# Cluster 3

@app.route('/canary/add', methods=['POST'])
def add_canary():
    """ Add a canary token to the prompt """
    logger.info(f'({request.path}) Adding canary token to prompt')
    prompt = check_field(request.json, 'prompt', str)
    always = check_field(request.json, 'always', bool, required=False)
    length = check_field(request.json, 'length', int, required=False)
    header = check_field(request.json, 'header', str, required=False)
    updated_prompt = vigil.canary_tokens.add(prompt=prompt, always=always if always else False, length=length if length else 16, header=header if header else '<-@!-- {canary} --@!->')
    logger.info(f'({request.path}) Returning response')
    return jsonify({'success': True, 'timestamp': timestamp_str(), 'result': updated_prompt})

def check_field(data, field_name: str, field_type: type, required: bool=True) -> str:
    field_data = data.get(field_name, None)
    if field_data is None:
        if required:
            logger.error(f'Missing "{field_name}" field')
            abort(400, f'Missing "{field_name}" field')
        return None
    if not isinstance(field_data, field_type):
        logger.error(f'Invalid data type; "{field_name}" value must be a {field_type.__name__}')
        abort(400, f'Invalid data type; "{field_name}" value must be a {field_type.__name__}')
    return field_data

def timestamp_str():
    return datetime.isoformat(datetime.utcnow())

@app.route('/canary/check', methods=['POST'])
def check_canary():
    """ Check if the prompt contains a canary token """
    logger.info(f'({request.path}) Checking prompt for canary token')
    prompt = check_field(request.json, 'prompt', str)
    result = vigil.canary_tokens.check(prompt=prompt)
    if result:
        message = 'Canary token found in prompt'
    else:
        message = 'No canary token found in prompt'
    logger.info(f'({request.path}) Returning response')
    return jsonify({'success': True, 'timestamp': timestamp_str(), 'result': result, 'message': message})

@app.route('/add/texts', methods=['POST'])
def add_texts():
    """ Add text to the vector database (embedded at index) """
    texts = check_field(request.json, 'texts', list)
    metadatas = check_field(request.json, 'metadatas', list)
    logger.info(f'({request.path}) Adding text to VectorDB')
    res, ids = vigil.vectordb.add_texts(texts, metadatas)
    if res is False:
        logger.error(f'({request.path}) Error adding text to VectorDB')
        abort(500, 'Error adding text to VectorDB')
    logger.info(f'({request.path}) Returning response')
    return jsonify({'success': True, 'timestamp': timestamp_str(), 'ids': ids})

@app.route('/analyze/response', methods=['POST'])
def analyze_response():
    """ Analyze a prompt and its response """
    logger.info(f'({request.path}) Received scan request')
    input_prompt = check_field(request.json, 'prompt', str)
    out_data = check_field(request.json, 'response', str)
    start_time = time.time()
    result = vigil.output_scanner.perform_scan(input_prompt, out_data)
    result['elapsed'] = round(time.time() - start_time, 6)
    logger.info(f'({request.path}) Returning response')
    return jsonify(result)

@app.route('/analyze/prompt', methods=['POST'])
def analyze_prompt():
    """ Analyze a prompt against a set of scanners """
    logger.info(f'({request.path}) Received scan request')
    input_prompt = check_field(request.json, 'prompt', str)
    cached_response = lru_cache.get(input_prompt)
    if cached_response:
        logger.info(f'({request.path}) Found response in cache!')
        cached_response['cached'] = True
        return jsonify(cached_response)
    start_time = time.time()
    result = vigil.input_scanner.perform_scan(input_prompt)
    result['elapsed'] = round(time.time() - start_time, 6)
    logger.info(f'({request.path}) Returning response')
    lru_cache.set(input_prompt, result)
    return jsonify(result)

class Manager:

    def __init__(self, scanners: List[BaseScanner], auto_update: bool=False, update_threshold: int=3, db_client=None, name: str='input'):
        self.name = f'dispatch:{name}'
        self.dispatcher = Scanner(scanners)
        self.auto_update = auto_update
        self.update_threshold = update_threshold
        self.db_client = db_client
        if self.auto_update:
            if self.db_client is None:
                logger.warn(f'{self.name} Auto-update disabled: db client is None')
            else:
                logger.info(f'{self.name} Auto-update vectordb enabled: threshold={self.update_threshold}')

    def perform_scan(self, prompt: str, prompt_response: str=None) -> dict:
        resp = ResponseModel(status='success', prompt=prompt, prompt_response=prompt_response, prompt_entropy=calculate_entropy(prompt))
        resp.uuid = str(resp.uuid)
        if not prompt:
            resp.errors.append('Input prompt value is empty')
            resp.status = 'failed'
            logger.error(f'{self.name} Input prompt value is empty')
            return resp.dict()
        logger.info(f'{self.name} Dispatching scan request id={resp.uuid}')
        scan_results = self.dispatcher.run(prompt=prompt, prompt_response=prompt_response, scan_id={resp.uuid})
        total_matches = 0
        for scanner_name, results in scan_results.items():
            if 'error' in results:
                resp.status = 'partial_success'
                resp.errors.append(f'Error in {scanner_name}: {results['error']}')
            else:
                resp.results[scanner_name] = {'matches': results}
                if len(results) > 0 and scanner_name != 'scanner:sentiment':
                    total_matches += 1
        for scanner_name, message in messages.items():
            if scanner_name in scan_results and len(scan_results[scanner_name]) > 0 and (message not in resp.messages):
                resp.messages.append(message)
        logger.info(f'{self.name} Total scanner matches: {total_matches}')
        if self.auto_update and total_matches >= self.update_threshold:
            logger.info(f'{self.name} (auto-update) Adding detected prompt to db id={resp.uuid}')
            doc_id = self.db_client.add_texts([prompt], [{'uuid': resp.uuid, 'source': 'auto-update', 'timestamp': timestamp_str(), 'threshold': self.update_threshold}])
            logger.success(f'{self.name} (auto-update) Successful doc_id={doc_id} id={resp.uuid}')
        logger.info(f'{self.name} Returning response object id={resp.uuid}')
        return resp.dict()

def calculate_entropy(text) -> float:
    prob = [text.count(c) / len(text) for c in set(text)]
    entropy = -sum((p * math.log2(p) for p in prob))
    return entropy

