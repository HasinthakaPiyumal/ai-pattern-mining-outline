# Cluster 31

class BlueprintRefactor:
    """
    Refactor Factorio naive blueprint policies using language models to generate more efficient and diverse code.
    """

    def __init__(self, config: RefactorConfig):
        self.config = config
        self.anthropic = Anthropic()
        self.deepseek = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com/v1')
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        log_dir = os.path.join(config.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('blueprint_refactor')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, 'refactor.log'))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.worker_loggers = {}
        for i in range(config.num_workers):
            worker_logger = logging.getLogger(f'worker_{i}')
            worker_logger.setLevel(logging.INFO)
            wfh = logging.FileHandler(os.path.join(log_dir, f'worker_{i}.log'))
            wfh.setFormatter(formatter)
            worker_logger.addHandler(wfh)
            worker_logger.addHandler(ch)
            self.worker_loggers[i] = worker_logger
        self.state_file = os.path.join(config.output_dir, 'processing_state.json')
        self.state = ProcessingState.load(self.state_file)
        self.metrics_df = self._init_metrics_df()
        self.servers = self._init_server_pool()
        self.worker_states = {}
        self.task_queue = queue.Queue()
        self.results_lock = threading.Lock()

    def _init_metrics_df(self) -> pd.DataFrame:
        """Initialize or load existing metrics DataFrame."""
        metrics_file = os.path.join(self.config.output_dir, f'refactor_metrics_{self.config.model}.csv')
        if os.path.exists(metrics_file):
            return pd.read_csv(metrics_file)
        return pd.DataFrame(columns=['model', 'blueprint_name', 'total_entities', 'distinct_entities', 'successful_attempts', 'total_attempts', 'success_rate', 'original_loc', 'min_refactored_loc', 'max_refactored_loc', 'avg_refactored_loc', 'best_compression_ratio', 'avg_compression_ratio'])

    def _update_metrics(self, base_name: str, blueprint: dict, original_loc: int, refactored_locs: List[int], successful_attempts: int, total_attempts: int):
        """
        Update metrics for a blueprint in a thread-safe manner.

        Args:
            base_name: Name of the blueprint
            blueprint: The blueprint dictionary
            original_loc: Lines of code in original implementation
            refactored_locs: List of lines of code in successful refactors
            successful_attempts: Number of successful refactoring attempts
            total_attempts: Total number of attempts made
        """
        with self.results_lock:
            total_entities = len(blueprint['entities'])
            distinct_entities = len({entity['name'] for entity in blueprint['entities']})
            if refactored_locs:
                min_loc = min(refactored_locs)
                max_loc = max(refactored_locs)
                avg_loc = sum(refactored_locs) / len(refactored_locs)
                best_compression = original_loc / min_loc if min_loc > 0 else 0
                avg_compression = original_loc / avg_loc if avg_loc > 0 else 0
            else:
                min_loc = max_loc = avg_loc = best_compression = avg_compression = 0
            success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0
            new_metrics = pd.DataFrame([{'model': self.config.model, 'blueprint_name': base_name, 'total_entities': total_entities, 'distinct_entities': distinct_entities, 'successful_attempts': successful_attempts, 'total_attempts': total_attempts, 'success_rate': success_rate, 'original_loc': original_loc, 'min_refactored_loc': min_loc, 'max_refactored_loc': max_loc, 'avg_refactored_loc': avg_loc, 'best_compression_ratio': best_compression, 'avg_compression_ratio': avg_compression}])
            self.metrics_df = self.metrics_df[~((self.metrics_df['blueprint_name'] == base_name) & (self.metrics_df['model'] == self.config.model))]
            self.metrics_df = pd.concat([self.metrics_df, new_metrics], ignore_index=True)
            metrics_path = os.path.join(self.config.output_dir, f'refactor_metrics_{self.config.model}.csv')
            self.metrics_df.to_csv(metrics_path, index=False)
            self.logger.info(f'Updated metrics for {base_name}:')
            self.logger.info(f'  Success rate: {success_rate:.2%}')
            self.logger.info(f'  Best compression: {best_compression:.2f}x')
            self.logger.info(f'  Average compression: {avg_compression:.2f}x')
            if successful_attempts >= self.config.max_attempts:
                self.state.mark_complete(base_name)
                self.state.save(self.state_file)
                self.logger.info(f'Marked {base_name} as complete')

    def _create_more_ore(self, position: Position, size=20):
        """
        We need to create more ore, because some mining templates don't fit on the lab scenario ore deposits.
        :param position: Position to create ore
        :param size: Size of patch
        :return: A lua script to create more ore
        """
        return f'\n/c local surface=game.players[1].surface\nlocal ore=nil\nlocal size={size}\nlocal density=10\nfor y=-size, size do\n\tfor x=-size, size do\n\t\ta=(size+1-math.abs(x))*10\n\t\tb=(size+1-math.abs(y))*10\n\t\tif a<b then\n\t\t\tore=math.random(a*density-a*(density-8), a*density+a*(density-8))\n\t\tend\n\t\tif b<a then\n\t\t\tore=math.random(b*density-b*(density-8), b*density+b*(density-8))\n\t\tend\n\t\tif surface.get_tile({position.x}+x, {position.y}+y).collides_with("ground-tile") then\n\t\t\tsurface.create_entity({{name="copper-ore", amount=ore, position={{{position.x}+x, {position.y}+y}}}})\n\t\tend\n\tend\nend\n'.strip()

    def _init_server_pool(self) -> List[ServerConfig]:
        """Initialize the pool of available Factorio servers."""
        if self.config.cluster_name:
            ips = get_public_ips(self.config.cluster_name)
            self.logger.info(f'Found {len(ips)} servers in ECS cluster {self.config.cluster_name}')
            return [ServerConfig(ip, 27000) for ip in ips]
        else:
            self.logger.info('No cluster name provided - using local server')
            self.config.num_workers = 1
            return [ServerConfig('localhost', 27000 + i) for i in range(self.config.num_workers)]

    def _worker_heartbeat(self, worker_id: int):
        """Update worker heartbeat and log status."""
        if worker_id in self.worker_states:
            self.worker_states[worker_id].update_heartbeat()
            worker_logger = self.worker_loggers[worker_id]
            state = self.worker_states[worker_id]
            worker_logger.debug(f'Heartbeat - Processing: {state.current_blueprint}')

    def _worker_process(self, worker_id: int):
        """Main worker process for handling blueprint refactoring tasks."""
        worker_logger = self.worker_loggers[worker_id]
        server_config = self.servers[worker_id % len(self.servers)]
        self.worker_states[worker_id] = WorkerState(server_config, worker_id)
        state = self.worker_states[worker_id]
        worker_logger.info(f'Worker {worker_id} starting on {server_config.address}:{server_config.tcp_port}')

        def heartbeat_routine():
            while state.instance is not None:
                self._worker_heartbeat(worker_id)
                time.sleep(10)
        heartbeat_thread = threading.Thread(target=heartbeat_routine, name=f'heartbeat-{worker_id}', daemon=True)
        heartbeat_thread.start()
        try:
            state.instance = FactorioInstance(address=server_config.address, tcp_port=server_config.tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={})
            time.sleep(3)
            copper_ore_patch = state.instance.get_resource_patch(Resource.CopperOre, state.instance.nearest(Resource.CopperOre))
            center_position = copper_ore_patch.bounding_box.center
            create_more_ore_command = self._create_more_ore(center_position)
            state.instance.add_command(create_more_ore_command, raw=True)
            state.instance.execute_transaction()
            expanded_copper_ore_patch = state.instance.get_resource_patch(Resource.CopperOre, state.instance.nearest(Resource.CopperOre))
            assert expanded_copper_ore_patch.size != copper_ore_patch.size, f'Failed to expand copper ore patch from {copper_ore_patch.size} to {expanded_copper_ore_patch.size}'
            while True:
                try:
                    blueprint_path, code_path = self.task_queue.get_nowait()
                    base_name = os.path.splitext(os.path.basename(code_path))[0]
                    state.current_blueprint = base_name
                    state.busy = True
                    worker_logger.info(f'Processing blueprint: {base_name}')
                    self._process_single_blueprint(blueprint_path, code_path, state.instance, worker_logger)
                    state.busy = False
                    state.current_blueprint = None
                    self.task_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            worker_logger.error(f'Worker {worker_id} failed: {str(e)}', exc_info=True)
        finally:
            worker_logger.info(f'Worker {worker_id} shutting down')
            if state.instance:
                state.instance.close()

    def code_contains_required_actions(self, code, required_actions=['connect_entities', 'place_entity_next_to']):
        return any((action in code for action in required_actions))

    def _process_single_blueprint(self, blueprint_path: str, code_path: str, instance: FactorioInstance, logger: logging.Logger):
        """Process a single blueprint with the given Factorio instance."""
        base_name = os.path.splitext(os.path.basename(code_path))[0]
        blueprint, original_code = self.load_blueprint_and_code(blueprint_path, code_path)
        original_loc = count_code_lines(original_code)
        original_code = original_code.replace('game.', '')
        analyzer = BlueprintAnalyzer(blueprint)
        inventory = analyzer.get_inventory()
        base_output_dir = os.path.join(self.config.output_dir, base_name)
        os.makedirs(base_output_dir, exist_ok=True)
        successful_attempts = 0
        total_attempts = 0
        refactored_locs = []
        while successful_attempts < self.config.max_attempts and total_attempts < self.config.max_attempts:
            try:
                logger.info(f'Attempt {total_attempts + 1} for `{base_name}`')
                refactored_code = self.get_refactored_code(blueprint, original_code, base_name)
                refactored_loc = count_code_lines(refactored_code)
                refactored_code = refactored_code.replace('```python', '').replace('```', '')
                logger.info('Verifying placement...')
                if self.verify_placement(refactored_code, blueprint, instance, inventory) and self.code_contains_required_actions(refactored_code):
                    output_path = os.path.join(base_output_dir, f'place_entity_next_to_{self.config.model}_{successful_attempts + 1}.py')
                    with open(output_path, 'w') as f:
                        f.write(refactored_code)
                    successful_attempts += 1
                    refactored_locs.append(refactored_loc)
                    logger.info(f'Successfully generated refactor {successful_attempts} (Compression: {original_loc / refactored_loc:.2f}x)')
                else:
                    logger.warning('Refactor failed verification')
            except Exception as e:
                logger.error(f'Error during refactoring attempt: {str(e)}', exc_info=True)
            total_attempts += 1
            with self.results_lock:
                self.state.add_attempt(base_name, self.config.model)
                self.state.save(self.state_file)
        self._update_metrics(base_name, blueprint, original_loc, refactored_locs, successful_attempts, total_attempts)

    def process_directory(self, implementation_dir: str, blueprints_dir: str):
        """Process all blueprints in parallel across available servers."""
        for filename in os.listdir(implementation_dir):
            if filename.endswith('.py'):
                base_name = os.path.splitext(filename)[0]
                if base_name in self.state.completed_blueprints:
                    self.logger.info(f'Skipping {filename} - already completed')
                    continue
                blueprint_path = os.path.join(blueprints_dir, filename.replace('.py', '.json'))
                code_path = os.path.join(implementation_dir, filename)
                if os.path.exists(blueprint_path):
                    self.task_queue.put((blueprint_path, code_path))
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            workers = []
            for i in range(self.config.num_workers):
                workers.append(executor.submit(self._worker_process, i))
            while any((not w.done() for w in workers)):
                self._log_worker_status()
                time.sleep(30)
        self.logger.info('All workers completed')

    def _log_worker_status(self):
        """Log current status of all workers."""
        status = []
        for worker_id, state in self.worker_states.items():
            status.append(f'Worker {worker_id} - {('Busy' if state.busy else 'Idle')} - Blueprint: {state.current_blueprint or 'None'} - Healthy: {state.is_healthy()}')
        self.logger.info('Worker Status:\n' + '\n'.join(status))

    def load_blueprint_and_code(self, blueprint_path: str, code_path: str) -> tuple[dict, str]:
        """Load the blueprint JSON and corresponding Python code."""
        with open(blueprint_path, 'r') as f:
            blueprint = json.loads(f.read())
        with open(code_path, 'r') as f:
            original_code = f.read()
        return (blueprint, original_code)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_refactored_code(self, blueprint: dict, original_code: str, base_name: str) -> str:
        """Get refactored code from the specified LLM."""
        prompt = f"You are an expert Python programmer and player of the game Factorio.\n\nHere is a entity layout in JSON called '{base_name}':\n```json\n{json.dumps(blueprint, indent=2)}\n```\n\nThe current Python implementation that creates this layout is as follows:\n```python\n{original_code}\n``` \n\nPlease rewrite this code to use `connect_entities` and `place_entity_next_to` methods, while preserving exactly the same functionality. \nThe code must produce identical entity placements. Don't redefine any used classes, as these will be imported for you. Think about the overall purpose of the code before refactoring it.\n\nKey requirements:\n1. Must use the same Position, Direction, and Prototype classes\n2. Must start from the same origin calculation\n3. Must use `place_entity_next_to` function to place entities next to each other.\n4. Must use `connect_entities` function to create lines of transport belts.\n4. Can use any Python features but must maintain compatibility\n5. Focus on making the code better, not just shorter\n6. Add comments before each section that explains the purpose of the following code given that we are creating a '{base_name}'.\n7. Avoid magic numbers and hard-coded values, declare them as variables instead where possible. \n    7a. Specifically, ensure that the bounding box parameters are calculated in terms of the number of entities that we are placing.\n\n:example: place_entity_next_to(Prototype.WoodenChest, Position(x=0, y=0), direction=Direction.UP, spacing=1)\n:return: Entity placed (with position of x=0, y=-1)\n\n:example: connect_entities(Position(x=0, y=0), Position(x=0, y=1)), connection_type=Prototype.TransportBelt)\n:return: Entity group of transport belts created\n\nOnly return python code between ```python and ``` tags, nothing else.\n"
        if 'gpt' in self.config.model:
            response = self.openai_client.chat.completions.create(model=self.config.model, messages=[{'role': 'user', 'content': prompt}], temperature=self.config.temperature)
            return response.choices[0].message.content
        elif self.config.model == 'claude-3':
            response = self.anthropic.messages.create(model='claude-3-5-sonnet-20241022', temperature=self.config.temperature, messages=[{'role': 'user', 'content': prompt}], max_tokens=4096)
            return response.content[0].text
        elif self.config.model == 'deepseek-coder':
            try:
                response = self.deepseek.chat.completions.create(model='deepseek-coder', messages=[{'role': 'user', 'content': prompt}], max_tokens=4096, temperature=self.config.temperature)
            except openai.error.OpenAIError as e:
                print(f'Error during DeepSeek request: {str(e)}')
            return response.choices[0].message.content

    def verify_placement(self, code: str, blueprint: dict, instance: FactorioInstance, inventory: dict) -> bool:
        """Verify that the refactored code produces identical entity placement."""
        try:
            instance.reset()
            instance.set_inventory(inventory)
            score, goal, result = instance.eval_with_error(code.replace('game.', ''), timeout=60)
            if 'error' in result:
                return False
            game_entities = instance.get_entities()
            analyzer = BlueprintAnalyzer(blueprint)
            analyzer.verify_placement(game_entities)
            return True
        except Exception as e:
            print(f'Verification failed: {str(e)}')
            return False

    def calculate_blueprint_complexity(self, blueprint: dict) -> tuple[int, int]:
        """Calculate complexity metrics for a blueprint."""
        total_entities = len(blueprint['entities'])
        distinct_entities = len({entity['name'] for entity in blueprint['entities']})
        return (total_entities, distinct_entities)

    def save_metrics(self):
        """Save metrics to CSV file."""
        metrics_path = os.path.join(self.config.output_dir, f'refactor_metrics_(place_next_to)_{self.config.model}.csv')
        self.metrics_df.to_csv(metrics_path, index=False)

    def analyze_metrics(self) -> pd.DataFrame:
        """Analyze and return summary statistics of the metrics."""
        summary = self.metrics_df.groupby('model').agg({'success_rate': ['mean', 'std'], 'total_entities': ['mean', 'std'], 'distinct_entities': ['mean', 'std'], 'best_compression_ratio': ['mean', 'std', 'max'], 'avg_compression_ratio': ['mean', 'std']}).round(3)
        correlations = pd.DataFrame({'total_entities_correlation': self.metrics_df.groupby('model').apply(lambda x: x['total_entities'].corr(x['success_rate'])), 'distinct_entities_correlation': self.metrics_df.groupby('model').apply(lambda x: x['distinct_entities'].corr(x['success_rate'])), 'complexity_compression_correlation': self.metrics_df.groupby('model').apply(lambda x: x['total_entities'].corr(x['best_compression_ratio']))})
        return pd.concat([summary, correlations], axis=1)

def get_public_ips(cluster_name):
    ecs_client = boto3.client('ecs')
    ec2_client = boto3.client('ec2')
    tasks = []
    paginator = ecs_client.get_paginator('list_tasks')
    for page in paginator.paginate(cluster=cluster_name, desiredStatus='RUNNING'):
        tasks.extend(page['taskArns'])
    if not tasks:
        print(f'No running tasks found in cluster {cluster_name}')
        return []
    public_ips = []
    for i in range(0, len(tasks), 100):
        batch = tasks[i:i + 100]
        task_details = ecs_client.describe_tasks(cluster=cluster_name, tasks=batch)
        for task in task_details['tasks']:
            eni_id = None
            for attachment in task['attachments']:
                for detail in attachment['details']:
                    if detail['name'] == 'networkInterfaceId':
                        eni_id = detail['value']
                        break
                if eni_id:
                    break
            if not eni_id:
                print(f'Warning: No ENI found for task {task['taskArn']}')
                continue
            try:
                eni_details = ec2_client.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
                if 'Association' in eni_details['NetworkInterfaces'][0]:
                    public_ip = eni_details['NetworkInterfaces'][0]['Association']['PublicIp']
                    public_ips.append(public_ip)
            except Exception as e:
                print(f'Error getting public IP for ENI {eni_id}: {str(e)}')
    return public_ips

def connect_to_server(ip_address):
    focus_factorio()
    pyautogui.press('esc')
    pyautogui.press('esc')
    pyautogui.press('esc')
    pyautogui.click(MULTIPLAYER_BUTTON)
    pyautogui.click(CONNECT_TO_ADDRESS_BUTTON)
    pyautogui.click(IP_INPUT_FIELD)
    pyautogui.hotkey('command', 'a')
    pyautogui.press('delete')
    pyautogui.write(f'{ip_address}:34197')
    pyautogui.click(CONNECT_BUTTON)
    time.sleep(5)
    pyautogui.press('esc')
    pyautogui.click(ESC_MENU_QUIT_BUTTON)
    time.sleep(3)

def focus_factorio():
    applescript = '\n    tell application "Factorio" to activate\n    '
    subprocess.run(['osascript', '-e', applescript])
    time.sleep(1)

def main(cluster_name):
    ip_addresses = get_public_ips(cluster_name)
    ip_addresses = get_uninitialised_ips(ip_addresses)
    factorio_process = launch_factorio()
    for ip in ip_addresses:
        connect_to_server(ip)
        print(f'Connected to and quit from {ip}')
    factorio_process.terminate()

def get_uninitialised_ips(ip_addresses: List[str], tcp_ports: List[str], max_workers: int=8) -> List[str]:
    """
    Check multiple IP addresses in parallel using ThreadPoolExecutor.

    Args:
        ip_addresses: List of IP addresses to check
        max_workers: Maximum number of concurrent threads (default: 20)

    Returns:
        List of IP addresses that successfully initialized
    """
    invalid_ips = []
    total_ips = len(ip_addresses)
    print(f'Starting initialization check for {total_ips} IP addresses...')
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ip = {executor.submit(is_initialised, ip, port): ip for ip, port in zip(ip_addresses, tcp_ports)}
        for i, future in enumerate(as_completed(future_to_ip), 1):
            ip = future_to_ip[future]
            try:
                if future.result():
                    print(f'Progress: {i}/{total_ips} - {ip} is valid')
                else:
                    invalid_ips.append(ip)
                    print(f'Progress: {i}/{total_ips} - {ip} is invalid')
            except Exception as e:
                print(f'Progress: {i}/{total_ips} - Unexpected error with {ip}: {str(e)}')
    elapsed_time = time.time() - start_time
    print(f'\nCompleted in {elapsed_time:.2f} seconds')
    print(f'Found {len(invalid_ips)} uninitialised IPs out of {total_ips}')
    return invalid_ips

def launch_factorio():
    process = subprocess.Popen(['open', '-a', 'Factorio'])
    time.sleep(10)
    return process

