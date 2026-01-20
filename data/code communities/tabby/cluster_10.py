# Cluster 10

def rewrite_prompt(args):
    binary = args['tabby_path']
    language = args['language']
    segments = generate_completion_segments(args)
    serve_command = [binary, 'serve', '--model', 'TabbyML/T5P-220M']
    process = subprocess.Popen(serve_command)
    try:
        if not wait_for_online(5):
            logging.error('Tabby server is not online')
            return
        completion_url = f'http://127.0.0.1:{PORT}/v1/completions'
        for s in segments:
            req = {'language': language, 'segments': s}
            r = requests.post(completion_url, json=req)
            logging.info(r.status_code)
    finally:
        process.terminate()

def generate_completion_segments(args):
    binary = args['tabby_path']
    sample_repo_url = args['sample_repo_url']
    language = args['language']
    prompt_count = args['prompt_count']
    segments = []
    sample_path = os.path.expanduser('~/.tabby/eval_sample')
    sample_config_file_path = os.path.join(sample_path, 'config.toml')
    config = {'repositories': [{'git_url': sample_repo_url}]}
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)
    with open(sample_config_file_path, 'w+') as f:
        toml.dump(config, f)
    sample_index_command = [binary, 'scheduler', '--now']
    subprocess.run(sample_index_command, env={'TABBY_ROOT': sample_path})
    contents = []
    dataset_path = os.path.join(sample_path, 'dataset')
    files = os.listdir(dataset_path)
    for file_name in files:
        dataset_file_path = os.path.join(dataset_path, file_name)
        with jsonlines.open(dataset_file_path) as dataset:
            for obj in dataset:
                if obj['language'] != language:
                    continue
                contents.append(obj['content'])
    for _ in range(prompt_count):
        content = ''
        while not content:
            file_content = random.randrange(len(contents))
            content = contents[file_content]
        cursor = random.randrange(len(content))
        lb = 0
        pc = cursor
        while True:
            if pc < 0:
                break
            if content[pc] == '\n':
                lb += 1
                if lb == 10:
                    break
            pc -= 1
        prefix = content[pc + 1:cursor + 1]
        lb = 0
        sc = cursor + 1
        while True:
            if sc >= len(content):
                break
            if content[sc] == '\n':
                lb += 1
                if lb == 10:
                    break
            sc += 1
        suffix = content[cursor + 1:sc]
        segments.append({'prefix': prefix, 'suffix': suffix})
    return segments

def wait_for_online(timeout):
    logging.info('Trying to connect to tabby')
    health_url = f'http://127.0.0.1:{PORT}/v1/health'
    is_online = False
    till = time.time() + timeout * 1000
    while time.time() < till:
        try:
            r = requests.post(health_url)
            if r.status_code == 200:
                logging.info('Tabby is online now')
                is_online = True
                break
        except:
            logging.info('Retrying to connect')
        time.sleep(1)
    return is_online

def main():
    args = toml.load('eval.toml')
    index(args)
    rewrite_prompt(args)

def index(args):
    binary = args['tabby_path']
    index_repo_url = args['index_repo_url']
    config_file_path = os.path.expanduser('~/.tabby/config.toml')
    config = {'repositories': [{'git_url': index_repo_url}], 'experimental': {'enable_prompt_rewrite': True}}
    with open(config_file_path, 'w+') as f:
        toml.dump(config, f)
    cmd = [binary, 'scheduler', '--now']
    subprocess.run(cmd)

