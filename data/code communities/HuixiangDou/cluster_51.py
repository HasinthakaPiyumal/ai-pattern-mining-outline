# Cluster 51

def read_config_ini_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'config.ini':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config = pytoml.load(f)
                    print((file_path, config['llm']['server']['remote_llm_max_text_length']))
                    config['llm']['server']['remote_llm_max_text_length'] = 40000
                    with open(file_path, 'w', encoding='utf8') as f:
                        pytoml.dump(config, f)
                except Exception as e:
                    print(f'An error occurred while reading {file_path}: {e}')

