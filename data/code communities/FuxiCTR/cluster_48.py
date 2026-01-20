# Cluster 48

def print_to_json(data, sort_keys=True):
    new_data = dict(((k, str(v)) for k, v in data.items()))
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def enumerate_params(config_file, exclude_expid=[]):
    with open(config_file, 'r') as cfg:
        config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
    tune_dict = config_dict['tuner_space']
    for k, v in tune_dict.items():
        if not isinstance(v, list):
            tune_dict[k] = [v]
    experiment_id = config_dict['base_expid']
    if 'model_config' in config_dict:
        model_dict = config_dict['model_config'][experiment_id]
    else:
        base_config_dir = config_dict.get('base_config', os.path.dirname(config_file))
        model_dict = load_model_config(base_config_dir, experiment_id)
    dataset_id = config_dict.get('dataset_id', model_dict['dataset_id'])
    if 'dataset_config' in config_dict:
        dataset_dict = config_dict['dataset_config'][dataset_id]
    else:
        dataset_dict = load_dataset_config(base_config_dir, dataset_id)
    if model_dict['dataset_id'] == 'TBD':
        model_dict['dataset_id'] = dataset_id
        experiment_id = model_dict['model'] + '_' + dataset_id
    tuner_keys = set(tune_dict.keys())
    base_keys = set(model_dict.keys()).union(set(dataset_dict.keys()))
    if len(tuner_keys - base_keys) > 0:
        raise RuntimeError('Invalid params in tuner config: {}'.format(tuner_keys - base_keys))
    config_dir = config_file.replace('.yaml', '')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    dataset_dict = {k: tune_dict[k] if k in tune_dict else [v] for k, v in dataset_dict.items()}
    dataset_para_keys = list(dataset_dict.keys())
    dataset_para_combs = dict()
    for idx, values in enumerate(itertools.product(*map(dataset_dict.get, dataset_para_keys))):
        dataset_params = dict(zip(dataset_para_keys, values))
        if dataset_params['data_format'] == 'npz' or (dataset_params['data_format'] == 'parquet' and dataset_params.get('rebuild_dataset') == False):
            dataset_para_combs[dataset_id] = dataset_params
        else:
            hash_id = hashlib.md5(''.join(sorted(print_to_json(dataset_params))).encode('utf-8')).hexdigest()[0:8]
            dataset_para_combs[dataset_id + '_{}'.format(hash_id)] = dataset_params
    dataset_config = os.path.join(config_dir, 'dataset_config.yaml')
    with open(dataset_config, 'w') as fw:
        yaml.dump(dataset_para_combs, fw, default_flow_style=None, indent=4)
    model_dict = {k: tune_dict[k] if k in tune_dict else [v] for k, v in model_dict.items()}
    model_para_keys = list(model_dict.keys())
    model_param_combs = dict()
    for idx, values in enumerate(itertools.product(*map(model_dict.get, model_para_keys))):
        model_param_combs[idx + 1] = dict(zip(model_para_keys, values))
    merged_param_combs = dict()
    for idx, item in enumerate(itertools.product(model_param_combs.values(), dataset_para_combs.keys())):
        para_dict = item[0]
        para_dict['dataset_id'] = item[1]
        del para_dict['model_id']
        random_str = ''
        if para_dict['debug_mode']:
            random_str = '{:06d}'.format(np.random.randint(1000000.0))
        hash_id = hashlib.md5((''.join(sorted(print_to_json(para_dict))) + random_str).encode('utf-8')).hexdigest()[0:8]
        hash_expid = experiment_id + '_{:03d}_{}'.format(idx + 1, hash_id)
        if hash_expid not in exclude_expid:
            merged_param_combs[hash_expid] = para_dict.copy()
    model_config = os.path.join(config_dir, 'model_config.yaml')
    with open(model_config, 'w') as fw:
        yaml.dump(merged_param_combs, fw, default_flow_style=None, indent=4)
    print('Enumerate all tuner configurations done.')
    return config_dir

def load_model_config(config_dir, experiment_id):
    model_configs = glob.glob(os.path.join(config_dir, 'model_config.yaml'))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, 'model_config/*.yaml'))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    params = found_params.get('Base', {})
    params.update(found_params.get(experiment_id, {}))
    assert 'dataset_id' in params, f'expid={experiment_id} is not valid in config.'
    params['model_id'] = experiment_id
    return params

def load_dataset_config(config_dir, dataset_id):
    params = {'dataset_id': dataset_id}
    dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config.yaml'))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config/*.yaml'))
    for config in dataset_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                return params
    raise RuntimeError(f'dataset_id={dataset_id} is not found in config.')

