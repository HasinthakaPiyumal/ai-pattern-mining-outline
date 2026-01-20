# Cluster 4

def evaluate_dataset_with_and_without_cafe(ds, seed, methods, metric_used, prompt_id='v2', max_time=300, overwrite=False):
    """Evaluates a dataframe with and without feature extension."""
    ds, df_train, df_test, df_train_old, df_test_old = get_data_split(ds, seed)
    ds, df_train, df_test = evaluate_dataset_helper_extend_df(df_train, df_test, ds, prompt_id, seed)
    print('SHAPE BEFORE', df_train_old.shape, 'AFTER', df_train.shape)
    for method in methods:
        method_str = method if type(method) == str else 'transformer'
        data_dir = os.environ.get('DATA_DIR', 'data/')
        path = f'{data_dir}/evaluations/result_{ds[0]}_{prompt_id}_{seed}_{method_str}.txt'
        if os.path.exists(path) and (not overwrite):
            print(f'Skipping {path}')
            continue
        print(ds[0], method_str, prompt_id, seed)
        r = evaluate_dataset(df_train=df_train, df_test=df_test, prompt_id=prompt_id, name=ds[0], method=method, metric_used=metric_used, max_time=max_time, seed=seed, target_name=ds[4][-1])
        f = open(path, 'wb')
        pickle.dump(r, f)
        f.close()

def evaluate_dataset_helper_extend_df(df_train, df_test, ds, prompt_id, seed, code_overwrite=None):
    target_train = df_train[ds[4][-1]]
    target_test = df_test[ds[4][-1]]
    df_train = df_train.drop(columns=[ds[4][-1]])
    df_test = df_test.drop(columns=[ds[4][-1]])
    if prompt_id == 'dfs':
        df_train, df_test = extend_using_dfs(df_train, df_test, target_train)
    elif prompt_id == 'autofeat':
        df_train, df_test = extend_using_autofeat(df_train, df_test, target_train)
    elif prompt_id == 'v4' or prompt_id == 'v3':
        df_train, df_test = extend_using_caafe(df_train, df_test, ds, seed, prompt_id, code_overwrite=code_overwrite)
    elif prompt_id == 'v4+dfs' or prompt_id == 'v3+dfs':
        df_train, df_test = extend_using_caafe(df_train, df_test, ds, seed, prompt_id[0:2])
        df_train, df_test = extend_using_dfs(df_train, df_test, target_train)
    elif prompt_id == 'v4+autofeat' or prompt_id == 'v3+autofeat':
        df_train, df_test = extend_using_caafe(df_train, df_test, ds, seed, prompt_id[0:2])
        df_train, df_test = extend_using_autofeat(df_train, df_test, target_train)
    df_train[ds[4][-1]] = target_train
    df_test[ds[4][-1]] = target_test
    ds[3] = []
    ds[2] = []
    return (ds, df_train, df_test)

