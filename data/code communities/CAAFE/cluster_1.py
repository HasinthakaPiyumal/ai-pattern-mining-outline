# Cluster 1

def generate_and_save_feats(i, seed=0, iterative_method=None, iterations=10):
    if iterative_method is None:
        iterative_method = tabpfn
    ds = cc_test_datasets_multiclass[i]
    ds, df_train, df_test, df_train_old, df_test_old = get_data_split(ds, seed)
    code, prompt, messages = generate_features(ds, df_train, just_print_prompt=False, model=model, iterative=iterations, metric_used=metric_used, iterative_method=iterative_method, display_method='print')
    data_dir = os.environ.get('DATA_DIR', 'data/')
    f = open(f'{data_dir}/generated_code/{ds[0]}_{prompt_id}_{seed}_prompt.txt', 'w')
    f.write(prompt)
    f.close()
    f = open(f'{data_dir}/generated_code/{ds[0]}_{prompt_id}_{seed}_code.txt', 'w')
    f.write(code)
    f.close()

def get_data_split(ds, seed):

    def get_df(X, y):
        df = pd.DataFrame(data=np.concatenate([X, np.expand_dims(y, -1)], -1), columns=ds[4])
        cat_features = ds[3]
        for c in cat_features:
            if len(np.unique(df.iloc[:, c])) > 50:
                cat_features.remove(c)
                continue
            df[df.columns[c]] = df[df.columns[c]].astype('int32')
        return df.infer_objects()
    ds = copy.deepcopy(ds)
    X = ds[1].numpy() if type(ds[1]) == torch.Tensor else ds[1]
    y = ds[2].numpy() if type(ds[2]) == torch.Tensor else ds[2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    df_train = get_df(X_train, y_train)
    df_test = get_df(X_test, y_test)
    df_train.iloc[:, -1] = df_train.iloc[:, -1].astype('category')
    df_test.iloc[:, -1] = df_test.iloc[:, -1].astype('category')
    df_test_old = copy.deepcopy(df_test)
    df_train_old = copy.deepcopy(df_train)
    data_dir = os.environ.get('DATA_DIR', 'data/')
    source = '' if ds[0].startswith('kaggle') else 'openml_'
    path = f'{data_dir}/dataset_descriptions/{source}{ds[0]}.txt'
    try:
        with open(path) as f:
            ds[-1] = f.read()
    except:
        print(f'Using initial description (tried reading {path})')
    return (ds, df_train, df_test, df_train_old, df_test_old)

def get_df(X, y):
    df = pd.DataFrame(data=np.concatenate([X, np.expand_dims(y, -1)], -1), columns=ds[4])
    cat_features = ds[3]
    for c in cat_features:
        if len(np.unique(df.iloc[:, c])) > 50:
            cat_features.remove(c)
            continue
        df[df.columns[c]] = df[df.columns[c]].astype('int32')
    return df.infer_objects()

