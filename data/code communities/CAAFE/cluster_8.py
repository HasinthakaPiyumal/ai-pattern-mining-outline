# Cluster 8

def load_all_data():
    cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(benchmark_ids, multiclass=True, shuffled=True, filter_for_nan=False, max_samples=10000, num_feats=25, return_capped=False)
    cc_test_datasets_multiclass += load_kaggle()
    return postprocess_datasets(cc_test_datasets_multiclass)

def load_openml_list(dids, filter_for_nan=False, num_feats=100, min_samples=100, max_samples=400, multiclass=True, max_num_classes=10, shuffled=True, return_capped=False):
    """Load a list of openml datasets and return the data in the correct format."""
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    print(f'Number of datasets: {len(openml_list)}')
    datalist = pd.DataFrame.from_dict(openml_list, orient='index')
    if filter_for_nan:
        datalist = datalist[datalist['NumberOfInstancesWithMissingValues'] == 0]
        print(f'Number of datasets after Nan and feature number filtering: {len(datalist)}')
    for ds in datalist.index:
        modifications = {'samples_capped': False, 'classes_capped': False, 'feats_capped': False}
        entry = datalist.loc[ds]
        print('Loading', entry['name'], entry.did, '..')
        if entry['NumberOfClasses'] == 0.0:
            raise Exception('Regression not supported')
        else:
            X, y, categorical_feats, attribute_names, description = get_openml_classification(int(entry.did), max_samples, multiclass=multiclass, shuffled=shuffled)
        if X is None:
            continue
        if X.shape[1] > num_feats:
            if return_capped:
                X = X[:, 0:num_feats]
                categorical_feats = [c for c in categorical_feats if c < num_feats]
                modifications['feats_capped'] = True
            else:
                print('Too many features')
                continue
        if X.shape[0] == max_samples:
            modifications['samples_capped'] = True
        if X.shape[0] < min_samples:
            print(f'Too few samples left')
            continue
        if len(np.unique(y)) > max_num_classes:
            if return_capped:
                X = X[y < np.unique(y)[10]]
                y = y[y < np.unique(y)[10]]
                modifications['classes_capped'] = True
            else:
                print(f'Too many classes')
                continue
        datasets += [[entry['name'], X, y, categorical_feats, attribute_names, modifications, description]]
    return (datasets, datalist)

def get_openml_classification(did, max_samples, multiclass=True, shuffled=True):
    """Load an openml dataset and return the data in the correct format."""
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format='array', target=dataset.default_target_attribute)
    description = refactor_openml_description(dataset.description)
    if not multiclass:
        X = X[y < 2]
        y = y[y < 2]
    if multiclass and (not shuffled):
        raise NotImplementedError("This combination of multiclass and shuffling isn't implemented")
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        print('Not a NP Array, skipping')
        return (None, None, None, None)
    if not shuffled:
        sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
        pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
        X, y = (X[sort][-pos * 2:], y[sort][-pos * 2:])
        y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
        X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    else:
        order = np.arange(y.shape[0])
        np.random.seed(13)
        np.random.shuffle(order)
        X, y = (torch.tensor(X[order]), torch.tensor(y[order]))
    if max_samples:
        X, y = (X[:max_samples], y[:max_samples])
    return (X, y, list(np.where(categorical_indicator)[0]), attribute_names + [list(dataset.features.values())[-1].name], description)

def load_kaggle():
    cc_test_datasets_multiclass = []
    for name in kaggle_dataset_ids:
        try:
            df_all = pd.read_csv(f'datasets_kaggle/{name[0]}/{name[1]}.csv')
            df_train, df_test = train_test_split(df_all, test_size=0.25, random_state=0)
            ds = ['kaggle_' + name[0], df_all.copy().drop(columns=[name[2]], inplace=False).values, df_all[name[2]].values, [], df_train.copy().drop(columns=[name[2]], inplace=False).columns.tolist() + [name[2]], '']
            data_dir = os.environ.get('DATA_DIR', 'data/')
            path = f'{data_dir}/dataset_descriptions/kaggle_{name[0]}.txt'
            try:
                with open(path) as f:
                    ds[-1] = f.read()
            except:
                print('Using initial description')
            cc_test_datasets_multiclass += [ds]
        except:
            print(f'{name[0]} at datasets_kaggle/{name[0]}/{name[1]}.csv not found, skipping...')
    for name in kaggle_competition_ids:
        try:
            df_all = pd.read_csv(f'datasets_kaggle/{name}/train.csv')
            df_train, df_test = train_test_split(df_all, test_size=0.25, random_state=0)
            ds = ['kaggle_' + name, df_all[df_all.columns[:-1]].values, df_all[df_all.columns[-1]].values, [], df_train.columns.tolist(), '']
            path = f'dataset_descriptions/kaggle_{name}.txt'
            try:
                with open(path) as f:
                    ds[-1] = f.read()
            except:
                print('Using initial description')
            cc_test_datasets_multiclass += [ds]
        except:
            print(f'{name} at datasets_kaggle/{name}/train.csv not found, skipping...')
    return cc_test_datasets_multiclass

