# Cluster 7

def make_dataset_numeric(df: pd.DataFrame, mappings: Dict[str, Dict[int, str]]) -> pd.DataFrame:
    """
    Converts the categorical columns in the given dataframe to integer values using the given mappings.

    Parameters:
    df (pandas.DataFrame): The dataframe to convert.
    mappings (Dict[str, Dict[int, str]]): The mappings to use for the conversion.

    Returns:
    pandas.DataFrame: The converted dataframe.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(lambda col: convert_categorical_to_integer_f(col, mapping=mappings.get(col.name)), axis=0)
    df = df.astype(float)
    return df

def convert_categorical_to_integer_f(column: pd.Series, mapping: Optional[Dict[int, str]]=None) -> pd.Series:
    """
    Converts a categorical column to integer values using the given mapping.

    Parameters:
    column (pandas.Series): The column to convert.
    mapping (Dict[int, str], optional): The mapping to use for the conversion. Defaults to None.

    Returns:
    pandas.Series: The converted column.
    """
    if mapping is not None:
        if column.dtype.name == 'category':
            column = column.cat.add_categories([-1])
        return column.map(mapping).fillna(-1).astype(int)
    return column

def make_datasets_numeric(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame], target_column: str, return_mappings: Optional[bool]=False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Dict[str, Dict[int, str]]]]:
    """
    Converts the categorical columns in the given training and test dataframes to integer values using mappings created from the training dataframe.

    Parameters:
    df_train (pandas.DataFrame): The training dataframe to convert.
    df_test (pandas.DataFrame, optional): The test dataframe to convert. Defaults to None.
    target_column (str): The name of the target column.
    return_mappings (bool, optional): Whether to return the mappings used for the conversion. Defaults to False.

    Returns:
    Tuple[pandas.DataFrame, Optional[pandas.DataFrame], Optional[Dict[str, Dict[int, str]]]]: The converted training dataframe, the converted test dataframe (if it exists), and the mappings used for the conversion (if `return_mappings` is True).
    """
    df_train = copy.deepcopy(df_train)
    df_train = df_train.infer_objects()
    if df_test is not None:
        df_test = copy.deepcopy(df_test)
        df_test = df_test.infer_objects()
    mappings = create_mappings(df_train)
    non_target = [c for c in df_train.columns if c != target_column]
    df_train[non_target] = make_dataset_numeric(df_train[non_target], mappings)
    if df_test is not None:
        df_test[non_target] = make_dataset_numeric(df_test[non_target], mappings)
    if return_mappings:
        return (df_train, df_test, mappings)
    return (df_train, df_test)

def create_mappings(df_train: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """
    Creates a dictionary of mappings for categorical columns in the given dataframe.

    Parameters:
    df_train (pandas.DataFrame): The dataframe to create mappings for.

    Returns:
    Dict[str, Dict[int, str]]: A dictionary of mappings for categorical columns in the dataframe.
    """
    mappings = {}
    for col in df_train.columns:
        if df_train[col].dtype.name == 'category' or df_train[col].dtype.name == 'object':
            mappings[col] = {v: i for i, v in enumerate(df_train[col].astype('category').cat.categories)}
    return mappings

def evaluate_dataset(df_train: pd.DataFrame, df_test: pd.DataFrame, prompt_id, name, method, metric_used, target_name, max_time=300, seed=0):
    df_train, df_test = (copy.deepcopy(df_train), copy.deepcopy(df_test))
    df_train, _, mappings = make_datasets_numeric(df_train, None, target_name, return_mappings=True)
    df_test = make_dataset_numeric(df_test, mappings=mappings)
    if df_test is not None:
        test_x, test_y = get_X_y(df_test, target_name=target_name)
    x, y = get_X_y(df_train, target_name=target_name)
    feature_names = list(df_train.drop(target_name, axis=1).columns)
    np.random.seed(0)
    if method == 'autogluon' or method == 'autosklearn2':
        if method == 'autogluon':
            from tabpfn.scripts.tabular_baselines import autogluon_metric
            clf = autogluon_metric
        elif method == 'autosklearn2':
            from tabpfn.scripts.tabular_baselines import autosklearn2_metric
            clf = autosklearn2_metric
        metric, ys, res = clf(x, y, test_x, test_y, feature_names, metric_used, max_time=max_time)
    elif type(method) == str:
        if method == 'gp':
            from tabpfn.scripts.tabular_baselines import gp_metric
            clf = gp_metric
        elif method == 'knn':
            from tabpfn.scripts.tabular_baselines import knn_metric
            clf = knn_metric
        elif method == 'xgb':
            from tabpfn.scripts.tabular_baselines import xgb_metric
            clf = xgb_metric
        elif method == 'catboost':
            from tabpfn.scripts.tabular_baselines import catboost_metric
            clf = catboost_metric
        elif method == 'random_forest':
            from tabpfn.scripts.tabular_baselines import random_forest_metric
            clf = random_forest_metric
        elif method == 'logistic':
            from tabpfn.scripts.tabular_baselines import logistic_metric
            clf = logistic_metric
        metric, ys, res = clf(x, y, test_x, test_y, [], metric_used, max_time=max_time, no_tune={})
    elif isinstance(method, BaseEstimator):
        method.fit(X=x, y=y.long())
        ys = method.predict_proba(test_x)
    else:
        metric, ys, res = method(x, y, test_x, test_y, [], metric_used)
    acc = tabpfn.scripts.tabular_metrics.accuracy_metric(test_y, ys)
    roc = tabpfn.scripts.tabular_metrics.auc_metric(test_y, ys)
    method_str = method if type(method) == str else 'transformer'
    return {'acc': float(acc.numpy()), 'roc': float(roc.numpy()), 'prompt': prompt_id, 'seed': seed, 'name': name, 'size': len(df_train), 'method': method_str, 'max_time': max_time, 'feats': x.shape[-1]}

def get_X_y(df_train, target_name):
    y = torch.tensor(df_train[target_name].astype(int).to_numpy())
    x = torch.tensor(df_train.drop(target_name, axis=1).to_numpy())
    return (x, y)

def get_leave_one_out_importance(df_train, df_test, ds, method, metric_used, max_time=30):
    """Get the importance of each feature for a dataset by dropping it in the training and prediction."""
    res_base = evaluate_dataset(ds, df_train, df_test, prompt_id='', name=ds[0], method=method, metric_used=metric_used, max_time=max_time)
    importances = {}
    for feat_idx, feat in enumerate(set(df_train.columns)):
        if feat == ds[4][-1]:
            continue
        df_train_ = df_train.copy().drop(feat, axis=1)
        df_test_ = df_test.copy().drop(feat, axis=1)
        ds_ = copy.deepcopy(ds)
        res = evaluate_dataset(ds_, df_train_, df_test_, prompt_id='', name=ds[0], method=method, metric_used=metric_used, max_time=max_time)
        importances[feat] = (round(res_base['roc'] - res['roc'], 3),)
    return importances

