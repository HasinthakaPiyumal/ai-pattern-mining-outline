# Cluster 6

class CAAFEClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses the CAAFE algorithm to generate features and a base classifier to make predictions.

    Parameters:
    base_classifier (object, optional): The base classifier to use. If None, a default TabPFNClassifier will be used. Defaults to None.
    optimization_metric (str, optional): The metric to optimize during feature generation. Can be 'accuracy' or 'auc'. Defaults to 'accuracy'.
    iterations (int, optional): The number of iterations to run the CAAFE algorithm. Defaults to 10.
    llm_model (str, optional): The LLM model to use for generating features. Defaults to 'gpt-3.5-turbo'.
    n_splits (int, optional): The number of cross-validation splits to use during feature generation. Defaults to 10.
    n_repeats (int, optional): The number of times to repeat the cross-validation during feature generation. Defaults to 2.
    """

    def __init__(self, base_classifier: Optional[object]=None, optimization_metric: str='accuracy', iterations: int=10, llm_model: str='gpt-3.5-turbo', n_splits: int=10, n_repeats: int=2) -> None:
        self.base_classifier = base_classifier
        if self.base_classifier is None:
            from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
            import torch
            from functools import partial
            self.base_classifier = TabPFNClassifier(N_ensemble_configurations=16, device='cuda' if torch.cuda.is_available() else 'cpu')
            self.base_classifier.fit = partial(self.base_classifier.fit, overwrite_warning=True)
        self.llm_model = llm_model
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def fit_pandas(self, df, dataset_description, target_column_name, **kwargs):
        """
        Fit the classifier to a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame to fit the classifier to.
        dataset_description (str): A description of the dataset.
        target_column_name (str): The name of the target column in the DataFrame.
        **kwargs: Additional keyword arguments to pass to the base classifier's fit method.
        """
        feature_columns = list(df.drop(columns=[target_column_name]).columns)
        X, y = (df.drop(columns=[target_column_name]).values, df[target_column_name].values)
        return self.fit(X, y, dataset_description, feature_columns, target_column_name, **kwargs)

    def fit(self, X, y, dataset_description, feature_names, target_name, disable_caafe=False):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.
        dataset_description : str
            A description of the dataset.
        feature_names : List[str]
            The names of the features in the dataset.
        target_name : str
            The name of the target variable in the dataset.
        disable_caafe : bool, optional
            Whether to disable the CAAFE algorithm, by default False.
        """
        self.dataset_description = dataset_description
        self.feature_names = list(feature_names)
        self.target_name = target_name
        self.X_ = X
        self.y_ = y
        if X.shape[0] > 3000 and self.base_classifier.__class__.__name__ == 'TabPFNClassifier':
            print('WARNING: TabPFN may take a long time to run on large datasets. Consider using alternatives (e.g. RandomForestClassifier)')
        elif X.shape[0] > 10000 and self.base_classifier.__class__.__name__ == 'TabPFNClassifier':
            print('WARNING: CAAFE may take a long time to run on large datasets.')
        ds = ['dataset', X, y, [], self.feature_names + [target_name], {}, dataset_description]
        df_train = pd.DataFrame(X, columns=self.feature_names)
        df_train[target_name] = y
        if disable_caafe:
            self.code = ''
        else:
            self.code, prompt, messages = generate_features(ds, df_train, model=self.llm_model, iterative=self.iterations, metric_used=auc_metric, iterative_method=self.base_classifier, display_method='markdown', n_splits=self.n_splits, n_repeats=self.n_repeats)
        df_train = run_llm_code(self.code, df_train)
        df_train, _, self.mappings = make_datasets_numeric(df_train, df_test=None, target_column=target_name, return_mappings=True)
        df_train, y = split_target_column(df_train, target_name)
        X, y = (df_train.values, y.values.astype(int))
        self.classes_ = unique_labels(y)
        self.base_classifier.fit(X, y)
        return self

    def predict_preprocess(self, X):
        """
        Helper functions for preprocessing the data before making predictions.

        Parameters:
        X (pandas.DataFrame): The DataFrame to make predictions on.

        Returns:
        numpy.ndarray: The preprocessed input data.
        """
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=self.X_.columns)
        X, _ = split_target_column(X, self.target_name)
        X = run_llm_code(self.code, X)
        X = make_dataset_numeric(X, mappings=self.mappings)
        X = X.values
        return X

    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict_proba(X)

    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict(X)

def run_llm_code(code: str, df: pd.DataFrame, convert_categorical_to_integer: Optional[bool]=True, fill_na: Optional[bool]=True) -> pd.DataFrame:
    """
    Executes the given code on the given dataframe and returns the resulting dataframe.

    Parameters:
    code (str): The code to execute.
    df (pandas.DataFrame): The dataframe to execute the code on.
    convert_categorical_to_integer (bool, optional): Whether to convert categorical columns to integer values. Defaults to True.
    fill_na (bool, optional): Whether to fill NaN values in object columns with empty strings. Defaults to True.

    Returns:
    pandas.DataFrame: The resulting dataframe after executing the code.
    """
    try:
        loc = {}
        df = copy.deepcopy(df)
        if fill_na and False:
            df.loc[:, df.dtypes == object] = df.loc[:, df.dtypes == object].fillna('')
        if convert_categorical_to_integer and False:
            df = df.apply(convert_categorical_to_integer_f)
        access_scope = {'df': df, 'pd': pd, 'np': np}
        parsed = ast.parse(code)
        check_ast(parsed)
        exec(compile(parsed, filename='<ast>', mode='exec'), access_scope, loc)
        df = copy.deepcopy(df)
    except Exception as e:
        print('Code could not be executed', e)
        raise e
    return df

def split_target_column(df: pd.DataFrame, target: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Splits the given dataframe into the feature dataframe and the target column.

    Parameters:
    df (pandas.DataFrame): The dataframe to split.
    target (str, optional): The name of the target column. Defaults to None.

    Returns:
    Tuple[pandas.DataFrame, Optional[pandas.Series]]: The feature dataframe and the target column (if it exists).
    """
    return (df[[c for c in df.columns if c != target]], df[target].astype(int) if target and target in df.columns else None)

