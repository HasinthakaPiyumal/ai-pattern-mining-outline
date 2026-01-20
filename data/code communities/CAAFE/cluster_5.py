# Cluster 5

def build_prompt_from_df(ds, df, iterative=1):
    data_description_unparsed = ds[-1]
    feature_importance = {}
    samples = ''
    df_ = df.head(10)
    for i in list(df_):
        nan_freq = '%s' % float('%.2g' % (df[i].isna().mean() * 100))
        s = df_[i].tolist()
        if str(df[i].dtype) == 'float64':
            s = [round(sample, 2) for sample in s]
        samples += f'{df_[i].name} ({df[i].dtype}): NaN-freq [{nan_freq}%], Samples {s}\n'
    kwargs = {'data_description_unparsed': data_description_unparsed, 'samples': samples, 'feature_importance': {k: '%s' % float('%.2g' % feature_importance[k]) for k in feature_importance}}
    prompt = get_prompt(df, ds, data_description_unparsed=data_description_unparsed, iterative=iterative, samples=samples)
    return prompt

def get_prompt(df, ds, iterative=1, data_description_unparsed=None, samples=None, **kwargs):
    how_many = 'up to 10 useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance.' if iterative == 1 else 'exactly one useful column'
    return f"""\nThe dataframe `df` is loaded and in memory. Columns are also named attributes.\nDescription of the dataset in `df` (column dtypes might be inaccurate):\n"{data_description_unparsed}"\n\nColumns in `df` (true feature dtypes listed here, categoricals encoded as int):\n{samples}\n    \nThis code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.\nNumber of samples (rows) in training dataset: {int(len(df))}\n    \nThis code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting "{ds[4][-1]}".\nAdditional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.\nThe scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.\nThis code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.\nThe classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.\nAdded columns can be used in other codeblocks, dropped columns are not available anymore.\n\nCode formatting for each added column:\n```python\n# (Feature name and description)\n# Usefulness: (Description why this adds useful real world knowledge to classify "{ds[4][-1]}" according to dataset description and attributes.)\n# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)\n(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)\n```end\n\nCode formatting for dropping columns:\n```python\n# Explanation why the column XX is dropped\ndf.drop(columns=['XX'], inplace=True)\n```end\n\nEach codeblock generates {how_many} and can drop unused columns (Feature selection).\nEach codeblock ends with ```end and starts with "```python"\nCodeblock:\n"""

