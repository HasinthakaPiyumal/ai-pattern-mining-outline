# Cluster 1

def load_data_and_labels(filename):
    """Load sentences and labels"""
    df = pd.read_csv(filename, compression='zip', dtype={'consumer_complaint_narrative': object})
    selected = ['product', 'consumer_complaint_narrative']
    non_selected = list(set(df.columns) - set(selected))
    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    return (x_raw, y_raw, df, labels)

def clean_str(s):
    """Clean sentence"""
    s = re.sub("[^A-Za-z0-9(),!?\\'\\`]", ' ', s)
    s = re.sub("\\'s", " 's", s)
    s = re.sub("\\'ve", " 've", s)
    s = re.sub("n\\'t", " n't", s)
    s = re.sub("\\'re", " 're", s)
    s = re.sub("\\'d", " 'd", s)
    s = re.sub("\\'ll", " 'll", s)
    s = re.sub(',', ' , ', s)
    s = re.sub('!', ' ! ', s)
    s = re.sub('\\(', ' \\( ', s)
    s = re.sub('\\)', ' \\) ', s)
    s = re.sub('\\?', ' \\? ', s)
    s = re.sub('\\s{2,}', ' ', s)
    s = re.sub('\\S*(x{2,}|X{2,})\\S*', 'xxx', s)
    s = re.sub('[^\\x00-\\x7F]+', '', s)
    return s.strip().lower()

