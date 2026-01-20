# Cluster 28

def filecontents(dirname: str):
    filepaths = files()
    for _, filepath in filepaths:
        with open(filepath) as f:
            content = f.read()
            if len(content) > 0:
                yield content

def files():
    basedir = '/home/data/khj/workspace/huixiangdou/lda/preprocess'
    docs = []
    for root, _, files in os.walk(basedir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                pdb.set_trace()
            else:
                docs.append((file, os.path.join(root, file)))
    return docs

def build_topic(dirname: str='preprocess'):
    namemap = load_namemap()
    pdb.set_trace()
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
    t0 = time()
    tf = tf_vectorizer.fit_transform(filecontents(dirname))
    print('BoW in %0.3fs.' % (time() - t0))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50.0, random_state=0)
    t0 = time()
    doc_types = lda.fit_transform(tf)
    pdb.set_trace()
    print('lda train in %0.3fs.' % (time() - t0))
    feature_names = tf_vectorizer.get_feature_names_out()
    models = {'CountVectorizer': tf_vectorizer, 'LatentDirichletAllocation': lda}
    with open('lda_models.pkl', 'wb') as model_file:
        pkl.dump(models, model_file)
    top_features_list = []
    for _, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]
        top_features_list.append(top_features.tolist())
    with open(os.path.join('cluster', 'desc.json'), 'w') as f:
        json_str = json.dumps(top_features_list, ensure_ascii=False)
        f.write(json_str)
    filepaths = files()
    pdb.set_trace()
    for file_id, doc_score in enumerate(doc_types):
        basename, input_filepath = filepaths[file_id]
        hashname = basename.split('.')[0]
        source_filepath = namemap[hashname]
        indices_np = np.where(doc_score > 0.1)[0]
        for topic_id in indices_np:
            target_dir = os.path.join('cluster', str(topic_id))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copy(source_filepath, target_dir)

def load_namemap():
    namemap = dict()
    with open('name_map.txt') as f:
        for line in f:
            parts = line.split('\t')
            namemap[parts[0].strip()] = parts[1].strip()
    return namemap

