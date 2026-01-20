# Cluster 3

def serialize_tfrecords(tfrecords_fn, datadir='ml-1m', download=False):
    if download is True:
        print('Downloading MovieLens-1M dataset ...')
        _download_and_unzip(datadir + '.zip')
    users_data = _load_data(datadir + '/users.dat', columns=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    movies_data = _load_data(datadir + '/movies.dat', columns=['MovieID', 'Title', 'Genres'])
    ratings_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    writer = tf.io.TFRecordWriter(tfrecords_fn)
    shuffled_filename = _shuffle_data(datadir + '/ratings.dat')
    f = open(shuffled_filename, 'r', encoding='unicode_escape')
    for line in f:
        ls = line.strip().split('::')
        rating = dict(zip(ratings_columns, ls))
        rating.update(users_data.get(ls[0]))
        rating.update(movies_data.get(ls[1]))
        for c in ['Age', 'Occupation', 'Rating', 'Timestamp']:
            rating[c] = int(rating[c])
        for c in ['UserID', 'MovieID', 'Gender', 'Zip-code', 'Title']:
            rating[c] = rating[c].encode('utf-8')
        rating['Genres'] = [x.encode('utf-8') for x in rating['Genres'].split('|')]
        serialized = _serialize_example(rating)
        writer.write(serialized)
    writer.close()
    f.close()

def _download_and_unzip(filename='ml-1m.zip'):
    import requests
    import zipfile
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)
    f = zipfile.ZipFile(filename)
    f.extractall()

def _load_data(filename, columns):
    data = {}
    with open(filename, 'r', encoding='unicode_escape') as f:
        for line in f:
            ls = line.strip('\n').split('::')
            data[ls[0]] = dict(zip(columns[1:], ls[1:]))
    return data

def _shuffle_data(filename):
    shuffled_filename = f'{filename}.shuffled'
    with open(filename, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(shuffled_filename, 'w') as f:
        f.writelines(lines)
    return shuffled_filename

def _serialize_example(feature):
    serialize_feature = {}
    for c in ['Age', 'Occupation', 'Rating', 'Timestamp']:
        serialize_feature[c] = tf.train.Feature(int64_list=tf.train.Int64List(value=[feature[c]]))
    for c in ['UserID', 'MovieID', 'Gender', 'Zip-code', 'Title']:
        serialize_feature[c] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature[c]]))
    serialize_feature['Genres'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=feature['Genres']))
    example_proto = tf.train.Example(features=tf.train.Features(feature=serialize_feature))
    return example_proto.SerializeToString()

