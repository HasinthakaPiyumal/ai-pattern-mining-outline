# Cluster 1

def pre_build_data_cache(infile, outfile, feature_cnt, batch_size):
    wt = open(outfile, 'wb')
    for labels, features in load_data_from_file_batching(infile, batch_size):
        input_in_sp = prepare_data_4_sp(labels, features, feature_cnt)
        pickle.dump(input_in_sp, wt)
    wt.close()

def load_data_from_file_batching(file, batch_size):
    labels = []
    features = []
    cnt = 0
    with open(file, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            cnt += 1
            if '#' in line:
                punc_idx = line.index('#')
            else:
                punc_idx = len(line)
            label = float(line[0:1])
            if label > 1:
                label = 1
            feature_line = line[2:punc_idx]
            words = feature_line.split(' ')
            cur_feature_list = []
            for word in words:
                if not word:
                    continue
                tokens = word.split(':')
                if len(tokens[1]) <= 0:
                    tokens[1] = '0'
                cur_feature_list.append([int(tokens[0]) - 1, float(tokens[1])])
            features.append(cur_feature_list)
            labels.append(label)
            if cnt == batch_size:
                yield (labels, features)
                labels = []
                features = []
                cnt = 0
    if cnt > 0:
        yield (labels, features)

def prepare_data_4_sp(labels, features, dim):
    instance_cnt = len(labels)
    indices = []
    values = []
    values_2 = []
    shape = [instance_cnt, dim]
    feature_indices = []
    for i in range(instance_cnt):
        m = len(features[i])
        for j in range(m):
            indices.append([i, features[i][j][0]])
            values.append(features[i][j][1])
            values_2.append(features[i][j][1] * features[i][j][1])
            feature_indices.append(features[i][j][0])
    res = {}
    res['indices'] = np.asarray(indices, dtype=np.int64)
    res['values'] = np.asarray(values, dtype=np.float32)
    res['values2'] = np.asarray(values_2, dtype=np.float32)
    res['shape'] = np.asarray(shape, dtype=np.int64)
    res['labels'] = np.asarray([[label] for label in labels], dtype=np.float32)
    res['feature_indices'] = np.asarray(feature_indices, dtype=np.int64)
    return res

def pre_build_data_cache_if_need(infile, feature_cnt, batch_size):
    outfile = infile.replace('.csv', '.pkl').replace('.txt', '.pkl')
    if not os.path.isfile(outfile):
        print('pre_build_data_cache for ', infile)
        pre_build_data_cache(infile, outfile, feature_cnt, batch_size)
        print('pre_build_data_cache finished.')

def pre_build_data_cache(infile, outfile, batch_size):
    wt = open(outfile, 'wb')
    for labels, features, qids, docids in load_data_from_file_batching(infile, batch_size):
        input_in_sp = prepare_data_4_sp(labels, features, FEATURE_COUNT)
        pickle.dump((input_in_sp, qids, docids), wt)
    wt.close()

def pre_build_data_cache_if_need(infile, batch_size, rebuild_cache):
    outfile = infile.replace('.csv', '.pkl').replace('.txt', '.pkl')
    if not os.path.isfile(outfile) or rebuild_cache:
        print('pre_build_data_cache for ', infile)
        pre_build_data_cache(infile, outfile, batch_size)
        print('pre_build_data_cache finished.')

