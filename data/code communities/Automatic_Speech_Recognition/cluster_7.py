# Cluster 7

def wav2feature(rootdir, save_directory, mode, feature_len, level, keywords, win_len, win_step, seq2seq, save):
    feat_dir = os.path.join(save_directory, level, keywords, mode)
    label_dir = os.path.join(save_directory, level, keywords, 'label')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fullFilename = os.path.join(subdir, file)
            filenameNoSuffix = os.path.splitext(fullFilename)[0]
            if file.endswith('.WAV'):
                rate, sig = (16000, np.fromfile(fullFilename, dtype=np.int16)[512:])
                feat = calcfeat_delta_delta(sig, rate, win_length=win_len, win_step=win_step, mode=mode, feature_len=feature_len)
                feat = preprocessing.scale(feat)
                feat = np.transpose(feat)
                print(feat.shape)
                if level == 'phn':
                    labelFilename = filenameNoSuffix + '.PHN'
                    phenome = []
                    with open(labelFilename, 'r') as f:
                        if seq2seq is True:
                            phenome.append(len(phn))
                        for line in f.read().splitlines():
                            s = line.split(' ')[2]
                            p_index = phn.index(s)
                            phenome.append(p_index)
                        if seq2seq is True:
                            phenome.append(len(phn) + 1)
                        print(phenome)
                    phenome = np.array(phenome)
                elif level == 'cha':
                    labelFilename = filenameNoSuffix + '.WRD'
                    phenome = []
                    sentence = ''
                    with open(labelFilename, 'r') as f:
                        for line in f.read().splitlines():
                            s = line.split(' ')[2]
                            sentence += s + ' '
                            if seq2seq is True:
                                phenome.append(28)
                            for c in s:
                                if c == "'":
                                    phenome.append(27)
                                else:
                                    phenome.append(ord(c) - 96)
                            phenome.append(0)
                        phenome = phenome[:-1]
                        if seq2seq is True:
                            phenome.append(29)
                    print(phenome)
                    print(sentence)
                count += 1
                print('file index:', count)
                if save:
                    speaker, sentence_name = filenameNoSuffix.split('/')[-2:]
                    feature_filename = '{}/{}-{}.npy'.format(feat_dir, speaker, sentence_name)
                    np.save(feature_filename, feat)
                    label_filename = '{}/{}-{}.npy'.format(label_dir, speaker, sentence_name)
                    print(label_filename)
                    np.save(label_filename, phenome)

