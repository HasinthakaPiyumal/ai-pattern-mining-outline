# Cluster 10

def wav2feature(root_directory, save_directory, name, win_len, win_step, mode, feature_len, seq2seq, save):
    count = 0
    dirid = 0
    level = 'cha' if seq2seq is False else 'seq2seq'
    data_dir = os.path.join(root_directory, name)
    preprocess(data_dir)
    for subdir, dirs, files in os.walk(data_dir):
        for f in files:
            fullFilename = os.path.join(subdir, f)
            filenameNoSuffix = os.path.splitext(fullFilename)[0]
            if f.endswith('.wav'):
                rate = None
                sig = None
                try:
                    rate, sig = wav.read(fullFilename)
                except ValueError as e:
                    if e.message == "File format 'NIST'... not understood.":
                        sf = Sndfile(fullFilename, 'r')
                    nframes = sf.nframes
                    sig = sf.read_frames(nframes)
                    rate = sf.samplerate
                feat = calcfeat_delta_delta(sig, rate, win_length=win_len, win_step=win_step, mode=mode, feature_len=feature_len)
                feat = preprocessing.scale(feat)
                feat = np.transpose(feat)
                print(feat.shape)
                labelFilename = filenameNoSuffix + '.label'
                with open(labelFilename, 'r') as f:
                    characters = f.readline().strip().lower()
                targets = []
                if seq2seq is True:
                    targets.append(28)
                for c in characters:
                    if c == ' ':
                        targets.append(0)
                    elif c == "'":
                        targets.append(27)
                    else:
                        targets.append(ord(c) - 96)
                if seq2seq is True:
                    targets.append(29)
                print(targets)
                if save:
                    count += 1
                    if count % 4000 == 0:
                        dirid += 1
                    print('file index:', count)
                    print('dir index:', dirid)
                    label_dir = os.path.join(save_directory, level, name, str(dirid), 'label')
                    feat_dir = os.path.join(save_directory, level, name, str(dirid), 'feature')
                    if not os.path.isdir(label_dir):
                        os.makedirs(label_dir)
                    if not os.path.isdir(feat_dir):
                        os.makedirs(feat_dir)
                    featureFilename = os.path.join(feat_dir, filenameNoSuffix.split('/')[-1] + '.npy')
                    np.save(featureFilename, feat)
                    t_f = os.path.join(label_dir, filenameNoSuffix.split('/')[-1] + '.npy')
                    print(t_f)
                    np.save(t_f, targets)

def preprocess(root_directory):
    """
    Function to walk through the directory and convert flac to wav files
    """
    try:
        check_call(['flac'])
    except OSError:
        raise OSError('Flac not installed. Install using apt-get install flac')
    for subdir, dirs, files in os.walk(root_directory):
        for f in files:
            filename = os.path.join(subdir, f)
            if f.endswith('.flac'):
                try:
                    check_call(['flac', '-d', filename])
                    os.remove(filename)
                except CalledProcessError as e:
                    print('Failed to convert file {}'.format(filename))
            elif f.endswith('.TXT'):
                os.remove(filename)
            elif f.endswith('.txt'):
                with open(filename, 'r') as fp:
                    lines = fp.readlines()
                    for line in lines:
                        sub_n = line.split(' ')[0] + '.label'
                        subfile = os.path.join(subdir, sub_n)
                        sub_c = ' '.join(line.split(' ')[1:])
                        sub_c = sub_c.lower()
                        with open(subfile, 'w') as sp:
                            sp.write(sub_c)
            elif f.endswith('.wav'):
                if not os.path.isfile(os.path.splitext(filename)[0] + '.label'):
                    raise ValueError('.label file not found for {}'.format(filename))
            else:
                pass

