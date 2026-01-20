# Cluster 6

def get_num_classes(level):
    if level == 'phn':
        num_classes = 62
    elif level == 'cha':
        num_classes = 29
    elif level == 'seq2seq':
        num_classes = 30
    else:
        raise ValueError('level must be phn, cha or seq2seq, but the given level is %s' % str(level))
    return num_classes

def check_path_exists(path):
    """ check a path exists or not
    """
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    elif not os.path.exists(path):
        os.makedirs(path)

class CorpusGardener(object):
    """
    Preprocessing multiple language corpuses, and gathering
    them into batches
    """

    def __init__(self, remove_duplicate_space=True):
        self.remove_dubplicate_space = remove_duplicate_space
        self.save_dir = '/home/pony/github/data/spellingChecker/raw'

    def process_poetry(self, data_dir='/media/pony/DLdigest/data/languageModel/chinese-poetry/json'):
        """
        Process Tang and Song poems dataset
        """
        save_dir = os.path.join(self.save_dir, 'poem')
        check_path_exists(save_dir)
        count = 0
        for entry in os.scandir(data_dir):
            if entry.name.startswith('poet'):
                with open(entry.path, 'r') as json_file:
                    poems = json.load(json_file)
                    for p in poems:
                        paras = HanziConv.toSimplified(''.join(p['paragraphs']).replace('\n', ''))
                        paras = filter_punctuation(paras)
                        for para in paras.split(' '):
                            if len(para.strip()) > 1:
                                pys = ' '.join(np.array(pinyin(para)).flatten())
                                with open(os.path.join(save_dir, str(count // 400000 + 1) + '.txt'), 'a') as f:
                                    f.write(para + ',' + pys + '\n')
                                count += 1

    def process_dureader(self, data_dir='/media/pony/DLdigest/data/languageModel/dureader-raw/'):
        """
        Processing Baidu released QA Reader Dataset
        """
        save_dir = os.path.join(self.save_dir, 'dureader')
        check_path_exists(save_dir)
        count = 0
        for entry in os.scandir(data_dir):
            if entry.name.endswith('json'):
                print(entry.path)
                with open(entry.path, 'r') as f:
                    for line in f:
                        contents = json.loads(line)
                        con = []
                        try:
                            answers = ''.join(contents['answers'])
                            con.append(answers)
                            questions = contents['question']
                            con.append(questions)
                            for doc in contents['documents']:
                                paragraphs = ''.join(doc['paragraphs'])
                                title = doc['title']
                                con.append(paragraphs)
                                con.append(title)
                            con = HanziConv.toSimplified(''.join(con).replace('\n', ''))
                            cons = filter_punctuation(con)
                            for c in cons.split(' '):
                                if len(c.strip()) > 1:
                                    pys = ' '.join(np.array(pinyin(c)).flatten())
                                    count += 1
                                    with open(os.path.join(save_dir, str(count // 400000 + 1) + '.txt'), 'a') as f:
                                        f.write(c + ',' + pys + '\n')
                        except KeyError:
                            continue

    def process_audioLabels(self, data_dir='/media/pony/DLdigest/data/ASR_zh/'):
        """
        Processing label files in collected Chinese audio dataset
        """
        save_dir = os.path.join(self.save_dir, 'audioLabels')
        check_path_exists(save_dir)
        count = 0
        for subdir, dirs, files in os.walk(data_dir):
            print(subdir)
            for f in files:
                if f.endswith('label'):
                    fullFilename = os.path.join(subdir, f)
                    with open(fullFilename, 'r') as f:
                        line = f.read()
                        con = HanziConv.toSimplified(line)
                        con = filter_punctuation(con)
                        for c in con.split(' '):
                            if len(c.strip()) > 1:
                                pys = ' '.join(np.array(pinyin(c)).flatten())
                                count += 1
                                with open(os.path.join(save_dir, str(count // 400000 + 1) + '.txt'), 'a') as f:
                                    f.write(c + ',' + pys + '\n')

def split_data_by_s5(src_dir, des_dir, keywords=['train_si284', 'test_eval92', 'test_dev93']):
    count = 0
    for key in keywords:
        wav_file_list = os.path.join(src_dir, key + '.flist')
        label_file_list = os.path.join(src_dir, key + '.txt')
        new_path = check_path_exists(os.path.join(des_dir, key))
        with open(wav_file_list, 'r') as wfl:
            wfl_contents = wfl.readlines()
            for line in wfl_contents:
                line = line.strip()
                if os.path.isfile(line):
                    shutil.copyfile(line, os.path.join(des_dir, key, line.split('/')[-1]))
                    print(line)
                else:
                    tmp = '/'.join(line.split('/')[:-1] + [line.split('/')[-1].upper()])
                    shutil.copyfile(tmp, os.path.join(des_dir, key, line.split('/')[-1].replace('WV1', 'wv1')))
                    print(tmp)
        with open(label_file_list, 'r') as lfl:
            lfl_contents = lfl.readlines()
            for line in lfl_contents:
                label = ' '.join(line.strip().split(' ')[1:])
                with open(os.path.join(des_dir, key, line.strip().split(' ')[0] + '.label'), 'w') as lf:
                    lf.writelines(label)
                print(key, label)

