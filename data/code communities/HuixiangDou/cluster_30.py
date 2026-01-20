# Cluster 30

def process_data(documents: list, pid: int):
    t0 = time.time()
    new_words = load_newwords()
    for w in new_words:
        jieba.add_word(w, tag='n')
    stop_words = load_stopwords()
    print('{} start..'.format(pid))
    bad_patterns = [image_name, chapter2, chapter3]
    for filename, filepath in documents:
        d = ''
        with open(filepath) as f:
            d = f.read()
            head_length = int(len(d) * 0.8)
            d = d[0:head_length]
        cuts = [w.word for w in jp.cut(d)]
        filtered = []
        for c in cuts:
            c = c.strip()
            if c in stop_words:
                continue
            if 'images' == c:
                continue
            skip = False
            for bad_pattern in bad_patterns:
                if bad_pattern.match(c):
                    skip = True
                    break
            if skip:
                continue
            filtered.append(c)
        if len(filtered) < 1:
            continue
        new_content = ' '.join(filtered)
        if len(new_content) < 300:
            continue
        dirname = os.path.join('preprocess', str(pid))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        hashname = content_hash(new_content)
        outfilepath = os.path.join(dirname, hashname + '.md')
        with open('name_map.txt', 'a') as f:
            f.write('{}\t {}'.format(hashname, filepath))
            f.write('\n')
        with open(outfilepath, 'w') as f:
            f.write(new_content)
            f.flush()
    print('{} finish, timecost {}'.format(pid, time.time() - t0))

def load_newwords():
    words = []
    basename = './newwords'
    files = os.listdir(basename)
    for filename in files:
        filepath = os.path.join(basename, filename)
        with open(filepath, encoding='utf8') as f:
            words += json.load(f)
        print('load {}'.format(filepath))
    return words

def load_stopwords():
    sw = []
    with open('cn_en_stopwords.txt') as f:
        for line in f:
            if len(line.strip()) > 0:
                sw.append(line.strip())
    return sw

def content_hash(input_str: str):
    hash_object = hashlib.sha256()
    hash_object.update(input_str.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig[:6]

def main():
    debug_mode = False
    processes = []
    split_documents = load_documents(n=_get_num_processes())
    for process_id, documents in enumerate(split_documents):
        print(f'Distributing to process[{process_id}]...')
        if debug_mode:
            process_data(documents, process_id)
        else:
            process = Process(target=process_data, args=(documents, process_id))
            process.start()
            print(f'Distributed to process[{process_id}].')
            processes.append(process)
    for process in processes:
        process.join()

def load_documents(n: int=1):
    basedir = '/home/data/khj/workspace/huixiangdou/repodir.lda'
    docs = []
    for root, _, files in os.walk(basedir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                pdb.set_trace()
            else:
                docs.append((file, os.path.join(root, file)))
    length = len(docs)
    step = length // n
    remainder = length % n
    result = []
    start = 0
    for i in range(n):
        end = start + step + (1 if i < remainder else 0)
        result.append(docs[start:end])
        start = end
    return result

def _get_num_processes():
    num_processes = cpu_count() - 1
    return num_processes

