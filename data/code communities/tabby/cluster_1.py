# Cluster 1

def read_dataframe_from_file(language: str, file: str) -> pd.DataFrame:
    whole_path_file = './data/' + MODEL_ID.split('/')[-1] + '/' + language + '/' + file
    objs = []
    with open(whole_path_file) as fin:
        for line in fin:
            obj = json.loads(line)
            if 'crossfile_context' in obj.keys():
                obj['raw_prompt'] = obj['crossfile_context']['text'] + obj['prompt']
            else:
                obj['raw_prompt'] = obj['prompt']
            objs.append(obj)
    df = pd.DataFrame(objs)
    return df

def write_log(log: str):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('./modal/log.txt', 'a') as f:
        f.write(f'{now} : {log}')
        f.write('\n')

def chunker(seq, size) -> List:
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

