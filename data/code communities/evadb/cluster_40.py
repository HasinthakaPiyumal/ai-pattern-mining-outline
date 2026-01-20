# Cluster 40

def main():
    story_path = download_story()
    ask_question(story_path)

def ask_question(story_path: str):
    llm = GPT4All('ggml-model-gpt4all-falcon-q4_0.bin')
    path = os.path.dirname(evadb.__file__)
    cursor = evadb.connect().cursor()
    story_table = 'TablePPText'
    story_feat_table = 'FeatTablePPText'
    index_table = 'IndexTable'
    timestamps = {}
    t_i = 0
    timestamps[t_i] = perf_counter()
    print('Setup Function')
    Text_feat_function_query = f"CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor\n            IMPL  '{path}/functions/sentence_feature_extractor.py';\n            "
    cursor.query('DROP FUNCTION IF EXISTS SentenceFeatureExtractor;').execute()
    cursor.query(Text_feat_function_query).execute()
    cursor.query(f'DROP TABLE IF EXISTS {story_table};').execute()
    cursor.query(f'DROP TABLE IF EXISTS {story_feat_table};').execute()
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()
    print(f'Time: {(timestamps[t_i] - timestamps[t_i - 1]) * 1000:.3f} ms')
    print('Create table')
    cursor.query(f'CREATE TABLE {story_table} (id INTEGER, data TEXT(1000));').execute()
    for i, text in enumerate(read_text_line(story_path)):
        print('text: --' + text + '--')
        ascii_text = unidecode(text)
        cursor.query(f"INSERT INTO {story_table} (id, data)\n                VALUES ({i}, '{ascii_text}');").execute()
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()
    print(f'Time: {(timestamps[t_i] - timestamps[t_i - 1]) * 1000:.3f} ms')
    print('Extract features')
    cursor.query(f'CREATE TABLE {story_feat_table} AS\n        SELECT SentenceFeatureExtractor(data), data FROM {story_table};').execute()
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()
    print(f'Time: {(timestamps[t_i] - timestamps[t_i - 1]) * 1000:.3f} ms')
    print('Create index')
    cursor.query(f'CREATE INDEX {index_table} ON {story_feat_table} (features) USING QDRANT;').execute()
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()
    print(f'Time: {(timestamps[t_i] - timestamps[t_i - 1]) * 1000:.3f} ms')
    print('Query')
    question = 'Who is Count Cyril Vladmirovich?'
    ascii_question = unidecode(question)
    res_batch = cursor.query(f"SELECT data FROM {story_feat_table}\n        ORDER BY Similarity(SentenceFeatureExtractor('{ascii_question}'),features)\n        LIMIT 5;").execute()
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()
    print(f'Time: {(timestamps[t_i] - timestamps[t_i - 1]) * 1000:.3f} ms')
    print('Merge')
    context_list = []
    for i in range(len(res_batch)):
        context_list.append(res_batch.frames[f'{story_feat_table.lower()}.data'][i])
    context = '\n'.join(context_list)
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()
    print(f'Time: {(timestamps[t_i] - timestamps[t_i - 1]) * 1000:.3f} ms')
    print('LLM')
    query = f'If the context is not relevant, please answer the question by using your own knowledge about the topic.\n    \n    {context}\n    \n    Question : {question}'
    full_response = llm.generate(query)
    print(full_response)
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()
    print(f'Time: {(timestamps[t_i] - timestamps[t_i - 1]) * 1000:.3f} ms')
    print(f'Total Time: {(timestamps[t_i] - timestamps[0]) * 1000:.3f} ms')

