# Cluster 46

def query(question):
    context_docs = cursor.query(f"\n        SELECT data\n        FROM embedding_table\n        ORDER BY Similarity(embedding('{question}'), features)\n        LIMIT 5;\n    ").df()
    context = '\n'.join(context_docs['embedding_table.data'])
    llm = GPT4All('ggml-model-gpt4all-falcon-q4_0.bin')
    llm.set_thread_count(16)
    message = f'If the context is not relevant, please answer the question by using your own knowledge about the topic.\n    \n    {context}\n    \n    Question : {question}'
    answer = llm.generate(message)
    print('\n> Answer:', answer)

