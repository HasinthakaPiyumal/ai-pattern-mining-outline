# Cluster 6

def get_gpt_emb(prompt):
    embedding = openai.Embedding.create(input=prompt, model='text-embedding-ada-002')['data'][0]['embedding']
    return np.array(embedding)

