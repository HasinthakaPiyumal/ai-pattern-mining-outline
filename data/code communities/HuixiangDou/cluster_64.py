# Cluster 64

def save(_id, sentence):
    if _id not in queries:
        queries[_id] = [sentence]
    else:
        queries[_id].append(sentence)

