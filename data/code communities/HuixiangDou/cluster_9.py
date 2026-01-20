# Cluster 9

def test_split_python_code():
    path = 'huixiangdou/main.py'
    with open(path) as f:
        content = f.read()
    chunks = split_python_code(filepath=path, text=content, metadata={'test': 'meta'})
    for chunk in chunks:
        print(chunk)

