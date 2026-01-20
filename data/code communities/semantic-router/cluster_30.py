# Cluster 30

def load_img(url):
    resp = requests.get(url)
    return Image.open(BytesIO(resp.content))

