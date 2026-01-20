# Cluster 2

def extract_urls(text):
    url_pattern = re.compile('(https?://\\S+)')
    return url_pattern.findall(text)

