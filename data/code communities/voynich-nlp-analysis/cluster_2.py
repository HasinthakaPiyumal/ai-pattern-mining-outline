# Cluster 2

def strip_suffix(word):
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix):
            return re.sub(f'{suffix}$', '', word)
    return word

