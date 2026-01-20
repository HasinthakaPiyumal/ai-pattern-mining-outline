# Cluster 95

def try_to_import_sentence_transformers():
    try:
        import sentence_transformers
    except ImportError:
        raise ValueError('Could not import sentence-transformers python package.\n                Please install it with `pip install sentence-transformers`.')

