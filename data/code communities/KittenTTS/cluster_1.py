# Cluster 1

def basic_english_tokenize(text):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re
    tokens = re.findall('\\w+|[^\\w\\s]', text)
    return tokens

