# Cluster 0

def main():
    """Loads the model and processes it.
    
    The model used can be installed by running this command on your CMD/Terminal:

    python -m spacy download es_core_news_md
    
    """
    corpus = open('transcript_clean.txt', 'r', encoding='utf-8').read()
    nlp = spacy.load('es_core_news_md')
    nlp.max_length = len(corpus)
    doc = nlp(corpus)
    get_tokens(doc)
    get_entities(doc)
    get_sentences(doc)

