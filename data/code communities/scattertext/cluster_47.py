# Cluster 47

def tweet_tokenizier_factory(tweet_tokenizer):
    """
    Parameters
    ----------
    tweet_tokenizer

    Doc of tweets

    Notes
    -------
    Requires NLTK to be installed :(

    """
    return nltk_tokenzier_factory(tweet_tokenizer)

def nltk_tokenzier_factory(nltk_tokenizer):
    """
    Parameters
    ----------
    nltk_tokenizer : nltk.tokenize.* instance (e.g., nltk.TreebankWordTokenizer())

    Returns
    -------
    Doc of tweets

    Notes
    -------
    Requires NLTK to be installed
    """

    def tokenize(text):
        toks = []
        for tok in nltk_tokenizer.tokenize(text):
            if len(tok) > 0:
                toks.append(Tok(_get_pos_tag(tok), tok.lower(), tok.lower(), ent_type='', tag=''))
        return Doc([toks], text)
    return tokenize

