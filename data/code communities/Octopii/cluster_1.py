# Cluster 1

def keywords_classify_pii(rules, intelligible_text_list):
    scores = {}
    for key, rule in rules.items():
        scores[key] = 0
        keywords = rule.get('keywords', [])
        if keywords is not None:
            for intelligible_text_word in intelligible_text_list:
                for keywords_word in keywords:
                    if similarity(intelligible_text_word.lower().replace('.', '').replace("'", '').replace('-', '').replace('_', '').replace(',', ''), keywords_word.lower()) > 80:
                        scores[key] += 1
    return scores

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio() * 100

