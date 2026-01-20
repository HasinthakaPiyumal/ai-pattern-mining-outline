# Cluster 4

def gen_topics():
    elements = list(range(8))
    combinations = list(itertools.combinations(elements, 2))
    expanded_combinations = []
    while len(expanded_combinations) < total:
        expanded_combinations.extend(combinations)
    expanded_combinations = expanded_combinations[:total]
    random.shuffle(expanded_combinations)
    return expanded_combinations

