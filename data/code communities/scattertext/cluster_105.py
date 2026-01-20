# Cluster 105

def extract_JK(pos_seq):
    """The 'JK' method in Handler et al. 2016.
	Returns token positions of valid ngrams."""

    def find_ngrams(input_list, num_):
        """get ngrams of len n from input list"""
        return zip(*[input_list[i:] for i in range(num_)])
    patterns = set(['AN', 'NN', 'AAN', 'ANN', 'NAN', 'NNN', 'NPN'])
    pos_seq = [tag2coarse.get(tag, 'O') for tag in pos_seq]
    pos_seq = [(i, p) for i, p in enumerate(pos_seq)]
    ngrams = [ngram for n in range(1, 4) for ngram in find_ngrams(pos_seq, n)]

    def stringify(s):
        return ''.join((a[1] for a in s))

    def positionify(s):
        return tuple((a[0] for a in s))
    ngrams = filter(lambda x: stringify(x) in patterns, ngrams)
    return [set(positionify(n)) for n in ngrams]

def find_ngrams(input_list, num_):
    """get ngrams of len n from input list"""
    return zip(*[input_list[i:] for i in range(num_)])

def stringify(s):
    return ''.join((a[1] for a in s))

def positionify(s):
    return tuple((a[0] for a in s))

