# Cluster 11

class TermDocMatrixFilter(object):
    """
	Filter out terms below a particular frequency or pmi threshold.
	"""

    def __init__(self, pmi_threshold_coef=DEFAULT_PMI_THRESHOLD_COEFFICIENT, minimum_term_freq=3):
        """
		Parameters
		----------
		pmi_threshold_coef : float
			Bigram filtering threshold (2 * PMI). Default 2.
		minimum_term_freq : int
			Minimum number of times term has to appear.  Default 3.

		"""
        self._threshold_coef = pmi_threshold_coef
        self._min_freq = minimum_term_freq

    def filter(self, term_doc_matrix):
        """
		Parameters
		----------
		term_doc_matrix  : TermDocMatrix

		Returns
		-------
		TermDocMatrix pmi-filterd term doc matrix
		"""
        df = term_doc_matrix.get_term_freq_df()
        if len(df) == 0:
            return term_doc_matrix
        low_pmi_bigrams = get_low_pmi_bigrams(self._threshold_coef, df).index
        infrequent_terms = df[df.sum(axis=1) < self._min_freq].index
        filtered_term_doc_mat = term_doc_matrix.remove_terms(set(low_pmi_bigrams) | set(infrequent_terms))
        try:
            filtered_term_doc_mat.get_term_freq_df()
        except ValueError:
            raise AtLeastOneCategoryHasNoTermsException()
        return filtered_term_doc_mat

def get_low_pmi_bigrams(threshold_coef, word_freq_df):
    is_bigram = np.array([' ' in word for word in word_freq_df.index])
    unigram_freq = word_freq_df[~is_bigram].sum(axis=1)
    bigram_freq = word_freq_df[is_bigram].sum(axis=1)
    bigram_prob = bigram_freq / bigram_freq.sum()
    unigram_prob = unigram_freq / unigram_freq.sum()

    def get_pmi(bigram):
        try:
            return np.log(bigram_prob[bigram] / np.product([unigram_prob[word] for word in bigram.split(' ')])) / np.log(2)
        except:
            return 0
    low_pmi_bigrams = bigram_prob[bigram_prob.index.map(get_pmi) < threshold_coef * 2]
    return low_pmi_bigrams

