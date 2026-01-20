# Cluster 98

class UngarCoefficients(CoefficientBase):

    def get_coefficient_df(self, corpus, document_scores):
        from statsmodels.regression.linear_model import OLS
        '\n\n        :param corpus: TermDocMatrix, should just have unigrams\n        :param document_scores: np.array, continuous value for each document score\n        :return: pd.DataFrame\n        '
        assert document_scores.shape == (corpus.get_num_docs(),)
        if any((' ' in t for t in self._get_terms(corpus))):
            logging.warning('UngerCoefficients is currently designed for only unigram terms. Run corpus.get_unigram_corpus() before using this.')
        X = ungar_transform(self._get_tdm(corpus))
        model = OLS(document_scores, X.T).fit()
        df = pd.DataFrame({'Word': self._get_terms(corpus), 'Beta': model.params, 'Tstat': model.tvalues, 'Frequency': corpus.get_term_doc_mat().sum(axis=0)[0].A1}).set_index('Word')
        return df

def ungar_transform(tdm):
    pw = tdm.sum(axis=0).A1 / tdm.sum()
    tdmd = tdm.todense()
    tdmdpw = tdmd.T
    tdmdpwln = (tdmdpw / tdmdpw.sum(axis=1)).T
    print(tdmdpwln.shape, tdm.shape)
    X = 2 * np.sqrt(tdmdpwln + 3.0 / 8)
    return X.T

