# Cluster 51

class TermDocMatrixWithoutCategoriesFromPandas(TermDocMatrixFactory):

    def __init__(self, data_frame, text_col, clean_function=lambda x: x, nlp=None, feats_from_spacy_doc=None, verbose=False):
        """Creates a TermDocMatrix from a pandas data frame.

        Parameters
        ----------
        data_frame : pd.DataFrame
            The data frame that contains columns for the category of interest
            and the document text.
        text_col : str
            The name of the column which contains each document's raw text.
        clean_function : function, optional
            A function that strips invalid characters out of the document text string,
            returning the new string.
        nlp : function, optional
        feats_from_spacy_doc : FeatsFromSpacyDoc or None
        verbose : boolean, optional
            If true, prints a message every time a document index % 100 is 0.

        See Also
        --------
        TermDocMatrixFactory
        """
        TermDocMatrixFactory.__init__(self, clean_function=clean_function, nlp=nlp, feats_from_spacy_doc=feats_from_spacy_doc)
        self.data_frame = data_frame.reset_index()
        self._text_col = text_col
        self._verbose = verbose

    def build(self):
        """Constructs the term doc matrix.

        Returns
        -------
        TermDocMatrix
        """
        X_factory = CSRMatrixFactory()
        mX_factory = CSRMatrixFactory()
        term_idx_store = IndexStore()
        metadata_idx_store = IndexStore()
        parse_pipeline = ParsePipelineFactoryWithoutCategories(self.get_nlp(), X_factory, mX_factory, term_idx_store, metadata_idx_store, self)
        df = self._clean_and_filter_nulls_and_empties_from_dataframe()
        tdm = self._apply_pipeline_and_get_build_instance(X_factory, mX_factory, df, parse_pipeline, term_idx_store, metadata_idx_store)
        return tdm

    def _apply_pipeline_and_get_build_instance(self, X_factory, mX_factory, df, parse_pipeline, term_idx_store, metadata_idx_store):
        df.apply(parse_pipeline.parse, axis=1)
        X, mX = build_sparse_matrices_with_num_docs(len(df), X_factory, mX_factory)
        tdm = TermDocMatrixWithoutCategories(X, mX, term_idx_store, metadata_idx_store)
        return tdm

    def _clean_and_filter_nulls_and_empties_from_dataframe(self):
        df = self.data_frame.loc[self.data_frame[[self._text_col]].dropna().index]
        df = df[df[self._text_col] != ''].reset_index()
        return df

def build_sparse_matrices_with_num_docs(num_docs, X_factory, mX_factory):
    return (X_factory.set_last_row_idx(num_docs - 1).get_csr_matrix(), mX_factory.set_last_row_idx(num_docs - 1).get_csr_matrix())

