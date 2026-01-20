# Cluster 13

class CorpusFromPandas(TermDocMatrixFromPandas):
    """Creates a Corpus from a pandas data frame.  A Corpus is a Term Document Matrix
	preserves the original texts.

	Parameters
	----------
	data_frame : pd.DataFrame
		The data frame that contains columns for the category of interest
		and the document text.
	text_col : str
		The name of the column which contains the document text.
	category_col : str
		The name of the column which contains the category of interest.
	clean_function : function, optional
		A function that strips invalid characters out of the document text string, returning
		a new string.
	nlp : function, optional
	verbose : boolean, optional
		If true, prints a message every time a document index % 100 is 0.

	See Also
	--------
	TermDocMatrixFromPandas
	"""

    def _apply_pipeline_and_get_build_instance(self, X_factory, mX_factory, category_idx_store, df, parse_pipeline, term_idx_store, metadata_idx_store, y):
        """
		Parameters
		----------
		X_factory
		mX_factory
		category_idx_store
		df
		parse_pipeline
		term_idx_store
		metadata_idx_store
		y

		Returns
		-------
		CorpusDF
		"""
        df.apply(parse_pipeline.parse, axis=1)
        y = np.array(y)
        X, mX = build_sparse_matrices(y, X_factory, mX_factory)
        return CorpusDF(df, X, mX, y, self._text_col, term_idx_store, category_idx_store, metadata_idx_store)

def build_sparse_matrices(y, X_factory, mX_factory):
    return build_sparse_matrices_with_num_docs(len(y), X_factory, mX_factory)

class CorpusFromFeatureDict(object):

    def __init__(self, df, category_col, text_col, feature_col, metadata_col=None, parsed_col=None):
        """
		Parameters
		----------
		df : pd.DataFrame
		 contains category_col, and parse_col, were parsed col is entirely spacy docs
		category_col : str
				name of category column in convention_df
		text_col : str
				The name of the column which contains each document's raw text.
		feature_col : str
				name of column in convention_df with a feature dictionary
		metadata_col : str, optional
				name of column in convention_df with a meatadata dictionary
		parsed_col : str, optional
				name of column in convention_df with parsed strings
		"""
        self._df = df.reset_index()
        self._category_col = category_col
        self._text_col = text_col
        self._feature_col = feature_col
        self._parsed_col = parsed_col
        self._metadata_col = metadata_col
        self._category_idx_store = IndexStore()
        self._X_factory = CSRMatrixFactory()
        self._mX_factory = CSRMatrixFactory()
        self._term_idx_store = IndexStore()
        self._metadata_idx_store = IndexStore()

    def build(self):
        """Constructs the term doc matrix.

		Returns
		-------
		scattertext.ParsedCorpus.ParsedCorpus
		"""
        self._y = self._get_y_and_populate_category_idx_store()
        self._df.apply(self._add_to_x_factory, axis=1)
        self._X = self._X_factory.set_last_row_idx(len(self._y) - 1).get_csr_matrix()
        self._mX = self._mX_factory.set_last_row_idx(len(self._y) - 1).get_csr_matrix()
        if self._parsed_col is not None and self._parsed_col in self._df:
            return ParsedCorpus(self._df, self._X, self._mX, self._y, self._term_idx_store, self._category_idx_store, self._metadata_idx_store, self._parsed_col, self._category_col)
        else:
            return CorpusDF(self._df, self._X, self._mX, self._y, self._text_col, self._term_idx_store, self._category_idx_store, self._metadata_idx_store)

    def _get_y_and_populate_category_idx_store(self):
        return np.array(self._df[self._category_col].apply(str).apply(self._category_idx_store.getidx))

    def _add_to_x_factory(self, row):
        for feat, count in row[self._feature_col].items():
            feat_idx = self._term_idx_store.getidx(feat)
            self._X_factory[row.name, feat_idx] = count
        if self._metadata_col in self._df:
            for meta, count in row[self._metadata_col].items():
                meta_idx = self._metadata_idx_store.getidx(meta)
                self._mX_factory[row.name, meta_idx] = count

    def _make_new_term_doc_matrix(self, new_X, new_mX, new_y, new_term_idx_store, new_category_idx_store, new_metadata_idx_store, new_y_mask):
        if self._parsed_col is not None and self._parsed_col in self._df:
            return ParsedCorpus(self._df[new_y_mask], new_X, new_mX, new_y, new_term_idx_store, new_category_idx_store, new_metadata_idx_store, self._parsed_col, self._category_col)
        else:
            return CorpusDF(self._df[new_y_mask], new_X, new_mX, new_y, self._text_col, new_term_idx_store, new_category_idx_store, new_metadata_idx_store, self._df[self._text_col][new_y_mask])

class TermDocMatrixFromPandas(TermDocMatrixFactory):

    def __init__(self, data_frame, category_col, text_col, clean_function=lambda x: x, nlp=None, feats_from_spacy_doc=None, verbose=False):
        """Creates a TermDocMatrix from a pandas data frame.

        Parameters
        ----------
        data_frame : pd.DataFrame
            The data frame that contains columns for the category of interest
            and the document text.
        text_col : str
            The name of the column which contains each document's raw text.
        category_col : str
            The name of the column which contains the category of interest.
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
        self._category_col = category_col
        self._verbose = verbose

    def build(self):
        """Constructs the term doc matrix.

        Returns
        -------
        TermDocMatrix
        """
        X_factory, mX_factory, category_idx_store, term_idx_store, metadata_idx_store, y = self._init_term_doc_matrix_variables()
        parse_pipeline = ParsePipelineFactory(self.get_nlp(), X_factory, mX_factory, category_idx_store, term_idx_store, metadata_idx_store, y, self)
        df = self._clean_and_filter_nulls_and_empties_from_dataframe()
        tdm = self._apply_pipeline_and_get_build_instance(X_factory, mX_factory, category_idx_store, df, parse_pipeline, term_idx_store, metadata_idx_store, y)
        return tdm

    def _apply_pipeline_and_get_build_instance(self, X_factory, mX_factory, category_idx_store, df, parse_pipeline, term_idx_store, metadata_idx_store, y):
        df.apply(parse_pipeline.parse, axis=1)
        y = np.array(y)
        X, mX = self._build_sparse_matrices(y, X_factory, mX_factory)
        tdm = TermDocMatrix(X, mX, y, term_idx_store, category_idx_store, metadata_idx_store)
        return tdm

    def _build_sparse_matrices(self, y, X_factory, mX_factory):
        return build_sparse_matrices(y, X_factory, mX_factory)

    def _init_term_doc_matrix_variables(self):
        return CorpusFactoryHelper.init_term_doc_matrix_variables()

    def _clean_and_filter_nulls_and_empties_from_dataframe(self):
        return self.data_frame.loc[lambda df: df[[self._category_col, self._text_col]].dropna().index][lambda df: df[self._text_col] != ''].reset_index()

class CorpusFromTermFrequencies(object):

    def __init__(self, X: csr_matrix, term_vocabulary: List[str], mX: Optional[csr_matrix]=None, y: Optional[np.array]=None, category_names: Optional[str]=None, metadata_vocabulary: Optional[List[str]]=None, text_df: Optional[pd.DataFrame]=None, text_col: Optional[str]=None, parsed_col: Optional[str]=None, category_col: Optional[str]=None, unigram_frequency_path: Optional[str]=None):
        """
        Parameters
        ----------
        X: csr_matrix; term-document frequency matrix; columns represent terms and rows documents
        term_vocabulary: List[str]; Each entry corresponds to a term
        mX: Optional[csr_matrix]; metadata csr matrix
        y: Optional[np.array[int]]; indices of category names for each document
        category_names: Optional[List[str]], names of categories for y
        text_df: pd.DataFrame with a row containing the raw document text
        text_col: str; name of row containing the text of each document
        parsed_col: str; name of row containing the parsed text of each document
        unigram_frequency_path: str (see TermDocMatrix)

        """
        self.X = X
        self.term_idx_store = IndexStoreFromList.build(term_vocabulary)
        assert self.X.shape[1] == len(term_vocabulary)
        self.metadata_idx_store = IndexStore()
        if y is None:
            self.y = np.zeros(self.X.shape[0], dtype=int)
            self.category_idx_store = IndexStoreFromList.build(['_'])
            assert category_names is None
        else:
            self.y = y
            assert len(category_names) == len(set(y))
            self.category_idx_store = IndexStoreFromList.build(category_names)
        if metadata_vocabulary is not None:
            assert mX.shape[1] == metadata_vocabulary
            self.mX = mX
            self.metadata_idx_store = IndexStoreFromList.build(metadata_vocabulary)
        else:
            assert metadata_vocabulary is None
            self.mX = csr_matrix((0, 0))
            self.metadata_idx_store = IndexStore()
        self.text_df = text_df
        if parsed_col is not None:
            assert parsed_col in text_df
        if text_col is not None:
            assert text_col in text_df
        if category_col is not None:
            assert category_col in text_df
        self.category_col = category_col
        self.text_col = text_col
        self.parsed_col = parsed_col
        self.unigram_frequency_path = unigram_frequency_path

    def build(self):
        """
        Returns
        -------
        CorpusDF
        """
        if self.text_df is not None:
            if self.parsed_col is not None:
                if self.category_col is None:
                    self.text_df = self.text_df.assign(Category=self.category_idx_store.getvalbatch(self.y))
                    self.category_col = 'Category'
                return ParsedCorpus(df=self.text_df, X=self.X, mX=self.mX, y=self.y, parsed_col=self.parsed_col, term_idx_store=self.term_idx_store, category_idx_store=self.category_idx_store, metadata_idx_store=self.metadata_idx_store, unigram_frequency_path=self.unigram_frequency_path, category_col=self.category_col)
            elif self.text_col is not None:
                return CorpusDF(df=self.text_df, X=self.X, mX=self.mX, y=self.y, text_col=self.text_col, term_idx_store=self.term_idx_store, category_idx_store=self.category_idx_store, metadata_idx_store=self.metadata_idx_store, unigram_frequency_path=self.unigram_frequency_path)
        return TermDocMatrix(X=self.X, mX=self.mX, y=self.y, term_idx_store=self.term_idx_store, category_idx_store=self.category_idx_store, metadata_idx_store=self.metadata_idx_store, unigram_frequency_path=self.unigram_frequency_path)

class CorpusFromParsedDocuments(object):

    def __init__(self, df, category_col, parsed_col, feats_from_spacy_doc=FeatsFromSpacyDoc()):
        """
        Parameters
        ----------
        df : pd.DataFrame
         contains category_col, and parse_col, were parsed col is entirely spacy docs
        category_col : str
            name of category column in convention_df
        parsed_col : str
            name of spacy parsed column in convention_df
        feats_from_spacy_doc : FeatsFromSpacyDoc
        """
        self._df = df.reset_index()
        self._category_col = category_col
        self._parsed_col = parsed_col
        self._category_idx_store = IndexStore()
        self._X_factory = CSRMatrixFactory()
        self._mX_factory = CSRMatrixFactory()
        self._term_idx_store = IndexStore()
        self._metadata_idx_store = IndexStore()
        self._feats_from_spacy_doc = feats_from_spacy_doc

    def build(self, show_progress=False) -> ParsedCorpus:
        """Constructs the term doc matrix.

        Returns
        -------
        scattertext.ParsedCorpus.ParsedCorpus
        """
        y = self._get_y_and_populate_category_idx_store(self._df[self._category_col].apply(str))
        if show_progress is True:
            self._df.progress_apply(self._add_to_x_factory, axis=1)
        else:
            self._df.apply(self._add_to_x_factory, axis=1)
        self._mX = self._mX_factory.set_last_row_idx(len(y) - 1).get_csr_matrix()
        return ParsedCorpus(df=self._df, X=self._X_factory.set_last_row_idx(len(y) - 1).get_csr_matrix(), mX=self._mX_factory.set_last_row_idx(len(y) - 1).get_csr_matrix(), y=y, term_idx_store=self._term_idx_store, category_idx_store=self._category_idx_store, metadata_idx_store=self._metadata_idx_store, parsed_col=self._parsed_col, category_col=self._category_col)

    def _get_y_and_populate_category_idx_store(self, categories):
        return np.array(categories.apply(self._category_idx_store.getidx))

    def _add_to_x_factory(self, row):
        parsed_text = row[self._parsed_col]
        for term, count in self._feats_from_spacy_doc.get_feats(parsed_text).items():
            term_idx = self._term_idx_store.getidx(term)
            self._X_factory[row.name, term_idx] = count
        for meta, val in self._feats_from_spacy_doc.get_doc_metadata(parsed_text).items():
            meta_idx = self._metadata_idx_store.getidx(meta)
            self._mX_factory[row.name, meta_idx] = val

