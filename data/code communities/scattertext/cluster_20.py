# Cluster 20

class TermDocMatrixWithoutCategories(object):

    def __init__(self, X: csr_matrix, mX: csr_matrix, term_idx_store: IndexStore, metadata_idx_store: IndexStore, unigram_frequency_path: str=None):
        """

        Parameters
        ----------
        X : csr_matrix
            term document matrix
        mX : csr_matrix
            metadata-document matrix
        term_idx_store : IndexStore
            Term indices
        metadata_idx_store : IndexStore
          Document metadata indices
        unigram_frequency_path : str or None
            Path to term frequency file.
        """
        self._X = X
        self._mX = mX
        self._term_idx_store = term_idx_store
        self._metadata_idx_store = metadata_idx_store
        self._unigram_frequency_path = unigram_frequency_path
        self._background_corpus = None
        self._strict_unigram_definition = True

    def get_default_stoplist(self):
        return MY_ENGLISH_STOP_WORDS

    def allow_single_quotes_in_unigrams(self) -> Self:
        """
        Don't filter out single quotes in unigrams
        :return: self
        """
        self._strict_unigram_definition = False
        return self

    def compact(self, compactor, non_text: bool=False) -> Self:
        """
        Compact term document matrix.

        Parameters
        ----------
        compactor : object
            Object that takes a Term Doc Matrix as its first argument, and has a compact function which returns a
            Term Doc Matrix like argument
        non_text : bool
            Use non text features. False by default.
        Returns
        -------
        TermDocMatrix
        """
        return compactor.compact(self, non_text)

    def select(self, compactor, non_text: bool=False) -> Self:
        """
        Same as compact
        """
        return compactor.compact(self, non_text)

    def get_num_terms(self, non_text: bool=False) -> int:
        """
        Returns
        -------
        The number of terms registered in the term doc matrix
        """
        if non_text:
            return self.get_num_metadata()
        return len(self._term_idx_store)

    def get_num_docs(self) -> int:
        """
        Returns
        -------
        int, number of documents
        """
        return self._X.shape[0]

    def get_num_metadata(self) -> int:
        """
        Returns
        -------
        int, number of unique metadata items
        """
        return len(self.get_metadata())

    def set_background_corpus(self, background) -> Self:
        """
        Parameters
        ----------
        background

        """
        if issubclass(type(background), TermDocMatrixWithoutCategories):
            self._background_corpus = pd.DataFrame(background.get_term_freq_df().sum(axis=1), columns=['background']).reset_index()
            self._background_corpus.columns = ['word', 'background']
        elif type(background) == pd.DataFrame and set(background.columns) == set(['word', 'background']):
            self._background_corpus = background
        elif type(background) == pd.Series:
            self._background_corpus = background
        else:
            raise Exception('The argument named background must be a subclass of TermDocMatrix or a ' + 'DataFrame with columns "word" and "background", where "word" ' + 'is the term text, and "background" is its frequency.')
        return self

    def get_background_corpus(self):
        if self._background_corpus is not None:
            if type(self._background_corpus) == pd.DataFrame:
                return self._background_corpus['background']
            elif type(self._background_corpus) == pd.Series:
                return self._background_corpus
            return self._background_corpus
        return DefaultBackgroundFrequencies.get_background_frequency_df(self._unigram_frequency_path)

    def get_term_and_background_counts(self) -> pd.DataFrame:
        """
        Returns
        -------
        A pd.DataFrame consisting of unigram term counts of words occurring
         in the TermDocumentMatrix and their corresponding background corpus
         counts.  The dataframe has two columns, corpus and background.

        >>> corpus.get_unigram_corpus().get_term_and_background_counts()
                          corpus  background
        obama              702.0    565739.0
        romney             570.0    695398.0
        barack             248.0    227861.0
        ...
        """
        background_df = self._get_background_unigram_frequencies()
        if type(background_df) == pd.DataFrame:
            background_df = background_df['background']
        corpus_freq_df = self.get_term_count_df()
        corpus_unigram_freq = self._get_corpus_unigram_freq(corpus_freq_df)
        return pd.DataFrame({'background': background_df, 'corpus': corpus_unigram_freq['corpus']}).fillna(0)

    def get_term_count_df(self):
        return pd.DataFrame({'corpus': self._X.sum(axis=0).A1, 'term': self.get_terms()}).set_index('term')

    def _get_corpus_unigram_freq(self, corpus_freq_df: pd.DataFrame) -> pd.DataFrame:
        unigram_validator = re.compile('^[A-Za-z]+$')
        corpus_unigram_freq = corpus_freq_df.loc[[term for term in corpus_freq_df.index if unigram_validator.match(term) is not None]]
        return corpus_unigram_freq

    def _get_background_unigram_frequencies(self) -> pd.DataFrame:
        if self.get_background_corpus() is not None:
            return self.get_background_corpus()
        return DefaultBackgroundFrequencies.get_background_frequency_df(self._unigram_frequency_path)

    def list_extra_features(self, use_metadata: bool=True) -> List[Dict[str, str]]:
        """
        Returns
        -------
        List of dicts.  One dict for each document, keys are metadata, values are counts
        """
        return FeatureLister(self._get_relevant_X(use_metadata), self._get_relevant_idx_store(use_metadata), self.get_num_docs()).output()

    def get_terms(self, use_metadata=False) -> List[str]:
        """
        Returns
        -------
        np.array of unique terms
        """
        if use_metadata:
            return self.get_metadata()
        return self._term_idx_store._i2val

    def get_metadata(self) -> List[str]:
        """
        Returns
        -------
        np.array of unique metadata
        """
        return self._metadata_idx_store._i2val

    def get_total_unigram_count(self) -> int:
        return self._get_unigram_term_freq_df().sum()

    def _get_unigram_term_freq_df(self) -> pd.DataFrame:
        return self._get_corpus_unigram_freq(self.get_term_count_df()['corpus'])

    def _get_X_after_delete_terms(self, idx_to_delete_list: List[int], non_text: bool=False) -> Tuple[csr_matrix, IndexStore]:
        new_term_idx_store = self._get_relevant_idx_store(non_text).batch_delete_idx(idx_to_delete_list)
        new_X = delete_columns(self._get_relevant_X(non_text), idx_to_delete_list)
        return (new_X, new_term_idx_store)

    def _get_relevant_X(self, non_text: bool) -> csr_matrix:
        return self._mX if non_text else self._X

    def _get_relevant_idx_store(self, non_text: bool) -> IndexStore:
        return self._metadata_idx_store if non_text else self._term_idx_store

    def remove_infrequent_words(self, minimum_term_count: int, term_ranker: Type[TermRanker]=AbsoluteFrequencyRanker, non_text: bool=False) -> Self:
        """
        Returns
        -------
        A new TermDocumentMatrix consisting of only terms which occur at least minimum_term_count.
        """
        ranker = term_ranker(self)
        if non_text:
            ranker = ranker.use_non_text_features()
        tdf = ranker.get_ranks().sum(axis=1)
        return self.remove_terms(list(tdf[tdf <= minimum_term_count].index), non_text=non_text)

    def remove_infrequent_terms(self, minimum_term_count: int, term_ranker: Type[TermRanker]=AbsoluteFrequencyRanker, non_text: bool=False) -> Self:
        return self.remove_infrequent_words(minimum_term_count=minimum_term_count, term_ranker=term_ranker, non_text=non_text)

    def remove_word_by_document_pct(self, min_document_pct: float=0.0, max_document_pct: float=1.0, non_text: bool=False) -> Self:
        """
        Returns a copy of the corpus with terms that occur in a document percentage range.

        :param min_document_pct: float, minimum document percentage. 0 by default
        :param max_document_pct: float, maximum document percentage. 1 by default
        :param non_text: bool, use metadata?
        :return: Corpus
        """
        tdm = self.get_term_doc_mat(non_text=non_text) > 0
        tdmpct = (tdm.sum(axis=0) / tdm.shape[0]).A1
        mask = (tdmpct >= min_document_pct) & (tdmpct <= max_document_pct)
        return self.whitelist_terms(np.array(self.get_terms(use_metadata=non_text))[mask], non_text=non_text)

    def remove_entity_tags(self, non_text: bool=False) -> Self:
        """
        Returns
        -------
        A new TermDocumentMatrix consisting of only terms in the current TermDocumentMatrix
         that aren't spaCy entity tags.

        Note: Used if entity types are censored using FeatsFromSpacyDoc(tag_types_to_censor=...).
        :param non_text: bool, use metadata?
        """
        terms_to_remove = [term for term in self.get_terms(use_metadata=non_text) if any([word in SPACY_ENTITY_TAGS for word in term.split()])]
        return self.remove_terms(terms_to_remove, non_text=non_text)

    def remove_terms(self, terms: List[str], ignore_absences: bool=False, non_text: bool=False) -> Self:
        """Non-destructive term removal.

        Parameters
        ----------
        terms : list
            list of terms to remove
        ignore_absences : bool, False by default
            If term does not appear, don't raise an error, just move on.
        non_text : bool, False by default
            Remove metadata terms instead of regular terms

        Returns
        -------
        TermDocMatrix, new object with terms removed.
        """
        idx_to_delete_list = self._build_term_index_list(ignore_absences, terms, non_text)
        return self.remove_terms_by_indices(idx_to_delete_list, non_text)

    def whitelist_terms(self, whitelist_terms: List[str], non_text: bool=False) -> Self:
        """

        :param whitelist_terms: list[str], terms to whitelist
        :param non_text: bool, use non text featurs, default False
        :return: TermDocMatrix, new object with only terms in parameter
        """
        return self.remove_terms(list(set(self.get_terms(use_metadata=non_text)) - set(whitelist_terms)), non_text=non_text)

    def _build_term_index_list(self, ignore_absences: bool, terms: List[str], non_text=False) -> List[int]:
        idx_to_delete_list = []
        my_term_idx_store = self._get_relevant_idx_store(non_text)
        for term in terms:
            if term not in my_term_idx_store:
                if not ignore_absences:
                    raise KeyError('Term %s not found' % term)
                continue
            idx_to_delete_list.append(my_term_idx_store.getidx(term))
        return idx_to_delete_list

    def _make_new_term_doc_matrix(self, new_X=None, new_mX=None, new_y=None, new_term_idx_store=None, new_category_idx_store=None, new_metadata_idx_store=None, new_y_mask=None) -> Self:
        return TermDocMatrixWithoutCategories(X=new_X if new_X is not None else self._X, mX=new_mX if new_mX is not None else self._mX, term_idx_store=new_term_idx_store if new_term_idx_store is not None else self._term_idx_store, metadata_idx_store=new_metadata_idx_store if new_metadata_idx_store is not None else self._metadata_idx_store, unigram_frequency_path=self._unigram_frequency_path)

    def remove_terms_used_in_less_than_num_docs(self, threshold: int, non_text: bool=False) -> Self:
        """
        Parameters
        ----------
        threshold: int
            Minimum number of documents term should appear in to be kept
        non_text: bool
            Use non-text features instead of terms

        Returns
        -------
        TermDocMatrix, new object with terms removed.
        """
        term_counts = self._get_relevant_X(non_text).astype(bool).astype(int).sum(axis=0).A[0]
        terms_to_remove = np.where(term_counts < threshold)[0]
        return self.remove_terms_by_indices(terms_to_remove, non_text)

    def remove_document_ids(self, document_ids: List[int], remove_unused_terms: bool=True, remove_unused_metadata: bool=False) -> Self:
        """

        :param document_ids: List[int], list of document ids to remove
        :return: Corpus
        """
        y_mask = ~np.isin(np.arange(self.get_num_docs()), np.array(document_ids))
        updated_tdm = self._make_new_term_doc_matrix(new_X=self._X, new_mX=self._mX, new_y=None, new_category_idx_store=None, new_term_idx_store=self._term_idx_store, new_metadata_idx_store=self._metadata_idx_store, new_y_mask=y_mask)
        if remove_unused_terms:
            unused_term_idx = np.where(self._X[y_mask, :].sum(axis=0) == 0)[1]
            updated_tdm = updated_tdm.remove_terms_by_indices(unused_term_idx, non_text=False)
        if remove_unused_metadata:
            unused_metadata_mask = np.mask(self._mX[y_mask, :].sum(axis=0) == 0)[0]
            updated_tdm = updated_tdm.remove_terms_by_indices(unused_metadata_mask, non_text=True)
        return updated_tdm

    def remove_documents_less_than_length(self, max_length: int, non_text: bool=False) -> Self:
        """
            `

        :param max_length: int, length of document in terms registered in corpus
        :return: Corpus
        """
        tdm = self.get_metadata_doc_mat() if non_text else self.get_term_doc_mat()
        doc_ids_to_remove = np.where(tdm.sum(axis=1).T.A1 < max_length)
        return self.remove_document_ids(doc_ids_to_remove)

    def get_unigram_corpus(self) -> Self:
        """
        Returns
        -------
        A new TermDocumentMatrix consisting of only unigrams in the current TermDocumentMatrix.
        """
        terms_to_ignore = self._get_non_unigrams()
        return self.remove_terms(terms_to_ignore)

    def _get_non_unigrams(self) -> List[str]:
        return [term for term in self._term_idx_store._i2val if ' ' in term or (self._strict_unigram_definition and "'" in term)]

    def get_stoplisted_unigram_corpus(self, stoplist: Optional[List[str]]=None, non_text: bool=False) -> Self:
        """
        Parameters
        -------
        stoplist : list, optional

        Returns
        -------
        A new TermDocumentMatrix consisting of only unigrams in the current TermDocumentMatrix.
        """
        if stoplist is None:
            stoplist = self.get_default_stoplist()
        else:
            stoplist = [w.lower() for w in stoplist]
        return self._remove_terms_from_list_and_all_non_unigrams(stoplist, non_text=non_text)

    def get_stoplisted_corpus(self, stoplist=None, non_text: bool=False):
        """
        Parameters
        -------
        stoplist : list, optional
        non_text : bool

        Returns
        -------
        A new TermDocumentMatrix consisting of only unigrams in the current TermDocumentMatrix.
        """
        if stoplist is None:
            stoplist = self.get_default_stoplist()
        return self.remove_terms([w.lower() for w in stoplist], ignore_absences=True, non_text=non_text)

    def get_stoplisted_unigram_corpus_and_custom(self, custom_stoplist, non_text: bool=False):
        """
        Parameters
        -------
        stoplist : list of lower-cased words, optional

        Returns
        -------
        A new TermDocumentMatrix consisting of only unigrams in the current TermDocumentMatrix.
        """
        if type(custom_stoplist) == str:
            custom_stoplist = [custom_stoplist]
        return self._remove_terms_from_list_and_all_non_unigrams(set(self.get_default_stoplist()) | set((w.lower() for w in custom_stoplist)), non_text=non_text)

    def filter_out(self, filter_func, non_text=False):
        """

        :param filter_func: function which takes a string and returns true or false
        :return: A new TermDocumentMatrix consisting of only unigrams in the current TermDocumentMatrix.
        """
        return self.remove_terms([x for x in self.get_terms(use_metadata=non_text) if filter_func(x)], non_text=non_text)

    def _remove_terms_from_list_and_all_non_unigrams(self, stoplist, non_text=False):
        terms_to_ignore = [term for term in (self._metadata_idx_store._i2val if non_text else self._term_idx_store._i2val) if ' ' in term or (self._strict_unigram_definition and ("'" in term or '’' in term)) or term in stoplist]
        return self.remove_terms(terms_to_ignore, non_text=non_text)

    def metadata_in_use(self) -> bool:
        """
        Returns True if metadata values are in term doc matrix.

        Returns
        -------
        bool
        """
        return len(self._metadata_idx_store) > 0

    def _make_all_positive_data_ones(self, new_x: csr_matrix) -> csr_matrix:
        return (new_x > 0).astype(np.int32)

    def remove_terms_by_indices(self, idx_to_delete_list: List[int], non_text: bool=False) -> Self:
        """
        Parameters
        ----------
        idx_to_delete_list, list
        non_text, bool
            Should we remove non text features or just terms?

        Returns
        -------
        TermDocMatrix
        """
        new_X, new_idx_store = self._get_X_after_delete_terms(idx_to_delete_list, non_text)
        return self._make_new_term_doc_matrix(new_X=self._X if non_text else new_X, new_mX=new_X if non_text else self._mX, new_y=None, new_category_idx_store=None, new_term_idx_store=self._term_idx_store if non_text else new_idx_store, new_metadata_idx_store=new_idx_store if non_text else self._metadata_idx_store, new_y_mask=np.ones(new_X.shape[0]).astype(np.bool))

    def get_scaled_f_scores_vs_background(self, scaler_algo: str=DEFAULT_BACKGROUND_SCALER_ALGO, beta: float=DEFAULT_BACKGROUND_BETA) -> pd.DataFrame:
        """
        Parameters
        ----------
        scaler_algo : str
            see get_scaled_f_scores, default 'none'
        beta : float
          default 1.
        Returns
        -------
        pd.DataFrame of scaled_f_score scores compared to background corpus
        """
        df = self.get_term_and_background_counts()
        df['Scaled f-score'] = ScaledFScore.get_scores_for_category(df['corpus'], df['background'], scaler_algo, beta)
        return df.sort_values(by='Scaled f-score', ascending=False)

    def get_term_doc_mat(self, non_text: bool=False) -> csr_matrix:
        """
        Returns sparse matrix representation of term-doc-matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if non_text:
            return self.get_metadata_doc_mat()
        return self._X

    def get_term_freqs(self, non_text: bool=False) -> np.array:
        return self.get_term_doc_mat(non_text=non_text).sum(axis=0).A1

    def get_term_doc_mat_coo(self, non_text: bool=False) -> coo_matrix:
        """
        Returns sparse matrix representation of term-doc-matrix

        Returns
        -------
        scipy.sparse.coo_matrix
        """
        return self.get_term_doc_mat(non_text=non_text).astype(np.double).tocoo()

    def get_metadata_doc_mat(self) -> csr_matrix:
        """
        Returns sparse matrix representation of term-doc-matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return self._mX

    def term_doc_lists(self) -> Dict:
        """
        Returns
        -------
        dict
        """
        doc_ids = self._X.transpose().tolil().rows
        terms = self._term_idx_store.values()
        return dict(zip(terms, doc_ids))

    def apply_ranker(self, term_ranker: Type[TermRanker], use_non_text_features: bool, label_append: str=' freq') -> pd.DataFrame:
        """
        Parameters
        ----------
        term_ranker : TermRanker

        Returns
        -------
        pd.Dataframe
        """
        if use_non_text_features:
            return term_ranker(self).use_non_text_features().get_ranks(label_append=label_append)
        return term_ranker(self).get_ranks(label_append=label_append)

    def add_doc_names_as_metadata(self, doc_names: List[str]) -> Self:
        """
        :param doc_names: array-like[str], document names of reach document
        :return: Corpus-like object with doc names as metadata. If two documents share the same name
        (doc number) will be appended to their names.
        """
        if len(doc_names) != self.get_num_docs():
            raise Exception('The parameter doc_names contains %s elements. It should have %s elements, one per document.' % (len(doc_names), self.get_num_docs()))
        doc_names_counter = collections.Counter(np.array(doc_names))
        metafact = CSRMatrixFactory()
        metaidxstore = IndexStore()
        doc_id_uses = collections.Counter()
        for i in range(self.get_num_docs()):
            doc_id = doc_names[i]
            if doc_names_counter[doc_id] > 1:
                doc_id_uses[doc_id] += 1
                doc_name_idx = metaidxstore.getidx('%s (%s)' % (doc_id, doc_id_uses[doc_id]))
            else:
                doc_name_idx = metaidxstore.getidx(doc_id)
            metafact[i, i] = doc_name_idx
        return self.add_metadata(metafact.get_csr_matrix(), metaidxstore)

    def get_term_index(self, term: str) -> int:
        return self._term_idx_store.getidxstrict(term)

    def get_metadata_index(self, term: str) -> int:
        return self._metadata_idx_store.getidxstrict(term)

    def get_metadata_from_index(self, index: int) -> str:
        return self._metadata_idx_store.getval(index)

    def get_term_from_index(self, index: int) -> str:
        return self._term_idx_store.getval(index)

    def get_term_index_store(self, non_text=False) -> IndexStore:
        return self._metadata_idx_store if non_text else self._term_idx_store

    def get_document_ids_with_terms(self, terms: List[str], use_non_text_features: bool=False) -> np.array:
        return np.where(self._get_relevant_X(use_non_text_features)[:, [self._term_idx_store.getidx(x) for x in terms if x in self._term_idx_store]].sum(axis=1).A1 > 0)[0]

    def add_metadata(self, metadata_matrix: csr_matrix, meta_index_store: IndexStore) -> Self:
        """
        Returns a new corpus with the metadata matrix and index store integrated.

        :param metadata_matrix: scipy.sparse matrix (# docs, # metadata)
        :param meta_index_store: IndexStore of metadata values
        :return: TermDocMatrixWithoutCategories
        """
        assert isinstance(meta_index_store, IndexStore)
        assert len(metadata_matrix.shape) == 2
        assert metadata_matrix.shape[0] == self.get_num_docs()
        return self._make_new_term_doc_matrix(new_X=self._X, new_y=None, new_category_idx_store=None, new_y_mask=np.ones(self.get_num_docs()).astype(bool), new_mX=metadata_matrix, new_term_idx_store=self._term_idx_store, new_metadata_idx_store=meta_index_store)

    def rename_metadata(self, old_to_new_vals: List[Tuple[str, str]], policy: MetadataReplacementRetentionPolicy=MetadataReplacementRetentionPolicy.KEEP_ONLY_NEW) -> Self:
        new_mX, new_metadata_idx_store = self._remap_metadata(old_to_new_vals, policy)
        return self._make_new_term_doc_matrix(new_X=self._X, new_mX=new_mX, new_term_idx_store=self._term_idx_store, new_metadata_idx_store=new_metadata_idx_store)

    def _remap_metadata(self, old_to_new_vals: List[Tuple[str, str]], policy: MetadataReplacementRetentionPolicy) -> Tuple[csr_matrix, IndexStore]:
        old_to_new_df = self._get_old_to_new_metadata_mapping_df(old_to_new_vals)
        keep_vals = self._get_metadata_mapped_values_to_keep(old_to_new_df)
        new_val_mX = np.zeros(shape=(self._mX.shape[0], old_to_new_df.New.nunique()))
        if policy.value == MetadataReplacementRetentionPolicy.KEEP_UNMODIFIED.value:
            new_metadata_idx_store = IndexStoreFromList.build(keep_vals)
        elif policy.value == MetadataReplacementRetentionPolicy.KEEP_ONLY_NEW.value:
            new_metadata_idx_store = IndexStore()
        else:
            raise Exception(f'Policy {policy} not supporteds')
        for new_val_i, (new_name, new_df) in enumerate(old_to_new_df.groupby('New')):
            new_metadata_idx_store.getidx(new_name)
            new_val_counts = self._mX[:, self._metadata_idx_store.getidxstrictbatch(new_df.Old.values)]
            new_val_mX[:, new_val_i] = new_val_counts.sum(axis=1).T[0]
        if policy.value == MetadataReplacementRetentionPolicy.KEEP_UNMODIFIED.value:
            keep_mX = self._mX[:, self._metadata_idx_store.getidxstrictbatch(keep_vals)]
            new_mX = scipy.sparse.hstack([keep_mX, new_val_mX], format='csr', dtype=self._mX.dtype)
        else:
            new_mX = scipy.sparse.csr_matrix(new_val_mX, dtype=self._mX.dtype)
        return (new_mX, new_metadata_idx_store)

    def _get_metadata_mapped_values_to_keep(self, old_to_new_df: pd.DataFrame) -> List[str]:
        keep_vals = [x for x in self.get_metadata() if x not in set(old_to_new_df.Old.unique()) | set(old_to_new_df.New.unique())]
        return keep_vals

    def _get_old_to_new_metadata_mapping_df(self, old_to_new_vals: List[Tuple[str, str]]) -> pd.DataFrame:
        old_to_new_vals = [(old, new) for old, new in old_to_new_vals if old in self._metadata_idx_store]
        return pd.DataFrame(old_to_new_vals, columns=['Old', 'New'])

class TestTermDocMat(TestCase):

    @classmethod
    def setUp(cls):
        cls.tdm = make_a_test_term_doc_matrix()

    def test_get_num_terms(self):
        self.assertEqual(self.tdm.get_num_terms(), self.tdm._X.shape[1])

    def test_get_term_freq_df(self):
        df = self.tdm.get_term_freq_df().sort_values('b freq', ascending=False)[:3]
        self.assertEqual(list(df.index), ['another', 'blah', 'blah blah'])
        self.assertEqual(list(df['a freq']), [0, 0, 0])
        self.assertEqual(list(df['b freq']), [4, 3, 2])
        self.assertEqual(list(self.tdm.get_term_freq_df().sort_values('a freq', ascending=False)[:3]['a freq']), [2, 2, 1])

    def test_single_category_term_doc_matrix_should_error(self):
        """
        with self.assertRaisesRegex(
                expected_exception=CannotCreateATermDocMatrixWithASignleCategoryException,
                expected_regex='Documents must be labeled with more than one category. '
                               'All documents were labeled with category: "a"'):
            single_category_tdm = build_from_category_whitespace_delimited_text(
                [['a', text]
                 for category, text
                 in get_test_categories_and_documents()]
            )
        """
        single_category_tdm = build_from_category_whitespace_delimited_text([['a', text] for category, text in get_test_categories_and_documents()])

    def test_total_unigram_count(self):
        self.assertEqual(self.tdm.get_total_unigram_count(), 36)

    def test_get_term_df(self):
        categories, documents = get_docs_categories()
        df = pd.DataFrame({'category': categories, 'text': documents})
        tdm_factory = TermDocMatrixFromPandas(df, 'category', 'text', nlp=whitespace_nlp)
        term_doc_matrix = tdm_factory.build()
        term_df = term_doc_matrix.get_term_freq_df()
        self.assertEqual(dict(term_df.loc['speak up']), {'??? freq': 2, 'hamlet freq': 0, 'jay-z/r. kelly freq': 1})
        self.assertEqual(dict(term_df.loc['that']), {'??? freq': 0, 'hamlet freq': 2, 'jay-z/r. kelly freq': 0})

    def test_get_terms(self):
        tdm = make_a_test_term_doc_matrix()
        self.assertEqual(tdm.get_terms(), ['hello', 'my', 'name', 'is', 'joe.', 'hello my', 'my name', 'name is', 'is joe.', "i've", 'got', 'a', 'wife', 'and', 'three', 'kids', "i'm", 'working.', "i've got", 'got a', 'a wife', 'wife and', 'and three', 'three kids', 'kids and', "and i'm", "i'm working.", 'in', 'button', 'factory', 'in a', 'a button', 'button factory', 'this', 'another', 'type', 'of', 'document', 'this is', 'is another', 'another type', 'type of', 'of document', 'sentence', 'another sentence', 'sentence in', 'in another', 'another document', "isn't", 'joe', 'here', "name isn't", "isn't joe", 'joe here', 'document.', 'another document.', 'blah', 'blah blah'])

    def test_get_unigram_corpus(self):
        tdm = make_a_test_term_doc_matrix()
        uni_tdm = tdm.get_unigram_corpus()
        term_df = tdm.get_term_freq_df()
        uni_term_df = uni_tdm.get_term_freq_df()
        self.assertEqual(set((term for term in term_df.index if ' ' not in term and "'" not in term)), set(uni_term_df.index))

    def test_remove_entity_tags(self):
        tdm = make_a_test_term_doc_matrix()
        removed_tags_tdm = tdm.remove_entity_tags()
        term_df = tdm.get_term_freq_df()
        removed_tags_term_df = removed_tags_tdm.get_term_freq_df()
        expected_terms = set((term for term in term_df.index if not any((t in SPACY_ENTITY_TAGS for t in term.split()))))
        removed_terms = set(removed_tags_term_df.index)
        (self.assertEqual(expected_terms, removed_terms),)

    def test_get_stoplisted_unigram_corpus(self):
        tdm = make_a_test_term_doc_matrix()
        uni_tdm = tdm.get_stoplisted_unigram_corpus()
        term_df = tdm.get_term_freq_df()
        uni_term_df = uni_tdm.get_term_freq_df()
        (self.assertEqual(set((term for term in term_df.index if ' ' not in term and "'" not in term and (term not in MY_ENGLISH_STOP_WORDS))), set(uni_term_df.index)),)

    def test_allow_single_quotes_in_unigrams(self):
        tdm = make_a_test_term_doc_matrix()
        self.assertEqual(type(tdm.allow_single_quotes_in_unigrams()), type(tdm))
        uni_tdm = tdm.get_stoplisted_unigram_corpus()
        term_df = tdm.get_term_freq_df()
        uni_term_df = uni_tdm.get_term_freq_df()
        (self.assertEqual(set((term for term in term_df.index if ' ' not in term and term not in MY_ENGLISH_STOP_WORDS)), set(uni_term_df.index)),)

    def test_get_stoplisted_unigram_corpus_and_custom(self):
        tdm = make_a_test_term_doc_matrix()
        uni_tdm = tdm.get_stoplisted_unigram_corpus_and_custom(['joe'])
        self._assert_stoplisted_minus_joe(tdm, uni_tdm)
        uni_tdm = tdm.get_stoplisted_unigram_corpus_and_custom('joe')
        self._assert_stoplisted_minus_joe(tdm, uni_tdm)

    def _assert_stoplisted_minus_joe(self, tdm, uni_tdm):
        term_df = tdm.get_term_freq_df()
        uni_term_df = uni_tdm.get_term_freq_df()
        (self.assertEqual(set((term for term in term_df.index if ' ' not in term and 'joe' != term.lower() and ("'" not in term) and (term not in MY_ENGLISH_STOP_WORDS))), set(uni_term_df.index)),)

    def test_term_doc_lists(self):
        term_doc_lists = self.tdm.term_doc_lists()
        self.assertEqual(type(term_doc_lists), dict)
        self.assertEqual(term_doc_lists['this'], [1, 2])
        self.assertEqual(term_doc_lists['another document'], [1])
        self.assertEqual(term_doc_lists['is'], [0, 1, 2])

    def test_remove_terms(self):
        tdm = make_a_test_term_doc_matrix()
        with self.assertRaises(KeyError):
            tdm.remove_terms(['elephant'])
        tdm_removed = tdm.remove_terms(['hello', 'this', 'is'])
        removed_df = tdm_removed.get_term_freq_df()
        df = tdm.get_term_freq_df()
        self.assertEqual(tdm_removed.get_num_docs(), tdm.get_num_docs())
        self.assertEqual(len(removed_df), len(df) - 3)
        self.assertNotIn('hello', removed_df.index)
        self.assertIn('hello', df.index)

    def test_remove_terms_non_text(self):
        hamlet = get_hamlet_term_doc_matrix()
        doc_names = [str(i) for i in range(hamlet.get_num_docs())]
        hamlet_meta = hamlet.add_doc_names_as_metadata(doc_names)
        with self.assertRaises(KeyError):
            hamlet_meta.remove_terms(['xzcljzxsdjlksd'], non_text=True)
        tdm_removed = hamlet_meta.remove_terms(['2', '4', '6'], non_text=True)
        removed_df = tdm_removed.get_metadata_freq_df()
        df = hamlet_meta.get_metadata_freq_df()
        self.assertEqual(tdm_removed.get_num_docs(), hamlet_meta.get_num_docs())
        self.assertEqual(len(removed_df), len(df) - 3)
        self.assertNotIn('2', removed_df.index)
        self.assertIn('2', df.index)

    def test_whitelist_terms(self):
        tdm = make_a_test_term_doc_matrix()
        tdm_removed = tdm.whitelist_terms(['hello', 'this', 'is'])
        removed_df = tdm_removed.get_term_freq_df()
        df = tdm.get_term_freq_df()
        self.assertEqual(tdm_removed.get_num_docs(), tdm.get_num_docs())
        self.assertEqual(len(removed_df), 3)
        self.assertIn('hello', removed_df.index)
        self.assertIn('hello', df.index)
        self.assertIn('my', df.index)
        self.assertNotIn('my', removed_df.index)

    def test_remove_terms_used_less_than_num_docs(self):
        tdm = make_a_test_term_doc_matrix()
        tdm2 = tdm.remove_terms_used_in_less_than_num_docs(2)
        self.assertTrue(all(tdm2.get_term_freq_df().sum(axis=1) >= 2))

    def test_term_scores(self):
        df = self.tdm.get_term_freq_df()
        df['posterior ratio'] = self.tdm.get_posterior_mean_ratio_scores('b')
        scores = self.tdm.get_scaled_f_scores('b', scaler_algo='percentile')
        df['scaled_f_score'] = np.array(scores)
        with self.assertRaises(InvalidScalerException):
            self.tdm.get_scaled_f_scores('a', scaler_algo='x')
        self.tdm.get_scaled_f_scores('a', scaler_algo='percentile')
        self.tdm.get_scaled_f_scores('a', scaler_algo='normcdf')
        df['rudder'] = self.tdm.get_rudder_scores('b')
        df['corner'] = self.tdm.get_corner_scores('b')
        df['fisher oddsratio'], df['fisher pval'] = self.tdm.get_fisher_scores('b')
        self.assertEqual(list(df.sort_values(by='posterior ratio', ascending=False).index[:3]), ['another', 'blah', 'blah blah'])
        self.assertEqual(list(df.sort_values(by='scaled_f_score', ascending=False).index[:3]), ['another', 'blah', 'blah blah'])
        self.assertEqual(list(df.sort_values(by='rudder', ascending=True).index[:3]), ['another', 'blah', 'blah blah'])

    def test_term_scores_background(self):
        hamlet = get_hamlet_term_doc_matrix()
        df = hamlet.get_scaled_f_scores_vs_background(scaler_algo='none')
        self.assertEqual({u'corpus', u'background', u'Scaled f-score'}, set(df.columns))
        self.assertEqual(list(df.index[:3]), ['polonius', 'laertes', 'osric'])
        df = hamlet.get_posterior_mean_ratio_scores_vs_background()
        self.assertEqual({u'corpus', u'background', u'Log Posterior Mean Ratio'}, set(df.columns))
        self.assertEqual(list(df.index[:3]), ['hamlet', 'horatio', 'claudius'])

    def test_keep_only_these_categories(self):
        df = pd.DataFrame(data=np.array(get_docs_categories_semiotic()).T, columns=['category', 'text'])
        corpus = CorpusFromPandas(df, 'category', 'text', nlp=whitespace_nlp).build()
        hamlet_swift_corpus = corpus.keep_only_these_categories(['hamlet', 'swift'])
        self.assertEqual(hamlet_swift_corpus.get_categories(), ['hamlet', 'swift'])
        self.assertGreater(len(corpus.get_terms()), len(hamlet_swift_corpus.get_terms()))
        with self.assertRaises(AssertionError):
            corpus.keep_only_these_categories(['hamlet', 'swift', 'asdjklasfd'])
        corpus.keep_only_these_categories(['hamlet', 'swift', 'asdjklasfd'], True)

    def test_remove_categories(self):
        df = pd.DataFrame(data=np.array(get_docs_categories_semiotic()).T, columns=['category', 'text'])
        corpus = CorpusFromPandas(df, 'category', 'text', nlp=whitespace_nlp).build()
        swiftless = corpus.remove_categories(['swift'])
        swiftless_constructed = CorpusFromPandas(df[df['category'] != 'swift'], 'category', 'text', nlp=whitespace_nlp).build()
        np.testing.assert_equal([i for i in corpus._y if i != corpus.get_categories().index('swift')], swiftless._y)
        self.assertEqual(swiftless._y.shape[0], swiftless._X.shape[0])
        self.assertEqual(swiftless_constructed._X.shape, swiftless._X.shape)
        self.assertEqual(set(swiftless_constructed.get_terms()), set(swiftless.get_terms()))
        pd.testing.assert_series_equal(swiftless_constructed.get_texts(), swiftless.get_texts())
        np.testing.assert_equal(swiftless.get_category_names_by_row(), swiftless_constructed.get_category_names_by_row())

    def test_get_category_names_by_row2(self):
        hamlet = get_hamlet_term_doc_matrix()
        returned = hamlet.get_category_names_by_row()
        self.assertEqual(len(hamlet._y), len(returned))
        np.testing.assert_almost_equal([hamlet.get_categories().index(x) for x in returned], hamlet._y)

    def test_set_background_corpus(self):
        tdm = get_hamlet_term_doc_matrix()
        with self.assertRaisesRegex(Exception, 'The argument.+'):
            tdm.set_background_corpus(1)
        with self.assertRaisesRegex(Exception, 'The argument.+'):
            back_df = pd.DataFrame()
            tdm.set_background_corpus(back_df)
        with self.assertRaisesRegex(Exception, 'The argument.+'):
            back_df = pd.DataFrame({'word': ['a', 'bee'], 'backgasdround': [3, 1]})
            tdm.set_background_corpus(back_df)
        back_df = pd.DataFrame({'word': ['a', 'bee'], 'background': [3, 1]})
        tdm.set_background_corpus(back_df)
        tdm.set_background_corpus(tdm)

    def test_compact(self):
        x = get_hamlet_term_doc_matrix().compact(CompactTerms(minimum_term_count=3))
        self.assertEqual(type(x), TermDocMatrix)

    def _test_get_background_corpus(self):
        tdm = get_hamlet_term_doc_matrix()
        back_df = pd.DataFrame({'word': ['a', 'bee'], 'background': [3, 1]})
        tdm.set_background_corpus(back_df)
        print(tdm.get_background_corpus().to_dict())
        self.assertEqual(tdm.get_background_corpus().to_dict(), back_df.to_dict())
        tdm.set_background_corpus(tdm)
        self.assertEqual(set(tdm.get_background_corpus().to_dict().keys()), set(['word', 'background']))

    def test_log_reg(self):
        hamlet = get_hamlet_term_doc_matrix()
        df = hamlet.get_term_freq_df()
        df['logreg'], acc, baseline = hamlet.get_logistic_regression_coefs_l2('hamlet', clf=LinearRegression())
        l1scores, acc, baseline = hamlet.get_logistic_regression_coefs_l1('hamlet', clf=LinearRegression())
        self.assertGreaterEqual(acc, 0)
        self.assertGreaterEqual(baseline, 0)
        self.assertGreaterEqual(1, acc)
        self.assertGreaterEqual(1, baseline)
        self.assertEqual(list(df.sort_values(by='logreg', ascending=False).index[:3]), ['the', 'starts', 'incorporal'])

    def test_get_category_ids(self):
        hamlet = get_hamlet_term_doc_matrix()
        np.testing.assert_array_equal(hamlet.get_category_ids(), hamlet._y)

    def test_add_metadata(self):
        hamlet = get_hamlet_term_doc_matrix()
        meta_index_store = IndexStore()
        meta_fact = CSRMatrixFactory()
        for i in range(hamlet.get_num_docs()):
            meta_fact[i, i] = meta_index_store.getidx(str(i))
        other_hamlet = hamlet.add_metadata(meta_fact.get_csr_matrix(), meta_index_store)
        assert other_hamlet != hamlet
        meta_index_store = IndexStore()
        meta_fact = CSRMatrixFactory()
        for i in range(hamlet.get_num_docs() - 5):
            meta_fact[i, i] = meta_index_store.getidx(str(i))
        with self.assertRaises(AssertionError):
            hamlet.add_metadata(meta_fact.get_csr_matrix(), meta_index_store)

    def test_add_doc_names_as_metadata(self):
        hamlet = get_hamlet_term_doc_matrix()
        doc_names = [str(i) for i in range(hamlet.get_num_docs())]
        hamlet_meta = hamlet.add_doc_names_as_metadata(doc_names)
        self.assertNotEqual(hamlet, hamlet_meta)
        self.assertEqual(hamlet.get_metadata(), [])
        self.assertEqual(hamlet_meta.get_metadata(), doc_names)
        self.assertEqual(hamlet_meta.get_metadata_doc_mat().shape, (hamlet.get_num_docs(), hamlet.get_num_docs()))
        self.assertNotEqual(hamlet.get_metadata_doc_mat().shape, (hamlet.get_num_docs(), hamlet.get_num_docs()))

    def test_use_doc_labeled_terms_as_metadata(self):
        hamlet = get_hamlet_term_doc_matrix()
        doc_labels = [str(i % 3) for i in range(hamlet.get_num_docs())]
        new_hamlet = hamlet.use_doc_labeled_terms_as_metadata(doc_labels, '++')
        np.testing.assert_array_equal(hamlet.get_term_doc_mat().sum(axis=1), new_hamlet.get_metadata_doc_mat().sum(axis=1))
        metadata_freq_df = new_hamlet.get_metadata_freq_df()
        term_freq_df = hamlet.get_term_freq_df()
        assert term_freq_df.loc['a'].sum() == metadata_freq_df.loc[['0++a', '1++a', '2++a']].values.sum()

    def test_metadata_in_use(self):
        hamlet = get_hamlet_term_doc_matrix()
        self.assertFalse(hamlet.metadata_in_use())
        hamlet_meta = build_hamlet_jz_corpus_with_meta()
        self.assertTrue(hamlet_meta.metadata_in_use())

    def test_get_term_doc_mat(self):
        hamlet = get_hamlet_term_doc_matrix()
        X = hamlet.get_term_doc_mat()
        np.testing.assert_array_equal(X.shape, (hamlet.get_num_docs(), hamlet.get_num_terms()))

    def test_get_metadata_doc_mat(self):
        hamlet_meta = build_hamlet_jz_corpus_with_meta()
        mX = hamlet_meta.get_metadata_doc_mat()
        np.testing.assert_array_equal(mX.shape, (hamlet_meta.get_num_docs(), len(hamlet_meta.get_metadata_freq_df())))

    def test_get_metadata_freq_df(self):
        hamlet_meta = build_hamlet_jz_corpus_with_meta()
        mdf = hamlet_meta.get_metadata_freq_df()
        self.assertEqual(list(mdf.columns), ['hamlet freq', 'jay-z/r. kelly freq'])
        mdf = hamlet_meta.get_metadata_freq_df('')
        self.assertEqual(list(mdf.columns), ['hamlet', 'jay-z/r. kelly'])

    def test_get_metadata(self):
        hamlet_meta = build_hamlet_jz_corpus_with_meta()
        self.assertEqual(hamlet_meta.get_metadata(), ['cat1'])

    def test_get_category_index_store(self):
        hamlet = get_hamlet_term_doc_matrix()
        idxstore = hamlet.get_category_index_store()
        self.assertIsInstance(idxstore, IndexStore)
        self.assertEquals(idxstore.getvals(), {'hamlet', 'not hamlet'})

    def test_use_categories_as_metadata(self):
        hamlet = get_hamlet_term_doc_matrix()
        meta_hamlet = hamlet.use_categories_as_metadata()
        self.assertEqual(hamlet.get_metadata_doc_mat().toarray(), [[0]])
        self.assertEqual(meta_hamlet.get_metadata(), ['hamlet', 'not hamlet'])
        self.assertEqual(meta_hamlet.get_metadata_doc_mat().shape, (meta_hamlet.get_num_docs(), len(meta_hamlet.get_metadata())))
        self.assertTrue(all(meta_hamlet.get_metadata_doc_mat().todense().T[0].astype(bool).A1 == (meta_hamlet.get_category_names_by_row() == 'hamlet')))
        self.assertTrue(all(meta_hamlet.get_metadata_doc_mat().todense().T[1].astype(bool).A1 == (meta_hamlet.get_category_names_by_row() == 'not hamlet')))

    def test_use_categories_as_metadata_and_replace_terms(self):
        hamlet = build_hamlet_jz_corpus_with_meta()
        meta_hamlet = hamlet.use_categories_as_metadata_and_replace_terms()
        np.testing.assert_array_almost_equal(hamlet.get_metadata_doc_mat().toarray(), np.array([[2] for _ in range(8)]))
        self.assertEqual(meta_hamlet.get_metadata(), ['hamlet', 'jay-z/r. kelly'])
        self.assertEqual(meta_hamlet.get_metadata_doc_mat().shape, (meta_hamlet.get_num_docs(), len(meta_hamlet.get_metadata())))
        self.assertTrue(all(meta_hamlet.get_metadata_doc_mat().todense().T[0].astype(bool).A1 == (meta_hamlet.get_category_names_by_row() == 'hamlet')))
        self.assertTrue(all(meta_hamlet.get_metadata_doc_mat().todense().T[1].astype(bool).A1 == (meta_hamlet.get_category_names_by_row() == 'jay-z/r. kelly')))
        np.testing.assert_array_equal(meta_hamlet.get_term_doc_mat().todense(), hamlet.get_metadata_doc_mat().toarray())

    def test_copy_terms_to_metadata(self):
        tdm = get_hamlet_term_doc_matrix()
        tdm_meta = tdm.copy_terms_to_metadata()
        self.assertEqual(tdm.get_metadata_doc_mat().shape, (1, 1))
        self.assertEqual(tdm.get_term_doc_mat().shape, (211, 26875))
        self.assertEqual(tdm_meta.get_term_doc_mat().shape, (211, 26875))
        self.assertEqual(tdm_meta.get_metadata_doc_mat().shape, (211, 26875))
        np.testing.assert_array_equal(tdm.get_term_freq_df().index, tdm_meta.get_metadata_freq_df().index)

    def test_get_num_metadata(self):
        self.assertEqual(get_hamlet_term_doc_matrix().use_categories_as_metadata().get_num_metadata(), 2)

    def test_get_num_categories(self):
        self.assertEqual(get_hamlet_term_doc_matrix().get_num_categories(), 2)

    def test_recategorize(self):
        hamlet = get_hamlet_term_doc_matrix()
        newcats = ['cat' + str(i % 3) for i in range(hamlet.get_num_docs())]
        newhamlet = hamlet.recategorize(newcats)
        self.assertEquals(set(newhamlet.get_term_freq_df('').columns), {'cat0', 'cat1', 'cat2'})
        self.assertEquals(set(hamlet.get_term_freq_df('').columns), {'hamlet', 'not hamlet'})

    def test_get_metadata_count_mat(self):
        corpus = build_hamlet_jz_corpus_with_meta()
        np.testing.assert_array_almost_equal(corpus.get_metadata_count_mat(), [[4, 4]])

    def test_get_metadata_doc_count_df(self):
        corpus = build_hamlet_jz_corpus_with_meta()
        np.testing.assert_array_almost_equal(corpus.get_metadata_doc_count_df(), [[4, 4]])
        self.assertEqual(list(corpus.get_metadata_doc_count_df().columns), ['hamlet freq', 'jay-z/r. kelly freq'])
        self.assertEqual(list(corpus.get_metadata_doc_count_df().index), ['cat1'])

    def test_change_categories(self):
        corpus = build_hamlet_jz_corpus_with_meta()
        with self.assertRaisesRegex(Exception, 'The number of category names passed \\(0\\) needs to equal the number of categories in the corpus \\(2\\)\\.'):
            corpus.change_category_names([])
        with self.assertRaisesRegex(Exception, 'The number of category names passed \\(1\\) needs to equal the number of categories in the corpus \\(2\\)\\.'):
            corpus.change_category_names(['a'])
        new_corpus = corpus.change_category_names(['aaa', 'bcd'])
        self.assertEquals(new_corpus.get_categories(), ['aaa', 'bcd'])
        self.assertEquals(corpus.get_categories(), ['hamlet', 'jay-z/r. kelly'])

def build_hamlet_jz_corpus_with_meta():

    def empath_mock(doc, **kwargs):
        toks = list(doc)
        num_toks = min(3, len(toks))
        return {'cat' + str(len(tok)): val for val, tok in enumerate(toks[:num_toks])}
    categories, documents = get_docs_categories()
    clean_function = lambda text: '' if text.startswith('[') else text
    df = pd.DataFrame({'category': categories, 'parsed': [whitespace_nlp(clean_function(doc)) for doc in documents]})
    df = df[df['parsed'].apply(lambda x: len(str(x).strip()) > 0)]
    return CorpusFromParsedDocuments(df=df, category_col='category', parsed_col='parsed', feats_from_spacy_doc=FeatsFromSpacyDocAndEmpath(empath_analyze_function=empath_mock)).build()

class TestZScores(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.corpus = build_hamlet_jz_corpus()

    def test_get_scores(self):
        result = ZScores(self.corpus).set_categories('hamlet').get_scores()
        self.assertEquals(type(result), pd.Series)
        np.testing.assert_array_equal(np.array(result.index), self.corpus.get_terms())

    def test_get_name(self):
        self.assertEquals(ZScores(self.corpus).set_categories('hamlet').get_name(), "Z-Score from Welch's T-Test")

    def test_get_ranks_meta(self):
        corpus = build_hamlet_jz_corpus_with_meta()
        self.assertEquals(ZScores(corpus).set_term_ranker(OncePerDocFrequencyRanker).set_categories('hamlet').get_name(), "Z-Score from Welch's T-Test")

class TestScatterChart(TestCase):

    def test_to_json(self):
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0).to_dict('hamlet')
        self.assertEqual(set(j.keys()), set(['info', 'data']))
        self.assertEqual(set(j['info'].keys()), set(['not_category_name', 'category_name', 'category_terms', 'not_category_terms', 'category_internal_name', 'not_category_internal_names', 'neutral_category_internal_names', 'extra_category_internal_names', 'categories']))
        expected = {'x': 0.0, 'y': 0.42, 'ox': 0, 'oy': 0.42, 'term': 'art', 'cat25k': 758, 'ncat25k': 0, 'neut25k': 0, 'neut': 0, 'extra25k': 0, 'extra': 0, 's': 0.5, 'os': 3, 'bg': 3}
        datum = self._get_data_example(j)
        for var in ['cat25k', 'ncat25k']:
            np.testing.assert_almost_equal(expected[var], datum[var], decimal=1)
        self.assertEqual(set(expected.keys()), set(datum.keys()))
        self.assertEqual(expected['term'], datum['term'])

    def test_to_dict_without_categories(self):
        tdm = get_term_doc_matrix_without_categories()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        with self.assertRaises(NeedToInjectCoordinatesException):
            scatter_chart.to_dict_without_categories()
        x_coords = tdm.get_term_doc_mat().sum(axis=0).A1
        y_coords = tdm.get_term_doc_mat().astype(bool).astype(int).sum(axis=0).A1
        scatter_chart.inject_coordinates(original_x=x_coords, original_y=y_coords, x_coords=scale(x_coords), y_coords=scale(y_coords))
        j = scatter_chart.to_dict_without_categories()
        self.assertIsInstance(j, dict)
        self.assertEqual(set(j.keys()), set(['data']))
        self.assertEqual(len(j['data']), tdm.get_num_terms())
        self.assertEqual(j['data'][-1], {'cat': 4, 'cat25k': 735, 'ox': 4, 'oy': 3, 'term': 'speak', 'x': 1.0, 'y': 1.0})

    def test_resuse_is_disabled(self):
        corpus = get_test_corpus()
        sc = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0)
        sc.to_dict('hamlet')
        with self.assertRaises(Exception):
            sc.to_dict('hamlet')

    def test_score_transform(self):
        corpus = get_test_corpus()
        sc = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0)
        d1 = sc.to_dict('hamlet')
        sc = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0, score_transform=lambda x: x)
        d2 = sc.to_dict('hamlet')
        assert sum([datum['s'] for datum in d1['data']]) != sum([datum['s'] for datum in d2['data']])

    def test_multi_categories(self):
        corpus = get_test_corpus()
        j_vs_all = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0).to_dict('hamlet')
        j_vs_swift = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0).to_dict('hamlet', not_categories=['swift'])
        self.assertNotEqual(set(j_vs_all['info']['not_category_internal_names']), set(j_vs_swift['info']['not_category_internal_names']))
        self.assertEqual(j_vs_all['info']['categories'], corpus.get_categories())
        self.assertEqual(j_vs_swift['info']['categories'], corpus.get_categories())

    def test_title_case_names(self):
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0).to_dict('hamlet', 'HAMLET', 'NOT HAMLET')
        self.assertEqual(j['info']['category_name'], 'HAMLET')
        self.assertEqual(j['info']['not_category_name'], 'NOT HAMLET')
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0).to_dict('hamlet', 'HAMLET', 'NOT HAMLET', title_case_names=True)
        self.assertEqual(j['info']['category_name'], 'Hamlet')
        self.assertEqual(j['info']['not_category_name'], 'Not Hamlet')

    def _get_data_example(self, j):
        return [t for t in j['data'] if t['term'] == 'art'][0]

    def test_terms_to_include(self):
        tdm = build_hamlet_jz_term_doc_mat()
        terms_to_include = list(sorted(['both worlds', 'thou', 'the', 'of', 'st', 'returned', 'best']))
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, terms_to_include=terms_to_include).to_dict('hamlet', 'HAMLET', 'NOT HAMLET')
        self.assertEqual(list(sorted((t['term'] for t in j['data']))), terms_to_include)

    def test_p_vals(self):
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, term_significance=LogOddsRatioUninformativeDirichletPrior()).to_dict('hamlet')
        datum = self._get_data_example(j)
        self.assertIn('p', datum.keys())

    def test_inject_coordinates(self):
        tdm = build_hamlet_jz_term_doc_mat()
        freq_df = tdm.get_term_freq_df()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates([], [])
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(freq_df[freq_df.columns[0]], [])
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates([], freq_df[freq_df.columns[0]])
        x = freq_df[freq_df.columns[1]].astype(np.float64)
        y = freq_df[freq_df.columns[0]].astype(np.float64)
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(x, y)
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(x, y / y.max())
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(x / x.max(), y)
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(-x / x.max(), -y / y.max())
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(-x / x.max(), y / y.max())
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(x / x.max(), -y / y.max())
        scatter_chart.inject_coordinates(x / x.max(), y / y.max())

    def test_inject_metadata_term_lists(self):
        tdm = build_hamlet_jz_term_doc_mat()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        with self.assertRaises(TermDocMatrixHasNoMetadataException):
            scatter_chart.inject_metadata_term_lists({'blah': ['a', 'adsf', 'asfd']})
        scatter_chart = ScatterChart(term_doc_matrix=build_hamlet_jz_corpus_with_meta(), minimum_term_frequency=0, use_non_text_features=True)
        with self.assertRaises(TypeError):
            scatter_chart.inject_metadata_term_lists({'blash': [3, 1]})
        with self.assertRaises(TypeError):
            scatter_chart.inject_metadata_term_lists({3: ['a', 'b']})
        with self.assertRaises(TypeError):
            scatter_chart.inject_metadata_term_lists({'a': {'a', 'b'}})
        with self.assertRaises(TypeError):
            scatter_chart.inject_metadata_term_lists(3)
        self.assertEqual(type(scatter_chart.inject_metadata_term_lists({'a': ['a', 'b']})), ScatterChart)
        j = scatter_chart.to_dict('hamlet')
        self.assertEqual(set(j.keys()), set(['info', 'data', 'metalists']))
        self.assertEqual(set(j['info'].keys()), set(['not_category_name', 'category_name', 'category_terms', 'not_category_terms', 'category_internal_name', 'not_category_internal_names', 'extra_category_internal_names', 'neutral_category_internal_names', 'categories']))

    def test_inject_metadata_descriptions(self):
        tdm = build_hamlet_jz_corpus_with_meta()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        with self.assertRaises(AssertionError):
            scatter_chart.inject_metadata_descriptions(3323)
        if sys.version_info > (3, 0):
            "\n            with self.assertRaisesRegex(Exception, 'The following meta data terms are not present: blah'):\n                scatter_chart.inject_metadata_descriptions({'blah': 'asjdkflasdjklfsadjk jsdkafsd'})\n            with self.assertRaisesRegex(Exception, 'The following meta data terms are not present: cat2'):\n                scatter_chart.inject_metadata_descriptions({'cat1': 'asjdkflasdjklfsadjk jsdkafsd', 'cat2': 'asdf'})\n            "
        assert scatter_chart == scatter_chart.inject_metadata_descriptions({'cat1': 'asjdkflasdjklfsadjk jsdkafsd'})
        j = scatter_chart.to_dict('hamlet')
        self.assertEqual(set(j.keys()), set(['info', 'data', 'metadescriptions']))

    def test_inject_term_colors(self):
        tdm = build_hamlet_jz_corpus_with_meta()
        freq_df = tdm.get_term_freq_df()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        scatter_chart.inject_term_colors({'t1': '00ffee'})
        j = scatter_chart.to_dict('hamlet')
        self.assertIn('term_colors', j['info'])

    def test_inject_coordinates_original(self):
        tdm = build_hamlet_jz_term_doc_mat()
        freq_df = tdm.get_term_freq_df()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        x = freq_df[freq_df.columns[1]].astype(np.float64)
        y = freq_df[freq_df.columns[0]].astype(np.float64)
        scatter_chart.inject_coordinates(x / x.max(), y / y.max(), original_x=x, original_y=y)
        j = scatter_chart.to_dict('hamlet')
        self.assertEqual(j['data'][0].keys(), {'x', 'os', 'y', 'ncat25k', 'neut', 'cat25k', 'ox', 'neut25k', 'extra25k', 'extra', 'oy', 'term', 's', 'bg'})
        and_term = [t for t in j['data'] if t['term'] == 'and'][0]
        self.assertEqual(and_term['ox'], 0)
        self.assertEqual(and_term['oy'], 1)

    def test_to_json_use_non_text_features(self):
        tdm = build_hamlet_jz_corpus_with_meta()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, use_non_text_features=True).to_dict('hamlet')
        self.assertEqual(set(j.keys()), set(['info', 'data']))
        self.assertEqual(set(j['info'].keys()), set(['not_category_name', 'category_name', 'category_terms', 'not_category_terms', 'category_internal_name', 'not_category_internal_names', 'extra_category_internal_names', 'neutral_category_internal_names', 'categories']))
        self.assertEqual({t['term'] for t in j['data']}, {'cat1'})

    def test_max_terms(self):
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, max_terms=2).to_dict('hamlet')
        self.assertEqual(2, len(j['data']))
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, max_terms=10).to_dict('hamlet')
        self.assertEqual(10, len(j['data']))
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, pmi_threshold_coefficient=0, max_terms=10000).to_dict('hamlet')
        self.assertEqual(len(tdm.get_term_freq_df()), len(j['data']))
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, pmi_threshold_coefficient=0, max_terms=None).to_dict('hamlet')
        self.assertEqual(len(tdm.get_term_freq_df()), len(j['data']))

class TestFeatureList(TestCase):

    def test_main(self):
        tdm = build_hamlet_jz_corpus_with_meta()
        features = FeatureLister(tdm._mX, tdm._metadata_idx_store, tdm.get_num_docs()).output()
        expected = [{'cat4': 2, 'cat3': 1}, {'cat4': 2}, {'cat5': 1, 'cat3': 2}, {'cat6': 2, 'cat9': 1}, {'cat4': 2, 'cat3': 1}, {'cat2': 1, 'cat1': 2}, {'cat2': 2, 'cat5': 1}, {'cat4': 1, 'cat3': 2}]
        expected = [{'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}]
        self.assertEqual(features, expected)

