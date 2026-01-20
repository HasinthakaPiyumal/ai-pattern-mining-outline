# Cluster 18

class TermDocMatrixFactory(object):

    def __init__(self, category_text_iter=None, clean_function=lambda x: x, nlp=None, feats_from_spacy_doc=None):
        """
        Class for easy construction of a term document matrix.
       This class let's you define an iterator for each document (text_iter),
       an iterator for each document's category name (category_iter),
       and a document cleaning function that's applied to each document
       before it's parsed.

       Parameters
       ----------
       category_text_iter : iter<str: category, unicode: document)>
           An iterator of pairs. The first element is a string category
           name, the second the text of a document.  You can also set this
           using the function set_category_text_iter.
       clean_function : function (default lambda x: x)
           A function that strips invalid characters out of a string, returning
           the new string.
       post_nlp_clean_function : function (default lambda x: x)
           A function that takes a spaCy Doc
       nlp : spacy.load('en_core_web_sm') (default None)
           The spaCy parser used to parse documents.  If it's None,
           the class will go through the expensive operation of
           creating one to parse the text
       feats_from_spacy_doc : FeatsFromSpacyDoc (default None)
           Class for extraction of features from spacy
       Attributes
       ----------
       _clean_function : Callable
           function that takes a unicode document and returns
           a cleaned version of that document
       _text_iter : iter<unicode>
           an iterator that iterates through the unicode text of each
            document
       _category_iter : iter<str>
           an iterator the same size as text iter that gives a string or
           unicode name of each document catgory
       Examples
       --------
       >>> import scattertext as ST
       >>> documents = [u'What art thou that usurp''st this time of night,',
       ...u'Together with that fair and warlike form',
       ...u'In which the majesty of buried Denmark',
       ...u'Did sometimes march? by heaven I charge thee, speak!',
         ...u'Halt! Who goes there?',
         ...u'[Intro]',
         ...u'It is I sire Tone from Brooklyn.',
         ...u'Well, speak up man what is it?',
         ...u'News from the East sire! THE BEST OF BOTH WORLDS HAS RETURNED!']
       >>> categories = ['hamlet'] * 4 + ['jay-z/r. kelly'] * 5
       >>> clean_function = lambda text: '' if text.startswith('[') else text
       >>> term_doc_mat = ST.TermDocMatrixFactory(category_text_iter = zip(categories, documents),clean_function = clean_function).build()
        """
        self._category_text_iter = category_text_iter
        self._clean_function = clean_function
        self._nlp = nlp
        self._entity_types_to_censor = set()
        if feats_from_spacy_doc is None:
            self._feats_from_spacy_doc = FeatsFromSpacyDoc()
        else:
            self._feats_from_spacy_doc = feats_from_spacy_doc

    def set_category_text_iter(self, category_text_iter):
        """Initializes the category_text_iter

       Paramters
       ----------
       category_text_iter : iter<str: category, unicode: document)>
               An iterator of pairs. The first element is a string category
               name, the second the text of a document.

         Returns
         ----------
         self: TermDocMatrixFactory
        """
        self._category_text_iter = category_text_iter
        return self

    def set_nlp(self, nlp):
        """Adds a spaCy-compatible nlp function

       Paramters
       ----------
       nlp : spacy model

         Returns
         ----------
         self: TermDocMatrixFactory
        """
        self._nlp = nlp
        return self

    def build(self):
        """Generate a TermDocMatrix from data in parameters.

         Returns
         ----------
         term_doc_matrix : TermDocMatrix
            The object that this factory class builds.
        """
        if self._category_text_iter is None:
            raise CategoryTextIterNotSetError()
        nlp = self.get_nlp()
        category_document_iter = ((str(category), self._clean_function(raw_text)) for category, raw_text in self._category_text_iter)
        term_doc_matrix = self._build_from_category_spacy_doc_iter(((category, nlp(text)) for category, text in category_document_iter if text.strip() != ''))
        return term_doc_matrix

    def get_nlp(self):
        nlp = self._nlp
        if nlp is None:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        return nlp

    def censor_entity_types(self, entity_types):
        """
        Entity types to exclude from feature construction. Terms matching
        specificed entities, instead of labeled by their lower case orthographic
        form or lemma, will be labeled by their entity type.

        Parameters
        ----------
        entity_types : set of entity types outputted by spaCy



        Returns
        ---------
        self
        """
        assert type(entity_types) == set
        self._entity_types_to_censor = entity_types
        self._feats_from_spacy_doc = FeatsFromSpacyDoc(use_lemmas=self._use_lemmas, entity_types_to_censor=self._entity_types_to_censor)
        return self

    def _build_from_category_spacy_doc_iter(self, category_doc_iter):
        """
        Parameters
        ----------
        category_doc_iter : iterator of (string category name, spacy.tokens.doc.Doc) pairs

        Returns
        ----------
        t : TermDocMatrix
        """
        term_idx_store = IndexStore()
        category_idx_store = IndexStore()
        metadata_idx_store = IndexStore()
        X, mX, y = self._get_features_and_labels_from_documents_and_indexes(category_doc_iter, category_idx_store, term_idx_store, metadata_idx_store)
        return TermDocMatrix(X, mX, y, term_idx_store=term_idx_store, category_idx_store=category_idx_store, metadata_idx_store=metadata_idx_store)

    def _get_features_and_labels_from_documents_and_indexes(self, category_doc_iter, category_idx_store, term_idx_store, metadata_idx_store):
        y = []
        X_factory = CSRMatrixFactory()
        mX_factory = CSRMatrixFactory()
        for document_index, (category, parsed_text) in enumerate(category_doc_iter):
            self._register_doc_and_category(X_factory, mX_factory, category, category_idx_store, document_index, parsed_text, term_idx_store, metadata_idx_store, y)
        X = X_factory.get_csr_matrix()
        mX = mX_factory.get_csr_matrix()
        y = np.array(y)
        return (X, mX, y)

    def _old_register_doc_and_category(self, X_factory, category, category_idx_store, document_index, parsed_text, term_idx_store, y):
        y.append(category_idx_store.getidx(category))
        document_features = self._get_features_from_parsed_text(parsed_text, term_idx_store)
        self._register_document_features_with_X_factory(X_factory, document_index, document_features)

    def _register_doc_and_category(self, X_factory, mX_factory, category, category_idx_store, document_index, parsed_text, term_idx_store, metadata_idx_store, y):
        self._register_doc(X_factory, mX_factory, document_index, parsed_text, term_idx_store, metadata_idx_store)
        self._register_category(category, category_idx_store, y)

    def _register_doc(self, X_factory, mX_factory, document_index, parsed_text, term_idx_store, metadata_idx_store):
        for term, count in self._feats_from_spacy_doc.get_feats(parsed_text).items():
            term_idx = term_idx_store.getidx(term)
            X_factory[document_index, term_idx] = count
        for term, val in self._feats_from_spacy_doc.get_doc_metadata(parsed_text).items():
            meta_idx = metadata_idx_store.getidx(term)
            mX_factory[document_index, meta_idx] = val

    def _register_category(self, category, category_idx_store, y):
        y.append(category_idx_store.getidx(category))

    def _register_document_features_with_X_factory(self, X_factory, doci, term_freq):
        for word_idx, freq in term_freq.items():
            X_factory[doci, word_idx] = freq

    def _get_features_from_parsed_text(self, parsed_text, term_idx_store):
        return {term_idx_store.getidxstrict(k): v for k, v in self._feats_from_spacy_doc.get_feats(parsed_text).items() if k in term_idx_store}

class CorpusWithoutCategoriesFromParsedDocuments(object):

    def __init__(self, df, parsed_col, feats_from_spacy_doc=FeatsFromSpacyDoc()):
        """
        Parameters
        ----------
        df : pd.DataFrame
         contains category_col, and parse_col, were parsed col is entirely spacy docs
        parsed_col : str
            name of spacy parsed column in convention_df
        feats_from_spacy_doc : FeatsFromSpacyDoc
        """
        self.df = df
        self.parsed_col = parsed_col
        self.feats_from_spacy_doc = feats_from_spacy_doc

    def build(self):
        """

        :return: ParsedCorpus
        """
        category_col = 'Category'
        while category_col in self.df:
            category_col = 'Category_' + ''.join((np.random.choice(string.ascii_letters) for _ in range(5)))
        return CorpusFromParsedDocuments(self.df.assign(**{category_col: '_'}), category_col, self.parsed_col, feats_from_spacy_doc=self.feats_from_spacy_doc).build()

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

def get_docs_categories():
    documents = [u"What art thou that usurp'st this time of night,", u'Together with that fair and warlike form', u'In which the majesty of buried Denmark', u'Did sometimes march? by heaven I charge thee, speak!', u'Halt! Who goes there?', u'[Intro]', u'It is I sire Tone from Brooklyn.', u'Well, speak up man what is it?', u'News from the East sire! THE BEST OF BOTH WORLDS HAS RETURNED!', u'Speak up, speak up, this is a repeat bigram.']
    categories = ['hamlet'] * 4 + ['jay-z/r. kelly'] * 5 + ['???']
    return (categories, documents)

class TestDocsAndLabelsFromCorpus(TestCase):

    @classmethod
    def setUp(cls):
        cls.categories, cls.documents = get_docs_categories()
        cls.parsed_docs = []
        for doc in cls.documents:
            cls.parsed_docs.append(whitespace_nlp(doc))
        cls.df = pd.DataFrame({'category': cls.categories, 'parsed': cls.parsed_docs, 'orig': [d.upper() for d in cls.documents]})
        cls.parsed_corpus = CorpusFromParsedDocuments(cls.df, 'category', 'parsed').build()
        cls.corpus = CorpusFromPandas(cls.df, 'category', 'orig', nlp=whitespace_nlp).build()

    def test_categories(self):
        for obj in [DocsAndLabelsFromCorpusSample(self.parsed_corpus, 1), DocsAndLabelsFromCorpus(self.parsed_corpus)]:
            output = obj.get_labels_and_texts()
            self.assertEqual(output['categories'], ['hamlet', 'jay-z/r. kelly', '???'])
            metadata = ['element 0 0', 'element 1 0', 'element 2 0', 'element 3 0', 'element 4 1', 'element 5 1', 'element 6 1', 'element 7 1', 'element 8 1', 'element 9 2']
            output = obj.get_labels_and_texts_and_meta(metadata)
            self.assertEqual(output['categories'], ['hamlet', 'jay-z/r. kelly', '???'])

    def test_main(self):
        d = DocsAndLabelsFromCorpus(self.parsed_corpus)
        output = d.get_labels_and_texts()
        self.assertTrue('texts' in output)
        self.assertTrue('labels' in output)
        self.assertEqual(self.parsed_corpus._y.astype(int).tolist(), list(output['labels']))
        self.assertEqual(self.parsed_corpus.get_texts().tolist(), list(output['texts']))

    def test_extra_features(self):
        corpus = build_hamlet_jz_corpus_with_meta()
        d = DocsAndLabelsFromCorpus(corpus).use_non_text_features()
        metadata = ['meta%s' % i for i in range(corpus.get_num_docs())]
        output = d.get_labels_and_texts_and_meta(metadata)
        extra_val = [{'cat3': 1, 'cat4': 2}, {'cat4': 2}, {'cat5': 1, 'cat3': 2}, {'cat9': 1, 'cat6': 2}, {'cat3': 1, 'cat4': 2}, {'cat1': 2, 'cat2': 1}, {'cat5': 1, 'cat2': 2}, {'cat3': 2, 'cat4': 1}]
        extra_val = [{'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}, {'cat1': 2}]
        output['labels'] = list(output['labels'])
        self.assertEqual(output, {'categories': ['hamlet', 'jay-z/r. kelly'], 'texts': ["what art thou that usurp'st this time of night,", 'together with that fair and warlike form', 'in which the majesty of buried denmark', 'did sometimes march? by heaven i charge thee, speak!', 'halt! who goes there?', 'it is i sire tone from brooklyn.', 'well, speak up man what is it?', 'news from the east sire! the best of both worlds has returned!'], 'meta': ['meta0', 'meta1', 'meta2', 'meta3', 'meta4', 'meta5', 'meta6', 'meta7'], 'labels': [0, 0, 0, 0, 1, 1, 1, 1], 'extra': extra_val})

    def test_alternative_text_field(self):
        DocsAndLabelsFromCorpus(self.corpus)
        DocsAndLabelsFromCorpus(self.parsed_corpus)
        with self.assertRaises(CorpusShouldBeParsedCorpusException):
            DocsAndLabelsFromCorpus(self.corpus, alternative_text_field='orig')
        d = DocsAndLabelsFromCorpus(self.parsed_corpus, alternative_text_field='orig')
        self.assertEqual(d.get_labels_and_texts()['texts'][0], d.get_labels_and_texts()['texts'][0].upper())
        d = DocsAndLabelsFromCorpus(self.parsed_corpus)
        self.assertNotEqual(d.get_labels_and_texts()['texts'][0], d.get_labels_and_texts()['texts'][0].upper())
        d = DocsAndLabelsFromCorpusSample(self.parsed_corpus, 2, alternative_text_field='orig', seed=0)
        texts = d.get_labels_and_texts()['texts']
        self.assertEqual(texts[0], texts[0].upper())
        d = DocsAndLabelsFromCorpusSample(self.parsed_corpus, 2)
        self.assertNotEqual(d.get_labels_and_texts()['texts'][0], d.get_labels_and_texts()['texts'][0].upper())

    def test_metadata(self):
        d = DocsAndLabelsFromCorpus(self.parsed_corpus)
        metadata = ['element 0 0', 'element 1 0', 'element 2 0', 'element 3 0', 'element 4 1', 'element 5 1', 'element 6 1', 'element 7 1', 'element 8 1', 'element 9 2']
        output = d.get_labels_and_texts_and_meta(metadata)
        self.assertTrue('texts' in output)
        self.assertTrue('labels' in output)
        self.assertTrue('meta' in output)
        self.assertEqual(output['meta'], metadata)

    def test_max_per_category(self):
        docs_and_labels = DocsAndLabelsFromCorpusSample(self.parsed_corpus, max_per_category=2, seed=0)
        metadata = np.array(['element 0 0', 'element 1 0', 'element 2 0', 'element 3 0', 'element 4 1', 'element 5 1', 'element 6 1', 'element 7 1', 'element 8 1', 'element 9 2'])
        output = docs_and_labels.get_labels_and_texts_and_meta(metadata)
        self.assertTrue('texts' in output)
        self.assertTrue('labels' in output)
        self.assertTrue('meta' in output)
        self.assertTrue('extra' not in output)
        d = {}
        for text, lab, meta in zip(output['texts'], output['labels'], output['meta']):
            d.setdefault(lab, []).append(text)
        for lab, documents in d.items():
            self.assertLessEqual(len(documents), 2)
        json.dumps(d)
        docs_and_labels = DocsAndLabelsFromCorpusSample(self.parsed_corpus, max_per_category=2)
        output = docs_and_labels.get_labels_and_texts()
        self.assertTrue('texts' in output)
        self.assertTrue('labels' in output)
        self.assertTrue('meta' not in output)
        self.assertTrue('extra' not in output)
        d = {}
        for text, lab in zip(output['texts'], output['labels']):
            d.setdefault(lab, []).append(text)
        for lab, documents in d.items():
            self.assertLessEqual(len(documents), 2)
        json.dumps(d)
        docs_and_labels = DocsAndLabelsFromCorpusSample(self.parsed_corpus, max_per_category=2).use_non_text_features()
        output = docs_and_labels.get_labels_and_texts()
        self.assertTrue('texts' in output)
        self.assertTrue('labels' in output)
        self.assertTrue('meta' not in output)
        self.assertTrue('extra' in output)
        d = {}
        for text, lab in zip(output['texts'], output['labels']):
            d.setdefault(lab, []).append(text)
        for lab, documents in d.items():
            self.assertLessEqual(len(documents), 2)
        json.dumps(d)

def whitespace_nlp(doc, entity_type=None, tag_type=None):
    toks = _regex_parse_sentence(doc, entity_type, tag_type)
    return Doc([toks])

class TestSemioticSquareFromAxes(TestCase):

    @classmethod
    def setUp(cls):
        categories, documents = get_docs_categories()
        cls.df = pd.DataFrame({'category': categories, 'text': documents})
        cls.corpus = CorpusFromPandas(cls.df, 'category', 'text', nlp=whitespace_nlp).build()

    def test_main(self):
        terms = self.corpus.get_terms()
        axes = pd.DataFrame({'x': [len(x) for x in terms], 'y': [sum([ord(c) for c in x]) * 1.0 / len(x) for x in terms]}, index=terms)
        axes['x'] = axes['x'] - axes['x'].median()
        axes['y'] = axes['y'] - axes['y'].median()
        x_axis_label = 'len'
        y_axis_label = 'alpha'
        with self.assertRaises(AssertionError):
            SemioticSquareFromAxes(self.corpus, axes.iloc[:3], x_axis_label, y_axis_label)
        with self.assertRaises(AssertionError):
            axes2 = axes.copy()
            axes2.loc['asdjfksafjd'] = pd.Series({'x': 3, 'y': 3})
            SemioticSquareFromAxes(self.corpus, axes2, x_axis_label, y_axis_label)
        with self.assertRaises(AssertionError):
            SemioticSquareFromAxes(self.corpus, axes2[['x']], x_axis_label, y_axis_label)
        with self.assertRaises(AssertionError):
            axes2 = axes.copy()
            axes2['a'] = 1
            SemioticSquareFromAxes(self.corpus, axes2, x_axis_label, y_axis_label)
        semsq = SemioticSquareFromAxes(self.corpus, axes, x_axis_label, y_axis_label)
        self.assertEqual(semsq.get_labels(), {'a_and_b_label': 'alpha', 'a_and_not_b_label': 'not-len', 'a_label': 'not-len; alpha', 'b_and_not_a_label': 'len', 'b_label': 'len; alpha', 'not_a_and_not_b_label': 'not-alpha', 'not_a_label': 'len; not-alpha', 'not_b_label': 'not-len; not-alpha'})
        self.assertEqual(semsq.get_axes().to_csv(), axes.to_csv())
        self.assertEqual(semsq.get_lexicons(3), {'a': ['st', 'up', 'usurp'], 'a_and_b': ['usurp', 'worlds', 'thou'], 'a_and_not_b': ['and', 'did', 'i'], 'b': ['sometimes', 'brooklyn', 'returned'], 'b_and_not_a': ['sometimes march', 'together with', 'did sometimes'], 'not_a': ['i charge', 'fair and', 'charge thee'], 'not_a_and_not_b': ['is a', 'is i', 'i charge'], 'not_b': ['is a', 'is i', 'it is']})

class TestEmbeddingsResolver(TestCase):

    @classmethod
    def setUp(cls):
        categories, documents = get_docs_categories()
        cls.df = pd.DataFrame({'category': categories, 'text': documents})
        cls.df['parsed'] = cls.df.text.apply(whitespace_nlp)
        cls.corpus = CorpusFromParsedDocuments(cls.df, 'category', 'parsed').build()

    def test_resolve_embeddings(self):
        tdm = self.corpus.get_unigram_corpus().select(ClassPercentageCompactor(term_count=1))
        embeddings_resolver = EmbeddingsResolver(tdm)
        embeddings_resolver = embeddings_resolver.set_embeddings(tdm.get_term_doc_mat())
        if self.assertRaisesRegex:
            with self.assertRaisesRegex(Exception, 'You have already set embeddings by running set_embeddings or set_embeddings_model.'):
                embeddings_resolver.set_embeddings_model(None)
        embeddings_resolver = EmbeddingsResolver(tdm)
        embeddings_resolver = embeddings_resolver.set_embeddings_model(MockWord2Vec(tdm.get_terms()))
        if self.assertRaisesRegex:
            with self.assertRaisesRegex(Exception, 'You have already set embeddings by running set_embeddings or set_embeddings_model.'):
                embeddings_resolver.set_embeddings(tdm.get_term_doc_mat())
        c, axes = embeddings_resolver.project_embeddings(projection_model=TruncatedSVD(3))
        self.assertIsInstance(c, ParsedCorpus)
        self.assertEqual(axes.to_dict(), pd.DataFrame(index=['speak'], data={'x': [0.0], 'y': [0.0]}).to_dict())

class TestUnigramsFromSpacyDoc(TestCase):

    def test_get_feats(self):
        doc = whitespace_nlp('A a bb cc.')
        term_freq = UnigramsFromSpacyDoc().get_feats(doc)
        self.assertEqual(Counter({'a': 2, 'bb': 1, 'cc': 1}), term_freq)

class TestFeatsFromSpacyDoc(TestCase):

    def test_main(self):
        doc = whitespace_nlp('A a bb cc.')
        term_freq = FeatsFromSpacyDoc().get_feats(doc)
        self.assertEqual(Counter({'a': 2, 'bb': 1, 'a bb': 1, 'cc': 1, 'a a': 1, 'bb cc': 1}), term_freq)

    def test_singleton_with_sentences(self):
        doc = whitespace_nlp_with_sentences('Blah')
        term_freq = FeatsFromSpacyDoc().get_feats(doc)
        self.assertEqual(Counter({'blah': 1}), term_freq)

    def test_lemmas(self):
        doc = whitespace_nlp('A a bb ddddd.')
        term_freq = FeatsFromSpacyDoc(use_lemmas=True).get_feats(doc)
        self.assertEqual(Counter({'a': 2, 'bb': 1, 'a bb': 1, 'dd': 1, 'a a': 1, 'bb dd': 1}), term_freq)

    def test_feats_from_spacy_doc_only_chunks(self):
        doc = whitespace_nlp_with_fake_chunks('This is a fake noun chunk generating sentence.')
        term_freq = FeatsFromSpacyDocOnlyNounChunks().get_feats(doc)
        self.assertEqual(term_freq, Counter({'this is': 1, 'is a': 1}))

    def test_empty(self):
        doc = whitespace_nlp('')
        term_freq = FeatsFromSpacyDoc().get_feats(doc)
        self.assertEqual(Counter(), term_freq)

    def test_entity_types_to_censor_not_a_set(self):
        doc = whitespace_nlp('A a bb cc.', {'bb': 'A'})
        with self.assertRaises(AssertionError):
            FeatsFromSpacyDoc(entity_types_to_censor='A').get_feats(doc)

    def test_entity_censor(self):
        doc = whitespace_nlp('A a bb cc.', {'bb': 'BAD'})
        term_freq = FeatsFromSpacyDoc(entity_types_to_censor=set(['BAD'])).get_feats(doc)
        self.assertEqual(Counter({'a': 2, 'a _BAD': 1, '_BAD cc': 1, 'cc': 1, 'a a': 1, '_BAD': 1}), term_freq)

    def test_entity_tags(self):
        doc = whitespace_nlp('A a bb cc Bob.', {'bb': 'BAD'}, {'Bob': 'NNP'})
        term_freq = FeatsFromSpacyDoc(entity_types_to_censor=set(['BAD'])).get_feats(doc)
        self.assertEqual(Counter({'a': 2, 'a _BAD': 1, '_BAD cc': 1, 'cc': 1, 'a a': 1, '_BAD': 1, 'bob': 1, 'cc bob': 1}), term_freq)
        term_freq = FeatsFromSpacyDoc(entity_types_to_censor=set(['BAD']), tag_types_to_censor=set(['NNP'])).get_feats(doc)
        self.assertEqual(Counter({'a': 2, 'a _BAD': 1, '_BAD cc': 1, 'cc': 1, 'a a': 1, '_BAD': 1, 'NNP': 1, 'cc NNP': 1}), term_freq)

    def test_strip_final_period(self):
        doc = bad_whitespace_nlp("I CAN'T ANSWER THAT\n QUESTION.\n I HAVE NOT ASKED THEM\n SPECIFICALLY IF THEY HAVE\n ENOUGH.")
        feats = FeatsFromSpacyDoc().get_feats(doc)
        self.assertEqual(feats, Counter({'i': 2, 'have': 2, 'that question.': 1, 'answer': 1, 'question.': 1, 'enough.': 1, 'i have': 1, 'them specifically': 1, 'have enough.': 1, 'not asked': 1, 'they have': 1, 'have not': 1, 'specifically': 1, 'answer that': 1, 'question. i': 1, "can't": 1, 'if': 1, 'they': 1, "can't answer": 1, 'asked': 1, 'them': 1, 'if they': 1, 'asked them': 1, 'that': 1, 'not': 1, "i can't": 1, 'specifically if': 1}))
        feats = FeatsFromSpacyDoc(strip_final_period=True).get_feats(doc)
        self.assertEqual(feats, Counter({'i': 2, 'have': 2, 'that question': 1, 'answer': 1, 'question': 1, 'enough': 1, 'i have': 1, 'them specifically': 1, 'have enough': 1, 'not asked': 1, 'they have': 1, 'have not': 1, 'specifically': 1, 'answer that': 1, 'question i': 1, "can't": 1, 'if': 1, 'they': 1, "can't answer": 1, 'asked': 1, 'them': 1, 'if they': 1, 'asked them': 1, 'that': 1, 'not': 1, "i can't": 1, 'specifically if': 1}))

class TestParsedCorpus(TestCase):

    @classmethod
    def setUp(cls):
        cls.categories, cls.documents = get_docs_categories()
        cls.parsed_docs = []
        for doc in cls.documents:
            cls.parsed_docs.append(whitespace_nlp(doc))
        cls.df = pd.DataFrame({'category': cls.categories, 'author': ['a', 'a', 'c', 'c', 'c', 'c', 'd', 'd', 'e', 'e'], 'parsed': cls.parsed_docs, 'document_lengths': [len(doc) for doc in cls.documents]})
        cls.corpus = CorpusFromParsedDocuments(cls.df, 'category', 'parsed').build()

    def test_get_text(self):
        self.assertEqual(len([x for x in self.corpus.get_texts()]), len(self.documents))
        self.assertEqual([str(x) for x in self.corpus.get_texts()][0], "what art thou that usurp'st this time of night,")

    def test_get_field(self):
        self.assertEqual(list(self.corpus.get_field('author')), list(self.df.author))

    def test_get_parsed_docs(self):
        doc = [x for x in self.corpus.get_parsed_docs()][0]
        doc.sents

    def test_get_unigram_corpus(self):
        unicorp = self.corpus.get_unigram_corpus()
        self.assertEqual(len([x for x in unicorp.get_texts()]), len(self.documents))
        self.assertEqual([str(x) for x in unicorp.get_texts()][0], "what art thou that usurp'st this time of night,")

    def test_search(self):
        self.assertEqual(len(self.corpus.search('bigram')), 1)
        df = self.corpus.search('bigram')
        d = dict(df.iloc[0])
        self.assertEqual(d['category'], '???')
        self.assertEqual(d['document_lengths'], 44)
        self.assertEqual(str(d['parsed']), 'speak up, speak up, this is a repeat bigram.')
        self.assertEqual(len(self.corpus.search('the')), 2)

    def test_term_group_freq_df(self):
        """
		Returns
		-------
		return pd.DataFrame indexed on terms with columns giving how many attributes in convention_df

		"""
        group_df = self.corpus.term_group_freq_df('author')
        self.assertEqual(set(group_df.index), set(self.corpus._term_idx_store.values()))
        self.assertEqual(dict(group_df.loc['of']), {'??? freq': 0, 'hamlet freq': 2, 'jay-z/r. kelly freq': 1})
        self.assertEqual(dict(group_df.loc['speak up']), {'??? freq': 1, 'hamlet freq': 0, 'jay-z/r. kelly freq': 1})

class TestCategoryColorAssigner(TestCase):

    def test_main(self):
        categories, documents = get_docs_categories()
        df = pd.DataFrame({'category': categories, 'text': documents})
        corpus = CorpusFromPandas(df, 'category', 'text', nlp=whitespace_nlp).build()
        self.assertEqual(CategoryColorAssigner(corpus).get_category_colors().to_dict(), {'???': [255, 127, 14], 'hamlet': [174, 199, 232], 'jay-z/r. kelly': [31, 119, 180]})
        term_colors = CategoryColorAssigner(corpus).get_term_colors()
        self.assertEqual(term_colors['this time'], '#aec7e8')
        self.assertEqual(term_colors['sire'], '#1f77b4')
        self.assertEqual(len(term_colors), corpus.get_num_terms())
        mfact = CSRMatrixFactory()
        mis = IndexStore()
        for i, c in enumerate(df['category']):
            mfact[i, mis.getidx(c)] = 1
        corpus = corpus.add_metadata(mfact.get_csr_matrix(), mis)
        meta_colors = CategoryColorAssigner(corpus, use_non_text_features=True).get_term_colors()
        self.assertEqual(meta_colors, {'hamlet': '#aec7e8', 'jay-z/r. kelly': '#1f77b4', '???': '#ff7f0e'})
        self.assertNotEqual(CategoryColorAssigner(corpus).get_term_colors(), meta_colors)

class TestFeatsFsromSpacyDocAndEmpath(TestCase):

    def test_main(self):
        feat_getter = FeatsFromSpacyDocAndEmpath(empath_analyze_function=mock_empath_analyze)
        sys.modules['empath'] = Mock(analyze=mock_empath_analyze)
        FeatsFromSpacyDocAndEmpath()
        doc = whitespace_nlp('Hello this is a document.')
        term_freq = feat_getter.get_feats(doc)
        self.assertEqual(set(term_freq.items()), set({'document': 1, 'hello': 1, 'is': 1, 'this': 1, 'a document': 1, 'hello this': 1, 'is a': 1, 'a': 1, 'this is': 1}.items()))
        metadata_freq = feat_getter.get_doc_metadata(doc)
        self.assertEqual(metadata_freq['ridicule'], 1)
        self.assertNotIn('empath_fashion', metadata_freq)

    def test_empath_not_presesnt(self):
        sys.modules['empath'] = None
        if sys.version_info.major == 3:
            with self.assertRaisesRegex(Exception, 'Please install the empath library to use FeatsFromSpacyDocAndEmpath.'):
                FeatsFromSpacyDocAndEmpath()
        else:
            with self.assertRaises(Exception):
                FeatsFromSpacyDocAndEmpath()

class TestFeatsFromOnlyEmpath(TestCase):

    def test_main(self):
        sys.modules['empath'] = Mock(analyze=mock_empath_analyze)
        FeatsFromOnlyEmpath()
        feat_getter = FeatsFromOnlyEmpath(empath_analyze_function=mock_empath_analyze)
        doc = whitespace_nlp('Hello this is a document.')
        term_freq = feat_getter.get_feats(doc)
        metadata_freq = feat_getter.get_doc_metadata(doc)
        self.assertEqual(term_freq, Counter())
        self.assertEqual(metadata_freq['ridicule'], 1)
        self.assertNotIn('fashion', metadata_freq)
        self.assertNotIn('document', metadata_freq)
        self.assertNotIn('a document', metadata_freq)

    def test_empath_not_presesnt(self):
        sys.modules['empath'] = None
        if sys.version_info.major == 3:
            with self.assertRaisesRegex(Exception, 'Please install the empath library to use FeatsFromSpacyDocAndEmpath.'):
                FeatsFromSpacyDocAndEmpath()
        else:
            with self.assertRaises(Exception):
                FeatsFromSpacyDocAndEmpath()

class TestCorpusFromParsedDocuments(TestCase):

    @classmethod
    def setUp(cls):
        cls.categories, cls.documents = get_docs_categories()
        cls.parsed_docs = []
        for doc in cls.documents:
            cls.parsed_docs.append(whitespace_nlp(doc))
        cls.df = pd.DataFrame({'category': cls.categories, 'parsed': cls.parsed_docs})
        cls.corpus_fact = CorpusFromParsedDocuments(cls.df, 'category', 'parsed')

    def test_same_as_term_doc_matrix(self):
        term_doc_matrix = build_term_doc_matrix()
        corpus = self._make_political_corpus()
        self.assertEqual(term_doc_matrix._X.shape, corpus._X.shape)
        self.assertEqual((corpus._X != term_doc_matrix._X).nnz, 0)
        corpus_scores = corpus.get_scaled_f_scores('democrat')
        term_doc_matrix_scores = corpus.get_scaled_f_scores('democrat')
        self.assertTrue(np.array_equal(term_doc_matrix_scores, corpus_scores))

    def _make_political_corpus(self):
        clean = clean_function_factory()
        data = []
        for party, speech in iter_party_speech_pairs():
            cleaned_speech = clean(speech)
            if cleaned_speech and cleaned_speech != '':
                parsed_speech = whitespace_nlp(cleaned_speech)
                data.append({'party': party, 'text': parsed_speech})
        corpus = CorpusFromParsedDocuments(pd.DataFrame(data), category_col='party', parsed_col='text').build()
        return corpus

    def test_get_y_and_populate_category_idx_store(self):
        corpus = self.corpus_fact.build()
        self.assertEqual([0, 0, 0, 0, 1, 1, 1, 1, 1, 2], list(corpus._y))
        self.assertEqual([(0, 'hamlet'), (1, 'jay-z/r. kelly'), (2, '???')], list(sorted(list(corpus._category_idx_store.items()))))

    def test_get_term_idx_and_x(self):
        docs = [whitespace_nlp('aa aa bb.'), whitespace_nlp('bb aa a.')]
        df = pd.DataFrame({'category': ['a', 'b'], 'parsed': docs})
        corpus_fact = CorpusFromParsedDocuments(df, category_col='category', parsed_col='parsed')
        corpus = corpus_fact.build()
        kvs = list(corpus_fact._term_idx_store.items())
        keys = [k for k, v in kvs]
        values = [v for k, v in kvs]
        self.assertEqual(sorted(keys), list(range(7)))
        self.assertEqual(sorted(values), ['a', 'aa', 'aa a', 'aa aa', 'aa bb', 'bb', 'bb aa'])

        def assert_word_in_doc_cnt(doc, word, count):
            self.assertEqual(corpus._X[doc, corpus._term_idx_store.getidx(word)], count)
        assert_word_in_doc_cnt(0, 'aa', 2)
        assert_word_in_doc_cnt(0, 'bb', 1)
        assert_word_in_doc_cnt(0, 'aa aa', 1)
        assert_word_in_doc_cnt(0, 'aa bb', 1)
        assert_word_in_doc_cnt(0, 'bb aa', 0)
        assert_word_in_doc_cnt(1, 'bb', 1)
        assert_word_in_doc_cnt(1, 'aa', 1)
        assert_word_in_doc_cnt(1, 'a', 1)
        assert_word_in_doc_cnt(1, 'bb aa', 1)
        assert_word_in_doc_cnt(1, 'aa aa', 0)
        assert_word_in_doc_cnt(1, 'aa a', 1)
        self.assertTrue(isinstance(corpus, ParsedCorpus))

    def test_hamlet(self):
        raw_docs = get_hamlet_docs()
        categories = [get_hamlet_snippet_binary_category(doc) for doc in raw_docs]
        docs = [whitespace_nlp(doc) for doc in raw_docs]
        df = pd.DataFrame({'category': categories, 'parsed': docs})
        corpus_fact = CorpusFromParsedDocuments(df, 'category', 'parsed')
        corpus = corpus_fact.build()
        tdf = corpus.get_term_freq_df()
        self.assertEqual(list(tdf.loc['play']), [37, 5])
        self.assertFalse(any(corpus.search('play').apply(lambda x: 'plfay' in str(x['parsed']), axis=1)))
        self.assertTrue(all(corpus.search('play').apply(lambda x: 'play' in str(x['parsed']), axis=1)))
        play_term_idx = corpus_fact._term_idx_store.getidx('play')
        play_X = corpus._X.todok()[:, play_term_idx]
        self.assertEqual(play_X.sum(), 37 + 5)

def get_term_doc_matrix_without_categories():
    categories, documents = get_docs_categories()
    df = pd.DataFrame({'text': documents})
    tdm = TermDocMatrixWithoutCategoriesFromPandas(df, 'text', nlp=whitespace_nlp).build()
    return tdm

class TestCorpusFromPandasWithoutCategories(TestCase):

    def test_term_category_matrix_from_pandas_without_categories(self):
        tdm = get_term_doc_matrix_without_categories()
        categories, documents = get_docs_categories()
        reg_tdm = TermDocMatrixFromPandas(pd.DataFrame({'text': documents, 'categories': categories}), text_col='text', category_col='categories', nlp=whitespace_nlp).build()
        self.assertIsInstance(tdm, TermDocMatrixWithoutCategories)
        self.assertEqual(tdm.get_terms(), reg_tdm.get_terms())
        self.assertEqual(tdm.get_num_docs(), reg_tdm.get_num_docs())
        np.testing.assert_equal(tdm.get_term_doc_mat().data, reg_tdm.get_term_doc_mat().data)

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

def build_hamlet_jz_df():
    categories, documents = get_docs_categories()
    clean_function = lambda text: '' if text.startswith('[') else text
    df = pd.DataFrame({'category': categories, 'parsed': [whitespace_nlp(clean_function(doc)) for doc in documents]})
    df = df[df['parsed'].apply(lambda x: len(str(x).strip()) > 0)]
    return df

def build_hamlet_jz_df_with_alt_text():
    categories, documents = get_docs_categories()
    clean_function = lambda text: '' if text.startswith('[') else text
    df = pd.DataFrame({'category': categories, 'parsed': [whitespace_nlp(clean_function(doc)) for doc in documents], 'alt': [doc.upper() for doc in documents]})
    df = df[df['parsed'].apply(lambda x: len(str(x).strip()) > 0)]
    return df

class TestGensimPhraseAdder(TestCase):

    @classmethod
    def setUp(cls):
        cls.categories, cls.documents = get_docs_categories()
        cls.parsed_docs = []
        for doc in cls.documents:
            cls.parsed_docs.append(whitespace_nlp(doc))
        cls.df = pd.DataFrame({'category': cls.categories, 'author': ['a', 'a', 'c', 'c', 'c', 'c', 'd', 'd', 'e', 'e'], 'parsed': cls.parsed_docs, 'document_lengths': [len(doc) for doc in cls.documents]})
        cls.corpus = CorpusFromParsedDocuments(cls.df, 'category', 'parsed').build()

    def test_add_phrase(self):
        adder = GensimPhraseAdder()

class TestWord2VecFromParsedCorpus(TestCase):

    @classmethod
    def setUp(cls):
        cls.categories, cls.documents = get_docs_categories()
        cls.parsed_docs = []
        for doc in cls.documents:
            cls.parsed_docs.append(whitespace_nlp(doc))
        cls.df = pd.DataFrame({'category': cls.categories, 'author': ['a', 'a', 'c', 'c', 'c', 'c', 'd', 'd', 'e', 'e'], 'parsed': cls.parsed_docs, 'document_lengths': [len(doc) for doc in cls.documents]})
        cls.corpus = CorpusFromParsedDocuments(cls.df, 'category', 'parsed').build()

    def test_make(self):
        gensim_is_present_and_working = False
        try:
            from gensim.models import word2vec
            gensim_is_present_and_working = True
        except:
            pass
        if gensim_is_present_and_working:
            Word2VecFromParsedCorpus(self.corpus)
            Word2VecFromParsedCorpus(self.corpus, word2vec.Word2Vec())

    def test_train(self):
        gensim_is_present_and_working = False
        try:
            from gensim.models import word2vec
            gensim_is_present_and_working = True
        except:
            pass
        if gensim_is_present_and_working:
            Word2VecFromParsedCorpus(self.corpus).train()

    def test_bigrams(self):
        gensim_is_present_and_working = False
        try:
            from gensim.models import word2vec
            gensim_is_present_and_working = True
        except:
            pass
        if gensim_is_present_and_working:
            Word2VecFromParsedCorpusBigrams(self.corpus).train()

class TestOneClassScatterChart(TestCase):

    def test_main(self):
        df = build_hamlet_jz_df()

