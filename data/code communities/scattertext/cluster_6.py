# Cluster 6

def zero_centered_scale(ar):
    ar[ar > 0] = scale(ar[ar > 0])
    ar[ar < 0] = -scale(-ar[ar < 0])
    return (ar + 1) / 2.0

def scale(vec, other_vec=None):
    if other_vec is None:
        other_vec = vec
    return (other_vec - vec.min()) / (vec.max() - vec.min())

def dense_rank(vec: np.array, terms=None, other_vec=None) -> np.array:
    ranks = rankdata(vec, method='dense') - 1
    if ranks.max() == 0:
        return np.ones(len(vec)) * 0.5
    return ranks / ranks.max()

def zero_centered_scale(ar):
    ar[ar > 0] = scale(ar[ar > 0])
    ar[ar < 0] = -scale(-ar[ar < 0])
    return (ar + 1) / 2.0

def scale(ar):
    return (ar - ar.min()) / (ar.max() - ar.min())

def main():
    nlp = spacy.load('en_core_web_sm')
    convention_df = SampleCorpora.ConventionData2012.get_data()
    corpus = CorpusFromPandas(convention_df, category_col='party', text_col='text', nlp=nlp).build()
    html = word_similarity_explorer(corpus, category='democrat', category_name='Democratic', not_category_name='Republican', target_term='jobs', minimum_term_frequency=5, width_in_pixels=1000, metadata=convention_df['speaker'], alpha=0.01, max_p_val=0.1, save_svg_button=True)
    open('./demo_similarity.html', 'wb').write(html.encode('utf-8'))
    print('Open ./demo_similarlity.html in Chrome or Firefox.')

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

def get_docs_categories_semiotic():
    documents = [u"What art thou that usurp'st this time of night,", u'Together with that fair and warlike form', u'In which the majesty of buried Denmark', u'Did sometimes march? by heaven I charge thee, speak!', u'Halt! Who goes there?', u'[Intro]', u'It is I sire Tone from Brooklyn.', u'Well, speak up man what is it?', u'News from the East sire! THE BEST OF BOTH WORLDS HAS RETURNED!', u'I think it therefore manifest, from what I have here advanced,', u'that the main Point of Skill and Address, is to furnish Employment', u'for this Redundancy of Vapour, and prudently to adjust the Season 1', u'of it ; by which ,means it may certainly become of Cardinal']
    categories = ['hamlet'] * 4 + ['jay-z/r. kelly'] * 5 + ['swift'] * 4
    return (categories, documents)

class TestTermDocMatrixFromPandas(TestCase):

    def test_main(self):
        categories, documents = get_docs_categories()
        df = pd.DataFrame({'category': categories, 'text': documents})
        tdm_factory = TermDocMatrixFromPandas(df, 'category', 'text', nlp=whitespace_nlp)
        term_doc_matrix = tdm_factory.build()
        self.assertIsInstance(term_doc_matrix, TermDocMatrix)
        self.assertEqual(set(term_doc_matrix.get_categories()), set(['hamlet', 'jay-z/r. kelly']))
        self.assertEqual(term_doc_matrix.get_num_docs(), 9)
        term_doc_df = term_doc_matrix.get_term_freq_df()
        self.assertEqual(term_doc_df.loc['of'].sum(), 3)

    def test_one_word_per_docs(self):
        records = [(0, 'verified', 'RAs'), (1, 'view', 'RAs'), (2, 'laminectomy', 'RAs'), (3, 'recognition', 'RAs'), (4, 'possibility', 'RAs'), (5, 'possibility', 'RAs'), (6, 'possibility', 'RAs'), (7, 'observations', 'RAs'), (8, 'observation', 'RAs'), (9, 'observation', 'RAs'), (10, 'observation', 'RAs'), (11, 'observation', 'RAs'), (12, 'observation', 'RAs'), (13, 'implication', 'RAs'), (14, 'idea', 'RAs'), (15, 'hypothesis', 'RAs'), (16, 'fact', 'RAs'), (17, 'fact', 'RAs'), (18, 'fact', 'RAs'), (19, 'fact', 'RAs'), (20, 'fact', 'RAs'), (21, 'surprising', 'RAs'), (22, 'surprising', 'RAs'), (23, 'surprising', 'RAs'), (24, 'suggests', 'RAs'), (25, 'suggests', 'RAs'), (26, 'suggests', 'RAs'), (27, 'suggests', 'RAs'), (28, 'suggests', 'RAs'), (29, 'suggests', 'RAs'), (30, 'suggests', 'RAs'), (31, 'suggests', 'RAs'), (32, 'suggests', 'RAs'), (33, 'suggests', 'RAs'), (34, 'suggests', 'RAs'), (35, 'suggests', 'RAs'), (36, 'suggests', 'RAs'), (37, 'suggests', 'RAs'), (38, 'suggests', 'RAs'), (39, 'suggests', 'RAs'), (40, 'suggests', 'RAs'), (41, 'suggests', 'RAs'), (42, 'suggests', 'RAs'), (43, 'suggests', 'RAs'), (44, 'suggests', 'RAs'), (45, 'suggests', 'RAs'), (46, 'suggests', 'RAs'), (47, 'suggests', 'RAs'), (48, 'suggesting', 'RAs'), (49, 'suggesting', 'RAs'), (50, 'suggesting', 'RAs'), (51, 'suggesting', 'RAs'), (52, 'suggesting', 'RAs'), (53, 'suggesting', 'RAs'), (54, 'suggesting', 'RAs'), (55, 'suggesting', 'RAs'), (56, 'suggesting', 'RAs'), (57, 'suggesting', 'RAs'), (58, 'suggesting', 'RAs'), (59, 'suggesting', 'RAs'), (60, 'suggesting', 'RAs'), (61, 'suggesting', 'RAs'), (62, 'suggesting', 'RAs'), (63, 'suggesting', 'RAs'), (64, 'suggesting', 'RAs'), (65, 'suggesting', 'RAs'), (66, 'suggesting', 'RAs'), (67, 'suggesting', 'RAs'), (68, 'suggesting', 'RAs'), (69, 'suggesting', 'RAs'), (70, 'suggesting', 'RAs'), (71, 'suggesting', 'RAs'), (72, 'suggesting', 'RAs'), (73, 'suggesting', 'RAs'), (74, 'suggesting', 'RAs'), (75, 'suggested', 'RAs'), (76, 'suggested', 'RAs'), (77, 'suggested', 'RAs'), (78, 'suggested', 'RAs'), (79, 'suggested', 'RAs'), (80, 'suggest', 'RAs'), (81, 'suggest', 'RAs'), (82, 'suggest', 'RAs'), (83, 'suggest', 'RAs'), (84, 'suggest', 'RAs'), (85, 'suggest', 'RAs'), (86, 'suggest', 'RAs'), (87, 'suggest', 'RAs'), (88, 'suggest', 'RAs'), (89, 'suggest', 'RAs'), (90, 'suggest', 'RAs'), (91, 'suggest', 'RAs'), (92, 'suggest', 'RAs'), (93, 'suggest', 'RAs'), (94, 'suggest', 'RAs'), (95, 'suggest', 'RAs'), (96, 'suggest', 'RAs'), (97, 'suggest', 'RAs'), (98, 'suggest', 'RAs'), (99, 'suggest', 'RAs'), (100, 'suggest', 'RAs'), (101, 'suggest', 'RAs'), (102, 'suggest', 'RAs'), (103, 'suggest', 'RAs'), (104, 'suggest', 'RAs'), (105, 'suggest', 'RAs'), (106, 'suggest', 'RAs'), (107, 'suggest', 'RAs'), (108, 'suggest', 'RAs'), (109, 'suggest', 'RAs'), (110, 'suggest', 'RAs'), (111, 'suggest', 'RAs'), (112, 'suggest', 'RAs'), (113, 'suggest', 'RAs'), (114, 'suggest', 'RAs'), (115, 'suggest', 'RAs'), (116, 'suggest', 'RAs'), (117, 'suggest', 'RAs'), (118, 'suggest', 'RAs'), (119, 'suggest', 'RAs'), (120, 'suggest', 'RAs'), (121, 'suggest', 'RAs'), (122, 'suggest', 'RAs'), (123, 'suggest', 'RAs'), (124, 'suggest', 'RAs'), (125, 'suggest', 'RAs'), (126, 'suggest', 'RAs'), (127, 'suggest', 'RAs'), (128, 'speculate', 'RAs'), (129, 'speculate', 'RAs'), (130, 'speculate', 'RAs'), (131, 'shows', 'RAs'), (132, 'shows', 'RAs'), (133, 'shows', 'RAs'), (134, 'shows', 'RAs'), (135, 'shows', 'RAs'), (136, 'shown', 'RAs'), (137, 'shown', 'RAs'), (138, 'shown', 'RAs'), (139, 'shown', 'RAs'), (140, 'showing', 'RAs'), (141, 'showing', 'RAs'), (142, 'showing', 'RAs'), (143, 'showing', 'RAs'), (144, 'showing', 'RAs'), (145, 'showing', 'RAs'), (146, 'showed', 'RAs'), (147, 'showed', 'RAs'), (148, 'showed', 'RAs'), (149, 'showed', 'RAs'), (150, 'showed', 'RAs'), (151, 'showed', 'RAs'), (152, 'showed', 'RAs'), (153, 'showed', 'RAs'), (154, 'showed', 'RAs'), (155, 'showed', 'RAs'), (156, 'showed', 'RAs'), (157, 'showed', 'RAs'), (158, 'showed', 'RAs'), (159, 'showed', 'RAs'), (160, 'showed', 'RAs'), (161, 'showed', 'RAs'), (162, 'showed', 'RAs'), (163, 'showed', 'RAs'), (164, 'showed', 'RAs'), (165, 'showed', 'RAs'), (166, 'showed', 'RAs'), (167, 'showed', 'RAs'), (168, 'showed', 'RAs'), (169, 'show', 'RAs'), (170, 'show', 'RAs'), (171, 'show', 'RAs'), (172, 'show', 'RAs'), (173, 'show', 'RAs'), (174, 'show', 'RAs'), (175, 'show', 'RAs'), (176, 'show', 'RAs'), (177, 'show', 'RAs'), (178, 'show', 'RAs'), (179, 'show', 'RAs'), (180, 'show', 'RAs'), (181, 'show', 'RAs'), (182, 'show', 'RAs'), (183, 'show', 'RAs'), (184, 'show', 'RAs'), (185, 'show', 'RAs'), (186, 'show', 'RAs'), (187, 'show', 'RAs'), (188, 'show', 'RAs'), (189, 'show', 'RAs'), (190, 'show', 'RAs'), (191, 'show', 'RAs'), (192, 'revealing', 'RAs'), (193, 'revealed', 'RAs'), (194, 'revealed', 'RAs'), (195, 'revealed', 'RAs'), (196, 'revealed', 'RAs'), (197, 'revealed', 'RAs'), (198, 'revealed', 'RAs'), (199, 'reveal', 'RAs'), (200, 'requires', 'RAs'), (201, 'requires', 'RAs'), (202, 'requires', 'RAs'), (203, 'report', 'RAs'), (204, 'report', 'RAs'), (205, 'reasoned', 'RAs'), (206, 'reasoned', 'RAs'), (207, 'reasoned', 'RAs'), (208, 'reasoned', 'RAs'), (209, 'rationale', 'RAs'), (210, 'observations', 'RAs'), (211, 'findings', 'RAs'), (212, 'postulated', 'RAs'), (213, 'postulate', 'RAs'), (214, 'possible', 'RAs'), (215, 'possible', 'RAs'), (216, 'possible', 'RAs'), (217, 'possible', 'RAs'), (218, 'possible', 'RAs'), (219, 'possible', 'RAs'), (220, 'possible', 'RAs'), (221, 'possible', 'RAs'), (222, 'possible', 'RAs'), (223, 'possible', 'RAs'), (224, 'possible', 'RAs'), (225, 'possible', 'RAs'), (226, 'possible', 'RAs'), (227, 'possible', 'RAs'), (228, 'possibility', 'RAs'), (229, 'possibility', 'RAs'), (230, 'possibility', 'RAs'), (231, 'possibility', 'RAs'), (232, 'possibility', 'RAs'), (233, 'possibility', 'RAs'), (234, 'possibility', 'RAs'), (235, 'possibility', 'RAs'), (236, 'explanation', 'RAs'), (237, 'possibility', 'RAs'), (238, 'One', 'RAs'), (239, 'interpretation', 'RAs'), (240, 'observed', 'RAs'), (241, 'observed', 'RAs'), (242, 'observations', 'RAs'), (243, 'observations', 'RAs'), (244, 'observation', 'RAs'), (245, 'noteworthy', 'RAs'), (246, 'noted', 'RAs'), (247, 'noted', 'RAs'), (248, 'noted', 'RAs'), (249, 'note', 'RAs'), (250, 'known', 'RAs'), (251, 'evidence', 'RAs'), (252, 'doubt', 'RAs'), (253, 'means', 'RAs'), (254, 'means', 'RAs'), (255, 'likely', 'RAs'), (256, 'likely', 'RAs'), (257, 'likely', 'RAs'), (258, 'likely', 'RAs'), (259, 'likely', 'RAs'), (260, 'likely', 'RAs'), (261, 'likely', 'RAs'), (262, 'likely', 'RAs'), (263, 'likely', 'RAs'), (264, 'possible', 'RAs'), (265, 'possible', 'RAs'), (266, 'possible', 'RAs'), (267, 'interesting', 'RAs'), (268, 'infer', 'RAs'), (269, 'inevitable', 'RAs'), (270, 'indicating', 'RAs'), (271, 'indicating', 'RAs'), (272, 'indicating', 'RAs'), (273, 'indicating', 'RAs'), (274, 'indicating', 'RAs'), (275, 'indicating', 'RAs'), (276, 'indicating', 'RAs'), (277, 'indicating', 'RAs'), (278, 'indicating', 'RAs'), (279, 'indicates', 'RAs'), (280, 'indicates', 'RAs'), (281, 'indicates', 'RAs'), (282, 'indicates', 'RAs'), (283, 'indicates', 'RAs'), (284, 'indicates', 'RAs'), (285, 'indicated', 'RAs'), (286, 'indicated', 'RAs'), (287, 'indicated', 'RAs'), (288, 'indicated', 'RAs'), (289, 'indicated', 'RAs'), (290, 'indicated', 'RAs'), (291, 'indicated', 'RAs'), (292, 'indicate', 'RAs'), (293, 'indicate', 'RAs'), (294, 'indicate', 'RAs'), (295, 'indicate', 'RAs'), (296, 'indicate', 'RAs'), (297, 'indicate', 'RAs'), (298, 'indicate', 'RAs'), (299, 'indicate', 'RAs'), (300, 'indicate', 'RAs'), (301, 'indicate', 'RAs'), (302, 'indicate', 'RAs'), (303, 'indicate', 'RAs'), (304, 'indicate', 'RAs'), (305, 'indicate', 'RAs'), (306, 'indicate', 'RAs'), (307, 'indicate', 'RAs'), (308, 'indicate', 'RAs'), (309, 'indicate', 'RAs'), (310, 'indicate', 'RAs'), (311, 'indicate', 'RAs'), (312, 'indicate', 'RAs'), (313, 'indicate', 'RAs'), (314, 'implying', 'RAs'), (315, 'imply', 'RAs'), (316, 'imply', 'RAs'), (317, 'implies', 'RAs'), (318, 'idea', 'RAs'), (319, 'idea', 'RAs'), (320, 'hypothesized', 'RAs'), (321, 'hypothesized', 'RAs'), (322, 'hypothesized', 'RAs'), (323, 'shown', 'RAs'), (324, 'given', 'RAs'), (325, 'given', 'RAs'), (326, 'given', 'RAs'), (327, 'given', 'RAs'), (328, 'evidence', 'RAs'), (329, 'found', 'RAs'), (330, 'found', 'RAs'), (331, 'found', 'RAs'), (332, 'found', 'RAs'), (333, 'found', 'RAs'), (334, 'found', 'RAs'), (335, 'found', 'RAs'), (336, 'found', 'RAs'), (337, 'found', 'RAs'), (338, 'found', 'RAs'), (339, 'found', 'RAs'), (340, 'found', 'RAs'), (341, 'found', 'RAs'), (342, 'found', 'RAs'), (343, 'found', 'RAs'), (344, 'found', 'RAs'), (345, 'found', 'RAs'), (346, 'found', 'RAs'), (347, 'found', 'RAs'), (348, 'found', 'RAs'), (349, 'found', 'RAs'), (350, 'found', 'RAs'), (351, 'finding', 'RAs'), (352, 'find', 'RAs'), (353, 'feel', 'RAs'), (354, 'fact', 'RAs'), (355, 'extent', 'RAs'), (356, 'expected', 'RAs'), (357, 'evidence', 'RAs'), (358, 'evidence', 'RAs'), (359, 'evidence', 'RAs'), (360, 'evidence', 'RAs'), (361, 'estimated', 'RAs'), (362, 'estimated', 'RAs'), (363, 'estimated', 'RAs'), (364, 'estimate', 'RAs'), (365, 'established', 'RAs'), (366, 'established', 'RAs'), (367, 'emphasize', 'RAs'), (368, 'determined', 'RAs'), (369, 'demonstration', 'RAs'), (370, 'demonstrating', 'RAs'), (371, 'demonstrates', 'RAs'), (372, 'demonstrated', 'RAs'), (373, 'demonstrated', 'RAs'), (374, 'demonstrate', 'RAs'), (375, 'demonstrate', 'RAs'), (376, 'demonstrate', 'RAs'), (377, 'demonstrate', 'RAs'), (378, 'demonstrate', 'RAs'), (379, 'demonstrate', 'RAs'), (380, 'demonstrate', 'RAs'), (381, 'demonstrate', 'RAs'), (382, 'demonstrate', 'RAs'), (383, 'argued', 'RAs'), (384, 'confirming', 'RAs'), (385, 'confirming', 'RAs'), (386, 'confirming', 'RAs'), (387, 'confirming', 'RAs'), (388, 'confirmed', 'RAs'), (389, 'confirmed', 'RAs'), (390, 'confirmed', 'RAs'), (391, 'confirmed', 'RAs'), (392, 'confirmed', 'RAs'), (393, 'confirm', 'RAs'), (394, 'confirm', 'RAs'), (395, 'conclusion', 'RAs'), (396, 'conclude', 'RAs'), (397, 'conclude', 'RAs'), (398, 'conclude', 'RAs'), (399, 'conclude', 'RAs'), (400, 'conclude', 'RAs'), (401, 'conclude', 'RAs'), (402, 'conclude', 'RAs'), (403, 'conclude', 'RAs'), (404, 'believe', 'RAs'), (405, 'believe', 'RAs'), (406, 'believe', 'RAs'), (407, 'believe', 'RAs'), (408, 'believe', 'RAs'), (409, 'appears', 'RAs'), (410, 'appeared', 'RAs'), (411, 'appeared', 'RAs'), (412, 'anticipated', 'RAs'), (413, 'acknowledged', 'RAs'), (414, 'acknowledge', 'RAs'), (415, 'accept', 'RAs'), (416, 'limitation', 'RAs'), (417, 'explanation', 'RAs'), (418, 'finding', 'RAs'), (419, 'decision', 'RAs'), (420, 'well-known', 'RAs'), (421, 'view', 'RAs'), (422, 'observation', 'RAs'), (423, 'fact', 'RAs'), (424, 'fact', 'RAs'), (425, 'reports', 'RAs'), (426, 'possibility', 'RAs'), (427, 'indication', 'RAs'), (428, 'exclude', 'RAs'), (429, 'reported', 'RAs'), (430, 'indicated', 'RAs'), (431, 'observation', 'RAs'), (432, 'observation', 'RAs'), (433, 'suggests', 'RAs'), (434, 'suggesting', 'RAs'), (435, 'suggesting', 'RAs'), (436, 'suggesting', 'RAs'), (437, 'suggesting', 'RAs'), (438, 'suggesting', 'RAs'), (439, 'suggested', 'RAs'), (440, 'suggested', 'RAs'), (441, 'suggested', 'RAs'), (442, 'suggested', 'RAs'), (443, 'suggested', 'RAs'), (444, 'suggested', 'RAs'), (445, 'suggested', 'RAs'), (446, 'suggested', 'RAs'), (447, 'suggested', 'RAs'), (448, 'suggest', 'RAs'), (449, 'suggest', 'RAs'), (450, 'suggest', 'RAs'), (451, 'suggest', 'RAs'), (452, 'shown', 'RAs'), (453, 'shown', 'RAs'), (454, 'shown', 'RAs'), (455, 'shown', 'RAs'), (456, 'shown', 'RAs'), (457, 'shown', 'RAs'), (458, 'shown', 'RAs'), (459, 'shown', 'RAs'), (460, 'shown', 'RAs'), (461, 'shown', 'RAs'), (462, 'shown', 'RAs'), (463, 'shown', 'RAs'), (464, 'shown', 'RAs'), (465, 'showing', 'RAs'), (466, 'showed', 'RAs'), (467, 'showed', 'RAs'), (468, 'showed', 'RAs'), (469, 'showed', 'RAs'), (470, 'showed', 'RAs'), (471, 'show', 'RAs'), (472, 'show', 'RAs'), (473, 'show', 'RAs'), (474, 'revealed', 'RAs'), (475, 'revealed', 'RAs'), (476, 'revealed', 'RAs'), (477, 'reported', 'RAs'), (478, 'reported', 'RAs'), (479, 'reported', 'RAs'), (480, 'reported', 'RAs'), (481, 'reported', 'RAs'), (482, 'reported', 'RAs'), (483, 'evidence', 'RAs'), (484, 'proposed', 'RAs'), (485, 'reports', 'RAs'), (486, 'observations', 'RAs'), (487, 'postulated', 'RAs'), (488, 'observations', 'RAs'), (489, 'observations', 'RAs'), (490, 'observation', 'RAs'), (491, 'notion', 'RAs'), (492, 'noted', 'RAs'), (493, 'noted', 'RAs'), (494, 'thought', 'RAs'), (495, 'increasing', 'RAs'), (496, 'indicates', 'RAs'), (497, 'indicated', 'RAs'), (498, 'indicate', 'RAs'), (499, 'indicate', 'RAs'), (500, 'evidence', 'RAs'), (501, 'hypothesized', 'RAs'), (502, 'found', 'RAs'), (503, 'found', 'RAs'), (504, 'found', 'RAs'), (505, 'found', 'RAs'), (506, 'found', 'RAs'), (507, 'found', 'RAs'), (508, 'found', 'RAs'), (509, 'found', 'RAs'), (510, 'found', 'RAs'), (511, 'found', 'RAs'), (512, 'findings', 'RAs'), (513, 'findings', 'RAs'), (514, 'findings', 'RAs'), (515, 'find', 'RAs'), (516, 'evidence', 'RAs'), (517, 'evidence', 'RAs'), (518, 'established', 'RAs'), (519, 'established', 'RAs'), (520, 'documented', 'RAs'), (521, 'demonstrated', 'RAs'), (522, 'demonstrated', 'RAs'), (523, 'demonstrated', 'RAs'), (524, 'demonstrated', 'RAs'), (525, 'demonstrated', 'RAs'), (526, 'demonstrated', 'RAs'), (527, 'demonstrated', 'RAs'), (528, 'demonstrated', 'RAs'), (529, 'demonstrated', 'RAs'), (530, 'confirmed', 'RAs'), (531, 'concluded', 'RAs'), (532, 'claimed', 'RAs'), (533, 'believed', 'RAs'), (534, 'argued', 'RAs'), (535, 'reports', 'RAs'), (536, 'prove', 'RAs'), (537, 'confirm', 'RAs'), (538, 'show', 'RAs'), (539, 'types', 'RAs'), (540, 'analysis', 'RAs'), (541, 'fact', 'RAs'), (542, 'showing', 'RAs'), (543, 'recognize', 'RAs'), (544, 'reassuring', 'RAs'), (545, 'provided', 'RAs'), (546, 'note', 'RAs'), (547, 'limitation', 'RAs'), (548, 'knowing', 'RAs'), (549, 'expected', 'RAs'), (550, 'indicating', 'RAs'), (551, 'indicates', 'RAs'), (552, 'indicated', 'RAs'), (553, 'included', 'RAs'), (554, 'given', 'RAs'), (555, 'estimated', 'RAs'), (556, 'estimated', 'RAs'), (557, 'established', 'RAs'), (558, 'ensured', 'RAs'), (559, 'ensure', 'RAs'), (560, 'ensure', 'RAs'), (561, 'ensure', 'RAs'), (562, 'effect', 'RAs'), (563, 'dependence', 'RAs'), (564, 'confirm', 'RAs'), (565, 'confirm', 'RAs'), (566, 'condition', 'RAs'), (567, 'assuming', 'RAs'), (568, 'assumed', 'RAs'), (569, 'acknowledge', 'RAs'), (570, 'method', 'RAs'), (571, 'limitation', 'RAs'), (572, 'difference', 'RAs'), (573, 'length', 'RAs'), (574, 'view', 'RAs'), (575, 'theory', 'RAs'), (576, 'notion', 'RAs'), (577, 'notion', 'RAs'), (578, 'idea', 'RAs'), (579, 'hypothesis', 'RAs'), (580, 'suggests', 'RAs'), (581, 'recognises', 'RAs'), (582, 'probability', 'RAs'), (583, 'postulated', 'RAs'), (584, 'postulated', 'RAs'), (585, 'hypothesis', 'RAs'), (586, 'hypothesis', 'RAs'), (587, 'hypothesis', 'RAs'), (588, 'account', 'RAs'), (589, 'account', 'RAs'), (590, 'theory', 'RAs'), (591, 'idea', 'RAs'), (592, 'unlikely', 'RAs'), (593, 'understand', 'RAs'), (594, 'uncovered', 'RAs'), (595, 'time', 'RAs'), (596, 'potential', 'RAs'), (597, 'possibility', 'RAs'), (598, 'finding', 'RAs'), (599, 'fact', 'RAs'), (600, 'fact', 'RAs'), (601, 'plausibility', 'RAs'), (602, 'suggests', 'RAs'), (603, 'suggests', 'RAs'), (604, 'suggests', 'RAs'), (605, 'suggests', 'RAs'), (606, 'suggests', 'RAs'), (607, 'suggesting', 'RAs'), (608, 'suggesting', 'RAs'), (609, 'suggesting', 'RAs'), (610, 'suggesting', 'RAs'), (611, 'suggesting', 'RAs'), (612, 'suggesting', 'RAs'), (613, 'suggesting', 'RAs'), (614, 'suggesting', 'RAs'), (615, 'suggesting', 'RAs'), (616, 'suggesting', 'RAs'), (617, 'suggesting', 'RAs'), (618, 'suggested', 'RAs'), (619, 'suggest', 'RAs'), (620, 'suggest', 'RAs'), (621, 'suggest', 'RAs'), (622, 'suggest', 'RAs'), (623, 'suggest', 'RAs'), (624, 'suggest', 'RAs'), (625, 'suggest', 'RAs'), (626, 'suggest', 'RAs'), (627, 'suggest', 'RAs'), (628, 'suggest', 'RAs'), (629, 'suggest', 'RAs'), (630, 'suggest', 'RAs'), (631, 'suggest', 'RAs'), (632, 'suggest', 'RAs'), (633, 'suggest', 'RAs'), (634, 'suggest', 'RAs'), (635, 'suggest', 'RAs'), (636, 'shows', 'RAs'), (637, 'shown', 'RAs'), (638, 'shown', 'RAs'), (639, 'shown', 'RAs'), (640, 'shown', 'RAs'), (641, 'showing', 'RAs'), (642, 'showing', 'RAs'), (643, 'showing', 'RAs'), (644, 'showing', 'RAs'), (645, 'showing', 'RAs'), (646, 'showing', 'RAs'), (647, 'showing', 'RAs'), (648, 'showing', 'RAs'), (649, 'showed', 'RAs'), (650, 'showed', 'RAs'), (651, 'showed', 'RAs'), (652, 'showed', 'RAs'), (653, 'showed', 'RAs'), (654, 'show', 'RAs'), (655, 'show', 'RAs'), (656, 'show', 'RAs'), (657, 'show', 'RAs'), (658, 'show', 'RAs'), (659, 'show', 'RAs'), (660, 'show', 'RAs'), (661, 'revealed', 'RAs'), (662, 'revealed', 'RAs'), (663, 'revealed', 'RAs'), (664, 'revealed', 'RAs'), (665, 'revealed', 'RAs'), (666, 'reported', 'RAs'), (667, 'possible', 'RAs'), (668, 'possible', 'RAs'), (669, 'possible', 'RAs'), (670, 'possible', 'RAs'), (671, 'possible', 'RAs'), (672, 'possible', 'RAs'), (673, 'possible', 'RAs'), (674, 'possible', 'RAs'), (675, 'observation', 'RAs'), (676, 'hypothesis', 'RAs'), (677, 'observed', 'RAs'), (678, 'observed', 'RAs'), (679, 'observed', 'RAs'), (680, 'observed', 'RAs'), (681, 'observed', 'RAs'), (682, 'observed', 'RAs'), (683, 'observed', 'RAs'), (684, 'observed', 'RAs'), (685, 'observed', 'RAs'), (686, 'observed', 'RAs'), (687, 'observed', 'RAs'), (688, 'observed', 'RAs'), (689, 'observed', 'RAs'), (690, 'observed', 'RAs'), (691, 'noted', 'RAs'), (692, 'note', 'RAs'), (693, 'note', 'RAs'), (694, 'lower', 'RAs'), (695, 'likely', 'RAs'), (696, 'indicating', 'RAs'), (697, 'indicating', 'RAs'), (698, 'indicating', 'RAs'), (699, 'indicates', 'RAs'), (700, 'indicated', 'RAs'), (701, 'indicate', 'RAs'), (702, 'indicate', 'RAs'), (703, 'indicate', 'RAs'), (704, 'indicate', 'RAs'), (705, 'illustrate', 'RAs'), (706, 'illustrate', 'RAs'), (707, 'hypothesized', 'RAs'), (708, 'higher', 'RAs'), (709, 'given', 'RAs'), (710, 'Given', 'RAs'), (711, 'found', 'RAs'), (712, 'found', 'RAs'), (713, 'found', 'RAs'), (714, 'found', 'RAs'), (715, 'found', 'RAs'), (716, 'found', 'RAs'), (717, 'found', 'RAs'), (718, 'found', 'RAs'), (719, 'found', 'RAs'), (720, 'found', 'RAs'), (721, 'found', 'RAs'), (722, 'found', 'RAs'), (723, 'found', 'RAs'), (724, 'found', 'RAs'), (725, 'found', 'RAs'), (726, 'found', 'RAs'), (727, 'found', 'RAs'), (728, 'feasible', 'RAs'), (729, 'Evidence', 'RAs'), (730, 'established', 'RAs'), (731, 'discovered', 'RAs'), (732, 'determined', 'RAs'), (733, 'demonstrating', 'RAs'), (734, 'demonstrated', 'RAs'), (735, 'demonstrated', 'RAs'), (736, 'demonstrate', 'RAs'), (737, 'demonstrate', 'RAs'), (738, 'demonstrate', 'RAs'), (739, 'demonstrate', 'RAs'), (740, 'demonstrate', 'RAs'), (741, 'demonstrate', 'RAs'), (742, 'demonstrate', 'RAs'), (743, 'confirming', 'RAs'), (744, 'confirming', 'RAs'), (745, 'confirmed', 'RAs'), (746, 'confirm', 'RAs'), (747, 'confirm', 'RAs'), (748, 'conclude', 'RAs'), (749, 'conclude', 'RAs'), (750, 'conclude', 'RAs'), (751, 'interpretation', 'RAs'), (752, 'observed', 'RAs'), (753, 'Given', 'RAs'), (754, 'given', 'RAs'), (755, 'evidence', 'RAs'), (756, 'hypothesis', 'RAs'), (757, 'notion', 'RAs'), (758, 'fact', 'RAs'), (759, 'discovery', 'RAs'), (760, 'suggests', 'RAs'), (761, 'suggests', 'RAs'), (762, 'suggests', 'RAs'), (763, 'suggested', 'RAs'), (764, 'evidence', 'RAs'), (765, 'shown', 'RAs'), (766, 'shown', 'RAs'), (767, 'shown', 'RAs'), (768, 'shown', 'RAs'), (769, 'shown', 'RAs'), (770, 'shown', 'RAs'), (771, 'shown', 'RAs'), (772, 'shown', 'RAs'), (773, 'showed', 'RAs'), (774, 'show', 'RAs'), (775, 'show', 'RAs'), (776, 'revealed', 'RAs'), (777, 'revealed', 'RAs'), (778, 'reported', 'RAs'), (779, 'reported', 'RAs'), (780, 'reported', 'RAs'), (781, 'reported', 'RAs'), (782, 'recommending', 'RAs'), (783, 'reported', 'RAs'), (784, 'indicates', 'RAs'), (785, 'indicates', 'RAs'), (786, 'indicates', 'RAs'), (787, 'indicates', 'RAs'), (788, 'indicate', 'RAs'), (789, 'hypothesis', 'RAs'), (790, 'found', 'RAs'), (791, 'demonstrated', 'RAs'), (792, 'demonstrated', 'RAs'), (793, 'confirmed', 'RAs'), (794, 'confirm', 'RAs'), (795, 'awareness', 'RAs'), (796, 'caveat', 'RAs'), (797, 'fact', 'RAs'), (798, 'show', 'RAs'), (799, 'reasoned', 'RAs'), (800, 'posit', 'RAs'), (801, 'hypothesized', 'RAs'), (802, 'hypothesized', 'RAs'), (803, 'hypothesized', 'RAs'), (804, 'hypothesized', 'RAs'), (805, 'hypothesized', 'RAs'), (806, 'hypothesized', 'RAs'), (807, 'hypothesized', 'RAs'), (808, 'envision', 'RAs'), (809, 'believe', 'RAs'), (810, 'anticipated', 'RAs'), (811, 'anticipate', 'RAs'), (812, 'ensure', 'RAs'), (813, 'possibility', 'RAs'), (814, 'suggests', 'RAs'), (815, 'suggests', 'RAs'), (816, 'shown', 'RAs'), (817, 'seems', 'RAs'), (818, 'probability', 'RAs'), (819, 'possible', 'RAs'), (820, 'noting', 'RAs'), (821, 'note', 'RAs'), (822, 'given', 'RAs'), (823, 'exclude', 'RAs'), (824, 'assumption', 'RAs'), (825, 'assumption', 'RAs'), (826, 'assumption', 'RAs'), (827, 'assumed', 'RAs'), (828, 'acknowledge', 'RAs'), (829, 'limitation', 'RAs'), (830, 'hypothesis', 'RAs'), (831, 'suggesting', 'RAs'), (832, 'possibility', 'RAs'), (833, 'hypothesis ', 'RAs'), (834, 'What', 'Theses'), (835, 'unlikely', 'Theses'), (836, 'unlikely', 'Theses'), (837, 'unlikely', 'Theses'), (838, 'speculation', 'Theses'), (839, 'result', 'Theses'), (840, 'question', 'Theses'), (841, 'problem', 'Theses'), (842, 'possibility', 'Theses'), (843, 'observations', 'Theses'), (844, 'observation', 'Theses'), (845, 'indication', 'Theses'), (846, 'evidence', 'Theses'), (847, 'evidence', 'Theses'), (848, 'findings', 'Theses'), (849, 'fact', 'Theses'), (850, 'fact', 'Theses'), (851, 'fact', 'Theses'), (852, 'fact', 'Theses'), (853, 'fact', 'Theses'), (854, 'fact', 'Theses'), (855, 'expectation', 'Theses'), (856, 'observation', 'Theses'), (857, 'discordance', 'Theses'), (858, 'observation', 'Theses'), (859, 'evidence', 'Theses'), (860, 'conclusion', 'Theses'), (861, 'surprising', 'Theses'), (862, 'surprising', 'Theses'), (863, 'surprising', 'Theses'), (864, 'surprising', 'Theses'), (865, 'suggests', 'Theses'), (866, 'suggests', 'Theses'), (867, 'suggests', 'Theses'), (868, 'suggests', 'Theses'), (869, 'suggests', 'Theses'), (870, 'suggests', 'Theses'), (871, 'suggests', 'Theses'), (872, 'suggests', 'Theses'), (873, 'suggests', 'Theses'), (874, 'suggests', 'Theses'), (875, 'suggests', 'Theses'), (876, 'suggests', 'Theses'), (877, 'suggests', 'Theses'), (878, 'suggests', 'Theses'), (879, 'suggests', 'Theses'), (880, 'suggests', 'Theses'), (881, 'suggests', 'Theses'), (882, 'suggests', 'Theses'), (883, 'suggests', 'Theses'), (884, 'suggests', 'Theses'), (885, 'suggests', 'Theses'), (886, 'suggests', 'Theses'), (887, 'suggests', 'Theses'), (888, 'suggests', 'Theses'), (889, 'suggests', 'Theses'), (890, 'suggests', 'Theses'), (891, 'suggests', 'Theses'), (892, 'suggests', 'Theses'), (893, 'suggests', 'Theses'), (894, 'suggests', 'Theses'), (895, 'suggests', 'Theses'), (896, 'suggesting', 'Theses'), (897, 'suggesting', 'Theses'), (898, 'suggesting', 'Theses'), (899, 'suggesting', 'Theses'), (900, 'suggesting', 'Theses'), (901, 'suggesting', 'Theses'), (902, 'suggesting', 'Theses'), (903, 'suggesting', 'Theses'), (904, 'suggesting', 'Theses'), (905, 'suggesting', 'Theses'), (906, 'suggesting', 'Theses'), (907, 'suggesting', 'Theses'), (908, 'suggesting', 'Theses'), (909, 'suggesting', 'Theses'), (910, 'suggesting', 'Theses'), (911, 'suggested', 'Theses'), (912, 'suggested', 'Theses'), (913, 'suggested', 'Theses'), (914, 'suggested', 'Theses'), (915, 'suggested', 'Theses'), (916, 'suggested', 'Theses'), (917, 'suggested', 'Theses'), (918, 'suggested', 'Theses'), (919, 'suggested', 'Theses'), (920, 'suggest', 'Theses'), (921, 'suggest', 'Theses'), (922, 'suggest', 'Theses'), (923, 'suggest', 'Theses'), (924, 'suggest', 'Theses'), (925, 'suggest', 'Theses'), (926, 'suggest', 'Theses'), (927, 'suggest', 'Theses'), (928, 'suggest', 'Theses'), (929, 'suggest', 'Theses'), (930, 'suggest', 'Theses'), (931, 'suggest', 'Theses'), (932, 'suggest', 'Theses'), (933, 'suggest', 'Theses'), (934, 'suggest', 'Theses'), (935, 'suggest', 'Theses'), (936, 'suggest', 'Theses'), (937, 'suggest', 'Theses'), (938, 'suggest', 'Theses'), (939, 'suggest', 'Theses'), (940, 'suggest', 'Theses'), (941, 'suggest', 'Theses'), (942, 'suggest', 'Theses'), (943, 'suggest', 'Theses'), (944, 'suggest', 'Theses'), (945, 'suggest', 'Theses'), (946, 'suggest', 'Theses'), (947, 'striking', 'Theses'), (948, 'striking', 'Theses'), (949, 'speculated', 'Theses'), (950, 'speculated', 'Theses'), (951, 'speculated', 'Theses'), (952, 'speculate', 'Theses'), (953, 'signifying', 'Theses'), (954, 'shows', 'Theses'), (955, 'shows', 'Theses'), (956, 'shows', 'Theses'), (957, 'shows', 'Theses'), (958, 'shows', 'Theses'), (959, 'shows', 'Theses'), (960, 'shown', 'Theses'), (961, 'shown', 'Theses'), (962, 'shown', 'Theses'), (963, 'shown', 'Theses'), (964, 'shown', 'Theses'), (965, 'shown', 'Theses'), (966, 'shown', 'Theses'), (967, 'shown', 'Theses'), (968, 'shown', 'Theses'), (969, 'showing', 'Theses'), (970, 'showed', 'Theses'), (971, 'showed', 'Theses'), (972, 'showed', 'Theses'), (973, 'showed', 'Theses'), (974, 'showed', 'Theses'), (975, 'showed', 'Theses'), (976, 'showed', 'Theses'), (977, 'showed', 'Theses'), (978, 'showed', 'Theses'), (979, 'showed', 'Theses'), (980, 'showed', 'Theses'), (981, 'showed', 'Theses'), (982, 'showed', 'Theses'), (983, 'showed', 'Theses'), (984, 'showed', 'Theses'), (985, 'showed', 'Theses'), (986, 'showed', 'Theses'), (987, 'showed', 'Theses'), (988, 'showed', 'Theses'), (989, 'showed', 'Theses'), (990, 'showed', 'Theses'), (991, 'showed', 'Theses'), (992, 'show', 'Theses'), (993, 'show', 'Theses'), (994, 'show', 'Theses'), (995, 'show', 'Theses'), (996, 'seem', 'Theses'), (997, 'revealed', 'Theses'), (998, 'revealed', 'Theses'), (999, 'revealed', 'Theses'), (1000, 'report', 'Theses'), (1001, 'recognized', 'Theses'), (1002, 'proposed', 'Theses'), (1003, 'predicting', 'Theses'), (1004, 'possible', 'Theses'), (1005, 'possible', 'Theses'), (1006, 'possible', 'Theses'), (1007, 'possible', 'Theses'), (1008, 'possible', 'Theses'), (1009, 'possible', 'Theses'), (1010, 'possible', 'Theses'), (1011, 'possible', 'Theses'), (1012, 'plausible', 'Theses'), (1013, 'plausible', 'Theses'), (1014, 'observation', 'Theses'), (1015, 'observed', 'Theses'), (1016, 'observed', 'Theses'), (1017, 'observed', 'Theses'), (1018, 'observed', 'Theses'), (1019, 'noting', 'Theses'), (1020, 'noticeable', 'Theses'), (1021, 'noteworthy', 'Theses'), (1022, 'noteworthy', 'Theses'), (1023, 'noted', 'Theses'), (1024, 'noted', 'Theses'), (1025, 'noted', 'Theses'), (1026, 'Note', 'Theses'), (1027, 'note', 'Theses'), (1028, 'note', 'Theses'), (1029, 'note', 'Theses'), (1030, 'note', 'Theses'), (1031, 'means', 'Theses'), (1032, 'meaning', 'Theses'), (1033, 'meaning', 'Theses'), (1034, 'meaning', 'Theses'), (1035, 'mean', 'Theses'), (1036, 'mean', 'Theses'), (1037, 'likely', 'Theses'), (1038, 'likely', 'Theses'), (1039, 'likely', 'Theses'), (1040, 'likely', 'Theses'), (1041, 'likely', 'Theses'), (1042, 'likely', 'Theses'), (1043, 'suggested', 'Theses'), (1044, 'interesting', 'Theses'), (1045, 'interesting', 'Theses'), (1046, 'indicating', 'Theses'), (1047, 'indicating', 'Theses'), (1048, 'indicating', 'Theses'), (1049, 'indicating', 'Theses'), (1050, 'indicating', 'Theses'), (1051, 'indicating', 'Theses'), (1052, 'indicating', 'Theses'), (1053, 'indicating', 'Theses'), (1054, 'indicating', 'Theses'), (1055, 'indicating', 'Theses'), (1056, 'indicating', 'Theses'), (1057, 'indicating', 'Theses'), (1058, 'indicating', 'Theses'), (1059, 'indicating', 'Theses'), (1060, 'indicating', 'Theses'), (1061, 'indicates', 'Theses'), (1062, 'indicates', 'Theses'), (1063, 'indicates', 'Theses'), (1064, 'indicates', 'Theses'), (1065, 'indicates', 'Theses'), (1066, 'indicates', 'Theses'), (1067, 'indicates', 'Theses'), (1068, 'indicates', 'Theses'), (1069, 'indicates', 'Theses'), (1070, 'indicated', 'Theses'), (1071, 'indicated', 'Theses'), (1072, 'indicated', 'Theses'), (1073, 'indicated', 'Theses'), (1074, 'indicated', 'Theses'), (1075, 'indicated', 'Theses'), (1076, 'indicated', 'Theses'), (1077, 'indicate', 'Theses'), (1078, 'indicate', 'Theses'), (1079, 'indicate', 'Theses'), (1080, 'indicate', 'Theses'), (1081, 'indicate', 'Theses'), (1082, 'indicate', 'Theses'), (1083, 'indicate', 'Theses'), (1084, 'indicate', 'Theses'), (1085, 'indicate', 'Theses'), (1086, 'implying', 'Theses'), (1087, 'imply', 'Theses'), (1088, 'Given', 'Theses'), (1089, 'Given', 'Theses'), (1090, 'Given', 'Theses'), (1091, 'Given', 'Theses'), (1092, 'Given', 'Theses'), (1093, 'given', 'Theses'), (1094, 'given', 'Theses'), (1095, 'found', 'Theses'), (1096, 'found', 'Theses'), (1097, 'found', 'Theses'), (1098, 'found', 'Theses'), (1099, 'found', 'Theses'), (1100, 'found', 'Theses'), (1101, 'found', 'Theses'), (1102, 'found', 'Theses'), (1103, 'found', 'Theses'), (1104, 'found', 'Theses'), (1105, 'found', 'Theses'), (1106, 'found', 'Theses'), (1107, 'found', 'Theses'), (1108, 'found', 'Theses'), (1109, 'found', 'Theses'), (1110, 'evidence', 'Theses'), (1111, 'established', 'Theses'), (1112, 'established', 'Theses'), (1113, 'ensure', 'Theses'), (1114, 'doubt', 'Theses'), (1115, 'discovered', 'Theses'), (1116, 'determining', 'Theses'), (1117, 'demonstrating', 'Theses'), (1118, 'demonstrating', 'Theses'), (1119, 'demonstrating', 'Theses'), (1120, 'demonstrating', 'Theses'), (1121, 'demonstrating', 'Theses'), (1122, 'demonstrates', 'Theses'), (1123, 'demonstrates', 'Theses'), (1124, 'demonstrates', 'Theses'), (1125, 'demonstrates', 'Theses'), (1126, 'demonstrates', 'Theses'), (1127, 'demonstrated', 'Theses'), (1128, 'demonstrated', 'Theses'), (1129, 'demonstrated', 'Theses'), (1130, 'demonstrated', 'Theses'), (1131, 'demonstrated', 'Theses'), (1132, 'demonstrated', 'Theses'), (1133, 'demonstrated', 'Theses'), (1134, 'demonstrated', 'Theses'), (1135, 'demonstrated', 'Theses'), (1136, 'demonstrated', 'Theses'), (1137, 'demonstrated', 'Theses'), (1138, 'demonstrated', 'Theses'), (1139, 'demonstrated', 'Theses'), (1140, 'demonstrated', 'Theses'), (1141, 'demonstrated', 'Theses'), (1142, 'demonstrated', 'Theses'), (1143, 'demonstrated', 'Theses'), (1144, 'demonstrated', 'Theses'), (1145, 'demonstrated', 'Theses'), (1146, 'demonstrated', 'Theses'), (1147, 'demonstrated', 'Theses'), (1148, 'demonstrated', 'Theses'), (1149, 'demonstrated', 'Theses'), (1150, 'demonstrated', 'Theses'), (1151, 'demonstrated', 'Theses'), (1152, 'demonstrated', 'Theses'), (1153, 'demonstrated', 'Theses'), (1154, 'demonstrated', 'Theses'), (1155, 'demonstrated', 'Theses'), (1156, 'demonstrated', 'Theses'), (1157, 'demonstrated', 'Theses'), (1158, 'demonstrated', 'Theses'), (1159, 'demonstrated', 'Theses'), (1160, 'demonstrated', 'Theses'), (1161, 'demonstrated', 'Theses'), (1162, 'demonstrated', 'Theses'), (1163, 'demonstrated', 'Theses'), (1164, 'demonstrated', 'Theses'), (1165, 'demonstrated', 'Theses'), (1166, 'demonstrated', 'Theses'), (1167, 'demonstrated', 'Theses'), (1168, 'demonstrate', 'Theses'), (1169, 'demonstrate', 'Theses'), (1170, 'demonstrate', 'Theses'), (1171, 'demonstrate', 'Theses'), (1172, 'demonstrate', 'Theses'), (1173, 'corroborated', 'Theses'), (1174, 'Considering', 'Theses'), (1175, 'Considering', 'Theses'), (1176, 'confirms', 'Theses'), (1177, 'confirms', 'Theses'), (1178, 'confirming', 'Theses'), (1179, 'confirming', 'Theses'), (1180, 'confirmed', 'Theses'), (1181, 'confirmed', 'Theses'), (1182, 'confirmed', 'Theses'), (1183, 'confirmed', 'Theses'), (1184, 'concluded', 'Theses'), (1185, 'conclude', 'Theses'), (1186, 'conclude', 'Theses'), (1187, 'concerns', 'Theses'), (1188, 'clear', 'Theses'), (1189, 'assuming', 'Theses'), (1190, 'appears', 'Theses'), (1191, 'appears', 'Theses'), (1192, 'appears', 'Theses'), (1193, 'appears', 'Theses'), (1194, 'appeared', 'Theses'), (1195, 'apparent', 'Theses'), (1196, 'observation', 'Theses'), (1197, 'agrees', 'Theses'), (1198, 'agreed', 'Theses'), (1199, 'affirming', 'Theses'), (1200, 'thought', 'Theses'), (1201, 'thought', 'Theses'), (1202, 'thought', 'Theses'), (1203, 'thought', 'Theses'), (1204, 'thought', 'Theses'), (1205, 'thought', 'Theses'), (1206, 'thought', 'Theses'), (1207, 'thought', 'Theses'), (1208, 'thought', 'Theses'), (1209, 'thought', 'Theses'), (1210, 'thickened', 'Theses'), (1211, 'findings', 'Theses'), (1212, 'fact', 'Theses'), (1213, 'fact', 'Theses'), (1214, 'fact', 'Theses'), (1215, 'fact', 'Theses'), (1216, 'demonstration', 'Theses'), (1217, 'evidence', 'Theses'), (1218, 'surprising', 'Theses'), (1219, 'surprising', 'Theses'), (1220, 'surprise', 'Theses'), (1221, 'supported', 'Theses'), (1222, 'suggests', 'Theses'), (1223, 'suggests', 'Theses'), (1224, 'suggests', 'Theses'), (1225, 'suggests', 'Theses'), (1226, 'suggests', 'Theses'), (1227, 'suggests', 'Theses'), (1228, 'suggests', 'Theses'), (1229, 'suggests', 'Theses'), (1230, 'suggests', 'Theses'), (1231, 'suggests', 'Theses'), (1232, 'suggesting', 'Theses'), (1233, 'suggesting', 'Theses'), (1234, 'suggesting', 'Theses'), (1235, 'suggesting', 'Theses'), (1236, 'suggesting', 'Theses'), (1237, 'suggesting', 'Theses'), (1238, 'suggesting', 'Theses'), (1239, 'suggested', 'Theses'), (1240, 'suggested', 'Theses'), (1241, 'suggested', 'Theses'), (1242, 'suggested', 'Theses'), (1243, 'suggested', 'Theses'), (1244, 'suggested', 'Theses'), (1245, 'suggested', 'Theses'), (1246, 'suggested', 'Theses'), (1247, 'suggested', 'Theses'), (1248, 'suggested', 'Theses'), (1249, 'suggested', 'Theses'), (1250, 'suggested', 'Theses'), (1251, 'suggested', 'Theses'), (1252, 'suggest', 'Theses'), (1253, 'suggest', 'Theses'), (1254, 'suggest', 'Theses'), (1255, 'suggest', 'Theses'), (1256, 'suggest', 'Theses'), (1257, 'suggest', 'Theses'), (1258, 'submit', 'Theses'), (1259, 'stating', 'Theses'), (1260, 'shows', 'Theses'), (1261, 'shown', 'Theses'), (1262, 'shown', 'Theses'), (1263, 'shown', 'Theses'), (1264, 'shown', 'Theses'), (1265, 'shown', 'Theses'), (1266, 'shown', 'Theses'), (1267, 'shown', 'Theses'), (1268, 'shown', 'Theses'), (1269, 'shown', 'Theses'), (1270, 'shown', 'Theses'), (1271, 'shown', 'Theses'), (1272, 'shown', 'Theses'), (1273, 'shown', 'Theses'), (1274, 'shown', 'Theses'), (1275, 'shown', 'Theses'), (1276, 'shown', 'Theses'), (1277, 'showing', 'Theses'), (1278, 'showed', 'Theses'), (1279, 'showed', 'Theses'), (1280, 'showed', 'Theses'), (1281, 'showed', 'Theses'), (1282, 'showed', 'Theses'), (1283, 'showed', 'Theses'), (1284, 'showed', 'Theses'), (1285, 'showed', 'Theses'), (1286, 'showed', 'Theses'), (1287, 'showed', 'Theses'), (1288, 'showed', 'Theses'), (1289, 'showed', 'Theses'), (1290, 'showed', 'Theses'), (1291, 'showed', 'Theses'), (1292, 'showed', 'Theses'), (1293, 'showed', 'Theses'), (1294, 'seen', 'Theses'), (1295, 'seems', 'Theses'), (1296, 'revealed', 'Theses'), (1297, 'reveal', 'Theses'), (1298, 'reported', 'Theses'), (1299, 'reported', 'Theses'), (1300, 'reported', 'Theses'), (1301, 'reported', 'Theses'), (1302, 'reported', 'Theses'), (1303, 'reported', 'Theses'), (1304, 'reported', 'Theses'), (1305, 'reported', 'Theses'), (1306, 'reported', 'Theses'), (1307, 'reported', 'Theses'), (1308, 'reported', 'Theses'), (1309, 'reported', 'Theses'), (1310, 'reported', 'Theses'), (1311, 'data', 'Theses'), (1312, 'recommended', 'Theses'), (1313, 'recognised', 'Theses'), (1314, 'recognised', 'Theses'), (1315, 'recognised', 'Theses'), (1316, 'recognised', 'Theses'), (1317, 'proposed', 'Theses'), (1318, 'propose', 'Theses'), (1319, 'propose', 'Theses'), (1320, 'projected', 'Theses'), (1321, 'postulated', 'Theses'), (1322, 'plausible', 'Theses'), (1323, 'understanding', 'Theses'), (1324, 'observed', 'Theses'), (1325, 'mentioning', 'Theses'), (1326, 'means', 'Theses'), (1327, 'likely', 'Theses'), (1328, 'likely', 'Theses'), (1329, 'likely', 'Theses'), (1330, 'likely', 'Theses'), (1331, 'likely', 'Theses'), (1332, 'known', 'Theses'), (1333, 'known', 'Theses'), (1334, 'indicating', 'Theses'), (1335, 'indicates', 'Theses'), (1336, 'indicates', 'Theses'), (1337, 'indicates', 'Theses'), (1338, 'indicated', 'Theses'), (1339, 'indicated', 'Theses'), (1340, 'indicated', 'Theses'), (1341, 'indicate', 'Theses'), (1342, 'implying', 'Theses'), (1343, 'implying', 'Theses'), (1344, 'implicating', 'Theses'), (1345, 'highlight', 'Theses'), (1346, 'shown', 'Theses'), (1347, 'shown', 'Theses'), (1348, 'shown', 'Theses'), (1349, 'Given', 'Theses'), (1350, 'found', 'Theses'), (1351, 'found', 'Theses'), (1352, 'found', 'Theses'), (1353, 'found', 'Theses'), (1354, 'found', 'Theses'), (1355, 'found', 'Theses'), (1356, 'found', 'Theses'), (1357, 'evidence', 'Theses'), (1358, 'evidence', 'Theses'), (1359, 'evidence', 'Theses'), (1360, 'evidence', 'Theses'), (1361, 'evidence', 'Theses'), (1362, 'estimated', 'Theses'), (1363, 'estimated', 'Theses'), (1364, 'estimated', 'Theses'), (1365, 'established', 'Theses'), (1366, 'documented', 'Theses'), (1367, 'documented', 'Theses'), (1368, 'discovered', 'Theses'), (1369, 'described', 'Theses'), (1370, 'demonstrated', 'Theses'), (1371, 'demonstrated', 'Theses'), (1372, 'demonstrated', 'Theses'), (1373, 'demonstrated', 'Theses'), (1374, 'demonstrated', 'Theses'), (1375, 'demonstrated', 'Theses'), (1376, 'demonstrated', 'Theses'), (1377, 'demonstrated', 'Theses'), (1378, 'demonstrated', 'Theses'), (1379, 'demonstrated', 'Theses'), (1380, 'demonstrated', 'Theses'), (1381, 'demonstrated', 'Theses'), (1382, 'demonstrated', 'Theses'), (1383, 'demonstrated', 'Theses'), (1384, 'demonstrated', 'Theses'), (1385, 'demonstrated', 'Theses'), (1386, 'demonstrated', 'Theses'), (1387, 'demonstrated', 'Theses'), (1388, 'demonstrated', 'Theses'), (1389, 'demonstrated', 'Theses'), (1390, 'demonstrated', 'Theses'), (1391, 'demonstrate', 'Theses'), (1392, 'demonstrate', 'Theses'), (1393, 'credible', 'Theses'), (1394, 'consistent', 'Theses'), (1395, 'Considering', 'Theses'), (1396, 'Considering', 'Theses'), (1397, 'considering', 'Theses'), (1398, 'considering', 'Theses'), (1399, 'confirmed', 'Theses'), (1400, 'confirmed', 'Theses'), (1401, 'concluded', 'Theses'), (1402, 'concluded', 'Theses'), (1403, 'believed', 'Theses'), (1404, 'believed', 'Theses'), (1405, 'believed', 'Theses'), (1406, 'believed', 'Theses'), (1407, 'believed', 'Theses'), (1408, 'believe', 'Theses'), (1409, 'recognition', 'Theses'), (1410, 'acknowledging', 'Theses'), (1411, 'accepted', 'Theses'), (1412, 'accepted', 'Theses'), (1413, 'range', 'Theses'), (1414, 'hope', 'Theses'), (1415, 'hoped', 'Theses'), (1416, 'hoped', 'Theses'), (1417, 'hoped', 'Theses'), (1418, 'hoped', 'Theses'), (1419, 'hoped', 'Theses'), (1420, 'hoped', 'Theses'), (1421, 'hoped', 'Theses'), (1422, 'hoped', 'Theses'), (1423, 'hoped', 'Theses'), (1424, 'hoped', 'Theses'), (1425, 'expect', 'Theses'), (1426, 'confirm', 'Theses'), (1427, 'ensure', 'Theses'), (1428, 'affirm', 'Theses'), (1429, 'rationale', 'Theses'), (1430, 'probability', 'Theses'), (1431, 'observation', 'Theses'), (1432, 'fact', 'Theses'), (1433, 'difference', 'Theses'), (1434, 'difference', 'Theses'), (1435, 'suggests', 'Theses'), (1436, 'suggests', 'Theses'), (1437, 'showed', 'Theses'), (1438, 'recognize', 'Theses'), (1439, 'probability', 'Theses'), (1440, 'application', 'Theses'), (1441, 'noteworthy', 'Theses'), (1442, 'noteworthy', 'Theses'), (1443, 'noted', 'Theses'), (1444, 'noted', 'Theses'), (1445, 'doubt', 'Theses'), (1446, 'method', 'Theses'), (1447, 'insist', 'Theses'), (1448, 'found', 'Theses'), (1449, 'ensure', 'Theses'), (1450, 'ensure', 'Theses'), (1451, 'ensure', 'Theses'), (1452, 'ensure', 'Theses'), (1453, 'ensure    ', 'Theses'), (1454, 'ensure', 'Theses'), (1455, 'demonstrated', 'Theses'), (1456, 'corroborated', 'Theses'), (1457, 'concern', 'Theses'), (1458, 'check', 'Theses'), (1459, 'assumes', 'Theses'), (1460, 'assumes', 'Theses'), (1461, 'ascertained', 'Theses'), (1462, 'reason', 'Theses'), (1463, 'unlikely', 'Theses'), (1464, 'thought', 'Theses'), (1465, 'thought', 'Theses'), (1466, 'advantage', 'Theses'), (1467, 'idea', 'Theses'), (1468, 'hypothesis', 'Theses'), (1469, 'hypothesis', 'Theses'), (1470, 'hypothesis', 'Theses'), (1471, 'hypothesis', 'Theses'), (1472, 'hypothesis', 'Theses'), (1473, 'hypothesis', 'Theses'), (1474, 'fact', 'Theses'), (1475, 'hypothesis', 'Theses'), (1476, 'suggesting', 'Theses'), (1477, 'suggest', 'Theses'), (1478, 'proposed', 'Theses'), (1479, 'presumed', 'Theses'), (1480, 'postulate', 'Theses'), (1481, 'possible', 'Theses'), (1482, 'possible', 'Theses'), (1483, 'possibility', 'Theses'), (1484, 'model', 'Theses'), (1485, 'likely', 'Theses'), (1486, 'likely', 'Theses'), (1487, 'likely', 'Theses'), (1488, 'concern', 'Theses'), (1489, 'implies', 'Theses'), (1490, 'hypothesized', 'Theses'), (1491, 'hypothesized', 'Theses'), (1492, 'hypothesize', 'Theses'), (1493, 'hypothesize', 'Theses'), (1494, 'hypothesised', 'Theses'), (1495, 'hypothesised', 'Theses'), (1496, 'Hypothesis', 'Theses'), (1497, 'hypothesis', 'Theses'), (1498, 'hypothesis', 'Theses'), (1499, 'hypothesis', 'Theses'), (1500, 'hypothesis', 'Theses'), (1501, 'hoped', 'Theses'), (1502, 'hoped', 'Theses'), (1503, 'hoped', 'Theses'), (1504, 'hoped', 'Theses'), (1505, 'expected', 'Theses'), (1506, 'expected', 'Theses'), (1507, 'expected', 'Theses'), (1508, 'demonstrating', 'Theses'), (1509, 'Assuming', 'Theses'), (1510, 'anticipated', 'Theses'), (1511, 'theory', 'Theses'), (1512, 'hypothesise', 'Theses')]
        df = pd.DataFrame.from_records(records, columns=['index', 'Text', 'Genre']).set_index('index')
        c = TermDocMatrixFromPandas(df, category_col='Genre', text_col='Text', nlp=whitespace_nlp).build()
        c.get_term_freq_df()
        c = CorpusFromPandas(df, category_col='Genre', text_col='Text', nlp=whitespace_nlp).build()
        df = c.get_term_freq_df()

class TestSemioticSquare(TestCase):

    def test_constructor(self):
        df = pd.DataFrame(data=np.array(get_docs_categories_semiotic()).T, columns=['category', 'text'])
        corpus = CorpusFromPandas(df, 'category', 'text', nlp=whitespace_nlp).build()
        SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', ['swift'])
        with self.assertRaises(AssertionError):
            SemioticSquare(corpus, 'XXXhamlet', 'jay-z/r. kelly', ['swift'])
        with self.assertRaises(AssertionError):
            SemioticSquare(corpus, 'hamlet', 'jay-z/r. kellyXXX', ['swift'])
        with self.assertRaises(AssertionError):
            SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', ['swift', 'asd'])
        with self.assertRaises(EmptyNeutralCategoriesError):
            SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', [])

    def test_get_labels(self):
        corpus = get_test_corpus()
        semsq = SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', ['swift'])
        a, b = ('hamlet', 'jay-z/r. kelly')
        default_labels = {'a': a, 'not_a': 'Not ' + a, 'b': b, 'not_b': 'Not ' + b, 'a_and_b': a + ' + ' + b, 'not_a_and_not_b': 'Not ' + a + ' + Not ' + b, 'a_and_not_b': a + ' + Not ' + b, 'b_and_not_a': 'Not ' + a + ' + ' + b}
        labels = semsq.get_labels()
        for name, default_label in default_labels.items():
            self.assertTrue(name + '_label' in labels)
            self.assertEqual(labels[name + '_label'], default_label)
        semsq = SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', ['swift'], labels={'a': 'AAA'})
        labels = semsq.get_labels()
        for name, default_label in default_labels.items():
            if name == 'a':
                self.assertEqual(labels[name + '_label'], 'AAA')
            else:
                self.assertTrue(name + '_label' in labels)
                self.assertEqual(labels[name + '_label'], default_label)

    def test_get_lexicons(self):
        semsq = get_test_semiotic_square()
        lexicons = semsq.get_lexicons()
        for category in self.categories():
            self.assertIn(category, lexicons)
            self.assertLessEqual(len(lexicons[category]), 10)
        lexicons = semsq.get_lexicons(5)
        for category in self.categories():
            self.assertIn(category, lexicons)
            self.assertLessEqual(len(lexicons[category]), 5)

    def test_get_axes(self):
        semsq = get_test_semiotic_square()
        ax = semsq.get_axes()
        self.assertEqual(list(sorted(ax.index)), list(sorted(semsq.term_doc_matrix_.get_terms())))

    def categories(self):
        return ['a', 'b', 'not_a', 'not_b', 'a_and_not_b', 'b_and_not_a', 'a_and_b', 'not_a_and_not_b']

class TestLogOddsUninformativePriorScore(TestCase):

    def test_get_score(self):
        cat_counts, not_cat_counts = self._get_counts()
        scores = LogOddsUninformativePriorScore.get_score(cat_counts, not_cat_counts)
        np.testing.assert_almost_equal(scores, np.array([0.4447054, 0.9433088, 0.4447054, -0.9971462]))
    '\n\tdef test_get_delta_hats(self):\n\t\tcat_counts, not_cat_counts = self._get_counts()\n\t\tscores = LogOddsUninformativePriorScore.get_delta_hats(cat_counts, not_cat_counts)\n\t\tnp.testing.assert_almost_equal(scores,\n\t\t                               np.array([-0.6095321, -1.0345766, -0.6095321,  1.5201005]))\n\t'

    def test_get_score_threshold(self):
        cat_counts = np.array([1, 5, 2, 7, 10])
        not_cat_counts = np.array([10, 10, 1, 5, 10])
        scores = LogOddsUninformativePriorScore.get_thresholded_score(cat_counts, not_cat_counts, alpha_w=0.01, threshold=0.1)
        np.testing.assert_almost_equal(scores, np.array([-0.9593012, -0.0, 0.0, 0.8197493, 0.0]))

    def test__turn_pvals_into_scores(self):
        p_vals = np.array([0.01, 0.99, 0.5, 0.1, 0.9])
        scores = LogOddsUninformativePriorScore._turn_pvals_into_scores(p_vals)
        np.testing.assert_almost_equal(scores, [0.98, -0.98, -0.0, 0.8, -0.8])

    def test__turn_counts_into_matrix(self):
        cat_counts, not_cat_counts = self._get_counts()
        X = LogOddsUninformativePriorScore._turn_counts_into_matrix(cat_counts, not_cat_counts)
        np.testing.assert_almost_equal(X, np.array([[1, 100], [5, 510], [1, 100], [9, 199]]))

    def _get_counts(self):
        cat_counts = np.array([1, 5, 1, 9])
        not_cat_counts = np.array([100, 510, 100, 199])
        return (cat_counts, not_cat_counts)

class TestFourSquareAxes(TestCase):

    def test_build(self):
        corpus = self._get_test_corpus()
        with self.assertRaises(AssertionError):
            fs = FourSquareAxes(corpus, 'hamlet', ['jay-z/r. kelly'], ['swift'], ['dylan'])
        with self.assertRaises(AssertionError):
            fs = FourSquareAxes(corpus, ['hamlet'], 'jay-z/r. kelly', ['swift'], ['dylan'])
        with self.assertRaises(AssertionError):
            fs = FourSquareAxes(corpus, ['hamlet'], ['jay-z/r. kelly'], 'swift', ['dylan'])
        with self.assertRaises(AssertionError):
            fs = FourSquareAxes(corpus, ['hamlet'], ['jay-z/r. kelly'], ['swift'], 'dylan')
        fs = FourSquareAxes(corpus, ['hamlet'], ['jay-z/r. kelly'], ['swift'], ['dylan'])
        self.assertEqual(fs.get_labels(), {'a_and_b_label': 'swift', 'a_and_not_b_label': 'hamlet', 'a_label': '', 'b_and_not_a_label': 'jay-z/r. kelly', 'b_label': '', 'not_a_and_not_b_label': 'dylan', 'not_a_label': '', 'not_b_label': ''})
        fs = FourSquareAxes(corpus, ['hamlet'], ['jay-z/r. kelly'], ['swift'], ['dylan'], labels={'a': 'swiftham', 'b': 'swiftj'})
        self.assertEqual(fs.get_labels(), {'a_and_b_label': 'swift', 'a_and_not_b_label': 'hamlet', 'a_label': 'swiftham', 'b_and_not_a_label': 'jay-z/r. kelly', 'b_label': 'swiftj', 'not_a_and_not_b_label': 'dylan', 'not_a_label': '', 'not_b_label': ''})
        axes = fs.get_axes()
        self.assertEqual(len(axes), len(corpus.get_terms()))
        self.assertEqual(set(axes.columns), {'x', 'y', 'counts'})
        fs.lexicons

    def _get_test_corpus(self):
        cats, docs = get_docs_categories_four()
        df = pd.DataFrame({'category': cats, 'text': docs})
        corpus = CorpusFromPandas(df, 'category', 'text', nlp=whitespace_nlp).build()
        return corpus

    def _get_test_semiotic_square(self):
        corpus = self._get_test_corpus()
        semsq = FourSquareAxes(corpus, ['hamlet'], ['jay-z/r. kelly'], ['swift'], ['dylan'])
        return semsq

def get_docs_categories_four():
    documents = [u"What art thou that usurp'st this time of night,", u'Together with that fair and warlike form', u'In which the majesty of buried Denmark', u'Did sometimes march? by heaven I charge thee, speak!', u'Halt! Who goes there?', u'[Intro]', u'It is I sire Tone from Brooklyn.', u'Well, speak up man what is it?', u'News from the East sire! THE BEST OF BOTH WORLDS HAS RETURNED!', u'I think it therefore manifest, from what I have here advanced,', u'that the main Point of Skill and Address, is to furnish Employment', u'for this Redundancy of Vapour, and prudently to adjust the Season 1', u'of it ; by which ,means it may certainly become of Cardinal', u"Ain't it just like the night to play tricks when you're tryin' to be so quiet?", u"We sit here stranded, though we're all doin' our best to deny it", u"And Louise holds a handful of rain, temptin' you to defy it", u'Lights flicker from the opposite loft', u'In this room the heat pipes just cough', u'The country music station plays soft']
    categories = ['hamlet'] * 4 + ['jay-z/r. kelly'] * 5 + ['swift'] * 4 + ['dylan'] * 6
    return (categories, documents)

class TestCorpusFromPandas(TestCase):

    def test_term_doc(self):
        self.assertIsInstance(self.corpus, CorpusDF)
        self.assertEqual(set(self.corpus.get_categories()), set(['hamlet', 'jay-z/r. kelly', '???']))
        self.assertEqual(self.corpus.get_num_docs(), 10)
        term_doc_df = self.corpus.get_term_freq_df()
        self.assertEqual(term_doc_df.loc['of'].sum(), 3)
        self.corpus.get_df()

    def test_chinese_error(self):
        with self.assertRaises(Exception):
            CorpusFromPandas(self.df, 'category', 'text', nlp=chinese_nlp).build()

    def test_get_texts(self):
        self.assertTrue(all(self.df['text'] == self.corpus.get_texts()))

    def test_search(self):
        expected = pd.DataFrame({'text': ["What art thou that usurp'st this time of night,", 'Together with that fair and warlike form'], 'category': ['hamlet', 'hamlet'], 'index': [0, 1]})
        self.assertIsInstance(self.corpus, CorpusDF)
        returned = self.corpus.search('that')
        pd.testing.assert_frame_equal(expected, returned[expected.columns])

    def test_search_bigram(self):
        expected = pd.DataFrame({'text': [u'Well, speak up man what is it?', u'Speak up, speak up, this is a repeat bigram.'], 'category': ['jay-z/r. kelly', '???'], 'index': [7, 9]}).reset_index(drop=True)
        self.assertIsInstance(self.corpus, CorpusDF)
        returned = self.corpus.search('speak up').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, returned[expected.columns])

    def test_search_index(self):
        expected = np.array([7, 9])
        self.assertIsInstance(self.corpus, CorpusDF)
        returned = self.corpus.search_index('speak up')
        np.testing.assert_array_equal(expected, returned)

    @classmethod
    def setUp(cls):
        categories, documents = get_docs_categories()
        cls.df = pd.DataFrame({'category': categories, 'text': documents})
        cls.corpus = CorpusFromPandas(cls.df, 'category', 'text', nlp=whitespace_nlp).build()

class TestTermDocMatrixFromScikit(TestCase):

    def test_build(self):
        from sklearn.feature_extraction.text import CountVectorizer
        categories, docs = get_docs_categories_semiotic()
        idx_store = IndexStore()
        y = np.array([idx_store.getidx(c) for c in categories])
        count_vectorizer = CountVectorizer()
        X_counts = count_vectorizer.fit_transform(docs)
        term_doc_mat = TermDocMatrixFromScikit(X=X_counts, y=y, feature_vocabulary=count_vectorizer.vocabulary_, category_names=idx_store.values()).build()
        self.assertEqual(term_doc_mat.get_categories()[:2], ['hamlet', 'jay-z/r. kelly'])
        self.assertEqual(term_doc_mat.get_term_freq_df().assign(score=term_doc_mat.get_scaled_f_scores('hamlet')).sort_values(by='score', ascending=False).index.tolist()[:5], ['that', 'march', 'did', 'majesty', 'sometimes'])

class DenseRankCharacteristicness(CharacteristicScorer):

    def get_scores(self, corpus):
        """
		Parameters
		----------
		corpus

		Returns
		-------
		(float, pd.Series)
		float: point on x-axis at even characteristicness
		pd.Series: term -> value between 0 and 1, sorted by score in a descending manner
			Background scores from corpus
		"""
        term_ranks = self.term_ranker(corpus).get_ranks()
        freq_df = pd.DataFrame({'corpus': term_ranks.sum(axis=1), 'standard': self.background_frequencies.get_background_frequency_df()['background']})
        freq_df = freq_df.loc[freq_df['corpus'].dropna().index].fillna(0)
        corpus_rank = rankdata(freq_df.corpus, 'dense')
        standard_rank = rankdata(freq_df.standard, 'dense')
        scores = corpus_rank / corpus_rank.max() - standard_rank / standard_rank.max()
        if self.rerank_ranks:
            rank_scores, zero_marker = self._rerank_scores(scores)
            freq_df['score'] = pd.Series(rank_scores, index=freq_df.index)
        else:
            if scores.min() < 0 and scores.max() > 0:
                zero_marker = -scores.min() / (scores.max() - scores.min())
            elif scores.min() > 0:
                zero_marker = 0
            else:
                zero_marker = 1
            freq_df['score'] = scale(scores)
        return (zero_marker, freq_df.sort_values(by='score', ascending=False)['score'])

