# Cluster 26

class TestCohensD(TestCase):

    def test_get_cohens_d_scores(self):
        corpus = build_hamlet_jz_corpus()
        np.testing.assert_almost_equal(CohensD(corpus).set_term_ranker(OncePerDocFrequencyRanker).set_categories('hamlet').get_scores()[:5], [-0.2303607, 0.8838835, 0.8838835, 1.4028612, 0.8838835])

    def test_get_cohens_d_scores_zero_robust(self):
        corpus = build_hamlet_jz_corpus()
        corpus._X[1, :] = 0
        np.testing.assert_almost_equal(CohensD(corpus).set_term_ranker(OncePerDocFrequencyRanker).set_categories('hamlet').get_scores()[:5], [-0.2303607, 0.8838835, 0.8838835, 0.8838835, 0.8838835])

    def test_get_cohens_d_score_df(self):
        corpus = build_hamlet_jz_corpus()
        columns = CohensD(corpus).set_term_ranker(OncePerDocFrequencyRanker).set_categories('hamlet').get_score_df().columns
        self.assertEqual(set(columns), set(['cohens_d', 'cohens_d_se', 'cohens_d_z', 'cohens_d_p', 'hedges_g', 'hedges_g_se', 'hedges_g_z', 'hedges_g_p', 'm1', 'm2', 'count1', 'count2', 'docs1', 'docs2']))

    def test_get_cohens_d_score_df_p_vals(self):
        corpus = build_hamlet_jz_corpus()
        columns = CohensD(corpus).set_term_ranker(OncePerDocFrequencyRanker).set_categories('hamlet').get_score_df().columns
        self.assertEqual(set(columns), set(['cohens_d', 'cohens_d_se', 'cohens_d_z', 'cohens_d_p', 'hedges_g', 'hedges_g_se', 'hedges_g_z', 'hedges_g_p', 'm1', 'm2', 'count1', 'count2', 'docs1', 'docs2']))

    def test_get_name(self):
        corpus = build_hamlet_jz_corpus()
        self.assertEqual(CohensD(corpus).set_categories('hamlet').get_name(), "Cohen's d")

    def test_get_name_hedges(self):
        corpus = build_hamlet_jz_corpus()
        self.assertEqual(HedgesG(corpus).set_categories('hamlet').get_name(), "Hedge's g")
        self.assertEqual(len(HedgesG(corpus).set_categories('hamlet').get_scores()), corpus.get_num_terms())

def build_hamlet_jz_corpus():
    df = build_hamlet_jz_df()
    return CorpusFromParsedDocuments(df=df, category_col='category', parsed_col='parsed').build()

class TestBetaPosterior(TestCase):

    def test_get_score_df(self):
        corpus = build_hamlet_jz_corpus()
        beta_posterior = BetaPosterior(corpus).set_categories('hamlet')
        score_df = beta_posterior.get_score_df()
        scores = beta_posterior.get_scores()
        np.testing.assert_almost_equal(scores[:5], [-0.3194860824225506, 1.0294085051562822, 1.0294085051562822, 1.234664219528909, 1.0294085051562822])

    def test_get_name(self):
        corpus = build_hamlet_jz_corpus()
        self.assertEqual(BetaPosterior(corpus).get_name(), 'Beta Posterior')

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

class TestRelativeEntropy(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.corpus = build_hamlet_jz_corpus()

    def test_get_scores(self):
        result = RelativeEntropy(self.corpus).set_categories('hamlet').get_scores()
        self.assertEquals(type(result), pd.Series)
        np.testing.assert_array_equal(np.array(result.index), self.corpus.get_terms())

    def test_get_name(self):
        self.assertEquals(RelativeEntropy(self.corpus).set_categories('hamlet').get_name(), 'Frankhauser Relative Entropy')

class TestBM25Difference(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.corpus = build_hamlet_jz_corpus()

    def test_get_scores(self):
        result = BM25Difference(self.corpus).set_categories('hamlet').get_scores()
        self.assertEquals(type(result), pd.Series)
        np.testing.assert_array_equal(np.array(result.index), self.corpus.get_terms())

    def test_get_name(self):
        self.assertEquals(BM25Difference(self.corpus).set_categories('hamlet').get_name(), 'BM25 difference')

class TestCredTFIDF(TestCase):

    def test_get_score_df(self):
        corpus = build_hamlet_jz_corpus()
        tfidf = CredTFIDF(corpus).set_term_ranker(OncePerDocFrequencyRanker).set_categories('hamlet')
        np.testing.assert_almost_equal(tfidf.get_scores()[:5], [3.0757237e-05, 0.041256023, 0.041256023, 0.055708409, 0.041256023])
        self.assertEqual(list(tfidf.get_score_df().columns), ['pos_cred_tfidf', 'neg_cred_tfidf', 'delta_cred_tf_idf'])

    def test_get_name(self):
        corpus = build_hamlet_jz_corpus()
        self.assertEqual(CredTFIDF(corpus).get_name(), 'Delta mean cred-tf-idf')

class TestScatterChartExplorer(TestCase):

    def test_to_dict(self):
        np.random.seed(0)
        random.seed(0)
        corpus = build_hamlet_jz_corpus()
        j = ScatterChartExplorer(corpus, minimum_term_frequency=0).to_dict('hamlet')
        self.assertEqual(set(j.keys()), set(['info', 'data', 'docs']))
        self.assertEqual(set(j['info'].keys()), set(['not_category_name', 'category_name', 'category_terms', 'not_category_internal_names', 'not_category_terms', 'category_internal_name', 'categories', 'neutral_category_name', 'extra_category_name', 'neutral_category_internal_names', 'extra_category_internal_names']))
        self.assertEqual(list(j['docs']['labels']), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(list(j['docs']['texts']), ["what art thou that usurp'st this time of night,", 'together with that fair and warlike form', 'in which the majesty of buried denmark', 'did sometimes march? by heaven i charge thee, speak!', 'halt! who goes there?', 'it is i sire tone from brooklyn.', 'well, speak up man what is it?', 'news from the east sire! the best of both worlds has returned!'])
        expected = {'y': 0.5, 'ncat': 0, 'ncat25k': 0, 'bg': 5, 'cat': 1, 's': 0.5, 'term': 'art', 'os': 0.5192, 'extra': 0, 'extra25k': 0, 'cat25k': 758, 'x': 0.06, 'neut': 0, 'neut25k': 0, 'ox': 5, 'oy': 3}
        actual = [t for t in j['data'] if t['term'] == 'art'][0]
        '\n\t\tfor var in expected.keys():\n\t\t\ttry:\n\t\t\t\t#np.testing.assert_almost_equal(actual[var], expected[var],decimal=1)\n\t\t\texcept TypeError:\n\t\t\t\tself.assertEqual(actual[var], expected[var])\n\t\t'
        self.assertEqual(set(expected.keys()), set(actual.keys()))
        self.assertEqual(expected['term'], actual['term'])
        self.assertEqual(j['docs'].keys(), {'texts', 'labels', 'categories'})
        j = ScatterChartExplorer(corpus, minimum_term_frequency=0).inject_term_metadata({'art': {'display': 'blah blah blah', 'color': 'red'}}).to_dict('hamlet')
        actual = [t for t in j['data'] if t['term'] == 'art'][0]
        expected = {'y': 0.5, 'ncat': 0, 'ncat25k': 0, 'bg': 5, 'cat': 1, 's': 0.5, 'term': 'art', 'os': 0.5192, 'extra': 0, 'extra25k': 0, 'cat25k': 758, 'x': 0.06, 'neut': 0, 'neut25k': 0, 'ox': 5, 'oy': 3, 'etc': {'display': 'blah blah blah', 'color': 'red'}}
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        self.assertEqual(actual['etc'], expected['etc'])
        actual = [t for t in j['data'] if t['term'] != 'art'][0]
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        self.assertEqual(actual['etc'], {})

    def test_hide_terms(self):
        corpus = build_hamlet_jz_corpus().get_unigram_corpus()
        terms_to_hide = ['thou', 'heaven']
        sc = ScatterChartExplorer(corpus, minimum_term_frequency=0).hide_terms(terms_to_hide)
        self.assertEquals(type(sc), ScatterChartExplorer)
        j = sc.to_dict('hamlet', include_term_category_counts=True)
        self.assertTrue(all(['display' in t and t['display'] == False for t in j['data'] if t['term'] in terms_to_hide]))
        self.assertTrue(all(['display' not in t for t in j['data'] if t['term'] not in terms_to_hide]))

    def test_include_term_category_counts(self):
        corpus = build_hamlet_jz_corpus().get_unigram_corpus()
        j = ScatterChartExplorer(corpus, minimum_term_frequency=0).to_dict('hamlet', include_term_category_counts=True)
        self.assertEqual(set(j.keys()), set(['info', 'data', 'docs', 'termCounts']))
        self.assertEqual(len(j['termCounts']), corpus.get_num_categories())
        term_idx_set = set()
        for cat_counts in j['termCounts']:
            term_idx_set |= set(cat_counts.keys())
            self.assertTrue(all([freq >= docs for freq, docs in cat_counts.values()]))
        self.assertEqual(len(term_idx_set), corpus.get_num_terms())

    def test_multi_categories(self):
        corpus = get_test_corpus()
        j_vs_all = ScatterChartExplorer(corpus=corpus, minimum_term_frequency=0).to_dict('hamlet')
        j_vs_swift = ScatterChartExplorer(corpus=corpus, minimum_term_frequency=0).to_dict('hamlet', not_categories=['swift'])
        self.assertNotEqual(set(j_vs_all['info']['not_category_internal_names']), set(j_vs_swift['info']['not_category_internal_names']))
        self.assertEqual(list(j_vs_all['docs']['labels']), list(j_vs_swift['docs']['labels']))
        self.assertEqual(list(j_vs_all['docs']['categories']), list(j_vs_swift['docs']['categories']))

    def test_metadata(self):
        corpus = build_hamlet_jz_corpus()
        meta = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
        j = ScatterChartExplorer(corpus, minimum_term_frequency=0).to_dict('hamlet', metadata=meta)
        self.maxDiff = None
        j['docs']['labels'] = list(j['docs']['labels'])
        self.assertEqual(j['docs'], {'labels': [0, 0, 0, 0, 1, 1, 1, 1], 'categories': ['hamlet', 'jay-z/r. kelly'], 'meta': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'], 'texts': ["what art thou that usurp'st this time of night,", 'together with that fair and warlike form', 'in which the majesty of buried denmark', 'did sometimes march? by heaven i charge thee, speak!', 'halt! who goes there?', 'it is i sire tone from brooklyn.', 'well, speak up man what is it?', 'news from the east sire! the best of both worlds has returned!']})

    def test_alternative_text(self):
        corpus = build_hamlet_jz_corpus_with_alt_text()
        j = ScatterChartExplorer(corpus, minimum_term_frequency=0).to_dict('hamlet', alternative_text_field='alt')
        self.assertEqual(j['docs']['texts'][0], j['docs']['texts'][0].upper())
        j = ScatterChartExplorer(corpus, minimum_term_frequency=0).to_dict('hamlet')
        self.assertNotEqual(j['docs']['texts'][0], j['docs']['texts'][0].upper())

    def test_extra_features(self):
        corpus = build_hamlet_jz_corpus_with_meta()
        meta = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
        j = ScatterChartExplorer(corpus, minimum_term_frequency=0, use_non_text_features=True).to_dict('hamlet', metadata=meta)
        extras = [{'cat3': 1, 'cat4': 2}, {'cat4': 2}, {'cat3': 2, 'cat5': 1}, {'cat6': 2, 'cat9': 1}, {'cat3': 1, 'cat4': 2}, {'cat1': 2, 'cat2': 1}, {'cat2': 2, 'cat5': 1}, {'cat3': 2, 'cat4': 1}]
        extras = [{'cat1': 2}] * 8
        self.maxDiff = None
        j['docs']['labels'] = list(j['docs']['labels'])
        self.assertEqual(j['docs'], {'labels': [0, 0, 0, 0, 1, 1, 1, 1], 'categories': ['hamlet', 'jay-z/r. kelly'], 'extra': extras, 'meta': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'], 'texts': ["what art thou that usurp'st this time of night,", 'together with that fair and warlike form', 'in which the majesty of buried denmark', 'did sometimes march? by heaven i charge thee, speak!', 'halt! who goes there?', 'it is i sire tone from brooklyn.', 'well, speak up man what is it?', 'news from the east sire! the best of both worlds has returned!']})

def build_hamlet_jz_corpus_with_alt_text():
    df = build_hamlet_jz_df_with_alt_text()
    return CorpusFromParsedDocuments(df=df, category_col='category', parsed_col='parsed').build()

def _get_category_scatter_chart_explorer(category_projection, scaler, term_ranker, verbose):
    category_scatter_chart_explorer = ScatterChartExplorer(category_projection.get_corpus(), minimum_term_frequency=0, minimum_not_category_term_frequency=0, pmi_threshold_coefficient=0, filter_unigrams=False, jitter=0, max_terms=None, use_non_text_features=True, term_significance=None, terms_to_include=None, verbose=verbose, dont_filter=True)
    proj_df = category_projection.get_pandas_projection()
    category_scatter_chart_explorer.inject_coordinates(x_coords=scaler(proj_df['x']), y_coords=scaler(proj_df['y']), original_x=proj_df['x'], original_y=proj_df['y'])
    return category_scatter_chart_explorer

class TestCredTFIDF(TestCase):

    def test_get_score_df(self):
        corpus = build_hamlet_jz_corpus()
        self.assertEqual(set(CredTFIDF(corpus).set_categories('hamlet').get_score_df().columns), set(['pos_cred_tfidf', 'neg_cred_tfidf', 'delta_cred_tf_idf']))

