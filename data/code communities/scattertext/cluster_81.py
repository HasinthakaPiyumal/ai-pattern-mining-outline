# Cluster 81

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

