# Cluster 45

class ScatterChartExplorer(ScatterChart):

    def __init__(self, corpus, verbose=False, **kwargs):
        """See ScatterChart.  This lets you click on terms to see what contexts they tend to appear in.
        Running the `to_dict` function outputs
        """
        ScatterChart.__init__(self, corpus, verbose, **kwargs)
        self._term_metadata = None

    def to_dict(self, category, category_name=None, not_category_name=None, scores=None, metadata=None, max_docs_per_category=None, transform=percentile_alphabetical, alternative_text_field=None, title_case_names=False, not_categories=None, neutral_categories=None, extra_categories=None, neutral_category_name=None, extra_category_name=None, background_scorer=None, include_term_category_counts=False, use_offsets=False, **kwargs):
        """

        Parameters
        ----------
        category : str
            Category to annotate.  Exact value of category.
        category_name : str, optional
            Name of category which will appear on web site. Default None is same as category.
        not_category_name : str, optional
            Name of ~category which will appear on web site. Default None is same as "not " + category.
        scores : np.array, optional
            Scores to use for coloring.  Defaults to None, or RankDifference scores
        metadata, None or array-like.
          List of metadata for each document.  Defaults to a list of blank strings.
        max_docs_per_category, None or int, optional
          Maximum number of documents to store per category.  Defaults to 4.
        transform : function, optional
            Function for ranking terms.  Defaults to scattertext.Scalers.percentile_lexicographic.
        alternative_text_field : str or None, optional
            Field in from dataframe used to make corpus to display in place of parsed text. Only
            can be used if corpus is a ParsedCorpus instance.
        title_case_names : bool, default False
          Should the program title-case the category and not-category names?
        not_categories : list, optional
            List of categories to use as "not category".  Defaults to all others.
        neutral_categories : list, optional
            List of categories to use as neutral.  Defaults [].
        extra_categories : list, optional
            List of categories to use as extra.  Defaults [].
        neutral_category_name : str
            "Neutral" by default. Only active if show_neutral is True.  Name of the neutra l
            column.
        extra_category_name : str
            "Extra" by default. Only active if show_neutral and show_extra are true. Name of the
            extra column.
        background_scorer : CharacteristicScorer, optional
            Used for bg scores
        include_term_category_counts : bool, default False
            Includes term-category counts in keyed off 'term-category-count'. If use_non_text_features,
            use metadata counts instead.

        Returns
        -------
        dictionary {info: {'category_name': full category name, ...},
                    docs: {'texts': [doc1text, ...],
                            'labels': [1, 0, ...],
                            'meta': ['<b>blah</b>', '<b>blah</b>']},

                    // if include_term_category_counts
                    termCounts: [term num -> [total occurrences, total documents, variance], ... for the number of categories]

                    data: {term:term,
                           x:frequency [0-1],
                           y:frequency [0-1],
                           s: score,
                           bg: background score,
                           as: association score,
                           cat25k: freq per 25k in category,
                           cat: count in category,
                           ncat: count in non-category,
                           catdocs: [docnum, ...],
                           ncatdocs: [docnum, ...]
                           ncat25k: freq per 25k in non-category}
                           etc: term specific dictionary (if inject_term_metadata is called and contains terms)}
        """
        if kwargs is not {} and self.verbose:
            logging.info('Excessive arguments passed to ScatterChartExplorer.to_dict: ' + str(kwargs))
        json_data = ScatterChart.to_dict(self, category, category_name=category_name, not_category_name=not_category_name, scores=scores, transform=transform, title_case_names=title_case_names, not_categories=not_categories, neutral_categories=neutral_categories, extra_categories=extra_categories, background_scorer=background_scorer, use_offsets=use_offsets)
        docs_getter = self._make_docs_getter(max_docs_per_category, alternative_text_field)
        if neutral_category_name is None:
            neutral_category_name = 'Neutral'
        if extra_category_name is None:
            extra_category_name = 'Extra'
        metadata_series = metadata
        if callable(metadata):
            metadata_series = metadata(self.term_doc_matrix)
        json_data['docs'] = self._get_docs_structure(docs_getter, metadata_series)
        json_data['info']['neutral_category_name'] = neutral_category_name
        json_data['info']['extra_category_name'] = extra_category_name
        if include_term_category_counts:
            terms = np.array([term_struct['term'] for term_struct in json_data['data']])
            json_data['termCounts'] = self._get_term_doc_counts(terms)
        return json_data

    def _get_term_doc_counts(self, terms):
        term_counts = []
        if self.scatterchartdata.use_non_text_features:
            term_doc_counts = self.term_doc_matrix.get_metadata_doc_count_df('').loc[terms]
            term_doc_freq = self.term_doc_matrix.get_metadata_freq_df('').loc[terms]
        else:
            term_doc_counts = self.term_doc_matrix.get_term_doc_count_df('').loc[terms]
            term_doc_freq = self.term_doc_matrix.get_term_freq_df('').loc[terms]
        term2idx = pd.Series(np.arange(len(terms)), index=terms)
        for category_i, category in enumerate(self.term_doc_matrix.get_categories()):
            category = str(category)
            term_ser = term_doc_freq[category]
            doc_ser = term_doc_counts[category]
            term_ser = term_ser[term_ser.values > 0]
            doc_ser = doc_ser[doc_ser.values > 0]
            category_counts = pd.Series(np.array([term_ser.values, doc_ser.values]).T.tolist(), index=term2idx[term_ser.index].values).to_dict()
            term_counts.append(category_counts)
        return term_counts

    def _make_docs_getter(self, max_docs_per_category, alternative_text_field):
        if max_docs_per_category is None:
            docs_getter = DocsAndLabelsFromCorpus(self.term_doc_matrix, alternative_text_field=alternative_text_field)
        else:
            docs_getter = DocsAndLabelsFromCorpusSample(self.term_doc_matrix, max_docs_per_category, alternative_text_field=alternative_text_field)
        if self.scatterchartdata.use_non_text_features or self.scatterchartdata.add_extra_features:
            docs_getter = docs_getter.use_non_text_features()
            if self.scatterchartdata.add_extra_features:
                docs_getter = docs_getter.use_terms_for_extra_features()
        return docs_getter

    def _get_docs_structure(self, docs_getter, metadata):
        if metadata is not None:
            return docs_getter.get_labels_and_texts_and_meta(np.array(metadata))
        else:
            return docs_getter.get_labels_and_texts()

    def _add_term_freq_to_json_df(self, json_df, term_freq_df, category):
        json_df = ScatterChart._add_term_freq_to_json_df(self, json_df, term_freq_df, category).assign(cat=term_freq_df[str(category) + ' freq'].astype(int), ncat=term_freq_df['not cat freq'].astype(int))
        if self._term_metadata is not None:
            json_df = json_df.assign(etc=term_freq_df['term'].apply(lambda term: self._term_metadata.get(term, {})))
        return json_df

    def inject_term_metadata(self, metadata):
        """

        :param metadata: dict, maps terms to a dictionary which will be added to term's json structure
        :return: ScatterChartExplorer
        """
        self._term_metadata = metadata
        return self

    def inject_term_metadata_df(self, metadata_df):
        """

        :param metadata_df: pd.DataFrame, indexed on terms with columns as structure
        :return: ScatterChartExplorer
        """
        term_metadata_dict = metadata_df.T.to_dict()
        return self.inject_term_metadata(term_metadata_dict)

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

