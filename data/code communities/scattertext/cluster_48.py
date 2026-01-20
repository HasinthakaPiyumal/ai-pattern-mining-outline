# Cluster 48

def produce_scattertext_digraph(df, text_col, source_col, dest_col, source_name='Source', dest_name='Destination', graph_width=500, graph_height=500, metadata_func=None, enable_pan_and_zoom=True, engine='dot', graph_params=None, node_params=None, **kwargs):
    """

    :param df: pd.DataFrame
    :param text_col: str
    :param source_col: str
    :param dest_col: str
    :param source_name: str
    :param dest_name: str
    :param graph_width: int
    :param graph_height: int
    :param metadata_func: lambda
    :param enable_pan_and_zoom: bool
    :param engine: str, The graphviz engine (e.g., dot or neat)
    :param graph_params dict or None, graph parameters in graph viz
    :param node_params dict or None, node parameters in graph viz
    :param kwargs: dicdt
    :return:
    """
    graph_df = pd.concat([df.assign(__text=lambda df: df[source_col], __alttext=lambda df: df[text_col], __category='source'), df.assign(__text=lambda df: df[dest_col], __alttext=lambda df: df[text_col], __category='target')])
    corpus = CorpusFromParsedDocuments(graph_df, category_col='__category', parsed_col='__text', feats_from_spacy_doc=UseFullDocAsMetadata()).build()
    edges = corpus.get_df()[[source_col, dest_col]].rename(columns={source_col: 'source', dest_col: 'target'}).drop_duplicates()
    component_graph = SimpleDiGraph(edges).make_component_digraph(graph_params=graph_params, node_params=node_params)
    graph_renderer = ComponentDiGraphHTMLRenderer(component_graph, height=graph_height, width=graph_width, enable_pan_and_zoom=enable_pan_and_zoom, engine=engine)
    alternative_term_func = '(function(termDict) {\n        document.querySelectorAll(".dotgraph").forEach(svg => svg.style.display = \'none\');\n        showTermGraph(termDict[\'term\']);\n        return true;\n    })'
    scatterplot_structure = produce_scattertext_explorer(corpus, category='source', category_name=source_name, not_category_name=dest_name, minimum_term_frequency=0, pmi_threshold_coefficient=0, alternative_text_field='__alttext', use_non_text_features=True, transform=dense_rank, metadata=corpus.get_df().apply(metadata_func, axis=1) if metadata_func else None, return_scatterplot_structure=True, width_in_pixels=kwargs.get('width_in_pixels', 700), max_overlapping=kwargs.get('max_overlapping', 3), color_func=kwargs.get('color_func', '(function(x) {return "#5555FF"})'), alternative_term_func=alternative_term_func, **kwargs)
    html = GraphStructure(scatterplot_structure, graph_renderer=graph_renderer).to_html()
    return html

def whitespace_nlp_with_sentences(doc, entity_type=None, tag_type=None, tok_splitter_re=DEFAULT_TOK_SPLITTER_RE):
    sentence_split_pat = re.compile('([^\\.!?]*?[\\.!?$])', re.M)
    sents = []
    raw_sents = sentence_split_pat.findall(doc)
    if len(raw_sents) == 0:
        raw_sents = [doc]
    sent_start_idx = 0
    for sentence in raw_sents:
        toks = []
        start_idx_in_sentence = 0
        for tok in tok_splitter_re.split(sentence):
            if len(tok.strip()) > 0:
                toks.append(Tok(_get_pos_tag(tok), tok[:2].lower(), tok.lower(), ent_type='' if entity_type is None else entity_type.get(tok, ''), tag='' if tag_type is None else tag_type.get(tok, ''), idx=sent_start_idx + start_idx_in_sentence))
            start_idx_in_sentence += len(tok)
        sents.append(toks)
        sent_start_idx += len(sentence)
    return Doc(sents, doc)

class TestUseFullDocAsMetadata(TestCase):

    def test_get_feats(self):
        doc = whitespace_nlp_with_sentences('A a bb cc.')
        term_freq = UseFullDocAsMetadata().get_doc_metadata(doc)
        self.assertEqual(Counter({'A a bb cc.': 1}), term_freq)

class TestUseFullDocAsFeature(TestCase):

    def test_get_feats(self):
        doc = whitespace_nlp_with_sentences('A a bb cc.')
        term_freq = UseFullDocAsFeature().get_feats(doc)
        self.assertEqual(Counter({'A a bb cc.': 1}), term_freq)

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

class TestWhitespaceNLP(TestCase):

    def test_whitespace_nlp(self):
        raw = 'Hi! My name\n\t\tis Jason.  You can call me\n\t\tMr. J.  Is that your name too?\n\t\tHa. Ha ha.\n\t\t'
        doc = whitespace_nlp(raw)
        self.assertEqual(len(list(doc)), 55)
        self.assertEqual(len(doc.sents), 1)
        tok = Tok('WORD', 'Jason', 'jason', 'Name', 'NNP')
        self.assertEqual(len(tok), 5)
        self.assertEqual(str(tok), 'jason')
        self.assertEqual(str(Doc([[Tok('WORD', 'Jason', 'jason', 'Name', 'NNP'), Tok('WORD', 'a', 'a', 'Name', 'NNP')]], raw='asdfbasdfasd')), 'asdfbasdfasd')
        self.assertEqual(str(Doc([[Tok('WORD', 'Blah', 'blah', 'Name', 'NNP'), Tok('Space', ' ', ' ', ' ', ' '), Tok('WORD', 'a', 'a', 'Name', 'NNP')]])), 'blah a')

    def test_whitespace_nlp_with_sentences(self):
        raw = 'Hi! My name\n\t\tis Jason.  You can call me\n\t\tMr. J.  Is that your name too?\n\t\tHa. Ha ha.\n\t\t'
        doc = whitespace_nlp_with_sentences(raw)
        self.assertEqual(doc.text, raw)
        self.assertEqual(len(doc.sents), 7)
        self.assertEqual(doc[3].orth_, 'name')
        self.assertEqual(doc[25].orth_, '.')
        self.assertEqual(len(doc), 26)
        self.assertEqual(doc[3].idx, 7)
        self.assertEqual(raw[doc[3].idx:doc[3].idx + len(doc[3].orth_)], 'name')

    def test_whitespace_nlp_with_sentences_singleton(self):
        raw = 'Blah'
        self.assertEqual(whitespace_nlp_with_sentences(raw).text, raw)
        self.assertEqual(len(whitespace_nlp_with_sentences(raw).sents), 1)
        self.assertEqual(len(whitespace_nlp_with_sentences(raw).sents[0]), 1)
        raw = 'Blah.'
        self.assertEqual(whitespace_nlp_with_sentences(raw).text, raw)
        self.assertEqual(len(whitespace_nlp_with_sentences(raw).sents), 1)
        self.assertEqual(len(whitespace_nlp_with_sentences(raw).sents[0]), 2)

class TestFeatsFromScoredLexicon(TestCase):

    def test_main(self):
        lexicon_df = pd.DataFrame({'activation': {'a': 1.3846, 'abandon': 2.375, 'abandoned': 2.1, 'abandonment': 2.0, 'abated': 1.3333}, 'imagery': {'a': 1.0, 'abandon': 2.4, 'abandoned': 3.0, 'abandonment': 1.4, 'abated': 1.2}, 'pleasantness': {'a': 2.0, 'abandon': 1.0, 'abandoned': 1.1429, 'abandonment': 1.0, 'abated': 1.6667}})
        with self.assertRaises(AssertionError):
            FeatsFromScoredLexicon(3)
        feats_from_scored_lexicon = FeatsFromScoredLexicon(lexicon_df)
        self.assertEqual(set(feats_from_scored_lexicon.get_top_model_term_lists().keys()), set(['activation', 'imagery', 'pleasantness']))
        features = feats_from_scored_lexicon.get_doc_metadata(whitespace_nlp_with_sentences('I abandoned a wallet.'))
        np.testing.assert_almost_equal(features[['activation', 'imagery', 'pleasantness']], np.array([1.7423, 2.0, 1.57145]))

