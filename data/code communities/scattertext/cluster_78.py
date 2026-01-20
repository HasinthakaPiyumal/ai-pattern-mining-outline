# Cluster 78

def _get_pos_tag(tok):
    pos = 'WORD'
    if tok.strip() == '':
        pos = 'SPACE'
    elif ord(tok[0]) in VALID_EMOJIS:
        pos = 'EMJOI'
    elif PUNCT_MATCHER.match(tok):
        pos = 'PUNCT'
    return pos

def bad_whitespace_nlp(doc):
    toks = []
    for tok in doc.split():
        pos = 'WORD'
        if tok.strip() == '':
            pos = 'SPACE'
        elif re.match('^\\W+$', tok):
            pos = 'PUNCT'
        toks.append(Tok(pos, tok[:2].lower(), tok.lower(), ent_type='', tag=''))
    return Doc([toks])

def whitespace_nlp_with_fake_chunks(doc, entity_type=None, tag_type=None):
    toks = _regex_parse_sentence(doc, entity_type, tag_type)
    words = [t for t in toks if t.pos_ == 'WORD']
    if len(words) < 5:
        return Doc([toks])
    else:
        return Doc([toks], noun_chunks=[Span(words[:2]), Span(words[1:3])])

def _regex_parse_sentence(doc, entity_type, tag_type):
    toks = _toks_from_sentence(doc, entity_type, tag_type)
    return toks

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

def _testing_nlp(doc):
    toks = []
    for tok in re.split('(\\W)', doc):
        pos = 'WORD'
        ent = ''
        tag = ''
        if tok.strip() == '':
            pos = 'SPACE'
        elif re.match('^\\W+$', tok):
            pos = 'PUNCT'
        if tok == 'Tone':
            ent = 'PERSON'
        if tok == 'Brooklyn':
            ent = 'GPE'
        toks.append(Tok(pos, tok[:2].lower(), tok.lower(), ent, tag))
    return Doc([toks])

class TransformerTokenizerWrapper(ABC):
    """
    Encapsulates the roberta tokenizer
    """

    def __init__(self, tokenizer, decoder=None, entity_type=None, tag_type=None, lower_case=False, name=None):
        self.tokenizer = tokenizer
        self.name = tokenizer.name_or_path if name is None else name
        if decoder is None:
            try:
                from text_unidecode import unidecode
            except:
                raise Exception("Please install the text_unicode package to preprocess documents. If you'd like to bypass this step, pass a text preprocessing (e.g., lambda x: x) function into the decode parameter of this class.")
            self.decoder = unidecode
        else:
            self.decoder = decoder
        self.entity_type = entity_type
        self.tag_type = tag_type
        self.lower_case = lower_case

    def tokenize(self, doc):
        """
        doc: str, text to be tokenized
        """
        sents = []
        if self.lower_case:
            doc = doc.lower()
        decoded_text = self.decoder(doc)
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer(decoded_text)['input_ids'], skip_special_tokens=True)
        last_idx = 0
        toks = []
        for raw_token in tokens:
            token_surface_string = self._get_surface_string(raw_token)
            if ord(raw_token[0]) == 266:
                last_idx += len(raw_token)
                continue
            try:
                token_idx = decoded_text.index(token_surface_string, last_idx)
            except Exception as e:
                print(decoded_text)
                print(token_surface_string)
                raise e
            toks.append(Tok(_get_pos_tag(token_surface_string), token_surface_string.lower(), raw_token.lower(), ent_type='' if self.entity_type is None else self.entity_type.get(token_surface_string, ''), tag='' if self.tag_type is None else self.tag_type.get(token_surface_string, ''), idx=token_idx))
            last_idx = token_idx + len(token_surface_string)
            if token_surface_string in ['.', '!', '?']:
                sents.append(toks)
                toks = []
        if len(toks) > 0:
            sents.append(toks)
        return Doc(sents, decoded_text)

    @abc.abstractmethod
    def _get_surface_string(self, raw_token: str) -> str:
        raise NotImplementedError()

    def get_subword_encoding_name(self) -> str:
        raise self.name

