# Cluster 49

class FeatsFromTopicModel(FeatsFromTopicModelBase, FeatsFromSpacyDoc):

    def __init__(self, topic_model, use_lemmas=False, entity_types_to_censor=set(), entity_types_to_use=None, tag_types_to_censor=set(), strip_final_period=False, keyword_processor_args={'case_sensitive': False}):
        self._keyword_processor = KeywordProcessor(**keyword_processor_args)
        self._topic_model = topic_model.copy()
        if keyword_processor_args.get('case_sensitive', None) is False:
            for k, v in self._topic_model.items():
                self._topic_model[k] = [e.lower() for e in v]
        for keyphrase in reduce(lambda x, y: set(x) | set(y), self._topic_model.values()):
            self._keyword_processor.add_keyword(keyphrase)
        FeatsFromSpacyDoc.__init__(self, use_lemmas, entity_types_to_censor, tag_types_to_censor, strip_final_period)
        FeatsFromTopicModelBase.__init__(self, topic_model)

    def get_top_model_term_lists(self):
        return self._topic_model

    def _get_terms_from_doc(self, doc):
        return Counter(self._keyword_processor.extract_keywords(str(doc)))

    def get_feats(self, doc):
        return Counter(self._get_terms_from_doc(str(doc)))

class PyatePhrases(FeatsFromSpacyDoc):

    def __init__(self, extractor=None, **args):
        import pyate
        self._extractor = pyate.combo_basic if extractor is None else extractor
        FeatsFromSpacyDoc.__init__(self, **args)

    def get_feats(self, doc):
        return Counter(self._extractor(str(doc)).to_dict())

class FeatsFromSpacyDocAndEmpath(FeatsFromSpacyDoc):

    def __init__(self, use_lemmas=False, entity_types_to_censor=set(), tag_types_to_censor=set(), strip_final_period=False, empath_analyze_function=None, **kwargs):
        """
        Parameters
        ----------
        empath_analyze_function: function (default=empath.Empath().analyze)
            Function that produces a dictionary mapping Empath categories to

        Other parameters from FeatsFromSpacyDoc.__init__
        """
        if empath_analyze_function is None:
            try:
                import empath
            except ImportError:
                raise Exception('Please install the empath library to use FeatsFromSpacyDocAndEmpath.')
            self._empath_analyze_function = empath.Empath().analyze
        else:
            self._empath_analyze_function = partial(empath_analyze_function, kwargs={'tokenizer': 'bigram'})
        FeatsFromSpacyDoc.__init__(self, use_lemmas, entity_types_to_censor, tag_types_to_censor, strip_final_period)

    def get_doc_metadata(self, doc, prefix=''):
        empath_counter = Counter()
        if version_info[0] >= 3:
            doc = str(doc)
        for empath_category, score in self._empath_analyze_function(doc).items():
            if score > 0:
                empath_counter[prefix + empath_category] = int(score)
        return empath_counter

    def has_metadata_term_list(self):
        return True

    def get_top_model_term_lists(self):
        try:
            import empath
        except ImportError:
            raise Exception('Please install the empath library to use FeatsFromSpacyDocAndEmpath.')
        return dict(empath.Empath().cats)

class PyTextRankPhrases(FeatsFromSpacyDoc):

    def __init__(self, use_lemmas=False, entity_types_to_censor=set(), tag_types_to_censor=set(), strip_final_period=False):
        FeatsFromSpacyDoc.__init__(self, use_lemmas, entity_types_to_censor, tag_types_to_censor, strip_final_period)
        self._include_chunks = False
        self._rank_smoothing_constant = 0

    def include_chunks(self):
        """
        Use each chunk in a phrase instead of just the span identified as a phrase
        :return: self
        """
        self._include_chunks = True
        return self

    def set_rank_smoothing_constant(self, rank_smoothing_constant):
        """
        Add a quantity

        :param rank_smoothing_constant: float
        :return: self
        """
        self._rank_smoothing_constant = rank_smoothing_constant
        return self

    def get_doc_metadata(self, doc):
        phrase_counter = Counter()
        try:
            for phrase in doc._.phrases:
                if self._include_chunks:
                    for chunk in phrase.chunks:
                        phrase_counter[str(chunk)] += phrase.rank + self._rank_smoothing_constant
                else:
                    phrase_counter[phrase.text] += phrase.count * (phrase.rank + self._rank_smoothing_constant)
        except:
            import pytextrank
            tr = pytextrank.TextRank()
            tr.doc = doc
            phrases = tr.calc_textrank()
            for phrase in phrases:
                if self._include_chunks:
                    for chunk in phrase.chunks:
                        phrase_counter[str(chunk)] += phrase.rank + self._rank_smoothing_constant
                else:
                    phrase_counter[phrase.text] += phrase.count * (phrase.rank + self._rank_smoothing_constant)
        return phrase_counter

    def get_feats(self, doc):
        return Counter()

class SpacyEntities(FeatsFromSpacyDoc):

    def __init__(self, use_lemmas=False, entity_types_to_censor=set(), entity_types_to_use=None, tag_types_to_censor=set(), strip_final_period=False):
        self._entity_types_to_use = entity_types_to_use
        FeatsFromSpacyDoc.__init__(self, use_lemmas, entity_types_to_censor, tag_types_to_censor, strip_final_period)

    def get_feats(self, doc):
        return Counter([' '.join(str(ent).split()).lower() for ent in doc.ents if (self._entity_types_to_use is None or ent.label_ in self._entity_types_to_use) and ent.label_ not in self._entity_types_to_censor])

class FlexibleNGrams(FeatsFromSpacyDoc, FlexibleNGramFeaturesBase):

    def __init__(self, ngram_sizes: Optional[List[int]]=None, exclude_ngram_filter: Optional[Callable]=None, text_from_token: Optional[Callable]=None, validate_token: Optional[Callable]=None, exclude_sentence_filter: Optional[Callable[[str], bool]]=None):
        FeatsFromSpacyDoc.__init__(self)
        FlexibleNGramFeaturesBase.__init__(self, exclude_ngram_filter, ngram_sizes, text_from_token, validate_token, whitespace_substitute=None, exclude_sentence_filter=exclude_sentence_filter)

    def get_feats(self, doc):
        return self._doc_to_feature_representation(doc)

    def _add_ngram_to_token_stats(self, ngram, offset_tokens):
        toktext = ' '.join((self.text_from_token(tok) for tok in ngram))
        offset_tokens.setdefault(toktext, 0)
        offset_tokens[toktext] += 1

