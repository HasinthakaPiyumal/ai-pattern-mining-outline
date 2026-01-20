# Cluster 60

class NgramPercentageCompactor(BaseTermCompactor):

    def __init__(self, term_ranker: type=AbsoluteFrequencyRanker, minimum_term_count: int=0, usage_portion: float=0.7, token_split_function: Callable[[str], List[str]]=lambda x: x.split(), token_join_function: Callable[[List[str]], str]=' '.join, verbose: bool=False):
        """

        Parameters
        ----------
        term_ranker : TermRanker
            Default AbsoluteFrequencyRanker
        minimum_term_count : int
            Default 0
        usage_portion : float
            Portion of times term is used in a containing n-gram for it to be eliminated
            Default 0.8
        token_split_function : Callable[[str], List[str]]
            Function to split string into parts,
            Default lambda x: x.split()
        token_join_function : Callable[[List[str]], str]
            Function to join parsts into a string
            Default lambda x: ' '.join(x)
        verbose : bool
            Show progress bar
        """
        self.term_ranker = term_ranker
        self.minimum_term_count = minimum_term_count
        self.usage_portion = usage_portion
        self.verbose = verbose
        self.token_split_function = token_split_function
        self.token_join_function = token_join_function

    def compact(self, term_doc_matrix, non_text=False):
        elim_df = self.get_elimination_df(term_doc_matrix, non_text)
        if self.verbose:
            print(f'Ngram percentage compactor removed {len(elim_df)} terms.')
        return term_doc_matrix.remove_terms(terms=elim_df.Eliminations, ignore_absences=True, non_text=non_text)

    def get_elimination_df(self, term_doc_matrix, non_text=False) -> pd.DataFrame:
        freq_df = pd.DataFrame({'Count': self.term_ranker(term_doc_matrix).set_non_text(non_text).get_ranks().sum(axis=1)})[lambda df: df.Count >= self.minimum_term_count]
        max_subgramsize = max((len(self.token_split_function(tok)) for tok in freq_df.index)) - 1
        eliminations = []
        eliminators = []
        it = freq_df.iterrows()
        if self.verbose:
            it = tqdm(freq_df.iterrows(), total=len(freq_df))
        for row_i, row in it:
            toks = self.token_split_function(row.name)
            gram_len = len(toks)
            if gram_len > 1:
                subgrams = []
                for i in range(min(max_subgramsize, gram_len - 1), 0, -1):
                    for subtoks in sequence_window(toks, i):
                        subgrams.append(self.token_join_function(subtoks))
                found_subgrams = freq_df.index.intersection(subgrams)
                to_elim = list(freq_df.loc[found_subgrams][lambda df: row.Count > df.Count * self.usage_portion].index)
                if to_elim:
                    eliminations += to_elim
                    eliminators += [row.name] * len(to_elim)
        return pd.DataFrame({'Eliminations': eliminations, 'Eliminators': eliminators})

def sequence_window(seq, n=2):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield list(win)
    for e in it:
        win.append(e)
        yield list(win)

class FlexibleNGramFeaturesBase:

    def __init__(self, exclude_ngram_filter: Optional[Callable[[str], bool]]=None, ngram_sizes: Optional[List[int]]=None, text_from_token: Optional[Callable[[str], str]]=None, validate_token: Optional[Callable[[spacy.tokens.Token], bool]]=None, whitespace_substitute: Optional[Callable[[str], str]]=None, exclude_sentence_filter: Optional[Callable[[str], bool]]=None):
        self.exclude_ngram_filter = (lambda x: False) if exclude_ngram_filter is None else exclude_ngram_filter
        self.ngram_sizes = [1, 2, 3] if ngram_sizes is None else ngram_sizes
        self.text_from_token = (lambda tok: tok.lower_) if text_from_token is None else text_from_token
        self.validate_token = (lambda tok: tok.tag_ != '_SP' and tok.orth_.strip() != '') if validate_token is None else validate_token
        self.whitespace_substitute = ' ' if whitespace_substitute is None else whitespace_substitute
        self.exclude_sentence_filter = (lambda s: False) if exclude_sentence_filter is None else exclude_sentence_filter

    def _doc_to_feature_representation(self, doc) -> Dict:
        offset_tokens = {}
        for sent in doc.sents:
            if not self.exclude_sentence_filter(sent):
                for ngram_size in self.ngram_sizes:
                    if len(sent) >= ngram_size:
                        for ngram in sequence_window(sent, ngram_size):
                            if not self.exclude_ngram_filter(ngram):
                                self._add_ngram_to_token_stats(ngram, offset_tokens)
        return offset_tokens

    def _add_ngram_to_token_stats(self, ngram, offset_tokens):
        toktext = self.whitespace_substitute.join((self.text_from_token(tok) for tok in ngram))
        token_stats = offset_tokens.setdefault(toktext, [0, []])
        token_stats[0] += 1
        start = ngram[0].idx
        end = ngram[-1].idx + len(ngram[-1].orth_)
        token_stats[1].append((start, end))

    def _sent_to_token_features(self, sent):
        sent_features = []
        for tok in sent:
            if self.validate_token(tok):
                sent_features.append([tok.idx, tok.idx + len(tok), self.text_from_token(tok)])
        return sent_features

