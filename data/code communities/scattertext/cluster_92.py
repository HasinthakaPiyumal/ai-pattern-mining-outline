# Cluster 92

def category_specific_inter_arrivals_from_offset_corpus(corpus: OffsetCorpus, weibull_fit_func: Optional[Callable]=None, verbose: bool=False) -> pd.DataFrame:
    if not isinstance(corpus, OffsetCorpus):
        raise Exception(f'The corpus argument was of type {type(corpus)}. Use offset_corpus_to_concatenated_inter_arrivals instead.')
    if weibull_fit_func is None:
        from reliability.Fitters import Fit_Weibull_2P
        weibull_fit_func = lambda failures: Fit_Weibull_2P(failures=failures, show_probability_plot=False, print_results=False)
    cat_term_ias = {cat: offset_corpus_to_concatenated_inter_arrivals(corpus, categories=[cat], verbose=verbose) for cat in corpus.get_categories()}
    data = []
    for cat, term_ias in cat_term_ias.items():
        for term, ias in tqdm(term_ias.items()):
            if len(ias) > 1:
                data.append({'term': term, 'cat': cat, 'freq': len(ias), **__get_term_stats(ias, weibull_fit_func)})
    return pd.DataFrame(data)

def offset_corpus_to_concatenated_inter_arrivals(corpus: OffsetCorpus, categories: Optional[List[str]]=None, generator: Optional[np.random._generator.Generator]=None, domains_to_preserve: Optional[List[str]]=None, join_text: str='\n', verbose: bool=False, nlp: Optional[spacy.Language]=None) -> Dict[str, List[int]]:
    if not isinstance(corpus, OffsetCorpus):
        raise Exception(f'The corpus argument was of type {type(corpus)}. Use offset_corpus_to_concatenated_inter_arrivals instead.')
    doc_df = __order_docs_to_concat(categories, corpus, domains_to_preserve, generator, join_text)
    doc = __concatenate_doc(corpus, doc_df, join_text, nlp)
    doc_id_to_offset = dict(doc_df[['_OrigIdx', 'StartOffset']].set_index('_OrigIdx')['StartOffset'])
    term_inter_arrivals = {}
    it = corpus.get_offsets().items()
    if verbose:
        it = tqdm(it, total=len(corpus.get_offsets()))
    for term, doc_offsets in it:
        new_offsets = _translate_offsets_to_concatenated_doc(doc_id_to_offset, doc_offsets)
        term_inter_arrivals[term] = _collect_term_inter_arrivals_on_concatenated_doc(doc, new_offsets)
    return term_inter_arrivals

