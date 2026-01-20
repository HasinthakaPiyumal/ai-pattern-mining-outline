# Cluster 59

class DispersionRanker(TermRanker):

    def get_ranks(self, label_append: str=f' {metric}') -> pd.DataFrame:
        compare_dispersion_df = get_category_dispersion(self._corpus, metric=metric, corpus_to_parts=corpus_to_parts, non_text=self._use_non_text_features)
        rank_df = pd.DataFrame({cat + label_append: compare_dispersion_df[f'{cat}_{metric}'] for cat_i, cat in enumerate(self._corpus.get_categories())})
        if metric in NULL_VALUES:
            rank_df = rank_df.fillna(NULL_VALUES[metric])
        else:
            for cat in self._corpus.get_categories():
                cat_compare_df = compare_dispersion_df[[f'{cat}_Frequency', f'{cat}_{metric}']].dropna()
                r = pearsonr(x=cat_compare_df.dropna()[f'{cat}_Frequency'], y=cat_compare_df.dropna()[f'{cat}_{metric}']).statistic
                if r < 0:
                    fill = compare_dispersion_df[f'{cat}_{metric}'].min() * impute_null_coef
                else:
                    fill = compare_dispersion_df[f'{cat}_{metric}'].max() / impute_null_coef
                rank_df[cat + label_append] = rank_df[cat + label_append].fillna(fill)
        return rank_df

def get_category_dispersion(corpus: TermDocMatrix, metric: str, corpus_to_parts: Optional[Callable[['TermDocMatrix'], List]]=None, include_residual: bool=False, include_residual_regressor: Optional[object]=None, non_text: bool=False) -> pd.DataFrame:
    """

    :param corpus:  TermDocMatrix to process
    :param metric: a metric present in Dispersion.get_df. May be "DA".
    :param corpus_to_parts: Optional function which takes a TermDocMatrix and returns a list of parts of each doc. None indicates each doc is a separate part.
    :param non_text: Use non text features. False by default
    :param include_residual: Include the residual
    :param include_residual_regressor: Use a regressor for the residual computation
    :return: Dataframe giving category-specific features
    """
    data = {}
    for category in corpus.get_categories():
        category_corpus = corpus.remove_categories([c for c in corpus.get_categories() if c != category])
        if corpus_to_parts is not None:
            category_corpus = category_corpus.recategorize(corpus_to_parts)
        dispersion = Dispersion(category_corpus, non_text=non_text, use_categories_as_documents=corpus_to_parts is not None, vocabulary=corpus.get_terms(use_metadata=non_text), add_smoothing_part=True)
        dispersion_df = dispersion.get_df(include_da=metric == 'DA')
        data[category + '_Frequency'] = dispersion_df.Frequency
        data[category + '_' + metric] = dispersion_df[metric]
        if include_residual:
            residual_df = dispersion.get_adjusted_metric_df(metric=metric)
            data[f'{category}_{metric}_Residual'] = residual_df['Residual']
            data[f'{category}_{metric}_Estimate'] = residual_df['Estimate']
    return pd.DataFrame(data)

