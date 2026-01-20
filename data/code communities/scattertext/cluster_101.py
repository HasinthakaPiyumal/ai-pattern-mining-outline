# Cluster 101

class CategoryProjector(CategoryProjectorBase):

    def __init__(self, weighter=LengthNormalizer(), normalizer=StandardScaler(), selector=AssociationCompactor(1000, RankDifference), projector=PCA(2), fit_transform_kwargs=None, use_metadata=False):
        """

        :param weighter: instance of an sklearn class with fit_transform to weight X category corpus.
        :param normalizer: instance of an sklearn class with fit_transform to normalize term X category corpus.
        :param selector: instance of a compactor class, if None, no compaction will be done.
        :param projector: instance an sklearn class with fit_transform
        :param fit_transform_kwargs: optional, dict of kwargs to fit_transform
        :param use_metadata: bool, use metadata features
        """
        self.weighter_ = weighter
        self.normalizer_ = normalizer
        self.selector_ = selector
        self.projector_ = projector
        self.fit_transform_kwargs_ = {} if fit_transform_kwargs is None else fit_transform_kwargs
        self.use_metadata_ = use_metadata

    def use_metadata(self) -> 'CategoryProjector':
        self.use_metadata_ = True
        return self

    def get_category_embeddings(self, category_corpus):
        raw_category_counts = self._get_raw_category_counts(category_corpus)
        weighted_counts = self.weight(raw_category_counts)
        normalized_counts = self.normalize(weighted_counts)
        if type(normalized_counts) is not pd.DataFrame:
            normalized_counts = pd.DataFrame(normalized_counts.todense() if scipy.sparse.issparse(normalized_counts) else normalized_counts, columns=raw_category_counts.columns, index=raw_category_counts.index)
        return normalized_counts

    def _get_raw_category_counts(self, category_corpus):
        return category_corpus.get_freq_df(label_append='')

    def weight(self, category_counts):
        if self.weighter_ is None:
            return category_counts
        return self.weighter_.fit_transform(category_counts)

    def normalize(self, weighted_category_counts):
        if self.normalizer_ is not None:
            normalized_vals = self.normalizer_.fit_transform(weighted_category_counts)
            if issparse(normalized_vals):
                return normalized_vals
            if not isinstance(normalized_vals, DataFrame):
                return DataFrame(data=normalized_vals, columns=weighted_category_counts.columns, index=weighted_category_counts.index)
            else:
                return normalized_vals
        return weighted_category_counts

    def select(self, corpus):
        if self.selector_ is None:
            return corpus
        if self.use_metadata_:
            self.selector_ = self.selector_.set_use_non_text_features(self.use_metadata_)
        return corpus.select(self.selector_, non_text=self.use_metadata_)

    def _project_category_corpus(self, category_corpus, x_dim=0, y_dim=1):
        normalized_counts = self.get_category_embeddings(category_corpus)
        proj = self.projector_.fit_transform(normalized_counts.T, **self.fit_transform_kwargs_)
        return CategoryProjection(category_corpus, normalized_counts, proj, x_dim=x_dim, y_dim=y_dim)

    def _get_category_metadata_corpus(self, corpus):
        return self.select(corpus).use_categories_as_metadata()

    def _get_category_metadata_corpus_and_replace_terms(self, corpus):
        return self.select(corpus).use_categories_as_metadata_and_replace_terms()

