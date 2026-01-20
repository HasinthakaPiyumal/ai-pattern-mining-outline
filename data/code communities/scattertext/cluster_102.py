# Cluster 102

class Doc2VecCategoryProjector(CategoryProjectorBase):

    def __init__(self, doc2vec_builder=None, projector=PCA(2)):
        """

        :param doc2vec_builder: Doc2VecBuilder, optional
            If None, a default model will be used
        :param projector: object
            Has fit_transform method
        """
        if doc2vec_builder is None:
            try:
                import gensim
            except:
                raise Exception('Please install gensim before using Doc2VecCategoryProjector/')
            self.doc2vec_builder = Doc2VecBuilder(gensim.models.Doc2Vec(vector_size=100, window=5, min_count=5, workers=6, alpha=0.025, min_alpha=0.025, epochs=50))
        else:
            assert type(doc2vec_builder) == Doc2VecBuilder
            self.doc2vec_builder = doc2vec_builder
        self.projector = projector

    def _project_category_corpus(self, corpus, x_dim=0, y_dim=1):
        try:
            import gensim
        except:
            raise Exception('Please install gensim before using Doc2VecCategoryProjector/')
        category_corpus = corpus.use_categories_as_metadata()
        category_counts = corpus.get_term_freq_df('')
        self.doc2vec_builder.train(corpus)
        proj = self.projector.fit_transform(self.doc2vec_builder.project())
        return CategoryProjectionWithDoc2Vec(category_corpus, category_counts, proj, x_dim=x_dim, y_dim=y_dim, doc2vec_model=self.doc2vec_builder)

    def _get_category_metadata_corpus(self, corpus):
        return corpus.use_categories_as_metadata()

    def _get_category_metadata_corpus_and_replace_terms(self, corpus):
        return corpus.use_categories_as_metadata_and_replace_terms()

    def get_category_embeddings(self, corpus):
        return self.doc2vec_builder.project()

