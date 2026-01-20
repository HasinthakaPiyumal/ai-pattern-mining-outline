# Cluster 100

class CategoryProjectionBase(ABC):
    """

    """

    def _pseduo_init(self, category_corpus, category_counts, projection, x_dim=0, y_dim=1, term_projection=None):
        self.category_corpus = category_corpus
        self.category_counts = category_counts
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.projection = projection
        self.term_projection = term_projection

    def project_with_alternative_dimensions(self, x_dim, y_dim):
        return CategoryProjection(self.category_corpus, self.category_counts, self.projection, x_dim, y_dim)

    def project_with_alternate_axes(self, x_axis=None, y_axis=None):
        if x_axis is None:
            x_axis = self._get_x_axis()
        if y_axis is None:
            y_axis = self._get_y_axis()
        return CategoryProjectionAlternateAxes(self.category_corpus, self.category_counts, self.projection, self.get_category_embeddings(), self.x_dim, self.y_dim, x_axis=x_axis, y_axis=y_axis)

    def get_pandas_projection(self):
        """

        :param x_dim: int
        :param y_dim: int
        :return: pd.DataFrame
        """
        to_ret = pd.DataFrame({'term': self.category_corpus.get_metadata(), 'x': self._get_x_axis(), 'y': self._get_y_axis()}).set_index('term')
        return to_ret

    def _get_x_axis(self):
        try:
            return self.projection.T[self.x_dim]
        except:
            return self.projection.T.loc[self.x_dim]

    def _get_y_axis(self):
        try:
            return self.projection.T[self.y_dim]
        except:
            try:
                return self.projection.T.loc[self.y_dim]
            except:
                import pdb
                pdb.set_trace()
                raise e

    def get_axes_labels(self, num_terms=5):
        df = self.get_term_projection()
        return {'right': list(df.sort_values(by='x', ascending=False).index[:num_terms]), 'left': list(df.sort_values(by='x', ascending=True).index[:num_terms]), 'top': list(df.sort_values(by='y', ascending=False).index[:num_terms]), 'bottom': list(df.sort_values(by='y', ascending=True).index[:num_terms])}

    def get_nearest_terms(self, num_terms: int=5) -> dict:
        return term_coordinates_to_halo(term_coordinates_df=self.get_term_projection(), num_terms=num_terms)

    def get_term_projection(self):
        if self.term_projection is None:
            dim_term = np.matmul(self.category_counts.values, self._get_x_y_projection())
        else:
            dim_term = self.term_projection
        df = pd.DataFrame(dim_term, index=self.category_corpus.get_terms(), columns=['x', 'y'])
        return df

    def _get_x_y_projection(self):
        return np.array([self._get_x_axis(), self._get_y_axis()]).T

    def get_projection(self):
        return self.projection

    @abstractmethod
    def use_alternate_projection(self, projection):
        pass

    @abstractmethod
    def get_category_embeddings(self):
        pass

    def get_corpus(self):
        return self.category_corpus

def term_coordinates_to_halo(term_coordinates_df: pd.DataFrame, num_terms: int=5) -> dict:
    return dict(add_radial_parts_and_mag_to_term_coordinates(term_coordinates_df=term_coordinates_df).sort_values(by='Mag', ascending=False).groupby('Part').apply(lambda gdf: list(gdf.iloc[:num_terms].index)))

