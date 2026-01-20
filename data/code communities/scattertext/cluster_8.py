# Cluster 8

class ScatterChart:

    def __init__(self, term_doc_matrix, verbose=False, **kwargs):
        """
        Parameters
        ----------
        term_doc_matrix: term document matrix to create chart from

        Remaining parameters are from ScatterChartData
        """
        self.term_doc_matrix = term_doc_matrix
        self.scatterchartdata = ScatterChartData(**kwargs)
        self.x_coords = None
        self.y_coords = None
        self.original_x = None
        self.original_y = None
        self._rescale_x = None
        self._rescale_y = None
        self.used = False
        self.metadata_term_lists = None
        self.metadata_descriptions = None
        self.term_colors = None
        self.hidden_terms = None
        self.category_scores = None
        self.verbose = verbose

    def inject_metadata_term_lists(self, term_dict):
        """
        Inserts dictionary of meta data terms into object.

        Parameters
        ----------
        term_dict: dict {metadataname: [term1, term2, ....], ...}

        Returns
        -------
        self: ScatterChart
        """
        check_topic_model_string_format(term_dict)
        if not self.term_doc_matrix.metadata_in_use():
            raise TermDocMatrixHasNoMetadataException('No metadata is present in the term document matrix')
        self.metadata_term_lists = term_dict
        return self

    def inject_category_scores(self, category_scores: Union[np.array, List[List[float]]]) -> Self:
        if type(category_scores) == np.array:
            category_scores = category_scores.tolist()
        if not len(category_scores) == self.term_doc_matrix.get_num_categories():
            raise Exception('Number of rows in category scores must be the number of categories in corpus')
        if not all((len(scores) == self.term_doc_matrix.get_num_terms(non_text=self.scatterchartdata.use_non_text_features) for scores in category_scores)):
            raise Exception('Number of columns in category scores must be the number of terms or metadata in corpus')
        self.category_scores = category_scores
        return self

    def inject_metadata_descriptions(self, term_dict):
        """
        Inserts a set of descriptions of meta data terms.  These will be displayed
        below the scatter plot when a meta data term is clicked. All keys in the term dict
        must occur as meta data.

        Parameters
        ----------
        term_dict: dict {metadataname: str: 'explanation to insert', ...}

        Returns
        -------
        self: ScatterChart
        """
        assert type(term_dict) == dict
        if not self.term_doc_matrix.metadata_in_use():
            raise TermDocMatrixHasNoMetadataException('No metadata is present in the term document matrix')
        if sys.version_info[0] == 2:
            assert set([type(v) for v in term_dict.values()]) - set([str, unicode]) == set()
        else:
            assert set([type(v) for v in term_dict.values()]) - set([str]) == set()
        self.metadata_descriptions = term_dict
        return self

    def inject_term_colors(self, term_to_color_dict):
        """

        :param term_to_color_dict: dict, mapping a term to a color
        :return: self
        """
        self.term_colors = term_to_color_dict

    def inject_coordinates(self, x_coords, y_coords, rescale_x=None, rescale_y=None, original_x=None, original_y=None):
        """
        Inject custom x and y coordinates for each term into chart.

        Parameters
        ----------
        x_coords: array-like
            positions on x-axis \\in [0,1]
        y_coords: array-like
            positions on y-axis \\in [0,1]
        rescale_x: lambda list[0,1]: list[0,1], default identity
            Rescales x-axis after filtering
        rescale_y: lambda list[0,1]: list[0,1], default identity
            Rescales y-axis after filtering
        original_x : array-like, optional
            Original, unscaled x-values.  Defaults to x_coords
        original_y : array-like, optional
            Original, unscaled y-values.  Defaults to y_coords
        Returns
        -------
        self: ScatterChart

        """
        self._verify_coordinates(x_coords, 'x')
        self._verify_coordinates(y_coords, 'y')
        self.x_coords = x_coords
        self.y_coords = y_coords
        self._rescale_x = rescale_x
        self._rescale_y = rescale_y
        self.original_x = x_coords if original_x is None else original_x
        self.original_y = y_coords if original_y is None else original_y

    def _verify_coordinates(self, coords, name):
        if self.scatterchartdata.use_non_text_features and len(coords) != len(self.term_doc_matrix.get_metadata()):
            raise CoordinatesNotRightException('Length of %s_coords must be the same as the number of non-text features in the term_doc_matrix.' % name)
        if not self.scatterchartdata.use_non_text_features and len(coords) != self.term_doc_matrix.get_num_terms():
            raise CoordinatesNotRightException('Length of %s_coords must be the same as the number of terms in the term_doc_matrix.' % name)
        if max(coords) > 1:
            raise CoordinatesNotRightException('Max value of %s_coords must be <= 1.' % name)
        if min(coords) < 0:
            raise CoordinatesNotRightException('Min value of %s_coords must be >= 0.' % name)

    def hide_terms(self, terms):
        """
        Mark terms which won't be displayed in the visualization.

        :param terms: iter[str]
            Terms to mark as hidden.
        :return: ScatterChart
        """
        self.hidden_terms = set(terms)
        return self

    def to_dict(self, category, category_name=None, not_category_name=None, scores=None, transform=percentile_alphabetical, title_case_names=False, not_categories=None, neutral_categories=None, extra_categories=None, background_scorer=None, use_offsets=False, **kwargs):
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
        transform : function, optional
            Function for ranking terms.  Defaults to scattertext.Scalers.percentile_lexicographic.
        title_case_names : bool, default False
          Title case category name and no-category name?
        not_categories : list, optional
            List of categories to use as "not category".  Defaults to all others.
        neutral_categories : list, optional
            List of categories to use as neutral.  Defaults [].
        extra_categories : list, optional
            List of categories to use as extra.  Defaults [].
        background_scorer : CharacteristicScorer, optional
            Used for bg scores

        Returns
        -------
        Dictionary that encodes the scatter chart
        information. The dictionary can be dumped as a json document, and
        used in scattertext.html
         {info: {category_name: ..., not_category_name},
          data: [{term:,
                  x:frequency [0-1],
                  y:frequency [0-1],
                  ox: score,
                  oy: score,
                  s: score,
                  os: original score,
                  p: p-val,
                  cat25k: freq per 25k in category,
                  cat: count in category,
                  ncat: count in non-category,
                  catdocs: [docnum, ...],
                  ncatdocs: [docnum, ...]
                  ncat25k: freq per 25k in non-category}, ...]}}

        """
        if self.used:
            raise Exception('Cannot reuse a ScatterChart constructor')
        if kwargs is not {} and self.verbose:
            logging.info('Excessive arguments passed to ScatterChart.to_dict: ' + str(kwargs))
        all_categories = self.term_doc_matrix.get_categories()
        assert category in all_categories
        if not_categories is None:
            not_categories = [c for c in all_categories if c != category]
            neutral_categories = []
            extra_categories = []
        elif neutral_categories is None:
            neutral_categories = [c for c in all_categories if c not in [category] + not_categories]
            extra_categories = []
        elif extra_categories is None:
            extra_categories = [c for c in all_categories if c not in [category] + not_categories + neutral_categories]
        all_categories = [category] + not_categories + neutral_categories + extra_categories
        df = self._add_x_and_y_coords_to_term_df_if_injected(self._get_term_category_frequencies())
        if scores is None:
            scores = self._get_default_scores(category, not_categories, df)
        category_column_name = str(category) + ' freq'
        df['category score'] = CornerScore.get_scores_for_category(df[category_column_name], df[[str(c) + ' freq' for c in not_categories]].sum(axis=1))
        if self.scatterchartdata.term_significance is not None:
            df['p'] = get_p_vals(df, category_column_name, self.scatterchartdata.term_significance)
        df['not category score'] = CornerScore.get_scores_for_category(df[[str(c) + ' freq' for c in not_categories]].sum(axis=1), df[category_column_name])
        df['color_scores'] = scores
        if self.scatterchartdata.terms_to_include is None and self.scatterchartdata.dont_filter is False:
            df = self._filter_bigrams_by_minimum_not_category_term_freq(category_column_name, not_categories, df)
            df = filter_bigrams_by_pmis(self._filter_by_minimum_term_frequency(all_categories, df), threshold_coef=self.scatterchartdata.pmi_threshold_coefficient)
        if self.scatterchartdata.filter_unigrams:
            df = filter_out_unigrams_that_only_occur_in_one_bigram(df)
        if len(df) == 0:
            raise NoWordMeetsTermFrequencyRequirementsError()
        df['category score rank'] = rankdata(df['category score'], method='ordinal')
        df['not category score rank'] = rankdata(df['not category score'], method='ordinal')
        if self.scatterchartdata.max_terms and self.scatterchartdata.max_terms < len(df):
            assert self.scatterchartdata.max_terms > 0
            df = self._limit_max_terms(category, df)
        df = df.reset_index()
        if self.x_coords is None:
            self.x_coords, self.y_coords = self._get_coordinates_from_transform_and_jitter_frequencies(category, df, not_categories, transform)
            df['x'], df['y'] = (self.x_coords, self.y_coords)
            df['ox'], df['oy'] = (self.x_coords, self.y_coords)
        df['not cat freq'] = df[[str(x) + ' freq' for x in not_categories]].sum(axis=1)
        if neutral_categories != []:
            df['neut cat freq'] = df[[str(x) + ' freq' for x in neutral_categories]].sum(axis=1).fillna(0)
        if extra_categories != []:
            df['extra cat freq'] = df[[str(x) + ' freq' for x in extra_categories]].sum(axis=1).fillna(0)
        json_df = df[['x', 'y', 'ox', 'oy', 'term'] + (['p'] if self.scatterchartdata.term_significance else [])]
        json_df = self._add_term_freq_to_json_df(json_df, df, category).assign(s=self.scatterchartdata.score_transform(df['color_scores']), os=df['color_scores'])
        if background_scorer:
            bg_scores = background_scorer.get_scores(self.term_doc_matrix)
            json_df = json_df.assign(bg=lambda json_df: bg_scores[1].loc[json_df.term].values)
        elif not self.scatterchartdata.use_non_text_features:
            json_df = json_df.assign(bg=lambda json_df: self._get_corpus_characteristic_scores(json_df))
        self._preform_axis_rescale(json_df, self._rescale_x, 'x')
        self._preform_axis_rescale(json_df, self._rescale_y, 'y')
        if self.scatterchartdata.terms_to_include is not None:
            json_df = self._use_only_selected_terms(json_df)
        category_terms = list(json_df.sort_values('s', ascending=False)['term'].iloc[:10])
        not_category_terms = list(json_df.sort_values('s', ascending=True)['term'].iloc[:10])
        if category_name is None:
            category_name = category
        if not_category_name is None:
            not_category_name = 'Not ' + str(category_name)

        def better_title(x):
            if title_case_names:
                return ' '.join([t[0].upper() + t[1:].lower() for t in x.split()])
            else:
                return x
        j = {'info': {'category_name': better_title(category_name), 'not_category_name': better_title(not_category_name), 'category_terms': category_terms, 'not_category_terms': not_category_terms, 'category_internal_name': category, 'not_category_internal_names': not_categories, 'categories': self.term_doc_matrix.get_categories(), 'neutral_category_internal_names': neutral_categories, 'extra_category_internal_names': extra_categories}}
        if self.metadata_term_lists is not None:
            j['metalists'] = self.metadata_term_lists
        if self.metadata_descriptions is not None:
            j['metadescriptions'] = self.metadata_descriptions
        if self.term_colors is not None:
            j['info']['term_colors'] = self.term_colors
        j['data'] = json_df.to_dict(orient='records')
        if self.hidden_terms is not None:
            for term_obj in j['data']:
                if term_obj['term'] in self.hidden_terms:
                    term_obj['display'] = False
        if use_offsets:
            j['offsets'] = self.term_doc_matrix.get_offsets()
        if self.category_scores is not None:
            j['category_scores'] = self.category_scores
        return j

    def _add_x_and_y_coords_to_term_df_if_injected(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.x_coords is not None:
            df = df.assign(x=self.x_coords, y=self.y_coords)
        if not self.original_x is None:
            try:
                df = df.assign(ox=self.original_x.values)
            except AttributeError:
                df = df.assign(ox=self.original_x)
        if not self.original_y is None:
            try:
                df = df.assign(oy=self.original_y.values)
            except AttributeError:
                df = df.assign(oy=self.original_y)
        return df

    def _get_term_category_frequencies(self):
        return self.term_doc_matrix.get_term_category_frequencies(self.scatterchartdata)

    def _use_only_selected_terms(self, json_df):
        term_df = pd.DataFrame({'term': self.scatterchartdata.terms_to_include})
        return pd.merge(json_df, term_df, on='term', how='inner')

    def _preform_axis_rescale(self, json_df, rescaler, variable_to_rescale):
        if rescaler is not None:
            json_df.loc[:, variable_to_rescale] = rescaler(json_df[variable_to_rescale])
            assert json_df[variable_to_rescale].min() >= 0 and json_df[variable_to_rescale].max() <= 1

    def _get_corpus_characteristic_scores(self, json_df):
        bg_terms = self.term_doc_matrix.get_scaled_f_scores_vs_background()
        bg_terms = bg_terms['Scaled f-score']
        bg_terms.name = 'bg'
        bg_terms = bg_terms.reset_index()
        bg_terms.columns = ['term' if x in ['index', 'word'] else x for x in bg_terms.columns]
        json_df = pd.merge(json_df, bg_terms, on='term', how='left')
        return json_df.loc[:, 'bg'].fillna(0)

    def _add_term_freq_to_json_df(self, json_df, term_freq_df, category) -> pd.DataFrame:
        """
        json_df['cat25k'] = (((term_freq_df[category + ' freq'] * 1.
                               / term_freq_df[category + ' freq'].sum()) * 25000).fillna(0)
                             .apply(np.round).astype(int))
        json_df['ncat25k'] = (((term_freq_df['not cat freq'] * 1.
                                / term_freq_df['not cat freq'].sum()) * 25000).fillna(0)
                              .apply(np.round).astype(int))
        """
        json_df = json_df.assign(cat25k=(term_freq_df[str(category) + ' freq'] * 1.0 / term_freq_df[str(category) + ' freq'].sum() * 25000).fillna(0).apply(np.round).astype(int), ncat25k=(term_freq_df['not cat freq'] * 1.0 / term_freq_df['not cat freq'].sum() * 25000).fillna(0).apply(np.round).astype(int), neut25k=0, neut=0, extra25k=0, extra=0)
        if 'neut cat freq' in term_freq_df:
            json_df = json_df.assign(neut25k=(term_freq_df['neut cat freq'] * 1.0 / term_freq_df['neut cat freq'].sum() * 25000).fillna(0).apply(np.round).astype(int), neut=term_freq_df['neut cat freq'])
        if 'extra cat freq' in term_freq_df:
            json_df = json_df.assign(extra25k=(term_freq_df['extra cat freq'] * 1.0 / term_freq_df['extra cat freq'].sum() * 25000).fillna(0).apply(np.round).astype(int), extra=term_freq_df['extra cat freq'])
        return json_df

    def _get_category_names(self, category):
        other_categories = [val + ' freq' for val in self.term_doc_matrix.get_categories() if val != category]
        all_categories = other_categories + [str(category) + ' freq']
        return (all_categories, other_categories)

    def _get_coordinates_from_transform_and_jitter_frequencies(self, category, df, other_categories, transform):
        not_counts = df[[str(c) + ' freq' for c in other_categories]].sum(axis=1)
        counts = df[str(category) + ' freq']
        x_data_raw = transform(not_counts, df.index, counts)
        y_data_raw = transform(counts, df.index, not_counts)
        x_data = self._add_jitter(x_data_raw)
        y_data = self._add_jitter(y_data_raw)
        return (x_data, y_data)

    def _add_jitter(self, vec):
        """
        :param vec: array to jitter
        :return: array, jittered version of arrays
        """
        if self.scatterchartdata.jitter == 0 or self.scatterchartdata.jitter is None:
            return vec
        return vec + np.random.rand(1, len(vec))[0] * self.scatterchartdata.jitter

    def _term_rank_score_and_frequency_df(self, all_categories, category, other_categories, scores):
        df = self._add_x_and_y_coords_to_term_df_if_injected(self._get_term_category_frequencies())
        if scores is None:
            scores = self._get_default_scores(category, other_categories, df)
        category_column_name = str(category) + ' freq'
        df = df['category score'] = CornerScore.get_scores_for_category(df[str(category_column_name)], df[[str(c) + ' freq' for c in other_categories]].sum(axis=1))
        if self.scatterchartdata.term_significance is not None:
            df = df.assign(p=get_p_vals(df, category_column_name, self.scatterchartdata.term_significance))
        df['not category score'] = CornerScore.get_scores_for_category(df[[str(c) + ' freq' for c in other_categories]].sum(axis=1), df[str(category_column_name)])
        df['color_scores', :] = scores
        if self.scatterchartdata.terms_to_include is None and self.scatterchartdata.dont_filter is False:
            df = self._filter_bigrams_by_minimum_not_category_term_freq(category_column_name, other_categories, df)
            df = filter_bigrams_by_pmis(self._filter_by_minimum_term_frequency(all_categories, df), threshold_coef=self.scatterchartdata.pmi_threshold_coefficient)
        if self.scatterchartdata.filter_unigrams:
            df = filter_out_unigrams_that_only_occur_in_one_bigram(df)
        if len(df) == 0:
            raise NoWordMeetsTermFrequencyRequirementsError()
        df['category score rank'] = rankdata(df['category score'], method='ordinal')
        df['not category score rank'] = rankdata(df['not category score'], method='ordinal')
        if self.scatterchartdata.max_terms and self.scatterchartdata.max_terms < len(df):
            assert self.scatterchartdata.max_terms > 0
            df = self._limit_max_terms(category, df)
        df = df.reset_index()
        return df

    def _filter_bigrams_by_minimum_not_category_term_freq(self, category_column_name, other_categories, df):
        if self.scatterchartdata.terms_to_include is None and self.scatterchartdata.dont_filter is False:
            return df[(df[str(category_column_name)] > 0) | (df[[str(c) + ' freq' for c in other_categories]].sum(axis=1) >= self.scatterchartdata.minimum_not_category_term_frequency)]
        else:
            return df

    def _filter_by_minimum_term_frequency(self, all_categories, df):
        if self.scatterchartdata.terms_to_include is None and self.scatterchartdata.dont_filter is False:
            df = df[lambda df: df[[str(c) + ' freq' for c in all_categories]].sum(axis=1) > self.scatterchartdata.minimum_term_frequency]
        return df

    def _limit_max_terms(self, category, df):
        df['score'] = self._term_importance_ranks(category, df)
        df = df.loc[df.sort_values('score').iloc[:self.scatterchartdata.max_terms].index]
        return df[[c for c in df.columns if c != 'score']]

    def _get_default_scores(self, category, other_categories, df):
        category_column_name = str(category) + ' freq'
        cat_word_counts = df[category_column_name]
        not_cat_word_counts = df[[str(c) + ' freq' for c in other_categories]].sum(axis=1)
        scores = RankDifference().get_scores(cat_word_counts, not_cat_word_counts)
        return scores

    def _term_importance_ranks(self, category, df):
        return np.array([df['category score rank'], df['not category score rank']]).min(axis=0)

    def draw(self, category, num_top_words_to_annotate=4, words_to_annotate=[], scores=None, transform=percentile_alphabetical):
        """Outdated.  MPLD3 drawing.

        Parameters
        ----------
        category
        num_top_words_to_annotate
        words_to_annotate
        scores
        transform

        Returns
        -------
        pd.DataFrame, html of fgure
        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise Exception('matplotlib and mpld3 need to be installed to use this function.')
        try:
            from mpld3 import plugins, fig_to_html
        except:
            raise Exception('mpld3 need to be installed to use this function.')
        all_categories, other_categories = self._get_category_names(category)
        df = self._term_rank_score_and_frequency_df(all_categories, category, other_categories, scores)
        if self.x_coords is None:
            df['x'], df['y'] = self._get_coordinates_from_transform_and_jitter_frequencies(category, df, other_categories, transform)
        df_to_annotate = df[(df['not category score rank'] <= num_top_words_to_annotate) | (df['category score rank'] <= num_top_words_to_annotate) | df['term'].isin(words_to_annotate)]
        words = list(df['term'])
        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 'large'}
        fig, ax = plt.subplots()
        plt.figure(figsize=(10, 10))
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.gcf().subplots_adjust(right=0.2)
        points = ax.scatter(self.x_coords, self.y_coords, c=-df['color_scores'], cmap='seismic', s=10, edgecolors='none', alpha=0.9)
        tooltip = plugins.PointHTMLTooltip(points, ['<span id=a>%s</span>' % w for w in words], css='#a {background-color: white;}')
        plugins.connect(fig, tooltip)
        ax.set_ylim([-0.2, 1.2])
        ax.set_xlim([-0.2, 1.2])
        ax.xaxis.set_ticks([0.0, 0.5, 1.0])
        ax.yaxis.set_ticks([0.0, 0.5, 1.0])
        ax.set_ylabel(category.title() + ' Frequency Percentile', fontdict=font, labelpad=20)
        ax.set_xlabel('Not ' + category.title() + ' Frequency Percentile', fontdict=font, labelpad=20)
        for i, row in df_to_annotate.iterrows():
            alignment_criteria = i % 2 == 0
            horizontalalignment = 'right' if alignment_criteria else 'left'
            verticalalignment = 'bottom' if alignment_criteria else 'top'
            term = row['term']
            ax.annotate(term, (self.x_coords[i], self.y_data[i]), size=15, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment)
        plt.show()
        return (df, fig_to_html(fig))

    def to_dict_without_categories(self):
        if self.y_coords is None or self.x_coords is None or self.original_x is None or (self.original_y is None):
            raise NeedToInjectCoordinatesException('This function requires you run inject_coordinates.')
        return {'data': self._add_x_and_y_coords_to_term_df_if_injected(self.term_doc_matrix.get_term_count_df().rename(columns={'corpus': 'cat'}).assign(cat25k=lambda df: (df['cat'] * 1.0 / df['cat'].sum() * 25000).apply(np.round).astype(int))).reset_index().sort_values(by=['x', 'y', 'term']).to_dict(orient='records')}

class CategoryColorAssigner(object):

    def __init__(self, corpus, scorer=RankDifference(), ranker=AbsoluteFrequencyRanker, use_non_text_features=False, color_palette=QUALITATIVE_COLORS):
        """
        Assigns scores to colors for categories

        :param corpus: TermDocMatrix
        :param scorer: scorer
        :param color_palette: list of colors [[red, green, blue], ...]
        """
        self.corpus = corpus
        self.scorer = scorer
        self.color_palette = color_palette
        my_ranker = ranker(corpus)
        if use_non_text_features:
            my_ranker.use_non_text_features()
        tdf = my_ranker.get_ranks()
        tdf_sum = tdf.sum(axis=1)
        term_scores = {}
        for cat in tdf.columns:
            term_scores[cat[:-5]] = pd.Series(self.scorer.get_scores(tdf[cat], tdf_sum - tdf[cat]), index=tdf.index)
        self.term_cat = pd.DataFrame(term_scores).idxmax(axis=1)
        ranked_list_categories = pd.Series(corpus.get_category_names_by_row()).value_counts().index
        self.category_colors = pd.Series(self.color_palette[:len(ranked_list_categories)], index=ranked_list_categories)

    def get_category_colors(self):
        return self.category_colors

    def get_term_colors(self):
        """

        :return: dict, term -> color
        """
        term_color = pd.Series(self.category_colors[self.term_cat].values, index=self.term_cat.index)
        return term_color.apply(get_hex_color).to_dict()

def scale_neg_1_to_1_with_zero_mean_rank_abs_max(v):
    rankv = v * 2 - 1
    pos_v = rankv[rankv > 0]
    pos_v = rankdata(pos_v, 'dense')
    pos_v = pos_v / pos_v.max()
    neg_v = rankv[rankv < 0]
    neg_v = rankdata(neg_v, 'dense')
    neg_v = neg_v / neg_v.max()
    rankv[rankv > 0] = pos_v
    rankv[rankv < 0] = -(neg_v.max() - neg_v)
    return scale_neg_1_to_1_with_zero_mean_abs_max(rankv)

def scale_neg_1_to_1_with_zero_mean_abs_max(vec):
    max_abs = max(vec.max(), -vec.min())
    return (vec > 0).astype(float) * (vec / max_abs) * 0.5 + 0.5 + (vec < 0).astype(float) * (vec / max_abs) * 0.5

def scale_neg_1_to_1_with_zero_mean_log_abs_max(v):
    """
    !!! not working
    """
    df = pd.DataFrame({'v': v, 'sign': (v > 0) * 2 - 1})
    df['lg'] = np.log(np.abs(v)) / np.log(1.96)
    df['exclude'] = np.isinf(df.lg) | np.isneginf(df.lg)
    for mask in [(df['sign'] == -1) & (df['exclude'] == False), (df['sign'] == 1) & (df['exclude'] == False)]:
        df[mask]['lg'] = df[mask]['lg'].max() - df[mask]['lg']
    df['lg'] *= df['sign']
    df['lg'] = df['lg'].fillna(0)
    print(df[df['exclude']]['lg'].values)
    df['to_out'] = scale_neg_1_to_1_with_zero_mean_abs_max(df['lg'])
    print('right')
    print(df.sort_values(by='lg').iloc[:5])
    print(df.sort_values(by='lg').iloc[-5:])
    print('to_out')
    print(df.sort_values(by='to_out').iloc[:5])
    print(df.sort_values(by='to_out').iloc[-5:])
    print(len(df), len(df.dropna()))
    return df['to_out']

def produce_characteristic_explorer(corpus, category, category_name=None, not_category_name=None, not_categories=None, characteristic_scorer=DenseRankCharacteristicness(), term_ranker=termranking.AbsoluteFrequencyRanker, term_scorer=RankDifference(), x_label='Characteristic to Corpus', y_label=None, y_axis_labels=None, scores=None, vertical_lines=None, **kwargs):
    """
    Parameters
    ----------
    corpus : Corpus
        It is highly recommended to use a stoplisted, unigram corpus-- `corpus.get_stoplisted_unigram_corpus()`
    category : str
    category_name : str
    not_category_name : str
    not_categories : list
    characteristic_scorer : CharacteristicScorer
    term_ranker
    term_scorer
    term_acceptance_re : SRE_Pattern
        Regular expression to identify valid terms
    kwargs : dict
        remaining produce_scattertext_explorer keywords

    Returns
    -------
    str HTML of visualization

    """
    if not_categories is None:
        not_categories = [c for c in corpus.get_categories() if c != category]
    category_name, not_category_name = get_category_names(category, category_name, not_categories, not_category_name)
    zero_point, characteristic_scores = characteristic_scorer.get_scores(corpus)
    corpus = corpus.remove_terms(set(corpus.get_terms()) - set(characteristic_scores.index))
    characteristic_scores = characteristic_scores.loc[corpus.get_terms()]
    term_freq_df = term_ranker(corpus).get_ranks()
    scores = term_scorer.get_scores(term_freq_df[str(category) + ' freq'], term_freq_df[[str(c) + ' freq' for c in not_categories]].sum(axis=1)) if scores is None else scores
    scores_scaled_for_charting = scale_neg_1_to_1_with_zero_mean_abs_max(scores)
    html = produce_scattertext_explorer(corpus=corpus, category=category, category_name=category_name, not_category_name=not_category_name, not_categories=not_categories, minimum_term_frequency=0, sort_by_dist=False, x_coords=characteristic_scores, y_coords=scores_scaled_for_charting, y_axis_labels=['More ' + not_category_name, 'Even', 'More ' + category_name] if y_axis_labels is None else y_axis_labels, x_label=x_label, y_label=term_scorer.get_name() if y_label is None else y_label, vertical_lines=[] if vertical_lines is None else vertical_lines, characteristic_scorer=characteristic_scorer, **kwargs)
    return html

def get_category_names(category, category_name, not_categories, not_category_name):
    category_name = str(category_name)
    if category_name is None:
        category_name = category
    if not_category_name is None:
        if not_categories is not None and len(not_categories) == 1:
            not_category_name = not_categories[0]
        else:
            not_category_name = ('Not' if category_name[0].isupper() else 'not') + ' ' + category_name
    return (category_name, not_category_name)

class FourSquareAxes(SemioticSquare):
    """
    This creates a semiotic square where the complex term is considered the "top" category, the
    neutral term is the "bottom" category, the positive dexis is the "left" category, and the
    negative dexis is the "right" category.
    """

    def __init__(self, term_doc_matrix, left_categories, right_categories, top_categories, bottom_categories, left_category_name=None, right_category_name=None, top_category_name=None, bottom_category_name=None, x_scorer=RankDifference(), y_scorer=RankDifference(), term_ranker=AbsoluteFrequencyRanker, labels=None):
        for param in [left_categories, right_categories, top_categories, bottom_categories]:
            assert type(param) == list
            assert set(param) - set(term_doc_matrix.get_categories()) == set()
            assert len(param) > 0
        self.term_doc_matrix_ = term_doc_matrix
        self._labels = labels
        self.left_category_name_ = left_category_name if left_category_name is not None else left_categories[0]
        self.right_category_name_ = right_category_name if right_category_name is not None else right_categories[0]
        self.top_category_name_ = top_category_name if top_category_name is not None else top_categories[0]
        self.bottom_category_name_ = bottom_category_name if bottom_category_name is not None else bottom_categories[0]
        self.x_scorer_ = x_scorer
        self.y_scorer_ = y_scorer
        self.term_ranker_ = term_ranker
        self.left_categories_, self.right_categories_, self.top_categories_, self.bottom_categories_ = (left_categories, right_categories, top_categories, bottom_categories)
        self.axes = self._build_axes()
        self.lexicons = self._build_lexicons()

    def _get_all_categories(self):
        return self.left_categories_ + self.right_categories_ + self.top_categories_ + self.bottom_categories_

    def _build_axes(self, scorer=None):
        tdf = self.term_ranker_(self.term_doc_matrix_).get_ranks()
        tdf.columns = [c[:-5] for c in tdf.columns]
        tdf = tdf[self._get_all_categories()]
        counts = tdf.sum(axis=1)
        tdf['x'] = self.x_scorer_.get_scores(tdf[self.left_categories_].sum(axis=1), tdf[self.right_categories_].sum(axis=1))
        tdf.loc[np.isnan(tdf['x']), 'x'] = self.x_scorer_.get_default_score()
        tdf['y'] = self.y_scorer_.get_scores(tdf[self.top_categories_].sum(axis=1), tdf[self.bottom_categories_].sum(axis=1))
        tdf.loc[np.isnan(tdf['y']), 'y'] = self.y_scorer_.get_default_score()
        tdf['counts'] = counts
        return tdf[['x', 'y', 'counts']]

    def get_labels(self):
        a = self._get_default_a_label()
        b = self._get_default_b_label()
        default_labels = {'a': a, 'not_a': '' if a == '' else 'Not ' + a, 'b': b, 'not_b': '' if b == '' else 'Not ' + b, 'a_and_b': self.top_category_name_, 'not_a_and_not_b': self.bottom_category_name_, 'a_and_not_b': self.left_category_name_, 'b_and_not_a': self.right_category_name_}
        labels = self._labels
        if labels is None:
            labels = {}
        return {name + '_label': labels.get(name, default_labels[name]) for name in default_labels}

    def _get_default_b_label(self):
        return ''

    def _get_default_a_label(self):
        return ''

class SemioticSquare(SemioticSquareBase):
    """
    Create a visualization of a semiotic square.  Requires Corpus to have
    at least three categories.
    >>> newsgroups_train = fetch_20newsgroups(subset='train',
    ...   remove=('headers', 'footers', 'quotes'))
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(newsgroups_train.data)
    >>> corpus = st.CorpusFromScikit(
    ... 	X=X,
    ... 	y=newsgroups_train.target,
    ... 	feature_vocabulary=vectorizer.vocabulary_,
    ... 	category_names=newsgroups_train.target_names,
    ... 	raw_texts=newsgroups_train.data
    ... 	).build()
    >>> semseq = SemioticSquare(corpus,
    ... 	category_a = 'alt.atheism',
    ... 	category_b = 'soc.religion.christian',
    ... 	neutral_categories = ['talk.religion.misc']
    ... )
    >>> # A simple HTML table
    >>> html = SemioticSquareViz(semseq).to_html()
    >>> # The table with an interactive scatterplot below it
    >>> html = st.produce_semiotic_square_explorer(semiotic_square,
    ...                                            x_label='More Atheism, Less Xtnity',
    ...                                            y_label='General Religious Talk')
    """

    def __init__(self, term_doc_matrix, category_a, category_b, neutral_categories, labels=None, term_ranker=AbsoluteFrequencyRanker, scorer=None, non_text=False):
        """
        Parameters
        ----------
        term_doc_matrix : TermDocMatrix
            TermDocMatrix (or descendant) which will be used in constructing square.
        category_a : str
            Category name for term A
        category_b : str
            Category name for term B (in opposition to A)
        neutral_categories : list[str]
            List of category names that A and B will be contrasted to.  Should be in same domain.
        labels : dict
            None by default. Labels are dictionary of {'a_and_b': 'A and B', ...} to be shown
            above each category.
        term_ranker : TermRanker
            Class for returning a term-frequency convention_df
        scorer : termscoring class, optional
            Term scoring class for lexicon mining. Default: `scattertext.termscoring.ScaledFScore`
        non_text : bool, default False
            Use metadata/non-text
        """
        assert category_a in term_doc_matrix.get_categories()
        assert category_b in term_doc_matrix.get_categories()
        for category in neutral_categories:
            assert category in term_doc_matrix.get_categories()
        if len(neutral_categories) == 0:
            raise EmptyNeutralCategoriesError()
        self.category_a_ = category_a
        self.category_b_ = category_b
        self.neutral_categories_ = neutral_categories
        self.non_text = non_text
        self._build_square(term_doc_matrix, term_ranker, labels, scorer)

    def _build_square(self, term_doc_matrix, term_ranker, labels, scorer):
        self.term_doc_matrix_ = term_doc_matrix
        self.term_ranker = term_ranker(term_doc_matrix).set_non_text(non_text=self.non_text)
        self.scorer = RankDifference() if scorer is None else scorer
        self.axes = self._build_axes(scorer)
        self.lexicons = self._build_lexicons()
        self._labels = labels

    def get_axes(self, scorer=None):
        """
        Returns
        -------
        pd.DataFrame
        """
        if scorer:
            return self._build_axes(scorer)
        return self.axes

    def get_lexicons(self, num_terms=10):
        """
        Parameters
        ----------
        num_terms, int

        Returns
        -------
        dict
        """
        return {k: v.index[:num_terms] for k, v in self.lexicons.items()}

    def get_labels(self):
        a = self._get_default_a_label()
        b = self._get_default_b_label()
        default_labels = {'a': a, 'not_a': 'Not ' + a, 'b': b, 'not_b': 'Not ' + b, 'a_and_b': a + ' + ' + b, 'not_a_and_not_b': 'Not ' + a + ' + Not ' + b, 'a_and_not_b': a + ' + Not ' + b, 'b_and_not_a': 'Not ' + a + ' + ' + b}
        labels = self._labels
        if labels is None:
            labels = {}
        return {name + '_label': labels.get(name, default_labels[name]) for name in default_labels}

    def _get_default_b_label(self):
        return self.category_b_

    def _get_default_a_label(self):
        return self.category_a_

    def _build_axes(self, scorer):
        if scorer is None:
            scorer = self.scorer
        tdf = self._get_term_doc_count_df()
        default_score = self.scorer.get_default_score()
        counts = tdf.sum(axis=1)
        tdf['x'] = self._get_x_axis(scorer, tdf)
        tdf.loc[np.isnan(tdf['x']), 'x'] = default_score
        tdf['y'] = self._get_y_axis(scorer, tdf)
        tdf.loc[np.isnan(tdf['y']), 'y'] = default_score
        tdf['counts'] = counts
        if default_score == 0.5:
            tdf['x'] = 2 * tdf['x'] - 1
            tdf['y'] = 2 * tdf['y'] - 1
        return tdf[['x', 'y', 'counts']]

    def _get_x_axis(self, scorer, tdf):
        return scorer.get_scores(tdf[self.category_a_ + ' freq'], tdf[self.category_b_ + ' freq'])

    def _get_y_axis(self, scorer, tdf):
        return scorer.get_scores(tdf[[t + ' freq' for t in [self.category_a_, self.category_b_]]].sum(axis=1), tdf[[t + ' freq' for t in self.neutral_categories_]].sum(axis=1))

    def _get_term_doc_count_df(self):
        return self.term_ranker.get_ranks()[[t + ' freq' for t in self._get_all_categories()]]

    def _get_all_categories(self):
        return [self.category_a_, self.category_b_] + self.neutral_categories_

    def _build_lexicons(self):
        axes_parts_df = add_radial_parts_and_mag_to_term_coordinates(term_coordinates_df=self.axes)
        self.axes['color'] = axes_parts_df.Part.apply(lambda x: HALO_COLORS.get(x.replace('left', 'RIGHT').replace('right', 'left').replace('RIGHT', 'right')))
        self.lexicons = {semiotic_square_label: axes_parts_df[lambda df: df.Part == part].sort_values(by='Mag', ascending=False) for semiotic_square_label, part in SEMIOTIC_SQUARE_TO_PART.items()}
        return self.lexicons

class RankEmbedder(CategoryEmbedderABC):

    def __init__(self, scorer_function: Optional[Callable[[np.array, np.array], np.array]]=None, term_scorer: Optional[CorpusBasedTermScorer]=None, rank_threshold: int=10, term_scorer_kwargs: Optional[Dict]=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorer_function = RankDifference().get_scores if scorer_function is None else scorer_function
        self.term_scorer = term_scorer
        self.rank_threshold = rank_threshold
        self.term_scorer_kwargs = {} if term_scorer_kwargs is None else term_scorer_kwargs

    def embed_categories(self, corpus: TermDocMatrix, non_text: bool=False) -> np.array:
        tdf = corpus.get_freq_df(use_metadata=non_text, label_append='')
        term_freqs = tdf.sum(axis=1)
        score_df = pd.DataFrame({category: pd.Series(self.__get_scores_for_category(str(category), tdf, term_freqs, non_text, corpus), index=corpus.get_terms(use_metadata=non_text)).sort_values(ascending=False).head(self.rank_threshold) for category in corpus.get_categories()})
        return score_df.fillna(0).T.values

    def __get_scores_for_category(self, category, tdf, term_freqs, non_text, corpus):
        if self.term_scorer is not None:
            if inherits_from(self.term_scorer, 'CorpusBasedTermScorer') and type(self.term_scorer) == ABCMeta:
                scorer = self.term_scorer(corpus, **self.term_scorer_kwargs)
            else:
                scorer = self.term_scorer
            if non_text:
                scorer = scorer.use_metadata()
            scorer = scorer.set_categories(category_name=category)
            return scorer.get_scores()
        return self.scorer_function(tdf[str(category)], term_freqs - tdf[str(category)])

class DomainCompactor(object):

    def __init__(self, doc_domains, min_domain_count=None, max_domain_count=None):
        """

		Parameters
		----------
		doc_domains : np.array like
			Length of documents in corpus. Specifies a single domain for each document.
		min_domain_count : int, None
			Term should appear in at least this number of domains
			Default 0
		max_domain_count : int, None
			Term should appear in at most this number of domains
			Default is the number of domains in doc_domains
		"""
        self.doc_domains = doc_domains
        if max_domain_count is None and min_domain_count is None:
            raise NeedsMaxOrMinDomainCountException('Either max_domain_count or min_domain_count must be entered')
        self.min_domain_count = 0 if min_domain_count is None else min_domain_count
        self.max_domain_count = len(doc_domains) if max_domain_count is None else max_domain_count

    def compact(self, term_doc_matrix, non_text=False):
        """
		Parameters
		----------
		term_doc_matrix : TermDocMatrix
			Term document matrix object to compact

		Returns
		-------
		New term doc matrix
		"""
        domain_mat = CombineDocsIntoDomains(term_doc_matrix).get_new_term_doc_mat(self.doc_domains, non_text)
        domain_count = (domain_mat > 0).sum(axis=0)
        valid_term_mask = (self.max_domain_count >= domain_count) & (domain_count >= self.min_domain_count)
        indices_to_compact = np.arange(self._get_num_terms(term_doc_matrix, non_text))[~valid_term_mask.A1]
        return term_doc_matrix.remove_terms_by_indices(indices_to_compact, non_text=non_text)

    def _get_num_terms(self, term_doc_matrix, non_text):
        return term_doc_matrix.get_num_metadata() if non_text else term_doc_matrix.get_num_terms()

class TestDenseRankCharacteristicness(TestCase):

    def test_get_scores(self):
        c = get_hamlet_term_doc_matrix()
        zero_point, scores = DenseRankCharacteristicness().get_scores(c)
        self.assertGreater(zero_point, 0)
        self.assertLessEqual(zero_point, 1)
        self.assertGreater(len(scores), 100)

def get_hamlet_term_doc_matrix():
    hamlet_docs = get_hamlet_docs()
    hamlet_term_doc_matrix = build_from_category_whitespace_delimited_text([(get_hamlet_snippet_binary_category(text), text) for i, text in enumerate(hamlet_docs)])
    return hamlet_term_doc_matrix

class TestScalers(TestCase):

    def test_stretch_0_to_1(self):
        a = np.array([0.8, 0.5, 0.0, -0.2, -0.3, 0.4])
        out = stretch_0_to_1(a)
        np.testing.assert_almost_equal(out, np.array([1.0, 0.8125, 0.5, 0.16666667, 0.0, 0.75]))
        np.testing.assert_almost_equal(a, np.array([0.8, 0.5, 0.0, -0.2, -0.3, 0.4]))
        out = stretch_0_to_1(np.array([]))
        np.testing.assert_almost_equal(out, np.array([]))
        out = stretch_0_to_1(np.array([1, 0.5]))
        np.testing.assert_almost_equal(out, np.array([1.0, 0.75]))
        out = stretch_0_to_1(np.array([-1, -0.5]))
        np.testing.assert_almost_equal(out, np.array([0, 0.25]))

def stretch_0_to_1(vec):
    a = stretch_neg1_to_1(vec)
    return 0.5 * (a + 1)

class TestDomainCompactor(TestCase):

    def test_compact(self):
        hamlet = get_hamlet_term_doc_matrix()
        domains = np.arange(hamlet.get_num_docs()) % 3
        with self.assertRaises(NeedsMaxOrMinDomainCountException):
            hamlet_compact = hamlet.compact(DomainCompactor(domains))
        hamlet_compact = hamlet.compact(DomainCompactor(domains, min_domain_count=2))
        self.assertLess(hamlet_compact.get_num_terms(), hamlet.get_num_terms())
        self.assertEqual(hamlet_compact.get_num_docs(), hamlet.get_num_docs())
        hamlet_compact = hamlet.compact(DomainCompactor(domains, max_domain_count=2))
        self.assertLess(hamlet_compact.get_num_terms(), hamlet.get_num_terms())
        self.assertEqual(hamlet_compact.get_num_docs(), hamlet.get_num_docs())
        hamlet_compact = hamlet.compact(DomainCompactor(domains, max_domain_count=2, min_domain_count=2))
        self.assertLess(hamlet_compact.get_num_terms(), hamlet.get_num_terms())
        self.assertEqual(hamlet_compact.get_num_docs(), hamlet.get_num_docs())

class TestAssociationCompactor(TestCase):

    def test_compact(self):
        term_doc_mat = get_hamlet_term_doc_matrix()
        new_tdm = AssociationCompactor(max_terms=213).compact(term_doc_mat)
        self.assertEqual(len(term_doc_mat.get_terms()), 26875)
        self.assertEqual(len(new_tdm.get_terms()), 213)

    def test_get_term_ranks(self):
        term_doc_mat = get_hamlet_term_doc_matrix()
        ranks = TermCategoryRanker().get_rank_df(term_doc_mat)
        self.assertEqual(len(ranks), term_doc_mat.get_num_terms())
        self.assertGreaterEqual(ranks.min().min(), 0)

    def test_compact_by_rank(self):
        term_doc_mat = get_hamlet_term_doc_matrix()
        compact_tdm4 = AssociationCompactorByRank(rank=4).compact(term_doc_mat)
        compact_tdm8 = AssociationCompactorByRank(rank=8).compact(term_doc_mat)
        self.assertLess(compact_tdm4.get_num_terms(), compact_tdm8.get_num_terms())
        self.assertLess(compact_tdm8.get_num_terms(), term_doc_mat.get_num_terms())

    def test_get_max_rank(self):
        term_doc_mat = get_hamlet_term_doc_matrix()
        self.assertEqual(TermCategoryRanker().get_max_rank(term_doc_mat), 322)

class TestCombineDocsIntoDomains(TestCase):

    def test_get_new_term_doc_mat(self):
        hamlet = get_hamlet_term_doc_matrix()
        domains = np.arange(hamlet.get_num_docs()) % 3
        tdm = CombineDocsIntoDomains(hamlet).get_new_term_doc_mat(domains)
        self.assertEquals(tdm.shape, (3, hamlet.get_num_terms()))
        self.assertEquals(tdm.sum(), hamlet.get_term_doc_mat().sum())

class TestPMIFiltering(TestCase):

    def test_main(self):
        term_doc_mat = get_hamlet_term_doc_matrix()
        pmi_filter = TermDocMatrixFilter(pmi_threshold_coef=4, minimum_term_freq=3)
        filtered_term_doc_mat = pmi_filter.filter(term_doc_mat)
        self.assertLessEqual(len(filtered_term_doc_mat.get_term_freq_df()), len(term_doc_mat.get_term_freq_df()))

    def _test_nothing_passes_filter_raise_error(self):
        term_doc_mat = get_hamlet_term_doc_matrix()
        pmi_filter = TermDocMatrixFilter(pmi_threshold_coef=4000, minimum_term_freq=3000)
        with self.assertRaises(AtLeastOneCategoryHasNoTermsException):
            pmi_filter.filter(term_doc_mat)

    def test_filter_bigrams_by_pmis(self):
        term_doc_mat = get_hamlet_term_doc_matrix()
        df = term_doc_mat.get_term_freq_df()
        filtered_df = filter_bigrams_by_pmis(df, threshold_coef=3)
        self.assertLess(len(filtered_df), len(df))

    def test_unigrams_that_only_occur_in_one_bigram(self):
        bigrams = set(['the cat', 'the saw', 'horses are', 'are pigs', 'pigs horses'])
        expected = {'cat', 'saw'}
        self.assertEqual(expected, unigrams_that_only_occur_in_one_bigram(bigrams))

    def test_filter_out_unigrams_that_only_occur_in_one_bigram(self):
        bigrams = ['the cat', 'the saw', 'horses are', 'are pigs', 'pigs horses']
        df = TermDocMatrixFromPandas(data_frame=pd.DataFrame({'text': bigrams, 'category': ['a', 'a', 'a', 'b', 'b']}), category_col='category', text_col='text', nlp=whitespace_nlp).build().get_term_freq_df()
        new_df = filter_out_unigrams_that_only_occur_in_one_bigram(df)
        self.assertFalse('cat' in new_df.index)
        self.assertFalse('saw' in new_df.index)
        self.assertTrue('the' in new_df.index)
        self.assertTrue('horses' in new_df.index)
        self.assertTrue('pigs' in new_df.index)
        self.assertEqual(set(bigrams) & set(new_df.index), set(bigrams))

def produce_category_focused_pairplot(corpus, category, category_projector=CategoryProjector(projector=TruncatedSVD(20)), category_projection=None, **kwargs):
    """
    Produces a pair-plot which is focused on a single category.

    :param corpus: TermDocMatrix
    :param category: str, name of a category in the corpus
    :param category_projector: CategoryProjector, a factor analysis of the category/feature vector
    :param category_projection: CategoryProjection, None by default. If present, overrides category projector
    :param kwargs: remaining kwargs for produce_pairplot
    :return: str, HTML
    """
    category_num = corpus.get_categories().index(category)
    uncorrelated_components_projection = category_projection
    if category_projection is None:
        if 'use_metadata' in kwargs and kwargs['use_metadata']:
            uncorrelated_components_projection = category_projector.project_with_metadata(corpus)
        else:
            uncorrelated_components_projection = category_projector.project(corpus)
    distances = cosine_distances(uncorrelated_components_projection.get_category_embeddings().T)
    similarity_to_category_scores = -2 * (rankdata(distances[category_num]) - 0.5)
    uncorrelated_components = uncorrelated_components_projection.get_projection()
    least_correlated_dimension = min(([(np.abs(pearsonr(similarity_to_category_scores, uncorrelated_components.T[i])[0]), i)] for i in range(uncorrelated_components.shape[1])))[0][1]
    projection_to_plot = np.array([uncorrelated_components.T[least_correlated_dimension], similarity_to_category_scores]).T
    return produce_pairplot(corpus, initial_category=category, category_projection=uncorrelated_components_projection.use_alternate_projection(projection_to_plot), category_focused=True, **kwargs)

def produce_pairplot(corpus, asian_mode=False, category_width_in_pixels=500, category_height_in_pixels=700, term_width_in_pixels=500, term_height_in_pixels=700, terms_to_show=3000, scaler=scale_neg_1_to_1_with_zero_mean, term_ranker=AbsoluteFrequencyRanker, use_metadata=False, category_projector=CategoryProjector(), category_projection=None, topic_model_term_lists=None, topic_model_preview_size=10, metadata_descriptions=None, initial_category=None, x_dim=0, y_dim=1, show_halo=True, num_terms_in_halo=5, category_color_func='(function(x) {return "#5555FF"})', protocol='https', d3_url_struct=D3URLs(), category_focused=False, verbose=False, use_full_doc=True, default_to_term_comparison=True, category_x_label='', category_y_label='', category_tooltip_func='(function(d) {return d.term})', term_tooltip_func='(function(d) {return d.term})', category_show_axes_and_cross_hairs=False, highlight_selected_category=True, term_x_label=None, term_y_label=None, wordfish_style=False, category_metadata_df=None, return_structure=False, **kwargs):
    if category_projection is None:
        if use_metadata:
            category_projection = category_projector.use_metadata().project_with_metadata(corpus, x_dim=x_dim, y_dim=y_dim)
        else:
            category_projection = category_projector.project(corpus, x_dim=x_dim, y_dim=y_dim)
    if initial_category is None:
        initial_category = corpus.get_categories()[0]
    category_scatter_chart_explorer = _get_category_scatter_chart_explorer(category_projection, scaler, term_ranker, verbose)
    if category_metadata_df is not None:
        if type(category_metadata_df) != pd.DataFrame:
            category_metadata_df = category_metadata_df(corpus)
        category_scatter_chart_explorer = category_scatter_chart_explorer.inject_term_metadata_df(category_metadata_df)
    category_scatter_chart_data = category_scatter_chart_explorer.to_dict(category=initial_category, max_docs_per_category=0)
    term_plot_change_func = _get_term_plot_change_js_func(wordfish_style, category_focused, initial_category)
    category_scatterplot_structure = ScatterplotStructure(VizDataAdapter(category_scatter_chart_data), width_in_pixels=category_width_in_pixels, height_in_pixels=category_height_in_pixels, asian_mode=asian_mode, use_non_text_features=True, show_characteristic=False, x_label=category_x_label, y_label=category_y_label, show_axes_and_cross_hairs=category_show_axes_and_cross_hairs, full_data='getCategoryDataAndInfo()', show_top_terms=False, get_tooltip_content=category_tooltip_func, color_func=category_color_func, show_axes=False, horizontal_line_y_position=0, vertical_line_x_position=0, unified_context=not wordfish_style, show_category_headings=False, show_cross_axes=True, div_name='cat-plot', alternative_term_func=term_plot_change_func, highlight_selected_category=highlight_selected_category)
    compacted_corpus = AssociationCompactor(terms_to_show, use_non_text_features=use_metadata).compact(corpus)
    terms_to_hide = set(corpus.get_terms(use_metadata=use_metadata)) - set(compacted_corpus.get_terms(use_metadata=use_metadata))
    if verbose:
        print('num terms to hide', len(terms_to_hide))
        print('num terms to show', compacted_corpus.get_num_terms())
    term_corpus = category_projection.get_corpus()
    term_scatter_chart_explorer = ScatterChartExplorer(term_corpus, minimum_term_frequency=0, minimum_not_category_term_frequency=0, pmi_threshold_coefficient=0, term_ranker=term_ranker, use_non_text_features=False, add_extra_features=use_metadata, score_transform=stretch_0_to_1, verbose=verbose, dont_filter=True).hide_terms(terms_to_hide)
    if default_to_term_comparison:
        if topic_model_term_lists is not None:
            term_scatter_chart_explorer.inject_metadata_term_lists(topic_model_term_lists)
        if metadata_descriptions is not None:
            term_scatter_chart_explorer.inject_metadata_descriptions(metadata_descriptions)
        if use_metadata:
            tdf = corpus.get_metadata_freq_df('')
        else:
            tdf = corpus.get_term_freq_df('')
        scores = RankDifference().get_scores(tdf[initial_category], tdf[[c for c in corpus.get_categories() if c != initial_category]].sum(axis=1))
        term_scatter_chart_data = term_scatter_chart_explorer.to_dict(category=initial_category, scores=scores, include_term_category_counts=True, transform=dense_rank, **kwargs)
        y_label = (initial_category,)
        x_label = ('Not ' + initial_category,)
        color_func = None
        show_top_terms = True
        show_axes = False
    else:
        term_projection = category_projection.get_term_projection()
        original_x = term_projection['x']
        original_y = term_projection['y']
        x_coords = scaler(term_projection['x'])
        y_coords = scaler(term_projection['y'])
        x_label = term_x_label if term_x_label is not None else ''
        y_label = term_y_label if term_y_label is not None else ''
        show_axes = True
        horizontal_line_y_position = 0
        vertical_line_x_position = 0
        term_scatter_chart_explorer.inject_coordinates(x_coords, y_coords, original_x=original_x, original_y=original_y)
        if topic_model_term_lists is not None:
            term_scatter_chart_explorer.inject_metadata_term_lists(topic_model_term_lists)
        if metadata_descriptions is not None:
            term_scatter_chart_explorer.inject_metadata_descriptions(metadata_descriptions)
        term_scatter_chart_data = term_scatter_chart_explorer.to_dict(category=initial_category, category_name=initial_category, include_term_category_counts=True, **kwargs)
        color_func = '(function(x) {return "#5555FF"})'
        show_top_terms = False
    term_scatterplot_structure = ScatterplotStructure(VizDataAdapter(term_scatter_chart_data), width_in_pixels=term_width_in_pixels, height_in_pixels=term_height_in_pixels, use_full_doc=use_metadata or use_full_doc, asian_mode=asian_mode, use_non_text_features=use_metadata, show_characteristic=False, x_label=x_label, y_label=y_label, full_data='getTermDataAndInfo()', show_top_terms=show_top_terms, get_tooltip_content=term_tooltip_func, color_func=color_func, show_axes=show_axes, topic_model_preview_size=topic_model_preview_size, show_category_headings=False, div_name='d3-div-1', unified_context=not wordfish_style, highlight_selected_category=highlight_selected_category)
    pair_plot_structure = PairPlotFromScatterplotStructure(category_scatterplot_structure, term_scatterplot_structure, category_projection, category_width_in_pixels, category_height_in_pixels, num_terms=num_terms_in_halo, show_halo=show_halo, d3_url_struct=d3_url_struct, x_dim=x_dim, y_dim=y_dim, protocol=protocol)
    if return_structure:
        return pair_plot_structure
    return pair_plot_structure.to_html()

def get_optimal_category_projection(corpus, n_dims=3, n_steps=10, projector=lambda n_terms, n_dims: CategoryProjector(selector=AssociationCompactor(n_terms, scorer=RankDifference), projector=PCA(n_dims)), optimizer: Optional[Callable]=None, term_counts: Optional[np.array]=None, verbose=False):
    min_dev = None
    best_k = None
    best_x = None
    best_y = None
    best_projector = None
    optimizer = ProjectionQuality().ripley_poisson_difference if optimizer is None else optimizer
    term_counts = np.power(2, np.linspace(np.log(corpus.get_num_categories()) / np.log(2), np.log(corpus.get_num_terms()) / np.log(2), n_steps)).astype(int) if term_counts is None else term_counts
    for k in term_counts:
        category_projector = projector(k, n_dims)
        category_projection = category_projector.project(corpus)
        for dim_1 in range(0, n_dims):
            for dim_2 in range(dim_1 + 1, n_dims):
                proj = category_projection.projection[:, [dim_1, dim_2]]
                scaled_proj = np.array([stretch_0_to_1(proj.T[0]), stretch_0_to_1(proj.T[1])]).T
                dev = optimizer(scaled_proj)
                category_projection.x_dim = dim_1
                category_projection.y_dim = dim_2
                tproj = category_projection.get_term_projection().values
                print(proj.shape)
                print(tproj.shape)
                scaled_tproj = np.array([stretch_0_to_1(tproj.T[0]), stretch_0_to_1(tproj.T[1])]).T
                tdev = optimizer(scaled_tproj)
                print(dev, tdev)
                best = False
                if min_dev is None or dev < min_dev:
                    min_dev = dev
                    best_k = k
                    best_projector = category_projector
                    best_x, best_y = (dim_1, dim_2)
                    best = True
                if verbose:
                    print(k, dim_1, dim_2, dev, best_k, best_x, best_y, min_dev, f'best={best}')
    if verbose:
        print(best_k, best_x, best_y)
    return best_projector.project(corpus, best_x, best_y)

def get_optimal_category_projection_by_rank(corpus, n_dims=2, n_steps=20, projector=lambda rank, n_dims: CategoryProjector(AssociationCompactorByRank(rank), projector=PCA(n_dims)), verbose=False):
    try:
        from astropy.stats import RipleysKEstimator
    except:
        raise Exception('Please install astropy')
    ripley = RipleysKEstimator(area=1.0, x_max=1.0, y_max=1.0, x_min=0.0, y_min=0.0)
    min_dev = None
    best_rank = None
    best_x = None
    best_y = None
    best_projector = None
    for rank in np.linspace(1, TermCategoryRanker().get_max_rank(corpus), n_steps):
        r = np.linspace(0, np.sqrt(2), 100)
        category_projector = projector(rank, n_dims)
        category_projection = category_projector.project(corpus)
        for dim_1 in range(0, n_dims):
            for dim_2 in range(dim_1 + 1, n_dims):
                proj = category_projection.projection[:, [dim_1, dim_2]]
                scaled_proj = np.array([stretch_0_to_1(proj.T[0]), stretch_0_to_1(proj.T[1])]).T
                dev = np.sum(np.abs(ripley(scaled_proj, r, mode='ripley') - ripley.poisson(r)))
                if min_dev is None or dev < min_dev:
                    min_dev = dev
                    best_rank = rank
                    best_projector = category_projector
                    best_x, best_y = (dim_1, dim_2)
                if verbose:
                    print('rank', rank, 'dims', dim_1, dim_2, 'K', dev)
                    print('     best rank', best_rank, 'dims', best_x, best_y, 'K', min_dev)
    if verbose:
        print(best_rank, best_x, best_y)
    return best_projector.project(corpus, best_x, best_y)

class RipleyKCategoryProjectorEvaluator(CategoryProjectionEvaluator):

    def __init__(self, max_distance=np.sqrt(2)):
        self.max_distance = max_distance

    def evaluate(self, category_projection):
        assert type(category_projection) == CategoryProjection
        try:
            from astropy.stats import RipleysKEstimator
        except:
            raise Exception('Please install astropy')
        assert issubclass(type(category_projection), CategoryProjectionBase)
        ripley_estimator = RipleysKEstimator(area=1.0, x_max=1.0, y_max=1.0, x_min=0.0, y_min=0.0)
        proj = category_projection.projection[:, [category_projection.x_dim, category_projection.y_dim]]
        scaled_proj = np.array([stretch_0_to_1(proj.T[0]), stretch_0_to_1(proj.T[1])]).T
        radii = np.linspace(0, self.max_distance, 1000)
        deviances = np.abs(ripley_estimator(scaled_proj, radii, mode='ripley') - ripley_estimator.poisson(radii))
        return np.trapz(deviances, x=radii)

class MeanMorisitaIndexEvaluator(CategoryProjectionEvaluator):

    def __init__(self, num_bin_range=None):
        self.num_bin_range = num_bin_range if num_bin_range is not None else [10, 1000]

    def evaluate(self, category_projection):
        assert issubclass(type(category_projection), CategoryProjectionBase)
        proj = category_projection.projection[:, [category_projection.x_dim, category_projection.y_dim]]
        scaled_proj = np.array([stretch_0_to_1(proj.T[0]), stretch_0_to_1(proj.T[1])]).T
        morista_sum = 0
        N = scaled_proj.shape[0]
        for i in range(self.num_bin_range[0], self.num_bin_range[1]):
            bins, _, _ = np.histogram2d(scaled_proj.T[0], scaled_proj.T[1], i)
            Q = len(bins)
            morista_sum += Q * np.sum(np.ravel(bins) * (np.ravel(bins) - 1)) / (N * (N - 1))
        return morista_sum / (self.num_bin_range[1] - self.num_bin_range[0])

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

class SentencesForTopicModeling(object):
    """
	Creates a topic model from a set of key terms based on sentence level co-occurrence.
	"""

    def __init__(self, corpus, use_offsets=False):
        """

		Parameters
		----------
		corpus
		use_offsets

		"""
        assert isinstance(corpus, ParsedCorpus)
        self.corpus = corpus
        self.use_offsets = use_offsets
        if not use_offsets:
            self.termidxstore = corpus._term_idx_store
            matfact = CSRMatrixFactory()
            self.doclabs = []
            self.sentlabs = []
            self.sentdocs = []
            senti = 0
            for doci, doc in enumerate(corpus.get_parsed_docs()):
                for sent in doc.sents:
                    validsent = False
                    for t in sent:
                        try:
                            termi = self.termidxstore.getidxstrict(t.lower_)
                        except:
                            continue
                        if validsent is False:
                            senti += 1
                            self.sentlabs.append(corpus._y[doci])
                            self.sentdocs.append(doci)
                            validsent = True
                        matfact[senti, termi] = 1
            self.sentX = matfact.get_csr_matrix().astype(bool)
        else:
            self.termidxstore = corpus._metadata_idx_store
            doc_sent_offsets = [pd.IntervalIndex.from_breaks([sent[0].idx for sent in doc.sents] + [len(str(doc))], closed='left') for doc_i, doc in enumerate(corpus.get_parsed_docs())]
            doc_sent_count = []
            tally = 0
            for doc_offsets in doc_sent_offsets:
                doc_sent_count.append(tally)
                tally += len(doc_offsets)
            matfact = CSRMatrixFactory()
            for term, term_offsets in corpus.get_offsets().items():
                term_index = corpus.get_metadata_index(term)
                for doc_i, offsets in term_offsets.items():
                    for offset in offsets:
                        doc_sent_i = doc_sent_offsets[doc_i].get_loc(offset[0]) + doc_sent_count[doc_i]
                        matfact[doc_sent_i, term_index] = 1
            self.sentX = matfact.get_csr_matrix()

    def get_sentence_word_mat(self):
        return self.sentX.astype(np.double).tocoo()

    def get_topic_weights_df(self, pipe=None) -> pd.DataFrame:
        pipe = self._fit_model(pipe)
        return pd.DataFrame(pipe._final_estimator.components_.T, index=self.corpus.get_terms(use_metadata=self.use_offsets))

    def get_topics_from_model(self, pipe=None, num_terms_per_topic=10) -> dict:
        """

		Parameters
		----------
		pipe : Pipeline
			For example, `Pipeline([
				('tfidf', TfidfTransformer(sublinear_tf=True)),
				('nmf', (NMF(n_components=30, l1_ratio=.5, random_state=0)))])`
			The last transformer must populate a `components_` attribute when finished.
		num_terms_per_topic : int

		Returns
		-------
		dict: {term: [term1, ...], ...}
		"""
        pipe = self._fit_model(pipe)
        topic_model = {}
        for topic_idx, topic in enumerate(pipe._final_estimator.components_):
            term_list = [self.termidxstore.getval(i) for i in topic.argsort()[:-num_terms_per_topic - 1:-1] if topic[i] > 0]
            if len(term_list) > 0:
                topic_model['%s. %s' % (topic_idx, term_list[0])] = term_list
            else:
                Warning('Topic %s has no terms with scores > 0. Omitting.' % topic_idx)
        return topic_model

    def _fit_model(self, pipe):
        if pipe is None:
            pipe = Pipeline([('tfidf', TfidfTransformer(sublinear_tf=True)), ('nmf', NMF(n_components=30, l1_ratio=0.5, random_state=0))])
        pipe.fit_transform(self.sentX)
        return pipe

    def get_topics_from_terms(self, terms=None, num_terms_per_topic=10, scorer=RankDifference()):
        """
		Parameters
		----------
		terms : list or None
			If terms is list, make these the seed terms for the topoics
			If none, use the first 30 terms in get_scaled_f_scores_vs_background
		num_terms_per_topic : int, default 10
			Use this many terms per topic
		scorer : TermScorer
			Implements get_scores, default is RankDifferce, which tends to work best

		Returns
		-------
		dict: {term: [term1, ...], ...}
		"""
        topic_model = {}
        if terms is None:
            terms = self.corpus.get_scaled_f_scores_vs_background().index[:30]
        for term in terms:
            termidx = self.termidxstore.getidxstrict(term)
            labels = self.sentX[:, termidx].astype(bool).todense().A1
            poscnts = self.sentX[labels, :].astype(bool).sum(axis=0).A1
            negcnts = self.sentX[~labels, :].astype(bool).sum(axis=0).A1
            scores = scorer.get_scores(poscnts, negcnts)
            topic_model[term] = [self.termidxstore.getval(i) for i in np.argsort(-scores)[:num_terms_per_topic]]
        return topic_model

class GanttChart(object):
    """
	Note: the Gantt charts listed here are inspired by
	Dustin Arendt and Svitlana Volkova. ESTEEM: A Novel Framework for Qualitatively Evaluating and
	Visualizing Spatiotemporal Embeddings in Social Media. ACL System Demonstrations. 2017.
	http://www.aclweb.org/anthology/P/P17/P17-4005.pdf

	In order to use the make chart function, Altair must be installed.
	"""

    def __init__(self, term_doc_matrix, category_to_timestep_func, is_gap_between_sequences_func, timesteps_to_lag=4, num_top_terms_each_timestep=10, num_terms_to_include=40, starting_time_step=None, term_ranker=AbsoluteFrequencyRanker, term_scorer=RankDifference()):
        """
		Parameters
		----------
		term_doc_matrix : TermDocMatrix
		category_to_timestep_func : lambda
		is_gap_between_sequences_func : lambda
			timesteps_to_lag : int
		num_top_terms_each_timestep : int
		num_terms_to_include : int
		starting_time_step : object
		term_ranker : TermRanker
		term_scorer : TermScorer
		"""
        self.corpus = term_doc_matrix
        self.timesteps_to_lag = timesteps_to_lag
        self.num_top_terms_each_timestep = num_top_terms_each_timestep
        self.num_terms_to_include = num_terms_to_include
        self.is_gap_between_sequences_func = is_gap_between_sequences_func
        self.category_to_timestep_func = category_to_timestep_func
        self.term_ranker = term_ranker
        self.term_scorer = term_scorer
        categories = list(sorted(self.corpus.get_categories()))
        if len(categories) <= timesteps_to_lag:
            raise Exception('The number of categories in the term doc matrix is <= ' + str(timesteps_to_lag))
        if starting_time_step is None:
            starting_time_step = categories[timesteps_to_lag + 1]
        self.starting_time_step = starting_time_step

    def make_chart(self):
        """
		Returns
		-------
		altair.Chart
		"""
        task_df = self.get_task_df()
        import altair as alt
        chart = alt.Chart(task_df).mark_bar().encode(x='start', x2='end', y='term')
        return chart

    def get_temporal_score_df(self):
        """
		Returns
		-------

		"""
        scoredf = {}
        tdf = self.term_ranker(self.corpus).get_ranks()
        for cat in sorted(self.corpus.get_categories()):
            if cat >= self.starting_time_step:
                negative_categories = self._get_negative_categories(cat, tdf)
                scores = self.term_scorer.get_scores(tdf[cat + ' freq'].astype(int), tdf[negative_categories].sum(axis=1))
                scoredf[cat + ' score'] = scores
                scoredf[cat + ' freq'] = tdf[cat + ' freq'].astype(int)
        return pd.DataFrame(scoredf)

    def _get_negative_categories(self, cat, tdf):
        return sorted([x for x in tdf.columns if x < cat])[-self.timesteps_to_lag:]

    def _get_term_time_df(self):
        data = []
        tdf = self.term_ranker(self.corpus).get_ranks()
        for cat in sorted(self.corpus.get_categories()):
            if cat >= self.starting_time_step:
                negative_categories = self._get_negative_categories(cat, tdf)
                scores = self.term_scorer.get_scores(tdf[cat + ' freq'].astype(int), tdf[negative_categories].sum(axis=1))
                top_term_indices = np.argsort(-scores)[:self.num_top_terms_each_timestep]
                for term in tdf.index[top_term_indices]:
                    data.append({'time': self.category_to_timestep_func(cat), 'term': term, 'top': 1})
        return pd.DataFrame(data)

    def get_task_df(self):
        """
		Returns
		-------

		"""
        term_time_df = self._get_term_time_df()
        terms_to_include = term_time_df.groupby('term')['top'].sum().sort_values(ascending=False).iloc[:self.num_terms_to_include].index
        task_df = term_time_df[term_time_df.term.isin(terms_to_include)][['time', 'term']].groupby('term').apply(lambda x: pd.Series(self._find_sequences(x['time']))).reset_index().rename({0: 'sequence'}, axis=1).reset_index().assign(start=lambda x: x['sequence'].apply(lambda x: x[0])).assign(end=lambda x: x['sequence'].apply(lambda x: x[1]))[['term', 'start', 'end']]
        return task_df

    def _find_sequences(self, time_steps):
        min_timestep = None
        last_timestep = None
        sequences = []
        cur_sequence = []
        for cur_timestep in sorted(time_steps):
            if min_timestep is None:
                cur_sequence = [cur_timestep]
                min_timestep = cur_timestep
            elif not self.is_gap_between_sequences_func(last_timestep, cur_timestep):
                cur_sequence.append(cur_timestep)
                min_timestep = cur_timestep
            else:
                sequences.append([cur_sequence[0], cur_sequence[-1]])
                cur_sequence = [cur_timestep]
            last_timestep = cur_timestep
        if len(cur_sequence) != []:
            sequences.append([cur_sequence[0], cur_sequence[-1]])
        return sequences

