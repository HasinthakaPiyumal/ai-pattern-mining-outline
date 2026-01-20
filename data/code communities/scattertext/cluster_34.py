# Cluster 34

def dataframe_scattertext(corpus: Corpus, plot_df: pd.DataFrame, **kwargs):
    assert 'X' in plot_df
    assert 'Y' in plot_df
    if 'Xpos' not in plot_df:
        plot_df['Xpos'] = Scalers.scale(plot_df['X'])
    if 'Ypos' not in plot_df:
        plot_df['Ypos'] = Scalers.scale(plot_df['Y'])
    use_metadata = kwargs.get('use_non_text_features', False)
    excess_terms = list(set(corpus.get_terms(use_metadata=use_metadata)) - set(plot_df.index))
    if excess_terms:
        print(f'There are {('metadata' if use_metadata else 'terms')} in the corpus which are not in the index of plot_df. These will not be available in the visualization. These are: {excess_terms}.s')
        corpus = corpus.remove_terms(terms=excess_terms, non_text=True)
    plot_df = plot_df.reindex(corpus.get_terms(use_metadata=use_metadata))
    assert len(plot_df) > 0
    if 'term_description_columns' not in kwargs:
        kwargs['term_description_columns'] = [x for x in plot_df.columns if x not in ['X', 'Y', 'Xpos', 'Ypos', 'ColorScore']]
    if 'tooltip_columns' not in kwargs:
        kwargs['tooltip_columns'] = ['Xpos', 'Ypos']
        kwargs['tooltip_column_names'] = {'Xpos': kwargs.get('x_label', 'X'), 'Ypos': kwargs.get('y_label', 'Y')}
    (kwargs.setdefault('metadata', None),)
    (kwargs.setdefault('scores', plot_df['Score'] if 'Score' in plot_df else 0),)
    kwargs.setdefault('minimum_term_frequency', 0)
    kwargs.setdefault('pmi_threshold_coefficient', 0)
    kwargs.setdefault('category', corpus.get_categories()[0])
    kwargs.setdefault('original_x', plot_df['X'].values)
    kwargs.setdefault('original_y', plot_df['Y'].values)
    kwargs.setdefault('x_coords', plot_df['Xpos'].values)
    kwargs.setdefault('y_coords', plot_df['Ypos'].values)
    kwargs.setdefault('use_global_scale', True)
    kwargs.setdefault('ignore_categories', True)
    kwargs.setdefault('unified_context', kwargs['ignore_categories'])
    kwargs.setdefault('show_axes_and_cross_hairs', 0)
    kwargs.setdefault('show_top_terms', False)
    kwargs.setdefault('x_label', 'X')
    kwargs.setdefault('y_label', 'Y')
    return produce_scattertext_explorer(corpus, term_metadata_df=plot_df, **kwargs)

def get_trend_scatterplot_structure(corpus: TermDocMatrix, trend_plot_settings: TrendPlotSettings, d3_url_struct: Optional[D3URLs]=None, non_text: bool=False, plot_height: int=500, plot_width: int=600, show_chart: bool=True, show_category_headings: bool=False, category_order: Optional[List[str]]=None, kwargs: Optional[Dict]=None):
    if kwargs is None:
        kwargs = {}
    add_to_plot_df = {}
    line_df = None
    if isinstance(trend_plot_settings, DispersionPlotSettings):
        dispersion = Dispersion(corpus, use_categories_as_documents=True, non_text=non_text, regressor=trend_plot_settings.regressor, term_ranker=trend_plot_settings.term_ranker)
        dispersion_metric = trend_plot_settings.metric
        terms = dispersion.get_names()
        if trend_plot_settings.use_residual:
            dispersion_df = dispersion.get_adjusted_metric_df(metric=dispersion_metric)
            Y = dispersion_df['Residual']
            YPos = trend_plot_settings.dispersion_scaler(Y)
            line_y = 0.5
        else:
            dispersion_df = dispersion.get_adjusted_metric_df(metric=dispersion_metric)
            Y = dispersion_df['Metric']
            all_scale = trend_plot_settings.dispersion_scaler(np.concatenate([dispersion_df['Metric'].values, dispersion_df['Estimate'].values]))
            YPos = all_scale[:len(dispersion_df)]
            line_y = all_scale[len(dispersion_df):]
        x_axis = trend_plot_settings.get_x_axis(corpus=corpus, non_text=non_text)
        XPos = x_axis.scaled
        X = x_axis.orig
        line_df = pd.DataFrame({'x': x_axis.scaled, 'y': line_y}).sort_values(by='x')
    elif isinstance(trend_plot_settings, CorrelationPlotSettings):
        correlations = Correlations(use_non_text=non_text).set_correlation_type(correlation_type=trend_plot_settings.correlation_type)
        correlation_df = correlations.get_correlation_df(corpus=corpus, document_scores=trend_plot_settings.get_category_ranks(corpus=corpus))
        x_axis = trend_plot_settings.get_x_axis(corpus=corpus, non_text=non_text)
        XPos = x_axis.scaled
        X = x_axis.orig
        line_df = pd.DataFrame({'x': x_axis.scaled, 'y': 0.5}).sort_values(by='x')
        Y = correlation_df[Correlations.get_notation_name(correlation_type=trend_plot_settings.correlation_type)]
        YPos = Scalers.scale_neg_1_to_1_with_zero_mean_abs_max(Y)
        terms = list(correlation_df.index)
    elif isinstance(trend_plot_settings, TimePlotSettings):
        position_df = TimePlotPositioner(corpus=corpus, category_order=trend_plot_settings.category_order, non_text=non_text, dispersion_metric=trend_plot_settings.y_axis_metric, use_residual=trend_plot_settings.use_residual).get_position_df()
        X = position_df.Mean
        XPos = X / corpus.get_num_categories()
        terms = list(position_df.index)
        add_to_plot_df = position_df
        Y = position_df.Dispersion
        YPos = trend_plot_settings.dispersion_scaler(Y)
    else:
        raise Exception('Invalid trend_plot_settings type: ' + str(type(trend_plot_settings)))
    plot_params = trend_plot_settings.get_plot_params()
    plot_df = pd.DataFrame().assign(X=X, Frequency=lambda df: df.X, Xpos=XPos, Y=Y, Ypos=YPos, Color='#ffbf00', term=terms).set_index('term')
    for k, v in add_to_plot_df.items():
        plot_df[k] = v
    kwargs.setdefault('top_terms_left_buffer', 10)
    kwargs.setdefault('ignore_categories', False)
    kwargs.setdefault('unified_context', True)
    kwargs['category_order'] = category_order
    if d3_url_struct is None:
        d3_url_struct = D3URLs()
    scatterplot_structure = dataframe_scattertext(corpus, plot_df=plot_df, x_label=plot_params.x_label, y_label=plot_params.y_label, y_axis_labels=plot_params.y_axis_labels, x_axis_labels=plot_params.x_axis_labels, color_column='Color', tooltip_columns=plot_params.tooltip_columns, tooltip_column_names=plot_params.tooltip_column_names, header_names=plot_params.header_names, left_list_column=plot_params.left_list_column, line_coordinates=line_df.to_dict('records') if line_df is not None else None, use_non_text_features=non_text, return_scatterplot_structure=True, width_in_pixels=plot_width, height_in_pixels=plot_height, d3_url=d3_url_struct.get_d3_url(), d3_scale_chromatic_url=d3_url_struct.get_d3_scale_chromatic_url(), show_chart=show_chart, show_category_headings=show_category_headings, **kwargs)
    return scatterplot_structure

class GraphStructure(object):

    def __init__(self, scatterplot_structure, graph_renderer, scatterplot_width=500, scatterplot_height=700, d3_url_struct=None, protocol='http', template_file_name=None):
        """,
        Parameters
        ----------
        scatterplot_structure: ScatterplotStructure
        graph_renderer: GraphRenderer
        scatterplot_width: int
        scatterplot_height: int
        d3_url_struct: D3URLs
        protocol: str
            http or https
        template_file_name: file name to use as template
        """
        self.graph_renderer = graph_renderer
        self.scatterplot_structure = scatterplot_structure
        self.d3_url_struct = d3_url_struct if d3_url_struct else D3URLs()
        ExternalJSUtilts.ensure_valid_protocol(protocol)
        self.protocol = protocol
        self.scatterplot_width = scatterplot_width
        self.scatterplot_height = scatterplot_height
        self.template_file_name = GRAPH_VIZ_FILE_NAME if template_file_name is None else template_file_name

    def to_html(self):
        """
        Returns
        -------
        str, the html file representation

        """
        javascript_to_insert = self._get_javascript_to_insert()
        autocomplete_css = PackedDataUtils.full_content_of_default_autocomplete_css()
        html_template = self._get_html_template()
        html_content = self._replace_html_template(autocomplete_css, html_template, javascript_to_insert)
        return html_content

    def _get_javascript_to_insert(self):
        return '\n'.join([PackedDataUtils.full_content_of_javascript_files(), self.scatterplot_structure._visualization_data.to_javascript(), self.scatterplot_structure.get_js_to_call_build_scatterplot_with_a_function('termPlotInterface'), PackedDataUtils.javascript_post_build_viz('termSearch', 'plotInterface'), self.graph_renderer.get_javascript()])

    def _replace_html_template(self, autocomplete_css, html_template, javascript_to_insert):
        return HELLO + html_template.replace('/***AUTOCOMPLETE CSS***/', autocomplete_css, 1).replace('<!-- INSERT SCRIPT -->', javascript_to_insert, 1).replace('<!--D3URL-->', self.d3_url_struct.get_d3_url(), 1).replace('<!-- INSERT GRAPH -->', self.graph_renderer.get_graph()).replace('<!--D3SCALECHROMATIC-->', self.d3_url_struct.get_d3_scale_chromatic_url()).replace('<!--USEZOOM-->', self.get_zoom_script_import()).replace('<!--FONTIMPORT-->', self.get_font_import()).replace('http://', self.protocol + '://').replace('{width}', str(self.scatterplot_width)).replace('{height}', str(self.scatterplot_height)).replace('{cellheight}', str(int(self.scatterplot_height * (6 / 12))))

    def get_font_import(self):
        return '<link href="https://fonts.googleapis.com/css?family=IBM+Plex+Sans&display=swap" rel="stylesheet">'

    def get_zoom_script_import(self):
        return '<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.0/dist/svg-pan-zoom.min.js"></script>'

    def _get_html_template(self):
        return PackedDataUtils.get_packaged_html_template_content(self.template_file_name)

class DispersionPlotSettings(TrendPlotSettings):

    def __init__(self, category_order: List=None, metric: str='DA', use_residual: bool=True, term_ranker: Optional[TermRanker]=None, frequency_scaler: Optional[Callable[[np.array], np.array]]=None, dispersion_scaler: Optional[Callable[[np.array], np.array]]=None, regressor: Optional=None):
        TrendPlotSettings.__init__(self, category_order=category_order)
        self.metric = metric
        self.use_residual = use_residual
        self.frequency_scaler = dense_rank if frequency_scaler is None else frequency_scaler
        self.dispersion_scaler = (scale_center_zero_abs if use_residual else scale) if dispersion_scaler is None else dispersion_scaler
        self.term_ranker = AbsoluteFrequencyRanker if term_ranker is None else term_ranker
        self.regressor = MeanIsotonic() if regressor is None else regressor

    def get_plot_params(self) -> TrendPlotPresets:
        return TrendPlotPresets(x_label=' '.join(['Frequency', get_scaler_name(self.frequency_scaler).strip()]), y_label=('Frequency-adjusted ' if self.use_residual else '') + self.metric, x_axis_labels=['Low', 'Medium', 'High'], y_axis_labels=['More Concentrated', 'Medium', 'More Dispersion'], tooltip_column_names={'Frequency': 'Frequency', 'Y': 'Residual ' + self.metric}, tooltip_columns=['Frequency', 'Y'], header_names={'upper': 'Dispersed', 'lower': 'Concentrated'}, left_list_column='Y')

    def get_x_axis(self, corpus: TermDocMatrix, non_text: bool=False) -> ScaledAxis:
        rank_df = self.term_ranker(corpus).set_non_text(non_text=non_text).get_ranks('')
        frequencies = rank_df.sum(axis=1)
        return ScaledAxis(orig=frequencies, scaled=self.frequency_scaler(frequencies))

def get_scaler_name(scaler: Callable) -> str:
    if scaler == dense_rank:
        return 'Dense Rank'
    if scaler == log_scale:
        return 'Log'
    if scaler == scale:
        return ''
    if scaler == sqrt_scale:
        return 'Sqrt'
    return ''

class CorrelationPlotSettings(TrendPlotSettings):

    def __init__(self, category_order: List=None, correlation_type: str='spearmanr', term_ranker: Optional[TermRanker]=None, frequency_scaler: Optional[Callable[[np.array], np.array]]=None):
        TrendPlotSettings.__init__(self, category_order=category_order)
        self.correlation_type = correlation_type
        self.term_ranker = AbsoluteFrequencyRanker if term_ranker is None else term_ranker
        self.frequency_scaler = dense_rank if frequency_scaler is None else frequency_scaler

    def get_plot_params(self) -> TrendPlotPresets:
        return TrendPlotPresets(x_label=' '.join(['Frequency', get_scaler_name(self.frequency_scaler).strip()]), y_label=self.correlation_type.title(), x_axis_labels=['Low', 'Medium', 'High'], y_axis_labels=['Anti-correlated', 'No-correlation', 'Correlated'], tooltip_column_names={'Frequency': 'Frequency', 'Y': Correlations.get_notation_name(self.correlation_type)}, tooltip_columns=['Frequency', 'Y'], header_names={'upper': 'Most correlated', 'lower': 'Most anti-correlated'}, left_list_column='Y')

    def get_x_axis(self, corpus: TermDocMatrix, non_text: bool=False) -> ScaledAxis:
        rank_df = self.term_ranker(corpus).set_non_text(non_text=non_text).get_ranks('')
        frequencies = rank_df.sum(axis=1)
        return ScaledAxis(orig=frequencies, scaled=self.frequency_scaler(frequencies))

class TimePlotSettings(TrendPlotSettings):

    def __init__(self, category_order: List=None, dispersion_metric: str='DA', use_residual: bool=False, dispersion_scaler: Optional[Callable[[np.array], np.array]]=None, term_ranker: Optional[TermRanker]=None, regressor: Optional=None):
        TrendPlotSettings.__init__(self, category_order=category_order)
        self.y_axis_metric = dispersion_metric
        self.dispersion_scaler = dispersion_scaler
        self.use_residual = use_residual
        self.term_ranker = AbsoluteFrequencyRanker if term_ranker is None else term_ranker
        self.regressor = KNeighborsRegressor(weights='distance') if regressor is None else regressor

    def get_plot_params(self) -> TrendPlotPresets:
        return TrendPlotPresets(x_label='Mean Category Position', y_label=' '.join([get_scaler_name(self.dispersion_scaler), ('Residual ' if self.use_residual else '') + self.y_axis_metric]).strip(), y_axis_labels=['High ' + self.y_axis_metric, '', 'Low ' + self.y_axis_metric], x_axis_labels=[self.category_order[0], self.category_order[round(len(self.category_order) / 2)], self.category_order[-1]], tooltip_column_names={'Frequency': 'Frequency', 'MeanCategory': 'Mean'}, tooltip_columns=['Frequency', 'MeanCategory'], header_names={'upper': 'Most Dispersed', 'lower': 'Most Concentrated'}, left_list_column='Y')

class TimePlotPositioner:

    def __init__(self, corpus: TermDocMatrix, category_order: List, non_text: bool=True, dispersion_metric: str='DA', use_residual: bool=False):
        self.corpus = corpus
        self.category_order = category_order
        assert set(category_order) == set(corpus.get_categories())
        self.non_text = non_text
        self.dispersion_metric = dispersion_metric
        self.use_residual = use_residual

    def get_position_df(self) -> pd.DataFrame:
        category_order_idx = IndexStoreFromList.build(self.category_order)
        category_values = np.array([category_order_idx.getidx(v) for v in self.corpus.get_category_names_by_row()])
        tdm = self.corpus.get_term_doc_mat(non_text=self.non_text)
        freq = tdm.sum(axis=0).A1
        dispersion = Dispersion(corpus=self.corpus, non_text=self.non_text, use_categories_as_documents=True)
        if self.use_residual:
            dispersion_df = dispersion.get_adjusted_metric_df(metric=self.dispersion_metric)
            dispersion_value = dispersion_df['Residual'].values
        else:
            dispersion_df = dispersion.get_df(include_da=self.dispersion_metric == 'DA')
            dispersion_value = dispersion_df[self.dispersion_metric].values
        position_df = pd.DataFrame({'Frequency': freq, 'Mean': category_values * tdm / freq, 'term': self.corpus.get_terms(use_metadata=self.non_text), 'Dispersion': dispersion_value}).set_index('term').assign(MeanCategory=lambda df: np.array(self.category_order)[df.Mean.round().astype(int)])
        return position_df

class PairPlotFromScatterplotStructure(object):

    def __init__(self, category_scatterplot_structure, term_scatterplot_structure, category_projection, category_width, category_height, include_category_labels=True, show_halo=True, num_terms=5, d3_url_struct=None, x_dim=0, y_dim=1, protocol='http', term_plot_interface='termPlotInterface', category_plot_interface='categoryPlotInterface'):
        """,
        Parameters
        ----------
        category_scatterplot_structure: ScatterplotStructure
        term_scatterplot_structure: ScatterplotStructure,
        category_projection: CategoryProjection
        category_height: int
        category_width: int
        show_halo: bool
        num_terms: int, default 5
        include_category_labels: bool, default True
        d3_url_struct: D3URLs
        x_dim: int, 0
        y_dim: int, 1
        protocol: str
            http or https
        term_plot_interface : str
        category_plot_interface : str
        """
        self.category_scatterplot_structure = category_scatterplot_structure
        self.term_scatterplot_structure = term_scatterplot_structure
        self.category_projection = category_projection
        self.d3_url_struct = d3_url_struct if d3_url_struct else D3URLs()
        ExternalJSUtilts.ensure_valid_protocol(protocol)
        self.protocol = protocol
        self.category_width = category_width
        self.category_height = category_height
        self.num_terms = num_terms
        self.show_halo = show_halo
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.include_category_labels = include_category_labels
        self.term_plot_interface = term_plot_interface
        self.category_plot_interface = category_plot_interface

    def to_html(self):
        """
        Returns
        -------
        str, the html file representation

        """
        javascript_to_insert = '\n'.join([PackedDataUtils.full_content_of_javascript_files(), self.category_scatterplot_structure._visualization_data.to_javascript('getCategoryDataAndInfo'), self.category_scatterplot_structure.get_js_to_call_build_scatterplot_with_a_function(self.category_plot_interface), self.term_scatterplot_structure._visualization_data.to_javascript('getTermDataAndInfo'), self.term_scatterplot_structure.get_js_to_call_build_scatterplot_with_a_function(self.term_plot_interface), self.term_scatterplot_structure.get_js_reset_function(values_to_set=[self.category_plot_interface, self.term_plot_interface], functions_to_reset=['build' + self.category_plot_interface, 'build' + self.term_plot_interface]), PackedDataUtils.javascript_post_build_viz('categorySearch', self.category_plot_interface), PackedDataUtils.javascript_post_build_viz('termSearch', self.term_plot_interface)])
        autocomplete_css = PackedDataUtils.full_content_of_default_autocomplete_css()
        html_template = self._get_html_template()
        html_content = HELLO + html_template.replace('/***AUTOCOMPLETE CSS***/', autocomplete_css, 1).replace('<!-- INSERT SCRIPT -->', javascript_to_insert, 1).replace('<!--D3URL-->', self.d3_url_struct.get_d3_url(), 1).replace('<!--D3SCALECHROMATIC-->', self.d3_url_struct.get_d3_scale_chromatic_url())
        html_content = html_content.replace('http://', self.protocol + '://')
        if self.show_halo:
            axes_labels = self.category_projection.get_nearest_terms(num_terms=self.num_terms)
            for position, terms in axes_labels.items():
                html_content = html_content.replace('{%s}' % position, self._get_lexicon_html(terms))
        cellheight, cellheightshort = cell_height_and_cell_height_short_from_height(self.category_height)
        return html_content.replace('{width}', str(self.category_width)).replace('{height}', str(self.category_height)).replace('{cellheight}', str(cellheight)).replace('{cellheightshort}', str(cellheightshort))

    def _get_html_template(self):
        if self.show_halo:
            return PackedDataUtils.get_packaged_html_template_content(PAIR_PLOT_HTML_VIZ_FILE_NAME)
        return PackedDataUtils.get_packaged_html_template_content(PAIR_PLOT_WITHOUT_HALO_HTML_VIZ_FILE_NAME)

    def _get_lexicon_html(self, terms):
        lexicon_html = ''
        for i, term in enumerate(terms):
            lexicon_html += '<b>' + ClickableTerms.get_clickable_term(term, self.term_plot_interface) + '</b>'
            if self.include_category_labels:
                category = self.category_projection.category_counts.loc[term].idxmax()
                lexicon_html += ' (<i>%s</i>)' % ClickableTerms.get_clickable_term(category, self.category_plot_interface, self.term_plot_interface)
            if i != len(terms) - 1:
                lexicon_html += ',\n'
        return lexicon_html

