# Cluster 33

def produce_scattertext_table(corpus: TermDocMatrix, num_rows: int=10, non_text: bool=False, plot_width=800, plot_height=600, category_order: Optional[List]=None, heading_categories: Optional[List]=None, heading_category_order: Optional[List]=None, d3_url_struct: Optional[D3URLs]=None, all_category_scorer: Optional[Union[AllCategoryScorer, Callable]]=None, trend_plot_settings: Optional[TrendPlotSettings]=None, show_chart: bool=True, show_category_headings: bool=False, header_clickable: bool=False, category_term_scores: Optional[np.array]=None, category_term_freqs: Optional[np.array]=None, **kwargs):
    """
    Parameters
    -----

    corpus: TermDocMatrix
    num_rows: int, Num rows in table, default 10
    non_text: bool, Use non-text features in table, default False
    plot_width: int, Scatterplot width in pixels, default 800
    plot_height: int, Scatterplot height in pixels, default 600
    category_order: Optional[List], list of categories in chronological order, default to sorted list of cats
    heading_categories: Optional[List], list of new, compacted categories per document
    heading_category_order: Optional[List], order of new, cmpacted categories
    d3_url_struct: Optional[D3URLs]
    all_category_scorer: Optional[Union[AllCategoryScorer, Callable]] = None
    trend_plot_settings: Optional[TrendPlotSettings] = None, default dispersion
    show_chart: bool = True, default, show line chart of ordered categories
    show_category_headings: bool = False, show list of category headings
    header_clickable: bool = False, allow clicking on header to show category scores
    term_category_scores: Optional[np.array]
    term_category_freqs: Optional[np.array]
    """
    alternative_term_func = '(function(termDict) {\n       //document.querySelectorAll(".dotgraph").forEach(svg => svg.style.display = \'none\');\n       //showTermGraph(termDict[\'term\']);\n       //alert(termDict[\'term\'])\n       return true;\n    })'
    if 'use_non_text_features' in kwargs or 'non_text' in kwargs or 'use_metadata' in kwargs:
        non_text = kwargs['use_non_text_features']
    heading_corpus = corpus
    if heading_categories is not None:
        assert len(heading_categories) == corpus.get_num_docs()
        heading_corpus = heading_corpus.recategorize(heading_categories)
    else:
        heading_categories = corpus.get_category_names_by_row()
    if heading_category_order is not None:
        assert set(heading_categories) == set(heading_category_order)
        assert len(heading_category_order) == len(set(heading_categories))
    else:
        heading_category_order = list(sorted(set(heading_categories)))
    table_maker = CategoryTableMaker(corpus=heading_corpus, num_rows=num_rows, non_text=non_text, category_order=heading_category_order, all_category_scorer_factory=all_category_scorer, header_clickable=header_clickable, term_category_scores=category_term_scores, term_category_freqs=category_term_freqs)
    if header_clickable:
        kwargs['include_term_category_counts'] = True
    if trend_plot_settings is None:
        trend_plot_settings = DispersionPlotSettings(metric='DA', category_order=category_order)
    scatterplot_structure = get_trend_scatterplot_structure(corpus=corpus, trend_plot_settings=trend_plot_settings, d3_url_struct=d3_url_struct, non_text=non_text, plot_width=plot_width, plot_height=plot_height, show_chart=show_chart, show_category_headings=show_category_headings, category_order=heading_category_order, kwargs=kwargs)
    html = TableStructure(scatterplot_structure, graph_renderer=table_maker, d3_url_struct=d3_url_struct).to_html()
    return html

