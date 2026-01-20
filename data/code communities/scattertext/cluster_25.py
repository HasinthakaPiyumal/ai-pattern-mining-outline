# Cluster 25

def produce_scattertext_explorer(corpus: object, category: object, category_name: object=None, not_category_name: object=None, protocol: object='https', pmi_threshold_coefficient: object=DEFAULT_MINIMUM_TERM_FREQUENCY, minimum_term_frequency: object=DEFAULT_PMI_THRESHOLD_COEFFICIENT, minimum_not_category_term_frequency: object=0, max_terms: object=None, filter_unigrams: object=False, height_in_pixels: object=None, width_in_pixels: object=None, max_snippets: object=None, max_docs_per_category: object=None, metadata: object=None, scores: object=None, x_coords: object=None, y_coords: object=None, original_x: object=None, original_y: object=None, rescale_x: object=None, rescale_y: object=None, singleScoreMode: object=False, sort_by_dist: object=False, reverse_sort_scores_for_not_category: object=True, use_full_doc: object=False, transform: object=percentile_alphabetical, jitter: object=0, gray_zero_scores: object=False, term_ranker: object=None, asian_mode: object=False, match_full_line: object=False, use_non_text_features: object=False, show_top_terms: object=True, show_characteristic: object=None, word_vec_use_p_vals: object=False, max_p_val: object=0.1, p_value_colors: object=False, term_significance: object=None, save_svg_button: object=False, x_label: object=None, y_label: object=None, d3_url: object=None, d3_scale_chromatic_url: object=None, pmi_filter_thresold: object=None, alternative_text_field: object=None, terms_to_include: object=None, semiotic_square: object=None, num_terms_semiotic_square: object=None, not_categories: object=None, neutral_categories: object=[], extra_categories: object=[], show_neutral: object=False, neutral_category_name: object=None, get_tooltip_content: object=None, x_axis_values: object=None, y_axis_values: object=None, x_axis_values_format: object=None, y_axis_values_format: object=None, color_func: object=None, term_scorer: object=None, term_scorer_kwargs: object=None, show_axes: object=True, show_axes_and_cross_hairs: object=False, show_diagonal: object=False, use_global_scale: object=False, horizontal_line_y_position: object=None, vertical_line_x_position: object=None, show_cross_axes: object=True, show_extra: object=False, extra_category_name: object=None, censor_points: object=True, center_label_over_points: object=False, x_axis_labels: object=None, y_axis_labels: object=None, topic_model_term_lists: object=None, topic_model_preview_size: object=10, metadata_descriptions: object=None, vertical_lines: object=None, characteristic_scorer: object=None, term_colors: object=None, unified_context: object=False, show_category_headings: object=True, highlight_selected_category: object=False, include_term_category_counts: object=False, div_name: object=None, alternative_term_func: object=None, term_metadata: object=None, term_metadata_df: object=None, max_overlapping: object=-1, include_all_contexts: object=False, show_corpus_stats: object=True, sort_doc_labels_by_name: object=False, enable_term_category_description: object=True, always_jump: object=True, get_custom_term_html: object=None, header_names: object=None, header_sorting_algos: object=None, ignore_categories: object=False, d3_color_scale: object=None, background_labels: object=None, tooltip_columns: object=None, tooltip_column_names: object=None, term_description_columns: object=None, term_description_column_names: object=None, term_word_in_term_description: object='Term', color_column: object=None, color_score_column: object=None, label_priority_column: object=None, text_color_column: object=None, text_size_column: object=None, suppress_text_column: object=None, background_color: object=None, left_list_column: object=None, censor_point_column: object=None, right_order_column: object=None, line_coordinates: object=None, subword_encoding: object=None, top_terms_length: object=14, top_terms_left_buffer: object=0, dont_filter: object=False, use_offsets: object=False, get_column_header_html: object=None, show_term_etc: object=True, sort_contexts_by_meta: object=False, show_chart: object=False, return_data: object=False, suppress_circles: object=False, category_colors: object=None, document_word: object='document', document_word_plural: object=None, category_order: object=None, include_gradient: bool=False, left_gradient_term: Optional[str]=None, middle_gradient_term: Optional[str]=None, right_gradient_term: Optional[str]=None, gradient_text_color: Optional[str]=None, gradient_colors: Optional[List[str]]=None, category_term_scores: Optional[list[List[float]]]=None, category_term_score_scaler: Optional[str]=None, return_scatterplot_structure: object=False) -> object:
    """Returns html code of visualization.

    Parameters
    ----------
    corpus : Corpus
        Corpus to use.
    category : str
        Name of category column as it appears in original data frame.
    category_name : str
        Name of category to use.  E.g., "5-star reviews."
        Optional, defaults to category name.
    not_category_name : str
        Name of everything that isn't in category.  E.g., "Below 5-star reviews".
        Optional defaults to "N(n)ot " + category_name, with the case of the 'n' dependent
        on the case of the first letter in category_name.
    protocol : str, optional
        Protocol to use.  Either http or https.  Default is https.
    pmi_threshold_coefficient : int, optional
        Filter out bigrams with a PMI of < 2 * pmi_threshold_coefficient. Default is 6
    minimum_term_frequency : int, optional
        Minimum number of times word needs to appear to make it into visualization.
    minimum_not_category_term_frequency : int, optional
      If an n-gram does not occur in the category, minimum times it
       must be seen to be included. Default is 0.
    max_terms : int, optional
        Maximum number of terms to include in visualization.
    filter_unigrams : bool, optional
        Default False, do we filter out unigrams that only occur in one bigram
    width_in_pixels : int, optional
        Width of viz in pixels, if None, default to JS's choice
    height_in_pixels : int, optional
        Height of viz in pixels, if None, default to JS's choice
    max_snippets : int, optional
        Maximum number of snippets to show when term is clicked.  If None, all are shown.
    max_docs_per_category: int, optional
        Maximum number of documents to store per category.  If None, by default, all are stored.
    metadata : list or function, optional
        list of meta data strings that will be included for each document, if a function, called on corpus
    scores : np.array, optional
        Array of term scores or None.
    x_coords : np.array, optional
        Array of term x-axis positions or None.  Must be in [0,1].
        If present, y_coords must also be present.
    y_coords : np.array, optional
        Array of term y-axis positions or None.  Must be in [0,1].
        If present, x_coords must also be present.
    original_x : array-like
        Original, unscaled x-values.  Defaults to x_coords
    original_y : array-like
        Original, unscaled y-values.  Defaults to y_coords
    rescale_x : lambda list[0,1]: list[0,1], optional
        Array of term x-axis positions or None.  Must be in [0,1].
        Rescales x-axis after filtering
    rescale_y : lambda list[0,1]: list[0,1], optional
        Array of term y-axis positions or None.  Must be in [0,1].
        Rescales y-axis after filtering
    singleScoreMode : bool, optional
        Label terms based on score vs distance from corner.  Good for topic scores. Show only one color.
    sort_by_dist: bool, optional
        Label terms based distance from corner. True by default.  Negated by singleScoreMode.
    reverse_sort_scores_for_not_category: bool, optional
        If using a custom score, score the not-category class by
        lowest-score-as-most-predictive. Turn this off for word vector
        or topic similarity. Default True.
    use_full_doc : bool, optional
        Use the full document in snippets.  False by default.
    transform : function, optional
        not recommended for editing.  change the way terms are ranked.  default is st.Scalers.percentile_ordinal
    jitter : float, optional
        percentage of axis to jitter each point.  default is 0.
    gray_zero_scores : bool, optional
        If True, color points with zero-scores a light shade of grey.  False by default.
    term_ranker : TermRanker, optional
        TermRanker class for determining term frequency ranks.
    asian_mode : bool, optional
        Use a special Javascript regular expression that's specific to chinese or japanese
    match_full_line : bool, optional
        Has the javascript regex match the full line instead of part of it
    use_non_text_features : bool, optional
        Show non-bag-of-words features (e.g., Empath) instead of text.  False by default.
    show_top_terms : bool, default True
        Show top terms on the left-hand side of the visualization
    show_characteristic: bool, default None
        Show characteristic terms on the far left-hand side of the visualization
    word_vec_use_p_vals: bool, default False
        Sort by harmonic mean of score and distance.
    max_p_val : float, default 0.1
        If word_vec_use_p_vals, the minimum p val to use.
    p_value_colors : bool, default False
      Color points differently if p val is above 1-max_p_val, below max_p_val, or
       in between.
    term_significance : TermSignificance instance or None
        Way of getting signfiance scores.  If None, p values will not be added.
    save_svg_button : bool, default False
        Add a save as SVG button to the page.
    x_label : str, default None
        Custom x-axis label
    y_label : str, default None
        Custom y-axis label
    d3_url, str, None by default.  The url (or path) of d3.
        URL of d3, to be inserted into <script src="..."/>.  Overrides `protocol`.
      By default, this is `DEFAULT_D3_URL` declared in `ScatterplotStructure`.
    d3_scale_chromatic_url, str, None by default.  Overrides `protocol`.
      URL of d3 scale chromatic, to be inserted into <script src="..."/>
      By default, this is `DEFAULT_D3_SCALE_CHROMATIC` declared in `ScatterplotStructure`.
    pmi_filter_thresold : (DEPRECATED) int, None by default
      DEPRECATED.  Use pmi_threshold_coefficient instead.
    alternative_text_field : str or None, optional
        Field in from dataframe used to make corpus to display in place of parsed text. Only
        can be used if corpus is a ParsedCorpus instance.
    terms_to_include : list or None, optional
        Whitelist of terms to include in visualization.
    semiotic_square : SemioticSquareBase
        None by default.  SemioticSquare based on corpus.  Includes square above visualization.
    num_terms_semiotic_square : int
        10 by default. Number of terms to show in semiotic square.
        Only active if semiotic square is present.
    not_categories : list
        All categories other than category by default.  Documents labeled
        with remaining category.
    neutral_categories : list
        [] by default.  Documents labeled neutral.
    extra_categories : list
        [] by default.  Documents labeled extra.
    show_neutral : bool
        False by default.  Show a third column listing contexts in the
        neutral categories.
    neutral_category_name : str
        "Neutral" by default. Only active if show_neutral is True.  Name of the neutral
        column.
    get_tooltip_content : str
        Javascript function to control content of tooltip.  Function takes a parameter
        which is a dictionary entry produced by `ScatterChartExplorer.to_dict` and
        returns a string.
    x_axis_values : list, default None
        Value-labels to show on x-axis. Low, medium, high are defaults.
    y_axis_values : list, default None
        Value-labels to show on y-axis. Low, medium, high are defaults.
    x_axis_values_format : str, default None
        d3 format of x-axis values
    y_axis_values_format : str, default None
        d3 format of y-axis values
    color_func : str, default None
        Javascript function to control color of a point.  Function takes a parameter
        which is a dictionary entry produced by `ScatterChartExplorer.to_dict` and
        returns a string.
    term_scorer : Object, default None
        In lieu of scores, object with a get_scores(a,b) function that returns a set of scores,
        where a and b are term counts.  Scorer optionally has a get_term_freqs function. Also could be a
        CorpusBasedTermScorer instance.
    term_scorer_kwargs : Optional[Dict], default None
        Arguments to be placed in the term_scorer constructor after the corpus
    show_axes : bool, default True
        Show the ticked axes on the plot.  If false, show inner axes as a crosshair.
    show_axes_and_cross_hairs : bool, default False
        Show both peripheral axis labels and cross axes.
    show_diagonal : bool, default False
        Show a diagonal line leading from the lower-left ot the upper-right; only makes
        sense to use this if use_global_scale is true.
    use_global_scale : bool, default False
        Use same scale for both axes
    vertical_line_x_position : float, default None
    horizontal_line_y_position : float, default None
    show_cross_axes : bool, default True
        If show_axes is False, do we show cross-axes?
    show_extra : bool
        False by default.  Show a fourth column listing contexts in the
        extra categories.
    extra_category_name : str, default None
        "Extra" by default. Only active if show_neutral is True and show_extra is True.  Name
        of the extra column.
    censor_points : bool, default True
        Don't label over points.
    center_label_over_points : bool, default False
        Center a label over points, or try to find a position near a point that
        doesn't overlap anything else.
    x_axis_labels: list, default None
        List of string value-labels to show at evenly spaced intervals on the x-axis.
        Low, medium, high are defaults.
    y_axis_labels : list, default None
        List of string value-labels to show at evenly spaced intervals on the y-axis.
        Low, medium, high are defaults.
    topic_model_term_lists : dict default None
        Dict of metadata name (str) -> List of string terms in metadata. These will be bolded
        in query in context results.
    topic_model_preview_size : int default 10
        Number of terms in topic model to show as a preview.
    metadata_descriptions : dict default None
        Dict of metadata name (str) -> str of metadata description. These will be shown when a meta data term is
        clicked.
    vertical_lines : list default None
        List of floats corresponding to points on the x-axis to draw vertical lines
    characteristic_scorer : CharacteristicScorer default None
        Used for bg scores
    term_colors : dict, default None
        Dictionary mapping term to color
    unified_context : bool, default False
        Boolean displays contexts in a single pane as opposed to separate columns.
    show_category_headings : bool, default True
        Show category headings if unified_context is True.
    highlight_selected_category : bool, default False
        Highlight selected category if unified_context is True.
    include_term_category_counts : bool, default False
        Include the termCounts object in the plot definition.
    div_name : str, None by default
        Give the scatterplot div name a non-default value
    alternative_term_func: str, default None
        Javascript function which take a term JSON object and returns a bool.  If the return value is true,
        execute standard term click pipeline. Ex.: `'(function(termDict) {return true;})'`.
    term_metadata : dict, None by default
        Dict mapping terms to dictionaries containing additional information which can be used in the color_func
        or the get_tooltip_content function. These will appear in termDict.etc
    term_metadata_df : pd.DataFrame, None by default
        Dataframe version of term_metadata
    include_all_contexts: bool, default False
        Include all contexts, even non-matching ones, in interface
    max_overlapping: int, default -1
        Number of overlapping terms to dislay. If -1, display all. (default)
    show_corpus_stats: bool, default True
        Show the corpus stats div
    sort_doc_labels_by_name: bool default False
        If unified, sort the document labels by name
    always_jump: bool, default True
        Always jump to term contexts if a term is clicked
    enable_term_category_description: bool, default True
        List term/metadata statistics under category
    get_custom_term_html: str, default None
        Javascript function which displays term summary from term info
    header_names: Dict[str, str], default None
        Dictionary giving names of term lists shown to the right of the plot. Valid keys are
        upper, lower and right.
    header_sorting_algos: Dict[str, str], default None
        Dictionary giving javascript sorting algorithms for panes. Valid keys are upper, lower
        and right. Value is a JS function which takes the "data" object.
    ignore_categories: bool, default False
        Signals the plot shouldn't display category names. Used in single category plots.
    suppress_text_column: str, default None
        Column in term_metadata_df which indicates term should be hidden
    left_list_column: str, default None
        Column in term_metadata_df which should be used for sorting words into upper and lower
        parts of left word-list sections. Highest values in upper, lowest in lower.
    tooltip_columns: List[str]
    tooltip_column_names: Dict[str, str]
    term_description_columns: List[str]
    term_description_column_names: Dict[str]
    term_word_in_term_description: str, default None
    color_column: str, default None:
        column in term_metadata_df which indicates color
    color_score_column: str, default None
        column in term_metadata df; contains value between 0 and 1 which will be used to assign a color
    label_priority_column : str, default None
        Column in term_metadata_df; larger values in the column indicate a term should be labeled first
    censor_point_column : str, default None
        Should we allow labels to be drawn over point?
    right_order_column : str, default None
        Order for right column ("characteristic" by default); largest first
    background_color : str, default None
        Changes document.body's background color to background_color
    line_coordinates : list, default None
        Coordinates for drawing a line under the plot
    subword_encoding : str, default None
        Type of subword encoding to use, None if none, currently supports "RoBERTa"
    top_terms_length : int, default 14
        Number of words to list in most/least associated lists on left-hand side
    top_terms_left_buffer : int, default 0
        Number of pixels left to shift top terms list
    dont_filter : bool, default False
        Don't filter any terms when charting
    get_column_header_html : str, default None
        Javascript function to return html over each column. Matches header
        (Column Name, occurrences per 25k, occs, # occs * 1000/num docs, term info)
    show_term_etc: bool, default True
        Shows list of etc values after clicking term
    use_offsets : bool, default False
        Enable the use of metadata offsets
    sort_contexts_by_meta : bool, default False
        Sort context by meta instead of match strength
    suppress_circles : bool, default False
        Label terms over circles and hide circless
    show_chart : bool, default False
        Show line chart if unified context is true
    return_data : bool default False
        Return a dict containing the output of `ScatterChartExplosrer.to_dict` instead of
        an html.
    category_colors : dict, optional defaut None
        Dictionary matching category names to colors
    document_word : str, default "document"
    document_word_plural : Optional[str], default "document"
    category_order : Optional[list[str]], default None
        Order of categories in line chart
    include_gradient : bool, False
        Include gradient at the top of the chart
    left_gradient_term : Optional[str], None by default
        Text of left gradient label. category_name by default
    middle_gradient_term : Optional[str], None by default
        Text of middle grad label. If None, not shown.
    right_gradient_term: Optional[str], None by default
        Text of right gradient label, not_category_name by default
    gradient_text_color: str, white by default
        Color of text in gradient
    gradient_colors: Optional[List[str]], None by default, follows d3_color_scale
        Colors of gradient, as a list of hex values (e.g, ['#0000ff', '#fe0100', '#00ff00'])
    category_term_scores: Optional[List[List[float]], None by default
        score[category, term] for table visualization
    category_term_score_scaler: Optional[str], None by default
        Javascript function which scales a set of categories scores to between 0 and 1
    return_scatterplot_structure : bool, default False
        return ScatterplotStructure instead of html

    Returns
    -------
    str
    html of visualization

    """
    if singleScoreMode or word_vec_use_p_vals:
        d3_color_scale = 'd3.interpolatePurples'
    if singleScoreMode or not sort_by_dist:
        sort_by_dist = False
    else:
        sort_by_dist = True
    if term_ranker is None:
        term_ranker = termranking.AbsoluteFrequencyRanker
    category_name, not_category_name = get_category_names(category, category_name, not_categories, not_category_name)
    if not_categories is None:
        not_categories = [c for c in corpus.get_categories() if c != category]
    term_scorer = _initialize_term_scorer_if_needed(category, corpus, neutral_categories, not_categories, show_neutral, term_scorer, use_non_text_features, term_ranker, term_scorer_kwargs)
    if term_scorer:
        scores = get_term_scorer_scores(category, corpus, neutral_categories, not_categories, show_neutral, term_ranker, term_scorer, use_non_text_features)
    if pmi_filter_thresold is not None:
        pmi_threshold_coefficient = pmi_filter_thresold
        warnings.warn("The argument name 'pmi_filter_thresold' has been deprecated. Use 'pmi_threshold_coefficient' in its place", DeprecationWarning)
    if use_non_text_features:
        pmi_threshold_coefficient = 0
    scatter_chart_explorer = ScatterChartExplorer(corpus, minimum_term_frequency=minimum_term_frequency, minimum_not_category_term_frequency=minimum_not_category_term_frequency, pmi_threshold_coefficient=pmi_threshold_coefficient, filter_unigrams=filter_unigrams, jitter=jitter, max_terms=max_terms, term_ranker=term_ranker, use_non_text_features=use_non_text_features, term_significance=term_significance, terms_to_include=terms_to_include, dont_filter=dont_filter)
    if x_coords is None and y_coords is not None or (y_coords is None and x_coords is not None):
        raise Exception('Both x_coords and y_coords need to be passed or both left blank')
    if x_coords is not None:
        scatter_chart_explorer.inject_coordinates(x_coords, y_coords, rescale_x=rescale_x, rescale_y=rescale_y, original_x=original_x, original_y=original_y)
    if topic_model_term_lists is not None:
        scatter_chart_explorer.inject_metadata_term_lists(topic_model_term_lists)
    if metadata_descriptions is not None:
        scatter_chart_explorer.inject_metadata_descriptions(metadata_descriptions)
    if term_colors is not None:
        scatter_chart_explorer.inject_term_colors(term_colors)
        if color_func is None:
            color_func = '(function(d) {return modelInfo.term_colors[d.term]})'
    if term_metadata_df is not None and term_metadata is not None:
        raise Exception('Both term_metadata_df and term_metadata cannot be values which are not None.')
    if term_metadata_df is not None:
        scatter_chart_explorer.inject_term_metadata_df(term_metadata_df)
    if term_metadata is not None:
        scatter_chart_explorer.inject_term_metadata(term_metadata)
    html_base = None
    if semiotic_square:
        html_base = get_semiotic_square_html(num_terms_semiotic_square, semiotic_square)
    if category_term_scores is not None:
        scatter_chart_explorer.inject_category_scores(category_scores=category_term_scores)
    scatter_chart_data = scatter_chart_explorer.to_dict(category=category, category_name=category_name, not_category_name=not_category_name, not_categories=not_categories, transform=transform, scores=scores, max_docs_per_category=max_docs_per_category, metadata=metadata if not callable(metadata) else metadata(corpus), alternative_text_field=alternative_text_field, neutral_category_name=neutral_category_name, extra_category_name=extra_category_name, neutral_categories=neutral_categories, extra_categories=extra_categories, background_scorer=characteristic_scorer, include_term_category_counts=include_term_category_counts, use_offsets=use_offsets)
    if line_coordinates is not None:
        scatter_chart_data['line'] = line_coordinates
    if return_data:
        return scatter_chart_data
    if tooltip_columns is not None:
        assert get_tooltip_content is None
        get_tooltip_content = get_tooltip_js_function(term_metadata_df, tooltip_column_names, tooltip_columns)
    if term_description_columns is not None:
        assert get_custom_term_html is None
        get_custom_term_html = get_custom_term_info_js_function(term_metadata_df, term_description_column_names, term_description_columns, term_word_in_term_description)
    if color_column:
        assert color_func is None
        color_func = '(function(d) {return d.etc["%s"]})' % color_column
    if color_score_column:
        assert color_func is None
        color_func = '(function(d) {return %s(d.etc["%s"])})' % (d3_color_scale if d3_color_scale is not None else 'd3.interpolateRdYlBu', color_score_column)
    if header_sorting_algos is not None:
        assert 'upper' in header_sorting_algos
        assert 'lower' in header_sorting_algos
    if left_list_column is not None:
        assert term_metadata_df is not None
        assert left_list_column in term_metadata_df
        header_sorting_algos = {'upper': '((a,b) => b.etc["' + left_list_column + '"] - a.etc["' + left_list_column + '"])', 'lower': '((a,b) => a.etc["' + left_list_column + '"] - b.etc["' + left_list_column + '"])'}
    if right_order_column is not None:
        assert right_order_column in term_metadata_df
    if show_characteristic is None:
        show_characteristic = not (asian_mode or use_non_text_features)
    scatterplot_structure = ScatterplotStructure(VizDataAdapter(scatter_chart_data), width_in_pixels=width_in_pixels, height_in_pixels=height_in_pixels, max_snippets=max_snippets, color=d3_color_scale, grey_zero_scores=gray_zero_scores, sort_by_dist=sort_by_dist, reverse_sort_scores_for_not_category=reverse_sort_scores_for_not_category, use_full_doc=use_full_doc, asian_mode=asian_mode, match_full_line=match_full_line, use_non_text_features=use_non_text_features, show_characteristic=show_characteristic, word_vec_use_p_vals=word_vec_use_p_vals, max_p_val=max_p_val, save_svg_button=save_svg_button, p_value_colors=p_value_colors, x_label=x_label, y_label=y_label, show_top_terms=show_top_terms, show_neutral=show_neutral, get_tooltip_content=get_tooltip_content, x_axis_values=x_axis_values, y_axis_values=y_axis_values, color_func=color_func, show_axes=show_axes, horizontal_line_y_position=horizontal_line_y_position, vertical_line_x_position=vertical_line_x_position, show_extra=show_extra, do_censor_points=censor_points, center_label_over_points=center_label_over_points, x_axis_labels=x_axis_labels, y_axis_labels=y_axis_labels, topic_model_preview_size=topic_model_preview_size, vertical_lines=vertical_lines, unified_context=unified_context, show_category_headings=show_category_headings, highlight_selected_category=highlight_selected_category, show_cross_axes=show_cross_axes, div_name=div_name, alternative_term_func=alternative_term_func, include_all_contexts=include_all_contexts, show_axes_and_cross_hairs=show_axes_and_cross_hairs, show_diagonal=show_diagonal, use_global_scale=use_global_scale, x_axis_values_format=x_axis_values_format, y_axis_values_format=y_axis_values_format, max_overlapping=max_overlapping, show_corpus_stats=show_corpus_stats, sort_doc_labels_by_name=sort_doc_labels_by_name, enable_term_category_description=enable_term_category_description, always_jump=always_jump, get_custom_term_html=get_custom_term_html, header_names=header_names, header_sorting_algos=header_sorting_algos, ignore_categories=ignore_categories, background_labels=background_labels, label_priority_column=label_priority_column, text_color_column=text_color_column, text_size_column=text_size_column, suppress_text_column=suppress_text_column, background_color=background_color, censor_point_column=censor_point_column, right_order_column=right_order_column, subword_encoding=subword_encoding, top_terms_length=top_terms_length, top_terms_left_buffer=top_terms_left_buffer, get_column_header_html=get_column_header_html, term_word=term_word_in_term_description, show_term_etc=show_term_etc, sort_contexts_by_meta=sort_contexts_by_meta, suppress_circles=suppress_circles, category_colors=category_colors, document_word=document_word, document_word_plural=document_word_plural, category_order=category_order, include_gradient=include_gradient, left_gradient_term=left_gradient_term, middle_gradient_term=middle_gradient_term, right_gradient_term=right_gradient_term, gradient_text_color=gradient_text_color, gradient_colors=gradient_colors, category_term_score_scaler=category_term_score_scaler, show_chart=show_chart)
    if return_scatterplot_structure:
        return scatterplot_structure
    return BasicHTMLFromScatterplotStructure(scatterplot_structure).to_html(protocol=protocol, d3_url=d3_url, d3_scale_chromatic_url=d3_scale_chromatic_url, html_base=html_base)

def get_tooltip_js_function(plot_df, tooltip_column_names, tooltip_columns):
    if len(tooltip_columns) > 2:
        raise Exception('You can have at most two columns in a tooltip.')
    tooltip_content = ''
    tooltip_column_names = {} if tooltip_column_names is None else tooltip_column_names
    for col in tooltip_columns:
        if col not in plot_df:
            raise Exception(f'Column {col} not in plot_df')
        formatting = ''
        if pd.api.types.is_float(plot_df[col].iloc[0]):
            formatting = '.toFixed(6)'
        tooltip_content += '+ "<br />%s: " + d.etc["%s"]%s' % (html.escape(tooltip_column_names.get(col, col)), col.replace('"', '\\"').replace("'", "\\'"), formatting)
    tooltip_content_js_function = '(function(d) {return d.term %s;})' % tooltip_content
    return tooltip_content_js_function

def get_custom_term_info_js_function(plot_df, term_description_column_names, term_description_columns, term_word_in_term_description):
    custom_term_html = ''
    term_description_column_names = {} if term_description_column_names is None else term_description_column_names
    for col in term_description_columns:
        if col not in plot_df:
            raise Exception(f'Column {col} not in plot_df')
        formatting = '.toFixed(6)' if pd.api.types.is_float(plot_df[col].iloc[0]) else ''
        custom_term_html += '+ "<b>%s:</b> " + d.etc["%s"]%s + "<br />"' % (html.escape(term_description_column_names.get(col, col)), col.replace('"', '\\"').replace("'", "\\'"), formatting)
    if custom_term_html != '':
        custom_term_html += '+'
    custom_term_info_js_function = '(d => "%s: "+d.term+"<div class=topic_preview>"%s"</div>")' % (term_word_in_term_description, custom_term_html)
    return custom_term_info_js_function

def _initialize_term_scorer_if_needed(category, corpus, neutral_categories, not_categories, show_neutral, term_scorer, use_non_text_features, term_ranker, term_scorer_kwargs):
    if inherits_from(term_scorer, 'CorpusBasedTermScorer') and type(term_scorer) == ABCMeta:
        term_scorer_kwargs = {} if term_scorer_kwargs is None else term_scorer_kwargs
        term_scorer = term_scorer(corpus, **term_scorer_kwargs)
    if inherits_from(type(term_scorer), 'CorpusBasedTermScorer'):
        if use_non_text_features:
            term_scorer = term_scorer.use_metadata()
        if term_ranker is not None:
            term_scorer = term_scorer.set_term_ranker(term_ranker=term_ranker)
        if not term_scorer.is_category_name_set():
            if show_neutral:
                term_scorer = term_scorer.set_categories(category, not_categories, neutral_categories)
            else:
                term_scorer = term_scorer.set_categories(category, not_categories)
    return term_scorer

def inherits_from(child, parent_name):
    if inspect.isclass(child):
        return parent_name in [c.__name__ for c in inspect.getmro(child)[1:]]
    return False

def get_term_scorer_scores(category, corpus, neutral_categories, not_categories, show_neutral, term_ranker, term_scorer, use_non_text_features):
    tdf = corpus.apply_ranker(term_ranker, use_non_text_features)
    cat_freqs = tdf[str(category) + ' freq']
    if not_categories:
        not_cat_freqs = tdf[[str(c) + ' freq' for c in not_categories]].sum(axis=1)
    else:
        not_cat_freqs = tdf.sum(axis=1) - tdf[str(category) + ' freq']
    if inherits_from(type(term_scorer), 'CorpusBasedTermScorer'):
        return term_scorer.get_scores()
    return term_scorer.get_scores(cat_freqs, not_cat_freqs)

def word_similarity_explorer_gensim(corpus, category, target_term, category_name=None, not_category_name=None, word2vec=None, alpha=0.01, max_p_val=0.1, term_significance=None, **kwargs):
    """
        Parameters
        ----------
        corpus : Corpus
            Corpus to use.
        category : str
            Name of category column as it appears in original data frame.
        category_name : str
            Name of category to use.  E.g., "5-star reviews."
        not_category_name : str
            Name of everything that isn't in category.  E.g., "Below 5-star reviews".
        target_term : str
            Word or phrase for semantic similarity comparison
        word2vec : word2vec.Word2Vec
          Gensim-compatible Word2Vec model of lower-cased corpus. If none, o
          ne will be trained using Word2VecFromParsedCorpus(corpus).train()
        alpha : float, default = 0.01
            Uniform dirichlet prior for p-value calculation
        max_p_val : float, default = 0.1
            Max p-val to use find set of terms for similarity calculation
        term_significance : TermSignificance
            Significance finder

        Remaining arguments are from `produce_scattertext_explorer`.
        Returns
        -------
            str, html of visualization
        """
    if word2vec is None:
        word2vec = Word2VecFromParsedCorpus(corpus).train()
    if term_significance is None:
        term_significance = LogOddsRatioUninformativeDirichletPrior(alpha)
    assert issubclass(type(term_significance), TermSignificance)
    scores = []
    for tok in corpus._term_idx_store._i2val:
        try:
            scores.append(word2vec.similarity(target_term, tok.replace(' ', '_')))
        except:
            try:
                scores.append(np.mean([word2vec.similarity(target_term, tok_part) for tok_part in tok.split()]))
            except:
                scores.append(0)
    scores = np.array(scores)
    return produce_scattertext_explorer(corpus, category, category_name, not_category_name, scores=scores, sort_by_dist=False, reverse_sort_scores_for_not_category=False, word_vec_use_p_vals=True, term_significance=term_significance, max_p_val=max_p_val, p_value_colors=True, **kwargs)

def word_similarity_explorer(corpus, category, category_name, not_category_name, target_term, nlp=None, alpha=0.01, max_p_val=0.1, **kwargs):
    """
    Parameters
    ----------
    corpus : Corpus
        Corpus to use.
    category : str
        Name of category column as it appears in original data frame.
    category_name : str
        Name of category to use.  E.g., "5-star reviews."
    not_category_name : str
        Name of everything that isn't in category.  E.g., "Below 5-star reviews".
    target_term : str
        Word or phrase for semantic similarity comparison
    nlp : spaCy-like parsing function
        E.g., spacy.load('en_core_web_sm'), whitespace_nlp, etc...
    alpha : float, default = 0.01
        Uniform dirichlet prior for p-value calculation
    max_p_val : float, default = 0.1
        Max p-val to use find set of terms for similarity calculation
    Remaining arguments are from `produce_scattertext_explorer`.
    Returns
    -------
        str, html of visualization
    """
    if nlp is None:
        import spacy
        nlp = spacy.load('en_core_web_sm')
    base_term = nlp(target_term)
    scores = np.array([base_term.similarity(nlp(tok)) for tok in corpus._term_idx_store._i2val])
    return produce_scattertext_explorer(corpus, category, category_name, not_category_name, scores=scores, sort_by_dist=False, reverse_sort_scores_for_not_category=False, word_vec_use_p_vals=True, term_significance=LogOddsRatioUninformativeDirichletPrior(alpha), max_p_val=max_p_val, p_value_colors=True, **kwargs)

def produce_frequency_explorer(corpus, category, category_name=None, not_category_name=None, term_ranker=termranking.AbsoluteFrequencyRanker, alpha=0.01, use_term_significance=False, term_scorer=None, not_categories=None, grey_threshold=0, y_axis_values=None, frequency_transform=lambda x: scale(np.log(x) - np.log(1)), score_scaler=None, **kwargs):
    """
    Produces a Monroe et al. style visualization, with the x-axis being the log frequency

    Parameters
    ----------
    corpus : Corpus
        Corpus to use.
    category : str
        Name of category column as it appears in original data frame.
    category_name : str or None
        Name of category to use.  E.g., "5-star reviews."
        Defaults to category
    not_category_name : str or None
        Name of everything that isn't in category.  E.g., "Below 5-star reviews".
        Defaults to "Not " + category_name
    term_ranker : TermRanker
        TermRanker class for determining term frequency ranks.
    alpha : float, default = 0.01
        Uniform dirichlet prior for p-value calculation
    use_term_significance : bool, True by default
        Use term scorer
    term_scorer : TermSignificance
        Subclass of TermSignificance to use as for scores and significance
    not_categories : list
        All categories other than category by default.  Documents labeled
        with remaining category.
    grey_threshold : float
        Score to grey points. Default is 1.96
    y_axis_values : list
        Custom y-axis values. Defaults to linspace
    frequency_transfom : lambda, default lambda x: scale(np.log(x) - np.log(1))
        Takes a vector of frequencies and returns their x-axis scale.
    score_scaler: lambda, default scale_neg_1_to_1_with_zero_mean_abs_max
    Remaining arguments are from `produce_scattertext_explorer`.'
    Returns
    -------
        str, html of visualization
    """
    if not_categories is None:
        not_categories = [c for c in corpus.get_categories() if c != category]
    if term_scorer is None:
        term_scorer = LogOddsRatioUninformativeDirichletPrior(alpha)
    my_term_ranker = term_ranker(corpus)
    term_scorer = _initialize_term_scorer_if_needed(category=category, corpus=corpus, neutral_categories=kwargs.get('neutral_categories', False), not_categories=not_categories, show_neutral=kwargs.get('show_neutral', False), term_scorer=term_scorer, use_non_text_features=kwargs.get('use_non_text_features', False), term_ranker=term_ranker, term_scorer_kwargs=kwargs.get('term_scorer_kwargs', None))
    if kwargs.get('use_non_text_features', False):
        my_term_ranker.use_non_text_features()
    term_freq_df = my_term_ranker.get_ranks() + 1
    freqs = term_freq_df[[str(c) + ' freq' for c in [category] + not_categories]].sum(axis=1).values
    x_axis_values = [round_downer(10 ** x) for x in np.linspace(0, np.log(freqs.max()) / np.log(10), 5)]
    x_axis_values = [x for x in x_axis_values if x > 1 and x <= freqs.max()]
    frequencies_log_scaled = frequency_transform(freqs)
    if 'scores' not in kwargs:
        kwargs['scores'] = get_term_scorer_scores(category, corpus, kwargs.get('neutral_categories', False), not_categories, kwargs.get('show_neutral', False), term_ranker, term_scorer, kwargs.get('use_non_text_features', False))
    if kwargs.get('rescale_y', None) is None:

        def y_axis_rescale(coords):
            return ((coords - 0.5) / np.abs(coords - 0.5).max() + 1) / 2
        kwargs['rescale_y'] = y_axis_rescale

    def round_to_1(x):
        if x == 0:
            return 0
        return round(x, -int(np.floor(np.log10(abs(x)))))
    if y_axis_values is None:
        max_score = np.floor(np.max(kwargs['scores']) * 100) / 100
        min_score = np.ceil(np.min(kwargs['scores']) * 100) / 100
        if min_score < 0 and max_score > 0:
            central = 0
        else:
            central = 0.5
        y_axis_values = [x for x in [min_score, central, max_score] if x >= min_score and x <= max_score]
    if score_scaler is None:
        score_scaler = scale_neg_1_to_1_with_zero_mean_abs_max
    scores_scaled_for_charting = score_scaler(kwargs['scores'])
    if use_term_significance:
        kwargs['term_significance'] = term_scorer
    kwargs['y_label'] = kwargs.get('y_label', term_scorer.get_name())
    kwargs['color_func'] = kwargs.get('color_func', '(function(d) {\n\treturn (Math.abs(d.os) < %s) \n\t ? d3.interpolate(d3.rgb(230, 230, 230), d3.rgb(130, 130, 130))(Math.abs(d.os)/%s) \n\t : d3.interpolateRdYlBu(d.y);\n\t})' % (grey_threshold, grey_threshold))
    return produce_scattertext_explorer(corpus, category=category, category_name=category_name, not_category_name=not_category_name, x_coords=frequencies_log_scaled, y_coords=scores_scaled_for_charting, original_x=freqs, original_y=kwargs['scores'], x_axis_values=x_axis_values, y_axis_values=y_axis_values, rescale_x=scale, sort_by_dist=False, term_ranker=term_ranker, not_categories=not_categories, x_label=kwargs.get('x_label', 'Log Frequency'), **kwargs)

def round_downer(x):
    power_of_ten = 10 ** (len(str(int(x))) - 1)
    num = power_of_ten * (x // power_of_ten)
    return num

def produce_semiotic_square_explorer(semiotic_square: SemioticSquare, x_label, y_label, category_name=None, not_category_name=None, neutral_category_name=None, num_terms_semiotic_square=None, get_tooltip_content=None, x_axis_values=None, y_axis_values=None, color_func=None, axis_scaler=scale_neg_1_to_1_with_zero_mean, **kwargs):
    """
    Produces a semiotic square visualization.

    Parameters
    ----------
    semiotic_square : SemioticSquare
        The basis of the visualization
    x_label : str
        The x-axis label in the scatter plot.  Relationship between `category_a` and `category_b`.
    y_label
        The y-axis label in the scatter plot.  Relationship neutral term and complex term.
    category_name : str or None
        Name of category to use.  Defaults to category_a.
    not_category_name : str or None
        Name of everything that isn't in category.  Defaults to category_b.
    neutral_category_name : str or None
        Name of neutral set of data.  Defaults to "Neutral".
    num_terms_semiotic_square : int or None
        10 by default. Number of terms to show in semiotic square.
    get_tooltip_content : str or None
        Defaults to tooltip showing z-scores on both axes.
    x_axis_values : list, default None
        Value-labels to show on x-axis. [-2.58, -1.96, 0, 1.96, 2.58] is the default
    y_axis_values : list, default None
        Value-labels to show on y-axis. [-2.58, -1.96, 0, 1.96, 2.58] is the default
    color_func : str, default None
        Javascript function to control color of a point.  Function takes a parameter
        which is a dictionary entry produced by `ScatterChartExplorer.to_dict` and
        returns a string. Defaults to RdYlBl on x-axis, and varying saturation on y-axis.
    axis_scaler : lambda, default scale_neg_1_to_1_with_zero_mean_abs_max
        Scale values to fit axis
    Remaining arguments are from `produce_scattertext_explorer`.

    Returns
    -------
        str, html of visualization
    """
    if category_name is None:
        category_name = semiotic_square.category_a_
    if not_category_name is None:
        not_category_name = semiotic_square.category_b_
    if get_tooltip_content is None:
        get_tooltip_content = '(function(d) {return d.term + "<br/>%s: " + Math.round(d.ox*1000)/1000+"<br/>%s: " + Math.round(d.oy*1000)/1000})' % (x_label, y_label)
    if color_func is None:
        color_func = '(function(d) {return d3.interpolateRdYlBu(d.x)})'
    '\n    my_scaler = scale_neg_1_to_1_with_zero_mean_abs_max\n    if foveate:\n        my_scaler = scale_neg_1_to_1_with_zero_mean_rank_abs_max\n    '
    axes = semiotic_square.get_axes()
    return produce_scattertext_explorer(semiotic_square.term_doc_matrix_, category=semiotic_square.category_a_, category_name=category_name, not_category_name=not_category_name, not_categories=[semiotic_square.category_b_], scores=-axes['x'], sort_by_dist=False, x_coords=axis_scaler(-axes['x']), y_coords=axis_scaler(axes['y']), original_x=-axes['x'], original_y=axes['y'], show_characteristic=False, show_top_terms=False, x_label=x_label, y_label=y_label, semiotic_square=semiotic_square, neutral_categories=semiotic_square.neutral_categories_, show_neutral=True, neutral_category_name=neutral_category_name, num_terms_semiotic_square=num_terms_semiotic_square, get_tooltip_content=get_tooltip_content, x_axis_values=x_axis_values, y_axis_values=y_axis_values, term_colors=axes['color'].to_dict(), text_color_column='color', term_metadata_df=axes, show_axes=False, **kwargs)

def produce_four_square_explorer(four_square, x_label=None, y_label=None, a_category_name=None, b_category_name=None, not_a_category_name=None, not_b_category_name=None, num_terms_semiotic_square=None, get_tooltip_content=None, x_axis_values=None, y_axis_values=None, color_func=None, axis_scaler=scale_neg_1_to_1_with_zero_mean, **kwargs):
    """
    Produces a semiotic square visualization.

    Parameters
    ----------
    four_square : FourSquare
        The basis of the visualization
    x_label : str
        The x-axis label in the scatter plot.  Relationship between `category_a` and `category_b`.
    y_label
        The y-axis label in the scatter plot.  Relationship neutral term and complex term.
    a_category_name : str or None
        Name of category to use.  Defaults to category_a.
    b_category_name : str or None
        Name of everything that isn't in category.  Defaults to category_b.
    not_a_category_name : str or None
        Name of neutral set of data.  Defaults to "Neutral".
    not_b_category_name: str or None
        Name of neutral set of data.  Defaults to "Extra".
    num_terms_semiotic_square : int or None
        10 by default. Number of terms to show in semiotic square.
    get_tooltip_content : str or None
        Defaults to tooltip showing z-scores on both axes.
    x_axis_values : list, default None
        Value-labels to show on x-axis. [-2.58, -1.96, 0, 1.96, 2.58] is the default
    y_axis_values : list, default None
        Value-labels to show on y-axis. [-2.58, -1.96, 0, 1.96, 2.58] is the default
    color_func : str, default None
        Javascript function to control color of a point.  Function takes a parameter
        which is a dictionary entry produced by `ScatterChartExplorer.to_dict` and
        returns a string. Defaults to RdYlBl on x-axis, and varying saturation on y-axis.
    axis_scaler : lambda, default scale_neg_1_to_1_with_zero_mean_abs_max
        Scale values to fit axis
    Remaining arguments are from `produce_scattertext_explorer`.

    Returns
    -------
        str, html of visualization
    """
    if a_category_name is None:
        a_category_name = four_square.get_labels()['a_label']
        if a_category_name is None or a_category_name == '':
            a_category_name = four_square.category_a_list_[0]
    if b_category_name is None:
        b_category_name = four_square.get_labels()['b_label']
        if b_category_name is None or b_category_name == '':
            b_category_name = four_square.category_b_list_[0]
    if not_a_category_name is None:
        not_a_category_name = four_square.get_labels()['not_a_label']
        if not_a_category_name is None or not_a_category_name == '':
            not_a_category_name = four_square.not_category_a_list_[0]
    if not_b_category_name is None:
        not_b_category_name = four_square.get_labels()['not_b_label']
        if not_b_category_name is None or not_b_category_name == '':
            not_b_category_name = four_square.not_category_b_list_[0]
    if x_label is None:
        x_label = a_category_name + '-' + b_category_name
    if y_label is None:
        y_label = not_a_category_name + '-' + not_b_category_name
    if get_tooltip_content is None:
        get_tooltip_content = '(function(d) {return d.term + "<br/>%s: " + Math.round(d.ox*1000)/1000+"<br/>%s: " + Math.round(d.oy*1000)/1000})' % (x_label, y_label)
    "\n    # Commenting due to label color change in semiotic square viewer\n    if color_func is None:\n        # this desaturates\n        # color_func = '(function(d) {var c = d3.hsl(d3.interpolateRdYlBu(d.x)); c.s *= d.y; return c;})'\n        color_func = '(function(d) {return d3.interpolateRdYlBu(d.x)})'\n    "
    '\n    my_scaler = scale_neg_1_to_1_with_zero_mean_abs_max\n    if foveate:\n        my_scaler = scale_neg_1_to_1_with_zero_mean_rank_abs_max\n    '
    axes = four_square.get_axes()
    if 'scores' not in kwargs:
        kwargs['scores'] = -axes['x']
    return produce_scattertext_explorer(four_square.term_doc_matrix_, category=list(set(four_square.category_a_list_) - set(four_square.category_b_list_))[0], category_name=a_category_name, not_category_name=b_category_name, not_categories=four_square.category_b_list_, neutral_categories=four_square.not_category_a_list_, extra_categories=four_square.not_category_b_list_, sort_by_dist=False, x_coords=axis_scaler(-axes['x']), y_coords=axis_scaler(axes['y']), original_x=-axes['x'], original_y=axes['y'], show_characteristic=False, show_top_terms=False, x_label=x_label, y_label=y_label, semiotic_square=four_square, show_neutral=True, neutral_category_name=not_a_category_name, show_extra=True, extra_category_name=not_b_category_name, num_terms_semiotic_square=num_terms_semiotic_square, get_tooltip_content=get_tooltip_content, x_axis_values=x_axis_values, y_axis_values=y_axis_values, color_func=color_func, term_colors=axes['color'].to_dict(), text_color_column='color', term_metadata_df=axes, show_axes=False, **kwargs)

def produce_four_square_axes_explorer(four_square_axes, x_label=None, y_label=None, num_terms_semiotic_square=None, get_tooltip_content=None, x_axis_values=None, y_axis_values=None, color_func=None, axis_scaler=scale_neg_1_to_1_with_zero_mean, **kwargs):
    """
    Produces a semiotic square visualization.

    Parameters
    ----------
    four_square : FourSquareAxes
        The basis of the visualization
    x_label : str
        The x-axis label in the scatter plot.  Relationship between `category_a` and `category_b`.
    y_label
        The y-axis label in the scatter plot.  Relationship neutral term and complex term.
    not_b_category_name: str or None
        Name of neutral set of data.  Defaults to "Extra".
    num_terms_semiotic_square : int or None
        10 by default. Number of terms to show in semiotic square.
    get_tooltip_content : str or None
        Defaults to tooltip showing z-scores on both axes.
    x_axis_values : list, default None
        Value-labels to show on x-axis. [-2.58, -1.96, 0, 1.96, 2.58] is the default
    y_axis_values : list, default None
        Value-labels to show on y-axis. [-2.58, -1.96, 0, 1.96, 2.58] is the default
    color_func : str, default None
        Javascript function to control color of a point.  Function takes a parameter
        which is a dictionary entry produced by `ScatterChartExplorer.to_dict` and
        returns a string. Defaults to RdYlBl on x-axis, and varying saturation on y-axis.
    axis_scaler : lambda, default scale_neg_1_to_1_with_zero_mean_abs_max
        Scale values to fit axis
    Remaining arguments are from `produce_scattertext_explorer`.

    Returns
    -------
        str, html of visualization
    """
    if x_label is None:
        x_label = four_square_axes.left_category_name_ + '-' + four_square_axes.right_category_name_
    if y_label is None:
        y_label = four_square_axes.top_category_name_ + '-' + four_square_axes.bottom_category_name_
    if get_tooltip_content is None:
        get_tooltip_content = '(function(d) {return d.term + "<br/>%s: " + Math.round(d.ox*1000)/1000+"<br/>%s: " + Math.round(d.oy*1000)/1000})' % (x_label, y_label)
    "\n    if color_func is None:\n        # this desaturates\n        # color_func = '(function(d) {var c = d3.hsl(d3.interpolateRdYlBu(d.x)); c.s *= d.y; return c;})'\n        color_func = '(function(d) {return d3.interpolateRdYlBu(d.x)})'\n    "
    axes = four_square_axes.get_axes()
    if 'scores' not in kwargs:
        kwargs['scores'] = -axes['x']
    '\n    my_scaler = scale_neg_1_to_1_with_zero_mean_abs_max\n    if foveate:\n        my_scaler = scale_neg_1_to_1_with_zero_mean_rank_abs_max\n    '
    return produce_scattertext_explorer(four_square_axes.term_doc_matrix_, category=four_square_axes.left_categories_[0], category_name=four_square_axes.left_category_name_, not_categories=four_square_axes.right_categories_, not_category_name=four_square_axes.right_category_name_, neutral_categories=four_square_axes.top_categories_, neutral_category_name=four_square_axes.top_category_name_, extra_categories=four_square_axes.bottom_categories_, extra_category_name=four_square_axes.bottom_category_name_, sort_by_dist=False, x_coords=axis_scaler(-axes['x']), y_coords=axis_scaler(axes['y']), original_x=-axes['x'], original_y=axes['y'], show_characteristic=False, show_top_terms=False, x_label=x_label, y_label=y_label, semiotic_square=four_square_axes, show_neutral=True, show_extra=True, num_terms_semiotic_square=num_terms_semiotic_square, get_tooltip_content=get_tooltip_content, x_axis_values=x_axis_values, y_axis_values=y_axis_values, color_func=color_func, term_colors=axes['color'].to_dict(), text_color_column='color', term_metadata_df=axes, show_axes=False, **kwargs)

def produce_projection_explorer(corpus, category, word2vec_model=None, projection_model=None, embeddings=None, term_acceptance_re=re.compile('[a-z]{3,}'), show_axes=False, **kwargs):
    """
    Parameters
    ----------
    corpus : ParsedCorpus
        It is highly recommended to use a stoplisted, unigram corpus-- `corpus.get_stoplisted_unigram_corpus()`
    category : str
    word2vec_model : Word2Vec
        A gensim word2vec model.  A default model will be used instead. See Word2VecFromParsedCorpus for the default
        model.
    projection_model : sklearn-style dimensionality reduction model.
        By default: umap.UMAP(min_dist=0.5, metric='cosine')
      You could also use, e.g., sklearn.manifold.TSNE(perplexity=10, n_components=2, init='pca', n_iter=2500, random_state=23)
    embeddings : array[len(corpus.get_terms()), X]
        Word embeddings.  If None (default), will train them using word2vec Model
    term_acceptance_re : SRE_Pattern,
        Regular expression to identify valid terms
    show_axes : bool, default False
        Show the ticked axes on the plot.  If false, show inner axes as a crosshair.
    kwargs : dict
        Remaining produce_scattertext_explorer keywords get_tooltip_content

    Returns
    -------
    str
    HTML of visualization

    """
    embeddings_resolover = EmbeddingsResolver(corpus)
    if embeddings is not None:
        embeddings_resolover.set_embeddings(embeddings)
    else:
        embeddings_resolover.set_embeddings_model(word2vec_model, term_acceptance_re)
    corpus, word_axes = embeddings_resolover.project_embeddings(projection_model, x_dim=0, y_dim=1)
    html = produce_scattertext_explorer(corpus=corpus, category=category, minimum_term_frequency=0, sort_by_dist=False, x_coords=scale(word_axes['x']), y_coords=scale(word_axes['y']), y_label='', x_label='', show_axes=show_axes, **kwargs)
    return html

def produce_pca_explorer(corpus, category, word2vec_model=None, projection_model=None, embeddings=None, projection=None, term_acceptance_re=re.compile('[a-z]{3,}'), x_dim=0, y_dim=1, scaler=scale, show_axes=False, show_dimensions_on_tooltip=True, x_label='', y_label='', **kwargs):
    """
    Parameters
    ----------
    corpus : ParsedCorpus
        It is highly recommended to use a stoplisted, unigram corpus-- `corpus.get_stoplisted_unigram_corpus()`
    category : str
    word2vec_model : Word2Vec
        A gensim word2vec model.  A default model will be used instead. See Word2VecFromParsedCorpus for the default
        model.
    projection_model : sklearn-style dimensionality reduction model. Ignored if 'projection' is presents
        By default: umap.UMAP(min_dist=0.5, metric='cosine') unless projection is present. If so,
        You could also use, e.g., sklearn.manifold.TSNE(perplexity=10, n_components=2, init='pca', n_iter=2500, random_state=23)
    embeddings : array[len(corpus.get_terms()), X]
        Word embeddings.  If None (default), and no value is passed into projection, use word2vec_model
    projection : DataFrame('x': array[len(corpus.get_terms())], 'y': array[len(corpus.get_terms())])
        If None (default), produced using projection_model
    term_acceptance_re : SRE_Pattern,
        Regular expression to identify valid terms
    x_dim : int, default 0
        Dimension of transformation matrix for x-axis
    y_dim : int, default 1
        Dimension of transformation matrix for y-axis
    scalers : function , default scattertext.Scalers.scale
        Function used to scale projection
    show_axes : bool, default False
        Show the ticked axes on the plot.  If false, show inner axes as a crosshair.
    show_dimensions_on_tooltip : bool, False by default
        If true, shows dimension positions on tooltip, along with term name. Otherwise, default to the
         get_tooltip_content parameter.
    kwargs : dict
        Remaining produce_scattertext_explorer keywords get_tooltip_content

    Returns
    -------
    str
    HTML of visualization
    """
    if projection is None:
        embeddings_resolover = EmbeddingsResolver(corpus)
        if embeddings is not None:
            embeddings_resolover.set_embeddings(embeddings)
        else:
            embeddings_resolover.set_embeddings_model(word2vec_model, term_acceptance_re)
        corpus, projection = embeddings_resolover.project_embeddings(projection_model, x_dim=x_dim, y_dim=y_dim)
    else:
        assert type(projection) == pd.DataFrame
        assert 'x' in projection and 'y' in projection
        if kwargs.get('use_non_text_features', False):
            assert set(projection.index) == set(corpus.get_metadata())
        else:
            assert set(projection.index) == set(corpus.get_terms())
    if show_dimensions_on_tooltip:
        kwargs['get_tooltip_content'] = '(function(d) {\n     return  d.term + "<br/>Dim %s: " + Math.round(d.ox*1000)/1000 + "<br/>Dim %s: " + Math.round(d.oy*1000)/1000 \n    })' % (x_dim, y_dim)
    html = produce_scattertext_explorer(corpus=corpus, category=category, minimum_term_frequency=0, sort_by_dist=False, original_x=projection['x'], original_y=projection['y'], x_coords=scaler(projection['x']), y_coords=scaler(projection['y']), y_label=y_label, x_label=x_label, show_axes=show_axes, horizontal_line_y_position=kwargs.get('horizontal_line_y_position', None), vertical_line_x_position=kwargs.get('vertical_line_x_position', None), **kwargs)
    return html

def sparse_explorer(corpus, category, scores, category_name=None, not_category_name=None, **kwargs):
    """
    Parameters
    ----------
    corpus : Corpus
        Corpus to use.
    category : str
        Name of category column as it appears in original data frame.
    category_name : str
        Name of category to use.  E.g., "5-star reviews."
    not_category_name : str
        Name of everything that isn't in category.  E.g., "Below 5-star reviews".
    scores : np.array
        Scores to display in visualization.  Zero scores are grey.

    Remaining arguments are from `produce_scattertext_explorer`.

    Returns
    -------
        str, html of visualization
    """
    return produce_scattertext_explorer(corpus, category, category_name, not_category_name, scores=scores, sort_by_dist=False, gray_zero_scores=True, **kwargs)

def produce_two_axis_plot(corpus, x_score_df, y_score_df, x_label, y_label, statistic_column='cohens_d', p_value_column='cohens_d_p', statistic_name='d', use_non_text_features=False, pick_color=pick_color, axis_scaler=scale_neg_1_to_1_with_zero_mean, distance_measure=EuclideanDistance, semiotic_square_labels=None, x_tooltip_label=None, y_tooltip_label=None, **kwargs):
    """

    :param corpus: Corpus
    :param x_score_df: pd.DataFrame, contains effect_size_column, p_value_column. outputted by CohensD
    :param y_score_df: pd.DataFrame, contains effect_size_column, p_value_column. outputted by CohensD
    :param x_label: str
    :param y_label: str
    :param statistic_column: str, column in x_score_df, y_score_df giving statistics, default cohens_d
    :param p_value_column: str, column in x_score_df, y_score_df giving effect sizes, default cohens_d_p
    :param statistic_name: str, column which corresponds to statistic name, defauld d
    :param use_non_text_features: bool, default True
    :param pick_color: func, returns color, default is pick_color
    :param axis_scaler: func, scaler default is scale_neg_1_to_1_with_zero_mean
    :param distance_measure: DistanceMeasureBase, default EuclideanDistance
        This is how parts of the square are populated
    :param semiotic_square_labels: dict, semiotic square position labels
    :param x_tooltip_label: str, if None, x_label
    :param y_tooltip_label: str, if None, y_label
    :param kwargs: dict, other arguments
    :return: str, html
    """
    if use_non_text_features:
        terms = corpus.get_metadata()
    else:
        terms = corpus.get_terms()
    axes = pd.DataFrame({'x': x_score_df[statistic_column], 'y': y_score_df[statistic_column]}).loc[terms]
    merged_scores = pd.merge(x_score_df, y_score_df, left_index=True, right_index=True).loc[terms]
    x_tooltip_label = x_label if x_tooltip_label is None else x_tooltip_label
    y_tooltip_label = y_label if y_tooltip_label is None else y_tooltip_label

    def generate_term_metadata(term_struct):
        if p_value_column + '_corr_x' in term_struct:
            x_p = term_struct[p_value_column + '_corr_x']
        elif p_value_column + '_x' in term_struct:
            x_p = term_struct[p_value_column + '_x']
        else:
            x_p = None
        if p_value_column + '_corr_y' in term_struct:
            y_p = term_struct[p_value_column + '_corr_y']
        elif p_value_column + '_y' in term_struct:
            y_p = term_struct[p_value_column + '_y']
        else:
            y_p = None
        if x_p is not None:
            x_p = min(x_p, 1.0 - x_p)
        if y_p is not None:
            y_p = min(y_p, 1.0 - y_p)
        x_d = term_struct[statistic_column + '_x']
        y_d = term_struct[statistic_column + '_y']
        tooltip = '%s: %s: %0.3f' % (x_tooltip_label, statistic_name, x_d)
        if x_p is not None:
            tooltip += '; p: %0.4f' % x_p
        tooltip += '<br/>'
        tooltip += '%s: %s: %0.3f' % (y_tooltip_label, statistic_name, y_d)
        if y_p is not None:
            tooltip += '; p: %0.4f' % y_p
        return {'tooltip': tooltip, 'color': pick_color(x_p, y_p, np.abs(x_d), np.abs(y_d))}
    explanations = merged_scores.apply(generate_term_metadata, axis=1)
    semiotic_square = SemioticSquareFromAxes(corpus, axes, x_axis_name=x_label, y_axis_name=y_label, labels=semiotic_square_labels, distance_measure=distance_measure)
    get_tooltip_content = kwargs.get('get_tooltip_content', '(function(d) {return d.term + "<br/> " + d.etc.tooltip})')
    color_func = kwargs.get('color_func', '(function(d) {return d.etc.color})')
    html = produce_scattertext_explorer(corpus, category=corpus.get_categories()[0], sort_by_dist=False, x_coords=axis_scaler(axes['x']), y_coords=axis_scaler(axes['y']), original_x=axes['x'], original_y=axes['y'], show_characteristic=False, show_top_terms=False, show_category_headings=True, x_label=x_label, y_label=y_label, semiotic_square=semiotic_square, get_tooltip_content=get_tooltip_content, x_axis_values=None, y_axis_values=None, unified_context=True, color_func=color_func, show_axes=False, term_metadata=explanations.to_dict(), use_non_text_features=use_non_text_features, **kwargs)
    return html

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

class MultiCategoryAssociationScorer(MultiCategoryAssociationBase):

    def get_category_association(self, ranker: Union[TermRanker, Type]=None, scorer=None, verbose=False):
        ranker, scorer = self._resolve_ranker_and_scorer(ranker, scorer)
        data = []
        it = self.corpus.get_categories()
        if verbose:
            it = tqdm(it)
        for cat in it:
            scores = self.__get_scores(cat=cat, scorer=scorer, ranker=ranker)
            for term_rank, (term, score) in enumerate(scores.sort_values(ascending=False).items()):
                data.append({'Category': cat, 'Term': term, 'Rank': term_rank, 'Score': score})
        return pd.DataFrame(data)

    def get_category_association_and_freqs(self, ranker: Union[TermRanker, Type]=None, scorer=None, verbose=False):
        ranker, scorer = self._resolve_ranker_and_scorer(ranker, scorer)
        data = []
        it = self.corpus.get_categories()
        if verbose:
            it = tqdm(it)
        term_freq_df = ranker.get_ranks('')
        for cat in it:
            scores = self.__get_scores(cat=cat, scorer=scorer, ranker=ranker)
            freqs = term_freq_df[str(cat)]
            for term_rank, (term, score) in enumerate(scores.sort_values(ascending=False).items()):
                data.append({'Category': cat, 'Term': term, 'Freq': freqs.loc[term], 'Rank': term_rank, 'Score': score})
        return pd.DataFrame(data)

    def __get_scores(self, cat, scorer, ranker) -> pd.Series:
        if inherits_from(type(scorer), 'CorpusBasedTermScorer'):
            if self.use_metadata:
                scorer = scorer.use_metadata()
            scorer = scorer.set_categories(category_name=cat)
            if ranker is not None:
                scorer = scorer.set_term_ranker(term_ranker=ranker)
            return scorer.get_scores()
        term_freq_df = ranker.get_ranks('')
        try:
            cat_freq = term_freq_df[cat]
        except KeyError:
            cat_freq = term_freq_df[str(cat)]
        global_freq = term_freq_df.sum(axis=1)
        return scorer.get_scores(cat_freq, global_freq - cat_freq)

class MultiCategoryAssociationBase:

    def __init__(self, corpus, use_metadata=False, non_text=False, use_non_text_features=False, ranker=None, scorer=None):
        self.corpus = corpus
        self.use_metadata = use_metadata or non_text or use_non_text_features
        self.ranker, self.scorer = self._resolve_ranker_and_scorer(ranker=ranker, scorer=scorer)

    def get_category_association(self, **kwargs):
        raise NotImplementedError()

    def get_category_association_and_freqs(self, **kwargs):
        raise NotImplementedError()

    def _resolve_ranker_and_scorer(self, ranker, scorer):
        if scorer is None:
            scorer = RankDifference()
        if inherits_from(scorer, 'CorpusBasedTermScorer'):
            scorer = scorer(self.corpus, use_metadata=self.use_metadata)
        elif type(scorer) == type:
            scorer = scorer()
        if ranker is None:
            ranker = AbsoluteFrequencyRanker(self.corpus)
        if inherits_from(ranker, 'TermRanker'):
            ranker = ranker(self.corpus)
        if self.use_metadata:
            ranker = ranker.use_non_text_features()
        return (ranker, scorer)

class TestWord2VecFromParsedCorpus(TestCase):

    @classmethod
    def setUp(cls):
        cls.categories, cls.documents = get_docs_categories()
        cls.parsed_docs = []
        for doc in cls.documents:
            cls.parsed_docs.append(whitespace_nlp(doc))
        cls.df = pd.DataFrame({'category': cls.categories, 'author': ['a', 'a', 'c', 'c', 'c', 'c', 'd', 'd', 'e', 'e'], 'parsed': cls.parsed_docs, 'document_lengths': [len(doc) for doc in cls.documents]})
        cls.corpus = CorpusFromParsedDocuments(cls.df, 'category', 'parsed').build()

    def test_make(self):
        gensim_is_present_and_working = False
        try:
            from gensim.models import word2vec
            gensim_is_present_and_working = True
        except:
            pass
        if gensim_is_present_and_working:
            Word2VecFromParsedCorpus(self.corpus)
            Word2VecFromParsedCorpus(self.corpus, word2vec.Word2Vec())

    def test_train(self):
        gensim_is_present_and_working = False
        try:
            from gensim.models import word2vec
            gensim_is_present_and_working = True
        except:
            pass
        if gensim_is_present_and_working:
            Word2VecFromParsedCorpus(self.corpus).train()

    def test_bigrams(self):
        gensim_is_present_and_working = False
        try:
            from gensim.models import word2vec
            gensim_is_present_and_working = True
        except:
            pass
        if gensim_is_present_and_working:
            Word2VecFromParsedCorpusBigrams(self.corpus).train()

class InterArrivalCounts:

    def __init__(self, corpus: OffsetCorpus, non_text: bool, domains_to_preserve: Optional[List[str]]=None, verbose: bool=False, random_generator: Optional[np.random._generator.Generator]=None):
        self.corpus = corpus
        self.rng = random_generator
        self.domains_to_preserve = domains_to_preserve
        if not non_text:
            assert inherits_from(type(corpus), 'OffsetCorpus')
        self.term_inter_arrivals = {}
        self.term_category_inter_arrivals = {}
        self.__populate_inter_arrival_stats(verbose=verbose)

    def __populate_inter_arrival_stats(self, verbose: bool):
        it = enumerate(self.corpus.get_terms(use_metadata=True))
        if verbose:
            it = tqdm(it, total=self.corpus.get_num_metadata())
        for term_idx, term in it:
            term_offsets = self.corpus.get_offsets()[term]
            interarrivals = []
            interarrival_cat = {}
            for doc_i, row in self.corpus.get_df().iterrows():
                cat = row[self.corpus.get_category_column()]
                doc = row[self.corpus.get_parsed_column()]
                last_end = None
                interarrival_cat.setdefault(cat, [])
                term_doc_offsets = term_offsets.get(doc_i, [])
                if term_doc_offsets:
                    for offset_start, offset_end in term_doc_offsets:
                        if self.__not_the_first_example_of_a_term(last_end):
                            intervening_token_count = self.__get_intervening_token_count(doc, last_end, offset_start)
                            self.__append_inter_arrival(cat, interarrival_cat, interarrivals, intervening_token_count)
                        last_end = offset_end
                    end_inter_arrival_tokens = self.__get_intervening_tokens_for_final_term(doc, term_doc_offsets)
                    self.__append_inter_arrival(cat, interarrival_cat, interarrivals, end_inter_arrival_tokens)
            self.term_inter_arrivals[term] = interarrivals
            self.term_category_inter_arrivals[term] = copy(interarrival_cat)

    def __get_intervening_tokens_for_final_term(self, doc, term_doc_offsets):
        tokens_before_first = doc.char_span(0, term_doc_offsets[0][0], alignment_mode='contract')
        token_count_before_first = 0 if tokens_before_first is None else len(tokens_before_first)
        tokens_after_last = doc.char_span(term_doc_offsets[-1][1], len(str(doc)), alignment_mode='contract')
        token_count_after_last = 0 if tokens_after_last is None else len(tokens_after_last)
        end_inter_arrival_tokens = token_count_before_first + token_count_after_last + 1
        return end_inter_arrival_tokens

    def __get_intervening_token_count(self, doc, last_end, offset_start):
        intervening_tokens = doc.char_span(last_end, offset_start, alignment_mode='contract')
        intervening_token_count = (0 if intervening_tokens is None else len(intervening_tokens)) + 1
        return intervening_token_count

    def __not_the_first_example_of_a_term(self, last_end: Optional[int]) -> bool:
        return last_end is not None

    def __append_inter_arrival(self, cat: str, interarrival_cat: Dict, interarrivals, intervening_token_count: int) -> None:
        interarrivals.append(intervening_token_count)
        interarrival_cat[cat].append(intervening_token_count)

class CorpusBasedTermScorer(with_metaclass(ABCMeta, object)):

    def __init__(self, corpus: 'ParsedCorpus', *args, **kwargs):
        self.corpus_ = corpus
        self.category_ids_ = corpus._y
        self.tdf_ = None
        self.use_metadata_ = kwargs.get('non_text', False) or kwargs.get('use_metadata', False) or kwargs.get('use_non_text_features', False)
        self.category_name_is_set_ = False
        self._doc_sizes = None
        self._set_scorer_args(**kwargs)
        self.term_ranker_ = self._get_default_ranker(corpus)

    def _get_default_ranker(self, corpus):
        return AbsoluteFrequencyRanker(corpus).set_non_text(non_text=self.use_metadata_)

    @abstractmethod
    def _set_scorer_args(self, **kwargs):
        pass

    def set_doc_sizes(self, doc_sizes: np.array) -> 'CorpusBasedTermScorer':
        assert len(doc_sizes) == self.corpus_.get_num_docs()
        self._doc_sizes = doc_sizes
        return self

    def use_token_counts_as_doc_sizes(self) -> 'CorpusBasedTermScorer':
        return self.set_doc_sizes(doc_sizes=self.corpus_.get_parsed_docs().apply(len).values)

    def get_doc_sizes(self) -> np.array:
        if self._doc_sizes is None:
            return self._get_X().sum(axis=1)
        return self._doc_sizes

    def _get_cat_size(self) -> float:
        return self.get_doc_sizes()[self._get_cat_x_row_mask()].sum()

    def _get_ncat_size(self) -> float:
        return self.get_doc_sizes()[self._get_ncat_x_row_mask()].sum()

    def use_metadata(self) -> 'CorpusBasedTermScorer':
        self.use_metadata_ = True
        self.term_ranker_ = self.term_ranker_.use_non_text_features()
        return self

    def set_term_ranker(self, term_ranker) -> 'CorpusBasedTermScorer':
        if inherits_from(term_ranker, 'TermRanker'):
            self.term_ranker_ = term_ranker(self.corpus_)
        else:
            self.term_ranker_ = term_ranker
        if self.use_metadata_:
            self.term_ranker_.use_non_text_features()
        return self

    def get_term_ranker(self) -> TermRanker:
        return self.term_ranker_

    def is_category_name_set(self):
        return self.category_name_is_set_

    def set_categories(self, category_name, not_category_names=[], neutral_category_names=[]):
        """
        Specify the category to score. Optionally, score against a specific set of categories.
        """
        tdf = self.term_ranker_.set_non_text(non_text=self.use_metadata_).get_ranks(label_append='')
        d = {'cat': tdf[str(category_name)]}
        if not_category_names == []:
            not_category_names = [str(c) for c in self.corpus_.get_categories() if c != category_name]
        else:
            not_category_names = [str(c) for c in not_category_names]
        d['ncat'] = tdf[not_category_names].sum(axis=1)
        if neutral_category_names == []:
            pass
        else:
            neutral_category_names = [str(c) for c in neutral_category_names]
        for i, c in enumerate(neutral_category_names):
            d['neut%s' % i] = tdf[c]
        self.tdf_ = pd.DataFrame(d)
        self.category_name = category_name
        self.not_category_names = [c for c in not_category_names]
        self.neutral_category_names = [c for c in neutral_category_names]
        self.category_name_is_set_ = True
        return self

    def _get_X(self):
        return self.term_ranker_.get_term_doc_mat()

    def get_t_statistics(self):
        """
        In this case, parameters a and b aren't used, since this information is taken
        directly from the corpus categories.

        Returns
        -------

        """
        X = self._get_X()
        cat_X, ncat_X = self._get_cat_and_ncat(X)
        mean_delta = self._get_mean_delta(cat_X, ncat_X)
        cat_var = sparse_var(cat_X)
        ncat_var = sparse_var(ncat_X)
        cat_n = cat_X.shape[0]
        ncat_n = ncat_X.shape[0]
        pooled_stderr = np.sqrt(cat_var / cat_n + ncat_var / ncat_n)
        tt = mean_delta / pooled_stderr
        degs_of_freedom = (cat_var ** 2 / cat_n + ncat_var ** 2 / ncat_n) ** 2 / ((cat_var ** 2 / cat_n) ** 2 / (cat_n - 1) + (ncat_var ** 2 / ncat_n) ** 2 / (ncat_n - 1))
        only_in_neutral_mask = self.tdf_[['cat', 'ncat']].sum(axis=1) == 0
        pvals = stats.t.sf(np.abs(tt), degs_of_freedom)
        tt[only_in_neutral_mask] = 0
        pvals[only_in_neutral_mask] = 0
        return (tt, pvals)

    def _get_mean_delta(self, cat_X, ncat_X):
        return np.array(cat_X.mean(axis=0) - ncat_X.mean(axis=0))[0]

    def _get_cat_and_ncat(self, X):
        if self.category_name_is_set_ is False:
            raise NeedToSetCategoriesException()
        try:
            if X.format == 'coo':
                X = X.tocsr()
        except:
            pass
        cat_X = X[self._get_cat_x_row_mask(), :]
        ncat_X = X[self._get_ncat_x_row_mask(), :]
        if self.neutral_category_names:
            neut_X = X[self._get_neut_row_mask(), :]
            cat_X = vstack([cat_X, neut_X])
            ncat_X = vstack([ncat_X, neut_X])
        return (cat_X, ncat_X)

    def _get_neut_row_mask(self):
        return np.isin(self.corpus_.get_category_names_by_row(), self.neutral_category_names)

    def _get_ncat_x_row_mask(self):
        return np.isin(self.corpus_.get_category_names_by_row(), self.not_category_names)

    def _get_cat_x_row_mask(self):
        return np.isin(self.corpus_.get_category_names_by_row(), [self.category_name])

    def _get_index(self):
        return self.corpus_.get_metadata() if self.use_metadata_ else self.corpus_.get_terms()

    @abstractmethod
    def get_scores(self, *args):
        """
        Args are ignored

        Returns
        -------
        """

    @abstractmethod
    def get_name(self):
        pass

    def _get_terms(self):
        return self.corpus_.get_terms(use_metadata=self.use_metadata_)

    def _get_num_terms(self):
        return self.corpus_.get_num_terms(non_text=self.use_metadata_)

    def get_score_df(self, label_append=''):
        return self.get_term_ranker().get_ranks(label_append=label_append).assign(Metric=self.get_scores()).sort_values(by='Metric', ascending=True).rename(columns={'Metric': self.get_name()})

    def __get_f1_f2_from_args(self, args) -> Tuple[np.array, np.array]:
        f1, f2 = args
        assert len(f1) == len(f2)
        assert len(f1) == len(self._get_terms())
        return (f1, f2)

