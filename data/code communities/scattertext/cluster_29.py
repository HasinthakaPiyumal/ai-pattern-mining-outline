# Cluster 29

def get_semiotic_square_html(num_terms_semiotic_square, semiotic_square):
    """

    :param num_terms_semiotic_square: int
    :param semiotic_square: SemioticSquare
    :return: str
    """
    semiotic_square_html = None
    if semiotic_square:
        semiotic_square_viz = HTMLSemioticSquareViz(semiotic_square)
        if num_terms_semiotic_square:
            semiotic_square_html = semiotic_square_viz.get_html(num_terms_semiotic_square)
        else:
            semiotic_square_html = semiotic_square_viz.get_html()
    return semiotic_square_html

class TestHTMLSemioticSquareViz(TestCase):

    def test_get_html(self):
        semsq = get_test_semiotic_square()
        html_default = HTMLSemioticSquareViz(semsq).get_html()
        html_6 = HTMLSemioticSquareViz(semsq).get_html(num_terms=6)
        self.assertNotEqual(html_default, html_6)

def get_test_semiotic_square():
    corpus = get_test_corpus()
    semsq = SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', ['swift'])
    return semsq

def get_test_corpus():
    df = pd.DataFrame(data=np.array(get_docs_categories_semiotic()).T, columns=['category', 'text'])
    corpus = CorpusFromPandas(df, 'category', 'text', nlp=whitespace_nlp).build()
    return corpus

class TestSemioticSquare(TestCase):

    def test_constructor(self):
        df = pd.DataFrame(data=np.array(get_docs_categories_semiotic()).T, columns=['category', 'text'])
        corpus = CorpusFromPandas(df, 'category', 'text', nlp=whitespace_nlp).build()
        SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', ['swift'])
        with self.assertRaises(AssertionError):
            SemioticSquare(corpus, 'XXXhamlet', 'jay-z/r. kelly', ['swift'])
        with self.assertRaises(AssertionError):
            SemioticSquare(corpus, 'hamlet', 'jay-z/r. kellyXXX', ['swift'])
        with self.assertRaises(AssertionError):
            SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', ['swift', 'asd'])
        with self.assertRaises(EmptyNeutralCategoriesError):
            SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', [])

    def test_get_labels(self):
        corpus = get_test_corpus()
        semsq = SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', ['swift'])
        a, b = ('hamlet', 'jay-z/r. kelly')
        default_labels = {'a': a, 'not_a': 'Not ' + a, 'b': b, 'not_b': 'Not ' + b, 'a_and_b': a + ' + ' + b, 'not_a_and_not_b': 'Not ' + a + ' + Not ' + b, 'a_and_not_b': a + ' + Not ' + b, 'b_and_not_a': 'Not ' + a + ' + ' + b}
        labels = semsq.get_labels()
        for name, default_label in default_labels.items():
            self.assertTrue(name + '_label' in labels)
            self.assertEqual(labels[name + '_label'], default_label)
        semsq = SemioticSquare(corpus, 'hamlet', 'jay-z/r. kelly', ['swift'], labels={'a': 'AAA'})
        labels = semsq.get_labels()
        for name, default_label in default_labels.items():
            if name == 'a':
                self.assertEqual(labels[name + '_label'], 'AAA')
            else:
                self.assertTrue(name + '_label' in labels)
                self.assertEqual(labels[name + '_label'], default_label)

    def test_get_lexicons(self):
        semsq = get_test_semiotic_square()
        lexicons = semsq.get_lexicons()
        for category in self.categories():
            self.assertIn(category, lexicons)
            self.assertLessEqual(len(lexicons[category]), 10)
        lexicons = semsq.get_lexicons(5)
        for category in self.categories():
            self.assertIn(category, lexicons)
            self.assertLessEqual(len(lexicons[category]), 5)

    def test_get_axes(self):
        semsq = get_test_semiotic_square()
        ax = semsq.get_axes()
        self.assertEqual(list(sorted(ax.index)), list(sorted(semsq.term_doc_matrix_.get_terms())))

    def categories(self):
        return ['a', 'b', 'not_a', 'not_b', 'a_and_not_b', 'b_and_not_a', 'a_and_b', 'not_a_and_not_b']

class TestHTMLVisualizationAssembly(TestCase):

    def get_params(self, param_dict={}):
        params = ['1000', '600', 'null', 'null', 'true', 'false', 'false', 'false', 'false', 'true', 'false', 'false', 'true', '0.1', 'false', 'undefined', 'undefined', 'getDataAndInfo()', 'true', 'false', 'null', 'null', 'null', 'null', 'true', 'false', 'true', 'false', 'null', 'null', '10', 'null', 'null', 'null', 'false', 'true', 'true', '"' + DEFAULT_DIV_ID + '"', 'null', 'false', 'false', '"' + DEFAULT_D3_AXIS_VALUE_FORMAT + '"', '"' + DEFAULT_D3_AXIS_VALUE_FORMAT + '"', 'false', '-1', 'true', 'false', 'true', 'false', 'false', 'false', 'true', 'null', 'null', 'null', 'false', 'null', 'undefined', 'undefined', 'undefined', 'undefined', 'undefined', 'undefined', 'undefined', '14', '0', 'null', '"Term"', 'true', 'false', 'false', 'undefined', 'null', '"document"', '"documents"', 'null', 'false', 'null', 'null', 'null', 'null', 'null', 'null', 'false']
        for i, val in param_dict.items():
            params[i] = val
        return 'buildViz(' + ',\n'.join(params) + ');\n'

    def make_assembler(self):
        scatterplot_structure = ScatterplotStructure(self.make_adapter())
        return BasicHTMLFromScatterplotStructure(scatterplot_structure)

    def make_adapter(self):
        words_dict = {'info': {'not_category_name': 'Republican', 'category_name': 'Democratic'}, 'data': [{'y': 0.33763837638376387, 'term': 'crises', 'ncat25k': 0, 'cat25k': 1, 'x': 0.0, 's': 0.878755930416447}, {'y': 0.5, 'term': 'something else', 'ncat25k': 0, 'cat25k': 1, 'x': 0.0, 's': 0.5}]}
        visualization_data = VizDataAdapter(words_dict)
        return visualization_data

    def test_main(self):
        assembler = self.make_assembler()
        html = assembler.to_html()
        if sys.version_info.major == 2:
            self.assertEqual(type(html), unicode)
        else:
            self.assertEqual(type(html), str)
        self.assertFalse('<!-- EXTRA LIBS -->' in html)
        self.assertFalse('<!-- INSERT SCRIPT -->' in html)
        self.assertTrue('<!-- INSERT SEMIOTIC SQUARE -->' in html)
        self.assertTrue('Republican' in html)

    def test_semiotic_square(self):
        semsq = get_test_semiotic_square()
        assembler = self.make_assembler()
        html = assembler.to_html(html_base=HTMLSemioticSquareViz(semsq).get_html(num_terms=6))
        if sys.version_info.major == 2:
            self.assertEqual(type(html), unicode)
        else:
            self.assertEqual(type(html), str)
        self.assertFalse('<!-- EXTRA LIBS -->' in html)
        self.assertFalse('<!-- INSERT SCRIPT -->' in html)
        self.assertTrue('Republican' in html)

    def test_save_svg_button(self):
        scatterplot_structure = ScatterplotStructure(self.make_adapter(), save_svg_button=True)
        assembly = BasicHTMLFromScatterplotStructure(scatterplot_structure)
        html = assembly.to_html()
        self.assertEqual(scatterplot_structure.call_build_visualization_in_javascript(), self.get_params({11: 'true'}))
        self.assertFalse('<!-- INSERT SCRIPT -->' in html)

    def test_protocol_is_https(self):
        html = self.make_assembler().to_html(protocol='https')
        self.assertTrue(self._https_script_is_present(html))
        self.assertFalse(self._http_script_is_present(html))

    def _test_protocol_is_http(self):
        html = self.make_assembler().to_html(protocol='http')
        self.assertFalse(self._https_script_is_present(html))
        self.assertTrue(self._http_script_is_present(html))

    def _http_script_is_present(self, html):
        return 'src="http://' in html

    def _https_script_is_present(self, html):
        return 'src="https://' in html

    def test_protocol_default_d3_url(self):
        html = self.make_assembler().to_html()
        self.assertTrue(DEFAULT_D3_URL in html)
        html = self.make_assembler().to_html(d3_url='d3.js')
        self.assertTrue(DEFAULT_D3_URL not in html)
        self.assertTrue('d3.js' in html)

    def test_protocol_default_d3_chromatic_url(self):
        html = self.make_assembler().to_html()
        self.assertTrue(DEFAULT_D3_SCALE_CHROMATIC in html)
        html = self.make_assembler().to_html(d3_scale_chromatic_url='d3-scale-chromatic.v1.min.js')
        self.assertTrue(DEFAULT_D3_SCALE_CHROMATIC not in html)
        self.assertTrue('d3-scale-chromatic.v1.min.js' in html)

    def test_protocol_defaults_to_http(self):
        self.assertEqual(self.make_assembler().to_html(protocol='http'), self.make_assembler().to_html())

    def test_raise_invalid_protocol_exception(self):
        with self.assertRaisesRegexp(BaseException, 'Invalid protocol: ftp.  Protocol must be either http or https.'):
            self.make_assembler().to_html(protocol='ftp')

    def test_height_width_default(self):
        scatterplot_structure = ScatterplotStructure(self.make_adapter())
        self.assertEqual(scatterplot_structure.call_build_visualization_in_javascript(), self.get_params())

    def test_color(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, color='d3.interpolatePurples').call_build_visualization_in_javascript(), self.get_params({3: 'd3.interpolatePurples'}))

    def test_full_doc(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, use_full_doc=True).call_build_visualization_in_javascript(), self.get_params({5: 'true'}))

    def test_grey_zero_scores(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, grey_zero_scores=True).call_build_visualization_in_javascript(), self.get_params({6: 'true'}))

    def test_chinese_mode(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, asian_mode=True).call_build_visualization_in_javascript(), self.get_params({7: 'true'}))

    def test_reverse_sort_scores_for_not_category(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, reverse_sort_scores_for_not_category=False).call_build_visualization_in_javascript(), self.get_params({12: 'false'}))

    def test_height_width_nondefault(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000).call_build_visualization_in_javascript(), self.get_params({0: '1000'}))
        self.assertEqual(ScatterplotStructure(visualization_data, height_in_pixels=60).call_build_visualization_in_javascript(), self.get_params({1: '60'}))
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60).call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60'}))

    def test_use_non_text_features(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60, use_non_text_features=True).call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60', 8: 'true'}))

    def test_show_characteristic(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60, show_characteristic=False).call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60', 9: 'false'}))

    def test_max_snippets(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60, max_snippets=None).call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60'}))
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60, max_snippets=100).call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60', 2: '100'}))

    def test_word_vec_use_p_vals(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60, word_vec_use_p_vals=True).call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60', 10: 'true'}))

    def test_max_p_val(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60, word_vec_use_p_vals=True, max_p_val=0.01).call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60', 10: 'true', 13: '0.01'}))

    def test_p_value_colors(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60, word_vec_use_p_vals=True, p_value_colors=True).call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60', 10: 'true', 14: 'true'}))

    def test_x_label(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60, x_label='x label').call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60', 15: '"x label"'}))

    def test_y_label(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, width_in_pixels=1000, height_in_pixels=60, y_label='y label').call_build_visualization_in_javascript(), self.get_params({0: '1000', 1: '60', 16: '"y label"'}))

    def test_full_data(self):
        visualization_data = self.make_adapter()
        full_data = 'customFullDataFunction()'
        self.assertEqual(ScatterplotStructure(visualization_data, full_data=full_data).call_build_visualization_in_javascript(), self.get_params({17: full_data}))

    def test_show_top_terms(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, show_top_terms=False).call_build_visualization_in_javascript(), self.get_params({18: 'false'}))
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, show_top_terms=True).call_build_visualization_in_javascript(), self.get_params({18: 'true'}))
        self.assertEqual(ScatterplotStructure(visualization_data).call_build_visualization_in_javascript(), self.get_params({18: 'true'}))

    def test_show_neutral(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data).call_build_visualization_in_javascript(), self.get_params({19: 'false'}))
        self.assertEqual(ScatterplotStructure(visualization_data, show_neutral=True).call_build_visualization_in_javascript(), self.get_params({19: 'true'}))

    def test_get_tooltip_content(self):
        visualization_data = self.make_adapter()
        f = "(function(x) {return 'Original X: ' + x.ox;})"
        self.assertEqual(ScatterplotStructure(visualization_data, get_tooltip_content=f).call_build_visualization_in_javascript(), self.get_params({20: f}))

    def test_x_axis_labels(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, x_axis_values=[1, 2, 3]).call_build_visualization_in_javascript(), self.get_params({21: '[1, 2, 3]'}))

    def test_y_axis_labels(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, y_axis_values=[4, 5, 6]).call_build_visualization_in_javascript(), self.get_params({22: '[4, 5, 6]'}))

    def test_color_func(self):
        visualization_data = self.make_adapter()
        color_func = 'function colorFunc(d) {var c = d3.hsl(d3.interpolateRdYlBu(d.x)); c.s *= d.y;\treturn c;}'
        self.assertEqual(ScatterplotStructure(visualization_data, color_func=color_func).call_build_visualization_in_javascript(), self.get_params({23: color_func}))

    def test_show_axes(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, show_axes=False).call_build_visualization_in_javascript(), self.get_params({24: 'false'}))

    def test_show_extra(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, show_extra=True).call_build_visualization_in_javascript(), self.get_params({25: 'true'}))

    def test_do_censor_points(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, do_censor_points=False).call_build_visualization_in_javascript(), self.get_params({26: 'false'}))

    def test_center_label_over_points(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, center_label_over_points=True).call_build_visualization_in_javascript(), self.get_params({27: 'true'}))

    def test_x_axis_labels_over_points(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, x_axis_labels=['Lo', 'Hi']).call_build_visualization_in_javascript(), self.get_params({28: '["Lo", "Hi"]'}))

    def test_y_axis_labels_over_points(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, y_axis_labels=['Lo', 'Hi']).call_build_visualization_in_javascript(), self.get_params({29: '["Lo", "Hi"]'}))

    def test_topic_model_preview_size(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, topic_model_preview_size=20).call_build_visualization_in_javascript(), self.get_params({30: '20'}))

    def test_vertical_lines(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, vertical_lines=[20, 31]).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({31: '[20, 31]'}))

    def test_horizontal_line_y_position(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, horizontal_line_y_position=0).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({32: '0'}))

    def test_vertical_line_x_position(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, vertical_line_x_position=3).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({33: '3'}))

    def test_unifed_context(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, unified_context=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({34: 'true'}))

    def test_show_category_headings(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, show_category_headings=False).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({35: 'false'}))

    def test_show_cross_axes(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, show_cross_axes=False).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({36: 'false'}))

    def test_div_name(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, div_name='divvydivvy').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({37: '"divvydivvy"'}))

    def test_alternative_term_func(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, alternative_term_func='(function(termDict) {return true;})').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({38: '(function(termDict) {return true;})'}))

    def test_include_all_contexts(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, include_all_contexts=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({39: 'true'}))

    def test_show_axes_and_cross_hairs(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, show_axes_and_cross_hairs=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({40: 'true'}))

    def test_x_axis_values_format(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, x_axis_values_format='.4f').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({41: '".4f"'}))

    def test_y_axis_values_format(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, y_axis_values_format='.5f').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({42: '".5f"'}))

    def test_match_full_line(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, match_full_line=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({43: 'true'}))

    def test_max_overlapping(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, max_overlapping=10).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({44: '10'}))

    def test_show_corpus_stats(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, show_corpus_stats=False).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({45: 'false'}))

    def test_sort_doc_labels_by_name(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, sort_doc_labels_by_name=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({46: 'true'}))

    def test_always_jump(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, always_jump=False).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({47: 'false'}))

    def test_highlight_selected_category(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, highlight_selected_category=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({48: 'true'}))

    def test_show_diagonal(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, show_diagonal=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({49: 'true'}))

    def test_use_global_scale(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, use_global_scale=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({50: 'true'}))

    def test_enable_term_category_description(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, enable_term_category_description=False).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({51: 'false'}))

    def test_get_custom_term_html(self):
        visualization_data = self.make_adapter()
        html = '(function(x) {return "Term: " + x.term})'
        params = ScatterplotStructure(visualization_data, get_custom_term_html=html).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({52: html}))

    def test_header_names(self):
        visualization_data = self.make_adapter()
        header_names = {'upper': 'Upper Header Name', 'lower': 'Lower Header Name'}
        params = ScatterplotStructure(visualization_data, header_names=header_names).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({53: '{"upper": "Upper Header Name", "lower": "Lower Header Name"}'}))

    def test_header_sorting_algos(self):
        visualization_data = self.make_adapter()
        header_sorting_algos = {'upper': '(function(a, b) {return b.s - a.s})', 'lower': '(function(a, b) {return a.s - b.s})'}
        params = ScatterplotStructure(visualization_data, header_sorting_algos=header_sorting_algos).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({54: '{"lower": (function(a, b) {return a.s - b.s}), "upper": (function(a, b) {return b.s - a.s})}'}))

    def test_ignore_categories(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, ignore_categories=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({55: 'true'}))

    def test_background_labels(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, background_labels=[{'Text': 'Topic 0', 'X': 0.5242579971278757, 'Y': 0.8272937510221724}, {'Text': 'Topic 1', 'X': 0.7107755717675702, 'Y': 0.5034326824672314}, {'Text': 'Topic 2', 'X': 0.09014690078982, 'Y': 0.6261596586530888}]).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({56: '[{"Text": "Topic 0", "X": 0.5242579971278757, "Y": 0.8272937510221724}, {"Text": "Topic 1", "X": 0.7107755717675702, "Y": 0.5034326824672314}, {"Text": "Topic 2", "X": 0.09014690078982, "Y": 0.6261596586530888}]'}))

    def test_label_priority_column(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, label_priority_column='LabelPriority').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({57: '"LabelPriority"'}))

    def test_text_color_column(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, text_color_column='TextColor').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({58: '"TextColor"'}))

    def test_suppress_label_column(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, suppress_text_column='Suppress').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({59: '"Suppress"'}))

    def test_background_color(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, background_color='#444444').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({60: '"#444444"'}))

    def test_censor_point_column(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, censor_point_column='CensorPoint').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({61: '"CensorPoint"'}))

    def test_right_order_column(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, right_order_column='Priority').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({62: '"Priority"'}))

    def test_sentence_piece(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, subword_encoding='RoBERTa').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({63: '"RoBERTa"'}))

    def test_top_terms_length(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, top_terms_length=5).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({64: '5'}))

    def test_top_terms_left_buffer(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, top_terms_left_buffer=10).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({65: '10'}))

    def test_get_column_header_html(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, get_column_header_html='function(a,b,c,d,e) {return "X"}').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({66: 'function(a,b,c,d,e) {return "X"}'}))

    def test_term_word(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, term_word='Phone').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({67: '"Phone"'}))

    def test_show_term_etc(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, show_term_etc=False).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({68: 'false'}))

    def test_sort_contexts_by_meta(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, sort_contexts_by_meta=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({69: 'true'}))

    def test_suppress_circles(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, suppress_circles=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({70: 'true'}))

    def test_text_size_column(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, text_size_column='TextSize').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({71: '"TextSize"'}))

    def test_category_colors(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, category_colors={'Democratic': '0000FF', 'Republican': 'FF0000'}).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({72: '{"Democratic": "0000FF", "Republican": "FF0000"}'}))

    def test_document_word(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, document_word='paragraph').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({73: '"paragraph"', 74: '"paragraphs"'}))

    def test_document_word_plural(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, document_word_plural='multiple paragraphs').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({74: '"multiple paragraphs"'}))

    def test_category_order(self):
        visualization_data = self.make_adapter()
        self.assertEqual(ScatterplotStructure(visualization_data, category_order=['Dem', 'Rep']).call_build_visualization_in_javascript(), self.get_params({75: '["Dem", "Rep"]'}))

    def test_include_gradient(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, include_gradient=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({76: 'true'}))

    def test_left_gradient_term(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, left_gradient_term='Left Gradient').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({77: '"Left Gradient"'}))

    def test_middle_gradient_term(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, middle_gradient_term='Middle Gradient').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({78: '"Middle Gradient"'}))

    def test_right_gradient_term(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, right_gradient_term='Right Gradient').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({79: '"Right Gradient"'}))

    def test_gradient_text_color(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, gradient_text_color='black').call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({80: '"black"'}))

    def test_gradient_colors(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, gradient_colors=['#0000ff', '#fe0100', '#00ff00']).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({81: '["#0000ff", "#fe0100", "#00ff00"]'}))

    def test_category_term_score_scaler(self):
        visualization_data = self.make_adapter()
        cat_term_scaler = '(scores => (maxScore=>scores.flatMap(score=>-0.5*(1 + score/maxScore)))(Math.max(...scores.flatMap(Math.abs))))'
        params = ScatterplotStructure(visualization_data, category_term_score_scaler=cat_term_scaler).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({82: cat_term_scaler}))

    def test_show_chart(self):
        visualization_data = self.make_adapter()
        params = ScatterplotStructure(visualization_data, show_chart=True).call_build_visualization_in_javascript()
        self.assertEqual(params, self.get_params({83: 'true'}))

class TestScatterChart(TestCase):

    def test_to_json(self):
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0).to_dict('hamlet')
        self.assertEqual(set(j.keys()), set(['info', 'data']))
        self.assertEqual(set(j['info'].keys()), set(['not_category_name', 'category_name', 'category_terms', 'not_category_terms', 'category_internal_name', 'not_category_internal_names', 'neutral_category_internal_names', 'extra_category_internal_names', 'categories']))
        expected = {'x': 0.0, 'y': 0.42, 'ox': 0, 'oy': 0.42, 'term': 'art', 'cat25k': 758, 'ncat25k': 0, 'neut25k': 0, 'neut': 0, 'extra25k': 0, 'extra': 0, 's': 0.5, 'os': 3, 'bg': 3}
        datum = self._get_data_example(j)
        for var in ['cat25k', 'ncat25k']:
            np.testing.assert_almost_equal(expected[var], datum[var], decimal=1)
        self.assertEqual(set(expected.keys()), set(datum.keys()))
        self.assertEqual(expected['term'], datum['term'])

    def test_to_dict_without_categories(self):
        tdm = get_term_doc_matrix_without_categories()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        with self.assertRaises(NeedToInjectCoordinatesException):
            scatter_chart.to_dict_without_categories()
        x_coords = tdm.get_term_doc_mat().sum(axis=0).A1
        y_coords = tdm.get_term_doc_mat().astype(bool).astype(int).sum(axis=0).A1
        scatter_chart.inject_coordinates(original_x=x_coords, original_y=y_coords, x_coords=scale(x_coords), y_coords=scale(y_coords))
        j = scatter_chart.to_dict_without_categories()
        self.assertIsInstance(j, dict)
        self.assertEqual(set(j.keys()), set(['data']))
        self.assertEqual(len(j['data']), tdm.get_num_terms())
        self.assertEqual(j['data'][-1], {'cat': 4, 'cat25k': 735, 'ox': 4, 'oy': 3, 'term': 'speak', 'x': 1.0, 'y': 1.0})

    def test_resuse_is_disabled(self):
        corpus = get_test_corpus()
        sc = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0)
        sc.to_dict('hamlet')
        with self.assertRaises(Exception):
            sc.to_dict('hamlet')

    def test_score_transform(self):
        corpus = get_test_corpus()
        sc = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0)
        d1 = sc.to_dict('hamlet')
        sc = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0, score_transform=lambda x: x)
        d2 = sc.to_dict('hamlet')
        assert sum([datum['s'] for datum in d1['data']]) != sum([datum['s'] for datum in d2['data']])

    def test_multi_categories(self):
        corpus = get_test_corpus()
        j_vs_all = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0).to_dict('hamlet')
        j_vs_swift = ScatterChart(term_doc_matrix=corpus, minimum_term_frequency=0).to_dict('hamlet', not_categories=['swift'])
        self.assertNotEqual(set(j_vs_all['info']['not_category_internal_names']), set(j_vs_swift['info']['not_category_internal_names']))
        self.assertEqual(j_vs_all['info']['categories'], corpus.get_categories())
        self.assertEqual(j_vs_swift['info']['categories'], corpus.get_categories())

    def test_title_case_names(self):
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0).to_dict('hamlet', 'HAMLET', 'NOT HAMLET')
        self.assertEqual(j['info']['category_name'], 'HAMLET')
        self.assertEqual(j['info']['not_category_name'], 'NOT HAMLET')
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0).to_dict('hamlet', 'HAMLET', 'NOT HAMLET', title_case_names=True)
        self.assertEqual(j['info']['category_name'], 'Hamlet')
        self.assertEqual(j['info']['not_category_name'], 'Not Hamlet')

    def _get_data_example(self, j):
        return [t for t in j['data'] if t['term'] == 'art'][0]

    def test_terms_to_include(self):
        tdm = build_hamlet_jz_term_doc_mat()
        terms_to_include = list(sorted(['both worlds', 'thou', 'the', 'of', 'st', 'returned', 'best']))
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, terms_to_include=terms_to_include).to_dict('hamlet', 'HAMLET', 'NOT HAMLET')
        self.assertEqual(list(sorted((t['term'] for t in j['data']))), terms_to_include)

    def test_p_vals(self):
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, term_significance=LogOddsRatioUninformativeDirichletPrior()).to_dict('hamlet')
        datum = self._get_data_example(j)
        self.assertIn('p', datum.keys())

    def test_inject_coordinates(self):
        tdm = build_hamlet_jz_term_doc_mat()
        freq_df = tdm.get_term_freq_df()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates([], [])
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(freq_df[freq_df.columns[0]], [])
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates([], freq_df[freq_df.columns[0]])
        x = freq_df[freq_df.columns[1]].astype(np.float64)
        y = freq_df[freq_df.columns[0]].astype(np.float64)
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(x, y)
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(x, y / y.max())
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(x / x.max(), y)
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(-x / x.max(), -y / y.max())
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(-x / x.max(), y / y.max())
        with self.assertRaises(CoordinatesNotRightException):
            scatter_chart.inject_coordinates(x / x.max(), -y / y.max())
        scatter_chart.inject_coordinates(x / x.max(), y / y.max())

    def test_inject_metadata_term_lists(self):
        tdm = build_hamlet_jz_term_doc_mat()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        with self.assertRaises(TermDocMatrixHasNoMetadataException):
            scatter_chart.inject_metadata_term_lists({'blah': ['a', 'adsf', 'asfd']})
        scatter_chart = ScatterChart(term_doc_matrix=build_hamlet_jz_corpus_with_meta(), minimum_term_frequency=0, use_non_text_features=True)
        with self.assertRaises(TypeError):
            scatter_chart.inject_metadata_term_lists({'blash': [3, 1]})
        with self.assertRaises(TypeError):
            scatter_chart.inject_metadata_term_lists({3: ['a', 'b']})
        with self.assertRaises(TypeError):
            scatter_chart.inject_metadata_term_lists({'a': {'a', 'b'}})
        with self.assertRaises(TypeError):
            scatter_chart.inject_metadata_term_lists(3)
        self.assertEqual(type(scatter_chart.inject_metadata_term_lists({'a': ['a', 'b']})), ScatterChart)
        j = scatter_chart.to_dict('hamlet')
        self.assertEqual(set(j.keys()), set(['info', 'data', 'metalists']))
        self.assertEqual(set(j['info'].keys()), set(['not_category_name', 'category_name', 'category_terms', 'not_category_terms', 'category_internal_name', 'not_category_internal_names', 'extra_category_internal_names', 'neutral_category_internal_names', 'categories']))

    def test_inject_metadata_descriptions(self):
        tdm = build_hamlet_jz_corpus_with_meta()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        with self.assertRaises(AssertionError):
            scatter_chart.inject_metadata_descriptions(3323)
        if sys.version_info > (3, 0):
            "\n            with self.assertRaisesRegex(Exception, 'The following meta data terms are not present: blah'):\n                scatter_chart.inject_metadata_descriptions({'blah': 'asjdkflasdjklfsadjk jsdkafsd'})\n            with self.assertRaisesRegex(Exception, 'The following meta data terms are not present: cat2'):\n                scatter_chart.inject_metadata_descriptions({'cat1': 'asjdkflasdjklfsadjk jsdkafsd', 'cat2': 'asdf'})\n            "
        assert scatter_chart == scatter_chart.inject_metadata_descriptions({'cat1': 'asjdkflasdjklfsadjk jsdkafsd'})
        j = scatter_chart.to_dict('hamlet')
        self.assertEqual(set(j.keys()), set(['info', 'data', 'metadescriptions']))

    def test_inject_term_colors(self):
        tdm = build_hamlet_jz_corpus_with_meta()
        freq_df = tdm.get_term_freq_df()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        scatter_chart.inject_term_colors({'t1': '00ffee'})
        j = scatter_chart.to_dict('hamlet')
        self.assertIn('term_colors', j['info'])

    def test_inject_coordinates_original(self):
        tdm = build_hamlet_jz_term_doc_mat()
        freq_df = tdm.get_term_freq_df()
        scatter_chart = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0)
        x = freq_df[freq_df.columns[1]].astype(np.float64)
        y = freq_df[freq_df.columns[0]].astype(np.float64)
        scatter_chart.inject_coordinates(x / x.max(), y / y.max(), original_x=x, original_y=y)
        j = scatter_chart.to_dict('hamlet')
        self.assertEqual(j['data'][0].keys(), {'x', 'os', 'y', 'ncat25k', 'neut', 'cat25k', 'ox', 'neut25k', 'extra25k', 'extra', 'oy', 'term', 's', 'bg'})
        and_term = [t for t in j['data'] if t['term'] == 'and'][0]
        self.assertEqual(and_term['ox'], 0)
        self.assertEqual(and_term['oy'], 1)

    def test_to_json_use_non_text_features(self):
        tdm = build_hamlet_jz_corpus_with_meta()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, use_non_text_features=True).to_dict('hamlet')
        self.assertEqual(set(j.keys()), set(['info', 'data']))
        self.assertEqual(set(j['info'].keys()), set(['not_category_name', 'category_name', 'category_terms', 'not_category_terms', 'category_internal_name', 'not_category_internal_names', 'extra_category_internal_names', 'neutral_category_internal_names', 'categories']))
        self.assertEqual({t['term'] for t in j['data']}, {'cat1'})

    def test_max_terms(self):
        tdm = build_hamlet_jz_term_doc_mat()
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, max_terms=2).to_dict('hamlet')
        self.assertEqual(2, len(j['data']))
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, max_terms=10).to_dict('hamlet')
        self.assertEqual(10, len(j['data']))
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, pmi_threshold_coefficient=0, max_terms=10000).to_dict('hamlet')
        self.assertEqual(len(tdm.get_term_freq_df()), len(j['data']))
        j = ScatterChart(term_doc_matrix=tdm, minimum_term_frequency=0, pmi_threshold_coefficient=0, max_terms=None).to_dict('hamlet')
        self.assertEqual(len(tdm.get_term_freq_df()), len(j['data']))

class TestPriorFactory(TestCase):

    def test_all_categories(self):
        corpus = get_test_corpus()
        priors, my_corpus = PriorFactory(corpus, starting_count=0, category='hamlet').use_all_categories().build()
        tdf = corpus.get_term_freq_df()
        self.assertEqual(len(priors), len(tdf))
        np.testing.assert_equal(priors.values, corpus.get_term_freq_df().sum(axis=1).values)

    def test_neutral_categories(self):
        corpus = get_test_corpus()
        priors = PriorFactory(corpus, 'hamlet', starting_count=0.001, not_categories=['swift']).use_neutral_categories().get_priors()
        self.assertEqual(priors.min(), 0.001)
        self.assertEqual(priors.shape[0], corpus._X.shape[1])
        corpus = get_test_corpus()
        priors = PriorFactory(corpus, 'hamlet', starting_count=0.001, not_categories=['swift']).use_neutral_categories().drop_zero_priors().get_priors()
        jzcnts = corpus.get_term_freq_df()['jay-z/r. kelly freq'].where(lambda x: x > 0).dropna()
        np.testing.assert_equal(priors.values, jzcnts.values + 0.001)

    def test_get_general_term_frequencies(self):
        corpus = get_test_corpus()
        fact = PriorFactory(corpus, category='hamlet', not_categories=['swift'], starting_count=0).use_general_term_frequencies().use_all_categories()
        priors, clean_corpus = fact.build()
        expected_prior = pd.merge(corpus.get_term_doc_count_df(), corpus.get_term_and_background_counts()[['background']], left_index=True, right_index=True, how='left').fillna(0.0).sum(axis=1)
        np.testing.assert_allclose(priors.values, expected_prior.values)

    def test_align_to_target(self):
        full_corpus = get_test_corpus()
        corpus = full_corpus.remove_categories(['swift'])
        priors = PriorFactory(full_corpus).use_all_categories().get_priors()
        with self.assertRaises(ValueError):
            LogOddsRatioInformativeDirichletPrior(priors).get_scores(*corpus.get_term_freq_df().values.T)
        priors = PriorFactory(full_corpus).use_all_categories().align_to_target(corpus).get_priors()
        LogOddsRatioInformativeDirichletPrior(priors).get_scores(*corpus.get_term_freq_df().values.T)

    def test_use_categories(self):
        full_corpus = get_test_corpus()
        priors = PriorFactory(full_corpus).use_categories(['swift']).get_priors()
        corpus = full_corpus.remove_categories(['swift'])
        with self.assertRaises(ValueError):
            LogOddsRatioInformativeDirichletPrior(priors).get_scores(*corpus.get_term_freq_df().values.T)
        priors = PriorFactory(full_corpus).use_all_categories().align_to_target(corpus).get_priors()
        LogOddsRatioInformativeDirichletPrior(priors).get_scores(*corpus.get_term_freq_df().values.T)

    def test_get_custom_term_frequencies(self):
        corpus = get_test_corpus()
        fact = PriorFactory(corpus, starting_count=0.04).use_custom_term_frequencies(pd.Series({'halt': 3, 'i': 8})).drop_zero_priors()
        priors, clean_corpus = fact.build()
        self.assertEqual(set(clean_corpus.get_terms()), {'i', 'halt'})
        np.testing.assert_equal(priors.sort_values().values, [3.04, 8.04])

