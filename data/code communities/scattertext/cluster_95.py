# Cluster 95

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

def cell_height_and_cell_height_short_from_height(height: int) -> Tuple[int, int]:
    cellheight = int(height * (4.5 / 12))
    cellheightshort = height - 2 * cellheight
    return (cellheight, cellheightshort)

