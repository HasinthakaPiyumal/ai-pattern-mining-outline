# Cluster 28

class HTMLSemioticSquareViz(object):

    def __init__(self, semiotic_square):
        """
		Parameters
		----------
		semiotic_square : SemioticSquare
		"""
        self.semiotic_square_ = semiotic_square

    def get_html(self, num_terms=10):
        return self._get_style() + self._get_table(num_terms)

    def _get_style(self):
        return get_halo_td_style()

    def _get_table(self, num_terms):
        lexicons = self.semiotic_square_.get_lexicons(num_terms=num_terms)
        template = self._get_template()
        formatters = {category: self._lexicon_to_html(lexicon) for category, lexicon in lexicons.items()}
        formatters.update(self.semiotic_square_.get_labels())
        for k, v in formatters.items():
            template = template.replace('{' + k + '}', v)
        return template

    def _lexicon_to_html(self, lexicon):
        return ClickableTerms.get_clickable_lexicon(lexicon)

    def _get_template(self):
        return pkgutil.get_data('scattertext', SEMIOTIC_SQUARE_HTML_PATH).decode('utf-8')

def get_halo_td_style():
    return '\n\t\t\t<style>\n\t\t\ttd {\n\n\t                border-collapse: collapse;\n\t                box-sizing: border-box;\n\t                color: rgb(0, 0, 0);\n\t                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;\n\t                font-size: 12px;\n\t                height: auto ;\n\t                line-height: normal;\n\t                text-align: right;\n\t                text-size-adjust:100% ;\n\t                -webkit-border-horizontal-spacing: 0px;\n\t                -webkit-border-vertical-spacing:0px;\n\t\t\t}\n\t\t\t</style>'

