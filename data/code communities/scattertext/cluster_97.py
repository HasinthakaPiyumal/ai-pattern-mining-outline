# Cluster 97

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

