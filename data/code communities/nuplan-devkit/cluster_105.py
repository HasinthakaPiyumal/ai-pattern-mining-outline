# Cluster 105

class MetricSummaryCallback(AbstractMainCallback):
    """Callback to render histograms for metrics and metric aggregator."""

    def __init__(self, metric_save_path: str, metric_aggregator_save_path: str, summary_output_path: str, pdf_file_name: str, num_bins: int=20):
        """Callback to handle metric files at the end of process."""
        self._metric_save_path = Path(metric_save_path)
        self._metric_aggregator_save_path = Path(metric_aggregator_save_path)
        self._summary_output_path = Path(summary_output_path)
        if not is_s3_path(self._summary_output_path):
            self._summary_output_path.mkdir(parents=True, exist_ok=True)
        self._pdf_file_name = pdf_file_name
        self._num_bins = num_bins
        self._color_index = 0
        color_palette = cmap.get_cmap('Set1').colors + cmap.get_cmap('Set2').colors + cmap.get_cmap('Set3').colors
        self._color_choices = [mcolors.rgb2hex(color) for color in color_palette]
        self._metric_aggregator_dataframes: Dict[str, pd.DataFrame] = {}
        self._metric_statistics_dataframes: Dict[str, MetricStatisticsDataFrame] = {}

    @staticmethod
    def _read_metric_parquet_files(metric_save_path: Path, metric_reader: Callable[[Path], Any]) -> METRIC_DATAFRAME_TYPE:
        """
        Read metric parquet files with different readers.
        :param metric_save_path: Metric save path.
        :param metric_reader: Metric reader to read metric parquet files.
        :return A dictionary of {file_index: {file_name: MetricStatisticsDataFrame or pandas dataframe}}.
        """
        metric_dataframes: Dict[str, Union[MetricStatisticsDataFrame, pd.DataFrame]] = defaultdict()
        metric_file = metric_save_path.rglob('*.parquet')
        for file_index, file in enumerate(metric_file):
            try:
                if file.is_dir():
                    continue
                data_frame = metric_reader(file)
                metric_dataframes[file.stem] = data_frame
            except (FileNotFoundError, Exception):
                pass
        return metric_dataframes

    def _aggregate_metric_statistic_histogram_data(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate metric statistic histogram data.
        :return A dictionary of metric names and their aggregated data.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        for dataframe_filename, dataframe in self._metric_statistics_dataframes.items():
            histogram_data_list = aggregate_metric_statistics_dataframe_histogram_data(metric_statistics_dataframe=dataframe, metric_statistics_dataframe_index=0, metric_choices=[], scenario_types=None)
            if histogram_data_list:
                data[dataframe.metric_statistic_name] += histogram_data_list
        return data

    def _aggregate_scenario_type_score_histogram_data(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate scenario type score histogram data.
        :return A dictionary of scenario type metric name and their scenario type scores.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        for index, (dataframe_filename, dataframe) in enumerate(self._metric_aggregator_dataframes.items()):
            histogram_data_list = aggregate_metric_aggregator_dataframe_histogram_data(metric_aggregator_dataframe=dataframe, metric_aggregator_dataframe_index=index, scenario_types=['all'], dataframe_file_name=dataframe_filename)
            if histogram_data_list:
                data[f'{HistogramConstantConfig.SCENARIO_TYPE_SCORE_HISTOGRAM_NAME}_{dataframe_filename}'] += histogram_data_list
        return data

    def _assign_planner_colors(self) -> Dict[str, Any]:
        """
        Assign colors to planners.
        :return A dictionary of planner and colors.
        """
        planner_color_maps = {}
        for dataframe_filename, dataframe in self._metric_statistics_dataframes.items():
            planner_names = dataframe.planner_names
            for planner_name in planner_names:
                if planner_name not in planner_color_maps:
                    planner_color_maps[planner_name] = self._color_choices[self._color_index % len(self._color_choices)]
                    self._color_index += 1
        return planner_color_maps

    def _save_to_pdf(self, matplotlib_plots: List[Any]) -> None:
        """
        Save a list of matplotlib plots to a pdf file.
        :param matplotlib_plots: A list of matplotlib plots.
        """
        file_name = safe_path_to_string(self._summary_output_path / self._pdf_file_name)
        pp = PdfPages(file_name)
        for fig in matplotlib_plots[::-1]:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close()

    @staticmethod
    def _render_ax_hist(ax: Any, x_values: npt.NDArray[np.float64], x_axis_label: str, y_axis_label: str, bins: npt.NDArray[np.float64], label: str, color: str, ax_title: str) -> None:
        """
        Render axis with histogram bins.
        :param ax: Matplotlib axis.
        :param x_values: An array of histogram x-axis values.
        :param x_axis_label: Label in the x-axis.
        :param y_axis_label: Label in the y-axis.
        :param bins: An array of histogram bins.
        :param label: Legend name for the bins.
        :param color: Color for the bins.
        :param ax_title: Axis title.
        """
        ax.hist(x=x_values, bins=bins, label=label, color=color, weights=np.ones(len(x_values)) / len(x_values))
        ax.set_xlabel(x_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.x_axis_label_size)
        ax.set_ylabel(y_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.y_axis_label_size)
        ax.set_title(ax_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.axis_title_size)
        ax.set_ylim(ymin=0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis='both', which='major', labelsize=HistogramTabMatPlotLibPlotStyleConfig.axis_ticker_size)
        ax.legend(fontsize=HistogramTabMatPlotLibPlotStyleConfig.legend_font_size)

    @staticmethod
    def _render_ax_bar_hist(ax: Any, x_values: Union[npt.NDArray[np.float64], List[str]], x_axis_label: str, y_axis_label: str, x_range: List[str], label: str, color: str, ax_title: str) -> None:
        """
        Render axis with bar histogram.
        :param ax: Matplotlib axis.
        :param x_values: An array of histogram x-axis values.
        :param x_axis_label: Label in the x-axis.
        :param y_axis_label: Label in the y-axis.
        :param x_range: A list of histogram category names.
        :param label: Legend name for the bins.
        :param color: Color for the bins.
        :param ax_title: Axis title.
        """
        value_categories = {key: 0.0 for key in x_range}
        for value in x_values:
            value_categories[str(value)] += 1.0
        category_names = list(value_categories.keys())
        category_values: List[float] = list(value_categories.values())
        num_scenarios = sum(category_values)
        if num_scenarios != 0:
            category_values = [value / num_scenarios * 100 for value in category_values]
            category_values = np.round(category_values, decimals=HistogramTabFigureStyleConfig.decimal_places)
        ax.bar(category_names, category_values, label=label, color=color)
        ax.set_xlabel(x_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.x_axis_label_size)
        ax.set_ylabel(y_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.y_axis_label_size)
        ax.set_title(ax_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.axis_title_size)
        ax.set_ylim(ymin=0)
        ax.tick_params(axis='both', which='major', labelsize=HistogramTabMatPlotLibPlotStyleConfig.axis_ticker_size)
        ax.legend(fontsize=HistogramTabMatPlotLibPlotStyleConfig.legend_font_size)

    def _draw_histogram_plots(self, planner_color_maps: Dict[str, Any], histogram_data_dict: HistogramConstantConfig.HistogramDataType, histogram_edges: HistogramConstantConfig.HistogramEdgesDataType, n_cols: int=2) -> None:
        """
        :param planner_color_maps: Color maps from planner names.
        :param histogram_data_dict: A dictionary of histogram data.
        :param histogram_edges: A dictionary of histogram edges (bins) data.
        :param n_cols: Number of columns in subplot.
        """
        matplotlib_plots = []
        for histogram_title, histogram_data_list in tqdm(histogram_data_dict.items(), desc='Rendering histograms'):
            for histogram_data in histogram_data_list:
                color = planner_color_maps.get(histogram_data.planner_name, None)
                if not color:
                    planner_color_maps[histogram_data.planner_name] = self._color_choices[self._color_index % len(self._color_choices)]
                    color = planner_color_maps.get(histogram_data.planner_name)
                    self._color_index += 1
                n_rows = math.ceil(len(histogram_data.statistics) / n_cols)
                fig_size = min(max(6, len(histogram_data.statistics) // 5 * 5), 24)
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_size, fig_size))
                flatten_axs = axs.flatten()
                fig.suptitle(histogram_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.main_title_size)
                for index, (statistic_name, statistic) in enumerate(histogram_data.statistics.items()):
                    unit = statistic.unit
                    bins: npt.NDArray[np.float64] = np.unique(histogram_edges[histogram_title].get(statistic_name, None))
                    assert bins is not None, f'Count edge data for {statistic_name} cannot be None!'
                    x_range = get_histogram_plot_x_range(unit=unit, data=bins)
                    values = np.round(statistic.values, HistogramTabFigureStyleConfig.decimal_places)
                    if unit in ['count']:
                        self._render_ax_bar_hist(ax=flatten_axs[index], x_values=values, x_range=x_range, x_axis_label=unit, y_axis_label='Frequency (%)', label=histogram_data.planner_name, color=color, ax_title=statistic_name)
                    elif unit in ['bool', 'boolean']:
                        values = ['True' if value else 'False' for value in values]
                        self._render_ax_bar_hist(ax=flatten_axs[index], x_values=values, x_range=x_range, x_axis_label=unit, y_axis_label='Frequency (%)', label=histogram_data.planner_name, color=color, ax_title=statistic_name)
                    else:
                        self._render_ax_hist(ax=flatten_axs[index], x_values=values, bins=bins, x_axis_label=unit, y_axis_label='Frequency (%)', label=histogram_data.planner_name, color=color, ax_title=statistic_name)
                if n_rows * n_cols != len(histogram_data.statistics.values()):
                    flatten_axs[-1].set_axis_off()
                plt.tight_layout()
                matplotlib_plots.append(fig)
        self._save_to_pdf(matplotlib_plots=matplotlib_plots)

    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        start_time = time.perf_counter()
        if not self._metric_save_path.exists() and (not self._metric_aggregator_save_path.exists()):
            return
        self._metric_aggregator_dataframes = self._read_metric_parquet_files(metric_save_path=self._metric_aggregator_save_path, metric_reader=metric_aggregator_reader)
        self._metric_statistics_dataframes = self._read_metric_parquet_files(metric_save_path=self._metric_save_path, metric_reader=metric_statistics_reader)
        planner_color_maps = self._assign_planner_colors()
        histogram_data_dict = self._aggregate_metric_statistic_histogram_data()
        scenario_type_histogram_data_dict = self._aggregate_scenario_type_score_histogram_data()
        histogram_data_dict.update(scenario_type_histogram_data_dict)
        histogram_edge_data = compute_histogram_edges(bins=self._num_bins, aggregated_data=histogram_data_dict)
        self._draw_histogram_plots(planner_color_maps=planner_color_maps, histogram_data_dict=histogram_data_dict, histogram_edges=histogram_edge_data)
        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time_s))
        logger.info('Metric summary: {} [HH:MM:SS]'.format(time_str))

def get_histogram_plot_x_range(unit: str, data: npt.NDArray[np.float64]) -> Union[List[str], FactorRange]:
    """
    Get Histogram x_range based on unit and data.
    :param unit: Histogram unit.
    :param data: Histogram data.
    :return x_range in histogram plot.
    """
    x_range = None
    if unit in ['bool', 'boolean']:
        x_range = ['False', 'True']
    elif unit in ['count']:
        x_range = [str(count) for count in data]
    return x_range

class HistogramTab(BaseTab):
    """Histogram tab in nuBoard."""

    def __init__(self, doc: Document, experiment_file_data: ExperimentFileData, bins: int=HistogramTabBinSpinnerConfig.default_bins, max_scenario_names: int=20):
        """
        Histogram for metric results about simulation.
        :param doc: Bokeh html document.
        :param experiment_file_data: Experiment file data.
        :param bins: Default number of bins in histograms.
        :param max_scenario_names: Show the maximum list of scenario names in each bin, 0 or None to disable
        """
        super().__init__(doc=doc, experiment_file_data=experiment_file_data)
        self._bins = bins
        self._max_scenario_names = max_scenario_names
        self.planner_checkbox_group.name = HistogramConstantConfig.PLANNER_CHECKBOX_GROUP_NAME
        self.planner_checkbox_group.js_on_change('active', HistogramTabLoadingJSCode.get_js_code())
        self._scenario_type_multi_choice = MultiChoice(**HistogramTabScenarioTypeMultiChoiceConfig.get_config())
        self._scenario_type_multi_choice.on_change('value', self._scenario_type_multi_choice_on_change)
        self._scenario_type_multi_choice.js_on_change('value', HistogramTabUpdateWindowsSizeJSCode.get_js_code())
        self._metric_name_multi_choice = MultiChoice(**HistogramTabMetricNameMultiChoiceConfig.get_config())
        self._metric_name_multi_choice.on_change('value', self._metric_name_multi_choice_on_change)
        self._metric_name_multi_choice.js_on_change('value', HistogramTabUpdateWindowsSizeJSCode.get_js_code())
        self._bin_spinner = Spinner(**HistogramTabBinSpinnerConfig.get_config())
        self._histogram_modal_query_btn = Button(**HistogramTabModalQueryButtonConfig.get_config())
        self._histogram_modal_query_btn.js_on_click(HistogramTabLoadingJSCode.get_js_code())
        self._histogram_modal_query_btn.on_click(self._setting_modal_query_button_on_click)
        self._default_div = Div(**HistogramTabDefaultDivConfig.get_config())
        self._histogram_plots = column(self._default_div, **HistogramTabPlotConfig.get_config())
        self._histogram_plots.js_on_change('children', HistogramTabLoadingEndJSCode.get_js_code())
        self._histogram_figures: Optional[column] = None
        self._aggregated_data: Optional[HistogramConstantConfig.HistogramDataType] = None
        self._histogram_edges: Optional[HistogramConstantConfig.HistogramEdgesDataType] = None
        self._plot_data: Dict[str, List[glyph]] = defaultdict(list)
        self._init_selection()

    @property
    def bin_spinner(self) -> Spinner:
        """Return a bin spinner."""
        return self._bin_spinner

    @property
    def scenario_type_multi_choice(self) -> MultiChoice:
        """Return scenario_type_multi_choice."""
        return self._scenario_type_multi_choice

    @property
    def metric_name_multi_choice(self) -> MultiChoice:
        """Return metric_name_multi_choice."""
        return self._metric_name_multi_choice

    @property
    def histogram_plots(self) -> column:
        """Return histogram_plots."""
        return self._histogram_plots

    @property
    def histogram_modal_query_btn(self) -> Button:
        """Return histogram modal query button."""
        return self._histogram_modal_query_btn

    def _click_planner_checkbox_group(self, attr: Any) -> None:
        """
        Click event handler for planner_checkbox_group.
        :param attr: Clicked attributes.
        """
        if not self._aggregated_data and (not self._histogram_edges):
            return
        self._histogram_figures = self._render_histograms()
        self._doc.add_next_tick_callback(self._update_histogram_layouts)

    def file_paths_on_change(self, experiment_file_data: ExperimentFileData, experiment_file_active_index: List[int]) -> None:
        """
        Interface to update layout when file_paths is changed.
        :param experiment_file_data: Experiment file data.
        :param experiment_file_active_index: Active indexes for experiment files.
        """
        self._experiment_file_data = experiment_file_data
        self._experiment_file_active_index = experiment_file_active_index
        self._init_selection()
        self._update_histograms()

    def _update_histogram_layouts(self) -> None:
        """Update histogram layouts."""
        self._histogram_plots.children[0] = layout(self._histogram_figures)

    def _update_histograms(self) -> None:
        """Update histograms."""
        self._aggregated_data = self._aggregate_statistics()
        aggregated_scenario_type_score_data = self._aggregate_scenario_type_score_histogram()
        self._aggregated_data.update(aggregated_scenario_type_score_data)
        self._histogram_edges = compute_histogram_edges(aggregated_data=self._aggregated_data, bins=self._bins)
        self._histogram_figures = self._render_histograms()
        self._doc.add_next_tick_callback(self._update_histogram_layouts)

    def _setting_modal_query_button_on_click(self) -> None:
        """Setting modal query button on click helper function."""
        if self._metric_name_multi_choice.tags:
            self.window_width = self._metric_name_multi_choice.tags[0]
            self.window_height = self._metric_name_multi_choice.tags[1]
        if self._bin_spinner.value:
            self._bins = self._bin_spinner.value
        self._update_histograms()

    def _metric_name_multi_choice_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in histogram metric name.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        if self._metric_name_multi_choice.tags:
            self.window_width = self._metric_name_multi_choice.tags[0]
            self.window_height = self._metric_name_multi_choice.tags[1]

    def _scenario_type_multi_choice_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in histogram scenario type.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        if self._scenario_type_multi_choice.tags:
            self.window_width = self._scenario_type_multi_choice.tags[0]
            self.window_height = self.scenario_type_multi_choice.tags[1]

    def _adjust_plot_width_size(self, n_bins: int) -> int:
        """
        Adjust plot width size based on number of bins.
        :param n_bins: Number of bins.
        :return Width size of a histogram plot.
        """
        base_plot_width: int = self.plot_sizes[0]
        if n_bins < 20:
            return base_plot_width
        width_multiplier_factor: int = n_bins // 20 * 100
        width_size: int = min(base_plot_width + width_multiplier_factor, HistogramTabFigureStyleConfig.maximum_plot_width)
        return width_size

    def _init_selection(self) -> None:
        """Init histogram and scalar selection options."""
        planner_name_list: List[str] = []
        self.planner_checkbox_group.labels = []
        self.planner_checkbox_group.active = []
        for index, metric_statistics_dataframes in enumerate(self.experiment_file_data.metric_statistics_dataframes):
            if index not in self._experiment_file_active_index:
                continue
            for metric_statistics_dataframe in metric_statistics_dataframes:
                planner_names = metric_statistics_dataframe.planner_names
                planner_name_list += planner_names
        sorted_planner_name_list = sorted(list(set(planner_name_list)))
        self.planner_checkbox_group.labels = sorted_planner_name_list
        self.planner_checkbox_group.active = [index for index in range(len(sorted_planner_name_list))]
        self._init_multi_search_criteria_selection(scenario_type_multi_choice=self._scenario_type_multi_choice, metric_name_multi_choice=self._metric_name_multi_choice)

    def plot_vbar(self, histogram_figure_data: HistogramFigureData, counts: npt.NDArray[np.int64], category: List[str], planner_name: str, legend_label: str, color: str, scenario_names: List[str], x_values: List[str], width: float=0.4, histogram_file_name: Optional[str]=None) -> None:
        """
        Plot a vertical bar plot.
        :param histogram_figure_data: Figure class.
        :param counts: An array of counts for each category.
        :param category: A list of category (x-axis label).
        :param planner_name: Planner name.
        :param legend_label: Legend label.
        :param color: Legend color.
        :param scenario_names: A list of scenario names.
        :param x_values: X-axis values.
        :param width: Bar width.
        :param histogram_file_name: Histogram file name for the histogram data.
        """
        y_values = deepcopy(counts)
        bottom: npt.NDArray[np.int64] = np.zeros_like(counts) if histogram_figure_data.frequency_array is None else histogram_figure_data.frequency_array
        count_position = counts > 0
        bottom_arrays: npt.NDArray[np.int64] = bottom * count_position
        top = counts + bottom_arrays
        histogram_file_names = [histogram_file_name] * len(top)
        data_source = ColumnDataSource(dict(x=category, top=top, bottom=bottom_arrays, y_values=y_values, x_values=x_values, scenario_names=scenario_names, histogram_file_name=histogram_file_names))
        figure_plot = histogram_figure_data.figure_plot
        vbar = figure_plot.vbar(x='x', top='top', bottom='bottom', fill_color=color, legend_label=legend_label, width=width, source=data_source, **HistogramTabHistogramBarStyleConfig.get_config())
        self._plot_data[planner_name].append(vbar)
        HistogramTabHistogramBarStyleConfig.update_histogram_bar_figure_style(histogram_figure=figure_plot)

    def plot_histogram(self, histogram_figure_data: HistogramFigureData, hist: npt.NDArray[np.float64], edges: npt.NDArray[np.float64], planner_name: str, legend_label: str, color: str, scenario_names: List[str], x_values: List[str], histogram_file_name: Optional[str]=None) -> None:
        """
        Plot a histogram.
        Reference from https://docs.bokeh.org/en/latest/docs/gallery/histogram.html.
        :param histogram_figure_data: Histogram figure data.
        :param hist: Histogram data.
        :param edges: Histogram bin data.
        :param planner_name: Planner name.
        :param legend_label: Legend label.
        :param color: Legend color.
        :param scenario_names: A list of scenario names.
        :param x_values: A list of x value names.
        :param histogram_file_name: Histogram file name for the histogram data.
        """
        bottom: npt.NDArray[np.int64] = np.zeros_like(hist) if histogram_figure_data.frequency_array is None else histogram_figure_data.frequency_array
        hist_position = hist > 0
        bottom_arrays: npt.NDArray[np.int64] = bottom * hist_position
        top = hist + bottom_arrays
        histogram_file_names = [histogram_file_name] * len(top)
        data_source = ColumnDataSource(dict(top=top, bottom=bottom_arrays, left=edges[:-1], right=edges[1:], y_values=hist, x_values=x_values, scenario_names=scenario_names, histogram_file_name=histogram_file_names))
        figure_plot = histogram_figure_data.figure_plot
        quad = figure_plot.quad(top='top', bottom='bottom', left='left', right='right', fill_color=color, legend_label=legend_label, **HistogramTabHistogramBarStyleConfig.get_config(), source=data_source)
        self._plot_data[planner_name].append(quad)
        HistogramTabHistogramBarStyleConfig.update_histogram_bar_figure_style(histogram_figure=figure_plot)

    def _render_histogram_plot(self, title: str, x_axis_label: str, x_range: Optional[Union[List[str], FactorRange]]=None, histogram_file_name: Optional[str]=None) -> HistogramFigureData:
        """
        Render a histogram plot.
        :param title: Title.
        :param x_axis_label: x-axis label.
        :param x_range: A list of category data if specified.
        :param histogram_file_name: Histogram file name for the histogram plot.
        :return a figure.
        """
        if x_range is None:
            len_plot_width = 1
        elif isinstance(x_range, list):
            len_plot_width = len(x_range)
        else:
            len_plot_width = len(x_range.factors)
        plot_width = self._adjust_plot_width_size(n_bins=len_plot_width)
        tooltips = [('Frequency', '@y_values'), ('Values', '@x_values{safe}'), ('Scenarios', '@scenario_names{safe}')]
        if histogram_file_name:
            tooltips.append(('File', '@histogram_file_name'))
        hover_tool = HoverTool(tooltips=tooltips, point_policy='follow_mouse')
        statistic_figure = figure(**HistogramTabFigureStyleConfig.get_config(title=title, x_axis_label=x_axis_label, width=plot_width, height=self.plot_sizes[1], x_range=x_range), tools=['pan', 'wheel_zoom', 'save', 'reset', hover_tool])
        HistogramTabFigureStyleConfig.update_histogram_figure_style(histogram_figure=statistic_figure)
        return HistogramFigureData(figure_plot=statistic_figure)

    def _render_histogram_layout(self, histograms: HistogramConstantConfig.HistogramFigureDataType) -> List[column]:
        """
        Render histogram layout.
        :param histograms: A dictionary of histogram names and their histograms.
        :return: A list of lists of figures (a list per row).
        """
        layouts = []
        ncols = self.get_plot_cols(plot_width=self.plot_sizes[0], default_ncols=HistogramConstantConfig.HISTOGRAM_TAB_DEFAULT_NUMBER_COLS)
        for metric_statistics_name, statistics_data in histograms.items():
            title_div = Div(**HistogramTabFigureTitleDivStyleConfig.get_config(title=metric_statistics_name))
            figures = [histogram_figure.figure_plot for statistic_name, histogram_figure in statistics_data.items()]
            grid_plot = gridplot(figures, **HistogramTabFigureGridPlotStyleConfig.get_config(ncols=ncols, height=self.plot_sizes[1]))
            grid_layout = column(title_div, grid_plot)
            layouts.append(grid_layout)
        return layouts

    def _aggregate_scenario_type_score_histogram(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate metric aggregator data.
        :return: A dictionary of metric aggregator names and their metric scores.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        selected_scenario_types = self._scenario_type_multi_choice.value
        for index, metric_aggregator_dataframes in enumerate(self.experiment_file_data.metric_aggregator_dataframes):
            if index not in self._experiment_file_active_index:
                continue
            for metric_aggregator_filename, metric_aggregator_dataframe in metric_aggregator_dataframes.items():
                histogram_data_list = aggregate_metric_aggregator_dataframe_histogram_data(metric_aggregator_dataframe_index=index, metric_aggregator_dataframe=metric_aggregator_dataframe, scenario_types=selected_scenario_types, dataframe_file_name=metric_aggregator_filename)
                if histogram_data_list:
                    data[HistogramConstantConfig.SCENARIO_TYPE_SCORE_HISTOGRAM_NAME] += histogram_data_list
        return data

    def _aggregate_statistics(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate statistics data.
        :return A dictionary of metric names and their aggregated data.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        scenario_types = self._scenario_type_multi_choice.value
        metric_choices = self._metric_name_multi_choice.value
        if not len(scenario_types) and (not len(metric_choices)):
            return data
        if 'all' in scenario_types:
            scenario_types = None
        else:
            scenario_types = tuple(scenario_types)
        for index, metric_statistics_dataframes in enumerate(self.experiment_file_data.metric_statistics_dataframes):
            if index not in self._experiment_file_active_index:
                continue
            for metric_statistics_dataframe in metric_statistics_dataframes:
                histogram_data_list = aggregate_metric_statistics_dataframe_histogram_data(metric_statistics_dataframe=metric_statistics_dataframe, metric_statistics_dataframe_index=index, scenario_types=scenario_types, metric_choices=metric_choices)
                if histogram_data_list:
                    data[metric_statistics_dataframe.metric_statistic_name] += histogram_data_list
        return data

    def _plot_bool_histogram(self, histogram_figure_data: HistogramFigureData, values: npt.NDArray[np.float64], scenarios: List[str], planner_name: str, legend_name: str, color: str, histogram_file_name: Optional[str]=None) -> None:
        """
        Plot boolean type of histograms.
        :param histogram_figure_data: Histogram figure data.
        :param values: An array of values.
        :param scenarios: A list of scenario names.
        :param planner_name: Planner name.
        :param legend_name: Legend name.
        :param color: Plot color.
        :param histogram_file_name: Histogram file name for the histogram data.
        """
        num_true = np.nansum(values)
        num_false = len(values[values == 0])
        scenario_names: List[List[str]] = [[] for _ in range(2)]
        for index, scenario in enumerate(scenarios):
            scenario_name_index = 1 if values[index] else 0
            if not self._max_scenario_names or len(scenario_names[scenario_name_index]) < self._max_scenario_names:
                scenario_names[scenario_name_index].append(scenario)
        scenario_names_flatten = ['<br>'.join(names) if names else '' for names in scenario_names]
        counts: npt.NDArray[np.int64] = np.asarray([num_false, num_true])
        x_range = ['False', 'True']
        x_values = ['False', 'True']
        self.plot_vbar(histogram_figure_data=histogram_figure_data, category=x_range, counts=counts, planner_name=planner_name, legend_label=legend_name, color=color, scenario_names=scenario_names_flatten, x_values=x_values, histogram_file_name=histogram_file_name)
        counts = np.asarray(counts)
        if histogram_figure_data.frequency_array is None:
            histogram_figure_data.frequency_array = deepcopy(counts)
        else:
            histogram_figure_data.frequency_array += counts

    def _plot_count_histogram(self, histogram_figure_data: HistogramFigureData, values: npt.NDArray[np.float64], scenarios: List[str], planner_name: str, legend_name: str, color: str, edges: npt.NDArray[np.float64], histogram_file_name: Optional[str]=None) -> None:
        """
        Plot count type of histograms.
        :param histogram_figure_data: Histogram figure data.
        :param values: An array of values.
        :param scenarios: A list of scenario names.
        :param planner_name: Planner name.
        :param legend_name: Legend name.
        :param color: Plot color.
        :param edges: Count edges.
        :param histogram_file_name: Histogram file name for the histogram data.
        """
        uniques: Any = np.unique(values, return_inverse=True)
        unique_values: npt.NDArray[np.float64] = uniques[0]
        unique_index: npt.NDArray[np.int64] = uniques[1]
        counts = {value: 0 for value in edges}
        bin_count = np.bincount(unique_index)
        for index, count_value in enumerate(bin_count):
            counts[unique_values[index]] = count_value
        scenario_names: List[List[str]] = [[] for _ in range(len(counts))]
        for index, bin_index in enumerate(unique_index):
            if not self._max_scenario_names or len(scenario_names[bin_index]) < self._max_scenario_names:
                scenario_names[bin_index].append(scenarios[index])
        scenario_names_flatten = ['<br>'.join(names) if names else '' for names in scenario_names]
        category = [str(key) for key in counts.keys()]
        count_values: npt.NDArray[np.int64] = np.asarray(list(counts.values()))
        self.plot_vbar(histogram_figure_data=histogram_figure_data, category=category, counts=count_values, planner_name=planner_name, legend_label=legend_name, color=color, scenario_names=scenario_names_flatten, width=0.1, x_values=category, histogram_file_name=histogram_file_name)
        if histogram_figure_data.frequency_array is None:
            histogram_figure_data.frequency_array = deepcopy(count_values)
        else:
            histogram_figure_data.frequency_array += count_values

    def _plot_bin_histogram(self, histogram_figure_data: HistogramFigureData, values: npt.NDArray[np.float64], scenarios: List[str], planner_name: str, legend_name: str, color: str, edges: npt.NDArray[np.float64], histogram_file_name: Optional[str]=None) -> None:
        """
        Plot bin type of histograms.
        :param histogram_figure_data: Histogram figure data.
        :param values: An array of values.
        :param scenarios: A list of scenario names.
        :param planner_name: Planner name.
        :param legend_name: Legend name.
        :param color: Plot color.
        :param edges: Histogram bin edges.
        :param histogram_file_name: Histogram file name for the histogram data.
        """
        hist, bins = np.histogram(values, bins=edges)
        value_bin_index: npt.NDArray[np.int64] = np.asarray(np.digitize(values, bins=bins[:-1]))
        scenario_names: List[List[str]] = [[] for _ in range(len(hist))]
        for index, bin_index in enumerate(value_bin_index):
            if not self._max_scenario_names or len(scenario_names[bin_index - 1]) < self._max_scenario_names:
                scenario_names[bin_index - 1].append(scenarios[index])
        scenario_names_flatten = ['<br>'.join(names) if names else '' for names in scenario_names]
        bins = np.round(bins, HistogramTabFigureStyleConfig.decimal_places)
        x_values = [str(value) + ' - ' + str(bins[index + 1]) for index, value in enumerate(bins[:-1])]
        self.plot_histogram(histogram_figure_data=histogram_figure_data, planner_name=planner_name, legend_label=legend_name, hist=hist, edges=edges, color=color, scenario_names=scenario_names_flatten, x_values=x_values, histogram_file_name=histogram_file_name)
        if histogram_figure_data.frequency_array is None:
            histogram_figure_data.frequency_array = deepcopy(hist)
        else:
            histogram_figure_data.frequency_array += hist

    def _draw_histogram_data(self) -> HistogramConstantConfig.HistogramFigureDataType:
        """
        Draw histogram data based on aggregated data.
        :return A dictionary of metric names and theirs histograms.
        """
        histograms: HistogramConstantConfig.HistogramFigureDataType = defaultdict()
        if self._aggregated_data is None or self._histogram_edges is None:
            return histograms
        for metric_statistics_name, aggregated_histogram_data in self._aggregated_data.items():
            if metric_statistics_name not in histograms:
                histograms[metric_statistics_name] = {}
            for histogram_data in aggregated_histogram_data:
                legend_name = histogram_data.planner_name + f' ({self.get_file_path_last_name(histogram_data.experiment_index)})'
                if histogram_data.planner_name not in self.enable_planner_names:
                    continue
                color = self.experiment_file_data.file_path_colors[histogram_data.experiment_index][histogram_data.planner_name]
                for statistic_name, statistic in histogram_data.statistics.items():
                    unit = statistic.unit
                    data: npt.NDArray[np.float64] = np.unique(self._histogram_edges[metric_statistics_name].get(statistic_name, None))
                    assert data is not None, f'Count edge data for {statistic_name} cannot be None!'
                    if statistic_name not in histograms[metric_statistics_name]:
                        x_range = get_histogram_plot_x_range(unit=unit, data=data)
                        histograms[metric_statistics_name][statistic_name] = self._render_histogram_plot(title=statistic_name, x_axis_label=unit, x_range=x_range, histogram_file_name=histogram_data.histogram_file_name)
                    histogram_figure_data = histograms[metric_statistics_name][statistic_name]
                    values = statistic.values
                    if unit in ['bool', 'boolean']:
                        self._plot_bool_histogram(histogram_figure_data=histogram_figure_data, values=values, scenarios=statistic.scenarios, planner_name=histogram_data.planner_name, legend_name=legend_name, color=color, histogram_file_name=histogram_data.histogram_file_name)
                    else:
                        edges = self._histogram_edges[metric_statistics_name][statistic_name]
                        if edges is None:
                            continue
                        if unit in ['count']:
                            self._plot_count_histogram(histogram_figure_data=histogram_figure_data, values=values, scenarios=statistic.scenarios, planner_name=histogram_data.planner_name, legend_name=legend_name, color=color, edges=edges, histogram_file_name=histogram_data.histogram_file_name)
                        else:
                            self._plot_bin_histogram(histogram_figure_data=histogram_figure_data, values=values, scenarios=statistic.scenarios, planner_name=histogram_data.planner_name, legend_name=legend_name, color=color, edges=edges, histogram_file_name=histogram_data.histogram_file_name)
        sorted_histograms = {}
        if HistogramConstantConfig.SCENARIO_TYPE_SCORE_HISTOGRAM_NAME in histograms:
            sorted_histograms[HistogramConstantConfig.SCENARIO_TYPE_SCORE_HISTOGRAM_NAME] = histograms[HistogramConstantConfig.SCENARIO_TYPE_SCORE_HISTOGRAM_NAME]
        sorted_histogram_keys = sorted((key for key in histograms.keys() if key != HistogramConstantConfig.SCENARIO_TYPE_SCORE_HISTOGRAM_NAME), reverse=False)
        sorted_histograms.update({key: histograms[key] for key in sorted_histogram_keys})
        return sorted_histograms

    def _render_histograms(self) -> List[column]:
        """
        Render histograms across all scenarios based on a scenario type.
        :return: A list of lists of figures (a list per row).
        """
        histograms = self._draw_histogram_data()
        layouts = self._render_histogram_layout(histograms)
        if not layouts:
            layouts = [column(self._default_div, width=HistogramTabPlotConfig.default_width, **HistogramTabPlotConfig.get_config())]
        return layouts

