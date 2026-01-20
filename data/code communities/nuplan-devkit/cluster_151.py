# Cluster 151

@dataclass
class AgentStatePlot(BaseScenarioPlot):
    """A dataclass for agent state plot."""
    data_sources: Dict[int, Dict[str, ColumnDataSource]] = field(default_factory=dict)
    plots: Dict[str, GlyphRenderer] = field(default_factory=dict)
    track_id_history: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        """Initialize track id history."""
        super().__post_init__()
        if not self.track_id_history:
            self.track_id_history = {}

    def _get_track_id(self, track_id: str) -> Union[int, float]:
        """
        Get a number representation for track ids.
        :param track_id: Agent track id.
        :return A number representation for a track id.
        """
        if track_id == 'null' or not self.track_id_history:
            return np.nan
        number_track_id = self.track_id_history.get(track_id, None)
        if not number_track_id:
            self.track_id_history[track_id] = len(self.track_id_history)
            number_track_id = len(self.track_id_history)
        return number_track_id

    def update_plot(self, main_figure: Figure, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param frame_index: Frame index.
        :param doc: Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return
        self.render_event.set()
        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                data_sources = self.data_sources.get(frame_index, None)
                if not data_sources:
                    return
                for category, data_source in data_sources.items():
                    plot = self.plots.get(category, None)
                    data = dict(data_source.data)
                    if plot is None:
                        agent_color = simulation_tile_agent_style.get(category)
                        self.plots[category] = main_figure.multi_polygons(xs='xs', ys='ys', fill_color=agent_color['fill_color'], fill_alpha=agent_color['fill_alpha'], line_color=agent_color['line_color'], line_width=agent_color['line_width'], source=data)
                        agent_hover = HoverTool(renderers=[self.plots[category]], tooltips=[('center_x [m]', '@center_xs{0.2f}'), ('center_y [m]', '@center_ys{0.2f}'), ('velocity_x [m/s]', '@velocity_xs{0.2f}'), ('velocity_y [m/s]', '@velocity_ys{0.2f}'), ('speed [m/s]', '@speeds{0.2f}'), ('heading [rad]', '@headings{0.2f}'), ('type', '@agent_type'), ('track token', '@track_token')])
                        main_figure.add_tools(agent_hover)
                    else:
                        self.plots[category].data_source.data = data
                self.render_event.clear()
            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update agents data sources.
        :param history: SimulationHistory time-series data.
        """
        if not self.data_source_condition:
            return
        with self.data_source_condition:
            for frame_index, sample in enumerate(history.data):
                if not isinstance(sample.observation, DetectionsTracks):
                    continue
                tracked_objects = sample.observation.tracked_objects
                frame_dict = {}
                for tracked_object_type_name, tracked_object_type in tracked_object_types.items():
                    corner_xs = []
                    corner_ys = []
                    track_ids = []
                    track_tokens = []
                    agent_types = []
                    center_xs = []
                    center_ys = []
                    velocity_xs = []
                    velocity_ys = []
                    speeds = []
                    headings = []
                    for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type):
                        agent_corners = tracked_object.box.all_corners()
                        corners_x = [corner.x for corner in agent_corners]
                        corners_y = [corner.y for corner in agent_corners]
                        corners_x.append(corners_x[0])
                        corners_y.append(corners_y[0])
                        corner_xs.append([[corners_x]])
                        corner_ys.append([[corners_y]])
                        center_xs.append(tracked_object.center.x)
                        center_ys.append(tracked_object.center.y)
                        velocity_xs.append(tracked_object.velocity.x)
                        velocity_ys.append(tracked_object.velocity.y)
                        speeds.append(tracked_object.velocity.magnitude())
                        headings.append(tracked_object.center.heading)
                        agent_types.append(tracked_object_type.fullname)
                        track_ids.append(self._get_track_id(tracked_object.track_token))
                        track_tokens.append(tracked_object.track_token)
                    agent_states = BokehAgentStates(xs=corner_xs, ys=corner_ys, track_id=track_ids, track_token=track_tokens, agent_type=agent_types, center_xs=center_xs, center_ys=center_ys, velocity_xs=velocity_xs, velocity_ys=velocity_ys, speeds=speeds, headings=headings)
                    frame_dict[tracked_object_type_name] = ColumnDataSource(agent_states._asdict())
                self.data_sources[frame_index] = frame_dict
                self.data_source_condition.notify()

@dataclass
class AgentStateHeadingPlot(BaseScenarioPlot):
    """A dataclass for agent state heading plot."""
    data_sources: Dict[int, Dict[str, ColumnDataSource]] = field(default_factory=dict)
    plots: Dict[str, GlyphRenderer] = field(default_factory=dict)

    def update_plot(self, main_figure: Figure, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param frame_index: Frame index.
        :param doc: Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return
        self.render_event.set()
        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                data_sources = self.data_sources.get(frame_index, None)
                if not data_sources:
                    return
                for category, data_source in data_sources.items():
                    plot = self.plots.get(category, None)
                    data = dict(data_source.data)
                    if plot is None:
                        agent_color = simulation_tile_agent_style.get(category)
                        self.plots[category] = main_figure.multi_line(xs='trajectory_x', ys='trajectory_y', line_color=agent_color['line_color'], line_width=agent_color['line_width'], source=data)
                    else:
                        self.plots[category].data_source.data = data
                self.render_event.clear()
            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update agent heading data sources.
        :param history: SimulationHistory time-series data.
        """
        if not self.data_source_condition:
            return
        with self.data_source_condition:
            for frame_index, sample in enumerate(history.data):
                if not isinstance(sample.observation, DetectionsTracks):
                    continue
                tracked_objects = sample.observation.tracked_objects
                frame_dict: Dict[str, Any] = {}
                for tracked_object_type_name, tracked_object_type in tracked_object_types.items():
                    trajectory_xs = []
                    trajectory_ys = []
                    for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type):
                        object_box = tracked_object.box
                        agent_trajectory = translate_longitudinally(object_box.center, distance=object_box.length / 2 + 1)
                        trajectory_xs.append([object_box.center.x, agent_trajectory.x])
                        trajectory_ys.append([object_box.center.y, agent_trajectory.y])
                    trajectories = ColumnDataSource(dict(trajectory_x=trajectory_xs, trajectory_y=trajectory_ys))
                    frame_dict[tracked_object_type_name] = trajectories
                self.data_sources[frame_index] = frame_dict
                self.data_source_condition.notify()

