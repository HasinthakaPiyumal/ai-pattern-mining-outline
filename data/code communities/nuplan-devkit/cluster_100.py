# Cluster 100

def approximate_derivatives(y: npt.NDArray[np.float32], x: npt.NDArray[np.float32], window_length: int=5, poly_order: int=2, deriv_order: int=1, axis: int=-1) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))
    if not poly_order < window_length:
        raise ValueError(f'{poly_order} < {window_length} does not hold!')
    dx = np.diff(x)
    if not (dx > 0).all():
        raise RuntimeError('dx is not monotonically increasing!')
    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(y, polyorder=poly_order, window_length=window_length, deriv=deriv_order, delta=dx, axis=axis)
    return derivative

class TestMetricFileCallback(TestCase):
    """Tests metrics files generation at the end fo the simulation."""

    def setUp(self) -> None:
        """Setup mocks for the tests"""
        self.mock_metric_file_callback = Mock(spec=MetricFileCallback)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp_dir.name)
        self.path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up tmp dir."""
        self.tmp_dir.cleanup()

    def test_metric_callback_init(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        metric_file_callback = MetricFileCallback(metric_file_output_path=self.tmp_dir.name, scenario_metric_paths=[self.tmp_dir.name])
        self.assertEqual(metric_file_callback._metric_file_output_path, self.path)
        self.assertEqual(metric_file_callback._scenario_metric_paths, [self.path])

    @patch('nuplan.planning.simulation.main_callback.metric_file_callback.logger')
    def test_on_run_simulation_end(self, logger: MagicMock) -> None:
        """
        Tests if the callback is called with the correct parameters.
        """
        metric_file_callback = MetricFileCallback(metric_file_output_path=self.tmp_dir.name, scenario_metric_paths=[self.tmp_dir.name])
        metric_file_callback.on_run_simulation_end()
        logger.info.assert_has_calls([call('Metric files integration: 00:00:00 [HH:MM:SS]')])

class TestMetricSummaryCallback(unittest.TestCase):
    """Test metric_summary callback functionality."""

    def set_up_dummy_metric(self, metric_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy metric results.
        :param metric_path: Metric path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        statistics = [Statistic(name='ego_max_acceleration', unit='meters_per_second_squared', value=2.0, type=MetricStatisticsType.MAX), Statistic(name='ego_min_acceleration', unit='meters_per_second_squared', value=0.0, type=MetricStatisticsType.MIN), Statistic(name='ego_p90_acceleration', unit='meters_per_second_squared', value=1.0, type=MetricStatisticsType.P90)]
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit='meters_per_second_squared', time_stamps=list(time_stamps), values=list(accel))
        result = MetricStatistics(metric_computator='ego_acceleration', name='ego_acceleration_statistics', statistics=statistics, time_series=time_series, metric_category='Dynamic', metric_score=1)
        key = MetricFileKey(metric_name='ego_acceleration', scenario_name=scenario_name, log_name=log_name, scenario_type=scenario_type, planner_name=planner_name)
        metric_engine = MetricsEngine(main_save_path=metric_path)
        metric_files = {'ego_acceleration': [MetricFile(key=key, metric_statistics=[result])]}
        metric_engine.write_to_files(metric_files=metric_files)
        metric_file_callback = MetricFileCallback(metric_file_output_path=str(metric_path), scenario_metric_paths=[str(metric_path)], delete_scenario_metric_files=True)
        metric_file_callback.on_run_simulation_end()

    def setUp(self) -> None:
        """Set up a nuboard base tab."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        log_name = 'dummy_log'
        planner_name = 'SimplePlanner'
        scenario_type = 'Test'
        scenario_name = 'Dummy_scene'
        metric_path = Path(self.tmp_dir.name) / 'metrics'
        metric_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_metric(metric_path=metric_path, log_name=log_name, planner_name=planner_name, scenario_name=scenario_name, scenario_type=scenario_type)
        self.aggregator_save_path = Path(self.tmp_dir.name) / 'aggregator_metric'
        self.weighted_average_metric_aggregator = WeightedAverageMetricAggregator(name='weighted_average_metric_aggregator', metric_weights={'default': 1.0, 'dummy_metric': 0.5}, file_name='test_weighted_average_metric_aggregator.parquet', aggregator_save_path=self.aggregator_save_path, multiple_metrics=[])
        self.metric_statistics_dataframes = {}
        for metric_parquet_file in metric_path.iterdir():
            print(metric_parquet_file)
            data_frame = MetricStatisticsDataFrame.load_parquet(metric_parquet_file)
            self.metric_statistics_dataframes[data_frame.metric_statistic_name] = data_frame
        self.metric_summary_output_path = Path(self.tmp_dir.name) / 'summary'
        self.metric_summary_callback = MetricSummaryCallback(metric_save_path=str(metric_path), metric_aggregator_save_path=str(self.aggregator_save_path), summary_output_path=str(self.metric_summary_output_path), pdf_file_name='summary.pdf')

    def test_metric_summary_callback_on_simulation_end(self) -> None:
        """Test on_simulation_end in metric summary callback."""
        self.weighted_average_metric_aggregator(metric_dataframes=self.metric_statistics_dataframes)
        self.metric_summary_callback.on_run_simulation_end()
        pdf_files = self.metric_summary_output_path.rglob('*.pdf')
        self.assertEqual(len(list(pdf_files)), 1)

    def tearDown(self) -> None:
        """Remove all temporary folders and files."""
        self.tmp_dir.cleanup()

def is_agent_ahead(ego_state: StateSE2, agent_state: StateSE2, angle_tolerance: float=30) -> bool:
    """
    Determines if an agent is ahead of the ego
    :param ego_state: ego's pose
    :param agent_state: agent's pose
    :param angle_tolerance: tolerance to consider if agent is ahead, where zero is the heading of the ego [deg]
    :return: true if agent is ahead, false otherwise.
    """
    return bool(get_agent_relative_angle(ego_state, agent_state) < np.deg2rad(angle_tolerance))

@lru_cache(maxsize=256)
def get_agent_relative_angle(ego_state: StateSE2, agent_state: StateSE2) -> float:
    """
    Get the the relative angle of an agent position to the ego
    :param ego_state: pose of ego
    :param agent_state: pose of an agent
    :return: relative angle in radians.
    """
    agent_vector: npt.NDArray[np.float32] = np.array([agent_state.x - ego_state.x, agent_state.y - ego_state.y])
    ego_vector: npt.NDArray[np.float32] = np.array([np.cos(ego_state.heading), np.sin(ego_state.heading)])
    dot_product = np.dot(ego_vector, agent_vector / np.linalg.norm(agent_vector))
    return float(np.arccos(dot_product))

def is_agent_behind(ego_state: StateSE2, agent_state: StateSE2, angle_tolerance: float=150) -> bool:
    """
    Determines if an agent is behind of the ego
    :param ego_state: ego's pose
    :param agent_state: agent's pose
    :param angle_tolerance: tolerance to consider if agent is behind, where zero is the heading of the ego [deg]
    :return: true if agent is behind, false otherwise
    """
    return bool(get_agent_relative_angle(ego_state, agent_state) > np.deg2rad(angle_tolerance))

def compute_yaw_rate_from_states(agent_states_horizon: List[List[StateSE2]], time_stamps: List[TimePoint]) -> npt.NDArray[np.float32]:
    """
    Computes the yaw rate of all agents over the trajectory from heading
    :param agent_states_horizon: agent trajectories [num_frames, num_agents, 1]
           where each state is represented by StateSE2
    :param time_stamps: the time stamps of each frame
    :return: <np.ndarray: num_frames, num_agents, 1> where last dimension is the yaw rate
    """
    yaw: npt.NDArray[np.float32] = np.array([[agent.heading for agent in frame] for frame in agent_states_horizon], dtype=np.float32)
    yaw_rate_horizon = approximate_derivatives(yaw.transpose(), np.array([stamp.time_s for stamp in time_stamps]), window_length=3)
    return cast(npt.NDArray[np.float32], yaw_rate_horizon)

class TestBaseTab(unittest.TestCase):
    """Test base_tab functionality."""

    def set_up_dummy_simulation(self, simulation_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy simulation data.
        :param simulation_path: Simulation path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        save_path = simulation_path / planner_name / scenario_type / log_name / scenario_name
        save_path.mkdir(parents=True, exist_ok=True)
        simulation_data = create_sample_simulation_log(save_path / 'test_base_tab_simulation_log.msgpack.xz')
        simulation_data.save_to_file()

    def set_up_dummy_metric(self, metric_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy metric results.
        :param metric_path: Metric path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        statistics = [Statistic(name='ego_max_acceleration', unit='meters_per_second_squared', value=2.0, type=MetricStatisticsType.MAX), Statistic(name='ego_min_acceleration', unit='meters_per_second_squared', value=0.0, type=MetricStatisticsType.MIN), Statistic(name='ego_p90_acceleration', unit='meters_per_second_squared', value=1.0, type=MetricStatisticsType.P90)]
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit='meters_per_second_squared', time_stamps=list(time_stamps), values=list(accel))
        result = MetricStatistics(metric_computator='ego_acceleration', name='ego_acceleration_statistics', statistics=statistics, time_series=time_series, metric_category='Dynamic', metric_score=1)
        key = MetricFileKey(metric_name='ego_acceleration', scenario_name=scenario_name, log_name=log_name, scenario_type=scenario_type, planner_name=planner_name)
        metric_engine = MetricsEngine(main_save_path=metric_path)
        metric_files = {'ego_acceleration': [MetricFile(key=key, metric_statistics=[result])]}
        metric_engine.write_to_files(metric_files=metric_files)
        metric_file_callback = MetricFileCallback(metric_file_output_path=str(metric_path), scenario_metric_paths=[str(metric_path)])
        metric_file_callback.on_run_simulation_end()

    def setUp(self) -> None:
        """Set up a nuboard base tab."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', aggregator_metric_folder='aggregator_metric', current_path=Path(self.tmp_dir.name))
        doc = Document()
        log_name = 'dummy_log'
        planner_name = 'SimplePlanner'
        scenario_type = 'Test'
        scenario_name = 'Dummy_scene'
        metric_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_metric(metric_path=metric_path, log_name=log_name, planner_name=planner_name, scenario_name=scenario_name, scenario_type=scenario_type)
        simulation_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_simulation(simulation_path, log_name=log_name, planner_name=planner_name, scenario_type=scenario_type, scenario_name=scenario_name)
        color_palettes = Category20[20] + Set3[12] + Bokeh[8]
        experiment_file_data = ExperimentFileData(file_paths=[], color_palettes=color_palettes)
        self.base_tab = BaseTab(doc=doc, experiment_file_data=experiment_file_data)

    def test_update_experiment_file_data(self) -> None:
        """Test update experiment file data."""
        self.base_tab.experiment_file_data.update_data(file_paths=[self.nuboard_file])
        self.assertEqual(len(self.base_tab.experiment_file_data.available_metric_statistics_names), 1)
        self.assertEqual(len(self.base_tab.experiment_file_data.simulation_scenario_keys), 1)

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change feature."""
        self.base_tab.experiment_file_data.update_data(file_paths=[self.nuboard_file])
        self.assertRaises(NotImplementedError, self.base_tab.file_paths_on_change, self.base_tab.experiment_file_data, [0])

    def tearDown(self) -> None:
        """Remove all temporary folders and files."""
        self.tmp_dir.cleanup()

class SkeletonTestTab(unittest.TestCase):
    """Base class for nuBoard tab unit tests."""

    @staticmethod
    def set_up_dummy_simulation(simulation_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy simulation data.
        :param simulation_path: Simulation path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        save_path = simulation_path / planner_name / scenario_type / log_name / scenario_name
        save_path.mkdir(parents=True, exist_ok=True)
        simulation_data = create_sample_simulation_log(save_path / f'{uuid4()}.msgpack.xz')
        simulation_data.save_to_file()

    @staticmethod
    def set_up_dummy_metric(metric_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy metric results.
        :param metric_path: Metric path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        statistics = [Statistic(name='ego_max_acceleration', unit='meters_per_second_squared', value=2.0, type=MetricStatisticsType.MAX), Statistic(name='ego_min_acceleration', unit='meters_per_second_squared', value=0.0, type=MetricStatisticsType.MIN), Statistic(name='ego_p90_acceleration', unit='meters_per_second_squared', value=1.0, type=MetricStatisticsType.P90), Statistic(name='ego_count_acceleration', unit=MetricStatisticsType.COUNT.unit, value=2, type=MetricStatisticsType.COUNT), Statistic(name='ego_boolean_acceleration', unit=MetricStatisticsType.BOOLEAN.unit, value=True, type=MetricStatisticsType.BOOLEAN)]
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit='meters_per_second_squared', time_stamps=list(time_stamps), values=list(accel))
        result = MetricStatistics(metric_computator='ego_acceleration', name='ego_acceleration_statistics', statistics=statistics, time_series=time_series, metric_category='Dynamic', metric_score=1.0)
        key = MetricFileKey(metric_name='ego_acceleration', log_name=log_name, scenario_name=scenario_name, scenario_type=scenario_type, planner_name=planner_name)
        metric_engine = MetricsEngine(main_save_path=metric_path)
        metric_files = {scenario_name: [MetricFile(key=key, metric_statistics=[result])]}
        metric_engine.write_to_files(metric_files=metric_files)
        metric_file_callback = MetricFileCallback(metric_file_output_path=str(metric_path), scenario_metric_paths=[str(metric_path)])
        metric_file_callback.on_run_simulation_end()

    def setUp(self) -> None:
        """
        Set up common data for nuboard unit tests.
        """
        self.doc = Document()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', aggregator_metric_folder='aggregator_metric', current_path=Path(self.tmp_dir.name))
        log_name = 'dummy_log'
        planner_name = 'SimplePlanner'
        scenario_type = 'Test'
        scenario_name = 'Dummy_scene'
        metric_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_metric(metric_path=metric_path, log_name=log_name, planner_name=planner_name, scenario_name=scenario_name, scenario_type=scenario_type)
        simulation_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_simulation(simulation_path, log_name=log_name, planner_name=planner_name, scenario_type=scenario_type, scenario_name=scenario_name)
        self.nuboard_file_name = Path(self.tmp_dir.name) / ('nuboard_file' + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.experiment_file_data = ExperimentFileData(file_paths=[self.nuboard_file])

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()

class MetricsEngine:
    """The metrics engine aggregates and manages the instantiated metrics for a scenario."""

    def __init__(self, main_save_path: Path, metrics: Optional[List[AbstractMetricBuilder]]=None) -> None:
        """
        Initializer for MetricsEngine class
        :param metrics: Metric objects.
        """
        self._main_save_path = main_save_path
        if not is_s3_path(self._main_save_path):
            self._main_save_path.mkdir(parents=True, exist_ok=True)
        if metrics is None:
            self._metrics: List[AbstractMetricBuilder] = []
        else:
            self._metrics = metrics

    @property
    def metrics(self) -> List[AbstractMetricBuilder]:
        """Retrieve a list of metric results."""
        return self._metrics

    def add_metric(self, metric_builder: AbstractMetricBuilder) -> None:
        """TODO: Create the list of types needed from the history"""
        self._metrics.append(metric_builder)

    def write_to_files(self, metric_files: Dict[str, List[MetricFile]]) -> None:
        """
        Write to a file by constructing a dataframe
        :param metric_files: A dictionary of scenario names and a list of their metric files.
        """
        for scenario_name, metric_files in metric_files.items():
            file_name = scenario_name + JSON_FILE_EXTENSION
            save_path = self._main_save_path / file_name
            dataframes = []
            for metric_file in metric_files:
                metric_file_key = metric_file.key
                for metric_statistic in metric_file.metric_statistics:
                    dataframe = construct_dataframe(log_name=metric_file_key.log_name, scenario_name=metric_file_key.scenario_name, scenario_type=metric_file_key.scenario_type, planner_name=metric_file_key.planner_name, metric_statistics=metric_statistic)
                    dataframes.append(dataframe)
            if len(dataframes):
                save_object_as_pickle(save_path, dataframes)

    def compute_metric_results(self, history: SimulationHistory, scenario: AbstractScenario) -> Dict[str, List[MetricStatistics]]:
        """
        Compute metrics in the engine
        :param history: History from simulation
        :param scenario: Scenario running this metric engine
        :return A list of metric statistics.
        """
        metric_results = {}
        for metric in self._metrics:
            try:
                start_time = time.perf_counter()
                metric_results[metric.name] = metric.compute(history, scenario=scenario)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.debug(f'Metric: {metric.name} running time: {elapsed_time:.2f} seconds.')
            except (NotImplementedError, Exception) as e:
                logger.error(f'Running {metric.name} with error: {e}')
                raise RuntimeError(f'Metric Engine failed with: {e}')
        return metric_results

    def compute(self, history: SimulationHistory, scenario: AbstractScenario, planner_name: str) -> Dict[str, List[MetricFile]]:
        """
        Compute metrics and return in a format of MetricStorageResult for each metric computation
        :param history: History from simulation
        :param scenario: Scenario running this metric engine
        :param planner_name: name of the planner
        :return A dictionary of scenario name and list of MetricStorageResult.
        """
        all_metrics_results = self.compute_metric_results(history=history, scenario=scenario)
        metric_files = defaultdict(list)
        for metric_name, metric_statistics_results in all_metrics_results.items():
            metric_file_key = MetricFileKey(metric_name=metric_name, log_name=scenario.log_name, scenario_name=scenario.scenario_name, scenario_type=scenario.scenario_type, planner_name=planner_name)
            metric_file = MetricFile(key=metric_file_key, metric_statistics=metric_statistics_results)
            metric_file_name = scenario.scenario_type + '_' + scenario.scenario_name + '_' + planner_name
            metric_files[metric_file_name].append(metric_file)
        return metric_files

def compute_traj_heading_errors(ego_traj: List[StateSE2], expert_traj: List[StateSE2]) -> npt.NDArray:
    """
    Compute the heading (yaw) errors between the ego trajectory and expert trajectory
    :param ego_traj: a list of StateSE2 that describe ego position with yaw
    :param expert_traj: a list of StateSE2 that describe expert position with yaw
    :return An array of yaw errors.
    """
    yaw_displacements: npt.NDArray[np.float32] = np.array([ego_traj[i].heading - expert_traj[i].heading for i in range(len(ego_traj))])
    heading_errors = np.abs(principal_value(yaw_displacements))
    return heading_errors

def compute_traj_errors(ego_traj: Union[List[Point2D], List[StateSE2]], expert_traj: Union[List[Point2D], List[StateSE2]], discount_factor: float=1.0, heading_diff_weight: float=1.0) -> npt.NDArray:
    """
    Compute the errors between the position/position_with_yaw of ego trajectory and expert trajectory
    :param ego_traj: a list of Point2D or StateSE2 that describe ego position/position with yaw
    :param expert_traj: a list of Point2D or StateSE2 that describe expert position/position with yaw
    :param discount_factor: Displacements corresponding to the k^th timestep will
    be discounted by a factor of discount_factor^k., defaults to 1.0
    :param heading_diff_weight: factor to weight heading differences if yaw errors are also
    considered, defaults to 1.0
    :return an array of displacement errors.
    """
    traj_len = len(ego_traj)
    expert_traj_len = len(expert_traj)
    assert traj_len != 0, 'ego_traj should be a nonempty list'
    assert traj_len == expert_traj_len or traj_len == expert_traj_len - 1, 'ego and expert have different trajectory lengths'
    displacements = np.zeros((traj_len, 2))
    for i in range(traj_len):
        displacements[i, :] = [ego_traj[i].x - expert_traj[i].x, ego_traj[i].y - expert_traj[i].y]
    dist_seq = np.hypot(displacements[:, 0], displacements[:, 1])
    if isinstance(ego_traj[0], StateSE2) and isinstance(expert_traj[0], StateSE2) and (heading_diff_weight != 0):
        heading_errors = compute_traj_heading_errors(ego_traj, expert_traj)
        weighted_heading_errors = heading_errors * heading_diff_weight
        dist_seq = dist_seq + weighted_heading_errors
    if discount_factor != 1:
        discount_weights = get_discount_weights(discount_factor=discount_factor, traj_len=traj_len)
        dist_seq = np.multiply(dist_seq, discount_weights)
    return dist_seq

def get_route(map_api: AbstractMap, poses: List[Point2D]) -> List[List[GraphEdgeMapObject]]:
    """
    Returns and sets the sequence of lane and lane connectors corresponding to the trajectory
    :param map_api: map
    :param poses: a list of xy coordinates
    :return list of route objects.
    """
    if not len(poses):
        raise ValueError('invalid poses passed to get_route()')
    route_objs: List[List[GraphEdgeMapObject]] = []
    curr_route_obj: List[GraphEdgeMapObject] = []
    for ind, pose in enumerate(poses):
        if curr_route_obj:
            curr_route_obj = get_route_obj_with_candidates(pose, curr_route_obj)
        if not curr_route_obj:
            curr_route_obj = get_current_route_objects(map_api, pose)
            if ind > 1 and route_objs[-1] and isinstance(route_objs[-1][0], LaneConnector) and (curr_route_obj and isinstance(curr_route_obj[0], LaneConnector) or (not curr_route_obj and map_api.is_in_layer(pose, SemanticMapLayer.INTERSECTION))):
                previous_proximal_route_obj = [obj for obj in route_objs[-1] if obj.polygon.distance(Point(*pose)) < 5]
                if previous_proximal_route_obj:
                    curr_route_obj = previous_proximal_route_obj
        route_objs.append(curr_route_obj)
    improved_route_obj = remove_extra_lane_connectors(route_objs)
    return improved_route_obj

def get_route_obj_with_candidates(pose: Point2D, candidate_route_objs: List[GraphEdgeMapObject]) -> List[GraphEdgeMapObject]:
    """
    This function uses a candidate set of lane/lane-connectors and return the lane/lane-connector that correponds to the pose
    by checking if pose belongs to one of the route objs in candidate_route_objs or their outgoing_edges
    :param pose: ego_pose
    :param candidate_route_objs: a list of route objects
    :return: a list of route objects corresponding to the pose
    """
    if not len(candidate_route_objs):
        raise ValueError('candidate_route_objs list is empty, no candidates to start with')
    route_objects_with_pose = [one_route_obj for one_route_obj in candidate_route_objs if one_route_obj.contains_point(pose)]
    if not route_objects_with_pose and len(candidate_route_objs) == 1:
        route_objects_with_pose = [next_route_obj for next_route_obj in candidate_route_objs[0].outgoing_edges if next_route_obj.contains_point(pose)]
    return route_objects_with_pose

def get_current_route_objects(map_api: AbstractMap, pose: Point2D) -> List[GraphEdgeMapObject]:
    """
    Gets the list including the lane or lane_connectors the pose corresponds to if there exists one, and empty list o.w
    :param map_api: map
    :param pose: xy coordinates
    :return the corresponding route object.
    """
    curr_lane = map_api.get_one_map_object(pose, SemanticMapLayer.LANE)
    if curr_lane is None:
        curr_lane_connectors = map_api.get_all_map_objects(pose, SemanticMapLayer.LANE_CONNECTOR)
        route_objects_with_pose = curr_lane_connectors
    else:
        route_objects_with_pose = [curr_lane]
    return route_objects_with_pose

def remove_extra_lane_connectors(route_objs: List[List[GraphEdgeMapObject]]) -> List[List[GraphEdgeMapObject]]:
    """
    # This function iterate through route object and replace field with multiple lane_connectors
    # with the one lane_connector ego ends up in.
    :param route_objs: a list of route objects.
    """
    last_to_first_route_list = route_objs[::-1]
    enum = enumerate(last_to_first_route_list)
    for ind, curr_last_obj in enum:
        if ind == 0 or len(curr_last_obj) <= 1:
            continue
        if len(curr_last_obj) > len(last_to_first_route_list[ind - 1]):
            curr_route_obj_ids = [obj.id for obj in curr_last_obj]
            if all([obj.id in curr_route_obj_ids for obj in last_to_first_route_list[ind - 1]]):
                last_to_first_route_list[ind] = last_to_first_route_list[ind - 1]
        if len(curr_last_obj) <= 1:
            continue
        if last_to_first_route_list[ind - 1] and isinstance(last_to_first_route_list[ind - 1][0], Lane):
            next_lane_incoming_edge_ids = [obj.id for obj in last_to_first_route_list[ind - 1][0].incoming_edges]
            objs_to_keep = [obj for obj in curr_last_obj if obj.id in next_lane_incoming_edge_ids]
            if objs_to_keep:
                last_to_first_route_list[ind] = objs_to_keep
    return last_to_first_route_list[::-1]

def extract_corners_route(map_api: AbstractMap, ego_footprint_list: List[OrientedBox]) -> List[CornersGraphEdgeMapObject]:
    """
    Extracts lists of lane/lane connectors of corners of ego from history
    :param map_api: AbstractMap
    :param ego_corners_list: List of OrientedBoxes
    :return List of CornersGraphEdgeMapObject class containing list of lane/lane connectors of each
    corner of ego in the history.
    """
    if not len(ego_footprint_list):
        logger.warning('Invalid poses passed to extract_corners_route()')
        return []
    corners_route: List[CornersGraphEdgeMapObject] = []
    curr_candid_route_obj: List[GraphEdgeMapObject] = []
    for ind, ego_footprint in enumerate(ego_footprint_list):
        corners_route_objs = CornersGraphEdgeMapObject([], [], [], [])
        ego_corners = ego_footprint.all_corners()
        next_candid_route_obj: List[GraphEdgeMapObject] = []
        for ego_corner, corner_type in zip(ego_corners, corners_route_objs.__dict__.keys()):
            route_object = []
            if curr_candid_route_obj:
                route_object = get_route_obj_with_candidates(ego_corner, curr_candid_route_obj)
            if not route_object:
                route_object = get_current_route_objects(map_api, ego_corner)
                if ind == 1:
                    curr_candid_route_obj += [obj for obj in route_object if obj.id not in [candid.id for candid in curr_candid_route_obj]]
            next_candid_route_obj += [obj for obj in route_object if obj.id not in [candid.id for candid in next_candid_route_obj]]
            corners_route_objs.__setattr__(corner_type, route_object)
        corners_route.append(corners_route_objs)
        curr_candid_route_obj = next_candid_route_obj
    return corners_route

def get_connecting_route_object(corners_route_obj_list: List[List[GraphEdgeMapObject]], corners_route_obj_ids: List[Set[str]], obj_id_dict: dict[str, GraphEdgeMapObject]) -> Set[GraphEdgeMapObject]:
    """
    Extracts connecting (outgoing or incoming) lane/lane connectors of corners
    :param corners_route_obj_list: List of route objects of corners of ego
    :param corners_route_obj_ids: List of ids of route objects of corners of ego
    :param obj_id_dict: dictionary of ids and corresponding route objects
    :return set of connecting route objects, returns an empty set of no connecting object is found.
    """
    all_corners_connecting_obj_ids = set()
    front_left_route_obj, rear_left_route_obj, rear_right_route_obj, front_right_route_obj = corners_route_obj_list
    front_left_route_obj_ids, rear_left_route_obj_ids, rear_right_route_obj_ids, front_right_route_obj_ids = corners_route_obj_ids
    rear_right_route_obj_out_edge_dict = get_outgoing_edges_obj_dict(rear_right_route_obj)
    rear_left_route_obj_out_edge_dict = get_outgoing_edges_obj_dict(rear_left_route_obj)
    obj_id_dict = {**obj_id_dict, **rear_right_route_obj_out_edge_dict, **rear_left_route_obj_out_edge_dict}
    rear_right_obj_or_outgoing_edge = rear_right_route_obj_ids.union(set(rear_right_route_obj_out_edge_dict.keys()))
    rear_left_in_rear_right_obj_or_outgoing_edge = rear_left_route_obj_ids.intersection(rear_right_obj_or_outgoing_edge)
    rear_left_obj_or_outgoing_edge = rear_left_route_obj_ids.union(set(rear_left_route_obj_out_edge_dict.keys()))
    rear_right_in_rear_left_obj_or_outgoing_edge = rear_right_route_obj_ids.intersection(rear_left_obj_or_outgoing_edge)
    rear_corners_connecting_obj_ids = rear_left_in_rear_right_obj_or_outgoing_edge.union(rear_right_in_rear_left_obj_or_outgoing_edge)
    if len(rear_corners_connecting_obj_ids) > 0:
        front_left_route_obj_in_edge_dict = get_incoming_edges_obj_dict(front_left_route_obj)
        front_left_obj_or_incoming_edge = front_left_route_obj_ids.union(set(front_left_route_obj_in_edge_dict.keys()))
        front_left_rear_right_common_obj_ids = front_left_obj_or_incoming_edge.intersection(rear_right_obj_or_outgoing_edge)
        front_right_route_obj_in_edge_dict = get_incoming_edges_obj_dict(front_right_route_obj)
        front_right_obj_or_incoming_edge = front_right_route_obj_ids.union(set(front_right_route_obj_in_edge_dict.keys()))
        front_right_rear_left_common_obj_ids = front_right_obj_or_incoming_edge.intersection(rear_left_obj_or_outgoing_edge)
        all_corners_connecting_obj_ids = {obj_id_dict[id] for id in set.intersection(front_left_rear_right_common_obj_ids, front_right_rear_left_common_obj_ids)}
    return all_corners_connecting_obj_ids

def get_outgoing_edges_obj_dict(corner_route_object: List[GraphEdgeMapObject]) -> dict[str, GraphEdgeMapObject]:
    """
    :param corner_route_object: List of lane/lane connectors
    :return dictionary of id and itscorresponding route object of outgoing edges of a given route object
    """
    return {obj_edge.id: obj_edge for obj in corner_route_object for obj_edge in obj.outgoing_edges}

def get_incoming_edges_obj_dict(corner_route_object: List[GraphEdgeMapObject]) -> dict[str, GraphEdgeMapObject]:
    """
    :param corner_route_object: List of lane/lane connectors
    :return dictionary of id and itscorresponding route object of incoming edges of a given route object
    """
    return {obj_edge.id: obj_edge for obj in corner_route_object for obj_edge in obj.incoming_edges}

def extract_common_or_connecting_route_objs(corners_route_obj: CornersGraphEdgeMapObject) -> Optional[Set[GraphEdgeMapObject]]:
    """
    Extracts common or connecting (outgoing or incoming) lane/lane connectors of corners
    :param corners_route_obj: Class containing list of lane/lane connectors of each corner
    :return common or connecting lane/lane connectors of corners if exists, else None.
    If all corners are in nondrivable area, returns an empty set.
    """
    corners_route_obj_list = [*corners_route_obj.__iter__()]
    not_in_lane_or_laneconn = [True if len(corner_route_obj) == 0 else False for corner_route_obj in corners_route_obj_list]
    if np.all(not_in_lane_or_laneconn):
        return set()
    if np.any(not_in_lane_or_laneconn):
        return None
    obj_id_dict = {obj.id: obj for corner_route_obj in corners_route_obj_list for obj in corner_route_obj}
    corners_route_obj_ids = [{obj.id for obj in corner_route_obj} for corner_route_obj in corners_route_obj_list]
    all_corners_common_obj = get_common_route_object(corners_route_obj_ids, obj_id_dict)
    if len(all_corners_common_obj) > 0:
        return all_corners_common_obj
    all_corners_connecting_obj = get_connecting_route_object(corners_route_obj_list, corners_route_obj_ids, obj_id_dict)
    if len(all_corners_connecting_obj) > 0:
        return all_corners_connecting_obj
    return None

def get_common_route_object(corners_route_obj_ids: List[Set[str]], obj_id_dict: dict[str, GraphEdgeMapObject]) -> Set[GraphEdgeMapObject]:
    """
    Extracts common lane/lane connectors of corners
    :param corners_route_obj_ids: List of ids of route objects of corners of ego
    :param obj_id_dict: dictionary of ids and corresponding route objects
    :return set of common route objects, returns an empty set of no common object is found.
    """
    return {obj_id_dict[id] for id in set.intersection(*corners_route_obj_ids)}

def get_common_or_connected_route_objs_of_corners(corners_route: List[CornersGraphEdgeMapObject]) -> List[Optional[Set[GraphEdgeMapObject]]]:
    """
    Returns a list of common or connected lane/lane connectors of corners.
    :param corners_route: List of class conatining list of lane/lane connectors of corners of ego
    :return list of common or connected lane/lane connectors of corners if exist, empty list if all corners are
    in non_drivable area and None if corners are in different lane/lane connectors.
    """
    history_common_or_connecting_route_objs: List[Optional[Set[GraphEdgeMapObject]]] = []
    prev_corners_route_obj = corners_route[0]
    corners_common_or_connecting_route_objs = extract_common_or_connecting_route_objs(prev_corners_route_obj)
    history_common_or_connecting_route_objs.append(corners_common_or_connecting_route_objs)
    for curr_corners_route_obj in corners_route[1:]:
        if curr_corners_route_obj != prev_corners_route_obj:
            corners_common_or_connecting_route_objs = extract_common_or_connecting_route_objs(curr_corners_route_obj)
        history_common_or_connecting_route_objs.append(corners_common_or_connecting_route_objs)
        prev_corners_route_obj = curr_corners_route_obj
    return history_common_or_connecting_route_objs

def get_fault_type_statistics(all_at_fault_collisions: Dict[TrackedObjectType, List[float]]) -> List[Statistic]:
    """
    :param all_at_fault_collisions: Dict of at_fault collisions.
    :return: List of Statistics for all collision track types.
    """
    statistics = []
    track_types_collisions_energy_dict: Dict[str, List[float]] = {}
    for collision_track_type, collision_name in zip([VRU_types, [TrackedObjectType.VEHICLE], object_types], ['VRUs', 'vehicles', 'objects']):
        track_types_collisions_energy_dict[collision_name] = [colision_energy for track_type in collision_track_type for colision_energy in all_at_fault_collisions[track_type]]
        statistics.extend([Statistic(name=f'number_of_at_fault_collisions_with_{collision_name}', unit=MetricStatisticsType.COUNT.unit, value=len(track_types_collisions_energy_dict[collision_name]), type=MetricStatisticsType.COUNT)])
    for collision_name, track_types_collisions_energy in track_types_collisions_energy_dict.items():
        if len(track_types_collisions_energy) > 0:
            statistics.extend([Statistic(name=f'max_collision_energy_with_{collision_name}', unit='meters_per_second', value=max(track_types_collisions_energy), type=MetricStatisticsType.MAX), Statistic(name=f'min_collision_energy_with_{collision_name}', unit='meters_per_second', value=min(track_types_collisions_energy), type=MetricStatisticsType.MIN), Statistic(name=f'mean_collision_energy_with_{collision_name}', unit='meters_per_second', value=np.mean(track_types_collisions_energy), type=MetricStatisticsType.MEAN)])
    return statistics

def extract_ego_jerk(ego_states: List[EgoState], acceleration_coordinate: str, decimals: int=8, deriv_order: int=1, poly_order: int=2, window_length: int=15) -> npt.NDArray[np.float32]:
    """
    Extract jerk of ego pose in simulation history
    :param ego_states: A list of ego states
    :param acceleration_coordinate: x, y or 'magnitude' in acceleration
    :param decimals: Decimal precision
    :return An array of valid ego pose jerk and timestamps.
    """
    time_points = extract_ego_time_point(ego_states)
    ego_acceleration = extract_ego_acceleration(ego_states=ego_states, acceleration_coordinate=acceleration_coordinate)
    jerk = approximate_derivatives(ego_acceleration, time_points / 1000000.0, deriv_order=deriv_order, poly_order=poly_order, window_length=min(window_length, len(ego_acceleration)))
    jerk = np.round(jerk, decimals=decimals)
    return jerk

def extract_ego_time_point(ego_states: List[EgoState]) -> npt.NDArray[np.int32]:
    """
    Extract time point in simulation history
    :param ego_states: A list of ego stets
    :return An array of time in micro seconds.
    """
    time_point: npt.NDArray[np.int32] = np.array([ego_state.time_point.time_us for ego_state in ego_states])
    return time_point

def extract_ego_acceleration(ego_states: List[EgoState], acceleration_coordinate: str, decimals: int=8, poly_order: int=2, window_length: int=8) -> npt.NDArray[np.float32]:
    """
    Extract acceleration of ego pose in simulation history
    :param ego_states: A list of ego states
    :param acceleration_coordinate: 'x', 'y', or 'magnitude'
    :param decimals: Decimal precision
    :return An array of ego pose acceleration.
    """
    if acceleration_coordinate == 'x':
        acceleration: npt.NDArray[np.float32] = np.asarray([ego_state.dynamic_car_state.center_acceleration_2d.x for ego_state in ego_states])
    elif acceleration_coordinate == 'y':
        acceleration = np.asarray([ego_state.dynamic_car_state.center_acceleration_2d.y for ego_state in ego_states])
    elif acceleration_coordinate == 'magnitude':
        acceleration = np.array([ego_state.dynamic_car_state.acceleration for ego_state in ego_states])
    else:
        raise ValueError(f'acceleration_coordinate option: {acceleration_coordinate} not available. Available options are: x, y or magnitude')
    acceleration = savgol_filter(acceleration, polyorder=poly_order, window_length=min(window_length, len(acceleration)))
    acceleration = np.round(acceleration, decimals=decimals)
    return acceleration

def extract_ego_yaw_rate(ego_states: List[EgoState], deriv_order: int=1, poly_order: int=2, decimals: int=8, window_length: int=15) -> npt.NDArray[np.float32]:
    """
    Extract ego rates
    :param ego_states: A list of ego states
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param decimals: Decimal precision
    :return An array of ego yaw rates.
    """
    ego_headings = extract_ego_heading(ego_states)
    ego_timestamps = extract_ego_time_point(ego_states)
    ego_yaw_rate = approximate_derivatives(phase_unwrap(ego_headings), ego_timestamps / 1000000.0, deriv_order=deriv_order, poly_order=poly_order)
    ego_yaw_rate = np.round(ego_yaw_rate, decimals=decimals)
    return ego_yaw_rate

def extract_ego_heading(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Extract yaw headings of ego pose in simulation history
    :param ego_states: A list of ego states
    :return An array of ego pose yaw heading.
    """
    heading: npt.NDArray[np.float32] = np.array([ego_state.rear_axle.heading for ego_state in ego_states])
    return heading

def get_route_simplified(route_list: List[List[LaneGraphEdgeMapObject]]) -> List[List[LaneGraphEdgeMapObject]]:
    """
    This function simplifies the route by removing repeated consequtive route objects
    :param route_list: A list of route objects representing ego's corresponding route objects at each instance
    :return A simplified list of route objects that shows the order of route objects ego has been in.
    """
    try:
        ind = next((iteration for iteration, iter_route_list in enumerate(route_list) if iter_route_list))
    except StopIteration:
        logger.warning('All route_list elements are empty, returning an empty list from get_route_simplified()')
        return []
    route_simplified = [route_list[ind]]
    for route_object in route_list[ind + 1:]:
        repeated_entries = [obj_id for obj_id in [prev_obj.id for prev_obj in route_simplified[-1]] if obj_id in [one_route_obj.id for one_route_obj in route_object]]
        if route_object and (not repeated_entries):
            route_simplified.append(route_object)
    return route_simplified

@nuplan_test(path='json/route_extractor/route_extractor.json')
def test_corners_route_extraction(scene: Dict[str, Any]) -> None:
    """
    Test getting ego's corners route objects.
    """
    map_api = map_factory.build_map_from_name(scene['map']['area'])
    vehicle_parameters = get_pacifica_parameters()
    expert_footprints = []
    for marker in scene['markers']:
        expert_footprints.append(CarFootprint.build_from_center(StateSE2(*marker['pose'][:3]), vehicle_parameters))
    corners_route = extract_corners_route(map_api=map_api, ego_footprint_list=expert_footprints)
    assert len(corners_route) == len(expert_footprints)
    all_route_obj = [map_object for corners_objects in corners_route for corner in corners_objects.__dict__.values() for map_object in corner]
    unique_route_obj_ids = {obj.id for obj in all_route_obj}
    assert len(unique_route_obj_ids) == 4

class ViolationMetricBase(MetricBase):
    """Base class for evaluation of violation metrics."""

    def __init__(self, name: str, category: str, max_violation_threshold: int=0, metric_score_unit: Optional[str]=None) -> None:
        """
        Initializes the ViolationMetricBase class
        :param name: Metric name
        :param category: Metric category
        :param max_violation_threshold: Maximum threshold for the violation when computing the score.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._max_violation_threshold = max_violation_threshold
        self.number_of_violations = 0

    def aggregate_metric_violations(self, metric_violations: List[MetricViolation], scenario: AbstractScenario, time_series: Optional[TimeSeries]=None) -> List[MetricStatistics]:
        """
        Aggregates (possibly) multiple MetricViolations to a MetricStatistics.
        All the violations must be of the same metric.
        :param metric_violations: The list of violations for a single metric name.
        :param scenario: Scenario running this metric.
        :param time_series: Time series metrics.
        :return Statistics about the violations.
        """
        if not metric_violations:
            statistics = [Statistic(name=f'{self.name}', unit=MetricStatisticsType.BOOLEAN.unit, value=True, type=MetricStatisticsType.BOOLEAN)]
        else:
            sample_violation = metric_violations[0]
            name = sample_violation.name
            unit = sample_violation.unit
            extrema = []
            mean_values = []
            durations = []
            for violation in metric_violations:
                assert name == violation.name
                extrema.append(violation.extremum)
                mean_values.append(violation.mean)
                durations.append(violation.duration)
            max_val = max(extrema)
            min_val = min(extrema)
            mean_val = np.sum([mean_value * duration for mean_value, duration in zip(mean_values, durations)]) / sum(durations)
            statistics = [Statistic(name=f'number_of_violations_of_{self.name}', unit=MetricStatisticsType.COUNT.unit, value=len(metric_violations), type=MetricStatisticsType.COUNT), Statistic(name=f'max_violation_of_{self.name}', unit=unit, value=max_val, type=MetricStatisticsType.MAX), Statistic(name=f'min_violation_of_{self.name}', unit=unit, value=min_val, type=MetricStatisticsType.MIN), Statistic(name=f'mean_violation_of_{self.name}', unit=unit, value=mean_val, type=MetricStatisticsType.MEAN), Statistic(name=f'{self.name}', unit=MetricStatisticsType.BOOLEAN.unit, value=False, type=MetricStatisticsType.BOOLEAN)]
        self.number_of_violations = len(metric_violations)
        results: list[MetricStatistics] = self._construct_metric_results(metric_statistics=statistics, scenario=scenario, time_series=time_series, metric_score_unit=self.metric_score_unit)
        return results

    def _compute_violation_metric_score(self, number_of_violations: int) -> float:
        """
        Compute a metric score based on a violation threshold. It is 1 - (x / (max_violation_threshold + 1))
        The score will be 0 if the number of violations exceeds this value
        :param number_of_violations: Total number of violations
        :return A metric score between 0 and 1.
        """
        return max(0.0, 1.0 - number_of_violations / (self._max_violation_threshold + 1))

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        return self._compute_violation_metric_score(number_of_violations=self.number_of_violations)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        raise NotImplementedError

class MetricBase(AbstractMetricBuilder):
    """Base class for evaluation of metrics."""

    def __init__(self, name: str, category: str, metric_score_unit: Optional[str]=None) -> None:
        """
        Initializer for MetricBase
        :param name: Metric name
        :param category: Metric category.
        :param metric_score_unit: Metric final score unit.
        """
        self._name = name
        self._category = category
        self._metric_score_unit = metric_score_unit

    @property
    def name(self) -> str:
        """
        Returns the metric name
        :return the metric name.
        """
        return self._name

    @property
    def category(self) -> str:
        """
        Returns the metric category
        :return the metric category.
        """
        return self._category

    @property
    def metric_score_unit(self) -> Optional[str]:
        """
        Returns the metric final score unit.
        """
        return self._metric_score_unit

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> Optional[float]:
        """Inherited, see superclass."""
        return None

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        raise NotImplementedError

    def _compute_time_series_statistic(self, time_series: TimeSeries, statistics_type_list: Optional[List[MetricStatisticsType]]=None) -> List[Statistic]:
        """
        Compute metric statistics in time series.
        :param time_series: time series (with float values).
        :param statistics_type_list: List of available types such as [MetricStatisticsType.MAX,
        MetricStatisticsType.MIN, MetricStatisticsType.MEAN, MetricStatisticsType.P90]. Use all if set to None.
        :return A list of metric statistics.
        """
        values = time_series.values
        assert values, 'Time series values cannot be empty!'
        unit = time_series.unit
        if statistics_type_list is None:
            statistics_type_list = [MetricStatisticsType.MAX, MetricStatisticsType.MIN, MetricStatisticsType.MEAN, MetricStatisticsType.P90]
        statistics = []
        for statistics_type in statistics_type_list:
            if statistics_type == MetricStatisticsType.MAX:
                name = f'max_{self.name}'
                value = np.nanmax(values)
            elif statistics_type == MetricStatisticsType.MEAN:
                name = f'avg_{self.name}'
                value = np.nanmean(values)
            elif statistics_type == MetricStatisticsType.MIN:
                name = f'min_{self.name}'
                value = np.nanmin(values)
            elif statistics_type == MetricStatisticsType.P90:
                name = f'p90_{self.name}'
                value = np.nanpercentile(values, 90, method='closest_observation')
            else:
                raise TypeError('Other metric types statistics cannot be created by compute_statistics()')
            statistics.append(Statistic(name=name, unit=unit, value=value, type=statistics_type))
        return statistics

    def _construct_metric_results(self, metric_statistics: List[Statistic], scenario: AbstractScenario, metric_score_unit: Optional[str]=None, time_series: Optional[TimeSeries]=None) -> List[MetricStatistics]:
        """
        Construct metric results with statistics, scenario, and time series
        :param metric_statistics: A list of metric statistics
        :param scenario: Scenario running this metric to compute a metric score
        :param metric_score_unit: Unit for the metric final score.
        :param time_series: Time series object.
        :return: A list of metric statistics.
        """
        score = self.compute_score(scenario=scenario, metric_statistics=metric_statistics, time_series=time_series)
        result = MetricStatistics(metric_computator=self.name, name=self.name, statistics=metric_statistics, time_series=time_series, metric_category=self.category, metric_score=score, metric_score_unit=metric_score_unit)
        return [result]

    def _construct_open_loop_metric_results(self, scenario: AbstractScenario, comparison_horizon: List[int], maximum_threshold: float, metric_values: npt.NDArray[np.float64], name: str, unit: str, timestamps_sampled: List[int], metric_score_unit: str, selected_frames: List[int]) -> List[MetricStatistics]:
        """
        Construct metric results with statistics, scenario, and time series for open_loop metrics.
        :param scenario: Scenario running this metric to compute a metric score.
        :param comparison_horizon: List of horizon times in future (s) to find displacement errors.
        :param maximum_threshold: Maximum acceptable error threshold.
        :param metric_values: Time series object.
        :param name: name of timeseries.
        :param unit: metric unit.
        :param timestamps_sampled:A list of sampled timestamps.
        :param metric_score_unit: Unit for the metric final score.
        :param selected_frames: List sampled indices for nuboard Timeseries frames
        :return: A list of metric statistics.
        """
        metric_statistics: List[Statistic] = [Statistic(name=f'{name}_horizon_{horizon}', unit=unit, value=np.mean(metric_values[ind]), type=MetricStatisticsType.MEAN) for ind, horizon in enumerate(comparison_horizon)]
        metric_statistics.extend([Statistic(name=f'{self.name}', unit=MetricStatisticsType.BOOLEAN.unit, value=np.mean(metric_values) <= maximum_threshold, type=MetricStatisticsType.BOOLEAN), Statistic(name=f'avg_{name}_over_all_horizons', unit=unit, value=np.mean(metric_values), type=MetricStatisticsType.MEAN)])
        metric_values_over_horizons_at_each_time = np.mean(metric_values, axis=0)
        time_series = TimeSeries(unit=f'avg_{name}_over_all_horizons [{unit}]', time_stamps=timestamps_sampled, values=list(metric_values_over_horizons_at_each_time), selected_frames=selected_frames)
        results: List[MetricStatistics] = self._construct_metric_results(metric_statistics=metric_statistics, scenario=scenario, metric_score_unit=metric_score_unit, time_series=time_series)
        return results

class WithinBoundMetricBase(MetricBase):
    """Base class for evaluation of within_bound metrics."""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializes the WithinBoundMetricBase class
        :param name: Metric name
        :param category: Metric category.
        """
        super().__init__(name=name, category=category)
        self.within_bound_status: Optional[bool] = False

    @staticmethod
    def _compute_within_bound_metric_score(within_bound_status: bool) -> float:
        """
        Compute a metric score based on within bound condition
        :param within_bound_status: True if the value is within the bound, otherwise false
        :return 1.0 if within_bound_status is true otherwise 0.
        """
        return 1.0 if within_bound_status else 0.0

    def compute_score(self, scenario: AbstractScenario, metric_statistics: Dict[str, Statistic], time_series: Optional[TimeSeries]=None) -> Optional[float]:
        """Inherited, see superclass."""
        return None

    @staticmethod
    def _compute_within_bound(time_series: TimeSeries, min_within_bound_threshold: Optional[float]=None, max_within_bound_threshold: Optional[float]=None) -> Optional[bool]:
        """
        Compute if value is within bound based on the thresholds
        :param time_series: Time series object
        :param min_within_bound_threshold: Minimum threshold to check if value is within bound
        :param max_within_bound_threshold: Maximum threshold to check if value is within bound.
        """
        ego_pose_values: npt.NDArray[np.float32] = np.array(time_series.values)
        if not min_within_bound_threshold and (not max_within_bound_threshold):
            return None
        if min_within_bound_threshold is None:
            min_within_bound_threshold = float(-np.inf)
        if max_within_bound_threshold is None:
            max_within_bound_threshold = float(np.inf)
        ego_pose_value_within_bound = (ego_pose_values > min_within_bound_threshold) & (ego_pose_values < max_within_bound_threshold)
        return bool(np.all(ego_pose_value_within_bound))

    def _compute_statistics(self, history: SimulationHistory, scenario: AbstractScenario, statistic_unit_name: str, extract_function: Any, extract_function_params: Dict[str, Any], min_within_bound_threshold: Optional[float]=None, max_within_bound_threshold: Optional[float]=None) -> List[MetricStatistics]:
        """
        Compute metrics following the same structure
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :param statistic_unit_name: Statistic unit name
        :param extract_function: Function used to extract certain values
        :param extract_function_params: Params used in extract_function
        :param min_within_bound_threshold: Minimum threshold to check if value is within bound
        :param max_within_bound_threshold: Maximum threshold to check if value is within bound.
        """
        ego_pose_states = history.extract_ego_state
        ego_pose_values = extract_function(ego_pose_states, **extract_function_params)
        ego_pose_timestamps = extract_ego_time_point(ego_pose_states)
        time_series = TimeSeries(unit=statistic_unit_name, time_stamps=list(ego_pose_timestamps), values=list(ego_pose_values))
        statistics_type_list = [MetricStatisticsType.MAX, MetricStatisticsType.MIN, MetricStatisticsType.MEAN, MetricStatisticsType.P90]
        metric_statistics = self._compute_time_series_statistic(time_series=time_series, statistics_type_list=statistics_type_list)
        self.within_bound_status = self._compute_within_bound(time_series=time_series, min_within_bound_threshold=min_within_bound_threshold, max_within_bound_threshold=max_within_bound_threshold)
        if self.within_bound_status is not None:
            metric_statistics.append(Statistic(name=f'abs_{self.name}_within_bounds', unit=MetricStatisticsType.BOOLEAN.unit, value=self.within_bound_status, type=MetricStatisticsType.BOOLEAN))
        results: List[MetricStatistics] = self._construct_metric_results(metric_statistics=metric_statistics, time_series=time_series, scenario=scenario)
        return results

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        raise NotImplementedError

class EgoProgressAlongExpertRouteStatistics(MetricBase):
    """Ego progress along the expert route metric."""

    def __init__(self, name: str, category: str, score_progress_threshold: float=2, metric_score_unit: Optional[str]=None) -> None:
        """
        Initializes the EgoProgressAlongExpertRouteStatistics class
        :param name: Metric name
        :param category: Metric category
        :param score_progress_threshold: Progress distance threshold for the score.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._score_progress_threshold = score_progress_threshold
        self.results: List[MetricStatistics] = []

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[-1].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the ego progress along the expert route metric
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric
        :return: Ego progress along expert route statistics.
        """
        ego_states = history.extract_ego_state
        ego_poses = extract_ego_center(ego_states)
        expert_states = scenario.get_expert_ego_trajectory()
        expert_poses = extract_ego_center(expert_states)
        expert_route = get_route(map_api=history.map_api, poses=expert_poses)
        expert_route_simplified = get_route_simplified(expert_route)
        if not expert_route_simplified:
            statistics = [Statistic(name='expert_total_progress_along_route', unit='meters', value=0.0, type=MetricStatisticsType.VALUE), Statistic(name='ego_expert_progress_along_route_ratio', unit=MetricStatisticsType.RATIO.unit, value=1.0, type=MetricStatisticsType.RATIO)]
            self.results = self._construct_metric_results(metric_statistics=statistics, scenario=scenario)
        else:
            route_baseline_roadblock_pairs = get_route_baseline_roadblock_linkedlist(history.map_api, expert_route_simplified)
            ego_progress_computer = PerFrameProgressAlongRouteComputer(route_roadblocks=route_baseline_roadblock_pairs)
            ego_progress = ego_progress_computer(ego_poses=ego_poses)
            overall_ego_progress = np.sum(ego_progress)
            expert_progress_computer = PerFrameProgressAlongRouteComputer(route_roadblocks=route_baseline_roadblock_pairs)
            expert_progress = expert_progress_computer(ego_poses=expert_poses)
            overall_expert_progress = np.sum(expert_progress)
            if overall_ego_progress < -self._score_progress_threshold:
                ego_expert_progress_along_route_ratio = 0
            else:
                ego_expert_progress_along_route_ratio = min(1.0, max(overall_ego_progress, self._score_progress_threshold) / max(overall_expert_progress, self._score_progress_threshold))
            ego_timestamps = extract_ego_time_point(ego_states)
            time_series = TimeSeries(unit='meters', time_stamps=list(ego_timestamps), values=list(ego_progress))
            statistics = [Statistic(name='expert_total_progress_along_route', unit='meters', value=float(overall_expert_progress), type=MetricStatisticsType.VALUE), Statistic(name='ego_total_progress_along_route', unit='meters', value=float(overall_ego_progress), type=MetricStatisticsType.VALUE), Statistic(name='ego_expert_progress_along_route_ratio', unit=MetricStatisticsType.RATIO.unit, value=ego_expert_progress_along_route_ratio, type=MetricStatisticsType.RATIO)]
            self.results = self._construct_metric_results(metric_statistics=statistics, scenario=scenario, time_series=time_series, metric_score_unit=self.metric_score_unit)
        return self.results

def extract_ego_center(ego_states: List[EgoState]) -> List[Point2D]:
    """
    Extract xy position of center from a list of ego_states
    :param ego_states: list of ego states
    :return List of ego center positions.
    """
    xy_poses: List[Point2D] = [ego_state.center.point for ego_state in ego_states]
    return xy_poses

def get_route_baseline_roadblock_linkedlist(map_api: AbstractMap, expert_route: List[List[LaneGraphEdgeMapObject]]) -> RouteRoadBlockLinkedList:
    """
    This function generates a linked list of baseline & unique road-block pairs
    (RouteBaselineRoadBlockPair) from a simplified route
    :param map_api: Corresponding map
    :param expert_route: A route list
    :return A linked list of RouteBaselineRoadBlockPair.
    """
    route_baseline_roadblock_list = RouteRoadBlockLinkedList()
    prev_roadblock_id = None
    for route_object in expert_route:
        if route_object:
            roadblock_id = route_object[0].get_roadblock_id()
            if roadblock_id != prev_roadblock_id:
                prev_roadblock_id = roadblock_id
                if isinstance(route_object[0], Lane):
                    road_block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
                else:
                    road_block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
                ref_baseline_path = route_object[0].baseline_path
                if route_baseline_roadblock_list.head is None:
                    prev_route_baseline_roadblock = RouteBaselineRoadBlockPair(base_line=ref_baseline_path, road_block=road_block)
                    route_baseline_roadblock_list.head = prev_route_baseline_roadblock
                else:
                    prev_route_baseline_roadblock.next = RouteBaselineRoadBlockPair(base_line=ref_baseline_path, road_block=road_block)
                    prev_route_baseline_roadblock = prev_route_baseline_roadblock.next
    return route_baseline_roadblock_list

class EgoIsComfortableStatistics(MetricBase):
    """
    Check if ego trajectory is comfortable based on min_ego_lon_acceleration, max_ego_lon_acceleration,
    max_ego_abs_lat_acceleration, max_ego_abs_yaw_rate, max_ego_abs_yaw_acceleration, max_ego_abs_jerk_lon,
    max_ego_abs_jerk.
    """

    def __init__(self, name: str, category: str, ego_jerk_metric: EgoJerkStatistics, ego_lat_acceleration_metric: EgoLatAccelerationStatistics, ego_lon_acceleration_metric: EgoLonAccelerationStatistics, ego_lon_jerk_metric: EgoLonJerkStatistics, ego_yaw_acceleration_metric: EgoYawAccelerationStatistics, ego_yaw_rate_metric: EgoYawRateStatistics, metric_score_unit: Optional[str]=None) -> None:
        """
        Initializes the EgoIsComfortableStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_jerk_metric: Ego jerk metric
        :param ego_lat_acceleration_metric: Ego lat acceleration metric
        :param ego_lon_acceleration_metric: Ego lon acceleration metric
        :param ego_lon_jerk_metric: Ego lon jerk metric
        :param ego_yaw_acceleration_metric: Ego yaw acceleration metric
        :param ego_yaw_rate_metric: Ego yaw rate metric.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._comfortability_metrics = [ego_jerk_metric, ego_lat_acceleration_metric, ego_lon_acceleration_metric, ego_lon_jerk_metric, ego_yaw_acceleration_metric, ego_yaw_rate_metric]

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)

    def check_ego_is_comfortable(self, history: SimulationHistory, scenario: AbstractScenario) -> bool:
        """
        Check if ego trajectory is comfortable
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return Ego comfortable status.
        """
        metrics_results = [metric.within_bound_status for metric in self._comfortability_metrics]
        ego_is_comfortable = bool(np.all(metrics_results))
        return ego_is_comfortable

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        ego_is_comfortable = self.check_ego_is_comfortable(history=history, scenario=scenario)
        statistics = [Statistic(name='ego_is_comfortable', unit=MetricStatisticsType.BOOLEAN.unit, value=ego_is_comfortable, type=MetricStatisticsType.BOOLEAN)]
        results: List[MetricStatistics] = self._construct_metric_results(metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit)
        return results

class PlannerExpertAverageL2ErrorStatistics(MetricBase):
    """Average displacement error metric between the planned ego pose and expert."""

    def __init__(self, name: str, category: str, comparison_horizon: List[int], comparison_frequency: int, max_average_l2_error_threshold: float, metric_score_unit: Optional[str]=None) -> None:
        """
        Initialize the PlannerExpertL2ErrorStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param comparison_horizon: List of horizon times in future (s) to find displacement errors.
        :param comparison_frequency: Frequency to sample expert and planner trajectory.
        :param max_average_l2_error_threshold: Maximum acceptable error threshold.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self.comparison_horizon = comparison_horizon
        self._comparison_frequency = comparison_frequency
        self._max_average_l2_error_threshold = max_average_l2_error_threshold
        self.maximum_displacement_errors: npt.NDArray[np.float64] = np.array([0])
        self.final_displacement_errors: npt.NDArray[np.float64] = np.array([0])
        self.expert_timestamps_sampled: List[int] = []
        self.average_heading_errors: npt.NDArray[np.float64] = np.array([0])
        self.final_heading_errors: npt.NDArray[np.float64] = np.array([0])
        self.selected_frames: List[int] = [0]

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        return float(max(0, 1 - metric_statistics[-1].value / self._max_average_l2_error_threshold))

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        expert_frequency = 1 / scenario.database_interval
        step_size = int(expert_frequency / self._comparison_frequency)
        sampled_indices = list(range(0, len(history.data), step_size))
        expert_states = list(itertools.chain(list(scenario.get_expert_ego_trajectory())[0::step_size], scenario.get_ego_future_trajectory(sampled_indices[-1], max(self.comparison_horizon), max(self.comparison_horizon) // self._comparison_frequency)))
        expert_traj_poses = extract_ego_center_with_heading(expert_states)
        expert_timestamps_sampled = extract_ego_time_point(expert_states)
        planned_trajectories = list((history.data[index].trajectory for index in sampled_indices))
        average_displacement_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        maximum_displacement_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        final_displacement_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        average_heading_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        final_heading_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        for curr_frame, curr_ego_planned_traj in enumerate(planned_trajectories):
            future_horizon_frame = int(curr_frame + max(self.comparison_horizon))
            planner_interpolated_traj = list((curr_ego_planned_traj.get_state_at_time(TimePoint(int(timestamp))) for timestamp in expert_timestamps_sampled[curr_frame:future_horizon_frame + 1] if timestamp <= curr_ego_planned_traj.end_time.time_us))
            if len(planner_interpolated_traj) < max(self.comparison_horizon) + 1:
                planner_interpolated_traj = list(itertools.chain(planner_interpolated_traj, [curr_ego_planned_traj.get_sampled_trajectory()[-1]]))
                expert_traj = expert_traj_poses[curr_frame + 1:future_horizon_frame] + [InterpolatedTrajectory(expert_states).get_state_at_time(curr_ego_planned_traj.end_time).center]
            else:
                expert_traj = expert_traj_poses[curr_frame + 1:future_horizon_frame + 1]
            planner_interpolated_traj_poses = extract_ego_center_with_heading(planner_interpolated_traj)
            displacement_errors = compute_traj_errors(planner_interpolated_traj_poses[1:], expert_traj, heading_diff_weight=0)
            heading_errors = compute_traj_heading_errors(planner_interpolated_traj_poses[1:], expert_traj)
            for ind, horizon in enumerate(self.comparison_horizon):
                horizon_index = horizon // self._comparison_frequency
                average_displacement_errors[ind, curr_frame] = np.mean(displacement_errors[:horizon_index])
                maximum_displacement_errors[ind, curr_frame] = np.max(displacement_errors[:horizon_index])
                final_displacement_errors[ind, curr_frame] = displacement_errors[horizon_index - 1]
                average_heading_errors[ind, curr_frame] = np.mean(heading_errors[:horizon_index])
                final_heading_errors[ind, curr_frame] = heading_errors[horizon_index - 1]
        self.ego_timestamps_sampled = expert_timestamps_sampled[:len(sampled_indices)]
        self.selected_frames = sampled_indices
        results: List[MetricStatistics] = self._construct_open_loop_metric_results(scenario, self.comparison_horizon, self._max_average_l2_error_threshold, metric_values=average_displacement_errors, name='planner_expert_ADE', unit='meter', timestamps_sampled=self.ego_timestamps_sampled, metric_score_unit=self.metric_score_unit, selected_frames=sampled_indices)
        self.maximum_displacement_errors = maximum_displacement_errors
        self.final_displacement_errors = final_displacement_errors
        self.average_heading_errors = average_heading_errors
        self.final_heading_errors = final_heading_errors
        return results

def extract_ego_center_with_heading(ego_states: List[EgoState]) -> List[StateSE2]:
    """
    Extract xy position of center and heading from a list of ego_states
    :param ego_states: list of ego states
    :return a list of StateSE2.
    """
    xy_poses_and_heading: List[StateSE2] = [ego_state.center for ego_state in ego_states]
    return xy_poses_and_heading

def _get_collision_type(ego_state: EgoState, tracked_object: TrackedObject, stopped_speed_threshold: float=0.05) -> CollisionType:
    """
    Classify collision between ego and the track.
    :param ego_state: Ego's state at the current timestamp.
    :param tracked_object: Tracked object.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return Collision type.
    """
    is_ego_stopped = ego_state.dynamic_car_state.speed <= stopped_speed_threshold
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION
    elif is_track_stopped(tracked_object):
        collision_type = CollisionType.STOPPED_TRACK_COLLISION
    elif is_agent_behind(ego_state.rear_axle, tracked_object.box.center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION
    elif LineString([ego_state.car_footprint.oriented_box.geometry.exterior.coords[0], ego_state.car_footprint.oriented_box.geometry.exterior.coords[3]]).intersects(tracked_object.box.geometry):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION
    return collision_type

def is_track_stopped(tracked_object: TrackedObject, stopped_speed_threshhold: float=0.05) -> bool:
    """
    Evaluates if a tracked object is stopped
    :param tracked_object: tracked_object representation
    :param stopped_speed_threshhold: Threshhold for 0 speed due to noise
    :return: True if track is stopped else False.
    """
    return True if not isinstance(tracked_object, Agent) else bool(tracked_object.velocity.magnitude() <= stopped_speed_threshhold)

def find_new_collisions(ego_state: EgoState, observation: DetectionsTracks, collided_track_ids: Set[str]) -> Tuple[Set[str], Dict[str, CollisionData]]:
    """
    Identify and classify new collisions in a given timestamp. We assume that ego can only collide with an agent
    once in the scenario. Collided tracks will be removed from metrics evaluation at future timestamps.
    :param ego_state: Ego's state at the current timestamp.
    :param observation: DetectionsTracks at the current timestamp.
    :param collided_track_ids: Set of all collisions happend before the current timestamp.
    :return Updated set of collided track ids and a dict of new collided tracks and their CollisionData.
    """
    collisions_id_data: Dict[str, CollisionData] = {}
    for tracked_object in observation.tracked_objects:
        if tracked_object.track_token not in collided_track_ids and in_collision(ego_state.car_footprint.oriented_box, tracked_object.box):
            collided_track_ids.add(tracked_object.track_token)
            collision_delta_v = ego_delta_v_collision(ego_state, tracked_object)
            collision_type = _get_collision_type(ego_state, tracked_object)
            collisions_id_data[tracked_object.track_token] = CollisionData(collision_delta_v, collision_type, tracked_object.tracked_object_type)
    return (collided_track_ids, collisions_id_data)

def ego_delta_v_collision(ego_state: EgoState, scene_object: SceneObject, ego_mass: float=2000, agent_mass: float=2000) -> float:
    """
    Compute the ego delta V (loss of velocity during the collision). Delta V represents the intensity of the collision
    of the ego with other agents.
    :param ego_state: The state of ego.
    :param scene_object: The scene_object ego is colliding with.
    :param ego_mass: mass of ego.
    :param agent_mass: mass of the agent.
    :return The delta V measure for ego.
    """
    ego_mass_ratio = agent_mass / (agent_mass + ego_mass)
    scene_object_speed = scene_object.velocity.magnitude() if isinstance(scene_object, Agent) else 0
    sum_speed_squared = ego_state.dynamic_car_state.speed ** 2 + scene_object_speed ** 2
    cos_rule_term = 2 * ego_state.dynamic_car_state.speed * scene_object_speed * np.cos(ego_state.rear_axle.heading - scene_object.center.heading)
    velocity_component = float(np.sqrt(sum_speed_squared - cos_rule_term))
    return ego_mass_ratio * velocity_component

class EgoAtFaultCollisionStatistics(MetricBase):
    """
    Statistics on number and energy of collisions of ego.
    A collision is defined as the event of ego intersecting another bounding box. If the same collision lasts for
    multiple frames, it still counts as a single one.
    """

    def __init__(self, name: str, category: str, ego_lane_change_metric: EgoLaneChangeStatistics, max_violation_threshold_vru: int=0, max_violation_threshold_vehicle: int=0, max_violation_threshold_object: int=1, metric_score_unit: Optional[str]=None) -> None:
        """
        Initialize the EgoAtFaultCollisionStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param ego_lane_change_metric: Lane change metric computed prior to calling the current metric.
        :param max_violation_threshold_vru: Maximum threshold for the collision with VRUs.
        :param max_violation_threshold_vehicle: Maximum threshold for the collision with vehicles.
        :param max_violation_threshold_object: Maximum threshold for the collision with objects.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._max_violation_threshold_vru = max_violation_threshold_vru
        self._max_violation_threshold_vehicle = max_violation_threshold_vehicle
        self._max_violation_threshold_object = max_violation_threshold_object
        self.results: List[MetricStatistics] = []
        self.all_collisions: List[Collisions] = []
        self.all_at_fault_collisions: Dict[TrackedObjectType, List[float]] = defaultdict(list)
        self.timestamps_at_fault_collisions: List[int] = []
        self._ego_lane_change_metric = ego_lane_change_metric

    def _compute_collision_score(self, number_of_collisions: int, max_violation_threshold: int) -> float:
        """
        Compute a score based on a maximum violation threshold. The score is max( 0, 1 - (x / (max_violation_threshold + 1)))
        The score will be 0 if the number of collisions exceeds this value.
        :param max_violation_threshold: Total number of allowed collisions.
        :return A metric score between 0 and 1.
        """
        return max(0.0, 1.0 - number_of_collisions / (max_violation_threshold + 1))

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> Optional[float]:
        """Inherited, see superclass.
        The total score for this metric is defined as the product of the scores for VRUs, vehicles and object track types. If no at fault collision exist, the score is 1.
        """
        return 1 if metric_statistics[0].value else self._compute_collision_score(metric_statistics[2].value, self._max_violation_threshold_vru) * self._compute_collision_score(metric_statistics[3].value, self._max_violation_threshold_vehicle) * self._compute_collision_score(metric_statistics[4].value, self._max_violation_threshold_object)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the collision metric.
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated collision energy and counts.
        """
        assert self._ego_lane_change_metric.results, 'ego_lane_change_metric must be run prior to calling {}'.format(self.name)
        timestamps_in_common_or_connected_route_objs: List[int] = self._ego_lane_change_metric.timestamps_in_common_or_connected_route_objs
        all_collisions: List[Collisions] = []
        collided_track_ids: Set[str] = set()
        for sample in history.data:
            ego_state = sample.ego_state
            observation = sample.observation
            timestamp = ego_state.time_point.time_us
            collided_track_ids, collisions_id_data = find_new_collisions(ego_state, observation, collided_track_ids)
            if len(collisions_id_data):
                all_collisions.append(Collisions(timestamp, collisions_id_data))
        self.timestamps_at_fault_collisions, self.all_at_fault_collisions = classify_at_fault_collisions(all_collisions, timestamps_in_common_or_connected_route_objs)
        number_of_at_fault_collisions = sum((len(track_collisions) for track_collisions in self.all_at_fault_collisions.values()))
        statistics = [Statistic(name=f'{self.name}', unit=MetricStatisticsType.BOOLEAN.unit, value=number_of_at_fault_collisions == 0, type=MetricStatisticsType.BOOLEAN), Statistic(name='number_of_all_at_fault_collisions', unit=MetricStatisticsType.COUNT.unit, value=number_of_at_fault_collisions, type=MetricStatisticsType.COUNT)]
        statistics.extend(get_fault_type_statistics(self.all_at_fault_collisions))
        self.results = self._construct_metric_results(metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit)
        self.all_collisions = all_collisions
        return self.results

def classify_at_fault_collisions(all_collisions: List[Collisions], timestamps_in_common_or_connected_route_objs: List[int]) -> Tuple[List[int], Dict[TrackedObjectType, List[float]]]:
    """
    Return a list of timestamps that at fault collisions happened and a dictionary of track types and collision energy.

    We consider at_fault_collisions as collisions that could have been prevented if planner
    performed differently. For simplicity we call these collisions at fault although the proposed classification is
    not complete and there are more cases to be considered.

    :param all_collisions: List of all collisions in the history.
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
    lanes/lane connectors.
    :return: A list of timestamps that at fault collisions happened and a dictionary of track types and collision energy.
    """
    at_fault_collisions: Dict[TrackedObjectType, List[float]] = defaultdict(list)
    timestamps_at_fault_collisions: List[int] = []
    for collision in all_collisions:
        timestamp = collision.timestamp
        ego_in_multiple_lanes_or_nondrivable_area = timestamp not in timestamps_in_common_or_connected_route_objs
        for _id, collision_data in collision.collisions_id_data.items():
            collisions_at_stopped_track_or_active_front = collision_data.collision_type in [CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION]
            collision_at_lateral = collision_data.collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
            if collisions_at_stopped_track_or_active_front or (ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral):
                timestamps_at_fault_collisions.append(timestamp)
                at_fault_collisions[collision_data.tracked_object_type].append(collision_data.collision_ego_delta_v)
    return (timestamps_at_fault_collisions, at_fault_collisions)

class SpeedLimitComplianceStatistics(ViolationMetricBase):
    """Statistics on speed limit compliance of ego."""

    def __init__(self, name: str, category: str, lane_change_metric: EgoLaneChangeStatistics, max_violation_threshold: int, max_overspeed_value_threshold: float, metric_score_unit: Optional[str]=None) -> None:
        """
        Initializes the SpeedLimitComplianceStatistics class
        :param name: Metric name
        :param category: Metric category
        :param lane_change_metric: lane change metric
        :param max_violation_threshold: Maximum threshold for the number of violation
        :param max_overspeed_value_threshold: A threshold for overspeed value driving above which is considered more
        dangerous.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, max_violation_threshold=max_violation_threshold, metric_score_unit=metric_score_unit)
        self._max_overspeed_value_threshold = max_overspeed_value_threshold
        self._lane_change_metric = lane_change_metric

    def _compute_violation_metric_score(self, time_series: TimeSeries) -> float:
        """
        Compute a metric score based on the durtaion and magnitude of the violation compared to the scenario
        duration and a threshold for overspeed value.
        :param time_series: A time series for the overspeed
        :return: A metric score between 0 and 1.
        """
        dt_in_sec = np.mean(np.diff(time_series.time_stamps)) * 1e-06
        scenario_duration_in_sec = (time_series.time_stamps[-1] - time_series.time_stamps[0]) * 1e-06
        if scenario_duration_in_sec <= 0:
            logger.warning('Scenario duration is 0 or less!')
            return 1.0
        max_overspeed_value_threshold = max(self._max_overspeed_value_threshold, 0.001)
        violation_loss = np.sum(time_series.values) * dt_in_sec / (max_overspeed_value_threshold * scenario_duration_in_sec)
        return float(max(0.0, 1.0 - violation_loss))

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        if metric_statistics[-1].value:
            return 1.0
        return float(self._compute_violation_metric_score(time_series=time_series))

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        ego_route = self._lane_change_metric.ego_driven_route
        extractor = SpeedLimitViolationExtractor(history=history, metric_name=self._name, category=self._category)
        extractor.extract_metric(ego_route=ego_route)
        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_series = TimeSeries(unit='over_speeding[meters_per_second]', time_stamps=list(time_stamps), values=extractor.violation_depths)
        violation_statistics: List[MetricStatistics] = self.aggregate_metric_violations(metric_violations=extractor.violations, scenario=scenario, time_series=time_series)
        return violation_statistics

class EgoLatAccelerationStatistics(WithinBoundMetricBase):
    """Ego lateral acceleration metric."""

    def __init__(self, name: str, category: str, max_abs_lat_accel: float) -> None:
        """
        Initializes the EgoLatAccelerationStatistics class
        :param name: Metric name
        :param category: Metric category
        :param max_abs_lat_accel: Maximum threshold to define if absolute lateral acceleration is within bound.
        """
        super().__init__(name=name, category=category)
        self._max_abs_lat_accel = max_abs_lat_accel

    @staticmethod
    def compute_comfortability(history: SimulationHistory, max_abs_lat_accel: float) -> bool:
        """
        Compute comfortability based on max_abs_lat_accel
        :param history: History from a simulation engine
        :param max_abs_lat_accel: Threshold for the absolute lat jerk
        :return True if within the threshold otherwise false.
        """
        ego_pose_states = history.extract_ego_state
        ego_pose_lat_accels = extract_ego_acceleration(ego_pose_states, acceleration_coordinate='y')
        lat_accels_within_bounds = np.abs(ego_pose_lat_accels) < max_abs_lat_accel
        return bool(np.all(lat_accels_within_bounds))

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the lateral acceleration metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated lateral acceleration metric.
        """
        metric_statistics: List[MetricStatistics] = self._compute_statistics(history=history, scenario=scenario, statistic_unit_name='meters_per_second_squared', extract_function=extract_ego_acceleration, extract_function_params={'acceleration_coordinate': 'y'}, min_within_bound_threshold=-self._max_abs_lat_accel, max_within_bound_threshold=self._max_abs_lat_accel)
        return metric_statistics

class DrivableAreaComplianceStatistics(MetricBase):
    """Statistics on drivable area compliance of ego."""

    def __init__(self, name: str, category: str, lane_change_metric: EgoLaneChangeStatistics, max_violation_threshold: float, metric_score_unit: Optional[str]=None) -> None:
        """
        Initialize the DrivableAreaComplianceStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param lane_change_metric: lane change metric.
        :param max_violation_threshold: [m] tolerance threshold.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self.results: List[MetricStatistics] = []
        self._lane_change_metric = lane_change_metric
        self._max_violation_threshold = max_violation_threshold

    @staticmethod
    def not_in_drivable_area_with_route_object(pose: Point2D, route_object: List[GraphEdgeMapObject], map_api: AbstractMap) -> bool:
        """
        Return a boolean is_in_drivable_area.
        :param pose: pose.
        :param route_object: lane/lane connector of that pose or empty list.
        :param map_api: map.
        :return: a boolean is_in_drivable_area.
        """
        return not route_object and (not map_api.is_in_layer(pose, layer=SemanticMapLayer.DRIVABLE_AREA))

    @staticmethod
    def compute_distance_to_map_objects_list(pose: Point2D, map_objects: List[GraphEdgeMapObject]) -> float:
        """
        Compute the min distance to a list of map objects.
        :param pose: pose.
        :param map_objects: list of map objects.
        :return: distance.
        """
        return float(min((obj.polygon.distance(Point(*pose)) for obj in map_objects)))

    def is_corner_far_from_drivable_area(self, map_api: AbstractMap, center_lane_lane_connector: List[GraphEdgeMapObject], ego_corner: Point2D) -> bool:
        """
        Return a boolean that shows if ego_corner is far from drivable area according to the threshold.
        :param map_api: map api.
        :param center_lane_lane_connector: ego's center route obj in iteration.
        :param ego_corner: one of ego's corners.
        :return: boolean is_corner_far_from_drivable_area.
        """
        if center_lane_lane_connector:
            distance = self.compute_distance_to_map_objects_list(ego_corner, center_lane_lane_connector)
            if distance < self._max_violation_threshold:
                return False
        id_distance_tuple = map_api.get_distance_to_nearest_map_object(ego_corner, layer=SemanticMapLayer.DRIVABLE_AREA)
        return id_distance_tuple[1] is None or id_distance_tuple[1] >= self._max_violation_threshold

    def compute_violation_for_iteration(self, map_api: AbstractMap, ego_corners: List[Point2D], corners_lane_lane_connector: CornersGraphEdgeMapObject, center_lane_lane_connector: List[GraphEdgeMapObject], far_from_drivable_area: bool) -> Tuple[bool, bool]:
        """
        Compute violation of drivable area for an iteration.
        :param map_api: map api.
        :param ego_corners: 4 corners of ego (FL, RL, RR, FR) in iteration.
        :param corners_lane_lane_connector: object holding corners route objects.
        :param center_lane_lane_connector: ego's center route obj in iteration.
        :param far_from_drivable_area: boolean showing if ego got far from drivable_area in a previous iteration.
        :return: booleans not_in_drivable_area, far_from_drivable_area.
        """
        outside_drivable_area_objs = [ind for ind, obj in enumerate(corners_lane_lane_connector) if self.not_in_drivable_area_with_route_object(ego_corners[ind], obj, map_api)]
        not_in_drivable_area = len(outside_drivable_area_objs) > 0
        far_from_drivable_area = far_from_drivable_area or any((self.is_corner_far_from_drivable_area(map_api, center_lane_lane_connector, ego_corners[ind]) for ind in outside_drivable_area_objs))
        return (not_in_drivable_area, far_from_drivable_area)

    def extract_metric(self, history: SimulationHistory) -> Tuple[List[float], bool]:
        """
        Extract the drivable area violations from the history of Ego poses to evaluate drivable area compliance.
        :param history: SimulationHistory.
        :param corners_lane_lane_connector_list: List of corners lane and lane connectors.
        :return: list of float that shows if corners are in drivable area.
        """
        ego_states = history.extract_ego_state
        map_api = history.map_api
        all_ego_corners = extract_ego_corners(ego_states)
        corners_lane_lane_connector_list = self._lane_change_metric.corners_route
        center_route = self._lane_change_metric.ego_driven_route
        corners_in_drivable_area = []
        far_from_drivable_area = False
        for ego_corners, corners_lane_lane_connector, center_lane_lane_connector in zip(all_ego_corners, corners_lane_lane_connector_list, center_route):
            not_in_drivable_area, far_from_drivable_area = self.compute_violation_for_iteration(map_api, ego_corners, corners_lane_lane_connector, center_lane_lane_connector, far_from_drivable_area)
            corners_in_drivable_area.append(float(not not_in_drivable_area))
        return (corners_in_drivable_area, far_from_drivable_area)

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return: the estimated metric.
        """
        corners_in_drivable_area, far_from_drivable_area = self.extract_metric(history=history)
        statistics = [Statistic(name=f'{self.name}', unit=MetricStatisticsType.BOOLEAN.unit, value=float(not far_from_drivable_area), type=MetricStatisticsType.BOOLEAN)]
        self.results = self._construct_metric_results(metric_statistics=statistics, scenario=scenario, metric_score_unit=self._metric_score_unit)
        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_series = TimeSeries(unit='boolean', time_stamps=list(time_stamps), values=corners_in_drivable_area)
        corners_statistics = [Statistic(name='corners_in_drivable_area', unit=MetricStatisticsType.BOOLEAN.unit, value=float(np.all(corners_in_drivable_area)), type=MetricStatisticsType.BOOLEAN)]
        corners_statistics_result = MetricStatistics(metric_computator=self.name, name='corners_in_drivable_area', statistics=corners_statistics, time_series=time_series, metric_category=self.category)
        self.results.append(corners_statistics_result)
        return self.results

class EgoExpertL2ErrorStatistics(MetricBase):
    """Ego pose L2 error metric w.r.t expert."""

    def __init__(self, name: str, category: str, discount_factor: float) -> None:
        """
        Initializes the EgoExpertL2ErrorStatistics class
        :param name: Metric name
        :param category: Metric category
        :param discount_factor: Displacement at step i is discounted by discount_factor^i.
        """
        super().__init__(name=name, category=category)
        self._discount_factor = discount_factor

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        ego_states = history.extract_ego_state
        expert_states = scenario.get_expert_ego_trajectory()
        ego_traj = extract_ego_center(ego_states)
        expert_traj = extract_ego_center(expert_states)
        error = compute_traj_errors(ego_traj=ego_traj, expert_traj=expert_traj, discount_factor=self._discount_factor)
        ego_timestamps = extract_ego_time_point(ego_states)
        statistics_type_list = [MetricStatisticsType.MAX, MetricStatisticsType.MEAN, MetricStatisticsType.P90]
        time_series = TimeSeries(unit='meters', time_stamps=list(ego_timestamps), values=list(error))
        metric_statistics = self._compute_time_series_statistic(time_series=time_series, statistics_type_list=statistics_type_list)
        results: List[MetricStatistics] = self._construct_metric_results(metric_statistics=metric_statistics, scenario=scenario, time_series=time_series)
        return results

class PlannerMissRateStatistics(MetricBase):
    """Miss rate defined based on the maximum L2 error of planned ego pose w.r.t expert."""

    def __init__(self, name: str, category: str, planner_expert_average_l2_error_within_bound_metric: PlannerExpertAverageL2ErrorStatistics, max_displacement_threshold: List[float], max_miss_rate_threshold: float, metric_score_unit: Optional[str]=None) -> None:
        """
        Initialize the PlannerMissRateStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param planner_expert_average_l2_error_within_bound_metric: planner_expert_average_l2_error_within_bound metric for each horizon.
        :param max_displacement_threshold: A List of thresholds at different horizons
        :param max_miss_rate_threshold: maximum acceptable miss rate threshold.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._max_displacement_threshold = max_displacement_threshold
        self._max_miss_rate_threshold = max_miss_rate_threshold
        self._planner_expert_average_l2_error_within_bound_metric = planner_expert_average_l2_error_within_bound_metric

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[-1].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        maximum_displacement_errors = self._planner_expert_average_l2_error_within_bound_metric.maximum_displacement_errors
        comparison_horizon = self._planner_expert_average_l2_error_within_bound_metric.comparison_horizon
        miss_rates: npt.NDArray[np.float64] = np.array([np.mean(maximum_displacement_errors[i] > self._max_displacement_threshold[i]) for i in range(len(comparison_horizon))])
        metric_statistics = [Statistic(name=f'planner_miss_rate_horizon_{comparison_horizon[ind]}', unit=MetricStatisticsType.RATIO.unit, value=miss_rate, type=MetricStatisticsType.RATIO) for ind, miss_rate in enumerate(miss_rates)]
        metric_statistics.append(Statistic(name=f'{self.name}', unit=MetricStatisticsType.BOOLEAN.unit, value=float(np.all(miss_rates <= self._max_miss_rate_threshold)), type=MetricStatisticsType.BOOLEAN))
        results: List[MetricStatistics] = self._construct_metric_results(metric_statistics=metric_statistics, scenario=scenario, metric_score_unit=self.metric_score_unit)
        return results

class EgoExpertL2ErrorWithYawStatistics(MetricBase):
    """Ego pose and heading L2 error metric w.r.t expert."""

    def __init__(self, name: str, category: str, discount_factor: float, heading_diff_weight: float=2.5) -> None:
        """
        Initializes the EgoExpertL2ErrorWithYawStatistics class
        :param name: Metric name
        :param category: Metric category
        :param discount_factor: Displacement at step i is dicounted by discount_factor^i
        :heading_diff_weight: The weight of heading differences.
        """
        super().__init__(name=name, category=category)
        self._discount_factor = discount_factor
        self._heading_diff_weight = heading_diff_weight

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        ego_states = history.extract_ego_state
        expert_states = scenario.get_expert_ego_trajectory()
        ego_traj = extract_ego_center_with_heading(ego_states)
        expert_traj = extract_ego_center_with_heading(expert_states)
        error = compute_traj_errors(ego_traj=ego_traj, expert_traj=expert_traj, discount_factor=self._discount_factor, heading_diff_weight=self._heading_diff_weight)
        ego_timestamps = extract_ego_time_point(ego_states)
        statistics_type_list = [MetricStatisticsType.MAX, MetricStatisticsType.MEAN, MetricStatisticsType.P90]
        time_series = TimeSeries(unit='None', time_stamps=list(ego_timestamps), values=list(error))
        metric_statistics = self._compute_time_series_statistic(time_series=time_series, statistics_type_list=statistics_type_list)
        results: List[MetricStatistics] = self._construct_metric_results(metric_statistics=metric_statistics, scenario=scenario, time_series=time_series)
        return results

def extract_tracks_info_excluding_collided_tracks(ego_states: List[EgoState], ego_timestamps: npt.NDArray[np.int64], observations: List[Observation], all_collisions: List[Collisions], timestamps_in_common_or_connected_route_objs: List[int], map_api: AbstractMap) -> TRACKS_POSE_SPEED_BOX:
    """
    Extracts arrays of tracks pose, speed and oriented box for TTC: all lead and cross tracks, plus lateral tracks if ego is in
    between lanes or in nondrivable area or in intersection.

    :param ego_states: A list of ego states
    :param ego_timestamps: Array of times in time_us
    :param observations: A list of observations
    :param all_collisions: List of all collisions in the history
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
        lanes/lane connectors
    :param map_api: map api.
    :return: A tuple of lists of arrays of tracks pose, speed and represented box at each timestep.
    """
    collided_track_ids: Set[str] = set()
    history_tracks_poses: List[npt.NDArray[np.float64]] = []
    history_tracks_speed: List[npt.NDArray[np.float64]] = []
    history_tracks_boxes: List[npt.NDArray[OrientedBox]] = []
    collision_time_dict = {collision.timestamp: list(collision.collisions_id_data.keys()) for collision in all_collisions}
    for ego_state, timestamp, observation in zip(ego_states, ego_timestamps, observations):
        collided_track_ids = collided_track_ids.union(set(collision_time_dict.get(timestamp, [])))
        ego_not_in_common_or_connected_route_objs = timestamp not in timestamps_in_common_or_connected_route_objs
        tracked_objects = [tracked_object for tracked_object in observation.tracked_objects if tracked_object.track_token not in collided_track_ids and (is_agent_ahead(ego_state.rear_axle, tracked_object.center) or ((ego_not_in_common_or_connected_route_objs or map_api.is_in_layer(ego_state.rear_axle, layer=SemanticMapLayer.INTERSECTION)) and (not is_agent_behind(ego_state.rear_axle, tracked_object.center))))]
        poses: List[npt.NDArray[np.float64]] = [np.array([*tracked_object.center], dtype=np.float64) for tracked_object in tracked_objects]
        speeds: List[npt.NDArray[np.float64]] = [np.array(tracked_object.velocity.magnitude(), dtype=np.float64) if isinstance(tracked_object, Agent) else 0 for tracked_object in tracked_objects]
        boxes: List[OrientedBox] = [tracked_object.box for tracked_object in tracked_objects]
        history_tracks_poses.append(np.array(poses))
        history_tracks_speed.append(np.array(speeds))
        history_tracks_boxes.append(np.array(boxes))
    return (history_tracks_poses, history_tracks_speed, history_tracks_boxes)

def compute_time_to_collision(ego_states: List[EgoState], ego_timestamps: npt.NDArray[np.int64], observations: List[Observation], timestamps_in_common_or_connected_route_objs: List[int], all_collisions: List[Collisions], timestamps_at_fault_collisions: List[int], map_api: AbstractMap, time_step_size: float, time_horizon: float, stopped_speed_threshold: float=0.005) -> npt.NDArray[np.float64]:
    """
    Computes an estimate of the minimal time to collision with other agents. Ego and agents are projected
    with constant velocity until there is a collision or the maximal time window is reached.
    :param ego_states: A list of ego states.
    :param ego_timestamps: Array of times in time_us.
    :param observations: Observations to consider collisions with ego states.
    :param timestamps_in_common_or_connected_route_objs: List of timestamps where ego is in same or connected
    lanes/lane connectors.
    :param all_collisions: List of all collisions in the history.
    :param timestamps_at_fault_collisions: List of timestamps corresponding to at-fault-collisions in the history.
    :param map_api: Map to consider.
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :param time_horizon: [s] Time horizon for collision checking.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return: The minimal TTC for each sample, inf if no collision is found within the projection horizon.
    """
    ego_velocities = extract_ego_velocity(ego_states)
    history_tracks_poses, history_tracks_speed, history_tracks_boxes = extract_tracks_info_excluding_collided_tracks(ego_states, ego_timestamps, observations, all_collisions, timestamps_in_common_or_connected_route_objs, map_api)
    time_to_collision: npt.NDArray[np.float64] = np.asarray([np.inf] * len(ego_states), dtype=np.float64)
    for timestamp_index, (timestamp, ego_state, ego_speed, tracks_poses, tracks_speed, tracks_boxes) in enumerate(zip(ego_timestamps, ego_states, ego_velocities, history_tracks_poses, history_tracks_speed, history_tracks_boxes)):
        ttc_at_index = _compute_time_to_collision_at_timestamp(timestamp, ego_state, ego_speed, tracks_poses, tracks_speed, tracks_boxes, timestamps_at_fault_collisions, time_step_size, time_horizon, stopped_speed_threshold)
        if ttc_at_index is not None:
            time_to_collision[timestamp_index] = ttc_at_index
    return time_to_collision

def extract_ego_velocity(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Extract velocity of ego pose from list of ego states
    :param ego_states: A list of ego states
    :return An array of ego pose velocity.
    """
    velocity: npt.NDArray[np.float32] = np.array([ego_state.dynamic_car_state.speed for ego_state in ego_states])
    return velocity

class TimeToCollisionStatistics(MetricBase):
    """
    Ego time to collision metric, reports the minimal time for a projected collision if agents proceed with
    zero acceleration.
    """

    def __init__(self, name: str, category: str, ego_lane_change_metric: EgoLaneChangeStatistics, no_ego_at_fault_collisions_metric: EgoAtFaultCollisionStatistics, time_step_size: float, time_horizon: float, least_min_ttc: float, metric_score_unit: Optional[str]=None):
        """
        Initializes the TimeToCollisionStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_lane_change_metric: Lane chang metric computed prior to calling the current metric
        :param no_ego_at_fault_collisions_metric: Ego at fault collisions computed prior to the current metric
        :param time_step_size: [s] Step size for the propagation of collision agents
        :param time_horizon: [s] Time horizon for collision checking
        :param least_min_ttc: minimum desired TTC.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._time_step_size = time_step_size
        self._time_horizon = time_horizon
        self._least_min_ttc = least_min_ttc
        self._ego_lane_change_metric = ego_lane_change_metric
        self._no_ego_at_fault_collisions_metric = no_ego_at_fault_collisions_metric
        self.results: List[MetricStatistics] = []

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[-1].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the time to collision statistics
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the time to collision metric
        """
        timestamps_in_common_or_connected_route_objs: List[int] = self._ego_lane_change_metric.timestamps_in_common_or_connected_route_objs
        assert self._no_ego_at_fault_collisions_metric.results, 'no_ego_at_fault_collisions metric must be run prior to calling {}'.format(self.name)
        all_collisions = self._no_ego_at_fault_collisions_metric.all_collisions
        timestamps_at_fault_collisions = self._no_ego_at_fault_collisions_metric.timestamps_at_fault_collisions
        ego_states = history.extract_ego_state
        ego_timestamps = extract_ego_time_point(ego_states)
        observations = [sample.observation for sample in history.data]
        time_to_collision = compute_time_to_collision(ego_states, ego_timestamps, observations, timestamps_in_common_or_connected_route_objs, all_collisions, timestamps_at_fault_collisions, history.map_api, self._time_step_size, self._time_horizon)
        time_to_collision_within_bounds = self._least_min_ttc < np.array(time_to_collision, dtype=np.float64)
        time_series = TimeSeries(unit='time_to_collision_under_' + f'{self._time_horizon}' + '_seconds [s]', time_stamps=list(ego_timestamps), values=list(time_to_collision))
        metric_statistics = [Statistic(name='min_time_to_collision', unit='seconds', value=np.min(time_to_collision), type=MetricStatisticsType.MIN), Statistic(name=f'{self.name}', unit=MetricStatisticsType.BOOLEAN.unit, value=bool(np.all(time_to_collision_within_bounds)), type=MetricStatisticsType.BOOLEAN)]
        self.results = self._construct_metric_results(metric_statistics=metric_statistics, time_series=time_series, scenario=scenario, metric_score_unit=self.metric_score_unit)
        return self.results

class EgoMeanSpeedStatistics(MetricBase):
    """Ego mean speed metric."""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializes the EgoMeanSpeedStatistics class
        :param name: Metric name
        :param category: Metric category.
        """
        super().__init__(name=name, category=category)

    @staticmethod
    def ego_avg_speed(history: SimulationHistory) -> Any:
        """
        Compute mean of ego speed over the scenario duration
        :param history: History from a simulation engine
        :return mean of ego speed (m/s).
        """
        ego_states = history.extract_ego_state
        ego_velocities = extract_ego_velocity(ego_states)
        mean_speed = np.mean(ego_velocities)
        return mean_speed

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the mean of ego speed over the scenario duration
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the mean of ego speed.
        """
        mean_speed = self.ego_avg_speed(history=history)
        statistics = [Statistic(name='ego_mean_speed_value', unit='meters_per_second', value=mean_speed, type=MetricStatisticsType.VALUE)]
        results: List[MetricStatistics] = self._construct_metric_results(metric_statistics=statistics, time_series=None, scenario=scenario)
        return results

def _ego_ends_lane_change(open_lane_change: LaneChangeStartRecord, final_lane: Set[GraphEdgeMapObject], end_timestamp: int) -> LaneChangeData:
    """
    Stores the information if ego ends a lane change
    :param open_lane_change: Record of the currently open lane change
    :param final_lane: Set of common/connected route objects of corners of ego when completing a lane change
    :param end_timestamp: The current timestamp
    :return LaneChangeData.
    """
    if not final_lane:
        return LaneChangeData(open_lane_change, end_timestamp - open_lane_change.start_timestamp, final_lane=None, success=False)
    initial_lane = open_lane_change.initial_lane
    initial_lane_ids = {obj.id for obj in initial_lane}
    initial_lane_out_edge_ids = set(get_outgoing_edges_obj_dict(initial_lane).keys())
    initial_lane_or_out_edge_ids = initial_lane_ids.union(initial_lane_out_edge_ids)
    final_lane_ids = {obj.id for obj in final_lane}
    return LaneChangeData(open_lane_change, end_timestamp - open_lane_change.start_timestamp, final_lane, success=False if len(set.intersection(initial_lane_or_out_edge_ids, final_lane_ids)) else True)

def find_lane_changes(ego_timestamps: npt.NDArray[np.int32], common_or_connected_route_objs: List[Optional[Set[GraphEdgeMapObject]]]) -> List[LaneChangeData]:
    """
    Extracts the lane changes in the scenario
    :param ego_timestamps: Array of times in time_us
    :param common_or_connected_route_objs: list of common or connected lane/lane connectors of corners
    :return List of lane change data in the scenario.
    """
    lane_changes: List[LaneChangeData] = []
    open_lane_change = None
    if common_or_connected_route_objs[0] is None:
        logging.debug('Scenario starts with corners in different route objects')
    for prev_ind, curr_obj in enumerate(common_or_connected_route_objs[1:]):
        if open_lane_change is None:
            if curr_obj is None:
                open_lane_change = _ego_starts_lane_change(initial_lane=common_or_connected_route_objs[prev_ind], start_timestamp=ego_timestamps[prev_ind + 1])
        elif curr_obj is not None:
            lane_change_data = _ego_ends_lane_change(open_lane_change, final_lane=curr_obj, end_timestamp=ego_timestamps[prev_ind + 1])
            lane_changes.append(lane_change_data)
            open_lane_change = None
    if open_lane_change:
        lane_changes.append(LaneChangeData(open_lane_change, ego_timestamps[-1] - open_lane_change.start_timestamp, final_lane=None, success=False))
    return lane_changes

def _ego_starts_lane_change(initial_lane: Optional[Set[GraphEdgeMapObject]], start_timestamp: int) -> Optional[LaneChangeStartRecord]:
    """
    Opens lane change window and stores the information
    :param initial_lane: Set of common/connected route objects of corners of ego at previous timestamp
    :param start_timestamp: The current timestamp
    :return information on starts of a lane change if exists, otherwise None.
    """
    return LaneChangeStartRecord(start_timestamp, initial_lane) if initial_lane else None

class EgoLaneChangeStatistics(MetricBase):
    """Statistics on lane change."""

    def __init__(self, name: str, category: str, max_fail_rate: float) -> None:
        """
        Initializes the EgoLaneChangeStatistics class
        :param name: Metric name
        :param category: Metric category
        :param max_fail_rate: maximum acceptable ratio of failed to total number of lane changes.
        """
        super().__init__(name=name, category=category)
        self._max_fail_rate = max_fail_rate
        self.ego_driven_route: List[List[Optional[GraphEdgeMapObject]]] = []
        self.corners_route: List[CornersGraphEdgeMapObject] = [CornersGraphEdgeMapObject([], [], [], [])]
        self.timestamps_in_common_or_connected_route_objs: List[int] = []
        self.results: List[MetricStatistics] = []

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the lane chane metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated lane change duration in micro seconds and status.
        """
        ego_states = history.extract_ego_state
        ego_poses = extract_ego_center(ego_states)
        self.ego_driven_route = get_route(history.map_api, ego_poses)
        ego_timestamps = extract_ego_time_point(ego_states)
        ego_footprint_list = [ego_state.car_footprint for ego_state in ego_states]
        corners_route = extract_corners_route(history.map_api, ego_footprint_list)
        self.corners_route = corners_route
        common_or_connected_route_objs = get_common_or_connected_route_objs_of_corners(corners_route)
        timestamps_in_common_or_connected_route_objs = get_timestamps_in_common_or_connected_route_objs(common_or_connected_route_objs, ego_timestamps)
        self.timestamps_in_common_or_connected_route_objs = timestamps_in_common_or_connected_route_objs
        lane_changes = find_lane_changes(ego_timestamps, common_or_connected_route_objs)
        if len(lane_changes) == 0:
            metric_statistics = [Statistic(name=f'number_of_{self.name}', unit=MetricStatisticsType.COUNT.unit, value=0, type=MetricStatisticsType.COUNT), Statistic(name=f'{self.name}_fail_rate_below_threshold', unit=MetricStatisticsType.BOOLEAN.unit, value=True, type=MetricStatisticsType.BOOLEAN)]
        else:
            lane_change_durations = [lane_change.duration_us * 1e-06 for lane_change in lane_changes]
            failed_lane_changes = [lane_change for lane_change in lane_changes if not lane_change.success]
            failed_ratio = len(failed_lane_changes) / len(lane_changes)
            fail_rate_below_threshold = 1 if self._max_fail_rate >= failed_ratio else 0
            metric_statistics = [Statistic(name=f'number_of_{self.name}', unit=MetricStatisticsType.COUNT.unit, value=len(lane_changes), type=MetricStatisticsType.COUNT), Statistic(name=f'max_{self.name}_duration', unit='seconds', value=np.max(lane_change_durations), type=MetricStatisticsType.MAX), Statistic(name=f'avg_{self.name}_duration', unit='seconds', value=float(np.mean(lane_change_durations)), type=MetricStatisticsType.MEAN), Statistic(name=f'ratio_of_failed_{self.name}', unit=MetricStatisticsType.RATIO.unit, value=failed_ratio, type=MetricStatisticsType.RATIO), Statistic(name=f'{self.name}_fail_rate_below_threshold', unit=MetricStatisticsType.BOOLEAN.unit, value=bool(fail_rate_below_threshold), type=MetricStatisticsType.BOOLEAN)]
        results: List[MetricStatistics] = self._construct_metric_results(metric_statistics=metric_statistics, time_series=None, scenario=scenario)
        self.results = results
        return results

def get_timestamps_in_common_or_connected_route_objs(common_or_connected_route_objs: List[Optional[Set[GraphEdgeMapObject]]], ego_timestamps: npt.NDArray[np.int32]) -> List[int]:
    """
    Extract timestamps when ego's corners are in common or connected lane/lane connectors.
    :param common_or_connected_route_objs: list of common or connected lane/lane connectors of corners if exist,
    empty list if all corners are in non_drivable area and None if corners are in different lane/lane connectors
    :param ego_timestamps: Array of times in time_us
    :return List of ego_timestamps where all corners of ego are in common or connected route objects
    """
    return [timestamp for route_obj, timestamp in zip(common_or_connected_route_objs, ego_timestamps) if route_obj]

class EgoIsMakingProgressStatistics(MetricBase):
    """
    Check if ego trajectory is making progress along expert route more than a minimum required progress.
    """

    def __init__(self, name: str, category: str, ego_progress_along_expert_route_metric: EgoProgressAlongExpertRouteStatistics, min_progress_threshold: float, metric_score_unit: Optional[str]=None) -> None:
        """
        Initializes the EgoIsMakingProgressStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_progress_along_expert_route_metric: Ego progress along expert route metric
        :param min_progress_threshold: minimimum required progress threshold
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._min_progress_threshold = min_progress_threshold
        self._ego_progress_along_expert_route_metric = ego_progress_along_expert_route_metric

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the ego_is_making_progress metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        ego_is_making_progress = self._ego_progress_along_expert_route_metric.results[0].statistics[-1].value >= self._min_progress_threshold
        statistics = [Statistic(name='ego_is_making_progress', unit='boolean', value=ego_is_making_progress, type=MetricStatisticsType.BOOLEAN)]
        results = self._construct_metric_results(metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit)
        return results

class DrivingDirectionComplianceStatistics(MetricBase):
    """Driving direction compliance metric.
    This metric traces if ego has been driving against the traffic flow more than some threshold during some time interval of ineterst.
    """

    def __init__(self, name: str, category: str, lane_change_metric: EgoLaneChangeStatistics, driving_direction_compliance_threshold: float=2, driving_direction_violation_threshold: float=6, time_horizon: float=1, metric_score_unit: Optional[str]=None) -> None:
        """
        Initialize the DrivingDirectionComplianceStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param lane_change_metric: Lane change metric.
        :param driving_direction_compliance_threshold: Driving in opposite direction up to this threshold isn't considered violation
        :param driving_direction_violation_threshold: Driving in opposite direction above this threshold isn't tolerated
        :param time_horizon: Movement of the vehicle along baseline direction during a horizon time_horizon is
        considered for evaluation.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._lane_change_metric = lane_change_metric
        self._driving_direction_compliance_threshold = driving_direction_compliance_threshold
        self._driving_direction_violation_threshold = driving_direction_violation_threshold
        self._time_horizon = time_horizon

    @staticmethod
    def _extract_metric(ego_poses: List[Point2D], ego_driven_route: List[List[GraphEdgeMapObject]], n_horizon: int) -> List[float]:
        """Compute the movement of ego during the past n_horizon samples along the direction of baselines.
        :param ego_poses: List of  ego poses.
        :param ego_driven_route: List of lanes/lane_connectors ego belongs to.
        :param n_horizon: Number of samples to sum the movement over.
        :return: A list of floats including ego's overall movements in the past n_horizon samples.
        """
        progress_along_baseline = []
        distance_to_start = None
        prev_distance_to_start = None
        prev_route_obj_id = None
        if ego_driven_route[0]:
            prev_route_obj_id = ego_driven_route[0][0].id
        for ego_pose, ego_route_object in zip(ego_poses, ego_driven_route):
            if not ego_route_object:
                progress_along_baseline.append(0.0)
                continue
            if prev_route_obj_id and ego_route_object[0].id == prev_route_obj_id:
                distance_to_start = get_distance_of_closest_baseline_point_to_its_start(ego_route_object[0].baseline_path, ego_pose)
                progress_made = distance_to_start - prev_distance_to_start if prev_distance_to_start is not None and distance_to_start else 0.0
                progress_along_baseline.append(progress_made)
                prev_distance_to_start = distance_to_start
            else:
                distance_to_start = None
                prev_distance_to_start = None
                progress_along_baseline.append(0.0)
                prev_route_obj_id = ego_route_object[0].id
        progress_over_n_horizon = [sum(progress_along_baseline[max(0, ind - n_horizon):ind + 1]) for ind, _ in enumerate(progress_along_baseline)]
        return progress_over_n_horizon

    def compute_score(self, scenario: AbstractScenario, metric_statistics: List[Statistic], time_series: Optional[TimeSeries]=None) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the driving direction compliance metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return: driving direction compliance statistics.
        """
        ego_states = history.extract_ego_state
        ego_poses = extract_ego_center(ego_states)
        ego_driven_route = self._lane_change_metric.ego_driven_route
        ego_timestamps = extract_ego_time_point(ego_states)
        n_horizon = int(self._time_horizon * 1000000.0 / np.mean(np.diff(ego_timestamps)))
        progress_over_interval = self._extract_metric(ego_poses, ego_driven_route, n_horizon)
        max_negative_progress_over_interval = abs(min(progress_over_interval))
        if max_negative_progress_over_interval < self._driving_direction_compliance_threshold:
            driving_direction_score = 1.0
        elif max_negative_progress_over_interval < self._driving_direction_violation_threshold:
            driving_direction_score = 0.5
        else:
            driving_direction_score = 0.0
        time_series = TimeSeries(unit='progress_along_driving_direction_in_last_' + f'{self._time_horizon}' + '_seconds_[m]', time_stamps=list(ego_timestamps), values=list(progress_over_interval))
        statistics = [Statistic(name=f'{self.name}' + '_score', unit='value', value=float(driving_direction_score), type=MetricStatisticsType.VALUE), Statistic(name='min_progress_along_driving_direction_in_' + f'{self._time_horizon}' + '_second_interval', unit='meters', value=float(-max_negative_progress_over_interval), type=MetricStatisticsType.MIN)]
        self.results: List[MetricStatistics] = self._construct_metric_results(metric_statistics=statistics, scenario=scenario, time_series=time_series, metric_score_unit=self.metric_score_unit)
        return self.results

class EgoStopAtStopLineStatistics(ViolationMetricBase):
    """
    Ego stopped at stop line metric.
    """

    def __init__(self, name: str, category: str, max_violation_threshold: int, distance_threshold: float, velocity_threshold: float) -> None:
        """
        Initializes the EgoProgressAlongExpertRouteStatistics class
        Rule formulation: 1. Get the nearest stop polygon (less than the distance threshold).
                          2. Check if the stop polygon is in any lanes.
                          3. Check if front corners of ego cross the stop polygon.
                          4. Check if no any leading agents.
                          5. Get min_velocity(distance_stop_line) until the ego leaves the stop polygon.
        :param name: Metric name
        :param category: Metric category
        :param max_violation_threshold: Maximum threshold for the violation when computing the score
        :param distance_threshold: Distances between ego front side and stop line lower than this threshold
        assumed to be the first vehicle before the stop line
        :param velocity_threshold: Velocity threshold to consider an ego stopped.
        """
        super().__init__(name=name, category=category, max_violation_threshold=max_violation_threshold)
        self._distance_threshold = distance_threshold
        self._velocity_threshold = velocity_threshold
        self._stopping_velocity_data: List[VelocityData] = []
        self._previous_stop_polygon_fid: Optional[str] = None

    @staticmethod
    def get_nearest_stop_line(map_api: AbstractMap, ego_pose_front: LineString) -> Optional[Tuple[str, Polygon]]:
        """
        Retrieve the nearest stop polygon
        :param map_api: AbstractMap map api
        :param ego_pose_front: Ego pose front corner line
        :return Nearest stop polygon fid if distance is less than the threshold.
        """
        center_x, center_y = ego_pose_front.centroid.xy
        center = Point2D(center_x[0], center_y[0])
        if not map_api.is_in_layer(center, layer=SemanticMapLayer.LANE):
            return None
        stop_line_fid, distance = map_api.get_distance_to_nearest_map_object(center, SemanticMapLayer.STOP_LINE)
        if stop_line_fid is None:
            return None
        stop_line: StopLine = map_api.get_map_object(stop_line_fid, SemanticMapLayer.STOP_LINE)
        lane: Optional[Lane] = map_api.get_one_map_object(center, SemanticMapLayer.LANE)
        if lane is not None:
            return (stop_line_fid, stop_line.polygon if stop_line.polygon.intersects(lane.polygon) else None)
        return None

    @staticmethod
    def check_for_leading_agents(detections: Observation, ego_state: EgoState, map_api: AbstractMap) -> bool:
        """
        Get the nearest leading agent
        :param detections: Detection class
        :param ego_state: Ego in oriented box representation
        :param map_api: AbstractMap api
        :return True if there is a leading agent, False otherwise
        """
        if isinstance(detections, DetectionsTracks):
            if len(detections.tracked_objects.tracked_objects) == 0:
                return False
            ego_agent = ego_state.agent
            for index, box in enumerate(detections.tracked_objects):
                if box.token is None:
                    box.token = str(index + 1)
            scene_objects: List[SceneObject] = [ego_agent]
            scene_objects.extend([scene_object for scene_object in detections.tracked_objects])
            occupancy_map = STRTreeOccupancyMapFactory.get_from_boxes(scene_objects)
            agent_states = {scene_object.token: StateSE2(x=scene_object.center.x, y=scene_object.center.y, heading=scene_object.center.heading) for scene_object in scene_objects}
            ego_pose: StateSE2 = agent_states['ego']
            lane = map_api.get_one_map_object(ego_pose, SemanticMapLayer.LANE)
            ego_baseline = lane.baseline_path
            ego_progress = ego_baseline.get_nearest_arc_length_from_position(ego_pose)
            progress_path = create_path_from_se2(ego_baseline.discrete_path)
            ego_path_to_go = trim_path_up_to_progress(progress_path, ego_progress)
            ego_path_to_go = path_to_linestring(ego_path_to_go)
            intersecting_agents = occupancy_map.intersects(ego_path_to_go.buffer(scene_objects[0].box.width / 2, cap_style=CAP_STYLE.flat))
            if intersecting_agents.size > 1:
                return True
        return False

    def _compute_velocity_statistics(self, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Compute statistics in each stop line
        :param scenario: Scenario running this metric
        :return A list of metric statistics.
        """
        if not self._stopping_velocity_data:
            return []
        mean_ego_min_distance_to_stop_line = []
        mean_ego_min_velocity_before_stop_line = []
        aggregated_timestamp_velocity = []
        aggregated_timestamps = []
        ego_stop_status = []
        for velocity_data in self._stopping_velocity_data:
            min_distance_velocity_record = velocity_data.min_distance_stop_line_record
            mean_ego_min_distance_to_stop_line.append(min_distance_velocity_record.distance_to_stop_line)
            mean_ego_min_velocity_before_stop_line.append(min_distance_velocity_record.velocity)
            if min_distance_velocity_record.distance_to_stop_line < self._distance_threshold and min_distance_velocity_record.velocity < self._velocity_threshold:
                stop_status = True
            else:
                stop_status = False
            ego_stop_status.append(stop_status)
            aggregated_timestamp_velocity.append(velocity_data.velocity_np)
            aggregated_timestamps.append(velocity_data.timestamp_np)
        statistics = [Statistic(name='number_of_ego_stop_before_stop_line', unit=MetricStatisticsType.COUNT.unit, value=sum(ego_stop_status), type=MetricStatisticsType.COUNT), Statistic(name='number_of_ego_before_stop_line', unit=MetricStatisticsType.COUNT.unit, value=len(ego_stop_status), type=MetricStatisticsType.COUNT), Statistic(name='mean_ego_min_distance_to_stop_line', unit='meters', value=float(np.mean(mean_ego_min_distance_to_stop_line)), type=MetricStatisticsType.VALUE), Statistic(name='mean_ego_min_velocity_before_stop_line', unit='meters_per_second_squared', value=float(np.mean(mean_ego_min_velocity_before_stop_line)), type=MetricStatisticsType.VALUE)]
        aggregated_timestamp_velocity = np.hstack(aggregated_timestamp_velocity)
        aggregated_timestamps = np.hstack(aggregated_timestamps)
        velocity_time_series = TimeSeries(unit='meters_per_second_squared', time_stamps=list(aggregated_timestamps), values=list(aggregated_timestamp_velocity))
        results = self._construct_metric_results(metric_statistics=statistics, time_series=velocity_time_series, scenario=scenario)
        return results

    def _save_stopping_velocity(self, current_stop_polygon_fid: str, history_data: SimulationHistorySample, stop_polygon_in_lane: Polygon, ego_pose_front: LineString) -> None:
        """
        Save velocity, timestamp and distance to a stop line if the ego is stopping
        :param current_stop_polygon_fid: Current stop polygon fid
        :param history_data: History sample data at current timestamp
        :param stop_polygon_in_lane: The stop polygon where the ego is in
        :param ego_pose_front: Front line string (front right corner and left corner) of the ego.
        """
        stop_line: LineString = LineString(stop_polygon_in_lane.exterior.coords[:2])
        distance_ego_front_stop_line = stop_line.distance(ego_pose_front)
        current_velocity = history_data.ego_state.dynamic_car_state.speed
        current_timestamp = history_data.ego_state.time_point.time_us
        if current_stop_polygon_fid == self._previous_stop_polygon_fid:
            self._stopping_velocity_data[-1].add_data(velocity=current_velocity, timestamp=current_timestamp, distance_to_stop_line=distance_ego_front_stop_line)
        else:
            self._previous_stop_polygon_fid = current_stop_polygon_fid
            velocity_data = VelocityData([])
            velocity_data.add_data(velocity=current_velocity, timestamp=current_timestamp, distance_to_stop_line=distance_ego_front_stop_line)
            self._stopping_velocity_data.append(velocity_data)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the ego stopped at stop line metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated ego stopped at stop line metric.
        """
        ego_states: List[EgoState] = history.extract_ego_state
        ego_pose_fronts: List[LineString] = [LineString([state.car_footprint.oriented_box.geometry.exterior.coords[0], state.car_footprint.oriented_box.geometry.exterior.coords[3]]) for state in ego_states]
        scenario_map: AbstractMap = history.map_api
        for ego_pose_front, ego_state, history_data in zip(ego_pose_fronts, ego_states, history.data):
            stop_polygon_info: Optional[Tuple[str, Polygon]] = self.get_nearest_stop_line(map_api=scenario_map, ego_pose_front=ego_pose_front)
            if stop_polygon_info is None:
                continue
            fid, stop_polygon_in_lane = stop_polygon_info
            ego_pose_front_stop_polygon_distance: float = ego_pose_front.distance(stop_polygon_in_lane)
            if ego_pose_front_stop_polygon_distance != 0:
                continue
            detections: Observation = history_data.observation
            has_leading_agent = self.check_for_leading_agents(detections=detections, ego_state=ego_state, map_api=scenario_map)
            if has_leading_agent:
                continue
            self._save_stopping_velocity(current_stop_polygon_fid=fid, history_data=history_data, stop_polygon_in_lane=stop_polygon_in_lane, ego_pose_front=ego_pose_front)
        results = self._compute_velocity_statistics(scenario=scenario)
        return results

