# Cluster 163

class PerFrameProgressAlongRouteComputer:
    """Class that computes progress per frame along a route."""

    def __init__(self, route_roadblocks: RouteRoadBlockLinkedList):
        """Class initializer
        :param route_roadblocks: A route roadblock linked list.
        """
        self.curr_roadblock_pair = route_roadblocks.head
        self.progress = [float(0)]
        self.prev_distance_to_start = float(0)
        self.next_roadblock_pair: Optional[RouteBaselineRoadBlockPair] = None
        self.skipped_roadblock_pair: Optional[RouteBaselineRoadBlockPair] = None

    @staticmethod
    def get_some_baseline_point(baseline: PolylineMapObject, ind: str) -> Optional[Point2D]:
        """Gets the first or last point on a given baselinePath
        :param baseline: A baseline path
        :param ind: Either 'last' or 'first' strings to show which point function should return
        :return: A point.
        """
        if ind == 'last':
            return Point2D(baseline.linestring.xy[0][-1], baseline.linestring.xy[1][-1])
        elif ind == 'first':
            return Point2D(baseline.linestring.xy[0][0], baseline.linestring.xy[1][0])
        else:
            raise ValueError('invalid position argument')

    def compute_progress_for_skipped_road_block(self) -> float:
        """Computes progress for skipped road_blocks (when ego pose exits one road block in a route and it does not
        enter the next one)
        :return: progress_for_skipped_roadblock
        """
        assert self.next_roadblock_pair is not None
        if self.skipped_roadblock_pair:
            prev_roadblock_last_point = self.get_some_baseline_point(self.skipped_roadblock_pair.base_line, 'last')
        else:
            prev_roadblock_last_point = self.get_some_baseline_point(self.curr_roadblock_pair.base_line, 'last')
        self.skipped_roadblock_pair = self.next_roadblock_pair
        skipped_distance_to_start = get_distance_of_closest_baseline_point_to_its_start(self.skipped_roadblock_pair.base_line, prev_roadblock_last_point)
        self.next_roadblock_pair = self.next_roadblock_pair.next
        next_roadblock_first_point = self.get_some_baseline_point(self.next_roadblock_pair.base_line, 'first')
        next_baseline_start_dist_to_skipped = get_distance_of_closest_baseline_point_to_its_start(self.skipped_roadblock_pair.base_line, next_roadblock_first_point)
        progress_for_skipped_roadblock: float = next_baseline_start_dist_to_skipped - skipped_distance_to_start
        return progress_for_skipped_roadblock

    def get_progress_including_skipped_roadblocks(self, ego_pose: Point2D, progress_for_skipped_roadblock: float) -> float:
        """Computes ego's progress when it first enters a new road-block in the route by considering possible progress
        for roadblocks it has skipped as multi_block_progress = (progress along the baseline of prev ego roadblock)
        + (progress along the baseline of the roadblock ego is in now) + (progress along skipped roadblocks if any).
        :param ego_pose: ego pose
        :param progress_for_skipped_roadblock: Prgoress for skipped road_blocks (zero if no roadblocks is skipped)
        :return: multi_block_progress
        """
        assert self.next_roadblock_pair is not None
        progress_in_prev_roadblock = self.curr_roadblock_pair.base_line.linestring.length - self.prev_distance_to_start
        prev_roadblock_last_point = self.get_some_baseline_point(self.curr_roadblock_pair.base_line, 'last')
        self.curr_roadblock_pair = self.next_roadblock_pair
        distance_to_start = get_distance_of_closest_baseline_point_to_its_start(self.curr_roadblock_pair.base_line, ego_pose)
        last_baseline_point_dist_to_start = get_distance_of_closest_baseline_point_to_its_start(self.curr_roadblock_pair.base_line, prev_roadblock_last_point)
        progress_in_new_roadblock = distance_to_start - last_baseline_point_dist_to_start
        multi_block_progress = progress_in_prev_roadblock + progress_in_new_roadblock + progress_for_skipped_roadblock
        self.prev_distance_to_start = distance_to_start
        return float(multi_block_progress)

    def get_multi_block_progress(self, ego_pose: Point2D) -> float:
        """When ego pose exits previous roadblock this function takes next road blocks in the expert route one by one
        until it finds one (if any) that pose belongs to. Once found, ego progress for multiple roadblocks including
        possible skipped roadblocks is computed and returned
        :param ego_pose: ego pose
        :return: multi block progress
        """
        multi_block_progress = float(0)
        progress_for_skipped_roadblocks = float(0)
        self.next_roadblock_pair = self.curr_roadblock_pair.next
        self.skipped_roadblock_pair = None
        while self.next_roadblock_pair is not None:
            if self.next_roadblock_pair.road_block.contains_point(ego_pose):
                multi_block_progress = self.get_progress_including_skipped_roadblocks(ego_pose, progress_for_skipped_roadblocks)
                break
            elif not self.next_roadblock_pair.next:
                break
            else:
                progress_for_skipped_roadblocks += self.compute_progress_for_skipped_road_block()
        return multi_block_progress

    def __call__(self, ego_poses: List[Point2D]) -> List[float]:
        """
        Computes per frame progress along the route baselines for ego poses
        :param ego_poses: ego poses
        :return: progress along the route.
        """
        self.prev_distance_to_start = get_distance_of_closest_baseline_point_to_its_start(self.curr_roadblock_pair.base_line, ego_poses[0])
        for ego_pose in ego_poses[1:]:
            if self.curr_roadblock_pair.road_block.contains_point(ego_pose):
                distance_to_start = get_distance_of_closest_baseline_point_to_its_start(self.curr_roadblock_pair.base_line, ego_pose)
                self.progress.append(distance_to_start - self.prev_distance_to_start)
                self.prev_distance_to_start = distance_to_start
            else:
                multi_block_progress = self.get_multi_block_progress(ego_pose)
                self.progress.append(multi_block_progress)
        return self.progress

def get_distance_of_closest_baseline_point_to_its_start(base_line: PolylineMapObject, pose: Point2D) -> float:
    """Computes distance of "closest point on the baseline to pose" to the beginning of the baseline
    :param base_line: A baseline path
    :param pose: An ego pose
    :return: distance to start.
    """
    return float(base_line.linestring.project(Point(*pose)))

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

