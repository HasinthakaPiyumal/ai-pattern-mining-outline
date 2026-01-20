# Cluster 165

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

