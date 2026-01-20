# Cluster 161

class TestViolationMetricBase(unittest.TestCase):
    """Creates mock violations for testing."""

    def setUp(self) -> None:
        """Set up mock violations."""
        self.violation_metric_base = ViolationMetricBase(name='metric_1', category='Dynamics', max_violation_threshold=1)
        self.mock_abstract_scenario = MockAbstractScenario()
        self.violation_metric_1 = [self._create_mock_violation('metric_1', duration=3, extremum=12.23, mean=8.9), self._create_mock_violation('metric_1', duration=1, extremum=123.23, mean=111.1), self._create_mock_violation('metric_1', duration=10, extremum=12.23, mean=4.92)]
        self.violation_metric_2 = [self._create_mock_violation('metric_2', duration=13, extremum=1.2, mean=0.0)]

    def _create_mock_violation(self, metric_name: str, duration: int, extremum: float, mean: float) -> MetricViolation:
        """Creates a simple violation
        :param metric_name: name of the metric
        :param duration: duration of the violation
        :param extremum: maximally violating value
        :param mean: mean value of violation depth
        :return: a MetricViolation with the given parameters.
        """
        return MetricViolation(metric_computator=self.violation_metric_base.name, name=metric_name, metric_category=self.violation_metric_base.category, unit='unit', start_timestamp=0, duration=duration, extremum=extremum, mean=mean)

    def test_successful_aggregation(self) -> None:
        """Checks that the aggregation of MetricViolations works as intended."""
        aggregated_metrics = self.violation_metric_base.aggregate_metric_violations(metric_violations=self.violation_metric_1, scenario=self.mock_abstract_scenario)[0]
        self.assertEqual(aggregated_metrics.metric_computator, self.violation_metric_base.name)
        self.assertEqual(aggregated_metrics.metric_category, self.violation_metric_base.category)
        statistics = aggregated_metrics.statistics
        self.assertEqual(len(self.violation_metric_1), statistics[0].value)
        self.assertAlmostEqual(statistics[1].value, 123.23, 2)
        self.assertAlmostEqual(statistics[2].value, 12.23, 3)
        self.assertAlmostEqual(statistics[3].value, 13.357, 3)

    def test_failure_on_mixed_metrics(self) -> None:
        """Checks that the aggregation fails when called on MetricViolations from different metrics."""
        with self.assertRaises(AssertionError):
            self.violation_metric_base.aggregate_metric_violations(self.violation_metric_1 + self.violation_metric_2, scenario=self.mock_abstract_scenario)

    def test_empty_statistics_on_empty_violations(self) -> None:
        """Checks that for an empty list of MetricViolations we get a MetricStatistics with zero violations."""
        empty_statistics = self.violation_metric_base.aggregate_metric_violations([], self.mock_abstract_scenario)[0]
        self.assertTrue(empty_statistics.statistics[0].value)

class SpeedLimitViolationExtractor:
    """Class to extract speed limit violations."""

    def __init__(self, history: SimulationHistory, metric_name: str, category: str) -> None:
        """
        Initializes the SpeedLimitViolationExtractor class
        :param history: History from a simulation engine
        :param metric_name: Metric name
        :param category: Metric category.
        """
        self.history = history
        self.open_violation: Optional[GenericViolation] = None
        self.violations: List[MetricViolation] = []
        self.violation_depths: List[float] = []
        self.metric_name = metric_name
        self.category = category

    def extract_metric(self, ego_route: List[List[GraphEdgeMapObject]]) -> None:
        """Extracts the drivable area violations from the history of Ego poses."""
        timestamp = None
        for sample, curr_ego_route in zip(self.history.data, ego_route):
            ego_state = sample.ego_state
            timestamp = ego_state.time_point.time_us
            if not curr_ego_route:
                violation = None
            else:
                violation = self._get_speed_limit_violation(ego_state, timestamp, curr_ego_route)
            if violation:
                if not self.open_violation:
                    self.start_violation(violation)
                else:
                    self.update_violation(violation)
                self.violation_depths.append(violation.violation_depths[0])
            else:
                self.violation_depths.append(0)
                if self.open_violation:
                    self.end_violation(timestamp, higher_is_worse=True)
        if timestamp and self.open_violation:
            self.end_violation(timestamp)

    def start_violation(self, violation: GenericViolation) -> None:
        """
        Opens the violation window of the given IDs, as they now starting to violate the metric
        :param violation: The current violation.
        """
        self.open_violation = violation

    def update_violation(self, violation: GenericViolation) -> None:
        """
        Updates the violation if the maximum depth of violation is greater than the current maximum
        :param violation: The current violation.
        """
        assert isinstance(self.open_violation, GenericViolation), 'There is no open violation, cannot update it!'
        self.open_violation.violation_depths.extend(violation.violation_depths)

    def end_violation(self, timestamp: int, higher_is_worse: bool=True) -> None:
        """
        Closes the violation window, as Ego re-enters the non-violating regime
        :param timestamp: The current timestamp
        :param higher_is_worse: True if the violation gravity is monotonic increasing with violation depth.
        """
        assert isinstance(self.open_violation, GenericViolation), 'There is no open violation, cannot end it!'
        maximal_violation = max(self.open_violation.violation_depths) if higher_is_worse else min(self.open_violation.violation_depths)
        self.violations.append(MetricViolation(name='speed_limit_violation', metric_computator=self.metric_name, metric_category=self.category, unit='meters_per_second', start_timestamp=self.open_violation.timestamp, duration=timestamp - self.open_violation.timestamp, extremum=maximal_violation, mean=statistics.mean(self.open_violation.violation_depths)))
        self.open_violation = None

    @staticmethod
    def _get_speed_limit_violation(ego_state: EgoState, timestamp: int, ego_lane_or_laneconnector: List[GraphEdgeMapObject]) -> Optional[GenericViolation]:
        """
        Computes by how much ego is exceeding the speed limit
        :param ego_state: The current state of Ego
        :param timestamp: The current timestamp
        :return: By how much ego is exceeding the speed limit, none if not violation is present or unable to find
        the speed limit.
        """
        if isinstance(ego_lane_or_laneconnector[0], Lane):
            assert len(ego_lane_or_laneconnector) == 1, 'Ego should can assigned to one lane only'
            speed_limits = [ego_lane_or_laneconnector[0].speed_limit_mps]
        else:
            speed_limits = []
            for map_obj in ego_lane_or_laneconnector:
                edges = map_obj.outgoing_edges + map_obj.incoming_edges
                speed_limits.extend([lane.speed_limit_mps for lane in edges])
        if all(speed_limits):
            max_speed_limit = max(speed_limits)
            exceeding_speed = ego_state.dynamic_car_state.speed - max_speed_limit
            return GenericViolation(timestamp, violation_depths=[exceeding_speed]) if exceeding_speed > 0 else None
        return None

