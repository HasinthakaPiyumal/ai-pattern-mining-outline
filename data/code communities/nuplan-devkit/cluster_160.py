# Cluster 160

def build_mock_history_scenario_test(scene: Dict[str, Any]) -> Tuple[SimulationHistory, MockAbstractScenario]:
    """
    A common template to create a test history and scenario.
    :param scene: A json format to represent a scene.
    :return The mock history and scenario.
    """
    goal_pose = None
    if 'goal' in scene and 'pose' in scene['goal'] and scene['goal']['pose']:
        goal_pose = StateSE2(x=scene['goal']['pose'][0], y=scene['goal']['pose'][1], heading=scene['goal']['pose'][2])
    if 'ego' in scene and 'time_us' in scene['ego'] and ('ego_future_states' in scene) and scene['ego_future_states'] and ('time_us' in scene['ego_future_states'][0]):
        initial_time_us = TimePoint(time_us=scene['ego']['time_us'])
        time_step = (scene['ego_future_states'][0]['time_us'] - scene['ego']['time_us']) * 1e-06
        mock_abstract_scenario = MockAbstractScenario(initial_time_us=initial_time_us, time_step=time_step)
    else:
        mock_abstract_scenario = MockAbstractScenario()
    if goal_pose is not None:
        mock_abstract_scenario.get_mission_goal = lambda: goal_pose
    history = setup_history(scene, mock_abstract_scenario)
    return (history, mock_abstract_scenario)

def metric_statistic_test(scene: Dict[str, Any], metric: AbstractMetricBuilder, history: Optional[SimulationHistory]=None, mock_abstract_scenario: Optional[MockAbstractScenario]=None) -> MetricStatistics:
    """
    A common template to test metric statistics.
    :param scene: A json format to represent a scene.
    :param metric: An evaluation metric.
    :param history: A SimulationHistory history.
    :param mock_abstract_scenario: A scenario.
    :return Metric statistics.
    """
    if not history or not mock_abstract_scenario:
        history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    metric_results = metric.compute(history, mock_abstract_scenario)
    expected_statistics_list = scene['expected']
    if not isinstance(expected_statistics_list, list):
        expected_statistics_list = [expected_statistics_list]
    for ind, metric_result in enumerate(metric_results):
        statistics = metric_result.statistics
        expected_statistic = expected_statistics_list[ind]['statistics']
        assert len(expected_statistic) == len(statistics), f'Length of actual ({len(statistics)}) and expected ({len(expected_statistic)}) statistics must be same!'
        for expected_statistic, statistic in zip(expected_statistic, statistics):
            expected_type, expected_value = expected_statistic
            assert expected_type == str(statistic.type), f"Statistic types don't match. Actual: {statistic.type}, Expected: {expected_type}"
            assert np.isclose(expected_value, statistic.value, atol=0.01), f"Statistic values don't match. Actual: {statistic.value}, Expected: {expected_value}"
        expected_time_series = expected_statistics_list[ind].get('time_series', None)
        if expected_time_series and metric_result.time_series is not None:
            time_series = metric_result.time_series
            expected_time_series = expected_statistics_list[ind]['time_series']
            assert isinstance(time_series, TimeSeries), 'Time series type not correct.'
            assert time_series.time_stamps == expected_time_series['time_stamps'], 'Time stamps are not correct.'
            assert np.all(np.round(time_series.values, 2) == expected_time_series['values']), 'Time stamp values are not correct.'
    return metric_result

class TestMetricEngine(unittest.TestCase):
    """Run metric_engine unit tests."""

    def setUp(self) -> None:
        """Set up a metric engine."""
        goal = StateSE2(x=664430.1930625531, y=3997650.6249544094, heading=0)
        self.scenario = MockAbstractScenario(mission_goal=goal)
        self.metric_names = ['ego_acceleration', 'ego_jerk']
        ego_acceleration_metric = EgoAccelerationStatistics(name=self.metric_names[0], category='Dynamics')
        ego_jerk = EgoJerkStatistics(name=self.metric_names[1], category='Dynamics', max_abs_mag_jerk=10.0)
        self.planner_name = 'planner'
        self.metric_engine = MetricsEngine(metrics=[ego_acceleration_metric], main_save_path=Path(''))
        self.metric_engine.add_metric(ego_jerk)
        self.history = self.setup_history()

    def setup_history(self) -> SimulationHistory:
        """Set up a history."""
        history = SimulationHistory(self.scenario.map_api, self.scenario.get_mission_goal())
        scene_objects = [SceneObject.from_raw_params('1', '1', 1, 1, center=StateSE2(664436.5810496865, 3997678.37696938, -1.50403628994573), size=(1.8634377032974847, 4.555735325993202, 1.5))]
        vehicle_parameters = get_pacifica_parameters()
        ego_states = [EgoState.build_from_rear_axle(StateSE2(664430.3396621217, 3997673.373507501, -1.534863576938717), rear_axle_velocity_2d=StateVector2D(x=0.0, y=0.0), rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0), tire_steering_angle=0.0, time_point=TimePoint(1000000), vehicle_parameters=vehicle_parameters), EgoState.build_from_rear_axle(StateSE2(664431.1930625531, 3997675.3735075, -1.534863576938717), rear_axle_velocity_2d=StateVector2D(x=1.0, y=0.0), rear_axle_acceleration_2d=StateVector2D(x=0.5, y=0.0), tire_steering_angle=0.0, time_point=TimePoint(2000000), vehicle_parameters=vehicle_parameters), EgoState.build_from_rear_axle(StateSE2(664432.1930625531, 3997678.3735075, -1.534863576938717), rear_axle_velocity_2d=StateVector2D(x=0.5, y=0.0), rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0), tire_steering_angle=0.0, time_point=TimePoint(3000000), vehicle_parameters=vehicle_parameters), EgoState.build_from_rear_axle(StateSE2(664432.1930625531, 3997678.3735075, -1.534863576938717), rear_axle_velocity_2d=StateVector2D(x=0.5, y=0.0), rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0), tire_steering_angle=0.0, time_point=TimePoint(4000000), vehicle_parameters=vehicle_parameters), EgoState.build_from_rear_axle(StateSE2(664434.1930625531, 3997679.3735075, -1.534863576938717), rear_axle_velocity_2d=StateVector2D(x=0.5, y=0.0), rear_axle_acceleration_2d=StateVector2D(x=1.0, y=0.0), tire_steering_angle=0.0, time_point=TimePoint(5000000), vehicle_parameters=vehicle_parameters), EgoState.build_from_rear_axle(StateSE2(664434.1930625531, 3997679.3735075, -1.534863576938717), rear_axle_velocity_2d=StateVector2D(x=0.0, y=0.0), rear_axle_acceleration_2d=StateVector2D(x=2.0, y=0.0), tire_steering_angle=0.0, time_point=TimePoint(6000000), vehicle_parameters=vehicle_parameters)]
        simulation_iterations = [SimulationIteration(TimePoint(1000000), 0), SimulationIteration(TimePoint(2000000), 1), SimulationIteration(TimePoint(3000000), 2), SimulationIteration(TimePoint(4000000), 3), SimulationIteration(TimePoint(5000000), 4)]
        trajectories = [InterpolatedTrajectory([ego_states[0], ego_states[1]]), InterpolatedTrajectory([ego_states[1], ego_states[2]]), InterpolatedTrajectory([ego_states[2], ego_states[3]]), InterpolatedTrajectory([ego_states[3], ego_states[4]]), InterpolatedTrajectory([ego_states[4], ego_states[5]])]
        for ego_state, simulation_iteration, trajectory in zip(ego_states, simulation_iterations, trajectories):
            history.add_sample(SimulationHistorySample(iteration=simulation_iteration, ego_state=ego_state, trajectory=trajectory, observation=DetectionsTracks(TrackedObjects(scene_objects)), traffic_light_status=self.scenario.get_traffic_light_status_at_iteration(simulation_iteration.index)))
        return history

    def test_compute(self) -> None:
        """Test compute() in MetricEngine."""
        expected_values = [[0.81, 0.04, 0.3, 0.81], [0.58, -0.28, 0.15, 0.58]]
        expected_time_stamps = [1000000, 2000000, 3000000, 4000000, 5000000]
        expected_time_series_values = [[0.21, 0.04, 0.09, 0.34, 0.81], [-0.28, -0.06, 0.15, 0.36, 0.58]]
        metric_dict = self.metric_engine.compute(history=self.history, planner_name=self.planner_name, scenario=self.scenario)
        metric_files = metric_dict['mock_scenario_type_mock_scenario_name_planner']
        self.assertEqual(len(metric_files), 2)
        for index, metric_file in enumerate(metric_files):
            key = metric_file.key
            self.assertEqual(key.metric_name, self.metric_names[index])
            self.assertEqual(key.scenario_type, self.scenario.scenario_type)
            self.assertEqual(key.scenario_name, self.scenario.scenario_name)
            self.assertEqual(key.planner_name, self.planner_name)
            metric_statistics = metric_file.metric_statistics
            for statistic_result in metric_statistics:
                statistics = statistic_result.statistics
                self.assertEqual(np.round(statistics[0].value, 2), expected_values[index][0])
                self.assertEqual(np.round(statistics[1].value, 2), expected_values[index][1])
                self.assertEqual(np.round(statistics[2].value, 2), expected_values[index][2])
                self.assertEqual(np.round(statistics[3].value, 2), expected_values[index][3])
                time_series = statistic_result.time_series
                assert isinstance(time_series, TimeSeries)
                self.assertEqual(time_series.time_stamps, expected_time_stamps)
                self.assertEqual(np.round(time_series.values, 2).tolist(), expected_time_series_values[index])

@nuplan_test(path='json/ego_lon_jerk/ego_lon_jerk.json')
def test_ego_longitudinal_jerk(scene: Dict[str, Any]) -> None:
    """
    Tests ego longitudinal jerk statistics as expected.
    :param scene: the json scene
    """
    metric = EgoLonJerkStatistics('ego_lon_jerk_statistics', 'Dynamics', max_abs_lon_jerk=8.0)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/driving_direction_compliance/ego_does_not_drive_backward.json')
def test_ego_no_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when there's no route.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivingDirectionComplianceStatistics('driving_direction_compliance', 'Planning', lane_change_metric, driving_direction_compliance_threshold=2, driving_direction_violation_threshold=6, time_horizon=1)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/driving_direction_compliance/ego_drives_backward.json')
def test_ego_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when ego drives backward more than driving_direction_violation_threshold.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivingDirectionComplianceStatistics('driving_direction_compliance', 'Planning', lane_change_metric, driving_direction_compliance_threshold=2, driving_direction_violation_threshold=6, time_horizon=1)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/driving_direction_compliance/ego_slightly_drives_backward.json')
def test_ego_slightly_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when ego drives backward more than driving_direction_compliance_threshold but less than driving_direction_violation_threshold.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivingDirectionComplianceStatistics('driving_direction_compliance', 'Planning', lane_change_metric, driving_direction_compliance_threshold=2, driving_direction_violation_threshold=15, time_horizon=1)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/drivable_area_compliance/drivable_area_violation.json')
def test_violations_detected_and_reported(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric, by checking the detection and the depth of violation on a made up scenario.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivableAreaComplianceStatistics('drivable_area_compliance', 'Planning', lane_change_metric, 0.3)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/drivable_area_compliance/no_drivable_area_violation.json')
def test_works_with_no_violations(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric, by checking the detection and the depth of violation on a made up scenario.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivableAreaComplianceStatistics('drivable_area_compliance', 'Planning', lane_change_metric, 0.3)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/drivable_area_compliance/small_drivable_area_violation.json')
def test_works_with_small_violations(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric when ego's footprint overapproximation is slightly outside drivable area.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivableAreaComplianceStatistics('drivable_area_compliance', 'Planning', lane_change_metric, 0.3)
    metric_statistic_test(scene=scene, metric=metric)
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.1)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivableAreaComplianceStatistics('drivable_area_compliance', 'Planning', lane_change_metric, 0.3)
    metric.compute(history, mock_abstract_scenario)
    assert np.isclose(metric.results[0].statistics[0].value, 0, atol=0.01)

@nuplan_test(path='json/planner_expert_average_heading_error_within_bound/low_average_heading_error.json')
def test_planner_expert_average_heading_error(scene: Dict[str, Any]) -> None:
    """
    Tests planner_expert_average_heading_error is expected value.
    :param scene: the json scene.
    """
    planner_expert_average_l2_error_within_bound_metric = PlannerExpertAverageL2ErrorStatistics('planner_expert_average_l2_error', 'Planning', comparison_horizon=[3, 5, 8], comparison_frequency=1, max_average_l2_error_threshold=8)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    planner_expert_average_l2_error_within_bound_metric.compute(history, mock_abstract_scenario)
    metric = PlannerExpertAverageHeadingErrorStatistics('planner_expert_average_heading_error_within_bound', 'Planning', planner_expert_average_l2_error_within_bound_metric, max_average_heading_error_threshold=0.8)
    metric_statistic_test(scene, metric, history, mock_abstract_scenario)

def _run_time_to_collision_test(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision
    :param scene: the json scene
    """
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]
    no_ego_at_fault_collisions_metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Dynamics', ego_lane_change_metric)
    no_ego_at_fault_collisions_metric.compute(history, mock_abstract_scenario)[0]
    metric = TimeToCollisionStatistics('time_to_collision_statistics', 'Planning', ego_lane_change_metric, no_ego_at_fault_collisions_metric, **scene['metric_parameters'])
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/time_to_collision_within_bound/time_to_collision_above_threshold.json')
def test_time_to_collision_above_threshold(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision when above threshold.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)

@nuplan_test(path='json/time_to_collision_within_bound/in_collision.json')
def test_time_to_collision_in_collision(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision in case where there is a collision.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)

@nuplan_test(path='json/time_to_collision_within_bound/ego_stopped.json')
def test_time_to_collision_ego_stopped(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision when ego is stopped.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)

@nuplan_test(path='json/time_to_collision_within_bound/no_collisions.json')
def test_time_to_collision_no_collisions(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision when there are relevant tracks, but ego will not collide.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)

@nuplan_test(path='json/time_to_collision_within_bound/no_relevant_tracks.json')
def test_time_to_collision_no_relevant_tracks(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision when no relevant tracks.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)

@nuplan_test(path='json/ego_mean_speed/ego_mean_speed.json')
def test_ego_mean_speed(scene: Dict[str, Any]) -> None:
    """
    Tests ego mean speed statistics as expected.
    :param scene: the json scene
    """
    metric = EgoMeanSpeedStatistics('ego_lon_jerk_statistics', 'Dynamics')
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_acceleration/ego_acceleration.json')
def test_ego_expected_acceleration(scene: Dict[str, Any]) -> None:
    """
    Tests ego acceleration by checking if it is the expected acceleration.
    :param scene: the json scene
    """
    metric = EgoAccelerationStatistics('ego_acceleration_statistics', 'Dynamics')
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_is_comfortable/ego_is_comfortable.json')
def test_ego_is_comfortable(scene: Dict[str, Any]) -> None:
    """
    Tests ego is comfortable by checking if it is the expected comfortable.
    :param scene: the json scene
    """
    ego_jerk_metric = EgoJerkStatistics('ego_jerk', 'Dynamics', max_abs_mag_jerk=8.37)
    ego_lat_accel_metric = EgoLatAccelerationStatistics('ego_lat_accel', 'Dynamics', max_abs_lat_accel=4.89)
    ego_lon_accel_metric = EgoLonAccelerationStatistics('ego_lon_accel', 'Dynamics', min_lon_accel=-4.05, max_lon_accel=2.4)
    ego_lon_jerk_metric = EgoLonJerkStatistics('ego_lon_jerk', 'dynamic', max_abs_lon_jerk=4.13)
    ego_yaw_accel_metric = EgoYawAccelerationStatistics('ego_yaw_accel', 'dynamic', max_abs_yaw_accel=1.93)
    ego_yaw_rate_metric = EgoYawRateStatistics('ego_yaw_rate', 'dynamic', max_abs_yaw_rate=0.95)
    metric = EgoIsComfortableStatistics(name='ego_is_comfortable_statistics', category='Dynamics', ego_jerk_metric=ego_jerk_metric, ego_lat_acceleration_metric=ego_lat_accel_metric, ego_lon_acceleration_metric=ego_lon_accel_metric, ego_lon_jerk_metric=ego_lon_jerk_metric, ego_yaw_acceleration_metric=ego_yaw_accel_metric, ego_yaw_rate_metric=ego_yaw_rate_metric)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_jerk/ego_jerk.json')
def test_ego_jerk(scene: Dict[str, Any]) -> None:
    """
    Tests ego jerk statistics as expected.
    :param scene: the json scene
    """
    metric = EgoJerkStatistics('ego_jerk_statistics', 'Dynamics', max_abs_mag_jerk=7.0)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/no_ego_at_fault_collision/no_collision.json')
def test_no_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is no collision as expected.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]
    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_result = metric_statistic_test(scene=scene, metric=metric)
    statistics = metric_result.statistics
    assert statistics[1].value == 0
    assert len(metric.all_collisions) == 0

@nuplan_test(path='json/no_ego_at_fault_collision/active_front_collision.json')
def test_active_front_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one front collision in this scene.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]
    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type == CollisionType.ACTIVE_FRONT_COLLISION

@nuplan_test(path='json/no_ego_at_fault_collision/active_lateral_collision.json')
def test_active_lateral_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one lateral collision in this scene which is at fault.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]
    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

@nuplan_test(path='json/no_ego_at_fault_collision/active_rear_collision.json')
def test_active_rear_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one rear collision in this scene which is not at fault.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]
    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type == CollisionType.ACTIVE_REAR_COLLISION

@nuplan_test(path='json/no_ego_at_fault_collision/stopped_track_collision.json')
def test_stopped_track_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one collision with a stopped track in this scene which is at fault.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]
    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type == CollisionType.STOPPED_TRACK_COLLISION

@nuplan_test(path='json/no_ego_at_fault_collision/stopped_ego_collision.json')
def test_stopped_ego_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one collision when ego is stopped in this scene which is not at fault.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]
    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 1
    assert list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type == CollisionType.STOPPED_EGO_COLLISION

@nuplan_test(path='json/no_ego_at_fault_collision/multiple_collisions.json')
def test_multiple_collisions(scene: Dict[str, Any]) -> None:
    """
    Tests there are 4 tracks and 3 collisions in this scene, and there are 2 at-fault-collisions for which
    we find the violation metric.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]
    metric = EgoAtFaultCollisionStatistics('no_ego_at_fault_collisions_statistics', 'Planning', ego_lane_change_metric)
    metric_statistic_test(scene=scene, metric=metric)
    assert np.sum([len(collision.collisions_id_data) for collision in metric.all_collisions]) == 3
    assert list(metric.all_collisions[0].collisions_id_data.values())[0].collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
    assert list(metric.all_collisions[1].collisions_id_data.values())[0].collision_type == CollisionType.ACTIVE_FRONT_COLLISION
    assert list(metric.all_collisions[1].collisions_id_data.values())[1].collision_type == CollisionType.ACTIVE_FRONT_COLLISION

@nuplan_test(path='json/ego_yaw_rate/ego_yaw_rate.json')
def test_ego_yaw_rate(scene: Dict[str, Any]) -> None:
    """
    Tests ego yaw rate statistics as expected.
    :param scene: the json scene
    """
    metric = EgoYawRateStatistics('ego_yaw_rate_statistics', 'Dynamics', max_abs_yaw_rate=5.0)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_lon_acceleration/ego_lon_acceleration.json')
def test_ego_longitudinal_acceleration(scene: Dict[str, Any]) -> None:
    """
    Tests ego longitudinal acceleration statistics as expected
    :param scene: the json scene.
    """
    metric = EgoLonAccelerationStatistics('ego_lon_acceleration_statistics', 'Dynamics', min_lon_accel=0.0, max_lon_accel=10.0)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/planner_miss_rate_within_bound/high_miss_rate.json')
def test_planner_miss_rate(scene: Dict[str, Any]) -> None:
    """
    Tests planner_miss_rate is expected value.
    :param scene: the json scene.
    """
    planner_expert_average_l2_error_within_bound_metric = PlannerExpertAverageL2ErrorStatistics('planner_expert_average_l2_error_within_bound', 'Planning', comparison_horizon=[3, 5, 8], comparison_frequency=1, max_average_l2_error_threshold=8)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    planner_expert_average_l2_error_within_bound_metric.compute(history, mock_abstract_scenario)
    metric = PlannerMissRateStatistics('planner_miss_rate_within_bound_statistics', 'Planning', planner_expert_average_l2_error_within_bound_metric, max_displacement_threshold=[6, 8, 16], max_miss_rate_threshold=0.3)
    metric_statistic_test(scene, metric, history, mock_abstract_scenario)

@nuplan_test(path='json/ego_expert_l2_error_with_yaw/ego_expert_l2_error_with_yaw.json')
def test_ego_expert_l2_error_with_yaw(scene: Dict[str, Any]) -> None:
    """
    Tests ego expert l2 error with yaw is expected value.
    :param scene: the json scene
    """
    metric = EgoExpertL2ErrorWithYawStatistics('ego_expert_L2_error_with_yaw', 'Dynamics', discount_factor=1.0, heading_diff_weight=2.5)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_expert_l2_error_with_yaw/ego_expert_l2_error_with_yaw_zero.json')
def test_ego_expert_l2_error_with_yaw_zero(scene: Dict[str, Any]) -> None:
    """
    Tests ego expert l2 error with yaw is zero.
    :param scene: the json scene
    """
    metric = EgoExpertL2ErrorWithYawStatistics('ego_expert_L2_error_with_yaw', 'Dynamics', discount_factor=1.0, heading_diff_weight=2.5)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_yaw_acceleration/ego_yaw_acceleration.json')
def test_ego_yaw_acceleration(scene: Dict[str, Any]) -> None:
    """
    Tests ego yaw acceleration statistics as expected.
    :param scene: the json scene
    """
    metric = EgoYawAccelerationStatistics('ego_yaw_acceleration_statistics', 'Dynamics', max_abs_yaw_accel=3.0)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_lat_jerk/ego_lat_jerk.json')
def test_ego_lateral_jerk(scene: Dict[str, Any]) -> None:
    """
    Tests ego lateral jerk statistics as expected.
    :param scene: the json scene
    """
    metric = EgoLatJerkStatistics('ego_lat_jerk_statistics', 'Dynamics')
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/planner_expert_final_heading_error_within_bound/low_final_heading_error.json')
def test_planner_expert_final_heading_error(scene: Dict[str, Any]) -> None:
    """
    Tests planner_expert_final_heading_error is expected value.
    :param scene: the json scene.
    """
    planner_expert_average_l2_error_within_bound = PlannerExpertAverageL2ErrorStatistics('planner_expert_average_l2_error', 'Planning', comparison_horizon=[3, 5, 8], comparison_frequency=1, max_average_l2_error_threshold=8)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    planner_expert_average_l2_error_within_bound.compute(history, mock_abstract_scenario)
    metric = PlannerExpertFinalHeadingErrorStatistics('planner_expert_final_heading_error_within_bound', 'Planning', planner_expert_average_l2_error_within_bound, max_final_heading_error_threshold=0.8)
    metric_statistic_test(scene, metric, history, mock_abstract_scenario)

@nuplan_test(path='json/ego_expert_l2_error/ego_expert_l2_error.json')
def test_ego_expert_l2_error(scene: Dict[str, Any]) -> None:
    """
    Tests ego expert l2 error is expected value.
    :param scene: the json scene
    """
    metric = EgoExpertL2ErrorStatistics('ego_expert_L2_error', 'Dynamics', discount_factor=1.0)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_lane_change/ego_lane_change.json')
def test_ego_lane_change(scene: Dict[str, Any]) -> None:
    """
    Tests ego lane change statistics as expected.
    :param scene: the json scene
    """
    metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/planner_expert_average_l2_error_within_bound/high_average_l2_error.json')
def test_planner_miss_rate(scene: Dict[str, Any]) -> None:
    """
    Tests planner_expert_average_l2_error is expected value.
    :param scene: the json scene.
    """
    metric = PlannerExpertAverageL2ErrorStatistics('planner_expert_average_l2_error', 'Planning', comparison_horizon=[3, 5, 8], comparison_frequency=1, max_average_l2_error_threshold=8)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_lat_acceleration/ego_lat_acceleration.json')
def test_ego_lateral_acceleration(scene: Dict[str, Any]) -> None:
    """
    Tests ego lateral acceleration statistics as expected.
    :param scene: the json scene
    """
    metric = EgoLatAccelerationStatistics('ego_lat_acceleration_statistics', 'Dynamics', max_abs_lat_accel=10.0)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/speed_limit_compliance/speed_limit_violation.json')
def test_speed_limit_violation(scene: Dict[str, Any]) -> None:
    """
    Tests speed limit violation, by checking the detection and the depth of compliance on a made up scenario
    :param scene: the json scene.
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = SpeedLimitComplianceStatistics('speed_limit_compliance', '', lane_change_metric=lane_change_metric, max_violation_threshold=1, max_overspeed_value_threshold=2.23)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/speed_limit_compliance/no_speed_limit_violation.json')
def test_no_violations(scene: Dict[str, Any]) -> None:
    """
    Tests speed limit violation, by checking that the metric works without violations
    :param scene: the json scene.
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = SpeedLimitComplianceStatistics('speed_limit_compliance', '', lane_change_metric=lane_change_metric, max_violation_threshold=1, max_overspeed_value_threshold=2.23)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_progress_along_expert_route/ego_progress_along_expert_route.json')
def test_ego_progress_to_goal(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress along expert route statistics as expected.
    :param scene: the json scene
    """
    metric = EgoProgressAlongExpertRouteStatistics('ego_progress_along_expert_route_statistics', 'Dynamics', score_progress_threshold=2)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_progress_along_expert_route/ego_no_progress_along_expert_route.json')
def test_ego_no_progress_to_goal(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress along expert route statistics when expert isn't assigned a route at first and ego isn't making enough progress.
    :param scene: the json scene
    """
    metric = EgoProgressAlongExpertRouteStatistics('ego_progress_along_expert_route_statistics', 'Dynamics', score_progress_threshold=2)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_progress_along_expert_route/ego_no_route.json')
def test_no_route(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when there's no route.
    :param scene: the json scene
    """
    metric = EgoProgressAlongExpertRouteStatistics('ego_progress_along_expert_route_statistics', 'Dynamics', score_progress_threshold=2)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_progress_along_expert_route/ego_drives_backward.json')
def test_ego_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when ego drives backward.
    :param scene: the json scene
    """
    metric = EgoProgressAlongExpertRouteStatistics('ego_progress_along_expert_route_statistics', 'Dynamics', score_progress_threshold=2)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/ego_is_making_progress/ego_is_making_progress.json')
def test_ego_progress_to_goal(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress along expert route statistics as expected.
    :param scene: the json scene
    """
    ego_progress_along_expert_route_metric = EgoProgressAlongExpertRouteStatistics('ego_progress_along_expert_route_statistics', 'Dynamics', score_progress_threshold=0.1)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    ego_progress_along_expert_route_metric.compute(history, mock_abstract_scenario)[0]
    metric = EgoIsMakingProgressStatistics('ego_is_making_progress_statistics', 'Plannning', ego_progress_along_expert_route_metric, min_progress_threshold=0.2)
    metric_statistic_test(scene=scene, metric=metric)

@nuplan_test(path='json/planner_expert_final_l2_error_within_bound/high_final_l2_error.json')
def test_planner_expert_final_l2_error(scene: Dict[str, Any]) -> None:
    """
    Tests planner_expert_final_l2_error is expected value.
    :param scene: the json scene.
    """
    planner_expert_average_l2_error_within_bound_metric = PlannerExpertAverageL2ErrorStatistics('planner_expert_average_l2_error', 'Planning', comparison_horizon=[3, 5, 8], comparison_frequency=1, max_average_l2_error_threshold=8)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    planner_expert_average_l2_error_within_bound_metric.compute(history, mock_abstract_scenario)
    metric = PlannerExpertFinalL2ErrorStatistics('planner_expert_final_l2_error_within_bound', 'Planning', planner_expert_average_l2_error_within_bound_metric, max_final_l2_error_threshold=8)
    metric_statistic_test(scene, metric, history, mock_abstract_scenario)

