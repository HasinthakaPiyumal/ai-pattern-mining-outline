# Cluster 1

def _create_dummy_simple_planner(acceleration: List[float], horizon_seconds: float=10.0, sampling_time: float=20.0) -> SimplePlanner:
    """
    Create a dummy simple planner.
    :param acceleration: [m/s^2] constant ego acceleration, till limited by max_velocity.
    :param horizon_seconds: [s] time horizon being run.
    :param sampling_time: [s] sampling timestep.
    :return: dummy simple planner.
    """
    acceleration_np: npt.NDArray[np.float32] = np.asarray(acceleration)
    return SimplePlanner(horizon_seconds=horizon_seconds, sampling_time=sampling_time, acceleration=acceleration_np)

def _create_dummy_simulation_history_buffer(scenario: AbstractScenario, iteration: int=0, time_horizon: int=2, num_samples: int=2, buffer_size: int=2) -> SimulationHistoryBuffer:
    """
    Create dummy SimulationHistoryBuffer.
    :param scenario: Scenario.
    :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
    :param time_horizon: the desired horizon to the future.
    :param num_samples: number of entries in the future.
    :param buffer_size: size of buffer.
    :return: SimulationHistoryBuffer.
    """
    past_observation = list(scenario.get_past_tracked_objects(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples))
    past_ego_states = list(scenario.get_ego_past_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples))
    history_buffer = SimulationHistoryBuffer.initialize_from_list(buffer_size=buffer_size, ego_states=past_ego_states, observations=past_observation, sample_interval=scenario.database_interval)
    return history_buffer

def serialize_scenario(scenario: AbstractScenario, num_poses: int=12, future_time_horizon: float=6.0) -> SimulationHistory:
    """
    Serialize a scenario to a list of scene dicts.
    :param scenario: Scenario.
    :param num_poses: Number of poses in trajectory.
    :param future_time_horizon: Future time horizon in trajectory.
    :return: SimulationHistory containing all scenes.
    """
    simulation_history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
    ego_controller = PerfectTrackingController(scenario)
    simulation_time_controller = StepSimulationTimeController(scenario)
    observations = TracksObservation(scenario)
    history_buffer = _create_dummy_simulation_history_buffer(scenario=scenario)
    for _ in range(simulation_time_controller.number_of_iterations()):
        iteration = simulation_time_controller.get_iteration()
        ego_state = ego_controller.get_state()
        observation = observations.get_observation()
        traffic_light_status = list(scenario.get_traffic_light_status_at_iteration(iteration.index))
        current_state = scenario.get_ego_state_at_iteration(iteration.index)
        states = scenario.get_ego_future_trajectory(iteration.index, future_time_horizon, num_poses)
        trajectory = InterpolatedTrajectory(list(itertools.chain([current_state], states)))
        simulation_history.add_sample(SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status))
        next_iteration = simulation_time_controller.next_iteration()
        if next_iteration:
            ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
            observations.update_observation(iteration, next_iteration, history_buffer)
    return simulation_history

def get_sensor_data_token_timestamp_from_db(log_file: str, sensor_source: SensorDataSource, token: str) -> Optional[int]:
    """
    Get the timestamp associated with an individual lidar_pc token.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param token: The token for which to grab the timestamp.
    :return: The timestamp associated with the token, if found.
    """
    query = f'\n    SELECT timestamp\n    FROM {sensor_source.table}\n    WHERE token = ?;\n    '
    result = execute_one(query, (bytearray.fromhex(token),), log_file)
    return None if result is None else int(result['timestamp'])

def get_future_waypoints_for_agents_from_db(log_file: str, track_tokens: Union[Generator[str, None, None], List[str]], start_timestamp: int, end_timestamp: int) -> Generator[Tuple[str, Waypoint], None, None]:
    """
    Obtain the future waypoints for the selected agents from the DB in the provided time window.
    Results are sorted by track token, then by timestamp in ascending order.

    :param log_file: The log file to query.
    :param track_tokens: The track_tokens for which to query.
    :param start_timestamp: The starting timestamp for which to query.
    :param end_timestamp: The maximal time for which to query.
    :return: A generator of tuples of (track_token, Waypoint), sorted by track_token, then by timestamp in ascending order.
    """
    if not isinstance(track_tokens, list):
        track_tokens = list(track_tokens)
    query = f'\n        SELECT  lb.x,\n                lb.y,\n                lb.z,\n                lb.yaw,\n                lb.width,\n                lb.length,\n                lb.height,\n                lb.vx,\n                lb.vy,\n                lb.track_token,\n                lp.timestamp\n        FROM lidar_box AS lb\n        INNER JOIN lidar_pc AS lp\n            ON lp.token = lb.lidar_pc_token\n        WHERE   lp.timestamp >= ?\n            AND lp.timestamp <= ?\n            AND lb.track_token IN\n            ({('?,' * len(track_tokens))[:-1]})\n        ORDER BY lb.track_token ASC, lp.timestamp ASC;\n    '
    args = [start_timestamp, end_timestamp] + [bytearray.fromhex(t) for t in track_tokens]
    for row in execute_many(query, args, log_file):
        pose = StateSE2(row['x'], row['y'], row['yaw'])
        oriented_box = OrientedBox(pose, width=row['width'], height=row['height'], length=row['length'])
        velocity = StateVector2D(row['vx'], row['vy'])
        yield (row['track_token'].hex(), Waypoint(TimePoint(row['timestamp']), oriented_box, velocity))

class TestNuPlanScenarioQueries(unittest.TestCase):
    """
    Test suite for the NuPlan scenario queries.
    """
    generation_parameters: DBGenerationParameters

    @staticmethod
    def getDBFilePath() -> Path:
        """
        Get the location for the temporary SQLite file used for the test DB.
        :return: The filepath for the test data.
        """
        return Path('/tmp/test_nuplan_scenario_queries.sqlite3')

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create the mock DB data.
        """
        db_file_path = TestNuPlanScenarioQueries.getDBFilePath()
        if db_file_path.exists():
            db_file_path.unlink()
        cls.generation_parameters = DBGenerationParameters(num_lidars=1, num_cameras=2, num_sensor_data_per_sensor=50, num_lidarpc_per_image_ratio=2, num_scenes=10, num_traffic_lights_per_lidar_pc=5, num_agents_per_lidar_pc=3, num_static_objects_per_lidar_pc=2, scene_scenario_tag_mapping={5: ['first_tag'], 6: ['first_tag', 'second_tag'], 7: ['second_tag']}, file_path=str(db_file_path))
        generate_minimal_nuplan_db(cls.generation_parameters)

    def setUp(self) -> None:
        """
        The method to run before each test.
        """
        self.db_file_name = str(TestNuPlanScenarioQueries.getDBFilePath())
        self.sensor_source = SensorDataSource('lidar_pc', 'lidar', 'lidar_token', 'channel')

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Destroy the mock DB data.
        """
        db_file_path = TestNuPlanScenarioQueries.getDBFilePath()
        if os.path.exists(db_file_path):
            os.remove(db_file_path)

    def test_get_sensor_token_from_index(self) -> None:
        """
        Test the get_sensor_token_from_index query.
        """
        for sample_index in [0, 12, 24]:
            retrieved_token = get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, sample_index)
            self.assertEqual(sample_index / self.generation_parameters.num_lidars, str_token_to_int(retrieved_token))
        self.assertIsNone(get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, 100000))
        with self.assertRaises(ValueError):
            get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, -2)

    def test_get_end_sensor_time_from_db(self) -> None:
        """
        Test the get_end_sensor_time_from_db query.
        """
        log_end_time = get_end_sensor_time_from_db(self.db_file_name, sensor_source=self.sensor_source)
        self.assertEqual(49 * 1000000.0, log_end_time)

    def test_get_sensor_token_timestamp_from_db(self) -> None:
        """
        Test the get_sensor_data_token_timestamp_from_db query.
        """
        for token in [0, 3, 7]:
            expected_timestamp = token * 1000000.0
            actual_timestamp = get_sensor_data_token_timestamp_from_db(self.db_file_name, self.sensor_source, int_to_str_token(token))
            self.assertEqual(expected_timestamp, actual_timestamp)
        self.assertIsNone(get_sensor_data_token_timestamp_from_db(self.db_file_name, self.sensor_source, int_to_str_token(1000)))

    def test_get_sensor_token_map_name_from_db(self) -> None:
        """
        Test the get_sensor_token_map_name_from_db query.
        """
        for token in [0, 2, 6]:
            expected_map_name = 'map_version'
            actual_map_name = get_sensor_token_map_name_from_db(self.db_file_name, self.sensor_source, int_to_str_token(token))
            self.assertEqual(expected_map_name, actual_map_name)
        self.assertIsNone(get_sensor_token_map_name_from_db(self.db_file_name, self.sensor_source, int_to_str_token(1000)))

    def test_get_sampled_sensor_tokens_in_time_window_from_db(self) -> None:
        """
        Test the get_sampled_lidarpc_tokens_in_time_window_from_db query.
        """
        expected_tokens = [10, 13, 16, 19]
        actual_tokens = list((str_token_to_int(v) for v in get_sampled_sensor_tokens_in_time_window_from_db(log_file=self.db_file_name, sensor_source=self.sensor_source, start_timestamp=int(10 * 1000000.0), end_timestamp=int(20 * 1000000.0), subsample_interval=3)))
        self.assertEqual(expected_tokens, actual_tokens)

    def test_get_sensor_data_from_sensor_data_tokens_from_db(self) -> None:
        """
        Test the get_sensor_data_from_sensor_data_tokens_from_db query.
        """
        lidar_pc_tokens = [int_to_str_token(v) for v in [10, 13, 21]]
        image_tokens = [int_to_str_token(v) for v in [1100000]]
        lidar_pcs = [cast(LidarPc, sensor_data) for sensor_data in get_sensor_data_from_sensor_data_tokens_from_db(self.db_file_name, self.sensor_source, LidarPc, lidar_pc_tokens)]
        images = [cast(Image, sensor_data) for sensor_data in get_sensor_data_from_sensor_data_tokens_from_db(self.db_file_name, SensorDataSource('image', 'camera', 'camera_token', 'camera_0'), Image, image_tokens)]
        self.assertEqual(len(lidar_pc_tokens), len(lidar_pcs))
        self.assertEqual(len(image_tokens), len(images))
        lidar_pcs.sort(key=lambda x: int(x.timestamp))
        self.assertEqual(10, str_token_to_int(lidar_pcs[0].token))
        self.assertEqual(13, str_token_to_int(lidar_pcs[1].token))
        self.assertEqual(21, str_token_to_int(lidar_pcs[2].token))
        self.assertEqual(1100000, str_token_to_int(images[0].token))

    def test_get_lidar_transform_matrix_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_sensor_transform_matrix_for_sensor_data_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            xform_mat = get_sensor_transform_matrix_for_sensor_data_token_from_db(self.db_file_name, self.sensor_source, int_to_str_token(sample_token))
            self.assertIsNotNone(xform_mat)
            self.assertEqual(xform_mat[0, 3], 0)

    def test_get_mission_goal_for_sensor_data_token_from_db(self) -> None:
        """
        Test the get_mission_goal_for_sensor_data_token_from_db query.
        """
        query_lidarpc_token = int_to_str_token(12)
        expected_ego_pose_x = 14
        expected_ego_pose_y = 15
        result = get_mission_goal_for_sensor_data_token_from_db(self.db_file_name, self.sensor_source, query_lidarpc_token)
        self.assertIsNotNone(result)
        self.assertEqual(expected_ego_pose_x, result.x)
        self.assertEqual(expected_ego_pose_y, result.y)

    def test_get_roadblock_ids_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_roadblock_ids_for_lidarpc_token_from_db query.
        """
        result = get_roadblock_ids_for_lidarpc_token_from_db(self.db_file_name, int_to_str_token(0))
        self.assertEqual(result, ['0', '1', '2'])

    def test_get_statese2_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_statese2_for_lidarpc_token_from_db query.
        """
        query_lidarpc_token = int_to_str_token(13)
        expected_ego_pose_x = 13
        expected_ego_pose_y = 14
        result = get_statese2_for_lidarpc_token_from_db(self.db_file_name, query_lidarpc_token)
        self.assertIsNotNone(result)
        self.assertEqual(expected_ego_pose_x, result.x)
        self.assertEqual(expected_ego_pose_y, result.y)

    def test_get_sampled_lidarpcs_from_db(self) -> None:
        """
        Test the get_sampled_lidarpcs_from_db query.
        """
        test_cases = [{'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': True, 'expected_return_tokens': [5, 6, 7]}, {'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': False, 'expected_return_tokens': [3, 4, 5]}, {'initial_token': 7, 'sample_indexes': [0, 3, 12], 'future': False, 'expected_return_tokens': [4, 7]}, {'initial_token': 0, 'sample_indexes': [1000], 'future': True, 'expected_return_tokens': []}]
        for test_case in test_cases:
            initial_token = int_to_str_token(test_case['initial_token'])
            expected_return_tokens = [int_to_str_token(v) for v in test_case['expected_return_tokens']]
            actual_returned_lidarpcs = list(get_sampled_lidarpcs_from_db(self.db_file_name, initial_token, self.sensor_source, test_case['sample_indexes'], test_case['future']))
            self.assertEqual(len(expected_return_tokens), len(actual_returned_lidarpcs))
            for i in range(len(expected_return_tokens)):
                self.assertEqual(expected_return_tokens[i], actual_returned_lidarpcs[i].token)

    def test_get_sampled_ego_states_from_db(self) -> None:
        """
        Test the get_sampled_ego_states_from_db query.
        """
        test_cases = [{'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': True, 'expected_row_indexes': [5, 6, 7]}, {'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': False, 'expected_row_indexes': [3, 4, 5]}, {'initial_token': 7, 'sample_indexes': [0, 3, 12], 'future': False, 'expected_row_indexes': [4, 7]}, {'initial_token': 0, 'sample_indexes': [1000], 'future': True, 'expected_row_indexes': []}]
        for test_case in test_cases:
            initial_token = int_to_str_token(test_case['initial_token'])
            expected_row_indexes = test_case['expected_row_indexes']
            actual_returned_ego_states = list(get_sampled_ego_states_from_db(self.db_file_name, initial_token, self.sensor_source, test_case['sample_indexes'], test_case['future']))
            self.assertEqual(len(expected_row_indexes), len(actual_returned_ego_states))
            for i in range(len(expected_row_indexes)):
                self.assertEqual(expected_row_indexes[i] * 1000000.0, actual_returned_ego_states[i].time_point.time_us)

    def test_get_ego_state_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_ego_state_for_lidarpc_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            returned_pose = get_ego_state_for_lidarpc_token_from_db(self.db_file_name, query_token)
            self.assertEqual(sample_token * 1000000.0, returned_pose.time_point.time_us)

    def test_get_traffic_light_status_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_traffic_light_status_for_lidarpc_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            traffic_light_statuses = list(get_traffic_light_status_for_lidarpc_token_from_db(self.db_file_name, query_token))
            self.assertEqual(5, len(traffic_light_statuses))
            for tl_status in traffic_light_statuses:
                self.assertEqual(sample_token * 1000000.0, tl_status.timestamp)

    def test_get_tracked_objects_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_tracked_objects_for_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            tracked_objects = list(get_tracked_objects_for_lidarpc_token_from_db(self.db_file_name, query_token))
            self.assertEqual(5, len(tracked_objects))
            agent_count = 0
            static_object_count = 0
            track_token_base_id = 600000
            token_base_id = 500000
            token_sample_step = 10000
            for idx, tracked_object in enumerate(tracked_objects):
                expected_track_token = track_token_base_id + idx
                expected_token = token_base_id + token_sample_step * sample_token + idx
                self.assertEqual(int_to_str_token(expected_track_token), tracked_object.track_token)
                self.assertEqual(int_to_str_token(expected_token), tracked_object.token)
                if isinstance(tracked_object, Agent):
                    agent_count += 1
                    self.assertEqual(TrackedObjectType.VEHICLE, tracked_object.tracked_object_type)
                    self.assertEqual(0, len(tracked_object.predictions))
                elif isinstance(tracked_object, StaticObject):
                    static_object_count += 1
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, tracked_object.tracked_object_type)
                else:
                    raise ValueError(f'Unexpected type: {type(tracked_object)}')
            self.assertEqual(3, agent_count)
            self.assertEqual(2, static_object_count)

    def test_get_tracked_objects_within_time_interval_from_db(self) -> None:
        """
        Test the get_tracked_objects_within_time_interval_from_db query.
        """
        expected_num_windows = {0: 3, 30: 5, 48: 4}
        expected_backward_offset = {0: 0, 30: -2, 48: -2}
        for sample_token in expected_num_windows.keys():
            start_timestamp = int(1000000.0 * (sample_token - 2))
            end_timestamp = int(1000000.0 * (sample_token + 2))
            tracked_objects = list(get_tracked_objects_within_time_interval_from_db(self.db_file_name, start_timestamp, end_timestamp, filter_track_tokens=None))
            expected_num_tokens = expected_num_windows[sample_token] * 5
            self.assertEqual(expected_num_tokens, len(tracked_objects))
            agent_count = 0
            static_object_count = 0
            track_token_base_id = 600000
            token_base_id = 500000
            token_sample_step = 10000
            for idx, tracked_object in enumerate(tracked_objects):
                expected_track_token = track_token_base_id + idx % 5
                expected_token = token_base_id + token_sample_step * (sample_token + expected_backward_offset[sample_token] + math.floor(idx / 5)) + idx % 5
                self.assertEqual(int_to_str_token(expected_track_token), tracked_object.track_token)
                self.assertEqual(int_to_str_token(expected_token), tracked_object.token)
                if isinstance(tracked_object, Agent):
                    agent_count += 1
                    self.assertEqual(TrackedObjectType.VEHICLE, tracked_object.tracked_object_type)
                    self.assertEqual(0, len(tracked_object.predictions))
                elif isinstance(tracked_object, StaticObject):
                    static_object_count += 1
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, tracked_object.tracked_object_type)
                else:
                    raise ValueError(f'Unexpected type: {type(tracked_object)}')
            self.assertEqual(3 * expected_num_windows[sample_token], agent_count)
            self.assertEqual(2 * expected_num_windows[sample_token], static_object_count)

    def test_get_future_waypoints_for_agents_from_db(self) -> None:
        """
        Test the get_future_waypoints_for_agents_from_db query.
        """
        track_tokens = [600000, 600001, 600002]
        start_timestamp = 0
        end_timestamp = int(20 * 1000000.0 - 1)
        query_output: Dict[str, List[Waypoint]] = {}
        for token, waypoint in get_future_waypoints_for_agents_from_db(self.db_file_name, (int_to_str_token(t) for t in track_tokens), start_timestamp, end_timestamp):
            if token not in query_output:
                query_output[token] = []
            query_output[token].append(waypoint)
        expected_keys = ['{:08d}'.format(t) for t in track_tokens]
        self.assertEqual(len(expected_keys), len(query_output))
        for expected_key in expected_keys:
            self.assertTrue(expected_key in query_output)
            collected_waypoints = query_output[expected_key]
            self.assertEqual(20, len(collected_waypoints))
            for i in range(0, len(collected_waypoints), 1):
                self.assertEqual(i * 1000000.0, collected_waypoints[i].time_point.time_us)

    def test_get_scenarios_from_db(self) -> None:
        """
        Test the get_scenarios_from_db_query.
        """
        no_filter_output: List[int] = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            no_filter_output.append(str_token_to_int(row['token'].hex()))
        self.assertEqual(list(range(10, 40, 1)), no_filter_output)
        filter_tokens = [int_to_str_token(v) for v in [15, 30]]
        tokens_filter_output: List[int] = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=filter_tokens, filter_types=None, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            tokens_filter_output.append(row['token'].hex())
        self.assertEqual(filter_tokens, tokens_filter_output)
        filter_scenarios = ['first_tag']
        extracted_rows: List[Tuple[int, str]] = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=filter_scenarios, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            extracted_rows.append((str_token_to_int(row['token'].hex()), row['scenario_type']))
        self.assertEqual(2, len(extracted_rows))
        self.assertEqual(25, extracted_rows[0][0])
        self.assertEqual('first_tag', extracted_rows[0][1])
        self.assertEqual(30, extracted_rows[1][0])
        self.assertEqual('first_tag', extracted_rows[1][1])
        filter_scenarios = ['second_tag']
        extracted_rows = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=filter_scenarios, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            extracted_rows.append((str_token_to_int(row['token'].hex()), row['scenario_type']))
        self.assertEqual(2, len(extracted_rows))
        self.assertEqual(30, extracted_rows[0][0])
        self.assertEqual('second_tag', extracted_rows[0][1])
        self.assertEqual(35, extracted_rows[1][0])
        self.assertEqual('second_tag', extracted_rows[1][1])
        filter_maps = ['map_version']
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=filter_maps, include_invalid_mission_goals=False, include_cameras=False)))
        self.assertLess(0, row_cnt)
        filter_maps = ['map_that_does_not_exist']
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=filter_maps, include_invalid_mission_goals=False, include_cameras=False)))
        self.assertEqual(0, row_cnt)
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=True)))
        self.assertEqual(15, row_cnt)
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=[int_to_str_token(25)], filter_types=['first_tag'], filter_map_names=['map_version'], include_invalid_mission_goals=False, include_cameras=False)))
        self.assertEqual(1, row_cnt)

    def test_get_lidarpc_tokens_with_scenario_tag_from_db(self) -> None:
        """
        Test the get_lidarpc_tokens_with_scenario_tag_from_db query.
        """
        tuples = list(get_lidarpc_tokens_with_scenario_tag_from_db(self.db_file_name))
        self.assertEqual(4, len(tuples))
        expected_tuples = [('first_tag', int_to_str_token(25)), ('first_tag', int_to_str_token(30)), ('second_tag', int_to_str_token(30)), ('second_tag', int_to_str_token(35))]
        for tup in tuples:
            self.assertTrue(tup in expected_tuples)

    def test_get_sensor_token(self) -> None:
        """Test the get_lidarpc_token_from_index query."""
        retrieved_token = get_sensor_token(self.db_file_name, 'lidar', 'channel')
        self.assertEqual(700000, str_token_to_int(retrieved_token))
        with self.assertRaisesRegex(RuntimeError, 'Channel missing_channel not found in table lidar!'):
            self.assertIsNone(get_sensor_token(self.db_file_name, 'lidar', 'missing_channel'))

    def test_get_images_from_lidar_tokens(self) -> None:
        """Test the get_images_from_lidar_tokens query."""
        token = int_to_str_token(20)
        retrieved_images = list(get_images_from_lidar_tokens(self.db_file_name, [token], ['camera_0', 'camera_1'], 50000, 50000))
        self.assertEqual(2, len(retrieved_images))
        self.assertEqual(1100020, str_token_to_int(retrieved_images[0].token))
        self.assertEqual(1100070, str_token_to_int(retrieved_images[1].token))
        self.assertEqual('camera_0', retrieved_images[0].channel)
        self.assertEqual('camera_1', retrieved_images[1].channel)

    def test_get_cameras(self) -> None:
        """Test the get_cameras query."""
        retrieved_cameras = list(get_cameras(self.db_file_name, ['camera_0', 'camera_1']))
        self.assertEqual(2, len(retrieved_cameras))
        self.assertEqual(1000000, str_token_to_int(retrieved_cameras[0].token))
        self.assertEqual(1000001, str_token_to_int(retrieved_cameras[1].token))
        self.assertEqual('camera_0', retrieved_cameras[0].channel)
        self.assertEqual('camera_1', retrieved_cameras[1].channel)
        retrieved_cameras = list(get_cameras(self.db_file_name, ['camera_1']))
        self.assertEqual(1, len(retrieved_cameras))
        self.assertEqual(1000001, str_token_to_int(retrieved_cameras[0].token))
        self.assertEqual('camera_1', retrieved_cameras[0].channel)

class TestEgoState(unittest.TestCase):
    """Tests EgoState class"""

    def setUp(self) -> None:
        """Creates sample parameters for testing"""
        self.ego_state = get_sample_ego_state()
        self.vehicle = get_pacifica_parameters()
        self.dynamic_car_state = get_sample_dynamic_car_state(self.vehicle.rear_axle_to_center)

    def test_ego_state_extended_construction(self) -> None:
        """Tests that the ego state extended can be constructed from a pre-existing ego state."""
        ego_state_ext = EgoState.build_from_rear_axle(rear_axle_pose=self.ego_state.rear_axle, rear_axle_velocity_2d=self.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=self.dynamic_car_state.rear_axle_acceleration_2d, tire_steering_angle=self.ego_state.tire_steering_angle, time_point=self.ego_state.time_point, angular_vel=self.dynamic_car_state.angular_velocity, angular_accel=self.dynamic_car_state.angular_acceleration, is_in_auto_mode=True, vehicle_parameters=self.vehicle)
        self.assertTrue(ego_state_ext.dynamic_car_state == self.dynamic_car_state)
        self.assertTrue(ego_state_ext.center == self.ego_state.center)
        wp = ego_state_ext.waypoint
        self.assertEqual(wp.time_point, ego_state_ext.time_point)
        self.assertEqual(wp.oriented_box, ego_state_ext.car_footprint)
        self.assertEqual(wp.velocity, ego_state_ext.dynamic_car_state.rear_axle_velocity_2d)

    def test_to_split_state(self) -> None:
        """Tests that the state gets split as expected"""
        split_state = self.ego_state.to_split_state()
        self.assertEqual(len(split_state.linear_states), 8)
        self.assertEqual(split_state.fixed_states, [self.ego_state.car_footprint.vehicle_parameters])
        self.assertEqual(split_state.angular_states, [self.ego_state.rear_axle.heading])

    def test_from_split_state(self) -> None:
        """Tests that the object gets created as expected from the split state"""
        split_state = SplitState([0, 1, 2, 3, 4, 5, 6, 7], [8], [self.ego_state.car_footprint.vehicle_parameters])
        ego_from_split = EgoState.from_split_state(split_state)
        self.assertEqual(self.ego_state.car_footprint.vehicle_parameters, ego_from_split.car_footprint.vehicle_parameters)
        self.assertAlmostEqual(ego_from_split.time_us, 0)
        self.assertAlmostEqual(ego_from_split.rear_axle.x, 1)
        self.assertAlmostEqual(ego_from_split.rear_axle.y, 2)
        self.assertAlmostEqual(ego_from_split.rear_axle.heading, 8)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_velocity_2d.x, 3)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_velocity_2d.y, 4)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_acceleration_2d.x, 5)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_acceleration_2d.y, 6)
        self.assertAlmostEqual(ego_from_split.tire_steering_angle, 7)

class TestTrackedObjects(unittest.TestCase):
    """Tests TrackedObjects class"""

    def setUp(self) -> None:
        """Creates sample agents for testing"""
        self.agents = [get_sample_agent('foo', TrackedObjectType.PEDESTRIAN), get_sample_agent('bar', TrackedObjectType.VEHICLE), get_sample_agent('bar_out_the_car', TrackedObjectType.PEDESTRIAN)]

    def test_construction(self) -> None:
        """Tests that the object can be created correctly."""
        tracked_objects = TrackedObjects(self.agents)
        expected_type_and_set_of_tokens: Dict[TrackedObjectType, Any] = {object_type: set() for object_type in TrackedObjectType}
        expected_type_and_set_of_tokens[TrackedObjectType.PEDESTRIAN].update({'foo', 'bar_out_the_car'})
        expected_type_and_set_of_tokens[TrackedObjectType.VEHICLE].update({'bar'})
        for tracked_object_type in TrackedObjectType:
            if tracked_object_type not in expected_type_and_set_of_tokens:
                continue
            self.assertEqual(expected_type_and_set_of_tokens[tracked_object_type], {tracked_object.token for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type)})

    def test_get_subset(self) -> None:
        """Tests that the object can be created correctly."""
        tracked_objects = TrackedObjects(self.agents)
        agents = tracked_objects.get_agents()
        static_objects = tracked_objects.get_static_objects()
        self.assertEqual(3, len(agents))
        self.assertEqual(0, len(static_objects))

    def test_get_tracked_objects_of_types(self) -> None:
        """Test get_tracked_objects_of_types()"""
        tracked_objects = TrackedObjects(self.agents)
        track_types = [TrackedObjectType.PEDESTRIAN, TrackedObjectType.VEHICLE]
        tracks = tracked_objects.get_tracked_objects_of_types(track_types)
        self.assertEqual(3, len(tracks))

class TestSceneObject(unittest.TestCase):
    """Tests SceneObject class"""

    @patch('nuplan.common.actor_state.tracked_objects_types.TrackedObjectType')
    @patch('nuplan.common.actor_state.oriented_box.OrientedBox')
    def test_initialization(self, mock_box: Mock, mock_tracked_object_type: Mock) -> None:
        """Tests that agents can be initialized correctly"""
        scene_object = SceneObject(mock_tracked_object_type, mock_box, SceneObjectMetadata(1, '123', 1, '456'))
        self.assertEqual('123', scene_object.token)
        self.assertEqual('456', scene_object.track_token)
        self.assertEqual(mock_box, scene_object.box)
        self.assertEqual(mock_tracked_object_type, scene_object.tracked_object_type)

    @patch('nuplan.common.actor_state.scene_object.StateSE2')
    @patch('nuplan.common.actor_state.scene_object.OrientedBox')
    @patch('nuplan.common.actor_state.scene_object.TrackedObjectType')
    @patch('nuplan.common.actor_state.scene_object.SceneObject.__init__')
    def test_construction(self, mock_init: Mock, mock_type: Mock, mock_box_object: Mock, mock_state: Mock) -> None:
        """Test that agents can be constructed correctly."""
        mock_init.return_value = None
        mock_box = Mock()
        mock_box_object.return_value = mock_box
        _ = SceneObject.from_raw_params('123', '123', 1, 1, mock_state, size=(3, 2, 1))
        mock_box_object.assert_called_with(mock_state, width=3, length=2, height=1)
        mock_init.assert_called_with(metadata=SceneObjectMetadata(token='123', track_token='123', timestamp_us=1, track_id=1), tracked_object_type=mock_type.GENERIC_OBJECT, oriented_box=mock_box)

def interpolate_agent(agent: AgentTemporalState, horizon_len_s: float, interval_s: float) -> AgentTemporalState:
    """
    Interpolate agent's future predictions and past trajectory based on the predefined length and interval
    :param agent: to be interpolated
    :param horizon_len_s: [s] horizon of predictions
    :param interval_s: [s] interval between two states
    :return: interpolated agent, where missing waypoints are replaced with None
    """
    interpolated_agent = agent
    if interpolated_agent.predictions:
        interpolated_agent.predictions = [PredictedTrajectory(waypoints=interpolate_future_waypoints(mode.waypoints, horizon_len_s=horizon_len_s, interval_s=interval_s), probability=mode.probability) for mode in interpolated_agent.predictions]
    past_trajectory = interpolated_agent.past_trajectory
    if past_trajectory:
        interpolated_agent.past_trajectory = PredictedTrajectory(waypoints=interpolate_past_waypoints(past_trajectory.waypoints, horizon_len_s=horizon_len_s, interval_s=interval_s), probability=past_trajectory.probability)
    return interpolated_agent

def interpolate_future_waypoints(waypoints: List[InterpolatableState], horizon_len_s: float, interval_s: float) -> List[Optional[InterpolatableState]]:
    """
    Interpolate waypoints which are in the future. If not enough waypoints are provided, we append None
    :param waypoints: list of waypoints, there needs to be at least one
    :param horizon_len_s: [s] time distance to future
    :param interval_s: [s] interval between two states
    :return: interpolated waypoints
    """
    _validate_waypoints(waypoints)
    start_timestamp = waypoints[0].time_us
    end_timestamp = int(start_timestamp + horizon_len_s * 1000000.0)
    target_timestamps, num_future_boxes = _compute_desired_time_steps(start_timestamp, end_timestamp, horizon_len_s=horizon_len_s, interval_s=interval_s)
    if len(waypoints) == 1:
        return waypoints + cast(List[Optional[InterpolatableState]], [None] * (num_future_boxes - 1))
    return _interpolate_waypoints(waypoints, target_timestamps)

def interpolate_past_waypoints(waypoints: List[InterpolatableState], horizon_len_s: float, interval_s: float) -> List[Optional[InterpolatableState]]:
    """
    Interpolate waypoints which are in the past. We assume that they are still monotonically increasing.
        If not enough waypoints are provided, we append None
    :param waypoints: list of waypoints, there needs to be at least one
    :param horizon_len_s: [s] time distance to past
    :param interval_s: [s] interval between two states
    :return: interpolated waypoints
    """
    _validate_waypoints(waypoints)
    end_timestamp = waypoints[-1].time_us
    start_timestamp = max(int(end_timestamp - horizon_len_s * 1000000.0), 0)
    target_timestamps, num_future_boxes = _compute_desired_time_steps(start_timestamp, end_timestamp, horizon_len_s=horizon_len_s, interval_s=interval_s)
    if len(waypoints) == 1:
        return cast(List[Optional[InterpolatableState]], [None] * (num_future_boxes - 1)) + waypoints
    sampled_trajectory = _interpolate_waypoints(waypoints, target_timestamps)
    if not sampled_trajectory[-1]:
        raise RuntimeError('Last state of the trajectory has to be existent!')
    return sampled_trajectory

def interpolate_tracks(tracked_objects: Union[TrackedObjects, List[TrackedObject]], horizon_len_s: float, interval_s: float) -> List[TrackedObject]:
    """
    Interpolate agent's predictions and past trajectory, if not enough states are present, add NONE!
    :param tracked_objects: agents to be interpolated
    :param horizon_len_s: [s] horizon from initial waypoint
    :param interval_s: [s] interval between two states
    :return: interpolated agents
    """
    all_tracked_objects = tracked_objects if isinstance(tracked_objects, TrackedObjects) else TrackedObjects(tracked_objects)
    return [interpolate_agent(agent, horizon_len_s=horizon_len_s, interval_s=interval_s) for agent in all_tracked_objects.get_agents()] + cast(List[TrackedObject], all_tracked_objects.get_static_objects())

def _interpolate_waypoints(waypoints: List[InterpolatableState], target_timestamps: npt.NDArray[np.float64], pad_with_none: bool=True) -> List[Optional[InterpolatableState]]:
    """
    Interpolate waypoints when required from target_timestamps
    :param waypoints: to be interpolated
    :param target_timestamps: desired sampling
    :param pad_with_none: if True, the output will have None for states that can not be interpolated
    :return: list of existent interpolations, if an interpolation is not possible, it will be replaced with None
    """
    trajectory = InterpolatedTrajectory(waypoints)
    if pad_with_none:
        return [trajectory.get_state_at_time(TimePoint(t)) if trajectory.is_in_range(TimePoint(t)) else None for t in target_timestamps]
    return [trajectory.get_state_at_time(TimePoint(t)) for t in target_timestamps if trajectory.is_in_range(TimePoint(t))]

def _validate_waypoints(waypoints: List[InterpolatableState]) -> None:
    """
    Make sure that waypoints are valid for interpolation
        raise in case they are empty or they are not monotonically increasing
    :param waypoints: list of waypoints to be interpolated
    """
    if not waypoints:
        raise RuntimeError('There are no waypoints!')
    if not np.all(np.diff([w.time_us for w in waypoints]) > 0):
        raise ValueError(f'The waypoints are not monotonically increasing: {[w.time_us for w in waypoints]}!')

def _compute_desired_time_steps(start_timestamp: int, end_timestamp: int, horizon_len_s: float, interval_s: float) -> Tuple[npt.NDArray[np.float64], int]:
    """
    Compute the desired sampling
    :param start_timestamp: [us] starting time stamp
    :param end_timestamp: [us] ending time stamp
    :param horizon_len_s: [s] length of horizon
    :param interval_s: [s] interval between states
    :return: array of time stamps, and the desired length
    """
    num_future_boxes = int(horizon_len_s / interval_s)
    num_target_timestamps = num_future_boxes + 1
    return (np.linspace(start=start_timestamp, stop=end_timestamp, num=num_target_timestamps), num_target_timestamps)

@nuplan_test(path='json/interpolate_future.json')
def test_interpolate_tracked_object(scene: Dict[str, Any]) -> None:
    """Test that we can interpolate agents with various initial length."""
    tracked_objects = from_scene_to_tracked_objects_with_predictions(scene['world'], scene['prediction'])
    future_horizon_len_s = 8.0
    future_interval_s = 0.5
    agents = interpolate_tracks(tracked_objects, future_horizon_len_s, future_interval_s)
    desired_length = int(future_horizon_len_s / future_interval_s) + 1
    for agent in agents:
        assert agent.predictions, 'Predictions have to exist!'
        for prediction in agent.predictions:
            last_original_prediction_state = [json_prediction['states'][-1] for json_prediction in scene['prediction'] if json_prediction['id'] == agent.metadata.track_id]
            assert len(last_original_prediction_state) == 1, 'We did not find original prediction?'
            last_time_stamp = last_original_prediction_state[0]['timestamp']
            nonzero = [w for w in prediction.waypoints if w]
            if last_time_stamp < future_interval_s:
                assert len(nonzero) == 1, 'The length of non zero predictions has to be 1!'
            if future_interval_s < last_time_stamp < 2 * future_interval_s:
                assert len(nonzero) == 2, 'The length of non zero predictions has to be 2!'
            assert len(prediction.waypoints) == desired_length, 'Prediction does not have desired length!'
            for index, waypoint in enumerate(prediction.waypoints):
                if index != 0 and waypoint:
                    time_interval = waypoint.time_point.time_s - prediction.waypoints[index - 1].time_point.time_s
                    assert time_interval == pytest.approx(future_interval_s), 'The sampling is not correct!'

class Simulation:
    """
    This class queries data for initialization of a planner, and propagates simulation a step forward based on the
        planned trajectory of a planner.
    """

    def __init__(self, simulation_setup: SimulationSetup, callback: Optional[AbstractCallback]=None, simulation_history_buffer_duration: float=2):
        """
        Create Simulation.
        :param simulation_setup: Configuration that describes the simulation.
        :param callback: A callback to be executed for this simulation setup
        :param simulation_history_buffer_duration: [s] Duration to pre-load scenario into the buffer.
        """
        if simulation_history_buffer_duration < simulation_setup.scenario.database_interval:
            raise ValueError(f'simulation_history_buffer_duration {simulation_history_buffer_duration} has to be larger than the scenario database_interval {simulation_setup.scenario.database_interval}')
        self._setup = simulation_setup
        self._time_controller = simulation_setup.time_controller
        self._ego_controller = simulation_setup.ego_controller
        self._observations = simulation_setup.observations
        self._scenario = simulation_setup.scenario
        self._callback = MultiCallback([]) if callback is None else callback
        self._history = SimulationHistory(self._scenario.map_api, self._scenario.get_mission_goal())
        self._simulation_history_buffer_duration = simulation_history_buffer_duration + self._scenario.database_interval
        self._history_buffer_size = int(self._simulation_history_buffer_duration / self._scenario.database_interval) + 1
        self._history_buffer: Optional[SimulationHistoryBuffer] = None
        self._is_simulation_running = True

    def __reduce__(self) -> Tuple[Type[Simulation], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._setup, self._callback, self._simulation_history_buffer_duration))

    def is_simulation_running(self) -> bool:
        """
        Check whether a simulation reached the end
        :return True if simulation hasn't reached the end, otherwise false.
        """
        return not self._time_controller.reached_end() and self._is_simulation_running

    def reset(self) -> None:
        """
        Reset all internal states of simulation.
        """
        self._history.reset()
        self._setup.reset()
        self._history_buffer = None
        self._is_simulation_running = True

    def initialize(self) -> PlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
        :return data needed for planner initialization.
        """
        self.reset()
        self._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(self._history_buffer_size, self._scenario, self._observations.observation_type())
        self._observations.initialize()
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())
        return PlannerInitialization(route_roadblock_ids=self._scenario.get_route_roadblock_ids(), mission_goal=self._scenario.get_mission_goal(), map_api=self._scenario.map_api)

    def get_planner_input(self) -> PlannerInput:
        """
        Construct inputs to the planner for the current iteration step
        :return Inputs to the planner.
        """
        if self._history_buffer is None:
            raise RuntimeError('Simulation was not initialized!')
        if not self.is_simulation_running():
            raise RuntimeError('Simulation is not running, stepping can not be performed!')
        iteration = self._time_controller.get_iteration()
        traffic_light_data = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))
        logger.debug(f'Executing {iteration.index}!')
        return PlannerInput(iteration=iteration, history=self._history_buffer, traffic_light_data=traffic_light_data)

    def propagate(self, trajectory: AbstractTrajectory) -> None:
        """
        Propagate the simulation based on planner's trajectory and the inputs to the planner
        This function also decides whether simulation should still continue. This flag can be queried through
        reached_end() function
        :param trajectory: computed trajectory from planner.
        """
        if self._history_buffer is None:
            raise RuntimeError('Simulation was not initialized!')
        if not self.is_simulation_running():
            raise RuntimeError('Simulation is not running, simulation can not be propagated!')
        iteration = self._time_controller.get_iteration()
        ego_state, observation = self._history_buffer.current_state
        traffic_light_status = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))
        logger.debug(f'Adding to history: {iteration.index}')
        self._history.add_sample(SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status))
        next_iteration = self._time_controller.next_iteration()
        if next_iteration:
            self._ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
            self._observations.update_observation(iteration, next_iteration, self._history_buffer)
        else:
            self._is_simulation_running = False
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: used scenario in this simulation.
        """
        return self._scenario

    @property
    def setup(self) -> SimulationSetup:
        """
        :return: Setup for this simulation.
        """
        return self._setup

    @property
    def callback(self) -> AbstractCallback:
        """
        :return: Callback for this simulation.
        """
        return self._callback

    @property
    def history(self) -> SimulationHistory:
        """
        :return History from the simulation.
        """
        return self._history

    @property
    def history_buffer(self) -> SimulationHistoryBuffer:
        """
        :return SimulationHistoryBuffer from the simulation.
        """
        if self._history_buffer is None:
            raise RuntimeError('_history_buffer is None. Please initialize the buffer by calling Simulation.initialize()')
        return self._history_buffer

class TestSimulation(unittest.TestCase):
    """
    Tests Simulation class which is updating simulation
    """

    def setUp(self) -> None:
        """Setup Mock classes."""
        self.scenario = MockAbstractScenario(number_of_past_iterations=10)
        self.sim_manager = StepSimulationTimeController(self.scenario)
        self.observation = TracksObservation(self.scenario)
        self.controller = PerfectTrackingController(self.scenario)
        self.setup = SimulationSetup(time_controller=self.sim_manager, observations=self.observation, ego_controller=self.controller, scenario=self.scenario)
        self.simulation_history_buffer_duration = 2
        self.stepper = Simulation(simulation_setup=self.setup, callback=MultiCallback([]), simulation_history_buffer_duration=self.simulation_history_buffer_duration)

    def test_stepper_initialize(self) -> None:
        """Test initialization method."""
        initialization = self.stepper.initialize()
        self.assertEqual(initialization.mission_goal, self.scenario.get_mission_goal())
        self.assertEqual(self.stepper._history_buffer.current_state[0].rear_axle, self.scenario.get_ego_state_at_iteration(0).rear_axle)

    def test_stepper_planner_input(self) -> None:
        """Test query to planner input function."""
        stepper = Simulation(simulation_setup=self.setup, callback=MultiCallback([]), simulation_history_buffer_duration=self.simulation_history_buffer_duration)
        stepper.initialize()
        planner_input = stepper.get_planner_input()
        self.assertEqual(planner_input.iteration.index, 0)

    def test_run_callbacks(self) -> None:
        """Test whether all callbacks are called"""
        callback = MagicMock()
        planner = SimplePlanner(2, 0.5, [0, 0])
        stepper = Simulation(simulation_setup=self.setup, callback=MultiCallback([callback]), simulation_history_buffer_duration=self.simulation_history_buffer_duration)
        runner = SimulationRunner(stepper, planner)
        runner.run()
        callback.on_simulation_start.assert_has_calls([call(stepper.setup)])
        callback.on_initialization_end.assert_has_calls([call(stepper.setup, planner)])
        callback.on_initialization_start.assert_has_calls([call(stepper.setup, planner)])
        callback.on_step_start.assert_has_calls([call(stepper.setup, planner)])
        callback.on_planner_start.assert_has_calls([call(stepper.setup, planner)])
        callback.on_step_end.assert_has_calls([call(stepper.setup, planner, stepper.history.last())])
        callback.on_simulation_end.assert_has_calls([call(stepper.setup, planner, stepper.history)])

    def test_buffer_simulation_duration(self) -> None:
        """Test initialization method."""
        self.stepper.initialize()
        simulation_buffer = self.stepper.history_buffer
        self.assertGreaterEqual(simulation_buffer.duration, self.simulation_history_buffer_duration)

class TestSimulationBufferInitialization(unittest.TestCase):
    """
    A class to test the simulation buffer initialization
    """

    def _test_simulation_buffer(self, time_step: float, simulation_history_buffer_duration: float) -> None:
        """
        Test the simulation buffer duration is equal or greater than simulation_history_buffer_duration.
        :param time_step: [s] The time interval of the simulation buffer.
        :param simulation_history_buffer_duration: [s] The requested simulation buffer duration.
        """
        scenario = MockAbstractScenario(number_of_past_iterations=200, time_step=time_step)
        sim_manager = StepSimulationTimeController(scenario)
        observation = TracksObservation(scenario)
        controller = PerfectTrackingController(scenario)
        setup = SimulationSetup(time_controller=sim_manager, observations=observation, ego_controller=controller, scenario=scenario)
        stepper = Simulation(simulation_setup=setup, callback=MultiCallback([]), simulation_history_buffer_duration=simulation_history_buffer_duration)
        stepper.initialize()
        simulation_buffer = stepper.history_buffer
        self.assertGreaterEqual(simulation_buffer.duration, simulation_history_buffer_duration)

    @settings(deadline=1000)
    @given(time_step=st.floats(min_value=0.05, max_value=1), simulation_history_buffer_duration=st.floats(min_value=1.0, max_value=10.0))
    @example(time_step=0.5, simulation_history_buffer_duration=2.0)
    def test_simulation_buffer(self, time_step: float, simulation_history_buffer_duration: float) -> None:
        """Test the simulation buffer initialization."""
        if simulation_history_buffer_duration % time_step > 1e-06:
            return
        if time_step > simulation_history_buffer_duration:
            with self.assertRaises(ValueError):
                self._test_simulation_buffer(time_step, simulation_history_buffer_duration)
        else:
            self._test_simulation_buffer(time_step, simulation_history_buffer_duration)

class TestPerfectTracking(unittest.TestCase):
    """
    Tests Tracker
    """

    def test_perfect_tracker(self) -> None:
        """
        Test the basic functionality of perfect tracker
        """
        initial_time_point = TimePoint(0)
        scenario = MockAbstractScenario(initial_time_us=initial_time_point)
        trajectory = InterpolatedTrajectory(list(scenario.get_expert_ego_trajectory()))
        tracker = PerfectTrackingController(scenario)
        desired_state = scenario.initial_ego_state
        state = scenario.initial_ego_state
        self.assertAlmostEqual(state.rear_axle.x, desired_state.rear_axle.x)
        self.assertAlmostEqual(state.rear_axle.y, desired_state.rear_axle.y)
        self.assertAlmostEqual(state.rear_axle.heading, desired_state.rear_axle.heading)
        tracker.update_state(current_iteration=SimulationIteration(time_point=initial_time_point, index=0), next_iteration=SimulationIteration(time_point=TimePoint(int(1 * 1000000.0)), index=1), ego_state=scenario.initial_ego_state, trajectory=trajectory)
        next_state = tracker.get_state()
        desired_state = scenario.get_ego_state_at_iteration(2)
        self.assertAlmostEqual(next_state.rear_axle.x, desired_state.rear_axle.x)
        self.assertAlmostEqual(next_state.rear_axle.y, desired_state.rear_axle.y)
        self.assertAlmostEqual(next_state.rear_axle.heading, desired_state.rear_axle.heading)

class KinematicBicycleModel(AbstractMotionModel):
    """
    A class describing the kinematic motion model where the rear axle is the point of reference.
    """

    def __init__(self, vehicle: VehicleParameters, max_steering_angle: float=np.pi / 3, accel_time_constant: float=0.2, steering_angle_time_constant: float=0.05):
        """
        Construct KinematicBicycleModel.

        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        :param accel_time_constant: low pass filter time constant for acceleration in s
        :param steering_angle_time_constant: low pass filter time constant for steering angle in s
        """
        self._vehicle = vehicle
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """Inherited, see super class."""
        longitudinal_speed = state.dynamic_car_state.rear_axle_velocity_2d.x
        x_dot = longitudinal_speed * np.cos(state.rear_axle.heading)
        y_dot = longitudinal_speed * np.sin(state.rear_axle.heading)
        yaw_dot = longitudinal_speed * np.tan(state.tire_steering_angle) / self._vehicle.wheel_base
        return EgoStateDot.build_from_rear_axle(rear_axle_pose=StateSE2(x=x_dot, y=y_dot, heading=yaw_dot), rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_acceleration_2d, rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=state.dynamic_car_state.tire_steering_rate, time_point=state.time_point, is_in_auto_mode=True, vehicle_parameters=self._vehicle)

    def _update_commands(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """
        This function applies some first order control delay/a low pass filter to acceleration/steering.

        :param state: Ego state
        :param ideal_dynamic_state: The desired dynamic state for propagation
        :param sampling_time: The time duration to propagate for
        :return: propagating_state including updated dynamic_state
        """
        dt_control = sampling_time.time_s
        accel = state.dynamic_car_state.rear_axle_acceleration_2d.x
        steering_angle = state.tire_steering_angle
        ideal_accel_x = ideal_dynamic_state.rear_axle_acceleration_2d.x
        ideal_steering_angle = dt_control * ideal_dynamic_state.tire_steering_rate + steering_angle
        updated_accel_x = dt_control / (dt_control + self._accel_time_constant) * (ideal_accel_x - accel) + accel
        updated_steering_angle = dt_control / (dt_control + self._steering_angle_time_constant) * (ideal_steering_angle - steering_angle) + steering_angle
        updated_steering_rate = (updated_steering_angle - steering_angle) / dt_control
        dynamic_state = DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist=state.car_footprint.rear_axle_to_center_dist, rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=StateVector2D(updated_accel_x, 0), tire_steering_rate=updated_steering_rate)
        propagating_state = EgoState(car_footprint=state.car_footprint, dynamic_car_state=dynamic_state, tire_steering_angle=state.tire_steering_angle, is_in_auto_mode=True, time_point=state.time_point)
        return propagating_state

    def propagate_state(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """Inherited, see super class."""
        propagating_state = self._update_commands(state, ideal_dynamic_state, sampling_time)
        state_dot = self.get_state_dot(propagating_state)
        next_x = forward_integrate(propagating_state.rear_axle.x, state_dot.rear_axle.x, sampling_time)
        next_y = forward_integrate(propagating_state.rear_axle.y, state_dot.rear_axle.y, sampling_time)
        next_heading = forward_integrate(propagating_state.rear_axle.heading, state_dot.rear_axle.heading, sampling_time)
        next_heading = principal_value(next_heading)
        next_point_velocity_x = forward_integrate(propagating_state.dynamic_car_state.rear_axle_velocity_2d.x, state_dot.dynamic_car_state.rear_axle_velocity_2d.x, sampling_time)
        next_point_velocity_y = 0.0
        next_point_tire_steering_angle = np.clip(forward_integrate(propagating_state.tire_steering_angle, state_dot.tire_steering_angle, sampling_time), -self._max_steering_angle, self._max_steering_angle)
        next_point_angular_velocity = next_point_velocity_x * np.tan(next_point_tire_steering_angle) / self._vehicle.wheel_base
        rear_axle_accel = [state_dot.dynamic_car_state.rear_axle_velocity_2d.x, state_dot.dynamic_car_state.rear_axle_velocity_2d.y]
        angular_accel = (next_point_angular_velocity - state.dynamic_car_state.angular_velocity) / sampling_time.time_s
        return EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(next_x, next_y, next_heading), rear_axle_velocity_2d=StateVector2D(next_point_velocity_x, next_point_velocity_y), rear_axle_acceleration_2d=StateVector2D(rear_axle_accel[0], rear_axle_accel[1]), tire_steering_angle=float(next_point_tire_steering_angle), time_point=propagating_state.time_point + sampling_time, vehicle_parameters=self._vehicle, is_in_auto_mode=True, angular_vel=next_point_angular_velocity, angular_accel=angular_accel, tire_steering_rate=state_dot.tire_steering_angle)

class LogFuturePlanner(AbstractPlanner):
    """
    Planner which just looks as future GT and returns it as a desired trajectory
    the input to this planner are detections.
    """
    requires_scenario: bool = True

    def __init__(self, scenario: AbstractScenario, num_poses: int, future_time_horizon: float):
        """
        Constructor of LogFuturePlanner.
        :param scenario: The scenario the planner is running on.
        :param num_poses: The number of poses to plan for.
        :param future_time_horizon: [s] The horizon length to plan for.
        """
        self._scenario = scenario
        self._num_poses = num_poses
        self._future_time_horizon = future_time_horizon
        self._trajectory: Optional[AbstractTrajectory] = None

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        pass

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        current_state = self._scenario.get_ego_state_at_iteration(current_input.iteration.index)
        try:
            states = self._scenario.get_ego_future_trajectory(current_input.iteration.index, self._future_time_horizon, self._num_poses)
            self._trajectory = InterpolatedTrajectory(list(itertools.chain([current_state], states)))
        except AssertionError:
            logger.warning('Cannot retrieve future ego trajectory. Using previous computed trajectory.')
            if self._trajectory is None:
                raise RuntimeError('Future ego trajectory cannot be retrieved from the scenario!')
        return self._trajectory

class MockIDMPlanner(AbstractIDMPlanner):
    """
    Mock IDMPlanner class for testing the AbstractIDMPlanner interface
    """
    requires_scenario: bool = False

    def __init__(self, target_velocity: float, min_gap_to_lead_agent: float, headway_time: float, accel_max: float, decel_max: float, planned_trajectory_samples: int, planned_trajectory_sample_interval: float, occupancy_map_radius: float):
        """
        Constructor for IDMPlanner
        :param target_velocity: [m/s] Desired velocity in free traffic.
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle.
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front.
        :param accel_max: [m/s^2] maximum acceleration.
        :param decel_max: [m/s^2] maximum deceleration (positive value).
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param occupancy_map_radius: [m] The range around the ego to add objects to be considered.
        """
        super(MockIDMPlanner, self).__init__(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max, planned_trajectory_samples, planned_trajectory_sample_interval, occupancy_map_radius)
        self._scenario = MockAbstractScenario()
        self._scenario_buffer = 10

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        self._map_api = initialization[0].map_api
        self._initialize_route_plan(initialization[0].route_roadblock_ids)
        self._ego_path: InterpolatedPath = create_path_from_ego_state(self._scenario.get_ego_future_trajectory(0, self._scenario_buffer, 10))
        self._ego_path_linestring = LineString()

    def compute_planner_trajectory(self, current_input: List[PlannerInput]) -> List[AbstractTrajectory]:
        """Inherited, see superclass."""
        return [InterpolatedTrajectory(self._ego_path.get_sampled_path())]

class AbstractMLAgents(AbstractObservation):
    """
    Simulate agents based on an ML model.
    """

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario) -> None:
        """
        Initializes the AbstractEgoCentricMLAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        self._model_loader = ModelLoader(model)
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval_us = model.future_trajectory_sampling.step_time * 1000000.0
        self._num_output_dim = model.future_trajectory_sampling.num_poses
        self._scenario = scenario
        self._ego_anchor_state = scenario.initial_ego_state
        self.step_time = None
        self._agents: Optional[Dict[str, TrackedObject]] = None

    @abstractmethod
    def _infer_model(self, features: FeaturesType) -> TargetsType:
        """
        Makes a single inference on a Pytorch/Torchscript model.
        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        pass

    @abstractmethod
    def _update_observation_with_predictions(self, agent_predictions: TargetsType) -> None:
        """
        Update smart agent using the predictions from the ML model
        :param agent_predictions: The prediction output from the ML_model
        """
        pass

    def _initialize_agents(self) -> None:
        """
        Initializes the agents based on the first step of the scenario
        """
        unique_agents = {tracked_object.track_token: tracked_object for tracked_object in self._scenario.initial_tracked_objects.tracked_objects if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE}
        self._agents = sort_dict(unique_agents)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._initialize_agents()

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._initialize_agents()
        self._model_loader.initialize()

    def update_observation(self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer) -> None:
        """Inherited, see superclass."""
        self.step_time = next_iteration.time_point - iteration.time_point
        self._ego_anchor_state, _ = history.current_state
        initialization = PlannerInitialization(mission_goal=self._scenario.get_mission_goal(), route_roadblock_ids=self._scenario.get_route_roadblock_ids(), map_api=self._scenario.map_api)
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(next_iteration.index)
        current_input = PlannerInput(next_iteration, history, traffic_light_data)
        features = self._model_loader.build_features(current_input, initialization)
        predictions = self._infer_model(features)
        self._update_observation_with_predictions(predictions)

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert self._agents, 'ML agent observation has not been initialized!Please make sure initialize() is called before getting the observation.'
        return DetectionsTracks(TrackedObjects(list(self._agents.values())))

class IDMAgentManager:
    """IDM smart-agents manager."""

    def __init__(self, agents: UniqueIDMAgents, agent_occupancy: OccupancyMap, map_api: AbstractMap):
        """
        Constructor for IDMAgentManager.
        :param agents: A dictionary pairing the agent's token to it's IDM representation.
        :param agent_occupancy: An occupancy map describing the spatial relationship between agents.
        :param map_api: AbstractMap API
        """
        self.agents: UniqueIDMAgents = agents
        self.agent_occupancy = agent_occupancy
        self._map_api = map_api

    def propagate_agents(self, ego_state: EgoState, tspan: float, iteration: int, traffic_light_status: Dict[TrafficLightStatusType, List[str]], open_loop_detections: List[TrackedObject], radius: float) -> None:
        """
        Propagate each active agent forward in time.

        :param ego_state: the ego's current state in the simulation.
        :param tspan: the interval of time to simulate.
        :param iteration: the simulation iteration.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :param open_loop_detections: A list of open loop detections the IDM agents should be responsive to.
        :param radius: [m] The radius around the ego state
        """
        self.agent_occupancy.set('ego', ego_state.car_footprint.geometry)
        track_ids = []
        for track in open_loop_detections:
            track_ids.append(track.track_token)
            self.agent_occupancy.insert(track.track_token, track.box.geometry)
        self._filter_agents_out_of_range(ego_state, radius)
        for agent_token, agent in self.agents.items():
            if agent.is_active(iteration) and agent.has_valid_path():
                agent.plan_route(traffic_light_status)
                stop_lines = self._get_relevant_stop_lines(agent, traffic_light_status)
                inactive_stop_line_tokens = self._insert_stop_lines_into_occupancy_map(stop_lines)
                agent_path = path_to_linestring(agent.get_path_to_go())
                intersecting_agents = self.agent_occupancy.intersects(agent_path.buffer(agent.width / 2, cap_style=CAP_STYLE.flat))
                assert intersecting_agents.contains(agent_token), "Agent's baseline does not intersect the agent itself"
                if intersecting_agents.size > 1:
                    nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(agent_token)
                    agent_heading = agent.to_se2().heading
                    if 'ego' in nearest_id:
                        ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
                        longitudinal_velocity = np.hypot(ego_velocity.x, ego_velocity.y)
                        relative_heading = ego_state.rear_axle.heading - agent_heading
                    elif 'stop_line' in nearest_id:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0
                    elif nearest_id in self.agents:
                        nearest_agent = self.agents[nearest_id]
                        longitudinal_velocity = nearest_agent.velocity
                        relative_heading = nearest_agent.to_se2().heading - agent_heading
                    else:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0
                    relative_heading = principal_value(relative_heading)
                    projected_velocity = rotate_angle(StateSE2(longitudinal_velocity, 0, 0), relative_heading).x
                    length_rear = 0
                else:
                    projected_velocity = 0.0
                    relative_distance = agent.get_progress_to_go()
                    length_rear = agent.length / 2
                agent.propagate(IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=length_rear), tspan)
                self.agent_occupancy.set(agent_token, agent.projected_footprint)
                self.agent_occupancy.remove(inactive_stop_line_tokens)
        self.agent_occupancy.remove(track_ids)

    def get_active_agents(self, iteration: int, num_samples: int, sampling_time: float) -> DetectionsTracks:
        """
        Returns all agents as DetectionsTracks.
        :param iteration: the current simulation iteration.
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: agents as DetectionsTracks.
        """
        return DetectionsTracks(TrackedObjects([agent.get_agent_with_planned_trajectory(num_samples, sampling_time) for agent in self.agents.values() if agent.is_active(iteration)]))

    def _filter_agents_out_of_range(self, ego_state: EgoState, radius: float=100) -> None:
        """
        Filter out agents that are out of range.
        :param ego_state: The ego state used as the center of the given radius
        :param radius: [m] The radius around the ego state
        """
        if len(self.agents) == 0:
            return
        agents: npt.NDArray[np.int32] = np.array([agent.to_se2().point.array for agent in self.agents.values()])
        distances = cdist(np.expand_dims(ego_state.center.point.array, axis=0), agents)
        remove_indices = np.argwhere(distances.flatten() > radius)
        remove_tokens = np.array(list(self.agents.keys()))[remove_indices.flatten()]
        self.agent_occupancy.remove(remove_tokens)
        for token in remove_tokens:
            self.agents.pop(token)

    def _get_relevant_stop_lines(self, agent: IDMAgent, traffic_light_status: Dict[TrafficLightStatusType, List[str]]) -> List[StopLine]:
        """
        Retrieve the stop lines that are affecting the given agent.
        :param agent: The IDM agent of interest.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :return: A list of stop lines associated with the given traffic light status.
        """
        relevant_lane_connectors = list({segment.id for segment in agent.get_route()} & set(traffic_light_status[TrafficLightStatusType.RED]))
        lane_connectors = [self._map_api.get_map_object(lc_id, SemanticMapLayer.LANE_CONNECTOR) for lc_id in relevant_lane_connectors]
        return [stop_line for lc in lane_connectors if lc for stop_line in lc.stop_lines]

    def _insert_stop_lines_into_occupancy_map(self, stop_lines: List[StopLine]) -> List[str]:
        """
        Insert stop lines into the occupancy map.
        :param stop_lines: A list of stop lines to be inserted.
        :return: A list of token corresponding to the inserted stop lines.
        """
        stop_line_tokens: List[str] = []
        for stop_line in stop_lines:
            stop_line_token = f'stop_line_{stop_line.id}'
            if not self.agent_occupancy.contains(stop_line_token):
                self.agent_occupancy.set(stop_line_token, stop_line.polygon)
                stop_line_tokens.append(stop_line_token)
        return stop_line_tokens

@dataclass
class PredictedTrajectory:
    """Stores a predicted trajectory, along with its probability."""
    probability: float
    waypoints: List[Optional[WaypointTypes]]

    @property
    def valid_waypoints(self) -> List[WaypointTypes]:
        """
        Interface to get only valid waypoints
        :return: waypoints which are not None
        """
        return [w for w in self.waypoints if w]

    @cached_property
    def trajectory(self) -> AbstractTrajectory:
        """
        Interface to compute trajectory from waypoints
        :return: trajectory from waypoints
        """
        return InterpolatedTrajectory(self.valid_waypoints)

    def __len__(self) -> int:
        """
        :return: number of waypoints in trajectory
        """
        return len(self.waypoints)

class TestInterpolatedTrajectory(unittest.TestCase):
    """Tests implementation of InterpolatedTrajectory."""

    def setUp(self) -> None:
        """Test setup."""
        self.split_state_1 = Mock(linear_states=[123], angular_states=[2.13], fixed_states=['fix'], autspec=SplitState)
        self.split_state_2 = Mock(linear_states=[456], angular_states=[3.13], fixed_states=['fix'], autspec=SplitState)
        self.start_time_point = TimePoint(0)
        self.end_time_point = TimePoint(int(1000000.0))
        self.points = [MagicMock(time_point=self.start_time_point, time_us=self.start_time_point.time_us, to_split_state=lambda: self.split_state_1, spec=MockPoint), MagicMock(time_point=self.end_time_point, time_us=self.end_time_point.time_us, to_split_state=lambda: self.split_state_2, spec=MockPoint)]
        self.trajectory = InterpolatedTrajectory(self.points)

    def tearDown(self) -> None:
        """Resets mock objects."""
        MockPoint.reset_calls()

    @patch('nuplan.planning.simulation.trajectory.interpolated_trajectory.sp_interp')
    @patch('nuplan.planning.simulation.trajectory.interpolated_trajectory.np')
    @patch('nuplan.planning.simulation.trajectory.interpolated_trajectory.AngularInterpolator', autospec=True)
    def test_initialization(self, mock_interp_angular: Mock, mock_np: Mock, mock_sp_interp: Mock) -> None:
        """Tests that initialization works as intended."""
        mock_sp_interp.interp1d.return_value = 'interp_function'
        mock_np.array.return_value = 'array'
        trajectory = InterpolatedTrajectory(self.points)
        self.assertEqual(trajectory._trajectory_class, MockPoint)
        self.assertEqual(trajectory._fixed_state, ['fix'])
        mock_sp_interp.interp1d.assert_called_with([0, 1000000], mock_np.array.return_value, axis=0)
        self.assertEqual(trajectory._function_interp_linear, mock_sp_interp.interp1d.return_value)
        mock_interp_angular.assert_called_with([0, 1000000], 'array')
        self.assertEqual(trajectory._angular_interpolator, mock_interp_angular.return_value)
        with self.assertRaises(AssertionError):
            InterpolatedTrajectory([MagicMock()])

    def test_start_end_time(self) -> None:
        """Tests that properties return correct members."""
        self.assertEqual(self.start_time_point, self.trajectory.start_time)
        self.assertEqual(self.end_time_point, self.trajectory.end_time)

    def test_get_state_at_time(self) -> None:
        """Tests interpolation method."""
        time_point = TimePoint(int(0.5 * 1000000.0))
        state = self.trajectory.get_state_at_time(time_point)
        self.assertEqual('foo', state)
        interpolated_state = SplitState(linear_states=[289.5], angular_states=[2.63], fixed_states=['fix'])
        self.assertEqual(MockPoint.from_split_state.calls, [interpolated_state])
        time_point_outside_interval = TimePoint(int(5 * 1000000.0))
        with self.assertRaises(AssertionError):
            self.trajectory.get_state_at_time(time_point_outside_interval)

    def test_get_state_at_times(self) -> None:
        """Tests batch interpolation method."""
        time_points = [TimePoint(0), TimePoint(int(0.5 * 1000000.0))]
        states = self.trajectory.get_state_at_times(time_points)
        self.assertEqual(['foo', 'foo'], states)
        initial_state = SplitState(linear_states=[123], angular_states=[2.13], fixed_states=['fix'])
        interpolated_state = SplitState(linear_states=[289.5], angular_states=[2.63], fixed_states=['fix'])
        self.assertEqual(MockPoint.from_split_state.calls, [initial_state, interpolated_state])
        time_point_outside_interval = TimePoint(int(5 * 1000000.0))
        with self.assertRaises(AssertionError):
            self.trajectory.get_state_at_times([time_point_outside_interval])

    def test_get_sampled_trajectory(self) -> None:
        """Tests getter for entire trajectory."""
        self.assertEqual(self.points, self.trajectory.get_sampled_trajectory())

class TestSimulationHistory(TestCase):
    """Tests for SimulationHistory buffer."""

    def setUp(self) -> None:
        """
        Setup mocks for the tests
        """
        self.map = MagicMock(spec=AbstractMap)
        self.se2 = MagicMock(spec=StateSE2)
        self.sample = MagicMock(spec=SimulationHistorySample)
        self.sh = SimulationHistory(self.map, self.se2)

    def test_init(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        self.assertEqual(self.sh.map_api, self.map)
        self.assertEqual(self.sh.mission_goal, self.se2)
        self.assertEqual(self.sh.data, [])

    def test_add_sample(self) -> None:
        """
        Test if the add_sample method adds the passed sample to the data list.
        """
        with patch.object(self.sh, 'data', append=MagicMock()) as data:
            self.sh.add_sample(self.sample)
            data.append.assert_called_once_with(self.sample)

    def test_last(self) -> None:
        """Test if the last method works as expected."""
        self.sh.data = None
        with self.assertRaises(RuntimeError):
            self.sh.last()
        self.sh.data = [self.sample, self.sample]
        self.assertEqual(self.sh.last(), self.sample)

    def test_extract_ego_state(self) -> None:
        """Test if the extract_ego_state property works as expected."""
        with patch('nuplan.planning.simulation.history.simulation_history.SimulationHistorySample', autospec=True):
            mock_data = [SimulationHistorySample(iteration=MagicMock(), ego_state=MagicMock(side_effect=lambda: i), trajectory=MagicMock(), observation=MagicMock(), traffic_light_status=MagicMock()) for i in range(DATA_LEN)]
            self.sh.data = mock_data
            ego_states = self.sh.extract_ego_state
            self.assertTrue([ego_states[i]() == i for i in range(len(ego_states))])

    def test_clear(self) -> None:
        """
        Tests if the clear method clears the data list.
        """
        with patch.object(self.sh, 'data', clear=MagicMock()) as data:
            self.sh.reset()
            data.clear.assert_called_once()

    @patch('nuplan.planning.simulation.history.simulation_history.len', return_value=DATA_LEN)
    def test_len(self, len_mock: MagicMock) -> None:
        """
        Tests if the len method returns the length of the data list.
        """
        result = len(self.sh)
        self.assertEqual(result, DATA_LEN)
        len_mock.assert_called_once_with(self.sh.data)

    def test_interval_seconds(self) -> None:
        """Tests for the correct behavior of the interval_seconds property."""
        self.sh.data = None
        with self.assertRaises(ValueError):
            self.sh.interval_seconds
        self.sh.data = []
        with self.assertRaises(ValueError):
            self.sh.interval_seconds
        with patch('nuplan.planning.simulation.history.simulation_history.SimulationHistorySample', autospec=True):
            self.sh.data = [self.sample]
            with self.assertRaises(ValueError):
                self.sh.interval_seconds
            mock_data = [SimulationHistorySample(iteration=SimulationIteration(index=0, time_point=TimePoint(i * 1000.0)), ego_state=MagicMock(), trajectory=MagicMock(), observation=MagicMock(), traffic_light_status=MagicMock()) for i in range(DATA_LEN)]
            self.sh.data = mock_data
            expected_interval_seconds = mock_data[1].iteration.time_s - mock_data[0].iteration.time_s
            self.assertEqual(expected_interval_seconds, self.sh.interval_seconds)

class TestSimulationHistoryBuffer(unittest.TestCase):
    """Test suite for SimulationHistoryBuffer"""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = MockAbstractScenario(number_of_past_iterations=20)
        self.buffer_size = 10

    def test_initialize_with_box(self) -> None:
        """Test the initialize function"""
        tracks_observation = TracksObservation(self.scenario)
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(buffer_size=self.buffer_size, scenario=self.scenario, observation_type=tracks_observation.observation_type())
        self.assertEqual(len(history_buffer), self.buffer_size)

    def test_initialize_with_lidar_pc(self) -> None:
        """Test the initialize function"""
        lidar_pc_observation = LidarPcObservation(self.scenario)
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(buffer_size=self.buffer_size, scenario=self.scenario, observation_type=lidar_pc_observation.observation_type())
        self.assertEqual(len(history_buffer), self.buffer_size)

    def test_initialize_from_list(self) -> None:
        """Test the initialization from lists"""
        history_buffer = SimulationHistoryBuffer.initialize_from_list(buffer_size=self.buffer_size, ego_states=[self.scenario.initial_ego_state], observations=[self.scenario.initial_tracked_objects], sample_interval=0.05)
        self.assertEqual(len(history_buffer), 1)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state])
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects])

    def test_append(self) -> None:
        """Test the append function"""
        history_buffer = SimulationHistoryBuffer(ego_state_buffer=deque([Mock()], maxlen=1), observations_buffer=deque([Mock()], maxlen=1))
        history_buffer.append(self.scenario.initial_ego_state, self.scenario.initial_tracked_objects)
        self.assertEqual(len(history_buffer), 1)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state])
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects])

    def test_extend(self) -> None:
        """Test the extend function"""
        history_buffer = SimulationHistoryBuffer(ego_state_buffer=deque([Mock()], maxlen=2), observations_buffer=deque([Mock()], maxlen=2))
        history_buffer.extend([self.scenario.initial_ego_state] * 2, [self.scenario.initial_tracked_objects] * 2)
        self.assertEqual(len(history_buffer), 2)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state] * 2)
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects] * 2)

class TestMetricRunner(unittest.TestCase):
    """Tests MetricRunner class which is computing metric."""

    def setUp(self) -> None:
        """Setup Mock classes."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.scenario = MockAbstractScenario(number_of_past_iterations=10)
        self.history = SimulationHistory(self.scenario.map_api, self.scenario.get_mission_goal())
        state_0 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=self.scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(0))
        state_1 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=self.scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(1000))
        self.history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_0, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=self.scenario.get_traffic_light_status_at_iteration(0)))
        self.history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_1, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=self.scenario.get_traffic_light_status_at_iteration(0)))
        save_path = Path(self.tmp_dir.name)
        planner = SimplePlanner(2, 0.5, [0, 0])
        self.simulation_log = SimulationLog(file_path=save_path / 'simulation_logs', simulation_history=self.history, scenario=self.scenario, planner=planner)
        self.metric_engine = MetricsEngine(metrics=[], main_save_path=save_path / 'metrics')
        self.metric_callback = MetricCallback(metric_engine=self.metric_engine)
        self.metric_runner = MetricRunner(simulation_log=self.simulation_log, metric_callback=self.metric_callback)

    def tearDown(self) -> None:
        """Clean up folders."""
        self.tmp_dir.cleanup()

    def test_run_metric_runner(self) -> None:
        """Test to run metric_runner."""
        self.metric_runner.run()

def _save_log_to_file(file_name: pathlib.Path, scenario: AbstractScenario, planner: AbstractPlanner, history: SimulationHistory) -> None:
    """
    Create SimulationLog and save it to disk.
    :param file_name: to write to.
    :param scenario: to store in the log.
    :param planner: to store in the log.
    :param history: to store in the log.
    """
    simulation_log = SimulationLog(file_path=file_name, scenario=scenario, planner=planner, simulation_history=history)
    simulation_log.save_to_file()

class SimulationLogCallback(AbstractCallback):
    """
    Callback for simulation logging/object serialization to disk.
    """

    def __init__(self, output_directory: Union[str, pathlib.Path], simulation_log_dir: Union[str, pathlib.Path], serialization_type: str, worker_pool: Optional[WorkerPool]=None):
        """
        Construct simulation log callback.
        :param output_directory: where scenes should be serialized.
        :param simulation_log_dir: Folder where to save simulation logs.
        :param serialization_type: A way to serialize output, options: ["json", "pickle", "msgpack"].
        """
        available_formats = ['pickle', 'msgpack']
        if serialization_type not in available_formats:
            raise ValueError(f'The simulation log callback will not store files anywhere!Choose at least one format from {available_formats} instead of {serialization_type}!')
        self._output_directory = pathlib.Path(output_directory) / simulation_log_dir
        self._serialization_type = serialization_type
        if serialization_type == 'pickle':
            file_suffix = '.pkl.xz'
        elif serialization_type == 'msgpack':
            file_suffix = '.msgpack.xz'
        else:
            raise ValueError(f'Unknown option: {serialization_type}')
        self._file_suffix = file_suffix
        self._pool = worker_pool
        self._futures: List[Future[None]] = []

    @property
    def futures(self) -> List[Future[None]]:
        """
        Returns a list of futures, eg. for the main process to block on.
        :return: any futures generated by running any part of the callback asynchronously.
        """
        return self._futures

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Create directory at initialization
        :param setup: simulation setup
        :param planner: planner before initialization
        """
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        if not is_s3_path(scenario_directory):
            scenario_directory.mkdir(exist_ok=True, parents=True)

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """
        On reached_end validate that all steps were correctly serialized.
        :param setup: simulation setup.
        :param planner: planner when simulation ends.
        :param history: resulting from simulation.
        """
        number_of_scenes = len(history)
        if number_of_scenes == 0:
            raise RuntimeError('Number of scenes has to be greater than 0')
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        scenario = setup.scenario
        file_name = scenario_directory / (scenario.scenario_name + self._file_suffix)
        if self._pool is not None:
            self._futures = []
            self._futures.append(self._pool.submit(Task(_save_log_to_file, num_cpus=1, num_gpus=0), file_name, scenario, planner, history))
        else:
            _save_log_to_file(file_name, scenario, planner, history)

    def _get_scenario_folder(self, planner_name: str, scenario: AbstractScenario) -> pathlib.Path:
        """
        Compute scenario folder directory where all files will be stored.
        :param planner_name: planner name.
        :param scenario: for which to compute directory name.
        :return directory path.
        """
        return self._output_directory / planner_name / scenario.scenario_type / scenario.log_name / scenario.scenario_name

class SkeletonTestSerializationCallback(unittest.TestCase):
    """Base class for TestsSerializationCallback* classes."""

    def _setUp(self) -> None:
        """Setup mocks for our tests."""
        self._serialization_type_to_extension_map = {'json': '.json', 'pickle': '.pkl.xz', 'msgpack': '.msgpack.xz'}
        self._serialization_type = getattr(self, '_serialization_type', '')
        self.assertIn(self._serialization_type, self._serialization_type_to_extension_map)
        self.output_folder = tempfile.TemporaryDirectory()
        self.callback = SerializationCallback(output_directory=self.output_folder.name, folder_name='sim', serialization_type=self._serialization_type, serialize_into_single_file=True)
        self.sim_manager = Mock(spec=AbstractSimulationTimeController)
        self.observation = Mock(spec=AbstractObservation)
        self.controller = Mock(spec=AbstractEgoController)
        super().setUp()

    @settings(deadline=None)
    @given(mock_timestamp=st.one_of(st.just(0), st.integers(min_value=1627066061949808, max_value=18446744073709551615)))
    def _dump_test_scenario(self, mock_timestamp: int) -> None:
        """
        Tests whether a scene can be dumped into a file and check that the keys are in the dumped scene.
        :param mock_timestamp: Mocked timestamp to pass to mock_get_traffic_light_status_at_iteration.
        """

        def mock_get_traffic_light_status_at_iteration(iteration: int) -> Generator[TrafficLightStatusData, None, None]:
            """Mocks MockAbstractScenario.get_traffic_light_status_at_iteration to return large numbers."""
            dummy_tl_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=mock_timestamp)
            yield dummy_tl_data
        scenario = MockAbstractScenario()
        scenario.get_traffic_light_status_at_iteration = Mock(spec=scenario.get_traffic_light_status_at_iteration)
        scenario.get_traffic_light_status_at_iteration.side_effect = mock_get_traffic_light_status_at_iteration
        self.setup = SimulationSetup(observations=self.observation, scenario=scenario, time_controller=self.sim_manager, ego_controller=self.controller)
        planner = Mock()
        planner.name = Mock(return_value='DummyPlanner')
        directory = self.callback._get_scenario_folder(planner.name(), scenario)
        self.assertEqual(str(directory), self.output_folder.name + '/sim/DummyPlanner/mock_scenario_type/mock_log_name/mock_scenario_name')
        self.callback.on_initialization_start(self.setup, planner)
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
        state_0 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(0))
        state_1 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(1000))
        history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_0, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=scenario.get_traffic_light_status_at_iteration(0)))
        history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_1, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=scenario.get_traffic_light_status_at_iteration(0)))
        for data in history.data:
            self.callback.on_step_end(self.setup, planner, data)
        self.callback.on_simulation_end(self.setup, planner, history)
        filename = 'mock_scenario_name' + self._serialization_type_to_extension_map[self._serialization_type]
        path = pathlib.Path(self.output_folder.name + '/sim/DummyPlanner/mock_scenario_type/mock_log_name/mock_scenario_name/' + filename)
        self.assertTrue(path.exists())
        if self._serialization_type == 'json':
            with open(path.absolute()) as f:
                data = json.load(f)
        elif self._serialization_type == 'msgpack':
            with lzma.open(str(path), 'rb') as f:
                data = msgpack.unpackb(f.read())
        elif self._serialization_type == 'pickle':
            with lzma.open(str(path), 'rb') as f:
                data = pickle.load(f)
        self.assertTrue(len(data) > 0)
        data = data[0]
        self.assertTrue('world' in data.keys())
        self.assertTrue('ego' in data.keys())
        self.assertTrue('trajectories' in data.keys())
        self.assertTrue('map' in data.keys())
        expected_traffic_light_data = next(scenario.get_traffic_light_status_at_iteration(0))
        actual_traffic_light_data_dict = data['traffic_light_status'][0]
        self.assertEqual(actual_traffic_light_data_dict['timestamp'], expected_traffic_light_data.timestamp)

class TestMetricCallback(TestCase):
    """Tests metrics callback."""

    def setUp(self) -> None:
        """
        Setup mocks for the tests
        """
        self.mock_metric_engine = Mock(spec=MetricsEngine)
        self.mock_metric_engine.compute = Mock(return_value=METRICS_LIST)
        self.mock_setup = Mock()
        self.mock_planner = Mock(spec=AbstractPlanner)
        self.mock_planner.name = Mock(return_value=PLANNER_NAME)
        self.mock_history = Mock()
        return super().setUp()

    def test_metric_callback_init(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        mc = MetricCallback(self.mock_metric_engine)
        self.assertEqual(mc._metric_engine, self.mock_metric_engine)

    @patch('nuplan.planning.simulation.callback.metric_callback.logger')
    def test_on_simulation_end(self, logger: MagicMock) -> None:
        """
        Tests if the metric engine compute is called with the correct parameters.
        Tests if the metric engine save_metric_files is called with compute's result.
        Tests if the logger is called with the correct parameters.
        """
        mc = MetricCallback(self.mock_metric_engine)
        mc.on_simulation_end(self.mock_setup, self.mock_planner, self.mock_history)
        logger.debug.assert_has_calls([call('Starting metrics computation...'), call('Finished metrics computation!'), call('Saving metric statistics!'), call('Saved metrics!')])
        self.mock_planner.name.assert_called_once()
        self.mock_metric_engine.compute.assert_called_once_with(self.mock_history, scenario=self.mock_setup.scenario, planner_name=PLANNER_NAME)
        self.mock_metric_engine.write_to_files.assert_called_once_with(METRICS_LIST)

def objects_are_equal(a: object, b: object) -> bool:
    """
    Recursively checks if two objects are equal by value.

    This method supports objects that are compositions of:
        * built-in types (int, float, bool, etc)
        * callable objects
        * numpy arrays
        * objects supporting `__dict__`
        * compositions of the above objects

    Other types are currently unsupported.

    :param a: a in a == b, must implement __dict__ or be directly comparable.
    :param b: b in a == b, must implement __dict__ or be directly comparable.
    :return: true if both objects are the same, otherwise false.
    """
    if not hasattr(a, '__dict__') and (not hasattr(b, '__dict__')):
        return a == b
    a_dict = a.__dict__
    b_dict = b.__dict__
    if set(a_dict.keys()) != set(b_dict.keys()):
        return False
    for key in a_dict:
        if type(a_dict[key]) != type(b_dict[key]):
            return False
        if callable(a_dict[key]):
            if not callable_name_matches(a_dict[key], b_dict[key]):
                return False
        elif hasattr(a_dict[key], '__dict__'):
            if not objects_are_equal(a_dict[key], b_dict[key]):
                return False
        elif isinstance(a_dict[key], np.ndarray):
            if not np.allclose(a_dict[key], b_dict[key]):
                return False
        elif hasattr(a_dict[key], '__iter__'):
            if not iterator_is_equal(a_dict[key], b_dict[key]):
                return False
        else:
            return objects_are_equal(a_dict[key], b_dict[key])
    return True

def callable_name_matches(a: Callable[..., Any], b: Callable[..., Any]) -> bool:
    """
    Checks that callable names match.
    :param a: first callable to compare.
    :param b: second callable to compare.
    :return: true if the names match, otherwise false.
    """
    if hasattr(a, '__name__'):
        if a.__name__ != b.__name__:
            return False
    elif 'object at' in (a_repr := repr(a)):
        address_ind = a_repr.index('object at')
        a_name = a_repr[1:address_ind - 1]
        b_name = repr(b)[1:address_ind - 1]
        if a_name != b_name:
            return False
    else:
        raise NotImplementedError
    return True

def iterator_is_equal(a: Iterable[Any], b: Iterable[Any]) -> bool:
    """
    Checks that two iterables are equal by value.
    :param a: a in a == b.
    :param b: b in a == b.
    :return: true if the iterable contents match.
    """
    for a_item, b_item in zip((a_iter := iter(a)), (b_iter := iter(b))):
        if not objects_are_equal(a_item, b_item):
            return False
    try:
        next(a_iter)
        return False
    except StopIteration:
        try:
            next(b_iter)
            return False
        except StopIteration:
            return True

class TestSimulationLogCallback(unittest.TestCase):
    """Tests simulation_log_callback."""

    def setUp(self) -> None:
        """Setup Mocked classes."""
        self.output_folder = tempfile.TemporaryDirectory()
        self.callback = SimulationLogCallback(output_directory=self.output_folder.name, simulation_log_dir='simulation_log', serialization_type='msgpack')
        self.sim_manager = Mock(spec=AbstractSimulationTimeController)
        self.observation = Mock(spec=AbstractObservation)
        self.controller = Mock(spec=AbstractEgoController)

    def tearDown(self) -> None:
        """Clean up folder."""
        self.output_folder.cleanup()

    def test_callback(self) -> None:
        """
        Tests whether a scene can be dumped into a simulation log, checks that the keys are correct,
        and checks that the log contains the expected data after being re-loaded from disk.
        """
        scenario = MockAbstractScenario()
        self.setup = SimulationSetup(observations=self.observation, scenario=scenario, time_controller=self.sim_manager, ego_controller=self.controller)
        planner = SimplePlanner(2, 0.5, [0, 0])
        directory = self.callback._get_scenario_folder(planner.name(), scenario)
        self.assertEqual(str(directory), self.output_folder.name + '/simulation_log/SimplePlanner/mock_scenario_type/mock_log_name/mock_scenario_name')
        self.callback.on_initialization_start(self.setup, planner)
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
        state_0 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(0))
        state_1 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(1000))
        history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_0, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0))))
        history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_1, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0))))
        for data in history.data:
            self.callback.on_step_end(self.setup, planner, data)
        self.callback.on_simulation_end(self.setup, planner, history)
        path = pathlib.Path(self.output_folder.name + '/simulation_log/SimplePlanner/mock_scenario_type/mock_log_name/mock_scenario_name/mock_scenario_name.msgpack.xz')
        self.assertTrue(path.exists())
        simulation_log = SimulationLog.load_data(file_path=path)
        self.assertEqual(simulation_log.file_path, path)
        self.assertTrue(objects_are_equal(simulation_log.simulation_history, history))

def filter_agents(tracked_objects_history: List[TrackedObjects], reverse: bool=False, allowable_types: Optional[Set[TrackedObjectType]]=None) -> List[TrackedObjects]:
    """
    Filter detections to keep only agents of specified types which appear in the first frame (or last frame if reverse=True)
    :param tracked_objects_history: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the last element in the list will be used as the filter
    :param allowable_types: TrackedObjectTypes to filter for (optional: defaults to VEHICLE)
    :return: filtered agents in the same format [num_frames, num_agents]
    """
    if allowable_types is None:
        allowable_types = {TrackedObjectType.VEHICLE}
    if reverse:
        agent_tokens = [box.track_token for object_type in allowable_types for box in tracked_objects_history[-1].get_tracked_objects_of_type(object_type)]
    else:
        agent_tokens = [box.track_token for object_type in allowable_types for box in tracked_objects_history[0].get_tracked_objects_of_type(object_type)]
    filtered_agents = [TrackedObjects([agent for object_type in allowable_types for agent in tracked_objects.get_tracked_objects_of_type(object_type) if agent.track_token in agent_tokens]) for tracked_objects in tracked_objects_history]
    return filtered_agents

def _create_tracked_objects(num_frames: int, num_agents: int, object_type: TrackedObjectType=TrackedObjectType.VEHICLE) -> List[TrackedObjects]:
    """
    Generate dummy agent trajectories
    :param num_frames: length of the trajectory to be generate
    :param num_agents: number of agents to generate
    :param object_type: agent type.
    :return: agent trajectories [num_frames, num_agents, 1]
    """
    return [TrackedObjects([_create_scene_object(str(num), object_type) for num in range(num_agents)]) for _ in range(num_frames)]

class TestAgentsFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs features with vectorized agent information."""

    def setUp(self) -> None:
        """Set up test case."""
        self.num_frames = 8
        self.num_agents = 10
        self.num_missing_agents = 2
        self.agent_trajectories = [*_create_tracked_objects(5, self.num_agents), *_create_tracked_objects(3, self.num_agents - self.num_missing_agents)]
        self.time_stamps = [TimePoint(step) for step in range(self.num_frames)]

    def test_build_ego_features(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        ego_trajectory = _create_ego_trajectory(num_frames)
        ego_features = build_ego_features(ego_trajectory)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(np.allclose(ego_features[0], np.array([0, 0, 0])))
        ego_features_reversed = build_ego_features(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(np.allclose(ego_features_reversed[-1], np.array([0, 0, 0])))

    def test_extract_and_pad_agent_poses(self) -> None:
        """
        Test when there is agent pose trajectory is incomplete
        """
        padded_poses, availability = extract_and_pad_agent_poses(self.agent_trajectories)
        availability = np.asarray(availability)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses])
        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)
        self.assertEqual(len(availability.shape), 2)
        self.assertEqual(availability.shape[0], self.num_frames)
        self.assertEqual(availability.shape[1], self.num_agents)
        self.assertTrue(availability[:5, :].all())
        self.assertTrue(availability[:, :self.num_agents - self.num_missing_agents].all())
        self.assertTrue((~availability[5:, -self.num_missing_agents:]).all())
        padded_poses_reversed, availability_reversed = extract_and_pad_agent_poses(self.agent_trajectories[::-1], reverse=True)
        availability_reversed = np.asarray(availability_reversed)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses_reversed])
        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)
        self.assertEqual(len(availability_reversed.shape), 2)
        self.assertEqual(availability_reversed.shape[0], self.num_frames)
        self.assertEqual(availability_reversed.shape[1], self.num_agents)
        self.assertTrue(availability_reversed[-5:, :].all())
        self.assertTrue(availability_reversed[:, :self.num_agents - self.num_missing_agents].all())
        self.assertTrue((~availability_reversed[:3, -self.num_missing_agents:]).all())

    def test_extract_and_pad_agent_sizes(self) -> None:
        """
        Test when there is agent size trajectory is incomplete
        """
        padded_sizes, _ = extract_and_pad_agent_sizes(self.agent_trajectories)
        stacked_sizes = np.stack(padded_sizes)
        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)
        padded_sizes_reversed, _ = extract_and_pad_agent_sizes(self.agent_trajectories[::-1], reverse=True)
        stacked_sizes = np.stack(padded_sizes_reversed)
        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)

    def test_extract_and_pad_agent_velocities(self) -> None:
        """
        Test when there is agent velocity trajectory is incomplete
        """
        padded_velocities, _ = extract_and_pad_agent_velocities(self.agent_trajectories)
        stacked_velocities = np.stack([[agent.serialize() for agent in frame] for frame in padded_velocities])
        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)
        padded_velocities_reversed, _ = extract_and_pad_agent_velocities(self.agent_trajectories[::-1], reverse=True)
        stacked_velocities = np.stack([[agent.serialize() for agent in frame] for frame in padded_velocities_reversed])
        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)

    def test_compute_yaw_rate_from_states(self) -> None:
        """
        Test computing yaw from the agent pose trajectory
        """
        padded_poses, _ = extract_and_pad_agent_poses(self.agent_trajectories)
        yaw_rates = compute_yaw_rate_from_states(padded_poses, self.time_stamps)
        self.assertEqual(yaw_rates.transpose().shape[0], self.num_frames)
        self.assertEqual(yaw_rates.transpose().shape[1], self.num_agents)

    def test_filter_agents(self) -> None:
        """
        Test agent filtering
        """
        num_frames = 8
        num_agents = 5
        missing_agents = 2
        tracked_objects_history = [*_create_tracked_objects(num_frames=5, num_agents=num_agents, object_type=TrackedObjectType.VEHICLE), *_create_tracked_objects(num_frames=2, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.BICYCLE), *_create_tracked_objects(num_frames=1, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.VEHICLE)]
        filtered_agents = filter_agents(tracked_objects_history)
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[0].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)
        filtered_agents = filter_agents(tracked_objects_history, reverse=True)
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[-1].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)
        tracked_objects_history = [*_create_tracked_objects(num_frames=5, num_agents=num_agents, object_type=TrackedObjectType.BICYCLE), *_create_tracked_objects(num_frames=2, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.VEHICLE), *_create_tracked_objects(num_frames=1, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.BICYCLE)]
        filtered_agents = filter_agents(tracked_objects_history, allowable_types=[TrackedObjectType.BICYCLE])
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[0].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)

    def test_build_ego_features_from_tensor(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        zeros = torch.tensor([0, 0, 0], dtype=torch.float32)
        ego_trajectory = _create_ego_trajectory_tensor(num_frames)
        ego_features = build_ego_features_from_tensor(ego_trajectory)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(torch.allclose(ego_features[0], zeros, atol=1e-07))
        ego_features_reversed = build_ego_features_from_tensor(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(torch.allclose(ego_features_reversed[-1], zeros, atol=1e-07))

    def test_build_generic_ego_features_from_tensor(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        ego_trajectory = _create_ego_trajectory_tensor(num_frames)
        ego_features = build_generic_ego_features_from_tensor(ego_trajectory)
        self.assertEqual((num_frames, GenericEgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(torch.allclose(ego_features[0], zeros, atol=1e-07))
        ego_features_reversed = build_generic_ego_features_from_tensor(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, GenericEgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(torch.allclose(ego_features_reversed[-1], zeros, atol=1e-07))

    def test_convert_absolute_quantities_to_relative(self) -> None:
        """
        Test the conversion routine between absolute and relative quantities
        """

        def get_dummy_states() -> List[torch.Tensor]:
            """
            Create a series of dummy agent tensors
            """
            dummy_agent_state = _create_tracked_object_agent_tensor(7)
            dummy_states = [dummy_agent_state + i for i in range(5)]
            return dummy_states
        zeros = torch.tensor([0, 0, 0], dtype=torch.float32)
        dummy_states = get_dummy_states()
        ego_pose = torch.tensor([4, 4, 4, 2, 2, 2, 2], dtype=torch.float32)
        transformed = convert_absolute_quantities_to_relative(dummy_states, ego_pose)
        for i in range(0, len(transformed), 1):
            should_be_zero_row = 4 - i
            check_tensor = torch.tensor([transformed[i][should_be_zero_row, AgentInternalIndex.x()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.y()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.heading()].item()], dtype=torch.float32)
            self.assertTrue(torch.allclose(check_tensor, zeros, atol=1e-07))
        dummy_states = get_dummy_states()
        ego_pose = torch.tensor([2, 2, 4, 4, 4, 4, 4], dtype=torch.float32)
        transformed = convert_absolute_quantities_to_relative(dummy_states, ego_pose)
        for i in range(0, len(transformed), 1):
            should_be_zero_row = 4 - i
            check_tensor = torch.tensor([transformed[i][should_be_zero_row, AgentInternalIndex.vx()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.vy()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.heading()].item()], dtype=torch.float32)
            self.assertTrue(torch.allclose(check_tensor, zeros, atol=1e-07))

    def test_pad_agent_states(self) -> None:
        """
        Test the pad agent states functionality
        """
        forward_dummy_states = [_create_tracked_object_agent_tensor(7), _create_tracked_object_agent_tensor(5), _create_tracked_object_agent_tensor(6)]
        padded = pad_agent_states(forward_dummy_states, reverse=False)
        self.assertTrue(len(padded) == 3)
        self.assertEqual((7, AgentInternalIndex.dim()), padded[0].shape)
        for i in range(1, len(padded)):
            self.assertTrue(torch.allclose(padded[0], padded[i]))
        backward_dummy_states = [_create_tracked_object_agent_tensor(6), _create_tracked_object_agent_tensor(5), _create_tracked_object_agent_tensor(7)]
        padded_reverse = pad_agent_states(backward_dummy_states, reverse=True)
        self.assertTrue(len(padded_reverse) == 3)
        self.assertEqual((7, AgentInternalIndex.dim()), padded_reverse[2].shape)
        for i in range(0, len(padded_reverse) - 1):
            self.assertTrue(torch.allclose(padded_reverse[2], padded_reverse[i]))

    def test_compute_yaw_rate_from_state_tensors(self) -> None:
        """
        Test compute yaw rate functionality
        """
        num_frames = 6
        num_agents = 5
        agent_states = [_create_tracked_object_agent_tensor(num_agents) + i for i in range(num_frames)]
        time_stamps = torch.tensor([int(i * 1000000.0) for i in range(num_frames)], dtype=torch.int64)
        yaw_rate = compute_yaw_rate_from_state_tensors(agent_states, time_stamps)
        self.assertEqual((num_frames, num_agents), yaw_rate.shape)
        self.assertTrue(torch.allclose(torch.ones((num_frames, num_agents), dtype=torch.float64), yaw_rate))

    def test_filter_agents_tensor(self) -> None:
        """
        Test filter agents
        """
        dummy_states = [_create_tracked_object_agent_tensor(7), _create_tracked_object_agent_tensor(8), _create_tracked_object_agent_tensor(6)]
        filtered = filter_agents_tensor(dummy_states, reverse=False)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered[0].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered[1].shape)
        self.assertEqual((6, AgentInternalIndex.dim()), filtered[2].shape)
        dummy_states = [_create_tracked_object_agent_tensor(6), _create_tracked_object_agent_tensor(8), _create_tracked_object_agent_tensor(7)]
        filtered_reverse = filter_agents_tensor(dummy_states, reverse=True)
        self.assertEqual((6, AgentInternalIndex.dim()), filtered_reverse[0].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered_reverse[1].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered_reverse[2].shape)

    def test_sampled_past_ego_states_to_tensor(self) -> None:
        """
        Test the conversion routine to convert ego states to tensors.
        """
        num_egos = 6
        test_egos = []
        for i in range(num_egos):
            footprint = CarFootprint(center=StateSE2(x=i, y=i, heading=i), vehicle_parameters=VehicleParameters(vehicle_name='vehicle_name', vehicle_type='vehicle_type', width=i, front_length=i, rear_length=i, cog_position_from_rear_axle=i, wheel_base=i, height=i))
            dynamic_car_state = DynamicCarState(rear_axle_to_center_dist=i, rear_axle_velocity_2d=StateVector2D(x=i + 5, y=i + 5), rear_axle_acceleration_2d=StateVector2D(x=i, y=i), angular_velocity=i, angular_acceleration=i, tire_steering_rate=i)
            test_ego = EgoState(car_footprint=footprint, dynamic_car_state=dynamic_car_state, tire_steering_angle=i, is_in_auto_mode=i, time_point=TimePoint(time_us=i))
            test_egos.append(test_ego)
        tensor = sampled_past_ego_states_to_tensor(test_egos)
        self.assertEqual((6, EgoInternalIndex.dim()), tensor.shape)
        for i in range(0, tensor.shape[0], 1):
            ego = test_egos[i]
            self.assertEqual(ego.rear_axle.x, tensor[i, EgoInternalIndex.x()].item())
            self.assertEqual(ego.rear_axle.y, tensor[i, EgoInternalIndex.y()].item())
            self.assertEqual(ego.rear_axle.heading, tensor[i, EgoInternalIndex.heading()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_velocity_2d.x, tensor[i, EgoInternalIndex.vx()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_velocity_2d.y, tensor[i, EgoInternalIndex.vy()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_acceleration_2d.x, tensor[i, EgoInternalIndex.ax()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_acceleration_2d.y, tensor[i, EgoInternalIndex.ay()].item())

    def test_sampled_past_timestamps_to_tensor(self) -> None:
        """
        Test the conversion routine to convert timestamps to tensors.
        """
        points = [TimePoint(time_us=i) for i in range(10)]
        tensor = sampled_past_timestamps_to_tensor(points)
        self.assertEqual((10,), tensor.shape)
        for i in range(tensor.shape[0]):
            self.assertEqual(i, int(tensor[i].item()))

    def test_tracked_objects_to_tensor_list(self) -> None:
        """
        Test the conversion routine to convert tracked objects to tensors.
        """
        num_frames = 5
        test_tracked_objects = _create_dummy_tracked_objects_tensor(num_frames)
        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)
            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))
        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects, object_type=TrackedObjectType.BICYCLE)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)
            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))
        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects, object_type=TrackedObjectType.PEDESTRIAN)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)
            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))

    def test_pack_agents_tensor(self) -> None:
        """
        Test the routine used to convert local buffers into the final feature.
        """
        num_agents = 4
        num_timestamps = 3
        agents_tensors = [_create_tracked_object_agent_tensor(num_agents) for _ in range(num_timestamps)]
        yaw_rates = torch.ones((num_timestamps, num_agents)) * 100
        packed = pack_agents_tensor(agents_tensors, yaw_rates)
        self.assertEqual((num_timestamps, num_agents, AgentFeatureIndex.dim()), packed.shape)
        for ts in range(num_timestamps):
            for agent in range(num_agents):
                for col in range(AgentFeatureIndex.dim()):
                    if col == AgentFeatureIndex.yaw_rate():
                        self.assertEqual(100, packed[ts, agent, col])
                    else:
                        self.assertEqual(agent, packed[ts, agent, col])

def create_sample_simulation_log(output_path: Path) -> SimulationLog:
    """
    Generates a sample simulation log for use in tests.
    :param output_path: to write to.
    """
    scenario = MockAbstractScenario()
    planner = SimplePlanner(horizon_seconds=2, sampling_time=0.5, acceleration=np.array([0.0, 0.0]))
    history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
    state_0 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(0))
    state_1 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(1000))
    history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_0, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0))))
    history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_1, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0))))
    return SimulationLog(file_path=Path(output_path), scenario=scenario, planner=planner, simulation_history=history)

class SimulationTile:
    """Scenario simulation tile for visualization."""

    def __init__(self, doc: Document, experiment_file_data: ExperimentFileData, vehicle_parameters: VehicleParameters, map_factory: AbstractMapFactory, period_milliseconds: int=5000, radius: float=300.0, async_rendering: bool=True, frame_rate_cap_hz: int=60):
        """
        Scenario simulation tile.
        :param doc: Bokeh HTML document.
        :param experiment_file_data: Experiment file data.
        :param vehicle_parameters: Ego pose parameters.
        :param map_factory: Map factory for building maps.
        :param period_milliseconds: Milliseconds to update the tile.
        :param radius: Map radius.
        :param async_rendering: When true, will use threads to render asynchronously.
        :param frame_rate_cap_hz: Maximum frames to render per second. Internally this value is capped at 60.
        """
        self._doc = doc
        self._vehicle_parameters = vehicle_parameters
        self._map_factory = map_factory
        self._experiment_file_data = experiment_file_data
        self._period_milliseconds = period_milliseconds
        self._radius = radius
        self._selected_scenario_keys: List[SimulationScenarioKey] = []
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._maps: Dict[str, AbstractMap] = {}
        self._figures: List[SimulationFigure] = []
        self._nearest_vector_map: Dict[SemanticMapLayer, List[MapObject]] = {}
        self._async_rendering = async_rendering
        self._plot_render_queue: Optional[Tuple[SimulationFigure, int]] = None
        self._doc.add_periodic_callback(self._periodic_callback, period_milliseconds=1000)
        self._last_frame_time = time.time()
        self._current_frame_index = 0
        self._last_frame_index = 0
        self._playback_callback_handle: Optional[PeriodicCallback] = None
        if frame_rate_cap_hz < 1 or frame_rate_cap_hz > 60:
            raise ValueError('frame_rate_cap_hz should be between 1 and 60')
        self._minimum_frame_time_seconds = 1.0 / float(frame_rate_cap_hz)
        logger.info('Minimum frame time=%4.3f s', self._minimum_frame_time_seconds)

    @property
    def get_figure_data(self) -> List[SimulationFigure]:
        """Return figure data."""
        return self._figures

    @property
    def is_in_playback(self) -> bool:
        """Returns True if we're currently rendering a playback of a figure."""
        return self._playback_callback_handle is not None

    def _on_mouse_move(self, event: PointEvent, figure_index: int) -> None:
        """
        Event when mouse moving in a figure.
        :param event: Point event.
        :param figure_index: Figure index where the mouse is moving.
        """
        main_figure = self._figures[figure_index]
        main_figure.x_y_coordinate_title.text = f'x [m]: {np.round(event.x, simulation_tile_style['decimal_points'])}, y [m]: {np.round(event.y, simulation_tile_style['decimal_points'])}'

    def _create_frame_control_button(self, button_config: ScenarioTabFrameButtonConfig, click_callback: EventCallback, figure_index: int) -> Button:
        """
        Helper function to create a frame control button (prev, play, etc.) based on the provided config.
        :param button_config: Configuration object for the frame control button.
        :param click_callback: Button click event callback that will be registered to the created button.
        :param figure_index: The figure index to be passed to the button's click event callback.
        :return: The created Bokeh Button instance.
        """
        button_instance = Button(label=button_config.label, margin=button_config.margin, css_classes=button_config.css_classes, width=button_config.width)
        button_instance.on_click(partial(click_callback, figure_index=figure_index))
        return button_instance

    def _create_initial_figure(self, figure_index: int, figure_sizes: List[int], backend: Optional[str]='webgl') -> SimulationFigure:
        """
        Create an initial Bokeh figure.
        :param figure_index: Figure index.
        :param figure_sizes: width and height in pixels.
        :param backend: Bokeh figure backend.
        :return: A Bokeh figure.
        """
        selected_scenario_key = self._selected_scenario_keys[figure_index]
        experiment_path = Path(self._experiment_file_data.file_paths[selected_scenario_key.nuboard_file_index].metric_main_path)
        planner_name = selected_scenario_key.planner_name
        presented_planner_name = planner_name + f' ({experiment_path.stem})'
        simulation_figure = Figure(x_range=(-self._radius, self._radius), y_range=(-self._radius, self._radius), width=figure_sizes[0], height=figure_sizes[1], title=f'{presented_planner_name}', tools=['pan', 'wheel_zoom', 'save', 'reset'], match_aspect=True, active_scroll='wheel_zoom', margin=simulation_tile_style['figure_margins'], background_fill_color=simulation_tile_style['background_color'], output_backend=backend)
        simulation_figure.on_event('mousemove', partial(self._on_mouse_move, figure_index=figure_index))
        simulation_figure.axis.visible = False
        simulation_figure.xgrid.visible = False
        simulation_figure.ygrid.visible = False
        simulation_figure.title.text_font_size = simulation_tile_style['figure_title_text_font_size']
        x_y_coordinate_title = Title(text='x [m]: , y [m]: ')
        simulation_figure.add_layout(x_y_coordinate_title, 'below')
        slider = Slider(start=0, end=1, value=0, step=1, title='Frame', margin=simulation_tile_style['slider_margins'], css_classes=['scenario-frame-slider'])
        slider.on_change('value', partial(self._slider_on_change, figure_index=figure_index))
        video_button = Button(label='Render video', margin=simulation_tile_style['video_button_margins'], css_classes=['scenario-video-button'])
        video_button.on_click(partial(self._video_button_on_click, figure_index=figure_index))
        first_button = self._create_frame_control_button(first_button_config, self._first_button_on_click, figure_index)
        prev_button = self._create_frame_control_button(prev_button_config, self._prev_button_on_click, figure_index)
        play_button = self._create_frame_control_button(play_button_config, self._play_button_on_click, figure_index)
        next_button = self._create_frame_control_button(next_button_config, self._next_button_on_click, figure_index)
        last_button = self._create_frame_control_button(last_button_config, self._last_button_on_click, figure_index)
        assert len(selected_scenario_key.files) == 1, 'Expected one file containing the serialized SimulationLog.'
        simulation_file = next(iter(selected_scenario_key.files))
        simulation_log = SimulationLog.load_data(simulation_file)
        simulation_figure_data = SimulationFigure(figure=simulation_figure, file_path_index=selected_scenario_key.nuboard_file_index, figure_title_name=presented_planner_name, slider=slider, video_button=video_button, first_button=first_button, prev_button=prev_button, play_button=play_button, next_button=next_button, last_button=last_button, vehicle_parameters=self._vehicle_parameters, planner_name=planner_name, scenario=simulation_log.scenario, simulation_history=simulation_log.simulation_history, x_y_coordinate_title=x_y_coordinate_title)
        return simulation_figure_data

    def _map_api(self, map_name: str) -> AbstractMap:
        """
        Get a map api.
        :param map_name: Map name.
        :return Map api.
        """
        if map_name not in self._maps:
            self._maps[map_name] = self._map_factory.build_map_from_name(map_name)
        return self._maps[map_name]

    def init_simulations(self, figure_sizes: List[int]) -> None:
        """
        Initialization of the visualization of simulation panel.
        :param figure_sizes: Width and height in pixels.
        """
        self._figures = []
        for figure_index in range(len(self._selected_scenario_keys)):
            simulation_figure = self._create_initial_figure(figure_index=figure_index, figure_sizes=figure_sizes)
            self._figures.append(simulation_figure)

    @property
    def figures(self) -> List[SimulationFigure]:
        """
        Access bokeh figures.
        :return A list of bokeh figures.
        """
        return self._figures

    def _render_simulation_layouts(self) -> List[SimulationData]:
        """
        Render simulation layouts.
        :return: A list of columns or rows.
        """
        grid_layouts: List[SimulationData] = []
        for simulation_figure in self.figures:
            grid_layouts.append(SimulationData(planner_name=simulation_figure.planner_name, simulation_figure=simulation_figure, plot=gridplot([[simulation_figure.slider], [row([simulation_figure.first_button, simulation_figure.prev_button, simulation_figure.play_button, simulation_figure.next_button, simulation_figure.last_button])], [simulation_figure.figure], [simulation_figure.video_button]], toolbar_location='left')))
        return grid_layouts

    def render_simulation_tiles(self, selected_scenario_keys: List[SimulationScenarioKey], figure_sizes: List[int]=simulation_tile_style['figure_sizes'], hidden_glyph_names: Optional[List[str]]=None) -> List[SimulationData]:
        """
        Render simulation tiles.
        :param selected_scenario_keys: A list of selected scenario keys.
        :param figure_sizes: Width and height in pixels.
        :param hidden_glyph_names: A list of glyph names to be hidden.
        :return A list of bokeh layouts.
        """
        self._selected_scenario_keys = selected_scenario_keys
        self.init_simulations(figure_sizes=figure_sizes)
        for main_figure in tqdm(self._figures, desc='Rendering a scenario'):
            self._render_scenario(main_figure, hidden_glyph_names=hidden_glyph_names)
        layouts = self._render_simulation_layouts()
        return layouts

    @gen.coroutine
    @without_document_lock
    def _video_button_on_click(self, figure_index: int) -> None:
        """
        Callback to video button click event.
        Note that this callback in run on a background thread.
        :param figure_index: Figure index.
        """
        self._figures[figure_index].video_button.disabled = True
        self._figures[figure_index].video_button.label = 'Rendering video now...'
        self._executor.submit(self._video_button_next_tick, figure_index)

    def _reset_video_button(self, figure_index: int) -> None:
        """
        Reset a video button after exporting is done.
        :param figure_index: Figure index.
        """
        self.figures[figure_index].video_button.label = 'Render video'
        self.figures[figure_index].video_button.disabled = False

    def _update_video_button_label(self, figure_index: int, label: str) -> None:
        """
        Update a video button label to show progress when rendering a video.
        :param figure_index: Figure index.
        :param label: New video button text.
        """
        self.figures[figure_index].video_button.label = label

    def _video_button_next_tick(self, figure_index: int) -> None:
        """
        Synchronous callback to the video button on click event.
        :param figure_index: Figure index.
        """
        if not len(self._figures):
            return
        images = []
        scenario_key = self._selected_scenario_keys[figure_index]
        scenario_name = scenario_key.scenario_name
        scenario_type = scenario_key.scenario_type
        planner_name = scenario_key.planner_name
        video_name = scenario_type + '_' + planner_name + '_' + scenario_name + '.avi'
        nuboard_file_index = scenario_key.nuboard_file_index
        video_path = Path(self._experiment_file_data.file_paths[nuboard_file_index].simulation_main_path) / 'video_screenshot'
        if not video_path.exists():
            video_path.mkdir(parents=True, exist_ok=True)
        video_save_path = video_path / video_name
        scenario = self.figures[figure_index].scenario
        database_interval = scenario.database_interval
        selected_simulation_figure = self._figures[figure_index]
        try:
            if len(selected_simulation_figure.ego_state_plot.data_sources):
                chrome_options = webdriver.ChromeOptions()
                chrome_options.headless = True
                driver = webdriver.Chrome(chrome_options=chrome_options)
                driver.set_window_size(1920, 1080)
                shape = None
                simulation_figure = self._create_initial_figure(figure_index=figure_index, backend='canvas', figure_sizes=simulation_tile_style['render_figure_sizes'])
                simulation_figure.copy_datasources(selected_simulation_figure)
                self._render_scenario(main_figure=simulation_figure)
                length = len(selected_simulation_figure.ego_state_plot.data_sources)
                for frame_index in tqdm(range(length), desc='Rendering video'):
                    self._render_plots(main_figure=simulation_figure, frame_index=frame_index)
                    image = get_screenshot_as_png(column(simulation_figure.figure), driver=driver)
                    shape = image.size
                    images.append(image)
                    label = f'Rendering video now... ({frame_index}/{length})'
                    self._doc.add_next_tick_callback(partial(self._update_video_button_label, figure_index=figure_index, label=label))
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                if database_interval:
                    fps = 1 / database_interval
                else:
                    fps = 20
                video_obj = cv2.VideoWriter(filename=str(video_save_path), fourcc=fourcc, fps=fps, frameSize=shape)
                for index, image in enumerate(images):
                    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    video_obj.write(cv2_image)
                video_obj.release()
                logger.info('Video saved to %s' % str(video_save_path))
        except (RuntimeError, Exception) as e:
            logger.warning('%s' % e)
        self._doc.add_next_tick_callback(partial(self._reset_video_button, figure_index=figure_index))

    def _first_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the first button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._request_specific_frame(figure=figure, frame_index=0)

    def _prev_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the prev button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._request_previous_frame(figure)

    def _play_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the play button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._process_play_request(figure)

    def _next_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the next button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._request_next_frame(figure)

    def _last_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the last button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._request_specific_frame(figure=figure, frame_index=len(figure.simulation_history.data) - 1)

    def _slider_on_change(self, attr: str, old: int, frame_index: int, figure_index: int) -> None:
        """
        The function that's called every time the slider's value has changed.
        All frame requests are routed through slider's event handling since currently there's no way to manually
        set the slider's value programatically (to sync the slider value) without triggering this event.
        :param attr: Attribute name.
        :param old: Old value.
        :param frame_index: The new value of the slider, which is the requested frame index.
        :param figure_index: Figure index.
        """
        del attr, old
        selected_figure = self._figures[figure_index]
        self._request_plot_rendering(figure=selected_figure, frame_index=frame_index)

    def _request_specific_frame(self, figure: SimulationFigure, frame_index: int) -> None:
        """
        Requests to render the previous frame of the specified SimulationFigure.
        :param figure: The SimulationFigure render.
        :param frame_index: The frame index to render
        """
        figure.slider.value = frame_index

    def _request_previous_frame(self, figure: SimulationFigure) -> None:
        """
        Requests to render the previous frame of the specified SimulationFigure.
        :param figure: The SimulationFigure render.
        """
        if self._current_frame_index > 0:
            figure.slider.value = self._current_frame_index - 1

    def _request_next_frame(self, figure: SimulationFigure) -> bool:
        """
        Requests to render next frame of the specified SimulationFigure.
        :param figure: The SimulationFigure render.
        :return True if the request is valid, False otherwise.
        """
        result = False
        if self._current_frame_index < len(figure.simulation_history.data) - 1:
            figure.slider.value = self._current_frame_index + 1
            result = True
        return result

    def _request_plot_rendering(self, figure: SimulationFigure, frame_index: int) -> None:
        """
        Request the SimulationTile to render a frame of the plot. The requested frame will be enqueued if frame rate cap
        is reached or the figure is currently rendering a frame.
        :param figure: The SimulationFigure to render.
        :param frame_index: The requested frame index to render.
        """
        current_time = time.time()
        if current_time - self._last_frame_time < self._minimum_frame_time_seconds or figure.is_rendering():
            logger.info('Frame deferred: %d', frame_index)
            self._plot_render_queue = (figure, frame_index)
        else:
            self._process_plot_render_request(figure=figure, frame_index=frame_index)
            self._last_frame_time = time.time()

    def _stop_playback(self, figure: SimulationFigure) -> None:
        """
        Stops the playback for the given figure.
        :param figure: SimulationFigure to stop rendering.
        """
        if self._playback_callback_handle:
            self._doc.remove_periodic_callback(self._playback_callback_handle)
            self._playback_callback_handle = None
            figure.play_button.label = 'play'

    def _start_playback(self, figure: SimulationFigure) -> None:
        """
        Starts the playback for the given figure.
        :param figure: SimulationFigure to stop rendering.
        """
        callback_period_seconds = figure.simulation_history.interval_seconds
        callback_period_seconds = max(self._minimum_frame_time_seconds, callback_period_seconds)
        callback_period_ms = 1000.0 * callback_period_seconds
        self._playback_callback_handle = self._doc.add_periodic_callback(partial(self._playback_callback, figure), callback_period_ms)
        figure.play_button.label = 'stop'

    def _playback_callback(self, figure: SimulationFigure) -> None:
        """The callback that will advance the simulation frame. Will automatically stop the playback once we reach the final frame."""
        if not self._request_next_frame(figure):
            self._stop_playback(figure)

    def _process_play_request(self, figure: SimulationFigure) -> None:
        """
        Processes play request. When play mode is activated, the frame auto-advances, at the rate of the currently set frame rate cap.
        :param figure: The SimulationFigure to render.
        """
        if self._playback_callback_handle:
            self._stop_playback(figure)
        else:
            self._start_playback(figure)

    def _process_plot_render_request(self, figure: SimulationFigure, frame_index: int) -> None:
        """
        Process plot render requests, coming either from the slider or the render queue.
        :param figure: The SimulationFigure to render.
        :param frame_index: The requested frame index to render.
        """
        if frame_index != len(figure.simulation_history.data):
            if self._async_rendering:
                thread = threading.Thread(target=self._render_plots, kwargs={'main_figure': figure, 'frame_index': frame_index}, daemon=True)
                thread.start()
            else:
                self._render_plots(main_figure=figure, frame_index=frame_index)

    def _render_scenario(self, main_figure: SimulationFigure, hidden_glyph_names: Optional[List[str]]=None) -> None:
        """
        Render scenario.
        :param main_figure: Simulation figure object.
        :param hidden_glyph_names: A list of glyph names to be hidden.
        """
        if self._async_rendering:

            def render() -> None:
                """Wrapper for the non-map-dependent parts of the rendering logic."""
                main_figure.update_data_sources()
                self._render_expert_trajectory(main_figure=main_figure)
                mission_goal = main_figure.scenario.get_mission_goal()
                if mission_goal is not None:
                    main_figure.render_mission_goal(mission_goal_state=mission_goal)
                self._render_plots(main_figure=main_figure, frame_index=0, hidden_glyph_names=hidden_glyph_names)

            def render_map_dependent() -> None:
                """Wrapper for the map-dependent parts of the rendering logic."""
                self._load_map_data(main_figure=main_figure)
                main_figure.update_map_dependent_data_sources()
                self._render_map(main_figure=main_figure)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            executor.submit(render)
            executor.submit(render_map_dependent)
            executor.shutdown(wait=False)
        else:
            main_figure.update_data_sources()
            self._load_map_data(main_figure=main_figure)
            main_figure.update_map_dependent_data_sources()
            self._render_map(main_figure=main_figure)
            self._render_expert_trajectory(main_figure=main_figure)
            mission_goal = main_figure.scenario.get_mission_goal()
            if mission_goal is not None:
                main_figure.render_mission_goal(mission_goal_state=mission_goal)
            self._render_plots(main_figure=main_figure, frame_index=0, hidden_glyph_names=hidden_glyph_names)

    def _load_map_data(self, main_figure: SimulationFigure) -> None:
        """
        Load the map data of the simulation tile.
        :param main_figure: Simulation figure.
        """
        map_name = main_figure.scenario.map_api.map_name
        map_api = self._map_api(map_name)
        layer_names = [SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.LANE, SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE, SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA]
        assert main_figure.simulation_history.data, 'No simulation history samples, unable to render the map.'
        ego_pose = main_figure.simulation_history.data[0].ego_state.center
        center = Point2D(ego_pose.x, ego_pose.y)
        self._nearest_vector_map = map_api.get_proximal_map_objects(center, self._radius, layer_names)
        if SemanticMapLayer.STOP_LINE in self._nearest_vector_map:
            stop_polygons = self._nearest_vector_map[SemanticMapLayer.STOP_LINE]
            self._nearest_vector_map[SemanticMapLayer.STOP_LINE] = [stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP]
        main_figure.lane_connectors = {lane_connector.id: lane_connector for lane_connector in self._nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]}

    def _render_map_polygon_layers(self, main_figure: SimulationFigure) -> None:
        """Renders the polygon layers of the map."""
        polygon_layer_names = [(SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.LANE]), (SemanticMapLayer.INTERSECTION, simulation_map_layer_color[SemanticMapLayer.INTERSECTION]), (SemanticMapLayer.STOP_LINE, simulation_map_layer_color[SemanticMapLayer.STOP_LINE]), (SemanticMapLayer.CROSSWALK, simulation_map_layer_color[SemanticMapLayer.CROSSWALK]), (SemanticMapLayer.WALKWAYS, simulation_map_layer_color[SemanticMapLayer.WALKWAYS]), (SemanticMapLayer.CARPARK_AREA, simulation_map_layer_color[SemanticMapLayer.CARPARK_AREA])]
        roadblock_ids = main_figure.scenario.get_route_roadblock_ids()
        if roadblock_ids:
            polygon_layer_names.append((SemanticMapLayer.ROADBLOCK, simulation_map_layer_color[SemanticMapLayer.ROADBLOCK]))
        for layer_name, color in polygon_layer_names:
            map_polygon = MapPoint(point_2d=[])
            if layer_name == SemanticMapLayer.ROADBLOCK:
                layer = self._nearest_vector_map[SemanticMapLayer.LANE] + self._nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]
                for map_obj in layer:
                    roadblock_id = map_obj.get_roadblock_id()
                    if roadblock_id in roadblock_ids:
                        coords = map_obj.polygon.exterior.coords
                        points = [Point2D(x=x, y=y) for x, y in coords]
                        map_polygon.point_2d.append(points)
            else:
                layer = self._nearest_vector_map[layer_name]
                for map_obj in layer:
                    coords = map_obj.polygon.exterior.coords
                    points = [Point2D(x=x, y=y) for x, y in coords]
                    map_polygon.point_2d.append(points)
            polygon_source = ColumnDataSource(dict(xs=map_polygon.polygon_xs, ys=map_polygon.polygon_ys))
            layer_map_polygon_plot = main_figure.figure.multi_polygons(xs='xs', ys='ys', fill_color=color['fill_color'], fill_alpha=color['fill_color_alpha'], line_color=color['line_color'], source=polygon_source)
            layer_map_polygon_plot.level = 'underlay'
            main_figure.map_polygon_plots[layer_name.name] = layer_map_polygon_plot

    def _render_map_line_layers(self, main_figure: SimulationFigure) -> None:
        """Renders the line layers of the map."""
        line_layer_names = [(SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.BASELINE_PATHS]), (SemanticMapLayer.LANE_CONNECTOR, simulation_map_layer_color[SemanticMapLayer.LANE_CONNECTOR])]
        for layer_name, color in line_layer_names:
            layer = self._nearest_vector_map[layer_name]
            map_line = MapPoint(point_2d=[])
            for map_obj in layer:
                path = map_obj.baseline_path.discrete_path
                points = [Point2D(x=pose.x, y=pose.y) for pose in path]
                map_line.point_2d.append(points)
            line_source = ColumnDataSource(dict(xs=map_line.line_xs, ys=map_line.line_ys))
            layer_map_line_plot = main_figure.figure.multi_line(xs='xs', ys='ys', line_color=color['line_color'], line_alpha=color['line_color_alpha'], line_width=0.5, line_dash='dashed', source=line_source)
            layer_map_line_plot.level = 'underlay'
            main_figure.map_line_plots[layer_name.name] = layer_map_line_plot

    def _render_map(self, main_figure: SimulationFigure) -> None:
        """
        Render a map.
        :param main_figure: Simulation figure.
        """

        def render() -> None:
            """Wrapper for the actual render logic, for multi-threading compatibility."""
            self._render_map_polygon_layers(main_figure)
            self._render_map_line_layers(main_figure)
        self._doc.add_next_tick_callback(lambda: render())

    @staticmethod
    def _render_expert_trajectory(main_figure: SimulationFigure) -> None:
        """
        Render expert trajectory.
        :param main_figure: Main simulation figure.
        """
        expert_ego_trajectory = main_figure.scenario.get_expert_ego_trajectory()
        source = extract_source_from_states(expert_ego_trajectory)
        main_figure.render_expert_trajectory(expert_ego_trajectory_state=source)

    def _render_plots(self, main_figure: SimulationFigure, frame_index: int, hidden_glyph_names: Optional[List[str]]=None) -> None:
        """
        Render plot with a frame index.
        :param main_figure: Main figure to render.
        :param frame_index: A frame index.
        :param hidden_glyph_names: A list of glyph names to be hidden.
        """
        if main_figure.lane_connectors is not None and len(main_figure.lane_connectors):
            main_figure.traffic_light_plot.update_plot(main_figure=main_figure.figure, frame_index=frame_index, doc=self._doc)
        main_figure.ego_state_plot.update_plot(main_figure=main_figure.figure, frame_index=frame_index, radius=self._radius, doc=self._doc)
        main_figure.ego_state_trajectory_plot.update_plot(main_figure=main_figure.figure, frame_index=frame_index, doc=self._doc)
        main_figure.agent_state_plot.update_plot(main_figure=main_figure.figure, frame_index=frame_index, doc=self._doc)
        main_figure.agent_state_heading_plot.update_plot(main_figure=main_figure.figure, frame_index=frame_index, doc=self._doc)

        def update_decorations() -> None:
            main_figure.figure.title.text = main_figure.figure_title_name_with_timestamp(frame_index=frame_index)
            main_figure.update_glyphs_visibility(glyph_names=hidden_glyph_names)
        self._doc.add_next_tick_callback(lambda: update_decorations())
        self._last_frame_index = self._current_frame_index
        self._current_frame_index = frame_index

    def _periodic_callback(self) -> None:
        """Periodic callback registered to the bokeh.Document."""
        if self._plot_render_queue:
            figure, frame_index = self._plot_render_queue
            last_frame_direction = math.copysign(1, self._current_frame_index - self._last_frame_index)
            request_frame_direction = math.copysign(1, frame_index - self._current_frame_index)
            if request_frame_direction != last_frame_direction:
                logger.info('Frame dropped %d', frame_index)
                self._plot_render_queue = None
            elif not figure.is_rendering():
                logger.info('Processing render queue for frame %d', frame_index)
                self._plot_render_queue = None
                self._process_plot_render_request(figure=figure, frame_index=frame_index)

class TestSimulationTile(unittest.TestCase):
    """Test simulation_tile functionality."""

    def set_up_simulation_log(self, output_path: Path) -> None:
        """
        Create a simulation log and save it to disk.
        :param output path: to write the simulation log to.
        """
        simulation_log = create_sample_simulation_log(output_path)
        simulation_log.save_to_file()

    def setUp(self) -> None:
        """Set up simulation tile with nuboard file."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.vehicle_parameters = get_pacifica_parameters()
        simulation_log_path = Path(self.tmp_dir.name) / 'test_simulation_tile_simulation_log.msgpack.xz'
        self.set_up_simulation_log(simulation_log_path)
        nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulation', aggregator_metric_folder='aggregator_metric', current_path=Path(self.tmp_dir.name))
        self.scenario_keys = [SimulationScenarioKey(nuboard_file_index=0, log_name='dummy_log', planner_name='SimplePlanner', scenario_type='common', scenario_name='test', files=[simulation_log_path])]
        self.doc = Document()
        self.map_factory = MockMapFactory()
        self.experiment_file_data = ExperimentFileData(file_paths=[nuboard_file])
        self.simulation_tile = SimulationTile(doc=self.doc, map_factory=self.map_factory, vehicle_parameters=self.vehicle_parameters, radius=80, experiment_file_data=self.experiment_file_data)

    @given(frame_rate_cap=st.integers(min_value=1, max_value=60))
    def test_valid_frame_rate_cap_range(self, frame_rate_cap: int) -> None:
        """Tests valid frame rate cap range."""
        SimulationTile(doc=self.doc, map_factory=self.map_factory, vehicle_parameters=self.vehicle_parameters, radius=80, experiment_file_data=self.experiment_file_data, frame_rate_cap_hz=frame_rate_cap)

    @given(frame_rate_cap=st.integers().filter(lambda x: x < 1 or x > 60))
    def test_invalid_frame_rate_cap_range(self, frame_rate_cap: int) -> None:
        """Tests invalid frame rate cap range."""
        with self.assertRaises(ValueError):
            SimulationTile(doc=self.doc, map_factory=self.map_factory, vehicle_parameters=self.vehicle_parameters, radius=80, experiment_file_data=self.experiment_file_data, frame_rate_cap_hz=frame_rate_cap)

    def test_simulation_tile_layout(self) -> None:
        """Test layout design."""
        layout = self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
        self.assertEqual(len(layout), 1)

    def test_periodic_callback(self) -> None:
        """Tests that _periodic_callback is registered correctly to the bokeh Document."""
        with patch.object(SimulationTile, '_periodic_callback', autospec=True) as mock_periodic_callback:
            SimulationTile(doc=self.doc, map_factory=self.map_factory, vehicle_parameters=self.vehicle_parameters, radius=80, experiment_file_data=self.experiment_file_data)
            for cb in self.doc.callbacks.session_callbacks:
                cb.callback()
            self.assertEqual(mock_periodic_callback.call_count, 1)

    def _trigger_button_click_event(self, figure_index: int, button_name: str) -> None:
        """
        Trigger a bokeh.model.Button click event.
        :param figure_index: The index of the SimulationTile figure.
        :param button_name: The name of SimulationTile button.
        """
        button = getattr(self.simulation_tile.figures[figure_index], button_name)
        button._trigger_event(ButtonClick(button))

    def _test_frame_index_request_button(self, button_name: str, frame_index_request: FrameIndexRequest) -> None:
        """
        Helper function to test that frame index request buttons (first, prev, next, last) work correctly.
        :param click_callback_name: Button click callback function name in SimulationTile that's registered to bokeh.
        :param button_name: The name of the button in the SimulationTile class.
        :param frame_index_request: FrameIndexRequest object representing the frame index requested.
        """
        with patch.object(self.simulation_tile, '_render_plots'):
            self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
            figure_index = 0
            figure = self.simulation_tile.figures[figure_index]
            if frame_index_request == FrameIndexRequest.FIRST or frame_index_request == FrameIndexRequest.LAST:
                self._trigger_button_click_event(figure_index, button_name)
                frame_index = len(figure.simulation_history) - 1 if frame_index_request == FrameIndexRequest.LAST else 0
                self.assertEqual(figure.slider.value, frame_index)
            elif frame_index_request == FrameIndexRequest.NEXT:
                self.simulation_tile._current_frame_index = 0
                self._trigger_button_click_event(figure_index, button_name)
                self.assertEqual(figure.slider.value, self.simulation_tile._current_frame_index + 1)
                self.simulation_tile._current_frame_index = len(figure.simulation_history.data) - 1
                self._trigger_button_click_event(figure_index, button_name)
                self.assertEqual(figure.slider.value, self.simulation_tile._current_frame_index)
            elif frame_index_request == FrameIndexRequest.PREV:
                self.simulation_tile._current_frame_index = len(figure.simulation_history.data) - 1
                self._trigger_button_click_event(figure_index, button_name)
                self.assertEqual(figure.slider.value, self.simulation_tile._current_frame_index - 1)
                self.simulation_tile._current_frame_index = 0
                self._trigger_button_click_event(figure_index, button_name)
                self.assertEqual(figure.slider.value, self.simulation_tile._current_frame_index)

    def test_first_frame_button(self) -> None:
        """Tests that go to first frame button works correctly."""
        self._test_frame_index_request_button(button_name='first_button', frame_index_request=FrameIndexRequest.FIRST)

    def test_last_frame_button(self) -> None:
        """Tests that go to last frame button works correctly."""
        self._test_frame_index_request_button(button_name='last_button', frame_index_request=FrameIndexRequest.LAST)

    def _test_symbolic_frame_request_callback_called(self, button_name: str, frame_request_callback_name: str) -> None:
        """
        Helper function to test that the provided symbolic frame request (previous, next, play/stop) callback is called when a button is clicked
        :param button_name: The name of the button in the SimulationTile class.
        :param frame_request_callback_name: Frame request callback function name in SimulationTile that's supposed to be called.
        """
        with patch.object(self.simulation_tile, frame_request_callback_name, autospec=True) as mock_request_frame:
            self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
            figure_index = 0
            button = getattr(self.simulation_tile.figures[figure_index], button_name)
            button._trigger_event(ButtonClick(button))
            mock_request_frame.assert_called_once_with(self.simulation_tile.figures[figure_index])

    def test_prev_button(self) -> None:
        """Tests that show prev frame button works correctly."""
        self._test_frame_index_request_button(button_name='prev_button', frame_index_request=FrameIndexRequest.PREV)

    def test_next_button(self) -> None:
        """Tests that show next frame button works correctly."""
        self._test_frame_index_request_button(button_name='next_button', frame_index_request=FrameIndexRequest.NEXT)

    def test_play_button(self) -> None:
        """Tests that the play button works correctly."""
        self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
        figure_index = 0
        button_name = 'play_button'
        self.assertFalse(self.simulation_tile.is_in_playback)
        self._trigger_button_click_event(figure_index, button_name)
        self.assertTrue(self.simulation_tile.is_in_playback)
        self._trigger_button_click_event(figure_index, button_name)
        self.assertFalse(self.simulation_tile.is_in_playback)

    def test_playback_callback(self) -> None:
        """Tests that the playback callback is registered correctly to the bokeh Document & behaves correctly."""
        self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
        figure_index = 0
        figure = self.simulation_tile.figures[figure_index]
        button_name = 'play_button'
        previous_request_index = figure.slider.value
        self._trigger_button_click_event(figure_index, button_name)
        for cb in self.doc.callbacks.session_callbacks:
            cb.callback()
        self.assertTrue(self.simulation_tile.is_in_playback)
        self.assertTrue(figure.slider.value, previous_request_index + 1)
        self.simulation_tile._current_frame_index = len(figure.simulation_history) - 1
        for cb in self.doc.callbacks.session_callbacks:
            cb.callback()
        self.assertFalse(self.simulation_tile.is_in_playback)

    def test_deferred_plot_rendering(self) -> None:
        """Tests that plot rendering request will be deferred if successive requests are triggered faster than the frame rate cap configured."""
        self.assertIsNone(self.simulation_tile._plot_render_queue)
        with patch.object(self.simulation_tile, '_last_frame_time', new=time.time()):
            self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
            figure_index = 0
            figure = self.simulation_tile.figures[figure_index]
            trigger_count = 2
            for _ in range(trigger_count):
                figure.slider.trigger(attr='value', old=0, new=1)
            self.assertIsNotNone(self.simulation_tile._plot_render_queue)

    def tearDown(self) -> None:
        """Clean up temporary folder and files."""
        self.tmp_dir.cleanup()

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

def build_simulations(cfg: DictConfig, worker: WorkerPool, callbacks: List[AbstractCallback], callbacks_worker: Optional[WorkerPool]=None, pre_built_planners: Optional[List[AbstractPlanner]]=None) -> List[SimulationRunner]:
    """
    Build simulations.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param callbacks: Callbacks for simulation.
    :param worker: Worker for job execution.
    :param callbacks_worker: worker pool to use for callbacks from sim
    :param pre_built_planners: List of pre-built planners to run in simulation.
    :return A dict of simulation engines with challenge names.
    """
    logger.info('Building simulations...')
    simulations = list()
    logger.info('Extracting scenarios...')
    if not int(os.environ.get('NUPLAN_SIMULATION_ALLOW_ANY_BUILDER', '0')) and (not is_target_type(cfg.scenario_builder, NuPlanScenarioBuilder)):
        raise ValueError(f'Simulation framework only runs with NuPlanScenarioBuilder. Got {cfg.scenario_builder}')
    scenario_filter = DistributedScenarioFilter(cfg=cfg, worker=worker, node_rank=int(os.environ.get('NODE_RANK', 0)), num_nodes=int(os.environ.get('NUM_NODES', 1)), synchronization_path=cfg.output_dir, timeout_seconds=cfg.distributed_timeout_seconds, distributed_mode=DistributedMode[cfg.distributed_mode])
    scenarios = scenario_filter.get_scenarios()
    metric_engines_map = {}
    if cfg.run_metric:
        logger.info('Building metric engines...')
        metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)
        logger.info('Building metric engines...DONE')
    else:
        logger.info('Metric engine is disable')
    logger.info('Building simulations from %d scenarios...', len(scenarios))
    for scenario in scenarios:
        if pre_built_planners is None:
            if 'planner' not in cfg.keys():
                raise KeyError('Planner not specified in config. Please specify a planner using "planner" field.')
            planners = build_planners(cfg.planner, scenario)
        else:
            planners = pre_built_planners
        for planner in planners:
            ego_controller: AbstractEgoController = instantiate(cfg.ego_controller, scenario=scenario)
            simulation_time_controller: AbstractSimulationTimeController = instantiate(cfg.simulation_time_controller, scenario=scenario)
            observations: AbstractObservation = build_observations(cfg.observation, scenario=scenario)
            metric_engine = metric_engines_map.get(scenario.scenario_type, None)
            if metric_engine is not None:
                stateful_callbacks = [MetricCallback(metric_engine=metric_engine, worker_pool=callbacks_worker)]
            else:
                stateful_callbacks = []
            if 'simulation_log_callback' in cfg.callback:
                stateful_callbacks.append(instantiate(cfg.callback['simulation_log_callback'], worker_pool=callbacks_worker))
            simulation_setup = SimulationSetup(time_controller=simulation_time_controller, observations=observations, ego_controller=ego_controller, scenario=scenario)
            simulation = Simulation(simulation_setup=simulation_setup, callback=MultiCallback(callbacks + stateful_callbacks), simulation_history_buffer_duration=cfg.simulation_history_buffer_duration)
            simulations.append(SimulationRunner(simulation, planner))
    logger.info('Building simulations...DONE!')
    return simulations

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute metrics with simulation logs only.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    assert cfg.simulation_log_main_path is not None, 'Simulation_log_main_path must be set when running metrics.'
    pl.seed_everything(cfg.seed, workers=True)
    profiler_name = 'building_metrics'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)
    simulation_logs = build_simulation_logs(cfg=cfg)
    runners = build_metric_runners(cfg=cfg, simulation_logs=simulation_logs)
    if common_builder.profiler:
        common_builder.profiler.save_profiler(profiler_name)
    logger.info('Running metrics...')
    run_runners(runners=runners, common_builder=common_builder, cfg=cfg, profiler_name='running_metrics')
    logger.info('Finished running metrics!')

def build_simulation_logs(cfg: DictConfig) -> List[SimulationLog]:
    """
    Build a list of simulation logs.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return A list of simulation logs.
    """
    logger.info('Building simulation logs...')
    simulation_logs = []
    simulation_log_path = Path(cfg.simulation_log_main_path) / cfg.callback.simulation_log_callback.simulation_log_dir
    for planner_dir_folder in simulation_log_path.iterdir():
        for scenario_type_folder in planner_dir_folder.iterdir():
            for log_name_folder in scenario_type_folder.iterdir():
                for scenario_name_folder in log_name_folder.iterdir():
                    for scenario_log_file in scenario_name_folder.iterdir():
                        simulation_log = SimulationLog.load_data(file_path=scenario_log_file)
                        simulation_logs.append(simulation_log)
    logger.info(f'Building simulation logs: {len(simulation_logs)}...DONE!')
    return simulation_logs

def build_metric_runners(cfg: DictConfig, simulation_logs: List[SimulationLog]) -> List[MetricRunner]:
    """
    Build metric runners.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param simulation_logs: A list of simulation logs.
    :return A list of metric runners.
    """
    logger.info('Building metric runners...')
    metric_runners = list()
    logger.info('Extracting scenarios...')
    scenarios = [simulation_log.scenario for simulation_log in simulation_logs]
    logger.info('Extracting scenarios...DONE!')
    logger.info('Building metric engines...')
    metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)
    logger.info('Building metric engines...DONE')
    logger.info(f'Building metric_runner from {len(scenarios)} scenarios...')
    for simulation_log in simulation_logs:
        scenario = simulation_log.scenario
        metric_engine = metric_engines_map.get(scenario.scenario_type, None)
        if not metric_engine:
            raise ValueError(f'{scenario.scenario_type} not found in a metric engine.')
        if not simulation_log:
            raise ValueError(f'{scenario.scenario_name} not found in simulation logs.')
        metric_callback = MetricCallback(metric_engine=metric_engine)
        metric_runner = MetricRunner(simulation_log=simulation_log, metric_callback=metric_callback)
        metric_runners.append(metric_runner)
    logger.info('Building metric runners...DONE!')
    return metric_runners

def build_metrics_engines(cfg: DictConfig, scenarios: List[AbstractScenario]) -> Dict[str, MetricsEngine]:
    """
    Build a metric engine for each different scenario type.
    :param cfg: Config.
    :param scenarios: list of scenarios for which metrics should be build.
    :return Dict of scenario types to metric engines.
    """
    main_save_path = pathlib.Path(cfg.output_dir) / cfg.metric_dir
    selected_metrics = cfg.selected_simulation_metrics
    if isinstance(selected_metrics, str):
        selected_metrics = [selected_metrics]
    simulation_metrics = cfg.simulation_metric
    low_level_metrics: DictConfig = simulation_metrics.get('low_level', {})
    high_level_metrics: DictConfig = simulation_metrics.get('high_level', {})
    metric_engines = {}
    for scenario in scenarios:
        if scenario.scenario_type in metric_engines:
            continue
        metric_engine = MetricsEngine(main_save_path=main_save_path)
        scenario_type = scenario.scenario_type
        scenario_metrics: DictConfig = simulation_metrics.get(scenario_type, {})
        metrics_in_scope = low_level_metrics.copy()
        metrics_in_scope.update(scenario_metrics)
        high_level_metric_in_scope = high_level_metrics.copy()
        if selected_metrics is not None:
            metrics_in_scope = {metric_name: metrics_in_scope[metric_name] for metric_name in selected_metrics if metric_name in metrics_in_scope}
            high_level_metric_in_scope = {metric_name: high_level_metrics[metric_name] for metric_name in selected_metrics if metric_name in high_level_metric_in_scope}
        base_metrics = {metric_name: instantiate(metric_config) for metric_name, metric_config in metrics_in_scope.items()}
        for metric in base_metrics.values():
            metric_engine.add_metric(metric)
        for metric_name, metric in high_level_metric_in_scope.items():
            high_level_metric = build_high_level_metric(cfg=metric, base_metrics=base_metrics)
            metric_engine.add_metric(high_level_metric)
            base_metrics[metric_name] = high_level_metric
        metric_engines[scenario_type] = metric_engine
    return metric_engines

def build_high_level_metric(cfg: DictConfig, base_metrics: Dict[str, AbstractMetricBuilder]) -> AbstractMetricBuilder:
    """
    Build a high level metric.
    :param cfg: High level metric config.
    :param base_metrics: A dict of base metrics.
    :return A high level metric.
    """
    OmegaConf.set_struct(cfg, False)
    required_metrics: Dict[str, str] = cfg.pop('required_metrics', {})
    OmegaConf.set_struct(cfg, True)
    metric_params = {}
    for metric_param, metric_name in required_metrics.items():
        metric_params[metric_param] = base_metrics[metric_name]
    return instantiate(cfg, **metric_params)

def _process_future_trajectories_for_windowed_agents(log_file: str, tracked_objects: List[TrackedObject], agent_indexes: Dict[int, Dict[str, int]], future_trajectory_sampling: TrajectorySampling) -> List[TrackedObject]:
    """
    A helper method to interpolate and parse the future trajectories for windowed agents.
    :param log_file: The log file to query.
    :param tracked_objects: The tracked objects to parse.
    :param agent_indexes: A mapping of [timestamp, [track_token, tracked_object_idx]]
    :param future_trajectory_sampling: The future trajectory sampling to use for future waypoints.
    :return: The tracked objects with predicted trajectories included.
    """
    agent_future_trajectories: Dict[int, Dict[str, List[Waypoint]]] = {}
    for timestamp in agent_indexes:
        agent_future_trajectories[timestamp] = {}
        for token in agent_indexes[timestamp]:
            agent_future_trajectories[timestamp][token] = []
    for timestamp_time in agent_future_trajectories:
        end_time = timestamp_time + int(1000000.0 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length))
        for track_token, waypoint in get_future_waypoints_for_agents_from_db(log_file, list(agent_indexes[timestamp_time].keys()), timestamp_time, end_time):
            agent_future_trajectories[timestamp_time][track_token].append(waypoint)
    for timestamp in agent_future_trajectories:
        for key in agent_future_trajectories[timestamp]:
            if len(agent_future_trajectories[timestamp][key]) == 1:
                tracked_objects[agent_indexes[timestamp][key]]._predictions = [PredictedTrajectory(1.0, agent_future_trajectories[timestamp][key])]
            elif len(agent_future_trajectories[timestamp][key]) > 1:
                tracked_objects[agent_indexes[timestamp][key]]._predictions = [PredictedTrajectory(1.0, interpolate_future_waypoints(agent_future_trajectories[timestamp][key], future_trajectory_sampling.time_horizon, future_trajectory_sampling.interval_length))]
    return tracked_objects

def extract_tracked_objects_within_time_window(token: str, log_file: str, past_time_horizon: float, future_time_horizon: float, filter_track_tokens: Optional[Set[str]]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> TrackedObjects:
    """
    Extracts the tracked objects in a time window centered on a token.
    :param token: The token on which to center the time window.
    :param past_time_horizon: The time in the past for which to search.
    :param future_time_horizon: The time in the future for which to search.
    :param filter_track_tokens: If provided, objects with track_tokens missing from the set will be excluded.
    :param future_trajectory_sampling: If provided, the future trajectory sampling to use for future waypoints.
    :return: The retrieved TrackedObjects.
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[int, Dict[str, int]] = {}
    token_timestamp = get_sensor_data_token_timestamp_from_db(log_file, get_lidarpc_sensor_data(), token)
    start_time = int(token_timestamp - 1000000.0 * past_time_horizon)
    end_time = int(token_timestamp + 1000000.0 * future_time_horizon)
    for idx, tracked_object in enumerate(get_tracked_objects_within_time_interval_from_db(log_file, start_time, end_time, filter_track_tokens)):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            if tracked_object.metadata.timestamp_us not in agent_indexes:
                agent_indexes[tracked_object.metadata.timestamp_us] = {}
            agent_indexes[tracked_object.metadata.timestamp_us][tracked_object.metadata.track_token] = idx
        tracked_objects.append(tracked_object)
    if future_trajectory_sampling:
        _process_future_trajectories_for_windowed_agents(log_file, tracked_objects, agent_indexes, future_trajectory_sampling)
    return TrackedObjects(tracked_objects=tracked_objects)

def extract_tracked_objects(token: str, log_file: str, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> TrackedObjects:
    """
    Extracts all boxes from a lidarpc.
    :param lidar_pc: Input lidarpc.
    :param future_trajectory_sampling: If provided, the future trajectory sampling to use for future waypoints.
    :return: Tracked objects contained in the lidarpc.
    """
    tracked_objects: List[TrackedObject] = []
    agent_indexes: Dict[str, int] = {}
    agent_future_trajectories: Dict[str, List[Waypoint]] = {}
    for idx, tracked_object in enumerate(get_tracked_objects_for_lidarpc_token_from_db(log_file, token)):
        if future_trajectory_sampling and isinstance(tracked_object, Agent):
            agent_indexes[tracked_object.metadata.track_token] = idx
            agent_future_trajectories[tracked_object.metadata.track_token] = []
        tracked_objects.append(tracked_object)
    if future_trajectory_sampling and len(tracked_objects) > 0:
        timestamp_time = get_sensor_data_token_timestamp_from_db(log_file, get_lidarpc_sensor_data(), token)
        end_time = timestamp_time + int(1000000.0 * (future_trajectory_sampling.time_horizon + future_trajectory_sampling.interval_length))
        for track_token, waypoint in get_future_waypoints_for_agents_from_db(log_file, list(agent_indexes.keys()), timestamp_time, end_time):
            agent_future_trajectories[track_token].append(waypoint)
        for key in agent_future_trajectories:
            if len(agent_future_trajectories[key]) == 1:
                tracked_objects[agent_indexes[key]]._predictions = [PredictedTrajectory(1.0, agent_future_trajectories[key])]
            elif len(agent_future_trajectories[key]) > 1:
                tracked_objects[agent_indexes[key]]._predictions = [PredictedTrajectory(1.0, interpolate_future_waypoints(agent_future_trajectories[key], future_trajectory_sampling.time_horizon, future_trajectory_sampling.interval_length))]
    return TrackedObjects(tracked_objects=tracked_objects)

class NuPlanScenario(AbstractScenario):
    """Scenario implementation for the nuPlan dataset that is used in training and simulation."""

    def __init__(self, data_root: str, log_file_load_path: str, initial_lidar_token: str, initial_lidar_timestamp: int, scenario_type: str, map_root: str, map_version: str, map_name: str, scenario_extraction_info: Optional[ScenarioExtractionInfo], ego_vehicle_parameters: VehicleParameters, sensor_root: Optional[str]=None) -> None:
        """
        Initialize the nuPlan scenario.
        :param data_root: The prefix for the log file. e.g. "/data/root/nuplan". For remote paths, this is where the file will be downloaded if necessary.
        :param log_file_load_path: Name of the log that this scenario belongs to. e.g. "/data/sets/nuplan-v1.1/splits/mini/2021.07.16.20.45.29_veh-35_01095_01486.db", "s3://path/to/db.db"
        :param initial_lidar_token: Token of the scenario's initial lidarpc.
        :param initial_lidar_timestamp: The timestamp of the initial lidarpc.
        :param scenario_type: Type of scenario (e.g. ego overtaking).
        :param map_root: The root path for the map db
        :param map_version: The version of maps to load
        :param map_name: The map name to use for the scenario
        :param scenario_extraction_info: Structure containing information used to extract the scenario.
            None means the scenario has no length and it is comprised only by the initial lidarpc.
        :param ego_vehicle_parameters: Structure containing the vehicle parameters.
        :param sensor_root: The root path for the sensor blobs.
        """
        self._local_store: Optional[LocalStore] = None
        self._remote_store: Optional[S3Store] = None
        self._data_root = data_root
        self._log_file_load_path = log_file_load_path
        self._initial_lidar_token = initial_lidar_token
        self._initial_lidar_timestamp = initial_lidar_timestamp
        self._scenario_type = scenario_type
        self._map_root = map_root
        self._map_version = map_version
        self._map_name = map_name
        self._scenario_extraction_info = scenario_extraction_info
        self._ego_vehicle_parameters = ego_vehicle_parameters
        self._sensor_root = sensor_root
        if self._scenario_extraction_info is not None:
            skip_rows = 1.0 / self._scenario_extraction_info.subsample_ratio
            if abs(int(skip_rows) - skip_rows) > 0.001:
                raise ValueError(f'Subsample ratio is not valid. Must resolve to an integer number of skipping rows, instead received {self._scenario_extraction_info.subsample_ratio}, which would skip {skip_rows} rows.')
        self._database_row_interval = 0.05
        self._log_file = download_file_if_necessary(self._data_root, self._log_file_load_path)
        self._log_name: str = absolute_path_to_log_name(self._log_file)

    def __reduce__(self) -> Tuple[Type[NuPlanScenario], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._data_root, self._log_file_load_path, self._initial_lidar_token, self._initial_lidar_timestamp, self._scenario_type, self._map_root, self._map_version, self._map_name, self._scenario_extraction_info, self._ego_vehicle_parameters, self._sensor_root))

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return self._ego_vehicle_parameters

    @cached_property
    def _lidarpc_tokens(self) -> List[str]:
        """
        :return: list of lidarpc tokens in the scenario
        """
        if self._scenario_extraction_info is None:
            return [self._initial_lidar_token]
        lidarpc_tokens = list(extract_sensor_tokens_as_scenario(self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_timestamp, self._scenario_extraction_info))
        return cast(List[str], lidarpc_tokens)

    @cached_property
    def _route_roadblock_ids(self) -> List[str]:
        """
        return: Route roadblock ids extracted from expert trajectory.
        """
        expert_trajectory = list(self._extract_expert_trajectory())
        return get_roadblock_ids_from_trajectory(self.map_api, expert_trajectory)

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._initial_lidar_token

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self.token

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_type

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        return get_maps_api(self._map_root, self._map_version, self._map_name)

    @property
    def map_root(self) -> str:
        """Get the map root folder."""
        return self._map_root

    @property
    def map_version(self) -> str:
        """Get the map version."""
        return self._map_version

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        if self._scenario_extraction_info is None:
            return 0.05
        return float(0.05 / self._scenario_extraction_info.subsample_ratio)

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return len(self._lidarpc_tokens)

    def get_lidar_to_ego_transform(self) -> Transform:
        """Inherited, see superclass."""
        return get_sensor_transform_matrix_for_sensor_data_token_from_db(self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token)

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Inherited, see superclass."""
        return get_mission_goal_for_sensor_data_token_from_db(self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token)

    def get_route_roadblock_ids(self) -> List[str]:
        """Inherited, see superclass."""
        roadblock_ids = get_roadblock_ids_for_lidarpc_token_from_db(self._log_file, self._initial_lidar_token)
        assert roadblock_ids is not None, 'Unable to find Roadblock ids for current scenario'
        return cast(List[str], roadblock_ids)

    def get_expert_goal_state(self) -> StateSE2:
        """Inherited, see superclass."""
        return get_statese2_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[-1])

    def get_time_point(self, iteration: int) -> TimePoint:
        """Inherited, see superclass."""
        return TimePoint(time_us=get_sensor_data_token_timestamp_from_db(self._log_file, get_lidarpc_sensor_data(), self._lidarpc_tokens[iteration]))

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""
        return get_ego_state_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[iteration])

    def get_tracked_objects_at_iteration(self, iteration: int, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f'Iteration is out of scenario: {iteration}!'
        return DetectionsTracks(extract_tracked_objects(self._lidarpc_tokens[iteration], self._log_file, future_trajectory_sampling))

    def get_tracked_objects_within_time_window_at_iteration(self, iteration: int, past_time_horizon: float, future_time_horizon: float, filter_track_tokens: Optional[Set[str]]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f'Iteration is out of scenario: {iteration}!'
        return DetectionsTracks(extract_tracked_objects_within_time_window(self._lidarpc_tokens[iteration], self._log_file, past_time_horizon, future_time_horizon, filter_track_tokens, future_trajectory_sampling))

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]]=None) -> Sensors:
        """Inherited, see superclass."""
        channels = [LidarChannel.MERGED_PC] if channels is None else channels
        lidar_pc = next(get_sensor_data_from_sensor_data_tokens_from_db(self._log_file, get_lidarpc_sensor_data(), LidarPc, [self._lidarpc_tokens[iteration]]))
        return self._get_sensor_data_from_lidar_pc(cast(LidarPc, lidar_pc), channels)

    def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TimePoint(lidar_pc.timestamp)

    def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TimePoint(lidar_pc.timestamp)

    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)
        return cast(Generator[EgoState, None, None], get_sampled_ego_states_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=False))

    def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)
        return cast(Generator[EgoState, None, None], get_sampled_ego_states_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=True))

    def get_past_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_future_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_past_sensors(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, channels: Optional[List[SensorChannel]]=None) -> Generator[Sensors, None, None]:
        """Inherited, see superclass."""
        channels = [LidarChannel.MERGED_PC] if channels is None else channels
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield self._get_sensor_data_from_lidar_pc(lidar_pc, channels)

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Inherited, see superclass."""
        token = self._lidarpc_tokens[iteration]
        return cast(Generator[TrafficLightStatusData, None, None], get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, token))

    def get_past_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets past traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the past.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the past.
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TrafficLightStatuses(list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token)))

    def get_future_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets future traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the future.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the future.
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TrafficLightStatuses(list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token)))

    def get_scenario_tokens(self) -> List[str]:
        """Return the list of lidarpc tokens from the DB that are contained in the scenario."""
        return self._lidarpc_tokens

    def _find_matching_lidar_pcs(self, iteration: int, num_samples: Optional[int], time_horizon: float, look_into_future: bool) -> Generator[LidarPc, None, None]:
        """
        Find the best matching lidar_pcs to the desired samples and time horizon
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future, if None it will be deduced from the DB
        :param time_horizon: the desired horizon to the future
        :param look_into_future: if True, we will iterate into next lidar_pc otherwise we will iterate through prev
        :return: lidar_pcs matching to database indices
        """
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)
        return cast(Generator[LidarPc, None, None], get_sampled_lidarpcs_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, look_into_future))

    def _extract_expert_trajectory(self, max_future_seconds: int=60) -> Generator[EgoState, None, None]:
        """
        Extract expert trajectory with specified time parameters. If initial lidar pc does not have enough history/future
            only available time will be extracted
        :param max_future_seconds: time to future which should be considered for route extraction [s]
        :return: list of expert ego states
        """
        minimal_required_future_time_available = 0.5
        end_log_time_us = get_end_sensor_time_from_db(self._log_file, get_lidarpc_sensor_data())
        max_future_time = min((end_log_time_us - self._initial_lidar_timestamp) * 1e-06, max_future_seconds)
        if max_future_time < minimal_required_future_time_available:
            return
        for traj in self.get_ego_future_trajectory(0, max_future_time):
            yield traj

    def _create_blob_store_if_needed(self) -> Tuple[LocalStore, Optional[S3Store]]:
        """
        A convenience method that creates the blob stores if it's not already created.
        :return: The created or cached LocalStore and S3Store objects.
        """
        if self._local_store is not None and self._remote_store is not None:
            return (self._local_store, self._remote_store)
        if self._sensor_root is None:
            raise ValueError('sensor_root is not set. Please set the sensor_root to access sensor data.')
        Path(self._sensor_root).mkdir(exist_ok=True)
        self._local_store = LocalStore(self._sensor_root)
        if os.getenv('NUPLAN_DATA_STORE', '') == 's3':
            s3_url = os.getenv('NUPLAN_DATA_ROOT_S3_URL', '')
            self._remote_store = S3Store(os.path.join(s3_url, 'sensor_blobs'), show_progress=True)
        return (self._local_store, self._remote_store)

    def _get_sensor_data_from_lidar_pc(self, lidar_pc: LidarPc, channels: List[SensorChannel]) -> Sensors:
        """
        Loads Sensor data given a database LidarPC object.
        :param lidar_pc: The lidar_pc for which to grab the point cloud.
        :param channels: The sensor channels to return.
        :return: The corresponding sensor data.
        """
        local_store, remote_store = self._create_blob_store_if_needed()
        retrieved_images = get_images_from_lidar_tokens(self._log_file, [lidar_pc.token], [cast(str, channel.value) for channel in channels])
        lidar_pcs = {LidarChannel.MERGED_PC: load_point_cloud(cast(LidarPc, lidar_pc), local_store, remote_store)} if LidarChannel.MERGED_PC in channels else None
        images = {CameraChannel[image.channel]: load_image(image, local_store, remote_store) for image in retrieved_images}
        return Sensors(pointcloud=lidar_pcs, images=images if images else None)

