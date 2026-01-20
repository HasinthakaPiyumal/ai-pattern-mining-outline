# Cluster 3

def visualize_scenario(scenario: NuPlanScenario, save_dir: str='/tmp/scenario_visualization/', bokeh_port: int=8899) -> None:
    """
    Visualize a scenario in Bokeh.
    :param scenario: Scenario object to be visualized.
    :param save_dir: Dir to save serialization and visualization artifacts.
    :param bokeh_port: Port that the server bokeh starts to render the generate the visualization will run on.
    """
    map_factory = NuPlanMapFactory(get_maps_db(map_root=scenario.map_root, map_version=scenario.map_version))
    simulation_history = serialize_scenario(scenario)
    simulation_scenario_key = save_scenes_to_dir(scenario=scenario, save_dir=save_dir, simulation_history=simulation_history)
    visualize_scenarios([simulation_scenario_key], map_factory, Path(save_dir), bokeh_port=bokeh_port)

def get_pacifica_parameters() -> VehicleParameters:
    """
    :return VehicleParameters containing parameters of Pacifica Vehicle.
    """
    return VehicleParameters(vehicle_name='pacifica', vehicle_type='gen1', width=1.1485 * 2.0, front_length=4.049, rear_length=1.127, wheel_base=3.089, cog_position_from_rear_axle=1.67, height=1.777)

def get_default_scenario_extraction(scenario_duration: float=15.0, extraction_offset: float=-2.0, subsample_ratio: float=0.5) -> ScenarioExtractionInfo:
    """
    Get default scenario extraction instructions used in visualization.
    :param scenario_duration: [s] Duration of scenario.
    :param extraction_offset: [s] Offset of scenario (e.g. -2 means start scenario 2s before it starts).
    :param subsample_ratio: Scenario resolution.
    :return: Scenario extraction info object.
    """
    return ScenarioExtractionInfo(DEFAULT_SCENARIO_NAME, scenario_duration, extraction_offset, subsample_ratio)

def get_default_scenario_from_token(data_root: str, log_file_full_path: str, token: str, map_root: str, map_version: str) -> NuPlanScenario:
    """
    Build a scenario with default parameters for visualization.
    :param data_root: The root directory to use for looking for db files.
    :param log_file_full_path: The full path to the log db file to use.
    :param token: Lidar pc token to be used as anchor for the scenario.
    :param map_root: The root directory to use for looking for maps.
    :param map_version: The map version to use.
    :return: Instantiated scenario object.
    """
    timestamp = get_sensor_data_token_timestamp_from_db(log_file_full_path, get_lidarpc_sensor_data(), token)
    map_name = get_sensor_token_map_name_from_db(log_file_full_path, get_lidarpc_sensor_data(), token)
    return NuPlanScenario(data_root=data_root, log_file_load_path=log_file_full_path, initial_lidar_token=token, initial_lidar_timestamp=timestamp, scenario_type=DEFAULT_SCENARIO_NAME, map_root=map_root, map_version=map_version, map_name=map_name, scenario_extraction_info=get_default_scenario_extraction(), ego_vehicle_parameters=get_pacifica_parameters())

def get_sensor_token_map_name_from_db(log_file: str, sensor_source: SensorDataSource, token: str) -> Optional[str]:
    """
    Get the map name for a provided sensor token.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param token: The token for which to get the map name.
    :return: The map name for the token, if found.
    """
    query = f'\n    SELECT map_version\n    FROM log AS l\n    INNER JOIN {sensor_source.sensor_table} AS sensor\n        ON sensor.log_token = l.token\n    INNER JOIN {sensor_source.table} AS sensor_data\n        ON sensor_data.{sensor_source.sensor_token_column} = sensor.token\n    WHERE sensor_data.token = ?;\n    '
    result = execute_one(query, (bytearray.fromhex(token),), log_file)
    return None if result is None else result['map_version']

def scenario_dropdown_handler(change: Any) -> None:
    """
        Dropdown handler that randomly chooses a scenario from the selected scenario type and renders it.
        :param change: Object containing scenario selection.
        """
    with out:
        clear_output()
        logger.info('Randomly rendering a scenario...')
        scenario_type = str(change.new)
        log_db_file, token = random.choice(scenario_type_token_map[scenario_type])
        scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)
        visualize_scenario(scenario, bokeh_port=bokeh_port)

class TestTutorialUtils(unittest.TestCase):
    """Unit tests for tutorial_utils.py."""

    def test_scenario_visualization_utils(self) -> None:
        """Test if scenario visualization utils work as expected."""
        visualize_nuplan_scenarios(data_root=NUPLAN_DATA_ROOT, db_files=NUPLAN_DB_FILES, map_root=NUPLAN_MAPS_ROOT, map_version=NUPLAN_MAP_VERSION)

    def test_scenario_rendering(self) -> None:
        """Test if scenario rendering works."""
        bokeh_port = 8999
        output_notebook()
        scenario_type_token_map = get_scenario_type_token_map(NUPLAN_DB_FILES)
        available_keys = list(scenario_type_token_map.keys())
        log_db, token = scenario_type_token_map[available_keys[0]][0]
        scenario = get_default_scenario_from_token(NUPLAN_DATA_ROOT, log_db, token, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION)
        for _ in range(2):
            visualize_scenario(scenario, bokeh_port=bokeh_port)
        for server in curstate().uuid_to_server.values():
            self.assertEqual(bokeh_port, server.port)

    def test_start_event_loop_if_needed(self) -> None:
        """Tests if start_event_loop_if_needed works."""

        async def test_fn() -> int:
            """Minimal async function"""
            return 1
        start_event_loop_if_needed()
        _ = asyncio.get_event_loop()
        start_event_loop_if_needed()
        _ = asyncio.get_event_loop()
        asyncio.run(test_fn())
        with self.assertRaises(RuntimeError):
            _ = asyncio.get_event_loop()
        start_event_loop_if_needed()
        _ = asyncio.get_event_loop()

def get_sensor_token_by_index_from_db(log_file: str, sensor_source: SensorDataSource, index: int) -> Optional[str]:
    """
    Get the N-th sensor token ordered chronologically by timestamp from a particular channel.
    This is primarily used for unit testing.
    If the index does not exist (e.g. index = 10,000 in a log file with 1000 entries),
        then the result will be None.
    Only non-negative integer indexes are supported.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param index: The 0-indexed integer index of the lidarpc token to retrieve.
    :return: The token, if it exists.
    """
    if index < 0:
        raise ValueError(f'Index of {index} was supplied to get_lidarpc_token_by_index_from_db(), which is negative.')
    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)
    query = f'\n    WITH ordered AS\n    (\n        SELECT  token,\n                lidar_token,\n                ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS row_num\n        FROM {sensor_source.table}\n    )\n    SELECT token\n    FROM ordered\n    WHERE (row_num - 1) = ?\n        AND {sensor_source.sensor_token_column} = ?;\n    '
    result = execute_one(query, [index, bytearray.fromhex(sensor_token)], log_file)
    return None if result is None else str(result['token'].hex())

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

def get_sample_ego_state(center: Optional[StateSE2]=None, time_us: Optional[int]=0) -> EgoState:
    """
    Creates a sample EgoState.
    :param center: Vehicle's position. If none it uses the same position returned by get_sample_pose()
    :param time_us: Time in microseconds
    :return: A sample EgoState with arbitrary parameters
    """
    return EgoState(car_footprint=get_sample_car_footprint(center), dynamic_car_state=get_sample_dynamic_car_state(), tire_steering_angle=0.2, time_point=TimePoint(time_us), is_in_auto_mode=False)

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

def get_sample_dynamic_car_state(rear_axle_to_center_dist: float=1.44) -> DynamicCarState:
    """
    Creates a sample DynamicCarState.
    :param rear_axle_to_center_dist: distance between rear axle and center [m]
    :return: A sample DynamicCarState with arbitrary parameters
    """
    return DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist, StateVector2D(1.0, 2.0), StateVector2D(0.1, 0.2))

def get_sample_car_footprint(center: Optional[StateSE2]=None) -> CarFootprint:
    """
    Creates a sample CarFootprint.
    :param center: Vehicle's position. If none it uses the same position returned by get_sample_pose()
    :return: A sample CarFootprint with arbitrary parameters
    """
    if center:
        return CarFootprint.build_from_center(center=center, vehicle_parameters=get_pacifica_parameters())
    else:
        return CarFootprint.build_from_center(center=get_sample_oriented_box().center, vehicle_parameters=get_pacifica_parameters())

class TestCarFootprint(unittest.TestCase):
    """Tests CarFoorprint class"""

    def setUp(self) -> None:
        """Sets sample parameters for testing"""
        self.center_position_from_rear_axle = get_pacifica_parameters().rear_axle_to_center

    def test_car_footprint_creation(self) -> None:
        """Checks that the car footprint is created correctly, in particular the point of interest."""
        car_footprint = CarFootprint.build_from_rear_axle(get_sample_pose(), get_pacifica_parameters())
        self.assertAlmostEqual(car_footprint.rear_axle_to_center_dist, self.center_position_from_rear_axle)
        expected_values = {OrientedBoxPointType.FRONT_BUMPER: (1.0, 6.049), OrientedBoxPointType.REAR_BUMPER: (1.0, 0.873), OrientedBoxPointType.FRONT_LEFT: (-0.1485, 6.049), OrientedBoxPointType.REAR_LEFT: (-0.1485, 0.873), OrientedBoxPointType.REAR_RIGHT: (2.1485, 0.873), OrientedBoxPointType.FRONT_RIGHT: (2.1485, 6.049), OrientedBoxPointType.CENTER: (1.0, 3.461)}
        for point, position in expected_values.items():
            np.testing.assert_array_almost_equal(position, tuple(car_footprint.corner(point)))
        np.testing.assert_array_almost_equal(expected_values[OrientedBoxPointType.FRONT_LEFT], tuple(car_footprint.get_point_of_interest(OrientedBoxPointType.FRONT_LEFT)), 6)

class TestColor(TestCase):
    """
    Test color.
    """

    def setUp(self) -> None:
        """
        Set up.
        """
        self.red = 0.1
        self.green = 0.2
        self.blue = 0.3
        self.alpha = 0.5
        self.color = Color(self.red, self.green, self.blue, self.alpha, ColorType.FLOAT)
        self.color_255 = Color(self.red, self.green, self.blue, self.alpha, ColorType.INT)

    def test_init(self) -> None:
        """
        Test initialisation.
        """
        self.assertEqual(self.color.red, self.red)
        self.assertEqual(self.color.green, self.green)
        self.assertEqual(self.color.blue, self.blue)
        self.assertEqual(self.color.alpha, self.alpha)
        self.assertEqual(self.color.serialize_to, ColorType.FLOAT)
        self.assertEqual(self.color_255.serialize_to, ColorType.INT)

    def test_post_init_invalid_type(self) -> None:
        """
        Tests that post init raises TypeError when passing any non-float types.
        """
        with self.assertRaises(TypeError):
            Color(1.0, 0.5, 0.0, '1')

    def test_post_init_invalid_range(self) -> None:
        """
        Tests that post init raises ValueError when passing values outside of range 0-255.
        """
        with self.assertRaises(ValueError):
            Color(1.0, 0.5, 0.0, 100.0)
        with self.assertRaises(ValueError):
            Color(1.0, 0.5, 0.0, -1.0)

    def test_iter(self) -> None:
        """
        Tests iteration of RGBA components.
        """
        result = [color for color in self.color]
        self.assertEqual(result[0], self.red)
        self.assertEqual(result[1], self.green)
        self.assertEqual(result[2], self.blue)
        self.assertEqual(result[3], self.alpha)

    def test_iter_255(self) -> None:
        """
        Tests iteration of RGBA components, with color type specified as int.
        """
        result = [color for color in self.color_255]
        self.assertEqual(result[0], int(self.red * 255))
        self.assertEqual(result[1], int(self.green * 255))
        self.assertEqual(result[2], int(self.blue * 255))
        self.assertEqual(result[3], int(self.alpha * 255))

    def test_to_list(self) -> None:
        """
        Tests to list method.
        """
        result = self.color.to_list()
        self.assertEqual(result, [self.red, self.green, self.blue, self.alpha])

    def test_mul(self) -> None:
        """
        Tests multiplication operation without clamping ie. results already in range (0-255).
        """
        result = self.color * 2
        self.assertEqual(result, Color(self.red * 2, self.green * 2, self.blue * 2, self.alpha * 2))

    def test_mul_clamp(self) -> None:
        """
        Tests clamping of values to range (0-255) after multiplication.
        """
        red = 0.5
        green = 0.7
        blue = 0.0
        alpha = 1.0
        color = Color(red, green, blue, alpha) * 2
        self.assertEqual(color.red, 1.0)
        self.assertEqual(color.green, 1.0)
        self.assertEqual(color.blue, 0.0)
        self.assertEqual(color.alpha, 1.0)

    def test_mul_255(self) -> None:
        """
        Tests multiplication operation with a color of integer color type preserves color type
        """
        result = self.color_255 * 2
        self.assertEqual(result, Color(self.red * 2, self.green * 2, self.blue * 2, self.alpha * 2, ColorType.INT))

    @patch('nuplan.planning.utils.color.Color.__mul__')
    def test_rmul(self, mock_mul: Mock) -> None:
        """
        Tests reverse multiplication operation.
        """
        result = 2 * self.color
        mock_mul.assert_called_once_with(2)
        self.assertEqual(result, mock_mul.return_value)

def to_scene_ego_from_ego_state(ego_pose: Union[EgoState, EgoTemporalState]) -> EgoScene:
    """
    :param ego_pose: temporal state trajectory
    :return serialized scene
    """
    ego_temporal_state = EgoTemporalState(ego_pose) if isinstance(ego_pose, EgoState) else ego_pose
    current_state = ego_temporal_state.ego_current_state
    future = [to_scene_waypoint(state, -current_state.time_point.time_s) for prediction in ego_temporal_state.predictions for state in prediction.valid_waypoints] if ego_temporal_state.predictions else []
    past = [to_scene_waypoint(state, -current_state.time_point.time_s) for state in ego_temporal_state.past_trajectory.valid_waypoints] if ego_temporal_state.past_trajectory else []
    predictions = {'color': Color(red=1, green=0, blue=0, alpha=1, serialize_to=ColorType.FLOAT).to_list(), 'states': past + future}
    rear_axle = current_state.rear_axle
    return EgoScene(acceleration=0.0, pose=rear_axle, speed=current_state.dynamic_car_state.speed, prediction=predictions)

def to_scene_waypoint(waypoint: Waypoint, time_offset: Optional[float]=None) -> Dict[str, Any]:
    """
    Convert waypoint to scene object that can be visualized as predictions, and offset timestamp if desired
    :param waypoint: to be converted
    :param time_offset: if None, no offset will be done, otherwise offset time stamp by this number
    :return: serialized scene
    """
    return {'pose': [waypoint.center.x, waypoint.center.y, waypoint.center.heading], 'timestamp': waypoint.time_point.time_s + time_offset if time_offset else 0.0}

def to_scene_trajectory_state_from_ego_state(ego_state: EgoState) -> TrajectoryState:
    """
    Convert ego state into scene structure for states in a trajectory.
    :param ego_state: ego state.
    :return: state in scene format.
    """
    return TrajectoryState(pose=ego_state.rear_axle, speed=ego_state.dynamic_car_state.speed, velocity_2d=[ego_state.dynamic_car_state.rear_axle_velocity_2d.x, ego_state.dynamic_car_state.rear_axle_velocity_2d.y], lateral=[0.0, 0.0], acceleration=[ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, ego_state.dynamic_car_state.rear_axle_acceleration_2d.y], tire_steering_angle=ego_state.tire_steering_angle)

def to_scene_trajectory_from_list_ego_state(trajectory: List[EgoState], color: Color) -> Trajectory:
    """
    Convert list of ego states and a color into a scene structure for a trajectory.
    :param trajectory: a list of states.
    :param color: color [R, G, B, A].
    :return: Trajectory in scene format.
    """
    trajectory_states = [to_scene_trajectory_state_from_ego_state(state) for state in trajectory]
    return Trajectory(color=color, states=trajectory_states)

def to_scene_trajectory_state_from_waypoint(waypoint: Waypoint) -> TrajectoryState:
    """
    Convert ego state into scene structure for states in a trajectory.
    :param waypoint: waypoint in a trajectory.
    :return: state in scene format.
    """
    return TrajectoryState(pose=waypoint.center, speed=waypoint.velocity.magnitude(), velocity_2d=[waypoint.velocity.x, waypoint.velocity.y] if waypoint.velocity else [0, 0], lateral=[0.0, 0.0])

def to_scene_trajectory_from_list_waypoint(trajectory: List[Waypoint], color: Color) -> Trajectory:
    """
    Convert list of waypoints and a color into a scene structure for a trajectory.
    :param trajectory: a list of states.
    :param color: color [R, G, B, A].
    :return: Trajectory in scene format.
    """
    trajectory_states = [to_scene_trajectory_state_from_waypoint(state) for state in trajectory]
    return Trajectory(color=color, states=trajectory_states)

def to_scene_goal_from_state(state: StateSE2) -> GoalScene:
    """
    Convert car footprint to scene structure for ego.
    :param car_footprint: CarFootprint of ego.
    :return Ego in scene format.
    """
    return GoalScene(pose=state)

def to_scene_ego_from_car_footprint(car_footprint: CarFootprint) -> EgoScene:
    """
    Convert car footprint to scene structure for ego.
    :param car_footprint: CarFootprint of ego.
    :return Ego in scene format.
    """
    return EgoScene(acceleration=0.0, pose=car_footprint.rear_axle, speed=0.0)

def to_scene_boxes(tracked_objects: TrackedObjects) -> Dict[str, Any]:
    """
    Convert tracked_objects into a scene.
    :param tracked_objects: List of boxes in global coordinates.
    :return dictionary which should be placed into scene["world"].
    """
    tracked_object_dictionaries = {}
    for track_object_type_name, tracked_object_type in tracked_object_types.items():
        objects = [to_scene_box(tracked_object, track_id=tracked_object.track_token) for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type)]
        tracked_object_dictionaries[track_object_type_name] = objects
    return tracked_object_dictionaries

def to_scene_box(tracked_object: TrackedObject, track_id: str) -> Dict[str, Any]:
    """
    Convert tracked_object into json representation.
    :param tracked_object: tracked_object representation.
    :param track_id: unique id of a track.
    :return json representation of an agent.
    """
    center_x = tracked_object.center.x
    center_y = tracked_object.center.y
    center_heading = tracked_object.center.heading
    if tracked_object.tracked_object_type in AGENT_TYPES:
        speed = np.hypot(tracked_object.velocity.x, tracked_object.velocity.y)
    else:
        speed = 0
    if track_id is None:
        track_id = 'null'
    scene = {'active': True, 'real': True, 'speed': speed if not np.isnan(speed) else 0.0, 'box': {'pose': [center_x, center_y, center_heading], 'size': [tracked_object.box.width, tracked_object.box.length]}, 'id': track_id, 'type': tracked_object.tracked_object_type.fullname, 'tooltip': f'avtest_track_id: {track_id}\ntrack_token: {tracked_object.metadata.track_token}\ntoken: {tracked_object.metadata.token}\ncategory_name: {tracked_object.metadata.category_name}\ntrack_id: {tracked_object.metadata.track_id}\ntype: {tracked_object.tracked_object_type.fullname}\nvelocity: {tracked_object.velocity}'}
    if tracked_object.tracked_object_type == TrackedObjectType.PEDESTRIAN:
        scene['box']['radius'] = 0.5
    return scene

def to_scene_from_ego_and_boxes(ego_pose: StateSE2, tracked_objects: TrackedObjects, map_name: str, set_camera: bool=False) -> Dict[str, Any]:
    """
    Extract scene from ego_pose and boxes.
    :param ego_pose: Ego Position.
    :param tracked_objects: list of actors in global coordinate frame.
    :param map_name: map name.
    :param set_camera: True if we wish to also set camera view.
    :return scene.
    """
    world = to_scene_boxes(tracked_objects)
    map_name_without_suffix = str(pathlib.Path(map_name).stem)
    scene: Dict[str, Any] = {'map': {'area': map_name_without_suffix}, 'world': world, 'ego': dict(to_scene_ego_from_car_footprint(CarFootprint.build_from_center(ego_pose, get_pacifica_parameters())))}
    if set_camera:
        ego_pose = scene['ego']['pose']
        ego_x = ego_pose[0]
        ego_y = ego_pose[1]
        ego_heading = ego_pose[2]
        bearing_rad = np.fmod(ego_heading, np.pi * 2)
        if bearing_rad < 0:
            bearing_rad += np.pi * 2
        bearing_rad = 1.75 - bearing_rad / (np.pi * 2)
        if bearing_rad >= 1:
            bearing_rad -= 1.0
        scene['camera'] = {'pitch': 50, 'scale': 2500000, 'bearing': bearing_rad, 'lookat': [ego_x, ego_y, 0.0]}
    return scene

def _to_scene_agent_prediction(tracked_object: TrackedObject, color: Color) -> Dict[str, Any]:
    """
    Extract agent's predicted states from TrackedObject to scene.
    :param tracked_object: tracked_object representation.
    :param color: color [R, G, B, A].
    :return a prediction scene.
    """

    def extract_prediction_state(pose: StateSE2, time_delta: float, speed: float) -> Dict[str, Any]:
        """
        Extract the representation of prediction state for scene.
        :param pose: Track pose.
        :param time_delta: Time difference from initial timestamp.
        :param speed: Speed of track.
        :return: Scene-like dict containing prediction state.
        """
        return {'pose': [pose.x, pose.y, pose.heading], 'polygon': [[pose.x, pose.y]], 'timestamp': time_delta, 'speed': speed}
    past_states = [] if not tracked_object.past_trajectory else [extract_prediction_state(waypoint.oriented_box.center, tracked_object.metadata.timestamp_s - waypoint.time_point.time_s, waypoint.velocity.magnitude() if waypoint.velocity is not None else 0) for waypoint in tracked_object.past_trajectory.waypoints if waypoint]
    future_states = [extract_prediction_state(waypoint.oriented_box.center, waypoint.time_point.time_s - mode.waypoints[0].time_point.time_s, waypoint.velocity.magnitude() if waypoint.velocity is not None else 0) for mode in tracked_object.predictions for waypoint in mode.waypoints if waypoint]
    return {'id': tracked_object.metadata.track_id, 'color': color.to_list(), 'size': [tracked_object.box.width, tracked_object.box.length], 'states': past_states + future_states}

def extract_prediction_state(pose: StateSE2, time_delta: float, speed: float) -> Dict[str, Any]:
    """
        Extract the representation of prediction state for scene.
        :param pose: Track pose.
        :param time_delta: Time difference from initial timestamp.
        :param speed: Speed of track.
        :return: Scene-like dict containing prediction state.
        """
    return {'pose': [pose.x, pose.y, pose.heading], 'polygon': [[pose.x, pose.y]], 'timestamp': time_delta, 'speed': speed}

def to_scene_agent_prediction_from_boxes(tracked_objects: TrackedObjects, color: Color) -> List[Dict[str, Any]]:
    """
    Convert predicted observations into prediction dictionary.
    :param tracked_objects: List of tracked_objects in global coordinates.
    :param color: color [R, G, B, A].
    :return scene.
    """
    return [_to_scene_agent_prediction(tracked_object, color) for tracked_object in tracked_objects if tracked_object.predictions is not None]

def to_scene_agent_prediction_from_boxes_separate_color(tracked_objects: TrackedObjects, color_vehicles: Color, color_pedestrians: Color, color_bikes: Color) -> List[Dict[str, Any]]:
    """
    Convert predicted observations into prediction dictionary.
    :param tracked_objects: List of tracked_objects in global coordinates.
    :param color_vehicles: color [R, G, B, A] for vehicles predictions.
    :param color_pedestrians: color [R, G, B, A] for pedestrians predictions.
    :param color_bikes: color [R, G, B, A] for bikes predictions.
    :return scene.
    """
    predictions = []
    for tracked_object in tracked_objects:
        if tracked_object.predictions is None:
            continue
        if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE:
            color = color_vehicles
        elif tracked_object.tracked_object_type == TrackedObjectType.PEDESTRIAN:
            color = color_pedestrians
        elif tracked_object.tracked_object_type == TrackedObjectType.BICYCLE:
            color = color_bikes
        else:
            color = Color(0, 0, 0, 1, ColorType.FLOAT)
        predictions.append(_to_scene_agent_prediction(tracked_object, color))
    return predictions

class TestTrajectoryState(unittest.TestCase):
    """
    Test scene dataclass TrajectoryState
    """

    def setUp(self) -> None:
        """
        Set up
        """
        self.pose_x = 1.12
        self.pose_y = 2.11
        self.pose_heading = 0.29
        self.pose = StateSE2(self.pose_x, self.pose_y, self.pose_heading)
        self.speed = 1.23
        self.velocity_2d = [0.12, 0.54]
        self.lateral = [0.0, 0.0]
        self.acceleration = [0.32, 0.43]
        self.trajectory_state = TrajectoryState(pose=self.pose, speed=self.speed, velocity_2d=self.velocity_2d, lateral=self.lateral, acceleration=self.acceleration)

    def test_init(self) -> None:
        """
        Tests TrajectoryState initialization
        """
        self.assertEqual(self.trajectory_state.pose, self.pose)
        self.assertEqual(self.trajectory_state.speed, self.speed)
        self.assertEqual(self.trajectory_state.velocity_2d, self.velocity_2d)
        self.assertEqual(self.trajectory_state.lateral, self.lateral)
        self.assertEqual(self.trajectory_state.acceleration, self.acceleration)
        self.assertIsNone(self.trajectory_state.tire_steering_angle)

    def test_serialize(self) -> None:
        """
        Tests whether TrajectoryState is serializable
        """
        result = dict(self.trajectory_state)
        self.assertEqual(result, {'pose': [self.pose_x, self.pose_y, self.pose_heading], 'speed': self.speed, 'velocity_2d': self.velocity_2d, 'lateral': self.lateral, 'acceleration': self.acceleration})
        self.assertFalse('tire_steering_angle' in result.keys())

    def test_update(self) -> None:
        """
        Tests whether TrajectoryState is compatible with dict.update()
        """
        scene = {'example': 'unchanged', 'pose': 'old_pose', 'speed': 'old_speed'}
        scene.update(self.trajectory_state)
        self.assertEqual(scene, {'example': 'unchanged', 'pose': [self.pose_x, self.pose_y, self.pose_heading], 'speed': self.speed, 'velocity_2d': self.velocity_2d, 'lateral': self.lateral, 'acceleration': self.acceleration})

class TestToScene(unittest.TestCase):
    """
    Test scene conversions in to_scene.py
    """

    def test_to_scene_trajectory_state_from_ego_state(self) -> None:
        """
        Tests conversion from ego state to trajectory state (scene class)
        """
        ego_state = Mock(spec=EgoState)
        ego_state.rear_axle = [1.12, 2.11, 0.29]
        ego_state.dynamic_car_state.speed = 1.23
        ego_state.dynamic_car_state.rear_axle_velocity_2d.x = 0.12
        ego_state.dynamic_car_state.rear_axle_velocity_2d.y = 0.54
        ego_state.dynamic_car_state.rear_axle_acceleration_2d.x = 0.32
        ego_state.dynamic_car_state.rear_axle_acceleration_2d.y = 0.43
        ego_state.tire_steering_angle = 0.21
        result = to_scene_trajectory_state_from_ego_state(ego_state)
        self.assertEqual(result.pose, [1.12, 2.11, 0.29])
        self.assertEqual(result.speed, 1.23)
        self.assertEqual(result.velocity_2d, [0.12, 0.54])
        self.assertEqual(result.lateral, [0.0, 0.0])
        self.assertEqual(result.acceleration, [0.32, 0.43])
        self.assertEqual(result.tire_steering_angle, 0.21)

    @patch('nuplan.planning.utils.serialization.to_scene.to_scene_trajectory_state_from_ego_state')
    def test_to_scene_trajectory_from_list_ego_state(self, mock_to_trajectory_state: Mock) -> None:
        """
        Tests conversion of list of ego states to trajectory structure (scene class)
        """
        mock_to_trajectory_state.side_effect = lambda state: 't_' + state
        ego_states = ['s1', 's2']
        color = Color(0.5, 0.2, 0.5, 1)
        result = to_scene_trajectory_from_list_ego_state(ego_states, color)
        self.assertEqual(result.color.to_list(), [0.5, 0.2, 0.5, 1])
        self.assertEqual(result.states, ['t_s1', 't_s2'])

    def test_to_scene_trajectory_state_from_waypoint(self) -> None:
        """
        Tests conversion from waypoint to trajectory state (scene class)
        """
        waypoint = Mock(spec=Waypoint)
        waypoint.center = [1.12, 2.11, 0.29]
        waypoint.velocity.magnitude.return_value = 1.23
        waypoint.velocity.x = 0.12
        waypoint.velocity.y = 0.54
        result = to_scene_trajectory_state_from_waypoint(waypoint)
        self.assertEqual(result.pose, [1.12, 2.11, 0.29])
        self.assertEqual(result.speed, 1.23)
        self.assertEqual(result.velocity_2d, [0.12, 0.54])
        self.assertEqual(result.lateral, [0.0, 0.0])
        self.assertEqual(result.acceleration, None)
        self.assertEqual(result.tire_steering_angle, None)

    @patch('nuplan.planning.utils.serialization.to_scene.to_scene_trajectory_state_from_waypoint')
    def test_to_scene_trajectory_from_list_waypoint(self, mock_to_trajectory_state: Mock) -> None:
        """
        Tests conversion of list of waypoints to trajectory structure (scene class)
        """
        mock_to_trajectory_state.side_effect = lambda state: 'w_' + state
        waypoints = ['s1', 's2']
        color = Color(0.5, 0.2, 0.5, 1)
        result = to_scene_trajectory_from_list_waypoint(waypoints, color)
        self.assertEqual(result.color.to_list(), [0.5, 0.2, 0.5, 1])
        self.assertEqual(result.states, ['w_s1', 'w_s2'])

class TestTrajectory(unittest.TestCase):
    """
    Test scene dataclass Trajectory
    """

    def setUp(self) -> None:
        """
        Set up
        """
        self.color = Color(1, 0.5, 0, 1, ColorType.INT)
        self.states = [Mock(spec=TrajectoryState), Mock(spec=TrajectoryState)]
        self.trajectory_structure = Trajectory(color=self.color, states=self.states)

    def test_init(self) -> None:
        """
        Tests TrajectoryState initialization
        """
        self.assertEqual(self.trajectory_structure.color, self.color)
        self.assertEqual(self.trajectory_structure.states, self.states)

    @patch('nuplan.planning.utils.serialization.scene.type')
    def test_serialize(self, mock_type: Mock) -> None:
        """
        Tests whether TrajectoryState is serializable
        """
        self.states[0].__iter__ = Mock(return_value=iter([['state_0', 'value_0']]))
        self.states[1].__iter__ = Mock(return_value=iter([['state_1', 'value_1']]))
        mock_type.side_effect = lambda x: TrajectoryState if isinstance(x, TrajectoryState) else type(x)
        result = dict(self.trajectory_structure)
        self.assertEqual(result, {'color': self.color.to_list(), 'states': [{'state_0': 'value_0'}, {'state_1': 'value_1'}]})

    def test_update(self) -> None:
        """
        Tests whether Trajectory is compatible with dict.update()
        """
        scene = {'example': 'unchanged', 'color': 'old_color'}
        scene.update(self.trajectory_structure)
        self.assertEqual(scene, {'example': 'unchanged', 'color': self.color.to_list(), 'states': self.states})

class LQRTracker(AbstractTracker):
    """
    Implements an LQR tracker for a kinematic bicycle model.

    We decouple into two subsystems, longitudinal and lateral, with small angle approximations for linearization.
    We then solve two sequential LQR subproblems to find acceleration and steering rate inputs.

    Longitudinal Subsystem:
        States: [velocity]
        Inputs: [acceleration]
        Dynamics (continuous time):
            velocity_dot = acceleration

    Lateral Subsystem (After Linearization/Small Angle Approximation):
        States: [lateral_error, heading_error, steering_angle]
        Inputs: [steering_rate]
        Parameters: [velocity, curvature]
        Dynamics (continuous time):
            lateral_error_dot  = velocity * heading_error
            heading_error_dot  = velocity * (steering_angle / wheelbase_length - curvature)
            steering_angle_dot = steering_rate

    The continuous time dynamics are discretized using Euler integration and zero-order-hold on the input.
    In case of a stopping reference, we use a simplified stopping P controller instead of LQR.

    The final control inputs passed on to the motion model are:
        - acceleration
        - steering_rate
    """

    def __init__(self, q_longitudinal: npt.NDArray[np.float64], r_longitudinal: npt.NDArray[np.float64], q_lateral: npt.NDArray[np.float64], r_lateral: npt.NDArray[np.float64], discretization_time: float, tracking_horizon: int, jerk_penalty: float, curvature_rate_penalty: float, stopping_proportional_gain: float, stopping_velocity: float, vehicle: VehicleParameters=get_pacifica_parameters()):
        """
        Constructor for LQR controller
        :param q_longitudinal: The weights for the Q matrix for the longitudinal subystem.
        :param r_longitudinal: The weights for the R matrix for the longitudinal subystem.
        :param q_lateral: The weights for the Q matrix for the lateral subystem.
        :param r_lateral: The weights for the R matrix for the lateral subystem.
        :param discretization_time: [s] The time interval used for discretizing the continuous time dynamics.
        :param tracking_horizon: How many discrete time steps ahead to consider for the LQR objective.
        :param stopping_proportional_gain: The proportional_gain term for the P controller when coming to a stop.
        :param stopping_velocity: [m/s] The velocity below which we are deemed to be stopping and we don't use LQR.
        :param vehicle: Vehicle parameters
        """
        assert len(q_longitudinal) == 1, 'q_longitudinal should have 1 element (velocity).'
        assert len(r_longitudinal) == 1, 'r_longitudinal should have 1 element (acceleration).'
        self._q_longitudinal: npt.NDArray[np.float64] = np.diag(q_longitudinal)
        self._r_longitudinal: npt.NDArray[np.float64] = np.diag(r_longitudinal)
        assert len(q_lateral) == 3, 'q_lateral should have 3 elements (lateral_error, heading_error, steering_angle).'
        assert len(r_lateral) == 1, 'r_lateral should have 1 element (steering_rate).'
        self._q_lateral: npt.NDArray[np.float64] = np.diag(q_lateral)
        self._r_lateral: npt.NDArray[np.float64] = np.diag(r_lateral)
        for attr in ['_q_lateral', '_q_longitudinal']:
            assert np.all(np.diag(getattr(self, attr)) >= 0.0), f'self.{attr} must be positive semidefinite.'
        for attr in ['_r_lateral', '_r_longitudinal']:
            assert np.all(np.diag(getattr(self, attr)) > 0.0), f'self.{attr} must be positive definite.'
        assert discretization_time > 0.0, 'The discretization_time should be positive.'
        assert tracking_horizon > 1, 'We expect the horizon to be greater than 1 - else steering_rate has no impact with Euler integration.'
        self._discretization_time = discretization_time
        self._tracking_horizon = tracking_horizon
        self._wheel_base = vehicle.wheel_base
        assert jerk_penalty > 0.0, 'The jerk penalty must be positive.'
        assert curvature_rate_penalty > 0.0, 'The curvature rate penalty must be positive.'
        self._jerk_penalty = jerk_penalty
        self._curvature_rate_penalty = curvature_rate_penalty
        assert stopping_proportional_gain > 0, 'stopping_proportional_gain has to be greater than 0.'
        assert stopping_velocity > 0, 'stopping_velocity has to be greater than 0.'
        self._stopping_proportional_gain = stopping_proportional_gain
        self._stopping_velocity = stopping_velocity

    def track_trajectory(self, current_iteration: SimulationIteration, next_iteration: SimulationIteration, initial_state: EgoState, trajectory: AbstractTrajectory) -> DynamicCarState:
        """Inherited, see superclass."""
        initial_velocity, initial_lateral_state_vector = self._compute_initial_velocity_and_lateral_state(current_iteration, initial_state, trajectory)
        reference_velocity, curvature_profile = self._compute_reference_velocity_and_curvature_profile(current_iteration, trajectory)
        should_stop = reference_velocity <= self._stopping_velocity and initial_velocity <= self._stopping_velocity
        if should_stop:
            accel_cmd, steering_rate_cmd = self._stopping_controller(initial_velocity, reference_velocity)
        else:
            accel_cmd = self._longitudinal_lqr_controller(initial_velocity, reference_velocity)
            velocity_profile = _generate_profile_from_initial_condition_and_derivatives(initial_condition=initial_velocity, derivatives=np.ones(self._tracking_horizon, dtype=np.float64) * accel_cmd, discretization_time=self._discretization_time)[:self._tracking_horizon]
            steering_rate_cmd = self._lateral_lqr_controller(initial_lateral_state_vector, velocity_profile, curvature_profile)
        return DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist, rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=StateVector2D(accel_cmd, 0), tire_steering_rate=steering_rate_cmd)

    def _compute_initial_velocity_and_lateral_state(self, current_iteration: SimulationIteration, initial_state: EgoState, trajectory: AbstractTrajectory) -> Tuple[float, npt.NDArray[np.float64]]:
        """
        This method projects the initial tracking error into vehicle/Frenet frame.  It also extracts initial velocity.
        :param current_iteration: Used to get the current time.
        :param initial_state: The current state for ego.
        :param trajectory: The reference trajectory we are tracking.
        :return: Initial velocity [m/s] and initial lateral state.
        """
        initial_trajectory_state = trajectory.get_state_at_time(current_iteration.time_point)
        x_error = initial_state.rear_axle.x - initial_trajectory_state.rear_axle.x
        y_error = initial_state.rear_axle.y - initial_trajectory_state.rear_axle.y
        heading_reference = initial_trajectory_state.rear_axle.heading
        lateral_error = -x_error * np.sin(heading_reference) + y_error * np.cos(heading_reference)
        heading_error = angle_diff(initial_state.rear_axle.heading, heading_reference, 2 * np.pi)
        initial_velocity = initial_state.dynamic_car_state.rear_axle_velocity_2d.x
        initial_lateral_state_vector: npt.NDArray[np.float64] = np.array([lateral_error, heading_error, initial_state.tire_steering_angle], dtype=np.float64)
        return (initial_velocity, initial_lateral_state_vector)

    def _compute_reference_velocity_and_curvature_profile(self, current_iteration: SimulationIteration, trajectory: AbstractTrajectory) -> Tuple[float, npt.NDArray[np.float64]]:
        """
        This method computes reference velocity and curvature profile based on the reference trajectory.
        We use a lookahead time equal to self._tracking_horizon * self._discretization_time.
        :param current_iteration: Used to get the current time.
        :param trajectory: The reference trajectory we are tracking.
        :return: The reference velocity [m/s] and curvature profile [rad] to track.
        """
        times_s, poses = get_interpolated_reference_trajectory_poses(trajectory, self._discretization_time)
        velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile = get_velocity_curvature_profiles_with_derivatives_from_poses(discretization_time=self._discretization_time, poses=poses, jerk_penalty=self._jerk_penalty, curvature_rate_penalty=self._curvature_rate_penalty)
        reference_time = current_iteration.time_point.time_s + self._tracking_horizon * self._discretization_time
        reference_velocity = np.interp(reference_time, times_s[:-1], velocity_profile)
        profile_times = [current_iteration.time_point.time_s + x * self._discretization_time for x in range(self._tracking_horizon)]
        reference_curvature_profile = np.interp(profile_times, times_s[:-1], curvature_profile)
        return (float(reference_velocity), reference_curvature_profile)

    def _stopping_controller(self, initial_velocity: float, reference_velocity: float) -> Tuple[float, float]:
        """
        Apply proportional controller when at near-stop conditions.
        :param initial_velocity: [m/s] The current velocity of ego.
        :param reference_velocity: [m/s] The reference velocity to track.
        :return: Acceleration [m/s^2] and zero steering_rate [rad/s] command.
        """
        accel = -self._stopping_proportional_gain * (initial_velocity - reference_velocity)
        return (accel, 0.0)

    def _longitudinal_lqr_controller(self, initial_velocity: float, reference_velocity: float) -> float:
        """
        This longitudinal controller determines an acceleration input to minimize velocity error at a lookahead time.
        :param initial_velocity: [m/s] The current velocity of ego.
        :param reference_velocity: [m/s] The reference_velocity to track at a lookahead time.
        :return: Acceleration [m/s^2] command based on LQR.
        """
        A: npt.NDArray[np.float64] = np.array([1.0], dtype=np.float64)
        B: npt.NDArray[np.float64] = np.array([self._tracking_horizon * self._discretization_time], dtype=np.float64)
        accel_cmd = self._solve_one_step_lqr(initial_state=np.array([initial_velocity], dtype=np.float64), reference_state=np.array([reference_velocity], dtype=np.float64), Q=self._q_longitudinal, R=self._r_longitudinal, A=A, B=B, g=np.zeros(1, dtype=np.float64), angle_diff_indices=[])
        return float(accel_cmd)

    def _lateral_lqr_controller(self, initial_lateral_state_vector: npt.NDArray[np.float64], velocity_profile: npt.NDArray[np.float64], curvature_profile: npt.NDArray[np.float64]) -> float:
        """
        This lateral controller determines a steering_rate input to minimize lateral errors at a lookahead time.
        It requires a velocity sequence as a parameter to ensure linear time-varying lateral dynamics.
        :param initial_lateral_state_vector: The current lateral state of ego.
        :param velocity_profile: [m/s] The velocity over the entire self._tracking_horizon-step lookahead.
        :param curvature_profile: [rad] The curvature over the entire self._tracking_horizon-step lookahead..
        :return: Steering rate [rad/s] command based on LQR.
        """
        assert len(velocity_profile) == self._tracking_horizon, f'The linearization velocity sequence should have length {self._tracking_horizon} but is {len(velocity_profile)}.'
        assert len(curvature_profile) == self._tracking_horizon, f'The linearization curvature sequence should have length {self._tracking_horizon} but is {len(curvature_profile)}.'
        n_lateral_states = len(LateralStateIndex)
        I: npt.NDArray[np.float64] = np.eye(n_lateral_states, dtype=np.float64)
        A: npt.NDArray[np.float64] = I
        B: npt.NDArray[np.float64] = np.zeros((n_lateral_states, 1), dtype=np.float64)
        g: npt.NDArray[np.float64] = np.zeros(n_lateral_states, dtype=np.float64)
        idx_lateral_error = LateralStateIndex.LATERAL_ERROR
        idx_heading_error = LateralStateIndex.HEADING_ERROR
        idx_steering_angle = LateralStateIndex.STEERING_ANGLE
        input_matrix: npt.NDArray[np.float64] = np.zeros((n_lateral_states, 1), np.float64)
        input_matrix[idx_steering_angle] = self._discretization_time
        for index_step, (velocity, curvature) in enumerate(zip(velocity_profile, curvature_profile)):
            state_matrix_at_step: npt.NDArray[np.float64] = np.eye(n_lateral_states, dtype=np.float64)
            state_matrix_at_step[idx_lateral_error, idx_heading_error] = velocity * self._discretization_time
            state_matrix_at_step[idx_heading_error, idx_steering_angle] = velocity * self._discretization_time / self._wheel_base
            affine_term: npt.NDArray[np.float64] = np.zeros(n_lateral_states, dtype=np.float64)
            affine_term[idx_heading_error] = -velocity * curvature * self._discretization_time
            A = state_matrix_at_step @ A
            B = state_matrix_at_step @ B + input_matrix
            g = state_matrix_at_step @ g + affine_term
        steering_rate_cmd = self._solve_one_step_lqr(initial_state=initial_lateral_state_vector, reference_state=np.zeros(n_lateral_states, dtype=np.float64), Q=self._q_lateral, R=self._r_lateral, A=A, B=B, g=g, angle_diff_indices=[idx_heading_error, idx_steering_angle])
        return float(steering_rate_cmd)

    @staticmethod
    def _solve_one_step_lqr(initial_state: npt.NDArray[np.float64], reference_state: npt.NDArray[np.float64], Q: npt.NDArray[np.float64], R: npt.NDArray[np.float64], A: npt.NDArray[np.float64], B: npt.NDArray[np.float64], g: npt.NDArray[np.float64], angle_diff_indices: List[int]=[]) -> npt.NDArray[np.float64]:
        """
        This function uses LQR to find an optimal input to minimize tracking error in one step of dynamics.
        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.
        :param initial_state: The current state.
        :param reference_state: The desired state in 1 step (according to A,B,g dynamics).
        :param Q: The state tracking 2-norm cost matrix.
        :param R: The input 2-norm cost matrix.
        :param A: The state dynamics matrix.
        :param B: The input dynamics matrix.
        :param g: The offset/affine dynamics term.
        :param angle_diff_indices: The set of state indices for which we need to apply angle differences, if defined.
        :return: LQR optimal input for the 1-step problem.
        """
        state_error_zero_input = A @ initial_state + g - reference_state
        for angle_diff_index in angle_diff_indices:
            state_error_zero_input[angle_diff_index] = angle_diff(state_error_zero_input[angle_diff_index], 0.0, 2 * np.pi)
        lqr_input = -np.linalg.inv(B.T @ Q @ B + R) @ B.T @ Q @ state_error_zero_input
        return lqr_input

class TestKinematicMotionModel(unittest.TestCase):
    """
    Run tests for Kinematic Bicycle Model.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.vehicle = get_pacifica_parameters()
        self.ego_state = get_sample_ego_state()
        self.sampling_time = TimePoint(1000000)
        self.motion_model = KinematicBicycleModel(self.vehicle)
        wheel_base = self.vehicle.wheel_base
        self.longitudinal_speed = self.ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        self.x_dot = self.longitudinal_speed * np.cos(self.ego_state.rear_axle.heading)
        self.y_dot = self.longitudinal_speed * np.sin(self.ego_state.rear_axle.heading)
        self.yaw_dot = self.longitudinal_speed * np.tan(self.ego_state.tire_steering_angle) / wheel_base

    def test_get_state_dot(self) -> None:
        """
        Test get_state_dot for expected results
        """
        state_dot = self.motion_model.get_state_dot(self.ego_state)
        self.assertEqual(state_dot.rear_axle, StateSE2(self.x_dot, self.y_dot, self.yaw_dot))
        self.assertEqual(state_dot.dynamic_car_state.rear_axle_velocity_2d, self.ego_state.dynamic_car_state.rear_axle_acceleration_2d)
        self.assertEqual(state_dot.dynamic_car_state.rear_axle_acceleration_2d, StateVector2D(0, 0))
        self.assertEqual(state_dot.tire_steering_angle, self.ego_state.dynamic_car_state.tire_steering_rate)

    def test_propagate_state(self) -> None:
        """
        Test propagate_state
        """
        state = self.motion_model.propagate_state(self.ego_state, self.ego_state.dynamic_car_state, self.sampling_time)
        self.assertEqual(state.rear_axle, StateSE2(forward_integrate(self.ego_state.rear_axle.x, self.x_dot, self.sampling_time), forward_integrate(self.ego_state.rear_axle.y, self.y_dot, self.sampling_time), forward_integrate(self.ego_state.rear_axle.heading, self.yaw_dot, self.sampling_time)))
        self.assertEqual(state.dynamic_car_state.rear_axle_velocity_2d, StateVector2D(forward_integrate(self.ego_state.dynamic_car_state.rear_axle_velocity_2d.x, self.ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, self.sampling_time), 0.0))
        self.assertEqual(state.dynamic_car_state.rear_axle_acceleration_2d, StateVector2D(0.1, 0.0))
        self.assertEqual(state.tire_steering_angle, forward_integrate(self.ego_state.tire_steering_angle, self.ego_state.dynamic_car_state.tire_steering_rate, self.sampling_time))
        self.assertEqual(state.dynamic_car_state.angular_velocity, state.dynamic_car_state.rear_axle_velocity_2d.x * np.tan(state.tire_steering_angle) / self.vehicle.wheel_base)

    def test_limit_steering_angle(self) -> None:
        """
        Test whether the KinematicBicycleModel correct enforces steering angle
        limits.
        """
        dynamic_car_state = DynamicCarState.build_from_rear_axle(self.vehicle.rear_axle_to_center, rear_axle_velocity_2d=StateVector2D(0.0, 0.0), rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_rate=10.0)
        car_footprint = CarFootprint.build_from_rear_axle(rear_axle_pose=StateSE2(x=0.0, y=0.0, heading=0.0), vehicle_parameters=self.vehicle)
        ego_state = EgoState(car_footprint, dynamic_car_state, tire_steering_angle=self.motion_model._max_steering_angle - 0.0001, is_in_auto_mode=True, time_point=TimePoint(0))
        propagated_state = self.motion_model.propagate_state(ego_state, dynamic_car_state, self.sampling_time)
        self.assertEqual(propagated_state.tire_steering_angle, self.motion_model._max_steering_angle)

    def test_update_command(self) -> None:
        """
        Test whether the update_command function performs as expected:
        1) returns same commands if time constants are set to zero (no delay)
        2) returns an smaller command (in the absolute sense) when filter is applied
        """
        dynamic_car_state = DynamicCarState.build_from_rear_axle(self.vehicle.rear_axle_to_center, rear_axle_velocity_2d=StateVector2D(0.0, 0.0), rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_rate=0.0)
        car_footprint = CarFootprint.build_from_rear_axle(rear_axle_pose=StateSE2(x=0.0, y=0.0, heading=0.0), vehicle_parameters=self.vehicle)
        state = EgoState(car_footprint, dynamic_car_state, tire_steering_angle=self.motion_model._max_steering_angle - 0.0001, is_in_auto_mode=True, time_point=TimePoint(0))
        ideal_dynamic_state = DynamicCarState.build_from_rear_axle(self.vehicle.rear_axle_to_center, rear_axle_velocity_2d=StateVector2D(0.0, 0.0), rear_axle_acceleration_2d=StateVector2D(1.0, 0.0), tire_steering_rate=0.5)
        no_delay_motion_model = KinematicBicycleModel(self.vehicle, accel_time_constant=0, steering_angle_time_constant=0)
        no_delay_propagating_state = no_delay_motion_model._update_commands(state, ideal_dynamic_state, self.sampling_time)
        self.assertEqual(round(no_delay_propagating_state.dynamic_car_state.rear_axle_acceleration_2d.x, 10), ideal_dynamic_state.rear_axle_acceleration_2d.x)
        self.assertEqual(round(no_delay_propagating_state.dynamic_car_state.tire_steering_rate, 10), ideal_dynamic_state.tire_steering_rate)
        propagating_state = self.motion_model._update_commands(state, ideal_dynamic_state, self.sampling_time)
        self.assertTrue(propagating_state.dynamic_car_state.rear_axle_acceleration_2d.x < ideal_dynamic_state.rear_axle_acceleration_2d.x)
        self.assertLess(propagating_state.dynamic_car_state.tire_steering_rate, ideal_dynamic_state.tire_steering_rate)

class RemotePlanner(AbstractPlanner):
    """
    Remote planner delegates computation of trajectories to a docker container, with which communicates through
    grpc.
    """

    def __init__(self, submission_container_manager: Optional[SubmissionContainerManager]=None, submission_image: Optional[str]=None, container_name: Optional[str]=None, compute_trajectory_timeout: float=1) -> None:
        """
        Prepares the remote container for planning.
        :param submission_container_manager: Optional manager, if provided a container will be started by RemotePlanner
        :param submission_image: Docker image name for the submission_container_factory
        :param container_name: Name to assign to the submission container
        :param compute_trajectory_timeout: Timeout for computation of trajectory.
        """
        if submission_container_manager:
            missing_parameter_message = 'Parameters for SubmissionContainer are missing!'
            assert submission_image, missing_parameter_message
            assert container_name, missing_parameter_message
            self.port = None
        else:
            self.port = os.getenv('SUBMISSION_CONTAINER_PORT', 50051)
        self.submission_container_manager = submission_container_manager
        self.submission_image = submission_image
        self.container_name = container_name
        self._channel = None
        self._stub = None
        self.serialized_observation: Optional[List[bytes]] = None
        self.serialized_state: Optional[List[bytes]] = None
        self.sample_interval: Optional[float] = None
        self._compute_trajectory_timeout = compute_trajectory_timeout

    def __reduce__(self) -> Tuple[Type[RemotePlanner], Tuple[Optional[SubmissionContainerManager], Optional[str], Optional[str]]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class
        """
        return (self.__class__, (self.submission_container_manager, self.submission_image, self.container_name))

    def name(self) -> str:
        """Inherited, see superclass."""
        return 'RemotePlanner'

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    @staticmethod
    def _planner_initializations_to_message(initialization: PlannerInitialization) -> chpb.PlannerInitializationLight:
        """
        Converts a PlannerInitialization to the message specified in the protocol files.
        :param initialization: The initialization parameters for the planner
        :return: A initialization message
        """
        try:
            mission_goal = proto_se2_from_se2(initialization.mission_goal)
        except AttributeError as e:
            logger.error('Mission goal was None!')
            raise e
        planner_initialization = chpb.PlannerInitializationLight(route_roadblock_ids=initialization.route_roadblock_ids, mission_goal=mission_goal, map_name=initialization.map_api.map_name)
        return planner_initialization

    def initialize(self, initialization: PlannerInitialization, timeout: float=5) -> None:
        """
        Creates the container manager, and runs the specified docker image. The communication port is created using
        the PID from the ray worker. Sends a request to initialize the remote planner.
        :param initialization: List of PlannerInitialization objects
        :param timeout: for planner initialization
        """
        if self.submission_container_manager:
            submission_container = try_n_times(self.submission_container_manager.get_submission_container, [self.submission_image, self.container_name, find_free_port_number()], {}, (docker.errors.APIError,), max_tries=10)
            self.port = submission_container.port
            submission_container.start()
            submission_container.wait_until_running(timeout=5)
        self._channel = grpc.insecure_channel(f'{NETWORK}:{self.port}')
        self._stub = chpb_grpc.DetectionTracksChallengeStub(self._channel)
        logger.info('Client sending planner initialization request...')
        planner_initializations_message = self._planner_initializations_to_message(initialization)
        logger.info(f'Trying to communicate on port {NETWORK}:{self.port}')
        try:
            _, _ = keep_trying(self._stub.InitializePlanner, [planner_initializations_message], {}, errors=(grpc.RpcError,), timeout=timeout)
        except Exception as e:
            submission_logger.error('Planner initialization failed!')
            submission_logger.error(e)
            raise e
        logger.info('Planner initialized!')

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Planner input for which trajectory should be computed
        :return: Trajectory representing the predicted ego's position in future for every input iteration
        """
        logger.debug('Client sending planner input: %s' % current_input)
        trajectory = self._compute_trajectory(self._stub, current_input=current_input)
        return trajectory

    def _compute_trajectory(self, stub: chpb_grpc.DetectionTracksChallengeStub, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Sends a request to compute the trajectory given the PlannerInput to the remote planner.
        :param stub: Service interface
        :param current_input: Planner input for which a trajectory should be computed.
        :return: Trajectory representing the predicted ego's position in future for every input iteration
        """
        logging.debug('Client sending observation...')
        self.serialized_state, self.serialized_observation, self.sample_interval = self._get_history_update(current_input)
        serialized_simulation_iteration = chpb.SimulationIteration(time_us=current_input.iteration.time_us, index=current_input.iteration.index)
        if self.sample_interval:
            serialized_buffer = chpb.SimulationHistoryBuffer(ego_states=self.serialized_state, observations=self.serialized_observation, sample_interval=self.sample_interval)
        else:
            serialized_buffer = chpb.SimulationHistoryBuffer(ego_states=self.serialized_state, observations=self.serialized_observation, sample_interval=None)
        tl_data = self._build_tl_message_from_planner_input(current_input)
        planner_input = chpb.PlannerInput(simulation_iteration=serialized_simulation_iteration, simulation_history_buffer=serialized_buffer, traffic_light_data=tl_data)
        try:
            trajectory_message = stub.ComputeTrajectory(planner_input, timeout=self._compute_trajectory_timeout)
        except grpc.RpcError as e:
            submission_logger.error('Trajectory computation service failed!')
            submission_logger.error(e)
            raise e
        return interp_traj_from_proto_traj(trajectory_message)

    def _get_history_update(self, planner_input: PlannerInput) -> Tuple[List[bytes], List[bytes], Optional[float]]:
        """
        Gets the new states and observations from the input. If no cache is present, the entire history is
        serialized, otherwise just the last element.
        :param planner_input: The input for planners
        :return: Tuple with new serialized state and observations.
        """
        keep_all_history = not self.serialized_state and (not self.serialized_observation)
        if keep_all_history:
            serialized_state = [pickle.dumps(state) for state in planner_input.history.ego_states]
            serialized_observation = [pickle.dumps(obs) for obs in planner_input.history.observations]
        else:
            last_ego_state, last_observations = planner_input.history.current_state
            serialized_state = [pickle.dumps(last_ego_state)]
            serialized_observation = [pickle.dumps(last_observations)]
        sample_interval = planner_input.history.sample_interval if not self.sample_interval else None
        return (serialized_state, serialized_observation, sample_interval)

    @staticmethod
    def _build_tl_message_from_planner_input(planner_input: PlannerInput) -> chpb.TrafficLightStatusData:
        tl_status_data: List[List[chpb.TrafficLightStatusData]]
        if planner_input.traffic_light_data is None:
            tl_status_data = [[]]
        else:
            tl_status_data = [proto_tl_status_data_from_tl_status_data(tl_status_data) for tl_status_data in planner_input.traffic_light_data]
        return tl_status_data

def proto_se2_from_se2(se2: StateSE2) -> chpb.StateSE2:
    """
    Serializes StateSE2 to a StateSE2 message
    :param se2: The StateSE2 object
    :return: The corresponding StateSE2 message
    """
    return chpb.StateSE2(x=se2.x, y=se2.y, heading=se2.heading)

def interp_traj_from_proto_traj(trajectory: chpb.Trajectory) -> InterpolatedTrajectory:
    """
    Deserializes Trajectory message to a InterpolatedTrajectory object
    :param trajectory: The proto Trajectory message
    :return: The corresponding InterpolatedTrajectory object
    """
    return InterpolatedTrajectory([ego_state_from_proto_ego_state(state) for state in trajectory.ego_states])

class TestAbstractIDMPlanner(unittest.TestCase):
    """Test the AbstractIDMPlanner interface"""
    TEST_FILE_PATH = 'nuplan.planning.simulation.planner.abstract_idm_planner'

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self.scenario = get_test_nuplan_scenario()
        self.planned_trajectory_samples = 10
        self.planner = MockIDMPlanner(target_velocity=10, min_gap_to_lead_agent=0.5, headway_time=1.5, accel_max=1.0, decel_max=2.0, planned_trajectory_samples=self.planned_trajectory_samples, planned_trajectory_sample_interval=0.2, occupancy_map_radius=20)

    def test_name(self) -> None:
        """Test name"""
        self.assertEqual(self.planner.name(), 'MockIDMPlanner')

    def test_observation_type(self) -> None:
        """Test observation_type"""
        self.assertEqual(self.planner.observation_type(), DetectionsTracks)

    def test__initialize_route_plan_assertion_error(self) -> None:
        """Test raise if _map_api is uninitialized"""
        with self.assertRaises(AssertionError):
            self.planner._initialize_route_plan([])

    def test__initialize_route_plan(self) -> None:
        """Test _map_api is uninitialized."""
        with patch.object(self.planner, '_map_api') as _map_api:
            _map_api.get_map_object = Mock()
            _map_api.get_map_object.side_effect = [MagicMock(), None, MagicMock()]
            mock_route_roadblock_ids = ['a']
            self.planner._initialize_route_plan(mock_route_roadblock_ids)
            _map_api.get_map_object.assert_called_with('a', SemanticMapLayer.ROADBLOCK)
            mock_route_roadblock_ids = ['b']
            self.planner._initialize_route_plan(mock_route_roadblock_ids)
            _map_api.get_map_object.assert_called_with('b', SemanticMapLayer.ROADBLOCK_CONNECTOR)

    def test__construct_occupancy_map_value_error(self) -> None:
        """Test raise if observation type is incorrect"""
        with self.assertRaises(ValueError):
            self.planner._construct_occupancy_map(Mock(), Mock())

    @patch(f'{TEST_FILE_PATH}.STRTreeOccupancyMapFactory.get_from_boxes')
    def test__construct_occupancy_map(self, mock_get_from_boxes: Mock) -> None:
        """Test raise if observation type is incorrect"""
        mock_observations = self.scenario.initial_tracked_objects
        mock_ego_state = self.scenario.initial_ego_state
        self.planner._construct_occupancy_map(mock_ego_state, mock_observations)
        mock_get_from_boxes.assert_called_once()

    def test__propagate(self) -> None:
        """Test _propagate()"""
        with patch.object(self.planner, '_policy') as _policy:
            init_progress = 1
            init_velocity = 2
            tspan = 0.5
            mock_ego_idm_state = IDMAgentState(init_progress, init_velocity)
            mock_lead_agent = Mock()
            _policy.solve_forward_euler_idm_policy = Mock(return_value=IDMAgentState(3, 4))
            self.planner._propagate(mock_ego_idm_state, mock_lead_agent, tspan)
            _policy.solve_forward_euler_idm_policy.assert_called_once_with(IDMAgentState(0, init_velocity), mock_lead_agent, tspan)
            self.assertEqual(init_progress + _policy.solve_forward_euler_idm_policy().progress, mock_ego_idm_state.progress)
            self.assertEqual(_policy.solve_forward_euler_idm_policy().velocity, mock_ego_idm_state.velocity)

    def test__get_planned_trajectory_error(self) -> None:
        """Test raise if _ego_path_linestring has not been initialized"""
        with self.assertRaises(AssertionError):
            self.planner._get_planned_trajectory(Mock(), Mock(), Mock())

    @patch(f'{TEST_FILE_PATH}.InterpolatedTrajectory')
    @patch(f'{TEST_FILE_PATH}.AbstractIDMPlanner._propagate')
    @patch(f'{TEST_FILE_PATH}.AbstractIDMPlanner._get_leading_object')
    @patch(f'{TEST_FILE_PATH}.AbstractIDMPlanner._idm_state_to_ego_state')
    def test__get_planned_trajectory(self, mock_idm_state_to_ego_state: Mock, mock_get_leading_object: Mock, mock_propagate: Mock, mock_trajectory: Mock) -> None:
        """Test _get_planned_trajectory"""
        with patch.object(self.planner, '_ego_path_linestring') as _ego_path_linestring:
            _ego_path_linestring.project = call()
            mock_idm_state_to_ego_state.return_value = Mock()
            mock_get_leading_object.return_value = Mock()
            self.planner._get_planned_trajectory(MagicMock(), MagicMock(), MagicMock())
            _ego_path_linestring.project.assert_called_once()
            mock_idm_state_to_ego_state.assert_called()
            mock_get_leading_object.assert_called()
            mock_propagate.assert_called()
            mock_trajectory.assert_called_once()

    def test__idm_state_to_ego_state_error(self) -> None:
        """Test raise if _ego_path has not been initialized"""
        with self.assertRaises(AssertionError):
            self.planner._idm_state_to_ego_state(Mock(), Mock(), Mock())

    @patch(f'{TEST_FILE_PATH}.EgoState.build_from_center')
    @patch(f'{TEST_FILE_PATH}.max')
    @patch(f'{TEST_FILE_PATH}.min')
    def test__idm_state_to_ego_state(self, mock_max: Mock, mock_min: Mock, mock_build_from_center: Mock) -> None:
        """Test _idm_state_to_ego_state"""
        with patch.object(self.planner, '_ego_path') as _ego_path:
            mock_new_center = MagicMock(autospec=True)
            mock_ego_idm_state = IDMAgentState(0, 1)
            mock_time_point = Mock()
            mock_vehicle_params = Mock()
            _ego_path.get_state_at_progress = Mock(return_value=mock_new_center)
            self.planner._idm_state_to_ego_state(mock_ego_idm_state, mock_time_point, mock_vehicle_params)
            mock_max.assert_called_once()
            mock_min.assert_called_once()
            mock_build_from_center.assert_called_with(center=StateSE2(mock_new_center.x, mock_new_center.y, mock_new_center.heading), center_velocity_2d=StateVector2D(mock_ego_idm_state.velocity, 0), center_acceleration_2d=StateVector2D(0, 0), tire_steering_angle=0.0, time_point=mock_time_point, vehicle_parameters=mock_vehicle_params)

    def test__annotate_occupancy_map_error(self) -> None:
        """Test raise if _map_api or _candidate_lane_edge_ids has not been initialized"""
        with self.assertRaises(AssertionError):
            with patch.object(self.planner, '_map_api'):
                self.planner._annotate_occupancy_map(Mock(), Mock())
        with self.assertRaises(AssertionError):
            with patch.object(self.planner, '_candidate_lane_edge_ids'):
                self.planner._annotate_occupancy_map(Mock(), Mock())

    @patch(f'{TEST_FILE_PATH}.trim_path')
    @patch(f'{TEST_FILE_PATH}.unary_union')
    @patch(f'{TEST_FILE_PATH}.path_to_linestring')
    def test__get_expanded_ego_path(self, mock_path_to_linestring: MagicMock, mock_unary_union: Mock, mock_trim_path: Mock) -> None:
        """Test _get_expanded_ego_path"""
        mock_ego_idm_state = IDMAgentState(0, 1)
        mock_ego_state = MagicMock(spec_set=EgoState)
        mock_trim_path.return_value = Mock()
        with patch.object(self.planner, '_ego_path') as _ego_path:
            _ego_path.get_start_progress = Mock(return_value=0)
            _ego_path.get_end_progress = Mock(return_value=10)
            self.planner._get_expanded_ego_path(mock_ego_state, mock_ego_idm_state)
            mock_trim_path.assert_called_once()
            mock_path_to_linestring.assert_called_once_with(mock_trim_path.return_value)
            mock_unary_union.assert_called_once()

    @patch(f'{TEST_FILE_PATH}.transform')
    @patch(f'{TEST_FILE_PATH}.principal_value')
    def test__get_leading_idm_agent(self, mock_principal_value: Mock, mock_transform: Mock) -> None:
        """Test _get_leading_idm_agent when an Agent object is passed"""
        mock_agent = MagicMock(spec_set=Agent)
        mock_transform.return_value = StateSE2(1, 0, 0)
        mock_relative_distance = 2
        result = self.planner._get_leading_idm_agent(MagicMock(spec_set=EgoState), mock_agent, mock_relative_distance)
        self.assertEqual(mock_relative_distance, result.progress)
        self.assertEqual(mock_transform.return_value.x, result.velocity)
        self.assertEqual(0.0, result.length_rear)
        mock_principal_value.assert_called_once()
        mock_transform.assert_called_once()

    def test__get_leading_idm_agent_static(self) -> None:
        """Test _get_leading_idm_agent when a Staic object is passed"""
        mock_relative_distance = 2
        result = self.planner._get_leading_idm_agent(Mock(spec_set=EgoState), Mock(), mock_relative_distance)
        self.assertEqual(mock_relative_distance, result.progress)
        self.assertEqual(0.0, result.velocity)
        self.assertEqual(0.0, result.length_rear)

    def test__get_free_road_leading_idm_state(self) -> None:
        """Test _get_free_road_leading_idm_state"""
        mock_ego_idm_state = IDMAgentState(0, 1)
        mock_ego_state = self.scenario.initial_ego_state
        with patch.object(self.planner, '_ego_path', spec_set=AbstractPath) as _ego_path:
            _ego_path.get_start_progress = Mock(return_value=0)
            _ego_path.get_end_progress = Mock(return_value=10)
            result = self.planner._get_free_road_leading_idm_state(mock_ego_state, mock_ego_idm_state)
            self.assertEqual(_ego_path.get_end_progress() - mock_ego_idm_state.progress, result.progress)
            self.assertEqual(0.0, result.velocity)
            self.assertEqual(mock_ego_state.car_footprint.length / 2, result.length_rear)

    def test__get_red_light_leading_idm_state(self) -> None:
        """Test _get_red_light_leading_idm_state"""
        mock_relative_distance = 2
        result = self.planner._get_red_light_leading_idm_state(mock_relative_distance)
        self.assertEqual(mock_relative_distance, result.progress)
        self.assertEqual(0.0, result.velocity)
        self.assertEqual(0.0, result.length_rear)

    def test__get_leading_object(self) -> None:
        """Test _get_leading_object"""
        mock_occupancy_map = MagicMock(spec_set=OccupancyMap)
        mock_intersecting_agents = MagicMock(spec_set=OccupancyMap)
        mock_intersecting_agents.size = 1
        mock_intersecting_agents.get_nearest_entry_to = Mock(return_value=('red_light', Mock(), 0.0))
        mock_occupancy_map.intersects = Mock(return_value=mock_intersecting_agents)
        with patch.object(self.planner, '_get_red_light_leading_idm_state') as mock_handle_traffic_light:
            with patch.object(self.planner, '_get_expanded_ego_path') as mock_get_expanded_ego_path:
                self.planner._get_leading_object(Mock(), MagicMock(), mock_occupancy_map, Mock())
                mock_handle_traffic_light.assert_called_once_with(0.0)
                mock_get_expanded_ego_path.assert_called_once()
        mock_intersecting_agents.get_nearest_entry_to = Mock(return_value=('', Mock(), 0.0))
        with patch.object(self.planner, '_get_leading_idm_agent') as mock_handle_tracks:
            with patch.object(self.planner, '_get_expanded_ego_path') as mock_get_expanded_ego_path:
                self.planner._get_leading_object(Mock(), MagicMock(), mock_occupancy_map, MagicMock())
                mock_handle_tracks.assert_called_once()
                mock_get_expanded_ego_path.assert_called_once()

    def test__get_leading_object_free_road(self) -> None:
        """Test _get_leading_object in the case where there are no leading agents"""
        mock_occupancy_map = MagicMock(spec_set=OccupancyMap)
        mock_intersecting_agents = MagicMock(spec_set=OccupancyMap)
        mock_intersecting_agents.size = 0
        mock_occupancy_map.intersects = Mock(return_value=mock_intersecting_agents)
        with patch.object(self.planner, '_get_free_road_leading_idm_state') as mock_handle_free_road_case:
            with patch.object(self.planner, '_get_expanded_ego_path') as mock_get_expanded_ego_path:
                self.planner._get_leading_object(Mock(), MagicMock(), mock_occupancy_map, Mock())
                mock_handle_free_road_case.assert_called_once()
                mock_get_expanded_ego_path.assert_called_once()

@lru_cache(maxsize=1)
def get_test_nuplan_scenario(use_multi_sample: bool=False, lidar_pc_index: Optional[int]=None, sensor_root: Optional[str]=None) -> NuPlanScenario:
    """
    Retrieve a sample scenario from the db.
    :param use_multi_sample: Whether to extract multiple temporal samples in the scenario.
    :param lidar_pc_index: The initial lidarpc_token for the sceanrio. If None, then a default example (corresponding to DEFAULT_LIDARPC_INDEX) will be used.
    :param sensor_root: The directory for which the sensor data should be saved to.
    :return: A sample db scenario.
    """
    load_path = NUPLAN_DB_FILES[4]
    lidar_pc_index = DEFAULT_LIDARPC_INDEX if lidar_pc_index is None else lidar_pc_index
    token = get_sensor_token_by_index_from_db(load_path, get_lidarpc_sensor_data(), lidar_pc_index)
    timestamp = get_sensor_data_token_timestamp_from_db(load_path, get_lidarpc_sensor_data(), token)
    map_name = get_sensor_token_map_name_from_db(load_path, get_lidarpc_sensor_data(), token)
    if timestamp is None or map_name is None:
        raise ValueError(f'Token {token} not found in log.')
    scenario = NuPlanScenario(data_root=NUPLAN_DATA_ROOT, log_file_load_path=load_path, initial_lidar_token=token, initial_lidar_timestamp=timestamp, scenario_type=DEFAULT_SCENARIO_NAME, map_root=NUPLAN_MAPS_ROOT, map_version=NUPLAN_MAP_VERSION, map_name=map_name, scenario_extraction_info=ScenarioExtractionInfo() if use_multi_sample else None, ego_vehicle_parameters=get_pacifica_parameters(), sensor_root=NUPLAN_SENSOR_ROOT if sensor_root is None else sensor_root)
    return scenario

class TestAbstractIDMPlanner(unittest.TestCase):
    """Test the AbstractIDMPlanner interface"""
    TEST_FILE_PATH = 'nuplan.planning.simulation.planner.idm_planner'

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self.scenario = get_test_nuplan_scenario()
        self.planned_trajectory_samples = 10
        self.planner = IDMPlanner(target_velocity=10, min_gap_to_lead_agent=0.5, headway_time=1.5, accel_max=1.0, decel_max=2.0, planned_trajectory_samples=self.planned_trajectory_samples, planned_trajectory_sample_interval=0.2, occupancy_map_radius=20)

    def test_name(self) -> None:
        """Test name"""
        self.assertEqual(self.planner.name(), 'IDMPlanner')

    def test_observation_type(self) -> None:
        """Test observation_type"""
        self.assertEqual(self.planner.observation_type(), DetectionsTracks)

    def test__initialize_route_plan_assertion_error(self) -> None:
        """Test raise if _map_api is uninitialized"""
        with self.assertRaises(AssertionError):
            self.planner._initialize_route_plan([])

    @patch(f'{TEST_FILE_PATH}.IDMPlanner._initialize_route_plan')
    def test_initialize(self, mock_initialize_route_plan: Mock) -> None:
        """Test initialize"""
        initialization = MagicMock()
        self.planner.initialize(initialization)
        mock_initialize_route_plan.assert_called_once_with(initialization.route_roadblock_ids)

    @patch(f'{TEST_FILE_PATH}.path_to_linestring')
    @patch(f'{TEST_FILE_PATH}.create_path_from_se2')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._breadth_first_search')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._get_starting_edge')
    def test__initialize_ego_path(self, mock_get_starting_edge: Mock, mock_breadth_first_search: Mock, mock_create_path_from_se2: Mock, mock_path_to_linestring: Mock) -> None:
        """Test _initialize_ego_path()"""
        mock_starting_edge = Mock()
        mock_lane = MagicMock()
        mock_lane.speed_limit_mps = 0
        ego_state = self.scenario.initial_ego_state
        mock_breadth_first_search.return_value = ([mock_lane], True)
        mock_get_starting_edge.return_value = mock_starting_edge
        with patch.object(self.planner, '_route_roadblocks'):
            self.planner._initialize_ego_path(ego_state)
            mock_breadth_first_search.assert_called_once_with(ego_state)
            mock_create_path_from_se2.assert_called_once_with([])
            mock_path_to_linestring.assert_called_once_with([])

    def test__get_starting_edge(self) -> None:
        """Test _get_starting_edge()"""
        mock_edge = MagicMock(spec_set=LaneGraphEdgeMapObject)
        mock_edge.contains_point.side_effect = [False, True]
        mock_edge.polygon.distance.side_effect = [0, 0]
        mock_roadblock = MagicMock(spec_set=RoadBlockGraphEdgeMapObject)
        mock_roadblock.interior_edges = [mock_edge]
        self.planner._route_roadblocks = [mock_roadblock, mock_roadblock]
        result = self.planner._get_starting_edge(Mock(spec=EgoState))
        mock_edge.contains_point.assert_called()
        mock_edge.polygon.distance.assert_called()
        self.assertEqual(result, mock_edge)

    @patch(f'{TEST_FILE_PATH}.IDMPlanner._initialize_ego_path')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._construct_occupancy_map')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._annotate_occupancy_map')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._get_planned_trajectory')
    def test_compute_trajectory(self, mock_get_planned_trajectory: Mock, mock_annotate_occupancy_map: Mock, mock_construct_occupancy_map: Mock, mock_initialize_ego_path: Mock) -> None:
        """Test compute_trajectory"""
        planner_input = MagicMock()
        mock_ego_state = Mock()
        mock_traffic_light_data = call()
        planner_input.history.current_state = (mock_ego_state, Mock())
        planner_input.traffic_light_data = mock_traffic_light_data
        mock_occupancy_map = Mock()
        mock_unique_observations = Mock()
        mock_construct_occupancy_map.return_value = (mock_occupancy_map, mock_unique_observations)
        self.planner.compute_trajectory(planner_input)
        mock_initialize_ego_path.assert_called_once_with(mock_ego_state)
        mock_construct_occupancy_map.assert_called_once_with(*planner_input.history.current_state)
        mock_annotate_occupancy_map.assert_called_once_with(mock_traffic_light_data, mock_occupancy_map)
        mock_get_planned_trajectory.assert_called_once_with(mock_ego_state, mock_occupancy_map, mock_unique_observations)

    def test_compute_trajectory_integration(self) -> None:
        """Test the IDMPlanner in full using mock data"""
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(10, self.scenario, DetectionsTracks)
        self.planner.initialize(PlannerInitialization(self.scenario.get_route_roadblock_ids(), self.scenario.get_mission_goal(), self.scenario.map_api))
        trajectories = self.planner.compute_trajectory(PlannerInput(SimulationIteration(self.scenario.get_time_point(0), 0), history_buffer, list(self.scenario.get_traffic_light_status_at_iteration(0))))
        self.assertEqual(self.planned_trajectory_samples + 1, len(trajectories.get_sampled_trajectory()))

class TestMlPlanner(unittest.TestCase):
    """
    Test MLPlanner with two models.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.scenario = get_test_nuplan_scenario()

    def test_simple_vector_net_model(self) -> None:
        """Test Model Vector Map Simple"""
        self.run_test_ml_planner(construct_simple_vector_map_ml_planner())

    def test_raster_net_model(self) -> None:
        """Test Raster Net model"""
        self.run_test_ml_planner(construct_raster_ml_planner())

    def test_urban_driver_open_loop_model(self) -> None:
        """Test UrbanDriverOpenLoop model"""
        self.run_test_ml_planner(construct_urban_driver_open_loop_ml_planner())

    def run_test_ml_planner(self, planner: MLPlanner) -> None:
        """Tests if progress is calculated correctly"""
        scenario = self.scenario
        simulation_history_buffer_duration = 2
        buffer_size = int(simulation_history_buffer_duration / self.scenario.database_interval + 1)
        history = SimulationHistoryBuffer.initialize_from_scenario(buffer_size=buffer_size, scenario=self.scenario, observation_type=DetectionsTracks)
        initialization = PlannerInitialization(route_roadblock_ids=scenario.get_route_roadblock_ids(), mission_goal=scenario.get_mission_goal(), map_api=scenario.map_api)
        planner.initialize(initialization)
        trajectory = planner.compute_trajectory(PlannerInput(iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history, traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0))))
        self.assertNotEqual(trajectory, None)
        self.assertEqual(len(trajectory.get_sampled_trajectory()), planner._num_output_dim + 1)

class TestProfileIDM(unittest.TestCase):
    """
    Profiling test for IDM agents.
    """

    def setUp(self) -> None:
        """
        Inherited, see super class.
        """
        self.n_repeat_trials = 1
        self.display_results = True
        self.scenario = get_test_nuplan_scenario()

    def test_profile_idm_agent_observation(self) -> None:
        """Profile IDMAgents."""
        profiler = Profiler(interval=0.0001)
        profiler.start()
        for _ in range(self.n_repeat_trials):
            observation = IDMAgents(target_velocity=10, min_gap_to_lead_agent=0.5, headway_time=1.5, accel_max=1.0, decel_max=2.0, scenario=self.scenario, open_loop_detections_types=[])
            for step in range(self.scenario.get_number_of_iterations() - 1):
                iteration = SimulationIteration(time_point=self.scenario.get_time_point(step), index=step)
                next_iteration = SimulationIteration(time_point=self.scenario.get_time_point(step + 1), index=step + 1)
                buffer = SimulationHistoryBuffer.initialize_from_list(1, [self.scenario.get_ego_state_at_iteration(step)], [self.scenario.get_tracked_objects_at_iteration(step)], next_iteration.time_point.time_s - iteration.time_point.time_s)
                observation.update_observation(iteration, next_iteration, buffer)
        profiler.stop()
        if self.display_results:
            logger.info(profiler.output_text(unicode=True, color=True))

def convert_sample_to_scene(map_name: str, database_interval: float, traffic_light_status: Generator[TrafficLightStatusData, None, None], mission_goal: Optional[StateSE2], expert_trajectory: List[EgoState], data: SimulationHistorySample, colors: TrajectoryColors=TrajectoryColors()) -> Dict[str, Any]:
    """
    Serialize history and scenario.
    :param map_name: name of the map used for this scenario.
    :param database_interval: Database interval (fps).
    :param traffic_light_status: Traffic light status.
    :param mission_goal: if mission goal is present, this is goal of this mission.
    :param expert_trajectory: trajectory of an expert driver.
    :param data: single sample from history.
    :param colors: colors for trajectories.
    :return: serialized dictionary.
    """
    scene: Dict[str, Any] = {'timestamp_us': data.ego_state.time_us}
    trajectories: Dict[str, Dict[str, Any]] = {}
    if mission_goal is not None:
        scene['goal'] = dict(to_scene_goal_from_state(mission_goal))
    else:
        scene['goal'] = None
    scene['ego'] = dict(to_scene_ego_from_car_footprint(CarFootprint.build_from_center(data.ego_state.center, get_pacifica_parameters())))
    scene['ego']['timestamp_us'] = data.ego_state.time_us
    map_name_without_suffix = str(pathlib.Path(map_name).with_suffix(''))
    scene['map'] = {'area': map_name_without_suffix}
    scene['map_name'] = map_name
    if isinstance(data.observation, DetectionsTracks):
        scene['world'] = to_scene_boxes(data.observation.tracked_objects)
        scene['prediction'] = to_scene_agent_prediction_from_boxes(data.observation.tracked_objects, colors.agents_predicted_trajectory)
    trajectories['ego_predicted_trajectory'] = dict(to_scene_trajectory_from_list_ego_state(data.trajectory.get_sampled_trajectory(), colors.ego_predicted_trajectory))
    trajectories['ego_expert_trajectory'] = dict(to_scene_trajectory_from_list_ego_state(expert_trajectory, colors.ego_expert_trajectory))
    scene['trajectories'] = trajectories
    scene['traffic_light_status'] = [traffic_light.serialize() for traffic_light in traffic_light_status]
    scene['database_interval'] = database_interval
    return scene

class SerializationCallback(AbstractCallback):
    """Callback for serializing scenes at the end of the simulation."""

    def __init__(self, output_directory: Union[str, pathlib.Path], folder_name: Union[str, pathlib.Path], serialization_type: str, serialize_into_single_file: bool):
        """
        Construct serialization callback
        :param output_directory: where scenes should be serialized
        :param folder_name: folder where output should be serialized
        :param serialization_type: A way to serialize output, options: ["json", "pickle", "msgpack"]
        :param serialize_into_single_file: if true all data will be in single file, if false, each time step will
                be serialized into a separate file
        """
        available_formats = ['json', 'pickle', 'msgpack']
        if serialization_type not in available_formats:
            raise ValueError(f'The serialization callback will not store files anywhere!Choose at least one format from {available_formats} instead of {serialization_type}!')
        self._output_directory = pathlib.Path(output_directory) / folder_name
        self._serialization_type = serialization_type
        self._serialize_into_single_file = serialize_into_single_file

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Create directory at initialization
        :param setup: simulation setup
        :param planner: planner before initialization
        """
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
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
        On reached_end validate that all steps were correctly serialized
        :param setup: simulation setup
        :param planner: planner when simulation ends
        :param history: resulting from simulation
        """
        number_of_scenes = len(history)
        if number_of_scenes == 0:
            raise RuntimeError('Number of scenes has to be greater than 0')
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        scenario = setup.scenario
        expert_trajectory = list(scenario.get_expert_ego_trajectory())
        scenes = [convert_sample_to_scene(map_name=scenario.map_api.map_name, database_interval=scenario.database_interval, traffic_light_status=scenario.get_traffic_light_status_at_iteration(index), expert_trajectory=expert_trajectory, mission_goal=scenario.get_mission_goal(), data=sample, colors=TrajectoryColors()) for index, sample in enumerate(history.data)]
        self._serialize_scenes(scenes, scenario_directory)

    def _serialize_scenes(self, scenes: List[Dict[str, Any]], scenario_directory: pathlib.Path) -> None:
        """
        Serialize scenes based on callback setup to json/pickle or other
        :param scenes: scenes to be serialized
        :param scenario_directory: directory where they should be serialized
        """
        if not self._serialize_into_single_file:
            for scene in scenes:
                file_name = scenario_directory / str(scene['ego']['timestamp_us'])
                _dump_to_file(file_name, scene, self._serialization_type)
        else:
            file_name = scenario_directory / scenario_directory.name
            _dump_to_file(file_name, scenes, self._serialization_type)

    def _get_scenario_folder(self, planner_name: str, scenario: AbstractScenario) -> pathlib.Path:
        """
        Compute scenario folder directory where all files will be stored
        :param planner_name: planner name
        :param scenario: for which to compute directory name
        :return directory path
        """
        return self._output_directory / planner_name / scenario.scenario_type / scenario.log_name / scenario.scenario_name

class ScenarioSceneConverter(SceneConverter):
    """
    Scene writer that converts a scenario sample to scene.
    """

    def __init__(self, ego_trajectory_horizon: float, ego_trajectory_poses: int) -> None:
        """
        Initialize scene writer.
        :param ego_trajectory_horizon: the horizon to get ego's future trajectory.
        :param ego_trajectory_poses: number of poses for ego's future trajectory.
        """
        self._ego_trajectory_horizon = ego_trajectory_horizon
        self._ego_trajectory_poses = ego_trajectory_poses

    def __call__(self, scenario: AbstractScenario, features: FeaturesType, targets: TargetsType, predictions: FeaturesType) -> List[Dict[str, Any]]:
        """Inherited, see superclass."""
        index = 0
        ego_trajectory = [scenario.get_ego_state_at_iteration(index)] + list(scenario.get_ego_future_trajectory(index, self._ego_trajectory_horizon, self._ego_trajectory_poses))
        sample = SimulationHistorySample(iteration=SimulationIteration(time_point=scenario.get_time_point(index), index=index), ego_state=scenario.get_ego_state_at_iteration(index), trajectory=InterpolatedTrajectory(ego_trajectory), observation=scenario.get_tracked_objects_at_iteration(index), traffic_light_status=scenario.get_traffic_light_status_at_iteration(index))
        scene = convert_sample_to_scene(map_name=scenario.map_api.map_name, database_interval=scenario.database_interval, traffic_light_status=scenario.get_traffic_light_status_at_iteration(index), expert_trajectory=ego_trajectory, mission_goal=scenario.get_mission_goal(), data=sample)
        return [scene]

class TestVectorMapFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs map features in vectorized format."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = get_test_nuplan_scenario()

    def test_vector_map_feature_builder(self) -> None:
        """
        Test VectorMapFeatureBuilder
        """
        feature_builder = VectorMapFeatureBuilder(radius=20, connection_scales=[2])
        self.assertEqual(feature_builder.get_feature_type(), VectorMap)
        features = feature_builder.get_features_from_scenario(self.scenario)
        self.assertEqual(type(features), VectorMap)
        ego_state = self.scenario.initial_ego_state
        detections = self.scenario.initial_tracked_objects
        meta_data = PlannerInitialization(map_api=self.scenario.map_api, mission_goal=self.scenario.get_mission_goal(), route_roadblock_ids=self.scenario.get_route_roadblock_ids())
        history = SimulationHistoryBuffer.initialize_from_list(1, [ego_state], [detections], self.scenario.database_interval)
        iteration = SimulationIteration(TimePoint(0), 0)
        tl_data = self.scenario.get_traffic_light_status_at_iteration(iteration.index)
        current_input = PlannerInput(iteration=iteration, history=history, traffic_light_data=tl_data)
        features_sim = feature_builder.get_features_from_simulation(current_input=current_input, initialization=meta_data)
        self.assertEqual(type(features_sim), VectorMap)
        self.assertTrue(np.allclose(features_sim.coords[0], features.coords[0], atol=0.0001))
        for connections, connections_simulation in zip(features_sim.multi_scale_connections[0].values(), features.multi_scale_connections[0].values()):
            self.assertTrue(np.allclose(connections, connections_simulation))
        for lane in range(len(features_sim.lane_groupings[0])):
            for lane_groupings, lane_groupings_simulation in zip(features_sim.lane_groupings[0][lane], features.lane_groupings[0][lane]):
                self.assertTrue(np.allclose(lane_groupings, lane_groupings_simulation))
        self.assertTrue(np.allclose(features_sim.on_route_status[0], features.on_route_status[0], atol=0.0001))
        self.assertTrue(np.allclose(features_sim.traffic_light_data[0], features.traffic_light_data[0]))

    def test_vector_map_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the VectorMapFeatureBuilder can be scripted properly.
        """
        feature_builder = VectorMapFeatureBuilder(radius=20, connection_scales=[2])
        self.assertEqual(feature_builder.get_feature_type(), VectorMap)
        scripted_builder = torch.jit.script(feature_builder)
        self.assertIsNotNone(scripted_builder)
        config = scripted_builder.precomputed_feature_config()
        self.assertTrue('initial_ego_state' in config)
        self.assertTrue('neighbor_vector_map' in config)
        self.assertTrue('radius' in config['neighbor_vector_map'])
        self.assertEqual('20', config['neighbor_vector_map']['radius'])
        num_lane_segment = 5
        num_connections = 7
        tensor_data = {'lane_segment_coords': torch.rand((num_lane_segment, 2, 2), dtype=torch.float64), 'lane_segment_conns': torch.zeros((num_connections, 2), dtype=torch.int64), 'on_route_status': torch.zeros((num_lane_segment, 2), dtype=torch.float32), 'traffic_light_array': torch.zeros((num_lane_segment, 4), dtype=torch.float32), 'anchor_state': torch.zeros((3,), dtype=torch.float64)}
        list_tensor_data = {'lane_segment_groupings': [torch.zeros(size=(2,), dtype=torch.int64) for _ in range(num_lane_segment)]}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_tensor_output, scripted_list_output, scripted_list_list_output = scripted_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        py_tensor_output, py_list_output, py_list_list_output = feature_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        self.assertEqual(0, len(scripted_tensor_output))
        self.assertEqual(0, len(py_tensor_output))
        self.assertEqual(len(scripted_list_output), len(py_list_output))
        for key in py_list_output:
            self.assertEqual(len(py_list_output[key]), len(scripted_list_output[key]))
            for i in range(len(py_list_output[key])):
                torch.testing.assert_close(py_list_output[key][i], scripted_list_output[key][i])
        self.assertEqual(len(py_list_list_output), len(scripted_list_list_output))
        for key in py_list_list_output:
            py_list = py_list_list_output[key]
            scripted_list = scripted_list_list_output[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                py = py_list[i]
                script = scripted_list[i]
                self.assertEqual(len(py), len(script))
                for j in range(len(py)):
                    torch.testing.assert_close(py[j], script[j])

class TestRasterUtils(unittest.TestCase):
    """Test raster building utility functions."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        scenario = get_test_nuplan_scenario()
        self.x_range = [-56.0, 56.0]
        self.y_range = [-56.0, 56.0]
        self.raster_shape = (224, 224)
        self.resolution = 0.5
        self.thickness = 2
        self.ego_state = scenario.initial_ego_state
        self.map_api = scenario.map_api
        self.tracked_objects = scenario.initial_tracked_objects
        self.map_features = {'LANE': 255, 'INTERSECTION': 255, 'STOP_LINE': 128, 'CROSSWALK': 128}
        ego_width = 2.297
        ego_front_length = 4.049
        ego_rear_length = 1.127
        self.ego_longitudinal_offset = 0.0
        self.ego_width_pixels = int(ego_width / self.resolution)
        self.ego_front_length_pixels = int(ego_front_length / self.resolution)
        self.ego_rear_length_pixels = int(ego_rear_length / self.resolution)

    def test_get_roadmap_raster(self) -> None:
        """
        Test get_roadmap_raster / get_agents_raster / get_baseline_paths_raster
        """
        self.assertGreater(len(self.tracked_objects.tracked_objects), 0)
        roadmap_raster = get_roadmap_raster(self.ego_state, self.map_api, self.map_features, self.x_range, self.y_range, self.raster_shape, self.resolution)
        agents_raster = get_agents_raster(self.ego_state, self.tracked_objects, self.x_range, self.y_range, self.raster_shape)
        ego_raster = get_ego_raster(self.raster_shape, self.ego_longitudinal_offset, self.ego_width_pixels, self.ego_front_length_pixels, self.ego_rear_length_pixels)
        baseline_paths_raster = get_baseline_paths_raster(self.ego_state, self.map_api, self.x_range, self.y_range, self.raster_shape, self.resolution, self.thickness)
        self.assertEqual(roadmap_raster.shape, self.raster_shape)
        self.assertEqual(agents_raster.shape, self.raster_shape)
        self.assertEqual(ego_raster.shape, self.raster_shape)
        self.assertEqual(baseline_paths_raster.shape, self.raster_shape)
        self.assertTrue(np.any(roadmap_raster))
        self.assertTrue(np.any(agents_raster))
        self.assertTrue(np.any(ego_raster))
        self.assertTrue(np.any(baseline_paths_raster))

class MockAbstractScenario(AbstractScenario):
    """Mock abstract scenario class used for testing."""

    def __init__(self, initial_time_us: TimePoint=TimePoint(time_us=1621641671099), time_step: float=0.5, number_of_future_iterations: int=10, number_of_past_iterations: int=0, initial_velocity: StateVector2D=StateVector2D(x=1.0, y=0.0), fixed_acceleration: StateVector2D=StateVector2D(x=0.0, y=0.0), number_of_detections: int=10, initial_ego_state: StateSE2=StateSE2(x=0.0, y=0.0, heading=0.0), mission_goal: StateSE2=StateSE2(10, 0, 0), tracked_object_types: List[TrackedObjectType]=[TrackedObjectType.VEHICLE]):
        """
        Create mocked scenario where ego starts with an initial velocity [m/s] and has a constant acceleration
            throughout (0 m/s^2 by default). The ego does not turn.
        :param initial_time_us: initial time from start point of scenario [us]
        :param time_step: time step in [s]
        :param number_of_future_iterations: number of iterations in the future
        :param number_of_past_iterations: number of iterations in the past
        :param initial_velocity: [m/s] velocity assigned to the ego at iteration 0
        :param fixed_acceleration: [m/s^2] constant ego acceleration throughout scenario
        :param number_of_detections: number of detections in the scenario
        :param initial_ego_state: Initial state of ego
        :param mission_goal: Dummy mission goal
        :param tracked_object_types: Types of tracked objects to mock
        """
        self._initial_time_us = initial_time_us
        self._time_step = time_step
        self._number_of_past_iterations = number_of_past_iterations
        self._number_of_future_iterations = number_of_future_iterations
        self._current_iteration = number_of_past_iterations
        self._total_iterations = number_of_past_iterations + number_of_future_iterations + 1
        self._tracked_object_types = tracked_object_types
        start_time_us = max(TimePoint(int(number_of_past_iterations * time_step * 1000000.0)), initial_time_us)
        time_horizon = (number_of_past_iterations + number_of_future_iterations) * time_step
        history_buffer = SimulationHistoryBuffer.initialize_from_list(buffer_size=10, ego_states=[EgoState.build_from_rear_axle(StateSE2(x=initial_ego_state.x, y=initial_ego_state.y, heading=initial_ego_state.heading), time_point=start_time_us, rear_axle_velocity_2d=initial_velocity, tire_steering_angle=0.0, rear_axle_acceleration_2d=fixed_acceleration, vehicle_parameters=self.ego_vehicle_parameters)], observations=[DetectionsTracks(TrackedObjects())], sample_interval=time_step)
        planner_input = PlannerInput(iteration=SimulationIteration(start_time_us, 0), history=history_buffer)
        planner = SimplePlanner(horizon_seconds=time_horizon, sampling_time=time_step, acceleration=fixed_acceleration.array)
        self._ego_states = planner.compute_trajectory(planner_input).get_sampled_trajectory()
        self._tracked_objects = [DetectionsTracks(TrackedObjects([get_sample_agent(token=str(idx + type_idx * number_of_detections), agent_type=agent_type, num_future_states=0) for idx in range(number_of_detections) for type_idx, agent_type in enumerate(self._tracked_object_types)])) for _ in range(self._total_iterations)]
        self._sensors = [Sensors(pointcloud={LidarChannel.MERGED_PC: np.eye(3) for _ in range(number_of_detections)}, images=None) for _ in range(self._total_iterations)]
        if len(self._ego_states) != len(self._tracked_objects) or len(self._ego_states) != self._total_iterations:
            raise RuntimeError('The dimensions of detections and ego trajectory is not the same!')
        self._mission_goal = mission_goal
        self._map_api = MockAbstractMap()
        self._token_suffix = str(uuid.uuid4())

    @property
    def token(self) -> str:
        """Implemented. See interface."""
        return f'mock_token_{self._token_suffix}'

    @property
    def log_name(self) -> str:
        """Implemented. See interface."""
        return 'mock_log_name'

    @property
    def scenario_name(self) -> str:
        """Implemented. See interface."""
        return 'mock_scenario_name'

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return get_pacifica_parameters()

    @property
    def scenario_type(self) -> str:
        """Implemented. See interface."""
        return 'mock_scenario_type'

    @property
    def map_api(self) -> AbstractMap:
        """Implemented. See interface."""
        return self._map_api

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        return self._time_step

    def get_number_of_iterations(self) -> int:
        """Implemented. See interface."""
        return self._number_of_future_iterations

    def get_time_point(self, iteration: int) -> TimePoint:
        """Implemented. See interface."""
        return self._ego_states[self._current_iteration + iteration].time_point

    def get_lidar_to_ego_transform(self) -> Transform:
        """Implemented. See interface."""
        return np.eye(4)

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Implemented. See interface."""
        return self._mission_goal

    def get_route_roadblock_ids(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def get_expert_goal_state(self) -> StateSE2:
        """Implemented. See interface."""
        return self._mission_goal

    def get_tracked_objects_at_iteration(self, iteration: int, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Implemented. See interface."""
        return self._tracked_objects[self._current_iteration + iteration]

    def get_tracked_objects_within_time_window_at_iteration(self, iteration: int, past_time_horizon: float, future_time_horizon: float, filter_track_tokens: Optional[Set[str]]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Implemented. See interface."""
        raise NotImplementedError

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]]=None) -> Sensors:
        """Implemented. See interface."""
        raise NotImplementedError

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Implemented. See interface."""
        return self._ego_states[self._current_iteration + iteration]

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Implemented. see interface."""
        dummy_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=1627066061949808)
        yield dummy_data

    def get_past_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """Gets past traffic light status."""
        dummy_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=1627066061949808)
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        for _ in range(num_samples):
            yield TrafficLightStatuses([dummy_data])

    def get_future_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """Gets future traffic light status."""
        dummy_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=1627066061949808)
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        for _ in range(num_samples):
            yield TrafficLightStatuses([dummy_data])

    def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Implemented. See interface."""
        ego_states = self.get_ego_future_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
        for state in ego_states:
            yield state.time_point

    def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Implemented. See interface."""
        ego_states = self.get_ego_past_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
        for state in ego_states:
            yield state.time_point

    def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Implemented. See interface."""
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._number_of_future_iterations - iteration >= indices[-1], f'Requested time horizon of {time_horizon}s is too long! Scenario future has length {(self._number_of_future_iterations - iteration) * self._time_step}s from the iteration {iteration}'
        for idx in indices:
            yield self._ego_states[self._current_iteration + iteration + idx]

    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Implemented. See interface."""
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._current_iteration + iteration >= indices[-1], f'Requested time horizon of {time_horizon}s is too long! Scenario past has length {(self._current_iteration + iteration) * self._time_step}s from the iteration {iteration}'
        for idx in reversed(indices):
            yield self._ego_states[self._current_iteration + iteration - idx]

    def get_past_sensors(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, channels: Optional[List[SensorChannel]]=None) -> Generator[Sensors, None, None]:
        """Implemented. See interface."""
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        for idx in indices:
            yield self._sensors[self._current_iteration + iteration - idx - 1]

    def get_past_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Implemented. See interface."""
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        if self._current_iteration + iteration < indices[-1]:
            raise ValueError(f'Requested time horizon of {time_horizon}s is too long! Scenario past has length {(self._current_iteration + iteration) * self._time_step}s from the iteration {iteration}')
        for idx in reversed(indices):
            yield self._tracked_objects[self._current_iteration + iteration - idx]

    def get_future_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Implemented. See interface."""
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._number_of_future_iterations - iteration >= indices[-1], f'Requested time horizon of {time_horizon}s is too long! Scenario future has length {(self._number_of_future_iterations - iteration) * self._time_step}s from the iteration {iteration}'
        for idx in indices:
            yield self._tracked_objects[self._current_iteration + iteration + idx]

class TestNuPlanScenarioIntegration(unittest.TestCase):
    """Integration test cases for nuplan_scenario.py"""

    def test_get_sensors_at_iteration_download(self) -> None:
        """
        Test that get_sensors_at_iteration is able to pull data from s3 correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario = get_test_nuplan_scenario(sensor_root=tmp_dir)
            sensor_path = Path(f'{tmp_dir}/{scenario.log_name}')

            def _get_image_paths() -> List[Path]:
                """:return: The expected path to the test image file."""
                return list(sensor_path.joinpath(f'{CameraChannel.CAM_R0.value}').glob('*.jpg'))

            def _get_pointcloud_paths() -> List[Path]:
                """:return: The expected path to the test pointcloud file."""
                return list(sensor_path.joinpath(f'{LidarChannel.MERGED_PC.value}').glob('*.pcd'))
            self.assertFalse(os.path.exists(sensor_path))
            sensors = scenario.get_sensors_at_iteration(0, [CameraChannel.CAM_R0, LidarChannel.MERGED_PC])
            self.assertIsNotNone(sensors.pointcloud)
            self.assertIsNotNone(sensors.images)
            self.assertTrue(os.path.exists(sensor_path))
            self.assertTrue(os.path.exists(_get_image_paths()[0]))
            self.assertTrue(os.path.exists(_get_pointcloud_paths()[0]))

class TestNuPlanScenario(unittest.TestCase):
    """
    Tests scenario for NuPlan
    """

    def _make_test_scenario(self) -> NuPlanScenario:
        """
        Creates a sample scenario to use for testing.
        """
        return NuPlanScenario(data_root='data_root/', log_file_load_path='data_root/log_name.db', initial_lidar_token=int_to_str_token(1234), initial_lidar_timestamp=2345, scenario_type='scenario_type', map_root='map_root', map_version='map_version', map_name='map_name', scenario_extraction_info=ScenarioExtractionInfo(scenario_name='scenario_name', scenario_duration=20, extraction_offset=1, subsample_ratio=0.5), ego_vehicle_parameters=get_pacifica_parameters(), sensor_root='sensor_root')

    def _get_sampled_sensor_tokens_in_time_window_patch(self, expected_log_file: str, expected_sensor_data_source: SensorDataSource, expected_start_timestamp: int, expected_end_timestamp: int, expected_subsample_step: int) -> Callable[[str, SensorDataSource, int, int, int], Generator[str, None, None]]:
        """
        Creates a patch for the get_sampled_lidarpc_tokens_in_time_window function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_start_timestamp: The expected start timestamp with which the function is expected to be called.
        :param expected_end_timestamp: The expected end timestamp with which the function is expected to be called.
        :param expected_subsample_step: The expected subsample step with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_sensor_data_source: SensorDataSource, actual_start_timestamp: int, actual_end_timestamp: int, actual_subsample_step: int) -> Generator[str, None, None]:
            """
            The patch function for get_sampled_lidarpc_tokens_in_time_window.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_start_timestamp, actual_start_timestamp)
            self.assertEqual(expected_end_timestamp, actual_end_timestamp)
            self.assertEqual(expected_subsample_step, actual_subsample_step)
            num_tokens = int((expected_end_timestamp - expected_start_timestamp) / (expected_subsample_step * 1000000.0))
            for token in range(num_tokens):
                yield int_to_str_token(token)
        return fxn

    def _get_download_file_if_necessary_patch(self, expected_data_root: str, expected_log_file_load_path: str) -> Callable[[str, str], str]:
        """
        Creates a patch for the download_file_if_necessary function that validates the arguments.
        :param expected_data_root: The data_root with which the function is expected to be called.
        :param expected_log_file_load_path: The log_file_load_path with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_data_root: str, actual_log_file_load_path: str) -> str:
            """
            The generated patch function.
            """
            self.assertEqual(expected_data_root, actual_data_root)
            self.assertEqual(expected_log_file_load_path, actual_log_file_load_path)
            return actual_log_file_load_path
        return fxn

    def _get_sensor_data_from_sensor_data_tokens_from_db_patch(self, expected_log_file: str, expected_sensor_data_source: SensorDataSource, expected_sensor_class: Type[SensorDataTableRow], expected_tokens: List[str]) -> Callable[[str, SensorDataSource, Type[SensorDataTableRow], List[str]], Generator[SensorDataTableRow, None, None]]:
        """
        Creates a patch for the get_images_from_lidar_tokens_patch function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_sensor_data_source: The sensor source with which the function is expected to be called.
        :param expected_sensor_class: The sensor class with which the function is expected to be called.
        :param expected_tokens: The tokens with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_sensor_data_source: SensorDataSource, actual_sensor_class: Type[SensorDataTableRow], actual_tokens: List[str]) -> Generator[SensorDataTableRow, None, None]:
            """
            The patch function for get_sensor_data_from_sensor_data_tokens_from_db.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_sensor_class, actual_sensor_class)
            self.assertEqual(expected_tokens, actual_tokens)
            lidar_token = actual_tokens[0]
            if expected_sensor_class == LidarPc:
                yield LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token))
            elif expected_sensor_class == ImageDBRow.Image:
                camera_token = str_token_to_int(lidar_token) + CAMERA_OFFSET
                yield ImageDBRow.Image(token=int_to_str_token(camera_token), next_token=int_to_str_token(camera_token), prev_token=int_to_str_token(camera_token), ego_pose_token=int_to_str_token(camera_token), camera_token=int_to_str_token(camera_token), filename_jpg=f'image_{camera_token}', timestamp=camera_token, channel=CameraChannel.CAM_R0.value)
            else:
                self.fail(f'Unexpected type: {expected_sensor_class}.')
        return fxn

    def _load_point_cloud_patch(self, expected_lidar_pc: LidarPc, expected_local_store: LocalStore, expected_s3_store: S3Store) -> Callable[[LidarPc, LocalStore, S3Store], LidarPointCloud]:
        """
        Creates a patch for the _load_point_cloud function that validates the arguments.
        :param expected_lidar_pc: The lidar pc with which the function is expected to be called.
        :param expected_local_store: The LocalStore with which the function is expected to be called.
        :param expected_s3_store: The S3Store with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_lidar_pc: LidarPc, actual_local_store: LocalStore, actual_s3_store: S3Store) -> LidarPointCloud:
            """
            The patch function for load_point_cloud.
            """
            self.assertEqual(expected_lidar_pc, actual_lidar_pc)
            self.assertEqual(expected_local_store, actual_local_store)
            self.assertEqual(expected_s3_store, actual_s3_store)
            return LidarPointCloud(np.eye(3))
        return fxn

    def _load_image_patch(self, expected_local_store: LocalStore, expected_s3_store: S3Store) -> Callable[[ImageDBRow.Image, LocalStore, S3Store], Image]:
        """
        Creates a patch for the _load_image_patch function and validates that argument is an Image object.
        :param expected_local_store: The LocalStore with which the function is expected to be called.
        :param expected_s3_store: The S3Store with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_image: ImageDBRow.Image, actual_local_store: LocalStore, actual_s3_store: S3Store) -> Image:
            """
            The patch function for load_image.
            """
            self.assertEqual(expected_local_store, actual_local_store)
            self.assertEqual(expected_s3_store, actual_s3_store)
            self.assertTrue(isinstance(actual_image, ImageDBRow.Image))
            return Image(PilImg.new('RGB', (500, 500)))
        return fxn

    def _get_images_from_lidar_tokens_patch(self, expected_log_file: str, expected_tokens: List[str], expected_channels: List[str], expected_lookahead_window_us: int, expected_lookback_window_us: int) -> Callable[[str, List[str], List[str], int, int], Generator[ImageDBRow.Image, None, None]]:
        """
        Creates a patch for the get_images_from_lidar_tokens_patch function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_tokens: The expected tokens with which the function is expected to be called.
        :param expected_channels: The expected channels with which the function is expected to be called.
        :param expected_lookahead_window_us: The expected lookahead window with which the function is expected to be called.
        :param expected_lookahead_window_us: The expected lookback window with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_tokens: List[str], actual_channels: List[str], actual_lookahead_window_us: int=50000, actual_lookback_window_us: int=50000) -> Generator[ImageDBRow.Image, None, None]:
            """
            The patch function for get_images_from_lidar_tokens.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_tokens, actual_tokens)
            self.assertEqual(expected_channels, actual_channels)
            self.assertEqual(expected_lookahead_window_us, actual_lookahead_window_us)
            self.assertEqual(expected_lookback_window_us, actual_lookback_window_us)
            for camera_token, channel in enumerate(actual_channels):
                if channel != LidarChannel.MERGED_PC.value:
                    yield ImageDBRow.Image(token=int_to_str_token(camera_token), next_token=int_to_str_token(camera_token), prev_token=int_to_str_token(camera_token), ego_pose_token=int_to_str_token(camera_token), camera_token=int_to_str_token(camera_token), filename_jpg=f'image_{camera_token}', timestamp=camera_token, channel=channel)
        return fxn

    def _get_sampled_lidarpcs_from_db_patch(self, expected_log_file: str, expected_initial_token: str, expected_sensor_data_source: SensorDataSource, expected_sample_indexes: Union[Generator[int, None, None], List[int]], expected_future: bool) -> Callable[[str, str, SensorDataSource, Union[Generator[int, None, None], List[int]], bool], Generator[LidarPc, None, None]]:
        """
        Creates a patch for the get_sampled_lidarpcs_from_db function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_initial_token: The initial token name with which the function is expected to be called.
        :param expected_sensor_data_source: The sensor source with which the function is expected to be called.
        :param expected_sample_indexes: The sample indexes with which the function is expected to be called.
        :param expected_future: The future with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_initial_token: str, actual_sensor_data_source: SensorDataSource, actual_sample_indexes: Union[Generator[int, None, None], List[int]], actual_future: bool) -> Generator[LidarPc, None, None]:
            """
            The patch function for get_images_from_lidar_tokens.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_initial_token, actual_initial_token)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_sample_indexes, actual_sample_indexes)
            self.assertEqual(expected_future, actual_future)
            for idx in actual_sample_indexes:
                lidar_token = int_to_str_token(idx)
                yield LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token))
        return fxn

    def test_implements_abstract_scenario_interface(self) -> None:
        """
        Tests that NuPlanScenario properly implements AbstractScenario interface.
        """
        assert_class_properly_implements_interface(AbstractScenario, NuPlanScenario)

    def test_token(self) -> None:
        """
        Tests that the token method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(int_to_str_token(1234), scenario.token)

    def test_log_name(self) -> None:
        """
        Tests that the log_name method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual('log_name', scenario.log_name)

    def test_scenario_name(self) -> None:
        """
        Tests that the scenario_name method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(int_to_str_token(1234), scenario.scenario_name)

    def test_ego_vehicle_parameters(self) -> None:
        """
        Tests that the ego_vehicle_parameters method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(get_pacifica_parameters(), scenario.ego_vehicle_parameters)

    def test_scenario_type(self) -> None:
        """
        Tests that the scenario_type method works properly
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual('scenario_type', scenario.scenario_type)

    def test_database_interval(self) -> None:
        """
        Tests that the database_interval method works properly
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(0.1, scenario.database_interval)

    def test_get_number_of_iterations(self) -> None:
        """
        Tests that the get_number_of_iterations method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn):
                scenario = self._make_test_scenario()
                self.assertEqual(10, scenario.get_number_of_iterations())

    def test_get_time_point(self) -> None:
        """
        Tests that the get_time_point method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        for iter_val in [0, 3, 5]:

            def token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch method for get_lidarpc_token_timstamp_from_db that validates the arguments.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(SensorDataSource(table='lidar_pc', sensor_table='lidar', sensor_token_column='lidar_token', channel='MergedPointCloud'), sensor_source)
                self.assertEqual(int_to_str_token(iter_val), token)
                return int(str_token_to_int(iter_val) + 5)
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.get_sensor_data_token_timestamp_from_db', token_timestamp_patch):
                scenario = self._make_test_scenario()
                self.assertEqual(iter_val + 5, scenario.get_time_point(iter_val).time_us)

    def test_get_tracked_objects_at_iteration(self) -> None:
        """
        Tests that the get_tracked_objects_at_iteration method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)
        for iter_val in [0, 2, 3]:

            def get_token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch for get_sensor_data_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(SensorDataSource(table='lidar_pc', sensor_table='lidar', sensor_token_column='lidar_token', channel='MergedPointCloud'), sensor_source)
                self.assertEqual(int_to_str_token(iter_val), token)
                return int(iter_val * 1000000.0)

            def tracked_objects_for_token_patch(log_file: str, token: str) -> Generator[TrackedObject, None, None]:
                """
                The patch for get_tracked_objects_for_lidarpc_token that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(int_to_str_token(iter_val), token)
                for idx in range(0, 4, 1):
                    box = OrientedBox(center=StateSE2(x=10, y=10, heading=10), length=10, width=10, height=10)
                    metadata = SceneObjectMetadata(token=int_to_str_token(idx + str_token_to_int(token)), track_token=int_to_str_token(idx + str_token_to_int(token) + 100), track_id=None, timestamp_us=0, category_name='foo')
                    if idx < 2:
                        yield Agent(tracked_object_type=TrackedObjectType.VEHICLE, oriented_box=box, velocity=StateVector2D(x=10, y=10), metadata=metadata)
                    else:
                        yield StaticObject(tracked_object_type=TrackedObjectType.CZONE_SIGN, oriented_box=box, metadata=metadata)

            def interpolate_future_waypoints_patch(waypoints: List[InterpolatableState], time_horizon: float, interval_s: float) -> List[Optional[InterpolatableState]]:
                """
                The patch for interpolate_future_waypoints that validates the arguments and generates fake data.
                """
                self.assertEqual(4, len(waypoints))
                self.assertEqual(0.5, interval_s)
                self.assertEqual(5, time_horizon)
                return waypoints

            def future_waypoints_for_agents_patch(log_file: str, agents_tokens: List[str], start_time: int, end_time: int) -> Generator[Tuple[str, Waypoint], None, None]:
                """
                The patch for get_future_waypoints_for_agents_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(iter_val * 1000000.0, start_time)
                self.assertEqual((iter_val + 5.5) * 1000000.0, end_time)
                self.assertEqual(2, len(agents_tokens))
                check_tokens = [str_token_to_int(t) for t in agents_tokens]
                check_tokens.sort()
                self.assertEqual(iter_val + 100, check_tokens[0])
                self.assertEqual(iter_val + 100 + 1, check_tokens[1])
                for i in range(8):
                    waypoint = Waypoint(time_point=TimePoint(time_us=i), oriented_box=OrientedBox(center=StateSE2(x=i, y=i, heading=i), length=i, width=i, height=i), velocity=None)
                    token = check_tokens[0] if i < 4 else check_tokens[1]
                    yield (int_to_str_token(token), waypoint)
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_for_lidarpc_token_from_db', tracked_objects_for_token_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db', future_waypoints_for_agents_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sensor_data_token_timestamp_from_db', get_token_timestamp_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.interpolate_future_waypoints', interpolate_future_waypoints_patch):
                scenario = self._make_test_scenario()
                agents = scenario.get_tracked_objects_at_iteration(iter_val, ground_truth_predictions)
                objects = agents.tracked_objects.tracked_objects
                self.assertEqual(4, len(objects))
                objects.sort(key=lambda x: str_token_to_int(x.metadata.token))
                for i in range(0, 2, 1):
                    test_obj = objects[i]
                    self.assertTrue(isinstance(test_obj, Agent))
                    self.assertEqual(iter_val + i, str_token_to_int(test_obj.metadata.token))
                    self.assertEqual(iter_val + i + 100, str_token_to_int(test_obj.metadata.track_token))
                    self.assertEqual(TrackedObjectType.VEHICLE, test_obj.tracked_object_type)
                    self.assertIsNotNone(test_obj.predictions)
                    object_waypoints = test_obj.predictions[0].waypoints
                    self.assertEqual(4, len(object_waypoints))
                    for j in range(len(object_waypoints)):
                        self.assertEqual(j + i * len(object_waypoints), object_waypoints[j].x)
                for i in range(2, 4, 1):
                    test_obj = objects[i]
                    self.assertTrue(isinstance(test_obj, StaticObject))
                    self.assertEqual(iter_val + i, str_token_to_int(test_obj.metadata.token))
                    self.assertEqual(iter_val + i + 100, str_token_to_int(test_obj.metadata.track_token))
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, test_obj.tracked_object_type)

    def test_get_tracked_objects_within_time_window_at_iteration(self) -> None:
        """
        Tests that the get_tracked_objects_within_time_window_at_iteration method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)
        for iter_val in [3, 4]:

            def get_token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch for get_sensor_data_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(SensorDataSource(table='lidar_pc', sensor_table='lidar', sensor_token_column='lidar_token', channel='MergedPointCloud'), sensor_source)
                self.assertEqual(int_to_str_token(iter_val), token)
                return int(iter_val * 1000000.0)

            def tracked_objects_within_time_interval_patch(log_file: str, start_timestamp: int, end_timestamp: int, filter_tokens: Optional[Set[str]]) -> Generator[TrackedObject, None, None]:
                """
                The patch for get_tracked_objects_for_lidarpc_token that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual((iter_val - 2) * 1000000.0, start_timestamp)
                self.assertEqual((iter_val + 2) * 1000000.0, end_timestamp)
                self.assertIsNone(filter_tokens)
                for time_idx in range(-2, 3, 1):
                    for idx in range(0, 4, 1):
                        box = OrientedBox(center=StateSE2(x=10, y=10, heading=10), length=10, width=10, height=10)
                        metadata = SceneObjectMetadata(token=int_to_str_token(idx + iter_val), track_token=int_to_str_token(idx + iter_val + 100), track_id=None, timestamp_us=(iter_val + time_idx) * 1000000.0, category_name='foo')
                        if idx < 2:
                            yield Agent(tracked_object_type=TrackedObjectType.VEHICLE, oriented_box=box, velocity=StateVector2D(x=10, y=10), metadata=metadata)
                        else:
                            yield StaticObject(tracked_object_type=TrackedObjectType.CZONE_SIGN, oriented_box=box, metadata=metadata)

            def interpolate_future_waypoints_patch(waypoints: List[InterpolatableState], time_horizon: float, interval_s: float) -> List[Optional[InterpolatableState]]:
                """
                The patch for interpolate_future_waypoints that validates the arguments and generates fake data.
                """
                self.assertEqual(4, len(waypoints))
                self.assertEqual(0.5, interval_s)
                self.assertEqual(5, time_horizon)
                return waypoints

            def future_waypoints_for_agents_patch(log_file: str, agents_tokens: List[str], start_time: int, end_time: int) -> Generator[Tuple[str, Waypoint], None, None]:
                """
                The patch for get_future_waypoints_for_agents_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(end_time - start_time, 5.5 * 1000000.0)
                self.assertEqual(2, len(agents_tokens))
                check_tokens = [str_token_to_int(t) for t in agents_tokens]
                check_tokens.sort()
                self.assertEqual(iter_val + 100, check_tokens[0])
                self.assertEqual(iter_val + 100 + 1, check_tokens[1])
                for i in range(8):
                    waypoint = Waypoint(time_point=TimePoint(time_us=i), oriented_box=OrientedBox(center=StateSE2(x=i, y=i, heading=i), length=i, width=i, height=i), velocity=None)
                    token = check_tokens[0] if i < 4 else check_tokens[1]
                    yield (int_to_str_token(token), waypoint)
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_within_time_interval_from_db', tracked_objects_within_time_interval_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db', future_waypoints_for_agents_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sensor_data_token_timestamp_from_db', get_token_timestamp_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.interpolate_future_waypoints', interpolate_future_waypoints_patch):
                scenario = self._make_test_scenario()
                agents = scenario.get_tracked_objects_within_time_window_at_iteration(iter_val, 2, 2, future_trajectory_sampling=ground_truth_predictions)
                objects = agents.tracked_objects.tracked_objects
                self.assertEqual(20, len(objects))
                num_objects = 2
                for window in range(0, 5, 1):
                    for object_num in range(0, 2, 1):
                        start_agent_idx = window * 2
                        test_obj = objects[start_agent_idx + object_num]
                        self.assertTrue(isinstance(test_obj, Agent))
                        self.assertEqual(iter_val + object_num, str_token_to_int(test_obj.metadata.token))
                        self.assertEqual(iter_val + object_num + 100, str_token_to_int(test_obj.metadata.track_token))
                        self.assertEqual(TrackedObjectType.VEHICLE, test_obj.tracked_object_type)
                        self.assertIsNotNone(test_obj.predictions)
                        object_waypoints = test_obj.predictions[0].waypoints
                        self.assertEqual(4, len(object_waypoints))
                        for j in range(len(object_waypoints)):
                            self.assertEqual(j + object_num * len(object_waypoints), object_waypoints[j].x)
                        start_obj_idx = 10 + window * 2
                        test_obj = objects[start_obj_idx + object_num]
                        self.assertTrue(isinstance(test_obj, StaticObject))
                        self.assertEqual(iter_val + object_num + num_objects, str_token_to_int(test_obj.metadata.token))
                        self.assertEqual(iter_val + object_num + num_objects + 100, str_token_to_int(test_obj.metadata.track_token))
                        self.assertEqual(TrackedObjectType.CZONE_SIGN, test_obj.tracked_object_type)

    def test_nuplan_scenario_memory_usage(self) -> None:
        """
        Test that repeatedly creating and destroying nuplan scenario does not cause memory leaks.
        """
        starting_usage = 0
        ending_usage = 0
        num_iterations = 5
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            hpy = guppy.hpy()
            hpy.setrelheap()
            for i in range(0, num_iterations, 1):
                scenario = self._make_test_scenario()
                _ = scenario.token
                gc.collect()
                heap = hpy.heap()
                _ = heap.size
                if i == num_iterations - 2:
                    starting_usage = heap.size
                if i == num_iterations - 1:
                    ending_usage = heap.size
            memory_difference_in_mb = (ending_usage - starting_usage) / (1024 * 1024)
            max_allowable_growth_mb = max(0.1, 0.1 * starting_usage / (1024 * 1024))
            self.assertGreater(max_allowable_growth_mb, memory_difference_in_mb)

    @patch(f'{TEST_PATH}.LocalStore', autospec=True)
    @patch(f'{TEST_PATH}.S3Store', autospec=True)
    @patch(f'{TEST_PATH}.os.getenv')
    def test_get_sensors_at_iteration(self, mock_get_env: Mock, mock_s3_store: Mock, mock_local_store: Mock) -> None:
        """Test get_sensors_at_iteration."""
        mock_url = 'url'
        mock_get_env.side_effect = ['s3', mock_url]
        mock_s3_store.return_value = Mock(spec_set=S3Store)
        mock_local_store.return_value = Mock(spec_set=LocalStore)
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0) + 2345, expected_end_timestamp=int(21 * 1000000.0) + 2345, expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch(f'{TEST_PATH}.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
        for iter_val in [0, 3, 5]:
            lidar_token = int_to_str_token(iter_val)
            get_sensor_data_from_sensor_data_tokens_from_db_fxn = self._get_sensor_data_from_sensor_data_tokens_from_db_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_sensor_class=LidarPc, expected_tokens=[lidar_token])
            get_images_from_lidar_tokens_fxn = self._get_images_from_lidar_tokens_patch(expected_log_file='data_root/log_name.db', expected_tokens=[lidar_token], expected_channels=[CameraChannel.CAM_R0.value, LidarChannel.MERGED_PC.value], expected_lookahead_window_us=50000, expected_lookback_window_us=50000)
            load_lidar_fxn = self._load_point_cloud_patch(LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token)), mock_local_store.return_value, mock_s3_store.return_value)
            load_image_fxn = self._load_image_patch(mock_local_store.return_value, mock_s3_store.return_value)
            with mock.patch(f'{TEST_PATH_UTILS}.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch(f'{TEST_PATH}.get_sensor_data_from_sensor_data_tokens_from_db', get_sensor_data_from_sensor_data_tokens_from_db_fxn), mock.patch(f'{TEST_PATH}.get_images_from_lidar_tokens', get_images_from_lidar_tokens_fxn), mock.patch(f'{TEST_PATH}.load_point_cloud', load_lidar_fxn), mock.patch(f'{TEST_PATH}.load_image', load_image_fxn):
                sensors = scenario.get_sensors_at_iteration(iter_val, [CameraChannel.CAM_R0, LidarChannel.MERGED_PC])
                self.assertEqual(LidarChannel.MERGED_PC, list(sensors.pointcloud.keys())[0])
                self.assertEqual(CameraChannel.CAM_R0, list(sensors.images.keys())[0])
                mock_local_store.assert_called_with('sensor_root')
                mock_s3_store.assert_called_with(f'{mock_url}/sensor_blobs', show_progress=True)

    @patch(f'{TEST_PATH}.LocalStore', autospec=True)
    @patch(f'{TEST_PATH}.S3Store', autospec=True)
    @patch(f'{TEST_PATH}.os.getenv')
    def test_get_past_sensors(self, mock_get_env: Mock, mock_s3_store: Mock, mock_local_store: Mock) -> None:
        """Test get_past_sensors."""
        mock_url = 'url'
        mock_get_env.side_effect = ['s3', mock_url]
        mock_s3_store.return_value = Mock(spec_set=S3Store)
        mock_local_store.return_value = Mock(spec_set=LocalStore)
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        lidar_token = int_to_str_token(9)
        get_sampled_lidarpcs_from_db_fxn = self._get_sampled_lidarpcs_from_db_patch(expected_log_file='data_root/log_name.db', expected_initial_token=int_to_str_token(0), expected_sensor_data_source=get_lidarpc_sensor_data(), expected_sample_indexes=[9], expected_future=False)
        get_images_from_lidar_tokens_fxn = self._get_images_from_lidar_tokens_patch(expected_log_file='data_root/log_name.db', expected_tokens=[lidar_token], expected_channels=[CameraChannel.CAM_R0.value, LidarChannel.MERGED_PC.value], expected_lookahead_window_us=50000, expected_lookback_window_us=50000)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        load_lidar_fxn = self._load_point_cloud_patch(LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token)), mock_local_store.return_value, mock_s3_store.return_value)
        load_image_fxn = self._load_image_patch(mock_local_store.return_value, mock_s3_store.return_value)
        with mock.patch(f'{TEST_PATH}.download_file_if_necessary', download_file_patch_fxn), mock.patch(f'{TEST_PATH_UTILS}.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch(f'{TEST_PATH}.get_sampled_lidarpcs_from_db', get_sampled_lidarpcs_from_db_fxn), mock.patch(f'{TEST_PATH}.get_images_from_lidar_tokens', get_images_from_lidar_tokens_fxn), mock.patch(f'{TEST_PATH}.load_point_cloud', load_lidar_fxn), mock.patch(f'{TEST_PATH}.load_image', load_image_fxn):
            scenario = self._make_test_scenario()
            past_sensors = list(scenario.get_past_sensors(iteration=0, time_horizon=0.4, num_samples=1, channels=[CameraChannel.CAM_R0, LidarChannel.MERGED_PC]))
            self.assertEqual(1, len(past_sensors))
            self.assertEqual(LidarChannel.MERGED_PC, list(past_sensors[0].pointcloud.keys())[0])
            self.assertEqual(CameraChannel.CAM_R0, list(past_sensors[0].images.keys())[0])
            mock_local_store.assert_called_with('sensor_root')
            mock_s3_store.assert_called_with(f'{mock_url}/sensor_blobs', show_progress=True)

    @patch(f'{TEST_PATH}.download_file_if_necessary', Mock())
    @patch(f'{TEST_PATH}.absolute_path_to_log_name', Mock())
    @patch(f'{TEST_PATH}.get_images_from_lidar_tokens', Mock(return_value=[]))
    @patch(f'{TEST_PATH}.NuPlanScenario._find_matching_lidar_pcs')
    @patch(f'{TEST_PATH}.load_point_cloud')
    @patch(f'{TEST_PATH}.load_image')
    def test_get_past_sensors_no_channels(self, mock_load_image: Mock, mock_load_point_cloud: Mock, mock__find_matching_lidar_pcs: Mock) -> None:
        """Test get_past_sensors when no channels are passed."""
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.token = 'token'
        mock_load_point_cloud.return_value = Mock(spec_set=LidarPointCloud)
        mock__find_matching_lidar_pcs.return_value = iter([mock_lidar_pc])
        scenario = self._make_test_scenario()
        past_sensors = list(scenario.get_past_sensors(iteration=0, time_horizon=0.4, num_samples=1, channels=None))
        mock__find_matching_lidar_pcs.assert_called_once()
        mock_load_point_cloud.assert_called_once()
        mock_load_image.assert_not_called()
        self.assertIsNone(past_sensors[0].images)
        self.assertIsNotNone(past_sensors[0].pointcloud)

    @patch(f'{TEST_PATH}.download_file_if_necessary', Mock())
    @patch(f'{TEST_PATH}.absolute_path_to_log_name', Mock())
    @patch(f'{TEST_PATH}.get_images_from_lidar_tokens', Mock(return_value=[]))
    @patch(f'{TEST_PATH}.extract_sensor_tokens_as_scenario', Mock(return_value=[None]))
    @patch(f'{TEST_PATH}.get_sensor_data_from_sensor_data_tokens_from_db')
    @patch(f'{TEST_PATH}.load_point_cloud')
    @patch(f'{TEST_PATH}.load_image')
    def test_get_sensors_at_iteration_no_channels(self, mock_load_image: Mock, mock_load_point_cloud: Mock, mock_get_sensor_data_from_sensor_data_tokens_from_db: Mock) -> None:
        """Test get_past_sensors when no channels are passed."""
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.token = 'token'
        mock_load_point_cloud.return_value = Mock(spec_set=LidarPointCloud)
        mock_get_sensor_data_from_sensor_data_tokens_from_db.return_value = iter([mock_lidar_pc])
        scenario = self._make_test_scenario()
        sensors = scenario.get_sensors_at_iteration(iteration=0, channels=None)
        mock_get_sensor_data_from_sensor_data_tokens_from_db.assert_called_once()
        mock_load_point_cloud.assert_called_once()
        mock_load_image.assert_not_called()
        self.assertIsNone(sensors.images)
        self.assertIsNotNone(sensors.pointcloud)

class DetectionTracksChallengeServicer(chpb_grpc.DetectionTracksChallengeServicer):
    """
    Servicer for exposing initialization and trajectory computation services to the client.
    It keeps a rolling history buffer to avoid unnecessary serialization/deserialization.
    """

    def __init__(self, planner_config: DictConfig, map_manager: MapManager):
        """
        :param planner_config: The planner configuration to instantiate the planner.
        :param map_manager: The map manager.
        """
        self.planner: Optional[AbstractPlanner] = None
        self._planner_config = planner_config
        self.map_manager = map_manager
        self.simulation_history_buffer: Optional[SimulationHistoryBuffer] = None
        self._initialized = False

    @staticmethod
    def _extract_simulation_iteration(planner_input_message: chpb.PlannerInput) -> SimulationIteration:
        return SimulationIteration(TimePoint(planner_input_message.simulation_iteration.time_us), planner_input_message.simulation_iteration.index)

    def _build_planner_input(self, planner_input_message: chpb.PlannerInput, buffer: Optional[SimulationHistoryBuffer]) -> PlannerInput:
        """
        Builds a PlannerInput from a serialized PlannerInput message and an existing data buffer
        :param planner_input_message: the serialized message
        :param buffer: The history buffer
        :return: PlannerInput object
        """
        simulation_iteration = self._extract_simulation_iteration(planner_input_message)
        new_data = planner_input_message.simulation_history_buffer
        states = []
        observations = []
        for serialized_state, serialized_observation in zip(new_data.ego_states, new_data.observations):
            states.append(pickle.loads(serialized_state))
            observations.append(pickle.loads(serialized_observation))
        if buffer is not None:
            buffer.extend(states, observations)
        else:
            buffer = SimulationHistoryBuffer.initialize_from_list(len(states), states, observations, new_data.sample_interval)
            self.simulation_history_buffer = buffer
        tl_data_messages = planner_input_message.traffic_light_data
        tl_data = [tl_status_data_from_proto_tl_status_data(tl_data_message) for tl_data_message in tl_data_messages]
        return PlannerInput(iteration=simulation_iteration, history=buffer, traffic_light_data=tl_data)

    def InitializePlanner(self, planner_initialization_message: chpb.PlannerInitializationLight, context: Any) -> chpb.Empty:
        """
        Service to initialize the planner given the initialization request.
        :param planner_initialization_message: Message containing initialization details
        :param context
        """
        planners = build_planners(self._planner_config, None)
        assert len(planners) == 1, f'Configuration should build exactly 1 planner, got {len(planners)} instead!'
        self.planner = planners[0]
        logger.info('Initialization request received..')
        route_roadblock_ids = planner_initialization_message.route_roadblock_ids
        mission_goal = se2_from_proto_se2(planner_initialization_message.mission_goal)
        map_api = self.map_manager.get_map(planner_initialization_message.map_name)
        map_api.initialize_all_layers()
        planner_initialization = PlannerInitialization(route_roadblock_ids=route_roadblock_ids, mission_goal=mission_goal, map_api=map_api)
        self.simulation_history_buffer = None
        self.planner.initialize(planner_initialization)
        logging.info('Planner initialized!')
        self._initialized = True
        return chpb.Empty()

    def ComputeTrajectory(self, planner_input_message: chpb.PlannerInput, context: Any) -> chpb.Trajectory:
        """
        Service to compute a trajectory given a planner input message
        :param planner_input_message: Message containing the input to the planner
        :param context
        :return Message containing the computed trajectories
        """
        assert self._initialized, 'Planner has not been initialized. Please call InitializePlanner'
        planner_inputs = self._build_planner_input(planner_input_message, self.simulation_history_buffer)
        if isinstance(self.planner, AbstractPlanner):
            trajectory = self.planner.compute_trajectory(planner_inputs)
            return proto_traj_from_inter_traj(trajectory)
        raise RuntimeError('The planner was not initialized correctly!')

def proto_traj_from_inter_traj(trajectory: AbstractTrajectory) -> chpb.Trajectory:
    """
    Serializes AbstractTrajectory to a Trajectory message
    :param trajectory: The AbstractTrajectory object
    :return: The corresponding Trajectory message
    """
    return chpb.Trajectory(ego_states=[proto_ego_state_from_ego_state(state) for state in trajectory.get_sampled_trajectory()])

def vector_2d_from_proto_vector_2d(vector: chpb.StateVector2D) -> StateVector2D:
    """
    Deserializes StateVector2D message to a StateVector2D object
    :param vector: The proto StateVector2D message
    :return: The corresponding StateVector2D object
    """
    return StateVector2D(x=vector.x, y=vector.y)

def proto_ego_state_from_ego_state(ego_state: EgoState) -> chpb.EgoState:
    """
    Serializes EgoState to a EgoState message
    :param ego_state: The EgoState object
    :return: The corresponding EgoState message
    """
    return chpb.EgoState(rear_axle_pose=proto_se2_from_se2(ego_state.rear_axle), rear_axle_velocity_2d=proto_vector_2d_from_vector_2d(ego_state.dynamic_car_state.rear_axle_velocity_2d), rear_axle_acceleration_2d=proto_vector_2d_from_vector_2d(ego_state.dynamic_car_state.rear_axle_acceleration_2d), tire_steering_angle=ego_state.tire_steering_angle, time_us=ego_state.time_us, angular_vel=ego_state.dynamic_car_state.angular_velocity, angular_accel=ego_state.dynamic_car_state.angular_acceleration)

def proto_vector_2d_from_vector_2d(vector: StateVector2D) -> chpb.StateVector2D:
    """
    Serializes StateVector2D to a StateVector2D message
    :param vector: The StateVector2D object
    :return: The corresponding StateVector2D message
    """
    return chpb.StateVector2D(x=vector.x, y=vector.y)

def ego_state_from_proto_ego_state(ego_state: chpb.EgoState) -> EgoState:
    """
    Deserializes EgoState message to a EgoState object
    :param ego_state: The proto EgoState message
    :return: The corresponding EgoState object
    """
    vehicle_parameters = get_pacifica_parameters()
    return EgoState.build_from_rear_axle(rear_axle_pose=se2_from_proto_se2(ego_state.rear_axle_pose), rear_axle_velocity_2d=vector_2d_from_proto_vector_2d(ego_state.rear_axle_velocity_2d), rear_axle_acceleration_2d=vector_2d_from_proto_vector_2d(ego_state.rear_axle_acceleration_2d), tire_steering_angle=ego_state.tire_steering_angle, time_point=TimePoint(ego_state.time_us), angular_vel=ego_state.angular_vel, angular_accel=ego_state.angular_accel, vehicle_parameters=vehicle_parameters)

class TestProtoConverters(unittest.TestCase):
    """Tests proto converters by checking if composition is idempotent."""

    def test_trajectory_conversions(self) -> None:
        """Tests conversions between trajectory object and messages."""
        trajectory = InterpolatedTrajectory([get_sample_ego_state(StateSE2(0, 1, 2)), get_sample_ego_state(StateSE2(1, 2, 3), time_us=1)])
        result = interp_traj_from_proto_traj(proto_traj_from_inter_traj(trajectory))
        for result_state, trajectory_state in zip(result.get_sampled_trajectory(), trajectory.get_sampled_trajectory()):
            np.allclose(result_state.to_split_state().linear_states, trajectory_state.to_split_state().linear_states)
            np.allclose(result_state.to_split_state().angular_states, trajectory_state.to_split_state().angular_states)

    def test_tl_status_type_conversions(self) -> None:
        """Tests conversions between TL status data and messages."""
        tl_status_type = TrafficLightStatusType.RED
        result = tl_status_type_from_proto_tl_status_type(proto_tl_status_type_from_tl_status_type(tl_status_type))
        self.assertEqual(tl_status_type, result)

    def test_tl_status_data_conversions(self) -> None:
        """Tests conversions between TL status type and messages."""
        tl_status = TrafficLightStatusData(TrafficLightStatusType.RED, 123, 456)
        result = tl_status_data_from_proto_tl_status_data(proto_tl_status_data_from_tl_status_data(tl_status))
        self.assertEqual(tl_status, result)

