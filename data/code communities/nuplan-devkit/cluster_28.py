# Cluster 28

class TestCamera(unittest.TestCase):
    """Test class Camera"""

    def setUp(self) -> None:
        """
        Initializes a test Camera
        """
        self.camera = get_test_nuplan_camera()

    @patch('nuplan.database.nuplan_db_orm.camera.inspect', autospec=True)
    def test_session(self, inspect: Mock) -> None:
        """
        Tests _session method
        """
        mock_session = PropertyMock()
        inspect.return_value = Mock()
        inspect.return_value.session = mock_session
        result = self.camera._session()
        inspect.assert_called_once_with(self.camera)
        mock_session.assert_called_once()
        self.assertEqual(result, mock_session.return_value)

    @patch('nuplan.database.nuplan_db_orm.camera.simple_repr', autospec=True)
    def test_repr(self, simple_repr: Mock) -> None:
        """
        Tests string representation
        """
        result = self.camera.__repr__()
        simple_repr.assert_called_once_with(self.camera)
        self.assertEqual(result, simple_repr.return_value)

    @patch('nuplan.database.nuplan_db_orm.camera.np.array', autospec=True)
    def test_intrinsic_np(self, np_array: Mock) -> None:
        """
        Test property - camera intrinsic.
        """
        result = self.camera.intrinsic_np
        np_array.assert_called_once_with(self.camera.intrinsic)
        self.assertEqual(result, np_array.return_value)

    @patch('nuplan.database.nuplan_db_orm.camera.np.array', autospec=True)
    def test_distortion_np(self, np_array: Mock) -> None:
        """
        Test property - camera distrotion.
        """
        result = self.camera.distortion_np
        np_array.assert_called_once_with(self.camera.distortion)
        self.assertEqual(result, np_array.return_value)

    @patch('nuplan.database.nuplan_db_orm.camera.np.array', autospec=True)
    def test_translation_np(self, np_array: Mock) -> None:
        """
        Test property - translation.
        """
        result = self.camera.translation_np
        np_array.assert_called_once_with(self.camera.translation)
        self.assertEqual(result, np_array.return_value)

    def test_quaternion(self) -> None:
        """
        Test property - rotation in quaternion.
        """
        result = self.camera.quaternion
        np.testing.assert_array_equal(self.camera.rotation, result.elements)

    def test_trans_matrix_and_inv(self) -> None:
        """
        Test two properties - transformation matrix and its inverse.
        """
        trans_mat = self.camera.trans_matrix
        inv_trans_mat = self.camera.trans_matrix_inv
        np.testing.assert_allclose(trans_mat @ inv_trans_mat, np.eye(4), atol=0.001)

@lru_cache(maxsize=1)
def get_test_nuplan_camera() -> Camera:
    """Get a nuPlan camera object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.camera[DEFAULT_TEST_CAMERA_INDEX]

class TestLidarPc(unittest.TestCase):
    """Tests the LidarBox class"""

    def setUp(self) -> None:
        """Sets up for the tests cases"""
        self.lidar_pc = get_test_nuplan_lidarpc()
        self.lidar_pc_with_blob = get_test_nuplan_lidarpc_with_blob()

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the _session property"""
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock
        result = self.lidar_pc._session
        inspect_mock.assert_called_once_with(self.lidar_pc)
        self.assertEqual(result, session_mock)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the __repr__ method"""
        result = self.lidar_pc.__repr__()
        simple_repr_mock.assert_called_once_with(self.lidar_pc)
        self.assertEqual(result, simple_repr_mock.return_value)

    def test_log(self) -> None:
        """Tests the log property"""
        result = self.lidar_pc.log
        self.assertIsInstance(result, Log)

    def test_future_ego_pose_has_next(self) -> None:
        """Tests the future_ego_pose method when there is a future ego pose"""
        result = self.lidar_pc.future_ego_pose()
        self.assertEqual(result, self.lidar_pc.next.ego_pose)

    def test_future_ego_pose_no_next(self) -> None:
        """Tests the future_ego_pose method when there is no future ego pose"""
        lidar_pc = deepcopy(self.lidar_pc)
        lidar_pc.next = None
        result = lidar_pc.future_ego_pose()
        self.assertEqual(result, None)

    def test_past_ego_pose_has_prev(self) -> None:
        """Tests the past_ego_pose method when there is a past ego pose"""
        result = self.lidar_pc.past_ego_pose()
        self.assertEqual(result, self.lidar_pc.prev.ego_pose)

    def test_past_ego_pose_no_prev(self) -> None:
        """Tests the past_ego_pose method when there is no past ego pose"""
        lidar_pc = deepcopy(self.lidar_pc)
        lidar_pc.prev = None
        result = lidar_pc.past_ego_pose()
        self.assertEqual(result, None)

    def test_future_or_past_ego_poses_prev_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_poses"""
        number, mode, direction = (1, 'n_poses', 'prev')
        result = self.lidar_pc.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_seconds"""
        number, mode, direction = (1, 'n_seconds', 'prev')
        result = self.lidar_pc.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev and mode is unknown"""
        number, mode, direction = (1, 'unknown_mode', 'prev')
        with self.assertRaises(ValueError):
            self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_next_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_poses"""
        number, mode, direction = (1, 'n_poses', 'next')
        result = self.lidar_pc.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_seconds"""
        number, mode, direction = (1, 'n_seconds', 'next')
        result = self.lidar_pc.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next and mode is unknown"""
        number, mode, direction = (1, 'unknown_mode', 'next')
        with self.assertRaises(ValueError):
            self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_unknown_direction(self) -> None:
        """Tests the future_or_past_ego_poses when direction is unknown"""
        number, mode, direction = (1, 'unknown_mode', 'unknown_direction')
        with self.assertRaises(ValueError):
            self.lidar_pc.future_or_past_ego_poses(number, mode, direction)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.LidarPointCloud.from_buffer', autospec=True)
    def test_load_channel_is_merged_point_cloud(self, from_buffer_mock: Mock) -> None:
        """Tests the load method when lidar channel is MergedPointCloud"""
        db = get_test_nuplan_db()
        result = self.lidar_pc.load(db)
        self.assertEqual(result, from_buffer_mock.return_value)

    def test_load_channel_is_not_implemented(self) -> None:
        """Tests the load method when lidar channel is not implemented"""
        db = get_test_nuplan_db()
        lidar_pc = deepcopy(self.lidar_pc)
        lidar_pc.lidar.channel = 'UnknownPointCloud'
        with self.assertRaises(NotImplementedError):
            lidar_pc.load(db)

    def test_load_bytes(self) -> None:
        """Tests the load bytes method"""
        db = get_test_nuplan_db()
        result = self.lidar_pc_with_blob.load_bytes(db)
        self.assertIsNotNone(result)

    def test_path(self) -> None:
        """Tests the path property"""
        db = get_test_nuplan_db()
        result = self.lidar_pc_with_blob.path(db)
        self.assertIsInstance(result, str)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.get_boxes', autospec=True)
    def test_boxes(self, get_boxes_mock: Mock) -> None:
        """Tests the boxes method"""
        result = self.lidar_pc.boxes()
        self.assertEqual(result, get_boxes_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.pack_future_boxes', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.lidar_pc.get_future_box_sequence', autospec=True)
    def test_boxes_with_future_waypoints(self, get_future_box_sequence_mock: Mock, pack_future_boxes_mock: Mock) -> None:
        """Tests the boxes_with_future_waypoints method"""
        future_horizon_len_s, future_interval_s = (1.0, 1.0)
        result = self.lidar_pc.boxes_with_future_waypoints(future_horizon_len_s, future_interval_s)
        get_future_box_sequence_mock.assert_called_once()
        pack_future_boxes_mock.assert_called_once_with(get_future_box_sequence_mock.return_value, future_interval_s, future_horizon_len_s)
        self.assertEqual(result, pack_future_boxes_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_pc.render_on_map', autospec=True)
    def test_render(self, render_on_map_mock: Mock) -> None:
        """Tests the render method"""
        db = get_test_nuplan_db()
        result = self.lidar_pc_with_blob.render(db)
        render_on_map_mock.assert_called_once()
        self.assertIsInstance(result, Axes)

    def test_past_ego_poses(self) -> None:
        """Test if past ego poses are returned correctly."""
        n_ego_poses = 4
        lidar_pc = self.lidar_pc.next.next.next
        past_ego_poses = lidar_pc.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(ego_pose.timestamp, past_ego_poses[i].timestamp, 'Timestamps of current EgoPose must be greater than past EgoPoses.')

    def test_future_ego_poses(self) -> None:
        """Test if future ego poses are returned correctly."""
        n_ego_poses = 4
        future_ego_poses = self.lidar_pc.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(ego_pose.timestamp, future_ego_poses[i].timestamp, 'Timestamps of current EgoPose must be less that future EgoPoses.')

@lru_cache(maxsize=1)
def get_test_nuplan_lidarpc(index: Union[int, str]=DEFAULT_TEST_LIDAR_PC_INDEX) -> LidarPc:
    """Get a nuPlan lidarpc object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar_pc[index]

@lru_cache(maxsize=1)
def get_test_nuplan_lidarpc_with_blob() -> LidarPc:
    """Get a nuPlan lidarpc object with blob with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar_pc[DEFAULT_TEST_LIDAR_PC_WITH_BLOB_TOKEN]

@lru_cache(maxsize=1)
def get_test_nuplan_db() -> NuPlanDB:
    """Get a nuPlan DB object with default settings to be used in testing."""
    return get_test_nuplan_db_nocache()

class TestLidarBox(unittest.TestCase):
    """Tests the LidarBox class"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.lidar_box_vehicle = get_test_nuplan_lidar_box_vehicle()
        self.lidar_box = get_test_nuplan_lidar_box()

    @patch('nuplan.database.nuplan_db_orm.lidar_box.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the _session property"""
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock
        result = self.lidar_box._session()
        inspect_mock.assert_called_once_with(self.lidar_box)
        self.assertEqual(result, session_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the __repr__ method"""
        result = self.lidar_box.__repr__()
        simple_repr_mock.assert_called_once_with(self.lidar_box)
        self.assertEqual(result, simple_repr_mock.return_value)

    def test_log(self) -> None:
        """Tests the log property"""
        result = self.lidar_box.log
        self.assertIsInstance(result, Log)

    def test_category(self) -> None:
        """Tests the category property"""
        result = self.lidar_box.category
        self.assertIsInstance(result, Category)

    def test_timestamp(self) -> None:
        """Tests the timestamp property"""
        result = self.lidar_box.timestamp
        self.assertIsInstance(result, int)

    def test_distance_to_ego(self) -> None:
        """Tests the distance_to_ego property"""
        x = self.lidar_box.x
        y = self.lidar_box.y
        x_ego = self.lidar_box.lidar_pc.ego_pose.x
        y_ego = self.lidar_box.lidar_pc.ego_pose.y
        expected_result = math.sqrt((x - x_ego) * (x - x_ego) + (y - y_ego) * (y - y_ego))
        actual_result = self.lidar_box.distance_to_ego
        self.assertEqual(expected_result, actual_result)

    def test_size(self) -> None:
        """Tests the size property"""
        result = self.lidar_box.size
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], self.lidar_box.width)
        self.assertEqual(result[1], self.lidar_box.length)
        self.assertEqual(result[2], self.lidar_box.height)

    def test_translation(self) -> None:
        """Tests the translation property"""
        result = self.lidar_box.translation
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], self.lidar_box.x)
        self.assertEqual(result[1], self.lidar_box.y)
        self.assertEqual(result[2], self.lidar_box.z)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.Quaternion', autospec=True)
    def test_rotation(self, quaternion_mock: Mock) -> None:
        """Tests the rotation property"""
        result = self.lidar_box.rotation
        self.assertIsInstance(result, list)
        quaternion_mock.assert_called()

    @patch('nuplan.database.nuplan_db_orm.lidar_box.Quaternion', autospec=True)
    def test_quaternion(self, quaternion_mock: Mock) -> None:
        """Tests the quaternion property"""
        result = self.lidar_box.quaternion
        self.assertEqual(result, quaternion_mock.return_value)
        quaternion_mock.assert_called()

    @patch('nuplan.database.nuplan_db_orm.lidar_box.np.array', autospec=True)
    def test_translation_np(self, np_array_mock: Mock) -> None:
        """Tests the translation_np property"""
        result = self.lidar_box.translation_np
        np_array_mock.assert_called_once_with(self.lidar_box.translation)
        self.assertEqual(result, np_array_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.np.array', autospec=True)
    def test_size_np(self, np_array_mock: Mock) -> None:
        """Tests the size_np property"""
        result = self.lidar_box.size_np
        np_array_mock.assert_called_once_with(self.lidar_box.size)
        self.assertEqual(result, np_array_mock.return_value)

    def test_get_box_items(self) -> None:
        """Tests the _get_box_items method"""
        result = self.lidar_box._get_box_items()
        self.assertEqual(len(result), 2)

    def test_find_box_out_of_bounds(self) -> None:
        """Tests the _find_box method index is out of bounds"""
        result = self.lidar_box._find_box(maxsize)
        self.assertEqual(result, None)

    def test_find_box_within_bounds(self) -> None:
        """Tests the _find_box method index is within bounds"""
        result = self.lidar_box._find_box(0)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_poses"""
        number, mode, direction = (1, 'n_poses', 'prev')
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_seconds"""
        number, mode, direction = (1, 'n_seconds', 'prev')
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev and mode is unknown"""
        number, mode, direction = (1, 'unknown_mode', 'prev')
        with self.assertRaises(ValueError):
            self.lidar_box.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_next_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_poses"""
        number, mode, direction = (1, 'n_poses', 'next')
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_seconds"""
        number, mode, direction = (1, 'n_seconds', 'next')
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next and mode is unknown"""
        number, mode, direction = (1, 'unknown_mode', 'next')
        with self.assertRaises(ValueError):
            self.lidar_box.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_unknown_direction(self) -> None:
        """Tests the future_or_past_ego_poses when direction is unknown"""
        number, mode, direction = (1, 'unknown_mode', 'unknown_direction')
        with self.assertRaises(ValueError):
            self.lidar_box.future_or_past_ego_poses(number, mode, direction)

    def test_temporal_neighbours_prev_exists(self) -> None:
        """Tests the _temporal_neighbours method when prev exists"""
        result = self.lidar_box._temporal_neighbors()
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], self.lidar_box.prev)

    def test_temporal_neighbours_prev_is_empty(self) -> None:
        """Tests the _temporal_neighbours method when prev does not exist"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.prev = None
        result = lidar_box._temporal_neighbors()
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], lidar_box)

    def test_temporal_neighbours_next_exists(self) -> None:
        """Tests the _temporal_neighbours method when next exists"""
        result = self.lidar_box._temporal_neighbors()
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1], self.lidar_box.next)

    def test_temporal_neighbours_next_is_empty(self) -> None:
        """Tests the _temporal_neighbours method when next does not exist"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next = None
        result = lidar_box._temporal_neighbors()
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1], lidar_box)

    def test_velocity_no_next_and_prev(self) -> None:
        """Tests the velocity property when next and prev does not exist"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next = None
        lidar_box.prev = None
        result = lidar_box.velocity
        self.assertTrue(np.isnan(result).any())

    def test_velocity_time_diff_exceed_limit(self) -> None:
        """Tests the velocity property when the difference between timestamps exceed limit"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next.lidar_pc.timestamp = lidar_box.prev.lidar_pc.timestamp + 1000000000
        result = lidar_box.velocity
        self.assertTrue(np.isnan(result).any())

    def test_velocity_default(self) -> None:
        """Tests the default velocity property, should not return any NaN values"""
        result = self.lidar_box.velocity
        self.assertFalse(np.isnan(result).any())

    def test_angular_velocity_no_next_and_prev(self) -> None:
        """Tests the angular_velocity property when next and prev does not exist"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next = None
        lidar_box.prev = None
        result = lidar_box.angular_velocity
        self.assertTrue(np.isnan(result))

    def test_angular_velocity_time_diff_exceed_limit(self) -> None:
        """Tests the angular_velocity property when the difference between timestamps exceed limit"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next.lidar_pc.timestamp = lidar_box.prev.lidar_pc.timestamp + 1000000000
        result = lidar_box.angular_velocity
        self.assertTrue(np.isnan(result))

    def test_angular_velocity_default(self) -> None:
        """Tests the default angular_velocity property, should not return any NaN values"""
        result = self.lidar_box.angular_velocity
        self.assertFalse(np.isnan(result))

    def test_box(self) -> None:
        """Tests the box method"""
        result = self.lidar_box.box()
        self.assertIsInstance(result, Box3D)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.PredictedTrajectory', autospec=True)
    def test_tracked_object_is_agent(self, predicted_trajectory_mock: Mock) -> None:
        """Tests the tracked_object method"""
        future_waypoints = Mock()
        predicted_trajectory_mock.return_value.probability = 1.0
        result = self.lidar_box_vehicle.tracked_object(future_waypoints)
        predicted_trajectory_mock.assert_called_once_with(1.0, future_waypoints)
        self.assertIsInstance(result, Agent)

    def test_tracked_object_is_static_object(self) -> None:
        """Tests the tracked_object method"""
        future_waypoints = Mock()
        result = self.lidar_box.tracked_object(future_waypoints)
        self.assertIsInstance(result, StaticObject)

    def test_velocity(self) -> None:
        """Test if velocity is calculated correctly."""
        self.assertTrue(self.lidar_box.prev is not None)
        self.assertTrue(self.lidar_box.next is not None)
        prev_lidar_box: LidarBox = self.lidar_box.prev
        next_lidar_box: LidarBox = self.lidar_box.next
        time_diff = 1e-06 * (next_lidar_box.timestamp - prev_lidar_box.timestamp)
        pos_diff = self.lidar_box.velocity * time_diff
        pos_next = next_lidar_box.translation_np
        pos_next_pred = prev_lidar_box.translation_np + pos_diff
        np.testing.assert_array_almost_equal(pos_next[:2], pos_next_pred[:2], decimal=4)

    def test_angular_velocity(self) -> None:
        """Test if angular velocity is calculated correctly."""
        self.assertTrue(self.lidar_box.prev is not None)
        self.assertTrue(self.lidar_box.next is not None)
        prev_lidar_box: LidarBox = self.lidar_box.prev
        next_lidar_box: LidarBox = self.lidar_box.next
        time_diff = 1e-06 * (next_lidar_box.timestamp - prev_lidar_box.timestamp)
        yaw_diff = self.lidar_box.angular_velocity * time_diff
        yaw_prev = quaternion_yaw(prev_lidar_box.quaternion)
        q_yaw_prev = Quaternion(np.array([np.cos(yaw_prev / 2), 0, 0, np.sin(yaw_prev / 2)]))
        q_yaw_next_pred = Quaternion(np.array([np.cos(yaw_diff / 2), 0, 0, np.sin(yaw_diff / 2)])) * q_yaw_prev
        yaw_next_pred = quaternion_yaw(q_yaw_next_pred)
        yaw_next = quaternion_yaw(next_lidar_box.quaternion)
        self.assertAlmostEqual(yaw_next, yaw_next_pred, delta=0.0001)

    def test_next(self) -> None:
        """Test next."""
        self.assertGreater(self.lidar_box.next.timestamp, self.lidar_box.timestamp, 'Timestamp of succeeding box must be greater then current box.')

    def test_prev(self) -> None:
        """Test prev."""
        self.assertLess(self.lidar_box.prev.timestamp, self.lidar_box.timestamp, 'Timestamp of preceding box must be lower then current box.')

    def test_past_ego_poses(self) -> None:
        """Test if past ego poses are returned correctly."""
        n_ego_poses = 4
        past_ego_poses = self.lidar_box.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = self.lidar_box.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(ego_pose.timestamp, past_ego_poses[i].timestamp, 'Timestamp of current EgoPose must be greater than past EgoPoses')

    def test_future_ego_poses(self) -> None:
        """Test if future ego poses are returned correctly."""
        n_ego_poses = 4
        future_ego_poses = self.lidar_box.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.lidar_box.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(ego_pose.timestamp, future_ego_poses[i].timestamp, 'Timestamp of current EgoPose must be less than future EgoPoses ')

    def test_get_box_items_to_iterate(self) -> None:
        """Tests the get_box_items_to_iterate method"""
        result = self.lidar_box.get_box_items_to_iterate()
        self.assertTrue(self.lidar_box.timestamp in result)
        self.assertEqual(self.lidar_box.prev, result[self.lidar_box.timestamp][0])
        self.assertEqual(self.lidar_box.next, result[self.lidar_box.timestamp][1])

    @patch('nuplan.database.nuplan_db_orm.lidar_box.IterableLidarBox', autospec=True)
    def test_iter(self, iterable_lidar_box_mock: Mock) -> None:
        """Tests the iterator for LidarBox"""
        result = iter(self.lidar_box)
        iterable_lidar_box_mock.assert_called_once_with(self.lidar_box)
        self.assertEqual(result, iterable_lidar_box_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.IterableLidarBox', autospec=True)
    def test_reverse_iter(self, iterable_lidar_box_mock: Mock) -> None:
        """Tests the reverse iterator for LidarBox"""
        result = reversed(self.lidar_box)
        iterable_lidar_box_mock.assert_called_once_with(self.lidar_box, reverse=True)
        self.assertEqual(result, iterable_lidar_box_mock.return_value)

@lru_cache(maxsize=1)
def get_test_nuplan_lidar_box_vehicle() -> LidarBox:
    """Get a nuPlan lidar box object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar_box[DEFAULT_TEST_LIDAR_BOX_INDEX_VEHICLE]

@lru_cache(maxsize=1)
def get_test_nuplan_lidar_box() -> LidarBox:
    """Get a nuPlan lidar box object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar_box[DEFAULT_TEST_LIDAR_BOX_INDEX]

class TestPointCloudPreparation(unittest.TestCase):
    """Test preparation of point cloud method (standalone method)."""

    def setUp(self) -> None:
        """Setup funciton for class."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()

    def test_prepare_pointcloud_points(self) -> None:
        """
        Tests if the lidar point clouds are properly filtered when loaded and decorations are correctly applied.
        """
        pc_none = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=False, use_ring=False, use_lidar_index=False, lidar_indices=None, sample_apillar_lidar_rings=False)
        pc_intensity = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=True, use_ring=False, use_lidar_index=False, lidar_indices=None, sample_apillar_lidar_rings=False)
        pc_ring = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=False, use_ring=True, use_lidar_index=False, lidar_indices=None, sample_apillar_lidar_rings=False)
        pc_lidar = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=False, use_ring=False, use_lidar_index=True, lidar_indices=None, sample_apillar_lidar_rings=False)
        pc_all = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=True, use_ring=True, use_lidar_index=True, lidar_indices=None, sample_apillar_lidar_rings=False)
        pc_single_lidar = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=True, use_ring=True, use_lidar_index=True, lidar_indices=(0,), sample_apillar_lidar_rings=True)
        pc_all_0 = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=True, use_ring=True, use_lidar_index=True, lidar_indices=(0, 1, 2, 3, 4), sample_apillar_lidar_rings=True)
        pc_all_1 = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=True, use_ring=True, use_lidar_index=True, lidar_indices=None, sample_apillar_lidar_rings=True)
        pc_all_no_sample_apillar = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=True, use_ring=True, use_lidar_index=True, lidar_indices=None, sample_apillar_lidar_rings=False)
        pc_sample_apillar_0 = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=True, use_ring=True, use_lidar_index=True, lidar_indices=(0, 3, 4), sample_apillar_lidar_rings=True)
        pc_sample_apillar_1 = prepare_pointcloud_points(self.lidar_pc.load(self.db), use_intensity=True, use_ring=True, use_lidar_index=True, lidar_indices=(0, 3, 4), sample_apillar_lidar_rings=False)
        pt_cloud = self.lidar_pc.load(self.db)
        self.assertEqual(pc_none.points.shape[0], 3)
        self.assertEqual(pc_intensity.points.shape[0], 4)
        self.assertEqual(pc_ring.points.shape[0], 4)
        self.assertEqual(pc_lidar.points.shape[0], 4)
        self.assertEqual(pc_all.points.shape[0], 6)
        self.assertTrue(pt_cloud.nbr_points() > pc_single_lidar.nbr_points())
        self.assertTrue((pc_all_0.points == pc_all_1.points).all())
        self.assertTrue((pc_single_lidar.points[5] == 0).all())
        self.assertTrue(np.isin(pc_all_0.points[5], [0, 1, 2, 3, 4]).all())
        self.assertTrue(np.array_equal(pc_sample_apillar_0.points, pc_sample_apillar_1.points))
        self.assertTrue(pc_all_no_sample_apillar.points[:, np.logical_or(pc_all_no_sample_apillar.points[5] == 1, pc_all_no_sample_apillar.points[5] == 2)].shape[1] > pc_all_0.points[:, np.logical_or(pc_all_0.points[5] == 1, pc_all_0.points[5] == 2)].shape[1])

class TestNuPlanDBLidarMethods(unittest.TestCase):
    """Tests for NuPlanDBLidarMethods (Helper methods for interacting with NuPlanDB's lidar samples)."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()

    def test_get_past_future_sweep(self) -> None:
        """
        Go N sweeps back and N sweeps forth and see if we are back at the original.
        """
        for sweep_idx in range(-10, 10):
            sweep_lidarpc_rec = _get_past_future_sweep(self.lidar_pc, sweep_idx)
            if sweep_lidarpc_rec is None:
                continue
            return_lidarpc_rec = _get_past_future_sweep(sweep_lidarpc_rec, -sweep_idx)
            self.assertEqual(self.lidar_pc, return_lidarpc_rec)

    def test_load_pointcloud_from_pc(self) -> None:
        """
        Test loading of point cloud from LidarPc based on distance, data shape, map filtering and timestmap.
        """
        min_dist = 0.9
        max_dist = 50.0
        pc = load_pointcloud_from_pc(nuplandb=self.db, token=self.lidar_pc.token, nsweeps=1, max_distance=max_dist, min_distance=min_dist, use_intensity=False, use_ring=False, use_lidar_index=False)
        pc_intensity = load_pointcloud_from_pc(nuplandb=self.db, token=self.lidar_pc.token, nsweeps=1, max_distance=max_dist, min_distance=min_dist, use_intensity=True, use_ring=False, use_lidar_index=False)
        pc_ring = load_pointcloud_from_pc(nuplandb=self.db, token=self.lidar_pc.token, nsweeps=1, max_distance=max_dist, min_distance=min_dist, use_intensity=False, use_ring=True, use_lidar_index=False)
        pc_lidar_index = load_pointcloud_from_pc(nuplandb=self.db, token=self.lidar_pc.token, nsweeps=1, max_distance=max_dist, min_distance=min_dist, use_intensity=False, use_ring=False, use_lidar_index=True)
        pc_multiple_sweeps = load_pointcloud_from_pc(nuplandb=self.db, token=self.lidar_pc.token, nsweeps=3, max_distance=max_dist, min_distance=min_dist, use_intensity=True, use_ring=False, use_lidar_index=False)
        pc_multiple_sweeps_new_format = load_pointcloud_from_pc(nuplandb=self.db, token=self.lidar_pc.token, nsweeps=list(range(-3 + 1, 0 + 1)), max_distance=max_dist, min_distance=min_dist, use_intensity=True, use_ring=False, use_lidar_index=False)
        pc_map_filtered_random = load_pointcloud_from_pc(nuplandb=self.db, token=self.lidar_pc.token, nsweeps=1, max_distance=max_dist, min_distance=min_dist, use_intensity=False, use_ring=False, use_lidar_index=False)
        pc_past_future = load_pointcloud_from_pc(nuplandb=self.db, token=self.lidar_pc.token, nsweeps=[-2, 0, 1], max_distance=max_dist, min_distance=min_dist, use_intensity=False, use_ring=False, use_lidar_index=False)
        pc_dist_from_orig = np.linalg.norm(pc.points[:2, :], axis=0)
        pc_multiple_sweeps_dist_from_orig = np.linalg.norm(pc_multiple_sweeps.points[:2, :], axis=0)
        self.assertEqual(pc.points.shape[0], 4)
        self.assertEqual(pc_map_filtered_random.points.shape[0], 4)
        self.assertEqual(pc_intensity.points.shape[0], 5)
        self.assertEqual(pc_ring.points.shape[0], 5)
        self.assertEqual(pc_lidar_index.points.shape[0], 5)
        self.assertEqual(pc_multiple_sweeps.points.shape[0], 5)
        self.assertTrue((pc_dist_from_orig >= min_dist).all() and (pc_dist_from_orig <= max_dist).all())
        self.assertTrue((pc_multiple_sweeps_dist_from_orig <= max_dist).all())
        self.assertTrue(pc_multiple_sweeps.points.shape[1] >= pc.points.shape[1])
        self.assertTrue(pc_map_filtered_random.points.shape[1] <= pc.points.shape[1])
        timestamps = np.unique(pc_past_future.points[3, :])
        past_timestamp = (self.lidar_pc.timestamp - self.lidar_pc.prev.prev.timestamp) / 1000000.0
        future_timestamp = (self.lidar_pc.timestamp - self.lidar_pc.next.timestamp) / 1000000.0
        self.assertAlmostEqual(past_timestamp, timestamps[2])
        self.assertAlmostEqual(0, timestamps[1])
        self.assertAlmostEqual(future_timestamp, timestamps[0])
        self.assertTrue(len(timestamps) == 3)
        self.assertTrue(np.all(pc_multiple_sweeps.points == pc_multiple_sweeps_new_format.points))

class TestLoadBoxes(unittest.TestCase):
    """Tests for get_boxes() and get_future_box_sequence()"""

    def setUp(self) -> None:
        """Set up the test case."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc()
        self.future_horizon_len_s = 1
        self.future_interval_s = 0.05

    def test_can_run_get_future_box_sequence(self) -> None:
        """Test get future box sequence."""
        get_future_box_sequence(lidar_pcs=[self.lidar_pc, self.lidar_pc.next], frame=Frame.VEHICLE, future_horizon_len_s=self.future_horizon_len_s, trans_matrix_ego=self.lidar_pc.ego_pose.trans_matrix_inv, future_interval_s=self.future_interval_s)

    def test_pack_future_boxes(self) -> None:
        """Test pack future boxes."""
        track_token_2_box_sequence = get_future_box_sequence(lidar_pcs=[self.lidar_pc, self.lidar_pc.next], frame=Frame.VEHICLE, future_horizon_len_s=self.future_horizon_len_s, trans_matrix_ego=self.lidar_pc.ego_pose.trans_matrix_inv, future_interval_s=self.future_interval_s)
        boxes_with_futures = pack_future_boxes(track_token_2_box_sequence=track_token_2_box_sequence, future_horizon_len_s=self.future_horizon_len_s, future_interval_s=self.future_interval_s)
        for box in boxes_with_futures:
            for horizon_idx, horizon_s in enumerate(box.get_all_future_horizons_s()):
                future_center = box.get_future_center_at_horizon(horizon_s)
                future_orientation = box.get_future_orientation_at_horizon(horizon_s)
                self.assertTrue(box.track_token is not None)
                expected_future_box = track_token_2_box_sequence[box.track_token][horizon_idx + 1]
                if expected_future_box is None:
                    np.testing.assert_array_equal(future_center, [np.nan, np.nan, np.nan])
                    self.assertEqual(future_orientation, None)
                else:
                    np.testing.assert_array_equal(expected_future_box.center, future_center)
                    self.assertEqual(expected_future_box.orientation, future_orientation)

    def test_load_boxes_from_lidarpc(self) -> None:
        """Test load all boxes from a lidar pc."""
        boxes = load_boxes_from_lidarpc(self.db, self.lidar_pc, ['pedestrian', 'vehicle'], False, 80.04, self.future_horizon_len_s, self.future_interval_s, {'pedestrian': 0, 'vehicle': 1})
        self.assertSetEqual({'pedestrian', 'vehicle'}, set(boxes.keys()))
        self.assertEqual(len(boxes['pedestrian']), 70)
        self.assertEqual(len(boxes['vehicle']), 29)

class TestGetFutureEgoTrajectory(unittest.TestCase):
    """Test getting future ego trajectory."""

    def setUp(self) -> None:
        """Set up test case."""
        self.lidar_pc = get_test_nuplan_lidarpc()
        self.future_lidarpc_recs: List[LidarPc] = [self.lidar_pc]
        while len(self.future_lidarpc_recs) < 200:
            self.future_lidarpc_recs.append(self.future_lidarpc_recs[-1].next)
        self.future_ego_poses = [rec.ego_pose for rec in self.future_lidarpc_recs]

    def test_get_future_ego_trajectory(self) -> None:
        """Test getting future ego trajectory."""
        future_ego_traj = get_future_ego_trajectory(self.lidar_pc, self.future_ego_poses, np.eye(4), 5.0, 0.5)
        self.assertEqual(future_ego_traj[0, 3], self.lidar_pc.ego_pose.timestamp)
        self.assertEqual(len(future_ego_traj), 11)
        self.assertLessEqual(abs((future_ego_traj[-1, 3] - future_ego_traj[0, 3]) / 1000000.0 - 5.0), 0.5)

    def test_get_future_ego_trajectory_not_enough(self) -> None:
        """Test getting future ego trajectory when there are not enough ego poses."""
        future_ego_traj = get_future_ego_trajectory(self.lidar_pc, self.future_ego_poses[:50], np.eye(4), 5.0, 0.5)
        self.assertEqual(future_ego_traj[0, 3], self.lidar_pc.ego_pose.timestamp)
        self.assertEqual(len(future_ego_traj), 11)
        np.testing.assert_equal(future_ego_traj[-1, :], [np.nan, np.nan, np.nan, np.nan])

class TestRenderOnMap(unittest.TestCase):
    """Test rendering on map."""

    def setUp(self) -> None:
        """Set up test case."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()
        self.future_lidarpc_recs: List[LidarPc] = [self.lidar_pc]
        while len(self.future_lidarpc_recs) < 200:
            self.future_lidarpc_recs.append(self.future_lidarpc_recs[-1].next)
        self.future_ego_poses = [rec.ego_pose for rec in self.future_lidarpc_recs]

    def test_render_on_map(self) -> None:
        """Test render on map."""
        render_on_map(self.lidar_pc, self.db, self.lidar_pc.boxes(), self.future_ego_poses, render_boxes_with_velocity=True, render_map_raster=False, render_vector_map=True, with_random_color=True, render_future_ego_poses=True)

class TestEgoPose(unittest.TestCase):
    """Tests the EgoPose class"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.ego_pose = get_test_nuplan_egopose()

    @patch('nuplan.database.nuplan_db_orm.ego_pose.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the _session property"""
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock
        result = self.ego_pose._session
        inspect_mock.assert_called_once_with(self.ego_pose)
        self.assertEqual(result, session_mock)

    @patch('nuplan.database.nuplan_db_orm.ego_pose.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the __repr__ method"""
        result = self.ego_pose.__repr__()
        simple_repr_mock.assert_called_once_with(self.ego_pose)
        self.assertEqual(result, simple_repr_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.ego_pose.Quaternion', autospec=True)
    def test_quaternion(self, quaternion_mock: Mock) -> None:
        """Tests the quaternion method"""
        result = self.ego_pose.quaternion
        quaternion_mock.assert_called_once_with(self.ego_pose.qw, self.ego_pose.qx, self.ego_pose.qy, self.ego_pose.qz)
        self.assertEqual(result, quaternion_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.ego_pose.np.array', autospec=True)
    def test_translation_np(self, np_array_mock: Mock) -> None:
        """Tests the translation_np method"""
        result = self.ego_pose.translation_np
        np_array_mock.assert_called_with([self.ego_pose.x, self.ego_pose.y, self.ego_pose.z])
        self.assertEqual(result, np_array_mock.return_value)

    def test_trans_matrix_and_inv(self) -> None:
        """Tests the transformation matrix and it's inverse method"""
        trans_matrix = self.ego_pose.trans_matrix
        trans_matrix_inv = self.ego_pose.trans_matrix_inv
        result = np.matmul(trans_matrix, trans_matrix_inv)
        np.testing.assert_allclose(result, np.identity(4), atol=0.001)

    def test_rotate_2d_points2d_to_ego_vehicle_frame(self) -> None:
        """Tests the rotate_2d_points2d_to_ego_vehicle_frame method"""
        points2d: npt.NDArray[np.float32] = np.ones([1, 2], dtype=np.float32)
        result = self.ego_pose.rotate_2d_points2d_to_ego_vehicle_frame(points2d)
        self.assertEqual(result.ndim, 2)

    def test_get_map_crop_dimensions(self) -> None:
        """
        Test that map crop method produces map of the correct dimensions.
        Test time: 10.569s
        """
        xrange = (-60, 60)
        yrange = (-60, 60)
        rotate_face_up = False
        map_layer_description = 'intensity'
        map_layer_precision = 0.1
        map_scale = 1 / map_layer_precision
        num_samples = 10
        db = get_test_nuplan_db()
        selected_indices = random.sample(list(range(len(db.ego_pose))), num_samples)
        expected_dimensions = ((xrange[1] - xrange[0]) * map_scale, (yrange[1] - yrange[0]) * map_scale)
        ego_pose_list = db.ego_pose
        for i in selected_indices:
            current_ego_pose = ego_pose_list[i]
            if current_ego_pose.lidar_pc is None:
                continue
            map_crop = current_ego_pose.get_map_crop(maps_db=db.maps_db, xrange=xrange, yrange=yrange, map_layer_name=map_layer_description, rotate_face_up=rotate_face_up)
            self.assertTrue(map_crop[0] is not None)
            self.assertEqual(expected_dimensions, map_crop[0].shape, f'Dimensions failed at ego pose index {i}')

    def test_get_vector_map(self) -> None:
        """Tests the get vector map method"""
        xrange = (-60, 60)
        yrange = (-60, 60)
        db = get_test_nuplan_db()
        num_samples = 10
        selected_indices = random.sample(list(range(len(db.ego_pose))), num_samples)
        ego_pose_list = db.ego_pose
        for i in selected_indices:
            current_ego_pose = ego_pose_list[i]
            if current_ego_pose.lidar_pc is None:
                continue
            result = current_ego_pose.get_vector_map(db.maps_db, xrange, yrange)
            self.assertIsNotNone(result)

@lru_cache(maxsize=1)
def get_test_nuplan_egopose() -> EgoPose:
    """Get a nuPlan egopose object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.ego_pose[DEFAULT_TEST_EGO_POSE_INDEX]

class TestNuPlanDB(unittest.TestCase):
    """Test main nuPlan database class."""

    def setUp(self) -> None:
        """Set up test case."""
        self.db = get_test_nuplan_db()
        self.db.add_ref()

    def test_pickle(self) -> None:
        """Test dumping and loading the object through pickle."""
        db_binary = pickle.dumps(self.db)
        re_db: NuPlanDB = pickle.loads(db_binary)
        self.assertEqual(self.db.data_root, re_db.data_root)
        self.assertEqual(self.db.name, re_db.name)
        self.assertEqual(self.db._verbose, re_db._verbose)

    def test_table_getters(self) -> None:
        """Test the table getters."""
        self.assertTrue(isinstance(self.db.category, Table))
        self.assertTrue(isinstance(self.db.camera, Table))
        self.assertTrue(isinstance(self.db.lidar, Table))
        self.assertTrue(isinstance(self.db.image, Table))
        self.assertTrue(isinstance(self.db.lidar_pc, Table))
        self.assertTrue(isinstance(self.db.lidar_box, Table))
        self.assertTrue(isinstance(self.db.track, Table))
        self.assertTrue(isinstance(self.db.scene, Table))
        self.assertTrue(isinstance(self.db.scenario_tag, Table))
        self.assertTrue(isinstance(self.db.traffic_light_status, Table))
        self.assertSetEqual(self.db.cam_channels, {'CAM_R2', 'CAM_R1', 'CAM_R0', 'CAM_F0', 'CAM_L2', 'CAM_L1', 'CAM_B0', 'CAM_L0'})
        self.assertSetEqual(self.db.lidar_channels, {'MergedPointCloud'})

    def test_nuplan_memory_usage(self) -> None:
        """
        Test that repeatedly creating and destroying nuplan DB objects does not cause memory leaks.
        """

        def spin_up_db() -> None:
            db = get_test_nuplan_db_nocache()
            db.remove_ref()
        starting_usage = 0
        ending_usage = 0
        num_iterations = 5
        hpy = guppy.hpy()
        hpy.setrelheap()
        for i in range(0, num_iterations, 1):
            spin_up_db()
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

class TestTrack(unittest.TestCase):
    """Test class Track"""

    def setUp(self) -> None:
        """
        Initializes a test Track
        """
        self.track = get_test_nuplan_track()

    @patch('nuplan.database.nuplan_db_orm.track.inspect', autospec=True)
    def test_session(self, inspect: Mock) -> None:
        """
        Tests _session method
        """
        mock_session = PropertyMock()
        inspect.return_value = Mock()
        inspect.return_value.session = mock_session
        result = self.track._session()
        inspect.assert_called_once_with(self.track)
        mock_session.assert_called_once()
        self.assertEqual(result, mock_session.return_value)

    @patch('nuplan.database.nuplan_db_orm.track.simple_repr', autospec=True)
    def test_repr(self, simple_repr: Mock) -> None:
        """
        Tests string representation
        """
        result = self.track.__repr__()
        simple_repr.assert_called_once_with(self.track)
        self.assertEqual(result, simple_repr.return_value)

    def test_nbr_lidar_boxes(self) -> None:
        """
        Tests property - number of boxes along the track.
        """
        result = self.track.nbr_lidar_boxes
        self.assertGreater(result, 0)
        self.assertIsInstance(result, int)

    def test_first_last_lidar_box(self) -> None:
        """
        Tests properties - first and last lidar box along the track.
        """
        first_lidar_box = self.track.first_lidar_box
        last_lidar_box = self.track.last_lidar_box
        self.assertGreaterEqual(last_lidar_box.timestamp, first_lidar_box.timestamp)

    @patch('nuplan.database.nuplan_db_orm.track.Track.first_lidar_box', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.track.Track.last_lidar_box', autospec=True)
    def test_duration(self, mock_last_box: Mock, mock_first_box: Mock) -> None:
        """
        Tests property - duration of Track.
        """
        mock_first_box.timestamp = 1000
        mock_last_box.timestamp = 5000
        result = self.track.duration
        self.assertEqual(result, 4000)

    def test_distances_to_ego(self) -> None:
        """
        Tests property - distances of all boxes in the track from ego vehicle.
        """
        result = self.track.distances_to_ego
        self.assertEqual(len(result), self.track.nbr_lidar_boxes)

    def test_min_max_distance_to_ego(self) -> None:
        """
        Tests two properties - min and max distance to ego
        """
        min_result = self.track.min_distance_to_ego
        max_result = self.track.max_distance_to_ego
        self.assertGreaterEqual(max_result, min_result)

@lru_cache(maxsize=1)
def get_test_nuplan_track() -> Track:
    """Get a nuPlan track object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.track[DEFAULT_TEST_TRACK_INDEX]

class TestRendering(unittest.TestCase):
    """Some of these tests don't assert anything, but they will fail if the rendering code throws an exception."""

    def setUp(self) -> None:
        """Set up"""
        self.db = get_test_nuplan_db()
        self.lidar_box = get_test_nuplan_lidar_box()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()

    def test_closest_image(self) -> None:
        """Tests the closest_image method"""
        result = lidar_pc_closest_image(self.lidar_pc)
        self.assertNotEqual(len(result), 0)

    def test_lidar_pc_render(self) -> None:
        """Test Lidar PC render."""
        self.lidar_pc.render(self.db)

    @patch('nuplan.database.nuplan_db_orm.rendering_utils.Axes.imshow', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.image.Image.load_as', autospec=True)
    def test_lidar_box_render_img_found(self, loadas_mock: Mock, axes_mock: Mock) -> None:
        """Test Lidar Box render when the image is found"""
        render_lidar_box(self.lidar_box, self.db)
        loadas_mock.assert_called_once()
        axes_mock.assert_called_once()

    @patch('nuplan.database.nuplan_db_orm.rendering_utils.box_in_image', autospec=True)
    def test_lidar_box_render_img_not_found(self, box_in_image_mock: Mock) -> None:
        """Test Lidar Box render in the event that the image is not found"""
        box_in_image_mock.return_value = False
        with self.assertRaises(AssertionError):
            render_lidar_box(self.lidar_box, self.db)

class TestImage(unittest.TestCase):
    """Test class Image"""

    def setUp(self) -> None:
        """
        Initializes a test Image
        """
        self.db = get_test_nuplan_db()
        self.image = get_test_nuplan_image()

    @patch('nuplan.database.nuplan_db_orm.image.inspect', autospec=True)
    def test_session(self, inspect: Mock) -> None:
        """
        Tests _session method
        """
        mock_session = PropertyMock()
        inspect.return_value = Mock()
        inspect.return_value.session = mock_session
        result = self.image._session()
        inspect.assert_called_once_with(self.image)
        mock_session.assert_called_once()
        self.assertEqual(result, mock_session.return_value)

    @patch('nuplan.database.nuplan_db_orm.image.simple_repr', autospec=True)
    def test_repr(self, simple_repr: Mock) -> None:
        """
        Tests string representation
        """
        result = self.image.__repr__()
        simple_repr.assert_called_once_with(self.image)
        self.assertEqual(result, simple_repr.return_value)

    def test_log(self) -> None:
        """
        Tests property log
        """
        log = self.image.log
        self.assertEqual(log, self.image.camera.log)

    @patch('nuplan.database.nuplan_db_orm.image.func')
    @patch('nuplan.database.nuplan_db_orm.image.Image._session')
    def test_lidar_pc(self, mock_session: Mock, mock_func: Mock) -> None:
        """
        Tests property lidar_pc
        """
        mock_query = mock_session.query
        mock_lidar_pc = mock_query.return_value
        mock_lidar_pc_ordered = mock_lidar_pc.order_by.return_value
        mock_first = mock_lidar_pc_ordered.first.return_value
        result = self.image.lidar_pc
        mock_query.assert_called_once_with(LidarPc)
        self.assertTrue(mock_func.abs.call_args[0][0].compare(LidarPc.timestamp - self.image.timestamp))
        mock_lidar_pc.order_by.assert_called_once_with(mock_func.abs.return_value)
        mock_lidar_pc_ordered.first.assert_called_once()
        self.assertEqual(result, mock_first)

    @patch('nuplan.database.nuplan_db_orm.image.Image.lidar_pc')
    def test_lidar_boxes(self, mock_lidar_pc: Mock) -> None:
        """
        Tests property lidar_boxes
        """
        result = self.image.lidar_boxes
        self.assertEqual(result, mock_lidar_pc.lidar_boxes)

    @patch('nuplan.database.nuplan_db_orm.image.Image.lidar_pc')
    def test_scene(self, mock_lidar_pc: Mock) -> None:
        """
        Tests property scene
        """
        result = self.image.scene
        self.assertEqual(result, mock_lidar_pc.scene)

    @patch('nuplan.database.nuplan_db_orm.image.PIL.Image.open')
    @patch('nuplan.database.nuplan_db_orm.image.Image.load_bytes_jpg')
    def test_load_as_pil(self, mock_load_bytes: Mock, mock_pil_open: Mock) -> None:
        """
        Tests load_as with PIL image type
        """
        mock_db = Mock()
        img = self.image.load_as(mock_db, 'pil')
        mock_load_bytes.assert_called_once_with(mock_db)
        mock_pil_open.assert_called_once_with(mock_load_bytes.return_value)
        self.assertEqual(img, mock_pil_open.return_value)

    @patch('nuplan.database.nuplan_db_orm.image.np.array')
    @patch('nuplan.database.nuplan_db_orm.image.PIL.Image.open')
    @patch('nuplan.database.nuplan_db_orm.image.Image.load_bytes_jpg')
    def test_load_as_np(self, mock_load_bytes: Mock, mock_pil_open: Mock, mock_np_array: Mock) -> None:
        """
        Tests load_as with numpy array image type
        """
        mock_db = Mock()
        img = self.image.load_as(mock_db, 'np')
        mock_load_bytes.assert_called_once_with(mock_db)
        mock_pil_open.assert_called_once_with(mock_load_bytes.return_value)
        mock_np_array.assert_called_once_with(mock_pil_open.return_value)
        self.assertEqual(img, mock_np_array.return_value)

    @patch('nuplan.database.nuplan_db_orm.image.cv2.COLOR_RGB2BGR')
    @patch('nuplan.database.nuplan_db_orm.image.cv2.cvtColor')
    @patch('nuplan.database.nuplan_db_orm.image.np.array')
    @patch('nuplan.database.nuplan_db_orm.image.PIL.Image.open')
    @patch('nuplan.database.nuplan_db_orm.image.Image.load_bytes_jpg')
    def test_load_as_cv2(self, mock_load_bytes: Mock, mock_pil_open: Mock, mock_np_array: Mock, mock_cvtColor: Mock, mock_rgb2bgr: Mock) -> None:
        """
        Tests load_as with cv2 image type
        """
        mock_db = Mock()
        img = self.image.load_as(mock_db, 'cv2')
        mock_load_bytes.assert_called_once_with(mock_db)
        mock_pil_open.assert_called_once_with(mock_load_bytes.return_value)
        mock_np_array.assert_called_once_with(mock_pil_open.return_value)
        mock_cvtColor.assert_called_once_with(mock_np_array.return_value, mock_rgb2bgr)
        self.assertEqual(img, mock_cvtColor.return_value)

    def test_load_as_invalid(self) -> None:
        """
        Tests load_as with invalid image type
        """
        mock_db = Mock()
        with self.assertRaises(AssertionError):
            self.image.load_as(mock_db, 'invalid')

    def test_filename(self) -> None:
        """
        Tests property filename
        """
        filename = self.image.filename
        self.assertEqual(filename, self.image.filename_jpg)

    @patch('nuplan.database.nuplan_db_orm.image.osp.join')
    @patch('nuplan.database.nuplan_db_orm.image.Image.filename')
    def test_load_bytes_jpg(self, mock_filename: Mock, mock_osp_join: Mock) -> None:
        """
        Tests method to load bytes of the jpg data db.load_blob(osp.join("sensor_blobs", self.filename))
        """
        mock_load_blob = Mock()
        mock_db = Mock(load_blob=mock_load_blob)
        result = self.image.load_bytes_jpg(mock_db)
        mock_osp_join.assert_called_once_with('sensor_blobs', mock_filename)
        mock_load_blob.assert_called_once_with(mock_osp_join.return_value)
        self.assertEqual(result, mock_load_blob.return_value)

    @patch('nuplan.database.nuplan_db_orm.image.osp.join')
    def test_path(self, mock_osp_join: Mock) -> None:
        """
        Tests image path based on DB data root
        """
        mock_db = Mock(data_root='data_root')
        path = self.image.path(mock_db)
        mock_osp_join.assert_called_once_with('data_root', self.image.filename)
        self.assertEqual(path, mock_osp_join.return_value)

    @patch('nuplan.database.nuplan_db_orm.image.get_boxes')
    @patch('nuplan.database.nuplan_db_orm.image.Image.camera')
    @patch('nuplan.database.nuplan_db_orm.image.Image.ego_pose')
    def test_boxes(self, mock_egopose: Mock, mock_camera: Mock, mock_get_boxes: Mock) -> None:
        """
        Test loading of boxes associated with this Image
        """
        boxes = self.image.boxes('Frame')
        mock_get_boxes.assert_called_once_with(self.image, 'Frame', mock_egopose.trans_matrix_inv, mock_camera.trans_matrix_inv)
        self.assertEqual(boxes, mock_get_boxes.return_value)

    def test_future_ego_poses(self) -> None:
        """
        Test method to get n future poses
        """
        n_ego_poses = 4
        future_ego_poses = self.image.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.image.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(ego_pose.timestamp, future_ego_poses[i].timestamp, 'Timestamps of current EgoPose must be less that future EgoPoses.')

    def test_past_ego_poses(self) -> None:
        """
        Test method to get n past poses
        """
        n_ego_poses = 4
        past_ego_poses = self.image.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = self.image.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(ego_pose.timestamp, past_ego_poses[i].timestamp, 'Timestamps of current EgoPose must be greater than past EgoPoses ')

    def test_invalid_ego_poses(self) -> None:
        """
        Test method to get poses, with invalid inputs
        """
        with self.assertRaises(ValueError):
            self.image.future_or_past_ego_poses(number=1, mode='n_poses', direction='invalid')
        with self.assertRaises(NotImplementedError):
            self.image.future_or_past_ego_poses(number=1, mode='invalid', direction='prev')

    @patch('nuplan.database.nuplan_db_orm.image.Image.boxes')
    @patch('nuplan.database.nuplan_db_orm.image.box_in_image')
    @patch('nuplan.database.nuplan_db_orm.image.Image.load_as', autospec=True)
    def test_render(self, mock_load: Mock, mock_box_in_image: Mock, mock_boxes: Mock) -> None:
        """
        Test render method
        """
        mock_ax = Mock(spec=Axes)
        mock_box = Mock(spec=Box3D, token='token')
        mock_box.render = Mock()
        mock_boxes.return_value = [mock_box]
        mock_db = MagicMock()
        mock_db.lidar_box = MagicMock()
        mock_box_in_image.return_value = True
        self.image.render(mock_db, with_3d_anns=True, box_vis_level='box_vis_level', ax=mock_ax)
        mock_boxes.assert_called_once()
        self.assertEqual(mock_box_in_image.call_args.args[0], mock_box)
        np.testing.assert_array_equal(mock_box_in_image.call_args.args[1], self.image.camera.intrinsic_np)
        np.testing.assert_array_equal(mock_box_in_image.call_args.args[2], (self.image.camera.width, self.image.camera.height))
        self.assertEqual(mock_box_in_image.call_args.kwargs, {'vis_level': 'box_vis_level'})
        mock_box.render.assert_called_once()

@lru_cache(maxsize=1)
def get_test_nuplan_image() -> Image:
    """Get a nuPlan image object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.image[DEFAULT_TEST_IMAGE_WITH_BLOB_TOKEN]

class TestLidar(unittest.TestCase):
    """Test class Lidar"""

    def setUp(self) -> None:
        """
        Initializes a test Lidar
        """
        self.lidar = get_test_nuplan_lidar()

    @patch('nuplan.database.nuplan_db_orm.lidar.inspect', autospec=True)
    def test_session(self, inspect: Mock) -> None:
        """
        Tests _session method
        """
        mock_session = PropertyMock()
        inspect.return_value = Mock()
        inspect.return_value.session = mock_session
        result = self.lidar._session()
        inspect.assert_called_once_with(self.lidar)
        mock_session.assert_called_once()
        self.assertEqual(result, mock_session.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar.simple_repr', autospec=True)
    def test_repr(self, simple_repr: Mock) -> None:
        """
        Tests string representation
        """
        result = self.lidar.__repr__()
        simple_repr.assert_called_once_with(self.lidar)
        self.assertEqual(result, simple_repr.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar.np.array', autospec=True)
    def test_translation_np(self, np_array: Mock) -> None:
        """
        Test property - translation.
        """
        result = self.lidar.translation_np
        np_array.assert_called_once_with(self.lidar.translation)
        self.assertEqual(result, np_array.return_value)

    def test_quaternion(self) -> None:
        """
        Test property - rotation in quaternion.
        """
        result = self.lidar.quaternion
        np.testing.assert_array_equal(self.lidar.rotation, result.elements)

    def test_trans_matrix_and_inv(self) -> None:
        """
        Test two properties - transformation matrix and its inverse.
        """
        trans_mat = self.lidar.trans_matrix
        inv_trans_mat = self.lidar.trans_matrix_inv
        np.testing.assert_allclose(trans_mat @ inv_trans_mat, np.eye(4), atol=0.001)

@lru_cache(maxsize=1)
def get_test_nuplan_lidar() -> Lidar:
    """Get a nuPlan lidar object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar[DEFAULT_TEST_LIDAR_INDEX]

class TestNuplan(unittest.TestCase):
    """Test Nuplan DB."""

    def test_nuplan(self) -> None:
        """
        Check whether the nuPlan DB can be loaded without errors.
        """
        db = get_test_nuplan_db()
        self.assertIsNotNone(db)

