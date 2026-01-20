# Cluster 32

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

def load_pointcloud_from_pc(nuplandb: NuPlanDB, token: str, nsweeps: Union[int, List[int]], max_distance: float, min_distance: float, drivable_area: bool=False, map_dilation: float=0.0, use_intensity: bool=True, use_ring: bool=False, use_lidar_index: bool=False, lidar_indices: Optional[Tuple[int, ...]]=None, sample_apillar_lidar_rings: bool=False, sweep_map: str='time_lag') -> LidarPointCloud:
    """
    Loads one or more sweeps of a LIDAR pointcloud from the database using a SampleData record of NuPlanDB.
    :param nuplandb: The multimodal database used in this dataset.
    :param token: Token for the Lidar pointcloud.
    :param nsweeps: The number of past LIDAR sweeps used in the model.
        Alternatively, it is possible to provide a list of relative sweep indices, with:
        - Negative numbers corresponding to past sweeps.
        - 0 corresponding to the present sweep.
        - Positive numbers corresponding to future sweeps.
    :param max_distance: Radius outside which the points will be removed. Helps speed up caching and building the
        GT database.
    :param min_distance: Radius below which near points will be removed. This is usually recommended by the lidar
        manufacturer.
    :param drivable_area: Whether the pointcloud should be filtered based on drivable_area mask.
    :param map_dilation: Map dilation factor in meters.
    :param use_intensity: See prepare_pointcloud_points documentation for details.
    :param use_ring: See prepare_pointcloud_points documentation for details.
    :param use_lidar_index: Whether to use lidar index as a decoration.
    :param lidar_indices: See prepare_pointcloud_points documentation for details.
    :param sample_apillar_lidar_rings: Whether you want to sample rings for the A-pillar lidars.
    :param sweep_map: What to append to the lidar points to give information about what sweep it belongs to.
        Options: 'time_lag' and 'sweep_idx'.
    :return: The pointcloud.
    """
    assert sweep_map in ['time_lag', 'sweep_idx']
    if isinstance(nsweeps, int):
        nsweeps = list(range(-nsweeps + 1, 0 + 1))
    elif isinstance(nsweeps, list):
        assert 0 in nsweeps, f'Error: Present sweep (0) must be included! nsweeps is: {nsweeps}'
    else:
        raise TypeError('Invalid nsweeps type: {}'.format(type(nsweeps)))
    assert sorted(nsweeps) == nsweeps, 'Error: nsweeps must be sorted in ascending order!'
    lidarpc_rec = nuplandb.lidar_pc[token]
    time_current = lidarpc_rec.timestamp
    if len(nsweeps) > 1:
        car_from_lidar = lidarpc_rec.lidar.trans_matrix
        car_from_global = lidarpc_rec.ego_pose.trans_matrix_inv
        lidar_from_car = lidarpc_rec.lidar.trans_matrix_inv
    init = False
    for rel_sweep_idx, sweep_idx in enumerate(nsweeps):
        sweep_lidarpc_rec = _get_past_future_sweep(lidarpc_rec, sweep_idx)
        if sweep_lidarpc_rec is None:
            continue
        sweep_pc = sweep_lidarpc_rec.load(nuplandb)
        sweep_pc = prepare_pointcloud_points(sweep_pc, use_intensity=use_intensity, use_ring=use_ring, use_lidar_index=use_lidar_index, lidar_indices=lidar_indices, sample_apillar_lidar_rings=sample_apillar_lidar_rings)
        sweep_pc.remove_close(min_distance)
        if sweep_idx != 0:
            sweep_pose_rec = sweep_lidarpc_rec.ego_pose
            global_from_car = sweep_pose_rec.trans_matrix
            trans_matrix = reduce(np.dot, [lidar_from_car, car_from_global, global_from_car, car_from_lidar])
            sweep_pc.transform(trans_matrix)
        sweep_pc.radius_filter(max_distance)
        if sweep_map == 'sweep_idx':
            rel_sweep_idx_pixor = np.array(rel_sweep_idx, dtype=np.float32) + 1
            assert rel_sweep_idx_pixor > 0
            sweep_vector = rel_sweep_idx_pixor * np.ones((1, sweep_pc.nbr_points()), dtype=np.float32)
        elif sweep_map == 'time_lag':
            time_lag = time_current - sweep_lidarpc_rec.timestamp if sweep_idx != 0 else 0
            sweep_vector = 1e-06 * time_lag * np.ones((1, sweep_pc.nbr_points()), dtype=np.float32)
        else:
            raise ValueError('Cannot recognize sweep_map type: {}'.format(sweep_map))
        sweep_pc.points = np.concatenate((sweep_pc.points, sweep_vector), axis=0)
        if not init:
            pc: LidarPointCloud = sweep_pc
            init = True
        else:
            pc.points = np.hstack((pc.points, sweep_pc.points))
    return pc

class TestLoadPointcloudFromSampledataUsingMocks(unittest.TestCase):
    """Test Loading PointCloud."""

    @unittest.mock.patch('nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points')
    def test_distance_filtering(self, prepare_pointcloud_points_mock: Mock) -> None:
        """
        Make sure close and far points are filtered properly.
        """
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(points=np.array([[0.1, -0.1, 10, -10, 1000, 1000], [0.2, -0.2, 20, 20, 2000, -2000]]))
        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(nuplandb, token='abc', nsweeps=1, max_distance=1000, min_distance=1)
        expected_points = np.array([[10, -10], [20, 20], [0, 0]], dtype=np.float32)
        self.assertTrue(np.allclose(loaded_pc.points, expected_points))

    @unittest.mock.patch('nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points')
    def test_3_sweeps(self, prepare_pointcloud_points_mock: Mock) -> None:
        """
        Make sure points and timestamps accumulate properly with multiple sweeps.
        """
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(points=np.array([[100, -100], [200, 200], [300, 300]]))
        mock_lidarpc_rec.prev.load.return_value = LidarPointCloud(points=np.array([[10, -10], [20, 20], [30, 30]]))
        mock_lidarpc_rec.prev.prev.load.return_value = LidarPointCloud(points=np.array([[1, -1], [2, 2], [3, 3]]))
        mock_lidarpc_rec.timestamp = 507
        mock_lidarpc_rec.prev.timestamp = 504
        mock_lidarpc_rec.prev.prev.timestamp = 500
        mock_lidarpc_rec.lidar.trans_matrix = np.eye(4)
        mock_lidarpc_rec.lidar.trans_matrix_inv = np.eye(4)
        mock_lidarpc_rec.ego_pose.trans_matrix_inv = np.eye(4)
        mock_lidarpc_rec.prev.ego_pose.trans_matrix = np.eye(4)
        mock_lidarpc_rec.prev.prev.ego_pose.trans_matrix = np.eye(4)
        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(nuplandb, token='abc', nsweeps=3, max_distance=1000, min_distance=0)
        expected_points = np.array([[1, -1, 10, -10, 100, -100], [2, 2, 20, 20, 200, 200], [3, 3, 30, 30, 300, 300], [7e-06, 7e-06, 3e-06, 3e-06, 0, 0]], dtype=np.float32)
        self.assertTrue(np.allclose(loaded_pc.points, expected_points))

    @unittest.mock.patch('nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points')
    def test_3_sweeps_past_future(self, prepare_pointcloud_points_mock: Mock) -> None:
        """
        Make sure points and timestamps accumulate properly with multiple sweeps, using past and future data.
        """
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(points=np.array([[100, -100], [200, 200], [300, 300]]))
        mock_lidarpc_rec.next.next.load.return_value = LidarPointCloud(points=np.array([[10, -10], [20, 20], [30, 30]]))
        mock_lidarpc_rec.prev.prev.load.return_value = LidarPointCloud(points=np.array([[1, -1], [2, 2], [3, 3]]))
        mock_lidarpc_rec.prev.prev.timestamp = 500
        mock_lidarpc_rec.timestamp = 504
        mock_lidarpc_rec.next.next.timestamp = 507
        mock_lidarpc_rec.lidar.trans_matrix = np.eye(4)
        mock_lidarpc_rec.lidar.trans_matrix_inv = np.eye(4)
        mock_lidarpc_rec.ego_pose.trans_matrix_inv = np.eye(4)
        mock_lidarpc_rec.prev.prev.ego_pose.trans_matrix = np.eye(4)
        mock_lidarpc_rec.next.next.ego_pose.trans_matrix = np.eye(4)
        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(nuplandb, token='abc', nsweeps=[-2, 0, 2], max_distance=1000, min_distance=0)
        expected_points = np.array([[1, -1, 100, -100, 10, -10], [2, 2, 200, 200, 20, 20], [3, 3, 300, 300, 30, 30], [4e-06, 4e-06, 0, 0, -3e-06, -3e-06]], dtype=np.float32)
        self.assertTrue(np.allclose(loaded_pc.points, expected_points))

    @unittest.mock.patch('nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points')
    def test_5_sweeps_moving_vehicle(self, prepare_pointcloud_points_mock: Mock) -> None:
        """Test accumulating sweeps with moving vehicle."""
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points
        point_111 = np.ones((3, 1), dtype=np.float32)
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(points=point_111)
        mock_lidarpc_rec.prev.load.return_value = LidarPointCloud(points=point_111)
        mock_lidarpc_rec.prev.prev.load.return_value = LidarPointCloud(points=point_111)
        mock_lidarpc_rec.prev.prev.prev.load.return_value = LidarPointCloud(points=point_111)
        mock_lidarpc_rec.prev.prev.prev.prev.load.return_value = LidarPointCloud(points=point_111)
        mock_lidarpc_rec.timestamp = 504
        mock_lidarpc_rec.prev.timestamp = 503
        mock_lidarpc_rec.prev.prev.timestamp = 502
        mock_lidarpc_rec.prev.prev.prev.timestamp = 501
        mock_lidarpc_rec.prev.prev.prev.prev.timestamp = 500
        mock_lidarpc_rec.lidar.trans_matrix = np.eye(4)
        mock_lidarpc_rec.lidar.trans_matrix_inv = np.eye(4)

        def addition_transform(x: float, y: float, z: float) -> npt.NDArray[np.float64]:
            """
            Create a 4 by 4 transformation matrix given translation.
            :return: <np.float: 4, 4>. The transformation matrix.
            """
            return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float32)
        mock_lidarpc_rec.ego_pose.trans_matrix_inv = np.eye(4)
        mock_lidarpc_rec.prev.ego_pose.trans_matrix = addition_transform(1, 2, 3)
        mock_lidarpc_rec.prev.prev.ego_pose.trans_matrix = addition_transform(2, 3, 4)
        mock_lidarpc_rec.prev.prev.prev.ego_pose.trans_matrix = addition_transform(3, 4, 5)
        mock_lidarpc_rec.prev.prev.prev.prev.ego_pose.trans_matrix = addition_transform(4, 5, 6)
        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(nuplandb, token='abc', nsweeps=5, max_distance=1000, min_distance=0)
        expected_points = np.array([[5, 4, 3, 2, 1], [6, 5, 4, 3, 1], [7, 6, 5, 4, 1], [4e-06, 3e-06, 2e-06, 1e-06, 0]], dtype=np.float32)
        self.assertTrue(np.allclose(loaded_pc.points, expected_points))

    @unittest.mock.patch('nuplan.database.nuplan_db_orm.utils.prepare_pointcloud_points')
    def test_coordinate_transforms(self, prepare_pointcloud_points_mock: Mock) -> None:
        """
        Make sure points and timestamps accumulate properly with multiple sweeps.
        """
        prepare_pointcloud_points_mock.side_effect = mock_prepare_pointcloud_points
        mock_lidarpc_rec = Mock()
        mock_lidarpc_rec.load.return_value = LidarPointCloud(points=np.array([[100], [200], [300]], dtype=np.float32))
        mock_lidarpc_rec.prev.load.return_value = LidarPointCloud(points=np.array([[10], [20], [30]], dtype=np.float32))
        mock_lidarpc_rec.prev.prev.load.return_value = LidarPointCloud(points=np.array([[1], [2], [3]], dtype=np.float32))
        mock_lidarpc_rec.timestamp = 507
        mock_lidarpc_rec.prev.timestamp = 504
        mock_lidarpc_rec.prev.prev.timestamp = 500
        mock_lidarpc_rec.lidar.trans_matrix = np.eye(4)
        mock_lidarpc_rec.lidar.trans_matrix_inv = np.eye(4)
        mock_lidarpc_rec.ego_pose.trans_matrix_inv = np.eye(4)
        mock_lidarpc_rec.prev.ego_pose.trans_matrix = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        mock_lidarpc_rec.prev.prev.ego_pose.trans_matrix = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        nuplandb = MagicMock()
        nuplandb.lidar_pc.__getitem__.return_value = mock_lidarpc_rec
        loaded_pc = load_pointcloud_from_pc(nuplandb, token='abc', nsweeps=3, max_distance=1000, min_distance=0, sweep_map='sweep_idx')
        expected_points = np.array([[2, -20, 100], [-1, 10, 200], [3, 30, 300], [1, 2, 3]], dtype=np.float32)
        self.assertTrue(np.allclose(loaded_pc.points, expected_points))

class TestPointCloud(unittest.TestCase):
    """Test Class for Point Cloud."""

    def test_load_pcd_bin_v1(self) -> None:
        """Testing if points in binary format v1 can be read."""
        pcd_expected = np.array([[3.5999999, -3.0999999, 0, 1, 0.5, -1], [1.0, -3.01, 10.0, 0.4, 10, -1], [4.5999999, -2.90001, -1.0, 0.1, 1.5, -1]], dtype=np.float32)
        file_path = tempfile.NamedTemporaryFile()
        with open(file_path.name, 'w+b'):
            for point in pcd_expected:
                file_path.write(struct.pack('5f', point[0], point[1], point[2], point[3], point[4]))
            _ = file_path.seek(0)
            pcd = LidarPointCloud.load_pcd_bin(file_path.name)
            assert np.all(pcd == pcd_expected.T)

    def test_load_pcd_bin_v2(self) -> None:
        """Testing if points in binary format v2 can be read."""
        pcd_expected = np.array([[3.5999999, -3.0999999, 0, 1, 0.5, -1], [1.0, -3.01, 10.0, 0.4, 10, -1], [4.5999999, -2.90001, -1.0, 0.1, 1.5, -1]], dtype=np.float32)
        file_path = tempfile.NamedTemporaryFile()
        with open(file_path.name, 'w+b'):
            for point in pcd_expected:
                file_path.write(struct.pack('6f', point[0], point[1], point[2], point[3], point[4], point[5]))
            _ = file_path.seek(0)
            pcd = LidarPointCloud.load_pcd_bin(file_path.name, 2)
            assert np.all(pcd == pcd_expected.T)

    def test_nbr_points(self) -> None:
        """Testing if the number of points in the pointcloud is returned."""
        test_pointcloud = np.array([[35, 35, 0, 0, 0], [20.0, 30.0, 2000, 0, 0], [30.0, 20.0, 0, 0, 0], [8.0, 8.0, 0, 0, 0], [0.0, 15.0, 10, 0, 0]])
        pc = LidarPointCloud(test_pointcloud.T)
        self.assertEqual(pc.nbr_points(), 5)

    def test_subsample(self) -> None:
        """Testing if the correct number of points are sampled given the ratio."""
        test_pointcloud = np.zeros((100, 5))
        pc = LidarPointCloud(test_pointcloud.T)
        pc.subsample(ratio=0.5)
        self.assertEqual(pc.nbr_points(), 50)
        pc.subsample(ratio=0.2)
        self.assertEqual(pc.nbr_points(), 10)
        pc.subsample(ratio=0.18)
        self.assertEqual(pc.nbr_points(), 1)

    def test_1d_array_input(self) -> None:
        """Testing if can do translate/rotate function from single point input array."""
        pc = LidarPointCloud(np.array([0, 0, 0, 0, 0]))
        test_translate = np.array([0, 0, 1])
        pc.translate(test_translate)
        assert_array_equal(pc.points[:, 0], np.array([0, 0, 1, 0, 0]))
        theta = np.pi
        test_rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]])
        pc.rotate(Quaternion(matrix=test_rot_matrix))
        self.assertAlmostEqual(pc.points[0, 0], 0)
        self.assertAlmostEqual(pc.points[1, 0], 0)
        self.assertAlmostEqual(pc.points[2, 0], -1)

    def test_remove_close(self) -> None:
        """Testing if points within a certain radius from origin (in bird view) are correctly removed."""
        test_pointcloud = np.array([[35, 35, 0, 0, 0], [20.0, 30.0, 2000, 0, 0], [30.0, 20.0, 0, 0, 0], [8.0, 8.0, 0, 0, 0], [0.0, 15.0, 10, 0, 0]])
        pc = LidarPointCloud(test_pointcloud.T)
        pc.remove_close(5)
        self.assertEqual(pc.nbr_points(), 5)
        pc.remove_close(12)
        self.assertEqual(pc.nbr_points(), 4)
        pc.remove_close(15)
        self.assertEqual(pc.nbr_points(), 4)
        pc.remove_close(36.1)
        self.assertEqual(pc.nbr_points(), 1)

    def test_radius_filter(self) -> None:
        """Testing if points within a certain radius from origin (in bird view) is correctly removed."""
        test_pointcloud = np.array([[35, 35, 0, 0, 0], [20.0, 30.0, 2000, 0, 0], [30.0, 20.0, 0, 0, 0], [8.0, 8.0, 0, 0, 0], [0.0, 15.0, 10, 0, 0]])
        pointcloud = LidarPointCloud(test_pointcloud.T)
        pc = pointcloud.copy()
        pc.radius_filter(5)
        self.assertEqual(pc.nbr_points(), 0)
        pc = pointcloud.copy()
        pc.radius_filter(12)
        self.assertEqual(pc.nbr_points(), 1)
        pc = pointcloud.copy()
        pc.radius_filter(15)
        self.assertEqual(pc.nbr_points(), 2)
        pc = pointcloud.copy()
        pc.radius_filter(36.1)
        self.assertEqual(pc.nbr_points(), 4)

    def test_scale(self) -> None:
        """Testing if the lidar xyz coordinates are scaled."""
        test_pointcloud = np.array([[35, 35, 0, 0, 0], [20.0, 30.0, 2000, 0, 0], [30.0, 20.0, 0, 0, 0], [8.0, 8.0, 0, 0, 0], [0.0, 15.0, 10, 0, 0]])
        test_pc = test_pointcloud.copy()
        pc = LidarPointCloud(test_pc.T)
        pc.scale((2, 2, 2))
        test_pc_scaled = test_pointcloud.copy()
        test_pc_scaled[:, 0:3] *= 2
        pc_scaled = LidarPointCloud(test_pc_scaled.T)
        self.assertEqual(pc, pc_scaled)

    def test_translate_simple(self) -> None:
        """Testing if points are translated correctly given a translate vector."""
        pc = LidarPointCloud(np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0, 0.0]]).T)
        test_translate = np.array([5.2, 10.4, 15.1])
        pc.translate(test_translate)
        assert_array_equal(pc.points[:, 0], np.array([5.2, 10.4, 15.1, 0, 0]))
        assert_array_equal(pc.points[:, 1], np.array([6.2, 12.4, 18.1, 0, 0]))

    def test_rotate_simple(self) -> None:
        """Testing if points are rotated correctly given a rotation matrix."""
        theta = np.pi / 4
        test_rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]])
        pc = LidarPointCloud(np.array([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]).T)
        pc.rotate(Quaternion(matrix=test_rot_matrix))
        self.assertAlmostEqual(pc.points[0, 0], 0)
        self.assertAlmostEqual(pc.points[1, 0], -1 / np.sqrt(2))
        self.assertAlmostEqual(pc.points[2, 0], 1 / np.sqrt(2))

    def test_copy(self) -> None:
        """Verify that copy works as expected."""
        pc_orig = LidarPointCloud.make_random()
        pc_copy = pc_orig.copy()
        self.assertEqual(pc_orig, pc_copy)
        pc_orig.points[0, 0] += 1
        self.assertNotEqual(pc_orig, pc_copy)

    def test_read_pcd_ascii_xyz(self) -> None:
        """Test making a LidarPointCloud with x, y, and z fields from a .pcd file with ascii data."""
        pcd_contents = b'#.PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 3\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 3\nDATA ascii\n3.5999999 -3.0999999 0\n1.0 -3.01 10.0\n4.5999999 -2.90001 -1.0'
        temp_file = tempfile.NamedTemporaryFile(suffix='.pcd')
        temp_file.write(pcd_contents)
        _ = temp_file.seek(0)
        pcd = LidarPointCloud.from_file(temp_file.name)
        self.assertEqual(pcd.nbr_points(), 3)
        expected_points = np.array([[3.5999999, -3.0999999, 0, 0], [1.0, -3.01, 10, 0], [4.5999999, -2.90001, -1.0, 0]]).T
        self.assertEqual(np.all(np.isclose(pcd.points, expected_points)), True)

    def test_read_pcd_ascii_xyzi(self) -> None:
        """Test making a LidarPointCloud with x, y, z, and intensity fields from a .pcd file with ascii data."""
        pcd_contents = b'#.PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z r intensity rcs\nSIZE 4 4 4 4 4 4\nTYPE F F F F F F\nCOUNT 1 1 1 1 1 1\nWIDTH 3\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 3\nDATA ascii\n3.5999999 -3.0999999 0 1 0.5 7.5\n1.0 -3.01 10.0 0.4 10 2.5\n4.5999999 -2.90001 -1.0 0.1 1.5 -3.5'
        temp_file = tempfile.NamedTemporaryFile(suffix='.pcd')
        temp_file.write(pcd_contents)
        _ = temp_file.seek(0)
        pcd = LidarPointCloud.from_file(temp_file.name)
        self.assertEqual(pcd.nbr_points(), 3)
        expected_points = np.array([[3.5999999, -3.0999999, 0, 0.5], [1.0, -3.01, 10, 10], [4.5999999, -2.90001, -1.0, 1.5]]).T
        self.assertEqual(np.all(np.isclose(pcd.points, expected_points)), True)

    def test_read_pcd_ascii_xyzit(self) -> None:
        """Test making a LidarPointCloud with x, y, z, intensity, and time fields from a .pcd file with ascii data."""
        pcd_contents = f'#.PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z r intensity rcs {PCD_TIMESTAMP_FIELD_NAME}\nSIZE 4 4 4 4 4 4 4\nTYPE F F F F F F F\nCOUNT 1 1 1 1 1 1 1\nWIDTH 3\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 3\nDATA ascii\n3.5999999 -3.0999999 0 1 0.5 7.5 0\n1.0 -3.01 10.0 0.4 10 2.5 0.05\n4.5999999 -2.90001 -1.0 0.1 1.5 -3.5 0.1'.encode('utf-8')
        temp_file = tempfile.NamedTemporaryFile(suffix='.pcd')
        temp_file.write(pcd_contents)
        _ = temp_file.seek(0)
        pcd = LidarPointCloud.from_file(temp_file.name)
        self.assertEqual(pcd.nbr_points(), 3)
        expected_points = np.array([[3.5999999, -3.0999999, 0, 0.5, 0], [1.0, -3.01, 10, 10, 0.05], [4.5999999, -2.90001, -1.0, 1.5, 0.1]]).T
        self.assertEqual(np.all(np.isclose(pcd.points, expected_points)), True)

    def test_read_pcd_ascii_shuffled_field_order(self) -> None:
        """
        Test making a LidarPointCloud with x, y, z, intensity, and time fields from a .pcd file
        with ascii data where the fields are in an unusual order.
        """
        pcd_contents = f'#.PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS {PCD_TIMESTAMP_FIELD_NAME} intensity r rcs x y z\nSIZE 4 4 4 4 4 4 4\nTYPE F F F F F F F\nCOUNT 1 1 1 1 1 1 1\nWIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA ascii\n1 2 3 4 5 6 7\n8 9 10 11 12 13 14'.encode('utf-8')
        temp_file = tempfile.NamedTemporaryFile(suffix='.pcd')
        temp_file.write(pcd_contents)
        _ = temp_file.seek(0)
        pcd = LidarPointCloud.from_file(temp_file.name)
        self.assertEqual(pcd.nbr_points(), 2)
        expected_points = np.array([[5, 6, 7, 2, 1], [12, 13, 14, 9, 8]]).T
        self.assertEqual(np.all(np.isclose(pcd.points, expected_points)), True)

    def test_range_filter(self) -> None:
        """Test if Range filter works as expected."""
        points_orig = np.array([[2.26, -0.76, 4.72, -5.46, 9.54, -8.89, 5.45, 7.05, -0.89, 8.58], [-0.88, 1.81, -9.12, 3.32, 3.13, -8.67, -5.11, 6.22, 9.39, -3.25], [4.42, -9.08, 0.12, 2.5, -4.23, 2.08, 8.12, 9.22, -8.71, 3.9], [2.25, 4.32, 4.53, 2.88, 2.84, 0.79, 7.62, 1.21, 3.3, 0.52], [9.72, 9.43, 3.67, 9.99, 5.56, 3.15, 0.02, 7.07, 8.64, 6.16]], dtype=float)
        pc = LidarPointCloud(points_orig)
        pc.range_filter(xrange=(-2, 2))
        should_match = np.array([[-0.76, 1.81, -9.08, 4.32, 9.43], [-0.89, 9.39, -8.71, 3.3, 8.64]]).T
        self.assertTrue(np.array_equal(pc.points, should_match))
        pc = LidarPointCloud(points_orig)
        pc.range_filter(xrange=(5, 10), yrange=(-5, 0), zrange=(3, 5))
        should_match = np.array([[8.58, -3.25, 3.9, 0.52, 6.16]]).T
        self.assertTrue(np.array_equal(pc.points, should_match))
        pc = LidarPointCloud(points_orig)
        pc.range_filter(xrange=(1000, 2000))
        self.assertEqual(pc.nbr_points(), 0)
        pc = LidarPointCloud(points_orig)
        pc.range_filter(xrange=(-100, 100), yrange=(-100, 100), zrange=(-100, 100))
        self.assertTrue(np.array_equal(pc.points, points_orig))

    def test_transform(self) -> None:
        """
        Test the transform function (example transformation matrices taken from
        https://www.springer.com/cda/content/document/cda_downloaddocument/9789048137756-c2.pdf?SGWID=0-0-45-1123955-p173940737
        """
        test_points = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [0.0, 0.0, 0.0]])
        pc = LidarPointCloud(test_points.copy())
        pc.transform(np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0.0, 0.0, 0.0, 1]]))
        shouldMatch = np.array([[2, 1, 2], [1, 2, 2], [0, 0, 1], [0.0, 0.0, 0.0]])
        self.assertTrue(np.array_equal(pc.points, shouldMatch))
        pc = LidarPointCloud(test_points.copy())
        pc.transform(np.array([[0, 0, 1, 4], [1, 0, 0, -3], [0, 1, 0, 7], [0.0, 0.0, 0.0, 1]]))
        shouldMatch = np.array([[4, 4, 5], [-2, -3, -2], [7, 8, 8], [0.0, 0.0, 0.0]])
        self.assertTrue(np.array_equal(pc.points, shouldMatch))

    def test_equality(self) -> None:
        """Test equality of two points cloud based on element-wise difference."""
        test_points = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [0.0, 0.0, 0.0]])
        pc = LidarPointCloud(test_points.copy())
        test_points_2 = np.asarray([[1.0000001, 1e-07, 1], [1e-07, 1.0000001, 1], [0, 0.0, 1], [0.0, 0.0, 0.0]])
        pc2 = LidarPointCloud(test_points_2.copy())
        self.assertEqual(pc, pc2)
        pc = LidarPointCloud.make_random()
        pc2 = LidarPointCloud.make_random()
        self.assertNotEqual(pc, pc2)

    def test_rotate_composite(self) -> None:
        """Testing if points are rotated correctly for a composite rotation sequence."""
        test_point = np.array([[0, 0, -1, 0, 0], [0, -1, 0, 0, 0]]).T
        alpha, beta, gamma = (np.pi, np.pi / 2, np.pi / 2)
        test_rot_matrix_alpha = np.array([[1.0, 0.0, 0.0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        test_rot_matrix_beta = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        test_rot_matrix_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
        rotated_test_point = np.array([[0, 1, 0, 0, 0], [-1, 0, 0, 0, 0]]).T
        pc = LidarPointCloud(test_point)
        pc.rotate(Quaternion(matrix=test_rot_matrix_alpha))
        pc.rotate(Quaternion(matrix=test_rot_matrix_beta))
        pc.rotate(Quaternion(matrix=test_rot_matrix_gamma))
        assert_array_equal(pc.points, rotated_test_point)

