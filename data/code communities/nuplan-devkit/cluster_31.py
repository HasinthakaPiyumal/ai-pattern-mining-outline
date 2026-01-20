# Cluster 31

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

def _get_past_future_sweep(present_lidarpc: LidarPc, sweep_idx: int) -> LidarPc:
    """
    Find a past or future sweep given the present sweep and its index.
    :param present_lidarpc: The present sweep.
    :param sweep_idx: The sweep index.
        - Negative numbers corresponding to past sweeps.
        - 0 corresponding to the present sweep.
        - Positive numbers corresponding to future sweeps.
    :returns: The specified sweep or None if we hit the start or end of an extraction.
    """
    cur_lidarpc = present_lidarpc
    for _ in range(abs(sweep_idx)):
        if sweep_idx > 0:
            if cur_lidarpc.next is None:
                return None
            else:
                cur_lidarpc = cur_lidarpc.next
        elif sweep_idx < 0:
            if cur_lidarpc.prev is None:
                return None
            else:
                cur_lidarpc = cur_lidarpc.prev
    return cur_lidarpc

