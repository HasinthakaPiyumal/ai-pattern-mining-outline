# Cluster 30

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

def prepare_pointcloud_points(pc: LidarPointCloud, use_intensity: bool=True, use_ring: bool=False, use_lidar_index: bool=False, lidar_indices: Optional[Tuple[int, ...]]=None, sample_apillar_lidar_rings: bool=False) -> LidarPointCloud:
    """
    Prepare the lidar points.
    There are two independent steps:
        - filter points to only use a subset of the lidars
        - change the decorations (intensity and ring)
    :param pc: Pointcloud input.
    :param use_intensity: Whether to use intensity or not.
    :param use_ring: Whether to use ring index or not.
    :param use_lidar_index: Whether to use lidar index as a decoration.
    :param lidar_indices: Which lidars to keep.
        MergedPointCloud has following options:
            0: top lidar
            1: right A pillar lidar
            2: left A pillar lidar
            3: back lidar
            4: front lidar
            None: Use all lidars
    :param sample_apillar_lidar_rings: Whether you want to sample rings for the A-pillar lidars.
    :return: Modified pointcloud.
    """
    a_pillar_lidar_indices = (1, 2)
    ring_indices_to_keep = [0, 1, 2, 3, 4, 5, 6, 8, 11, 17, 23, 29, 35, 38, 39]
    if lidar_indices is None:
        if sample_apillar_lidar_rings:
            keep = np.zeros(pc.points.shape[1])
            keep = np.logical_or(keep, (pc.points[5] != a_pillar_lidar_indices[0]) & (pc.points[5] != a_pillar_lidar_indices[1]))
            for index in a_pillar_lidar_indices:
                keep = np.logical_or(keep, (pc.points[5] == index) & np.isin(pc.points[4], ring_indices_to_keep))
            pc.points = pc.points[:, keep]
    else:
        keep = np.zeros(pc.points.shape[1])
        for index in lidar_indices:
            if sample_apillar_lidar_rings and index in a_pillar_lidar_indices:
                current_keep = (pc.points[5] == index) & np.isin(pc.points[4], ring_indices_to_keep)
            else:
                current_keep = pc.points[5] == index
            keep = np.logical_or(keep, current_keep)
        pc.points = pc.points[:, keep]
    decoration_index = [0, 1, 2]
    if use_intensity:
        decoration_index += [3]
    if use_ring:
        decoration_index += [4]
    if use_lidar_index:
        decoration_index += [5]
    pc.points = pc.points[np.array(decoration_index)]
    return pc

