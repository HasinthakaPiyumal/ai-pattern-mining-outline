# Cluster 33

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

def load_boxes_from_lidarpc(nuplandb: NuPlanDB, lidarpc_rec: LidarPc, target_category_names: List[str], filter_boxes: bool, max_distance: float, future_horizon_len_s: float=0.0, future_interval_s: float=0.5, category2id: Optional[Dict[str, int]]=None, map_dilation: float=0.0) -> Dict[str, List[Box3D]]:
    """
    Load all the boxes for a LidarPc.
    :param nuplandb: The multimodal database used in this dataset.
    :param lidarpc_rec: Lidar sample record.
    :param target_category_names: Global names corresponding to the boxes we are interested in obtaining.
    :param filter_boxes: Whether to filter the boxes to be on the drivable area + dilation factor.
    :param max_distance: Radius outside which the boxes will be removed. Helps speed up caching and building the
        GT database.
    :param future_horizon_len_s: Num seconds in the future where we want a future box.
        If a value is provided, the center coordinates and orientation for each box will be provided at 0.5 sec
        intervals. If the value is 0 (default), the function will not provide future center coordinates or orientation.
    :param future_interval_s: Time interval between future waypoints in seconds.
    :param category2id: Mapping from category name to id. This parameter is optional and if provided, it is used to
        populate the box.label property when applicable.
    :param map_dilation: Map dilation factor in meters.
    :return: Dictionary mapping global names of desired categories to list of corresponding boxes.
    """
    if future_horizon_len_s:
        assert 0 < future_interval_s <= future_horizon_len_s
        all_boxes = lidarpc_rec.boxes_with_future_waypoints(future_horizon_len_s=future_horizon_len_s, future_interval_s=future_interval_s)
    else:
        all_boxes = lidarpc_rec.boxes()
    global2boxes: Dict[str, List[Box3D]] = {global_name: [] for global_name in target_category_names}
    for box in all_boxes:
        current_global_name = nuplandb.lidar_box[box.token].category.name
        if current_global_name in target_category_names:
            if category2id and current_global_name in list(category2id.keys()):
                box.label = category2id[current_global_name]
            global2boxes[current_global_name].append(box)
    for global_name, boxes in global2boxes.items():
        car_from_global = lidarpc_rec.ego_pose.trans_matrix_inv
        lidar_from_car = lidarpc_rec.lidar.trans_matrix_inv
        trans_matrix = reduce(np.dot, [lidar_from_car, car_from_global])
        transformed_boxes = [_box_transform(box, trans_matrix) for box in boxes]
        filtered_boxes = [box for box in transformed_boxes if box.distance_plane < max_distance]
        global2boxes[global_name] = filtered_boxes
    return global2boxes

