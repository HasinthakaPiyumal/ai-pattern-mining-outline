# Cluster 22

class EgoPose(Base):
    """
    Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system.
    """
    __tablename__ = 'ego_pose'
    token = Column(sql_types.HexLen8, primary_key=True)
    timestamp = Column(Integer)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    qw: float = Column(Float)
    qx: float = Column(Float)
    qy: float = Column(Float)
    qz: float = Column(Float)
    vx = Column(Float)
    vy = Column(Float)
    vz = Column(Float)
    acceleration_x = Column(Float)
    acceleration_y = Column(Float)
    acceleration_z = Column(Float)
    angular_rate_x = Column(Float)
    angular_rate_y = Column(Float)
    angular_rate_z = Column(Float)
    epsg = Column(Integer)
    log_token = Column(sql_types.HexLen8, ForeignKey('log.token'), nullable=False)

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def quaternion(self) -> Quaternion:
        """
        Get the orientation of ego vehicle as quaternion respect to global coordinate system.
        :return: The orientation in quaternion.
        """
        return Quaternion(self.qw, self.qx, self.qy, self.qz)

    @property
    def translation_np(self) -> npt.NDArray[np.float64]:
        """
        Position of ego vehicle respect to global coordinate system.
        :return: <np.float: 3> Translation.
        """
        return np.array([self.x, self.y, self.z])

    @property
    def trans_matrix(self) -> npt.NDArray[np.float64]:
        """
        Get the transformation matrix.
        :return: <np.float: 4, 4>. Transformation matrix.
        """
        tm: npt.NDArray[np.float64] = self.quaternion.transformation_matrix
        tm[:3, 3] = self.translation_np
        return tm

    @property
    def trans_matrix_inv(self) -> npt.NDArray[np.float64]:
        """
        Get the inverse transformation matrix.
        :return: <np.float: 4, 4>. Inverse transformation matrix.
        """
        tm: npt.NDArray[np.float64] = np.eye(4)
        rot_inv = self.quaternion.rotation_matrix.T
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(np.transpose(-self.translation_np))
        return tm

    def rotate_2d_points2d_to_ego_vehicle_frame(self, points2d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Rotate 2D points from global frame to ego-vehicle frame.
        :param points2d: <np.float: num_points, 2>. 2D points in global frame.
        :return: <np.float: num_points, 2>. 2D points rotated to ego-vehicle frame.
        """
        points3d: npt.NDArray[np.float32] = np.concatenate((points2d, np.zeros_like(points2d[:, 0:1])), axis=-1)
        rotation = R.from_matrix(self.quaternion.rotation_matrix.T)
        ego_rotation_angle = rotation.as_euler('zxy', degrees=True)[0]
        xy_rotation = R.from_euler('z', ego_rotation_angle, degrees=True)
        rotated_points3d = xy_rotation.apply(points3d)
        rotated_points2d: npt.NDArray[np.float64] = rotated_points3d[:, :2]
        return rotated_points2d

    def get_map_crop(self, maps_db: Optional[GPKGMapsDB], xrange: Tuple[float, float], yrange: Tuple[float, float], map_layer_name: str, rotate_face_up: bool, target_imsize_xy: Optional[Tuple[float, float]]=None) -> Tuple[Optional[npt.NDArray[np.float64]], npt.NDArray[np.float64], Tuple[float, ...]]:
        """
        This function returns the crop of the map centered at the current ego-pose with the given xrange and yrange.
        :param maps_db: Map database associated with this database.
        :param xrange: The range in x direction in meters relative to the current ego-pose. Eg: (-60, 60]).
        :param yrange: The range in y direction in meters relative to the current ego-pose Eg: (-60, 60).
        :param map_layer_name: A relevant map layer. Eg: 'drivable_area' or 'intensity'.
        :param rotate_face_up: Boolean indicating whether to rotate the image face up with respect to ego-pose.
        :param target_imsize_xy: The target grid xy dimensions for the output array. The xy resolution in meters / grid
            may be scaled by zooming to the desired dimensions.
        :return: (map_crop, map_translation, map_scale). Where:
            map_crop: The desired crop of the map.
            map_translation: The translation in map coordinates from the origin to the ego-pose.
            map_scale: Map scale (inverse of the map precision). This will be a tuple specifying the zoom in both the x
                and y direction if the target_imsize_xy parameter was set, which causes the resolution to change.

            map_scale and map_translation are useful for transforming objects like pointcloud/boxes to the map_crop.
            Refer to render_on_map().
        """
        if maps_db is None:
            precision: float = 1

            def to_pixel_coords(x: float, y: float) -> Tuple[float, float]:
                """
                Get the image coordinates given the x-y coordinates of point. This implementation simply returns the
                same coordinates.
                :param x: Global x coordinate.
                :param y: Global y coordinate.
                :return: Pixel coordinates in map.
                """
                return (x, y)
        else:
            map_layer = maps_db.load_layer(self.log.map_version, map_layer_name)
            precision = map_layer.precision
            to_pixel_coords = map_layer.to_pixel_coords
        map_scale: Tuple[float, ...] = (1.0 / precision, 1.0 / precision, 1.0)
        ego_translation = self.translation_np
        center_x, center_y = to_pixel_coords(ego_translation[0], ego_translation[1])
        center_x, center_y = (int(center_x), int(center_y))
        top_left = (int(xrange[0] * map_scale[0]), int(yrange[0] * map_scale[1]))
        bottom_right = (int(xrange[1] * map_scale[0]), int(yrange[1] * map_scale[1]))
        rotation = R.from_matrix(self.quaternion.rotation_matrix.T)
        ego_rotation_angle = rotation.as_euler('zxy', degrees=True)[0]
        xy_rotation = R.from_euler('z', ego_rotation_angle, degrees=True)
        map_rotate = 0
        rotated = xy_rotation.apply([[top_left[0], top_left[1], 0], [top_left[0], bottom_right[1], 0], [bottom_right[0], top_left[1], 0], [bottom_right[0], bottom_right[1], 0]])[:, :2]
        rect = cv2.minAreaRect(np.hstack([rotated[:, :1] + center_x, rotated[:, 1:] + center_y]).astype(int))
        rect_angle = rect[2]
        cropped_dimensions: npt.NDArray[np.float32] = np.array([map_scale[0] * (xrange[1] - xrange[0]), map_scale[1] * (yrange[1] - yrange[0])])
        rect = (rect[0], cropped_dimensions, rect_angle)
        rect_angle = rect[2]
        cropped_dimensions = np.array([map_scale[0] * (xrange[1] - xrange[0]), map_scale[1] * (yrange[1] - yrange[0])])
        if rect_angle >= 0:
            rect = (rect[0], cropped_dimensions, rect_angle - 90)
        else:
            rect = (rect[0], cropped_dimensions, rect_angle)
        if ego_rotation_angle < -90:
            map_rotate = -90
        if -90 < ego_rotation_angle < 0:
            map_rotate = 0
        if 0 < ego_rotation_angle < 90:
            map_rotate = 90
        if 90 < ego_rotation_angle < 180:
            map_rotate = 180
        if map_layer is None:
            map_crop = None
        else:
            map_crop = crop_rect(map_layer.data, rect)
            map_crop = ndimage.rotate(map_crop, map_rotate, reshape=False)
            if rotate_face_up:
                map_crop = np.rot90(map_crop)
        if map_layer is None:
            map_upper_left_offset_from_global_coordinate_origin = np.zeros((2,))
        else:
            map_upper_left_offset_from_global_coordinate_origin = np.array([-map_layer.transform_matrix[0, -1], map_layer.transform_matrix[1, -1]])
        ego_offset_from_map_upper_left: npt.NDArray[np.float32] = np.array([center_x, -center_y])
        crop_upper_left_offset_from_ego: npt.NDArray[np.float32] = np.array([xrange[0] * map_scale[0], yrange[0] * map_scale[1]])
        map_translation: npt.NDArray[np.float64] = -map_upper_left_offset_from_global_coordinate_origin - ego_offset_from_map_upper_left - crop_upper_left_offset_from_ego
        map_translation_with_z: npt.NDArray[np.float64] = np.array([map_translation[0], map_translation[1], 0])
        if target_imsize_xy is not None:
            zoom_size_x = target_imsize_xy[0] / cropped_dimensions[0]
            zoom_size_y = target_imsize_xy[1] / cropped_dimensions[1]
            map_crop = ndimage.zoom(map_crop, [zoom_size_x, zoom_size_y])
            map_scale = (zoom_size_x, zoom_size_y)
        return (map_crop, map_translation_with_z, map_scale)

    def get_vector_map(self, maps_db: Optional[GPKGMapsDB], xrange: Tuple[float, float], yrange: Tuple[float, float], connection_scales: Optional[List[int]]=None) -> VectorMapNp:
        """
        This function returns the crop of baseline paths (blps) map centered at the current ego-pose with
        the given xrange and yrange.
        :param maps_db: Map database associated with this database.
        :param xrange: The range in x direction in meters relative to the current ego-pose. Eg: [-60, 60].
        :param yrange: The range in y direction in meters relative to the current ego-pose Eg: [-60, 60].
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        map_version = self.lidar_pc.log.map_version.replace('.gpkg', '')
        blps_gdf = maps_db.load_vector_layer(map_version, 'baseline_paths')
        lane_poly_gdf = maps_db.load_vector_layer(map_version, 'lanes_polygons')
        intersections_gdf = maps_db.load_vector_layer(map_version, 'intersections')
        lane_connectors_gdf = maps_db.load_vector_layer(map_version, 'lane_connectors')
        lane_groups_gdf = maps_db.load_vector_layer(map_version, 'lane_groups_polygons')
        if blps_gdf is None or lane_poly_gdf is None or intersections_gdf is None or (lane_connectors_gdf is None) or (lane_groups_gdf is None):
            coords: npt.NDArray[np.float32] = np.empty([0, 2, 2], dtype=np.float32)
            if not connection_scales:
                connection_scales = [1]
            multi_scale_connections: Dict[int, Any] = {scale: np.empty([0, 2], dtype=np.int64) for scale in connection_scales}
            return VectorMapNp(coords=coords, multi_scale_connections=multi_scale_connections)
        blps_in_lanes = blps_gdf[blps_gdf['lane_fid'].notna()]
        blps_in_intersections = blps_gdf[blps_gdf['lane_connector_fid'].notna()]
        lane_group_info = lane_poly_gdf[['lane_fid', 'lane_group_fid']]
        blps_in_lanes = blps_in_lanes.merge(lane_group_info, on='lane_fid', how='outer')
        lane_connectors_gdf['lane_connector_fid'] = lane_connectors_gdf['fid']
        lane_conns_info = lane_connectors_gdf[['lane_connector_fid', 'intersection_fid', 'exit_lane_fid', 'entry_lane_fid']]
        lane_conns_info = lane_conns_info.astype({'lane_connector_fid': int})
        blps_in_intersections = blps_in_intersections.astype({'lane_connector_fid': int})
        blps_in_intersections = blps_in_intersections.merge(lane_conns_info, on='lane_connector_fid', how='outer')
        lane_blps_info = blps_in_lanes[['fid', 'lane_fid']]
        from_blps_info = lane_blps_info.rename(columns={'fid': 'from_blp', 'lane_fid': 'exit_lane_fid'})
        to_blps_info = lane_blps_info.rename(columns={'fid': 'to_blp', 'lane_fid': 'entry_lane_fid'})
        blps_in_intersections = blps_in_intersections.merge(from_blps_info, on='exit_lane_fid', how='inner')
        blps_in_intersections = blps_in_intersections.merge(to_blps_info, on='entry_lane_fid', how='inner')
        candidate_lane_groups, candidate_intersections = get_candidates(self.translation_np, xrange, yrange, lane_groups_gdf, intersections_gdf)
        candidate_blps_in_lanes = blps_in_lanes[blps_in_lanes['lane_group_fid'].isin(candidate_lane_groups['fid'].astype(int))]
        candidate_blps_in_intersections = blps_in_intersections[blps_in_intersections['intersection_fid'].isin(candidate_intersections['fid'].astype(int))]
        ls_coordinates_list: List[List[List[float]]] = []
        ls_connections_list: List[List[int]] = []
        ls_groupings_list: List[List[int]] = []
        cross_blp_connection: Dict[str, List[int]] = dict()
        build_lane_segments_from_blps(candidate_blps_in_lanes, ls_coordinates_list, ls_connections_list, ls_groupings_list, cross_blp_connection)
        build_lane_segments_from_blps(candidate_blps_in_intersections, ls_coordinates_list, ls_connections_list, ls_groupings_list, cross_blp_connection)
        for blp_id, blp_info in cross_blp_connection.items():
            connect_blp_predecessor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_connections_list)
            connect_blp_successor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_connections_list)
        ls_coordinates: npt.NDArray[np.float64] = np.asarray(ls_coordinates_list, self.translation_np.dtype)
        ls_connections: npt.NDArray[np.int64] = np.asarray(ls_connections_list, np.int64)
        ls_coordinates = ls_coordinates.reshape(-1, 2)
        ls_coordinates = ls_coordinates - self.translation_np[:2]
        ls_coordinates = self.rotate_2d_points2d_to_ego_vehicle_frame(ls_coordinates)
        ls_coordinates = ls_coordinates.reshape(-1, 2, 2).astype(np.float32)
        if connection_scales:
            multi_scale_connections = generate_multi_scale_connections(ls_connections, connection_scales)
        else:
            multi_scale_connections = {1: ls_connections}
        return VectorMapNp(coords=ls_coordinates, multi_scale_connections=multi_scale_connections)

def get_candidates(position: Union[Tuple[float, float], npt.NDArray[np.float64]], xrange: Union[Tuple[float, float], npt.NDArray[np.float64]], yrange: Union[Tuple[float, float], npt.NDArray[np.float64]], lane_groups_gdf: gpd.geodataframe, intersections_gdf: gpd.geodataframe) -> Tuple[gpd.geodataframe, gpd.geodataframe]:
    """
    Given a sample ego_pose position, find applicable lane_groups and intersections within its range.
    :param position: Ego pose position.
    :param xrange: only inside or intersects with xrange would lane_groups and intersections be considered.
    :param yrange: only inside or intersects with yrange would lane_groups and intersections be considered.
    :param lane_groups_gdf: dataframe of lane_groups data
    :param intersections_gdf: dataframe of intersections data
    :return: selected lane_groups dataframe and intersections dataframe within the range of sample ego-pose.
    """
    x_min, x_max = (position[0] + xrange[0], position[0] + xrange[1])
    y_min, y_max = (position[1] + yrange[0], position[1] + yrange[1])
    patch = geometry.box(x_min, y_min, x_max, y_max)
    candidate_lane_groups = lane_groups_gdf[lane_groups_gdf['geometry'].intersects(patch)]
    candidate_intersections = intersections_gdf[intersections_gdf['geometry'].intersects(patch)]
    return (candidate_lane_groups, candidate_intersections)

def build_lane_segments_from_blps(candidate_blps: gpd.geodataframe, ls_coords: List[List[List[float]]], ls_conns: List[List[int]], ls_groupings: List[List[int]], cross_blp_conns: Dict[str, List[int]]) -> None:
    """
    Process candidate baseline paths to small portions of lane-segments with connection info recorded.
    :param candidate_blps: Candidate baseline paths to be cut to lane_segments
    :param ls_coords: Output data recording lane-segment coordinates in format of [N, 2, 2]
    :param ls_conns: Output data recording lane-segment connection relations in format of [M, 2]
    :param ls_groupings: Output data recording lane-segment indices associated with each lane in format
        [num_lanes, num_segments_in_lane]
    :param: cross_blp_conns: Output data recording start_idx/end_idx for each baseline path with id as key.
    """
    for _, blp in candidate_blps.iterrows():
        blp_id = blp['fid']
        px, py = blp.geometry.coords.xy
        ls_num = len(px) - 1
        blp_start_ls = len(ls_coords)
        blp_end_ls = blp_start_ls + ls_num - 1
        ls_grouping = []
        for idx in range(ls_num):
            curr_pt, next_pt = ([px[idx], py[idx]], [px[idx + 1], py[idx + 1]])
            ls_idx = len(ls_coords)
            if idx > 0:
                ls_conns.append([ls_idx - 1, ls_idx])
            ls_coords.append([curr_pt, next_pt])
            ls_grouping.append(ls_idx)
        ls_groupings.append(ls_grouping)
        cross_blp_conns[blp_id] = [blp_start_ls, blp_end_ls]

def connect_blp_predecessor(blp_id: str, lane_conn_info: gpd.geodataframe, cross_blp_conns: Dict[str, List[int]], ls_conns: List[List[int]]) -> None:
    """
    Given a specific baseline path id, find its predecessor and update info in ls_connections information.
    :param blp_id: a specific baseline path id to query
    :param lane_conn_info: baseline paths information in intersections contains the from_blp/to_blp info
    :param cross_blp_conns: Dict to record the baseline path id as key(str) and [blp_start_ls_idx, blp_end_ls_idx] pair
        as value (List[int])
    :param ls_conns: lane_segment_connection to record the [from_ls_idx, to_ls_idx] connection info, updated with
        predecessors found.
    """
    blp_start, blp_end = cross_blp_conns[blp_id]
    predecessor_blp = lane_conn_info[lane_conn_info['to_blp'] == blp_id]
    predecessor_list = predecessor_blp['fid'].to_list()
    for predecessor_id in predecessor_list:
        predecessor_start, predecessor_end = cross_blp_conns[predecessor_id]
        ls_conns.append([predecessor_end, blp_start])

def connect_blp_successor(blp_id: str, lane_conn_info: gpd.geodataframe, cross_blp_conns: Dict[str, List[int]], ls_conns: List[List[int]]) -> None:
    """
    Given a specific baseline path id, find its successor and update info in ls_connections information.
    :param blp_id: a specific baseline path id to query
    :param lane_conn_info: baseline paths information in intersections contains the from_blp/to_blp info
    :param cross_blp_conns: Dict to record the baseline path id as key(str) and [blp_start_ls_idx, blp_end_ls_idx] pair
        as value (List[int])
    :param ls_conns: lane_segment_connnection to record the [from_ls_idx, to_ls_idx] connection info, updated with
        predecessors found.
    """
    blp_start, blp_end = cross_blp_conns[blp_id]
    successor_blp = lane_conn_info[lane_conn_info['from_blp'] == blp_id]
    successor_list = successor_blp['fid'].to_list()
    for successor_id in successor_list:
        successor_start, successor_end = cross_blp_conns[successor_id]
        ls_conns.append([blp_end, successor_start])

def generate_multi_scale_connections(connections: npt.NDArray[np.float64], scales: List[int]) -> Dict[int, npt.NDArray[np.float64]]:
    """
    Generate multi-scale connections by finding the neighors up to max(scales) hops away for each node.

    :param connections: <np.float: num_connections, 2>. 1-hop connections.
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
    """
    node_idx_to_neighbor_dict: Dict[int, Dict[str, Set[int]]] = {}
    for connection in connections:
        start_idx, end_idx = list(connection)
        if start_idx not in node_idx_to_neighbor_dict:
            node_idx_to_neighbor_dict[start_idx] = {'1_hop_neighbors': set()}
        if end_idx not in node_idx_to_neighbor_dict:
            node_idx_to_neighbor_dict[end_idx] = {'1_hop_neighbors': set()}
        node_idx_to_neighbor_dict[start_idx]['1_hop_neighbors'].add(end_idx)
    for scale in range(2, max(scales) + 1):
        for neighbor_dict in node_idx_to_neighbor_dict.values():
            neighbor_dict[f'{scale}_hop_neighbors'] = set()
            for n_hop_neighbor in neighbor_dict[f'{scale - 1}_hop_neighbors']:
                for n_plus_1_hop_neighbor in node_idx_to_neighbor_dict[n_hop_neighbor]['1_hop_neighbors']:
                    neighbor_dict[f'{scale}_hop_neighbors'].add(n_plus_1_hop_neighbor)
    multi_scale_connections: Dict[int, npt.NDArray[np.float64]] = {}
    for scale in scales:
        scale_connections = []
        for node_idx, neighbor_dict in node_idx_to_neighbor_dict.items():
            for n_hop_neighbor in neighbor_dict[f'{scale}_hop_neighbors']:
                scale_connections.append([node_idx, n_hop_neighbor])
        multi_scale_connections[scale] = np.array(scale_connections)
    return multi_scale_connections

class TestGenerateMultiScaleConnections(unittest.TestCase):
    """
    Test generation of multi-scale connections
    """

    def test_generate_multi_scale_connections(self) -> None:
        """Test generate_multi_scale_connections()"""
        connections = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 3], [2, 4]], dtype=np.float64)
        scales = [1, 2, 4]
        expected_multi_scale_connections = {1: connections, 2: np.array([[0, 2], [1, 3], [2, 4], [3, 5], [0, 4], [1, 4], [2, 5]]), 4: np.array([[0, 4], [0, 5], [1, 5]])}
        multi_scale_connections = generate_multi_scale_connections(connections, scales)

        def _convert_to_connection_set(connection_array: npt.NDArray[np.float64]) -> Set[Tuple[float, float]]:
            """
            Convert connections from array to set.

            :param connection_array: <np.float: N, 2>. Connection in array format.
            :return: Connection in set format.
            """
            return {(connection[0], connection[1]) for connection in connection_array}
        self.assertEqual(multi_scale_connections.keys(), expected_multi_scale_connections.keys())
        for key in multi_scale_connections:
            connection_set = _convert_to_connection_set(multi_scale_connections[key])
            expected_connection_set = _convert_to_connection_set(expected_multi_scale_connections[key])
            self.assertEqual(connection_set, expected_connection_set)

class TestVectorMapNp(unittest.TestCase):
    """
    Tests the VectorMapNp class
    """

    def setUp(self) -> None:
        """
        Sets up for the test cases
        """
        coords: npt.NDArray[np.float64] = np.ones([1, 2, 2], dtype=np.float32)
        multi_scale_connections = Dict[int, npt.NDArray[np.float64]]
        self.vector_map_np = VectorMapNp(coords, multi_scale_connections)

    def test_translate(self) -> None:
        """
        Tests the translate method
        """
        vector_map_np = self.vector_map_np
        translate = [1.0, 1.0, 0.0]
        expected_coords = 2.0 * np.ones([1, 2, 2], dtype=np.float32)
        result = vector_map_np.translate(translate)
        self.assertTrue(np.array_equal(result.coords, expected_coords))

    @patch('nuplan.database.nuplan_db_orm.vector_map_np.np.dot', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.vector_map_np.np.concatenate', autospec=True)
    def test_rotate(self, concatenate_mock: Mock, dot_mock: Mock) -> None:
        """
        Tests the rotate method
        """
        vector_map_np = self.vector_map_np
        quarternion = Mock()
        vector_map_np.rotate(quarternion)
        dot_mock.assert_called_once()
        concatenate_mock.assert_called_once()

    def test_scale(self) -> None:
        """
        Tests the scale method
        """
        vector_map_np = self.vector_map_np
        scale = [3.0, 3.0, 3.0]
        expected_coords = 3.0 * np.ones([1, 2, 2], dtype=np.float32)
        result = vector_map_np.scale(scale)
        self.assertTrue(np.array_equal(result.coords, expected_coords))

    def test_xflip(self) -> None:
        """
        Tests the xflip method
        """
        vector_map_np = self.vector_map_np
        expected_coords: npt.NDArray[np.float64] = np.array([[[-1, 1], [-1, 1]]])
        result = vector_map_np.xflip()
        self.assertTrue(np.array_equal(result.coords, expected_coords))

    def test_yflip(self) -> None:
        """
        Tests the yflip method
        """
        vector_map_np = self.vector_map_np
        expected_coords: npt.NDArray[np.float64] = np.array([[[1, -1], [1, -1]]])
        result = vector_map_np.yflip()
        self.assertTrue(np.array_equal(result.coords, expected_coords))

