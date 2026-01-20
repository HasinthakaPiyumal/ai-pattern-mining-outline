# Cluster 61

def parametrize_dir(absdirpath: Optional[str], files: List[str], relpath: Optional[str]) -> List[Any]:
    """
    Converts a target json file as a source of parameters for pytest.
    :param absdirpath: Absolute path of the directory containing the json files
    :param files: Name of the json files
    :param relpath: Relative path to the json file
    :return A list of pytest parameters
    """
    parameters = [pytest.param(None, id='<newname>', marks=[pytest.mark.nuplan_test(relpath=relpath, absdirpath=absdirpath, params=None)])]
    for file in files:
        if file.endswith('.json'):
            parameters.append(parametrize_filebased(absdirpath, file, relpath))
    return parameters

def parametrize_filebased(abspath: Optional[str], filename: str, relpath: Optional[str]) -> Any:
    """
    Converts a target json file as a source of parameters for pytest.
    :param abspath: Absolute path of the json file
    :param filename: Name of the json file
    :param relpath: Relative path to the json file
    :return A pytest parameter
    """
    if filename.endswith('.json'):
        id_ = filename[:-5]
        return pytest.param(None, id=id_, marks=[pytest.mark.nuplan_test(relpath=relpath, absdirpath=abspath, params=id_)])
    else:
        return pytest.param(None, id='-', marks=[pytest.mark.nuplan_test(relpath=relpath, absdirpath=None, params=None)])

@functools.wraps(nuplan_test)
@pytest.mark.nuplan_test(type='hardcoded', params=None, absdirpath=None, relpath=None)
@pytest.mark.usefixtures('scene')
@pytest.mark.parametrize(argnames='nuplan_test', argvalues=[None], ids=['-'])
def testwrapper(*args: Any, **kwargs: Any) -> Any:
    return nuplan_test(*args, **kwargs)

def nuplan_test(path: Optional[str]=None) -> Any:
    """
    This decorator enable pytest to load a sample scene from a json file. The test can then be run normally with pytest
    if the plugin is added to PYTEST_PLUGINS. It can be replaced with any other framework for visual testing/debugging.
    """

    def impl_decorate(nuplan_test: Any) -> Any:
        if path is not None:
            name = sys.modules.get(nuplan_test.__module__).__file__
            abspath = join(dirname(name), path)
            if isdir(abspath):

                @functools.wraps(nuplan_test)
                @pytest.mark.usefixtures('scene')
                @pytest.mark.parametrize(argnames='nuplan_test', argvalues=parametrize_dir(abspath, os.listdir(abspath), path))
                def testwrapper(*args: Any, **kwargs: Any) -> Any:
                    return nuplan_test(*args, **kwargs)
                return testwrapper
            else:

                @functools.wraps(nuplan_test)
                @pytest.mark.usefixtures('scene')
                @pytest.mark.parametrize(argnames='nuplan_test', argvalues=[parametrize_filebased(dirname(abspath), basename(abspath), path)])
                def testwrapper(*args: Any, **kwargs: Any) -> Any:
                    return nuplan_test(*args, **kwargs)
                return testwrapper
        else:

            @functools.wraps(nuplan_test)
            @pytest.mark.nuplan_test(type='hardcoded', params=None, absdirpath=None, relpath=None)
            @pytest.mark.usefixtures('scene')
            @pytest.mark.parametrize(argnames='nuplan_test', argvalues=[None], ids=['-'])
            def testwrapper(*args: Any, **kwargs: Any) -> Any:
                return nuplan_test(*args, **kwargs)
            return testwrapper
    return impl_decorate

def add_map_objects_to_scene(scene: Dict[str, Any], map_object: List[AbstractMapObject], layer: Optional[SemanticMapLayer]=None) -> None:
    """
    Serialize and append map objects to the scene.
    :param scene: scene dict.
    :param map_object: The map object to be added.
    :param layer: SemanticMapLayer type.
    """
    for obj in map_object:
        if isinstance(obj, (StopLine, PolygonMapObject, Intersection, RoadBlockGraphEdgeMapObject)):
            add_polygon_to_scene(scene, obj.polygon, obj.id, _color_to_object_mapping(layer))
        elif isinstance(obj, GraphEdgeMapObject):
            add_polyline_to_scene(scene, obj.baseline_path.discrete_path)

def add_polygon_to_scene(scene: Dict[str, Any], polygon: Polygon, polygon_id: str, color: List[float]) -> None:
    """
    Serialize and append a Polygon to the scene.
    :param scene: scene dict.
    :param polygon: The polygon to be added.
    :param polygon_id: A unique id of the polygon.
    :param color: color of polygon.
    """
    if 'shapes' not in scene.keys():
        scene['shapes'] = dict()
    scene['shapes'][str(polygon_id)] = {'color': color, 'filled': True, 'objects': [[[x, y] for x, y in zip(*polygon.exterior.xy)]]}

def _color_to_object_mapping(layer: SemanticMapLayer) -> List[float]:
    color_mapping = {SemanticMapLayer.STOP_LINE: [1.0, 0.0, 0.0, 1.0], SemanticMapLayer.CROSSWALK: [0.0, 0.0, 1.0, 1.0], SemanticMapLayer.INTERSECTION: [0.0, 1.0, 0.0, 1.0], SemanticMapLayer.ROADBLOCK: [0.0, 1.0, 1.0, 1.0], SemanticMapLayer.ROADBLOCK_CONNECTOR: [0.0, 1.0, 1.0, 1.0]}
    try:
        return color_mapping[layer]
    except KeyError:
        return [1.0, 1.0, 1.0, 0.5]

def add_polyline_to_scene(scene: Dict[str, Any], polyline: List[StateSE2]) -> None:
    """
    Serialize and append a polyline to the scene.
    :param scene: scene dict.
    :param polyline: The polyline to be added.
    """
    if 'path_info' not in scene.keys():
        scene['path_info'] = []
    scene['path_info'].extend([[pose.x, pose.y, pose.heading] for pose in polyline])

def build_lane_segments_from_blps_with_trim(point: Point2D, radius: float, map_obj: MapObject, start_lane_seg_idx: int) -> Union[None, Tuple[List[List[List[float]]], List[Tuple[int, int]], List[List[int]], List[str], List[str], Tuple[int, int]]]:
    """
    Process baseline paths of associated lanes/lane connectors to series of lane-segments along with connection info.
    :param point: [m] x, y coordinates in global frame.
    :param radius [m] floating number about vector map query range.
    :param map_obj: Lane or LaneConnector for building lane segments from associated baseline path.
    :param start_lane_seg_idx: Starting index for lane segments.
    :return
        obj_coords: Data recording lane-segment coordinates in format of [N, 2, 2].
        obj_conns: Data recording lane-segment connection relations in format of [M, 2].
        obj_groupings: Data recording lane-segment indices associated with each lane in format
            [num_lanes, num_segments_in_lane].
        obj_lane_ids: Data recording map object ids of lane/lane connector containing lane-segment.
        obj_roadblock_ids: Data recording map object ids of roadblock/roadblock connector containing lane-segment.
        obj_cross_blp_conn: Data storing indices of first and last lane segments of a given map object's baseline path
            as [blp_start_lane_seg_idx, blp_end_lane_seg_idx].
    """
    map_obj_id = map_obj.id
    roadblock_id = map_obj.get_roadblock_id()
    nodes = map_obj.baseline_path.discrete_path
    nodes = trim_lane_nodes(point, radius, nodes)
    if len(nodes) <= 2:
        return None
    lane_seg_num = len(nodes) - 1
    end_lane_seg_idx = start_lane_seg_idx + lane_seg_num - 1
    obj_coords = split_blp_lane_segments(nodes, lane_seg_num)
    obj_conns = connect_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    obj_groupings = group_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    obj_lane_ids = [map_obj_id for _ in range(lane_seg_num)]
    obj_roadblock_ids = [roadblock_id for _ in range(lane_seg_num)]
    obj_cross_blp_conn = (start_lane_seg_idx, end_lane_seg_idx)
    return (obj_coords, obj_conns, obj_groupings, obj_lane_ids, obj_roadblock_ids, obj_cross_blp_conn)

def trim_lane_nodes(point: Point2D, radius: float, lane_nodes: List[StateSE2]) -> List[StateSE2]:
    """
    Trim the discretized baseline path nodes to be within the radius. To ensure the continuity of
    the lane coords, only the end points of the lane/lane connectors are trimmed. For example, given
    the points in lane as [p_1, ..., p_n], only points at the end of the lane [p_1,...p_f], or
    [p_e, ... p_n] will be trimmed if they are further than the radius. The points between p_f and
    p_e will be kept regardless their distance to the ego.
    :param point: [m] x, y coordinates in global frame.
    :param radius [m] floating number about vector map query range.
    :param lane_nodes: The list of lane nodes to be filtered.
    :return obj_groupings: Data recording lane-segment indices associated with given lane/lane connector.
    """
    radius_squared = radius ** 2
    for index, node in enumerate(lane_nodes):
        if (node.x - point.x) ** 2 + (node.y - point.y) ** 2 <= radius_squared:
            start_index = index
            break
    else:
        return []
    for index, node in enumerate(lane_nodes[::-1]):
        if (node.x - point.x) ** 2 + (node.y - point.y) ** 2 <= radius_squared:
            end_index = len(lane_nodes) - index
            break
    return lane_nodes[start_index:end_index]

def split_blp_lane_segments(nodes: List[StateSE2], lane_seg_num: int) -> List[List[List[float]]]:
    """
    Split baseline path points into series of lane segment coordinate vectors.
    :param nodes: Baseline path nodes to be cut to lane_segments.
    :param lane_seg_num: Number of lane segments to split from baseline path.
    :return obj_coords: Data recording lane segment coordinates in format of [N, 2, 2].
    """
    obj_coords: List[List[List[float]]] = []
    for idx in range(lane_seg_num):
        curr_pt = [nodes[idx].x, nodes[idx].y]
        next_pt = [nodes[idx + 1].x, nodes[idx + 1].y]
        obj_coords.append([curr_pt, next_pt])
    return obj_coords

def connect_blp_lane_segments(start_lane_seg_idx: int, lane_seg_num: int) -> List[Tuple[int, int]]:
    """
    Add connection info for neighboring segments in baseline path.
    :param start_lane_seg_idx: Index for first lane segment in baseline path.
    :param lane_seg_num: Number of lane segments.
    :return obj_conns: Data recording lane-segment connection relations [from_lane_seg_idx, to_lane_seg_idx].
    """
    obj_conns: List[Tuple[int, int]] = []
    for lane_seg_idx in range(start_lane_seg_idx + 1, start_lane_seg_idx + lane_seg_num):
        obj_conns.append((lane_seg_idx - 1, lane_seg_idx))
    return obj_conns

def group_blp_lane_segments(start_lane_seg_idx: int, lane_seg_num: int) -> List[List[int]]:
    """
    Collect lane segment indices across lane/lane connector baseline path.
    :param start_lane_seg_idx: Index for first lane segment in baseline path.
    :param lane_seg_num: Number of lane segments.
    :return obj_groupings: Data recording lane-segment indices associated with given lane/lane connector.
    """
    obj_grouping: List[int] = []
    for lane_seg_idx in range(start_lane_seg_idx, start_lane_seg_idx + lane_seg_num):
        obj_grouping.append(lane_seg_idx)
    return [obj_grouping]

def build_lane_segments_from_blps(map_obj: MapObject, start_lane_seg_idx: int) -> Tuple[List[List[List[float]]], List[Tuple[int, int]], List[List[int]], List[str], List[str], Tuple[int, int]]:
    """
    Process baseline paths of associated lanes/lane connectors to series of lane-segments along with connection info.
    :param map_obj: Lane or LaneConnector for building lane segments from associated baseline path.
    :param start_lane_seg_idx: Starting index for lane segments.
    :return
        obj_coords: Data recording lane-segment coordinates in format of [N, 2, 2].
        obj_conns: Data recording lane-segment connection relations in format of [M, 2].
        obj_groupings: Data recording lane-segment indices associated with each lane in format
            [num_lanes, num_segments_in_lane].
        obj_lane_ids: Data recording map object ids of lane/lane connector containing lane-segment.
        obj_roadblock_ids: Data recording map object ids of roadblock/roadblock connector containing lane-segment.
        obj_cross_blp_conn: Data storing indices of first and last lane segments of a given map object's baseline path
            as [blp_start_lane_seg_idx, blp_end_lane_seg_idx].
    """
    map_obj_id = map_obj.id
    roadblock_id = map_obj.get_roadblock_id()
    nodes = map_obj.baseline_path.discrete_path
    lane_seg_num = len(nodes) - 1
    end_lane_seg_idx = start_lane_seg_idx + lane_seg_num - 1
    obj_coords = split_blp_lane_segments(nodes, lane_seg_num)
    obj_conns = connect_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    obj_groupings = group_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    obj_lane_ids = [map_obj_id for _ in range(lane_seg_num)]
    obj_roadblock_ids = [roadblock_id for _ in range(lane_seg_num)]
    obj_cross_blp_conn = (start_lane_seg_idx, end_lane_seg_idx)
    return (obj_coords, obj_conns, obj_groupings, obj_lane_ids, obj_roadblock_ids, obj_cross_blp_conn)

def extract_polygon_from_map_object(map_object: MapObject) -> List[Point2D]:
    """
    Extract polygon from map object.
    :param map_object: input MapObject.
    :return: polygon as list of Point2D.
    """
    x_coords, y_coords = map_object.polygon.exterior.coords.xy
    return [Point2D(x, y) for x, y in zip(x_coords, y_coords)]

def get_roadblock_ids_from_trajectory(map_api: AbstractMap, ego_states: List[EgoState]) -> List[str]:
    """
    Extract ids of roadblocks and roadblock connectors containing points in specified trajectory.
    :param map_api: map to perform extraction on.
    :param ego_states: sequence of agent states representing trajectory.
    :return roadblock_ids: List of ids of roadblocks/roadblock connectors containing trajectory points.
    """
    roadblock_ids: List[str] = []
    roadblock_candidates: List[RoadBlockGraphEdgeMapObject] = []
    last_roadblock = None
    points = [ego_state.rear_axle.point for ego_state in ego_states]
    for point in points:
        if last_roadblock and last_roadblock.contains_point(point):
            continue
        if last_roadblock and (not roadblock_candidates):
            roadblock_candidates = last_roadblock.outgoing_edges
        roadblock_candidates = [roadblock for roadblock in roadblock_candidates if roadblock.contains_point(point)]
        if len(roadblock_candidates) == 1:
            last_roadblock = roadblock_candidates.pop()
            roadblock_ids.append(last_roadblock.id)
        elif not roadblock_candidates:
            roadblock_objects = extract_roadblock_objects(map_api, point)
            if len(roadblock_objects) == 1:
                last_roadblock = roadblock_objects.pop()
                roadblock_ids.append(last_roadblock.id)
            else:
                roadblock_candidates = roadblock_objects
    return roadblock_ids

def extract_roadblock_objects(map_api: AbstractMap, point: Point2D) -> List[RoadBlockGraphEdgeMapObject]:
    """
    Extract roadblock or roadblock connectors from map containing point if they exist.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :return List of roadblocks/roadblock connectors containing point if they exist.
    """
    roadblock = map_api.get_one_map_object(point, SemanticMapLayer.ROADBLOCK)
    if roadblock:
        return [roadblock]
    else:
        roadblock_conns = map_api.get_all_map_objects(point, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        return cast(List[RoadBlockGraphEdgeMapObject], roadblock_conns)

@nuplan_test(path='json/stop_lines/nearby.json')
def test_get_nearby_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test getting nearby stop lines.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_distance, expected_id in zip(scene['markers'], scene['xtr']['expected_nearest_distance'], scene['xtr']['expected_nearest_id']):
        pose = marker['pose']
        stop_line_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.STOP_LINE)
        assert stop_line_id is not None
        assert expected_distance == distance
        assert expected_id == stop_line_id
        stop_line: StopLine = nuplan_map.get_map_object(stop_line_id, SemanticMapLayer.STOP_LINE)
        add_map_objects_to_scene(scene, [stop_line])

@nuplan_test(path='json/stop_lines/on_stopline.json')
def test_get_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test getting stop lines at a point.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_id in zip(scene['markers'], scene['xtr']['expected_nearest_id']):
        pose = marker['pose']
        stop_line: StopLine = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.STOP_LINE)
        assert stop_line is not None
        assert expected_id == stop_line.id
        assert stop_line.contains_point(Point2D(pose[0], pose[1]))
        add_map_objects_to_scene(scene, [stop_line])

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_baseline_queries_in_lane(scene: Dict[str, Any]) -> None:
    """
    Test baseline queries.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    expected_arc_length = scene['xtr']['expected_arc_length']
    expected_pose = scene['xtr']['expected_pose']
    expected_curvature = scene['xtr']['expected_curvature']
    poses = {}
    for marker, exp_arc_length, exp_pose, exp_curv in zip(scene['markers'], expected_arc_length, expected_pose.values(), expected_curvature):
        pose = marker['pose']
        point = Point2D(pose[0], pose[1])
        lane = nuplan_map.get_one_map_object(point, SemanticMapLayer.LANE)
        assert lane is not None
        assert lane.contains_point(point)
        add_map_objects_to_scene(scene, [lane])
        lane_blp = lane.baseline_path
        arc_length = lane_blp.get_nearest_arc_length_from_position(point)
        pose = lane_blp.get_nearest_pose_from_position(point)
        curv = lane_blp.get_curvature_at_arc_length(arc_length)
        poses[marker['id']] = pose
        assert arc_length == pytest.approx(exp_arc_length)
        assert pose == StateSE2(exp_pose[0], exp_pose[1], exp_pose[2])
        assert curv == pytest.approx(exp_curv)
        constructed_blp = NuPlanPolylineMapObject(get_row_with_value(lane._baseline_paths_df, 'lane_fid', lane.id))
        constructed_blp_arc_length = constructed_blp.get_nearest_arc_length_from_position(point)
        constructed_blp_pose = constructed_blp.get_nearest_pose_from_position(point)
        constructed_blp_curv = constructed_blp.get_curvature_at_arc_length(constructed_blp_arc_length)
        assert arc_length == pytest.approx(constructed_blp_arc_length)
        assert pose == constructed_blp_pose
        assert curv == pytest.approx(constructed_blp_curv)
    for pose_id, pose in poses.items():
        add_marker_to_scene(scene, str(pose_id), pose)

def add_marker_to_scene(scene: Dict[str, Any], marker_id: str, pose: StateSE2) -> None:
    """
    Serialize and append a marker to the scene.
    :param scene: scene dict.
    :param marker_id: A unique id of the marker.
    :param pose: The pose of the marker.
    """
    if 'markers' not in scene.keys():
        scene['markers'] = []
    scene['markers'].append({'id': int(marker_id), 'name': marker_id, 'pose': pose.serialize(), 'shape': 'arrow'})

@nuplan_test(path='json/intersections/on_intersection.json')
def test_get_intersections(scene: Dict[str, Any]) -> None:
    """
    Test getting intersections at a point.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_id in zip(scene['markers'], scene['xtr']['expected_nearest_id']):
        pose = marker['pose']
        intersection: Intersection = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.INTERSECTION)
        assert intersection is not None
        assert expected_id == intersection.id
        assert intersection.contains_point(Point2D(pose[0], pose[1]))
        add_map_objects_to_scene(scene, [intersection])

@nuplan_test(path='json/intersections/nearby.json')
def test_get_nearby_intersection(scene: Dict[str, Any]) -> None:
    """
    Test getting nearby crosswalks.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_distance, expected_id in zip(scene['markers'], scene['xtr']['expected_nearest_distance'], scene['xtr']['expected_nearest_id']):
        pose = marker['pose']
        intersection_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.INTERSECTION)
        assert intersection_id is not None
        assert expected_distance == distance
        assert expected_id == intersection_id
        intersection: Intersection = nuplan_map.get_map_object(intersection_id, SemanticMapLayer.INTERSECTION)
        add_map_objects_to_scene(scene, [intersection])

@nuplan_test(path='json/crosswalks/nearby.json')
def test_get_nearby_crosswalks(scene: Dict[str, Any]) -> None:
    """
    Test getting nearby crosswalks.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_distance, expected_id in zip(scene['markers'], scene['xtr']['expected_nearest_distance'], scene['xtr']['expected_nearest_id']):
        pose = marker['pose']
        crosswalk_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.CROSSWALK)
        assert crosswalk_id is not None
        assert expected_distance == distance
        assert expected_id == crosswalk_id
        crosswalk: PolygonMapObject = nuplan_map.get_map_object(crosswalk_id, SemanticMapLayer.CROSSWALK)
        add_map_objects_to_scene(scene, [crosswalk])

@nuplan_test(path='json/crosswalks/on_crosswalk.json')
def test_get_crosswalk(scene: Dict[str, Any]) -> None:
    """
    Test getting crosswalk at a point.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_id in zip(scene['markers'], scene['xtr']['expected_nearest_id']):
        pose = marker['pose']
        crosswalk: PolygonMapObject = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.CROSSWALK)
        assert crosswalk is not None
        assert expected_id == crosswalk.id
        assert crosswalk.contains_point(Point2D(pose[0], pose[1]))
        add_map_objects_to_scene(scene, [crosswalk])

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_incoming_outgoing_lanes(scene: Dict[str, Any]) -> None:
    """
    Test getting incoming and outgoing lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)
        assert len(lane_connectors) > 0
        incoming_edges = lane_connectors[0].incoming_edges
        outgoing_edges = lane_connectors[0].outgoing_edges
        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_lane_left_boundaries(scene: Dict[str, Any]) -> None:
    """
    Test getting left boundaries of lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)
        assert len(lane_connectors) > 0
        left_boundary = lane_connectors[0].left_boundary
        assert left_boundary is not None
        assert isinstance(left_boundary, PolylineMapObject)
        add_polyline_to_scene(scene, left_boundary.discrete_path)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_lane_right_boundaries(scene: Dict[str, Any]) -> None:
    """
    Test getting right boundaries of lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)
        assert len(lane_connectors) > 0
        right_boundary = lane_connectors[0].right_boundary
        assert right_boundary is not None
        assert isinstance(right_boundary, PolylineMapObject)
        add_polyline_to_scene(scene, right_boundary.discrete_path)

@nuplan_test(path='json/intersections/on_intersection_with_stop_lines.json')
def test_get_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test getting stop lines from lane connector.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)
        assert len(lane_connectors) > 0
        stop_lines = lane_connectors[0].stop_lines
        assert len(stop_lines) > 0
        add_map_objects_to_scene(scene, stop_lines)

@nuplan_test(path='json/intersections/on_intersection_with_no_stop_lines.json')
def test_get_stop_lines_empty(scene: Dict[str, Any]) -> None:
    """
    Test getting stop lines from lane connector when there are no stop lines.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)
        assert len(lane_connectors) > 0
        stop_lines = lane_connectors[0].stop_lines
        assert len(stop_lines) == 0
        add_map_objects_to_scene(scene, stop_lines)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_polygon(scene: Dict[str, Any]) -> None:
    """
    Test getting polygons from lane_connector.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        point = Point(pose[0], pose[1])
        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)
        assert len(lane_connectors) > 0
        polygon = lane_connectors[0].polygon
        assert polygon.contains(point)
        add_map_objects_to_scene(scene, lane_connectors)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_incoming_outgoing_roadblock_connectors(scene: Dict[str, Any]) -> None:
    """
    Test getting incoming and outgoing roadblock connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK)
        assert roadblock is not None
        incoming_edges = roadblock.incoming_edges
        outgoing_edges = roadblock.outgoing_edges
        assert len(incoming_edges) > 0
        assert len(outgoing_edges) > 0
        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)

@nuplan_test(path='json/connections/no_end_connection.json')
def test_no_end_roadblock_connector(scene: Dict[str, Any]) -> None:
    """
    Test when there are not outgoing roadblock connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK)
        assert roadblock is not None
        incoming_edges = roadblock.incoming_edges
        outgoing_edges = roadblock.outgoing_edges
        assert not outgoing_edges
        add_map_objects_to_scene(scene, incoming_edges)

@nuplan_test(path='json/connections/no_start_connection.json')
def test_no_start_roadblock_connector(scene: Dict[str, Any]) -> None:
    """
    Test when there are not incoming lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK)
        assert roadblock is not None
        incoming_edges = roadblock.incoming_edges
        outgoing_edges = roadblock.outgoing_edges
        assert not incoming_edges
        add_map_objects_to_scene(scene, outgoing_edges)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_roadblock_interior_edges(scene: Dict[str, Any]) -> None:
    """
    Test getting roadblock's interior lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK)
        assert roadblock is not None
        interior_edges = roadblock.interior_edges
        assert len(interior_edges) > 0
        add_map_objects_to_scene(scene, interior_edges)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_roadblock_polygon(scene: Dict[str, Any]) -> None:
    """
    Test getting roadblock's polygon.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK)
        assert roadblock is not None
        polygon = roadblock.polygon
        assert polygon
        assert isinstance(polygon, Polygon)

def test_connect_blp_lane_segments() -> None:
    """
    Test connecting lane indices.
    """
    start_lane_seg_idx = 0
    lane_seg_num = 10
    obj_conns = connect_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    assert len(obj_conns) == lane_seg_num - 1
    assert len(obj_conns[0]) == 2
    assert isinstance(obj_conns, List)
    assert isinstance(obj_conns[0], tuple)
    assert isinstance(obj_conns[0][0], int)

def test_group_blp_lane_segments() -> None:
    """
    Test grouping lane indices belonging to same lane/lane connector.
    """
    start_lane_seg_idx = 0
    lane_seg_num = 10
    obj_groupings = group_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    assert len(obj_groupings) == 1
    assert len(obj_groupings[0]) == lane_seg_num
    assert isinstance(obj_groupings, List)
    assert isinstance(obj_groupings[0], List)
    assert isinstance(obj_groupings[0][0], int)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_build_lane_segments_from_blps_with_trim(scene: Dict[str, Any]) -> None:
    """
    Test build and trim the lane segments from the baseline paths.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        radius = 20
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is not None
        start_idx = 0
        trimmed_obj_coords, trimmed_obj_conns, trimmed_obj_groupings, trimmed_obj_lane_ids, trimmed_obj_roadblock_ids, trimmed_obj_cross_blp_conn = build_lane_segments_from_blps_with_trim(Point2D(pose[0], pose[1]), radius, lane, start_idx)
        start_idx = 0
        obj_coords, obj_conns, obj_groupings, obj_lane_ids, obj_roadblock_ids, obj_cross_blp_conn = build_lane_segments_from_blps(lane, start_idx)
        assert len(trimmed_obj_coords) > 0
        assert len(trimmed_obj_conns) > 0
        assert len(trimmed_obj_groupings) > 0
        assert len(trimmed_obj_lane_ids) > 0
        assert len(trimmed_obj_roadblock_ids) > 0
        assert len(trimmed_obj_cross_blp_conn) == 2
        assert len(trimmed_obj_coords) == len(trimmed_obj_conns) + 1
        assert len(trimmed_obj_coords) == len(trimmed_obj_groupings[0])
        assert len(trimmed_obj_coords) == len(trimmed_obj_lane_ids)
        assert len(trimmed_obj_coords) == len(trimmed_obj_roadblock_ids)
        assert len(trimmed_obj_coords) <= len(obj_coords)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_connect_trimmed_lane_conn_predecessor(scene: Dict[str, Any]) -> None:
    """
    Test connecting trimmed lane connector to incoming lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane_connector: LaneConnector = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)[0]
        assert lane_connector is not None
        incoming_edges = lane_connector.incoming_edges
        assert len(incoming_edges) > 0
        lane: Lane = lane_connector.incoming_edges[0]
        assert lane is not None
        start_idx = 0
        radius = 20
        trim_nodes = build_lane_segments_from_blps_with_trim(Point2D(pose[0], pose[1]), radius, lane, start_idx)
        if trim_nodes is not None:
            obj_coords, obj_conns, obj_groupings, obj_lane_ids, obj_roadblock_ids, obj_cross_blp_conn = trim_nodes
        else:
            continue
        cross_blp_conns: Dict[str, List[int]] = {}
        cross_blp_conns[lane_connector.id] = [0, 0]
        cross_blp_conns[incoming_edges[0].id] = [0, 0]
        lane_seg_pred_conns = connect_trimmed_lane_conn_predecessor(obj_coords, lane_connector, cross_blp_conns)
        assert len(lane_seg_pred_conns) > 0
        assert isinstance(lane_seg_pred_conns, List)
        assert isinstance(lane_seg_pred_conns[0], tuple)
        assert isinstance(lane_seg_pred_conns[0][0], int)

def connect_trimmed_lane_conn_predecessor(lane_coords: Tuple[List[List[List[float]]]], lane_conn: LaneConnector, cross_blp_conns: Dict[str, Tuple[int, int]], distance_threshold: float=0.3) -> List[Tuple[int, int]]:
    """
    Given a specific lane connector, find its predecessor lane and return new connection info. To
                       handle the case where the end points of lane connector or/and the predecissor
                       lane being trimmed, a distance check is performed to make sure the end points
                       of the predecissor lane is close enough to be connected.
    :param: lane_coords: the lane segment cooridnates
    :param lane_conn: a specific lane connector.
    :param cross_blp_conns: Dict recording the map object id as key(str) and corresponding [first segment index,
        last segment index] pair as value (Tuple[int, int]).
    :param distance_threshold: the distance to determine if the end points are close enough to be
        connected in the lane graph.
    :return lane_seg_pred_conns: container recording the connection [from_lane_seg_idx, to_lane_seg_idx] between
        last predecessor segment and first segment of given lane connector.
    """
    lane_seg_pred_conns: List[Tuple[int, int]] = []
    lane_conn_start_seg_idx, lane_conn_end_seg_idx = cross_blp_conns[lane_conn.id]
    incoming_lanes = [incoming_edge for incoming_edge in lane_conn.incoming_edges if isinstance(incoming_edge, Lane)]
    for incoming_lane in incoming_lanes:
        lane_id = incoming_lane.id
        if lane_id in cross_blp_conns.keys():
            predecessor_start_idx, predecessor_end_idx = cross_blp_conns[lane_id]
            if np.linalg.norm(np.array(lane_coords[predecessor_end_idx][1]) - np.array(lane_coords[lane_conn_start_seg_idx][0])) < distance_threshold:
                lane_seg_pred_conns.append((predecessor_end_idx, lane_conn_start_seg_idx))
    return lane_seg_pred_conns

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_connect_trimmed_lane_conn_successor(scene: Dict[str, Any]) -> None:
    """
    Test connecting trimmed lane connector to outgoing lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane_connector: LaneConnector = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)[0]
        assert lane_connector is not None
        outgoing_edges = lane_connector.outgoing_edges
        assert len(outgoing_edges) > 0
        lane: Lane = lane_connector.outgoing_edges[0]
        assert lane is not None
        start_idx = 0
        radius = 20
        trim_nodes = build_lane_segments_from_blps_with_trim(Point2D(pose[0], pose[1]), radius, lane, start_idx)
        if trim_nodes is not None:
            obj_coords, obj_conns, obj_groupings, obj_lane_ids, obj_roadblock_ids, obj_cross_blp_conn = trim_nodes
        else:
            continue
        cross_blp_conns: Dict[str, List[int]] = {}
        cross_blp_conns[lane_connector.id] = [0, 0]
        cross_blp_conns[outgoing_edges[0].id] = [0, 0]
        lane_seg_suc_conns = connect_trimmed_lane_conn_successor(obj_coords, lane_connector, cross_blp_conns)
        assert len(lane_seg_suc_conns) > 0
        assert isinstance(lane_seg_suc_conns, List)
        assert isinstance(lane_seg_suc_conns[0], tuple)
        assert isinstance(lane_seg_suc_conns[0][0], int)

def connect_trimmed_lane_conn_successor(lane_coords: Tuple[List[List[List[float]]]], lane_conn: LaneConnector, cross_blp_conns: Dict[str, Tuple[int, int]], distance_threshold: float=0.3) -> List[Tuple[int, int]]:
    """
    Given a specific lane connector, find its successor lane and return new connection info. To
                       handle the case where the end points of lane connector or/and the predecissor
                       lane being trimmed, a distance check is performed to make sure the end points
                       of the predecissor lane is close enough to be connected.
    :param: lane_coords: the lane segment cooridnates
    :param lane_conn: a specific lane connector.
    :param cross_blp_conns: Dict recording the map object id as key(str) and corresponding [first segment index,
        last segment index] pair as value (Tuple[int, int]).
    :param distance_threshold: the distance to determine if the end points are close enough to be
        connected in the lane graph.
    :return lane_seg_suc_conns: container recording the connection [from_lane_seg_idx, to_lane_seg_idx] between
        last segment of given lane connector and first successor lane segment.
    """
    lane_seg_suc_conns: List[Tuple[int, int]] = []
    lane_conn_start_seg_idx, lane_conn_end_seg_idx = cross_blp_conns[lane_conn.id]
    outgoing_lanes = [outgoing_edge for outgoing_edge in lane_conn.outgoing_edges if isinstance(outgoing_edge, Lane)]
    for outgoing_lane in outgoing_lanes:
        lane_id = outgoing_lane.id
        if lane_id in cross_blp_conns.keys():
            successor_start_idx, successor_end_seg_idx = cross_blp_conns[lane_id]
            if np.linalg.norm(np.array(lane_coords[lane_conn_end_seg_idx][1]) - np.array(lane_coords[successor_start_idx][0])) < distance_threshold:
                lane_seg_suc_conns.append((lane_conn_end_seg_idx, successor_start_idx))
    return lane_seg_suc_conns

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_build_lane_segments_from_blps(scene: Dict[str, Any]) -> None:
    """
    Test building lane segments from baseline paths.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is not None
        start_idx = 0
        obj_coords, obj_conns, obj_groupings, obj_lane_ids, obj_roadblock_ids, obj_cross_blp_conn = build_lane_segments_from_blps(lane, start_idx)
        assert len(obj_coords) > 0
        assert len(obj_conns) > 0
        assert len(obj_groupings) > 0
        assert len(obj_lane_ids) > 0
        assert len(obj_roadblock_ids) > 0
        assert len(obj_cross_blp_conn) == 2
        assert len(obj_coords) == len(obj_conns) + 1
        assert len(obj_coords) == len(obj_groupings[0])
        assert len(obj_coords) == len(obj_lane_ids)
        assert len(obj_coords) == len(obj_roadblock_ids)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_connect_lane_conn_predecessor(scene: Dict[str, Any]) -> None:
    """
    Test connecting lane connector to incoming lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane_connector: LaneConnector = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)[0]
        assert lane_connector is not None
        incoming_edges = lane_connector.incoming_edges
        assert len(incoming_edges) > 0
        cross_blp_conns: Dict[str, List[int]] = {}
        cross_blp_conns[lane_connector.id] = [0, 0]
        cross_blp_conns[incoming_edges[0].id] = [0, 0]
        lane_seg_pred_conns = connect_lane_conn_predecessor(lane_connector, cross_blp_conns)
        assert len(lane_seg_pred_conns) > 0
        assert isinstance(lane_seg_pred_conns, List)
        assert isinstance(lane_seg_pred_conns[0], tuple)
        assert isinstance(lane_seg_pred_conns[0][0], int)

def connect_lane_conn_predecessor(lane_conn: LaneConnector, cross_blp_conns: Dict[str, Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Given a specific lane connector, find its predecessor lane and return new connection info.
    :param lane_conn: a specific lane connector.
    :param cross_blp_conns: Dict recording the map object id as key(str) and corresponding [first segment index,
        last segment index] pair as value (Tuple[int, int]).
    :return lane_seg_pred_conns: container recording the connection [from_lane_seg_idx, to_lane_seg_idx] between
        last predecessor segment and first segment of given lane connector.
    """
    lane_seg_pred_conns: List[Tuple[int, int]] = []
    lane_conn_start_seg_idx, lane_conn_end_seg_idx = cross_blp_conns[lane_conn.id]
    incoming_lanes = [incoming_edge for incoming_edge in lane_conn.incoming_edges if isinstance(incoming_edge, Lane)]
    for incoming_lane in incoming_lanes:
        lane_id = incoming_lane.id
        if lane_id in cross_blp_conns.keys():
            predecessor_start_idx, predecessor_end_idx = cross_blp_conns[lane_id]
            lane_seg_pred_conns.append((predecessor_end_idx, lane_conn_start_seg_idx))
    return lane_seg_pred_conns

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_connect_lane_conn_successor(scene: Dict[str, Any]) -> None:
    """
    Test connecting lane connector to outgoing lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane_connector: LaneConnector = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)[0]
        assert lane_connector is not None
        outgoing_edges = lane_connector.outgoing_edges
        assert len(outgoing_edges) > 0
        cross_blp_conns: Dict[str, List[int]] = {}
        cross_blp_conns[lane_connector.id] = [0, 0]
        cross_blp_conns[outgoing_edges[0].id] = [0, 0]
        lane_seg_suc_conns = connect_lane_conn_successor(lane_connector, cross_blp_conns)
        assert len(lane_seg_suc_conns) > 0
        assert isinstance(lane_seg_suc_conns, List)
        assert isinstance(lane_seg_suc_conns[0], tuple)
        assert isinstance(lane_seg_suc_conns[0][0], int)

def connect_lane_conn_successor(lane_conn: LaneConnector, cross_blp_conns: Dict[str, Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Given a specific lane connector, find its successor lane and return new connection info.
    :param lane_conn: a specific lane connector.
    :param cross_blp_conns: Dict recording the map object id as key(str) and corresponding [first segment index,
        last segment index] pair as value (Tuple[int, int]).
    :return lane_seg_suc_conns: container recording the connection [from_lane_seg_idx, to_lane_seg_idx] between
        last segment of given lane connector and first successor lane segment.
    """
    lane_seg_suc_conns: List[Tuple[int, int]] = []
    lane_conn_start_seg_idx, lane_conn_end_seg_idx = cross_blp_conns[lane_conn.id]
    outgoing_lanes = [outgoing_edge for outgoing_edge in lane_conn.outgoing_edges if isinstance(outgoing_edge, Lane)]
    for outgoing_lane in outgoing_lanes:
        lane_id = outgoing_lane.id
        if lane_id in cross_blp_conns.keys():
            successor_start_idx, successor_end_seg_idx = cross_blp_conns[lane_id]
            lane_seg_suc_conns.append((lane_conn_end_seg_idx, successor_start_idx))
    return lane_seg_suc_conns

@nuplan_test(path='json/crosswalks/nearby.json')
def test_extract_polygon_from_map_object_crosswalk(scene: Dict[str, Any]) -> None:
    """
    Test extracting polygon from map object. Tests crosswalks.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    radius = 20
    for marker in scene['markers']:
        pose = marker['pose']
        layers = nuplan_map.get_proximal_map_objects(Point2D(pose[0], pose[1]), radius, [SemanticMapLayer.CROSSWALK])
        crosswalks = layers[SemanticMapLayer.CROSSWALK]
        assert len(crosswalks) > 0
        crosswalk_polygon = extract_polygon_from_map_object(crosswalks[0])
        assert isinstance(crosswalk_polygon, List)
        assert len(crosswalk_polygon) > 0
        assert isinstance(crosswalk_polygon[0], Point2D)

@nuplan_test(path='json/stop_lines/nearby.json')
def test_extract_polygon_from_map_object_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test extracting polygon from map object. Tests stop lines.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    radius = 20
    for marker in scene['markers']:
        pose = marker['pose']
        layers = nuplan_map.get_proximal_map_objects(Point2D(pose[0], pose[1]), radius, [SemanticMapLayer.STOP_LINE])
        stop_lines = layers[SemanticMapLayer.STOP_LINE]
        assert len(stop_lines) > 0
        stop_line_polygon = extract_polygon_from_map_object(stop_lines[0])
        assert isinstance(stop_line_polygon, List)
        assert len(stop_line_polygon) > 0
        assert isinstance(stop_line_polygon[0], Point2D)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_extract_roadblock_objects_roadblocks(scene: Dict[str, Any]) -> None:
    """
    Test extract roadblock or roadblock connectors from map containing point. Tests roadblocks.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock_objects = extract_roadblock_objects(nuplan_map, Point2D(pose[0], pose[1]))
        assert isinstance(roadblock_objects, List)
        assert len(roadblock_objects) > 0
        roadblock_object = roadblock_objects[0]
        assert isinstance(roadblock_object, RoadBlockGraphEdgeMapObject)
        roadblock_polygon = extract_polygon_from_map_object(roadblock_object)
        assert isinstance(roadblock_polygon, List)
        assert len(roadblock_polygon) > 0
        assert isinstance(roadblock_polygon[0], Point2D)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_extract_roadblock_objects_roadblock_connectors(scene: Dict[str, Any]) -> None:
    """
    Test extract roadblock or roadblock connectors from map containing point. Tests roadblock connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock_objects = extract_roadblock_objects(nuplan_map, Point2D(pose[0], pose[1]))
        assert isinstance(roadblock_objects, List)
        assert len(roadblock_objects) > 0
        roadblock_object = roadblock_objects[0]
        assert isinstance(roadblock_object, RoadBlockGraphEdgeMapObject)
        roadblock_polygon = extract_polygon_from_map_object(roadblock_object)
        assert isinstance(roadblock_polygon, List)
        assert len(roadblock_polygon) > 0
        assert isinstance(roadblock_polygon[0], Point2D)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_roadblock_ids_from_trajectory(scene: Dict[str, Any]) -> None:
    """
    Test extracting ids of roadblocks and roadblock connectors containing points specified in trajectory.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    trajectory: List[EgoState] = []
    for marker in scene['markers']:
        pose = marker['pose']
        ego_state = get_sample_ego_state()
        ego_state.car_footprint.rear_axle = StateSE2(pose[0], pose[1], pose[2])
        trajectory.append(ego_state)
    roadblock_ids = get_roadblock_ids_from_trajectory(nuplan_map, trajectory)
    assert isinstance(roadblock_ids, List)
    for roadblock_id in roadblock_ids:
        assert isinstance(roadblock_id, str)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_distance_between_map_object_and_point_lanes_roadblocks(scene: Dict[str, Any]) -> None:
    """
    Test get distance between point and nearest surface of specified map object.
    Tests lane/connectors and roadblock/connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    radius = 35
    pose = scene['markers'][0]['pose']
    point = Point2D(pose[0], pose[1])
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    layers = nuplan_map.get_proximal_map_objects(point, radius, layer_names)
    for layer_name in layer_names:
        map_objects = layers[layer_name]
        assert len(map_objects) > 0
        dist = get_distance_between_map_object_and_point(point, map_objects[0])
        assert dist <= radius

def get_distance_between_map_object_and_point(point: Point2D, map_object: MapObject) -> float:
    """
    Get distance between point and nearest surface of specified map object.
    :param point: Point to calculate distance between.
    :param map_object: MapObject (containing underlying polygon) to check distance between.
    :return: Computed distance.
    """
    return float(geom.Point(point.x, point.y).distance(map_object.polygon))

@nuplan_test(path='json/crosswalks/nearby.json')
def test_get_distance_between_map_object_and_point_crosswalks(scene: Dict[str, Any]) -> None:
    """
    Test get distance between point and nearest surface of specified map object. Tests crosswalks.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    radius = 35
    pose = scene['markers'][0]['pose']
    point = Point2D(pose[0], pose[1])
    layers = nuplan_map.get_proximal_map_objects(point, radius, [SemanticMapLayer.CROSSWALK])
    map_objects = layers[SemanticMapLayer.CROSSWALK]
    assert len(map_objects) > 0
    dist = get_distance_between_map_object_and_point(point, map_objects[0])
    assert dist <= radius

@nuplan_test(path='json/stop_lines/nearby.json')
def test_get_distance_between_map_object_and_point_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test get distance between point and nearest surface of specified map object. Tests stop lines.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    radius = 35
    pose = scene['markers'][0]['pose']
    point = Point2D(pose[0], pose[1])
    layers = nuplan_map.get_proximal_map_objects(point, radius, [SemanticMapLayer.STOP_LINE])
    map_objects = layers[SemanticMapLayer.STOP_LINE]
    assert len(map_objects) > 0
    dist = get_distance_between_map_object_and_point(point, map_objects[0])
    assert dist <= radius

def assert_helper(first_markers: List[Dict[str, List[float]]], second_markers: List[Dict[str, List[float]]], assertion: Callable[[Lane, Lane, bool], None], map: AbstractMap, inverse: bool) -> None:
    """
    Helper function to remove redundant lane instantiation and checking
    """
    for first_marker, second_marker in zip(first_markers, second_markers):
        first_point = Point2D(*first_marker['pose'][:2])
        second_point = Point2D(*second_marker['pose'][:2])
        first_lane = map.get_one_map_object(first_point, SemanticMapLayer.LANE)
        second_lane = map.get_one_map_object(second_point, SemanticMapLayer.LANE)
        assertion(first_lane, second_lane, inverse)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_incoming_outgoing_lane_connectors(scene: Dict[str, Any]) -> None:
    """
    Test getting incoming and outgoing lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is not None
        incoming_edges = lane.incoming_edges
        outgoing_edges = lane.outgoing_edges
        assert len(incoming_edges) > 0
        assert len(outgoing_edges) > 0
        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)

@nuplan_test(path='json/connections/no_end_connection.json')
def test_no_end_lane_connector(scene: Dict[str, Any]) -> None:
    """
    Test when there are not outgoing lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is not None
        incoming_edges = lane.incoming_edges
        outgoing_edges = lane.outgoing_edges
        assert not outgoing_edges
        add_map_objects_to_scene(scene, incoming_edges)

@nuplan_test(path='json/connections/no_start_connection.json')
def test_no_start_lane_connector(scene: Dict[str, Any]) -> None:
    """
    Test when there are not incoming lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is not None
        incoming_edges = lane.incoming_edges
        outgoing_edges = lane.outgoing_edges
        assert not incoming_edges
        add_map_objects_to_scene(scene, outgoing_edges)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_lane_left_boundaries(scene: Dict[str, Any]) -> None:
    """
    Test getting left boundaries of lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is not None
        left_boundary = lane.left_boundary
        assert left_boundary is not None
        assert isinstance(left_boundary, PolylineMapObject)
        add_polyline_to_scene(scene, left_boundary.discrete_path)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_lane_right_boundaries(scene: Dict[str, Any]) -> None:
    """
    Test getting right boundaries of lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is not None
        right_boundary = lane.right_boundary
        assert right_boundary is not None
        assert isinstance(right_boundary, PolylineMapObject)
        add_polyline_to_scene(scene, right_boundary.discrete_path)

@nuplan_test(path='json/lanes/lanes_in_same_roadblock.json')
def test_lane_is_same_roadblock(scene: Dict[str, Any]) -> None:
    """
    Test if lanes are in the same roadblock
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])

    def is_same_roadblock(first_lane: Lane, second_lane: Lane, inverse: bool) -> None:
        if not inverse:
            assert first_lane.is_same_roadblock(second_lane)
        else:
            assert not first_lane.is_same_roadblock(second_lane)
    assert_helper(scene['markers'][:4:2], scene['markers'][1:4:2], is_same_roadblock, nuplan_map, False)
    assert_helper(scene['markers'][4::2], scene['markers'][5::2], is_same_roadblock, nuplan_map, True)

@nuplan_test(path='json/lanes/lanes_are_adjacent.json')
def test_lane_is_adjacent_to(scene: Dict[str, Any]) -> None:
    """
    Test if lanes are adjacent
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])

    def is_adjacent_to(first_lane: Lane, second_lane: Lane, inverse: bool) -> None:
        if not inverse:
            assert first_lane.is_adjacent_to(second_lane)
        else:
            assert not first_lane.is_adjacent_to(second_lane)
    assert_helper(scene['markers'][:4:2], scene['markers'][1:4:2], is_adjacent_to, nuplan_map, False)
    assert_helper(scene['markers'][4::2], scene['markers'][5::2], is_adjacent_to, nuplan_map, True)

@nuplan_test(path='json/lanes/lane_is_left_of.json')
def test_lane_is_left_of(scene: Dict[str, Any]) -> None:
    """
    Test if first is left of second
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])

    def is_left_of(first_lane: Lane, second_lane: Lane, inverse: bool) -> None:
        if not inverse:
            assert first_lane.is_left_of(second_lane)
        else:
            assert not first_lane.is_left_of(second_lane)
    assert_helper(scene['markers'][:4:2], scene['markers'][1:4:2], is_left_of, nuplan_map, False)
    assert_helper(scene['markers'][4::2], scene['markers'][5::2], is_left_of, nuplan_map, True)

@nuplan_test(path='json/lanes/lane_is_left_of.json')
def test_lane_is_right_of(scene: Dict[str, Any]) -> None:
    """
    Test if first is right of second
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])

    def is_right_of(first_lane: Lane, second_lane: Lane, inverse: bool) -> None:
        if not inverse:
            assert first_lane.is_right_of(second_lane)
        else:
            assert not first_lane.is_right_of(second_lane)
    assert_helper(scene['markers'][1:4:2], scene['markers'][:4:2], is_right_of, nuplan_map, False)
    assert_helper(scene['markers'][5::2], scene['markers'][4::2], is_right_of, nuplan_map, True)

@nuplan_test(path='json/lanes/get_adjacent_lanes.json')
def test_get_lane_adjacent_lanes(scene: Dict[str, Any]) -> None:
    """
    Test if getting correct adjacent lanes
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        left_lane, right_lane = lane.adjacent_edges
        assert left_lane or right_lane
        if left_lane:
            assert left_lane.is_left_of(lane)
            assert left_lane.is_adjacent_to(lane)
        if right_lane:
            assert right_lane.is_right_of(lane)
            assert right_lane.is_adjacent_to(lane)

@nuplan_test(path='json/lanes/lane_index.json')
def test_get_lane_index(scene: Dict[str, Any]) -> None:
    """
    Test if getting correct lane index
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_index in zip(scene['markers'], scene['xtr']['expected_lane_index']):
        pose = marker['pose']
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is not None
        assert lane.index == expected_index

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_incoming_outgoing_roadblock(scene: Dict[str, Any]) -> None:
    """
    Test getting incoming and outgoing lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock_connectors: List[RoadBlockGraphEdgeMapObject] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR)
        assert len(roadblock_connectors) > 0
        incoming_edges = roadblock_connectors[0].incoming_edges
        outgoing_edges = roadblock_connectors[0].outgoing_edges
        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_roadblock_connector_interior_edges(scene: Dict[str, Any]) -> None:
    """
    Test getting roadblock connector's interior lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock_connectors: List[RoadBlockGraphEdgeMapObject] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR)
        assert len(roadblock_connectors) > 0
        interior_edges = roadblock_connectors[0].interior_edges
        assert len(interior_edges) > 0
        add_map_objects_to_scene(scene, interior_edges)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_roadblock_connector_polygon(scene: Dict[str, Any]) -> None:
    """
    Test getting roadblock connector's polygon.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock_connectors: List[RoadBlockGraphEdgeMapObject] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR)
        assert len(roadblock_connectors) > 0
        polygon = roadblock_connectors[0].polygon
        assert polygon
        assert isinstance(polygon, Polygon)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_is_in_layer_lane(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test is in lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        assert nuplan_map.is_in_layer(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_is_in_layer_intersection(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test is in intersection.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        assert nuplan_map.is_in_layer(Point2D(pose[0], pose[1]), SemanticMapLayer.INTERSECTION)

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_lane(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting one lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_speed_limit in zip(scene['markers'], scene['xtr']['expected_speed_limit']):
        pose = marker['pose']
        point = Point2D(pose[0], pose[1])
        lane = nuplan_map.get_one_map_object(point, SemanticMapLayer.LANE)
        assert lane is not None
        assert lane.contains_point(point)
        assert lane.speed_limit_mps == pytest.approx(expected_speed_limit)
        add_map_objects_to_scene(scene, [lane])

@nuplan_test(path='json/baseline/no_baseline.json')
def test_no_baseline(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test when there is no baseline.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is None
        lane_connector = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)
        assert not lane_connector

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_lane_connector(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    idx = 0
    for marker in scene['markers']:
        pose = marker['pose']
        point = Point2D(pose[0], pose[1])
        lane_connectors = nuplan_map.get_all_map_objects(point, SemanticMapLayer.LANE_CONNECTOR)
        assert lane_connectors is not None
        add_map_objects_to_scene(scene, lane_connectors)
        for lane_connector in lane_connectors:
            assert lane_connector.contains_point(point)
            assert lane_connector.speed_limit_mps == pytest.approx(scene['xtr']['expected_speed_limit'][idx])
            idx += 1
    pose = scene['markers'][0]['pose']
    with pytest.raises(AssertionError):
        nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)

@nuplan_test(path='json/get_nearest/lane.json')
def test_get_nearest_lane(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting nearest lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_distance, expected_id in zip(scene['markers'], scene['xtr']['expected_nearest_distance'], scene['xtr']['expected_nearest_id']):
        pose = marker['pose']
        lane_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane_id == expected_id
        assert distance == expected_distance
        lane = nuplan_map.get_map_object(str(lane_id), SemanticMapLayer.LANE)
        add_map_objects_to_scene(scene, [lane])

@nuplan_test(path='json/get_nearest/lane_connector.json')
def test_get_nearest_lane_connector(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting nearest lane connector.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker, expected_distance, expected_id in zip(scene['markers'], scene['xtr']['expected_nearest_distance'], scene['xtr']['expected_nearest_id']):
        pose = marker['pose']
        lane_connector_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)
        lane_connector = nuplan_map.get_map_object(str(lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)
        add_map_objects_to_scene(scene, [lane_connector])

@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_roadblock(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting one roadblock.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        point = Point2D(pose[0], pose[1])
        roadblock = nuplan_map.get_one_map_object(point, SemanticMapLayer.ROADBLOCK)
        assert roadblock is not None
        assert roadblock.contains_point(point)
        add_map_objects_to_scene(scene, [roadblock])

@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_roadblock_connector(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting roadblock connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        point = Point2D(pose[0], pose[1])
        roadblock_connectors = nuplan_map.get_all_map_objects(point, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        assert roadblock_connectors is not None
        add_map_objects_to_scene(scene, roadblock_connectors)
        for roadblock_connector in roadblock_connectors:
            assert roadblock_connector.contains_point(point)
    pose = scene['markers'][0]['pose']
    with pytest.raises(AssertionError):
        nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR)

@nuplan_test(path='json/get_nearest/lane.json')
def test_get_nearest_roadblock(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting nearest roadblock.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK)
        roadblock = nuplan_map.get_map_object(str(roadblock_id), SemanticMapLayer.ROADBLOCK)
        assert roadblock_id
        add_map_objects_to_scene(scene, [roadblock])

@nuplan_test(path='json/get_nearest/lane_connector.json')
def test_get_nearest_roadblock_connector(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting nearest roadblock connector.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    for marker in scene['markers']:
        pose = marker['pose']
        roadblock_connector_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR)
        assert roadblock_connector_id != -1
        assert distance != np.NaN
        roadblock_connector = nuplan_map.get_map_object(str(roadblock_connector_id), SemanticMapLayer.ROADBLOCK_CONNECTOR)
        assert roadblock_connector
        add_map_objects_to_scene(scene, [roadblock_connector])

@nuplan_test(path='json/neighboring/all_map_objects.json')
def test_get_proximal_map_objects(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test get_neighbor_lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene['map']['area'])
    marker = scene['markers'][0]
    pose = marker['pose']
    map_objects = nuplan_map.get_proximal_map_objects(Point2D(pose[0], pose[1]), 40, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR, SemanticMapLayer.STOP_LINE, SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION])
    assert len(map_objects[SemanticMapLayer.LANE]) == scene['xtr']['expected_num_lanes']
    assert len(map_objects[SemanticMapLayer.LANE_CONNECTOR]) == scene['xtr']['expected_num_lane_connectors']
    assert len(map_objects[SemanticMapLayer.ROADBLOCK]) == scene['xtr']['expected_num_roadblocks']
    assert len(map_objects[SemanticMapLayer.ROADBLOCK_CONNECTOR]) == scene['xtr']['expected_num_roadblock_connectors']
    assert len(map_objects[SemanticMapLayer.STOP_LINE]) == scene['xtr']['expected_num_stop_lines']
    assert len(map_objects[SemanticMapLayer.CROSSWALK]) == scene['xtr']['expected_num_cross_walks']
    assert len(map_objects[SemanticMapLayer.INTERSECTION]) == scene['xtr']['expected_num_intersections']
    for layer, map_objects in map_objects.items():
        add_map_objects_to_scene(scene, map_objects, layer)

@nuplan_test()
def test_unsupported_neighbor_map_objects(map_factory: NuPlanMapFactory) -> None:
    """
    Test throw if unsupported layer is queried.
    """
    nuplan_map = map_factory.build_map_from_name('us-nv-las-vegas-strip')
    with pytest.raises(AssertionError):
        nuplan_map.get_proximal_map_objects(Point2D(0, 0), 15, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR, SemanticMapLayer.STOP_LINE, SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.TRAFFIC_LIGHT])

@nuplan_test()
def test_get_available_map_objects(map_factory: NuPlanMapFactory) -> None:
    """
    Test getting available map objects for all SemanticMapLayers.
    """
    nuplan_map = map_factory.build_map_from_name('us-nv-las-vegas-strip')
    assert set(nuplan_map.get_available_map_objects()) == {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR, SemanticMapLayer.STOP_LINE, SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA}

class TestStateRepresentation(unittest.TestCase):
    """Test StateSE2 and Point2D"""

    def test_point2d(self) -> None:
        """Test Point2D"""
        x = 1.2222
        y = 3.553435
        point = Point2D(x=x, y=y)
        self.assertAlmostEqual(point.x, x)
        self.assertAlmostEqual(point.y, y)

    def test_state_se2(self) -> None:
        """Test StateSE2"""
        x = 1.2222
        y = 3.553435
        heading = 1.32498
        state = StateSE2(x, y, heading)
        self.assertAlmostEqual(state.x, x)
        self.assertAlmostEqual(state.y, y)
        self.assertAlmostEqual(state.heading, heading)

def rotate_2d(point: Point2D, rotation_matrix: npt.NDArray[np.float64]) -> Point2D:
    """
    Rotate 2D point with a 2d rotation matrix
    :param point: to be rotated
    :param rotation_matrix: [[R11, R12], [R21, R22]]
    :return: rotated point
    """
    assert rotation_matrix.shape == (2, 2)
    rotated_point = np.array([point.x, point.y]) @ rotation_matrix
    return Point2D(rotated_point[0], rotated_point[1])

class TestTransform(unittest.TestCase):
    """Tests for transform functions"""

    def test_rotate_2d(self) -> None:
        """Tests rotation of 2D point"""
        point = Point2D(1, 0)
        rotation_matrix: npt.NDArray[np.float32] = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        result = rotate_2d(point, rotation_matrix)
        self.assertEqual(result, Point2D(0, 1))

    def test_translate(self) -> None:
        """Tests translate"""
        pose = StateSE2(3, 5, np.pi / 4)
        translation: npt.NDArray[np.float32] = np.array([1, 2], dtype=np.float32)
        result = translate(pose, translation)
        self.assertEqual(result, StateSE2(4, 7, np.pi / 4))

    def test_rotate(self) -> None:
        """Tests rotation of SE2 pose by rotation matrix"""
        pose = StateSE2(1, 2, np.pi / 4)
        rotation_matrix: npt.NDArray[np.float32] = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        result = rotate(pose, rotation_matrix)
        self.assertAlmostEqual(result.x, -2)
        self.assertAlmostEqual(result.y, 1)
        self.assertAlmostEqual(result.heading, -np.pi / 4)

    def test_rotate_angle(self) -> None:
        """Tests rotation of SE2 pose by angle (in radian)"""
        pose = StateSE2(1, 2, np.pi / 4)
        angle = -np.pi / 2
        result = rotate_angle(pose, angle)
        self.assertAlmostEqual(result.x, -2)
        self.assertAlmostEqual(result.y, 1)
        self.assertAlmostEqual(result.heading, -np.pi / 4)

    def test_transform(self) -> None:
        """Tests transformation of SE2 pose"""
        pose = StateSE2(1, 2, 0)
        transform_matrix: npt.NDArray[np.float32] = np.array([[-3, -2, 5], [0, -1, 4], [0, 0, 1]], dtype=np.float32)
        result = transform(pose, transform_matrix)
        self.assertAlmostEqual(result.x, 2)
        self.assertAlmostEqual(result.y, 0)
        self.assertAlmostEqual(result.heading, np.pi, places=4)

    @patch('nuplan.common.geometry.transform.translate')
    def test_translate_longitudinally(self, mock_translate: Mock) -> None:
        """Tests longitudinal translation"""
        pose = StateSE2(1, 2, np.arctan(1 / 3))
        result = translate_longitudinally(pose, np.sqrt(10))
        np.testing.assert_array_almost_equal(mock_translate.call_args.args[1], np.array([3, 1]))
        self.assertEqual(result, mock_translate.return_value)

    @patch('nuplan.common.geometry.transform.translate')
    def test_translate_laterally(self, mock_translate: Mock) -> None:
        """Tests lateral translation"""
        pose = StateSE2(1, 2, np.arctan(1 / 3))
        result = translate_laterally(pose, np.sqrt(10))
        np.testing.assert_array_almost_equal(mock_translate.call_args.args[1], np.array([-1, 3]))
        self.assertEqual(result, mock_translate.return_value)

    @patch('nuplan.common.geometry.transform.translate')
    def test_translate_longitudinally_and_laterally(self, mock_translate: Mock) -> None:
        """Tests longitudinal and lateral translation"""
        pose = StateSE2(1, 2, np.arctan(1 / 3))
        result = translate_longitudinally_and_laterally(pose, np.sqrt(10), np.sqrt(10))
        np.testing.assert_array_almost_equal(mock_translate.call_args.args[1], np.array([2, 4]))
        self.assertEqual(result, mock_translate.return_value)

class VectorMapFeatureBuilder(ScriptableFeatureBuilder):
    """
    Feature builder for constructing map features in a vector-representation.
    """

    def __init__(self, radius: float, connection_scales: Optional[List[int]]=None) -> None:
        """
        Initialize vector map builder with configuration parameters.
        :param radius:  The query radius scope relative to the current ego-pose.
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        super().__init__()
        self._radius = radius
        self._connection_scales = connection_scales

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return VectorMap

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'vector_map'

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            lane_seg_coords, lane_seg_conns, lane_seg_groupings, lane_seg_lane_ids, lane_seg_roadblock_ids = get_neighbor_vector_map(scenario.map_api, ego_coords, self._radius)
            on_route_status = get_on_route_status(scenario.get_route_roadblock_ids(), lane_seg_roadblock_ids)
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(lane_seg_coords, lane_seg_conns, lane_seg_groupings, on_route_status, traffic_light_data, ego_state.rear_axle)
            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            ego_state = current_input.history.ego_states[-1]
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            lane_seg_coords, lane_seg_conns, lane_seg_groupings, lane_seg_lane_ids, lane_seg_roadblock_ids = get_neighbor_vector_map(initialization.map_api, ego_coords, self._radius)
            on_route_status = get_on_route_status(initialization.route_roadblock_ids, lane_seg_roadblock_ids)
            if current_input.traffic_light_data is None:
                raise ValueError('Cannot build VectorMap feature. PlannerInput.traffic_light_data is None')
            traffic_light_data = current_input.traffic_light_data
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(lane_seg_coords, lane_seg_conns, lane_seg_groupings, on_route_status, traffic_light_data, ego_state.rear_axle)
            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.ignore
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> VectorMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorMap.
        """
        multi_scale_connections: Dict[int, torch.Tensor] = {}
        for key in list_tensor_data:
            if key.startswith('vector_map.multi_scale_connections_'):
                multi_scale_connections[int(key[len('vector_map.multi_scale_connections_'):])] = list_tensor_data[key][0].detach().numpy()
        lane_groupings = [t.detach().numpy() for t in list_list_tensor_data['vector_map.lane_groupings'][0]]
        return VectorMap(coords=[list_tensor_data['vector_map.coords'][0].detach().numpy()], lane_groupings=[lane_groupings], multi_scale_connections=[multi_scale_connections], on_route_status=[list_tensor_data['vector_map.on_route_status'][0].detach().numpy()], traffic_light_data=[list_tensor_data['vector_map.traffic_light_data'][0].detach().numpy()])

    @torch.jit.ignore
    def _pack_to_feature_tensor_dict(self, lane_coords: LaneSegmentCoords, lane_conns: LaneSegmentConnections, lane_groupings: LaneSegmentGroupings, lane_on_route_status: LaneOnRouteStatusData, traffic_light_data: LaneSegmentTrafficLightData, anchor_state: StateSE2) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature tranform.
        :param lane_coords: The LaneSegmentCoords returned from `get_neighbor_vector_map` to transform.
        :param lane_conns: The LaneSegmentConnections returned from `get_neighbor_vector_map` to transform.
        :param lane_groupings: The LaneSegmentGroupings returned from `get_neighbor_vector_map` to transform.
        :param lane_on_route_status: The LaneOnRouteStatusData returned from `get_neighbor_vector_map` to transform.
        :param traffic_light_data: The LaneSegmentTrafficLightData returned from `get_neighbor_vector_map` to transform.
        :param anchor_state: The ego state to transform to vector.
        """
        lane_segment_coords: torch.tensor = torch.tensor(lane_coords.to_vector(), dtype=torch.float64)
        lane_segment_conns: torch.tensor = torch.tensor(lane_conns.to_vector(), dtype=torch.int64)
        on_route_status: torch.tensor = torch.tensor(lane_on_route_status.to_vector(), dtype=torch.float32)
        traffic_light_array: torch.tensor = torch.tensor(traffic_light_data.to_vector(), dtype=torch.float32)
        lane_segment_groupings: List[torch.tensor] = []
        for lane_grouping in lane_groupings.to_vector():
            lane_segment_groupings.append(torch.tensor(lane_grouping, dtype=torch.int64))
        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)
        return ({'lane_segment_coords': lane_segment_coords, 'lane_segment_conns': lane_segment_conns, 'on_route_status': on_route_status, 'traffic_light_array': traffic_light_array, 'anchor_state': anchor_state_tensor}, {'lane_segment_groupings': lane_segment_groupings}, {})

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        lane_segment_coords = tensor_data['lane_segment_coords']
        anchor_state = tensor_data['anchor_state']
        lane_segment_conns = tensor_data['lane_segment_conns']
        if len(lane_segment_conns.shape) == 1:
            if lane_segment_conns.shape[0] == 0:
                lane_segment_conns = torch.zeros((0, 2), device=lane_segment_coords.device, layout=lane_segment_coords.layout, dtype=torch.int64)
            else:
                raise ValueError(f'Unexpected shape for lane_segment_conns: {lane_segment_conns.shape}')
        lane_segment_coords = lane_segment_coords.reshape(-1, 2)
        lane_segment_coords = coordinates_to_local_frame(lane_segment_coords, anchor_state, precision=torch.float64)
        lane_segment_coords = lane_segment_coords.reshape(-1, 2, 2).float()
        if self._connection_scales is not None:
            multi_scale_connections = _generate_multi_scale_connections(lane_segment_conns, self._connection_scales)
        else:
            multi_scale_connections = {1: lane_segment_conns}
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {'vector_map.lane_groupings': [list_tensor_data['lane_segment_groupings']]}
        list_tensor_output: Dict[str, List[torch.Tensor]] = {'vector_map.coords': [lane_segment_coords], 'vector_map.on_route_status': [tensor_data['on_route_status']], 'vector_map.traffic_light_data': [tensor_data['traffic_light_array']]}
        for key in multi_scale_connections:
            list_tensor_output[f'vector_map.multi_scale_connections_{key}'] = [multi_scale_connections[key]]
        tensor_output: Dict[str, torch.Tensor] = {}
        return (tensor_output, list_tensor_output, list_list_tensor_output)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        return {'neighbor_vector_map': {'radius': str(self._radius)}, 'initial_ego_state': empty}

def get_neighbor_vector_map(map_api: AbstractMap, point: Point2D, radius: float) -> Tuple[LaneSegmentCoords, LaneSegmentConnections, LaneSegmentGroupings, LaneSegmentLaneIDs, LaneSegmentRoadBlockIDs]:
    """
    Extract neighbor vector map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :return
        lane_seg_coords: lane_segment coords in shape of [num_lane_segment, 2, 2].
        lane_seg_conns: lane_segment connections [start_idx, end_idx] in shape of [num_connection, 2].
        lane_seg_groupings: collection of lane_segment indices in each lane in shape of
            [num_lane, num_lane_segment_in_lane].
        lane_seg_lane_ids: lane ids of segments at given index in coords in shape of [num_lane_segment 1].
        lane_seg_roadblock_ids: roadblock ids of segments at given index in coords in shape of [num_lane_segment 1].
    """
    lane_seg_coords: List[List[List[float]]] = []
    lane_seg_conns: List[Tuple[int, int]] = []
    lane_seg_groupings: List[List[int]] = []
    lane_seg_lane_ids: List[str] = []
    lane_seg_roadblock_ids: List[str] = []
    cross_blp_conns: Dict[str, Tuple[int, int]] = dict()
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    nearest_vector_map = map_api.get_proximal_map_objects(point, radius, layer_names)
    for layer_name in layer_names:
        for map_obj in nearest_vector_map[layer_name]:
            start_lane_seg_idx = len(lane_seg_coords)
            trim_nodes = build_lane_segments_from_blps_with_trim(point, radius, map_obj, start_lane_seg_idx)
            if trim_nodes is not None:
                obj_coords, obj_conns, obj_groupings, obj_lane_ids, obj_roadblock_ids, obj_cross_blp_conn = trim_nodes
                lane_seg_coords += obj_coords
                lane_seg_conns += obj_conns
                lane_seg_groupings += obj_groupings
                lane_seg_lane_ids += obj_lane_ids
                lane_seg_roadblock_ids += obj_roadblock_ids
                cross_blp_conns[map_obj.id] = obj_cross_blp_conn
    for lane_conn in nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]:
        if lane_conn.id in cross_blp_conns:
            lane_seg_conns += connect_trimmed_lane_conn_predecessor(lane_seg_coords, lane_conn, cross_blp_conns)
            lane_seg_conns += connect_trimmed_lane_conn_successor(lane_seg_coords, lane_conn, cross_blp_conns)
    return (lane_segment_coords_from_lane_segment_vector(lane_seg_coords), LaneSegmentConnections(lane_seg_conns), LaneSegmentGroupings(lane_seg_groupings), LaneSegmentLaneIDs(lane_seg_lane_ids), LaneSegmentRoadBlockIDs(lane_seg_roadblock_ids))

def get_on_route_status(route_roadblock_ids: List[str], roadblock_ids: LaneSegmentRoadBlockIDs) -> LaneOnRouteStatusData:
    """
    Identify whether given lane segments lie within goal route.
    :param route_roadblock_ids: List of ids of roadblocks (lane groups) within goal route.
    :param roadblock_ids: Roadblock ids (lane group associations) pertaining to associated lane segments.
    :return on_route_status: binary encoding of on route status for each input roadblock id.
    """
    if route_roadblock_ids:
        route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids, set(roadblock_ids.roadblock_ids))
        on_route_status = np.full((len(roadblock_ids.roadblock_ids), len(OnRouteStatusType) - 1), LaneOnRouteStatusData.encode(OnRouteStatusType.OFF_ROUTE))
        on_route_indices = np.arange(on_route_status.shape[0])[np.in1d(roadblock_ids.roadblock_ids, route_roadblock_ids)]
        on_route_status[on_route_indices] = LaneOnRouteStatusData.encode(OnRouteStatusType.ON_ROUTE)
    else:
        on_route_status = np.full((len(roadblock_ids.roadblock_ids), len(OnRouteStatusType) - 1), LaneOnRouteStatusData.encode(OnRouteStatusType.UNKNOWN))
    return LaneOnRouteStatusData(list(map(tuple, on_route_status)))

def get_traffic_light_encoding(lane_seg_ids: LaneSegmentLaneIDs, traffic_light_data: List[TrafficLightStatusData]) -> LaneSegmentTrafficLightData:
    """
    Encode the lane segments with traffic light data.
    :param lane_seg_ids: The lane_segment ids [num_lane_segment].
    :param traffic_light_data: A list of all available data at the current time step.
    :returns: Encoded traffic light data per segment.
    """
    traffic_light_encoding = np.full((len(lane_seg_ids.lane_ids), len(TrafficLightStatusType)), LaneSegmentTrafficLightData.encode(TrafficLightStatusType.UNKNOWN))
    green_lane_connectors = [str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.GREEN]
    red_lane_connectors = [str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.RED]
    for tl_id in green_lane_connectors:
        indices = np.argwhere(np.array(lane_seg_ids.lane_ids) == tl_id)
        traffic_light_encoding[indices] = LaneSegmentTrafficLightData.encode(TrafficLightStatusType.GREEN)
    for tl_id in red_lane_connectors:
        indices = np.argwhere(np.array(lane_seg_ids.lane_ids) == tl_id)
        traffic_light_encoding[indices] = LaneSegmentTrafficLightData.encode(TrafficLightStatusType.RED)
    return LaneSegmentTrafficLightData(list(map(tuple, traffic_light_encoding)))

class VectorSetMapFeatureBuilder(ScriptableFeatureBuilder):
    """
    Feature builder for constructing map features in a vector set representation, similar to that of
        VectorNet ("VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation").
    """

    def __init__(self, map_features: List[str], max_elements: Dict[str, int], max_points: Dict[str, int], radius: float, interpolation_method: str) -> None:
        """
        Initialize vector set map builder with configuration parameters.
        :param map_features: name of map features to be extracted.
        :param max_elements: maximum number of elements to extract per feature layer.
        :param max_points: maximum number of points per feature to extract per feature layer.
        :param radius:  [m ]The query radius scope relative to the current ego-pose.
        :param interpolation_method: Interpolation method to apply when interpolating to maintain fixed size
            map elements.
        :return: Vector set map data including map element coordinates and traffic light status info.
        """
        super().__init__()
        self.map_features = map_features
        self.max_elements = max_elements
        self.max_points = max_points
        self.radius = radius
        self.interpolation_method = interpolation_method
        self._traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()
        for feature_name in self.map_features:
            try:
                VectorFeatureLayer[feature_name]
            except KeyError:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')
            if feature_name not in self.max_elements:
                raise RuntimeError(f'Max elements unavailable for {feature_name} feature layer!')
            if feature_name not in self.max_points:
                raise RuntimeError(f'Max points unavailable for {feature_name} feature layer!')

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return VectorSetMap

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'vector_set_map'

    @torch.jit.unused
    def get_scriptable_input_from_scenario(self, scenario: AbstractScenario) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
        coords, traffic_light_data = get_neighbor_vector_set_map(scenario.map_api, self.map_features, ego_coords, self.radius, route_roadblock_ids, [TrafficLightStatuses(traffic_light_data)])
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(coords, traffic_light_data[0], ego_state.rear_axle)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_scriptable_input_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the simulation objects
        :param current_input: planner input from sim
        :param initialization: planner initialization from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = current_input.history.ego_states[-1]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = initialization.route_roadblock_ids
        if current_input.traffic_light_data is None:
            raise ValueError('Cannot build VectorSetMap feature. PlannerInput.traffic_light_data is None')
        traffic_light_data = current_input.traffic_light_data
        coords, traffic_light_data = get_neighbor_vector_set_map(initialization.map_api, self.map_features, ego_coords, self.radius, route_roadblock_ids, [TrafficLightStatuses(traffic_light_data)])
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(coords, traffic_light_data[0], ego_state.rear_axle)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_scenario(scenario)
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_simulation(current_input, initialization)
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> VectorSetMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorSetMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorSetMap.
        """
        coords: Dict[str, List[FeatureDataType]] = {}
        traffic_light_data: Dict[str, List[FeatureDataType]] = {}
        availabilities: Dict[str, List[FeatureDataType]] = {}
        for key in list_tensor_data:
            if key.startswith('vector_set_map.coords.'):
                feature_name = key[len('vector_set_map.coords.'):]
                coords[feature_name] = [list_tensor_data[key][0].detach().numpy()]
            if key.startswith('vector_set_map.traffic_light_data.'):
                feature_name = key[len('vector_set_map.traffic_light_data.'):]
                traffic_light_data[feature_name] = [list_tensor_data[key][0].detach().numpy()]
            if key.startswith('vector_set_map.availabilities.'):
                feature_name = key[len('vector_set_map.availabilities.'):]
                availabilities[feature_name] = [list_tensor_data[key][0].detach().numpy()]
        return VectorSetMap(coords=coords, traffic_light_data=traffic_light_data, availabilities=availabilities)

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(self, coords: Dict[str, MapObjectPolylines], traffic_light_data: Dict[str, LaneSegmentTrafficLightData], anchor_state: StateSE2) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature transform.
        :param coords: Dictionary mapping feature name to polyline vector sets.
        :param traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
        :param anchor_state: The ego state to transform to vector.
        :return
           tensor_data: Packed tensor data.
           list_tensor_data: Packed List[tensor] data.
           list_list_tensor_data: Packed List[List[tensor]] data.
        """
        tensor_data: Dict[str, torch.Tensor] = {}
        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)
        tensor_data['anchor_state'] = anchor_state_tensor
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        for feature_name, feature_coords in coords.items():
            list_feature_coords: List[torch.Tensor] = []
            for element_coords in feature_coords.to_vector():
                list_feature_coords.append(torch.tensor(element_coords, dtype=torch.float64))
            list_tensor_data[f'coords.{feature_name}'] = list_feature_coords
            if feature_name in traffic_light_data:
                list_feature_tl_data: List[torch.Tensor] = []
                for element_tl_data in traffic_light_data[feature_name].to_vector():
                    list_feature_tl_data.append(torch.tensor(element_tl_data, dtype=torch.float32))
                list_tensor_data[f'traffic_light_data.{feature_name}'] = list_feature_tl_data
        return (tensor_data, list_tensor_data, {})

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        tensor_output: Dict[str, torch.Tensor] = {}
        list_tensor_output: Dict[str, List[torch.Tensor]] = {}
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {}
        anchor_state = tensor_data['anchor_state']
        for feature_name in self.map_features:
            if f'coords.{feature_name}' in list_tensor_data:
                feature_coords = list_tensor_data[f'coords.{feature_name}']
                feature_tl_data = [list_tensor_data[f'traffic_light_data.{feature_name}']] if f'traffic_light_data.{feature_name}' in list_tensor_data else None
                coords, tl_data, avails = convert_feature_layer_to_fixed_size(feature_coords, feature_tl_data, self.max_elements[feature_name], self.max_points[feature_name], self._traffic_light_encoding_dim, interpolation=self.interpolation_method if feature_name in [VectorFeatureLayer.LANE.name, VectorFeatureLayer.LEFT_BOUNDARY.name, VectorFeatureLayer.RIGHT_BOUNDARY.name, VectorFeatureLayer.ROUTE_LANES.name] else None)
                coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)
                list_tensor_output[f'vector_set_map.coords.{feature_name}'] = [coords]
                list_tensor_output[f'vector_set_map.availabilities.{feature_name}'] = [avails]
                if tl_data is not None:
                    list_tensor_output[f'vector_set_map.traffic_light_data.{feature_name}'] = [tl_data[0]]
        return (tensor_output, list_tensor_output, list_list_tensor_output)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        max_elements: List[str] = [f'{feature_name}.{feature_max_elements}' for feature_name, feature_max_elements in self.max_elements.items()]
        max_points: List[str] = [f'{feature_name}.{feature_max_points}' for feature_name, feature_max_points in self.max_points.items()]
        return {'neighbor_vector_set_map': {'radius': str(self.radius), 'interpolation_method': self.interpolation_method, 'map_features': ','.join(self.map_features), 'max_elements': ','.join(max_elements), 'max_points': ','.join(max_points)}, 'initial_ego_state': empty}

def get_neighbor_vector_set_map(map_api: AbstractMap, map_features: List[str], point: Point2D, radius: float, route_roadblock_ids: List[str], traffic_light_statuses_over_time: List[TrafficLightStatuses]) -> Tuple[Dict[str, MapObjectPolylines], List[Dict[str, LaneSegmentTrafficLightData]]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param route_roadblock_ids: List of ids of roadblocks/roadblock connectors (lane groups) within goal route.
    :param traffic_light_statuses_over_time: A list of available traffic light statuses data, indexed by time step.
    :return:
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """
    coords: Dict[str, MapObjectPolylines] = {}
    feature_layers: List[VectorFeatureLayer] = []
    traffic_light_data_over_time: List[Dict[str, LaneSegmentTrafficLightData]] = []
    for feature_name in map_features:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(f'Object representation for layer: {feature_name} is unavailable')
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(map_api, point, radius)
        coords[VectorFeatureLayer.LANE.name] = lanes_mid
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lanes_left.polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lanes_right.polylines)
        for traffic_lights in traffic_light_statuses_over_time:
            traffic_light_data_at_t: Dict[str, LaneSegmentTrafficLightData] = {}
            traffic_light_data_at_t[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(lane_ids, traffic_lights.traffic_lights)
            traffic_light_data_over_time.append(traffic_light_data_at_t)
    if VectorFeatureLayer.ROUTE_LANES in feature_layers:
        route_polylines = get_route_lane_polylines_from_roadblock_ids(map_api, point, radius, route_roadblock_ids)
        coords[VectorFeatureLayer.ROUTE_LANES.name] = route_polylines
    for feature_layer in feature_layers:
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
            polygons = get_map_object_polygons(map_api, point, radius, VectorFeatureLayerMapping.semantic_map_layer(feature_layer))
            coords[feature_layer.name] = polygons
    return (coords, traffic_light_data_over_time)

def lane_segment_coords_from_lane_segment_vector(coords: List[List[List[float]]]) -> LaneSegmentCoords:
    """
    Convert lane segment coords [N, 2, 2] to nuPlan LaneSegmentCoords.
    :param coords: lane segment coordinates in vector form.
    :return: lane segment coordinates as LaneSegmentCoords.
    """
    return LaneSegmentCoords([(Point2D(*start), Point2D(*end)) for start, end in coords])

def get_lane_polylines(map_api: AbstractMap, point: Point2D, radius: float) -> Tuple[MapObjectPolylines, MapObjectPolylines, MapObjectPolylines, LaneSegmentLaneIDs]:
    """
    Extract ids, baseline path polylines, and boundary polylines of neighbor lanes and lane connectors around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :return:
        lanes_mid: extracted lane/lane connector baseline polylines.
        lanes_left: extracted lane/lane connector left boundary polylines.
        lanes_right: extracted lane/lane connector right boundary polylines.
        lane_ids: ids of lanes/lane connector associated polylines were extracted from.
    """
    lanes_mid: List[List[Point2D]] = []
    lanes_left: List[List[Point2D]] = []
    lanes_right: List[List[Point2D]] = []
    lane_ids: List[str] = []
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)
    map_objects: List[MapObject] = []
    for layer_name in layer_names:
        map_objects += layers[layer_name]
    map_objects.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))
    for map_obj in map_objects:
        baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.baseline_path.discrete_path]
        lanes_mid.append(baseline_path_polyline)
        lanes_left.append([Point2D(node.x, node.y) for node in map_obj.left_boundary.discrete_path])
        lanes_right.append([Point2D(node.x, node.y) for node in map_obj.right_boundary.discrete_path])
        lane_ids.append(map_obj.id)
    return (MapObjectPolylines(lanes_mid), MapObjectPolylines(lanes_left), MapObjectPolylines(lanes_right), LaneSegmentLaneIDs(lane_ids))

def get_map_object_polygons(map_api: AbstractMap, point: Point2D, radius: float, layer_name: SemanticMapLayer) -> MapObjectPolylines:
    """
    Extract polygons of neighbor map object around ego vehicle for specified semantic layers.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :param layer_name: semantic layer to query.
    :return extracted map object polygons.
    """
    map_objects = map_api.get_proximal_map_objects(point, radius, [layer_name])[layer_name]
    map_objects.sort(key=lambda map_obj: get_distance_between_map_object_and_point(point, map_obj))
    polygons = [extract_polygon_from_map_object(map_obj) for map_obj in map_objects]
    return MapObjectPolylines(polygons)

def get_route_polygon_from_roadblock_ids(map_api: AbstractMap, point: Point2D, radius: float, route_roadblock_ids: List[str]) -> MapObjectPolylines:
    """
    Extract route polygon from map for route specified by list of roadblock ids. Polygon is represented as collection of
        polygons of roadblocks/roadblock connectors encompassing route.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :param route_roadblock_ids: ids of roadblocks/roadblock connectors specifying route.
    :return: A route as sequence of roadblock/roadblock connector polygons.
    """
    route_polygons: List[List[Point2D]] = []
    layer_names = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)
    roadblock_ids: Set[str] = set()
    for layer_name in layer_names:
        roadblock_ids = roadblock_ids.union({map_object.id for map_object in layers[layer_name]})
    route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids, roadblock_ids)
    for route_roadblock_id in route_roadblock_ids:
        roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK)
        if not roadblock_obj:
            roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if roadblock_obj:
            polygon = extract_polygon_from_map_object(roadblock_obj)
            route_polygons.append(polygon)
    return MapObjectPolylines(route_polygons)

def prune_route_by_connectivity(route_roadblock_ids: List[str], roadblock_ids: Set[str]) -> List[str]:
    """
    Prune route by overlap with extracted roadblock elements within query radius to maintain connectivity in route
    feature. Assumes route_roadblock_ids is ordered and connected to begin with.
    :param route_roadblock_ids: List of roadblock ids representing route.
    :param roadblock_ids: Set of ids of extracted roadblocks within query radius.
    :return: List of pruned roadblock ids (connected and within query radius).
    """
    pruned_route_roadblock_ids: List[str] = []
    route_start = False
    for roadblock_id in route_roadblock_ids:
        if roadblock_id in roadblock_ids:
            pruned_route_roadblock_ids.append(roadblock_id)
            route_start = True
        elif route_start:
            break
    return pruned_route_roadblock_ids

def get_route_lane_polylines_from_roadblock_ids(map_api: AbstractMap, point: Point2D, radius: float, route_roadblock_ids: List[str]) -> MapObjectPolylines:
    """
    Extract route polylines from map for route specified by list of roadblock ids. Route is represented as collection of
        baseline polylines of all children lane/lane connectors or roadblock/roadblock connectors encompassing route.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :param route_roadblock_ids: ids of roadblocks/roadblock connectors specifying route.
    :return: A route as sequence of lane/lane connector polylines.
    """
    route_lane_polylines: List[List[Point2D]] = []
    map_objects = []
    layer_names = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)
    roadblock_ids: Set[str] = set()
    for layer_name in layer_names:
        roadblock_ids = roadblock_ids.union({map_object.id for map_object in layers[layer_name]})
    route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids, roadblock_ids)
    for route_roadblock_id in route_roadblock_ids:
        roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK)
        if not roadblock_obj:
            roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if roadblock_obj:
            map_objects += roadblock_obj.interior_edges
    map_objects.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))
    for map_obj in map_objects:
        baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.baseline_path.discrete_path]
        route_lane_polylines.append(baseline_path_polyline)
    return MapObjectPolylines(route_lane_polylines)

class TestVectorUtils(unittest.TestCase):
    """Test vector building utility functions."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        scenario = MockAbstractScenario()
        ego_state = scenario.initial_ego_state
        self.ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        self.map_api = scenario.map_api
        self.route_roadblock_ids = scenario.get_route_roadblock_ids()
        self.traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
        self.radius = 35
        self.map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        self._num_past_poses = 1
        self._past_time_horizon = 1.0
        self._num_future_poses = 5
        self._future_time_horizon = 5.0
        current_tl = [TrafficLightStatuses(list(scenario.get_traffic_light_status_at_iteration(iteration=0)))]
        past_tl = scenario.get_past_traffic_light_status_history(iteration=0, num_samples=self._num_past_poses, time_horizon=self._past_time_horizon)
        future_tl = scenario.get_future_traffic_light_status_history(iteration=0, num_samples=self._num_future_poses, time_horizon=self._future_time_horizon)
        past_tl_list = list(past_tl)
        future_tl_list = list(future_tl)
        self.traffic_light_data_over_time = past_tl_list + current_tl + future_tl_list

    def test_prune_route_by_connectivity(self) -> None:
        """
        Test pruning route roadblock ids by those within query radius (specified in roadblock_ids)
        maintaining connectivity.
        """
        route_roadblock_ids = ['-1', '0', '1', '2', '3']
        roadblock_ids = {'0', '1', '3'}
        pruned_route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids, roadblock_ids)
        self.assertEqual(pruned_route_roadblock_ids, ['0', '1'])

    def test_get_lane_polylines(self) -> None:
        """
        Test extracting lane/lane connector baseline path and boundary polylines from given map api.
        """
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(self.map_api, self.ego_coords, self.radius)
        assert type(lanes_mid) == MapObjectPolylines
        assert type(lanes_left) == MapObjectPolylines
        assert type(lanes_right) == MapObjectPolylines
        assert type(lane_ids) == LaneSegmentLaneIDs

    def test_get_map_object_polygons(self) -> None:
        """
        Test extracting map object polygons from map.
        """
        for layer in [SemanticMapLayer.CROSSWALK, SemanticMapLayer.STOP_LINE]:
            polygons = get_map_object_polygons(self.map_api, self.ego_coords, self.radius, layer)
            assert type(polygons) == MapObjectPolylines

    def test_get_route_polygon_from_roadblock_ids(self) -> None:
        """
        Test extracting route polygon from map given list of roadblock ids.
        """
        route = get_route_polygon_from_roadblock_ids(self.map_api, self.ego_coords, self.radius, self.route_roadblock_ids)
        assert type(route) == MapObjectPolylines

    def test_get_route_lane_polylines_from_roadblock_ids(self) -> None:
        """
        Test extracting route lane polylines from map given list of roadblock ids.
        """
        route = get_route_lane_polylines_from_roadblock_ids(self.map_api, self.ego_coords, self.radius, self.route_roadblock_ids)
        assert type(route) == MapObjectPolylines

    def test_get_on_route_status(self) -> None:
        """
        Test identifying whether given roadblock lie within goal route.
        """
        route_roadblock_ids = ['0']
        roadblock_ids = LaneSegmentRoadBlockIDs(['0', '1'])
        on_route_status = get_on_route_status(route_roadblock_ids, roadblock_ids)
        assert type(on_route_status) == LaneOnRouteStatusData
        assert len(on_route_status.on_route_status) == LaneOnRouteStatusData.encoding_dim()
        assert on_route_status.on_route_status[0] == on_route_status.encode(OnRouteStatusType.ON_ROUTE)
        assert on_route_status.on_route_status[1] == on_route_status.encode(OnRouteStatusType.OFF_ROUTE)

    def test_get_neighbor_vector_map(self) -> None:
        """
        Test extracting neighbor vector map information from map api.
        """
        lane_seg_coords, lane_seg_conns, lane_seg_groupings, lane_seg_lane_ids, lane_seg_roadblock_ids = get_neighbor_vector_map(self.map_api, self.ego_coords, self.radius)
        assert type(lane_seg_coords) == LaneSegmentCoords
        assert type(lane_seg_conns) == LaneSegmentConnections
        assert type(lane_seg_groupings) == LaneSegmentGroupings
        assert type(lane_seg_lane_ids) == LaneSegmentLaneIDs
        assert type(lane_seg_roadblock_ids) == LaneSegmentRoadBlockIDs

    def test_get_neighbor_vector_set_map(self) -> None:
        """
        Test extracting neighbor vector set map information from map api.
        """
        coords, traffic_light_data = get_neighbor_vector_set_map(self.map_api, self.map_features, self.ego_coords, self.radius, self.route_roadblock_ids, [TrafficLightStatuses(self.traffic_light_data)])
        for feature_name in self.map_features:
            assert feature_name in coords
            assert type(coords[feature_name]) == MapObjectPolylines
        assert len(traffic_light_data) == 1
        assert 'LANE' in traffic_light_data[0]
        assert type(traffic_light_data[0]['LANE']) == LaneSegmentTrafficLightData

    def test_get_neighbor_vector_set_map_for_time_horizon(self) -> None:
        """
        Test extracting neighbor vector set map information from map api.
        """
        coords, traffic_light_data_list = get_neighbor_vector_set_map(self.map_api, self.map_features, self.ego_coords, self.radius, self.route_roadblock_ids, self.traffic_light_data_over_time)
        for feature_name in self.map_features:
            assert feature_name in coords
            assert type(coords[feature_name]) == MapObjectPolylines
        for traffic_light_data in traffic_light_data_list:
            assert 'LANE' in traffic_light_data
            assert type(traffic_light_data['LANE']) == LaneSegmentTrafficLightData

@dataclass
class TrafficLightPlot(BaseScenarioPlot):
    """A dataclass for traffic light plot."""
    data_sources: Dict[int, ColumnDataSource] = field(default_factory=dict)
    plot: Optional[GlyphRenderer] = None

    def update_plot(self, main_figure: Figure, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param frame_index: Frame index.
        :param doc: The Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return
        self.render_event.set()
        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                data_sources = dict(self.data_sources[frame_index].data)
                if self.plot is None:
                    self.plot = main_figure.multi_line(xs='xs', ys='ys', line_color='line_colors', line_alpha='line_color_alphas', line_width=3.0, line_dash='dashed', source=data_sources)
                else:
                    self.plot.data_source.data = data_sources
                self.render_event.clear()
            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(self, scenario: AbstractScenario, history: SimulationHistory, lane_connectors: Dict[str, LaneConnector]) -> None:
        """
        Update traffic light status datasource of each frame.
        :param scenario: Scenario traffic light status information.
        :param history: SimulationHistory time-series data.
        :param lane_connectors: Lane connectors.
        """
        if not self.data_source_condition:
            return
        with self.data_source_condition:
            for frame_index in range(len(history.data)):
                traffic_light_status = history.data[frame_index].traffic_light_status
                traffic_light_map_line = TrafficLightMapLine(point_2d=[], line_colors=[], line_color_alphas=[])
                lane_connector_colors = simulation_map_layer_color[SemanticMapLayer.LANE_CONNECTOR]
                for traffic_light in traffic_light_status:
                    lane_connector = lane_connectors.get(str(traffic_light.lane_connector_id), None)
                    if lane_connector is not None:
                        path = lane_connector.baseline_path.discrete_path
                        points = [Point2D(x=pose.x, y=pose.y) for pose in path]
                        traffic_light_map_line.line_colors.append(traffic_light.status.name)
                        traffic_light_map_line.line_color_alphas.append(lane_connector_colors['line_color_alpha'])
                        traffic_light_map_line.point_2d.append(points)
                line_source = ColumnDataSource(dict(xs=traffic_light_map_line.line_xs, ys=traffic_light_map_line.line_ys, line_colors=traffic_light_map_line.line_colors, line_color_alphas=traffic_light_map_line.line_color_alphas))
                self.data_sources[frame_index] = line_source
                self.data_source_condition.notify()

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

@nuplan_test(path='json/route_extractor/route_extractor.json')
def test_get_route_and_simplify(scene: Dict[str, Any]) -> None:
    """
    Test getting route from ego pose and simplifying.
    """
    map_api = map_factory.build_map_from_name(scene['map']['area'])
    poses = []
    for marker in scene['markers']:
        poses.append(Point2D(*marker['pose'][:2]))
    expert_route = get_route(map_api=map_api, poses=poses)
    assert len(expert_route) == len(poses)
    all_route_obj = [map_object for map_objects in expert_route for map_object in map_objects]
    assert len(all_route_obj) == len(poses)
    route_simplified = get_route_simplified(expert_route)
    assert len(route_simplified) == 3

class PerFrameProgressAlongRouteComputer:
    """Class that computes progress per frame along a route."""

    def __init__(self, route_roadblocks: RouteRoadBlockLinkedList):
        """Class initializer
        :param route_roadblocks: A route roadblock linked list.
        """
        self.curr_roadblock_pair = route_roadblocks.head
        self.progress = [float(0)]
        self.prev_distance_to_start = float(0)
        self.next_roadblock_pair: Optional[RouteBaselineRoadBlockPair] = None
        self.skipped_roadblock_pair: Optional[RouteBaselineRoadBlockPair] = None

    @staticmethod
    def get_some_baseline_point(baseline: PolylineMapObject, ind: str) -> Optional[Point2D]:
        """Gets the first or last point on a given baselinePath
        :param baseline: A baseline path
        :param ind: Either 'last' or 'first' strings to show which point function should return
        :return: A point.
        """
        if ind == 'last':
            return Point2D(baseline.linestring.xy[0][-1], baseline.linestring.xy[1][-1])
        elif ind == 'first':
            return Point2D(baseline.linestring.xy[0][0], baseline.linestring.xy[1][0])
        else:
            raise ValueError('invalid position argument')

    def compute_progress_for_skipped_road_block(self) -> float:
        """Computes progress for skipped road_blocks (when ego pose exits one road block in a route and it does not
        enter the next one)
        :return: progress_for_skipped_roadblock
        """
        assert self.next_roadblock_pair is not None
        if self.skipped_roadblock_pair:
            prev_roadblock_last_point = self.get_some_baseline_point(self.skipped_roadblock_pair.base_line, 'last')
        else:
            prev_roadblock_last_point = self.get_some_baseline_point(self.curr_roadblock_pair.base_line, 'last')
        self.skipped_roadblock_pair = self.next_roadblock_pair
        skipped_distance_to_start = get_distance_of_closest_baseline_point_to_its_start(self.skipped_roadblock_pair.base_line, prev_roadblock_last_point)
        self.next_roadblock_pair = self.next_roadblock_pair.next
        next_roadblock_first_point = self.get_some_baseline_point(self.next_roadblock_pair.base_line, 'first')
        next_baseline_start_dist_to_skipped = get_distance_of_closest_baseline_point_to_its_start(self.skipped_roadblock_pair.base_line, next_roadblock_first_point)
        progress_for_skipped_roadblock: float = next_baseline_start_dist_to_skipped - skipped_distance_to_start
        return progress_for_skipped_roadblock

    def get_progress_including_skipped_roadblocks(self, ego_pose: Point2D, progress_for_skipped_roadblock: float) -> float:
        """Computes ego's progress when it first enters a new road-block in the route by considering possible progress
        for roadblocks it has skipped as multi_block_progress = (progress along the baseline of prev ego roadblock)
        + (progress along the baseline of the roadblock ego is in now) + (progress along skipped roadblocks if any).
        :param ego_pose: ego pose
        :param progress_for_skipped_roadblock: Prgoress for skipped road_blocks (zero if no roadblocks is skipped)
        :return: multi_block_progress
        """
        assert self.next_roadblock_pair is not None
        progress_in_prev_roadblock = self.curr_roadblock_pair.base_line.linestring.length - self.prev_distance_to_start
        prev_roadblock_last_point = self.get_some_baseline_point(self.curr_roadblock_pair.base_line, 'last')
        self.curr_roadblock_pair = self.next_roadblock_pair
        distance_to_start = get_distance_of_closest_baseline_point_to_its_start(self.curr_roadblock_pair.base_line, ego_pose)
        last_baseline_point_dist_to_start = get_distance_of_closest_baseline_point_to_its_start(self.curr_roadblock_pair.base_line, prev_roadblock_last_point)
        progress_in_new_roadblock = distance_to_start - last_baseline_point_dist_to_start
        multi_block_progress = progress_in_prev_roadblock + progress_in_new_roadblock + progress_for_skipped_roadblock
        self.prev_distance_to_start = distance_to_start
        return float(multi_block_progress)

    def get_multi_block_progress(self, ego_pose: Point2D) -> float:
        """When ego pose exits previous roadblock this function takes next road blocks in the expert route one by one
        until it finds one (if any) that pose belongs to. Once found, ego progress for multiple roadblocks including
        possible skipped roadblocks is computed and returned
        :param ego_pose: ego pose
        :return: multi block progress
        """
        multi_block_progress = float(0)
        progress_for_skipped_roadblocks = float(0)
        self.next_roadblock_pair = self.curr_roadblock_pair.next
        self.skipped_roadblock_pair = None
        while self.next_roadblock_pair is not None:
            if self.next_roadblock_pair.road_block.contains_point(ego_pose):
                multi_block_progress = self.get_progress_including_skipped_roadblocks(ego_pose, progress_for_skipped_roadblocks)
                break
            elif not self.next_roadblock_pair.next:
                break
            else:
                progress_for_skipped_roadblocks += self.compute_progress_for_skipped_road_block()
        return multi_block_progress

    def __call__(self, ego_poses: List[Point2D]) -> List[float]:
        """
        Computes per frame progress along the route baselines for ego poses
        :param ego_poses: ego poses
        :return: progress along the route.
        """
        self.prev_distance_to_start = get_distance_of_closest_baseline_point_to_its_start(self.curr_roadblock_pair.base_line, ego_poses[0])
        for ego_pose in ego_poses[1:]:
            if self.curr_roadblock_pair.road_block.contains_point(ego_pose):
                distance_to_start = get_distance_of_closest_baseline_point_to_its_start(self.curr_roadblock_pair.base_line, ego_pose)
                self.progress.append(distance_to_start - self.prev_distance_to_start)
                self.prev_distance_to_start = distance_to_start
            else:
                multi_block_progress = self.get_multi_block_progress(ego_pose)
                self.progress.append(multi_block_progress)
        return self.progress

class EgoStopAtStopLineStatistics(ViolationMetricBase):
    """
    Ego stopped at stop line metric.
    """

    def __init__(self, name: str, category: str, max_violation_threshold: int, distance_threshold: float, velocity_threshold: float) -> None:
        """
        Initializes the EgoProgressAlongExpertRouteStatistics class
        Rule formulation: 1. Get the nearest stop polygon (less than the distance threshold).
                          2. Check if the stop polygon is in any lanes.
                          3. Check if front corners of ego cross the stop polygon.
                          4. Check if no any leading agents.
                          5. Get min_velocity(distance_stop_line) until the ego leaves the stop polygon.
        :param name: Metric name
        :param category: Metric category
        :param max_violation_threshold: Maximum threshold for the violation when computing the score
        :param distance_threshold: Distances between ego front side and stop line lower than this threshold
        assumed to be the first vehicle before the stop line
        :param velocity_threshold: Velocity threshold to consider an ego stopped.
        """
        super().__init__(name=name, category=category, max_violation_threshold=max_violation_threshold)
        self._distance_threshold = distance_threshold
        self._velocity_threshold = velocity_threshold
        self._stopping_velocity_data: List[VelocityData] = []
        self._previous_stop_polygon_fid: Optional[str] = None

    @staticmethod
    def get_nearest_stop_line(map_api: AbstractMap, ego_pose_front: LineString) -> Optional[Tuple[str, Polygon]]:
        """
        Retrieve the nearest stop polygon
        :param map_api: AbstractMap map api
        :param ego_pose_front: Ego pose front corner line
        :return Nearest stop polygon fid if distance is less than the threshold.
        """
        center_x, center_y = ego_pose_front.centroid.xy
        center = Point2D(center_x[0], center_y[0])
        if not map_api.is_in_layer(center, layer=SemanticMapLayer.LANE):
            return None
        stop_line_fid, distance = map_api.get_distance_to_nearest_map_object(center, SemanticMapLayer.STOP_LINE)
        if stop_line_fid is None:
            return None
        stop_line: StopLine = map_api.get_map_object(stop_line_fid, SemanticMapLayer.STOP_LINE)
        lane: Optional[Lane] = map_api.get_one_map_object(center, SemanticMapLayer.LANE)
        if lane is not None:
            return (stop_line_fid, stop_line.polygon if stop_line.polygon.intersects(lane.polygon) else None)
        return None

    @staticmethod
    def check_for_leading_agents(detections: Observation, ego_state: EgoState, map_api: AbstractMap) -> bool:
        """
        Get the nearest leading agent
        :param detections: Detection class
        :param ego_state: Ego in oriented box representation
        :param map_api: AbstractMap api
        :return True if there is a leading agent, False otherwise
        """
        if isinstance(detections, DetectionsTracks):
            if len(detections.tracked_objects.tracked_objects) == 0:
                return False
            ego_agent = ego_state.agent
            for index, box in enumerate(detections.tracked_objects):
                if box.token is None:
                    box.token = str(index + 1)
            scene_objects: List[SceneObject] = [ego_agent]
            scene_objects.extend([scene_object for scene_object in detections.tracked_objects])
            occupancy_map = STRTreeOccupancyMapFactory.get_from_boxes(scene_objects)
            agent_states = {scene_object.token: StateSE2(x=scene_object.center.x, y=scene_object.center.y, heading=scene_object.center.heading) for scene_object in scene_objects}
            ego_pose: StateSE2 = agent_states['ego']
            lane = map_api.get_one_map_object(ego_pose, SemanticMapLayer.LANE)
            ego_baseline = lane.baseline_path
            ego_progress = ego_baseline.get_nearest_arc_length_from_position(ego_pose)
            progress_path = create_path_from_se2(ego_baseline.discrete_path)
            ego_path_to_go = trim_path_up_to_progress(progress_path, ego_progress)
            ego_path_to_go = path_to_linestring(ego_path_to_go)
            intersecting_agents = occupancy_map.intersects(ego_path_to_go.buffer(scene_objects[0].box.width / 2, cap_style=CAP_STYLE.flat))
            if intersecting_agents.size > 1:
                return True
        return False

    def _compute_velocity_statistics(self, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Compute statistics in each stop line
        :param scenario: Scenario running this metric
        :return A list of metric statistics.
        """
        if not self._stopping_velocity_data:
            return []
        mean_ego_min_distance_to_stop_line = []
        mean_ego_min_velocity_before_stop_line = []
        aggregated_timestamp_velocity = []
        aggregated_timestamps = []
        ego_stop_status = []
        for velocity_data in self._stopping_velocity_data:
            min_distance_velocity_record = velocity_data.min_distance_stop_line_record
            mean_ego_min_distance_to_stop_line.append(min_distance_velocity_record.distance_to_stop_line)
            mean_ego_min_velocity_before_stop_line.append(min_distance_velocity_record.velocity)
            if min_distance_velocity_record.distance_to_stop_line < self._distance_threshold and min_distance_velocity_record.velocity < self._velocity_threshold:
                stop_status = True
            else:
                stop_status = False
            ego_stop_status.append(stop_status)
            aggregated_timestamp_velocity.append(velocity_data.velocity_np)
            aggregated_timestamps.append(velocity_data.timestamp_np)
        statistics = [Statistic(name='number_of_ego_stop_before_stop_line', unit=MetricStatisticsType.COUNT.unit, value=sum(ego_stop_status), type=MetricStatisticsType.COUNT), Statistic(name='number_of_ego_before_stop_line', unit=MetricStatisticsType.COUNT.unit, value=len(ego_stop_status), type=MetricStatisticsType.COUNT), Statistic(name='mean_ego_min_distance_to_stop_line', unit='meters', value=float(np.mean(mean_ego_min_distance_to_stop_line)), type=MetricStatisticsType.VALUE), Statistic(name='mean_ego_min_velocity_before_stop_line', unit='meters_per_second_squared', value=float(np.mean(mean_ego_min_velocity_before_stop_line)), type=MetricStatisticsType.VALUE)]
        aggregated_timestamp_velocity = np.hstack(aggregated_timestamp_velocity)
        aggregated_timestamps = np.hstack(aggregated_timestamps)
        velocity_time_series = TimeSeries(unit='meters_per_second_squared', time_stamps=list(aggregated_timestamps), values=list(aggregated_timestamp_velocity))
        results = self._construct_metric_results(metric_statistics=statistics, time_series=velocity_time_series, scenario=scenario)
        return results

    def _save_stopping_velocity(self, current_stop_polygon_fid: str, history_data: SimulationHistorySample, stop_polygon_in_lane: Polygon, ego_pose_front: LineString) -> None:
        """
        Save velocity, timestamp and distance to a stop line if the ego is stopping
        :param current_stop_polygon_fid: Current stop polygon fid
        :param history_data: History sample data at current timestamp
        :param stop_polygon_in_lane: The stop polygon where the ego is in
        :param ego_pose_front: Front line string (front right corner and left corner) of the ego.
        """
        stop_line: LineString = LineString(stop_polygon_in_lane.exterior.coords[:2])
        distance_ego_front_stop_line = stop_line.distance(ego_pose_front)
        current_velocity = history_data.ego_state.dynamic_car_state.speed
        current_timestamp = history_data.ego_state.time_point.time_us
        if current_stop_polygon_fid == self._previous_stop_polygon_fid:
            self._stopping_velocity_data[-1].add_data(velocity=current_velocity, timestamp=current_timestamp, distance_to_stop_line=distance_ego_front_stop_line)
        else:
            self._previous_stop_polygon_fid = current_stop_polygon_fid
            velocity_data = VelocityData([])
            velocity_data.add_data(velocity=current_velocity, timestamp=current_timestamp, distance_to_stop_line=distance_ego_front_stop_line)
            self._stopping_velocity_data.append(velocity_data)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the ego stopped at stop line metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated ego stopped at stop line metric.
        """
        ego_states: List[EgoState] = history.extract_ego_state
        ego_pose_fronts: List[LineString] = [LineString([state.car_footprint.oriented_box.geometry.exterior.coords[0], state.car_footprint.oriented_box.geometry.exterior.coords[3]]) for state in ego_states]
        scenario_map: AbstractMap = history.map_api
        for ego_pose_front, ego_state, history_data in zip(ego_pose_fronts, ego_states, history.data):
            stop_polygon_info: Optional[Tuple[str, Polygon]] = self.get_nearest_stop_line(map_api=scenario_map, ego_pose_front=ego_pose_front)
            if stop_polygon_info is None:
                continue
            fid, stop_polygon_in_lane = stop_polygon_info
            ego_pose_front_stop_polygon_distance: float = ego_pose_front.distance(stop_polygon_in_lane)
            if ego_pose_front_stop_polygon_distance != 0:
                continue
            detections: Observation = history_data.observation
            has_leading_agent = self.check_for_leading_agents(detections=detections, ego_state=ego_state, map_api=scenario_map)
            if has_leading_agent:
                continue
            self._save_stopping_velocity(current_stop_polygon_fid=fid, history_data=history_data, stop_polygon_in_lane=stop_polygon_in_lane, ego_pose_front=ego_pose_front)
        results = self._compute_velocity_statistics(scenario=scenario)
        return results

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

