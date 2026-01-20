# Cluster 15

def extract_discrete_polyline(polyline: geom.LineString) -> List[StateSE2]:
    """
    Returns a discretized polyline composed of StateSE2 as nodes.
    :param polyline: the polyline of interest.
    :returns: linestring as a list of waypoints represented by StateSE2.
    """
    assert polyline.length > 0.0, 'The length of the polyline has to be greater than 0!'
    headings = compute_linestring_heading(polyline)
    x_coords, y_coords = polyline.coords.xy
    return [StateSE2(x, y, heading) for x, y, heading in zip(x_coords, y_coords, headings)]

def compute_linestring_heading(linestring: geom.linestring.LineString) -> List[float]:
    """
    Compute the heading of each coordinate to its successor coordinate. The last coordinate will have the same heading
        as the second last coordinate.
    :param linestring: linestring as a shapely LineString.
    :return: a list of headings associated to each starting coordinate.
    """
    coords: npt.NDArray[np.float64] = np.asarray(linestring.coords)
    vectors = np.diff(coords, axis=0)
    angles = np.arctan2(vectors.T[1], vectors.T[0])
    angles = np.append(angles, angles[-1])
    assert len(angles) == len(coords), 'Calculated heading must have the same length as input coordinates'
    return list(angles)

class NuPlanPolylineMapObject(PolylineMapObject):
    """
    NuPlanMap implementation of Polyline Map Object.
    """

    def __init__(self, polyline: Series, distance_for_curvature_estimation: float=2.0, distance_for_heading_estimation: float=0.5):
        """
        Constructor of polyline map layer.
        :param polyline: a pandas series representing the polyline.
        :param distance_for_curvature_estimation: [m] distance of the split between 3-points curvature estimation.
        :param distance_for_heading_estimation: [m] distance between two points on the polyline to calculate
                                                    the relative heading.
        """
        super().__init__(polyline['fid'])
        self._polyline: LineString = polyline.geometry
        assert self._polyline.length > 0.0, 'The length of the polyline has to be greater than 0!'
        self._distance_for_curvature_estimation = distance_for_curvature_estimation
        self._distance_for_heading_estimation = distance_for_heading_estimation

    @property
    def linestring(self) -> LineString:
        """Inherited from superclass."""
        return self._polyline

    @property
    def length(self) -> float:
        """Inherited from superclass."""
        return float(self._polyline.length)

    @cached_property
    def discrete_path(self) -> List[StateSE2]:
        """Inherited from superclass."""
        return cast(List[StateSE2], extract_discrete_polyline(self._polyline))

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """Inherited from superclass."""
        return self._polyline.project(Point(point.x, point.y))

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """Inherited from superclass."""
        arc_length = self.get_nearest_arc_length_from_position(point)
        state1 = self._polyline.interpolate(arc_length)
        state2 = self._polyline.interpolate(arc_length + self._distance_for_heading_estimation)
        if state1 == state2:
            state2 = self._polyline.interpolate(arc_length - self._distance_for_heading_estimation)
            heading = _get_heading(state2, state1)
        else:
            heading = _get_heading(state1, state2)
        return StateSE2(state1.x, state1.y, heading)

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """Inherited from superclass."""
        curvature = estimate_curvature_along_path(self._polyline, arc_length, self._distance_for_curvature_estimation)
        return float(curvature)

def _get_heading(pt1: Point, pt2: Point) -> float:
    """
    Computes the angle two points makes to the x-axis.
    :param pt1: origin point.
    :param pt2: end point.
    :return: [rad] resulting angle.
    """
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.atan2(y_diff, x_diff)

def test_split_blp_lane_segments() -> None:
    """
    Test splitting baseline paths node list into lane segments.
    """
    nodes = [StateSE2(0.0, 0.0, 0.0), StateSE2(0.0, 0.0, 0.0), StateSE2(0.0, 0.0, 0.0)]
    lane_seg_num = 2
    obj_coords = split_blp_lane_segments(nodes, lane_seg_num)
    assert len(obj_coords) == 2
    assert len(obj_coords[0]) == 2
    assert len(obj_coords[0][0]) == 2
    assert isinstance(obj_coords, List)
    assert isinstance(obj_coords[0], List)
    assert isinstance(obj_coords[0][0], List)
    assert isinstance(obj_coords[0][0][0], float)

class AgentState(SceneObject):
    """
    Class describing Agent State (including dynamics) in the scene, representing Vehicles, Bicycles and Pedestrians.
    """

    def __init__(self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, velocity: StateVector2D, metadata: SceneObjectMetadata, angular_velocity: Optional[float]=None):
        """
        Representation of an Agent in the scene (Vehicles, Pedestrians, Bicyclists and GenericObjects).
        :param tracked_object_type: Type of the current agent.
        :param oriented_box: Geometrical representation of the Agent.
        :param velocity: Velocity (vectorial) of Agent.
        :param metadata: Agent's metadata.
        :param angular_velocity: The scalar angular velocity of the agent, if available.
        """
        super().__init__(tracked_object_type=tracked_object_type, oriented_box=oriented_box, metadata=metadata)
        self._velocity = velocity
        self._angular_velocity = angular_velocity

    @property
    def velocity(self) -> StateVector2D:
        """
        Getter for velocity.
        :return: The agent vectorial velocity.
        """
        return self._velocity

    @property
    def angular_velocity(self) -> Optional[float]:
        """
        Getter for angular.
        :return: The agent angular velocity.
        """
        return self._angular_velocity

    @classmethod
    def from_new_pose(cls, agent: AgentState, pose: StateSE2) -> AgentState:
        """
        Initializer that create the same agent in a different pose.
        :param agent: A sample agent.
        :param pose: The new pose.
        :return: A new agent.
        """
        return AgentState(tracked_object_type=agent.tracked_object_type, oriented_box=OrientedBox.from_new_pose(agent.box, pose), velocity=agent.velocity, angular_velocity=agent.angular_velocity, metadata=copy.deepcopy(agent.metadata))

class CarFootprint(OrientedBox):
    """Class that represent the car semantically, with geometry and relevant point of interest."""

    def __init__(self, center: StateSE2, vehicle_parameters: VehicleParameters):
        """
        :param center: The pose of ego in the specified frame
        :param vehicle_parameters: The parameters of ego
        """
        super().__init__(center=center, width=vehicle_parameters.width, length=vehicle_parameters.length, height=vehicle_parameters.height)
        self._vehicle_parameters = vehicle_parameters

    @property
    def vehicle_parameters(self) -> VehicleParameters:
        """
        :return: vehicle parameters corresponding to the footprint
        """
        return self._vehicle_parameters

    def get_point_of_interest(self, point_of_interest: OrientedBoxPointType) -> Point2D:
        """
        Getter for the point of interest of ego.
        :param point_of_interest: The query point of the car
        :return: The position of the query point.
        """
        return self.corner(point_of_interest)

    @property
    def oriented_box(self) -> OrientedBox:
        """
        Getter for Ego's OrientedBox
        :return: OrientedBox of Ego
        """
        return self

    @property
    def rear_axle_to_center_dist(self) -> float:
        """
        Getter for the distance from the rear axle to the center of mass of Ego.
        :return: Distance from rear axle to COG
        """
        return float(self._vehicle_parameters.rear_axle_to_center)

    @cached_property
    def rear_axle(self) -> StateSE2:
        """
        Getter for the pose at the middle of the rear axle
        :return: SE2 Pose of the rear axle.
        """
        return translate_longitudinally(self.oriented_box.center, -self.rear_axle_to_center_dist)

    @classmethod
    def build_from_rear_axle(cls, rear_axle_pose: StateSE2, vehicle_parameters: VehicleParameters) -> CarFootprint:
        """
        Construct Car Footprint from rear axle position
        :param rear_axle_pose: SE2 position of rear axle
        :param vehicle_parameters: parameters of vehicle
        :return: CarFootprint
        """
        center = translate_longitudinally(rear_axle_pose, vehicle_parameters.rear_axle_to_center)
        return cls(center=center, vehicle_parameters=vehicle_parameters)

    @classmethod
    def build_from_cog(cls, cog_pose: StateSE2, vehicle_parameters: VehicleParameters) -> CarFootprint:
        """
        Construct Car Footprint from COG position
        :param cog_pose: SE2 position of COG
        :param vehicle_parameters: parameters of vehicle
        :return: CarFootprint
        """
        cog_to_center = vehicle_parameters.rear_axle_to_center - vehicle_parameters.cog_position_from_rear_axle
        center = translate_longitudinally(cog_pose, cog_to_center)
        return cls(center=center, vehicle_parameters=vehicle_parameters)

    @classmethod
    def build_from_center(cls, center: StateSE2, vehicle_parameters: VehicleParameters) -> CarFootprint:
        """
        Construct Car Footprint from geometric center of vehicle
        :param center: SE2 position of geometric center of vehicle
        :param vehicle_parameters: parameters of vehicle
        :return: CarFootprint
        """
        return cls(center=center, vehicle_parameters=vehicle_parameters)

def translate_longitudinally(pose: StateSE2, distance: float) -> StateSE2:
    """
    Translate an SE2 pose longitudinally (along heading direction)
    :param pose: SE2 pose to be translated
    :param distance: [m] distance by which point (x, y, heading) should be translated longitudinally
    :return translated se2
    """
    translation: npt.NDArray[np.float64] = np.array([distance * np.cos(pose.heading), distance * np.sin(pose.heading)])
    return translate(pose, translation)

def in_collision(box1: OrientedBox, box2: OrientedBox, radius_threshold: Optional[float]=None) -> bool:
    """
    Check for collision between two boxes. First do a quick check by approximating each box with a circle of given radius,
    if there is an overlap, check for the exact intersection using geometry Polygon
    :param box1: Oriented box (e.g., of ego)
    :param box2: Oriented box (e.g., of other tracks)
    :param radius: Radius for quick collision check
    :return True if there is a collision between the two boxes.
    """
    return bool(box1.geometry.intersects(box2.geometry)) if collision_by_radius_check(box1, box2, radius_threshold) else False

def collision_by_radius_check(box1: OrientedBox, box2: OrientedBox, radius_threshold: Optional[float]) -> bool:
    """
    Quick check for whether two boxes are in collision using a radius check, if radius_threshold is None, an over-approximated circle around each box is considered to determine the radius
    :param box1: Oriented box (e.g., of ego)
    :param box2: Oriented box (e.g., of other tracks)
    :param radius_threshold: Radius threshold for quick collision check
    :return False if the distance between centers of the two boxes is larger than radius_threshold else True. If radius_threshold is None, radius_threshold is defined as the sum of the radius of the smallest over-approximated circles around each box
    centered at the box center (i.e., the radius_threshold is defined when over-approximated circles are external tangents).
    """
    if not radius_threshold:
        w1, l1 = (box1.width, box1.length)
        w2, l2 = (box2.width, box2.length)
        radius_threshold = (np.hypot(w1, l1) + np.hypot(w2, l2)) / 2.0
    distance_between_centers = box1.center.distance_to(box2.center)
    return bool(distance_between_centers < radius_threshold)

class TestOrientedBox(unittest.TestCase):
    """Tests OrientedBox class"""

    def setUp(self) -> None:
        """Creates sample parameters for testing"""
        self.center = StateSE2(1, 2, m.pi / 8)
        self.length = 4.0
        self.width = 2.0
        self.height = 1.5
        self.expected_vertices = [(2.47, 3.69), (-1.23, 2.16), (-0.47, 0.31), (3.23, 1.84)]

    def test_construction(self) -> None:
        """Tests that the object is created correctly, including the polygon representing its geometry."""
        test_box = OrientedBox(self.center, self.length, self.width, self.height)
        self.assertTrue(self.center == test_box.center)
        self.assertEqual(self.length, test_box.length)
        self.assertEqual(self.width, test_box.width)
        self.assertEqual(self.height, test_box.height)
        self.assertFalse('geometry' in test_box.__dict__)
        for vertex, expected_vertex in zip(test_box.geometry.exterior.coords, self.expected_vertices):
            self.assertAlmostEqual(vertex[0], expected_vertex[0], 2)
            self.assertAlmostEqual(vertex[1], expected_vertex[1], 2)
        self.assertTrue('geometry' in test_box.__dict__)

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

def pose_from_matrix(transform_matrix: npt.NDArray[np.float32]) -> StateSE2:
    """
    Converts a 3x3 transformation matrix to a 2D pose
    :param transform_matrix: 3x3 transformation matrix
    :return: 2D pose (x, y, yaw)
    """
    if transform_matrix.shape != (3, 3):
        raise RuntimeError(f'Expected a 3x3 transformation matrix, got {transform_matrix.shape}')
    heading = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
    return StateSE2(transform_matrix[0, 2], transform_matrix[1, 2], heading)

def absolute_to_relative_poses(absolute_poses: List[StateSE2]) -> List[StateSE2]:
    """
    Converts a list of SE2 poses from absolute to relative coordinates with the first pose being the origin
    :param absolute_poses: list of absolute poses to convert
    :return: list of converted relative poses
    """
    absolute_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(pose) for pose in absolute_poses])
    origin_transform = np.linalg.inv(absolute_transforms[0])
    relative_transforms = origin_transform @ absolute_transforms
    relative_poses = [pose_from_matrix(transform_matrix) for transform_matrix in relative_transforms]
    return relative_poses

def matrix_from_pose(pose: StateSE2) -> npt.NDArray[np.float64]:
    """
    Converts a 2D pose to a 3x3 transformation matrix

    :param pose: 2D pose (x, y, yaw)
    :return: 3x3 transformation matrix
    """
    return np.array([[np.cos(pose.heading), -np.sin(pose.heading), pose.x], [np.sin(pose.heading), np.cos(pose.heading), pose.y], [0, 0, 1]])

def relative_to_absolute_poses(origin_pose: StateSE2, relative_poses: List[StateSE2]) -> List[StateSE2]:
    """
    Converts a list of SE2 poses from relative to absolute coordinates using an origin pose.
    :param origin_pose: Reference origin pose
    :param relative_poses: list of relative poses to convert
    :return: list of converted absolute poses
    """
    relative_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(pose) for pose in relative_poses])
    origin_transform = matrix_from_pose(origin_pose)
    absolute_transforms: npt.NDArray[np.float32] = origin_transform @ relative_transforms
    absolute_poses = [pose_from_matrix(transform_matrix) for transform_matrix in absolute_transforms]
    return absolute_poses

def numpy_array_to_absolute_velocity(origin_absolute_state: StateSE2, velocities: npt.NDArray[np.float32]) -> List[StateVector2D]:
    """
    Converts an array of relative numpy velocities to a list of absolute StateVector2D objects.
    :param velocities: list of velocities to convert
    :param origin_absolute_state: Reference origin pose
    :return: list of StateVector2D
    """
    assert velocities.shape[1] == 2, f'Expected poses shape of (*, 2), got {velocities.shape}'
    velocities = np.pad(velocities.astype(np.float64), ((0, 0), (0, 1)), 'constant', constant_values=0.0)
    relative_states = [StateSE2.deserialize(pose) for pose in velocities]
    return [StateVector2D(state.x, state.y) for state in relative_to_absolute_poses(origin_absolute_state, relative_states)]

def numpy_array_to_absolute_pose(origin_absolute_state: StateSE2, poses: npt.NDArray[np.float32]) -> List[StateSE2]:
    """
    Converts an array of relative numpy poses to a list of absolute StateSE2 objects.
    :param poses: list of poses to convert
    :param origin_absolute_state: Reference origin pose
    :return: list of StateSE2
    """
    assert poses.shape[1] == 3, f'Expected poses shape of (*, 3), got {poses.shape}'
    relative_states = [StateSE2.deserialize(pose) for pose in poses]
    return relative_to_absolute_poses(origin_absolute_state, relative_states)

def translate(pose: StateSE2, translation: npt.NDArray[np.float64]) -> StateSE2:
    """ "
    Applies a 2D translation
    :param pose: The pose to be transformed
    :param translation: The translation to be applied
    :return: The translated pose
    """
    assert translation.shape == (2,) or translation.shape == (2, 1)
    return StateSE2(pose.x + translation[0], pose.y + translation[1], pose.heading)

def rotate(pose: StateSE2, rotation_matrix: npt.NDArray[np.float64]) -> StateSE2:
    """
    Applies a 2D rotation to an SE2 Pose
    :param pose: The pose to be transformed
    :param rotation_matrix: The 2x2 rotation matrix representing the rotation
    :return: The rotated pose
    """
    assert rotation_matrix.shape == (2, 2)
    rotated_point = np.array([pose.x, pose.y]) @ rotation_matrix
    rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[1, 1])
    return StateSE2(rotated_point[0], rotated_point[1], pose.heading + rotation_angle)

def translate_laterally(pose: StateSE2, distance: float) -> StateSE2:
    """
    Translate an SE2 pose laterally
    :param pose: SE2 pose to be translated
    :param distance: [m] distance by which point (x, y, heading) should be translated longitudinally
    :return translated se2
    """
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.array([distance * np.cos(pose.heading + half_pi), distance * np.sin(pose.heading + half_pi)])
    return translate(pose, translation)

def signed_lateral_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    Computes the minimal lateral distance of ego from another polygon
    :param ego_state: the state of ego
    :param other: the query polygon
    :return: the signed lateral distance
    """
    ego_half_width = get_pacifica_parameters().half_width
    ego_left = translate_laterally(ego_state, ego_half_width)
    ego_right = translate_laterally(ego_state, -ego_half_width)
    vertices = list(zip(*other.exterior.coords.xy))
    distance_left = max(min((lateral_distance(ego_left, Point2D(*vertex)) for vertex in vertices)), 0)
    distance_right = max(min((-lateral_distance(ego_right, Point2D(*vertex)) for vertex in vertices)), 0)
    return distance_left if distance_left > distance_right else -distance_right

def lateral_distance(reference: StateSE2, other: Point2D) -> float:
    """
    Lateral distance from a point to a reference pose
    :param reference: the reference pose
    :param other: the query point
    :return: the lateral distance
    """
    return float(-np.sin(reference.heading) * (other.x - reference.x) + np.cos(reference.heading) * (other.y - reference.y))

def signed_longitudinal_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    Computes the minimal longitudinal distance of ego from another polygon
    :param ego_state: the state of ego
    :param other: the query polygon
    :return: the signed lateral distance
    """
    ego_half_length = get_pacifica_parameters().half_length
    ego_front = translate_longitudinally(ego_state, ego_half_length)
    ego_back = translate_longitudinally(ego_state, -ego_half_length)
    vertices = list(zip(*other.exterior.coords.xy))
    distance_front = max(min((longitudinal_distance(ego_front, Point2D(*vertex)) for vertex in vertices)), 0)
    distance_back = max(min((-longitudinal_distance(ego_back, Point2D(*vertex)) for vertex in vertices)), 0)
    return distance_front if distance_front > distance_back else -distance_back

def longitudinal_distance(reference: StateSE2, other: Point2D) -> float:
    """
    Longitudinal distance from a point to a reference pose
    :param reference: the reference pose
    :param other: the query point
    :return: the longitudinal distance
    """
    return float(np.cos(reference.heading) * (other.x - reference.x) + np.sin(reference.heading) * (other.y - reference.y))

def se2_box_distances(query: StateSE2, targets: list[StateSE2], box_size: Dimension, consider_flipped: bool=True) -> List[float]:
    """
    Computes the minimal distance [m] from a query to a list of targets. The distance is computed using the norm of the
    euclidean distances between the corners of a box spawned using the pose as center and given dimensions.
    The query box is also rotated by 180deg and the minimum of the two distances is used.
    :param query: The query pose.
    :param targets: The targets to compute the distance.
    :param box_size: The size of the box to be constructed.
    :param consider_flipped: Whether to also check for the same query pose, but rotated by 180 degrees.
    :return: A list of distances [m] from query to targets
    """
    query_box = OrientedBox(query, box_size.length, box_size.width, box_size.height)
    backwards_query_box = OrientedBox.from_new_pose(query_box, StateSE2(query.x, query.y, query.heading + np.pi))
    target_boxes = [OrientedBox(target, box_size.length, box_size.width, box_size.height) for target in targets]
    if consider_flipped:
        return [min(l2_euclidean_corners_distance(query_box, target_box), l2_euclidean_corners_distance(backwards_query_box, target_box)) for target_box in target_boxes]
    else:
        return [l2_euclidean_corners_distance(query_box, target_box) for target_box in target_boxes]

def l2_euclidean_corners_distance(box1: OrientedBox, box2: OrientedBox) -> float:
    """
    Computes the L2 norm [m] of the euclidean distance between the corners of an OrientedBox in two configurations.
    :param box1: The first box configuration.
    :param box2: The second box configuration.
    :return: [m] The norm of the euclidean distance.
    """
    distances = [np.linalg.norm(box1_corner.array - box2_corner.array) for box1_corner, box2_corner in zip(box1.all_corners(), box2.all_corners())]
    return float(np.linalg.norm(distances))

class TestConvert(unittest.TestCase):
    """Tests for convert functions"""

    def test_pose_from_matrix(self) -> None:
        """Tests conversion from 3x3 transformation matrix to a 2D pose"""
        transform_matrix: npt.NDArray[np.float32] = np.array([[np.sqrt(3) / 2, -0.5, 2], [0.5, np.sqrt(3) / 2, 2], [0, 0, 1]], dtype=np.float32)
        expected_pose = StateSE2(2, 2, np.pi / 6)
        result = pose_from_matrix(transform_matrix=transform_matrix)
        self.assertAlmostEqual(result.x, expected_pose.x)
        self.assertAlmostEqual(result.y, expected_pose.y)
        self.assertAlmostEqual(result.heading, expected_pose.heading)
        with self.assertRaises(RuntimeError):
            bad_matrix: npt.NDArray[np.float32] = np.array([[np.sqrt(3) / 2, -0.5, 2], [0.5, np.sqrt(3) / 2, 2]], dtype=np.float32)
            _ = pose_from_matrix(transform_matrix=bad_matrix)

    def test_matrix_from_pose(self) -> None:
        """Tests conversion from 2D pose to a 3x3 transformation matrix"""
        pose = StateSE2(2, 2, np.pi / 6)
        expected_transform_matrix: npt.NDArray[np.float32] = np.array([[np.sqrt(3) / 2, -0.5, 2], [0.5, np.sqrt(3) / 2, 2], [0, 0, 1]], dtype=np.float32)
        result = matrix_from_pose(pose=pose)
        np.testing.assert_array_almost_equal(result, expected_transform_matrix)

    def test_absolute_to_relative_poses(self) -> None:
        """Tests conversion of a list of SE2 poses from absolute to relative coordinates"""
        inv_sqrt_2 = 1 / np.sqrt(2)
        origin = StateSE2(1, 1, np.pi / 4)
        poses = [origin, StateSE2(1, 1, np.pi / 2), StateSE2(1, 1, np.pi / 4), StateSE2(2, 3, 0), StateSE2(3, 2, 0)]
        expected_poses = [StateSE2(0, 0, 0), StateSE2(0, 0, np.pi / 4), StateSE2(0, 0, 0), StateSE2(3 * inv_sqrt_2, inv_sqrt_2, -np.pi / 4), StateSE2(3 * inv_sqrt_2, -inv_sqrt_2, -np.pi / 4)]
        result = absolute_to_relative_poses(poses)
        for i in range(len(result)):
            self.assertAlmostEqual(result[i].x, expected_poses[i].x)
            self.assertAlmostEqual(result[i].y, expected_poses[i].y)
            self.assertAlmostEqual(result[i].heading, expected_poses[i].heading)

    def test_relative_to_absolute_poses(self) -> None:
        """Tests conversion of a list of SE2 poses from relative to absolute coordinates"""
        inv_sqrt_2 = 1 / np.sqrt(2)
        origin = StateSE2(1, 1, np.pi / 4)
        poses = [StateSE2(0, 0, np.pi / 4), StateSE2(0, 0, 0), StateSE2(3 * inv_sqrt_2, inv_sqrt_2, -np.pi / 4), StateSE2(3 * inv_sqrt_2, -inv_sqrt_2, -np.pi / 4)]
        expected_poses = [StateSE2(1, 1, np.pi / 2), StateSE2(1, 1, np.pi / 4), StateSE2(2, 3, 0), StateSE2(3, 2, 0)]
        result = relative_to_absolute_poses(origin, poses)
        for i in range(len(result)):
            self.assertAlmostEqual(result[i].x, expected_poses[i].x)
            self.assertAlmostEqual(result[i].y, expected_poses[i].y)
            self.assertAlmostEqual(result[i].heading, expected_poses[i].heading)

    def test_input_numpy_array_to_absolute_velocity(self) -> None:
        """Tests input validation of numpy_array_to_absolute_velocity"""
        np_velocities = np.random.random(size=(10, 3))
        with self.assertRaises(AssertionError):
            numpy_array_to_absolute_velocity(StateSE2(0, 0, 0), np_velocities)

    @patch('nuplan.common.geometry.convert.relative_to_absolute_poses')
    def test_numpy_array_to_absolute_velocity(self, mock_relative_to_absolute_poses: Mock) -> None:
        """Tests conversion from relative numpy velocities to list of absolute velocities"""
        np_velocities: npt.NDArray[np.float32] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        num_velocities = len(np_velocities)
        mock_relative_to_absolute_poses.side_effect = lambda _, r_s: r_s
        result = numpy_array_to_absolute_velocity('origin', np_velocities)
        mock_relative_to_absolute_poses.assert_called_once()
        self.assertEqual(num_velocities, len(result))
        for i in range(num_velocities):
            self.assertEqual(result[i].x, np_velocities[i][0])
            self.assertEqual(result[i].y, np_velocities[i][1])

    def test_input_numpy_array_to_absolute_pose_input(self) -> None:
        """Tests input validation of numpy_array_to_absolute_pose_input"""
        np_poses = np.random.random((10, 2))
        with self.assertRaises(AssertionError):
            numpy_array_to_absolute_pose(StateSE2(0, 0, 0), np_poses)

    @patch('nuplan.common.geometry.convert.relative_to_absolute_poses')
    def test_numpy_array_to_absolute_pose(self, mock_relative_to_absolute_poses: Mock) -> None:
        """Tests conversion from relative numpy poses to list of absolute StateSE2 objects."""
        np_poses = np.random.random((10, 3))
        mock_relative_to_absolute_poses.side_effect = lambda _, r_s: r_s
        result = numpy_array_to_absolute_pose('origin', np_poses)
        mock_relative_to_absolute_poses.assert_called_once()
        for np_p, se2_p in zip(np_poses, result):
            self.assertEqual(np_p[0], se2_p.x)
            self.assertEqual(np_p[1], se2_p.y)
            self.assertEqual(np_p[2], se2_p.heading)

    @patch('nuplan.common.geometry.convert.np')
    @patch('nuplan.common.geometry.convert.StateVector2D')
    def test_vector_2d_from_magnitude_angle(self, vector: Mock, mock_np: Mock) -> None:
        """Tests that projection to vector works as expected."""
        magnitude = Mock()
        angle = Mock()
        result = vector_2d_from_magnitude_angle(magnitude, angle)
        self.assertEqual(result, vector.return_value)
        vector.assert_called_once_with(mock_np.cos() * magnitude, mock_np.sin() * angle)

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

class TestCompute(unittest.TestCase):
    """Tests for compute functions"""

    @patch('nuplan.common.geometry.compute.get_pacifica_parameters', autospec=True)
    def test_signed_lateral_distance(self, mock_pacifica: Mock) -> None:
        """Tests signed lateral distance of ego to polygon"""
        mock_pacifica.return_value = Mock(half_width=1)
        result_0 = signed_lateral_distance(StateSE2(1, 1, -np.pi / 2), Polygon(((3, 2), (4, 3), (6, 1), (5, 0))))
        result_1 = signed_lateral_distance(StateSE2(1, 1, np.pi / 2), Polygon(((3, 2), (4, 3), (6, 1), (5, 0))))
        self.assertAlmostEqual(result_0, 1)
        self.assertAlmostEqual(result_1, -1)

    @patch('nuplan.common.geometry.compute.get_pacifica_parameters', autospec=True)
    def test_signed_longitudinal_distance(self, mock_pacifica: Mock) -> None:
        """Tests signed longitudinal distance of ego to polygon"""
        mock_pacifica.return_value = Mock(half_length=1)
        result_0 = signed_longitudinal_distance(StateSE2(1, 1, 0), Polygon(((3, 2), (4, 3), (6, 1), (5, 0))))
        result_1 = signed_longitudinal_distance(StateSE2(1, 1, np.pi), Polygon(((3, 2), (4, 3), (6, 1), (5, 0))))
        self.assertAlmostEqual(result_0, 1)
        self.assertAlmostEqual(result_1, -1)

    def test_compute_distance(self) -> None:
        """Tests distance between two points"""
        point_0 = StateSE2(8, 8, np.pi)
        point_1 = StateSE2(4, 5, 0)
        result_0 = compute_distance(point_0, point_1)
        result_1 = compute_distance(point_1, point_0)
        self.assertEqual(result_0, 5)
        self.assertEqual(result_1, 5)

    def test_compute_lateral_displacements(self) -> None:
        """Tests lateral distance between a list of points"""
        state_0 = StateSE2(0, 0, 0)
        state_1 = StateSE2(0, 1, 0)
        state_2 = StateSE2(0, 2, 0)
        state_3 = StateSE2(0, 3, 0)
        result = compute_lateral_displacements([state_0, state_1, state_2, state_3])
        for i in range(3):
            self.assertEqual(result[i], 1)

    def test_principal_value(self) -> None:
        """Tests principal angle calculation"""
        values: npt.NDArray[np.float64] = np.array([0, np.pi, 2 * np.pi, 3 * np.pi, -4 * np.pi, -3 * np.pi])
        expected_wrapped_0_to_pi: npt.NDArray[np.float64] = np.array([0, np.pi, 0, np.pi, 0, np.pi])
        expected_wrapped_neg_pi_to_pi: npt.NDArray[np.float64] = np.array([0, -np.pi, 0, -np.pi, 0, -np.pi])
        actual_wrapped_0_to_pi = principal_value(values, min_=0)
        actual_wrapped_neg_pi_to_pi = principal_value(values)
        np.testing.assert_allclose(expected_wrapped_0_to_pi, actual_wrapped_0_to_pi)
        np.testing.assert_allclose(expected_wrapped_neg_pi_to_pi, actual_wrapped_neg_pi_to_pi)

    def test_l2_euclidean_corners_distance(self) -> None:
        """Tests computation of distances between"""
        box_dimension = Dimension(4, 3, 1)
        box1 = OrientedBox(StateSE2(0, 0, 0), box_dimension.length, box_dimension.width, box_dimension.height)
        box2 = OrientedBox(StateSE2(2, 0, 0), box_dimension.length, box_dimension.width, box_dimension.height)
        box3 = OrientedBox(StateSE2(0, 2, 0), box_dimension.length, box_dimension.width, box_dimension.height)
        box4 = OrientedBox(StateSE2(3, 4, 0), box_dimension.length, box_dimension.width, box_dimension.height)
        box1_rot = OrientedBox(StateSE2(0, 0, np.pi), box_dimension.length, box_dimension.width, box_dimension.height)
        box5 = OrientedBox(StateSE2(1, 2, 3), box_dimension.length, box_dimension.width, box_dimension.height)
        self.assertEqual(0, l2_euclidean_corners_distance(box1, box1))
        self.assertEqual(4.0, l2_euclidean_corners_distance(box1, box2))
        self.assertEqual(l2_euclidean_corners_distance(box1, box2), l2_euclidean_corners_distance(box1, box3))
        self.assertEqual(10.0, l2_euclidean_corners_distance(box1, box4))
        self.assertEqual(10.0, l2_euclidean_corners_distance(box1, box1_rot))
        self.assertTrue(math.isclose(10.931588394648887, l2_euclidean_corners_distance(box1, box5)))

    def test_se2_box_distances(self) -> None:
        """Tests computation of distances between SE2 poses using OrientedBox"""
        box_dimension = Dimension(4, 3, 1)
        query = StateSE2(0, 0, 0)
        targets = [StateSE2(0, 0, 0), StateSE2(0, 0, np.pi), StateSE2(2, 0, 0)]
        self.assertEqual([0, 0, 4.0], se2_box_distances(query, targets, box_dimension))
        self.assertEqual([0, 10.0, 4.0], se2_box_distances(query, targets, box_dimension, consider_flipped=False))

def compute_distance(lhs: StateSE2, rhs: StateSE2) -> float:
    """
    Compute the euclidean distance between two points
    :param lhs: first point
    :param rhs: second point
    :return distance between two points
    """
    return float(np.hypot(lhs.x - rhs.x, lhs.y - rhs.y))

def compute_lateral_displacements(poses: List[StateSE2]) -> List[float]:
    """
    Computes the lateral displacements (y_t - y_t-1) from a list of poses

    :param poses: list of N poses to compute displacements from
    :return: list of N-1 lateral displacements
    """
    return [poses[idx].y - poses[idx - 1].y for idx in range(1, len(poses))]

def to_state_from_scene(scene: Dict[str, Any]) -> StateSE2:
    """
    Extract state se2 from pose.
    :param scene: position from scene.
    :return StateSE2.
    """
    return StateSE2(x=scene['pose'][0], y=scene['pose'][1], heading=scene['pose'][2])

def to_ego_center_from_scene(scene: Dict[str, Any], vehicle: VehicleParameters) -> StateSE2:
    """
    :param scene: from scene['ego'].
    :param vehicle: vehicle parameters.
    :return the extracted State in the center of ego's bounding box.
    """
    ego_pose = scene['pose']
    ego_x = ego_pose[0]
    ego_y = ego_pose[1]
    ego_heading = ego_pose[2]
    distance = vehicle.rear_axle_to_center
    return translate_longitudinally(StateSE2(ego_x, ego_y, ego_heading), distance)

def to_agent_state_from_scene(scene: Dict[str, Any], vehicle: VehicleParameters, time_us: int=10, to_cog: bool=True) -> EgoState:
    """
    Extract agent state from scene.
    :param scene: from json.
    :param vehicle: parameters.
    :param time_us: [us] initial time.
    :param to_cog: If true, xy will be translated to the COG of the car. Otherwise xy is assumed to be at rear axle.
    :return EgoState.
    """
    if to_cog:
        ego_state2d = to_ego_center_from_scene(scene, vehicle)
    else:
        ego_pose = scene['pose']
        ego_state2d = StateSE2(ego_pose[0], ego_pose[1], ego_pose[2])
    if 'velocity_x' not in scene or 'velocity_y' not in scene:
        velocity_x = scene['speed']
        velocity_y = 0.0
    else:
        velocity_x = scene['velocity_x']
        velocity_y = scene['velocity_y']
    if 'acceleration_x' not in scene or 'acceleration_y' not in scene:
        acceleration_x = scene['acceleration']
        acceleration_y = 0.0
    else:
        acceleration_x = scene['acceleration_x']
        acceleration_y = scene['acceleration_y']
    return EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(x=ego_state2d.x, y=ego_state2d.y, heading=ego_state2d.heading), time_point=TimePoint(time_us), rear_axle_velocity_2d=StateVector2D(x=velocity_x, y=velocity_y), rear_axle_acceleration_2d=StateVector2D(x=acceleration_x, y=acceleration_y), tire_steering_angle=0, vehicle_parameters=get_pacifica_parameters())

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

def _se2_vel_acc_to_ego_state(state: StateSE2, velocity: npt.NDArray[np.float32], acceleration: npt.NDArray[np.float32], timestamp: float, vehicle: VehicleParameters) -> EgoState:
    """
    Convert StateSE2, velocity and acceleration to EgoState given a timestamp.

    :param state: input SE2 state
    :param velocity: [m/s] longitudinal velocity, lateral velocity
    :param acceleration: [m/s^2] longitudinal acceleration, lateral acceleration
    :param timestamp: [s] timestamp of state
    :return: output agent state
    """
    return EgoState.build_from_rear_axle(rear_axle_pose=state, rear_axle_velocity_2d=StateVector2D(*velocity), rear_axle_acceleration_2d=StateVector2D(*acceleration), tire_steering_angle=0.0, time_point=TimePoint(int(timestamp * 1000000.0)), vehicle_parameters=vehicle, is_in_auto_mode=True)

def _get_velocity_and_acceleration(ego_poses: List[StateSE2], ego_history: Deque[EgoState], timesteps: List[float]) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Given the past, current and planned ego poses, estimate the velocity and acceleration by taking the derivatives.

    :param ego_poses: a list of the planned ego poses
    :param ego_history: the ego history that includes the current
    :param timesteps: [s] timesteps of the planned ego poses
    :return: the approximated velocity and acceleration in ego centric frame
    """
    ego_history_len = len(ego_history)
    current_ego_state = ego_history[-1]
    timesteps_past_current = [state.time_point.time_s for state in ego_history]
    ego_poses_past_current: npt.NDArray[np.float32] = np.stack([np.array(state.rear_axle.serialize()) for state in ego_history])
    dt = current_ego_state.time_point.time_s - ego_history[-2].time_point.time_s
    timesteps_current_planned: npt.NDArray[np.float32] = np.array([current_ego_state.time_point.time_s] + timesteps)
    ego_poses_current_planned: npt.NDArray[np.float32] = np.stack([current_ego_state.rear_axle.serialize()] + [pose.serialize() for pose in ego_poses])
    ego_poses_interpolate = interp1d(timesteps_current_planned, ego_poses_current_planned, axis=0, fill_value='extrapolate')
    timesteps_current_planned_interp = np.arange(start=current_ego_state.time_point.time_s, stop=timesteps[-1] + 1e-06, step=dt)
    ego_poses_current_planned_interp = ego_poses_interpolate(timesteps_current_planned_interp)
    timesteps_past_current_planned = [*timesteps_past_current, *timesteps_current_planned_interp[1:]]
    ego_poses_past_current_planned: npt.NDArray[np.float32] = np.concatenate([ego_poses_past_current, ego_poses_current_planned_interp[1:]], axis=0)
    ego_velocity_past_current_planned = approximate_derivatives(ego_poses_past_current_planned[:, :2], timesteps_past_current_planned, axis=0)
    ego_acceleration_past_current_planned = approximate_derivatives(ego_poses_past_current_planned[:, :2], timesteps_past_current_planned, axis=0, deriv_order=2)
    ego_velocity_planned_xy = ego_velocity_past_current_planned[ego_history_len:]
    ego_acceleration_planned_xy = ego_acceleration_past_current_planned[ego_history_len:]
    ego_velocity_planned_ds = _project_from_global_to_ego_centric_ds(ego_poses_current_planned_interp[1:], ego_velocity_planned_xy)
    ego_acceleration_planned_ds = _project_from_global_to_ego_centric_ds(ego_poses_current_planned_interp[1:], ego_acceleration_planned_xy)
    ego_velocity_interp_back = interp1d(timesteps_past_current_planned[ego_history_len:], ego_velocity_planned_ds, axis=0, fill_value='extrapolate')
    ego_acceleration_interp_back = interp1d(timesteps_past_current_planned[ego_history_len:], ego_acceleration_planned_ds, axis=0, fill_value='extrapolate')
    ego_velocity_planned_ds = ego_velocity_interp_back(timesteps)
    ego_acceleration_planned_ds = ego_acceleration_interp_back(timesteps)
    return (ego_velocity_planned_ds, ego_acceleration_planned_ds)

def _project_from_global_to_ego_centric_ds(ego_poses: npt.NDArray[np.float32], values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Project value from the global xy frame to the ego centric ds frame.

    :param ego_poses: [x, y, heading] with size [planned steps, 3].
    :param values: values in global frame with size [planned steps, 2]
    :return: values projected onto the new frame with size [planned steps, 2]
    """
    headings = ego_poses[:, -1:]
    values_lon = values[:, :1] * np.cos(headings) + values[:, 1:2] * np.sin(headings)
    values_lat = values[:, :1] * np.sin(headings) - values[:, 1:2] * np.cos(headings)
    values = np.concatenate((values_lon, values_lat), axis=1)
    return values

def _get_absolute_agent_states_from_numpy_poses(poses: npt.NDArray[np.float32], ego_history: Deque[EgoState], timesteps: List[float]) -> List[EgoState]:
    """
    Converts an array of relative numpy poses to a list of absolute EgoState objects.

    :param poses: input relative poses
    :param ego_history: the history of the ego state, including the current
    :param timesteps: timestamps corresponding to each state
    :return: list of agent states
    """
    ego_state = ego_history[-1]
    relative_states = [StateSE2.deserialize(pose) for pose in poses]
    absolute_states = relative_to_absolute_poses(ego_state.rear_axle, relative_states)
    velocities, accelerations = _get_velocity_and_acceleration(absolute_states, ego_history, timesteps)
    agent_states = [_se2_vel_acc_to_ego_state(state, velocity, acceleration, timestep, ego_state.car_footprint.vehicle_parameters) for state, velocity, acceleration, timestep in zip(absolute_states, velocities, accelerations, timesteps)]
    return agent_states

def transform_predictions_to_states(predicted_poses: npt.NDArray[np.float32], ego_history: Deque[EgoState], future_horizon: float, step_interval: float, include_ego_state: bool=True) -> List[EgoState]:
    """
    Transform an array of pose predictions to a list of EgoState.

    :param predicted_poses: input relative poses
    :param ego_history: the history of the ego state, including the current
    :param future_horizon: [s] future time horizon
    :param step_interval: [s] interval between steps in the array
    :param include_ego_state: whether to include the current ego state as the initial state
    :return: transformed absolute states
    """
    ego_state = ego_history[-1]
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    states = _get_absolute_agent_states_from_numpy_poses(predicted_poses, ego_history, timesteps)
    if include_ego_state:
        states.insert(0, ego_state)
    return states

def _get_fixed_timesteps(state: EgoState, future_horizon: float, step_interval: float) -> List[float]:
    """
    Get a fixed array of timesteps starting from a state's time.

    :param state: input state
    :param future_horizon: [s] future time horizon
    :param step_interval: [s] interval between steps in the array
    :return: constructed timestep list
    """
    timesteps = np.arange(0.0, future_horizon, step_interval) + step_interval
    timesteps += state.time_point.time_s
    return list(timesteps.tolist())

class MLPlanner(AbstractPlanner):
    """
    Implements abstract planner interface.
    Used for simulating any ML planner trained through the nuPlan training framework.
    """

    def __init__(self, model: TorchModuleWrapper) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval = model.future_trajectory_sampling.step_time
        self._num_output_dim = model.future_trajectory_sampling.num_poses
        self._model_loader = ModelLoader(model)
        self._initialization: Optional[PlannerInitialization] = None
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        predictions = self._model_loader.infer(features)
        trajectory_predicted = cast(Trajectory, predictions['trajectory'])
        trajectory_tensor = trajectory_predicted.data
        trajectory = trajectory_tensor.cpu().detach().numpy()[0]
        return cast(npt.NDArray[np.float32], trajectory)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._model_loader.initialize()
        self._initialization = initialization

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        history = current_input.history
        start_time = time.perf_counter()
        features = self._model_loader.build_features(current_input, self._initialization)
        self._feature_building_runtimes.append(time.perf_counter() - start_time)
        start_time = time.perf_counter()
        predictions = self._infer_model(features)
        self._inference_runtimes.append(time.perf_counter() - start_time)
        states = transform_predictions_to_states(predictions, history.ego_states, self._future_horizon, self._step_interval)
        trajectory = InterpolatedTrajectory(states)
        return trajectory

    def generate_planner_report(self, clear_stats: bool=True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(compute_trajectory_runtimes=self._compute_trajectory_runtimes, feature_building_runtimes=self._feature_building_runtimes, inference_runtimes=self._inference_runtimes)
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []
        return report

class TestTransformUtils(unittest.TestCase):
    """
    Unit tests for transform_utils.py
    """

    def test_transform_predictions_to_states(self) -> None:
        """
        Test transform predictions to states
        """
        predicted_poses: npt.NDArray[np.float32] = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
        ego_history: List[MagicMock] = []
        for i in range(5):
            s = MagicMock()
            s.time_point.time_s = i * 0.1
            s.car_footprint.vehicle_parameters = VehicleParameters(width=2, front_length=4, rear_length=1, cog_position_from_rear_axle=2, height=2, wheel_base=3, vehicle_name='mock', vehicle_type='mock')
            s.rear_axle = StateSE2.deserialize([i * 0.1, i * 0.1, np.pi / 4])
            ego_history.append(s)
        future_horizon = 3
        time_interval = 1
        states = transform_predictions_to_states(predicted_poses, ego_history, future_horizon, time_interval)
        np.testing.assert_allclose(ego_history[-1].rear_axle.serialize(), states[0].rear_axle.serialize())
        gt_poses = [[0.4 + i * np.cos(np.pi / 4), 0.4 + i * np.sin(np.pi / 4), np.pi / 4] for i in range(1, 4)]
        np.testing.assert_allclose(gt_poses, [s.rear_axle.serialize() for s in states[1:]])
        np.testing.assert_allclose([0.4, 1.4, 2.4, 3.4], [s.time_point.time_s for s in states])
        np.testing.assert_allclose([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], [s.dynamic_car_state.center_velocity_2d.array for s in states[1:]], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [s.dynamic_car_state.center_acceleration_2d.array for s in states[1:]], rtol=1e-06, atol=1e-06)

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

def sort_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sort dictionary according to the key
    :param dictionary: the dictionary to be sorted
    :return: a sorted dictionary
    """
    return {key: dictionary[key] for key in sorted(dictionary.keys())}

def _convert_prediction_to_predicted_trajectory(agent: TrackedObject, poses: List[StateSE2], xy_velocities: List[StateVector2D], step_interval_us: float) -> PredictedTrajectory:
    """
    Convert each agent predictions into a PredictedTrajectory.
    :param agent: The agent the predictions are for.
    :param poses: A list of poses that makes up the predictions
    :param xy_velocities: A list of velocities in world frame corresponding to each pose.
    :return: The predictions parsed into PredictedTrajectory.
    """
    waypoints = [Waypoint(TimePoint(0), agent.box, agent.velocity)]
    waypoints += [Waypoint(TimePoint(int((step + 1) * step_interval_us)), OrientedBox.from_new_pose(agent.box, pose), velocity) for step, (pose, velocity) in enumerate(zip(poses, xy_velocities))]
    return PredictedTrajectory(1.0, waypoints)

class EgoCentricMLAgents(AbstractMLAgents):
    """
    Simulate agents based on an ML model.
    """

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario) -> None:
        """
        Initializes the EgoCentricMLAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        super().__init__(model, scenario)
        self.prediction_type = 'agents_trajectory'

    @property
    def _ego_velocity_anchor_state(self) -> StateSE2:
        """
        Returns the ego's velocity state vector as an anchor state for transformation.
        :return: A StateSE2 representing ego's velocity state as an anchor state
        """
        ego_velocity = self._ego_anchor_state.dynamic_car_state.rear_axle_velocity_2d
        return StateSE2(ego_velocity.x, ego_velocity.y, self._ego_anchor_state.rear_axle.heading)

    def _infer_model(self, features: FeaturesType) -> TargetsType:
        """Inherited, see superclass."""
        predictions = self._model_loader.infer(features)
        if self.prediction_type not in predictions:
            raise ValueError(f"Prediction does not have the output '{self.prediction_type}'")
        agents_prediction_tensor = cast(AgentsTrajectories, predictions[self.prediction_type]).data
        agents_prediction = agents_prediction_tensor[0].cpu().detach().numpy()
        return {self.prediction_type: AgentsTrajectories([cast(npt.NDArray[np.float32], agents_prediction)]).get_agents_only_trajectories()}

    def _update_observation_with_predictions(self, predictions: TargetsType) -> None:
        """Inherited, see superclass."""
        assert self._agents, 'The agents have not been initialized. Please make sure they are initialized!'
        agent_predictions = cast(AgentsTrajectories, predictions[self.prediction_type])
        agent_predictions.reshape_to_agents()
        agent_poses = agent_predictions.poses[0]
        agent_velocities = agent_predictions.xy_velocity[0]
        for agent_token, agent, poses_horizon, xy_velocity_horizon in zip(self._agents, self._agents.values(), agent_poses, agent_velocities):
            poses = numpy_array_to_absolute_pose(self._ego_anchor_state.rear_axle, poses_horizon)
            xy_velocities = numpy_array_to_absolute_velocity(self._ego_velocity_anchor_state, xy_velocity_horizon)
            future_trajectory = _convert_prediction_to_predicted_trajectory(agent, poses, xy_velocities, self._step_interval_us)
            new_state = future_trajectory.trajectory.get_state_at_time(self.step_time)
            new_agent = Agent(tracked_object_type=agent.tracked_object_type, oriented_box=new_state.oriented_box, velocity=new_state.velocity, metadata=agent.metadata)
            new_agent.predictions = [future_trajectory]
            self._agents[agent_token] = new_agent

def get_closest_agent_in_position(ego_state: EgoState, observations: DetectionsTracks, is_in_position: Callable[[StateSE2, StateSE2], bool], collided_track_ids: Set[str]=set(), lateral_distance_threshold: float=0.5) -> Tuple[Optional[Agent], float]:
    """
    Searches for the closest agent in a specified position
    :param ego_state: ego's state
    :param observations: agents as DetectionTracks
    :param is_in_position: a function to determine the positional relationship to the ego
    :param collided_track_ids: Set of collided track tokens, default {}
    :param lateral_distance_threshold: Agents laterally further away than this threshold are not considered, default 0.5 meters
    :return: the closest agent in the position and the corresponding shortest distance.
    """
    closest_distance = np.inf
    closest_agent = None
    for agent in observations.tracked_objects.get_agents():
        if is_in_position(ego_state.rear_axle, agent.center) and agent.track_token not in collided_track_ids and (abs(signed_lateral_distance(ego_state.rear_axle, agent.box.geometry)) < lateral_distance_threshold):
            distance = abs(ego_state.car_footprint.oriented_box.geometry.distance(agent.box.geometry))
            if distance < closest_distance:
                closest_distance = distance
                closest_agent = agent
    return (closest_agent, float(closest_distance))

class IDMAgent:
    """IDM smart-agent."""

    def __init__(self, start_iteration: int, initial_state: IDMInitialState, route: List[LaneGraphEdgeMapObject], policy: IDMPolicy, minimum_path_length: float, max_route_len: int=5):
        """
        Constructor for IDMAgent.
        :param start_iteration: scenario iteration where agent first appeared
        :param initial_state: agent initial state
        :param route: agent initial route plan
        :param policy: policy controlling the agent behavior
        :param minimum_path_length: [m] The minimum path length
        :param max_route_len: The max number of route elements to store
        """
        self._start_iteration = start_iteration
        self._initial_state = initial_state
        self._state = IDMAgentState(initial_state.path_progress, initial_state.velocity.x)
        self._route: Deque[LaneGraphEdgeMapObject] = deque(route, maxlen=max_route_len)
        self._path = self._convert_route_to_path()
        self._policy = policy
        self._minimum_path_length = minimum_path_length
        self._size = (initial_state.box.width, initial_state.box.length, initial_state.box.height)
        self._requires_state_update: bool = True
        self._full_agent_state: Optional[Agent] = None

    def propagate(self, lead_agent: IDMLeadAgentState, tspan: float) -> None:
        """
        Propagate agent forward according to the IDM policy.

        :param lead_agent: the agent leading this agent
        :param tspan: the interval of time to propagate for
        """
        speed_limit = self.end_segment.speed_limit_mps
        if speed_limit is not None and speed_limit > 0.0:
            self._policy.target_velocity = speed_limit
        solution = self._policy.solve_forward_euler_idm_policy(IDMAgentState(0, self._state.velocity), lead_agent, tspan)
        self._state.progress += solution.progress
        self._state.velocity = max(solution.velocity, 0)
        self._requires_state_update = True

    @property
    def agent(self) -> Agent:
        """:return: the agent as a Agent object"""
        return self._get_agent_at_progress(self._get_bounded_progress())

    @property
    def polygon(self) -> Polygon:
        """:return: the agent as a Agent object"""
        return self.agent.box.geometry

    def get_route(self) -> List[LaneGraphEdgeMapObject]:
        """:return: The route the IDM agent is following."""
        return list(self._route)

    @property
    def projected_footprint(self) -> Polygon:
        """
        Returns the agent's projected footprint along it's planned path. The extended length is proportional
        to it's current velocity
        :return: The agent's projected footprint as a Polygon.
        """
        start_progress = self._clamp_progress(self.progress - self.length / 2)
        end_progress = self._clamp_progress(self.progress + self.length / 2 + self.velocity * self._policy.headway_time)
        projected_path = path_to_linestring(trim_path(self._path, start_progress, end_progress))
        return unary_union([projected_path.buffer(self.width / 2, cap_style=CAP_STYLE.flat), self.polygon])

    @property
    def width(self) -> float:
        """:return: [m] agent's width"""
        return float(self._initial_state.box.width)

    @property
    def length(self) -> float:
        """:return: [m] agent's length"""
        return float(self._initial_state.box.length)

    @property
    def progress(self) -> float:
        """:return: [m] agent's progress"""
        return self._state.progress

    @property
    def velocity(self) -> float:
        """:return: [m/s] agent's velocity along the path"""
        return self._state.velocity

    @property
    def end_segment(self) -> LaneGraphEdgeMapObject:
        """
        Returns the last segment in the agent's route
        :return: End segment as a LaneGraphEdgeMapObject
        """
        return self._route[-1]

    def to_se2(self) -> StateSE2:
        """
        :return: the agent as a StateSE2 object
        """
        return self._get_agent_at_progress(self._get_bounded_progress()).box.center

    def is_active(self, iteration: int) -> bool:
        """
        Return if the agent should be active at a simulation iteration

        :param iteration: the current simulation iteration
        :return: true if active, false otherwise
        """
        return self._start_iteration <= iteration

    def has_valid_path(self) -> bool:
        """
        :return: true if agent has a valid path, false otherwise
        """
        return self._path is not None

    def _get_bounded_progress(self) -> float:
        """
        :return: [m] The agent's progress. The progress is clamped between the start and end progress of it's path
        """
        return self._clamp_progress(self._state.progress)

    def get_path_to_go(self) -> List[ProgressStateSE2]:
        """
        :return: The agent's path trimmed to start at the agent's current progress
        """
        return trim_path_up_to_progress(self._path, self._get_bounded_progress())

    def get_progress_to_go(self) -> float:
        """
        return: [m] the progress left until the end of the path
        """
        return self._path.get_end_progress() - self.progress

    def get_agent_with_planned_trajectory(self, num_samples: int, sampling_time: float) -> Agent:
        """
        Samples the the agent's trajectory. The velocity is assumed to be constant over the sampled trajectory
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: the agent's trajectory as a list of Agent
        """
        return self._get_agent_at_progress(self._get_bounded_progress(), num_samples, sampling_time)

    def plan_route(self, traffic_light_status: Dict[TrafficLightStatusType, List[str]]) -> None:
        """
        The planning logic for the agent.
            - Prefers going straight. Selects edge with the lowest curvature.
            - Looks to add a segment to the route if:
                - the progress to go is less than the agent's desired velocity multiplied by the desired headway time
                  plus the minimum path length
                - the outgoing segment is active

        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information
        """
        while self.get_progress_to_go() < self._minimum_path_length + self._policy.target_velocity * self._policy.headway_time:
            outgoing_edges = self.end_segment.outgoing_edges
            selected_outgoing_edges = []
            for edge in outgoing_edges:
                if edge.has_traffic_lights():
                    if edge.id in traffic_light_status[TrafficLightStatusType.GREEN]:
                        selected_outgoing_edges.append(edge)
                elif edge.id not in traffic_light_status[TrafficLightStatusType.RED]:
                    selected_outgoing_edges.append(edge)
            if not selected_outgoing_edges:
                break
            curvatures = [abs(edge.baseline_path.get_curvature_at_arc_length(0.0)) for edge in selected_outgoing_edges]
            idx = np.argmin(curvatures)
            new_segment = selected_outgoing_edges[idx]
            self._route.append(new_segment)
            self._path = create_path_from_se2(self.get_path_to_go() + new_segment.baseline_path.discrete_path)
            self._state.progress = 0

    def _get_agent_at_progress(self, progress: float, num_samples: Optional[int]=None, sampling_time: Optional[float]=None) -> Agent:
        """
        Returns the agent as a box at a given progress
        :param progress: the arc length along the agent's path
        :return: the agent as a Agent object at the given progress
        """
        if not self._requires_state_update:
            return self._full_agent_state
        if self._path is not None:
            init_pose = self._path.get_state_at_progress(progress)
            box = OrientedBox.from_new_pose(self._initial_state.box, StateSE2(init_pose.x, init_pose.y, init_pose.heading))
            future_trajectory = None
            if num_samples and sampling_time:
                progress_samples = [self._clamp_progress(progress + self.velocity * sampling_time * (step + 1)) for step in range(num_samples)]
                future_poses = self._path.get_state_at_progresses(progress_samples)
                time_stamps = [TimePoint(int(1000000.0 * sampling_time * (step + 1))) for step in range(num_samples)]
                init_way_point = [Waypoint(TimePoint(0), box, self._velocity_to_global_frame(init_pose.heading))]
                waypoints = [Waypoint(time, OrientedBox.from_new_pose(self._initial_state.box, pose), self._velocity_to_global_frame(pose.heading)) for time, pose in zip(time_stamps, future_poses)]
                future_trajectory = PredictedTrajectory(1.0, init_way_point + waypoints)
            self._full_agent_state = Agent(metadata=self._initial_state.metadata, oriented_box=box, velocity=self._velocity_to_global_frame(init_pose.heading), tracked_object_type=self._initial_state.tracked_object_type, predictions=[future_trajectory] if future_trajectory is not None else [])
        else:
            self._full_agent_state = Agent(metadata=self._initial_state.metadata, oriented_box=self._initial_state.box, velocity=self._initial_state.velocity, tracked_object_type=self._initial_state.tracked_object_type, predictions=self._initial_state.predictions)
        self._requires_state_update = False
        return self._full_agent_state

    def _clamp_progress(self, progress: float) -> float:
        """
        Clamp the progress to be between the agent's path bounds
        :param progress: [m] the progress along the agent's path
        :return: [m] the progress clamped between the start and end progress of the agent's path
        """
        return max(self._path.get_start_progress(), min(progress, self._path.get_end_progress()))

    def _convert_route_to_path(self) -> InterpolatedPath:
        """
        Converts the route into an InterpolatedPath
        :return: InterpolatedPath from the agent's route
        """
        blp: List[StateSE2] = []
        for segment in self._route:
            blp.extend(segment.baseline_path.discrete_path)
        return create_path_from_se2(blp)

    def _velocity_to_global_frame(self, heading: float) -> StateVector2D:
        """
        Transform agent's velocity along the path to global frame
        :param heading: [rad] The heading defining the transform to global frame.
        :return: The velocity vector in global frame.
        """
        return StateVector2D(self.velocity * np.cos(heading), self.velocity * np.sin(heading))

class InterpolatedPath(AbstractPath):
    """A path that is interpolated from a list of points."""

    def __init__(self, path: List[ProgressStateSE2]):
        """
        Constructor of InterpolatedPath.

        :param path: List of states creating a path.
            The path has to have at least 2 elements, otherwise it is considered invalid and the class will raise.
        """
        assert len(path) > 1, 'Path has to has more than 1 element!'
        self._path = path
        progress = [point.progress for point in self._path]
        linear_states = []
        angular_states = []
        for point in path:
            linear_states.append([point.progress, point.x, point.y])
            angular_states.append([point.heading])
        linear_states = np.array(linear_states, dtype='float64')
        angular_states = np.array(angular_states, dtype='float64')
        self._function_interp_linear = sp_interp.interp1d(progress, linear_states, axis=0)
        self._angular_interpolator = AngularInterpolator(progress, angular_states)

    def get_start_progress(self) -> float:
        """Inherited, see superclass."""
        return self._path[0].progress

    def get_end_progress(self) -> float:
        """Inherited, see superclass."""
        return self._path[-1].progress

    def get_state_at_progress(self, progress: float) -> ProgressStateSE2:
        """Inherited, see superclass."""
        self._assert_progress(progress)
        linear_states = list(self._function_interp_linear(progress))
        angular_states = list(self._angular_interpolator.interpolate(progress))
        return ProgressStateSE2.deserialize(linear_states + angular_states)

    def get_state_at_progresses(self, progresses: List[float]) -> List[ProgressStateSE2]:
        """Inherited, see superclass."""
        self._assert_progress(min(progresses))
        self._assert_progress(max(progresses))
        linear_states_batch = self._function_interp_linear(progresses)
        angular_states_batch = self._angular_interpolator.interpolate(progresses)
        return [ProgressStateSE2.deserialize(list(linear_states) + list(angular_states)) for linear_states, angular_states in zip(linear_states_batch, angular_states_batch)]

    def get_sampled_path(self) -> List[ProgressStateSE2]:
        """Inherited, see superclass."""
        return self._path

    def _assert_progress(self, progress: float) -> None:
        """Check if queried progress is within bounds"""
        start_progress = self.get_start_progress()
        end_progress = self.get_end_progress()
        assert start_progress <= progress <= end_progress, f'Progress exceeds path! {start_progress} <= {progress} <= {end_progress}'

class TestPathUtils(unittest.TestCase):
    """Tests path util functions."""

    def setUp(self) -> None:
        """Test setup."""
        self.path = [StateSE2(0, 0, 0), StateSE2(3, 4, 1), StateSE2(7, 7, 2), StateSE2(10, 10, 3)]

    def test_calculate_progress(self) -> None:
        """Tests if progress is calculated correctly"""
        progress = calculate_progress(self.path)
        self.assertEqual([0.0, 5.0, 10.0, 14.242640687119284], progress)

    def test_convert_se2_path_to_progress_path(self) -> None:
        """Tests if conversion to List[ProgressStateSE2] is calculated correctly"""
        progress_path = convert_se2_path_to_progress_path(self.path)
        self.assertEqual([0.0, 5.0, 10.0, 14.242640687119284], [point.progress for point in progress_path])
        self.assertEqual(self.path, [StateSE2(x=point.x, y=point.y, heading=point.heading) for point in progress_path])

def extract_and_pad_agent_states(agent_trajectories: List[TrackedObjects], state_extractor: Callable[[TrackedObjects], Any], reverse: bool) -> Tuple[List[List[Any]], List[List[bool]]]:
    """
    Extract the agent states and pads it with the most recent available states. The order of the agents is also
    preserved. Note: only agents that appear in the current time step will be computed for. Agents appearing in the
    future or past will be discarded.

     t1      t2           t1      t2
    |a1,t1| |a1,t2|  pad |a1,t1| |a1,t2|
    |a2,t1| |a3,t2|  ->  |a2,t1| |a2,t1| (padded with agent 2 state at t1)
    |a3,t1| |     |      |a3,t1| |a3,t2|


    If reverse is True, the padding direction will start from the end of the trajectory towards the start

     tN-1    tN             tN-1    tN
    |a1,tN-1| |a1,tN|  pad |a1,tN-1| |a1,tN|
    |a2,tN  | |a2,tN|  <-  |a3,tN-1| |a2,tN| (padded with agent 2 state at tN)
    |a3,tN-1| |a3,tN|      |       | |a3,tN|

    :param agent_trajectories: agent trajectories [num_frames, num_agents, 1]
    :param state_extractor: a function to extract a state from a SceneObject instance
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states, and an availability array indicate whether a agent's
    future state is available at each frame.
    """
    if reverse:
        agent_trajectories = agent_trajectories[::-1]
    current_agents_state = {scene_object.track_token: state_extractor(scene_object) for scene_object in agent_trajectories[0].tracked_objects}
    current_agents_state = sort_dict(current_agents_state)
    agent_states_horizon: List[List[Any]] = []
    agent_availabilities: List[List[bool]] = []
    non_availability = {agent_token: False for agent_token in current_agents_state.keys()}
    for tracked_objects in agent_trajectories:
        next_agents_states = {scene_object.track_token: state_extractor(scene_object) for scene_object in tracked_objects.tracked_objects}
        current_agents_state = {**current_agents_state, **next_agents_states}
        agent_states_horizon.append(list(current_agents_state.values()))
        next_agents_available = {scene_object.track_token: True for scene_object in tracked_objects.tracked_objects}
        current_agents_availability = {**non_availability, **next_agents_available}
        agent_availabilities.append(list(current_agents_availability.values()))
    if reverse:
        agent_states_horizon = agent_states_horizon[::-1]
        agent_availabilities = agent_availabilities[::-1]
    return (agent_states_horizon, agent_availabilities)

def extract_and_pad_agent_poses(agent_trajectories: List[TrackedObjects], reverse: bool=False) -> Tuple[List[List[StateSE2]], List[List[bool]]]:
    """
    Extract and pad agent poses along the given trajectory. For details see extract_and_pad_agent_states.
    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of StateSE2 for all agents, and an availability array indicate whether a agent's
    future state is available at each frame.
    """
    return extract_and_pad_agent_states(agent_trajectories, lambda scene_object: StateSE2(scene_object.center.x, scene_object.center.y, scene_object.center.heading), reverse)

def extract_and_pad_agent_sizes(agent_trajectories: List[TrackedObjects], reverse: bool=False) -> Tuple[List[List[npt.NDArray[np.float32]]], List[List[bool]]]:
    """
    Extract and pad agent sizes along the given trajectory. For details see extract_and_pad_agent_states.
    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of sizes for all agents, and an availability array indicate whether a agent's
    future state is available at each frame.
    """
    return extract_and_pad_agent_states(agent_trajectories, lambda agent: np.array([agent.box.width, agent.box.length], np.float32), reverse)

def extract_and_pad_agent_velocities(agent_trajectories: List[TrackedObjects], reverse: bool=False) -> Tuple[List[List[StateSE2]], List[List[bool]]]:
    """
    Extract and pad agent sizes along the given trajectory. For details see extract_and_pad_agent_states.
    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of velocities for all agents, and an availability array indicate whether a agent's
    future state is available at each frame.
    """
    return extract_and_pad_agent_states(agent_trajectories, lambda box: StateSE2(0, 0, 0) if np.isnan(box.velocity.array).any() else StateSE2(box.velocity.x, box.velocity.y, box.center.heading), reverse)

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

@dataclass
class AgentStatePlot(BaseScenarioPlot):
    """A dataclass for agent state plot."""
    data_sources: Dict[int, Dict[str, ColumnDataSource]] = field(default_factory=dict)
    plots: Dict[str, GlyphRenderer] = field(default_factory=dict)
    track_id_history: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        """Initialize track id history."""
        super().__post_init__()
        if not self.track_id_history:
            self.track_id_history = {}

    def _get_track_id(self, track_id: str) -> Union[int, float]:
        """
        Get a number representation for track ids.
        :param track_id: Agent track id.
        :return A number representation for a track id.
        """
        if track_id == 'null' or not self.track_id_history:
            return np.nan
        number_track_id = self.track_id_history.get(track_id, None)
        if not number_track_id:
            self.track_id_history[track_id] = len(self.track_id_history)
            number_track_id = len(self.track_id_history)
        return number_track_id

    def update_plot(self, main_figure: Figure, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param frame_index: Frame index.
        :param doc: Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return
        self.render_event.set()
        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                data_sources = self.data_sources.get(frame_index, None)
                if not data_sources:
                    return
                for category, data_source in data_sources.items():
                    plot = self.plots.get(category, None)
                    data = dict(data_source.data)
                    if plot is None:
                        agent_color = simulation_tile_agent_style.get(category)
                        self.plots[category] = main_figure.multi_polygons(xs='xs', ys='ys', fill_color=agent_color['fill_color'], fill_alpha=agent_color['fill_alpha'], line_color=agent_color['line_color'], line_width=agent_color['line_width'], source=data)
                        agent_hover = HoverTool(renderers=[self.plots[category]], tooltips=[('center_x [m]', '@center_xs{0.2f}'), ('center_y [m]', '@center_ys{0.2f}'), ('velocity_x [m/s]', '@velocity_xs{0.2f}'), ('velocity_y [m/s]', '@velocity_ys{0.2f}'), ('speed [m/s]', '@speeds{0.2f}'), ('heading [rad]', '@headings{0.2f}'), ('type', '@agent_type'), ('track token', '@track_token')])
                        main_figure.add_tools(agent_hover)
                    else:
                        self.plots[category].data_source.data = data
                self.render_event.clear()
            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update agents data sources.
        :param history: SimulationHistory time-series data.
        """
        if not self.data_source_condition:
            return
        with self.data_source_condition:
            for frame_index, sample in enumerate(history.data):
                if not isinstance(sample.observation, DetectionsTracks):
                    continue
                tracked_objects = sample.observation.tracked_objects
                frame_dict = {}
                for tracked_object_type_name, tracked_object_type in tracked_object_types.items():
                    corner_xs = []
                    corner_ys = []
                    track_ids = []
                    track_tokens = []
                    agent_types = []
                    center_xs = []
                    center_ys = []
                    velocity_xs = []
                    velocity_ys = []
                    speeds = []
                    headings = []
                    for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type):
                        agent_corners = tracked_object.box.all_corners()
                        corners_x = [corner.x for corner in agent_corners]
                        corners_y = [corner.y for corner in agent_corners]
                        corners_x.append(corners_x[0])
                        corners_y.append(corners_y[0])
                        corner_xs.append([[corners_x]])
                        corner_ys.append([[corners_y]])
                        center_xs.append(tracked_object.center.x)
                        center_ys.append(tracked_object.center.y)
                        velocity_xs.append(tracked_object.velocity.x)
                        velocity_ys.append(tracked_object.velocity.y)
                        speeds.append(tracked_object.velocity.magnitude())
                        headings.append(tracked_object.center.heading)
                        agent_types.append(tracked_object_type.fullname)
                        track_ids.append(self._get_track_id(tracked_object.track_token))
                        track_tokens.append(tracked_object.track_token)
                    agent_states = BokehAgentStates(xs=corner_xs, ys=corner_ys, track_id=track_ids, track_token=track_tokens, agent_type=agent_types, center_xs=center_xs, center_ys=center_ys, velocity_xs=velocity_xs, velocity_ys=velocity_ys, speeds=speeds, headings=headings)
                    frame_dict[tracked_object_type_name] = ColumnDataSource(agent_states._asdict())
                self.data_sources[frame_index] = frame_dict
                self.data_source_condition.notify()

@dataclass
class AgentStateHeadingPlot(BaseScenarioPlot):
    """A dataclass for agent state heading plot."""
    data_sources: Dict[int, Dict[str, ColumnDataSource]] = field(default_factory=dict)
    plots: Dict[str, GlyphRenderer] = field(default_factory=dict)

    def update_plot(self, main_figure: Figure, frame_index: int, doc: Document) -> None:
        """
        Update the plot.
        :param main_figure: The plotting figure.
        :param frame_index: Frame index.
        :param doc: Bokeh document that the plot lives in.
        """
        if not self.data_source_condition:
            return
        self.render_event.set()
        with self.data_source_condition:
            while self.data_sources.get(frame_index, None) is None:
                self.data_source_condition.wait()

            def update_main_figure() -> None:
                """Wrapper for the main_figure update logic to support multi-threading."""
                data_sources = self.data_sources.get(frame_index, None)
                if not data_sources:
                    return
                for category, data_source in data_sources.items():
                    plot = self.plots.get(category, None)
                    data = dict(data_source.data)
                    if plot is None:
                        agent_color = simulation_tile_agent_style.get(category)
                        self.plots[category] = main_figure.multi_line(xs='trajectory_x', ys='trajectory_y', line_color=agent_color['line_color'], line_width=agent_color['line_width'], source=data)
                    else:
                        self.plots[category].data_source.data = data
                self.render_event.clear()
            doc.add_next_tick_callback(lambda: update_main_figure())

    def update_data_sources(self, history: SimulationHistory) -> None:
        """
        Update agent heading data sources.
        :param history: SimulationHistory time-series data.
        """
        if not self.data_source_condition:
            return
        with self.data_source_condition:
            for frame_index, sample in enumerate(history.data):
                if not isinstance(sample.observation, DetectionsTracks):
                    continue
                tracked_objects = sample.observation.tracked_objects
                frame_dict: Dict[str, Any] = {}
                for tracked_object_type_name, tracked_object_type in tracked_object_types.items():
                    trajectory_xs = []
                    trajectory_ys = []
                    for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type):
                        object_box = tracked_object.box
                        agent_trajectory = translate_longitudinally(object_box.center, distance=object_box.length / 2 + 1)
                        trajectory_xs.append([object_box.center.x, agent_trajectory.x])
                        trajectory_ys.append([object_box.center.y, agent_trajectory.y])
                    trajectories = ColumnDataSource(dict(trajectory_x=trajectory_xs, trajectory_y=trajectory_ys))
                    frame_dict[tracked_object_type_name] = trajectories
                self.data_sources[frame_index] = frame_dict
                self.data_source_condition.notify()

def _get_relevant_tracks(ego_pose: npt.NDArray[np.float64], ego_box: OrientedBox, ego_dx: float, ego_dy: float, tracks_poses: npt.NDArray[np.float64], tracks_boxes: List[OrientedBox], tracks_dxy: npt.NDArray[np.float64], time_step_size: float, time_horizon: float) -> npt.NDArray[np.int64]:
    """
    Find relevant tracks affecting time to collision, determined by overlapping boxes elongated according to current
      movement.
    :param ego_pose: Ego pose.
    :param ego_box: Oriented box of ego.
    :param ego_dx: Movement in x axis in global frame at each time_step_size.
    :param ego_dy: Movement in y axis in global frame at each time_step_size.
    :param tracks_poses: Pose for each track.
    :param tracks_boxes: Oriented box for each track.
    :param tracks_dxy: Tracks' movements in the global frame
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :param time_horizon: [s] Time horizon for collision checking.
    :return: Indices for tracks revlevant to time to collision calculations.
    """
    ego_elongated_box_center_pose: npt.NDArray[np.float64] = np.array([time_horizon / time_step_size / 2 * ego_dx + ego_pose[0], time_horizon / time_step_size / 2 * ego_dy + ego_pose[1], ego_pose[2]], dtype=np.float64)
    ego_elongated_box = OrientedBox(StateSE2(*ego_elongated_box_center_pose), _get_elongated_box_length(ego_box.length, ego_dx, ego_dy, time_step_size, time_horizon), ego_box.width, ego_box.height)
    tracks_elongated_box_center_poses: npt.NDArray[np.float64] = np.concatenate((time_horizon / time_step_size / 2 * tracks_dxy + tracks_poses[:, :2], tracks_poses[:, 2].reshape(-1, 1)), axis=1)
    tracks_elongated_boxes = [OrientedBox(StateSE2(*track_elongated_box_center_pose), _get_elongated_box_length(track_box.length, track_dxy[0], track_dxy[1], time_step_size, time_horizon), track_box.width, track_box.height) for track_box, track_dxy, track_elongated_box_center_pose in zip(tracks_boxes, tracks_dxy, tracks_elongated_box_center_poses)]
    relevant_tracks_mask = np.where([in_collision(ego_elongated_box, track_elongated_box) for track_elongated_box in tracks_elongated_boxes])[0]
    return relevant_tracks_mask

def _get_elongated_box_length(length: float, dx: float, dy: float, time_step_size: float, time_horizon: float) -> float:
    """
    Helper to find the length of an elongated box projected up to a given time horizon.
    :param length: The length of the OrientedBox.
    :param dx: Movement in x axis in global frame at each time_step_size.
    :param dy: Movement in y axis in global frame at each time_step_size.
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :param time_horizon: [s] Time horizon for collision checking.
    :return: Length of elonated box up to time horizon.
    """
    return float(length + np.hypot(dx * time_horizon / time_step_size, dy * time_horizon / time_step_size))

def _compute_time_to_collision_at_timestamp(timestamp: int, ego_state: EgoState, ego_speed: npt.NDArray[np.float64], tracks_poses: npt.NDArray[np.float64], tracks_speed: npt.NDArray[np.float64], tracks_boxes: List[OrientedBox], timestamps_at_fault_collisions: List[int], time_step_size: float, time_horizon: float, stopped_speed_threshold: float) -> Optional[float]:
    """
    Helper function for compute_time_to_collision. Computes time to collision value at given timestamp.
    :param timestamp: Time in time_us.
    :param ego_state: Ego state.
    :param ego_speed: Ego speed.
    :param tracks_poses: Pose for each track.
    :param tracks_speed: Array of tracks speeds.
    :param tracks_boxes: Oriented box for each track.
    :param timestamps_at_fault_collisions: List of timestamps corresponding to at-fault-collisions in the history.
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :param time_horizon: [s] Time horizon for collision checking.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return: Computed time to collision if available, otherwise None.
    """
    ego_in_at_fault_collision = timestamp in timestamps_at_fault_collisions
    if ego_in_at_fault_collision:
        return 0.0
    if len(tracks_poses) == 0 or ego_speed <= stopped_speed_threshold:
        return None
    displacement_info = _get_ego_tracks_displacement_info(ego_state, ego_speed, tracks_poses, tracks_speed, time_step_size)
    relevant_tracks_mask = _get_relevant_tracks(displacement_info.ego_pose, displacement_info.ego_box, displacement_info.ego_dx, displacement_info.ego_dy, tracks_poses, tracks_boxes, displacement_info.tracks_dxy, time_step_size, time_horizon)
    if not len(relevant_tracks_mask):
        return None
    for time_to_collision in np.arange(time_step_size, time_horizon, time_step_size):
        displacement_info.ego_pose[:2] += (displacement_info.ego_dx, displacement_info.ego_dy)
        projected_ego_box = OrientedBox.from_new_pose(displacement_info.ego_box, StateSE2(*displacement_info.ego_pose))
        tracks_poses[:, :2] += displacement_info.tracks_dxy
        for track_box, track_pose in zip(tracks_boxes[relevant_tracks_mask], tracks_poses[relevant_tracks_mask]):
            projected_track_box = OrientedBox.from_new_pose(track_box, StateSE2(*track_pose))
            if in_collision(projected_ego_box, projected_track_box):
                return float(time_to_collision)
    return None

def _get_ego_tracks_displacement_info(ego_state: EgoState, ego_speed: npt.NDArray[np.float64], tracks_poses: npt.NDArray[np.float64], tracks_speed: npt.NDArray[np.float64], time_step_size: float) -> EgoTracksDisplacementInfo:
    """
    Helper function for compute_time_to_collision. Gets relevent pose, displacement values for TTC calculations.
    :param ego_state: Ego state.
    :param ego_speed: Ego speed.
    :param tracks_poses: Array of tracks poses.
    :param tracks_speed: Array of tracks speeds.
    :param time_step_size: [s] Step size for the propagation of collision agents.
    :return: Relevent pose, displacement information for ego and tracks supporting time to collision calculations.
    """
    ego_pose: npt.NDArray[np.float64] = np.array([*ego_state.center], dtype=np.float64)
    ego_box = ego_state.car_footprint.oriented_box
    ego_dx = np.cos(ego_pose[2]) * ego_speed * time_step_size
    ego_dy = np.sin(ego_pose[2]) * ego_speed * time_step_size
    tracks_dxy: npt.NDArray[np.float64] = np.array([np.cos(tracks_poses[:, 2]) * tracks_speed * time_step_size, np.sin(tracks_poses[:, 2]) * tracks_speed * time_step_size], dtype=np.float64).T
    return EgoTracksDisplacementInfo(ego_pose, ego_box, ego_dx, ego_dy, tracks_dxy)

