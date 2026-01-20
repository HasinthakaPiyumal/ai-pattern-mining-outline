# Cluster 98

class IDMPlanner(AbstractIDMPlanner):
    """
    The IDM planner is composed of two parts:
        1. Path planner that constructs a route to the same road block as the goal pose.
        2. IDM policy controller to control the longitudinal movement of the ego along the planned route.
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
        super(IDMPlanner, self).__init__(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max, planned_trajectory_samples, planned_trajectory_sample_interval, occupancy_map_radius)
        self._initialized = False

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._map_api = initialization.map_api
        self._initialize_route_plan(initialization.route_roadblock_ids)
        self._initialized = False

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        ego_state, observations = current_input.history.current_state
        if not self._initialized:
            self._initialize_ego_path(ego_state)
            self._initialized = True
        occupancy_map, unique_observations = self._construct_occupancy_map(ego_state, observations)
        traffic_light_data = current_input.traffic_light_data
        self._annotate_occupancy_map(traffic_light_data, occupancy_map)
        return self._get_planned_trajectory(ego_state, occupancy_map, unique_observations)

    def _initialize_ego_path(self, ego_state: EgoState) -> None:
        """
        Initializes the ego path from the ground truth driven trajectory
        :param ego_state: The ego state at the start of the scenario.
        """
        route_plan, _ = self._breadth_first_search(ego_state)
        ego_speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        speed_limit = route_plan[0].speed_limit_mps or self._policy.target_velocity
        self._policy.target_velocity = speed_limit if speed_limit > ego_speed else ego_speed
        discrete_path = []
        for edge in route_plan:
            discrete_path.extend(edge.baseline_path.discrete_path)
        self._ego_path = create_path_from_se2(discrete_path)
        self._ego_path_linestring = path_to_linestring(discrete_path)

    def _get_starting_edge(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Get the starting edge based on ego state. If a lane graph object does not contain the ego state then
        the closest one is taken instead.
        :param ego_state: Current ego state.
        :return: The starting LaneGraphEdgeMapObject.
        """
        assert self._route_roadblocks is not None, '_route_roadblocks has not yet been initialized. Please call the initialize() function first!'
        assert len(self._route_roadblocks) >= 2, '_route_roadblocks should have at least 2 elements!'
        starting_edge = None
        closest_distance = math.inf
        for edge in self._route_roadblocks[0].interior_edges + self._route_roadblocks[1].interior_edges:
            if edge.contains_point(ego_state.center):
                starting_edge = edge
                break
            distance = edge.polygon.distance(ego_state.car_footprint.geometry)
            if distance < closest_distance:
                starting_edge = edge
                closest_distance = distance
        assert starting_edge, 'Starting edge for IDM path planning could not be found!'
        return starting_edge

    def _breadth_first_search(self, ego_state: EgoState) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Performs iterative breath first search to find a route to the target roadblock.
        :param ego_state: Current ego state.
        :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end roadblock. If unsuccessful a longest route is given.
        """
        assert self._route_roadblocks is not None, '_route_roadblocks has not yet been initialized. Please call the initialize() function first!'
        assert self._candidate_lane_edge_ids is not None, '_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!'
        starting_edge = self._get_starting_edge(ego_state)
        graph_search = BreadthFirstSearch(starting_edge, self._candidate_lane_edge_ids)
        offset = 1 if starting_edge.get_roadblock_id() == self._route_roadblocks[1].id else 0
        route_plan, path_found = graph_search.search(self._route_roadblocks[-1], len(self._route_roadblocks[offset:]))
        if not path_found:
            logger.warning('IDMPlanner could not find valid path to the target roadblock. Using longest route found instead')
        return (route_plan, path_found)

class TestBreadthFirstSearch(unittest.TestCase):
    """Test class for BreadthFirstSearch"""
    TEST_FILE_PATH = 'nuplan.planning.simulation.planner.utils.breadth_first_search'

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self._mock_edge = MagicMock(spec=LaneGraphEdgeMapObject)
        self._graph_search = BreadthFirstSearch(self._mock_edge, ['a'])
        self._mock_edge.id = 'a'
        self._mock_edge.get_roadblock_id.side_effect = ['1', '2', '3']
        self._mock_edge.outgoing_edges = [self._mock_edge]

    @patch(f'{TEST_FILE_PATH}.BreadthFirstSearch._check_end_condition')
    @patch(f'{TEST_FILE_PATH}.BreadthFirstSearch._check_goal_condition')
    @patch(f'{TEST_FILE_PATH}.BreadthFirstSearch._construct_path')
    def test__breadth_first_search(self, mock_construct_path: Mock, mock_check_goal_condition: Mock, mock_check_end_condition: Mock) -> None:
        """Test search()"""
        mock_check_goal_condition.side_effect = [False, False, True]
        mock_check_end_condition.side_effect = [False, False, False, False, False]
        _, path_found = self._graph_search.search(Mock(), 3)
        self.assertTrue(path_found)
        mock_check_goal_condition.assert_called()
        mock_check_end_condition.assert_called()
        mock_construct_path.assert_called_once()

    def test__construct_path(self) -> None:
        """Test _construct_path()"""
        mock_edge_1 = MagicMock(spec=LaneGraphEdgeMapObject)
        mock_edge_1.id = 'a'
        mock_edge_2 = MagicMock(spec=LaneGraphEdgeMapObject)
        mock_edge_2.id = 'b'
        self._graph_search._parent = {'a_2': mock_edge_2, 'b_1': None}
        path = self._graph_search._construct_path(mock_edge_1, 2)
        self.assertEqual(path, [mock_edge_2, mock_edge_1])

    def test__check_end_condition(self) -> None:
        """Test _check_end_condition()"""
        self.assertTrue(self._graph_search._check_end_condition(1, 0))
        self.assertFalse(self._graph_search._check_end_condition(1, 1))

    def test__check_goal_condition(self) -> None:
        """Test _check_goal_condition()"""
        mock_edge = MagicMock(spec=LaneGraphEdgeMapObject)
        mock_edge.get_roadblock_id.side_effect = ['1', '2', '3', '3']
        mock_target_block = MagicMock(spec=LaneGraphEdgeMapObject)
        mock_target_block.id = '3'
        self.assertFalse(self._graph_search._check_goal_condition(mock_edge, mock_target_block, 0, 3))
        self.assertFalse(self._graph_search._check_goal_condition(mock_edge, mock_target_block, 3, 3))
        self.assertFalse(self._graph_search._check_goal_condition(mock_edge, mock_target_block, 0, 3))
        self.assertTrue(self._graph_search._check_goal_condition(mock_edge, mock_target_block, 3, 3))

