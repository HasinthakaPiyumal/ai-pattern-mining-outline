# Cluster 96

class AbstractIDMPlanner(AbstractPlanner, ABC):
    """
    An interface for IDM based planners. Inherit from this class to use IDM policy to control the longitudinal
    behaviour of the ego.
    """

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
        self._policy = IDMPolicy(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max)
        self._planned_trajectory_samples = planned_trajectory_samples
        self._planned_trajectory_sample_interval = planned_trajectory_sample_interval
        self._planned_horizon = planned_trajectory_samples * planned_trajectory_sample_interval
        self._occupancy_map_radius = occupancy_map_radius
        self._max_path_length = self._policy.target_velocity * self._planned_horizon
        self._ego_token = 'ego_token'
        self._red_light_token = 'red_light'
        self._route_roadblocks: List[RoadBlockGraphEdgeMapObject] = []
        self._candidate_lane_edge_ids: Optional[List[str]] = None
        self._map_api: Optional[AbstractMap] = None
        self._ego_path: Optional[AbstractPath] = None
        self._ego_path_linestring: Optional[LineString] = None

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def _initialize_route_plan(self, route_roadblock_ids: List[str]) -> None:
        """
        Initializes the route plan with roadblocks.
        :param route_roadblock_ids: A list of roadblock ids that make up the ego's route
        """
        assert self._map_api, '_map_api has not yet been initialized. Please call the initialize() function first!'
        self._route_roadblocks = []
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)
        self._candidate_lane_edge_ids = [edge.id for block in self._route_roadblocks if block for edge in block.interior_edges]
        assert self._route_roadblocks, 'Cannot create route plan. No roadblocks were extracted from the given route_roadblock_ids!'

    def _get_expanded_ego_path(self, ego_state: EgoState, ego_idm_state: IDMAgentState) -> Polygon:
        """
        Returns the ego's expanded path as a Polygon.
        :return: A polygon representing the ego's path.
        """
        assert self._ego_path, '_ego_path has not yet been initialized. Please call the initialize() function first!'
        ego_footprint = ego_state.car_footprint
        path_to_go = trim_path(self._ego_path, max(self._ego_path.get_start_progress(), min(ego_idm_state.progress, self._ego_path.get_end_progress())), max(self._ego_path.get_start_progress(), min(ego_idm_state.progress + abs(self._policy.target_velocity) * self._planned_horizon, self._ego_path.get_end_progress())))
        expanded_path = path_to_linestring(path_to_go).buffer(ego_footprint.width / 2, cap_style=CAP_STYLE.square)
        return unary_union([expanded_path, ego_state.car_footprint.geometry])

    @staticmethod
    def _get_leading_idm_agent(ego_state: EgoState, agent: SceneObject, relative_distance: float) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state that represents another static and dynamic agent.
        :param agent: A scene object.
        :param relative_distance: [m] The relative distance from the scene object to the ego.
        :return: A IDM lead agents state
        """
        if isinstance(agent, Agent):
            longitudinal_velocity = agent.velocity.magnitude()
            relative_heading = principal_value(agent.center.heading - ego_state.center.heading)
            projected_velocity = transform(StateSE2(longitudinal_velocity, 0, 0), StateSE2(0, 0, relative_heading).as_matrix()).x
        else:
            projected_velocity = 0.0
        return IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=0.0)

    def _get_free_road_leading_idm_state(self, ego_state: EgoState, ego_idm_state: IDMAgentState) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state when there is no leading agent.
        :return: A IDM lead agents state.
        """
        assert self._ego_path, '_ego_path has not yet been initialized. Please call the initialize() function first!'
        projected_velocity = 0.0
        relative_distance = self._ego_path.get_end_progress() - ego_idm_state.progress
        length_rear = ego_state.car_footprint.length / 2
        return IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=length_rear)

    @staticmethod
    def _get_red_light_leading_idm_state(relative_distance: float) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state that represents a red light intersection.
        :param relative_distance: [m] The relative distance from the intersection to the ego.
        :return: A IDM lead agents state.
        """
        return IDMLeadAgentState(progress=relative_distance, velocity=0, length_rear=0)

    def _get_leading_object(self, ego_idm_state: IDMAgentState, ego_state: EgoState, occupancy_map: OccupancyMap, unique_observations: UniqueObjects) -> IDMLeadAgentState:
        """
        Get the most suitable leading object based on the occupancy map.
        :param ego_idm_state: The ego's IDM state at current iteration.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        """
        intersecting_agents = occupancy_map.intersects(self._get_expanded_ego_path(ego_state, ego_idm_state))
        if intersecting_agents.size > 0:
            intersecting_agents.insert(self._ego_token, ego_state.car_footprint.geometry)
            nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(self._ego_token)
            if self._red_light_token in nearest_id:
                return self._get_red_light_leading_idm_state(relative_distance)
            return self._get_leading_idm_agent(ego_state, unique_observations[nearest_id], relative_distance)
        else:
            return self._get_free_road_leading_idm_state(ego_state, ego_idm_state)

    def _construct_occupancy_map(self, ego_state: EgoState, observation: Observation) -> Tuple[OccupancyMap, UniqueObjects]:
        """
        Constructs an OccupancyMap from Observations.
        :param ego_state: Current EgoState
        :param observation: Observations of other agents and static objects in the scene.
        :return:
            - OccupancyMap.
            - A mapping between the object token and the object itself.
        """
        if isinstance(observation, DetectionsTracks):
            unique_observations = {detection.track_token: detection for detection in observation.tracked_objects.tracked_objects if np.linalg.norm(ego_state.center.array - detection.center.array) < self._occupancy_map_radius}
            return (STRTreeOccupancyMapFactory.get_from_boxes(list(unique_observations.values())), unique_observations)
        else:
            raise ValueError(f'IDM planner only supports DetectionsTracks. Got {observation.detection_type()}')

    def _propagate(self, ego: IDMAgentState, lead_agent: IDMLeadAgentState, tspan: float) -> None:
        """
        Propagate agent forward according to the IDM policy.
        :param ego: The ego's IDM state.
        :param lead_agent: The agent leading this agent.
        :param tspan: [s] The interval of time to propagate for.
        """
        solution = self._policy.solve_forward_euler_idm_policy(IDMAgentState(0, ego.velocity), lead_agent, tspan)
        ego.progress += solution.progress
        ego.velocity = max(solution.velocity, 0)

    def _get_planned_trajectory(self, ego_state: EgoState, occupancy_map: OccupancyMap, unique_observations: UniqueObjects) -> InterpolatedTrajectory:
        """
        Plan a trajectory w.r.t. the occupancy map.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        :return: A trajectory representing the predicted ego's position in future.
        """
        assert self._ego_path_linestring, '_ego_path_linestring has not yet been initialized. Please call the initialize() function first!'
        ego_progress = self._ego_path_linestring.project(Point(*ego_state.center.point.array))
        ego_idm_state = IDMAgentState(progress=ego_progress, velocity=ego_state.dynamic_car_state.center_velocity_2d.x)
        vehicle_parameters = ego_state.car_footprint.vehicle_parameters
        current_time_point = ego_state.time_point
        projected_ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)
        planned_trajectory: List[EgoState] = [projected_ego_state]
        for _ in range(self._planned_trajectory_samples):
            leading_agent = self._get_leading_object(ego_idm_state, ego_state, occupancy_map, unique_observations)
            self._propagate(ego_idm_state, leading_agent, self._planned_trajectory_sample_interval)
            current_time_point += TimePoint(int(self._planned_trajectory_sample_interval * 1000000.0))
            ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)
            planned_trajectory.append(ego_state)
        return InterpolatedTrajectory(planned_trajectory)

    def _idm_state_to_ego_state(self, idm_state: IDMAgentState, time_point: TimePoint, vehicle_parameters: VehicleParameters) -> EgoState:
        """
        Convert IDMAgentState to EgoState
        :param idm_state: The IDMAgentState to be converted.
        :param time_point: The TimePoint corresponding to the state.
        :param vehicle_parameters: VehicleParameters of the ego.
        """
        assert self._ego_path, '_ego_path has not yet been initialized. Please call the initialize() function first!'
        new_ego_center = self._ego_path.get_state_at_progress(max(self._ego_path.get_start_progress(), min(idm_state.progress, self._ego_path.get_end_progress())))
        return EgoState.build_from_center(center=StateSE2(new_ego_center.x, new_ego_center.y, new_ego_center.heading), center_velocity_2d=StateVector2D(idm_state.velocity, 0), center_acceleration_2d=StateVector2D(0, 0), tire_steering_angle=0.0, time_point=time_point, vehicle_parameters=vehicle_parameters)

    def _annotate_occupancy_map(self, traffic_light_data: List[TrafficLightStatusData], occupancy_map: OccupancyMap) -> None:
        """
        Add red light lane connectors on the route plan to the occupancy map. Note: the function works inline, hence,
        the occupancy map will be modified in this function.
        :param traffic_light_data: A list of all available traffic status data.
        :param occupancy_map: The occupancy map to be annotated.
        """
        assert self._map_api, '_map_api has not yet been initialized. Please call the initialize() function first!'
        assert self._candidate_lane_edge_ids is not None, '_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!'
        for data in traffic_light_data:
            if data.status == TrafficLightStatusType.RED and str(data.lane_connector_id) in self._candidate_lane_edge_ids:
                id_ = str(data.lane_connector_id)
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                occupancy_map.insert(f'{self._red_light_token}_{id_}', lane_conn.polygon)

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

class IDMAgents(AbstractObservation):
    """
    Simulate agents based on IDM policy.
    """

    def __init__(self, target_velocity: float, min_gap_to_lead_agent: float, headway_time: float, accel_max: float, decel_max: float, open_loop_detections_types: List[str], scenario: AbstractScenario, minimum_path_length: float=20, planned_trajectory_samples: Optional[int]=None, planned_trajectory_sample_interval: Optional[float]=None, radius: float=100):
        """
        Constructor for IDMAgents

        :param target_velocity: [m/s] Desired velocity in free traffic
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front
        :param accel_max: [m/s^2] maximum acceleration
        :param decel_max: [m/s^2] maximum deceleration (positive value)
        :param scenario: scenario
        :param open_loop_detections_types: The open-loop detection types to include.
        :param minimum_path_length: [m] The minimum path length to maintain.
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param radius: [m] Only agents within this radius around the ego will be simulated.
        """
        self.current_iteration = 0
        self._target_velocity = target_velocity
        self._min_gap_to_lead_agent = min_gap_to_lead_agent
        self._headway_time = headway_time
        self._accel_max = accel_max
        self._decel_max = decel_max
        self._scenario = scenario
        self._open_loop_detections_types: List[TrackedObjectType] = []
        self._minimum_path_length = minimum_path_length
        self._planned_trajectory_samples = planned_trajectory_samples
        self._planned_trajectory_sample_interval = planned_trajectory_sample_interval
        self._radius = radius
        self._idm_agent_manager: Optional[IDMAgentManager] = None
        self._initialize_open_loop_detection_types(open_loop_detections_types)

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0
        self._idm_agent_manager = None

    def _initialize_open_loop_detection_types(self, open_loop_detections: List[str]) -> None:
        """
        Initializes open-loop detections with the enum types from TrackedObjectType
        :param open_loop_detections: A list of open-loop detections types as strings
        :return: A list of open-loop detections types as strings as the corresponding TrackedObjectType
        """
        for _type in open_loop_detections:
            try:
                self._open_loop_detections_types.append(TrackedObjectType[_type])
            except KeyError:
                raise ValueError(f'The given detection type {_type} does not exist or is not supported!')

    def _get_idm_agent_manager(self) -> IDMAgentManager:
        """
        Create idm agent manager in case it does not already exists
        :return: IDMAgentManager
        """
        if not self._idm_agent_manager:
            agents, agent_occupancy = build_idm_agents_on_map_rails(self._target_velocity, self._min_gap_to_lead_agent, self._headway_time, self._accel_max, self._decel_max, self._minimum_path_length, self._scenario, self._open_loop_detections_types)
            self._idm_agent_manager = IDMAgentManager(agents, agent_occupancy, self._scenario.map_api)
        return self._idm_agent_manager

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        detections = self._get_idm_agent_manager().get_active_agents(self.current_iteration, self._planned_trajectory_samples, self._planned_trajectory_sample_interval)
        if self._open_loop_detections_types:
            open_loop_detections = self._get_open_loop_track_objects(self.current_iteration)
            detections.tracked_objects.tracked_objects.extend(open_loop_detections)
        return detections

    def update_observation(self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
        tspan = next_iteration.time_s - iteration.time_s
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(self.current_iteration)
        traffic_light_status: Dict[TrafficLightStatusType, List[str]] = defaultdict(list)
        for data in traffic_light_data:
            traffic_light_status[data.status].append(str(data.lane_connector_id))
        ego_state, _ = history.current_state
        self._get_idm_agent_manager().propagate_agents(ego_state, tspan, self.current_iteration, traffic_light_status, self._get_open_loop_track_objects(self.current_iteration), self._radius)

    def _get_open_loop_track_objects(self, iteration: int) -> List[TrackedObject]:
        """
        Get open-loop tracked objects from scenario.
        :param iteration: The simulation iteration.
        :return: A list of TrackedObjects.
        """
        detections = self._scenario.get_tracked_objects_at_iteration(iteration)
        return detections.tracked_objects.get_tracked_objects_of_types(self._open_loop_detections_types)

def build_idm_agents_on_map_rails(target_velocity: float, min_gap_to_lead_agent: float, headway_time: float, accel_max: float, decel_max: float, minimum_path_length: float, scenario: AbstractScenario, open_loop_detections_types: List[TrackedObjectType]) -> Tuple[UniqueIDMAgents, OccupancyMap]:
    """
    Build unique agents from a scenario. InterpolatedPaths are created for each agent according to their driven path

    :param target_velocity: Desired velocity in free traffic [m/s]
    :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
    :param headway_time: Desired time headway. The minimum possible time to the vehicle in front [s]
    :param accel_max: maximum acceleration [m/s^2]
    :param decel_max: maximum deceleration (positive value) [m/s^2]
    :param minimum_path_length: [m] The minimum path length
    :param scenario: scenario
    :param open_loop_detections_types: The open-loop detection types to include.
    :return: a dictionary of IDM agent uniquely identified by a track_token
    """
    unique_agents: UniqueIDMAgents = {}
    detections = scenario.initial_tracked_objects
    map_api = scenario.map_api
    ego_agent = scenario.get_ego_state_at_iteration(0).agent
    open_loop_detections = detections.tracked_objects.get_tracked_objects_of_types(open_loop_detections_types)
    init_agent_occupancy = STRTreeOccupancyMapFactory.get_from_boxes(open_loop_detections)
    init_agent_occupancy.insert(ego_agent.token, ego_agent.box.geometry)
    occupancy_map = STRTreeOccupancyMap({})
    desc = 'Converting detections to smart agents'
    agent: Agent
    for agent in tqdm(detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE), desc=desc, leave=False):
        if agent.track_token not in unique_agents:
            route, progress = get_starting_segment(agent, map_api)
            if route is None:
                continue
            state_on_path = route.baseline_path.get_nearest_pose_from_position(agent.center.point)
            box_on_baseline = OrientedBox.from_new_pose(agent.box, StateSE2(state_on_path.x, state_on_path.y, state_on_path.heading))
            if not init_agent_occupancy.intersects(box_on_baseline.geometry).is_empty():
                continue
            init_agent_occupancy.insert(agent.track_token, box_on_baseline.geometry)
            occupancy_map.insert(agent.track_token, box_on_baseline.geometry)
            if np.isnan(agent.velocity.array).any():
                ego_state = scenario.get_ego_state_at_iteration(0)
                logger.debug(f"Agents has nan velocity. Setting velocity to ego's velocity of {ego_state.dynamic_car_state.speed}")
                velocity = StateVector2D(ego_state.dynamic_car_state.speed, 0.0)
            else:
                velocity = StateVector2D(np.hypot(agent.velocity.x, agent.velocity.y), 0)
            initial_state = IDMInitialState(metadata=agent.metadata, tracked_object_type=agent.tracked_object_type, box=box_on_baseline, velocity=velocity, path_progress=progress, predictions=agent.predictions)
            target_velocity = route.speed_limit_mps or target_velocity
            unique_agents[agent.track_token] = IDMAgent(start_iteration=0, initial_state=initial_state, route=[route], policy=IDMPolicy(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max), minimum_path_length=minimum_path_length)
    return (unique_agents, occupancy_map)

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

def get_starting_segment(agent: Agent, map_api: AbstractMap) -> Tuple[Optional[LaneGraphEdgeMapObject], Optional[float]]:
    """
    Gets the map object that the agent is on and the progress along the segment.
    :param agent: The agent of interested.
    :param map_api: An AbstractMap instance.
    :return: GraphEdgeMapObject and progress along the segment. If no map object is found then None.
    """
    if map_api.is_in_layer(agent.center, SemanticMapLayer.LANE):
        layer = SemanticMapLayer.LANE
    elif map_api.is_in_layer(agent.center, SemanticMapLayer.INTERSECTION):
        layer = SemanticMapLayer.LANE_CONNECTOR
    else:
        return (None, None)
    segments: List[LaneGraphEdgeMapObject] = map_api.get_all_map_objects(agent.center, layer)
    if not segments:
        return (None, None)
    heading_diff = [segment.baseline_path.get_nearest_pose_from_position(agent.center).heading - agent.center.heading for segment in segments]
    closest_segment = segments[np.argmin(np.abs(heading_diff))]
    progress = closest_segment.baseline_path.get_nearest_arc_length_from_position(agent.center)
    return (closest_segment, progress)

class IDMPolicy:
    """
    An agent policy that describes the agent's behaviour w.r.t to a lead agent. The policy only controls the
    longitudinal states (progress, velocity) of the agent. This longitudinal states are used to propagate the agent
    along a given path.
    """

    def __init__(self, target_velocity: float, min_gap_to_lead_agent: float, headway_time: float, accel_max: float, decel_max: float):
        """
        Constructor for IDMPolicy

        :param target_velocity: Desired velocity in free traffic [m/s]
        :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
        :param headway_time: Desired time headway. The minimum possible time to the vehicle in front [s]
        :param accel_max: maximum acceleration [m/s^2]
        :param decel_max: maximum deceleration (positive value) [m/s^2]
        """
        self._target_velocity = target_velocity
        self._min_gap_to_lead_agent = min_gap_to_lead_agent
        self._headway_time = headway_time
        self._accel_max = accel_max
        self._decel_max = decel_max

    @property
    def idm_params(self) -> List[float]:
        """Returns the policy parameters as a list"""
        return [self._target_velocity, self._min_gap_to_lead_agent, self._headway_time, self._accel_max, self._decel_max]

    @property
    def target_velocity(self) -> float:
        """
        The policy's desired velocity in free traffic [m/s]
        :return: target velocity
        """
        return self._target_velocity

    @target_velocity.setter
    def target_velocity(self, target_velocity: float) -> None:
        """
        Sets the policy's desired velocity in free traffic [m/s]
        """
        self._target_velocity = target_velocity
        assert target_velocity > 0, f'The target velocity must be greater than 0! {target_velocity} > 0'

    @property
    def headway_time(self) -> float:
        """
        The policy's minimum possible time to the vehicle in front [s]
        :return: Desired time headway
        """
        return self._headway_time

    @property
    def decel_max(self) -> float:
        """
        The policy's maximum deceleration (positive value) [m/s^2]
        :return: Maximum deceleration
        """
        return self._decel_max

    @staticmethod
    def idm_model(time_points: List[float], state_variables: List[float], lead_agent: List[float], params: List[float]) -> List[Any]:
        """
        Defines the differential equations for IDM.

        :param state_variables: vector of the state variables:
                  state_variables = [x_agent: progress,
                                     v_agent: velocity]
        :param time_points: time A sequence of time points for which to solve for the state variables
        :param lead_agent: vector of the state variables for the lead vehicle:
                  lead_agent = [x_lead: progress,
                                v_lead: velocity,
                                l_r_lead: half length of the leading vehicle]
        :param params:vector of the parameters:
                  params = [target_velocity: desired velocity in free traffic,
                            min_gap_to_lead_agent: minimum relative distance to lead vehicle,
                            headway_time: desired time headway. The minimum possible time to the vehicle in front,
                            accel_max: maximum acceleration,
                            decel_max: maximum deceleration (positive value)]

        :return: system of differential equations
        """
        x_agent, v_agent = state_variables
        x_lead, v_lead, l_r_lead = lead_agent
        target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max = params
        acceleration_exponent = 4
        s_star = min_gap_to_lead_agent + v_agent * headway_time + v_agent * (v_agent - v_lead) / (2 * sqrt(accel_max * decel_max))
        s_alpha = max(x_lead - x_agent - l_r_lead, min_gap_to_lead_agent)
        x_dot = v_agent
        v_agent_dot = accel_max * (1 - (v_agent / target_velocity) ** acceleration_exponent - (s_star / s_alpha) ** 2)
        return [x_dot, v_agent_dot]

    def solve_forward_euler_idm_policy(self, agent: IDMAgentState, lead_agent: IDMLeadAgentState, sampling_time: float) -> IDMAgentState:
        """
        Solves Solves an initial value problem for a system of ODEs using forward euler.
        This has the benefit of being differentiable

        :param agent: the agent of interest
        :param lead_agent: the lead vehicle
        :param sampling_time: interval of integration
        :return: solution to the differential equations
        """
        params = self.idm_params
        x_dot, v_agent_dot = self.idm_model([], agent.to_array(), lead_agent.to_array(), params)
        return IDMAgentState(agent.progress + sampling_time * x_dot, agent.velocity + sampling_time * min(max(-self._decel_max, v_agent_dot), self._accel_max))

    def solve_odeint_idm_policy(self, agent: IDMAgentState, lead_agent: IDMLeadAgentState, sampling_time: float, solve_points: int=10) -> IDMAgentState:
        """
        Solves an initial value problem for a system of ODEs using scipy odeint

        :param agent: the agent of interest
        :param lead_agent: the lead vehicle
        :param sampling_time: interval of integration
        :param solve_points: number of points for temporal resolution
        :return: solution to the differential equations
        """
        t = np.linspace(0, sampling_time, solve_points)
        solution = odeint(self.idm_model, agent.to_array(), t, args=(lead_agent.to_array(), self.idm_params), tfirst=True)
        return IDMAgentState(solution[-1][0], solution[-1][1])

    def solve_ivp_idm_policy(self, agent: IDMAgentState, lead_agent: IDMLeadAgentState, sampling_time: float) -> IDMAgentState:
        """
        Solves an initial value problem for a system of ODEs using scipy RK45

        :param agent: the agent of interest
        :param lead_agent: the lead vehicle
        :param sampling_time: interval of integration
        :return: solution to the differential equations
        """
        t = (0, sampling_time)
        solution = solve_ivp(self.idm_model, t, agent.to_array(), args=(lead_agent.to_array(), self.idm_params), method='RK45')
        return IDMAgentState(solution.y[0][-1], solution.y[1][-1])

def _build_idm_agents(agents: List[Agent], map_api: AbstractMap, policy: IDMPolicy) -> UniqueIDMAgents:
    """
    Builds idm agents.
    :param agents: list of agents represented by Agent
    :param map_api: AbstractMap
    :param policy: IDM policy
    :return: A dictionary of unique agents
    """
    unique_agents: UniqueIDMAgents = {}
    for agent in agents:
        route, progress = get_starting_segment(agent, map_api)
        initial_state = IDMInitialState(metadata=agent.metadata, tracked_object_type=agent.tracked_object_type, box=agent.box, velocity=agent.velocity, path_progress=progress, predictions=None)
        unique_agents[str(agent.token)] = IDMAgent(start_iteration=0, initial_state=initial_state, route=[route], policy=policy, minimum_path_length=10)
    return unique_agents

def build_idm_manager(scene: Dict[str, Any], map_factory: AbstractMapFactory, policy: IDMPolicy) -> IDMAgentManager:
    """
    Builds IDMAgentManager from scene
    :param scene: scene dictionary
    :param map_factory: AbstractMapFactory
    :param policy: IDM policy
    :return: IDMAgentManager object
    """
    map_name = scene['map']['area']
    map_maps_db = map_factory.build_map_from_name(map_name)
    agents = from_scene_to_tracked_objects(scene['world']).get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
    unique_agents = _build_idm_agents(agents, map_maps_db, policy)
    occupancy_map = STRTreeOccupancyMapFactory().get_from_boxes(agents)
    idm_manager = IDMAgentManager(unique_agents, occupancy_map, map_maps_db)
    return idm_manager

class IDMPolicyTests(unittest.TestCase):
    """Tests implementation of IDMPolicy"""

    def setUp(self):
        """Test setup"""
        self.idm = IDMPolicy(target_velocity=30, min_gap_to_lead_agent=2, headway_time=1.5, accel_max=0.73, decel_max=1.67)
        self.sampling_time = 0.5
        self.agent = IDMAgentState(5, 3)
        self.lead_agent = IDMLeadAgentState(15, 2, 5)

    def test_idm_model(self):
        """Tests the model correctness"""
        model = self.idm.idm_model([], self.agent.to_array(), self.lead_agent.to_array(), self.idm.idm_params)
        self.assertEqual(3, model[0])
        self.assertAlmostEqual(-1.073366, model[1])

    def test_solve_forward_euler_idm_policy(self):
        """Tests expected behaviour of forward euler method"""
        solution = self.idm.solve_forward_euler_idm_policy(self.agent, self.lead_agent, self.sampling_time)
        self.assertEqual(6.5, solution.progress)
        self.assertAlmostEqual(2.46331699693, solution.velocity)

    def test_non_differential_idm_policy(self):
        """Tests expected behaviour of odeint integrator"""
        solution = self.idm.solve_odeint_idm_policy(self.agent, self.lead_agent, self.sampling_time, 2)
        self.assertAlmostEqual(6.3558523392415, solution.progress)
        self.assertAlmostEqual(2.4058965769308, solution.velocity)

    def test_solve_ivp_idm_policy(self):
        """Tests expected behaviour of inital value problem integrator"""
        solution = self.idm.solve_ivp_idm_policy(self.agent, self.lead_agent, self.sampling_time)
        self.assertAlmostEqual(6.355856711603, solution.progress)
        self.assertAlmostEqual(2.40590847399835, solution.velocity)

@nuplan_test(path='json/idm_manager/')
def test_idm_manager(scene: Dict[str, Any]) -> None:
    """
    Test idm agent manager behaviour when ego is in lane
    """
    simulation_step = 20
    idm_manager = build_idm_manager(scene, map_factory, policy)
    ego_agent = to_agent_state_from_scene(scene['ego'], get_pacifica_parameters(), to_cog=False)
    traffic_light_status = {TrafficLightStatusType.GREEN: cast(List[str], scene['active_lane_connectors']), TrafficLightStatusType.RED: cast(List[str], scene['inactive_lane_connectors'])}
    for step in range(simulation_step):
        idm_manager.propagate_agents(ego_state=ego_agent, tspan=0.5, iteration=0, traffic_light_status=traffic_light_status, open_loop_detections=[], radius=100)
    for geom1, geom2 in itertools.combinations(idm_manager.agent_occupancy.get_all_geometries(), 2):
        assert not geom1.intersects(geom2)

