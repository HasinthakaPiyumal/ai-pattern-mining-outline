# Cluster 97

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

def trim_path(path: AbstractPath, start: float, end: float) -> List[ProgressStateSE2]:
    """
    Returns a trimmed path to be between given start and end progress. Everything else is discarded.
    :param path: the path to be trimmed
    :param start: the progress where the path should start.
    :param end: the progress where the path should end.
    :return: the trimmed discrete sampled path starting and ending from the given progress
    """
    start_progress = path.get_start_progress()
    end_progress = path.get_end_progress()
    assert start <= end, f'Start progress has to be less than the end progress {start} <= {end}'
    assert start_progress <= start, f'Start progress exceeds path! {start_progress} <= {start}'
    assert end <= end_progress, f'End progress exceeds path! {end} <= {end_progress}'
    start_state, end_state = path.get_state_at_progresses([start, end])
    progress_list: npt.NDArray[np.float_] = np.array([point.progress for point in path.get_sampled_path()])
    trim_front_indices = np.argwhere(progress_list > start)
    trim_tail_indices = np.argwhere(progress_list < end)
    if trim_front_indices.size > 0:
        trim_front_index = trim_front_indices.flatten()[0]
    else:
        return path.get_sampled_path()[-2:]
    if trim_tail_indices.size > 0:
        trim_end_index = trim_tail_indices.flatten()[-1]
    else:
        return path.get_sampled_path()[:2]
    return [start_state] + path.get_sampled_path()[trim_front_index:trim_end_index + 1] + [end_state]

def path_to_linestring(path: List[StateSE2]) -> LineString:
    """
    Converts a List of StateSE2 into a LineString
    :param path: path to be converted
    :return: LineString.
    """
    return LineString([(point.x, point.y) for point in path])

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

def create_path_from_se2(states: List[StateSE2]) -> InterpolatedPath:
    """
    Constructs an InterpolatedPath from a list of StateSE2.
    :param states: Waypoints to construct an InterpolatedPath.
    :return: InterpolatedPath.
    """
    progress_list = calculate_progress(states)
    progress_diff = np.diff(progress_list)
    repeated_states_mask = np.isclose(progress_diff, 0.0)
    progress_states = [ProgressStateSE2(progress=progress, x=point.x, y=point.y, heading=point.heading) for point, progress, is_repeated in zip(states, progress_list, repeated_states_mask) if not is_repeated]
    return InterpolatedPath(progress_states)

class MockIDMPlanner(AbstractIDMPlanner):
    """
    Mock IDMPlanner class for testing the AbstractIDMPlanner interface
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
        super(MockIDMPlanner, self).__init__(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max, planned_trajectory_samples, planned_trajectory_sample_interval, occupancy_map_radius)
        self._scenario = MockAbstractScenario()
        self._scenario_buffer = 10

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        self._map_api = initialization[0].map_api
        self._initialize_route_plan(initialization[0].route_roadblock_ids)
        self._ego_path: InterpolatedPath = create_path_from_ego_state(self._scenario.get_ego_future_trajectory(0, self._scenario_buffer, 10))
        self._ego_path_linestring = LineString()

    def compute_planner_trajectory(self, current_input: List[PlannerInput]) -> List[AbstractTrajectory]:
        """Inherited, see superclass."""
        return [InterpolatedTrajectory(self._ego_path.get_sampled_path())]

def create_path_from_ego_state(states: List[EgoState]) -> InterpolatedPath:
    """
    Constructs an InterpolatedPath from a list of EgoState.
    :param states: waypoints to construct an InterpolatedPath.
    :return InterpolatedPath.
    """
    return create_path_from_se2(ego_path_to_se2(states))

def calculate_progress(path: List[StateSE2]) -> List[float]:
    """
    Calculate the cumulative progress of a given path

    :param path: a path consisting of StateSE2 as waypoints
    :return: a cumulative list of progress
    """
    x_position = [point.x for point in path]
    y_position = [point.y for point in path]
    x_diff = np.diff(x_position)
    y_diff = np.diff(y_position)
    points_diff: npt.NDArray[np.float_] = np.concatenate(([x_diff], [y_diff]), axis=0)
    progress_diff = np.append(0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff).tolist()

def ego_path_to_se2(path: List[EgoState]) -> List[StateSE2]:
    """
    Convert a list of EgoState into a list of StateSE2.
    :param path: The path to be converted.
    :return: A list of StateSE2.
    """
    return [state.center for state in path]

def ego_path_to_linestring(path: List[EgoState]) -> LineString:
    """
    Converts a List of EgoState into a LineString
    :param path: path to be converted
    :return: LineString.
    """
    return path_to_linestring(ego_path_to_se2(path))

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

def trim_path_up_to_progress(path: AbstractPath, progress: float) -> List[ProgressStateSE2]:
    """
    Returns a trimmed path where the starting pose is starts at the given progress. Everything before is discarded
    :param path: the path to be trimmed
    :param progress: the progress where the path should start.
    :return: the trimmed discrete sampled path starting from the given progress
    """
    start_progress = path.get_start_progress()
    end_progress = path.get_end_progress()
    assert start_progress <= progress <= end_progress, f'Progress exceeds path! {start_progress} <= {progress} <= {end_progress}'
    cut_path = [path.get_state_at_progress(progress)]
    progress_list: npt.NDArray[np.float_] = np.array([point.progress for point in path.get_sampled_path()])
    trim_indices = np.argwhere(progress_list > progress)
    if trim_indices.size > 0:
        trim_index = trim_indices.flatten()[0]
        cut_path += path.get_sampled_path()[trim_index:]
        return cut_path
    return path.get_sampled_path()[-2:]

def convert_se2_path_to_progress_path(path: List[StateSE2]) -> List[ProgressStateSE2]:
    """
    Converts a list of StateSE2 to a list of ProgressStateSE2

    :return: a list of ProgressStateSE2
    """
    progress_list = calculate_progress(path)
    return [ProgressStateSE2(progress=progress, x=point.x, y=point.y, heading=point.heading) for point, progress in zip(path, progress_list)]

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

class TestInterpolatedPath(unittest.TestCase):
    """Tests implementation of InterpolatedPath."""

    def setUp(self) -> None:
        """Test setup."""
        self.path = [StateSE2(0, 0, 0), StateSE2(3, 4, 1), StateSE2(7, 7, 2), StateSE2(10, 10, 3)]
        self.interpolated_path = InterpolatedPath(convert_se2_path_to_progress_path(self.path))

    def test_get_start_progress(self) -> None:
        """Check start progress"""
        self.assertEqual(self.interpolated_path.get_start_progress(), self.interpolated_path.get_sampled_path()[0].progress)

    def test_get_end_progress(self) -> None:
        """Check end progress"""
        self.assertEqual(self.interpolated_path.get_end_progress(), self.interpolated_path.get_sampled_path()[-1].progress)

    def test_get_state_at_progress(self) -> None:
        """Check if the interpolated states are calculated correctly progress"""
        state = self.interpolated_path.get_state_at_progress(5)
        self.assertEqual(5, state.progress)
        self.assertEqual(3, state.x)
        self.assertEqual(4, state.y)
        self.assertEqual(1, state.heading)

    def test_get_state_at_progresses(self) -> None:
        """Check if the interpolated states are calculated correctly given a list of progresses."""
        states = self.interpolated_path.get_state_at_progresses([0, 5])
        self.assertEqual(0, states[0].progress)
        self.assertEqual(0, states[0].x)
        self.assertEqual(0, states[0].y)
        self.assertEqual(0, states[0].heading)
        self.assertEqual(5, states[1].progress)
        self.assertEqual(3, states[1].x)
        self.assertEqual(4, states[1].y)
        self.assertEqual(1, states[1].heading)

    def test_get_state_at_progress_expect_throw(self) -> None:
        """Check if assertion is raised for invalid calls"""
        self.assertRaises(AssertionError, self.interpolated_path.get_state_at_progress, 100)
        self.assertRaises(AssertionError, self.interpolated_path.get_state_at_progress, -1)

    def test_get_sampled_path(self) -> None:
        """Test if the sampled path is the same as the one originally given"""
        sample_path = self.interpolated_path.get_sampled_path()
        self.assertEqual(self.interpolated_path._path, sample_path)

    def test_trimmed_path_up_to_progress(self) -> None:
        """Test if path is trimmed correctly"""
        sample_path = trim_path_up_to_progress(self.interpolated_path, 2)
        self.assertEqual([self.interpolated_path.get_state_at_progress(2)] + self.interpolated_path._path[1:], sample_path)
        trimmed_sample_path = trim_path_up_to_progress(self.interpolated_path, 5)
        self.assertEqual(self.interpolated_path._path[1:], trimmed_sample_path)

    def test_trim_path(self) -> None:
        """Test if path is trimmed correctly"""
        sample_path = trim_path(self.interpolated_path, 2, 10)
        self.assertEqual([self.interpolated_path.get_state_at_progress(2)] + [self.interpolated_path._path[1]] + [self.interpolated_path.get_state_at_progress(10)], sample_path)
        sample_path = trim_path(self.interpolated_path, 2, 3)
        self.assertEqual([self.interpolated_path.get_state_at_progress(2)] + [self.interpolated_path.get_state_at_progress(3)], sample_path)
        sample_path = trim_path(self.interpolated_path, 0, 0)
        self.assertEqual(self.interpolated_path._path[:2], sample_path)

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

