# Cluster 2

def interpolate_waypoints(waypoints: List[Waypoint], trajectory_sampling: TrajectorySampling) -> List[Waypoint]:
    """
    Interpolates a list of waypoints given sampling time and horizon, starting at the first waypoint timestamp.
    :param waypoints: The sample waypoints
    :param trajectory_sampling: The sampling parameters
    :return: A list of interpolated waypoints
    """
    waypoint_trajectory = InterpolatedTrajectory(waypoints)
    start_time_us = waypoints[0].time_us
    end_time_us = waypoints[-1].time_us
    time_horizon_us = int(trajectory_sampling.time_horizon * 1000000.0)
    step_time_us = int(trajectory_sampling.step_time * 1000000.0)
    max_horizon_time = min(end_time_us, start_time_us + time_horizon_us + step_time_us)
    interpolation_times = np.arange(start_time_us, max_horizon_time, step_time_us)
    interpolated_waypoints = [waypoint_trajectory.get_state_at_time(TimePoint(int(interpolation_time))) for interpolation_time in interpolation_times]
    return interpolated_waypoints

def get_interpolated_waypoints(lidar_pc: LidarPc, future_trajectory_sampling: TrajectorySampling) -> Dict[str, List[Waypoint]]:
    """
    Gets the interpolated future waypoints for the agents detected in the given LidarPC. The sampling is determined
    by the horizon length and the sampling time.
    :param lidar_pc: The starting lidar pc
    :param future_trajectory_sampling: Sampling parameters for future predictions
    :return: A dict containing interpolated waypoints for each agent track_token, empty if no waypoint available
    """
    horizon_end = lidar_pc.timestamp + int(future_trajectory_sampling.time_horizon * 1000000.0)
    future_waypoints = {box.track_token: get_waypoints_for_agent(box, horizon_end) for box in lidar_pc.lidar_boxes}
    agents_interpolated_waypoints = {}
    for track_token, waypoints in future_waypoints.items():
        if len(waypoints) >= 2:
            interpolated = interpolate_waypoints(waypoints, future_trajectory_sampling)
            agents_interpolated_waypoints[track_token] = interpolated if len(interpolated) > 1 else []
        else:
            agents_interpolated_waypoints[track_token] = []
    return agents_interpolated_waypoints

class TestPredictionConstruction(unittest.TestCase):
    """Tests free function for prediction construction given future ground truth"""

    @patch('nuplan.database.nuplan_db_orm.prediction_construction.LidarBox', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.prediction_construction.StateSE2', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.prediction_construction.OrientedBox', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.prediction_construction.StateVector2D', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.prediction_construction.TimePoint', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.prediction_construction.Waypoint', autospec=True)
    def test__waypoint_from_lidar_box(self, waypoint: Mock, time_point: Mock, state_vector: Mock, oriented_box: Mock, state_se2: Mock, lidar_box: Mock) -> None:
        """Tests Waypoint creation from LidarBox"""
        lidar_box.translation = ['x', 'y']
        lidar_box.yaw = 'yaw'
        lidar_box.size = ['w', 'l', 'h']
        lidar_box.vx = 'vx'
        lidar_box.vy = 'vy'
        lidar_box.timestamp = 'timestamp'
        result = _waypoint_from_lidar_box(lidar_box)
        state_se2.assert_called_once_with('x', 'y', 'yaw')
        oriented_box.assert_called_once_with(state_se2.return_value, width='w', length='l', height='h')
        state_vector.assert_called_once_with('vx', 'vy')
        time_point.assert_called_once_with('timestamp')
        waypoint.assert_called_once_with(time_point.return_value, oriented_box.return_value, state_vector.return_value)
        self.assertEqual(waypoint.return_value, result)

    @patch('nuplan.database.nuplan_db_orm.prediction_construction.LidarBox', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.prediction_construction._waypoint_from_lidar_box', autospec=True)
    def test_get_waypoints_for_agent(self, waypoint_from_lidar_box: Mock, lidar_box: Mock) -> None:
        """Tests extraction of future waypoints for a single agent"""
        end_timestamp = 5
        lidar_box.timestamp = 0

        def increase_timestamp() -> Any:
            """Increases the lidar_box timestamp"""
            lidar_box.timestamp += 1
            return DEFAULT
        type(lidar_box).next = PropertyMock(return_value=lidar_box, side_effect=increase_timestamp)
        result = get_waypoints_for_agent(lidar_box, end_timestamp)
        calls = [call(lidar_box)] * 5
        waypoint_from_lidar_box.assert_has_calls(calls)
        self.assertTrue(5, len(result))

    @patch('nuplan.database.nuplan_db_orm.prediction_construction.LidarBox', autospec=True)
    def test_get_waypoints_for_agent_empty_on_invalid_time(self, lidar_box: Mock) -> None:
        """Tests extraction of future waypoints for a single agent"""
        end_timestamp = 1
        lidar_box.timestamp = 2
        result = get_waypoints_for_agent(lidar_box, end_timestamp)
        self.assertEqual([], result)

    @patch('nuplan.database.nuplan_db_orm.prediction_construction.InterpolatedTrajectory', autospec=True)
    @patch('numpy.arange')
    @patch('nuplan.database.nuplan_db_orm.prediction_construction.TimePoint', autospec=True)
    def test_interpolate_waypoints(self, time_point: Mock, arange: Mock, interpolated_trajectory: Mock) -> None:
        """Tests interpolation of waypoints for a single agent"""
        waypoints = [Mock(time_us=0, spec_set=Waypoint)]
        arange.return_value = [1.12, 2.23]
        time_point.side_effect = ['tp1', 'tp2']
        trajectory_sampling = Mock(time_horizon=5, step_time=1, spec=TrajectorySampling)
        result = interpolate_waypoints(waypoints, trajectory_sampling)
        arange.assert_called_once_with(0, 5 * 1000000.0, 1 * 1000000.0)
        time_point_calls = [call(1), call(2)]
        time_point.assert_has_calls(time_point_calls)
        calls = [call('tp1'), call('tp2')]
        interpolated_trajectory.return_value.get_state_at_time.assert_has_calls(calls)
        self.assertEqual(result, [interpolated_trajectory.return_value.get_state_at_time.return_value] * 2)

    @patch('nuplan.database.nuplan_db_orm.prediction_construction.get_waypoints_for_agent', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.prediction_construction.interpolate_waypoints', autospec=True)
    def test_get_interpolated_waypoints(self, mock_interpolate_waypoints: Mock, mock_get_waypoints_for_agent: Mock) -> None:
        """Tests extraction and interpolation of waypoints for a list of agents"""
        box_1 = Mock(track_token='1')
        box_2 = Mock(track_token='2')
        mock_lidar_pc = Mock(timestamp=0, lidar_boxes=[box_1, box_2])
        future_trajectory_sampling = Mock(time_horizon=5)
        mock_get_waypoints_for_agent.side_effect = ['waypoints_1', 'waypoints_2']
        result = get_interpolated_waypoints(mock_lidar_pc, future_trajectory_sampling)
        get_waypoints_calls = [call(box_1, 5 * 1000000.0), call(box_2, 5 * 1000000.0)]
        mock_get_waypoints_for_agent.assert_has_calls(get_waypoints_calls)
        interpolate_waypoints_calls = [call('waypoints_1', future_trajectory_sampling), call('waypoints_2', future_trajectory_sampling)]
        mock_interpolate_waypoints.assert_has_calls(interpolate_waypoints_calls)
        self.assertEqual(result, {'1': mock_interpolate_waypoints.return_value, '2': mock_interpolate_waypoints.return_value})

    @patch('nuplan.database.nuplan_db_orm.prediction_construction.get_waypoints_for_agent', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.prediction_construction.interpolate_waypoints', autospec=True)
    def test_get_interpolated_waypoints_no_waypoitns(self, mock_interpolate_waypoints: Mock, mock_get_waypoints_for_agent: Mock) -> None:
        """Tests extraction and interpolation of waypoints for a list of agents"""
        box_1 = Mock(track_token='1')
        box_2 = Mock(track_token='2')
        mock_lidar_pc = Mock(timestamp=0, lidar_boxes=[box_1, box_2])
        future_trajectory_sampling = Mock(time_horizon=5)
        mock_get_waypoints_for_agent.side_effect = [[], ['waypoint']]
        result = get_interpolated_waypoints(mock_lidar_pc, future_trajectory_sampling)
        get_waypoints_calls = [call(box_1, 5 * 1000000.0), call(box_2, 5 * 1000000.0)]
        mock_get_waypoints_for_agent.assert_has_calls(get_waypoints_calls)
        mock_interpolate_waypoints.assert_not_called()
        self.assertEqual(result, {'1': [], '2': []})

class EgoState(InterpolatableState):
    """Represent the current state of ego, along with its dynamic attributes."""

    def __init__(self, car_footprint: CarFootprint, dynamic_car_state: DynamicCarState, tire_steering_angle: float, is_in_auto_mode: bool, time_point: TimePoint):
        """
        :param car_footprint: The CarFootprint of Ego
        :param dynamic_car_state: The current dynamical state of ego
        :param tire_steering_angle: The current steering angle of the tires
        :param is_in_auto_mode: If the state refers to car in autonomous mode
        :param time_point: Time stamp of the state
        """
        self._car_footprint = car_footprint
        self._tire_steering_angle = tire_steering_angle
        self._is_in_auto_mode = is_in_auto_mode
        self._time_point = time_point
        self._dynamic_car_state = dynamic_car_state

    @cached_property
    def waypoint(self) -> Waypoint:
        """
        :return: waypoint corresponding to this ego state
        """
        return Waypoint(time_point=self.time_point, oriented_box=self.car_footprint, velocity=self.dynamic_car_state.rear_axle_velocity_2d)

    @staticmethod
    def deserialize(vector: List[Union[int, float]], vehicle: VehicleParameters) -> EgoState:
        """
        Deserialize object, ordering kept for backward compatibility
        :param vector: List of variables for deserialization
        :param vehicle: Vehicle parameters
        """
        if len(vector) != 9:
            raise RuntimeError(f'Expected a vector of size 9, got {len(vector)}')
        return EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(vector[1], vector[2], vector[3]), rear_axle_velocity_2d=StateVector2D(vector[4], vector[5]), rear_axle_acceleration_2d=StateVector2D(vector[6], vector[7]), tire_steering_angle=vector[8], time_point=TimePoint(int(vector[0])), vehicle_parameters=vehicle)

    def __iter__(self) -> Iterable[Union[int, float]]:
        """Iterable over ego parameters"""
        return iter((self.time_us, self.rear_axle.x, self.rear_axle.y, self.rear_axle.heading, self.dynamic_car_state.rear_axle_velocity_2d.x, self.dynamic_car_state.rear_axle_velocity_2d.y, self.dynamic_car_state.rear_axle_acceleration_2d.x, self.dynamic_car_state.rear_axle_acceleration_2d.y, self.tire_steering_angle))

    def to_split_state(self) -> SplitState:
        """Inherited, see superclass."""
        linear_states = [self.time_us, self.rear_axle.x, self.rear_axle.y, self.dynamic_car_state.rear_axle_velocity_2d.x, self.dynamic_car_state.rear_axle_velocity_2d.y, self.dynamic_car_state.rear_axle_acceleration_2d.x, self.dynamic_car_state.rear_axle_acceleration_2d.y, self.tire_steering_angle]
        angular_states = [self.rear_axle.heading]
        fixed_state = [self.car_footprint.vehicle_parameters]
        return SplitState(linear_states, angular_states, fixed_state)

    @staticmethod
    def from_split_state(split_state: SplitState) -> EgoState:
        """Inherited, see superclass."""
        if len(split_state) != 10:
            raise RuntimeError(f'Expected a variable state vector of size 10, got {len(split_state)}')
        return EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(split_state.linear_states[1], split_state.linear_states[2], split_state.angular_states[0]), rear_axle_velocity_2d=StateVector2D(split_state.linear_states[3], split_state.linear_states[4]), rear_axle_acceleration_2d=StateVector2D(split_state.linear_states[5], split_state.linear_states[6]), tire_steering_angle=split_state.linear_states[7], time_point=TimePoint(int(split_state.linear_states[0])), vehicle_parameters=split_state.fixed_states[0])

    @property
    def is_in_auto_mode(self) -> bool:
        """
        :return: True if ego is in auto mode, False otherwise.
        """
        return self._is_in_auto_mode

    @property
    def car_footprint(self) -> CarFootprint:
        """
        Getter for Ego's Car footprint
        :return: Ego's car footprint
        """
        return self._car_footprint

    @property
    def tire_steering_angle(self) -> float:
        """
        Getter for Ego's tire steering angle
        :return: Ego's tire steering angle
        """
        return self._tire_steering_angle

    @property
    def center(self) -> StateSE2:
        """
        Getter for Ego's center pose (center of mass)
        :return: Ego's center pose
        """
        return self._car_footprint.oriented_box.center

    @property
    def rear_axle(self) -> StateSE2:
        """
        Getter for Ego's rear axle pose (middle of the rear axle)
        :return: Ego's rear axle pose
        """
        return self.car_footprint.rear_axle

    @property
    def time_point(self) -> TimePoint:
        """
        Time stamp of the EgoState
        :return: EgoState time stamp
        """
        return self._time_point

    @property
    def time_us(self) -> int:
        """
        Time in micro seconds
        :return: [us].
        """
        return int(self.time_point.time_us)

    @property
    def time_seconds(self) -> float:
        """
        Time in seconds
        :return: [s]
        """
        return float(self.time_us * 1e-06)

    @property
    def dynamic_car_state(self) -> DynamicCarState:
        """
        Getter for the dynamic car state of Ego.
        :return: The dynamic car state
        """
        return self._dynamic_car_state

    @property
    def scene_object_metadata(self) -> SceneObjectMetadata:
        """
        :return: create scene object metadata
        """
        return SceneObjectMetadata(token='ego', track_token='ego', track_id=-1, timestamp_us=self.time_us)

    @cached_property
    def agent(self) -> AgentState:
        """
        Casts the EgoState to an Agent object.
        :return: An Agent object with the parameters of EgoState
        """
        return AgentState(metadata=self.scene_object_metadata, tracked_object_type=TrackedObjectType.EGO, oriented_box=self.car_footprint.oriented_box, velocity=self.dynamic_car_state.center_velocity_2d)

    @classmethod
    def build_from_rear_axle(cls, rear_axle_pose: StateSE2, rear_axle_velocity_2d: StateVector2D, rear_axle_acceleration_2d: StateVector2D, tire_steering_angle: float, time_point: TimePoint, vehicle_parameters: VehicleParameters, is_in_auto_mode: bool=True, angular_vel: float=0.0, angular_accel: float=0.0, tire_steering_rate: float=0.0) -> EgoState:
        """
        Initializer using raw parameters, assumes that the reference frame is CAR_POINT.REAR_AXLE
        :param rear_axle_pose: Pose of ego's rear axle
        :param rear_axle_velocity_2d: Vectorial velocity of Ego's rear axle
        :param rear_axle_acceleration_2d: Vectorial acceleration of Ego's rear axle
        :param angular_vel: Angular velocity of Ego
        :param angular_accel: Angular acceleration of Ego,
        :param tire_steering_angle: Angle of the tires
        :param is_in_auto_mode: True if ego is in auto mode, false otherwise
        :param time_point: Timestamp of the ego state
        :param vehicle_parameters: Vehicle parameters
        :param tire_steering_rate: Steering rate of tires [rad/s]
        :return: The initialized EgoState
        """
        car_footprint = CarFootprint.build_from_rear_axle(rear_axle_pose=rear_axle_pose, vehicle_parameters=vehicle_parameters)
        dynamic_ego_state = DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist=car_footprint.rear_axle_to_center_dist, rear_axle_velocity_2d=rear_axle_velocity_2d, rear_axle_acceleration_2d=rear_axle_acceleration_2d, angular_velocity=angular_vel, angular_acceleration=angular_accel, tire_steering_rate=tire_steering_rate)
        return cls(car_footprint=car_footprint, dynamic_car_state=dynamic_ego_state, tire_steering_angle=tire_steering_angle, time_point=time_point, is_in_auto_mode=is_in_auto_mode)

    @classmethod
    def build_from_center(cls, center: StateSE2, center_velocity_2d: StateVector2D, center_acceleration_2d: StateVector2D, tire_steering_angle: float, time_point: TimePoint, vehicle_parameters: VehicleParameters, is_in_auto_mode: bool=True, angular_vel: float=0.0, angular_accel: float=0.0) -> EgoState:
        """
        Initializer using raw parameters, assumes that the reference frame is center frame
        :param center: Pose of ego center
        :param center_velocity_2d: Vectorial velocity of Ego's center
        :param center_acceleration_2d: Vectorial acceleration of Ego's center
        :param tire_steering_angle: Angle of the tires
        :param time_point: Timestamp of the ego state
        :param vehicle_parameters: Vehicle parameters
        :param is_in_auto_mode: True if ego is in auto mode, false otherwise, defaults to True
        :param angular_vel: Angular velocity of Ego, defaults to 0.0
        :param angular_accel: Angular acceleration of Ego, defaults to 0.0
        :return: The initialized EgoState
        """
        car_footprint = CarFootprint.build_from_center(center, vehicle_parameters)
        rear_axle_to_center_dist = car_footprint.rear_axle_to_center_dist
        displacement = StateVector2D(-rear_axle_to_center_dist, 0.0)
        rear_axle_velocity_2d = get_velocity_shifted(displacement, center_velocity_2d, angular_vel)
        rear_axle_acceleration_2d = get_acceleration_shifted(displacement, center_acceleration_2d, angular_vel, angular_accel)
        dynamic_ego_state = DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist=rear_axle_to_center_dist, rear_axle_velocity_2d=rear_axle_velocity_2d, rear_axle_acceleration_2d=rear_axle_acceleration_2d, angular_velocity=angular_vel, angular_acceleration=angular_accel)
        return cls(car_footprint=car_footprint, dynamic_car_state=dynamic_ego_state, tire_steering_angle=tire_steering_angle, time_point=time_point, is_in_auto_mode=is_in_auto_mode)

class Agent(AgentTemporalState, AgentState):
    """
    AgentState with future and past trajectory.
    """

    def __init__(self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, velocity: StateVector2D, metadata: SceneObjectMetadata, angular_velocity: Optional[float]=None, predictions: Optional[List[PredictedTrajectory]]=None, past_trajectory: Optional[PredictedTrajectory]=None):
        """
        Representation of an Agent in the scene (Vehicles, Pedestrians, Bicyclists and GenericObjects).
        :param tracked_object_type: Type of the current agent.
        :param oriented_box: Geometrical representation of the Agent.
        :param velocity: Velocity (vectorial) of Agent.
        :param metadata: Agent's metadata.
        :param angular_velocity: The scalar angular velocity of the agent, if available.
        :param predictions: Optional list of (possibly multiple) predicted trajectories.
        :param past_trajectory: Optional past trajectory of this agent.
        """
        AgentTemporalState.__init__(self, initial_time_stamp=TimePoint(metadata.timestamp_us), predictions=predictions, past_trajectory=past_trajectory)
        AgentState.__init__(self, tracked_object_type=tracked_object_type, oriented_box=oriented_box, metadata=metadata, velocity=velocity, angular_velocity=angular_velocity)

    @classmethod
    def from_agent_state(cls, agent: AgentState) -> Agent:
        """
        Create Agent from AgentState.
        :param agent: input single agent state.
        :return: Agent with None for future and past trajectory.
        """
        return cls(tracked_object_type=agent.tracked_object_type, oriented_box=agent.box, velocity=agent.velocity, metadata=agent.metadata, angular_velocity=agent.angular_velocity, predictions=None, past_trajectory=None)

class Waypoint(InterpolatableState):
    """Represents a waypoint which is part of a trajectory. Optionals to allow for geometric trajectory"""

    def __init__(self, time_point: TimePoint, oriented_box: OrientedBox, velocity: Optional[StateVector2D]=None):
        """
        :param time_point: TimePoint corresponding to the Waypoint
        :param oriented_box: Position of the oriented box at the Waypoint
        :param velocity: Optional velocity information
        """
        self._time_point = time_point
        self._oriented_box = oriented_box
        self._velocity = velocity

    def __iter__(self) -> Iterable[Union[int, float]]:
        """
        Iterator for waypoint variables.
        :return: An iterator to the variables of the Waypoint.
        """
        return iter((self.time_us, self._oriented_box.center.x, self._oriented_box.center.y, self._oriented_box.center.heading, self._velocity.x if self._velocity is not None else None, self._velocity.y if self._velocity is not None else None))

    def __eq__(self, other: Any) -> bool:
        """
        Comparison between two Waypoints.
        :param other: Other object.
        :return True if both objects are same.
        """
        if not isinstance(other, Waypoint):
            return NotImplemented
        return other.oriented_box == self._oriented_box and other.time_point == self.time_point and (other.velocity == self._velocity)

    def __repr__(self) -> str:
        """
        :return: A string describing the object.
        """
        return self.__class__.__qualname__ + '(' + ', '.join([f'{f}={v}' for f, v in self.__dict__.items()]) + ')'

    @property
    def center(self) -> StateSE2:
        """
        Getter for center position of the waypoint
        :return: StateSE2 referring to position of the waypoint
        """
        return self._oriented_box.center

    @property
    def time_point(self) -> TimePoint:
        """
        Getter for time point corresponding to the waypoint
        :return: The time point
        """
        return self._time_point

    @property
    def oriented_box(self) -> OrientedBox:
        """
        Getter for the oriented box corresponding to the waypoint
        :return: The oriented box
        """
        return self._oriented_box

    @property
    def x(self) -> float:
        """
        Getter for the x position of the waypoint
        :return: The x position
        """
        return self._oriented_box.center.x

    @property
    def y(self) -> float:
        """
        Getter for the y position of the waypoint
        :return: The y position
        """
        return self._oriented_box.center.y

    @property
    def heading(self) -> float:
        """
        Getter for the heading of the waypoint
        :return: The heading
        """
        return self._oriented_box.center.heading

    @property
    def velocity(self) -> Optional[StateVector2D]:
        """
        Getter for the velocity corresponding to the waypoint
        :return: The velocity, None if not available
        """
        return self._velocity

    def serialize(self) -> List[Union[int, float]]:
        """
        Serializes the object as a list
        :return: Serialized object as a list
        """
        return [self.time_point.time_us, self._oriented_box.center.x, self._oriented_box.center.y, self._oriented_box.center.heading, self._oriented_box.length, self._oriented_box.width, self._oriented_box.height, self._velocity.x if self._velocity is not None else None, self._velocity.y if self._velocity is not None else None]

    @staticmethod
    def deserialize(vector: List[Union[int, float]]) -> Waypoint:
        """
        Deserializes the object.
        :param vector: a list of data to initialize a waypoint
        :return: Waypoint
        """
        assert len(vector) == 9, f'Expected a vector of size 9, got {len(vector)}'
        return Waypoint(time_point=TimePoint(int(vector[0])), oriented_box=OrientedBox(StateSE2(vector[1], vector[2], vector[3]), vector[4], vector[5], vector[6]), velocity=StateVector2D(vector[7], vector[8]) if vector[7] is not None and vector[8] is not None else None)

    def to_split_state(self) -> SplitState:
        """Inherited, see superclass."""
        linear_states = [self.time_point.time_us, self._oriented_box.center.x, self._oriented_box.center.y, self._velocity.x if self._velocity is not None else None, self._velocity.y if self._velocity is not None else None]
        angular_states = [self._oriented_box.center.heading]
        fixed_state = [self._oriented_box.width, self._oriented_box.length, self._oriented_box.height]
        return SplitState(linear_states, angular_states, fixed_state)

    @staticmethod
    def from_split_state(split_state: SplitState) -> Waypoint:
        """Inherited, see superclass."""
        total_state_length = len(split_state)
        assert total_state_length == 9, f'Expected a vector of size 9, got {total_state_length}'
        return Waypoint(time_point=TimePoint(int(split_state.linear_states[0])), oriented_box=OrientedBox(StateSE2(split_state.linear_states[1], split_state.linear_states[2], split_state.angular_states[0]), length=split_state.fixed_states[1], width=split_state.fixed_states[0], height=split_state.fixed_states[2]), velocity=StateVector2D(split_state.linear_states[3], split_state.linear_states[4]) if split_state.linear_states[3] is not None and split_state.linear_states[4] is not None else None)

class TestTimePoint(unittest.TestCase):
    """Tests for TimePoint class."""

    def test_initialization(self) -> None:
        """Tests initialization fails with negative values and works otherwise."""
        with self.assertRaises(AssertionError):
            _ = TimePoint(-42)
        t1 = TimePoint(123456)
        self.assertEqual(t1.time_us, 123456)

    def test_comparisons(self) -> None:
        """Test basic comparison operators."""
        t1 = TimePoint(123123)
        t2 = TimePoint(234234)
        self.assertTrue(t2 > t1)
        self.assertFalse(t2 < t1)
        self.assertTrue(t1 < t2)
        self.assertFalse(t1 > t2)
        self.assertTrue(t1 == t1)
        self.assertFalse(t1 == t2)
        self.assertTrue(t1 >= t1)
        self.assertTrue(t1 <= t1)

    def test_addition(self) -> None:
        """Tests addition and subtractions."""
        t1 = TimePoint(123)
        dt = TimeDuration.from_us(100)
        self.assertEqual(t1 + dt, TimePoint(223))
        self.assertEqual(dt + t1, TimePoint(223))
        self.assertEqual(t1 - dt, TimePoint(23))

class TestEgoState(unittest.TestCase):
    """Tests EgoState class"""

    def setUp(self) -> None:
        """Creates sample parameters for testing"""
        self.ego_state = get_sample_ego_state()
        self.vehicle = get_pacifica_parameters()
        self.dynamic_car_state = get_sample_dynamic_car_state(self.vehicle.rear_axle_to_center)

    def test_ego_state_extended_construction(self) -> None:
        """Tests that the ego state extended can be constructed from a pre-existing ego state."""
        ego_state_ext = EgoState.build_from_rear_axle(rear_axle_pose=self.ego_state.rear_axle, rear_axle_velocity_2d=self.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=self.dynamic_car_state.rear_axle_acceleration_2d, tire_steering_angle=self.ego_state.tire_steering_angle, time_point=self.ego_state.time_point, angular_vel=self.dynamic_car_state.angular_velocity, angular_accel=self.dynamic_car_state.angular_acceleration, is_in_auto_mode=True, vehicle_parameters=self.vehicle)
        self.assertTrue(ego_state_ext.dynamic_car_state == self.dynamic_car_state)
        self.assertTrue(ego_state_ext.center == self.ego_state.center)
        wp = ego_state_ext.waypoint
        self.assertEqual(wp.time_point, ego_state_ext.time_point)
        self.assertEqual(wp.oriented_box, ego_state_ext.car_footprint)
        self.assertEqual(wp.velocity, ego_state_ext.dynamic_car_state.rear_axle_velocity_2d)

    def test_to_split_state(self) -> None:
        """Tests that the state gets split as expected"""
        split_state = self.ego_state.to_split_state()
        self.assertEqual(len(split_state.linear_states), 8)
        self.assertEqual(split_state.fixed_states, [self.ego_state.car_footprint.vehicle_parameters])
        self.assertEqual(split_state.angular_states, [self.ego_state.rear_axle.heading])

    def test_from_split_state(self) -> None:
        """Tests that the object gets created as expected from the split state"""
        split_state = SplitState([0, 1, 2, 3, 4, 5, 6, 7], [8], [self.ego_state.car_footprint.vehicle_parameters])
        ego_from_split = EgoState.from_split_state(split_state)
        self.assertEqual(self.ego_state.car_footprint.vehicle_parameters, ego_from_split.car_footprint.vehicle_parameters)
        self.assertAlmostEqual(ego_from_split.time_us, 0)
        self.assertAlmostEqual(ego_from_split.rear_axle.x, 1)
        self.assertAlmostEqual(ego_from_split.rear_axle.y, 2)
        self.assertAlmostEqual(ego_from_split.rear_axle.heading, 8)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_velocity_2d.x, 3)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_velocity_2d.y, 4)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_acceleration_2d.x, 5)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_acceleration_2d.y, 6)
        self.assertAlmostEqual(ego_from_split.tire_steering_angle, 7)

class TestActorTemporalState(unittest.TestCase):
    """Test suite for the AgentTemporalState class"""

    def setUp(self) -> None:
        """Setup initial waypoints."""
        self.current_time_us = int(10 * 1000000.0)
        mock_oriented_box = Mock()
        self.future_waypoints: List[Optional[Waypoint]] = [Waypoint(time_point=TimePoint(self.current_time_us), oriented_box=mock_oriented_box), Waypoint(time_point=TimePoint(self.current_time_us + int(1000000.0)), oriented_box=mock_oriented_box)]
        self.past_waypoints: List[Optional[Waypoint]] = [Waypoint(time_point=TimePoint(self.current_time_us - int(1000000.0)), oriented_box=mock_oriented_box), Waypoint(time_point=TimePoint(self.current_time_us), oriented_box=mock_oriented_box)]

    def test_past_setting_successful(self) -> None:
        """Test that we can set past trajectory."""
        past_waypoints = [None] + self.past_waypoints
        actor = AgentTemporalState(initial_time_stamp=TimePoint(self.current_time_us), past_trajectory=PredictedTrajectory(waypoints=past_waypoints, probability=1.0))
        self.assertEqual(actor.past_trajectory.probability, 1.0)
        self.assertEqual(len(actor.past_trajectory.valid_waypoints), 2)
        self.assertEqual(len(actor.past_trajectory), 3)
        self.assertEqual(actor.previous_state, self.past_waypoints[0])

    def test_past_setting_fail(self) -> None:
        """Test that we can raise if past trajectory does not start at current state."""
        past_waypoints = list(reversed(self.past_waypoints))
        with self.assertRaises(ValueError):
            AgentTemporalState(initial_time_stamp=TimePoint(self.current_time_us), past_trajectory=PredictedTrajectory(waypoints=past_waypoints, probability=1.0))

    def test_future_trajectory_successful(self) -> None:
        """Test that we can set future predictions."""
        future_waypoints = self.future_waypoints
        actor = AgentTemporalState(initial_time_stamp=TimePoint(self.current_time_us), predictions=[PredictedTrajectory(waypoints=future_waypoints, probability=1.0)])
        self.assertEqual(len(actor.predictions), 1)
        self.assertEqual(actor.predictions[0].probability, 1.0)

    def test_trajectory_successful_none(self) -> None:
        """Test that we can set future predictions with None."""
        actor = AgentTemporalState(initial_time_stamp=TimePoint(self.current_time_us), predictions=None, past_trajectory=None)
        self.assertEqual(len(actor.predictions), 0)
        self.assertEqual(actor.past_trajectory, None)

    def test_future_trajectory_fail(self) -> None:
        """Test that we can set future predictions, but it will fail if all conditions are not met."""
        future_waypoints = self.future_waypoints
        with self.assertRaises(ValueError):
            AgentTemporalState(initial_time_stamp=TimePoint(self.current_time_us), predictions=[PredictedTrajectory(waypoints=future_waypoints, probability=0.4)])

class TestSceneSimpleTrajectory(unittest.TestCase):
    """
    Tests the class SceneSimpleTrajectory
    """

    def setUp(self) -> None:
        """
        Sets up for the test cases
        """
        state1: Dict[str, Any] = {'timestamp': 1, 'pose': [1, 2, 3]}
        state2: Dict[str, Any] = {'timestamp': 2, 'pose': [3, 4, 5]}
        prediction_states: List[Dict[str, Any]] = [state1, state2]
        self.width = 3
        self.length = 6
        self.height = 2
        self.scene_simple_trajectory = SceneSimpleTrajectory(prediction_states, width=self.width, length=self.length, height=self.height)

    def test_init(self) -> None:
        """
        Tests the init of SceneSiimpleTrajectory
        """
        state1: Dict[str, Any] = {'timestamp': 1, 'pose': [1, 2, 3]}
        state2: Dict[str, Any] = {'timestamp': 2, 'pose': [3, 4, 5]}
        prediction_states: List[Dict[str, Any]] = [state1, state2]
        result = SceneSimpleTrajectory(prediction_states, width=self.width, length=self.length, height=self.height)
        self.assertEqual(result._start_time, 1)
        self.assertEqual(result._end_time, 2)

    def test_start_time(self) -> None:
        """
        Tests the start time property
        """
        scene_simple_trajectory = self.scene_simple_trajectory
        result = scene_simple_trajectory.start_time
        self.assertEqual(result, 1)

    def test_end_time(self) -> None:
        """
        Tests the start time property
        """
        scene_simple_trajectory = self.scene_simple_trajectory
        result = scene_simple_trajectory.end_time
        self.assertEqual(result, 2)

    def test_get_state_at_time(self) -> None:
        """
        Tests the get state at time method
        """
        scene_simple_trajectory = self.scene_simple_trajectory
        result = scene_simple_trajectory.get_state_at_time(TimePoint(int(1000000.0)))
        self.assertEqual(result.x, 1)
        self.assertEqual(result.y, 2)

    def test_get_sampled_trajectory(self) -> None:
        """
        Tests the get sampled method
        """
        scene_simple_trajectory = self.scene_simple_trajectory
        result = scene_simple_trajectory.get_sampled_trajectory()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].x, 1)
        self.assertEqual(result[1].x, 3)

class Simulation:
    """
    This class queries data for initialization of a planner, and propagates simulation a step forward based on the
        planned trajectory of a planner.
    """

    def __init__(self, simulation_setup: SimulationSetup, callback: Optional[AbstractCallback]=None, simulation_history_buffer_duration: float=2):
        """
        Create Simulation.
        :param simulation_setup: Configuration that describes the simulation.
        :param callback: A callback to be executed for this simulation setup
        :param simulation_history_buffer_duration: [s] Duration to pre-load scenario into the buffer.
        """
        if simulation_history_buffer_duration < simulation_setup.scenario.database_interval:
            raise ValueError(f'simulation_history_buffer_duration {simulation_history_buffer_duration} has to be larger than the scenario database_interval {simulation_setup.scenario.database_interval}')
        self._setup = simulation_setup
        self._time_controller = simulation_setup.time_controller
        self._ego_controller = simulation_setup.ego_controller
        self._observations = simulation_setup.observations
        self._scenario = simulation_setup.scenario
        self._callback = MultiCallback([]) if callback is None else callback
        self._history = SimulationHistory(self._scenario.map_api, self._scenario.get_mission_goal())
        self._simulation_history_buffer_duration = simulation_history_buffer_duration + self._scenario.database_interval
        self._history_buffer_size = int(self._simulation_history_buffer_duration / self._scenario.database_interval) + 1
        self._history_buffer: Optional[SimulationHistoryBuffer] = None
        self._is_simulation_running = True

    def __reduce__(self) -> Tuple[Type[Simulation], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._setup, self._callback, self._simulation_history_buffer_duration))

    def is_simulation_running(self) -> bool:
        """
        Check whether a simulation reached the end
        :return True if simulation hasn't reached the end, otherwise false.
        """
        return not self._time_controller.reached_end() and self._is_simulation_running

    def reset(self) -> None:
        """
        Reset all internal states of simulation.
        """
        self._history.reset()
        self._setup.reset()
        self._history_buffer = None
        self._is_simulation_running = True

    def initialize(self) -> PlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
        :return data needed for planner initialization.
        """
        self.reset()
        self._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(self._history_buffer_size, self._scenario, self._observations.observation_type())
        self._observations.initialize()
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())
        return PlannerInitialization(route_roadblock_ids=self._scenario.get_route_roadblock_ids(), mission_goal=self._scenario.get_mission_goal(), map_api=self._scenario.map_api)

    def get_planner_input(self) -> PlannerInput:
        """
        Construct inputs to the planner for the current iteration step
        :return Inputs to the planner.
        """
        if self._history_buffer is None:
            raise RuntimeError('Simulation was not initialized!')
        if not self.is_simulation_running():
            raise RuntimeError('Simulation is not running, stepping can not be performed!')
        iteration = self._time_controller.get_iteration()
        traffic_light_data = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))
        logger.debug(f'Executing {iteration.index}!')
        return PlannerInput(iteration=iteration, history=self._history_buffer, traffic_light_data=traffic_light_data)

    def propagate(self, trajectory: AbstractTrajectory) -> None:
        """
        Propagate the simulation based on planner's trajectory and the inputs to the planner
        This function also decides whether simulation should still continue. This flag can be queried through
        reached_end() function
        :param trajectory: computed trajectory from planner.
        """
        if self._history_buffer is None:
            raise RuntimeError('Simulation was not initialized!')
        if not self.is_simulation_running():
            raise RuntimeError('Simulation is not running, simulation can not be propagated!')
        iteration = self._time_controller.get_iteration()
        ego_state, observation = self._history_buffer.current_state
        traffic_light_status = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))
        logger.debug(f'Adding to history: {iteration.index}')
        self._history.add_sample(SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status))
        next_iteration = self._time_controller.next_iteration()
        if next_iteration:
            self._ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
            self._observations.update_observation(iteration, next_iteration, self._history_buffer)
        else:
            self._is_simulation_running = False
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: used scenario in this simulation.
        """
        return self._scenario

    @property
    def setup(self) -> SimulationSetup:
        """
        :return: Setup for this simulation.
        """
        return self._setup

    @property
    def callback(self) -> AbstractCallback:
        """
        :return: Callback for this simulation.
        """
        return self._callback

    @property
    def history(self) -> SimulationHistory:
        """
        :return History from the simulation.
        """
        return self._history

    @property
    def history_buffer(self) -> SimulationHistoryBuffer:
        """
        :return SimulationHistoryBuffer from the simulation.
        """
        if self._history_buffer is None:
            raise RuntimeError('_history_buffer is None. Please initialize the buffer by calling Simulation.initialize()')
        return self._history_buffer

class TestAbstractPredictor(unittest.TestCase):
    """Test the AbstractPredictor interface"""

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self.predictor = MockAbstractPredictor()

    def test_initialize(self) -> None:
        """Test initialization"""
        mock_initialization = get_mock_predictor_initialization()
        self.predictor.initialize(mock_initialization)
        self.assertEqual(self.predictor._map_api, mock_initialization.map_api)

    def test_name(self) -> None:
        """Test name"""
        self.assertEqual(self.predictor.name(), 'MockAbstractPredictor')

    def test_observation_type(self) -> None:
        """Test observation_type"""
        self.assertEqual(self.predictor.observation_type(), DetectionsTracks)

    def test_compute_predictions(self) -> None:
        """Test compute_predictions"""
        predictor_input = get_mock_predictor_input()
        start_time = time.perf_counter()
        detections = self.predictor.compute_predictions(predictor_input)
        compute_predictions_time = time.perf_counter() - start_time
        self.assertEqual(type(detections), DetectionsTracks)
        predictor_report = self.predictor.generate_predictor_report()
        self.assertEqual(len(predictor_report.compute_predictions_runtimes), 1)
        self.assertNotIsInstance(predictor_report, MLPredictorReport)
        self.assertAlmostEqual(predictor_report.compute_predictions_runtimes[0], compute_predictions_time, delta=0.1)

def get_mock_predictor_initialization() -> PredictorInitialization:
    """
    Returns a mock PredictorInitialization for testing.
    :return: PredictorInitialization.
    """
    return PredictorInitialization(MockAbstractMap())

def get_mock_predictor_input(buffer_size: int=1) -> PredictorInput:
    """
    Returns a mock PredictorInput for testing.
    :return: PredictorInput.
    """
    scenario = MockAbstractScenario()
    history_buffer = SimulationHistoryBuffer.initialize_from_list(buffer_size, [scenario.initial_ego_state for _ in range(buffer_size)], [scenario.initial_tracked_objects for _ in range(buffer_size)], 0.5)
    return PredictorInput(iteration=SimulationIteration(TimePoint(0), 0), history=history_buffer, traffic_light_data=None)

class TestLogFuturePredictor(unittest.TestCase):
    """
    Test LogFuturePredictor class.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.scenario = MockAbstractScenario()
        self.future_trajectory_sampling = TrajectorySampling(num_poses=1, time_horizon=1.0)
        self.predictor = LogFuturePredictor(self.scenario, self.future_trajectory_sampling)

    def test_compute_predicted_trajectories(self) -> None:
        """Test compute_predicted_trajectories."""
        predictor_input = get_mock_predictor_input()
        start_time = time.perf_counter()
        detections = self.predictor.compute_predictions(predictor_input)
        compute_predictions_time = time.perf_counter() - start_time
        _, input_detections = predictor_input.history.current_state
        self.assertEqual(len(detections.tracked_objects), len(input_detections.tracked_objects))
        for agent in detections.tracked_objects.get_agents():
            self.assertTrue(agent.predictions is not None)
            for prediction in agent.predictions:
                self.assertEqual(len(prediction.valid_waypoints), self.future_trajectory_sampling.num_poses)
        predictor_report = self.predictor.generate_predictor_report()
        self.assertEqual(len(predictor_report.compute_predictions_runtimes), 1)
        self.assertNotIsInstance(predictor_report, MLPredictorReport)
        self.assertAlmostEqual(predictor_report.compute_predictions_runtimes[0], compute_predictions_time, delta=0.1)

class ILQRTracker(AbstractTracker):
    """
    Tracker using an iLQR solver with a kinematic bicycle model.
    """

    def __init__(self, n_horizon: int, ilqr_solver: ILQRSolver) -> None:
        """
        Initialize tracker parameters, primarily the iLQR solver.
        :param n_horizon: Maximum time horizon (number of discrete time steps) that we should plan ahead.
                          Please note the associated discretization_time is specified in the ilqr_solver.
        :param ilqr_solver: Solver used to compute inputs to apply.
        """
        assert n_horizon > 0, 'The time horizon length should be positive.'
        self._n_horizon = n_horizon
        self._ilqr_solver = ilqr_solver

    def track_trajectory(self, current_iteration: SimulationIteration, next_iteration: SimulationIteration, initial_state: EgoState, trajectory: AbstractTrajectory) -> DynamicCarState:
        """Inherited, see superclass."""
        current_state: DoubleMatrix = np.array([initial_state.rear_axle.x, initial_state.rear_axle.y, initial_state.rear_axle.heading, initial_state.dynamic_car_state.rear_axle_velocity_2d.x, initial_state.tire_steering_angle])
        reference_trajectory = self._get_reference_trajectory(current_iteration, trajectory)
        solutions = self._ilqr_solver.solve(current_state, reference_trajectory)
        optimal_inputs = solutions[-1].input_trajectory
        accel_cmd = optimal_inputs[0, 0]
        steering_rate_cmd = optimal_inputs[0, 1]
        return DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist, rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=StateVector2D(accel_cmd, 0), tire_steering_rate=steering_rate_cmd)

    def _get_reference_trajectory(self, current_iteration: SimulationIteration, trajectory: AbstractTrajectory) -> DoubleMatrix:
        """
        Determines reference trajectory, (z_{ref,k})_k=0^self._n_horizon.
        In case the query timestep exceeds the trajectory length, we return a smaller trajectory (z_{ref,k})_k=0^M,
        where M < self._n_horizon.  The shorter reference will then be handled downstream by the solver appropriately.
        :param current_iteration: Provides the current time from which we interpolate.
        :param trajectory: The full planned trajectory from which we perform state interpolation.
        :return a (M+1 or self._n_horizon+1) by self._n_states array.
        """
        assert trajectory.start_time.time_s <= current_iteration.time_s, 'Current time is before trajectory start.'
        assert current_iteration.time_s <= trajectory.end_time.time_s, 'Current time is after trajectory end'
        discretization_time = self._ilqr_solver._solver_params.discretization_time
        time_deltas_s: DoubleMatrix = np.array([x * discretization_time for x in range(0, self._n_horizon + 1)], dtype=np.float64)
        states_interp = []
        for tm_delta_s in time_deltas_s:
            timepoint = TimePoint(int(tm_delta_s * 1000000.0)) + current_iteration.time_point
            if timepoint > trajectory.end_time:
                break
            state = trajectory.get_state_at_time(timepoint)
            states_interp.append([state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading, state.dynamic_car_state.rear_axle_velocity_2d.x, state.tire_steering_angle])
        return np.array(states_interp)

class TestILQRTracker(unittest.TestCase):
    """
    Tests the functionality of the ILQRTracker class.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.initial_time_point = TimePoint(1000000)
        self.scenario = MockAbstractScenario(initial_time_us=self.initial_time_point)
        self.trajectory = InterpolatedTrajectory(list(self.scenario.get_expert_ego_trajectory()))
        solver_params = ILQRSolverParameters(discretization_time=0.2, state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0], input_cost_diagonal_entries=[1.0, 10.0], state_trust_region_entries=[1.0] * 5, input_trust_region_entries=[1.0] * 2, max_ilqr_iterations=100, convergence_threshold=1e-06, max_solve_time=0.05, max_acceleration=3.0, max_steering_angle=np.pi / 3.0, max_steering_angle_rate=0.5, min_velocity_linearization=0.01)
        warm_start_params = ILQRWarmStartParameters(k_velocity_error_feedback=0.5, k_steering_angle_error_feedback=0.05, lookahead_distance_lateral_error=15.0, k_lateral_error=0.1, jerk_penalty_warm_start_fit=0.0001, curvature_rate_penalty_warm_start_fit=0.01)
        self.tracker = ILQRTracker(n_horizon=40, ilqr_solver=ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params))
        self.discretization_time_us = int(1000000.0 * self.tracker._ilqr_solver._solver_params.discretization_time)

    def test_track_trajectory(self) -> None:
        """Ensure that we can run a single solver call to track a trajectory."""
        current_iteration = SimulationIteration(time_point=self.initial_time_point, index=0)
        time_point_delta = TimePoint(self.discretization_time_us)
        next_iteration = SimulationIteration(time_point=self.initial_time_point + time_point_delta, index=1)
        self.tracker.track_trajectory(current_iteration=current_iteration, next_iteration=next_iteration, initial_state=self.scenario.initial_ego_state, trajectory=self.trajectory)

    def test__get_reference_trajectory(self) -> None:
        """Test reference trajectory extraction for the solver."""
        current_iteration_before_trajectory_start = SimulationIteration(time_point=self.trajectory.start_time - TimePoint(1), index=0)
        with self.assertRaises(AssertionError):
            self.tracker._get_reference_trajectory(current_iteration_before_trajectory_start, self.trajectory)
        current_iteration_after_trajectory_end = SimulationIteration(time_point=self.trajectory.end_time + TimePoint(1), index=0)
        with self.assertRaises(AssertionError):
            self.tracker._get_reference_trajectory(current_iteration_after_trajectory_end, self.trajectory)
        start_time_us = self.trajectory.start_time.time_us
        end_time_us = self.trajectory.end_time.time_us
        mid_time_us = int((start_time_us + end_time_us) / 2)
        for test_time_us in [start_time_us, mid_time_us, end_time_us]:
            expected_trajectory_length = min((end_time_us - test_time_us) // self.discretization_time_us + 1, self.tracker._n_horizon + 1)
            current_iteration = SimulationIteration(time_point=TimePoint(test_time_us), index=0)
            reference_trajectory = self.tracker._get_reference_trajectory(current_iteration, self.trajectory)
            self.assertEqual(len(reference_trajectory), expected_trajectory_length)
            first_state_reference_trajectory = reference_trajectory[0]
            first_ego_state_expected = self.trajectory.get_state_at_time(current_iteration.time_point)
            np_test.assert_allclose(first_state_reference_trajectory[0], first_ego_state_expected.rear_axle.x)
            np_test.assert_allclose(first_state_reference_trajectory[1], first_ego_state_expected.rear_axle.y)
            np_test.assert_allclose(first_state_reference_trajectory[2], first_ego_state_expected.rear_axle.heading)
            np_test.assert_allclose(first_state_reference_trajectory[3], first_ego_state_expected.dynamic_car_state.rear_axle_velocity_2d.x)
            np_test.assert_allclose(first_state_reference_trajectory[4], first_ego_state_expected.tire_steering_angle)

class TestLQRTracker(unittest.TestCase):
    """
    Tests LQR Tracker.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.initial_time_point = TimePoint(0)
        self.scenario = MockAbstractScenario(initial_time_us=self.initial_time_point)
        self.trajectory = InterpolatedTrajectory(list(self.scenario.get_expert_ego_trajectory()))
        self.sampling_time = 0.5
        self.tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, jerk_penalty=0.0001, curvature_rate_penalty=0.01, stopping_proportional_gain=0.5, stopping_velocity=0.2)

    def test_track_trajectory(self) -> None:
        """Ensure we are able to run track trajectory using LQR."""
        dynamic_state = self.tracker.track_trajectory(current_iteration=SimulationIteration(self.initial_time_point, 0), next_iteration=SimulationIteration(TimePoint(int(self.sampling_time * 1000000.0)), 1), initial_state=self.scenario.initial_ego_state, trajectory=self.trajectory)
        self.assertIsInstance(dynamic_state._rear_axle_to_center_dist, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_velocity_2d.x, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_velocity_2d.y, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_acceleration_2d.x, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_acceleration_2d.y, (int, float))
        self.assertIsInstance(dynamic_state.tire_steering_rate, (int, float))
        self.assertGreater(dynamic_state._rear_axle_to_center_dist, 0.0)
        self.assertEqual(dynamic_state.rear_axle_acceleration_2d.y, 0.0)

    def test__compute_initial_velocity_and_lateral_state(self) -> None:
        """
        This essentially checks that our projection to vehicle/Frenet frame works by reconstructing specified errors.
        """
        current_iteration = SimulationIteration(self.initial_time_point, 0)
        base_initial_state = self.trajectory.get_state_at_time(self.initial_time_point)
        base_pose_rear_axle = base_initial_state.car_footprint.rear_axle
        test_lateral_errors = [-3.0, 3.0]
        test_heading_errors = [-0.1, 0.1]
        test_longitudinal_errors = [-3.0, 3.0]
        error_product = itertools.product(test_lateral_errors, test_heading_errors, test_longitudinal_errors)
        for lateral_error, heading_error, longitudinal_error in error_product:
            theta = base_pose_rear_axle.heading
            delta_x = longitudinal_error * np.cos(theta) - lateral_error * np.sin(theta)
            delta_y = longitudinal_error * np.sin(theta) + lateral_error * np.cos(theta)
            perturbed_pose_rear_axle = StateSE2(x=base_pose_rear_axle.x + delta_x, y=base_pose_rear_axle.y + delta_y, heading=theta + heading_error)
            perturbed_car_footprint = CarFootprint.build_from_rear_axle(rear_axle_pose=perturbed_pose_rear_axle, vehicle_parameters=base_initial_state.car_footprint.vehicle_parameters)
            perturbed_initial_state = EgoState(car_footprint=perturbed_car_footprint, dynamic_car_state=base_initial_state.dynamic_car_state, tire_steering_angle=base_initial_state.tire_steering_angle, is_in_auto_mode=base_initial_state.is_in_auto_mode, time_point=base_initial_state.time_point)
            initial_velocity, initial_lateral_state_vector = self.tracker._compute_initial_velocity_and_lateral_state(current_iteration=current_iteration, initial_state=perturbed_initial_state, trajectory=self.trajectory)
            self.assertEqual(initial_velocity, base_initial_state.dynamic_car_state.rear_axle_velocity_2d.x)
            np_test.assert_allclose(initial_lateral_state_vector, [lateral_error, heading_error, base_initial_state.tire_steering_angle])

    def test__compute_reference_velocity_and_curvature_profile(self) -> None:
        """
        This test just checks functionality of computing a reference velocity / curvature profile.
        Detailed evaluation of the result is handled in test_tracker_utils and omitted here.
        """
        current_iteration = SimulationIteration(self.initial_time_point, 0)
        reference_velocity, curvature_profile = self.tracker._compute_reference_velocity_and_curvature_profile(current_iteration=current_iteration, trajectory=self.trajectory)
        tracking_horizon = self.tracker._tracking_horizon
        discretization_time = self.tracker._discretization_time
        lookahead_time_point = TimePoint(current_iteration.time_point.time_us + int(1000000.0 * tracking_horizon * discretization_time))
        expected_lookahead_ego_state = self.trajectory.get_state_at_time(lookahead_time_point)
        np_test.assert_allclose(np.sign(reference_velocity), np.sign(expected_lookahead_ego_state.dynamic_car_state.rear_axle_velocity_2d.x))
        self.assertEqual(curvature_profile.shape, (tracking_horizon,))

    def test__stopping_controller(self) -> None:
        """Test P controller for when we are coming to a stop."""
        initial_velocity = 5.0
        accel, steering_rate_cmd = self.tracker._stopping_controller(initial_velocity=initial_velocity, reference_velocity=0.5 * initial_velocity)
        self.assertLess(accel, 0.0)
        self.assertEqual(steering_rate_cmd, 0.0)
        accel, steering_rate_cmd = self.tracker._stopping_controller(initial_velocity=-initial_velocity, reference_velocity=0.0)
        self.assertGreater(accel, 0.0)
        self.assertEqual(steering_rate_cmd, 0.0)

    def test__longitudinal_lqr_controller(self) -> None:
        """Test longitudinal control for simple cases of speed above or below the reference velocity."""
        test_initial_velocities = [2.0, 6.0]
        reference_velocity = float(np.mean(test_initial_velocities))
        for initial_velocity in test_initial_velocities:
            accel_cmd = self.tracker._longitudinal_lqr_controller(initial_velocity=initial_velocity, reference_velocity=reference_velocity)
            np_test.assert_allclose(np.sign(accel_cmd), -np.sign(initial_velocity - reference_velocity))

    def test__lateral_lqr_controller_straight_road(self) -> None:
        """Test how the controller handles non-zero initial tracking error on a straight road."""
        test_velocity_profile = 5.0 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)
        test_curvature_profile = 0.0 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)
        test_lateral_errors = [-3.0, 3.0]
        for lateral_error in test_lateral_errors:
            initial_lateral_state_vector_lateral_only: npt.NDArray[np.float64] = np.array([lateral_error, 0.0, 0.0], dtype=np.float64)
            steering_rate_cmd = self.tracker._lateral_lqr_controller(initial_lateral_state_vector=initial_lateral_state_vector_lateral_only, velocity_profile=test_velocity_profile, curvature_profile=test_curvature_profile)
            np_test.assert_allclose(np.sign(steering_rate_cmd), -np.sign(lateral_error))
        test_heading_errors = [-0.1, 0.1]
        for heading_error in test_heading_errors:
            initial_lateral_state_vector_heading_only: npt.NDArray[np.float64] = np.array([0.0, heading_error, 0.0], dtype=np.float64)
            steering_rate_cmd = self.tracker._lateral_lqr_controller(initial_lateral_state_vector=initial_lateral_state_vector_heading_only, velocity_profile=test_velocity_profile, curvature_profile=test_curvature_profile)
            np_test.assert_allclose(np.sign(steering_rate_cmd), -np.sign(heading_error))

    def test__lateral_lqr_controller_curved_road(self) -> None:
        """Test how the controller handles a curved road with zero initial tracking error and zero steering angle."""
        test_velocity_profile = 5.0 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)
        test_curvature_profile = 0.1 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)
        test_initial_lateral_state_vector: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        steering_rate_cmd = self.tracker._lateral_lqr_controller(initial_lateral_state_vector=test_initial_lateral_state_vector, velocity_profile=test_velocity_profile, curvature_profile=test_curvature_profile)
        np_test.assert_allclose(np.sign(steering_rate_cmd), np.sign(test_curvature_profile[0]))

    def test__solve_one_step_lqr(self) -> None:
        """Test LQR on a simple linear system."""
        A: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        B: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        g: npt.NDArray[np.float64] = np.zeros(A.shape[0], dtype=np.float64)
        Q: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        R: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        for component_1, component_2 in itertools.product([-5.0, 5.0], [-10.0, 10.0]):
            initial_state: npt.NDArray[np.float64] = np.array([component_1, component_2], dtype=np.float64)
            solution = self.tracker._solve_one_step_lqr(initial_state=initial_state, reference_state=np.zeros_like(initial_state), Q=Q, R=R, A=A, B=B, g=g, angle_diff_indices=[])
            np_test.assert_allclose(np.sign(solution), -np.sign(initial_state))

class TestAbstractIDMPlanner(unittest.TestCase):
    """Test the AbstractIDMPlanner interface"""
    TEST_FILE_PATH = 'nuplan.planning.simulation.planner.idm_planner'

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self.scenario = get_test_nuplan_scenario()
        self.planned_trajectory_samples = 10
        self.planner = IDMPlanner(target_velocity=10, min_gap_to_lead_agent=0.5, headway_time=1.5, accel_max=1.0, decel_max=2.0, planned_trajectory_samples=self.planned_trajectory_samples, planned_trajectory_sample_interval=0.2, occupancy_map_radius=20)

    def test_name(self) -> None:
        """Test name"""
        self.assertEqual(self.planner.name(), 'IDMPlanner')

    def test_observation_type(self) -> None:
        """Test observation_type"""
        self.assertEqual(self.planner.observation_type(), DetectionsTracks)

    def test__initialize_route_plan_assertion_error(self) -> None:
        """Test raise if _map_api is uninitialized"""
        with self.assertRaises(AssertionError):
            self.planner._initialize_route_plan([])

    @patch(f'{TEST_FILE_PATH}.IDMPlanner._initialize_route_plan')
    def test_initialize(self, mock_initialize_route_plan: Mock) -> None:
        """Test initialize"""
        initialization = MagicMock()
        self.planner.initialize(initialization)
        mock_initialize_route_plan.assert_called_once_with(initialization.route_roadblock_ids)

    @patch(f'{TEST_FILE_PATH}.path_to_linestring')
    @patch(f'{TEST_FILE_PATH}.create_path_from_se2')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._breadth_first_search')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._get_starting_edge')
    def test__initialize_ego_path(self, mock_get_starting_edge: Mock, mock_breadth_first_search: Mock, mock_create_path_from_se2: Mock, mock_path_to_linestring: Mock) -> None:
        """Test _initialize_ego_path()"""
        mock_starting_edge = Mock()
        mock_lane = MagicMock()
        mock_lane.speed_limit_mps = 0
        ego_state = self.scenario.initial_ego_state
        mock_breadth_first_search.return_value = ([mock_lane], True)
        mock_get_starting_edge.return_value = mock_starting_edge
        with patch.object(self.planner, '_route_roadblocks'):
            self.planner._initialize_ego_path(ego_state)
            mock_breadth_first_search.assert_called_once_with(ego_state)
            mock_create_path_from_se2.assert_called_once_with([])
            mock_path_to_linestring.assert_called_once_with([])

    def test__get_starting_edge(self) -> None:
        """Test _get_starting_edge()"""
        mock_edge = MagicMock(spec_set=LaneGraphEdgeMapObject)
        mock_edge.contains_point.side_effect = [False, True]
        mock_edge.polygon.distance.side_effect = [0, 0]
        mock_roadblock = MagicMock(spec_set=RoadBlockGraphEdgeMapObject)
        mock_roadblock.interior_edges = [mock_edge]
        self.planner._route_roadblocks = [mock_roadblock, mock_roadblock]
        result = self.planner._get_starting_edge(Mock(spec=EgoState))
        mock_edge.contains_point.assert_called()
        mock_edge.polygon.distance.assert_called()
        self.assertEqual(result, mock_edge)

    @patch(f'{TEST_FILE_PATH}.IDMPlanner._initialize_ego_path')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._construct_occupancy_map')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._annotate_occupancy_map')
    @patch(f'{TEST_FILE_PATH}.IDMPlanner._get_planned_trajectory')
    def test_compute_trajectory(self, mock_get_planned_trajectory: Mock, mock_annotate_occupancy_map: Mock, mock_construct_occupancy_map: Mock, mock_initialize_ego_path: Mock) -> None:
        """Test compute_trajectory"""
        planner_input = MagicMock()
        mock_ego_state = Mock()
        mock_traffic_light_data = call()
        planner_input.history.current_state = (mock_ego_state, Mock())
        planner_input.traffic_light_data = mock_traffic_light_data
        mock_occupancy_map = Mock()
        mock_unique_observations = Mock()
        mock_construct_occupancy_map.return_value = (mock_occupancy_map, mock_unique_observations)
        self.planner.compute_trajectory(planner_input)
        mock_initialize_ego_path.assert_called_once_with(mock_ego_state)
        mock_construct_occupancy_map.assert_called_once_with(*planner_input.history.current_state)
        mock_annotate_occupancy_map.assert_called_once_with(mock_traffic_light_data, mock_occupancy_map)
        mock_get_planned_trajectory.assert_called_once_with(mock_ego_state, mock_occupancy_map, mock_unique_observations)

    def test_compute_trajectory_integration(self) -> None:
        """Test the IDMPlanner in full using mock data"""
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(10, self.scenario, DetectionsTracks)
        self.planner.initialize(PlannerInitialization(self.scenario.get_route_roadblock_ids(), self.scenario.get_mission_goal(), self.scenario.map_api))
        trajectories = self.planner.compute_trajectory(PlannerInput(SimulationIteration(self.scenario.get_time_point(0), 0), history_buffer, list(self.scenario.get_traffic_light_status_at_iteration(0))))
        self.assertEqual(self.planned_trajectory_samples + 1, len(trajectories.get_sampled_trajectory()))

class TestLogFuturePlanner(unittest.TestCase):
    """
    Test LogFuturePlanner class
    """

    def _get_mock_planner_input(self) -> PlannerInput:
        """
        Returns a mock PlannerInput for testing.
        :return: PlannerInput.
        """
        buffer = SimulationHistoryBuffer.initialize_from_list(1, [self.scenario.initial_ego_state], [self.scenario.initial_tracked_objects])
        return PlannerInput(iteration=SimulationIteration(TimePoint(0), 0), history=buffer, traffic_light_data=None)

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.scenario = MockAbstractScenario(number_of_future_iterations=20)
        self.num_poses = 10
        self.future_time_horizon = 5
        self.planner = LogFuturePlanner(self.scenario, self.num_poses, self.future_time_horizon)

    def test_name(self) -> None:
        """Tests planner name is set correctly."""
        result = self.planner.name()
        self.assertEqual(result, 'LogFuturePlanner')

    @patch('nuplan.planning.simulation.planner.log_future_planner.DetectionsTracks')
    def test_observation_type(self, mock_detection_tracks: Mock) -> None:
        """Tests observation type is set correctly."""
        result = self.planner.observation_type()
        self.assertEqual(result, mock_detection_tracks)

    def test_compute_trajectory(self) -> None:
        """Test compute_trajectory"""
        planner_input = self._get_mock_planner_input()
        start_time = time.perf_counter()
        result = self.planner.compute_trajectory(planner_input)
        compute_trajectory_time = time.perf_counter() - start_time
        self.assertEqual(len(result.get_sampled_trajectory()), self.num_poses + 1)
        planner_report = self.planner.generate_planner_report()
        self.assertEqual(len(planner_report.compute_trajectory_runtimes), 1)
        self.assertNotIsInstance(planner_report, MLPlannerReport)
        self.assertAlmostEqual(planner_report.compute_trajectory_runtimes[0], compute_trajectory_time, delta=0.1)

    def test_compute_trajectory_fail_extraction_previous_available(self) -> None:
        """
        Test compute_trajectory when future ego extraction from scenario fails and planner should fall back on previous
        trajectory.
        """
        previous_trajectory = Mock()
        self.planner._trajectory = previous_trajectory
        planner_input = self._get_mock_planner_input()
        with patch.object(self.scenario, 'get_ego_future_trajectory', side_effect=AssertionError):
            result = self.planner.compute_trajectory(planner_input)
        self.assertEqual(result, previous_trajectory)

    def test_compute_trajectory_fail_extraction_no_previous(self) -> None:
        """
        Test compute_trajectory when future ego extraction from scenario fails and there is no prior trajectory
        to fall back on.
        """
        self.planner._trajectory = None
        planner_input = self._get_mock_planner_input()
        with patch.object(self.scenario, 'get_ego_future_trajectory', side_effect=AssertionError):
            with self.assertRaises(RuntimeError):
                _ = self.planner.compute_trajectory(planner_input)

class TestRemotePlanner(TestCase):
    """Tests RemotePlanner class"""

    @patch('nuplan.planning.simulation.planner.remote_planner.SubmissionContainerManager', autospec=True)
    def setUp(self, mock_factory: Mock) -> None:
        """Sets variables for testing"""
        self.planner = RemotePlanner()
        self.planner_with_container = RemotePlanner(submission_container_manager=Mock(), submission_image='foo', container_name='bar')

    @patch('nuplan.planning.simulation.planner.remote_planner.SubmissionContainerManager', autospec=True)
    def test_initialization(self, mock_factory: Mock) -> None:
        """Tests that the class is initialized as intended."""
        mock_planner = RemotePlanner()
        self.assertEqual(None, mock_planner.submission_container_manager)
        self.assertEqual(50051, mock_planner.port)
        mock_planner = RemotePlanner(submission_container_manager=mock_factory, submission_image='foo', container_name='bar')
        self.assertEqual(mock_factory, mock_planner.submission_container_manager)
        self.assertEqual('foo', mock_planner.submission_image)
        self.assertEqual('bar', mock_planner.container_name)
        self.assertEqual(None, mock_planner.port)
        with self.assertRaises(AssertionError):
            _ = RemotePlanner(submission_container_manager=Mock())

    def test_name(self) -> None:
        """Tests planner name is set correctly"""
        self.assertEqual('RemotePlanner', self.planner.name())

    def test_observation_type(self) -> None:
        """Tests observation type is set correctly"""
        self.assertEqual(DetectionsTracks, self.planner.observation_type())

    def test_initialization_message_creation(self) -> None:
        """Tests that the message for the initialization request is built correctly."""
        mock_state_1 = Mock(x=0, y=1, heading=0.2)
        mock_map_api = Mock(map_name='test')
        mock_initialization = Mock(mission_goal=mock_state_1, map_api=mock_map_api, route_roadblock_ids=['a', 'b', 'c'])
        with self.assertRaises(AttributeError):
            self.planner._planner_initializations_to_message(Mock(mission_goal=None, map_api=mock_map_api))
        initialization_message = self.planner._planner_initializations_to_message(mock_initialization)
        self.assertAlmostEqual(mock_state_1.x, initialization_message.mission_goal.x)
        self.assertAlmostEqual(mock_state_1.y, initialization_message.mission_goal.y)
        self.assertAlmostEqual(mock_state_1.heading, initialization_message.mission_goal.heading)
        self.assertEqual(mock_map_api.map_name, initialization_message.map_name)
        self.assertEqual(initialization_message.route_roadblock_ids, ['a', 'b', 'c'])

    @patch.object(RemotePlanner, '_planner_initializations_to_message', return_value=123, autospec=True)
    @patch('grpc.insecure_channel')
    @patch('nuplan.submission.challenge_pb2_grpc.DetectionTracksChallengeStub', autospec=True)
    @patch('nuplan.planning.simulation.planner.remote_planner.SubmissionContainerManager', Mock(spec_set=SubmissionContainerManager))
    @patch('nuplan.planning.simulation.planner.remote_planner.find_free_port_number')
    def test_initialize(self, mock_find_port: Mock, mock_stub_function: Mock, mock_channel: Mock, initialization_to_message: Mock) -> None:
        """Tests that the initialization request is called correctly."""
        mock_initialization = Mock()
        mock_stub = Mock()
        mock_stub_function.return_value = mock_stub
        self.planner.initialize(mock_initialization)
        mock_channel.assert_called()
        initialization_to_message.assert_called_with(mock_initialization)
        self.planner._stub.InitializePlanner.assert_called_with(123)
        self.planner_with_container.initialize(mock_initialization)
        self.planner_with_container.submission_container_manager.get_submission_container.assert_called_with(self.planner_with_container.submission_image, self.planner_with_container.container_name, mock_find_port())
        self.planner_with_container.submission_container_manager.get_submission_container().start.assert_called()

    @patch.object(RemotePlanner, '_compute_trajectory')
    @patch('grpc.insecure_channel', Mock())
    @patch('nuplan.submission.challenge_pb2_grpc.DetectionTracksChallengeStub', Mock(spec_set=DetectionTracksChallengeStub))
    def test_compute_trajectory_interface(self, mock_compute_trajectory: Mock) -> None:
        """Tests that the interface for the trajectory computation request is called correctly."""
        mock_compute_trajectory.return_value = 'trajectories'
        mock_input = [Mock()]
        trajectories = self.planner.compute_trajectory(mock_input)
        mock_compute_trajectory.assert_called_with(self.planner._stub, current_input=mock_input)
        self.assertEqual('trajectories', trajectories)

    @patch('nuplan.planning.simulation.planner.remote_planner.interp_traj_from_proto_traj', Mock)
    @patch('nuplan.planning.simulation.planner.remote_planner.proto_tl_status_data_from_tl_status_data')
    @patch('nuplan.submission.challenge_pb2.PlannerInput')
    @patch('nuplan.submission.challenge_pb2.SimulationIteration')
    @patch('nuplan.submission.challenge_pb2.SimulationHistoryBuffer')
    def test_compute_trajectory(self, history_buffer: Mock, simulation_iteration: Mock, planner_input: Mock, mock_proto_tl_status_data: Mock) -> None:
        """Tests deserialization and serialization of the input/output for the trajectory computation interface."""
        with patch.object(self.planner, '_get_history_update', MagicMock()) as get_history_update:
            get_history_update.return_value = [['states'], ['observations'], ['intervals']]
            mock_stub = MagicMock()
            mock_tl_data = Mock()
            mock_input = Mock(iteration=Mock(time_us=1, index=0), history=Mock(ego_states='fake_input'), traffic_light_data=[mock_tl_data])
            mock_input.history.ego_states = ['fake_input']
            planner_input.return_value = 'planner input'
            simulation_iteration.return_value = 'iter_1'
            history_buffer.return_value = 'hb_1'
            self.planner._compute_trajectory(mock_stub, mock_input)
            get_history_update.assert_called_once_with(mock_input)
            mock_proto_tl_status_data.assert_called_once_with(mock_tl_data)
            simulation_iteration.assert_has_calls([call(time_us=1, index=0)])
            planner_input.assert_has_calls([call(simulation_iteration='iter_1', simulation_history_buffer='hb_1', traffic_light_data=[mock_proto_tl_status_data.return_value])])
            mock_stub.ComputeTrajectory.assert_called_once_with(planner_input.return_value, timeout=1)

    @patch('pickle.dumps')
    def test_get_history_update(self, mock_dumps: Mock) -> None:
        """Tests that the history update is built correctly."""
        planner_input = Mock()
        planner_input.history.ego_states = [1, 2]
        planner_input.history.observations = [4, 5]
        planner_input.history.current_state = (6, 7)
        serialized_states, serialized_observations, sample_interval = self.planner._get_history_update(planner_input)
        calls = [call(1), call(2), call(4), call(5)]
        mock_dumps.assert_has_calls(calls)
        self.planner.serialized_state = serialized_states
        self.planner.serialized_observation = serialized_observations
        self.planner.sample_intervals = sample_interval
        _, _, _ = self.planner._get_history_update(planner_input)
        calls = calls + [call(6), call(7)]
        mock_dumps.assert_has_calls(calls)

class TestMlPlanner(unittest.TestCase):
    """
    Test MLPlanner with two models.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.scenario = get_test_nuplan_scenario()

    def test_simple_vector_net_model(self) -> None:
        """Test Model Vector Map Simple"""
        self.run_test_ml_planner(construct_simple_vector_map_ml_planner())

    def test_raster_net_model(self) -> None:
        """Test Raster Net model"""
        self.run_test_ml_planner(construct_raster_ml_planner())

    def test_urban_driver_open_loop_model(self) -> None:
        """Test UrbanDriverOpenLoop model"""
        self.run_test_ml_planner(construct_urban_driver_open_loop_ml_planner())

    def run_test_ml_planner(self, planner: MLPlanner) -> None:
        """Tests if progress is calculated correctly"""
        scenario = self.scenario
        simulation_history_buffer_duration = 2
        buffer_size = int(simulation_history_buffer_duration / self.scenario.database_interval + 1)
        history = SimulationHistoryBuffer.initialize_from_scenario(buffer_size=buffer_size, scenario=self.scenario, observation_type=DetectionsTracks)
        initialization = PlannerInitialization(route_roadblock_ids=scenario.get_route_roadblock_ids(), mission_goal=scenario.get_mission_goal(), map_api=scenario.map_api)
        planner.initialize(initialization)
        trajectory = planner.compute_trajectory(PlannerInput(iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history, traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0))))
        self.assertNotEqual(trajectory, None)
        self.assertEqual(len(trajectory.get_sampled_trajectory()), planner._num_output_dim + 1)

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

class TestEgoCentricMLAgents(unittest.TestCase):
    """Test important functions of EgoCentricMLAgents class"""

    def setUp(self) -> None:
        """Initialize scenario and model for constructing AgentCentricMLAgents class."""
        self.scenario = MockAbstractScenario(number_of_detections=1)
        self.model = Mock(spec=TorchModuleWrapper)
        self.model.future_trajectory_sampling = TrajectorySampling(num_poses=1, time_horizon=1.0)
        self.pred_trajectory = AgentsTrajectories(data=[np.array([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]])])

    def test_update_observation_with_predictions(self) -> None:
        """Test the _update_observation_with_predictions fucntion."""
        obs = EgoCentricMLAgents(model=self.model, scenario=self.scenario)
        obs.initialize()
        self.assertEqual(len(obs._agents), 1)
        self.assertEqual(obs._agents['0'].center.x, 1.0)
        self.assertEqual(obs._agents['0'].center.y, 2.0)
        self.assertAlmostEqual(obs._agents['0'].center.heading, np.pi / 2)
        obs.step_time = TimePoint(1000000.0)
        predictions = {'agents_trajectory': self.pred_trajectory}
        obs._update_observation_with_predictions(predictions)
        self.assertEqual(len(obs._agents), 1)
        self.assertAlmostEqual(obs._agents['0'].center.x, 1.0)
        self.assertAlmostEqual(obs._agents['0'].center.y, 1.0)
        self.assertAlmostEqual(obs._agents['0'].center.heading, 0.0)

    @patch('nuplan.planning.simulation.planner.ml_planner.model_loader.ModelLoader.infer')
    def test_infer_model(self, mock_infer: Mock) -> None:
        """Test _infer_model function."""
        predictions = {'agents_trajectory': self.pred_trajectory.to_feature_tensor()}
        mock_infer.return_value = predictions
        obs = EgoCentricMLAgents(model=self.model, scenario=self.scenario)
        obs.initialize()
        agents_raster = Mock(spec=Agents)
        features = {'agents': agents_raster}
        results = obs._infer_model(features)
        mock_infer.assert_called_with(features)
        self.assertIn(obs.prediction_type, results)
        self.assertIsInstance(results[obs.prediction_type], AgentsTrajectories)
        self.assertIsInstance(results[obs.prediction_type].data[0], np.ndarray)

class TestProfileIDM(unittest.TestCase):
    """
    Profiling test for IDM agents.
    """

    def setUp(self) -> None:
        """
        Inherited, see super class.
        """
        self.n_repeat_trials = 1
        self.display_results = True
        self.scenario = get_test_nuplan_scenario()

    def test_profile_idm_agent_observation(self) -> None:
        """Profile IDMAgents."""
        profiler = Profiler(interval=0.0001)
        profiler.start()
        for _ in range(self.n_repeat_trials):
            observation = IDMAgents(target_velocity=10, min_gap_to_lead_agent=0.5, headway_time=1.5, accel_max=1.0, decel_max=2.0, scenario=self.scenario, open_loop_detections_types=[])
            for step in range(self.scenario.get_number_of_iterations() - 1):
                iteration = SimulationIteration(time_point=self.scenario.get_time_point(step), index=step)
                next_iteration = SimulationIteration(time_point=self.scenario.get_time_point(step + 1), index=step + 1)
                buffer = SimulationHistoryBuffer.initialize_from_list(1, [self.scenario.get_ego_state_at_iteration(step)], [self.scenario.get_tracked_objects_at_iteration(step)], next_iteration.time_point.time_s - iteration.time_point.time_s)
                observation.update_observation(iteration, next_iteration, buffer)
        profiler.stop()
        if self.display_results:
            logger.info(profiler.output_text(unicode=True, color=True))

@nuplan_test(path='json/idm_agent_observation/baseline.json')
def test_idm_observations(scene: Dict[str, Any]) -> None:
    """
    Overall integration test of IDM smart agents
    """
    simulation_step = 17
    observation = IDMAgents(target_velocity=10, min_gap_to_lead_agent=0.5, headway_time=1.5, accel_max=1.0, decel_max=2.0, scenario=scenario, open_loop_detections_types=[])
    agent_history: Dict[str, List[StateSE2]] = dict()
    for step in range(simulation_step):
        iteration = SimulationIteration(time_point=scenario.get_time_point(step), index=step)
        next_iteration = SimulationIteration(time_point=scenario.get_time_point(step + 1), index=step + 1)
        buffer = SimulationHistoryBuffer.initialize_from_list(1, [scenario.get_ego_state_at_iteration(step)], [scenario.get_tracked_objects_at_iteration(step)], sample_interval=next_iteration.time_point.time_s - iteration.time_point.time_s)
        observation.update_observation(iteration, next_iteration, buffer)
        for agent_id, agent in observation._idm_agent_manager.agents.items():
            if agent_id not in agent_history:
                agent_history[agent_id] = []
            agent_history[agent_id].append(agent.to_se2())

class AbstractTrajectory(metaclass=ABCMeta):
    """
    Generic agent or ego trajectory interface.
    """

    @property
    @abstractmethod
    def start_time(self) -> TimePoint:
        """
        Get the trajectory start time.
        :return: Start time.
        """
        pass

    @property
    @abstractmethod
    def end_time(self) -> TimePoint:
        """
        Get the trajectory end time.
        :return: End time.
        """
        pass

    @property
    def duration(self) -> float:
        """
        :return: the time duration of the trajectory
        """
        return self.end_time.time_s - self.start_time.time_s

    @property
    def duration_us(self) -> int:
        """
        :return: the time duration of the trajectory in micro seconds
        """
        return int(self.end_time.time_us - self.start_time.time_us)

    @abstractmethod
    def get_state_at_time(self, time_point: TimePoint) -> Any:
        """
        Get the state of the actor at the specified time point.
        :param time_point: Time for which are want to query a state.
        :return: State at the specified time.

        :raises AssertionError: Throws an exception in case a time_point is beyond range of a trajectory.
        """
        pass

    @abstractmethod
    def get_state_at_times(self, time_points: List[TimePoint]) -> List[Any]:
        """
        Get the state of the actor at the specified time points.
        :param time_points: List of time points for which are want to query a state.
        :return: States at the specified time.

        :raises AssertionError: Throws an exception in case a time_point is beyond range of a trajectory.
        """
        pass

    @abstractmethod
    def get_sampled_trajectory(self) -> List[Any]:
        """
        Get the sampled states along the trajectory.
        :return: Discrete trajectory consisting of states.
        """
        pass

    def is_in_range(self, time_point: Union[TimePoint, int]) -> bool:
        """
        Check whether a time point is in range of trajectory.
        :return: True if it is, False otherwise.
        """
        if isinstance(time_point, int):
            time_point = TimePoint(time_point)
        return bool(self.start_time <= time_point <= self.end_time)

class InterpolatedTrajectory(AbstractTrajectory):
    """Class representing a trajectory that can be interpolated from a list of points."""

    def __init__(self, trajectory: List[InterpolatableState]):
        """
        :param trajectory: List of states creating a trajectory.
            The trajectory has to have at least 2 elements, otherwise it is considered invalid and the class will raise.
        """
        assert trajectory, "Trajectory can't be empty!"
        assert isinstance(trajectory[0], InterpolatableState)
        self._trajectory_class = trajectory[0].__class__
        assert all((isinstance(point, self._trajectory_class) for point in trajectory))
        if len(trajectory) <= 1:
            raise ValueError(f'There is not enough states in trajectory: {len(trajectory)}!')
        self._trajectory = trajectory
        time_series = [point.time_us for point in trajectory]
        linear_states = []
        angular_states = []
        for point in trajectory:
            split_state = point.to_split_state()
            linear_states.append(split_state.linear_states)
            angular_states.append(split_state.angular_states)
        self._fixed_state = trajectory[0].to_split_state().fixed_states
        linear_states = np.array(linear_states, dtype='float64')
        angular_states = np.array(angular_states, dtype='float64')
        self._function_interp_linear = sp_interp.interp1d(time_series, linear_states, axis=0)
        self._angular_interpolator = AngularInterpolator(time_series, angular_states)

    def __reduce__(self) -> Tuple[Type[InterpolatedTrajectory], Tuple[Any, ...]]:
        """
        Helper for pickling.
        """
        return (self.__class__, (self._trajectory,))

    @property
    def start_time(self) -> TimePoint:
        """Inherited, see superclass."""
        return self._trajectory[0].time_point

    @property
    def end_time(self) -> TimePoint:
        """Inherited, see superclass."""
        return self._trajectory[-1].time_point

    def get_state_at_time(self, time_point: TimePoint) -> InterpolatableState:
        """Inherited, see superclass."""
        start_time = self.start_time
        end_time = self.end_time
        assert start_time <= time_point <= end_time, f'Interpolation time time_point={time_point!r} not in trajectory time window! \nstart_time.time_us={start_time.time_us!r} <= time_point.time_us={time_point.time_us!r} <= end_time.time_us={end_time.time_us!r}'
        linear_states = list(self._function_interp_linear(time_point.time_us))
        angular_states = list(self._angular_interpolator.interpolate(time_point.time_us))
        return self._trajectory_class.from_split_state(SplitState(linear_states, angular_states, self._fixed_state))

    def get_state_at_times(self, time_points: List[TimePoint]) -> List[InterpolatableState]:
        """Inherited, see superclass."""
        start_time = self.start_time
        end_time = self.end_time
        assert start_time <= min(time_points), f'Interpolation time not in trajectory time window! The following is not satisfied:Trajectory start time: ({start_time.time_s}) <= Earliest interpolation time ({min(time_points).time_s}) {max(time_points).time_s} <= {end_time.time_s} '
        assert max(time_points) <= end_time, f'Interpolation time not in trajectory time window! The following is not satisfied:Trajectory end time: ({end_time.time_s}) >= Latest interpolation time ({max(time_points).time_s}) '
        interpolation_times = [t.time_us for t in time_points]
        linear_states = list(self._function_interp_linear(interpolation_times))
        angular_states = list(self._angular_interpolator.interpolate(interpolation_times))
        return [self._trajectory_class.from_split_state(SplitState(lin_state, ang_state, self._fixed_state)) for lin_state, ang_state in zip(linear_states, angular_states)]

    def get_sampled_trajectory(self) -> List[InterpolatableState]:
        """Inherited, see superclass."""
        return self._trajectory

class TestInterpolatedTrajectory(unittest.TestCase):
    """Tests implementation of InterpolatedTrajectory."""

    def setUp(self) -> None:
        """Test setup."""
        self.split_state_1 = Mock(linear_states=[123], angular_states=[2.13], fixed_states=['fix'], autspec=SplitState)
        self.split_state_2 = Mock(linear_states=[456], angular_states=[3.13], fixed_states=['fix'], autspec=SplitState)
        self.start_time_point = TimePoint(0)
        self.end_time_point = TimePoint(int(1000000.0))
        self.points = [MagicMock(time_point=self.start_time_point, time_us=self.start_time_point.time_us, to_split_state=lambda: self.split_state_1, spec=MockPoint), MagicMock(time_point=self.end_time_point, time_us=self.end_time_point.time_us, to_split_state=lambda: self.split_state_2, spec=MockPoint)]
        self.trajectory = InterpolatedTrajectory(self.points)

    def tearDown(self) -> None:
        """Resets mock objects."""
        MockPoint.reset_calls()

    @patch('nuplan.planning.simulation.trajectory.interpolated_trajectory.sp_interp')
    @patch('nuplan.planning.simulation.trajectory.interpolated_trajectory.np')
    @patch('nuplan.planning.simulation.trajectory.interpolated_trajectory.AngularInterpolator', autospec=True)
    def test_initialization(self, mock_interp_angular: Mock, mock_np: Mock, mock_sp_interp: Mock) -> None:
        """Tests that initialization works as intended."""
        mock_sp_interp.interp1d.return_value = 'interp_function'
        mock_np.array.return_value = 'array'
        trajectory = InterpolatedTrajectory(self.points)
        self.assertEqual(trajectory._trajectory_class, MockPoint)
        self.assertEqual(trajectory._fixed_state, ['fix'])
        mock_sp_interp.interp1d.assert_called_with([0, 1000000], mock_np.array.return_value, axis=0)
        self.assertEqual(trajectory._function_interp_linear, mock_sp_interp.interp1d.return_value)
        mock_interp_angular.assert_called_with([0, 1000000], 'array')
        self.assertEqual(trajectory._angular_interpolator, mock_interp_angular.return_value)
        with self.assertRaises(AssertionError):
            InterpolatedTrajectory([MagicMock()])

    def test_start_end_time(self) -> None:
        """Tests that properties return correct members."""
        self.assertEqual(self.start_time_point, self.trajectory.start_time)
        self.assertEqual(self.end_time_point, self.trajectory.end_time)

    def test_get_state_at_time(self) -> None:
        """Tests interpolation method."""
        time_point = TimePoint(int(0.5 * 1000000.0))
        state = self.trajectory.get_state_at_time(time_point)
        self.assertEqual('foo', state)
        interpolated_state = SplitState(linear_states=[289.5], angular_states=[2.63], fixed_states=['fix'])
        self.assertEqual(MockPoint.from_split_state.calls, [interpolated_state])
        time_point_outside_interval = TimePoint(int(5 * 1000000.0))
        with self.assertRaises(AssertionError):
            self.trajectory.get_state_at_time(time_point_outside_interval)

    def test_get_state_at_times(self) -> None:
        """Tests batch interpolation method."""
        time_points = [TimePoint(0), TimePoint(int(0.5 * 1000000.0))]
        states = self.trajectory.get_state_at_times(time_points)
        self.assertEqual(['foo', 'foo'], states)
        initial_state = SplitState(linear_states=[123], angular_states=[2.13], fixed_states=['fix'])
        interpolated_state = SplitState(linear_states=[289.5], angular_states=[2.63], fixed_states=['fix'])
        self.assertEqual(MockPoint.from_split_state.calls, [initial_state, interpolated_state])
        time_point_outside_interval = TimePoint(int(5 * 1000000.0))
        with self.assertRaises(AssertionError):
            self.trajectory.get_state_at_times([time_point_outside_interval])

    def test_get_sampled_trajectory(self) -> None:
        """Tests getter for entire trajectory."""
        self.assertEqual(self.points, self.trajectory.get_sampled_trajectory())

class TestSimulationHistoryBuffer(unittest.TestCase):
    """Test suite for SimulationHistoryBuffer"""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = MockAbstractScenario(number_of_past_iterations=20)
        self.buffer_size = 10

    def test_initialize_with_box(self) -> None:
        """Test the initialize function"""
        tracks_observation = TracksObservation(self.scenario)
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(buffer_size=self.buffer_size, scenario=self.scenario, observation_type=tracks_observation.observation_type())
        self.assertEqual(len(history_buffer), self.buffer_size)

    def test_initialize_with_lidar_pc(self) -> None:
        """Test the initialize function"""
        lidar_pc_observation = LidarPcObservation(self.scenario)
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(buffer_size=self.buffer_size, scenario=self.scenario, observation_type=lidar_pc_observation.observation_type())
        self.assertEqual(len(history_buffer), self.buffer_size)

    def test_initialize_from_list(self) -> None:
        """Test the initialization from lists"""
        history_buffer = SimulationHistoryBuffer.initialize_from_list(buffer_size=self.buffer_size, ego_states=[self.scenario.initial_ego_state], observations=[self.scenario.initial_tracked_objects], sample_interval=0.05)
        self.assertEqual(len(history_buffer), 1)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state])
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects])

    def test_append(self) -> None:
        """Test the append function"""
        history_buffer = SimulationHistoryBuffer(ego_state_buffer=deque([Mock()], maxlen=1), observations_buffer=deque([Mock()], maxlen=1))
        history_buffer.append(self.scenario.initial_ego_state, self.scenario.initial_tracked_objects)
        self.assertEqual(len(history_buffer), 1)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state])
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects])

    def test_extend(self) -> None:
        """Test the extend function"""
        history_buffer = SimulationHistoryBuffer(ego_state_buffer=deque([Mock()], maxlen=2), observations_buffer=deque([Mock()], maxlen=2))
        history_buffer.extend([self.scenario.initial_ego_state] * 2, [self.scenario.initial_tracked_objects] * 2)
        self.assertEqual(len(history_buffer), 2)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state] * 2)
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects] * 2)

class StepSimulationTimeController(AbstractSimulationTimeController):
    """
    Class handling simulation time and completion.
    """

    def __init__(self, scenario: AbstractScenario):
        """
        Initialize simulation control.
        """
        self.current_iteration_index = 0
        self.scenario = scenario

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration_index = 0

    def get_iteration(self) -> SimulationIteration:
        """Inherited, see superclass."""
        scenario_time = self.scenario.get_time_point(self.current_iteration_index)
        return SimulationIteration(time_point=scenario_time, index=self.current_iteration_index)

    def next_iteration(self) -> Optional[SimulationIteration]:
        """Inherited, see superclass."""
        self.current_iteration_index += 1
        return None if self.reached_end() else self.get_iteration()

    def reached_end(self) -> bool:
        """Inherited, see superclass."""
        return self.current_iteration_index >= self.number_of_iterations() - 1

    def number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return cast(int, self.scenario.get_number_of_iterations())

class TestAgentsFeatureBuilder(unittest.TestCase):
    """Test builder that constructs agent features during training and simulation."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.batch_size = 1
        self.past_time_horizon = 4.0
        self.num_past_poses = 4
        self.num_total_past_poses = self.num_past_poses + 1

    @given(number_of_detections=st.sampled_from([0, 10]), feature_builder_type=st.sampled_from([TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]))
    def test_agent_feature_builder(self, number_of_detections: int, feature_builder_type: TrackedObjectType) -> None:
        """
        Test AgentFeatureBuilder with and without agents in the scene for both pedestrian and vehicles
        """
        feature_builder = AgentsFeatureBuilder(TrajectorySampling(self.num_past_poses, self.past_time_horizon), feature_builder_type)
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=number_of_detections, tracked_object_types=[TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN])
        feature = feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), Agents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), Agents.ego_state_dim())
        self.assertEqual(len(feature.agents), self.batch_size)
        self.assertEqual(len(feature.agents[0]), self.num_total_past_poses)
        self.assertEqual(feature.agents[0].shape[1], number_of_detections)
        self.assertEqual(feature.agents[0].shape[2], Agents.agents_states_dim())

    @given(feature_builder_type=st.sampled_from([TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]))
    def test_get_feature_from_simulation(self, feature_builder_type: TrackedObjectType) -> None:
        """
        Test get feature from simulation
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=10, tracked_object_types=[TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN])
        mock_meta_data = PlannerInitialization(map_api=MockAbstractMap(), route_roadblock_ids=None, mission_goal=StateSE2(0, 0, 0))
        ego_past_states = list(scenario.get_ego_past_trajectory(iteration=0, num_samples=10, time_horizon=5))
        ego_initial_state = scenario.initial_ego_state
        ego_history = ego_past_states + [ego_initial_state]
        past_observations = list(scenario.get_past_tracked_objects(iteration=0, num_samples=10, time_horizon=5))
        initial_observation = scenario.initial_tracked_objects
        observation_history = past_observations + [initial_observation]
        history = SimulationHistoryBuffer.initialize_from_list(len(ego_history), ego_history, observation_history, scenario.database_interval)
        current_input = PlannerInput(iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history, traffic_light_data=scenario.get_traffic_light_status_at_iteration(0))
        feature_builder = AgentsFeatureBuilder(TrajectorySampling(self.num_past_poses, self.past_time_horizon), feature_builder_type)
        feature = feature_builder.get_features_from_simulation(current_input=current_input, initialization=mock_meta_data)
        self.assertEqual(type(feature), Agents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), Agents.ego_state_dim())
        self.assertEqual(len(feature.agents), self.batch_size)
        self.assertEqual(len(feature.agents[0]), self.num_total_past_poses)
        self.assertEqual(feature.agents[0].shape[1], 10)
        self.assertEqual(feature.agents[0].shape[2], Agents.agents_states_dim())

    @given(feature_builder_type=st.sampled_from([TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]))
    @settings(deadline=1000)
    def test_agents_feature_builder_scripts_properly(self, feature_builder_type: TrackedObjectType) -> None:
        """
        Tests that the Agents Feature Builder scripts properly
        """
        feature_builder = AgentsFeatureBuilder(TrajectorySampling(self.num_past_poses, self.past_time_horizon), feature_builder_type)
        config = feature_builder.precomputed_feature_config()
        for expected_key in ['past_ego_states', 'past_time_stamps', 'past_tracked_objects']:
            self.assertIn(expected_key, config)
            config_dict = config[expected_key]
            self.assertEqual(len(config_dict), 3)
            self.assertEqual(0, int(config_dict['iteration']))
            self.assertEqual(self.num_past_poses, int(config_dict['num_samples']))
            self.assertEqual(self.past_time_horizon, int(float(config_dict['time_horizon'])))
        num_frames = 5
        num_agents = 3
        ego_dim = EgoInternalIndex.dim()
        agent_dim = AgentInternalIndex.dim()
        past_ego_states = torch.zeros((num_frames, ego_dim), dtype=torch.float32)
        past_timestamps = torch.tensor([i * 50 for i in range(num_frames)], dtype=torch.int64)
        past_tracked_objects = [torch.ones((num_agents, agent_dim), dtype=torch.float32) for _ in range(num_frames)]
        for i in range(num_frames):
            for j in range(num_agents):
                past_tracked_objects[i][j, :] *= j + 1
        tensor_data = {'past_ego_states': past_ego_states, 'past_time_stamps': past_timestamps}
        list_tensor_data = {'past_tracked_objects': past_tracked_objects}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_builder = torch.jit.script(feature_builder)
        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        py_tensors, py_list_tensors, py_list_list_tensors = feature_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        self.assertEqual(0, len(scripted_tensors))
        self.assertEqual(0, len(py_tensors))
        self.assertEqual(len(scripted_list_tensors), len(py_list_tensors))
        for key in py_list_tensors:
            scripted_list = scripted_list_tensors[key]
            py_list = py_list_tensors[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                scripted = scripted_list[i]
                py = py_list[i]
                torch.testing.assert_allclose(py, scripted, atol=0.01, rtol=0.01)
        self.assertEqual(0, len(scripted_list_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))

class TestVectorMapFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs map features in vectorized format."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = get_test_nuplan_scenario()

    def test_vector_map_feature_builder(self) -> None:
        """
        Test VectorMapFeatureBuilder
        """
        feature_builder = VectorMapFeatureBuilder(radius=20, connection_scales=[2])
        self.assertEqual(feature_builder.get_feature_type(), VectorMap)
        features = feature_builder.get_features_from_scenario(self.scenario)
        self.assertEqual(type(features), VectorMap)
        ego_state = self.scenario.initial_ego_state
        detections = self.scenario.initial_tracked_objects
        meta_data = PlannerInitialization(map_api=self.scenario.map_api, mission_goal=self.scenario.get_mission_goal(), route_roadblock_ids=self.scenario.get_route_roadblock_ids())
        history = SimulationHistoryBuffer.initialize_from_list(1, [ego_state], [detections], self.scenario.database_interval)
        iteration = SimulationIteration(TimePoint(0), 0)
        tl_data = self.scenario.get_traffic_light_status_at_iteration(iteration.index)
        current_input = PlannerInput(iteration=iteration, history=history, traffic_light_data=tl_data)
        features_sim = feature_builder.get_features_from_simulation(current_input=current_input, initialization=meta_data)
        self.assertEqual(type(features_sim), VectorMap)
        self.assertTrue(np.allclose(features_sim.coords[0], features.coords[0], atol=0.0001))
        for connections, connections_simulation in zip(features_sim.multi_scale_connections[0].values(), features.multi_scale_connections[0].values()):
            self.assertTrue(np.allclose(connections, connections_simulation))
        for lane in range(len(features_sim.lane_groupings[0])):
            for lane_groupings, lane_groupings_simulation in zip(features_sim.lane_groupings[0][lane], features.lane_groupings[0][lane]):
                self.assertTrue(np.allclose(lane_groupings, lane_groupings_simulation))
        self.assertTrue(np.allclose(features_sim.on_route_status[0], features.on_route_status[0], atol=0.0001))
        self.assertTrue(np.allclose(features_sim.traffic_light_data[0], features.traffic_light_data[0]))

    def test_vector_map_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the VectorMapFeatureBuilder can be scripted properly.
        """
        feature_builder = VectorMapFeatureBuilder(radius=20, connection_scales=[2])
        self.assertEqual(feature_builder.get_feature_type(), VectorMap)
        scripted_builder = torch.jit.script(feature_builder)
        self.assertIsNotNone(scripted_builder)
        config = scripted_builder.precomputed_feature_config()
        self.assertTrue('initial_ego_state' in config)
        self.assertTrue('neighbor_vector_map' in config)
        self.assertTrue('radius' in config['neighbor_vector_map'])
        self.assertEqual('20', config['neighbor_vector_map']['radius'])
        num_lane_segment = 5
        num_connections = 7
        tensor_data = {'lane_segment_coords': torch.rand((num_lane_segment, 2, 2), dtype=torch.float64), 'lane_segment_conns': torch.zeros((num_connections, 2), dtype=torch.int64), 'on_route_status': torch.zeros((num_lane_segment, 2), dtype=torch.float32), 'traffic_light_array': torch.zeros((num_lane_segment, 4), dtype=torch.float32), 'anchor_state': torch.zeros((3,), dtype=torch.float64)}
        list_tensor_data = {'lane_segment_groupings': [torch.zeros(size=(2,), dtype=torch.int64) for _ in range(num_lane_segment)]}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_tensor_output, scripted_list_output, scripted_list_list_output = scripted_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        py_tensor_output, py_list_output, py_list_list_output = feature_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        self.assertEqual(0, len(scripted_tensor_output))
        self.assertEqual(0, len(py_tensor_output))
        self.assertEqual(len(scripted_list_output), len(py_list_output))
        for key in py_list_output:
            self.assertEqual(len(py_list_output[key]), len(scripted_list_output[key]))
            for i in range(len(py_list_output[key])):
                torch.testing.assert_close(py_list_output[key][i], scripted_list_output[key][i])
        self.assertEqual(len(py_list_list_output), len(scripted_list_list_output))
        for key in py_list_list_output:
            py_list = py_list_list_output[key]
            scripted_list = scripted_list_list_output[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                py = py_list[i]
                script = scripted_list[i]
                self.assertEqual(len(py), len(script))
                for j in range(len(py)):
                    torch.testing.assert_close(py[j], script[j])

class TestVectorSetMapFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs map features in vector set format."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = MockAbstractScenario()
        self.batch_size = 1
        self.radius = 35
        self.interpolation_method = 'linear'
        self.map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        self.max_elements = {'LANE': 30, 'LEFT_BOUNDARY': 30, 'RIGHT_BOUNDARY': 30, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 30}
        self.max_points = {'LANE': 20, 'LEFT_BOUNDARY': 20, 'RIGHT_BOUNDARY': 20, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 20}
        self.feature_builder = VectorSetMapFeatureBuilder(map_features=self.map_features, max_elements=self.max_elements, max_points=self.max_points, radius=self.radius, interpolation_method=self.interpolation_method)

    def test_vector_set_map_feature_builder(self) -> None:
        """
        Tests VectorSetMapFeatureBuilder.
        """
        self.assertEqual(self.feature_builder.get_feature_type(), VectorSetMap)
        features = self.feature_builder.get_features_from_scenario(self.scenario)
        self.assertEqual(type(features), VectorSetMap)
        self.assertEqual(features.batch_size, self.batch_size)
        for feature_name in self.map_features:
            self.assertEqual(features.coords[feature_name][0].shape, (self.max_elements[feature_name], self.max_points[feature_name], 2))
            self.assertEqual(features.availabilities[feature_name][0].shape, (self.max_elements[feature_name], self.max_points[feature_name]))
            np.testing.assert_array_equal(features.availabilities[feature_name][0], np.zeros((self.max_elements[feature_name], self.max_points[feature_name]), dtype=np.bool_))

    def test_vector_set_map_feature_builder_simulation_and_scenario_match(self) -> None:
        """
        Tests that get_features_from_scenario and get_features_from_simulation give same results.
        """
        features = self.feature_builder.get_features_from_scenario(self.scenario)
        ego_state = self.scenario.initial_ego_state
        detections = self.scenario.initial_tracked_objects
        meta_data = PlannerInitialization(map_api=self.scenario.map_api, mission_goal=self.scenario.get_mission_goal(), route_roadblock_ids=self.scenario.get_route_roadblock_ids())
        history = SimulationHistoryBuffer.initialize_from_list(1, [ego_state], [detections], self.scenario.database_interval)
        iteration = SimulationIteration(TimePoint(0), 0)
        tl_data = self.scenario.get_traffic_light_status_at_iteration(iteration.index)
        current_input = PlannerInput(iteration=iteration, history=history, traffic_light_data=tl_data)
        features_sim = self.feature_builder.get_features_from_simulation(current_input=current_input, initialization=meta_data)
        self.assertEqual(type(features_sim), VectorSetMap)
        self.assertEqual(set(features_sim.coords.keys()), set(features.coords.keys()))
        for feature_name in features_sim.coords.keys():
            np.testing.assert_allclose(features_sim.coords[feature_name][0], features.coords[feature_name][0], atol=0.0001)
        self.assertEqual(set(features_sim.traffic_light_data.keys()), set(features.traffic_light_data.keys()))
        for feature_name in features_sim.traffic_light_data.keys():
            np.testing.assert_allclose(features_sim.traffic_light_data[feature_name][0], features.traffic_light_data[feature_name][0])
        self.assertEqual(set(features_sim.availabilities.keys()), set(features.availabilities.keys()))
        for feature_name in features_sim.availabilities.keys():
            np.testing.assert_array_equal(features_sim.availabilities[feature_name][0], features.availabilities[feature_name][0])

    def test_vector_set_map_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the VectorSetMapFeatureBuilder can be scripted properly.
        """
        self.assertEqual(self.feature_builder.get_feature_type(), VectorSetMap)
        scripted_builder = torch.jit.script(self.feature_builder)
        self.assertIsNotNone(scripted_builder)
        config = scripted_builder.precomputed_feature_config()
        self.assertTrue('initial_ego_state' in config)
        self.assertTrue('neighbor_vector_set_map' in config)
        self.assertTrue('radius' in config['neighbor_vector_set_map'])
        self.assertEqual(str(self.radius), config['neighbor_vector_set_map']['radius'])
        self.assertEqual(str(self.interpolation_method), config['neighbor_vector_set_map']['interpolation_method'])
        self.assertEqual(','.join(self.map_features), config['neighbor_vector_set_map']['map_features'])
        max_elements: List[str] = [f'{feature_name}.{feature_max_elements}' for feature_name, feature_max_elements in self.max_elements.items()]
        max_points: List[str] = [f'{feature_name}.{feature_max_points}' for feature_name, feature_max_points in self.max_points.items()]
        self.assertEqual(','.join(max_elements), config['neighbor_vector_set_map']['max_elements'])
        self.assertEqual(','.join(max_points), config['neighbor_vector_set_map']['max_points'])
        tensor_data = {'anchor_state': torch.zeros((3,))}
        for feature_name in self.map_features:
            feature_max_elements = self.max_elements[feature_name]
            feature_max_points = self.max_points[feature_name]
            tensor_data[f'coords.{feature_name}'] = torch.rand((feature_max_elements, feature_max_points, 2), dtype=torch.float64)
            tensor_data[f'traffic_light_data.{feature_name}'] = torch.zeros((feature_max_elements, feature_max_points, 4), dtype=torch.int64)
            tensor_data[f'availabilities.{feature_name}'] = torch.zeros((feature_max_elements, feature_max_points), dtype=torch.bool)
        list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_tensor_output, scripted_list_output, scripted_list_list_output = scripted_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        py_tensor_output, py_list_output, py_list_list_output = self.feature_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        self.assertEqual(0, len(scripted_tensor_output))
        self.assertEqual(0, len(py_tensor_output))
        self.assertEqual(len(scripted_list_output), len(py_list_output))
        for key in py_list_output:
            self.assertEqual(len(py_list_output[key]), len(scripted_list_output[key]))
            for i in range(len(py_list_output[key])):
                torch.testing.assert_close(py_list_output[key][i], scripted_list_output[key][i])
        self.assertEqual(len(py_list_list_output), len(scripted_list_list_output))
        for key in py_list_list_output:
            py_list = py_list_list_output[key]
            scripted_list = scripted_list_list_output[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                py = py_list[i]
                script = scripted_list[i]
                self.assertEqual(len(py), len(script))
                for j in range(len(py)):
                    torch.testing.assert_close(py[j], script[j])

class TestGenericAgentsFeatureBuilder(unittest.TestCase):
    """Test builder that constructs agent features during training and simulation."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.batch_size = 1
        self.past_time_horizon = 4.0
        self.num_agents = 10
        self.num_past_poses = 4
        self.num_total_past_poses = self.num_past_poses + 1
        self.agent_features = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'TRAFFIC_CONE', 'BARRIER', 'CZONE_SIGN', 'GENERIC_OBJECT']
        self.tracked_object_types: List[TrackedObjectType] = []
        for feature_name in self.agent_features:
            try:
                self.tracked_object_types.append(TrackedObjectType[feature_name])
            except KeyError:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')
        self.feature_builder = GenericAgentsFeatureBuilder(self.agent_features, TrajectorySampling(num_poses=self.num_past_poses, time_horizon=self.past_time_horizon))

    def test_generic_agent_feature_builder(self) -> None:
        """
        Test GenericAgentFeatureBuilder
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents, tracked_object_types=self.tracked_object_types)
        feature = self.feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_no_agents(self) -> None:
        """
        Test when there are no agents
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=0, tracked_object_types=self.tracked_object_types)
        feature = self.feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), 0)
            self.assertEqual(feature.agents[feature_name][0].shape[1], 0)
            self.assertEqual(feature.agents[feature_name][0].shape[2], GenericAgents.agents_states_dim())

    def test_get_feature_from_simulation(self) -> None:
        """
        Test get feature from simulation
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents, tracked_object_types=self.tracked_object_types)
        mock_meta_data = PlannerInitialization(map_api=MockAbstractMap(), route_roadblock_ids=None, mission_goal=StateSE2(0, 0, 0))
        ego_past_states = list(scenario.get_ego_past_trajectory(iteration=0, num_samples=10, time_horizon=5))
        ego_initial_state = scenario.initial_ego_state
        ego_history = ego_past_states + [ego_initial_state]
        past_observations = list(scenario.get_past_tracked_objects(iteration=0, num_samples=10, time_horizon=5))
        initial_observation = scenario.initial_tracked_objects
        observation_history = past_observations + [initial_observation]
        history = SimulationHistoryBuffer.initialize_from_list(len(ego_history), ego_history, observation_history, scenario.database_interval)
        current_input = PlannerInput(iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history, traffic_light_data=scenario.get_traffic_light_status_at_iteration(0))
        feature = self.feature_builder.get_features_from_simulation(current_input=current_input, initialization=mock_meta_data)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_agents_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the Generic Agents Feature Builder scripts properly
        """
        config = self.feature_builder.precomputed_feature_config()
        for expected_key in ['past_ego_states', 'past_time_stamps']:
            self.assertTrue(expected_key in config)
            config_dict = config[expected_key]
            self.assertTrue(len(config_dict) == 3)
            self.assertEqual(0, int(config_dict['iteration']))
            self.assertEqual(self.num_past_poses, int(config_dict['num_samples']))
            self.assertEqual(self.past_time_horizon, int(float(config_dict['time_horizon'])))
        tracked_objects_config_dict = config['past_tracked_objects']
        self.assertTrue(len(tracked_objects_config_dict) == 4)
        self.assertEqual(0, int(tracked_objects_config_dict['iteration']))
        self.assertEqual(self.num_past_poses, int(tracked_objects_config_dict['num_samples']))
        self.assertEqual(self.past_time_horizon, int(float(tracked_objects_config_dict['time_horizon'])))
        self.assertTrue('agent_features' in tracked_objects_config_dict)
        self.assertEqual(','.join(self.agent_features), tracked_objects_config_dict['agent_features'])
        num_frames = 5
        num_agents = 3
        ego_dim = EgoInternalIndex.dim()
        agent_dim = AgentInternalIndex.dim()
        past_ego_states = torch.zeros((num_frames, ego_dim), dtype=torch.float32)
        past_timestamps = torch.tensor([i * 50 for i in range(num_frames)], dtype=torch.int64)
        past_tracked_objects = [torch.ones((num_agents, agent_dim), dtype=torch.float32) for _ in range(num_frames)]
        for i in range(num_frames):
            for j in range(num_agents):
                past_tracked_objects[i][j, :] *= j + 1
        tensor_data = {'past_ego_states': past_ego_states, 'past_time_stamps': past_timestamps}
        list_tensor_data = {f'past_tracked_objects.{feature_name}': past_tracked_objects for feature_name in self.agent_features}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_builder = torch.jit.script(self.feature_builder)
        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        py_tensors, py_list_tensors, py_list_list_tensors = self.feature_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        self.assertEqual(0, len(scripted_tensors))
        self.assertEqual(0, len(py_tensors))
        self.assertEqual(len(scripted_list_tensors), len(py_list_tensors))
        for key in py_list_tensors:
            scripted_list = scripted_list_tensors[key]
            py_list = py_list_tensors[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                scripted = scripted_list[i]
                py = py_list[i]
                torch.testing.assert_allclose(py, scripted, atol=0.05, rtol=0.05)
        self.assertEqual(0, len(scripted_list_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))

class MockAbstractScenario(AbstractScenario):
    """Mock abstract scenario class used for testing."""

    def __init__(self, initial_time_us: TimePoint=TimePoint(time_us=1621641671099), time_step: float=0.5, number_of_future_iterations: int=10, number_of_past_iterations: int=0, initial_velocity: StateVector2D=StateVector2D(x=1.0, y=0.0), fixed_acceleration: StateVector2D=StateVector2D(x=0.0, y=0.0), number_of_detections: int=10, initial_ego_state: StateSE2=StateSE2(x=0.0, y=0.0, heading=0.0), mission_goal: StateSE2=StateSE2(10, 0, 0), tracked_object_types: List[TrackedObjectType]=[TrackedObjectType.VEHICLE]):
        """
        Create mocked scenario where ego starts with an initial velocity [m/s] and has a constant acceleration
            throughout (0 m/s^2 by default). The ego does not turn.
        :param initial_time_us: initial time from start point of scenario [us]
        :param time_step: time step in [s]
        :param number_of_future_iterations: number of iterations in the future
        :param number_of_past_iterations: number of iterations in the past
        :param initial_velocity: [m/s] velocity assigned to the ego at iteration 0
        :param fixed_acceleration: [m/s^2] constant ego acceleration throughout scenario
        :param number_of_detections: number of detections in the scenario
        :param initial_ego_state: Initial state of ego
        :param mission_goal: Dummy mission goal
        :param tracked_object_types: Types of tracked objects to mock
        """
        self._initial_time_us = initial_time_us
        self._time_step = time_step
        self._number_of_past_iterations = number_of_past_iterations
        self._number_of_future_iterations = number_of_future_iterations
        self._current_iteration = number_of_past_iterations
        self._total_iterations = number_of_past_iterations + number_of_future_iterations + 1
        self._tracked_object_types = tracked_object_types
        start_time_us = max(TimePoint(int(number_of_past_iterations * time_step * 1000000.0)), initial_time_us)
        time_horizon = (number_of_past_iterations + number_of_future_iterations) * time_step
        history_buffer = SimulationHistoryBuffer.initialize_from_list(buffer_size=10, ego_states=[EgoState.build_from_rear_axle(StateSE2(x=initial_ego_state.x, y=initial_ego_state.y, heading=initial_ego_state.heading), time_point=start_time_us, rear_axle_velocity_2d=initial_velocity, tire_steering_angle=0.0, rear_axle_acceleration_2d=fixed_acceleration, vehicle_parameters=self.ego_vehicle_parameters)], observations=[DetectionsTracks(TrackedObjects())], sample_interval=time_step)
        planner_input = PlannerInput(iteration=SimulationIteration(start_time_us, 0), history=history_buffer)
        planner = SimplePlanner(horizon_seconds=time_horizon, sampling_time=time_step, acceleration=fixed_acceleration.array)
        self._ego_states = planner.compute_trajectory(planner_input).get_sampled_trajectory()
        self._tracked_objects = [DetectionsTracks(TrackedObjects([get_sample_agent(token=str(idx + type_idx * number_of_detections), agent_type=agent_type, num_future_states=0) for idx in range(number_of_detections) for type_idx, agent_type in enumerate(self._tracked_object_types)])) for _ in range(self._total_iterations)]
        self._sensors = [Sensors(pointcloud={LidarChannel.MERGED_PC: np.eye(3) for _ in range(number_of_detections)}, images=None) for _ in range(self._total_iterations)]
        if len(self._ego_states) != len(self._tracked_objects) or len(self._ego_states) != self._total_iterations:
            raise RuntimeError('The dimensions of detections and ego trajectory is not the same!')
        self._mission_goal = mission_goal
        self._map_api = MockAbstractMap()
        self._token_suffix = str(uuid.uuid4())

    @property
    def token(self) -> str:
        """Implemented. See interface."""
        return f'mock_token_{self._token_suffix}'

    @property
    def log_name(self) -> str:
        """Implemented. See interface."""
        return 'mock_log_name'

    @property
    def scenario_name(self) -> str:
        """Implemented. See interface."""
        return 'mock_scenario_name'

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return get_pacifica_parameters()

    @property
    def scenario_type(self) -> str:
        """Implemented. See interface."""
        return 'mock_scenario_type'

    @property
    def map_api(self) -> AbstractMap:
        """Implemented. See interface."""
        return self._map_api

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        return self._time_step

    def get_number_of_iterations(self) -> int:
        """Implemented. See interface."""
        return self._number_of_future_iterations

    def get_time_point(self, iteration: int) -> TimePoint:
        """Implemented. See interface."""
        return self._ego_states[self._current_iteration + iteration].time_point

    def get_lidar_to_ego_transform(self) -> Transform:
        """Implemented. See interface."""
        return np.eye(4)

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Implemented. See interface."""
        return self._mission_goal

    def get_route_roadblock_ids(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def get_expert_goal_state(self) -> StateSE2:
        """Implemented. See interface."""
        return self._mission_goal

    def get_tracked_objects_at_iteration(self, iteration: int, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Implemented. See interface."""
        return self._tracked_objects[self._current_iteration + iteration]

    def get_tracked_objects_within_time_window_at_iteration(self, iteration: int, past_time_horizon: float, future_time_horizon: float, filter_track_tokens: Optional[Set[str]]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Implemented. See interface."""
        raise NotImplementedError

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]]=None) -> Sensors:
        """Implemented. See interface."""
        raise NotImplementedError

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Implemented. See interface."""
        return self._ego_states[self._current_iteration + iteration]

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Implemented. see interface."""
        dummy_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=1627066061949808)
        yield dummy_data

    def get_past_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """Gets past traffic light status."""
        dummy_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=1627066061949808)
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        for _ in range(num_samples):
            yield TrafficLightStatuses([dummy_data])

    def get_future_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """Gets future traffic light status."""
        dummy_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=1627066061949808)
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        for _ in range(num_samples):
            yield TrafficLightStatuses([dummy_data])

    def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Implemented. See interface."""
        ego_states = self.get_ego_future_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
        for state in ego_states:
            yield state.time_point

    def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Implemented. See interface."""
        ego_states = self.get_ego_past_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
        for state in ego_states:
            yield state.time_point

    def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Implemented. See interface."""
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._number_of_future_iterations - iteration >= indices[-1], f'Requested time horizon of {time_horizon}s is too long! Scenario future has length {(self._number_of_future_iterations - iteration) * self._time_step}s from the iteration {iteration}'
        for idx in indices:
            yield self._ego_states[self._current_iteration + iteration + idx]

    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Implemented. See interface."""
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._current_iteration + iteration >= indices[-1], f'Requested time horizon of {time_horizon}s is too long! Scenario past has length {(self._current_iteration + iteration) * self._time_step}s from the iteration {iteration}'
        for idx in reversed(indices):
            yield self._ego_states[self._current_iteration + iteration - idx]

    def get_past_sensors(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, channels: Optional[List[SensorChannel]]=None) -> Generator[Sensors, None, None]:
        """Implemented. See interface."""
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        for idx in indices:
            yield self._sensors[self._current_iteration + iteration - idx - 1]

    def get_past_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Implemented. See interface."""
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        if self._current_iteration + iteration < indices[-1]:
            raise ValueError(f'Requested time horizon of {time_horizon}s is too long! Scenario past has length {(self._current_iteration + iteration) * self._time_step}s from the iteration {iteration}')
        for idx in reversed(indices):
            yield self._tracked_objects[self._current_iteration + iteration - idx]

    def get_future_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Implemented. See interface."""
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._number_of_future_iterations - iteration >= indices[-1], f'Requested time horizon of {time_horizon}s is too long! Scenario future has length {(self._number_of_future_iterations - iteration) * self._time_step}s from the iteration {iteration}'
        for idx in indices:
            yield self._tracked_objects[self._current_iteration + iteration + idx]

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

class DetectionTracksChallengeServicer(chpb_grpc.DetectionTracksChallengeServicer):
    """
    Servicer for exposing initialization and trajectory computation services to the client.
    It keeps a rolling history buffer to avoid unnecessary serialization/deserialization.
    """

    def __init__(self, planner_config: DictConfig, map_manager: MapManager):
        """
        :param planner_config: The planner configuration to instantiate the planner.
        :param map_manager: The map manager.
        """
        self.planner: Optional[AbstractPlanner] = None
        self._planner_config = planner_config
        self.map_manager = map_manager
        self.simulation_history_buffer: Optional[SimulationHistoryBuffer] = None
        self._initialized = False

    @staticmethod
    def _extract_simulation_iteration(planner_input_message: chpb.PlannerInput) -> SimulationIteration:
        return SimulationIteration(TimePoint(planner_input_message.simulation_iteration.time_us), planner_input_message.simulation_iteration.index)

    def _build_planner_input(self, planner_input_message: chpb.PlannerInput, buffer: Optional[SimulationHistoryBuffer]) -> PlannerInput:
        """
        Builds a PlannerInput from a serialized PlannerInput message and an existing data buffer
        :param planner_input_message: the serialized message
        :param buffer: The history buffer
        :return: PlannerInput object
        """
        simulation_iteration = self._extract_simulation_iteration(planner_input_message)
        new_data = planner_input_message.simulation_history_buffer
        states = []
        observations = []
        for serialized_state, serialized_observation in zip(new_data.ego_states, new_data.observations):
            states.append(pickle.loads(serialized_state))
            observations.append(pickle.loads(serialized_observation))
        if buffer is not None:
            buffer.extend(states, observations)
        else:
            buffer = SimulationHistoryBuffer.initialize_from_list(len(states), states, observations, new_data.sample_interval)
            self.simulation_history_buffer = buffer
        tl_data_messages = planner_input_message.traffic_light_data
        tl_data = [tl_status_data_from_proto_tl_status_data(tl_data_message) for tl_data_message in tl_data_messages]
        return PlannerInput(iteration=simulation_iteration, history=buffer, traffic_light_data=tl_data)

    def InitializePlanner(self, planner_initialization_message: chpb.PlannerInitializationLight, context: Any) -> chpb.Empty:
        """
        Service to initialize the planner given the initialization request.
        :param planner_initialization_message: Message containing initialization details
        :param context
        """
        planners = build_planners(self._planner_config, None)
        assert len(planners) == 1, f'Configuration should build exactly 1 planner, got {len(planners)} instead!'
        self.planner = planners[0]
        logger.info('Initialization request received..')
        route_roadblock_ids = planner_initialization_message.route_roadblock_ids
        mission_goal = se2_from_proto_se2(planner_initialization_message.mission_goal)
        map_api = self.map_manager.get_map(planner_initialization_message.map_name)
        map_api.initialize_all_layers()
        planner_initialization = PlannerInitialization(route_roadblock_ids=route_roadblock_ids, mission_goal=mission_goal, map_api=map_api)
        self.simulation_history_buffer = None
        self.planner.initialize(planner_initialization)
        logging.info('Planner initialized!')
        self._initialized = True
        return chpb.Empty()

    def ComputeTrajectory(self, planner_input_message: chpb.PlannerInput, context: Any) -> chpb.Trajectory:
        """
        Service to compute a trajectory given a planner input message
        :param planner_input_message: Message containing the input to the planner
        :param context
        :return Message containing the computed trajectories
        """
        assert self._initialized, 'Planner has not been initialized. Please call InitializePlanner'
        planner_inputs = self._build_planner_input(planner_input_message, self.simulation_history_buffer)
        if isinstance(self.planner, AbstractPlanner):
            trajectory = self.planner.compute_trajectory(planner_inputs)
            return proto_traj_from_inter_traj(trajectory)
        raise RuntimeError('The planner was not initialized correctly!')

class SubmissionComputesTrajectoryValidator(BaseSubmissionValidator):
    """Checks if a submission is able to compute a trajectory"""

    def validate(self, submission: str) -> bool:
        """
        Checks if the provided submission is able to provide a trajectory given an input.
        :param submission: The submission image
        :return: Whether the submission is able to compute a trajectory
        """
        logger.info('validating trajectory computation')
        scenario = get_test_nuplan_scenario()
        step = 0
        iteration = SimulationIteration(time_point=scenario.get_time_point(0), index=0)
        history = SimulationHistoryBuffer.initialize_from_list(1, [scenario.get_ego_state_at_iteration(step)], [scenario.get_tracked_objects_at_iteration(step)], scenario.database_interval)
        planner_input = PlannerInput(iteration, history)
        container_name = container_name_from_image_name(submission)
        container_manager = SubmissionContainerManager(SubmissionContainerFactory())
        planner = RemotePlanner(container_manager, submission, container_name)
        planner_initialization = PlannerInitialization(scenario.get_mission_goal(), scenario.get_route_roadblock_ids(), scenario.map_api)
        planner.initialize([planner_initialization])
        trajectory = planner.compute_trajectory([planner_input])
        if trajectory:
            logger.debug(f'Computed trajectory {trajectory}')
            return True
        logger.error('Submission failed to compute trajectory')
        self._failing_validator = SubmissionComputesTrajectoryValidator
        return False

class TestSubmissionContainerManager(unittest.TestCase):
    """Tests for SubmissionContainerManager class"""

    @patch('nuplan.submission.submission_container_manager.SubmissionContainerFactory')
    def setUp(self, mock_container_factory: Mock) -> None:
        """Sets variables for testing"""
        self.container_manager = SubmissionContainerManager(mock_container_factory)

    @patch('nuplan.submission.submission_container_manager.SubmissionContainerFactory')
    def test_initialization(self, mock_container_factory: Mock) -> None:
        """Tests that objects are initialized correctly."""
        submission_container_manager = SubmissionContainerManager(mock_container_factory)
        self.assertEqual(mock_container_factory, submission_container_manager.submission_container_factory)
        self.assertEqual({}, submission_container_manager.submission_containers)

    def test_get_submission_container(self) -> None:
        """Tests that maps are retrieved from cache, if not present created and added to it."""
        image_name = 'image_name'
        container_name = 'container_name'
        port = 123
        self.container_manager.submission_container_factory.build_submission_container.return_value = 'container'
        _container = self.container_manager.get_submission_container(image_name, container_name, port)
        self.container_manager.submission_container_factory.build_submission_container.assert_called_once_with(image_name, container_name, port)
        self.assertTrue(container_name in self.container_manager.submission_containers)
        self.assertEqual('container', _container)
        _ = self.container_manager.get_submission_container(image_name, container_name, port)
        self.container_manager.submission_container_factory.build_submission_container.assert_called_once_with(image_name, container_name, port)

