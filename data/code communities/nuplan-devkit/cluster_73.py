# Cluster 73

class TestTimeDuration(unittest.TestCase):
    """Tests for TimeDurationClass"""

    def test_default_initialization(self) -> None:
        """Checks raising when constructor is called directly unless flagged."""
        with self.assertRaises(RuntimeError):
            _ = TimeDuration(time_us=42)
        dt = TimeDuration(time_us=42, _direct=False)
        self.assertEqual(dt.time_us, 42)

    def test_constructors(self) -> None:
        """Checks constructors perform correct conversions"""
        dt_s = TimeDuration.from_s(42)
        dt_ms = TimeDuration.from_ms(42)
        dt_us = TimeDuration.from_us(42)
        self.assertEqual(dt_s.time_us, 42000000)
        self.assertEqual(dt_ms.time_us, 42000)
        self.assertEqual(dt_us.time_us, 42)

    def test_getters(self) -> None:
        """Checks getters work as intended"""
        dt = TimeDuration.from_s(42)
        value_s = dt.time_s
        value_ms = dt.time_ms
        value_us = dt.time_us
        self.assertEqual(value_s, 42)
        self.assertEqual(value_ms, 42000)
        self.assertEqual(value_us, 42000000)

    def test_operators(self) -> None:
        """Tests basic math operators."""
        t1 = TimeDuration.from_s(1)
        t2 = TimeDuration.from_s(2)
        self.assertTrue(t2 > t1)
        self.assertFalse(t2 < t1)
        self.assertTrue(t1 < t2)
        self.assertFalse(t1 > t2)
        self.assertTrue(t1 == t1)
        self.assertFalse(t1 == t2)
        self.assertTrue(t1 >= t1)
        self.assertTrue(t1 <= t1)
        self.assertEqual((t1 + t2).time_s, 3)
        self.assertEqual((t1 - t2).time_s, -1)
        self.assertEqual((t1 * 3).time_s, 3)
        self.assertEqual((3 * t1).time_s, 3)
        self.assertEqual((t2 / 2).time_s, 1)
        self.assertEqual((t2 // 3).time_s, 0)

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

class AbstractScenario(abc.ABC):
    """
    Interface for a generic scenarios from any database.
    """

    @property
    @abc.abstractmethod
    def token(self) -> str:
        """
        Unique identifier of a scenario
        :return: str representing unique token.
        """
        pass

    @property
    @abc.abstractmethod
    def log_name(self) -> str:
        """
        Log name for from which this scenario was created
        :return: str representing log name.
        """
        pass

    @property
    @abc.abstractmethod
    def scenario_name(self) -> str:
        """
        Name of this scenario, e.g. extraction_xxxx
        :return: str representing name of this scenario.
        """
        pass

    @property
    @abc.abstractmethod
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """
        Query the vehicle parameters of ego
        :return: VehicleParameters struct.
        """
        pass

    @property
    @abc.abstractmethod
    def scenario_type(self) -> str:
        """
        :return: type of scenario e.g. [lane_change, lane_follow, ...].
        """
        pass

    @property
    @abc.abstractmethod
    def map_api(self) -> AbstractMap:
        """
        Return the Map API for this scenario
        :return: AbstractMap.
        """
        pass

    @property
    @abc.abstractmethod
    def database_interval(self) -> float:
        """
        Database interval in seconds
        :return: [s] database interval.
        """
        pass

    @abc.abstractmethod
    def get_number_of_iterations(self) -> int:
        """
        Get how many frames does this scenario contain
        :return: [int] representing number of scenarios.
        """
        pass

    @abc.abstractmethod
    def get_time_point(self, iteration: int) -> TimePoint:
        """
        Get time point of the iteration
        :param iteration: iteration in scenario 0 <= iteration < number_of_iterations
        :return: global time point.
        """
        pass

    @property
    def start_time(self) -> TimePoint:
        """
        Get the start time of a scenario
        :return: starting time.
        """
        return self.get_time_point(0)

    @property
    def end_time(self) -> TimePoint:
        """
        Get end time of the scenario
        :return: end time point.
        """
        return self.get_time_point(self.get_number_of_iterations() - 1)

    @property
    def duration_s(self) -> TimeDuration:
        """
        Get the duration of the scenario in seconds
        :return: the difference in seconds between the scenario's final and first timepoints.
        """
        return TimeDuration.from_s(self.end_time.time_s - self.start_time.time_s)

    @abc.abstractmethod
    def get_lidar_to_ego_transform(self) -> Transform:
        """
        Return the transformation matrix between lidar and ego
        :return: [4x4] rotation and translation matrix.
        """
        pass

    @abc.abstractmethod
    def get_mission_goal(self) -> Optional[StateSE2]:
        """
        Goal far into future (in generally more than 100m far beyond scenario length).
        :return: StateSE2 for the final state.
        """
        pass

    @abc.abstractmethod
    def get_route_roadblock_ids(self) -> List[str]:
        """
        Get list of roadblock ids comprising goal route.
        :return: List of roadblock id strings.
        """
        pass

    @abc.abstractmethod
    def get_expert_goal_state(self) -> StateSE2:
        """
        Get the final state which the expert driver achieved at the end of the scenario
        :return: StateSE2 for the final state.
        """
        pass

    @abc.abstractmethod
    def get_tracked_objects_at_iteration(self, iteration: int, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """
        Return tracked objects from iteration
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :param future_trajectory_sampling: sampling parameters of agent future ground truth predictions if desired.
        :return: DetectionsTracks.
        """
        pass

    @abc.abstractmethod
    def get_tracked_objects_within_time_window_at_iteration(self, iteration: int, past_time_horizon: float, future_time_horizon: float, filter_track_tokens: Optional[Set[str]]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """
        Gets all tracked objects present within a time window that stretches from past_time_horizon before the iteration to future_time_horizon afterwards.
        Also optionally filters the included results on the provided track_tokens.
        Results will be sorted by object type, then by timestamp, then by track token.
        :param iteration: The iteration of the scenario to query.
        :param past_time_horizon [s]: The amount of time to look into the past from the iteration timestamp.
        :param future_time_horizon [s]: The amount of time to look into the future from the iteration timestamp.
        :param filter_track_tokens: If provided, then the results will be filtered to only contain objects with
            track_tokens included in the provided set. If None, then all results are returned.
        :param future_trajectory_sampling: sampling parameters of agent future ground truth predictions if desired.
        :return: The retrieved detection tracks.
        """
        pass

    @property
    def initial_tracked_objects(self) -> DetectionsTracks:
        """
        Get initial tracked objects
        :return: DetectionsTracks.
        """
        return self.get_tracked_objects_at_iteration(0)

    @abc.abstractmethod
    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]]=None) -> Sensors:
        """
        Return sensor from iteration
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :param channels: The sensor channels to return.
        :return: Sensors.
        """
        pass

    @property
    def initial_sensors(self) -> Sensors:
        """
        Return the initial sensors (e.g. pointcloud)
        :return: Sensors.
        """
        return self.get_sensors_at_iteration(0)

    @abc.abstractmethod
    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """
        Return ego (expert) state in a dataset
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: EgoState of ego.
        """
        pass

    @property
    def initial_ego_state(self) -> EgoState:
        """
        Return the initial ego state
        :return: EgoState of ego.
        """
        return self.get_ego_state_at_iteration(0)

    @abc.abstractmethod
    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """
        Get traffic light status at an iteration.
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return traffic light status at the iteration.
        """
        pass

    @abc.abstractmethod
    def get_past_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets past traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the past.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the past.
        """
        pass

    @abc.abstractmethod
    def get_future_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets future traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the future.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the future.
        """
        pass

    def get_expert_ego_trajectory(self) -> Generator[EgoState, None, None]:
        """
        Return trajectory that was taken by the expert-driver
        :return: sequence of agent states taken by ego.
        """
        return (self.get_ego_state_at_iteration(index) for index in range(self.get_number_of_iterations()))

    def get_ego_trajectory_slice(self, start_idx: int, end_idx: int) -> Generator[EgoState, None, None]:
        """
        Return trajectory that was taken by the expert-driver between start_idx and end_idx
        :param start_idx: starting index for ego's trajectory
        :param end_idx: ending index for ego's trajectory
        :return: sequence of agent states taken by ego
        timestamp (best matching to the database).
        """
        return (self.get_ego_state_at_iteration(index) for index in range(start_idx, end_idx))

    @abc.abstractmethod
    def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """
        Find timesteps in future
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon [s]: the desired horizon to the future
        :return: the future timestamps with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """
        Find timesteps in past
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the past
        :param time_horizon [s]: the desired horizon to the past
        :return: the future timestamps with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """
        Find ego future trajectory
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon [s]: the desired horizon to the future
        :return: the future ego trajectory with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """
        Find ego past trajectory
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon [s]: the desired horizon to the future
        :return: the past ego trajectory with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_past_sensors(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, channels: Optional[List[SensorChannel]]=None) -> Generator[Sensors, None, None]:
        """
        Find past sensors
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param time_horizon: [s] the desired horizon to the future
        :param num_samples: number of entries in the future
        :param channels: The sensor channels to return.
        :return: the past sensors with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_past_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """
        Find past detections.
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param num_samples: number of entries in the future.
        :param time_horizon [s]: the desired horizon to the future.
        :param future_trajectory_sampling: sampling parameters of agent future ground truth predictions if desired.
        :return: the past detections.
        """
        pass

    @abc.abstractmethod
    def get_future_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """
        Find future detections.
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param num_samples: number of entries in the future.
        :param time_horizon [s]: the desired horizon to the future.
        :param future_trajectory_sampling: sampling parameters of agent future ground truth predictions if desired.
        :return: the past detections.
        """
        pass

