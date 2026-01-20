# Cluster 14

class LidarBox(Base):
    """
    Lidar box from tracker.
    """
    __tablename__ = 'lidar_box'
    token: str = Column(sql_types.HexLen8, primary_key=True)
    lidar_pc_token: str = Column(sql_types.HexLen8, ForeignKey('lidar_pc.token'), nullable=False)
    track_token: str = Column(sql_types.HexLen8, ForeignKey('track.token'))
    next_token = Column(sql_types.HexLen8, ForeignKey('lidar_box.token'), nullable=True)
    prev_token = Column(sql_types.HexLen8, ForeignKey('lidar_box.token'), nullable=True)
    x: float = Column(Float)
    y: float = Column(Float)
    z: float = Column(Float)
    width: float = Column(Float)
    length: float = Column(Float)
    height: float = Column(Float)
    vx: float = Column(Float)
    vy: float = Column(Float)
    vz: float = Column(Float)
    yaw: float = Column(Float)
    confidence: float = Column(Float)
    next = relationship('LidarBox', foreign_keys=[next_token], remote_side=[token])
    prev = relationship('LidarBox', foreign_keys=[prev_token], remote_side=[token])

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __iter__(self) -> IterableLidarBox:
        """
        Returns a iterator object for LidarBox.
        :return: The iterator object.
        """
        return IterableLidarBox(self)

    def __reversed__(self) -> IterableLidarBox:
        """
        Returns a iterator object for LidarBox that traverses in reverse.
        :return: The iterator object.
        """
        return IterableLidarBox(self, reverse=True)

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def log(self) -> Log:
        """
        Returns the Log containing the LidarBox.
        :return: The log containing the lidar box.
        """
        return self.lidar_pc.log

    @property
    def category(self) -> Category:
        """
        Returns the Category of the LidarBox.
        :return: The category of the lidar box.
        """
        return self.track.category

    @property
    def timestamp(self) -> int:
        """
        Returns the timestamp of the LidarBox.
        :return: The timestamp of the lidar box.
        """
        return int(self.lidar_pc.timestamp)

    @property
    def distance_to_ego(self) -> float:
        """
        Returns the distance of detection from Ego Vehicle.
        :return: The distance to ego vehicle.
        """
        return float(np.sqrt((self.x - self.lidar_pc.ego_pose.x) ** 2 + (self.y - self.lidar_pc.ego_pose.y) ** 2))

    @property
    def size(self) -> List[float]:
        """
        Get the box size.
        :return: The box size.
        """
        return [self.width, self.length, self.height]

    @property
    def translation(self) -> List[float]:
        """
        Get the box location.
        :return: The box location.
        """
        return [self.x, self.y, self.z]

    @property
    def rotation(self) -> List[float]:
        """
        Get the box rotation in euler angles.
        :return: The box rotation in euler angles.
        """
        qx = Quaternion(axis=(1, 0, 0), radians=0.0)
        qy = Quaternion(axis=(0, 1, 0), radians=0.0)
        qz = Quaternion(axis=(0, 0, 1), radians=self.yaw)
        return list(qx * qy * qz)

    @property
    def quaternion(self) -> Quaternion:
        """
        Get the box rotation in quaternion.
        :return: The box rotation in quaternion.
        """
        return Quaternion(self.rotation)

    @property
    def translation_np(self) -> npt.NDArray[np.float64]:
        """
        Get the box translation in numpy.
        :return: <np.float: 3> Translation.
        """
        return np.array(self.translation)

    @property
    def size_np(self) -> npt.NDArray[np.float64]:
        """
        Get the box size in numpy.
        :return: <np.float, 3> Width, length and height.
        """
        return np.array(self.size)

    @cached(cache=LRUCache(maxsize=LIDAR_BOX_LRU_CACHE_SIZE), key=lambda self: hashkey(self.track_token))
    def _get_box_items(self) -> Tuple[List[Integer], List[LidarBox]]:
        """
        Get all boxes along the track.
        :return: The list of timestamps and boxes along the track.
        """
        box_list: List[LidarBox] = self._session.query(LidarBox).filter(LidarBox.track_token == self.track_token).all()
        sorted_box_list = sorted(box_list, key=lambda x: x.timestamp)
        return ([b.timestamp for b in sorted_box_list], sorted_box_list)

    @cached(cache=LRUCache(maxsize=LIDAR_BOX_LRU_CACHE_SIZE), key=lambda self: hashkey(self.track_token))
    def get_box_items_to_iterate(self) -> Dict[int, Tuple[Optional[LidarBox], Optional[LidarBox]]]:
        """
        Get all boxes along the track.
        :return: Dict. Key is timestamp of box, value is Tuple of (prev,next) LidarBox.
        """
        box_list = self._session.query(LidarBox).filter(LidarBox.track_token == self.track_token).all()
        sorted_box_list = sorted(box_list, key=lambda x: x.timestamp)
        return {box.timestamp: (prev, next) for box, prev, next in zip(sorted_box_list, [None] + sorted_box_list[:-1], sorted_box_list[1:] + [None])}

    def _find_box(self, step: int=0) -> Optional[LidarBox]:
        """
        Find the next box along the track with the given step.
        :param: step: The number of steps to look ahead, defaults to zero.
        :return: The found box if any.
        """
        timestamp_list, sorted_box_list = self._get_box_items()
        i = bisect.bisect_left(timestamp_list, self.timestamp)
        j = i + step
        if j < 0 or j >= len(sorted_box_list):
            return None
        return sorted_box_list[j]

    def future_or_past_ego_poses(self, number: int, mode: str, direction: str) -> List[EgoPose]:
        """
        Get n future or past vehicle poses. Note here the frequency of pose differs from frequency of LidarBox.
        :param number: Number of poses to fetch or number of seconds of ego poses to fetch.
        :param mode: Either n_poses or n_seconds.
        :param direction: Future or past ego poses to fetch, could be 'prev' or 'next'.
        :return: List of up to n or n seconds future or past ego poses.
        """
        if direction == 'prev':
            if mode == 'n_poses':
                return self._session.query(EgoPose).filter(EgoPose.timestamp < self.lidar_pc.ego_pose.timestamp, self.lidar_pc.lidar.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.desc()).limit(number).all()
            elif mode == 'n_seconds':
                return self._session.query(EgoPose).filter(EgoPose.timestamp - self.lidar_pc.ego_pose.timestamp < 0, EgoPose.timestamp - self.lidar_pc.ego_pose.timestamp >= -number * 1000000.0, self.lidar_pc.lidar.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.desc()).all()
            else:
                raise ValueError(f'Unknown mode: {mode}.')
        elif direction == 'next':
            if mode == 'n_poses':
                return self._session.query(EgoPose).filter(EgoPose.timestamp > self.lidar_pc.ego_pose.timestamp, self.lidar_pc.lidar.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.asc()).limit(number).all()
            elif mode == 'n_seconds':
                return self._session.query(EgoPose).filter(EgoPose.timestamp - self.lidar_pc.ego_pose.timestamp > 0, EgoPose.timestamp - self.lidar_pc.ego_pose.timestamp <= number * 1000000.0, self.lidar_pc.lidar.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.asc()).all()
            else:
                raise ValueError(f'Unknown mode: {mode}.')
        else:
            raise ValueError(f'Unknown direction: {direction}.')

    def _temporal_neighbors(self) -> Tuple[LidarBox, LidarBox, bool, bool]:
        """
        Find temporal neighbors to calculate velocity and angular velocity.
        :return: The previous box, next box and their existences. If the previous or next box do not exist, they will
            be set to the current box itself.
        """
        has_prev = self.prev is not None
        has_next = self.next is not None
        if has_prev:
            prev_lidar_box = self.prev
        else:
            prev_lidar_box = self
        if has_next:
            next_lidar_box = self.next
        else:
            next_lidar_box = self
        return (prev_lidar_box, next_lidar_box, has_prev, has_next)

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        """
        Estimate box velocity for a box.
        :return: The estimated box velocity of the box.
        """
        max_time_diff = 1.5
        prev_lidar_box, next_lidar_box, has_prev, has_next = self._temporal_neighbors()
        if not has_prev and (not has_next):
            return np.array([np.nan, np.nan, np.nan])
        pos_next: npt.NDArray[np.float64] = np.array(next_lidar_box.translation)
        pos_prev: npt.NDArray[np.float64] = np.array(prev_lidar_box.translation)
        pos_diff: npt.NDArray[np.float64] = pos_next - pos_prev
        pos_diff[2] = 0
        time_next = 1e-06 * next_lidar_box.timestamp
        time_prev = 1e-06 * prev_lidar_box.timestamp
        time_diff = time_next - time_prev
        if has_next and has_prev:
            max_time_diff *= 2
        if time_diff > max_time_diff:
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    @property
    def angular_velocity(self) -> float:
        """
        Estimate box angular velocity for a box.
        :return: The estimated box angular velocity of the box.
        """
        max_time_diff = 1.5
        prev_lidar_box, next_lidar_box, has_prev, has_next = self._temporal_neighbors()
        if not has_prev and (not has_next):
            return np.nan
        time_next = 1e-06 * next_lidar_box.timestamp
        time_prev = 1e-06 * prev_lidar_box.timestamp
        time_diff = time_next - time_prev
        if has_next and has_prev:
            max_time_diff *= 2
        if time_diff > max_time_diff:
            return np.nan
        else:
            yaw_diff = next_lidar_box.yaw - prev_lidar_box.yaw
            if yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            elif yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi
            return float(yaw_diff / time_diff)

    def box(self) -> Box3D:
        """
        Get the Box3D representation of the box.
        :return: The box3d representation of the box.
        """
        label_local = raw_mapping['global2local'][self.category.name]
        label_int = raw_mapping['local2id'][label_local]
        return Box3D(center=self.translation, size=self.size, orientation=self.quaternion, token=self.token, label=label_int, track_token=self.track_token)

    def tracked_object(self, future_waypoints: Optional[List[Waypoint]]) -> TrackedObject:
        """
        Creates an Agent object
        :param future_waypoints: Optional future poses, which will be used as predicted trajectory
        """
        pose = StateSE2(self.translation[0], self.translation[1], self.yaw)
        oriented_box = OrientedBox(pose, width=self.size[0], length=self.size[1], height=self.size[2])
        label_local = raw_mapping['global2local'][self.category.name]
        tracked_object_type = TrackedObjectType[local2agent_type[label_local]]
        if tracked_object_type in AGENT_TYPES:
            return Agent(tracked_object_type=tracked_object_type, oriented_box=oriented_box, velocity=StateVector2D(self.vx, self.vy), predictions=[PredictedTrajectory(1.0, future_waypoints)] if future_waypoints else [], angular_velocity=np.nan, metadata=SceneObjectMetadata(token=self.token, track_token=self.track_token, track_id=None, timestamp_us=self.timestamp, category_name=self.category.name))
        else:
            return StaticObject(tracked_object_type=tracked_object_type, oriented_box=oriented_box, metadata=SceneObjectMetadata(token=self.token, track_token=self.track_token, track_id=None, timestamp_us=self.timestamp, category_name=self.category.name))

def _waypoint_from_lidar_box(lidar_box: LidarBox) -> Waypoint:
    """
    Creates a Waypoint from a LidarBox
    :param lidar_box: the input LidarBox
    :return: the corresponding Waypoint
    """
    pose = StateSE2(lidar_box.translation[0], lidar_box.translation[1], lidar_box.yaw)
    oriented_box = OrientedBox(pose, width=lidar_box.size[0], length=lidar_box.size[1], height=lidar_box.size[2])
    velocity = StateVector2D(lidar_box.vx, lidar_box.vy)
    waypoint = Waypoint(TimePoint(lidar_box.timestamp), oriented_box, velocity)
    return waypoint

def get_waypoints_for_agent(agent_box: LidarBox, end_timestamp: int) -> List[Waypoint]:
    """
    Extracts waypoints from a LidarBox by looking into the future samples
    :param agent_box: The first LidarBox
    :param end_timestamp: The maximal timestamp, used to stop extraction
    :return: Waypoints of the agent up to end_timestamp
    """
    agent_waypoints: List[Waypoint] = []
    tolerance_us = 60000
    while agent_box.timestamp <= end_timestamp + tolerance_us:
        agent_waypoints.append(_waypoint_from_lidar_box(agent_box))
        agent_box = agent_box.next
        if agent_box is None:
            break
    return agent_waypoints

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

def _parse_tracked_object_row(row: sqlite3.Row) -> TrackedObject:
    """
    A convenience method to parse a TrackedObject from a sqlite3 row.
    :param row: The row from the DB query.
    :return: The parsed TrackedObject.
    """
    category_name = row['category_name']
    pose = StateSE2(row['x'], row['y'], row['yaw'])
    oriented_box = OrientedBox(pose, width=row['width'], length=row['length'], height=row['height'])
    label_local = raw_mapping['global2local'][category_name]
    tracked_object_type = TrackedObjectType[local2agent_type[label_local]]
    if tracked_object_type in AGENT_TYPES:
        return Agent(tracked_object_type=tracked_object_type, oriented_box=oriented_box, velocity=StateVector2D(row['vx'], row['vy']), predictions=[], angular_velocity=np.nan, metadata=SceneObjectMetadata(token=row['token'].hex(), track_token=row['track_token'].hex(), track_id=get_unique_incremental_track_id(str(row['track_token'].hex())), timestamp_us=row['timestamp'], category_name=category_name))
    else:
        return StaticObject(tracked_object_type=tracked_object_type, oriented_box=oriented_box, metadata=SceneObjectMetadata(token=row['token'].hex(), track_token=row['track_token'].hex(), track_id=get_unique_incremental_track_id(str(row['track_token'].hex())), timestamp_us=row['timestamp'], category_name=category_name))

@functools.lru_cache(maxsize=None)
@static_vars(id=-1)
def get_unique_incremental_track_id(_: str) -> int:
    """
    Generate a unique ID (increasing number)
    :return int Unique ID
    """
    get_unique_incremental_track_id.id += 1
    return get_unique_incremental_track_id.id

def static_vars(**kwargs: Any) -> GenericCallable:
    """
    Decorator to assign static variables to functions
    """

    def decorate(func: GenericCallable) -> GenericCallable:
        for key, value in kwargs.items():
            setattr(func, key, value)
        return func
    return decorate

class TestGetUniqueJobId(unittest.TestCase):
    """Test suite for the generation of unique job IDs"""

    def test_unique_job_id_no_job_id(self) -> None:
        """
        Tests that if NUPLAN_JOB_ID is not set, the same job ID is returned.
        """
        if os.environ.get('NUPLAN_JOB_ID') is not None:
            del os.environ['NUPLAN_JOB_ID']
        get_unique_job_id.cache_clear()
        job_id_1 = get_unique_job_id()
        job_id_2 = get_unique_job_id()
        self.assertEqual(job_id_1, job_id_2)

    def test_unique_job_id_with_job_id(self) -> None:
        """
        Tests that if NUPLAN_JOB_ID is not set, the same job ID is returned.
        """
        os.environ['NUPLAN_JOB_ID'] = '12345'
        get_unique_job_id.cache_clear()
        job_id_1 = get_unique_job_id()
        job_id_2 = get_unique_job_id()
        self.assertEqual(job_id_1, job_id_2)
        del os.environ['NUPLAN_JOB_ID']

    def test_uniqueness_job_id(self) -> None:
        """
        Tests that the returned job ids are unique if they are not cached.
        """
        if os.environ.get('NUPLAN_JOB_ID') is not None:
            del os.environ['NUPLAN_JOB_ID']
        job_id_1 = get_unique_job_id()
        get_unique_job_id.cache_clear()
        job_id_2 = get_unique_job_id()
        self.assertNotEqual(job_id_1, job_id_2)

    def test_get_unique_incremental_track_id(self) -> None:
        """Tests creation of unique track ids."""
        track_token_and_expected_id_0 = ('track_0', 0)
        track_token_and_expected_id_1 = ('track_1', 1)
        self.assertEqual(get_unique_incremental_track_id(track_token_and_expected_id_0[0]), track_token_and_expected_id_0[1])
        self.assertEqual(get_unique_incremental_track_id(track_token_and_expected_id_1[0]), track_token_and_expected_id_1[1])
        self.assertEqual(get_unique_incremental_track_id(track_token_and_expected_id_0[0]), track_token_and_expected_id_0[1])

class TrackedObjects:
    """Class representing tracked objects, a collection of SceneObjects"""

    def __init__(self, tracked_objects: Optional[List[TrackedObject]]=None):
        """
        :param tracked_objects: List of tracked objects
        """
        tracked_objects = tracked_objects if tracked_objects is not None else []
        self.tracked_objects = sorted(tracked_objects, key=lambda agent: agent.tracked_object_type.value)

    def __iter__(self) -> Iterable[TrackedObject]:
        """When iterating return the tracked objects."""
        return iter(self.tracked_objects)

    @classmethod
    def from_oriented_boxes(cls, boxes: List[OrientedBox]) -> TrackedObjects:
        """When iterating return the tracked objects."""
        scene_objects = [SceneObject(TrackedObjectType.GENERIC_OBJECT, box, SceneObjectMetadata(timestamp_us=i, token=str(i), track_token=None, track_id=None)) for i, box in enumerate(boxes)]
        return TrackedObjects(scene_objects)

    @cached_property
    def _ranges_per_type(self) -> Dict[TrackedObjectType, Tuple[int, int]]:
        """
        Returns the start and end index of the range of agents for each agent type
        in the list of agents (sorted by agent type). The ranges are cached for subsequent calls.
        """
        ranges_per_type: Dict[TrackedObjectType, Tuple[int, int]] = {}
        if self.tracked_objects:
            last_agent_type = self.tracked_objects[0].tracked_object_type
            start_range = 0
            end_range = len(self.tracked_objects)
            for idx, agent in enumerate(self.tracked_objects):
                if agent.tracked_object_type is not last_agent_type:
                    ranges_per_type[last_agent_type] = (start_range, idx)
                    start_range = idx
                    last_agent_type = agent.tracked_object_type
            ranges_per_type[last_agent_type] = (start_range, end_range)
            ranges_per_type.update({agent_type: (end_range, end_range) for agent_type in TrackedObjectType if agent_type not in ranges_per_type})
        return ranges_per_type

    def get_tracked_objects_of_type(self, tracked_object_type: TrackedObjectType) -> List[TrackedObject]:
        """
        Gets the sublist of agents of a particular TrackedObjectType
        :param tracked_object_type: The query TrackedObjectType
        :return: List of the present agents of the query type. Throws an error if the key is invalid.
        """
        if tracked_object_type in self._ranges_per_type:
            start_idx, end_idx = self._ranges_per_type[tracked_object_type]
            return self.tracked_objects[start_idx:end_idx]
        else:
            return []

    def get_agents(self) -> List[Agent]:
        """
        Getter for the tracked objects which are Agents
        :return: list of Agents
        """
        agents = []
        for agent_type in AGENT_TYPES:
            agents.extend(self.get_tracked_objects_of_type(agent_type))
        return agents

    def get_static_objects(self) -> List[StaticObject]:
        """
        Getter for the tracked objects which are StaticObjects
        :return: list of StaticObjects
        """
        static_objects = []
        for static_object_type in STATIC_OBJECT_TYPES:
            static_objects.extend(self.get_tracked_objects_of_type(static_object_type))
        return static_objects

    def __len__(self) -> int:
        """
        :return: The number of tracked objects in the class
        """
        return len(self.tracked_objects)

    def get_tracked_objects_of_types(self, tracked_object_types: List[TrackedObjectType]) -> List[TrackedObject]:
        """
        Gets the sublist of agents of particular TrackedObjectTypes
        :param tracked_object_types: The query TrackedObjectTypes
        :return: List of the present agents of the query types. Throws an error if the key is invalid.
        """
        open_loop_tracked_objects = []
        for _type in tracked_object_types:
            open_loop_tracked_objects.extend(self.get_tracked_objects_of_type(_type))
        return open_loop_tracked_objects

def get_velocity_shifted(displacement: StateVector2D, ref_velocity: StateVector2D, ref_angular_vel: float) -> StateVector2D:
    """
    Computes the velocity at a query point on the same planar rigid body as a reference point.
    :param displacement: [m] The displacement vector from the reference to the query point
    :param ref_velocity: [m/s] The velocity vector at the reference point
    :param ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis
    :return: [m/s] The velocity vector at the given displacement.
    """
    velocity_shift_term: npt.NDArray[np.float64] = np.array([-displacement.y * ref_angular_vel, displacement.x * ref_angular_vel])
    return StateVector2D(*ref_velocity.array + velocity_shift_term)

def get_acceleration_shifted(displacement: StateVector2D, ref_accel: StateVector2D, ref_angular_vel: float, ref_angular_accel: float) -> StateVector2D:
    """
    Computes the acceleration at a query point on the same planar rigid body as a reference point.
    :param displacement: [m] The displacement vector from the reference to the query point
    :param ref_accel: [m/s^2] The acceleration vector at the reference point
    :param ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis
    :param ref_angular_accel: [rad/s^2] The angular acceleration of the body around the vertical axis
    :return: [m/s^2] The acceleration vector at the given displacement.
    """
    centripetal_acceleration_term = displacement.array * ref_angular_vel ** 2
    angular_acceleration_term = displacement.array * ref_angular_accel
    return StateVector2D(*ref_accel.array + centripetal_acceleration_term + angular_acceleration_term)

class DynamicCarState:
    """Contains the various dynamic attributes of ego."""

    def __init__(self, rear_axle_to_center_dist: float, rear_axle_velocity_2d: StateVector2D, rear_axle_acceleration_2d: StateVector2D, angular_velocity: float=0.0, angular_acceleration: float=0.0, tire_steering_rate: float=0.0):
        """
        :param rear_axle_to_center_dist:[m]  Distance (positive) from rear axle to the geometrical center of ego
        :param rear_axle_velocity_2d: [m/s]Velocity vector at the rear axle
        :param rear_axle_acceleration_2d: [m/s^2] Acceleration vector at the rear axle
        :param angular_velocity: [rad/s] Angular velocity of ego
        :param angular_acceleration: [rad/s^2] Angular acceleration of ego
        :param tire_steering_rate: [rad/s] Tire steering rate of ego
        """
        self._rear_axle_to_center_dist = rear_axle_to_center_dist
        self._angular_velocity = angular_velocity
        self._angular_acceleration = angular_acceleration
        self._rear_axle_velocity_2d = rear_axle_velocity_2d
        self._rear_axle_acceleration_2d = rear_axle_acceleration_2d
        self._tire_steering_rate = tire_steering_rate

    @property
    def rear_axle_velocity_2d(self) -> StateVector2D:
        """
        Returns the vectorial velocity at the middle of the rear axle.
        :return: StateVector2D Containing the velocity at the rear axle
        """
        return self._rear_axle_velocity_2d

    @property
    def rear_axle_acceleration_2d(self) -> StateVector2D:
        """
        Returns the vectorial acceleration at the middle of the rear axle.
        :return: StateVector2D Containing the acceleration at the rear axle
        """
        return self._rear_axle_acceleration_2d

    @cached_property
    def center_velocity_2d(self) -> StateVector2D:
        """
        Returns the vectorial velocity at the geometrical center of Ego.
        :return: StateVector2D Containing the velocity at the geometrical center of Ego
        """
        displacement = StateVector2D(self._rear_axle_to_center_dist, 0.0)
        return get_velocity_shifted(displacement, self.rear_axle_velocity_2d, self.angular_velocity)

    @cached_property
    def center_acceleration_2d(self) -> StateVector2D:
        """
        Returns the vectorial acceleration at the geometrical center of Ego.
        :return: StateVector2D Containing the acceleration at the geometrical center of Ego
        """
        displacement = StateVector2D(self._rear_axle_to_center_dist, 0.0)
        return get_acceleration_shifted(displacement, self.rear_axle_acceleration_2d, self.angular_velocity, self.angular_acceleration)

    @property
    def angular_velocity(self) -> float:
        """
        Getter for the angular velocity of ego.
        :return: [rad/s] Angular velocity
        """
        return self._angular_velocity

    @property
    def angular_acceleration(self) -> float:
        """
        Getter for the angular acceleration of ego.
        :return: [rad/s^2] Angular acceleration
        """
        return self._angular_acceleration

    @property
    def tire_steering_rate(self) -> float:
        """
        Getter for the tire steering rate of ego.
        :return: [rad/s] Tire steering rate
        """
        return self._tire_steering_rate

    @cached_property
    def speed(self) -> float:
        """
        Magnitude of the speed of the center of ego.
        :return: [m/s] 1D speed
        """
        return float(self._rear_axle_velocity_2d.magnitude())

    @cached_property
    def acceleration(self) -> float:
        """
        Magnitude of the acceleration of the center of ego.
        :return: [m/s^2] 1D acceleration
        """
        return float(self._rear_axle_acceleration_2d.magnitude())

    def __eq__(self, other: object) -> bool:
        """
        Compare two instances whether they are numerically close
        :param other: object
        :return: true if the classes are almost equal
        """
        if not isinstance(other, DynamicCarState):
            return NotImplemented
        return self.rear_axle_velocity_2d == other.rear_axle_velocity_2d and self.rear_axle_acceleration_2d == other.rear_axle_acceleration_2d and math.isclose(self._angular_acceleration, other._angular_acceleration) and math.isclose(self._angular_velocity, other._angular_velocity) and math.isclose(self._rear_axle_to_center_dist, other._rear_axle_to_center_dist) and math.isclose(self._tire_steering_rate, other._tire_steering_rate)

    def __repr__(self) -> str:
        """Repr magic method"""
        return f'Rear Axle| velocity: {self.rear_axle_velocity_2d}, acceleration: {self.rear_axle_acceleration_2d}\nCenter   | velocity: {self.center_velocity_2d}, acceleration: {self.center_acceleration_2d}\nangular velocity: {self.angular_velocity}, angular acceleration: {self._angular_acceleration}\nrear_axle_to_center_dist: {self._rear_axle_to_center_dist} \n_tire_steering_rate: {self._tire_steering_rate} \n'

    @staticmethod
    def build_from_rear_axle(rear_axle_to_center_dist: float, rear_axle_velocity_2d: StateVector2D, rear_axle_acceleration_2d: StateVector2D, angular_velocity: float=0.0, angular_acceleration: float=0.0, tire_steering_rate: float=0.0) -> DynamicCarState:
        """
        Construct ego state from rear axle parameters
        :param rear_axle_to_center_dist: [m] distance between center and rear axle
        :param rear_axle_velocity_2d: [m/s] velocity at rear axle
        :param rear_axle_acceleration_2d: [m/s^2] acceleration at rear axle
        :param angular_velocity: [rad/s] angular velocity
        :param angular_acceleration: [rad/s^2] angular acceleration
        :param tire_steering_rate: [rad/s] tire steering_rate
        :return: constructed DynamicCarState of ego.
        """
        return DynamicCarState(rear_axle_to_center_dist=rear_axle_to_center_dist, rear_axle_velocity_2d=rear_axle_velocity_2d, rear_axle_acceleration_2d=rear_axle_acceleration_2d, angular_velocity=angular_velocity, angular_acceleration=angular_acceleration, tire_steering_rate=tire_steering_rate)

    @staticmethod
    def build_from_cog(wheel_base: float, rear_axle_to_center_dist: float, cog_speed: float, cog_acceleration: float, steering_angle: float, angular_acceleration: float=0.0, tire_steering_rate: float=0.0) -> DynamicCarState:
        """
        Construct ego state from rear axle parameters
        :param wheel_base: distance between axles [m]
        :param rear_axle_to_center_dist: distance between center and rear axle [m]
        :param cog_speed: magnitude of speed COG [m/s]
        :param cog_acceleration: magnitude of acceleration at COG [m/s^s]
        :param steering_angle: steering angle at tire [rad]
        :param angular_acceleration: angular acceleration
        :param tire_steering_rate: tire steering rate
        :return: constructed DynamicCarState of ego.
        """
        beta = _get_beta(steering_angle, wheel_base)
        rear_axle_longitudinal_velocity, rear_axle_lateral_velocity = _projected_velocities_from_cog(beta, cog_speed)
        angular_velocity = _angular_velocity_from_cog(cog_speed, wheel_base, beta, steering_angle)
        longitudinal_acceleration, lateral_acceleration = _project_accelerations_from_cog(rear_axle_longitudinal_velocity, angular_velocity, cog_acceleration, beta)
        return DynamicCarState(rear_axle_to_center_dist=rear_axle_to_center_dist, rear_axle_velocity_2d=StateVector2D(rear_axle_longitudinal_velocity, rear_axle_lateral_velocity), rear_axle_acceleration_2d=StateVector2D(longitudinal_acceleration, lateral_acceleration), angular_velocity=angular_velocity, angular_acceleration=angular_acceleration, tire_steering_rate=tire_steering_rate)

def _get_beta(steering_angle: float, wheel_base: float) -> float:
    """
    Computes beta, the angle from rear axle to COG at instantaneous center of rotation
    :param [rad] steering_angle: steering angle of the car
    :param [m] wheel_base: distance between the axles
    :return: [rad] Value of beta
    """
    beta = math.atan2(math.tan(steering_angle), wheel_base)
    return beta

def _projected_velocities_from_cog(beta: float, cog_speed: float) -> Tuple[float, float]:
    """
    Computes the projected velocities at the rear axle using the Bicycle kinematic model using COG data
    :param beta: [rad] the angle from rear axle to COG at instantaneous center of rotation
    :param cog_speed: [m/s] Magnitude of velocity vector at COG
    :return: Tuple with longitudinal and lateral velocities [m/s] at the rear axle
    """
    rear_axle_forward_velocity = math.cos(beta) * cog_speed
    rear_axle_lateral_velocity = 0
    return (rear_axle_forward_velocity, rear_axle_lateral_velocity)

def _angular_velocity_from_cog(cog_speed: float, length_rear_axle_to_cog: float, beta: float, steering_angle: float) -> float:
    """
    Computes the angular velocity using the Bicycle kinematic model using COG data.
    :param cog_speed: [m/s] Magnitude of velocity vector at COG
    :param length_rear_axle_to_cog: [m] Distance from rear axle to COG
    :param beta: [rad] angle from rear axle to COG at instantaneous center of rotation
    :param steering_angle: [rad] of the car
    """
    return cog_speed / length_rear_axle_to_cog * math.cos(beta) * math.tan(steering_angle)

def _project_accelerations_from_cog(rear_axle_longitudinal_velocity: float, angular_velocity: float, cog_acceleration: float, beta: float) -> Tuple[float, float]:
    """
    Computes the projected accelerations at the rear axle using the Bicycle kinematic model using COG data
    :param rear_axle_longitudinal_velocity: [m/s] Longitudinal component of velocity vector at COG
    :param angular_velocity: [rad/s] Angular velocity at COG
    :param cog_acceleration: [m/s^2] Magnitude of acceleration vector at COG
    :param beta: [rad] ]the angle from rear axle to COG at instantaneous center of rotation
    :return: Tuple with longitudinal and lateral velocities [m/s] at the rear axle
    """
    rear_axle_longitudinal_acceleration = math.cos(beta) * cog_acceleration
    rear_axle_lateral_acceleration = rear_axle_longitudinal_velocity * angular_velocity
    return (rear_axle_longitudinal_acceleration, rear_axle_lateral_acceleration)

class EgoTemporalState(AgentTemporalState):
    """
    Temporal ego state, with future and past trajectory
    """

    def __init__(self, current_state: EgoState, past_trajectory: Optional[PredictedTrajectory]=None, predictions: Optional[List[PredictedTrajectory]]=None):
        """
        Initialize temporal state
        :param current_state: current state of ego
        :param past_trajectory: past trajectory, where last waypoint represents the same position as current state
        :param predictions: multimodal predictions, or future trajectory
        """
        super().__init__(initial_time_stamp=current_state.time_point, predictions=predictions, past_trajectory=past_trajectory)
        self._ego_current_state = current_state

    @property
    def ego_current_state(self) -> EgoState:
        """
        :return: the current ego state
        """
        return self._ego_current_state

    @property
    def ego_previous_state(self) -> Optional[EgoState]:
        """
        :return: the previous ego state if exists. This is just a proxy to make sure the return type is correct.
        """
        return self.previous_state

    @cached_property
    def agent(self) -> Agent:
        """
        Casts the EgoTemporalState to an Agent object.
        :return: An Agent object with the parameters of EgoState
        """
        return Agent(metadata=self.ego_current_state.scene_object_metadata, tracked_object_type=TrackedObjectType.EGO, oriented_box=self.ego_current_state.car_footprint.oriented_box, velocity=self.ego_current_state.dynamic_car_state.center_velocity_2d, past_trajectory=self.past_trajectory, predictions=self.predictions)

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

class StaticObject(SceneObject):
    """Represents static objects in the scene."""

    def __init__(self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, metadata: SceneObjectMetadata):
        """
        :param tracked_object_type: Classification type of the object.
        :param oriented_box: OrientedBox representing the StaticObject geometrically.
        :param metadata: Metadata of a static object.
        """
        super().__init__(tracked_object_type, oriented_box, metadata)
        self.predictions = None
        self.past_trajectory = None
        self.velocity = StateVector2D(0.0, 0.0)

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

class SceneObject:
    """Class describing SceneObjects, i.e. objects present in a planning scene"""

    def __init__(self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, metadata: SceneObjectMetadata):
        """
        Representation of an Agent in the scene.
        :param tracked_object_type: Type of the current static object
        :param oriented_box: Geometrical representation of the static object
        :param metadata: High-level information about the object
        """
        self._metadata = metadata
        self.instance_token = None
        self._tracked_object_type = tracked_object_type
        self._box: OrientedBox = oriented_box

    @property
    def metadata(self) -> SceneObjectMetadata:
        """
        Getter for object metadata
        :return: Object's metadata
        """
        return self._metadata

    @property
    def token(self) -> str:
        """
        Getter for object unique token, different for same object in different samples
        :return: The unique token
        """
        return self._metadata.token

    @property
    def track_token(self) -> Optional[str]:
        """
        Getter for object unique token tracked across samples, same for same objects in different samples
        :return: The unique track token
        """
        return self._metadata.track_token

    @property
    def tracked_object_type(self) -> TrackedObjectType:
        """
        Getter for object classification type
        :return: The object classification type
        """
        return self._tracked_object_type

    @property
    def box(self) -> OrientedBox:
        """
        Getter for object OrientedBox
        :return: The object oriented box
        """
        return self._box

    @property
    def center(self) -> StateSE2:
        """
        Getter for object center pose
        :return: The center pose
        """
        return self.box.center

    @classmethod
    def make_random(cls, token: str, object_type: TrackedObjectType) -> SceneObject:
        """
        Instantiates a random SceneObject.
        :param token: Unique token
        :param object_type: Classification type
        :return: SceneObject instance.
        """
        center = random.sample(range(50), 2)
        heading = np.random.uniform(-np.pi, np.pi)
        size = random.sample(range(1, 50), 3)
        track_id = random.sample(range(1, 10), 1)[0]
        timestamp_us = random.sample(range(1, 10), 1)[0]
        return SceneObject(metadata=SceneObjectMetadata(token=token, track_id=track_id, track_token=token, timestamp_us=timestamp_us), tracked_object_type=object_type, oriented_box=OrientedBox(StateSE2(*center, heading), size[0], size[1], size[2]))

    @classmethod
    def from_raw_params(cls, token: str, track_token: str, timestamp_us: int, track_id: int, center: StateSE2, size: Tuple[float, float, float]) -> SceneObject:
        """
        Instantiates a generic SceneObject.
        :param token: The token of the object.
        :param track_token: The track token of the object.
        :param timestamp_us: [us] timestamp for the object.
        :param track_id: Human readable track id.
        :param center: Center pose.
        :param size: Size of the geometrical box (width, length, height).
        :return: SceneObject instance.
        """
        box = OrientedBox(center, width=size[0], length=size[1], height=size[2])
        return SceneObject(metadata=SceneObjectMetadata(token=token, track_token=track_token, timestamp_us=timestamp_us, track_id=track_id), tracked_object_type=TrackedObjectType.GENERIC_OBJECT, oriented_box=box)

class TestWaypoint(unittest.TestCase):
    """Tests Waypoint class"""

    def setUp(self) -> None:
        """Sets sample parameters for testing"""
        mock_time_point = Mock(time_us=0)
        mock_box = Mock(center=Mock(x='center_x', y='center_y', heading='center_heading'), length='length', width='width', height='height')
        mock_velocity = Mock(x='velocity_x', y='velocity_y')
        self.waypoint = Waypoint(mock_time_point, mock_box, mock_velocity)
        self.waypoint_no_vel = Waypoint(mock_time_point, mock_box)

    def test_iterable(self) -> None:
        """Test that the iterable gets built correctly."""
        iterable_waypoint = iter(self.waypoint)
        iterable_expected = [0, 'center_x', 'center_y', 'center_heading', 'velocity_x', 'velocity_y']
        for expected, actual in zip(iterable_expected, iterable_waypoint):
            self.assertEqual(expected, actual)
        iterable_waypoint_no_vel = iter(self.waypoint_no_vel)
        iterable_expected = [0, 'center_x', 'center_y', 'center_heading', None, None]
        for expected, actual in zip(iterable_expected, iterable_waypoint_no_vel):
            self.assertEqual(expected, actual)

    def test_serialize(self) -> None:
        """Tests that the serialization works as expected."""
        serialized_waypoint = self.waypoint.serialize()
        serialized_expected = [0, 'center_x', 'center_y', 'center_heading', 'length', 'width', 'height', 'velocity_x', 'velocity_y']
        self.assertEqual(serialized_expected, serialized_waypoint)
        serialized_waypoint_no_vel = self.waypoint_no_vel.serialize()
        serialized_no_vel_expected = [0, 'center_x', 'center_y', 'center_heading', 'length', 'width', 'height', None, None]
        self.assertEqual(serialized_no_vel_expected, serialized_waypoint_no_vel)

    @patch('nuplan.common.actor_state.waypoint.StateVector2D')
    @patch('nuplan.common.actor_state.waypoint.OrientedBox')
    @patch('nuplan.common.actor_state.waypoint.TimePoint')
    @patch('nuplan.common.actor_state.waypoint.StateSE2')
    @patch('nuplan.common.actor_state.waypoint.Waypoint')
    def test_deserialize(self, mock_waypoint: Mock, mock_se2: Mock, mock_time_point: Mock, mock_box: Mock, mock_velocity: Mock) -> None:
        """Tests that the object is deserialized correctly."""
        mock_se2.return_value = 'se2'
        mock_time_point.return_value = 'time_point'
        mock_box.return_value = 'mock_box'
        mock_velocity.return_value = 'velocity'
        waypoint = self.waypoint.deserialize([0, 1, 2, 3, 4, 5, 6, 7, 8])
        mock_time_point.assert_called_once_with(0)
        mock_se2.assert_called_once_with(1, 2, 3)
        mock_box.assert_called_once_with(mock_se2.return_value, 4, 5, 6)
        mock_velocity.assert_called_once_with(7, 8)
        mock_waypoint.assert_called_with(time_point=mock_time_point.return_value, oriented_box=mock_box.return_value, velocity=mock_velocity.return_value)
        self.assertEqual(mock_waypoint.return_value, waypoint)

    @patch('nuplan.common.actor_state.waypoint.StateVector2D')
    @patch('nuplan.common.actor_state.waypoint.OrientedBox')
    @patch('nuplan.common.actor_state.waypoint.TimePoint')
    @patch('nuplan.common.actor_state.waypoint.StateSE2')
    @patch('nuplan.common.actor_state.waypoint.Waypoint')
    def test_deserialize_no_velocity(self, mock_waypoint: Mock, mock_se2: Mock, mock_time_point: Mock, mock_box: Mock, mock_velocity: Mock) -> None:
        """Tests that the object is deserialized correctly when no velocity is provided."""
        mock_se2.return_value = 'se2'
        mock_time_point.return_value = 'time_point'
        mock_box.return_value = 'mock_box'
        mock_velocity.return_value = 'velocity'
        waypoint = self.waypoint.deserialize([0, 1, 2, 3, 4, 5, 6, None, None])
        mock_time_point.assert_called_once_with(0)
        mock_se2.assert_called_once_with(1, 2, 3)
        mock_box.assert_called_once_with(mock_se2.return_value, 4, 5, 6)
        mock_velocity.assert_not_called()
        mock_waypoint.assert_called_with(time_point=mock_time_point.return_value, oriented_box=mock_box.return_value, velocity=None)
        self.assertEqual(mock_waypoint.return_value, waypoint)

    @patch('nuplan.common.actor_state.waypoint.SplitState', autospec=True)
    def test_to_split_state(self, mock_split_state: Mock) -> None:
        """Tests that the object is split correctly"""
        result = self.waypoint.to_split_state()
        expected_linear_states = [0, 'center_x', 'center_y', 'velocity_x', 'velocity_y']
        expected_angular_states = ['center_heading']
        expected_fixed_states = ['width', 'length', 'height']
        mock_split_state.assert_called_once_with(expected_linear_states, expected_angular_states, expected_fixed_states)
        self.assertEqual(result, mock_split_state.return_value)

    @patch('nuplan.common.actor_state.waypoint.StateVector2D', autospec=True)
    @patch('nuplan.common.actor_state.waypoint.OrientedBox', autospec=True)
    @patch('nuplan.common.actor_state.waypoint.TimePoint', autospec=True)
    @patch('nuplan.common.actor_state.waypoint.StateSE2', autospec=True)
    def test_from_split_state(self, mock_se2: Mock, mock_time_point: Mock, mock_box: Mock, mock_vector: Mock) -> None:
        """Tests that the object is recreated correctly from a split state"""
        split_state = self.waypoint.to_split_state()
        result = self.waypoint.from_split_state(split_state)
        mock_time_point.assert_called_once_with(0)
        mock_se2.assert_called_once_with('center_x', 'center_y', 'center_heading')
        mock_vector.assert_called_once_with('velocity_x', 'velocity_y')
        mock_box.assert_called_once_with(mock_se2.return_value, length='length', width='width', height='height')
        self.assertEqual(result.time_point, mock_time_point.return_value)
        self.assertEqual(result.oriented_box, mock_box.return_value)
        self.assertEqual(result.velocity, mock_vector.return_value)

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

class TestTrackedObjects(unittest.TestCase):
    """Tests TrackedObjects class"""

    def setUp(self) -> None:
        """Creates sample agents for testing"""
        self.agents = [get_sample_agent('foo', TrackedObjectType.PEDESTRIAN), get_sample_agent('bar', TrackedObjectType.VEHICLE), get_sample_agent('bar_out_the_car', TrackedObjectType.PEDESTRIAN)]

    def test_construction(self) -> None:
        """Tests that the object can be created correctly."""
        tracked_objects = TrackedObjects(self.agents)
        expected_type_and_set_of_tokens: Dict[TrackedObjectType, Any] = {object_type: set() for object_type in TrackedObjectType}
        expected_type_and_set_of_tokens[TrackedObjectType.PEDESTRIAN].update({'foo', 'bar_out_the_car'})
        expected_type_and_set_of_tokens[TrackedObjectType.VEHICLE].update({'bar'})
        for tracked_object_type in TrackedObjectType:
            if tracked_object_type not in expected_type_and_set_of_tokens:
                continue
            self.assertEqual(expected_type_and_set_of_tokens[tracked_object_type], {tracked_object.token for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type)})

    def test_get_subset(self) -> None:
        """Tests that the object can be created correctly."""
        tracked_objects = TrackedObjects(self.agents)
        agents = tracked_objects.get_agents()
        static_objects = tracked_objects.get_static_objects()
        self.assertEqual(3, len(agents))
        self.assertEqual(0, len(static_objects))

    def test_get_tracked_objects_of_types(self) -> None:
        """Test get_tracked_objects_of_types()"""
        tracked_objects = TrackedObjects(self.agents)
        track_types = [TrackedObjectType.PEDESTRIAN, TrackedObjectType.VEHICLE]
        tracks = tracked_objects.get_tracked_objects_of_types(track_types)
        self.assertEqual(3, len(tracks))

def get_sample_agent(token: str='test', agent_type: TrackedObjectType=TrackedObjectType.VEHICLE, num_past_states: Optional[int]=1, num_future_states: Optional[int]=1) -> Agent:
    """
    Creates a sample Agent, the token and agent type can be specified for various testing purposes.
    :param token: The unique token to assign to the agent.
    :param agent_type: Classification of the agent.
    :param num_past_states: How many states to generate in the past trajectory. With None, that will be assigned to
    the past_trajectory otherwise the current state + num_past_states will be added.
    :param num_future_states: How many states to generate in the future trajectory. If `None` is passed, `None` will
    be assigned to the predictions; otherwise the current state + num_future_states will be added.
    :return: A sample Agent.
    """
    initial_timestamp = 10
    sample_oriented_box = get_sample_oriented_box()
    return Agent(agent_type, sample_oriented_box, metadata=SceneObjectMetadata(timestamp_us=initial_timestamp, track_token=token, track_id=None, token=token), velocity=StateVector2D(0.0, 0.0), predictions=[PredictedTrajectory(1.0, [Waypoint(time_point=TimePoint(initial_timestamp + i * 5), oriented_box=sample_oriented_box) for i in range(num_future_states + 1)])] if num_future_states is not None else None, past_trajectory=PredictedTrajectory(1.0, [Waypoint(time_point=TimePoint(initial_timestamp - i * 5), oriented_box=sample_oriented_box) for i in reversed(range(num_past_states + 1))]) if num_past_states is not None else None)

class TestAgent(unittest.TestCase):
    """Test suite for the Agent class"""

    def setUp(self) -> None:
        """Setup parameters for tests"""
        self.sample_token = 'abc123'
        self.track_token = 'abc123'
        self.timestamp = 123
        self.agent_type = TrackedObjectType.VEHICLE
        self.sample_pose = StateSE2(1.0, 2.0, np.pi / 2.0)
        self.wlh = (2.0, 4.0, 1.5)
        self.velocity = StateVector2D(1.0, 2.2)

    def test_agent_state(self) -> None:
        """Test AgentState."""
        angular_velocity = 10.0
        oriented_box = get_sample_oriented_box()
        metadata = SceneObjectMetadata(token=self.sample_token, track_token=self.track_token, timestamp_us=self.timestamp, track_id=None)
        agent_state = AgentState(TrackedObjectType.VEHICLE, oriented_box=oriented_box, velocity=self.velocity, metadata=metadata, angular_velocity=angular_velocity)
        self.assertEqual(agent_state.tracked_object_type, TrackedObjectType.VEHICLE)
        self.assertEqual(agent_state.box, oriented_box)
        self.assertEqual(agent_state.metadata, metadata)
        self.assertEqual(agent_state.velocity, self.velocity)
        self.assertEqual(agent_state.angular_velocity, angular_velocity)
        self.assertEqual(agent_state.token, metadata.token)
        self.assertEqual(agent_state.track_token, metadata.track_token)

    def test_agent_types(self) -> None:
        """Test that enum works for both existing and missing keys"""
        self.assertEqual(TrackedObjectType(0), TrackedObjectType.VEHICLE)
        self.assertEqual(TrackedObjectType.VEHICLE.fullname, 'vehicle')
        with self.assertRaises(ValueError):
            TrackedObjectType('missing_key')

    def test_construction(self) -> None:
        """Test that agents can be constructed correctly."""
        oriented_box = get_sample_oriented_box()
        agent = Agent(metadata=SceneObjectMetadata(token=self.sample_token, track_token=self.track_token, timestamp_us=self.timestamp, track_id=None), tracked_object_type=self.agent_type, oriented_box=oriented_box, velocity=self.velocity)
        self.assertTrue(agent.angular_velocity is None)

    def test_set_predictions(self) -> None:
        """Tests assignment of predictions to agents, and that this fails if the probabilities don't sum to one."""
        agent = get_sample_agent()
        waypoints = [Waypoint(TimePoint(t), get_sample_oriented_box(), StateVector2D(0.0, 0.0)) for t in range(5)]
        predictions = [PredictedTrajectory(0.3, waypoints), PredictedTrajectory(0.7, waypoints)]
        agent.predictions = predictions
        self.assertEqual(len(agent.predictions), 2)
        self.assertEqual(0.3, agent.predictions[0].probability)
        self.assertEqual(0.7, agent.predictions[1].probability)
        predictions += predictions
        with self.assertRaises(ValueError):
            agent.predictions = predictions

    def test_set_past_trajectory(self) -> None:
        """Tests assignment of past trajectory to agents."""
        agent = get_sample_agent()
        waypoints = [Waypoint(TimePoint(t), get_sample_oriented_box(), StateVector2D(0.0, 0.0)) for t in range(agent.metadata.timestamp_us + 1)]
        agent.past_trajectory = PredictedTrajectory(1, waypoints)
        self.assertEqual(len(agent.past_trajectory.waypoints), 11)
        with self.assertRaises(ValueError):
            agent.past_trajectory = PredictedTrajectory(1, [Waypoint(TimePoint(t), get_sample_oriented_box(), StateVector2D(0.0, 0.0)) for t in range(3)])

def get_sample_oriented_box() -> OrientedBox:
    """
    Creates a sample OrientedBox.
    :return: A sample OrientedBox with arbitrary parameters
    """
    return OrientedBox(get_sample_pose(), 4.0, 2.0, 1.5)

def get_sample_pose() -> StateSE2:
    """
    Creates a sample SE2 Pose.
    :return: A sample SE2 Pose with arbitrary parameters
    """
    return StateSE2(1.0, 2.0, math.pi / 2.0)

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

class TestDynamicCarState(unittest.TestCase):
    """Tests DynamicCarState class and helper functions"""

    def setUp(self) -> None:
        """Sets sample variables for testing"""
        self.displacement = StateVector2D(2.0, 2.0)
        self.reference_vector = StateVector2D(2.3, 3.4)
        self.angular_velocity = 0.2
        self.dynamic_car_state = DynamicCarState(rear_axle_to_center_dist=1, rear_axle_velocity_2d=self.reference_vector, rear_axle_acceleration_2d=StateVector2D(0.1, 0.2), angular_velocity=2, angular_acceleration=2.5, tire_steering_rate=0.5)

    def test_velocity_transfer(self) -> None:
        """Tests behavior of velocity transfer formula for planar rigid bodies."""
        actual_velocity = get_velocity_shifted(self.displacement, self.reference_vector, self.angular_velocity)
        expected_velocity_p2 = StateVector2D(1.9, 3.8)
        np.testing.assert_array_almost_equal(expected_velocity_p2.array, actual_velocity.array, 6)
        actual_velocity = get_velocity_shifted(StateVector2D(0.0, 0.0), self.reference_vector, self.angular_velocity)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_velocity.array, 6)
        actual_velocity = get_velocity_shifted(self.displacement, self.reference_vector, 0)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_velocity.array, 6)

    def test_acceleration_transfer(self) -> None:
        """Tests behavior of acceleration transfer formula for planar rigid bodies."""
        angular_acceleration = 0.234
        actual_acceleration = get_acceleration_shifted(self.displacement, self.reference_vector, self.angular_velocity, angular_acceleration)
        np.testing.assert_array_almost_equal(StateVector2D(2.848, 3.948).array, actual_acceleration.array, 6)
        actual_acceleration = get_acceleration_shifted(StateVector2D(0.0, 0.0), self.reference_vector, self.angular_velocity, angular_acceleration)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_acceleration.array, 6)
        actual_acceleration = get_acceleration_shifted(self.displacement, self.reference_vector, 0, 0)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_acceleration.array, 6)

    def test_initialization(self) -> None:
        """Tests that object initialization works as intended"""
        self.assertEqual(1, self.dynamic_car_state._rear_axle_to_center_dist)
        self.assertEqual(self.reference_vector, self.dynamic_car_state._rear_axle_velocity_2d)
        self.assertEqual(StateVector2D(0.1, 0.2), self.dynamic_car_state._rear_axle_acceleration_2d)
        self.assertEqual(2, self.dynamic_car_state._angular_velocity)
        self.assertEqual(2.5, self.dynamic_car_state._angular_acceleration)
        self.assertEqual(0.5, self.dynamic_car_state._tire_steering_rate)

    def test_properties(self) -> None:
        """Checks that the properties return the expected variables."""
        self.assertTrue(self.dynamic_car_state.rear_axle_velocity_2d is self.dynamic_car_state._rear_axle_velocity_2d)
        self.assertTrue(self.dynamic_car_state.rear_axle_acceleration_2d is self.dynamic_car_state._rear_axle_acceleration_2d)
        self.assertTrue(self.dynamic_car_state.tire_steering_rate is self.dynamic_car_state._tire_steering_rate)
        self.assertTrue(self.dynamic_car_state.tire_steering_rate is self.dynamic_car_state._tire_steering_rate)
        self.assertAlmostEqual(4.104875150354758, self.dynamic_car_state.speed)
        self.assertEqual(0.22360679774997896, self.dynamic_car_state.acceleration)

    @patch('nuplan.common.actor_state.dynamic_car_state.StateVector2D', Mock())
    @patch('nuplan.common.actor_state.dynamic_car_state.DynamicCarState', autospec=DynamicCarState)
    def test_build_from_rear_axle(self, mock_dynamic_car_state: Mock) -> None:
        """Tests that constructor from rear axle behaves as intended."""
        mock_velocity = Mock()
        mock_acceleration = Mock()
        self.dynamic_car_state.build_from_rear_axle(1, mock_velocity, mock_acceleration, 4, 5, 6)
        mock_dynamic_car_state.assert_called_with(rear_axle_to_center_dist=1, rear_axle_velocity_2d=mock_velocity, rear_axle_acceleration_2d=mock_acceleration, angular_velocity=4, angular_acceleration=5, tire_steering_rate=6)

    @patch('nuplan.common.actor_state.dynamic_car_state.StateVector2D')
    @patch('nuplan.common.actor_state.dynamic_car_state.math', Mock())
    @patch('nuplan.common.actor_state.dynamic_car_state._angular_velocity_from_cog')
    @patch('nuplan.common.actor_state.dynamic_car_state._projected_velocities_from_cog')
    @patch('nuplan.common.actor_state.dynamic_car_state._project_accelerations_from_cog')
    @patch('nuplan.common.actor_state.dynamic_car_state._get_beta')
    @patch('nuplan.common.actor_state.dynamic_car_state.DynamicCarState', autospec=DynamicCarState)
    def test_build_from_cog(self, mock_dynamic_car_state: Mock, mock_beta: Mock, mock_accelerations: Mock, mock_velocities: Mock, mock_angular_velocity: Mock, mock_vector: Mock) -> None:
        """Checks that constructor from COG computes the correct projections."""
        wheel_base = MagicMock(return_value='wheel_base')
        rear_axle_to_center = MagicMock(return_value='rear_axle_to_center')
        cog_speed = MagicMock(return_value='cog_speed')
        cog_acceleration = MagicMock(return_value='cog_acceleration')
        steering_angle = MagicMock(return_value='steering_angle')
        angular_accel = MagicMock(return_value='angular_accel')
        tire_steering_rate = MagicMock(return_value='tire_steering_rate')
        mock_velocities.return_value = ('x_vel', 'y_vel')
        mock_accelerations.return_value = ('x_acc', 'y_acc')
        self.dynamic_car_state.build_from_cog(wheel_base, rear_axle_to_center, cog_speed, cog_acceleration, steering_angle, angular_accel, tire_steering_rate)
        mock_beta.assert_called_once_with(steering_angle, wheel_base)
        mock_velocities.assert_called_once_with(mock_beta.return_value, cog_speed)
        mock_angular_velocity.assert_called_once_with(cog_speed, wheel_base, mock_beta.return_value, steering_angle)
        mock_accelerations.assert_called_once_with('x_vel', mock_angular_velocity.return_value, cog_acceleration, mock_beta.return_value)
        mock_dynamic_car_state.assert_called_with(rear_axle_to_center_dist=rear_axle_to_center, rear_axle_velocity_2d=mock_vector(mock_velocities.return_value), rear_axle_acceleration_2d=mock_vector(mock_accelerations.return_value), angular_velocity=mock_angular_velocity.return_value, angular_acceleration=angular_accel, tire_steering_rate=tire_steering_rate)

class TestCarFootprint(unittest.TestCase):
    """Tests CarFoorprint class"""

    def setUp(self) -> None:
        """Sets sample parameters for testing"""
        self.center_position_from_rear_axle = get_pacifica_parameters().rear_axle_to_center

    def test_car_footprint_creation(self) -> None:
        """Checks that the car footprint is created correctly, in particular the point of interest."""
        car_footprint = CarFootprint.build_from_rear_axle(get_sample_pose(), get_pacifica_parameters())
        self.assertAlmostEqual(car_footprint.rear_axle_to_center_dist, self.center_position_from_rear_axle)
        expected_values = {OrientedBoxPointType.FRONT_BUMPER: (1.0, 6.049), OrientedBoxPointType.REAR_BUMPER: (1.0, 0.873), OrientedBoxPointType.FRONT_LEFT: (-0.1485, 6.049), OrientedBoxPointType.REAR_LEFT: (-0.1485, 0.873), OrientedBoxPointType.REAR_RIGHT: (2.1485, 0.873), OrientedBoxPointType.FRONT_RIGHT: (2.1485, 6.049), OrientedBoxPointType.CENTER: (1.0, 3.461)}
        for point, position in expected_values.items():
            np.testing.assert_array_almost_equal(position, tuple(car_footprint.corner(point)))
        np.testing.assert_array_almost_equal(expected_values[OrientedBoxPointType.FRONT_LEFT], tuple(car_footprint.get_point_of_interest(OrientedBoxPointType.FRONT_LEFT)), 6)

class TestSceneObject(unittest.TestCase):
    """Tests SceneObject class"""

    @patch('nuplan.common.actor_state.tracked_objects_types.TrackedObjectType')
    @patch('nuplan.common.actor_state.oriented_box.OrientedBox')
    def test_initialization(self, mock_box: Mock, mock_tracked_object_type: Mock) -> None:
        """Tests that agents can be initialized correctly"""
        scene_object = SceneObject(mock_tracked_object_type, mock_box, SceneObjectMetadata(1, '123', 1, '456'))
        self.assertEqual('123', scene_object.token)
        self.assertEqual('456', scene_object.track_token)
        self.assertEqual(mock_box, scene_object.box)
        self.assertEqual(mock_tracked_object_type, scene_object.tracked_object_type)

    @patch('nuplan.common.actor_state.scene_object.StateSE2')
    @patch('nuplan.common.actor_state.scene_object.OrientedBox')
    @patch('nuplan.common.actor_state.scene_object.TrackedObjectType')
    @patch('nuplan.common.actor_state.scene_object.SceneObject.__init__')
    def test_construction(self, mock_init: Mock, mock_type: Mock, mock_box_object: Mock, mock_state: Mock) -> None:
        """Test that agents can be constructed correctly."""
        mock_init.return_value = None
        mock_box = Mock()
        mock_box_object.return_value = mock_box
        _ = SceneObject.from_raw_params('123', '123', 1, 1, mock_state, size=(3, 2, 1))
        mock_box_object.assert_called_with(mock_state, width=3, length=2, height=1)
        mock_init.assert_called_with(metadata=SceneObjectMetadata(token='123', track_token='123', timestamp_us=1, track_id=1), tracked_object_type=mock_type.GENERIC_OBJECT, oriented_box=mock_box)

def vector_2d_from_magnitude_angle(magnitude: float, angle: float) -> StateVector2D:
    """
    Projects magnitude and angle into a vector of x-y components.
    :param magnitude: The magnitude of the vector.
    :param angle: The angle of the vector.
    :return: A state vector.
    """
    return StateVector2D(np.cos(angle) * magnitude, np.sin(angle) * magnitude)

def from_scene_to_tracked_objects_with_predictions(scene: Dict[str, Any], predictions: List[Dict[str, Any]]) -> TrackedObjects:
    """
    Creates tracked objects, adding prediction from the given parameter.
    :param scene: The input scene loaded from the json file.
    :param predictions: Predictions for the tracked objects in the scene.
    :return: Tracked objects from the scene, with predictions loaded from the input.
    """
    tracked_objects = from_scene_to_tracked_objects(scene)
    for tracked_object in tracked_objects:
        for prediction in predictions:
            if str(prediction['id']) == str(tracked_object.token):
                box = tracked_object.box
                tracked_object.predictions = [PredictedTrajectory(probability=1.0, waypoints=SceneSimpleTrajectory(prediction['states'], width=box.width, length=box.length, height=box.height).get_sampled_trajectory())]
                del prediction
                break
    return tracked_objects

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

class SceneSimpleTrajectory(AbstractTrajectory):
    """
    Simple trajectory that is used to represent scene's predictions.
    """

    def __init__(self, prediction_states: List[Dict[str, Any]], width: float, length: float, height: float):
        """
        Constructor.

        :param prediction_states: Dictionary of states.
        :param width: [m] Width of the agent.
        :param length: [m] Length of the agent.
        :param height: [m] Height of the agent.
        """
        self._states: List[Waypoint] = []
        self._state_at_time: Dict[TimePoint, Waypoint] = {}
        for state in prediction_states:
            time = TimePoint(int(state['timestamp'] * 1000000.0))
            coordinates: List[float] = state['pose']
            center = StateSE2(x=coordinates[0], y=coordinates[1], heading=coordinates[2])
            self._states.append(Waypoint(time_point=time, oriented_box=OrientedBox(center=center, width=width, length=length, height=height)))
            self._state_at_time[time] = self._states[-1]
        self._start_time = prediction_states[0]['timestamp']
        self._end_time = prediction_states[-1]['timestamp']

    @property
    def start_time(self) -> TimePoint:
        """
        Get the trajectory start time.
        :return: Start time.
        """
        return self._start_time

    @property
    def end_time(self) -> TimePoint:
        """
        Get the trajectory end time.
        :return: End time.
        """
        return self._end_time

    def get_state_at_time(self, time_point: TimePoint) -> Any:
        """
        Get the state of the actor at the specified time point.
        :param time_point: Time for which are want to query a state.
        :return: State at the specified time.

        :raises Exception: Throws an exception in case a time_point is beyond range of a trajectory.
        """
        return self._state_at_time[time_point]

    def get_state_at_times(self, time_points: List[TimePoint]) -> List[Any]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_sampled_trajectory(self) -> List[Any]:
        """
        Get the sampled states along the trajectory.
        :return: Discrete trajectory consisting of states.
        """
        return self._states

def from_scene_tracked_object(scene: Dict[str, Any], object_type: TrackedObjectType) -> TrackedObject:
    """
    Convert scene to a TrackedObject.
    :param scene: scene of an agent.
    :param object_type: type of the resulting object.
    :return Agent extracted from a scene.
    """
    token = scene['id']
    box = scene['box']
    pose = box['pose']
    size = box['size'] if 'size' in box.keys() else [0.5, 0.5]
    default_height = 1.5
    box = OrientedBox(StateSE2(*pose), width=size[0], length=size[1], height=default_height)
    if object_type in AGENT_TYPES:
        return Agent(metadata=SceneObjectMetadata(token=str(token), track_token=str(token), track_id=token, timestamp_us=0), tracked_object_type=object_type, oriented_box=box, velocity=StateVector2D(scene['speed'], 0))
    else:
        return StaticObject(metadata=SceneObjectMetadata(token=str(token), track_token=str(token), track_id=token, timestamp_us=0), tracked_object_type=object_type, oriented_box=box)

def from_scene_to_tracked_objects(scene: Dict[str, Any]) -> TrackedObjects:
    """
    Convert scene["world"] into boxes
    :param scene: scene["world"] coming from json
    :return List of boxes representing all agents
    """
    if 'world' in scene.keys():
        raise ValueError("You need to pass only the 'world' field of scene, not the whole dict!")
    tracked_objects: List[TrackedObject] = []
    scene_labels_map = {'vehicles': TrackedObjectType.VEHICLE, 'bicycles': TrackedObjectType.BICYCLE, 'pedestrians': TrackedObjectType.PEDESTRIAN}
    for label, object_type in scene_labels_map.items():
        if label in scene:
            tracked_objects.extend([from_scene_tracked_object(scene_object, object_type) for scene_object in scene[label]])
    return TrackedObjects(tracked_objects)

def from_scene_to_tracked_objects_with_scene_predictions(scene: Dict[str, Any]) -> TrackedObjects:
    """
    Creates tracked objects, loading the predictions directly from the scene json.
    :param scene: The input scene loaded from the json file.
    :return: Tracked objects from the scene, with predictions loaded from the scene json.
    """
    tracked_objects = from_scene_to_tracked_objects(scene['world'])
    tracked_objects_map: Dict[str, TrackedObject] = {track.token: track for track in tracked_objects}
    for prediction in scene['prediction']:
        prediction_id = str(prediction['id'])
        if prediction_id not in tracked_objects_map:
            logger.warning('Json scene file contains prediction not assigned to any track: %s.', prediction_id)
            continue
        box = tracked_objects_map[prediction_id].box
        current_state = {'timestamp': tracked_objects_map[prediction_id].metadata.timestamp_s, 'pose': list(tracked_objects_map[prediction_id].center)}
        tracked_objects_map[prediction_id].predictions = [PredictedTrajectory(probability=mode['probability'], waypoints=SceneSimpleTrajectory(_validate_an_unite_predictions(current_state, mode['states']), width=box.width, length=box.length, height=box.height).get_sampled_trajectory()) for mode in prediction['modes']]
    return tracked_objects

@nuplan_test(path='json/load_from_scene.json')
def test_load_from_scene(scene: Dict[str, Any]) -> None:
    """
    Tests loading tracked objects with predictions from a scene json.
    :param scene: The input scene loaded from the json file.
    """
    tracked_objects = from_scene_to_tracked_objects_with_scene_predictions(scene)
    agent = tracked_objects.tracked_objects[0]
    assert agent.track_token == '0'
    assert agent.tracked_object_type == TrackedObjectType.VEHICLE
    assert list(agent.box.center) == [1, 2, 0]
    assert len(agent.predictions) == 2
    assert agent.predictions[0].probability == 0.9
    assert agent.predictions[1].probability == 0.1
    assert agent.box.width == 2.0
    assert agent.box.length == 4.7
    for i, state in enumerate(agent.predictions[0].waypoints):
        assert list(state.center) == pytest.approx([1 + 0.01 * i, 2 + 0.01 * i, 0.01 * i])
        assert state.time_us == pytest.approx(agent.metadata.timestamp_us + int(0.5 * i * 1000000.0))

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

class KinematicBicycleModel(AbstractMotionModel):
    """
    A class describing the kinematic motion model where the rear axle is the point of reference.
    """

    def __init__(self, vehicle: VehicleParameters, max_steering_angle: float=np.pi / 3, accel_time_constant: float=0.2, steering_angle_time_constant: float=0.05):
        """
        Construct KinematicBicycleModel.

        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        :param accel_time_constant: low pass filter time constant for acceleration in s
        :param steering_angle_time_constant: low pass filter time constant for steering angle in s
        """
        self._vehicle = vehicle
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """Inherited, see super class."""
        longitudinal_speed = state.dynamic_car_state.rear_axle_velocity_2d.x
        x_dot = longitudinal_speed * np.cos(state.rear_axle.heading)
        y_dot = longitudinal_speed * np.sin(state.rear_axle.heading)
        yaw_dot = longitudinal_speed * np.tan(state.tire_steering_angle) / self._vehicle.wheel_base
        return EgoStateDot.build_from_rear_axle(rear_axle_pose=StateSE2(x=x_dot, y=y_dot, heading=yaw_dot), rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_acceleration_2d, rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=state.dynamic_car_state.tire_steering_rate, time_point=state.time_point, is_in_auto_mode=True, vehicle_parameters=self._vehicle)

    def _update_commands(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """
        This function applies some first order control delay/a low pass filter to acceleration/steering.

        :param state: Ego state
        :param ideal_dynamic_state: The desired dynamic state for propagation
        :param sampling_time: The time duration to propagate for
        :return: propagating_state including updated dynamic_state
        """
        dt_control = sampling_time.time_s
        accel = state.dynamic_car_state.rear_axle_acceleration_2d.x
        steering_angle = state.tire_steering_angle
        ideal_accel_x = ideal_dynamic_state.rear_axle_acceleration_2d.x
        ideal_steering_angle = dt_control * ideal_dynamic_state.tire_steering_rate + steering_angle
        updated_accel_x = dt_control / (dt_control + self._accel_time_constant) * (ideal_accel_x - accel) + accel
        updated_steering_angle = dt_control / (dt_control + self._steering_angle_time_constant) * (ideal_steering_angle - steering_angle) + steering_angle
        updated_steering_rate = (updated_steering_angle - steering_angle) / dt_control
        dynamic_state = DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist=state.car_footprint.rear_axle_to_center_dist, rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=StateVector2D(updated_accel_x, 0), tire_steering_rate=updated_steering_rate)
        propagating_state = EgoState(car_footprint=state.car_footprint, dynamic_car_state=dynamic_state, tire_steering_angle=state.tire_steering_angle, is_in_auto_mode=True, time_point=state.time_point)
        return propagating_state

    def propagate_state(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """Inherited, see super class."""
        propagating_state = self._update_commands(state, ideal_dynamic_state, sampling_time)
        state_dot = self.get_state_dot(propagating_state)
        next_x = forward_integrate(propagating_state.rear_axle.x, state_dot.rear_axle.x, sampling_time)
        next_y = forward_integrate(propagating_state.rear_axle.y, state_dot.rear_axle.y, sampling_time)
        next_heading = forward_integrate(propagating_state.rear_axle.heading, state_dot.rear_axle.heading, sampling_time)
        next_heading = principal_value(next_heading)
        next_point_velocity_x = forward_integrate(propagating_state.dynamic_car_state.rear_axle_velocity_2d.x, state_dot.dynamic_car_state.rear_axle_velocity_2d.x, sampling_time)
        next_point_velocity_y = 0.0
        next_point_tire_steering_angle = np.clip(forward_integrate(propagating_state.tire_steering_angle, state_dot.tire_steering_angle, sampling_time), -self._max_steering_angle, self._max_steering_angle)
        next_point_angular_velocity = next_point_velocity_x * np.tan(next_point_tire_steering_angle) / self._vehicle.wheel_base
        rear_axle_accel = [state_dot.dynamic_car_state.rear_axle_velocity_2d.x, state_dot.dynamic_car_state.rear_axle_velocity_2d.y]
        angular_accel = (next_point_angular_velocity - state.dynamic_car_state.angular_velocity) / sampling_time.time_s
        return EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(next_x, next_y, next_heading), rear_axle_velocity_2d=StateVector2D(next_point_velocity_x, next_point_velocity_y), rear_axle_acceleration_2d=StateVector2D(rear_axle_accel[0], rear_axle_accel[1]), tire_steering_angle=float(next_point_tire_steering_angle), time_point=propagating_state.time_point + sampling_time, vehicle_parameters=self._vehicle, is_in_auto_mode=True, angular_vel=next_point_angular_velocity, angular_accel=angular_accel, tire_steering_rate=state_dot.tire_steering_angle)

class TestKinematicMotionModel(unittest.TestCase):
    """
    Run tests for Kinematic Bicycle Model.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.vehicle = get_pacifica_parameters()
        self.ego_state = get_sample_ego_state()
        self.sampling_time = TimePoint(1000000)
        self.motion_model = KinematicBicycleModel(self.vehicle)
        wheel_base = self.vehicle.wheel_base
        self.longitudinal_speed = self.ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        self.x_dot = self.longitudinal_speed * np.cos(self.ego_state.rear_axle.heading)
        self.y_dot = self.longitudinal_speed * np.sin(self.ego_state.rear_axle.heading)
        self.yaw_dot = self.longitudinal_speed * np.tan(self.ego_state.tire_steering_angle) / wheel_base

    def test_get_state_dot(self) -> None:
        """
        Test get_state_dot for expected results
        """
        state_dot = self.motion_model.get_state_dot(self.ego_state)
        self.assertEqual(state_dot.rear_axle, StateSE2(self.x_dot, self.y_dot, self.yaw_dot))
        self.assertEqual(state_dot.dynamic_car_state.rear_axle_velocity_2d, self.ego_state.dynamic_car_state.rear_axle_acceleration_2d)
        self.assertEqual(state_dot.dynamic_car_state.rear_axle_acceleration_2d, StateVector2D(0, 0))
        self.assertEqual(state_dot.tire_steering_angle, self.ego_state.dynamic_car_state.tire_steering_rate)

    def test_propagate_state(self) -> None:
        """
        Test propagate_state
        """
        state = self.motion_model.propagate_state(self.ego_state, self.ego_state.dynamic_car_state, self.sampling_time)
        self.assertEqual(state.rear_axle, StateSE2(forward_integrate(self.ego_state.rear_axle.x, self.x_dot, self.sampling_time), forward_integrate(self.ego_state.rear_axle.y, self.y_dot, self.sampling_time), forward_integrate(self.ego_state.rear_axle.heading, self.yaw_dot, self.sampling_time)))
        self.assertEqual(state.dynamic_car_state.rear_axle_velocity_2d, StateVector2D(forward_integrate(self.ego_state.dynamic_car_state.rear_axle_velocity_2d.x, self.ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, self.sampling_time), 0.0))
        self.assertEqual(state.dynamic_car_state.rear_axle_acceleration_2d, StateVector2D(0.1, 0.0))
        self.assertEqual(state.tire_steering_angle, forward_integrate(self.ego_state.tire_steering_angle, self.ego_state.dynamic_car_state.tire_steering_rate, self.sampling_time))
        self.assertEqual(state.dynamic_car_state.angular_velocity, state.dynamic_car_state.rear_axle_velocity_2d.x * np.tan(state.tire_steering_angle) / self.vehicle.wheel_base)

    def test_limit_steering_angle(self) -> None:
        """
        Test whether the KinematicBicycleModel correct enforces steering angle
        limits.
        """
        dynamic_car_state = DynamicCarState.build_from_rear_axle(self.vehicle.rear_axle_to_center, rear_axle_velocity_2d=StateVector2D(0.0, 0.0), rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_rate=10.0)
        car_footprint = CarFootprint.build_from_rear_axle(rear_axle_pose=StateSE2(x=0.0, y=0.0, heading=0.0), vehicle_parameters=self.vehicle)
        ego_state = EgoState(car_footprint, dynamic_car_state, tire_steering_angle=self.motion_model._max_steering_angle - 0.0001, is_in_auto_mode=True, time_point=TimePoint(0))
        propagated_state = self.motion_model.propagate_state(ego_state, dynamic_car_state, self.sampling_time)
        self.assertEqual(propagated_state.tire_steering_angle, self.motion_model._max_steering_angle)

    def test_update_command(self) -> None:
        """
        Test whether the update_command function performs as expected:
        1) returns same commands if time constants are set to zero (no delay)
        2) returns an smaller command (in the absolute sense) when filter is applied
        """
        dynamic_car_state = DynamicCarState.build_from_rear_axle(self.vehicle.rear_axle_to_center, rear_axle_velocity_2d=StateVector2D(0.0, 0.0), rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_rate=0.0)
        car_footprint = CarFootprint.build_from_rear_axle(rear_axle_pose=StateSE2(x=0.0, y=0.0, heading=0.0), vehicle_parameters=self.vehicle)
        state = EgoState(car_footprint, dynamic_car_state, tire_steering_angle=self.motion_model._max_steering_angle - 0.0001, is_in_auto_mode=True, time_point=TimePoint(0))
        ideal_dynamic_state = DynamicCarState.build_from_rear_axle(self.vehicle.rear_axle_to_center, rear_axle_velocity_2d=StateVector2D(0.0, 0.0), rear_axle_acceleration_2d=StateVector2D(1.0, 0.0), tire_steering_rate=0.5)
        no_delay_motion_model = KinematicBicycleModel(self.vehicle, accel_time_constant=0, steering_angle_time_constant=0)
        no_delay_propagating_state = no_delay_motion_model._update_commands(state, ideal_dynamic_state, self.sampling_time)
        self.assertEqual(round(no_delay_propagating_state.dynamic_car_state.rear_axle_acceleration_2d.x, 10), ideal_dynamic_state.rear_axle_acceleration_2d.x)
        self.assertEqual(round(no_delay_propagating_state.dynamic_car_state.tire_steering_rate, 10), ideal_dynamic_state.tire_steering_rate)
        propagating_state = self.motion_model._update_commands(state, ideal_dynamic_state, self.sampling_time)
        self.assertTrue(propagating_state.dynamic_car_state.rear_axle_acceleration_2d.x < ideal_dynamic_state.rear_axle_acceleration_2d.x)
        self.assertLess(propagating_state.dynamic_car_state.tire_steering_rate, ideal_dynamic_state.tire_steering_rate)

class SimplePlanner(AbstractPlanner):
    """
    Planner going straight.
    """

    def __init__(self, horizon_seconds: float, sampling_time: float, acceleration: npt.NDArray[np.float32], max_velocity: float=5.0, steering_angle: float=0.0):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param acceleration: [m/s^2] constant ego acceleration, till limited by max_velocity.
        :param max_velocity: [m/s] ego max velocity.
        :param steering_angle: [rad] ego steering angle.
        """
        self.horizon_seconds = TimePoint(int(horizon_seconds * 1000000.0))
        self.sampling_time = TimePoint(int(sampling_time * 1000000.0))
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        pass

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """
        history = current_input.history
        ego_state, _ = history.current_state
        state = EgoState(car_footprint=ego_state.car_footprint, dynamic_car_state=DynamicCarState.build_from_rear_axle(ego_state.car_footprint.rear_axle_to_center_dist, ego_state.dynamic_car_state.rear_axle_velocity_2d, self.acceleration), tire_steering_angle=self.steering_angle, is_in_auto_mode=True, time_point=ego_state.time_point)
        trajectory: List[EgoState] = [state]
        for _ in range(int(self.horizon_seconds.time_us / self.sampling_time.time_us)):
            if state.dynamic_car_state.speed > self.max_velocity:
                accel = self.max_velocity - state.dynamic_car_state.speed
                state = EgoState.build_from_rear_axle(rear_axle_pose=state.rear_axle, rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=StateVector2D(accel, 0), tire_steering_angle=state.tire_steering_angle, time_point=state.time_point, vehicle_parameters=state.car_footprint.vehicle_parameters, is_in_auto_mode=True, angular_vel=state.dynamic_car_state.angular_velocity, angular_accel=state.dynamic_car_state.angular_acceleration)
            state = self.motion_model.propagate_state(state, state.dynamic_car_state, self.sampling_time)
            trajectory.append(state)
        return InterpolatedTrajectory(trajectory)

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

class MockPoint(InterpolatableState, ABC):
    """Mock point for trajectory tests."""

    @staticmethod
    @static_vars(calls=[])
    def from_split_state(split_state: Mock) -> str:
        """Mock from_split_state."""
        MockPoint.from_split_state.calls.append(split_state)
        return 'foo'

    @staticmethod
    def reset_calls() -> None:
        """Resets spy."""
        MockPoint.from_split_state.calls = []

def sampled_tracked_objects_to_tensor_list(past_tracked_objects: List[TrackedObjects], object_type: TrackedObjectType=TrackedObjectType.VEHICLE) -> List[torch.Tensor]:
    """
    Tensorizes the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each tensor as described in `_extract_agent_tensor()`.
    :param past_tracked_objects: The tracked objects to tensorize.
    :param object_type: TrackedObjectType to filter agents by.
    :return: The tensorized objects.
    """
    output: List[torch.Tensor] = []
    track_token_ids: Dict[str, int] = {}
    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids = _extract_agent_tensor(past_tracked_objects[i], track_token_ids, object_type)
        output.append(tensorized)
    return output

def _extract_agent_tensor(tracked_objects: TrackedObjects, track_token_ids: Dict[str, int], object_type: TrackedObjectType) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Extracts the relevant data from the agents present in a past detection into a tensor.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a tensor as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a tensor.
    :track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :object_type: TrackedObjectType to filter agents by.
    :return: The generated tensor and the updated track_token_ids dict.
    """
    agents = tracked_objects.get_tracked_objects_of_type(object_type)
    output = torch.zeros((len(agents), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)
    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]
        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
    return (output, track_token_ids)

def _create_scene_object(token: str, object_type: TrackedObjectType) -> Agent:
    """
    :param token: a unique instance token
    :param object_type: agent type.
    :return: a random Agent
    """
    scene = SceneObject.make_random(token, object_type)
    return Agent(tracked_object_type=object_type, oriented_box=scene.box, velocity=StateVector2D(0, 0), metadata=SceneObjectMetadata(token=token, track_token=token, track_id=None, timestamp_us=0))

def _create_dummy_tracked_objects_tensor(num_frames: int) -> List[TrackedObjects]:
    """
    Generates some dummy tracked objects for use with testing the tensorization functions.
    :param num_frames: The number of frames for which to generate the objects.
    :return: The generated dummy objects.
    """
    test_tracked_objects = []
    for i in range(num_frames):
        num_agents_in_frame = i + 1
        num_non_agents_in_frame = num_frames + 1 - num_agents_in_frame
        objects_in_frame: List[TrackedObject] = []
        for j in range(num_agents_in_frame):
            objects_in_frame.append(Agent(tracked_object_type=TrackedObjectType.VEHICLE, oriented_box=OrientedBox(center=StateSE2(x=j + 6, y=j + 7, heading=j + 3), length=j + 5, width=j + 4, height=-1), velocity=StateVector2D(x=j + 1, y=j + 2), metadata=SceneObjectMetadata(timestamp_us=1, token=f'agent_{j}', track_id=f'agent_{j}', track_token=f'agent_{j}'), angular_velocity=-1, predictions=None, past_trajectory=None))
            objects_in_frame.append(Agent(tracked_object_type=TrackedObjectType.BICYCLE, oriented_box=OrientedBox(center=StateSE2(x=j + 6, y=j + 7, heading=j + 3), length=j + 5, width=j + 4, height=-1), velocity=StateVector2D(x=j + 1, y=j + 2), metadata=SceneObjectMetadata(timestamp_us=1, token=f'agent_{j}', track_id=f'agent_{j}', track_token=f'agent_{j}'), angular_velocity=-1, predictions=None, past_trajectory=None))
            objects_in_frame.append(Agent(tracked_object_type=TrackedObjectType.PEDESTRIAN, oriented_box=OrientedBox(center=StateSE2(x=j + 6, y=j + 7, heading=j + 3), length=j + 5, width=j + 4, height=-1), velocity=StateVector2D(x=j + 1, y=j + 2), metadata=SceneObjectMetadata(timestamp_us=1, token=f'agent_{j}', track_id=f'agent_{j}', track_token=f'agent_{j}'), angular_velocity=-1, predictions=None, past_trajectory=None))
        for j in range(num_non_agents_in_frame):
            jj = j + 100
            objects_in_frame.append(StaticObject(tracked_object_type=TrackedObjectType.GENERIC_OBJECT, oriented_box=OrientedBox(center=StateSE2(x=jj, y=jj, heading=jj), length=jj, width=jj, height=jj), metadata=SceneObjectMetadata(timestamp_us=jj, token=f'static_{jj}', track_id=f'static_{jj}', track_token=f'static_{jj}')))
        test_tracked_objects.append(TrackedObjects(objects_in_frame))
    return test_tracked_objects

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

class TestNuPlanScenario(unittest.TestCase):
    """
    Tests scenario for NuPlan
    """

    def _make_test_scenario(self) -> NuPlanScenario:
        """
        Creates a sample scenario to use for testing.
        """
        return NuPlanScenario(data_root='data_root/', log_file_load_path='data_root/log_name.db', initial_lidar_token=int_to_str_token(1234), initial_lidar_timestamp=2345, scenario_type='scenario_type', map_root='map_root', map_version='map_version', map_name='map_name', scenario_extraction_info=ScenarioExtractionInfo(scenario_name='scenario_name', scenario_duration=20, extraction_offset=1, subsample_ratio=0.5), ego_vehicle_parameters=get_pacifica_parameters(), sensor_root='sensor_root')

    def _get_sampled_sensor_tokens_in_time_window_patch(self, expected_log_file: str, expected_sensor_data_source: SensorDataSource, expected_start_timestamp: int, expected_end_timestamp: int, expected_subsample_step: int) -> Callable[[str, SensorDataSource, int, int, int], Generator[str, None, None]]:
        """
        Creates a patch for the get_sampled_lidarpc_tokens_in_time_window function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_start_timestamp: The expected start timestamp with which the function is expected to be called.
        :param expected_end_timestamp: The expected end timestamp with which the function is expected to be called.
        :param expected_subsample_step: The expected subsample step with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_sensor_data_source: SensorDataSource, actual_start_timestamp: int, actual_end_timestamp: int, actual_subsample_step: int) -> Generator[str, None, None]:
            """
            The patch function for get_sampled_lidarpc_tokens_in_time_window.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_start_timestamp, actual_start_timestamp)
            self.assertEqual(expected_end_timestamp, actual_end_timestamp)
            self.assertEqual(expected_subsample_step, actual_subsample_step)
            num_tokens = int((expected_end_timestamp - expected_start_timestamp) / (expected_subsample_step * 1000000.0))
            for token in range(num_tokens):
                yield int_to_str_token(token)
        return fxn

    def _get_download_file_if_necessary_patch(self, expected_data_root: str, expected_log_file_load_path: str) -> Callable[[str, str], str]:
        """
        Creates a patch for the download_file_if_necessary function that validates the arguments.
        :param expected_data_root: The data_root with which the function is expected to be called.
        :param expected_log_file_load_path: The log_file_load_path with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_data_root: str, actual_log_file_load_path: str) -> str:
            """
            The generated patch function.
            """
            self.assertEqual(expected_data_root, actual_data_root)
            self.assertEqual(expected_log_file_load_path, actual_log_file_load_path)
            return actual_log_file_load_path
        return fxn

    def _get_sensor_data_from_sensor_data_tokens_from_db_patch(self, expected_log_file: str, expected_sensor_data_source: SensorDataSource, expected_sensor_class: Type[SensorDataTableRow], expected_tokens: List[str]) -> Callable[[str, SensorDataSource, Type[SensorDataTableRow], List[str]], Generator[SensorDataTableRow, None, None]]:
        """
        Creates a patch for the get_images_from_lidar_tokens_patch function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_sensor_data_source: The sensor source with which the function is expected to be called.
        :param expected_sensor_class: The sensor class with which the function is expected to be called.
        :param expected_tokens: The tokens with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_sensor_data_source: SensorDataSource, actual_sensor_class: Type[SensorDataTableRow], actual_tokens: List[str]) -> Generator[SensorDataTableRow, None, None]:
            """
            The patch function for get_sensor_data_from_sensor_data_tokens_from_db.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_sensor_class, actual_sensor_class)
            self.assertEqual(expected_tokens, actual_tokens)
            lidar_token = actual_tokens[0]
            if expected_sensor_class == LidarPc:
                yield LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token))
            elif expected_sensor_class == ImageDBRow.Image:
                camera_token = str_token_to_int(lidar_token) + CAMERA_OFFSET
                yield ImageDBRow.Image(token=int_to_str_token(camera_token), next_token=int_to_str_token(camera_token), prev_token=int_to_str_token(camera_token), ego_pose_token=int_to_str_token(camera_token), camera_token=int_to_str_token(camera_token), filename_jpg=f'image_{camera_token}', timestamp=camera_token, channel=CameraChannel.CAM_R0.value)
            else:
                self.fail(f'Unexpected type: {expected_sensor_class}.')
        return fxn

    def _load_point_cloud_patch(self, expected_lidar_pc: LidarPc, expected_local_store: LocalStore, expected_s3_store: S3Store) -> Callable[[LidarPc, LocalStore, S3Store], LidarPointCloud]:
        """
        Creates a patch for the _load_point_cloud function that validates the arguments.
        :param expected_lidar_pc: The lidar pc with which the function is expected to be called.
        :param expected_local_store: The LocalStore with which the function is expected to be called.
        :param expected_s3_store: The S3Store with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_lidar_pc: LidarPc, actual_local_store: LocalStore, actual_s3_store: S3Store) -> LidarPointCloud:
            """
            The patch function for load_point_cloud.
            """
            self.assertEqual(expected_lidar_pc, actual_lidar_pc)
            self.assertEqual(expected_local_store, actual_local_store)
            self.assertEqual(expected_s3_store, actual_s3_store)
            return LidarPointCloud(np.eye(3))
        return fxn

    def _load_image_patch(self, expected_local_store: LocalStore, expected_s3_store: S3Store) -> Callable[[ImageDBRow.Image, LocalStore, S3Store], Image]:
        """
        Creates a patch for the _load_image_patch function and validates that argument is an Image object.
        :param expected_local_store: The LocalStore with which the function is expected to be called.
        :param expected_s3_store: The S3Store with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_image: ImageDBRow.Image, actual_local_store: LocalStore, actual_s3_store: S3Store) -> Image:
            """
            The patch function for load_image.
            """
            self.assertEqual(expected_local_store, actual_local_store)
            self.assertEqual(expected_s3_store, actual_s3_store)
            self.assertTrue(isinstance(actual_image, ImageDBRow.Image))
            return Image(PilImg.new('RGB', (500, 500)))
        return fxn

    def _get_images_from_lidar_tokens_patch(self, expected_log_file: str, expected_tokens: List[str], expected_channels: List[str], expected_lookahead_window_us: int, expected_lookback_window_us: int) -> Callable[[str, List[str], List[str], int, int], Generator[ImageDBRow.Image, None, None]]:
        """
        Creates a patch for the get_images_from_lidar_tokens_patch function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_tokens: The expected tokens with which the function is expected to be called.
        :param expected_channels: The expected channels with which the function is expected to be called.
        :param expected_lookahead_window_us: The expected lookahead window with which the function is expected to be called.
        :param expected_lookahead_window_us: The expected lookback window with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_tokens: List[str], actual_channels: List[str], actual_lookahead_window_us: int=50000, actual_lookback_window_us: int=50000) -> Generator[ImageDBRow.Image, None, None]:
            """
            The patch function for get_images_from_lidar_tokens.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_tokens, actual_tokens)
            self.assertEqual(expected_channels, actual_channels)
            self.assertEqual(expected_lookahead_window_us, actual_lookahead_window_us)
            self.assertEqual(expected_lookback_window_us, actual_lookback_window_us)
            for camera_token, channel in enumerate(actual_channels):
                if channel != LidarChannel.MERGED_PC.value:
                    yield ImageDBRow.Image(token=int_to_str_token(camera_token), next_token=int_to_str_token(camera_token), prev_token=int_to_str_token(camera_token), ego_pose_token=int_to_str_token(camera_token), camera_token=int_to_str_token(camera_token), filename_jpg=f'image_{camera_token}', timestamp=camera_token, channel=channel)
        return fxn

    def _get_sampled_lidarpcs_from_db_patch(self, expected_log_file: str, expected_initial_token: str, expected_sensor_data_source: SensorDataSource, expected_sample_indexes: Union[Generator[int, None, None], List[int]], expected_future: bool) -> Callable[[str, str, SensorDataSource, Union[Generator[int, None, None], List[int]], bool], Generator[LidarPc, None, None]]:
        """
        Creates a patch for the get_sampled_lidarpcs_from_db function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_initial_token: The initial token name with which the function is expected to be called.
        :param expected_sensor_data_source: The sensor source with which the function is expected to be called.
        :param expected_sample_indexes: The sample indexes with which the function is expected to be called.
        :param expected_future: The future with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_initial_token: str, actual_sensor_data_source: SensorDataSource, actual_sample_indexes: Union[Generator[int, None, None], List[int]], actual_future: bool) -> Generator[LidarPc, None, None]:
            """
            The patch function for get_images_from_lidar_tokens.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_initial_token, actual_initial_token)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_sample_indexes, actual_sample_indexes)
            self.assertEqual(expected_future, actual_future)
            for idx in actual_sample_indexes:
                lidar_token = int_to_str_token(idx)
                yield LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token))
        return fxn

    def test_implements_abstract_scenario_interface(self) -> None:
        """
        Tests that NuPlanScenario properly implements AbstractScenario interface.
        """
        assert_class_properly_implements_interface(AbstractScenario, NuPlanScenario)

    def test_token(self) -> None:
        """
        Tests that the token method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(int_to_str_token(1234), scenario.token)

    def test_log_name(self) -> None:
        """
        Tests that the log_name method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual('log_name', scenario.log_name)

    def test_scenario_name(self) -> None:
        """
        Tests that the scenario_name method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(int_to_str_token(1234), scenario.scenario_name)

    def test_ego_vehicle_parameters(self) -> None:
        """
        Tests that the ego_vehicle_parameters method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(get_pacifica_parameters(), scenario.ego_vehicle_parameters)

    def test_scenario_type(self) -> None:
        """
        Tests that the scenario_type method works properly
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual('scenario_type', scenario.scenario_type)

    def test_database_interval(self) -> None:
        """
        Tests that the database_interval method works properly
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(0.1, scenario.database_interval)

    def test_get_number_of_iterations(self) -> None:
        """
        Tests that the get_number_of_iterations method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn):
                scenario = self._make_test_scenario()
                self.assertEqual(10, scenario.get_number_of_iterations())

    def test_get_time_point(self) -> None:
        """
        Tests that the get_time_point method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        for iter_val in [0, 3, 5]:

            def token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch method for get_lidarpc_token_timstamp_from_db that validates the arguments.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(SensorDataSource(table='lidar_pc', sensor_table='lidar', sensor_token_column='lidar_token', channel='MergedPointCloud'), sensor_source)
                self.assertEqual(int_to_str_token(iter_val), token)
                return int(str_token_to_int(iter_val) + 5)
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.get_sensor_data_token_timestamp_from_db', token_timestamp_patch):
                scenario = self._make_test_scenario()
                self.assertEqual(iter_val + 5, scenario.get_time_point(iter_val).time_us)

    def test_get_tracked_objects_at_iteration(self) -> None:
        """
        Tests that the get_tracked_objects_at_iteration method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)
        for iter_val in [0, 2, 3]:

            def get_token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch for get_sensor_data_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(SensorDataSource(table='lidar_pc', sensor_table='lidar', sensor_token_column='lidar_token', channel='MergedPointCloud'), sensor_source)
                self.assertEqual(int_to_str_token(iter_val), token)
                return int(iter_val * 1000000.0)

            def tracked_objects_for_token_patch(log_file: str, token: str) -> Generator[TrackedObject, None, None]:
                """
                The patch for get_tracked_objects_for_lidarpc_token that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(int_to_str_token(iter_val), token)
                for idx in range(0, 4, 1):
                    box = OrientedBox(center=StateSE2(x=10, y=10, heading=10), length=10, width=10, height=10)
                    metadata = SceneObjectMetadata(token=int_to_str_token(idx + str_token_to_int(token)), track_token=int_to_str_token(idx + str_token_to_int(token) + 100), track_id=None, timestamp_us=0, category_name='foo')
                    if idx < 2:
                        yield Agent(tracked_object_type=TrackedObjectType.VEHICLE, oriented_box=box, velocity=StateVector2D(x=10, y=10), metadata=metadata)
                    else:
                        yield StaticObject(tracked_object_type=TrackedObjectType.CZONE_SIGN, oriented_box=box, metadata=metadata)

            def interpolate_future_waypoints_patch(waypoints: List[InterpolatableState], time_horizon: float, interval_s: float) -> List[Optional[InterpolatableState]]:
                """
                The patch for interpolate_future_waypoints that validates the arguments and generates fake data.
                """
                self.assertEqual(4, len(waypoints))
                self.assertEqual(0.5, interval_s)
                self.assertEqual(5, time_horizon)
                return waypoints

            def future_waypoints_for_agents_patch(log_file: str, agents_tokens: List[str], start_time: int, end_time: int) -> Generator[Tuple[str, Waypoint], None, None]:
                """
                The patch for get_future_waypoints_for_agents_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(iter_val * 1000000.0, start_time)
                self.assertEqual((iter_val + 5.5) * 1000000.0, end_time)
                self.assertEqual(2, len(agents_tokens))
                check_tokens = [str_token_to_int(t) for t in agents_tokens]
                check_tokens.sort()
                self.assertEqual(iter_val + 100, check_tokens[0])
                self.assertEqual(iter_val + 100 + 1, check_tokens[1])
                for i in range(8):
                    waypoint = Waypoint(time_point=TimePoint(time_us=i), oriented_box=OrientedBox(center=StateSE2(x=i, y=i, heading=i), length=i, width=i, height=i), velocity=None)
                    token = check_tokens[0] if i < 4 else check_tokens[1]
                    yield (int_to_str_token(token), waypoint)
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_for_lidarpc_token_from_db', tracked_objects_for_token_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db', future_waypoints_for_agents_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sensor_data_token_timestamp_from_db', get_token_timestamp_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.interpolate_future_waypoints', interpolate_future_waypoints_patch):
                scenario = self._make_test_scenario()
                agents = scenario.get_tracked_objects_at_iteration(iter_val, ground_truth_predictions)
                objects = agents.tracked_objects.tracked_objects
                self.assertEqual(4, len(objects))
                objects.sort(key=lambda x: str_token_to_int(x.metadata.token))
                for i in range(0, 2, 1):
                    test_obj = objects[i]
                    self.assertTrue(isinstance(test_obj, Agent))
                    self.assertEqual(iter_val + i, str_token_to_int(test_obj.metadata.token))
                    self.assertEqual(iter_val + i + 100, str_token_to_int(test_obj.metadata.track_token))
                    self.assertEqual(TrackedObjectType.VEHICLE, test_obj.tracked_object_type)
                    self.assertIsNotNone(test_obj.predictions)
                    object_waypoints = test_obj.predictions[0].waypoints
                    self.assertEqual(4, len(object_waypoints))
                    for j in range(len(object_waypoints)):
                        self.assertEqual(j + i * len(object_waypoints), object_waypoints[j].x)
                for i in range(2, 4, 1):
                    test_obj = objects[i]
                    self.assertTrue(isinstance(test_obj, StaticObject))
                    self.assertEqual(iter_val + i, str_token_to_int(test_obj.metadata.token))
                    self.assertEqual(iter_val + i + 100, str_token_to_int(test_obj.metadata.track_token))
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, test_obj.tracked_object_type)

    def test_get_tracked_objects_within_time_window_at_iteration(self) -> None:
        """
        Tests that the get_tracked_objects_within_time_window_at_iteration method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)
        for iter_val in [3, 4]:

            def get_token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch for get_sensor_data_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(SensorDataSource(table='lidar_pc', sensor_table='lidar', sensor_token_column='lidar_token', channel='MergedPointCloud'), sensor_source)
                self.assertEqual(int_to_str_token(iter_val), token)
                return int(iter_val * 1000000.0)

            def tracked_objects_within_time_interval_patch(log_file: str, start_timestamp: int, end_timestamp: int, filter_tokens: Optional[Set[str]]) -> Generator[TrackedObject, None, None]:
                """
                The patch for get_tracked_objects_for_lidarpc_token that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual((iter_val - 2) * 1000000.0, start_timestamp)
                self.assertEqual((iter_val + 2) * 1000000.0, end_timestamp)
                self.assertIsNone(filter_tokens)
                for time_idx in range(-2, 3, 1):
                    for idx in range(0, 4, 1):
                        box = OrientedBox(center=StateSE2(x=10, y=10, heading=10), length=10, width=10, height=10)
                        metadata = SceneObjectMetadata(token=int_to_str_token(idx + iter_val), track_token=int_to_str_token(idx + iter_val + 100), track_id=None, timestamp_us=(iter_val + time_idx) * 1000000.0, category_name='foo')
                        if idx < 2:
                            yield Agent(tracked_object_type=TrackedObjectType.VEHICLE, oriented_box=box, velocity=StateVector2D(x=10, y=10), metadata=metadata)
                        else:
                            yield StaticObject(tracked_object_type=TrackedObjectType.CZONE_SIGN, oriented_box=box, metadata=metadata)

            def interpolate_future_waypoints_patch(waypoints: List[InterpolatableState], time_horizon: float, interval_s: float) -> List[Optional[InterpolatableState]]:
                """
                The patch for interpolate_future_waypoints that validates the arguments and generates fake data.
                """
                self.assertEqual(4, len(waypoints))
                self.assertEqual(0.5, interval_s)
                self.assertEqual(5, time_horizon)
                return waypoints

            def future_waypoints_for_agents_patch(log_file: str, agents_tokens: List[str], start_time: int, end_time: int) -> Generator[Tuple[str, Waypoint], None, None]:
                """
                The patch for get_future_waypoints_for_agents_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(end_time - start_time, 5.5 * 1000000.0)
                self.assertEqual(2, len(agents_tokens))
                check_tokens = [str_token_to_int(t) for t in agents_tokens]
                check_tokens.sort()
                self.assertEqual(iter_val + 100, check_tokens[0])
                self.assertEqual(iter_val + 100 + 1, check_tokens[1])
                for i in range(8):
                    waypoint = Waypoint(time_point=TimePoint(time_us=i), oriented_box=OrientedBox(center=StateSE2(x=i, y=i, heading=i), length=i, width=i, height=i), velocity=None)
                    token = check_tokens[0] if i < 4 else check_tokens[1]
                    yield (int_to_str_token(token), waypoint)
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_within_time_interval_from_db', tracked_objects_within_time_interval_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db', future_waypoints_for_agents_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sensor_data_token_timestamp_from_db', get_token_timestamp_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.interpolate_future_waypoints', interpolate_future_waypoints_patch):
                scenario = self._make_test_scenario()
                agents = scenario.get_tracked_objects_within_time_window_at_iteration(iter_val, 2, 2, future_trajectory_sampling=ground_truth_predictions)
                objects = agents.tracked_objects.tracked_objects
                self.assertEqual(20, len(objects))
                num_objects = 2
                for window in range(0, 5, 1):
                    for object_num in range(0, 2, 1):
                        start_agent_idx = window * 2
                        test_obj = objects[start_agent_idx + object_num]
                        self.assertTrue(isinstance(test_obj, Agent))
                        self.assertEqual(iter_val + object_num, str_token_to_int(test_obj.metadata.token))
                        self.assertEqual(iter_val + object_num + 100, str_token_to_int(test_obj.metadata.track_token))
                        self.assertEqual(TrackedObjectType.VEHICLE, test_obj.tracked_object_type)
                        self.assertIsNotNone(test_obj.predictions)
                        object_waypoints = test_obj.predictions[0].waypoints
                        self.assertEqual(4, len(object_waypoints))
                        for j in range(len(object_waypoints)):
                            self.assertEqual(j + object_num * len(object_waypoints), object_waypoints[j].x)
                        start_obj_idx = 10 + window * 2
                        test_obj = objects[start_obj_idx + object_num]
                        self.assertTrue(isinstance(test_obj, StaticObject))
                        self.assertEqual(iter_val + object_num + num_objects, str_token_to_int(test_obj.metadata.token))
                        self.assertEqual(iter_val + object_num + num_objects + 100, str_token_to_int(test_obj.metadata.track_token))
                        self.assertEqual(TrackedObjectType.CZONE_SIGN, test_obj.tracked_object_type)

    def test_nuplan_scenario_memory_usage(self) -> None:
        """
        Test that repeatedly creating and destroying nuplan scenario does not cause memory leaks.
        """
        starting_usage = 0
        ending_usage = 0
        num_iterations = 5
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            hpy = guppy.hpy()
            hpy.setrelheap()
            for i in range(0, num_iterations, 1):
                scenario = self._make_test_scenario()
                _ = scenario.token
                gc.collect()
                heap = hpy.heap()
                _ = heap.size
                if i == num_iterations - 2:
                    starting_usage = heap.size
                if i == num_iterations - 1:
                    ending_usage = heap.size
            memory_difference_in_mb = (ending_usage - starting_usage) / (1024 * 1024)
            max_allowable_growth_mb = max(0.1, 0.1 * starting_usage / (1024 * 1024))
            self.assertGreater(max_allowable_growth_mb, memory_difference_in_mb)

    @patch(f'{TEST_PATH}.LocalStore', autospec=True)
    @patch(f'{TEST_PATH}.S3Store', autospec=True)
    @patch(f'{TEST_PATH}.os.getenv')
    def test_get_sensors_at_iteration(self, mock_get_env: Mock, mock_s3_store: Mock, mock_local_store: Mock) -> None:
        """Test get_sensors_at_iteration."""
        mock_url = 'url'
        mock_get_env.side_effect = ['s3', mock_url]
        mock_s3_store.return_value = Mock(spec_set=S3Store)
        mock_local_store.return_value = Mock(spec_set=LocalStore)
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0) + 2345, expected_end_timestamp=int(21 * 1000000.0) + 2345, expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch(f'{TEST_PATH}.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
        for iter_val in [0, 3, 5]:
            lidar_token = int_to_str_token(iter_val)
            get_sensor_data_from_sensor_data_tokens_from_db_fxn = self._get_sensor_data_from_sensor_data_tokens_from_db_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_sensor_class=LidarPc, expected_tokens=[lidar_token])
            get_images_from_lidar_tokens_fxn = self._get_images_from_lidar_tokens_patch(expected_log_file='data_root/log_name.db', expected_tokens=[lidar_token], expected_channels=[CameraChannel.CAM_R0.value, LidarChannel.MERGED_PC.value], expected_lookahead_window_us=50000, expected_lookback_window_us=50000)
            load_lidar_fxn = self._load_point_cloud_patch(LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token)), mock_local_store.return_value, mock_s3_store.return_value)
            load_image_fxn = self._load_image_patch(mock_local_store.return_value, mock_s3_store.return_value)
            with mock.patch(f'{TEST_PATH_UTILS}.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch(f'{TEST_PATH}.get_sensor_data_from_sensor_data_tokens_from_db', get_sensor_data_from_sensor_data_tokens_from_db_fxn), mock.patch(f'{TEST_PATH}.get_images_from_lidar_tokens', get_images_from_lidar_tokens_fxn), mock.patch(f'{TEST_PATH}.load_point_cloud', load_lidar_fxn), mock.patch(f'{TEST_PATH}.load_image', load_image_fxn):
                sensors = scenario.get_sensors_at_iteration(iter_val, [CameraChannel.CAM_R0, LidarChannel.MERGED_PC])
                self.assertEqual(LidarChannel.MERGED_PC, list(sensors.pointcloud.keys())[0])
                self.assertEqual(CameraChannel.CAM_R0, list(sensors.images.keys())[0])
                mock_local_store.assert_called_with('sensor_root')
                mock_s3_store.assert_called_with(f'{mock_url}/sensor_blobs', show_progress=True)

    @patch(f'{TEST_PATH}.LocalStore', autospec=True)
    @patch(f'{TEST_PATH}.S3Store', autospec=True)
    @patch(f'{TEST_PATH}.os.getenv')
    def test_get_past_sensors(self, mock_get_env: Mock, mock_s3_store: Mock, mock_local_store: Mock) -> None:
        """Test get_past_sensors."""
        mock_url = 'url'
        mock_get_env.side_effect = ['s3', mock_url]
        mock_s3_store.return_value = Mock(spec_set=S3Store)
        mock_local_store.return_value = Mock(spec_set=LocalStore)
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        lidar_token = int_to_str_token(9)
        get_sampled_lidarpcs_from_db_fxn = self._get_sampled_lidarpcs_from_db_patch(expected_log_file='data_root/log_name.db', expected_initial_token=int_to_str_token(0), expected_sensor_data_source=get_lidarpc_sensor_data(), expected_sample_indexes=[9], expected_future=False)
        get_images_from_lidar_tokens_fxn = self._get_images_from_lidar_tokens_patch(expected_log_file='data_root/log_name.db', expected_tokens=[lidar_token], expected_channels=[CameraChannel.CAM_R0.value, LidarChannel.MERGED_PC.value], expected_lookahead_window_us=50000, expected_lookback_window_us=50000)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        load_lidar_fxn = self._load_point_cloud_patch(LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token)), mock_local_store.return_value, mock_s3_store.return_value)
        load_image_fxn = self._load_image_patch(mock_local_store.return_value, mock_s3_store.return_value)
        with mock.patch(f'{TEST_PATH}.download_file_if_necessary', download_file_patch_fxn), mock.patch(f'{TEST_PATH_UTILS}.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch(f'{TEST_PATH}.get_sampled_lidarpcs_from_db', get_sampled_lidarpcs_from_db_fxn), mock.patch(f'{TEST_PATH}.get_images_from_lidar_tokens', get_images_from_lidar_tokens_fxn), mock.patch(f'{TEST_PATH}.load_point_cloud', load_lidar_fxn), mock.patch(f'{TEST_PATH}.load_image', load_image_fxn):
            scenario = self._make_test_scenario()
            past_sensors = list(scenario.get_past_sensors(iteration=0, time_horizon=0.4, num_samples=1, channels=[CameraChannel.CAM_R0, LidarChannel.MERGED_PC]))
            self.assertEqual(1, len(past_sensors))
            self.assertEqual(LidarChannel.MERGED_PC, list(past_sensors[0].pointcloud.keys())[0])
            self.assertEqual(CameraChannel.CAM_R0, list(past_sensors[0].images.keys())[0])
            mock_local_store.assert_called_with('sensor_root')
            mock_s3_store.assert_called_with(f'{mock_url}/sensor_blobs', show_progress=True)

    @patch(f'{TEST_PATH}.download_file_if_necessary', Mock())
    @patch(f'{TEST_PATH}.absolute_path_to_log_name', Mock())
    @patch(f'{TEST_PATH}.get_images_from_lidar_tokens', Mock(return_value=[]))
    @patch(f'{TEST_PATH}.NuPlanScenario._find_matching_lidar_pcs')
    @patch(f'{TEST_PATH}.load_point_cloud')
    @patch(f'{TEST_PATH}.load_image')
    def test_get_past_sensors_no_channels(self, mock_load_image: Mock, mock_load_point_cloud: Mock, mock__find_matching_lidar_pcs: Mock) -> None:
        """Test get_past_sensors when no channels are passed."""
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.token = 'token'
        mock_load_point_cloud.return_value = Mock(spec_set=LidarPointCloud)
        mock__find_matching_lidar_pcs.return_value = iter([mock_lidar_pc])
        scenario = self._make_test_scenario()
        past_sensors = list(scenario.get_past_sensors(iteration=0, time_horizon=0.4, num_samples=1, channels=None))
        mock__find_matching_lidar_pcs.assert_called_once()
        mock_load_point_cloud.assert_called_once()
        mock_load_image.assert_not_called()
        self.assertIsNone(past_sensors[0].images)
        self.assertIsNotNone(past_sensors[0].pointcloud)

    @patch(f'{TEST_PATH}.download_file_if_necessary', Mock())
    @patch(f'{TEST_PATH}.absolute_path_to_log_name', Mock())
    @patch(f'{TEST_PATH}.get_images_from_lidar_tokens', Mock(return_value=[]))
    @patch(f'{TEST_PATH}.extract_sensor_tokens_as_scenario', Mock(return_value=[None]))
    @patch(f'{TEST_PATH}.get_sensor_data_from_sensor_data_tokens_from_db')
    @patch(f'{TEST_PATH}.load_point_cloud')
    @patch(f'{TEST_PATH}.load_image')
    def test_get_sensors_at_iteration_no_channels(self, mock_load_image: Mock, mock_load_point_cloud: Mock, mock_get_sensor_data_from_sensor_data_tokens_from_db: Mock) -> None:
        """Test get_past_sensors when no channels are passed."""
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.token = 'token'
        mock_load_point_cloud.return_value = Mock(spec_set=LidarPointCloud)
        mock_get_sensor_data_from_sensor_data_tokens_from_db.return_value = iter([mock_lidar_pc])
        scenario = self._make_test_scenario()
        sensors = scenario.get_sensors_at_iteration(iteration=0, channels=None)
        mock_get_sensor_data_from_sensor_data_tokens_from_db.assert_called_once()
        mock_load_point_cloud.assert_called_once()
        mock_load_image.assert_not_called()
        self.assertIsNone(sensors.images)
        self.assertIsNotNone(sensors.pointcloud)

class SimplePlanner(AbstractPlanner):
    """
    Planner going straight
    """

    def __init__(self, horizon_seconds: float, sampling_time: float, acceleration: npt.NDArray[np.float32], max_velocity: float=5.0, steering_angle: float=0.0):
        self.horizon_seconds = TimePoint(int(horizon_seconds * 1000000.0))
        self.sampling_time = TimePoint(int(sampling_time * 1000000.0))
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """ Inherited, see superclass. """
        pass

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """ Inherited, see superclass. """
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> List[AbstractTrajectory]:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """
        iteration = current_input.iteration
        history = current_input.history
        ego_state = history.ego_states[-1]
        state = EgoState(car_footprint=ego_state.car_footprint, dynamic_car_state=DynamicCarState.build_from_rear_axle(ego_state.car_footprint.rear_axle_to_center_dist, ego_state.dynamic_car_state.rear_axle_velocity_2d, self.acceleration), tire_steering_angle=self.steering_angle, is_in_auto_mode=True, time_point=ego_state.time_point)
        trajectory: List[EgoState] = [state]
        for _ in np.arange(iteration.time_us + self.sampling_time.time_us, iteration.time_us + self.horizon_seconds.time_us, self.sampling_time.time_us):
            if state.dynamic_car_state.speed > self.max_velocity:
                accel = self.max_velocity - state.dynamic_car_state.speed
                state = EgoState.build_from_rear_axle(rear_axle_pose=state.rear_axle, rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=StateVector2D(accel, 0), tire_steering_angle=state.tire_steering_angle, time_point=state.time_point, vehicle_parameters=state.car_footprint.vehicle_parameters, is_in_auto_mode=True, angular_vel=state.dynamic_car_state.angular_velocity, angular_accel=state.dynamic_car_state.angular_acceleration)
            state = self.motion_model.propagate_state(state, state.dynamic_car_state, self.sampling_time)
            trajectory.append(state)
        return InterpolatedTrajectory(trajectory)

