# Cluster 11

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

class TestIterableLidarBox(unittest.TestCase):
    """Tests the IterableLidarBox class and it's methods"""

    def test_IterableLidarBox_init(self) -> None:
        """Checks the correctness of the `IterableLidarBox` class' constructor."""
        box = Mock()
        iterable = IterableLidarBox(box)
        box.get_box_items_to_iterate.assert_called_once()
        self.assertEqual(iterable._begin, box)
        self.assertEqual(iterable.box, box)
        self.assertEqual(iterable._current, box)
        self.assertEqual(iterable._reverse, False)
        self.assertEqual(iterable._items_dict, box.get_box_items_to_iterate.return_value)

    def test_IterableLidarBox_iter(self) -> None:
        """Checks the correctness of the `IterableLidarBox` class' `__iter__` method."""
        box = Mock()
        iterable = IterableLidarBox(box)
        iter_val = iter(iterable)
        self.assertEqual(iterable, iter_val)

    def test_IterableLidarBox_next_not_end(self) -> None:
        """
        Checks the correctness of the `IterableLidarBox` class' `__next__` method.
         When the current box is not the end box, the current box should be returned
         and the `.next` box should become the current one.
        """
        box = Mock()
        box.timestamp = 1
        box.get_box_items_to_iterate.return_value = {1: (Mock(), Mock())}
        iterable = IterableLidarBox(box)
        result = next(iterable)
        self.assertEqual(result, box)
        self.assertEqual(iterable._current, box.get_box_items_to_iterate.return_value[1][1])

    def test_IterableLidarBox_getitem(self) -> None:
        """
        Checks the correctness of the `IterableLidarBox` class' `__getitem__` method.
        Should return the box at the given index.
        """
        box = Mock()
        box.timestamp = 1
        box.get_box_items_to_iterate.return_value = {1: (Mock(), Mock())}
        iterable = IterableLidarBox(box)
        box_0 = iterable[0]
        self.assertEqual(box_0, box)

