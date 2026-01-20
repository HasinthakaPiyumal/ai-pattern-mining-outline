# Cluster 13

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

def get_future_box_sequence(lidar_pcs: List[LidarPc], frame: Frame, future_horizon_len_s: float, future_interval_s: float, extrapolation_threshold_ms: float=100000.0, trans_matrix_ego: Optional[npt.NDArray[np.float64]]=None, trans_matrix_sensor: Optional[npt.NDArray[np.float64]]=None) -> Dict[str, List[Box3D]]:
    """
    Get a mapping from track token to box sequence over time for each box in the input data. Box
    annotations are sampled at a frequency of 20Hz.
    :param lidar_pcs: List of LidarPc.
    :param frame: An enumeration of Frame (global/vehicle/sensor).
    :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
    :param future_interval_s: Timestamp interval of the future waypoints in seconds.
    :param extrapolation_threshold_ms: If a target interpolation timestamp extends beyond the timestamp of the
        last recorded bounding box for an actor, then the values for the box position at the target timestamp will only
        be extrapolated if the target timestamp is within the specified number of microseconds of the last recorded
        bounding box. Otherwise the box at the target timestamp will be set to None.
    :param trans_matrix_ego:
        Transformation matrix to transform the boxes from the global frame to the ego-vehicle frame.
    :param trans_matrix_sensor:
        Transformation matrix to transform the boxes from the ego-vehicle frame to the sensor frame.
    :return: Mapping from track token to list of corresponding boxes at each timestamp in the global coordinate
        frame, where the first box corresponds to the current timestamp t0.
    """
    if frame == Frame.VEHICLE:
        assert trans_matrix_ego is not None
    if frame == Frame.SENSOR:
        assert trans_matrix_ego is not None
        assert trans_matrix_sensor is not None
    num_future_boxes = int(future_horizon_len_s / future_interval_s)
    num_target_timestamps = num_future_boxes + 1
    future_horizon_len_ms = future_horizon_len_s * 1000000.0
    start_timestamp = lidar_pcs[0].timestamp
    tracks_dict = {lidar_box.track_token: [lidar_box.box()] for lidar_box in lidar_pcs[0].lidar_boxes}
    timestamps = [lidar_pc.timestamp for lidar_pc in lidar_pcs]
    tracks_dict = add_future_boxes(tracks_dict, lidar_pcs)
    target_timestamps: Union[npt.NDArray[np.float64], List[float]] = np.linspace(start=start_timestamp, stop=start_timestamp + future_horizon_len_ms, num=num_target_timestamps)
    for track_token, track in tracks_dict.items():
        last_box_index = get_last_box_index(box_sequence=track)
        if last_box_index == 0:
            tracks_dict[track_token] = [track[0]] + [None] * num_future_boxes
            continue
        last_box_timestamp = timestamps[last_box_index]
        target_timestamps = [t for t in target_timestamps if t <= last_box_timestamp + extrapolation_threshold_ms]
        box_indices = [i for i, box in enumerate(track) if box is not None]
        interpolated_boxes: List[Optional[Box3D]] = []
        interpolated_boxes.extend(interpolate_boxes(target_timestamps=target_timestamps, timestamps=np.array([float(timestamps[i]) for i in box_indices]), box_sequence=[track[i] for i in box_indices]))
        if frame in [Frame.VEHICLE, Frame.SENSOR]:
            for box in interpolated_boxes:
                if box is not None:
                    box.transform(trans_matrix_ego)
        if frame == Frame.SENSOR:
            for box in interpolated_boxes:
                if box is not None:
                    box.transform(trans_matrix_sensor)
        num_missing_final_boxes = num_target_timestamps - len(interpolated_boxes)
        if num_missing_final_boxes:
            interpolated_boxes.extend([None] * num_missing_final_boxes)
        tracks_dict[track_token] = interpolated_boxes
    return tracks_dict

def add_future_boxes(tracks_dict: Dict[str, List[Box3D]], lidar_pcs: List[LidarPc]) -> Dict[str, List[Box3D]]:
    """
    Iterate over future samples, adding boxes to the box sequence associated with each track token
    :param tracks_dict: Dictionary of boxes associated with track tokens.
    :param lidar_pcs: List of LidarPc.
    :return: Updated Dictionary that includes future boxes.
    """
    for idx, next_lidar_pc in enumerate(lidar_pcs[1:]):
        next_lidar_boxes = next_lidar_pc.lidar_boxes
        for lidar_box in next_lidar_boxes:
            track_token = lidar_box.track_token
            if track_token not in tracks_dict:
                continue
            tracks_dict[track_token].append(lidar_box.box())
        for track_token, track in tracks_dict.items():
            if len(track) < idx + 1:
                tracks_dict[track_token].append(None)
    return tracks_dict

def pack_future_boxes(track_token_2_box_sequence: Dict[str, List[Box3D]], future_horizon_len_s: float, future_interval_s: float) -> List[Box3D]:
    """
    Given a mapping from all the track tokens to the list of corresponding boxes at each future
        timestamp, this function packs the "future" data into the individual Box3D boxes in the current Sample,
        such that each box contains its future center positions and future orientations in subsequent frames.
    :param track_token_2_box_sequence: Mapping from track token to list of corresponding boxes at each timestamp
        in the global coordinate frame, returned by function get_future_box_sequence()
    :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
    :param future_interval_s: Timestamp interval of the future waypoints in seconds.
    :return: List of boxes in a frame, where each box contains future center positions and future orientations in
        subsequent frames.
    """
    boxes_out: List[Box3D] = []
    for track_token, box_sequence in track_token_2_box_sequence.items():
        current_box = box_sequence[0]
        future_centers = [[box.center if box else (np.nan, np.nan, np.nan) for box in box_sequence[1:]]]
        future_orientations = [[box.orientation if box else None for box in box_sequence[1:]]]
        mode_probs = [1.0]
        box_with_future = Box3D(center=current_box.center, size=current_box.size, orientation=current_box.orientation, label=current_box.label, score=current_box.score, velocity=current_box.velocity, angular_velocity=current_box.angular_velocity, payload=current_box.payload, token=current_box.token, track_token=current_box.track_token, future_horizon_len_s=future_horizon_len_s, future_interval_s=future_interval_s, future_centers=future_centers, future_orientations=future_orientations, mode_probs=mode_probs)
        boxes_out.append(box_with_future)
    return boxes_out

class LidarPc(Base):
    """
    A lidar point cloud.
    """
    __tablename__ = 'lidar_pc'
    token = Column(sql_types.HexLen8, primary_key=True)
    next_token = Column(sql_types.HexLen8, ForeignKey('lidar_pc.token'), nullable=True)
    prev_token = Column(sql_types.HexLen8, ForeignKey('lidar_pc.token'), nullable=True)
    ego_pose_token = Column(sql_types.HexLen8, ForeignKey('ego_pose.token'), nullable=False)
    lidar_token = Column(sql_types.HexLen8, ForeignKey('lidar.token'), nullable=False)
    scene_token = Column(sql_types.HexLen8, ForeignKey('scene.token'), nullable=False)
    filename = Column(String(128))
    timestamp = Column(Integer)
    next = relationship('LidarPc', foreign_keys=[next_token], remote_side=[token])
    prev = relationship('LidarPc', foreign_keys=[prev_token], remote_side=[token])
    ego_pose = relationship('EgoPose', foreign_keys=[ego_pose_token], back_populates='lidar_pc')
    scene = relationship('Scene', foreign_keys=[scene_token], back_populates='lidar_pcs')
    lidar_boxes = relationship('LidarBox', foreign_keys='LidarBox.lidar_pc_token', back_populates='lidar_pc')

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def log(self) -> Log:
        """
        Returns the Log containing the LidarPC.
        :return: The log containing the LidarPC.
        """
        return self.lidar.log

    def future_ego_pose(self) -> Optional[EgoPose]:
        """
        Get future ego poses.
        :return: Ego pose at next pointcloud if any.
        """
        if self.next is not None:
            return self.next.ego_pose
        return None

    def past_ego_pose(self) -> Optional[EgoPose]:
        """
        Get past ego poses.
        :return: Ego pose at previous pointcloud if any.
        """
        if self.prev is not None:
            return self.prev.ego_pose
        return None

    def future_or_past_ego_poses(self, number: int, mode: str, direction: str) -> List[EgoPose]:
        """
        Get n future or past vehicle poses. Note here the frequency of pose differs from frequency of LidarPc.
        :param number: Number of poses to fetch or number of seconds of ego poses to fetch.
        :param mode: Either n_poses or n_seconds.
        :param direction: Future or past ego poses to fetch, could be 'prev' or 'next'.
        :return: List of up to n or n seconds future or past ego poses.
        """
        if direction == 'prev':
            if mode == 'n_poses':
                return self._session.query(EgoPose).filter(EgoPose.timestamp < self.ego_pose.timestamp, self.lidar.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.desc()).limit(number).all()
            elif mode == 'n_seconds':
                return self._session.query(EgoPose).filter(EgoPose.timestamp - self.ego_pose.timestamp < 0, EgoPose.timestamp - self.ego_pose.timestamp >= -number * 1000000.0, self.lidar.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.desc()).all()
            else:
                raise ValueError(f'Unknown mode: {mode}.')
        elif direction == 'next':
            if mode == 'n_poses':
                return self._session.query(EgoPose).filter(EgoPose.timestamp > self.ego_pose.timestamp, self.lidar.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.asc()).limit(number).all()
            elif mode == 'n_seconds':
                return self._session.query(EgoPose).filter(EgoPose.timestamp - self.ego_pose.timestamp > 0, EgoPose.timestamp - self.ego_pose.timestamp <= number * 1000000.0, self.lidar.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.asc()).all()
            else:
                raise ValueError(f'Unknown mode: {mode}.')
        else:
            raise ValueError(f'Unknown direction: {direction}.')

    def load(self, db: NuPlanDB, remove_close: bool=True) -> LidarPointCloud:
        """
        Load a point cloud.
        :param db: Log Database.
        :param remove_close: If true, remove nearby points, defaults to True.
        :return: Loaded point cloud.
        """
        if self.lidar.channel == 'MergedPointCloud':
            if self.filename.endswith('bin2'):
                return LidarPointCloud.from_buffer(self.load_bytes(db), 'bin2')
            else:
                assert self.filename.endswith('pcd'), f'.pcd file is expected but get {self.filename}'
                return LidarPointCloud.from_buffer(self.load_bytes(db), 'pcd')
        else:
            raise NotImplementedError

    def load_bytes(self, db: NuPlanDB) -> BinaryIO:
        """
        Load the point cloud in binary.
        :param db: Log Database.
        :return: Point cloud bytes.
        """
        blob: BinaryIO = db.load_blob(os.path.join('sensor_blobs', self.filename))
        return blob

    def path(self, db: NuPlanDB) -> str:
        """
        Get the path to the point cloud file.
        :param db: Log Database.
        :return: Point cloud file path.
        """
        self.load_bytes(db)
        return osp.join(db.data_root, self.filename)

    def boxes(self, frame: Frame=Frame.GLOBAL) -> List[Box3D]:
        """
        Loads all boxes associated with this LidarPc record. Boxes are returned in the global frame by default.
        :param frame: Specify the frame in which the boxes will be returned.
        :return: The list of boxes.
        """
        boxes: List[Box3D] = get_boxes(self, frame, self.ego_pose.trans_matrix_inv, self.lidar.trans_matrix_inv)
        return boxes

    def boxes_with_future_waypoints(self, future_horizon_len_s: float, future_interval_s: float, frame: Frame=Frame.GLOBAL) -> List[Box3D]:
        """
        Loads all boxes and future boxes associated with this LidarPc record. Boxes are returned in the global frame by
            default and annotations are sampled at a frequency of ~0.5 seconds.
        :param future_horizon_len_s: Timestep horizon of the future waypoints in seconds.
        :param future_interval_s: Timestep interval of the future waypoints in seconds.
        :param frame: Specify the frame in which the boxes will be returned.
        :return: List of boxes in sample data that includes box centers and orientations at future timesteps.
        """
        TIMESTAMP_MARGIN_MS = 1000000.0
        future_horizon_len_ms = future_horizon_len_s * 1000000.0
        query = self._session.query(LidarPc).filter(LidarPc.timestamp - self.timestamp >= 0, LidarPc.timestamp - self.timestamp <= future_horizon_len_ms + TIMESTAMP_MARGIN_MS).order_by(LidarPc.timestamp.asc()).all()
        lidar_pcs = [lidar_pc for lidar_pc in list(query)]
        track_token_2_box_sequence = get_future_box_sequence(lidar_pcs=lidar_pcs, frame=frame, future_horizon_len_s=future_horizon_len_s, future_interval_s=future_interval_s, trans_matrix_ego=self.ego_pose.trans_matrix_inv, trans_matrix_sensor=self.lidar.trans_matrix_inv)
        boxes_with_future_waypoints: List[Box3D] = pack_future_boxes(track_token_2_box_sequence=track_token_2_box_sequence, future_interval_s=future_interval_s, future_horizon_len_s=future_horizon_len_s)
        return boxes_with_future_waypoints

    def render(self, db: NuPlanDB, render_future_waypoints: bool=False, render_map_raster: bool=False, render_vector_map: bool=False, render_track_color: bool=False, render_future_ego_poses: bool=False, track_token: Optional[str]=None, with_anns: bool=True, axes_limit: float=80.0, ax: Axes=None) -> plt.axes:
        """
        Render the Lidar pointcloud with appropriate boxes and (optionally) the map raster.
        :param db: Log database.
        :param render_future_waypoints: Whether to render future waypoints.
        :param render_map_raster: Whether to render the map raster.
        :param render_vector_map: Whether to render the vector map.
        :param render_track_color: Whether to render the tracks with different random color.
        :param render_future_ego_poses: Whether to render future ego poses.
        :param track_token: Which instance to render, if it's None, render all the instances.
        :param with_anns: Whether you want to render the annotations?
        :param axes_limit: The range of Lidar pointcloud that will be rendered will be between
            (-axes_limit, axes_limit).
        :param ax: Axes object.
        :return: Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(25, 25))
        if with_anns:
            if render_future_waypoints:
                DEFAULT_FUTURE_HORIZON_LEN_S = 6.0
                DEFAULT_FUTURE_INTERVAL_S = 0.5
                boxes = self.boxes_with_future_waypoints(DEFAULT_FUTURE_HORIZON_LEN_S, DEFAULT_FUTURE_INTERVAL_S, Frame.SENSOR)
            else:
                boxes = self.boxes(Frame.SENSOR)
        else:
            boxes = []
        if render_future_ego_poses:
            DEFAULT_FUTURE_HORIZON_LEN_S = 6
            TIMESTAMP_MARGIN_S = 1
            ego_poses = self.future_or_past_ego_poses(DEFAULT_FUTURE_HORIZON_LEN_S + TIMESTAMP_MARGIN_S, 'n_seconds', 'next')
        else:
            ego_poses = [self.ego_pose]
        labelmap = {lid: Label(raw_mapping['id2local'][lid], raw_mapping['id2color'][lid]) for lid in raw_mapping['id2local'].keys()}
        render_on_map(lidarpc_rec=self, db=db, boxes_lidar=boxes, ego_poses=ego_poses, radius=axes_limit, ax=ax, labelmap=labelmap, render_map_raster=render_map_raster, render_vector_map=render_vector_map, track_token=track_token, with_random_color=render_track_color, render_future_ego_poses=render_future_ego_poses)
        plt.axis('equal')
        ax.set_title('PC {} from {} in {}'.format(self.token, self.lidar.channel, self.log.location))
        return ax

class TestLidarBox(unittest.TestCase):
    """Tests the LidarBox class"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.lidar_box_vehicle = get_test_nuplan_lidar_box_vehicle()
        self.lidar_box = get_test_nuplan_lidar_box()

    @patch('nuplan.database.nuplan_db_orm.lidar_box.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the _session property"""
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock
        result = self.lidar_box._session()
        inspect_mock.assert_called_once_with(self.lidar_box)
        self.assertEqual(result, session_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the __repr__ method"""
        result = self.lidar_box.__repr__()
        simple_repr_mock.assert_called_once_with(self.lidar_box)
        self.assertEqual(result, simple_repr_mock.return_value)

    def test_log(self) -> None:
        """Tests the log property"""
        result = self.lidar_box.log
        self.assertIsInstance(result, Log)

    def test_category(self) -> None:
        """Tests the category property"""
        result = self.lidar_box.category
        self.assertIsInstance(result, Category)

    def test_timestamp(self) -> None:
        """Tests the timestamp property"""
        result = self.lidar_box.timestamp
        self.assertIsInstance(result, int)

    def test_distance_to_ego(self) -> None:
        """Tests the distance_to_ego property"""
        x = self.lidar_box.x
        y = self.lidar_box.y
        x_ego = self.lidar_box.lidar_pc.ego_pose.x
        y_ego = self.lidar_box.lidar_pc.ego_pose.y
        expected_result = math.sqrt((x - x_ego) * (x - x_ego) + (y - y_ego) * (y - y_ego))
        actual_result = self.lidar_box.distance_to_ego
        self.assertEqual(expected_result, actual_result)

    def test_size(self) -> None:
        """Tests the size property"""
        result = self.lidar_box.size
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], self.lidar_box.width)
        self.assertEqual(result[1], self.lidar_box.length)
        self.assertEqual(result[2], self.lidar_box.height)

    def test_translation(self) -> None:
        """Tests the translation property"""
        result = self.lidar_box.translation
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], self.lidar_box.x)
        self.assertEqual(result[1], self.lidar_box.y)
        self.assertEqual(result[2], self.lidar_box.z)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.Quaternion', autospec=True)
    def test_rotation(self, quaternion_mock: Mock) -> None:
        """Tests the rotation property"""
        result = self.lidar_box.rotation
        self.assertIsInstance(result, list)
        quaternion_mock.assert_called()

    @patch('nuplan.database.nuplan_db_orm.lidar_box.Quaternion', autospec=True)
    def test_quaternion(self, quaternion_mock: Mock) -> None:
        """Tests the quaternion property"""
        result = self.lidar_box.quaternion
        self.assertEqual(result, quaternion_mock.return_value)
        quaternion_mock.assert_called()

    @patch('nuplan.database.nuplan_db_orm.lidar_box.np.array', autospec=True)
    def test_translation_np(self, np_array_mock: Mock) -> None:
        """Tests the translation_np property"""
        result = self.lidar_box.translation_np
        np_array_mock.assert_called_once_with(self.lidar_box.translation)
        self.assertEqual(result, np_array_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.np.array', autospec=True)
    def test_size_np(self, np_array_mock: Mock) -> None:
        """Tests the size_np property"""
        result = self.lidar_box.size_np
        np_array_mock.assert_called_once_with(self.lidar_box.size)
        self.assertEqual(result, np_array_mock.return_value)

    def test_get_box_items(self) -> None:
        """Tests the _get_box_items method"""
        result = self.lidar_box._get_box_items()
        self.assertEqual(len(result), 2)

    def test_find_box_out_of_bounds(self) -> None:
        """Tests the _find_box method index is out of bounds"""
        result = self.lidar_box._find_box(maxsize)
        self.assertEqual(result, None)

    def test_find_box_within_bounds(self) -> None:
        """Tests the _find_box method index is within bounds"""
        result = self.lidar_box._find_box(0)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_poses"""
        number, mode, direction = (1, 'n_poses', 'prev')
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev, mode=n_seconds"""
        number, mode, direction = (1, 'n_seconds', 'prev')
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_prev_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=prev and mode is unknown"""
        number, mode, direction = (1, 'unknown_mode', 'prev')
        with self.assertRaises(ValueError):
            self.lidar_box.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_next_nposes(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_poses"""
        number, mode, direction = (1, 'n_poses', 'next')
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_nseconds(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next, mode=n_seconds"""
        number, mode, direction = (1, 'n_seconds', 'next')
        result = self.lidar_box.future_or_past_ego_poses(number, mode, direction)
        self.assertIsNotNone(result)

    def test_future_or_past_ego_poses_next_unknown_mode(self) -> None:
        """Tests the future_or_past_ego_poses when direction=next and mode is unknown"""
        number, mode, direction = (1, 'unknown_mode', 'next')
        with self.assertRaises(ValueError):
            self.lidar_box.future_or_past_ego_poses(number, mode, direction)

    def test_future_or_past_ego_poses_unknown_direction(self) -> None:
        """Tests the future_or_past_ego_poses when direction is unknown"""
        number, mode, direction = (1, 'unknown_mode', 'unknown_direction')
        with self.assertRaises(ValueError):
            self.lidar_box.future_or_past_ego_poses(number, mode, direction)

    def test_temporal_neighbours_prev_exists(self) -> None:
        """Tests the _temporal_neighbours method when prev exists"""
        result = self.lidar_box._temporal_neighbors()
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], self.lidar_box.prev)

    def test_temporal_neighbours_prev_is_empty(self) -> None:
        """Tests the _temporal_neighbours method when prev does not exist"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.prev = None
        result = lidar_box._temporal_neighbors()
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], lidar_box)

    def test_temporal_neighbours_next_exists(self) -> None:
        """Tests the _temporal_neighbours method when next exists"""
        result = self.lidar_box._temporal_neighbors()
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1], self.lidar_box.next)

    def test_temporal_neighbours_next_is_empty(self) -> None:
        """Tests the _temporal_neighbours method when next does not exist"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next = None
        result = lidar_box._temporal_neighbors()
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1], lidar_box)

    def test_velocity_no_next_and_prev(self) -> None:
        """Tests the velocity property when next and prev does not exist"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next = None
        lidar_box.prev = None
        result = lidar_box.velocity
        self.assertTrue(np.isnan(result).any())

    def test_velocity_time_diff_exceed_limit(self) -> None:
        """Tests the velocity property when the difference between timestamps exceed limit"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next.lidar_pc.timestamp = lidar_box.prev.lidar_pc.timestamp + 1000000000
        result = lidar_box.velocity
        self.assertTrue(np.isnan(result).any())

    def test_velocity_default(self) -> None:
        """Tests the default velocity property, should not return any NaN values"""
        result = self.lidar_box.velocity
        self.assertFalse(np.isnan(result).any())

    def test_angular_velocity_no_next_and_prev(self) -> None:
        """Tests the angular_velocity property when next and prev does not exist"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next = None
        lidar_box.prev = None
        result = lidar_box.angular_velocity
        self.assertTrue(np.isnan(result))

    def test_angular_velocity_time_diff_exceed_limit(self) -> None:
        """Tests the angular_velocity property when the difference between timestamps exceed limit"""
        lidar_box = deepcopy(self.lidar_box)
        lidar_box.next.lidar_pc.timestamp = lidar_box.prev.lidar_pc.timestamp + 1000000000
        result = lidar_box.angular_velocity
        self.assertTrue(np.isnan(result))

    def test_angular_velocity_default(self) -> None:
        """Tests the default angular_velocity property, should not return any NaN values"""
        result = self.lidar_box.angular_velocity
        self.assertFalse(np.isnan(result))

    def test_box(self) -> None:
        """Tests the box method"""
        result = self.lidar_box.box()
        self.assertIsInstance(result, Box3D)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.PredictedTrajectory', autospec=True)
    def test_tracked_object_is_agent(self, predicted_trajectory_mock: Mock) -> None:
        """Tests the tracked_object method"""
        future_waypoints = Mock()
        predicted_trajectory_mock.return_value.probability = 1.0
        result = self.lidar_box_vehicle.tracked_object(future_waypoints)
        predicted_trajectory_mock.assert_called_once_with(1.0, future_waypoints)
        self.assertIsInstance(result, Agent)

    def test_tracked_object_is_static_object(self) -> None:
        """Tests the tracked_object method"""
        future_waypoints = Mock()
        result = self.lidar_box.tracked_object(future_waypoints)
        self.assertIsInstance(result, StaticObject)

    def test_velocity(self) -> None:
        """Test if velocity is calculated correctly."""
        self.assertTrue(self.lidar_box.prev is not None)
        self.assertTrue(self.lidar_box.next is not None)
        prev_lidar_box: LidarBox = self.lidar_box.prev
        next_lidar_box: LidarBox = self.lidar_box.next
        time_diff = 1e-06 * (next_lidar_box.timestamp - prev_lidar_box.timestamp)
        pos_diff = self.lidar_box.velocity * time_diff
        pos_next = next_lidar_box.translation_np
        pos_next_pred = prev_lidar_box.translation_np + pos_diff
        np.testing.assert_array_almost_equal(pos_next[:2], pos_next_pred[:2], decimal=4)

    def test_angular_velocity(self) -> None:
        """Test if angular velocity is calculated correctly."""
        self.assertTrue(self.lidar_box.prev is not None)
        self.assertTrue(self.lidar_box.next is not None)
        prev_lidar_box: LidarBox = self.lidar_box.prev
        next_lidar_box: LidarBox = self.lidar_box.next
        time_diff = 1e-06 * (next_lidar_box.timestamp - prev_lidar_box.timestamp)
        yaw_diff = self.lidar_box.angular_velocity * time_diff
        yaw_prev = quaternion_yaw(prev_lidar_box.quaternion)
        q_yaw_prev = Quaternion(np.array([np.cos(yaw_prev / 2), 0, 0, np.sin(yaw_prev / 2)]))
        q_yaw_next_pred = Quaternion(np.array([np.cos(yaw_diff / 2), 0, 0, np.sin(yaw_diff / 2)])) * q_yaw_prev
        yaw_next_pred = quaternion_yaw(q_yaw_next_pred)
        yaw_next = quaternion_yaw(next_lidar_box.quaternion)
        self.assertAlmostEqual(yaw_next, yaw_next_pred, delta=0.0001)

    def test_next(self) -> None:
        """Test next."""
        self.assertGreater(self.lidar_box.next.timestamp, self.lidar_box.timestamp, 'Timestamp of succeeding box must be greater then current box.')

    def test_prev(self) -> None:
        """Test prev."""
        self.assertLess(self.lidar_box.prev.timestamp, self.lidar_box.timestamp, 'Timestamp of preceding box must be lower then current box.')

    def test_past_ego_poses(self) -> None:
        """Test if past ego poses are returned correctly."""
        n_ego_poses = 4
        past_ego_poses = self.lidar_box.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='prev')
        ego_pose = self.lidar_box.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertGreater(ego_pose.timestamp, past_ego_poses[i].timestamp, 'Timestamp of current EgoPose must be greater than past EgoPoses')

    def test_future_ego_poses(self) -> None:
        """Test if future ego poses are returned correctly."""
        n_ego_poses = 4
        future_ego_poses = self.lidar_box.future_or_past_ego_poses(number=n_ego_poses, mode='n_poses', direction='next')
        ego_pose = self.lidar_box.lidar_pc.ego_pose
        for i in range(n_ego_poses):
            self.assertLess(ego_pose.timestamp, future_ego_poses[i].timestamp, 'Timestamp of current EgoPose must be less than future EgoPoses ')

    def test_get_box_items_to_iterate(self) -> None:
        """Tests the get_box_items_to_iterate method"""
        result = self.lidar_box.get_box_items_to_iterate()
        self.assertTrue(self.lidar_box.timestamp in result)
        self.assertEqual(self.lidar_box.prev, result[self.lidar_box.timestamp][0])
        self.assertEqual(self.lidar_box.next, result[self.lidar_box.timestamp][1])

    @patch('nuplan.database.nuplan_db_orm.lidar_box.IterableLidarBox', autospec=True)
    def test_iter(self, iterable_lidar_box_mock: Mock) -> None:
        """Tests the iterator for LidarBox"""
        result = iter(self.lidar_box)
        iterable_lidar_box_mock.assert_called_once_with(self.lidar_box)
        self.assertEqual(result, iterable_lidar_box_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.lidar_box.IterableLidarBox', autospec=True)
    def test_reverse_iter(self, iterable_lidar_box_mock: Mock) -> None:
        """Tests the reverse iterator for LidarBox"""
        result = reversed(self.lidar_box)
        iterable_lidar_box_mock.assert_called_once_with(self.lidar_box, reverse=True)
        self.assertEqual(result, iterable_lidar_box_mock.return_value)

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculates the yaw angle from a quaternion.
    Follow convention: R = Rz(yaw)Ry(pitch)Px(roll)
    Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """
    a = 2.0 * (q[0] * q[3] + q[1] * q[2])
    b = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
    return math.atan2(a, b)

class TestGetBoxes(unittest.TestCase):
    """Test get box."""

    def _box_A(self) -> Box3D:
        """
        Helper method to get one box.
        :return: One box.
        """
        return Box3D(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), orientation=Quaternion(axis=[1, 0, 0], angle=0), velocity=(0.0, 0.0, 0.0), angular_velocity=0.0)

    def _box_B(self) -> Box3D:
        """
        Helper method to get one box.
        :return: One box.
        """
        return Box3D(center=(1.0, 2.0, 3.0), size=(1.0, 1.0, 1.0), orientation=Quaternion(axis=[1, 0, 0], angle=2), velocity=(5.0, 6.0, 7.0), angular_velocity=8.0)

    def _box_quarterway_between_A_and_B(self) -> Box3D:
        """
        Helper method to get one box.
        :return: One box.
        """
        return Box3D(center=(0.25, 0.5, 0.75), size=(1.0, 1.0, 1.0), orientation=Quaternion(axis=[1, 0, 0], angle=0.5), velocity=(1.25, 1.5, 1.75), angular_velocity=2.0)

    def _box_halfway_between_A_and_B(self) -> Box3D:
        """
        Helper method to get one box.
        :return: One box.
        """
        return Box3D(center=(0.5, 1.0, 1.5), size=(1.0, 1.0, 1.0), orientation=Quaternion(axis=[1, 0, 0], angle=1), velocity=(2.5, 3, 3.5), angular_velocity=4.0)

    def _annotation_A(self, track_token: str) -> Mock:
        """
        Helper method to get one annotation.
        :param track_token: Track token to use.
        :return: Mocked annotation.
        """
        ann = Mock()
        ann.x = 0.0
        ann.y = 0.0
        ann.z = 0.0
        ann.translation_np = np.array([ann.x, ann.y, ann.z])
        ann.width = 1.0
        ann.length = 1.0
        ann.height = 1.0
        ann.size = (ann.width, ann.length, ann.height)
        ann.roll = 0.0
        ann.pitch = 0.0
        ann.yaw = 0.0
        ann.quaternion = Quaternion(axis=[1, 0, 0], angle=0)
        ann.vx = 0.0
        ann.vy = 0.0
        ann.vz = 0.0
        ann.velocity = np.array([ann.vx, ann.vy, ann.vz])
        ann.angular_velocity = 0.0
        ann.box.return_value = self._box_A()
        ann.track_token = track_token
        return ann

    def _annotation_B(self, track_token: str) -> Mock:
        """
        Helper method to get one annotation.
        :param track_token: Track token to use.
        :return: Mocked annotation.
        """
        ann = Mock()
        ann.x = 1.0
        ann.y = 2.0
        ann.z = 3.0
        ann.translation_np = np.array([ann.x, ann.y, ann.z])
        ann.width = 1.0
        ann.length = 1.0
        ann.height = 1.0
        ann.size = (ann.width, ann.length, ann.height)
        ann.roll = 0.0
        ann.pitch = 0.0
        ann.yaw = 0.0
        ann.quaternion = Quaternion(axis=[1, 0, 0], angle=2)
        ann.vx = 5.0
        ann.vy = 6.0
        ann.vz = 7.0
        ann.velocity = np.array([ann.vx, ann.vy, ann.vz])
        ann.angular_velocity = 8.0
        ann.box.return_value = self._box_B()
        ann.track_token = track_token
        return ann

    def _trans_matrix_ego(self) -> npt.NDArray[np.float64]:
        """
        Helper method to get a transformation.
        :return: <np.float: 4, 4> Transformation matrix.
        """
        return np.array([[0, 1, 0, 1], [-1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])

    def _trans_matrix_sensor(self) -> npt.NDArray[np.float64]:
        """
        Helper method to get a transformation.
        :return: <np.float: 4, 4> Transformation matrix.
        """
        return np.array([[0, 0, 1, 4], [0, -1, 0, 5], [1, 0, 0, 6], [0, 0, 0, 1]])

    def test_frame_vehicle(self) -> None:
        """
        Test putting resulting boxes in vehicle coordinates.
        """
        lidarpc = Mock()
        lidarpc.lidar_boxes = [self._annotation_B(track_token='456')]
        lidarpc.prev = object()
        box_b_vehicle_frame = self._box_B()
        box_b_vehicle_frame.transform(self._trans_matrix_ego())
        self.assertEqual(get_boxes(lidarpc, frame=Frame.VEHICLE, trans_matrix_ego=self._trans_matrix_ego()), [box_b_vehicle_frame])

    def test_frame_sensor(self) -> None:
        """
        Test putting resulting boxes in sensor coordinates.
        """
        lidarpc = Mock()
        lidarpc.lidar_boxes = [self._annotation_B(track_token='456')]
        lidarpc.prev = object()
        box_b_sensor_frame = self._box_B()
        box_b_sensor_frame.transform(self._trans_matrix_ego())
        box_b_sensor_frame.transform(self._trans_matrix_sensor())
        self.assertEqual(get_boxes(lidarpc, frame=Frame.SENSOR, trans_matrix_ego=self._trans_matrix_ego(), trans_matrix_sensor=self._trans_matrix_sensor()), [box_b_sensor_frame])

class TestLoadBoxes(unittest.TestCase):
    """Tests for get_boxes() and get_future_box_sequence()"""

    def setUp(self) -> None:
        """Set up the test case."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc()
        self.future_horizon_len_s = 1
        self.future_interval_s = 0.05

    def test_can_run_get_future_box_sequence(self) -> None:
        """Test get future box sequence."""
        get_future_box_sequence(lidar_pcs=[self.lidar_pc, self.lidar_pc.next], frame=Frame.VEHICLE, future_horizon_len_s=self.future_horizon_len_s, trans_matrix_ego=self.lidar_pc.ego_pose.trans_matrix_inv, future_interval_s=self.future_interval_s)

    def test_pack_future_boxes(self) -> None:
        """Test pack future boxes."""
        track_token_2_box_sequence = get_future_box_sequence(lidar_pcs=[self.lidar_pc, self.lidar_pc.next], frame=Frame.VEHICLE, future_horizon_len_s=self.future_horizon_len_s, trans_matrix_ego=self.lidar_pc.ego_pose.trans_matrix_inv, future_interval_s=self.future_interval_s)
        boxes_with_futures = pack_future_boxes(track_token_2_box_sequence=track_token_2_box_sequence, future_horizon_len_s=self.future_horizon_len_s, future_interval_s=self.future_interval_s)
        for box in boxes_with_futures:
            for horizon_idx, horizon_s in enumerate(box.get_all_future_horizons_s()):
                future_center = box.get_future_center_at_horizon(horizon_s)
                future_orientation = box.get_future_orientation_at_horizon(horizon_s)
                self.assertTrue(box.track_token is not None)
                expected_future_box = track_token_2_box_sequence[box.track_token][horizon_idx + 1]
                if expected_future_box is None:
                    np.testing.assert_array_equal(future_center, [np.nan, np.nan, np.nan])
                    self.assertEqual(future_orientation, None)
                else:
                    np.testing.assert_array_equal(expected_future_box.center, future_center)
                    self.assertEqual(expected_future_box.orientation, future_orientation)

    def test_load_boxes_from_lidarpc(self) -> None:
        """Test load all boxes from a lidar pc."""
        boxes = load_boxes_from_lidarpc(self.db, self.lidar_pc, ['pedestrian', 'vehicle'], False, 80.04, self.future_horizon_len_s, self.future_interval_s, {'pedestrian': 0, 'vehicle': 1})
        self.assertSetEqual({'pedestrian', 'vehicle'}, set(boxes.keys()))
        self.assertEqual(len(boxes['pedestrian']), 70)
        self.assertEqual(len(boxes['vehicle']), 29)

def birdview_corner_angle_mean_distance_box(a: Box3D, b: Box3D, period: float) -> float:
    """
    Calculates ad-hoc birdview distance of two Box3D instances.
    :param a: Box3D 1.
    :param b: Box3D 2.
    :param period: Periodicity for assessing angle difference.
    :return: Birdview distance.
    """
    error = 0.0
    error += abs(a.center[0] - b.center[0])
    error += abs(a.center[1] - b.center[1])
    error += abs(a.wlh[0] - b.wlh[0])
    error += abs(a.wlh[1] - b.wlh[1])
    a_yaw = quaternion_yaw(a.orientation)
    b_yaw = quaternion_yaw(b.orientation)
    error += abs(angle_diff(a_yaw, b_yaw, period))
    return error / 5

def birdview_pseudo_iou_box(a: Box3D, b: Box3D, period: float) -> float:
    """
    Calculates ad-hoc birdview IoU of two Box3D instances.
    :param a: Box3D 1.
    :param b: Box3D 2.
    :param period: Periodicity for assessing angle difference.
    :return: Birdview IoU.
    """
    return 1 / (1 + birdview_corner_angle_mean_distance_box(a, b, period))

def footprint(box: TwoDimBox) -> Polygon:
    """
        Get footprint polygon.
        :param box: Input 2-d box.
        :return: A polygon representation of the 2d box.
        """
    x, y, w, l, head = box
    rot = np.array([[math.cos(head), -math.sin(head)], [math.sin(head), math.cos(head)]])
    q0 = np.array([x, y])[:, None]
    q1 = np.array([-w / 2, -l / 2])[:, None]
    q2 = np.array([-w / 2, l / 2])[:, None]
    q3 = np.array([w / 2, l / 2])[:, None]
    q4 = np.array([w / 2, -l / 2])[:, None]
    q1 = np.dot(rot, q1) + q0
    q2 = np.dot(rot, q2) + q0
    q3 = np.dot(rot, q3) + q0
    q4 = np.dot(rot, q4) + q0
    return Polygon([(q1.item(0), q1.item(1)), (q2.item(0), q2.item(1)), (q3.item(0), q3.item(1)), (q4.item(0), q4.item(1))])

def hausdorff_distance_box(obsbox: Box3D, gtbox: Box3D) -> float:
    """
    Calculate Hausdorff distance between two 2d-boxes in Box3D class.
    :param obsbox: Observation box.
    :param gtbox: Ground truth box.
    :return: Hausdorff distance.
    """

    def footprint(box: Box3D) -> Polygon:
        """
        Get footprint polygon.
        :param box: (center_x <float>, center_y <float>, width <float>, length <float>, theta <float>).
        :return: <Polygon>. A polygon representation of the 2d box.
        """
        x, y, w, l, head = (box.center[0], box.center[1], box.wlh[0], box.wlh[1], quaternion_yaw(box.orientation))
        rot = np.array([[math.cos(head), -math.sin(head)], [math.sin(head), math.cos(head)]])
        q0 = np.array([x, y])[:, None]
        q1 = np.array([-w / 2, -l / 2])[:, None]
        q2 = np.array([-w / 2, l / 2])[:, None]
        q3 = np.array([w / 2, l / 2])[:, None]
        q4 = np.array([w / 2, -l / 2])[:, None]
        q1 = np.dot(rot, q1) + q0
        q2 = np.dot(rot, q2) + q0
        q3 = np.dot(rot, q3) + q0
        q4 = np.dot(rot, q4) + q0
        return Polygon([(q1.item(0), q1.item(1)), (q2.item(0), q2.item(1)), (q3.item(0), q3.item(1)), (q4.item(0), q4.item(1))])
    obs_poly = footprint(obsbox)
    gt_poly = footprint(gtbox)
    distance = 0.0
    for p in list(gt_poly.exterior.coords):
        new_dist = float(obs_poly.distance(Point(p)))
        if new_dist > distance:
            distance = new_dist
    for p in list(obs_poly.exterior.coords):
        new_dist = float(gt_poly.distance(Point(p)))
        if new_dist > distance:
            distance = new_dist
    return distance

def hausdorff_distance(obsbox: TwoDimBox, gtbox: TwoDimBox) -> float:
    """
    Calculate Hausdorff distance between two 2d-boxes.
    :param obsbox: Observation 2d box.
    :param gtbox: Ground truth 2d box.
    :return: Hausdorff distance.
    """

    def footprint(box: TwoDimBox) -> Polygon:
        """
        Get footprint polygon.
        :param box: Input 2-d box.
        :return: A polygon representation of the 2d box.
        """
        x, y, w, l, head = box
        rot = np.array([[math.cos(head), -math.sin(head)], [math.sin(head), math.cos(head)]])
        q0 = np.array([x, y])[:, None]
        q1 = np.array([-w / 2, -l / 2])[:, None]
        q2 = np.array([-w / 2, l / 2])[:, None]
        q3 = np.array([w / 2, l / 2])[:, None]
        q4 = np.array([w / 2, -l / 2])[:, None]
        q1 = np.dot(rot, q1) + q0
        q2 = np.dot(rot, q2) + q0
        q3 = np.dot(rot, q3) + q0
        q4 = np.dot(rot, q4) + q0
        return Polygon([(q1.item(0), q1.item(1)), (q2.item(0), q2.item(1)), (q3.item(0), q3.item(1)), (q4.item(0), q4.item(1))])
    obs_poly = footprint(obsbox)
    gt_poly = footprint(gtbox)
    distance = 0.0
    for p in list(gt_poly.exterior.coords):
        new_dist = float(obs_poly.distance(Point(p)))
        if new_dist > distance:
            distance = new_dist
    for p in list(obs_poly.exterior.coords):
        new_dist = float(gt_poly.distance(Point(p)))
        if new_dist > distance:
            distance = new_dist
    return distance

def points_in_box_bev(box: Box3D, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Checks whether points are inside the box in birds eyed view.
    :param box: Box3D instance.
    :param points: Trajectory given as <np.float: 3, n_way_points)
    :return: A boolean mask whether points are in the box in BEV world.
    """
    box = box.copy()
    points = points.copy()
    points[2, :] = box.center[2]
    return points_in_box(box, points)

def points_in_box(box: Box3D, points: npt.NDArray[np.float64], wlh_factor: float=1.0) -> npt.NDArray[np.float64]:
    """
    Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579.

    :param box: A Box3D instance.
    :param points: Points given as <np.float: 3, n_way_points)
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >. Mask for points in box or not.
    """
    assert points.shape[0] == 3, 'Expect 3D pts'
    assert points.ndim == 2, 'Expect 2D inputs'
    r = ((box.wlh / 2) ** 2).sum() ** 0.5
    w, l, h = box.wlh
    w, l, h, r = (w * wlh_factor, l * wlh_factor, h * wlh_factor, r * wlh_factor)
    cx, cy, cz = box.center
    x, y, z = points
    pts_mask = functools.reduce(np.logical_and, [x >= cx - r, x <= cx + r, y >= cy - r, y <= cy + r, z >= cz - r, z <= cz + r])
    pts = points[:, pts_mask]
    rot = box.orientation.inverse.rotation_matrix.astype(np.float32)
    x, y, z = rot @ pts + (rot @ -box.center.astype(np.float32)).reshape(-1, 1)
    mask = functools.reduce(np.logical_and, [np.logical_and(x >= -l / 2, x <= l / 2), np.logical_and(y >= -w / 2, y <= w / 2), np.logical_and(z >= -h / 2, z <= h / 2)])
    pts_index = np.nonzero(pts_mask)
    pts_mask[pts_index] = mask
    return pts_mask

class Box3D(BoxInterface):
    """Simple data class representing a 3d box including, label, score and velocity."""
    MAX_LABELS = 100
    _labelmap = None
    _min_size = np.finfo(np.float32).eps
    RENDER_MODE_PROB_THRESHOLD = 0.1

    def __init__(self, center: Tuple[float, float, float], size: Tuple[float, float, float], orientation: Quaternion, label: int=np.nan, score: float=np.nan, velocity: Tuple[float, float, float]=(np.nan, np.nan, np.nan), angular_velocity: float=np.nan, payload: Optional[Dict[str, Any]]=None, token: Optional[str]=None, track_token: Optional[str]=None, future_horizon_len_s: Optional[float]=None, future_interval_s: Optional[float]=None, future_centers: Optional[List[List[Tuple[float, float, float]]]]=None, future_orientations: Optional[List[List[Quaternion]]]=None, mode_probs: Optional[List[float]]=None) -> None:
        """
        The convention is that: x points forward, y to the left, z up when this box is initialized with an orientation
        of zero.
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box3D orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box3D velocity in x, y, z direction.
        :param angular_velocity: Box3D angular velocity in yaw direction.
        :param payload: Box3D payload, optional. For example, can be used to denote category name or provide boolean
            data regarding whether the box trajectory goes off the driveable area. The format should be a dictionary
            so that different types of metadata can be stored here, e.g., payload['category_name'] and
            payload['timestamp_2_on_road_bool'].
        :param token: Unique token (optional). Usually DB annotation token. In NuPlanDB, 3D annotations are present in
            the LidarBox table, in which case the token provided corresponds to the LidarBox token.
        :param track_token: Track token in the "track" table that corresponds to a particular box.
        :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
        :param future_interval_s: Timestamp interval of the future waypoints in seconds.
        :param future_centers: List of future center coordinates given as (x, y, z), where the list indices increase
            with time and are spaced apart at the specified intervals. If the box is missing at a future timestamp, then
            the future center coordinates at the corresponding list index will have the format (np.nan, np.nan, np.nan)
        :param future_orientations: List of future Box3D orientations, where the list indices increase with time and
            are spaced apart at the specified intervals. If the box is missing at a future timestamp, then
            the future orientation at the corresponding list index will be represented as None.
        :param mode_probs: Mode probabilities.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert len(velocity) == 3
        assert type(orientation) == Quaternion
        assert size[0] > self._min_size, 'Error: box Width must be larger than {} cm'.format(100 * self._min_size)
        assert size[1] > self._min_size, 'Error: box Length must be larger than {} cm'.format(100 * self._min_size)
        assert size[2] > self._min_size, 'Error: box Height must be larger than {} cm'.format(100 * self._min_size)
        assert size[0] * size[1] * size[2] > self._min_size, 'Invalid box volume'
        self.center = np.array(center, dtype=float)
        self.size = size
        self.wlh = np.array(size, dtype=float)
        self.orientation = orientation.__copy__()
        self._label = int(label) if not np.isnan(label) else label
        self._score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity, dtype=float)
        self.angular_velocity = float(angular_velocity) if not np.isnan(angular_velocity) else angular_velocity
        self.payload = payload if payload is not None else {}
        assert type(self.payload) == dict, 'Error: box payload is not a dict'
        self.token = token
        self._color = None
        self.track_token = track_token
        self.init_trajectory_fields(future_horizon_len_s, future_interval_s, future_centers, future_orientations, mode_probs)

    @classmethod
    def set_labelmap(cls, labelmap: Dict[int, Label]) -> None:
        """
        :param labelmap: {id: label}. Map from label id to Label.
        """
        cls._labelmap = labelmap

    @property
    def color(self) -> Color:
        """RGBA color of Box3D."""
        if self._color is None:
            self._set_color()
        return self._color

    @property
    def width(self) -> float:
        """Width of the box."""
        return float(self.wlh[0])

    @width.setter
    def width(self, width: float) -> None:
        """Implemented. See interface."""
        self.wlh[0] = width

    @property
    def length(self) -> float:
        """Length of the box."""
        return float(self.wlh[1])

    @length.setter
    def length(self, length: float) -> None:
        """Implemented. See interface."""
        self.wlh[1] = length

    @property
    def height(self) -> float:
        """Height of the box."""
        return float(self.wlh[2])

    @height.setter
    def height(self, height: float) -> None:
        """Implemented. See interface."""
        self.wlh[2] = height

    @property
    def yaw(self) -> float:
        """Yaw of the box."""
        return quaternion_yaw(self.orientation)

    @property
    def distance_plane(self) -> float:
        """
        The euclidean distance of the box center from the z-axis passing through the origin of the coordinate system
        (sensor/world). Refer to the axial/radial distance in a cylindrical coordinate system:
        https://en.wikipedia.org/wiki/Cylindrical_coordinate_system.
        """
        return float((self.center[0] ** 2 + self.center[1] ** 2) ** 0.5)

    @property
    def distance_3d(self) -> float:
        """
        The euclidean distance of the box center from the origin of the coordinate system (sensor/world). Refer to the
        radial distance in a spherical coordinate system: https://en.wikipedia.org/wiki/Spherical_coordinate_system.
        """
        return float((self.center[0] ** 2 + self.center[1] ** 2 + self.center[2] ** 2) ** 0.5)

    def init_trajectory_fields(self, future_horizon_len_s: Optional[float]=None, future_interval_s: Optional[float]=None, future_centers: Optional[List[List[Tuple[float, float, float]]]]=None, future_orientations: Optional[List[List[Quaternion]]]=None, mode_probs: Optional[List[float]]=None) -> None:
        """
        Checks that values for future horizon length, interval length, future orientations and future centers are either
        all provided or all None. Check that future centers and future orientations are the expected length, if
        applicable.
        :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
        :param future_interval_s: Timestamp interval of the future waypoints in seconds.
        :param future_centers: List of future center coordinates given as (x, y, z), where the list indices increase
            with time and are spaced apart at the specified intervals. If the box is missing at a future timestamp, then
            the future center coordinates at the corresponding list index will have the format (np.nan, np.nan, np.nan)
        :param future_orientations: List of future Box3D orientations, where the list indices increase with time and
            are spaced apart at the specified intervals. If the box is missing at a future timestamp, then
            the future orientation at the corresponding list index will be represented as None.
        :param mode_probs: Mode probabilities.
        """
        if future_centers is None:
            assert future_horizon_len_s is None
            assert future_interval_s is None
            assert future_orientations is None
            assert mode_probs is None
            self.future_horizon_len_s = None
            self.future_interval_s = None
            self.future_centers = None
            self.future_orientations = None
            self.mode_probs = None
            self.num_modes = None
            self.num_future_timesteps = None
            return
        assert future_horizon_len_s is not None
        assert future_interval_s is not None
        assert future_orientations is not None
        assert mode_probs is not None
        self.future_horizon_len_s = future_horizon_len_s
        self.future_interval_s = future_interval_s
        self.future_centers = np.array(future_centers, dtype=float)
        self.future_orientations = future_orientations
        self.mode_probs = np.array(mode_probs, dtype=float)
        assert self.future_centers.ndim == 3
        if not self.mode_probs.shape[0] == self.future_centers.shape[0] == len(self.future_orientations):
            raise ValueError(f'Future parameters have different number of modes:\nself.mode_probs.shape: {self.mode_probs.shape}\nself.future_centers.shape: {self.future_centers.shape}\nlen(self.future_orientations): {len(self.future_orientations)}')
        self.num_modes = self.mode_probs.shape[0]
        if self.future_centers.shape[1] != len(self.future_orientations[0]):
            raise ValueError(f'Future parameters have different number of timesteps:\nself.future_centers.shape: {self.future_centers.shape}\nlen(self.future_orientations[0]): {len(self.future_orientations[0])}')
        self.num_future_timesteps = self.future_centers.shape[1]
        if self.future_horizon_len_s != self.future_interval_s * self.num_future_timesteps:
            raise ValueError(f'Future horizon length ({self.future_horizon_len_s}) should equal to future interval ({self.future_interval_s}) times number of timesteps ({self.num_future_timesteps}).')

    def _set_color(self) -> None:
        """Sets color based on label."""
        if self._labelmap is None or self.label not in self._labelmap:
            if self.label is None or np.isnan(self.label):
                self._color = (255, 61, 99, 0)
            else:
                fixed_colors = [(255, 61, 99, 0), (255, 158, 0, 0), (0, 0, 230, 0)]
                colors = [el + (255,) for el in rainbow(self.MAX_LABELS - 3)]
                random.Random(1).shuffle(colors)
                colors = fixed_colors + colors
                self._color = colors[self.label % self.MAX_LABELS]
        else:
            self._color = self._labelmap[self.label].color

    @property
    def name(self) -> str:
        """Name of Box3D."""
        if self._labelmap is None or self.label is np.nan:
            return 'not_set'
        elif self.label not in self._labelmap:
            return 'unknown'
        else:
            return self._labelmap[self.label].name

    @property
    def label(self) -> int:
        """Implemented. See interface."""
        return self._label

    @label.setter
    def label(self, label: int) -> None:
        """Implemented. See interface."""
        self._label = label

    @property
    def score(self) -> float:
        """Implemented. See interface."""
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        """Implemented. See interface."""
        self._score = score

    @property
    def has_future_waypoints(self) -> bool:
        """Whether this box has future waypoints."""
        return self.future_centers is not None

    def equate_orientations(self, other: object) -> bool:
        """
        Compare orientations of two Box3D Objects.
        :param other: The other Box3D object.
        :return: True if orientations of both objects are the same, otherwise False.
        """
        if (self.future_orientations is None) != (other.future_orientations is None):
            return False
        if self.future_orientations is not None and other.future_orientations is not None:
            for mode_idx in range(self.num_modes):
                for horizon_idx in range(self.num_future_timesteps):
                    self_future_orientation = self.future_orientations[mode_idx][horizon_idx]
                    other_future_orientation = other.future_orientations[mode_idx][horizon_idx]
                    if (self_future_orientation is None) != (other_future_orientation is None):
                        return False
                    if self_future_orientation is not None and other_future_orientation is not None:
                        if not np.allclose(self.future_orientations[mode_idx][horizon_idx].rotation_matrix, other.future_orientations[mode_idx][horizon_idx].rotation_matrix, atol=0.0001):
                            return False
        return True

    def __eq__(self, other: object) -> bool:
        """
        Compares the two Box3D object are the same.
        :param other: The other Box3D object.
        :return: True if both objects are the same, otherwise False.
        """
        if not isinstance(other, Box3D):
            return NotImplemented
        center = np.allclose(self.center, other.center, atol=0.0001)
        wlh = np.allclose(self.wlh, other.wlh, atol=0.0001)
        orientation = np.allclose(self.orientation.rotation_matrix, other.orientation.rotation_matrix, atol=0.0001)
        label = self.label == other.label or (np.isnan(self.label) and np.isnan(other.label))
        score = self.score == other.score or (np.isnan(self.score) and np.isnan(other.score))
        vel = np.allclose(self.velocity, other.velocity, atol=0.0001) or (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity)))
        angular_vel = np.isclose(self.angular_velocity, other.angular_velocity, atol=0.0001) or (np.isnan(self.angular_velocity) and np.isnan(other.angular_velocity))
        payload = self.payload == other.payload
        if not (center and wlh and orientation and label and score and vel and angular_vel and payload):
            return False
        if self.future_horizon_len_s != other.future_horizon_len_s:
            return False
        if self.future_interval_s != other.future_interval_s:
            return False
        if self.num_future_timesteps != other.num_future_timesteps:
            return False
        if self.num_modes != other.num_modes:
            return False
        if (self.future_centers is None) != (other.future_centers is None):
            return False
        if self.future_centers is not None and other.future_centers is not None:
            if not np.array_equal(np.isnan(self.future_centers), np.isnan(other.future_centers)):
                return False
            if not np.allclose(self.future_centers[~np.isnan(self.future_centers)], other.future_centers[~np.isnan(other.future_centers)], atol=0.0001):
                return False
        if not self.equate_orientations(other):
            return False
        if (self.mode_probs is None) != (other.mode_probs is None):
            return False
        if self.mode_probs is not None and other.mode_probs is not None:
            if not np.allclose(self.mode_probs, other.mode_probs, atol=0.0001):
                return False
        return True

    def __repr__(self) -> str:
        """
        Represent a box using a string.
        :return: A string to represent a box.
        """
        arguments = 'center={}, size={}, orientation={}'.format(tuple(self.center), tuple(self.wlh), self.orientation.__repr__())
        if not np.isnan(self.label):
            arguments += ', label={}'.format(self.label)
        if not np.isnan(self.score):
            arguments += ', score={}'.format(self.score)
        if not all(np.isnan(self.velocity)):
            arguments += ', velocity={}'.format(tuple(self.velocity))
        if not np.isnan(self.angular_velocity):
            arguments += ', angular_velocity={}'.format(self.angular_velocity)
        if self.payload is not None:
            arguments += ", payload='{}'".format(self.payload)
        if self.token is not None:
            arguments += ", token='{}'".format(self.token)
        if self.track_token is not None:
            arguments += ", track_token='{}'".format(self.track_token)
        if self.future_horizon_len_s is not None:
            arguments += ", future_horizon_len_s='{}'".format(self.future_horizon_len_s)
        if self.future_interval_s is not None:
            arguments += ", future_interval_s='{}'".format(self.future_interval_s)
        if self.future_centers is not None:
            arguments += ", future_centers='{}'".format(self.future_centers)
        if self.future_orientations is not None:
            arguments += ", future_orientations='{}'".format(self.future_orientations)
        if self.mode_probs is not None:
            arguments += ", mode_probs='{}'".format(self.mode_probs)
        return 'Box3D({})'.format(arguments)

    def serialize(self) -> Dict[str, Any]:
        """
        Implemented. See interface.
        :return: Dict of field name to field values.
        """
        future_orientations_serialized = [[orientation.elements.tolist() if orientation is not None else None for orientation in future_orientations_of_mode] for future_orientations_of_mode in self.future_orientations] if self.future_orientations is not None else None
        return {'center': self.center.tolist(), 'wlh': self.wlh.tolist(), 'orientation': self.orientation.elements.tolist(), 'label': self.label, 'score': self.score, 'velocity': self.velocity.tolist(), 'angular_velocity': self.angular_velocity, 'payload': self.payload, 'token': self.token, 'track_token': self.track_token, 'future_horizon_len_s': self.future_horizon_len_s, 'future_interval_s': self.future_interval_s, 'future_centers': self.future_centers.tolist() if self.future_centers is not None else None, 'future_orientations': future_orientations_serialized, 'mode_probs': self.mode_probs.tolist() if self.mode_probs is not None else None}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Box3D:
        """
        Implemented. See interface.
        :param data: Output from serialize.
        :return: Deserialized Box3D.
        """
        if type(data) is dict:
            future_orientations = [[Quaternion(orientation) if orientation is not None else None for orientation in orientations_of_mode] for orientations_of_mode in data['future_orientations']] if data['future_orientations'] is not None else None
            return Box3D(data['center'], data['wlh'], Quaternion(data['orientation']), label=data['label'], score=data['score'], velocity=data['velocity'], angular_velocity=data['angular_velocity'], payload=data['payload'], token=data['token'], track_token=data['track_token'], future_horizon_len_s=data['future_horizon_len_s'], future_interval_s=data['future_interval_s'], future_centers=data['future_centers'], future_orientations=future_orientations, mode_probs=data['mode_probs'])
        else:
            raise TypeError('Type of data should be a dictionary.')

    @classmethod
    def arbitrary_box(cls) -> Box3D:
        """Instantiates an arbitrary box."""
        return Box3D(center=(1.1, 2.2, 3.3), size=(2.2, 5.5, 3.1), orientation=Quaternion(1, 2, 3, 4), label=1, score=0.5, velocity=(1.1, 2.3, 3.3), angular_velocity=0.314, payload={'def': 'hij'}, token='abc', track_token='wxy')

    @classmethod
    def make_random(cls) -> Box3D:
        """
        Instantiates a random box.
        :return: Box3D instance.
        """
        center = random.sample(range(50), 3)
        size = random.sample(range(1, 50), 3)
        quaternion = Quaternion(random.sample(range(10), 4))
        label = random.choice(range(cls.MAX_LABELS))
        score = random.uniform(0, 1)
        velocity = tuple((random.uniform(0, 10) for _ in range(3)))
        angular_velocity = np.random.uniform(-np.pi, np.pi)
        return Box3D(center=center, size=size, orientation=quaternion, label=label, score=score, velocity=velocity, angular_velocity=angular_velocity)

    def copy(self) -> Box3D:
        """
        Create a copy of self.
        :return: Box3D instance.
        """
        return Box3D(center=self.center, size=self.wlh, orientation=self.orientation, label=self.label, score=self.score, velocity=self.velocity, angular_velocity=self.angular_velocity, payload=self.payload, token=self.token, track_token=self.track_token, future_horizon_len_s=self.future_horizon_len_s, future_interval_s=self.future_interval_s, future_centers=self.future_centers, future_orientations=self.future_orientations, mode_probs=self.mode_probs)

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """
        Returns a rotation matrix.
        :return: <np.float: (3, 3)>.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: npt.NDArray[np.float64]) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3>. Translation in x, y, z direction.
        """
        self.center += x
        if self.future_centers is not None:
            assert x.ndim == 1
            assert x.shape[-1] == self.future_centers.shape[-1]
            self.future_centers += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates a box.
        :param quaternion: Rotation to apply.
        """
        self.orientation = quaternion * self.orientation
        rotation_matrix = quaternion.rotation_matrix
        self.center = np.dot(rotation_matrix, self.center)
        self.velocity = np.dot(rotation_matrix, self.velocity)
        if self.future_centers is not None:
            for mode_idx in range(self.num_modes):
                for horizon_idx in range(self.num_future_timesteps):
                    self.future_centers[mode_idx][horizon_idx] = np.dot(rotation_matrix, self.future_centers[mode_idx][horizon_idx])
        if self.future_orientations is not None:
            for mode_idx in range(self.num_modes):
                for horizon_idx in range(self.num_future_timesteps):
                    if self.future_orientations[mode_idx][horizon_idx] is None:
                        continue
                    self.future_orientations[mode_idx][horizon_idx] = quaternion * self.future_orientations[mode_idx][horizon_idx]

    def transform(self, trans_matrix: npt.NDArray[np.float64]) -> None:
        """
        Applies a transformation matrix to the box
        :param trans_matrix: <np.float: 4, 4>. Homogeneous transformation matrix.
        """
        self.rotate(Quaternion(matrix=trans_matrix[:3, :3]))
        self.translate(trans_matrix[:3, 3])

    def scale(self, s: Tuple[float, float, float]) -> None:
        """
        Scales the box coordinate system.
        :param s: Scale parameter in x, y, z direction.
        """
        scale = np.asarray(s)
        assert len(scale) == 3
        self.center *= scale
        self.wlh *= scale
        self.velocity *= scale
        if self.future_centers is not None:
            assert scale.ndim == 1
            assert scale.shape[-1] == self.future_centers.shape[-1]
            self.future_centers *= scale

    def xflip(self) -> None:
        """Flip the box along the X-axis."""
        self.center[0] *= -1
        self.velocity[0] *= -1
        self.angular_velocity *= -1
        if self.future_centers is not None:
            self.future_centers[:, :, 0] *= -1
        current_yaw = quaternion_yaw(self.orientation)
        final_yaw = -current_yaw + np.pi
        self.orientation = Quaternion(axis=(0, 0, 1), angle=final_yaw)
        if self.future_orientations is not None:
            for mode_idx in range(self.num_modes):
                for horizon_idx in range(self.num_future_timesteps):
                    orientation = self.future_orientations[mode_idx][horizon_idx]
                    if orientation is None:
                        continue
                    current_yaw = quaternion_yaw(orientation)
                    final_yaw = -current_yaw + np.pi
                    self.future_orientations[mode_idx][horizon_idx] = Quaternion(axis=(0, 0, 1), angle=final_yaw)

    def yflip(self) -> None:
        """Flip the box along the Y-axis."""
        self.center[1] *= -1
        self.velocity[1] *= -1
        self.angular_velocity *= -1
        if self.future_centers is not None:
            self.future_centers[:, :, 1] *= -1
        current_yaw = quaternion_yaw(self.orientation)
        final_yaw = -current_yaw
        self.orientation = Quaternion(axis=(0, 0, 1), angle=final_yaw)
        if self.future_orientations is not None:
            for mode_idx in range(self.num_modes):
                for horizon_idx in range(self.num_future_timesteps):
                    orientation = self.future_orientations[mode_idx][horizon_idx]
                    if orientation is None:
                        continue
                    current_yaw = quaternion_yaw(orientation)
                    final_yaw = -current_yaw
                    self.future_orientations[mode_idx][horizon_idx] = Quaternion(axis=(0, 0, 1), angle=final_yaw)

    def corners(self, wlh_factor: float=1.0) -> npt.NDArray[np.float64]:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w: float = self.wlh[0] * wlh_factor
        l: float = self.wlh[1] * wlh_factor
        h: float = self.wlh[2] * wlh_factor
        center = tuple(self.center.flatten())
        rotation_matrix = tuple(self.rotation_matrix.flatten())
        return self._calc_corners(w, l, h, center, rotation_matrix)

    @property
    def front_corners(self) -> npt.NDArray[np.float64]:
        """
        Returns the four corners of the front face of the box. First two are on top face while the last two are on the
        bottom face.
        :return: <np.float: 3, 4>. Front corners.
        """
        return self.corners()[:, :4]

    @property
    def rear_corners(self) -> npt.NDArray[np.float64]:
        """
        Returns the four corners of the rear face of the box. First two are on top face while the last two are on the
        bottom face.
        :return: <np.float: 3, 4>. Rear corners.
        """
        return self.corners()[:, 4:]

    @property
    def bottom_corners(self) -> npt.NDArray[np.float64]:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    @property
    def center_bottom_forward(self) -> npt.NDArray[np.float64]:
        """
        Returns the coordinate of the following point: the center of the intersection of the bottom and forward faces
        of the box.
        :return: <np.float: 3, 1>.
        """
        return np.expand_dims(np.mean(self.corners().T[2:4], axis=0), 0).T

    @property
    def front_center(self) -> npt.NDArray[np.float64]:
        """
        Returns the coordinate of the center of the front face of the box.
        :return: <np.float: 3>.
        """
        return np.mean(self.front_corners, axis=1)

    @property
    def rear_center(self) -> npt.NDArray[np.float64]:
        """
        Returns the coordinate of the center of the rear face of the box.
        :return: <np.float: 3>.
        """
        return np.mean(self.rear_corners, axis=1)

    @property
    def bottom_center(self) -> npt.NDArray[np.float64]:
        """
        Returns the coordinate of the bottom face center.
        :return: <np.float: 3>.
        """
        return np.mean(self.bottom_corners, axis=1)

    @property
    def velocity_endpoint(self) -> npt.NDArray[np.float64]:
        """
        Extends the velocity vector from the front bottom center.
        :return: <np.float: 3, 1>.
        """
        return self.center_bottom_forward + np.expand_dims(self.velocity.T, axis=1)

    def get_future_horizon_idx(self, future_horizon_s: float) -> int:
        """
        Gets the index of a future horizon.
        :param future_horizon_s: Future horizon in seconds.
        :return: The index of the future horizon.
        """
        if self.future_horizon_len_s is None or self.future_interval_s is None:
            raise ValueError(f'Future horizon information is not available. Invalid variable values:\nfuture_horizon_len_s={self.future_horizon_len_s}\nfuture_interval_s={self.future_interval_s}.')
        if not 0.0 < future_horizon_s <= self.future_horizon_len_s:
            raise ValueError(f'Future horizon ({future_horizon_s}) should be in (0, {self.future_horizon_len_s}].')
        horizon_idx = round(future_horizon_s / self.future_interval_s - 1, 1)
        if not horizon_idx.is_integer():
            raise ValueError(f'Future horizon ({future_horizon_s}) divided by future interval ({self.future_interval_s}) is not an integer.')
        horizon_idx = int(horizon_idx)
        assert 0 <= horizon_idx < self.num_future_timesteps
        return horizon_idx

    def get_all_future_horizons_s(self) -> List[float]:
        """
        Gets the list of all future horizons.
        :return: The list of all future horizons.
        """
        return [round((horizon_idx + 1) * self.future_interval_s, 2) for horizon_idx in range(self.num_future_timesteps)]

    def get_future_center_at_horizon(self, future_horizon_s: float) -> npt.NDArray[np.float64]:
        """
        Gets future center of the highest probability trajectory at a given horizon.
        :param future_horizon_s: Future horizon in seconds.
        :return: Future center at the given horizon.
        """
        if self.future_centers is None:
            raise ValueError('Future center is not available.')
        highest_prob_mode_idx = self.get_highest_prob_mode_idx()
        horizon_idx = self.get_future_horizon_idx(future_horizon_s)
        return self.future_centers[highest_prob_mode_idx, horizon_idx]

    def get_future_centers_at_horizons(self, future_horizons_s: List[float]) -> npt.NDArray[np.float64]:
        """
        Gets future centers at the given horizons.
        :param future_horizons_s: Future horizons in seconds.
        :return: Future centers at the given horizons.
        """
        if self.future_centers is None:
            raise ValueError('Future center is not available.')
        highest_prob_mode_idx = self.get_highest_prob_mode_idx()
        horizon_indices = [self.get_future_horizon_idx(future_horizon_s) for future_horizon_s in future_horizons_s]
        return self.future_centers[highest_prob_mode_idx, horizon_indices]

    def get_future_orientation_at_horizon(self, future_horizon_s: float) -> Quaternion:
        """
        Gets future orientation of the highest probability trajectory at a given horizon.
        :param future_horizon_s: Future horizon in seconds.
        :return: Future orientation at the given horizon.
        """
        if self.future_orientations is None:
            raise ValueError('Future orientation is not available.')
        highest_prob_mode_idx = self.get_highest_prob_mode_idx()
        horizon_idx = self.get_future_horizon_idx(future_horizon_s)
        return self.future_orientations[highest_prob_mode_idx][horizon_idx]

    def get_future_orientations_at_horizons(self, future_horizons_s: List[float]) -> List[Quaternion]:
        """
        Gets future orientation of the highest probability trajectory at the given horizons.
        :param future_horizons_s: Future horizons in seconds.
        :return: Future orientations at the given horizons.
        """
        if self.future_orientations is None:
            raise ValueError('Future orientation is not available.')
        highest_prob_mode_idx = self.get_highest_prob_mode_idx()
        horizon_indices = [self.get_future_horizon_idx(future_horizon_s) for future_horizon_s in future_horizons_s]
        return [self.future_orientations[highest_prob_mode_idx][horizon_idx] for horizon_idx in horizon_indices]

    def get_topk_future_center_at_horizon(self, future_horizon_s: float, topk: int) -> npt.NDArray[np.float64]:
        """
        Gets top-k future centers at a given horizon.
        :param future_horizon_s: Future horizon in seconds.
        :param topk: The number of top-k modes.
        :return: Future center at the given horizon.
        """
        if self.future_centers is None:
            raise ValueError('Future centers are not available.')
        topk_mode_indices = self.get_topk_mode_indices(topk)
        horizon_idx = self.get_future_horizon_idx(future_horizon_s)
        return self.future_centers[topk_mode_indices, horizon_idx]

    def get_topk_future_orientation_at_horizon(self, future_horizon_s: float, topk: int) -> List[Quaternion]:
        """
        Gets top-k future orientations at a given horizon.
        :param future_horizon_s: Future horizon in seconds.
        :param topk: The number of top-k modes.
        :return: Future orientation at the given horizon.
        """
        if self.future_orientations is None:
            raise ValueError('Future orientations are not available.')
        topk_mode_indices = self.get_topk_mode_indices(topk)
        horizon_idx = self.get_future_horizon_idx(future_horizon_s)
        return [self.future_orientations[mode_idx][horizon_idx] for mode_idx in topk_mode_indices]

    def get_topk_mode_indices(self, topk: int) -> List[int]:
        """
        Gets the indices for the top-k highest probability modes.
        :param topk: Number of top-k modes.
        :return: The list of top-k highest probability mode indices.
        """
        if self.mode_probs is None:
            raise ValueError('Mode probabilities are not available.')
        return self.mode_probs.argsort()[::-1][:topk]

    def get_highest_prob_mode_idx(self) -> int:
        """
        Gets the index of the highest probability mode.
        :return: The index of the highest probability mode.
        """
        return self.get_topk_mode_indices(1)[0]

    def draw_line(self, canvas: Union[plt.Axes, npt.NDArray[np.uint8]], from_x: float, to_x: float, from_y: float, to_y: float, color: Tuple[Union[float, str], Union[float, str], Union[float, str]], linewidth: float, marker: Optional[str]=None, alpha: float=1.0) -> None:
        """
        Draws a line on a matplotlib/cv2 canvas.
        :param canvas: <matplotlib.pyplot.axis> OR <np.array: width, height, 3>.
        Axis/Image onto which the box should be drawn.
        :param from_x: The start x coordinates of vertices.
        :param to_x: The end x coordinates of vertices.
        :param from_y: The start y coordinates of vertices.
        :param to_y: The end y coordinates of vertices.
        :param color: The color used to draw line.
        :param linewidth: Width in pixel of the box sides.
        :param marker: Marker style string to draw line.
        :param alpha: The degree of transparency (or opacity) of a color.
        """
        if isinstance(canvas, np.ndarray):
            color_int = tuple((int(c * 255) for c in color))
            cv2.line(canvas, (int(from_x), int(from_y)), (int(to_x), int(to_y)), color_int[::-1], linewidth)
        else:
            canvas.plot([from_x, to_x], [from_y, to_y], color=color, linewidth=linewidth, marker=marker, alpha=alpha)

    def draw_rect(self, canvas: Union[plt.Axes, npt.NDArray[np.uint8]], selected_corners: npt.NDArray[np.float64], color: Tuple[float, float, float], linewidth: float) -> None:
        """
        Draws a rectangle on a matplotlib/cv2 canvas.
        :param canvas: <matplotlib.pyplot.axis> OR <np.array: width, height, 3>.
        Axis/Image onto which the box should be drawn.
        :param selected_corners: The selected corners for a rectangle.
        :param color: The color used to draw rectangle.
        :param linewidth: Width in pixel of the box sides.
        """
        prev = selected_corners[-1]
        for corner in selected_corners:
            self.draw_line(canvas, prev[0], corner[0], prev[1], corner[1], color=color, linewidth=linewidth)
            prev = corner

    def draw_text(self, canvas: Union[plt.Axes, npt.NDArray[np.uint8]], x: float, y: float, text: str) -> None:
        """
        Draws text on a matplotlib/cv2 canvas.
        :param canvas: <matplotlib.pyplot.axis> OR <np.array: width, height, 3>.
        Axis/Image onto which the box should be drawn.
        :param x: The x coordinates of vertices.
        :param y: The y coordinates of vertices.
        :param text: The text to draw.
        """
        if isinstance(canvas, np.ndarray):
            cv2.putText(canvas, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            canvas.text(x, y, text)

    def render(self, canvas: Union[plt.Axes, npt.NDArray[np.uint8]], view: npt.NDArray[np.float64]=np.eye(3), normalize: bool=False, colors: Tuple[MatplotlibColor, MatplotlibColor, MatplotlibColor]=None, linewidth: float=2, marker: str='o', with_direction: bool=True, with_velocity: bool=False, with_label: bool=False) -> None:
        """
        Renders the box. Canvas can be either a Matplotlib axis or a numpy array image (using cv2).
        :param canvas: <matplotlib.pyplot.axis> OR <np.array: width, height, 3>.
            Axis/Image onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            rear/top and bottom.
        :param linewidth: Width in pixel of the box sides.
        :param marker: Marker style string to draw line.
        :param with_direction: Whether to draw a line indicating box direction.
        :param with_velocity: Whether to draw a line indicating box velocity.
        :param with_label: Whether to render the label.
        """
        corners = self.corners()
        sel = corners[2, :] < 0
        corners[2, sel] *= -1
        corners = view_points(corners, view, normalize=normalize)[:2, :]
        if colors is None:
            color = tuple((c / 255 for c in self.color[:3]))
            colors = (color, color, 'k')
        colors = tuple((matplotlib.colors.to_rgb(c) if isinstance(c, str) else c for c in colors))
        for i in [2, 3]:
            self.draw_line(canvas, corners.T[i][0], corners.T[i + 4][0], corners.T[i][1], corners.T[i + 4][1], color=colors[2], linewidth=linewidth)
        for i in [0, 1]:
            self.draw_line(canvas, corners.T[i][0], corners.T[i + 4][0], corners.T[i][1], corners.T[i + 4][1], color=colors[1], linewidth=linewidth)
        self.draw_rect(canvas, corners.T[:4], colors[0], linewidth)
        self.draw_rect(canvas, corners.T[4:], colors[1], linewidth)
        if with_direction:
            center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
            center_bottom_forward = np.mean(corners.T[2:4], axis=0)
            self.draw_line(canvas, center_bottom[0], center_bottom_forward[0], center_bottom[1], center_bottom_forward[1], color=colors[1], linewidth=linewidth)
        if with_velocity and (not any(np.isnan(self.velocity))):
            center_bottom_forward = np.mean(corners.T[2:4], axis=0)
            velocity_end = view_points(self.velocity_endpoint, view, normalize=normalize)[:2, 0]
            self.draw_line(canvas, center_bottom_forward[0], velocity_end[0], center_bottom_forward[1], velocity_end[1], color=colors[1], linewidth=linewidth * 2, marker='o')
        if with_label:
            org_center = np.expand_dims(self.center, axis=0).T
            proj_center = view_points(org_center, view, normalize=normalize)[:2, 0]
            self.draw_text(canvas, proj_center[0], proj_center[1], str(self.label))
        if self.future_centers is not None:
            for mode_idx in range(self.num_modes):
                mode_prob = self.mode_probs[mode_idx]
                if mode_prob < self.RENDER_MODE_PROB_THRESHOLD:
                    continue
                prev_x, prev_y, _ = self.center
                for horizon_idx in range(self.num_future_timesteps):
                    if self.num_future_timesteps > 1:
                        color_int = tuple((int(c * 255) for c in colors[0]))
                        color = self.fade_color(color_int, horizon_idx, self.num_future_timesteps - 1)
                        color = tuple((c / 255 for c in color))
                    else:
                        color = colors[0]
                    waypoint = self.future_centers[mode_idx, horizon_idx]
                    if waypoint is not None and (not np.isnan(waypoint).any()):
                        next_x, next_y, _ = waypoint
                        alpha = max(1.0 - horizon_idx * 0.1, 0.1) * mode_prob
                        self.draw_line(from_x=prev_x, to_x=next_x, from_y=prev_y, to_y=next_y, color=color, marker=marker, linewidth=linewidth, canvas=canvas, alpha=alpha)
                        prev_x, prev_y = (next_x, next_y)

    @staticmethod
    def fade_color(color: Tuple[int, int, int], step: int, total_number_of_steps: int) -> Tuple[int, int, int]:
        """
        Fades a color so that future observations are darker in the image.
        :param color: Tuple of ints describing an RGB color.
        :param step: The current time step.
        :param total_number_of_steps: The total number of time steps the agent has in the image.
        :return: Tuple representing faded rgb color.
        """
        LOWEST_VALUE = 0.2
        hsv_color = colorsys.rgb_to_hsv(*color)
        increment = (float(hsv_color[2]) / 255.0 - LOWEST_VALUE) / total_number_of_steps
        new_value = float(hsv_color[2]) / 255.0 - step * increment
        new_rgb = colorsys.hsv_to_rgb(float(hsv_color[0]), float(hsv_color[1]), new_value * 255.0)
        new_rgb_int = tuple((int(c) for c in new_rgb))
        return new_rgb_int

    @staticmethod
    @functools.lru_cache()
    def _calc_corners(width: float, length: float, height: float, center: Tuple[float], rotation_matrix: Tuple[float]) -> npt.NDArray[np.float64]:
        """
        Cached helper function to calculate corners from center and size.
        :param w: Width of box.
        :param l: Length of box.
        :param h: Height of box.
        :param center: Center of box.
        :param rotation_matrix: Rotation matrix of box.
        :return: Corners of box given as <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        corners = np.array([[1, 1, 1, 1, -1, -1, -1, -1], [1, -1, -1, 1, 1, -1, -1, 1], [1, 1, -1, -1, 1, 1, -1, -1]], dtype=float)
        corners[0] *= length / 2
        corners[1] *= width / 2
        corners[2] *= height / 2
        rot_mat = np.array(rotation_matrix).reshape(3, 3)
        corners = np.dot(rot_mat, corners)
        corners += np.array(center).reshape((-1, 1))
        return corners

class TestBox3DEncoding(unittest.TestCase):
    """Test Box3D Encoding."""

    def test_simple(self) -> None:
        """Test a Box3D object is still the same after serialize and deserialize."""
        box = Box3D((1, 2, 3), (1, 2, 3), Quaternion(0, 0, 0, 0), label=1, score=1.4)
        self.assertEqual(box, Box3D.deserialize(box.serialize()))

    def test_only_mandatory(self) -> None:
        """Test the only mandatory fields to instantiate a Box3D object."""
        box = Box3D((1, 2, 3), (1, 2, 3), Quaternion(0, 0, 0, 0))
        self.assertEqual(box, Box3D.deserialize(box.serialize()))

    def test_all(self) -> None:
        """Test all the fields to instantiate a Box3D object."""
        box = Box3D((1, 2, 3), (1, 2, 3), Quaternion(0, 0, 0, 0), label=1, score=1.2, velocity=(1, 2, 3), angular_velocity=1, payload=dict({'abc': 'def'}))
        self.assertEqual(box, Box3D.deserialize(box.serialize()))

    def test_random(self) -> None:
        """Test random box. After serialize and deserialize, the box is still the same."""
        for i in range(100):
            box = Box3D.make_random()
            self.assertEqual(box, Box3D.deserialize(box.serialize()))

class TestBox3D(unittest.TestCase):
    """Test Box3D."""

    def test_points_in_box(self) -> None:
        """Test the point_in_box method."""
        vel = (np.nan, np.nan, np.nan)

        def qyaw(yaw: float) -> Quaternion:
            """
            Return a Quaternion given yaw angle.
            :param yaw: Yaw angle.
            :return: A Quaternion object.
            """
            return Quaternion(axis=(0, 0, 1), angle=yaw)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 1.0), qyaw(0.0), 1, 2.0, vel)
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask.all(), True)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 1.0), qyaw(0.0), 1, 2.0, vel)
        points = np.array([[0.1, 0.0, 0.0], [0.5, -1.1, 0.0]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask.all(), False)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 1.0), qyaw(0.0), 1, 2.0, vel)
        points = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask.all(), True)
        rot = 45
        trans = [1.0, 1.0]
        box = Box3D((0.0 + trans[0], 0.0 + trans[1], 0.0), (2.0, 2.0, 1.0), qyaw(rot / 180.0 * np.pi), 1, 2.0, vel)
        points = np.array([[0.7 + trans[0], 0.7 + trans[1], 0.0], [0.71 + 1.0, 0.71 + 1.0, 0.0]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask[0], True)
        self.assertEqual(mask[1], False)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), qyaw(0.0), 1, 2.0, vel)
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask.all(), True)
        for wlh_factor in [0.5, 1.0, 1.5, 10.0]:
            box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 1.0), qyaw(0.0), 1, 2.0, vel)
            points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]).transpose()
            mask = points_in_box(box, points, wlh_factor=wlh_factor)
            self.assertEqual(mask.all(), True)
        for wlh_factor in [0.1, 0.49]:
            box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 1.0), qyaw(0.0), 1, 2.0, vel)
            points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]).transpose()
            mask = points_in_box(box, points, wlh_factor=wlh_factor)
            self.assertEqual(mask[0], True)
            self.assertEqual(mask[1], False)

    def test_points_in_box_bev(self) -> None:
        """Test the points_in_box_bev method."""
        vel = (np.nan, np.nan, np.nan)

        def qyaw(yaw: float) -> Quaternion:
            """
            Return a Quaternion given yaw angle.
            :param yaw: Yaw angle.
            :return: A Quaternion object.
            """
            return Quaternion(axis=(0, 0, 1), angle=yaw)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 1.0), qyaw(0.0), 1, 2.0, vel)
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]).transpose()
        mask = points_in_box_bev(box, points)
        self.assertEqual(mask.all(), True)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 1.0), qyaw(0.0), 1, 2.0, vel)
        points = np.array([[0.1, 0.0, 0.0], [0.5, -1.1, 0.0]]).transpose()
        mask = points_in_box_bev(box, points)
        self.assertEqual(mask.all(), False)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 1.0), qyaw(0.0), 1, 2.0, vel)
        points = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]).transpose()
        mask = points_in_box_bev(box, points)
        self.assertEqual(mask.all(), True)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), qyaw(0.0), 1, 2.0, vel)
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]).transpose()
        mask = points_in_box(box, points, wlh_factor=1.0)
        self.assertEqual(mask.all(), True)
        for center_z in [0.5, 1.0, 1.5, 10.0, 100]:
            box = Box3D((0.0, 0.0, center_z), (2.0, 2.0, 1.0), qyaw(0.0), 1, 2.0, vel)
            points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]).transpose()
            mask = points_in_box_bev(box, points)
            self.assertEqual(mask.all(), True)

    def test_rotate(self) -> None:
        """Test if rotate correctly rotates the box."""
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        theta = np.pi / 2
        box.rotate(Quaternion(axis=(0.0, 0.0, 1.0), angle=theta))
        assert_array_almost_equal(box.bottom_corners[:, 0], np.array([1.0, 1.0, -1.0]))
        assert_array_almost_equal(box.bottom_corners[:, 1], np.array([-1.0, 1.0, -1.0]))
        assert_array_almost_equal(box.bottom_corners[:, 2], np.array([-1.0, -1.0, -1.0]))
        assert_array_almost_equal(box.bottom_corners[:, 3], np.array([1.0, -1.0, -1.0]))

    def test_box_in_image(self) -> None:
        """Test Box at different location in Image."""
        box = Box3D((150.0, 150.0, 150.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        intrinsic = np.eye(3)
        imsize = (300, 300)
        box_in_img = box_in_image(box, intrinsic, imsize)
        self.assertEqual(box_in_img, True)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        box_in_img = box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ALL)
        self.assertEqual(box_in_img, False)
        box = Box3D((0.0, 0.0, 0.0), (0.01, 0.01, 0.05), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        box_in_img = box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY)
        self.assertEqual(box_in_img, False)
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        box_in_img = box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.NONE)
        self.assertEqual(box_in_img, True)
        box = Box3D((-10.0, -90.0, -100.0), (2.0, 2.0, 2.0), Quaternion(axis=(10.0, 20.0, 1.4), angle=20))
        box_in_img = box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.NONE)
        self.assertEqual(box_in_img, True)
        box = Box3D((0.0, 0.0, 3.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        box_in_img = box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY)
        self.assertEqual(box_in_img, True)
        box = Box3D((-2.0, -2.0, -2.0), (1.0, 1.0, 1.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        box_in_img = box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY)
        self.assertEqual(box_in_img, False)
        box = Box3D((10.0, 10.0, 0.51), (1.0, 1.0, 1.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        box_in_img = box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY)
        self.assertEqual(box_in_img, True)
        box = Box3D((150.0, 150.0, 150.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0), velocity=(10.0, 20.0, 3.0))
        box_in_img = box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ALL, with_velocity=True)
        self.assertEqual(box_in_img, True)
        box = Box3D((150.0, 150.0, 2.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0), velocity=(2000.0, 20.0, 3.0))
        box_in_img = box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ALL, with_velocity=True)
        self.assertEqual(box_in_img, False)

    def test_copy(self) -> None:
        """Verify that box copy works as expected."""
        box_orig = Box3D.make_random()
        box_copy = box_orig.copy()
        self.assertEqual(box_orig, box_copy)
        box_orig.center[0] += 1
        self.assertNotEqual(box_orig, box_copy)
        box_orig = Box3D.make_random()
        box_copy = box_orig.copy()
        box_orig.wlh[0] += 1
        self.assertNotEqual(box_orig, box_copy)
        box_orig = Box3D.make_random()
        box_copy = box_orig.copy()
        box_orig.orientation.q[0] += 1
        self.assertNotEqual(box_orig, box_copy)
        box_orig = Box3D.make_random()
        box_copy = box_orig.copy()
        box_orig.label += 1
        self.assertNotEqual(box_orig, box_copy)
        box_orig = Box3D.make_random()
        box_copy = box_orig.copy()
        box_orig.score += 1
        self.assertNotEqual(box_orig, box_copy)
        box_orig = Box3D.make_random()
        box_copy = box_orig.copy()
        box_orig.velocity[0] += 1
        self.assertNotEqual(box_orig, box_copy)
        box_orig = Box3D.make_random()
        box_copy = box_orig.copy()
        box_orig.angular_velocity += 1
        self.assertNotEqual(box_orig, box_copy)
        box_orig = Box3D.make_random()
        box_copy = box_orig.copy()
        box_orig.payload = {'abc': 'def'}
        self.assertNotEqual(box_orig, box_copy)

    def test_translate(self) -> None:
        """Tests box translation performs as expected."""
        box = Box3D((150.0, 120.0, 10.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.2, 0.4, 1.43), angle=30))
        box.translate(np.array([12.3, 0.0, 1.4], dtype=float))
        self.assertTrue(np.array_equal(box.center, [162.3, 120.0, 11.4]))
        box = Box3D((10.0, 1220.0, 1.0), (2.0, 2.0, 2.0), Quaternion(axis=(2.2, 0.24, 0), angle=20))
        box.translate(np.array([-990.0, 10.0, -0.4], dtype=float))
        self.assertTrue(np.array_equal(box.center, [-980.0, 1230.0, 0.6]))
        box = Box3D((10.0, 1220.0, 1.0), (2.0, 2.0, 2.0), Quaternion(axis=(2.2, 0.24, 0), angle=20))
        box.translate(np.array([0.0, 0.0, 0.0], dtype=float))
        self.assertTrue(np.array_equal(box.center, [10.0, 1220.0, 1.0]))

    def test_transform(self) -> None:
        """Tests the equivalence of using box.transform compared to box.translation followed by box.rotation."""
        box1 = Box3D.arbitrary_box()
        box2 = Box3D.arbitrary_box()
        self.assertEqual(box1, box2)
        r1 = Quaternion(np.random.rand(4))
        t1 = np.random.rand(3)
        r2 = Quaternion(np.random.rand(4))
        t2 = np.random.rand(3)
        tf1 = r1.transformation_matrix
        tf1[:3, 3] = t1
        tf2 = r2.transformation_matrix
        tf2[:3, 3] = t2
        tf = np.dot(tf2, tf1)
        box1.rotate(r1)
        box1.translate(t1)
        box1.rotate(r2)
        box1.translate(t2)
        box2.transform(tf)
        self.assertEqual(box1, box2)

    def test_xflip_no_flip(self) -> None:
        """Tests that there is no change."""
        for input_yaw in (np.pi / 2, -np.pi / 2):
            box = Box3D((0, 0, 0), (1, 1, 1), Quaternion(axis=(0, 0, 1), angle=input_yaw))
            box.xflip()
            assert_almost_equal(quaternion_yaw(box.orientation), input_yaw)

    def test_xflip_180_flip(self) -> None:
        """Test flip from left to right and right to left."""
        input_yaw = (0, np.pi)
        output_yaw = (np.pi, 0)
        for in_yaw, out_yaw in zip(input_yaw, output_yaw):
            box = Box3D((0, 0, 0), (1, 1, 1), Quaternion(axis=(0, 0, 1), angle=in_yaw))
            box.xflip()
            assert_almost_equal(quaternion_yaw(box.orientation), out_yaw)

    def test_xflip_pos_yaw(self) -> None:
        """Test flips when starting with positive yaw."""
        for yaw in np.linspace(0, np.pi, 100):
            box = Box3D((0, 0, 0), (1, 1, 1), Quaternion(axis=(0, 0, 1), angle=yaw))
            box.xflip()
            assert_almost_equal(quaternion_yaw(box.orientation), np.pi - yaw)

    def test_xflip_neg_yaw(self) -> None:
        """Test flips when starting with negative yaw."""
        for yaw in np.linspace(-np.pi, -0.0001, 100):
            box = Box3D((0, 0, 0), (1, 1, 1), Quaternion(axis=(0, 0, 1), angle=yaw))
            box.xflip()
            assert_almost_equal(quaternion_yaw(box.orientation), -np.pi - yaw)

    def test_yflip_no_flip(self) -> None:
        """Test that there is no change."""
        for input_yaw in (0, np.pi):
            box = Box3D((0, 0, 0), (1, 1, 1), Quaternion(axis=(0, 0, 1), angle=input_yaw))
            box.yflip()
            assert_almost_equal(quaternion_yaw(box.orientation), -input_yaw)

    def test_yflip_180_flip(self) -> None:
        """Test flip from left to right and right to left."""
        input_yaw = (-np.pi / 2, np.pi / 2)
        output_yaw = (np.pi / 2, -np.pi / 2)
        for in_yaw, out_yaw in zip(input_yaw, output_yaw):
            box = Box3D((0, 0, 0), (1, 1, 1), Quaternion(axis=(0, 0, 1), angle=in_yaw))
            box.yflip()
            assert_almost_equal(quaternion_yaw(box.orientation), out_yaw)

    def test_yflip_pos_yaw(self) -> None:
        """Test flips when starting with positive yaw."""
        for yaw in np.linspace(0, np.pi, 100):
            box = Box3D((0, 0, 0), (1, 1, 1), Quaternion(axis=(0, 0, 1), angle=yaw))
            box.yflip()
            assert_almost_equal(quaternion_yaw(box.orientation), -yaw)

    def test_yflip_neg_yaw(self) -> None:
        """Test flips when starting with negative yaw."""
        for yaw in np.linspace(-np.pi, -0.0001, 100):
            box = Box3D((0, 0, 0), (1, 1, 1), Quaternion(axis=(0, 0, 1), angle=yaw))
            box.yflip()
            assert_almost_equal(quaternion_yaw(box.orientation), -yaw)

    def test_arbitrary_box(self) -> None:
        """Tests arbitrary_box method could initiate a box correctly."""
        box = Box3D.arbitrary_box()
        self.assertTrue(box)
        self.assertEqual(box, Box3D.deserialize(box.serialize()))

    def test_center_bottom_forward(self) -> None:
        """Tests the point of the center of the intersection of the bottom and forward faces of the box."""
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        self.assertEqual(box.center_bottom_forward[0], 1)
        self.assertEqual(box.center_bottom_forward[1], 0)
        self.assertEqual(box.center_bottom_forward[2], -1)

    def test_front_center(self) -> None:
        """Tests the center of the front face of the box."""
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        self.assertEqual(box.front_center[0], 1)
        self.assertEqual(box.front_center[1], 0)
        self.assertEqual(box.front_center[2], 0)

    def test_rear_center(self) -> None:
        """Tests the center of the rear face of the box."""
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        self.assertEqual(box.rear_center[0], -1)
        self.assertEqual(box.rear_center[1], 0)
        self.assertEqual(box.rear_center[2], 0)

    def test_bottom_center(self) -> None:
        """Tests the bottom face center of the box."""
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        self.assertEqual(box.bottom_center[0], 0)
        self.assertEqual(box.bottom_center[1], 0)
        self.assertEqual(box.bottom_center[2], -1)

    def test_velocity_endpoint(self) -> None:
        """Tests the velocity vector is correct."""
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0), velocity=(1.0, 1.0, 1.0))
        self.assertEqual(box.velocity_endpoint[0], 2)
        self.assertEqual(box.velocity_endpoint[1], 1)
        self.assertEqual(box.velocity_endpoint[2], 0)

    def test_corners(self) -> None:
        """Tests if corners change after translation."""
        box = Box3D.make_random()
        corners = box.corners()
        translation: npt.NDArray[np.float64] = np.array([4, 4, 4])
        box.translate(translation)
        corners_translated: npt.NDArray[np.float64] = corners + translation.reshape(-1, 1)
        self.assertTrue(np.allclose(box.corners(), corners_translated))
        box = Box3D.make_random()
        corners = box.corners()
        translation = np.array([np.random.randint(-box.center[0] - CONST_NUM, 0), np.random.randint(-box.center[1] - CONST_NUM, 0), np.random.randint(-box.center[2] - CONST_NUM, 0)])
        box.translate(translation)
        corners_translated = corners + translation.reshape(-1, 1)
        self.assertTrue(np.allclose(box.corners(), corners_translated))

    def test_front_corners(self) -> None:
        """Tests the four corners of the front face of the box."""
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        assert_array_almost_equal(box.front_corners[:, 0], np.array([1, 1, 1]))
        assert_array_almost_equal(box.front_corners[:, 1], np.array([1, -1, 1]))
        assert_array_almost_equal(box.front_corners[:, 2], np.array([1, -1, -1]))
        assert_array_almost_equal(box.front_corners[:, 3], np.array([1, 1, -1]))

    def test_rear_corners(self) -> None:
        """Tests the four corners of the rear face of the box."""
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        assert_array_almost_equal(box.rear_corners[:, 0], np.array([-1, 1, 1]))
        assert_array_almost_equal(box.rear_corners[:, 1], np.array([-1, -1, 1]))
        assert_array_almost_equal(box.rear_corners[:, 2], np.array([-1, -1, -1]))
        assert_array_almost_equal(box.rear_corners[:, 3], np.array([-1, 1, -1]))

    def test_bottom_corners(self) -> None:
        """Tests the four bottom corners of the box."""
        box = Box3D((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), Quaternion(axis=(0.0, 0.0, 1.0), angle=0))
        assert_array_almost_equal(box.bottom_corners[:, 0], np.array([1, -1, -1]))
        assert_array_almost_equal(box.bottom_corners[:, 1], np.array([1, 1, -1]))
        assert_array_almost_equal(box.bottom_corners[:, 2], np.array([-1, 1, -1]))
        assert_array_almost_equal(box.bottom_corners[:, 3], np.array([-1, -1, -1]))

    def test_box_only_size_error(self) -> None:
        """Tests that invalid box sizes get rejected."""
        center = (1, 1, 1)
        quaternion = Quaternion(axis=(0.0, 0.0, 1.0), angle=0)
        size = (-1, 1, 1)
        self.assertRaises(AssertionError, Box3D, center=center, size=size, orientation=quaternion)
        size = (1, -1, 1)
        self.assertRaises(AssertionError, Box3D, center=center, size=size, orientation=quaternion)
        size = (1, 1, -1)
        self.assertRaises(AssertionError, Box3D, center=center, size=size, orientation=quaternion)
        size = (-1, -1, -1)
        self.assertRaises(AssertionError, Box3D, center=center, size=size, orientation=quaternion)

class TestAngleDiff(unittest.TestCase):
    """Unittests for angle difference."""

    def test_angle_diff_2pi(self) -> None:
        """Tests angle diff function for 2 pi."""
        period = 2 * math.pi
        x, y = (math.pi, math.pi)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)
        x, y = (math.pi, -math.pi)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)
        x, y = (-math.pi / 6, math.pi / 6)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -math.pi / 3)
        x, y = (2 * math.pi / 3, -2 * math.pi / 3)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -2 * math.pi / 3)
        x, y = (8 * math.pi / 3, -2 * math.pi / 3)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -2 * math.pi / 3)
        x, y = (0, math.pi)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -math.pi)

    def test_angle_diff_pi(self) -> None:
        """Tests angle diff function for pi."""
        period = math.pi
        x, y = (math.pi, math.pi)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)
        x, y = (math.pi, -math.pi)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)
        x, y = (-math.pi / 6, math.pi / 6)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), -math.pi / 3)
        x, y = (2 * math.pi / 3, -2 * math.pi / 3)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), math.pi / 3)
        x, y = (8 * math.pi / 3, -2 * math.pi / 3)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), math.pi / 3)
        x, y = (0, math.pi)
        self.assertAlmostEqual(measure.angle_diff(x, y, period), 0)

    def test_quaternion(self) -> None:
        """Tests the angle difference between two yaw angles from two quaternions."""
        x = quaternion_yaw(Quaternion(axis=(0, 0, 1), angle=1.1 * np.pi))
        y = quaternion_yaw(Quaternion(axis=(0, 0, 1), angle=0.9 * np.pi))
        diff = measure.angle_diff(x, y, period=2 * np.pi)
        self.assertAlmostEqual(diff, 0.2 * np.pi)

class TestBirdviewCenterDistanceBox(unittest.TestCase):
    """Unit test for birdview center distance."""

    def test_birdview_center_distance(self) -> None:
        """Test the l2 distance between birdview bounding box centers."""
        dist = measure.birdview_center_distance((0.0, 0.0, 1.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(dist, 0)
        dist = measure.birdview_center_distance((0.0, 0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(dist, 1)
        dist = measure.birdview_center_distance((0.0, 0.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0))
        self.assertAlmostEqual(dist, 1.4142135623730951)

    def test_birdview_center_distance_box(self) -> None:
        """Test the l2 distance between birdview bounding box centers in Box3D class format."""
        dist = measure.birdview_center_distance_box(Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)))
        self.assertEqual(dist, 0)
        dist = measure.birdview_center_distance_box(Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)))
        self.assertEqual(dist, 1)
        dist = measure.birdview_center_distance_box(Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 1, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)))
        self.assertAlmostEqual(dist, 1.4142135623730951)
        dist1 = measure.birdview_center_distance_box(Box3D((4, 5, 0), (2, 2, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 4, 0), (2, 4, 1), Quaternion(axis=(0, 0, 1), angle=np.pi / 3)))
        dist2 = measure.birdview_center_distance((4.0, 5.0, 2.0, 2.0, 0.0), (1.0, 4.0, 2.0, 4.0, np.pi / 3.0))
        self.assertEqual(dist1, dist2)

class TestHausdorffDistance(unittest.TestCase):
    """Unit test for hausdorff_distance"""

    def test_hausdorff_distance(self) -> None:
        """Test Hausdorff distance between two 2d-boxes"""
        dist = measure.hausdorff_distance((0.0, 0.0, 1.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(dist, 0)
        dist = measure.hausdorff_distance((0.0, 0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(dist, 1.0)
        dist = measure.hausdorff_distance((0.0, 0.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0))
        self.assertAlmostEqual(dist, 1.4142135623730951)
        dist = measure.hausdorff_distance((1.0, 1.0, 1.0, 2.0, 0.0), (1.0, 1.0, 1.0, 2.0, np.pi / 2.0))
        self.assertAlmostEqual(dist, 0.5)
        dist = measure.hausdorff_distance((1.0, 1.0, 1.0, 2.0, 0.0), (1.0, 1.5, 1.0, 2.0, 0.0))
        self.assertAlmostEqual(dist, 0.5)
        dist = measure.hausdorff_distance((1.0, 1.0, 1.0, 2.0, 0.0), (1.0, 2.0, 2.0, 4.0, 0.0))
        self.assertAlmostEqual(dist, np.sqrt(0.5 ** 2 + 2 ** 2))
        dist = measure.hausdorff_distance((0.0, 0.0, 2.0 / np.sqrt(2.0), 2.0 / np.sqrt(2.0), 0.0), (0.0, 1.0 / np.sqrt(2.0), 1.0, 1.0, np.pi / 4.0))
        self.assertAlmostEqual(dist, 1)

    def test_hausdorff_distance_box(self) -> None:
        """Test Hausdorff distance between two 2d-boxes in Box3D class."""
        dist = measure.hausdorff_distance_box(Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((0, 0, 0), (1, 1, 10), Quaternion(0, 0, 0, 0)))
        self.assertEqual(dist, 0)
        dist = measure.hausdorff_distance_box(Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)))
        self.assertEqual(dist, 1.0)
        dist = measure.hausdorff_distance_box(Box3D((0, 0, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 1, 0), (1, 1, 1), Quaternion(0, 0, 0, 0)))
        self.assertAlmostEqual(dist, 1.4142135623730951)
        dist1 = measure.hausdorff_distance_box(Box3D((4, 5, 0), (2, 2, 1), Quaternion(0, 0, 0, 0)), Box3D((1, 4, 0), (2, 4, 1), Quaternion(axis=(0, 0, 1), angle=np.pi / 3)))
        dist2 = measure.hausdorff_distance((4.0, 5.0, 2.0, 2.0, 0.0), (1.0, 4.0, 2.0, 4.0, np.pi / 3.0))
        self.assertEqual(dist1, dist2)

class TestPseudoIOU(unittest.TestCase):
    """Test the birdview_pseudo_iou metric."""

    def test_pseudo_distance_2pi(self) -> None:
        """Test ad-hoc birdview distance of two 2-d boxes with period of 2 pi."""
        period = 2 * np.pi
        a = (0.0, 0.0, 1.0, 2.0, 0.0)
        b = (0.0, 0.0, 1.0, 2.0, 0.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)
        a = (0.0, 0.0, 1.0, 2.0, 0.0)
        b = (0.0, 0.0, 1.0, 2.0, 2.0 * math.pi)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)
        a = (-10.0, 10.0, 0.1, 20.0, 0.0)
        b = (-10.0, 10.0, 0.1, 20.0, math.pi / 2.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), math.pi / 2 / 5)
        a = (-10, 10, 0.1, 20, 0)
        b = (-10, 10, 0.1, 20, math.pi)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), math.pi / 5)
        a = (-100.0, -100.0, 100.0, 100.0, 0.0)
        b = (0.0, 0.0, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 398 / 5)
        a = (-100.0, -100.0, 100.0, 100.0, 0.0)
        b = (0.0, 0.0, 1.0, 1.0, math.pi / 2.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), (398 + math.pi / 2) / 5)

    def test_pseudo_distance_pi(self) -> None:
        """Test ad-hoc birdview distance of two 2-d boxes with period of pi."""
        period = np.pi
        a = (0.0, 0.0, 1.0, 2.0, 0.0)
        b = (0.0, 0.0, 1.0, 2.0, 0.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)
        a = (0.0, 0.0, 1.0, 2.0, 0.0)
        b = (0.0, 0.0, 1.0, 2.0, 2.0 * math.pi)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)
        a = (-10.0, 10.0, 0.1, 20.0, 0.0)
        b = (-10.0, 10.0, 0.1, 20.0, math.pi / 2.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), math.pi / 2 / 5)
        a = (-10.0, 10.0, 0.1, 20.0, 0.0)
        b = (-10.0, 10.0, 0.1, 20.0, math.pi)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 0)
        a = (-100.0, -100.0, 100.0, 100.0, 0.0)
        b = (0.0, 0.0, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), 398 / 5)
        a = (-100.0, -100.0, 100.0, 100.0, 0.0)
        b = (0.0, 0.0, 1.0, 1.0, math.pi / 2.0)
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance(a, b, period=period), (398 + math.pi / 2) / 5)

    def test_pseudo_distance_box_pi(self) -> None:
        """Unit test for calculating ad-hoc birdview distance of two Box3D instances with period of pi."""
        period = np.pi
        a = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)
        a = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=2 * math.pi))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)
        a = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi / 2))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), math.pi / 2 / 5)
        a = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)
        a = Box3D(center=(-100, -100, 0), size=(100, 100, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 1, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 398 / 5)
        a = Box3D(center=(-100, -100, 0), size=(100, 100, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 1, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi / 2))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), (398 + math.pi / 2) / 5)

    def test_pseudo_distance_box_2pi(self) -> None:
        """Unit test for calculating ad-hoc birdview distance of two Box3D instances with period of 2 * pi."""
        period = 2 * np.pi
        a = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)
        a = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 2, 1), orientation=Quaternion(axis=[0, 0, 1], angle=2 * math.pi))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 0)
        a = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi / 2))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), math.pi / 2 / 5)
        a = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(-10, 10, 0), size=(0.1, 20, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), math.pi / 5)
        a = Box3D(center=(-100, -100, 0), size=(100, 100, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 1, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), 398 / 5)
        a = Box3D(center=(-100, -100, 0), size=(100, 100, 1), orientation=Quaternion(axis=[0, 0, 1], angle=0))
        b = Box3D(center=(0, 0, 0), size=(1, 1, 1), orientation=Quaternion(axis=[0, 0, 1], angle=math.pi / 2))
        self.assertAlmostEqual(measure.birdview_corner_angle_mean_distance_box(a, b, period=period), (398 + math.pi / 2) / 5)

class TestQuaternionYaw(unittest.TestCase):
    """Test QuaternionYaw."""

    def test_quaternion_yaw(self) -> None:
        """Test valid and invalid inputs for quaternion_yaw()."""
        for yaw_in in np.linspace(-10, 10, 100):
            q = Quaternion(axis=(0, 0, 1), angle=yaw_in)
            yaw_true = yaw_in % (2 * np.pi)
            if yaw_true > np.pi:
                yaw_true -= 2 * np.pi
            yaw_test = quaternion_yaw(q)
            self.assertAlmostEqual(yaw_true, yaw_test)
        yaw_in = np.pi / 4
        q = Quaternion(axis=(0, 0, 0.5), angle=yaw_in)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)
        yaw_in = np.pi / 4
        q = Quaternion(axis=(0, 0, -1), angle=yaw_in)
        yaw_test = -quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)
        yaw_in = np.pi / 4
        q = Quaternion(axis=(0, 1, 0), angle=yaw_in)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(0, yaw_test)
        yaw_in = np.pi / 2
        q = Quaternion(axis=(0, 1, 1), angle=yaw_in)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)
        yaw_in = np.pi / 2
        q = Quaternion(axis=(0, 0, 1), angle=yaw_in) * Quaternion(axis=(0, 1, 0), angle=0.5821)
        yaw_test = quaternion_yaw(q)
        self.assertAlmostEqual(yaw_in, yaw_test)

