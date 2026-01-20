# Cluster 12

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

def simple_repr(record: Any) -> str:
    """
    Simple renderer for a SQL table
    :param record: A table record.
    :return: A string description of the record.
    """
    out = '{:28}: {}\n'.format('token', record.token)
    columns = None
    if hasattr(record, '__table__'):
        columns = {c for c in record.__table__.columns.keys()}
    for field, value in vars(record).items():
        if columns and field not in columns:
            continue
        if not (field[0] == '_' or field == 'token'):
            out += '{:28}: {}\n'.format(field, value)
    return out + '\n'

class TrafficLightStatus(Base):
    """
    Traffic Light Statuses in a Log.
    """
    __tablename__ = 'traffic_light_status'
    token: str = Column(sql_types.HexLen8, primary_key=True)
    lidar_pc_token: str = Column(sql_types.HexLen8, ForeignKey('lidar_pc.token'), nullable=False)
    lane_connector_id: int = Column(Integer)
    status: str = Column(String(8))
    lidar_pc: LidarPc = relationship('LidarPc', foreign_keys=[lidar_pc_token], back_populates='traffic_lights')

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

class Camera(Base):
    """
    Defines a calibrated camera used to record a particular log.
    """
    __tablename__ = 'camera'
    token = Column(sql_types.HexLen8, primary_key=True)
    log_token = Column(sql_types.HexLen8, ForeignKey('log.token'), nullable=False)
    channel = Column(String(64))
    model = Column(String(64))
    translation = Column(sql_types.SqlTranslation)
    rotation = Column(sql_types.SqlRotation)
    intrinsic = Column(sql_types.SqlCameraIntrinsic)
    distortion = Column(PickleType)
    width = Column(Integer)
    height = Column(Integer)

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return : The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def intrinsic_np(self) -> npt.NDArray[np.float64]:
        """
        Get the intrinsic in numpy format.
        :return: <np.float: 3, 3> Camera intrinsic.
        """
        return np.array(self.intrinsic)

    @property
    def distortion_np(self) -> npt.NDArray[np.float64]:
        """
        Get the distortion in numpy format.
        :return: <np.float: N> Camera distrotion.
        """
        return np.array(self.distortion)

    @property
    def translation_np(self) -> npt.NDArray[np.float64]:
        """
        Get the translation in numpy format.
        :return: <np.float: 3> Translation.
        """
        return np.array(self.translation)

    @property
    def quaternion(self) -> Quaternion:
        """
        Get the rotation in quaternion.
        :return: Rotation in quaternion.
        """
        return Quaternion(self.rotation)

    @property
    def trans_matrix(self) -> npt.NDArray[np.float64]:
        """
        Get the transformation matrix.
        :return: <np.float: 4, 4>. Transformation matrix.
        """
        tm: npt.NDArray[np.float64] = self.quaternion.transformation_matrix
        tm[:3, 3] = self.translation_np
        return tm

    @property
    def trans_matrix_inv(self) -> npt.NDArray[np.float64]:
        """
        Get the inverse transformation matrix.
        :return: <np.float: 4, 4>. Inverse transformation matrix.
        """
        tm: npt.NDArray[np.float64] = np.eye(4)
        rot_inv = self.quaternion.rotation_matrix.T
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(np.transpose(-self.translation_np))
        return tm

class Category(Base):
    """
    A category within our taxonomy. Includes both things (e.g. cars) or stuff (e.g. lanes, sidewalks).
    Subcategories are delineated by a period.
    """
    __tablename__ = 'category'
    token = Column(sql_types.HexLen8, primary_key=True)
    name = Column(String(64))
    description = Column(Text)

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def color(self) -> Tuple[int, int, int]:
        """
        Get category color.
        :return: The category color tuple.
        """
        c: Tuple[int, int, int] = default_color(self.name)
        return c

    @property
    def color_np(self) -> npt.NDArray[np.float64]:
        """
        Get category color in numpy.
        :return: The category color in numpy.
        """
        c: npt.NDArray[np.float64] = default_color_np(self.name)
        return c

class Scene(Base):
    """
    Scenes in a Log.
    """
    __tablename__ = 'scene'
    token: str = Column(sql_types.HexLen8, primary_key=True)
    log_token: str = Column(sql_types.HexLen8, ForeignKey('log.token'), nullable=False)
    name: str = Column(Text)
    goal_ego_pose_token: str = Column(sql_types.HexLen8, ForeignKey('ego_pose.token'), nullable=True)
    roadblock_ids: str = Column(Text)
    goal_ego_pose: EgoPose = relationship('EgoPose', foreign_keys=[goal_ego_pose_token], back_populates='scene')

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

class ScenarioTag(Base):
    """
    Scenarios Tags for a scene.
    """
    __tablename__ = 'scenario_tag'
    token: str = Column(sql_types.HexLen8, primary_key=True)
    lidar_pc_token: str = Column(sql_types.HexLen8, ForeignKey('lidar_pc.token'), nullable=False)
    type: str = Column(Text)
    agent_track_token: str = Column(sql_types.HexLen8, ForeignKey('track.token'), nullable=False)
    lidar_pc: LidarPc = relationship('LidarPc', foreign_keys=[lidar_pc_token], back_populates='scenario_tags')

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

class EgoPose(Base):
    """
    Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system.
    """
    __tablename__ = 'ego_pose'
    token = Column(sql_types.HexLen8, primary_key=True)
    timestamp = Column(Integer)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    qw: float = Column(Float)
    qx: float = Column(Float)
    qy: float = Column(Float)
    qz: float = Column(Float)
    vx = Column(Float)
    vy = Column(Float)
    vz = Column(Float)
    acceleration_x = Column(Float)
    acceleration_y = Column(Float)
    acceleration_z = Column(Float)
    angular_rate_x = Column(Float)
    angular_rate_y = Column(Float)
    angular_rate_z = Column(Float)
    epsg = Column(Integer)
    log_token = Column(sql_types.HexLen8, ForeignKey('log.token'), nullable=False)

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def quaternion(self) -> Quaternion:
        """
        Get the orientation of ego vehicle as quaternion respect to global coordinate system.
        :return: The orientation in quaternion.
        """
        return Quaternion(self.qw, self.qx, self.qy, self.qz)

    @property
    def translation_np(self) -> npt.NDArray[np.float64]:
        """
        Position of ego vehicle respect to global coordinate system.
        :return: <np.float: 3> Translation.
        """
        return np.array([self.x, self.y, self.z])

    @property
    def trans_matrix(self) -> npt.NDArray[np.float64]:
        """
        Get the transformation matrix.
        :return: <np.float: 4, 4>. Transformation matrix.
        """
        tm: npt.NDArray[np.float64] = self.quaternion.transformation_matrix
        tm[:3, 3] = self.translation_np
        return tm

    @property
    def trans_matrix_inv(self) -> npt.NDArray[np.float64]:
        """
        Get the inverse transformation matrix.
        :return: <np.float: 4, 4>. Inverse transformation matrix.
        """
        tm: npt.NDArray[np.float64] = np.eye(4)
        rot_inv = self.quaternion.rotation_matrix.T
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(np.transpose(-self.translation_np))
        return tm

    def rotate_2d_points2d_to_ego_vehicle_frame(self, points2d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Rotate 2D points from global frame to ego-vehicle frame.
        :param points2d: <np.float: num_points, 2>. 2D points in global frame.
        :return: <np.float: num_points, 2>. 2D points rotated to ego-vehicle frame.
        """
        points3d: npt.NDArray[np.float32] = np.concatenate((points2d, np.zeros_like(points2d[:, 0:1])), axis=-1)
        rotation = R.from_matrix(self.quaternion.rotation_matrix.T)
        ego_rotation_angle = rotation.as_euler('zxy', degrees=True)[0]
        xy_rotation = R.from_euler('z', ego_rotation_angle, degrees=True)
        rotated_points3d = xy_rotation.apply(points3d)
        rotated_points2d: npt.NDArray[np.float64] = rotated_points3d[:, :2]
        return rotated_points2d

    def get_map_crop(self, maps_db: Optional[GPKGMapsDB], xrange: Tuple[float, float], yrange: Tuple[float, float], map_layer_name: str, rotate_face_up: bool, target_imsize_xy: Optional[Tuple[float, float]]=None) -> Tuple[Optional[npt.NDArray[np.float64]], npt.NDArray[np.float64], Tuple[float, ...]]:
        """
        This function returns the crop of the map centered at the current ego-pose with the given xrange and yrange.
        :param maps_db: Map database associated with this database.
        :param xrange: The range in x direction in meters relative to the current ego-pose. Eg: (-60, 60]).
        :param yrange: The range in y direction in meters relative to the current ego-pose Eg: (-60, 60).
        :param map_layer_name: A relevant map layer. Eg: 'drivable_area' or 'intensity'.
        :param rotate_face_up: Boolean indicating whether to rotate the image face up with respect to ego-pose.
        :param target_imsize_xy: The target grid xy dimensions for the output array. The xy resolution in meters / grid
            may be scaled by zooming to the desired dimensions.
        :return: (map_crop, map_translation, map_scale). Where:
            map_crop: The desired crop of the map.
            map_translation: The translation in map coordinates from the origin to the ego-pose.
            map_scale: Map scale (inverse of the map precision). This will be a tuple specifying the zoom in both the x
                and y direction if the target_imsize_xy parameter was set, which causes the resolution to change.

            map_scale and map_translation are useful for transforming objects like pointcloud/boxes to the map_crop.
            Refer to render_on_map().
        """
        if maps_db is None:
            precision: float = 1

            def to_pixel_coords(x: float, y: float) -> Tuple[float, float]:
                """
                Get the image coordinates given the x-y coordinates of point. This implementation simply returns the
                same coordinates.
                :param x: Global x coordinate.
                :param y: Global y coordinate.
                :return: Pixel coordinates in map.
                """
                return (x, y)
        else:
            map_layer = maps_db.load_layer(self.log.map_version, map_layer_name)
            precision = map_layer.precision
            to_pixel_coords = map_layer.to_pixel_coords
        map_scale: Tuple[float, ...] = (1.0 / precision, 1.0 / precision, 1.0)
        ego_translation = self.translation_np
        center_x, center_y = to_pixel_coords(ego_translation[0], ego_translation[1])
        center_x, center_y = (int(center_x), int(center_y))
        top_left = (int(xrange[0] * map_scale[0]), int(yrange[0] * map_scale[1]))
        bottom_right = (int(xrange[1] * map_scale[0]), int(yrange[1] * map_scale[1]))
        rotation = R.from_matrix(self.quaternion.rotation_matrix.T)
        ego_rotation_angle = rotation.as_euler('zxy', degrees=True)[0]
        xy_rotation = R.from_euler('z', ego_rotation_angle, degrees=True)
        map_rotate = 0
        rotated = xy_rotation.apply([[top_left[0], top_left[1], 0], [top_left[0], bottom_right[1], 0], [bottom_right[0], top_left[1], 0], [bottom_right[0], bottom_right[1], 0]])[:, :2]
        rect = cv2.minAreaRect(np.hstack([rotated[:, :1] + center_x, rotated[:, 1:] + center_y]).astype(int))
        rect_angle = rect[2]
        cropped_dimensions: npt.NDArray[np.float32] = np.array([map_scale[0] * (xrange[1] - xrange[0]), map_scale[1] * (yrange[1] - yrange[0])])
        rect = (rect[0], cropped_dimensions, rect_angle)
        rect_angle = rect[2]
        cropped_dimensions = np.array([map_scale[0] * (xrange[1] - xrange[0]), map_scale[1] * (yrange[1] - yrange[0])])
        if rect_angle >= 0:
            rect = (rect[0], cropped_dimensions, rect_angle - 90)
        else:
            rect = (rect[0], cropped_dimensions, rect_angle)
        if ego_rotation_angle < -90:
            map_rotate = -90
        if -90 < ego_rotation_angle < 0:
            map_rotate = 0
        if 0 < ego_rotation_angle < 90:
            map_rotate = 90
        if 90 < ego_rotation_angle < 180:
            map_rotate = 180
        if map_layer is None:
            map_crop = None
        else:
            map_crop = crop_rect(map_layer.data, rect)
            map_crop = ndimage.rotate(map_crop, map_rotate, reshape=False)
            if rotate_face_up:
                map_crop = np.rot90(map_crop)
        if map_layer is None:
            map_upper_left_offset_from_global_coordinate_origin = np.zeros((2,))
        else:
            map_upper_left_offset_from_global_coordinate_origin = np.array([-map_layer.transform_matrix[0, -1], map_layer.transform_matrix[1, -1]])
        ego_offset_from_map_upper_left: npt.NDArray[np.float32] = np.array([center_x, -center_y])
        crop_upper_left_offset_from_ego: npt.NDArray[np.float32] = np.array([xrange[0] * map_scale[0], yrange[0] * map_scale[1]])
        map_translation: npt.NDArray[np.float64] = -map_upper_left_offset_from_global_coordinate_origin - ego_offset_from_map_upper_left - crop_upper_left_offset_from_ego
        map_translation_with_z: npt.NDArray[np.float64] = np.array([map_translation[0], map_translation[1], 0])
        if target_imsize_xy is not None:
            zoom_size_x = target_imsize_xy[0] / cropped_dimensions[0]
            zoom_size_y = target_imsize_xy[1] / cropped_dimensions[1]
            map_crop = ndimage.zoom(map_crop, [zoom_size_x, zoom_size_y])
            map_scale = (zoom_size_x, zoom_size_y)
        return (map_crop, map_translation_with_z, map_scale)

    def get_vector_map(self, maps_db: Optional[GPKGMapsDB], xrange: Tuple[float, float], yrange: Tuple[float, float], connection_scales: Optional[List[int]]=None) -> VectorMapNp:
        """
        This function returns the crop of baseline paths (blps) map centered at the current ego-pose with
        the given xrange and yrange.
        :param maps_db: Map database associated with this database.
        :param xrange: The range in x direction in meters relative to the current ego-pose. Eg: [-60, 60].
        :param yrange: The range in y direction in meters relative to the current ego-pose Eg: [-60, 60].
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        map_version = self.lidar_pc.log.map_version.replace('.gpkg', '')
        blps_gdf = maps_db.load_vector_layer(map_version, 'baseline_paths')
        lane_poly_gdf = maps_db.load_vector_layer(map_version, 'lanes_polygons')
        intersections_gdf = maps_db.load_vector_layer(map_version, 'intersections')
        lane_connectors_gdf = maps_db.load_vector_layer(map_version, 'lane_connectors')
        lane_groups_gdf = maps_db.load_vector_layer(map_version, 'lane_groups_polygons')
        if blps_gdf is None or lane_poly_gdf is None or intersections_gdf is None or (lane_connectors_gdf is None) or (lane_groups_gdf is None):
            coords: npt.NDArray[np.float32] = np.empty([0, 2, 2], dtype=np.float32)
            if not connection_scales:
                connection_scales = [1]
            multi_scale_connections: Dict[int, Any] = {scale: np.empty([0, 2], dtype=np.int64) for scale in connection_scales}
            return VectorMapNp(coords=coords, multi_scale_connections=multi_scale_connections)
        blps_in_lanes = blps_gdf[blps_gdf['lane_fid'].notna()]
        blps_in_intersections = blps_gdf[blps_gdf['lane_connector_fid'].notna()]
        lane_group_info = lane_poly_gdf[['lane_fid', 'lane_group_fid']]
        blps_in_lanes = blps_in_lanes.merge(lane_group_info, on='lane_fid', how='outer')
        lane_connectors_gdf['lane_connector_fid'] = lane_connectors_gdf['fid']
        lane_conns_info = lane_connectors_gdf[['lane_connector_fid', 'intersection_fid', 'exit_lane_fid', 'entry_lane_fid']]
        lane_conns_info = lane_conns_info.astype({'lane_connector_fid': int})
        blps_in_intersections = blps_in_intersections.astype({'lane_connector_fid': int})
        blps_in_intersections = blps_in_intersections.merge(lane_conns_info, on='lane_connector_fid', how='outer')
        lane_blps_info = blps_in_lanes[['fid', 'lane_fid']]
        from_blps_info = lane_blps_info.rename(columns={'fid': 'from_blp', 'lane_fid': 'exit_lane_fid'})
        to_blps_info = lane_blps_info.rename(columns={'fid': 'to_blp', 'lane_fid': 'entry_lane_fid'})
        blps_in_intersections = blps_in_intersections.merge(from_blps_info, on='exit_lane_fid', how='inner')
        blps_in_intersections = blps_in_intersections.merge(to_blps_info, on='entry_lane_fid', how='inner')
        candidate_lane_groups, candidate_intersections = get_candidates(self.translation_np, xrange, yrange, lane_groups_gdf, intersections_gdf)
        candidate_blps_in_lanes = blps_in_lanes[blps_in_lanes['lane_group_fid'].isin(candidate_lane_groups['fid'].astype(int))]
        candidate_blps_in_intersections = blps_in_intersections[blps_in_intersections['intersection_fid'].isin(candidate_intersections['fid'].astype(int))]
        ls_coordinates_list: List[List[List[float]]] = []
        ls_connections_list: List[List[int]] = []
        ls_groupings_list: List[List[int]] = []
        cross_blp_connection: Dict[str, List[int]] = dict()
        build_lane_segments_from_blps(candidate_blps_in_lanes, ls_coordinates_list, ls_connections_list, ls_groupings_list, cross_blp_connection)
        build_lane_segments_from_blps(candidate_blps_in_intersections, ls_coordinates_list, ls_connections_list, ls_groupings_list, cross_blp_connection)
        for blp_id, blp_info in cross_blp_connection.items():
            connect_blp_predecessor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_connections_list)
            connect_blp_successor(blp_id, candidate_blps_in_intersections, cross_blp_connection, ls_connections_list)
        ls_coordinates: npt.NDArray[np.float64] = np.asarray(ls_coordinates_list, self.translation_np.dtype)
        ls_connections: npt.NDArray[np.int64] = np.asarray(ls_connections_list, np.int64)
        ls_coordinates = ls_coordinates.reshape(-1, 2)
        ls_coordinates = ls_coordinates - self.translation_np[:2]
        ls_coordinates = self.rotate_2d_points2d_to_ego_vehicle_frame(ls_coordinates)
        ls_coordinates = ls_coordinates.reshape(-1, 2, 2).astype(np.float32)
        if connection_scales:
            multi_scale_connections = generate_multi_scale_connections(ls_connections, connection_scales)
        else:
            multi_scale_connections = {1: ls_connections}
        return VectorMapNp(coords=ls_coordinates, multi_scale_connections=multi_scale_connections)

class Image(Base):
    """
    An image.
    """
    __tablename__ = 'image'
    token = Column(sql_types.HexLen8, primary_key=True)
    next_token = Column(sql_types.HexLen8, ForeignKey('image.token'), nullable=True)
    prev_token = Column(sql_types.HexLen8, ForeignKey('image.token'), nullable=True)
    ego_pose_token = Column(sql_types.HexLen8, ForeignKey('ego_pose.token'), nullable=False)
    camera_token = Column(sql_types.HexLen8, ForeignKey('camera.token'), nullable=False)
    filename_jpg = Column(String(128))
    timestamp = Column(Integer)
    next = relationship('Image', foreign_keys=[next_token], remote_side=[token])
    prev = relationship('Image', foreign_keys=[prev_token], remote_side=[token])
    camera = relationship('Camera', foreign_keys=[camera_token], back_populates='images')
    ego_pose = relationship('EgoPose', foreign_keys=[ego_pose_token], back_populates='image')

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

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
        Returns the Log containing the image.
        :return: The log containing this image.
        """
        return self.camera.log

    @property
    def lidar_pc(self) -> LidarPc:
        """
        Get the closest LidarPc by timestamp
        :return: LidarPc closest to the Image by time
        """
        lidar_pc = self._session.query(LidarPc).order_by(func.abs(LidarPc.timestamp - self.timestamp)).first()
        return lidar_pc

    @property
    def scene(self) -> Scene:
        """
        Get the corresponding scene by finding the closest LidarPc by timestamp.
        :return: Scene corresponding to the Image.
        """
        return self.lidar_pc.scene

    @property
    def lidar_boxes(self) -> LidarBox:
        """
        Get the list of boxes associated with this Image, based on closest LidarPc
        :return: List of boxes associated with this Image
        """
        return self.lidar_pc.lidar_boxes

    def load_as(self, db: NuPlanDB, img_type: str) -> Any:
        """
        Loads the image as a desired type.
        :param db: Log Database.
        :param img_type: Can be either 'pil' or 'np' or 'cv2'. If the img_type is cv2, the image is returned in BGR
            format, otherwise it is returned in RGB format.
        :return: The image.
        """
        assert img_type in ['pil', 'cv2', 'np'], f'Expected img_type to be pil, cv2 or np. Received {img_type}'
        pil_img = PIL.Image.open(self.load_bytes_jpg(db))
        if img_type == 'pil':
            return pil_img
        elif img_type == 'np':
            return np.array(pil_img)
        else:
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @property
    def filename(self) -> str:
        """
        Get the file name.
        :return: The file name.
        """
        return self.filename_jpg

    def load_bytes_jpg(self, db: NuPlanDB) -> BinaryIO:
        """
        Returns the bytes of the jpg data for this image.
        :param db: Log Database.
        :return: The image bytes.
        """
        blob: BinaryIO = db.load_blob(osp.join('sensor_blobs', self.filename))
        return blob

    def path(self, db: NuPlanDB) -> str:
        """
        Get the path to image file.
        :param db: Log Database.
        :return: The image file path.
        """
        return osp.join(db.data_root, self.filename)

    def boxes(self, frame: Frame=Frame.GLOBAL) -> List[Box3D]:
        """
        Loads all boxes associated with this Image record. Boxes are returned in the global frame by default.
        :param frame: Specify the frame in which the boxes will be returned.
        :return: List of boxes.
        """
        boxes: List[Box3D] = get_boxes(self, frame, self.ego_pose.trans_matrix_inv, self.camera.trans_matrix_inv)
        return boxes

    def future_or_past_ego_poses(self, number: int, mode: str, direction: str) -> List[EgoPose]:
        """
        Get n future or past vehicle poses. Note here the frequency of pose differs from frequency of Image.
        :param number: Number of poses to fetch or number of seconds of ego poses to fetch.
        :param mode: Either n_poses or n_seconds.
        :param direction: Future or past ego poses to fetch, could be 'prev' or 'next'.
        :return: List of up to n or n seconds future or past ego poses.
        """
        ego_poses: List[EgoPose]
        if direction == 'prev':
            if mode == 'n_poses':
                ego_poses = self._session.query(EgoPose).filter(EgoPose.timestamp < self.ego_pose.timestamp, self.camera.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.desc()).limit(number).all()
                return ego_poses
            elif mode == 'n_seconds':
                ego_poses = self._session.query(EgoPose).filter(EgoPose.timestamp - self.ego_pose.timestamp < 0, EgoPose.timestamp - self.ego_pose.timestamp >= -number * 1000000.0, self.camera.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.desc()).all()
                return ego_poses
            else:
                raise NotImplementedError('Only n_poses and n_seconds two modes are supported for now!')
        elif direction == 'next':
            if mode == 'n_poses':
                ego_poses = self._session.query(EgoPose).filter(EgoPose.timestamp > self.ego_pose.timestamp, self.camera.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.asc()).limit(number).all()
                return ego_poses
            elif mode == 'n_seconds':
                ego_poses = self._session.query(EgoPose).filter(EgoPose.timestamp - self.ego_pose.timestamp > 0, EgoPose.timestamp - self.ego_pose.timestamp <= number * 1000000.0, self.camera.log_token == EgoPose.log_token).order_by(EgoPose.timestamp.asc()).all()
                return ego_poses
            else:
                raise NotImplementedError('Only n_poses and n_seconds two modes are supported!')
        else:
            raise ValueError('Only prev and next two directions are supported!')

    def render(self, db: NuPlanDB, with_3d_anns: bool=True, box_vis_level: BoxVisibility=BoxVisibility.ANY, ax: Optional[Axes]=None) -> None:
        """
        Render the image with all 3d and 2d annotations.
        :param db: Log Database.
        :param with_3d_anns: Whether you want to render 3D boxes?
        :param box_vis_level: One of the enumerations of <BoxVisibility>.
        :param ax: Axes object or array of Axes objects.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 16))
        ax.imshow(self.load_as(db, img_type='pil'))
        if with_3d_anns:
            for box in self.boxes(Frame.SENSOR):
                ann_record = db.lidar_box[box.token]
                c = ann_record.category.color_np
                color = (c, c, np.array([0, 0, 0]))
                if box_in_image(box, self.camera.intrinsic_np, (self.camera.width, self.camera.height), vis_level=box_vis_level):
                    box.render(ax, view=self.camera.intrinsic_np, normalize=True, colors=color)
        ax.set_xlim(0, self.camera.width)
        ax.set_ylim(self.camera.height, 0)
        ax.set_title(self.camera.channel)

class Log(Base):
    """
    Information about the log from which the data was extracted.
    """
    __tablename__ = 'log'
    token = Column(sql_types.HexLen8, primary_key=True)
    vehicle_name = Column(String(64))
    date = Column(String(64))
    timestamp = Column(Integer)
    logfile = Column(String(64))
    location = Column(String(64))
    map_version = Column(String(64))
    cameras = relationship('Camera', foreign_keys='Camera.log_token', back_populates='log')
    ego_poses = relationship('EgoPose', foreign_keys='EgoPose.log_token', back_populates='log')
    lidars = relationship('Lidar', foreign_keys='Lidar.log_token', back_populates='log')
    scenes = relationship('Scene', foreign_keys='Scene.log_token', back_populates='log')

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    @property
    def images(self) -> List[Image]:
        """
        Returns list of Images contained in the Log.
        :return: The list of Images contained in the log.
        """
        log_images = []
        for camera in self.cameras:
            log_images.extend(camera.images)
        return log_images

    @property
    def lidar_pcs(self) -> List[LidarPc]:
        """
        Returns list of Lidar PCs in the Log.
        :return: The list of Lidar PCs in the log.
        """
        log_lidar_pcs = []
        for lidar in self.lidars:
            log_lidar_pcs.extend(lidar.lidar_pcs)
        return log_lidar_pcs

    @property
    def lidar_boxes(self) -> List[LidarBox]:
        """
        Returns list of Lidar Boxes in the Log.
        :return: The list of Lidar Boxes in the log.
        """
        log_lidar_boxes = []
        for lidar_pc in self.lidar_pcs:
            log_lidar_boxes.extend(lidar_pc.lidar_boxes)
        return log_lidar_boxes

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

class Lidar(Base):
    """
    Defines a calibrated lidar used to record a particular log.
    """
    __tablename__ = 'lidar'
    token = Column(sql_types.HexLen8, primary_key=True)
    log_token = Column(sql_types.HexLen8, ForeignKey('log.token'), nullable=False)
    channel = Column(String(64))
    model = Column(String(64))
    translation = Column(sql_types.SqlTranslation)
    rotation = Column(sql_types.SqlRotation)
    lidar_pcs = relationship('LidarPc', foreign_keys='LidarPc.lidar_token', back_populates='lidar')

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def translation_np(self) -> npt.NDArray[np.float64]:
        """
        Get the translation in numpy format.
        :return: <np.float: 3> Translation.
        """
        return np.array(self.translation)

    @property
    def quaternion(self) -> Quaternion:
        """
        Get the rotation in quaternion.
        :return: The rotation in quaternion.
        """
        return Quaternion(self.rotation)

    @property
    def trans_matrix(self) -> npt.NDArray[np.float64]:
        """
        Get the transformation matrix.
        :return: <np.float: 4, 4>. Transformation matrix.
        """
        tm: npt.NDArray[np.float64] = self.quaternion.transformation_matrix
        tm[:3, 3] = self.translation_np
        return tm

    @property
    def trans_matrix_inv(self) -> npt.NDArray[np.float64]:
        """
        Get the inverse transformation matrix.
        :return: <np.float: 4, 4>. Inverse transformation matrix.
        """
        tm: npt.NDArray[np.float64] = np.eye(4)
        rot_inv = self.quaternion.rotation_matrix.T
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(np.transpose(-self.translation_np))
        return tm

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

class Track(Base):
    """
    Track from tracker output. A track represents a bunch of lidar boxes with the same instance id in a given log.
    """
    __tablename__ = 'track'
    token: str = Column(sql_types.HexLen8, primary_key=True)
    category_token: str = Column(sql_types.HexLen8, ForeignKey('category.token'), nullable=False)
    width: float = Column(Float)
    length: float = Column(Float)
    height: float = Column(Float)
    lidar_boxes: List[LidarBox] = relationship('LidarBox', foreign_keys=[LidarBox.track_token], back_populates='track')
    scenario_tags: List[ScenarioTag] = relationship('ScenarioTag', foreign_keys=[ScenarioTag.agent_track_token], back_populates='agent_track')
    category: Category = relationship('Category', foreign_keys=[category_token], back_populates='tracks')

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
    def nbr_lidar_boxes(self) -> int:
        """
        Returns number of boxes in the Track.
        :return: Number of boxes.
        """
        nbr: int = self._session.query(LidarBox).filter(LidarBox.track_token == self.token).count()
        return nbr

    @property
    def first_lidar_box(self) -> LidarBox:
        """
        Returns first lidar box along the track.
        :return: First lidar box along the track.
        """
        box: LidarBox = self._session.query(LidarBox).filter(LidarBox.track_token == self.token).join(LidarPc).order_by(LidarPc.timestamp.asc()).first()
        return box

    @property
    def last_lidar_box(self) -> LidarBox:
        """
        Returns last lidar box along the track.
        :return: Last lidar box along the track.
        """
        box: LidarBox = self._session.query(LidarBox).filter(LidarBox.track_token == self.token).join(LidarPc).order_by(LidarPc.timestamp.desc()).first()
        return box

    @property
    def duration(self) -> int:
        """
        Returns duration of Track.
        :return: Duration of the track.
        """
        d: int = self.last_lidar_box.timestamp - self.first_lidar_box.timestamp
        return d

    @property
    def distances_to_ego(self) -> npt.NDArray[np.float64]:
        """
        Returns array containing distances of all boxes in the Track from ego vehicle.
        :return: Distances of all boxes in the track from ego vehicle.
        """
        return np.asarray([lidar_box.distance_to_ego for lidar_box in self.lidar_boxes])

    @property
    def min_distance_to_ego(self) -> float:
        """
        Returns minimum distance of Track from Ego Vehicle.
        :return: The minimum distance of the track from ego vehicle.
        """
        min_dist: float = np.amin(self.distances_to_ego)
        return min_dist

    @property
    def max_distance_to_ego(self) -> float:
        """
        Returns maximum distance of Track from Ego Vehicle.
        :return: The maximum distance of the tack from ego vehicle.
        """
        max_dist: float = np.amax(self.distances_to_ego)
        return max_dist

