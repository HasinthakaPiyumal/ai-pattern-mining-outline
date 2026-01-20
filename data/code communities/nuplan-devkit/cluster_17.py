# Cluster 17

def get_future_ego_trajectory(lidarpc_rec: LidarPc, future_ego_poses: List[EgoPose], transformmatrix: npt.NDArray[np.float64], future_horizon_len_s: float, future_interval_s: float=0.5, extrapolation_threshold_ms: float=100000.0) -> npt.NDArray[np.float64]:
    """
    Extract ego trajectory data starting from current sample timestamp for a duration of
        future horizon length in seconds.
    :param lidarpc_rec: Lidar point cloud record.
    :param future_ego_poses: future ego poses for a duration of horizon length.
    :param transformmatrix: Transformation matrix to transform the boxes from the global frame to the map_crop frame.
    :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
    :param future_interval_s: Timestamp interval of the future waypoints in seconds.
    :param extrapolation_threshold_ms: If the ego interpolation timestamp extends beyond the timestamp of the
        last recorded pose for the ego, then the values for the box position at the target timestamp will only
        be extrapolated if the target timestamp is within the specified number of microseconds of the last recorded
        pose. Otherwise the pose at the target timestamp will be set to None.
    :return: 2d numpy array of extracted trajectory data. Columns are
        (x_map, y_map, z_map, timestamp)
    """
    num_future_poses = int(future_horizon_len_s / future_interval_s)
    num_target_timestamps = num_future_poses + 1
    start_timestamp = lidarpc_rec.ego_pose.timestamp
    ego_traj: List[Tuple[float, ...]] = [(lidarpc_rec.ego_pose.x, lidarpc_rec.ego_pose.y, lidarpc_rec.ego_pose.z)]
    timestamps = [start_timestamp]
    ego_traj.extend([(pose.x, pose.y, pose.z) for pose in future_ego_poses])
    timestamps.extend([pose.timestamp for pose in future_ego_poses])
    target_timestamps: Union[npt.NDArray[np.float64], List[float]] = np.linspace(start=start_timestamp, stop=start_timestamp + future_horizon_len_s * 1000000.0, num=num_target_timestamps)
    last_ego_timestamp = timestamps[-1]
    target_timestamps = [t for t in target_timestamps if t <= last_ego_timestamp + extrapolation_threshold_ms]
    interpolated_ego_traj = interpolate_coordinates(target_timestamps=target_timestamps, box_timestamps=np.array([float(ts) for ts in timestamps]), box_coordinates=ego_traj)
    ego_traj_np = np.zeros((len(interpolated_ego_traj), 4))
    for i, wp in enumerate(interpolated_ego_traj):
        ego_traj_np[i, :] = [wp[0], wp[1], wp[2], target_timestamps[i]]
    num_waypoint = ego_traj_np.shape[0]
    if num_waypoint < num_target_timestamps:
        num_missing_rows = num_target_timestamps - num_waypoint
        padded_row = np.array([np.nan, np.nan, np.nan, np.nan])
        padding = np.tile(padded_row, (num_missing_rows, 1))
        ego_traj_np = np.concatenate((ego_traj_np, padding), axis=0)
    ego_poses = transform_ego_traj(ego_traj_np, lidarpc_rec.ego_pose.trans_matrix_inv)
    transf_matrix = transformmatrix.astype(np.float32)
    ego_poses = transformmatrix[:3, :3] @ ego_traj_np[:, 0:3].T + transf_matrix[:3, 3].reshape((-1, 1))
    ego_traj_np[:, 0:3] = ego_poses.T
    return ego_traj_np

def interpolate_coordinates(target_timestamps: Union[npt.NDArray[np.float64], List[float]], box_timestamps: Union[npt.NDArray[np.float64], List[float]], box_coordinates: List[Union[npt.NDArray[np.float64], Tuple[float, ...]]]) -> List[npt.NDArray[np.float64]]:
    """
    Given a sequence of boxes representing box positions over time along with their corresponding timestamps,
    interpolate the box coordinates at the target timestamps. Target timestamps should lie within
    the range of the raw timestamps corresponding to recorded data.
    :param target_timestamps: Times at which box coordinates will be interpolated, sorted in increasing order.
    :param box_timestamps: Times corresponding to each box coordinate in the box sequence, sorted in increasing order.
    :param box_coordinates: Sequence of box coordinates at each timestamp corresponding to an actor's position over
        time, from the raw data. The first box corresponds to the current frame.
    :return: Sequence of array coordinate positions in np.array(x, y, z) format at each timestamp corresponding to an
        actor's position over time, where position is interpolated at the target timestamps.
    """
    xs = list(np.interp(x=target_timestamps, xp=box_timestamps, fp=np.array([coordinate[0] for coordinate in box_coordinates])))
    ys = list(np.interp(x=target_timestamps, xp=box_timestamps, fp=np.array([coordinate[1] for coordinate in box_coordinates])))
    zs = list(np.interp(x=target_timestamps, xp=box_timestamps, fp=np.array([coordinate[2] for coordinate in box_coordinates])))
    centers = [np.array([x, y, z]) for x, y, z in zip(xs, ys, zs)]
    return centers

def transform_ego_traj(ego_poses: npt.NDArray[np.float64], transform_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Transform the ego trajectory to the first ego pose.
    :param ego_poses: Ego trajectory to transform.
    :param transform_matrix: Transformation to apply.
    :return The transformed ego poses.
    """
    ego_poses_new = transform_matrix[:3, :3] @ ego_poses[:, 0:3].T + transform_matrix[:3, 3].reshape((-1, 1))
    ego_poses[:, 0:3] = ego_poses_new.T
    return ego_poses

def transform(inp: npt.NDArray[np.float64], trans_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Transform a vector.
    :param inp: Vector to transform.
    :param trans_matrix: Transformation matrix.
    :return: Transformed vector.
    """
    inp = rotate(inp, Quaternion(matrix=trans_matrix[:3, :3]))
    inp = translate(inp, trans_matrix[:3, 3])
    return inp

def rotate(inp: npt.NDArray[np.float64], quaternion: Quaternion) -> npt.NDArray[np.float64]:
    """
    Rotate a vector.
    :param inp: Vector to rotate.
    :param quaternion: Rotation.
    :return: Rotated vector.
    """
    rotation_matrix: npt.NDArray[np.float64] = quaternion.rotation_matrix
    return np.dot(rotation_matrix, inp)

def translate(inp: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Translate a vector.
    :param inp: Vector to translate.
    :param x: Translation.
    :return: Translated vector.
    """
    return inp + x

def draw_future_ego_poses(ego_box: Box3D, ego_poses_np: npt.NDArray, color: Tuple[float, float, float], ax: Union[npt.NDArray[np.float64], Axes]) -> None:
    """
    Draw Future Ego Poses
    :param ego_box: Ego Vehicle Box.
    :param ego_poses_np: Numpy array containing future Ego Poses.
    :param color: Color to use.
    :param ax: Canvas to draw.
    """
    prev_x, prev_y = (ego_box.center[0], ego_box.center[1])
    for idx in range(1, ego_poses_np.shape[0]):
        next_x, next_y = (ego_poses_np[idx, 0], ego_poses_np[idx, 1])
        alpha = max(1.0 - idx * 0.1, 0.1)
        draw_line(from_x=prev_x, to_x=next_x, from_y=prev_y, to_y=next_y, color=color, marker='o', linewidth=1.0, canvas=ax, alpha=alpha)
        prev_x, prev_y = (next_x, next_y)

def draw_line(canvas: Union[npt.NDArray[np.float64], Axes], from_x: int, to_x: int, from_y: int, to_y: int, color: Tuple[Union[int, float], Union[int, float], Union[int, float]], linewidth: float, marker: Optional[str]=None, alpha: float=1.0) -> None:
    """
    Draw a line on a matplotlib/cv2 canvas. Note that marker is not used in cv2.
    :param canvas: Canvas to draw.
    :param from_x: Start x position.
    :param to_x: End x position.
    :param from_y: Start y position.
    :param to_y: End y position.
    :param color: Color of the line.
    :param linewidth: Width of the line.
    :param marker: Marker to use, defaults to None
    :param alpha: Alpha channel of the color, defaults to 1.0
    """
    if isinstance(canvas, np.ndarray):
        color_int = tuple((int(c * 255) for c in color))
        cv2.line(canvas, (int(from_x), int(from_y)), (int(to_x), int(to_y)), color_int[::-1], linewidth)
    else:
        canvas.plot([from_x, to_x], [from_y, to_y], color=color, linewidth=linewidth, marker=marker, alpha=alpha)

def render_on_map(lidarpc_rec: LidarPc, db: NuPlanDB, boxes_lidar: List[Box3D], ego_poses: List[EgoPose], points_to_render: Optional[npt.NDArray[np.float64]]=None, radius: float=80.0, ax: Axes=None, labelmap: Optional[Dict[int, Label]]=None, render_boxes_with_velocity: bool=False, render_map_raster: bool=False, render_vector_map: bool=False, track_token: Optional[str]=None, with_random_color: bool=False, render_future_ego_poses: bool=False) -> plt.axes:
    """
    This function is used to render a LidarPC and boxes (in the lidar frame) on the map.
    :param lidarpc_rec: LidarPc record from NuPlanDB.
    :param db: Log database.
    :param boxes_lidar: List of boxes in the lidar frame.
    :param ego_poses: Ego poses to render.
    :param points_to_render: <np.float: nbr_indices, nbr_points>. If the user wants to visualize only a specific set
        of points (example points from selective rings/drivable area filtered/...) and not the entire pointcloud, they
        can pass those points along. Note that nbr_indices >=2 i.e. the user should at least pass (x, y).
    :param radius: The radius (centered on the Lidar) outside which we won't keep any points or boxes.
    :param ax: Axis on which to render.
    :param labelmap: The labelmap is used to color the boxes. If not provided, default colors from box.render() will be
        used.
    :param render_boxes_with_velocity: Whether you want to show the velocity arrow when you render the box?
    :param render_map_raster: Boolean indicating whether to include visualization of map layers from rasterized map.
    :param render_vector_map: Boolean indicating whether to include visualization of baseline paths from vector map.
    :param track_token: Which track to render, if it's None, render all the tracks.
    :param with_random_color: Whether to render the instances with different random color.
    :param render_future_ego_poses: Whether to render future EgoPoses.
    :return: plt.axes corresponding to BEV image with specified visualizations.
    """
    xrange = (-radius, radius)
    yrange = (-radius, radius)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))
    intensity_map_crop, intensity_map_translation, intensity_map_scale = lidarpc_rec.ego_pose.get_map_crop(db.maps_db, xrange, yrange, 'intensity', rotate_face_up=True)
    map_translation = intensity_map_translation
    map_scale = intensity_map_scale
    lidar_to_ego = lidarpc_rec.lidar.trans_matrix
    ego_to_global = lidarpc_rec.ego_pose.trans_matrix
    map_align_rot = R.from_matrix(lidarpc_rec.ego_pose.quaternion.rotation_matrix.T)
    map_align_rot_angle = map_align_rot.as_euler('zxy')[0] + math.pi / 2
    map_align_transform = Quaternion(axis=[0, 0, 1], angle=map_align_rot_angle).transformation_matrix
    if render_map_raster:
        map_raster, map_translation, map_scale = lidarpc_rec.ego_pose.get_map_crop(maps_db=db.maps_db, xrange=xrange, yrange=yrange, map_layer_name='drivable_area', rotate_face_up=True)
        ax.imshow(map_raster[::-1, :], cmap='gray')
    elif intensity_map_crop is not None:
        ax.imshow(intensity_map_crop[::-1, :], cmap='gray')
    if intensity_map_crop is not None:
        ax.set_ylim(ax.get_ylim()[::-1])
    pointcloud = lidarpc_rec.load(db)
    if points_to_render is not None:
        pointcloud.points = points_to_render
    keep = np.sqrt(pointcloud.points[0] ** 2 + pointcloud.points[1] ** 2) < radius
    pointcloud.points = pointcloud.points[:, keep]
    global_to_crop = np.array([[map_scale[0], 0, 0, map_translation[0]], [0, map_scale[1], 0, map_translation[1]], [0, 0, map_scale[2], 0], [0, 0, 0, 1]])
    lidar_to_crop = reduce(np.dot, [global_to_crop, ego_to_global, lidar_to_ego, map_align_transform])
    front_length = 4.049
    rear_length = 1.127
    ego_car_length = front_length + rear_length
    ego_car_width = 1.1485 * 2.0
    ego_pose_np = np.array([ego_poses[0].x, ego_poses[0].y, ego_poses[0].z, 1])
    ego_box = Box3D(center=(ego_pose_np[0], ego_pose_np[1], ego_pose_np[2]), size=(ego_car_width, ego_car_length, 1.78), orientation=ego_poses[0].quaternion)
    ego_box.transform(ego_poses[0].trans_matrix_inv)
    ego_box.transform(map_align_transform)
    ego_box.transform(lidar_to_ego)
    ego_box.transform(ego_to_global)
    ego_box.scale(map_scale)
    ego_box.translate(map_translation)
    color = (1.0, 0.0, 0.0)
    colors: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], str]] = (color, color, 'k')
    ego_box.render(ax, colors=colors)
    if render_future_ego_poses:
        ego_poses_np = get_future_ego_trajectory(lidarpc_rec=lidarpc_rec, future_ego_poses=ego_poses, transformmatrix=lidar_to_crop, future_horizon_len_s=6.0, future_interval_s=0.5)
        draw_future_ego_poses(ego_box, ego_poses_np, color, ax)
    if render_vector_map:
        vector_map = lidarpc_rec.ego_pose.get_vector_map(maps_db=db.maps_db, xrange=xrange, yrange=yrange)
        lane_coords = vector_map.coords
        for coords in lane_coords:
            start = np.array([coords[0][0], coords[0][1], 0.0])
            end = np.array([coords[1][0], coords[1][1], 0.0])
            start = transform(start, map_align_transform)
            end = transform(end, map_align_transform)
            start = transform(start, lidar_to_ego)
            end = transform(end, lidar_to_ego)
            start = transform(start, ego_to_global)
            end = transform(end, ego_to_global)
            start = scale(start, map_scale)
            end = scale(end, map_scale)
            start = translate(start, map_translation)
            end = translate(end, map_translation)
            line = geometry.LineString([start[:-1], end[:-1]])
            xx, yy = line.coords.xy
            ax.plot(xx, yy, color='y', alpha=0.3)
    pointcloud.transform(lidar_to_crop)
    ax.scatter(pointcloud.points[0, :], pointcloud.points[1, :], c='g', s=1, alpha=0.2)
    if track_token is None and with_random_color:
        cmap = plt.cm.get_cmap('Dark2', len(boxes_lidar))
    for idx, box in enumerate(boxes_lidar):
        box_copy = box.copy()
        if track_token is not None:
            if box_copy.track_token != track_token:
                continue
        if np.abs(box_copy.center[0]) <= radius and np.abs(box_copy.center[1]) <= radius:
            colors, marker = get_colors_marker(labelmap, box_copy)
            if track_token is None and with_random_color:
                c = np.array(cmap(idx)[:3])
                colors = (c, c, 'k')
            box_copy.transform(map_align_transform)
            box_copy.transform(lidar_to_ego)
            box_copy.transform(ego_to_global)
            box_copy.scale(map_scale)
            box_copy.translate(map_translation)
            box_copy.render(ax, colors=colors, marker=marker, with_velocity=render_boxes_with_velocity)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    return ax

def scale(inp: npt.NDArray[np.float64], scale: Tuple[float, float, float]) -> npt.NDArray[np.float64]:
    """
    Scale a vector.
    :param inp: Vector to scale.
    :param scale: Scale factors.
    :return: Scaled vector.
    """
    scale_np = np.asarray(scale)
    assert len(scale_np) == 3
    return inp * scale_np

def get_colors_marker(labelmap: Optional[Dict[int, Label]], box: Box3D) -> Tuple[Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], str]], Optional[str]]:
    """
    Get the color and marker to use.
    :param labelmap: The labelmap is used to color the boxes. If not provided, default colors from box.render() will be
        used.
    :param box: The box for which color and marker are to be returned.
    :return: The color and marker to be used.
    """
    if labelmap is not None:
        c = np.array(labelmap[box.label].color)[:-1] / 255.0
        colors = (c, c, 'k')
    else:
        colors = None
    if box.label == 2:
        marker = None
    else:
        marker = 'o'
    return (colors, marker)

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

class TestGetFutureEgoTrajectory(unittest.TestCase):
    """Test getting future ego trajectory."""

    def setUp(self) -> None:
        """Set up test case."""
        self.lidar_pc = get_test_nuplan_lidarpc()
        self.future_lidarpc_recs: List[LidarPc] = [self.lidar_pc]
        while len(self.future_lidarpc_recs) < 200:
            self.future_lidarpc_recs.append(self.future_lidarpc_recs[-1].next)
        self.future_ego_poses = [rec.ego_pose for rec in self.future_lidarpc_recs]

    def test_get_future_ego_trajectory(self) -> None:
        """Test getting future ego trajectory."""
        future_ego_traj = get_future_ego_trajectory(self.lidar_pc, self.future_ego_poses, np.eye(4), 5.0, 0.5)
        self.assertEqual(future_ego_traj[0, 3], self.lidar_pc.ego_pose.timestamp)
        self.assertEqual(len(future_ego_traj), 11)
        self.assertLessEqual(abs((future_ego_traj[-1, 3] - future_ego_traj[0, 3]) / 1000000.0 - 5.0), 0.5)

    def test_get_future_ego_trajectory_not_enough(self) -> None:
        """Test getting future ego trajectory when there are not enough ego poses."""
        future_ego_traj = get_future_ego_trajectory(self.lidar_pc, self.future_ego_poses[:50], np.eye(4), 5.0, 0.5)
        self.assertEqual(future_ego_traj[0, 3], self.lidar_pc.ego_pose.timestamp)
        self.assertEqual(len(future_ego_traj), 11)
        np.testing.assert_equal(future_ego_traj[-1, :], [np.nan, np.nan, np.nan, np.nan])

class TestRenderOnMap(unittest.TestCase):
    """Test rendering on map."""

    def setUp(self) -> None:
        """Set up test case."""
        self.db = get_test_nuplan_db()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()
        self.future_lidarpc_recs: List[LidarPc] = [self.lidar_pc]
        while len(self.future_lidarpc_recs) < 200:
            self.future_lidarpc_recs.append(self.future_lidarpc_recs[-1].next)
        self.future_ego_poses = [rec.ego_pose for rec in self.future_lidarpc_recs]

    def test_render_on_map(self) -> None:
        """Test render on map."""
        render_on_map(self.lidar_pc, self.db, self.lidar_pc.boxes(), self.future_ego_poses, render_boxes_with_velocity=True, render_map_raster=False, render_vector_map=True, with_random_color=True, render_future_ego_poses=True)

class TestLabel(unittest.TestCase):
    """Test Label Serialization."""

    def test_serialize(self) -> None:
        """Tests a serialized label are still the same after serializing."""
        label = Label('my_name', (1, 3, 4, 1))
        self.assertEqual(label, Label.deserialize(json.loads(json.dumps(label.serialize()))))

class TestParseLabelmap(unittest.TestCase):
    """Test Parsing LabMap."""

    def setUp(self) -> None:
        """Setup function."""
        self.label1 = Label('label1', (1, 1, 1, 1))
        self.label2 = Label('label2', (2, 2, 2, 2))

    def test_empty(self) -> None:
        """Tests empty label map case."""
        id2name, id2color = parse_labelmap_dataclass({})
        self.assertIsInstance(id2name, OrderedDict)
        self.assertIsInstance(id2color, OrderedDict)
        self.assertEqual(len(id2name), 0)
        self.assertEqual(len(id2color), 0)

    def test_one(self) -> None:
        """Tests one label case."""
        num = 1
        mapping = {num: self.label1}
        id2name, id2color = parse_labelmap_dataclass(mapping)
        self.assertEqual(len(id2name), len(mapping))
        self.assertEqual(id2name[num], self.label1.name)
        self.assertEqual(len(id2color), len(mapping))
        self.assertEqual(id2color[num], self.label1.color)

    def test_multiple(self) -> None:
        """Tests multiple labels case."""
        num1, num2 = (1, 2)
        mapping = {num1: self.label1, num2: self.label2}
        id2name, id2color = parse_labelmap_dataclass(mapping)
        self.assertEqual(len(id2name), len(mapping))
        self.assertEqual(len(id2color), len(mapping))
        self.assertEqual(id2name[num1], self.label1.name)
        self.assertEqual(id2name[num2], self.label2.name)
        self.assertEqual(id2color[num1], self.label1.color)
        self.assertEqual(id2color[num2], self.label2.color)
        self.assertEqual(list(id2name.keys())[0], min(num1, num2))
        self.assertEqual(list(id2name.keys())[1], max(num1, num2))
        self.assertEqual(list(id2color.keys())[0], min(num1, num2))
        self.assertEqual(list(id2color.keys())[1], max(num1, num2))

class OrientedBox:
    """Represents the physical space occupied by agents on the plane."""

    def __init__(self, center: StateSE2, length: float, width: float, height: float):
        """
        :param center: The pose of the geometrical center of the box
        :param length: The length of the OrientedBox
        :param width: The width of the OrientedBox
        :param height: The height of the OrientedBox
        """
        self._center = center
        self._length = length
        self._width = width
        self._height = height

    @property
    def dimensions(self) -> Dimension:
        """
        :return: Dimensions of this oriented box in meters
        """
        return Dimension(length=self.length, width=self.width, height=self.height)

    @lru_cache()
    def corner(self, point: OrientedBoxPointType) -> Point2D:
        """
        Extract a point of oriented box
        :param point: which point you want to query
        :return: Coordinates of a point on oriented box.
        """
        if point == OrientedBoxPointType.FRONT_LEFT:
            return translate_longitudinally_and_laterally(self.center, self.half_length, self.half_width).point
        elif point == OrientedBoxPointType.FRONT_RIGHT:
            return translate_longitudinally_and_laterally(self.center, self.half_length, -self.half_width).point
        elif point == OrientedBoxPointType.REAR_LEFT:
            return translate_longitudinally_and_laterally(self.center, -self.half_length, self.half_width).point
        elif point == OrientedBoxPointType.REAR_RIGHT:
            return translate_longitudinally_and_laterally(self.center, -self.half_length, -self.half_width).point
        elif point == OrientedBoxPointType.CENTER:
            return self._center.point
        elif point == OrientedBoxPointType.FRONT_BUMPER:
            return translate_longitudinally_and_laterally(self.center, self.half_length, 0.0).point
        elif point == OrientedBoxPointType.REAR_BUMPER:
            return translate_longitudinally_and_laterally(self.center, -self.half_length, 0.0).point
        elif point == OrientedBoxPointType.LEFT:
            return translate_longitudinally_and_laterally(self.center, 0, self.half_width).point
        elif point == OrientedBoxPointType.RIGHT:
            return translate_longitudinally_and_laterally(self.center, 0, -self.half_width).point
        else:
            raise RuntimeError(f'Unknown point: {point}!')

    def all_corners(self) -> List[Point2D]:
        """
        Return 4 corners of oriented box (FL, RL, RR, FR)
        :return: all corners of a oriented box in a list
        """
        return [self.corner(OrientedBoxPointType.FRONT_LEFT), self.corner(OrientedBoxPointType.REAR_LEFT), self.corner(OrientedBoxPointType.REAR_RIGHT), self.corner(OrientedBoxPointType.FRONT_RIGHT)]

    @property
    def width(self) -> float:
        """
        Returns the width of the OrientedBox
        :return: The width of the OrientedBox
        """
        return self._width

    @property
    def half_width(self) -> float:
        """
        Returns the half width of the OrientedBox
        :return: The half width of the OrientedBox
        """
        return self._width / 2.0

    @property
    def length(self) -> float:
        """
        Returns the length of the OrientedBox
        :return: The length of the OrientedBox
        """
        return self._length

    @property
    def half_length(self) -> float:
        """
        Returns the half length of the OrientedBox
        :return: The half length of the OrientedBox
        """
        return self._length / 2.0

    @property
    def height(self) -> float:
        """
        Returns the height of the OrientedBox
        :return: The height of the OrientedBox
        """
        return self._height

    @property
    def half_height(self) -> float:
        """
        Returns the half height of the OrientedBox
        :return: The half height of the OrientedBox
        """
        return self._height / 2.0

    @property
    def center(self) -> StateSE2:
        """
        Returns the pose of the center of the OrientedBox
        :return: The pose of the center
        """
        return self._center

    @cached_property
    def geometry(self) -> Polygon:
        """
        Returns the Polygon describing the OrientedBox, if not done yet it will build it lazily.
        :return: The Polygon of the OrientedBox
        """
        corners = [tuple(corner) for corner in self.all_corners()]
        return Polygon(corners)

    def __hash__(self) -> int:
        """
        :return: hash for this object
        """
        return hash((self.center, self.width, self.height, self.length))

    def __eq__(self, other: object) -> bool:
        """
        Compare two oriented boxes
        :param other: object
        :return: true if other and self is equal
        """
        if not isinstance(other, OrientedBox):
            return NotImplemented
        return math.isclose(self.width, other.width) and math.isclose(self.height, other.height) and math.isclose(self.length, other.length) and (self.center == other.center)

    @classmethod
    def from_new_pose(cls, box: OrientedBox, pose: StateSE2) -> OrientedBox:
        """
        Initializer that create the same oriented box in a different pose.
        :param box: A sample box
        :param pose: The new pose
        :return: A new OrientedBox
        """
        return cls(pose, box.length, box.width, box.height)

def translate_longitudinally_and_laterally(pose: StateSE2, lon: float, lat: float) -> StateSE2:
    """
    Translate the position component of an SE2 pose longitudinally and laterally
    :param pose: SE2 pose to be translated
    :param lon: [m] distance by which a point should be translated in longitudinal direction
    :param lat: [m] distance by which a point should be translated in lateral direction
    :return Point2D translated position
    """
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.array([lat * np.cos(pose.heading + half_pi) + lon * np.cos(pose.heading), lat * np.sin(pose.heading + half_pi) + lon * np.sin(pose.heading)])
    return translate(pose, translation)

def get_front_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the front left corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, half_length, half_width).point

def get_front_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the front right corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, half_length, -half_width).point

def get_rear_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the rear left corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, -half_length, half_width).point

def get_rear_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the rear right corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, -half_length, -half_width).point

def rotate_angle(pose: StateSE2, theta: float) -> StateSE2:
    """
    Rotates the scene object by the given angle.
    :param pose: The input pose
    :param theta: The rotation angle.
    """
    cos_theta, sin_theta = (np.cos(theta), np.sin(theta))
    rotation_matrix: npt.NDArray[np.float64] = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return rotate(pose, rotation_matrix)

def transform(pose: StateSE2, transform_matrix: npt.NDArray[np.float64]) -> StateSE2:
    """
    Applies an SE2 transform
    :param pose: The input pose
    :param transform_matrix: The transform matrix, can be 2D (3x3) or 3D (4x4)
    """
    rotated_pose = rotate(pose, transform_matrix[:2, :2])
    return translate(rotated_pose, transform_matrix[:2, 2])

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

class IDMAgentManager:
    """IDM smart-agents manager."""

    def __init__(self, agents: UniqueIDMAgents, agent_occupancy: OccupancyMap, map_api: AbstractMap):
        """
        Constructor for IDMAgentManager.
        :param agents: A dictionary pairing the agent's token to it's IDM representation.
        :param agent_occupancy: An occupancy map describing the spatial relationship between agents.
        :param map_api: AbstractMap API
        """
        self.agents: UniqueIDMAgents = agents
        self.agent_occupancy = agent_occupancy
        self._map_api = map_api

    def propagate_agents(self, ego_state: EgoState, tspan: float, iteration: int, traffic_light_status: Dict[TrafficLightStatusType, List[str]], open_loop_detections: List[TrackedObject], radius: float) -> None:
        """
        Propagate each active agent forward in time.

        :param ego_state: the ego's current state in the simulation.
        :param tspan: the interval of time to simulate.
        :param iteration: the simulation iteration.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :param open_loop_detections: A list of open loop detections the IDM agents should be responsive to.
        :param radius: [m] The radius around the ego state
        """
        self.agent_occupancy.set('ego', ego_state.car_footprint.geometry)
        track_ids = []
        for track in open_loop_detections:
            track_ids.append(track.track_token)
            self.agent_occupancy.insert(track.track_token, track.box.geometry)
        self._filter_agents_out_of_range(ego_state, radius)
        for agent_token, agent in self.agents.items():
            if agent.is_active(iteration) and agent.has_valid_path():
                agent.plan_route(traffic_light_status)
                stop_lines = self._get_relevant_stop_lines(agent, traffic_light_status)
                inactive_stop_line_tokens = self._insert_stop_lines_into_occupancy_map(stop_lines)
                agent_path = path_to_linestring(agent.get_path_to_go())
                intersecting_agents = self.agent_occupancy.intersects(agent_path.buffer(agent.width / 2, cap_style=CAP_STYLE.flat))
                assert intersecting_agents.contains(agent_token), "Agent's baseline does not intersect the agent itself"
                if intersecting_agents.size > 1:
                    nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(agent_token)
                    agent_heading = agent.to_se2().heading
                    if 'ego' in nearest_id:
                        ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
                        longitudinal_velocity = np.hypot(ego_velocity.x, ego_velocity.y)
                        relative_heading = ego_state.rear_axle.heading - agent_heading
                    elif 'stop_line' in nearest_id:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0
                    elif nearest_id in self.agents:
                        nearest_agent = self.agents[nearest_id]
                        longitudinal_velocity = nearest_agent.velocity
                        relative_heading = nearest_agent.to_se2().heading - agent_heading
                    else:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0
                    relative_heading = principal_value(relative_heading)
                    projected_velocity = rotate_angle(StateSE2(longitudinal_velocity, 0, 0), relative_heading).x
                    length_rear = 0
                else:
                    projected_velocity = 0.0
                    relative_distance = agent.get_progress_to_go()
                    length_rear = agent.length / 2
                agent.propagate(IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=length_rear), tspan)
                self.agent_occupancy.set(agent_token, agent.projected_footprint)
                self.agent_occupancy.remove(inactive_stop_line_tokens)
        self.agent_occupancy.remove(track_ids)

    def get_active_agents(self, iteration: int, num_samples: int, sampling_time: float) -> DetectionsTracks:
        """
        Returns all agents as DetectionsTracks.
        :param iteration: the current simulation iteration.
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: agents as DetectionsTracks.
        """
        return DetectionsTracks(TrackedObjects([agent.get_agent_with_planned_trajectory(num_samples, sampling_time) for agent in self.agents.values() if agent.is_active(iteration)]))

    def _filter_agents_out_of_range(self, ego_state: EgoState, radius: float=100) -> None:
        """
        Filter out agents that are out of range.
        :param ego_state: The ego state used as the center of the given radius
        :param radius: [m] The radius around the ego state
        """
        if len(self.agents) == 0:
            return
        agents: npt.NDArray[np.int32] = np.array([agent.to_se2().point.array for agent in self.agents.values()])
        distances = cdist(np.expand_dims(ego_state.center.point.array, axis=0), agents)
        remove_indices = np.argwhere(distances.flatten() > radius)
        remove_tokens = np.array(list(self.agents.keys()))[remove_indices.flatten()]
        self.agent_occupancy.remove(remove_tokens)
        for token in remove_tokens:
            self.agents.pop(token)

    def _get_relevant_stop_lines(self, agent: IDMAgent, traffic_light_status: Dict[TrafficLightStatusType, List[str]]) -> List[StopLine]:
        """
        Retrieve the stop lines that are affecting the given agent.
        :param agent: The IDM agent of interest.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :return: A list of stop lines associated with the given traffic light status.
        """
        relevant_lane_connectors = list({segment.id for segment in agent.get_route()} & set(traffic_light_status[TrafficLightStatusType.RED]))
        lane_connectors = [self._map_api.get_map_object(lc_id, SemanticMapLayer.LANE_CONNECTOR) for lc_id in relevant_lane_connectors]
        return [stop_line for lc in lane_connectors if lc for stop_line in lc.stop_lines]

    def _insert_stop_lines_into_occupancy_map(self, stop_lines: List[StopLine]) -> List[str]:
        """
        Insert stop lines into the occupancy map.
        :param stop_lines: A list of stop lines to be inserted.
        :return: A list of token corresponding to the inserted stop lines.
        """
        stop_line_tokens: List[str] = []
        for stop_line in stop_lines:
            stop_line_token = f'stop_line_{stop_line.id}'
            if not self.agent_occupancy.contains(stop_line_token):
                self.agent_occupancy.set(stop_line_token, stop_line.polygon)
                stop_line_tokens.append(stop_line_token)
        return stop_line_tokens

def get_rectangle_corners(center: StateSE2, half_width: float, half_length: float) -> Polygon:
    """
    Get all four corners of actor's footprint
    :param center: StateSE2 object for the center of the actor
    :param half_width: rectangle width divided by 2
    :param half_length: rectangle length divided by 2.
    """
    corners = Polygon([get_front_left_corner(center, half_length, half_width), get_rear_left_corner(center, half_length, half_width), get_rear_right_corner(center, half_length, half_width), get_front_right_corner(center, half_length, half_width)])
    return corners

