# Cluster 26

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

def load_point_cloud(lidar_pc: LidarPc, local_store: LocalStore, remote_store: S3Store) -> Optional[LidarPointCloud]:
    """
    Loads a point cloud given a database LidarPC object.
    :param lidar_pc: The lidar_pc for which to grab the point cloud.
    :param local_store: Local blob store for loading blobs from local file system.
    :param remote_store: S3 blob store for loading blobs from AWS S3.
    :return: The corresponding point cloud.
    """
    file_type = lidar_pc.filename.split('.')[-1]
    blob = download_and_cache(lidar_pc.filename, local_store, remote_store)
    return LidarPointCloud.from_buffer(blob, file_type) if blob is not None else None

def download_and_cache(key: str, local_store: LocalStore, remote_store: S3Store) -> Optional[BinaryIO]:
    """
    Downloads and cache the key given. This function assumes that the local and remotes stores are already configured.
    Data will be downloaded from the remote store's s3 bucket and saved relative to the data root of the local store.
    This method will initialize the scenario's blob store if it does not already exist.
    :param key: The key for which to grab the sensor data.
    :param local_store: Local blob store for loading blobs from local file system.
    :param remote_store: S3 blob store for loading blobs from AWS S3.
    :return: The sensor data.
    """
    if local_store.exists(key):
        return cast(BinaryIO, local_store.get(key))
    if remote_store is None:
        raise RuntimeError("Remote store is not set and key was not found locally. Try setting NUPLAN_DATA_STORE to 's3'.")
    try:
        blob = remote_store.get(key)
        local_store.put(key, blob)
        return cast(BinaryIO, local_store.get(key))
    except RuntimeError as error:
        logging.warning(f'Could not find sensor data locally or remotely. Returning None\nCause: {error}')
        return None

def load_image(image: ImageDBRow.Image, local_store: LocalStore, remote_store: S3Store) -> Optional[Image]:
    """
    Loads an image given a database Image object.
    :param image: The image for which to grab the image.
    :param local_store: Local blob store for loading blobs from local file system.
    :param remote_store: S3 blob store for loading blobs from AWS S3.
    :return: The corresponding image.
    """
    blob = download_and_cache(image.filename_jpg, local_store, remote_store)
    return Image.from_buffer(blob) if blob is not None else None

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

class TestNuPlanScenarioUtils(unittest.TestCase):
    """Test cases for nuplan_scenario_utils.py"""

    def test_convert_legacy_nuplan_path_to_latest(self) -> None:
        """Test that convert_legacy_nuplan_path_to_latest works as expected."""
        legacy_path = Path(NUPLAN_DATA_ROOT) / 'nuplan-v1.1/mini/2021.09.16.15.12.03_veh-42_01037_01434.db'
        legacy_path_str = str(legacy_path)
        expected_latest_path = Path(NUPLAN_DATA_ROOT) / 'nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db'
        expected_latest_path_str = str(expected_latest_path)
        actual_latest_path = convert_legacy_nuplan_path_to_latest(legacy_path_str)
        self.assertEqual(expected_latest_path_str, actual_latest_path)
        actual_latest_path = convert_legacy_nuplan_path_to_latest(legacy_path_str, NUPLAN_DATA_ROOT)
        self.assertEqual(expected_latest_path_str, actual_latest_path)
        data_root_without_slash = NUPLAN_DATA_ROOT.rstrip('/')
        actual_latest_path = convert_legacy_nuplan_path_to_latest(legacy_path_str, data_root_without_slash)
        self.assertEqual(expected_latest_path_str, actual_latest_path)

    def test_convert_legacy_nuplan_path_to_latest_invalid_path(self) -> None:
        """Test that convert_legacy_nuplan_path_to_latest will throw if path does not contain version info."""
        invalid_legacy_path = Path(NUPLAN_DATA_ROOT) / 'mini/2021.09.16.15.12.03_veh-42_01037_01434.db'
        invalid_legacy_path_str = str(invalid_legacy_path)
        with self.assertRaises(ValueError):
            _ = convert_legacy_nuplan_path_to_latest(invalid_legacy_path_str)

    def test_infer_remote_key_from_local_path(self) -> None:
        """Test that infer_remote_key_from_local_path works as expected."""
        local_path = Path(NUPLAN_DATA_ROOT) / 'nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db'
        local_path_str = str(local_path)
        expected_remote_key = 'splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db'
        actual_remote_key = infer_remote_key_from_local_path(local_path_str)
        self.assertEqual(expected_remote_key, actual_remote_key)
        actual_remote_key = infer_remote_key_from_local_path(local_path_str, NUPLAN_DATA_ROOT)
        self.assertEqual(expected_remote_key, actual_remote_key)
        data_root_without_slash = NUPLAN_DATA_ROOT.rstrip('/')
        actual_remote_key = infer_remote_key_from_local_path(local_path_str, data_root_without_slash)
        self.assertEqual(expected_remote_key, actual_remote_key)

    @patch(f'{TEST_PATH}.LidarPointCloud.from_buffer')
    @patch(f'{TEST_PATH}.download_and_cache')
    def test_load_point_cloud(self, mock_load_sensor: Mock, mock_from_buffer: Mock) -> None:
        """Test load_point_cloud."""
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.filename = 'pcd'
        mock_local_store = Mock(spec=LocalStore)
        mock_remote_store = Mock(spec=S3Store)
        mock_load_sensor.return_value = Mock()
        load_point_cloud(mock_lidar_pc, mock_local_store, mock_remote_store)
        mock_load_sensor.assert_called_with(mock_lidar_pc.filename, mock_local_store, mock_remote_store)
        mock_from_buffer.assert_called_with(mock_load_sensor.return_value, mock_lidar_pc.filename)

    @patch(f'{TEST_PATH}.Image.from_buffer')
    @patch(f'{TEST_PATH}.download_and_cache')
    def test_load_image(self, mock_load_sensor: Mock, mock_from_buffer: Mock) -> None:
        """Test load_point_cloud."""
        mock_image = Mock(spec=Image)
        mock_image.filename_jpg = 'image'
        mock_local_store = Mock(spec=LocalStore)
        mock_remote_store = Mock(spec=S3Store)
        mock_load_sensor.return_value = Mock()
        load_image(mock_image, mock_local_store, mock_remote_store)
        mock_load_sensor.assert_called_with(mock_image.filename_jpg, mock_local_store, mock_remote_store)
        mock_from_buffer.assert_called_with(mock_load_sensor.return_value)

    def test_download_and_cache(self) -> None:
        """Test download_and_cache."""
        mock_key = 'key'
        mock_image = Mock(spec=Image)
        mock_image.filename_jpg = 'image'
        mock_local_store = Mock(spec=LocalStore)
        mock_local_store.exists.side_effect = [True, False, False]
        mock_local_store.get.return_value = Mock(spec=BinaryIO)
        mock_local_store.put = Mock()
        mock_remote_store = Mock(spec=S3Store)
        mock_remote_store.get = Mock(return_value=Mock(spec=BinaryIO))
        blob = download_and_cache(mock_key, mock_local_store, mock_remote_store)
        self.assertEqual(mock_local_store.get.return_value, blob)
        self.assertTrue(isinstance(blob, BinaryIO))
        with self.assertRaises(RuntimeError):
            download_and_cache(mock_key, mock_local_store, None)
        blob = download_and_cache(mock_key, mock_local_store, mock_remote_store)
        mock_remote_store.get.assert_called_with(mock_key)
        mock_local_store.put.assert_called_with(mock_key, mock_remote_store.get.return_value)
        self.assertTrue(isinstance(blob, BinaryIO))

