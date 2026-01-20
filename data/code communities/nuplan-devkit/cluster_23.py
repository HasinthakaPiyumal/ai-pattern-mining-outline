# Cluster 23

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

def get_boxes(sample_data: Union[LidarPc, Image], frame: Frame=Frame.GLOBAL, trans_matrix_ego: Optional[npt.NDArray[np.float64]]=None, trans_matrix_sensor: Optional[npt.NDArray[np.float64]]=None) -> List[Box3D]:
    """
    Given a LidarPc/Image record, this function returns a list of boxes (in the global coordinate frame by
    default) associated with that record. It simply converts the annotations to boxes.
    :param sample_data: Either a Lidar pointcloud or an Image.
        Note: Having the type Union[LidarPc, Image] for this throws error for TRT with Python 3.8.
    :param frame: An enumeration of Frame (global/vehicle/sensor).
    :param trans_matrix_ego:
        Transformation matrix to transform the boxes from the global frame to the ego-vehicle frame.
    :param trans_matrix_sensor:
        Transformation matrix to transform the boxes from the ego-vehicle frame to the sensor frame.
    :return: List of boxes in the global coordinate frame.
    """
    if frame == Frame.VEHICLE:
        assert trans_matrix_ego is not None
    if frame == Frame.SENSOR:
        assert trans_matrix_ego is not None
        assert trans_matrix_sensor is not None
    boxes = [sa.box() for sa in sample_data.lidar_boxes]
    if frame in [Frame.VEHICLE, Frame.SENSOR]:
        for box in boxes:
            box.transform(trans_matrix_ego)
    if frame == Frame.SENSOR:
        for box in boxes:
            box.transform(trans_matrix_sensor)
    return boxes

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

