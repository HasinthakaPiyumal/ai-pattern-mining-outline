# Cluster 18

def render_pc(sample_data: LidarPc, with_anns: bool=True, view_3d: npt.NDArray[np.float64]=np.eye(4), axes_limit: float=40, ax: Optional[Axes]=None, title: Optional[str]=None) -> None:
    """
    This is a naive rendering of the Lidar pointclouds with appropriate boxes. This is naive in the sense that it
    only renders the points but not the velocity associated with those points.
    :param sample_data: The Lidar pointcloud.
        Note: Having the type Union[LidarPc] for this throws error for TRT with Python 3.5.
    :param with_anns: Whether you want to render the annotations?
    :param view_3d: <np.float: 4, 4>. Define a projection needed (e.g. for drawing projection in an image).
    :param axes_limit: The range of that will be rendered will be between (-axes_limit, axes_limit).
    :param ax: Axes object or array of Axes objects.
    :param title: Title of the plot you want to render.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))
    points = view_points(sample_data.load().points[:3, :], view_3d, normalize=False)
    ax.scatter(points[0, :], points[1, :], c=points[2, :], s=2)
    if with_anns:
        for box in sample_data.boxes(Frame.SENSOR):
            ann_record = sample_data.lidar_box[box.payload]
            if not ann_record.track:
                logger.error('Wrong 3d instance mapping', ann_record)
                c: npt.NDArray[np.float64] = np.array([128, 0, 128]) / 255.0
            else:
                c = ann_record.track.category.color_np
            color = (c, c, np.array([0, 0, 0]))
            box.render(ax, view=view_3d, colors=color)
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_title('{}'.format(title))

def view_points(points: npt.NDArray[np.float64], view: npt.NDArray[np.float64], normalize: bool) -> npt.NDArray[np.float64]:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view
    nbr_points = points.shape[1]
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    return points

def boxes_lidar_to_img(lidar_record: LidarPc, img_record: Image, boxes_lidar: List[Box3D]) -> List[Box3D]:
    """
    This function transforms the boxes in the Lidar frame to the image frame.
    :param lidar_record: The SampleData record for the point cloud.
    :param img_record: The SampleData record for the image.
    :param boxes_lidar: List of boxes in the Lidar frame (given by lidar_record).
    :return: List of boxes in the image frame (given by img_record).
    """
    cam_intrinsic = img_record.camera.intrinsic_np
    imsize = (img_record.camera.width, img_record.camera.height)
    ego_from_lidar = lidar_record.lidar.trans_matrix
    global_from_ego = lidar_record.ego_pose.trans_matrix
    ego_from_global = img_record.ego_pose.trans_matrix_inv
    img_from_ego = img_record.camera.trans_matrix_inv
    trans_matrix = reduce(np.dot, [img_from_ego, ego_from_global, global_from_ego, ego_from_lidar])
    boxes_img = []
    for box in boxes_lidar:
        box = box.copy()
        box.transform(trans_matrix)
        if box_in_image(box, cam_intrinsic, imsize):
            boxes_img.append(box)
    return boxes_img

def box_in_image(box: Box3D, intrinsic: npt.NDArray[np.float64], imsize: Tuple[float, float], vis_level: int=BoxVisibility.ANY, front: int=2, min_front_th: float=0.1, with_velocity: bool=False) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: Box3D instance.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: Image (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :param front: Which axis represents depth. Default is z-axis (2) but can be set to y-axis (1) or x-axis (0).
    :param min_front_th: Corners' depth must be greater than this threshold for a box to be in the image.
        Note that 0.1 is a number that we found to produce reasonable plots.
    :param with_velocity: If True, include the velocity endpoint as one of the corners.
    :return True if visibility condition is satisfied.
    """
    corners_3d = box.corners()
    if with_velocity and (not np.isnan(box.velocity_endpoint).any()):
        corners_3d = np.concatenate((corners_3d, box.velocity_endpoint), axis=1)
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
    in_front = corners_3d[front, :] > min_front_th
    corners_img = corners_img[:, in_front]
    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError('vis_level: {} not valid'.format(vis_level))

def project_lidarpcs_to_camera(pc: LidarPointCloud, transform: npt.NDArray[np.float64], camera_intrinsic: npt.NDArray[np.float64], width: int, height: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool8]]:
    """
    Project lidar pcs to a camera and return pcs with coordinate in a camera view.
    :param pc: Lidar point clouds.
    :param transform: <4, 4>. Matrix to transform point clouds to a camera view.
    :param camera_intrinsic: <3, 3>. Intrinsic matrix of a camera.
    :param width: Image width.
    :param height: Image height.
    :return points: <np.float: 3, number of points>. Point cloud with their coordinates in a camera.
            masks: <np.bool: number of points>. A 1d-array of boolean to indicate which points are available in
            the camera.
    """
    pc.transform(transform)
    depths = pc.points[2, :]
    points = view_points(pc.points[:3, :], camera_intrinsic, normalize=True)
    mask = np.ones(pc.points.shape[1], dtype=bool)
    mask = np.logical_and(mask, depths > 0.0)
    mask = np.logical_and(mask, points[0, :] > 0)
    mask = np.logical_and(mask, points[0, :] < width - 1)
    mask = np.logical_and(mask, points[1, :] > 0)
    mask = np.logical_and(mask, points[1, :] < height - 1)
    points = points[:, mask]
    points[2, :] = depths[mask]
    return (points, mask)

def render_pointcloud_in_image(db: NuPlanDB, lidar_pc: LidarPc, dot_size: int=5, color_channel: int=2, max_radius: float=np.inf, image_channel: str='CAM_F0') -> None:
    """
    Scatter-plots pointcloud on top of image.
    :param db: Log Database.
    :param sample: LidarPc Sample.
    :param dot_size: Scatter plot dot size.
    :param color_channel: Set to 2 for coloring dots by height, 3 for intensity.
    :param max_radius: Max xy radius of lidar points to include in visualization.
        Set to np.inf to include all points.
    :param image_channel: Which image to render.
    """
    image = lidar_pc_closest_image(lidar_pc, [image_channel])[0]
    points, coloring, im = map_pointcloud_to_image(db, lidar_pc, image, color_channel=color_channel, max_radius=max_radius)
    plt.figure(figsize=(9, 16))
    plt.imshow(im)
    plt.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    plt.axis('off')

def lidar_pc_closest_image(lidar_pc: LidarPc, camera_channels: Optional[List[str]]=None) -> List[Image]:
    """
    Find the closest images to LidarPc.
    :param camera_channels: List of image channels to find closest image of.
    :return: List of Images from the provided channels closest to LidarPc.
    """
    if camera_channels is None:
        camera_channels = ['CAM_F0', 'CAM_B0', 'CAM_L0', 'CAM_L1', 'CAM_R0', 'CAM_R1']
    imgs = []
    for channel in camera_channels:
        img = lidar_pc._session.query(Image).join(Camera).filter(Image.camera_token == Camera.token).filter(Camera.channel == channel).filter(Camera.log_token == lidar_pc.lidar.log_token).order_by(func.abs(Image.timestamp - lidar_pc.timestamp)).first()
        imgs.append(img)
    return imgs

def map_pointcloud_to_image(db: NuPlanDB, lidar_pc: LidarPc, img: Image, color_channel: int=2, max_radius: float=np.inf) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], PIL.Image.Image]:
    """
    Given a lidar and camera sample_data, load point-cloud and map it to the image plane.
    :param db: Log Database.
    :param lidar_pc: Lidar sample_data record.
    :param img: Camera sample_data record.
    :param color_channel: Set to 2 for coloring dots by depth, 3 for intensity.
    :param max_radius: Max xy radius of lidar points to include in visualization.
        Set to np.inf to include all points.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    assert isinstance(lidar_pc, LidarPc), 'first input must be a lidar_pc modality'
    assert isinstance(img, Image), 'second input must be a camera modality'
    pc = lidar_pc.load()
    im = img.load_as(db, img_type='pil')
    radius = np.sqrt(pc.points[0] ** 2 + pc.points[1] ** 2)
    keep = radius <= max_radius
    pc.points = pc.points[:, keep]
    transform = reduce(np.dot, [img.camera.trans_matrix_inv, img.ego_pose.trans_matrix_inv, lidar_pc.ego_pose.trans_matrix, lidar_pc.lidar.trans_matrix])
    pc.transform(transform)
    coloring = pc.points[color_channel, :]
    depths = pc.points[2, :]
    points = view_points(pc.points[:3, :], img.camera.intrinsic_np, normalize=True)
    mask: npt.NDArray[np.bool8] = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    return (points, coloring, im)

def render_lidar_box(lidar_box: LidarBox, db: NuPlanDB, ax: Optional[List[Axes]]=None) -> None:
    """
    Render LidarBox on an image and a lidar.
    :param lidar_box: A LidarBox object
    :param db: Log Database.
    :param ax: Array of Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    pc = lidar_box.lidar_pc
    imgs = lidar_pc_closest_image(lidar_box.lidar_pc)
    found = False
    for img in imgs:
        cam = img.camera
        box = lidar_box.box()
        box.transform(img.ego_pose.trans_matrix_inv)
        box.transform(cam.trans_matrix_inv)
        if box_in_image(box, cam.intrinsic_np, (cam.width, cam.height), vis_level=BoxVisibility.ANY):
            found = True
            break
    assert found, 'Could not find image where annotation is visible'
    if not lidar_box.category:
        logger.error('Wrong 3d instance mapping', lidar_box)
        c: npt.NDArray[np.float64] = np.array([128, 0, 128]) / 255.0
    else:
        c = lidar_box.category.color_np
    color = (c, c, np.array([0, 0, 0]))
    ax[0].imshow(img.load_as(db, img_type='pil'))
    box.render(ax[0], view=img.camera.intrinsic_np, normalize=True, colors=color)
    ax[0].set_title(img.camera.channel)
    ax[0].axis('off')
    ax[0].set_aspect('equal')
    box = lidar_box.box()
    box.transform(pc.ego_pose.trans_matrix_inv)
    box.transform(pc.lidar.trans_matrix_inv)
    view = np.eye(4)
    pc.load(db).render_height(ax[1], view=view)
    box.render(ax[1], view=view, colors=color)
    corners = view_points(box.corners(), view, False)[:2, :]
    ax[1].set_xlim([np.amin(corners[0, :]) - 10, np.amax(corners[0, :]) + 10])
    ax[1].set_ylim([np.amin(corners[1, :]) - 10, np.amax(corners[1, :]) + 10])
    ax[1].axis('off')
    ax[1].set_aspect('equal')

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

class TestRendering(unittest.TestCase):
    """Some of these tests don't assert anything, but they will fail if the rendering code throws an exception."""

    def setUp(self) -> None:
        """Set up"""
        self.db = get_test_nuplan_db()
        self.lidar_box = get_test_nuplan_lidar_box()
        self.lidar_pc = get_test_nuplan_lidarpc_with_blob()

    def test_closest_image(self) -> None:
        """Tests the closest_image method"""
        result = lidar_pc_closest_image(self.lidar_pc)
        self.assertNotEqual(len(result), 0)

    def test_lidar_pc_render(self) -> None:
        """Test Lidar PC render."""
        self.lidar_pc.render(self.db)

    @patch('nuplan.database.nuplan_db_orm.rendering_utils.Axes.imshow', autospec=True)
    @patch('nuplan.database.nuplan_db_orm.image.Image.load_as', autospec=True)
    def test_lidar_box_render_img_found(self, loadas_mock: Mock, axes_mock: Mock) -> None:
        """Test Lidar Box render when the image is found"""
        render_lidar_box(self.lidar_box, self.db)
        loadas_mock.assert_called_once()
        axes_mock.assert_called_once()

    @patch('nuplan.database.nuplan_db_orm.rendering_utils.box_in_image', autospec=True)
    def test_lidar_box_render_img_not_found(self, box_in_image_mock: Mock) -> None:
        """Test Lidar Box render in the event that the image is not found"""
        box_in_image_mock.return_value = False
        with self.assertRaises(AssertionError):
            render_lidar_box(self.lidar_box, self.db)

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

class TestViewPoints(unittest.TestCase):
    """Test ViewPoints."""

    def test_view_points(self) -> None:
        """Test expected value of view_points()."""
        for _ in range(100):
            intrinsic = np.eye(3)
            focal = random.uniform(0.0, 10.0)
            intrinsic[0, 0] = focal
            intrinsic[1, 1] = focal
            pc1 = np.random.uniform(-100.0, 100.0, (3, 100))
            pc2: npt.NDArray[np.float64] = np.random.uniform(-100.0, 100.0) * pc1
            pc1_in_img = view_points(pc1, intrinsic, True)
            pc2_in_img = view_points(pc2, intrinsic, True)
            assert_array_almost_equal(pc1_in_img, pc2_in_img)
        for _ in range(100):
            intrinsic = np.eye(3)
            focal = random.uniform(0.0, 10.0)
            intrinsic[0, 0] = focal
            intrinsic[1, 1] = focal
            x_trans = random.uniform(-100.0, 100.0)
            y_trans = random.uniform(-100.0, 100.0)
            intrinsic[0, 2] = x_trans
            intrinsic[1, 2] = y_trans
            pc3 = np.random.uniform(-100.0, 100.0, (3, 100))
            pc4: npt.NDArray[np.float64] = np.random.uniform(-100.0, 100.0) * pc3
            pc3_in_img = view_points(pc3, intrinsic, True)
            pc4_in_img = view_points(pc4, intrinsic, True)
            assert_array_almost_equal(pc3_in_img, pc4_in_img)

class LidarPointCloud:
    """Simple data class representing a point cloud."""

    def __init__(self, points: npt.NDArray[np.float32]) -> None:
        """
        Class for manipulating and viewing point clouds.
        :param points: <np.float: f, n>. Input point cloud matrix with f features per point and n points.
        """
        if points.ndim == 1:
            points = np.atleast_2d(points).T
        self.points = points

    @staticmethod
    def load_pcd_bin(pcd_bin: Union[str, IO[Any], ByteString], pcd_bin_version: int=1) -> npt.NDArray[np.float32]:
        """
        Loads from pcd binary format:
            version 1: a numpy array with 5 cols (x, y, z, intensity, ring).
            version 2: a numpy array with 6 cols (x, y, z, intensity, ring, lidar_id).
        :param pcd_bin: File path or a file-like object or raw bytes.
        :param pcd_bin_version: 1 or 2, see above.
        :return: <np.float: 6, n>. Point cloud matrix[(x, y, z, intensity, ring, lidar_id)].
        """
        if isinstance(pcd_bin, str):
            scan = np.fromfile(pcd_bin, dtype=np.float32)
        else:
            if not isinstance(pcd_bin, bytes):
                pcd_bin = pcd_bin.read()
            scan = np.frombuffer(pcd_bin, dtype=np.float32)
            scan = np.copy(scan)
        if pcd_bin_version == 1:
            points = scan.reshape((-1, 5))
            points = np.hstack((points, -1 * np.ones((points.shape[0], 1), dtype=np.float32)))
        elif pcd_bin_version == 2:
            points = scan.reshape((-1, 6))
        else:
            pytest.fail('Unknown pcd bin file version: %d' % pcd_bin_version)
        return points.T

    @staticmethod
    def load_pcd(pcd_data: Union[IO[Any], ByteString]) -> npt.NDArray[np.float32]:
        """
        Loads a pcd file.
        :param pcd_data: File path or a file-like object or raw bytes.
        :return: <np.float: 6, n>. Point cloud matrix[(x, y, z, intensity, ring, lidar_id)].
        """
        if not isinstance(pcd_data, bytes):
            pcd_data = pcd_data.read()
        return PointCloud.parse(pcd_data).to_pcd_bin2()

    @classmethod
    def from_file(cls, file_name: str) -> LidarPointCloud:
        """
        Instantiates from a .pcl, .pcd, .npy, or .bin file.
        :param file_name: Path of the pointcloud file on disk.
        :return: A LidarPointCloud object.
        """
        if file_name.endswith('.bin'):
            points = cls.load_pcd_bin(file_name, 1)
        elif file_name.endswith('.bin2'):
            points = cls.load_pcd_bin(file_name, 2)
        elif file_name.endswith('.pcl') or file_name.endswith('.pcd'):
            points = pcd_to_numpy(file_name).T
        elif file_name.endswith('.npy'):
            points = np.load(file_name)
        else:
            raise ValueError('Unsupported filetype {}'.format(file_name))
        return cls(points)

    @classmethod
    def from_buffer(cls, pcd_data: Union[IO[Any], ByteString], content_type: str='bin') -> LidarPointCloud:
        """
        Instantiates from buffer.
        :param pcd_data: File path or a file-like object or raw bytes.
        :param content_type: Type of the point cloud content, such as 'bin', 'bin2', 'pcd'.
        :return: A LidarPointCloud object.
        """
        if content_type == 'bin':
            return cls(cls.load_pcd_bin(pcd_data, 1))
        elif content_type == 'bin2':
            return cls(cls.load_pcd_bin(pcd_data, 2))
        elif content_type == 'pcd':
            return cls(cls.load_pcd(pcd_data))
        else:
            raise NotImplementedError('Not implemented content type: %s' % content_type)

    @classmethod
    def make_random(cls) -> LidarPointCloud:
        """
        Instantiates a random point cloud.
        :return: LidarPointCloud instance.
        """
        return LidarPointCloud(points=np.random.normal(0, 100, size=(4, 100)))

    def __eq__(self, other: object) -> bool:
        """
        Checks if two LidarPointCloud are equal.
        :param other: Other object.
        :return: True if both objects are equal otherwise False.
        """
        if not isinstance(other, LidarPointCloud):
            return NotImplemented
        return np.allclose(self.points, other.points, atol=1e-06)

    def copy(self) -> LidarPointCloud:
        """
        Creates a copy of self.
        :return: LidarPointCloud instance.
        """
        return LidarPointCloud(points=self.points.copy())

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return int(self.points.shape[1])

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        assert 0 < ratio < 1
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, min_dist: float) -> None:
        """
        Removes points too close within a certain distance from origin from bird view (so dist = sqrt(x^2+y^2)).
        :param min_dist: The distance threshold.
        """
        dist_from_orig = np.linalg.norm(self.points[:2, :], axis=0)
        self.points = self.points[:, dist_from_orig >= min_dist]

    def radius_filter(self, radius: float) -> None:
        """
        Removes points outside the given radius.
        :param radius: Radius in meters.
        """
        keep = np.sqrt(self.points[0] ** 2 + self.points[1] ** 2) <= radius
        self.points = self.points[:, keep]

    def range_filter(self, xrange: Tuple[float, float]=(-np.inf, np.inf), yrange: Tuple[float, float]=(-np.inf, np.inf), zrange: Tuple[float, float]=(-np.inf, np.inf)) -> None:
        """
        Restricts points to specified ranges.
        :param xrange: (xmin, xmax).
        :param yrange: (ymin, ymax).
        :param zrange: (zmin, zmax).
        """
        keep_x = np.logical_and(xrange[0] <= self.points[0], self.points[0] <= xrange[1])
        keep_y = np.logical_and(yrange[0] <= self.points[1], self.points[1] <= yrange[1])
        keep_z = np.logical_and(zrange[0] <= self.points[2], self.points[2] <= zrange[1])
        keep = np.logical_and(keep_x, np.logical_and(keep_y, keep_z))
        self.points = self.points[:, keep]

    def translate(self, x: npt.NDArray[np.float64]) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3,>. Translation in x, y, z.
        """
        self.points[:3] += x.reshape((-1, 1))

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Applies a rotation.
        :param quaternion: Rotation to apply.
        """
        self.points[:3] = np.dot(quaternion.rotation_matrix.astype(np.float32), self.points[:3])

    def transform(self, transf_matrix: npt.NDArray[np.float64]) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogeneous transformation matrix.
        """
        transf_matrix = transf_matrix.astype(np.float32)
        self.points[:3, :] = transf_matrix[:3, :3] @ self.points[:3] + transf_matrix[:3, 3].reshape((-1, 1))

    def scale(self, scale: Tuple[float, float, float]) -> None:
        """
        Scales the lidar xyz coordinates.
        :param scale: The scaling parameter.
        """
        scale_arr = np.array(scale)
        scale_arr.shape = (3, 1)
        self.points[:3, :] *= np.tile(scale_arr, (1, self.nbr_points()))

    def render_image(self, canvas_size: Tuple[int, int]=(1001, 1001), view: npt.NDArray[np.float64]=np.array([[10, 0, 0, 500], [0, 10, 0, 500], [0, 0, 10, 0]]), color_dim: int=2) -> Image.Image:
        """
        Renders pointcloud to an array with 3 channels appropriate for viewing as an image. The image is color coded
        according the color_dim dimension of points (typically the height).
        :param canvas_size: (width, height). Size of the canvas on which to render the image.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param color_dim: The dimension of the points to be visualized as color. Default is 2 for height.
        :return: A Image instance.
        """
        heights = self.points[2, :]
        points = view_points(self.points[:3, :], view, normalize=False)
        points[2, :] = heights
        mask = np.ones(points.shape[1], dtype=bool)
        mask = np.logical_and(mask, points[0, :] < canvas_size[0] - 1)
        mask = np.logical_and(mask, points[0, :] > 0)
        mask = np.logical_and(mask, points[1, :] < canvas_size[1] - 1)
        mask = np.logical_and(mask, points[1, :] > 0)
        points = points[:, mask]
        color_values = points[color_dim, :]
        color_values = 255.0 * (color_values - np.amin(color_values)) / (np.amax(color_values) - np.amin(color_values))
        points = np.int16(np.round(points[:2, :]))
        color_values = np.int16(np.round(color_values))
        cmap = [cm.jet(i / 255, bytes=True)[:3] for i in range(256)]
        render = np.tile(np.expand_dims(np.zeros(canvas_size, dtype=np.uint8), axis=2), [1, 1, 3])
        color_value_array: npt.NDArray[np.float64] = -1 * np.ones(canvas_size, dtype=float)
        for (col, row), color_value in zip(points.T, color_values.T):
            if color_value > color_value_array[row, col]:
                color_value_array[row, col] = color_value
                render[row, col] = cmap[color_value]
        return Image.fromarray(render)

    def render_height(self, ax: axes.Axes, view: npt.NDArray[np.float64]=np.eye(4), x_lim: Tuple[float, float]=(-20, 20), y_lim: Tuple[float, float]=(-20, 20), marker_size: float=1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(self.points[2, :], ax, view, x_lim, y_lim, marker_size)

    def render_intensity(self, ax: axes.Axes, view: npt.NDArray[np.float64]=np.eye(4), x_lim: Tuple[float, float]=(-20, 20), y_lim: Tuple[float, float]=(-20, 20), marker_size: float=1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(self.points[3, :], ax, view, x_lim, y_lim, marker_size)

    def render_label(self, ax: axes.Axes, id2color: Optional[Dict[int, Tuple[float, float, float, float]]]=None, view: npt.NDArray[np.float64]=np.eye(4), x_lim: Tuple[float, float]=(-20, 20), y_lim: Tuple[float, float]=(-20, 20), marker_size: float=1.0) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points. Each points is colored based
        on labels through the label color mapping, If no mapping provided, we use the rainbow function to assign
        the colors.
        :param id2color: {label_id : (R, G, B, A)}. Id to color mapping where RGBA is within [0, 255].
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        label = self.points[-1]
        colors: Dict[int, Tuple[Any, ...]] = {}
        if id2color is None:
            unique_label = np.unique(label)
            color_rainbow = rainbow(len(unique_label), normalized=True)
            for label_id, c in zip(unique_label, color_rainbow):
                colors[label_id] = c
        else:
            for key, color in id2color.items():
                colors[key] = np.array(color) / 255.0
        color_list = list(map(lambda x: colors.get(x, np.array((1.0, 1.0, 1.0, 0.0))), label))
        self._render_helper(color_list, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(self, colors: Union[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]], ax: axes.Axes, view: npt.NDArray[np.float64], x_lim: Tuple[float, float], y_lim: Tuple[float, float], marker_size: float) -> None:
        """
        Helper function for rendering.
        :param colors: Array-like or list of colors or color input for scatter function.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=colors, s=marker_size)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

