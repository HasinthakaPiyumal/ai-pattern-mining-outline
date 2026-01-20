# Cluster 37

def _color_prep(ncolors: Optional[int]=None, alpha: int=128, colors: Optional[Union[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int, int]]]]=None) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Prepares colors for image_with_boxes and draw_masks.
    :param ncolors: Total number of colors.
    :param alpha: Alpha-matting value to use for fill (0-255).
    :param colors: {id: (R, G, B) OR (R, G, B, A)}.
    :return: {id: (R, G, B, A)}.
    """
    if colors is None:
        assert ncolors is not None, 'If no colors are supplied, need to include ncolors'
        colors = [tuple(color) + (alpha,) for color in rainbow(ncolors - 1)]
    else:
        if ncolors is not None:
            assert ncolors == len(colors), 'Number of supplied colors {} disagrees with supplied ncolor: {}'.format(len(colors), ncolors)
        for _id, color in colors.items():
            if isinstance(color, list):
                color = tuple(color)
            if len(color) == 3:
                color = color + (alpha,)
            colors[_id] = color
    return colors

def rainbow(nbr_colors: int, normalized: bool=False) -> List[Tuple[Any, ...]]:
    """
    Returns colors that are maximally different in HSV color space.
    :param nbr_colors: Number of colors to generate.
    :param normalized: Whether to normalize colors in 0-1. Else it is between 0-255.
    :return: <[(R <TYPE>, G <TYPE>, B <TYPE>)]>. Color <TYPE> varies depending on whether they are normalized.
    """
    hsv_tuples = [(x * 1.0 / nbr_colors, 0.5, 1) for x in range(nbr_colors)]
    colors = 255 * np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)))
    if normalized:
        colors = colors / 255.0
        return list(colors)
    else:
        return [tuple([int(c) for c in color]) for color in colors]

def image_with_boxes(img: npt.NDArray[np.uint8], boxes: Optional[List[Tuple[float, float, float, float]]]=None, labels: Optional[List[int]]=None, ncolors: Optional[int]=None, alpha: int=128, labelset: Optional[Dict[int, str]]=None, scores: Optional[List[float]]=None, colors: Optional[Union[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int, int]]]]=None) -> Image:
    """
    Simple plotting function to view image with boxes.
    :param img: <np.uint8: nrows, ncols, 3>. Input image.
    :param boxes: [(xmin, ymin, xmax, ymax)]. Bounding boxes.
    :param labels: Box3D labels.
    :param ncolors: Total number of colors needed (ie number of foreground classes).
    :param alpha: Alpha-matting value to use for fill (0-255).
    :param labelset: {id: name}. Maps label ids to names.
    :param scores: Prediction scores.
    :param colors: {id: (R, G, B) OR (R, G, B, A)}.
    :return: Image instance with overlaid boxes.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if not boxes or len(boxes) == 0:
        return img
    if not labels:
        labels = [1] * len(boxes)
    if not scores:
        scores = [None] * len(boxes)
    colors = _color_prep(ncolors, alpha, colors)
    draw = ImageDraw.Draw(img, 'RGBA')
    for box, label, score in zip(boxes, labels, scores):
        color = colors[label]
        bbox = [int(b) for b in box]
        draw.rectangle(bbox, outline=color[:3], fill=color)
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] - 1, bbox[3] - 1], outline=color[:3], fill=None)
        text = labelset[label] if labelset else '{:.0f}'.format(label)
        if score:
            text += ': {:.0f}'.format(100 * score)
        draw.text((box[0], box[1]), text)
    return img

def draw_masks(img: Image, target: npt.NDArray[np.uint8], ncolors: Optional[int]=None, colors: Optional[Union[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int, int]]]]=None, alpha: int=128) -> None:
    """
    Utility function for overlaying masks on images.
    :param img: Input image.
    :param target: <np.uint8: nrows, ncols>. Same size as image. Indicates the label of each pixel.
    :param ncolors: Total number of colors needed (ie number of foreground classes).
    :param colors: {id: (R, G, B) OR (R, G, B, A)}.
    :param alpha: Alpha-matting value to use for fill (0-255).
    """
    assert isinstance(img, Image.Image), 'img should be PIL type.'
    alpha_img = img.convert('RGBA')
    colors_prep = _color_prep(ncolors, alpha, colors)
    color_mask = build_color_mask(target, colors_prep)
    color_mask_image = Image.fromarray(color_mask, mode='RGBA')
    alpha_img.alpha_composite(color_mask_image)
    img.paste(alpha_img.convert('RGB'))

def build_color_mask(target: npt.NDArray[np.uint8], colors: Dict[int, Tuple[int, int, int, int]]) -> npt.NDArray[np.uint8]:
    """
    Builds color mask based on color dictionary.
    :param target: <np.uint8: nrows, ncols>. Same size as image. Indicates the label of each pixel.
    :param colors: {id: (R, G, B, A)}. Color dictionary.
    :return: Color mask.
    """
    nrows, ncols = target.shape
    color_mask = np.zeros(shape=(nrows, ncols, 4), dtype='uint8')
    for i in np.unique(target):
        color_mask[target == i] = colors[i]
    return color_mask

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

class TestRainbow(unittest.TestCase):
    """Test the rainbow."""

    def test_number_colors(self) -> None:
        """Check that correct number of colors is returned."""
        n_list = [3, 5, 7]
        for n in n_list:
            colors = rainbow(n)
            self.assertEqual(len(colors), n)

    def test_normalized(self) -> None:
        """Check that the colors are normalized."""
        n = 7
        colors = rainbow(n, normalized=True)
        for color in colors:
            for c in color:
                self.assertTrue(isinstance(c, float))
                self.assertTrue(0.0 <= c <= 1.0)

    def test_non_normalized(self) -> None:
        """Check that the colors are not normalized."""
        n = 7
        colors = rainbow(n, normalized=False)
        for color in colors:
            for c in color:
                self.assertTrue(isinstance(c, int))
                self.assertTrue(0 <= c <= 255)
        max_value = max([max(color) for color in colors])
        self.assertTrue(max_value > 1)

class TestBuildColorMask(unittest.TestCase):
    """Test build color mask function."""

    def test_build_color_mask(self) -> None:
        """Check if correct color mask is built."""
        colors = {0: (0, 0, 0, 0), 1: (128, 20, 20, 10), 2: (255, 100, 100, 255)}
        test_array = np.array([[0, 1], [2, 2]])
        target_mask = np.array([[[0, 0, 0, 0], [128, 20, 20, 10]], [[255, 100, 100, 255], [255, 100, 100, 255]]])
        color_mask = build_color_mask(test_array, colors)
        self.assertEqual(np.array_equal(color_mask, target_mask), True)

    def test_build_color_mask_invalid_key(self) -> None:
        """Check if build_color_mask throws a KeyError exception for invalid keys."""
        colors = {100: (0, 0, 0, 0), 1: (128, 20, 20, 10), 2: (255, 100, 100, 255)}
        test_array = np.array([[0, 1], [2, 2]])
        with self.assertRaises(KeyError):
            build_color_mask(test_array, colors)

class TestImageWithBoxes(unittest.TestCase):
    """Test drawing of input image with boxes and labels."""

    @mock.patch('nuplan.database.utils.plot._color_prep')
    def test_image_with_boxes(self, mock__color_prep) -> None:
        """Test function of viewing image with boxes."""
        target_image_array = np.zeros((100, 100, 3), np.uint8)
        target_image_array[10:41, 10:41] = (255, 0, 0)
        target_image_array[60:91, 60:91] = (0, 255, 0)
        target_image_array[10, 40] = (0, 0, 0)
        target_image_array[40, 10] = (0, 0, 0)
        target_image_array[60, 90] = (0, 0, 0)
        target_image_array[90, 60] = (0, 0, 0)
        target_image = Image.fromarray(target_image_array)
        draw = ImageDraw.Draw(target_image, 'RGB')
        test_image = np.zeros((100, 100, 3), np.uint8)
        labelset = {3: 'green', 2: 'red'}
        boxes = [(11.0, 11.0, 40.0, 40.0), (61.0, 61.0, 90.0, 90.0)]
        scores = [0.01, 0.01]
        labels = [2, 3]
        colors = {2: (255, 0, 0, 255), 3: (0, 255, 0, 255)}
        mock__color_prep.return_value = colors
        draw.text((boxes[0][0], boxes[0][1]), 'red: 1')
        draw.text((boxes[1][0], boxes[1][1]), 'green: 1')
        image = image_with_boxes(test_image, boxes, labels, 2, 255, labelset, scores, colors)
        image = np.array(image.convert('RGB'))
        target_image_converted = np.array(target_image.convert('RGB'))
        mock__color_prep.assert_called_with(2, 255, colors)
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(target_image_converted, np.ndarray)
        self.assertEqual(np.array_equal(target_image_converted, image), True)

class TestDrawMasks(unittest.TestCase):
    """Test draw_masks function."""

    @mock.patch('nuplan.database.utils.plot._color_prep')
    def test_draw_masks(self, mock__color_prep) -> None:
        """Test Drawing Masks on Image."""
        test_image = np.zeros((100, 100, 3), np.uint8)
        test_image[0:50, 0:50, :] = (255, 0, 0)
        test_image[0:50, 50:100, :] = (0, 255, 0)
        test_image[50:100, 0:50, :] = (0, 0, 255)
        test_image[50:, 50:, :] = (255, 255, 255)
        test_target = np.zeros((100, 100))
        test_target[:50, :50] = 1
        test_target[:50, 50:] = 2
        test_target[50:, :50] = 3
        test_target[50:, 50:] = 4
        colors = {1: (255, 0, 0, 255), 2: (0, 255, 0, 255), 3: (0, 0, 255, 255), 4: (255, 255, 255, 255)}
        mock__color_prep.return_value = colors
        input_image = Image.fromarray(np.zeros((100, 100, 3), np.uint8))
        draw_masks(input_image, test_target, ncolors=4, colors=colors, alpha=255)
        mock__color_prep.assert_called_with(4, 255, colors)
        input_image = np.array(input_image)
        self.assertEqual(np.array_equal(input_image, test_image), True)

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

