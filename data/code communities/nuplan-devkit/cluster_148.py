# Cluster 148

@dataclass
class VectorMap(AbstractModelFeature):
    """
    Vector map data struture, including:
        coords: List[<np.ndarray: num_lane_segments, 2, 2>].
            The (x, y) coordinates of the start and end point of the lane segments.
        lane_groupings: List[List[<np.ndarray: num_lane_segments_in_lane>]].
            Each lane grouping or polyline is represented by an array of indices of lane segments
            in coords belonging to the given lane. Each batch contains a List of lane groupings.
        multi_scale_connections: List[Dict of {scale: connections_of_scale}].
            Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
            and each column in the array is [from_lane_segment_idx, to_lane_segment_idx].
        on_route_status: List[<np.ndarray: num_lane_segments, 2>].
            Binary encoding of on route status for lane segment at given index.
            Encoding: off route [0, 1], on route [1, 0], unknown [0, 0]
        traffic_light_data: List[<np.ndarray: num_lane_segments, 4>]
            One-hot encoding of on traffic light status for lane segment at given index.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]

    In all cases, the top level List represent number of batches. This is a special feature where
    each batch entry can have different size. Similarly, each lane grouping within a batch can have
    a variable number of elements. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """
    coords: List[FeatureDataType]
    lane_groupings: List[List[FeatureDataType]]
    multi_scale_connections: List[Dict[int, FeatureDataType]]
    on_route_status: List[FeatureDataType]
    traffic_light_data: List[FeatureDataType]
    _lane_coord_dim: int = 2
    _on_route_status_encoding_dim: int = LaneOnRouteStatusData.encoding_dim()

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        if len(self.coords) != len(self.multi_scale_connections):
            raise RuntimeError(f'Not consistent length of batches! {len(self.coords)} != {len(self.multi_scale_connections)}')
        if len(self.coords) != len(self.lane_groupings):
            raise RuntimeError(f'Not consistent length of batches! {len(self.coords)} != {len(self.lane_groupings)}')
        if len(self.coords) != len(self.on_route_status):
            raise RuntimeError(f'Not consistent length of batches! {len(self.coords)} != {len(self.on_route_status)}')
        if len(self.coords) != len(self.traffic_light_data):
            raise RuntimeError(f'Not consistent length of batches! {len(self.coords)} != {len(self.traffic_light_data)}')
        if len(self.coords) == 0:
            raise RuntimeError('Batch size has to be > 0!')
        for coords in self.coords:
            if coords.shape[1] != 2 or coords.shape[2] != 2:
                raise RuntimeError('The dimension of coords is not correct!')
        for coords, traffic_lights in zip(self.coords, self.traffic_light_data):
            if coords.shape[0] != traffic_lights.shape[0]:
                raise RuntimeError('Number of segments are inconsistent')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return len(self.coords) > 0 and len(self.coords[0]) > 0 and (len(self.lane_groupings) > 0) and (len(self.lane_groupings[0]) > 0) and (len(self.lane_groupings[0][0]) > 0) and (len(self.on_route_status) > 0) and (len(self.on_route_status[0]) > 0) and (len(self.traffic_light_data) > 0) and (len(self.traffic_light_data[0]) > 0) and (len(self.multi_scale_connections) > 0) and (len(list(self.multi_scale_connections[0].values())[0]) > 0)

    @property
    def num_of_batches(self) -> int:
        """
        :return: number of batches
        """
        return len(self.coords)

    def num_lanes_in_sample(self, sample_idx: int) -> int:
        """
        :param sample_idx: sample index in batch
        :return: number of lanes represented by lane_groupings in sample
        """
        return len(self.lane_groupings[sample_idx])

    @classmethod
    def lane_coord_dim(cls) -> int:
        """
        :return: dimension of coords, should be 2 (x, y)
        """
        return cls._lane_coord_dim

    @classmethod
    def on_route_status_encoding_dim(cls) -> int:
        """
        :return: dimension of route following status encoding
        """
        return cls._on_route_status_encoding_dim

    @classmethod
    def flatten_lane_coord_dim(cls) -> int:
        """
        :return: dimension of flattened start and end coords, should be 4 = 2 x (x, y)
        """
        return 2 * cls._lane_coord_dim

    def get_lane_coords(self, sample_idx: int) -> FeatureDataType:
        """
        Retrieve lane coordinates at given sample index.
        :param sample_idx: the batch index of interest.
        :return: lane coordinate features.
        """
        return self.coords[sample_idx]

    @classmethod
    def collate(cls, batch: List[VectorMap]) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(coords=[data for sample in batch for data in sample.coords], lane_groupings=[data for sample in batch for data in sample.lane_groupings], multi_scale_connections=[data for sample in batch for data in sample.multi_scale_connections], on_route_status=[data for sample in batch for data in sample.on_route_status], traffic_light_data=[data for sample in batch for data in sample.traffic_light_data])

    def to_feature_tensor(self) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(coords=[to_tensor(coords).contiguous() for coords in self.coords], lane_groupings=[[to_tensor(lane_grouping).contiguous() for lane_grouping in lane_groupings] for lane_groupings in self.lane_groupings], multi_scale_connections=[{scale: to_tensor(connection).contiguous() for scale, connection in multi_scale_connections.items()} for multi_scale_connections in self.multi_scale_connections], on_route_status=[to_tensor(status).contiguous() for status in self.on_route_status], traffic_light_data=[to_tensor(data).contiguous() for data in self.traffic_light_data])

    def to_device(self, device: torch.device) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(coords=[coords.to(device=device) for coords in self.coords], lane_groupings=[[lane_grouping.to(device=device) for lane_grouping in lane_groupings] for lane_groupings in self.lane_groupings], multi_scale_connections=[{scale: connection.to(device=device) for scale, connection in multi_scale_connections.items()} for multi_scale_connections in self.multi_scale_connections], on_route_status=[status.to(device=device) for status in self.on_route_status], traffic_light_data=[data.to(device=device) for data in self.traffic_light_data])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(coords=data['coords'], lane_groupings=data['lane_groupings'], multi_scale_connections=data['multi_scale_connections'], on_route_status=data['on_route_status'], traffic_light_data=data['traffic_light_data'])

    def unpack(self) -> List[VectorMap]:
        """Implemented. See interface."""
        return [VectorMap([coords], [lane_groupings], [multi_scale_connections], [on_route_status], [traffic_light_data]) for coords, lane_groupings, multi_scale_connections, on_route_status, traffic_light_data in zip(self.coords, self.lane_groupings, self.multi_scale_connections, self.on_route_status, self.traffic_light_data)]

    def rotate(self, quaternion: Quaternion) -> VectorMap:
        """
        Rotate the vector map.
        :param quaternion: Rotation to apply.
        """
        for coord in self.coords:
            validate_type(coord, np.ndarray)
        return VectorMap(coords=[rotate_coords(data, quaternion) for data in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def translate(self, translation_value: FeatureDataType) -> VectorMap:
        """
        Translate the vector map.
        :param translation_value: Translation in x, y, z.
        """
        assert translation_value.size == 3, 'Translation value must have dimension of 3 (x, y, z)'
        are_the_same_type(translation_value, self.coords[0])
        return VectorMap(coords=[translate_coords(coords, translation_value) for coords in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def scale(self, scale_value: FeatureDataType) -> VectorMap:
        """
        Scale the vector map.
        :param scale_value: <np.float: 3,>. Scale in x, y, z.
        """
        assert scale_value.size == 3, f'Scale value has incorrect dimension: {scale_value.size}!'
        are_the_same_type(scale_value, self.coords[0])
        return VectorMap(coords=[scale_coords(coords, scale_value) for coords in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def xflip(self) -> VectorMap:
        """
        Flip the vector map along the X-axis.
        """
        return VectorMap(coords=[xflip_coords(coords) for coords in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def yflip(self) -> VectorMap:
        """
        Flip the vector map along the Y-axis.
        """
        return VectorMap(coords=[yflip_coords(coords) for coords in self.coords], lane_groupings=self.lane_groupings, multi_scale_connections=self.multi_scale_connections, on_route_status=self.on_route_status, traffic_light_data=self.traffic_light_data)

    def extract_lane_polyline(self, sample_idx: int, lane_idx: int) -> FeatureDataType:
        """
        Extract start points (first coordinate) for segments in lane, specified by segment indices
            in lane_groupings.
        :param sample_idx: sample index in batch
        :param lane_idx: lane index in sample
        :return: lane_polyline: <np.ndarray: num_lane_segments_in_lane, 2>. Array of start points
            for each segment in lane.
        """
        lane_grouping = self.lane_groupings[sample_idx][lane_idx]
        return self.coords[sample_idx][lane_grouping, 0]

def are_the_same_type(lhs: Any, rhs: Any) -> None:
    """
    Validate that lhs and rhs are of the same type
    :param lhs: left argument
    :param rhs: right argument
    """
    lhs_type = type(lhs)
    rhs_type = type(rhs)
    assert lhs_type == rhs_type, f'Lhs and Rhs are not of the same type! {lhs_type} != {rhs_type}!'

def translate_coords(coords: FeatureDataType, translation_value: FeatureDataType, avails: Optional[FeatureDataType]=None) -> FeatureDataType:
    """
    Translate all vector coordinates within input tensor along x, y dimensions of input translation tensor.
        Note: Z-dimension ignored.
    :param coords: coordinates to translate: <num_map_elements, num_points_per_element, 2>.
    :param translation_value: <np.float: 3,>. Translation in x, y, z.
    :param avails: Optional mask to specify real vs zero-padded data to ignore in coords:
        <num_map_elements, num_points_per_element>.
    :return translated coords.
    :raise ValueError: If translation_value dimensions are not valid or coords and avails have inconsistent shape.
    """
    if translation_value.shape[0] != 3:
        raise ValueError(f'Translation value has incorrect dimensions: {translation_value.shape[0]}! Expected: 3 (x, y, z)')
    _validate_coords_shape(coords)
    are_the_same_type(coords, translation_value)
    if avails is not None and coords.shape[:2] != avails.shape:
        raise ValueError(f'Mismatching shape between coords and availabilities: {coords.shape[:2]}, {avails.shape}')
    coords = coords + translation_value[:2]
    if avails is not None:
        coords[~avails] = 0.0
    return coords

def scale_coords(coords: FeatureDataType, scale_value: FeatureDataType) -> FeatureDataType:
    """
    Scale all vector coordinates within input tensor along x, y dimensions of input scaling tensor.
        Note: Z-dimension ignored.
    :param coords: coordinates to scale: <num_map_elements, num_points_per_element, 2>.
    :param scale_value: <np.float: 3,>. Scale in x, y, z.
    :return scaled coords.
    :raise ValueError: If scale_value dimensions are not valid.
    """
    if scale_value.shape[0] != 3:
        raise ValueError(f'Scale value has incorrect dimensions: {scale_value.shape[0]}! Expected: 3 (x, y, z)')
    _validate_coords_shape(coords)
    are_the_same_type(coords, scale_value)
    return coords * scale_value[:2]

def xflip_coords(coords: FeatureDataType) -> FeatureDataType:
    """
    Flip all vector coordinates within input tensor along X-axis.
    :param coords: coordinates to flip: <num_map_elements, num_points_per_element, 2>.
    :return flipped coords.
    """
    _validate_coords_shape(coords)
    coords = deepcopy(coords)
    coords[:, :, 0] *= -1
    return coords

def yflip_coords(coords: FeatureDataType) -> FeatureDataType:
    """
    Flip all vector coordinates within input tensor along Y-axis.
    :param coords: coordinates to flip: <num_map_elements, num_points_per_element, 2>.
    :return flipped coords.
    """
    _validate_coords_shape(coords)
    coords = deepcopy(coords)
    coords[:, :, 1] *= -1
    return coords

def _validate_coords_shape(coords: FeatureDataType) -> None:
    """
    Validate coordinates have proper shape: <num_map_elements, num_points_per_element, 2>.
    :param coords: Coordinates to validate.
    :raise ValueError: If coordinates dimensions are not valid.
    """
    if len(coords.shape) != 3 or coords.shape[2] != 2:
        raise ValueError(f'Unexpected coords shape: {coords.shape}. Expected shape: (*, *, 2)')

@dataclass
class VectorSetMap(AbstractModelFeature):
    """
    Vector set map data structure, including:
        coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample in batch,
                indexed by map feature.
        traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample
                in batch, indexed by map feature. Same indexing as coords.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
        availabilities: Dict[str, List[<np.ndarray: num_elements, num_points>]].
            Boolean indicator of whether feature data (coords as well as traffic light status if it exists for feature)
                is available for point at given index or if it is zero-padded.

    Feature formulation as sets of vectors for each map element similar to that of VectorNet ("VectorNet: Encoding HD
    Maps and Agent Dynamics from Vectorized Representation"), except map elements are encoded as sets of singular x, y
    points instead of start, end point pairs.

    Coords, traffic light status, and availabilities data are each keyed by map feature name, with dimensionality
    (availabilities don't include feature dimension):
    B: number of samples per batch (variable)
    N: number of map elements (fixed for a given map feature)
    P: number of points (fixed for a given map feature)
    F: number of features (2 for coords, 4 for traffic light status)

    Data at the same index represent the same map element/point among coords, traffic_light_data, and availabilities,
    with traffic_light_data only optionally included. For each map feature, the top level List represents number of
    samples per batch. This is a special feature where each batch entry can have a different size. For that reason, the
    features can not be placed to a single tensor, and we batch the feature with a custom `collate` function.
    """
    coords: Dict[str, List[FeatureDataType]]
    traffic_light_data: Dict[str, List[FeatureDataType]]
    availabilities: Dict[str, List[FeatureDataType]]
    _polyline_coord_dim: int = 2
    _traffic_light_status_dim: int = LaneSegmentTrafficLightData.encoding_dim()

    def __post_init__(self) -> None:
        """
        Sanitize attributes of the dataclass.
        :raise RuntimeError if dimensions invalid.
        """
        if not len(self.coords) > 0:
            raise RuntimeError('Coords cannot be empty!')
        if not all([len(coords) > 0 for coords in self.coords.values()]):
            raise RuntimeError('Batch size has to be > 0!')
        self._sanitize_feature_consistency()
        self._sanitize_data_dimensionality()

    def _sanitize_feature_consistency(self) -> None:
        """
        Check data dimensionality consistent across and within map features.
        :raise RuntimeError if dimensions invalid.
        """
        if not all([len(coords) == len(list(self.coords.values())[0]) for coords in self.coords.values()]):
            raise RuntimeError('Batch size inconsistent across features!')
        for feature_name, feature_coords in self.coords.items():
            if feature_name not in self.availabilities:
                raise RuntimeError('No matching feature in coords for availabilities data!')
            feature_avails = self.availabilities[feature_name]
            if len(feature_avails) != len(feature_coords):
                raise RuntimeError(f'Batch size between coords and availabilities data inconsistent! {len(feature_coords)} != {len(feature_avails)}')
            feature_size = self.feature_size(feature_name)
            if feature_size[1] == 0:
                raise RuntimeError('Features cannot be empty!')
            for coords in feature_coords:
                if coords.shape[0:2] != feature_size:
                    raise RuntimeError(f"Coords for {feature_name} feature don't have consistent feature size! {coords.shape[0:2] != feature_size}")
            for avails in feature_avails:
                if avails.shape[0:2] != feature_size:
                    raise RuntimeError(f"Availabilities for {feature_name} feature don't have consistent feature size! {avails.shape[0:2] != feature_size}")
        for feature_name, feature_tl_data in self.traffic_light_data.items():
            if feature_name not in self.coords:
                raise RuntimeError('No matching feature in coords for traffic light data!')
            feature_coords = self.coords[feature_name]
            if len(feature_tl_data) != len(self.coords[feature_name]):
                raise RuntimeError(f'Batch size between coords and traffic light data inconsistent! {len(feature_coords)} != {len(feature_tl_data)}')
            feature_size = self.feature_size(feature_name)
            for tl_data in feature_tl_data:
                if tl_data.shape[0:2] != feature_size:
                    raise RuntimeError(f"Traffic light data for {feature_name} feature don't have consistent feature size! {tl_data.shape[0:2] != feature_size}")

    def _sanitize_data_dimensionality(self) -> None:
        """
        Check data dimensionality as expected.
        :raise RuntimeError if dimensions invalid.
        """
        for feature_coords in self.coords.values():
            for sample in feature_coords:
                if sample.shape[2] != self._polyline_coord_dim:
                    raise RuntimeError('The dimension of coords is not correct!')
        for feature_tl_data in self.traffic_light_data.values():
            for sample in feature_tl_data:
                if sample.shape[2] != self._traffic_light_status_dim:
                    raise RuntimeError('The dimension of traffic light data is not correct!')
        for feature_avails in self.availabilities.values():
            for sample in feature_avails:
                if len(sample.shape) != 2:
                    raise RuntimeError('The dimension of availabilities is not correct!')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return all([len(feature_coords) > 0 for feature_coords in self.coords.values()]) and all([feature_coords[0].shape[0] > 0 for feature_coords in self.coords.values()]) and all([feature_coords[0].shape[1] > 0 for feature_coords in self.coords.values()]) and all([len(feature_tl_data) > 0 for feature_tl_data in self.traffic_light_data.values()]) and all([feature_tl_data[0].shape[0] > 0 for feature_tl_data in self.traffic_light_data.values()]) and all([feature_tl_data[0].shape[1] > 0 for feature_tl_data in self.traffic_light_data.values()]) and all([len(features_avails) > 0 for features_avails in self.availabilities.values()]) and all([features_avails[0].shape[0] > 0 for features_avails in self.availabilities.values()]) and all([features_avails[0].shape[1] > 0 for features_avails in self.availabilities.values()])

    @property
    def batch_size(self) -> int:
        """
        Batch size across features.
        :return: number of batches.
        """
        return len(list(self.coords.values())[0])

    def feature_size(self, feature_name: str) -> Tuple[int, int]:
        """
        Number of map elements for given feature, points per element.
        :param feature_name: name of map feature to access.
        :return: [num_elements, num_points]
        :raise: RuntimeError if empty feature.
        """
        map_feature = self.coords[feature_name][0]
        if map_feature.size == 0:
            raise RuntimeError('Feature is empty!')
        return (map_feature.shape[0], map_feature.shape[1])

    @classmethod
    def coord_dim(cls) -> int:
        """
        Coords dimensionality, should be 2 (x, y).
        :return: dimension of coords.
        """
        return cls._polyline_coord_dim

    @classmethod
    def traffic_light_status_dim(cls) -> int:
        """
        Traffic light status dimensionality, should be 4.
        :return: dimension of traffic light status.
        """
        return cls._traffic_light_status_dim

    def get_lane_coords(self, sample_idx: int) -> FeatureDataType:
        """
        Retrieve lane coordinates at given sample index.
        :param sample_idx: the batch index of interest.
        :return: lane coordinate features.
        """
        lane_coords = self.coords[VectorFeatureLayer.LANE.name][sample_idx]
        if lane_coords.size == 0:
            raise RuntimeError('Lane feature is empty!')
        return lane_coords

    @classmethod
    def collate(cls, batch: List[VectorSetMap]) -> VectorSetMap:
        """Implemented. See interface."""
        coords: Dict[str, List[FeatureDataType]] = defaultdict(list)
        traffic_light_data: Dict[str, List[FeatureDataType]] = defaultdict(list)
        availabilities: Dict[str, List[FeatureDataType]] = defaultdict(list)
        for sample in batch:
            for feature_name, feature_coords in sample.coords.items():
                coords[feature_name] += feature_coords
            for feature_name, feature_tl_data in sample.traffic_light_data.items():
                traffic_light_data[feature_name] += feature_tl_data
            for feature_name, feature_avails in sample.availabilities.items():
                availabilities[feature_name] += feature_avails
        return VectorSetMap(coords=coords, traffic_light_data=traffic_light_data, availabilities=availabilities)

    def to_feature_tensor(self) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(coords={feature_name: [to_tensor(sample).contiguous() for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data={feature_name: [to_tensor(sample).contiguous() for sample in feature_tl_data] for feature_name, feature_tl_data in self.traffic_light_data.items()}, availabilities={feature_name: [to_tensor(sample).contiguous() for sample in feature_avails] for feature_name, feature_avails in self.availabilities.items()})

    def to_device(self, device: torch.device) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(coords={feature_name: [sample.to(device=device) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data={feature_name: [sample.to(device=device) for sample in feature_tl_data] for feature_name, feature_tl_data in self.traffic_light_data.items()}, availabilities={feature_name: [sample.to(device=device) for sample in feature_avails] for feature_name, feature_avails in self.availabilities.items()})

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(coords=data['coords'], traffic_light_data=data['traffic_light_data'], availabilities=data['availabilities'])

    def unpack(self) -> List[VectorSetMap]:
        """Implemented. See interface."""
        return [VectorSetMap({feature_name: [feature_coords[sample_idx]] for feature_name, feature_coords in self.coords.items()}, {feature_name: [feature_tl_data[sample_idx]] for feature_name, feature_tl_data in self.traffic_light_data.items()}, {feature_name: [feature_avails[sample_idx]] for feature_name, feature_avails in self.availabilities.items()}) for sample_idx in range(self.batch_size)]

    def rotate(self, quaternion: Quaternion) -> VectorSetMap:
        """
        Rotate the vector set map.
        :param quaternion: Rotation to apply.
        :return rotated VectorSetMap.
        """
        for feature_coords in self.coords.values():
            for sample in feature_coords:
                validate_type(sample, np.ndarray)
        return VectorSetMap(coords={feature_name: [rotate_coords(sample, quaternion) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

    def translate(self, translation_value: FeatureDataType) -> VectorSetMap:
        """
        Translate the vector set map.
        :param translation_value: Translation in x, y, z.
        :return translated VectorSetMap.
        :raise ValueError if translation_value dimensions invalid.
        """
        if translation_value.size != 3:
            raise ValueError(f'Translation value has incorrect dimensions: {translation_value.size}! Expected: 3 (x, y, z)')
        are_the_same_type(translation_value, list(self.coords.values())[0])
        return VectorSetMap(coords={feature_name: [translate_coords(sample_coords, translation_value, sample_avails) for sample_coords, sample_avails in zip(self.coords[feature_name], self.availabilities[feature_name])] for feature_name in self.coords}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

    def scale(self, scale_value: FeatureDataType) -> VectorSetMap:
        """
        Scale the vector set map.
        :param scale_value: <np.float: 3,>. Scale in x, y, z.
        :return scaled VectorSetMap.
        :raise ValueError if scale_value dimensions invalid.
        """
        if scale_value.size != 3:
            raise ValueError(f'Scale value has incorrect dimensions: {scale_value.size}! Expected: 3 (x, y, z)')
        are_the_same_type(scale_value, list(self.coords.values())[0])
        return VectorSetMap(coords={feature_name: [scale_coords(sample, scale_value) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

    def xflip(self) -> VectorSetMap:
        """
        Flip the vector set map along the X-axis.
        :return flipped VectorSetMap.
        """
        return VectorSetMap(coords={feature_name: [xflip_coords(sample) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

    def yflip(self) -> VectorSetMap:
        """
        Flip the vector set map along the Y-axis.
        :return flipped VectorSetMap.
        """
        return VectorSetMap(coords={feature_name: [yflip_coords(sample) for sample in feature_coords] for feature_name, feature_coords in self.coords.items()}, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

class TestVectorUtils(unittest.TestCase):
    """Test vector-based feature utility functions."""

    def setUp(self) -> None:
        """Set up test case."""
        self.coords: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [-1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [-1.0, -1.0], [1.0, -1.0]]])
        self.avails: npt.NDArray[np.bool_] = np.array([[False, True, True], [True, True, True]])

    def test_rotate_coords(self) -> None:
        """
        Test vector feature coordinate rotation.
        """
        quaternion = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        expected_result: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]], [[1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]]])
        result = rotate_coords(self.coords, quaternion)
        np.testing.assert_allclose(expected_result, result)

    def test_translate_coords(self) -> None:
        """
        Test vector feature coordinate translation.
        """
        translation_value: npt.NDArray[np.float32] = np.array([1.0, 0.0, -1.0])
        expected_result: npt.NDArray[np.float32] = np.array([[[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]], [[2.0, 0.0], [0.0, -1.0], [2.0, -1.0]]])
        result = translate_coords(self.coords, translation_value)
        np.testing.assert_allclose(expected_result, result)
        result = translate_coords(self.coords, translation_value, self.avails)
        expected_result[0][0] = [0.0, 0.0]
        np.testing.assert_allclose(expected_result, result)
        result = translate_coords(torch.from_numpy(self.coords), torch.from_numpy(translation_value), torch.from_numpy(self.avails))
        torch.testing.assert_allclose(torch.from_numpy(expected_result), result)

    def test_scale_coords(self) -> None:
        """
        Test vector feature coordinate scaling.
        """
        scale_value: npt.NDArray[np.float32] = np.array([-2.0, 0.0, -1.0])
        expected_result: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0]], [[-2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]])
        result = scale_coords(self.coords, scale_value)
        np.testing.assert_allclose(expected_result, result)
        result = scale_coords(torch.from_numpy(self.coords), torch.from_numpy(scale_value))
        torch.testing.assert_allclose(torch.from_numpy(expected_result), result)

    def test_xflip_coords(self) -> None:
        """
        Test flipping vector feature coordinates about X-axis.
        """
        expected_result: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]], [[-1.0, 0.0], [1.0, -1.0], [-1.0, -1.0]]])
        result = xflip_coords(self.coords)
        np.testing.assert_allclose(expected_result, result)
        result = xflip_coords(torch.from_numpy(self.coords))
        torch.testing.assert_allclose(torch.from_numpy(expected_result), result)

    def test_yflip_coords(self) -> None:
        """
        Test flipping vector feature coordinates about Y-axis.
        """
        expected_result: npt.NDArray[np.float32] = np.array([[[0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]], [[1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]]])
        result = yflip_coords(self.coords)
        np.testing.assert_allclose(expected_result, result)
        result = yflip_coords(torch.from_numpy(self.coords))
        torch.testing.assert_allclose(torch.from_numpy(expected_result), result)

class TestUtilsType(unittest.TestCase):
    """Test utils_type functions."""

    def test_is_TorchModuleWrapper_config(self) -> None:
        """Tests that is_TorchModuleWrapper_config works as expected."""
        mock_config = DictConfig({'model_config': 'some_value', 'checkpoint_path': 'some_value', 'some_other_key': 'some_value'})
        expect_true = is_TorchModuleWrapper_config(mock_config)
        self.assertTrue(expect_true)
        mock_config.pop('some_other_key')
        expect_true = is_TorchModuleWrapper_config(mock_config)
        self.assertTrue(expect_true)
        mock_config.pop('model_config')
        expect_false = is_TorchModuleWrapper_config(mock_config)
        self.assertFalse(expect_false)
        mock_config.pop('checkpoint_path')
        expect_false = is_TorchModuleWrapper_config(mock_config)
        self.assertFalse(expect_false)

    def test_is_target_type(self) -> None:
        """Tests that is_target_type works as expected."""
        mock_config_test_utils_mock_type = DictConfig({'_target_': f'{__name__}.TestUtilsTypeMockType'})
        mock_config_test_utils_another_mock_type = DictConfig({'_target_': f'{__name__}.TestUtilsTypeAnotherMockType'})
        expect_true = is_target_type(mock_config_test_utils_mock_type, TestUtilsTypeMockType)
        self.assertTrue(expect_true)
        expect_true = is_target_type(mock_config_test_utils_another_mock_type, TestUtilsTypeAnotherMockType)
        self.assertTrue(expect_true)
        expect_false = is_target_type(mock_config_test_utils_mock_type, TestUtilsTypeAnotherMockType)
        self.assertFalse(expect_false)
        expect_false = is_target_type(mock_config_test_utils_another_mock_type, TestUtilsTypeMockType)
        self.assertFalse(expect_false)

    def test_validate_type(self) -> None:
        """Tests that validate_type works as expected."""
        test_utils_type_mock_type = TestUtilsTypeMockType()
        validate_type(test_utils_type_mock_type, TestUtilsTypeMockType)
        with self.assertRaises(AssertionError):
            validate_type(test_utils_type_mock_type, TestUtilsTypeAnotherMockType)

    def test_are_the_same_type(self) -> None:
        """Tests that are_the_same_type works as expected."""
        test_utils_type_mock_type = TestUtilsTypeMockType()
        another_test_utils_type_mock_type = TestUtilsTypeMockType()
        test_utils_type_another_mock_type = TestUtilsTypeAnotherMockType()
        are_the_same_type(test_utils_type_mock_type, another_test_utils_type_mock_type)
        with self.assertRaises(AssertionError):
            are_the_same_type(test_utils_type_mock_type, test_utils_type_another_mock_type)

    def test_validate_dict_type(self) -> None:
        """Tests that validate_dict_type works as expected."""
        mock_config = DictConfig({'_convert_': 'all', 'correct_object': {'_target_': f'{__name__}.TestUtilsTypeMockType', 'a': 1, 'b': 2.5}, 'correct_object_2': {'_target_': f'{__name__}.TestUtilsTypeMockType', 'a': 1, 'b': 2.5}})
        instantiated_config = hydra.utils.instantiate(mock_config)
        validate_dict_type(instantiated_config, TestUtilsTypeMockType)
        mock_config.other_object = {'_target_': f'{__name__}.TestUtilsTypeAnotherMockType', 'c': 1}
        instantiated_config = hydra.utils.instantiate(mock_config)
        with self.assertRaises(AssertionError):
            validate_dict_type(instantiated_config, TestUtilsTypeMockType)

    def test_find_builder_in_config(self) -> None:
        """Tests that find_builder_in_config works as expected."""
        mock_config = DictConfig({'correct_object': {'_target_': f'{__name__}.TestUtilsTypeMockType', 'a': 1, 'b': 2.5}, 'other_object': {'_target_': f'{__name__}.TestUtilsTypeAnotherMockType', 'c': 1}})
        test_utils_mock_type = find_builder_in_config(mock_config, TestUtilsTypeMockType)
        self.assertTrue(is_target_type(test_utils_mock_type, TestUtilsTypeMockType))
        test_utils_another_mock_type = find_builder_in_config(mock_config, TestUtilsTypeAnotherMockType)
        self.assertTrue(is_target_type(test_utils_another_mock_type, TestUtilsTypeAnotherMockType))
        del mock_config.other_object
        with self.assertRaises(ValueError):
            find_builder_in_config(mock_config, TestUtilsTypeAnotherMockType)

