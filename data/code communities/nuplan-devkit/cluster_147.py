# Cluster 147

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

def to_tensor(data: FeatureDataType) -> torch.Tensor:
    """
    Convert data to tensor
    :param data which is either numpy or Tensor
    :return torch.Tensor
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        raise ValueError(f'Unknown type: {type(data)}')

@dataclass
class GenericAgents(AbstractModelFeature):
    """
    Model input feature representing the present and past states of the ego and agents.

    The structure includes:
        ego: List[<np.ndarray: num_frames, 7>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The last dimension is the ego pose (x, y, heading) velocities (vx, vy) accelerations (ax, ay) at time t.
            Example dimensions: 8 (batch_size) x 5 (1 present + 4 past frames) x 7
        agents: Dict[str, List[<np.ndarray: num_frames, num_agents, 8>]].
            Agent features indexed by agent feature type.
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The num_agents is padded to fit the largest number of agents across all frames.
            The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate)
             and size (length, width) at time t.

    The present/past frames dimension is populated in increasing chronological order, i.e. (t_-N, ..., t_-1, t_0)
    where N is the number of frames in the feature

    In both cases, the outer List represent number of batches. This is a special feature where each batch entry
    can have different size. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """
    ego: List[FeatureDataType]
    agents: Dict[str, List[FeatureDataType]]

    def __post_init__(self) -> None:
        """Sanitize attributes of dataclass."""
        if not all([len(self.ego) == len(agent) for agent in self.agents.values()]):
            raise AssertionError('Batch size inconsistent across features!')
        if len(self.ego) == 0:
            raise AssertionError('Batch size has to be > 0!')
        if self.ego[0].ndim != 2:
            raise AssertionError(f'Ego feature samples does not conform to feature dimensions! Got ndim: {self.ego[0].ndim} , expected 2 [num_frames, 7]')
        if 'EGO' in self.agents.keys():
            raise AssertionError('EGO not a valid agents feature type!')
        for feature_name in self.agents.keys():
            if feature_name not in TrackedObjectType._member_names_:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')
        for agent in self.agents.values():
            if agent[0].ndim != 3:
                raise AssertionError(f'Agent feature samples does not conform to feature dimensions! Got ndim: {agent[0].ndim} , expected 3 [num_frames, num_agents, 8]')
        for sample_idx in range(len(self.ego)):
            if int(self.ego[sample_idx].shape[0]) != self.num_frames or not all([int(agent[sample_idx].shape[0]) == self.num_frames for agent in self.agents.values()]):
                raise AssertionError('Agent feature samples have different number of frames!')

    def _validate_ego_query(self, sample_idx: int) -> None:
        """
        Validate ego sample query is valid.
        :param sample_idx: the batch index of interest.
        :raise
            ValueError if sample_idx invalid.
            RuntimeError if feature at given sample index is empty.
        """
        if self.batch_size < sample_idx:
            raise ValueError(f'Requsted sample index {sample_idx} larger than batch size {self.batch_size}!')
        if self.ego[sample_idx].size == 0:
            raise RuntimeError('Feature is empty!')

    def _validate_agent_query(self, agent_type: str, sample_idx: int) -> None:
        """
        Validate agent type, sample query is valid.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :raise ValueError if agent_type or sample_idx invalid.
        """
        if agent_type not in TrackedObjectType._member_names_:
            raise ValueError(f'Invalid agent type: {agent_type}')
        if agent_type not in self.agents.keys():
            raise ValueError(f'Agent type: {agent_type} is unavailable!')
        if self.batch_size < sample_idx:
            raise ValueError(f'Requsted sample index {sample_idx} larger than batch size {self.batch_size}!')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return len(self.ego) > 0 and all([len(agent) > 0 for agent in self.agents.values()]) and all([len(self.ego) == len(agent) for agent in self.agents.values()]) and (len(self.ego[0]) > 0) and all([len(agent[0]) > 0 for agent in self.agents.values()]) and all([len(self.ego[0]) == len(agent[0]) > 0 for agent in self.agents.values()]) and (self.ego[0].shape[-1] == self.ego_state_dim()) and all([agent[0].shape[-1] == self.agents_states_dim() for agent in self.agents.values()])

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches.
        """
        return len(self.ego)

    @classmethod
    def collate(cls, batch: List[GenericAgents]) -> GenericAgents:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        agents: Dict[str, List[FeatureDataType]] = defaultdict(list)
        for sample in batch:
            for agent_name, agent in sample.agents.items():
                agents[agent_name] += [agent[0]]
        return GenericAgents(ego=[item.ego[0] for item in batch], agents=agents)

    def to_feature_tensor(self) -> GenericAgents:
        """Implemented. See interface."""
        return GenericAgents(ego=[to_tensor(sample) for sample in self.ego], agents={agent_name: [to_tensor(sample) for sample in agent] for agent_name, agent in self.agents.items()})

    def to_device(self, device: torch.device) -> GenericAgents:
        """Implemented. See interface."""
        return GenericAgents(ego=[to_tensor(ego).to(device=device) for ego in self.ego], agents={agent_name: [to_tensor(sample).to(device=device) for sample in agent] for agent_name, agent in self.agents.items()})

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> GenericAgents:
        """Implemented. See interface."""
        return GenericAgents(ego=data['ego'], agents=data['agents'])

    def unpack(self) -> List[GenericAgents]:
        """Implemented. See interface."""
        return [GenericAgents(ego=[self.ego[sample_idx]], agents={agent_name: [agent[sample_idx]] for agent_name, agent in self.agents.items()}) for sample_idx in range(self.batch_size)]

    def num_agents_in_sample(self, agent_type: str, sample_idx: int) -> int:
        """
        Returns the number of agents at a given batch for given agent feature type.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: number of agents in the given batch.
        """
        self._validate_agent_query(agent_type, sample_idx)
        return self.agents[agent_type][sample_idx].shape[1]

    @staticmethod
    def ego_state_dim() -> int:
        """
        :return: ego state dimension.
        """
        return GenericEgoFeatureIndex.dim()

    @staticmethod
    def agents_states_dim() -> int:
        """
        :return: agent state dimension.
        """
        return GenericAgentFeatureIndex.dim()

    @property
    def num_frames(self) -> int:
        """
        :return: number of frames.
        """
        return int(self.ego[0].shape[0])

    @property
    def ego_feature_dim(self) -> int:
        """
        :return: ego feature dimension.
        """
        return GenericAgents.ego_state_dim() * self.num_frames

    @property
    def agents_features_dim(self) -> int:
        """
        :return: ego feature dimension.
        """
        return GenericAgents.agents_states_dim() * self.num_frames

    def has_agents(self, agent_type: str, sample_idx: int) -> bool:
        """
        Check whether agents of specified type exist in the feature.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: whether agents exist in the feature.
        """
        self._validate_agent_query(agent_type, sample_idx)
        return self.num_agents_in_sample(agent_type, sample_idx) > 0

    def agent_processing_by_type(self, processing_function: Callable[[str, int], FeatureDataType], sample_idx: int) -> FeatureDataType:
        """
        Apply agent processing functions across all agent types in features for given batch sample.
        :param processing_function: function to apply across agent types
        :param sample_idx: the batch index of interest.
        :return Processed agent feature across agent types.
        """
        agents: List[FeatureDataType] = []
        for agent_type in self.agents.keys():
            if self.has_agents(agent_type, sample_idx):
                agents.append(processing_function(agent_type, sample_idx))
        if len(agents) == 0:
            if isinstance(self.ego[sample_idx], torch.Tensor):
                return torch.empty((0, len(self.agents.keys()) * self.num_frames * GenericAgentFeatureIndex.dim()), dtype=self.ego[sample_idx].dtype, device=self.ego[sample_idx].device)
            else:
                return np.empty((0, len(self.agents.keys()) * self.num_frames * GenericAgentFeatureIndex.dim()), dtype=self.ego[sample_idx].dtype)
        elif isinstance(agents[0], torch.Tensor):
            return torch.cat(agents, dim=0)
        else:
            return np.concatenate(agents, axis=0)

    def get_flatten_agents_features_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Flatten agents' features of specified type by stacking the agents' states along the num_frame dimension
        <np.ndarray: num_frames, num_agents, 8>] -> <np.ndarray: num_agents, num_frames x 8>].

        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, num_frames x 8>] agent feature.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.num_agents_in_sample(agent_type, sample_idx) == 0:
            if isinstance(self.ego[sample_idx], torch.Tensor):
                return torch.empty((0, self.num_frames * GenericAgentFeatureIndex.dim()), dtype=self.ego[sample_idx].dtype, device=self.ego[sample_idx].device)
            else:
                return np.empty((0, self.num_frames * GenericAgentFeatureIndex.dim()), dtype=self.ego[sample_idx].dtype)
        data = self.agents[agent_type][sample_idx]
        axes = (1, 0) if isinstance(data, torch.Tensor) else (1, 0, 2)
        return data.transpose(*axes).reshape(data.shape[1], -1)

    def get_flatten_agents_features_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Flatten agents' features of all types by stacking the agents' states along the num_frame dimension
        <np.ndarray: num_frames, num_agents, 8>] -> <np.ndarray: num_agents, num_frames x 8>].

        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents, num_frames x 8>] agent feature.
        """
        return self.agent_processing_by_type(self.get_flatten_agents_features_by_type_in_sample, sample_idx)

    def get_present_ego_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present ego in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: 8>. ego at sample index.
        """
        self._validate_ego_query(sample_idx)
        return self.ego[sample_idx][-1]

    def get_present_agents_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Return the present agents of specified type in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, 8>. all agents at sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError('Feature is empty!')
        return self.agents[agent_type][sample_idx][-1]

    def get_present_agents_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present agents of all types in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents, 8>. all agents at sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        return self.agent_processing_by_type(self.get_present_agents_by_type_in_sample, sample_idx)

    def get_ego_agents_center_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return ego center in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: 2>. (x, y) positions of the ego's center at sample index.
        """
        self._validate_ego_query(sample_idx)
        return self.get_present_ego_in_sample(sample_idx)[:GenericEgoFeatureIndex.y() + 1]

    def get_agents_centers_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's centers in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, 2>. (x, y) positions of the agents' centers at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError('Feature is empty!')
        return self.get_present_agents_by_type_in_sample(agent_type, sample_idx)[:, :GenericAgentFeatureIndex.y() + 1]

    def get_agents_centers_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of all types' centers in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents, 2>.
            (x, y) positions of the agents' centers at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        return self.agent_processing_by_type(self.get_agents_centers_by_type_in_sample, sample_idx)

    def get_agents_length_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's length at the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents>. lengths of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError('Feature is empty!')
        return self.get_present_agents_by_type_in_sample(agent_type, sample_idx)[:, GenericAgentFeatureIndex.length()]

    def get_agents_length_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of all types' length at the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents>. lengths of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        return self.agent_processing_by_type(self.get_agents_length_by_type_in_sample, sample_idx)

    def get_agents_width_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's width in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents>. width of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError('Feature is empty!')
        return self.get_present_agents_by_type_in_sample(agent_type, sample_idx)[:, GenericAgentFeatureIndex.width()]

    def get_agents_width_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of all types' width in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents>. width of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty
        """
        return self.agent_processing_by_type(self.get_agents_width_by_type_in_sample, sample_idx)

    def get_agent_corners_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's corners in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, 4, 3>. (x, y, 1) positions of all the agents' corners at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError('Feature is empty!')
        widths = self.get_agents_width_by_type_in_sample(agent_type, sample_idx)
        lengths = self.get_agents_length_by_type_in_sample(agent_type, sample_idx)
        half_widths = widths / 2.0
        half_lengths = lengths / 2.0
        feature_cls = np.array if isinstance(widths, np.ndarray) else torch.Tensor
        return feature_cls([[[half_length, half_width, 1.0], [-half_length, half_width, 1.0], [-half_length, -half_width, 1.0], [half_length, -half_width, 1.0]] for half_width, half_length in zip(half_widths, half_lengths)])

    def get_agent_corners_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of all types' corners in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_types, num_agents, 4, 3>.
            (x, y, 1) positions of all the agents' corners at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        return self.agent_processing_by_type(self.get_agent_corners_by_type_in_sample, sample_idx)

@dataclass
class Agents(AbstractModelFeature):
    """
    Model input feature representing the present and past states of the ego and agents.

    The structure inludes:
        ego: List[<np.ndarray: num_frames, 3>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The last dimension is the ego pose (x, y, heading) at time t.
            Example dimensions: 8 (batch_size) x 5 (1 present + 4 past frames) x 3
        agents: List[<np.ndarray: num_frames, num_agents, 8>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The num_agents is padded to fit the largest number of agents across all frames.
            The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate)
             and size (length, width) at time t.

    The present/past frames dimension is populated in increasing chronological order, i.e. (t_-N, ..., t_-1, t_0)
    where N is the number of frames in the feature

    In both cases, the outer List represent number of batches. This is a special feature where each batch entry
    can have different size. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """
    ego: List[FeatureDataType]
    agents: List[FeatureDataType]

    def __post_init__(self) -> None:
        """Sanitize attributes of dataclass."""
        if len(self.ego) != len(self.agents):
            raise AssertionError(f'Not consistent length of batches! {len(self.ego)} != {len(self.agents)}')
        if len(self.ego) == 0:
            raise AssertionError('Batch size has to be > 0!')
        if self.ego[0].ndim != 2:
            raise AssertionError(f'Ego feature samples does not conform to feature dimensions! Got ndim: {self.ego[0].ndim} , expected 2 [num_frames, 3]')
        if self.agents[0].ndim != 3:
            raise AssertionError(f'Agent feature samples does not conform to feature dimensions! Got ndim: {self.agents[0].ndim} , expected 3 [num_frames, num_agents, 8]')
        for i in range(len(self.ego)):
            if int(self.ego[i].shape[0]) != self.num_frames or int(self.agents[i].shape[0]) != self.num_frames:
                raise AssertionError('Agent feature samples have different number of frames!')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return len(self.ego) > 0 and len(self.agents) > 0 and (len(self.ego) == len(self.agents)) and (len(self.ego[0]) > 0) and (len(self.agents[0]) > 0) and (len(self.ego[0]) == len(self.agents[0]) > 0) and (self.ego[0].shape[-1] == self.ego_state_dim()) and (self.agents[0].shape[-1] == self.agents_states_dim())

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        return len(self.ego)

    @classmethod
    def collate(cls, batch: List[Agents]) -> Agents:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return Agents(ego=[item.ego[0] for item in batch], agents=[item.agents[0] for item in batch])

    def to_feature_tensor(self) -> Agents:
        """Implemented. See interface."""
        return Agents(ego=[to_tensor(ego) for ego in self.ego], agents=[to_tensor(agents) for agents in self.agents])

    def to_device(self, device: torch.device) -> Agents:
        """Implemented. See interface."""
        return Agents(ego=[to_tensor(ego).to(device=device) for ego in self.ego], agents=[to_tensor(agents).to(device=device) for agents in self.agents])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Agents:
        """Implemented. See interface."""
        return Agents(ego=data['ego'], agents=data['agents'])

    def unpack(self) -> List[Agents]:
        """Implemented. See interface."""
        return [Agents([ego], [agents]) for ego, agents in zip(self.ego, self.agents)]

    def num_agents_in_sample(self, sample_idx: int) -> int:
        """
        Returns the number of agents at a given batch
        :param sample_idx: the batch index of interest
        :return: number of agents in the given batch
        """
        return self.agents[sample_idx].shape[1]

    @staticmethod
    def ego_state_dim() -> int:
        """
        :return: ego state dimension
        """
        return EgoFeatureIndex.dim()

    @staticmethod
    def agents_states_dim() -> int:
        """
        :return: agent state dimension
        """
        return AgentFeatureIndex.dim()

    @property
    def num_frames(self) -> int:
        """
        :return: number of frames.
        """
        return int(self.ego[0].shape[0])

    @property
    def ego_feature_dim(self) -> int:
        """
        :return: ego feature dimension. Note, the plus one is to account for the present frame
        """
        return Agents.ego_state_dim() * self.num_frames

    @property
    def agents_features_dim(self) -> int:
        """
        :return: ego feature dimension. Note, the plus one is to account for the present frame
        """
        return Agents.agents_states_dim() * self.num_frames

    def has_agents(self, batch_idx: int) -> bool:
        """
        Check whether agents exist in the feature.
        :param batch_idx: the batch index of interest
        :return: whether agents exist in the feature
        """
        return self.num_agents_in_sample(batch_idx) > 0

    def get_flatten_agents_features_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Flatten agents' features by stacking the agents' states along the num_frame dimension
        <np.ndarray: num_frames, num_agents, 8>] -> <np.ndarray: num_agents, num_frames x 8>]

        :param sample_idx: the sample index of interest
        :return: <FeatureDataType: num_agents, num_frames x 8>] agent feature
        """
        if self.num_agents_in_sample(sample_idx) == 0:
            if isinstance(self.ego[sample_idx], torch.Tensor):
                return torch.empty((0, self.num_frames * AgentFeatureIndex.dim()), dtype=self.ego[sample_idx].dtype, device=self.ego[sample_idx].device)
            else:
                return np.empty((0, self.num_frames * AgentFeatureIndex.dim()), dtype=self.ego[sample_idx].dtype)
        data = self.agents[sample_idx]
        axes = (1, 0) if isinstance(data, torch.Tensor) else (1, 0, 2)
        return data.transpose(*axes).reshape(data.shape[1], -1)

    def get_present_ego_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present ego in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 8>. ego at sample index
        """
        return self.ego[sample_idx][-1]

    def get_present_agents_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return the present agents in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: num_agents, 8>. all agents at sample index
        """
        return self.agents[sample_idx][-1]

    def get_ego_agents_center_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return ego center in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: 2>. (x, y) positions of the ego's center at sample index
        """
        return self.get_present_ego_in_sample(sample_idx)[:EgoFeatureIndex.y() + 1]

    def get_agents_centers_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents'centers in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: num_agents, 2>. (x, y) positions of the agents' centers at the sample index
        """
        return self.get_present_agents_in_sample(sample_idx)[:, :AgentFeatureIndex.y() + 1]

    def get_agents_length_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents' length in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: num_agents>. lengths of all the agents at the sample index
        """
        return self.get_present_agents_in_sample(sample_idx)[:, AgentFeatureIndex.length()]

    def get_agents_width_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents' width in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: num_agents>. width of all the agents at the sample index
        """
        return self.get_present_agents_in_sample(sample_idx)[:, AgentFeatureIndex.width()]

    def get_agent_corners_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents' corners in the given sample index
        :param sample_idx: the batch index of interest
        :return: <FeatureDataType: num_agents, 4, 3>. (x, y, 1) positions of all the agents' corners at the sample index
        """
        widths = self.get_agents_width_in_sample(sample_idx)
        lengths = self.get_agents_length_in_sample(sample_idx)
        half_widths = widths / 2.0
        half_lengths = lengths / 2.0
        feature_cls = np.array if isinstance(widths, np.ndarray) else torch.Tensor
        return feature_cls([[[half_length, half_width, 1.0], [-half_length, half_width, 1.0], [-half_length, -half_width, 1.0], [half_length, -half_width, 1.0]] for half_width, half_length in zip(half_widths, half_lengths)])

@dataclass
class AgentsTrajectories(AbstractModelFeature):
    """
    Model input feature representing the present and past states of the ego and agents.

    The structure inludes:
        agents: List[<np.ndarray: num_frames, num_agents, 6>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The num_agents is padded to fit the largest number of agents across all frames.
            The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate) at time t.

    The present/future frames dimension is populated in ascending chronological order, i.e. (t_1, t_2, ..., t_n)

    The outer List represent number of batches. This is a special feature where each batch entry
    can have different size. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """
    data: List[FeatureDataType]

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        if len(self.data) == 0:
            raise AssertionError('Batch size has to be > 0!')

    @property
    def batch_size(self) -> int:
        """
        :return: batch size
        """
        return len(self.data)

    @staticmethod
    def states_dim() -> int:
        """
        :return: agent state dimension
        """
        return 6

    @property
    def num_frames(self) -> int:
        """
        :return: number of future frames. Note: this excludes the present frame
        """
        return int(self.data[0].shape[0])

    @property
    def features_dim(self) -> int:
        """
        :return: ego feature dimension
        """
        return self.num_frames * AgentsTrajectories.states_dim()

    @classmethod
    def collate(cls, batch: List[AgentsTrajectories]) -> AgentsTrajectories:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return AgentsTrajectories(data=[item.data[0] for item in batch])

    def to_feature_tensor(self) -> AgentsTrajectories:
        """Implemented. See interface."""
        return AgentsTrajectories(data=[to_tensor(data) for data in self.data])

    def to_device(self, device: torch.device) -> AgentsTrajectories:
        """Implemented. See interface."""
        return AgentsTrajectories(data=[data.to(device=device) for data in self.data])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AgentsTrajectories:
        """Implemented. See interface."""
        return AgentsTrajectories(data=data['data'])

    def unpack(self) -> List[AgentsTrajectories]:
        """Implemented. See interface."""
        return [AgentsTrajectories([data]) for data in self.data]

    def num_agents_in_sample(self, sample_idx: int) -> int:
        """
        Returns the number of agents at a given batch
        :param sample_idx: the batch index of interest
        :return: number of agents in the given batch
        """
        return int(self.data[sample_idx].shape[1])

    def has_agents(self, batch_idx: int) -> bool:
        """
        Check whether agents exist in the feature.
        :param batch_idx: the batch index of interest
        :return: whether agents exist in the feature
        """
        return self.num_agents_in_sample(batch_idx) > 0

    @property
    def xy(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: num_frames, num_agents, 2>] x, y of all agent across all frames
        """
        return [sample[..., :2] for sample in self.data]

    @property
    def heading(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: num_frames, num_agents, 1>] yaw of all agent across all frames
        """
        return [sample[..., 2] for sample in self.data]

    @property
    def poses(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: num_frames, num_agents, 3>] x, y, yaw of all agents across all frames
        """
        return [sample[..., :3] for sample in self.data]

    @property
    def xy_velocity(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: num_frames, num_agents, 2>] x velocity, y velocity of all agent across all frames
        """
        return [sample[..., 3:5] for sample in self.data]

    @property
    def yaw_rate(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: num_frames, num_agents, 1>] yaw_rate of all agents across all frames
        """
        return [sample[..., 5] for sample in self.data]

    @property
    def terminal_xy(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: terminal_frame, num_agents, 2>] x, y of all agents at terminal frame
        """
        return [sample[-1, :, :2] for sample in self.data]

    @property
    def terminal_heading(self) -> FeatureDataType:
        """
        :return: List[<np.ndarray: terminal_frame, num_agents, 1>] heading of all agents at terminal frame
        """
        return [sample[-1, :, 3] for sample in self.data]

    def get_agents_only_trajectories(self) -> AgentsTrajectories:
        """
        :return: A new AgentsTrajectories isntance with only trajecotries data of agents (ignoring ego AV).
        """
        return AgentsTrajectories([sample[1:] for sample in self.data])

    def reshape_to_agents(self) -> None:
        """
        Reshapes predicted agent data by number of agents
        """
        axes = (1, 0) if isinstance(self.data[0], torch.Tensor) else (1, 0, 2)
        self.data = [sample.transpose(*axes).reshape(-1, self.num_frames, self.states_dim()) for sample in self.data]

@dataclass
class Trajectory(AbstractModelFeature):
    """
    Dataclass that holds trajectory signals produced from the model or from the dataset for supervision.

    :param data: either a [num_batches, num_states, 3] or [num_states, 3] representing the trajectory
                 where se2_state is [x, y, heading] with units [meters, meters, radians].
    """
    data: FeatureDataType

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        array_dims = self.num_dimensions
        state_size = self.data.shape[-1]
        if array_dims != 2 and array_dims != 3:
            raise RuntimeError(f'Invalid trajectory array. Expected 2 or 3 dims, got {array_dims}.')
        if state_size != self.state_size():
            raise RuntimeError(f'Invalid trajectory array. Expected {self.state_size()} variables per state, got {state_size}.')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return len(self.data) > 0 and self.data.shape[-2] > 0 and (self.data.shape[-1] == self.state_size())

    def to_device(self, device: torch.device) -> Trajectory:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return Trajectory(data=self.data.to(device=device))

    def to_feature_tensor(self) -> Trajectory:
        """Inherited, see superclass."""
        return Trajectory(data=to_tensor(self.data))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Trajectory:
        """Implemented. See interface."""
        return Trajectory(data=data['data'])

    def unpack(self) -> List[Trajectory]:
        """Implemented. See interface."""
        return [Trajectory(data[None]) for data in self.data]

    @staticmethod
    def state_size() -> int:
        """
        Size of each SE2 state of the trajectory.
        """
        return 3

    @property
    def xy(self) -> FeatureDataType:
        """
        :return: tensor of positions [..., x, y]
        """
        return self.data[..., :2]

    @property
    def terminal_position(self) -> FeatureDataType:
        """
        :return: tensor of terminal position [..., x, y]
        """
        return self.data[..., -1, :2]

    @property
    def terminal_heading(self) -> FeatureDataType:
        """
        :return: tensor of terminal position [..., heading]
        """
        return self.data[..., -1, 2]

    @property
    def position_x(self) -> FeatureDataType:
        """
        Array of x positions of trajectory.
        """
        return self.data[..., 0]

    @property
    def numpy_position_x(self) -> FeatureDataType:
        """
        Array of x positions of trajectory.
        """
        return np.asarray(self.data[..., 0])

    @property
    def position_y(self) -> FeatureDataType:
        """
        Array of y positions of trajectory.
        """
        return self.data[..., 1]

    @property
    def numpy_position_y(self) -> FeatureDataType:
        """
        Array of y positions of trajectory.
        """
        return np.asarray(self.data[..., 1])

    @property
    def heading(self) -> FeatureDataType:
        """
        Array of heading positions of trajectory.
        """
        return self.data[..., 2]

    @property
    def num_dimensions(self) -> int:
        """
        :return: dimensions of underlying data
        """
        return len(self.data.shape)

    @property
    def num_of_iterations(self) -> int:
        """
        :return: number of states in a trajectory
        """
        return int(self.data.shape[-2])

    @property
    def num_batches(self) -> Optional[int]:
        """
        :return: number of batches in the trajectory, None if trajectory does not have batch dimension
        """
        return None if self.num_dimensions <= 2 else self.data.shape[0]

    def state_at_index(self, index: int) -> FeatureDataType:
        """
        Query state at index along trajectory horizon
        :param index: along horizon
        :return: state corresponding to the index along trajectory horizon
        @raise in case index is not within valid range: 0 < index <= num_of_iterations
        """
        assert 0 <= index < self.num_of_iterations, f'Index is out of bounds! 0 <= {index} < {self.num_of_iterations}!'
        return self.data[..., index, :]

    def extract_number_of_last_states(self, number_of_states: int) -> Trajectory:
        """
        Extract last number_of_states from a trajectory
        :param number_of_states: from last point
        :return: shorter trajectory containing number_of_states from end of trajectory
        @raise in case number_of_states is not within valid range: 0 < number_of_states <= length
        """
        assert number_of_states > 0, f'number_of_states has to be > 0, {number_of_states} > 0!'
        length = self.num_of_iterations
        assert number_of_states <= length, f'number_of_states has to be smaller than length, {number_of_states} <= {length}!'
        return self.extract_trajectory_between(length - number_of_states, length)

    def extract_trajectory_between(self, start_index: int, end_index: Optional[int]) -> Trajectory:
        """
        Extract partial trajectory based on [start_index, end_index]
        :param start_index: starting index
        :param end_index: ending index
        :return: Trajectory
        @raise in case the desired ranges are not valid
        """
        if not end_index:
            end_index = self.num_of_iterations
        assert 0 <= start_index < self.num_of_iterations, f'Start index is out of bounds! 0 <= {start_index} < {self.num_of_iterations}!'
        assert 0 <= end_index <= self.num_of_iterations, f'Start index is out of bounds! 0 <= {end_index} <= {self.num_of_iterations}!'
        assert start_index < end_index, f'Start Index has to be smaller then end, {start_index} < {end_index}!'
        return Trajectory(data=self.data[..., start_index:end_index, :])

    @classmethod
    def append_to_trajectory(cls, trajectory: Trajectory, new_state: torch.Tensor) -> Trajectory:
        """
        Extend trajectory with a new state, in this case we require that both trajectory and new_state has dimension
        of 3, that means that they both have batch dimension
        :param trajectory: to be extended
        :param new_state: state with which trajectory should be extended
        :return: extended trajectory
        """
        assert trajectory.num_dimensions == 3, f'Trajectory dimension {trajectory.num_dimensions} != 3!'
        assert len(new_state.shape) == 3, f'New state dimension {new_state.shape} != 3!'
        if new_state.shape[0] != trajectory.data.shape[0]:
            raise RuntimeError(f'Not compatible shapes {new_state.shape} != {trajectory.data.shape}!')
        if new_state.shape[-1] != trajectory.data.shape[-1]:
            raise RuntimeError(f'Not compatible shapes {new_state.shape} != {trajectory.data.shape}!')
        return Trajectory(data=torch.cat((trajectory.data, new_state.clone()), dim=1))

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

