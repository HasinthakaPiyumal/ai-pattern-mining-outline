# Cluster 84

def function_to_load_model(dummy_var: Any) -> Tuple[bool, int]:
    """
    Dummy function
    return: gpu_available, num_threads avaialble for torch
    """
    model = RasterModel(feature_builders=[], target_builders=[], model_name='resnet50', pretrained=True, num_input_channels=4, num_features_per_pose=3, future_trajectory_sampling=TrajectorySampling(num_poses=10, time_horizon=5))
    gpu_available = torch.cuda.is_available()
    device = torch.device('cuda' if gpu_available else 'cpu')
    model.to(device)
    sleep(1)
    return (gpu_available, torch.get_num_threads())

def construct_simple_vector_map_ml_planner() -> MLPlanner:
    """
    Construct vector map simple planner
    :return: MLPlanner with vector map model
    """
    future_trajectory_param = TrajectorySampling(time_horizon=6.0, num_poses=12)
    past_trajectory_param = TrajectorySampling(time_horizon=2.0, num_poses=4)
    model = VectorMapSimpleMLP(num_output_features=36, hidden_size=128, vector_map_feature_radius=20, past_trajectory_sampling=past_trajectory_param, future_trajectory_sampling=future_trajectory_param)
    return MLPlanner(model=model)

def construct_raster_ml_planner() -> MLPlanner:
    """
    Construct Raster ML Planner
    :return: MLPlanner with raster model
    """
    future_trajectory_param = TrajectorySampling(time_horizon=6.0, num_poses=12)
    model = RasterModel(model_name='resnet50', pretrained=True, num_input_channels=4, num_features_per_pose=3, future_trajectory_sampling=future_trajectory_param, feature_builders=[RasterFeatureBuilder(map_features={'LANE': 1.0, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5}, num_input_channels=4, target_width=224, target_height=224, target_pixel_size=0.5, ego_width=2.297, ego_front_length=4.049, ego_rear_length=1.127, ego_longitudinal_offset=0.0, baseline_path_thickness=1)], target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_param)])
    return MLPlanner(model=model)

def construct_urban_driver_open_loop_ml_planner() -> MLPlanner:
    """
    Construct UrbanDriverOpenLoop ML Planner
    :return: MLPlanner with urban_driver_open_loop model
    """
    model_params = UrbanDriverOpenLoopModelParams(local_embedding_size=256, global_embedding_size=256, num_subgraph_layers=3, global_head_dropout=0.0)
    feature_params = UrbanDriverOpenLoopModelFeatureParams(feature_types={'NONE': -1, 'EGO': 0, 'VEHICLE': 1, 'BICYCLE': 2, 'PEDESTRIAN': 3, 'LANE': 4, 'STOP_LINE': 5, 'CROSSWALK': 6, 'LEFT_BOUNDARY': 7, 'RIGHT_BOUNDARY': 8, 'ROUTE_LANES': 9}, total_max_points=20, feature_dimension=8, agent_features=['VEHICLE', 'BICYCLE', 'PEDESTRIAN'], ego_dimension=3, agent_dimension=8, max_agents=30, past_trajectory_sampling=TrajectorySampling(time_horizon=2.0, num_poses=4), map_features=['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES'], max_elements={'LANE': 30, 'LEFT_BOUNDARY': 30, 'RIGHT_BOUNDARY': 30, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 30}, max_points={'LANE': 20, 'LEFT_BOUNDARY': 20, 'RIGHT_BOUNDARY': 20, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 20}, vector_set_map_feature_radius=35, interpolation_method='linear', disable_map=False, disable_agents=False)
    target_params = UrbanDriverOpenLoopModelTargetParams(num_output_features=36, future_trajectory_sampling=TrajectorySampling(time_horizon=6.0, num_poses=12))
    model = UrbanDriverOpenLoopModel(model_params, feature_params, target_params)
    return MLPlanner(model=model)

class TestMlPlanner(unittest.TestCase):
    """
    Test MLPlanner with two models.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.scenario = get_test_nuplan_scenario()

    def test_simple_vector_net_model(self) -> None:
        """Test Model Vector Map Simple"""
        self.run_test_ml_planner(construct_simple_vector_map_ml_planner())

    def test_raster_net_model(self) -> None:
        """Test Raster Net model"""
        self.run_test_ml_planner(construct_raster_ml_planner())

    def test_urban_driver_open_loop_model(self) -> None:
        """Test UrbanDriverOpenLoop model"""
        self.run_test_ml_planner(construct_urban_driver_open_loop_ml_planner())

    def run_test_ml_planner(self, planner: MLPlanner) -> None:
        """Tests if progress is calculated correctly"""
        scenario = self.scenario
        simulation_history_buffer_duration = 2
        buffer_size = int(simulation_history_buffer_duration / self.scenario.database_interval + 1)
        history = SimulationHistoryBuffer.initialize_from_scenario(buffer_size=buffer_size, scenario=self.scenario, observation_type=DetectionsTracks)
        initialization = PlannerInitialization(route_roadblock_ids=scenario.get_route_roadblock_ids(), mission_goal=scenario.get_mission_goal(), map_api=scenario.map_api)
        planner.initialize(initialization)
        trajectory = planner.compute_trajectory(PlannerInput(iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history, traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0))))
        self.assertNotEqual(trajectory, None)
        self.assertEqual(len(trajectory.get_sampled_trajectory()), planner._num_output_dim + 1)

class TestTrajectorySampling(unittest.TestCase):
    """
    Test trajectory sampling parameters
    """

    def test_wrong_setup(self) -> None:
        """Raise in case the sampling args are not consistent."""
        with self.assertRaises(ValueError):
            TrajectorySampling(num_poses=10, time_horizon=8, interval_length=0.5)
        with self.assertRaises(ValueError):
            TrajectorySampling()
        with self.assertRaises(ValueError):
            TrajectorySampling(num_poses=10)
            TrajectorySampling(time_horizon=10)
            TrajectorySampling(interval_length=10)

    def test_num_poses(self) -> None:
        """Test that num poses are set correctly."""
        sampling = TrajectorySampling(time_horizon=8, interval_length=0.5)
        self.assertEqual(sampling.time_horizon, 8)
        self.assertEqual(sampling.interval_length, 0.5)
        self.assertEqual(sampling.num_poses, 16)

    def test_num_poses_floating(self) -> None:
        """Test that num poses are set correctly even with floating point numbers."""
        sampling = TrajectorySampling(time_horizon=0.5, interval_length=0.1)
        self.assertEqual(sampling.time_horizon, 0.5)
        self.assertEqual(sampling.interval_length, 0.1)
        self.assertEqual(sampling.num_poses, 5)

    def test_interval(self) -> None:
        """Test that interval are set correctly."""
        sampling = TrajectorySampling(time_horizon=8, num_poses=16)
        self.assertEqual(sampling.time_horizon, 8)
        self.assertEqual(sampling.interval_length, 0.5)
        self.assertEqual(sampling.num_poses, 16)

    def test_time_horizon(self) -> None:
        """Test that time_horizon are set correctly."""
        sampling = TrajectorySampling(interval_length=0.5, num_poses=16)
        self.assertEqual(sampling.time_horizon, 8)
        self.assertEqual(sampling.interval_length, 0.5)
        self.assertEqual(sampling.num_poses, 16)

class LaneGCN(TorchModuleWrapper):
    """
    Vector-based model that uses a series of MLPs to encode ego and agent signals, a lane graph to encode vector-map
    elements and a fusion network to capture lane & agent intra/inter-interactions through attention layers.
    Dynamic map elements such as traffic light status and ego route information are also encoded in the fusion network.

    Implementation of the original LaneGCN paper ("Learning Lane Graph Representations for Motion Forecasting").
    """

    def __init__(self, map_net_scales: int, num_res_blocks: int, num_attention_layers: int, a2a_dist_threshold: float, l2a_dist_threshold: float, num_output_features: int, feature_dim: int, vector_map_feature_radius: int, vector_map_connection_scales: Optional[List[int]], past_trajectory_sampling: TrajectorySampling, future_trajectory_sampling: TrajectorySampling):
        """
        :param map_net_scales: Number of scales to extend the predecessor and successor lane nodes.
        :param num_res_blocks: Number of residual blocks for the GCN (LaneGCN uses 4).
        :param num_attention_layers: Number of times to repeatedly apply the attention layer.
        :param a2a_dist_threshold: [m] distance threshold for aggregating actor-to-actor nodes
        :param l2a_dist_threshold: [m] distance threshold for aggregating map-to-actor nodes
        :param num_output_features: number of target features
        :param feature_dim: hidden layer dimension
        :param vector_map_feature_radius: The query radius scope relative to the current ego-pose.
        :param vector_map_connection_scales: The hops of lane neighbors to extract, default 1 hop
        :param past_trajectory_sampling: Sampling parameters for past trajectory
        :param future_trajectory_sampling: Sampling parameters for future trajectory
        """
        super().__init__(feature_builders=[VectorMapFeatureBuilder(radius=vector_map_feature_radius, connection_scales=vector_map_connection_scales), AgentsFeatureBuilder(trajectory_sampling=past_trajectory_sampling)], target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_sampling)], future_trajectory_sampling=future_trajectory_sampling)
        self.feature_dim = feature_dim
        self.connection_scales = list(range(map_net_scales)) if vector_map_connection_scales is None else vector_map_connection_scales
        self.ego_input_dim = (past_trajectory_sampling.num_poses + 1) * Agents.ego_state_dim()
        self.agent_input_dim = (past_trajectory_sampling.num_poses + 1) * Agents.agents_states_dim()
        self.lane_net = LaneNet(lane_input_len=2, lane_feature_len=self.feature_dim, num_scales=map_net_scales, num_residual_blocks=num_res_blocks, is_map_feat=False)
        self.ego_feature_extractor = torch.nn.Sequential(nn.Linear(self.ego_input_dim, self.feature_dim), nn.ReLU(inplace=True), nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(inplace=True), LinearWithGroupNorm(self.feature_dim, self.feature_dim, num_groups=1, activation=False))
        self.agent_feature_extractor = torch.nn.Sequential(nn.Linear(self.agent_input_dim, self.feature_dim), nn.ReLU(inplace=True), nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(), LinearWithGroupNorm(self.feature_dim, self.feature_dim, num_groups=1, activation=False))
        self.actor2lane_attention = Actor2LaneAttention(actor_feature_len=self.feature_dim, lane_feature_len=self.feature_dim, num_attention_layers=num_attention_layers, dist_threshold_m=l2a_dist_threshold)
        self.lane2actor_attention = Lane2ActorAttention(lane_feature_len=self.feature_dim, actor_feature_len=self.feature_dim, num_attention_layers=num_attention_layers, dist_threshold_m=l2a_dist_threshold)
        self.actor2actor_attention = Actor2ActorAttention(actor_feature_len=self.feature_dim, num_attention_layers=num_attention_layers, dist_threshold_m=a2a_dist_threshold)
        self._mlp = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(), nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(), nn.Linear(self.feature_dim, num_output_features))

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_map": VectorMap,
                            "agents": Agents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        vector_map_data = cast(VectorMap, features['vector_map'])
        ego_agent_features = cast(Agents, features['agents'])
        ego_past_trajectory = ego_agent_features.ego
        batch_size = ego_agent_features.batch_size
        ego_features = []
        for sample_idx in range(batch_size):
            sample_ego_feature = self.ego_feature_extractor(ego_past_trajectory[sample_idx].reshape(1, -1))
            sample_ego_center = ego_agent_features.get_ego_agents_center_in_sample(sample_idx)
            if not vector_map_data.is_valid:
                num_coords = 1
                coords = torch.zeros((num_coords, 2, 2), device=sample_ego_feature.device, dtype=sample_ego_feature.dtype, layout=sample_ego_feature.layout)
                connections = {}
                for scale in self.connection_scales:
                    connections[scale] = torch.zeros((num_coords, 2), device=sample_ego_feature.device).long()
                lane_meta_tl = torch.zeros((num_coords, LaneSegmentTrafficLightData._encoding_dim), device=sample_ego_feature.device)
                lane_meta_route = torch.zeros((num_coords, LaneOnRouteStatusData._encoding_dim), device=sample_ego_feature.device)
                lane_meta = torch.cat((lane_meta_tl, lane_meta_route), dim=1)
            else:
                coords = vector_map_data.coords[sample_idx]
                connections = vector_map_data.multi_scale_connections[sample_idx]
                lane_meta_tl = vector_map_data.traffic_light_data[sample_idx]
                lane_meta_route = vector_map_data.on_route_status[sample_idx]
                lane_meta = torch.cat((lane_meta_tl, lane_meta_route), dim=1)
            lane_features = self.lane_net(coords, connections)
            lane_centers = coords.mean(axis=1)
            if ego_agent_features.has_agents(sample_idx):
                sample_agents_feature = self.agent_feature_extractor(ego_agent_features.get_flatten_agents_features_in_sample(sample_idx))
                sample_agents_center = ego_agent_features.get_agents_centers_in_sample(sample_idx)
            else:
                flattened_agents = torch.zeros((1, self.agent_input_dim), device=sample_ego_feature.device, dtype=sample_ego_feature.dtype, layout=sample_ego_feature.layout)
                sample_agents_feature = self.agent_feature_extractor(flattened_agents)
                sample_agents_center = torch.zeros_like(sample_ego_center).unsqueeze(dim=0)
            ego_agents_feature = torch.cat([sample_ego_feature, sample_agents_feature], dim=0)
            ego_agents_center = torch.cat([sample_ego_center.unsqueeze(dim=0), sample_agents_center], dim=0)
            lane_features = self.actor2lane_attention(ego_agents_feature, ego_agents_center, lane_features, lane_meta, lane_centers)
            ego_agents_feature = self.lane2actor_attention(lane_features, lane_centers, ego_agents_feature, ego_agents_center)
            ego_agents_feature = self.actor2actor_attention(ego_agents_feature, ego_agents_center)
            ego_features.append(ego_agents_feature[0])
        ego_features = torch.cat(ego_features).view(batch_size, -1)
        predictions = self._mlp(ego_features)
        return {'trajectory': Trajectory(data=convert_predictions_to_trajectory(predictions))}

class VectorMapSimpleMLP(ScriptableTorchModuleWrapper):
    """Simple vector-based model that encodes agents and map elements through an MLP."""

    def __init__(self, num_output_features: int, hidden_size: int, vector_map_feature_radius: int, past_trajectory_sampling: TrajectorySampling, future_trajectory_sampling: TrajectorySampling):
        """
        Initialize the simple vector map model.
        :param num_output_features: number of target features
        :param hidden_size: size of hidden layers of MLP
        :param vector_map_feature_radius: The query radius scope relative to the current ego-pose.
        :param past_trajectory_sampling: Sampling parameters for past trajectory
        :param future_trajectory_sampling: Sampling parameters for future trajectory
        """
        super().__init__(feature_builders=[VectorMapFeatureBuilder(radius=vector_map_feature_radius), AgentsFeatureBuilder(past_trajectory_sampling)], target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling)], future_trajectory_sampling=future_trajectory_sampling)
        self._hidden_size = hidden_size
        self.vectormap_mlp = create_mlp(input_size=2 * VectorMap.lane_coord_dim(), output_size=self._hidden_size, hidden_size=self._hidden_size)
        self.ego_mlp = create_mlp(input_size=(past_trajectory_sampling.num_poses + 1) * Agents.ego_state_dim(), output_size=self._hidden_size, hidden_size=self._hidden_size)
        self._agent_mlp_dim = (past_trajectory_sampling.num_poses + 1) * Agents.agents_states_dim()
        self.agent_mlp = create_mlp(input_size=self._agent_mlp_dim, output_size=self._hidden_size, hidden_size=self._hidden_size)
        self._mlp = create_mlp(input_size=3 * self._hidden_size, output_size=num_output_features, hidden_size=self._hidden_size)
        self._vector_map_flatten_lane_coord_dim = VectorMap.flatten_lane_coord_dim()
        self._trajectory_state_size = Trajectory.state_size()

    @torch.jit.unused
    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_map": VectorMap,
                            "agents": Agents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        vector_map_data = cast(VectorMap, features['vector_map'])
        ego_agents_feature = cast(Agents, features['agents'])
        tensor_inputs: Dict[str, torch.Tensor] = {}
        list_tensor_inputs = {'vector_map.coords': vector_map_data.coords, 'agents.ego': ego_agents_feature.ego, 'agents.agents': ego_agents_feature.agents}
        list_list_tensor_inputs: Dict[str, List[List[torch.Tensor]]] = {}
        output_tensors, output_list_tensors, output_list_list_tensors = self.scriptable_forward(tensor_inputs, list_tensor_inputs, list_list_tensor_inputs)
        return {'trajectory': Trajectory(data=output_tensors['trajectory'])}

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        ego_past_trajectory = list_tensor_data['agents.ego']
        ego_agents_agents = list_tensor_data['agents.agents']
        vector_map_coords = list_tensor_data['vector_map.coords']
        if len(vector_map_coords) != len(ego_agents_agents) or len(vector_map_coords) != len(ego_past_trajectory):
            raise ValueError(f'Mixed batch sizes passed to scriptable_forward: vector_map.coords = {len(vector_map_coords)}, agents.agents = {len(ego_agents_agents)}, agents.ego_past_trajectory={len(ego_past_trajectory)}')
        batch_size = len(vector_map_coords)
        vector_map_feature: List[torch.Tensor] = []
        agents_feature: List[torch.Tensor] = []
        ego_feature: List[torch.Tensor] = []
        for sample_idx in range(batch_size):
            sample_ego_feature = self.ego_mlp(ego_past_trajectory[sample_idx].view(1, -1))
            ego_feature.append(torch.max(sample_ego_feature, dim=0).values)
            vectormap_coords = vector_map_coords[sample_idx].reshape(-1, self._vector_map_flatten_lane_coord_dim)
            if vectormap_coords.numel() == 0:
                vectormap_coords = torch.zeros((1, self._vector_map_flatten_lane_coord_dim), dtype=vectormap_coords.dtype, device=vectormap_coords.device)
            sample_vectormap_feature = self.vectormap_mlp(vectormap_coords)
            vector_map_feature.append(torch.max(sample_vectormap_feature, dim=0).values)
            this_agents_feature = ego_agents_agents[sample_idx]
            agents_multiplier = float(min(this_agents_feature.shape[1], 1))
            if this_agents_feature.shape[1] > 0:
                orig_shape = this_agents_feature.shape
                flattened_agents = this_agents_feature.transpose(1, 0).reshape(orig_shape[1], -1)
            else:
                flattened_agents = torch.zeros((this_agents_feature.shape[0], self._agent_mlp_dim), device=sample_vectormap_feature.device, dtype=sample_vectormap_feature.dtype, layout=sample_vectormap_feature.layout)
            sample_agent_feature = self.agent_mlp(flattened_agents)
            sample_agent_feature *= agents_multiplier
            agents_feature.append(torch.max(sample_agent_feature, dim=0).values)
        vector_map_feature = torch.cat(vector_map_feature).reshape(batch_size, -1)
        ego_feature = torch.cat(ego_feature).reshape(batch_size, -1)
        agents_feature = torch.cat(agents_feature).reshape(batch_size, -1)
        input_features = torch.cat([vector_map_feature, ego_feature, agents_feature], dim=1)
        predictions = self._mlp(input_features)
        output_tensors: Dict[str, torch.Tensor] = {'trajectory': convert_predictions_to_trajectory(predictions, self._trajectory_state_size)}
        output_list_tensors: Dict[str, List[torch.Tensor]] = {}
        output_list_list_tensors: Dict[str, List[List[torch.Tensor]]] = {}
        return (output_tensors, output_list_tensors, output_list_list_tensors)

def create_mlp(input_size: int, output_size: int, hidden_size: int=128) -> torch.nn.Module:
    """
    Create MLP
    :param input_size: input feature size
    :param output_size: output feature size
    :param hidden_size: hidden layer
    :return: sequential network
    """
    return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

class TestUrbanDriverOpenLoop(unittest.TestCase):
    """Test UrbanDriverOpenLoopModel model."""

    def setUp(self) -> None:
        """Set up the test."""
        self.model_params = UrbanDriverOpenLoopModelParams(local_embedding_size=256, global_embedding_size=256, num_subgraph_layers=3, global_head_dropout=0.0)
        self.feature_params = UrbanDriverOpenLoopModelFeatureParams(feature_types={'NONE': -1, 'EGO': 0, 'VEHICLE': 1, 'BICYCLE': 2, 'PEDESTRIAN': 3, 'LANE': 4, 'STOP_LINE': 5, 'CROSSWALK': 6, 'LEFT_BOUNDARY': 7, 'RIGHT_BOUNDARY': 8, 'ROUTE_LANES': 9}, total_max_points=20, feature_dimension=8, agent_features=['VEHICLE', 'BICYCLE', 'PEDESTRIAN'], ego_dimension=3, agent_dimension=8, max_agents=30, past_trajectory_sampling=TrajectorySampling(time_horizon=2.0, num_poses=4), map_features=['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES'], max_elements={'LANE': 30, 'LEFT_BOUNDARY': 30, 'RIGHT_BOUNDARY': 30, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 30}, max_points={'LANE': 20, 'LEFT_BOUNDARY': 20, 'RIGHT_BOUNDARY': 20, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 20}, vector_set_map_feature_radius=35, interpolation_method='linear', disable_map=False, disable_agents=False)
        self.target_params = UrbanDriverOpenLoopModelTargetParams(num_output_features=36, future_trajectory_sampling=TrajectorySampling(time_horizon=6.0, num_poses=12))

    def _build_model(self) -> UrbanDriverOpenLoopModel:
        """
        Creates a new instance of a UrbanDriverOpenLoop with some default parameters.
        """
        model = UrbanDriverOpenLoopModel(self.model_params, self.feature_params, self.target_params)
        return model

    def _build_input_features(self, device: torch.device, include_agents: bool) -> FeaturesType:
        """
        Creates a set of input features for use with unit testing.
        :param device: The device on which to create the tensors.
        :param include_agents: If true, the generated input features will have agents.
            If not, then there will be no agents in the agents feature.
        :return: FeaturesType to be consumed by the model
        """
        num_frames = 5
        num_agents = num_frames if include_agents else 0
        coords: Dict[str, List[torch.Tensor]] = dict()
        traffic_light_data: Dict[str, List[torch.Tensor]] = dict()
        availabilities: Dict[str, List[torch.BoolTensor]] = dict()
        for feature_name in self.feature_params.map_features:
            coords[feature_name] = [torch.zeros((self.feature_params.max_elements[feature_name], self.feature_params.max_points[feature_name], VectorSetMap.coord_dim()), dtype=torch.float32, device=device)]
            availabilities[feature_name] = [torch.ones((self.feature_params.max_elements[feature_name], self.feature_params.max_points[feature_name]), dtype=torch.bool, device=device)]
        traffic_light_data['LANE'] = [torch.zeros((self.feature_params.max_elements['LANE'], self.feature_params.max_points['LANE'], VectorSetMap.traffic_light_status_dim()), dtype=torch.float32, device=device)]
        vector_set_map_feature = VectorSetMap(coords=coords, traffic_light_data=traffic_light_data, availabilities=availabilities)
        ego_agents = [torch.zeros((num_frames, GenericAgents.ego_state_dim()), dtype=torch.float32, device=device)]
        agent_agents = {feature_name: [torch.zeros((num_frames, num_agents, GenericAgents.agents_states_dim()), dtype=torch.float32, device=device)] for feature_name in self.feature_params.agent_features}
        generic_agents_feature = GenericAgents(ego=ego_agents, agents=agent_agents)
        return {'vector_set_map': vector_set_map_feature, 'generic_agents': generic_agents_feature}

    def _find_free_port(self) -> int:
        """
        Finds a free port to use for gloo server.
        :return: A port not in use.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            address, port = s.getsockname()
            return int(port)

    def _init_distributed_process_group(self) -> None:
        """
        Sets up the torch distributed processing server.
        :param port: The port to use for the gloo server.
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self._find_free_port())
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        torch.distributed.init_process_group(backend='gloo')

    def _assert_valid_output(self, model_output: TargetsType) -> None:
        """
        Validates that the output from the model has the correct keys and that the tensor is of the correct type.
        :param model_output: The output from the model.
        """
        self.assertTrue('trajectory' in model_output)
        self.assertTrue(isinstance(model_output['trajectory'], Trajectory))
        predicted_trajectory: Trajectory = model_output['trajectory']
        self.assertIsNotNone(predicted_trajectory.data)

    def _perform_backprop_step(self, optimizer: torch.optim.Optimizer, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], predictions: TargetsType) -> None:
        """
        Performs a backpropagation step.
        :param optimizer: The optimizer to use for training.
        :param loss_function: The loss function to use.
        :param predictions: The output from the model.
        """
        loss = loss_function(predictions['trajectory'].data, torch.zeros_like(predictions['trajectory'].data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_backprop(self) -> None:
        """
        Tests that the UrbanDriverOpenLoop model can train with DDP.
        This test was developed in response to an error related to zero agent input
        """
        self._init_distributed_process_group()
        device = torch.device('cpu')
        model = self._build_model().to(device)
        ddp_model = DDP(model, device_ids=None, output_device=None)
        optimizer = torch.optim.RMSprop(ddp_model.parameters())
        loss_function = torch.nn.MSELoss()
        num_epochs = 3
        for _ in range(num_epochs):
            for include_agents in [True, False]:
                input_features = self._build_input_features(device, include_agents=include_agents)
                predictions = ddp_model.forward(input_features)
                self._assert_valid_output(predictions)
                self._perform_backprop_step(optimizer, loss_function, predictions)

class TestActor2ActorAttention(unittest.TestCase):
    """Test actor-to-actor attention layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.actor_feature_len = 4
        self.num_attention_layers = 2
        self.dist_threshold_m = 6.0
        self.model = Actor2ActorAttention(self.actor_feature_len, self.num_attention_layers, self.dist_threshold_m)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_actors = 3
        actor_features = torch.zeros((num_actors, self.actor_feature_len))
        actor_centers = torch.zeros((num_actors, 2))
        output = self.model.forward(actor_features, actor_centers)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_actors, self.actor_feature_len))

class TestLane2ActorAttention(unittest.TestCase):
    """Test lane-to-actor attention layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.lane_feature_len = 4
        self.actor_feature_len = 4
        self.num_attention_layers = 2
        self.dist_threshold_m = 6.0
        self.model = Lane2ActorAttention(self.lane_feature_len, self.actor_feature_len, self.num_attention_layers, self.dist_threshold_m)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_lanes = 2
        num_actors = 3
        lane_features = torch.zeros((num_lanes, self.lane_feature_len))
        lane_centers = torch.zeros((num_lanes, 2))
        actor_features = torch.zeros((num_actors, self.actor_feature_len))
        actor_centers = torch.zeros((num_actors, 2))
        output = self.model.forward(lane_features, lane_centers, actor_features, actor_centers)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_actors, self.actor_feature_len))

class TestActor2LaneAttention(unittest.TestCase):
    """Test actor-to-lane attention layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.lane_feature_len = 4
        self.actor_feature_len = 4
        self.num_attention_layers = 2
        self.dist_threshold_m = 6.0
        self.model = Actor2LaneAttention(self.actor_feature_len, self.lane_feature_len, self.num_attention_layers, self.dist_threshold_m)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_lanes = 2
        num_actors = 3
        meta_info_len = 6
        lane_features = torch.zeros((num_lanes, self.lane_feature_len))
        lane_meta = torch.zeros((num_lanes, meta_info_len))
        lane_centers = torch.zeros((num_lanes, 2))
        actor_features = torch.zeros((num_actors, self.actor_feature_len))
        actor_centers = torch.zeros((num_actors, 2))
        output = self.model.forward(actor_features, actor_centers, lane_features, lane_meta, lane_centers)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_lanes, self.actor_feature_len))

class TestLaneNet(unittest.TestCase):
    """Test lane net layer."""

    def setUp(self) -> None:
        """Set up the test."""
        self.lane_input_len = 2
        self.lane_feature_len = 4
        self.num_scales = 2
        self.num_res_blocks = 3
        self.model = LaneNet(lane_input_len=self.lane_input_len, lane_feature_len=self.lane_feature_len, num_scales=self.num_scales, num_residual_blocks=self.num_res_blocks, is_map_feat=False)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_lanes = 4
        lane_input = torch.zeros((num_lanes, self.lane_input_len, 2))
        multi_scale_connections = {1: torch.tensor([[0, 1], [1, 2], [2, 3]]), 2: torch.tensor([[0, 2], [1, 3]])}
        vector_map = Munch(multi_scale_connections=multi_scale_connections)
        output = self.model.forward(lane_input, vector_map.multi_scale_connections)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_lanes, self.lane_feature_len))

class TestLaneGCN(unittest.TestCase):
    """Test LaneGCN model."""

    def _build_model(self) -> LaneGCN:
        """
        Creates a new instance of a LaneGCN with some default parameters.
        """
        model = LaneGCN(map_net_scales=4, num_res_blocks=4, num_attention_layers=5, a2a_dist_threshold=20, l2a_dist_threshold=20, num_output_features=12, feature_dim=32, vector_map_feature_radius=30, vector_map_connection_scales=[1, 2, 3, 4], past_trajectory_sampling=TrajectorySampling(num_poses=4, time_horizon=1.5), future_trajectory_sampling=TrajectorySampling(num_poses=12, time_horizon=6))
        return model

    def _build_input_features(self, device: torch.device, include_agents: bool, include_lanes: bool) -> FeaturesType:
        """
        Creates a set of input features for use with unit testing.
        :param device: The device on which to create the tensors.
        :param include_agents: If true, the generated input features will have agents.
            If not, then there will be no agents in the agents feature.
        :param include_lanes: If true, the generated input features will have lanes.
            If not, then there will be no lanes in the vectormap feature.
        :return: FeaturesType to be consumed by the model
        """
        num_frames = 5
        num_coords = 1000
        num_groupings = 100
        num_multi_scale_connections = 800
        num_lanes = num_coords if include_lanes else 0
        num_connections = num_multi_scale_connections if include_lanes else 0
        num_agents = num_frames if include_agents else 0
        vector_map_coords = [torch.zeros((num_lanes, VectorMap.lane_coord_dim(), VectorMap.lane_coord_dim()), dtype=torch.float32, device=device)]
        vector_map_lane_groupings = [[torch.zeros(num_groupings, device=device)]]
        multi_scale_connections = [{1: torch.zeros((num_connections, 2), device=device).long(), 2: torch.zeros((num_connections, 2), device=device).long(), 3: torch.zeros((num_connections, 2), device=device).long(), 4: torch.zeros((num_connections, 2), device=device).long()}]
        on_route_status = [torch.zeros((num_lanes, VectorMap.on_route_status_encoding_dim()), device=device)]
        traffic_light_data = [torch.zeros((num_lanes, 4), device=device)]
        vector_map_feature = VectorMap(coords=vector_map_coords, lane_groupings=vector_map_lane_groupings, multi_scale_connections=multi_scale_connections, on_route_status=on_route_status, traffic_light_data=traffic_light_data)
        ego_agents = [torch.zeros((num_frames, Agents.ego_state_dim()), dtype=torch.float32, device=device)]
        agent_agents = [torch.zeros((num_frames, num_agents, Agents.agents_states_dim()), dtype=torch.float32, device=device)]
        agents_feature = Agents(ego=ego_agents, agents=agent_agents)
        return {'vector_map': vector_map_feature, 'agents': agents_feature}

    def _find_free_port(self) -> int:
        """
        Finds a free port to use for gloo server.
        :return: A port not in use.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            address, port = s.getsockname()
            return int(port)

    def _init_distributed_process_group(self) -> None:
        """
        Sets up the torch distributed processing server.
        :param port: The port to use for the gloo server.
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self._find_free_port())
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        torch.distributed.init_process_group(backend='gloo')

    def _assert_valid_output(self, model_output: TargetsType) -> None:
        """
        Validates that the output from the model has the correct keys and that the tensor is of the correct type.
        :param model_output: The output from the model.
        """
        self.assertTrue('trajectory' in model_output)
        self.assertTrue(isinstance(model_output['trajectory'], Trajectory))
        predicted_trajectory: Trajectory = model_output['trajectory']
        self.assertIsNotNone(predicted_trajectory.data)

    def _perform_backprop_step(self, optimizer: torch.optim.Optimizer, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], predictions: TargetsType) -> None:
        """
        Performs a backpropagation step.
        :param optimizer: The optimizer to use for training.
        :param loss_function: The loss function to use.
        :param predictions: The output from the model.
        """
        loss = loss_function(predictions['trajectory'].data, torch.zeros_like(predictions['trajectory'].data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_backprop(self) -> None:
        """
        Tests that the LaneGCN model can train with DDP.
        This test was developed in response to an error related to zero agent input.
        """
        self._init_distributed_process_group()
        device = torch.device('cpu')
        model = self._build_model().to(device)
        ddp_model = DDP(model, device_ids=None, output_device=None)
        optimizer = torch.optim.RMSprop(ddp_model.parameters())
        loss_function = torch.nn.MSELoss()
        num_epochs = 3
        for _ in range(num_epochs):
            for include_agents in [True, False]:
                for include_lanes in [True, False]:
                    input_features = self._build_input_features(device, include_agents=include_agents, include_lanes=include_lanes)
                    predictions = ddp_model.forward(input_features)
                    self._assert_valid_output(predictions)
                    self._perform_backprop_step(optimizer, loss_function, predictions)

def _create_empty_vector_map_for_test(device: torch.device) -> VectorMap:
    """
    Helper function to create a dummy empty vector map solely for testing purposes.
    :param device: The device on which to place the constructed vector map.
    :return: A VectorMap with batch size 1 but otherwise empty features.
    """
    coords = [torch.zeros(size=(0, 2, 2), dtype=torch.float32, device=device)]
    lane_groupings = [[torch.zeros(size=(0,), dtype=torch.float32, device=device)]]
    multi_scale_connections = {1: [torch.zeros(size=(0, 2), dtype=torch.float32, device=device)]}
    on_route_status = [torch.zeros(size=(0, VectorMap.on_route_status_encoding_dim()), dtype=torch.float32, device=device)]
    traffic_light_data = [torch.zeros(size=(0, 4), dtype=torch.float32, device=device)]
    return VectorMap(coords=coords, lane_groupings=lane_groupings, multi_scale_connections=multi_scale_connections, on_route_status=on_route_status, traffic_light_data=traffic_light_data)

class TestVectorMapSimpleMLP(unittest.TestCase):
    """Test graph attention layer."""

    def _build_model(self) -> VectorMapSimpleMLP:
        """
        Creates a new instance of a VectorMapSimpleMLP with some default parameters.
        """
        num_output_features = 36
        hidden_size = 128
        vector_map_feature_radius = 20
        past_trajectory_sampling = TrajectorySampling(num_poses=4, time_horizon=1.5)
        future_trajectory_sampling = TrajectorySampling(num_poses=12, time_horizon=6)
        model = VectorMapSimpleMLP(num_output_features=num_output_features, hidden_size=hidden_size, vector_map_feature_radius=vector_map_feature_radius, past_trajectory_sampling=past_trajectory_sampling, future_trajectory_sampling=future_trajectory_sampling)
        return model

    def _build_input_features(self, device: torch.device, include_agents: bool) -> FeaturesType:
        """
        Creates a set of input features for use with unit testing.
        :param device: The device on which to create the tensors.
        :param include_agents: If true, the generated input features will have agents.
            If not, then there will be no agents in the agents feature.
        :return: FeaturesType to be consumed by the model
        """
        num_frames = 5
        num_coords = 1000
        num_groupings = 100
        num_multi_scale_connections = 800
        num_agents = num_frames if include_agents else 0
        vector_map_coords = [torch.zeros((num_coords, VectorMap.lane_coord_dim(), VectorMap.lane_coord_dim()), dtype=torch.float32, device=device)]
        vector_map_lane_groupings = [[torch.zeros(num_groupings, device=device)]]
        multi_scale_connections = {1: [torch.zeros((num_multi_scale_connections, 2), device=device)]}
        on_route_status = [torch.zeros((num_coords, VectorMap.on_route_status_encoding_dim()), device=device)]
        traffic_light_data = [torch.zeros((num_coords, 4), device=device)]
        vector_map_feature = VectorMap(coords=vector_map_coords, lane_groupings=vector_map_lane_groupings, multi_scale_connections=multi_scale_connections, on_route_status=on_route_status, traffic_light_data=traffic_light_data)
        ego_agents = [torch.zeros((num_frames, Agents.ego_state_dim()), dtype=torch.float32, device=device)]
        agent_agents = [torch.zeros((num_frames, num_agents, Agents.agents_states_dim()), dtype=torch.float32, device=device)]
        agents_feature = Agents(ego=ego_agents, agents=agent_agents)
        return {'vector_map': vector_map_feature, 'agents': agents_feature}

    def _assert_valid_output(self, model_output: TargetsType) -> None:
        """
        Validates that the output from the model has the correct keys and that the tensor is of the correct type.
        :param model_output: The output from the model.
        """
        self.assertTrue('trajectory' in model_output)
        self.assertTrue(isinstance(model_output['trajectory'], Trajectory))
        predicted_trajectory: Trajectory = model_output['trajectory']
        self.assertIsNotNone(predicted_trajectory.data)

    def _perform_backprop_step(self, optimizer: torch.optim.Optimizer, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], predictions: TargetsType) -> None:
        """
        Performs a backpropagation step.
        :param optimizer: The optimizer to use for training.
        :param loss_function: The loss function to use.
        :param predictions: The output from the model.
        """
        loss = loss_function(predictions['trajectory'].data, torch.zeros_like(predictions['trajectory'].data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _find_free_port(self) -> int:
        """
        Finds a free port to use for gloo server.
        :return: A port not in use.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            address, port = s.getsockname()
            return int(port)

    def _init_distributed_process_group(self) -> None:
        """
        Sets up the torch distributed processing server.
        :param port: The starting to use for the gloo server. If taken, it will increment by 1 until a free port is found.
        :param max_port: The maximum port number to try.
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self._find_free_port())
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        torch.distributed.init_process_group(backend='gloo')

    def _assert_valid_gradients_for_model(self, model: torch.nn.Module) -> None:
        """
        Validates that trainable parameters in a model have gradients after a backprop operation.
        :param model: The model with parameters to update following a forward/backward pass.
        """
        all_gradients_computed = all((param.grad is not None for param in model.parameters() if param.requires_grad))
        self.assertTrue(all_gradients_computed)

    def test_can_train_distributed(self) -> None:
        """
        Tests that the model can train with DDP.
        This test was developed in response to an error like this one:
        https://discuss.pytorch.org/t/need-help-runtimeerror-expected-to-have-finished-reduction-in-the-prior-iteration-before-starting-a-new-one/119247
        """
        self._init_distributed_process_group()
        device = torch.device('cpu')
        model = self._build_model().to(device)
        ddp_model = DDP(model, device_ids=None, output_device=None)
        optimizer = torch.optim.RMSprop(ddp_model.parameters())
        loss_function = torch.nn.MSELoss()
        num_epochs = 3
        for _ in range(num_epochs):
            for include_agents in [True, False]:
                input_features = self._build_input_features(device, include_agents=include_agents)
                predictions = ddp_model.forward(input_features)
                self._assert_valid_output(predictions)
                self._perform_backprop_step(optimizer, loss_function, predictions)
                self._assert_valid_gradients_for_model(ddp_model)

    def test_scripts_properly(self) -> None:
        """
        Test that the VectorMapSimpleMLP model scripts properly.
        """
        model = self._build_model()
        device = torch.device('cpu')
        input_features = self._build_input_features(device, include_agents=True)
        dummy_tensor_input: Dict[str, torch.Tensor] = {}
        dummy_list_tensor_input = {'vector_map.coords': input_features['vector_map'].coords, 'agents.ego': input_features['agents'].ego, 'agents.agents': input_features['agents'].agents}
        dummy_list_list_tensor_input: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_module = torch.jit.script(model)
        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_module.scriptable_forward(dummy_tensor_input, dummy_list_tensor_input, dummy_list_list_tensor_input)
        py_tensors, py_list_tensors, py_list_list_tensors = model.scriptable_forward(dummy_tensor_input, dummy_list_tensor_input, dummy_list_list_tensor_input)
        self.assertEqual(1, len(scripted_tensors))
        self.assertEqual(0, len(scripted_list_tensors))
        self.assertEqual(0, len(scripted_list_list_tensors))
        self.assertEqual(1, len(py_tensors))
        self.assertEqual(0, len(py_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))
        torch.testing.assert_allclose(py_tensors['trajectory'], scripted_tensors['trajectory'])

    def test_can_train_with_empty_vector_map(self) -> None:
        """In case of zero length vector map features, model training should not crash."""
        device = torch.device('cpu')
        test_features = self._build_input_features(device=device, include_agents=True)
        test_features['vector_map'] = _create_empty_vector_map_for_test(device=device)
        self.assertFalse(test_features['vector_map'].is_valid)
        model = self._build_model().to(device)
        optimizer = torch.optim.RMSprop(model.parameters())
        loss_function = torch.nn.MSELoss()
        predictions = model.forward(test_features)
        self._assert_valid_output(predictions)
        self._perform_backprop_step(optimizer, loss_function, predictions)
        self._assert_valid_gradients_for_model(model)

class TestKinematicHistoryAgentAugmentation(unittest.TestCase):
    """
    Test agent augmentation that perturbs the current ego position and generates a feasible trajectory history that
    satisfies a set of kinematic constraints.
    """

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)
        self.radius = 50
        self.features = {}
        self.features['agents'] = Agents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.004325964, -0.00069646863, -9.3163371e-06], [0.0024353617, -0.00037753209, 4.7789731e-06], [0.0011352128, -0.0001273104, 3.8040514e-05], [1.1641532e-10, 0.0, -3.0870851e-19]]), np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.004325964, -0.00069646863, -9.3163371e-06], [0.0024353617, -0.00037753209, 4.7789731e-06], [0.0011352128, -0.0001273104, 3.8040514e-05], [1.1641532e-10, 0.0, -3.0870851e-19]])], agents=[self.radius * np.random.rand(5, 1, 8) + self.radius / 2, self.radius * np.random.rand(5, 1, 8) + self.radius / 2])
        self.aug_feature_gt = {}
        self.aug_feature_gt['agents'] = Agents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.0120681393, -0.00109217957, 0.00104624288], [0.0268775601, -0.00105475327, 0.00400813782], [0.0512891984, -0.000897311768, 0.00889057227], [0.0852192154, -0.000480500022, 0.0156771013]])], agents=[self.radius * np.random.rand(5, 1, 8) + self.radius / 2])
        self.targets: Dict[str, Any] = {}
        augment_prob = 1.0
        dt = 0.1
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        self.gaussian_augmentor = KinematicHistoryAgentAugmentor(dt, mean, std, low, high, augment_prob, use_uniform_noise=False)
        self.uniform_augmentor = KinematicHistoryAgentAugmentor(dt, mean, std, low, high, augment_prob, use_uniform_noise=True)

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['agents'].ego[0] - self.aug_feature_gt['agents'].ego[0]) < 0.1).all())

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        original_feature_ego = self.features['agents'].ego[1].copy()
        aug_feature, _ = self.uniform_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['agents'].ego[1] - original_feature_ego) <= 0.1).all())

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.gaussian_augmentor._augment_prob = 0.0
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] == self.features['agents'].ego[0]).all())

class TestSimpleAgentAugmentation(unittest.TestCase):
    """Test agent augmentation that simply adds noise to the current ego position."""

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)
        self.features = {}
        self.features['agents'] = Agents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.004325964, -0.00069646863, -9.3163371e-06], [0.0024353617, -0.00037753209, 4.7789731e-06], [0.0011352128, -0.0001273104, 3.8040514e-05], [1.1641532e-10, 0.0, -3.0870851e-19]]), np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.004325964, -0.00069646863, -9.3163371e-06], [0.0024353617, -0.00037753209, 4.7789731e-06], [0.0011352128, -0.0001273104, 3.8040514e-05], [1.1641532e-10, 0.0, -3.0870851e-19]])], agents=[np.random.randn(5, 1, 8), np.random.randn(5, 1, 8)])
        self.aug_feature_gt = {}
        self.aug_feature_gt['agents'] = Agents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.004325964, -0.00069646863, -9.3163371e-06], [0.0024353617, -0.00037753209, 4.7789731e-06], [0.0011352128, -0.0001273104, 3.8040514e-05], [0.362865111, 0.0867895137, 0.429461646]])], agents=[np.array([[[-0.000527899086, -0.274901425, -0.139285562, 1.98468616, 0.282109326, 0.760808658, 0.300981606, 0.540297269]], [[0.373497287, 0.377813394, -0.0902131926, -2.30594327, 1.14276002, -1.53565429, -0.863752018, 1.01654494]], [[1.03396388, -0.824492228, 0.0189048564, -0.383343556, -0.304185475, 0.997291506, -0.127273841, -1.4758859]], [[-1.94090633, 0.833648924, -0.567217888, 1.17448696, 0.319068832, 0.190870428, 0.369270181, -0.101147863]], [[-0.941809489, -1.40414171, 2.08064701, -0.120316234, 0.759791879, 1.82743214, -0.660727087, -0.807806261]]])])
        self.targets: Dict[str, Any] = {}
        augment_prob = 1.0
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        self.gaussian_augmentor = SimpleAgentAugmentor(mean, std, low, high, augment_prob, use_uniform_noise=False)
        self.uniform_augmentor = SimpleAgentAugmentor(mean, std, low, high, augment_prob, use_uniform_noise=True)

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] - self.aug_feature_gt['agents'].ego[0] < 0.0001).all())

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        original_feature_ego = self.features['agents'].ego[1].copy()
        aug_feature, _ = self.uniform_augmentor.augment(self.features, self.targets)
        print(f'{original_feature_ego}, \n {aug_feature}')
        self.assertTrue((abs(aug_feature['agents'].ego[1] - original_feature_ego) <= 0.1).all())

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.gaussian_augmentor._augment_prob = 0.0
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] == self.features['agents'].ego[0]).all())

class TestKinematicAgentAugmentation(unittest.TestCase):
    """Test agent augmentation with kinematic constraints."""

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)
        self.features = {}
        self.features['agents'] = Agents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.004325964, -0.00069646863, -9.3163371e-06], [0.0024353617, -0.00037753209, 4.7789731e-06], [0.0011352128, -0.0001273104, 3.8040514e-05], [1.1641532e-10, 0.0, -3.0870851e-19]])], agents=[np.random.randn(5, 1, 8)])
        self.targets = {}
        self.targets['trajectory'] = Trajectory(data=np.array([[-0.0012336078, 0.0002229698, -2.075062e-05], [0.0032337871, 0.00035673147, -0.00011526359], [0.025042057, 0.00046393462, -0.00045901173], [0.24698858, -0.0015322007, -0.0013717031], [0.82662332, -0.0071887751, -0.0039011773], [1.7506398, -0.017746322, -0.0072191255], [3.0178127, -0.033933811, -0.0090915877], [4.5618219, -0.053034388, -0.0048586642], [6.3618584, -0.065912366, 0.00026488048], [8.3739414, -0.069805034, 0.0040571247], [10.576758, -0.044418037, 0.0074823718], [12.969443, -0.017768066, 0.0097025689]]))
        self.aug_feature_gt = {}
        self.aug_feature_gt['agents'] = Agents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.004325964, -0.00069646863, -9.3163371e-06], [0.0024353617, -0.00037753209, 4.7789731e-06], [0.0011352128, -0.0001273104, 3.8040514e-05], [0.36286512, 0.0867895111, 0.429461658]])], agents=[np.array([[[-0.000527899086, -0.274901425, -0.139285562, 1.98468616, 0.282109326, 0.760808658, 0.300981606, 0.540297269]], [[0.373497287, 0.377813394, -0.0902131926, -2.30594327, 1.14276002, -1.53565429, -0.863752018, 1.01654494]], [[1.03396388, -0.824492228, 0.0189048564, -0.383343556, -0.304185475, 0.997291506, -0.127273841, -1.4758859]], [[-1.94090633, 0.833648924, -0.567217888, 1.17448696, 0.319068832, 0.190870428, 0.369270181, -0.101147863]], [[-0.941809489, -1.40414171, 2.08064701, -0.120316234, 0.759791879, 1.82743214, -0.660727087, -0.807806261]]])])
        self.gaussian_aug_targets_gt = {}
        self.gaussian_aug_targets_gt['trajectory'] = Trajectory(data=np.array([[0.41521129, 0.11039978, 0.41797668], [0.5046286, 0.14907575, 0.39849171], [0.63200253, 0.2006533, 0.37100676], [0.79846221, 0.26203236, 0.33552179], [1.0052546, 0.3291364, 0.29203683], [1.2535783, 0.39687237, 0.24055186], [1.5443755, 0.45909974, 0.1810669], [1.8780817, 0.50862163, 0.11358193], [2.2541707, 0.53959757, 0.050773341], [2.6713488, 0.55327171, 0.014758691], [3.1287551, 0.55699998, 0.0015426531], [3.6260972, 0.55770481, 0.0012917991]]))
        self.uniform_aug_targets_gt = {}
        self.uniform_aug_targets_gt['trajectory'] = Trajectory(data=np.array([[0.05273135, -0.04831281, -0.08689969], [0.11795828, -0.05359042, -0.07457177], [0.22317114, -0.06049316, -0.05645524], [0.3684539, -0.06721046, -0.03595094], [0.553826, -0.07214818, -0.01731013], [0.77925223, -0.0745298, -0.00381898], [1.0446922, -0.07455366, 0.00363919], [1.3501287, -0.07300503, 0.00650118], [1.6955612, -0.07065626, 0.00709759], [2.080992, -0.06789713, 0.00721934], [2.5064206, -0.06473273, 0.00765666], [2.9717717, -0.06097136, 0.00850872]]))
        N = 12
        dt = 0.1
        augment_prob = 1.0
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        self.gaussian_augmentor = KinematicAgentAugmentor(N, dt, mean, std, low, high, augment_prob, use_uniform_noise=False)
        self.uniform_augmentor = KinematicAgentAugmentor(N, dt, mean, std, low, high, augment_prob, use_uniform_noise=True)

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, aug_targets = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] - self.aug_feature_gt['agents'].ego[0] < 0.0001).all())
        self.assertTrue((aug_targets['trajectory'].data - self.gaussian_aug_targets_gt['trajectory'].data < 0.0001).all())

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        features_ego = self.features['agents'].ego[0].copy()
        aug_feature, aug_targets = self.uniform_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['agents'].ego[0] - features_ego) <= 0.1).all())
        self.assertTrue((abs(aug_targets['trajectory'].data - self.uniform_aug_targets_gt['trajectory'].data) <= 0.1).all())

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.gaussian_augmentor._augment_prob = 0.0
        aug_feature, aug_targets = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] == self.features['agents'].ego[0]).all())
        self.assertTrue((aug_targets['trajectory'].data == self.targets['trajectory'].data).all())

    def test_input_validation(self) -> None:
        """
        Test the augmentor's validation check.
        """
        features = {'agents': None, 'test_feature': None}
        targets = {'trajectory': None, 'test_target': None}
        self.gaussian_augmentor.validate(features, targets)
        features = {'test_feature': None}
        targets = {'test_target': None}
        self.assertRaises(AssertionError, self.gaussian_augmentor.validate, features, targets)

class TestAgentDropoutAugmentation(unittest.TestCase):
    """Test agent augmentation that drops out random agents from the scene."""

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)
        self.features = {}
        self.features['agents'] = Agents(ego=[np.random.randn(5, 3), np.random.randn(5, 3)], agents=[np.random.randn(5, 20, 8), np.random.randn(5, 50, 8)])
        self.targets: Dict[str, Any] = {}
        augment_prob = 1.0
        self.dropout_rate = 0.5
        self.augmentor = AgentDropoutAugmentor(augment_prob, self.dropout_rate)

    def test_augment(self) -> None:
        """
        Test augmentation.
        """
        features = deepcopy(self.features)
        aug_features, _ = self.augmentor.augment(features, self.targets)
        for agents, aug_agents in zip(self.features['agents'].agents, aug_features['agents'].agents):
            self.assertLess(aug_agents.shape[1], agents.shape[1])

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.augmentor._augment_prob = 0.0
        aug_features, _ = self.augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_features['agents'].agents[0] == self.features['agents'].agents[0]).all())

class TestGaussianSmoothAgentAugmentation(unittest.TestCase):
    """Test agent augmentation with gaussian smooth noise."""

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)
        self.features = {}
        self.features['agents'] = Agents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.004325964, -0.00069646863, -9.3163371e-06], [0.0024353617, -0.00037753209, 4.7789731e-06], [0.0011352128, -0.0001273104, 3.8040514e-05], [1.1641532e-10, 0.0, -3.0870851e-19]])], agents=[np.random.randn(5, 1, 8)])
        self.targets = {}
        self.targets['trajectory'] = Trajectory(data=np.array([[-0.0012336078, 0.0002229698, -2.075062e-05], [0.0032337871, 0.00035673147, -0.00011526359], [0.025042057, 0.00046393462, -0.00045901173], [0.24698858, -0.0015322007, -0.0013717031], [0.82662332, -0.0071887751, -0.0039011773], [1.7506398, -0.017746322, -0.0072191255], [3.0178127, -0.033933811, -0.0090915877], [4.5618219, -0.053034388, -0.0048586642], [6.3618584, -0.065912366, 0.00026488048], [8.3739414, -0.069805034, 0.0040571247], [10.576758, -0.044418037, 0.0074823718], [12.969443, -0.017768066, 0.0097025689]]))
        self.aug_feature_gt = {}
        self.aug_feature_gt['agents'] = Agents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05], [0.004325964, -0.00069646863, -9.3163371e-06], [0.0024353617, -0.00037753209, 4.7789731e-06], [0.0011352128, -0.0001273104, 3.8040514e-05], [0.267742378, 0.0587639301, 0.305916953]])], agents=[np.array([[[-0.000527899086, -0.274901425, -0.139285562, 1.98468616, 0.282109326, 0.760808658, 0.300981606, 0.540297269]], [[0.373497287, 0.377813394, -0.0902131926, -2.30594327, 1.14276002, -1.53565429, -0.863752018, 1.01654494]], [[1.03396388, -0.824492228, 0.0189048564, -0.383343556, -0.304185475, 0.997291506, -0.127273841, -1.4758859]], [[-1.94090633, 0.833648924, -0.567217888, 1.17448696, 0.319068832, 0.190870428, 0.369270181, -0.101147863]], [[-0.941809489, -1.40414171, 2.08064701, -0.120316234, 0.759791879, 1.82743214, -0.660727087, -0.807806261]]])])
        self.gaussian_aug_targets_gt = {}
        self.gaussian_aug_targets_gt['trajectory'] = Trajectory(data=np.array([[0.179909768, 0.0346292143, 0.169823954], [0.10860577, 0.0197756017, 0.042442404], [0.0955989353, 0.00718025938, 0.0104373998], [0.32935287, -0.00092038409, 0.000919450476], [0.915527184, -0.00788396897, -0.00340438637], [1.84272996, -0.0188512783, -0.00617531861], [3.09345437, -0.0344966704, -0.0072624205], [4.62998953, -0.0514234703, -0.00448675278], [6.41906077, -0.0632653042, -1.90822629e-05], [8.4253027, -0.062546126, 0.00395074816], [10.5772538, -0.0442685352, 0.00714697555], [11.9537668, -0.0290942382, 0.00873650844]]))
        self.uniform_aug_targets_gt = {}
        self.uniform_aug_targets_gt['trajectory'] = Trajectory(data=np.array([[-0.0123269903, 0.00395750476, -0.00366945959], [0.0052339853, 0.00876677051, 0.00482984929], [0.0811338362, 0.00287675577, -0.00247428679], [0.341575812, -0.00256694967, -0.00236505408], [0.919201714, -0.00857337111, -0.00399194094], [1.84200781, -0.0191452871, -0.00689646769], [3.09258798, -0.0346222537, -0.00760822625], [4.62963856, -0.0514547445, -0.00456260133], [6.41901491, -0.0632771927, -4.26804288e-05], [8.42531047, -0.0625500536, 0.0039344597], [10.5772518, -0.0442706406, 0.00713952217], [11.9537637, -0.0290951136, 0.00873232027]]))
        augment_prob = 1.0
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        sigma = 5.0
        self.gaussian_augmentor = GaussianSmoothAgentAugmentor(mean, std, low, high, sigma, augment_prob, use_uniform_noise=False)
        self.uniform_augmentor = GaussianSmoothAgentAugmentor(mean, std, low, high, sigma, augment_prob, use_uniform_noise=True)

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, aug_targets = self.gaussian_augmentor.augment(self.features, self.targets)
        print(aug_feature, aug_targets)
        self.assertTrue((aug_feature['agents'].ego[0] - self.aug_feature_gt['agents'].ego[0] < 0.0001).all())
        self.assertTrue((aug_targets['trajectory'].data - self.gaussian_aug_targets_gt['trajectory'].data < 0.0001).all())

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        original_features_ego = self.features['agents'].ego[0].copy()
        aug_feature, aug_targets = self.uniform_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['agents'].ego[0] - original_features_ego) < 0.1).all())
        self.assertTrue((aug_targets['trajectory'].data - self.uniform_aug_targets_gt['trajectory'].data < 0.0001).all())

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.gaussian_augmentor._augment_prob = 0.0
        aug_feature, aug_targets = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] == self.features['agents'].ego[0]).all())
        self.assertTrue((aug_targets['trajectory'].data == self.targets['trajectory'].data).all())

def cache_scenarios_internal(args: List[Dict[str, Union[List[AbstractScenario], DictConfig]]]) -> List[CacheResult]:
    node_id = int(os.environ.get('NODE_RANK', 0))
    thread_id = str(uuid.uuid4())
    scenarios: List[AbstractScenario] = [a['scenario'] for a in args]
    cfg: DictConfig = args[0]['cfg']
    model = build_torch_module_wrapper(cfg.model)
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()
    del model
    assert cfg.cache.cache_path is not None, f'Cache path cannot be None when caching, got {cfg.cache.cache_path}'
    preprocessor = FeaturePreprocessor(cache_path=cfg.cache.cache_path, force_feature_computation=cfg.cache.force_feature_computation, feature_builders=feature_builders, target_builders=target_builders)
    logger.info('Extracted %s scenarios for thread_id=%s, node_id=%s.', str(len(scenarios)), thread_id, node_id)
    num_failures = 0
    num_successes = 0
    all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
    for idx, scenario in enumerate(scenarios):
        logger.info('Processing scenario %s / %s in thread_id=%s, node_id=%s', idx + 1, len(scenarios), thread_id, node_id)
        features, targets, file_cache_metadata = preprocessor.compute_features(scenario)
        scenario_num_failures = sum((0 if feature.is_valid else 1 for feature in itertools.chain(features.values(), targets.values())))
        scenario_num_successes = len(features.values()) + len(targets.values()) - scenario_num_failures
        num_failures += scenario_num_failures
        num_successes += scenario_num_successes
        all_file_cache_metadata += file_cache_metadata
    logger.info('Finished processing scenarios for thread_id=%s, node_id=%s', thread_id, node_id)
    return [CacheResult(failures=num_failures, successes=num_successes, cache_metadata=all_file_cache_metadata)]

def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """

    def cache_scenarios_internal(args: List[Dict[str, Union[List[AbstractScenario], DictConfig]]]) -> List[CacheResult]:
        node_id = int(os.environ.get('NODE_RANK', 0))
        thread_id = str(uuid.uuid4())
        scenarios: List[AbstractScenario] = [a['scenario'] for a in args]
        cfg: DictConfig = args[0]['cfg']
        model = build_torch_module_wrapper(cfg.model)
        feature_builders = model.get_list_of_required_feature()
        target_builders = model.get_list_of_computed_target()
        del model
        assert cfg.cache.cache_path is not None, f'Cache path cannot be None when caching, got {cfg.cache.cache_path}'
        preprocessor = FeaturePreprocessor(cache_path=cfg.cache.cache_path, force_feature_computation=cfg.cache.force_feature_computation, feature_builders=feature_builders, target_builders=target_builders)
        logger.info('Extracted %s scenarios for thread_id=%s, node_id=%s.', str(len(scenarios)), thread_id, node_id)
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        for idx, scenario in enumerate(scenarios):
            logger.info('Processing scenario %s / %s in thread_id=%s, node_id=%s', idx + 1, len(scenarios), thread_id, node_id)
            features, targets, file_cache_metadata = preprocessor.compute_features(scenario)
            scenario_num_failures = sum((0 if feature.is_valid else 1 for feature in itertools.chain(features.values(), targets.values())))
            scenario_num_successes = len(features.values()) + len(targets.values()) - scenario_num_failures
            num_failures += scenario_num_failures
            num_successes += scenario_num_successes
            all_file_cache_metadata += file_cache_metadata
        logger.info('Finished processing scenarios for thread_id=%s, node_id=%s', thread_id, node_id)
        return [CacheResult(failures=num_failures, successes=num_successes, cache_metadata=all_file_cache_metadata)]
    result = cache_scenarios_internal(args)
    gc.collect()
    return result

def create_dataset(samples: List[AbstractScenario], feature_preprocessor: FeaturePreprocessor, dataset_fraction: float, dataset_name: str, augmentors: Optional[List[AbstractAugmentor]]=None) -> torch.utils.data.Dataset:
    """
    Create a dataset from a list of samples.
    :param samples: List of dataset candidate samples.
    :param feature_preprocessor: Feature preprocessor object.
    :param dataset_fraction: Fraction of the dataset to load.
    :param dataset_name: Set name (train/val/test).
    :param scenario_type_loss_weights: Dictionary of scenario type loss weights.
    :param augmentors: List of augmentor objects for providing data augmentation to data samples.
    :return: The instantiated torch dataset.
    """
    num_keep = int(len(samples) * dataset_fraction)
    selected_scenarios = random.sample(samples, num_keep)
    logger.info(f'Number of samples in {dataset_name} set: {len(selected_scenarios)}')
    return ScenarioDataset(scenarios=selected_scenarios, feature_preprocessor=feature_preprocessor, augmentors=augmentors)

class DataModule(pl.LightningDataModule):
    """
    Datamodule wrapping all preparation and dataset creation functionality.
    """

    def __init__(self, feature_preprocessor: FeaturePreprocessor, splitter: AbstractSplitter, all_scenarios: List[AbstractScenario], train_fraction: float, val_fraction: float, test_fraction: float, dataloader_params: Dict[str, Any], scenario_type_sampling_weights: DictConfig, worker: WorkerPool, augmentors: Optional[List[AbstractAugmentor]]=None) -> None:
        """
        Initialize the class.
        :param feature_preprocessor: Feature preprocessor object.
        :param splitter: Splitter object used to retrieve lists of samples to construct train/val/test sets.
        :param train_fraction: Fraction of training examples to load.
        :param val_fraction: Fraction of validation examples to load.
        :param test_fraction: Fraction of test examples to load.
        :param dataloader_params: Parameter dictionary passed to the dataloaders.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__()
        assert train_fraction > 0.0, 'Train fraction has to be larger than 0!'
        assert val_fraction > 0.0, 'Validation fraction has to be larger than 0!'
        assert test_fraction >= 0.0, 'Test fraction has to be larger/equal than 0!'
        self._train_set: Optional[torch.utils.data.Dataset] = None
        self._val_set: Optional[torch.utils.data.Dataset] = None
        self._test_set: Optional[torch.utils.data.Dataset] = None
        self._feature_preprocessor = feature_preprocessor
        self._splitter = splitter
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction
        self._dataloader_params = dataloader_params
        self._all_samples = all_scenarios
        assert len(self._all_samples) > 0, 'No samples were passed to the datamodule'
        self._scenario_type_sampling_weights = scenario_type_sampling_weights
        self._augmentors = augmentors
        self._worker = worker

    @property
    def feature_and_targets_builder(self) -> FeaturePreprocessor:
        """Get feature and target builders."""
        return self._feature_preprocessor

    def setup(self, stage: Optional[str]=None) -> None:
        """
        Set up the dataset for each target set depending on the training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        if stage is None:
            return
        if stage == 'fit':
            train_samples = self._splitter.get_train_samples(self._all_samples, self._worker)
            assert len(train_samples) > 0, 'Splitter returned no training samples'
            self._train_set = create_dataset(train_samples, self._feature_preprocessor, self._train_fraction, 'train', self._augmentors)
            val_samples = self._splitter.get_val_samples(self._all_samples, self._worker)
            assert len(val_samples) > 0, 'Splitter returned no validation samples'
            self._val_set = create_dataset(val_samples, self._feature_preprocessor, self._val_fraction, 'validation')
        elif stage == 'test':
            test_samples = self._splitter.get_test_samples(self._all_samples, self._worker)
            assert len(test_samples) > 0, 'Splitter returned no test samples'
            self._test_set = create_dataset(test_samples, self._feature_preprocessor, self._test_fraction, 'test')
        else:
            raise ValueError(f'Stage must be one of ["fit", "test"], got ${stage}.')

    def teardown(self, stage: Optional[str]=None) -> None:
        """
        Clean up after a training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the training dataloader.
        :raises RuntimeError: If this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._train_set is None:
            raise DataModuleNotSetupError
        if self._scenario_type_sampling_weights.enable:
            weighted_sampler = distributed_weighted_sampler_init(scenario_dataset=self._train_set, scenario_sampling_weights=self._scenario_type_sampling_weights.scenario_type_weights)
        else:
            weighted_sampler = None
        return torch.utils.data.DataLoader(dataset=self._train_set, shuffle=weighted_sampler is None, collate_fn=FeatureCollate(), sampler=weighted_sampler, **self._dataloader_params)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the validation dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._val_set is None:
            raise DataModuleNotSetupError
        return torch.utils.data.DataLoader(dataset=self._val_set, **self._dataloader_params, collate_fn=FeatureCollate())

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the test dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._test_set is None:
            raise DataModuleNotSetupError
        return torch.utils.data.DataLoader(dataset=self._test_set, **self._dataloader_params, collate_fn=FeatureCollate())

    def transfer_batch_to_device(self, batch: Tuple[FeaturesType, ...], device: torch.device) -> Tuple[FeaturesType, ...]:
        """
        Transfer a batch to device.
        :param batch: Batch on origin device.
        :param device: Desired device.
        :return: Batch in new device.
        """
        return tuple((move_features_type_to_device(batch[0], device), move_features_type_to_device(batch[1], device), batch[2]))

class SkeletonTestDataloader(unittest.TestCase):
    """
    Skeleton with initialized dataloader used in testing.
    """

    def setUp(self) -> None:
        """
        Set up basic configs.
        """
        pl.seed_everything(2022, workers=True)
        self.splitter = LogSplitter(log_splits={'train': ['2021.07.16.20.45.29_veh-35_01095_01486'], 'val': ['2021.06.07.18.53.26_veh-26_00005_00427'], 'test': ['2021.10.06.07.26.10_veh-52_00006_00398']})
        feature_builders = [DummyVectorMapBuilder(), VectorMapFeatureBuilder(radius=20), AgentsFeatureBuilder(TrajectorySampling(num_poses=4, time_horizon=1.5)), RasterFeatureBuilder(map_features={'LANE': 1, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5}, num_input_channels=4, target_width=224, target_height=224, target_pixel_size=0.5, ego_width=2.297, ego_front_length=4.049, ego_rear_length=1.127, ego_longitudinal_offset=0.0, baseline_path_thickness=1)]
        target_builders = [EgoTrajectoryTargetBuilder(TrajectorySampling(num_poses=10, time_horizon=5.0))]
        self.feature_preprocessor = FeaturePreprocessor(cache_path=None, force_feature_computation=True, feature_builders=feature_builders, target_builders=target_builders)
        self.scenario_filter = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None, num_scenarios_per_type=None, limit_total_scenarios=150, expand_scenarios=True, remove_invalid_goals=False, shuffle=True, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
        self.augmentors = [KinematicAgentAugmentor(trajectory_length=10, dt=0.1, mean=[0.3, 0.1, np.pi / 12], std=[0.5, 0.1, np.pi / 12], low=[-0.2, 0.0, 0.0], high=[0.8, 0.2, np.pi / 6], augment_prob=0.5)]
        self.scenario_builder = get_test_nuplan_scenario_builder()

    def _test_dataloader(self, worker: WorkerPool) -> None:
        """
        Tests that the training dataloader can be iterated without errors
        """
        scenarios = self.scenario_builder.get_scenarios(self.scenario_filter, worker)
        self.assertGreater(len(scenarios), 0)
        batch_size = 4
        num_workers = 4
        scenario_type_sampling_weights = DictConfig({'enable': False, 'scenario_type_weights': {'unknown': 1.0}})
        datamodule = DataModule(feature_preprocessor=self.feature_preprocessor, splitter=self.splitter, train_fraction=1.0, val_fraction=0.1, test_fraction=0.1, all_scenarios=scenarios, augmentors=self.augmentors, worker=worker, scenario_type_sampling_weights=scenario_type_sampling_weights, dataloader_params={'batch_size': batch_size, 'num_workers': num_workers, 'drop_last': True})
        datamodule.setup('fit')
        self.assertGreater(len(datamodule.train_dataloader()), 0)
        for features, targets, scenarios in datamodule.train_dataloader():
            self.assertTrue('raster' in features.keys())
            self.assertTrue('vector_map' in features.keys())
            self.assertTrue('trajectory' in targets.keys())
            scenario_features: Raster = features['raster']
            trajectory_target: Trajectory = targets['trajectory']
            self.assertEqual(scenario_features.num_batches, trajectory_target.num_batches)
            self.assertIsInstance(scenario_features, Raster)
            self.assertIsInstance(trajectory_target, Trajectory)
            self.assertEqual(scenario_features.num_batches, batch_size)

    def tearDown(self) -> None:
        """
        Clean up.
        """
        if ray.is_initialized():
            ray.shutdown()

class TestVisualizationUtils(unittest.TestCase):
    """Unit tests for visualization utlities."""

    def test_raster_visualization(self) -> None:
        """
        Test raster visualization utils.
        """
        trajectory_1 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
        trajectory_2 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]))
        size = 224
        raster = Raster(data=np.zeros((1, size, size, 4)))
        image = get_raster_with_trajectories_as_rgb(raster, trajectory_1, trajectory_2)
        self.assertEqual(image.shape, (size, size, 3))
        self.assertTrue(np.any(image))

    def test_vector_map_agents_visualization(self) -> None:
        """
        Test vector map and agents visualization utils.
        """
        trajectory_1 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
        trajectory_2 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]))
        pixel_size = 0.5
        radius = 50.0
        size = int(2 * radius / pixel_size)
        vector_map = VectorMap(coords=[np.zeros((1000, 2, 2))], lane_groupings=[[]], multi_scale_connections=[{}], on_route_status=[np.zeros((1000, 2))], traffic_light_data=[np.zeros((1000, 4))])
        agents = Agents(ego=[np.array(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))], agents=[np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])])
        image = get_raster_from_vector_map_with_agents(vector_map, agents, trajectory_1, trajectory_2, pixel_size=pixel_size, radius=radius)
        self.assertEqual(image.shape, (size, size, 3))
        self.assertTrue(np.any(image))

    def test_vector_set_map_generic_agents_visualization(self) -> None:
        """
        Test vector set map and generic agents visualization utils.
        """
        trajectory_1 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
        trajectory_2 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]))
        pixel_size = 0.5
        radius = 50.0
        size = int(2 * radius / pixel_size)
        agent_features = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE']
        vector_set_map = VectorSetMap(coords={VectorFeatureLayer.LANE.name: [np.zeros((100, 100, 2))]}, traffic_light_data={}, availabilities={VectorFeatureLayer.LANE.name: [np.ones((100, 100), dtype=bool)]})
        agents = GenericAgents(ego=[np.array(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))], agents={feature_name: [np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])] for feature_name in agent_features})
        image = get_raster_from_vector_map_with_agents(vector_set_map, agents, trajectory_1, trajectory_2, pixel_size=pixel_size, radius=radius)
        self.assertEqual(image.shape, (size, size, 3))
        self.assertTrue(np.any(image))

class TestCollateDataLoader(unittest.TestCase):
    """
    Tests data loading functionality
    """

    def setUp(self) -> None:
        """Set up the test case."""
        self.batch_size = 4
        feature_preprocessor = FeaturePreprocessor(cache_path=None, feature_builders=[RasterFeatureBuilder(map_features={'LANE': 1.0, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5}, num_input_channels=4, target_width=224, target_height=224, target_pixel_size=0.5, ego_width=2.297, ego_front_length=4.049, ego_rear_length=1.127, ego_longitudinal_offset=0.0, baseline_path_thickness=1), VectorMapFeatureBuilder(radius=20)], target_builders=[EgoTrajectoryTargetBuilder(TrajectorySampling(time_horizon=6.0, num_poses=12))], force_feature_computation=False)
        scenario = get_test_nuplan_scenario()
        scenarios = [scenario] * 3
        dataset = ScenarioDataset(scenarios=scenarios, feature_preprocessor=feature_preprocessor)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=2, pin_memory=False, drop_last=True, collate_fn=FeatureCollate())

    def test_dataloader(self) -> None:
        """
        Tests that the training dataloader can be iterated without errors
        """
        dataloader = self.dataloader
        dataloader_iter = iter(dataloader)
        iterations = min(len(dataloader), NUM_BATCHES)
        for _ in range(iterations):
            features, targets, scenarios = next(dataloader_iter)
            self.assertTrue('vector_map' in features.keys())
            vector_map: VectorMap = features['vector_map']
            self.assertEqual(vector_map.num_of_batches, self.batch_size)
            self.assertEqual(len(vector_map.coords), self.batch_size)
            self.assertEqual(len(vector_map.multi_scale_connections), self.batch_size)

class TestFeaturePreprocessor(unittest.TestCase):
    """Tests preprocessing and caching functionality during training."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.cache_path = pathlib.Path('/tmp/test')

    def test_sample(self) -> None:
        """
        Test computation of a features for sample
        """
        raster_feature_builder = RasterFeatureBuilder(map_features={'LANE': 1.0, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5}, num_input_channels=4, target_width=224, target_height=224, target_pixel_size=0.5, ego_width=2.297, ego_front_length=4.049, ego_rear_length=1.127, ego_longitudinal_offset=0.0, baseline_path_thickness=1)
        vectormap_builder = VectorMapFeatureBuilder(radius=20)
        ego_trajectory_target_builder = EgoTrajectoryTargetBuilder(TrajectorySampling(num_poses=10, time_horizon=5.0))
        logging.basicConfig(level=logging.INFO)
        feature_preprocessor = FeaturePreprocessor(cache_path=str(self.cache_path), feature_builders=[raster_feature_builder, vectormap_builder], target_builders=[ego_trajectory_target_builder], force_feature_computation=False)
        scenario = get_test_nuplan_scenario()
        self._compute_features_and_check_builders(scenario, feature_preprocessor, 2, 1)

    def _compute_features_and_check_builders(self, sample: Any, feature_preprocessor: FeaturePreprocessor, number_of_features: int, number_of_targets: int) -> None:
        """
        :param sample: Input data sample to compute features/targets from.
        :param feature_preprocessor: Preprocessor object with caching mechanism.
        :param number_of_features: Number of expected features.
        :param number_of_targets: Number of expected targets.
        """
        features, targets, _ = feature_preprocessor.compute_features(sample)
        self.assertEqual(len(targets), number_of_targets)
        self.assertEqual(len(features), number_of_features)
        for builder in feature_preprocessor.feature_builders:
            self.assertTrue(builder.get_feature_unique_name() in features.keys())
            feature = features[builder.get_feature_unique_name()]
            self.assertIsInstance(feature, builder.get_feature_type())
            self.assertTrue(feature.is_valid)
        for builder in feature_preprocessor.target_builders:
            self.assertTrue(builder.get_feature_unique_name() in targets.keys())
            target = targets[builder.get_feature_unique_name()]
            self.assertIsInstance(target, builder.get_feature_type())
            self.assertTrue(target.is_valid)

class TestUtilsCache(unittest.TestCase):
    """Test caching utilities."""

    def setUp(self) -> None:
        """Set up test case."""
        local_cache_path = '/tmp/cache'
        s3_cache_path = 's3://tmp/cache'
        self.cache_paths = [local_cache_path, s3_cache_path]
        local_store = FeatureCachePickle()
        s3_store = FeatureCacheS3(s3_cache_path)
        s3_store._store = MockS3Store()
        self.cache_engines = [local_store, s3_store]

    def test_storing_to_cache_vector_map(self) -> None:
        """
        Test storing feature to cache
        """
        dim = 50
        feature = VectorMap(coords=[np.zeros((dim, 2, 2)).astype(np.float32)], lane_groupings=[[np.zeros(dim).astype(np.float32)]], multi_scale_connections=[{1: np.zeros((dim, 2)).astype(np.float32)}], on_route_status=[np.zeros((dim, 2)).astype(np.float32)], traffic_light_data=[np.zeros((dim, 4)).astype(np.float32)])
        for cache_path, cache in zip(self.cache_paths, self.cache_engines):
            folder = pathlib.Path(cache_path) / 'tmp_log_name' / 'tmp_scenario_token' / 'vector_map'
            if not str(folder).startswith('s3:/'):
                folder.parent.mkdir(parents=True, exist_ok=True)
            time_now = time.time()
            loaded_feature: VectorMap = self.store_and_load(cache, folder, feature)
            time_later = time.time()
            logger.debug(f'Cache: {type(cache)} = {time_later - time_now}')
            self.assertEqual(feature.num_of_batches, loaded_feature.num_of_batches)
            self.assertEqual(1, loaded_feature.num_of_batches)
            self.assertEqual(feature.coords[0].shape, loaded_feature.coords[0].shape)
            self.assertEqual(feature.lane_groupings[0][0].shape, loaded_feature.lane_groupings[0][0].shape)
            self.assertEqual(feature.multi_scale_connections[0][1].shape, loaded_feature.multi_scale_connections[0][1].shape)

    def test_storing_to_cache_raster(self) -> None:
        """
        Test storing feature to cache
        """
        feature = Raster(data=np.zeros((244, 244, 3)))
        for cache_path, cache in zip(self.cache_paths, self.cache_engines):
            folder = pathlib.Path(cache_path) / 'tmp_log_name' / 'tmp_scenario_token' / 'raster'
            if not str(folder).startswith('s3:/'):
                folder.parent.mkdir(parents=True, exist_ok=True)
            loaded_feature = self.store_and_load(cache, folder, feature)
            self.assertEqual(feature.data.shape, loaded_feature.data.shape)

    def store_and_load(self, cache: FeatureCache, folder: pathlib.Path, feature: AbstractModelFeature) -> AbstractModelFeature:
        """
        Store feature and load it back.
        :param cache: Caching mechanism to use.
        :param folder: Folder to store feature.
        :param feature: Feature to store.
        :return: Loaded feature.
        """
        time_now = time.time()
        cache.store_computed_feature_to_folder(folder, feature)
        logger.debug(f'store_computed_feature_to_folder: {type(cache)} = {time.time() - time_now}')
        time_now = time.time()
        out = cache.load_computed_feature_from_folder(folder, feature)
        logger.debug(f'load_computed_feature_from_folder: {type(cache)} = {time.time() - time_now}')
        self.assertIsInstance(out, type(feature))
        return out

class VectorMapFeatureBuilder(ScriptableFeatureBuilder):
    """
    Feature builder for constructing map features in a vector-representation.
    """

    def __init__(self, radius: float, connection_scales: Optional[List[int]]=None) -> None:
        """
        Initialize vector map builder with configuration parameters.
        :param radius:  The query radius scope relative to the current ego-pose.
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        super().__init__()
        self._radius = radius
        self._connection_scales = connection_scales

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return VectorMap

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'vector_map'

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            lane_seg_coords, lane_seg_conns, lane_seg_groupings, lane_seg_lane_ids, lane_seg_roadblock_ids = get_neighbor_vector_map(scenario.map_api, ego_coords, self._radius)
            on_route_status = get_on_route_status(scenario.get_route_roadblock_ids(), lane_seg_roadblock_ids)
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(lane_seg_coords, lane_seg_conns, lane_seg_groupings, on_route_status, traffic_light_data, ego_state.rear_axle)
            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            ego_state = current_input.history.ego_states[-1]
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            lane_seg_coords, lane_seg_conns, lane_seg_groupings, lane_seg_lane_ids, lane_seg_roadblock_ids = get_neighbor_vector_map(initialization.map_api, ego_coords, self._radius)
            on_route_status = get_on_route_status(initialization.route_roadblock_ids, lane_seg_roadblock_ids)
            if current_input.traffic_light_data is None:
                raise ValueError('Cannot build VectorMap feature. PlannerInput.traffic_light_data is None')
            traffic_light_data = current_input.traffic_light_data
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(lane_seg_coords, lane_seg_conns, lane_seg_groupings, on_route_status, traffic_light_data, ego_state.rear_axle)
            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.ignore
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> VectorMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorMap.
        """
        multi_scale_connections: Dict[int, torch.Tensor] = {}
        for key in list_tensor_data:
            if key.startswith('vector_map.multi_scale_connections_'):
                multi_scale_connections[int(key[len('vector_map.multi_scale_connections_'):])] = list_tensor_data[key][0].detach().numpy()
        lane_groupings = [t.detach().numpy() for t in list_list_tensor_data['vector_map.lane_groupings'][0]]
        return VectorMap(coords=[list_tensor_data['vector_map.coords'][0].detach().numpy()], lane_groupings=[lane_groupings], multi_scale_connections=[multi_scale_connections], on_route_status=[list_tensor_data['vector_map.on_route_status'][0].detach().numpy()], traffic_light_data=[list_tensor_data['vector_map.traffic_light_data'][0].detach().numpy()])

    @torch.jit.ignore
    def _pack_to_feature_tensor_dict(self, lane_coords: LaneSegmentCoords, lane_conns: LaneSegmentConnections, lane_groupings: LaneSegmentGroupings, lane_on_route_status: LaneOnRouteStatusData, traffic_light_data: LaneSegmentTrafficLightData, anchor_state: StateSE2) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature tranform.
        :param lane_coords: The LaneSegmentCoords returned from `get_neighbor_vector_map` to transform.
        :param lane_conns: The LaneSegmentConnections returned from `get_neighbor_vector_map` to transform.
        :param lane_groupings: The LaneSegmentGroupings returned from `get_neighbor_vector_map` to transform.
        :param lane_on_route_status: The LaneOnRouteStatusData returned from `get_neighbor_vector_map` to transform.
        :param traffic_light_data: The LaneSegmentTrafficLightData returned from `get_neighbor_vector_map` to transform.
        :param anchor_state: The ego state to transform to vector.
        """
        lane_segment_coords: torch.tensor = torch.tensor(lane_coords.to_vector(), dtype=torch.float64)
        lane_segment_conns: torch.tensor = torch.tensor(lane_conns.to_vector(), dtype=torch.int64)
        on_route_status: torch.tensor = torch.tensor(lane_on_route_status.to_vector(), dtype=torch.float32)
        traffic_light_array: torch.tensor = torch.tensor(traffic_light_data.to_vector(), dtype=torch.float32)
        lane_segment_groupings: List[torch.tensor] = []
        for lane_grouping in lane_groupings.to_vector():
            lane_segment_groupings.append(torch.tensor(lane_grouping, dtype=torch.int64))
        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)
        return ({'lane_segment_coords': lane_segment_coords, 'lane_segment_conns': lane_segment_conns, 'on_route_status': on_route_status, 'traffic_light_array': traffic_light_array, 'anchor_state': anchor_state_tensor}, {'lane_segment_groupings': lane_segment_groupings}, {})

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        lane_segment_coords = tensor_data['lane_segment_coords']
        anchor_state = tensor_data['anchor_state']
        lane_segment_conns = tensor_data['lane_segment_conns']
        if len(lane_segment_conns.shape) == 1:
            if lane_segment_conns.shape[0] == 0:
                lane_segment_conns = torch.zeros((0, 2), device=lane_segment_coords.device, layout=lane_segment_coords.layout, dtype=torch.int64)
            else:
                raise ValueError(f'Unexpected shape for lane_segment_conns: {lane_segment_conns.shape}')
        lane_segment_coords = lane_segment_coords.reshape(-1, 2)
        lane_segment_coords = coordinates_to_local_frame(lane_segment_coords, anchor_state, precision=torch.float64)
        lane_segment_coords = lane_segment_coords.reshape(-1, 2, 2).float()
        if self._connection_scales is not None:
            multi_scale_connections = _generate_multi_scale_connections(lane_segment_conns, self._connection_scales)
        else:
            multi_scale_connections = {1: lane_segment_conns}
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {'vector_map.lane_groupings': [list_tensor_data['lane_segment_groupings']]}
        list_tensor_output: Dict[str, List[torch.Tensor]] = {'vector_map.coords': [lane_segment_coords], 'vector_map.on_route_status': [tensor_data['on_route_status']], 'vector_map.traffic_light_data': [tensor_data['traffic_light_array']]}
        for key in multi_scale_connections:
            list_tensor_output[f'vector_map.multi_scale_connections_{key}'] = [multi_scale_connections[key]]
        tensor_output: Dict[str, torch.Tensor] = {}
        return (tensor_output, list_tensor_output, list_list_tensor_output)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        return {'neighbor_vector_map': {'radius': str(self._radius)}, 'initial_ego_state': empty}

class AgentsFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, trajectory_sampling: TrajectorySampling, object_type: TrackedObjectType=TrackedObjectType.VEHICLE) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        :param object_type: Type of agents (TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN) set to TrackedObjectType.VEHICLE by default
        """
        super().__init__()
        if object_type not in [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]:
            raise ValueError(f"The model's been tested just for vehicles and pedestrians types, but the provided object_type is {object_type}.")
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon
        self.object_type = object_type
        self._agents_states_dim = Agents.agents_states_dim()

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'agents'

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Agents

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> Agents:
        """Inherited, see superclass."""
        with torch.no_grad():
            anchor_ego_state = scenario.initial_ego_state
            past_ego_states = scenario.get_ego_past_trajectory(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)
            sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
            time_stamps = list(scenario.get_past_timestamps(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)) + [scenario.start_time]
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [tracked_objects.tracked_objects for tracked_objects in scenario.get_past_tracked_objects(iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses)]
            sampled_past_observations = past_tracked_objects + [present_tracked_objects]
            assert len(sampled_past_ego_states) == len(sampled_past_observations), f'Expected the trajectory length of ego and agent to be equal. Got ego: {len(sampled_past_ego_states)} and agent: {len(sampled_past_observations)}'
            assert len(sampled_past_observations) > 2, f'Trajectory of length of {len(sampled_past_observations)} needs to be at least 3'
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: Agents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> Agents:
        """Inherited, see superclass."""
        with torch.no_grad():
            history = current_input.history
            assert isinstance(history.observations[0], DetectionsTracks), f'Expected observation of type DetectionTracks, got {type(history.observations[0])}'
            present_ego_state, present_observation = history.current_state
            past_observations = history.observations[:-1]
            past_ego_states = history.ego_states[:-1]
            assert history.sample_interval, 'SimulationHistoryBuffer sample interval is None'
            indices = sample_indices_with_time_horizon(self.num_past_poses, self.past_time_horizon, history.sample_interval)
            try:
                sampled_past_observations = [cast(DetectionsTracks, past_observations[-idx]).tracked_objects for idx in reversed(indices)]
                sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
            except IndexError:
                raise RuntimeError(f'SimulationHistoryBuffer duration: {history.duration} is too short for requested past_time_horizon: {self.past_time_horizon}. Please increase the simulation_buffer_duration in default_simulation.yaml')
            sampled_past_observations = sampled_past_observations + [cast(DetectionsTracks, present_observation).tracked_objects]
            sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]
            time_stamps = [state.time_point for state in sampled_past_ego_states]
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: Agents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(self, past_ego_states: List[EgoState], past_time_stamps: List[TimePoint], past_tracked_objects: List[TrackedObjects]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param past_ego_states: The past states of the ego vehicle.
        :param past_time_stamps: The past time stamps of the input data.
        :param past_tracked_objects: The past tracked objects.
        :return: The packed tensors.
        """
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(past_ego_states)
        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)
        past_tracked_objects_tensor_list = sampled_tracked_objects_to_tensor_list(past_tracked_objects=past_tracked_objects, object_type=self.object_type)
        return ({'past_ego_states': past_ego_states_tensor, 'past_time_stamps': past_time_stamps_tensor}, {'past_tracked_objects': past_tracked_objects_tensor_list}, {})

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Agents:
        """
        Unpacks the data returned from the scriptable core into an Agents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed Agents object.
        """
        ego_features = [list_tensor_data['agents.ego'][0].detach().numpy()]
        agent_features = [list_tensor_data['agents.agents'][0].detach().numpy()]
        return Agents(ego=ego_features, agents=agent_features)

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        ego_history: torch.Tensor = tensor_data['past_ego_states']
        time_stamps: torch.Tensor = tensor_data['past_time_stamps']
        agents: List[torch.Tensor] = list_tensor_data['past_tracked_objects']
        anchor_ego_state = ego_history[-1, :].squeeze()
        agent_history = filter_agents_tensor(agents, reverse=True)
        if agent_history[-1].shape[0] == 0:
            agents_tensor: torch.Tensor = torch.zeros((len(agent_history), 0, self._agents_states_dim)).float()
        else:
            padded_agent_states = pad_agent_states(agent_history, reverse=True)
            local_coords_agent_states = convert_absolute_quantities_to_relative(padded_agent_states, anchor_ego_state)
            yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
            agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)
        ego_tensor = build_ego_features_from_tensor(ego_history, reverse=True)
        output_dict: Dict[str, torch.Tensor] = {}
        output_list_dict: Dict[str, List[torch.Tensor]] = {'agents.ego': [ego_tensor], 'agents.agents': [agents_tensor]}
        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}
        return (output_dict, output_list_dict, output_list_list_dict)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {'past_ego_states': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_time_stamps': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_tracked_objects': {'iteration': '0', 'time_horizon': str(self.past_time_horizon), 'num_samples': str(self.num_past_poses)}}

class TestAgentsFeatureBuilder(unittest.TestCase):
    """Test builder that constructs agent features during training and simulation."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.batch_size = 1
        self.past_time_horizon = 4.0
        self.num_past_poses = 4
        self.num_total_past_poses = self.num_past_poses + 1

    @given(number_of_detections=st.sampled_from([0, 10]), feature_builder_type=st.sampled_from([TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]))
    def test_agent_feature_builder(self, number_of_detections: int, feature_builder_type: TrackedObjectType) -> None:
        """
        Test AgentFeatureBuilder with and without agents in the scene for both pedestrian and vehicles
        """
        feature_builder = AgentsFeatureBuilder(TrajectorySampling(self.num_past_poses, self.past_time_horizon), feature_builder_type)
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=number_of_detections, tracked_object_types=[TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN])
        feature = feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), Agents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), Agents.ego_state_dim())
        self.assertEqual(len(feature.agents), self.batch_size)
        self.assertEqual(len(feature.agents[0]), self.num_total_past_poses)
        self.assertEqual(feature.agents[0].shape[1], number_of_detections)
        self.assertEqual(feature.agents[0].shape[2], Agents.agents_states_dim())

    @given(feature_builder_type=st.sampled_from([TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]))
    def test_get_feature_from_simulation(self, feature_builder_type: TrackedObjectType) -> None:
        """
        Test get feature from simulation
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=10, tracked_object_types=[TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN])
        mock_meta_data = PlannerInitialization(map_api=MockAbstractMap(), route_roadblock_ids=None, mission_goal=StateSE2(0, 0, 0))
        ego_past_states = list(scenario.get_ego_past_trajectory(iteration=0, num_samples=10, time_horizon=5))
        ego_initial_state = scenario.initial_ego_state
        ego_history = ego_past_states + [ego_initial_state]
        past_observations = list(scenario.get_past_tracked_objects(iteration=0, num_samples=10, time_horizon=5))
        initial_observation = scenario.initial_tracked_objects
        observation_history = past_observations + [initial_observation]
        history = SimulationHistoryBuffer.initialize_from_list(len(ego_history), ego_history, observation_history, scenario.database_interval)
        current_input = PlannerInput(iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history, traffic_light_data=scenario.get_traffic_light_status_at_iteration(0))
        feature_builder = AgentsFeatureBuilder(TrajectorySampling(self.num_past_poses, self.past_time_horizon), feature_builder_type)
        feature = feature_builder.get_features_from_simulation(current_input=current_input, initialization=mock_meta_data)
        self.assertEqual(type(feature), Agents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), Agents.ego_state_dim())
        self.assertEqual(len(feature.agents), self.batch_size)
        self.assertEqual(len(feature.agents[0]), self.num_total_past_poses)
        self.assertEqual(feature.agents[0].shape[1], 10)
        self.assertEqual(feature.agents[0].shape[2], Agents.agents_states_dim())

    @given(feature_builder_type=st.sampled_from([TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]))
    @settings(deadline=1000)
    def test_agents_feature_builder_scripts_properly(self, feature_builder_type: TrackedObjectType) -> None:
        """
        Tests that the Agents Feature Builder scripts properly
        """
        feature_builder = AgentsFeatureBuilder(TrajectorySampling(self.num_past_poses, self.past_time_horizon), feature_builder_type)
        config = feature_builder.precomputed_feature_config()
        for expected_key in ['past_ego_states', 'past_time_stamps', 'past_tracked_objects']:
            self.assertIn(expected_key, config)
            config_dict = config[expected_key]
            self.assertEqual(len(config_dict), 3)
            self.assertEqual(0, int(config_dict['iteration']))
            self.assertEqual(self.num_past_poses, int(config_dict['num_samples']))
            self.assertEqual(self.past_time_horizon, int(float(config_dict['time_horizon'])))
        num_frames = 5
        num_agents = 3
        ego_dim = EgoInternalIndex.dim()
        agent_dim = AgentInternalIndex.dim()
        past_ego_states = torch.zeros((num_frames, ego_dim), dtype=torch.float32)
        past_timestamps = torch.tensor([i * 50 for i in range(num_frames)], dtype=torch.int64)
        past_tracked_objects = [torch.ones((num_agents, agent_dim), dtype=torch.float32) for _ in range(num_frames)]
        for i in range(num_frames):
            for j in range(num_agents):
                past_tracked_objects[i][j, :] *= j + 1
        tensor_data = {'past_ego_states': past_ego_states, 'past_time_stamps': past_timestamps}
        list_tensor_data = {'past_tracked_objects': past_tracked_objects}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_builder = torch.jit.script(feature_builder)
        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        py_tensors, py_list_tensors, py_list_list_tensors = feature_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        self.assertEqual(0, len(scripted_tensors))
        self.assertEqual(0, len(py_tensors))
        self.assertEqual(len(scripted_list_tensors), len(py_list_tensors))
        for key in py_list_tensors:
            scripted_list = scripted_list_tensors[key]
            py_list = py_list_tensors[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                scripted = scripted_list[i]
                py = py_list[i]
                torch.testing.assert_allclose(py, scripted, atol=0.01, rtol=0.01)
        self.assertEqual(0, len(scripted_list_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))

class TestVectorMapFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs map features in vectorized format."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = get_test_nuplan_scenario()

    def test_vector_map_feature_builder(self) -> None:
        """
        Test VectorMapFeatureBuilder
        """
        feature_builder = VectorMapFeatureBuilder(radius=20, connection_scales=[2])
        self.assertEqual(feature_builder.get_feature_type(), VectorMap)
        features = feature_builder.get_features_from_scenario(self.scenario)
        self.assertEqual(type(features), VectorMap)
        ego_state = self.scenario.initial_ego_state
        detections = self.scenario.initial_tracked_objects
        meta_data = PlannerInitialization(map_api=self.scenario.map_api, mission_goal=self.scenario.get_mission_goal(), route_roadblock_ids=self.scenario.get_route_roadblock_ids())
        history = SimulationHistoryBuffer.initialize_from_list(1, [ego_state], [detections], self.scenario.database_interval)
        iteration = SimulationIteration(TimePoint(0), 0)
        tl_data = self.scenario.get_traffic_light_status_at_iteration(iteration.index)
        current_input = PlannerInput(iteration=iteration, history=history, traffic_light_data=tl_data)
        features_sim = feature_builder.get_features_from_simulation(current_input=current_input, initialization=meta_data)
        self.assertEqual(type(features_sim), VectorMap)
        self.assertTrue(np.allclose(features_sim.coords[0], features.coords[0], atol=0.0001))
        for connections, connections_simulation in zip(features_sim.multi_scale_connections[0].values(), features.multi_scale_connections[0].values()):
            self.assertTrue(np.allclose(connections, connections_simulation))
        for lane in range(len(features_sim.lane_groupings[0])):
            for lane_groupings, lane_groupings_simulation in zip(features_sim.lane_groupings[0][lane], features.lane_groupings[0][lane]):
                self.assertTrue(np.allclose(lane_groupings, lane_groupings_simulation))
        self.assertTrue(np.allclose(features_sim.on_route_status[0], features.on_route_status[0], atol=0.0001))
        self.assertTrue(np.allclose(features_sim.traffic_light_data[0], features.traffic_light_data[0]))

    def test_vector_map_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the VectorMapFeatureBuilder can be scripted properly.
        """
        feature_builder = VectorMapFeatureBuilder(radius=20, connection_scales=[2])
        self.assertEqual(feature_builder.get_feature_type(), VectorMap)
        scripted_builder = torch.jit.script(feature_builder)
        self.assertIsNotNone(scripted_builder)
        config = scripted_builder.precomputed_feature_config()
        self.assertTrue('initial_ego_state' in config)
        self.assertTrue('neighbor_vector_map' in config)
        self.assertTrue('radius' in config['neighbor_vector_map'])
        self.assertEqual('20', config['neighbor_vector_map']['radius'])
        num_lane_segment = 5
        num_connections = 7
        tensor_data = {'lane_segment_coords': torch.rand((num_lane_segment, 2, 2), dtype=torch.float64), 'lane_segment_conns': torch.zeros((num_connections, 2), dtype=torch.int64), 'on_route_status': torch.zeros((num_lane_segment, 2), dtype=torch.float32), 'traffic_light_array': torch.zeros((num_lane_segment, 4), dtype=torch.float32), 'anchor_state': torch.zeros((3,), dtype=torch.float64)}
        list_tensor_data = {'lane_segment_groupings': [torch.zeros(size=(2,), dtype=torch.int64) for _ in range(num_lane_segment)]}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_tensor_output, scripted_list_output, scripted_list_list_output = scripted_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        py_tensor_output, py_list_output, py_list_list_output = feature_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        self.assertEqual(0, len(scripted_tensor_output))
        self.assertEqual(0, len(py_tensor_output))
        self.assertEqual(len(scripted_list_output), len(py_list_output))
        for key in py_list_output:
            self.assertEqual(len(py_list_output[key]), len(scripted_list_output[key]))
            for i in range(len(py_list_output[key])):
                torch.testing.assert_close(py_list_output[key][i], scripted_list_output[key][i])
        self.assertEqual(len(py_list_list_output), len(scripted_list_list_output))
        for key in py_list_list_output:
            py_list = py_list_list_output[key]
            scripted_list = scripted_list_list_output[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                py = py_list[i]
                script = scripted_list[i]
                self.assertEqual(len(py), len(script))
                for j in range(len(py)):
                    torch.testing.assert_close(py[j], script[j])

class TestAgents(unittest.TestCase):
    """Test agent feature representation."""

    def setUp(self) -> None:
        """Set up test case."""
        self.ego: List[npt.NDArray[np.float32]] = [np.array(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))]
        self.ego_incorrect: List[npt.NDArray[np.float32]] = [np.array([0.0, 0.0, 0.0])]
        self.agents: List[npt.NDArray[np.float32]] = [np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])]
        self.agents_incorrect: List[npt.NDArray[np.float32]] = [np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])]

    def test_agent_feature(self) -> None:
        """
        Test the core functionality of features
        """
        feature = Agents(ego=self.ego, agents=self.agents)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(Agents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], np.ndarray)
        self.assertIsInstance(feature.agents[0], np.ndarray)
        self.assertIsInstance(feature.get_flatten_agents_features_in_sample(0), np.ndarray)
        self.assertEqual(feature.get_flatten_agents_features_in_sample(0).shape, (2, feature.agents_features_dim))
        feature = feature.to_feature_tensor()
        self.assertIsInstance(feature.get_flatten_agents_features_in_sample(0), torch.Tensor)
        self.assertEqual(feature.get_flatten_agents_features_in_sample(0).shape, (2, feature.agents_features_dim))
        self.assertIsInstance(feature.ego[0], torch.Tensor)
        self.assertIsInstance(feature.agents[0], torch.Tensor)

    def test_no_agents(self) -> None:
        """
        Test when there are no agents
        """
        agents: List[npt.NDArray[np.float32]] = [np.empty((self.ego[0].shape[0], 0, 8), dtype=np.float32)]
        feature = Agents(ego=self.ego, agents=agents)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(Agents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], np.ndarray)
        self.assertIsInstance(feature.agents[0], np.ndarray)
        self.assertIsInstance(feature.get_flatten_agents_features_in_sample(0), np.ndarray)
        self.assertEqual(feature.get_flatten_agents_features_in_sample(0).shape, (0, feature.agents_features_dim))
        feature = feature.to_feature_tensor()
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(Agents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], torch.Tensor)
        self.assertIsInstance(feature.agents[0], torch.Tensor)
        self.assertIsInstance(feature.get_flatten_agents_features_in_sample(0), torch.Tensor)
        self.assertEqual(feature.get_flatten_agents_features_in_sample(0).shape, (0, feature.agents_features_dim))

    def test_incorrect_dimension(self) -> None:
        """
        Test when inputs dimension are incorrect
        """
        with self.assertRaises(AssertionError):
            Agents(ego=self.ego, agents=self.agents_incorrect)
        with self.assertRaises(AssertionError):
            Agents(ego=self.ego_incorrect, agents=self.agents)
        agents: List[npt.NDArray[np.float32]] = [np.empty((self.ego[0].shape[0] + 1, 0, 8), dtype=np.float32)]
        with self.assertRaises(AssertionError):
            Agents(ego=self.ego, agents=agents)
        ego = copy.copy(self.ego)
        ego.append(np.zeros((self.ego[0].shape[0] + 1, self.ego[0].shape[1]), dtype=np.float32))
        with self.assertRaises(AssertionError):
            Agents(ego=ego, agents=self.agents)
        with self.assertRaises(AssertionError):
            Agents(ego=ego, agents=agents)

class TestEgoTrajectoryTargetBuilder(unittest.TestCase):
    """Test class for EgoTrajectoryTargetBuilder."""

    @given(test_parameters=_get_valid_test_parameters())
    @example(test_parameters={'time_step': 0.05, 'number_of_future_iterations': 100, 'num_poses': 10, 'time_horizon': 5.0, 'interval_length': None})
    def test_get_targets(self, test_parameters: Dict[str, Union[int, float]]) -> None:
        """
        Parametrized test for target trajectory extraction.
        :param test_parameters: The dictionary mapping parameters to sampled values to apply in the test.
        """
        test_scenario = MockAbstractScenario(time_step=test_parameters['time_step'], number_of_future_iterations=test_parameters['number_of_future_iterations'])
        test_future_trajectory_sampling = TrajectorySampling(num_poses=test_parameters['num_poses'], time_horizon=test_parameters['time_horizon'], interval_length=test_parameters['interval_length'])
        builder = EgoTrajectoryTargetBuilder(future_trajectory_sampling=test_future_trajectory_sampling)
        generated_features = builder.get_targets(test_scenario)
        self.assertEqual(generated_features.num_of_iterations, test_future_trajectory_sampling.num_poses)
        self.assertIsInstance(generated_features, builder.get_feature_type())

    @given(num_poses_diff=st.integers(min_value=-MIN_FUTURE_ITERATIONS + 1, max_value=MIN_FUTURE_ITERATIONS))
    def test_runtime_error_due_to_expected_pose_mismatch(self, num_poses_diff: int) -> None:
        """This tests the edge case if the number of returned poses doesn't match the expected one in the builder."""
        test_scenario = MockAbstractScenario()
        test_future_trajectory_sampling = TrajectorySampling(num_poses=max(test_scenario._number_of_future_iterations, MIN_FUTURE_ITERATIONS), interval_length=test_scenario._time_step)
        builder = EgoTrajectoryTargetBuilder(future_trajectory_sampling=test_future_trajectory_sampling)
        with patch(f'{PATCH_STR}.convert_absolute_to_relative_poses') as mock_convert_absolute_to_relative_poses:
            mock_convert_absolute_to_relative_poses.return_value = np.zeros((builder._num_future_poses + num_poses_diff, 3), dtype=np.float32)
            if num_poses_diff == 0:
                generated_features = builder.get_targets(test_scenario)
                self.assertIsInstance(generated_features, builder.get_feature_type())
            else:
                with self.assertRaisesRegex(RuntimeError, 'Expected.*num poses but got.*'):
                    builder.get_targets(test_scenario)
        mock_convert_absolute_to_relative_poses.assert_called_once()

@st.composite
def _get_valid_test_parameters(draw: Any) -> Dict[str, Union[int, float]]:
    """
    This function implements a strategy to define trajectory / time parameters
    for both MockAbstractScenario and TrajectorySampling, used by EgoTargetTrajectoryBuilder.
    :param draw: The draw function used to select a specific sample given the strategy.
    :return: A dictionary mapping parameters to sampled values to test.
    """
    test_time_step_samples = st.sampled_from(np.arange(MIN_TIME_INTERVAL, MIN_TIME_INTERVAL + MAX_TIME_INTERVAL, MIN_TIME_INTERVAL))
    test_number_of_future_iterations_samples = st.integers(min_value=MIN_FUTURE_ITERATIONS, max_value=MAX_FUTURE_ITERATIONS)
    test_field_to_set_none = st.one_of(st.none(), st.sampled_from(['num_poses', 'time_horizon', 'interval_length']))
    picked_time_step = draw(test_time_step_samples)
    picked_number_of_future_iterations = draw(test_number_of_future_iterations_samples)
    scenario_trajectory_full_horizon = picked_time_step * picked_number_of_future_iterations
    picked_interval_length = draw(st.sampled_from(np.arange(picked_time_step, scenario_trajectory_full_horizon - picked_time_step, picked_time_step)))
    picked_num_poses = draw(st.integers(min_value=1, max_value=int(scenario_trajectory_full_horizon / picked_interval_length)))
    picked_trajectory_horizon = picked_interval_length * picked_num_poses
    trajectory_sampling_dict = {'time_step': picked_time_step, 'number_of_future_iterations': picked_number_of_future_iterations, 'num_poses': picked_num_poses, 'time_horizon': picked_trajectory_horizon, 'interval_length': picked_interval_length}
    picked_field_to_set_none = draw(test_field_to_set_none)
    if picked_field_to_set_none is not None:
        trajectory_sampling_dict[picked_field_to_set_none] = None
    return trajectory_sampling_dict

