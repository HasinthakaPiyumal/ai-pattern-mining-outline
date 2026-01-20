# Cluster 110

class EgoCentricMLAgents(AbstractMLAgents):
    """
    Simulate agents based on an ML model.
    """

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario) -> None:
        """
        Initializes the EgoCentricMLAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        super().__init__(model, scenario)
        self.prediction_type = 'agents_trajectory'

    @property
    def _ego_velocity_anchor_state(self) -> StateSE2:
        """
        Returns the ego's velocity state vector as an anchor state for transformation.
        :return: A StateSE2 representing ego's velocity state as an anchor state
        """
        ego_velocity = self._ego_anchor_state.dynamic_car_state.rear_axle_velocity_2d
        return StateSE2(ego_velocity.x, ego_velocity.y, self._ego_anchor_state.rear_axle.heading)

    def _infer_model(self, features: FeaturesType) -> TargetsType:
        """Inherited, see superclass."""
        predictions = self._model_loader.infer(features)
        if self.prediction_type not in predictions:
            raise ValueError(f"Prediction does not have the output '{self.prediction_type}'")
        agents_prediction_tensor = cast(AgentsTrajectories, predictions[self.prediction_type]).data
        agents_prediction = agents_prediction_tensor[0].cpu().detach().numpy()
        return {self.prediction_type: AgentsTrajectories([cast(npt.NDArray[np.float32], agents_prediction)]).get_agents_only_trajectories()}

    def _update_observation_with_predictions(self, predictions: TargetsType) -> None:
        """Inherited, see superclass."""
        assert self._agents, 'The agents have not been initialized. Please make sure they are initialized!'
        agent_predictions = cast(AgentsTrajectories, predictions[self.prediction_type])
        agent_predictions.reshape_to_agents()
        agent_poses = agent_predictions.poses[0]
        agent_velocities = agent_predictions.xy_velocity[0]
        for agent_token, agent, poses_horizon, xy_velocity_horizon in zip(self._agents, self._agents.values(), agent_poses, agent_velocities):
            poses = numpy_array_to_absolute_pose(self._ego_anchor_state.rear_axle, poses_horizon)
            xy_velocities = numpy_array_to_absolute_velocity(self._ego_velocity_anchor_state, xy_velocity_horizon)
            future_trajectory = _convert_prediction_to_predicted_trajectory(agent, poses, xy_velocities, self._step_interval_us)
            new_state = future_trajectory.trajectory.get_state_at_time(self.step_time)
            new_agent = Agent(tracked_object_type=agent.tracked_object_type, oriented_box=new_state.oriented_box, velocity=new_state.velocity, metadata=agent.metadata)
            new_agent.predictions = [future_trajectory]
            self._agents[agent_token] = new_agent

class TestEgoCentricMLAgents(unittest.TestCase):
    """Test important functions of EgoCentricMLAgents class"""

    def setUp(self) -> None:
        """Initialize scenario and model for constructing AgentCentricMLAgents class."""
        self.scenario = MockAbstractScenario(number_of_detections=1)
        self.model = Mock(spec=TorchModuleWrapper)
        self.model.future_trajectory_sampling = TrajectorySampling(num_poses=1, time_horizon=1.0)
        self.pred_trajectory = AgentsTrajectories(data=[np.array([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]])])

    def test_update_observation_with_predictions(self) -> None:
        """Test the _update_observation_with_predictions fucntion."""
        obs = EgoCentricMLAgents(model=self.model, scenario=self.scenario)
        obs.initialize()
        self.assertEqual(len(obs._agents), 1)
        self.assertEqual(obs._agents['0'].center.x, 1.0)
        self.assertEqual(obs._agents['0'].center.y, 2.0)
        self.assertAlmostEqual(obs._agents['0'].center.heading, np.pi / 2)
        obs.step_time = TimePoint(1000000.0)
        predictions = {'agents_trajectory': self.pred_trajectory}
        obs._update_observation_with_predictions(predictions)
        self.assertEqual(len(obs._agents), 1)
        self.assertAlmostEqual(obs._agents['0'].center.x, 1.0)
        self.assertAlmostEqual(obs._agents['0'].center.y, 1.0)
        self.assertAlmostEqual(obs._agents['0'].center.heading, 0.0)

    @patch('nuplan.planning.simulation.planner.ml_planner.model_loader.ModelLoader.infer')
    def test_infer_model(self, mock_infer: Mock) -> None:
        """Test _infer_model function."""
        predictions = {'agents_trajectory': self.pred_trajectory.to_feature_tensor()}
        mock_infer.return_value = predictions
        obs = EgoCentricMLAgents(model=self.model, scenario=self.scenario)
        obs.initialize()
        agents_raster = Mock(spec=Agents)
        features = {'agents': agents_raster}
        results = obs._infer_model(features)
        mock_infer.assert_called_with(features)
        self.assertIn(obs.prediction_type, results)
        self.assertIsInstance(results[obs.prediction_type], AgentsTrajectories)
        self.assertIsInstance(results[obs.prediction_type].data[0], np.ndarray)

class TestAgentImitationObjective(unittest.TestCase):
    """Test agent imitation objective."""

    def setUp(self) -> None:
        """Set up test case."""
        self.target_data: List[npt.NDArray[np.float32]] = [np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])]
        self.prediction_data: List[npt.NDArray[np.float32]] = [np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]])]
        self.objective = AgentsImitationObjective(scenario_type_loss_weighting={})

    def test_compute_loss(self) -> None:
        """
        Test loss computation
        """
        prediction = AgentsTrajectories(data=self.prediction_data)
        target = AgentsTrajectories(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='lane_following_with_lead', scenario_type='') for _ in range(2)]
        loss = self.objective.compute({'agents_trajectory': prediction.to_feature_tensor()}, {'agents_trajectory': target.to_feature_tensor()}, scenarios)
        self.assertEqual(loss, torch.tensor(0.5))

    def test_zero_loss(self) -> None:
        """
        Test perfect prediction. The loss should be zero
        """
        target = AgentsTrajectories(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='lane_following_with_lead', scenario_type='') for _ in range(2)]
        loss = self.objective.compute({'agents_trajectory': target.to_feature_tensor()}, {'agents_trajectory': target.to_feature_tensor()}, scenarios)
        self.assertEqual(loss, torch.tensor(0.0))

class TestTrajectoryWeightDecayImitationObjective(unittest.TestCase):
    """Test weight decay imitation objective."""

    def setUp(self) -> None:
        """Set up test case."""
        self.target_data: npt.NDArray[np.float32] = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
        self.prediction_data: npt.NDArray[np.float32] = np.array([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])
        self.objective = TrajectoryWeightDecayImitationObjective(scenario_type_loss_weighting={})

    def test_compute_loss(self) -> None:
        """
        Test loss computation
        """
        prediction = Trajectory(data=self.prediction_data)
        target = Trajectory(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='lane_following_with_lead', scenario_type='') for _ in range(2)]
        loss = self.objective.compute({'trajectory': prediction.to_feature_tensor()}, {'trajectory': target.to_feature_tensor()}, scenarios)
        torch.testing.assert_allclose(loss, torch.tensor(0.60653, dtype=torch.float64))

    def test_zero_loss(self) -> None:
        """
        Test perfect prediction. The loss should be zero
        """
        target = Trajectory(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='lane_following_with_lead', scenario_type='') for _ in range(2)]
        loss = self.objective.compute({'trajectory': target.to_feature_tensor()}, {'trajectory': target.to_feature_tensor()}, scenarios)
        self.assertEqual(loss, torch.tensor(0.0))

class TestImitationObjective(unittest.TestCase):
    """Test weight decay imitation objective."""

    def setUp(self) -> None:
        """Set up test case."""
        self.target_data: npt.NDArray[np.float32] = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
        self.prediction_data: npt.NDArray[np.float32] = np.array([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])
        self.objective = ImitationObjective(scenario_type_loss_weighting={'unknown': 1.0, 'lane_following_with_lead': 2.0})

    def test_compute_loss(self) -> None:
        """
        Test loss computation
        """
        prediction = Trajectory(data=self.prediction_data)
        target = Trajectory(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='', scenario_type='lane_following_with_lead'), CachedScenario(log_name='', token='', scenario_type='unknown')]
        loss = self.objective.compute({'trajectory': prediction.to_feature_tensor()}, {'trajectory': target.to_feature_tensor()}, scenarios)
        self.assertEqual(loss, torch.tensor(1.5))

    def test_zero_loss(self) -> None:
        """
        Test perfect prediction. The loss should be zero
        """
        target = Trajectory(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='', scenario_type='lane_following_with_lead'), CachedScenario(log_name='', token='', scenario_type='unknown')]
        loss = self.objective.compute({'trajectory': target.to_feature_tensor()}, {'trajectory': target.to_feature_tensor()}, scenarios)
        self.assertEqual(loss, torch.tensor(0.0))

def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())

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

def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())

class RasterModel(TorchModuleWrapper):
    """
    Wrapper around raster-based CNN model that consumes ego, agent and map data in rasterized format
    and regresses ego's future trajectory.
    """

    def __init__(self, feature_builders: List[AbstractFeatureBuilder], target_builders: List[AbstractTargetBuilder], model_name: str, pretrained: bool, num_input_channels: int, num_features_per_pose: int, future_trajectory_sampling: TrajectorySampling):
        """
        Initialize model.
        :param feature_builders: list of builders for features
        :param target_builders: list of builders for targets
        :param model_name: name of the model (e.g. resnet_50, efficientnet_b3)
        :param pretrained: whether the model will be pretrained
        :param num_input_channels: number of input channel of the raster model.
        :param num_features_per_pose: number of features per single pose
        :param future_trajectory_sampling: parameters of predicted trajectory
        """
        super().__init__(feature_builders=feature_builders, target_builders=target_builders, future_trajectory_sampling=future_trajectory_sampling)
        num_output_features = future_trajectory_sampling.num_poses * num_features_per_pose
        self._model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=num_input_channels)
        mlp = torch.nn.Linear(in_features=self._model.num_features, out_features=num_output_features)
        if hasattr(self._model, 'classifier'):
            self._model.classifier = mlp
        elif hasattr(self._model, 'fc'):
            self._model.fc = mlp
        else:
            raise NameError('Expected output layer named "classifier" or "fc" in model')

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "raster": Raster,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        raster: Raster = features['raster']
        predictions = self._model.forward(raster.data)
        return {'trajectory': Trajectory(data=convert_predictions_to_trajectory(predictions))}

def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())

class UrbanDriverOpenLoopModel(TorchModuleWrapper):
    """
    Vector-based model that uses PointNet-based subgraph layers for collating loose collections of vectorized inputs
    into local feature descriptors to be used as input to a global Transformer.

    Adapted from L5Kit's implementation of "Urban Driver: Learning to Drive from Real-world Demonstrations
    Using Policy Gradients":
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py
    Only the open-loop  version of the model is here represented, with slight modifications to fit the nuPlan framework.
    Changes:
        1. Use nuPlan features from NuPlanScenario
        2. Format model for using pytorch_lightning
    """

    def __init__(self, model_params: UrbanDriverOpenLoopModelParams, feature_params: UrbanDriverOpenLoopModelFeatureParams, target_params: UrbanDriverOpenLoopModelTargetParams):
        """
        Initialize UrbanDriverOpenLoop model.
        :param model_params: internal model parameters.
        :param feature_params: agent and map feature parameters.
        :param target_params: target parameters.
        """
        super().__init__(feature_builders=[VectorSetMapFeatureBuilder(map_features=feature_params.map_features, max_elements=feature_params.max_elements, max_points=feature_params.max_points, radius=feature_params.vector_set_map_feature_radius, interpolation_method=feature_params.interpolation_method), GenericAgentsFeatureBuilder(feature_params.agent_features, feature_params.past_trajectory_sampling)], target_builders=[EgoTrajectoryTargetBuilder(target_params.future_trajectory_sampling)], future_trajectory_sampling=target_params.future_trajectory_sampling)
        self._model_params = model_params
        self._feature_params = feature_params
        self._target_params = target_params
        self.feature_embedding = nn.Linear(self._feature_params.feature_dimension, self._model_params.local_embedding_size)
        self.positional_embedding = SinusoidalPositionalEmbedding(self._model_params.local_embedding_size)
        self.type_embedding = TypeEmbedding(self._model_params.global_embedding_size, self._feature_params.feature_types)
        self.local_subgraph = LocalSubGraph(num_layers=self._model_params.num_subgraph_layers, dim_in=self._model_params.local_embedding_size)
        if self._model_params.global_embedding_size != self._model_params.local_embedding_size:
            self.global_from_local = nn.Linear(self._model_params.local_embedding_size, self._model_params.global_embedding_size)
        num_timesteps = self.future_trajectory_sampling.num_poses
        self.global_head = MultiheadAttentionGlobalHead(self._model_params.global_embedding_size, num_timesteps, self._target_params.num_output_features // num_timesteps, dropout=self._model_params.global_head_dropout)

    def extract_agent_features(self, ego_agent_features: GenericAgents, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ego and agent features into format expected by network and build accompanying availability matrix.
        :param ego_agent_features: agent features to be extracted (ego + other agents)
        :param batch_size: number of samples in batch to extract
        :return:
            agent_features: <torch.FloatTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element, feature_dimension>. Stacked ego, agent, and map features.
            agent_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        agent_features = []
        agent_avails = []
        for sample_idx in range(batch_size):
            sample_ego_feature = ego_agent_features.ego[sample_idx][..., :min(self._feature_params.ego_dimension, self._feature_params.feature_dimension)].unsqueeze(0)
            if min(self._feature_params.ego_dimension, GenericAgents.ego_state_dim()) < self._feature_params.feature_dimension:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.feature_dimension, dim=2)
            sample_ego_avails = torch.ones(sample_ego_feature.shape[0], sample_ego_feature.shape[1], dtype=torch.bool, device=sample_ego_feature.device)
            sample_ego_feature = torch.flip(sample_ego_feature, dims=[1])
            sample_ego_feature = sample_ego_feature[:, :self._feature_params.total_max_points, ...]
            sample_ego_avails = sample_ego_avails[:, :self._feature_params.total_max_points, ...]
            if sample_ego_feature.shape[1] < self._feature_params.total_max_points:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.total_max_points, dim=1)
                sample_ego_avails = pad_avails(sample_ego_avails, self._feature_params.total_max_points, dim=1)
            sample_features = [sample_ego_feature]
            sample_avails = [sample_ego_avails]
            for feature_name in self._feature_params.agent_features:
                if ego_agent_features.has_agents(feature_name, sample_idx):
                    sample_agent_features = torch.permute(ego_agent_features.agents[feature_name][sample_idx], (1, 0, 2))
                    sample_agent_features = sample_agent_features[..., :min(self._feature_params.agent_dimension, self._feature_params.feature_dimension)]
                    if min(self._feature_params.agent_dimension, GenericAgents.agents_states_dim()) < self._feature_params.feature_dimension:
                        sample_agent_features = pad_polylines(sample_agent_features, self._feature_params.feature_dimension, dim=2)
                    sample_agent_avails = torch.ones(sample_agent_features.shape[0], sample_agent_features.shape[1], dtype=torch.bool, device=sample_agent_features.device)
                    sample_agent_features = torch.flip(sample_agent_features, dims=[1])
                    sample_agent_features = sample_agent_features[:, :self._feature_params.total_max_points, ...]
                    sample_agent_avails = sample_agent_avails[:, :self._feature_params.total_max_points, ...]
                    if sample_agent_features.shape[1] < self._feature_params.total_max_points:
                        sample_agent_features = pad_polylines(sample_agent_features, self._feature_params.total_max_points, dim=1)
                        sample_agent_avails = pad_avails(sample_agent_avails, self._feature_params.total_max_points, dim=1)
                    sample_agent_features = sample_agent_features[:self._feature_params.max_agents, ...]
                    sample_agent_avails = sample_agent_avails[:self._feature_params.max_agents, ...]
                    if sample_agent_features.shape[0] < self._feature_params.max_agents:
                        sample_agent_features = pad_polylines(sample_agent_features, self._feature_params.max_agents, dim=0)
                        sample_agent_avails = pad_avails(sample_agent_avails, self._feature_params.max_agents, dim=0)
                else:
                    sample_agent_features = torch.zeros(self._feature_params.max_agents, self._feature_params.total_max_points, self._feature_params.feature_dimension, dtype=torch.float32, device=sample_ego_feature.device)
                    sample_agent_avails = torch.zeros(self._feature_params.max_agents, self._feature_params.total_max_points, dtype=torch.bool, device=sample_agent_features.device)
                sample_features.append(sample_agent_features)
                sample_avails.append(sample_agent_avails)
            sample_features = torch.cat(sample_features, dim=0)
            sample_avails = torch.cat(sample_avails, dim=0)
            agent_features.append(sample_features)
            agent_avails.append(sample_avails)
        agent_features = torch.stack(agent_features)
        agent_avails = torch.stack(agent_avails)
        return (agent_features, agent_avails)

    def extract_map_features(self, vector_set_map_data: VectorSetMap, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract map features into format expected by network and build accompanying availability matrix.
        :param vector_set_map_data: VectorSetMap features to be extracted
        :param batch_size: number of samples in batch to extract
        :return:
            map_features: <torch.FloatTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element, feature_dimension>. Stacked map features.
            map_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        map_features = []
        map_avails = []
        for sample_idx in range(batch_size):
            sample_map_features = []
            sample_map_avails = []
            for feature_name in self._feature_params.map_features:
                coords = vector_set_map_data.coords[feature_name][sample_idx]
                tl_data = vector_set_map_data.traffic_light_data[feature_name][sample_idx] if feature_name in vector_set_map_data.traffic_light_data else None
                avails = vector_set_map_data.availabilities[feature_name][sample_idx]
                if tl_data is not None:
                    coords = torch.cat((coords, tl_data), dim=2)
                coords = coords[:, :self._feature_params.total_max_points, ...]
                avails = avails[:, :self._feature_params.total_max_points]
                if coords.shape[1] < self._feature_params.total_max_points:
                    coords = pad_polylines(coords, self._feature_params.total_max_points, dim=1)
                    avails = pad_avails(avails, self._feature_params.total_max_points, dim=1)
                coords = coords[..., :self._feature_params.feature_dimension]
                if coords.shape[2] < self._feature_params.feature_dimension:
                    coords = pad_polylines(coords, self._feature_params.feature_dimension, dim=2)
                sample_map_features.append(coords)
                sample_map_avails.append(avails)
            map_features.append(torch.cat(sample_map_features))
            map_avails.append(torch.cat(sample_map_avails))
        map_features = torch.stack(map_features)
        map_avails = torch.stack(map_avails)
        return (map_features, map_avails)

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        vector_set_map_data = cast(VectorSetMap, features['vector_set_map'])
        ego_agent_features = cast(GenericAgents, features['generic_agents'])
        batch_size = ego_agent_features.batch_size
        agent_features, agent_avails = self.extract_agent_features(ego_agent_features, batch_size)
        map_features, map_avails = self.extract_map_features(vector_set_map_data, batch_size)
        features = torch.cat([agent_features, map_features], dim=1)
        avails = torch.cat([agent_avails, map_avails], dim=1)
        feature_embedding = self.feature_embedding(features)
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2)
        invalid_mask = ~avails
        invalid_polys = invalid_mask.all(-1)
        embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding)
        if hasattr(self, 'global_from_local'):
            embeddings = self.global_from_local(embeddings)
        embeddings = F.normalize(embeddings, dim=-1) * self._model_params.global_embedding_size ** 0.5
        embeddings = embeddings.transpose(0, 1)
        type_embedding = self.type_embedding(batch_size, self._feature_params.max_agents, self._feature_params.agent_features, self._feature_params.map_features, self._feature_params.max_elements, device=features.device).transpose(0, 1)
        if self._feature_params.disable_agents:
            invalid_polys[:, 1:1 + self._feature_params.max_agents * len(self._feature_params.agent_features)] = 1
        if self._feature_params.disable_map:
            invalid_polys[:, 1 + self._feature_params.max_agents * len(self._feature_params.agent_features):] = 1
        invalid_polys[:, 0] = 0
        outputs, attns = self.global_head(embeddings, type_embedding, invalid_polys)
        return {'trajectory': Trajectory(data=convert_predictions_to_trajectory(outputs))}

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

def distributed_weighted_sampler_init(scenario_dataset: ScenarioDataset, scenario_sampling_weights: Dict[str, float], replacement: bool=True) -> WeightedRandomSampler:
    """
    Initiliazes WeightedSampler object with sampling weights for each scenario_type and returns it.
    :param scenario_dataset: ScenarioDataset object
    :param replacement: Samples with replacement if True. By default set to True.
    return: Initialized Weighted sampler
    """
    scenarios = scenario_dataset._scenarios
    if not replacement:
        assert all((w > 0 for w in scenario_sampling_weights.values())), 'All scenario sampling weights must be positive when sampling without replacement.'
    default_scenario_sampling_weight = 1.0
    scenario_sampling_weights_per_idx = [scenario_sampling_weights[scenario.scenario_type] if scenario.scenario_type in scenario_sampling_weights else default_scenario_sampling_weight for scenario in scenarios]
    weighted_sampler = WeightedRandomSampler(weights=scenario_sampling_weights_per_idx, num_samples=len(scenarios), replacement=replacement)
    distributed_weighted_sampler = DistributedSamplerWrapper(weighted_sampler)
    return distributed_weighted_sampler

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

def move_features_type_to_device(batch: FeaturesType, device: torch.device) -> FeaturesType:
    """
    Move all features to a device
    :param batch: batch of features
    :param device: new device
    :return: batch moved to new device
    """
    output = {}
    for key, value in batch.items():
        output[key] = value.to_device(device)
    return output

class TestScenarioSamplingWeights(unittest.TestCase):
    """
    Tests data loading functionality in a sequential manner.
    """

    def setUp(self) -> None:
        """Set up test variables."""
        self.mock_scenario_sampling_weights = {DEFAULT_SCENARIO_NAME: 0.5}
        self.mock_scenario_types = [DEFAULT_SCENARIO_NAME, 'following_lane_with_lead']
        self.mock_scenarios = []
        for scenario_type in self.mock_scenario_types:
            self.mock_scenarios += [CachedScenario(log_name='', token='', scenario_type=scenario_type) for _ in range(3)]
        self.expected_sampler_weights = [self.mock_scenario_sampling_weights[DEFAULT_SCENARIO_NAME]] * 3 + [1.0] * 3

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

    def test_scenario_sampling_weight_initialises_correctly(self) -> None:
        """
        Test that the scenario sampling weights are correct.
        """
        self._init_distributed_process_group()
        scenarios_dataset = Mock(ScenarioDataset)
        scenarios_dataset._scenarios = self.mock_scenarios
        distributed_weight_sampler = distributed_weighted_sampler_init(scenario_dataset=scenarios_dataset, scenario_sampling_weights=self.mock_scenario_sampling_weights)
        self.assertEqual(list(distributed_weight_sampler.sampler.weights), self.expected_sampler_weights)

class TestDistributedSamplerWrapper(unittest.TestCase):
    """
    Skeleton with initialized dataloader used in testing.
    """

    def setUp(self) -> None:
        """
        Set up basic configs.
        """
        self.mock_sampler = self._get_sampler()
        self.num_replicas = 4
        self.expected_indices = [[i % 10 for i in range(j, j + 3)] for j in range(0, 12, 3)]

    def _get_sampler(self) -> SequentialSampler:
        mock_sampler = SequentialSampler([i for i in range(10)])
        return mock_sampler

    def test_distributed_sampler_wrapper(self) -> None:
        """
        Tests that the indices produced by the distributed sampler wrapper are as expected.
        """
        distributed_samplers = [DistributedSamplerWrapper(sampler=self.mock_sampler, num_replicas=self.num_replicas, rank=i) for i in range(self.num_replicas)]
        for i, distributed_sampler in enumerate(distributed_samplers):
            indices = list(distributed_sampler)
            self.assertEqual(self.expected_indices[i], indices)

class VisualizationCallback(pl.Callback):
    """
    Callback that visualizes planner model inputs/outputs and logs them in Tensorboard.
    """

    def __init__(self, images_per_tile: int, num_train_tiles: int, num_val_tiles: int, pixel_size: float):
        """
        Initialize the class.

        :param images_per_tile: number of images per tiles to visualize
        :param num_train_tiles: number of tiles from the training set
        :param num_val_tiles: number of tiles from the validation set
        :param pixel_size: [m] size of pixel in meters
        """
        super().__init__()
        self.custom_batch_size = images_per_tile
        self.num_train_images = num_train_tiles * images_per_tile
        self.num_val_images = num_val_tiles * images_per_tile
        self.pixel_size = pixel_size
        self.train_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.val_dataloader: Optional[torch.utils.data.DataLoader] = None

    def _initialize_dataloaders(self, datamodule: pl.LightningDataModule) -> None:
        """
        Initialize the dataloaders. This makes sure that the same examples are sampled
        every time for comparison during visualization.

        :param datamodule: lightning datamodule
        """
        train_set = datamodule.train_dataloader().dataset
        val_set = datamodule.val_dataloader().dataset
        self.train_dataloader = self._create_dataloader(train_set, self.num_train_images)
        self.val_dataloader = self._create_dataloader(val_set, self.num_val_images)

    def _create_dataloader(self, dataset: torch.utils.data.Dataset, num_samples: int) -> torch.utils.data.DataLoader:
        dataset_size = len(dataset)
        num_keep = min(dataset_size, num_samples)
        sampled_idxs = random.sample(range(dataset_size), num_keep)
        subset = torch.utils.data.Subset(dataset=dataset, indices=sampled_idxs)
        return torch.utils.data.DataLoader(dataset=subset, batch_size=self.custom_batch_size, collate_fn=FeatureCollate())

    def _log_from_dataloader(self, pl_module: pl.LightningModule, dataloader: torch.utils.data.DataLoader, loggers: List[Any], training_step: int, prefix: str) -> None:
        """
        Visualizes and logs all examples from the input dataloader.

        :param pl_module: lightning module used for inference
        :param dataloader: torch dataloader
        :param loggers: list of loggers from the trainer
        :param training_step: global step in training
        :param prefix: prefix to add to the log tag
        """
        for batch_idx, batch in enumerate(dataloader):
            features: FeaturesType = batch[0]
            targets: TargetsType = batch[1]
            predictions = self._infer_model(pl_module, move_features_type_to_device(features, pl_module.device))
            self._log_batch(loggers, features, targets, predictions, batch_idx, training_step, prefix)

    def _log_batch(self, loggers: List[Any], features: FeaturesType, targets: TargetsType, predictions: TargetsType, batch_idx: int, training_step: int, prefix: str) -> None:
        """
        Visualizes and logs a batch of data (features, targets, predictions) from the model.

        :param loggers: list of loggers from the trainer
        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :param batch_idx: index of total batches to visualize
        :param training_step: global training step
        :param prefix: prefix to add to the log tag
        """
        if 'trajectory' not in targets or 'trajectory' not in predictions:
            return
        if 'raster' in features:
            image_batch = self._get_images_from_raster_features(features, targets, predictions)
        elif ('vector_map' in features or 'vector_set_map' in features) and ('agents' in features or 'generic_agents' in features):
            image_batch = self._get_images_from_vector_features(features, targets, predictions)
        else:
            return
        tag = f'{prefix}_visualization_{batch_idx}'
        for logger in loggers:
            if isinstance(logger, torch.utils.tensorboard.writer.SummaryWriter):
                logger.add_images(tag=tag, img_tensor=torch.from_numpy(image_batch), global_step=training_step, dataformats='NHWC')

    def _get_images_from_raster_features(self, features: FeaturesType, targets: TargetsType, predictions: TargetsType) -> npt.NDArray[np.uint8]:
        """
        Create a list of RGB raster images from a batch of model data of raster features.

        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        images = list()
        for raster, target_trajectory, predicted_trajectory in zip(features['raster'].unpack(), targets['trajectory'].unpack(), predictions['trajectory'].unpack()):
            image = get_raster_with_trajectories_as_rgb(raster, target_trajectory, predicted_trajectory, pixel_size=self.pixel_size)
            images.append(image)
        return np.asarray(images)

    def _get_images_from_vector_features(self, features: FeaturesType, targets: TargetsType, predictions: TargetsType) -> npt.NDArray[np.uint8]:
        """
        Create a list of RGB raster images from a batch of model data of vectormap and agent features.

        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        images = list()
        vector_map_feature = 'vector_map' if 'vector_map' in features else 'vector_set_map'
        agents_feature = 'agents' if 'agents' in features else 'generic_agents'
        for vector_map, agents, target_trajectory, predicted_trajectory in zip(features[vector_map_feature].unpack(), features[agents_feature].unpack(), targets['trajectory'].unpack(), predictions['trajectory'].unpack()):
            image = get_raster_from_vector_map_with_agents(vector_map, agents, target_trajectory, predicted_trajectory, pixel_size=self.pixel_size)
            images.append(image)
        return np.asarray(images)

    def _infer_model(self, pl_module: pl.LightningModule, features: FeaturesType) -> TargetsType:
        """
        Make an inference of the input batch features given a model.

        :param pl_module: lightning model
        :param features: model inputs
        :return: model predictions
        """
        with torch.no_grad():
            pl_module.eval()
            predictions = move_features_type_to_device(pl_module(features), torch.device('cpu'))
            pl_module.train()
        return predictions

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional=None) -> None:
        """
        Visualizes and logs training examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, 'datamodule'), 'Trainer missing datamodule attribute'
        assert hasattr(trainer, 'global_step'), 'Trainer missing global_step attribute'
        if self.train_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)
        self._log_from_dataloader(pl_module, self.train_dataloader, trainer.logger.experiment, trainer.global_step, 'train')

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional=None) -> None:
        """
        Visualizes and logs validation examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, 'datamodule'), 'Trainer missing datamodule attribute'
        assert hasattr(trainer, 'global_step'), 'Trainer missing global_step attribute'
        if self.val_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)
        self._log_from_dataloader(pl_module, self.val_dataloader, trainer.logger.experiment, trainer.global_step, 'val')

def _score_model(pl_module: pl.LightningModule, features: FeaturesType, targets: TargetsType) -> Tuple[float, FeaturesType]:
    """
    Make an inference of the input batch feature given a model and score them through their objective
    :param pl_module: lightning model
    :param features: model inputs
    :param targets: training targets
    :return: tuple of score and prediction
    """
    objectives = pl_module.objectives
    with torch.no_grad():
        pl_module.eval()
        predictions = pl_module(features)
        pl_module.train()
    score = 0.0
    for objective in objectives:
        score += objective.compute(predictions, targets).to('cpu')
    return (score / len(objectives), move_features_type_to_device(predictions, torch.device('cpu')))

def _eval_model_and_write_to_scene(dataloader: torch.utils.data.DataLoader, pl_module: pl.LightningModule, scene_converter: SceneConverter, num_store: int, output_dir: Path) -> None:
    """
    Evaluate prediction of the model and write scenes based on their scores
    :param dataloader: pytorch data loader
    :param pl_module: lightning module
    :param scene_converter: converts data from the scored scenario into scene dictionary
    :param num_store: n number of scenarios to be written into scenes for each best, worst and random cases
    :param output_dir: output directory of scene file
    """
    scenario_dataset = dataloader.dataset
    score_record = torch.empty(len(scenario_dataset))
    predictions: List[TargetsType] = []
    for sample_idx, sample in enumerate(dataloader):
        features = cast(FeaturesType, sample[0])
        targets = cast(TargetsType, sample[1])
        score, prediction = _score_model(pl_module, move_features_type_to_device(features, pl_module.device), move_features_type_to_device(targets, pl_module.device))
        predictions.append(prediction)
        score_record[sample_idx] = score
    best_n_idx = torch.topk(score_record, num_store, largest=False).indices.tolist()
    worst_n_idx = torch.topk(score_record, num_store).indices.tolist()
    random_n_idx = random.sample(range(len(scenario_dataset)), num_store)
    for data_idx, score_type in zip((best_n_idx, worst_n_idx, random_n_idx), ('best', 'worst', 'random')):
        for idx in data_idx:
            features, targets, _ = scenario_dataset[idx]
            scenario = scenario_dataset._scenarios[idx]
            scenes = scene_converter(scenario, features, targets, predictions[idx])
            file_dir = output_dir / score_type / scenario.token
            if not is_s3_path(file_dir):
                file_dir.mkdir(parents=True, exist_ok=True)
            _dump_scenes(scenes, file_dir)

def _dump_scenes(scenes: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Dump a single scene file
    :param scenes: list of scenes to be written
    :param output_dir: final output directory
    """
    for scene in scenes:
        file_name = output_dir / str(scene['ego']['timestamp_us'])
        with open(str(file_name.with_suffix('.json')), 'w') as outfile:
            json.dump(scene, outfile, indent=4)

class ScenarioScoringCallback(pl.Callback):
    """
    Callback that performs an evaluation to score the model on each validation data.
    The n-best, n-worst and n-random data is written into a scene.

    The directory structure for the output of the scenes is:
        <output_dir>
            └── scenes
                ├── best
                │     ├── scenario_token_01
                │     │         ├── timestamp_01.json
                │     │         └── timestamp_02.json
                │     :                    :
                │     └── scenario_token_n
                ├── worst
                └── random
    """

    def __init__(self, scene_converter: SceneConverter, num_store: int, frequency: int, output_dir: Union[str, Path]):
        """
        Initialize the callback.
        :param scene_converter: Converts data from the scored scenario into scene dictionary.
        :param num_store: N number of scenarios to be written into scenes for each best, worst and random cases.
        :param frequency: Interval between epochs at which to perform the evaluation. Set 0 to skip the callback.
        :param output_dir: Output directory of scene file.
        """
        super().__init__()
        self._num_store = num_store
        self._frequency = frequency
        self._scene_converter = scene_converter
        self._output_dir = Path(output_dir) / 'scenes'
        self._val_dataloader: Optional[torch.utils.data.DataLoader] = None

    def _initialize_dataloaders(self, datamodule: pl.LightningDataModule) -> None:
        """
        Initialize the dataloaders. This makes sure that the same examples are sampled every time.
        :param datamodule: Lightning datamodule.
        """
        val_set = datamodule.val_dataloader().dataset
        assert isinstance(val_set, ScenarioDataset), 'invalid dataset type, dataset must be a scenario dataset'
        self._val_dataloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, shuffle=False, collate_fn=FeatureCollate())

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of each epoch validation.
        :param trainer: Lightning trainer.
        :param pl_module: lightning model.
        """
        if self._frequency == 0:
            return
        assert hasattr(trainer, 'datamodule'), 'Trainer missing datamodule attribute'
        assert hasattr(trainer, 'current_epoch'), 'Trainer missing current_epoch attribute'
        epoch = trainer.current_epoch
        if epoch % self._frequency == 0:
            if self._val_dataloader is None:
                self._initialize_dataloaders(trainer.datamodule)
            output_dir = self._output_dir / f'epoch={epoch}'
            _eval_model_and_write_to_scene(self._val_dataloader, pl_module, self._scene_converter, self._num_store, output_dir)

def mock_compute_features(scenario: AbstractScenario) -> Tuple[FeaturesType, TargetsType, None]:
    """
    Mock feature computation.
    :param scenario: Input scenario to extract features from.
    :return: Extracted features and targets.
    """
    mission_goal = scenario.get_mission_goal()
    data1 = torch.tensor(mission_goal.x)
    data2 = torch.tensor(mission_goal.y)
    data3 = torch.tensor(mission_goal.heading)
    mock_feature = DummyVectorMapFeature(data1=[data1], data2=[data2], data3=[{'test': data3}])
    mock_output = {'mock_feature': mock_feature}
    mock_cache_metadata = None
    return (mock_output, mock_output, mock_cache_metadata)

class TestScenarioScoringCallback(unittest.TestCase):
    """Test scenario scoring callback"""

    def setUp(self) -> None:
        """Set up test case."""
        self.output_dir = tempfile.TemporaryDirectory()
        preprocessor = Mock()
        preprocessor.compute_features.side_effect = mock_compute_features
        self.mock_scenarios = [MockAbstractScenario(mission_goal=StateSE2(x=1.0, y=0.0, heading=0.0)), MockAbstractScenario(mission_goal=StateSE2(x=0.0, y=0.0, heading=0.0))]
        self.scenario_time_stamp = self.mock_scenarios[0]._initial_time_us
        mock_scenario_dataset = ScenarioDataset(scenarios=self.mock_scenarios, feature_preprocessor=preprocessor)
        mock_datamodule = Mock()
        mock_datamodule.val_dataloader().dataset = mock_scenario_dataset
        self.trainer = Mock()
        self.trainer.datamodule = mock_datamodule
        self.trainer.current_epoch = 1
        mock_objective = Mock()
        mock_objective.compute.side_effect = mock_compute_objective
        self.pl_module = Mock()
        self.pl_module.device = 'cpu'
        self.pl_module.side_effect = mock_predict
        self.pl_module.objectives = [mock_objective]
        scenario_converter = ScenarioSceneConverter(ego_trajectory_horizon=1, ego_trajectory_poses=2)
        self.callback = ScenarioScoringCallback(scene_converter=scenario_converter, num_store=1, frequency=1, output_dir=self.output_dir.name)
        self.callback._initialize_dataloaders(self.trainer.datamodule)

    def test_initialize_dataloaders(self) -> None:
        """
        Test callback dataloader initialization.
        """
        invalid_datamodule = Mock()
        invalid_datamodule.val_dataloader().dataset = None
        with self.assertRaises(AssertionError):
            self.callback._initialize_dataloaders(invalid_datamodule)
        self.callback._initialize_dataloaders(self.trainer.datamodule)
        self.assertIsInstance(self.callback._val_dataloader, torch.utils.data.DataLoader)

    def test_score_model(self) -> None:
        """
        Test scoring of the model with mock features.
        """
        data1 = torch.tensor(1)
        data2 = torch.tensor(2)
        data3 = torch.tensor(3)
        mock_feature = DummyVectorMapFeature(data1=[data1], data2=[data2], data3=[{'test': data3}])
        mock_input = {'mock_feature': mock_feature}
        score, prediction = _score_model(self.pl_module, mock_input, mock_input)
        self.assertEqual(score, mock_feature.data1[0])
        self.assertEqual(prediction, mock_input)

    def test_on_validation_epoch_end(self) -> None:
        """
        Test on validation callback.
        """
        BEST_INDEX = 1
        WORST_INDEX = 0
        self.callback._initialize_dataloaders(self.trainer.datamodule)
        self.callback.on_validation_epoch_end(self.trainer, self.pl_module)
        best_score_path = pathlib.Path(self.output_dir.name + f'/scenes/epoch={self.trainer.current_epoch}' + f'/best/{self.mock_scenarios[BEST_INDEX].token}/{self.scenario_time_stamp.time_us}.json')
        self.assertTrue(best_score_path.exists())
        worst_score_path = pathlib.Path(self.output_dir.name + f'/scenes/epoch={self.trainer.current_epoch}' + f'/worst/{self.mock_scenarios[WORST_INDEX].token}/{self.scenario_time_stamp.time_us}.json')
        self.assertTrue(worst_score_path.exists())
        random_score_dir = pathlib.Path(self.output_dir.name + f'/scenes/epoch={self.trainer.current_epoch}/random/')
        random_score_paths = list(random_score_dir.glob(f'*/{self.scenario_time_stamp.time_us}.json'))
        self.assertEqual(len(random_score_paths), 1)
        with open(str(best_score_path), 'r') as f:
            best_data = json.load(f)
        with open(str(worst_score_path), 'r') as f:
            worst_data = json.load(f)
        self.assertEqual(worst_data['goal']['pose'][0], self.mock_scenarios[WORST_INDEX].get_mission_goal().x)
        self.assertEqual(best_data['goal']['pose'][0], self.mock_scenarios[BEST_INDEX].get_mission_goal().x)

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

def extract_initial_offset(scenario: AbstractScenario) -> npt.NDArray[np.float32]:
    """
    Return offset
    :param scenario: for which offset should be computed
    :return: offset of type [x, y, heading]
    """
    initial_ego_pose = ego_pose_to_array(scenario.get_ego_state_at_iteration(0))
    initial_offset: npt.NDArray[np.float32] = np.array([initial_ego_pose[0], initial_ego_pose[1], 0.0])
    return initial_offset

def ego_pose_to_array(ego_pose: EgoState) -> npt.NDArray[np.float32]:
    """
    Convert EgoState to array
    :param ego_pose: agent state
    :return: [x, y, heading]
    """
    return np.array([ego_pose.rear_axle.x, ego_pose.rear_axle.y, ego_pose.rear_axle.heading])

def extract_ego_trajectory(scenario: AbstractScenario, shorten_end_of_scenario: int, offset_start_of_scenario: int=0, subtract_initial_pose_offset: bool=True) -> Trajectory:
    """
    Extract ego trajectory from scenario
    :param scenario: for which ego trajectory should be extracted
    :param shorten_end_of_scenario: future poses
    :param offset_start_of_scenario: future poses
    :param subtract_initial_pose_offset: subtract offset from initial ego pose
    :return: Ego Trajectory
    """
    iteration_end = scenario.get_number_of_iterations() - shorten_end_of_scenario
    iteration_start = offset_start_of_scenario
    number_of_scenes = iteration_end - iteration_start
    trajectory_poses = np.zeros((number_of_scenes, Trajectory.state_size()))
    for index, index_in_scenario in enumerate(np.arange(iteration_start, iteration_end)):
        ego_pose = scenario.get_ego_state_at_iteration(index_in_scenario)
        trajectory_poses[index] = ego_pose_to_array(ego_pose)
    if subtract_initial_pose_offset:
        initial_offset = extract_initial_offset(scenario)
        trajectory_poses = trajectory_poses - initial_offset
    return Trajectory(data=trajectory_poses.astype(np.float32))

class TestCollate(unittest.TestCase):
    """Test feature collation functionality."""

    def test_list_as_batch(self) -> None:
        """
        Test collating lists
        """
        single_feature1: FeaturesType = {'VectorMap': DummyVectorMapFeature(data1=[torch.zeros((13, 3))], data2=[torch.zeros((13, 3))], data3=[{'test': torch.zeros((13, 3))}])}
        single_targets1: FeaturesType = {'Trajectory': Trajectory(data=torch.zeros((12, 3)))}
        singe_scenario: ScenarioListType = [CachedScenario(log_name='', token='', scenario_type='')]
        to_be_batched = [(single_feature1, single_targets1, singe_scenario), (single_feature1, single_targets1, singe_scenario)]
        collate = FeatureCollate()
        features, targets, scenarios = collate(to_be_batched)
        vector_map: DummyVectorMapFeature = features['VectorMap']
        self.assertEqual(vector_map.num_of_batches, 2)
        self.assertEqual(len(vector_map.data1), 2)
        self.assertEqual(vector_map.data1[0].shape, (13, 3))
        trajectory: Trajectory = targets['Trajectory']
        self.assertEqual(trajectory.data.shape, (2, 12, 3))
        self.assertEqual(len(scenarios), 2)

    def test_collate(self) -> None:
        """
        Test collating features
        """
        single_feature1: FeaturesType = {'Trajectory': Trajectory(data=torch.zeros((12, 3))), 'Raster': Raster(data=torch.zeros((244, 244, 3))), 'DummyVectorMapFeature': DummyVectorMapFeature(data1=[torch.zeros((13, 3))], data2=[torch.zeros((13, 3))], data3=[{'test': torch.zeros((13, 3))}])}
        single_targets1: FeaturesType = {'Trajectory': Trajectory(data=torch.zeros((12, 3))), 'Trajectory2': Trajectory(data=torch.zeros((12, 3)))}
        single_feature2: FeaturesType = {'Trajectory': Trajectory(data=torch.zeros((12, 3))), 'Raster': Raster(data=torch.zeros((244, 244, 3))), 'DummyVectorMapFeature': DummyVectorMapFeature(data1=[torch.zeros((13, 3))], data2=[torch.zeros((13, 3))], data3=[{'test': torch.zeros((13, 3))}])}
        single_targets2: FeaturesType = {'Trajectory': Trajectory(data=torch.zeros((12, 3))), 'Trajectory2': Trajectory(data=torch.zeros((12, 3)))}
        single_feature3: FeaturesType = {'Trajectory': Trajectory(data=torch.zeros((12, 3))), 'Raster': Raster(data=torch.zeros((244, 244, 3))), 'DummyVectorMapFeature': DummyVectorMapFeature(data1=[torch.zeros((13, 3))], data2=[torch.zeros((13, 3))], data3=[{'test': torch.zeros((13, 3))}])}
        single_targets3: FeaturesType = {'Trajectory': Trajectory(data=torch.zeros((12, 3))), 'Trajectory2': Trajectory(data=torch.zeros((12, 3)))}
        singe_scenario: ScenarioListType = [CachedScenario(log_name='', token='', scenario_type='')]
        to_be_batched = [(single_feature1, single_targets1, singe_scenario), (single_feature2, single_targets2, singe_scenario), (single_feature3, single_targets3, singe_scenario)]
        collate = FeatureCollate()
        features, targets, scenarios = collate(to_be_batched)
        self.assertEqual(features['Trajectory'].data.shape, (3, 12, 3))
        self.assertEqual(features['Raster'].data.shape, (3, 244, 244, 3))
        self.assertEqual(features['DummyVectorMapFeature'].num_of_batches, 3)
        self.assertEqual(targets['Trajectory'].data.shape, (3, 12, 3))
        self.assertEqual(len(scenarios), 3)

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

class MockTorchModuleWrapperTrajectoryPredictor(TorchModuleWrapper):
    """
    A simple implementation of the TorchModuleWrapper interface for use with unit tests.
    It validates the input tensor, and returns a trajectory object.
    """

    def __init__(self, future_trajectory_sampling: TrajectorySampling, feature_builders: List[AbstractFeatureBuilder], target_builders: List[AbstractTargetBuilder], raise_on_builder_access: bool=False, raise_on_forward: bool=False, expected_forward_tensor: Optional[torch.Tensor]=None, data_tensor_to_return: Optional[torch.Tensor]=None) -> None:
        """
        The init method.
        :param future_trajectory_sampling: The TrajectorySampling to use.
        :param feature_builders: The feature builders used by the model.
        :param target_builders: The target builders used by the model.
        :param raise_on_builder_access: If set, an exeption will be raised if the builders are accessed.
        :param raise_on_forward: If set, an exception will be raised if the forward function is called.
        :param expected_forward_tensor: The tensor that is expected to be provided to to the forward function.
        :param data_tensor_to_return: The tensor that expected to be returned from the forward function.
        """
        super().__init__(future_trajectory_sampling, feature_builders, target_builders)
        self.raise_on_builder_access = raise_on_builder_access
        self.raise_on_forward = raise_on_forward
        self.expected_forward_tensor = expected_forward_tensor
        self.data_tensor_to_return = data_tensor_to_return
        if not self.raise_on_builder_access:
            if self.feature_builders is None or len(self.feature_builders) == 0:
                raise ValueError(textwrap.dedent('\n                    raise_on_builder_access set to False with None or 0-length feature builders.\n                    This is likely a misconfigured unit test.\n                    '))
            if self.target_builders is None or len(self.target_builders) == 0:
                raise ValueError(textwrap.dedent('\n                    raise_on_builder_access set to False with None or 0-length target builders.\n                    This is likely a misconfigured unit test.\n                    '))
        if not self.raise_on_forward:
            if self.expected_forward_tensor is None:
                raise ValueError(textwrap.dedent('\n                    raise_on_forward set to false with None expected_forward_tensor.\n                    This is likely a misconfigured unit test.\n                    '))
            if self.data_tensor_to_return is None:
                raise ValueError(textwrap.dedent('\n                    raise_on_forward set to false with None data_tensor_to_return.\n                    This is likely a misconfigured unit test.\n                    '))

    def get_list_of_required_feature(self) -> List[AbstractFeatureBuilder]:
        """
        Implemented. See interface.
        """
        if self.raise_on_builder_access:
            raise ValueError('get_list_of_required_feature() called when raise_on_builder_access set.')
        result: List[AbstractFeatureBuilder] = TorchModuleWrapper.get_list_of_required_feature(self)
        return result

    def get_list_of_computed_target(self) -> List[AbstractTargetBuilder]:
        """
        Implemented. See interface.
        """
        if self.raise_on_builder_access:
            raise ValueError('get_list_of_computed_target() called when raise_on_builder_access set.')
        result: List[AbstractTargetBuilder] = TorchModuleWrapper.get_list_of_computed_target(self)
        return result

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Implemented. See interface.
        """
        if self.raise_on_forward:
            raise ValueError('forward() called when raise_on_forward set.')
        self._validate_input_feature(features)
        return {'trajectory': Trajectory(data=self.data_tensor_to_return)}

    def _validate_input_feature(self, features: FeaturesType) -> None:
        """
        Validates that the proper feature is provided.
        Raises an exception if it is not.
        :param features: The feature provided to the model.
        """
        if 'MockFeature' not in features:
            raise ValueError(f'MockFeature not in provided features. Available keys: {sorted(list(features.keys()))}')
        if len(features) != 1:
            raise ValueError(f'Expected a single feature. Instead got {len(features)}: {sorted(list(features.keys()))}')
        mock_feature = features['MockFeature']
        if not isinstance(mock_feature, MockFeature):
            raise ValueError(f'Expected feature of type MockFeature, but got {type(mock_feature)}')
        mock_feature_data = mock_feature.data
        torch.testing.assert_close(mock_feature_data, self.expected_forward_tensor)

class TestTrajectory(unittest.TestCase):
    """Test trajectory target representation."""

    def setUp(self) -> None:
        """Set up test case."""
        self.data = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        self.batched_data = default_collate([self.data, self.data])
        self.batched_trajectory = Trajectory(data=self.batched_data)

    def test_batches(self) -> None:
        """
        Test the number of batches in trajectory
        """
        self.assertEqual(self.batched_trajectory.num_batches, 2)
        self.assertEqual(Trajectory(data=self.data).num_batches, None)

    def test_extend_trajectory(self) -> None:
        """
        Test extending trajectory by a new state
        """
        feature_builder = Trajectory(data=torch.zeros((30, 10, 3)))
        new_state = torch.zeros((30, 3)).unsqueeze(1)
        new_trajectory = Trajectory.append_to_trajectory(feature_builder, new_state)
        self.assertEqual(feature_builder.num_of_iterations + 1, new_trajectory.num_of_iterations)
        self.assertEqual(feature_builder.num_batches, 30)
        self.assertEqual(new_trajectory.num_batches, 30)

    def test_extract_trajectory(self) -> None:
        """
        Test extracting part of a trajectory
        """
        extracted = self.batched_trajectory.extract_trajectory_between(0, 4)
        self.assertEqual(extracted.data.shape, (2, 4, 3))
        self.assertAlmostEqual(extracted.data[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(extracted.data[0, -1, 0].item(), 3.0)
        state_at = self.batched_trajectory.state_at_index(3)
        state_at = state_at.unsqueeze(1)
        self.assertEqual(state_at.shape, (2, 1, 3))
        self.assertAlmostEqual(state_at[0, 0, 0], 3)

class TestTrajectories(unittest.TestCase):
    """Test trajectories target representation."""

    def setUp(self) -> None:
        """Set up test case."""
        data = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        trajectory = Trajectory(data=data)
        self.trajectories = Trajectories(trajectories=[trajectory])

    def test_serialize_deserialize(self) -> None:
        """Test that serialization and deserialization work, and the resulting data matches."""
        serialized = self.trajectories.serialize()
        deserialized = Trajectories.deserialize(serialized)
        self.assertTrue(torch.allclose(self.trajectories.trajectories[0].data, deserialized.trajectories[0].data))

def create_scenario_from_paths(paths: List[Path]) -> List[AbstractScenario]:
    """
    Create scenario objects from a list of cache paths in the format of ".../log_name/scenario_token".
    :param paths: List of paths to load scenarios from.
    :return: List of created scenarios.
    """
    scenarios = [CachedScenario(log_name=path.parent.parent.name, token=path.name, scenario_type=path.parent.name) for path in paths]
    return scenarios

class TestNuPlanScenarioFilterUtils(unittest.TestCase):
    """
    Tests scenario filter utils for NuPlan
    """

    def _get_mock_scenario_dict(self) -> Dict[str, List[CachedScenario]]:
        """Gets mock scenario dict."""
        return {DEFAULT_SCENARIO_NAME: [CachedScenario(log_name='log/name', token=DEFAULT_SCENARIO_NAME, scenario_type=DEFAULT_SCENARIO_NAME) for i in range(500)], 'lane_following_with_lead': [CachedScenario(log_name='log/name', token='lane_following_with_lead', scenario_type='lane_following_with_lead') for i in range(80)], 'unprotected_left_turn': [CachedScenario(log_name='log/name', token='unprotected_left_turn', scenario_type='unprotected_left_turn') for i in range(120)]}

    def _get_mock_nuplan_scenario_dict_for_timestamp_filtering(self) -> Dict[str, List[CachedScenario]]:
        """Gets mock scenario dict."""
        mock_scenario_dict = {DEFAULT_SCENARIO_NAME: [Mock(NuPlanScenario) for _ in range(0, 100, 3)], 'lane_following_with_lead': [Mock(NuPlanScenario) for _ in range(0, 100, 6)], 'lane_following_without_lead': [Mock(NuPlanScenario) for _ in range(3)]}
        for i in range(0, len(mock_scenario_dict[DEFAULT_SCENARIO_NAME]) * int(1000000.0), int(1000000.0)):
            mock_scenario_dict[DEFAULT_SCENARIO_NAME][int(i / 1000000.0)]._initial_lidar_timestamp = i * 3
        for i in range(0, len(mock_scenario_dict['lane_following_with_lead']) * int(1000000.0), int(1000000.0)):
            mock_scenario_dict['lane_following_with_lead'][int(i / 1000000.0)]._initial_lidar_timestamp = i * 6
        mock_scenario_dict['lane_following_without_lead'][0]._initial_lidar_timestamp = 5.0 * int(1000000.0)
        mock_scenario_dict['lane_following_without_lead'][1]._initial_lidar_timestamp = 100.0 * int(1000000.0)
        mock_scenario_dict['lane_following_without_lead'][2]._initial_lidar_timestamp = 6.0 * int(1000000.0)
        return mock_scenario_dict

    def _get_mock_worker_map(self) -> Callable[..., List[Any]]:
        """
        Gets mock worker_map function.
        """

        def mock_worker_map(worker: WorkerPool, fn: Callable[..., List[Any]], input_objects: List[Any]) -> List[Any]:
            """
            Mock function for worker_map
            :param worker: Worker pool
            :param fn: Callable function
            :param input_objects: List of objects to be used as input
            :return: List of output objects
            """
            return fn(input_objects)
        return mock_worker_map

    def test_filter_total_num_scenarios_int_max_scenarios_requires_removing_known_scenario_types(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 100
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)
        self.assertTrue(len(final_scenario_dict['lane_following_with_lead']) < len(mock_scenario_dict['lane_following_with_lead']))
        self.assertTrue(len(final_scenario_dict['unprotected_left_turn']) < len(mock_scenario_dict['unprotected_left_turn']))
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), limit_total_scenarios)

    def test_filter_total_num_scenarios_int_max_scenarios_less_than_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 300
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertNotEqual(final_scenario_dict[DEFAULT_SCENARIO_NAME], mock_scenario_dict[DEFAULT_SCENARIO_NAME])
        self.assertEqual(final_scenario_dict['lane_following_with_lead'], mock_scenario_dict['lane_following_with_lead'])
        self.assertEqual(final_scenario_dict['unprotected_left_turn'], mock_scenario_dict['unprotected_left_turn'])
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), limit_total_scenarios)

    def test_filter_total_num_scenarios_int_max_scenarios_more_than_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 800
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertDictEqual(final_scenario_dict, mock_scenario_dict)

    def test_filter_total_num_scenarios_float_requires_removing_known_scenario_types(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an float, the actual number of scenarios,
        where the number of scenarios required is requires reomving known scenario types.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.2
        randomize = True
        final_num_of_scenarios = int(limit_total_scenarios * sum((len(scenarios) for scenarios in mock_scenario_dict.values())))
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)
        self.assertTrue(len(final_scenario_dict['lane_following_with_lead']) < len(mock_scenario_dict['lane_following_with_lead']))
        self.assertTrue(len(final_scenario_dict['unprotected_left_turn']) < len(mock_scenario_dict['unprotected_left_turn']))
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), final_num_of_scenarios)

    def test_filter_total_num_scenarios_float_removes_only_default_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an float, the actual number of scenarios,
        where the number of scenarios required is requires reomving known scenario types.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.5
        randomize = True
        final_num_of_scenarios = int(limit_total_scenarios * sum((len(scenarios) for scenarios in mock_scenario_dict.values())))
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertNotEqual(final_scenario_dict[DEFAULT_SCENARIO_NAME], mock_scenario_dict[DEFAULT_SCENARIO_NAME])
        self.assertEqual(final_scenario_dict['lane_following_with_lead'], mock_scenario_dict['lane_following_with_lead'])
        self.assertEqual(final_scenario_dict['unprotected_left_turn'], mock_scenario_dict['unprotected_left_turn'])
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), final_num_of_scenarios)

    def test_remove_all_scenarios_int_limit_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to 0. This should raise an assertion error.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0
        randomize = True
        with self.assertRaises(AssertionError):
            filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)

    def test_remove_all_scenarios_float_limit_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to 0. This should raise an assertion error.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.0
        randomize = True
        with self.assertRaises(AssertionError):
            filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)

    def test_remove_exactly_all_default_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to number of known scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 200
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)
        self.assertEqual(len(final_scenario_dict['lane_following_with_lead']), len(mock_scenario_dict['lane_following_with_lead']))
        self.assertEqual(len(final_scenario_dict['unprotected_left_turn']), len(mock_scenario_dict['unprotected_left_turn']))
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), limit_total_scenarios)

    def test_filter_scenarios_by_timestamp(self) -> None:
        """
        Tests filter_scenarios_by_timestamp with default threshold
        """
        mock_worker_map = self._get_mock_worker_map()
        mock_nuplan_scenario_dict = self._get_mock_nuplan_scenario_dict_for_timestamp_filtering()
        with patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.worker_map', mock_worker_map):
            final_scenario_dict = filter_scenarios_by_timestamp(mock_nuplan_scenario_dict.copy())
            self.assertEqual(len(final_scenario_dict['lane_following_with_lead']), len(mock_nuplan_scenario_dict['lane_following_with_lead']))
            self.assertEqual(len(final_scenario_dict[DEFAULT_SCENARIO_NAME]), len(mock_nuplan_scenario_dict[DEFAULT_SCENARIO_NAME]) * 0.5)
            self.assertEqual(len(final_scenario_dict['lane_following_without_lead']), len(mock_nuplan_scenario_dict['lane_following_without_lead']) - 1)

    def test_filter_fraction_lidarpc_tokens_in_set(self) -> None:
        """
        Test filter_fraction_lidarpc_tokens_in_set with fractional thresholds {0, 0.5, 1}.
        """
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
        mock_nuplan_scenarios = []
        for start_letter in range(4):
            mock_nuplan_scenario = Mock(NuPlanScenario)
            mock_nuplan_scenario.get_scenario_tokens.return_value = set(alphabet[start_letter:start_letter + 3])
            mock_nuplan_scenarios.append(mock_nuplan_scenario)
        full_intersection_scenario, two_intersection_scenario, one_intersection_scenario, no_intersection_scenario = mock_nuplan_scenarios
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json_path = Path(tmp_dir) / 'tmp_token_set.json'
            json.dump(['a', 'b', 'c'], open(tmp_json_path, 'w'))
            scenario_dict = {'on_pickup_dropoff': [no_intersection_scenario, one_intersection_scenario]}
            self.assertEqual(filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 0), {'on_pickup_dropoff': [one_intersection_scenario]})
            scenario_dict['on_pickup_dropoff'] = [one_intersection_scenario, two_intersection_scenario]
            self.assertEqual(filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 0.5), {'on_pickup_dropoff': [two_intersection_scenario]})
            scenario_dict['on_pickup_dropoff'] = [two_intersection_scenario, full_intersection_scenario]
            self.assertEqual(filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 1), {'on_pickup_dropoff': [full_intersection_scenario]})

    def test_filter_non_stationary_ego(self) -> None:
        """Test filter_non_stationary_ego with 0.5m displacement threshold"""
        stationary_ego_pudo_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.01, y=0.0))
        mobile_ego_pudo_scenario = MockAbstractScenario()
        scenario_dict = {'on_pickup_dropoff': [stationary_ego_pudo_scenario, mobile_ego_pudo_scenario]}
        filtered_scenario_dict = filter_non_stationary_ego(scenario_dict, minimum_threshold=0.5)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [mobile_ego_pudo_scenario])

    def test_filter_ego_starts(self) -> None:
        """Test filter_ego_starts with 0.1 m/s speed threshold"""
        slow_acceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=0.01, y=0.0), time_step=1)
        fast_acceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=1, y=0.0), time_step=1)
        scenario_dict = {'on_pickup_dropoff': [slow_acceleration_scenario, fast_acceleration_scenario]}
        filtered_scenario_dict = filter_ego_starts(scenario_dict, speed_threshold=0.1, speed_noise_tolerance=0.1)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [fast_acceleration_scenario])

    def test_filter_ego_stops(self) -> None:
        """Test filter_ego_stops with 0.1 m/s speed threshold"""
        slow_deceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=1.0, y=0.0), fixed_acceleration=StateVector2D(x=-0.01, y=0.0), time_step=1)
        fast_deceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=1.0, y=0.0), fixed_acceleration=StateVector2D(x=-1 / 9, y=0.0), time_step=1)
        scenario_dict = {'on_pickup_dropoff': [slow_deceleration_scenario, fast_deceleration_scenario]}
        filtered_scenario_dict = filter_ego_stops(scenario_dict, speed_threshold=0.1, speed_noise_tolerance=0.1)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [fast_deceleration_scenario])

    def test_ego_startstop_noise_tolerance(self) -> None:
        """Test filter_ego_starts with ego barely crossing speed threshold and noise tolerance higher than threshold"""
        fast_enough_acceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=0.11, y=0.0), time_step=1)
        scenario_dict = {'on_pickup_dropoff': [fast_enough_acceleration_scenario]}
        filtered_scenario_dict = filter_ego_starts(scenario_dict, speed_threshold=1, speed_noise_tolerance=2)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [])

    def test_filter_ego_has_route(self) -> None:
        """
        Test filter_ego_has_route with one route roadblock in the VectorMap (True case),
        and with no route-intersecting roadblocks (False case).
        """
        map_radius = 35
        scenario = MockAbstractScenario()
        scenario_dict = {'on_pickup_dropoff': [scenario]}
        with patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_neighbor_vector_map') as get_neighbor_vector_map:
            get_neighbor_vector_map.return_value = (None, None, None, None, LaneSegmentRoadBlockIDs(['a', 'b', 'c']))
            with patch.object(scenario, 'get_route_roadblock_ids') as get_route_roadblock_ids:
                get_route_roadblock_ids.return_value = ['d', 'e', 'a']
                self.assertEqual(filter_ego_has_route(scenario_dict, map_radius)['on_pickup_dropoff'], [scenario])
                get_route_roadblock_ids.return_value = ['d', 'e', 'f']
                self.assertEqual(filter_ego_has_route(scenario_dict, map_radius)['on_pickup_dropoff'], [])

class TestCachedScenario(unittest.TestCase):
    """
    Test suite for CachedScenario
    """

    def _make_cached_scenario(self) -> CachedScenario:
        return CachedScenario(log_name='log/name', token='token', scenario_type='type')

    def test_token(self) -> None:
        """
        Test the token method.
        """
        scenario = self._make_cached_scenario()
        self.assertEqual('token', scenario.token)

    def test_log_name(self) -> None:
        """
        Test the log_name method.
        """
        scenario = self._make_cached_scenario()
        self.assertEqual('log/name', scenario.log_name)

    def test_scenario_name_raises(self) -> None:
        """
        Test that the scenario_name method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.scenario_name

    def test_ego_vehicle_parameters_raises(self) -> None:
        """
        Test that the ego_vehicle_parameters method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.ego_vehicle_parameters

    def test_scenario_type(self) -> None:
        """
        Test that the scenario_type method returns scenario_type.
        """
        scenario = self._make_cached_scenario()
        self.assertEqual('type', scenario.scenario_type)

    def test_map_api_raises(self) -> None:
        """
        Test that the map_api method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.map_api

    def test_database_interval_raises(self) -> None:
        """
        Test that the database_interval method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.database_interval

    def test_get_number_of_iterations_raises(self) -> None:
        """
        Test that the get_number_of_iterations method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_number_of_iterations()

    def test_get_time_point_raises(self) -> None:
        """
        Test that the get_time_point method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_time_point()

    def test_get_lidar_to_ego_transform_raises(self) -> None:
        """
        Test that the get_lidar_to_ego_transform method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_lidar_to_ego_transform()

    def test_get_mission_goal_raises(self) -> None:
        """
        Test that the get_mission_goal method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_mission_goal()

    def test_get_route_roadblock_ids_raises(self) -> None:
        """
        Test that the get_route_roadblock_ids method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_route_roadblock_ids()

    def test_get_expert_goal_state_raises(self) -> None:
        """
        Test that the get_expert_goal_state method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_expert_goal_state()

    def test_get_tracked_objects_at_iteration_raises(self) -> None:
        """
        Test that the get_tracked_objects_at_iteration method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_tracked_objects_at_iteration(0)

    def test_get_sensors_at_iteration_raises(self) -> None:
        """
        Test that the get_sensors_at_iteration method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_sensors_at_iteration(0)

    def test_get_ego_state_at_iteration_raises(self) -> None:
        """
        Test that the get_ego_state_at_iteration method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_ego_state_at_iteration(0)

    def test_get_traffic_light_status_at_iteration_raises(self) -> None:
        """
        Test that the get_traffic_light_status_at_iteration method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_traffic_light_status_at_iteration(0)

    def test_get_future_timestamps_raises(self) -> None:
        """
        Test that the get_future_timestamps method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_future_timestamps(0, 0, 0)

    def test_get_past_timestamps_raises(self) -> None:
        """
        Test that the get_past_timestamps method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_past_timestamps(0, 0, 0)

    def test_ego_future_trajectory_raises(self) -> None:
        """
        Test that the ego_future_trajectory method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_ego_future_trajectory(0, 0, 0)

    def test_get_ego_past_trajectory_raises(self) -> None:
        """
        Test that the get_ego_past_trajectory method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_ego_past_trajectory(0, 0, 0)

    def test_get_past_sensors_raises(self) -> None:
        """
        Test that the get_past_sensors method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_past_sensors(0, 0, 0)

    def test_get_past_tracked_objects_raises(self) -> None:
        """
        Test that the get_past_tracked_objects method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_past_tracked_objects(0, 0, 0)

    def test_get_future_tracked_objects_raises(self) -> None:
        """
        Test that the get_future_tracked_objects method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_future_tracked_objects(0, 0, 0)

