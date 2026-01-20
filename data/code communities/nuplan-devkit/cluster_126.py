# Cluster 126

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

class TestSinusoidalPositionalEmbedding(unittest.TestCase):
    """Test SinusoidalPositionalEmbedding layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.embedding_size = 256
        self.model = SinusoidalPositionalEmbedding(self.embedding_size)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        batch_size = 2
        num_elements = 10
        num_points = 20
        inputs = torch.zeros((batch_size, num_elements, num_points, self.embedding_size))
        output = self.model.forward(inputs)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_points, 1, self.embedding_size))

class TestTypeEmbedding(unittest.TestCase):
    """Test TypeEmbedding layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.embedding_dim = 256
        self.feature_types = {'NONE': -1, 'EGO': 0, 'VEHICLE': 1, 'BICYCLE': 2, 'PEDESTRIAN': 3, 'LANE': 4, 'STOP_LINE': 5, 'CROSSWALK': 6, 'LEFT_BOUNDARY': 7, 'RIGHT_BOUNDARY': 8, 'ROUTE_LANES': 9}
        self.model = TypeEmbedding(self.embedding_dim, self.feature_types)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        device = torch.device('cpu')
        batch_size = 2
        max_agents = 30
        agent_features = ['VEHICLE', 'BICYCLE', 'PEDESTRIAN']
        map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        max_elements = {'LANE': 30, 'LEFT_BOUNDARY': 30, 'RIGHT_BOUNDARY': 30, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 30}
        num_elements = 1 + max_agents * len(agent_features) + sum(max_elements.values())
        output = self.model.forward(batch_size, max_agents, agent_features, map_features, max_elements, device)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, num_elements, self.embedding_dim))

class TestLocalSubGraph(unittest.TestCase):
    """Test LocalSubGraph layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.num_layers = 3
        self.dim_in = 256
        self.model = LocalSubGraph(self.num_layers, self.dim_in)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        batch_size = 2
        num_elements = 10
        num_points = 20
        inputs = torch.zeros((batch_size, num_elements, num_points, self.dim_in), dtype=torch.float32)
        invalid_mask = torch.zeros((batch_size, num_elements, num_points), dtype=torch.bool)
        pos_enc = torch.zeros((1, 1, num_points, self.dim_in), dtype=torch.float32)
        output = self.model.forward(inputs, invalid_mask, pos_enc)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, num_elements, self.dim_in))

class TestMultiheadAttentionGlobalHead(unittest.TestCase):
    """Test MultiheadAttentionGlobalHead layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.global_embedding_size = 256
        self.num_timesteps = 12
        self.num_outputs = 3
        self.model = MultiheadAttentionGlobalHead(self.global_embedding_size, self.num_timesteps, self.num_outputs)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        batch_size = 2
        num_elements = 10
        inputs = torch.zeros((num_elements, batch_size, self.global_embedding_size), dtype=torch.float32)
        type_embedding = torch.ones((num_elements, batch_size, self.global_embedding_size), dtype=torch.long)
        invalid_mask = torch.zeros((batch_size, num_elements), dtype=torch.bool)
        output, attns = self.model.forward(inputs, type_embedding, invalid_mask)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, self.num_timesteps, self.num_outputs))
        self.assertIsInstance(attns, torch.Tensor)
        assert attns is not None
        self.assertEqual(attns.shape, (batch_size, 1, num_elements))

class TestVectorSetMapFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs map features in vector set format."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = MockAbstractScenario()
        self.batch_size = 1
        self.radius = 35
        self.interpolation_method = 'linear'
        self.map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        self.max_elements = {'LANE': 30, 'LEFT_BOUNDARY': 30, 'RIGHT_BOUNDARY': 30, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 30}
        self.max_points = {'LANE': 20, 'LEFT_BOUNDARY': 20, 'RIGHT_BOUNDARY': 20, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 20}
        self.feature_builder = VectorSetMapFeatureBuilder(map_features=self.map_features, max_elements=self.max_elements, max_points=self.max_points, radius=self.radius, interpolation_method=self.interpolation_method)

    def test_vector_set_map_feature_builder(self) -> None:
        """
        Tests VectorSetMapFeatureBuilder.
        """
        self.assertEqual(self.feature_builder.get_feature_type(), VectorSetMap)
        features = self.feature_builder.get_features_from_scenario(self.scenario)
        self.assertEqual(type(features), VectorSetMap)
        self.assertEqual(features.batch_size, self.batch_size)
        for feature_name in self.map_features:
            self.assertEqual(features.coords[feature_name][0].shape, (self.max_elements[feature_name], self.max_points[feature_name], 2))
            self.assertEqual(features.availabilities[feature_name][0].shape, (self.max_elements[feature_name], self.max_points[feature_name]))
            np.testing.assert_array_equal(features.availabilities[feature_name][0], np.zeros((self.max_elements[feature_name], self.max_points[feature_name]), dtype=np.bool_))

    def test_vector_set_map_feature_builder_simulation_and_scenario_match(self) -> None:
        """
        Tests that get_features_from_scenario and get_features_from_simulation give same results.
        """
        features = self.feature_builder.get_features_from_scenario(self.scenario)
        ego_state = self.scenario.initial_ego_state
        detections = self.scenario.initial_tracked_objects
        meta_data = PlannerInitialization(map_api=self.scenario.map_api, mission_goal=self.scenario.get_mission_goal(), route_roadblock_ids=self.scenario.get_route_roadblock_ids())
        history = SimulationHistoryBuffer.initialize_from_list(1, [ego_state], [detections], self.scenario.database_interval)
        iteration = SimulationIteration(TimePoint(0), 0)
        tl_data = self.scenario.get_traffic_light_status_at_iteration(iteration.index)
        current_input = PlannerInput(iteration=iteration, history=history, traffic_light_data=tl_data)
        features_sim = self.feature_builder.get_features_from_simulation(current_input=current_input, initialization=meta_data)
        self.assertEqual(type(features_sim), VectorSetMap)
        self.assertEqual(set(features_sim.coords.keys()), set(features.coords.keys()))
        for feature_name in features_sim.coords.keys():
            np.testing.assert_allclose(features_sim.coords[feature_name][0], features.coords[feature_name][0], atol=0.0001)
        self.assertEqual(set(features_sim.traffic_light_data.keys()), set(features.traffic_light_data.keys()))
        for feature_name in features_sim.traffic_light_data.keys():
            np.testing.assert_allclose(features_sim.traffic_light_data[feature_name][0], features.traffic_light_data[feature_name][0])
        self.assertEqual(set(features_sim.availabilities.keys()), set(features.availabilities.keys()))
        for feature_name in features_sim.availabilities.keys():
            np.testing.assert_array_equal(features_sim.availabilities[feature_name][0], features.availabilities[feature_name][0])

    def test_vector_set_map_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the VectorSetMapFeatureBuilder can be scripted properly.
        """
        self.assertEqual(self.feature_builder.get_feature_type(), VectorSetMap)
        scripted_builder = torch.jit.script(self.feature_builder)
        self.assertIsNotNone(scripted_builder)
        config = scripted_builder.precomputed_feature_config()
        self.assertTrue('initial_ego_state' in config)
        self.assertTrue('neighbor_vector_set_map' in config)
        self.assertTrue('radius' in config['neighbor_vector_set_map'])
        self.assertEqual(str(self.radius), config['neighbor_vector_set_map']['radius'])
        self.assertEqual(str(self.interpolation_method), config['neighbor_vector_set_map']['interpolation_method'])
        self.assertEqual(','.join(self.map_features), config['neighbor_vector_set_map']['map_features'])
        max_elements: List[str] = [f'{feature_name}.{feature_max_elements}' for feature_name, feature_max_elements in self.max_elements.items()]
        max_points: List[str] = [f'{feature_name}.{feature_max_points}' for feature_name, feature_max_points in self.max_points.items()]
        self.assertEqual(','.join(max_elements), config['neighbor_vector_set_map']['max_elements'])
        self.assertEqual(','.join(max_points), config['neighbor_vector_set_map']['max_points'])
        tensor_data = {'anchor_state': torch.zeros((3,))}
        for feature_name in self.map_features:
            feature_max_elements = self.max_elements[feature_name]
            feature_max_points = self.max_points[feature_name]
            tensor_data[f'coords.{feature_name}'] = torch.rand((feature_max_elements, feature_max_points, 2), dtype=torch.float64)
            tensor_data[f'traffic_light_data.{feature_name}'] = torch.zeros((feature_max_elements, feature_max_points, 4), dtype=torch.int64)
            tensor_data[f'availabilities.{feature_name}'] = torch.zeros((feature_max_elements, feature_max_points), dtype=torch.bool)
        list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_tensor_output, scripted_list_output, scripted_list_list_output = scripted_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        py_tensor_output, py_list_output, py_list_list_output = self.feature_builder.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
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

class TestGenericAgentsFeatureBuilder(unittest.TestCase):
    """Test builder that constructs agent features during training and simulation."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.batch_size = 1
        self.past_time_horizon = 4.0
        self.num_agents = 10
        self.num_past_poses = 4
        self.num_total_past_poses = self.num_past_poses + 1
        self.agent_features = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'TRAFFIC_CONE', 'BARRIER', 'CZONE_SIGN', 'GENERIC_OBJECT']
        self.tracked_object_types: List[TrackedObjectType] = []
        for feature_name in self.agent_features:
            try:
                self.tracked_object_types.append(TrackedObjectType[feature_name])
            except KeyError:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')
        self.feature_builder = GenericAgentsFeatureBuilder(self.agent_features, TrajectorySampling(num_poses=self.num_past_poses, time_horizon=self.past_time_horizon))

    def test_generic_agent_feature_builder(self) -> None:
        """
        Test GenericAgentFeatureBuilder
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents, tracked_object_types=self.tracked_object_types)
        feature = self.feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_no_agents(self) -> None:
        """
        Test when there are no agents
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=0, tracked_object_types=self.tracked_object_types)
        feature = self.feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), 0)
            self.assertEqual(feature.agents[feature_name][0].shape[1], 0)
            self.assertEqual(feature.agents[feature_name][0].shape[2], GenericAgents.agents_states_dim())

    def test_get_feature_from_simulation(self) -> None:
        """
        Test get feature from simulation
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents, tracked_object_types=self.tracked_object_types)
        mock_meta_data = PlannerInitialization(map_api=MockAbstractMap(), route_roadblock_ids=None, mission_goal=StateSE2(0, 0, 0))
        ego_past_states = list(scenario.get_ego_past_trajectory(iteration=0, num_samples=10, time_horizon=5))
        ego_initial_state = scenario.initial_ego_state
        ego_history = ego_past_states + [ego_initial_state]
        past_observations = list(scenario.get_past_tracked_objects(iteration=0, num_samples=10, time_horizon=5))
        initial_observation = scenario.initial_tracked_objects
        observation_history = past_observations + [initial_observation]
        history = SimulationHistoryBuffer.initialize_from_list(len(ego_history), ego_history, observation_history, scenario.database_interval)
        current_input = PlannerInput(iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history, traffic_light_data=scenario.get_traffic_light_status_at_iteration(0))
        feature = self.feature_builder.get_features_from_simulation(current_input=current_input, initialization=mock_meta_data)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_agents_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the Generic Agents Feature Builder scripts properly
        """
        config = self.feature_builder.precomputed_feature_config()
        for expected_key in ['past_ego_states', 'past_time_stamps']:
            self.assertTrue(expected_key in config)
            config_dict = config[expected_key]
            self.assertTrue(len(config_dict) == 3)
            self.assertEqual(0, int(config_dict['iteration']))
            self.assertEqual(self.num_past_poses, int(config_dict['num_samples']))
            self.assertEqual(self.past_time_horizon, int(float(config_dict['time_horizon'])))
        tracked_objects_config_dict = config['past_tracked_objects']
        self.assertTrue(len(tracked_objects_config_dict) == 4)
        self.assertEqual(0, int(tracked_objects_config_dict['iteration']))
        self.assertEqual(self.num_past_poses, int(tracked_objects_config_dict['num_samples']))
        self.assertEqual(self.past_time_horizon, int(float(tracked_objects_config_dict['time_horizon'])))
        self.assertTrue('agent_features' in tracked_objects_config_dict)
        self.assertEqual(','.join(self.agent_features), tracked_objects_config_dict['agent_features'])
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
        list_tensor_data = {f'past_tracked_objects.{feature_name}': past_tracked_objects for feature_name in self.agent_features}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_builder = torch.jit.script(self.feature_builder)
        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        py_tensors, py_list_tensors, py_list_list_tensors = self.feature_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
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
                torch.testing.assert_allclose(py, scripted, atol=0.05, rtol=0.05)
        self.assertEqual(0, len(scripted_list_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))

