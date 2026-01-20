# Cluster 143

def build_ego_features(ego_trajectory: List[EgoState], reverse: bool=False) -> FeatureDataType:
    """
    Build agent features from the ego and agents trajectory
    :param ego_trajectory: ego trajectory comprising of EgoState [num_frames]
    :param reverse: if True, the last element in the list will be considered as the present ego state
    :return: ego_features: <np.ndarray: num_frames, 3>
                         The num_frames includes both present and past/future frames.
                         The last dimension is the ego pose (x, y, heading) at time t.
    """
    if reverse:
        anchor_ego_state = ego_trajectory[-1]
    else:
        anchor_ego_state = ego_trajectory[0]
    ego_poses = [ego_state.rear_axle for ego_state in ego_trajectory]
    ego_relative_poses = convert_absolute_to_relative_poses(anchor_ego_state.rear_axle, ego_poses)
    return ego_relative_poses

def convert_absolute_to_relative_poses(origin_absolute_state: StateSE2, absolute_states: List[StateSE2]) -> npt.NDArray[np.float32]:
    """
    Computes the relative poses from a list of absolute states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param absolute_states: list of absolute poses.
    :return: list of relative poses as numpy array.
    """
    relative_states = _convert_absolute_to_relative_states(origin_absolute_state, absolute_states)
    relative_poses: npt.NDArray[np.float32] = np.asarray([state.serialize() for state in relative_states]).astype(np.float32)
    return relative_poses

def build_ego_center_features(ego_trajectory: List[EgoState], reverse: bool=False) -> FeatureDataType:
    """
    Build agent features from the ego and agents trajectory, using center of ego OrientedBox as reference points.
    :param ego_trajectory: ego trajectory comprising of EgoState [num_frames]
    :param reverse: if True, the last element in the list will be considered as the present ego state
    :return: ego_features
             ego_features: <np.ndarray: num_frames, 3>
                         The num_frames includes both present and past/future frames.
                         The last dimension is the ego pose (x, y, heading) at time t.
    """
    if reverse:
        anchor_ego_state = ego_trajectory[-1]
    else:
        anchor_ego_state = ego_trajectory[0]
    ego_poses = [ego_state.center for ego_state in ego_trajectory]
    ego_relative_poses = convert_absolute_to_relative_poses(anchor_ego_state.center, ego_poses)
    return ego_relative_poses

def _create_ego_trajectory(num_frames: int) -> List[EgoState]:
    """
    Generate a dummy ego trajectory
    :param num_frames: length of the trajectory to be generate
    """
    return [EgoState.build_from_rear_axle(StateSE2(step, step, step), rear_axle_velocity_2d=StateVector2D(step, step), rear_axle_acceleration_2d=StateVector2D(step, step), tire_steering_angle=step, time_point=TimePoint(step), vehicle_parameters=get_pacifica_parameters()) for step in range(num_frames)]

class TestAgentsFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs features with vectorized agent information."""

    def setUp(self) -> None:
        """Set up test case."""
        self.num_frames = 8
        self.num_agents = 10
        self.num_missing_agents = 2
        self.agent_trajectories = [*_create_tracked_objects(5, self.num_agents), *_create_tracked_objects(3, self.num_agents - self.num_missing_agents)]
        self.time_stamps = [TimePoint(step) for step in range(self.num_frames)]

    def test_build_ego_features(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        ego_trajectory = _create_ego_trajectory(num_frames)
        ego_features = build_ego_features(ego_trajectory)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(np.allclose(ego_features[0], np.array([0, 0, 0])))
        ego_features_reversed = build_ego_features(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(np.allclose(ego_features_reversed[-1], np.array([0, 0, 0])))

    def test_extract_and_pad_agent_poses(self) -> None:
        """
        Test when there is agent pose trajectory is incomplete
        """
        padded_poses, availability = extract_and_pad_agent_poses(self.agent_trajectories)
        availability = np.asarray(availability)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses])
        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)
        self.assertEqual(len(availability.shape), 2)
        self.assertEqual(availability.shape[0], self.num_frames)
        self.assertEqual(availability.shape[1], self.num_agents)
        self.assertTrue(availability[:5, :].all())
        self.assertTrue(availability[:, :self.num_agents - self.num_missing_agents].all())
        self.assertTrue((~availability[5:, -self.num_missing_agents:]).all())
        padded_poses_reversed, availability_reversed = extract_and_pad_agent_poses(self.agent_trajectories[::-1], reverse=True)
        availability_reversed = np.asarray(availability_reversed)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses_reversed])
        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)
        self.assertEqual(len(availability_reversed.shape), 2)
        self.assertEqual(availability_reversed.shape[0], self.num_frames)
        self.assertEqual(availability_reversed.shape[1], self.num_agents)
        self.assertTrue(availability_reversed[-5:, :].all())
        self.assertTrue(availability_reversed[:, :self.num_agents - self.num_missing_agents].all())
        self.assertTrue((~availability_reversed[:3, -self.num_missing_agents:]).all())

    def test_extract_and_pad_agent_sizes(self) -> None:
        """
        Test when there is agent size trajectory is incomplete
        """
        padded_sizes, _ = extract_and_pad_agent_sizes(self.agent_trajectories)
        stacked_sizes = np.stack(padded_sizes)
        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)
        padded_sizes_reversed, _ = extract_and_pad_agent_sizes(self.agent_trajectories[::-1], reverse=True)
        stacked_sizes = np.stack(padded_sizes_reversed)
        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)

    def test_extract_and_pad_agent_velocities(self) -> None:
        """
        Test when there is agent velocity trajectory is incomplete
        """
        padded_velocities, _ = extract_and_pad_agent_velocities(self.agent_trajectories)
        stacked_velocities = np.stack([[agent.serialize() for agent in frame] for frame in padded_velocities])
        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)
        padded_velocities_reversed, _ = extract_and_pad_agent_velocities(self.agent_trajectories[::-1], reverse=True)
        stacked_velocities = np.stack([[agent.serialize() for agent in frame] for frame in padded_velocities_reversed])
        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)

    def test_compute_yaw_rate_from_states(self) -> None:
        """
        Test computing yaw from the agent pose trajectory
        """
        padded_poses, _ = extract_and_pad_agent_poses(self.agent_trajectories)
        yaw_rates = compute_yaw_rate_from_states(padded_poses, self.time_stamps)
        self.assertEqual(yaw_rates.transpose().shape[0], self.num_frames)
        self.assertEqual(yaw_rates.transpose().shape[1], self.num_agents)

    def test_filter_agents(self) -> None:
        """
        Test agent filtering
        """
        num_frames = 8
        num_agents = 5
        missing_agents = 2
        tracked_objects_history = [*_create_tracked_objects(num_frames=5, num_agents=num_agents, object_type=TrackedObjectType.VEHICLE), *_create_tracked_objects(num_frames=2, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.BICYCLE), *_create_tracked_objects(num_frames=1, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.VEHICLE)]
        filtered_agents = filter_agents(tracked_objects_history)
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[0].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)
        filtered_agents = filter_agents(tracked_objects_history, reverse=True)
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[-1].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)
        tracked_objects_history = [*_create_tracked_objects(num_frames=5, num_agents=num_agents, object_type=TrackedObjectType.BICYCLE), *_create_tracked_objects(num_frames=2, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.VEHICLE), *_create_tracked_objects(num_frames=1, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.BICYCLE)]
        filtered_agents = filter_agents(tracked_objects_history, allowable_types=[TrackedObjectType.BICYCLE])
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[0].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)

    def test_build_ego_features_from_tensor(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        zeros = torch.tensor([0, 0, 0], dtype=torch.float32)
        ego_trajectory = _create_ego_trajectory_tensor(num_frames)
        ego_features = build_ego_features_from_tensor(ego_trajectory)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(torch.allclose(ego_features[0], zeros, atol=1e-07))
        ego_features_reversed = build_ego_features_from_tensor(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(torch.allclose(ego_features_reversed[-1], zeros, atol=1e-07))

    def test_build_generic_ego_features_from_tensor(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        ego_trajectory = _create_ego_trajectory_tensor(num_frames)
        ego_features = build_generic_ego_features_from_tensor(ego_trajectory)
        self.assertEqual((num_frames, GenericEgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(torch.allclose(ego_features[0], zeros, atol=1e-07))
        ego_features_reversed = build_generic_ego_features_from_tensor(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, GenericEgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(torch.allclose(ego_features_reversed[-1], zeros, atol=1e-07))

    def test_convert_absolute_quantities_to_relative(self) -> None:
        """
        Test the conversion routine between absolute and relative quantities
        """

        def get_dummy_states() -> List[torch.Tensor]:
            """
            Create a series of dummy agent tensors
            """
            dummy_agent_state = _create_tracked_object_agent_tensor(7)
            dummy_states = [dummy_agent_state + i for i in range(5)]
            return dummy_states
        zeros = torch.tensor([0, 0, 0], dtype=torch.float32)
        dummy_states = get_dummy_states()
        ego_pose = torch.tensor([4, 4, 4, 2, 2, 2, 2], dtype=torch.float32)
        transformed = convert_absolute_quantities_to_relative(dummy_states, ego_pose)
        for i in range(0, len(transformed), 1):
            should_be_zero_row = 4 - i
            check_tensor = torch.tensor([transformed[i][should_be_zero_row, AgentInternalIndex.x()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.y()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.heading()].item()], dtype=torch.float32)
            self.assertTrue(torch.allclose(check_tensor, zeros, atol=1e-07))
        dummy_states = get_dummy_states()
        ego_pose = torch.tensor([2, 2, 4, 4, 4, 4, 4], dtype=torch.float32)
        transformed = convert_absolute_quantities_to_relative(dummy_states, ego_pose)
        for i in range(0, len(transformed), 1):
            should_be_zero_row = 4 - i
            check_tensor = torch.tensor([transformed[i][should_be_zero_row, AgentInternalIndex.vx()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.vy()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.heading()].item()], dtype=torch.float32)
            self.assertTrue(torch.allclose(check_tensor, zeros, atol=1e-07))

    def test_pad_agent_states(self) -> None:
        """
        Test the pad agent states functionality
        """
        forward_dummy_states = [_create_tracked_object_agent_tensor(7), _create_tracked_object_agent_tensor(5), _create_tracked_object_agent_tensor(6)]
        padded = pad_agent_states(forward_dummy_states, reverse=False)
        self.assertTrue(len(padded) == 3)
        self.assertEqual((7, AgentInternalIndex.dim()), padded[0].shape)
        for i in range(1, len(padded)):
            self.assertTrue(torch.allclose(padded[0], padded[i]))
        backward_dummy_states = [_create_tracked_object_agent_tensor(6), _create_tracked_object_agent_tensor(5), _create_tracked_object_agent_tensor(7)]
        padded_reverse = pad_agent_states(backward_dummy_states, reverse=True)
        self.assertTrue(len(padded_reverse) == 3)
        self.assertEqual((7, AgentInternalIndex.dim()), padded_reverse[2].shape)
        for i in range(0, len(padded_reverse) - 1):
            self.assertTrue(torch.allclose(padded_reverse[2], padded_reverse[i]))

    def test_compute_yaw_rate_from_state_tensors(self) -> None:
        """
        Test compute yaw rate functionality
        """
        num_frames = 6
        num_agents = 5
        agent_states = [_create_tracked_object_agent_tensor(num_agents) + i for i in range(num_frames)]
        time_stamps = torch.tensor([int(i * 1000000.0) for i in range(num_frames)], dtype=torch.int64)
        yaw_rate = compute_yaw_rate_from_state_tensors(agent_states, time_stamps)
        self.assertEqual((num_frames, num_agents), yaw_rate.shape)
        self.assertTrue(torch.allclose(torch.ones((num_frames, num_agents), dtype=torch.float64), yaw_rate))

    def test_filter_agents_tensor(self) -> None:
        """
        Test filter agents
        """
        dummy_states = [_create_tracked_object_agent_tensor(7), _create_tracked_object_agent_tensor(8), _create_tracked_object_agent_tensor(6)]
        filtered = filter_agents_tensor(dummy_states, reverse=False)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered[0].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered[1].shape)
        self.assertEqual((6, AgentInternalIndex.dim()), filtered[2].shape)
        dummy_states = [_create_tracked_object_agent_tensor(6), _create_tracked_object_agent_tensor(8), _create_tracked_object_agent_tensor(7)]
        filtered_reverse = filter_agents_tensor(dummy_states, reverse=True)
        self.assertEqual((6, AgentInternalIndex.dim()), filtered_reverse[0].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered_reverse[1].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered_reverse[2].shape)

    def test_sampled_past_ego_states_to_tensor(self) -> None:
        """
        Test the conversion routine to convert ego states to tensors.
        """
        num_egos = 6
        test_egos = []
        for i in range(num_egos):
            footprint = CarFootprint(center=StateSE2(x=i, y=i, heading=i), vehicle_parameters=VehicleParameters(vehicle_name='vehicle_name', vehicle_type='vehicle_type', width=i, front_length=i, rear_length=i, cog_position_from_rear_axle=i, wheel_base=i, height=i))
            dynamic_car_state = DynamicCarState(rear_axle_to_center_dist=i, rear_axle_velocity_2d=StateVector2D(x=i + 5, y=i + 5), rear_axle_acceleration_2d=StateVector2D(x=i, y=i), angular_velocity=i, angular_acceleration=i, tire_steering_rate=i)
            test_ego = EgoState(car_footprint=footprint, dynamic_car_state=dynamic_car_state, tire_steering_angle=i, is_in_auto_mode=i, time_point=TimePoint(time_us=i))
            test_egos.append(test_ego)
        tensor = sampled_past_ego_states_to_tensor(test_egos)
        self.assertEqual((6, EgoInternalIndex.dim()), tensor.shape)
        for i in range(0, tensor.shape[0], 1):
            ego = test_egos[i]
            self.assertEqual(ego.rear_axle.x, tensor[i, EgoInternalIndex.x()].item())
            self.assertEqual(ego.rear_axle.y, tensor[i, EgoInternalIndex.y()].item())
            self.assertEqual(ego.rear_axle.heading, tensor[i, EgoInternalIndex.heading()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_velocity_2d.x, tensor[i, EgoInternalIndex.vx()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_velocity_2d.y, tensor[i, EgoInternalIndex.vy()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_acceleration_2d.x, tensor[i, EgoInternalIndex.ax()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_acceleration_2d.y, tensor[i, EgoInternalIndex.ay()].item())

    def test_sampled_past_timestamps_to_tensor(self) -> None:
        """
        Test the conversion routine to convert timestamps to tensors.
        """
        points = [TimePoint(time_us=i) for i in range(10)]
        tensor = sampled_past_timestamps_to_tensor(points)
        self.assertEqual((10,), tensor.shape)
        for i in range(tensor.shape[0]):
            self.assertEqual(i, int(tensor[i].item()))

    def test_tracked_objects_to_tensor_list(self) -> None:
        """
        Test the conversion routine to convert tracked objects to tensors.
        """
        num_frames = 5
        test_tracked_objects = _create_dummy_tracked_objects_tensor(num_frames)
        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)
            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))
        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects, object_type=TrackedObjectType.BICYCLE)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)
            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))
        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects, object_type=TrackedObjectType.PEDESTRIAN)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)
            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))

    def test_pack_agents_tensor(self) -> None:
        """
        Test the routine used to convert local buffers into the final feature.
        """
        num_agents = 4
        num_timestamps = 3
        agents_tensors = [_create_tracked_object_agent_tensor(num_agents) for _ in range(num_timestamps)]
        yaw_rates = torch.ones((num_timestamps, num_agents)) * 100
        packed = pack_agents_tensor(agents_tensors, yaw_rates)
        self.assertEqual((num_timestamps, num_agents, AgentFeatureIndex.dim()), packed.shape)
        for ts in range(num_timestamps):
            for agent in range(num_agents):
                for col in range(AgentFeatureIndex.dim()):
                    if col == AgentFeatureIndex.yaw_rate():
                        self.assertEqual(100, packed[ts, agent, col])
                    else:
                        self.assertEqual(agent, packed[ts, agent, col])

class RasterFeatureBuilder(AbstractFeatureBuilder):
    """
    Raster builder responsible for constructing model input features.
    """

    def __init__(self, map_features: Dict[str, int], num_input_channels: int, target_width: int, target_height: int, target_pixel_size: float, ego_width: float, ego_front_length: float, ego_rear_length: float, ego_longitudinal_offset: float, baseline_path_thickness: int) -> None:
        """
        Initializes the builder.
        :param map_features: name of map features to be drawn and their color for encoding.
        :param num_input_channels: number of input channel of the raster model.
        :param target_width: [pixels] target width of the raster
        :param target_height: [pixels] target height of the raster
        :param target_pixel_size: [m] target pixel size in meters
        :param ego_width: [m] width of the ego vehicle
        :param ego_front_length: [m] distance between the rear axle and the front bumper
        :param ego_rear_length: [m] distance between the rear axle and the rear bumper
        :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
                                        0.0 means place the ego at 1/2 from the bottom of the raster image.
                                        0.25 means place the ego at 1/4 from the bottom of the raster image.
        :param baseline_path_thickness: [pixels] the thickness of baseline paths in the baseline_paths_raster.
        """
        self.map_features = map_features
        self.num_input_channels = num_input_channels
        self.target_width = target_width
        self.target_height = target_height
        self.target_pixel_size = target_pixel_size
        self.ego_longitudinal_offset = ego_longitudinal_offset
        self.baseline_path_thickness = baseline_path_thickness
        self.raster_shape = (self.target_width, self.target_height)
        x_size = self.target_width * self.target_pixel_size / 2.0
        y_size = self.target_height * self.target_pixel_size / 2.0
        x_offset = 2.0 * self.ego_longitudinal_offset * x_size
        self.x_range = (-x_size + x_offset, x_size + x_offset)
        self.y_range = (-y_size, y_size)
        self.ego_width_pixels = int(ego_width / self.target_pixel_size)
        self.ego_front_length_pixels = int(ego_front_length / self.target_pixel_size)
        self.ego_rear_length_pixels = int(ego_rear_length / self.target_pixel_size)

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'raster'

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Raster

    def get_features_from_scenario(self, scenario: AbstractScenario) -> Raster:
        """Inherited, see superclass."""
        ego_state = scenario.initial_ego_state
        detections = scenario.initial_tracked_objects
        map_api = scenario.map_api
        return self._compute_feature(ego_state, detections, map_api)

    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> Raster:
        """Inherited, see superclass."""
        history = current_input.history
        ego_state = history.ego_states[-1]
        observation = history.observations[-1]
        if isinstance(observation, DetectionsTracks):
            return self._compute_feature(ego_state, observation, initialization.map_api)
        else:
            raise TypeError(f'Observation was type {observation.detection_type()}. Expected DetectionsTracks')

    def _compute_feature(self, ego_state: EgoState, detections: DetectionsTracks, map_api: AbstractMap) -> Raster:
        roadmap_raster = get_roadmap_raster(ego_state.agent, map_api, self.map_features, self.x_range, self.y_range, self.raster_shape, self.target_pixel_size)
        agents_raster = get_agents_raster(ego_state, detections, self.x_range, self.y_range, self.raster_shape)
        ego_raster = get_ego_raster(self.raster_shape, self.ego_longitudinal_offset, self.ego_width_pixels, self.ego_front_length_pixels, self.ego_rear_length_pixels)
        baseline_paths_raster = get_baseline_paths_raster(ego_state.agent, map_api, self.x_range, self.y_range, self.raster_shape, self.target_pixel_size, self.baseline_path_thickness)
        collated_layers: npt.NDArray[np.float32] = np.dstack([ego_raster, agents_raster, roadmap_raster, baseline_paths_raster]).astype(np.float32)
        if collated_layers.shape[-1] != self.num_input_channels:
            raise RuntimeError(f'Invalid raster numpy array. Expected {self.num_input_channels} channels, got {collated_layers.shape[-1]} Shape is {collated_layers.shape}')
        return Raster(data=collated_layers)

def get_roadmap_raster(focus_agent: AgentState, map_api: AbstractMap, map_features: Dict[str, int], x_range: Tuple[float, float], y_range: Tuple[float, float], raster_shape: Tuple[int, int], resolution: float) -> npt.NDArray[np.float32]:
    """
    Construct the map layer of the raster by converting vector map to raster map.
    :param focus_agent: agent state representing ego.
    :param map_api: map api.
    :param map_features: name of map features to be drawn and its color for encoding.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :return roadmap_raster: the constructed map raster layer.
    """
    assert x_range[1] - x_range[0] == y_range[1] - y_range[0], f'Raster shape is assumed to be square but got width:             {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'
    radius = (x_range[1] - x_range[0]) / 2
    roadmap_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)
    for feature_name, feature_color in map_features.items():
        coords, _ = _get_layer_coords(focus_agent, map_api, SemanticMapLayer[feature_name], 'polygon', radius)
        roadmap_raster = _draw_polygon_image(roadmap_raster, coords, radius, resolution, feature_color)
    roadmap_raster = np.flip(roadmap_raster, axis=0)
    roadmap_raster = np.ascontiguousarray(roadmap_raster, dtype=np.float32)
    return roadmap_raster

def get_agents_raster(ego_state: EgoState, detections: DetectionsTracks, x_range: Tuple[float, float], y_range: Tuple[float, float], raster_shape: Tuple[int, int], polygon_bit_shift: int=9) -> npt.NDArray[np.float32]:
    """
    Construct the agents layer of the raster by transforming all detected boxes around the agent
    and creating polygons of them in a raster grid.
    :param ego_state: SE2 state of ego.
    :param detections: list of 3D bounding box of detected agents.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param polygon_bit_shift: bit shift of the polygon used in opencv.
    :return: constructed agents raster layer.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape
    agents_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)
    ego_to_global = ego_state.rear_axle.as_matrix()
    global_to_ego = np.linalg.inv(ego_to_global)
    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()
    tracked_objects = [deepcopy(tracked_object) for tracked_object in detections.tracked_objects]
    for tracked_object in tracked_objects:
        raster_object_matrix = north_aligned_transform @ global_to_ego @ tracked_object.center.as_matrix()
        raster_object_pose = StateSE2.from_matrix(raster_object_matrix)
        valid_x = x_range[0] < raster_object_pose.x < x_range[1]
        valid_y = y_range[0] < raster_object_pose.y < y_range[1]
        if not (valid_x and valid_y):
            continue
        raster_oriented_box = OrientedBox(raster_object_pose, tracked_object.box.length, tracked_object.box.width, tracked_object.box.height)
        box_bottom_corners = raster_oriented_box.all_corners()
        x_corners = np.asarray([corner.x for corner in box_bottom_corners])
        y_corners = np.asarray([corner.y for corner in box_bottom_corners])
        y_corners = (y_corners - ymin) / (ymax - ymin) * height
        x_corners = (x_corners - xmin) / (xmax - xmin) * width
        box_2d_coords = np.stack([x_corners, y_corners], axis=1)
        box_2d_coords = np.expand_dims(box_2d_coords, axis=0)
        box_2d_coords = (box_2d_coords * 2 ** polygon_bit_shift).astype(np.int32)
        cv2.fillPoly(agents_raster, box_2d_coords, color=1.0, shift=polygon_bit_shift, lineType=cv2.LINE_AA)
    agents_raster = np.asarray(agents_raster)
    agents_raster = np.flip(agents_raster, axis=0)
    agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)
    return agents_raster

def get_ego_raster(raster_shape: Tuple[int, int], ego_longitudinal_offset: float, ego_width_pixels: float, ego_front_length_pixels: float, ego_rear_length_pixels: float) -> npt.NDArray[np.float32]:
    """
    Construct the ego layer of the raster by drawing a polygon of the ego's extent in the middle of the grid.
    :param raster_shape: shape of the target raster.
    :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
    :param ego_width_pixels: width of the ego vehicle in pixels.
    :param ego_front_length_pixels: distance between the rear axle and the front bumper in pixels.
    :param ego_rear_length_pixels: distance between the rear axle and the rear bumper in pixels.
    :return: constructed ego raster layer.
    """
    ego_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)
    map_x_center = int(raster_shape[1] * 0.5)
    map_y_center = int(raster_shape[0] * (0.5 + ego_longitudinal_offset))
    ego_top_left = (map_x_center - ego_width_pixels // 2, map_y_center - ego_front_length_pixels)
    ego_bottom_right = (map_x_center + ego_width_pixels // 2, map_y_center + ego_rear_length_pixels)
    cv2.rectangle(ego_raster, ego_top_left, ego_bottom_right, 1, -1)
    return np.asarray(ego_raster)

def get_baseline_paths_raster(focus_agent: AgentState, map_api: AbstractMap, x_range: Tuple[float, float], y_range: Tuple[float, float], raster_shape: Tuple[int, int], resolution: float, baseline_path_thickness: int=1) -> npt.NDArray[np.float32]:
    """
    Construct the baseline paths layer by converting vector map to raster map.
    This funciton is for ego raster model, the baselin path only has one channel.
    :param ego_state: SE2 state of ego.
    :param map_api: map api
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :return baseline_paths_raster: the constructed baseline paths layer.
    """
    if x_range[1] - x_range[0] != y_range[1] - y_range[0]:
        raise ValueError(f'Raster shape is assumed to be square but got width:             {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}')
    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)
    for map_features in ['LANE', 'LANE_CONNECTOR']:
        baseline_paths_coords, lane_ids = _get_layer_coords(agent=focus_agent, map_api=map_api, map_layer_name=SemanticMapLayer[map_features], map_layer_geometry='linestring', radius=radius)
        lane_colors: npt.NDArray[np.uint8] = np.ones(len(lane_ids)).astype(np.uint8)
        baseline_paths_raster = _draw_linestring_image(image=baseline_paths_raster, object_coords=baseline_paths_coords, radius=radius, resolution=resolution, baseline_path_thickness=baseline_path_thickness, lane_colors=lane_colors)
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster

def _convert_absolute_to_relative_states(origin_absolute_state: StateSE2, absolute_states: List[StateSE2]) -> List[StateSE2]:
    """
    Computes the relative states from a list of absolute states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param absolute_states: list of absolute poses.
    :return: list of relative states.
    """
    origin_absolute_transform = origin_absolute_state.as_matrix()
    origin_transform = np.linalg.inv(origin_absolute_transform)
    absolute_transforms: npt.NDArray[np.float32] = np.array([state.as_matrix() for state in absolute_states])
    relative_transforms = origin_transform @ absolute_transforms.reshape(-1, 3, 3)
    relative_states = [StateSE2.from_matrix(transform) for transform in relative_transforms]
    return relative_states

def _convert_relative_to_absolute_states(origin_absolute_state: StateSE2, relative_states: List[StateSE2]) -> List[StateSE2]:
    """
    Computes the absolute states from a list of relative states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param relative_states: list of relative poses.
    :return: list of absolute states.
    """
    origin_transform = origin_absolute_state.as_matrix()
    relative_transforms: npt.NDArray[np.float32] = np.array([state.as_matrix() for state in relative_states])
    absolute_transforms = origin_transform @ relative_transforms.reshape(-1, 3, 3)
    absolute_states = [StateSE2.from_matrix(transform) for transform in absolute_transforms]
    return absolute_states

def convert_relative_to_absolute_poses(origin_absolute_state: StateSE2, relative_states: List[StateSE2]) -> npt.NDArray[np.float64]:
    """
    Computes the absolute poses from a list of relative states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param relative_states: list of absolute poses.
    :return: list of relative poses as numpy array.
    """
    absolute_states = _convert_relative_to_absolute_states(origin_absolute_state, relative_states)
    absolute_poses: npt.NDArray[np.float64] = np.asarray([state.serialize() for state in absolute_states]).astype(np.float64)
    return absolute_poses

def convert_absolute_to_relative_velocities(origin_absolute_velocity: StateSE2, absolute_velocities: List[StateSE2]) -> npt.NDArray[np.float32]:
    """
    Computes the relative velocities from a list of absolute velocities using an origin (anchor) velocity.

    :param origin_absolute_velocity: absolute velocities to be used as origin.
    :param absolute_velocities: list of absolute velocities.
    :return: list of relative velocities as numpy array.
    """
    relative_states = _convert_absolute_to_relative_states(origin_absolute_velocity, absolute_velocities)
    relative_velocities: npt.NDArray[np.float32] = np.asarray([[state.x, state.y] for state in relative_states]).astype(np.float32)
    return relative_velocities

def _get_layer_coords(agent: AgentState, map_api: AbstractMap, map_layer_name: SemanticMapLayer, map_layer_geometry: str, radius: float) -> Tuple[List[npt.NDArray[np.float64]], List[str]]:
    """
    Construct the map layer of the raster by converting vector map to raster map, based on the focus agent.
    :param agent: the focus agent used for raster generating.
    :param map_api: map api
    :param map_layer_name: name of the vector map layer to create a raster from.
    :param map_layer_geometry: geometric primitive of the vector map layer. i.e. either polygon or linestring.
    :param radius: [m] the radius of the square raster map.
    :return
        object_coords: the list of 2d coordinates which represent the shape of the map.
        lane_ids: the list of ids for the map objects.
    """
    ego_position = Point2D(agent.center.x, agent.center.y)
    nearest_vector_map = map_api.get_proximal_map_objects(layers=[map_layer_name], point=ego_position, radius=radius)
    geometry = nearest_vector_map[map_layer_name]
    if len(geometry):
        global_transform = np.linalg.inv(agent.center.as_matrix())
        map_align_transform = R.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
        transform = map_align_transform @ global_transform
        if map_layer_geometry == 'polygon':
            _object_coords = _polygon_to_coords(geometry)
        elif map_layer_geometry == 'linestring':
            _object_coords = _linestring_to_coords(geometry)
        else:
            raise RuntimeError(f'Layer geometry {map_layer_geometry} type not supported')
        object_coords: List[npt.NDArray[np.float64]] = [np.vstack(coords).T for coords in _object_coords]
        object_coords = [(transform @ _cartesian_to_projective_coords(coords).T).T[:, :2] for coords in object_coords]
        lane_ids = [lane.id for lane in geometry]
    else:
        object_coords = []
        lane_ids = []
    return (object_coords, lane_ids)

def _polygon_to_coords(geometry: List[PolygonMapObject]) -> List[Tuple[array[float]]]:
    """
    Get 2d coordinates of the vertices of a polygon.
    The polygon is a shapely.geometry.polygon.
    :param geometry: the polygon.
    :return: 2d coordinates of the vertices of the polygon.
    """
    return [element.polygon.exterior.coords.xy for element in geometry]

def _linestring_to_coords(geometry: List[PolylineMapObject]) -> List[Tuple[array[float]]]:
    """
    Get 2d coordinates of the endpoints of line segment string.
    The line segment string is a shapely.geometry.linestring.
    :param geometry: the line segment string.
    :return: 2d coordinates of the endpoints of line segment string.
    """
    return [element.baseline_path.linestring.coords.xy for element in geometry]

def _cartesian_to_projective_coords(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert from cartesian coordinates to projective coordinates.
    :param coords: the 2d coordinates of shape (N, 2) where N is the number of points.
    :return: the resulting projective coordinates of shape (N, 3).
    """
    return np.pad(coords, ((0, 0), (0, 1)), 'constant', constant_values=1.0)

def _draw_polygon_image(image: npt.NDArray[np.float32], object_coords: List[npt.NDArray[np.float64]], radius: float, resolution: float, color: float, bit_shift: int=12) -> npt.NDArray[np.float32]:
    """
    Draw a map feature consisting of polygons using a list of its coordinates.
    :param image: the raster map on which the map feature will be drawn
    :param object_coords: the coordinates that represents the shape of the map feature.
    :param radius: the radius of the square raster map.
    :param resolution: [m] pixel size in meters.
    :param color: color of the map feature.
    :param bit_shift: bit shift of the polygon used in opencv.
    :return: the resulting raster map with the map feature.
    """
    if len(object_coords):
        for coords in object_coords:
            index_coords = (radius + coords) / resolution
            shifted_index_coords = (index_coords * 2 ** bit_shift).astype(np.int64)
            cv2.fillPoly(image, shifted_index_coords[None], color=color, shift=bit_shift, lineType=cv2.LINE_AA)
    return image

def get_non_focus_agents_raster(focus_agent: AgentState, other_agents: List[Agent], x_range: Tuple[float, float], y_range: Tuple[float, float], raster_shape: Tuple[int, int], polygon_bit_shift: int=9) -> npt.NDArray[np.float32]:
    """
    Construct the agents layer of the raster by transforming all other agents around the focus agent
    and creating polygons of them in a raster grid.
    :param focus_agent: focus agent used for rasterization.
    :param other agents: list of agents including the ego AV but excluding the focus agent.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param polygon_bit_shift: bit shift of the polygon used in opencv.
    :return: constructed agents raster layer.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape
    agents_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)
    ego_to_global = focus_agent.center.as_matrix()
    global_to_ego = np.linalg.inv(ego_to_global)
    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()
    for tracked_object in other_agents:
        raster_object_matrix = north_aligned_transform @ global_to_ego @ tracked_object.center.as_matrix()
        raster_object_pose = StateSE2.from_matrix(raster_object_matrix)
        valid_x = x_range[0] < raster_object_pose.x < x_range[1]
        valid_y = y_range[0] < raster_object_pose.y < y_range[1]
        if not (valid_x and valid_y):
            continue
        raster_oriented_box = OrientedBox(raster_object_pose, tracked_object.box.length, tracked_object.box.width, tracked_object.box.height)
        box_bottom_corners = raster_oriented_box.all_corners()
        x_corners = np.asarray([corner.x for corner in box_bottom_corners])
        y_corners = np.asarray([corner.y for corner in box_bottom_corners])
        y_corners = (y_corners - ymin) / (ymax - ymin) * height
        x_corners = (x_corners - xmin) / (xmax - xmin) * width
        box_2d_coords = np.stack([x_corners, y_corners], axis=1)
        box_2d_coords = np.expand_dims(box_2d_coords, axis=0)
        box_2d_coords = (box_2d_coords * 2 ** polygon_bit_shift).astype(np.int32)
        cv2.fillPoly(agents_raster, box_2d_coords, color=1.0, shift=polygon_bit_shift, lineType=cv2.LINE_AA)
    agents_raster = np.asarray(agents_raster)
    agents_raster = np.flip(agents_raster, axis=0)
    agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)
    return agents_raster

def _draw_linestring_image(image: npt.NDArray[np.float32], object_coords: List[npt.NDArray[np.float64]], radius: float, resolution: float, baseline_path_thickness: int, lane_colors: npt.NDArray[np.uint8], bit_shift: int=13) -> npt.NDArray[np.float32]:
    """
    Draw a map feature consisting of linestring using a list of its coordinates.
    :param image: the raster map on which the map feature will be drawn
    :param object_coords: the coordinates that represents the shape of the map feature.
    :param radius: the radius of the square raster map.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :param lane_colors: an array indicate colors for each element of object_coords.
    :param bit_shift: bit shift of the polylines used in opencv.
    :return: the resulting raster map with the map feature.
    """
    if len(object_coords):
        assert len(object_coords) == len(lane_colors)
        for coords, lane_color in zip(object_coords, lane_colors):
            index_coords = (radius + coords) / resolution
            shifted_index_coords = (index_coords * 2 ** bit_shift).astype(np.int64)
            lane_color = int(lane_color) if np.isscalar(lane_color) else [int(item) for item in lane_color]
            cv2.polylines(image, [shifted_index_coords], isClosed=False, color=lane_color, thickness=baseline_path_thickness, shift=bit_shift, lineType=cv2.LINE_AA)
    return image

def get_baseline_paths_agents_raster(focus_agent: AgentState, map_api: AbstractMap, x_range: Tuple[float, float], y_range: Tuple[float, float], raster_shape: Tuple[int, int], resolution: float, traffic_light_connectors: Dict[TrafficLightStatusType, List[str]], baseline_path_thickness: int=1) -> npt.NDArray[np.float32]:
    """
    Construct the baseline paths layer by converting vector map to raster map.
    This function is for agents raster model, it has 3 channels for baseline path.
    :param focus_agent: agent state representing ego.
    :param map_api: map api
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param traffic_light_connectors: a dict mapping tl status type to a list of lane ids in this status.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :return baseline_paths_raster: the constructed baseline paths layer.
    """
    if x_range[1] - x_range[0] != y_range[1] - y_range[0]:
        raise ValueError(f'Raster shape is assumed to be square but got width:             {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}')
    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster: npt.NDArray[np.float32] = np.zeros((*raster_shape, 3), dtype=np.float32)
    for map_features in ['LANE', 'LANE_CONNECTOR']:
        baseline_paths_coords, lane_ids = _get_layer_coords(agent=focus_agent, map_api=map_api, map_layer_name=SemanticMapLayer[map_features], map_layer_geometry='linestring', radius=radius)
        lane_ids = np.asarray(lane_ids)
        lane_colors: npt.NDArray[np.uint8] = np.full((len(lane_ids), 3), BASELINE_TL_COLOR[TrafficLightStatusType.UNKNOWN], dtype=np.uint8)
        if len(traffic_light_connectors) > 0:
            for tl_status in TrafficLightStatusType:
                if tl_status != TrafficLightStatusType.UNKNOWN and len(traffic_light_connectors[tl_status]) > 0:
                    lanes_in_tl_status = np.isin(lane_ids, traffic_light_connectors[tl_status])
                    lane_colors[lanes_in_tl_status] = BASELINE_TL_COLOR[tl_status]
        baseline_paths_raster = _draw_linestring_image(image=baseline_paths_raster, object_coords=baseline_paths_coords, radius=radius, resolution=resolution, baseline_path_thickness=baseline_path_thickness, lane_colors=lane_colors)
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster

class TestRasterUtils(unittest.TestCase):
    """Test raster building utility functions."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        scenario = get_test_nuplan_scenario()
        self.x_range = [-56.0, 56.0]
        self.y_range = [-56.0, 56.0]
        self.raster_shape = (224, 224)
        self.resolution = 0.5
        self.thickness = 2
        self.ego_state = scenario.initial_ego_state
        self.map_api = scenario.map_api
        self.tracked_objects = scenario.initial_tracked_objects
        self.map_features = {'LANE': 255, 'INTERSECTION': 255, 'STOP_LINE': 128, 'CROSSWALK': 128}
        ego_width = 2.297
        ego_front_length = 4.049
        ego_rear_length = 1.127
        self.ego_longitudinal_offset = 0.0
        self.ego_width_pixels = int(ego_width / self.resolution)
        self.ego_front_length_pixels = int(ego_front_length / self.resolution)
        self.ego_rear_length_pixels = int(ego_rear_length / self.resolution)

    def test_get_roadmap_raster(self) -> None:
        """
        Test get_roadmap_raster / get_agents_raster / get_baseline_paths_raster
        """
        self.assertGreater(len(self.tracked_objects.tracked_objects), 0)
        roadmap_raster = get_roadmap_raster(self.ego_state, self.map_api, self.map_features, self.x_range, self.y_range, self.raster_shape, self.resolution)
        agents_raster = get_agents_raster(self.ego_state, self.tracked_objects, self.x_range, self.y_range, self.raster_shape)
        ego_raster = get_ego_raster(self.raster_shape, self.ego_longitudinal_offset, self.ego_width_pixels, self.ego_front_length_pixels, self.ego_rear_length_pixels)
        baseline_paths_raster = get_baseline_paths_raster(self.ego_state, self.map_api, self.x_range, self.y_range, self.raster_shape, self.resolution, self.thickness)
        self.assertEqual(roadmap_raster.shape, self.raster_shape)
        self.assertEqual(agents_raster.shape, self.raster_shape)
        self.assertEqual(ego_raster.shape, self.raster_shape)
        self.assertEqual(baseline_paths_raster.shape, self.raster_shape)
        self.assertTrue(np.any(roadmap_raster))
        self.assertTrue(np.any(agents_raster))
        self.assertTrue(np.any(ego_raster))
        self.assertTrue(np.any(baseline_paths_raster))

class EgoTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Trajectory builders constructed the desired ego's trajectory from a scenario."""

    def __init__(self, future_trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes the class.
        :param future_trajectory_sampling: parameters for sampled future trajectory
        """
        self._num_future_poses = future_trajectory_sampling.num_poses
        self._time_horizon = future_trajectory_sampling.time_horizon

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'trajectory'

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Trajectory

    def get_targets(self, scenario: AbstractScenario) -> Trajectory:
        """Inherited, see superclass."""
        current_absolute_state = scenario.initial_ego_state
        trajectory_absolute_states = scenario.get_ego_future_trajectory(iteration=0, num_samples=self._num_future_poses, time_horizon=self._time_horizon)
        trajectory_relative_poses = convert_absolute_to_relative_poses(current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states])
        if len(trajectory_relative_poses) != self._num_future_poses:
            raise RuntimeError(f'Expected {self._num_future_poses} num poses but got {len(trajectory_relative_poses)}')
        return Trajectory(data=trajectory_relative_poses)

