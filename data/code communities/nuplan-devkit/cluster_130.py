# Cluster 130

class TestGenericAgentDropoutAugmentation(unittest.TestCase):
    """Test agent augmentation that drops out random agents from the scene."""

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)
        self.features = {}
        self.agent_features = ['VEHICLE', 'BICYCLE', 'PEDESTRIAN']
        self.features['generic_agents'] = GenericAgents(ego=[np.random.randn(5, 3), np.random.randn(5, 3)], agents={feature_name: [np.random.randn(5, 20, 8), np.random.randn(5, 50, 8)] for feature_name in self.agent_features})
        self.targets: Dict[str, Any] = {}
        augment_prob = 1.0
        self.dropout_rate = 0.5
        self.augmentor = GenericAgentDropoutAugmentor(augment_prob, self.dropout_rate)

    def test_augment(self) -> None:
        """
        Test augmentation.
        """
        features = deepcopy(self.features)
        aug_features, _ = self.augmentor.augment(features, self.targets)
        for feature_name in self.agent_features:
            for agents, aug_agents in zip(self.features['generic_agents'].agents[feature_name], aug_features['generic_agents'].agents[feature_name]):
                self.assertLess(aug_agents.shape[1], agents.shape[1])

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.augmentor._augment_prob = 0.0
        aug_features, _ = self.augmentor.augment(self.features, self.targets)
        for feature_name in self.agent_features:
            self.assertTrue((aug_features['generic_agents'].agents[feature_name][0] == self.features['generic_agents'].agents[feature_name][0]).all())

class TestKinematicHistoryGenericAgentAugmentation(unittest.TestCase):
    """
    Test agent augmentation that perturbs the current ego position and generates a feasible trajectory history that
    satisfies a set of kinematic constraints.
    """

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)
        self.radius = 50
        self.features = {}
        self.agent_features = ['VEHICLE', 'BICYCLE', 'PEDESTRIAN']
        self.features['generic_agents'] = GenericAgents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05, 0.0, 0.0, 0.0, 0.0], [0.004325964, -0.00069646863, -9.3163371e-06, 0.0, 0.0, 0.0, 0.0], [0.0024353617, -0.00037753209, 4.7789731e-06, 0.0, 0.0, 0.0, 0.0], [0.0011352128, -0.0001273104, 3.8040514e-05, 0.0, 0.0, 0.0, 0.0], [1.1641532e-10, 0.0, -3.0870851e-19, 0.0, 0.0, 0.0, 0.0]]), np.array([[0.0069434252, -0.001094915, 2.1299818e-05, 0.0, 0.0, 0.0, 0.0], [0.004325964, -0.00069646863, -9.3163371e-06, 0.0, 0.0, 0.0, 0.0], [0.0024353617, -0.00037753209, 4.7789731e-06, 0.0, 0.0, 0.0, 0.0], [0.0011352128, -0.0001273104, 3.8040514e-05, 0.0, 0.0, 0.0, 0.0], [1.1641532e-10, 0.0, -3.0870851e-19, 0.0, 0.0, 0.0, 0.0]])], agents={feature_name: [self.radius * np.random.rand(5, 1, 8) + self.radius / 2, self.radius * np.random.rand(5, 1, 8) + self.radius / 2] for feature_name in self.agent_features})
        for sample_idx in range(len(self.features['generic_agents'].ego)):
            self.features['generic_agents'].ego[sample_idx][:-1, 3:5] = np.diff(self.features['generic_agents'].ego[sample_idx][:, :2], axis=0)
            self.features['generic_agents'].ego[sample_idx][:-1, 5:] = np.diff(self.features['generic_agents'].ego[sample_idx][:, 3:5], axis=0)
        self.aug_feature_gt = {}
        self.aug_feature_gt['generic_agents'] = GenericAgents(ego=[np.array([[0.0069434252, -0.001094915, 2.1299818e-05, 0.0, 0.0, 0.0, 0.0], [0.0120681393, -0.00109217957, 0.00104624288, 0.0, 0.0, 0.0, 0.0], [0.0268775601, -0.00105475327, 0.00400813782, 0.0, 0.0, 0.0, 0.0], [0.0512891984, -0.000897311768, 0.00889057227, 0.0, 0.0, 0.0, 0.0], [0.0852192154, -0.000480500022, 0.0156771013, 0.0, 0.0, 0.0, 0.0]])], agents={feature_name: [self.radius * np.random.rand(5, 1, 8) + self.radius / 2] for feature_name in self.agent_features})
        for sample_idx in range(len(self.aug_feature_gt['generic_agents'].ego)):
            self.aug_feature_gt['generic_agents'].ego[sample_idx][:-1, 3:5] = np.diff(self.aug_feature_gt['generic_agents'].ego[sample_idx][:, :2], axis=0)
            self.aug_feature_gt['generic_agents'].ego[sample_idx][:-1, 5:] = np.diff(self.aug_feature_gt['generic_agents'].ego[sample_idx][:, 3:5], axis=0)
        self.targets: Dict[str, Any] = {}
        augment_prob = 1.0
        dt = 0.1
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        self.gaussian_augmentor = KinematicHistoryGenericAgentAugmentor(dt, mean, std, low, high, augment_prob, use_uniform_noise=False)
        self.uniform_augmentor = KinematicHistoryGenericAgentAugmentor(dt, mean, std, low, high, augment_prob, use_uniform_noise=True)

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['generic_agents'].ego[0][:, :3] - self.aug_feature_gt['generic_agents'].ego[0][:, :3]) < 0.1).all())

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        original_feature_ego = self.features['generic_agents'].ego[1].copy()[:, :3]
        aug_feature, _ = self.uniform_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['generic_agents'].ego[1][:, :3] - original_feature_ego) <= 0.1).all())

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.gaussian_augmentor._augment_prob = 0.0
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['generic_agents'].ego[0] == self.features['generic_agents'].ego[0]).all())

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

def get_raster_with_trajectories_as_rgb(raster: Raster, target_trajectory: Optional[Trajectory]=None, predicted_trajectory: Optional[Trajectory]=None, pixel_size: float=0.5) -> npt.NDArray[np.uint8]:
    """
    Create an RGB images of the raster layers overlayed with predicted / ground truth trajectories

    :param raster: input raster to visualize
    :param target_trajectory: target (ground truth) trajectory to visualize
    :param predicted_trajectory: predicted trajectory to visualize
    :param background_color: desired color of the image's background
    :param roadmap_color: desired color of the map raster layer
    :param agents_color: desired color of the agents raster layer
    :param ego_color: desired color of the ego raster layer
    :param target_trajectory_color: desired color of the target trajectory
    :param predicted_trajectory_color: desired color of the predicted trajectory
    :param pixel_size: [m] size of pixel in meters
    :return: constructed RGB image
    """
    grid_shape = (raster.height, raster.width)
    image: npt.NDArray[np.uint8] = np.full((*grid_shape, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[raster.roadmap_layer[0] > 0] = Color.ROADMAP.value
    image[raster.baseline_paths_layer[0] > 0] = Color.BASELINE_PATHS.value
    image[raster.agents_layer.squeeze() > 0] = Color.AGENTS.value
    image[raster.ego_layer.squeeze() > 0] = Color.EGO.value
    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, Color.TARGET_TRAJECTORY, pixel_size, 2, 1)
    if predicted_trajectory is not None:
        _draw_trajectory(image, predicted_trajectory, Color.PREDICTED_TRAJECTORY, pixel_size, 2, 1)
    return image

def get_raster_from_vector_map_with_agents(vector_map: Union[VectorMap, VectorSetMap], agents: Union[Agents, GenericAgents], target_trajectory: Optional[Trajectory]=None, predicted_trajectory: Optional[Trajectory]=None, pixel_size: float=0.5, bit_shift: int=12, radius: float=50.0, vehicle_parameters: VehicleParameters=get_pacifica_parameters()) -> npt.NDArray[np.uint8]:
    """
    Create rasterized image from vector map and list of agents.

    :param vector_map: Vector map/vector set map feature to visualize.
    :param agents: Agents/GenericAgents feature to visualize.
    :param target_trajectory: Target trajectory to visualize.
    :param predicted_trajectory: Predicted trajectory to visualize.
    :param pixel_size: [m] Size of a pixel.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param radius: [m] Radius of raster.
    :param vehicle_parameters: Parameters of the ego vehicle.
    :return: Composed rasterized image.
    """
    size = int(2 * radius / pixel_size)
    map_raster = _create_map_raster(vector_map, radius, size, bit_shift, pixel_size)
    agents_raster = _create_agents_raster(agents, radius, size, bit_shift, pixel_size)
    ego_raster = _create_ego_raster(vehicle_parameters, pixel_size, size)
    image: npt.NDArray[np.uint8] = np.full((size, size, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[map_raster.nonzero()] = Color.BASELINE_PATHS.value
    image[agents_raster.nonzero()] = Color.AGENTS.value
    image[ego_raster.nonzero()] = Color.EGO.value
    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, Color.TARGET_TRAJECTORY, pixel_size)
    if predicted_trajectory is not None:
        _draw_trajectory(image, predicted_trajectory, Color.PREDICTED_TRAJECTORY, pixel_size)
    return image

def _create_map_raster(vector_map: Union[VectorMap, VectorSetMap], radius: float, size: int, bit_shift: int, pixel_size: float, color: int=1, thickness: int=2) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    vector_coords = vector_map.get_lane_coords(0)
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2 ** bit_shift).astype(np.int64)
    cv2.polylines(map_raster, shifted_index_coords, isClosed=False, color=color, thickness=thickness, shift=bit_shift, lineType=cv2.LINE_AA)
    map_raster = np.flipud(map_raster)
    return map_raster

def _create_agents_raster(agents: Union[Agents, GenericAgents], radius: float, size: int, bit_shift: int, pixel_size: float, color: int=1) -> npt.NDArray[np.uint8]:
    """
    Create agents raster layer to be visualized.

    :param agents: agents feature object (either Agents or GenericAgents).
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :return: Instantiated grid.
    """
    agents_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)
    agents_array: npt.NDArray[np.float32] = np.asarray(agents.get_present_agents_in_sample(0))
    agents_corners: npt.NDArray[np.float32] = np.asarray(agents.get_agent_corners_in_sample(0))
    if len(agents_array) == 0:
        return agents_raster
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    transform = Rotation.from_euler('z', agents_array[:, 2], degrees=False).as_matrix().astype(np.float32)
    transform[:, :2, 2] = agents_array[:, :2]
    points = (map_ortho_align @ transform @ agents_corners.transpose([0, 2, 1])).transpose([0, 2, 1])[..., :2]
    points[..., 0] = np.clip(points[..., 0], -radius, radius)
    points[..., 1] = np.clip(points[..., 1], -radius, radius)
    index_points = (radius + points) / pixel_size
    shifted_index_points = (index_points * 2 ** bit_shift).astype(np.int64)
    for box in shifted_index_points:
        cv2.fillPoly(agents_raster, box[None], color=color, shift=bit_shift, lineType=cv2.LINE_AA)
    agents_raster = np.flipud(agents_raster)
    return agents_raster

def _create_ego_raster(vehicle_parameters: VehicleParameters, pixel_size: float, size: int, color: int=1, thickness: int=-1) -> npt.NDArray[np.uint8]:
    """
    Create ego raster layer to be visualized.

    :param vehicle_parameters: Ego vehicle parameters dataclass object.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Box line thickness (-1 means fill).
    :return: Instantiated grid.
    """
    ego_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)
    ego_width = vehicle_parameters.width
    ego_front_length = vehicle_parameters.front_length
    ego_rear_length = vehicle_parameters.rear_length
    ego_width_pixels = int(ego_width / pixel_size)
    ego_front_length_pixels = int(ego_front_length / pixel_size)
    ego_rear_length_pixels = int(ego_rear_length / pixel_size)
    map_x_center = int(ego_raster.shape[1] * 0.5)
    map_y_center = int(ego_raster.shape[0] * 0.5)
    ego_top_left = (map_x_center - ego_width_pixels // 2, map_y_center - ego_front_length_pixels)
    ego_bottom_right = (map_x_center + ego_width_pixels // 2, map_y_center + ego_rear_length_pixels)
    cv2.rectangle(ego_raster, ego_top_left, ego_bottom_right, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return ego_raster

def _draw_trajectory(image: npt.NDArray[np.uint8], trajectory: Trajectory, color: Color, pixel_size: float, radius: int=7, thickness: int=3) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    :param radius: radius of each trajectory pose to be visualized
    :param thickness: thickness of lines connecting trajectory poses to be visualized
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2
    coords_x = (center_x - trajectory.numpy_position_x / pixel_size).astype(np.int32)
    coords_y = (center_y - trajectory.numpy_position_y / pixel_size).astype(np.int32)
    idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
    coords_x = coords_x[idxs]
    coords_y = coords_y[idxs]
    for point in zip(coords_y, coords_x):
        cv2.circle(image, point, radius=radius, color=color.value, thickness=-1)
    for point_1, point_2 in zip(zip(coords_y[:-1], coords_x[:-1]), zip(coords_y[1:], coords_x[1:])):
        cv2.line(image, point_1, point_2, color=color.value, thickness=thickness)

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

class GenericAgentsFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, agent_features: List[str], trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__()
        self.agent_features = agent_features
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon
        self._agents_states_dim = GenericAgents.agents_states_dim()
        if 'EGO' in self.agent_features:
            raise AssertionError('EGO not valid agents feature type!')
        for feature_name in self.agent_features:
            if feature_name not in TrackedObjectType._member_names_:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'generic_agents'

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return GenericAgents

    @torch.jit.unused
    def get_scriptable_input_from_scenario(self, scenario: AbstractScenario) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        anchor_ego_state = scenario.initial_ego_state
        past_ego_states = scenario.get_ego_past_trajectory(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)
        sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
        time_stamps = list(scenario.get_past_timestamps(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)) + [scenario.start_time]
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [tracked_objects.tracked_objects for tracked_objects in scenario.get_past_tracked_objects(iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses)]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        assert len(sampled_past_ego_states) == len(sampled_past_observations), f'Expected the trajectory length of ego and agent to be equal. Got ego: {len(sampled_past_ego_states)} and agent: {len(sampled_past_observations)}'
        assert len(sampled_past_observations) > 2, f'Trajectory of length of {len(sampled_past_observations)} needs to be at least 3'
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_scriptable_input_from_simulation(self, current_input: PlannerInput) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the simulation input
        :param current_input: planner input from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
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
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> GenericAgents:
        """Inherited, see superclass."""
        with torch.no_grad():
            tensors, list_tensors, list_list_tensors = self.get_scriptable_input_from_scenario(scenario)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: GenericAgents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> GenericAgents:
        """Inherited, see superclass."""
        with torch.no_grad():
            tensors, list_tensors, list_list_tensors = self.get_scriptable_input_from_simulation(current_input)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: GenericAgents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
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
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(past_ego_states)
        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)
        for feature_name in self.agent_features:
            past_tracked_objects_tensor_list = sampled_tracked_objects_to_tensor_list(past_tracked_objects, TrackedObjectType[feature_name])
            list_tensor_data[f'past_tracked_objects.{feature_name}'] = past_tracked_objects_tensor_list
        return ({'past_ego_states': past_ego_states_tensor, 'past_time_stamps': past_time_stamps_tensor}, list_tensor_data, {})

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> GenericAgents:
        """
        Unpacks the data returned from the scriptable core into an GenericAgents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed GenericAgents object.
        """
        ego_features = [list_tensor_data['generic_agents.ego'][0].detach().numpy()]
        agent_features = {}
        for key in list_tensor_data:
            if key.startswith('generic_agents.agents.'):
                feature_name = key[len('generic_agents.agents.'):]
                agent_features[feature_name] = [list_tensor_data[key][0].detach().numpy()]
        return GenericAgents(ego=ego_features, agents=agent_features)

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        output_dict: Dict[str, torch.Tensor] = {}
        output_list_dict: Dict[str, List[torch.Tensor]] = {}
        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}
        ego_history: torch.Tensor = tensor_data['past_ego_states']
        time_stamps: torch.Tensor = tensor_data['past_time_stamps']
        anchor_ego_state = ego_history[-1, :].squeeze()
        ego_tensor = build_generic_ego_features_from_tensor(ego_history, reverse=True)
        output_list_dict['generic_agents.ego'] = [ego_tensor]
        for feature_name in self.agent_features:
            if f'past_tracked_objects.{feature_name}' in list_tensor_data:
                agents: List[torch.Tensor] = list_tensor_data[f'past_tracked_objects.{feature_name}']
                agent_history = filter_agents_tensor(agents, reverse=True)
                if agent_history[-1].shape[0] == 0:
                    agents_tensor: torch.Tensor = torch.zeros((len(agent_history), 0, self._agents_states_dim)).float()
                else:
                    padded_agent_states = pad_agent_states(agent_history, reverse=True)
                    local_coords_agent_states = convert_absolute_quantities_to_relative(padded_agent_states, anchor_ego_state)
                    yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
                    agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)
                output_list_dict[f'generic_agents.agents.{feature_name}'] = [agents_tensor]
        return (output_dict, output_list_dict, output_list_list_dict)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {'past_ego_states': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_time_stamps': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_tracked_objects': {'iteration': '0', 'time_horizon': str(self.past_time_horizon), 'num_samples': str(self.num_past_poses), 'agent_features': ','.join(self.agent_features)}}

class VectorSetMapFeatureBuilder(ScriptableFeatureBuilder):
    """
    Feature builder for constructing map features in a vector set representation, similar to that of
        VectorNet ("VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation").
    """

    def __init__(self, map_features: List[str], max_elements: Dict[str, int], max_points: Dict[str, int], radius: float, interpolation_method: str) -> None:
        """
        Initialize vector set map builder with configuration parameters.
        :param map_features: name of map features to be extracted.
        :param max_elements: maximum number of elements to extract per feature layer.
        :param max_points: maximum number of points per feature to extract per feature layer.
        :param radius:  [m ]The query radius scope relative to the current ego-pose.
        :param interpolation_method: Interpolation method to apply when interpolating to maintain fixed size
            map elements.
        :return: Vector set map data including map element coordinates and traffic light status info.
        """
        super().__init__()
        self.map_features = map_features
        self.max_elements = max_elements
        self.max_points = max_points
        self.radius = radius
        self.interpolation_method = interpolation_method
        self._traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()
        for feature_name in self.map_features:
            try:
                VectorFeatureLayer[feature_name]
            except KeyError:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')
            if feature_name not in self.max_elements:
                raise RuntimeError(f'Max elements unavailable for {feature_name} feature layer!')
            if feature_name not in self.max_points:
                raise RuntimeError(f'Max points unavailable for {feature_name} feature layer!')

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return VectorSetMap

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'vector_set_map'

    @torch.jit.unused
    def get_scriptable_input_from_scenario(self, scenario: AbstractScenario) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
        coords, traffic_light_data = get_neighbor_vector_set_map(scenario.map_api, self.map_features, ego_coords, self.radius, route_roadblock_ids, [TrafficLightStatuses(traffic_light_data)])
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(coords, traffic_light_data[0], ego_state.rear_axle)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_scriptable_input_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the simulation objects
        :param current_input: planner input from sim
        :param initialization: planner initialization from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = current_input.history.ego_states[-1]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = initialization.route_roadblock_ids
        if current_input.traffic_light_data is None:
            raise ValueError('Cannot build VectorSetMap feature. PlannerInput.traffic_light_data is None')
        traffic_light_data = current_input.traffic_light_data
        coords, traffic_light_data = get_neighbor_vector_set_map(initialization.map_api, self.map_features, ego_coords, self.radius, route_roadblock_ids, [TrafficLightStatuses(traffic_light_data)])
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(coords, traffic_light_data[0], ego_state.rear_axle)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_scenario(scenario)
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_simulation(current_input, initialization)
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> VectorSetMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorSetMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorSetMap.
        """
        coords: Dict[str, List[FeatureDataType]] = {}
        traffic_light_data: Dict[str, List[FeatureDataType]] = {}
        availabilities: Dict[str, List[FeatureDataType]] = {}
        for key in list_tensor_data:
            if key.startswith('vector_set_map.coords.'):
                feature_name = key[len('vector_set_map.coords.'):]
                coords[feature_name] = [list_tensor_data[key][0].detach().numpy()]
            if key.startswith('vector_set_map.traffic_light_data.'):
                feature_name = key[len('vector_set_map.traffic_light_data.'):]
                traffic_light_data[feature_name] = [list_tensor_data[key][0].detach().numpy()]
            if key.startswith('vector_set_map.availabilities.'):
                feature_name = key[len('vector_set_map.availabilities.'):]
                availabilities[feature_name] = [list_tensor_data[key][0].detach().numpy()]
        return VectorSetMap(coords=coords, traffic_light_data=traffic_light_data, availabilities=availabilities)

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(self, coords: Dict[str, MapObjectPolylines], traffic_light_data: Dict[str, LaneSegmentTrafficLightData], anchor_state: StateSE2) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature transform.
        :param coords: Dictionary mapping feature name to polyline vector sets.
        :param traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
        :param anchor_state: The ego state to transform to vector.
        :return
           tensor_data: Packed tensor data.
           list_tensor_data: Packed List[tensor] data.
           list_list_tensor_data: Packed List[List[tensor]] data.
        """
        tensor_data: Dict[str, torch.Tensor] = {}
        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)
        tensor_data['anchor_state'] = anchor_state_tensor
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        for feature_name, feature_coords in coords.items():
            list_feature_coords: List[torch.Tensor] = []
            for element_coords in feature_coords.to_vector():
                list_feature_coords.append(torch.tensor(element_coords, dtype=torch.float64))
            list_tensor_data[f'coords.{feature_name}'] = list_feature_coords
            if feature_name in traffic_light_data:
                list_feature_tl_data: List[torch.Tensor] = []
                for element_tl_data in traffic_light_data[feature_name].to_vector():
                    list_feature_tl_data.append(torch.tensor(element_tl_data, dtype=torch.float32))
                list_tensor_data[f'traffic_light_data.{feature_name}'] = list_feature_tl_data
        return (tensor_data, list_tensor_data, {})

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        tensor_output: Dict[str, torch.Tensor] = {}
        list_tensor_output: Dict[str, List[torch.Tensor]] = {}
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {}
        anchor_state = tensor_data['anchor_state']
        for feature_name in self.map_features:
            if f'coords.{feature_name}' in list_tensor_data:
                feature_coords = list_tensor_data[f'coords.{feature_name}']
                feature_tl_data = [list_tensor_data[f'traffic_light_data.{feature_name}']] if f'traffic_light_data.{feature_name}' in list_tensor_data else None
                coords, tl_data, avails = convert_feature_layer_to_fixed_size(feature_coords, feature_tl_data, self.max_elements[feature_name], self.max_points[feature_name], self._traffic_light_encoding_dim, interpolation=self.interpolation_method if feature_name in [VectorFeatureLayer.LANE.name, VectorFeatureLayer.LEFT_BOUNDARY.name, VectorFeatureLayer.RIGHT_BOUNDARY.name, VectorFeatureLayer.ROUTE_LANES.name] else None)
                coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)
                list_tensor_output[f'vector_set_map.coords.{feature_name}'] = [coords]
                list_tensor_output[f'vector_set_map.availabilities.{feature_name}'] = [avails]
                if tl_data is not None:
                    list_tensor_output[f'vector_set_map.traffic_light_data.{feature_name}'] = [tl_data[0]]
        return (tensor_output, list_tensor_output, list_list_tensor_output)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        max_elements: List[str] = [f'{feature_name}.{feature_max_elements}' for feature_name, feature_max_elements in self.max_elements.items()]
        max_points: List[str] = [f'{feature_name}.{feature_max_points}' for feature_name, feature_max_points in self.max_points.items()]
        return {'neighbor_vector_set_map': {'radius': str(self.radius), 'interpolation_method': self.interpolation_method, 'map_features': ','.join(self.map_features), 'max_elements': ','.join(max_elements), 'max_points': ','.join(max_points)}, 'initial_ego_state': empty}

class TestVectorSetMap(unittest.TestCase):
    """Test vector set map feature representation."""

    def setUp(self) -> None:
        """Set up test case."""
        self.coords: Dict[str, List[npt.NDArray[np.float32]]] = {'LANE': [np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])], 'ROUTE': [np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])]}
        self.traffic_light_data: Dict[str, List[npt.NDArray[np.int64]]] = {'LANE': [np.array([[[0, 0, 0, 1], [1, 0, 0, 0]], [[0, 0, 0, 1], [1, 0, 0, 0]]])]}
        self.availabilities: Dict[str, List[npt.NDArray[np.bool_]]] = {'LANE': [np.array([[True, True], [True, True]])], 'ROUTE': [np.array([[True, True, False], [True, True, False]])]}

    def test_vector_set_map_feature(self) -> None:
        """
        Test the core functionality of features.
        """
        feature = VectorSetMap(coords=self.coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(VectorSetMap.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(list(feature.coords.values())[0][0], np.ndarray)
        self.assertIsInstance(list(feature.traffic_light_data.values())[0][0], np.ndarray)
        self.assertIsInstance(list(feature.availabilities.values())[0][0], np.ndarray)
        feature = feature.to_feature_tensor()
        self.assertIsInstance(list(feature.coords.values())[0][0], torch.Tensor)
        self.assertIsInstance(list(feature.traffic_light_data.values())[0][0], torch.Tensor)
        self.assertIsInstance(list(feature.availabilities.values())[0][0], torch.Tensor)

    def test_feature_layer_mismatch(self) -> None:
        """
        Test when same feature layers not present across feature.
        """
        coords: Dict[str, List[npt.NDArray[np.float32]]] = {'ROUTE': [np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])]}
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)
        availabilities: Dict[str, List[npt.NDArray[np.bool_]]] = {'LANE': [np.array([[True, True], [True, True]])]}
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=self.coords, traffic_light_data=self.traffic_light_data, availabilities=availabilities)

    def test_dimension_mismatch(self) -> None:
        """
        Test when feature dimensions don't match within or across feature layers.
        """
        coords: Dict[str, List[npt.NDArray[np.float32]]] = {'LANE': [np.array([[[0.0, 0.0]], [[0.0, 0.0]]])], 'ROUTE': [np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])]}
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)
        coords = {'LANE': [np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]), np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])], 'ROUTE': [np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]), np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])]}
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)
        coords = {'LANE': [np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])], 'ROUTE': [np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]), np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])]}
        availabilities: Dict[str, List[npt.NDArray[np.bool_]]] = {'LANE': [np.array([[True, True], [True, True]])], 'ROUTE': [np.array([[True, True, False], [True, True, False]]), np.array([[True, True, False], [True, True, False]])]}
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=availabilities)

    def test_bad_data(self) -> None:
        """
        Test data dimensions are wrong or missing.
        """
        coords: Dict[str, List[npt.NDArray[np.float32]]] = {'LANE': [np.array([[[0.0], [1.0]], [[0.0], [1.0]]])], 'ROUTE': [np.array([[[0.0], [1.0], [0.0]], [[0.0], [1.0], [0.0]]])]}
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)
        coords = {'LANE': [np.array([])], 'ROUTE': [np.array([])]}
        with self.assertRaises(RuntimeError):
            VectorSetMap(coords=coords, traffic_light_data=self.traffic_light_data, availabilities=self.availabilities)

class TestGenericAgents(unittest.TestCase):
    """Test agent feature representation."""

    def setUp(self) -> None:
        """Set up test case."""
        self.agent_features = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'TRAFFIC_CONE', 'BARRIER', 'CZONE_SIGN', 'GENERIC_OBJECT']
        self.ego: List[npt.NDArray[np.float32]] = [np.array(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))]
        self.ego_incorrect: List[npt.NDArray[np.float32]] = [np.array([0.0, 0.0, 0.0])]
        self.agents: Dict[str, List[npt.NDArray[np.float32]]] = {feature_name: [np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])] for feature_name in self.agent_features}
        self.agents_incorrect: Dict[str, List[npt.NDArray[np.float32]]] = {feature_name: [np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])] for feature_name in self.agent_features}

    def test_agent_feature(self) -> None:
        """
        Test the core functionality of features
        """
        feature = GenericAgents(ego=self.ego, agents=self.agents)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(GenericAgents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], np.ndarray)
        for feature_name in self.agent_features:
            self.assertIsInstance(feature.agents[feature_name][0], np.ndarray)
            self.assertIsInstance(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0), np.ndarray)
            self.assertEqual(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0).shape, (2, feature.agents_features_dim))
        feature = feature.to_feature_tensor()
        self.assertIsInstance(feature.ego[0], torch.Tensor)
        for feature_name in self.agent_features:
            self.assertIsInstance(feature.agents[feature_name][0], torch.Tensor)
            self.assertIsInstance(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0), torch.Tensor)
            self.assertEqual(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0).shape, (2, feature.agents_features_dim))

    def test_no_agents(self) -> None:
        """
        Test when there are no agents
        """
        agents: Dict[str, List[npt.NDArray[np.float32]]] = {feature_name: [np.empty((self.ego[0].shape[0], 0, 8), dtype=np.float32)] for feature_name in self.agent_features}
        feature = GenericAgents(ego=self.ego, agents=agents)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(GenericAgents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], np.ndarray)
        for feature_name in self.agent_features:
            self.assertIsInstance(feature.agents[feature_name][0], np.ndarray)
            self.assertIsInstance(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0), np.ndarray)
            self.assertEqual(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0).shape, (0, feature.agents_features_dim))
            self.assertEqual(feature.num_agents_in_sample(feature_name, 0), 0)
        feature = feature.to_feature_tensor()
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(GenericAgents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], torch.Tensor)
        for feature_name in self.agent_features:
            self.assertIsInstance(feature.agents[feature_name][0], torch.Tensor)
            self.assertIsInstance(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0), torch.Tensor)
            self.assertEqual(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0).shape, (0, feature.agents_features_dim))
            self.assertEqual(feature.num_agents_in_sample(feature_name, 0), 0)

    def test_incorrect_dimension(self) -> None:
        """
        Test when inputs dimension are incorrect
        """
        with self.assertRaises(AssertionError):
            GenericAgents(ego=self.ego, agents=self.agents_incorrect)
        with self.assertRaises(AssertionError):
            GenericAgents(ego=self.ego_incorrect, agents=self.agents)
        agents: Dict[str, List[npt.NDArray[np.float32]]] = {feature_name: [np.empty((self.ego[0].shape[0] + 1, 0, 8), dtype=np.float32)] for feature_name in self.agent_features}
        with self.assertRaises(AssertionError):
            GenericAgents(ego=self.ego, agents=agents)
        ego = copy.copy(self.ego)
        ego.append(np.zeros((self.ego[0].shape[0] + 1, self.ego[0].shape[1]), dtype=np.float32))
        with self.assertRaises(AssertionError):
            GenericAgents(ego=ego, agents=self.agents)
        with self.assertRaises(AssertionError):
            GenericAgents(ego=ego, agents=agents)

