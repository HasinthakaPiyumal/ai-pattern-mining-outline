# Cluster 136

class SimpleAgentAugmentor(AbstractAugmentor):
    """Simple data augmentation that adds Gaussian noise to the ego current position with specified mean and std."""

    def __init__(self, mean: List[float], std: List[float], low: List[float], high: List[float], augment_prob: float, use_uniform_noise: bool=False) -> None:
        """
        Initialize the augmentor.
        :param mean: mean of 3-dimensional Gaussian noise to [x, y, yaw]
        :param std: standard deviation of 3-dimenstional Gaussian noise to [x, y, yaw]
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)
        self._augment_prob = augment_prob

    def augment(self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario]=None) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return (features, targets)
        for batch_idx in range(len(features['agents'].ego)):
            features['agents'].ego[batch_idx][-1] += self._random_offset_generator.sample()
        return (features, targets)

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(param=self._augment_prob, param_name=f'self._augment_prob={self._augment_prob!r}'.partition('=')[0].split('.')[1], scaling_direction=ScalingDirection.MAX)

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(List[ParameterToScale], self._random_offset_generator.get_schedulable_attributes())

class AgentDropoutAugmentor(AbstractAugmentor):
    """Data augmentation that randomly drops out a part of agents in the scene."""

    def __init__(self, augment_prob: float, dropout_rate: float) -> None:
        """
        Initialize the augmentor.
        :param augment_prob: Probability between 0 and 1 of applying the data augmentation.
        :param dropout_rate: Rate of agents in the scenes to drop out - 0 means no dropout.
        """
        self._augment_prob = augment_prob
        self._dropout_rate = dropout_rate

    def augment(self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario]=None) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return (features, targets)
        for batch_idx in range(len(features['agents'].agents)):
            num_agents = features['agents'].agents[batch_idx].shape[1]
            keep_mask = np.random.choice([True, False], num_agents, p=[1.0 - self._dropout_rate, self._dropout_rate])
            agent_indices = np.arange(num_agents)[keep_mask]
            features['agents'].agents[batch_idx] = features['agents'].agents[batch_idx].take(agent_indices, axis=1)
        return (features, targets)

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(param=self._augment_prob, param_name=f'self._augment_prob={self._augment_prob!r}'.partition('=')[0].split('.')[1], scaling_direction=ScalingDirection.MAX)

class GenericAgentDropoutAugmentor(AbstractAugmentor):
    """Data augmentation that randomly drops out a part of agents in the scene."""

    def __init__(self, augment_prob: float, dropout_rate: float) -> None:
        """
        Initialize the augmentor.
        :param augment_prob: Probability between 0 and 1 of applying the data augmentation.
        :param dropout_rate: Rate of agents in the scenes to drop out - 0 means no dropout.
        """
        self._augment_prob = augment_prob
        self._dropout_rate = dropout_rate

    def augment(self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario]=None) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return (features, targets)
        for feature_name in features['generic_agents'].agents.keys():
            for batch_idx in range(len(features['generic_agents'].agents[feature_name])):
                num_agents = features['generic_agents'].agents[feature_name][batch_idx].shape[1]
                keep_mask = np.random.choice([True, False], num_agents, p=[1.0 - self._dropout_rate, self._dropout_rate])
                agent_indices = np.arange(num_agents)[keep_mask]
                features['generic_agents'].agents[feature_name][batch_idx] = features['generic_agents'].agents[feature_name][batch_idx].take(agent_indices, axis=1)
        return (features, targets)

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['generic_agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(param=self._augment_prob, param_name=f'self._augment_prob={self._augment_prob!r}'.partition('=')[0].split('.')[1], scaling_direction=ScalingDirection.MAX)

class KinematicHistoryGenericAgentAugmentor(AbstractAugmentor):
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory history that
    satisfies a set of kinematic constraints.

    This involves constrained minimization of the following objective:
    * minimize dist(perturbed_trajectory, ground_truth_trajectory)


    Simple data augmentation that adds Gaussian noise to the ego current position with specified mean and std.
    """

    def __init__(self, dt: float, mean: List[float], std: List[float], low: List[float], high: List[float], augment_prob: float, use_uniform_noise: bool=False) -> None:
        """
        Initialize the augmentor.
        :param dt: Time interval between trajectory points.
        :param mean: mean of 3-dimensional Gaussian noise to [x, y, yaw]
        :param std: standard deviation of 3-dimenstional Gaussian noise to [x, y, yaw]
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._dt = dt
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)
        self._augment_prob = augment_prob

    def safety_check(self, ego: npt.NDArray[np.float32], all_agents: List[npt.NDArray[np.float32]]) -> bool:
        """
        Check if the augmented trajectory violates any safety check (going backwards, collision with other agents).
        :param ego: Perturbed ego feature tensor to be validated.
        :param all_agents: List of agent features to validate against.
        :return: Bool reflecting feature validity.
        """
        if np.diff(ego, axis=0)[-1][0] < 0.0001:
            return False
        for agents in all_agents:
            dist_to_the_closest_agent = np.min(np.linalg.norm(np.array(agents)[:, :, :2] - ego[-1, :2], axis=1))
            if dist_to_the_closest_agent < 2.5:
                return False
        return True

    def augment(self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario]=None) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return (features, targets)
        for batch_idx in range(len(features['generic_agents'].ego)):
            trajectory_length = len(features['generic_agents'].ego[batch_idx]) - 1
            _optimizer = ConstrainedNonlinearSmoother(trajectory_length, self._dt)
            ego_trajectory: npt.NDArray[np.float32] = np.copy(features['generic_agents'].ego[batch_idx])
            ego_trajectory[-1][:3] += self._random_offset_generator.sample()
            ego_x, ego_y, ego_yaw, ego_vx, ego_vy, ego_ax, ego_ay = ego_trajectory.T
            ego_velocity = np.linalg.norm(ego_trajectory[:, 3:5], axis=1)
            x_curr = [ego_x[0], ego_y[0], ego_yaw[0], ego_velocity[0]]
            ref_traj = ego_trajectory[:, :3]
            _optimizer.set_reference_trajectory(x_curr, ref_traj)
            try:
                sol = _optimizer.solve()
            except RuntimeError:
                logger.error('Smoothing failed with status %s! Use G.T. instead' % sol.stats()['return_status'])
                return (features, targets)
            if not sol.stats()['success']:
                logger.warning('Smoothing failed with status %s! Use G.T. instead' % sol.stats()['return_status'])
                return (features, targets)
            ego_perturb: npt.NDArray[np.float32] = np.vstack([sol.value(_optimizer.position_x), sol.value(_optimizer.position_y), sol.value(_optimizer.yaw), sol.value(_optimizer.speed) * np.cos(sol.value(_optimizer.yaw)), sol.value(_optimizer.speed) * np.sin(sol.value(_optimizer.yaw)), np.concatenate((sol.value(_optimizer.accel), np.zeros(1))) * np.cos(sol.value(_optimizer.yaw)), np.concatenate((sol.value(_optimizer.accel), np.zeros(1))) * np.sin(sol.value(_optimizer.yaw))])
            ego_perturb = ego_perturb.T
            agents: List[npt.NDArray[np.float32]] = [agent_features[batch_idx] for agent_features in features['generic_agents'].agents.values()]
            if self.safety_check(ego_perturb, agents):
                features['generic_agents'].ego[batch_idx] = np.float32(ego_perturb)
        return (features, targets)

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['generic_agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(param=self._augment_prob, param_name=f'self._augment_prob={self._augment_prob!r}'.partition('=')[0].split('.')[1], scaling_direction=ScalingDirection.MAX)

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(List[ParameterToScale], self._random_offset_generator.get_schedulable_attributes())

class GaussianSmoothAgentAugmentor(AbstractAugmentor):
    """
    Augmentor that perturbs the ego's current position and future trajectory, then applies gaussian smoothing
    to generates a smooth trajectory over the current and future trajectory.
    """

    def __init__(self, mean: List[float], std: List[float], low: List[float], high: List[float], sigma: float, augment_prob: float, use_uniform_noise: bool=False) -> None:
        """
        Initialize the augmentor class.
        :param mean: Parameter to set mean vector of the Gaussian noise on [x, y, yaw].
        :param std: Parameter to set standard deviation vector of the Gaussian noise on [x, y, yaw].
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param sigma: Parameter to control the Gaussian smooth level.
        :param augment_prob: Probability between 0 and 1 of applying the data augmentation.
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._sigma = sigma
        self._augment_prob = augment_prob
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)

    def augment(self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario]=None) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return (features, targets)
        ego_trajectory: npt.NDArray[np.float32] = np.concatenate([features['agents'].ego[0][-1:, :], targets['trajectory'].data])
        trajectory_length, trajectory_dim = ego_trajectory.shape
        ego_trajectory += np.array([self._random_offset_generator.sample() for _ in range(trajectory_length)]) * np.expand_dims(np.exp(-np.arange(trajectory_length)), axis=1)
        ego_x, ego_y, ego_yaw = ego_trajectory.T
        step_t = np.linspace(0, 1, len(ego_x))
        step_resample_t = np.linspace(0, 1, 100)
        ego_resample_x = np.interp(step_resample_t, step_t, ego_x)
        ego_resample_y = np.interp(step_resample_t, step_t, ego_y)
        ego_resample_yaw = np.interp(step_resample_t, step_t, ego_yaw)
        ego_perturb_x = gaussian_filter1d(ego_resample_x, self._sigma)
        ego_perturb_y = gaussian_filter1d(ego_resample_y, self._sigma)
        ego_perturb_yaw = gaussian_filter1d(ego_resample_yaw, self._sigma)
        ego_perturb_x = np.interp(step_t, step_resample_t, ego_perturb_x)
        ego_perturb_y = np.interp(step_t, step_resample_t, ego_perturb_y)
        ego_perturb_yaw = np.interp(step_t, step_resample_t, ego_perturb_yaw)
        ego_perturb: npt.NDArray[np.float32] = np.vstack((ego_perturb_x, ego_perturb_y, ego_perturb_yaw)).T
        features['agents'].ego[0][-1] = ego_perturb[0]
        targets['trajectory'].data = ego_perturb[1:]
        return (features, targets)

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ['trajectory']

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(param=self._augment_prob, param_name=f'self._augment_prob={self._augment_prob!r}'.partition('=')[0].split('.')[1], scaling_direction=ScalingDirection.MAX)

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(List[ParameterToScale], self._random_offset_generator.get_schedulable_attributes())

class KinematicHistoryAgentAugmentor(AbstractAugmentor):
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory history that
    satisfies a set of kinematic constraints.

    This involves constrained minimization of the following objective:
    * minimize dist(perturbed_trajectory, ground_truth_trajectory)


    Simple data augmentation that adds Gaussian noise to the ego current position with specified mean and std.
    """

    def __init__(self, dt: float, mean: List[float], std: List[float], low: List[float], high: List[float], augment_prob: float, use_uniform_noise: bool=False) -> None:
        """
        Initialize the augmentor.
        :param dt: Time interval between trajectory points.
        :param mean: mean of 3-dimensional Gaussian noise to [x, y, yaw]
        :param std: standard deviation of 3-dimenstional Gaussian noise to [x, y, yaw]
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._dt = dt
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)
        self._augment_prob = augment_prob

    def safety_check(self, ego: npt.NDArray[np.float32], agents: npt.NDArray[np.float32]) -> bool:
        """
        Check if the augmented trajectory violates any safety check (going backwards, collision with other agents).
        :param ego: Perturbed ego feature tensor to be validated.
        :param agents: List of agent features to validate against.
        :return: Bool reflecting feature validity.
        """
        if np.diff(ego, axis=0)[-1][0] < 0.0001:
            return False
        dist_to_the_closest_agent = np.min(np.linalg.norm(np.array(agents)[:, :, :2] - ego[-1, :2], axis=1))
        if dist_to_the_closest_agent < 2.5:
            return False
        return True

    def augment(self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario]=None) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return (features, targets)
        for batch_idx in range(len(features['agents'].ego)):
            trajectory_length = len(features['agents'].ego[batch_idx]) - 1
            _optimizer = ConstrainedNonlinearSmoother(trajectory_length, self._dt)
            ego_trajectory: npt.NDArray[np.float32] = np.copy(features['agents'].ego[batch_idx])
            ego_trajectory[-1] += self._random_offset_generator.sample()
            ego_x, ego_y, ego_yaw = ego_trajectory.T
            ego_velocity = np.linalg.norm(np.diff(ego_trajectory[:, :2], axis=0), axis=1)
            x_curr = [ego_x[0], ego_y[0], ego_yaw[0], ego_velocity[0]]
            ref_traj = ego_trajectory
            _optimizer.set_reference_trajectory(x_curr, ref_traj)
            try:
                sol = _optimizer.solve()
            except RuntimeError:
                logger.error('Smoothing failed with status %s! Use G.T. instead' % sol.stats()['return_status'])
                return (features, targets)
            if not sol.stats()['success']:
                logger.warning('Smoothing failed with status %s! Use G.T. instead' % sol.stats()['return_status'])
                return (features, targets)
            ego_perturb: npt.NDArray[np.float32] = np.vstack([sol.value(_optimizer.position_x), sol.value(_optimizer.position_y), sol.value(_optimizer.yaw)])
            ego_perturb = ego_perturb.T
            if self.safety_check(ego_perturb, features['agents'].agents[batch_idx]):
                features['agents'].ego[batch_idx] = np.float32(ego_perturb)
        return (features, targets)

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(param=self._augment_prob, param_name=f'self._augment_prob={self._augment_prob!r}'.partition('=')[0].split('.')[1], scaling_direction=ScalingDirection.MAX)

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(List[ParameterToScale], self._random_offset_generator.get_schedulable_attributes())

class KinematicAgentAugmentor(AbstractAugmentor):
    """
    Data augmentation that perturbs the current ego position and generates a feasible future trajectory that
    satisfies a set of kinematic constraints.

    This involves constrained minimization of the following objective:
    * minimize dist(perturbed_trajectory, ground_truth_trajectory)
    """

    def __init__(self, trajectory_length: int, dt: float, mean: List[float], std: List[float], low: List[float], high: List[float], augment_prob: float, use_uniform_noise: bool=False) -> None:
        """
        Initialize the augmentor.
        :param trajectory_length: Length of trajectory to be augmented.
        :param dt: Time interval between trajecotry points.
        :param mean: Parameter to set mean vector of the Gaussian noise on [x, y, yaw].
        :param std: Parameter to set standard deviation vector of the Gaussian noise on [x, y, yaw].
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: Probability between 0 and 1 of applying the data augmentation.
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)
        self._augment_prob = augment_prob
        self._optimizer = ConstrainedNonlinearSmoother(trajectory_length, dt)

    def augment(self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario]=None) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return (features, targets)
        features['agents'].ego[0][-1] += self._random_offset_generator.sample()
        ego_trajectory: npt.NDArray[np.float32] = np.concatenate([features['agents'].ego[0][-1:, :], targets['trajectory'].data])
        ego_x, ego_y, ego_yaw = ego_trajectory.T
        ego_velocity = np.linalg.norm(np.diff(ego_trajectory[:, :2], axis=0), axis=1)
        x_curr = [ego_x[0], ego_y[0], ego_yaw[0], ego_velocity[0]]
        ref_traj = ego_trajectory
        self._optimizer.set_reference_trajectory(x_curr, ref_traj)
        try:
            sol = self._optimizer.solve()
        except RuntimeError:
            logger.error('Smoothing failed with status %s! Use G.T. instead' % sol.stats()['return_status'])
            return (features, targets)
        if not sol.stats()['success']:
            logger.warning('Smoothing failed with status %s! Use G.T. instead' % sol.stats()['return_status'])
            return (features, targets)
        ego_perturb: npt.NDArray[np.float32] = np.vstack([sol.value(self._optimizer.position_x), sol.value(self._optimizer.position_y), sol.value(self._optimizer.yaw)])
        ego_perturb = ego_perturb.T
        features['agents'].ego[0][-1] = np.float32(ego_perturb[0])
        targets['trajectory'].data = np.float32(ego_perturb[1:])
        return (features, targets)

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ['trajectory']

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(param=self._augment_prob, param_name=f'self._augment_prob={self._augment_prob!r}'.partition('=')[0].split('.')[1], scaling_direction=ScalingDirection.MAX)

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(List[ParameterToScale], self._random_offset_generator.get_schedulable_attributes())

class TestStepwiseAugmentationSheduler(unittest.TestCase):
    """Test scenario scoring callback"""

    def setUp(self) -> None:
        """Set up test case."""
        super().setUp()
        self.max_augment_prob = 0.8
        self.pct_time_increasing = 0.5
        self.max_aug_attribute_pct_increase = 0.2
        self.initial_augment_prob = 0.5
        self.milestones = [0.25, 0.5, 0.75, 1.0]
        self.cur_step = 1
        self.total_steps = 2
        self.mock_trajectory_length = 12
        self.mock_dt = 0.5
        self.mock_mean = [1.0, 0.0, 0.0]
        self.mock_std = [1.0, 1.0, 0.5]
        self.mock_low = [0.0, -1.0, -0.5]
        self.mock_high = [1.0, 1.0, 0.5]
        self.mock_augmentation_probability = 0.5
        self.mock_use_uniform_noise = False

    def test_scale_augmentor(self) -> None:
        """
        Test scale_augmentor function.
        """
        augmentation_attribute_scheduler = StepwiseAugmentationAttributeScheduler(self.max_aug_attribute_pct_increase, self.pct_time_increasing, 'linear', self.milestones)
        augmentation_probability_scheduler = StepwiseAugmentationProbabilityScheduler(self.max_augment_prob, self.pct_time_increasing, 'linear', self.milestones)
        mock_augmentor = Mock(AbstractAugmentor)
        mock_augmentor._random_offset_generator = GaussianNoise(self.mock_mean, self.mock_std)
        mock_augmentor._augment_prob = self.mock_augmentation_probability
        mock_augmentor.__name__ = Mock(return_value='mock_augmentor')
        mock_augmentor.augmentation_probability = ParameterToScale(param=self.mock_augmentation_probability, param_name='_augment_prob', scaling_direction=ScalingDirection.MAX)
        mock_augmentor.get_schedulable_attributes = mock_augmentor._random_offset_generator.get_schedulable_attributes()
        augmentation_attribute_scheduler._scale_augmentor(mock_augmentor, self.cur_step, self.total_steps)
        augmentation_probability_scheduler._scale_augmentor(mock_augmentor, self.cur_step, self.total_steps)
        expected_mean = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_mean)
        expected_std = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_std)
        expected_augmentation_probability = self.max_augment_prob
        self.assertEqual(mock_augmentor._augment_prob, expected_augmentation_probability)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, expected_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, expected_std))

    def test_handle_scheduling(self) -> None:
        """
        Test _handle_scheduling function to ensure scaling doesn't happen on non milestone steps.
        """
        augmentation_attribute_scheduler = StepwiseAugmentationAttributeScheduler(self.max_aug_attribute_pct_increase, self.pct_time_increasing, 'milestones', self.milestones)
        augmentation_probability_scheduler = StepwiseAugmentationProbabilityScheduler(self.max_augment_prob, self.pct_time_increasing, 'milestones', self.milestones)
        mock_augmentor = Mock(AbstractAugmentor)
        mock_augmentor._random_offset_generator = GaussianNoise(self.mock_mean, self.mock_std)
        mock_augmentor._augment_prob = self.mock_augmentation_probability
        mock_augmentor.__name__ = Mock(return_value='mock_augmentor')
        mock_augmentor.augmentation_probability = ParameterToScale(param=self.mock_augmentation_probability, param_name='_augment_prob', scaling_direction=ScalingDirection.MAX)
        mock_augmentor.get_schedulable_attributes = mock_augmentor._random_offset_generator.get_schedulable_attributes()
        mock_trainer = Mock(pl.Trainer)
        mock_trainer.datamodule = Mock()
        mock_trainer.datamodule._train_set._augmentors = [mock_augmentor]
        non_milestone_cur_step = 0.1
        pct_progress = round(non_milestone_cur_step / (self.total_steps * self.pct_time_increasing), 2)
        augmentation_attribute_scheduler._handle_scheduling(mock_trainer, non_milestone_cur_step, self.total_steps, pct_progress)
        augmentation_probability_scheduler._handle_scheduling(mock_trainer, non_milestone_cur_step, self.total_steps, pct_progress)
        self.assertEqual(mock_augmentor._augment_prob, self.initial_augment_prob)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, self.mock_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, self.mock_std))
        pct_progress = round(self.cur_step / (self.total_steps * self.pct_time_increasing), 2)
        augmentation_attribute_scheduler._handle_scheduling(mock_trainer, self.cur_step, self.total_steps, pct_progress)
        augmentation_probability_scheduler._handle_scheduling(mock_trainer, self.cur_step, self.total_steps, pct_progress)
        expected_mean = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_mean)
        expected_std = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_std)
        expected_augmentation_probability = self.max_augment_prob
        self.assertEqual(mock_augmentor._augment_prob, expected_augmentation_probability)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, expected_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, expected_std))

    def test_on_batch_end_milestones(self) -> None:
        """
        Test on_batch_end function to ensure scaling doesn't happen after scheduling is completed using milestones strategy.
        """
        augmentation_attribute_scheduler = StepwiseAugmentationAttributeScheduler(self.max_aug_attribute_pct_increase, self.pct_time_increasing, 'milestones', self.milestones)
        augmentation_probability_scheduler = StepwiseAugmentationProbabilityScheduler(self.max_augment_prob, self.pct_time_increasing, 'milestones', self.milestones)
        mock_augmentor = Mock(AbstractAugmentor)
        mock_augmentor._random_offset_generator = GaussianNoise(self.mock_mean, self.mock_std)
        mock_augmentor._augment_prob = self.mock_augmentation_probability
        mock_augmentor.__name__ = Mock(return_value='mock_augmentor')
        mock_augmentor.augmentation_probability = ParameterToScale(param=self.mock_augmentation_probability, param_name='_augment_prob', scaling_direction=ScalingDirection.MAX)
        mock_augmentor.get_schedulable_attributes = mock_augmentor._random_offset_generator.get_schedulable_attributes()
        mock_trainer = Mock(pl.Trainer)
        mock_trainer.max_epochs = 2
        mock_trainer.num_training_batches = 1
        mock_trainer.datamodule = Mock()
        mock_trainer.datamodule._train_set._augmentors = [mock_augmentor]
        mock_module = Mock(pl.LightningModule)
        mock_trainer.global_step = 0
        augmentation_attribute_scheduler.on_batch_end(mock_trainer, mock_module)
        augmentation_probability_scheduler.on_batch_end(mock_trainer, mock_module)
        expected_mean = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_mean)
        expected_std = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_std)
        expected_augmentation_probability = self.max_augment_prob
        self.assertEqual(mock_augmentor._augment_prob, expected_augmentation_probability)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, expected_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, expected_std))
        mock_trainer.global_step = 1
        augmentation_attribute_scheduler.on_batch_end(mock_trainer, mock_module)
        augmentation_probability_scheduler.on_batch_end(mock_trainer, mock_module)
        self.assertEqual(mock_augmentor._augment_prob, expected_augmentation_probability)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, expected_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, expected_std))

