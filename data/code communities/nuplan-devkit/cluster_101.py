# Cluster 101

class MLPlanner(AbstractPlanner):
    """
    Implements abstract planner interface.
    Used for simulating any ML planner trained through the nuPlan training framework.
    """

    def __init__(self, model: TorchModuleWrapper) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval = model.future_trajectory_sampling.step_time
        self._num_output_dim = model.future_trajectory_sampling.num_poses
        self._model_loader = ModelLoader(model)
        self._initialization: Optional[PlannerInitialization] = None
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        predictions = self._model_loader.infer(features)
        trajectory_predicted = cast(Trajectory, predictions['trajectory'])
        trajectory_tensor = trajectory_predicted.data
        trajectory = trajectory_tensor.cpu().detach().numpy()[0]
        return cast(npt.NDArray[np.float32], trajectory)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._model_loader.initialize()
        self._initialization = initialization

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        history = current_input.history
        start_time = time.perf_counter()
        features = self._model_loader.build_features(current_input, self._initialization)
        self._feature_building_runtimes.append(time.perf_counter() - start_time)
        start_time = time.perf_counter()
        predictions = self._infer_model(features)
        self._inference_runtimes.append(time.perf_counter() - start_time)
        states = transform_predictions_to_states(predictions, history.ego_states, self._future_horizon, self._step_interval)
        trajectory = InterpolatedTrajectory(states)
        return trajectory

    def generate_planner_report(self, clear_stats: bool=True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(compute_trajectory_runtimes=self._compute_trajectory_runtimes, feature_building_runtimes=self._feature_building_runtimes, inference_runtimes=self._inference_runtimes)
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []
        return report

class AbstractMLAgents(AbstractObservation):
    """
    Simulate agents based on an ML model.
    """

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario) -> None:
        """
        Initializes the AbstractEgoCentricMLAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        self._model_loader = ModelLoader(model)
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval_us = model.future_trajectory_sampling.step_time * 1000000.0
        self._num_output_dim = model.future_trajectory_sampling.num_poses
        self._scenario = scenario
        self._ego_anchor_state = scenario.initial_ego_state
        self.step_time = None
        self._agents: Optional[Dict[str, TrackedObject]] = None

    @abstractmethod
    def _infer_model(self, features: FeaturesType) -> TargetsType:
        """
        Makes a single inference on a Pytorch/Torchscript model.
        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        pass

    @abstractmethod
    def _update_observation_with_predictions(self, agent_predictions: TargetsType) -> None:
        """
        Update smart agent using the predictions from the ML model
        :param agent_predictions: The prediction output from the ML_model
        """
        pass

    def _initialize_agents(self) -> None:
        """
        Initializes the agents based on the first step of the scenario
        """
        unique_agents = {tracked_object.track_token: tracked_object for tracked_object in self._scenario.initial_tracked_objects.tracked_objects if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE}
        self._agents = sort_dict(unique_agents)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._initialize_agents()

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._initialize_agents()
        self._model_loader.initialize()

    def update_observation(self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer) -> None:
        """Inherited, see superclass."""
        self.step_time = next_iteration.time_point - iteration.time_point
        self._ego_anchor_state, _ = history.current_state
        initialization = PlannerInitialization(mission_goal=self._scenario.get_mission_goal(), route_roadblock_ids=self._scenario.get_route_roadblock_ids(), map_api=self._scenario.map_api)
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(next_iteration.index)
        current_input = PlannerInput(next_iteration, history, traffic_light_data)
        features = self._model_loader.build_features(current_input, initialization)
        predictions = self._infer_model(features)
        self._update_observation_with_predictions(predictions)

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert self._agents, 'ML agent observation has not been initialized!Please make sure initialize() is called before getting the observation.'
        return DetectionsTracks(TrackedObjects(list(self._agents.values())))

