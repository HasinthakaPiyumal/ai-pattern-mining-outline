# Cluster 51

class TestFunctionValidation(unittest.TestCase):
    """
    A class to test that the function validation method works properly.
    """

    def test_assert_functions_swappable(self) -> None:
        """
        Tests that the assert_functions_swappable method functions properly.
        """
        test_methods: List[MethodSpecification] = [MethodSpecification(name='_none_none', input_args={}, kw_only_args=None, return_type='None'), MethodSpecification(name='_int_none', input_args={'x': 'int'}, kw_only_args=None, return_type='None'), MethodSpecification(name='_intd1_none', input_args={'x': 'int = 1'}, kw_only_args=None, return_type='None'), MethodSpecification(name='_intk1_none', input_args={}, kw_only_args={'x': 'int = 1'}, return_type='None'), MethodSpecification(name='_int_int', input_args={'x': 'int'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_intd1_int', input_args={'x': 'int = 1'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_intd2_int', input_args={'x': 'int = 2'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_int_float', input_args={'x': 'int'}, kw_only_args=None, return_type='float'), MethodSpecification(name='_float_int', input_args={'x': 'float'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_list_int_int', input_args={'x': 'List[int]'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_list_float_int', input_args={'x': 'List[float]'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_int_int_int', input_args={'x': 'int', 'y': 'int'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_int_intk1_int', input_args={'x': 'int'}, kw_only_args={'y': 'int = 1'}, return_type='int'), MethodSpecification(name='_int_intk2_int', input_args={'x': 'int'}, kw_only_args={'y': 'int = 2'}, return_type='int'), MethodSpecification(name='_float_int_int', input_args={'x': 'float', 'y': 'int'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_ndarray_float32_int', input_args={'x': 'npt.NDArray[np.float32]'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_ndarray_float64_int', input_args={'x': 'npt.NDArray[np.float64]'}, kw_only_args=None, return_type='int'), MethodSpecification(name='_duplicate_of_int_int', input_args={'x': 'int'}, kw_only_args=None, return_type='int')]
        for spec in test_methods:
            method_text = _get_method_text(spec)
            exec(method_text)
        for first_func_definition in test_methods:
            for second_func_definition in test_methods:
                first_func_name = first_func_definition.name
                second_func_name = second_func_definition.name
                first_func = locals()[first_func_name]
                second_func = locals()[second_func_name]
                if first_func_name.replace('_duplicate_of', '') != second_func_name.replace('_duplicate_of', ''):
                    with self.assertRaises(TypeError):
                        assert_functions_swappable(first_func, second_func)
                else:
                    assert_functions_swappable(first_func, second_func)

def _get_method_text(spec: MethodSpecification) -> str:
    """
    Gets the text of a method to use for unit testing.
    This method does nothing and raises a `NotImplementedError()` if it is called.
    :param spec: The method specification.
    """
    input_signature_items = [f'{kvp[0]}: {kvp[1]}' for kvp in spec.input_args.items()]
    if spec.kw_only_args is not None:
        input_signature_items.append('*')
        input_signature_items += [f'{kvp[0]}: {kvp[1]}' for kvp in spec.kw_only_args.items()]
    input_signature = ', '.join(input_signature_items)
    method_text = textwrap.dedent(f'\n        def {spec.name}({input_signature}) -> {spec.return_type}:\n            raise NotImplementedError()\n        ')
    return method_text

def assert_functions_swappable(first_func: Callable[..., Any], second_func: Callable[..., Any]) -> None:
    """
    Asserts that a second function is swappable for the supplied first function.
    "Swappable" means that they contain the same arguments, same default arguments, and same return type.
    :param first_func: The first func that is being replaced.
    :param second_func: The second func that is being replaced.
    """
    _assert_function_signature_types_match(first_func, second_func)
    _assert_function_defaults_match(first_func, second_func)
    _assert_function_kwdefaults_match(first_func, second_func)

class TestInterfaceValidation(unittest.TestCase):
    """
    Tests that the interface_validation utils works properly.
    """

    def test_assert_class_properly_implements_interface_correct(self) -> None:
        """
        Tests that the validation passes when a class properly implements an interface.
        """
        assert_class_properly_implements_interface(ValidationInterface, CorrectConcrete)

    def test_assert_class_properly_implements_interface_swapped_args(self) -> None:
        """
        Tests that the validation fails if the args are swapped.
        """
        with self.assertRaisesRegex(TypeError, 'is not a subclass'):
            assert_class_properly_implements_interface(CorrectConcrete, ValidationInterface)

    def test_assert_class_properly_implements_interface_incorrect_method(self) -> None:
        """
        Tests that the validation fails when a class improperly implements an interface method.
        """
        with self.assertRaisesRegex(TypeError, 'Types in function signature.*do not match'):
            assert_class_properly_implements_interface(ValidationInterface, IncorrectConcrete)

    def test_assert_class_properly_implements_interface_missing_method(self) -> None:
        """
        Tests that the validation fails when a class missing the interface method is passed.
        """
        with self.assertRaisesRegex(TypeError, 'methods.*missing'):
            assert_class_properly_implements_interface(ValidationInterface, ConcreteMissingInterfaceMethod)

    def test_assert_class_properly_implements_interface_no_hierarchy(self) -> None:
        """
        Tests that the validation fails when the concrete does not derive from the interface.
        """
        with self.assertRaisesRegex(TypeError, 'is not a subclass'):
            assert_class_properly_implements_interface(ValidationInterface, ConcreteDoesNotDerive)

    def test_assert_class_properly_implements_interface_multiple_inheritance(self) -> None:
        """
        Tests that the validation passes with the multiple inheritance use case.
        """
        assert_class_properly_implements_interface(ValidationInterface, CorrectConcreteMulti)
        assert_class_properly_implements_interface(SecondValidationInterface, CorrectConcreteMulti)

def assert_class_properly_implements_interface(interface_class_type: Type[Any], derived_class_type: Type[Any]) -> None:
    """
    Asserts that a particular class implements a specified interface.
    This is done with the following checks:
        * Makes sure that derived_class is a subclass of interface_class.
        * Checks that all abstract public methods in interface are in derived.
        * Checks that the function signatures in derived class are swappable with abstract methods in interface.
    If the checks fail, a TypeError is raised.
    :param interface_class_type: The type of the interface class.
    :param derived_class_type: The type of the derived class.
    """
    _assert_derived_is_child_of_base(interface_class_type, derived_class_type)
    interface_abstract_methods = _get_public_methods(interface_class_type, only_abstract=True)
    derived_public_methods = _get_public_methods(derived_class_type, only_abstract=False)
    _assert_abstract_methods_present(interface_class_type, derived_class_type, {k for k in interface_abstract_methods.keys()}, {k for k in derived_public_methods.keys()})
    for key in interface_abstract_methods:
        interface_method = interface_abstract_methods[key]
        derived_method = derived_public_methods[key]
        assert_functions_swappable(interface_method, derived_method)

def _assert_derived_is_child_of_base(interface_class_type: Type[Any], derived_class_type: Type[Any]) -> None:
    """
    Checks that derived is an instance of base.
    Throws a TypeError if it is not.
    :param interface_class: The interface class.
    :param derived_class: The derived class.
    """
    if not issubclass(derived_class_type, interface_class_type):
        raise TypeError(textwrap.dedent(f'\n            {derived_class_type} is not a subclass of {interface_class_type}.\n            '))

def _get_public_methods(class_type: Type[Any], only_abstract: bool) -> Dict[str, Callable[..., Any]]:
    """
    Get all of the public methods exposed on a class.
    This excludes magic methods and underscore-prefixed methods.
    :param class_type: The type of the class for which to get the methods.
    :param only_abstract: If true, only returns abstract methods.
    :return: Mapping of (method_name -> function_object).
    """
    all_functions = {tup[0]: tup[1] for tup in inspect.getmembers(class_type, predicate=inspect.isfunction) if tup[1].__qualname__.startswith(class_type.__qualname__)}
    public_functions = {key: all_functions[key] for key in all_functions if not key.startswith('_')}
    if only_abstract:
        public_functions = {key: public_functions[key] for key in public_functions if hasattr(public_functions[key], '__isabstractmethod__') and public_functions[key].__isabstractmethod__}
    return public_functions

def _assert_abstract_methods_present(interface_class_type: Type[Any], derived_class_type: Type[Any], interface_abstract_method_names: Set[str], derived_public_method_names: Set[str]) -> None:
    """
    Asserts that all public methods in interface are in derived.
    :param interface_class_type: The class type of interface.
    :param derived_class_type: The class type of derived.
    :param interface_abstract_method_names: The interface abstract method names.
    :param derived_public_method_names: The derived public method names.
    """
    missing_methods = [im for im in interface_abstract_method_names if im not in derived_public_method_names]
    if len(missing_methods) > 0:
        missing_method_names = ', '.join(missing_methods)
        raise TypeError(textwrap.dedent(f'\n            The following methods are missing in {derived_class_type}, which are abstract in {interface_class_type}: {missing_method_names}\n            '))

def _assert_function_signature_types_match(first_func: Callable[..., Any], second_func: Callable[..., Any]) -> None:
    """
    Checks that the types in two method's function signatures match.
    If a difference is found, a TypeError is raised.
    :param first_func: The first function that is being seconded.
    :param second_func: The second that is being used.
    """
    first_annotations = first_func.__annotations__
    second_annotations = second_func.__annotations__
    if first_annotations != second_annotations:
        first_annotations_values = list(first_annotations.items()) if first_annotations is not None else []
        second_annotations_values = list(second_annotations.items()) if second_annotations is not None else []
        first_annotations_str = ', '.join([f'{kvp[0]}: {kvp[1]}' for kvp in first_annotations_values])
        second_annotations_str = ', '.join([f'{kvp[0]}: {kvp[1]}' for kvp in second_annotations_values])
        raise TypeError(textwrap.dedent(f'\n                Types in function signature for {first_func} do not match.\n                First func: {first_annotations_str}\n                Second func: {second_annotations_str}\n            '))

def _assert_function_defaults_match(first_func: Callable[..., Any], second_func: Callable[..., Any]) -> None:
    """
    Checks that the defaults set for the functions match.
    If a difference is found, a TypeError is raised.
    :param first_func: The first function that is being seconded.
    :param second_func: The second that is being used.
    """
    first_defaults = first_func.__defaults__
    second_defaults = second_func.__defaults__
    if first_defaults != second_defaults:
        raise TypeError(textwrap.dedent(f'\n                Default values for function {first_func} do not match.\n                First func: {first_defaults}\n                Second func: {second_defaults}\n            '))

def _assert_function_kwdefaults_match(first_func: Callable[..., Any], second_func: Callable[..., Any]) -> None:
    """
    Checks that the kwdefaults set for the functions match.
    If a difference is found, a TypeError is raised.
    :param first_func: The first function that is being seconded.
    :param second_func: The second that is being used.
    """
    first_kwdefaults = first_func.__kwdefaults__
    second_kwdefaults = second_func.__kwdefaults__
    if first_kwdefaults != second_kwdefaults:
        first_kwdefault_values = list(first_kwdefaults.items()) if first_kwdefaults is not None else []
        second_kwdefault_values = list(second_kwdefaults.items()) if second_kwdefaults is not None else []
        first_kwdefault_str = ', '.join([f'{kvp[0]}: {kvp[1]}' for kvp in first_kwdefault_values])
        second_kwdefault_str = ', '.join([f'{kvp[0]}: {kvp[1]}' for kvp in second_kwdefault_values])
        raise TypeError(textwrap.dedent(f'\n                Kwdefaults values in function signature for {first_func} do not match.\n                First func: {first_kwdefault_str}\n                Second func: {second_kwdefault_str}\n            '))

class RemotePlanner(AbstractPlanner):
    """
    Remote planner delegates computation of trajectories to a docker container, with which communicates through
    grpc.
    """

    def __init__(self, submission_container_manager: Optional[SubmissionContainerManager]=None, submission_image: Optional[str]=None, container_name: Optional[str]=None, compute_trajectory_timeout: float=1) -> None:
        """
        Prepares the remote container for planning.
        :param submission_container_manager: Optional manager, if provided a container will be started by RemotePlanner
        :param submission_image: Docker image name for the submission_container_factory
        :param container_name: Name to assign to the submission container
        :param compute_trajectory_timeout: Timeout for computation of trajectory.
        """
        if submission_container_manager:
            missing_parameter_message = 'Parameters for SubmissionContainer are missing!'
            assert submission_image, missing_parameter_message
            assert container_name, missing_parameter_message
            self.port = None
        else:
            self.port = os.getenv('SUBMISSION_CONTAINER_PORT', 50051)
        self.submission_container_manager = submission_container_manager
        self.submission_image = submission_image
        self.container_name = container_name
        self._channel = None
        self._stub = None
        self.serialized_observation: Optional[List[bytes]] = None
        self.serialized_state: Optional[List[bytes]] = None
        self.sample_interval: Optional[float] = None
        self._compute_trajectory_timeout = compute_trajectory_timeout

    def __reduce__(self) -> Tuple[Type[RemotePlanner], Tuple[Optional[SubmissionContainerManager], Optional[str], Optional[str]]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class
        """
        return (self.__class__, (self.submission_container_manager, self.submission_image, self.container_name))

    def name(self) -> str:
        """Inherited, see superclass."""
        return 'RemotePlanner'

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks

    @staticmethod
    def _planner_initializations_to_message(initialization: PlannerInitialization) -> chpb.PlannerInitializationLight:
        """
        Converts a PlannerInitialization to the message specified in the protocol files.
        :param initialization: The initialization parameters for the planner
        :return: A initialization message
        """
        try:
            mission_goal = proto_se2_from_se2(initialization.mission_goal)
        except AttributeError as e:
            logger.error('Mission goal was None!')
            raise e
        planner_initialization = chpb.PlannerInitializationLight(route_roadblock_ids=initialization.route_roadblock_ids, mission_goal=mission_goal, map_name=initialization.map_api.map_name)
        return planner_initialization

    def initialize(self, initialization: PlannerInitialization, timeout: float=5) -> None:
        """
        Creates the container manager, and runs the specified docker image. The communication port is created using
        the PID from the ray worker. Sends a request to initialize the remote planner.
        :param initialization: List of PlannerInitialization objects
        :param timeout: for planner initialization
        """
        if self.submission_container_manager:
            submission_container = try_n_times(self.submission_container_manager.get_submission_container, [self.submission_image, self.container_name, find_free_port_number()], {}, (docker.errors.APIError,), max_tries=10)
            self.port = submission_container.port
            submission_container.start()
            submission_container.wait_until_running(timeout=5)
        self._channel = grpc.insecure_channel(f'{NETWORK}:{self.port}')
        self._stub = chpb_grpc.DetectionTracksChallengeStub(self._channel)
        logger.info('Client sending planner initialization request...')
        planner_initializations_message = self._planner_initializations_to_message(initialization)
        logger.info(f'Trying to communicate on port {NETWORK}:{self.port}')
        try:
            _, _ = keep_trying(self._stub.InitializePlanner, [planner_initializations_message], {}, errors=(grpc.RpcError,), timeout=timeout)
        except Exception as e:
            submission_logger.error('Planner initialization failed!')
            submission_logger.error(e)
            raise e
        logger.info('Planner initialized!')

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Planner input for which trajectory should be computed
        :return: Trajectory representing the predicted ego's position in future for every input iteration
        """
        logger.debug('Client sending planner input: %s' % current_input)
        trajectory = self._compute_trajectory(self._stub, current_input=current_input)
        return trajectory

    def _compute_trajectory(self, stub: chpb_grpc.DetectionTracksChallengeStub, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Sends a request to compute the trajectory given the PlannerInput to the remote planner.
        :param stub: Service interface
        :param current_input: Planner input for which a trajectory should be computed.
        :return: Trajectory representing the predicted ego's position in future for every input iteration
        """
        logging.debug('Client sending observation...')
        self.serialized_state, self.serialized_observation, self.sample_interval = self._get_history_update(current_input)
        serialized_simulation_iteration = chpb.SimulationIteration(time_us=current_input.iteration.time_us, index=current_input.iteration.index)
        if self.sample_interval:
            serialized_buffer = chpb.SimulationHistoryBuffer(ego_states=self.serialized_state, observations=self.serialized_observation, sample_interval=self.sample_interval)
        else:
            serialized_buffer = chpb.SimulationHistoryBuffer(ego_states=self.serialized_state, observations=self.serialized_observation, sample_interval=None)
        tl_data = self._build_tl_message_from_planner_input(current_input)
        planner_input = chpb.PlannerInput(simulation_iteration=serialized_simulation_iteration, simulation_history_buffer=serialized_buffer, traffic_light_data=tl_data)
        try:
            trajectory_message = stub.ComputeTrajectory(planner_input, timeout=self._compute_trajectory_timeout)
        except grpc.RpcError as e:
            submission_logger.error('Trajectory computation service failed!')
            submission_logger.error(e)
            raise e
        return interp_traj_from_proto_traj(trajectory_message)

    def _get_history_update(self, planner_input: PlannerInput) -> Tuple[List[bytes], List[bytes], Optional[float]]:
        """
        Gets the new states and observations from the input. If no cache is present, the entire history is
        serialized, otherwise just the last element.
        :param planner_input: The input for planners
        :return: Tuple with new serialized state and observations.
        """
        keep_all_history = not self.serialized_state and (not self.serialized_observation)
        if keep_all_history:
            serialized_state = [pickle.dumps(state) for state in planner_input.history.ego_states]
            serialized_observation = [pickle.dumps(obs) for obs in planner_input.history.observations]
        else:
            last_ego_state, last_observations = planner_input.history.current_state
            serialized_state = [pickle.dumps(last_ego_state)]
            serialized_observation = [pickle.dumps(last_observations)]
        sample_interval = planner_input.history.sample_interval if not self.sample_interval else None
        return (serialized_state, serialized_observation, sample_interval)

    @staticmethod
    def _build_tl_message_from_planner_input(planner_input: PlannerInput) -> chpb.TrafficLightStatusData:
        tl_status_data: List[List[chpb.TrafficLightStatusData]]
        if planner_input.traffic_light_data is None:
            tl_status_data = [[]]
        else:
            tl_status_data = [proto_tl_status_data_from_tl_status_data(tl_status_data) for tl_status_data in planner_input.traffic_light_data]
        return tl_status_data

def proto_tl_status_data_from_tl_status_data(tl_status_data: TrafficLightStatusData) -> chpb.TrafficLightStatusData:
    """
    Serializes TrafficLightStatusData to a TrafficLightStatusData message
    :param tl_status_data: The TrafficLightStatusData object
    :return: The corresponding TrafficLightStatusData message
    """
    return chpb.TrafficLightStatusData(status=proto_tl_status_type_from_tl_status_type(tl_status_data.status), lane_connector_id=tl_status_data.lane_connector_id, timestamp=tl_status_data.timestamp)

class SkeletonTestSerializationCallback(unittest.TestCase):
    """Base class for TestsSerializationCallback* classes."""

    def _setUp(self) -> None:
        """Setup mocks for our tests."""
        self._serialization_type_to_extension_map = {'json': '.json', 'pickle': '.pkl.xz', 'msgpack': '.msgpack.xz'}
        self._serialization_type = getattr(self, '_serialization_type', '')
        self.assertIn(self._serialization_type, self._serialization_type_to_extension_map)
        self.output_folder = tempfile.TemporaryDirectory()
        self.callback = SerializationCallback(output_directory=self.output_folder.name, folder_name='sim', serialization_type=self._serialization_type, serialize_into_single_file=True)
        self.sim_manager = Mock(spec=AbstractSimulationTimeController)
        self.observation = Mock(spec=AbstractObservation)
        self.controller = Mock(spec=AbstractEgoController)
        super().setUp()

    @settings(deadline=None)
    @given(mock_timestamp=st.one_of(st.just(0), st.integers(min_value=1627066061949808, max_value=18446744073709551615)))
    def _dump_test_scenario(self, mock_timestamp: int) -> None:
        """
        Tests whether a scene can be dumped into a file and check that the keys are in the dumped scene.
        :param mock_timestamp: Mocked timestamp to pass to mock_get_traffic_light_status_at_iteration.
        """

        def mock_get_traffic_light_status_at_iteration(iteration: int) -> Generator[TrafficLightStatusData, None, None]:
            """Mocks MockAbstractScenario.get_traffic_light_status_at_iteration to return large numbers."""
            dummy_tl_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=mock_timestamp)
            yield dummy_tl_data
        scenario = MockAbstractScenario()
        scenario.get_traffic_light_status_at_iteration = Mock(spec=scenario.get_traffic_light_status_at_iteration)
        scenario.get_traffic_light_status_at_iteration.side_effect = mock_get_traffic_light_status_at_iteration
        self.setup = SimulationSetup(observations=self.observation, scenario=scenario, time_controller=self.sim_manager, ego_controller=self.controller)
        planner = Mock()
        planner.name = Mock(return_value='DummyPlanner')
        directory = self.callback._get_scenario_folder(planner.name(), scenario)
        self.assertEqual(str(directory), self.output_folder.name + '/sim/DummyPlanner/mock_scenario_type/mock_log_name/mock_scenario_name')
        self.callback.on_initialization_start(self.setup, planner)
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
        state_0 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(0))
        state_1 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(1000))
        history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_0, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=scenario.get_traffic_light_status_at_iteration(0)))
        history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_1, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=scenario.get_traffic_light_status_at_iteration(0)))
        for data in history.data:
            self.callback.on_step_end(self.setup, planner, data)
        self.callback.on_simulation_end(self.setup, planner, history)
        filename = 'mock_scenario_name' + self._serialization_type_to_extension_map[self._serialization_type]
        path = pathlib.Path(self.output_folder.name + '/sim/DummyPlanner/mock_scenario_type/mock_log_name/mock_scenario_name/' + filename)
        self.assertTrue(path.exists())
        if self._serialization_type == 'json':
            with open(path.absolute()) as f:
                data = json.load(f)
        elif self._serialization_type == 'msgpack':
            with lzma.open(str(path), 'rb') as f:
                data = msgpack.unpackb(f.read())
        elif self._serialization_type == 'pickle':
            with lzma.open(str(path), 'rb') as f:
                data = pickle.load(f)
        self.assertTrue(len(data) > 0)
        data = data[0]
        self.assertTrue('world' in data.keys())
        self.assertTrue('ego' in data.keys())
        self.assertTrue('trajectories' in data.keys())
        self.assertTrue('map' in data.keys())
        expected_traffic_light_data = next(scenario.get_traffic_light_status_at_iteration(0))
        actual_traffic_light_data_dict = data['traffic_light_status'][0]
        self.assertEqual(actual_traffic_light_data_dict['timestamp'], expected_traffic_light_data.timestamp)

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

def sample_indices_with_time_horizon(num_samples: int, time_horizon: float, time_interval: float) -> List[int]:
    """
    Samples the indices that can access N number of samples in a T time horizon from a sequence
    of temporal elemements with DT time interval.
    :param num_samples: number of elements to sample.
    :param time_horizon: [s] time horizon of sampled elements.
    :param time_interval: [s] time interval of sequence to sample from.
    :return: sampled indices that access the temporal sequence.
    """
    if time_horizon <= 0.0 or time_interval <= 0.0 or time_horizon < time_interval:
        raise ValueError(f'Time horizon {time_horizon} must be greater or equal than target time interval {time_interval} and both must be positive.')
    num_intervals = int(time_horizon / time_interval) + 1
    step_size = num_intervals // num_samples
    assert step_size > 0, f'Cannot get {num_samples} samples in a {time_horizon}s horizon at {time_interval}s intervals'
    indices = list(range(step_size, num_intervals + 1, step_size))
    indices = indices[:num_samples]
    assert len(indices) == num_samples, f'Expected {num_samples} samples but only {len(indices)} were sampled'
    return indices

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

def _fill_in_abstract_scenario_mock_parameters(scenario_mock: Mock, initial_ego_center_pose: StateSE2, initial_timestamp: int, traffic_light_statuses: List[TrafficLightStatusData], route_roadblock_ids: List[str]) -> None:
    """
    Helper function to configure a scenario mock with required parameters for VectorMapFeatureBuilder.
    :param scenario_mock: The AbstractScenario mock that we should update.
    :param initial_ego_center_pose: Ego vehicle center pose used to construct the scenario initial_ego_state.
    :param initial_timestamp: Initial timestamp corresponding to initial_ego_state for the test scenario.
    :param traffic_light_statuses: A list of TL statuses used to determine the traffic light scene at the 0th iteration.
    :param route_roadblock_ids: The route roadblock ids for the scenario.
    """
    scenario_mock.initial_ego_state = get_sample_ego_state(center=initial_ego_center_pose, time_us=initial_timestamp)
    scenario_mock.get_route_roadblock_ids.return_value = route_roadblock_ids

    def _get_traffic_light_status_at_iteration_patch(iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """A patch to populate traffic light states for the 0th iteration only."""
        if iteration != 0:
            raise ValueError('We expect the vector map builder to only use the 0th iteration TL states.')
        yield from traffic_light_statuses
    assert_functions_swappable(AbstractScenario.get_traffic_light_status_at_iteration, _get_traffic_light_status_at_iteration_patch)
    scenario_mock.get_traffic_light_status_at_iteration.side_effect = _get_traffic_light_status_at_iteration_patch

class TestVectorMapFeatureBuilderLaneMetadata(unittest.TestCase):
    """Test feature builder that constructs map features in vectorized format."""

    @patch(f'{PATCH_PREFIX}.AbstractScenario', autospec=True)
    def test_vectormap_example_metadata(self, mock_abstract_scenario: Mock) -> None:
        """
        Test VectorMapFeatureBuilder
        """
        test_radius = 50.0
        builder = VectorMapFeatureBuilder(radius=test_radius, connection_scales=None)
        initial_ego_center_pose = StateSE2(x=0.0, y=0.0, heading=0.0)
        timestamp = 1000000
        traffic_light_statuses = [TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=4000, timestamp=timestamp), TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=5000, timestamp=timestamp), TrafficLightStatusData(status=TrafficLightStatusType.YELLOW, lane_connector_id=5002, timestamp=timestamp), TrafficLightStatusData(status=TrafficLightStatusType.RED, lane_connector_id=5003, timestamp=timestamp), TrafficLightStatusData(status=TrafficLightStatusType.RED, lane_connector_id=5005, timestamp=timestamp)]
        route_roadblock_ids = ['60000', '70000']
        _fill_in_abstract_scenario_mock_parameters(scenario_mock=mock_abstract_scenario, initial_ego_center_pose=initial_ego_center_pose, initial_timestamp=timestamp, traffic_light_statuses=traffic_light_statuses, route_roadblock_ids=route_roadblock_ids)
        with patch_with_validation('nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder.get_neighbor_vector_map', _get_neighbor_vector_map_patch):
            vector_map_feature = builder.get_features_from_scenario(mock_abstract_scenario)
            self.assertIsInstance(vector_map_feature, VectorMap)
            self.assertEqual(vector_map_feature.num_of_batches, 1)
            self.assertTrue(vector_map_feature.is_valid)
            self.assertEqual(vector_map_feature.num_lanes_in_sample(0), 6)
            self.assertEqual(vector_map_feature.get_lane_coords(0).shape, (12, 2, 2))
            actual_traffic_light_data = vector_map_feature.traffic_light_data[0]
            expected_traffic_light_data = np.zeros_like(actual_traffic_light_data)
            tl_encoding_dict = LaneSegmentTrafficLightData._one_hot_encoding
            expected_traffic_light_data[[0, 1]] = tl_encoding_dict[TrafficLightStatusType.GREEN]
            expected_traffic_light_data[[6, 7, 10, 11]] = tl_encoding_dict[TrafficLightStatusType.RED]
            expected_traffic_light_data[[2, 3, 4, 5, 8, 9]] = tl_encoding_dict[TrafficLightStatusType.UNKNOWN]
            np_test.assert_allclose(actual_traffic_light_data, expected_traffic_light_data)
            actual_on_route_status = vector_map_feature.on_route_status[0]
            expected_on_route_status = np.zeros_like(actual_on_route_status)
            route_encoding_dict = LaneOnRouteStatusData._binary_encoding
            expected_on_route_status[[4, 5, 10, 11]] = route_encoding_dict[OnRouteStatusType.OFF_ROUTE]
            expected_on_route_status[[0, 1, 2, 3, 6, 7, 8, 9]] = route_encoding_dict[OnRouteStatusType.ON_ROUTE]
            np_test.assert_allclose(actual_on_route_status, expected_on_route_status)

class TestIndexTimeSampling(unittest.TestCase):
    """
    Tests the index time sampling functionality.
    """

    def test_round_time_horizon(self) -> None:
        """
        Tests the conversion of N number of samples and T time horizon (round) to sample indices.
        """
        time_interval = 0.05
        frames = np.arange(0, 20, time_interval)
        indices = sample_indices_with_time_horizon(num_samples=10, time_horizon=8.0, time_interval=time_interval)
        samples = frames[indices]
        assert np.allclose(samples, np.array([0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0]))

    def test_non_round_time_horizon(self) -> None:
        """
        Tests the conversion of N number of samples and T time horizon (non-round) to sample indices.
        """
        time_interval = 0.05
        frames = np.arange(0, 20, time_interval)
        indices = sample_indices_with_time_horizon(num_samples=12, time_horizon=1.2, time_interval=time_interval)
        samples = frames[indices]
        assert np.allclose(samples, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]))

    def test_raises_error(self) -> None:
        """
        Tests the edge case of receiving a smaller time horizon than time interval.
        """
        self.assertRaises(ValueError, sample_indices_with_time_horizon, num_samples=3, time_horizon=0.3, time_interval=0.5)

class MockAbstractScenario(AbstractScenario):
    """Mock abstract scenario class used for testing."""

    def __init__(self, initial_time_us: TimePoint=TimePoint(time_us=1621641671099), time_step: float=0.5, number_of_future_iterations: int=10, number_of_past_iterations: int=0, initial_velocity: StateVector2D=StateVector2D(x=1.0, y=0.0), fixed_acceleration: StateVector2D=StateVector2D(x=0.0, y=0.0), number_of_detections: int=10, initial_ego_state: StateSE2=StateSE2(x=0.0, y=0.0, heading=0.0), mission_goal: StateSE2=StateSE2(10, 0, 0), tracked_object_types: List[TrackedObjectType]=[TrackedObjectType.VEHICLE]):
        """
        Create mocked scenario where ego starts with an initial velocity [m/s] and has a constant acceleration
            throughout (0 m/s^2 by default). The ego does not turn.
        :param initial_time_us: initial time from start point of scenario [us]
        :param time_step: time step in [s]
        :param number_of_future_iterations: number of iterations in the future
        :param number_of_past_iterations: number of iterations in the past
        :param initial_velocity: [m/s] velocity assigned to the ego at iteration 0
        :param fixed_acceleration: [m/s^2] constant ego acceleration throughout scenario
        :param number_of_detections: number of detections in the scenario
        :param initial_ego_state: Initial state of ego
        :param mission_goal: Dummy mission goal
        :param tracked_object_types: Types of tracked objects to mock
        """
        self._initial_time_us = initial_time_us
        self._time_step = time_step
        self._number_of_past_iterations = number_of_past_iterations
        self._number_of_future_iterations = number_of_future_iterations
        self._current_iteration = number_of_past_iterations
        self._total_iterations = number_of_past_iterations + number_of_future_iterations + 1
        self._tracked_object_types = tracked_object_types
        start_time_us = max(TimePoint(int(number_of_past_iterations * time_step * 1000000.0)), initial_time_us)
        time_horizon = (number_of_past_iterations + number_of_future_iterations) * time_step
        history_buffer = SimulationHistoryBuffer.initialize_from_list(buffer_size=10, ego_states=[EgoState.build_from_rear_axle(StateSE2(x=initial_ego_state.x, y=initial_ego_state.y, heading=initial_ego_state.heading), time_point=start_time_us, rear_axle_velocity_2d=initial_velocity, tire_steering_angle=0.0, rear_axle_acceleration_2d=fixed_acceleration, vehicle_parameters=self.ego_vehicle_parameters)], observations=[DetectionsTracks(TrackedObjects())], sample_interval=time_step)
        planner_input = PlannerInput(iteration=SimulationIteration(start_time_us, 0), history=history_buffer)
        planner = SimplePlanner(horizon_seconds=time_horizon, sampling_time=time_step, acceleration=fixed_acceleration.array)
        self._ego_states = planner.compute_trajectory(planner_input).get_sampled_trajectory()
        self._tracked_objects = [DetectionsTracks(TrackedObjects([get_sample_agent(token=str(idx + type_idx * number_of_detections), agent_type=agent_type, num_future_states=0) for idx in range(number_of_detections) for type_idx, agent_type in enumerate(self._tracked_object_types)])) for _ in range(self._total_iterations)]
        self._sensors = [Sensors(pointcloud={LidarChannel.MERGED_PC: np.eye(3) for _ in range(number_of_detections)}, images=None) for _ in range(self._total_iterations)]
        if len(self._ego_states) != len(self._tracked_objects) or len(self._ego_states) != self._total_iterations:
            raise RuntimeError('The dimensions of detections and ego trajectory is not the same!')
        self._mission_goal = mission_goal
        self._map_api = MockAbstractMap()
        self._token_suffix = str(uuid.uuid4())

    @property
    def token(self) -> str:
        """Implemented. See interface."""
        return f'mock_token_{self._token_suffix}'

    @property
    def log_name(self) -> str:
        """Implemented. See interface."""
        return 'mock_log_name'

    @property
    def scenario_name(self) -> str:
        """Implemented. See interface."""
        return 'mock_scenario_name'

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return get_pacifica_parameters()

    @property
    def scenario_type(self) -> str:
        """Implemented. See interface."""
        return 'mock_scenario_type'

    @property
    def map_api(self) -> AbstractMap:
        """Implemented. See interface."""
        return self._map_api

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        return self._time_step

    def get_number_of_iterations(self) -> int:
        """Implemented. See interface."""
        return self._number_of_future_iterations

    def get_time_point(self, iteration: int) -> TimePoint:
        """Implemented. See interface."""
        return self._ego_states[self._current_iteration + iteration].time_point

    def get_lidar_to_ego_transform(self) -> Transform:
        """Implemented. See interface."""
        return np.eye(4)

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Implemented. See interface."""
        return self._mission_goal

    def get_route_roadblock_ids(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def get_expert_goal_state(self) -> StateSE2:
        """Implemented. See interface."""
        return self._mission_goal

    def get_tracked_objects_at_iteration(self, iteration: int, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Implemented. See interface."""
        return self._tracked_objects[self._current_iteration + iteration]

    def get_tracked_objects_within_time_window_at_iteration(self, iteration: int, past_time_horizon: float, future_time_horizon: float, filter_track_tokens: Optional[Set[str]]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Implemented. See interface."""
        raise NotImplementedError

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]]=None) -> Sensors:
        """Implemented. See interface."""
        raise NotImplementedError

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Implemented. See interface."""
        return self._ego_states[self._current_iteration + iteration]

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Implemented. see interface."""
        dummy_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=1627066061949808)
        yield dummy_data

    def get_past_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """Gets past traffic light status."""
        dummy_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=1627066061949808)
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        for _ in range(num_samples):
            yield TrafficLightStatuses([dummy_data])

    def get_future_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """Gets future traffic light status."""
        dummy_data = TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=1, timestamp=1627066061949808)
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        for _ in range(num_samples):
            yield TrafficLightStatuses([dummy_data])

    def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Implemented. See interface."""
        ego_states = self.get_ego_future_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
        for state in ego_states:
            yield state.time_point

    def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Implemented. See interface."""
        ego_states = self.get_ego_past_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
        for state in ego_states:
            yield state.time_point

    def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Implemented. See interface."""
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._number_of_future_iterations - iteration >= indices[-1], f'Requested time horizon of {time_horizon}s is too long! Scenario future has length {(self._number_of_future_iterations - iteration) * self._time_step}s from the iteration {iteration}'
        for idx in indices:
            yield self._ego_states[self._current_iteration + iteration + idx]

    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Implemented. See interface."""
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._current_iteration + iteration >= indices[-1], f'Requested time horizon of {time_horizon}s is too long! Scenario past has length {(self._current_iteration + iteration) * self._time_step}s from the iteration {iteration}'
        for idx in reversed(indices):
            yield self._ego_states[self._current_iteration + iteration - idx]

    def get_past_sensors(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, channels: Optional[List[SensorChannel]]=None) -> Generator[Sensors, None, None]:
        """Implemented. See interface."""
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        for idx in indices:
            yield self._sensors[self._current_iteration + iteration - idx - 1]

    def get_past_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Implemented. See interface."""
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        if self._current_iteration + iteration < indices[-1]:
            raise ValueError(f'Requested time horizon of {time_horizon}s is too long! Scenario past has length {(self._current_iteration + iteration) * self._time_step}s from the iteration {iteration}')
        for idx in reversed(indices):
            yield self._tracked_objects[self._current_iteration + iteration - idx]

    def get_future_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Implemented. See interface."""
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._number_of_future_iterations - iteration >= indices[-1], f'Requested time horizon of {time_horizon}s is too long! Scenario future has length {(self._number_of_future_iterations - iteration) * self._time_step}s from the iteration {iteration}'
        for idx in indices:
            yield self._tracked_objects[self._current_iteration + iteration + idx]

def get_num_samples(num_samples: Optional[int], time_horizon: float, database_interval: float) -> int:
    """
    Set num samples based on the time_horizon and database interval if num_samples is not set
    :param num_samples: if None, it will be computed based on  math.floor(time_horizon / database_interval)
    :param time_horizon: [s] horizon in which we want to look into
    :param database_interval: interval of the database
    :return: number of samples to iterate over
    """
    return num_samples if num_samples else int(time_horizon / database_interval)

class TestMockAbstractScenario(unittest.TestCase):
    """
    A class to test the MockAbstractScenario utility class.
    """

    def test_mock_abstract_scenario_implements_abstract_scenario(self) -> None:
        """
        Tests that the mock abstract scenario class properly implements the interface.
        """
        assert_class_properly_implements_interface(AbstractScenario, MockAbstractScenario)

class TestMockAbstractScenarioBuilder(unittest.TestCase):
    """
    A class to test the MockAbstractScenarioBuilder utility class.
    """

    def test_mock_abstract_scenario_builder_implements_abstract_scenario_builder(self) -> None:
        """
        Tests that the mock abstract scenario builder class properly implements the interface.
        """
        assert_class_properly_implements_interface(AbstractScenarioBuilder, MockAbstractScenarioBuilder)

class TestNuPlanScenarioBuilder(unittest.TestCase):
    """
    Tests scenario filtering and construction functionality.
    """

    def test_nuplan_scenario_builder_implements_abstract_scenario_builder(self) -> None:
        """
        Tests that the NuPlanScenarioBuilder implements the AbstractScenarioBuilder interface.
        """
        assert_class_properly_implements_interface(AbstractScenarioBuilder, NuPlanScenarioBuilder)

    def test_get_scenarios_no_filters(self) -> None:
        """
        Tests that the get_scenarios() method functions properly
        With no additional filters applied.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method that validates the input args.
            """
            self.assertIsNone(params.filter_tokens)
            self.assertIsNone(params.filter_types)
            self.assertIsNone(params.filter_map_names)
            self.assertFalse(params.include_cameras)
            m1 = MockNuPlanScenario(token='a', scenario_type='type1')
            m2 = MockNuPlanScenario(token='b', scenario_type='type1')
            m3 = MockNuPlanScenario(token='c', scenario_type='type2')
            return {'type1': [m1, m2], 'type2': [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method.
            """
            return ['filename']
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file', db_file_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs', discover_log_dbs_patch):
            scenario_builder = NuPlanScenarioBuilder(data_root='foo', map_root='bar', sensor_root='qux', db_files=None, map_version='baz', max_workers=None, verbose=False, scenario_mapping=None, vehicle_parameters=None, include_cameras=False)
            scenario_filter = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None, num_scenarios_per_type=None, limit_total_scenarios=None, expand_scenarios=False, remove_invalid_goals=False, shuffle=False, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
            result = scenario_builder.get_scenarios(scenario_filter, Sequential())
            self.assertEqual(3, len(result))
            result.sort(key=lambda s: s.token)
            self.assertEqual('a', result[0].token)
            self.assertEqual('b', result[1].token)
            self.assertEqual('c', result[2].token)

    def test_get_scenarios_db_filters(self) -> None:
        """
        Tests that the get_scenarios() method functions properly with db filters applied.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method.
            """
            self.assertEqual(params.filter_tokens, ['a', 'b', 'c', 'd', 'e', 'f'])
            self.assertEqual(params.filter_types, ['type1', 'type2', 'type3'])
            self.assertEqual(params.filter_map_names, ['map1', 'map2'])
            self.assertTrue(params.include_cameras)
            self.assertTrue(params.log_file_absolute_path in ['filename1', 'filename2'])
            m1 = MockNuPlanScenario(token='a', scenario_type='type1')
            m2 = MockNuPlanScenario(token='b', scenario_type='type1')
            m3 = MockNuPlanScenario(token='c', scenario_type='type2')
            return {'type1': [m1, m2], 'type2': [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method.
            """
            return ['filename1', 'filename2', 'filename3']
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file', db_file_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs', discover_log_dbs_patch):
            scenario_builder = NuPlanScenarioBuilder(data_root='foo', map_root='bar', sensor_root='qux', db_files=None, map_version='baz', max_workers=None, verbose=False, scenario_mapping=None, vehicle_parameters=None, include_cameras=True)
            scenario_filter = ScenarioFilter(scenario_types=['type1', 'type2', 'type3'], scenario_tokens=['a', 'b', 'c', 'd', 'e', 'f'], log_names=['filename1', 'filename2'], map_names=['map1', 'map2'], num_scenarios_per_type=None, limit_total_scenarios=None, expand_scenarios=False, remove_invalid_goals=False, shuffle=False, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
            result = scenario_builder.get_scenarios(scenario_filter, Sequential())
            self.assertEqual(6, len(result))
            result.sort(key=lambda s: s.token)
            self.assertEqual('a', result[0].token)
            self.assertEqual('a', result[1].token)
            self.assertEqual('b', result[2].token)
            self.assertEqual('b', result[3].token)
            self.assertEqual('c', result[4].token)
            self.assertEqual('c', result[5].token)

    def test_get_scenarios_num_scenarios_per_type_filter(self) -> None:
        """
        Tests that the get_scenarios() method functions properly
        With a num_scenarios_per_type filter applied.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method
            """
            self.assertEqual(params.filter_tokens, ['a', 'b', 'c', 'd', 'e', 'f'])
            self.assertEqual(params.filter_types, ['type1', 'type2', 'type3'])
            self.assertEqual(params.filter_map_names, ['map1', 'map2'])
            self.assertEqual(params.include_cameras, False)
            self.assertTrue(params.log_file_absolute_path in ['filename1', 'filename2'])
            m1 = MockNuPlanScenario(token='a', scenario_type='type1')
            m2 = MockNuPlanScenario(token='b', scenario_type='type1')
            m3 = MockNuPlanScenario(token='c', scenario_type='type2')
            return {'type1': [m1, m2], 'type2': [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method
            """
            return ['filename1', 'filename2', 'filename3']
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file', db_file_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs', discover_log_dbs_patch):
            scenario_builder = NuPlanScenarioBuilder(data_root='foo', map_root='bar', sensor_root='qux', db_files=None, map_version='baz', max_workers=None, verbose=False, scenario_mapping=None, vehicle_parameters=None, include_cameras=False)
            scenario_filter = ScenarioFilter(scenario_types=['type1', 'type2', 'type3'], scenario_tokens=['a', 'b', 'c', 'd', 'e', 'f'], log_names=['filename1', 'filename2'], map_names=['map1', 'map2'], num_scenarios_per_type=2, limit_total_scenarios=None, expand_scenarios=False, remove_invalid_goals=False, shuffle=False, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
            result = scenario_builder.get_scenarios(scenario_filter, Sequential())
            self.assertEqual(4, len(result))
            self.assertEqual(2, sum((1 if s.scenario_type == 'type1' else 0 for s in result)))
            self.assertEqual(2, sum((1 if s.scenario_type == 'type2' else 0 for s in result)))

    def test_get_scenarios_total_num_scenarios_filter(self) -> None:
        """
        Tests that the get_scenarios() method functions properly
        With a total_num_scenarios filter.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method
            """
            self.assertEqual(params.filter_tokens, ['a', 'b', 'c', 'd', 'e', 'f'])
            self.assertEqual(params.filter_types, ['type1', 'type2', 'type3'])
            self.assertEqual(params.filter_map_names, ['map1', 'map2'])
            self.assertFalse(params.include_cameras)
            self.assertTrue(params.log_file_absolute_path in ['filename1', 'filename2'])
            m1 = MockNuPlanScenario(token='a', scenario_type='type1')
            m2 = MockNuPlanScenario(token='b', scenario_type='type1')
            m3 = MockNuPlanScenario(token='c', scenario_type='type2')
            return {'type1': [m1, m2], 'type2': [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method
            """
            return ['filename1', 'filename2', 'filename3']
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file', db_file_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs', discover_log_dbs_patch):
            scenario_builder = NuPlanScenarioBuilder(data_root='foo', map_root='bar', sensor_root='qux', db_files=None, map_version='baz', max_workers=None, verbose=False, scenario_mapping=None, vehicle_parameters=None, include_cameras=False)
            scenario_filter = ScenarioFilter(scenario_types=['type1', 'type2', 'type3'], scenario_tokens=['a', 'b', 'c', 'd', 'e', 'f'], log_names=['filename1', 'filename2'], map_names=['map1', 'map2'], num_scenarios_per_type=None, limit_total_scenarios=5, expand_scenarios=False, remove_invalid_goals=False, shuffle=False, timestamp_threshold_s=None, ego_displacement_minimum_m=None, ego_start_speed_threshold=None, ego_stop_speed_threshold=None, speed_noise_tolerance=None, token_set_path=None, fraction_in_token_set_threshold=None)
            result = scenario_builder.get_scenarios(scenario_filter, Sequential())
            self.assertEqual(5, len(result))

class TestNuPlanScenario(unittest.TestCase):
    """
    Tests scenario for NuPlan
    """

    def _make_test_scenario(self) -> NuPlanScenario:
        """
        Creates a sample scenario to use for testing.
        """
        return NuPlanScenario(data_root='data_root/', log_file_load_path='data_root/log_name.db', initial_lidar_token=int_to_str_token(1234), initial_lidar_timestamp=2345, scenario_type='scenario_type', map_root='map_root', map_version='map_version', map_name='map_name', scenario_extraction_info=ScenarioExtractionInfo(scenario_name='scenario_name', scenario_duration=20, extraction_offset=1, subsample_ratio=0.5), ego_vehicle_parameters=get_pacifica_parameters(), sensor_root='sensor_root')

    def _get_sampled_sensor_tokens_in_time_window_patch(self, expected_log_file: str, expected_sensor_data_source: SensorDataSource, expected_start_timestamp: int, expected_end_timestamp: int, expected_subsample_step: int) -> Callable[[str, SensorDataSource, int, int, int], Generator[str, None, None]]:
        """
        Creates a patch for the get_sampled_lidarpc_tokens_in_time_window function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_start_timestamp: The expected start timestamp with which the function is expected to be called.
        :param expected_end_timestamp: The expected end timestamp with which the function is expected to be called.
        :param expected_subsample_step: The expected subsample step with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_sensor_data_source: SensorDataSource, actual_start_timestamp: int, actual_end_timestamp: int, actual_subsample_step: int) -> Generator[str, None, None]:
            """
            The patch function for get_sampled_lidarpc_tokens_in_time_window.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_start_timestamp, actual_start_timestamp)
            self.assertEqual(expected_end_timestamp, actual_end_timestamp)
            self.assertEqual(expected_subsample_step, actual_subsample_step)
            num_tokens = int((expected_end_timestamp - expected_start_timestamp) / (expected_subsample_step * 1000000.0))
            for token in range(num_tokens):
                yield int_to_str_token(token)
        return fxn

    def _get_download_file_if_necessary_patch(self, expected_data_root: str, expected_log_file_load_path: str) -> Callable[[str, str], str]:
        """
        Creates a patch for the download_file_if_necessary function that validates the arguments.
        :param expected_data_root: The data_root with which the function is expected to be called.
        :param expected_log_file_load_path: The log_file_load_path with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_data_root: str, actual_log_file_load_path: str) -> str:
            """
            The generated patch function.
            """
            self.assertEqual(expected_data_root, actual_data_root)
            self.assertEqual(expected_log_file_load_path, actual_log_file_load_path)
            return actual_log_file_load_path
        return fxn

    def _get_sensor_data_from_sensor_data_tokens_from_db_patch(self, expected_log_file: str, expected_sensor_data_source: SensorDataSource, expected_sensor_class: Type[SensorDataTableRow], expected_tokens: List[str]) -> Callable[[str, SensorDataSource, Type[SensorDataTableRow], List[str]], Generator[SensorDataTableRow, None, None]]:
        """
        Creates a patch for the get_images_from_lidar_tokens_patch function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_sensor_data_source: The sensor source with which the function is expected to be called.
        :param expected_sensor_class: The sensor class with which the function is expected to be called.
        :param expected_tokens: The tokens with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_sensor_data_source: SensorDataSource, actual_sensor_class: Type[SensorDataTableRow], actual_tokens: List[str]) -> Generator[SensorDataTableRow, None, None]:
            """
            The patch function for get_sensor_data_from_sensor_data_tokens_from_db.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_sensor_class, actual_sensor_class)
            self.assertEqual(expected_tokens, actual_tokens)
            lidar_token = actual_tokens[0]
            if expected_sensor_class == LidarPc:
                yield LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token))
            elif expected_sensor_class == ImageDBRow.Image:
                camera_token = str_token_to_int(lidar_token) + CAMERA_OFFSET
                yield ImageDBRow.Image(token=int_to_str_token(camera_token), next_token=int_to_str_token(camera_token), prev_token=int_to_str_token(camera_token), ego_pose_token=int_to_str_token(camera_token), camera_token=int_to_str_token(camera_token), filename_jpg=f'image_{camera_token}', timestamp=camera_token, channel=CameraChannel.CAM_R0.value)
            else:
                self.fail(f'Unexpected type: {expected_sensor_class}.')
        return fxn

    def _load_point_cloud_patch(self, expected_lidar_pc: LidarPc, expected_local_store: LocalStore, expected_s3_store: S3Store) -> Callable[[LidarPc, LocalStore, S3Store], LidarPointCloud]:
        """
        Creates a patch for the _load_point_cloud function that validates the arguments.
        :param expected_lidar_pc: The lidar pc with which the function is expected to be called.
        :param expected_local_store: The LocalStore with which the function is expected to be called.
        :param expected_s3_store: The S3Store with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_lidar_pc: LidarPc, actual_local_store: LocalStore, actual_s3_store: S3Store) -> LidarPointCloud:
            """
            The patch function for load_point_cloud.
            """
            self.assertEqual(expected_lidar_pc, actual_lidar_pc)
            self.assertEqual(expected_local_store, actual_local_store)
            self.assertEqual(expected_s3_store, actual_s3_store)
            return LidarPointCloud(np.eye(3))
        return fxn

    def _load_image_patch(self, expected_local_store: LocalStore, expected_s3_store: S3Store) -> Callable[[ImageDBRow.Image, LocalStore, S3Store], Image]:
        """
        Creates a patch for the _load_image_patch function and validates that argument is an Image object.
        :param expected_local_store: The LocalStore with which the function is expected to be called.
        :param expected_s3_store: The S3Store with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_image: ImageDBRow.Image, actual_local_store: LocalStore, actual_s3_store: S3Store) -> Image:
            """
            The patch function for load_image.
            """
            self.assertEqual(expected_local_store, actual_local_store)
            self.assertEqual(expected_s3_store, actual_s3_store)
            self.assertTrue(isinstance(actual_image, ImageDBRow.Image))
            return Image(PilImg.new('RGB', (500, 500)))
        return fxn

    def _get_images_from_lidar_tokens_patch(self, expected_log_file: str, expected_tokens: List[str], expected_channels: List[str], expected_lookahead_window_us: int, expected_lookback_window_us: int) -> Callable[[str, List[str], List[str], int, int], Generator[ImageDBRow.Image, None, None]]:
        """
        Creates a patch for the get_images_from_lidar_tokens_patch function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_tokens: The expected tokens with which the function is expected to be called.
        :param expected_channels: The expected channels with which the function is expected to be called.
        :param expected_lookahead_window_us: The expected lookahead window with which the function is expected to be called.
        :param expected_lookahead_window_us: The expected lookback window with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_tokens: List[str], actual_channels: List[str], actual_lookahead_window_us: int=50000, actual_lookback_window_us: int=50000) -> Generator[ImageDBRow.Image, None, None]:
            """
            The patch function for get_images_from_lidar_tokens.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_tokens, actual_tokens)
            self.assertEqual(expected_channels, actual_channels)
            self.assertEqual(expected_lookahead_window_us, actual_lookahead_window_us)
            self.assertEqual(expected_lookback_window_us, actual_lookback_window_us)
            for camera_token, channel in enumerate(actual_channels):
                if channel != LidarChannel.MERGED_PC.value:
                    yield ImageDBRow.Image(token=int_to_str_token(camera_token), next_token=int_to_str_token(camera_token), prev_token=int_to_str_token(camera_token), ego_pose_token=int_to_str_token(camera_token), camera_token=int_to_str_token(camera_token), filename_jpg=f'image_{camera_token}', timestamp=camera_token, channel=channel)
        return fxn

    def _get_sampled_lidarpcs_from_db_patch(self, expected_log_file: str, expected_initial_token: str, expected_sensor_data_source: SensorDataSource, expected_sample_indexes: Union[Generator[int, None, None], List[int]], expected_future: bool) -> Callable[[str, str, SensorDataSource, Union[Generator[int, None, None], List[int]], bool], Generator[LidarPc, None, None]]:
        """
        Creates a patch for the get_sampled_lidarpcs_from_db function that validates the arguments.
        :param expected_log_file: The log file name with which the function is expected to be called.
        :param expected_initial_token: The initial token name with which the function is expected to be called.
        :param expected_sensor_data_source: The sensor source with which the function is expected to be called.
        :param expected_sample_indexes: The sample indexes with which the function is expected to be called.
        :param expected_future: The future with which the function is expected to be called.
        :return: The patch function.
        """

        def fxn(actual_log_file: str, actual_initial_token: str, actual_sensor_data_source: SensorDataSource, actual_sample_indexes: Union[Generator[int, None, None], List[int]], actual_future: bool) -> Generator[LidarPc, None, None]:
            """
            The patch function for get_images_from_lidar_tokens.
            """
            self.assertEqual(expected_log_file, actual_log_file)
            self.assertEqual(expected_initial_token, actual_initial_token)
            self.assertEqual(expected_sensor_data_source, actual_sensor_data_source)
            self.assertEqual(expected_sample_indexes, actual_sample_indexes)
            self.assertEqual(expected_future, actual_future)
            for idx in actual_sample_indexes:
                lidar_token = int_to_str_token(idx)
                yield LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token))
        return fxn

    def test_implements_abstract_scenario_interface(self) -> None:
        """
        Tests that NuPlanScenario properly implements AbstractScenario interface.
        """
        assert_class_properly_implements_interface(AbstractScenario, NuPlanScenario)

    def test_token(self) -> None:
        """
        Tests that the token method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(int_to_str_token(1234), scenario.token)

    def test_log_name(self) -> None:
        """
        Tests that the log_name method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual('log_name', scenario.log_name)

    def test_scenario_name(self) -> None:
        """
        Tests that the scenario_name method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(int_to_str_token(1234), scenario.scenario_name)

    def test_ego_vehicle_parameters(self) -> None:
        """
        Tests that the ego_vehicle_parameters method works properly.
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(get_pacifica_parameters(), scenario.ego_vehicle_parameters)

    def test_scenario_type(self) -> None:
        """
        Tests that the scenario_type method works properly
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual('scenario_type', scenario.scenario_type)

    def test_database_interval(self) -> None:
        """
        Tests that the database_interval method works properly
        """
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
            self.assertEqual(0.1, scenario.database_interval)

    def test_get_number_of_iterations(self) -> None:
        """
        Tests that the get_number_of_iterations method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn):
                scenario = self._make_test_scenario()
                self.assertEqual(10, scenario.get_number_of_iterations())

    def test_get_time_point(self) -> None:
        """
        Tests that the get_time_point method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        for iter_val in [0, 3, 5]:

            def token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch method for get_lidarpc_token_timstamp_from_db that validates the arguments.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(SensorDataSource(table='lidar_pc', sensor_table='lidar', sensor_token_column='lidar_token', channel='MergedPointCloud'), sensor_source)
                self.assertEqual(int_to_str_token(iter_val), token)
                return int(str_token_to_int(iter_val) + 5)
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.get_sensor_data_token_timestamp_from_db', token_timestamp_patch):
                scenario = self._make_test_scenario()
                self.assertEqual(iter_val + 5, scenario.get_time_point(iter_val).time_us)

    def test_get_tracked_objects_at_iteration(self) -> None:
        """
        Tests that the get_tracked_objects_at_iteration method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)
        for iter_val in [0, 2, 3]:

            def get_token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch for get_sensor_data_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(SensorDataSource(table='lidar_pc', sensor_table='lidar', sensor_token_column='lidar_token', channel='MergedPointCloud'), sensor_source)
                self.assertEqual(int_to_str_token(iter_val), token)
                return int(iter_val * 1000000.0)

            def tracked_objects_for_token_patch(log_file: str, token: str) -> Generator[TrackedObject, None, None]:
                """
                The patch for get_tracked_objects_for_lidarpc_token that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(int_to_str_token(iter_val), token)
                for idx in range(0, 4, 1):
                    box = OrientedBox(center=StateSE2(x=10, y=10, heading=10), length=10, width=10, height=10)
                    metadata = SceneObjectMetadata(token=int_to_str_token(idx + str_token_to_int(token)), track_token=int_to_str_token(idx + str_token_to_int(token) + 100), track_id=None, timestamp_us=0, category_name='foo')
                    if idx < 2:
                        yield Agent(tracked_object_type=TrackedObjectType.VEHICLE, oriented_box=box, velocity=StateVector2D(x=10, y=10), metadata=metadata)
                    else:
                        yield StaticObject(tracked_object_type=TrackedObjectType.CZONE_SIGN, oriented_box=box, metadata=metadata)

            def interpolate_future_waypoints_patch(waypoints: List[InterpolatableState], time_horizon: float, interval_s: float) -> List[Optional[InterpolatableState]]:
                """
                The patch for interpolate_future_waypoints that validates the arguments and generates fake data.
                """
                self.assertEqual(4, len(waypoints))
                self.assertEqual(0.5, interval_s)
                self.assertEqual(5, time_horizon)
                return waypoints

            def future_waypoints_for_agents_patch(log_file: str, agents_tokens: List[str], start_time: int, end_time: int) -> Generator[Tuple[str, Waypoint], None, None]:
                """
                The patch for get_future_waypoints_for_agents_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(iter_val * 1000000.0, start_time)
                self.assertEqual((iter_val + 5.5) * 1000000.0, end_time)
                self.assertEqual(2, len(agents_tokens))
                check_tokens = [str_token_to_int(t) for t in agents_tokens]
                check_tokens.sort()
                self.assertEqual(iter_val + 100, check_tokens[0])
                self.assertEqual(iter_val + 100 + 1, check_tokens[1])
                for i in range(8):
                    waypoint = Waypoint(time_point=TimePoint(time_us=i), oriented_box=OrientedBox(center=StateSE2(x=i, y=i, heading=i), length=i, width=i, height=i), velocity=None)
                    token = check_tokens[0] if i < 4 else check_tokens[1]
                    yield (int_to_str_token(token), waypoint)
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_for_lidarpc_token_from_db', tracked_objects_for_token_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db', future_waypoints_for_agents_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sensor_data_token_timestamp_from_db', get_token_timestamp_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.interpolate_future_waypoints', interpolate_future_waypoints_patch):
                scenario = self._make_test_scenario()
                agents = scenario.get_tracked_objects_at_iteration(iter_val, ground_truth_predictions)
                objects = agents.tracked_objects.tracked_objects
                self.assertEqual(4, len(objects))
                objects.sort(key=lambda x: str_token_to_int(x.metadata.token))
                for i in range(0, 2, 1):
                    test_obj = objects[i]
                    self.assertTrue(isinstance(test_obj, Agent))
                    self.assertEqual(iter_val + i, str_token_to_int(test_obj.metadata.token))
                    self.assertEqual(iter_val + i + 100, str_token_to_int(test_obj.metadata.track_token))
                    self.assertEqual(TrackedObjectType.VEHICLE, test_obj.tracked_object_type)
                    self.assertIsNotNone(test_obj.predictions)
                    object_waypoints = test_obj.predictions[0].waypoints
                    self.assertEqual(4, len(object_waypoints))
                    for j in range(len(object_waypoints)):
                        self.assertEqual(j + i * len(object_waypoints), object_waypoints[j].x)
                for i in range(2, 4, 1):
                    test_obj = objects[i]
                    self.assertTrue(isinstance(test_obj, StaticObject))
                    self.assertEqual(iter_val + i, str_token_to_int(test_obj.metadata.token))
                    self.assertEqual(iter_val + i + 100, str_token_to_int(test_obj.metadata.track_token))
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, test_obj.tracked_object_type)

    def test_get_tracked_objects_within_time_window_at_iteration(self) -> None:
        """
        Tests that the get_tracked_objects_within_time_window_at_iteration method works properly
        """
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        ground_truth_predictions = TrajectorySampling(num_poses=10, time_horizon=5, interval_length=None)
        for iter_val in [3, 4]:

            def get_token_timestamp_patch(log_file: str, sensor_source: SensorDataSource, token: str) -> int:
                """
                The patch for get_sensor_data_token_timestamp_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(SensorDataSource(table='lidar_pc', sensor_table='lidar', sensor_token_column='lidar_token', channel='MergedPointCloud'), sensor_source)
                self.assertEqual(int_to_str_token(iter_val), token)
                return int(iter_val * 1000000.0)

            def tracked_objects_within_time_interval_patch(log_file: str, start_timestamp: int, end_timestamp: int, filter_tokens: Optional[Set[str]]) -> Generator[TrackedObject, None, None]:
                """
                The patch for get_tracked_objects_for_lidarpc_token that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual((iter_val - 2) * 1000000.0, start_timestamp)
                self.assertEqual((iter_val + 2) * 1000000.0, end_timestamp)
                self.assertIsNone(filter_tokens)
                for time_idx in range(-2, 3, 1):
                    for idx in range(0, 4, 1):
                        box = OrientedBox(center=StateSE2(x=10, y=10, heading=10), length=10, width=10, height=10)
                        metadata = SceneObjectMetadata(token=int_to_str_token(idx + iter_val), track_token=int_to_str_token(idx + iter_val + 100), track_id=None, timestamp_us=(iter_val + time_idx) * 1000000.0, category_name='foo')
                        if idx < 2:
                            yield Agent(tracked_object_type=TrackedObjectType.VEHICLE, oriented_box=box, velocity=StateVector2D(x=10, y=10), metadata=metadata)
                        else:
                            yield StaticObject(tracked_object_type=TrackedObjectType.CZONE_SIGN, oriented_box=box, metadata=metadata)

            def interpolate_future_waypoints_patch(waypoints: List[InterpolatableState], time_horizon: float, interval_s: float) -> List[Optional[InterpolatableState]]:
                """
                The patch for interpolate_future_waypoints that validates the arguments and generates fake data.
                """
                self.assertEqual(4, len(waypoints))
                self.assertEqual(0.5, interval_s)
                self.assertEqual(5, time_horizon)
                return waypoints

            def future_waypoints_for_agents_patch(log_file: str, agents_tokens: List[str], start_time: int, end_time: int) -> Generator[Tuple[str, Waypoint], None, None]:
                """
                The patch for get_future_waypoints_for_agents_from_db that validates the arguments and generates fake data.
                """
                self.assertEqual('data_root/log_name.db', log_file)
                self.assertEqual(end_time - start_time, 5.5 * 1000000.0)
                self.assertEqual(2, len(agents_tokens))
                check_tokens = [str_token_to_int(t) for t in agents_tokens]
                check_tokens.sort()
                self.assertEqual(iter_val + 100, check_tokens[0])
                self.assertEqual(iter_val + 100 + 1, check_tokens[1])
                for i in range(8):
                    waypoint = Waypoint(time_point=TimePoint(time_us=i), oriented_box=OrientedBox(center=StateSE2(x=i, y=i, heading=i), length=i, width=i, height=i), velocity=None)
                    token = check_tokens[0] if i < 4 else check_tokens[1]
                    yield (int_to_str_token(token), waypoint)
            with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_tracked_objects_within_time_interval_from_db', tracked_objects_within_time_interval_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_future_waypoints_for_agents_from_db', future_waypoints_for_agents_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.get_sensor_data_token_timestamp_from_db', get_token_timestamp_patch), mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils.interpolate_future_waypoints', interpolate_future_waypoints_patch):
                scenario = self._make_test_scenario()
                agents = scenario.get_tracked_objects_within_time_window_at_iteration(iter_val, 2, 2, future_trajectory_sampling=ground_truth_predictions)
                objects = agents.tracked_objects.tracked_objects
                self.assertEqual(20, len(objects))
                num_objects = 2
                for window in range(0, 5, 1):
                    for object_num in range(0, 2, 1):
                        start_agent_idx = window * 2
                        test_obj = objects[start_agent_idx + object_num]
                        self.assertTrue(isinstance(test_obj, Agent))
                        self.assertEqual(iter_val + object_num, str_token_to_int(test_obj.metadata.token))
                        self.assertEqual(iter_val + object_num + 100, str_token_to_int(test_obj.metadata.track_token))
                        self.assertEqual(TrackedObjectType.VEHICLE, test_obj.tracked_object_type)
                        self.assertIsNotNone(test_obj.predictions)
                        object_waypoints = test_obj.predictions[0].waypoints
                        self.assertEqual(4, len(object_waypoints))
                        for j in range(len(object_waypoints)):
                            self.assertEqual(j + object_num * len(object_waypoints), object_waypoints[j].x)
                        start_obj_idx = 10 + window * 2
                        test_obj = objects[start_obj_idx + object_num]
                        self.assertTrue(isinstance(test_obj, StaticObject))
                        self.assertEqual(iter_val + object_num + num_objects, str_token_to_int(test_obj.metadata.token))
                        self.assertEqual(iter_val + object_num + num_objects + 100, str_token_to_int(test_obj.metadata.track_token))
                        self.assertEqual(TrackedObjectType.CZONE_SIGN, test_obj.tracked_object_type)

    def test_nuplan_scenario_memory_usage(self) -> None:
        """
        Test that repeatedly creating and destroying nuplan scenario does not cause memory leaks.
        """
        starting_usage = 0
        ending_usage = 0
        num_iterations = 5
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.download_file_if_necessary', download_file_patch_fxn):
            hpy = guppy.hpy()
            hpy.setrelheap()
            for i in range(0, num_iterations, 1):
                scenario = self._make_test_scenario()
                _ = scenario.token
                gc.collect()
                heap = hpy.heap()
                _ = heap.size
                if i == num_iterations - 2:
                    starting_usage = heap.size
                if i == num_iterations - 1:
                    ending_usage = heap.size
            memory_difference_in_mb = (ending_usage - starting_usage) / (1024 * 1024)
            max_allowable_growth_mb = max(0.1, 0.1 * starting_usage / (1024 * 1024))
            self.assertGreater(max_allowable_growth_mb, memory_difference_in_mb)

    @patch(f'{TEST_PATH}.LocalStore', autospec=True)
    @patch(f'{TEST_PATH}.S3Store', autospec=True)
    @patch(f'{TEST_PATH}.os.getenv')
    def test_get_sensors_at_iteration(self, mock_get_env: Mock, mock_s3_store: Mock, mock_local_store: Mock) -> None:
        """Test get_sensors_at_iteration."""
        mock_url = 'url'
        mock_get_env.side_effect = ['s3', mock_url]
        mock_s3_store.return_value = Mock(spec_set=S3Store)
        mock_local_store.return_value = Mock(spec_set=LocalStore)
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0) + 2345, expected_end_timestamp=int(21 * 1000000.0) + 2345, expected_subsample_step=2)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        with mock.patch(f'{TEST_PATH}.download_file_if_necessary', download_file_patch_fxn):
            scenario = self._make_test_scenario()
        for iter_val in [0, 3, 5]:
            lidar_token = int_to_str_token(iter_val)
            get_sensor_data_from_sensor_data_tokens_from_db_fxn = self._get_sensor_data_from_sensor_data_tokens_from_db_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_sensor_class=LidarPc, expected_tokens=[lidar_token])
            get_images_from_lidar_tokens_fxn = self._get_images_from_lidar_tokens_patch(expected_log_file='data_root/log_name.db', expected_tokens=[lidar_token], expected_channels=[CameraChannel.CAM_R0.value, LidarChannel.MERGED_PC.value], expected_lookahead_window_us=50000, expected_lookback_window_us=50000)
            load_lidar_fxn = self._load_point_cloud_patch(LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token)), mock_local_store.return_value, mock_s3_store.return_value)
            load_image_fxn = self._load_image_patch(mock_local_store.return_value, mock_s3_store.return_value)
            with mock.patch(f'{TEST_PATH_UTILS}.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch(f'{TEST_PATH}.get_sensor_data_from_sensor_data_tokens_from_db', get_sensor_data_from_sensor_data_tokens_from_db_fxn), mock.patch(f'{TEST_PATH}.get_images_from_lidar_tokens', get_images_from_lidar_tokens_fxn), mock.patch(f'{TEST_PATH}.load_point_cloud', load_lidar_fxn), mock.patch(f'{TEST_PATH}.load_image', load_image_fxn):
                sensors = scenario.get_sensors_at_iteration(iter_val, [CameraChannel.CAM_R0, LidarChannel.MERGED_PC])
                self.assertEqual(LidarChannel.MERGED_PC, list(sensors.pointcloud.keys())[0])
                self.assertEqual(CameraChannel.CAM_R0, list(sensors.images.keys())[0])
                mock_local_store.assert_called_with('sensor_root')
                mock_s3_store.assert_called_with(f'{mock_url}/sensor_blobs', show_progress=True)

    @patch(f'{TEST_PATH}.LocalStore', autospec=True)
    @patch(f'{TEST_PATH}.S3Store', autospec=True)
    @patch(f'{TEST_PATH}.os.getenv')
    def test_get_past_sensors(self, mock_get_env: Mock, mock_s3_store: Mock, mock_local_store: Mock) -> None:
        """Test get_past_sensors."""
        mock_url = 'url'
        mock_get_env.side_effect = ['s3', mock_url]
        mock_s3_store.return_value = Mock(spec_set=S3Store)
        mock_local_store.return_value = Mock(spec_set=LocalStore)
        lidarpc_tokens_patch_fxn = self._get_sampled_sensor_tokens_in_time_window_patch(expected_log_file='data_root/log_name.db', expected_sensor_data_source=get_lidarpc_sensor_data(), expected_start_timestamp=int(1 * 1000000.0 + 2345), expected_end_timestamp=int(21 * 1000000.0 + 2345), expected_subsample_step=2)
        lidar_token = int_to_str_token(9)
        get_sampled_lidarpcs_from_db_fxn = self._get_sampled_lidarpcs_from_db_patch(expected_log_file='data_root/log_name.db', expected_initial_token=int_to_str_token(0), expected_sensor_data_source=get_lidarpc_sensor_data(), expected_sample_indexes=[9], expected_future=False)
        get_images_from_lidar_tokens_fxn = self._get_images_from_lidar_tokens_patch(expected_log_file='data_root/log_name.db', expected_tokens=[lidar_token], expected_channels=[CameraChannel.CAM_R0.value, LidarChannel.MERGED_PC.value], expected_lookahead_window_us=50000, expected_lookback_window_us=50000)
        download_file_patch_fxn = self._get_download_file_if_necessary_patch(expected_data_root='data_root/', expected_log_file_load_path='data_root/log_name.db')
        load_lidar_fxn = self._load_point_cloud_patch(LidarPc(token=lidar_token, next_token=lidar_token, prev_token=lidar_token, ego_pose_token=lidar_token, lidar_token=lidar_token, scene_token=lidar_token, filename=f'lidar_{lidar_token}', timestamp=str_token_to_int(lidar_token)), mock_local_store.return_value, mock_s3_store.return_value)
        load_image_fxn = self._load_image_patch(mock_local_store.return_value, mock_s3_store.return_value)
        with mock.patch(f'{TEST_PATH}.download_file_if_necessary', download_file_patch_fxn), mock.patch(f'{TEST_PATH_UTILS}.get_sampled_sensor_tokens_in_time_window_from_db', lidarpc_tokens_patch_fxn), mock.patch(f'{TEST_PATH}.get_sampled_lidarpcs_from_db', get_sampled_lidarpcs_from_db_fxn), mock.patch(f'{TEST_PATH}.get_images_from_lidar_tokens', get_images_from_lidar_tokens_fxn), mock.patch(f'{TEST_PATH}.load_point_cloud', load_lidar_fxn), mock.patch(f'{TEST_PATH}.load_image', load_image_fxn):
            scenario = self._make_test_scenario()
            past_sensors = list(scenario.get_past_sensors(iteration=0, time_horizon=0.4, num_samples=1, channels=[CameraChannel.CAM_R0, LidarChannel.MERGED_PC]))
            self.assertEqual(1, len(past_sensors))
            self.assertEqual(LidarChannel.MERGED_PC, list(past_sensors[0].pointcloud.keys())[0])
            self.assertEqual(CameraChannel.CAM_R0, list(past_sensors[0].images.keys())[0])
            mock_local_store.assert_called_with('sensor_root')
            mock_s3_store.assert_called_with(f'{mock_url}/sensor_blobs', show_progress=True)

    @patch(f'{TEST_PATH}.download_file_if_necessary', Mock())
    @patch(f'{TEST_PATH}.absolute_path_to_log_name', Mock())
    @patch(f'{TEST_PATH}.get_images_from_lidar_tokens', Mock(return_value=[]))
    @patch(f'{TEST_PATH}.NuPlanScenario._find_matching_lidar_pcs')
    @patch(f'{TEST_PATH}.load_point_cloud')
    @patch(f'{TEST_PATH}.load_image')
    def test_get_past_sensors_no_channels(self, mock_load_image: Mock, mock_load_point_cloud: Mock, mock__find_matching_lidar_pcs: Mock) -> None:
        """Test get_past_sensors when no channels are passed."""
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.token = 'token'
        mock_load_point_cloud.return_value = Mock(spec_set=LidarPointCloud)
        mock__find_matching_lidar_pcs.return_value = iter([mock_lidar_pc])
        scenario = self._make_test_scenario()
        past_sensors = list(scenario.get_past_sensors(iteration=0, time_horizon=0.4, num_samples=1, channels=None))
        mock__find_matching_lidar_pcs.assert_called_once()
        mock_load_point_cloud.assert_called_once()
        mock_load_image.assert_not_called()
        self.assertIsNone(past_sensors[0].images)
        self.assertIsNotNone(past_sensors[0].pointcloud)

    @patch(f'{TEST_PATH}.download_file_if_necessary', Mock())
    @patch(f'{TEST_PATH}.absolute_path_to_log_name', Mock())
    @patch(f'{TEST_PATH}.get_images_from_lidar_tokens', Mock(return_value=[]))
    @patch(f'{TEST_PATH}.extract_sensor_tokens_as_scenario', Mock(return_value=[None]))
    @patch(f'{TEST_PATH}.get_sensor_data_from_sensor_data_tokens_from_db')
    @patch(f'{TEST_PATH}.load_point_cloud')
    @patch(f'{TEST_PATH}.load_image')
    def test_get_sensors_at_iteration_no_channels(self, mock_load_image: Mock, mock_load_point_cloud: Mock, mock_get_sensor_data_from_sensor_data_tokens_from_db: Mock) -> None:
        """Test get_past_sensors when no channels are passed."""
        mock_lidar_pc = Mock(spec=LidarPc)
        mock_lidar_pc.token = 'token'
        mock_load_point_cloud.return_value = Mock(spec_set=LidarPointCloud)
        mock_get_sensor_data_from_sensor_data_tokens_from_db.return_value = iter([mock_lidar_pc])
        scenario = self._make_test_scenario()
        sensors = scenario.get_sensors_at_iteration(iteration=0, channels=None)
        mock_get_sensor_data_from_sensor_data_tokens_from_db.assert_called_once()
        mock_load_point_cloud.assert_called_once()
        mock_load_image.assert_not_called()
        self.assertIsNone(sensors.images)
        self.assertIsNotNone(sensors.pointcloud)

def tl_status_data_from_proto_tl_status_data(tl_status_data: chpb.TrafficLightStatusData) -> TrafficLightStatusData:
    """
    Deserializes TrafficLightStatusType message to a TrafficLightStatusType object
    :param tl_status_data: The proto TrafficLightStatusType message
    :return: The corresponding TrafficLightStatusType object
    """
    return TrafficLightStatusData(status=tl_status_type_from_proto_tl_status_type(tl_status_data.status), lane_connector_id=tl_status_data.lane_connector_id, timestamp=tl_status_data.timestamp)

def tl_status_type_from_proto_tl_status_type(tl_status_type: chpb.TrafficLightStatusType) -> TrafficLightStatusType:
    """
    Deserializes TrafficLightStatusType message to a TrafficLightStatusType object
    :param tl_status_type: The proto TrafficLightStatusType message
    :return: The corresponding TrafficLightStatusType object
    """
    return TrafficLightStatusType.deserialize(tl_status_type.status_name)

def proto_tl_status_type_from_tl_status_type(tl_status_type: TrafficLightStatusType) -> chpb.TrafficLightStatusType:
    """
    Serializes TrafficLightStatusType to a TrafficLightStatusType message
    :param tl_status_type: The TrafficLightStatusType object
    :return: The corresponding TrafficLightStatusType message
    """
    return chpb.TrafficLightStatusType(status_name=tl_status_type.serialize())

class TestProtoConverters(unittest.TestCase):
    """Tests proto converters by checking if composition is idempotent."""

    def test_trajectory_conversions(self) -> None:
        """Tests conversions between trajectory object and messages."""
        trajectory = InterpolatedTrajectory([get_sample_ego_state(StateSE2(0, 1, 2)), get_sample_ego_state(StateSE2(1, 2, 3), time_us=1)])
        result = interp_traj_from_proto_traj(proto_traj_from_inter_traj(trajectory))
        for result_state, trajectory_state in zip(result.get_sampled_trajectory(), trajectory.get_sampled_trajectory()):
            np.allclose(result_state.to_split_state().linear_states, trajectory_state.to_split_state().linear_states)
            np.allclose(result_state.to_split_state().angular_states, trajectory_state.to_split_state().angular_states)

    def test_tl_status_type_conversions(self) -> None:
        """Tests conversions between TL status data and messages."""
        tl_status_type = TrafficLightStatusType.RED
        result = tl_status_type_from_proto_tl_status_type(proto_tl_status_type_from_tl_status_type(tl_status_type))
        self.assertEqual(tl_status_type, result)

    def test_tl_status_data_conversions(self) -> None:
        """Tests conversions between TL status type and messages."""
        tl_status = TrafficLightStatusData(TrafficLightStatusType.RED, 123, 456)
        result = tl_status_data_from_proto_tl_status_data(proto_tl_status_data_from_tl_status_data(tl_status))
        self.assertEqual(tl_status, result)

