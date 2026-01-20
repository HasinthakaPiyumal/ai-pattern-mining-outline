# Cluster 60

class TestTryNTimes(unittest.TestCase, HelperTestingSetup):
    """Test suite for tests that lets tests run multiple times before declaring failure."""

    def setUp(self) -> None:
        """Inherited, see superclass"""
        HelperTestingSetup.__init__(self)

    def test_fails_on_invalid_number_of_tries(self) -> None:
        """Tests that we calling this method with zero tries result in failure."""
        with self.assertRaises(AssertionError):
            _ = try_n_times(self.passing_function, [], {}, self.errors, max_tries=0)

    def test_pass_on_valid_cases(self) -> None:
        """Tests that for nominal cases the output of the function is returned."""
        result = try_n_times(self.passing_function, self.args, self.kwargs, self.errors, max_tries=1)
        self.assertEqual('result', result)
        self.passing_function.assert_called_once_with(*self.args, **self.kwargs)

    @patch('time.sleep')
    def test_fail_on_invalid_case_after_n_tries(self, mock_sleep: Mock) -> None:
        """Tests that the helper throws after too many attempts."""
        with self.assertRaises(self.errors[0]):
            _ = try_n_times(self.failing_function, self.args, self.kwargs, self.errors, max_tries=2, sleep_time=4.2)
        calls = [call(*self.args, **self.kwargs)] * 2
        self.failing_function.assert_has_calls(calls)
        mock_sleep.assert_called_with(4.2)

def try_n_times(fn: Callable[..., Any], args: List[Any], kwargs: Dict[Any, Any], errors: Tuple[Any], max_tries: int, sleep_time: float=0) -> Any:
    """
    Keeps calling a function with given parameters until maximum number of tries, catching a set of given errors.
    :param fn: The function to call
    :param args: Argument list
    :param kwargs" Keyword arguments
    :param errors: Expected errors to be ignored
    :param max_tries: Maximal number of tries before raising error
    :param sleep_time: Time waited between subsequent tried to the function call.
    :return: The return value of the given function
    """
    assert max_tries > 0, 'Number of tries must be a positive integer'
    attempts = 0
    error = None
    while attempts < max_tries:
        try:
            return fn(*args, **kwargs)
        except errors as e:
            error = e
            attempts += 1
            logging.warning(f'Tried to call {fn} raised {e}, trying {max_tries - attempts} more times.')
            time.sleep(sleep_time)
            pass
    if error:
        raise error

class TestKeepTrying(unittest.TestCase, HelperTestingSetup):
    """Test suite for tests that lets tests run until a timeout is reached before declaring failure."""

    def setUp(self) -> None:
        """Inherited, see superclass"""
        HelperTestingSetup.__init__(self)

    def test_fails_on_invalid_number_of_tries(self) -> None:
        """Tests that we calling this method with zero tries result in failure."""
        with self.assertRaises(AssertionError):
            _ = keep_trying(self.passing_function, [], {}, self.errors, timeout=0.0)

    def test_pass_on_valid_cases(self) -> None:
        """Tests that for nominal cases the output of the function is returned."""
        result, _ = keep_trying(self.passing_function, self.args, self.kwargs, self.errors, timeout=1)
        self.assertEqual('result', result)
        self.passing_function.assert_called_once_with(*self.args, **self.kwargs)

    def test_fail_on_invalid_case_after_timeout(self) -> None:
        """Tests that the helper throws after timeout."""
        with self.assertRaises(TimeoutError):
            _ = keep_trying(self.failing_function, self.args, self.kwargs, self.errors, timeout=1e-06, sleep_time=1e-05)
        self.failing_function.assert_called_with(*self.args, **self.kwargs)

def keep_trying(fn: Callable[..., Any], args: List[Any], kwargs: Dict[Any, Any], errors: Tuple[Any], timeout: float, sleep_time: float=0.1) -> Any:
    """
    Keeps calling a function with given parameters until timeout (at least once), catching a set of given errors.
    :param fn: The function to call
    :param args: Argument list
    :param kwargs" Keyword arguments
    :param errors: Expected errors to be ignored
    :param timeout: Maximal time before timeout (seconds)
    :param sleep_time: Time waited between subsequent tried to the function call.
    :return: The return value of the given function
    """
    assert timeout > 0, 'Timeout must be a positive real number'
    start_time = time.time()
    max_time = start_time + timeout
    first_run = True
    while time.time() < max_time or first_run:
        try:
            return (fn(*args, **kwargs), time.time() - start_time)
        except errors:
            first_run = False
            time.sleep(sleep_time)
    raise TimeoutError(f'Timeout on function call {fn}({args}{kwargs}) catching {errors}')

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

def find_free_port_number() -> int:
    """
    Finds a free port number
    :return: the port number of a free port
    """
    skt = socket.socket()
    skt.bind(('', 0))
    port = skt.getsockname()[1]
    skt.close()
    return int(port)

class SubmissionContainer:
    """Class handling a submission Docker container"""

    def __init__(self, submission_image: str, container_name: str, port: int):
        """
        :param submission_image: Name of the docker image of the submission
        :param container_name: Name for the container to be run
        :param port: Port number to be used for communication
        """
        self.submission_image = submission_image
        self.container_name = container_name
        self.port = port
        self.client: docker.client.DockerClient | None = None

    def __del__(self) -> None:
        """Stop the running container when the destructor is called."""
        self.stop()

    def start(self, cpus: str='0,1', gpus: list[str] | None=None) -> Any:
        """
        Starts the submission container given a docker client, and the submission details. It exposes the specified
        port, and volume mounts (read only) the data directory to make it available to the container.
        :param cpus: CPUs to be used by the submission container
        :param gpus: GPUs to be used by the submission container
        """
        if gpus is None:
            gpus = ['0']
        self.client = docker.from_env()
        self.stop()
        ports = {f'{str(self.port)}': self.port}
        self.client.containers.run(self.submission_image, name=self.container_name, detach=True, ports=ports, tty=True, environment={'SUBMISSION_CONTAINER_PORT': str(self.port)}, device_requests=[docker.types.DeviceRequest(device_ids=gpus, capabilities=[['gpu']])], cpuset_cpus=cpus, volumes={os.getenv('NUPLAN_DATA_ROOT', '~/nuplan/dataset'): {'bind': '/data/sets/nuplan', 'mode': 'ro'}})
        logging.debug(f'Started submission container with image: {self.submission_image} with port: {self.port}')
        return self.client.containers.get(self.container_name)

    def stop(self) -> None:
        """Checks if the submission container is running, if it is it stops and removes it."""
        try:
            container = self.client.containers.get(self.container_name)
        except NotFound:
            pass
        else:
            logging.debug('Stopping and removing pre-existing container')
            try:
                container.kill()
            except docker.errors.APIError:
                pass
            container.remove()

    def wait_until_running(self, timeout: float=3) -> None:
        """
        Waits until a container is running until timeout.
        :param timeout: timeout in seconds
        """

        def is_running(manager: SubmissionContainer) -> bool:
            """
            Checks if the container is running
            :param manager: The container manager
            :returns: True if the container is in running state
            """
            return bool(manager.client.api.inspect_container(manager.container_name)['State']['Status'] == 'running')
        keep_trying(is_running, [self], {}, (docker.errors.NotFound,), timeout)

class SubmissionContainerFactory:
    """Factory for SubmissionContainer"""

    @staticmethod
    def build_submission_container(submission_image: str, container_name: str, port: int) -> SubmissionContainer:
        """
        Builds a SubmissionContainer given submission image, container name and port
        :param submission_image: Name of the Docker image
        :param container_name: Name for the Docker container
        :param port: Port number
        :return: The constructed SubmissionContainer
        """
        return SubmissionContainer(submission_image, container_name, port)

class ImageIsRunnableValidator(BaseSubmissionValidator):
    """Checks if an image is runnable without errors"""

    def validate(self, submission: str) -> bool:
        """
        Checks that the queried image is runnable.
        :param submission: Queried image name
        :return: False if the image is not runnable, or the next validator on the chain if the validation passes
        """
        container_name = container_name_from_image_name(submission)
        submission_container = SubmissionContainer(submission_image=submission, container_name=container_name, port=find_free_port_number())
        _ = submission_container.start()
        try:
            submission_container.wait_until_running(timeout=1)
            logger.debug('Image is runnable')
        except TimeoutError:
            logger.error('Image is not runnable')
            self._failing_validator = ImageIsRunnableValidator
            return False
        try:
            submission_container.stop()
        except docker.errors.APIError:
            pass
        return bool(super().validate(submission))

def container_name_from_image_name(image: str) -> str:
    """
    Creates a valid container name from an image name.
    :param image: Docker image name
    :return: A valid container name
    """
    return '_'.join(['test', *image.split(':')[0].split('/')])

class TestUtils(unittest.TestCase):
    """Tests for util functions"""

    @patch('socket.socket')
    def test_find_port(self, mock_socket: Mock) -> None:
        """Test that method uses socket to find a free port, and returns it."""
        mock_socket().getsockname.return_value = [0, '1234']
        port = find_free_port_number()
        mock_socket().bind.assert_called_once_with(('', 0))
        mock_socket().close.assert_called_once()
        self.assertEqual(1234, port)

class TestSubmissionContainer(TestCase):
    """Tests for SubmissionContainer class"""

    @patch('docker.from_env')
    def setUp(self, mock_from_env: Mock) -> None:
        """Sets variables for testing"""
        self.manager = SubmissionContainer(submission_image='foo/bar', container_name='foo_bar', port=314)

    @patch('docker.from_env', Mock())
    def test_initialization(self) -> None:
        """Tests that the container manager gets initialized correctly."""
        mock_manager = SubmissionContainer(submission_image='foo/bar', container_name='foo_bar', port=314)
        self.assertEqual('foo/bar', mock_manager.submission_image)
        self.assertEqual('foo_bar', mock_manager.container_name)
        self.assertEqual(314, mock_manager.port)

    @patch('docker.from_env')
    @patch.object(SubmissionContainer, 'stop')
    def test_start_submission_container(self, mock_stop_submission_container: Mock, mock_from_env: Mock) -> None:
        """Tests that the container is run with the correct arguments."""
        mock_env = Mock()
        mock_from_env.return_value = mock_env
        mock_env.containers.get.return_value = 'test_container'
        test_container = self.manager.start()
        mock_stop_submission_container.assert_called_once()
        self.manager.client.containers.run.assert_called_with('foo/bar', name='foo_bar', detach=True, ports={'314': 314}, tty=True, environment={'SUBMISSION_CONTAINER_PORT': '314'}, device_requests=[{'Driver': '', 'Count': 0, 'DeviceIDs': ['0'], 'Capabilities': [['gpu']], 'Options': {}}], cpuset_cpus='0,1', volumes={'/data/sets/nuplan': {'bind': '/data/sets/nuplan', 'mode': 'ro'}})
        self.assertEqual('test_container', test_container)

    def test_stop_missing_container(self) -> None:
        """Checks that trying to remove a missing container does not fail (is intended behavior)"""
        mock_container = Mock()
        self.manager.client = Mock()
        self.manager.client.containers.get.side_effect = ContainerNotFound('Container not found')
        self.manager.client.containers.get.return_value = mock_container
        self.manager.stop()
        self.manager.client.containers.get.assert_called_once()
        mock_container.stop.assert_not_called()
        mock_container.remove.assert_not_called()

    def test_stop_existing_container(self) -> None:
        """Checks that if a running container is found, it is stopped and removed."""
        mock_container = Mock()
        self.manager.client = Mock()
        self.manager.client.containers.get.return_value = mock_container
        self.manager.stop()
        self.manager.client.containers.get.assert_called_once()
        mock_container.kill.assert_called_once()
        mock_container.remove.assert_called_once()

