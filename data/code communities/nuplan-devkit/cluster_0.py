# Cluster 0

def save_scenes_to_dir(scenario: AbstractScenario, save_dir: str, simulation_history: SimulationHistory) -> SimulationScenarioKey:
    """
    Save scenes to a directory.
    :param scenario: Scenario.
    :param save_dir: Save path.
    :param simulation_history: Simulation history.
    :return: Scenario key of simulation.
    """
    planner_name = 'tutorial_planner'
    scenario_type = scenario.scenario_type
    scenario_name = scenario.scenario_name
    log_name = scenario.log_name
    save_path = Path(save_dir)
    file = save_path / planner_name / scenario_type / log_name / scenario_name / (scenario_name + '.msgpack.xz')
    file.parent.mkdir(exist_ok=True, parents=True)
    dummy_planner = _create_dummy_simple_planner(acceleration=[5.0, 5.0])
    simulation_log = SimulationLog(planner=dummy_planner, scenario=scenario, simulation_history=simulation_history, file_path=file)
    simulation_log.save_to_file()
    return SimulationScenarioKey(planner_name=planner_name, scenario_name=scenario_name, scenario_type=scenario_type, nuboard_file_index=0, log_name=log_name, files=[file])

def bokeh_app(doc: Document) -> None:
    """
        Run bokeh app in jupyter notebook.
        :param doc: Bokeh document to render.
        """
    nuboard_file = NuBoardFile(simulation_main_path=save_path.name, simulation_folder='', metric_main_path='', metric_folder='', aggregator_metric_folder='')
    experiment_file_data = ExperimentFileData(file_paths=[nuboard_file])
    simulation_tile = SimulationTile(doc=doc, map_factory=map_factory, experiment_file_data=experiment_file_data, vehicle_parameters=get_pacifica_parameters())
    simulation_scenario_data = simulation_tile.render_simulation_tiles(simulation_scenario_keys)
    simulation_figures = [data.plot for data in simulation_scenario_data]
    simulation_layouts = column(simulation_figures)
    doc.add_root(simulation_layouts)
    doc.add_next_tick_callback(complete_message)

class NuBoard:
    """NuBoard application class."""

    def __init__(self, nuboard_paths: List[str], scenario_builder: AbstractScenarioBuilder, vehicle_parameters: VehicleParameters, port_number: int=5006, profiler_path: Optional[Path]=None, resource_prefix: Optional[str]=None, async_scenario_rendering: bool=True, scenario_rendering_frame_rate_cap_hz: int=60):
        """
        Nuboard main class.
        :param nuboard_paths: A list of paths to nuboard files.
        :param scenario_builder: Scenario builder instance.
        :param vehicle_parameters: vehicle parameters.
        :param port_number: Bokeh port number.
        :param profiler_path: Path to save the profiler.
        :param resource_prefix: Prefix to the resource path in HTML.
        :param async_scenario_rendering: Whether to use asynchronous scenario rendering in the scenario tab.
        :param scenario_rendering_frame_rate_cap_hz: Maximum frames to render in the scenario tab per second.
            Use lower values when running nuBoard in the cloud to prevent frame queues due to latency. The rule of thumb
            is to match the frame rate with the expected latency, e.g 5Hz for 200ms round-trip latency.
            Internally this value is capped at 60.
        """
        self._profiler_path = profiler_path
        self._nuboard_paths = check_nuboard_file_paths(nuboard_paths)
        self._scenario_builder = scenario_builder
        self._port_number = port_number
        self._vehicle_parameters = vehicle_parameters
        self._doc: Optional[Document] = None
        self._resource_prefix = resource_prefix if resource_prefix else ''
        self._resource_path = Path(__file__).parents[0] / 'resource'
        self._profiler_file_name = 'nuboard'
        self._profiler: Optional[ProfileCallback] = None
        self._async_scenario_rendering = async_scenario_rendering
        if scenario_rendering_frame_rate_cap_hz < 1 or scenario_rendering_frame_rate_cap_hz > 60:
            raise ValueError('scenario_rendering_frame_rate_cap_hz should be between 1 and 60')
        self._scenario_rendering_frame_rate_cap_hz = scenario_rendering_frame_rate_cap_hz

    def stop_handler(self, sig: Any, frame: Any) -> None:
        """Helper to handle stop signals."""
        logger.info('Stopping the Bokeh application.')
        if self._profiler:
            self._profiler.save_profiler(self._profiler_file_name)
        IOLoop.current().stop()

    def run(self) -> None:
        """Run nuBoard WebApp."""
        logger.info(f'Opening Bokeh application on http://localhost:{self._port_number}/')
        logger.info(f'Async rendering is set to: {self._async_scenario_rendering}')
        io_loop = IOLoop.current()
        if self._profiler_path is not None:
            signal.signal(signal.SIGTERM, self.stop_handler)
            signal.signal(signal.SIGINT, self.stop_handler)
            self._profiler = ProfileCallback(output_dir=self._profiler_path)
            self._profiler.start_profiler(self._profiler_file_name)
        bokeh_app = Application(FunctionHandler(self.main_page))
        server = Server({'/': bokeh_app}, io_loop=io_loop, port=self._port_number, allow_websocket_origin=['*'], extra_patterns=[('/resource/(.*)', StaticFileHandler, {'path': str(self._resource_path)})])
        server.start()
        io_loop.add_callback(server.show, '/')
        try:
            io_loop.start()
        except RuntimeError as e:
            logger.warning(f'{e}')

    def main_page(self, doc: Document) -> None:
        """
        Main nuBoard page.
        :param doc: HTML document.
        """
        self._doc = doc
        template_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'templates'
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
        self._doc.template = env.get_template('index.html')
        self._doc.title = 'nuBoard'
        nuboard_files = read_nuboard_file_paths(file_paths=self._nuboard_paths)
        experiment_file_data = ExperimentFileData(file_paths=nuboard_files)
        overview_tab = OverviewTab(doc=self._doc, experiment_file_data=experiment_file_data)
        histogram_tab = HistogramTab(doc=self._doc, experiment_file_data=experiment_file_data)
        scenario_tab = ScenarioTab(experiment_file_data=experiment_file_data, scenario_builder=self._scenario_builder, doc=self._doc, vehicle_parameters=self._vehicle_parameters, async_rendering=self._async_scenario_rendering, frame_rate_cap_hz=self._scenario_rendering_frame_rate_cap_hz)
        configuration_tab = ConfigurationTab(experiment_file_data=experiment_file_data, doc=self._doc, tabs=[overview_tab, histogram_tab, scenario_tab])
        s3_tab = CloudTab(doc=self._doc, configuration_tab=configuration_tab)
        self._doc.add_root(configuration_tab.file_path_input)
        self._doc.add_root(configuration_tab.experiment_file_path_checkbox_group)
        self._doc.add_root(s3_tab.s3_bucket_name)
        self._doc.add_root(s3_tab.s3_bucket_text_input)
        self._doc.add_root(s3_tab.s3_error_text)
        self._doc.add_root(s3_tab.s3_access_key_id_text_input)
        self._doc.add_root(s3_tab.s3_secret_access_key_password_input)
        self._doc.add_root(s3_tab.s3_bucket_prefix_text_input)
        self._doc.add_root(s3_tab.s3_modal_query_btn)
        self._doc.add_root(s3_tab.s3_download_text_input)
        self._doc.add_root(s3_tab.s3_download_button)
        self._doc.add_root(s3_tab.data_table)
        self._doc.add_root(overview_tab.table)
        self._doc.add_root(overview_tab.planner_checkbox_group)
        self._doc.add_root(histogram_tab.scenario_type_multi_choice)
        self._doc.add_root(histogram_tab.metric_name_multi_choice)
        self._doc.add_root(histogram_tab.planner_checkbox_group)
        self._doc.add_root(histogram_tab.histogram_plots)
        self._doc.add_root(histogram_tab.bin_spinner)
        self._doc.add_root(histogram_tab.histogram_modal_query_btn)
        self._doc.add_root(scenario_tab.planner_checkbox_group)
        self._doc.add_root(scenario_tab.scenario_title_div)
        self._doc.add_root(scenario_tab.object_checkbox_group)
        self._doc.add_root(scenario_tab.traj_checkbox_group)
        self._doc.add_root(scenario_tab.map_checkbox_group)
        self._doc.add_root(scenario_tab.scalar_scenario_type_select)
        self._doc.add_root(scenario_tab.scalar_log_name_select)
        self._doc.add_root(scenario_tab.scalar_scenario_name_select)
        self._doc.add_root(scenario_tab.scenario_token_multi_choice)
        self._doc.add_root(scenario_tab.scenario_modal_query_btn)
        self._doc.add_root(scenario_tab.time_series_layout)
        self._doc.add_root(scenario_tab.ego_expert_states_layout)
        self._doc.add_root(scenario_tab.scenario_score_layout)
        self._doc.add_root(scenario_tab.simulation_tile_layout)

def check_nuboard_file_paths(main_paths: List[str]) -> List[Path]:
    """
    Check if given file paths are valid nuBoard files.
    :param main_paths: A list of file paths.
    :return A list of available nuBoard files.
    """
    available_paths = []
    for main_path in main_paths:
        main_folder_path: Path = Path(main_path)
        if main_folder_path.is_dir():
            files = list(main_folder_path.iterdir())
            event_files = [file for file in files if file.name.endswith(NuBoardFile.extension())]
            if len(event_files) > 0:
                event_files = sorted(event_files, reverse=True)
                available_paths.append(event_files[0])
        elif main_folder_path.is_file() and main_folder_path.name.endswith(NuBoardFile.extension()):
            available_paths.append(main_folder_path)
        else:
            raise RuntimeError(f'{str(main_folder_path)} is not a valid nuBoard file')
        if len(available_paths) == 0:
            logger.info('No available nuBoard files are found.')
    return available_paths

def read_nuboard_file_paths(file_paths: List[Path]) -> List[NuBoardFile]:
    """
    Read a list of file paths to NuBoardFile data class.
    :param file_paths: A list of file paths.
    :return A list of NuBoard files.
    """
    nuboard_files = []
    for file_path in file_paths:
        nuboard_file = NuBoardFile.load_nuboard_file(file_path)
        nuboard_file.current_path = file_path.parents[0]
        nuboard_files.append(nuboard_file)
    return nuboard_files

def check_s3_nuboard_files(s3_file_contents: Dict[str, S3FileContent], s3_path: str, s3_client: boto3.client) -> S3NuBoardFileResultMessage:
    """
    Return True in the message if there is a nuboard file and can load into nuBoard.
    :param s3_file_contents: S3 prefix with a dictionary of s3 file name and their contents.
    :Param s3_path: S3 Path starts with s3://.
    :param s3_client: s3 client session.
    :return S3NuBoardFileResultMessage to indicate if there is available nuboard file in the s3 prefix.
    """
    success = False
    return_message = 'No available nuboard files in the prefix'
    nuboard_file = None
    nuboard_filename = None
    if not s3_path.endswith('/'):
        s3_path = s3_path + '/'
    url = parse.urlparse(s3_path)
    for file_name, file_content in s3_file_contents.items():
        if file_name.endswith(NuBoardFile.extension()):
            try:
                nuboard_object = s3_client.get_object(Bucket=url.netloc, Key=file_name)
                file_stream = io.BytesIO(nuboard_object['Body'].read())
                nuboard_data = pickle.load(file_stream)
                nuboard_file = NuBoardFile.deserialize(nuboard_data)
                file_stream.close()
                nuboard_filename = Path(file_name).name
                return_message = f'Found available nuboard file: {nuboard_filename}'
                success = True
                break
            except Exception as e:
                logger.info(str(e))
                continue
    return S3NuBoardFileResultMessage(s3_connection_status=S3ConnectionStatus(success=success, return_message=return_message), nuboard_filename=nuboard_filename, nuboard_file=nuboard_file)

def download_s3_path(s3_path: str, s3_client: boto3.client, save_path: str, delimiter: str='/') -> S3ConnectionStatus:
    """
    Download a s3 path recursively given a s3 full path.
    :param s3_path: S3 full path.
    :param s3_client: A connecting S3 client.
    :param save_path: Local save path.
    :param delimiter: Delimiter to split folders.
    :return S3 connection status to indicate status of s3 connection.
    """
    return_message = f'Downloaded {s3_path}'
    try:
        if not s3_path.endswith('/'):
            s3_path = s3_path + '/'
        url = parse.urlparse(s3_path)
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=url.netloc, Prefix=url.path.lstrip('/'), Delimiter=delimiter)
        for page in page_iterator:
            common_prefixes = page.get('CommonPrefixes', [])
            for sub_folder in common_prefixes:
                sub_s3_path = os.path.join('s3://', url.netloc, sub_folder['Prefix'])
                local_save_sub_path = Path(save_path, sub_folder['Prefix'])
                local_save_sub_path.mkdir(parents=True, exist_ok=True)
                download_s3_path(s3_client=s3_client, s3_path=sub_s3_path, save_path=save_path)
            contents = page.get('Contents', [])
            for content in contents:
                file_name = str(content['Key'])
                file_size = content['Size']
                last_modified = content['LastModified']
                s3_file_path = os.path.join('s3://', url.netloc, file_name)
                local_folder = Path(save_path, file_name)
                local_folder.parents[0].mkdir(exist_ok=True, parents=True)
                file_content = S3FileContent(filename=file_name, size=file_size, last_modified=last_modified)
                download_s3_file(s3_path=s3_file_path, file_content=file_content, s3_client=s3_client, save_path=save_path)
        success = True
    except Exception as e:
        raise Boto3Error(e)
    s3_connection_status = S3ConnectionStatus(success=success, return_message=return_message)
    return s3_connection_status

def download_s3_file(s3_path: str, file_content: S3FileContent, s3_client: boto3.client, save_path: str) -> S3ConnectionStatus:
    """
    Download a s3 file given a s3 full path.
    :param s3_path: S3 full path.
    :param file_content: File content info.
    :param s3_client: A connecting S3 client.
    :param save_path: Local save path.
    :return S3 connection status to indicate status of s3 connection.
    """
    return_message = f'Downloaded {s3_path}'
    try:
        if s3_path.endswith('/'):
            return S3ConnectionStatus(success=False, return_message=f'{s3_path} is not a file')
        url = parse.urlparse(s3_path)
        file_name = file_content.filename if file_content.filename is not None else ''
        download_file_name = Path(save_path, file_name)
        remote_file_size = file_content.size if file_content.size is not None else 0
        local_file_size = os.path.getsize(str(download_file_name)) if download_file_name.exists() else 0
        if not download_file_name.exists() or local_file_size != float(remote_file_size):
            s3_client.download_file(url.netloc, file_name, str(download_file_name))
        success = True
    except Exception as e:
        raise Boto3Error(e)
    return S3ConnectionStatus(success=success, return_message=return_message)

class TestNuBoardUtils(unittest.TestCase):
    """Unit tests for utils in nuboard."""

    def setUp(self) -> None:
        """Set up a list of nuboard files."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_paths: List[str] = []
        self.nuboard_files: List[NuBoardFile] = []
        for i in range(2):
            main_path = os.path.join(self.tmp_dir.name, str(i))
            nuboard_file = NuBoardFile(simulation_main_path=main_path, metric_folder='metrics', simulation_folder='simulations', metric_main_path=main_path, aggregator_metric_folder='aggregator_metric')
            nuboard_file_name = os.path.join(main_path, 'nuboard_file' + NuBoardFile.extension())
            self.nuboard_files.append(nuboard_file)
            self.nuboard_paths.append(nuboard_file_name)

    def test_check_nuboard_file_paths(self) -> None:
        """Test if check_nuboard_file_paths works."""
        self.assertRaises(RuntimeError, check_nuboard_file_paths, self.nuboard_paths)
        for nuboard_file, nuboard_path in zip(self.nuboard_files, self.nuboard_paths):
            main_path = Path(nuboard_file.simulation_main_path)
            main_path.mkdir(parents=True, exist_ok=True)
            file = Path(nuboard_path)
            nuboard_file.save_nuboard_file(file)
        nuboard_paths = check_nuboard_file_paths(self.nuboard_paths)
        self.assertEqual(len(nuboard_paths), 2)
        self.assertIsInstance(nuboard_paths, list)
        for nuboard_path_name in nuboard_paths:
            self.assertIsInstance(nuboard_path_name, Path)
        nuboard_path_head = [os.path.dirname(nuboard_path) for nuboard_path in self.nuboard_paths]
        nuboard_paths = check_nuboard_file_paths(nuboard_path_head)
        self.assertEqual(len(nuboard_paths), 2)
        self.assertIsInstance(nuboard_paths, list)
        for nuboard_path_name in nuboard_paths:
            self.assertIsInstance(nuboard_path_name, Path)

    def test_read_nuboard_file_paths(self) -> None:
        """Test if read_nuboard_file_paths works."""
        nuboard_paths: List[Path] = []
        for nuboard_file, nuboard_path in zip(self.nuboard_files, self.nuboard_paths):
            main_path = Path(nuboard_file.simulation_main_path)
            main_path.mkdir(parents=True, exist_ok=True)
            file = Path(nuboard_path)
            nuboard_file.save_nuboard_file(file)
            nuboard_paths.append(file)
        nuboard_files = read_nuboard_file_paths(file_paths=nuboard_paths)
        self.assertEqual(len(nuboard_files), 2)
        for nuboard_file in nuboard_files:
            self.assertIsInstance(nuboard_file, NuBoardFile)

    def tearDown(self) -> None:
        """Remove and clean up the tmp folder."""
        self.tmp_dir.cleanup()

class TestNuBoardCloudUtil(unittest.TestCase):
    """Unit tests for cloud utils in nuboard."""

    def setUp(self) -> None:
        """Set up a list of nuboard files."""
        self.tmp_dir = tempfile.TemporaryDirectory()

    def test_check_s3_nuboard_files_fail(self) -> None:
        """Test if check_s3_nuboard_files fails when there is no nuboard file."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        dummy_file_result_message = {'dummy_a': S3FileContent(filename='dummy_a', size=10, last_modified=datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)), 'dummy_b': S3FileContent(filename='dummy_b', size=10, last_modified=datetime(day=3, month=8, year=1992, tzinfo=timezone.utc))}
        encoded_expected_messages = {'dummy_a': json.dumps(dummy_file_result_message['dummy_a'].serialize()).encode(), 'dummy_b': json.dumps(dummy_file_result_message['dummy_b'].serialize()).encode()}
        dummy_streaming_io_message_response = {}
        expected_params = {}
        for s3_key, result_message in dummy_file_result_message.items():
            dummy_streaming_io_message_response[s3_key] = {'Body': io.BytesIO(encoded_expected_messages[s3_key])}
            expected_params[s3_key] = {'Bucket': 'test-bucket', 'Key': s3_key}
        for s3_key, expected_param in expected_params.items():
            response = dummy_streaming_io_message_response[s3_key]
            stubber.add_response('get_object', response, expected_param)
        with stubber:
            s3_nuboard_file_result_message = check_s3_nuboard_files(s3_file_contents=dummy_file_result_message, s3_client=s3_client, s3_path='s3://test-bucket')
        self.assertIsNone(s3_nuboard_file_result_message.nuboard_file)
        self.assertFalse(s3_nuboard_file_result_message.s3_connection_status.success)

    def test_check_s3_nuboard_files_success(self) -> None:
        """Test if check_s3_nuboard_files success when there is a nuboard file."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', metric_main_path=self.tmp_dir.name, aggregator_metric_folder='aggregator_metric')
        nuboard_file_name = 'dummy_a' + NuBoardFile.extension()
        dummy_file_result_message = {nuboard_file_name: S3FileContent(filename=nuboard_file_name, size=12, last_modified=datetime(day=4, month=5, year=1992, tzinfo=timezone.utc))}
        encoded_expected_messages = {nuboard_file_name: pickle.dumps(nuboard_file.serialize())}
        dummy_streaming_io_message_response = {}
        expected_params = {}
        for s3_key, result_message in dummy_file_result_message.items():
            dummy_streaming_io_message_response[s3_key] = {'Body': io.BytesIO(encoded_expected_messages[s3_key])}
            expected_params[s3_key] = {'Bucket': 'test-bucket', 'Key': s3_key}
        for s3_key, expected_param in expected_params.items():
            response = dummy_streaming_io_message_response[s3_key]
            stubber.add_response('get_object', response, expected_param)
        with stubber:
            s3_nuboard_file_result_message = check_s3_nuboard_files(s3_file_contents=dummy_file_result_message, s3_client=s3_client, s3_path='s3://test-bucket')
        self.assertTrue(s3_nuboard_file_result_message.s3_connection_status.success)
        self.assertIsNotNone(s3_nuboard_file_result_message.nuboard_file)
        self.assertEqual(nuboard_file.simulation_main_path, s3_nuboard_file_result_message.nuboard_file.simulation_main_path)
        self.assertEqual(nuboard_file.metric_main_path, s3_nuboard_file_result_message.nuboard_file.metric_main_path)

    def test_get_s3_file_content(self) -> None:
        """Test if download_s3_file works."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        expected_response = {'CommonPrefixes': [{'Prefix': 'dummy_folder_a/log.txt'}, {'Prefix': 'dummy_folder_b/log_2.txt'}], 'Contents': [{'Key': 'dummy_a', 'Size': 15, 'LastModified': datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)}, {'Key': 'dummy_b', 'Size': 45, 'LastModified': datetime(day=6, month=7, year=1992, tzinfo=timezone.utc)}]}
        expected_params = {'Bucket': 'test-bucket', 'Prefix': '', 'Delimiter': '/'}
        s3_path = 's3://test-bucket'
        stubber.add_response('list_objects_v2', expected_response, expected_params)
        with stubber:
            s3_file_contents = get_s3_file_contents(s3_path=s3_path, client=s3_client, include_previous_folder=True)
            self.assertTrue(s3_file_contents.s3_connection_status.success)
            expected_file_names = ['dummy_folder_a/log.txt', 'dummy_folder_b/log_2.txt', 'dummy_a', 'dummy_b']
            for index, (file_name, _) in enumerate(s3_file_contents.file_contents.items()):
                self.assertEqual(file_name, expected_file_names[index])

    def test_s3_download_file(self) -> None:
        """Test s3_download_file in utils."""
        s3_client = boto3.Session().client('s3')
        dummy_s3_file_content = S3FileContent(filename='dummy_a', size=10, last_modified=datetime(day=2, month=7, year=1992, tzinfo=timezone.utc))
        s3_path = 's3://test-bucket'
        save_path = self.tmp_dir.name
        with self.assertRaises(Boto3Error):
            download_s3_file(s3_path=s3_path, s3_client=s3_client, save_path=save_path, file_content=dummy_s3_file_content)

    def test_s3_download_path(self) -> None:
        """Test s3_download_path in utils."""
        s3_client = boto3.Session().client('s3')
        stubber = Stubber(s3_client)
        expected_response = {'CommonPrefixes': [{'Prefix': 'dummy_folder_a/log.txt'}, {'Prefix': 'dummy_folder_b/log_2.txt'}], 'Contents': [{'Key': 'dummy_a', 'Size': 15, 'LastModified': datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)}, {'Key': 'dummy_b', 'Size': 45, 'LastModified': datetime(day=6, month=7, year=1992, tzinfo=timezone.utc)}]}
        expected_params = {'Bucket': 'test-bucket', 'Prefix': '', 'Delimiter': '/'}
        s3_path = 's3://test-bucket'
        stubber.add_response('list_objects_v2', expected_response, expected_params)
        save_path = self.tmp_dir.name
        with stubber:
            with self.assertRaises(Boto3Error):
                download_s3_path(s3_path=s3_path, s3_client=s3_client, save_path=save_path)

    def tearDown(self) -> None:
        """Remove and clean up the tmp folder."""
        self.tmp_dir.cleanup()

@dataclass
class ExperimentFileData:
    """Data for experiment files."""
    file_paths: List[NuBoardFile]
    color_palettes: List[str] = field(default_factory=list)
    expert_color_palettes: List[str] = field(default_factory=list)
    available_metric_statistics_names: List[str] = field(default_factory=list)
    metric_statistics_dataframes: List[List[MetricStatisticsDataFrame]] = field(default_factory=list)
    metric_aggregator_dataframes: List[Dict[str, pd.DataFrame]] = field(default_factory=list)
    simulation_files: Dict[str, Any] = field(default_factory=dict)
    simulation_scenario_keys: List[SimulationScenarioKey] = field(default_factory=list)
    available_scenario_types: List[str] = field(default_factory=list)
    available_scenarios: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    available_scenario_tokens: Dict[str, ScenarioTokenInfo] = field(default_factory=dict)
    file_path_colors: Dict[int, Dict[str, str]] = field(default_factory=dict)
    color_index: int = 0

    def __post_init__(self) -> None:
        """Post initialization."""
        if not self.simulation_files:
            self.simulation_files = defaultdict(set)
        if not self.available_scenario_tokens:
            self.available_scenario_tokens = defaultdict()
        if not self.color_palettes:
            self.color_palettes = Set1[9] + Set2[8] + Set3[12]
        if not self.expert_color_palettes:
            self.expert_color_palettes = Pastel2[8] + Pastel1[9] + Dark2[8]
        if not self.available_scenarios:
            self.available_scenarios = defaultdict(lambda: defaultdict(list))
        if self.file_paths:
            file_paths = self.file_paths
            self.file_paths = []
            self.update_data(file_paths=file_paths)

    def update_data(self, file_paths: List[NuBoardFile]) -> None:
        """
        Update experiment data with a new list of nuboard file paths.
        :param file_paths: A list of new nuboard file paths.
        """
        starting_file_path_index = len(self.file_paths)
        self._update_file_path_color(file_paths=file_paths, starting_file_path_index=starting_file_path_index)
        self._add_metric_files(file_paths=file_paths, starting_file_path_index=starting_file_path_index)
        self._add_metric_aggregator_files(file_paths=file_paths, starting_file_path_index=starting_file_path_index)
        self._add_simulation_files(file_paths=file_paths, starting_file_path_index=starting_file_path_index)
        self.file_paths += file_paths

    @staticmethod
    def _get_base_path(current_path: Path, base_path: Path, sub_folder: str) -> Path:
        """
        Get valid base path.
        :param current_path: Current nuboard file path.
        :Param base_path: Alternative base path.
        :param sub_folder: Sub folder.
        :return A base path.
        """
        default_path = base_path / sub_folder
        if current_path is None:
            return default_path
        base_folder = current_path / sub_folder
        if not base_folder.exists():
            base_folder = default_path
        return base_folder

    def _update_file_path_color(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Update file path colors.
        :param file_paths: A list of new nuboard file paths.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            self.file_path_colors[file_path_index] = defaultdict(str)
            metric_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.metric_main_path), sub_folder=file_path.metric_folder)
            planner_names: List[str] = []
            if not metric_path.exists():
                continue
            for file in metric_path.iterdir():
                try:
                    data_frame = MetricStatisticsDataFrame.load_parquet(file)
                    planner_names += data_frame.planner_names
                except (FileNotFoundError, Exception) as e:
                    logger.info(e)
                    pass
            if not planner_names:
                simulation_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.simulation_main_path), sub_folder=file_path.simulation_folder)
                if not simulation_path.exists():
                    continue
                planner_name_paths = simulation_path.iterdir()
                for planner_name_path in planner_name_paths:
                    planner_name = planner_name_path.name
                    planner_names.append(planner_name)
            planner_names = list(set(planner_names))
            for planner_name in planner_names:
                self.file_path_colors[file_path_index][planner_name] = self.color_palettes[self.color_index]
                self.color_index += 1

    def _add_metric_files(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Add and load metric files.
        Folder hierarchy: planner_name -> scenario_type -> metric result name -> scenario_name.pkl
        :param file_paths: A list of new nuboard files.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            self.metric_statistics_dataframes.append([])
            metric_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.metric_main_path), sub_folder=file_path.metric_folder)
            if not metric_path.exists():
                continue
            for file in metric_path.iterdir():
                if file.is_dir():
                    continue
                try:
                    data_frame = MetricStatisticsDataFrame.load_parquet(file)
                    self.metric_statistics_dataframes[file_path_index].append(data_frame)
                    self.available_metric_statistics_names.append(data_frame.metric_statistic_name)
                except (FileNotFoundError, Exception):
                    pass
        self.available_metric_statistics_names = sorted(list(set(self.available_metric_statistics_names)), reverse=False)

    def _add_metric_aggregator_files(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Load metric aggregator files.
        :param file_paths: A list of new nuboard files.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            self.metric_aggregator_dataframes.append({})
            metric_aggregator_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.metric_main_path), sub_folder=file_path.aggregator_metric_folder)
            if not metric_aggregator_path.exists():
                continue
            for file in metric_aggregator_path.iterdir():
                if file.is_dir():
                    continue
                try:
                    data_frame = pd.read_parquet(file)
                    self.metric_aggregator_dataframes[file_path_index][file.stem] = data_frame
                except (FileNotFoundError, Exception):
                    pass

    def _add_simulation_files(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Load simulation files.
        Folder hierarchy: planner_name -> scenario_type -> scenario_names -> iteration.pkl.
        :param file_paths: A list of new nuboard files.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            if file_path.simulation_folder is None:
                continue
            file_path_index = starting_file_path_index + index
            simulation_path = self._get_base_path(current_path=file_path.current_path, base_path=Path(file_path.simulation_main_path), sub_folder=file_path.simulation_folder)
            if not simulation_path.exists():
                continue
            planner_name_paths = simulation_path.iterdir()
            for planner_name_path in planner_name_paths:
                planner_name = planner_name_path.name
                scenario_type_paths = planner_name_path.iterdir()
                for scenario_type_path in scenario_type_paths:
                    log_name_paths = scenario_type_path.iterdir()
                    scenario_type = scenario_type_path.name
                    for log_name_path in log_name_paths:
                        scenario_name_paths = log_name_path.iterdir()
                        log_name = log_name_path.name
                        for scenario_name_path in scenario_name_paths:
                            scenario_name = scenario_name_path.name
                            scenario_key = f'{simulation_path.parents[0].name}/{planner_name}/{scenario_type}/{log_name}/{scenario_name}'
                            if scenario_key in self.simulation_files:
                                continue
                            files = scenario_name_path.iterdir()
                            for file in files:
                                self.simulation_files[scenario_key].add(file)
                            self.available_scenarios[scenario_type][log_name].append(scenario_name)
                            self.available_scenario_tokens[scenario_name] = ScenarioTokenInfo(scenario_name=scenario_name, scenario_token=scenario_name, scenario_type=scenario_type, log_name=log_name)
                            self.simulation_scenario_keys.append(SimulationScenarioKey(nuboard_file_index=file_path_index, log_name=log_name, planner_name=planner_name, scenario_type=scenario_type, scenario_name=scenario_name, files=list(self.simulation_files[scenario_key])))
        available_scenario_types = list(set(self.available_scenarios.keys()))
        self.available_scenario_types = sorted(available_scenario_types, reverse=False)

class TestSimulationTile(unittest.TestCase):
    """Test simulation_tile functionality."""

    def set_up_simulation_log(self, output_path: Path) -> None:
        """
        Create a simulation log and save it to disk.
        :param output path: to write the simulation log to.
        """
        simulation_log = create_sample_simulation_log(output_path)
        simulation_log.save_to_file()

    def setUp(self) -> None:
        """Set up simulation tile with nuboard file."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.vehicle_parameters = get_pacifica_parameters()
        simulation_log_path = Path(self.tmp_dir.name) / 'test_simulation_tile_simulation_log.msgpack.xz'
        self.set_up_simulation_log(simulation_log_path)
        nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulation', aggregator_metric_folder='aggregator_metric', current_path=Path(self.tmp_dir.name))
        self.scenario_keys = [SimulationScenarioKey(nuboard_file_index=0, log_name='dummy_log', planner_name='SimplePlanner', scenario_type='common', scenario_name='test', files=[simulation_log_path])]
        self.doc = Document()
        self.map_factory = MockMapFactory()
        self.experiment_file_data = ExperimentFileData(file_paths=[nuboard_file])
        self.simulation_tile = SimulationTile(doc=self.doc, map_factory=self.map_factory, vehicle_parameters=self.vehicle_parameters, radius=80, experiment_file_data=self.experiment_file_data)

    @given(frame_rate_cap=st.integers(min_value=1, max_value=60))
    def test_valid_frame_rate_cap_range(self, frame_rate_cap: int) -> None:
        """Tests valid frame rate cap range."""
        SimulationTile(doc=self.doc, map_factory=self.map_factory, vehicle_parameters=self.vehicle_parameters, radius=80, experiment_file_data=self.experiment_file_data, frame_rate_cap_hz=frame_rate_cap)

    @given(frame_rate_cap=st.integers().filter(lambda x: x < 1 or x > 60))
    def test_invalid_frame_rate_cap_range(self, frame_rate_cap: int) -> None:
        """Tests invalid frame rate cap range."""
        with self.assertRaises(ValueError):
            SimulationTile(doc=self.doc, map_factory=self.map_factory, vehicle_parameters=self.vehicle_parameters, radius=80, experiment_file_data=self.experiment_file_data, frame_rate_cap_hz=frame_rate_cap)

    def test_simulation_tile_layout(self) -> None:
        """Test layout design."""
        layout = self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
        self.assertEqual(len(layout), 1)

    def test_periodic_callback(self) -> None:
        """Tests that _periodic_callback is registered correctly to the bokeh Document."""
        with patch.object(SimulationTile, '_periodic_callback', autospec=True) as mock_periodic_callback:
            SimulationTile(doc=self.doc, map_factory=self.map_factory, vehicle_parameters=self.vehicle_parameters, radius=80, experiment_file_data=self.experiment_file_data)
            for cb in self.doc.callbacks.session_callbacks:
                cb.callback()
            self.assertEqual(mock_periodic_callback.call_count, 1)

    def _trigger_button_click_event(self, figure_index: int, button_name: str) -> None:
        """
        Trigger a bokeh.model.Button click event.
        :param figure_index: The index of the SimulationTile figure.
        :param button_name: The name of SimulationTile button.
        """
        button = getattr(self.simulation_tile.figures[figure_index], button_name)
        button._trigger_event(ButtonClick(button))

    def _test_frame_index_request_button(self, button_name: str, frame_index_request: FrameIndexRequest) -> None:
        """
        Helper function to test that frame index request buttons (first, prev, next, last) work correctly.
        :param click_callback_name: Button click callback function name in SimulationTile that's registered to bokeh.
        :param button_name: The name of the button in the SimulationTile class.
        :param frame_index_request: FrameIndexRequest object representing the frame index requested.
        """
        with patch.object(self.simulation_tile, '_render_plots'):
            self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
            figure_index = 0
            figure = self.simulation_tile.figures[figure_index]
            if frame_index_request == FrameIndexRequest.FIRST or frame_index_request == FrameIndexRequest.LAST:
                self._trigger_button_click_event(figure_index, button_name)
                frame_index = len(figure.simulation_history) - 1 if frame_index_request == FrameIndexRequest.LAST else 0
                self.assertEqual(figure.slider.value, frame_index)
            elif frame_index_request == FrameIndexRequest.NEXT:
                self.simulation_tile._current_frame_index = 0
                self._trigger_button_click_event(figure_index, button_name)
                self.assertEqual(figure.slider.value, self.simulation_tile._current_frame_index + 1)
                self.simulation_tile._current_frame_index = len(figure.simulation_history.data) - 1
                self._trigger_button_click_event(figure_index, button_name)
                self.assertEqual(figure.slider.value, self.simulation_tile._current_frame_index)
            elif frame_index_request == FrameIndexRequest.PREV:
                self.simulation_tile._current_frame_index = len(figure.simulation_history.data) - 1
                self._trigger_button_click_event(figure_index, button_name)
                self.assertEqual(figure.slider.value, self.simulation_tile._current_frame_index - 1)
                self.simulation_tile._current_frame_index = 0
                self._trigger_button_click_event(figure_index, button_name)
                self.assertEqual(figure.slider.value, self.simulation_tile._current_frame_index)

    def test_first_frame_button(self) -> None:
        """Tests that go to first frame button works correctly."""
        self._test_frame_index_request_button(button_name='first_button', frame_index_request=FrameIndexRequest.FIRST)

    def test_last_frame_button(self) -> None:
        """Tests that go to last frame button works correctly."""
        self._test_frame_index_request_button(button_name='last_button', frame_index_request=FrameIndexRequest.LAST)

    def _test_symbolic_frame_request_callback_called(self, button_name: str, frame_request_callback_name: str) -> None:
        """
        Helper function to test that the provided symbolic frame request (previous, next, play/stop) callback is called when a button is clicked
        :param button_name: The name of the button in the SimulationTile class.
        :param frame_request_callback_name: Frame request callback function name in SimulationTile that's supposed to be called.
        """
        with patch.object(self.simulation_tile, frame_request_callback_name, autospec=True) as mock_request_frame:
            self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
            figure_index = 0
            button = getattr(self.simulation_tile.figures[figure_index], button_name)
            button._trigger_event(ButtonClick(button))
            mock_request_frame.assert_called_once_with(self.simulation_tile.figures[figure_index])

    def test_prev_button(self) -> None:
        """Tests that show prev frame button works correctly."""
        self._test_frame_index_request_button(button_name='prev_button', frame_index_request=FrameIndexRequest.PREV)

    def test_next_button(self) -> None:
        """Tests that show next frame button works correctly."""
        self._test_frame_index_request_button(button_name='next_button', frame_index_request=FrameIndexRequest.NEXT)

    def test_play_button(self) -> None:
        """Tests that the play button works correctly."""
        self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
        figure_index = 0
        button_name = 'play_button'
        self.assertFalse(self.simulation_tile.is_in_playback)
        self._trigger_button_click_event(figure_index, button_name)
        self.assertTrue(self.simulation_tile.is_in_playback)
        self._trigger_button_click_event(figure_index, button_name)
        self.assertFalse(self.simulation_tile.is_in_playback)

    def test_playback_callback(self) -> None:
        """Tests that the playback callback is registered correctly to the bokeh Document & behaves correctly."""
        self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
        figure_index = 0
        figure = self.simulation_tile.figures[figure_index]
        button_name = 'play_button'
        previous_request_index = figure.slider.value
        self._trigger_button_click_event(figure_index, button_name)
        for cb in self.doc.callbacks.session_callbacks:
            cb.callback()
        self.assertTrue(self.simulation_tile.is_in_playback)
        self.assertTrue(figure.slider.value, previous_request_index + 1)
        self.simulation_tile._current_frame_index = len(figure.simulation_history) - 1
        for cb in self.doc.callbacks.session_callbacks:
            cb.callback()
        self.assertFalse(self.simulation_tile.is_in_playback)

    def test_deferred_plot_rendering(self) -> None:
        """Tests that plot rendering request will be deferred if successive requests are triggered faster than the frame rate cap configured."""
        self.assertIsNone(self.simulation_tile._plot_render_queue)
        with patch.object(self.simulation_tile, '_last_frame_time', new=time.time()):
            self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550])
            figure_index = 0
            figure = self.simulation_tile.figures[figure_index]
            trigger_count = 2
            for _ in range(trigger_count):
                figure.slider.trigger(attr='value', old=0, new=1)
            self.assertIsNotNone(self.simulation_tile._plot_render_queue)

    def tearDown(self) -> None:
        """Clean up temporary folder and files."""
        self.tmp_dir.cleanup()

class TestBaseTab(unittest.TestCase):
    """Test base_tab functionality."""

    def set_up_dummy_simulation(self, simulation_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy simulation data.
        :param simulation_path: Simulation path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        save_path = simulation_path / planner_name / scenario_type / log_name / scenario_name
        save_path.mkdir(parents=True, exist_ok=True)
        simulation_data = create_sample_simulation_log(save_path / 'test_base_tab_simulation_log.msgpack.xz')
        simulation_data.save_to_file()

    def set_up_dummy_metric(self, metric_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy metric results.
        :param metric_path: Metric path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        statistics = [Statistic(name='ego_max_acceleration', unit='meters_per_second_squared', value=2.0, type=MetricStatisticsType.MAX), Statistic(name='ego_min_acceleration', unit='meters_per_second_squared', value=0.0, type=MetricStatisticsType.MIN), Statistic(name='ego_p90_acceleration', unit='meters_per_second_squared', value=1.0, type=MetricStatisticsType.P90)]
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit='meters_per_second_squared', time_stamps=list(time_stamps), values=list(accel))
        result = MetricStatistics(metric_computator='ego_acceleration', name='ego_acceleration_statistics', statistics=statistics, time_series=time_series, metric_category='Dynamic', metric_score=1)
        key = MetricFileKey(metric_name='ego_acceleration', scenario_name=scenario_name, log_name=log_name, scenario_type=scenario_type, planner_name=planner_name)
        metric_engine = MetricsEngine(main_save_path=metric_path)
        metric_files = {'ego_acceleration': [MetricFile(key=key, metric_statistics=[result])]}
        metric_engine.write_to_files(metric_files=metric_files)
        metric_file_callback = MetricFileCallback(metric_file_output_path=str(metric_path), scenario_metric_paths=[str(metric_path)])
        metric_file_callback.on_run_simulation_end()

    def setUp(self) -> None:
        """Set up a nuboard base tab."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', aggregator_metric_folder='aggregator_metric', current_path=Path(self.tmp_dir.name))
        doc = Document()
        log_name = 'dummy_log'
        planner_name = 'SimplePlanner'
        scenario_type = 'Test'
        scenario_name = 'Dummy_scene'
        metric_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_metric(metric_path=metric_path, log_name=log_name, planner_name=planner_name, scenario_name=scenario_name, scenario_type=scenario_type)
        simulation_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_simulation(simulation_path, log_name=log_name, planner_name=planner_name, scenario_type=scenario_type, scenario_name=scenario_name)
        color_palettes = Category20[20] + Set3[12] + Bokeh[8]
        experiment_file_data = ExperimentFileData(file_paths=[], color_palettes=color_palettes)
        self.base_tab = BaseTab(doc=doc, experiment_file_data=experiment_file_data)

    def test_update_experiment_file_data(self) -> None:
        """Test update experiment file data."""
        self.base_tab.experiment_file_data.update_data(file_paths=[self.nuboard_file])
        self.assertEqual(len(self.base_tab.experiment_file_data.available_metric_statistics_names), 1)
        self.assertEqual(len(self.base_tab.experiment_file_data.simulation_scenario_keys), 1)

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change feature."""
        self.base_tab.experiment_file_data.update_data(file_paths=[self.nuboard_file])
        self.assertRaises(NotImplementedError, self.base_tab.file_paths_on_change, self.base_tab.experiment_file_data, [0])

    def tearDown(self) -> None:
        """Remove all temporary folders and files."""
        self.tmp_dir.cleanup()

class TestNuBoardFile(unittest.TestCase):
    """Test NuBoardFile functionality."""

    def setUp(self) -> None:
        """Set up a nuBoard file class."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', aggregator_metric_folder='aggregator_metric')
        self.nuboard_file_name = Path(self.tmp_dir.name) / ('nuboard_file' + self.nuboard_file.extension())

    def test_nuboard_save_and_load_file(self) -> None:
        """Test saving and loading a nuboard file."""
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.assertTrue(os.path.exists(self.nuboard_file_name))
        self.assertEqual(self.nuboard_file_name.suffix, self.nuboard_file.extension())
        nuboard_file = NuBoardFile.load_nuboard_file(self.nuboard_file_name)
        self.assertEqual(nuboard_file, self.nuboard_file)

    def tearDown(self) -> None:
        """Clean up temporary folder and files."""
        self.tmp_dir.cleanup()

class TestNuBoard(unittest.TestCase):
    """Test nuboard functionality."""

    def setUp(self) -> None:
        """Set up nuboard a bokeh main page."""
        self.vehicle_parameters = get_pacifica_parameters()
        self.doc = Document()
        self.scenario_builder = MockAbstractScenarioBuilder()
        self.tmp_dir = tempfile.TemporaryDirectory()
        if not os.getenv('NUPLAN_EXP_ROOT', None):
            os.environ['NUPLAN_EXP_ROOT'] = self.tmp_dir.name
        self.nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', aggregator_metric_folder='aggregator_metric')
        metric_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        simulation_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)
        self.nuboard_file_name = Path(self.tmp_dir.name) / ('nuboard_file' + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.main_paths = [str(self.nuboard_file_name)]
        self.nuboard = NuBoard(profiler_path=Path(self.tmp_dir.name), nuboard_paths=self.main_paths, scenario_builder=self.scenario_builder, vehicle_parameters=self.vehicle_parameters)

    def test_main_page(self) -> None:
        """Test if successfully construct a bokeh main page."""
        self.nuboard.main_page(doc=self.doc)
        self.assertEqual(len(self.doc.roots), 34)

    @given(frame_rate_cap=st.integers(min_value=1, max_value=60))
    def test_valid_frame_rate_cap_range(self, frame_rate_cap: int) -> None:
        """Tests valid frame rate cap range."""
        NuBoard(profiler_path=Path(self.tmp_dir.name), nuboard_paths=self.main_paths, scenario_builder=self.scenario_builder, vehicle_parameters=self.vehicle_parameters, scenario_rendering_frame_rate_cap_hz=frame_rate_cap)

    @given(frame_rate_cap=st.integers().filter(lambda x: x < 1 or x > 60))
    def test_invalid_frame_rate_cap_range(self, frame_rate_cap: int) -> None:
        """Tests invalid frame rate cap range."""
        with self.assertRaises(ValueError):
            NuBoard(profiler_path=Path(self.tmp_dir.name), nuboard_paths=self.main_paths, scenario_builder=self.scenario_builder, vehicle_parameters=self.vehicle_parameters, scenario_rendering_frame_rate_cap_hz=frame_rate_cap)

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()

class ScenarioTab(BaseTab):
    """Scenario tab in nuboard."""

    def __init__(self, doc: Document, experiment_file_data: ExperimentFileData, vehicle_parameters: VehicleParameters, scenario_builder: AbstractScenarioBuilder, async_rendering: bool=True, frame_rate_cap_hz: int=60):
        """
        Scenario tab to render metric results about a scenario.
        :param doc: Bokeh HTML document.
        :param experiment_file_data: Experiment file data.
        :param vehicle_parameters: Vehicle parameters.
        :param scenario_builder: nuPlan scenario builder instance.
        :param async_rendering: When true, will use threads to render SimulationTiles asynchronously.
        :param frame_rate_cap_hz: Maximum frames to render per second. Internally this value is capped at 60.
        """
        super().__init__(doc=doc, experiment_file_data=experiment_file_data)
        self._number_metrics_per_figure: int = 4
        self.planner_checkbox_group.name = 'scenario_planner_checkbox_group'
        self._scenario_builder = scenario_builder
        self._scenario_title_div = Div(**ScenarioTabTitleDivConfig.get_config())
        self._scalar_scenario_type_select = Select(name='scenario_scalar_scenario_type_select', css_classes=['scalar-scenario-type-select'])
        self._scalar_scenario_type_select.on_change('value', self._scalar_scenario_type_select_on_change)
        self._scalar_log_name_select = Select(name='scenario_scalar_log_name_select', css_classes=['scalar-log-name-select'])
        self._scalar_log_name_select.on_change('value', self._scalar_log_name_select_on_change)
        self._scalar_scenario_name_select = Select(name='scenario_scalar_name_select', css_classes=['scalar-scenario-name-select'])
        self._scalar_scenario_name_select.js_on_change('value', ScenarioTabUpdateWindowsSizeJSCode.get_js_code())
        self._scalar_scenario_name_select.on_change('value', self._scalar_scenario_name_select_on_change)
        self._scenario_token_multi_choice = MultiChoice(**ScenarioTabScenarioTokenMultiChoiceConfig.get_config())
        self._scenario_token_multi_choice.on_change('value', self._scenario_token_multi_choice_on_change)
        self._scenario_modal_query_btn = Button(**ScenarioTabModalQueryButtonConfig.get_config())
        self._scenario_modal_query_btn.js_on_click(ScenarioTabLoadingJSCode.get_js_code())
        self._scenario_modal_query_btn.on_click(self._scenario_modal_query_button_on_click)
        self.planner_checkbox_group.js_on_change('active', ScenarioTabLoadingJSCode.get_js_code())
        self._default_time_series_div = Div(text=' <p> No time series results, please add more experiments or\n                adjust the search filter.</p>', css_classes=['scenario-default-div'], margin=default_div_style['margin'], width=default_div_style['width'])
        self._time_series_layout = column(self._default_time_series_div, css_classes=['scenario-time-series-layout'], name='time_series_layout')
        self._default_ego_expert_states_div = Div(text=' <p> No expert and ego states, please add more experiments or\n                        adjust the search filter.</p>', css_classes=['scenario-default-div'], margin=default_div_style['margin'], width=default_div_style['width'])
        self._ego_expert_states_layout = column(self._default_ego_expert_states_div, css_classes=['scenario-ego-expert-states-layout'], name='ego_expert_states_layout')
        self._default_simulation_div = Div(text=' <p> No simulation data, please add more experiments or\n                adjust the search filter.</p>', css_classes=['scenario-default-div'], margin=default_div_style['margin'], width=default_div_style['width'])
        self._simulation_tile_layout = column(self._default_simulation_div, css_classes=['scenario-simulation-layout'], name='simulation_tile_layout')
        self._simulation_tile_layout.js_on_change('children', ScenarioTabLoadingEndJSCode.get_js_code())
        self.simulation_tile = SimulationTile(map_factory=self._scenario_builder.get_map_factory(), doc=self._doc, vehicle_parameters=vehicle_parameters, experiment_file_data=experiment_file_data, async_rendering=async_rendering, frame_rate_cap_hz=frame_rate_cap_hz)
        self._default_scenario_score_div = Div(text=' <p> No scenario score results, please add more experiments or\n                        adjust the search filter.</p>', css_classes=['scenario-default-div'], margin=default_div_style['margin'], width=default_div_style['width'])
        self._scenario_score_layout = column(self._default_scenario_score_div, css_classes=['scenario-score-layout'], name='scenario_score_layout')
        self._scenario_metric_score_data_figure_sizes = scenario_tab_style['scenario_metric_score_figure_sizes']
        self._scenario_metric_score_data: scenario_metric_score_dict_type = {}
        self._time_series_data: Dict[str, List[ScenarioTimeSeriesData]] = {}
        self._simulation_figure_data: List[SimulationData] = []
        self._available_scenario_names: List[str] = []
        self._simulation_plots: Optional[column] = None
        object_types = ['Ego', 'Vehicle', 'Pedestrian', 'Bicycle', 'Generic', 'Traffic Cone', 'Barrier', 'Czone Sign']
        self._object_checkbox_group = CheckboxGroup(labels=object_types, active=list(range(len(object_types))), css_classes=['scenario-object-checkbox-group'], name='scenario_object_checkbox_group')
        self._object_checkbox_group.on_change('active', self._object_checkbox_group_active_on_change)
        trajectories = ['Expert Trajectory', 'Ego Trajectory', 'Goal', 'Traffic Light', 'RoadBlock']
        self._traj_checkbox_group = CheckboxGroup(labels=trajectories, active=list(range(len(trajectories))), css_classes=['scenario-traj-checkbox-group'], name='scenario_traj_checkbox_group')
        self._traj_checkbox_group.on_change('active', self._traj_checkbox_group_active_on_change)
        map_objects = ['Lane', 'Intersection', 'Stop Line', 'Crosswalk', 'Walkway', 'Carpark', 'Lane Connector', 'Lane Line']
        self._map_checkbox_group = CheckboxGroup(labels=map_objects, active=list(range(len(map_objects))), css_classes=['scenario-map-checkbox-group'], name='scenario_map_checkbox_group')
        self._map_checkbox_group.on_change('active', self._map_checkbox_group_active_on_change)
        self.plot_state_keys = ['x [m]', 'y [m]', 'heading [rad]', 'velocity_x [m/s]', 'velocity_y [m/s]', 'speed [m/s]', 'acceleration_x [m/s^2]', 'acceleration_y [m/s^2]', 'acceleration [m/s^2]', 'steering_angle [rad]', 'yaw_rate [rad/s]']
        self.expert_planner_key = 'Expert'
        self._init_selection()

    @property
    def scenario_title_div(self) -> Div:
        """Return scenario title div."""
        return self._scenario_title_div

    @property
    def scalar_scenario_type_select(self) -> Select:
        """Return scalar_scenario_type_select."""
        return self._scalar_scenario_type_select

    @property
    def scalar_log_name_select(self) -> Select:
        """Return scalar_log_name_select."""
        return self._scalar_log_name_select

    @property
    def scalar_scenario_name_select(self) -> Select:
        """Return scalar_scenario_name_select."""
        return self._scalar_scenario_name_select

    @property
    def scenario_token_multi_choice(self) -> MultiChoice:
        """Return scenario_token multi choice."""
        return self._scenario_token_multi_choice

    @property
    def scenario_modal_query_btn(self) -> Button:
        """Return scenario_modal_query_button."""
        return self._scenario_modal_query_btn

    @property
    def object_checkbox_group(self) -> CheckboxGroup:
        """Return object checkbox group."""
        return self._object_checkbox_group

    @property
    def traj_checkbox_group(self) -> CheckboxGroup:
        """Return traj checkbox group."""
        return self._traj_checkbox_group

    @property
    def map_checkbox_group(self) -> CheckboxGroup:
        """Return map checkbox group."""
        return self._map_checkbox_group

    @property
    def time_series_layout(self) -> column:
        """Return time_series_layout."""
        return self._time_series_layout

    @property
    def scenario_score_layout(self) -> column:
        """Return scenario_score_layout."""
        return self._scenario_score_layout

    @property
    def simulation_tile_layout(self) -> column:
        """Return simulation_tile_layout."""
        return self._simulation_tile_layout

    @property
    def ego_expert_states_layout(self) -> column:
        """Return time_series_state_layout."""
        return self._ego_expert_states_layout

    def _update_glyph_checkbox_group(self, glyph_names: List[str]) -> None:
        """
        Update visibility of glyphs according to checkbox group.
        :param glyph_names: A list of updated glyph names.
        """
        for simulation_figure in self.simulation_tile.figures:
            simulation_figure.update_glyphs_visibility(glyph_names=glyph_names)

    def _traj_checkbox_group_active_on_change(self, attr: str, old: List[int], new: List[int]) -> None:
        """
        Helper function for traj checkbox group when the list of actives changes.
        :param attr: Attribute name.
        :param old: Old active index.
        :param new: New active index.
        """
        active_indices = list(set(old) - set(new)) + list(set(new) - set(old))
        active_labels = [self._traj_checkbox_group.labels[index] for index in active_indices]
        self._update_glyph_checkbox_group(glyph_names=active_labels)

    def _map_checkbox_group_active_on_change(self, attr: str, old: List[int], new: List[int]) -> None:
        """
        Helper function for map checkbox group when the list of actives changes.
        :param attr: Attribute name.
        :param old: Old active index.
        :param new: New active index.
        """
        active_indices = list(set(old) - set(new)) + list(set(new) - set(old))
        active_labels = [self._map_checkbox_group.labels[index] for index in active_indices]
        self._update_glyph_checkbox_group(glyph_names=active_labels)

    def _object_checkbox_group_active_on_change(self, attr: str, old: List[int], new: List[int]) -> None:
        """
        Helper function for object checkbox group when the list of actives changes.
        :param attr: Attribute name.
        :param old: Old active index.
        :param new: New active index.
        """
        active_indices = list(set(old) - set(new)) + list(set(new) - set(old))
        active_labels = [self._object_checkbox_group.labels[index] for index in active_indices]
        self._update_glyph_checkbox_group(glyph_names=active_labels)

    def file_paths_on_change(self, experiment_file_data: ExperimentFileData, experiment_file_active_index: List[int]) -> None:
        """
        Interface to update layout when file_paths is changed.
        :param experiment_file_data: Experiment file data.
        :param experiment_file_active_index: Active indexes for experiment files.
        """
        self._experiment_file_data = experiment_file_data
        self._experiment_file_active_index = experiment_file_active_index
        self.simulation_tile.init_simulations(figure_sizes=self.simulation_figure_sizes)
        self._init_selection()
        self._scenario_metric_score_data = self._update_aggregation_metric()
        self._update_scenario_plot()

    def _click_planner_checkbox_group(self, attr: Any) -> None:
        """
        Click event handler for planner_checkbox_group.
        :param attr: Clicked attributes.
        """
        scenario_metric_score_figure_data = self._render_scenario_metric_score()
        scenario_metric_score_layout = self._render_scenario_metric_layout(figure_data=scenario_metric_score_figure_data, default_div=self._default_scenario_score_div, plot_width=self._scenario_metric_score_data_figure_sizes[0], legend=False)
        self._scenario_score_layout.children[0] = layout(scenario_metric_score_layout)
        filtered_time_series_data: Dict[str, List[ScenarioTimeSeriesData]] = defaultdict(list)
        for key, time_series_data in self._time_series_data.items():
            for data in time_series_data:
                if data.planner_name not in self.enable_planner_names:
                    continue
                filtered_time_series_data[key].append(data)
        time_series_figure_data = self._render_time_series(aggregated_time_series_data=filtered_time_series_data)
        time_series_figures = self._render_scenario_metric_layout(figure_data=time_series_figure_data, default_div=self._default_time_series_div, plot_width=self.plot_sizes[0], legend=True)
        self._time_series_layout.children[0] = layout(time_series_figures)
        filtered_simulation_figures = [data for data in self._simulation_figure_data if data.planner_name in self.enable_planner_names]
        if not filtered_simulation_figures:
            simulation_layouts = column(self._default_simulation_div)
            ego_expert_state_layouts = column(self._default_ego_expert_states_div)
        else:
            simulation_layouts = gridplot([simulation_figure.plot for simulation_figure in filtered_simulation_figures], ncols=self.get_plot_cols(plot_width=self.simulation_figure_sizes[0], offset_width=scenario_tab_style['col_offset_width']), toolbar_location=None)
            ego_expert_state_layouts = self._render_ego_expert_states(simulation_figure_data=filtered_simulation_figures)
        self._simulation_tile_layout.children[0] = layout(simulation_layouts)
        self._ego_expert_states_layout.children[0] = layout(ego_expert_state_layouts)

    def _update_simulation_layouts(self) -> None:
        """Update simulation layouts."""
        self._simulation_tile_layout.children[0] = layout(self._simulation_plots)

    def _update_scenario_plot(self) -> None:
        """Update scenario plots when selection is made."""
        start_time = time.perf_counter()
        self._simulation_figure_data = []
        scenario_metric_score_figure_data = self._render_scenario_metric_score()
        scenario_metric_score_layout = self._render_scenario_metric_layout(figure_data=scenario_metric_score_figure_data, default_div=self._default_scenario_score_div, plot_width=self._scenario_metric_score_data_figure_sizes[0], legend=False)
        self._scenario_score_layout.children[0] = layout(scenario_metric_score_layout)
        self._time_series_data = self._aggregate_time_series_data()
        time_series_figure_data = self._render_time_series(aggregated_time_series_data=self._time_series_data)
        time_series_figures = self._render_scenario_metric_layout(figure_data=time_series_figure_data, default_div=self._default_time_series_div, plot_width=self.plot_sizes[0], legend=True)
        self._time_series_layout.children[0] = layout(time_series_figures)
        self._simulation_plots = self._render_simulations()
        ego_expert_state_layout = self._render_ego_expert_states(simulation_figure_data=self._simulation_figure_data)
        self._ego_expert_states_layout.children[0] = layout(ego_expert_state_layout)
        self._doc.add_next_tick_callback(self._update_simulation_layouts)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f'Rending scenario plot takes {elapsed_time:.4f} seconds.')

    def _update_planner_names(self) -> None:
        """Update planner name options in the checkbox widget."""
        self.planner_checkbox_group.labels = []
        self.planner_checkbox_group.active = []
        selected_keys = [key for key in self.experiment_file_data.simulation_scenario_keys if key.scenario_type == self._scalar_scenario_type_select.value and key.scenario_name == self._scalar_scenario_name_select.value]
        sorted_planner_names = sorted(list({key.planner_name for key in selected_keys}))
        self.planner_checkbox_group.labels = sorted_planner_names
        self.planner_checkbox_group.active = [index for index in range(len(sorted_planner_names))]

    def _scalar_scenario_type_select_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in scalar scenario type.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        if new == '':
            return
        available_log_names = self.load_log_name(scenario_type=self._scalar_scenario_type_select.value)
        self._scalar_log_name_select.options = [''] + available_log_names
        self._scalar_log_name_select.value = ''
        self._scalar_scenario_name_select.options = ['']
        self._scalar_scenario_name_select.value = ''

    def _scalar_log_name_select_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in scalar log name.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        if new == '':
            return
        available_scenario_names = self.load_scenario_names(scenario_type=self._scalar_scenario_type_select.value, log_name=self._scalar_log_name_select.value)
        self._scalar_scenario_name_select.options = [''] + available_scenario_names
        self._scalar_scenario_name_select.value = ''

    def _scalar_scenario_name_select_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in scalar scenario name.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        if self._scalar_scenario_name_select.tags:
            self.window_width = self._scalar_scenario_name_select.tags[0]
            self.window_height = self._scalar_scenario_name_select.tags[1]

    def _scenario_token_multi_choice_on_change(self, attr: str, old: List[str], new: List[str]) -> None:
        """
        Helper function to change event in scenario token multi choice.
        :param attr: Attribute.
        :param old: List of old values.
        :param new: List of new values.
        """
        available_scenario_tokens = self._experiment_file_data.available_scenario_tokens
        if not available_scenario_tokens or not new:
            return
        scenario_token_info = available_scenario_tokens.get(new[0])
        if self._scalar_scenario_type_select.value != scenario_token_info.scenario_type:
            self._scalar_scenario_type_select.value = scenario_token_info.scenario_type
        if self._scalar_log_name_select.value != scenario_token_info.log_name:
            self._scalar_log_name_select.value = scenario_token_info.log_name
        if self._scalar_scenario_name_select.value != scenario_token_info.scenario_name:
            self.scalar_scenario_name_select.value = scenario_token_info.scenario_name

    def _scenario_modal_query_button_on_click(self) -> None:
        """Helper function when click the modal query button."""
        if self._scalar_scenario_name_select.tags:
            self.window_width = self._scalar_scenario_name_select.tags[0]
            self.window_height = self._scalar_scenario_name_select.tags[1]
        self._update_planner_names()
        self._update_scenario_plot()

    def _init_selection(self) -> None:
        """Init histogram and scalar selection options."""
        self._scalar_scenario_type_select.value = ''
        self._scalar_scenario_type_select.options = []
        self._scalar_log_name_select.value = ''
        self._scalar_log_name_select.options = []
        self._scalar_scenario_name_select.value = ''
        self._scalar_scenario_name_select.options = []
        self._available_scenario_names = []
        self._simulation_figure_data = []
        if len(self._scalar_scenario_type_select.options) == 0:
            self._scalar_scenario_type_select.options = [''] + self.experiment_file_data.available_scenario_types
        if len(self._scalar_scenario_type_select.options) > 0:
            self._scalar_scenario_type_select.value = self._scalar_scenario_type_select.options[0]
        available_scenario_tokens = list(self._experiment_file_data.available_scenario_tokens.keys())
        self._scenario_token_multi_choice.options = available_scenario_tokens
        self._update_planner_names()

    @staticmethod
    def _render_scalar_figure(title: str, y_axis_label: str, hover: HoverTool, sizes: List[int], x_axis_label: Optional[str]=None, x_range: Optional[List[str]]=None, y_range: Optional[List[str]]=None) -> Figure:
        """
        Render a scalar figure.
        :param title: Plot title.
        :param y_axis_label: Y axis label.
        :param hover: Hover tool for the plot.
        :param sizes: Width and height in pixels.
        :param x_axis_label: Label in x axis.
        :param x_range: Labels in x major axis.
        :param y_range: Labels in y major axis.
        :return A time series plot.
        """
        scenario_scalar_figure = Figure(background_fill_color=PLOT_PALETTE['background_white'], title=title, css_classes=['time-series-figure'], margin=scenario_tab_style['time_series_figure_margins'], width=sizes[0], height=sizes[1], active_scroll='wheel_zoom', output_backend='webgl', x_range=x_range, y_range=y_range)
        scenario_scalar_figure.add_tools(hover)
        scenario_scalar_figure.title.text_font_size = scenario_tab_style['time_series_figure_title_text_font_size']
        scenario_scalar_figure.xaxis.axis_label_text_font_size = scenario_tab_style['time_series_figure_xaxis_axis_label_text_font_size']
        scenario_scalar_figure.xaxis.major_label_text_font_size = scenario_tab_style['time_series_figure_xaxis_major_label_text_font_size']
        scenario_scalar_figure.yaxis.axis_label_text_font_size = scenario_tab_style['time_series_figure_yaxis_axis_label_text_font_size']
        scenario_scalar_figure.yaxis.major_label_text_font_size = scenario_tab_style['time_series_figure_yaxis_major_label_text_font_size']
        scenario_scalar_figure.toolbar.logo = None
        scenario_scalar_figure.xaxis.major_label_orientation = np.pi / 4
        scenario_scalar_figure.yaxis.axis_label = y_axis_label
        scenario_scalar_figure.xaxis.axis_label = x_axis_label
        return scenario_scalar_figure

    def _update_aggregation_metric(self) -> scenario_metric_score_dict_type:
        """
        Update metric score for each scenario.
        :return A dict of log name: {scenario names and their metric scores}.
        """
        data: scenario_metric_score_dict_type = defaultdict(lambda: defaultdict(list))
        for index, metric_aggregator_dataframes in enumerate(self.experiment_file_data.metric_aggregator_dataframes):
            if index not in self._experiment_file_active_index:
                continue
            for file_index, (metric_aggregator_filename, metric_aggregator_dataframe) in enumerate(metric_aggregator_dataframes.items()):
                columns = set(list(metric_aggregator_dataframe.columns))
                non_metric_columns = {'scenario', 'log_name', 'scenario_type', 'num_scenarios', 'planner_name', 'aggregator_type'}
                metric_columns = sorted(list(columns - non_metric_columns))
                for _, row_data in metric_aggregator_dataframe.iterrows():
                    num_scenarios = row_data['num_scenarios']
                    if not np.isnan(num_scenarios):
                        continue
                    planner_name = row_data['planner_name']
                    scenario_name = row_data['scenario']
                    log_name = row_data['log_name']
                    for metric_column in metric_columns:
                        score = row_data[metric_column]
                        if score is not None:
                            data[log_name][scenario_name].append(ScenarioMetricScoreData(experiment_index=index, metric_aggregator_file_name=metric_aggregator_filename, metric_aggregator_file_index=file_index, planner_name=planner_name, metric_statistic_name=metric_column, score=np.round(score, 4)))
        return data

    def _aggregate_time_series_data(self) -> Dict[str, List[ScenarioTimeSeriesData]]:
        """
        Aggregate time series data.
        :return A dict of metric statistic names and their data.
        """
        aggregated_time_series_data: Dict[str, List[ScenarioTimeSeriesData]] = {}
        scenario_types = tuple([self._scalar_scenario_type_select.value]) if self._scalar_scenario_type_select.value else None
        log_names = tuple([self._scalar_log_name_select.value]) if self._scalar_log_name_select.value else None
        if not len(self._scalar_scenario_name_select.value):
            return aggregated_time_series_data
        for index, metric_statistics_dataframes in enumerate(self.experiment_file_data.metric_statistics_dataframes):
            if index not in self._experiment_file_active_index:
                continue
            for metric_statistics_dataframe in metric_statistics_dataframes:
                planner_names = metric_statistics_dataframe.planner_names
                if metric_statistics_dataframe.metric_statistic_name not in aggregated_time_series_data:
                    aggregated_time_series_data[metric_statistics_dataframe.metric_statistic_name] = []
                for planner_name in planner_names:
                    data_frame = metric_statistics_dataframe.query_scenarios(scenario_names=tuple([str(self._scalar_scenario_name_select.value)]), scenario_types=scenario_types, planner_names=tuple([planner_name]), log_names=log_names)
                    if not len(data_frame):
                        continue
                    time_series_headers = metric_statistics_dataframe.time_series_headers
                    time_series: pandas.DataFrame = data_frame[time_series_headers]
                    if time_series[time_series_headers[0]].iloc[0] is None:
                        continue
                    time_series_values: npt.NDArray[np.float64] = np.round(np.asarray(list(chain.from_iterable(time_series[metric_statistics_dataframe.time_series_values_column]))), 4)
                    time_series_timestamps = list(chain.from_iterable(time_series[metric_statistics_dataframe.time_series_timestamp_column]))
                    time_series_unit = time_series[metric_statistics_dataframe.time_series_unit_column].iloc[0]
                    time_series_selected_frames = metric_statistics_dataframe.get_time_series_selected_frames
                    scenario_time_series_data = ScenarioTimeSeriesData(experiment_index=index, planner_name=planner_name, time_series_values=time_series_values, time_series_timestamps=time_series_timestamps, time_series_unit=time_series_unit, time_series_selected_frames=time_series_selected_frames)
                    aggregated_time_series_data[metric_statistics_dataframe.metric_statistic_name].append(scenario_time_series_data)
        return aggregated_time_series_data

    def _render_time_series(self, aggregated_time_series_data: Dict[str, List[ScenarioTimeSeriesData]]) -> Dict[str, Figure]:
        """
        Render time series plots.
        :param aggregated_time_series_data: Aggregated scenario time series data.
        :return A dict of figure name and figures.
        """
        time_series_figures: Dict[str, Figure] = {}
        for metric_statistic_name, scenario_time_series_data in aggregated_time_series_data.items():
            for data in scenario_time_series_data:
                if not len(data.time_series_values):
                    continue
                if metric_statistic_name not in time_series_figures:
                    time_series_figures[metric_statistic_name] = self._render_scalar_figure(title=metric_statistic_name, y_axis_label=data.time_series_unit, x_axis_label='frame', hover=HoverTool(tooltips=[('Frame', '@x'), ('Value', '@y{0.0000}'), ('Time_us', '@time_us'), ('Planner', '$name')]), sizes=self.plot_sizes)
                planner_name = data.planner_name + f' ({self.get_file_path_last_name(data.experiment_index)})'
                color = self.experiment_file_data.file_path_colors[data.experiment_index][data.planner_name]
                time_series_figure = time_series_figures[metric_statistic_name]
                timestamp_frames = data.time_series_selected_frames if data.time_series_selected_frames is not None else list(range(len(data.time_series_timestamps)))
                data_source = ColumnDataSource(dict(x=timestamp_frames, y=data.time_series_values, time_us=data.time_series_timestamps))
                if data.time_series_selected_frames is not None:
                    time_series_figure.scatter(x='x', y='y', name=planner_name, color=color, legend_label=planner_name, source=data_source)
                else:
                    time_series_figure.line(x='x', y='y', name=planner_name, color=color, legend_label=planner_name, source=data_source)
        return time_series_figures

    def _render_scenario_metric_score_scatter(self, scatter_figure: Figure, scenario_metric_score_data: Dict[str, List[ScenarioMetricScoreData]]) -> None:
        """
        Render scatter plot with scenario metric score data.
        :param scatter_figure: A scatter figure.
        :param scenario_metric_score_data: Metric score data for a scenario.
        """
        data_sources: Dict[str, ScenarioMetricScoreDataSource] = {}
        for metric_name, metric_score_data in scenario_metric_score_data.items():
            for index, score_data in enumerate(metric_score_data):
                experiment_name = self.get_file_path_last_name(score_data.experiment_index)
                legend_label = f'{score_data.planner_name} ({experiment_name})'
                data_source_index = legend_label + f' - {score_data.metric_aggregator_file_index})'
                if data_source_index not in data_sources:
                    data_sources[data_source_index] = ScenarioMetricScoreDataSource(xs=[], ys=[], planners=[], aggregators=[], experiments=[], fill_colors=[], marker=self.get_scatter_sign(score_data.metric_aggregator_file_index), legend_label=legend_label)
                fill_color = self.experiment_file_data.file_path_colors[score_data.experiment_index][score_data.planner_name]
                data_sources[data_source_index].xs.append(score_data.metric_statistic_name)
                data_sources[data_source_index].ys.append(score_data.score)
                data_sources[data_source_index].planners.append(score_data.planner_name)
                data_sources[data_source_index].aggregators.append(score_data.metric_aggregator_file_name)
                data_sources[data_source_index].experiments.append(self.get_file_path_last_name(score_data.experiment_index))
                data_sources[data_source_index].fill_colors.append(fill_color)
        for legend_label, data_source in data_sources.items():
            sources = ColumnDataSource(dict(xs=data_source.xs, ys=data_source.ys, planners=data_source.planners, experiments=data_source.experiments, aggregators=data_source.aggregators, fill_colors=data_source.fill_colors, line_colors=data_source.fill_colors))
            glyph_renderer = self.get_scatter_render_func(scatter_sign=data_source.marker, scatter_figure=scatter_figure)
            glyph_renderer(x='xs', y='ys', size=10, fill_color='fill_colors', line_color='fill_colors', source=sources)

    def _render_scenario_metric_score(self) -> Dict[str, Figure]:
        """
        Render scenario metric score plot.
        :return A dict of figure names and figures.
        """
        if not self._scalar_log_name_select.value or not self._scalar_scenario_name_select.value or (not self._scenario_metric_score_data):
            return {}
        selected_scenario_metric_score: List[ScenarioMetricScoreData] = self._scenario_metric_score_data[self._scalar_log_name_select.value][self._scalar_scenario_name_select.value]
        data: Dict[str, List[ScenarioMetricScoreData]] = defaultdict(list)
        for scenario_metric_score_data in selected_scenario_metric_score:
            if scenario_metric_score_data.planner_name not in self.enable_planner_names:
                continue
            metric_statistic_name = scenario_metric_score_data.metric_statistic_name
            data[metric_statistic_name].append(scenario_metric_score_data)
        metric_statistic_names = sorted(list(set(data.keys())))
        if 'score' in metric_statistic_names:
            metric_statistic_names.remove('score')
            metric_statistic_names.append('score')
        hover = HoverTool(tooltips=[('Metric', '@xs'), ('Score', '@ys'), ('Planner', '@planners'), ('Experiment', '@experiments'), ('Aggregator', '@aggregators')])
        number_of_figures = ceil(len(metric_statistic_names) / self._number_metrics_per_figure)
        scenario_metric_score_figures: Dict[str, Figure] = defaultdict()
        for index in range(number_of_figures):
            starting_index = index * self._number_metrics_per_figure
            ending_index = starting_index + self._number_metrics_per_figure
            selected_metric_names = metric_statistic_names[starting_index:ending_index]
            scenario_metric_score_figure = self._render_scalar_figure(title='', y_axis_label='score', hover=hover, x_range=selected_metric_names, sizes=self._scenario_metric_score_data_figure_sizes)
            metric_score_data = {metric_name: data[metric_name] for metric_name in selected_metric_names}
            self._render_scenario_metric_score_scatter(scatter_figure=scenario_metric_score_figure, scenario_metric_score_data=metric_score_data)
            scenario_metric_score_figures[str(index)] = scenario_metric_score_figure
        return scenario_metric_score_figures

    def _render_grid_plot(self, figures: Dict[str, Figure], plot_width: int, legend: bool=True) -> LayoutDOM:
        """
        Render a grid plot.
        :param figures: A dict of figure names and figures.
        :param plot_width: Width of each plot.
        :param legend: If figures have legends.
        :return A grid plot.
        """
        figure_plot_list: List[Figure] = []
        for figure_name, figure_plot in figures.items():
            if legend:
                figure_plot.legend.label_text_font_size = scenario_tab_style['plot_legend_label_text_font_size']
                figure_plot.legend.background_fill_alpha = 0.0
                figure_plot.legend.click_policy = 'hide'
            figure_plot_list.append(figure_plot)
        grid_plot = gridplot(figure_plot_list, ncols=self.get_plot_cols(plot_width=plot_width), toolbar_location='left')
        return grid_plot

    def _render_scenario_metric_layout(self, figure_data: Dict[str, Figure], default_div: Div, plot_width: int, legend: bool=True) -> column:
        """
        Render a layout for scenario metric.
        :param figure_data: A dict of figure_data.
        :param default_div: Default message when there is no result.
        :param plot_width: Figure width.
        :param legend: If figures have legends.
        :return A bokeh column layout.
        """
        if not figure_data:
            return column(default_div)
        grid_plot = self._render_grid_plot(figures=figure_data, plot_width=plot_width, legend=legend)
        scenario_metric_layout = column(grid_plot)
        return scenario_metric_layout

    def _render_simulations(self) -> column:
        """
        Render simulation plot.
        :return: A list of Bokeh columns or rows.
        """
        selected_keys = [key for key in self.experiment_file_data.simulation_scenario_keys if key.scenario_type == self._scalar_scenario_type_select.value and key.log_name == self._scalar_log_name_select.value and (key.scenario_name == self._scalar_scenario_name_select.value) and (key.nuboard_file_index in self._experiment_file_active_index)]
        if not selected_keys:
            self._scenario_title_div.text = '-'
            simulation_layouts = column(self._default_simulation_div)
        else:
            hidden_glyph_names = [label for checkbox_group in [self._object_checkbox_group, self._traj_checkbox_group, self._map_checkbox_group] for index, label in enumerate(checkbox_group.labels) if index not in checkbox_group.active]
            self._simulation_figure_data = self.simulation_tile.render_simulation_tiles(selected_scenario_keys=selected_keys, figure_sizes=self.simulation_figure_sizes, hidden_glyph_names=hidden_glyph_names)
            simulation_figures = [data.plot for data in self._simulation_figure_data]
            simulation_layouts = gridplot(simulation_figures, ncols=self.get_plot_cols(plot_width=self.simulation_figure_sizes[0], offset_width=scenario_tab_style['col_offset_width']), toolbar_location=None)
            self._scenario_title_div.text = f'{self._scalar_scenario_type_select.value} - {self._scalar_log_name_select.value} - {self._scalar_scenario_name_select.value}'
        return simulation_layouts

    @staticmethod
    def _get_ego_expert_states(state_key: str, ego_state: EgoState) -> float:
        """
        Get states based on the state key.
        :param state_key: Ego state key.
        :param ego_state: Ego state.
        :return ego state based on the key.
        """
        if state_key == 'x [m]':
            return cast(float, ego_state.car_footprint.center.x)
        elif state_key == 'y [m]':
            return cast(float, ego_state.car_footprint.center.y)
        elif state_key == 'velocity_x [m/s]':
            return cast(float, ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
        elif state_key == 'velocity_y [m/s]':
            return cast(float, ego_state.dynamic_car_state.rear_axle_velocity_2d.y)
        elif state_key == 'speed [m/s]':
            return cast(float, ego_state.dynamic_car_state.speed)
        elif state_key == 'acceleration_x [m/s^2]':
            return cast(float, ego_state.dynamic_car_state.rear_axle_acceleration_2d.x)
        elif state_key == 'acceleration_y [m/s^2]':
            return cast(float, ego_state.dynamic_car_state.rear_axle_acceleration_2d.y)
        elif state_key == 'acceleration [m/s^2]':
            return cast(float, ego_state.dynamic_car_state.acceleration)
        elif state_key == 'heading [rad]':
            return cast(float, ego_state.car_footprint.center.heading)
        elif state_key == 'steering_angle [rad]':
            return cast(float, ego_state.dynamic_car_state.tire_steering_rate)
        elif state_key == 'yaw_rate [rad/s]':
            return cast(float, ego_state.dynamic_car_state.angular_velocity)
        else:
            raise ValueError(f'{state_key} not available!')

    def _render_ego_expert_state_glyph(self, ego_expert_plot_aggregated_states: scenario_ego_expert_state_figure_type, ego_expert_plot_colors: Dict[str, str]) -> column:
        """
        Render line and circle glyphs on ego_expert_state figures and get a grid plot.
        :param ego_expert_plot_aggregated_states: Aggregated ego and expert states over frames.
        :param ego_expert_plot_colors: Colors for different planners.
        :return Column layout for ego and expert states.
        """
        ego_expert_state_figures: Dict[str, Figure] = defaultdict()
        for plot_state_key in self.plot_state_keys:
            hover = HoverTool(tooltips=[('Frame', '@x'), ('Value', '@y{0.0000}'), ('Planner', '$name')])
            ego_expert_state_figure = self._render_scalar_figure(title='', y_axis_label=plot_state_key, x_axis_label='frame', hover=hover, sizes=scenario_tab_style['ego_expert_state_figure_sizes'])
            ego_expert_state_figure.yaxis.formatter = BasicTickFormatter(use_scientific=False)
            ego_expert_state_figures[plot_state_key] = ego_expert_state_figure
        for planner_name, plot_states in ego_expert_plot_aggregated_states.items():
            color = ego_expert_plot_colors.get(planner_name, None)
            if not color:
                color = None
            for plot_state_key, plot_state_values in plot_states.items():
                ego_expert_state_figure = ego_expert_state_figures[plot_state_key]
                data_source = ColumnDataSource(dict(x=list(range(len(plot_state_values))), y=np.round(plot_state_values, 2)))
                if self.expert_planner_key in planner_name:
                    ego_expert_state_figure.circle(x='x', y='y', name=planner_name, color=color, legend_label=planner_name, source=data_source, size=2)
                else:
                    ego_expert_state_figure.line(x='x', y='y', name=planner_name, color=color, legend_label=planner_name, source=data_source, line_width=1)
        ego_expert_states_layout = self._render_grid_plot(figures=ego_expert_state_figures, plot_width=scenario_tab_style['ego_expert_state_figure_sizes'][0], legend=True)
        return ego_expert_states_layout

    def _get_ego_expert_plot_color(self, planner_name: str, file_path_index: int, figure_planer_name: str) -> str:
        """
        Get color for ego expert plot states based on the planner name.
        :param planner_name: Plot planner name.
        :param file_path_index: File path index for the plot.
        :param figure_planer_name: Figure original planner name.
        """
        return cast(str, self.experiment_file_data.expert_color_palettes[file_path_index] if self.expert_planner_key in planner_name else self.experiment_file_data.file_path_colors[file_path_index][figure_planer_name])

    def _render_ego_expert_states(self, simulation_figure_data: List[SimulationData]) -> column:
        """
        Render expert and ego time series states. Make sure it is called after _render_simulation.
        :param simulation_figure_data: Simulation figure data after rendering simulation.
        :return Column layout for ego and expert states.
        """
        if not simulation_figure_data:
            return column(self._default_ego_expert_states_div)
        ego_expert_plot_aggregated_states: scenario_ego_expert_state_figure_type = defaultdict(lambda: defaultdict(list))
        ego_expert_plot_colors: Dict[str, str] = defaultdict()
        for figure_data in simulation_figure_data:
            experiment_file_index = figure_data.simulation_figure.file_path_index
            experiment_name = self.get_file_path_last_name(experiment_file_index)
            expert_planner_name = f'{self.expert_planner_key} - ({experiment_name})'
            ego_planner_name = f'{figure_data.planner_name} - ({experiment_name})'
            ego_expert_states = {expert_planner_name: figure_data.simulation_figure.scenario.get_expert_ego_trajectory(), ego_planner_name: figure_data.simulation_figure.simulation_history.extract_ego_state}
            for planner_name, planner_states in ego_expert_states.items():
                ego_expert_plot_colors[planner_name] = self._get_ego_expert_plot_color(planner_name=planner_name, figure_planer_name=figure_data.planner_name, file_path_index=figure_data.simulation_figure.file_path_index)
                if planner_name in ego_expert_plot_aggregated_states:
                    continue
                for planner_state in planner_states:
                    for plot_state_key in self.plot_state_keys:
                        state_key_value = self._get_ego_expert_states(state_key=plot_state_key, ego_state=planner_state)
                        ego_expert_plot_aggregated_states[planner_name][plot_state_key].append(state_key_value)
        ego_expert_states_layout = self._render_ego_expert_state_glyph(ego_expert_plot_aggregated_states=ego_expert_plot_aggregated_states, ego_expert_plot_colors=ego_expert_plot_colors)
        return ego_expert_states_layout

class CloudTab:
    """Cloud tab in nuboard."""

    def __init__(self, doc: Document, configuration_tab: ConfigurationTab, s3_bucket: Optional[str]=''):
        """
        Cloud tab for remote connection features.
        :param doc: Bokeh HTML document.
        :param configuration_tab: Configuration tab.
        :param s3_bucket: Aws s3 bucket name.
        """
        self._doc = doc
        self._configuration_tab = configuration_tab
        self._nuplan_exp_root = os.getenv('NUPLAN_EXP_ROOT', None)
        assert self._nuplan_exp_root is not None, 'Please set environment variable: NUPLAN_EXP_ROOT!'
        download_path = Path(self._nuplan_exp_root)
        download_path.mkdir(parents=True, exist_ok=True)
        self._default_datasource_dict = dict(object=['-'], last_modified=['-'], timestamp=['-'], size=['-'])
        self._s3_content_datasource = ColumnDataSource(data=self._default_datasource_dict)
        self._selected_column = TextInput()
        self._selected_row = TextInput()
        self.s3_bucket_name = Div(**S3TabBucketNameConfig.get_config())
        self.s3_bucket_name.js_on_change('text', S3TabDataTableUpdateJSCode.get_js_code())
        self.s3_error_text = Div(**S3TabErrorTextConfig.get_config())
        self.s3_download_text_input = TextInput(**S3TabDownloadTextInputConfig.get_config())
        self.s3_download_button = Button(**S3TabDownloadButtonConfig.get_config())
        self.s3_download_button.on_click(self._s3_download_button_on_click)
        self.s3_download_button.js_on_click(S3TabLoadingJSCode.get_js_code())
        self.s3_download_button.js_on_change('disabled', S3TabDownloadUpdateJSCode.get_js_code())
        self.s3_bucket_text_input = TextInput(**S3TabBucketTextInputConfig.get_config(), value=s3_bucket)
        self.s3_access_key_id_text_input = TextInput(**S3TabS3AccessKeyIDTextInputConfig.get_config())
        self.s3_secret_access_key_password_input = PasswordInput(**S3TabS3SecretAccessKeyPasswordTextInputConfig.get_config())
        self.s3_bucket_prefix_text_input = TextInput(**S3TabS3BucketPrefixTextInputConfig.get_config())
        self.s3_modal_query_btn = Button(**S3TabS3ModalQueryButtonConfig.get_config())
        self.s3_modal_query_btn.on_click(self._s3_modal_query_on_click)
        self.s3_modal_query_btn.js_on_click(S3TabLoadingJSCode.get_js_code())
        self._default_columns = [TableColumn(**S3TabObjectColumnConfig.get_config()), TableColumn(**S3TabLastModifiedColumnConfig.get_config()), TableColumn(**S3TabTimeStampColumnConfig.get_config()), TableColumn(**S3TabSizeColumnConfig.get_config())]
        self._s3_content_datasource = ColumnDataSource(data=self._default_datasource_dict)
        self._s3_content_datasource.js_on_change('data', S3TabDataTableUpdateJSCode.get_js_code())
        self._s3_content_datasource.selected.js_on_change('indices', S3TabContentDataSourceOnSelected.get_js_code(selected_column=self._selected_column, selected_row=self._selected_row))
        self._s3_content_datasource.selected.js_on_change('indices', S3TabContentDataSourceOnSelectedLoadingJSCode.get_js_code(source=self._s3_content_datasource, selected_column=self._selected_column))
        self._s3_content_datasource.selected.on_change('indices', self._s3_data_source_on_selected)
        self.data_table = DataTable(source=self._s3_content_datasource, columns=self._default_columns, **S3TabDataTableConfig.get_config())
        self._s3_client: Optional[boto3.client] = None
        if s3_bucket:
            self._update_blob_store(s3_bucket=s3_bucket, s3_prefix='')

    def _update_blob_store(self, s3_bucket: str, s3_prefix: str='') -> None:
        """
        :param s3_bucket:
        :param s3_prefix:
        """
        aws_profile_name = bytes(self.s3_access_key_id_text_input.value + self.s3_secret_access_key_password_input.value, encoding='utf-8')
        hash_md5 = hashlib.md5(aws_profile_name)
        profile = hash_md5.hexdigest()
        self._s3_client = get_s3_client(aws_access_key_id=self.s3_access_key_id_text_input.value, aws_secret_access_key=self.s3_secret_access_key_password_input.value, profile_name=profile)
        s3_path = os.path.join(s3_bucket, s3_prefix)
        s3_file_result_message = get_s3_file_contents(s3_path=s3_path, include_previous_folder=True, client=self._s3_client)
        self._load_s3_contents(s3_file_result_message=s3_file_result_message)
        self.s3_error_text.text = s3_file_result_message.s3_connection_status.return_message
        if s3_file_result_message.s3_connection_status.success:
            self.s3_bucket_name.text = s3_bucket

    def _s3_modal_query_on_click(self) -> None:
        """On click function for modal query button."""
        self._update_blob_store(s3_bucket=self.s3_bucket_text_input.value, s3_prefix=self.s3_bucket_prefix_text_input.value)

    def _s3_data_source_on_selected(self, attr: str, old: List[int], new: List[int]) -> None:
        """Helper function when select a row in data source."""
        if not new:
            return
        row_index = new[0]
        self._s3_content_datasource.selected.update(indices=[])
        column_index = int(self._selected_column.value)
        s3_prefix = self.data_table.source.data['object'][row_index]
        if column_index == 0:
            if not s3_prefix or s3_prefix == '-':
                return
            if '..' in s3_prefix:
                s3_prefix = Path(s3_prefix).parents[1].name
            self._update_blob_store(s3_bucket=self.s3_bucket_text_input.value, s3_prefix=s3_prefix)
        else:
            if '..' in s3_prefix or '-' == s3_prefix:
                return
            self.s3_download_text_input.value = s3_prefix

    def _update_data_table_source(self, data_sources: Dict[str, List[Any]]) -> None:
        """Update data table source."""
        self.data_table.source.data = data_sources

    def _load_s3_contents(self, s3_file_result_message: S3FileResultMessage) -> None:
        """
        Load s3 contents into a data table.
        :param s3_file_result_message: File content and return messages from s3 connection.
        """
        file_contents = s3_file_result_message.file_contents
        if not s3_file_result_message.s3_connection_status.success or len(s3_file_result_message.file_contents) <= 1:
            default_data_sources = self._default_datasource_dict
            self._doc.add_next_tick_callback(partial(self._update_data_table_source, data_sources=default_data_sources))
        else:
            data_sources: Dict[str, List[Any]] = {'object': [], 'last_modified': [], 'timestamp': [], 'size': []}
            for file_name, content in file_contents.items():
                data_sources['object'].append(file_name)
                data_sources['last_modified'].append(content.last_modified_day if content.last_modified is not None else '')
                data_sources['timestamp'].append(content.date_string if content.date_string is not None else '')
                data_sources['size'].append(content.kb_size() if content.kb_size() is not None else '')
            self._doc.add_next_tick_callback(partial(self._update_data_table_source, data_sources=data_sources))

    def _reset_s3_download_button(self) -> None:
        """Reset s3 download button."""
        self.s3_download_button.label = 'Download'
        self.s3_download_button.disabled = False
        self.s3_download_text_input.disabled = False

    def _update_error_text_label(self, text: str) -> None:
        """Update error text message in a sequential manner."""
        self.s3_error_text.text = text

    def _s3_download_prefixes(self) -> None:
        """Download s3 prefixes and update progress in a sequential manner."""
        try:
            start_time = time.perf_counter()
            if not self._s3_client:
                raise Boto3Error('No s3 connection!')
            selected_s3_bucket = str(self.s3_bucket_name.text).strip()
            selected_s3_prefix = str(self.s3_download_text_input.value).strip()
            selected_s3_path = os.path.join(selected_s3_bucket, selected_s3_prefix)
            s3_result_file_contents = get_s3_file_contents(s3_path=selected_s3_path, client=self._s3_client, include_previous_folder=False)
            s3_nuboard_file_result = check_s3_nuboard_files(s3_result_file_contents.file_contents, s3_client=self._s3_client, s3_path=selected_s3_path)
            if not s3_nuboard_file_result.s3_connection_status.success:
                raise Boto3Error(s3_nuboard_file_result.s3_connection_status.return_message)
            if not s3_result_file_contents.file_contents:
                raise Boto3Error(f'No objects exist in the path: {selected_s3_path}')
            self._download_s3_file_contents(s3_result_file_contents=s3_result_file_contents, selected_s3_bucket=selected_s3_bucket)
            self._update_s3_nuboard_file_main_path(s3_nuboard_file_result=s3_nuboard_file_result, selected_prefix=selected_s3_prefix)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            successful_message = f'Downloaded to {self._nuplan_exp_root} and took {elapsed_time:.4f} seconds'
            logger.info('Downloaded to {} and took {:.4f} seconds'.format(self._nuplan_exp_root, elapsed_time))
            self._doc.add_next_tick_callback(partial(self._update_error_text_label, text=successful_message))
        except Exception as e:
            logger.info(str(e))
            self.s3_error_text.text = str(e)
        self._doc.add_next_tick_callback(self._reset_s3_download_button)

    def _update_s3_nuboard_file_main_path(self, s3_nuboard_file_result: S3NuBoardFileResultMessage, selected_prefix: str) -> None:
        """
        Update nuboard file simulation and metric main path.
        :param s3_nuboard_file_result: S3 nuboard file result.
        :param selected_prefix: Selected prefix on s3.
        """
        nuboard_file = s3_nuboard_file_result.nuboard_file
        nuboard_filename = s3_nuboard_file_result.nuboard_filename
        if not nuboard_file or not nuboard_filename or (not self._nuplan_exp_root):
            return
        main_path = Path(self._nuplan_exp_root) / selected_prefix
        nuboard_file.simulation_main_path = str(main_path)
        nuboard_file.metric_main_path = str(main_path)
        metric_path = main_path / nuboard_file.metric_folder
        if not metric_path.exists():
            metric_path.mkdir(parents=True, exist_ok=True)
        simulation_path = main_path / nuboard_file.simulation_folder
        if not simulation_path.exists():
            simulation_path.mkdir(parents=True, exist_ok=True)
        aggregator_metric_path = main_path / nuboard_file.aggregator_metric_folder
        if not aggregator_metric_path.exists():
            aggregator_metric_path.mkdir(parents=True, exist_ok=True)
        save_path = main_path / nuboard_filename
        nuboard_file.save_nuboard_file(save_path)
        logger.info('Updated nubBard main path in {} to {}'.format(save_path, main_path))
        self._configuration_tab.add_nuboard_file_to_experiments(nuboard_file=s3_nuboard_file_result.nuboard_file)

    def _download_s3_file_contents(self, s3_result_file_contents: S3FileResultMessage, selected_s3_bucket: str) -> None:
        """
        Download s3 file contents.
        :param s3_result_file_contents: S3 file result contents.
        :param selected_s3_bucket: Selected s3 bucket name.
        """
        for index, (file_name, content) in enumerate(s3_result_file_contents.file_contents.items()):
            if '..' in file_name:
                continue
            s3_path = os.path.join(selected_s3_bucket, file_name)
            if not file_name.endswith('/'):
                s3_connection_message = download_s3_file(s3_path=s3_path, s3_client=self._s3_client, file_content=content, save_path=self._nuplan_exp_root)
            else:
                s3_connection_message = download_s3_path(s3_path=s3_path, s3_client=self._s3_client, save_path=self._nuplan_exp_root)
            if s3_connection_message.success:
                text_message = f'Downloaded {file_name} ({index + 1} / {len(s3_result_file_contents.file_contents)})'
                logger.info('Downloaded {} / ({}/{})'.format(file_name, index + 1, len(s3_result_file_contents.file_contents)))
                self._doc.add_next_tick_callback(partial(self._update_error_text_label, text=text_message))

    def _s3_download_button_on_click(self) -> None:
        """Function to call when the download button is click."""
        selected_s3_bucket = str(self.s3_bucket_name.text).strip()
        self.s3_download_button.label = 'Downloading...'
        self.s3_download_button.disabled = True
        self.s3_download_text_input.disabled = True
        if not selected_s3_bucket:
            self.s3_error_text.text = 'Please connect to a s3 bucket'
            self._doc.add_next_tick_callback(self._reset_s3_download_button)
            return
        selected_s3_prefix = str(self.s3_download_text_input.value).strip()
        if not selected_s3_prefix:
            self.s3_error_text.text = 'Please input a prefix'
            self._doc.add_next_tick_callback(self._reset_s3_download_button)
            return
        self._doc.add_next_tick_callback(self._s3_download_prefixes)

class ConfigurationTab:
    """Configuration tab for nuboard."""

    def __init__(self, doc: Document, experiment_file_data: ExperimentFileData, tabs: List[BaseTab]):
        """
        Configuration tab about configurating nuboard.
        :param experiment_file_data: Experiment file data.
        :param tabs: A list of tabs to be updated when configuration is changed.
        """
        self._doc = doc
        self._tabs = tabs
        self.experiment_file_data = experiment_file_data
        self._file_path_input = FileInput(accept=NuBoardFile.extension(), css_classes=['file-path-input'], margin=configuration_tab_style['file_path_input_margin'], name='file_path_input')
        self._file_path_input.on_change('value', self._add_experiment_file)
        self._experiment_file_path_checkbox_group = CheckboxGroup(labels=self.experiment_file_path_stems, active=[index for index in range(len(self.experiment_file_data.file_paths))], name='experiment_file_path_checkbox_group', css_classes=['experiment-file-path-checkbox-group'])
        self._experiment_file_path_checkbox_group.on_click(self._click_experiment_file_path_checkbox)
        if self.experiment_file_data.file_paths:
            self._file_paths_on_change()

    @property
    def experiment_file_path_stems(self) -> List[str]:
        """Return a list of file path stems."""
        experiment_paths = []
        for file_path in self.experiment_file_data.file_paths:
            metric_path = file_path.current_path / file_path.metric_folder
            if metric_path.exists():
                experiment_file_path_stem = file_path.current_path
            else:
                experiment_file_path_stem = file_path.metric_main_path
            if isinstance(experiment_file_path_stem, str):
                experiment_file_path_stem = pathlib.Path(experiment_file_path_stem)
            experiment_file_path_stem = '/'.join([experiment_file_path_stem.parts[-2], experiment_file_path_stem.parts[-1]])
            experiment_paths.append(experiment_file_path_stem)
        return experiment_paths

    @property
    def file_path_input(self) -> FileInput:
        """Return the file path input widget."""
        return self._file_path_input

    @property
    def experiment_file_path_checkbox_group(self) -> CheckboxGroup:
        """Return experiment file path checkboxgroup."""
        return self._experiment_file_path_checkbox_group

    def _click_experiment_file_path_checkbox(self, attr: Any) -> None:
        """
        Click event handler for experiment_file_path_checkbox_group.
        :param attr: Clicked attributes.
        """
        self._file_paths_on_change()

    def add_nuboard_file_to_experiments(self, nuboard_file: NuBoardFile) -> None:
        """
        Add nuboard files to experiments.
        :param nuboard_file: Added nuboard file.
        """
        nuboard_file.current_path = Path(nuboard_file.metric_main_path)
        if nuboard_file not in self.experiment_file_data.file_paths:
            self.experiment_file_data.update_data(file_paths=[nuboard_file])
            self._experiment_file_path_checkbox_group.labels = self.experiment_file_path_stems
            self._experiment_file_path_checkbox_group.active += [len(self.experiment_file_path_stems) - 1]
            self._file_paths_on_change()

    def _add_experiment_file(self, attr: str, old: bytes, new: bytes) -> None:
        """
        Event responds to file change.
        :param attr: Attribute name.
        :param old: Old value.
        :param new: New value.
        """
        if not new:
            return
        try:
            decoded_string = base64.b64decode(new)
            file_stream = io.BytesIO(decoded_string)
            data = pickle.load(file_stream)
            nuboard_file = NuBoardFile.deserialize(data=data)
            self.add_nuboard_file_to_experiments(nuboard_file=nuboard_file)
            file_stream.close()
        except (OSError, IOError) as e:
            logger.info(f'Error loading experiment file. {str(e)}.')

    def _file_paths_on_change(self) -> None:
        """Function to call when we change file paths."""
        for tab in self._tabs:
            tab.file_paths_on_change(experiment_file_data=self.experiment_file_data, experiment_file_active_index=self._experiment_file_path_checkbox_group.active)

class TestScenarioTab(SkeletonTestTab):
    """Test nuboard scenario tab functionality."""

    def setUp(self) -> None:
        """Set up a scenario tab."""
        super().setUp()
        vehicle_parameters = get_pacifica_parameters()
        scenario_builder = MockAbstractScenarioBuilder()
        self.experiment_file_data = ExperimentFileData(file_paths=[self.nuboard_file])
        self.scenario_tab = ScenarioTab(experiment_file_data=self.experiment_file_data, scenario_builder=scenario_builder, vehicle_parameters=vehicle_parameters, doc=self.doc)

    def test_update_scenario(self) -> None:
        """Test functions corresponding to selection changes work as expected."""
        self.scenario_tab.file_paths_on_change(experiment_file_data=self.experiment_file_data, experiment_file_active_index=[0])
        self.scenario_tab._scalar_scenario_type_select.value = self.scenario_tab._scalar_scenario_type_select.options[1]
        self.scenario_tab._scalar_log_name_select.value = self.scenario_tab._scalar_log_name_select.options[1]
        self.scenario_tab._scalar_scenario_name_select.value = self.scenario_tab._scalar_scenario_name_select.options[1]
        self.assertEqual(len(self.scenario_tab.simulation_tile_layout.children), 1)
        self.assertEqual(len(self.scenario_tab.time_series_layout.children), 1)

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change function."""
        new_experiment_file_data = ExperimentFileData(file_paths=[])
        self.scenario_tab.file_paths_on_change(experiment_file_data=new_experiment_file_data, experiment_file_active_index=[])
        self.assertEqual(self.scenario_tab._scalar_scenario_type_select.value, '')
        self.assertEqual(self.scenario_tab._scalar_scenario_type_select.options, [''])
        self.assertEqual(self.scenario_tab._scalar_scenario_name_select.value, '')
        self.assertEqual(self.scenario_tab._scalar_scenario_name_select.options, [])

    def test_update_scenario_legend(self) -> None:
        """Test functions corresponding to legend selection changes work as expected."""
        self.scenario_tab.file_paths_on_change(experiment_file_data=self.experiment_file_data, experiment_file_active_index=[0])
        self.scenario_tab._scalar_scenario_type_select.value = self.scenario_tab._scalar_scenario_type_select.options[1]
        self.scenario_tab._scalar_log_name_select.value = self.scenario_tab._scalar_log_name_select.options[1]
        self.scenario_tab._scalar_scenario_name_select.value = self.scenario_tab._scalar_scenario_name_select.options[1]
        self.scenario_tab._traj_checkbox_group.active = [0]
        self.scenario_tab._map_checkbox_group.active = [0, 1, 2]
        self.scenario_tab._object_checkbox_group.active = [3, 4]

    def test_modal_button_on_click(self) -> None:
        """Test modal button on click function."""
        self.scenario_tab._experiment_file_active_index = [0]
        self.scenario_tab._scalar_scenario_type_select.value = self.scenario_tab._scalar_scenario_type_select.options[1]
        self.scenario_tab._scalar_log_name_select.value = self.scenario_tab._scalar_log_name_select.options[1]
        self.scenario_tab._scalar_scenario_name_select.value = self.scenario_tab._scalar_scenario_name_select.options[1]
        self.scenario_tab._scenario_modal_query_button_on_click()
        self.assertEqual(self.scenario_tab.planner_checkbox_group.labels, ['SimplePlanner'])
        self.assertIn('ego_acceleration_statistics', self.scenario_tab._time_series_data)

    def test_planner_button_on_click(self) -> None:
        """Test checkbox button in planner."""
        self.scenario_tab._experiment_file_active_index = [0]
        self.scenario_tab._scalar_scenario_type_select.value = self.scenario_tab._scalar_scenario_type_select.options[1]
        self.scenario_tab._scalar_log_name_select.value = self.scenario_tab._scalar_log_name_select.options[1]
        self.scenario_tab._scalar_scenario_name_select.value = self.scenario_tab._scalar_scenario_name_select.options[1]
        self.scenario_tab._scenario_modal_query_button_on_click()
        self.scenario_tab.planner_checkbox_group.active = []
        self.assertEqual(len(self.scenario_tab.simulation_tile_layout.children), 1)
        self.assertEqual(len(self.scenario_tab.time_series_layout.children), 1)
        self.scenario_tab.planner_checkbox_group.active = [0]
        self.assertEqual(len(self.scenario_tab.simulation_tile_layout.children), 1)
        self.assertEqual(len(self.scenario_tab.time_series_layout.children), 1)
        with self.assertRaises(IndexError):
            self.scenario_tab.planner_checkbox_group.active = [1]

class TestS3Tab(unittest.TestCase):
    """Test nuboard s3 tab functionality."""

    def setUp(self) -> None:
        """Set up a configuration tab."""
        self.doc = Document()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', aggregator_metric_folder='aggregator_metric', current_path=Path(self.tmp_dir.name))
        metric_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        simulation_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)
        self.nuboard_file_name = Path(self.tmp_dir.name) / ('nuboard_file' + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.experiment_file_data = ExperimentFileData(file_paths=[self.nuboard_file])
        self.histogram_tab = HistogramTab(experiment_file_data=self.experiment_file_data, doc=self.doc)
        self.configuration_tab = ConfigurationTab(experiment_file_data=self.experiment_file_data, doc=self.doc, tabs=[self.histogram_tab])
        if not os.getenv('NUPLAN_EXP_ROOT', None):
            os.environ['NUPLAN_EXP_ROOT'] = self.tmp_dir.name
        self.s3_tab = CloudTab(doc=self.doc, configuration_tab=self.configuration_tab)
        self.dummy_file_result_message = S3FileResultMessage(s3_connection_status=S3ConnectionStatus(success=True, return_message='Connect successfully'), file_contents={'dummy_a': S3FileContent(filename='dummy_a', size=10, last_modified=datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)), 'dummy_b': S3FileContent(filename='dummy_b', size=10, last_modified=datetime(day=3, month=8, year=1992, tzinfo=timezone.utc))})

    def test_modal_query_btn(self) -> None:
        """Test if modal query btn works."""
        self.s3_tab._s3_modal_query_on_click()
        self.assertNotEqual(self.s3_tab._s3_client, None)

    def test_load_s3_contents_with_file_contents(self) -> None:
        """Test _load_s3_contents works if there are file contents."""
        self.s3_tab._load_s3_contents(s3_file_result_message=self.dummy_file_result_message)
        self.s3_tab.s3_error_text.text = self.dummy_file_result_message.s3_connection_status.return_message
        self.assertEqual(self.s3_tab.s3_error_text.text, self.dummy_file_result_message.s3_connection_status.return_message)

    def test_s3_data_source_on_selected(self) -> None:
        """Test _s3_data_source_on_selected work."""
        data_sources: Dict[str, List[Any]] = {'object': [], 'last_modified': [], 'timestamp': [], 'size': []}
        for file_name, content in self.dummy_file_result_message.file_contents.items():
            data_sources['object'].append(file_name)
            data_sources['last_modified'].append(content.last_modified_day if content.last_modified is not None else '')
            data_sources['timestamp'].append(content.date_string if content.date_string is not None else '')
            data_sources['size'].append(content.kb_size() if content.kb_size() is not None else '')
        self.s3_tab.data_table.source.data = data_sources
        self.s3_tab._selected_column.value = str(1)
        self.s3_tab._s3_data_source_on_selected(attr='indices', new=[0], old=[])
        self.assertEqual(self.s3_tab.s3_download_text_input.value, 'dummy_a')
        self.s3_tab._selected_column.value = str(2)
        self.s3_tab._s3_data_source_on_selected(attr='indices', new=[1], old=[])
        self.assertEqual(self.s3_tab.s3_download_text_input.value, 'dummy_b')
        self.s3_tab._selected_column.value = str(0)
        self.s3_tab._s3_data_source_on_selected(attr='indices', new=[0], old=[])
        self.assertNotEqual(self.s3_tab._s3_client, None)

    def test_s3_download_button_on_click(self) -> None:
        """Test if s3 download button on_click function works."""
        self.s3_tab.s3_bucket_name.text = 's3://test-bucket'
        self.s3_tab.s3_download_text_input.value = 'test-prefix'
        self.s3_tab._s3_download_button_on_click()
        self.assertEqual(self.s3_tab.s3_download_button.label, 'Downloading...')
        self.assertTrue(self.s3_tab.s3_download_button.disabled)

    def test_s3_download_prefixes_fail_without_s3_client(self) -> None:
        """Test s3 tab download_prefixes function fails when there is no s3 client."""
        self.s3_tab._s3_download_prefixes()
        self.assertEqual(self.s3_tab.s3_error_text.text, 'No s3 connection!')

    def test_s3_download_prefixes_fail_without_nuboard_files(self) -> None:
        """Test s3 tab download_prefixes function fails when there is no nuboard files."""
        self.s3_tab.s3_bucket_name.text = 's3://test-bucket'
        self.s3_tab.s3_download_text_input.value = 'test-prefix'
        s3_client = boto3.Session().client('s3')
        self.s3_tab._s3_client = s3_client
        stubber = Stubber(s3_client)
        expected_response = {'CommonPrefixes': [{'Prefix': 'dummy_folder_a/log.txt'}, {'Prefix': 'dummy_folder_b/log_2.txt'}], 'Contents': [{'Key': 'dummy_a', 'Size': 15, 'LastModified': datetime(day=2, month=7, year=1992, tzinfo=timezone.utc)}, {'Key': 'dummy_b', 'Size': 45, 'LastModified': datetime(day=6, month=7, year=1992, tzinfo=timezone.utc)}]}
        expected_params = {'Bucket': 'test-bucket', 'Prefix': 'test-prefix/', 'Delimiter': '/'}
        stubber.add_response('list_objects_v2', expected_response, expected_params)
        with stubber:
            self.s3_tab._s3_download_prefixes()
            self.assertEqual(self.s3_tab.s3_error_text.text, 'No available nuboard files in the prefix')

    def test_s3_update_nuboard_file_main_path(self) -> None:
        """Test s3 tab _update_s3_nuboard_file_main_path function updates main path based on the selected prefix."""
        s3_nuboard_file_result_message = S3NuBoardFileResultMessage(s3_connection_status=S3ConnectionStatus(success=True, return_message='Get s3 nuboasrd file'), nuboard_file=self.nuboard_file, nuboard_filename=self.nuboard_file_name.name)
        prefix = self.tmp_dir.name
        self.s3_tab._update_s3_nuboard_file_main_path(s3_nuboard_file_result=s3_nuboard_file_result_message, selected_prefix=prefix)
        nuboard_file = s3_nuboard_file_result_message.nuboard_file
        self.assertEqual(nuboard_file.simulation_main_path, self.tmp_dir.name)
        self.assertEqual(nuboard_file.metric_main_path, self.tmp_dir.name)

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()

class TestOverviewTab(SkeletonTestTab):
    """Test nuboard overview tab functionality."""

    def setUp(self) -> None:
        """Set up an overview tab."""
        super().setUp()
        self.overview_tab = OverviewTab(experiment_file_data=self.experiment_file_data, doc=self.doc)

    def test_update_table(self) -> None:
        """Test update table function."""
        self.overview_tab._overview_on_change()

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change function."""
        new_experiment_file_data = ExperimentFileData(file_paths=[])
        self.overview_tab.file_paths_on_change(experiment_file_data=new_experiment_file_data, experiment_file_active_index=[])

class SkeletonTestTab(unittest.TestCase):
    """Base class for nuBoard tab unit tests."""

    @staticmethod
    def set_up_dummy_simulation(simulation_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy simulation data.
        :param simulation_path: Simulation path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        save_path = simulation_path / planner_name / scenario_type / log_name / scenario_name
        save_path.mkdir(parents=True, exist_ok=True)
        simulation_data = create_sample_simulation_log(save_path / f'{uuid4()}.msgpack.xz')
        simulation_data.save_to_file()

    @staticmethod
    def set_up_dummy_metric(metric_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str) -> None:
        """
        Set up dummy metric results.
        :param metric_path: Metric path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        statistics = [Statistic(name='ego_max_acceleration', unit='meters_per_second_squared', value=2.0, type=MetricStatisticsType.MAX), Statistic(name='ego_min_acceleration', unit='meters_per_second_squared', value=0.0, type=MetricStatisticsType.MIN), Statistic(name='ego_p90_acceleration', unit='meters_per_second_squared', value=1.0, type=MetricStatisticsType.P90), Statistic(name='ego_count_acceleration', unit=MetricStatisticsType.COUNT.unit, value=2, type=MetricStatisticsType.COUNT), Statistic(name='ego_boolean_acceleration', unit=MetricStatisticsType.BOOLEAN.unit, value=True, type=MetricStatisticsType.BOOLEAN)]
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit='meters_per_second_squared', time_stamps=list(time_stamps), values=list(accel))
        result = MetricStatistics(metric_computator='ego_acceleration', name='ego_acceleration_statistics', statistics=statistics, time_series=time_series, metric_category='Dynamic', metric_score=1.0)
        key = MetricFileKey(metric_name='ego_acceleration', log_name=log_name, scenario_name=scenario_name, scenario_type=scenario_type, planner_name=planner_name)
        metric_engine = MetricsEngine(main_save_path=metric_path)
        metric_files = {scenario_name: [MetricFile(key=key, metric_statistics=[result])]}
        metric_engine.write_to_files(metric_files=metric_files)
        metric_file_callback = MetricFileCallback(metric_file_output_path=str(metric_path), scenario_metric_paths=[str(metric_path)])
        metric_file_callback.on_run_simulation_end()

    def setUp(self) -> None:
        """
        Set up common data for nuboard unit tests.
        """
        self.doc = Document()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', aggregator_metric_folder='aggregator_metric', current_path=Path(self.tmp_dir.name))
        log_name = 'dummy_log'
        planner_name = 'SimplePlanner'
        scenario_type = 'Test'
        scenario_name = 'Dummy_scene'
        metric_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_metric(metric_path=metric_path, log_name=log_name, planner_name=planner_name, scenario_name=scenario_name, scenario_type=scenario_type)
        simulation_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_simulation(simulation_path, log_name=log_name, planner_name=planner_name, scenario_type=scenario_type, scenario_name=scenario_name)
        self.nuboard_file_name = Path(self.tmp_dir.name) / ('nuboard_file' + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.experiment_file_data = ExperimentFileData(file_paths=[self.nuboard_file])

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()

class TestHistogramTab(SkeletonTestTab):
    """Test nuboard histogram tab functionality."""

    def setUp(self) -> None:
        """Set up a histogram tab."""
        super().setUp()
        self.histogram_tab = HistogramTab(experiment_file_data=self.experiment_file_data, doc=self.doc)

    def test_update_histograms(self) -> None:
        """Test update_histograms works as expected when we update choices."""
        self.histogram_tab.file_paths_on_change(experiment_file_data=self.experiment_file_data, experiment_file_active_index=[0])
        self.histogram_tab._scenario_type_multi_choice.value = ['Test']
        self.histogram_tab._metric_name_multi_choice.value = ['ego_acceleration_statistics']
        self.histogram_tab._setting_modal_query_button_on_click()
        self.assertIn('ego_acceleration_statistics', self.histogram_tab._aggregated_data)
        self.assertEqual(len(self.histogram_tab.histogram_plots.children), 1)

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change function."""
        new_experiment_file_data = ExperimentFileData(file_paths=[])
        self.histogram_tab.file_paths_on_change(experiment_file_data=new_experiment_file_data, experiment_file_active_index=[])
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.options, ['all'])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.options, [])

class TestConfigurationTab(unittest.TestCase):
    """Test nuboard configuration tab functionality."""

    def setUp(self) -> None:
        """Set up a configuration tab."""
        self.doc = Document()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(simulation_main_path=self.tmp_dir.name, metric_main_path=self.tmp_dir.name, metric_folder='metrics', simulation_folder='simulations', aggregator_metric_folder='aggregator_metric', current_path=Path(self.tmp_dir.name))
        metric_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        simulation_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)
        self.nuboard_file_name = Path(self.tmp_dir.name) / ('nuboard_file' + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.experiment_file_data = ExperimentFileData(file_paths=[self.nuboard_file])
        self.histogram_tab = HistogramTab(experiment_file_data=self.experiment_file_data, doc=self.doc)
        self.configuration_tab = ConfigurationTab(experiment_file_data=self.experiment_file_data, doc=self.doc, tabs=[self.histogram_tab])

    def test_file_path_on_change(self) -> None:
        """Test function when the file path is changed."""
        self.configuration_tab._file_paths = []
        self.configuration_tab._file_paths_on_change()
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.options, ['all'])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.options, [])

    def test_add_experiment_file(self) -> None:
        """Test add experiment file function."""
        attr = 'value'
        old = 'None'
        self.configuration_tab.experiment_file_data.file_paths = []
        self.configuration_tab._add_experiment_file(attr=attr, old=pickle.dumps(old), new=base64.b64encode(pickle.dumps(self.nuboard_file.serialize())))

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()

class MockAbstractScenarioBuilder(AbstractScenarioBuilder):
    """Mock abstract scenario builder class used for testing."""

    def __init__(self, num_scenarios: int=0):
        """
        The init method
        :param num_scenarios: The number of scenarios to return from get_scenarios()
        """
        self.num_scenarios = num_scenarios

    @classmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """Inherited. See superclass."""
        return cast(Type[AbstractScenario], MockAbstractScenario)

    def get_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[AbstractScenario]:
        """Implemented. See interface."""
        return [MockAbstractScenario() for _ in range(self.num_scenarios)]

    def get_map_factory(self) -> AbstractMapFactory:
        """Implemented. See interface."""
        return MockMapFactory()

    @property
    def repartition_strategy(self) -> RepartitionStrategy:
        """Implemented. See interface."""
        return RepartitionStrategy.INLINE

