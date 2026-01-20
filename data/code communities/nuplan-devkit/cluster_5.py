# Cluster 5

def get_lidarpc_sensor_data() -> SensorDataSource:
    """
    Builds the SensorDataSource for a lidar_pc.
    :return: The query parameters for lidar_pc.
    """
    return SensorDataSource('lidar_pc', 'lidar', 'lidar_token', 'MergedPointCloud')

def get_scenario_type_token_map(db_files: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get a map from scenario types to lists of all instances for a given scenario type in the database.
    :param db_files: db files to search for available scenario types.
    :return: dictionary mapping scenario type to list of db/token pairs of that type.
    """
    available_scenario_types = defaultdict(list)
    for db_file in db_files:
        for tag, token in get_lidarpc_tokens_with_scenario_tag_from_db(db_file):
            available_scenario_types[tag].append((db_file, token))
    return available_scenario_types

def get_lidarpc_tokens_with_scenario_tag_from_db(log_file: str) -> Generator[Tuple[str, str], None, None]:
    """
    Get the LidarPc tokens that are tagged with a scenario from the DB, sorted by scenario_type in ascending order.
    :param log_file: The log file to query.
    :return: A generator of (scenario_tag, token) tuples where `token` is tagged with `scenario_tag`
    """
    query = '\n    SELECT  st.type,\n            lp.token\n    FROM lidar_pc AS lp\n    LEFT OUTER JOIN scenario_tag AS st\n        ON lp.token=st.lidar_pc_token\n    WHERE st.type IS NOT NULL\n    ORDER BY st.type ASC NULLS LAST;\n    '
    for row in execute_many(query, (), log_file):
        yield (str(row['type']), row['token'].hex())

def _ensure_file_downloaded(data_root: str, potentially_remote_path: str) -> str:
    """
    Attempts to download the DB file from a remote URL if it does not exist locally.
    If the download fails, an error will be raised.
    :param data_root: The location to download the file, if necessary.
    :param potentially_remote_path: The path to the file.
    :return: The resulting file path. Will be one of a few options:
        * If potentially_remote_path points to a local file, will return potentially_remote_path
        * If potentially_remote_file points to a remote file, it does not exist currently, and the file can be successfully downloaded, it will return the path of the downloaded file.
        * In all other cases, an error will be raised.
    """
    output_file_path: str = download_file_if_necessary(data_root, potentially_remote_path)
    if not os.path.exists(output_file_path):
        raise ValueError(f'{potentially_remote_path} could not be downloaded.')
    return output_file_path

@cli.command()
def info(db_version: str=typer.Argument(NUPLAN_DB_VERSION, help='The database version.'), data_root: str=typer.Option(NUPLAN_DATA_ROOT, help='The root location of the database')) -> None:
    """
    Print out detailed information about the selected database.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    db_description = get_db_description(db_version)
    for table_name, table_description in db_description.tables.items():
        typer.echo(f'Table {table_name}: {table_description.row_count} rows')
        for column_name, column_description in table_description.columns.items():
            typer.echo(''.join([f'\tcolumn {column_name}: {column_description.data_type} ', 'NULL ' if column_description.nullable else 'NOT NULL ', 'PRIMARY KEY ' if column_description.is_primary_key else '']))
        typer.echo()

def get_db_description(log_file: str) -> DbDescription:
    """
    Get information about all tables that are present in the DB.
    :param log_file: The log file to describe.
    :return: A description of the tables present in the DB.
    """
    tables: Dict[str, TableDescription] = {}
    for table_name in _get_table_names_from_db(log_file):
        tables[table_name] = _get_table_description(log_file, table_name)
    return DbDescription(tables=tables)

@cli.command()
def duration(db_version: str=typer.Argument(NUPLAN_DB_VERSION, help='The database version.'), data_root: str=typer.Option(NUPLAN_DATA_ROOT, help='The root location of the database')) -> None:
    """
    Print out the duration of the selected db.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    db_duration_us = get_db_duration_in_us(db_version)
    db_duration_s = float(db_duration_us) / 1000000.0
    db_duration_str = time.strftime('%H:%M:%S', time.gmtime(db_duration_s))
    typer.echo(f'DB duration is {db_duration_str} [HH:MM:SS]')

def get_db_duration_in_us(log_file: str) -> int:
    """
    Get the duration of the database log in us, measured as (last_lidar_pc_timestamp) - (first_lidarpc_timestamp)
    :param log_file: The log file to query.
    :return: The db duration, in microseconds.
    """
    query = '\n    SELECT MAX(timestamp) - MIN(timestamp) AS diff_us\n    FROM lidar_pc;\n    '
    result = execute_one(query, (), log_file)
    return int(result['diff_us'])

@cli.command()
def log_duration(db_version: str=typer.Argument(NUPLAN_DB_VERSION, help='The database version.'), data_root: str=typer.Option(NUPLAN_DATA_ROOT, help='The root location of the database')) -> None:
    """
    Print out the duration of every log in the selected db.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    num_logs = 0
    for log_file_name, log_file_duration_us in get_db_log_duration(db_version):
        log_file_duration_s = float(log_file_duration_us) / 1000000.0
        log_file_duration_str = time.strftime('%H:%M:%S', time.gmtime(log_file_duration_s))
        typer.echo(f'The duration of log {log_file_name} is {log_file_duration_str} [HH:MM:SS]')
        num_logs += 1
    typer.echo(f'There are {num_logs} total logs.')

def get_db_log_duration(log_file: str) -> Generator[Tuple[str, int], None, None]:
    """
    Get the duration of each log present in the database, measured as (last_lidar_pc_timestamp) - (first_lidarpc_timestamp)
    :param log_file: The log file to query.
    :return: A tuple of (log_name, duration) pair, one for each log file present in the DB, sorted by log name.
    """
    query = '\n    SELECT  l.logfile,\n            MAX(lp.timestamp) - MIN(lp.timestamp) AS duration_us\n    FROM log AS l\n    INNER JOIN scene AS s\n        ON s.log_token = l.token\n    INNER JOIN lidar_pc AS lp\n        ON lp.scene_token = s.token\n    GROUP BY l.logfile\n    ORDER BY l.logfile ASC;\n    '
    for row in execute_many(query, (), log_file):
        yield (row['logfile'], row['duration_us'])

@cli.command()
def log_vehicle(db_version: str=typer.Argument(NUPLAN_DB_VERSION, help='The database version.'), data_root: str=typer.Option(NUPLAN_DATA_ROOT, help='The root location of the database')) -> None:
    """
    Print out vehicle information from every log in the selected database.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    for log_file, vehicle_name in get_db_log_vehicles(db_version):
        typer.echo(f'For the log {log_file}, vehicle {vehicle_name} was used.')

def get_db_log_vehicles(log_file: str) -> Generator[Tuple[str, str], None, None]:
    """
    Get the vehicle used for each log file in the DB, sorted by log file name.
    :param log_file: The log file to query.
    :return: A tuple of (log_name, vehicle_name) for each log file in the database.
    """
    query = '\n    SELECT  logfile,\n            vehicle_name\n    FROM log\n    ORDER BY logfile ASC;\n    '
    for row in execute_many(query, (), log_file):
        yield (row['logfile'], row['vehicle_name'])

@cli.command()
def scenarios(db_version: str=typer.Argument(NUPLAN_DB_VERSION, help='The database version.'), data_root: str=typer.Option(NUPLAN_DATA_ROOT, help='The root location of the database')) -> None:
    """
    Print out the available scenarios in the selected db.
    """
    db_version = _ensure_file_downloaded(data_root, db_version)
    total_count = 0
    for tag, num_scenarios in get_db_scenario_info(db_version):
        typer.echo(f'{tag}: {num_scenarios} scenarios.')
        total_count += num_scenarios
    typer.echo(f'TOTAL: {total_count} scenarios.')

def get_db_scenario_info(log_file: str) -> Generator[Tuple[str, int], None, None]:
    """
    Get the scenario types present in the dictionary and the number of occurances, ordered by occurance count.
    :param log_file: The log file to query.
    :return: A generator of (scenario_tag, count) tuples, ordered by count desc.
    """
    query = '\n    SELECT  type,\n            COUNT(*) AS cnt\n    FROM scenario_tag\n    GROUP BY type\n    ORDER BY cnt DESC;\n    '
    for row in execute_many(query, (), log_file):
        yield (row['type'], row['cnt'])

class TestNuPlanCli(unittest.TestCase):
    """
    Test nuplan cli with typer engine
    """

    def _get_ensure_file_downloaded_patch(self, expected_data_root: str, expected_remote_path: str) -> Callable[[str, str], str]:
        """
        Get the patch for ensure_file_downloaded.
        """

        def fxn(actual_data_root: str, actual_remote_path: str) -> str:
            """
            The patch for ensure_file_downloaded.
            """
            self.assertEqual(expected_data_root, actual_data_root)
            self.assertEqual(expected_remote_path, actual_remote_path)
            return actual_remote_path
        return fxn

    def test_db_info_info(self) -> None:
        """
        Test nuplan_cli.py db info command.
        """

        def _patch_get_db_description(log_name: str) -> DbDescription:
            """
            A patch for the get_db_description db function.
            """
            self.assertEqual('expected_log_name', log_name)
            return DbDescription(tables={'first_table': TableDescription(name='first_table', row_count=123, columns={'first_token': ColumnDescription(column_id=0, name='first_token', data_type='blob', nullable=False, is_primary_key=True), 'first_something': ColumnDescription(column_id=1, name='first_something', data_type='varchar(64)', nullable=True, is_primary_key=False)}), 'second_table': TableDescription(name='second_table', row_count=456, columns={'second_token': ColumnDescription(column_id=0, name='token', data_type='blob', nullable=False, is_primary_key=True), 'second_something': ColumnDescription(column_id=1, name='somthing', data_type='varchar(128)', nullable=True, is_primary_key=False)})})
        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch('/data/sets/nuplan', 'expected_log_name')
        with mock.patch('nuplan.cli.db_cli.get_db_description', _patch_get_db_description), mock.patch('nuplan.cli.db_cli._ensure_file_downloaded', ensure_file_downloaded_patch):
            result = runner.invoke(cli, ['db', 'info', 'expected_log_name'])
            self.assertEqual(0, result.exit_code)
            strings_of_interest = ['table first_table: 123 rows', 'table second_table: 456 rows', 'column first_token: blob not null primary key', 'column first_something: varchar(64) null', 'column second_token: blob not null primary key', 'column second_something: varchar(128) null']
            result_stdout = result.stdout.lower()
            for string_of_interest in strings_of_interest:
                self.assertTrue(string_of_interest in result_stdout)

    def test_db_cli_duration(self) -> None:
        """
        Test nuplan_cli.py db duration command.
        """

        def _patch_db_duration(log_name: str) -> int:
            """
            A patch for the get_db_duration function.
            """
            self.assertEqual('expected_log_name', log_name)
            return int(125 * 1000000.0)
        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch('/data/sets/nuplan', 'expected_log_name')
        with mock.patch('nuplan.cli.db_cli.get_db_duration_in_us', _patch_db_duration), mock.patch('nuplan.cli.db_cli._ensure_file_downloaded', ensure_file_downloaded_patch):
            result = runner.invoke(cli, ['db', 'duration', 'expected_log_name'])
            self.assertEqual(0, result.exit_code)
            self.assertTrue('00:02:05' in result.stdout)

    def test_db_cli_log_duration(self) -> None:
        """
        Test nuplan_cli.py db log-duration command.
        """

        def _patch_db_log_duration(log_name: str) -> Generator[Tuple[str, int], None, None]:
            """
            Patch for get_db_log_duration function.
            """
            self.assertEqual('expected_log_name', log_name)
            for i in range(0, 3, 1):
                yield (f'log_file_{i}', int((i + 1) * 67 * 1000000.0))
        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch('/data/sets/nuplan', 'expected_log_name')
        with mock.patch('nuplan.cli.db_cli.get_db_log_duration', _patch_db_log_duration), mock.patch('nuplan.cli.db_cli._ensure_file_downloaded', ensure_file_downloaded_patch):
            result = runner.invoke(cli, ['db', 'log-duration', 'expected_log_name'])
            self.assertEqual(0, result.exit_code)
            strings_of_interest = ['log_file_0 is 00:01:07', 'log_file_1 is 00:02:14', 'log_file_2 is 00:03:21', '3 total logs']
            for string_of_interest in strings_of_interest:
                self.assertTrue(string_of_interest in result.stdout)

    def test_db_cli_log_vehicle(self) -> None:
        """
        Test nuplan_cli.py log-vehicle command.
        """

        def _patch_db_log_vehicles(log_name: str) -> Generator[Tuple[str, str], None, None]:
            """
            Patch for get_db_log_vehicles function.
            """
            self.assertEqual('expected_log_name', log_name)
            for i in range(0, 3, 1):
                yield (f'log_file_{i}', f'vehicle_{i}')
        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch('/data/sets/nuplan', 'expected_log_name')
        with mock.patch('nuplan.cli.db_cli.get_db_log_vehicles', _patch_db_log_vehicles), mock.patch('nuplan.cli.db_cli._ensure_file_downloaded', ensure_file_downloaded_patch):
            result = runner.invoke(cli, ['db', 'log-vehicle', 'expected_log_name'])
            self.assertEqual(0, result.exit_code)
            for i in range(0, 3, 1):
                self.assertTrue(f'log_file_{i}, vehicle vehicle_{i}' in result.stdout)

    def test_db_cli_scenarios(self) -> None:
        """
        Test db_cli scenarios command.
        """

        def _patch_db_scenario_info(log_name: str) -> Generator[Tuple[str, int], None, None]:
            """
            Patch for get_db_scenario_info
            """
            self.assertEqual('expected_log_name', log_name)
            for i in range(0, 3, 1):
                yield (f'scenario_{i}', i + 5)
        ensure_file_downloaded_patch = self._get_ensure_file_downloaded_patch('/data/sets/nuplan', 'expected_log_name')
        with mock.patch('nuplan.cli.db_cli.get_db_scenario_info', _patch_db_scenario_info), mock.patch('nuplan.cli.db_cli._ensure_file_downloaded', ensure_file_downloaded_patch):
            result = runner.invoke(cli, ['db', 'scenarios', 'expected_log_name'])
            self.assertEqual(0, result.exit_code)
            strings_of_interest = ['scenario_0: 5', 'scenario_1: 6', 'scenario_2: 7', 'TOTAL: 18']
            for string_of_interest in strings_of_interest:
                self.assertTrue(string_of_interest in result.stdout)

class TestImageMock(unittest.TestCase):
    """Test suite for the Image class using mocks."""
    TEST_PATH = 'nuplan.database.utils.image'

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self.mock_pil_img = MagicMock(PilImg.Image)
        self.image = Image(self.mock_pil_img)

    def test_as_pil(self) -> None:
        """Test the function as_pil."""
        img = self.image.as_pil
        self.assertEqual(self.mock_pil_img, img)

    @patch(f'{TEST_PATH}.np.array', autospec=True)
    def test_as_numpy_nocache(self, mock_array: Mock) -> None:
        """Test the function as_numpy_nocache."""
        _ = self.image.as_numpy_nocache()
        mock_array.assert_called_with(self.mock_pil_img, dtype=np.uint8)

    @patch(f'{TEST_PATH}.Image.as_numpy_nocache', autospec=True)
    def test_as_numpy(self, mock_as_numpy_nocache: Mock) -> None:
        """Test the function as_numpy_nocache."""
        _ = self.image.as_numpy
        mock_as_numpy_nocache.assert_called_once()

    @patch(f'{TEST_PATH}.cv2.cvtColor', autospec=True)
    @patch(f'{TEST_PATH}.np.array', autospec=True)
    def test_as_cv2_nocache(self, mock_array: Mock, mock_cvtcolor: Mock) -> None:
        """Test the function as_cv2_nocache."""
        _ = self.image.as_cv2_nocache()
        mock_cvtcolor.assert_called_with(mock_array(self.mock_pil_img, np.uint8), cv2.COLOR_RGB2BGR)

    @patch(f'{TEST_PATH}.Image.as_cv2_nocache', autospec=True)
    def test_as_cv2(self, mock_as_cv2_nocache: Mock) -> None:
        """Test the function as_numpy_nocache."""
        _ = self.image.as_cv2
        mock_as_cv2_nocache.assert_called_once()

class TestImage(unittest.TestCase):
    """Test suite for the Image class using synthetic image."""

    def setUp(self) -> None:
        """Inherited, see superclass"""
        pil_img: PilImg.Image = PilImg.new('RGB', (500, 500))
        self.image = Image(pil_img)

    def _test_numpy_type(self, img: Any) -> None:
        """
        Checks if the given object is a numpy array with dtype uint8.
        :param img: The image object to test. Type hint any because the test should be valid for all objects.
        """
        self.assertEqual(np.ndarray, type(img))
        self.assertEqual(np.uint8, img.dtype)
        self.assertNotEqual(np.float64, img.dtype)

    def test_as_pil(self) -> None:
        """Test the function as_pil."""
        img = self.image.as_pil
        self.assertEqual(PilImg.Image, type(img))

    def test_as_numpy_nocache(self) -> None:
        """Test the function as_numpy_nocache."""
        img = self.image.as_numpy_nocache()
        self._test_numpy_type(img)

    def test_as_numpy(self) -> None:
        """Test the function as_numpy_nocache."""
        img = self.image.as_numpy
        self._test_numpy_type(img)

    def test_as_cv2_nocache(self) -> None:
        """Test the function as_cv2_nocache."""
        img = self.image.as_cv2_nocache()
        self._test_numpy_type(img)

    def test_as_cv2(self) -> None:
        """Test the function as_numpy_nocache."""
        img = self.image.as_cv2
        self._test_numpy_type(img)

def _get_table_columns_from_db(log_file: str, table_name: str) -> Generator[ColumnDescription, None, None]:
    """
    Get information about the columns that are present in the table.
    If the table does not exist, returns an empty generator.
    :param log_file: The log file to query.
    :param table_name: The table name to query.
    :return: A generator containing information about the columns in the table, ordered by column_id ascending.
    """
    query = f'\n    PRAGMA table_info({table_name});\n    '
    for row in execute_many(query, (), log_file):
        yield ColumnDescription(column_id=row['cid'], name=row['name'], data_type=row['type'], nullable=not row['notnull'], is_primary_key=row['pk'])

def execute_many(query_text: str, query_parameters: Any, db_file: str) -> Generator[sqlite3.Row, None, None]:
    """
    Runs a query with the provided arguments on a specified Sqlite DB file.
    This query can return any number of rows.
    :param query_text: The query to run.
    :param query_parameters: The parameters to provide to the query.
    :param db_file: The DB file on which to run the query.
    :return: A generator of rows emitted from the query.
    """
    connection = sqlite3.connect(db_file)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    try:
        cursor.execute(query_text, query_parameters)
        for row in cursor:
            yield row
    finally:
        cursor.close()
        connection.close()

def _get_table_row_count_from_db(log_file: str, table_name: str) -> int:
    """
    Get the number of rows in a table.
    Raises an error if the table does not exist.
    :param log_file: The log file to query.
    :param table_name: The table name to examine.
    :return: The number of rows in the table.
    """
    query = f'\n    SELECT COUNT(*) AS cnt\n    FROM {table_name};\n    '
    result = execute_one(query, (), log_file)
    if result is None:
        raise ValueError(f'Table {table_name} does not exist.')
    return int(result['cnt'])

def execute_one(query_text: str, query_parameters: Any, db_file: str) -> Optional[sqlite3.Row]:
    """
    Runs a query with the provided arguments on a specified Sqlite DB file.
    Validates that the query returns at most one row.
    :param query_text: The query to run.
    :param query_parameters: The parameters to provide to the query.
    :param db_file: The DB file on which to run the query.
    :return: The returned row, if it exists. None otherwise.
    """
    connection = sqlite3.connect(db_file)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    try:
        cursor.execute(query_text, query_parameters)
        result: Optional[sqlite3.Row] = cursor.fetchone()
        if result is not None and cursor.fetchone() is not None:
            raise RuntimeError('execute_one query returned multiple rows.')
        return result
    finally:
        cursor.close()
        connection.close()

def _get_table_description(log_file: str, table_name: str) -> TableDescription:
    """
    Get a description of the table.
    :param log_file: The log file to query.
    :param table_name: The table name to examine.
    :return: A struct filled with information about the table.
    """
    return TableDescription(name=table_name, columns={tc.name: tc for tc in _get_table_columns_from_db(log_file, table_name)}, row_count=_get_table_row_count_from_db(log_file, table_name))

def _get_table_names_from_db(log_file: str) -> Generator[str, None, None]:
    """
    Get the names of tables in the DB.
    :param log_file: The log file to examine.
    :return: A generator containing the table names.
    """
    query = "\n    SELECT tbl_name\n    FROM sqlite_schema\n    WHERE type='table'\n    ORDER BY tbl_name ASC;\n    "
    for row in execute_many(query, (), log_file):
        yield row['tbl_name']

def get_end_sensor_time_from_db(log_file: str, sensor_source: SensorDataSource) -> int:
    """
    Get the timestamp of the last sensor data recorded in the log file.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :return: The timestamp of the last sensor data.
    """
    query = f'\n    SELECT MAX(timestamp) AS max_time\n    FROM {sensor_source.table};\n    '
    result = execute_one(query, [], log_file)
    return int(result['max_time'])

def get_sampled_sensor_tokens_in_time_window_from_db(log_file: str, sensor_source: SensorDataSource, start_timestamp: int, end_timestamp: int, subsample_interval: int) -> Generator[str, None, None]:
    """
    For every token in a window defined by [start_timestamp, end_timestamp], retrieve every `subsample_interval`-th sensor token, ordered in increasing order by timestamp.

    E.g. for this table
    ```
    token | timestamp
    -----------------
    1     | 0
    2     | 1
    3     | 2
    4     | 3
    5     | 4
    6     | 5
    7     | 6
    ```

    query with start_timestamp=1, end_timestamp=5, subsample_interval=2, table=lidar_pc, will return tokens
    [1, 3, 5].

    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param start_timestamp: The start of the window to sample, inclusive.
    :param end_timestamp: The end of the window to sample, inclusive.
    :param subsample_interval: The interval at which to sample.
    :return: A generator of lidar_pc tokens that fit the provided parameters.
    """
    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)
    query = f'\n    WITH numbered AS\n    (\n        SELECT token, timestamp, ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS row_num\n        FROM {sensor_source.table}\n        WHERE timestamp >= ?\n        AND timestamp <= ?\n        AND {sensor_source.sensor_token_column} == ?\n    )\n    SELECT token\n    FROM numbered\n    WHERE ((row_num - 1) % ?) = 0\n    ORDER BY timestamp ASC;\n    '
    for row in execute_many(query, (start_timestamp, end_timestamp, bytearray.fromhex(sensor_token), subsample_interval), log_file):
        yield row['token'].hex()

def get_sensor_data_from_sensor_data_tokens_from_db(log_file: str, sensor_source: SensorDataSource, sensor_class: Type[SensorDataTableRow], tokens: Union[Generator[str, None, None], List[str]]) -> Generator[SensorDataTableRow, None, None]:
    """
    Given a collection of sensor tokens, builds the corresponding sensor_class objects.
    This function makes no restrictions on the ordering of returned values.
    :param sensor_source: Parameters for querying the correct table.
    :param sensor_class: Class holding a row of the SensorData table.
    :param log_file: The db file to query.
    :param tokens: The tokens for which to build the sensor_class objects.
    :return: A generator yielding sensor_class objects.
    """
    if not isinstance(tokens, list):
        tokens = list(tokens)
    query = f'\n        SELECT *\n        FROM {sensor_source.table}\n        WHERE token IN ({('?,' * len(tokens))[:-1]});\n    '
    for row in execute_many(query, [bytearray.fromhex(t) for t in tokens], log_file):
        yield sensor_class.from_db_row(row)

def get_sensor_transform_matrix_for_sensor_data_token_from_db(log_file: str, sensor_source: SensorDataSource, sensor_data_token: str) -> Optional[Transform]:
    """
    Get the associated lidar transform matrix from the DB for the given lidarpc_token.
    :param log_file: The log file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param sensor_data_token: The sensor data token to query.
    :return: The transform matrix. Reuturns None if the matrix does not exist in the DB (e.g. for a token that does not exist).
    """
    query = f'\n        SELECT  sensor.translation,\n                sensor.rotation\n        FROM {sensor_source.sensor_table} AS sensor\n        INNER JOIN {sensor_source.table} AS sensor_data\n            ON sensor_data.{sensor_source.sensor_token_column} = sensor.token\n        WHERE sensor_data.token = ?;\n    '
    row = execute_one(query, (bytearray.fromhex(sensor_data_token),), log_file)
    if row is None:
        return None
    translation = pickle.loads(row['translation'])
    rotation = pickle.loads(row['rotation'])
    output = Quaternion(rotation).transformation_matrix
    output[:3, 3] = np.array(translation)
    return output

def get_mission_goal_for_sensor_data_token_from_db(log_file: str, sensor_source: SensorDataSource, token: str) -> Optional[StateSE2]:
    """
    Get the goal pose for a given lidar_pc token.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param token: The token for which to query the goal state.
    :return: The goal state.
    """
    query = f'\n        SELECT  ep.x,\n                ep.y,\n                ep.qw,\n                ep.qx,\n                ep.qy,\n                ep.qz\n        FROM ego_pose AS ep\n        INNER JOIN scene AS s\n            ON s.goal_ego_pose_token = ep.token\n        INNER JOIN {sensor_source.table} AS sensor_data\n            ON sensor_data.scene_token = s.token\n        WHERE sensor_data.token = ?\n    '
    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None
    q = Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])
    return StateSE2(row['x'], row['y'], q.yaw_pitch_roll[0])

def get_roadblock_ids_for_lidarpc_token_from_db(log_file: str, lidarpc_token: str) -> Optional[List[str]]:
    """
    Get the scene roadblock ids from the db for a given lidar_pc token.
    :param log_file: The db file to query.
    :param lidarpc_token: The token for which to query the current state.
    :return: List of roadblock ids as str.
    """
    query = '\n        SELECT  s.roadblock_ids\n        FROM scene AS s\n        INNER JOIN lidar_pc AS lp\n            ON lp.scene_token = s.token\n        WHERE lp.token = ?\n    '
    row = execute_one(query, (bytearray.fromhex(lidarpc_token),), log_file)
    if row is None:
        return None
    return str(row['roadblock_ids']).split(' ')

def get_statese2_for_lidarpc_token_from_db(log_file: str, token: str) -> Optional[StateSE2]:
    """
    Get the ego pose as a StateSE2 from the db for a given lidar_pc token.
    :param log_file: The db file to query.
    :param token: The token for which to query the current state.
    :return: The current ego state, as a StateSE2 object.
    """
    query = '\n        SELECT  ep.x,\n                ep.y,\n                ep.qw,\n                ep.qx,\n                ep.qy,\n                ep.qz\n        FROM ego_pose AS ep\n        INNER JOIN lidar_pc AS lp\n            ON lp.ego_pose_token = ep.token\n        WHERE lp.token = ?\n    '
    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None
    q = Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])
    return StateSE2(row['x'], row['y'], q.yaw_pitch_roll[0])

def get_sampled_lidarpcs_from_db(log_file: str, initial_token: str, sensor_source: SensorDataSource, sample_indexes: Union[Generator[int, None, None], List[int]], future: bool) -> Generator[LidarPc, None, None]:
    """
    Given an anchor token, return the tokens of either the previous or future tokens, sampled by the provided indexes.

    The result is always sorted by timestamp ascending.

    For example, given the following table:
    token | timestamp
    -----------------
    0     | 0
    1     | 1
    2     | 2
    3     | 3
    4     | 4
    5     | 5
    6     | 6
    7     | 7
    8     | 8
    9     | 9
    10    | 10

    Some sample results:
    initial token | sample_indexes | future | returned tokens
    ---------------------------------------------------------
    5             | [0, 1, 2]      | True   | [5, 6, 7]
    5             | [0, 1, 2]      | False  | [3, 4, 5]
    7             | [0, 3, 12]     | False  | [4, 7]
    0             | [11]           | True   | []

    :param log_file: The db file to query.
    :param initial_token: The token on which to base the query.
    :param sensor_source: Parameters for querying the correct table.
    :param sample_indexes: The indexes for which to sample.
    :param future: If true, the indexes represent future times. If false, they represent previous times.
    :return: A generator of LidarPC objects representing the requested indexes
    """
    if not isinstance(sample_indexes, list):
        sample_indexes = list(sample_indexes)
    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)
    order_direction = 'ASC' if future else 'DESC'
    order_cmp = '>=' if future else '<='
    query = f'\n        WITH initial_lidarpc AS\n        (\n            SELECT token, timestamp\n            FROM lidar_pc\n            WHERE token = ?\n        ),\n        ordered AS\n        (\n            SELECT  lp.token,\n                    lp.next_token,\n                    lp.prev_token,\n                    lp.ego_pose_token,\n                    lp.lidar_token,\n                    lp.scene_token,\n                    lp.filename,\n                    lp.timestamp,\n                    ROW_NUMBER() OVER (ORDER BY lp.timestamp {order_direction}) AS row_num\n            FROM lidar_pc AS lp\n            CROSS JOIN initial_lidarpc AS il\n            WHERE   lp.timestamp {order_cmp} il.timestamp\n            AND lp.lidar_token = ?\n        )\n        SELECT  token,\n                next_token,\n                prev_token,\n                ego_pose_token,\n                lidar_token,\n                scene_token,\n                filename,\n                timestamp\n        FROM ordered\n\n        -- ROW_NUMBER() starts at 1, where consumers will expect sample_indexes to be 0-indexed\n        WHERE (row_num - 1) IN ({('?,' * len(sample_indexes))[:-1]})\n\n        ORDER BY timestamp ASC;\n    '
    args = [bytearray.fromhex(initial_token), bytearray.fromhex(sensor_token)] + sample_indexes
    for row in execute_many(query, args, log_file):
        yield LidarPc.from_db_row(row)

def get_sampled_ego_states_from_db(log_file: str, initial_token: str, sensor_source: SensorDataSource, sample_indexes: Union[Generator[int, None, None], List[int]], future: bool) -> Generator[EgoState, None, None]:
    """
    Given an anchor token, retrieve the ego states associated with tokens order by time, sampled by the provided indexes.

    The result is always sorted by timestamp ascending.

    For example, given the following table:
    token | timestamp | ego_state
    -----------------------------
    0     | 0         | A
    1     | 1         | B
    2     | 2         | C
    3     | 3         | D
    4     | 4         | E
    5     | 5         | F
    6     | 6         | G
    7     | 7         | H
    8     | 8         | I
    9     | 9         | J
    10    | 10        | K

    Some sample results:
    initial token | sample_indexes | future | returned states
    ---------------------------------------------------------
    5             | [0, 1, 2]      | True   | [F, G, H]
    5             | [0, 1, 2]      | False  | [D, E, F]
    7             | [0, 3, 12]     | False  | [E, H]
    0             | [11]           | True   | []

    :param log_file: The db file to query.
    :param initial_token: The token on which to base the query.
    :param sample_indexes: The indexes for which to sample.
    :param future: If true, the indexes represent future times. If false, they represent previous times.
    :return: A generator of EgoState objects associated with the given LidarPCs.
    """
    if not isinstance(sample_indexes, list):
        sample_indexes = list(sample_indexes)
    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)
    order_direction = 'ASC' if future else 'DESC'
    order_cmp = '>=' if future else '<='
    query = f'\n        WITH initial_lidarpc AS\n        (\n            SELECT token, timestamp\n            FROM lidar_pc\n            WHERE token = ?\n        ),\n        ordered AS\n        (\n            SELECT  lp.token,\n                    lp.next_token,\n                    lp.prev_token,\n                    lp.ego_pose_token,\n                    lp.lidar_token,\n                    lp.scene_token,\n                    lp.filename,\n                    lp.timestamp,\n                    ROW_NUMBER() OVER (ORDER BY lp.timestamp {order_direction}) AS row_num\n            FROM lidar_pc AS lp\n            CROSS JOIN initial_lidarpc AS il\n            WHERE   lp.timestamp {order_cmp} il.timestamp\n            AND lidar_token = ?\n        )\n        SELECT  ep.x,\n                ep.y,\n                ep.qw,\n                ep.qx,\n                ep.qy,\n                ep.qz,\n                -- ego_pose and lidar_pc timestamps are not the same, even when linked by token!\n                -- use the lidar_pc timestamp for compatibility with older code.\n                o.timestamp,\n                ep.vx,\n                ep.vy,\n                ep.acceleration_x,\n                ep.acceleration_y\n        FROM ego_pose AS ep\n        INNER JOIN ordered AS o\n            ON o.ego_pose_token = ep.token\n\n        -- ROW_NUMBER() starts at 1, where consumers will expect sample_indexes to be 0-indexed\n        WHERE (o.row_num - 1) IN ({('?,' * len(sample_indexes))[:-1]})\n\n        ORDER BY o.timestamp ASC;\n    '
    args = [bytearray.fromhex(initial_token), bytearray.fromhex(sensor_token)] + sample_indexes
    for row in execute_many(query, args, log_file):
        q = Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])
        yield EgoState.build_from_rear_axle(StateSE2(row['x'], row['y'], q.yaw_pitch_roll[0]), tire_steering_angle=0.0, vehicle_parameters=get_pacifica_parameters(), time_point=TimePoint(row['timestamp']), rear_axle_velocity_2d=StateVector2D(row['vx'], y=row['vy']), rear_axle_acceleration_2d=StateVector2D(x=row['acceleration_x'], y=row['acceleration_y']))

def get_ego_state_for_lidarpc_token_from_db(log_file: str, token: str) -> EgoState:
    """
    Get the ego state associated with an individual lidar_pc token from the db.

    :param log_file: The log file to query.
    :param token: The lidar_pc token to query.
    :return: The EgoState associated with the LidarPC.
    """
    query = '\n        SELECT  ep.x,\n                ep.y,\n                ep.qw,\n                ep.qx,\n                ep.qy,\n                ep.qz,\n                -- ego_pose and lidar_pc timestamps are not the same, even when linked by token!\n                -- use lidar_pc timestamp for backwards compatibility.\n                lp.timestamp,\n                ep.vx,\n                ep.vy,\n                ep.acceleration_x,\n                ep.acceleration_y\n        FROM ego_pose AS ep\n        INNER JOIN lidar_pc AS lp\n            ON lp.ego_pose_token = ep.token\n        WHERE lp.token = ?\n    '
    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None
    q = Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])
    return EgoState.build_from_rear_axle(StateSE2(row['x'], row['y'], q.yaw_pitch_roll[0]), tire_steering_angle=0.0, vehicle_parameters=get_pacifica_parameters(), time_point=TimePoint(row['timestamp']), rear_axle_velocity_2d=StateVector2D(row['vx'], y=row['vy']), rear_axle_acceleration_2d=StateVector2D(x=row['acceleration_x'], y=row['acceleration_y']))

def get_traffic_light_status_for_lidarpc_token_from_db(log_file: str, token: str) -> Generator[TrafficLightStatusData, None, None]:
    """
    Get the traffic light information associated with a given lidar_pc.
    :param log_file: The log file to query.
    :param token: The lidar_pc token for which to obtain the traffic light information.
    :return: The traffic light status data associated with the given lidar_pc.
    """
    query = '\n        SELECT  CASE WHEN tl.status == "green" THEN 0\n                     WHEN tl.status == "yellow" THEN 1\n                     WHEN tl.status == "red" THEN 2\n                     ELSE 3\n                END AS status,\n                tl.lane_connector_id,\n                lp.timestamp AS timestamp\n        FROM lidar_pc AS lp\n        INNER JOIN traffic_light_status AS tl\n            ON lp.token = tl.lidar_pc_token\n        WHERE lp.token = ?\n    '
    for row in execute_many(query, (bytearray.fromhex(token),), log_file):
        yield TrafficLightStatusData(status=TrafficLightStatusType(row['status']), lane_connector_id=row['lane_connector_id'], timestamp=row['timestamp'])

def get_tracked_objects_within_time_interval_from_db(log_file: str, start_timestamp: int, end_timestamp: int, filter_track_tokens: Optional[Set[str]]=None) -> Generator[TrackedObject, None, None]:
    """
    Gets all of the tracked objects between the provided timestamps, inclusive.
    Optionally filters on a user-provided set of track tokens.

    This query will not obtain the future waypoints.
    For that, call `get_future_waypoints_for_agents_from_db()`
    with the tokens of the agents of interest.

    :param log_file: The log file to query.
    :param start_timestamp: The starting timestamp for which to query, in uS.
    :param end_timestamp: The ending timestamp for which to query, in uS.
    :param filter_track_tokens: If provided, only agents with `track_tokens` present in the provided set will be returned.
      If not provided, then all agents present at every time stamp will be returned.
    :return: A generator of TrackedObjects, sorted by TimeStamp, then TrackedObject.
    """
    args: List[Union[int, bytearray]] = [start_timestamp, end_timestamp]
    filter_clause = ''
    if filter_track_tokens is not None:
        filter_clause = "\n            AND lb.track_token IN ({('?,'*len(filter_track_tokens))[:-1]})\n        "
        for token in filter_track_tokens:
            args.append(bytearray.fromhex(token))
    query = f'\n        SELECT  c.name AS category_name,\n                lb.x,\n                lb.y,\n                lb.z,\n                lb.yaw,\n                lb.width,\n                lb.length,\n                lb.height,\n                lb.vx,\n                lb.vy,\n                lb.token,\n                lb.track_token,\n                lp.timestamp\n        FROM lidar_box AS lb\n        INNER JOIN track AS t\n            ON t.token = lb.track_token\n        INNER JOIN category AS c\n            ON c.token = t.category_token\n        INNER JOIN lidar_pc AS lp\n            ON lp.token = lb.lidar_pc_token\n        WHERE lp.timestamp >= ?\n            AND lp.timestamp <= ?\n            {filter_clause}\n        ORDER BY lp.timestamp ASC, lb.track_token ASC;\n    '
    for row in execute_many(query, args, log_file):
        yield _parse_tracked_object_row(row)

def get_tracked_objects_for_lidarpc_token_from_db(log_file: str, token: str) -> Generator[TrackedObject, None, None]:
    """
    Get all tracked objects for a given lidar_pc.
    This includes both agents and static objects.
    The values are returned in random order.

    For agents, this query will not obtain the future waypoints.
    For that, call `get_future_waypoints_for_agents_from_db()`
        with the tokens of the agents of interest.

    :param log_file: The log file to query.
    :param token: The lidar_pc token for which to obtain the objects.
    :return: The tracked objects associated with the token.
    """
    query = '\n        SELECT  c.name AS category_name,\n                lb.x,\n                lb.y,\n                lb.z,\n                lb.yaw,\n                lb.width,\n                lb.length,\n                lb.height,\n                lb.vx,\n                lb.vy,\n                lb.token,\n                lb.track_token,\n                lp.timestamp\n        FROM lidar_box AS lb\n        INNER JOIN track AS t\n            ON t.token = lb.track_token\n        INNER JOIN category AS c\n            ON c.token = t.category_token\n        INNER JOIN lidar_pc AS lp\n            ON lp.token = lb.lidar_pc_token\n        WHERE lp.token = ?\n    '
    for row in execute_many(query, (bytearray.fromhex(token),), log_file):
        yield _parse_tracked_object_row(row)

def get_scenarios_from_db(log_file: str, filter_tokens: Optional[List[str]], filter_types: Optional[List[str]], filter_map_names: Optional[List[str]], include_invalid_mission_goals: bool=True, include_cameras: bool=False) -> Generator[sqlite3.Row, None, None]:
    """
    Get the scenarios present in the db file that match the specified filter criteria.
    If a filter is None, then it will be elided from the query.
    Results are sorted by timestamp ascending
    :param log_file: The log file to query.
    :param filter_tokens: If provided, the set of allowable tokens to return.
    :param filter_types: If provided, the set of allowable scenario types to return.
    :param filter_map_names: If provided, the set of allowable map names to return.
    :param include_cameras: If true, filter for lidar_pcs that has corresponding images.
    :param include_invalid_mission_goals: If true, then scenarios without a valid mission goal will be included
        (i.e. get_mission_goal_for_sensor_data_token_from_db(token) returns None)
        If False, then these scenarios will be filtered.
    :sensor_data_source: Table specification for data sourcing.
    :return: A sqlite3.Row object with the following fields:
        * token: The initial lidar_pc token of the scenario.
        * timestamp: The timestamp of the initial lidar_pc of the scenario.
        * map_name: The map name from which the scenario came.
        * scenario_type: One of the mapped scenario types for the scenario.
            This can be None if there are no matching rows in scenario_types table.
            If there are multiple matches, then one is selected from the set of allowable filter clauses at random.
    """
    filter_clauses = []
    args: List[Union[str, bytearray]] = []
    if filter_types is not None:
        filter_clauses.append(f'\n        st.type IN ({('?,' * len(filter_types))[:-1]})\n        ')
        args += filter_types
    if filter_tokens is not None:
        filter_clauses.append(f'\n        lp.token IN ({('?,' * len(filter_tokens))[:-1]})\n        ')
        args += [bytearray.fromhex(t) for t in filter_tokens]
    if filter_map_names is not None:
        filter_clauses.append(f'\n        l.map_version IN ({('?,' * len(filter_map_names))[:-1]})\n        ')
        args += filter_map_names
    if len(filter_clauses) > 0:
        filter_clause = 'WHERE ' + ' AND '.join(filter_clauses)
    else:
        filter_clause = ''
    if include_invalid_mission_goals:
        invalid_goals_joins = ''
    else:
        invalid_goals_joins = '\n        ---Join on ego_pose to filter scenarios that do not have a valid mission goal\n        INNER JOIN scene AS invalid_goal_scene\n            ON invalid_goal_scene.token = lp.scene_token\n        INNER JOIN ego_pose AS invalid_goal_ego_pose\n            ON invalid_goal_scene.goal_ego_pose_token = invalid_goal_ego_pose.token\n        '
    if include_cameras:
        matching_camera_clause = '\n        INNER JOIN image AS img\n            ON img.ego_pose_token = lp.ego_pose_token\n        '
    else:
        matching_camera_clause = ''
    query = f'\n        WITH ordered_scenes AS\n        (\n            SELECT  token,\n                    ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num\n            FROM scene\n        ),\n        num_scenes AS\n        (\n            SELECT  COUNT(*) AS cnt\n            FROM scene\n        ),\n        valid_scenes AS\n        (\n            SELECT  o.token\n            FROM ordered_scenes AS o\n            CROSS JOIN num_scenes AS n\n\n            -- Define "valid" scenes as those that have at least 2 before and 2 after\n            -- Note that the token denotes the beginning of a scene\n            WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1\n        )\n        SELECT  lp.token,\n                lp.timestamp,\n                l.map_version AS map_name,\n\n                -- scenarios can have multiple tags\n                -- Pick one arbitrarily from the list of acceptable tags\n                MAX(st.type) AS scenario_type\n        FROM lidar_pc AS lp\n        LEFT OUTER JOIN scenario_tag AS st\n            ON lp.token = st.lidar_pc_token\n        INNER JOIN lidar AS ld\n            ON ld.token = lp.lidar_token\n        INNER JOIN log AS l\n            ON ld.log_token = l.token\n        INNER JOIN valid_scenes AS vs\n            ON lp.scene_token = vs.token\n        {matching_camera_clause}\n        {invalid_goals_joins}\n        {filter_clause}\n        GROUP BY    lp.token,\n                    lp.timestamp,\n                    l.map_version\n        ORDER BY lp.timestamp ASC;\n    '
    for row in execute_many(query, args, log_file):
        yield row

def get_sensor_token(log_file: str, table: str, channel: str) -> str:
    """
    Get the sensor token of a particular channel for the given table.
    :param log_file: The DB file.
    :param table: The sensor table to query.
    :param channel: The channel to select.
    :return: The token of the sensor with the given channel.
    """
    q1 = f"\n        SELECT token\n        FROM {table}\n        WHERE channel == '{channel}';\n    "
    row = execute_one(q1, (), log_file)
    if row is None:
        raise RuntimeError(f'Channel {channel} not found in table {table}!')
    return str(row['token'].hex())

def get_images_from_lidar_tokens(log_file: str, tokens: List[str], channels: List[str], lookahead_window_us: int=50000, lookback_window_us: int=50000) -> Generator[Image, None, None]:
    """
    Get the images from the given channels for the given lidar_pc_tokens.
    Note: Both lookahead_window_us and lookback_window_us is defaulted to 50000us (0.05s). This means the search window
          is 0.1s centered around the queried lidar_pc timestamp. This is because lidar_pc are stored at 20hz and images
          are at 10hz for NuPlanDB. Hence, we can search the entire duration between lidar_pcs.
          Consider the example below where we want to query for images from the lidar_pc '4'. '|' represents a sample.

          iteration: 0    1    2    3   [4]   5    6
          timestamp: 0   0.05 0.1  0.15 0.2  0.25 0.3
          lidar_pc:  |    |    |    |    |    |    |
          Images:    |         |         |         |
          search window:            [---------]

          We set the search window to lookahead_window_us + lookback_window_us = 0.1s centered around lidar_pc '4'.
          This should guarantee that we retrieve the correct images associated with the queried lidar_pc.

    :param log_file: The log file to query.
    :param tokens: corresponding lidar_pc.
    :param channels: The channel to select.
    :param lookahead_window_us: [us] The time duration to look ahead relative to the lidar_pc for matching images.
    :param lookback_window_us: [us] The time duration to look back relative to the lidar_pc for matching images.
    :return: Images as a SensorDataTableRow.
    """
    query = f'\n            SELECT\n                img.token,\n                img.next_token,\n                img.prev_token,\n                img.ego_pose_token,\n                img.camera_token,\n                img.filename_jpg,\n                img.timestamp,\n                cam.channel\n            FROM image AS img\n              INNER JOIN lidar_pc AS lpc\n                ON  img.timestamp <= lpc.timestamp + ?\n                AND img.timestamp >= lpc.timestamp - ?\n              INNER JOIN camera AS cam\n                ON cam.token = img.camera_token\n            WHERE cam.channel IN ({('?,' * len(channels))[:-1]}) AND lpc.token IN ({('?,' * len(tokens))[:-1]})\n            ORDER BY lpc.timestamp ASC;\n    '
    args = [lookahead_window_us, lookback_window_us]
    args += channels
    args += [bytearray.fromhex(t) for t in tokens]
    for row in execute_many(query, args, log_file):
        yield Image.from_db_row(row)

def get_cameras(log_file: str, channels: List[str]) -> Generator[Camera, None, None]:
    """
    Get the cameras for the given channels.
    :param log_file: The log file to query.
    :param channels: The channel to select.
    :return: Cameras as a SensorDataTableRow.
    """
    query = f'\n            SELECT *\n            FROM camera AS cam\n            WHERE cam.channel IN ({('?,' * len(channels))[:-1]})\n    '
    for row in execute_many(query, channels, log_file):
        yield Camera.from_db_row(row)

class TestDbCliQueries(unittest.TestCase):
    """
    Test suite for the DB Cli queries.
    """

    @staticmethod
    def getDBFilePath() -> Path:
        """
        Get the location for the temporary SQLite file used for the test DB.
        :return: The filepath for the test data.
        """
        return Path('/tmp/test_db_cli_queries.sqlite3')

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create the mock DB data.
        """
        db_file_path = TestDbCliQueries.getDBFilePath()
        if db_file_path.exists():
            db_file_path.unlink()
        generation_parameters = DBGenerationParameters(num_lidars=1, num_cameras=2, num_sensor_data_per_sensor=50, num_lidarpc_per_image_ratio=2, num_scenes=10, num_traffic_lights_per_lidar_pc=5, num_agents_per_lidar_pc=3, num_static_objects_per_lidar_pc=2, scene_scenario_tag_mapping={5: ['first_tag'], 6: ['first_tag', 'second_tag']}, file_path=str(db_file_path))
        generate_minimal_nuplan_db(generation_parameters)

    def setUp(self) -> None:
        """
        The method to run before each test.
        """
        self.db_file_name = str(TestDbCliQueries.getDBFilePath())

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Destroy the mock DB data.
        """
        db_file_path = TestDbCliQueries.getDBFilePath()
        if os.path.exists(db_file_path):
            os.remove(db_file_path)

    def test_get_db_description(self) -> None:
        """
        Test the get_db_description queries.
        """
        db_description = get_db_description(self.db_file_name)
        expected_tables = ['category', 'ego_pose', 'lidar', 'lidar_box', 'lidar_pc', 'log', 'scenario_tag', 'scene', 'track', 'traffic_light_status', 'camera', 'image']
        self.assertEqual(len(expected_tables), len(db_description.tables))
        for expected_table in expected_tables:
            self.assertTrue(expected_table in db_description.tables)
        lidar_pc_table = db_description.tables['lidar_pc']
        self.assertEqual('lidar_pc', lidar_pc_table.name)
        self.assertEqual(50, lidar_pc_table.row_count)
        self.assertEqual(8, len(lidar_pc_table.columns))
        columns = sorted(lidar_pc_table.columns.values(), key=lambda x: x.column_id)

        def _validate_column(column: ColumnDescription, expected_id: int, expected_name: str, expected_data_type: str, expected_nullable: bool, expected_is_primary_key: bool) -> None:
            """
            A quick method to validate column info to reduce boilerplate.
            """
            self.assertEqual(expected_id, column.column_id)
            self.assertEqual(expected_name, column.name)
            self.assertEqual(expected_data_type, column.data_type)
            self.assertEqual(expected_nullable, column.nullable)
            self.assertEqual(expected_is_primary_key, column.is_primary_key)
        _validate_column(columns[0], 0, 'token', 'BLOB', False, True)
        _validate_column(columns[1], 1, 'next_token', 'BLOB', True, False)
        _validate_column(columns[2], 2, 'prev_token', 'BLOB', True, False)
        _validate_column(columns[3], 3, 'ego_pose_token', 'BLOB', False, False)
        _validate_column(columns[4], 4, 'lidar_token', 'BLOB', False, False)
        _validate_column(columns[5], 5, 'scene_token', 'BLOB', True, False)
        _validate_column(columns[6], 6, 'filename', 'VARCHAR(128)', True, False)
        _validate_column(columns[7], 7, 'timestamp', 'INTEGER', True, False)

    def test_get_db_duration_in_us(self) -> None:
        """
        Test the get_db_duration_in_us query
        """
        duration = get_db_duration_in_us(self.db_file_name)
        self.assertEqual(49 * 1000000.0, duration)

    def test_get_db_log_duration(self) -> None:
        """
        Test the get_db_log_duration query.
        """
        log_durations = list(get_db_log_duration(self.db_file_name))
        self.assertEqual(1, len(log_durations))
        self.assertEqual('logfile', log_durations[0][0])
        self.assertEqual(49 * 1000000.0, log_durations[0][1])

    def test_get_db_log_vehicles(self) -> None:
        """
        Test the get_db_log_vehicles query.
        """
        log_vehicles = list(get_db_log_vehicles(self.db_file_name))
        self.assertEqual(1, len(log_vehicles))
        self.assertEqual('logfile', log_vehicles[0][0])
        self.assertEqual('vehicle_name', log_vehicles[0][1])

    def test_get_db_scenario_info(self) -> None:
        """
        Test the get_db_scenario_info query.
        """
        scenario_info_tags = list(get_db_scenario_info(self.db_file_name))
        self.assertEqual(2, len(scenario_info_tags))
        self.assertEqual('first_tag', scenario_info_tags[0][0])
        self.assertEqual(2, scenario_info_tags[0][1])
        self.assertEqual('second_tag', scenario_info_tags[1][0])
        self.assertEqual(1, scenario_info_tags[1][1])

class TestSensorDataSource(unittest.TestCase):
    """Tests for the SensorDataSource class."""

    def test_initialization(self) -> None:
        """Tests correct initialization and raising of invalid configuration."""
        with self.assertRaisesRegex(AssertionError, 'Incompatible sensor_table: camera for table lidar_pc'):
            SensorDataSource('lidar_pc', 'camera', 'camera_token', '')
        with self.assertRaisesRegex(AssertionError, 'Incompatible sensor_table: lidar for table image'):
            SensorDataSource('image', 'lidar', 'lidar_token', '')
        with self.assertRaisesRegex(ValueError, 'Unknown requested sensor table: unknown'):
            SensorDataSource('unknown', '', '', '')
        with self.assertRaisesRegex(AssertionError, 'Incompatible sensor_token_column: lidar_token for sensor_table camera'):
            SensorDataSource('image', 'camera', 'lidar_token', '')
        _ = SensorDataSource('lidar_pc', 'lidar', 'lidar_token', '')
        valid_sensor_data_source = SensorDataSource('image', 'camera', 'camera_token', 'channel')
        self.assertEqual(valid_sensor_data_source.table, 'image')
        self.assertEqual(valid_sensor_data_source.sensor_table, 'camera')
        self.assertEqual(valid_sensor_data_source.sensor_token_column, 'camera_token')
        self.assertEqual(valid_sensor_data_source.channel, 'channel')

    def test_get_lidarpc_sensor_data(self) -> None:
        """Tests that utility function builds the correct object."""
        sensor_data = get_lidarpc_sensor_data()
        self.assertEqual(sensor_data.table, 'lidar_pc')
        self.assertEqual(sensor_data.sensor_table, 'lidar')
        self.assertEqual(sensor_data.sensor_token_column, 'lidar_token')
        self.assertEqual(sensor_data.channel, 'MergedPointCloud')

    def test_get_camera_channel_sensor_data(self) -> None:
        """Tests that utility function builds the correct object."""
        sensor_data = get_camera_channel_sensor_data('channel')
        self.assertEqual(sensor_data.table, 'image')
        self.assertEqual(sensor_data.sensor_table, 'camera')
        self.assertEqual(sensor_data.sensor_token_column, 'camera_token')
        self.assertEqual(sensor_data.channel, 'channel')

class TestNuPlanScenarioQueries(unittest.TestCase):
    """
    Test suite for the NuPlan scenario queries.
    """
    generation_parameters: DBGenerationParameters

    @staticmethod
    def getDBFilePath() -> Path:
        """
        Get the location for the temporary SQLite file used for the test DB.
        :return: The filepath for the test data.
        """
        return Path('/tmp/test_nuplan_scenario_queries.sqlite3')

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create the mock DB data.
        """
        db_file_path = TestNuPlanScenarioQueries.getDBFilePath()
        if db_file_path.exists():
            db_file_path.unlink()
        cls.generation_parameters = DBGenerationParameters(num_lidars=1, num_cameras=2, num_sensor_data_per_sensor=50, num_lidarpc_per_image_ratio=2, num_scenes=10, num_traffic_lights_per_lidar_pc=5, num_agents_per_lidar_pc=3, num_static_objects_per_lidar_pc=2, scene_scenario_tag_mapping={5: ['first_tag'], 6: ['first_tag', 'second_tag'], 7: ['second_tag']}, file_path=str(db_file_path))
        generate_minimal_nuplan_db(cls.generation_parameters)

    def setUp(self) -> None:
        """
        The method to run before each test.
        """
        self.db_file_name = str(TestNuPlanScenarioQueries.getDBFilePath())
        self.sensor_source = SensorDataSource('lidar_pc', 'lidar', 'lidar_token', 'channel')

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Destroy the mock DB data.
        """
        db_file_path = TestNuPlanScenarioQueries.getDBFilePath()
        if os.path.exists(db_file_path):
            os.remove(db_file_path)

    def test_get_sensor_token_from_index(self) -> None:
        """
        Test the get_sensor_token_from_index query.
        """
        for sample_index in [0, 12, 24]:
            retrieved_token = get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, sample_index)
            self.assertEqual(sample_index / self.generation_parameters.num_lidars, str_token_to_int(retrieved_token))
        self.assertIsNone(get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, 100000))
        with self.assertRaises(ValueError):
            get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, -2)

    def test_get_end_sensor_time_from_db(self) -> None:
        """
        Test the get_end_sensor_time_from_db query.
        """
        log_end_time = get_end_sensor_time_from_db(self.db_file_name, sensor_source=self.sensor_source)
        self.assertEqual(49 * 1000000.0, log_end_time)

    def test_get_sensor_token_timestamp_from_db(self) -> None:
        """
        Test the get_sensor_data_token_timestamp_from_db query.
        """
        for token in [0, 3, 7]:
            expected_timestamp = token * 1000000.0
            actual_timestamp = get_sensor_data_token_timestamp_from_db(self.db_file_name, self.sensor_source, int_to_str_token(token))
            self.assertEqual(expected_timestamp, actual_timestamp)
        self.assertIsNone(get_sensor_data_token_timestamp_from_db(self.db_file_name, self.sensor_source, int_to_str_token(1000)))

    def test_get_sensor_token_map_name_from_db(self) -> None:
        """
        Test the get_sensor_token_map_name_from_db query.
        """
        for token in [0, 2, 6]:
            expected_map_name = 'map_version'
            actual_map_name = get_sensor_token_map_name_from_db(self.db_file_name, self.sensor_source, int_to_str_token(token))
            self.assertEqual(expected_map_name, actual_map_name)
        self.assertIsNone(get_sensor_token_map_name_from_db(self.db_file_name, self.sensor_source, int_to_str_token(1000)))

    def test_get_sampled_sensor_tokens_in_time_window_from_db(self) -> None:
        """
        Test the get_sampled_lidarpc_tokens_in_time_window_from_db query.
        """
        expected_tokens = [10, 13, 16, 19]
        actual_tokens = list((str_token_to_int(v) for v in get_sampled_sensor_tokens_in_time_window_from_db(log_file=self.db_file_name, sensor_source=self.sensor_source, start_timestamp=int(10 * 1000000.0), end_timestamp=int(20 * 1000000.0), subsample_interval=3)))
        self.assertEqual(expected_tokens, actual_tokens)

    def test_get_sensor_data_from_sensor_data_tokens_from_db(self) -> None:
        """
        Test the get_sensor_data_from_sensor_data_tokens_from_db query.
        """
        lidar_pc_tokens = [int_to_str_token(v) for v in [10, 13, 21]]
        image_tokens = [int_to_str_token(v) for v in [1100000]]
        lidar_pcs = [cast(LidarPc, sensor_data) for sensor_data in get_sensor_data_from_sensor_data_tokens_from_db(self.db_file_name, self.sensor_source, LidarPc, lidar_pc_tokens)]
        images = [cast(Image, sensor_data) for sensor_data in get_sensor_data_from_sensor_data_tokens_from_db(self.db_file_name, SensorDataSource('image', 'camera', 'camera_token', 'camera_0'), Image, image_tokens)]
        self.assertEqual(len(lidar_pc_tokens), len(lidar_pcs))
        self.assertEqual(len(image_tokens), len(images))
        lidar_pcs.sort(key=lambda x: int(x.timestamp))
        self.assertEqual(10, str_token_to_int(lidar_pcs[0].token))
        self.assertEqual(13, str_token_to_int(lidar_pcs[1].token))
        self.assertEqual(21, str_token_to_int(lidar_pcs[2].token))
        self.assertEqual(1100000, str_token_to_int(images[0].token))

    def test_get_lidar_transform_matrix_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_sensor_transform_matrix_for_sensor_data_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            xform_mat = get_sensor_transform_matrix_for_sensor_data_token_from_db(self.db_file_name, self.sensor_source, int_to_str_token(sample_token))
            self.assertIsNotNone(xform_mat)
            self.assertEqual(xform_mat[0, 3], 0)

    def test_get_mission_goal_for_sensor_data_token_from_db(self) -> None:
        """
        Test the get_mission_goal_for_sensor_data_token_from_db query.
        """
        query_lidarpc_token = int_to_str_token(12)
        expected_ego_pose_x = 14
        expected_ego_pose_y = 15
        result = get_mission_goal_for_sensor_data_token_from_db(self.db_file_name, self.sensor_source, query_lidarpc_token)
        self.assertIsNotNone(result)
        self.assertEqual(expected_ego_pose_x, result.x)
        self.assertEqual(expected_ego_pose_y, result.y)

    def test_get_roadblock_ids_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_roadblock_ids_for_lidarpc_token_from_db query.
        """
        result = get_roadblock_ids_for_lidarpc_token_from_db(self.db_file_name, int_to_str_token(0))
        self.assertEqual(result, ['0', '1', '2'])

    def test_get_statese2_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_statese2_for_lidarpc_token_from_db query.
        """
        query_lidarpc_token = int_to_str_token(13)
        expected_ego_pose_x = 13
        expected_ego_pose_y = 14
        result = get_statese2_for_lidarpc_token_from_db(self.db_file_name, query_lidarpc_token)
        self.assertIsNotNone(result)
        self.assertEqual(expected_ego_pose_x, result.x)
        self.assertEqual(expected_ego_pose_y, result.y)

    def test_get_sampled_lidarpcs_from_db(self) -> None:
        """
        Test the get_sampled_lidarpcs_from_db query.
        """
        test_cases = [{'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': True, 'expected_return_tokens': [5, 6, 7]}, {'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': False, 'expected_return_tokens': [3, 4, 5]}, {'initial_token': 7, 'sample_indexes': [0, 3, 12], 'future': False, 'expected_return_tokens': [4, 7]}, {'initial_token': 0, 'sample_indexes': [1000], 'future': True, 'expected_return_tokens': []}]
        for test_case in test_cases:
            initial_token = int_to_str_token(test_case['initial_token'])
            expected_return_tokens = [int_to_str_token(v) for v in test_case['expected_return_tokens']]
            actual_returned_lidarpcs = list(get_sampled_lidarpcs_from_db(self.db_file_name, initial_token, self.sensor_source, test_case['sample_indexes'], test_case['future']))
            self.assertEqual(len(expected_return_tokens), len(actual_returned_lidarpcs))
            for i in range(len(expected_return_tokens)):
                self.assertEqual(expected_return_tokens[i], actual_returned_lidarpcs[i].token)

    def test_get_sampled_ego_states_from_db(self) -> None:
        """
        Test the get_sampled_ego_states_from_db query.
        """
        test_cases = [{'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': True, 'expected_row_indexes': [5, 6, 7]}, {'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': False, 'expected_row_indexes': [3, 4, 5]}, {'initial_token': 7, 'sample_indexes': [0, 3, 12], 'future': False, 'expected_row_indexes': [4, 7]}, {'initial_token': 0, 'sample_indexes': [1000], 'future': True, 'expected_row_indexes': []}]
        for test_case in test_cases:
            initial_token = int_to_str_token(test_case['initial_token'])
            expected_row_indexes = test_case['expected_row_indexes']
            actual_returned_ego_states = list(get_sampled_ego_states_from_db(self.db_file_name, initial_token, self.sensor_source, test_case['sample_indexes'], test_case['future']))
            self.assertEqual(len(expected_row_indexes), len(actual_returned_ego_states))
            for i in range(len(expected_row_indexes)):
                self.assertEqual(expected_row_indexes[i] * 1000000.0, actual_returned_ego_states[i].time_point.time_us)

    def test_get_ego_state_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_ego_state_for_lidarpc_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            returned_pose = get_ego_state_for_lidarpc_token_from_db(self.db_file_name, query_token)
            self.assertEqual(sample_token * 1000000.0, returned_pose.time_point.time_us)

    def test_get_traffic_light_status_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_traffic_light_status_for_lidarpc_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            traffic_light_statuses = list(get_traffic_light_status_for_lidarpc_token_from_db(self.db_file_name, query_token))
            self.assertEqual(5, len(traffic_light_statuses))
            for tl_status in traffic_light_statuses:
                self.assertEqual(sample_token * 1000000.0, tl_status.timestamp)

    def test_get_tracked_objects_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_tracked_objects_for_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            tracked_objects = list(get_tracked_objects_for_lidarpc_token_from_db(self.db_file_name, query_token))
            self.assertEqual(5, len(tracked_objects))
            agent_count = 0
            static_object_count = 0
            track_token_base_id = 600000
            token_base_id = 500000
            token_sample_step = 10000
            for idx, tracked_object in enumerate(tracked_objects):
                expected_track_token = track_token_base_id + idx
                expected_token = token_base_id + token_sample_step * sample_token + idx
                self.assertEqual(int_to_str_token(expected_track_token), tracked_object.track_token)
                self.assertEqual(int_to_str_token(expected_token), tracked_object.token)
                if isinstance(tracked_object, Agent):
                    agent_count += 1
                    self.assertEqual(TrackedObjectType.VEHICLE, tracked_object.tracked_object_type)
                    self.assertEqual(0, len(tracked_object.predictions))
                elif isinstance(tracked_object, StaticObject):
                    static_object_count += 1
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, tracked_object.tracked_object_type)
                else:
                    raise ValueError(f'Unexpected type: {type(tracked_object)}')
            self.assertEqual(3, agent_count)
            self.assertEqual(2, static_object_count)

    def test_get_tracked_objects_within_time_interval_from_db(self) -> None:
        """
        Test the get_tracked_objects_within_time_interval_from_db query.
        """
        expected_num_windows = {0: 3, 30: 5, 48: 4}
        expected_backward_offset = {0: 0, 30: -2, 48: -2}
        for sample_token in expected_num_windows.keys():
            start_timestamp = int(1000000.0 * (sample_token - 2))
            end_timestamp = int(1000000.0 * (sample_token + 2))
            tracked_objects = list(get_tracked_objects_within_time_interval_from_db(self.db_file_name, start_timestamp, end_timestamp, filter_track_tokens=None))
            expected_num_tokens = expected_num_windows[sample_token] * 5
            self.assertEqual(expected_num_tokens, len(tracked_objects))
            agent_count = 0
            static_object_count = 0
            track_token_base_id = 600000
            token_base_id = 500000
            token_sample_step = 10000
            for idx, tracked_object in enumerate(tracked_objects):
                expected_track_token = track_token_base_id + idx % 5
                expected_token = token_base_id + token_sample_step * (sample_token + expected_backward_offset[sample_token] + math.floor(idx / 5)) + idx % 5
                self.assertEqual(int_to_str_token(expected_track_token), tracked_object.track_token)
                self.assertEqual(int_to_str_token(expected_token), tracked_object.token)
                if isinstance(tracked_object, Agent):
                    agent_count += 1
                    self.assertEqual(TrackedObjectType.VEHICLE, tracked_object.tracked_object_type)
                    self.assertEqual(0, len(tracked_object.predictions))
                elif isinstance(tracked_object, StaticObject):
                    static_object_count += 1
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, tracked_object.tracked_object_type)
                else:
                    raise ValueError(f'Unexpected type: {type(tracked_object)}')
            self.assertEqual(3 * expected_num_windows[sample_token], agent_count)
            self.assertEqual(2 * expected_num_windows[sample_token], static_object_count)

    def test_get_future_waypoints_for_agents_from_db(self) -> None:
        """
        Test the get_future_waypoints_for_agents_from_db query.
        """
        track_tokens = [600000, 600001, 600002]
        start_timestamp = 0
        end_timestamp = int(20 * 1000000.0 - 1)
        query_output: Dict[str, List[Waypoint]] = {}
        for token, waypoint in get_future_waypoints_for_agents_from_db(self.db_file_name, (int_to_str_token(t) for t in track_tokens), start_timestamp, end_timestamp):
            if token not in query_output:
                query_output[token] = []
            query_output[token].append(waypoint)
        expected_keys = ['{:08d}'.format(t) for t in track_tokens]
        self.assertEqual(len(expected_keys), len(query_output))
        for expected_key in expected_keys:
            self.assertTrue(expected_key in query_output)
            collected_waypoints = query_output[expected_key]
            self.assertEqual(20, len(collected_waypoints))
            for i in range(0, len(collected_waypoints), 1):
                self.assertEqual(i * 1000000.0, collected_waypoints[i].time_point.time_us)

    def test_get_scenarios_from_db(self) -> None:
        """
        Test the get_scenarios_from_db_query.
        """
        no_filter_output: List[int] = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            no_filter_output.append(str_token_to_int(row['token'].hex()))
        self.assertEqual(list(range(10, 40, 1)), no_filter_output)
        filter_tokens = [int_to_str_token(v) for v in [15, 30]]
        tokens_filter_output: List[int] = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=filter_tokens, filter_types=None, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            tokens_filter_output.append(row['token'].hex())
        self.assertEqual(filter_tokens, tokens_filter_output)
        filter_scenarios = ['first_tag']
        extracted_rows: List[Tuple[int, str]] = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=filter_scenarios, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            extracted_rows.append((str_token_to_int(row['token'].hex()), row['scenario_type']))
        self.assertEqual(2, len(extracted_rows))
        self.assertEqual(25, extracted_rows[0][0])
        self.assertEqual('first_tag', extracted_rows[0][1])
        self.assertEqual(30, extracted_rows[1][0])
        self.assertEqual('first_tag', extracted_rows[1][1])
        filter_scenarios = ['second_tag']
        extracted_rows = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=filter_scenarios, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            extracted_rows.append((str_token_to_int(row['token'].hex()), row['scenario_type']))
        self.assertEqual(2, len(extracted_rows))
        self.assertEqual(30, extracted_rows[0][0])
        self.assertEqual('second_tag', extracted_rows[0][1])
        self.assertEqual(35, extracted_rows[1][0])
        self.assertEqual('second_tag', extracted_rows[1][1])
        filter_maps = ['map_version']
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=filter_maps, include_invalid_mission_goals=False, include_cameras=False)))
        self.assertLess(0, row_cnt)
        filter_maps = ['map_that_does_not_exist']
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=filter_maps, include_invalid_mission_goals=False, include_cameras=False)))
        self.assertEqual(0, row_cnt)
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=True)))
        self.assertEqual(15, row_cnt)
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=[int_to_str_token(25)], filter_types=['first_tag'], filter_map_names=['map_version'], include_invalid_mission_goals=False, include_cameras=False)))
        self.assertEqual(1, row_cnt)

    def test_get_lidarpc_tokens_with_scenario_tag_from_db(self) -> None:
        """
        Test the get_lidarpc_tokens_with_scenario_tag_from_db query.
        """
        tuples = list(get_lidarpc_tokens_with_scenario_tag_from_db(self.db_file_name))
        self.assertEqual(4, len(tuples))
        expected_tuples = [('first_tag', int_to_str_token(25)), ('first_tag', int_to_str_token(30)), ('second_tag', int_to_str_token(30)), ('second_tag', int_to_str_token(35))]
        for tup in tuples:
            self.assertTrue(tup in expected_tuples)

    def test_get_sensor_token(self) -> None:
        """Test the get_lidarpc_token_from_index query."""
        retrieved_token = get_sensor_token(self.db_file_name, 'lidar', 'channel')
        self.assertEqual(700000, str_token_to_int(retrieved_token))
        with self.assertRaisesRegex(RuntimeError, 'Channel missing_channel not found in table lidar!'):
            self.assertIsNone(get_sensor_token(self.db_file_name, 'lidar', 'missing_channel'))

    def test_get_images_from_lidar_tokens(self) -> None:
        """Test the get_images_from_lidar_tokens query."""
        token = int_to_str_token(20)
        retrieved_images = list(get_images_from_lidar_tokens(self.db_file_name, [token], ['camera_0', 'camera_1'], 50000, 50000))
        self.assertEqual(2, len(retrieved_images))
        self.assertEqual(1100020, str_token_to_int(retrieved_images[0].token))
        self.assertEqual(1100070, str_token_to_int(retrieved_images[1].token))
        self.assertEqual('camera_0', retrieved_images[0].channel)
        self.assertEqual('camera_1', retrieved_images[1].channel)

    def test_get_cameras(self) -> None:
        """Test the get_cameras query."""
        retrieved_cameras = list(get_cameras(self.db_file_name, ['camera_0', 'camera_1']))
        self.assertEqual(2, len(retrieved_cameras))
        self.assertEqual(1000000, str_token_to_int(retrieved_cameras[0].token))
        self.assertEqual(1000001, str_token_to_int(retrieved_cameras[1].token))
        self.assertEqual('camera_0', retrieved_cameras[0].channel)
        self.assertEqual('camera_1', retrieved_cameras[1].channel)
        retrieved_cameras = list(get_cameras(self.db_file_name, ['camera_1']))
        self.assertEqual(1, len(retrieved_cameras))
        self.assertEqual(1000001, str_token_to_int(retrieved_cameras[0].token))
        self.assertEqual('camera_1', retrieved_cameras[0].channel)

def str_token_to_int(val: Optional[str]) -> Optional[int]:
    """
    Convert a string token previously genreated with int_to_str_token() back to an int.
    :param val: The token to convert.
    :return: None if the input is None. Else, the int version of the string.
        The output is undefined if the token was not generated with int_to_str_token().
    """
    return None if val is None else int(val)

def int_to_str_token(val: Optional[int]) -> Optional[str]:
    """
    Convert an int to a string token used for DB access functions.
    :param val: The val to convert.
    :return: None if the input is None. Else, a string version of the input value to be used with db functions as a token.
    """
    return None if val is None else '{:08d}'.format(val)

def extract_sensor_tokens_as_scenario(log_file: str, sensor_data_source: SensorDataSource, anchor_timestamp: float, scenario_extraction_info: ScenarioExtractionInfo) -> Generator[str, None, None]:
    """
    Extract a list of sensor tokens that form a scenario around an anchor timestamp.
    :param log_file: The log file to access
    :param sensor_data_source: Parameters for querying the correct table.
    :param anchor_timestamp: Timestamp of Sensor representing the start of the scenario.
    :param scenario_extraction_info: Structure containing information used to extract the scenario.
    :return: List of extracted sensor tokens representing the scenario.
    """
    start_timestamp = int(anchor_timestamp + scenario_extraction_info.extraction_offset * 1000000.0)
    end_timestamp = int(start_timestamp + scenario_extraction_info.scenario_duration * 1000000.0)
    subsample_step = int(1.0 / scenario_extraction_info.subsample_ratio)
    return cast(Generator[str, None, None], get_sampled_sensor_tokens_in_time_window_from_db(log_file, sensor_data_source, start_timestamp, end_timestamp, subsample_step))

class NuPlanScenario(AbstractScenario):
    """Scenario implementation for the nuPlan dataset that is used in training and simulation."""

    def __init__(self, data_root: str, log_file_load_path: str, initial_lidar_token: str, initial_lidar_timestamp: int, scenario_type: str, map_root: str, map_version: str, map_name: str, scenario_extraction_info: Optional[ScenarioExtractionInfo], ego_vehicle_parameters: VehicleParameters, sensor_root: Optional[str]=None) -> None:
        """
        Initialize the nuPlan scenario.
        :param data_root: The prefix for the log file. e.g. "/data/root/nuplan". For remote paths, this is where the file will be downloaded if necessary.
        :param log_file_load_path: Name of the log that this scenario belongs to. e.g. "/data/sets/nuplan-v1.1/splits/mini/2021.07.16.20.45.29_veh-35_01095_01486.db", "s3://path/to/db.db"
        :param initial_lidar_token: Token of the scenario's initial lidarpc.
        :param initial_lidar_timestamp: The timestamp of the initial lidarpc.
        :param scenario_type: Type of scenario (e.g. ego overtaking).
        :param map_root: The root path for the map db
        :param map_version: The version of maps to load
        :param map_name: The map name to use for the scenario
        :param scenario_extraction_info: Structure containing information used to extract the scenario.
            None means the scenario has no length and it is comprised only by the initial lidarpc.
        :param ego_vehicle_parameters: Structure containing the vehicle parameters.
        :param sensor_root: The root path for the sensor blobs.
        """
        self._local_store: Optional[LocalStore] = None
        self._remote_store: Optional[S3Store] = None
        self._data_root = data_root
        self._log_file_load_path = log_file_load_path
        self._initial_lidar_token = initial_lidar_token
        self._initial_lidar_timestamp = initial_lidar_timestamp
        self._scenario_type = scenario_type
        self._map_root = map_root
        self._map_version = map_version
        self._map_name = map_name
        self._scenario_extraction_info = scenario_extraction_info
        self._ego_vehicle_parameters = ego_vehicle_parameters
        self._sensor_root = sensor_root
        if self._scenario_extraction_info is not None:
            skip_rows = 1.0 / self._scenario_extraction_info.subsample_ratio
            if abs(int(skip_rows) - skip_rows) > 0.001:
                raise ValueError(f'Subsample ratio is not valid. Must resolve to an integer number of skipping rows, instead received {self._scenario_extraction_info.subsample_ratio}, which would skip {skip_rows} rows.')
        self._database_row_interval = 0.05
        self._log_file = download_file_if_necessary(self._data_root, self._log_file_load_path)
        self._log_name: str = absolute_path_to_log_name(self._log_file)

    def __reduce__(self) -> Tuple[Type[NuPlanScenario], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (self.__class__, (self._data_root, self._log_file_load_path, self._initial_lidar_token, self._initial_lidar_timestamp, self._scenario_type, self._map_root, self._map_version, self._map_name, self._scenario_extraction_info, self._ego_vehicle_parameters, self._sensor_root))

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return self._ego_vehicle_parameters

    @cached_property
    def _lidarpc_tokens(self) -> List[str]:
        """
        :return: list of lidarpc tokens in the scenario
        """
        if self._scenario_extraction_info is None:
            return [self._initial_lidar_token]
        lidarpc_tokens = list(extract_sensor_tokens_as_scenario(self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_timestamp, self._scenario_extraction_info))
        return cast(List[str], lidarpc_tokens)

    @cached_property
    def _route_roadblock_ids(self) -> List[str]:
        """
        return: Route roadblock ids extracted from expert trajectory.
        """
        expert_trajectory = list(self._extract_expert_trajectory())
        return get_roadblock_ids_from_trajectory(self.map_api, expert_trajectory)

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._initial_lidar_token

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self.token

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_type

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        return get_maps_api(self._map_root, self._map_version, self._map_name)

    @property
    def map_root(self) -> str:
        """Get the map root folder."""
        return self._map_root

    @property
    def map_version(self) -> str:
        """Get the map version."""
        return self._map_version

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        if self._scenario_extraction_info is None:
            return 0.05
        return float(0.05 / self._scenario_extraction_info.subsample_ratio)

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return len(self._lidarpc_tokens)

    def get_lidar_to_ego_transform(self) -> Transform:
        """Inherited, see superclass."""
        return get_sensor_transform_matrix_for_sensor_data_token_from_db(self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token)

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Inherited, see superclass."""
        return get_mission_goal_for_sensor_data_token_from_db(self._log_file, get_lidarpc_sensor_data(), self._initial_lidar_token)

    def get_route_roadblock_ids(self) -> List[str]:
        """Inherited, see superclass."""
        roadblock_ids = get_roadblock_ids_for_lidarpc_token_from_db(self._log_file, self._initial_lidar_token)
        assert roadblock_ids is not None, 'Unable to find Roadblock ids for current scenario'
        return cast(List[str], roadblock_ids)

    def get_expert_goal_state(self) -> StateSE2:
        """Inherited, see superclass."""
        return get_statese2_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[-1])

    def get_time_point(self, iteration: int) -> TimePoint:
        """Inherited, see superclass."""
        return TimePoint(time_us=get_sensor_data_token_timestamp_from_db(self._log_file, get_lidarpc_sensor_data(), self._lidarpc_tokens[iteration]))

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""
        return get_ego_state_for_lidarpc_token_from_db(self._log_file, self._lidarpc_tokens[iteration])

    def get_tracked_objects_at_iteration(self, iteration: int, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f'Iteration is out of scenario: {iteration}!'
        return DetectionsTracks(extract_tracked_objects(self._lidarpc_tokens[iteration], self._log_file, future_trajectory_sampling))

    def get_tracked_objects_within_time_window_at_iteration(self, iteration: int, past_time_horizon: float, future_time_horizon: float, filter_track_tokens: Optional[Set[str]]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f'Iteration is out of scenario: {iteration}!'
        return DetectionsTracks(extract_tracked_objects_within_time_window(self._lidarpc_tokens[iteration], self._log_file, past_time_horizon, future_time_horizon, filter_track_tokens, future_trajectory_sampling))

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]]=None) -> Sensors:
        """Inherited, see superclass."""
        channels = [LidarChannel.MERGED_PC] if channels is None else channels
        lidar_pc = next(get_sensor_data_from_sensor_data_tokens_from_db(self._log_file, get_lidarpc_sensor_data(), LidarPc, [self._lidarpc_tokens[iteration]]))
        return self._get_sensor_data_from_lidar_pc(cast(LidarPc, lidar_pc), channels)

    def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TimePoint(lidar_pc.timestamp)

    def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TimePoint(lidar_pc.timestamp)

    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)
        return cast(Generator[EgoState, None, None], get_sampled_ego_states_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=False))

    def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)
        return cast(Generator[EgoState, None, None], get_sampled_ego_states_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, future=True))

    def get_past_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_future_tracked_objects(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, future_trajectory_sampling: Optional[TrajectorySampling]=None) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield DetectionsTracks(extract_tracked_objects(lidar_pc.token, self._log_file, future_trajectory_sampling))

    def get_past_sensors(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None, channels: Optional[List[SensorChannel]]=None) -> Generator[Sensors, None, None]:
        """Inherited, see superclass."""
        channels = [LidarChannel.MERGED_PC] if channels is None else channels
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield self._get_sensor_data_from_lidar_pc(lidar_pc, channels)

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Inherited, see superclass."""
        token = self._lidarpc_tokens[iteration]
        return cast(Generator[TrafficLightStatusData, None, None], get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, token))

    def get_past_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets past traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the past.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the past.
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, False):
            yield TrafficLightStatuses(list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token)))

    def get_future_traffic_light_status_history(self, iteration: int, time_horizon: float, num_samples: Optional[int]=None) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets future traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the future.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the future.
        """
        for lidar_pc in self._find_matching_lidar_pcs(iteration, num_samples, time_horizon, True):
            yield TrafficLightStatuses(list(get_traffic_light_status_for_lidarpc_token_from_db(self._log_file, lidar_pc.token)))

    def get_scenario_tokens(self) -> List[str]:
        """Return the list of lidarpc tokens from the DB that are contained in the scenario."""
        return self._lidarpc_tokens

    def _find_matching_lidar_pcs(self, iteration: int, num_samples: Optional[int], time_horizon: float, look_into_future: bool) -> Generator[LidarPc, None, None]:
        """
        Find the best matching lidar_pcs to the desired samples and time horizon
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future, if None it will be deduced from the DB
        :param time_horizon: the desired horizon to the future
        :param look_into_future: if True, we will iterate into next lidar_pc otherwise we will iterate through prev
        :return: lidar_pcs matching to database indices
        """
        num_samples = num_samples if num_samples else int(time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._database_row_interval)
        return cast(Generator[LidarPc, None, None], get_sampled_lidarpcs_from_db(self._log_file, self._lidarpc_tokens[iteration], get_lidarpc_sensor_data(), indices, look_into_future))

    def _extract_expert_trajectory(self, max_future_seconds: int=60) -> Generator[EgoState, None, None]:
        """
        Extract expert trajectory with specified time parameters. If initial lidar pc does not have enough history/future
            only available time will be extracted
        :param max_future_seconds: time to future which should be considered for route extraction [s]
        :return: list of expert ego states
        """
        minimal_required_future_time_available = 0.5
        end_log_time_us = get_end_sensor_time_from_db(self._log_file, get_lidarpc_sensor_data())
        max_future_time = min((end_log_time_us - self._initial_lidar_timestamp) * 1e-06, max_future_seconds)
        if max_future_time < minimal_required_future_time_available:
            return
        for traj in self.get_ego_future_trajectory(0, max_future_time):
            yield traj

    def _create_blob_store_if_needed(self) -> Tuple[LocalStore, Optional[S3Store]]:
        """
        A convenience method that creates the blob stores if it's not already created.
        :return: The created or cached LocalStore and S3Store objects.
        """
        if self._local_store is not None and self._remote_store is not None:
            return (self._local_store, self._remote_store)
        if self._sensor_root is None:
            raise ValueError('sensor_root is not set. Please set the sensor_root to access sensor data.')
        Path(self._sensor_root).mkdir(exist_ok=True)
        self._local_store = LocalStore(self._sensor_root)
        if os.getenv('NUPLAN_DATA_STORE', '') == 's3':
            s3_url = os.getenv('NUPLAN_DATA_ROOT_S3_URL', '')
            self._remote_store = S3Store(os.path.join(s3_url, 'sensor_blobs'), show_progress=True)
        return (self._local_store, self._remote_store)

    def _get_sensor_data_from_lidar_pc(self, lidar_pc: LidarPc, channels: List[SensorChannel]) -> Sensors:
        """
        Loads Sensor data given a database LidarPC object.
        :param lidar_pc: The lidar_pc for which to grab the point cloud.
        :param channels: The sensor channels to return.
        :return: The corresponding sensor data.
        """
        local_store, remote_store = self._create_blob_store_if_needed()
        retrieved_images = get_images_from_lidar_tokens(self._log_file, [lidar_pc.token], [cast(str, channel.value) for channel in channels])
        lidar_pcs = {LidarChannel.MERGED_PC: load_point_cloud(cast(LidarPc, lidar_pc), local_store, remote_store)} if LidarChannel.MERGED_PC in channels else None
        images = {CameraChannel[image.channel]: load_image(image, local_store, remote_store) for image in retrieved_images}
        return Sensors(pointcloud=lidar_pcs, images=images if images else None)

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

