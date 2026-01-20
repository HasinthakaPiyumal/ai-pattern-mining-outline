# Cluster 85

class MariaDbHandler(DBHandler):
    """
    Class for implementing the Maria DB handler as a backend store for
    EvaDB.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the handler.
        Args:
            name (str): name of the DB handler instance
            **kwargs: arbitrary keyword arguments for establishing the connection.
        """
        super().__init__(name)
        self.host = kwargs.get('host')
        self.port = kwargs.get('port')
        self.user = kwargs.get('user')
        self.password = kwargs.get('password')
        self.database = kwargs.get('database')

    def connect(self):
        """
        Establish connection to the database.
        Returns:
          DBHandlerStatus
        """
        try:
            self.connection = mariadb.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.database)
            self.connection.autocommit = True
            return DBHandlerStatus(status=True)
        except mariadb.Error as e:
            return DBHandlerStatus(status=False, error=str(e))

    def disconnect(self):
        """
        Disconnect from the database.
        """
        if self.connection:
            self.connection.close()

    def get_sqlalchmey_uri(self) -> str:
        return f'mariadb+mariadbconnector://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'

    def check_connection(self) -> DBHandlerStatus:
        """
        Method for checking the status of database connection.
        Returns:
          DBHandlerStatus
        """
        if self.connection:
            return DBHandlerStatus(status=True)
        else:
            return DBHandlerStatus(status=False, error='Not connected to the database.')

    def get_tables(self) -> DBHandlerResponse:
        """
        Method to get the list of tables from database.
        Returns:
          DBHandlerStatus
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = f"SELECT table_name as 'table_name' FROM information_schema.tables WHERE table_schema='{self.database}'"
            tables_df = pd.read_sql_query(query, self.connection)
            return DBHandlerResponse(data=tables_df)
        except mariadb.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def get_columns(self, table_name: str) -> DBHandlerResponse:
        """
        Method to retrieve the columns of the specified table from the database.
        Args:
          table_name (str): name of the table whose columns are to be retrieved.
        Returns:
          DBHandlerStatus
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = f"SELECT column_name as 'name', data_type as 'dtype' FROM information_schema.columns WHERE table_name='{table_name}'"
            columns_df = pd.read_sql_query(query, self.connection)
            columns_df['dtype'] = columns_df['dtype'].apply(self._mariadb_to_python_types)
            return DBHandlerResponse(data=columns_df)
        except mariadb.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _fetch_results_as_df(self, cursor):
        """
        Fetch results from the cursor for the executed query and return the
        query results as dataframe.
        """
        try:
            res = cursor.fetchall()
            res_df = pd.DataFrame(res, columns=[desc[0].lower() for desc in cursor.description])
            return res_df
        except mariadb.ProgrammingError as e:
            if str(e) == 'no results to fetch':
                return pd.DataFrame({'status': ['success']})
            raise e

    def execute_native_query(self, query_string: str) -> DBHandlerResponse:
        """
        Executes the native query on the database.
        Args:
            query_string (str): query in native format
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            cursor = self.connection.cursor()
            cursor.execute(query_string)
            return DBHandlerResponse(data=self._fetch_results_as_df(cursor))
        except mariadb.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _mariadb_to_python_types(self, mariadb_type: str):
        mapping = {'tinyint': int, 'smallint': int, 'mediumint': int, 'bigint': int, 'int': int, 'decimal': float, 'float': float, 'double': float, 'text': str, 'string literals': str, 'char': str, 'varchar': str, 'boolean': bool}
        if mariadb_type in mapping:
            return mapping[mariadb_type]
        else:
            raise Exception(f'Unsupported column {mariadb_type} encountered in the MariaDB. Please raise a feature request!')

class MysqlHandler(DBHandler):

    def __init__(self, name: str, **kwargs):
        super().__init__(name)
        self.host = kwargs.get('host')
        self.port = kwargs.get('port')
        self.user = kwargs.get('user')
        self.password = kwargs.get('password')
        self.database = kwargs.get('database')

    def connect(self):
        try:
            self.connection = mysql.connector.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.database)
            self.connection.autocommit = True
            return DBHandlerStatus(status=True)
        except mysql.connector.Error as e:
            return DBHandlerStatus(status=False, error=str(e))

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def get_sqlalchmey_uri(self) -> str:
        return f'mysql+mysqlconnector://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'

    def check_connection(self) -> DBHandlerStatus:
        if self.connection:
            return DBHandlerStatus(status=True)
        else:
            return DBHandlerStatus(status=False, error='Not connected to the database.')

    def get_tables(self) -> DBHandlerResponse:
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = f"SELECT table_name as 'table_name' FROM information_schema.tables WHERE table_schema='{self.database}'"
            tables_df = pd.read_sql_query(query, self.connection)
            return DBHandlerResponse(data=tables_df)
        except mysql.connector.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def get_columns(self, table_name: str) -> DBHandlerResponse:
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = f"SELECT column_name as 'name', data_type as dtype FROM information_schema.columns WHERE table_name='{table_name}'"
            columns_df = pd.read_sql_query(query, self.connection)
            columns_df['dtype'] = columns_df['dtype'].apply(self._mysql_to_python_types)
            return DBHandlerResponse(data=columns_df)
        except mysql.connector.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _fetch_results_as_df(self, cursor):
        """
        This is currently the only clean solution that we have found so far.
        Reference to MySQL API: https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-fetchall.html
        In short, currently there is no very clean programming way to differentiate
        CREATE, INSERT, SELECT. CREATE and INSERT do not return any result, so calling
        fetchall() on those will yield a programming error. Cursor has an attribute
        rowcount, but it indicates # of rows that are affected. In that case, for both
        INSERT and SELECT rowcount is not 0, so we also cannot use this API to
        differentiate INSERT and SELECT.
        """
        try:
            res = cursor.fetchall()
            if not res:
                return pd.DataFrame({'status': ['success']})
            res_df = pd.DataFrame(res, columns=[desc[0].lower() for desc in cursor.description])
            return res_df
        except mysql.connector.ProgrammingError as e:
            if str(e) == 'no results to fetch':
                return pd.DataFrame({'status': ['success']})
            raise e

    def execute_native_query(self, query_string: str) -> DBHandlerResponse:
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            cursor = self.connection.cursor()
            cursor.execute(query_string)
            return DBHandlerResponse(data=self._fetch_results_as_df(cursor))
        except mysql.connector.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _mysql_to_python_types(self, mysql_type: str):
        mapping = {'char': str, 'varchar': str, 'text': str, 'boolean': bool, 'integer': int, 'int': int, 'float': float, 'double': float}
        if mysql_type in mapping:
            return mapping[mysql_type]
        else:
            raise Exception(f'Unsupported column {mysql_type} encountered in the mysql table. Please raise a feature request!')

class SnowFlakeDbHandler(DBHandler):
    """
    Class for implementing the SnowFlake DB handler as a backend store for
    EvaDB.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the handler.
        Args:
            name (str): name of the DB handler instance
            **kwargs: arbitrary keyword arguments for establishing the connection.
        """
        super().__init__(name)
        self.user = kwargs.get('user')
        self.password = kwargs.get('password')
        self.database = kwargs.get('database')
        self.warehouse = kwargs.get('warehouse')
        self.account = kwargs.get('account')
        self.schema = kwargs.get('schema')

    def connect(self):
        """
        Establish connection to the database.
        Returns:
          DBHandlerStatus
        """
        try:
            self.connection = snowflake.connector.connect(user=self.user, password=self.password, database=self.database, warehouse=self.warehouse, schema=self.schema, account=self.account)
            self.connection.autocommit = True
            return DBHandlerStatus(status=True)
        except snowflake.connector.errors.Error as e:
            return DBHandlerStatus(status=False, error=str(e))

    def disconnect(self):
        """
        Disconnect from the database.
        """
        if self.connection:
            self.connection.close()

    def check_connection(self) -> DBHandlerStatus:
        """
        Method for checking the status of database connection.
        Returns:
          DBHandlerStatus
        """
        if self.connection:
            return DBHandlerStatus(status=True)
        else:
            return DBHandlerStatus(status=False, error='Not connected to the database.')

    def get_tables(self) -> DBHandlerResponse:
        """
        Method to get the list of tables from database.
        Returns:
          DBHandlerStatus
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = f"SELECT table_name as table_name FROM information_schema.tables WHERE table_schema='{self.schema}'"
            cursor = self.connection.cursor()
            cursor.execute(query)
            tables_df = self._fetch_results_as_df(cursor)
            return DBHandlerResponse(data=tables_df)
        except snowflake.connector.errors.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def get_columns(self, table_name: str) -> DBHandlerResponse:
        """
        Method to retrieve the columns of the specified table from the database.
        Args:
          table_name (str): name of the table whose columns are to be retrieved.
        Returns:
          DBHandlerStatus
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = f"SELECT column_name as name, data_type as dtype FROM information_schema.columns WHERE table_name='{table_name}'"
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns_df = self._fetch_results_as_df(cursor)
            columns_df['dtype'] = columns_df['dtype'].apply(self._snowflake_to_python_types)
            return DBHandlerResponse(data=columns_df)
        except snowflake.connector.errors.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _fetch_results_as_df(self, cursor):
        """
        Fetch results from the cursor for the executed query and return the
        query results as dataframe.
        """
        try:
            res = cursor.fetchall()
            res_df = pd.DataFrame(res, columns=[desc[0].lower() for desc in cursor.description])
            return res_df
        except snowflake.connector.errors.ProgrammingError as e:
            if str(e) == 'no results to fetch':
                return pd.DataFrame({'status': ['success']})
            raise e

    def execute_native_query(self, query_string: str) -> DBHandlerResponse:
        """
        Executes the native query on the database.
        Args:
            query_string (str): query in native format
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            cursor = self.connection.cursor()
            cursor.execute(query_string)
            return DBHandlerResponse(data=self._fetch_results_as_df(cursor))
        except snowflake.connector.errors.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _snowflake_to_python_types(self, snowflake_type: str):
        mapping = {'TEXT': str, 'NUMBER': int, 'INT': int, 'DECIMAL': float, 'STRING': str, 'CHAR': str, 'BOOLEAN': bool, 'BINARY': bytes, 'DATE': datetime.date, 'TIME': datetime.time, 'TIMESTAMP': datetime.datetime}
        if snowflake_type in mapping:
            return mapping[snowflake_type]
        else:
            raise Exception(f'Unsupported column {snowflake_type} encountered in the snowflake. Please raise a feature request!')

class PostgresHandler(DBHandler):

    def __init__(self, name: str, **kwargs):
        """
        Initialize the handler.
        Args:
            name (str): name of the DB handler instance
            **kwargs: arbitrary keyword arguments for establishing the connection.
        """
        super().__init__(name)
        self.host = kwargs.get('host')
        self.port = kwargs.get('port')
        self.user = kwargs.get('user')
        self.password = kwargs.get('password')
        self.database = kwargs.get('database')
        self.connection = None

    def connect(self) -> DBHandlerStatus:
        """
        Set up the connection required by the handler.
        Returns:
            DBHandlerStatus
        """
        try:
            self.connection = psycopg2.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.database)
            self.connection.autocommit = True
            return DBHandlerStatus(status=True)
        except psycopg2.Error as e:
            return DBHandlerStatus(status=False, error=str(e))

    def disconnect(self):
        """
        Close any existing connections.
        """
        if self.connection:
            self.connection.close()

    def get_sqlalchmey_uri(self) -> str:
        return f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'

    def check_connection(self) -> DBHandlerStatus:
        """
        Check connection to the handler.
        Returns:
            DBHandlerStatus
        """
        if self.connection:
            return DBHandlerStatus(status=True)
        else:
            return DBHandlerStatus(status=False, error='Not connected to the database.')

    def get_tables(self) -> DBHandlerResponse:
        """
        Return the list of tables in the database.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = "SELECT table_name FROM information_schema.tables WHERE table_schema NOT IN ('information_schema', 'pg_catalog')"
            tables_df = pd.read_sql_query(query, self.connection)
            return DBHandlerResponse(data=tables_df)
        except psycopg2.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def get_columns(self, table_name: str) -> DBHandlerResponse:
        """
        Returns the list of columns for the given table.
        Args:
            table_name (str): name of the table whose columns are to be retrieved.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = f"SELECT column_name as name, data_type as dtype, udt_name FROM information_schema.columns WHERE table_name='{table_name}'"
            columns_df = pd.read_sql_query(query, self.connection)
            columns_df['dtype'] = columns_df.apply(lambda x: self._pg_to_python_types(x['dtype'], x['udt_name']), axis=1)
            return DBHandlerResponse(data=columns_df)
        except psycopg2.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _fetch_results_as_df(self, cursor):
        """
        This is currently the only clean solution that we have found so far.
        Reference to Postgres API: https://www.psycopg.org/docs/cursor.html#fetch

        In short, currently there is no very clean programming way to differentiate
        CREATE, INSERT, SELECT. CREATE and INSERT do not return any result, so calling
        fetchall() on those will yield a programming error. Cursor has an attribute
        rowcount, but it indicates # of rows that are affected. In that case, for both
        INSERT and SELECT rowcount is not 0, so we also cannot use this API to
        differentiate INSERT and SELECT.
        """
        try:
            res = cursor.fetchall()
            res_df = pd.DataFrame(res, columns=[desc[0].lower() for desc in cursor.description])
            return res_df
        except psycopg2.ProgrammingError as e:
            if str(e) == 'no results to fetch':
                return pd.DataFrame({'status': ['success']})
            raise e

    def execute_native_query(self, query_string: str) -> DBHandlerResponse:
        """
        Executes the native query on the database.
        Args:
            query_string (str): query in native format
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            cursor = self.connection.cursor()
            cursor.execute(query_string)
            return DBHandlerResponse(data=self._fetch_results_as_df(cursor))
        except psycopg2.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _pg_to_python_types(self, pg_type: str, udt_name: str):
        primitive_type_mapping = {'integer': int, 'bigint': int, 'smallint': int, 'numeric': float, 'real': float, 'double precision': float, 'character': str, 'character varying': str, 'text': str, 'boolean': bool}
        user_defined_type_mapping = {'vector': np.ndarray}
        if pg_type in primitive_type_mapping:
            return primitive_type_mapping[pg_type]
        elif pg_type == 'USER-DEFINED' and udt_name in user_defined_type_mapping:
            return user_defined_type_mapping[udt_name]
        else:
            raise Exception(f'Unsupported column {pg_type} encountered in the postgres table. Please raise a feature request!')

class SQLiteHandler(DBHandler):

    def __init__(self, name: str, **kwargs):
        """
        Initialize the handler.
        Args:
            name (str): name of the DB handler instance
            **kwargs: arbitrary keyword arguments for establishing the connection.
        """
        super().__init__(name)
        self.database = kwargs.get('database')
        self.connection = None

    def connect(self):
        """
        Set up the connection required by the handler.
        Returns:
            DBHandlerStatus
        """
        try:
            self.connection = sqlite3.connect(database=self.database, isolation_level=None)
            return DBHandlerStatus(status=True)
        except sqlite3.Error as e:
            return DBHandlerStatus(status=False, error=str(e))

    def disconnect(self):
        """
        Close any existing connections.
        """
        if self.connection:
            self.connection.close()

    def get_sqlalchmey_uri(self) -> str:
        return f'sqlite:///{self.database}'

    def check_connection(self) -> DBHandlerStatus:
        """
        Check connection to the handler.
        Returns:
            DBHandlerStatus
        """
        if self.connection:
            return DBHandlerStatus(status=True)
        else:
            return DBHandlerStatus(status=False, error='Not connected to the database.')

    def get_tables(self) -> DBHandlerResponse:
        """
        Return the list of tables in the database.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = "SELECT name AS table_name FROM sqlite_master WHERE type = 'table'"
            tables_df = pd.read_sql_query(query, self.connection)
            return DBHandlerResponse(data=tables_df)
        except sqlite3.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def get_columns(self, table_name: str) -> DBHandlerResponse:
        """
        Returns the list of columns for the given table.
        Args:
            table_name (str): name of the table whose columns are to be retrieved.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        '\n        SQLite does not provide an in-built way to get the column names using a SELECT statement.\n        Hence we have to use the PRAGMA command and filter the required columns.\n        '
        try:
            query = f"PRAGMA table_info('{table_name}')"
            pragma_df = pd.read_sql_query(query, self.connection)
            columns_df = pragma_df[['name', 'type']].copy()
            columns_df.rename(columns={'type': 'dtype'}, inplace=True)
            columns_df['dtype'] = columns_df['dtype'].apply(self._sqlite_to_python_types)
            return DBHandlerResponse(data=columns_df)
        except sqlite3.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _fetch_results_as_df(self, cursor):
        try:
            res = cursor.fetchall()
            res_df = pd.DataFrame(res, columns=[desc[0].lower() for desc in cursor.description] if cursor.description else [])
            return res_df
        except sqlite3.ProgrammingError as e:
            if str(e) == 'no results to fetch':
                return pd.DataFrame({'status': ['success']})
            raise e

    def execute_native_query(self, query_string: str) -> DBHandlerResponse:
        """
        Executes the native query on the database.
        Args:
            query_string (str): query in native format
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            cursor = self.connection.cursor()
            cursor.execute(query_string)
            return DBHandlerResponse(data=self._fetch_results_as_df(cursor))
        except sqlite3.Error as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _sqlite_to_python_types(self, sqlite_type: str):
        mapping = {'INT': int, 'INTEGER': int, 'TINYINT': int, 'SMALLINT': int, 'MEDIUMINT': int, 'BIGINT': int, 'UNSIGNED BIG INT': int, 'INT2': int, 'INT8': int, 'CHARACTER': str, 'VARCHAR': str, 'VARYING CHARACTER': str, 'NCHAR': str, 'NATIVE CHARACTER': str, 'NVARCHAR': str, 'TEXT': str, 'CLOB': str, 'BLOB': bytes, 'REAL': float, 'DOUBLE': float, 'DOUBLE PRECISION': float, 'FLOAT': float, 'NUMERIC': float, 'DECIMAL': float, 'BOOLEAN': bool, 'DATE': datetime.date, 'DATETIME': datetime.datetime}
        sqlite_type = sqlite_type.split('(')[0].strip().upper()
        if sqlite_type in mapping:
            return mapping[sqlite_type]
        else:
            raise Exception(f'Unsupported column {sqlite_type} encountered in the sqlite table. Please raise a feature request!')

class GithubHandler(DBHandler):

    def __init__(self, name: str, **kwargs):
        """
        Initialize the handler.
        Args:
            name (str): name of the DB handler instance
            **kwargs: arbitrary keyword arguments for establishing the connection.
        """
        super().__init__(name)
        self.owner = kwargs.get('owner', '')
        self.repo = kwargs.get('repo', '')
        self.github_token = kwargs.get('github_token', '')

    @property
    def supported_table(self):

        def _stargazer_generator():
            for stargazer in self.connection.get_repo('{}/{}'.format(self.owner, self.repo)).get_stargazers():
                yield {property_name: getattr(stargazer, property_name) for property_name, _ in STARGAZERS_COLUMNS}
        mapping = {'stargazers': {'columns': STARGAZERS_COLUMNS, 'generator': _stargazer_generator()}}
        return mapping

    def connect(self):
        """
        Set up the connection required by the handler.
        Returns:
            DBHandlerStatus
        """
        try:
            if self.github_token:
                self.connection = github.Github(self.github_token)
            else:
                self.connection = github.Github()
            return DBHandlerStatus(status=True)
        except Exception as e:
            return DBHandlerStatus(status=False, error=str(e))

    def disconnect(self):
        """
        Close any existing connections.
        """
        pass

    def check_connection(self) -> DBHandlerStatus:
        """
        Check connection to the handler.
        Returns:
            DBHandlerStatus
        """
        if self.connection:
            return DBHandlerStatus(status=True)
        else:
            return DBHandlerStatus(status=False, error='Not connected to the database.')

    def get_tables(self) -> DBHandlerResponse:
        """
        Return the list of tables in the database.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            tables_df = pd.DataFrame(list(self.supported_table.keys()), columns=['table_name'])
            return DBHandlerResponse(data=tables_df)
        except Exception as e:
            return DBHandlerResponse(data=None, error=str(e))

    def get_columns(self, table_name: str) -> DBHandlerResponse:
        """
        Returns the list of columns for the given table.
        Args:
            table_name (str): name of the table whose columns are to be retrieved.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            columns_df = pd.DataFrame(self.supported_table[table_name]['columns'], columns=['name', 'dtype'])
            return DBHandlerResponse(data=columns_df)
        except Exception as e:
            return DBHandlerResponse(data=None, error=str(e))

    def select(self, table_name: str) -> DBHandlerResponse:
        """
        Returns a generator that yields the data from the given table.
        Args:
            table_name (str): name of the table whose data is to be retrieved.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            if table_name not in self.supported_table:
                return DBHandlerResponse(data=None, error='{} is not supported or does not exist.'.format(table_name))
            return DBHandlerResponse(data=None, data_generator=self.supported_table[table_name]['generator'])
        except Exception as e:
            return DBHandlerResponse(data=None, error=str(e))

class HackernewsSearchHandler(DBHandler):

    def connection():
        return requests.get('https://www.google.com/').status_code == 200

    def __init__(self, name: str, **kwargs):
        """
        Initialize the handler.
        Args:
            name (str): name of the DB handler instance
            **kwargs: arbitrary keyword arguments for establishing the connection.
        """
        super().__init__(name)
        self.query = kwargs.get('query', '')
        self.tags = kwargs.get('tags', '')

    @property
    def supported_table(self):

        def _hackernews_topics_generator():
            url = 'http://hn.algolia.com/api/v1/search?'
            url += 'query=' + self.query
            url += '&tags=' + ('story' if self.tags == '' else +self.tags)
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception('Could not reach website.')
            json_result = response.content
            dict_result = json.loads(json_result)
            for row in dict_result:
                yield {property_name: row[property_name] for property_name, _ in HACKERNEWS_COLUMNS}
        mapping = {'search_results': {'columns': HACKERNEWS_COLUMNS, 'generator': _hackernews_topics_generator()}}
        return mapping

    def connect(self):
        """
        Set up the connection required by the handler.
        Returns:
            DBHandlerStatus
        """
        return DBHandlerStatus(status=True)

    def disconnect(self):
        """
        Close any existing connections.
        """
        pass

    def check_connection(self) -> DBHandlerStatus:
        """
        Check connection to the handler.
        Returns:
            DBHandlerStatus
        """
        if self.connection():
            return DBHandlerStatus(status=True)
        else:
            return DBHandlerStatus(status=False, error='Not connected to the internet.')

    def get_tables(self) -> DBHandlerResponse:
        """
        Return the list of tables in the database.
        Returns:
            DBHandlerResponse
        """
        if not self.connection():
            return DBHandlerResponse(data=None, error='Not connected to the internet.')
        try:
            tables_df = pd.DataFrame(list(self.supported_table.keys()), columns=['table_name'])
            return DBHandlerResponse(data=tables_df)
        except Exception as e:
            return DBHandlerResponse(data=None, error=str(e))

    def get_columns(self, table_name: str) -> DBHandlerResponse:
        """
        Returns the list of columns for the given table.
        Args:
            table_name (str): name of the table whose columns are to be retrieved.
        Returns:
            DBHandlerResponse
        """
        if not self.connection():
            return DBHandlerResponse(data=None, error='Not connected to the internet.')
        try:
            columns_df = pd.DataFrame(self.supported_table[table_name]['columns'], columns=['name', 'dtype'])
            return DBHandlerResponse(data=columns_df)
        except Exception as e:
            return DBHandlerResponse(data=None, error=str(e))

    def select(self, table_name: str) -> DBHandlerResponse:
        """
        Returns a generator that yields the data from the given table.
        Args:
            table_name (str): name of the table whose data is to be retrieved.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            if table_name not in self.supported_table:
                return DBHandlerResponse(data=None, error='{} is not supported or does not exist.'.format(table_name))
            return DBHandlerResponse(data=None, data_generator=self.supported_table[table_name]['generator'])
        except Exception as e:
            return DBHandlerResponse(data=None, error=str(e))

class ClickHouseHandler(DBHandler):

    def __init__(self, name: str, **kwargs):
        """
        Initialize the handler.
        Args:
            name (str): name of the DB handler instance
            **kwargs: arbitrary keyword arguments for establishing the connection.
        """
        super().__init__(name)
        self.host = kwargs.get('host')
        self.port = kwargs.get('port')
        self.user = kwargs.get('user')
        self.password = kwargs.get('password')
        self.database = kwargs.get('database')
        self.protocol = kwargs.get('protocol')
        protocols_map = {'native': 'clickhouse+native', 'http': 'clickhouse+http', 'https': 'clickhouse+https'}
        if self.protocol in protocols_map:
            self.protocol = protocols_map[self.protocol]

    def connect(self):
        """
        Set up the connection required by the handler.
        Returns:
            DBHandlerStatus
        """
        try:
            protocol = self.protocol
            host = self.host
            port = self.port
            user = self.user
            password = self.password
            database = self.database
            url = f'{protocol}://{user}:{password}@{host}:{port}/{database}'
            if self.protocol == 'clickhouse+https':
                url = url + '?protocol=https'
            engine = create_engine(url)
            self.connection = engine.raw_connection()
            return DBHandlerStatus(status=True)
        except Exception as e:
            return DBHandlerStatus(status=False, error=str(e))

    def disconnect(self):
        """
        Close any existing connections.
        """
        if self.connection:
            self.disconnect()

    def check_connection(self) -> DBHandlerStatus:
        """
        Check connection to the handler.
        Returns:
            DBHandlerStatus
        """
        if self.connection:
            return DBHandlerStatus(status=True)
        else:
            return DBHandlerStatus(status=False, error='Not connected to the database.')

    def get_tables(self) -> DBHandlerResponse:
        """
        Return the list of tables in the database.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = f'SHOW TABLES FROM {self.connection_data['database']}'
            tables_df = pd.read_sql_query(query, self.connection)
            return DBHandlerResponse(data=tables_df)
        except Exception as e:
            return DBHandlerResponse(data=None, error=str(e))

    def get_columns(self, table_name: str) -> DBHandlerResponse:
        """
        Returns the list of columns for the given table.
        Args:
            table_name (str): name of the table whose columns are to be retrieved.
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            query = f'DESCRIBE {table_name}'
            columns_df = pd.read_sql_query(query, self.connection)
            columns_df['dtype'] = columns_df['dtype'].apply(self._clickhouse_to_python_types)
            return DBHandlerResponse(data=columns_df)
        except Exception as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _fetch_results_as_df(self, cursor):
        try:
            res = cursor.fetchall()
            if not res:
                return pd.DataFrame({'status': ['success']})
            res_df = pd.DataFrame(res, columns=[desc[0] for desc in cursor.description])
            return res_df
        except Exception as e:
            if str(e) == 'no results to fetch':
                return pd.DataFrame({'status': ['success']})
            raise e

    def execute_native_query(self, query_string: str) -> DBHandlerResponse:
        """
        Executes the native query on the database.
        Args:
            query_string (str): query in native format
        Returns:
            DBHandlerResponse
        """
        if not self.connection:
            return DBHandlerResponse(data=None, error='Not connected to the database.')
        try:
            cursor = self.connection.cursor()
            cursor.execute(query_string)
            return DBHandlerResponse(data=self._fetch_results_as_df(cursor))
        except Exception as e:
            return DBHandlerResponse(data=None, error=str(e))

    def _clickhouse_to_python_types(self, clickhouse_type: str):
        mapping = {'char': str, 'varchar': str, 'text': str, 'boolean': bool, 'integer': int, 'int': int, 'float': float, 'double': float}
        if clickhouse_type in mapping:
            return mapping[clickhouse_type]
        else:
            raise Exception(f'Unsupported column {clickhouse_type} encountered in the clickhouse table. Please raise a feature request!')

