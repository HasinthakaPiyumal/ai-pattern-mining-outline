# Cluster 6

class UtteranceDiff(BaseModel):
    """A list of Utterance objects that represent the differences between local and
    remote utterances.
    """
    diff: List[Utterance]

    @classmethod
    def from_utterances(cls, local_utterances: List[Utterance], remote_utterances: List[Utterance]):
        """Create a UtteranceDiff object from two lists of Utterance objects.

        :param local_utterances: A list of Utterance objects.
        :type local_utterances: List[Utterance]
        :param remote_utterances: A list of Utterance objects.
        :type remote_utterances: List[Utterance]
        """
        local_utterances_map = {x.to_str(include_metadata=True): x for x in local_utterances}
        remote_utterances_map = {x.to_str(include_metadata=True): x for x in remote_utterances}
        local_utterances_str = list(local_utterances_map.keys())
        local_utterances_str.sort()
        remote_utterances_str = list(remote_utterances_map.keys())
        remote_utterances_str.sort()
        differ = Differ()
        diff_obj = list(differ.compare(local_utterances_str, remote_utterances_str))
        utterance_diffs = []
        for line in diff_obj:
            utterance_str = line[2:]
            utterance_diff_tag = line[0]
            if utterance_diff_tag == '?':
                continue
            utterance = remote_utterances_map[utterance_str] if utterance_diff_tag == '+' else local_utterances_map[utterance_str]
            utterance.diff_tag = utterance_diff_tag
            utterance_diffs.append(utterance)
        return UtteranceDiff(diff=utterance_diffs)

    def to_utterance_str(self, include_metadata: bool=False) -> List[str]:
        """Outputs the utterance diff as a list of diff strings. Returns a list
        of strings showing what is different in the remote when compared to the
        local. For example:

        ["  route1: utterance1",
         "  route1: utterance2",
         "- route2: utterance3",
         "- route2: utterance4"]

        Tells us that the remote is missing "route2: utterance3" and "route2:
        utterance4", which do exist locally. If we see:

        ["  route1: utterance1",
         "  route1: utterance2",
         "+ route2: utterance3",
         "+ route2: utterance4"]

        This diff tells us that the remote has "route2: utterance3" and
        "route2: utterance4", which do not exist locally.

        :param include_metadata: Whether to include metadata in the string.
        :type include_metadata: bool
        :return: A list of diff strings.
        :rtype: List[str]
        """
        return [x.to_diff_str(include_metadata=include_metadata) for x in self.diff]

    def get_tag(self, diff_tag: str) -> List[Utterance]:
        """Get all utterances with a given diff tag.

        :param diff_tag: The diff tag to filter by. Must be one of "+", "-", or " ".
        :type diff_tag: str
        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        if diff_tag not in ['+', '-', ' ']:
            raise ValueError("diff_tag must be one of '+', '-', or ' '")
        return [x for x in self.diff if x.diff_tag == diff_tag]

    def get_sync_strategy(self, sync_mode: str) -> dict:
        """Generates the optimal synchronization plan for local and remote instances.

        :param sync_mode: The mode to sync the routes with the remote index.
        :type sync_mode: str
        :return: A dictionary describing the synchronization strategy.
        :rtype: dict
        """
        if sync_mode not in SYNC_MODES:
            raise ValueError(f'sync_mode must be one of {SYNC_MODES}')
        local_only = self.get_tag('-')
        local_only_mapper = {utt.route: (utt.function_schemas, utt.metadata) for utt in local_only}
        remote_only = self.get_tag('+')
        remote_only_mapper = {utt.route: (utt.function_schemas, utt.metadata) for utt in remote_only}
        local_and_remote = self.get_tag(' ')
        if sync_mode == 'error':
            if len(local_only) > 0 or len(remote_only) > 0:
                raise ValueError('There are utterances that exist in the local or remote instance that do not exist in the other instance. Please sync the routes before running this command.')
            else:
                return {'remote': {'upsert': [], 'delete': []}, 'local': {'upsert': [], 'delete': []}}
        elif sync_mode == 'local':
            return {'remote': {'upsert': local_only, 'delete': remote_only}, 'local': {'upsert': [], 'delete': []}}
        elif sync_mode == 'remote':
            return {'remote': {'upsert': [], 'delete': []}, 'local': {'upsert': remote_only, 'delete': local_only}}
        elif sync_mode == 'merge-force-local':
            local_route_names = set([utt.route for utt in local_only])
            local_route_utt_strs = set([utt.to_str() for utt in local_only])
            remote_to_keep = [utt for utt in remote_only if utt.route in local_route_names and utt.to_str() not in local_route_utt_strs]
            logger.info(f'local_only_mapper: {local_only_mapper}')
            remote_to_update = [Utterance(route=utt.route, utterance=utt.utterance, metadata=local_only_mapper[utt.route][1], function_schemas=local_only_mapper[utt.route][0]) for utt in remote_only if utt.route in local_only_mapper and (utt.metadata != local_only_mapper[utt.route][1] or utt.function_schemas != local_only_mapper[utt.route][0])]
            remote_to_keep = [Utterance(route=utt.route, utterance=utt.utterance, metadata=local_only_mapper[utt.route][1], function_schemas=local_only_mapper[utt.route][0]) for utt in remote_to_keep if utt.to_str() not in [x.to_str() for x in remote_to_update]]
            remote_to_delete = [utt for utt in remote_only if utt.route not in local_route_names]
            return {'remote': {'upsert': local_only + remote_to_update, 'delete': remote_to_delete}, 'local': {'upsert': remote_to_keep, 'delete': []}}
        elif sync_mode == 'merge-force-remote':
            remote_route_names = set([utt.route for utt in remote_only])
            remote_route_utt_strs = set([utt.to_str() for utt in remote_only])
            local_to_keep = [utt for utt in local_only if utt.route in remote_route_names and utt.to_str() not in remote_route_utt_strs]
            local_to_keep = [Utterance(route=utt.route, utterance=utt.utterance, metadata=remote_only_mapper[utt.route][1], function_schemas=remote_only_mapper[utt.route][0]) for utt in local_to_keep]
            local_to_delete = [utt for utt in local_only if utt.route not in remote_route_names]
            return {'remote': {'upsert': local_to_keep, 'delete': []}, 'local': {'upsert': remote_only, 'delete': local_to_delete}}
        elif sync_mode == 'merge':
            remote_only_updated = [Utterance(route=utt.route, utterance=utt.utterance, metadata=local_only_mapper[utt.route][1], function_schemas=local_only_mapper[utt.route][0]) if utt.route in local_only_mapper else utt for utt in remote_only]
            shared_updated = [Utterance(route=utt.route, utterance=utt.utterance, metadata=local_only_mapper[utt.route][1], function_schemas=local_only_mapper[utt.route][0]) for utt in local_and_remote if utt.route in local_only_mapper and (utt.metadata != local_only_mapper[utt.route][1] or utt.function_schemas != local_only_mapper[utt.route][0])]
            return {'remote': {'upsert': local_only + shared_updated + remote_only_updated, 'delete': []}, 'local': {'upsert': remote_only_updated + shared_updated, 'delete': []}}
        else:
            raise ValueError(f'sync_mode must be one of {SYNC_MODES}')

class BaseLLM(BaseModel):
    """Base class for LLMs typically used by dynamic routes.

    This class provides a base implementation for LLMs. It defines the common
    configuration and methods for all LLM classes.
    """
    name: str
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, name: str, **kwargs):
        """Initialize the BaseLLM.

        :param name: The name of the LLM.
        :type name: str
        :param **kwargs: Additional keyword arguments for the LLM.
        :type **kwargs: dict
        """
        super().__init__(name=name, **kwargs)

    def __call__(self, messages: List[Message]) -> Optional[str]:
        """Call the LLM.

        Must be implemented by subclasses.

        :param messages: The messages to pass to the LLM.
        :type messages: List[Message]
        :return: The response from the LLM.
        :rtype: Optional[str]
        """
        raise NotImplementedError('Subclasses must implement this method')

    def _check_for_mandatory_inputs(self, inputs: dict[str, Any], mandatory_params: List[str]) -> bool:
        """Check for mandatory parameters in inputs.

        :param inputs: The inputs to check for mandatory parameters.
        :type inputs: dict[str, Any]
        :param mandatory_params: The mandatory parameters to check for.
        :type mandatory_params: List[str]
        :return: True if all mandatory parameters are present, False otherwise.
        :rtype: bool
        """
        for name in mandatory_params:
            if name not in inputs:
                logger.error(f'Mandatory input {name} missing from query')
                return False
        return True

    def _check_for_extra_inputs(self, inputs: dict[str, Any], all_params: List[str]) -> bool:
        """Check for extra parameters not defined in the signature.

        :param inputs: The inputs to check for extra parameters.
        :type inputs: dict[str, Any]
        :param all_params: The all parameters to check for.
        :type all_params: List[str]
        :return: True if all extra parameters are present, False otherwise.
        :rtype: bool
        """
        input_keys = set(inputs.keys())
        param_keys = set(all_params)
        if not input_keys.issubset(param_keys):
            extra_keys = input_keys - param_keys
            logger.error(f'Extra inputs provided that are not in the signature: {extra_keys}')
            return False
        return True

    def _is_valid_inputs(self, inputs: List[Dict[str, Any]], function_schemas: List[Dict[str, Any]]) -> bool:
        """Determine if the functions chosen by the LLM exist within the function_schemas,
        and if the input arguments are valid for those functions.

        :param inputs: The inputs to check for validity.
        :type inputs: List[Dict[str, Any]]
        :param function_schemas: The function schemas to check against.
        :type function_schemas: List[Dict[str, Any]]
        :return: True if the inputs are valid, False otherwise.
        :rtype: bool
        """
        try:
            if len(inputs) != 1:
                logger.error('Only one set of function inputs is allowed.')
                return False
            if len(function_schemas) != 1:
                logger.error('Only one function schema is allowed.')
                return False
            if not self._validate_single_function_inputs(inputs[0], function_schemas[0]):
                return False
            return True
        except Exception as e:
            logger.error(f'Input validation error: {str(e)}')
            return False

    def _validate_single_function_inputs(self, inputs: Dict[str, Any], function_schema: Dict[str, Any]) -> bool:
        """Validate the extracted inputs against the function schema.

        :param inputs: The inputs to validate.
        :type inputs: Dict[str, Any]
        :param function_schema: The function schema to validate against.
        :type function_schema: Dict[str, Any]
        :return: True if the inputs are valid, False otherwise.
        :rtype: bool
        """
        try:
            signature = function_schema['signature']
            param_info = [param.strip() for param in signature[1:-1].split(',')]
            mandatory_params = []
            all_params = []
            for info in param_info:
                parts = info.split('=')
                name_type_pair = parts[0].strip()
                if ':' in name_type_pair:
                    name, _ = name_type_pair.split(':')
                else:
                    name = name_type_pair
                all_params.append(name)
                if len(parts) == 1:
                    mandatory_params.append(name)
            if not self._check_for_mandatory_inputs(inputs, mandatory_params):
                return False
            if not self._check_for_extra_inputs(inputs, all_params):
                return False
            return True
        except Exception as e:
            logger.error(f'Single input validation error: {str(e)}')
            return False

    def _extract_parameter_info(self, signature: str) -> tuple[List[str], List[str]]:
        """Extract parameter names and types from the function signature.

        :param signature: The function signature to extract parameter names and types from.
        :type signature: str
        :return: A tuple of parameter names and types.
        :rtype: tuple[List[str], List[str]]
        """
        param_info = [param.strip() for param in signature[1:-1].split(',')]
        param_names = [info.split(':')[0].strip() for info in param_info]
        param_types = [info.split(':')[1].strip().split('=')[0].strip() for info in param_info]
        return (param_names, param_types)

    def extract_function_inputs(self, query: str, function_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract the function inputs from the query.

        :param query: The query to extract the function inputs from.
        :type query: str
        :param function_schemas: The function schemas to extract the function inputs from.
        :type function_schemas: List[Dict[str, Any]]
        :return: The function inputs.
        :rtype: List[Dict[str, Any]]
        """
        logger.info('Extracting function input...')
        prompt = f"""\nYou are an accurate and reliable computer program that only outputs valid JSON. \nYour task is to output JSON representing the input arguments of a Python function.\n\nThis is the Python function's schema:\n\n### FUNCTION_SCHEMAS Start ###\n\t{function_schemas}\n### FUNCTION_SCHEMAS End ###\n\nThis is the input query.\n\n### QUERY Start ###\n\t{query}\n### QUERY End ###\n\nThe arguments that you need to provide values for, together with their datatypes, are stated in "signature" in the FUNCTION_SCHEMAS.\nThe values these arguments must take are made clear by the QUERY.\nUse the FUNCTION_SCHEMAS "description" too, as this might provide helpful clues about the arguments and their values.\nReturn only JSON, stating the argument names and their corresponding values.\n\n### FORMATTING_INSTRUCTIONS Start ###\n\tReturn a respones in valid JSON format. Do not return any other explanation or text, just the JSON.\n\tThe JSON-Keys are the names of the arguments, and JSON-values are the values those arguments should take.\n### FORMATTING_INSTRUCTIONS End ###\n\n### EXAMPLE Start ###\n\t=== EXAMPLE_INPUT_QUERY Start ===\n\t\t"How is the weather in Hawaii right now in International units?"\n\t=== EXAMPLE_INPUT_QUERY End ===\n\t=== EXAMPLE_INPUT_SCHEMA Start ===\n\t\t{{\n\t\t\t"name": "get_weather",\n\t\t\t"description": "Useful to get the weather in a specific location",\n\t\t\t"signature": "(location: str, degree: str) -> str",\n\t\t\t"output": "<class 'str'>",\n\t\t}}\n\t=== EXAMPLE_INPUT_QUERY End ===\n\t=== EXAMPLE_OUTPUT Start ===\n\t\t{{\n\t\t\t"location": "Hawaii",\n\t\t\t"degree": "Celsius",\n\t\t}}\n\t=== EXAMPLE_OUTPUT End ===\n### EXAMPLE End ###\n\nNote: I will tip $500 for an accurate JSON output. You will be penalized for an inaccurate JSON output.\n\nProvide JSON output now:\n"""
        llm_input = [Message(role='user', content=prompt)]
        output = self(llm_input)
        if not output:
            raise Exception('No output generated for extract function input')
        output = output.replace("'", '"').strip().rstrip(',')
        logger.info(f'LLM output: {output}')
        function_inputs = json.loads(output)
        if not isinstance(function_inputs, list):
            function_inputs = [function_inputs]
        logger.info(f'Function inputs: {function_inputs}')
        if not self._is_valid_inputs(function_inputs, function_schemas):
            raise ValueError('Invalid inputs')
        return function_inputs

class PineconeIndex(BaseIndex):
    index_prefix: str = 'semantic-router--'
    api_key: Optional[str] = None
    index_name: str = 'index'
    dimensions: Union[int, None] = None
    metric: str = 'dotproduct'
    cloud: str = 'aws'
    region: str = 'us-east-1'
    host: str = ''
    client: Any = Field(default=None, exclude=True)
    index: Optional[Any] = Field(default=None, exclude=True)
    ServerlessSpec: Any = Field(default=None, exclude=True)
    namespace: Optional[str] = ''
    base_url: Optional[str] = None
    headers: dict[str, str] = {}
    index_host: Optional[str] = 'http://localhost:5080'
    init_async_index: bool = False

    def __init__(self, api_key: Optional[str]=None, index_name: str='index', dimensions: Optional[int]=None, metric: str='dotproduct', cloud: str='aws', region: str='us-east-1', host: str='', namespace: Optional[str]='', base_url: Optional[str]='https://api.pinecone.io', init_async_index: bool=False):
        """Initialize PineconeIndex.

        :param api_key: Pinecone API key.
        :type api_key: Optional[str]
        :param index_name: Name of the index.
        :type index_name: str
        :param dimensions: Dimensions of the index.
        :type dimensions: Optional[int]
        :param metric: Metric of the index.
        :type metric: str
        :param cloud: Cloud provider of the index.
        :type cloud: str
        :param region: Region of the index.
        :type region: str
        :param host: Host of the index.
        :type host: str
        :param namespace: Namespace of the index.
        :type namespace: Optional[str]
        :param base_url: Base URL of the Pinecone API.
        :type base_url: Optional[str]
        :param init_async_index: Whether to initialize the index asynchronously.
        :type init_async_index: bool
        """
        super().__init__()
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise ValueError('Pinecone API key is required.')
        self.headers = {'Api-Key': self.api_key, 'Content-Type': 'application/json', 'User-Agent': 'source_tag=semanticrouter'}
        if base_url is not None or os.getenv('PINECONE_API_BASE_URL'):
            logger.info('Using pinecone remote API.')
            if os.getenv('PINECONE_API_BASE_URL'):
                self.base_url = os.getenv('PINECONE_API_BASE_URL')
            else:
                self.base_url = base_url
        if self.base_url and 'api.pinecone.io' in self.base_url:
            self.headers['X-Pinecone-API-Version'] = '2024-07'
        self.index_name = index_name
        self.dimensions = dimensions
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.host = host
        if namespace == 'sr_config':
            raise ValueError("Namespace 'sr_config' is reserved for internal use.")
        self.namespace = namespace
        self.type = 'pinecone'
        logger.warning('Default region changed from us-west-2 to us-east-1 in v0.1.0.dev6')
        self.client = self._initialize_client(api_key=self.api_key)
        if not init_async_index:
            self.index = self._init_index()

    def _initialize_client(self, api_key: Optional[str]=None):
        """Initialize the Pinecone client.

        :param api_key: Pinecone API key.
        :type api_key: Optional[str]
        :return: Pinecone client.
        :rtype: Pinecone
        """
        try:
            from pinecone import Pinecone, ServerlessSpec
            self.ServerlessSpec = ServerlessSpec
        except ImportError:
            raise ImportError("Please install pinecone-client to use PineconeIndex. You can install it with: `pip install 'semantic-router[pinecone]'`")
        pinecone_args = {'api_key': api_key, 'source_tag': 'semanticrouter', 'host': self.base_url}
        if self.namespace:
            pinecone_args['namespace'] = self.namespace
        return Pinecone(**pinecone_args)

    def _calculate_index_host(self):
        """Calculate the index host. Used to differentiate between normal
        Pinecone and Pinecone Local instance.

        :return: None
        :rtype: None
        """
        if self.index_host and self.base_url:
            if 'api.pinecone.io' in self.base_url:
                if not self.index_host.startswith('http'):
                    self.index_host = f'https://{self.index_host}'
            elif 'http' not in self.index_host:
                self.index_host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.index_host.split(':')[-1]}'
            elif not self.index_host.startswith('http://'):
                if 'localhost' in self.index_host:
                    self.index_host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.index_host.split(':')[-1]}'
                else:
                    self.index_host = f'http://{self.index_host}'

    def _init_index(self, force_create: bool=False) -> Union[Any, None]:
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        dimensions_given = self.dimensions is not None
        if self.index is None:
            index_exists = self.client.has_index(name=self.index_name)
            if dimensions_given and (not index_exists):
                self.client.create_index(name=self.index_name, dimension=self.dimensions, metric=self.metric, spec=self.ServerlessSpec(cloud=self.cloud, region=self.region))
                while not self.client.describe_index(self.index_name).status['ready']:
                    time.sleep(0.2)
                index = self.client.Index(self.index_name)
                self.index = index
                time.sleep(0.2)
            elif index_exists:
                self.index_host = self.client.describe_index(self.index_name).host
                self._calculate_index_host()
                index = self.client.Index(self.index_name, host=self.index_host)
                self.index = index
                self.dimensions = index.describe_index_stats()['dimension']
            elif force_create and (not dimensions_given):
                raise ValueError('Cannot create an index without specifying the dimensions.')
            else:
                logger.warning(f'Index could not be initialized. Init parameters: self.index_name={self.index_name!r}, self.dimensions={self.dimensions!r}, self.metric={self.metric!r}, self.cloud={self.cloud!r}, self.region={self.region!r}, self.host={self.host!r}, self.namespace={self.namespace!r}, force_create={force_create!r}')
                index = None
        else:
            index = self.index
        if self.index is not None and self.host == '':
            self.index_host = self.client.describe_index(self.index_name).host
            if self.index_host and self.base_url:
                self._calculate_index_host()
                index = self.client.Index(self.index_name, host=self.index_host)
                self.host = self.index_host
        return index

    async def _init_async_index(self, force_create: bool=False):
        """Initializing the index can be done after the object has been created
        to allow for the user to set the dimensions and other parameters.

        If the index doesn't exist and the dimensions are given, the index will
        be created. If the index exists, it will be returned. If the index doesn't
        exist and the dimensions are not given, the index will not be created and
        None will be returned.

        This method is used to initialize the index asynchronously.

        :param force_create: If True, the index will be created even if the
            dimensions are not given (which will raise an error).
        :type force_create: bool, optional
        """
        index_stats = None
        if self.dimensions is None:
            indexes = await self._async_list_indexes()
            index_names = [i['name'] for i in indexes['indexes']]
            index_exists = self.index_name in index_names
            if index_exists:
                index_stats = await self._async_describe_index(self.index_name)
                self.dimensions = index_stats['dimension']
            elif index_exists and (not force_create):
                logger.warning(f'Index could not be initialized. Init parameters: self.index_name={self.index_name!r}, self.dimensions={self.dimensions!r}, self.metric={self.metric!r}, self.cloud={self.cloud!r}, self.region={self.region!r}, self.host={self.host!r}, self.namespace={self.namespace!r}, force_create={force_create!r}')
            elif force_create:
                raise ValueError(f'Index could not be initialized. Init parameters: self.index_name={self.index_name!r}, self.dimensions={self.dimensions!r}, self.metric={self.metric!r}, ')
            else:
                raise NotImplementedError('Unexpected init conditions. Please report this issue in GitHub.')
        if self.dimensions:
            indexes = await self._async_list_indexes()
            index_names = [i['name'] for i in indexes['indexes']]
            index_exists = self.index_name in index_names
            if not index_exists:
                index_stats = await self._async_describe_index(self.index_name)
                index_status = index_stats.get('status', {})
                index_ready = index_status.get('ready', False) if isinstance(index_status, dict) else False
                if index_ready == 'true' or (isinstance(index_ready, bool) and index_ready):
                    self.index_host = index_stats['host']
                    self._calculate_index_host()
                    self.host = self.index_host
                    return index_stats
                else:
                    await self._async_create_index(name=self.index_name, dimension=self.dimensions, metric=self.metric, cloud=self.cloud, region=self.region)
                    index_ready = 'false'
                    while not (index_ready == 'true' or (isinstance(index_ready, bool) and index_ready)):
                        index_stats = await self._async_describe_index(self.index_name)
                        index_status = index_stats.get('status', {})
                        index_ready = index_status.get('ready', False) if isinstance(index_status, dict) else False
                        await asyncio.sleep(0.1)
                    self.index_host = index_stats['host']
                    self._calculate_index_host()
                    self.host = self.index_host
                    return index_stats
            else:
                index_stats = await self._async_describe_index(self.index_name)
                self.index_host = index_stats['host']
                self._calculate_index_host()
                self.host = self.index_host
                return index_stats
        if index_stats:
            self.index_host = index_stats['host']
            self._calculate_index_host()
            self.host = self.index_host
        else:
            self.host = ''

    def _batch_upsert(self, batch: List[Dict]):
        """Helper method for upserting a single batch of records.

        :param batch: The batch of records to upsert.
        :type batch: List[Dict]
        """
        if self.index is not None:
            self.index.upsert(vectors=batch, namespace=self.namespace)
        else:
            raise ValueError('Index is None, could not upsert.')

    def add(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[Optional[List[Dict[str, Any]]]]=None, metadata_list: List[Dict[str, Any]]=[], batch_size: int=100, sparse_embeddings: Optional[Optional[List[SparseEmbedding]]]=None, **kwargs):
        """Add vectors to Pinecone in batches.

        :param embeddings: List of embeddings to upsert.
        :type embeddings: List[List[float]]
        :param routes: List of routes to upsert.
        :type routes: List[str]
        :param utterances: List of utterances to upsert.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to upsert.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to upsert.
        :type metadata_list: List[Dict[str, Any]]
        :param batch_size: Number of vectors to upsert in a single batch.
        :type batch_size: int, optional
        :param sparse_embeddings: List of sparse embeddings to upsert.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        if self.index is None:
            self.dimensions = self.dimensions or len(embeddings[0])
            self.index = self._init_index(force_create=True)
        vectors_to_upsert = build_records(embeddings=embeddings, routes=routes, utterances=utterances, function_schemas=function_schemas, metadata_list=metadata_list, sparse_embeddings=sparse_embeddings)
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self._batch_upsert(batch)

    async def aadd(self, embeddings: List[List[float]], routes: List[str], utterances: List[str], function_schemas: Optional[Optional[List[Dict[str, Any]]]]=None, metadata_list: List[Dict[str, Any]]=[], batch_size: int=100, sparse_embeddings: Optional[Optional[List[SparseEmbedding]]]=None, **kwargs):
        """Add vectors to Pinecone in batches.

        :param embeddings: List of embeddings to upsert.
        :type embeddings: List[List[float]]
        :param routes: List of routes to upsert.
        :type routes: List[str]
        :param utterances: List of utterances to upsert.
        :type utterances: List[str]
        :param function_schemas: List of function schemas to upsert.
        :type function_schemas: Optional[List[Dict[str, Any]]]
        :param metadata_list: List of metadata to upsert.
        :type metadata_list: List[Dict[str, Any]]
        :param batch_size: Number of vectors to upsert in a single batch.
        :type batch_size: int, optional
        :param sparse_embeddings: List of sparse embeddings to upsert.
        :type sparse_embeddings: Optional[List[SparseEmbedding]]
        """
        vectors_to_upsert = build_records(embeddings=embeddings, routes=routes, utterances=utterances, function_schemas=function_schemas, metadata_list=metadata_list, sparse_embeddings=sparse_embeddings)
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            await self._async_upsert(vectors=batch, namespace=self.namespace or '')

    def _remove_and_sync(self, routes_to_delete: dict):
        """Remove specified routes from index if they exist.

        :param routes_to_delete: Routes to delete.
        :type routes_to_delete: dict
        """
        for route, utterances in routes_to_delete.items():
            remote_routes = self._get_routes_with_ids(route_name=route)
            ids_to_delete = [r['id'] for r in remote_routes if (r['route'], r['utterance']) in zip([route] * len(utterances), utterances)]
            if ids_to_delete and self.index:
                self.index.delete(ids=ids_to_delete, namespace=self.namespace)

    async def _async_remove_and_sync(self, routes_to_delete: dict):
        """Remove specified routes from index if they exist.

        This method is asyncronous.

        :param routes_to_delete: Routes to delete.
        :type routes_to_delete: dict
        """
        for route, utterances in routes_to_delete.items():
            remote_routes = await self._async_get_routes_with_ids(route_name=route)
            ids_to_delete = [r['id'] for r in remote_routes if (r['route'], r['utterance']) in zip([route] * len(utterances), utterances)]
            if ids_to_delete and self.index:
                await self._async_delete(ids=ids_to_delete, namespace=self.namespace or '')

    def _get_route_ids(self, route_name: str):
        """Get the IDs of the routes in the index.

        :param route_name: Name of the route to get the IDs for.
        :type route_name: str
        :return: List of IDs of the routes.
        :rtype: list[str]
        """
        clean_route = clean_route_name(route_name)
        ids, _ = self._get_all(prefix=f'{clean_route}#')
        return ids

    async def _async_get_route_ids(self, route_name: str):
        """Get the IDs of the routes in the index.

        :param route_name: Name of the route to get the IDs for.
        :type route_name: str
        :return: List of IDs of the routes.
        :rtype: list[str]
        """
        clean_route = clean_route_name(route_name)
        ids, _ = await self._async_get_all(prefix=f'{clean_route}#')
        return ids

    def _get_routes_with_ids(self, route_name: str):
        """Get the routes with their IDs from the index.

        :param route_name: Name of the route to get the routes with their IDs for.
        :type route_name: str
        :return: List of routes with their IDs.
        :rtype: list[dict]
        """
        clean_route = clean_route_name(route_name)
        ids, metadata = self._get_all(prefix=f'{clean_route}#', include_metadata=True)
        route_tuples = []
        for id, data in zip(ids, metadata):
            route_tuples.append({'id': id, 'route': data['sr_route'], 'utterance': data['sr_utterance']})
        return route_tuples

    async def _async_get_routes_with_ids(self, route_name: str):
        """Get the routes with their IDs from the index.

        :param route_name: Name of the route to get the routes with their IDs for.
        :type route_name: str
        :return: List of routes with their IDs.
        :rtype: list[dict]
        """
        clean_route = clean_route_name(route_name)
        ids, metadata = await self._async_get_all(prefix=f'{clean_route}#', include_metadata=True)
        route_tuples = []
        for id, data in zip(ids, metadata):
            route_tuples.append({'id': id, 'route': data['sr_route'], 'utterance': data['sr_utterance']})
        return route_tuples

    def _get_all(self, prefix: Optional[str]=None, include_metadata: bool=False):
        """Retrieves all vector IDs from the Pinecone index using pagination.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
        :rtype: tuple[list[str], list[dict]]
        """
        if self.index is None:
            raise ValueError('Index is None, could not retrieve vector IDs.')
        all_vector_ids = []
        metadata = []
        for ids in self.index.list(prefix=prefix, namespace=self.namespace):
            all_vector_ids.extend(ids)
            if include_metadata:
                for id in ids:
                    res_meta = self.index.fetch(ids=[id], namespace=self.namespace) if self.index else {}
                    metadata.extend([x['metadata'] for x in res_meta['vectors'].values()])
        return (all_vector_ids, metadata)

    def delete(self, route_name: str) -> list[str]:
        """Delete specified route from index if it exists. Returns the IDs of the vectors
        deleted.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        route_vec_ids = self._get_route_ids(route_name=route_name)
        if self.index is not None:
            logger.info('index is not None, deleting...')
            if self.base_url and 'api.pinecone.io' in self.base_url:
                self.index.delete(ids=route_vec_ids, namespace=self.namespace)
            else:
                response = requests.post(f'{self.index_host}/vectors/delete', json=DeleteRequest(ids=route_vec_ids, delete_all=True, namespace=self.namespace).model_dump(exclude_none=True), timeout=10)
                if response.status_code == 200:
                    logger.info(f'Deleted {len(route_vec_ids)} vectors from index {self.index_name}.')
                else:
                    error_message = response.text
                    raise Exception(f'Failed to delete vectors: {response.status_code} : {error_message}')
            return route_vec_ids
        else:
            raise ValueError('Index is None, could not delete.')

    async def adelete(self, route_name: str) -> list[str]:
        """Asynchronously delete specified route from index if it exists. Returns the IDs
        of the vectors deleted.

        :param route_name: Name of the route to delete.
        :type route_name: str
        :return: List of IDs of the vectors deleted.
        :rtype: list[str]
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        route_vec_ids = await self._async_get_route_ids(route_name=route_name)
        await self._async_delete(ids=route_vec_ids, namespace=self.namespace or '')
        return route_vec_ids

    def delete_all(self):
        """Delete all routes from index if it exists.

        :return: None
        :rtype: None
        """
        if self.index is not None:
            self.index.delete(delete_all=True, namespace=self.namespace)
        else:
            raise ValueError('Index is None, could not delete.')

    def describe(self) -> IndexConfig:
        """Describe the index.

        :return: IndexConfig
        :rtype: IndexConfig
        """
        if self.index is not None:
            stats = self.index.describe_index_stats()
            return IndexConfig(type=self.type, dimensions=stats['dimension'], vectors=stats['namespaces'][self.namespace]['vector_count'])
        else:
            return IndexConfig(type=self.type, dimensions=self.dimensions or 0, vectors=0)

    def is_ready(self) -> bool:
        """Checks if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        return self.index is not None

    def query(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Search the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param sparse_vector: An optional sparse vector to include in the query.
        :type sparse_vector: Optional[SparseEmbedding]
        :param kwargs: Additional keyword arguments for the query, including sparse_vector.
        :type kwargs: Any
        :return: A tuple containing an array of scores and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises ValueError: If the index is not populated.
        """
        if self.index is None:
            raise ValueError('Index is not populated.')
        query_vector_list = vector.tolist()
        if route_filter is not None:
            filter_query = {'sr_route': {'$in': route_filter}}
        else:
            filter_query = None
        if sparse_vector is not None:
            logger.error(f'sparse_vector exists:{sparse_vector}')
            if isinstance(sparse_vector, dict):
                sparse_vector = SparseEmbedding.from_dict(sparse_vector)
            if isinstance(sparse_vector, SparseEmbedding):
                sparse_vector = sparse_vector.to_pinecone()
        try:
            results = self.index.query(vector=[query_vector_list], sparse_vector=sparse_vector, top_k=top_k, filter=filter_query, include_metadata=True, namespace=self.namespace)
        except Exception:
            logger.error('retrying query with vector as str')
            results = self.index.query(vector=query_vector_list, sparse_vector=sparse_vector, top_k=top_k, filter=filter_query, include_metadata=True, namespace=self.namespace)
        scores = [result['score'] for result in results['matches']]
        route_names = [result['metadata']['sr_route'] for result in results['matches']]
        return (np.array(scores), route_names)

    def _read_config(self, field: str, scope: str | None=None) -> ConfigParameter:
        """Read a config parameter from the index.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        scope = scope or self.namespace
        if self.index is None:
            return ConfigParameter(field=field, value='', scope=scope)
        config_id = f'{field}#{scope}'
        config_record = self.index.fetch(ids=[config_id], namespace='sr_config')
        if config_record.get('vectors'):
            return ConfigParameter(field=field, value=config_record['vectors'][config_id]['metadata']['value'], created_at=config_record['vectors'][config_id]['metadata']['created_at'], scope=scope)
        else:
            logger.warning(f'Configuration for {field} parameter not found in index.')
            return ConfigParameter(field=field, value='', scope=scope)

    async def _async_read_config(self, field: str, scope: str | None=None) -> ConfigParameter:
        """Read a config parameter from the index asynchronously.

        :param field: The field to read.
        :type field: str
        :param scope: The scope to read.
        :type scope: str | None
        :return: The config parameter that was read.
        :rtype: ConfigParameter
        """
        scope = scope or self.namespace
        if self.index is None:
            return ConfigParameter(field=field, value='', scope=scope)
        config_id = f'{field}#{scope}'
        config_record = await self._async_fetch_metadata(vector_id=config_id, namespace='sr_config')
        if config_record:
            try:
                return ConfigParameter(field=field, value=config_record['value'], created_at=config_record['created_at'], scope=scope)
            except KeyError:
                raise ValueError(f'Found invalid config record during sync: {config_record}')
        else:
            logger.warning(f'Configuration for {field} parameter not found in index.')
            return ConfigParameter(field=field, value='', scope=scope)

    def _write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Method to write a config parameter to the remote Pinecone index.

        :param config: The config parameter to write to the index.
        :type config: ConfigParameter
        """
        config.scope = config.scope or self.namespace
        if self.index is None:
            raise ValueError('Index has not been initialized.')
        if self.dimensions is None:
            raise ValueError('Must set PineconeIndex.dimensions before writing config.')
        self.index.upsert(vectors=[config.to_pinecone(dimensions=self.dimensions)], namespace='sr_config')
        return config

    async def _async_write_config(self, config: ConfigParameter) -> ConfigParameter:
        """Method to write a config parameter to the remote Pinecone index.

        :param config: The config parameter to write to the index.
        :type config: ConfigParameter
        """
        config.scope = config.scope or self.namespace
        if self.dimensions is None:
            raise ValueError('Must set PineconeIndex.dimensions before writing config.')
        pinecone_config = config.to_pinecone(dimensions=self.dimensions)
        await self._async_upsert(vectors=[pinecone_config], namespace='sr_config')
        return config

    async def aquery(self, vector: np.ndarray, top_k: int=5, route_filter: Optional[List[str]]=None, sparse_vector: dict[int, float] | SparseEmbedding | None=None) -> Tuple[np.ndarray, List[str]]:
        """Asynchronously search the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: np.ndarray
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param route_filter: A list of route names to filter the search results, defaults to None.
        :type route_filter: Optional[List[str]], optional
        :param kwargs: Additional keyword arguments for the query, including sparse_vector.
        :type kwargs: Any
        :keyword sparse_vector: An optional sparse vector to include in the query.
        :type sparse_vector: Optional[dict]
        :return: A tuple containing an array of scores and a list of route names.
        :rtype: Tuple[np.ndarray, List[str]]
        :raises ValueError: If the index is not populated.
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        query_vector_list = vector.tolist()
        if route_filter is not None:
            filter_query = {'sr_route': {'$in': route_filter}}
        else:
            filter_query = None
        sparse_vector_obj: dict[str, Any] | None = None
        if sparse_vector is not None:
            if isinstance(sparse_vector, dict):
                sparse_vector_obj = SparseEmbedding.from_dict(sparse_vector)
            if isinstance(sparse_vector, SparseEmbedding):
                sparse_vector_obj = sparse_vector.to_pinecone()
        results = await self._async_query(vector=query_vector_list, sparse_vector=sparse_vector_obj, namespace=self.namespace or '', filter=filter_query, top_k=top_k, include_metadata=True)
        scores = [result['score'] for result in results['matches']]
        route_names = [result['metadata']['sr_route'] for result in results['matches']]
        return (np.array(scores), route_names)

    async def aget_routes(self) -> list[tuple]:
        """Asynchronously get a list of route and utterance objects currently
        stored in the index.

        :return: A list of (route_name, utterance) objects.
        :rtype: List[Tuple]
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        return await self._async_get_routes()

    def delete_index(self):
        """Delete the index.

        :return: None
        :rtype: None
        """
        self.client.delete_index(self.index_name)
        self.index = None

    async def adelete_index(self):
        """Asynchronously delete the index."""
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        if not self.base_url:
            raise ValueError('base_url is not set for PineconeIndex.')
        async with aiohttp.ClientSession() as session:
            async with session.delete(f'{self.base_url}/indexes/{self.index_name}', headers=self.headers) as response:
                res = await response.json(content_type=None)
                if response.status != 202:
                    raise Exception(f'Failed to delete index: {response.status}', res)
        self.host = ''
        return res

    async def _async_query(self, vector: list[float], sparse_vector: dict[str, Any] | None=None, namespace: str='', filter: Optional[dict]=None, top_k: int=5, include_metadata: bool=False):
        """Asynchronously query the index for the query vector and return the top_k results.

        :param vector: The query vector to search for.
        :type vector: list[float]
        :param sparse_vector: The sparse vector to search for.
        :type sparse_vector: dict[str, Any] | None
        :param namespace: The namespace to search for.
        :type namespace: str
        :param filter: The filter to search for.
        :type filter: Optional[dict]
        :param top_k: The number of top results to return, defaults to 5.
        :type top_k: int, optional
        :param include_metadata: Whether to include metadata in the results, defaults to False.
        :type include_metadata: bool, optional
        """
        params = {'vector': vector, 'sparse_vector': sparse_vector, 'namespace': namespace, 'filter': filter, 'top_k': top_k, 'include_metadata': include_metadata, 'topK': top_k, 'includeMetadata': include_metadata}
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        elif self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.host}/query', json=params, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f'Error in query response: {error_text}')
                    return {}
                try:
                    return await response.json(content_type=None)
                except JSONDecodeError as e:
                    logger.error(f'JSON decode error: {e}')
                    return {}

    async def ais_ready(self, client_only: bool=False) -> bool:
        """Checks if class attributes exist to be used for async operations.

        :param client_only: Whether to check only the client attributes. If False
            attributes will be checked for both client and index operations. If True
            only attributes for client operations will be checked. Defaults to False.
        :type client_only: bool, optional
        :return: True if the class attributes exist, False otherwise.
        :rtype: bool
        """
        if not (self.cloud or self.region or self.base_url):
            return False
        if not client_only:
            if not (self.index_name and self.dimensions and self.metric and self.host and (self.host != '')):
                await self._init_async_index()
                if not (self.index_name and self.dimensions and self.metric and self.host and (self.host != '')):
                    return False
        return True

    async def _async_list_indexes(self):
        """Asynchronously lists all indexes within the current Pinecone project.

        :return: List of indexes.
        :rtype: list[dict]
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.base_url}/indexes', headers=self.headers) as response:
                return await response.json(content_type=None)

    async def _async_upsert(self, vectors: list[dict], namespace: str=''):
        """Asynchronously upserts vectors into the index.

        :param vectors: The vectors to upsert.
        :type vectors: list[dict]
        :param namespace: The namespace to upsert the vectors into.
        :type namespace: str
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        params = {'vectors': vectors, 'namespace': namespace}
        if self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.host}/vectors/upsert', json=params, headers=self.headers) as response:
                res = await response.json(content_type=None)
                return res

    async def _async_create_index(self, name: str, dimension: int, cloud: str, region: str, metric: str='dotproduct'):
        """Asynchronously creates a new index in Pinecone.

        :param name: The name of the index to create.
        :type name: str
        :param dimension: The dimension of the index.
        :type dimension: int
        :param cloud: The cloud provider to create the index on.
        :type cloud: str
        :param region: The region to create the index in.
        :type region: str
        :param metric: The metric to use for the index, defaults to "dotproduct".
        :type metric: str, optional
        """
        params = {'name': name, 'dimension': dimension, 'metric': metric, 'spec': {'serverless': {'cloud': cloud, 'region': region}}, 'deletion_protection': 'disabled'}
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.base_url}/indexes', json=params, headers=self.headers) as response:
                return await response.json(content_type=None)

    async def _async_delete(self, ids: list[str], namespace: str=''):
        """Asynchronously deletes vectors from the index.

        :param ids: The IDs of the vectors to delete.
        :type ids: list[str]
        :param namespace: The namespace to delete the vectors from.
        :type namespace: str
        """
        params = {'ids': ids, 'namespace': namespace}
        if self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.host}/vectors/delete', json=params, headers=self.headers) as response:
                return await response.json(content_type=None)

    async def _async_describe_index(self, name: str):
        """Asynchronously describes the index.

        :param name: The name of the index to describe.
        :type name: str
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.base_url}/indexes/{name}', headers=self.headers) as response:
                return await response.json(content_type=None)

    async def _async_get_all(self, prefix: Optional[str]=None, include_metadata: bool=False) -> tuple[list[str], list[dict]]:
        """Retrieves all vector IDs from the Pinecone index using pagination
        asynchronously.

        :param prefix: The prefix to filter the vectors by.
        :type prefix: Optional[str]
        :param include_metadata: Whether to include metadata in the response.
        :type include_metadata: bool
        :return: A tuple containing a list of vector IDs and a list of metadata dictionaries.
        :rtype: tuple[list[str], list[dict]]
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        all_vector_ids = []
        next_page_token = None
        if prefix:
            prefix_str = f'?prefix={prefix}'
        else:
            prefix_str = ''
        if self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        list_url = f'{self.host}/vectors/list{prefix_str}'
        params: dict = {}
        if self.namespace:
            params['namespace'] = self.namespace
        metadata = []
        async with aiohttp.ClientSession() as session:
            while True:
                if next_page_token:
                    params['paginationToken'] = next_page_token
                async with session.get(list_url, params=params, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f'Error fetching vectors: {error_text}')
                        break
                    response_data = await response.json(content_type=None)
                vector_ids = [vec['id'] for vec in response_data.get('vectors', [])]
                if not vector_ids:
                    break
                all_vector_ids.extend(vector_ids)
                if include_metadata:
                    metadata_tasks = [self._async_fetch_metadata(id) for id in vector_ids]
                    metadata_results = await asyncio.gather(*metadata_tasks)
                    metadata.extend(metadata_results)
                next_page_token = response_data.get('pagination', {}).get('next')
                if not next_page_token:
                    break
        return (all_vector_ids, metadata)

    async def _async_fetch_metadata(self, vector_id: str, namespace: str | None=None) -> dict:
        """Fetch metadata for a single vector ID asynchronously using the
        ClientSession.

        :param vector_id: The ID of the vector to fetch metadata for.
        :type vector_id: str
        :param namespace: The namespace to fetch metadata for.
        :type namespace: str | None
        :return: A dictionary containing the metadata for the vector.
        :rtype: dict
        """
        if not await self.ais_ready():
            raise ValueError('Async index is not initialized.')
        if self.base_url and 'api.pinecone.io' in self.base_url:
            if not self.host.startswith('http'):
                logger.error(f'host exists:{self.host}')
                self.host = f'https://{self.host}'
        elif self.host.startswith('localhost') and self.base_url:
            self.host = f'http://{self.base_url.split(':')[-2].strip('/')}:{self.host.split(':')[-1]}'
        url = f'{self.host}/vectors/fetch'
        params = {'ids': [vector_id]}
        if namespace:
            params['namespace'] = [namespace]
        elif self.namespace:
            params['namespace'] = [self.namespace]
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f'Error fetching metadata: {error_text}')
                    return {}
                try:
                    response_data = await response.json(content_type=None)
                except Exception as e:
                    logger.warning(f'No metadata found for vector {vector_id}: {e}')
                    return {}
                return response_data.get('vectors', {}).get(vector_id, {}).get('metadata', {})

    def __len__(self):
        """Returns the total number of vectors in the index. If the index is not initialized
        returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        if self.index is None:
            logger.warning('Index is not initialized, returning 0')
            return 0
        namespace_stats = self.index.describe_index_stats()['namespaces'].get(self.namespace)
        if namespace_stats:
            return namespace_stats['vector_count']
        else:
            return 0

    async def alen(self):
        """Async version of __len__. Returns the total number of vectors in the index.
        If the index is not initialized, initializes it first or returns 0.

        :return: The total number of vectors.
        :rtype: int
        """
        if not await self.ais_ready():
            logger.warning('Index is not ready, returning 0')
            return 0
        namespace_stats = await self._async_describe_index_stats()
        if namespace_stats and 'namespaces' in namespace_stats:
            ns_stats = namespace_stats['namespaces'].get(self.namespace)
            if ns_stats:
                return ns_stats['vectorCount']
        return 0

    async def _async_describe_index_stats(self):
        """Async version of describe_index_stats.

        :return: Index statistics.
        :rtype: dict
        """
        url = f'{self.index_host}/describe_index_stats'
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json={'namespace': self.namespace}, timeout=aiohttp.ClientTimeout(total=300)) as response:
                response.raise_for_status()
                return await response.json()

class BM25Encoder(SparseEncoder, FittableMixin, AsymmetricSparseMixin):
    """BM25Encoder, running a vectorized version of ATIRE BM25 algorithm

    Concept:
    - BM25 uses scoring between queries & corpus to retrieve the most relevant documents ∈ corpus
    - most vector databases (VDB) store embedded documents and score them versus received queries for retrieval
    - we need to break up the BM25 formula into `encode_queries` and `encode_documents`, with the latter to be stored in VDB
    - dot product of `encode_queries(q)` and `encode_documents([D_0, D_1, ...])` is the BM25 score of the documents `[D_0, D_1, ...]` for the given query `q`
    - we train a BM25 encoder's normalization parameters on a sufficiently large corpus to capture target language distribution
    - these trained parameter allow us to balance TF & IDF of query & documents for retrieval (read more on how BM25 fixes issues with TF-IDF)

    ATIRE Paper: https://www.cs.otago.ac.nz/research/student-publications/atire-opensource.pdf
    Pinecone Implementation: https://github.com/pinecone-io/pinecone-text/blob/8399f9ff28c4652766c35165c0db9b0eff309077/pinecone_text/sparse/bm25_encoder.py

    :param k1: normalizer parameter that limits how much a single query term `q_i ∈ q` can affect score for document `D_n`
    :type k1: float
    :param b: normalizer parameter that balances the effect of a single document length compared to the average document length
    :type b: float
    :param corpus_size: number of documents in the trained corpus
    :type corpus_size: int, optional
    :param _avg_doc_len: float representing the average document length in the trained corpus
    :type _avg_doc_len: float, optional
    :param _documents_containing_word: (1, tokenizer.vocab_size) shaped array, denoting how many documents contain `token ∈ vocab`
    :type _documents_containing_word: class:`numpy.ndarray`, optional

    """
    type: str = 'sparse'
    k1: float = 1.5
    b: float = 0.75
    corpus_size: int | None = None
    _tokenizer: BaseTokenizer | None
    _avg_doc_len: np.float64 | float | None
    _documents_containing_word: np.ndarray | None

    def __init__(self, tokenizer: BaseTokenizer | None=None, name: str | None=None, k1: float=1.5, b: float=0.75, corpus_size: int | None=None, avg_doc_len: float | None=None, use_default_params: bool=True) -> None:
        if name is None:
            name = 'bm25'
        super().__init__(name=name)
        self.k1 = k1
        self.b = b
        self.corpus_size = corpus_size
        self._avg_doc_len = np.float64(avg_doc_len) if avg_doc_len else None
        if use_default_params and (not tokenizer):
            logger.info('Initializing default BM25 model parameters.')
            self._tokenizer = PretrainedTokenizer('google-bert/bert-base-uncased')
        elif tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            raise ValueError('Tokenizer not provided. Provide a tokenizer or set `use_default_params` to True')

    def _fit_validate(self, routes: List[Route]):
        if not isinstance(routes, list) or not isinstance(routes[0], Route):
            raise TypeError('`routes` parameter must be a list of Route objects.')

    def fit(self, routes: List[Route]) -> 'BM25Encoder':
        """Trains the encoder weights on the provided routes.

        :param routes: List of routes to train the encoder on.
        :type routes: List[Route]
        """
        if not self._tokenizer:
            raise ValueError('BM25 encoder not initialized. Provide a tokenizer or set `use_default_params` to True')
        self._fit_validate(routes)
        utterances = [utterance for route in routes for utterance in route.utterances]
        utterance_ids = self._tokenizer.tokenize(utterances, pad=True)
        corpus = self._tf(utterance_ids)
        self.corpus_size = len(utterances)
        doc_lengths = corpus.sum(axis=1)
        self._avg_doc_len = doc_lengths.mean()
        documents_containing_word = np.atleast_2d((corpus > 0).sum(axis=0))
        documents_containing_word[:, 0] *= 0
        self._documents_containing_word = documents_containing_word
        return self

    def _tf(self, docs: np.ndarray) -> np.ndarray:
        """Returns term frequency of query terms in trained corpus

        :param docs: 2D shaped array of each document's token ids
        :type docs: numpy.ndarray
        :return: Matrix where value @ (m, n) represents how many times token id `n` appears in document `m`
        :rtype: numpy.ndarray
        """
        if self._tokenizer is None:
            raise ValueError('Tokenizer not provided. Provide a tokenizer or set `use_default_params` to True')
        vocab_size = self._tokenizer.vocab_size
        bincount = partial(np.bincount, minlength=vocab_size)
        tf = np.apply_along_axis(bincount, 1, docs)
        tf[:, 0] *= 0
        return tf

    def _df(self, queries: np.ndarray) -> np.ndarray:
        """Returns the amount of times each token in the query appears in trained corpus

        This is done in a faster, vectorized way, instead of looping through each query

        :param queries: 2D shaped array of each query token ids
        :type queries: numpy.ndarray
        :return: Matrix where value @ (m, n) represents how many times token id `n` in query `m` appears in the trained corpus
        :rtype: numpy.ndarray
        """
        if self._documents_containing_word is None:
            raise ValueError('Encoder not fitted. `BM25Encoder.fit` a corpus, or `BM25Encoder.load` a pretrained encoder.')
        if self._tokenizer is None:
            raise ValueError('Tokenizer not provided. Provide a tokenizer or set `use_default_params` to True')
        n = queries.shape[0]
        row_indices = np.arange(n)[:, None]
        mask = np.zeros((n, self._tokenizer.vocab_size), dtype=bool)
        mask[row_indices, queries] = True
        query_df = mask * self._documents_containing_word
        return query_df

    def encode_queries(self, queries: list[str]) -> list[SparseEmbedding]:
        """Returns BM25 scores for queries using precomputed corpus scores.

        :param queries: List of queries to encode
        :type queries: list
        :return: BM25 scores for each query against the corpus
        :rtype: list[SparseEmbedding]
        """
        if self.corpus_size is None or self._avg_doc_len is None or self._documents_containing_word is None:
            raise ValueError('Encoder not fitted. Please `.fit` the model on a provided corpus or load a pretrained encoder')
        if not self._tokenizer:
            raise ValueError('BM25 encoder not initialized. Provide a tokenizer or set `use_default_params` to True')
        if queries == []:
            raise ValueError('No documents provided for encoding')
        queries_ids = self._tokenizer.tokenize(queries)
        df = self._df(queries_ids)
        N = self.corpus_size
        df = df + np.where(df > 0, 0.5, 0)
        idf = np.divide(N + 1, df, out=np.zeros_like(df), where=df != 0)
        idf = np.log(idf, out=np.zeros_like(df), where=df != 0)
        idf_norm = np.divide(idf, idf.sum(axis=1)[:, np.newaxis], out=np.zeros_like(idf), where=idf != 0)
        return self._array_to_sparse_embeddings(idf_norm)

    def encode_documents(self, documents: list[str], batch_size: int | None=None) -> list[SparseEmbedding]:
        """Returns document term frequency normed by itself & average trained corpus length
        (This is the right-hand side of the BM25 equation, which gets matmul-ed with the query IDF component)

        LaTeX: $\\frac{f(d_i, D)}{f(d_i, D) + k_1 \\times (1 - b + b \\times \\frac{|D|}{avgdl})}$
        where:
            f(d_i, D) is frequency of term `d_i ∈ D`
            |D| is the document length
            avgdl is average document length in trained corpus

        :param documents: List of queries to encode
        :type documents: list
        :return: Encoded queries (as either sparse or dict)
        :rtype: list[SparseEmbedding]
        """
        if self.corpus_size is None or self._avg_doc_len is None or self._documents_containing_word is None:
            raise ValueError('Encoder not fitted. Please `.fit` the model on a provided corpus or load a pretrained encoder')
        if not self._tokenizer:
            raise ValueError('BM25 encoder not initialized. Provide a tokenizer or set `use_default_params` to True')
        if documents == []:
            raise ValueError('No documents provided for encoding')
        batch_size = batch_size or len(documents)
        queries_ids = self._tokenizer.tokenize(documents, pad=True)
        tf = self._tf(queries_ids)
        tf_sum = tf.sum(axis=1)
        tf_normed = tf / (self.k1 * (1.0 - self.b * self.b * (tf_sum[:, np.newaxis] / self._avg_doc_len)) + tf)
        return self._array_to_sparse_embeddings(tf_normed)

    def model(self, docs: List[str]) -> list[SparseEmbedding]:
        """Encode documents using BM25, with different encoding for queries vs documents to be indexed.

        :param docs: List of documents to encode
        :param is_query: If True, use query encoding, else use document encoding
        :return: List of sparse embeddings
        """
        if not self._tokenizer:
            raise ValueError('Encoder not fitted. `BM25.index` a corpus, or `BM25.load` a pretrained encoder.')
        if self.corpus_size is None or self._avg_doc_len is None or self._documents_containing_word is None:
            raise ValueError('Encoder not fitted. Please `.fit` the model on a provided corpus or load a pretrained encoder')
        return self.encode_queries(docs)

    async def aencode_queries(self, docs: List[str]) -> List[SparseEmbedding]:
        return await asyncio.to_thread(lambda: self.encode_queries(docs))

    async def aencode_documents(self, docs: List[str]) -> List[SparseEmbedding]:
        return await asyncio.to_thread(lambda: self.encode_documents(docs))

    def __call__(self, docs: List[str]) -> List[SparseEmbedding]:
        return self.encode_queries(docs)

    async def acall(self, docs: List[Any]) -> List[SparseEmbedding]:
        return await asyncio.to_thread(lambda: self.__call__(docs))

class HFEndpointEncoder(DenseEncoder):
    """HFEndpointEncoder class to embeddings models using Huggingface's inference endpoints.

    The HFEndpointEncoder class is a subclass of DenseEncoder and utilizes a specified
    Huggingface endpoint to generate embeddings for given documents. It requires the URL
    of the Huggingface API endpoint and an API key for authentication. The class supports
    customization of the score threshold for filtering or processing the embeddings.

    Example usage:

    ```python
    from semantic_router.encoders import HFEndpointEncoder

    encoder = HFEndpointEncoder(
        huggingface_url="https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5",
        huggingface_api_key="your-hugging-face-api-key"
    )
    embeddings = encoder(["document1", "document2"])
    ```
    """
    name: str = 'hugging_face_custom_endpoint'
    huggingface_url: Optional[str] = None
    huggingface_api_key: Optional[str] = None

    def __init__(self, name: Optional[str]='hugging_face_custom_endpoint', huggingface_url: Optional[str]=None, huggingface_api_key: Optional[str]=None, score_threshold: float=0.8):
        """Initializes the HFEndpointEncoder with the specified parameters.

        :param name: The name of the encoder.
        :type name: str
        :param huggingface_url: The URL of the Hugging Face API endpoint.
        :type huggingface_url: str
        :param huggingface_api_key: The API key for the Hugging Face API.
        :type huggingface_api_key: str
        :param score_threshold: A threshold for processing the embeddings.
        :type score_threshold: float
        :raise ValueError: If either `huggingface_url` or `huggingface_api_key` is None.
        """
        huggingface_url = huggingface_url or os.getenv('HF_API_URL')
        huggingface_api_key = huggingface_api_key or os.getenv('HF_API_KEY')
        if score_threshold is None:
            score_threshold = 0.8
        super().__init__(name=name, score_threshold=score_threshold)
        if huggingface_url is None:
            raise ValueError("HuggingFace endpoint url cannot be 'None'.")
        if huggingface_api_key is None:
            raise ValueError("HuggingFace API key cannot be 'None'.")
        self.huggingface_url = huggingface_url or os.getenv('HF_API_URL')
        self.huggingface_api_key = huggingface_api_key or os.getenv('HF_API_KEY')
        try:
            self.query({'inputs': 'Hello World!', 'parameters': {}})
        except Exception as e:
            raise ValueError(f'HuggingFace endpoint client failed to initialize. Error: {e}') from e

    def __call__(self, docs: List[str]) -> List[List[float]]:
        """Encodes a list of documents into embeddings using the Hugging Face API.

        :param docs: A list of documents to encode.
        :type docs: List[str]
        :return: A list of embeddings for the given documents.
        :rtype: List[List[float]]
        :raise ValueError: If no embeddings are returned for a document.
        """
        embeddings = []
        for d in docs:
            try:
                output = self.query({'inputs': d, 'parameters': {}})
                if not output or len(output) == 0:
                    raise ValueError('No embeddings returned from the query.')
                embeddings.append(output)
            except Exception as e:
                raise ValueError(f'No embeddings returned for document. Error: {e}') from e
        return embeddings

    def query(self, payload, max_retries=3, retry_interval=5):
        """Sends a query to the Hugging Face API and returns the response.

        :param payload: The payload to send in the request.
        :type payload: dict
        :return: The response from the Hugging Face API.
        :rtype: dict
        :raise ValueError: If the query fails or the response status is not 200.
        """
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {self.huggingface_api_key}', 'Content-Type': 'application/json'}
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(self.huggingface_url, headers=headers, json=payload)
                if response.status_code == 503:
                    estimated_time = response.json().get('estimated_time', '')
                    if estimated_time:
                        logger.info(f'Model Initializing wait for - {estimated_time:.2f}s ')
                        time.sleep(estimated_time)
                        continue
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    logger.info(f'Retrying attempt: {attempt} for payload: {payload} ')
                    time.sleep(retry_interval)
                    retry_interval += attempt
                else:
                    raise ValueError(f'Query failed with status {response.status_code}: {response.text}')
        return response.json()

class OpenAIEncoder(DenseEncoder):
    """OpenAI encoder class for generating embeddings using OpenAI API.

    The OpenAIEncoder class is a subclass of DenseEncoder and utilizes the OpenAI API
    to generate embeddings for given documents. It requires an OpenAI API key and
    supports customization of the score threshold for filtering or processing the embeddings.
    """
    _client: Optional[openai.Client] = PrivateAttr(default=None)
    _async_client: Optional[openai.AsyncClient] = PrivateAttr(default=None)
    dimensions: Union[int, NotGiven] = NotGiven()
    token_limit: int = 8192
    _token_encoder: Any = PrivateAttr()
    type: str = 'openai'
    max_retries: int = 3

    def __init__(self, name: Optional[str]=None, openai_base_url: Optional[str]=None, openai_api_key: Optional[str]=None, openai_org_id: Optional[str]=None, score_threshold: Optional[float]=None, dimensions: Union[int, NotGiven]=NotGiven(), max_retries: int=3):
        """Initialize the OpenAIEncoder.

        :param name: The name of the embedding model to use.
        :type name: str
        :param openai_base_url: The base URL for the OpenAI API.
        :type openai_base_url: str
        :param openai_api_key: The OpenAI API key.
        :type openai_api_key: str
        :param openai_org_id: The OpenAI organization ID.
        :type openai_org_id: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        :param dimensions: The dimensions of the embeddings.
        :type dimensions: int
        :param max_retries: The maximum number of retries for the OpenAI API call.
        :type max_retries: int
        """
        if name is None:
            name = EncoderDefault.OPENAI.value['embedding_model']
        if score_threshold is None and name in model_configs:
            set_score_threshold = model_configs[name].threshold
        elif score_threshold is None:
            logger.warning(f'Score threshold not set for model: {name}. Using default value.')
            set_score_threshold = 0.82
        else:
            set_score_threshold = score_threshold
        super().__init__(name=name, score_threshold=set_score_threshold)
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        base_url = openai_base_url or os.getenv('OPENAI_BASE_URL')
        openai_org_id = openai_org_id or os.getenv('OPENAI_ORG_ID')
        if api_key is None or api_key.strip() == '':
            raise ValueError("OpenAI API key cannot be 'None' or empty.")
        if max_retries is not None:
            self.max_retries = max_retries
        try:
            self._client = openai.Client(base_url=base_url, api_key=api_key, organization=openai_org_id)
            self._async_client = openai.AsyncClient(base_url=base_url, api_key=api_key, organization=openai_org_id)
        except Exception as e:
            raise ValueError(f'OpenAI API client failed to initialize. Error: {e}') from e
        self.dimensions = dimensions
        if name in model_configs:
            self.token_limit = model_configs[name].token_limit
        self._token_encoder = tiktoken.encoding_for_model(name)

    def __call__(self, docs: List[str], truncate: bool=True) -> List[List[float]]:
        """Encode a list of text documents into embeddings using OpenAI API.

        :param docs: List of text documents to encode.
        :param truncate: Whether to truncate the documents to token limit. If
            False and a document exceeds the token limit, an error will be
            raised.
        :return: List of embeddings for each document."""
        if self._client is None:
            raise ValueError('OpenAI client is not initialized.')
        embeds = None
        if truncate:
            docs = [self._truncate(doc) for doc in docs]
        for j in range(self.max_retries + 1):
            try:
                logger.debug(f'Creating embeddings for {len(docs)} docs')
                embeds = self._client.embeddings.create(input=docs, model=self.name, dimensions=self.dimensions)
                if embeds.data:
                    break
            except OpenAIError as e:
                logger.error('Exception occurred', exc_info=True)
                if self.max_retries != 0 and j < self.max_retries:
                    sleep(2 ** j)
                    logger.warning(f'Retrying in {2 ** j} seconds due to OpenAIError: {e}')
                else:
                    raise
            except Exception as e:
                logger.error(f'OpenAI API call failed. Error: {e}')
                raise ValueError(f'OpenAI API call failed. Error: {str(e)}') from e
        if not embeds or not isinstance(embeds, CreateEmbeddingResponse) or (not embeds.data):
            logger.info(f'Returned embeddings: {embeds}')
            raise ValueError('No embeddings returned.')
        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings

    def _truncate(self, text: str) -> str:
        """Truncate a document to the token limit.

        :param text: The document to truncate.
        :type text: str
        :return: The truncated document.
        :rtype: str
        """
        tokens = self._token_encoder.encode_ordinary(text)
        if len(tokens) > self.token_limit:
            logger.warning(f'Document exceeds token limit: {len(tokens)} > {self.token_limit}\nTruncating document...')
            text = self._token_encoder.decode(tokens[:self.token_limit - 1])
            logger.info(f'Trunc length: {len(self._token_encoder.encode(text))}')
            return text
        return text

    async def acall(self, docs: List[str], truncate: bool=True) -> List[List[float]]:
        """Encode a list of text documents into embeddings using OpenAI API asynchronously.

        :param docs: List of text documents to encode.
        :param truncate: Whether to truncate the documents to token limit. If
            False and a document exceeds the token limit, an error will be
            raised.
        :return: List of embeddings for each document."""
        if self._async_client is None:
            raise ValueError('OpenAI async client is not initialized.')
        embeds = None
        if truncate:
            docs = [self._truncate(doc) for doc in docs]
        for j in range(self.max_retries + 1):
            try:
                embeds = await self._async_client.embeddings.create(input=docs, model=self.name, dimensions=self.dimensions)
                if embeds.data:
                    break
            except OpenAIError as e:
                logger.error('Exception occurred', exc_info=True)
                if self.max_retries != 0 and j < self.max_retries:
                    await asleep(2 ** j)
                    logger.warning(f'Retrying in {2 ** j} seconds due to OpenAIError: {e}')
                else:
                    raise
            except Exception as e:
                logger.error(f'OpenAI API call failed. Error: {e}')
                raise ValueError(f'OpenAI API call failed. Error: {e}') from e
        if not embeds or not isinstance(embeds, CreateEmbeddingResponse) or (not embeds.data):
            logger.info(f'Returned embeddings: {embeds}')
            raise ValueError('No embeddings returned.')
        embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
        return embeddings

class RouterConfig:
    """Generates a RouterConfig object that can be used for initializing routers."""
    routes: List[Route] = Field(default_factory=list)
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, routes: List[Route]=[], encoder_type: str='openai', encoder_name: Optional[str]=None):
        """Initialize a RouterConfig object.

        :param routes: A list of routes.
        :type routes: List[Route]
        :param encoder_type: The type of encoder to use.
        :type encoder_type: str
        :param encoder_name: The name of the encoder to use.
        :type encoder_name: Optional[str]
        """
        self.encoder_type = encoder_type
        if encoder_name is None:
            for encode_type in EncoderType:
                if encode_type.value == self.encoder_type:
                    if self.encoder_type == EncoderType.HUGGINGFACE.value:
                        raise NotImplementedError('HuggingFace encoder not supported by RouterConfig yet.')
                    encoder_name = EncoderDefault[encode_type.name].value['embedding_model']
                    break
            logger.info(f'Using default {encoder_type} encoder: {encoder_name}')
        self.encoder_name = encoder_name
        self.routes = routes

    @classmethod
    def from_file(cls, path: str) -> 'RouterConfig':
        """Initialize a RouterConfig from a file. Expects a JSON or YAML file with file
        extension .json, .yaml, or .yml.

        :param path: The path to the file to load the RouterConfig from.
        :type path: str
        """
        logger.info(f'Loading route config from {path}')
        _, ext = os.path.splitext(path)
        with open(path, 'r') as f:
            if ext == '.json':
                layer = json.load(f)
            elif ext in ['.yaml', '.yml']:
                layer = yaml.safe_load(f)
            else:
                raise ValueError('Unsupported file type. Only .json and .yaml are supported')
            if not is_valid(json.dumps(layer)):
                raise Exception('Invalid config JSON or YAML')
            encoder_type = layer['encoder_type']
            encoder_name = layer['encoder_name']
            routes = []
            for route_data in layer['routes']:
                if 'llm' in route_data and route_data['llm'] is not None:
                    llm_data = route_data.pop('llm')
                    llm_module_path = llm_data['module']
                    llm_module = importlib.import_module(llm_module_path)
                    llm_class = getattr(llm_module, llm_data['class'])
                    llm = llm_class(name=llm_data['model'])
                    route_data['llm'] = llm
                route = Route(**route_data)
                routes.append(route)
            return cls(encoder_type=encoder_type, encoder_name=encoder_name, routes=routes)

    @classmethod
    def from_tuples(cls, route_tuples: List[Tuple[str, str, Optional[List[Dict[str, Any]]], Dict[str, Any]]], encoder_type: str='openai', encoder_name: Optional[str]=None):
        """Initialize a RouterConfig from a list of tuples of routes and
        utterances.

        :param route_tuples: A list of tuples, each containing a route name and an
            associated utterance.
        :type route_tuples: List[Tuple[str, str]]
        :param encoder_type: The type of encoder to use, defaults to "openai".
        :type encoder_type: str, optional
        :param encoder_name: The name of the encoder to use, defaults to None.
        :type encoder_name: Optional[str], optional
        """
        routes_dict: Dict[str, Route] = {}
        for route_name, utterance, function_schema, metadata in route_tuples:
            if route_name not in routes_dict:
                routes_dict[route_name] = Route(name=route_name, utterances=[utterance], function_schemas=function_schema, metadata=metadata)
            else:
                routes_dict[route_name].utterances.append(utterance)
        routes: List[Route] = []
        for route_name, route in routes_dict.items():
            routes.append(route)
        return cls(routes=routes, encoder_type=encoder_type, encoder_name=encoder_name)

    @classmethod
    def from_index(cls, index: BaseIndex, encoder_type: str='openai', encoder_name: Optional[str]=None):
        """Initialize a RouterConfig from a BaseIndex object.

        :param index: The index to initialize the RouterConfig from.
        :type index: BaseIndex
        :param encoder_type: The type of encoder to use, defaults to "openai".
        :type encoder_type: str, optional
        :param encoder_name: The name of the encoder to use, defaults to None.
        :type encoder_name: Optional[str], optional
        """
        remote_routes = index.get_utterances(include_metadata=True)
        return cls.from_tuples(route_tuples=[utt.to_tuple() for utt in remote_routes], encoder_type=encoder_type, encoder_name=encoder_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the RouterConfig to a dictionary.

        :return: A dictionary representation of the RouterConfig.
        :rtype: Dict[str, Any]
        """
        return {'encoder_type': self.encoder_type, 'encoder_name': self.encoder_name, 'routes': [route.to_dict() for route in self.routes]}

    def to_file(self, path: str):
        """Save the routes to a file in JSON or YAML format.

        :param path: The path to save the RouterConfig to.
        :type path: str
        """
        logger.info(f'Saving route config to {path}')
        _, ext = os.path.splitext(path)
        if ext not in ['.json', '.yaml', '.yml']:
            raise ValueError('Unsupported file type. Only .json and .yaml are supported')
        dir_name = os.path.dirname(path)
        if dir_name and (not os.path.exists(dir_name)):
            os.makedirs(dir_name)
        with open(path, 'w') as f:
            if ext == '.json':
                json.dump(self.to_dict(), f, indent=4)
            elif ext in ['.yaml', '.yml']:
                yaml.safe_dump(self.to_dict(), f)

    def to_utterances(self) -> List[Utterance]:
        """Convert the routes to a list of Utterance objects.

        :return: A list of Utterance objects.
        :rtype: List[Utterance]
        """
        utterances = []
        for route in self.routes:
            utterances.extend([Utterance(route=route.name, utterance=x, function_schemas=route.function_schemas, metadata=route.metadata or {}) for x in route.utterances])
        return utterances

    def add(self, route: Route):
        """Add a route to the RouterConfig.

        :param route: The route to add.
        :type route: Route
        """
        self.routes.append(route)
        logger.info(f'Added route `{route.name}`')

    def get(self, name: str) -> Optional[Route]:
        """Get a route from the RouterConfig by name.

        :param name: The name of the route to get.
        :type name: str
        :return: The route if found, otherwise None.
        :rtype: Optional[Route]
        """
        for route in self.routes:
            if route.name == name:
                return route
        logger.error(f'Route `{name}` not found')
        return None

    def remove(self, name: str):
        """Remove a route from the RouterConfig by name.

        :param name: The name of the route to remove.
        :type name: str
        """
        if name not in [route.name for route in self.routes]:
            logger.error(f'Route `{name}` not found')
        else:
            self.routes = [route for route in self.routes if route.name != name]
            logger.info(f'Removed route `{name}`')

    def get_hash(self) -> ConfigParameter:
        """Get the hash of the RouterConfig. Used for syncing.

        :return: The hash of the RouterConfig.
        :rtype: ConfigParameter
        """
        layer = self.to_dict()
        return ConfigParameter(field='sr_hash', value=hashlib.sha256(json.dumps(layer).encode()).hexdigest())

class BaseRouter(BaseModel):
    """Base class for all routers."""
    encoder: DenseEncoder = Field(default_factory=OpenAIEncoder)
    sparse_encoder: Optional[SparseEncoder] = Field(default=None)
    index: BaseIndex = Field(default_factory=BaseIndex)
    score_threshold: Optional[float] = Field(default=None)
    routes: List[Route] = Field(default_factory=list)
    llm: Optional[BaseLLM] = None
    top_k: int = 5
    aggregation: str = 'mean'
    aggregation_method: Optional[Callable] = None
    auto_sync: Optional[str] = None
    init_async_index: bool = False
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, encoder: Optional[DenseEncoder]=None, sparse_encoder: Optional[SparseEncoder]=None, llm: Optional[BaseLLM]=None, routes: Optional[List[Route]]=None, index: Optional[BaseIndex]=None, top_k: int=5, aggregation: str='mean', auto_sync: Optional[str]=None, init_async_index: bool=False):
        """Initialize a BaseRouter object. Expected to be used as a base class only,
        not directly instantiated.

        :param encoder: The encoder to use.
        :type encoder: Optional[DenseEncoder]
        :param sparse_encoder: The sparse encoder to use.
        :type sparse_encoder: Optional[SparseEncoder]
        :param llm: The LLM to use.
        :type llm: Optional[BaseLLM]
        :param routes: The routes to use.
        :type routes: Optional[List[Route]]
        :param index: The index to use.
        :type index: Optional[BaseIndex]
        :param top_k: The number of routes to return.
        :type top_k: int
        :param aggregation: The aggregation method to use.
        :type aggregation: str
        :param auto_sync: The auto sync mode to use.
        :type auto_sync: Optional[str]
        """
        routes = routes.copy() if routes else []
        super().__init__(encoder=encoder, sparse_encoder=sparse_encoder, llm=llm, routes=routes, index=index, top_k=top_k, aggregation=aggregation, auto_sync=auto_sync)
        self.encoder = self._get_encoder(encoder=encoder)
        self.sparse_encoder = self._get_sparse_encoder(sparse_encoder=sparse_encoder)
        self.llm = llm
        self.routes = routes
        self.index = self._get_index(index=index)
        self._set_score_threshold()
        self.top_k = top_k
        if self.top_k < 1:
            raise ValueError(f'top_k needs to be >= 1, but was: {self.top_k}.')
        self.aggregation = aggregation
        if self.aggregation not in ['sum', 'mean', 'max']:
            raise ValueError(f"Unsupported aggregation method chosen: {aggregation}. Choose either 'SUM', 'MEAN', or 'MAX'.")
        self.aggregation_method = self._set_aggregation_method(self.aggregation)
        if isinstance(self.index, PostgresIndex):
            self.auto_sync = 'local'
        else:
            self.auto_sync = auto_sync
        for route in self.routes:
            if route.score_threshold is None:
                route.score_threshold = self.score_threshold
        if not init_async_index:
            self._init_index_state()

    def _get_index(self, index: Optional[BaseIndex]) -> BaseIndex:
        """Get the index to use.

        :param index: The index to use.
        :type index: Optional[BaseIndex]
        :return: The index to use.
        :rtype: BaseIndex
        """
        if index is None:
            logger.warning('No index provided. Using default LocalIndex.')
            index = LocalIndex()
        else:
            index = index
        return index

    def _get_encoder(self, encoder: Optional[DenseEncoder]) -> DenseEncoder:
        """Get the dense encoder to be used for creating dense vector embeddings.

        :param encoder: The encoder to use.
        :type encoder: Optional[DenseEncoder]
        :return: The encoder to use.
        :rtype: DenseEncoder
        """
        if encoder is None:
            logger.warning('No encoder provided. Using default OpenAIEncoder.')
            encoder = OpenAIEncoder()
        else:
            encoder = encoder
        return encoder

    def _get_sparse_encoder(self, sparse_encoder: Optional[SparseEncoder]) -> Optional[SparseEncoder]:
        """Get the sparse encoder to be used for creating sparse vector embeddings.

        :param sparse_encoder: The sparse encoder to use.
        :type sparse_encoder: Optional[SparseEncoder]
        :return: The sparse encoder to use.
        :rtype: Optional[SparseEncoder]
        """
        if sparse_encoder is None:
            return None
        raise NotImplementedError(f'Sparse encoder not implemented for {self.__class__.__name__}')

    def _init_index_state(self):
        """Initializes an index (where required) and runs auto_sync if active."""
        if self.index.dimensions is None:
            dims = len(self.encoder(['test'])[0])
            self.index.dimensions = dims
        if isinstance(self.index, PineconeIndex) or isinstance(self.index, PostgresIndex):
            self.index.index = self.index._init_index(force_create=True)
        if self.auto_sync:
            local_utterances = self.to_config().to_utterances()
            remote_utterances = self.index.get_utterances(include_metadata=True)
            diff = UtteranceDiff.from_utterances(local_utterances=local_utterances, remote_utterances=remote_utterances)
            sync_strategy = diff.get_sync_strategy(self.auto_sync)
            self._execute_sync_strategy(sync_strategy)

    async def _async_init_index_state(self):
        """Asynchronously initializes an index (where required) and runs auto_sync if active."""
        if self.index is None or self.index.dimensions is None:
            dims = len(self.encoder(['test'])[0])
            self.index.dimensions = dims
        if isinstance(self.index, PineconeIndex) or isinstance(self.index, PostgresIndex):
            await self.index._init_async_index(force_create=True)
        if self.auto_sync:
            local_utterances = self.to_config().to_utterances()
            remote_utterances = await self.index.aget_utterances(include_metadata=True)
            diff = UtteranceDiff.from_utterances(local_utterances=local_utterances, remote_utterances=remote_utterances)
            sync_strategy = diff.get_sync_strategy(self.auto_sync)
            await self._async_execute_sync_strategy(sync_strategy)

    def _set_score_threshold(self):
        """Set the score threshold for the layer based on the encoder
        score threshold.

        When no score threshold is used a default `None` value
        is used, which means that a route will always be returned when
        the layer is called."""
        if self.encoder.score_threshold is not None:
            self.score_threshold = self.encoder.score_threshold
            if self.score_threshold is None:
                logger.warning("No score threshold value found in encoder. Using the default 'None' value can lead to unexpected results.")

    def check_for_matching_routes(self, top_class: str) -> Optional[Route]:
        """Check for a matching route in the routes list.

        :param top_class: The top class to check for.
        :type top_class: str
        :return: The matching route if found, otherwise None.
        :rtype: Optional[Route]
        """
        matching_route = next((route for route in self.routes if route.name == top_class), None)
        if matching_route is None:
            logger.error(f'No route found with name {top_class}. Check to see if any Routes have been defined.')
            return None
        return matching_route

    def __call__(self, text: Optional[str]=None, vector: Optional[List[float] | np.ndarray]=None, simulate_static: bool=False, route_filter: Optional[List[str]]=None, limit: int | None=1) -> RouteChoice | List[RouteChoice]:
        """Call the router to get a route choice.

        :param text: The text to route.
        :type text: Optional[str]
        :param vector: The vector to route.
        :type vector: Optional[List[float] | np.ndarray]
        :param simulate_static: Whether to simulate a static route.
        :type simulate_static: bool
        :param route_filter: The route filter to use.
        :type route_filter: Optional[List[str]]
        :param limit: The number of routes to return, defaults to 1. If set to None, no
            limit is applied and all routes are returned.
        :type limit: int | None
        :return: The route choice.
        :rtype: RouteChoice | List[RouteChoice]
        """
        if not self.index.is_ready():
            raise ValueError('Index is not ready.')
        if vector is None:
            if text is None:
                raise ValueError('Either text or vector must be provided')
            vector = self._encode(text=[text], input_type='queries')
        vector = xq_reshape(vector)
        scores, routes = self.index.query(vector=vector[0], top_k=self.top_k, route_filter=route_filter)
        query_results = [{'route': d, 'score': s.item()} for d, s in zip(routes, scores)]
        scored_routes = self._score_routes(query_results=query_results)
        return self._pass_routes(scored_routes=scored_routes, simulate_static=simulate_static, text=text, limit=limit)

    def _pass_routes(self, scored_routes: List[Tuple[str, float, List[float]]], simulate_static: bool, text: Optional[str], limit: int | None) -> RouteChoice | list[RouteChoice]:
        """Returns a list of RouteChoice objects that passed the thresholds set.

        :param scored_routes: The scored routes to pass.
        :type scored_routes: List[Tuple[str, float, List[float]]]
        :param simulate_static: Whether to simulate a static route.
        :type simulate_static: bool
        :param text: The text to route.
        :type text: Optional[str]
        :param limit: The number of routes to return, defaults to 1. If set to None, no
            limit is applied and all routes are returned.
        :type limit: int | None
        :return: The route choice.
        :rtype: RouteChoice | list[RouteChoice]
        """
        passed_routes: list[RouteChoice] = []
        for route_name, total_score, scores in scored_routes:
            route = self.check_for_matching_routes(top_class=route_name)
            if route is None:
                continue
            if (current_threshold := (route.score_threshold if route.score_threshold is not None else self.score_threshold)):
                passed = total_score >= current_threshold
            else:
                passed = True
            if passed and route is not None and (not simulate_static):
                if route.function_schemas and text is None:
                    raise ValueError('Route has a function schema, but no text was provided.')
                if route.function_schemas and (not isinstance(route.llm, BaseLLM)):
                    if not self.llm:
                        logger.warning('No LLM provided for dynamic route, will use OpenAI LLM default. Ensure API key is set in OPENAI_API_KEY environment variable.')
                        self.llm = OpenAILLM()
                        route.llm = self.llm
                    else:
                        route.llm = self.llm
                route_choice = route(query=text)
                if route_choice is not None and route_choice.similarity_score is None:
                    route_choice.similarity_score = total_score
                passed_routes.append(route_choice)
            elif passed and route is not None and simulate_static:
                passed_routes.append(RouteChoice(name=route.name, function_call=None, similarity_score=None))
            if limit is None:
                continue
            if len(passed_routes) >= limit:
                if limit == 1:
                    return passed_routes[0]
                else:
                    return passed_routes
        if len(passed_routes) == 1:
            return passed_routes[0]
        elif len(passed_routes) > 1:
            return passed_routes
        else:
            return RouteChoice()

    async def _async_pass_routes(self, scored_routes: List[Tuple[str, float, List[float]]], simulate_static: bool, text: Optional[str], limit: int | None) -> RouteChoice | list[RouteChoice]:
        """Returns a list of RouteChoice objects that passed the thresholds set. Runs any
        dynamic route calls asynchronously. If there are no dynamic routes this method is
        equivalent to _pass_routes.

        :param scored_routes: The scored routes to pass.
        :type scored_routes: List[Tuple[str, float, List[float]]]
        :param simulate_static: Whether to simulate a static route.
        :type simulate_static: bool
        :param text: The text to route.
        :type text: Optional[str]
        :param limit: The number of routes to return, defaults to 1. If set to None, no
            limit is applied and all routes are returned.
        :type limit: int | None
        :return: The route choice.
        :rtype: RouteChoice | list[RouteChoice]
        """
        passed_routes: list[RouteChoice] = []
        for route_name, total_score, scores in scored_routes:
            route = self.check_for_matching_routes(top_class=route_name)
            if route is None:
                continue
            if (current_threshold := (route.score_threshold if route.score_threshold is not None else self.score_threshold)):
                passed = total_score >= current_threshold
            else:
                passed = True
            if passed and route is not None and (not simulate_static):
                if route.function_schemas and text is None:
                    raise ValueError('Route has a function schema, but no text was provided.')
                if route.function_schemas and (not isinstance(route.llm, BaseLLM)):
                    if not self.llm:
                        logger.warning('No LLM provided for dynamic route, will use OpenAI LLM default. Ensure API key is set in OPENAI_API_KEY environment variable.')
                        self.llm = OpenAILLM()
                        route.llm = self.llm
                    else:
                        route.llm = self.llm
                route_choice = await route.acall(query=text)
                if route_choice is not None and route_choice.similarity_score is None:
                    route_choice.similarity_score = total_score
                passed_routes.append(route_choice)
            elif passed and route is not None and simulate_static:
                passed_routes.append(RouteChoice(name=route.name, function_call=None, similarity_score=None))
            if limit is None:
                continue
            if len(passed_routes) >= limit:
                if limit == 1:
                    return passed_routes[0]
                else:
                    return passed_routes
        if len(passed_routes) == 1:
            return passed_routes[0]
        elif len(passed_routes) > 1:
            return passed_routes
        else:
            return RouteChoice()

    async def acall(self, text: Optional[str]=None, vector: Optional[List[float] | np.ndarray]=None, limit: int | None=1, simulate_static: bool=False, route_filter: Optional[List[str]]=None) -> RouteChoice | list[RouteChoice]:
        """Asynchronously call the router to get a route choice.

        :param text: The text to route.
        :type text: Optional[str]
        :param vector: The vector to route.
        :type vector: Optional[List[float] | np.ndarray]
        :param simulate_static: Whether to simulate a static route (ie avoid dynamic route
            LLM calls during fit or evaluate).
        :type simulate_static: bool
        :param route_filter: The route filter to use.
        :type route_filter: Optional[List[str]]
        :return: The route choice.
        :rtype: RouteChoice
        """
        if not await self.index.ais_ready():
            await self._async_init_index_state()
        if vector is None:
            if text is None:
                raise ValueError('Either text or vector must be provided')
            vector = await self._async_encode(text=[text], input_type='queries')
        vector = xq_reshape(vector)
        scores, routes = await self.index.aquery(vector=vector[0], top_k=self.top_k, route_filter=route_filter)
        query_results = [{'route': d, 'score': s.item()} for d, s in zip(routes, scores)]
        scored_routes = self._score_routes(query_results=query_results)
        return await self._async_pass_routes(scored_routes=scored_routes, simulate_static=simulate_static, text=text, limit=limit)

    def _index_ready(self) -> bool:
        """Method to check if the index is ready to be used.

        :return: True if the index is ready, False otherwise.
        :rtype: bool
        """
        if self.index.index is None or self.routes is None:
            return False
        if isinstance(self.index, QdrantIndex):
            info = self.index.describe()
            if info.vectors == 0:
                return False
        return True

    def sync(self, sync_mode: str, force: bool=False, wait: int=0) -> List[str]:
        """Runs a sync of the local routes with the remote index.

        :param sync_mode: The mode to sync the routes with the remote index.
        :type sync_mode: str
        :param force: Whether to force the sync even if the local and remote
            hashes already match. Defaults to False.
        :type force: bool, optional
        :param wait: The number of seconds to wait for the index to be unlocked
        before proceeding with the sync. If set to 0, will raise an error if
        index is already locked/unlocked.
        :type wait: int
        :return: A list of diffs describing the addressed differences between
            the local and remote route layers.
        :rtype: List[str]
        """
        if not force and self.is_synced():
            logger.warning('Local and remote route layers are already synchronized.')
            local_utterances = self.to_config().to_utterances()
            diff = UtteranceDiff.from_utterances(local_utterances=local_utterances, remote_utterances=local_utterances)
            return diff.to_utterance_str()
        try:
            diff_utt_str: list[str] = []
            _ = self.index.lock(value=True, wait=wait)
            try:
                local_utterances = self.to_config().to_utterances()
                remote_utterances = self.index.get_utterances(include_metadata=True)
                diff = UtteranceDiff.from_utterances(local_utterances=local_utterances, remote_utterances=remote_utterances)
                sync_strategy = diff.get_sync_strategy(sync_mode=sync_mode)
                self._execute_sync_strategy(sync_strategy)
                diff_utt_str = diff.to_utterance_str()
            except Exception as e:
                logger.error(f'Failed to create diff: {e}')
                raise e
            finally:
                _ = self.index.lock(value=False)
        except Exception as e:
            logger.error(f'Failed to lock index for sync: {e}')
            raise e
        return diff_utt_str

    async def async_sync(self, sync_mode: str, force: bool=False, wait: int=0) -> List[str]:
        """Runs a sync of the local routes with the remote index.

        :param sync_mode: The mode to sync the routes with the remote index.
        :type sync_mode: str
        :param force: Whether to force the sync even if the local and remote
            hashes already match. Defaults to False.
        :type force: bool, optional
        :param wait: The number of seconds to wait for the index to be unlocked
        before proceeding with the sync. If set to 0, will raise an error if
        index is already locked/unlocked.
        :type wait: int
        :return: A list of diffs describing the addressed differences between
            the local and remote route layers.
        :rtype: List[str]
        """
        if not force and await self.async_is_synced():
            logger.warning('Local and remote route layers are already synchronized.')
            local_utterances = self.to_config().to_utterances()
            diff = UtteranceDiff.from_utterances(local_utterances=local_utterances, remote_utterances=local_utterances)
            return diff.to_utterance_str()
        try:
            diff_utt_str: list[str] = []
            _ = await self.index.alock(value=True, wait=wait)
            try:
                local_utterances = self.to_config().to_utterances()
                remote_utterances = await self.index.aget_utterances(include_metadata=True)
                diff = UtteranceDiff.from_utterances(local_utterances=local_utterances, remote_utterances=remote_utterances)
                sync_strategy = diff.get_sync_strategy(sync_mode=sync_mode)
                await self._async_execute_sync_strategy(sync_strategy)
                diff_utt_str = diff.to_utterance_str()
            except Exception as e:
                logger.error(f'Failed to create diff: {e}')
                raise e
            finally:
                _ = await self.index.alock(value=False)
        except Exception as e:
            logger.error(f'Failed to lock index for sync: {e}')
            raise e
        return diff_utt_str

    def _execute_sync_strategy(self, strategy: Dict[str, Dict[str, List[Utterance]]]):
        """Executes the provided sync strategy, either deleting or upserting
        routes from the local and remote instances as defined in the strategy.

        :param strategy: The sync strategy to execute.
        :type strategy: Dict[str, Dict[str, List[Utterance]]]
        """
        if strategy['remote']['delete']:
            data_to_delete = {}
            for utt_obj in strategy['remote']['delete']:
                data_to_delete.setdefault(utt_obj.route, []).append(utt_obj.utterance)
            self.index._remove_and_sync(data_to_delete)
        if strategy['remote']['upsert']:
            utterances_text = [utt.utterance for utt in strategy['remote']['upsert']]
            self.index.add(embeddings=self.encoder(utterances_text), routes=[utt.route for utt in strategy['remote']['upsert']], utterances=utterances_text, function_schemas=[utt.function_schemas for utt in strategy['remote']['upsert']], metadata_list=[utt.metadata for utt in strategy['remote']['upsert']])
        if strategy['local']['delete']:
            self._local_delete(utterances=strategy['local']['delete'])
        if strategy['local']['upsert']:
            self._local_upsert(utterances=strategy['local']['upsert'])
        self._write_hash()

    async def _async_execute_sync_strategy(self, strategy: Dict[str, Dict[str, List[Utterance]]]):
        """Executes the provided sync strategy, either deleting or upserting
        routes from the local and remote instances as defined in the strategy.

        :param strategy: The sync strategy to execute.
        :type strategy: Dict[str, Dict[str, List[Utterance]]]
        """
        if strategy['remote']['delete']:
            data_to_delete = {}
            for utt_obj in strategy['remote']['delete']:
                data_to_delete.setdefault(utt_obj.route, []).append(utt_obj.utterance)
            await self.index._async_remove_and_sync(data_to_delete)
        if strategy['remote']['upsert']:
            utterances_text = [utt.utterance for utt in strategy['remote']['upsert']]
            await self.index.aadd(embeddings=await self.encoder.acall(docs=utterances_text), routes=[utt.route for utt in strategy['remote']['upsert']], utterances=utterances_text, function_schemas=[utt.function_schemas for utt in strategy['remote']['upsert']], metadata_list=[utt.metadata for utt in strategy['remote']['upsert']])
        if strategy['local']['delete']:
            self._local_delete(utterances=strategy['local']['delete'])
        if strategy['local']['upsert']:
            self._local_upsert(utterances=strategy['local']['upsert'])
        await self._async_write_hash()

    def _local_upsert(self, utterances: List[Utterance]):
        """Adds new routes to the SemanticRouter.

        :param utterances: The utterances to add to the local SemanticRouter.
        :type utterances: List[Utterance]
        """
        new_routes = {route.name: route for route in self.routes}
        for utt_obj in utterances:
            if utt_obj.route not in new_routes.keys():
                new_routes[utt_obj.route] = Route(name=utt_obj.route, utterances=[utt_obj.utterance], function_schemas=utt_obj.function_schemas, metadata=utt_obj.metadata)
            else:
                if utt_obj.utterance not in new_routes[utt_obj.route].utterances:
                    new_routes[utt_obj.route].utterances.append(utt_obj.utterance)
                new_routes[utt_obj.route].function_schemas = utt_obj.function_schemas
                new_routes[utt_obj.route].metadata = utt_obj.metadata
        self.routes = list(new_routes.values())

    def _local_delete(self, utterances: List[Utterance]):
        """Deletes routes from the local SemanticRouter.

        :param utterances: The utterances to delete from the local SemanticRouter.
        :type utterances: List[Utterance]
        """
        route_dict: dict[str, List[str]] = {}
        for utt in utterances:
            route_dict.setdefault(utt.route, []).append(utt.utterance)
        new_routes = []
        for route in self.routes:
            if route.name in route_dict.keys():
                new_utterances = list(set(route.utterances) - set(route_dict[route.name]))
                if len(new_utterances) == 0:
                    continue
                else:
                    new_routes.append(Route(name=route.name, utterances=new_utterances, function_schemas=route.function_schemas, metadata=route.metadata))
            else:
                new_routes.append(route)
        self.routes = new_routes

    def __str__(self):
        return f'{self.__class__.__name__}(encoder={self.encoder}, score_threshold={self.score_threshold}, routes={self.routes})'

    @classmethod
    def from_json(cls, file_path: str):
        """Load a RouterConfig from a JSON file.

        :param file_path: The path to the JSON file.
        :type file_path: str
        :return: The RouterConfig object.
        :rtype: RouterConfig
        """
        config = RouterConfig.from_file(file_path)
        encoder = AutoEncoder(type=config.encoder_type, name=config.encoder_name).model
        if isinstance(encoder, DenseEncoder):
            return cls(encoder=encoder, routes=config.routes)
        else:
            raise ValueError(f'{type(encoder)} not supported for loading from JSON.')

    @classmethod
    def from_yaml(cls, file_path: str):
        """Load a RouterConfig from a YAML file.

        :param file_path: The path to the YAML file.
        :type file_path: str
        :return: The RouterConfig object.
        :rtype: RouterConfig
        """
        config = RouterConfig.from_file(file_path)
        encoder = AutoEncoder(type=config.encoder_type, name=config.encoder_name).model
        if isinstance(encoder, DenseEncoder):
            return cls(encoder=encoder, routes=config.routes)
        else:
            raise ValueError(f'{type(encoder)} not supported for loading from YAML.')

    @classmethod
    def from_config(cls, config: RouterConfig, index: Optional[BaseIndex]=None):
        """Create a Router from a RouterConfig object.

        :param config: The RouterConfig object.
        :type config: RouterConfig
        :param index: The index to use.
        :type index: Optional[BaseIndex]
        """
        encoder = AutoEncoder(type=config.encoder_type, name=config.encoder_name).model
        if isinstance(encoder, DenseEncoder):
            return cls(encoder=encoder, routes=config.routes, index=index)
        else:
            raise ValueError(f'{type(encoder)} not supported for loading from config.')

    def add(self, routes: List[Route] | Route):
        """Add a route to the local SemanticRouter and index.

        :param route: The route to add.
        :type route: Route
        """
        raise NotImplementedError('This method must be implemented by subclasses.')

    async def aadd(self, routes: List[Route] | Route):
        """Add a route to the local SemanticRouter and index asynchronously.

        :param route: The route to add.
        :type route: Route
        """
        logger.warning('Async method not implemented.')
        return self.add(routes)

    def list_route_names(self) -> List[str]:
        return [route.name for route in self.routes]

    def update(self, name: str, threshold: Optional[float]=None, utterances: Optional[List[str]]=None):
        """Updates the route specified in name. Allows the update of
        threshold and/or utterances. If no values are provided via the
        threshold or utterances parameters, those fields are not updated.
        If neither field is provided raises a ValueError.

        The name must exist within the local SemanticRouter, if not a
        KeyError will be raised.

        :param name: The name of the route to update.
        :type name: str
        :param threshold: The threshold to update.
        :type threshold: Optional[float]
        :param utterances: The utterances to update.
        :type utterances: Optional[List[str]]
        """
        current_local_hash = self._get_hash()
        current_remote_hash = self.index._read_hash()
        if current_remote_hash.value == '':
            current_remote_hash = current_local_hash
        if threshold is None and utterances is None:
            raise ValueError("At least one of 'threshold' or 'utterances' must be provided.")
        if utterances:
            raise NotImplementedError('The update method cannot be used for updating utterances yet.')
        route = self.get(name)
        if route:
            if threshold:
                old_threshold = route.score_threshold
                route.score_threshold = threshold
                logger.info(f"Updated threshold for route '{route.name}' from {old_threshold} to {threshold}")
        else:
            raise ValueError(f"Route '{name}' not found. Nothing updated.")
        if current_local_hash.value == current_remote_hash.value:
            self._write_hash()
        else:
            logger.warning(f'Local and remote route layers were not aligned. Remote hash not updated. Use `{self.__class__.__name__}.get_utterance_diff()` to see details.')

    def delete(self, route_name: str):
        """Deletes a route given a specific route name.

        :param route_name: the name of the route to be deleted
        :type str:
        """
        if self.index._is_locked():
            raise ValueError('Index is locked. Cannot delete route.')
        current_local_hash = self._get_hash()
        current_remote_hash = self.index._read_hash()
        if current_remote_hash.value == '':
            current_remote_hash = current_local_hash
        if route_name not in [route.name for route in self.routes]:
            err_msg = f'Route `{route_name}` not found in {self.__class__.__name__}'
            logger.warning(err_msg)
            try:
                self.index.delete(route_name=route_name)
            except Exception as e:
                logger.error(f'Failed to delete route from the index: {e}')
        else:
            self.routes = [route for route in self.routes if route.name != route_name]
            self.index.delete(route_name=route_name)
        if current_local_hash.value == current_remote_hash.value:
            self._write_hash()
        else:
            logger.warning(f'Local and remote route layers were not aligned. Remote hash not updated. Use `{self.__class__.__name__}.get_utterance_diff()` to see details.')

    async def adelete(self, route_name: str):
        """Deletes a route given a specific route name asynchronously.

        :param route_name: the name of the route to be deleted
        :type str:
        """
        if await self.index._ais_locked():
            raise ValueError('Index is locked. Cannot delete route.')
        current_local_hash = self._get_hash()
        current_remote_hash = await self.index._async_read_hash()
        if current_remote_hash.value == '':
            current_remote_hash = current_local_hash
        if route_name not in [route.name for route in self.routes]:
            err_msg = f'Route `{route_name}` not found in {self.__class__.__name__}'
            logger.warning(err_msg)
            try:
                await self.index.adelete(route_name=route_name)
            except Exception as e:
                logger.error(f'Failed to delete route from the index: {e}')
        else:
            self.routes = [route for route in self.routes if route.name != route_name]
            await self.index.adelete(route_name=route_name)
        if current_local_hash.value == current_remote_hash.value:
            await self._async_write_hash()
        else:
            logger.warning(f'Local and remote route layers were not aligned. Remote hash not updated. Use `{self.__class__.__name__}.get_utterance_diff()` to see details.')

    def _refresh_routes(self):
        """Pulls out the latest routes from the index.

        Not yet implemented for BaseRouter.
        """
        raise NotImplementedError('This method has not yet been implemented.')
        route_mapping = {route.name: route for route in self.routes}
        index_routes = self.index.get_utterances()
        new_routes_names = []
        new_routes = []
        for route_name, utterance in index_routes:
            if route_name in route_mapping:
                if route_name not in new_routes_names:
                    existing_route = route_mapping[route_name]
                    new_routes.append(existing_route)
                new_routes.append(Route(name=route_name, utterances=[utterance]))
            route = route_mapping[route_name]
            self.routes.append(route)

    def _get_hash(self) -> ConfigParameter:
        """Get the hash of the current routes.

        :return: The hash of the current routes.
        :rtype: ConfigParameter
        """
        config = self.to_config()
        return config.get_hash()

    def _write_hash(self) -> ConfigParameter:
        """Write the hash of the current routes to the index.

        :return: The hash of the current routes.
        :rtype: ConfigParameter
        """
        config = self.to_config()
        hash_config = config.get_hash()
        self.index._write_config(config=hash_config)
        return hash_config

    async def _async_write_hash(self) -> ConfigParameter:
        """Write the hash of the current routes to the index asynchronously.

        :return: The hash of the current routes.
        :rtype: ConfigParameter
        """
        config = self.to_config()
        hash_config = config.get_hash()
        await self.index._async_write_config(config=hash_config)
        return hash_config

    def is_synced(self) -> bool:
        """Check if the local and remote route layer instances are
        synchronized.

        :return: True if the local and remote route layers are synchronized,
            False otherwise.
        :rtype: bool
        """
        local_hash = self._get_hash()
        remote_hash = self.index._read_hash()
        if local_hash.value == remote_hash.value:
            return True
        else:
            return False

    async def async_is_synced(self) -> bool:
        """Check if the local and remote route layer instances are
        synchronized asynchronously.

        :return: True if the local and remote route layers are synchronized,
            False otherwise.
        :rtype: bool
        """
        local_hash = self._get_hash()
        remote_hash = await self.index._async_read_hash()
        if local_hash.value == remote_hash.value:
            return True
        else:
            return False

    def get_utterance_diff(self, include_metadata: bool=False) -> List[str]:
        """Get the difference between the local and remote utterances. Returns
        a list of strings showing what is different in the remote when compared
        to the local. For example:

        ["  route1: utterance1",
         "  route1: utterance2",
         "- route2: utterance3",
         "- route2: utterance4"]

        Tells us that the remote is missing "route2: utterance3" and "route2:
        utterance4", which do exist locally. If we see:

        ["  route1: utterance1",
         "  route1: utterance2",
         "+ route2: utterance3",
         "+ route2: utterance4"]

        This diff tells us that the remote has "route2: utterance3" and
        "route2: utterance4", which do not exist locally.

        :param include_metadata: Whether to include metadata in the diff.
        :type include_metadata: bool
        :return: A list of strings showing the difference between the local and remote
            utterances.
        :rtype: List[str]
        """
        remote_utterances = self.index.get_utterances(include_metadata=include_metadata)
        local_utterances = self.to_config().to_utterances()
        diff_obj = UtteranceDiff.from_utterances(local_utterances=local_utterances, remote_utterances=remote_utterances)
        return diff_obj.to_utterance_str(include_metadata=include_metadata)

    async def aget_utterance_diff(self, include_metadata: bool=False) -> List[str]:
        """Get the difference between the local and remote utterances asynchronously.
        Returns a list of strings showing what is different in the remote when
        compared to the local. For example:

        ["  route1: utterance1",
         "  route1: utterance2",
         "- route2: utterance3",
         "- route2: utterance4"]

        Tells us that the remote is missing "route2: utterance3" and "route2:
        utterance4", which do exist locally. If we see:

        ["  route1: utterance1",
         "  route1: utterance2",
         "+ route2: utterance3",
         "+ route2: utterance4"]

        This diff tells us that the remote has "route2: utterance3" and
        "route2: utterance4", which do not exist locally.

        :param include_metadata: Whether to include metadata in the diff.
        :type include_metadata: bool
        :return: A list of strings showing the difference between the local and remote
            utterances.
        :rtype: List[str]
        """
        remote_utterances = await self.index.aget_utterances(include_metadata=include_metadata)
        local_utterances = self.to_config().to_utterances()
        diff_obj = UtteranceDiff.from_utterances(local_utterances=local_utterances, remote_utterances=remote_utterances)
        return diff_obj.to_utterance_str(include_metadata=include_metadata)

    def _extract_routes_details(self, routes: List[Route], include_metadata: bool=False) -> Tuple:
        """Extract the routes details.

        :param routes: The routes to extract the details from.
        :type routes: List[Route]
        :param include_metadata: Whether to include metadata in the details.
        :type include_metadata: bool
        :return: A tuple of the route names, utterances, and function schemas.
        """
        route_names = [route.name for route in routes for _ in route.utterances]
        utterances = [utterance for route in routes for utterance in route.utterances]
        function_schemas = [route.function_schemas[0] if route.function_schemas and len(route.function_schemas) > 0 else {} for route in routes for _ in route.utterances]
        if include_metadata:
            metadata = [route.metadata for route in routes for _ in route.utterances]
            return (route_names, utterances, function_schemas, metadata)
        return (route_names, utterances, function_schemas)

    def _encode(self, text: list[str], input_type: EncodeInputType) -> Any:
        """Generates embeddings for a given text.

        Must be implemented by a subclass.

        :param text: The text to encode.
        :type text: list[str]
        :param input_type: Specify whether encoding 'queries' or 'documents', used in asymmetric retrieval
        :type input_type: semantic_router.encoders.encode_input_type.EncodeInputType
        :return: The embeddings of the text.
        :rtype: Any
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    async def _async_encode(self, text: list[str], input_type: EncodeInputType) -> Any:
        """Asynchronously generates embeddings for a given text.

        Must be implemented by a subclass.

        :param text: The text to encode.
        :type text: list[str]
        :param input_type: Specify whether encoding 'queries' or 'documents', used in asymmetric retrieval
        :type input_type: semantic_router.encoders.encode_input_type.EncodeInputType
        :return: The embeddings of the text.
        :rtype: Any
        """
        raise NotImplementedError('This method should be implemented by subclasses.')

    def _set_aggregation_method(self, aggregation: str='sum'):
        """Set the aggregation method.

        :param aggregation: The aggregation method to use.
        :type aggregation: str
        :return: The aggregation method.
        :rtype: Callable
        """
        if aggregation == 'sum':
            return lambda x: sum(x)
        elif aggregation == 'mean':
            return lambda x: np.mean(x)
        elif aggregation == 'max':
            return lambda x: max(x)
        else:
            raise ValueError(f"Unsupported aggregation method chosen: {aggregation}. Choose either 'SUM', 'MEAN', or 'MAX'.")

    def _score_routes(self, query_results: list[dict]) -> list[tuple[str, float, list[float]]]:
        """Score the routes based on the query results.

        :param query_results: The query results to score.
        :type query_results: List[Dict]
        :return: A tuple of routes, their total scores, and their individual scores.
        """
        scores_by_class = self.group_scores_by_class(query_results)
        if self.aggregation_method is None:
            raise ValueError('self.aggregation_method is not set.')
        total_scores = [(route, self.aggregation_method(scores), scores) for route, scores in scores_by_class.items()]
        total_scores.sort(key=lambda x: x[1], reverse=True)
        return total_scores

    @deprecated('Direct use of `_semantic_classify` is deprecated. Use `__call__` or `acall` instead.')
    def _semantic_classify(self, query_results: List[Dict]) -> Tuple[str, List[float]]:
        """Classify the query results into a single class based on the highest total score.
        If no classification is found, return an empty string and an empty list.

        :param query_results: The query results to classify. Expected format is a list of
        dictionaries with "route" and "score" keys.
        :type query_results: List[Dict]
        :return: A tuple containing the top class and its associated scores.
        :rtype: Tuple[str, List[float]]
        """
        top_class, top_score, scores = self._score_routes(query_results)[0]
        if top_class is not None:
            return (str(top_class), scores)
        else:
            logger.warning('No classification found for semantic classifier.')
            return ('', [])

    def get(self, name: str) -> Optional[Route]:
        """Get a route by name.

        :param name: The name of the route to get.
        :type name: str
        :return: The route.
        :rtype: Optional[Route]
        """
        for route in self.routes:
            if route.name == name:
                return route
        logger.error(f'Route `{name}` not found')
        return None

    def group_scores_by_class(self, query_results: List[Dict]) -> Dict[str, List[float]]:
        """Group the scores by class.

        :param query_results: The query results to group. Expected format is a list of
        dictionaries with "route" and "score" keys.
        :type query_results: List[Dict]
        :return: A dictionary of route names and their associated scores.
        :rtype: Dict[str, List[float]]
        """
        scores_by_class: Dict[str, List[float]] = {}
        for result in query_results:
            score = result['score']
            route = result['route']
            if route in scores_by_class:
                scores_by_class[route].append(score)
            else:
                scores_by_class[route] = [score]
        return scores_by_class

    def _update_thresholds(self, route_thresholds: Optional[Dict[str, float]]=None):
        """Update the score thresholds for each route using a dictionary of
        route names and thresholds.

        :param route_thresholds: A dictionary of route names and thresholds.
        :type route_thresholds: Dict[str, float] | None
        """
        if route_thresholds:
            for route, threshold in route_thresholds.items():
                self.set_threshold(threshold=threshold, route_name=route)

    def set_threshold(self, threshold: float, route_name: str | None=None):
        """Set the score threshold for a specific route or all routes. A `threshold` of 0.0
        will mean that the route will be returned no matter how low it scores whereas
        a threshold of 1.0 will mean that a route must contain an exact utterance match
        to be returned.

        :param threshold: The threshold to set.
        :type threshold: float
        :param route_name: The name of the route to set the threshold for. If None, the
        threshold will be set for all routes.
        :type route_name: str | None
        """
        if route_name is None:
            for route in self.routes:
                route.score_threshold = threshold
            self.score_threshold = threshold
        else:
            route_get: Route | None = self.get(route_name)
            if route_get is not None:
                route_get.score_threshold = threshold
            else:
                logger.error(f'Route `{route_name}` not found')

    def to_config(self) -> RouterConfig:
        """Convert the router to a RouterConfig object.

        :return: The RouterConfig object.
        :rtype: RouterConfig
        """
        return RouterConfig(encoder_type=self.encoder.type, encoder_name=self.encoder.name, routes=self.routes)

    def to_json(self, file_path: str):
        """Convert the router to a JSON file.

        :param file_path: The path to the JSON file.
        :type file_path: str
        """
        config = self.to_config()
        config.to_file(file_path)

    def to_yaml(self, file_path: str):
        """Convert the router to a YAML file.

        :param file_path: The path to the YAML file.
        :type file_path: str
        """
        config = self.to_config()
        config.to_file(file_path)

    def get_thresholds(self) -> Dict[str, float]:
        """Get the score thresholds for each route.

        :return: A dictionary of route names and their associated thresholds.
        :rtype: Dict[str, float]
        """
        thresholds = {route.name: route.score_threshold or self.score_threshold or 0.0 for route in self.routes}
        return thresholds

    def fit(self, X: List[str], y: List[str], batch_size: int=500, max_iter: int=500, local_execution: bool=False):
        """Fit the router to the data. Works best with a large number of examples for each
        route and with many `None` utterances.

        :param X: The input data.
        :type X: List[str]
        :param y: The output data.
        :type y: List[str]
        :param batch_size: The batch size to use for fitting.
        :type batch_size: int
        :param max_iter: The maximum number of iterations to use for fitting.
        :type max_iter: int
        :param local_execution: Whether to execute the fitting locally.
        :type local_execution: bool
        """
        original_index = self.index
        if local_execution:
            from semantic_router.index.local import LocalIndex
            remote_utterances = self.index.get_utterances(include_metadata=True)
            routes = []
            utterances = []
            metadata = []
            for utterance in remote_utterances:
                routes.append(utterance.route)
                utterances.append(utterance.utterance)
                metadata.append(utterance.metadata)
            embeddings = self.encoder(utterances)
            self.index = LocalIndex()
            self.index.add(embeddings=embeddings, routes=routes, utterances=utterances, metadata_list=metadata)
        Xq: List[List[float]] = []
        for i in tqdm(range(0, len(X), batch_size), desc='Generating embeddings'):
            emb = np.array(self.encoder(X[i:i + batch_size]))
            Xq.extend(emb)
        best_acc = self._vec_evaluate(Xq_d=np.array(Xq), y=y)
        best_thresholds = self.get_thresholds()
        for _ in (pbar := tqdm(range(max_iter), desc='Training')):
            pbar.set_postfix({'acc': round(best_acc, 2)})
            thresholds = threshold_random_search(route_layer=self, search_range=0.8)
            self._update_thresholds(route_thresholds=thresholds)
            acc = self._vec_evaluate(Xq_d=Xq, y=y)
            if acc > best_acc:
                best_acc = acc
                best_thresholds = thresholds
        self._update_thresholds(route_thresholds=best_thresholds)
        if local_execution:
            self.index = original_index

    def evaluate(self, X: List[str], y: List[str], batch_size: int=500) -> float:
        """Evaluate the accuracy of the route selection.

        :param X: The input data.
        :type X: List[str]
        :param y: The output data.
        :type y: List[str]
        :param batch_size: The batch size to use for evaluation.
        :type batch_size: int
        :return: The accuracy of the route selection.
        :rtype: float
        """
        Xq: List[List[float]] = []
        for i in tqdm(range(0, len(X), batch_size), desc='Generating embeddings'):
            emb = np.array(self.encoder(X[i:i + batch_size]))
            Xq.extend(emb)
        accuracy = self._vec_evaluate(Xq_d=np.array(Xq), y=y)
        return accuracy

    def _vec_evaluate(self, Xq_d: Union[List[float], Any], y: List[str], **kwargs) -> float:
        """Evaluate the accuracy of the route selection.

        :param Xq_d: The input data.
        :type Xq_d: Union[List[float], Any]
        :param y: The output data.
        :type y: List[str]
        :return: The accuracy of the route selection.
        :rtype: float
        """
        correct = 0
        for xq, target_route in zip(Xq_d, y):
            route_choice = self(vector=xq, simulate_static=True)
            if isinstance(route_choice, list):
                route_name = route_choice[0].name
            else:
                route_name = route_choice.name
            if route_name == target_route:
                correct += 1
        accuracy = correct / len(Xq_d)
        return accuracy

    def _get_route_names(self) -> List[str]:
        """Get the names of the routes.

        :return: The names of the routes.
        :rtype: List[str]
        """
        return [route.name for route in self.routes]

    @deprecated('Use `__call__` or `acall` with `limit=None` instead.')
    def _semantic_classify_multiple_routes(self, query_results: list[dict]) -> list[dict]:
        """Classify the query results into a list of routes.

        :param query_results: The query results to classify.
        :type query_results: List[Dict]
        :return: Most similar results with scores.
        :rtype list[dict]:
        """
        raise NotImplementedError('This method has been deprecated. Use `__call__` or `acall` with `limit=None` instead.')

class TestPretrainedTokenizer:

    @pytest.fixture
    def tokenizer(self):
        return PretrainedTokenizer('google-bert/bert-base-uncased')

    def test_initialization(self, tokenizer):
        assert tokenizer.model_ident == 'google-bert/bert-base-uncased'
        assert tokenizer.add_special_tokens is False
        assert tokenizer.pad is True

    def test_vocab_size(self, tokenizer):
        assert isinstance(tokenizer.vocab_size, int)
        assert tokenizer.vocab_size > 0

    def test_config(self, tokenizer):
        config = tokenizer.config
        assert isinstance(config, dict)
        assert 'model_ident' in config
        assert 'add_special_tokens' in config
        assert 'pad' in config

    def test_tokenize_single_text(self, tokenizer):
        text = 'Hello world'
        tokens = tokenizer.tokenize(text)
        assert isinstance(tokens, np.ndarray)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 1
        assert tokens.shape[1] > 0

    def test_tokenize_multiple_texts(self, tokenizer):
        texts = ['Hello world', 'Testing tokenization']
        tokens = tokenizer.tokenize(texts)
        assert isinstance(tokens, np.ndarray)
        assert tokens.ndim == 2
        assert tokens.shape[0] == 2

    def test_save_load_cycle(self, tokenizer):
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            tokenizer.save(tmp.name)
            loaded = PretrainedTokenizer.load(tmp.name)
            assert isinstance(loaded, PretrainedTokenizer)
            assert loaded.model_ident == tokenizer.model_ident
            assert loaded.add_special_tokens == tokenizer.add_special_tokens
            assert loaded.pad == tokenizer.pad

