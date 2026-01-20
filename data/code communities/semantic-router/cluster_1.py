# Cluster 1

def is_valid(route_config: str) -> bool:
    """Check if the route config is valid.

    :param route_config: The route config to check.
    :type route_config: str
    :return: Whether the route config is valid.
    :rtype: bool
    """
    try:
        output_json = json.loads(route_config)
        required_keys = ['name', 'utterances']
        if isinstance(output_json, list):
            for item in output_json:
                missing_keys = [key for key in required_keys if key not in item]
                if missing_keys:
                    logger.warning(f'Missing keys in route config: {', '.join(missing_keys)}')
                    return False
            return True
        else:
            missing_keys = [key for key in required_keys if key not in output_json]
            if missing_keys:
                logger.warning(f'Missing keys in route config: {', '.join(missing_keys)}')
                return False
            else:
                return True
    except json.JSONDecodeError as e:
        logger.error(e)
        return False

class Route(BaseModel):
    """A route for the semantic router.

    :param name: The name of the route.
    :type name: str
    :param utterances: The utterances of the route.
    :type utterances: Union[List[str], List[Any]]
    :param description: The description of the route.
    :type description: Optional[str]
    :param function_schemas: The function schemas of the route.
    :type function_schemas: Optional[List[Dict[str, Any]]]
    :param llm: The LLM to use.
    :type llm: Optional[BaseLLM]
    :param score_threshold: The score threshold of the route.
    :type score_threshold: Optional[float]
    :param metadata: The metadata of the route.
    :type metadata: Optional[Dict[str, Any]]
    """
    name: str
    utterances: Union[List[str], List[Any]]
    description: Optional[str] = None
    function_schemas: Optional[List[Dict[str, Any]]] = None
    llm: Optional[BaseLLM] = None
    score_threshold: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = {}
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, query: Optional[str]=None) -> RouteChoice:
        """Call the route. If dynamic routes have been provided the query must have been
        provided and the llm attribute must be set.

        :param query: The query to pass to the route.
        :type query: Optional[str]
        :return: The route choice.
        :rtype: RouteChoice
        """
        if self.function_schemas:
            if not self.llm:
                raise ValueError('LLM is required for dynamic routes. Please ensure the `llm` attribute is set.')
            elif query is None:
                raise ValueError('Query is required for dynamic routes. Please ensure the `query` argument is passed.')
            try:
                extracted_inputs = self.llm.extract_function_inputs(query=query, function_schemas=self.function_schemas)
                func_call = extracted_inputs
            except Exception:
                logger.error('Error extracting function inputs', exc_info=True)
                func_call = None
        else:
            func_call = None
        return RouteChoice(name=self.name, function_call=func_call)

    async def acall(self, query: Optional[str]=None) -> RouteChoice:
        """Asynchronous call the route. If dynamic routes have been provided the query
        must have been provided and the llm attribute must be set.

        :param query: The query to pass to the route.
        :type query: Optional[str]
        :return: The route choice.
        :rtype: RouteChoice
        """
        if self.function_schemas:
            if not self.llm:
                raise ValueError('LLM is required for dynamic routes. Please ensure the `llm` attribute is set.')
            elif query is None:
                raise ValueError('Query is required for dynamic routes. Please ensure the `query` argument is passed.')
            try:
                extracted_inputs = await self.llm.async_extract_function_inputs(query=query, function_schemas=self.function_schemas)
                func_call = extracted_inputs
            except Exception:
                logger.error('Error extracting function inputs', exc_info=True)
                func_call = None
        else:
            func_call = None
        return RouteChoice(name=self.name, function_call=func_call)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the route to a dictionary.

        :return: The dictionary representation of the route.
        :rtype: Dict[str, Any]
        """
        data = self.dict()
        if self.llm is not None:
            data['llm'] = {'module': self.llm.__module__, 'class': self.llm.__class__.__name__, 'model': self.llm.name}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a Route object from a dictionary.

        :param data: The dictionary to create the route from.
        :type data: Dict[str, Any]
        :return: The created route.
        :rtype: Route
        """
        return cls(**data)

    @classmethod
    def from_dynamic_route(cls, llm: BaseLLM, entities: List[Union[BaseModel, Callable]], route_name: str):
        """Generate a dynamic Route object from a list of functions or Pydantic models
        using an LLM.

        :param llm: The LLM to use.
        :type llm: BaseLLM
        :param entities: The entities to use.
        :type entities: List[Union[BaseModel, Callable]]
        :param route_name: The name of the route.
        """
        schemas = function_call.get_schema_list(items=entities)
        dynamic_route = cls._generate_dynamic_route(llm=llm, function_schemas=schemas, route_name=route_name)
        dynamic_route.function_schemas = schemas
        return dynamic_route

    @classmethod
    def _parse_route_config(cls, config: str) -> str:
        """Parse the route config from the LLM output using regex. Expects the output
        content to be wrapped in <config></config> tags.

        :param config: The LLM output.
        :type config: str
        :return: The parsed route config.
        :rtype: str
        """
        config_pattern = '<config>(.*?)</config>'
        match = re.search(config_pattern, config, re.DOTALL)
        if match:
            config_content = match.group(1).strip()
            return config_content
        else:
            raise ValueError('No <config></config> tags found in the output.')

    @classmethod
    def _generate_dynamic_route(cls, llm: BaseLLM, function_schemas: List[Dict[str, Any]], route_name: str):
        """Generate a dynamic Route object from a list of function schemas using an LLM.

        :param llm: The LLM to use.
        :type llm: BaseLLM
        :param function_schemas: The function schemas to use.
        :type function_schemas: List[Dict[str, Any]]
        :param route_name: The name of the route.
        """
        formatted_schemas = '\n'.join([json.dumps(schema, indent=4) for schema in function_schemas])
        prompt = f'\n        You are tasked to generate a single JSON configuration for multiple function schemas. \n        Each function schema should contribute five example utterances. \n        Please follow the template below, no other tokens allowed:\n\n        <config>\n        {{\n            "name": "{route_name}",\n            "utterances": [\n                "<example_utterance_1>",\n                "<example_utterance_2>",\n                "<example_utterance_3>",\n                "<example_utterance_4>",\n                "<example_utterance_5>"]\n        }}\n        </config>\n\n        Only include the "name" and "utterances" keys in your answer.\n        The "name" should match the provided route name and the "utterances"\n        should comprise a list of 5 example phrases for each function schema that could be used to invoke\n        the functions. Use real values instead of placeholders.\n\n        Input schemas:\n        {formatted_schemas}\n        '
        llm_input = [Message(role='user', content=prompt)]
        output = llm(llm_input)
        if not output:
            raise Exception('No output generated for dynamic route')
        route_config = cls._parse_route_config(config=output)
        if is_valid(route_config):
            route_config_dict = json.loads(route_config)
            route_config_dict['llm'] = llm
            return Route.from_dict(route_config_dict)
        raise Exception('No config generated')

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

def test_is_valid_with_valid_json():
    valid_json = '{"name": "test_route", "utterances": ["hello", "hi"]}'
    assert is_valid(valid_json) is True

def test_is_valid_with_missing_keys():
    invalid_json = '{"name": "test_route"}'
    with patch('semantic_router.route.logger') as mock_logger:
        assert is_valid(invalid_json) is False
        mock_logger.warning.assert_called_once()

def test_is_valid_with_valid_json_list():
    valid_json_list = '[{"name": "test_route1", "utterances": ["hello"]}, {"name": "test_route2", "utterances": ["hi"]}]'
    assert is_valid(valid_json_list) is True

def test_is_valid_with_invalid_json_list():
    invalid_json_list = '[{"name": "test_route1"}, {"name": "test_route2", "utterances": ["hi"]}]'
    with patch('semantic_router.route.logger') as mock_logger:
        assert is_valid(invalid_json_list) is False
        mock_logger.warning.assert_called_once()

def test_is_valid_with_invalid_json():
    invalid_json = '{"name": "test_route", "utterances": ["hello", "hi" invalid json}'
    with patch('semantic_router.route.logger') as mock_logger:
        assert is_valid(invalid_json) is False
        mock_logger.error.assert_called_once()

