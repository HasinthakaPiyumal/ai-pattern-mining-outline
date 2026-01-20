# Cluster 47

def fix_json(string: str) -> str:
    string = fix_json_booleans(string)
    string = escape_json_values(string)
    return string

def fix_json_booleans(string: str) -> str:
    """
    Finds and replaces isolated "True" and "False" with "true" and "false".

    The '\x08' in the regex stands for a "word boundary", which ensures that
    we only match the full words and not substrings like "True" in "IsTrue".

    Args:
        json_string (str): The input JSON string.

    Returns:
        str: The modified JSON string with booleans in lowercase.
    """
    modified_string = regex.sub('\\bTrue\\b', 'true', string)
    modified_string = regex.sub('\\bFalse\\b', 'false', modified_string)
    return modified_string

def escape_json_values(string: str) -> str:

    def escape_value(match):
        raw_value = match.group(1)
        raw_value = raw_value.replace('\n', '\\n')
        return f'"{raw_value}"'

    def fix_json(match):
        raw_key = match.group(1)
        raw_value = match.group(2)
        raw_value = raw_value.replace('\n', '\\n')
        raw_value = regex.sub('(?<!\\\\)"', '\\"', raw_value)
        return f'"{raw_key}": "{raw_value}"'
    try:
        json.loads(string)
        return string
    except json.JSONDecodeError:
        pass
    try:
        string = regex.sub('(?<!\\\\)"', '\\"', string)
        pattern_key = '\\\\"([^"]+)\\\\"(?=\\s*:\\s*)'
        string = regex.sub(pattern_key, '"\\1"', string)
        pattern_value = '(?<=:\\s*)\\\\"((?:\\\\.|[^"\\\\])*)\\\\"'
        string = regex.sub(pattern_value, escape_value, string, flags=regex.DOTALL)
        pattern_nested_json = '"([^"]+)"\\s*:\\s*\\\\"([^"]*\\{+[\\S\\s]*?\\}+)[\\r\\n\\\\n]*"'
        string = regex.sub(pattern_nested_json, fix_json, string, flags=regex.DOTALL)
        json.loads(string)
        return string
    except json.JSONDecodeError:
        pass
    return string

def parse_json_from_text(text: str) -> List[str]:
    """
    Autoregressively extract JSON object from text 

    Args: 
        text (str): a text that includes JSON data 
    
    Returns:
        List[str]: a list of parsed JSON data
    """
    preferred, others = _extract_fenced_blocks(text)
    blocks = preferred or others or [text]
    json_pattern = '(?:\\{(?:[^{}]*|(?R))*\\}|\\[(?:[^\\[\\]]*|(?R))*\\])'
    pattern = regex.compile(json_pattern, regex.VERBOSE)
    matches: List[str] = []
    for block in blocks:
        found = pattern.findall(block)
        if found:
            matches.extend(found)
    if not matches:
        found = pattern.findall(text)
        matches.extend(found)
    matches = [fix_json(m) for m in matches]
    return matches

def _extract_fenced_blocks(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract fenced code blocks from the given text.

    Returns:
        preferred (List[str]): Code blocks explicitly labeled as json/yaml/yml.
        others (List[str]): Code blocks with other or missing language labels.
    """
    _FENCE_RE = re.compile('```(\\w+)?\\r?\\n(.*?)```', re.DOTALL)
    preferred, others = ([], [])
    for m in _FENCE_RE.finditer(text):
        lang = (m.group(1) or '').lower().strip()
        code = m.group(2)
        if lang in ('json', 'yaml', 'yml'):
            preferred.append(code)
        else:
            others.append(code)
    return (preferred, others)

def parse_json_from_llm_output(text: str) -> dict:
    """
    Extract JSON str from LLM outputs and convert it to dict. 
    """
    json_list = parse_json_from_text(text=text)
    if json_list:
        json_text = json_list[0]
        try:
            data = yaml.safe_load(json_text)
        except Exception:
            raise ValueError(f'The following generated text is not a valid JSON string!\n{json_text}')
    else:
        raise ValueError(f'The follwoing generated text does not contain JSON string!\n{text}')
    return data

class BaseModule(BaseModel, metaclass=MetaModule):
    """
    Base module class that serves as the foundation for all modules in the EvoAgentX framework.
    
    This class provides serialization/deserialization capabilities, supports creating instances from
    dictionaries, JSON, or files, and exporting instances to these formats.
    
    Attributes:
        class_name: The class name, defaults to None but is automatically set during subclass initialization
        model_config: Pydantic model configuration that controls type matching and behavior
    """
    class_name: str = None
    model_config = {'arbitrary_types_allowed': True, 'extra': 'allow', 'protected_namespaces': (), 'validate_assignment': False}

    def __init_subclass__(cls, **kwargs):
        """
        Subclass initialization method that automatically sets the class_name attribute.
        
        Args:
            cls (Type): The subclass being initialized
            **kwargs (Any): Additional keyword arguments
        """
        super().__init_subclass__(**kwargs)
        cls.class_name = cls.__name__

    def __init__(self, **kwargs):
        """
        Initializes a BaseModule instance.
        
        Args:
            **kwargs (Any): Keyword arguments used to initialize the instance
        
        Raises:
            ValidationError: When parameter validation fails
            Exception: When other errors occur during initialization
        """
        try:
            for field_name, _ in type(self).model_fields.items():
                field_value = kwargs.get(field_name, None)
                if field_value:
                    kwargs[field_name] = self._process_data(field_value)
            super().__init__(**kwargs)
            self.init_module()
        except (ValidationError, Exception) as e:
            exception_handler = callback_manager.get_callback('exception_buffer')
            if exception_handler is None:
                error_message = get_base_module_init_error_message(cls=self.__class__, data=kwargs, errors=e)
                logger.error(error_message)
                raise
            else:
                exception_handler.add(e)

    def init_module(self):
        """
        Module initialization method that subclasses can override to provide additional initialization logic.
        """
        pass

    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        
        Returns:
            str: String representation of the object
        """
        return self.to_str()

    @property
    def kwargs(self) -> dict:
        """
        Returns the extra fields of the model.
        
        Returns:
            dict: Dictionary containing all extra keyword arguments
        """
        return self.model_extra

    @classmethod
    def _create_instance(cls, data: Dict[str, Any]) -> 'BaseModule':
        """
        Internal method for creating an instance from a dictionary.
        
        Args:
            data: Dictionary containing instance data
        
        Returns:
            BaseModule: The created instance
        """
        processed_data = {k: cls._process_data(v) for k, v in data.items()}
        return cls.model_validate(processed_data)

    @classmethod
    def _process_data(cls, data: Any) -> Any:
        """
        Recursive method for processing data, with special handling for dictionaries containing class_name.
        
        Args:
            data: Data to be processed
        
        Returns:
            Processed data
        """
        if isinstance(data, dict):
            if 'class_name' in data:
                sub_class = MODULE_REGISTRY.get_module(data.get('class_name'))
                return sub_class._create_instance(data)
            else:
                return {k: cls._process_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [cls._process_data(x) for x in data]
        else:
            return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> 'BaseModule':
        """
        Instantiate the BaseModule from a dictionary.
        
        Args:
            data: Dictionary containing instance data
            **kwargs (Any): Additional keyword arguments, can include log to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            Exception: When errors occur during initialization
        """
        use_logger = kwargs.get('log', True)
        with exception_buffer() as buffer:
            try:
                class_name = data.get('class_name', None)
                if class_name:
                    cls = MODULE_REGISTRY.get_module(class_name)
                module = cls._create_instance(data)
                if len(buffer.exceptions) > 0:
                    error_message = get_base_module_init_error_message(cls, data, buffer.exceptions)
                    if use_logger:
                        logger.error(error_message)
                    raise Exception(get_error_message(buffer.exceptions))
            finally:
                pass
        return module

    @classmethod
    def from_json(cls, content: str, **kwargs) -> 'BaseModule':
        """
        Construct the BaseModule from a JSON string.
        
        This method uses yaml.safe_load to parse the JSON string into a Python object,
        which supports more flexible parsing than standard json.loads (including handling
        single quotes, trailing commas, etc). The parsed data is then passed to from_dict
        to create the instance.
        
        Args:
            content: JSON string
            **kwargs (Any): Additional keyword arguments, can include `log` to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            ValueError: When the input is not a valid JSON string
        """
        use_logger = kwargs.get('log', True)
        try:
            data = yaml.safe_load(content)
        except Exception:
            error_message = f'Can not instantiate {cls.__name__}. The input to {cls.__name__}.from_json is not a valid JSON string.'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        if not isinstance(data, (list, dict)):
            error_message = f'Can not instantiate {cls.__name__}. The input to {cls.__name__}.from_json is not a valid JSON string.'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        return cls.from_dict(data, log=use_logger)

    @classmethod
    def from_str(cls, content: str, **kwargs) -> 'BaseModule':
        """
        Construct the BaseModule from a string that may contain JSON.
        
        This method is more forgiving than `from_json` as it can extract valid JSON
        objects embedded within larger text. It uses `parse_json_from_text` to extract 
        all potential JSON strings from the input text, then tries to create an instance 
        from each extracted JSON string until successful.
        
        Args:
            content: Text that may contain JSON strings
            **kwargs (Any): Additional keyword arguments, can include `log` to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            ValueError: When the input does not contain valid JSON strings or the JSON is incompatible with the class
        """
        use_logger = kwargs.get('log', True)
        extracted_json_list = parse_json_from_text(content)
        if len(extracted_json_list) == 0:
            error_message = f'The input to {cls.__name__}.from_str does not contain any valid JSON str.'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        module = None
        for json_str in extracted_json_list:
            try:
                module = cls.from_json(json_str, log=False)
            except Exception:
                continue
            break
        if module is None:
            error_message = f'Can not instantiate {cls.__name__}. The input to {cls.__name__}.from_str either does not contain a valide JSON str, or the JSON str is incomplete or incompatable (incorrect variables or types) with {cls.__name__}.'
            error_message += f'\nInput:\n{content}'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        return module

    @classmethod
    def load_module(cls, path: str, **kwargs) -> dict:
        """
        Load the values for a module from a file.
        
        By default, it opens the specified file and uses `yaml.safe_load` to parse its contents 
        into a Python object (typically a dictionary).
        
        Args:
            path: The path of the file
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            dict: The JSON object instantiated from the file
        """
        with open(path, mode='r', encoding='utf-8') as file:
            content = yaml.safe_load(file.read())
        return content

    @classmethod
    def from_file(cls, path: str, load_function: Callable=None, **kwargs) -> 'BaseModule':
        """
        Construct the BaseModule from a file.
        
        This method reads and parses a file into a data structure, then creates
        a module instance from that data. It first verifies that the file exists,
        then uses either the provided `load_function` or the default `load_module`
        method to read and parse the file content, and finally calls `from_dict`
        to create the instance.
        
        Args:
            path: The path of the file
            load_function: The function used to load the data, takes a file path as input and returns a JSON object
            **kwargs (Any): Additional keyword arguments, can include `log` to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            ValueError: When the file does not exist
        """
        use_logger = kwargs.get('log', True)
        if not os.path.exists(path):
            error_message = f'File "{path}" does not exist!'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        function = load_function or cls.load_module
        content = function(path, **kwargs)
        module = cls.from_dict(content, log=use_logger)
        return module

    def to_dict(self, exclude_none: bool=True, ignore: List[str]=[], **kwargs) -> dict:
        """
        Convert the BaseModule to a dictionary.
        
        Args:
            exclude_none: Whether to exclude fields with None values
            ignore: List of field names to ignore
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            dict: Dictionary containing the object data
        """
        data = {}
        for field_name, _ in type(self).model_fields.items():
            if field_name in ignore:
                continue
            field_value = getattr(self, field_name, None)
            if exclude_none and field_value is None:
                continue
            if isinstance(field_value, BaseModule):
                data[field_name] = field_value.to_dict(exclude_none=exclude_none, ignore=ignore)
            elif isinstance(field_value, list):
                data[field_name] = [item.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(item, BaseModule) else item for item in field_value]
            elif isinstance(field_value, dict):
                data[field_name] = {key: value.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(value, BaseModule) else value for key, value in field_value.items()}
            else:
                data[field_name] = field_value
        return data

    def to_json(self, use_indent: bool=False, ignore: List[str]=[], **kwargs) -> str:
        """
        Convert the BaseModule to a JSON string.
        
        Args:
            use_indent: Whether to use indentation
            ignore: List of field names to ignore
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            str: The JSON string
        """
        if use_indent:
            kwargs['indent'] = kwargs.get('indent', 4)
        else:
            kwargs.pop('indent', None)
        if kwargs.get('default', None) is None:
            kwargs['default'] = custom_serializer
        data = self.to_dict(exclude_none=True)
        for ignore_field in ignore:
            data.pop(ignore_field, None)
        return json.dumps(data, **kwargs)

    def to_str(self, **kwargs) -> str:
        """
        Convert the BaseModule to a string. Use .to_json to output JSON string by default.
        
        Args:
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            str: The string
        """
        return self.to_json(use_indent=False)

    def save_module(self, path: str, ignore: List[str]=[], **kwargs) -> str:
        """
        Save the BaseModule to a file.
        
        This method will set non-serializable objects to None by default.
        If you want to save non-serializable objects, override this method.
        Remember to also override the `load_module` function to ensure the loaded
        object can be correctly parsed by `cls.from_dict`.
        
        Args:
            path: The path to save the file
            ignore: List of field names to ignore
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            str: The path where the file is saved, same as the input path
        """
        logger.info('Saving {} to {}', self.__class__.__name__, path)
        return save_json(self.to_json(use_indent=True, default=lambda x: None, ignore=ignore), path=path)

    def deepcopy(self):
        """Deep copy the module.

        This is a tweak to the default python deepcopy that only deep copies `self.parameters()`, and for other
        attributes, we just do the shallow copy.
        """
        try:
            return copy.deepcopy(self)
        except Exception:
            pass
        new_instance = self.__class__.__new__(self.__class__)
        for attr, value in self.__dict__.items():
            if isinstance(value, BaseModule):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    logging.warning(f"Failed to deep copy attribute '{attr}' of {self.__class__.__name__}, falling back to shallow copy or reference copy.")
                    try:
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        setattr(new_instance, attr, value)
        return new_instance

class LLMOutputParser(Parser):
    """A basic parser for LLM-generated content.
    
    This parser stores the raw text generated by an LLM in the `.content` attribute
    and provides methods to extract structured data from this text using different
    parsing strategies.
    
    Attributes:
        content: The raw text generated by the LLM.
    """
    content: str = Field(default=None, exclude=True, description='the text generated by LLM')

    @classmethod
    def get_attrs(cls, return_type: bool=False) -> List[Union[str, tuple]]:
        """Returns the attributes of the LLMOutputParser class.
        
        Excludes ["class_name", "content"] by default.

        Args:
            return_type: Whether to return the type of the attributes along with their names.
        
        Returns:
            If `return_type` is True, returns a list of tuples where each tuple contains 
            the attribute name and its type. Otherwise, returns a list of attribute names.
        """
        attrs = []
        exclude_attrs = ['class_name', 'content']
        for field, field_info in cls.model_fields.items():
            if field not in exclude_attrs:
                if return_type:
                    field_type = get_type_name(field_info.annotation)
                    attrs.append((field, field_type))
                else:
                    attrs.append(field)
        return attrs

    @classmethod
    def get_attr_descriptions(cls) -> dict:
        """Returns the attributes and their descriptions.
        
        Returns:
            A dictionary mapping attribute names to their descriptions.
        """
        attrs = cls.get_attrs()
        results = {}
        for field_name, field_info in cls.model_fields.items():
            if field_name not in attrs:
                continue
            field_desc = field_info.description if field_info.description is not None else 'None'
            results[field_name] = field_desc
        return results

    @classmethod
    def get_content_data(cls, content: str, parse_mode: str='json', parse_func: Optional[Callable]=None, **kwargs) -> dict:
        """Parses LLM-generated content into a dictionary.
        
        This method takes content from an LLM response and converts it to a structured
        dictionary based on the specified parsing mode.

        Args:
            content: The content to parse.
            parse_mode: The mode to parse the content. Must be one of:
                - 'str': Assigns the raw text content to all attributes of the parser. 
                - 'json': Extracts and parses JSON objects from LLM output. It will return a dictionary parsed from the first valid JSON string.
                - 'xml': Parses content using XML tags. It will return a dictionary parsed from the XML tags.
                - 'title': Parses content with Markdown-style headings.
                - 'custom': Uses custom parsing logic. Requires providing `parse_func` parameter as a custom parsing function.
            parse_func: The function to parse the content, only valid when parse_mode is 'custom'.
            **kwargs (Any): Additional arguments passed to the parsing function.
        
        Returns:
            The parsed content as a dictionary.
            
        Raises:
            ValueError: If parse_mode is invalid or if parse_func is not provided when parse_mode is 'custom'.
        """
        attrs = cls.get_attrs()
        if len(attrs) <= 0:
            return {}
        if parse_mode == 'str':
            parse_func = cls._parse_str_content
        elif parse_mode == 'json':
            parse_func = cls._parse_json_content
        elif parse_mode == 'xml':
            parse_func = cls._parse_xml_content
        elif parse_mode == 'title':
            parse_func = cls._parse_title_content
        elif parse_mode == 'custom':
            if parse_func is None:
                raise ValueError("`parse_func` must be provided when `parse_mode` is 'custom'.")
            signature = inspect.signature(parse_func)
            if 'content' not in signature.parameters:
                raise ValueError('`parse_func` must have an input argument `content`.')
            func_args = {}
            func_args['content'] = content
            for param_name, param in signature.parameters.items():
                if param_name == 'content':
                    continue
                if param_name in kwargs:
                    func_args[param_name] = kwargs[param_name]
            data = parse_func(**func_args)
            if not isinstance(data, dict):
                raise ValueError(f'The output of `parse_func` must be a dictionary, but found {type(data)}.')
            return data
        else:
            raise ValueError(f"Invalid value '{parse_mode}' detected for `parse_mode`. Available choices: {PARSER_VALID_MODE}")
        data = parse_func(content=content, **kwargs)
        return data

    @classmethod
    def _parse_str_content(cls, content: str, **kwargs) -> dict:
        """Parses content by setting all attributes to the raw content.
        
        Args:
            content: The content to parse.
            **kwargs: Additional arguments (not used).
        
        Returns:
            A dictionary mapping all attributes to the raw content.
        """
        attrs = cls.get_attrs()
        return {attr: content for attr in attrs}

    @classmethod
    def _parse_json_content(cls, content: str, **kwargs) -> dict:
        """Parses content by extracting and parsing a JSON object. 
        If the content contains multiple JSON objects, only the first one will be used. 
        
        Args:
            content: The content containing a JSON object.
            **kwargs: Additional arguments (not used).
        
        Returns:
            The parsed JSON as a dictionary.
            
        Raises:
            ValueError: If the content doesn't contain a valid JSON object.
        """
        extracted_json_list = parse_json_from_text(content)
        if len(extracted_json_list) > 0:
            json_str = extracted_json_list[0]
            try:
                data = yaml.safe_load(json_str)
                if not isinstance(data, dict):
                    if isinstance(data, list):
                        attrs = cls.get_attrs()
                        if len(attrs) == 1:
                            return {attrs[0]: data}
                        else:
                            raise ValueError('The generated content is a list of JSON strings, but the attribute name for the list is not specified. You should instruct the LLM to specify the attribute name for the list.')
                    else:
                        raise ValueError(f'The generated content is not a valid JSON string:\n{json_str}')
            except Exception:
                raise ValueError(f'The generated content is not a valid JSON string:\n{json_str}')
        else:
            raise ValueError(f'The following generated content does not contain JSON string!\n{content}')
        return data

    @classmethod
    def _parse_xml_content(cls, content: str, **kwargs) -> dict:
        """Parses content by extracting values from XML tags.
        
        Each attribute of the parser is expected to be enclosed in XML tags
        with the attribute name as the tag name.
        
        Args:
            content: The content containing XML tags.
            **kwargs: Additional arguments (not used).
        
        Returns:
            A dictionary mapping attributes to their extracted values.
            
        Raises:
            ValueError: If the content is missing expected XML tags or if the
                        extracted values can't be converted to the expected types.
        """
        attrs_with_types: List[tuple] = cls.get_attrs(return_type=True)
        data = {}
        for attr, attr_type in attrs_with_types:
            attr_raw_value_list = parse_xml_from_text(text=content, label=attr)
            if len(attr_raw_value_list) > 0:
                attr_raw_value = attr_raw_value_list[0]
                try:
                    attr_value = parse_data_from_text(text=attr_raw_value, datatype=attr_type)
                except Exception:
                    raise ValueError(f'Cannot parse text: {attr_raw_value} into {attr_type} data!')
            else:
                raise ValueError(f'The following generated content does not contain xml label <{attr}>xxx</{attr}>!\n{content}')
            data[attr] = attr_value
        return data

    @classmethod
    def _parse_title_content(cls, content: str, title_format: str='## {title}', **kwargs) -> dict:
        """Parses content with markdown-style titles.
        
        Extracts sections from content that are divided by titles following
        the specified format described in `title_format`. The default format is "## {title}".
        For example:
        ```
        ## title1
        content1
        ## title2
        content2
        ```
        This content will be parsed into:
        ```
        {
            "title1": "content1",
            "title2": "content2"
        }
        ```
        Args:
            content: The content with title-divided sections.
            title_format: The format of the titles, default is "## {title}".
            **kwargs: Additional arguments (not used).

        Returns:
            A dictionary mapping title names to their section contents.
        """
        attrs: List[str] = cls.get_attrs()
        if not attrs:
            return {}
        output_titles = [title_format.format(title=attr) for attr in attrs]

        def is_output_title(text: str):
            for title in output_titles:
                if text.strip().lower().startswith(title.lower()):
                    return (True, title)
            return (False, None)
        data = {}
        current_output_name: str = None
        current_output_content: list = None
        for line in content.split('\n'):
            is_title, title = is_output_title(line)
            if is_title:
                if current_output_name is not None and current_output_content is not None:
                    data[current_output_name] = '\n'.join(current_output_content)
                current_output_content = []
                current_output_name = title.replace('#', '').strip()
                output_titles.remove(title)
            elif current_output_content is not None:
                current_output_content.append(line)
        if current_output_name is not None and current_output_content is not None:
            data[current_output_name] = '\n'.join(current_output_content)
        return data

    @classmethod
    def parse(cls, content: str, parse_mode: str='json', parse_func: Optional[Callable]=None, **kwargs) -> 'LLMOutputParser':
        """Parses LLM-generated text into a structured parser instance.
        
        This is the main method for creating parser instances from LLM output.
        
        Args:
            content: The text generated by the LLM.
            parse_mode: The mode to parse the content, must be one of:
                - 'str': Assigns the raw text content to all attributes of the parser. 
                - 'json': Extracts and parses JSON objects from LLM output. Uses the first valid JSON string to create an instance of LLMOutputParser.
                - 'xml': Parses content using XML tags. Uses the XML tags to create an instance of LLMOutputParser.
                - 'title': Parses content with Markdown-style headings. Uses the Markdown-style headings to create an instance of LLMOutputParser. The default title format is "## {title}", you can change it by providing `title_format` parameter, which should be a string that contains `{title}` placeholder. 
                - 'custom': Uses custom parsing logic. Requires providing `parse_func` parameter as a custom parsing function. The `parse_func` must have a parameter named `content` and return a dictionary where the keys are the attribute names and the values are the parsed data. 
            parse_func: The function to parse the content, only valid when `parse_mode` is 'custom'.
            **kwargs (Any): Additional arguments passed to parsing functions, such as:
                - `title_format` for `parse_mode="title"`.
            
        Returns:
            An instance of LLMOutputParser containing the parsed data.
            
        Raises:
            ValueError: If parse_mode is invalid or if content is not a string.
        """
        if parse_mode not in PARSER_VALID_MODE:
            raise ValueError(f"'{parse_mode}' is an invalid value for `parse_mode`. Available choices: {PARSER_VALID_MODE}.")
        if not isinstance(content, str):
            raise ValueError(f'The input to {cls.__name__}.parse should be a str, but found {type(content)}.')
        data = cls.get_content_data(content=content, parse_mode=parse_mode, parse_func=parse_func, **kwargs)
        data.update({'content': content})
        parser = cls.from_dict(data, **kwargs)
        return parser

    def __str__(self) -> str:
        """
        Returns a string representation of the parser.
        """
        return self.to_str()

    def to_str(self, **kwargs) -> str:
        """
        Converts the parser to a string.
        """
        return self.content

    def get_structured_data(self) -> dict:
        """Extracts structured data from the parser.
        
        Returns:
            A dictionary containing only the defined attributes and their values,
            excluding metadata like class_name.
        """
        attrs = type(self).get_attrs()
        data = self.to_dict(ignore=['class_name'])
        structured_data = {key: value for key, value in data.items() if key in attrs}
        return structured_data

class CustomizeAction(Action):
    parse_mode: Optional[str] = Field(default='title', description="the parse mode of the action, must be one of: ['title', 'str', 'json', 'xml', 'custom']")
    parse_func: Optional[Callable] = Field(default=None, exclude=True, description='the function to parse the LLM output. It receives the LLM output and returns a dict.')
    title_format: Optional[str] = Field(default='## {title}', exclude=True, description="the format of the title. It is used when the `parse_mode` is 'title'.")
    custom_output_format: Optional[str] = Field(default=None, exclude=True, description='the format of the output. It is used when the `prompt_template` is provided.')
    tools: Optional[List[Toolkit]] = Field(default=None, description='The tools that the action can use')
    conversation: Optional[Message] = Field(default=None, description='Current conversation state')
    max_tool_try: int = Field(default=2, description='Maximum number of tool calling attempts allowed')

    def __init__(self, **kwargs):
        name = kwargs.pop('name', 'CustomizeAction')
        description = kwargs.pop('description', 'Customized action that can use tools to accomplish its task')
        super().__init__(name=name, description=description, **kwargs)
        if not self.prompt and (not self.prompt_template):
            raise ValueError('`prompt` or `prompt_template` is required when creating CustomizeAction action')
        if self.prompt and self.prompt_template:
            logger.warning('Both `prompt` and `prompt_template` are provided for CustomizeAction action. Prioritizing `prompt_template` and ignoring `prompt`.')
        if self.tools:
            self.tools_caller = {}
            self.add_tools(self.tools)

    def prepare_action_prompt(self, inputs: Optional[dict]=None, system_prompt: Optional[str]=None, **kwargs) -> Union[str, List[dict]]:
        """Prepare prompt for action execution.
        
        This helper function transforms the input dictionary into a formatted prompt
        for the language model, handling different prompting modes.
        
        Args:
            inputs: Dictionary of input parameters
            system_prompt: Optional system prompt to include
            
        Returns:
            Union[str, List[dict]]: Formatted prompt ready for LLM (string or chat messages)
            
        Raises:
            TypeError: If an input value type is not supported
            ValueError: If neither prompt nor prompt_template is available
        """
        if inputs is None:
            inputs = {}
        prompt_params_names = self.inputs_format.get_attrs()
        prompt_params_values = {}
        for param in prompt_params_names:
            value = inputs.get(param, '')
            if isinstance(value, str):
                prompt_params_values[param] = value
            elif isinstance(value, (dict, list)):
                prompt_params_values[param] = json.dumps(value, indent=4)
            else:
                raise TypeError(f'The input type {type(value)} is invalid! Valid types: [str, dict, list].')
        if self.prompt:
            prompt = self.prompt.format(**prompt_params_values) if prompt_params_values else self.prompt
            if self.tools:
                tools_schemas = [j['function'] for i in [tool.get_tool_schemas() for tool in self.tools] for j in i]
                prompt += '\n\n' + TOOL_CALLING_TEMPLATE.format(tools_description=tools_schemas)
            return prompt
        else:
            if self.tools:
                self.prompt_template.set_tools(self.tools)
            return self.prompt_template.format(system_prompt=system_prompt, values=prompt_params_values, inputs_format=self.inputs_format, outputs_format=self.outputs_format, parse_mode=self.parse_mode, title_format=self.title_format, custom_output_format=self.custom_output_format, tools=self.tools)

    def prepare_extraction_prompt(self, llm_output_content: str) -> str:
        """Prepare extraction prompt for fallback extraction when parsing fails.
        
        Args:
            self: The action instance
            llm_output_content: Raw output content from LLM
            
        Returns:
            str: Formatted extraction prompt
        """
        attr_descriptions: dict = self.outputs_format.get_attr_descriptions()
        output_description_list = []
        for i, (name, desc) in enumerate(attr_descriptions.items()):
            output_description_list.append(f'{i + 1}. {name}\nDescription: {desc}')
        output_description = '\n\n'.join(output_description_list)
        return OUTPUT_EXTRACTION_PROMPT.format(text=llm_output_content, output_description=output_description)

    def _get_unique_class_name(self, candidate_name: str) -> str:
        """
        Get a unique class name by checking if it already exists in the registry.
        If it does, append "Vx" to make it unique.
        """
        if not MODULE_REGISTRY.has_module(candidate_name):
            return candidate_name
        i = 1
        while True:
            unique_name = f'{candidate_name}V{i}'
            if not MODULE_REGISTRY.has_module(unique_name):
                break
            i += 1
        return unique_name

    def add_tools(self, tools: Union[Toolkit, List[Toolkit]]):
        if not tools:
            return
        if isinstance(tools, Toolkit):
            tools = [tools]
        if not all((isinstance(tool, Toolkit) for tool in tools)):
            raise TypeError('`tools` must be a Toolkit or list of Toolkit instances.')
        if not self.tools:
            self.tools_caller = {}
            self.tools = []
        for toolkit in tools:
            try:
                tool_callers = toolkit.get_tools()
                if not isinstance(tool_callers, list):
                    logger.warning(f"Expected list of tool functions from '{toolkit.name}.get_tools()', got {type(tool_callers)}.")
                    continue
                valid_tools_count = 0
                valid_tools_names, valid_tool_callers = ([], [])
                for tool_caller in tool_callers:
                    tool_caller_name = getattr(tool_caller, 'name', None)
                    if not tool_caller_name or not callable(tool_caller):
                        logger.warning(f"Invalid tool function in '{toolkit.name}': missing name or not callable.")
                        continue
                    if tool_caller_name in self.tools_caller:
                        logger.warning(f"Duplicate tool function '{tool_caller_name}' detected. Overwriting previous function.")
                    valid_tools_count += 1
                    valid_tools_names.append(tool_caller_name)
                    valid_tool_callers.append(tool_caller)
                if valid_tools_count == 0:
                    logger.info(f"No valid tools found in toolkit '{toolkit.name}'. Skipping.")
                    continue
                if valid_tools_count > 0 and all((name in self.tools_caller for name in valid_tools_names)):
                    logger.info(f"All tools from toolkit '{toolkit.name}' are already added. Skipping.")
                    continue
                if valid_tools_count > 0:
                    self.tools_caller.update({name: caller for name, caller in zip(valid_tools_names, valid_tool_callers)})
                existing_toolkit_names = {tkt.name for tkt in self.tools}
                if valid_tools_count > 0 and toolkit.name not in existing_toolkit_names:
                    self.tools.append(toolkit)
                if valid_tools_count > 0:
                    logger.info(f"Added toolkit '{toolkit.name}' with {valid_tools_count} valid tools in {self.name}: {valid_tools_names}.")
            except Exception as e:
                logger.error(f"Failed to load tools from toolkit '{toolkit.name}': {e}")

    def _extract_tool_calls(self, llm_output: str, llm: Optional[BaseLLM]=None) -> List[dict]:
        pattern = '<ToolCalling>\\s*(.*?)\\s*</ToolCalling>'
        matches = re.findall(pattern, llm_output, re.DOTALL)
        if not matches:
            return []
        parsed_tool_calls = []
        for match_content in matches:
            try:
                json_content = match_content.strip()
                json_list = parse_json_from_text(json_content)
                if not json_list:
                    logger.warning('No valid JSON found in ToolCalling block')
                    continue
                parsed_tool_call = json.loads(json_list[0])
                if isinstance(parsed_tool_call, dict):
                    parsed_tool_calls.append(parsed_tool_call)
                elif isinstance(parsed_tool_call, list):
                    parsed_tool_calls.extend(parsed_tool_call)
                else:
                    logger.warning(f'Invalid tool call format: {parsed_tool_call}')
                    continue
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f'Failed to parse tool calls from LLM output: {e}')
                if llm is not None:
                    retry_prompt = TOOL_CALLING_RETRY_PROMPT.format(text=match_content)
                    try:
                        fixed_output = llm.generate(prompt=retry_prompt).content.strip()
                        logger.info(f'Retrying tool call parse with fixed output:\n{fixed_output}')
                        fixed_list = parse_json_from_text(fixed_output)
                        if fixed_list:
                            parsed_tool_call = json.loads(fixed_list[0])
                            if isinstance(parsed_tool_call, dict):
                                parsed_tool_calls.append(parsed_tool_call)
                        elif isinstance(parsed_tool_call, list):
                            parsed_tool_calls.extend(parsed_tool_call)
                    except Exception as retry_err:
                        logger.error(f'Retry failed: {retry_err}')
                        continue
            else:
                continue
        return parsed_tool_calls

    def _extract_output(self, llm_output: Any, llm: BaseLLM=None, **kwargs):
        llm_output_content = getattr(llm_output, 'content', str(llm_output))
        output_attrs = self.outputs_format.get_attrs()
        if not output_attrs:
            output = self.outputs_format.parse(content=llm_output_content)
            return output
        try:
            parsed_output = self.outputs_format.parse(content=llm_output_content, parse_mode=self.parse_mode, parse_func=getattr(self, 'parse_func', None), title_format=getattr(self, 'title_format', '## {title}'))
            return parsed_output
        except Exception as e:
            logger.info(f"Failed to parse with action's parse settings: {e}")
            logger.info('Falling back to using LLM to extract outputs...')
            extraction_prompt = self.prepare_extraction_prompt(llm_output_content)
            llm_extracted_output: LLMOutputParser = llm.generate(prompt=extraction_prompt)
            llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
            output = self.outputs_format.from_dict(llm_extracted_data)
            return output

    async def _async_extract_output(self, llm_output: Any, llm: BaseLLM=None, **kwargs):
        llm_output_content = getattr(llm_output, 'content', str(llm_output))
        output_attrs = self.outputs_format.get_attrs()
        if not output_attrs:
            output = self.outputs_format.parse(content=llm_output_content)
            return output
        try:
            parsed_output = self.outputs_format.parse(content=llm_output_content, parse_mode=self.parse_mode, parse_func=getattr(self, 'parse_func', None), title_format=getattr(self, 'title_format', '## {title}'))
            return parsed_output
        except Exception as e:
            logger.info(f"Failed to parse with action's parse settings: {e}")
            logger.info('Falling back to using LLM to extract outputs...')
            extraction_prompt = self.prepare_extraction_prompt(llm_output_content)
            llm_extracted_output = await llm.async_generate(prompt=extraction_prompt)
            llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
            output = self.outputs_format.from_dict(llm_extracted_data)
            return output

    def _call_single_tool(self, function_param: dict) -> tuple:
        try:
            function_name = function_param.get('function_name')
            function_args = function_param.get('function_args') or {}
            if not function_name:
                return (None, 'No function name provided')
            callable_fn = self.tools_caller.get(function_name)
            if not callable(callable_fn):
                return (None, f"Function '{function_name}' not found or not callable")
            print('_____________________ Start Function Calling _____________________')
            print(f'Executing function calling: {function_name} with parameters: {function_args}')
            result = callable_fn(**function_args)
            return (result, None)
        except Exception as e:
            logger.error(f'Error executing tool {function_name}: {e}')
            return (None, f'Error executing tool {function_name}: {str(e)}')

    def _calling_tools(self, tool_call_args: List[dict]) -> dict:
        errors = []
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_tool = {executor.submit(self._call_single_tool, param): param for param in tool_call_args}
            for future in concurrent.futures.as_completed(future_to_tool):
                result, error = future.result()
                if error:
                    errors.append(error)
                if result is not None:
                    results.append(result)
        return {'result': results, 'error': errors}

    async def _async_call_single_tool(self, function_param: dict) -> tuple:
        try:
            function_name = function_param.get('function_name')
            function_args = function_param.get('function_args') or {}
            if not function_name:
                return (None, 'No function name provided')
            callable_fn = self.tools_caller.get(function_name)
            if not callable(callable_fn):
                return (None, f"Function '{function_name}' not found or not callable")
            print('_____________________ Start Function Calling _____________________')
            print(f'Executing function calling: {function_name} with parameters: {function_args}')
            if inspect.iscoroutinefunction(callable_fn):
                result = await callable_fn(**function_args)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: callable_fn(**function_args))
            return (result, None)
        except Exception as e:
            logger.error(f'Error executing tool {function_name}: {e}')
            return (None, f'Error executing tool {function_name}: {str(e)}')

    async def _async_calling_tools(self, tool_call_args: List[dict]) -> dict:
        tasks = [self._async_call_single_tool(param) for param in tool_call_args]
        results_with_errors = await asyncio.gather(*tasks)
        results = [res for res, err in results_with_errors if err is None and res is not None]
        errors = [err for _, err in results_with_errors if err is not None]
        return {'result': results, 'error': errors}

    def execute(self, llm: Optional[BaseLLM]=None, inputs: Optional[dict]=None, sys_msg: Optional[str]=None, return_prompt: bool=False, time_out=0, **kwargs):
        input_attributes: dict = self.inputs_format.get_attr_descriptions()
        if not inputs and input_attributes:
            logger.error('CustomizeAction action received invalid `inputs`: None or empty.')
            raise ValueError('The `inputs` to CustomizeAction action is None or empty.')
        if inputs is None:
            inputs = {}
        final_llm_response = None
        if self.prompt_template:
            if isinstance(self.prompt_template, ChatTemplate):
                conversation = self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)
            elif isinstance(self.prompt_template, StringTemplate):
                conversation = [{'role': 'system', 'content': self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
            else:
                raise ValueError(f'`prompt_template` must be a StringTemplate or ChatTemplate instance, but got {type(self.prompt_template)}')
        else:
            conversation = [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
        prompt_params_values = {k: inputs.get(k, '') for k in input_attributes.keys()}
        while True:
            if time_out > self.max_tool_try:
                current_prompt = self.prepare_action_prompt(inputs=prompt_params_values or {})
                content_to_extract = final_llm_response if final_llm_response is not None else '{content}'.format(content=conversation)
                if return_prompt:
                    return (self._extract_output(content_to_extract, llm=llm), current_prompt)
                return self._extract_output(content_to_extract, llm=llm)
            time_out += 1
            llm_response = llm.generate(messages=conversation)
            conversation.append({'role': 'assistant', 'content': llm_response.content})
            final_llm_response = llm_response
            tool_call_args = self._extract_tool_calls(llm_response.content)
            if not tool_call_args:
                break
            logger.info('Extracted tool call args:')
            logger.info(json.dumps(tool_call_args, indent=4))
            results = self._calling_tools(tool_call_args)
            logger.info('Tool call results:')
            logger.info(json.dumps(results, indent=4))
            conversation.append({'role': 'assistant', 'content': TOOL_CALLING_HISTORY_PROMPT.format(iteration_number=time_out, tool_call_args=f'{tool_call_args}', results=f'{results}')})
        current_prompt = self.prepare_action_prompt(inputs=prompt_params_values or {})
        content_to_extract = final_llm_response if final_llm_response is not None else '{content}'.format(content=conversation)
        if return_prompt:
            return (self._extract_output(content_to_extract, llm=llm), current_prompt)
        return self._extract_output(content_to_extract, llm=llm)

    async def async_execute(self, llm: Optional[BaseLLM]=None, inputs: Optional[dict]=None, sys_msg: Optional[str]=None, return_prompt: bool=False, time_out=0, **kwargs):
        input_attributes: dict = self.inputs_format.get_attr_descriptions()
        if not inputs and input_attributes:
            logger.error('CustomizeAction action received invalid `inputs`: None or empty.')
            raise ValueError('The `inputs` to CustomizeAction action is None or empty.')
        if inputs is None:
            inputs = {}
        final_llm_response = None
        if self.prompt_template:
            if isinstance(self.prompt_template, ChatTemplate):
                conversation = self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)
            elif isinstance(self.prompt_template, StringTemplate):
                conversation = [{'role': 'system', 'content': self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
            else:
                raise ValueError(f'`prompt_template` must be a StringTemplate or ChatTemplate instance, but got {type(self.prompt_template)}')
        else:
            conversation = [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': self.prepare_action_prompt(inputs=inputs, system_prompt=sys_msg)}]
        prompt_params_values = {k: inputs.get(k, '') for k in input_attributes.keys()}
        while True:
            if time_out > self.max_tool_try:
                current_prompt = self.prepare_action_prompt(inputs=prompt_params_values or {})
                content_to_extract = final_llm_response if final_llm_response is not None else '{content}'.format(content=conversation)
                if return_prompt:
                    return (await self._async_extract_output(content_to_extract, llm=llm), current_prompt)
                return await self._async_extract_output(content_to_extract, llm=llm)
            time_out += 1
            llm_response = await llm.async_generate(messages=conversation)
            conversation.append({'role': 'assistant', 'content': llm_response.content})
            final_llm_response = llm_response
            tool_call_args = self._extract_tool_calls(llm_response.content)
            if not tool_call_args:
                break
            logger.info('Extracted tool call args:')
            logger.info(json.dumps(tool_call_args, indent=4))
            results = self._calling_tools(tool_call_args)
            logger.info('Tool call results:')
            try:
                logger.info(json.dumps(results, indent=4))
            except Exception:
                logger.info(str(results))
            conversation.append({'role': 'assistant', 'content': TOOL_CALLING_HISTORY_PROMPT.format(iteration_number=time_out, tool_call_args=f'{tool_call_args}', results=f'{results}')})
        current_prompt = self.prepare_action_prompt(inputs=prompt_params_values or {})
        content_to_extract = final_llm_response if final_llm_response is not None else '{content}'.format(content=conversation)
        if return_prompt:
            return (await self._async_extract_output(content_to_extract, llm=llm), current_prompt)
        return await self._async_extract_output(content_to_extract, llm=llm)

