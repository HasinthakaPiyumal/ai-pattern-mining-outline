# Cluster 14

def _parse_type_node(node, names=None) -> Any:
    """Recursively parse an AST node representing a type annotation.

    This function converts Python's Abstract Syntax Tree (AST) nodes into actual Python types.
    It's used to parse type annotations in signature strings like "x: List[int] -> y: str".

    Examples:
        - For "x: int", the AST node represents 'int' and returns the int type
        - For "x: List[str]", it processes a subscript node to return typing.List[str]
        - For "x: Optional[int]", it handles the Union type to return Optional[int]
        - For "x: MyModule.CustomType", it processes attribute access to return the actual type

    Args:
        node: An AST node from Python's ast module, representing a type annotation.
            Common node types include:
            - ast.Name: Simple types like 'int', 'str'
            - ast.Attribute: Nested types like 'typing.List'
            - ast.Subscript: Generic types like 'List[int]'
        names: Optional dictionary mapping type names to their actual type objects.
            Defaults to Python's typing module contents plus NoneType.

    Returns:
        The actual Python type represented by the AST node.

    Raises:
        ValueError: If the AST node represents an unknown or invalid type annotation.
    """
    if names is None:
        names = dict(typing.__dict__)
        names['NoneType'] = type(None)

    def resolve_name(type_name: str):
        if type_name in names:
            return names[type_name]
        builtin_types = [int, str, float, bool, list, tuple, dict, set, frozenset, complex, bytes, bytearray]
        for t in builtin_types:
            if t.__name__ == type_name:
                return t
        try:
            mod = importlib.import_module(type_name)
            names[type_name] = mod
            return mod
        except ImportError:
            pass
        raise ValueError(f'Unknown name: {type_name}')
    if isinstance(node, ast.Module):
        if len(node.body) != 1:
            raise ValueError(f'Code is not syntactically valid: {ast.dump(node)}')
        return _parse_type_node(node.body[0], names)
    if isinstance(node, ast.Expr):
        return _parse_type_node(node.value, names)
    if isinstance(node, ast.Name):
        return resolve_name(node.id)
    if isinstance(node, ast.Attribute):
        base = _parse_type_node(node.value, names)
        attr_name = node.attr
        if hasattr(base, attr_name):
            return getattr(base, attr_name)
        else:
            raise ValueError(f'Unknown attribute: {attr_name} on {base}')
    if isinstance(node, ast.Subscript):
        base_type = _parse_type_node(node.value, names)
        slice_node = node.slice
        if isinstance(slice_node, ast.Index):
            slice_node = slice_node.value
        if isinstance(slice_node, ast.Tuple):
            arg_types = tuple((_parse_type_node(elt, names) for elt in slice_node.elts))
        else:
            arg_types = (_parse_type_node(slice_node, names),)
        if base_type is typing.Union:
            return typing.Union[arg_types]
        if base_type is typing.Optional:
            if len(arg_types) != 1:
                raise ValueError('Optional must have exactly one type argument')
            return typing.Optional[arg_types[0]]
        return base_type[arg_types]
    if isinstance(node, ast.Tuple):
        return tuple((_parse_type_node(elt, names) for elt in node.elts))
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'Field'):
        keys = [kw.arg for kw in node.keywords]
        values = []
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant):
                values.append(kw.value.value)
            else:
                values.append(_parse_type_node(kw.value, names))
        return Field(**dict(zip(keys, values)))
    raise ValueError(f'Failed to parse string-base Signature due to unhandled AST node type in annotation: {ast.dump(node)}. Please consider using class-based DSPy Signatures instead.')

def resolve_name(type_name: str):
    if type_name in names:
        return names[type_name]
    builtin_types = [int, str, float, bool, list, tuple, dict, set, frozenset, complex, bytes, bytearray]
    for t in builtin_types:
        if t.__name__ == type_name:
            return t
    try:
        mod = importlib.import_module(type_name)
        names[type_name] = mod
        return mod
    except ImportError:
        pass
    raise ValueError(f'Unknown name: {type_name}')

def _parse_field_string(field_string: str) -> Dict[str, str]:
    """Extract the field name and type from field string in the string-based Signature.

    It takes a string like "x: int, y: str" and returns a dictionary mapping field names to their types.
    For example, "x: int, y: str" -> [("x", int), ("y", str)]. This function utitlizes the Python AST to parse the
    fields and types.
    """
    args = ast.parse(f'def f({field_string}): pass').body[0].args.args
    names = [arg.arg for arg in args]
    types = [str if arg.annotation is None else _parse_type_node(arg.annotation) for arg in args]
    return zip(names, types)

def _parse_signature(signature: str) -> Dict[str, Tuple[Type, Field]]:
    if signature.count('->') != 1:
        raise ValueError(f"Invalid signature format: '{signature}', must contain exactly one '->'.")
    inputs_str, outputs_str = signature.split('->')
    fields = {}
    for field_name, field_type in _parse_field_string(inputs_str):
        fields[field_name] = (field_type, InputField())
    for field_name, field_type in _parse_field_string(outputs_str):
        fields[field_name] = (field_type, OutputField())
    return fields

def make_signature(signature: Union[str, Dict[str, Tuple[type, FieldInfo]]], instructions: Optional[str]=None, signature_name: str='StringSignature', extra_fields: Optional[Dict[str, Tuple[type, FieldInfo]]]=None) -> Type[Signature]:
    """Create a new Signature subclass with the specified fields and instructions."""
    fields = _parse_signature(signature) if isinstance(signature, str) else signature
    fixed_fields = {}
    for name, type_field in fields.items():
        if not isinstance(name, str):
            raise ValueError(f'Field names must be strings, but received: {name}.')
        if isinstance(type_field, FieldInfo):
            type_ = type_field.annotation
            field = type_field
        else:
            if not isinstance(type_field, tuple):
                raise ValueError(f'Field values must be tuples, but received: {type_field}.')
            type_, field = type_field
        if type_ is None:
            type_ = str
        if not isinstance(type_, (type, typing._GenericAlias, types.GenericAlias, typing._SpecialForm)):
            raise ValueError(f'Field types must be types, but received: {type_} of type {type(type_)}.')
        if not isinstance(field, FieldInfo):
            raise ValueError(f'Field values must be Field instances, but received: {field}.')
        fixed_fields[name] = (type_, field)
    if extra_fields:
        fixed_fields.update(extra_fields)
    if instructions is None:
        sig = Signature(signature, '')
        instructions = _default_instructions(sig)
    return create_model(signature_name, __base__=Signature, __doc__=instructions, **fixed_fields)

def _default_instructions(cls) -> str:
    inputs_ = ', '.join([f'`{field}`' for field in cls.input_fields])
    outputs_ = ', '.join([f'`{field}`' for field in cls.output_fields])
    return f'Given the fields {inputs_}, produce the fields {outputs_}.'

def signature_from_registry(registry: MiproRegistry) -> Dict[str, Type[Signature]]:
    signature_dict = {}
    signature_name2register_name = {}
    for key in registry.names():
        registered_element: Union[str, PromptTemplate] = registry.get(key)
        input_names = registry.get_input_names(key)
        output_names = registry.get_output_names(key)
        sig = {}
        if isinstance(registered_element, str):
            instructions = registered_element
        elif isinstance(registered_element, PromptTemplate):
            instructions = registered_element.instruction
        check_input_placeholders(instructions, input_names, key)
        for name in input_names:
            input_desc = registry.get_input_desc(key, name)
            if input_desc:
                sig[name] = (str, InputField(desc=input_desc))
            else:
                sig[name] = (str, InputField(desc=f'The Input for prompt `{key}`.'))
        for name in output_names:
            output_desc = registry.get_output_desc(key, name)
            if output_desc:
                sig[name] = (str, OutputField(desc=output_desc))
            else:
                sig[name] = (str, OutputField(desc=f'The Output for prompt `{key}`.'))
        if is_valid_identifier(key):
            signature_name = f'{key}Signature'
        else:
            print(f'Warning: The key `{key}` is not a valid identifier, so we will add an underscore to it.')
            signature_name = f'DefaultSignature_{len(signature_dict)}'
        signature_class = make_signature(signature=sig, instructions=instructions, signature_name=signature_name)
        signature_class.__pydantic_extra__ = {'register_name': key}
        signature_dict[signature_name] = signature_class
        signature_name2register_name[signature_name] = key
    return (signature_dict, signature_name2register_name)

def check_input_placeholders(instruction: str, input_names: list[str], key: str):
    placeholders = set(re.findall('\\{(\\w+)\\}', instruction))
    input_names_set = set(input_names or [])
    missing = placeholders - input_names_set
    if missing:
        warnings.warn(f'[{key}] Missing input_names for placeholders in instruction: {missing}')

def is_valid_identifier(key: str) -> bool:
    return key.isidentifier() and (not keyword.iskeyword(key))

class PromptTuningModule(dspy.Module):
    """
    A prompt tuning module that manages interactions between predictors,
    parameter registry, and program functions.
    
    This module coordinates prompt optimization through:
    1. Maintaining a set of predictors for different tasks
    2. Synchronizing optimized parameters back to the program
    3. Executing the program with updated parameters
    
    Parameters
    ----------
    program : Union[Callable[..., dict], Callable[..., Awaitable[dict]]]
        The main program function to execute. Can be either synchronous or asynchronous.
        Must return a dictionary containing execution results.
    signature_dict : Dict[str, dspy.Signature]
        A mapping of task names to their corresponding DSPy signatures.
        Each signature defines the input/output structure for a specific task.
    registry : ParamRegistry
        A registry that maintains tunable parameters shared between
        predictors and the program.
    """

    @classmethod
    def from_registry(cls, program: Union[Callable[..., dict], Callable[..., Awaitable[dict]]], registry: ParamRegistry) -> 'PromptTuningModule':
        """
        Factory method to create a PromptTuningModule from a registry and program.
        
        This method:
        1. Creates signatures for each field in the registry
        2. Initializes a PromptTuningModule with the program and signatures
        3. Sets up predictors for each signature
        
        Parameters
        ----------
        program : Union[Callable[..., dict], Callable[..., Awaitable[dict]]]
            The main program function to execute
        registry : ParamRegistry
            Registry containing tunable parameters
            
        Returns
        -------
        PromptTuningModule
            A configured PromptTuningModule instance
            
        Examples
        --------
        >>> registry = ParamRegistry()
        >>> registry.register("task1", "What is {topic}?")
        >>> registry.register("task2", PromptTemplate(system="You are helpful.", user="{query}"))
        >>> def my_program(**kwargs) -> dict:
        ...     return {"result": "done"}
        >>> module = PromptTuningModule.from_registry(my_program, registry)
        """
        from .signature_utils import signature_from_registry
        signature_dict, signature_name2register_name = signature_from_registry(registry=registry)
        return cls(program=program, signature_dict=signature_dict, registry=registry, signature_name2register_name=signature_name2register_name)

    def __init__(self, program: Union[Callable[..., dict], Callable[..., Awaitable[dict]]], signature_dict: Dict[str, dspy.Signature], registry: ParamRegistry, signature_name2register_name: Dict[str, str]):
        """
        Initialize a PromptTuningModule instance.
        
        Parameters
        ----------
        program : Union[Callable[..., dict], Callable[..., Awaitable[dict]]]
            The main program function to execute
        signature_dict : Dict[str, dspy.Signature]
            Mapping of task names to signatures
        registry : ParamRegistry
            Parameter registry
        signature_name2register_name : Dict[str, str]
            Mapping of signature names to register names
        """
        super().__init__()
        self.program = program
        self.predicts = []
        seen = set()
        for name, signature in signature_dict.items():
            if name in seen:
                raise ValueError(f'Duplicate name {name} in signature_dict')
            seen.add(name)
            self.predicts.append(dspy.Predict(signature, name=name))
        self.registry = registry
        self.signature_name2register_name = signature_name2register_name

    def reset(self):
        """
        Reset the module to its initial state.
        """
        self.registry.reset()
        for predict in self.predicts:
            signature = predict.signature
            signature_name = signature.__name__
            register_name = self.signature_name2register_name[signature_name]
            register_element = self.registry.get(register_name)
            if isinstance(register_element, PromptTemplate):
                predict.signature.instructions = register_element.instruction
                predict.demos = register_element.demonstrations
            elif isinstance(register_element, str):
                predict.signature.instructions = register_element
                predict.demos = []
            else:
                logger.warning(f'Unsupported register element type: {type(register_element)}')
        return self

    def escape_braces(self, text):
        """
        Escape all braces in the text.
        
        Parameters
        ----------
        text : str
            Text that needs escaping
            
        Returns
        -------
        str
            Escaped text
        """

        def helper(s, start=0):
            result = ''
            i = start
            while i < len(s):
                if s[i] == '{':
                    inner, new_i = helper(s, i + 1)
                    result += '{{' + inner + '}}'
                    i = new_i
                elif s[i] == '}':
                    return (result, i + 1)
                else:
                    result += s[i]
                    i += 1
            return (result, i)
        escaped, _ = helper(text)
        return escaped

    def _validate_prompt(self, prompt: str, input_names: List[str], verbose: bool=True) -> str:
        """
        Validate if the generated prompt is valid. Currently only checks if required inputs are wrapped in braces.
        
        Parameters
        ----------
        prompt : str
            The prompt to validate
        input_names : List[str]
            List of required input names
        verbose : bool, optional
            Whether to show detailed information, defaults to True
            
        Returns
        -------
        str
            Validated and potentially modified prompt
        """
        modified_messages = []
        required_inputs = input_names
        missing_required_inputs = [name for name in required_inputs if f'{{{name}}}' not in prompt]
        if missing_required_inputs:
            input_values = '\n\n'.join([f'{name}: {{{name}}}' for name in missing_required_inputs])
            prompt += f'\n\nThe followings are some required input values: \n{input_values}'
            modified_messages.append(f'added missing inputs: {', '.join(missing_required_inputs)}')
        prompt = self.escape_braces(prompt)
        for name in input_names:
            prompt = prompt.replace(f'{{{{{name}}}}}', f'{{{name}}}')
        prompt = prompt.replace('{{{{', '{{').replace('}}}}', '}}')
        return prompt

    def get_field_type(self, field: Field) -> str:
        """
        Get the type of the field.
        
        Parameters
        ----------
        field : Field
            The field to get type from
            
        Returns
        -------
        str
            The field type
        """
        return field.json_schema_extra.get('__dspy_field_type') if field.json_schema_extra.get('__dspy_field_type') else None

    def is_prompt_template(self, register_name: str) -> bool:
        """
        Check if the register name is a prompt template.
        
        Parameters
        ----------
        register_name : str
            The register name to check
            
        Returns
        -------
        bool
            Whether it is a prompt template
        """
        return self.registry.get(register_name) is not None and isinstance(self.registry.get(register_name), PromptTemplate)

    def get_demos(self, demos: list) -> List[dict]:
        result = []
        for demo in demos:
            if isinstance(demo, dspy.Example):
                demo = demo.toDict()
            result.append(demo)
        return result

    def _inject_demos_to_string(self, instruction: str, demos: List[dict], input_names: List[str], output_names: List[str]) -> str:
        """
        Inject demos to the instruction.
        """
        if not demos:
            return instruction

        def _escape_braces(text: str) -> str:
            return text.replace('{', '{{').replace('}', '}}')

        def format_demo(demo: dict) -> str:
            demo_str = 'Inputs:\n'
            inputs = {name: demo.get(name, 'Not provided') for name in input_names}
            demo_str += '\n'.join([f'{name}:\n{_escape_braces(str(value))}' for name, value in inputs.items()])
            demo_str += '\n\nOutputs:\n'
            outputs = {name: demo.get(name, 'Not provided') for name in output_names}
            demo_str += '\n'.join([f'{name}:\n{_escape_braces(str(value))}' for name, value in outputs.items()])
            return demo_str
        demos_string = '\n\n'.join([f'Example {i + 1}:\n{format_demo(demo)}' for i, demo in enumerate(demos)])
        prompt = f'{instruction}\n\nThe following are some examples:\n{demos_string}'
        return prompt

    def sync_predict_inputs_to_program(self):
        """
        Synchronize current input values from all predictors back to the registry.
        
        This method ensures that any optimized parameters in the predictors' configurations
        are properly reflected in the registry, which in turn affects program execution.
        
        Synchronization process:
        1. Iterate through all predictors
        2. For each predictor, check its signature's input fields
        3. If a field has a value in the predictor's config, update the registry
        
        Note: Values in predictor configs take precedence as they may contain
        optimized values from recent tuning iterations.
        """
        for predict in self.predicts:
            signature = predict.signature
            instruction = signature.instructions
            demos = predict.demos
            input_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'input']
            output_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'output']
            signature_name = signature.__name__
            register_name = self.signature_name2register_name[signature_name]
            if self.is_prompt_template(register_name):
                prompt_template: PromptTemplate = self.registry.get(register_name)
                prompt_template.instruction = instruction
                prompt_template.demonstrations = self.get_demos(demos)
                self.registry.set(register_name, prompt_template)
            else:
                instruction = self._validate_prompt(instruction, input_names)
                prompt = self._inject_demos_to_string(instruction, self.get_demos(demos), input_names, output_names)
                self.registry.set(register_name, prompt)

    def constrcut_trace(self, execution_data: dict) -> dict:
        """
        Construct the trace of the execution.
        
        Parameters
        ----------
        execution_data : dict
            Execution data
            
        Returns
        -------
        dict
            Trace information
        """
        trace: List[dict] = []
        for predict in self.predicts:
            input_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'input']
            output_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'output']
            input_dict = {}
            output_dict = {}
            for name in input_names:
                if name not in execution_data:
                    logger.warning(f'Input {name} not found in execution data')
            for name in output_names:
                if name not in execution_data:
                    logger.warning(f'Output {name} not found in execution data')
            for name in input_names:
                if name in execution_data:
                    input_dict[name] = execution_data[name]
            for name in output_names:
                if name in execution_data:
                    output_dict[name] = execution_data[name]
            trace_tuple = (predict, input_dict, output_dict)
            trace.append(trace_tuple)
        return trace

    def forward(self, **kwargs) -> dict:
        """
        Execute the program with synchronized parameters and optional inputs.
        
        This method:
        1. Synchronizes optimized prompts back to the program via registry
        2. Executes the program (handles both sync and async functions)
        3. Validates and returns the program's output
        
        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments to pass to the program function
            
        Returns
        -------
        dict
            The program's execution results
            
        Raises
        ------
        ValueError
            If the program doesn't return a dictionary
        """
        self.sync_predict_inputs_to_program()
        if asyncio.iscoroutinefunction(self.program):
            output, execution_data = asyncio.run(self.program(**kwargs)) if kwargs else asyncio.run(self.program())
        else:
            output, execution_data = self.program(**kwargs) if kwargs else self.program()
        trace = self.constrcut_trace(execution_data)
        if dspy.settings.trace is not None:
            dspy_trace = dspy.settings.trace
            dspy_trace.extend(trace)
        return output

    def deepcopy(self):
        """
        Deep copy the module.
        
        This is a tweak to the default Python deepcopy that only deep copies `self.parameters()`,
        and for other attributes, we just do a shallow copy.
        
        Returns
        -------
        PromptTuningModule
            A deep copy of the module
        """
        try:
            new_instance = copy.deepcopy(self)
            setattr(new_instance, 'program', self.program)
            return new_instance
        except Exception:
            pass
        new_instance = self.__class__.__new__(self.__class__)
        for attr, value in self.__dict__.items():
            if isinstance(value, dspy.Module):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    try:
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        setattr(new_instance, attr, value)
        setattr(new_instance, 'program', self.program)
        return new_instance

    def save(self, path, save_program=False):
        """Save the module.

        Save the module to a directory or a file. There are two modes:
        - `save_program=False`: Save only the state of the module to a json or pickle file, based on the value of
            the file extension.
        - `save_program=True`: Save the whole module to a directory via cloudpickle, which contains both the state and
            architecture of the model.

        We also save the dependency versions, so that the loaded model can check if there is a version mismatch on
        critical dependencies or DSPy version.

        Args:
            path (str): Path to the saved state file, which should be a .json or .pkl file when `save_program=False`,
                and a directory when `save_program=True`.
            save_program (bool): If True, save the whole module to a directory via cloudpickle, otherwise only save
                the state.
        """
        metadata = {}
        metadata['dependency_versions'] = get_dependency_versions()
        path = Path(path)
        if not path.is_dir():
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
        elif not path.exists():
            if not path.exists():
                path.mkdir(parents=True)
        if hasattr(self.program, 'save'):
            self.program.save(str(path))
            return
        if save_program:
            if path.suffix:
                raise ValueError(f'`path` must point to a directory without a suffix when `save_program=True`, but received: {path}')
            if path.exists() and (not path.is_dir()):
                raise NotADirectoryError(f"The path '{path}' exists but is not a directory.")
            try:
                with open(path / 'program.pkl', 'wb') as f:
                    cloudpickle.dump(self, f)
            except Exception as e:
                raise RuntimeError(f'Saving failed with error: {e}. Please remove the non-picklable attributes from your DSPy program, or consider using state-only saving by setting `save_program=False`.')
            with open(path / 'metadata.json', 'w') as f:
                ujson.dump(metadata, f, indent=4)
            return
        state = self.dump_state()
        state['metadata'] = metadata
        if path.suffix == '.json':
            try:
                with open(path, 'w') as f:
                    f.write(ujson.dumps(state, indent=4))
            except Exception as e:
                raise RuntimeError(f'Failed to save state to {path} with error: {e}. Your DSPy program may contain non json-serializable objects, please consider saving the state in .pkl by using `path` ending with `.pkl`, or saving the whole program by setting `save_program=True`.')
        elif path.suffix == '.pkl':
            with open(path, 'wb') as f:
                cloudpickle.dump(state, f)
        else:
            raise ValueError(f'`path` must end with `.json` or `.pkl` when `save_program=False`, but received: {path}')

    def load(self, path):
        """Load the saved module. You may also want to check out dspy.load, if you want to
        load an entire program, not just the state for an existing program.

        Args:
            path (str): Path to the saved state file, which should be a .json or a .pkl file
        """
        path = Path(path)
        if hasattr(self.program, 'load'):
            self.program.load(str(path))
            return
        if path.suffix == '.json':
            with open(path) as f:
                state = ujson.loads(f.read())
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                state = cloudpickle.load(f)
        else:
            raise ValueError(f'`path` must end with `.json` or `.pkl`, but received: {path}')
        dependency_versions = get_dependency_versions()
        saved_dependency_versions = state['metadata']['dependency_versions']
        for key, saved_version in saved_dependency_versions.items():
            if dependency_versions[key] != saved_version:
                logger.warning(f'There is a mismatch of {key} version between saved model and current environment. You saved with `{key}=={saved_version}`, but now you have `{key}=={dependency_versions[key]}`. This might cause errors or performance downgrade on the loaded model, please consider loading the model in the same environment as the saving environment.')
        self.load_state(state)
        self.sync_predict_inputs_to_program()

