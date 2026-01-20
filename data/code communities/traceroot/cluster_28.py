# Cluster 28

def test_get_openai_tool_schema():
    """Test that get_openai_tool_schema generates
    the correct schema for get_weather function.
    """
    result = get_openai_tool_schema(get_weather)
    assert result['type'] == 'function'

def get_openai_tool_schema(func: Callable) -> dict[str, Any]:
    """Generates an OpenAI JSON schema from a given Python function.

    This function creates a schema compatible with OpenAI's API specifications,
    based on the provided Python function. It processes the function's
    parameters, types, and docstrings, and constructs a schema accordingly.

    Note:
        - Each parameter in `func` must have a type annotation; otherwise, it's
          treated as 'Any'.
        - Variable arguments (*args) and keyword arguments (**kwargs) are not
          supported and will be ignored.
        - A functional description including a brief and detailed explanation
          should be provided in the docstring of `func`.
        - All parameters of `func` must be described in its docstring.
        - Supported docstring styles: ReST, Google, Numpydoc, and Epydoc.

    Args:
        func (Callable): The Python function to be converted into an OpenAI
                         JSON schema.

    Returns:
        dict[str, Any]: A dictionary representing the OpenAI JSON schema of
                        the provided function.

    See Also:
        `OpenAI API Reference
            <https://platform.openai.com/docs/api-reference/assistants/object>`_
    """
    params: Mapping[str, Parameter] = signature(func).parameters
    fields: dict[str, tuple[type, FieldInfo]] = {}
    for param_name, p in params.items():
        param_type = p.annotation
        param_default = p.default
        param_kind = p.kind
        param_annotation = p.annotation
        if param_kind == Parameter.VAR_POSITIONAL or param_kind == Parameter.VAR_KEYWORD:
            continue
        if param_annotation is Parameter.empty:
            param_type = Any
        if param_default is Parameter.empty:
            fields[param_name] = (param_type, FieldInfo())
        else:
            fields[param_name] = (param_type, FieldInfo(default=param_default))

    def _create_mol(name, field):
        return create_model(name, **field)
    model = _create_mol(to_pascal(func.__name__), fields)
    parameters_dict = model.model_json_schema()
    _remove_title_recursively(parameters_dict)
    docstring = parse(func.__doc__ or '')
    for param in docstring.params:
        if (name := param.arg_name) in parameters_dict['properties'] and (description := param.description):
            parameters_dict['properties'][name]['description'] = description
    short_description = docstring.short_description or ''
    long_description = docstring.long_description or ''
    if long_description:
        func_description = f'{short_description}\n{long_description}'
    else:
        func_description = short_description
    parameters_dict['additionalProperties'] = False
    openai_function_schema = {'name': func.__name__, 'description': func_description, 'strict': True, 'parameters': parameters_dict}
    openai_tool_schema = {'type': 'function', 'function': openai_function_schema}
    return openai_tool_schema

def _create_mol(name, field):
    return create_model(name, **field)

def to_pascal(snake: str) -> str:
    """Convert a snake_case string to PascalCase.

    Args:
        snake (str): The snake_case string to be converted.

    Returns:
        str: The converted PascalCase string.
    """
    if re.match('^[A-Z](?:[a-z0-9]*[A-Z])*[a-z0-9]*$', snake):
        return snake
    snake = snake.strip('_')
    snake = re.sub('_+', '_', snake)
    return re.sub('_([0-9A-Za-z])', lambda m: m.group(1).upper(), snake.title())

