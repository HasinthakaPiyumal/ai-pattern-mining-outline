# Cluster 85

def from_crewai_tool(crewai_tool: CrewaiBaseTool, *, name: Optional[str]=None, description: Optional[str]=None) -> Callable[..., Any]:
    """
    Convert a CrewAI tool to a plain Python function.

    Args:
        crewai_tool: The CrewAI tool to convert (BaseTool or similar)
        name: Optional override for the function name
        description: Optional override for the function docstring

    Returns:
        Callable[..., Any]: Function with correct signature and metadata.
    """
    if name:
        func_name = name
    elif hasattr(crewai_tool, 'name') and crewai_tool.name:
        func_name = crewai_tool.name.replace(' ', '_').lower()
    else:
        func_name = 'crewai_tool_func'
    if description:
        func_doc = description
    elif hasattr(crewai_tool, 'description') and crewai_tool.description:
        func_doc = crewai_tool.description
    else:
        func_doc = ''
    if hasattr(crewai_tool, 'func'):
        func = crewai_tool.func
        func.__name__ = func_name
        func.__doc__ = func_doc
        return func
    elif hasattr(crewai_tool, 'args_schema') and hasattr(crewai_tool, '_run'):
        return _create_function_from_schema(crewai_tool._run, crewai_tool.args_schema, func_name, func_doc)
    elif hasattr(crewai_tool, 'run'):

        def wrapper(*args, **kwargs):
            return crewai_tool.run(*args, **kwargs)
        wrapper.__name__ = func_name
        wrapper.__doc__ = func_doc
        return wrapper
    elif callable(crewai_tool):

        def wrapper(*args, **kwargs):
            return crewai_tool(*args, **kwargs)
        wrapper.__name__ = func_name
        wrapper.__doc__ = func_doc
        try:
            wrapper.__signature__ = inspect.signature(crewai_tool)
        except (ValueError, TypeError):
            pass
        return wrapper
    else:
        raise ValueError("CrewAI tool must have a 'func', '_run', 'run' method, or be callable.")

class TestConvertCrewaiToolToFunction:
    """Test cases for convert_crewai_tool_to_function."""

    def test_tool_decorated_function_conversion(self):
        """Test conversion of @tool decorated functions."""
        fn = from_crewai_tool(sample_multiply_tool)
        assert fn.__name__ == 'sample_multiply_tool'
        assert 'Multiply two numbers together' in fn.__doc__
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['first_number', 'second_number']
        assert sig.parameters['first_number'].annotation is int
        assert sig.parameters['second_number'].annotation is int
        result = fn(5, 3)
        assert result == '15'

    def test_tool_decorated_no_args_conversion(self):
        """Test conversion of @tool decorated functions with no arguments."""
        fn = from_crewai_tool(sample_no_args_tool)
        assert fn.__name__ == 'sample_no_args_tool'
        assert 'A tool that takes no arguments' in fn.__doc__
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 0
        result = fn()
        assert result == 'Hello World'

    def test_class_based_tool_with_required_args_conversion(self):
        """Test conversion of class-based tools with required arguments."""
        tool = MultiplyTool()
        fn = from_crewai_tool(tool)
        assert fn.__name__ == 'multiply'
        assert 'Multiply two numbers' in fn.__doc__
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['first_number', 'second_number']
        assert sig.parameters['first_number'].annotation is float
        assert sig.parameters['second_number'].annotation is float
        assert sig.parameters['first_number'].default == inspect.Parameter.empty
        assert sig.parameters['second_number'].default == inspect.Parameter.empty
        result = fn(3.5, 2.0)
        assert result == 7.0

    def test_class_based_tool_with_optional_args_conversion(self):
        """Test conversion of class-based tools with optional arguments."""
        tool = GreetTool()
        fn = from_crewai_tool(tool)
        assert fn.__name__ == 'greet'
        assert 'Greet someone with a custom message' in fn.__doc__
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['name', 'greeting']
        assert sig.parameters['name'].annotation is str
        assert sig.parameters['greeting'].annotation is str
        assert sig.parameters['greeting'].default == 'Hello'
        result = fn('Alice')
        assert result == 'Hello, Alice!'
        result = fn('Bob', 'Hi')
        assert result == 'Hi, Bob!'

    def test_class_based_tool_no_args_conversion(self):
        """Test conversion of class-based tools with no arguments."""
        tool = NoArgsTool()
        fn = from_crewai_tool(tool)
        assert fn.__name__ == 'no_args_tool'
        assert 'A tool that takes no arguments' in fn.__doc__
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 0
        result = fn()
        assert result == 'No args result'

    def test_name_sanitization(self):
        """Test that tool names with spaces are properly sanitized."""
        tool = NoArgsTool()
        tool.name = 'My Custom Tool With Spaces'
        fn = from_crewai_tool(tool)
        assert fn.__name__ == 'my_custom_tool_with_spaces'

    def test_name_and_description_override(self):
        """Test that name and description can be overridden."""
        tool = MultiplyTool()
        fn = from_crewai_tool(tool, name='custom_multiply', description='Custom multiply description')
        assert fn.__name__ == 'custom_multiply'
        assert fn.__doc__ == 'Custom multiply description'

    def test_fastmcp_integration(self):
        """Test that converted functions work with FastMCP."""
        fn1 = from_crewai_tool(sample_multiply_tool)
        fast_tool1 = FastTool.from_function(fn1)
        assert fast_tool1.name == 'sample_multiply_tool'
        multiply_tool = MultiplyTool()
        fn2 = from_crewai_tool(multiply_tool)
        fast_tool2 = FastTool.from_function(fn2)
        assert fast_tool2.name == 'multiply'
        greet_tool = GreetTool()
        fn3 = from_crewai_tool(greet_tool)
        fast_tool3 = FastTool.from_function(fn3)
        assert fast_tool3.name == 'greet'
        no_args_tool = NoArgsTool()
        fn4 = from_crewai_tool(no_args_tool)
        fast_tool4 = FastTool.from_function(fn4)
        assert fast_tool4.name == 'no_args_tool'

    def test_error_handling_invalid_tool(self):
        """Test error handling for invalid tools."""

        class InvalidTool:

            def __init__(self):
                self.name = 'invalid'
                self.description = 'invalid'
        invalid_tool = InvalidTool()
        with pytest.raises(ValueError, match='CrewAI tool must have'):
            from_crewai_tool(invalid_tool)

    def test_fallback_to_run_method(self):
        """Test fallback to run method when func and _run are not available."""
        tool = Mock()
        tool.name = 'fallback tool'
        tool.description = 'A fallback tool'
        tool.run = Mock(return_value='fallback result')
        del tool.func
        del tool._run
        del tool.args_schema
        fn = from_crewai_tool(tool)
        assert fn.__name__ == 'fallback_tool'
        assert fn.__doc__ == 'A fallback tool'
        result = fn('test')
        tool.run.assert_called_once_with('test')
        assert result == 'fallback result'

    def test_signature_correctness_for_fastmcp(self):
        """Test that function signatures are correctly preserved for FastMCP."""
        multiply_tool = MultiplyTool()
        fn = from_crewai_tool(multiply_tool)
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 2
        param_names = list(sig.parameters.keys())
        assert 'first_number' in param_names
        assert 'second_number' in param_names
        for param in sig.parameters.values():
            assert param.kind != inspect.Parameter.VAR_POSITIONAL
            assert param.kind != inspect.Parameter.VAR_KEYWORD

