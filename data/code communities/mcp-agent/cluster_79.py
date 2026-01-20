# Cluster 79

def from_langchain_tool(lc_tool: Union['BaseTool', object], *, name: Optional[str]=None, description: Optional[str]=None) -> Callable[..., Any]:
    """
    Convert a LangChain tool to a plain Python function.

    Args:
        lc_tool: The LangChain tool to convert (StructuredTool, BaseTool, or similar)
        name: Optional override for the function name
        description: Optional override for the function docstring

    Returns:
        Callable[..., Any]: Function with correct signature and metadata.
    """
    func_name = name or getattr(lc_tool, 'name', getattr(lc_tool, '__name__', 'tool_func'))
    func_doc = description or getattr(lc_tool, 'description', getattr(lc_tool, '__doc__', '') or '')
    if isinstance(lc_tool, StructuredTool):
        func = lc_tool.func
        func.__name__ = func_name
        func.__doc__ = func_doc
        return func
    elif hasattr(lc_tool, '_run'):
        run_method = lc_tool._run

        def wrapper(*args, **kwargs):
            return run_method(*args, **kwargs)
        wrapper.__signature__ = inspect.signature(run_method)
        wrapper.__name__ = func_name
        wrapper.__doc__ = func_doc
        return wrapper
    elif hasattr(lc_tool, 'run'):
        run_method = lc_tool.run

        def wrapper(*args, **kwargs):
            return run_method(*args, **kwargs)
        try:
            wrapper.__signature__ = inspect.signature(run_method)
        except (ValueError, TypeError):
            pass
        wrapper.__name__ = func_name
        wrapper.__doc__ = func_doc
        return wrapper
    elif callable(lc_tool):

        def wrapper(*args, **kwargs):
            return lc_tool(*args, **kwargs)
        try:
            wrapper.__signature__ = inspect.signature(lc_tool)
        except (ValueError, TypeError):
            pass
        wrapper.__name__ = func_name
        wrapper.__doc__ = func_doc
        return wrapper
    else:
        raise ValueError("LangChain tool must have a 'func', 'run', or '_run' method, or be callable.")

class TestConvertLangchainToolToFunction:
    """Test cases for convert_langchain_tool_to_function."""

    def test_tool_decorator_conversion(self):
        """Test conversion of @tool decorated functions."""
        fn = from_langchain_tool(multiply_decorator_tool)
        assert fn.__name__ == 'multiply_decorator_tool'
        assert 'Multiply two numbers' in fn.__doc__
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['a', 'b']
        assert sig.parameters['a'].annotation is int
        assert sig.parameters['b'].annotation is int
        result = fn(5, 3)
        assert result == 15

    def test_tool_decorator_no_args_conversion(self):
        """Test conversion of @tool decorated functions with no arguments."""
        fn = from_langchain_tool(no_args_decorator_tool)
        assert fn.__name__ == 'no_args_decorator_tool'
        assert 'A tool that takes no arguments' in fn.__doc__
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 0
        result = fn()
        assert result == 'Hello from decorator'

    def test_structured_tool_from_function_conversion(self):
        """Test conversion of StructuredTool.from_function() tools."""
        structured_tool = StructuredTool.from_function(func=multiply_func)
        fn = from_langchain_tool(structured_tool)
        assert fn.__name__ == 'multiply_func'
        assert 'Multiply two numbers using function' in fn.__doc__
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['a', 'b']
        assert sig.parameters['a'].annotation is int
        assert sig.parameters['b'].annotation is int
        result = fn(7, 4)
        assert result == 28

    def test_structured_tool_with_async_conversion(self):
        """Test conversion of StructuredTool with async coroutine."""
        structured_tool = StructuredTool.from_function(func=divide_func, coroutine=divide_async_func)
        fn = from_langchain_tool(structured_tool)
        assert fn.__name__ == 'divide_func'
        assert 'Divide two numbers' in fn.__doc__
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['numerator', 'denominator']
        assert sig.parameters['numerator'].annotation is float
        assert sig.parameters['denominator'].annotation is float
        result = fn(10.0, 2.0)
        assert result == 5.0
        with pytest.raises(ValueError, match='Cannot divide by zero'):
            fn(10.0, 0.0)

    def test_base_tool_with_run_method_conversion(self):
        """Test conversion of BaseTool with _run method."""
        tool = CustomBaseTool()
        fn = from_langchain_tool(tool)
        assert fn.__name__ == 'custom_base_tool'
        assert 'A custom tool that generates random numbers' in fn.__doc__
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['count', 'min_val', 'max_val']
        assert sig.parameters['count'].annotation is int
        assert sig.parameters['min_val'].annotation is float
        assert sig.parameters['max_val'].annotation is float
        assert sig.parameters['min_val'].default == 0.0
        assert sig.parameters['max_val'].default == 1.0
        result = fn(3, 0.5, 1.5)
        assert isinstance(result, list)
        assert len(result) == 3
        for val in result:
            assert 0.5 <= val <= 1.5

    def test_complex_base_tool_conversion(self):
        """Test conversion of complex BaseTool (from user's example)."""
        tool = GenerateRandomFloats()
        fn = from_langchain_tool(tool)
        assert fn.__name__ == 'generate_random_floats'
        assert 'Generate size random floats in the range [min, max]' in fn.__doc__
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['min', 'max', 'size']
        assert sig.parameters['min'].annotation is float
        assert sig.parameters['max'].annotation is float
        assert sig.parameters['size'].annotation is int
        result = fn(0.0, 1.0, 5)
        assert isinstance(result, tuple)
        content, array = result
        assert isinstance(content, str)
        assert isinstance(array, list)
        assert len(array) == 5
        assert 'Generated 5 floats' in content

    def test_base_tool_with_run_fallback(self):
        """Test fallback to run method when _run is not available."""
        tool = Mock()
        tool.name = 'mock_tool'
        tool.description = 'A mock tool'
        tool.run = Mock(return_value='mock result')
        del tool.func
        del tool._run
        fn = from_langchain_tool(tool)
        assert fn.__name__ == 'mock_tool'
        assert fn.__doc__ == 'A mock tool'
        result = fn('test_arg')
        tool.run.assert_called_once_with('test_arg')
        assert result == 'mock result'

    def test_callable_tool_conversion(self):
        """Test conversion of plain callable tools."""

        def simple_callable(x: str, y: int=42) -> str:
            """Simple callable function."""
            return f'{x}_{y}'
        fn = from_langchain_tool(simple_callable)
        assert fn.__name__ == 'simple_callable'
        assert 'Simple callable function' in fn.__doc__
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['x', 'y']
        assert sig.parameters['x'].annotation is str
        assert sig.parameters['y'].annotation is int
        assert sig.parameters['y'].default == 42
        result = fn('test')
        assert result == 'test_42'
        result = fn('hello', 100)
        assert result == 'hello_100'

    def test_name_and_description_override(self):
        """Test that name and description can be overridden."""
        fn = from_langchain_tool(multiply_decorator_tool, name='custom_multiply', description='Custom multiply description')
        assert fn.__name__ == 'custom_multiply'
        assert fn.__doc__ == 'Custom multiply description'
        result = fn(3, 4)
        assert result == 12

    def test_name_fallback_behavior(self):
        """Test name fallback behavior for tools without explicit names."""
        tool_with_name = CustomBaseTool()
        fn1 = from_langchain_tool(tool_with_name)
        assert fn1.__name__ == 'custom_base_tool'

        def named_func():
            pass
        fn2 = from_langchain_tool(named_func)
        assert fn2.__name__ == 'named_func'
        mock_tool = Mock()
        del mock_tool.name
        mock_tool.description = 'test'
        mock_tool.run = Mock(return_value='test')
        del mock_tool.func
        del mock_tool._run
        del mock_tool.__name__
        fn3 = from_langchain_tool(mock_tool)
        assert fn3.__name__ == 'tool_func'

    def test_description_fallback_behavior(self):
        """Test description fallback behavior for tools without explicit descriptions."""

        def func_with_docstring():
            """Function docstring."""
            pass
        fn1 = from_langchain_tool(func_with_docstring)
        assert fn1.__doc__ == 'Function docstring.'
        mock_tool = Mock()
        mock_tool.name = 'test_tool'
        del mock_tool.description
        mock_tool.run = Mock(return_value='test')
        del mock_tool.func
        del mock_tool._run
        mock_tool.__doc__ = 'Mock docstring'
        fn2 = from_langchain_tool(mock_tool)
        assert fn2.__doc__ == 'Mock docstring'
        mock_tool2 = Mock()
        mock_tool2.name = 'test_tool2'
        del mock_tool2.description
        mock_tool2.run = Mock(return_value='test')
        del mock_tool2.func
        del mock_tool2._run
        mock_tool2.__doc__ = None
        fn3 = from_langchain_tool(mock_tool2)
        assert fn3.__doc__ == ''

    def test_error_handling_invalid_tool(self):
        """Test error handling for invalid tools."""

        class InvalidTool:

            def __init__(self):
                self.name = 'invalid'
                self.description = 'invalid'
        invalid_tool = InvalidTool()
        with pytest.raises(ValueError, match='LangChain tool must have'):
            from_langchain_tool(invalid_tool)

    def test_fastmcp_integration(self):
        """Test that converted functions work with FastMCP."""
        fn1 = from_langchain_tool(multiply_decorator_tool)
        fast_tool1 = FastTool.from_function(fn1)
        assert fast_tool1.name == 'multiply_decorator_tool'
        structured_tool = StructuredTool.from_function(func=multiply_func)
        fn2 = from_langchain_tool(structured_tool)
        fast_tool2 = FastTool.from_function(fn2)
        assert fast_tool2.name == 'multiply_func'
        base_tool = CustomBaseTool()
        fn3 = from_langchain_tool(base_tool)
        fast_tool3 = FastTool.from_function(fn3)
        assert fast_tool3.name == 'custom_base_tool'

        def simple_func(x: int) -> int:
            return x * 2
        fn4 = from_langchain_tool(simple_func)
        fast_tool4 = FastTool.from_function(fn4)
        assert fast_tool4.name == 'simple_func'

    def test_signature_correctness_for_fastmcp(self):
        """Test that function signatures are correctly preserved for FastMCP."""
        tool = CustomBaseTool()
        fn = from_langchain_tool(tool)
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 3
        param_names = list(sig.parameters.keys())
        assert 'count' in param_names
        assert 'min_val' in param_names
        assert 'max_val' in param_names
        for param in sig.parameters.values():
            assert param.kind != inspect.Parameter.VAR_POSITIONAL
            assert param.kind != inspect.Parameter.VAR_KEYWORD

    def test_structured_tool_priority(self):
        """Test that StructuredTool uses func attribute with priority."""

        def primary_func(x: int) -> str:
            """Primary function."""
            return f'primary_{x}'

        def fallback_func(x: int) -> str:
            """Fallback function."""
            return f'fallback_{x}'
        tool = StructuredTool.from_function(func=primary_func)
        tool._run = fallback_func
        fn = from_langchain_tool(tool)
        result = fn(5)
        assert result == 'primary_5'
        assert fn.__name__ == 'primary_func'

    def test_multiple_conversion_idempotency(self):
        """Test that converting the same tool multiple times works correctly."""
        tool = multiply_decorator_tool
        fn1 = from_langchain_tool(tool)
        fn2 = from_langchain_tool(tool)
        assert fn1.__name__ == fn2.__name__
        assert fn1.__doc__ == fn2.__doc__
        assert fn1(3, 4) == fn2(3, 4) == 12

    def test_edge_case_empty_signatures(self):
        """Test tools with empty or unusual signatures."""

        @tool
        def no_params_tool():
            """No parameters tool."""
            return 'no params'
        fn = from_langchain_tool(no_params_tool)
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 0
        assert fn() == 'no params'

        def args_only_func(*args):
            """Args only function."""
            return sum(args)
        fn2 = from_langchain_tool(args_only_func)
        result = fn2(1, 2, 3)
        assert result == 6

        def kwargs_only_func(**kwargs):
            """Kwargs only function."""
            return len(kwargs)
        fn3 = from_langchain_tool(kwargs_only_func)
        result = fn3(a=1, b=2, c=3)
        assert result == 3

