# Cluster 27

class TestCreateFunctionFromSchema:
    """Test cases for _create_function_from_schema helper function."""

    def test_empty_schema(self):
        """Test schema with no fields."""
        mock_run = Mock(return_value='empty result')
        fn = _create_function_from_schema(mock_run, NoArgsToolSchema, 'test_func', 'Test doc')
        assert fn.__name__ == 'test_func'
        assert fn.__doc__ == 'Test doc'
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 0
        result = fn()
        mock_run.assert_called_once_with()
        assert result == 'empty result'

    def test_schema_with_required_fields(self):
        """Test schema with required fields."""
        mock_run = Mock(return_value='multiply result')
        fn = _create_function_from_schema(mock_run, MultiplyToolInput, 'test_multiply', 'Test multiply doc')
        assert fn.__name__ == 'test_multiply'
        assert fn.__doc__ == 'Test multiply doc'
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['first_number', 'second_number']
        assert sig.parameters['first_number'].annotation is float
        assert sig.parameters['second_number'].annotation is float
        assert sig.parameters['first_number'].default == inspect.Parameter.empty
        assert sig.parameters['second_number'].default == inspect.Parameter.empty
        fn(5.0, 3.0)
        mock_run.assert_called_with(first_number=5.0, second_number=3.0)

    def test_schema_with_optional_fields(self):
        """Test schema with optional fields."""
        mock_run = Mock(return_value='greet result')
        fn = _create_function_from_schema(mock_run, GreetToolInput, 'test_greet', 'Test greet doc')
        assert fn.__name__ == 'test_greet'
        assert fn.__doc__ == 'Test greet doc'
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params == ['name', 'greeting']
        assert sig.parameters['name'].annotation is str
        assert sig.parameters['greeting'].annotation is str
        assert sig.parameters['greeting'].default == 'Hello'
        fn('Alice', 'Hi')
        mock_run.assert_called_with(name='Alice', greeting='Hi')
        mock_run.reset_mock()
        fn('Bob')
        mock_run.assert_called_with(name='Bob', greeting='Hello')

    def test_parameter_binding_edge_cases(self):
        """Test edge cases for parameter binding."""
        mock_run = Mock(return_value='bound result')
        fn = _create_function_from_schema(mock_run, GreetToolInput, 'test_func', 'Test doc')
        fn('Alice', 'Hi')
        mock_run.assert_called_with(name='Alice', greeting='Hi')
        mock_run.reset_mock()
        fn(name='Bob', greeting='Hello')
        mock_run.assert_called_with(name='Bob', greeting='Hello')
        mock_run.reset_mock()
        fn('Charlie', greeting='Hey')
        mock_run.assert_called_with(name='Charlie', greeting='Hey')
        mock_run.reset_mock()
        fn('David')
        mock_run.assert_called_with(name='David', greeting='Hello')

def _create_function_from_schema(run_method: Callable, schema: type[BaseModel], func_name: str, func_doc: str) -> Callable:
    """Create a function with proper signature from a Pydantic schema."""
    if not hasattr(schema, 'model_fields') or not schema.model_fields:

        def schema_func():
            return run_method()
        schema_func.__name__ = func_name
        schema_func.__doc__ = func_doc
        return schema_func
    fields = schema.model_fields
    required_params = []
    optional_params = []
    annotations = {}
    for field_name, field_info in fields.items():
        annotations[field_name] = field_info.annotation
        if field_info.default is not ... and field_info.default is not PydanticUndefined:
            optional_params.append(inspect.Parameter(field_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=field_info.default, annotation=field_info.annotation))
        else:
            required_params.append(inspect.Parameter(field_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=field_info.annotation))
    params = required_params + optional_params
    sig = inspect.Signature(params)

    def schema_func(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return run_method(**bound.arguments)
    schema_func.__name__ = func_name
    schema_func.__doc__ = func_doc
    schema_func.__signature__ = sig
    schema_func.__annotations__ = annotations
    return schema_func

