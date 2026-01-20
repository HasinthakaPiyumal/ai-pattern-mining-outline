# Cluster 71

def deserialize_model(serialized_json: str) -> Type[BaseModel]:
    """
    Deserialize a JSON string back into a Pydantic model class.

    Args:
        serialized_json: The JSON string containing the serialized model

    Returns:
        The reconstructed Pydantic model class
    """
    serialized = json.loads(serialized_json, object_hook=json_object_hook)
    return PydanticTypeSerializer.deserialize_model_type(serialized)

def serialize_model(model_type: Type[BaseModel]) -> str:
    """
    Serialize a model type into a JSON string for transmission via Temporal.

    Args:
        model_type: The Pydantic model class to serialize

    Returns:
        A JSON string representing the serialized model
    """
    serialized = PydanticTypeSerializer.serialize_model_type(model_type)
    return json.dumps(serialized, cls=PydanticTypeEncoder)

def test_basic_model():
    """Test serialization and deserialization of a basic model."""
    serialized = serialize_model(Location)
    LocationReconstructed = deserialize_model(serialized)
    loc = LocationReconstructed(latitude=40.7128, longitude=-74.006)
    assert loc.latitude == 40.7128
    assert loc.longitude == -74.006
    original = Location.model_json_schema()
    recon = LocationReconstructed.model_json_schema()
    assert original == recon

def test_enum_serialization():
    """Test serialization of Enum types."""
    serialized = serialize_model(Status)
    StatusReconstructed = deserialize_model(serialized)
    assert StatusReconstructed.PENDING.value == 'pending'
    assert StatusReconstructed.ACTIVE.value == 'active'
    assert StatusReconstructed.INACTIVE.value == 'inactive'

def test_complex_model():
    """Test serialization of a complex model with nested types."""
    serialized = serialize_model(ComplexModel)
    ComplexModelReconstructed = deserialize_model(serialized)
    model = ComplexModelReconstructed(id=uuid.uuid4(), name='Test', created_at=datetime.now(), tags={'Tag1', 'tag2'}, location=Location(latitude=1.0, longitude=2.0))
    assert model.tags == {'Tag1', 'tag2'}
    assert getattr(ComplexModelReconstructed.model_config, 'validate_assignment', True)
    assert getattr(ComplexModelReconstructed.model_config, 'arbitrary_types_allowed', True)

def test_generic_model():
    """Test serialization of generic models."""
    StringContainer = GenericContainer[str]
    serialized = serialize_model(StringContainer)
    ContainerReconstructed = deserialize_model(serialized)
    container = ContainerReconstructed(value='test')
    assert container.value == 'test'

def test_forward_refs():
    """Test handling of forward references."""
    serialized = serialize_model(Node)
    NodeReconstructed = deserialize_model(serialized)
    node = NodeReconstructed(value='Parent', children=[NodeReconstructed(value='Child1'), NodeReconstructed(value='Child2')])
    assert node.value == 'Parent'
    assert len(node.children) == 2
    assert node.children[0].value == 'Child1'

def test_private_attributes():
    """Test handling of private attributes."""
    serialized = serialize_model(ComplexModel)
    ModelReconstructed = deserialize_model(serialized)
    assert hasattr(ModelReconstructed, '__private_attributes__')
    instance = ModelReconstructed(id=uuid.uuid4(), name='Test', created_at=datetime.now())
    assert hasattr(instance, '_secret')

def test_recursive_model():
    """Test serialization of recursive models."""
    serialized = serialize_model(Category)
    CategoryReconstructed = deserialize_model(serialized)
    parent = CategoryReconstructed(name='Parent')
    child = CategoryReconstructed(name='Child', parent=parent)
    parent.subcategories = [child]
    assert parent.name == 'Parent'
    assert parent.subcategories[0].name == 'Child'
    assert parent.subcategories[0].parent == parent

def test_literal_type():
    """Test handling of Literal types."""

    class LiteralModel(BaseModel):
        value: Literal['A', 'B', 'C'] = 'A'
    serialized = serialize_model(LiteralModel)
    ModelReconstructed = deserialize_model(serialized)
    instance = ModelReconstructed(value='B')
    assert instance.value == 'B'
    with pytest.raises(Exception):
        ModelReconstructed(value='D')

