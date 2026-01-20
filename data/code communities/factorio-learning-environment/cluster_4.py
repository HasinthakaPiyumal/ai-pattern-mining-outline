# Cluster 4

def detect_direction_system(blueprint: Dict[str, Any]) -> DirectionSystem:
    """
    Detect which direction system a blueprint uses by analyzing entity directions.

    The old system uses values 0-7, while the new system uses 0-15.

    Args:
        blueprint: Blueprint dictionary

    Returns:
        DirectionSystem enum indicating which system is in use
    """
    if 'entities' not in blueprint:
        return DirectionSystem.OLD_SYSTEM
    directions_found: Set[int] = set()
    for entity in blueprint['entities']:
        if 'direction' in entity and entity['direction'] is not None:
            direction = int(entity['direction'])
            directions_found.add(direction)
    if any((d >= 8 for d in directions_found)):
        return DirectionSystem.NEW_SYSTEM
    return DirectionSystem.OLD_SYSTEM

def create_flip_augmented_dataset(base_dataset: Dataset, include_flips: List[str]=None) -> MemoryDataset:
    """
    Create a flip-augmented dataset from a base dataset.

    Args:
        base_dataset: The original dataset
        include_flips: List of flip type names to include (e.g., ["none", "horizontal", "vertical", "both"])
                      If None, includes all flip types

    Returns:
        MemoryDataset with flipped variations
    """
    if include_flips is None:
        flip_types = list(FlipType)
    else:
        flip_map = {f.value: f for f in FlipType}
        shorthand_map = {'h': FlipType.HORIZONTAL, 'v': FlipType.VERTICAL, 'hv': FlipType.BOTH, 'original': FlipType.NONE, 'h_flip': FlipType.HORIZONTAL, 'v_flip': FlipType.VERTICAL, 'hv_flip': FlipType.BOTH}
        flip_types = []
        for name in include_flips:
            name_lower = name.lower()
            if name_lower in flip_map:
                flip_types.append(flip_map[name_lower])
            elif name_lower in shorthand_map:
                flip_types.append(shorthand_map[name_lower])
    augmented_samples = []
    for original_sample in base_dataset:
        blueprint = original_sample.metadata.get('blueprint', {})
        if not blueprint:
            augmented_samples.append(original_sample)
            continue
        direction_system = detect_direction_system(blueprint)
        flipped_blueprints = generate_flipped_blueprints(blueprint, direction_system)
        for flip_type in flip_types:
            flipped_blueprint = flipped_blueprints[flip_type]
            new_metadata = update_metadata_for_flip(original_sample.metadata, flip_type, direction_system)
            new_metadata['blueprint'] = flipped_blueprint
            new_metadata['original_filename'] = original_sample.metadata.get('filename', '')
            flip_suffix = {FlipType.NONE: 'original', FlipType.HORIZONTAL: 'h_flip', FlipType.VERTICAL: 'v_flip', FlipType.BOTH: 'hv_flip'}[flip_type]
            new_sample = Sample(input=original_sample.input, target=original_sample.target, metadata=new_metadata, id=f'{original_sample.id}_{flip_suffix}' if original_sample.id else None, files=original_sample.files)
            augmented_samples.append(new_sample)
    return MemoryDataset(samples=augmented_samples)

def generate_flipped_blueprints(blueprint: Dict[str, Any], direction_system: Optional[DirectionSystem]=None) -> Dict[FlipType, Dict[str, Any]]:
    """
    Generate all 4 flipped variations of a blueprint.

    Args:
        blueprint: Original blueprint dictionary
        direction_system: Direction system to use (auto-detected if None)

    Returns:
        Dictionary mapping flip type to flipped blueprint
    """
    if direction_system is None:
        direction_system = detect_direction_system(blueprint)
    flipped_blueprints = {}
    for flip_type in FlipType:
        flipped_blueprints[flip_type] = flip_blueprint(blueprint, flip_type, direction_system)
    return flipped_blueprints

def update_metadata_for_flip(metadata: Dict[str, Any], flip_type: FlipType, direction_system: DirectionSystem) -> Dict[str, Any]:
    """
    Update metadata to reflect the flip applied.

    Args:
        metadata: Original metadata dictionary
        flip_type: Applied flip type
        direction_system: Direction system used

    Returns:
        Updated metadata dictionary
    """
    updated_metadata = copy.deepcopy(metadata)
    updated_metadata['flip_type'] = flip_type.value
    updated_metadata['flip_suffix'] = get_flip_suffix(flip_type)
    updated_metadata['direction_system'] = direction_system.value
    if 'filename' in updated_metadata:
        base_filename = updated_metadata['filename']
        if '.' in base_filename:
            name, ext = base_filename.rsplit('.', 1)
            updated_metadata['filename'] = f'{name}_{get_flip_suffix(flip_type)}.{ext}'
        else:
            updated_metadata['filename'] = f'{base_filename}_{get_flip_suffix(flip_type)}'
    return updated_metadata

def create_single_flip_dataset(base_dataset: Dataset, flip: str) -> MemoryDataset:
    """
    Create a dataset with only a single flip type applied.

    Args:
        base_dataset: The original dataset
        flip: Flip type name ("none", "horizontal", "vertical", "both")
              or shorthand ("h", "v", "hv", "original")

    Returns:
        MemoryDataset with single flip type
    """
    return create_flip_augmented_dataset(base_dataset, [flip])

def generate_subchunks(blueprint: Dict[str, Any], config: SubchunkConfig) -> List[Dict[str, Any]]:
    """
    Generate all subchunks from a blueprint using sliding window.

    Args:
        blueprint: Original blueprint
        config: Subchunk configuration

    Returns:
        List of subchunk blueprints
    """
    entities = blueprint.get('entities', [])
    if not entities:
        return []
    min_x, min_y, max_x, max_y = get_blueprint_bounds(entities)
    chunk_width, chunk_height = config.chunk_size
    step_x, step_y = config.step_size
    subchunks = []
    y = min_y
    chunk_id = 0
    while y + chunk_height <= max_y + config.padding:
        x = min_x
        while x + chunk_width <= max_x + config.padding:
            chunk_min_x = x - config.padding
            chunk_min_y = y - config.padding
            chunk_max_x = x + chunk_width + config.padding
            chunk_max_y = y + chunk_height + config.padding
            chunk = extract_subchunk(blueprint, chunk_min_x, chunk_min_y, chunk_max_x, chunk_max_y, normalize=True)
            if len(chunk['entities']) >= config.min_entities:
                chunk['metadata']['subchunk']['id'] = chunk_id
                chunk['metadata']['subchunk']['grid_position'] = {'x': int((x - min_x) / step_x), 'y': int((y - min_y) / step_y)}
                subchunks.append(chunk)
                chunk_id += 1
            x += step_x
        y += step_y
    return subchunks

def format_position_from_dict(position: Dict[str, Union[int, float]]) -> str:
    """
    Format a position dictionary as Position(x={x}, y={y}).

    Args:
        position: Dictionary with 'x' and 'y' keys

    Returns:
        Formatted position string
    """
    x = position.get('x', 0)
    y = position.get('y', 0)
    return format_position(x, y)

def format_position(x: Union[int, float], y: Union[int, float]) -> str:
    """
    Format a position as Position(x={x}, y={y}).

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Formatted position string
    """
    return f'Position(x={x}, y={y})'

def flip_direction_new_system(direction: int, flip_type: FlipType) -> int:
    """
    Flip a direction in the new 16-direction system.

    New system uses 16 directions (0-15) representing 22.5° increments.
    """
    if direction is None or flip_type == FlipType.NONE:
        return direction
    if flip_type == FlipType.HORIZONTAL:
        horizontal_flip_map = {0: 0, 1: 15, 2: 14, 3: 13, 4: 12, 5: 11, 6: 10, 7: 9, 8: 8, 9: 7, 10: 6, 11: 5, 12: 4, 13: 3, 14: 2, 15: 1}
        return horizontal_flip_map.get(direction, direction)
    elif flip_type == FlipType.VERTICAL:
        vertical_flip_map = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0, 9: 15, 10: 14, 11: 13, 12: 12, 13: 11, 14: 10, 15: 9}
        return vertical_flip_map.get(direction, direction)
    elif flip_type == FlipType.BOTH:
        vertical_flip_direction = flip_direction_new_system(direction, FlipType.VERTICAL)
        final_direction = flip_direction_new_system(vertical_flip_direction, FlipType.HORIZONTAL)
        return final_direction
    return direction

def flip_direction(direction: Optional[int], flip_type: FlipType, direction_system: DirectionSystem) -> Optional[int]:
    """
    Flip a Factorio direction value using the appropriate system.

    Args:
        direction: Original direction value
        flip_type: Type of flip to apply
        direction_system: Which direction system to use

    Returns:
        New direction value
    """
    if direction is None:
        return None
    original_type = type(direction)
    direction_int = int(direction)
    if direction_system == DirectionSystem.OLD_SYSTEM:
        new_direction = flip_direction_old_system(direction_int, flip_type)
    else:
        new_direction = flip_direction_new_system(direction_int, flip_type)
    return original_type(new_direction) if original_type is float else new_direction

def flip_direction_old_system(direction: int, flip_type: FlipType) -> int:
    """
    Flip a direction in the old 8-direction system.

    Old system directions:
    - 0: North
    - 1: Northeast
    - 2: East
    - 3: Southeast
    - 4: South
    - 5: Southwest
    - 6: West
    - 7: Northwest
    """
    if direction is None or flip_type == FlipType.NONE:
        return direction
    horizontal_flip_map = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
    vertical_flip_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}
    both_flip_map = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}
    if flip_type == FlipType.HORIZONTAL:
        return horizontal_flip_map.get(direction, direction)
    elif flip_type == FlipType.VERTICAL:
        return vertical_flip_map.get(direction, direction)
    elif flip_type == FlipType.BOTH:
        return both_flip_map.get(direction, direction)
    return direction

def normalize_blueprint_positions(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize blueprint positions so the bounding box starts near (0, 0).

    Args:
        entities: List of blueprint entities

    Returns:
        List of entities with normalized positions
    """
    if not entities:
        return entities
    min_x, min_y, max_x, max_y = get_blueprint_bounds(entities)
    offset_x = -min_x
    offset_y = -min_y
    normalized_entities = []
    for entity in entities:
        new_entity = copy.deepcopy(entity)
        pos = new_entity.get('position', {})
        new_entity['position'] = {'x': pos.get('x', 0) + offset_x, 'y': pos.get('y', 0) + offset_y}
        normalized_entities.append(new_entity)
    return normalized_entities

def get_blueprint_bounds(entities: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """
    Get the bounding box of all entities in the blueprint.

    Args:
        entities: List of blueprint entities

    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
    """
    if not entities:
        return (0, 0, 0, 0)
    positions = []
    for entity in entities:
        pos = entity.get('position', {})
        x, y = (pos.get('x', 0), pos.get('y', 0))
        positions.append((x, y))
    xs, ys = zip(*positions)
    return (min(xs), min(ys), max(xs), max(ys))

def flip_entity(entity: Dict[str, Any], flip_type: FlipType, center_x: float, center_y: float, direction_system: DirectionSystem) -> Dict[str, Any]:
    """Flip a single entity with special handling for different entity types."""
    new_entity = copy.deepcopy(entity)
    entity_name = entity.get('name', '')
    pos = entity.get('position', {})
    x = pos.get('x', 0)
    y = pos.get('y', 0)
    if flip_type == FlipType.HORIZONTAL:
        new_x = center_x - (x - center_x)
        new_y = y
    elif flip_type == FlipType.VERTICAL:
        new_x = x
        new_y = center_y - (y - center_y)
    elif flip_type == FlipType.BOTH:
        new_x = center_x - (x - center_x)
        new_y = center_y - (y - center_y)
    else:
        new_x = x
        new_y = y
    new_entity['position'] = {'x': new_x, 'y': new_y}
    if 'direction' in entity and entity['direction'] is not None:
        new_entity['direction'] = flip_direction(entity['direction'], flip_type, direction_system)
    if 'underground-belt' in entity_name:
        if should_swap_underground_belt_type(entity, flip_type, direction_system):
            pass
    return new_entity

def should_swap_underground_belt_type(entity: Dict[str, Any], flip_type: FlipType, direction_system: DirectionSystem) -> bool:
    """
    Determine if an underground belt's type should be swapped based on flip type and direction.

    Underground belts need their type swapped when:
    1. BOTH flip (always swap - 180 degree rotation)
    2. Flipping along the axis the belt pair extends along

    Args:
        entity: The underground belt entity
        flip_type: Type of flip being applied
        direction_system: Which direction system is in use

    Returns:
        True if the belt type should be swapped
    """
    if flip_type == FlipType.BOTH:
        return True
    if flip_type == FlipType.NONE:
        return False
    direction = entity.get('direction', 0)
    if direction is None:
        direction = 0
    direction = int(direction)
    if direction_system == DirectionSystem.OLD_SYSTEM:
        north_south = direction in [0, 4]
        east_west = direction in [2, 6]
    else:
        north_south = direction in [0, 8]
        east_west = direction in [4, 12]
    if flip_type == FlipType.HORIZONTAL and east_west:
        return True
    elif flip_type == FlipType.VERTICAL and north_south:
        return True
    return False

def flip_blueprint(input_blueprint: Dict[str, Any], flip_type: FlipType, direction_system: Optional[DirectionSystem]=None) -> Dict[str, Any]:
    """
    Flip a blueprint by the specified type.

    Args:
        input_blueprint: Original blueprint
        flip_type: Type of flip to apply
        direction_system: Direction system to use (auto-detected if None)

    Returns:
        Flipped blueprint
    """
    blueprint = copy.deepcopy(input_blueprint)
    if direction_system is None:
        direction_system = detect_direction_system(blueprint)
    if 'entities' in blueprint:
        for entity in blueprint['entities']:
            if 'direction' not in entity:
                entity['direction'] = 0
    if flip_type == FlipType.NONE:
        flipped_blueprint = copy.deepcopy(blueprint)
        if 'entities' in flipped_blueprint:
            flipped_blueprint['entities'] = normalize_blueprint_positions(flipped_blueprint['entities'])
        return flipped_blueprint
    flipped_blueprint = copy.deepcopy(blueprint)
    if 'entities' not in flipped_blueprint:
        return flipped_blueprint
    entities = flipped_blueprint['entities']
    min_x, min_y, max_x, max_y = get_blueprint_bounds(entities)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    flipped_entities = []
    for entity in entities:
        flipped_entity = flip_entity(entity, flip_type, center_x, center_y, direction_system)
        flipped_entities.append(flipped_entity)
    flipped_entities = normalize_blueprint_positions(flipped_entities)
    flipped_blueprint['entities'] = flipped_entities
    if 'metadata' not in flipped_blueprint:
        flipped_blueprint['metadata'] = {}
    flipped_blueprint['metadata']['flip_type'] = flip_type.value
    flipped_blueprint['metadata']['direction_system'] = direction_system.value
    if direction_system == DirectionSystem.NEW_SYSTEM:
        n_entities = []
        for entity in flipped_entities:
            if entity['direction'] == 12:
                entity['direction'] = 6
            elif entity['direction'] == 8:
                entity['direction'] = 4
            elif entity['direction'] == 4:
                entity['direction'] = 2
            else:
                entity['direction'] = 0
            n_entities.append(entity)
        flipped_blueprint['entities'] = n_entities
    return flipped_blueprint

def get_flip_suffix(flip_type: FlipType) -> str:
    """
    Get a string suffix for the flip type.

    Args:
        flip_type: FlipType enum value

    Returns:
        String suffix like "original", "h_flip", etc.
    """
    suffix_map = {FlipType.NONE: 'original', FlipType.HORIZONTAL: 'h_flip', FlipType.VERTICAL: 'v_flip', FlipType.BOTH: 'hv_flip'}
    return suffix_map[flip_type]

def extract_subchunk(blueprint: Dict[str, Any], min_x: float, min_y: float, max_x: float, max_y: float, normalize: bool=True) -> Dict[str, Any]:
    """
    Extract a subchunk from a blueprint.

    Args:
        blueprint: Original blueprint
        min_x, min_y, max_x, max_y: Chunk boundaries
        normalize: Whether to normalize positions to start near (0, 0)

    Returns:
        New blueprint containing only entities in the chunk
    """
    chunk_blueprint = copy.deepcopy(blueprint)
    original_entities = blueprint.get('entities', [])
    chunk_entities = get_entities_in_region(original_entities, min_x, min_y, max_x, max_y)
    if normalize:
        normalized_entities = []
        for entity in chunk_entities:
            new_entity = copy.deepcopy(entity)
            pos = new_entity.get('position', {})
            new_entity['position'] = {'x': pos.get('x', 0) - min_x, 'y': pos.get('y', 0) - min_y}
            normalized_entities.append(new_entity)
        chunk_entities = normalized_entities
    chunk_blueprint['entities'] = chunk_entities
    if 'metadata' not in chunk_blueprint:
        chunk_blueprint['metadata'] = {}
    chunk_blueprint['metadata']['subchunk'] = {'original_bounds': {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}, 'chunk_size': {'width': max_x - min_x, 'height': max_y - min_y}, 'entity_count': len(chunk_entities), 'normalized': normalize}
    if 'label' in chunk_blueprint:
        chunk_blueprint['label'] = f'{chunk_blueprint['label']} (chunk)'
    return chunk_blueprint

def get_entities_in_region(entities: List[Dict[str, Any]], min_x: float, min_y: float, max_x: float, max_y: float) -> List[Dict[str, Any]]:
    """
    Get all entities within a rectangular region.

    Args:
        entities: List of blueprint entities
        min_x, min_y, max_x, max_y: Region boundaries

    Returns:
        List of entities within the region
    """
    entities_in_region = []
    for entity in entities:
        pos = entity.get('position', {})
        x = pos.get('x', 0)
        y = pos.get('y', 0)
        if min_x <= x <= max_x and min_y <= y <= max_y:
            entities_in_region.append(entity)
    return entities_in_region

def create_subchunk_augmented_dataset(base_dataset: Dataset, chunk_sizes: List[Tuple[int, int]]=None, step_sizes: List[Tuple[int, int]]=None, min_entities: int=3) -> MemoryDataset:
    """
    Create a subchunk-augmented dataset from a base dataset.

    Args:
        base_dataset: The original dataset
        chunk_sizes: List of (width, height) tuples for chunk sizes
        step_sizes: List of (x_step, y_step) tuples for step sizes
        min_entities: Minimum entities required in a chunk

    Returns:
        MemoryDataset with subchunk variations
    """
    if chunk_sizes is None:
        chunk_sizes = [(10, 10), (15, 15), (20, 20)]
    if step_sizes is None:
        step_sizes = [(5, 5), (10, 10)]
    augmented_samples = []
    for original_sample in base_dataset:
        blueprint = original_sample.metadata.get('blueprint', {})
        if not blueprint:
            augmented_samples.append(original_sample)
            continue
        augmented_samples.append(original_sample)
        for chunk_size in chunk_sizes:
            for step_size in step_sizes:
                config = SubchunkConfig(chunk_size=chunk_size, step_size=step_size, min_entities=min_entities)
                subchunks = generate_subchunks(blueprint, config)
                for i, chunk_blueprint in enumerate(subchunks):
                    new_metadata = copy.deepcopy(original_sample.metadata)
                    new_metadata['blueprint'] = chunk_blueprint
                    new_metadata['original_filename'] = original_sample.metadata.get('filename', '')
                    new_metadata['augmentation_type'] = 'subchunk'
                    new_metadata['chunk_config'] = {'chunk_size': chunk_size, 'step_size': step_size, 'chunk_index': i, 'total_chunks': len(subchunks)}
                    chunk_suffix = f'chunk_{chunk_size[0]}x{chunk_size[1]}_step_{step_size[0]}x{step_size[1]}_{i}'
                    new_sample = Sample(input=original_sample.input, target=original_sample.target, metadata=new_metadata, id=f'{original_sample.id}_{chunk_suffix}' if original_sample.id else None, files=original_sample.files)
                    augmented_samples.append(new_sample)
    return MemoryDataset(samples=augmented_samples)

def create_overlapping_subchunks(blueprint: Dict[str, Any], chunk_size: Tuple[int, int], overlap: float=0.5) -> List[Dict[str, Any]]:
    """
    Create overlapping subchunks with specified overlap ratio.

    Args:
        blueprint: Original blueprint
        chunk_size: (width, height) of each chunk
        overlap: Overlap ratio (0.5 = 50% overlap)

    Returns:
        List of overlapping subchunk blueprints
    """
    chunk_width, chunk_height = chunk_size
    step_x = int(chunk_width * (1 - overlap))
    step_y = int(chunk_height * (1 - overlap))
    config = SubchunkConfig(chunk_size=chunk_size, step_size=(step_x, step_y), min_entities=3)
    return generate_subchunks(blueprint, config)

def create_adaptive_subchunks(blueprint: Dict[str, Any], target_entities_per_chunk: int=20, max_chunks: int=10) -> List[Dict[str, Any]]:
    """
    Create subchunks with adaptive sizing based on entity density.

    Args:
        blueprint: Original blueprint
        target_entities_per_chunk: Target number of entities per chunk
        max_chunks: Maximum number of chunks to generate

    Returns:
        List of adaptively-sized subchunk blueprints
    """
    entities = blueprint.get('entities', [])
    if not entities:
        return []
    total_entities = len(entities)
    ideal_chunks = min(max_chunks, max(1, total_entities // target_entities_per_chunk))
    min_x, min_y, max_x, max_y = get_blueprint_bounds(entities)
    blueprint_width = max_x - min_x
    blueprint_height = max_y - min_y
    chunks_per_side = int(math.sqrt(ideal_chunks))
    chunk_width = int(blueprint_width / chunks_per_side)
    chunk_height = int(blueprint_height / chunks_per_side)
    config = SubchunkConfig(chunk_size=(chunk_width, chunk_height), step_size=(chunk_width, chunk_height), min_entities=3)
    return generate_subchunks(blueprint, config)

@solver
def generate_entity_name_questions(questions_per_blueprint: int=3, multiple_choice: bool=False) -> Solver:
    """
    Generate questions about entity properties using a model to create diverse Q&A pairs.

    Args:
        questions_per_blueprint: Number of questions to generate per blueprint
        multiple_choice: If True, generate multiple choice questions with distractor options
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        entities = blueprint.get('entities', [])
        direction_system = detect_direction_system(blueprint)
        if not entities:
            state.metadata['error'] = 'No entities found in blueprint'
            state.metadata['basic_questions'] = []
            return state
        basic_questions = []
        all_entity_names = list(set((entity.get('name', 'unknown') for entity in entities)))
        num_questions = min(questions_per_blueprint, len(entities))
        selected_entities = random.sample(entities, num_questions)
        for entity in selected_entities:
            position = entity.get('position', {})
            entity_name = entity.get('name', 'unknown')
            x, y = (position.get('x', 0), position.get('y', 0))
            entity['entity_number'] = None
            entity_properties = {k: v for k, v in entity.items() if v is not None}
            if 'direction' in entity_properties:
                dir_value = entity_properties['direction']
                compass_dir = convert_numeric_direction(dir_value, direction_system)
                entity_properties['direction_compass'] = compass_dir
            if multiple_choice:
                prompt = f'Given this Factorio entity and its properties, generate a SPECIFIC and UNAMBIGUOUS multiple choice question about the entity.\n\nEntity Properties:\n{entity_properties}\n\nAll entity types in blueprint: {all_entity_names}\n\nIMPORTANT GUIDELINES:\n1. Questions must be answerable from just looking at the blueprint image\n2. Always use exact positions when referring to entities (e.g., "at Position(x={x}, y={y})")\n3. Create 3 plausible distractor options that could appear in a Factorio blueprint\n4. For entity name questions, use other entity types from the blueprint as distractors when possible\n5. Make distractors realistic but clearly wrong when examining the blueprint\n\nExamples of GOOD multiple choice questions:\n- "What entity is located at Position(x={x}, y={y})?\n   A) transport-belt\n   B) inserter\n   C) assembly-machine-2\n   D) {entity_name}"\n\n- "What recipe is configured in the {entity_name} at Position(x={x}, y={y})?\n   A) copper-plate\n   B) iron-gear-wheel\n   C) electronic-circuit\n   D) [correct recipe]"\n\nThe correct answer should be the option at: {random.choice(['A', 'B', 'C', 'D'])}\n\nReturn your response in this exact JSON format:\n```json\n{{\n    "question": "Your specific question here",\n    "options": {{\n        "A": "First option",\n        "B": "Second option", \n        "C": "Third option",\n        "D": "Fourth option"\n    }},\n    "correct_answer": "The letter of the correct option (A, B, C, or D)",\n    "answer_text": "The actual answer value"\n}}\n```'
            else:
                prompt = f'Given this Factorio entity and its properties, generate a SPECIFIC and UNAMBIGUOUS question and answer pair about the positioning of the entity.\n\nEntity Properties:\n{entity_properties}\n\nIMPORTANT GUIDELINES:\n1. Questions must be answerable from just looking at the blueprint image\n2. Always use exact positions when referring to entities (e.g., "at Position(x={x}, y={y})")\n3. Be specific - if there are multiple entities of the same type, specify which one\n4. Avoid vague references like "the inserter" without position\n\nExamples of GOOD questions:\n- "What entity is located at Position(x={x}, y={y})?"\n- "What recipe is configured in the {entity_name} at Position(x={x}, y={y})?"\n- "How many filters are set on the {entity_name} at Position(x={x}, y={y})?"\n- "Is there a {entity_name} at Position(x={x}, y={y})?"\n\nReturn your response in this exact JSON format:\n```json\n{{\n    "question": "Your specific question here",\n    "answer": "The precise answer"\n}}\n```'
            state.messages = [ChatMessageUser(content=prompt)]
            response = await generate(state)
            try:
                completion = response.output.completion
                json_match = re.search('```json\\s*\\n(.*?)\\n```', completion, re.DOTALL)
                if json_match:
                    qa_data = json.loads(json_match.group(1))
                    if multiple_choice:
                        question = qa_data.get('question', f'What entity is at {format_position(x, y)}?')
                        options = qa_data.get('options', {})
                        correct_answer = qa_data.get('correct_answer', 'D')
                        answer_text = qa_data.get('answer_text', entity_name)
                        if not options or len(options) != 4:
                            distractors = [name for name in all_entity_names if name != entity_name][:3]
                            if len(distractors) < 3:
                                common_entities = ['transport-belt', 'inserter', 'assembly-machine-2', 'electric-mining-drill', 'stone-furnace', 'splitter']
                                distractors.extend([e for e in common_entities if e != entity_name and e not in distractors])[:3]
                            options = {'A': distractors[0] if len(distractors) > 0 else 'transport-belt', 'B': distractors[1] if len(distractors) > 1 else 'inserter', 'C': distractors[2] if len(distractors) > 2 else 'assembly-machine-2', 'D': entity_name}
                            correct_answer = 'D'
                        formatted_question = f'{question}\n'
                        for letter, option in sorted(options.items()):
                            formatted_question += f'   {letter}) {option}\n'
                        answer = correct_answer
                        question = formatted_question.rstrip()
                    else:
                        question = qa_data.get('question', f'What entity is at {format_position(x, y)}?')
                        answer = qa_data.get('answer', entity_name)
                elif multiple_choice:
                    distractors = [name for name in all_entity_names if name != entity_name][:3]
                    question = f'What entity is located at position {format_position(x, y)}?\n'
                    question += f'   A) {(distractors[0] if distractors else 'transport-belt')}\n'
                    question += f'   B) {(distractors[1] if len(distractors) > 1 else 'inserter')}\n'
                    question += f'   C) {(distractors[2] if len(distractors) > 2 else 'assembly-machine-2')}\n'
                    question += f'   D) {entity_name}'
                    answer = 'D'
                else:
                    question = f'What entity is located at position {format_position(x, y)}?'
                    answer = entity_name
            except (JSONDecodeError, AttributeError):
                if multiple_choice:
                    distractors = [name for name in all_entity_names if name != entity_name][:3]
                    question = f'What entity is located at position {format_position(x, y)}?\n'
                    question += f'   A) {(distractors[0] if distractors else 'transport-belt')}\n'
                    question += f'   B) {(distractors[1] if len(distractors) > 1 else 'inserter')}\n'
                    question += f'   C) {(distractors[2] if len(distractors) > 2 else 'assembly-machine-2')}\n'
                    question += f'   D) {entity_name}'
                    answer = 'D'
                else:
                    question = f'What entity is located at position {format_position(x, y)}?'
                    answer = entity_name
            qa_entry = {'question': question, 'answer': answer, 'entity': entity, 'position': position, 'entity_properties': entity_properties, 'question_type': 'multiple_choice' if multiple_choice else 'open_ended'}
            if multiple_choice and 'options' in locals():
                qa_entry['options'] = options
                qa_entry['answer_text'] = answer_text if 'answer_text' in locals() else entity_name
            basic_questions.append(qa_entry)
        state.metadata['basic_questions'] = basic_questions
        return state
    return solve

@solver
def generate_position_questions(questions_per_blueprint: int=3, multiple_choice: bool=False) -> Solver:
    """
    Generate questions asking for the position of entities using model-based generation.

    Args:
        questions_per_blueprint: Number of questions to generate per blueprint
        multiple_choice: If True, generate multiple choice questions with distractor positions
    """
    create_factorio_instance()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        entities = blueprint.get('entities', [])
        direction_system = detect_direction_system(blueprint)
        if not entities:
            state.metadata['error'] = 'No entities found in blueprint'
            state.metadata['position_questions'] = []
            return state
        position_questions = []
        entities_by_name = defaultdict(list)
        for entity in entities:
            entities_by_name[entity.get('name', 'unknown')].append(entity)
        all_positions = [(e.get('position', {}).get('x', 0), e.get('position', {}).get('y', 0)) for e in entities]
        num_questions = min(questions_per_blueprint, len(entities))
        selected_entities = random.sample(entities, num_questions)
        for entity in selected_entities:
            position = entity.get('position', {})
            entity_name = entity.get('name', 'unknown')
            x, y = (position.get('x', 0), position.get('y', 0))
            same_type_count = len(entities_by_name[entity_name])
            nearby_entities = []
            for other in entities:
                if other != entity:
                    other_pos = other.get('position', {})
                    ox, oy = (other_pos.get('x', 0), other_pos.get('y', 0))
                    distance = abs(ox - x) + abs(oy - y)
                    if distance <= 5:
                        nearby_entities.append({'name': other.get('name', 'unknown'), 'position': {'x': ox, 'y': oy}, 'distance': distance})
            nearby_entities.sort(key=lambda e: e['distance'])
            if multiple_choice:
                prompt = f'Given this Factorio entity and context, generate a SPECIFIC multiple choice question asking about its position.\n\nEntity: {entity_name}\nPosition: {format_position(x, y)}\nTotal {entity_name}s in blueprint: {same_type_count}\nNearby entities (within 5 tiles): {(nearby_entities[:3] if nearby_entities else 'None')}\nAll positions in blueprint: {all_positions[:10]}  # Show sample of positions\n\nIMPORTANT GUIDELINES:\n1. Create 3 distractor positions that are plausible but incorrect\n2. Distractors should be actual positions from the blueprint or nearby positions\n3. Make the question specific enough to have only one correct answer\n4. If there are multiple entities of the same type, use specific identifiers\n\nExamples of GOOD multiple choice position questions:\n{(f'- "Where is the {entity_name} located?' if same_type_count == 1 else f'- "Where is the northernmost {entity_name} located?')}\n   A) Position(x=5, y=2)\n   B) Position(x=3, y=-1)\n   C) Position(x={x}, y={y})\n   D) Position(x=0, y=4)"\n\nReturn your response in this exact JSON format:\n```json\n{{\n    "question": "Your specific position question here",\n    "options": {{\n        "A": "Position(x=?, y=?)",\n        "B": "Position(x=?, y=?)", \n        "C": "Position(x=?, y=?)",\n        "D": "Position(x=?, y=?)"\n    }},\n    "correct_answer": "The letter of the correct option (A, B, C, or D)",\n    "answer_text": "{format_position(x, y)}"\n}}\n```'
            else:
                prompt = f'''Given this Factorio entity and context, generate a SPECIFIC question asking about its position.\n\nEntity: {entity_name}\nPosition: {format_position(x, y)}\nTotal {entity_name}s in blueprint: {same_type_count}\nNearby entities (within 5 tiles): {(nearby_entities[:3] if nearby_entities else 'None')}\n\nIMPORTANT GUIDELINES:\n1. If there's only one {entity_name}, the question can be simple\n2. If there are multiple, use specific identifiers:\n   - Relative positions (northernmost, southernmost, etc.)\n   - Distance from other entities with their exact positions\n   - Unique characteristics visible in the image\n3. Always make the question answerable from just the visual image\n\nReturn your response in this exact JSON format:\n```json\n{{\n    "question": "Your specific position question here",\n    "answer": "{format_position(x, y)}"\n}}\n```'''
            state.messages = [ChatMessageUser(content=prompt)]
            response = await generate(state)
            try:
                completion = response.output.completion
                json_match = re.search('```json\\s*\\n(.*?)\\n```', completion, re.DOTALL)
                if json_match:
                    qa_data = json.loads(json_match.group(1))
                    if multiple_choice:
                        question = qa_data.get('question', f'Where is the {entity_name} located?')
                        options = qa_data.get('options', {})
                        correct_answer = qa_data.get('correct_answer', 'C')
                        answer_text = qa_data.get('answer_text', format_position(x, y))
                        if not options or len(options) != 4:
                            distractor_positions = []
                            for ox, oy in all_positions:
                                if (ox, oy) != (x, y):
                                    distractor_positions.append(format_position(ox, oy))
                            if len(distractor_positions) < 3:
                                for i in range(3 - len(distractor_positions)):
                                    offset_x = random.randint(-5, 5)
                                    offset_y = random.randint(-5, 5)
                                    if (x + offset_x, y + offset_y) != (x, y):
                                        distractor_positions.append(format_position(x + offset_x, y + offset_y))
                            random.shuffle(distractor_positions)
                            options = {'A': distractor_positions[0], 'B': distractor_positions[1], 'C': format_position(x, y), 'D': distractor_positions[2]}
                            correct_answer = 'C'
                        formatted_question = f'{question}\n'
                        for letter, option in sorted(options.items()):
                            formatted_question += f'{letter}) {option}\n'
                        answer = correct_answer
                        question = formatted_question.rstrip()
                    else:
                        question = qa_data.get('question', f'Where is the {entity_name} located?')
                        answer = qa_data.get('answer', format_position(x, y))
                elif multiple_choice:
                    distractor_positions = []
                    for ox, oy in random.sample(all_positions, min(3, len(all_positions) - 1)):
                        if (ox, oy) != (x, y):
                            distractor_positions.append(format_position(ox, oy))
                    question = f'Where is the {entity_name} located?\n'
                    options_list = distractor_positions[:3]
                    options_list.append(format_position(x, y))
                    random.shuffle(options_list)
                    correct_idx = options_list.index(format_position(x, y))
                    letters = ['A', 'B', 'C', 'D']
                    for i, opt in enumerate(options_list):
                        question += f'   {letters[i]}) {opt}\n'
                    answer = letters[correct_idx]
                    question = question.rstrip()
                else:
                    question = f'Where is the {entity_name} located?'
                    answer = format_position(x, y)
            except (json.JSONDecodeError, AttributeError):
                if multiple_choice:
                    question = f'Where is the {entity_name} located?\n'
                    question += f'   A) Position(x={x + 1}, y={y})\n'
                    question += f'   B) Position(x={x}, y={y + 1})\n'
                    question += f'   C) Position(x={x}, y={y})\n'
                    question += f'   D) Position(x={x - 1}, y={y - 1})'
                    answer = 'C'
                else:
                    question = f'Where is the {entity_name} located?'
                    answer = format_position(x, y)
            qa_entry = {'question': question, 'answer': answer, 'entity': entity, 'position': position, 'context': {'same_type_count': same_type_count, 'nearby_entities': nearby_entities[:3]}, 'question_type': 'multiple_choice' if multiple_choice else 'open_ended'}
            if multiple_choice and 'options' in locals():
                qa_entry['options'] = options
                qa_entry['answer_text'] = answer_text if 'answer_text' in locals() else format_position(x, y)
            position_questions.append(qa_entry)
        state.metadata['position_questions'] = position_questions
        return state
    return solve

@solver
def generate_counting_questions(questions_per_blueprint: int=2, multiple_choice: bool=False) -> Solver:
    """
    Generate questions about counting entities using model-based generation.

    Args:
        questions_per_blueprint: Number of counting questions to generate per blueprint
        multiple_choice: If True, generate multiple choice questions with distractor counts
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        entities = blueprint.get('entities', [])
        direction_system = detect_direction_system(blueprint)
        if not entities:
            state.metadata['error'] = 'No entities found in blueprint'
            state.metadata['counting_questions'] = []
            return state
        entity_counts = defaultdict(int)
        entity_by_direction = defaultdict(lambda: defaultdict(int))
        entity_in_regions = defaultdict(lambda: defaultdict(int))
        connected_entities = defaultdict(int)
        for entity in entities:
            entity_name = entity.get('name', 'unknown')
            entity_counts[entity_name] += 1
            direction = entity.get('direction', 0)
            compass_dir = convert_numeric_direction(direction, direction_system)
            entity_by_direction[entity_name][compass_dir] += 1
            pos = entity.get('position', {})
            x, y = (pos.get('x', 0), pos.get('y', 0))
            region = f'{('north' if y < 0 else 'south')}-{('west' if x < 0 else 'east')}'
            entity_in_regions[entity_name][region] += 1
            if entity.get('connections'):
                connected_entities[entity_name] += 1
        counting_questions = []
        for i in range(questions_per_blueprint):
            context = {'total_entities': len(entities), 'entity_types': list(entity_counts.keys()), 'entity_counts': dict(entity_counts), 'entities_by_direction': {k: dict(v) for k, v in entity_by_direction.items()}, 'entities_by_region': {k: dict(v) for k, v in entity_in_regions.items()}, 'connected_entity_counts': dict(connected_entities)}
            if multiple_choice:
                prompt = f'Given this Factorio blueprint analysis, generate a multiple choice counting question.\n\nBlueprint Statistics:\n- Total entities: {context['total_entities']}\n- Entity types and counts: {context['entity_counts']}\n- Entities by direction: {context['entities_by_direction']}\n- Entities by region: {context['entities_by_region']}\n- Connected entities: {context['connected_entity_counts']}\n\nGenerate a creative counting question with 4 options. The distractor numbers should be plausible but wrong.\n\nExamples:\n- "How many transport-belts are in this blueprint?\n   A) 12\n   B) 15\n   C) 18\n   D) 21"\n\n- "Count the number of inserters facing north:\n   A) 2\n   B) 4\n   C) 6\n   D) 8"\n\nGUIDELINES FOR DISTRACTORS:\n1. Make them close to the correct answer (within ±50%)\n2. Avoid obvious wrong answers like 0 or 1000\n3. Include common counting mistakes (off by one, double counting, etc.)\n\nThe correct answer should be the option at: {random.choice(['A', 'B', 'C', 'D'])}\n\nReturn your response in this exact JSON format:\n```json\n{{\n    "question": "Your counting question here",\n    "options": {{\n        "A": "number",\n        "B": "number", \n        "C": "number",\n        "D": "number"\n    }},\n    "correct_answer": "The letter of the correct option (A, B, C, or D)",\n    "answer_text": "The numeric answer",\n    "explanation": "Brief explanation of what was counted"\n}}\n```'
            else:
                prompt = f"""Given this Factorio blueprint analysis, generate a counting question and its answer.\n\nBlueprint Statistics:\n- Total entities: {context['total_entities']}\n- Entity types and counts: {context['entity_counts']}\n- Entities by direction: {context['entities_by_direction']}\n- Entities by region: {context['entities_by_region']}\n- Connected entities: {context['connected_entity_counts']}\n\nGenerate a creative counting question. Examples:\n- "How many transport-belts are in this blueprint?"\n- "Count the number of inserters facing north"\n- "How many assembly machines are in the eastern half of the blueprint?"\n- "What's the total number of connected entities?"\n- "How many different types of entities are used?"\n- "Count all entities that can move items"\n\nThink step by step.\n\nReturn your response in this exact JSON format:\n```json\n{{\n    "question": "Your counting question here",\n    "answer": "The numeric answer",\n    "explanation": "Brief explanation of what was counted"\n}}\n```"""
            state.messages = [ChatMessageUser(content=prompt)]
            response = await generate(state)
            try:
                completion = response.output.completion
                json_match = re.search('```json\\s*\\n(.*?)\\n```', completion, re.DOTALL)
                if json_match:
                    qa_data = json.loads(json_match.group(1))
                    if multiple_choice:
                        question = qa_data.get('question')
                        options = qa_data.get('options', {})
                        correct_answer = qa_data.get('correct_answer')
                        answer_text = qa_data.get('answer_text')
                        explanation = qa_data.get('explanation', '')
                        if question and correct_answer and options:
                            formatted_question = f'{question}\n'
                            for letter, option in sorted(options.items()):
                                formatted_question += f'   {letter}) {option}\n'
                            counting_questions.append({'question': formatted_question.rstrip(), 'answer': correct_answer, 'answer_text': answer_text, 'options': options, 'explanation': explanation, 'context': context, 'question_type': 'multiple_choice'})
                        else:
                            entity_name = random.choice(list(entity_counts.keys()))
                            correct_count = entity_counts[entity_name]
                            distractors = []
                            distractors.append(max(1, correct_count - random.randint(1, 3)))
                            distractors.append(correct_count + random.randint(1, 3))
                            distractors.append(max(1, int(correct_count * random.uniform(0.7, 0.9))))
                            options_list = distractors + [correct_count]
                            random.shuffle(options_list)
                            options = {'A': str(options_list[0]), 'B': str(options_list[1]), 'C': str(options_list[2]), 'D': str(options_list[3])}
                            correct_idx = options_list.index(correct_count)
                            correct_answer = ['A', 'B', 'C', 'D'][correct_idx]
                            question = f'How many {entity_name}s are in this blueprint?\n'
                            for letter, count in sorted(options.items()):
                                question += f'   {letter}) {count}\n'
                            counting_questions.append({'question': question.rstrip(), 'answer': correct_answer, 'answer_text': str(correct_count), 'options': options, 'explanation': f'Count of {entity_name} entities', 'context': context, 'question_type': 'multiple_choice'})
                    else:
                        question = qa_data.get('question')
                        answer = qa_data.get('answer')
                        explanation = qa_data.get('explanation', '')
                        if question and answer:
                            counting_questions.append({'question': question, 'answer': answer, 'explanation': explanation, 'context': context, 'question_type': 'open_ended'})
                        else:
                            entity_name = random.choice(list(entity_counts.keys()))
                            counting_questions.append({'question': f'How many {entity_name}s are in this blueprint?', 'answer': str(entity_counts[entity_name]), 'explanation': f'Count of {entity_name} entities', 'context': context, 'question_type': 'open_ended'})
            except (json.JSONDecodeError, AttributeError):
                if entity_counts:
                    entity_name = random.choice(list(entity_counts.keys()))
                    if multiple_choice:
                        correct_count = entity_counts[entity_name]
                        options = {'A': str(max(1, correct_count - 2)), 'B': str(correct_count + 1), 'C': str(correct_count), 'D': str(correct_count + 3)}
                        question = f'How many {entity_name}s are in this blueprint?\n'
                        for letter, count in sorted(options.items()):
                            question += f'   {letter}) {count}\n'
                        counting_questions.append({'question': question.rstrip(), 'answer': 'C', 'answer_text': str(correct_count), 'options': options, 'explanation': f'Count of {entity_name} entities', 'context': context, 'question_type': 'multiple_choice'})
                    else:
                        counting_questions.append({'question': f'How many {entity_name}s are in this blueprint?', 'answer': str(entity_counts[entity_name]), 'explanation': f'Count of {entity_name} entities', 'context': context, 'question_type': 'open_ended'})
        state.metadata['counting_questions'] = counting_questions
        return state
    return solve

def convert_numeric_direction(direction_value: Union[int, float, str], direction_system) -> str:
    """
    Convert numeric direction to compass string.

    Args:
        direction_value: Numeric direction (0,2,4,6) or string

    Returns:
        Compass direction string (north/east/south/west)
    """
    if isinstance(direction_value, (int, float)):
        direction = Direction.from_value(int(direction_value), direction_system)
        if direction:
            return direction.to_compass_string()
    return str(direction_value)

