# Cluster 3

@solver
def normalize_position_format() -> Solver:
    """
    Solver that converts position references from (x, y) format to Position(x={x}, y={y}) format.

    This solver ensures consistent position formatting across all QA pairs.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question_fields = ['basic_questions', 'position_questions', 'counting_questions', 'spatial_questions', 'state_questions', 'inventory_questions', 'qa_pairs', 'next_action_questions', 'construction_order_questions', 'throughput_questions', 'bottleneck_questions', 'optimization_questions', 'direction_questions']
        for field in question_fields:
            if field not in state.metadata:
                continue
            questions = state.metadata[field]
            if not isinstance(questions, list):
                continue
            normalized_questions = []
            for qa in questions:
                normalized_qa = normalize_position_references_in_qa(qa)
                normalized_questions.append(normalized_qa)
            state.metadata[field] = normalized_questions
        return state
    return solve

def normalize_position_references_in_qa(qa_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize position references in a QA pair to use Position(x={x}, y={y}) format.

    Args:
        qa_data: QA pair dictionary containing 'question' and 'answer' keys

    Returns:
        Updated QA pair with normalized position format
    """
    updated_qa = qa_data.copy()
    if 'question' in updated_qa:
        updated_qa['question'] = convert_coordinate_format_in_text(updated_qa['question'])
    if 'answer' in updated_qa:
        updated_qa['answer'] = convert_coordinate_format_in_text(updated_qa['answer'])
    return updated_qa

@solver
def render_blueprint_image() -> Solver:
    """
    Solver that renders and saves the blueprint image once per task.

    This solver ensures that only one image is generated per blueprint,
    preventing duplicate images when multiple solvers run on the same blueprint.

    Should be run early in the solver chain.
    """
    instance = create_factorio_instance()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if 'image' in state.metadata:
            return state
        blueprint = state.metadata.get('blueprint', {})
        if not blueprint:
            return state
        import copy
        blueprint_copy = copy.deepcopy(blueprint)
        image: RenderedImage = instance.namespace._render(blueprint=blueprint_copy)
        from data.vqa.image_utils import save_rendered_image
        image_id = save_rendered_image(image, blueprint, state.metadata)
        state.metadata['image'] = image_id
        return state
    return solve

def create_factorio_instance() -> FactorioInstance:
    """Create a Factorio instance for taking screenshots"""
    ips, udp_ports, tcp_ports = get_local_container_ips()
    instance = FactorioInstance(address=ips[-1], tcp_port=tcp_ports[-1], fast=True, cache_scripts=True, inventory={}, all_technologies_researched=False)
    return instance

def save_rendered_image(image: RenderedImage, blueprint: Optional[Dict[str, Any]]=None, metadata: Optional[Dict[str, Any]]=None, modification_info: Optional[str]=None, base_dir: str=os.getenv('VQA_DATASET_DIR'), is_map: bool=False, is_factory: bool=False) -> str:
    """
    Save a rendered image with associated metadata and return its unique ID.

    Args:
        image: The RenderedImage object to save
        blueprint: Optional blueprint data if this is a blueprint image
        metadata: Optional metadata to save with the image
        modification_info: Optional modification info to distinguish denoising variants
        base_dir: Base directory for saving images
        is_map: If True, save in 'terrain' subdirectory
        is_factory: If True, save in 'factory' subdirectory

    Returns:
        Relative path to the image including subdirectory (e.g., "blueprints/abc123.png")
    """
    if is_factory:
        subdirectory = 'factory'
        position = metadata.get('position', {'x': 0, 'y': 0})
        base_identifier = f'factory_x{int(position['x'])}_y{int(position['y'])}'
    elif is_map:
        subdirectory = 'terrain'
        position = metadata.get('position', {'x': 0, 'y': 0})
        base_identifier = f'terrain_x{int(position['x'])}_y{int(position['y'])}'
    elif blueprint is not None:
        subdirectory = 'blueprints'
        blueprint_str = json.dumps(blueprint, sort_keys=True)
        base_identifier = hashlib.md5(blueprint_str.encode()).hexdigest()[:12]
    else:
        subdirectory = ''
        base_identifier = hashlib.md5(str(metadata).encode()).hexdigest()[:12]
    if modification_info:
        identifier = f'{base_identifier}_{modification_info}'
    else:
        identifier = base_identifier
    if subdirectory:
        save_dir = Path(base_dir) / subdirectory
    else:
        save_dir = Path(base_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    image_path = save_dir / f'{identifier}.png'
    image.save(str(image_path))
    metadata_to_save = metadata.copy() if metadata else {}
    if blueprint:
        metadata_to_save['blueprint'] = blueprint
    metadata_to_save['image_id'] = identifier
    metadata_to_save['image_type'] = subdirectory if subdirectory else 'general'
    metadata_to_save['image_filename'] = f'{identifier}.png'
    metadata_to_save['image_path'] = f'{subdirectory}/{identifier}.png' if subdirectory else f'{identifier}.png'
    metadata_path = save_dir / f'{identifier}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata_to_save, f, indent=2)
    if subdirectory:
        return f'{subdirectory}/{identifier}.png'
    else:
        return f'{identifier}.png'

@solver
def attach_bounding_box() -> Solver:
    """
    Solver that calculates and attaches the blueprint bounding box to metadata.

    This ensures the bounding box information is available for grounding positions
    in questions and answers, and gets included in the JSONL output.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        if blueprint:
            bounding_box = calculate_blueprint_bounding_box(blueprint)
            state.metadata['bounding_box'] = bounding_box
            center_x = (bounding_box['min_x'] + bounding_box['max_x']) / 2
            center_y = (bounding_box['min_y'] + bounding_box['max_y']) / 2
            state.metadata['blueprint_center'] = {'x': center_x, 'y': center_y}
        return state
    return solve

def calculate_blueprint_bounding_box(blueprint: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate the bounding box of a blueprint from its entities.

    Args:
        blueprint: Blueprint dictionary containing entities

    Returns:
        Dictionary with min_x, min_y, max_x, max_y, width, height
    """
    entities = blueprint.get('entities', [])
    if not entities:
        return {'min_x': 0.0, 'min_y': 0.0, 'max_x': 0.0, 'max_y': 0.0, 'width': 0.0, 'height': 0.0}
    x_coords = []
    y_coords = []
    for entity in entities:
        position = entity.get('position', {})
        x = position.get('x', 0)
        y = position.get('y', 0)
        x_coords.append(x)
        y_coords.append(y)
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    width = max_x - min_x
    height = max_y - min_y
    return {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y, 'width': width, 'height': height}

@solver
def generate_direction_questions(questions_per_blueprint: int=2) -> Solver:
    """
    Solver that generates questions about entity orientations using Direction enums.

    This solver analyzes blueprint entities that have directional properties
    and generates questions about their orientations using the Direction enum.

    Args:
        questions_per_blueprint: Number of direction questions to generate per blueprint
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        entities = blueprint.get('entities', [])
        direction_system = detect_direction_system(blueprint)
        directional_entities = []
        for entity in entities:
            if 'direction' in entity and entity.get('direction') is not None:
                directional_entities.append(entity)
        if not directional_entities:
            state.metadata['direction_questions'] = []
            return state
        entity_info = []
        for entity in directional_entities[:10]:
            pos = entity.get('position', {})
            direction_val = entity.get('direction', 0)
            direction_enum = Direction.from_value(direction_val, direction_system)
            entity_info.append({'name': entity.get('name', 'unknown'), 'position': f'Position(x={pos.get('x', 0)}, y={pos.get('y', 0)})', 'direction': direction_enum.name if direction_enum else f'Direction({direction_val})'})
        direction_prompt = f"""You are analyzing a Factorio blueprint and need to generate {questions_per_blueprint} questions about entity orientations.\n\nBlueprint has {len(directional_entities)} entities with directional properties:\n{json.dumps(entity_info, indent=2)}\n\nGenerate {questions_per_blueprint} questions about entity orientations. Focus on:\n\n1. **Specific entity directions**: Ask about the direction/orientation of specific entities\n2. **Relative orientations**: Compare directions between entities  \n3. **Direction patterns**: Identify orientation patterns in the layout\n4. **Functional directions**: Questions about how entity directions affect function\n\n**Important guidelines:**\n- Use Direction enum values in answers: Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST\n- Reference entities by their exact positions using Position(x=X, y=Y) format\n- Be specific about which entity you're asking about\n- Focus on orientations that are visually apparent and functionally relevant\n\nReturn your response as a JSON array of question-answer pairs:\n```json\n[\n  {{\n    "question": "What direction is the [entity] facing at Position(x=X, y=Y)?",\n    "answer": "Direction.NORTH",\n    "entity_type": "entity_name",\n    "position": {{"x": X, "y": Y}},\n    "direction_value": 0,\n    "question_type": "entity_direction"\n  }}\n]\n```"""
        state.messages = [ChatMessageUser(content=direction_prompt)]
        response = await generate(state)
        try:
            completion = response.output.completion
            json_match = re.search('```json\\s*\\n(.*?)\\n```', completion, re.DOTALL)
            if json_match:
                direction_questions = json.loads(json_match.group(1))
                validated_questions = []
                for qa in direction_questions[:questions_per_blueprint]:
                    if isinstance(qa, dict) and 'question' in qa and ('answer' in qa):
                        answer = qa['answer']
                        if not answer.startswith('Direction.'):
                            direction = Direction.from_value(answer, direction_system)
                            if direction:
                                qa['answer'] = f'Direction.{direction.name}'
                        validated_questions.append(qa)
                state.metadata['direction_questions'] = validated_questions
            else:
                state.metadata['direction_questions'] = []
        except (json.JSONDecodeError, AttributeError):
            state.metadata['direction_questions'] = []
        return state
    return solve

@solver
def generate_spatial_context_question() -> Solver:
    """
    Alternative solver that generates more complex spatial reasoning questions
    for each QA pair that was already generated.
    """
    instance = create_factorio_instance()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        qa_pairs = state.metadata.get('qa_pairs', [])
        if not qa_pairs:
            removal_solver = entity_removal_denoising()
            state = await removal_solver(state, generate)
            qa_pairs = state.metadata.get('qa_pairs', [])
        spatial_qa_pairs = []
        for qa_pair in qa_pairs:
            removed_entity = qa_pair['removed_entity']
            modified_blueprint = qa_pair['modified_blueprint']
            entities = modified_blueprint.get('entities', [])
            removed_pos = removed_entity.get('position', {})
            rx, ry = (removed_pos.get('x', 0), removed_pos.get('y', 0))
            nearby_entities = []
            for entity in entities:
                pos = entity.get('position', {})
                ex, ey = (pos.get('x', 0), pos.get('y', 0))
                distance = abs(ex - rx) + abs(ey - ry)
                if distance <= 5:
                    nearby_entities.append({'entity': entity, 'distance': distance, 'relative_x': ex - rx, 'relative_y': ey - ry})
            nearby_entities.sort(key=lambda x: x['distance'])
            context_prompt = Templates.spatial_context_question(removed_entity=removed_entity, removed_position={'x': rx, 'y': ry}, nearby_entities=[{'name': ne['entity'].get('name'), 'relative_position': f'({ne['relative_x']}, {ne['relative_y']}) from missing entity'} for ne in nearby_entities[:3]], nearest_entity_name=nearby_entities[0]['entity'].get('name') if nearby_entities else 'nearest entity')
            state.messages = [ChatMessageUser(content=context_prompt)]
            question_response = await generate(state)
            spatial_question = question_response.output.completion.strip()
            spatial_qa = qa_pair.copy()
            spatial_qa['spatial_question'] = spatial_question
            spatial_qa['nearby_entities'] = nearby_entities[:3]
            blueprint = state.metadata.get('blueprint', {})
            image: RenderedImage = instance.namespace._render(blueprint=blueprint)
            from data.vqa.image_utils import save_rendered_image
            image_id = save_rendered_image(image, blueprint, state.metadata, 'spatial_qa', '../../images')
            spatial_qa['image'] = image_id
            spatial_qa_pairs.append(spatial_qa)
        state.metadata['qa_pairs'] = spatial_qa_pairs
        state.metadata['spatial_questions_added'] = True
        return state
    return solve

@solver
def entity_removal_denoising(qa_pairs_per_blueprint: int=5) -> Solver:
    """
    Solver that:
    1. Loads a blueprint
    2. Generates multiple QA pairs by removing different entities
    3. Stores all QA pairs for the blueprint

    Args:
        qa_pairs_per_blueprint: Number of QA pairs to generate per blueprint
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        qa_pairs = []
        entities = blueprint.get('entities', [])
        if not entities:
            state.metadata['error'] = 'No entities found in blueprint'
            state.metadata['qa_pairs'] = qa_pairs
            return state
        num_pairs = min(qa_pairs_per_blueprint, len(entities))
        selected_indices = random.sample(range(len(entities)), num_pairs)
        for idx in selected_indices:
            removed_entity = entities[idx].copy()
            modified_blueprint = blueprint.copy()
            modified_blueprint['entities'] = [e for i, e in enumerate(entities) if i != idx]
            position = removed_entity.get('position', {})
            entity_name = removed_entity.get('name', 'unknown')
            question_prompt = Templates.denoising_question(position=position, entity_name=entity_name)
            state.messages = [ChatMessageUser(content=question_prompt)]
            question_response = await generate(state)
            question = question_response.output.completion.strip()
            answer = entity_name
            qa_pair = {'question': question, 'answer': answer, 'removed_entity': removed_entity, 'position': position, 'modified_blueprint': modified_blueprint}
            qa_pairs.append(qa_pair)
        state.metadata['qa_pairs'] = qa_pairs
        state.metadata['num_qa_pairs'] = len(qa_pairs)
        return state
    return solve

def create_all_flips_dataset(base_dataset: Dataset) -> MemoryDataset:
    """
    Create a dataset with all possible flips (4x augmentation).

    Args:
        base_dataset: The original dataset

    Returns:
        MemoryDataset with all flip variations
    """
    return create_flip_augmented_dataset(base_dataset, None)

def raw_blueprint_dataset() -> MemoryDataset:
    blueprint_dir = find_blueprints_dir()
    samples = []
    for blueprint_path in blueprint_dir.glob('*.json'):
        with open(blueprint_path, 'r') as f:
            blueprint_json = f.read()
        blueprint = json.loads(blueprint_json)
        sample = Sample(input=blueprint['label'] if 'label' in blueprint else blueprint_path.name, metadata={'filename': blueprint_path.name, 'blueprint': blueprint})
        samples.append(sample)
    dataset = MemoryDataset(samples=samples)
    return dataset

def find_blueprints_dir() -> Path:
    """Walk up the directory tree until we find .fle directory."""
    current = Path.cwd()
    while current != current.parent:
        fle_dir = current / '.fle'
        if fle_dir.exists() and fle_dir.is_dir():
            return fle_dir / 'blueprints'
        current = current.parent
    return Path.cwd() / '.fle' / 'blueprints'

def augmented_blueprint_dataset() -> MemoryDataset:
    """
    Create an augmented blueprint dataset with rotations.

    Args:
        rotations: List of rotation names to include (e.g., ["north", "east"])
                  If None, includes all 4 rotations

    Returns:
        MemoryDataset with rotated blueprint variations
    """
    base_dataset = raw_blueprint_dataset()
    return create_all_flips_dataset(base_dataset)

def create_combined_augmented_dataset(base_dataset: Dataset, include_flips: List[str]=None, chunk_configs: List[SubchunkConfig]=None) -> MemoryDataset:
    """
    Create a dataset with both flip and subchunk augmentations.

    Args:
        base_dataset: The original dataset
        include_flips: List of flip types to include
        chunk_configs: List of SubchunkConfig objects

    Returns:
        MemoryDataset with combined augmentations
    """
    flip_augmented = create_flip_augmented_dataset(base_dataset, include_flips)
    if chunk_configs is None:
        chunk_configs = [SubchunkConfig((10, 10), (5, 5)), SubchunkConfig((15, 15), (10, 10)), SubchunkConfig((20, 20), (10, 10))]
    augmented_samples = []
    for sample in flip_augmented:
        blueprint = sample.metadata.get('blueprint', {})
        if not blueprint:
            augmented_samples.append(sample)
            continue
        augmented_samples.append(sample)
        for config in chunk_configs:
            subchunks = generate_subchunks(blueprint, config)
            for i, chunk_blueprint in enumerate(subchunks):
                new_metadata = copy.deepcopy(sample.metadata)
                new_metadata['blueprint'] = chunk_blueprint
                new_metadata['augmentation_type'] = 'combined'
                new_metadata['subchunk_config'] = {'chunk_size': config.chunk_size, 'step_size': config.step_size, 'chunk_index': i}
                flip_part = sample.metadata.get('flip_suffix', 'original')
                chunk_part = f'chunk_{config.chunk_size[0]}x{config.chunk_size[1]}_{i}'
                new_sample = Sample(input=sample.input, target=sample.target, metadata=new_metadata, id=f'{sample.input}_{chunk_part}_{flip_part}', files=sample.files)
                print(new_sample.id)
                augmented_samples.append(new_sample)
    return MemoryDataset(samples=augmented_samples)

def augmented_blueprint_dataset_with_chunks() -> MemoryDataset:
    """
    Create an augmented blueprint dataset with both rotations and subchunks.
    """
    base_dataset = raw_blueprint_dataset()
    chunk_configs = [SubchunkConfig((20, 20), (10, 10), min_entities=10)]
    return create_combined_augmented_dataset(base_dataset, include_flips=['none', 'horizontal', 'vertical', 'both'], chunk_configs=chunk_configs)

@task
def spatial_reasoning_sandbox_task(questions_per_blueprint: int=3) -> Task:
    """
    Spatial reasoning task using sandboxed Python execution.

    The agent writes Python code to analyze blueprints and generate
    diverse spatial reasoning questions. This allows for more complex
    analysis including pattern detection, clustering, and path finding.

    Args:
        questions_per_blueprint: Number of questions to generate per blueprint
    """
    return Task(dataset=raw_blueprint_dataset(), solver=[use_tools([bash(), python()]), system_message('You are an expert at spatial analysis in Factorio blueprints.\n                You can write Python code to analyze entity positions, calculate distances,\n                identify patterns, and generate creative spatial reasoning questions.\n\n                Focus on:\n                - Distance calculations (Manhattan, Euclidean)\n                - Directional relationships (north/south/east/west)\n                - Spatial patterns (lines, grids, clusters)\n                - Relative positions and proximity\n                - Path finding and connectivity\n                - Symmetry and alignment analysis\n\n                Your code has access to the blueprint data and can use standard Python\n                libraries for calculations.'), attach_bounding_box(), generate_spatial_reasoning_with_code(questions_per_blueprint=questions_per_blueprint), generate_direction_questions(), normalize_position_format(), validate_qa_answerability()], sandbox='docker', scorer=None)

@task
def spatial_context_sandbox_task(qa_pairs_per_blueprint: int=5) -> Task:
    """
    Spatial context denoising task using sandboxed Python execution.

    The agent writes Python code to analyze spatial relationships
    around removed entities and generate context-aware questions.
    This enables sophisticated pattern analysis and spatial reasoning.

    Args:
        qa_pairs_per_blueprint: Number of QA pairs to generate
    """
    return Task(dataset=raw_blueprint_dataset(), solver=[use_tools([bash(), python()]), system_message("You are an expert at spatial context analysis in Factorio.\n                Write Python code to analyze spatial relationships around missing entities\n                and generate questions that use spatial context to identify what's missing.\n\n                Consider:\n                - Nearby entity positions and types\n                - Patterns that would be broken by the missing entity\n                - Functional relationships (e.g., inserters need adjacent targets)\n                - Symmetry and alignment in the layout\n                - Connection patterns (belts, pipes, power)\n                - Production flow and logistics\n\n                Your code should identify sophisticated spatial patterns and generate\n                questions that require understanding these relationships."), attach_bounding_box(), entity_removal_denoising(qa_pairs_per_blueprint=qa_pairs_per_blueprint), generate_spatial_context_with_code(), generate_direction_questions(), normalize_position_format()], sandbox='docker', scorer=None)

def convert_coordinate_format_in_text(text: str) -> str:
    """
    Convert coordinate references in text from (x, y) format to Position(x={x}, y={y}) format.

    Args:
        text: Text containing coordinate references

    Returns:
        Text with updated coordinate format
    """
    coordinate_pattern = '\\((-?\\d+(?:\\.\\d+)?),\\s*(-?\\d+(?:\\.\\d+)?)\\)'

    def replace_coordinate(match):
        x = match.group(1)
        y = match.group(2)
        return f'Position(x={x}, y={y})'
    return re.sub(coordinate_pattern, replace_coordinate, text)

def save_map_render(image: RenderedImage, position: Optional[Dict[str, float]]=None, radius: int=64, metadata: Optional[Dict[str, Any]]=None, base_dir: str=os.getenv('VQA_DATASET_DIR')) -> str:
    """
    Convenience function specifically for saving game map renders.

    Args:
        image: RenderedImage to save
        position: Position dict with x,y coordinates
        radius: Render radius
        metadata: Additional metadata
        base_dir: Base directory for images

    Returns:
        Image ID for use in metadata
    """
    map_metadata = metadata or {}
    map_metadata.update({'position': position, 'radius': radius, 'timestamp': datetime.now().isoformat()})
    return save_rendered_image(image=image, blueprint=None, metadata=map_metadata, modification_info=f'radius_{radius}', base_dir=base_dir, is_map=True)

@task
def terrain_task(instance, include_nearest: bool=True, include_buildable: bool=True, include_resource_buildable: bool=True, include_tile_count: bool=False, include_character_loc: bool=True, multiple_choice: bool=True) -> Task:
    """
    Terrain analysis task including nearest buildable positions.

    Args:
        include_nearest: Include nearest resource questions
        include_buildable: Include nearest buildable position questions
        include_resource_buildable: Include resource-dependent buildable questions
        include_tile_count: Include tile counting questions
        include_character_loc: Include character localization questions
        multiple_choice: If True, generate multiple choice questions
    """
    solvers = [system_message('You are analyzing Factorio terrain to answer questions about \n            resources, buildable positions, and entity placement.\n            Consider terrain features, obstacles, and resource availability.'), attach_bounding_box(), render_terrain(instance)]
    if include_nearest:
        solvers.append(nearest_questions(multiple_choice=multiple_choice))
    if include_buildable:
        solvers.append(nearest_buildable_questions(questions_per_position=5, multiple_choice=multiple_choice))
    if include_resource_buildable:
        solvers.append(nearest_buildable_with_resources_questions(questions_per_position=3, multiple_choice=multiple_choice))
    if include_tile_count:
        solvers.append(tile_count_questions(multiple_choice=multiple_choice))
    if include_character_loc:
        solvers.append(character_localisation_question(multiple_choice=multiple_choice))
    return Task(name='terrain_task' + ('_mc' if multiple_choice else ''), dataset=raw_position_dataset(pattern='concentric'), solver=solvers, scorer=None)

@task
def denoising_blueprint_task(qa_pairs_per_blueprint: int=5) -> Task:
    """
    Task that creates denoising QA pairs from blueprints.

    This task removes entities from blueprints and asks questions about what's missing.
    It's useful for training models to understand blueprint completeness and entity relationships.

    Args:
        qa_pairs_per_blueprint: Number of QA pairs to generate per blueprint (default: 5)
    """
    return Task(dataset=augmented_blueprint_dataset(), solver=[system_message('You are an expert at analyzing Factorio blueprints and identifying missing components.'), attach_bounding_box(), entity_removal_denoising(qa_pairs_per_blueprint=qa_pairs_per_blueprint), generate_direction_questions(), normalize_position_format(), validate_qa_answerability()], scorer=None)

@task
def simple_denoising_blueprint_task(qa_pairs_per_blueprint: int=5) -> Task:
    """
    Task that creates denoising QA pairs from blueprints.

    This task removes entities from blueprints and asks questions about what's missing.
    It's useful for training models to understand blueprint completeness and entity relationships.

    Args:
        qa_pairs_per_blueprint: Number of QA pairs to generate per blueprint (default: 5)
    """
    return Task(dataset=augmented_blueprint_dataset(), solver=[system_message('You are an expert at analyzing Factorio blueprints and identifying missing components.'), attach_bounding_box(), entity_removal_denoising(qa_pairs_per_blueprint=qa_pairs_per_blueprint)], scorer=None)

@task
def contrastive_alignment_task(subset: Literal['title', 'purpose']='title', limit=4, variants=1) -> Task:
    """
    For each blueprint, we run a solver to compute the following metadata for it:
    1. A descriptive label
    2. A descriptive purpose
    """
    dataset = contrastive_alignment_dataset(subset=subset, limit=limit, num_variations=variants)
    return Task(name=f'contrastive_alignment_{subset}', dataset=dataset, solver=[render_blueprint_image(), passthrough_solver()], scorer=[])

@task
def entity_name_mc_task(questions_per_blueprint: int=10) -> Task:
    """
    Entity name task with multiple choice questions.
    Convenience function that calls entity_name_task with multiple_choice=True.
    """
    return entity_name_task(questions_per_blueprint=questions_per_blueprint, multiple_choice=True)

@task
def position_mc_task(questions_per_blueprint: int=10) -> Task:
    """
    Position task with multiple choice questions.
    Convenience function that calls position_task with multiple_choice=True.
    """
    return position_task(questions_per_blueprint=questions_per_blueprint, multiple_choice=True)

@task
def counting_mc_task(questions_per_blueprint: int=10) -> Task:
    """
    Counting task with multiple choice questions.
    Convenience function that calls counting_task with multiple_choice=True.
    """
    return counting_task(questions_per_blueprint=questions_per_blueprint, multiple_choice=True)

@task
def entity_name_task(questions_per_blueprint: int=10, multiple_choice: bool=False) -> Task:
    """
    Entity name task with rotation augmentation.

    Args:
        questions_per_blueprint: Number of questions to generate per blueprint
        multiple_choice: If True, generate multiple choice questions
    """
    return Task(name='entity_name_task' + ('_mc' if multiple_choice else ''), dataset=augmented_blueprint_dataset_with_chunks(), solver=[system_message('You are analyzing Factorio blueprints to identify entities. \n                Answer questions about what entities are located at specific positions.\n                The blueprints may be rotated.'), attach_bounding_box(), render_blueprint_image(), generate_entity_name_questions(questions_per_blueprint=questions_per_blueprint, multiple_choice=multiple_choice), normalize_position_format(), validate_qa_answerability()], scorer=None)

@task
def position_task(questions_per_blueprint: int=10, multiple_choice: bool=False) -> Task:
    """
    Position task with rotation augmentation.

    Args:
        questions_per_blueprint: Number of questions to generate per blueprint
        multiple_choice: If True, generate multiple choice questions
    """
    return Task(name='position_task' + ('_mc' if multiple_choice else ''), dataset=augmented_blueprint_dataset_with_chunks(), solver=[system_message('You are analyzing Factorio blueprints to locate entities. \n                Answer questions about where specific entities are positioned.\n                The blueprints may be rotated.'), attach_bounding_box(), render_blueprint_image(), generate_position_questions(questions_per_blueprint=questions_per_blueprint, multiple_choice=multiple_choice), normalize_position_format(), validate_qa_answerability()], scorer=None)

@task
def counting_task(questions_per_blueprint: int=10, multiple_choice: bool=False) -> Task:
    """
    Counting task with rotation augmentation.

    Args:
        questions_per_blueprint: Number of questions to generate per blueprint
        multiple_choice: If True, generate multiple choice questions
    """
    return Task(name='counting_task' + ('_mc' if multiple_choice else ''), dataset=augmented_blueprint_dataset_with_chunks(), solver=[system_message('You are analyzing Factorio blueprints to count entities. \n                Answer questions about how many entities of each type are present.\n                The blueprints may be rotated.'), attach_bounding_box(), render_blueprint_image(), generate_counting_questions(questions_per_blueprint=questions_per_blueprint, multiple_choice=multiple_choice), normalize_position_format(), validate_qa_answerability()], scorer=None)

@task
def direction_task(questions_per_blueprint: int=10, multiple_choice: bool=False) -> Task:
    """
    Direction task with rotation augmentation.

    Args:
        questions_per_blueprint: Number of questions to generate per blueprint
        multiple_choice: If True, generate multiple choice questions
    """
    return Task(name='direction_task' + ('_mc' if multiple_choice else ''), dataset=augmented_blueprint_dataset_with_chunks(), solver=[system_message('You are analyzing Factorio blueprints to identify entity directions. \n                Answer questions about which direction entities are facing.\n                The blueprints may be rotated.'), attach_bounding_box(), render_blueprint_image(), generate_direction_questions(questions_per_blueprint=questions_per_blueprint), normalize_position_format(), validate_qa_answerability()], scorer=None)

@solver
def render_terrain(instance) -> Solver:

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        x, y = (state.metadata['x'], state.metadata['y'])
        step = 32
        request = f'/c game.surfaces[0].request_to_generate_chunks({{{x * step}, {y * step}}}, 16)'
        instance.rcon_client.send_command(request)
        instance.rcon_client.send_command('/c game.player.surface.force_generate_chunk_requests()')
        try:
            instance.namespace.move_to(Position(x=x * step, y=y * step))
        except Exception:
            state.metadata['instance'] = instance
            state.metadata['renderer'] = None
            return state
        nearest = None
        attempt = 0
        bag = [Resource.IronOre, Resource.Water, Resource.Stone, Resource.CrudeOil, Resource.CopperOre, Resource.Coal, Resource.Wood]
        while nearest is None and bag:
            choice = random.choice(bag)
            try:
                nearest = instance.namespace.nearest(choice)
                instance.namespace.move_to(nearest)
                print('nearest:', nearest)
            except Exception:
                attempt += 1
                bag.remove(choice)
                continue
        visible_radius = 32
        player_position = instance.namespace.player_location
        character_position = {'x': player_position.x, 'y': player_position.y}
        image, renderer = instance.namespace._render(radius=visible_radius, position=nearest, return_renderer=True, max_render_radius=32)
        if nearest:
            state.metadata['position'] = {'x': int(nearest.x), 'y': int(nearest.y)}
        else:
            state.metadata['position'] = {'x': int(x * step), 'y': int(y * step)}
        state.metadata['character_position'] = character_position
        image_id = save_rendered_image(image, metadata=state.metadata, is_map=True)
        entities = instance.namespace.get_entities(radius=visible_radius, position=nearest)
        instance.namespace.move_to(Position(x=x * step, y=y * step))
        state.metadata['image'] = image_id
        state.metadata['renderer'] = renderer
        state.metadata['entities'] = entities
        state.metadata['instance'] = instance
        return state
    return solve

@solver
def nearest_questions(multiple_choice: bool=True) -> Solver:

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        instance = state.metadata['instance']
        state.metadata['renderer']
        state.metadata['nearest_questions'] = []
        bag = [Resource.IronOre, Resource.Water, Resource.Stone, Resource.CrudeOil, Resource.CopperOre, Resource.Coal, Resource.Wood]
        nearests = []
        for b in bag:
            choice = b
            try:
                nearest = instance.namespace.nearest(choice)
                nearests.append((choice, nearest))
            except Exception:
                continue
        for choice, nearest in nearests:
            choice_name, choice_entity = choice
            if not multiple_choice:
                question = f'What is the position of the nearest {choice_name} to you?'
                answer = f'Position({str(nearest)})'
                qa_entry = {'question': question, 'answer': answer, 'entity_properties': choice_name, 'nearest': nearest, 'question_type': 'open_ended'}
                state.metadata['nearest_questions'].append(qa_entry)
            else:
                other_options = random.sample([p for _, p in nearests], 3)
                alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
                other_options.append(nearest)
                random.shuffle(other_options)
                option_string = '\n'.join([f'{alphabet[i]}) Position({option})' for i, option in enumerate(other_options)])
                question = f'What is the position of the nearest {choice_name} to you?\nProvide the correct letter and nothing else.\n{option_string}'
                answer = str(alphabet[other_options.index(nearest)])
                qa_entry = {'question': question, 'answer': answer, 'entity_properties': choice_name, 'nearest': nearest, 'options': other_options, 'question_type': 'multiple_choice'}
                state.metadata['nearest_questions'].append(qa_entry)
            pass
        return state
    return solve

@solver
def nearest_buildable_questions(questions_per_position: int=5, multiple_choice: bool=True, prototype_subset: List[Prototype]=None) -> Solver:
    """
    Generate questions about nearest buildable positions for various prototypes.

    Args:
        questions_per_position: Number of questions to generate per terrain position
        multiple_choice: If True, generate multiple choice questions
        prototype_subset: Specific prototypes to test, if None uses default list
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        instance = state.metadata.get('instance')
        renderer = state.metadata.get('renderer')
        if not instance or not renderer:
            state.metadata['error'] = 'No instance found'
            state.metadata['nearest_buildable_questions'] = []
            return state
        if not renderer:
            return state
        characters = list(filter(lambda x: x.name == 'character', renderer.entities))
        player_position = None
        if len(characters) == 1:
            player_position = characters[0].position
        else:
            return state
        prototypes_to_test = prototype_subset or BUILDABLE_PROTOTYPES
        num_questions = min(questions_per_position, len(prototypes_to_test))
        selected_prototypes = random.sample(prototypes_to_test, num_questions)
        nearest_buildable_questions = []
        for prototype in selected_prototypes:
            try:
                width = prototype.WIDTH
                height = prototype.HEIGHT
                building_box = BuildingBox(width=width, height=height)
                player_pos = player_position
                center_pos = Position(x=player_pos.x, y=player_pos.y)
                buildable_area = instance.namespace.nearest_buildable(entity=prototype, building_box=building_box, center_position=center_pos)
                nearest_pos = buildable_area.center
                if not multiple_choice:
                    question = f'What is the position of the nearest place you can build a {prototype.value[0]}?'
                    answer = f'Position(x={nearest_pos.x}, y={nearest_pos.y})'
                    qa_entry = {'question': question, 'answer': answer, 'prototype': prototype.value[0], 'building_box': {'width': width, 'height': height}, 'center_position': {'x': center_pos.x, 'y': center_pos.y}, 'buildable_area': {'center': {'x': nearest_pos.x, 'y': nearest_pos.y}, 'left_top': {'x': buildable_area.left_top.x, 'y': buildable_area.left_top.y}, 'right_bottom': {'x': buildable_area.right_bottom.x, 'y': buildable_area.right_bottom.y}}, 'question_type': 'open_ended'}
                else:
                    distractors = []
                    offsets = [(-5, -5), (5, 5), (-10, 0), (0, 10), (-7, 3), (3, -7), (8, -2), (-2, 8)]
                    for offset_x, offset_y in random.sample(offsets, 3):
                        distractor_pos = Position(x=center_pos.x + offset_x, y=center_pos.y + offset_y)
                        distractors.append(distractor_pos)
                    options = distractors + [nearest_pos]
                    random.shuffle(options)
                    alphabet = ['a', 'b', 'c', 'd']
                    option_strings = []
                    for i, pos in enumerate(options):
                        option_strings.append(f'{alphabet[i]}) Position(x={pos.x}, y={pos.y})')
                    options_text = '\n'.join(option_strings)
                    correct_index = options.index(nearest_pos)
                    correct_letter = alphabet[correct_index]
                    question = f'What is the position of the nearest place you can build a {prototype.value[0]}?\nProvide the correct letter and nothing else.\n{options_text}'
                    qa_entry = {'question': question, 'answer': correct_letter, 'prototype': prototype.value[0], 'building_box': {'width': width, 'height': height}, 'center_position': {'x': center_pos.x, 'y': center_pos.y}, 'buildable_area': {'center': {'x': nearest_pos.x, 'y': nearest_pos.y}, 'left_top': {'x': buildable_area.left_top.x, 'y': buildable_area.left_top.y}, 'right_bottom': {'x': buildable_area.right_bottom.x, 'y': buildable_area.right_bottom.y}}, 'options': [{'x': pos.x, 'y': pos.y} for pos in options], 'correct_index': correct_index, 'question_type': 'multiple_choice'}
                nearest_buildable_questions.append(qa_entry)
            except Exception as e:
                print(f'Error finding buildable position for {prototype.value[0]}: {e}')
                continue
        state.metadata['nearest_buildable_questions'] = nearest_buildable_questions
        return state
    return solve

@solver
def nearest_buildable_with_resources_questions(questions_per_position: int=3, multiple_choice: bool=True) -> Solver:
    """
    Generate questions about nearest buildable positions for resource-dependent entities
    like mining drills that need to be placed on resource patches.
    """
    RESOURCE_DEPENDENT_PROTOTYPES = [(Prototype.BurnerMiningDrill, ['iron-ore', 'copper-ore', 'coal', 'stone']), (Prototype.ElectricMiningDrill, ['iron-ore', 'copper-ore', 'coal', 'stone']), (Prototype.PumpJack, ['crude-oil']), (Prototype.OffshorePump, ['water'])]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        instance = state.metadata.get('instance')
        renderer = state.metadata.get('renderer')
        if not instance or not renderer:
            state.metadata['error'] = 'No instance found'
            state.metadata['nearest_buildable_resource_questions'] = []
            return state
        questions = []
        if not renderer:
            return state
        characters = list(filter(lambda x: x.name == 'character', renderer.entities))
        player_position = None
        if len(characters) == 1:
            player_position = characters[0].position
        else:
            return state
        num_questions = min(questions_per_position, len(RESOURCE_DEPENDENT_PROTOTYPES))
        selected_items = random.sample(RESOURCE_DEPENDENT_PROTOTYPES, num_questions)
        for prototype, valid_resources in selected_items:
            try:
                width = prototype.WIDTH
                height = prototype.HEIGHT
                building_box = BuildingBox(width=width, height=height)
                player_pos = player_position
                center_pos = Position(x=player_pos.x, y=player_pos.y)
                buildable_area = instance.namespace.nearest_buildable(entity=prototype, building_box=building_box, center_position=center_pos)
                nearest_pos = buildable_area.center
                resource_type = 'a resource patch'
                if prototype == Prototype.PumpJack:
                    resource_type = 'crude oil'
                elif prototype == Prototype.OffshorePump:
                    resource_type = 'water'
                else:
                    resource_type = 'ore'
                if not multiple_choice:
                    question = f'What is the position of the nearest {resource_type} where I can build a {prototype.value[0]}?'
                    answer = f'Position(x={nearest_pos.x}, y={nearest_pos.y})'
                    qa_entry = {'question': question, 'answer': answer, 'prototype': prototype.value[0], 'resource_type': resource_type, 'building_box': {'width': width, 'height': height}, 'buildable_position': {'x': nearest_pos.x, 'y': nearest_pos.y}, 'question_type': 'open_ended'}
                else:
                    distractors = []
                    offsets = [(-8, -8), (10, 0), (0, -10), (7, 7), (-5, 5), (5, -5)]
                    for offset_x, offset_y in random.sample(offsets, 3):
                        distractor = Position(x=center_pos.x + offset_x, y=center_pos.y + offset_y)
                        distractors.append(distractor)
                    options = distractors + [nearest_pos]
                    random.shuffle(options)
                    alphabet = ['a', 'b', 'c', 'd']
                    option_strings = [f'{alphabet[i]}) Position(x={pos.x}, y={pos.y})' for i, pos in enumerate(options)]
                    correct_index = options.index(nearest_pos)
                    question = f'What is the position of the nearest {resource_type} where you can build a {prototype.value[0]}?\nProvide the correct letter and nothing else.\n{'\n'.join(option_strings)}'
                    qa_entry = {'question': question, 'answer': alphabet[correct_index], 'prototype': prototype.value[0], 'resource_type': resource_type, 'building_box': {'width': width, 'height': height}, 'buildable_position': {'x': nearest_pos.x, 'y': nearest_pos.y}, 'options': [{'x': pos.x, 'y': pos.y} for pos in options], 'correct_index': correct_index, 'question_type': 'multiple_choice'}
                questions.append(qa_entry)
            except Exception as e:
                print(f'Error with {prototype.value[0]}: {e}')
                continue
        state.metadata['nearest_buildable_resource_questions'] = questions
        return state
    return solve

@solver
def tile_count_questions(multiple_choice: bool=True) -> Solver:

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        renderer = state.metadata['renderer']
        counts = {}
        for entity in renderer.entities:
            name = entity.name.replace('-', ' ')
            name = re.sub('\\d+', '', name)
            if 'water' in name or 'cliff' in name:
                name += '-tile'
            if name.endswith('big'):
                name = name[:-3]
                name = 'big ' + name
            name = name.strip()
            if name not in counts:
                counts[name] = 0
            counts[name] += 1
        for entity in renderer.water_tiles:
            if entity['name'] not in counts:
                counts[entity['name']] = 0
            counts[entity['name']] += 1
        multiple_choice_bands = [0, 1, 2, 4, 8, 16, 32, 64, 128]
        state.metadata['tile_count_questions'] = []
        for key, value in counts.items():
            if multiple_choice:
                band = None
                for band in multiple_choice_bands:
                    if value > band:
                        continue
                    break
                removed_multiple_choice_bands = copy.deepcopy(multiple_choice_bands)
                removed_multiple_choice_bands.remove(band)
                alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
                other_options = random.sample(removed_multiple_choice_bands, 3)
                other_options.append(band)
                random.shuffle(other_options)
                option_string = '\n'.join([f'{alphabet[i]}){option}' for i, option in enumerate(other_options)])
                question = f'How many {key}s do you see?\n{option_string}\nProvide the letter of the best match and nothing else.'
                qa_entry = {'question': question, 'answer': str(alphabet[other_options.index(band)]), 'entity_properties': key, 'count': value, 'options': other_options, 'question_type': 'multiple_choice'}
                state.metadata['tile_count_questions'].append(qa_entry)
            else:
                qa_entry = {'question': f'How many {key}s do you see?', 'answer': str(value), 'entity_properties': key, 'question_type': 'open_ended'}
                state.metadata['tile_count_questions'].append(qa_entry)
        return state
    return solve

@solver
def character_localisation_question(multiple_choice: bool=False) -> Solver:

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        renderer = state.metadata['renderer']
        if not renderer:
            return state
        characters = list(filter(lambda x: x.name == 'character', renderer.entities))
        if len(characters) == 1:
            if multiple_choice:
                if len(renderer.entities) >= 3:
                    options = [entity.position for i, entity in enumerate(random.sample(renderer.entities, k=3))]
                else:
                    options = [entity.position for entity in renderer.entities]
                if characters[0].position not in options:
                    options.append(characters[0].position)
                random.shuffle(options)
                correct_index = str(options.index(characters[0].position) + 1)
                option_string = '\n'.join([f'{i + 1}) Position({str(option)})' for i, option in enumerate(options)])
                question = f'What is the position of your character?\n{option_string}\nOnly provide the correct number.'
                qa_entry = {'question': question, 'answer': str(correct_index), 'position': characters[0].position, 'entity_properties': characters[0], 'question_type': 'multiple_choice'}
            else:
                qa_entry = {'question': 'What is the position of your character?', 'answer': f'Position(x={characters[0].position.x}, y={characters[0].position.y})', 'position': characters[0].position, 'entity_properties': characters[0], 'question_type': 'open_ended'}
            state.metadata['character_localisation_question'] = [qa_entry]
        return state
    return solve

@task
def nearest_buildable_task(instance, multiple_choice: bool=True) -> Task:
    """
    Task focused only on nearest buildable position questions.
    """
    return Task(name='nearest_buildable_task', dataset=raw_position_dataset(pattern='concentric'), solver=[system_message('You are analyzing Factorio terrain to find valid building positions.\n                Consider space requirements, terrain obstacles, and resource coverage.'), attach_bounding_box(), render_terrain(instance), nearest_buildable_questions(questions_per_position=8, multiple_choice=multiple_choice), nearest_buildable_with_resources_questions(questions_per_position=4, multiple_choice=multiple_choice)], scorer=None)

def create_factorio_instances() -> List[FactorioInstance]:

    def init_instance(params: Tuple[str, int, int]) -> FactorioInstance:
        ip, udp_port, tcp_port = params
        return FactorioInstance(address=ip, tcp_port=tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={}, all_technologies_researched=False)
    ips, udp_ports, tcp_ports = get_local_container_ips()
    with futures.ThreadPoolExecutor() as executor:
        return list(executor.map(init_instance, zip(ips, udp_ports, tcp_ports)))

def terrain_position_dataset() -> MemoryDataset:
    """
    Generate terrain positions in a concentric spiral pattern.
    This ensures we explore from the origin outward, which is more
    efficient for finding resources and buildable areas.
    """
    return raw_position_dataset(pattern='concentric', limit=None)

@task
def nearest_entity_task(instance, questions_per_position: int=5, multiple_choice: bool=True) -> Task:
    """
    Task for finding and placing entities, then asking about nearest entity positions.

    Args:
        questions_per_position: Number of questions to generate per position
        multiple_choice: If True, generate multiple choice questions
    """
    instance.reset()
    return Task(name='nearest_entity_task' + ('_mc' if multiple_choice else ''), dataset=terrain_position_dataset(), solver=[system_message('You are analyzing a Factorio factory to answer questions about \n                entity positions. Consider the placement of various entities and their \n                relative positions.'), attach_bounding_box(), render_factory(instance), nearest_entity_questions(instance, questions_per_position=questions_per_position, multiple_choice=multiple_choice), normalize_position_format()], scorer=None)

@solver
def render_factory(instance) -> Solver:
    """
    Creates a factory instance and prepares for entity placement.
    """
    inv = {e.value[0]: 3 for e in PLACEABLE_ENTITIES}
    instance.set_inventory(inv)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        x, y = (state.metadata['x'], state.metadata['y'])
        step = 32
        request = f'/c game.surfaces[0].request_to_generate_chunks({{{x * step}, {y * step}}}, 4)'
        instance.rcon_client.send_command(request)
        instance.rcon_client.send_command('/c game.player.surface.force_generate_chunk_requests()')
        instance.namespace.move_to(Position(x=x * step, y=y * step))
        state.metadata['center_position'] = {'x': x * step, 'y': y * step}
        return state
    return solve

@solver
def nearest_entity_questions(instance, questions_per_position: int=5, multiple_choice: bool=True) -> Solver:
    """
    Places random entities and generates questions about nearest entity positions.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        center_x = state.metadata['center_position']['x']
        center_y = state.metadata['center_position']['y']
        placed_entities = []
        num_entities = random.randint(5, 15)
        for i in range(num_entities):
            entity_type = random.choice(PLACEABLE_ENTITIES)
            radius = random.randint(10, 40)
            angle = random.random() * 2 * 3.14159
            offset_x = int(radius * math.cos(angle))
            offset_y = int(radius * math.sin(angle))
            target_pos = Position(x=center_x + offset_x, y=center_y + offset_y)
            try:
                building_box = BuildingBox(width=entity_type.WIDTH + 1, height=entity_type.WIDTH + 1)
                buildable_pos = instance.namespace.nearest_buildable(entity_type, building_box, center_position=target_pos)
                if buildable_pos:
                    instance.namespace.move_to(buildable_pos.center)
                    placed = instance.namespace.place_entity(entity_type, direction=random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]), position=buildable_pos.left_top)
                    instance.namespace.move_to(Position(x=center_x, y=center_y))
                    if placed:
                        placed_entities.append(placed)
            except Exception as e:
                print(f'Failed to place {entity_type}: {e}')
                continue
        player_position = instance.namespace.player_location
        character_position = {'x': player_position.x, 'y': player_position.y}
        visible_radius = 32
        image, renderer = instance.namespace._render(radius=visible_radius, position=Position(x=center_x, y=center_y), return_renderer=True, max_render_radius=visible_radius)
        state.metadata['position'] = {'x': center_x, 'y': center_y}
        state.metadata['character_position'] = character_position
        image_id = save_rendered_image(image, metadata=state.metadata, is_factory=True)
        questions = []
        entities_by_type = {}
        for entity in placed_entities:
            entity_type = entity.name
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        valid_entity_types = list(entities_by_type.keys())
        if valid_entity_types:
            for _ in range(min(questions_per_position, len(valid_entity_types))):
                query_type = random.choice(valid_entity_types)
                instances = entities_by_type[query_type]
                closest = min(instances, key=lambda e: ((e.position.x - center_x) ** 2 + (e.position.y - center_y) ** 2) ** 0.5)
                entity_display_name = query_type.replace('-', ' ')
                if not multiple_choice:
                    question = f'What is the position of the nearest {entity_display_name}?'
                    answer = f'Position(x={closest.position.x}, y={closest.position.y})'
                    qa_entry = {'question': question, 'answer': answer, 'image': image_id, 'metadata': {'query_entity_type': query_type, 'center_position': state.metadata['center_position'], 'placed_entities': placed_entities}, 'question_type': 'open_ended'}
                else:
                    distractors = []
                    for entity in placed_entities:
                        if entity != closest:
                            distractors.append(entity.position)
                    while len(distractors) < 3:
                        fake_x = center_x + random.randint(-50, 50)
                        fake_y = center_y + random.randint(-50, 50)
                        fake_pos = Position(x=fake_x, y=fake_y)
                        distractors.append(fake_pos)
                    options = [closest.position] + random.sample(distractors, 3)
                    random.shuffle(options)
                    alphabet = ['a', 'b', 'c', 'd']
                    option_strings = [f'{alphabet[i]}) Position(x={pos.x}, y={pos.y})' for i, pos in enumerate(options)]
                    correct_index = options.index(closest.position)
                    question = f'What is the position of the nearest {entity_display_name}?\nProvide the correct letter and nothing else.\n{chr(10).join(option_strings)}'
                    answer = alphabet[correct_index]
                    qa_entry = {'question': question, 'answer': answer, 'image': image_id, 'metadata': {'query_entity_type': query_type, 'center_position': state.metadata['center_position'], 'placed_entities': placed_entities, 'options': [{'x': pos.x, 'y': pos.y} for pos in options], 'correct_index': correct_index}, 'question_type': 'multiple_choice'}
                questions.append(qa_entry)
        state.metadata['image'] = image_id
        state.metadata['renderer'] = renderer
        state.metadata['entities'] = instance.namespace.get_entities(radius=visible_radius)
        state.metadata['nearest_entity_questions'] = questions
        state.metadata['placed_entities'] = placed_entities
        return state
    return solve

@task
def entity_status_task(instance, questions_per_position: int=5, multiple_choice: bool=True) -> Task:
    """
    Task for asking about entity statuses in a factory.

    Args:
        questions_per_position: Number of questions to generate per position
        multiple_choice: If True, generate multiple choice questions
    """
    instance.reset()
    return Task(name='entity_status_task' + ('_mc' if multiple_choice else ''), dataset=terrain_position_dataset(), solver=[system_message('You are analyzing a Factorio factory to answer questions about \n                entity statuses. Consider whether entities are working, have power, \n                have ingredients, or have other status conditions.'), attach_bounding_box(), render_factory(instance), nearest_entity_questions(instance, questions_per_position=3, multiple_choice=multiple_choice), entity_status_questions(instance, questions_per_position=questions_per_position, multiple_choice=multiple_choice), normalize_position_format()], scorer=None)

@solver
def entity_status_questions(instance, questions_per_position: int=5, multiple_choice: bool=True) -> Solver:
    """
    Generate questions about entity statuses for entities in the rendered scene.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        entities = state.metadata.get('entities', [])
        state.metadata.get('placed_entities', [])
        image_id = state.metadata.get('image')
        if not entities or not image_id:
            state.metadata['entity_status_questions'] = []
            return state
        questions = []
        entities_with_status = []
        for entity in entities:
            if hasattr(entity, 'status') and entity.status:
                if isinstance(entity.status, str):
                    try:
                        status = EntityStatus.from_string(entity.status)
                    except:
                        continue
                else:
                    status = entity.status
                if status not in [EntityStatus.NORMAL, EntityStatus.WORKING]:
                    entities_with_status.append(entity)
        if len(entities_with_status) < questions_per_position:
            normal_entities = [e for e in entities if hasattr(e, 'status') and e.status in [EntityStatus.NORMAL, EntityStatus.WORKING]]
            entities_with_status.extend(random.sample(normal_entities, min(len(normal_entities), questions_per_position - len(entities_with_status))))
        num_questions = min(questions_per_position, len(entities_with_status))
        selected_entities = random.sample(entities_with_status, num_questions) if entities_with_status else []
        for entity in selected_entities:
            if isinstance(entity.status, str):
                status = EntityStatus.from_string(entity.status)
            else:
                status = entity.status
            entity_display_name = entity.name.replace('-', ' ')
            position_str = f'at Position(x={entity.position.x}, y={entity.position.y})'
            if not multiple_choice:
                question = f'What is the status of the {entity_display_name} {position_str}?'
                answer = STATUS_DESCRIPTIONS.get(status, status.value)
                qa_entry = {'question': question, 'answer': answer, 'image': image_id, 'metadata': {'entity_name': entity.name, 'entity_position': {'x': entity.position.x, 'y': entity.position.y}, 'entity_status': status.value}, 'question_type': 'open_ended'}
            else:
                all_statuses = list(STATUS_DESCRIPTIONS.keys())
                other_statuses = [s for s in all_statuses if s != status]
                distractors = random.sample(other_statuses, min(3, len(other_statuses)))
                options = [status] + distractors
                random.shuffle(options)
                alphabet = ['a', 'b', 'c', 'd']
                option_strings = [f'{alphabet[i]}) {STATUS_DESCRIPTIONS.get(opt, opt.value)}' for i, opt in enumerate(options)]
                correct_index = options.index(status)
                question = f'What is the status of the {entity_display_name} {position_str}?\nProvide the correct letter and nothing else.\n{chr(10).join(option_strings)}'
                answer = alphabet[correct_index]
                qa_entry = {'question': question, 'answer': answer, 'image': image_id, 'metadata': {'entity_name': entity.name, 'entity_position': {'x': entity.position.x, 'y': entity.position.y}, 'entity_status': status.value, 'options': [s.value for s in options], 'correct_index': correct_index}, 'question_type': 'multiple_choice'}
            questions.append(qa_entry)
        state.metadata['entity_status_questions'] = questions
        return state
    return solve

@task
def factory_task(instance, include_nearest_entity: bool=True, include_entity_status: bool=True, multiple_choice: bool=True) -> Task:
    """
    Comprehensive factory analysis task.

    Args:
        include_nearest_entity: Include nearest entity questions
        include_entity_status: Include entity status questions
        multiple_choice: If True, generate multiple choice questions
    """
    instance.reset()
    solvers = [system_message('You are analyzing a Factorio factory to answer questions about \n            entity positions, statuses, production chains, and factory layout.'), attach_bounding_box(), render_factory(instance)]
    if include_nearest_entity:
        solvers.append(nearest_entity_questions(instance, questions_per_position=5, multiple_choice=multiple_choice))
    if include_entity_status:
        solvers.append(entity_status_questions(instance, questions_per_position=5, multiple_choice=multiple_choice))
    solvers.append(normalize_position_format())
    return Task(name='factory_task' + ('_mc' if multiple_choice else ''), dataset=terrain_position_dataset(), solver=solvers, scorer=None)

@solver
def entity_removal_denoising(qa_pairs_per_blueprint: int=5) -> Solver:
    """
    Solver that:
    1. Loads a blueprint
    2. Generates multiple QA pairs by removing different entities
    3. Stores all QA pairs for the blueprint

    Args:
        qa_pairs_per_blueprint: Number of QA pairs to generate per blueprint
    """
    instance = create_factorio_instance()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        qa_pairs = []
        entities = blueprint.get('entities', [])
        if not entities:
            state.metadata['error'] = 'No entities found in blueprint'
            state.metadata['qa_pairs'] = qa_pairs
            return state
        num_pairs = min(qa_pairs_per_blueprint, len(entities))
        selected_indices = random.sample(range(len(entities)), num_pairs)
        for idx in selected_indices:
            removed_entity = entities[idx].copy()
            modified_blueprint = blueprint.copy()
            modified_blueprint['entities'] = [e for i, e in enumerate(entities) if i != idx]
            position = removed_entity.get('position', {})
            entity_name = removed_entity.get('name', 'unknown')
            question_prompt = Templates.denoising_question(position=position, entity_name=entity_name)
            state.messages = [ChatMessageUser(content=question_prompt)]
            question_response = await generate(state)
            question = question_response.output.completion.strip('"')
            if not question:
                continue
            image: RenderedImage = instance.namespace._render(blueprint=modified_blueprint)
            from data.vqa.image_utils import save_rendered_image
            modification_info = f'denoising_removed_{removed_entity.get('name', 'unknown')}_{idx}'
            image_id = save_rendered_image(image, modified_blueprint, state.metadata, modification_info)
            id = image_id
            answer = entity_name
            qa_pair = {'question': question, 'answer': answer, 'removed_entity': removed_entity, 'position': position, 'modified_blueprint': modified_blueprint, 'image': id}
            qa_pairs.append(qa_pair)
        state.metadata['qa_pairs'] = qa_pairs
        state.metadata['num_qa_pairs'] = len(qa_pairs)
        return state
    return solve

@solver
def validate_qa_answerability() -> Solver:
    """
    Followup solver that validates if generated questions are answerable and unambiguous.

    This solver checks each generated Q&A pair to ensure:
    1. The question is clear and specific
    2. The answer directly addresses the question
    3. There's enough context to answer the question
    4. The question avoids ambiguity

    It will regenerate questions that fail validation.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question_fields = ['basic_questions', 'position_questions', 'counting_questions', 'spatial_questions', 'state_questions', 'inventory_questions', 'qa_pairs', 'next_action_questions', 'construction_order_questions', 'throughput_questions', 'bottleneck_questions', 'optimization_questions', 'direction_questions']
        for field in question_fields:
            if field not in state.metadata:
                continue
            questions = state.metadata[field]
            if not isinstance(questions, list):
                continue
            validated_questions = []
            for qa in questions:
                question = qa.get('question', '')
                answer = qa.get('answer', '')
                if not question or not answer:
                    continue
                validation_prompt = f"""You are validating a Visual Question Answering (VQA) pair for a Factorio blueprint analysis task.\n                \nQuestion: \n```\n{question}\n```\nAnswer: `{answer}`\n\nPlease evaluate if this Q&A pair meets the following criteria:\n\n1. **Specificity**: Is the question specific enough that it has a single, unambiguous answer?\n2. **Visual Answerability**: Can the question be answered by looking at a blueprint image?\n3. **Clarity**: Is the question clearly worded without confusing terminology?\n4. **Answer Match**: Does the provided answer directly and completely answer the question?\n5. **Triviality/Tautology**: Is there actual informational content in the question? Or is it self-referential?\n\nCommon issues to check for:\n- Vague positional references (e.g., "the inserter" when there are multiple)\n- Unclear directional terms (using numbers instead of compass directions)\n- Ambiguous entity references without specific positions\n- Questions that require game knowledge beyond what's visible\n\nIf the Q&A pair has issues, provide a revised version that fixes them.\n\nIf the question includes multiple choice - it is critical that you keep them!\n\nReturn your response in this exact JSON format:\n```json\n{{\n    "is_valid": true/false,\n    "issues": ["list of specific issues if any"],\n    "revised_question": "improved question if needed",\n    "revised_answer": "improved answer if needed",\n    "explanation": "brief explanation of changes"\n}}\n```"""
                state.messages = [ChatMessageUser(content=validation_prompt)]
                response = await generate(state)
                try:
                    completion = response.output.completion
                    json_match = re.search('```json\\s*\\n(.*?)\\n```', completion, re.DOTALL)
                    if json_match:
                        validation_result = json.loads(json_match.group(1))
                        if validation_result.get('is_valid', False):
                            validated_questions.append(qa)
                        else:
                            revised_qa = qa.copy()
                            revised_qa['question'] = validation_result.get('revised_question', question)
                            revised_qa['answer'] = validation_result.get('revised_answer', answer)
                            revised_qa['validation_notes'] = {'original_question': question, 'original_answer': answer, 'issues': validation_result.get('issues', []), 'explanation': validation_result.get('explanation', '')}
                            validated_questions.append(revised_qa)
                    else:
                        validated_questions.append(qa)
                except (json.JSONDecodeError, AttributeError):
                    qa['validation_failed'] = True
                    validated_questions.append(qa)
            state.metadata[field] = validated_questions
        return state
    return solve

@task
def denoising_validation_task(qa_pairs_per_blueprint: int=5) -> Task:
    """
    Task that validates denoising QA pairs by testing if a model can answer them correctly.

    This task first generates denoising QA pairs, then validates them by having
    a model attempt to answer the questions.

    Args:
        qa_pairs_per_blueprint: Number of QA pairs to generate per blueprint (default: 5)
    """
    return Task(dataset=augmented_blueprint_dataset(), solver=[system_message('You are an expert at analyzing Factorio blueprints and identifying missing components.'), attach_bounding_box(), entity_removal_denoising(qa_pairs_per_blueprint=qa_pairs_per_blueprint), validate_denoising_qa(), generate_direction_questions(), normalize_position_format(), validate_qa_answerability()], scorer=None)

@solver
def validate_denoising_qa() -> Solver:
    """
    Solver that validates if another model can answer the denoising questions correctly.
    This should be run after entity_removal_denoising.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        qa_pairs = state.metadata.get('qa_pairs', [])
        if not qa_pairs:
            state.metadata['error'] = 'No QA pairs found'
            return state
        validated_pairs = []
        for qa_pair in qa_pairs:
            validation_prompt = Templates.denoising_validation(modified_blueprint=qa_pair['modified_blueprint'], question=qa_pair['question'])
            state.messages = [ChatMessageUser(content=validation_prompt)]
            validation_response = await generate(state)
            predicted_answer = validation_response.output.completion.strip().lower()
            correct_answer = qa_pair['answer'].lower()
            is_correct = correct_answer in predicted_answer or predicted_answer in correct_answer
            validated_qa = qa_pair.copy()
            validated_qa['validation_result'] = {'predicted': predicted_answer, 'correct': correct_answer, 'is_correct': is_correct}
            validated_pairs.append(validated_qa)
        state.metadata['qa_pairs'] = validated_pairs
        state.metadata['validation_complete'] = True
        return state
    return solve

@task
def direction_mc_task(questions_per_blueprint: int=10) -> Task:
    """
    Direction task with multiple choice questions.
    Currently returns regular direction task - update when MC support is added.
    """
    return direction_task(questions_per_blueprint=questions_per_blueprint)

@task
def action_sequence_generation_task(max_actions: int=10) -> Task:
    """
    Generate construction action sequences from blueprints.

    This task converts blueprints into imperative construction steps,
    creating a sequence of "place X at (y, z)" actions.

    Args:
        max_actions: Maximum number of construction actions to generate per blueprint
    """
    return Task(dataset=raw_blueprint_dataset(), solver=[system_message('You are planning the construction sequence for a Factorio blueprint. \n                Convert the blueprint into a logical series of construction steps.'), generate_action_sequence(max_actions=max_actions)], scorer=None)

@task
def next_action_prediction_task(num_questions: int=3) -> Task:
    """
    Action prediction VQA task: Predict the next action in a construction sequence.

    This task shows N-1 construction actions and asks the model to predict
    the Nth action. It tests understanding of construction logic and blueprints.

    Args:
        num_questions: Number of next-action prediction questions per blueprint
    """
    return Task(dataset=raw_blueprint_dataset(), solver=[system_message('You are an expert at Factorio construction planning. \n                Given a sequence of construction actions, predict what the next logical \n                action should be based on the blueprint and construction principles.'), attach_bounding_box(), generate_action_sequence(max_actions=10), generate_next_action_questions(num_questions=num_questions), generate_direction_questions(), normalize_position_format(), validate_qa_answerability()], scorer=None)

@task
def construction_order_task(num_questions: int=2) -> Task:
    """
    Construction order VQA task: Determine optimal build order for entities.

    This task asks about the optimal order to construct multiple entities,
    considering dependencies and efficiency.

    Args:
        num_questions: Number of construction order questions per blueprint
    """
    return Task(dataset=raw_blueprint_dataset(), solver=[system_message('You are an expert at Factorio construction planning. \n                Determine the optimal order to build entities considering power requirements, \n                dependencies, and construction efficiency.'), attach_bounding_box(), generate_construction_order_questions(num_questions=num_questions), generate_direction_questions(), normalize_position_format(), validate_qa_answerability()], scorer=None)

@task
def comprehensive_action_task(max_actions: int=8, next_action_questions: int=2, order_questions: int=1) -> Task:
    """
    Comprehensive action prediction task combining sequence generation and prediction.

    Args:
        max_actions: Maximum construction actions to generate
        next_action_questions: Number of next-action prediction questions
        order_questions: Number of construction order questions
    """
    return Task(dataset=raw_blueprint_dataset(), solver=[system_message('You are an expert at Factorio construction and automation. \n                Plan construction sequences, predict next actions, and determine optimal \n                build orders for efficient factory construction.'), attach_bounding_box(), generate_action_sequence(max_actions=max_actions), generate_next_action_questions(num_questions=next_action_questions), generate_construction_order_questions(num_questions=order_questions), generate_direction_questions(), normalize_position_format(), validate_qa_answerability()], scorer=None)

@solver
def entity_removal_denoising(qa_pairs_per_blueprint: int=5) -> Solver:
    """
    Solver that:
    1. Loads a blueprint
    2. Generates multiple QA pairs by removing different entities
    3. Stores all QA pairs for the blueprint

    Args:
        qa_pairs_per_blueprint: Number of QA pairs to generate per blueprint
    """
    instance = create_factorio_instance()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        qa_pairs = []
        entities = blueprint.get('entities', [])
        if not entities:
            state.metadata['error'] = 'No entities found in blueprint'
            state.metadata['qa_pairs'] = qa_pairs
            return state
        num_pairs = min(qa_pairs_per_blueprint, len(entities))
        selected_indices = random.sample(range(len(entities)), num_pairs)
        for idx in selected_indices:
            removed_entity = entities[idx].copy()
            modified_blueprint = blueprint.copy()
            modified_blueprint['entities'] = [e for i, e in enumerate(entities) if i != idx]
            position = removed_entity.get('position', {})
            entity_name = removed_entity.get('name', 'unknown')
            question = f'Name the missing entity at: Position(x={position['x']}, y={position['y']})'
            image: RenderedImage = instance.namespace._render(blueprint=modified_blueprint)
            from data.vqa.image_utils import save_rendered_image
            modification_info = f'denoising_removed_{removed_entity.get('name', 'unknown')}_{idx}'
            image_id = save_rendered_image(image, modified_blueprint, state.metadata, modification_info)
            id = image_id
            answer = entity_name
            qa_pair = {'question': question, 'answer': answer, 'removed_entity': removed_entity, 'position': position, 'modified_blueprint': modified_blueprint, 'image': id}
            qa_pairs.append(qa_pair)
        state.metadata['qa_pairs'] = qa_pairs
        state.metadata['num_qa_pairs'] = len(qa_pairs)
        return state
    return solve

@solver
def generate_spatial_reasoning_with_code(questions_per_blueprint: int=3) -> Solver:
    """
    Generate spatial reasoning questions using Python code written by the agent.
    """
    instance = create_factorio_instance()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        entities = blueprint.get('entities', [])
        image: RenderedImage = instance.namespace._render(blueprint=blueprint)
        from data.vqa.image_utils import save_rendered_image
        image_id = save_rendered_image(image, blueprint, state.metadata, 'spatial_reasoning')
        state.metadata['image'] = image_id
        if len(entities) < 2:
            state.metadata['error'] = 'Not enough entities for spatial reasoning'
            state.metadata['spatial_questions'] = []
            return state
        blueprint_data = f'blueprint = {json.dumps(blueprint, indent=2)}'
        await sandbox().write_file('/tmp/blueprint_data.py', blueprint_data)
        prompt = f"""I need you to analyze a Factorio blueprint and generate {questions_per_blueprint} spatial reasoning QA pairs.\n\nThe blueprint has {len(entities)} entities. I've saved the blueprint data to `/tmp/blueprint_data.py`.\n\nWrite Python code that:\n1. Imports the blueprint data using: `from blueprint_data import blueprint`\n2. Analyzes spatial relationships between entities\n3. Generates diverse spatial reasoning questions\n4. Prints the qa_pairs as JSON\n\nYour code should generate questions about:\n- Distances between entities (Manhattan, Euclidean)\n- Relative directions (north/south/east/west)\n- Spatial patterns (lines, grids, clusters)\n- Nearest/farthest entities\n- Entities within a certain radius\n\nThe output should be a JSON list of QA pairs, each with:\n- 'question': The spatial reasoning question\n- 'answer': The correct answer\n- 'metadata': Additional context about the spatial relationship\n\nExample code structure:\n```python\nimport json\nimport random\nimport math\nfrom blueprint_data import blueprint\n\nentities = blueprint.get('entities', [])\nqa_pairs = []\n\n# Generate distance questions\nfor _ in range(2):\n    if len(entities) >= 2:\n        e1, e2 = random.sample(entities, 2)\n        x1, y1 = e1['position']['x'], e1['position']['y']\n        x2, y2 = e2['position']['x'], e2['position']['y']\n\n        manhattan = abs(x2 - x1) + abs(y2 - y1)\n\n        qa_pairs.append({{\n        'question': f"What is the Manhattan distance between the {{e1['name']}} at ({{x1}}, {{y1}}) and the {{e2['name']}} at ({{x2}}, {{y2}})?",\n            'answer': str(manhattan),\n            'metadata': {{\n        'type': 'distance',\n                'entities': [e1['name'], e2['name']],\n                'positions': [(x1, y1), (x2, y2)]\n            }}\n        }})\n\n# Add more question types...\n\nprint(json.dumps(qa_pairs, indent=2))\n```\n\nUse the analyze_blueprint tool to execute your code."""
        state.messages = [ChatMessageUser(content=prompt)]
        state = await generate(state)
        qa_pairs = []
        for tool_call in reversed(state.messages):
            if isinstance(tool_call, ChatMessageTool):
                try:
                    qa_pairs = json.loads(tool_call.content)
                    break
                except json.JSONDecodeError:
                    continue
        state.metadata['spatial_questions'] = qa_pairs
        state.metadata['generation_method'] = 'sandbox_code'
        return state
    return solve

@solver
def generate_spatial_context_with_code() -> Solver:
    """
    Generate spatial context questions for denoising scenarios using sandbox Python execution.
    """
    instance = create_factorio_instance()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        qa_pairs = state.metadata.get('qa_pairs', [])
        if not qa_pairs:
            state.metadata['error'] = 'No denoising QA pairs found'
            return state
        core_qa_pairs = []
        images = []
        for pair in qa_pairs:
            core_pair = {}
            for key, value in pair.items():
                if key != 'image':
                    core_pair[key] = value
                else:
                    images.append(value)
            core_qa_pairs.append(core_pair)
        qa_data = f'qa_pairs = {json.dumps(core_qa_pairs, indent=2)}'
        await sandbox().write_file('/tmp/qa_pairs_data.py', qa_data)
        prompt = f"""I need you to enhance {len(qa_pairs)} denoising QA pairs with spatial context analysis.\n\nThe QA pairs data has been saved to `/tmp/qa_pairs_data.py`. Each pair contains:\n- 'removed_entity': The entity that was removed\n- 'modified_blueprint': The blueprint after removal\n- 'position': Where the entity was removed\n\nWrite Python code that:\n1. Imports the data: `from qa_pairs_data import qa_pairs`\n2. For each QA pair, analyzes nearby entities in the modified blueprint\n3. Generates spatial context questions about what's missing\n4. Creates enhanced QA pairs with spatial reasoning\n\nGenerate questions like:\n- "What entity is missing 2 tiles north of the [entity_name] at position ([x], [y])?"\n- "An entity was removed between two [entity_type]. What was it?"\n- "What's missing from the center of the 3x3 grid?"\n\nOutput format should be a JSON list of enhanced QA pairs with:\n- All original fields\n- 'spatial_question': A context-aware question\n- 'nearby_entities': List of nearby entities with distances and directions\n\nOnly print the final output to stdout, and nothing else.\n\nExample approach:\n```python\nimport json\nfrom qa_pairs_data import qa_pairs\n\ndef get_direction(from_pos, to_pos):\n    dx = to_pos['x'] - from_pos['x']\n    dy = to_pos['y'] - from_pos['y']\n\n    if abs(dx) > abs(dy):\n        return 'east' if dx > 0 else 'west'\n    else:\n        return 'south' if dy > 0 else 'north'\n\nenhanced_pairs = []\n\nfor qa in qa_pairs:\n    removed_pos = qa['position']\n    entities = qa['modified_blueprint']['entities']\n\n    # Find nearby entities\n    nearby = []\n    for entity in entities:\n        pos = entity['position']\n        dist = abs(pos['x'] - removed_pos['x']) + abs(pos['y'] - removed_pos['y'])\n        if dist <= 5:\n            nearby.append({{\n                'name': entity['name'],\n                'distance': dist,\n                'direction': get_direction(removed_pos, pos)\n            }})\n\n    nearby.sort(key=lambda x: x['distance'])\n\n    # Create spatial question\n    if nearby:\n        nearest = nearby[0]\n        spatial_q = f"What entity is missing {{nearest['distance']}} tiles {{nearest['direction']}} of the {{nearest['name']}}?"\n    else:\n        spatial_q = f"What entity was at position ({{removed_pos['x']}}, {{removed_pos['y']}})?"\n\n    enhanced = qa.copy()\n    enhanced['spatial_question'] = spatial_q\n    enhanced['nearby_entities'] = nearby[:3]\n    enhanced_pairs.append(enhanced)\n\nprint(json.dumps(enhanced_pairs, indent=2))\n```"""
        state.messages = [ChatMessageUser(content=prompt)]
        state = await generate(state)
        for tool_call in reversed(state.messages):
            if isinstance(tool_call, ChatMessageTool):
                try:
                    enhanced_pairs = json.loads(tool_call.content)
                    state.metadata['qa_pairs'] = enhanced_pairs
                    state.metadata['spatial_context_added'] = True
                    break
                except json.JSONDecodeError:
                    continue
        blueprint = state.metadata.get('blueprint', {})
        image: RenderedImage = instance.namespace._render(blueprint=blueprint)
        from data.vqa.image_utils import save_rendered_image
        image_id = save_rendered_image(image, blueprint, state.metadata, 'spatial_context')
        state.metadata['image'] = image_id
        return state
    return solve

def raw_blueprint_dataset() -> MemoryDataset:
    blueprint_dir = find_blueprints_dir()
    samples = []
    for blueprint_path in blueprint_dir.glob('*.json'):
        with open(blueprint_path, 'r') as f:
            blueprint_json = f.read()
        blueprint = json.loads(blueprint_json)
        sample = Sample(input=blueprint['label'] if 'label' in blueprint else blueprint_path.name, metadata={'filename': blueprint_path.name, 'blueprint': blueprint})
        samples.append(sample)
    dataset = MemoryDataset(samples=samples)
    return dataset

@task
def contrastive_blueprint_labelling_task(num_variations: int=3) -> Task:
    """
    For each blueprint, we run a solver to compute multiple variations of metadata:
    1. Descriptive labels
    2. Descriptive purposes

    Args:
        num_variations: Number of title/purpose variations to generate per blueprint
    """
    return Task(dataset=augmented_blueprint_dataset(), solver=[system_message('You are an expert Factorio player analyzing blueprints. \n                Generate clear, concise titles and purpose descriptions that would help \n                other players understand what each blueprint does.'), attach_bounding_box(), generate_blueprint_title_and_purpose(num_variations=num_variations)], scorer=[includes()])

@solver
def generate_blueprint_title_and_purpose(num_variations: int=3) -> Solver:
    """Generate multiple title and purpose descriptions for blueprints in a single LLM call.

    Args:
        num_variations: Number of different title/purpose pairs to generate (default: 3)
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        blueprint_copy = blueprint.copy()
        if 'label' in blueprint_copy:
            del blueprint_copy['label']
        prompt = f"""Analyze this Factorio blueprint and generate {num_variations} different metadata variations.\n\nBlueprint:\n{json.dumps(blueprint_copy, indent=2)}\n\nGenerate {num_variations} different variations, each with:\n1. A concise title (max 10 words) that describes what this blueprint builds\n2. A purpose description (1-2 sentences) explaining what it does and how it's used\n\nImportant guidelines:\n- Each variation should emphasize different aspects of the blueprint\n- Variation 1: Focus on the primary function and most obvious use case\n- Variation 2: Highlight efficiency, automation, or technical aspects\n- Variation 3: Emphasize scalability, integration, or advanced features\n{('- Additional variations: Consider alternative use cases, specialized applications, or unique benefits' if num_variations > 3 else '')}\n\nMake each title and purpose distinct while still being accurate.\n\nFormat your response as JSON:\n```json\n{{\n    "variations": [\n        {{\n            "title": "...",\n            "purpose": "..."\n        }},\n        {{\n            "title": "...",\n            "purpose": "..."\n        }},\n        ...\n    ]\n}}\n```"""
        state.messages[-1] = ChatMessageUser(content=prompt)
        response = await generate(state)
        completion = response.output.completion
        pattern = '```json\\s*\\n(.*?)\\n```'
        match = re.search(pattern, completion, re.DOTALL)
        all_titles = []
        all_purposes = []
        if match:
            json_content = match.group(1)
            try:
                data = json.loads(json_content)
                variations = data.get('variations', [])
                for variation in variations:
                    all_titles.append(variation.get('title', ''))
                    all_purposes.append(variation.get('purpose', ''))
            except json.JSONDecodeError as e:
                print(f'Error parsing JSON: {e}')
                pass
        state.metadata['titles'] = all_titles
        state.metadata['purposes'] = all_purposes
        if all_titles:
            state.metadata['title'] = all_titles[0]
            state.metadata['purpose'] = all_purposes[0]
        return state
    return solve

@solver
def passthrough_solver():

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.input
        answer = state.target.text
        alphabet = ['A', 'B', 'C', 'D', 'E']
        choices = [choice.value for choice in state.choices._choices]
        random.shuffle(choices)
        options = '\n'.join([f'{alphabet[i]}) {choice}' for i, choice in enumerate(choices)])
        question = question + '\n' + options
        answer_index = choices.index(answer)
        target = alphabet[answer_index]
        state.metadata['contrastive_alignment'] = [{'answer': target, 'question': question}]
        return state
    return solve

def contrastive_alignment_dataset(*args, subset: Literal['title', 'purpose'], limit=10, num_variations=3, model='anthropic/claude-opus-4-20250514') -> MemoryDataset:
    """
    Task that creates contrastive image-text alignment questions with multiple variations per blueprint.
    Given a blueprint image, the model must select the correct title/purpose from multiple options.

    Args:
        subset: Whether to use 'title' or 'purpose' for questions
        limit: Number of blueprints to process
        num_variations: Number of title/purpose variations to generate per blueprint
        model: Model to use for generation
    """
    instance = create_factorio_instance()
    result = eval(tasks=contrastive_blueprint_labelling_task(num_variations=num_variations), limit=limit, model=[model])
    all_titles = []
    all_purposes = []
    for s in result[0].samples:
        all_titles.append(s.metadata.get('titles', []))
        all_purposes.append(s.metadata.get('purposes', []))
    samples = []
    for i, s in enumerate(result[0].samples):
        variations = s.metadata.get('titles' if subset == 'title' else 'purposes', [])
        all_choices = []
        try:
            while len(all_choices) < 3:
                sample_index = random.randint(0, len(all_titles))
                if sample_index != i:
                    if subset == 'title':
                        all_choices.append(random.choice(all_titles[sample_index]))
                    else:
                        all_choices.append(random.choice(all_purposes[sample_index]))
        except IndexError:
            continue
        for variation_idx, correct_answer in enumerate(variations):
            if not correct_answer:
                continue
            distractors = [choice for choice in all_choices if choice != correct_answer]
            if len(distractors) >= 3:
                other_options = random.sample(distractors, 3)
            else:
                other_options = distractors.copy()
                dummy_options = ['Belt Balancer System' if subset == 'title' else 'Distributes items evenly across multiple belt lanes', 'Automated Train Station' if subset == 'title' else 'Loading and unloading point for trains with circuit control', 'Steam Power Plant' if subset == 'title' else 'Generates electricity using steam engines and boilers']
                while len(other_options) < 3 and dummy_options:
                    other_options.append(dummy_options.pop(0))
            all_choices_for_question = [correct_answer] + other_options
            random.shuffle(all_choices_for_question)
            try:
                image: RenderedImage = instance.namespace._render(blueprint=s.metadata['blueprint'])
                from data.vqa.image_utils import save_rendered_image
                image_id = save_rendered_image(image, s.metadata['blueprint'], {**s.metadata, 'variation_idx': variation_idx}, f'contrastive_v{variation_idx}', os.getenv('VQA_DATASET_DIR'))
                files = {'image': image_id}
            except Exception as e:
                print(f'Error rendering blueprint: {e}')
                continue
            input_text = 'What is the best title for this blueprint?' if subset == 'title' else 'What is the purpose of this blueprint?'
            sample = Sample(choices=all_choices_for_question, target=str(correct_answer), input=input_text, files=files, metadata={**s.metadata, 'variation_idx': variation_idx, 'total_variations': len(variations)})
            samples.append(sample)
    dataset = MemoryDataset(samples)
    return dataset

