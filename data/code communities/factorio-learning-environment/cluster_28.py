# Cluster 28

def dfs(entity: BlueprintEntity, current_segment: List[BlueprintEntity]):
    visited.add((entity.position['x'], entity.position['y']))
    current_segment.append(entity)
    for neighbor in get_neighbors(entity):
        pos = (neighbor.position['x'], neighbor.position['y'])
        if pos not in visited:
            dfs(neighbor, current_segment)

def get_neighbors(entity: BlueprintEntity) -> List[BlueprintEntity]:
    """Get adjacent belt entities."""
    x, y = (entity.position['x'], entity.position['y'])
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor_pos = (x + dx, y + dy)
        if neighbor_pos in pos_map:
            neighbors.append(pos_map[neighbor_pos])
    return neighbors

def find_belt_segments(entities: List[BlueprintEntity]) -> List[List[BlueprintEntity]]:
    """
    Find contiguous segments of transport belts.
    Returns list of lists, where each inner list is a contiguous belt segment.
    """
    belt_entities = [e for e in entities if is_transport_belt(e.name)]
    if not belt_entities:
        return []
    pos_map = {(e.position['x'], e.position['y']): e for e in belt_entities}

    def get_neighbors(entity: BlueprintEntity) -> List[BlueprintEntity]:
        """Get adjacent belt entities."""
        x, y = (entity.position['x'], entity.position['y'])
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_pos = (x + dx, y + dy)
            if neighbor_pos in pos_map:
                neighbors.append(pos_map[neighbor_pos])
        return neighbors
    segments = []
    visited = set()

    def dfs(entity: BlueprintEntity, current_segment: List[BlueprintEntity]):
        visited.add((entity.position['x'], entity.position['y']))
        current_segment.append(entity)
        for neighbor in get_neighbors(entity):
            pos = (neighbor.position['x'], neighbor.position['y'])
            if pos not in visited:
                dfs(neighbor, current_segment)
    for belt in belt_entities:
        pos = (belt.position['x'], belt.position['y'])
        if pos not in visited:
            current_segment = []
            dfs(belt, current_segment)
            segments.append(current_segment)
    return segments

def find_placement_references(entities: List[BlueprintEntity]) -> List[EntityPlacement]:
    """
    Determine relative positioning between entities.
    Returns list of EntityPlacement objects with reference entities and offsets.
    """
    placements = []
    placed_entities = {}
    sorted_entities = sorted(entities, key=lambda e: (get_entity_priority(e.name).value, e.entity_number))
    for entity in sorted_entities:
        current_pos = (entity.position['x'], entity.position['y'])
        closest_reference = None
        min_distance = float('inf')
        relative_pos = None
        for placed_pos, placed_num in placed_entities.items():
            dx = current_pos[0] - placed_pos[0]
            dy = current_pos[1] - placed_pos[1]
            distance = abs(dx) + abs(dy)
            if distance < min_distance and (dx == 0 or dy == 0):
                min_distance = distance
                closest_reference = placed_num
                relative_pos = (dx, dy)
        placements.append(EntityPlacement(entity, closest_reference, relative_pos))
        placed_entities[current_pos] = entity.entity_number
    return placements

def get_entity_priority(entity_name: str) -> EntityPriority:
    """Determine placement priority for different entity types."""
    if 'mining-drill' in entity_name:
        return EntityPriority.MINER
    elif 'assembling-machine' in entity_name:
        return EntityPriority.ASSEMBLER
    elif 'furnace' in entity_name:
        return EntityPriority.FURNACE
    elif 'inserter' in entity_name:
        return EntityPriority.INSERTER
    elif 'electric-pole' in entity_name or 'substation' in entity_name:
        return EntityPriority.POWER
    elif 'chest' in entity_name:
        return EntityPriority.CHEST
    elif 'belt' in entity_name or 'splitter' in entity_name:
        return EntityPriority.BELT
    return EntityPriority.CHEST

def convert_blueprint_to_trace(blueprint_json: str) -> List[str]:
    """Convert a Factorio blueprint JSON to a sequence of game commands."""
    blueprint = json.loads(blueprint_json)
    entities = [BlueprintEntity(**e) for e in blueprint['entities']]
    min_x = min((e.position['x'] for e in entities))
    min_y = min((e.position['y'] for e in entities))
    normalized_entities = []
    for entity in entities:
        new_entity = BlueprintEntity(entity_number=entity.entity_number, name=entity.name, position={'x': entity.position['x'] - min_x, 'y': entity.position['y'] - min_y}, direction=entity.direction, recipe=entity.recipe, neighbours=entity.neighbours, type=entity.type, items=entity.items)
        normalized_entities.append(new_entity)
    entities = normalized_entities
    resource = determine_resource_type(entities)
    miners = [p for p in entities if 'mining-drill' in p.name]
    try:
        find_valid_origin(entities, resource, instance)
    except Exception as e:
        raise ValueError(f'Failed to find valid origin: {e}. Skipping blueprint.')
    trace = []
    if resource and miners:
        trace.extend(create_origin_finding_code_trace(entities, resource))
    else:
        trace.append('origin = Position(x=0, y=0)')
        trace.append('')
    belt_entities = [e for e in entities if is_transport_belt(e.name)]
    non_belt_entities = [e for e in entities if not is_transport_belt(e.name)]
    placements = find_placement_references(non_belt_entities)
    belt_segments = find_belt_segments(belt_entities)
    tile_dimensions = get_tile_dimensions_of_all_entities(entities)
    placed_entity_vars = {}
    placed_entity = {}
    for placement in placements:
        entity = placement.entity
        entity_var = generate_entity_variable_name(entity.name, entities, entity.entity_number)
        placed_entity_vars[entity.entity_number] = entity_var
        placed_entity[entity.entity_number] = entity
        trace.append(f'# Place {entity.name}')
        if placement.reference_entity is None:
            trace.append(f'game.move_to(origin+Position(x={entity.position['x']},y={entity.position['y']}))')
            if 'mining-drill' in entity.name:
                trace.append(f'{entity_var} = game.place_entity(Prototype.{prototype_by_name[entity.name].name}, direction=Direction.{direction_to_enum(entity.direction)}, position=origin+Position(x={entity.position['x']},y={entity.position['y']}))')
            else:
                trace.append(f'{entity_var} = game.place_entity(Prototype.{prototype_by_name[entity.name].name}, direction=Direction.{direction_to_enum(entity.direction)}, position=origin+Position(x={entity.position['x']},y={entity.position['y']}))')
        else:
            ref_var = placed_entity_vars[placement.reference_entity]
            ref = placed_entity[placement.reference_entity]
            dx, dy = placement.relative_position
            direction_words = []
            position_ref_modifier = ['']
            spacing = 0
            if abs(dx) != 0 and abs(dy) != 0:
                raise ValueError('Diagonal placement not supported')
            if abs(dx) > abs(dy):
                direction_words = ['RIGHT' if dx > 0 else 'LEFT']
                spacing = abs(dx)
                spacing -= tile_dimensions[entity.name][0] / 2 + tile_dimensions[ref.name][0] / 2
                if 0 < abs(dy) < 1:
                    if dy > 0 and ref.name != 'burner-mining-drill':
                        position_ref_modifier = ['.above()'] * abs(dy)
                    if dy < 0:
                        position_ref_modifier = ['.below()'] * abs(dy)
                elif abs(dy) > 1:
                    position_ref_modifier = [f' + Position(x=0, y={dy})']
            else:
                direction_words = ['DOWN' if dy > 0 else 'UP']
                spacing = abs(dy)
                spacing -= tile_dimensions[entity.name][1] / 2 + tile_dimensions[ref.name][1] / 2
                if 0 < abs(dx) < 1:
                    if dx > 0 and ref.name != 'burner-mining-drill':
                        position_ref_modifier = ['.right()'] * abs(dx)
                    if dx < 0:
                        position_ref_modifier = ['.left()'] * abs(dx)
                elif abs(dx) > 1:
                    position_ref_modifier = [f' + Position(x={dx}, y=0)']
            placement_cmd = f'{entity_var} = game.place_entity_next_to(Prototype.{prototype_by_name[entity.name].name}, reference_position={ref_var}.position{''.join(position_ref_modifier)}, direction=Direction.{direction_words[0]}'
            if math.floor(spacing) != 0:
                placement_cmd += f', spacing={math.floor(spacing)})'
            else:
                placement_cmd += ')'
            trace.append(placement_cmd)
            trace.append(f'{entity_var} = game.rotate_entity({entity_var}, Direction.{direction_to_enum(entity.direction)})')
        trace.append(f"assert {entity_var}, 'Failed to place {entity.name}'")
        if entity.recipe:
            trace.append(f'game.set_entity_recipe({entity_var}, Prototype.{prototype_by_name[entity.recipe].name})')
        trace.append('')
    for segment_idx, segment in enumerate(belt_segments):
        start, end = find_segment_endpoints(segment)
        connection_type = 'Prototype.' + prototype_by_name[start.name].name
        if start.position['x'] > end.position['x']:
            direction = 6
        elif start.position['x'] < end.position['x']:
            direction = 2
        elif start.position['y'] > end.position['y']:
            direction = 0
        else:
            direction = 4
        trace.append(f'# Place transport belt segment {segment_idx + 1}')
        if len(segment) == 1:
            trace.append(f'game.move_to(origin+Position(x={segment[0].position['x']},y={segment[0].position['y']}))')
            trace.append(f'belt_segment_{segment_idx + 1} = game.place_entity({connection_type}, direction=Direction.{direction_to_enum(segment[0].direction)}, position=origin+Position(x={segment[0].position['x']}, y={segment[0].position['y']}))')
            trace.append(f"assert belt_segment_{segment_idx + 1}, 'Failed to place belt segment {segment_idx + 1}'")
            trace.append('')
            continue
        start_ref = None
        end_ref = None
        start_closest_entity, end_closest_entity = (None, None)
        start_closest_var, end_closest_var = (None, None)
        start_closest_distance, end_closest_distance = (float('inf'), float('inf'))
        for entity in non_belt_entities:
            start_distance = abs(entity.position['x'] - start.position['x']) + abs(entity.position['y'] - start.position['y'])
            end_distance = abs(entity.position['x'] - end.position['x']) + abs(entity.position['y'] - end.position['y'])
            if start_distance < start_closest_distance:
                start_closest_entity = entity
                start_closest_distance = start_distance
                start_closest_var = placed_entity_vars[entity.entity_number]
            if end_distance < end_closest_distance:
                end_closest_entity = entity
                end_closest_distance = end_distance
                end_closest_var = placed_entity_vars[entity.entity_number]
            if abs(entity.position['x'] - start.position['x']) < 1 and abs(entity.position['y'] - start.position['y']) < 1:
                start_ref = f'{placed_entity_vars[entity.entity_number]}.position'
            if abs(entity.position['x'] - end.position['x']) < 1 and abs(entity.position['y'] - end.position['y']) < 1:
                end_ref = f'{placed_entity_vars[entity.entity_number]}.position'
        if not start_ref:
            dx = start.position['x'] - start_closest_entity.position['x']
            dy = start.position['y'] - start_closest_entity.position['y']
            start_ref = f'Position(x={start_closest_var}.position.x+{dx}, y={start_closest_var}.position.y+{dy})'
        if not end_ref:
            dx = end.position['x'] - end_closest_entity.position['x']
            dy = end.position['y'] - end_closest_entity.position['y']
            end_ref = f'Position(x={end_closest_var}.position.x+{dx}, y={end_closest_var}.position.y+{dy})'
        if start.position['y'] < end.position['y'] and start.position['x'] == end.position['x'] and (direction == 4):
            start_ref, end_ref = (end_ref, start_ref)
        elif start.position['x'] < end.position['x'] and start.position['y'] == end.position['y'] and (direction == 6):
            start_ref, end_ref = (end_ref, start_ref)
        elif start.position['y'] > end.position['y'] and start.position['x'] == end.position['x'] and (direction == 0):
            start_ref, end_ref = (end_ref, start_ref)
        elif start.position['x'] > end.position['x'] and start.position['y'] == end.position['y'] and (direction == 2):
            start_ref, end_ref = (end_ref, start_ref)
        trace.append(f'belt_segment_{segment_idx + 1} = game.connect_entities({start_ref}, {end_ref}, connection_type={connection_type})')
        trace.append(f"assert belt_segment_{segment_idx + 1}, 'Failed to place belt segment {segment_idx + 1}'")
        trace.append('')
    trace.append("\nprint('Successfully placed all blueprint entities')")
    return trace

def determine_resource_type(entities: List[BlueprintEntity]) -> Resource:
    """
    Determine which resource type the miners are likely targeting based on their positions.
    """
    miners = [e for e in entities if 'mining-drill' in e.name]
    if not miners:
        return None
    first_miner = miners[0]
    if first_miner.position['x'] < 0:
        return Resource.CopperOre
    elif first_miner.position['x'] > 0:
        return Resource.IronOre
    else:
        return Resource.Coal

def find_valid_origin(entities: List[BlueprintEntity], resource: Resource, game: FactorioInstance) -> Position:
    """
    Find a valid origin position using nearest_buildable with bounding box.
    """
    miners = [e for e in entities if 'mining-drill' in e.name]
    if not miners:
        return Position(x=0, y=0)
    min_x = min((e.position['x'] for e in entities))
    max_x = max((e.position['x'] for e in entities))
    min_y = min((e.position['y'] for e in entities))
    max_y = max((e.position['y'] for e in entities))
    base_miner = entities[0]
    left_top = Position(x=min_x - base_miner.position['x'], y=min_y - base_miner.position['y'])
    right_bottom = Position(x=max_x - base_miner.position['x'], y=max_y - base_miner.position['y'])
    left_bottom = Position(x=min_x - base_miner.position['x'], y=max_y - base_miner.position['y'])
    right_top = Position(x=max_x - base_miner.position['x'], y=min_y - base_miner.position['y'])
    bounding_box = BoundingBox(left_top=left_top, right_bottom=right_bottom, left_bottom=left_bottom, right_top=right_top)
    return game.nearest_buildable(prototype_by_name[miners[0].name], bounding_box)

def create_origin_finding_code_trace(entities, resource):
    """
    Generate code trace for finding origin using nearest_buildable.
    """
    miners = [e for e in entities if 'mining-drill' in e.name]
    if not miners:
        return []
    min_x = min((e.position['x'] for e in miners))
    max_x = max((e.position['x'] for e in miners))
    min_y = min((e.position['y'] for e in miners))
    max_y = max((e.position['y'] for e in miners))
    base_miner = miners[0]
    trace = [f'# Find suitable origin position for miners on {resource[0]}', '', '# Calculate bounding box for miners', 'left_top = Position(', f'    x={min_x - base_miner.position['x']},', f'    y={min_y - base_miner.position['y']}', ')', 'right_bottom = Position(', f'    x={max_x - base_miner.position['x']},', f'    y={max_y - base_miner.position['y']}', ')', 'center = Position(', '    x=(left_top.x + right_bottom.x) / 2,', '    y=(left_top.y + right_bottom.y) / 2', ')', '', 'miner_box = BoundingBox(', '    left_top=left_top,', '    right_bottom=right_bottom,', '    center=center', ')', '', '# Find valid position for miners using nearest_buildable', 'origin = game.nearest_buildable(', f'    Prototype.{prototype_by_name[base_miner.name].name},', '    bounding_box=miner_box', ')', '', "assert origin, 'Could not find valid position for miners'", '', '# Move to origin position', 'game.move_to(origin)', '']
    return trace

def is_transport_belt(entity_name: str) -> bool:
    """Check if an entity is a transport belt."""
    return 'transport-belt' in entity_name

def get_tile_dimensions_of_all_entities(entities: List[BlueprintEntity]) -> Dict[str, Tuple[int, int]]:
    tile_dimensions = {}
    entity_counts = {}
    for entity in entities:
        entity_counts[entity.name] = entity_counts.get(entity.name, 0) + 1
    instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, cache_scripts=False, inventory=entity_counts)
    entity_names = set([entity.name for entity in entities])
    position = instance.nearest(Resource.IronOre)
    instance.move_to(position)
    for entity_name in entity_names:
        entity = instance.place_entity(prototype_by_name[entity_name], Direction.UP, position)
        tile_dimensions[entity_name] = (entity.tile_dimensions.tile_width, entity.tile_dimensions.tile_height)
        instance.pickup_entity(entity)
    return tile_dimensions

def generate_entity_variable_name(entity_name: str, entities: List[BlueprintEntity], entity_number: int) -> str:
    """
    Generate a variable name for an entity based on naming conventions:
    - Single entities use just the entity name
    - Multiple instances use name with index
    """
    name = entity_name.replace('-', '_')
    count = sum((1 for e in entities if e.name == entity_name))
    if count == 1:
        return name
    else:
        index = sum((1 for e in entities if e.name == entity_name and e.entity_number < entity_number))
        return f'{name}_{index + 1}'

def direction_to_enum(direction: int) -> str:
    """Convert numeric direction to Direction enum name."""
    direction_map = {0: 'UP', 2: 'RIGHT', 4: 'DOWN', 6: 'LEFT'}
    return direction_map.get(direction, 'UP')

def find_segment_endpoints(segment: List[BlueprintEntity]) -> Tuple[BlueprintEntity, BlueprintEntity]:
    """
    Find the start and end points of a belt segment.
    Returns (start_entity, end_entity).
    """
    if len(segment) == 1:
        return (segment[0], segment[0])
    pos_map = {(e.position['x'], e.position['y']): e for e in segment}
    endpoints = []
    for entity in segment:
        x, y = (entity.position['x'], entity.position['y'])
        neighbor_count = sum((1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] if (x + dx, y + dy) in pos_map))
        if neighbor_count == 1:
            endpoints.append(entity)
    if len(endpoints) == 2:
        endpoints.sort(key=lambda e: (e.position['x'], e.position['y']))
        return (endpoints[0], endpoints[1])
    segment.sort(key=lambda e: e.entity_number)
    return (segment[0], segment[-1])

def generate_trace(blueprint_json: str) -> str:
    """Generate the complete trace as a string."""
    trace_lines = convert_blueprint_to_trace(blueprint_json)
    return '\n'.join(trace_lines)

def verify_placement(game_entities, blueprint_json):

    def _get_hash(entities) -> str:
        pairs = []
        for i in range(len(entities)):
            for j in range(len(entities)):
                dx = entities[i].position['x'] - entities[j].position['x']
                dy = entities[i].position['y'] - entities[j].position['y']
                pairs.append((int(dx * 2), int(dy * 2)))
        pairs.sort()
        return (hash(tuple(pairs)), pairs)
    blueprint = json.loads(blueprint_json)
    entities = [BlueprintEntity(**e) for e in blueprint['entities']]
    hash1, blueprint_pairs = _get_hash(entities)
    positions = []
    for entity in game_entities:
        if isinstance(entity, EntityGroup):
            if hasattr(entity, 'belts'):
                positions.extend([(e.position.x, e.position.y) for e in entity.belts])
            elif hasattr(entity, 'pipes'):
                positions.extend([(e.position.x, e.position.y) for e in entity.pipes])
        else:
            positions.append((entity.position.x, entity.position.y))
    pairs = []
    for i in range(len(positions)):
        for j in range(len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            pairs.append((int(dx * 2), int(dy * 2)))
    pairs.sort()
    hash2 = hash(tuple(pairs))
    assert hash1 == hash2, f'The difference in entities is {set(blueprint_pairs) - set(pairs)}'

def _get_hash(entities) -> str:
    pairs = []
    for i in range(len(entities)):
        for j in range(len(entities)):
            dx = entities[i].position['x'] - entities[j].position['x']
            dy = entities[i].position['y'] - entities[j].position['y']
            pairs.append((int(dx * 2), int(dy * 2)))
    pairs.sort()
    return (hash(tuple(pairs)), pairs)

def get_inventory(blueprint_json):
    blueprint = json.loads(blueprint_json)
    entities = [BlueprintEntity(**e) for e in blueprint['entities']]
    entity_counts = {}
    for entity in entities:
        entity_counts[entity.name] = entity_counts.get(entity.name, 0) + 1
    return entity_counts

@profile_function('transport_belt.render', include_args=True)
def render(entity: Dict, grid, image_resolver: Callable) -> Optional[Image.Image]:
    """Render transport belt"""
    around = get_around(entity, grid)
    count = sum(around)
    direction = entity.get('direction', 0)
    if not isinstance(direction, int):
        direction = direction.value
    degree_offset = 90
    image = None
    if count in [0, 2, 3]:
        if direction in VERTICAL:
            image = image_resolver(f'{entity['name']}_vertical')
            degree_offset = -90
        else:
            image = image_resolver(f'{entity['name']}_horizontal')
    elif count == 1:
        if around[0] == 1:
            if direction in VERTICAL:
                image = image_resolver(f'{entity['name']}_vertical')
                degree_offset = -90
            elif direction == EAST:
                image = image_resolver(f'{entity['name']}_bend_left')
                degree_offset = 180
            elif direction == WEST:
                image = image_resolver(f'{entity['name']}_bend_right')
                degree_offset = 90
        elif around[1] == 1:
            if direction in HORIZONTAL:
                image = image_resolver(f'{entity['name']}_horizontal')
            elif direction == NORTH:
                image = image_resolver(f'{entity['name']}_bend_right')
                degree_offset = 90
            elif direction == SOUTH:
                image = image_resolver(f'{entity['name']}_bend_left')
                degree_offset = -180
        elif around[2] == 1:
            if direction in VERTICAL:
                image = image_resolver(f'{entity['name']}_vertical')
                degree_offset = -90
            elif direction == EAST:
                image = image_resolver(f'{entity['name']}_bend_right')
                degree_offset = 90
            elif direction == WEST:
                image = image_resolver(f'{entity['name']}_bend_left')
                degree_offset = 180
        elif around[3] == 1:
            if direction in HORIZONTAL:
                image = image_resolver(f'{entity['name']}_horizontal')
            elif direction == NORTH:
                image = image_resolver(f'{entity['name']}_bend_left')
                degree_offset = -180
            elif direction == SOUTH:
                image = image_resolver(f'{entity['name']}_bend_right')
                degree_offset = 90
    if image is None:
        return None
    rotation = direction * 45 - degree_offset
    if rotation != 0:
        image = image.rotate(-rotation, expand=True)
    return image

def profile_function(operation_name: str=None, include_args: bool=False):
    """Decorator to profile function execution time.

    Args:
        operation_name: Custom name for the operation (defaults to function name)
        include_args: Whether to include function arguments in metadata
    """

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not profiler.is_enabled():
                return func(*args, **kwargs)
            op_name = operation_name or f'{func.__module__}.{func.__name__}'
            metadata = {}
            if include_args:
                metadata['arg_count'] = len(args)
                metadata['kwarg_count'] = len(kwargs)
                if 'entity' in kwargs:
                    entity = kwargs['entity']
                    if isinstance(entity, dict):
                        metadata['entity_name'] = entity.get('name', 'unknown')
                    elif hasattr(entity, 'name'):
                        metadata['entity_name'] = entity.name
            with profiler.timer(op_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator

@profile_function('transport_belt.render_inventory', include_args=True)
def render_inventory(entity: Dict, grid, image_resolver: Callable) -> Optional[Image.Image]:
    """Transport belts display their contents on them"""
    inventory = entity.get('inventory', {})
    if not inventory or (not inventory.get('left') and (not inventory.get('right'))):
        return None
    direction = entity.get('direction', 0)
    if not isinstance(direction, int):
        direction = direction.value
    from PIL import Image
    import math
    from ..constants import VERTICAL, EAST, WEST, NORTH, SOUTH
    around = get_around(entity, grid)
    count = sum(around)
    degree_offset = 90
    belt_type = 'straight'
    if count == 1:
        if around[0] == 1:
            if direction == EAST:
                belt_type = 'bend_left'
                degree_offset = 180
            elif direction == WEST:
                belt_type = 'bend_right'
                degree_offset = 90
            elif direction in VERTICAL:
                belt_type = 'vertical'
                degree_offset = -90
        elif around[1] == 1:
            if direction == NORTH:
                belt_type = 'bend_right'
                degree_offset = 90
            elif direction == SOUTH:
                belt_type = 'bend_left'
                degree_offset = -180
            else:
                belt_type = 'horizontal'
        elif around[2] == 1:
            if direction == EAST:
                belt_type = 'bend_right'
                degree_offset = 90
            elif direction == WEST:
                belt_type = 'bend_left'
                degree_offset = 180
            elif direction in VERTICAL:
                belt_type = 'vertical'
                degree_offset = -90
        elif around[3] == 1:
            if direction == NORTH:
                belt_type = 'bend_left'
                degree_offset = -180
            elif direction == SOUTH:
                belt_type = 'bend_right'
                degree_offset = 90
            else:
                belt_type = 'horizontal'
    elif direction in VERTICAL:
        belt_type = 'vertical'
        degree_offset = -90
    else:
        belt_type = 'horizontal'
    rotation = direction * 45 - degree_offset
    overlay = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    item_size = 16
    max_items_per_lane = 4
    center = 32

    def place_items_on_lane(items_dict, is_left_lane):
        """Place items on a specific lane"""
        if not items_dict:
            return
        item_name = list(items_dict.keys())[0]
        item_count = min(items_dict[item_name], max_items_per_lane)
        choice = random.choice([1, 2, 3])
        item_icon = image_resolver(f'icon_{item_name}-{choice}', False)
        if not item_icon:
            item_icon = image_resolver(f'icon_{item_name}', False)
            if not item_icon:
                return
        item_icon = item_icon.resize((item_size, item_size), Image.Resampling.LANCZOS)
        positions = []
        spacing = 8
        if belt_type in ('horizontal', 'vertical'):
            for i in range(item_count):
                offset = -12 + i * spacing
                if direction in VERTICAL:
                    if direction == SOUTH:
                        if is_left_lane:
                            x = center + offset
                            y = center + 6
                        else:
                            x = center + offset
                            y = center - 6
                    elif is_left_lane:
                        x = center + offset
                        y = center - 6
                    else:
                        x = center + offset
                        y = center + 6
                elif is_left_lane:
                    x = center + offset
                    y = center - 6
                else:
                    x = center + offset
                    y = center + 6
                positions.append((x, y))
        elif belt_type == 'bend_left':
            for i in range(item_count):
                t = (i + 0.5) / max_items_per_lane
                if is_left_lane:
                    angle = t * math.pi / 2
                    radius = 18
                    center_x, center_y = (center - 10, center + 10)
                    x = center_x + radius * math.cos(angle)
                    y = center_y + radius * math.sin(angle)
                else:
                    angle = t * math.pi / 2
                    radius = 10
                    center_x, center_y = (center - 10, center + 10)
                    x = center_x + radius * math.sin(angle)
                    y = center_y + radius * math.cos(angle)
                positions.append((int(x), int(y)))
        elif belt_type == 'bend_right':
            for i in range(item_count):
                t = (i + 0.5) / max_items_per_lane
                if is_left_lane:
                    angle = t * math.pi / 2
                    radius = 10
                    center_x, center_y = (center + 10, center + 10)
                    x = center_x - radius * math.sin(angle)
                    y = center_y - radius * math.cos(angle)
                else:
                    angle = t * math.pi / 2
                    radius = 18
                    center_x, center_y = (center + 10, center + 10)
                    x = center_x - radius * math.cos(angle)
                    y = center_y - radius * math.sin(angle)
                positions.append((int(x), int(y)))
        for x, y in positions:
            paste_x = x - item_size // 2
            paste_y = y - item_size // 2
            overlay.paste(item_icon, (paste_x, paste_y), item_icon if item_icon.mode == 'RGBA' else None)
    place_items_on_lane(inventory.get('left', {}), True)
    place_items_on_lane(inventory.get('right', {}), False)
    if rotation != 0:
        overlay = overlay.rotate(-rotation, expand=False)
    return overlay

def place_items_on_lane(items_dict, is_left_lane):
    """Place items on a specific lane"""
    if not items_dict:
        return
    item_name = list(items_dict.keys())[0]
    item_count = min(items_dict[item_name], max_items_per_lane)
    choice = random.choice([1, 2, 3])
    item_icon = image_resolver(f'icon_{item_name}-{choice}', False)
    if not item_icon:
        item_icon = image_resolver(f'icon_{item_name}', False)
        if not item_icon:
            return
    item_icon = item_icon.resize((item_size, item_size), Image.Resampling.LANCZOS)
    positions = []
    spacing = 8
    if belt_type in ('horizontal', 'vertical'):
        for i in range(item_count):
            offset = -12 + i * spacing
            if direction in VERTICAL:
                if direction == SOUTH:
                    if is_left_lane:
                        x = center + offset
                        y = center + 6
                    else:
                        x = center + offset
                        y = center - 6
                elif is_left_lane:
                    x = center + offset
                    y = center - 6
                else:
                    x = center + offset
                    y = center + 6
            elif is_left_lane:
                x = center + offset
                y = center - 6
            else:
                x = center + offset
                y = center + 6
            positions.append((x, y))
    elif belt_type == 'bend_left':
        for i in range(item_count):
            t = (i + 0.5) / max_items_per_lane
            if is_left_lane:
                angle = t * math.pi / 2
                radius = 18
                center_x, center_y = (center - 10, center + 10)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
            else:
                angle = t * math.pi / 2
                radius = 10
                center_x, center_y = (center - 10, center + 10)
                x = center_x + radius * math.sin(angle)
                y = center_y + radius * math.cos(angle)
            positions.append((int(x), int(y)))
    elif belt_type == 'bend_right':
        for i in range(item_count):
            t = (i + 0.5) / max_items_per_lane
            if is_left_lane:
                angle = t * math.pi / 2
                radius = 10
                center_x, center_y = (center + 10, center + 10)
                x = center_x - radius * math.sin(angle)
                y = center_y - radius * math.cos(angle)
            else:
                angle = t * math.pi / 2
                radius = 18
                center_x, center_y = (center + 10, center + 10)
                x = center_x - radius * math.cos(angle)
                y = center_y - radius * math.sin(angle)
            positions.append((int(x), int(y)))
    for x, y in positions:
        paste_x = x - item_size // 2
        paste_y = y - item_size // 2
        overlay.paste(item_icon, (paste_x, paste_y), item_icon if item_icon.mode == 'RGBA' else None)

@profile_function('transport_belt.get_around')
def get_around(entity: Dict, grid) -> list:
    """Check surrounding connections"""
    return [is_transport_belt(grid.get_relative(0, -1), SOUTH) or is_splitter(grid.get_relative(0.5, -1), SOUTH) or is_splitter(grid.get_relative(-0.5, -1), SOUTH), is_transport_belt(grid.get_relative(1, 0), WEST) or is_splitter(grid.get_relative(1, 0.5), WEST) or is_splitter(grid.get_relative(1, -0.5), WEST), is_transport_belt(grid.get_relative(0, 1), NORTH) or is_splitter(grid.get_relative(0.5, 1), NORTH) or is_splitter(grid.get_relative(-0.5, 1), NORTH), is_transport_belt(grid.get_relative(-1, 0), EAST) or is_splitter(grid.get_relative(-1, 0.5), EAST) or is_splitter(grid.get_relative(-1, -0.5), EAST)]

def render(entity: Dict, grid, image_resolver: Callable) -> Optional[Image.Image]:
    """Render pipe based on connections"""
    around = get_around(entity, grid)
    count = sum(around)
    image_name = None
    if count == 0:
        image_name = 'pipe_straight_horizontal'
    elif count == 1:
        if around[0] == 1:
            image_name = 'pipe_ending_up'
        elif around[1] == 1:
            image_name = 'pipe_ending_right'
        elif around[2] == 1:
            image_name = 'pipe_ending_down'
        else:
            image_name = 'pipe_ending_left'
    elif count == 2:
        if around[0] == 1:
            if around[1] == 1:
                image_name = 'pipe_corner_up_right'
            elif around[2] == 1:
                image_name = 'pipe_straight_vertical'
            elif around[3] == 1:
                image_name = 'pipe_corner_up_left'
        elif around[1] == 1:
            if around[2] == 1:
                image_name = 'pipe_corner_down_right'
            elif around[3] == 1:
                image_name = 'pipe_straight_horizontal'
        else:
            image_name = 'pipe_corner_down_left'
    elif count == 3:
        if around[0] == 0:
            image_name = 'pipe_t_down'
        elif around[1] == 0:
            image_name = 'pipe_t_left'
        elif around[2] == 0:
            image_name = 'pipe_t_up'
        elif around[3] == 0:
            image_name = 'pipe_t_right'
    else:
        image_name = 'pipe_cross'
    return image_resolver(image_name)

def get_key(entity: Dict, grid) -> str:
    """Get cache key based on connections"""
    around = get_around(entity, grid)
    return '_'.join(map(str, around))

def get_name(entity: Dict, grid) -> str:
    """Get wall sprite name based on connections"""
    around = get_around(entity, grid)
    count = sum(around)
    if count == 0:
        return 'stone-wall_single'
    elif count == 1:
        if around[0] == 1:
            return 'stone-wall_single'
        elif around[1] == 1:
            return 'stone-wall_ending_right'
        elif around[2] == 1:
            return 'stone-wall_straight_vertical'
        else:
            return 'stone-wall_ending_left'
    elif count == 2:
        if around[0] == 1:
            if around[1] == 1:
                return 'stone-wall_ending_right'
            elif around[2] == 1:
                return 'stone-wall_straight_vertical'
            elif around[3] == 1:
                return 'stone-wall_ending_left'
        elif around[1] == 1:
            if around[2] == 1:
                return 'stone-wall_corner_right_down'
            elif around[3] == 1:
                return 'stone-wall_straight_horizontal'
        else:
            return 'stone-wall_corner_left_down'
    elif count == 3:
        if around[0] == 0:
            return 'stone-wall_t_up'
        elif around[1] == 0:
            return 'stone-wall_corner_left_down'
        elif around[2] == 0:
            return 'stone-wall_straight_horizontal'
        elif around[3] == 0:
            return 'stone-wall_corner_right_down'
    else:
        return 'stone-wall_t_up'

def get_key(entity: Dict, grid) -> str:
    """Get cache key based on connections"""
    around = get_around(entity, grid)
    return '_'.join(map(str, around))

def render(entity: Dict, grid, image_resolver: Callable) -> Optional[Image.Image]:
    """Render heat pipe based on connections"""
    around = get_around(entity, grid)
    count = sum(around)
    image_name = None
    if count == 0:
        image_name = 'heat-pipe_single'
    elif count == 1:
        if around[0] == 1:
            image_name = 'heat-pipe_ending_up'
        elif around[1] == 1:
            image_name = 'heat-pipe_ending_right'
        elif around[2] == 1:
            image_name = 'heat-pipe_ending_down'
        else:
            image_name = 'heat-pipe_ending_left'
    elif count == 2:
        if around[0] == 1:
            if around[1] == 1:
                image_name = 'heat-pipe_corner_right_up'
            elif around[2] == 1:
                image_name = 'heat-pipe_straight_vertical'
            elif around[3] == 1:
                image_name = 'heat-pipe_corner_left_up'
        elif around[1] == 1:
            if around[2] == 1:
                image_name = 'heat-pipe_corner_right_down'
            elif around[3] == 1:
                image_name = 'heat-pipe_straight_horizontal'
        else:
            image_name = 'heat-pipe_corner_left_down'
    elif count == 3:
        if around[0] == 0:
            image_name = 'heat-pipe_t_down'
        elif around[1] == 0:
            image_name = 'heat-pipe_t_left'
        elif around[2] == 0:
            image_name = 'heat-pipe_t_up'
        elif around[3] == 0:
            image_name = 'heat-pipe_t_right'
    else:
        image_name = 'heat-pipe_cross'
    return image_resolver(image_name)

def get_key(entity: Dict, grid) -> str:
    """Get cache key based on connections"""
    around = get_around(entity, grid)
    return '_'.join(map(str, around))

