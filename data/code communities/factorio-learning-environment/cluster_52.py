# Cluster 52

def is_entity_in_direction(entity: Optional[Dict], target: str, direction: int) -> bool:
    """Check if entity matches target name and direction."""
    if not is_entity(entity, target):
        return False
    return entity.get('direction', 0) == direction

def is_entity(entity: Optional[Dict], target: str) -> bool:
    """Check if entity matches target name."""
    if entity is None:
        return False
    return entity.get('name') == target

def get_around(entity: Dict, grid) -> list:
    """Check surrounding pipe connections"""
    return [is_pipe(grid.get_relative(0, -1), 4) or is_entity_in_direction(grid.get_relative(0, -1), 'offshore-pump', 0), is_pipe(grid.get_relative(1, 0), 6) or is_entity_in_direction(grid.get_relative(1, 0), 'offshore-pump', 2), is_pipe(grid.get_relative(0, 1), 0) or is_entity_in_direction(grid.get_relative(0, 1), 'offshore-pump', 4), is_pipe(grid.get_relative(-1, 0), 2) or is_entity_in_direction(grid.get_relative(-1, 0), 'offshore-pump', 6)]

def get_around(entity: Dict, grid) -> list:
    """Check surrounding heat connections"""
    return [is_heat_pipe(grid.get_relative(0, -1)) or is_entity_in_direction(grid.get_relative(0, -1.5), 'heat-exchanger', 0) or is_entity(grid.get_relative(-2, -3), 'nuclear-reactor') or is_entity(grid.get_relative(0, -3), 'nuclear-reactor') or is_entity(grid.get_relative(2, -3), 'nuclear-reactor'), is_heat_pipe(grid.get_relative(1, 0)) or is_entity_in_direction(grid.get_relative(1.5, 0), 'heat-exchanger', 2) or is_entity(grid.get_relative(3, -2), 'nuclear-reactor') or is_entity(grid.get_relative(3, 0), 'nuclear-reactor') or is_entity(grid.get_relative(3, 2), 'nuclear-reactor'), is_heat_pipe(grid.get_relative(0, 1)) or is_entity_in_direction(grid.get_relative(0, 1.5), 'heat-exchanger', 4) or is_entity(grid.get_relative(-2, 3), 'nuclear-reactor') or is_entity(grid.get_relative(0, 3), 'nuclear-reactor') or is_entity(grid.get_relative(2, 3), 'nuclear-reactor'), is_heat_pipe(grid.get_relative(-1, 0)) or is_entity_in_direction(grid.get_relative(-1.5, 0), 'heat-exchanger', 6) or is_entity(grid.get_relative(-3, -2), 'nuclear-reactor') or is_entity(grid.get_relative(-3, 0), 'nuclear-reactor') or is_entity(grid.get_relative(-3, 2), 'nuclear-reactor')]

