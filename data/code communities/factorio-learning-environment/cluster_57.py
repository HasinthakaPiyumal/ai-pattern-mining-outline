# Cluster 57

def get_key(entity: Dict, grid) -> str:
    """Get cache key"""
    orientation = get_orientation(entity)
    cliff_type = determine_cliff_type_from_orientation(orientation)
    return f'{cliff_type}_{orientation}'

def get_orientation(entity: Dict) -> str:
    """Extract orientation from entity data"""
    orientation = entity.get('cliff_orientation', '').strip('"')
    if not orientation:
        if 'direction' in entity:
            direction = str(entity['direction']).strip('"')
            if direction and '-to-' in direction:
                orientation = direction
    return orientation or 'west-to-east'

def determine_cliff_type_from_orientation(orientation: str) -> str:
    """
    Determine cliff type based on orientation pattern.

    In Factorio:
    - cliff-sides: Standard straight cliff pieces and basic corners
    - cliff-outer: Convex (outward) corners
    - cliff-inner: Concave (inward) corners
    - cliff-entrance: End pieces and special transitions
    """
    from_dir, to_dir = parse_orientation(orientation)
    if from_dir == 'none' or to_dir == 'none':
        return 'cliff-entrance'
    direction_order = ['north', 'east', 'south', 'west']
    if from_dir in direction_order and to_dir in direction_order:
        from_idx = direction_order.index(from_dir)
        to_idx = direction_order.index(to_dir)
        turn = (to_idx - from_idx) % 4
        if turn == 0:
            return 'cliff-sides'
        elif turn == 2:
            return 'cliff-sides'
        elif turn == 1:
            return 'cliff-outer'
        elif turn == 3:
            return 'cliff-inner'
    return 'cliff-sides'

