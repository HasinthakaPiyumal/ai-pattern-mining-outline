# Cluster 8

def get_relative_position_description(x: float, y: float, bounding_box: Dict[str, float]) -> str:
    """
    Get a relative position description within the bounding box.

    Args:
        x: X coordinate
        y: Y coordinate
        bounding_box: Bounding box dictionary

    Returns:
        String description like "northwest", "center", "southeast", etc.
    """
    center_x, center_y = get_blueprint_center(bounding_box)
    if x < center_x - bounding_box['width'] * 0.1:
        horizontal = 'west'
    elif x > center_x + bounding_box['width'] * 0.1:
        horizontal = 'east'
    else:
        horizontal = 'center'
    if y < center_y - bounding_box['height'] * 0.1:
        vertical = 'north'
    elif y > center_y + bounding_box['height'] * 0.1:
        vertical = 'south'
    else:
        vertical = 'center'
    if horizontal == 'center' and vertical == 'center':
        return 'center'
    elif horizontal == 'center':
        return vertical
    elif vertical == 'center':
        return horizontal
    else:
        return f'{vertical}{horizontal}'

def get_blueprint_center(bounding_box: Dict[str, float]) -> Tuple[float, float]:
    """
    Get the center point of a bounding box.

    Args:
        bounding_box: Bounding box dictionary

    Returns:
        Tuple of (center_x, center_y)
    """
    center_x = (bounding_box['min_x'] + bounding_box['max_x']) / 2
    center_y = (bounding_box['min_y'] + bounding_box['max_y']) / 2
    return (center_x, center_y)

