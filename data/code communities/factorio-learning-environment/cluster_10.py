# Cluster 10

def raw_position_dataset(pattern: str='concentric', limit: int=None) -> MemoryDataset:
    """
    Generate position dataset with various patterns.

    Args:
        pattern: One of "concentric", "spiral", "true_spiral", or "grid"
        limit: Maximum number of samples (None for all)

    Returns:
        MemoryDataset with position samples
    """
    samples = []
    if pattern == 'grid':
        positions = [(x, y) for x in range(-50, 51) for y in range(-50, 51)]
    elif pattern == 'concentric':
        positions = generate_concentric_spiral_positions(max_radius=50)
    elif pattern == 'spiral':
        positions = generate_spiral_positions(max_radius=50)
    elif pattern == 'true_spiral':
        positions = generate_true_spiral_positions(max_positions=10000)
    else:
        raise ValueError(f'Unknown pattern: {pattern}')
    if limit:
        positions = positions[:limit]
    for x, y in positions:
        sample = Sample(input=f'Position(x={x}, y={y})', metadata={'x': x, 'y': y})
        samples.append(sample)
    dataset = MemoryDataset(samples=samples[:10])
    return dataset

def generate_concentric_spiral_positions(max_radius: int=50) -> List[Tuple[int, int]]:
    """
    Generate positions in concentric squares expanding from origin.
    This creates a more predictable pattern than a true spiral.

    Args:
        max_radius: Maximum distance from origin

    Returns:
        List of (x, y) positions in concentric order
    """
    positions = [(0, 0)]
    for radius in range(1, max_radius + 1):
        for x in range(-radius, radius + 1):
            positions.append((x, -radius))
        for y in range(-radius + 1, radius):
            positions.append((radius, y))
        for x in range(radius, -radius - 1, -1):
            positions.append((x, radius))
        for y in range(radius - 1, -radius, -1):
            positions.append((-radius, y))
    return positions

def generate_spiral_positions(max_radius: int=50, step: int=1) -> List[Tuple[int, int]]:
    """
    Generate positions in a spiral pattern starting from origin.

    Args:
        max_radius: Maximum distance from origin to generate
        step: Step size between positions

    Returns:
        List of (x, y) positions in spiral order
    """
    positions = []
    x, y = (0, 0)
    dx, dy = (0, -step)
    positions.append((x, y))
    while max(abs(x), abs(y)) < max_radius:
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
            dx, dy = (-dy, dx)
        x, y = (x + dx, y + dy)
        positions.append((x, y))
    return positions

def generate_true_spiral_positions(max_positions: int=10000, spacing: float=1.0) -> List[Tuple[int, int]]:
    """
    Generate positions following an Archimedean spiral.

    Args:
        max_positions: Maximum number of positions to generate
        spacing: Distance between spiral arms

    Returns:
        List of (x, y) positions in spiral order
    """
    positions = []
    seen = set()
    theta = 0
    while len(positions) < max_positions:
        r = spacing * theta / (2 * math.pi)
        x = int(round(r * math.cos(theta)))
        y = int(round(r * math.sin(theta)))
        if (x, y) not in seen:
            positions.append((x, y))
            seen.add((x, y))
        theta += 0.1
        if r > 100:
            break
    return positions

def terrain_position_dataset() -> MemoryDataset:
    """
    Generate terrain positions in a concentric spiral pattern.
    This ensures we explore from the origin outward, which is more
    efficient for finding resources and buildable areas.
    """
    return raw_position_dataset(pattern='concentric', limit=None)

def raw_position_dataset_with_priority(max_radius: int=50, inner_radius_priority: int=10) -> MemoryDataset:
    """
    Generate position dataset with priority given to positions near origin.

    Args:
        max_radius: Maximum distance from origin
        inner_radius_priority: Positions within this radius are added first

    Returns:
        MemoryDataset with position samples
    """
    samples = []
    priority_positions = []
    regular_positions = []
    for x in range(-max_radius, max_radius + 1):
        for y in range(-max_radius, max_radius + 1):
            distance = math.sqrt(x * x + y * y)
            if distance <= inner_radius_priority:
                priority_positions.append((x, y, distance))
            elif distance <= max_radius:
                regular_positions.append((x, y, distance))
    priority_positions.sort(key=lambda p: p[2])
    regular_positions.sort(key=lambda p: p[2])
    for x, y, _ in priority_positions:
        sample = Sample(input=f'Position(x={x}, y={y})', metadata={'x': x, 'y': y, 'distance_from_origin': math.sqrt(x * x + y * y)})
        samples.append(sample)
    for x, y, _ in regular_positions:
        sample = Sample(input=f'Position(x={x}, y={y})', metadata={'x': x, 'y': y, 'distance_from_origin': math.sqrt(x * x + y * y)})
        samples.append(sample)
    dataset = MemoryDataset(samples=samples)
    return dataset

def raw_position_dataset(pattern: str='concentric', limit: int=None) -> MemoryDataset:
    """
    Generate position dataset with various patterns.

    Args:
        pattern: One of "concentric", "spiral", "true_spiral", or "grid"
        limit: Maximum number of samples (None for all)

    Returns:
        MemoryDataset with position samples
    """
    samples = []
    if pattern == 'grid':
        positions = [(x, y) for x in range(-50, 51) for y in range(-50, 51)]
    elif pattern == 'concentric':
        positions = generate_concentric_spiral_positions(max_radius=50)
    elif pattern == 'spiral':
        positions = generate_spiral_positions(max_radius=50)
    elif pattern == 'true_spiral':
        positions = generate_true_spiral_positions(max_positions=10000)
    else:
        raise ValueError(f'Unknown pattern: {pattern}')
    if limit:
        positions = positions[:limit]
    for x, y in positions:
        sample = Sample(input=f'Position(x={x}, y={y})', metadata={'x': x, 'y': y})
        samples.append(sample)
    dataset = MemoryDataset(samples=samples)
    return dataset

