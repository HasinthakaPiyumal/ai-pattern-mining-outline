# Cluster 7

def generate_image_path_and_id(content: Union[Dict[str, Any], None]=None, metadata: Dict[str, Any]=None, modification_info: str='', base_dir: str='../../../dataset/images', is_map: bool=False) -> tuple[str, str]:
    """
    Generate the folder structure image path and ID for blueprints or maps.

    Args:
        content: Blueprint dictionary or None for maps
        metadata: Metadata containing filename or map info
        modification_info: Additional info for variants (denoising, etc.)
        base_dir: Base directory for images
        is_map: Whether this is a game map render

    Returns:
        Tuple of (file_path, image_id) where:
        - file_path: Full path where image should be saved
        - image_id: ID to use in metadata (relative path from base_dir)
    """
    if is_map:
        name = get_map_name(metadata or {})
        folder_path = Path(base_dir) / 'maps'
    else:
        if not content:
            raise ValueError('Blueprint content required when is_map=False')
        name = get_blueprint_name(content, metadata or {})
        folder_path = Path(base_dir) / name
    variant_hash = generate_variant_hash(content, modification_info, metadata, is_map)
    prefix = ''
    if metadata:
        if 'flip_suffix' in metadata:
            prefix = metadata['flip_suffix'] + '_'
        elif is_map and 'view_angle' in metadata:
            prefix = f'angle_{metadata['view_angle']}_'
    if is_map:
        image_id = f'maps/{name}_{prefix}{variant_hash}'
    else:
        image_id = f'{name}_{prefix}{variant_hash}'
    file_path = folder_path / f'{name}_{prefix}{variant_hash}.png'
    return (str(file_path), image_id)

def get_map_name(metadata: Dict[str, Any]) -> str:
    """
    Get a clean name for game map folder structure.

    Args:
        metadata: Metadata containing map information

    Returns:
        Clean map name suitable for folder name
    """
    if 'map_name' in metadata and metadata['map_name']:
        name = metadata['map_name']
    elif 'location' in metadata and metadata['location']:
        name = f'map_{metadata['location']}'
    elif 'position' in metadata:
        pos = metadata['position']
        if isinstance(pos, dict) and 'x' in pos and ('y' in pos):
            name = f'map_{int(pos['x'])}_{int(pos['y'])}'
        else:
            name = f'map_{str(pos).replace(',', '_').replace(' ', '')}'
    elif 'x' in metadata and 'y' in metadata:
        name = f'map_{int(metadata['x'])}_{int(metadata['y'])}'
    else:
        name = f'map_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    clean_name = ''.join((c if c.isalnum() or c in '-_' else '_' for c in name))
    if not clean_name or clean_name == '_':
        clean_name = 'map_unknown'
    if len(clean_name) > 50:
        clean_name = clean_name[:50]
    return clean_name

def get_blueprint_name(blueprint: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """
    Get a clean blueprint name for folder structure.

    Args:
        blueprint: Blueprint dictionary
        metadata: Metadata containing filename

    Returns:
        Clean blueprint name suitable for folder name
    """
    if 'label' in blueprint and blueprint['label']:
        name = blueprint['label']
    else:
        filename = metadata.get('filename', 'unknown')
        name = Path(filename).stem
    clean_name = ''.join((c if c.isalnum() or c in '-_' else '_' for c in name))
    if not clean_name or clean_name == '_':
        clean_name = 'unknown'
    if len(clean_name) > 50:
        clean_name = clean_name[:50]
    return clean_name

def generate_variant_hash(content: Union[Dict[str, Any], None]=None, modification_info: str='', metadata: Dict[str, Any]=None, is_map: bool=False) -> str:
    """
    Generate a hash representing this specific variant of the blueprint or map.

    Args:
        content: Blueprint dictionary or None for maps
        modification_info: Additional info about modifications (for denoising, etc.)
        metadata: Metadata that may contain rotation info, position, etc.
        is_map: Whether this is a game map render

    Returns:
        Short hash string for this variant
    """
    variant_components = []
    if is_map and metadata:
        variant_components.extend(['map_render', str(metadata.get('position', '')), str(metadata.get('radius', 64)), str(metadata.get('layers', 'all')), str(metadata.get('include_status', False)), str(metadata.get('timestamp', ''))])
    elif content:
        variant_components.append(str(content))
    variant_components.append(modification_info)
    if metadata:
        rotation = metadata.get('rotation', '')
        rotation_degrees = metadata.get('rotation_degrees', '')
        variant_components.extend([rotation, str(rotation_degrees)])
    variant_string = '|'.join(variant_components)
    hash_object = hashlib.md5(variant_string.encode())
    return hash_object.hexdigest()[:12]

