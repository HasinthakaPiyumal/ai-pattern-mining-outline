# Cluster 55

def render(entity: Dict, grid, image_resolver: Callable) -> Optional[Image.Image]:
    """Render character based on state and direction."""
    direction = entity.get('direction', 0)
    state = entity.get('state', 'idle')
    level = entity.get('level', 1)
    has_gun = entity.get('has_gun', False)
    player_color = entity.get('color', DEFAULT_PLAYER_COLOR)
    animation_frame = entity.get('animation_frame', 0)
    config = get_sprite_config(state, has_gun)
    cols, rows = config['grid']
    direction_mapping = DIRECTION_MAPPINGS[config['directions']]
    if state == 'idle':
        if has_gun:
            base_name = f'level{level}_idle_gun'
            mask_name = f'level{level}_idle_gun_mask'
        else:
            base_name = f'level{level}_idle'
            mask_name = f'level{level}_idle_mask'
    elif state == 'running':
        if has_gun:
            base_name = f'level{level}_running_gun'
            mask_name = f'level{level}_running_gun_mask'
        else:
            base_name = f'level{level}_running'
            mask_name = f'level{level}_running_mask'
    elif state == 'mining':
        base_name = f'level{level}_mining_tool'
        mask_name = f'level{level}_mining_tool_mask'
    elif state == 'dead':
        base_name = f'level{level}_dead'
        mask_name = f'level{level}_dead_mask'
    else:
        base_name = f'level{level}_idle'
        mask_name = f'level{level}_idle_mask'
    if level > 1:
        base_name = base_name.replace(f'level{level}', f'level{level}addon')
        mask_name = mask_name.replace(f'level{level}', f'level{level}addon')
    variant = direction_mapping.get(direction, 0)
    direction_row = min(animation_frame, rows - 1)
    sprite_filename = f'{base_name}_{variant}_{direction_row}'
    mask_filename = f'{mask_name}_{variant}_{direction_row}'
    base_sprite = image_resolver(f'character/{sprite_filename}', False)
    if not base_sprite:
        base_sprite = image_resolver(sprite_filename, False)
    if not base_sprite:
        return None
    mask_sprite = image_resolver(f'character/{mask_filename}', False)
    if not mask_sprite:
        mask_sprite = image_resolver(mask_filename, False)
    if mask_sprite:
        colored_mask = apply_color_to_mask(mask_sprite, player_color)
        result = Image.new('RGBA', base_sprite.size, (0, 0, 0, 0))
        result.paste(base_sprite, (0, 0), base_sprite)
        result.paste(colored_mask, (9, 0), colored_mask)
        return result
    return base_sprite

def get_sprite_config(state: str, has_gun: bool=False) -> Dict:
    """Get the sprite configuration for a given state."""
    if state == 'idle':
        return SPRITE_CONFIGS['idle_gun' if has_gun else 'idle']
    elif state == 'running':
        return SPRITE_CONFIGS['running_gun' if has_gun else 'running']
    elif state == 'mining':
        return SPRITE_CONFIGS['mining']
    elif state == 'dead':
        return SPRITE_CONFIGS['dead']
    else:
        return SPRITE_CONFIGS['idle']

def apply_color_to_mask(mask: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
    """Apply color tinting to a mask image.

    Args:
        mask: The mask image (grayscale)
        color: RGB color tuple to apply

    Returns:
        Colored mask image
    """
    if mask.mode != 'RGBA':
        mask = mask.convert('RGBA')
    result = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    pixels = mask.load()
    result_pixels = result.load()
    for y in range(mask.height):
        for x in range(mask.width):
            r, g, b, a = pixels[x, y]
            brightness = (r + g + b) // 3
            if brightness > 0:
                result_pixels[x, y] = (int(color[0] * brightness / 255), int(color[1] * brightness / 255), int(color[2] * brightness / 255), a)
    return result

