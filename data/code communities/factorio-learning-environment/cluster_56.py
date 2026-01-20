# Cluster 56

def render_inventory2(entity: Dict, grid, image_resolver: Callable) -> Optional[Image.Image]:
    """Transport belts display their contents on them"""
    inventory = entity.get('inventory', {})
    if not inventory:
        return None
    direction = entity.get('direction', 0)
    if not isinstance(direction, int):
        direction = direction.value
    from PIL import Image
    import math
    canvas_size = 64
    overlay = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    from ..constants import VERTICAL, EAST, WEST, NORTH, SOUTH
    around = get_around(entity, grid)
    count = sum(around)
    degree_offset = 90
    is_bend = False
    bend_type = None
    if count == 1:
        if around[0] == 1:
            if direction == EAST:
                is_bend = True
                bend_type = 'left'
                degree_offset = 180
            elif direction == WEST:
                is_bend = True
                bend_type = 'right'
                degree_offset = 90
            elif direction in VERTICAL:
                degree_offset = -90
        elif around[1] == 1:
            if direction == NORTH:
                is_bend = True
                bend_type = 'right'
                degree_offset = 90
            elif direction == SOUTH:
                is_bend = True
                bend_type = 'left'
                degree_offset = -180
        elif around[2] == 1:
            if direction == EAST:
                is_bend = True
                bend_type = 'right'
                degree_offset = 90
            elif direction == WEST:
                is_bend = True
                bend_type = 'left'
                degree_offset = 180
            elif direction in VERTICAL:
                degree_offset = -90
        elif around[3] == 1:
            if direction == NORTH:
                is_bend = True
                bend_type = 'left'
                degree_offset = -180
            elif direction == SOUTH:
                is_bend = True
                bend_type = 'right'
                degree_offset = 90
    elif count in [0, 2, 3]:
        if direction in VERTICAL:
            degree_offset = -90
    rotation = direction * 45 - degree_offset
    item_size = 16
    center = canvas_size // 2

    def place_items_on_path(items_dict, is_left_lane):
        """Place items along the belt path"""
        if not items_dict:
            return
        item_name = list(items_dict.keys())[0]
        item_count = min(items_dict[item_name], 4)
        item_icon = image_resolver(f'icon_{item_name}', False)
        if not item_icon:
            return
        item_icon = item_icon.resize((item_size, item_size), Image.Resampling.LANCZOS)
        for i in range(item_count):
            if is_bend:
                t = (i + 0.5) / 4.0
                is_inside = bend_type == 'left' and is_left_lane or (bend_type == 'right' and (not is_left_lane))
                lane_offset = 6
                if bend_type == 'left':
                    angle = t * math.pi / 2
                    center_x = center - 6 * math.sin(angle)
                    center_y = center - 6 * (1 - math.cos(angle))
                    normal_x = -math.cos(angle)
                    normal_y = -math.sin(angle)
                    if not is_inside:
                        x = center_x - normal_x * lane_offset
                        y = center_y - normal_y * lane_offset
                    else:
                        x = center_x + normal_x * lane_offset
                        y = center_y + normal_y * lane_offset
                else:
                    angle = t * math.pi / 2
                    center_x = center + 6 * math.sin(angle)
                    center_y = center - 6 * (1 - math.cos(angle))
                    normal_x = math.cos(angle)
                    normal_y = math.sin(angle)
                    if not is_inside:
                        x = center_x - normal_x * lane_offset
                        y = center_y - normal_y * lane_offset
                    else:
                        x = center_x + normal_x * lane_offset
                        y = center_y - normal_y * lane_offset
            else:
                spacing = 8
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
            x_pos = int(x - item_size / 2)
            y_pos = int(y - item_size / 2)
            overlay.paste(item_icon, (x_pos, y_pos), item_icon if item_icon.mode == 'RGBA' else None)
    place_items_on_path(inventory.get('left', {}), True)
    place_items_on_path(inventory.get('right', {}), False)
    if rotation != 0:
        overlay = overlay.rotate(-rotation, expand=True)
    final_size = 32
    if overlay.size[0] > final_size or overlay.size[1] > final_size:
        left = (overlay.width - final_size) // 2
        top = (overlay.height - final_size) // 2
        right = left + final_size
        bottom = top + final_size
        overlay = overlay.crop((left, top, right, bottom))
    if inventory.get('left') or inventory.get('right'):
        return overlay
    return None

def place_items_on_path(items_dict, is_left_lane):
    """Place items along the belt path"""
    if not items_dict:
        return
    item_name = list(items_dict.keys())[0]
    item_count = min(items_dict[item_name], 4)
    item_icon = image_resolver(f'icon_{item_name}', False)
    if not item_icon:
        return
    item_icon = item_icon.resize((item_size, item_size), Image.Resampling.LANCZOS)
    for i in range(item_count):
        if is_bend:
            t = (i + 0.5) / 4.0
            is_inside = bend_type == 'left' and is_left_lane or (bend_type == 'right' and (not is_left_lane))
            lane_offset = 6
            if bend_type == 'left':
                angle = t * math.pi / 2
                center_x = center - 6 * math.sin(angle)
                center_y = center - 6 * (1 - math.cos(angle))
                normal_x = -math.cos(angle)
                normal_y = -math.sin(angle)
                if not is_inside:
                    x = center_x - normal_x * lane_offset
                    y = center_y - normal_y * lane_offset
                else:
                    x = center_x + normal_x * lane_offset
                    y = center_y + normal_y * lane_offset
            else:
                angle = t * math.pi / 2
                center_x = center + 6 * math.sin(angle)
                center_y = center - 6 * (1 - math.cos(angle))
                normal_x = math.cos(angle)
                normal_y = math.sin(angle)
                if not is_inside:
                    x = center_x - normal_x * lane_offset
                    y = center_y - normal_y * lane_offset
                else:
                    x = center_x + normal_x * lane_offset
                    y = center_y - normal_y * lane_offset
        else:
            spacing = 8
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
        x_pos = int(x - item_size / 2)
        y_pos = int(y - item_size / 2)
        overlay.paste(item_icon, (x_pos, y_pos), item_icon if item_icon.mode == 'RGBA' else None)

