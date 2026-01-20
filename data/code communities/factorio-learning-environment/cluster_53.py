# Cluster 53

class Renderer:
    """Factorio Blueprint representation."""

    @profile_method(include_args=True)
    def __init__(self, entities: Union[List[Dict], List[Entity]]=[], resources: List[Dict]=[], water_tiles: List[Dict]=[], sprites_dir: Optional[Path]=None, max_render_radius: Optional[float]=None, center_on_player: bool=True):
        """Initialize renderer with blueprint data.

        Args:
            entities: List of entities to render
            resources: List of resources to render
            water_tiles: List of water tiles to render
            sprites_dir: Optional directory path for sprite files
            max_render_radius: Optional maximum radius to render (for trimming captured area)
            center_on_player: Whether to center the rendering on the player position
        """
        self.icons = []
        self.max_render_radius = max_render_radius
        self.center_on_player = center_on_player
        flattened_entities = list(flatten_entities(entities))
        self.player_position = None
        if center_on_player:
            for entity in flattened_entities:
                if isinstance(entity, dict) and entity.get('name') == 'character':
                    pos = entity.get('position', {})
                    self.player_position = {'x': pos.get('x', 0), 'y': pos.get('y', 0)}
                    break
                elif hasattr(entity, 'name') and entity.name == 'character':
                    self.player_position = {'x': entity.position.x, 'y': entity.position.y}
                    break
        if self.player_position and center_on_player:
            self.offset_x = self.player_position['x']
            self.offset_y = self.player_position['y']
        else:
            min_x, min_y = self._find_min_coordinates(flattened_entities, resources, water_tiles)
            self.offset_x = min_x
            self.offset_y = min_y
        self.entities = self._normalize_positions(flattened_entities)
        self.resources = self._normalize_positions(resources)
        self.water_tiles = self._normalize_positions(water_tiles)
        self.entity_grid = entities_to_grid(self.entities)
        self.resource_grid = resources_to_grid(self.resources)
        self.sprites_dir = self._resolve_sprites_dir(sprites_dir)
        self.available_trees = build_available_trees_index(self.sprites_dir)
        self.tree_variants = self._precompute_tree_variants()
        self._sort_entities_for_rendering()

    @profile_method()
    def get_size(self) -> Dict:
        """Calculate blueprint bounds including resources and trees."""
        if self.max_render_radius is not None:
            return {'minX': -self.max_render_radius, 'minY': -self.max_render_radius, 'maxX': self.max_render_radius, 'maxY': self.max_render_radius, 'width': math.ceil(self.max_render_radius * 2), 'height': math.ceil(self.max_render_radius * 2)}
        bounds = self._calculate_bounds()
        content_width = bounds['max_width'] - bounds['min_width']
        content_height = bounds['max_height'] - bounds['min_height']
        max_dimension = max(content_width, content_height)
        width_diff = max_dimension - content_width
        height_diff = max_dimension - content_height
        adjusted_min_x = bounds['min_width'] - width_diff / 2
        adjusted_max_x = bounds['max_width'] + width_diff / 2
        adjusted_min_y = bounds['min_height'] - height_diff / 2
        adjusted_max_y = bounds['max_height'] + height_diff / 2
        return {'minX': adjusted_min_x, 'minY': adjusted_min_y, 'maxX': adjusted_max_x, 'maxY': adjusted_max_y, 'width': math.ceil(max_dimension), 'height': math.ceil(max_dimension)}

    def _calculate_bounds(self) -> Dict:
        """Calculate the bounding box for all entities and resources."""
        min_width = min_height = float('inf')
        max_width = max_height = float('-inf')
        for entity in self.entities:
            pos = entity.position
            size = renderer_manager.get_entity_size(entity)
            min_width = min(min_width, pos.x - size[0] / 2)
            min_height = min(min_height, pos.y - size[1] / 2)
            max_width = max(max_width, pos.x + size[0] / 2)
            max_height = max(max_height, pos.y + size[1] / 2)
        for resource in self.resources:
            pos = resource['position']
            min_width = min(min_width, pos['x'] - 0.5)
            min_height = min(min_height, pos['y'] - 0.5)
            max_width = max(max_width, pos['x'] + 0.5)
            max_height = max(max_height, pos['y'] + 0.5)
        for water_tile in self.water_tiles:
            pos = water_tile
            min_width = min(min_width, pos['x'] - 0.5)
            min_height = min(min_height, pos['y'] - 0.5)
            max_width = max(max_width, pos['x'] + 0.5)
            max_height = max(max_height, pos['y'] + 0.5)
        if min_width == float('inf'):
            min_width = -10
            max_width = 10
            min_height = -10
            max_height = 10
        return {'min_width': min_width, 'min_height': min_height, 'max_width': max_width, 'max_height': max_height}

    @profile_method()
    def _resolve_sprites_dir(self, sprites_dir: Optional[Path]) -> Path:
        """Resolve sprites directory location."""
        if sprites_dir is not None:
            return sprites_dir
        possible_dirs = [Path('.fle/sprites'), Path('sprites'), Path('images')]
        for dir_path in possible_dirs:
            if dir_path.exists():
                return dir_path
        return find_fle_sprites_dir()

    @profile_method()
    def _render_alert_overlays(self, img: Image.Image, entities, size: Dict, scaling: float, image_resolver) -> None:
        """Render alert overlays for entities with non-normal status."""
        status_alert_mapping = {EntityStatus.NO_POWER: 'alert-no-electricity', EntityStatus.LOW_POWER: 'alert-no-electricity', EntityStatus.NO_FUEL: 'alert-no-fuel', EntityStatus.EMPTY: 'alert-warning', EntityStatus.NOT_PLUGGED_IN_ELECTRIC_NETWORK: 'alert-disconnected', EntityStatus.CHARGING: 'alert-recharge-needed', EntityStatus.DISCHARGING: 'alert-recharge-needed', EntityStatus.FULLY_CHARGED: None, EntityStatus.NO_RECIPE: 'alert-no-building-materials', EntityStatus.NO_INGREDIENTS: 'alert-no-building-materials', EntityStatus.NOT_CONNECTED: 'alert-disconnected', EntityStatus.NO_INPUT_FLUID: 'alert-no-fluid', EntityStatus.NO_RESEARCH_IN_PROGRESS: 'alert-warning', EntityStatus.NO_MINABLE_RESOURCES: 'alert-warning', EntityStatus.LOW_INPUT_FLUID: 'alert-no-fluid', EntityStatus.FLUID_INGREDIENT_SHORTAGE: 'alert-no-fluid', EntityStatus.FULL_OUTPUT: 'alert-no-storage', EntityStatus.FULL_BURNT_RESULT_OUTPUT: 'alert-no-storage', EntityStatus.ITEM_INGREDIENT_SHORTAGE: 'alert-no-building-materials', EntityStatus.MISSING_REQUIRED_FLUID: 'alert-no-fluid', EntityStatus.MISSING_SCIENCE_PACKS: 'alert-no-building-materials', EntityStatus.WAITING_FOR_SOURCE_ITEMS: 'alert-logistic-delivery', EntityStatus.WAITING_FOR_SPACE_IN_DESTINATION: 'alert-no-storage', EntityStatus.PREPARING_ROCKET_FOR_LAUNCH: 'alert-warning', EntityStatus.WAITING_TO_LAUNCH_ROCKET: 'alert-warning', EntityStatus.LAUNCHING_ROCKET: 'alert-warning', EntityStatus.NO_AMMO: 'alert-no-ammo', EntityStatus.LOW_TEMPERATURE: 'alert-warning', EntityStatus.NOT_CONNECTED_TO_RAIL: 'alert-disconnected'}
        for entity in entities:
            if hasattr(entity, 'status'):
                status = entity.status
                entity_dict = entity.model_dump() if hasattr(entity, 'model_dump') else entity.__dict__
            elif isinstance(entity, dict) and 'status' in entity:
                status = entity['status']
                if isinstance(status, str):
                    status = EntityStatus.from_string(status)
                entity_dict = entity
            else:
                continue
            if status == EntityStatus.NO_POWER:
                if hasattr(entity, 'electrical_id'):
                    if not entity.electrical_id:
                        status = EntityStatus.NOT_PLUGGED_IN_ELECTRIC_NETWORK
                elif isinstance(entity, dict) and 'electrical_id' in entity:
                    if not entity['electrical_id']:
                        status = EntityStatus.NOT_PLUGGED_IN_ELECTRIC_NETWORK
            if status in (EntityStatus.NORMAL, EntityStatus.WORKING):
                continue
            alert_name = status_alert_mapping.get(status, 'alert-warning')
            if not alert_name:
                continue
            alert_icon = image_resolver(alert_name, False)
            if not alert_icon:
                alert_icon = image_resolver(f'icon_{alert_name}', False)
                if not alert_icon:
                    profiler.increment_counter('alert_icon_not_found')
                    continue
            pos = entity_dict.get('position', {})
            if hasattr(pos, 'x'):
                x, y = (pos.x, pos.y)
            else:
                x = pos.get('x', 0)
                y = pos.get('y', 0)
            if hasattr(entity, 'tile_dimensions'):
                entity_size = (entity.tile_dimensions.tile_width, entity.tile_dimensions.tile_height)
            else:
                entity_size = renderer_manager.get_entity_size(entity)
            base_scale_factor = 0.5 if entity_size[0] + entity_size[1] > 2 else 0.25
            scale_ratio = scaling / DEFAULT_SCALING
            scale_factor = base_scale_factor * scale_ratio
            new_width = int(alert_icon.width * scale_factor)
            new_height = int(alert_icon.height * scale_factor)
            alert_icon = alert_icon.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if alert_icon.mode != 'RGBA':
                alert_icon = alert_icon.convert('RGBA')
            pixels = alert_icon.load()
            for y_pixel in range(new_height):
                for x_pixel in range(new_width):
                    r, g, b, a = pixels[x_pixel, y_pixel]
                    pixels[x_pixel, y_pixel] = (r, g, b, int(a * 0.75))
            relative_x = x + abs(size['minX'])
            relative_y = y + abs(size['minY'])
            icon_offset_x = 0
            icon_offset_y = 0
            start_x = int(relative_x * scaling + scaling / 2 - new_width / 2 + icon_offset_x)
            start_y = int(relative_y * scaling + scaling / 2 - new_height / 2 + icon_offset_y)
            img.paste(alert_icon, (start_x, start_y), alert_icon)
            profiler.increment_counter('alert_overlays_rendered')

    @profile_method()
    def _precompute_tree_variants(self) -> Dict:
        """Pre-calculate tree variants for sorting."""
        tree_variants = {}
        for e in self.entities:
            entity = e.model_dump() if not isinstance(e, dict) else e
            if not is_tree_entity(entity['name']):
                continue
            x = entity['position']['x']
            y = entity['position']['y']
            tree_type = entity['name'].split('-')[-1] if '-' in entity['name'] else '01'
            if 'dead' not in entity['name'] and 'dry' not in entity['name']:
                variation, _ = get_tree_variant(x, y, tree_type, self.available_trees)
                tree_variants[id(entity)] = variation
            else:
                tree_variants[id(entity)] = 'z'
        return tree_variants

    @profile_method()
    def _sort_entities_for_rendering(self) -> None:
        """Sort entities for proper rendering order."""
        self.entities.sort(key=lambda e: (not is_tree_entity(e.name), -ord(self.tree_variants.get(id(e), 'a')) if is_tree_entity(e.name) else 0, not e.name.endswith('inserter'), e.position.y, e.position.x))

    def _get_position(self, item: Any) -> Optional[Dict[str, float]]:
        """Extract position from an item, handling both dict and object formats.

        Returns position as dict with 'x' and 'y' keys, or None if no position found.
        """
        if isinstance(item, dict) and 'x' in item and ('y' in item) and ('position' not in item):
            return {'x': item['x'], 'y': item['y']}
        if hasattr(item, 'position'):
            pos = item.position
            if hasattr(pos, 'x') and hasattr(pos, 'y'):
                return {'x': pos.x, 'y': pos.y}
            elif isinstance(pos, dict):
                return pos
        elif isinstance(item, dict) and 'position' in item:
            return item['position']
        return None

    def _set_position(self, item: Any, x: float, y: float) -> Any:
        """Set position on an item, handling both dict and object formats.

        Returns a copy of the item with updated position.
        """
        item_copy = copy.deepcopy(item)
        if isinstance(item_copy, dict) and 'x' in item_copy and ('y' in item_copy) and ('position' not in item_copy):
            item_copy['x'] = x
            item_copy['y'] = y
        elif hasattr(item_copy, 'position'):
            if hasattr(item_copy.position, 'x') and hasattr(item_copy.position, 'y'):
                item_copy.position.x = x
                item_copy.position.y = y
            elif isinstance(item_copy.position, dict):
                item_copy.position['x'] = x
                item_copy.position['y'] = y
        elif isinstance(item_copy, dict) and 'position' in item_copy:
            item_copy['position']['x'] = x
            item_copy['position']['y'] = y
        return item_copy

    def _find_min_coordinates(self, entities: List[Any], resources: List[Any], water_tiles: List[Any]) -> tuple[float, float]:
        """Find minimum x and y coordinates across all items."""
        min_x, min_y = (float('inf'), float('inf'))
        all_items = list(entities) + list(resources) + list(water_tiles)
        for item in all_items:
            pos = self._get_position(item)
            if pos:
                min_x = min(min_x, pos['x'])
                min_y = min(min_y, pos['y'])
        if min_x == float('inf'):
            min_x = 0
        if min_y == float('inf'):
            min_y = 0
        return (min_x, min_y)

    def _normalize_positions(self, items: List[Any]) -> List[Any]:
        """Normalize positions of all items by subtracting minimum coordinates."""
        normalized = []
        for item in items:
            pos = self._get_position(item)
            if pos:
                normalized_item = self._set_position(item, pos['x'] - self.offset_x, pos['y'] - self.offset_y)
                normalized.append(normalized_item)
            else:
                normalized.append(copy.deepcopy(item))
        return normalized

    @profile_method(include_args=True)
    def render(self, width: int, height: int, image_resolver) -> Image.Image:
        """Render blueprint to image.

        Args:
            width: Output image width
            height: Output image height
            image_resolver: Function to resolve sprite images

        Returns:
            Rendered PIL Image
        """
        size = self.get_size()
        scaling = min(width / size['width'], height / size['height'])
        img = self._create_base_image(width, height)
        self._draw_grid(img, size, scaling, width, height)
        tree_entities = [e.model_dump() if not isinstance(e, dict) else e for e in self.entities if is_tree_entity(e.name)]
        rock_entities = [e.model_dump() if not isinstance(e, dict) else e for e in self.entities if is_rock_entity(e.name)]
        player_entities = [e for e in self.entities if not is_tree_entity(e.name) and (not is_rock_entity(e.name))]
        player_entities = self._disintegrate_underground_belts(player_entities)
        grid_view = EntityGridView(self.entity_grid, 0, 0, self.available_trees)
        profiler.increment_counter('total_entities', len(self.entities))
        profiler.increment_counter('tree_entities', len(tree_entities))
        profiler.increment_counter('rock_entities', len(rock_entities))
        profiler.increment_counter('player_entities', len(player_entities))
        profiler.increment_counter('resources', len(self.resources))
        profiler.increment_counter('water_tiles', len(self.water_tiles))
        self._render_water_tiles(img, size, scaling, image_resolver)
        self._render_resources(img, size, scaling, image_resolver)
        self._render_tree_shadows(img, tree_entities, size, scaling, grid_view, image_resolver)
        self._render_trees(img, tree_entities, size, scaling, grid_view, image_resolver)
        self._render_decoratives(img, rock_entities, size, scaling, image_resolver)
        self._render_entity_shadows(img, player_entities, size, scaling, grid_view, image_resolver)
        self._render_rails(img, player_entities, size, scaling, image_resolver)
        self._render_entities(img, player_entities, size, scaling, grid_view, image_resolver)
        self._render_visible_inventories(img, player_entities, size, scaling, grid_view, image_resolver)
        self._render_alert_overlays(img, player_entities, size, scaling, image_resolver)
        return img

    def _disintegrate_underground_belts(self, player_entities):
        entities = []
        for entity in player_entities:
            if isinstance(entity, UndergroundBelt):
                entities.append(entity)
                output = copy.deepcopy(entity)
                output.is_input = False
                output.position = Position(x=output.output_position.x - self.offset_x, y=output.output_position.y - self.offset_y)
                entities.append(output)
            else:
                entities.append(entity)
        return entities

    @profile_method()
    def _create_base_image(self, width: int, height: int) -> Image.Image:
        """Create base image with background color."""
        return Image.new('RGB', (width, height), BACKGROUND_COLOR)

    @profile_method()
    def _draw_grid(self, img: Image.Image, size: Dict, scaling: float, width: int, height: int) -> None:
        """Draw grid lines on the image with different thicknesses based on game positions."""
        draw = ImageDraw.Draw(img)
        game_offset_x = round(self.offset_x)
        game_offset_y = round(self.offset_y)
        min_game_x = int(math.floor(size['minX'] + game_offset_x))
        max_game_x = int(math.ceil(size['maxX'] + game_offset_x))
        min_game_y = int(math.floor(size['minY'] + game_offset_y))
        max_game_y = int(math.ceil(size['maxY'] + game_offset_y))
        for game_x in range(min_game_x, max_game_x + 1):
            norm_x = game_x - game_offset_x
            pixel_x = (norm_x - size['minX']) * scaling
            if pixel_x < -5 or pixel_x > width + 5:
                continue
            if game_x % 10 == 0:
                line_width = 2
                line_color = GRID_COLOR_THICK
            elif game_x % 5 == 0:
                line_width = 1
                line_color = GRID_COLOR_MEDIUM
            else:
                line_width = 1
                line_color = GRID_COLOR_THIN
            x_center = int(pixel_x)
            half_width = line_width // 2
            x_start = x_center - half_width
            x_end = x_center + half_width
            if line_width % 2 == 1:
                x_end += 1
            x_start = max(0, x_start)
            x_end = min(width, x_end)
            if x_end > x_start:
                if line_width == 1:
                    draw.line([x_center, 0, x_center, height], fill=line_color, width=1)
                else:
                    draw.rectangle([x_start, 0, x_end - 1, height - 1], fill=line_color)
        for game_y in range(min_game_y, max_game_y + 1):
            norm_y = game_y - game_offset_y
            pixel_y = (norm_y - size['minY']) * scaling
            if pixel_y < -5 or pixel_y > height + 5:
                continue
            scale_ratio = scaling / DEFAULT_SCALING
            if game_y % 10 == 0:
                line_width = max(1, int(GRID_LINE_WIDTH_THICK * scale_ratio))
                line_color = GRID_COLOR_THICK
            elif game_y % 5 == 0:
                line_width = max(1, int(GRID_LINE_WIDTH_MEDIUM * scale_ratio))
                line_color = GRID_COLOR_MEDIUM
            else:
                line_width = max(1, int(GRID_LINE_WIDTH_THIN * scale_ratio))
                line_color = GRID_COLOR_THIN
            y_center = int(pixel_y)
            half_width = line_width // 2
            y_start = y_center - half_width
            y_end = y_center + half_width
            if line_width % 2 == 1:
                y_end += 1
            y_start = max(0, y_start)
            y_end = min(height, y_end)
            if y_end > y_start:
                if line_width == 1:
                    draw.line([0, y_center, width, y_center], fill=line_color, width=1)
                else:
                    draw.rectangle([0, y_start, width - 1, y_end - 1], fill=line_color)

    @profile_method()
    def _render_resources(self, img: Image.Image, size: Dict, scaling: float, image_resolver) -> None:
        """Render resource patches."""
        for resource in self.resources:
            pos = resource['position']
            relative_x = pos['x'] + abs(size['minX'])
            relative_y = pos['y'] + abs(size['minY'])
            if resource['name'] == 'crude-oil':
                volume = 1
                variant = get_resource_variant(pos['x'], pos['y'], max_variants=OIL_RESOURCE_VARIANTS)
            else:
                volume = get_resource_volume(resource.get('amount', 10000))
                variant = get_resource_variant(pos['x'], pos['y'])
            sprite_name = f'{resource['name']}_{variant}_{volume}'
            image = image_resolver(sprite_name, False)
            if image:
                self._paste_image(img, image, relative_x, relative_y, scaling)

    def _render_decoratives(self, img: Image.Image, decoratives: List[Dict], size: Dict, scaling: float, image_resolver) -> None:
        """Render decoratives."""
        for decorative in decoratives:
            pos = decorative['position']
            relative_x = pos['x'] + abs(size['minX'])
            relative_y = pos['y'] + abs(size['minY'])
            variant = get_resource_variant(pos['x'], pos['y'], max_variants=DEFAULT_ROCK_VARIANTS)
            sprite_name = f'{decorative['name']}_{variant}'
            image = image_resolver(sprite_name, False)
            if image:
                self._paste_image(img, image, relative_x, relative_y, scaling)
            else:
                while not image and variant < DEFAULT_ROCK_VARIANTS:
                    variant = variant + 1
                    sprite_name = f'{decorative['name']}_{variant}'
                    image = image_resolver(sprite_name, False)
                    if image:
                        self._paste_image(img, image, relative_x, relative_y, scaling)
                        break

    @profile_method()
    def _render_water_tiles(self, img: Image.Image, size: Dict, scaling: float, image_resolver) -> None:
        """Render water tiles."""
        for water in self.water_tiles:
            pos = water
            relative_x = pos['x'] + abs(size['minX']) + 0.5
            relative_y = pos['y'] + abs(size['minY']) + 0.5
            volume = 1
            variant = get_resource_variant(pos['x'], pos['y'], max_variants=DEFAULT_RESOURCE_VARIANTS)
            sprite_name = f'{water['name']}_{variant}_{volume}'
            image = image_resolver(sprite_name, False)
            if image:
                self._paste_image(img, image, relative_x, relative_y, scaling)

    @profile_method()
    def _render_tree_shadows(self, img: Image.Image, tree_entities, size: Dict, scaling: float, grid_view, image_resolver) -> None:
        """Render tree shadows."""
        for tree in tree_entities:
            pos = tree['position']
            relative_x = pos['x'] + abs(size['minX'])
            relative_y = pos['y'] + abs(size['minY'])
            grid_view.set_center(pos['x'], pos['y'])
            renderer = renderer_manager.get_renderer(tree['name'])
            if renderer and hasattr(renderer, 'render_shadow'):
                shadow_image = renderer.render_shadow(tree, grid_view, image_resolver)
                if shadow_image:
                    shadow_offset_x = 32
                    shadow_offset_y = 32
                    self._paste_image(img, shadow_image, relative_x, relative_y, scaling, shadow_offset_x, shadow_offset_y, is_shadow=True)

    @profile_method()
    def _render_trees(self, img: Image.Image, tree_entities, size: Dict, scaling: float, grid_view, image_resolver) -> None:
        """Render trees."""
        for tree in tree_entities:
            pos = tree['position']
            relative_x = pos['x'] + abs(size['minX'])
            relative_y = pos['y'] + abs(size['minY'])
            grid_view.set_center(pos['x'], pos['y'])
            renderer = renderer_manager.get_renderer(tree['name'])
            if renderer and hasattr(renderer, 'render'):
                tree_image = renderer.render(tree, grid_view, image_resolver)
                if tree_image:
                    self._paste_image(img, tree_image, relative_x, relative_y, scaling)

    @profile_method()
    def _render_entity_shadows(self, img: Image.Image, non_tree_entities, size: Dict, scaling: float, grid_view, image_resolver) -> None:
        """Render entity shadows."""
        for entity in non_tree_entities:
            entity = entity.model_dump() if hasattr(entity, 'model_dump') else entity
            pos = entity['position']
            relative_x = pos['x'] + abs(size['minX'])
            relative_y = pos['y'] + abs(size['minY'])
            grid_view.set_center(pos['x'], pos['y'])
            image = None
            if entity['name'] in RENDERERS:
                renderer = renderer_manager.get_renderer(entity['name'])
                if renderer and hasattr(renderer, 'render_shadow'):
                    if 'direction' in entity:
                        entity['direction'] = int(entity['direction'].value)
                    image = renderer.render_shadow(entity, grid_view, image_resolver)
            else:
                image = image_resolver(entity['name'], True)
            if image:
                if entity['name'] == 'character':
                    shadow_offset_x = 32
                    shadow_offset_y = 20
                    self._paste_image(img, image, relative_x, relative_y, scaling, shadow_offset_x, shadow_offset_y, is_shadow=True)
                else:
                    self._paste_image(img, image, relative_x, relative_y, scaling, is_shadow=True)

    @profile_method()
    def _render_visible_inventories(self, img: Image.Image, entities, size: Dict, scaling: float, grid_view, image_resolver) -> None:
        """Render entity shadows."""
        for entity in entities:
            entity = entity.model_dump() if hasattr(entity, 'model_dump') else entity
            pos = entity['position']
            relative_x = pos['x'] + abs(size['minX'])
            relative_y = pos['y'] + abs(size['minY'])
            grid_view.set_center(pos['x'], pos['y'])
            image = None
            if entity['name'] in RENDERERS:
                renderer = renderer_manager.get_renderer(entity['name'])
                if renderer and hasattr(renderer, 'render_inventory'):
                    image = renderer.render_inventory(entity, grid_view, image_resolver)
            if image:
                self._paste_image(img, image, relative_x, relative_y, scaling)

    @profile_method()
    def _render_rails(self, img: Image.Image, non_tree_entities, size: Dict, scaling: float, image_resolver) -> None:
        """Render rail entities with multiple passes."""
        passes = [1, 2, 3, 3.5, 4, 5]
        for pass_num in passes:
            for entity in non_tree_entities:
                entity = entity.model_dump() if hasattr(entity, 'model_dump') else entity
                if entity['name'] not in ['straight-rail', 'curved-rail', 'rail-signal', 'rail-chain-signal']:
                    continue
                pos = entity['position']
                relative_x = pos['x'] + abs(size['minX'])
                relative_y = pos['y'] + abs(size['minY'])
                direction = entity.get('direction', 0)
                image = None
                if entity.name == 'straight-rail':
                    if direction in [0, 4]:
                        image = image_resolver(f'{entity.name}_vertical_pass_{int(pass_num)}', False)
                    elif direction in [2, 6]:
                        image = image_resolver(f'{entity.name}_horizontal_pass_{int(pass_num)}', False)
                if image:
                    self._paste_image(img, image, relative_x, relative_y, scaling)

    @profile_method()
    def _render_entities(self, img: Image.Image, non_tree_entities, size: Dict, scaling: float, grid_view, image_resolver) -> None:
        """Render non-rail entities."""
        for entity in non_tree_entities:
            if entity.name in ['straight-rail', 'curved-rail']:
                continue
            pos = entity.position
            relative_x = pos.x + abs(size['minX'])
            relative_y = pos.y + abs(size['minY'])
            grid_view.set_center(pos.x, pos.y)
            image = None
            if entity.name in RENDERERS:
                renderer = renderer_manager.get_renderer(entity.name)
                if renderer and hasattr(renderer, 'render'):
                    entity_dict = entity.model_dump()
                    if 'direction' in entity_dict:
                        entity_dict['direction'] = int(entity_dict['direction'].value)
                    image = renderer.render(entity_dict, grid_view, image_resolver)
            else:
                image = image_resolver(entity.name, False)
            if image:
                self._paste_image(img, image, relative_x, relative_y, scaling)

    @profile_method()
    def _paste_image(self, img: Image.Image, sprite: Image.Image, relative_x: float, relative_y: float, scaling: float, offset_x: int=0, offset_y: int=0, is_shadow: bool=False) -> None:
        """Paste a sprite image onto the main image at the specified position.

        Args:
            img: Target image to paste onto
            sprite: Sprite image to paste
            relative_x: X position relative to origin
            relative_y: Y position relative to origin
            scaling: Current scaling factor
            offset_x: Additional X offset in pixels
            offset_y: Additional Y offset in pixels
            is_shadow: Whether this sprite is a shadow (will apply transparency)
        """
        scale_ratio = scaling / DEFAULT_SCALING
        if scale_ratio != 1.0:
            new_width = max(1, int(sprite.width * scale_ratio))
            new_height = max(1, int(sprite.height * scale_ratio))
            sprite = sprite.resize((new_width, new_height), Image.Resampling.LANCZOS)
            offset_x = int(offset_x * scale_ratio)
            offset_y = int(offset_y * scale_ratio)
        if is_shadow and SHADOW_INTENSITY < 1.0:
            sprite = sprite.copy()
            if sprite.mode != 'RGBA':
                sprite = sprite.convert('RGBA')
            pixels = sprite.load()
            for y in range(sprite.height):
                for x in range(sprite.width):
                    r, g, b, a = pixels[x, y]
                    pixels[x, y] = (r, g, b, int(a * SHADOW_INTENSITY))
        start_x = int(relative_x * scaling + scaling / 2 - sprite.width / 2) + offset_x
        start_y = int(relative_y * scaling + scaling / 2 - sprite.height / 2) + offset_y
        mask = sprite if sprite.mode == 'RGBA' else None
        img.paste(sprite, (start_x, start_y), mask)

def flatten_entities(entities: List[Union[Dict, Entity, EntityGroup]]) -> List[Union[Entity, EntityCore]]:
    max_direction = 0
    for entity in entities:
        if isinstance(entity, dict):
            if 'direction' not in entity:
                entity['direction'] = 0
            direction = entity['direction'] if 'direction' in entity else 0
            if direction > max_direction:
                max_direction = direction
    for entity in entities:
        if isinstance(entity, dict):
            try:
                entity['direction'] = entity['direction'] / 2 if max_direction > 6 else entity['direction']
                yield EntityCore(**entity)
            except Exception:
                pass
        elif isinstance(entity, EntityGroup):
            e_list = []
            if isinstance(entity, WallGroup):
                e_list = entity.entities
            elif isinstance(entity, BeltGroup):
                e_list = entity.belts
            elif isinstance(entity, PipeGroup):
                e_list = entity.pipes
            elif isinstance(entity, ElectricityGroup):
                e_list = entity.poles
            for e in e_list:
                yield e
        else:
            yield entity

def entities_to_grid(entities: List[Union[Dict, Entity]]) -> Dict:
    """Convert entity list to position grid."""
    grid = {}
    for entity in entities:
        if isinstance(entity, dict):
            x = entity['position']['x']
            y = entity['position']['y']
            if x not in grid:
                grid[x] = {}
            grid[x][y] = entity
        elif isinstance(entity, EntityGroup):
            e_list = []
            if isinstance(entity, WallGroup):
                e_list = entity.entities
            elif isinstance(entity, BeltGroup):
                e_list = entity.belts
            elif isinstance(entity, PipeGroup):
                e_list = entity.pipes
            elif isinstance(entity, ElectricityGroup):
                e_list = entity.poles
            for e in e_list:
                if e.position.x not in grid:
                    grid[e.position.x] = {}
                grid[e.position.x][e.position.y] = entity
        elif isinstance(entity, EntityCore):
            x = entity.position.x
            y = entity.position.y
            if x not in grid:
                grid[x] = {}
            grid[x][y] = entity
    return grid

def resources_to_grid(resources: List[Dict]) -> Dict:
    """Convert resource list to position grid."""
    grid = {}
    for resource in resources:
        x = resource['position']['x']
        y = resource['position']['y']
        if x not in grid:
            grid[x] = {}
        grid[x][y] = resource
    return grid

def build_available_trees_index(sprites_dir) -> Dict[str, Set[str]]:
    """
    Build an index of available tree sprites.

    Returns a dict mapping tree type to available variation-state combinations.
    Only includes variations that have exactly 10 files (complete set).
    e.g., {'03': {'a-full', 'a-medium', 'a-minimal', 'b-full', ...}}
    """
    from pathlib import Path
    file_counts = {}
    all_files = {}
    tree_pattern = re.compile('^(?:hr-)?tree-(\\d+)-([a-l])-(\\w+)\\.png$')
    sprites_dir = Path(sprites_dir)
    for sprite_file in sprites_dir.glob('*.png'):
        match = tree_pattern.match(sprite_file.name)
        if match:
            tree_type = match.group(1)
            variation = match.group(2)
            state = match.group(3)
            if state.endswith('-shadow'):
                continue
            key = (tree_type, variation)
            if key not in file_counts:
                file_counts[key] = 0
                all_files[key] = []
            file_counts[key] += 1
            all_files[key].append((state, sprite_file.name))
    available_trees = {}
    for (tree_type, variation), count in file_counts.items():
        if count == TREE_FILES_PER_VARIATION:
            if tree_type not in available_trees:
                available_trees[tree_type] = set()
            for state, _ in all_files[tree_type, variation]:
                available_trees[tree_type].add(f'{variation}-{state}')
        else:
            print(f'Skipping tree-{tree_type}-{variation}: has {count} files instead of {TREE_FILES_PER_VARIATION}')
    return available_trees

def profile_method(operation_name: str=None, include_args: bool=False):
    """Decorator to profile method execution time.

    Args:
        operation_name: Custom name for the operation (defaults to class.method)
        include_args: Whether to include method arguments in metadata
    """

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not profiler.is_enabled():
                return func(self, *args, **kwargs)
            op_name = operation_name or f'{self.__class__.__name__}.{func.__name__}'
            metadata = {}
            if include_args:
                metadata['arg_count'] = len(args)
                metadata['kwarg_count'] = len(kwargs)
            with profiler.timer(op_name, metadata):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

def find_fle_sprites_dir() -> Path:
    """Walk up the directory tree until we find .fle directory."""
    current = Path.cwd()
    while current != current.parent:
        fle_dir = current / '.fle'
        if fle_dir.exists() and fle_dir.is_dir():
            return fle_dir / 'sprites'
        current = current.parent
    return Path.cwd() / '.fle' / 'sprites'

def is_tree_entity(entity_name: str) -> bool:
    """Check if an entity is a tree."""
    return entity_name.startswith('tree-') or 'dead-tree' in entity_name or 'dry-tree' in entity_name or ('dead-grey-trunk' in entity_name)

def get_tree_variant(x: float, y: float, tree_type: str, available_trees: Dict[str, Set[str]]) -> Tuple[str, str]:
    """
    Calculate tree variant and foliage state based on position.
    Returns (variation_letter, foliage_state)

    Args:
        x, y: Position coordinates
        tree_type: Type of tree (e.g., '03', '04', '05')
        available_trees: Dict mapping tree_type to set of available variation-state combinations
                        e.g., {'03': {'a-full', 'a-medium', 'a-minimal', 'b-full', ...}}
    """
    seed = int((x * 113 + y * 157) * (x - y) * 31 + x * y * 73 + hash(tree_type) * 17)
    position_hash = int(x * 1000 % 97 + y * 1000 % 89 + (x + y) * 1000 % 83)
    seed = seed * 29 + position_hash
    variations = TREE_VARIATIONS
    available_for_type = available_trees.get(tree_type, set())
    valid_variations = []
    for var in variations:
        has_any_state = any((f'{var}-' in combo for combo in available_for_type))
        if has_any_state:
            valid_variations.append(var)
    if not valid_variations:
        valid_variations = ['a']
    variation_index = seed % len(valid_variations)
    variation = valid_variations[variation_index]
    foliage_weights = [('full', 70), ('medium', 25), ('minimal', 5), ('trunk_only', 0)]
    total_weight = sum((w for _, w in foliage_weights))
    choice = seed // len(variations) % total_weight
    cumulative = 0
    foliage_state = 'full'
    for state, weight in foliage_weights:
        cumulative += weight
        if choice < cumulative:
            foliage_state = state
            break
    if f'{variation}-{foliage_state}' not in available_for_type:
        for state in ['full', 'medium', 'minimal', 'trunk_only']:
            if f'{variation}-{state}' in available_for_type:
                foliage_state = state
                break
        else:
            for combo in available_for_type:
                if combo.startswith(f'{variation}-'):
                    foliage_state = combo.split('-', 1)[1]
                    break
    return (variation, foliage_state)

def is_rock_entity(entity_name: str) -> bool:
    return 'rock-' in entity_name

def get_resource_variant(x: float, y: float, max_variants: int=DEFAULT_RESOURCE_VARIANTS) -> int:
    """
    Calculate resource variant based on position using a hash-like function.
    Returns a variant number from 1 to max_variants.
    """
    hash_value = int(x * 7 + y * 13) % max_variants
    return hash_value + 1

def get_resource_volume(amount: int, max_amount: int=DEFAULT_MAX_RESOURCE_AMOUNT) -> int:
    """
    Calculate resource volume level (1-8) based on amount.
    8 = full, 1 = nearly empty
    """
    if amount <= 0:
        return MIN_RESOURCE_VOLUME
    percentage = min(amount / max_amount, 1.0)
    volume = max(MIN_RESOURCE_VOLUME, min(MAX_RESOURCE_VOLUME, int(percentage * MAX_RESOURCE_VOLUME)))
    return volume

class ImageResolver:
    """Resolve image paths and load images (simple PNG-based resolver)."""

    def __init__(self, images_dir: str='.fle/sprites'):
        """Initialize image resolver.

        Args:
            images_dir: Directory containing sprite images
        """
        self.images_dir = find_fle_sprites_dir()
        self.cache: Dict[str, Optional[Image.Image]] = {}

    @profile_method(include_args=True)
    def __call__(self, name: str, shadow: bool=False) -> Optional[Image.Image]:
        """Load and cache an image.

        Args:
            name: Name of the sprite (without extension)
            shadow: Whether to load shadow variant

        Returns:
            PIL Image if found, None otherwise
        """
        filename = f'{name}_shadow' if shadow else name
        if filename in self.cache and self.cache[filename]:
            profiler.increment_counter('image_cache_hits')
            return self.cache[filename]
        profiler.increment_counter('image_cache_misses')
        path = self.images_dir / f'{filename}.png'
        if not path.exists():
            self.cache[filename] = None
            profiler.increment_counter('image_not_found')
            return None
        try:
            with profiler.timer('image_load_from_disk'):
                image = Image.open(path).convert('RGBA')
            self.cache[filename] = image
            profiler.increment_counter('images_loaded')
            return image
        except Exception:
            self.cache[filename] = None
            profiler.increment_counter('image_load_errors')
            return None

class Render(Tool):

    def __init__(self, *args):
        super().__init__(*args)
        self.image_resolver = ImageResolver('.fle/sprites')
        self.decoder = Decoder()
        self.get_entities = GetEntities(*args)

    @profile_method(include_args=True)
    def _get_map_entities(self, include_status, radius, compression_level):
        try:
            result, _ = self.execute(self.player_index, include_status, radius, compression_level)
            decoded_result = self._decode_optimized_format(result)
            return decoded_result
        except Exception:
            result, _ = self.execute(self.player_index, include_status, radius, compression_level)
            pass

    @profile_method(include_args=True)
    def __call__(self, include_status: bool=False, radius: int=64, position: Optional[Position]=None, layers: Layer=Layer.ALL, compression_level: str='binary', blueprint: Union[str, List[Dict]]=None, return_renderer=False, max_render_radius: Optional[float]=None) -> Union[RenderedImage, Tuple[RenderedImage, Renderer]]:
        """
        Returns information about all entities, tiles, and resources within the specified radius of the player.

        Args:
            include_status: Whether to include status information for entities (optional)
            radius: Search radius around the player (default: 50)
            position: Center position for the search (optional, defaults to player position)
            layers: Which layers to include in the render
            compression_level: Compression level to use ('none', 'standard', 'binary', 'maximum')
                - 'none': No compression, raw data
                - 'standard': Run-length encoding for water, patch-based for resources (default)
                - 'binary': Binary encoding with base64 transport
                - 'maximum': Same as binary, reserved for future improvements
            blueprint: Either a Base64 encoded blueprint, or a decoded blueprint
            return_renderer: Whether to return the renderer, which contains the entities that were rendered

        Returns:
            RenderedImage containing the visual representation of the area
        """
        assert isinstance(include_status, bool), 'Include status must be boolean'
        assert isinstance(radius, (int, float)), 'Radius must be a number'
        if not blueprint:
            renderer = self.get_renderer_from_map(include_status, radius, compression_level, max_render_radius)
        else:
            renderer = self.get_renderer_from_blueprint(blueprint)
        size = renderer.get_size()
        if size['width'] == 0 or size['height'] == 0:
            raise Exception('Nothing to render.')
        width = size['width'] * DEFAULT_SCALING
        height = size['height'] * DEFAULT_SCALING
        max_dimension = 1024
        if width > max_dimension or height > max_dimension:
            aspect_ratio = width / height
            if width > height:
                width = max_dimension
                height = int(max_dimension / aspect_ratio)
            else:
                height = max_dimension
                width = int(max_dimension * aspect_ratio)
        width = max(1, width)
        height = max(1, height)
        image = renderer.render(width, height, self.image_resolver)
        if return_renderer:
            return (RenderedImage(image), renderer)
        else:
            return RenderedImage(image)

    def get_renderer_from_blueprint(self, blueprint):
        if isinstance(blueprint, str):
            raise NotImplementedError()
        else:
            if 'entities' not in blueprint:
                raise ValueError('Blueprint passed with no entities')
            entities = blueprint['entities']
            renderer = Renderer(entities=entities)
        return renderer

    def get_renderer_from_map(self, include_status: bool=False, radius: int=64, compression_level: str='binary', max_render_radius: Optional[float]=None) -> Renderer:
        result = self._get_map_entities(include_status, radius, compression_level)
        entities = self.parse_lua_dict(result['entities'])
        character_position = [c['position'] for c in list(filter(lambda x: x['name'] == 'character', entities))]
        char_pos = Position(character_position[0]['x'], character_position[0]['y'])
        ent = self.get_entities(position=char_pos, radius=radius)
        if ent:
            entities.extend(ent)
            pass
        water_tiles = result['water_tiles']
        resources = result['resources']
        renderer = Renderer(entities=entities, water_tiles=water_tiles, resources=resources, max_render_radius=max_render_radius)
        return renderer

    def _decode_optimized_format(self, result: Dict) -> Dict:
        """
        Decode the optimized format based on the version.

        Args:
            result: The raw result from the Lua execution

        Returns:
            Dictionary with decoded entities, water_tiles, and resources
        """
        meta = result.get('meta', {})
        format_version = meta.get('format', 'v1')
        if format_version == 'v2-binary':
            entities = result.get('entities', [])
            water_tiles = []
            if 'water_binary' in result:
                water_binary = self.decoder.decode_base64_urlsafe(result['water_binary'])
                water_runs = self.decoder.decode_water_binary(water_binary)
                water_tiles = self.decoder.decode_water_runs(water_runs)
            resources = []
            if 'resources_binary' in result:
                resources_binary = self.decoder.decode_base64_urlsafe(result['resources_binary'])
                resource_patches = self.decoder.decode_resources_binary(resources_binary)
                resources = self.decoder.decode_resource_patches(resource_patches)
            return {'entities': entities, 'water_tiles': water_tiles, 'resources': resources}
        elif format_version == 'v2':
            entities = result.get('entities', [])
            water_runs = result.get('water', [])
            resource_patches = result.get('resources', {})
            water_tiles = self.decoder.decode_water_runs(water_runs)
            resources = self.decoder.decode_resource_patches(resource_patches)
            return {'entities': entities, 'water_tiles': water_tiles, 'resources': resources}
        else:
            return {'entities': result.get('entities', []), 'water_tiles': result.get('water_tiles', []), 'resources': result.get('resources', [])}

