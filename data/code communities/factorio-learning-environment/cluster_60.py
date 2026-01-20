# Cluster 60

class Renderer:
    """
    Main renderer class for Factorio entities that composes all rendering components
    """

    def __init__(self, style: Optional[Dict]=None):
        """Initialize renderer with optional custom style"""
        self.config = RenderConfig(style)
        self.categorizer = EntityCategoriser()
        self.color_manager = ColourManager(self.config, self.categorizer)
        self.shape_renderer = ShapeRenderer(self.config)
        self.connection_renderer = ConnectionRenderer(self.color_manager)
        self.legend_renderer = LegendRenderer(self.config, self.color_manager, self.categorizer, self.shape_renderer)
        self.image_calculator = ImageCalculator(self.config)
        self.layer_renderers = {Layer.GRID: GridLayerRenderer(self.config), Layer.WATER: WaterLayerRenderer(self.config), Layer.RESOURCES: ResourcesLayerRenderer(self.config), Layer.NATURAL: NaturalLayerRenderer(self.config), Layer.ENTITIES: EntitiesLayerRenderer(self.config, self.categorizer, self.color_manager, self.shape_renderer), Layer.CONNECTIONS: ConnectionsLayerRenderer(self.config, self.color_manager, self.connection_renderer), Layer.PLAYER | Layer.ORIGIN: MarkersLayerRenderer(self.config, self.shape_renderer), Layer.ELECTRICITY: ElectricityLayerRenderer(self.config)}

    def render_entities(self, entities: List[Entity], center_pos: Optional[Position]=None, bounding_box: Optional[BoundingBox]=None, water_tiles: Optional[List[Dict]]=None, resource_entities: Optional[List[Dict]]=None, trees: Optional[List[Dict]]=None, rocks: Optional[List[Dict]]=None, electricity_networks: Optional[List[Dict]]=None, max_tiles: int=20, layers: Layer=Layer.ALL) -> Image.Image:
        """
        Render a list of Factorio entities to an image

        Args:
            entities: List of entities to render
            center_pos: Optional center position (e.g. player position)
            bounding_box: Optional bounding box to constrain the render area
            water_tiles: Optional list of water tiles to render
            resource_entities: Optional list of resource entities to render
            trees: Optional list of trees to render
            rocks: Optional list of rocks to render
            electricity_networks: Optional list of electricity network data to render
            max_tiles: Maximum number of tiles on each side of the center position
            layers: Layer flags to specify which elements to render

        Returns:
            PIL Image containing the rendered map
        """
        resources_present = set()
        natural_elements_present = set()
        statuses_present = set()
        network_colors = {}
        if Layer.WATER in layers and water_tiles and (len(water_tiles) > 0):
            resources_present.add('water')
        if Layer.RESOURCES in layers and resource_entities:
            for resource in resource_entities:
                if 'name' in resource:
                    resources_present.add(resource['name'])
        if Layer.TREES in layers and trees and (len(trees) > 0):
            natural_elements_present.add('tree')
        if Layer.ROCKS in layers and rocks and (len(rocks) > 0):
            natural_elements_present.add('rock')
        if Layer.ELECTRICITY in layers and electricity_networks:
            electricity_renderer = self.layer_renderers.get(Layer.ELECTRICITY)
            if electricity_renderer:
                network_ids = set()
                for network in electricity_networks:
                    if 'network_id' in network:
                        network_ids.add(network['network_id'])
                electricity_renderer._assign_network_colors(network_ids, network_colors)
        self.color_manager.assign_entity_colors(entities)
        boundaries = self.image_calculator.calculate_boundaries(entities, center_pos, bounding_box, max_tiles=max_tiles)
        filtered_entities = []
        for entity in entities:
            pos = entity.position
            if pos.x >= boundaries['min_x'] and pos.x <= boundaries['max_x'] and (pos.y >= boundaries['min_y']) and (pos.y <= boundaries['max_y']):
                filtered_entities.append(entity)
                if self.config.style['status_indicator_enabled'] and entity.status != EntityStatus.NORMAL:
                    statuses_present.add(entity.status)
        entities = filtered_entities
        self.config.style['legend_position'] = 'right_top'
        legend_dimensions = None
        if self.config.style['legend_enabled'] and (self.color_manager.entity_colors or resources_present or natural_elements_present or statuses_present or network_colors):
            BASE_CELL_SIZE = 20
            tmp_width = int((boundaries['max_x'] - boundaries['min_x']) * BASE_CELL_SIZE + 2 * self.config.style['margin'])
            tmp_height = int((boundaries['max_y'] - boundaries['min_y']) * BASE_CELL_SIZE + 2 * self.config.style['margin'])
            legend_dimensions = self.legend_renderer.calculate_legend_dimensions(tmp_width, tmp_height, resources_present, natural_elements_present, statuses_present, network_colors)
        dimensions = self.image_calculator.calculate_image_dimensions(legend_dimensions)
        img_width = dimensions['img_width']
        img_height = dimensions['img_height']
        img = Image.new('RGBA', (img_width, img_height), self.config.style['background_color'])
        draw = ImageDraw.Draw(img)
        game_to_img = self.image_calculator.get_game_to_image_coordinate_function()
        font = self._load_font()
        legend_font = self._load_legend_font()
        render_order = [Layer.WATER, Layer.GRID, Layer.RESOURCES, Layer.ROCKS, Layer.TREES, Layer.ELECTRICITY, Layer.ENTITIES, Layer.CONNECTIONS, Layer.ORIGIN, Layer.PLAYER]
        render_kwargs = {'entities': entities, 'water_tiles': water_tiles, 'resource_entities': resource_entities, 'trees': trees, 'rocks': rocks, 'electricity_networks': electricity_networks, 'center_pos': center_pos, 'font': font, 'layers': layers}
        for layer_type in render_order:
            if layer_type in layers:
                for renderer_key, renderer in self.layer_renderers.items():
                    if layer_type in renderer_key:
                        renderer.render(draw, game_to_img, boundaries, **render_kwargs)
                        break
        self.legend_renderer.draw_combined_legend(draw, img_width, img_height, legend_font, resources_present, natural_elements_present, statuses_present, network_colors)
        return img

    def _load_font(self) -> ImageFont.ImageFont:
        """Load a font for text rendering with fallbacks"""
        try:
            font = ImageFont.truetype('arial.ttf', size=10)
        except IOError:
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', size=10)
            except IOError:
                font = ImageFont.load_default()
        return font

    def _load_legend_font(self) -> ImageFont.ImageFont:
        """Load a font specifically for the legend with a consistent size"""
        legend_font_size = self.config.style.get('legend_font_size', 10)
        try:
            font = ImageFont.truetype('arial.ttf', size=legend_font_size)
        except IOError:
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', size=legend_font_size)
            except IOError:
                font = ImageFont.load_default()
        return font

