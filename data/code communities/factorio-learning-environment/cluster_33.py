# Cluster 33

def render_blueprints_from_directory(blueprints_dir: str, output_dir: str=None, show_images: bool=True):
    """
    Render all JSON blueprint files from a directory

    Args:
        blueprints_dir: Directory containing JSON blueprint files
        output_dir: Optional directory to save rendered images (default: blueprints_dir/rendered)
        show_images: Whether to display images on screen (default: True)
    """
    blueprints_path = Path(blueprints_dir)
    if not blueprints_path.exists():
        print(f'Error: Blueprint directory not found: {blueprints_dir}')
        return
    if output_dir is None:
        output_path = blueprints_path / 'rendered'
    else:
        output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    sprites_dir = Path('.fle/sprites')
    try:
        print('Using BasisImageResolver for .basis file support')
        image_resolver = ImageResolver('.fle/sprites')
    except ImportError:
        print('BasisImageResolver not found, using simple PNG resolver')
        print("Place PNG files in 'images' directory")
        image_resolver = ImageResolver(str(sprites_dir))
    json_files = list(blueprints_path.glob('*.json'))
    if not json_files:
        print(f'No JSON files found in {blueprints_dir}')
        return
    print(f'Found {len(json_files)} blueprint files to render')
    successful = 0
    failed = 0
    for json_file in json_files[:3]:
        try:
            print(f'\nProcessing: {json_file.name}')
            with open(json_file, 'r') as f:
                blueprint_data = json.load(f)
            if 'blueprint' in blueprint_data:
                blueprint_content = blueprint_data['blueprint']
            elif 'entities' in blueprint_data:
                blueprint_content = blueprint_data
            elif isinstance(blueprint_data, str):
                parsed = parse_blueprint(blueprint_data)
                blueprint_content = parsed.get('blueprint', parsed)
            else:
                print(f'  Warning: Unknown blueprint format in {json_file.name}')
                failed += 1
                continue
            blueprint = Renderer(sprites_dir, entities=blueprint_content['entities'])
            size = blueprint.get_size()
            if size['width'] == 0 or size['height'] == 0:
                print('  Warning: Blueprint has no entities to render')
                failed += 1
                continue
            scaling = 32
            width = min((size['width'] + 2) * scaling, 2048)
            height = min((size['height'] + 2) * scaling, 2048)
            image = blueprint.render(width, height, image_resolver)
            output_filename = json_file.stem + '.png'
            output_file = output_path / output_filename
            image.save(output_file)
            print(f'  Rendered to: {output_file} ({width}x{height})')
            if show_images:
                image.show()
            successful += 1
        except Exception as e:
            print(f'  Error processing {json_file.name}: {str(e)}')
            failed += 1
    print('\n=== Summary ===')
    print(f'Successfully rendered: {successful}')
    print(f'Failed: {failed}')
    print(f'Output directory: {output_path}')

def parse_blueprint(blueprint_string: str) -> Dict:
    """Parse blueprint string to JSON."""
    decoded = base64.b64decode(blueprint_string[1:])
    unzipped = zlib.decompress(decoded)
    return json.loads(unzipped)

def main():
    """Main function to render all blueprints in the specified directory"""
    blueprints_dir = '/Users/jackhopkins/PycharmProjects/PaperclipMaximiser/fle/agents/data/blueprints_to_policies/blueprints/other'
    render_blueprints_from_directory(blueprints_dir=blueprints_dir, output_dir=None, show_images=True)

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

