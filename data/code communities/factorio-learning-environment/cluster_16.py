# Cluster 16

def fle_eval(args):
    """Run evaluation/experiments with the given config."""
    try:
        from fle.eval.entrypoints.gym_eval import main as run_eval
        config_path = str(Path(args.config))
        asyncio.run(run_eval(config_path))
    except ImportError as e:
        print('Error: Evaluation functionality requires additional dependencies.', file=sys.stderr)
        print('Install with: pip install factorio-learning-environment[eval]', file=sys.stderr)
        print(f'Original error: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

def fle_cluster(args):
    """Handle cluster management commands."""
    if args.cluster_command == 'start':
        if not 1 <= args.number <= 33:
            print('Error: number of instances must be between 1 and 33.')
            sys.exit(1)
        if args.use_save and (not Path(args.use_save).exists()):
            print(f"Error: Save file '{args.use_save}' does not exist.")
            sys.exit(1)
        start_cluster(args.number, args.scenario, args.attach_mods, args.use_save)
    elif args.cluster_command == 'stop':
        stop_cluster()
    elif args.cluster_command == 'restart':
        restart_cluster()
    elif args.cluster_command == 'logs':
        manager = ClusterManager()
        manager.logs(getattr(args, 'service', 'factorio_0'))
    elif args.cluster_command == 'show':
        manager = ClusterManager()
        manager.show()
    else:
        print(f"Error: Unknown cluster command '{args.cluster_command}'")

def start_cluster(num_instances, scenario, attach_mod=False, save_file=None):
    manager = ClusterManager()
    manager.start(num_instances=num_instances, scenario=scenario, attach_mod=attach_mod, save_file=save_file)

def stop_cluster():
    manager = ClusterManager()
    manager.stop()

def restart_cluster():
    manager = ClusterManager()
    manager.restart()

def fle_sprites(args):
    try:
        print('Downloading spritemaps...')
        success = download_sprites_from_hf(output_dir=args.spritemap_dir, force=args.force, num_workers=args.workers)
        if not success:
            print('Failed to download spritemaps', file=sys.stderr)
            sys.exit(1)
        print('\nGenerating sprites...')
        success = generate_sprites(input_dir=args.spritemap_dir, output_dir=args.sprite_dir)
        if not success:
            print('Failed to generate sprites', file=sys.stderr)
            sys.exit(1)
        print('\nSprites successfully downloaded and generated!')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

def download_sprites_from_hf(repo_id: str='Noddybear/fle_images', output_dir: str='.fle/spritemaps', force: bool=False, num_workers: int=10, use_snapshot: bool=True, archive_name: Optional[str]=None) -> bool:
    """
    Optimized sprite download with multiple strategies

    Args:
        repo_id: Hugging Face dataset repository ID
        output_dir: Directory to save sprites
        force: Force re-download even if files exist
        num_workers: Number of parallel download workers
        use_snapshot: Use snapshot_download for faster bulk download
        archive_name: If sprites are in a single archive file, specify its name

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)
    if output_path.exists() and (not force):
        if any(output_path.iterdir()):
            print(f'Sprites already exist in {output_path}. Use --force to re-download.')
            return True
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        if archive_name or check_for_archive(repo_id):
            return download_archive_strategy(repo_id, output_path, archive_name)
        if use_snapshot:
            return download_snapshot_strategy(repo_id, output_path)
        return download_parallel_strategy(repo_id, output_path, num_workers)
    except Exception as e:
        print(f'Error downloading sprites: {e}')
        return False

def generate_sprites(input_dir: str='.fle/spritemaps', output_dir: str='.fle/sprites'):
    """
    Generate individual sprites from spritemaps

    Args:
        input_dir: Directory containing downloaded spritemaps
        output_dir: Directory to save extracted sprites
        data_path: Path to data.json file (optional)
    """
    import sys
    from pathlib import Path
    sprites_module_path = Path(__file__).parent.parent / 'data' / 'sprites'
    if sprites_module_path.exists():
        sys.path.insert(0, str(sprites_module_path))
    try:
        from fle.agents.data.sprites.extractors.entities import EntitySpritesheetExtractor
        from fle.agents.data.sprites.extractors.resources import ResourceSpriteExtractor
        from fle.agents.data.sprites.extractors.terrain import TerrainSpriteExtractor
        from fle.agents.data.sprites.extractors.trees import TreeSpriteExtractor
    except ImportError:
        print('Error: Could not import extractor modules.')
        print('Make sure the extractor modules are in the correct location.')
        return False
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.exists():
        print(f"Input directory {input_path} does not exist. Run 'fle sprites download' first.")
        return False
    print(f'Generating sprites from {input_path} to {output_path}...')
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        if (input_path / 'data.json').exists():
            entities = EntitySpritesheetExtractor(str(input_path), str(output_path))
            entities.extract_all()
        base_graphics = input_path / '__base__' / 'graphics'
        if base_graphics.exists():
            resources_path = base_graphics / 'resources'
            if resources_path.exists():
                resources = ResourceSpriteExtractor(str(resources_path), str(output_path))
                resources.extract_all_resources()
                resources.create_all_icons()
                trees = TreeSpriteExtractor(str(resources_path), str(output_path))
                trees.extract_all_trees()
            terrain_path = base_graphics / 'terrain'
            if terrain_path.exists():
                terrain = TerrainSpriteExtractor(str(terrain_path), str(output_path))
                terrain.extract_all_resources()
                terrain.create_all_icons()
            icons_path = base_graphics / 'icons'
            if icons_path.exists():
                icon = IconSpriteExtractor(str(icons_path), str(output_path))
                icon.extract_all_icons()
            alerts_path = icons_path / 'alerts'
            if alerts_path.exists():
                icon = AlertSpriteExtractor(str(alerts_path), str(output_path))
                icon.extract_all_alerts()
        else:
            print('No __base__/graphics structure found, copying PNG files directly...')
            png_files = list(input_path.rglob('*.png'))
            if not png_files:
                print('No PNG files found in spritemaps directory.')
                return False
            from tqdm import tqdm
            import shutil
            for png_file in tqdm(png_files, desc='Copying sprites'):
                rel_path = png_file.relative_to(input_path)
                dest_path = output_path / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(png_file, dest_path)
        print(f'Successfully generated sprites in {output_path}')
        return True
    except Exception as e:
        print(f'Error generating sprites: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main CLI entry point for FLE."""
    parser = argparse.ArgumentParser(prog='fle', description='Factorio Learning Environment CLI - Manage clusters and run experiments', formatter_class=argparse.RawDescriptionHelpFormatter, epilog="\nExamples:\n  fle init                                   # Initialize environment\n  fle cluster start                          # Start 1 Factorio instance\n  fle cluster start -n 4                     # Start 4 instances  \n  fle cluster start -s open_world            # Start with open world scenario\n  fle cluster stop                           # Stop all instances\n  fle cluster show                           # Show running services\n  fle cluster logs factorio_0                # View logs for specific service\n  fle cluster restart                        # Restart current cluster\n  fle eval --config configs/run_config.json  # Run experiment\n  fle eval --config configs/gym_run_config.json\n  fle cluster [start|stop|restart|help] [-n N] [-s SCENARIO]\n  fle sprites [--force] [--workers N]\n\nTips:\n  Use 'fle <command> -h' for command-specific help\n  Use 'fle cluster <subcommand> -h' for cluster subcommand help\n        ")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.add_parser('init', help='Initialize FLE environment (.env file)')
    cluster_parser = subparsers.add_parser('cluster', help='Manage Factorio server clusters', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  fle cluster start                    # Start 1 instance (default scenario)\n  fle cluster start -n 4               # Start 4 instances\n  fle cluster start -s open_world      # Start with open world scenario\n  fle cluster start -sv save.zip       # Start from a save file\n  fle cluster start -m                 # Start with mods attached\n  fle cluster stop                     # Stop all running instances\n  fle cluster restart                  # Restart current cluster\n  fle cluster show                     # Show running services and ports\n  fle cluster logs factorio_0          # Show logs for specific service\n        ')
    cluster_subparsers = cluster_parser.add_subparsers(dest='cluster_command', help='Cluster management commands')
    start_parser = cluster_subparsers.add_parser('start', help='Start Factorio instances')
    start_parser.add_argument('-n', '--number', type=int, default=1, help='Number of Factorio instances to run (1-33, default: 1)')
    start_parser.add_argument('-s', '--scenario', choices=['open_world', 'default_lab_scenario'], default='default_lab_scenario', help='Scenario to run (default: default_lab_scenario)')
    start_parser.add_argument('-sv', '--use_save', type=str, help='Use a .zip save file from Factorio')
    start_parser.add_argument('-m', '--attach_mods', action='store_true', help='Attach mods to the instances')
    cluster_subparsers.add_parser('stop', help='Stop all running instances')
    cluster_subparsers.add_parser('restart', help='Restart the current cluster')
    logs_parser = cluster_subparsers.add_parser('logs', help='Show service logs')
    logs_parser.add_argument('service', nargs='?', default='factorio_0', help='Service name (default: factorio_0)')
    cluster_subparsers.add_parser('show', help='Show running services and exposed ports')
    eval_parser = subparsers.add_parser('eval', help='Run experiments/evaluation')
    eval_parser.add_argument('--config', required=True, help='Path to run config JSON file')
    sprites_parser = subparsers.add_parser('sprites', help='Download and generate sprites')
    sprites_parser.add_argument('--force', action='store_true', help='Force re-download even if sprites exist')
    sprites_parser.add_argument('--workers', type=int, default=10, help='Number of parallel download workers (default: 10)')
    sprites_parser.add_argument('--spritemap-dir', type=str, default='.fle/spritemaps', help='Directory to save downloaded spritemaps (default: .fle/spritemaps)')
    sprites_parser.add_argument('--sprite-dir', type=str, default='.fle/sprites', help='Directory to save generated sprites (default: .fle/sprites)')
    args = parser.parse_args()
    if args.command == 'init':
        fle_init()
    elif args.command == 'cluster':
        if not args.cluster_command:
            cluster_parser.print_help()
            sys.exit(1)
        fle_cluster(args)
    elif args.command == 'eval':
        fle_init()
        fle_eval(args)
    elif args.command == 'sprites':
        fle_sprites(args)
    elif args.command is None:
        parser.print_help()
        sys.exit(1)
    else:
        print(f"Error: Unknown command '{args.command}'")
        parser.print_help()
        sys.exit(1)

def fle_init():
    """Initialize FLE environment by creating .env file and configs directory if they don't exist."""
    created_files = []
    if not Path('.env').exists():
        try:
            pkg = importlib.resources.files('fle')
            env_path = pkg / '.example.env'
            shutil.copy(str(env_path), '.env')
            created_files.append('.env file')
        except Exception as e:
            print(f'Error creating .env file: {e}', file=sys.stderr)
            sys.exit(1)
    configs_dir = Path('configs')
    if not configs_dir.exists():
        try:
            configs_dir.mkdir(exist_ok=True)
            pkg = importlib.resources.files('fle')
            config_path = pkg / 'eval' / 'configs' / 'gym_run_config.json'
            shutil.copy(str(config_path), configs_dir / 'gym_run_config.json')
            created_files.append('configs/ directory with gym_run_config.json')
        except Exception as e:
            print(f'Error creating configs directory: {e}', file=sys.stderr)
            sys.exit(1)
    if created_files:
        print(f'Created {', '.join(created_files)} - please edit .env with your API keys and DB config')

def main():
    """Main entry point"""
    base = Path.cwd()
    if base.name == 'sprites' or base.name == 'data':
        project_root = base.parent.parent.parent.parent.parent
        if base.name == 'sprites':
            project_root = base.parent.parent.parent.parent
    else:
        project_root = base.parent.parent.parent.parent.parent
    base_input_path = project_root / '.fle' / 'spritemaps' / '__base__' / 'graphics'
    resources_path = base_input_path / 'resources'
    terrain_path = base_input_path / 'terrain'
    decoratives_path = base_input_path / 'decorative'
    icons_path = base_input_path / 'icons'
    alerts_path = icons_path / 'alerts'
    entities_path = project_root / '.fle' / 'spritemaps'
    output_dir = project_root / '.fle' / 'sprites'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Project root: {project_root}')
    print(f'Input path: {entities_path}')
    print(f'Output path: {output_dir}')
    if not entities_path.exists():
        print(f'Error: Input directory does not exist: {entities_path}')
        print("Run 'fle sprites download' first to download the spritemaps.")
        return
    if icons_path.exists():
        print('\n=== Extracting Icon Sprites ===')
        icon = IconSpriteExtractor(str(icons_path), str(output_dir))
        icon.extract_all_icons()
    else:
        print(f'Warning: Icon path not found: {icons_path}')
    if alerts_path.exists():
        print('\n=== Extracting Alert Sprites ===')
        alerts = AlertSpriteExtractor(str(alerts_path), str(output_dir))
        alerts.extract_all_alerts()
    else:
        print(f'Warning: Alerts path not found: {alerts_path}')
    if decoratives_path.exists():
        print('\n=== Extracting Decorative Sprites ===')
        decoratives = DecorativeSpriteExtractor(str(decoratives_path), str(output_dir))
        decoratives.extract_all_decoratives()
    else:
        print(f'Warning: Decoratives path not found: {decoratives_path}')
    if (entities_path / 'data.json').exists():
        print('\n=== Extracting Entity Sprites ===')
        entities = EntitySpritesheetExtractor(str(entities_path), str(output_dir))
        entities.extract_all()
    else:
        print('Warning: data.json not found, skipping entity extraction')
    if resources_path.exists():
        print('\n=== Extracting Resource Sprites ===')
        resources = ResourceSpriteExtractor(str(resources_path), str(output_dir))
        resources.extract_all_resources()
        resources.create_all_icons()
    else:
        print(f'Warning: Resources path not found: {resources_path}')
    if resources_path.exists():
        print('\n=== Extracting Tree Sprites ===')
        trees = TreeSpriteExtractor(str(resources_path), str(output_dir))
        trees.extract_all_trees()
    if terrain_path.exists():
        print('\n=== Extracting Terrain Sprites ===')
        terrain = TerrainSpriteExtractor(str(terrain_path), str(output_dir))
        terrain.extract_all_resources()
        terrain.create_all_icons()
    else:
        print(f'Warning: Terrain path not found: {terrain_path}')
    character_path = base_input_path.parent / 'character'
    if character_path.exists():
        print('\n=== Extracting Character Sprites ===')
        character = CharacterSpriteExtractor(str(character_path), str(output_dir))
        character.extract_all_character_sprites()
        character.extract_single_sprites()
        character.create_character_mapping()
    else:
        print(f'Warning: Character path not found: {character_path}')
    print('\n=== Extraction Complete ===')

