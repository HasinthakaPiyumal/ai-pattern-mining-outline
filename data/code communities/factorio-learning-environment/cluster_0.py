# Cluster 0

def png_to_mp4(input_dir, output_file='output.mp4', framerate=24):
    """
    Convert numbered PNG files from specified directory to MP4.
    Creates a temporary directory with renamed files to ensure continuous sequence.
    """
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise ValueError(f'Directory not found: {input_dir}')
    png_files = list(input_path.glob('*.png'))
    if not png_files:
        raise ValueError(f'No PNG files found in {input_dir}')
    png_files.sort(key=lambda x: int(x.stem))
    temp_dir = input_path / 'temp_sequence'
    temp_dir.mkdir(exist_ok=True)
    try:
        for idx, file in enumerate(png_files, start=1):
            shutil.copy2(file, temp_dir / f'{idx:06d}.png')
        ffmpeg_cmd = ['ffmpeg', '-framerate', str(framerate), '-i', str(temp_dir / '%06d.png'), '-vf', 'scale=1920:1028:force_original_aspect_ratio=decrease,pad=1920:1028:(ow-iw)/2:(oh-ih)/2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(output_file)]
        subprocess.run(ffmpeg_cmd, check=True)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def get_program_state(conn, program_id: int):
    """Fetch a single program's full state by ID"""
    query = '\n    SELECT * FROM programs WHERE id = %s\n    '
    with conn.cursor() as cur:
        cur.execute(query, (program_id,))
        row = cur.fetchone()
        if not row:
            return None
        col_names = [desc[0] for desc in cur.description]
        return Program.from_row(dict(zip(col_names, row)))

def capture_screenshots_with_hooks(program_ids, output_dir: str, script_output_path: str, instance: FactorioInstance, conn, max_steps=1000):
    """
    Capture screenshots for each program state and after each entity placement,
    using sequential integer filenames.
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def get_highest_screenshot_number():
        existing_files = list(output_path.glob('*.png'))
        if not existing_files:
            return -1
        highest = -1
        for file in existing_files:
            try:
                num = int(file.stem)
                highest = max(highest, num)
            except ValueError:
                continue
        return highest
    screenshot_counter = get_highest_screenshot_number() + 1
    print(f'Starting screenshot numbering from {screenshot_counter}')
    instance.rcon_client.send_command('/c global.camera = nil')

    def capture_after_placement(tool_instance, result):
        nonlocal screenshot_counter
        screenshot_filename = f'{screenshot_counter:06d}.png'
        screenshot_path = str(output_path / screenshot_filename)
        instance.screenshot(script_output_path=script_output_path, save_path=screenshot_path, resolution='1920x1080', center_on_factory=True)
        print(f'Captured placement screenshot: {screenshot_filename}')
        screenshot_counter += 1
    for tool in ['place_entity', 'place_entity_next_to', 'connect_entities', 'harvest_resource', 'move_to', 'rotate_entity', 'shift_entity']:
        instance.register_post_tool_hook(tool, capture_after_placement)
    for idx, (program_id, created_at) in enumerate(program_ids):
        if idx >= max_steps:
            break
        program = get_program_state(conn, program_id)
        if not program or not program.state:
            print(f'Skipping program {program_id} - no state available')
            continue
        instance.reset(program.state)
        instance.eval(program.code)
        screenshot_filename = f'{screenshot_counter:06d}.png'
        screenshot_path = output_path / screenshot_filename
        instance.screenshot(script_output_path=script_output_path, save_path=str(screenshot_path), resolution='1920x1080', center_on_factory=True)
        print(f'Captured final program screenshot: {screenshot_filename}')
        screenshot_counter += 1
    for i in range(30):
        instance.eval('sleep(15)')
        screenshot_filename = f'{screenshot_counter:06d}.png'
        screenshot_path = output_path / screenshot_filename
        instance.screenshot(script_output_path=script_output_path, save_path=str(screenshot_path), resolution='1920x1080', center_on_factory=True)
        print(f'Captured final program screenshot: {screenshot_filename}')
        screenshot_counter += 1

def get_highest_screenshot_number():
    existing_files = list(output_path.glob('*.png'))
    if not existing_files:
        return -1
    highest = -1
    for file in existing_files:
        try:
            num = int(file.stem)
            highest = max(highest, num)
        except ValueError:
            continue
    return highest

def capture_screenshots(program_ids, output_dir: str, script_output_path: str, instance: FactorioInstance, conn, max_steps=1000):
    """Capture screenshots for each program state, skipping existing ones"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    existing_screenshots = get_existing_screenshots(output_path)
    total_needed = len(program_ids)
    existing_count = len(existing_screenshots)
    print(f'Found {existing_count} existing screenshots out of {total_needed} needed')
    instance.rcon_client.send_command('/c global.camera = nil')
    for idx, (program_id, created_at) in enumerate(program_ids):
        if idx in existing_screenshots:
            print(f'Skipping existing screenshot {idx + 1}/{total_needed}')
            continue
        if idx > max_steps:
            continue
        program = get_program_state(conn, program_id)
        if not program or not program.state:
            print(f'Skipping program {program_id} - no state available')
            continue
        instance.reset(program.state)
        instance.eval(program.code)
        screenshot_path = str(output_path / f'{idx:06d}.png')
        instance.screenshot(script_output_path=script_output_path, save_path=screenshot_path, resolution='1920x1080', center_on_factory=True)
        print(f'Captured screenshot {idx + 1}/{total_needed}')

def get_existing_screenshots(output_dir: Path) -> set:
    """Get a set of indices for screenshots that already exist"""
    existing = set()
    for file in output_dir.glob('*.png'):
        try:
            idx = int(file.stem)
            existing.add(idx)
        except ValueError:
            continue
    return existing

def main():
    backtracking_chain = True
    for version in [2755, 2757]:
        parser = argparse.ArgumentParser(description='Capture Factorio program evolution screenshots')
        parser.add_argument('--version', '-v', type=int, default=version, help=f'Program version to capture (default: {version})')
        parser.add_argument('--output-dir', '-o', default='screenshots', help='Output directory for screenshots and video')
        parser.add_argument('--framerate', '-f', type=int, default=30, help='Framerate for output video')
        parser.add_argument('--script_output_path', '-s', type=str, default='/Users/jackhopkins/Library/Application Support/factorio/script-output', help='path where the factorio script will save screenshots to')
        import sys
        if len(sys.argv) > 1:
            args = parser.parse_args()
        else:
            args = parser.parse_args([])
        output_base = Path(args.output_dir)
        version_dir = output_base / str(args.version)
        version_dir.mkdir(parents=True, exist_ok=True)
        conn = get_db_connection()
        try:
            print(f'Getting program chain for version {args.version}')
            program_ids = get_program_chain_backtracking(conn, args.version) if backtracking_chain else get_program_chain(conn, args.version)
            if not program_ids:
                print(f'No programs found for version {args.version}')
                return
            print(f'Found {len(program_ids)} programs')
            instance = create_factorio_instance()
            capture_screenshots_with_hooks(program_ids, str(version_dir), args.script_output_path, instance, conn)
            output_video = version_dir / 'output.mp4'
            png_to_mp4(str(version_dir), str(output_video), args.framerate)
            print(f'Created video: {output_video}')
        finally:
            conn.close()

def get_db_connection():
    """Create a database connection using environment variables"""
    return psycopg2.connect(host=os.getenv('SKILLS_DB_HOST'), port=os.getenv('SKILLS_DB_PORT'), dbname=os.getenv('SKILLS_DB_NAME'), user=os.getenv('SKILLS_DB_USER'), password=os.getenv('SKILLS_DB_PASSWORD'))

def get_program_chain_backtracking(conn, version: int):
    """Get the chain of programs for a specific version for the backtracking chain"""
    query = f'\n    SELECT meta, code, response, id, created_at FROM programs \n    WHERE version = {version}\n    ORDER BY created_at ASC\n    '
    model = 'anthropic/claude-3.5-sonnet-open-router'
    with conn.cursor() as cur:
        cur.execute(query)
        data = cur.fetchall()
    data = [(x[-2], x[-1]) for x in data if x[0]['model'] == model and (not x[0]['error_occurred'])]
    return data

def get_program_chain(conn, version: int):
    """Get the chain of programs for a specific version using recursive CTE"""
    latest_query = '\n    SELECT id FROM programs \n    WHERE version = %s \n    AND state_json IS NOT NULL \n    ORDER BY created_at DESC \n    LIMIT 1\n    '
    with conn.cursor() as cur:
        cur.execute(latest_query, (version,))
        latest_result = cur.fetchone()
        if not latest_result:
            return []
        latest_id = latest_result[0]
        recursive_query = '\n        WITH RECURSIVE program_trace AS (\n            -- Base case: start with most recent program\n            SELECT \n                id,\n                parent_id,\n                created_at\n            FROM programs\n            WHERE id = %s\n\n            UNION ALL\n\n            -- Recursive case: get the parent program\n            SELECT \n                p.id,\n                p.parent_id,\n                p.created_at\n            FROM programs p\n            INNER JOIN program_trace pt ON p.id = pt.parent_id\n        )\n        SELECT id, created_at FROM program_trace\n        ORDER BY created_at ASC\n        LIMIT 3000\n        '
        cur.execute(recursive_query, (latest_id,))
        return cur.fetchall()

def get_postgres_exceptions():
    """Get PostgreSQL exception classes if available, otherwise return empty tuple"""
    if PSYCOPG2_AVAILABLE:
        return (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.DatabaseError)
    return ()

def get_sqlite_exceptions():
    """Get SQLite exception classes"""
    return (sqlite3.OperationalError, sqlite3.InterfaceError, sqlite3.DatabaseError)

def create_default_postgres_db(**db_config) -> None:
    """Create PostgreSQL database with required schema if it doesn't exist"""
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("\n            SELECT EXISTS (\n                SELECT FROM information_schema.tables \n                WHERE table_schema = 'public' \n                AND table_name = 'programs'\n            )\n        ")
        table_exists = cursor.fetchone()[0]
        if not table_exists:
            print('Creating PostgreSQL database schema')
            cursor.execute("\n                CREATE TABLE programs (\n                    id SERIAL PRIMARY KEY,\n                    code TEXT NOT NULL,\n                    value REAL DEFAULT 0.0,\n                    visits INTEGER DEFAULT 0,\n                    parent_id INTEGER,\n                    state_json TEXT,\n                    conversation_json TEXT NOT NULL,\n                    completion_token_usage INTEGER,\n                    prompt_token_usage INTEGER,\n                    token_usage INTEGER,\n                    response TEXT,\n                    holdout_value REAL,\n                    raw_reward REAL,\n                    version INTEGER DEFAULT 1,\n                    version_description TEXT DEFAULT '',\n                    model TEXT DEFAULT 'gpt-4o',\n                    meta TEXT,\n                    achievements_json TEXT,\n                    instance INTEGER DEFAULT -1,\n                    depth REAL DEFAULT 0.0,\n                    advantage REAL DEFAULT 0.0,\n                    ticks INTEGER DEFAULT 0,\n                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                    timing_metrics_json TEXT\n                )\n            ")
            conn.commit()
            print('PostgreSQL database schema created successfully!')
    except Exception as e:
        print(f'Error creating PostgreSQL schema: {e}')
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def create_default_sqlite_db(db_file: str) -> None:
    """Create SQLite database with required schema if it doesn't exist"""
    db_path = Path(db_file)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        cursor = conn.cursor()
        cursor.execute("\n            SELECT name FROM sqlite_master \n            WHERE type='table' AND name='programs'\n        ")
        if not cursor.fetchone():
            print(f'Creating SQLite database schema in {db_file}')
            cursor.execute("\n                CREATE TABLE programs (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    code TEXT NOT NULL,\n                    value REAL DEFAULT 0.0,\n                    visits INTEGER DEFAULT 0,\n                    parent_id INTEGER,\n                    state_json TEXT,\n                    conversation_json TEXT NOT NULL,\n                    completion_token_usage INTEGER,\n                    prompt_token_usage INTEGER,\n                    token_usage INTEGER,\n                    response TEXT,\n                    holdout_value REAL,\n                    raw_reward REAL,\n                    version INTEGER DEFAULT 1,\n                    version_description TEXT DEFAULT '',\n                    model TEXT DEFAULT 'gpt-4o',\n                    meta TEXT,\n                    achievements_json TEXT,\n                    instance INTEGER DEFAULT -1,\n                    depth REAL DEFAULT 0.0,\n                    advantage REAL DEFAULT 0.0,\n                    ticks INTEGER DEFAULT 0,\n                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,\n                    timing_metrics_json TEXT\n                )\n            ")
            conn.commit()
            print('SQLite database schema created successfully!')
    finally:
        conn.close()

def get_program_state(conn, program_id: int):
    """Fetch a single program's full state by ID"""
    query = '\n    SELECT * FROM programs WHERE id = %s\n    '
    with conn.cursor() as cur:
        cur.execute(query, (program_id,))
        row = cur.fetchone()
        if not row:
            return None
        col_names = [desc[0] for desc in cur.description]
        return Program.from_row(dict(zip(col_names, row)))

def create_gym_environment(version: int) -> FactorioGymEnv:
    """Create a gym environment based on the task from the first program of a version"""
    available_envs = list_available_environments()
    print(f'Available gym environments: {available_envs[:5]}... (total: {len(available_envs)})')
    conn = get_db_connection()
    try:
        query = '\n        SELECT meta FROM programs\n        WHERE version = %s\n        AND meta IS NOT NULL\n        ORDER BY created_at ASC\n        LIMIT 1\n        '
        with conn.cursor() as cur:
            cur.execute(query, (version,))
            result = cur.fetchone()
            if not result:
                raise ValueError(f'No programs found for version {version}')
            meta = result[0]
            version_description = meta.get('version_description', '')
            if 'type:' in version_description:
                task_key = version_description.split('type:')[1].split('\n')[0].strip()
            else:
                task_key = available_envs[0] if available_envs else 'iron_plate_throughput'
                print(f'Warning: Could not determine task from version {version}, using default: {task_key}')
            env_id = task_key
            print(f'Trying to create environment: {env_id}')
            if env_id not in available_envs:
                print(f'Warning: Environment {env_id} not in available list. Available: {available_envs[:10]}')
                if available_envs:
                    env_id = available_envs[0]
                    print(f'Using fallback environment: {env_id}')
                else:
                    raise ValueError('No gym environments available!')
            try:
                gym_env = gym.make(env_id, run_idx=0)
                print(f'Successfully created gym environment: {env_id}')
                return gym_env
            except Exception as e:
                print(f'Failed to create gym environment {env_id}: {e}')
                if available_envs and env_id != available_envs[0]:
                    fallback_env_id = available_envs[0]
                    print(f'Trying final fallback environment: {fallback_env_id}')
                    gym_env = gym.make(fallback_env_id, run_idx=0)
                    return gym_env
                else:
                    raise e
    finally:
        conn.close()

def capture_camera_transition(instance, script_output_path, output_dir, screenshot_counter, start_camera, end_camera, transition_frames=15, easing_func=ease_in_out_cubic):
    """Capture a smooth camera transition between two states.

    Args:
        instance: Game instance
        script_output_path: Path to script output
        output_dir: Directory to save screenshots
        screenshot_counter: Current screenshot counter
        start_camera: Starting camera state dict with 'position' and 'zoom'
        end_camera: Ending camera state dict with 'position' and 'zoom'
        transition_frames: Number of interpolation frames
        easing_func: Easing function to use

    Returns:
        Updated screenshot_counter
    """
    if start_camera is None or end_camera is None:
        screenshot_filename = f'{screenshot_counter:06d}.png'
        save_path = str(output_dir / screenshot_filename)
        take_screenshot(instance, script_output_path, save_path=save_path)
        return screenshot_counter + 1
    pos_diff = math.sqrt((end_camera['position'][0] - start_camera['position'][0]) ** 2 + (end_camera['position'][1] - start_camera['position'][1]) ** 2)
    zoom_diff = abs(end_camera['zoom'] - start_camera['zoom'])
    if pos_diff < 5 and zoom_diff < 0.1:
        transition_frames = 1
    for i in range(transition_frames):
        progress = i / max(transition_frames - 1, 1)
        cam_x, cam_y, cam_zoom = interpolate_camera(start_camera['position'], end_camera['position'], start_camera['zoom'], end_camera['zoom'], progress, easing_func)
        screenshot_filename = f'{screenshot_counter:06d}.png'
        save_path = str(output_dir / screenshot_filename)
        take_screenshot(instance, script_output_path, save_path=save_path, camera_position=(cam_x, cam_y), camera_zoom=cam_zoom)
        screenshot_counter += 1
        time.sleep(0.05)
    return screenshot_counter

def interpolate_camera(start_pos, end_pos, start_zoom, end_zoom, progress, easing_func=ease_in_out_cubic):
    """Interpolate camera position and zoom with easing.

    Args:
        start_pos: (x, y) starting camera position
        end_pos: (x, y) ending camera position
        start_zoom: Starting zoom level
        end_zoom: Ending zoom level
        progress: Progress from 0.0 to 1.0
        easing_func: Easing function to use

    Returns:
        Tuple of (x, y, zoom) for interpolated camera state
    """
    eased_progress = easing_func(progress)
    x = start_pos[0] + (end_pos[0] - start_pos[0]) * eased_progress
    y = start_pos[1] + (end_pos[1] - start_pos[1]) * eased_progress
    if start_zoom > 0 and end_zoom > 0:
        log_start = math.log(start_zoom)
        log_end = math.log(end_zoom)
        log_zoom = log_start + (log_end - log_start) * eased_progress
        zoom = math.exp(log_zoom)
    else:
        zoom = start_zoom + (end_zoom - start_zoom) * eased_progress
    return (x, y, zoom)

def take_screenshot(instance, script_output_path: str, save_path: str=None, resolution: str='1920x1080', center_on_factory: bool=True, camera_position=None, camera_zoom=None):
    """Take a screenshot using Factorio's game.take_screenshot API."""
    instance.rcon_client.send_command('/sc rendering.clear()')
    if camera_position is not None and camera_zoom is not None:
        center_x, center_y = camera_position
        zoom = camera_zoom
        position_str = f', position={{x={center_x}, y={center_y}}}'
    elif center_on_factory:
        min_x, min_y, max_x, max_y = get_factory_bounds(instance)
        if max_x != 0 or max_y != 0:
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            factory_width = max_x - min_x + 20
            factory_height = max_y - min_y + 20
            zoom = calculate_optimal_zoom(factory_width, factory_height, resolution)
            position_str = f', position={{x={center_x}, y={center_y}}}'
        else:
            zoom = 1.0
            position_str = ''
    else:
        zoom = 1.0
        position_str = ''
    command = f'/sc game.take_screenshot({{zoom={zoom}, show_entity_info=true, hide_clouds=true, hide_fog=true{position_str}}})'
    instance.rcon_client.send_command(command)
    time.sleep(0.2)
    screenshot_path = get_latest_screenshot(script_output_path)
    if not screenshot_path:
        print('Screenshot file not found')
        return None
    if save_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            shutil.copy2(screenshot_path, save_path)
            return save_path
        except Exception as e:
            print(f'Failed to copy screenshot: {e}')
            return screenshot_path
    return screenshot_path

def get_factory_bounds(instance):
    """Get the bounding box of all entities in the factory."""
    bounds_cmd = '/sc local entities = game.surfaces[1].find_entities_filtered{force=game.forces.player}\n    if #entities == 0 then\n        rcon.print("0,0,0,0")\n    else\n        local min_x, min_y = math.huge, math.huge\n        local max_x, max_y = -math.huge, -math.huge\n        for _, e in pairs(entities) do\n            if e.position.x < min_x then min_x = e.position.x end\n            if e.position.y < min_y then min_y = e.position.y end\n            if e.position.x > max_x then max_x = e.position.x end\n            if e.position.y > max_y then max_y = e.position.y end\n        end\n        rcon.print(string.format("%.2f,%.2f,%.2f,%.2f", min_x, min_y, max_x, max_y))\n    end\n    '
    try:
        bounds_result = instance.rcon_client.send_command(bounds_cmd)
        min_x, min_y, max_x, max_y = map(float, bounds_result.split(','))
        return (min_x, min_y, max_x, max_y)
    except:
        return (0, 0, 0, 0)

def calculate_optimal_zoom(factory_width, factory_height, resolution='1920x1080'):
    """Calculate the optimal zoom level to fit the factory in the screenshot."""
    width, height = map(int, resolution.split('x'))
    aspect_ratio = width / height
    BASE_VISIBLE_HEIGHT = 25
    BASE_VISIBLE_WIDTH = BASE_VISIBLE_HEIGHT * aspect_ratio
    if factory_width > 0 and factory_height > 0:
        zoom_by_width = BASE_VISIBLE_WIDTH / factory_width
        zoom_by_height = BASE_VISIBLE_HEIGHT / factory_height
        optimal_zoom = min(zoom_by_width, zoom_by_height)
        optimal_zoom *= 1.2
        MIN_ZOOM = 0.1
        MAX_ZOOM = 4.0
        optimal_zoom = max(MIN_ZOOM, min(MAX_ZOOM, optimal_zoom))
        return round(optimal_zoom, 2)
    return 1.0

def get_latest_screenshot(script_output_path, max_wait=2):
    """Get the path to the latest screenshot in the script-output directory."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            screenshots = [f for f in os.listdir(script_output_path) if f.endswith('.png') and f.startswith('screenshot')]
            if screenshots:
                latest = max(screenshots, key=lambda x: os.path.getmtime(os.path.join(script_output_path, x)))
                return os.path.join(script_output_path, latest)
        except Exception as e:
            print(f'Error checking for screenshots: {e}')
        time.sleep(0.1)
    return None

def capture_screenshots_gym(program_ids, output_dir: Path, script_output_path: str, gym_env: FactorioGymEnv, conn, max_steps: int, capture_interval: float=0, transition_frames: int=10, easing: str='cubic', args=None):
    """
    Capture screenshots by replaying programs through a gym environment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def get_highest_screenshot_number():
        existing_files = list(output_dir.glob('*.png'))
        if not existing_files:
            return -1
        highest = -1
        for file in existing_files:
            try:
                num = int(file.stem)
                highest = max(highest, num)
            except ValueError:
                continue
        return highest
    screenshot_counter = get_highest_screenshot_number() + 1
    print(f'Starting screenshot numbering from {screenshot_counter}')
    instance = gym_env.unwrapped.instance
    if hasattr(args, 'hook_tools') and args.hook_tools:
        screenshot_hook_counter = {'count': 0}

        def capture_after_tool(tool_instance, result):
            """Hook to capture screenshot after tool execution"""
            nonlocal screenshot_counter
            if screenshot_hook_counter['count'] % args.hook_frequency == 0:
                screenshot_filename = f'{screenshot_counter:06d}.png'
                save_path = str(output_dir / screenshot_filename)
                current_camera = get_camera_state_for_factory(instance)
                if current_camera:
                    take_screenshot(instance, script_output_path, save_path=save_path, camera_position=current_camera['position'], camera_zoom=current_camera['zoom'])
                else:
                    take_screenshot(instance, script_output_path, save_path=save_path)
                screenshot_counter += 1
            screenshot_hook_counter['count'] += 1
        hook_tools = ['place_entity', 'place_entity_next_to', 'connect_entities', 'insert_item', 'pickup_entity', 'rotate_entity', 'move_to']
        for tool_name in hook_tools:
            LuaScriptManager.register_post_tool_hook(instance, tool_name, capture_after_tool)
        print(f'Registered screenshot hooks for: {', '.join(hook_tools)}')
    observation, info = gym_env.reset()
    initial_camera = get_camera_state_for_factory(instance)
    if initial_camera is None:
        initial_camera = {'position': (0, 0), 'zoom': 1.0}
    screenshot_filename = f'{screenshot_counter:06d}.png'
    save_path = str(output_dir / screenshot_filename)
    if take_screenshot(instance, script_output_path, save_path=save_path):
        print(f'Captured initial screenshot: {screenshot_filename}')
    screenshot_counter += 1
    current_game_state = None
    previous_camera = initial_camera
    for idx, (program_id, created_at) in enumerate(program_ids):
        if idx >= max_steps:
            break
        program = get_program_state(conn, program_id)
        if not program or not program.code:
            print(f'Skipping program {program_id} - no code available')
            continue
        print(f'Processing program {idx + 1}/{len(program_ids)}: {program_id}')
        print(f'Code: {program.code[:100]}...')
        try:
            action = Action(code=program.code, game_state=current_game_state)
            if capture_interval > 0:
                import threading
                import time as time_module
                stop_capture = threading.Event()
                capture_exception = None

                def capture_during_execution():
                    nonlocal screenshot_counter, capture_exception
                    last_capture_time = time_module.time()
                    while not stop_capture.is_set():
                        current_time = time_module.time()
                        if current_time - last_capture_time >= capture_interval:
                            screenshot_filename = f'{screenshot_counter:06d}.png'
                            save_path = str(output_dir / screenshot_filename)
                            try:
                                if take_screenshot(instance, script_output_path, save_path=save_path):
                                    print(f'  Captured mid-execution screenshot: {screenshot_filename}')
                                screenshot_counter += 1
                                last_capture_time = current_time
                            except Exception as e:
                                capture_exception = e
                        time_module.sleep(0.1)
                capture_thread = threading.Thread(target=capture_during_execution, daemon=True)
                capture_thread.start()
                observation, reward, terminated, truncated, info = gym_env.step(action)
                stop_capture.set()
                capture_thread.join(timeout=1.0)
                if capture_exception:
                    print(f'  Warning: Screenshot capture thread error: {capture_exception}')
            else:
                observation, reward, terminated, truncated, info = gym_env.step(action)
            print(f'  Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}')
            raw_text = BasicObservationFormatter.format_raw_text(observation['raw_text'])
            print(raw_text)
            current_game_state = program.state
            new_camera = get_camera_state_for_factory(instance)
            if new_camera is None:
                new_camera = previous_camera
            if easing == 'sine':
                easing_func = ease_in_out_sine
            elif easing == 'linear':
                easing_func = lambda t: t
            else:
                easing_func = ease_in_out_cubic
            print(f'  Capturing camera transition ({transition_frames} frames)...')
            screenshot_counter = capture_camera_transition(instance, script_output_path, output_dir, screenshot_counter, previous_camera, new_camera, transition_frames=transition_frames, easing_func=easing_func)
            previous_camera = new_camera
            if terminated:
                print(f'  Environment terminated after program {idx + 1}')
                break
        except Exception as e:
            print(f'  Error executing program {program_id}: {e}')
            if program.state:
                current_game_state = program.state
            error_camera = get_camera_state_for_factory(instance)
            if error_camera is None:
                error_camera = previous_camera
            for _ in range(3):
                screenshot_filename = f'{screenshot_counter:06d}.png'
                save_path = str(output_dir / screenshot_filename)
                if take_screenshot(instance, script_output_path, save_path=save_path, camera_position=error_camera['position'], camera_zoom=error_camera['zoom']):
                    print(f'  Captured error-state screenshot: {screenshot_filename}')
                screenshot_counter += 1
            previous_camera = error_camera
            continue
    print('Capturing final state frames...')
    for i in range(10):
        try:
            action = Action(code='sleep(15)')
            observation, reward, terminated, truncated, info = gym_env.step(action)
        except:
            pass
        screenshot_filename = f'{screenshot_counter:06d}.png'
        save_path = str(output_dir / screenshot_filename)
        if take_screenshot(instance, script_output_path, save_path=save_path):
            print(f'  Captured final frame {i + 1}/10: {screenshot_filename}')
        screenshot_counter += 1
    print(f'Screenshot capture complete. Total screenshots: {screenshot_counter - 1}')

def capture_after_tool(tool_instance, result):
    """Hook to capture screenshot after tool execution"""
    nonlocal screenshot_counter
    if screenshot_hook_counter['count'] % args.hook_frequency == 0:
        screenshot_filename = f'{screenshot_counter:06d}.png'
        save_path = str(output_dir / screenshot_filename)
        current_camera = get_camera_state_for_factory(instance)
        if current_camera:
            take_screenshot(instance, script_output_path, save_path=save_path, camera_position=current_camera['position'], camera_zoom=current_camera['zoom'])
        else:
            take_screenshot(instance, script_output_path, save_path=save_path)
        screenshot_counter += 1
    screenshot_hook_counter['count'] += 1

def get_camera_state_for_factory(instance, resolution='1920x1080'):
    """Calculate the optimal camera state for the current factory.

    Returns:
        Dict with 'position' and 'zoom' keys, or None if no factory
    """
    min_x, min_y, max_x, max_y = get_factory_bounds(instance)
    if max_x == 0 and max_y == 0:
        return None
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    factory_width = max_x - min_x + 20
    factory_height = max_y - min_y + 20
    zoom = calculate_optimal_zoom(factory_width, factory_height, resolution)
    return {'position': (center_x, center_y), 'zoom': zoom}

def capture_during_execution():
    nonlocal screenshot_counter, capture_exception
    last_capture_time = time_module.time()
    while not stop_capture.is_set():
        current_time = time_module.time()
        if current_time - last_capture_time >= capture_interval:
            screenshot_filename = f'{screenshot_counter:06d}.png'
            save_path = str(output_dir / screenshot_filename)
            try:
                if take_screenshot(instance, script_output_path, save_path=save_path):
                    print(f'  Captured mid-execution screenshot: {screenshot_filename}')
                screenshot_counter += 1
                last_capture_time = current_time
            except Exception as e:
                capture_exception = e
        time_module.sleep(0.1)

def process_version(version: int, output_base: Path, script_output_path: str, framerate: int, max_steps: int, with_hooks: bool, skip_screenshots: bool, skip_video: bool, capture_interval: float, transition_frames: int=10, easing: str='cubic', args=None):
    """Process a single version: generate screenshots and create video."""
    version_dir = output_base / str(version)
    version_dir.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    try:
        if not skip_screenshots:
            print(f'\nProcessing version {version}')
            print('Getting program chain from database...')
            program_ids = get_program_chain(conn, version)
            if not program_ids:
                print(f'No programs found for version {version}')
                return False
            print(f'Found {len(program_ids)} programs')
            print('Creating gym environment...')
            gym_env = create_gym_environment(version)
            print('Capturing screenshots using gym environment...')
            capture_screenshots_gym(program_ids, version_dir, script_output_path, gym_env, conn, max_steps, capture_interval=capture_interval, transition_frames=transition_frames, easing=easing, args=args)
        if not skip_video:
            output_video = version_dir / 'output.mp4'
            print('\nCreating video from screenshots...')
            success = png_to_mp4(version_dir, output_video, framerate)
            if success:
                print(f'Video saved to: {output_video}')
                return True
            else:
                print(f'Failed to create video for version {version}')
                return False
        return True
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description='Generate screenshots and create MP4 videos from Factorio program versions.', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  # Process single version with default settings\n  %(prog)s 2755\n\n  # Process multiple versions\n  %(prog)s 2755 2757 2760\n\n  # Custom output directory and framerate\n  %(prog)s 2755 --output-dir my_videos --framerate 60\n\n  # Only generate screenshots, skip video creation\n  %(prog)s 2755 --skip-video\n\n  # Only create video from existing screenshots\n  %(prog)s 2755 --skip-screenshots\n\n  # Disable hooks for faster processing (less detailed)\n  %(prog)s 2755 --no-hooks\n        ')
    parser.add_argument('versions', nargs='+', type=int, help='Version numbers to process')
    parser.add_argument('--output-dir', '-o', default='videos', help='Base output directory for screenshots and videos (default: videos)')
    parser.add_argument('--framerate', '-f', type=int, default=30, help='Framerate for output video (default: 30)')
    parser.add_argument('--script-output-path', '-s', type=str, help='Path where Factorio saves screenshots (defaults to auto-detect based on platform)')
    parser.add_argument('--max-steps', '-m', type=int, default=1000, help='Maximum number of program steps to capture (default: 1000)')
    parser.add_argument('--no-hooks', action='store_true', help='Disable hooks for entity placement (faster but less detailed)')
    parser.add_argument('--skip-screenshots', action='store_true', help='Skip screenshot generation, only create video from existing PNGs')
    parser.add_argument('--skip-video', action='store_true', help='Skip video creation, only generate screenshots')
    parser.add_argument('--capture-interval', '-c', type=float, default=0, help='Capture screenshots every N seconds during program execution (0 = disabled, only capture after each program)')
    parser.add_argument('--transition-frames', '-t', type=int, default=10, help='Number of frames for smooth camera transitions between programs (default: 10)')
    parser.add_argument('--easing', type=str, default='cubic', choices=['cubic', 'sine', 'linear'], help='Easing function for camera transitions (default: cubic)')
    parser.add_argument('--hook-tools', action='store_true', help='Capture screenshots after tool executions (place_entity, connect_entities, etc.)')
    parser.add_argument('--hook-frequency', type=int, default=1, help='Capture screenshot every N tool executions when using --hook-tools (default: 1)')
    args = parser.parse_args()
    if not args.script_output_path:
        if sys.platform == 'darwin':
            args.script_output_path = os.path.expanduser('~/Library/Application Support/factorio/script-output')
        elif sys.platform == 'linux':
            args.script_output_path = os.path.expanduser('~/.factorio/script-output')
        elif sys.platform == 'win32':
            args.script_output_path = os.path.expanduser('~/AppData/Roaming/Factorio/script-output')
        else:
            print(f'Warning: Could not auto-detect script output path for platform {sys.platform}')
            print('Please specify with --script-output-path')
            sys.exit(1)
    script_output_path = Path(args.script_output_path)
    if not script_output_path.exists():
        print(f'Warning: Script output path does not exist: {script_output_path}')
        print('Creating directory...')
        script_output_path.mkdir(parents=True, exist_ok=True)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    success_count = 0
    for version in args.versions:
        try:
            success = process_version(version, output_base, str(script_output_path), args.framerate, args.max_steps, not args.no_hooks, args.skip_screenshots, args.skip_video, args.capture_interval, args.transition_frames, args.easing, args)
            if success:
                success_count += 1
        except Exception as e:
            print(f'Error processing version {version}: {e}')
            import traceback
            traceback.print_exc()
    print(f'\nProcessed {success_count}/{len(args.versions)} versions successfully')
    if success_count == len(args.versions):
        sys.exit(0)
    else:
        sys.exit(1)

