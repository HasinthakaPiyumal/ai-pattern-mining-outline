# Cluster 9

def get_local_container_ips() -> tuple[List[str], List[int], List[int]]:
    """Get IP addresses of running Factorio containers in the local Docker setup."""
    cmd = ['docker', 'ps', '--filter', 'name=factorio_', '--format', '"{{.ID}}"']
    result = subprocess.run(cmd, capture_output=True, text=True)
    container_ids = result.stdout.strip().split('\n')
    container_ids = [id.strip('"') for id in container_ids]
    if not container_ids or container_ids[0] == '':
        print('No running Factorio containers found')
        return []
    ips = []
    udp_ports = []
    tcp_ports = []
    for container_id in container_ids:
        cmd = ['docker', 'inspect', container_id]
        result = subprocess.run(cmd, capture_output=True, text=True)
        container_info = json.loads(result.stdout)
        ports = container_info[0]['NetworkSettings']['Ports']
        for port, bindings in ports.items():
            if '/udp' in port and bindings:
                udp_port = bindings[0]['HostPort']
                udp_ports.append(int(udp_port))
            if '/tcp' in port and bindings:
                tcp_port = bindings[0]['HostPort']
                tcp_ports.append(int(tcp_port))
        ips.append('127.0.0.1')
    udp_ports.sort(key=lambda x: int(x))
    tcp_ports.sort(key=lambda x: int(x))
    return (ips, udp_ports, tcp_ports)

class FactorioControlPanel:
    """Minimal control panel for Factorio agent gameplay."""

    def __init__(self, env_id: str=None, model: str=None, mcp_server_path: str='fle/env/protocols/mcp/server.py', websocket_url: str='ws://localhost:8000/ws'):
        logger.info('Initializing FactorioControlPanel')
        self.ws_url = websocket_url
        self.ws = None
        self.mcp_server_path = mcp_server_path or 'fle/env/protocols/mcp/server.py'
        self.mcp_session = None
        self.update_queue = Queue()
        self.runner_thread = None
        self.current_step = 0
        self.max_steps = 100
        self.is_running = False
        self.env_id = env_id or 'steel_plate_throughput'
        self.model = model or 'openai/gpt-5-mini'
        logger.info('Configuration - env_id: %s, model: %s', self.env_id, self.model)
        self.icon_manager = IconManager()
        self.inventory_items = {}
        self.last_score = 0
        self.score_delta = 0
        self.production_history = {'timestamps': deque(maxlen=50), 'data': {}}
        self.chart = None
        self.coin_icon = None
        self.score_label = None
        self.camera_tracking_enabled = True
        self.camera_zoom_scale = 0.5
        self.last_camera_update = 0
        self.camera_update_interval = 3.0
        self.auto_start = True
        logger.info('Auto-start enabled: %s', self.auto_start)
        ips, udp_ports, tcp_ports = get_local_container_ips()
        rcon_client, address = FactorioInstance.connect_to_server(ips[0], tcp_ports[0])
        self.rcon_client = rcon_client

    def update_inventory_display(self):
        """Update the inventory grid display with sprite icons."""
        self.inventory_grid.clear()
        with self.inventory_grid:
            sorted_items = sorted(self.inventory_items.items(), key=lambda x: x[1], reverse=True)
            for item_name, count in sorted_items[:20]:
                with ui.card().classes('w-16 h-16 bg-gray-700 p-1 relative overflow-hidden'):
                    icon_data = self.icon_manager.get_icon_base64(item_name)
                    if icon_data:
                        ui.image(icon_data).classes('absolute inset-0 w-full h-full object-contain p-1')
                    else:
                        emoji = self.icon_manager.get_emoji_fallback(item_name)
                        ui.label(emoji).classes('text-2xl absolute top-1 left-1/2 transform -translate-x-1/2')
                    ui.label(str(count)).classes('text-sm text-white absolute bottom-0 right-1 bg-gray-900 px-1 rounded font-bold')
                    ui.tooltip(item_name)

    def update_production_chart(self, production_flows: Dict[str, Any]):
        """Update the production chart with new data."""
        if not production_flows:
            return
        self.production_history['timestamps'].append(time.time())
        outputs = production_flows.get('output', {})
        for item, amount in outputs.items():
            if item not in self.production_history['data']:
                self.production_history['data'][item] = deque(maxlen=50)
            self.production_history['data'][item].append(amount)
        total_value = 0
        for item, amount in outputs.items():
            if item not in self.production_history['data']:
                self.production_history['data'][item] = deque(maxlen=50)
            self.production_history['data'][item].append(amount)
            total_value += amount
        if 'cumulative_total' not in self.production_history['data']:
            self.production_history['data']['cumulative_total'] = deque(maxlen=50)
            self.cumulative_sum = 0
        self.cumulative_sum = total_value
        self.production_history['data']['cumulative_total'].append(self.cumulative_sum)
        max_len = len(self.production_history['timestamps'])
        for series in self.production_history['data'].values():
            while len(series) < max_len:
                series.appendleft(0)
        if self.chart:
            self.update_chart_display()

    def update_chart_display(self):
        """Update the chart visualization with icon legends."""
        if not self.chart:
            return
        x_data = list(range(len(self.production_history['timestamps'])))
        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
        top_items = sorted(self.production_history['data'].items(), key=lambda x: sum(x[1]) if x[1] else 0, reverse=True)[:20]
        series = []
        legend_data = []
        if 'cumulative_total' in self.production_history['data']:
            cumulative_values = self.production_history['data']['cumulative_total']
            series.append({'name': 'cumulative_total', 'type': 'line', 'data': list(cumulative_values), 'smooth': True, 'lineStyle': {'color': '#FFD700', 'width': 3, 'type': 'solid'}, 'itemStyle': {'color': '#FFD700'}, 'showSymbol': False, 'animationDuration': 30})
            legend_data.append({'name': 'Cumulative Total', 'icon': ''})
        for i, (item_name, values) in enumerate(top_items):
            color = colors[i % len(colors)]
            icon_data = self.icon_manager.get_icon_base64(item_name)
            legend_name = item_name[:20]
            series.append({'name': legend_name, 'type': 'line', 'data': list(values), 'smooth': True, 'lineStyle': {'color': color, 'width': 1, 'type': 'dashed'}, 'itemStyle': {'color': color}, 'showSymbol': False, 'animationDuration': 30})
            if icon_data:
                legend_data.append({'name': legend_name, 'icon': f'image://{icon_data}'})
        try:
            _series = series[:1] + series[2:]
            assert len(_series) == len(legend_data)
            chart_options = {'backgroundColor': 'transparent', 'xAxis': {'type': 'category', 'data': x_data, 'axisLabel': {'color': 'rgba(255, 255, 255, 0.7)'}, 'axisLine': {'lineStyle': {'color': 'rgba(255, 255, 255, 0.3)'}}}, 'yAxis': {'type': 'value', 'axisLabel': {'color': 'rgba(255, 255, 255, 0.7)'}, 'splitLine': {'lineStyle': {'color': 'rgba(255, 255, 255, 0.1)'}}}, 'series': _series, 'legend': {'data': legend_data if isinstance(legend_data[0], str) else [item['name'] for item in legend_data], 'textStyle': {'color': 'rgba(255, 255, 255, 0.7)'}, 'itemWidth': 20, 'itemHeight': 20, 'top': 0, 'icon': 'rect'}, 'grid': {'left': '5%', 'right': '5%', 'bottom': '5%', 'top': '15%', 'containLabel': True}, 'tooltip': {'trigger': 'axis', 'backgroundColor': 'rgba(0, 0, 0, 0.8)', 'borderColor': '#333', 'textStyle': {'color': '#fff'}}}
        except Exception as e:
            raise e
        self.chart.options['series'] = chart_options['series']
        self.chart.options['xAxis']['data'] = chart_options['xAxis']['data']
        self.chart.options['legend']['data'] = chart_options['legend']['data']
        try:
            self.chart.update()
        except Exception:
            pass

    def process_updates(self):
        """Process updates from the runner."""
        while not self.update_queue.empty():
            update = self.update_queue.get()
            if update['type'] == 'state_update':
                if 'inventory' in update and update['inventory']:
                    if self.inventory_items != update['inventory']:
                        self.inventory_items = update['inventory']
                        self.update_inventory_display()
                if 'score' in update:
                    score_text = f'{update['score']:.2f}'
                    self.score_label.set_content(score_text)
                if 'entities_count' in update:
                    self.entities_label.set_text(f'Entities: {update['entities_count']}')
                if 'game_tick' in update:
                    self.tick_label.set_text(f'Tick: {update['game_tick']:,}')
                if 'research_progress' in update:
                    research_pct = update['research_progress'] * 100
                    self.research_label.set_text(f'Research: {research_pct:.1f}%')
                if 'production_flows' in update:
                    self.update_production_chart(update['production_flows'])
                logger.debug(f'Processed state_update at {update.get('timestamp', 'unknown')}')
            elif update['type'] == 'init':
                self.max_steps = update.get('max_steps', 100)
                self.step_label.set_text(f'Step: 0 / {self.max_steps}')
                self.status_label.set_text('🟢 Running')
                coin_icon_data = self.icon_manager.get_icon_base64('coin')
                if coin_icon_data and self.coin_icon:
                    self.coin_icon.set_source(coin_icon_data)
                elif self.coin_icon:
                    self.coin_icon.set_source('')
                    self.coin_icon.classes('hidden')
            elif update['type'] == 'update':
                self.current_step = update['step']
                self.progress_bar.set_value(self.current_step / self.max_steps)
                self.step_label.set_text(f'Step: {self.current_step} / {self.max_steps}')
                new_score = update.get('score', 0)
                reward = update.get('reward', 0)
                score_text = f'{new_score:.2f}'
                if reward != 0:
                    if reward > 0:
                        reward_text = f'<span style="color: #10b981; font-size: 0.8em; vertical-align: super;">+{reward:.3f}</span>'
                    else:
                        reward_text = f'<span style="color: #ef4444; font-size: 0.8em; vertical-align: super;">{reward:.3f}</span>'
                    self.score_label.set_content(score_text + ' ' + reward_text)
                else:
                    self.score_label.set_content(score_text)
                self.last_score = new_score
                if 'observation' in update:
                    obs = update['observation']
                    self.entities_label.set_text(f'Entities: {obs.get('entities_count', 0)}')
                    self.tick_label.set_text(f'Tick: {obs.get('game_tick', 0):,}')
                    research_pct = obs.get('research_progress', 0) * 100
                    self.research_label.set_text(f'Research: {research_pct:.1f}%')
                if 'inventory' in update and update['inventory']:
                    self.inventory_items = update['inventory']
                    self.update_inventory_display()
                if 'production_flows' in update:
                    self.update_production_chart(update['production_flows'])
            elif update['type'] == 'complete':
                self.is_running = False
                status = '✅ Complete' if update.get('success') else '⏹️ Finished'
                self.status_label.set_text(status)
            elif update['type'] == 'error':
                self.status_label.set_text('❌ Error' + f'\n{update['message']}')
                self.is_running = False

    def create_ui(self):
        """Create the minimal control panel interface with production chart on the right."""
        if hasattr(self, '_ui_created'):
            logger.warning('UI already created, skipping duplicate creation')
            return
        self._ui_created = True
        with ui.row().classes('w-full h-screen p-4 bg-gray-900 justify-between'):
            with ui.column().classes('w-96 flex-shrink-0'):
                with ui.card().classes('w-full bg-gray-800 text-white'):
                    ui.label('Factorio Learning Environment').classes('text-xl font-bold text-center')
                    self.status_label = ui.label('⚪ Initializing...').classes('text-sm text-center')
                    ui.input(label='Task', value=self.env_id.split('-')[1] if '-' in self.env_id else self.env_id).props('readonly').classes('w-full text-xs')
                    ui.input(label='Model', value=self.model.split('/')[-1] if '/' in self.model else self.model).props('readonly').classes('w-full text-xs')
                    self.progress_bar = ui.linear_progress(value=0).classes('w-full')
                    with ui.row().classes('w-full justify-between'):
                        self.step_label = ui.label('Step: 0 / 100').classes('text-xs')
                        self.tick_label = ui.label('Tick: 0').classes('text-xs')
                    with ui.row().classes('w-full items-center gap-2 mt-2'):
                        ui.label('Camera:').classes('text-xs')
                        ui.switch('Track', value=True).bind_value(self, 'camera_tracking_enabled').classes('text-xs')
                        ui.slider(min=0.1, max=2.0, step=0.1, value=0.5).bind_value(self, 'camera_zoom_scale').classes('flex-1')
                with ui.card().classes('w-full bg-gray-800 text-white mt-2'):
                    ui.label('Statistics').classes('text-sm font-bold')
                    with ui.row().classes('items-center gap-1'):
                        self.coin_icon = ui.image('').classes('w-4 h-4')
                        self.score_label = ui.html('0.00').classes('text-sm')
                    with ui.row().classes('w-full justify-between text-xs mt-1'):
                        self.entities_label = ui.label('Entities: 0')
                        self.research_label = ui.label('Research: 0%')
                with ui.card().classes('w-full bg-gray-800 text-white mt-2'):
                    ui.label('Inventory').classes('text-sm font-bold')
                    self.inventory_grid = ui.row().classes('w-full flex-wrap gap-3')
                    with self.inventory_grid:
                        ui.label('No items yet').classes('text-xs text-gray-400')
            ui.element('div').classes('flex-1')
            with ui.column().classes('w-[500px] flex-shrink-0'):
                with ui.card().classes('w-full h-full bg-gray-800 text-white'):
                    ui.label('Production Output').classes('text-sm font-bold mb-2')
                    self.chart = ui.echart({'backgroundColor': 'transparent', 'xAxis': {'type': 'category', 'data': [], 'axisLabel': {'color': 'rgba(255, 255, 255, 0.7)'}, 'axisLine': {'lineStyle': {'color': 'rgba(255, 255, 255, 0.3)'}}}, 'yAxis': {'type': 'log', 'logBase': 10, 'axisLabel': {'color': 'rgba(255, 255, 255, 0.7)'}, 'splitLine': {'lineStyle': {'color': 'rgba(255, 255, 255, 0.1)'}}, 'min': 1}, 'series': [], 'legend': {'data': [], 'textStyle': {'color': 'rgba(255, 255, 255, 0.7)'}, 'itemWidth': 20, 'itemHeight': 20}, 'grid': {'left': '5%', 'right': '5%', 'bottom': '5%', 'top': '15%', 'containLabel': True}, 'tooltip': {'trigger': 'axis', 'backgroundColor': 'rgba(0, 0, 0, 0.8)', 'borderColor': '#333', 'textStyle': {'color': '#fff'}}}).classes('w-full h-[500px]')
        ui.timer(0.5, self.process_updates)
        logger.info('Process updates timer set up (0.5s interval)')
        if self.auto_start:
            logger.info('Auto-start is enabled, setting up timer to start in 2 seconds')
            ui.timer(2.0, lambda: self._auto_start_callback(), once=True)
        else:
            logger.info('Auto-start is disabled')

    def _auto_start_callback(self):
        """Callback for auto-start timer."""
        logger.info('Auto-start timer triggered')
        self.start_run()

    def start_run(self):
        """Start a new run automatically."""
        logger.info('start_run() called - is_running: %s', self.is_running)
        if not self.is_running:
            self.is_running = True
            self.status_label.set_text('🟡 Starting...')
            self.current_step = 0
            self.cumulative_score = 0
            if hasattr(self, 'score_history'):
                self.score_history.clear()
            else:
                self.score_history = deque(maxlen=50)
            self.inventory_items = {}
            self.production_history = {'timestamps': deque(maxlen=50), 'data': {}}
            self.cumulative_sum = 0
            logger.info('Starting trajectory runner thread')
            self.runner_thread = threading.Thread(target=self._run_trajectory)
            self.runner_thread.daemon = True
            self.runner_thread.start()
            logger.info('Trajectory runner thread started - thread alive: %s', self.runner_thread.is_alive())
        else:
            logger.warning('start_run() called but already running')

    def _run_trajectory(self):
        """Run the trajectory in a background thread."""
        logger.info('_run_trajectory thread started')
        try:
            logger.info('Starting asyncio.run for _async_run_trajectory')
            asyncio.run(self._async_run_trajectory())
            logger.info('_async_run_trajectory completed successfully')
        except Exception as e:
            logger.error('Exception in _run_trajectory: %s', e, exc_info=True)
            self.update_queue.put({'type': 'error', 'message': str(e)})

    async def start_polling(self):
        """Start polling for game state updates continuously."""
        logger.info('Starting game state polling in monitoring mode')
        poll_count = 0
        while self.is_running:
            try:
                poll_count += 1
                logger.debug(f'Polling iteration {poll_count} at {time.time()}')
                if hasattr(self, 'gym_env') and self.gym_env:
                    await self._update_camera_viewport()
                    logger.debug(f'Polling update {poll_count} sent to queue')
                await asyncio.sleep(3.0)
            except Exception as e:
                logger.error(f'Error in polling loop (iteration {poll_count}): {e}')
                await asyncio.sleep(3.0)
        logger.info(f'Stopping polling after {poll_count} iterations')

    async def _update_camera_viewport(self):
        """Update camera viewport to follow agent character."""
        if not self.camera_tracking_enabled:
            return
        current_time = time.time()
        if current_time - self.last_camera_update < self.camera_update_interval:
            return
        try:
            camera_cmd = '/sc if game.players[1] and global.agent_characters and global.agent_characters[1] then game.players[1].teleport(global.agent_characters[1].position) end'
            self.rcon_client.send_command(camera_cmd)
            logger.debug('Camera viewport updated to follow agent character')
            self.last_camera_update = current_time
        except Exception as e:
            logger.debug('Error updating camera viewport: %s', e)

    async def _initialize_overlay_session(self):
        """Initialize overlay session with cleanup Lua commands."""
        logger.info('Initializing overlay session with cleanup commands')
        try:
            logger.info('Show all map')
            self.rcon_client.send_command('/sc game.players[1].force.chart_all()')
            logger.info('Clearing rendering artifacts')
            self.rcon_client.send_command('/c rendering.clear()')
            logger.info('Removing hostile entities')
            enemy_cleanup_cmd = '/c local surface=game.players[1].surface for key, entity in pairs(surface.find_entities_filtered({force="enemy"})) do entity.destroy() end'
            self.rcon_client.send_command(enemy_cleanup_cmd)
        except Exception as e:
            logger.error('Error during overlay session initialization: %s', e, exc_info=True)
            self.update_queue.put({'type': 'error', 'message': f'Initialization warning: {str(e)}'})

    async def _async_run_trajectory(self):
        """Setup monitoring connection without running trajectory."""
        logger.info('Setting up monitoring connection - env_id: %s', self.env_id)
        try:
            logger.info('Getting environment info for %s', self.env_id)
            env_info = get_environment_info(self.env_id)
            if not env_info:
                logger.error('Could not get environment info for %s', self.env_id)
                self.update_queue.put({'type': 'error', 'message': f'Could not get environment info for {self.env_id}'})
                return
            logger.info('Environment info retrieved successfully')
            logger.info('Creating gym environment for monitoring')
            await self._initialize_overlay_session()
            self.update_queue.put({'type': 'init', 'task': self.env_id, 'max_steps': 100, 'num_agents': 1})
            self.update_queue.put({'type': 'update', 'step': 0, 'output': 'Monitoring mode - use Claude Code to control Factorio'})
            logger.info('Starting monitoring polling')
            await self.start_polling()
            logger.info('Monitoring completed')
        except Exception as e:
            logger.error('Fatal error in monitoring setup: %s', e, exc_info=True)
            self.update_queue.put({'type': 'error', 'message': f'Fatal error: {str(e)}'})
            raise

def run_async_safely(coro):
    """Run an async coroutine safely, handling both cases where event loop exists or not"""
    try:
        asyncio.get_running_loop()
        return _run_async_in_new_thread(coro)
    except RuntimeError:
        return asyncio.run(coro)

def _run_async_in_new_thread(coro):
    """Run an async coroutine in a new thread with its own event loop"""

    def run_in_thread():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()

def make_factorio_env(spec: GymEnvironmentSpec, run_idx: int) -> FactorioGymEnv:
    """Factory function to create a Factorio gym environment"""
    task = TaskFactory.create_task(spec.task_config_path)
    try:
        address = os.getenv('FACTORIO_SERVER_ADDRESS')
        tcp_port = os.getenv('FACTORIO_SERVER_PORT')
        if not address and (not tcp_port):
            try:
                ips, udp_ports, tcp_ports = get_local_container_ips()
            except ValueError:
                raise RuntimeError('No Factorio containers available')
            if len(tcp_ports) == 0:
                raise RuntimeError('No Factorio containers available')
            container_idx = PORT_OFFSET + run_idx
            if container_idx >= len(tcp_ports):
                raise RuntimeError(f'Container index {container_idx} (PORT_OFFSET={PORT_OFFSET} + run_idx={run_idx}) exceeds available containers ({len(tcp_ports)})')
            address = ips[container_idx]
            tcp_port = tcp_ports[container_idx]
        common_kwargs = {'address': address, 'tcp_port': int(tcp_port), 'num_agents': spec.num_agents, 'fast': True, 'cache_scripts': True, 'inventory': {}, 'all_technologies_researched': True}
        print(f'Using local Factorio container at {address}:{tcp_port}')
        if spec.num_agents > 1:
            instance = run_async_safely(A2AFactorioInstance.create(**common_kwargs))
        else:
            instance = FactorioInstance(**common_kwargs)
        instance.set_speed_and_unpause(10)
        task.setup(instance)
        env = FactorioGymEnv(instance=instance, task=task, enable_vision=spec.enable_vision)
        return env
    except Exception as e:
        raise RuntimeError(f'Failed to create Factorio environment: {e}')

class FactorioMCPState:
    """Manages the state of the Factorio MCP server"""

    def __init__(self):
        self.available_servers: Dict[int, FactorioServer] = {}
        self.active_server: Optional[FactorioInstance] = None
        self.server_entities: Dict[int, Dict[str, Any]] = {}
        self.server_resources: Dict[int, Dict[str, ResourcePatch]] = {}
        self.recipes: Dict[str, Recipe] = {}
        self.recipes_loaded = False
        self.checkpoints: Dict[int, Dict[str, str]] = {}
        self.current_task: Optional[str] = None
        self.last_entity_update = 0
        self.vcs_repos: Dict[int, 'FactorioMCPRepository'] = {}
        try:
            env_ids = list_available_environments()
            if not env_ids:
                raise Exception('No environments found')
            for id in env_ids:
                if 'open' in id:
                    print(f'DEBUG: Using open environment: {id}')
                    self.gym_env = gym.make(id, run_idx=0)
                    self.gym_env.reset()
                    return
            self.gym_env = gym.make(env_ids[0], run_idx=0)
        except IndexError as e:
            print(f'IndexError in __init__: {e}')
            print(f'env_ids length: {(len(env_ids) if 'env_ids' in locals() else 'Not available')}')
            print('Falling back to steel_plate_throughput environment')
            self.gym_env = gym.make('steel_plate_throughput', run_idx=0)
        except Exception as e:
            print(f'Error in __init__: {e}')
            print(f'Error type: {type(e)}')
            print('Falling back to steel_plate_throughput environment')
            self.gym_env = gym.make('steel_plate_throughput', run_idx=0)
        self.gym_env.reset()

    def create_factorio_instance(self, instance_id: int) -> FactorioInstance:
        """Create a single Factorio instance"""
        try:
            ips, udp_ports, tcp_ports = get_local_container_ips()
            if instance_id >= len(ips):
                raise IndexError(f'instance_id {instance_id} out of range for ips list of length {len(ips)}')
            if instance_id >= len(tcp_ports):
                raise IndexError(f'instance_id {instance_id} out of range for tcp_ports list of length {len(tcp_ports)}')
            instance = FactorioInstance(address=ips[instance_id], tcp_port=tcp_ports[instance_id], bounding_box=200, fast=True, cache_scripts=True, inventory={'stone-furnace': 1, 'burner-mining-drill': 1, 'wood': 5, 'iron-plate': 8}, all_technologies_researched=False)
            char_check = instance.rcon_client.send_command('/c rcon.print(global.agent_characters and #global.agent_characters or 0)')
            if int(char_check) == 0:
                instance.first_namespace._create_agent_characters(1)
            instance.set_speed(10)
            return instance
        except IndexError as e:
            print(f'IndexError in create_factorio_instance: {e}')
            try:
                print(f'Available IPs: {ips}')
                print(f'Available TCP ports: {tcp_ports}')
            except NameError:
                print('ERROR: Could not retrieve container IPs/ports')
            raise e
        except Exception as e:
            print(f'Error creating Factorio instance: {e}')
            print(f'Error type: {type(e)}')
            raise e

    async def scan_for_servers(self, ctx=None) -> List[FactorioServer]:
        """Scan for running Factorio servers"""
        try:
            ips, udp_ports, tcp_ports = get_local_container_ips()
            new_servers = {}
            for i in range(len(ips)):
                if ctx:
                    await ctx.report_progress(i, len(ips))
                instance_id = i
                if instance_id in self.available_servers:
                    server = self.available_servers[instance_id]
                    server.last_checked = time.time()
                    server.address = ips[i]
                    server.tcp_port = tcp_ports[i]
                    if not server.is_active:
                        try:
                            self.create_factorio_instance(i)
                            server.is_active = True
                        except Exception as e:
                            server.is_active = False
                            server.system_response = str(e)
                            print(str(e))
                    new_servers[instance_id] = server
                else:
                    server = FactorioServer(address=ips[i], tcp_port=int(tcp_ports[i]), instance_id=instance_id, name=f'Factorio Server {i + 1}', last_checked=time.time())
                    try:
                        self.create_factorio_instance(i)
                        server.is_active = True
                    except Exception as e:
                        server.is_active = False
                        server.system_response = str(e)
                    new_servers[instance_id] = server
                    if instance_id not in self.checkpoints:
                        self.checkpoints[instance_id] = {}
            self.available_servers = new_servers
            return list(self.available_servers.values())
        except Exception as e:
            raise e

    async def connect_to_server(self, instance_id: int) -> bool:
        """Connect to a Factorio server by instance ID"""
        if instance_id not in self.available_servers:
            return False
        server = self.available_servers[instance_id]
        if not server.is_active:
            return False
        try:
            instance = self.create_factorio_instance(instance_id)
            server.connected = True
            self.active_server = instance
            await self.refresh_game_data(instance_id)
            if not self.recipes:
                self.recipes = self.load_recipes_from_file()
            if instance_id not in self.vcs_repos:
                print('Initializing repo')
                self.vcs_repos[instance_id] = FactorioMCPRepository(instance)
            return True
        except Exception as e:
            print(f'Error connecting to Factorio server: {e}')
            return False

    def get_vcs(self):
        """Get the VCS repository for the active server"""
        if not self.active_server:
            return None
        instance_id = self.active_server.tcp_port
        if instance_id not in self.vcs_repos:
            self.vcs_repos[instance_id] = FactorioMCPRepository(self.active_server)
        return self.vcs_repos[instance_id]

    async def refresh_game_data(self, instance_id: int):
        """Refresh game data for a specific server instance"""
        if instance_id not in self.available_servers:
            return False
        self.last_entity_update = time.time()
        return True

    def load_recipes_from_file(self) -> Dict[str, Recipe]:
        """Load recipes from the jsonl file"""
        if self.recipes_loaded:
            return self.recipes
        recipes_path = Path(__file__).parent.parent / 'data' / 'recipes' / 'recipes.jsonl'
        if not recipes_path.exists():
            recipes_path = Path('/Users/jackhopkins/PycharmProjects/PaperclipMaximiser/data/recipes/recipes.jsonl')
        try:
            recipes = {}
            with open(recipes_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            recipe_data = json.loads(line)
                            ingredients = recipe_data.get('ingredients', [])
                            simplified_ingredients = []
                            for ingredient in ingredients:
                                simplified_ingredients.append({'name': ingredient.get('name', ''), 'amount': ingredient.get('amount', 1)})
                            results = [{'name': recipe_data.get('name', ''), 'amount': 1}]
                            recipes[recipe_data['name']] = Recipe(name=recipe_data['name'], ingredients=simplified_ingredients, results=results, energy_required=1.0)
                        except json.JSONDecodeError:
                            print(f'Warning: Could not parse recipe line: {line}')
                        except KeyError as e:
                            print(f'Warning: Missing key in recipe: {e}')
                        except Exception as e:
                            print(f'Warning: Error processing recipe: {e}')
            self.recipes_loaded = True
            return recipes
        except Exception as e:
            print(f'Error loading recipes from file: {e}')
            raise e

def create_factorio_instances() -> List[FactorioInstance]:

    def init_instance(params: Tuple[str, int, int]) -> FactorioInstance:
        ip, udp_port, tcp_port = params
        return FactorioInstance(address=ip, tcp_port=tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={}, all_technologies_researched=False)
    ips, udp_ports, tcp_ports = get_local_container_ips()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(init_instance, zip(ips, udp_ports, tcp_ports)))

def create_factorio_instances() -> List[FactorioInstance]:
    """Create Factorio instances in parallel from local servers"""

    def init_instance(params: Tuple[str, int, int]) -> FactorioInstance:
        ip, udp_port, tcp_port = params
        return FactorioInstance(address=ip, tcp_port=tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={})
    ips, udp_ports, tcp_ports = get_local_container_ips()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(init_instance, zip(ips, udp_ports, tcp_ports)))

class ServerManager:
    """Manages allocation of Factorio servers across concurrent jobs"""

    def __init__(self, max_allocation_time_hours: float=2.0):
        """Initialize server manager

        Args:
            max_allocation_time_hours: Maximum time to hold a server allocation
        """
        self._lock = threading.Lock()
        self._allocations: Dict[int, ServerAllocation] = {}
        self._available_servers: List[int] = []
        self._allocated_servers: Set[int] = set()
        self.max_allocation_time = timedelta(hours=max_allocation_time_hours)
        self._initialized = False

    def _discover_servers(self) -> bool:
        """Discover available Factorio servers

        Returns:
            True if servers were found, False otherwise
        """
        try:
            ips, udp_ports, tcp_ports = get_local_container_ips()
            if not tcp_ports:
                print('⚠️  No Factorio containers found')
                return False
            self._available_servers = list(range(len(tcp_ports)))
            self._server_info = {i: {'address': ips[i], 'tcp_port': tcp_ports[i], 'udp_port': udp_ports[i]} for i in range(len(tcp_ports))}
            print(f'🖥️  Discovered {len(tcp_ports)} Factorio servers:')
            for i, (ip, tcp_port) in enumerate(zip(ips, tcp_ports)):
                print(f'   Server {i}: {ip}:{tcp_port}')
            self._initialized = True
            return True
        except Exception as e:
            print(f'❌ Error discovering servers: {e}')
            return False

    def initialize(self) -> bool:
        """Initialize server discovery

        Returns:
            True if initialization was successful
        """
        with self._lock:
            if not self._initialized:
                return self._discover_servers()
            return True

    def get_available_server_count(self) -> int:
        """Get number of currently available servers"""
        with self._lock:
            if not self._initialized:
                self._discover_servers()
            return len(self._available_servers) - len(self._allocated_servers)

    def get_total_server_count(self) -> int:
        """Get total number of discovered servers"""
        with self._lock:
            if not self._initialized:
                self._discover_servers()
            return len(self._available_servers)

    def allocate_server(self, job_id: str, process_id: Optional[int]=None) -> Optional[ServerAllocation]:
        """Allocate a server for a job

        Args:
            job_id: Unique identifier for the job
            process_id: Optional process ID for tracking

        Returns:
            ServerAllocation if successful, None if no servers available
        """
        with self._lock:
            if not self._initialized:
                if not self._discover_servers():
                    return None
            self._cleanup_expired_allocations()
            available_servers = [server_id for server_id in self._available_servers if server_id not in self._allocated_servers]
            if not available_servers:
                print(f'⚠️  No servers available for job {job_id} (all {len(self._available_servers)} servers allocated)')
                return None
            server_id = available_servers[0]
            server_info = self._server_info[server_id]
            allocation = ServerAllocation(server_id=server_id, server_address=server_info['address'], tcp_port=server_info['tcp_port'], udp_port=server_info['udp_port'], job_id=job_id, allocated_at=datetime.now(), process_id=process_id)
            self._allocations[server_id] = allocation
            self._allocated_servers.add(server_id)
            print(f'🖥️  Allocated server {server_id} ({allocation.server_address}:{allocation.tcp_port}) to job {job_id}')
            return allocation

    def release_server(self, job_id: str) -> bool:
        """Release server allocation for a job

        Args:
            job_id: Job identifier to release

        Returns:
            True if server was found and released
        """
        with self._lock:
            server_id = None
            for sid, allocation in self._allocations.items():
                if allocation.job_id == job_id:
                    server_id = sid
                    break
            if server_id is not None:
                allocation = self._allocations.pop(server_id)
                self._allocated_servers.remove(server_id)
                print(f'🔓 Released server {server_id} from job {job_id}')
                return True
            return False

    def release_server_by_id(self, server_id: int) -> bool:
        """Release server allocation by server ID

        Args:
            server_id: Server ID to release

        Returns:
            True if server was found and released
        """
        with self._lock:
            if server_id in self._allocations:
                allocation = self._allocations.pop(server_id)
                self._allocated_servers.remove(server_id)
                print(f'🔓 Released server {server_id} (was allocated to {allocation.job_id})')
                return True
            return False

    def _cleanup_expired_allocations(self):
        """Clean up allocations that have been held too long (called with lock held)"""
        current_time = datetime.now()
        expired_servers = []
        for server_id, allocation in self._allocations.items():
            if current_time - allocation.allocated_at > self.max_allocation_time:
                expired_servers.append(server_id)
        for server_id in expired_servers:
            allocation = self._allocations.pop(server_id)
            self._allocated_servers.remove(server_id)
            print(f'⏰ Released expired allocation: server {server_id} (was allocated to {allocation.job_id})')

    def get_allocation_status(self) -> Dict:
        """Get current allocation status

        Returns:
            Dictionary with allocation information
        """
        with self._lock:
            if not self._initialized:
                self._discover_servers()
            self._cleanup_expired_allocations()
            return {'total_servers': len(self._available_servers), 'allocated_servers': len(self._allocated_servers), 'available_servers': len(self._available_servers) - len(self._allocated_servers), 'allocations': [allocation.to_dict() for allocation in self._allocations.values()], 'initialized': self._initialized}

    def get_server_assignment_for_job(self, job_id: str) -> Optional[Dict]:
        """Get server assignment for a specific job

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with server info or None if not found
        """
        with self._lock:
            for allocation in self._allocations.values():
                if allocation.job_id == job_id:
                    return {'server_id': allocation.server_id, 'address': allocation.server_address, 'tcp_port': allocation.tcp_port, 'udp_port': allocation.udp_port, 'allocated_at': allocation.allocated_at.isoformat()}
            return None

    def force_release_all(self):
        """Force release all server allocations (emergency cleanup)"""
        with self._lock:
            released_count = len(self._allocations)
            self._allocations.clear()
            self._allocated_servers.clear()
            if released_count > 0:
                print(f'🧹 Force released all {released_count} server allocations')

    def print_status(self):
        """Print current server allocation status"""
        status = self.get_allocation_status()
        print('🖥️  Server Allocation Status:')
        print(f'   Total servers: {status['total_servers']}')
        print(f'   Available: {status['available_servers']}')
        print(f'   Allocated: {status['allocated_servers']}')
        if status['allocations']:
            print('   Current allocations:')
            for alloc in status['allocations']:
                print(f'     Server {alloc['server_id']}: {alloc['job_id']} (since {alloc['allocated_at'][:19]})')

def create_factorio_instance(instance_id: int) -> FactorioInstance:
    """Create a single Factorio instance"""
    ips, udp_ports, tcp_ports = get_local_container_ips()
    instance = FactorioInstance(address=ips[instance_id], tcp_port=tcp_ports[instance_id], bounding_box=200, fast=True, cache_scripts=True, inventory={}, all_technologies_researched=True)
    instance.set_speed_and_unpause(10)
    return instance

def create_factorio_instances(start_index: int, count: int) -> List[FactorioInstance]:
    """Create Factorio instances with proper resource management"""
    ips, udp_ports, tcp_ports = get_local_container_ips()
    ips = ips[start_index:start_index + count]
    udp_ports = udp_ports[start_index:start_index + count]
    tcp_ports = tcp_ports[start_index:start_index + count]
    instances = []
    errors = []
    for ip, udp_port, tcp_port in zip(ips, udp_ports, tcp_ports):
        try:
            instance = FactorioInstance(address=ip, tcp_port=tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={}, all_technologies_researched=False)
            instance.set_speed(10)
            instances.append(instance)
        except Exception as e:
            errors.append(f'Failed to create instance at {ip}:{tcp_port} - {str(e)}')
    if errors:
        raise RuntimeError(f'Failed to create all instances: {'; '.join(errors)}')
    if not instances:
        raise RuntimeError('No instances were created successfully')
    return instances

def create_factorio_instances() -> List[FactorioInstance]:
    """Create Factorio instances in parallel from local servers"""

    def init_instance(params: Tuple[str, int, int]) -> FactorioInstance:
        ip, udp_port, tcp_port = params
        return FactorioInstance(address=ip, tcp_port=tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={})
    ips, udp_ports, tcp_ports = get_local_container_ips()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(init_instance, zip(ips, udp_ports, tcp_ports)))

def create_factorio_instances() -> List[FactorioInstance]:

    def init_instance(params: Tuple[str, int, int]) -> FactorioInstance:
        ip, udp_port, tcp_port = params
        return FactorioInstance(address=ip, tcp_port=tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={})
    ips, udp_ports, tcp_ports = get_local_container_ips()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(init_instance, zip(ips, udp_ports, tcp_ports)))

def create_factorio_instances() -> List[FactorioInstance]:
    """Create Factorio instances in parallel from local servers"""

    def init_instance(params: Tuple[str, int, int]) -> FactorioInstance:
        ip, udp_port, tcp_port = params
        return FactorioInstance(address=ip, tcp_port=tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={})
    ips, udp_ports, tcp_ports = get_local_container_ips()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(init_instance, zip(ips, udp_ports, tcp_ports)))

def test_drop_box_chest():
    ips, udp_ports, tcp_ports = get_local_container_ips()
    instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=tcp_ports[-1], fast=True, inventory={'burner-mining-drill': 1, 'iron-chest': 1, 'coal': 10})
    instance.get_system_prompt()
    instance.namespace.move_to(instance.namespace.nearest(Resource.IronOre))
    drill = instance.namespace.place_entity(Prototype.BurnerMiningDrill, Direction.UP, instance.namespace.nearest(Resource.IronOre))
    instance.namespace.place_entity(Prototype.IronChest, Direction.UP, drill.drop_position)
    instance.namespace.insert_item(Prototype.Coal, drill, 10)
    instance.namespace.sleep(10)
    drill = instance.namespace.get_entities({Prototype.BurnerMiningDrill})[0]
    state = GameState.from_instance(instance)
    instance.reset(state)
    drill = instance.namespace.get_entities({Prototype.BurnerMiningDrill})[0]
    assert not drill.warnings

def test_full_chest():
    ips, udp_ports, tcp_ports = get_local_container_ips()
    instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=tcp_ports[-1], fast=True, inventory={'burner-mining-drill': 1, 'wooden-chest': 1, 'coal': 2000})
    chest = instance.namespace.place_entity(Prototype.WoodenChest, Direction.UP)
    for i in range(16):
        instance.namespace.insert_item(Prototype.Coal, chest, 50)
    state = GameState.from_instance(instance)
    instance.reset(state)
    chest = instance.namespace.get_entities({Prototype.WoodenChest})[0]
    assert chest.warnings[0] == 'chest is full'

def test_nested_functions():
    ips, udp_ports, tcp_ports = get_local_container_ips()
    instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=tcp_ports[-1], fast=True, inventory={})
    score, goal, result = instance.eval_with_error('print(inspect_inventory())')
    assert result[3:] == '(Inventory({}),)'
    score, goal, result = instance.eval_with_error(embedded_function)
    assert result[3:] == '(Inventory({}),)'

@pytest.fixture(scope='session')
def instance(pytestconfig, worker_id):
    ips, udp_ports, tcp_ports = get_local_container_ips()
    xdist_count_env = os.environ.get('PYTEST_XDIST_WORKER_COUNT')
    try:
        opt_numproc = pytestconfig.getoption('numprocesses')
    except Exception:
        opt_numproc = None
    if xdist_count_env and xdist_count_env.isdigit():
        num_workers = int(xdist_count_env)
    elif isinstance(opt_numproc, int) and opt_numproc > 0:
        num_workers = opt_numproc
    else:
        num_workers = 1
    if worker_id == 'master':
        worker_index = 0
    elif worker_id.startswith('gw') and worker_id[2:].isdigit():
        worker_index = int(worker_id[2:])
    else:
        worker_index = 0
    ports_sorted = sorted(tcp_ports)
    if num_workers > 1:
        if len(ports_sorted) < num_workers:
            raise pytest.UsageError(f"pytest -n {num_workers} requested, but only {len(ports_sorted)} Factorio TCP ports were found: {ports_sorted}. Start {num_workers} servers, e.g. './run-envs.sh start -n {num_workers}'.")
        selected_port = ports_sorted[worker_index]
    else:
        port_env = os.getenv('FACTORIO_RCON_PORT')
        if port_env:
            selected_port = int(port_env)
        else:
            if not ports_sorted:
                raise pytest.UsageError('No Factorio TCP ports discovered. Did you start the headless server?')
            selected_port = ports_sorted[-1]
    try:
        instance = FactorioInstance(address='localhost', all_technologies_researched=True, tcp_port=selected_port, cache_scripts=True, fast=True, inventory={'coal': 50, 'copper-plate': 50, 'iron-plate': 50, 'iron-chest': 2, 'burner-mining-drill': 3, 'electric-mining-drill': 1, 'assembling-machine-1': 1, 'stone-furnace': 9, 'transport-belt': 50, 'boiler': 1, 'burner-inserter': 32, 'pipe': 15, 'steam-engine': 1, 'small-electric-pole': 10, 'fast-transport-belt': 10, 'express-transport-belt': 10})
        instance.set_speed(10.0)
        try:
            instance.default_initial_inventory = dict(instance.initial_inventory)
        except Exception:
            instance.default_initial_inventory = instance.initial_inventory
        yield instance
    except Exception as e:
        raise e
    finally:
        if 'instance' in locals():
            instance.cleanup()

class TestMCPResources:
    """Integration tests for MCP resources"""

    @classmethod
    def setup_class(cls):
        """Setup test fixtures once for all tests"""
        cls.instances = cls.create_factorio_instances()
        cls.test_instance = cls.instances[0] if cls.instances else None

    @classmethod
    def teardown_class(cls):
        """Cleanup after all tests"""
        if cls.instances:
            for instance in cls.instances:
                try:
                    instance.close()
                except:
                    pass

    @staticmethod
    def create_factorio_instances() -> List[FactorioInstance]:
        """Create Factorio instances in parallel from local servers"""

        def init_instance(params: Tuple[str, int, int]) -> FactorioInstance:
            ip, udp_port, tcp_port = params
            try:
                instance = FactorioInstance(address=ip, tcp_port=tcp_port, bounding_box=200, fast=True, cache_scripts=False, inventory={})
            except Exception as e:
                raise e
            instance.set_speed(100)
            return instance
        ips, udp_ports, tcp_ports = get_local_container_ips()
        with futures.ThreadPoolExecutor() as executor:
            return list(executor.map(init_instance, zip(ips, udp_ports, tcp_ports)))

    @pytest.fixture(autouse=True)
    async def setup_state(self):
        """Setup state before each test"""
        if self.test_instance:
            state.active_server = self.test_instance
            state.available_servers = {self.test_instance.tcp_port: MagicMock(name='TestServer', address='127.0.0.1', tcp_port=self.test_instance.tcp_port)}
            await initialize_session(None)
        yield
        state.active_server = None
        state.available_servers = {}

    @pytest.mark.asyncio
    async def test_render_with_position_default(self):
        """Test render_with_position resource with default position"""
        await reconnect.run({})
        resource = await render_at.create_resource('fle://render/', {'center_x': '0', 'center_y': '0'})
        result = await resource.read()
        assert result is not None
        if isinstance(result, ImageContent):
            assert result.type == 'image'
            assert result.mimeType == 'image/png'
            assert hasattr(result, 'data')
            if os.getenv('DISPLAY_TEST_IMAGES', 'false').lower() == 'true':
                image_data = base64.b64decode(result.data)
                img = PILImage.open(io.BytesIO(image_data))
                print(f'\nImage dimensions: {img.size}, mode: {img.mode}')
                img.show()
        elif isinstance(result, (str, bytes)):
            print(f'Render returned: {type(result)}')

    @pytest.mark.asyncio
    async def test_render_with_position_custom(self):
        """Test render_with_position with custom center coordinates"""
        await reconnect.run({})
        resource = await render_at.create_resource('fle://render/', {'center_x': '10.0', 'center_y': '10.0'})
        result = await resource.read()
        assert result is not None

    @pytest.mark.asyncio
    async def test_entities_resource_default(self):
        """Test entities resource with default parameters"""
        await reconnect.run({})
        resource = await entities.create_resource('fle://entities/', {'center_x': 'default', 'center_y': 'default', 'radius': 'default'})
        result = await resource.read()
        result = json.loads(result)
        assert result is not None
        assert isinstance(result, list)
        if result:
            first_entity = result[0]
            assert isinstance(first_entity, dict)

    @pytest.mark.asyncio
    async def test_entities_resource_custom(self):
        """Test entities resource with custom parameters"""
        await reconnect.run({})
        resource = await entities.create_resource('fle://entities/', {'center_x': '50.0', 'center_y': '50.0', 'radius': '100.0'})
        result = await resource.read()
        assert result is not None
        assert result == '[]'

    @pytest.mark.asyncio
    async def test_inventory_resource(self):
        """Test inventory resource retrieves current inventory"""
        await reconnect.run({})
        resource = inventory
        result = await resource.read()
        assert result is not None
        assert result[0] == '{'

    @pytest.mark.asyncio
    async def test_position_resource(self):
        """Test position resource gets player position"""
        await reconnect.run({})
        resource = position
        result = await resource.read()
        result = json.loads(result)
        assert result is not None
        assert isinstance(result, dict)
        assert 'x' in result
        assert 'y' in result
        assert isinstance(result['x'], (int, float))
        assert isinstance(result['y'], (int, float))

    @pytest.mark.asyncio
    async def test_entity_names_resource(self):
        """Test entity_names resource retrieves available entity prototypes"""
        await reconnect.run({})
        resource = prototypes
        result = await resource.read()
        result = json.loads(result)
        assert result is not None
        assert isinstance(result, list)
        if result:
            assert all((isinstance(name, str) for name in result))

    @pytest.mark.asyncio
    async def test_recipe_resource(self):
        """Test recipe resource with specific recipe name"""
        await reconnect.run({})
        names_resource = prototypes
        available_names = await names_resource.read()
        available_names = json.loads(available_names)
        if available_names and len(available_names) > 0:
            test_recipe_name = available_names[0]
            resource = await recipe.create_resource('fle://recipe/', {'name': test_recipe_name})
            result = await resource.read()
            assert result is not None
            assert isinstance(result, str)
            if 'not found' not in result:
                recipe_data = json.loads(result)
                assert 'name' in recipe_data
                assert 'ingredients' in recipe_data
                assert 'results' in recipe_data
                assert 'energy_required' in recipe_data
        invalid_resource = await recipe.create_resource('fle://recipe/', {'name': 'non_existent_recipe_xyz'})
        result_invalid = await invalid_resource.read()
        assert result_invalid is not None
        assert 'not found' in result_invalid

    @pytest.mark.asyncio
    async def test_schema_resource(self):
        """Test schema resource returns API documentation"""
        await reconnect.run({})
        resource = schema
        result = await resource.read()
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 100
        assert any((keyword in result.lower() for keyword in ['type', 'entity', 'class', 'def']))

    @pytest.mark.asyncio
    async def test_manual_resource(self):
        """Test manual resource with specific method name"""
        await reconnect.run({})
        test_methods = ['move_to', 'place_entity', 'craft_item']
        for method_name in test_methods:
            try:
                resource = await manual.create_resource('fle://api/manual/', {'method': method_name})
                result = await resource.read()
                assert result is not None
                assert isinstance(result, str)
                if 'Error' not in result:
                    assert len(result) > 50
                    break
            except Exception:
                continue
        invalid_resource = await manual.create_resource('fle://api/manual/', {'method': 'non_existent_method_xyz'})
        result_invalid = await invalid_resource.read()
        assert result_invalid is not None
        assert isinstance(result_invalid, str)
        assert 'Error' in result_invalid or 'not a valid' in result_invalid

    @pytest.mark.asyncio
    async def test_status_resource(self):
        """Test status resource checks server connection"""
        resource = status
        result = await resource.read()
        assert result is not None
        assert isinstance(result, str)
        assert 'Connected to Factorio server' in result or 'Initializing' in result

    @pytest.mark.asyncio
    async def test_resource_error_handling_no_connection(self):
        """Test resource error handling when server is not connected"""
        state.active_server = None
        inv_resource = inventory
        with pytest.raises(Exception, match='No active Factorio server connection'):
            await inv_resource.read()
        pos_resource = position
        with pytest.raises(Exception, match='No active Factorio server connection'):
            await pos_resource.read()
        ent_resource = await entities.create_resource('fle://entities/', {'center_x': '0', 'center_y': '0', 'radius': '100'})
        with pytest.raises(Exception, match='No active Factorio server connection'):
            await ent_resource.read()

    @pytest.mark.asyncio
    async def test_entities_parameter_conversion(self):
        """Test entities resource properly converts string parameters"""
        await reconnect.run({})
        test_cases = [{'center_x': '0', 'center_y': '0', 'radius': '500'}, {'center_x': '10.5', 'center_y': '-20.5', 'radius': '100.0'}, {'center_x': 'default', 'center_y': 'default', 'radius': 'default'}]
        for params in test_cases:
            resource = await entities.create_resource('fle://entities/', params)
            result = await resource.read()
            result = json.loads(result)
            assert result is not None
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_render_parameter_conversion(self):
        """Test render_with_position properly converts string parameters"""
        await reconnect.run({})
        test_cases = [{'center_x': '0', 'center_y': '0'}, {'center_x': '10.5', 'center_y': '-20.5'}, {'center_x': '-100', 'center_y': '100'}]
        for params in test_cases:
            resource = await render_at.create_resource('fle://render/', params)
            result = await resource.read()
            assert result is not None

    @pytest.mark.asyncio
    async def test_recipe_names_consistency(self):
        """Test that entity_names and recipe lookups are consistent"""
        await reconnect.run({})
        names_resource = prototypes
        names = await names_resource.read()
        names = json.loads(names)
        assert names is not None
        if len(names) > 5:
            sample_names = names[:5]
            for name in sample_names:
                resource = await recipe.create_resource('fle://recipe/', {'name': name})
                result = await resource.read()
                assert result is not None
                assert 'not found' not in result
                recipe_data = json.loads(result)
                assert recipe_data['name'] == name

    @pytest.mark.asyncio
    async def test_resources_without_params(self):
        """Test resources that don't take parameters"""
        await reconnect.run({})
        no_param_resources = [(inventory, 'fle://inventory', dict), (position, 'fle://position', dict), (prototypes, 'fle://prototypes', list), (schema, 'fle://api/schema', str), (status, 'fle://status', str)]
        for resource_template, uri, expected_type in no_param_resources:
            resource = await resource_template.create_resource(uri, {})
            result = await resource.read()
            assert result is not None
            assert isinstance(result, expected_type), f'{resource_template} should return {expected_type}, got {type(result)}'

    @pytest.mark.asyncio
    async def test_resources_with_path_params(self):
        """Test resources that take path parameters"""
        await reconnect.run({})
        entities_resource = await entities.create_resource('fle://entities/', {'center_x': '0', 'center_y': '0', 'radius': '100'})
        entities_result = await entities_resource.read()
        assert isinstance(entities_result, list)
        render_resource = await render_at.create_resource('fle://render/', {'center_x': '0', 'center_y': '0'})
        render_result = await render_resource.read()
        assert render_result is not None
        names_resource = await prototypes.create_resource('fle://prototypes', {})
        names = await names_resource.read()
        if names:
            recipe_resource = await recipe.create_resource('fle://recipe/', {'name': names[0]})
            recipe_result = await recipe_resource.read()
            assert isinstance(recipe_result, str)
        manual_resource = await manual.create_resource('fle://api/manual/', {'method': 'move_to'})
        manual_result = await manual_resource.read()
        assert isinstance(manual_result, str)

@pytest.fixture()
def game():
    instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
    instance.set_speed(1)
    instance.reset()
    yield instance.namespace

@pytest.fixture
def multi_instance(instance):
    """Creates a second FactorioInstance with a different player_index"""
    single_instance = instance
    inventory = single_instance.initial_inventory
    multi_instance = asyncio.run(A2AFactorioInstance.create(address='localhost', bounding_box=200, tcp_port=single_instance.tcp_port, cache_scripts=False, fast=True, inventory=inventory, num_agents=2))
    yield multi_instance
    multi_instance.reset()

@pytest.fixture()
def game(instance):
    initial_inventory = {'coal': 50, 'copper-plate': 50, 'iron-plate': 50, 'iron-chest': 2, 'burner-mining-drill': 3, 'assembling-machine-1': 1, 'boiler': 1, 'steam-engine': 1, 'stone-furnace': 10, 'burner-inserter': 32, 'offshore-pump': 4, 'pipe': 100, 'small-electric-pole': 50, 'transport-belt': 100, 'lab': 1, 'automation-science-pack': 10}
    ips, udp_ports, tcp_ports = get_local_container_ips()
    instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=tcp_ports[-1], fast=True, all_technologies_researched=False, inventory=initial_inventory)
    instance.reset()
    yield instance.namespace

