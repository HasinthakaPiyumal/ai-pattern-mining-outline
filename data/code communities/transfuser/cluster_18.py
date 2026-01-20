# Cluster 18

class WorldSR(World):
    restarted = False

    def restart(self):
        if self.restarted:
            return
        self.restarted = True
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        while self.player is None:
            print('Waiting for the ego vehicle...')
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == 'hero':
                    print('Ego vehicle found')
                    self.player = vehicle
                    break
        self.player_name = self.player.type_id
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def tick(self, clock):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            return False
        self.hud.tick(self, clock)
        return True

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return name[:truncate - 1] + u'…' if len(name) > truncate else name

class ModuleWorld(object):

    def __init__(self, name, args, timeout):
        self.client = None
        self.name = name
        self.args = args
        self.timeout = timeout
        self.server_fps = 0.0
        self.simulation_time = 0
        self.server_clock = pygame.time.Clock()
        self.world = None
        self.town_map = None
        self.actors_with_transforms = []
        self.module_hud = None
        self.module_input = None
        self.surface_size = [0, 0]
        self.prev_scaled_size = 0
        self.scaled_size = 0
        self.hero_actor = None
        self.spawned_hero = None
        self.hero_transform = None
        self.scale_offset = [0, 0]
        self.vehicle_id_surface = None
        self.result_surface = None
        self.traffic_light_surfaces = TrafficLightSurfaces()
        self.affected_traffic_light = None
        self.map_image = None
        self.border_round_surface = None
        self.original_surface_size = None
        self.hero_surface = None
        self.actors_surface = None

    def _get_data_from_carla(self):
        try:
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(self.timeout)
            if self.args.map is None:
                world = self.client.get_world()
            else:
                world = self.client.load_world(self.args.map)
            town_map = world.get_map()
            return (world, town_map)
        except RuntimeError as ex:
            logging.error(ex)
            exit_game()

    def start(self):
        self.world, self.town_map = self._get_data_from_carla()
        self.map_image = MapImage(carla_world=self.world, carla_map=self.town_map, pixels_per_meter=PIXELS_PER_METER, show_triggers=self.args.show_triggers, show_connections=self.args.show_connections, show_spawn_points=self.args.show_spawn_points)
        self.module_hud = module_manager.get_module(MODULE_HUD)
        self.module_input = module_manager.get_module(MODULE_INPUT)
        self.original_surface_size = min(self.module_hud.dim[0], self.module_hud.dim[1])
        self.surface_size = self.map_image.big_map_surface.get_width()
        self.scaled_size = int(self.surface_size)
        self.prev_scaled_size = int(self.surface_size)
        self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(COLOR_BLACK)
        self.vehicle_id_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.vehicle_id_surface.set_colorkey(COLOR_BLACK)
        self.border_round_surface = pygame.Surface(self.module_hud.dim, pygame.SRCALPHA).convert()
        self.border_round_surface.set_colorkey(COLOR_WHITE)
        self.border_round_surface.fill(COLOR_BLACK)
        center_offset = (int(self.module_hud.dim[0] / 2), int(self.module_hud.dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_ALUMINIUM_1, center_offset, int(self.module_hud.dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_WHITE, center_offset, int((self.module_hud.dim[1] - 8) / 2))
        scaled_original_size = self.original_surface_size * (1.0 / 0.9)
        self.hero_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()
        self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.result_surface.set_colorkey(COLOR_BLACK)
        self.select_hero_actor()
        self.hero_actor.set_autopilot(False)
        self.module_input.wheel_offset = HERO_DEFAULT_SCALE
        self.module_input.control = carla.VehicleControl()
        weak_self = weakref.ref(self)
        self.world.on_tick(lambda timestamp: ModuleWorld.on_world_tick(weak_self, timestamp))

    def select_hero_actor(self):
        hero_vehicles = [actor for actor in self.world.get_actors() if 'vehicle' in actor.type_id and actor.attributes['role_name'] == 'hero']
        if len(hero_vehicles) > 0:
            self.hero_actor = random.choice(hero_vehicles)
            self.hero_transform = self.hero_actor.get_transform()
        else:
            self._spawn_hero()

    def _spawn_hero(self):
        blueprint = random.choice(self.world.get_blueprint_library().filter(self.args.filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        while self.hero_actor is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.hero_actor = self.world.try_spawn_actor(blueprint, spawn_point)
        self.hero_transform = self.hero_actor.get_transform()
        self.spawned_hero = self.hero_actor

    def tick(self, clock):
        actors = self.world.get_actors()
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()
        self.update_hud_info(clock)

    def update_hud_info(self, clock):
        hero_mode_text = []
        if self.hero_actor is not None:
            hero_speed = self.hero_actor.get_velocity()
            hero_speed_text = 3.6 * math.sqrt(hero_speed.x ** 2 + hero_speed.y ** 2 + hero_speed.z ** 2)
            affected_traffic_light_text = 'None'
            if self.affected_traffic_light is not None:
                state = self.affected_traffic_light.state
                if state == carla.TrafficLightState.Green:
                    affected_traffic_light_text = 'GREEN'
                elif state == carla.TrafficLightState.Yellow:
                    affected_traffic_light_text = 'YELLOW'
                else:
                    affected_traffic_light_text = 'RED'
            affected_speed_limit_text = self.hero_actor.get_speed_limit()
            hero_mode_text = ['Hero Mode:                 ON', 'Hero ID:              %7d' % self.hero_actor.id, 'Hero Vehicle:  %14s' % get_actor_display_name(self.hero_actor, truncate=14), 'Hero Speed:          %3d km/h' % hero_speed_text, 'Hero Affected by:', '  Traffic Light: %12s' % affected_traffic_light_text, '  Speed Limit:       %3d km/h' % affected_speed_limit_text]
        else:
            hero_mode_text = ['Hero Mode:                OFF']
        self.server_fps = self.server_clock.get_fps()
        self.server_fps = 'inf' if self.server_fps == float('inf') else round(self.server_fps)
        module_info_text = ['Server:  % 16s FPS' % self.server_fps, 'Client:  % 16s FPS' % round(clock.get_fps()), 'Simulation Time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)), 'Map Name:          %10s' % self.town_map.name]
        module_info_text = module_info_text
        module_hud = module_manager.get_module(MODULE_HUD)
        module_hud.add_info(self.name, module_info_text)
        module_hud.add_info('HERO', hero_mode_text)

    @staticmethod
    def on_world_tick(weak_self, timestamp):
        self = weak_self()
        if not self:
            return
        self.server_clock.tick()
        self.server_fps = self.server_clock.get_fps()
        self.simulation_time = timestamp.elapsed_seconds

    def _split_actors(self):
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []
        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif 'walker' in actor.type_id:
                walkers.append(actor_with_transform)
        info_text = []
        if self.hero_actor is not None and len(vehicles) > 1:
            location = self.hero_transform.location
            vehicle_list = [x[0] for x in vehicles if x[0].id != self.hero_actor.id]

            def distance(v):
                return location.distance(v.get_location())
            for n, vehicle in enumerate(sorted(vehicle_list, key=distance)):
                if n > 15:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                info_text.append('% 5d %s' % (vehicle.id, vehicle_type))
        module_manager.get_module(MODULE_HUD).add_info('NEARBY VEHICLES', info_text)
        return (vehicles, traffic_lights, speed_limits, walkers)

    def _render_traffic_lights(self, surface, list_tl, world_to_pixel):
        self.affected_traffic_light = None
        for tl in list_tl:
            world_pos = tl.get_location()
            pos = world_to_pixel(world_pos)
            if self.args.show_triggers:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, COLOR_BUTTER_1, True, corners, 2)
            if self.hero_actor is not None:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                tl_t = tl.get_transform()
                transformed_tv = tl_t.transform(tl.trigger_volume.location)
                hero_location = self.hero_actor.get_location()
                d = hero_location.distance(transformed_tv)
                s = Util.length(tl.trigger_volume.extent) + Util.length(self.hero_actor.bounding_box.extent)
                if d <= s:
                    self.affected_traffic_light = tl
                    srf = self.traffic_light_surfaces.surfaces['h']
                    surface.blit(srf, srf.get_rect(center=pos))
            srf = self.traffic_light_surfaces.surfaces[tl.state]
            surface.blit(srf, srf.get_rect(center=pos))

    def _render_speed_limits(self, surface, list_sl, world_to_pixel, world_to_pixel_width):
        font_size = world_to_pixel_width(2)
        radius = world_to_pixel_width(2)
        font = pygame.font.SysFont('Arial', font_size)
        for sl in list_sl:
            x, y = world_to_pixel(sl.get_location())
            white_circle_radius = int(radius * 0.75)
            pygame.draw.circle(surface, COLOR_SCARLET_RED_1, (x, y), radius)
            pygame.draw.circle(surface, COLOR_ALUMINIUM_0, (x, y), white_circle_radius)
            limit = sl.type_id.split('.')[2]
            font_surface = font.render(limit, True, COLOR_ALUMINIUM_5)
            if self.args.show_triggers:
                corners = Util.get_bounding_box(sl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, COLOR_PLUM_2, True, corners, 2)
            if self.hero_actor is not None:
                angle = -self.hero_transform.rotation.yaw - 90.0
                font_surface = pygame.transform.rotate(font_surface, angle)
                offset = font_surface.get_rect(center=(x, y))
                surface.blit(font_surface, offset)
            else:
                surface.blit(font_surface, (x - radius / 2, y - radius / 2))

    def _render_walkers(self, surface, list_w, world_to_pixel):
        for w in list_w:
            color = COLOR_PLUM_0
            bb = w[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y), carla.Location(x=bb.x, y=-bb.y), carla.Location(x=bb.x, y=bb.y), carla.Location(x=-bb.x, y=bb.y)]
            w[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(self, surface, list_v, world_to_pixel):
        for v in list_v:
            color = COLOR_SKY_BLUE_0
            if int(v[0].attributes['number_of_wheels']) == 2:
                color = COLOR_CHOCOLATE_1
            if v[0].attributes['role_name'] == 'hero':
                color = COLOR_CHAMELEON_0
            bb = v[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y), carla.Location(x=bb.x - 0.8, y=-bb.y), carla.Location(x=bb.x, y=0), carla.Location(x=bb.x - 0.8, y=bb.y), carla.Location(x=-bb.x, y=bb.y), carla.Location(x=-bb.x, y=-bb.y)]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.lines(surface, color, False, corners, int(math.ceil(4.0 * self.map_image.scale)))

    def render_actors(self, surface, vehicles, traffic_lights, speed_limits, walkers):
        self._render_traffic_lights(surface, [tl[0] for tl in traffic_lights], self.map_image.world_to_pixel)
        self._render_speed_limits(surface, [sl[0] for sl in speed_limits], self.map_image.world_to_pixel, self.map_image.world_to_pixel_width)
        self._render_vehicles(surface, vehicles, self.map_image.world_to_pixel)
        self._render_walkers(surface, walkers, self.map_image.world_to_pixel)

    def clip_surfaces(self, clipping_rect):
        self.actors_surface.set_clip(clipping_rect)
        self.vehicle_id_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

    def _compute_scale(self, scale_factor):
        m = self.module_input.mouse_pos
        px = (m[0] - self.scale_offset[0]) / float(self.prev_scaled_size)
        py = (m[1] - self.scale_offset[1]) / float(self.prev_scaled_size)
        diff_between_scales = (float(self.prev_scaled_size) * px - float(self.scaled_size) * px, float(self.prev_scaled_size) * py - float(self.scaled_size) * py)
        self.scale_offset = (self.scale_offset[0] + diff_between_scales[0], self.scale_offset[1] + diff_between_scales[1])
        self.prev_scaled_size = self.scaled_size
        self.map_image.scale_map(scale_factor)

    def render(self, display):
        if self.actors_with_transforms is None:
            return
        self.result_surface.fill(COLOR_BLACK)
        vehicles, traffic_lights, speed_limits, walkers = self._split_actors()
        scale_factor = self.module_input.wheel_offset
        self.scaled_size = int(self.map_image.width * scale_factor)
        if self.scaled_size != self.prev_scaled_size:
            self._compute_scale(scale_factor)
        self.actors_surface.fill(COLOR_BLACK)
        self.render_actors(self.actors_surface, vehicles, traffic_lights, speed_limits, walkers)
        self.module_hud.render_vehicles_ids(self.vehicle_id_surface, vehicles, self.map_image.world_to_pixel, self.hero_actor, self.hero_transform)
        surfaces = ((self.map_image.surface, (0, 0)), (self.actors_surface, (0, 0)), (self.vehicle_id_surface, (0, 0)))
        angle = 0.0 if self.hero_actor is None else self.hero_transform.rotation.yaw + 90.0
        self.traffic_light_surfaces.rotozoom(-angle, self.map_image.scale)
        center_offset = (0, 0)
        if self.hero_actor is not None:
            hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
            hero_front = self.hero_transform.get_forward_vector()
            translation_offset = (hero_location_screen[0] - self.hero_surface.get_width() / 2 + hero_front.x * PIXELS_AHEAD_VEHICLE, hero_location_screen[1] - self.hero_surface.get_height() / 2 + hero_front.y * PIXELS_AHEAD_VEHICLE)
            clipping_rect = pygame.Rect(translation_offset[0], translation_offset[1], self.hero_surface.get_width(), self.hero_surface.get_height())
            self.clip_surfaces(clipping_rect)
            Util.blits(self.result_surface, surfaces)
            self.border_round_surface.set_clip(clipping_rect)
            self.hero_surface.fill(COLOR_ALUMINIUM_4)
            self.hero_surface.blit(self.result_surface, (-translation_offset[0], -translation_offset[1]))
            rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 0.9).convert()
            center = (display.get_width() / 2, display.get_height() / 2)
            rotation_pivot = rotated_result_surface.get_rect(center=center)
            display.blit(rotated_result_surface, rotation_pivot)
            display.blit(self.border_round_surface, (0, 0))
        else:
            translation_offset = (self.module_input.mouse_offset[0] * scale_factor + self.scale_offset[0], self.module_input.mouse_offset[1] * scale_factor + self.scale_offset[1])
            center_offset = (abs(display.get_width() - self.surface_size) / 2 * scale_factor, 0)
            clipping_rect = pygame.Rect(-translation_offset[0] - center_offset[0], -translation_offset[1], self.module_hud.dim[0], self.module_hud.dim[1])
            self.clip_surfaces(clipping_rect)
            Util.blits(self.result_surface, surfaces)
            display.blit(self.result_surface, (translation_offset[0] + center_offset[0], translation_offset[1]))

    def destroy(self):
        if self.spawned_hero is not None:
            self.spawned_hero.destroy()

