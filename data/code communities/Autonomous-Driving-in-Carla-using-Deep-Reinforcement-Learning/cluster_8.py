# Cluster 8

class CarlaEnvironment:

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:
        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start = True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()

    def reset(self):
        try:
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()
            vehicle_bp = self.get_vehicle(CAR_NAME)
            if self.town == 'Town07':
                transform = self.map.get_spawn_points()[38]
                self.total_distance = 750
            elif self.town == 'Town02':
                transform = self.map.get_spawn_points()[1]
                self.total_distance = 780
            else:
                transform = random.choice(self.map.get_spawn_points())
                self.total_distance = 250
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)
            self.camera_obj = CameraSensor(self.vehicle)
            while len(self.camera_obj.front_camera) == 0:
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 22
            self.max_speed = 25.0
            self.min_speed = 15.0
            self.max_distance_from_center = 3
            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0
            if self.fresh_start:
                self.current_waypoint_index = 0
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.town == 'Town07':
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == 'Town02':
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index
            self.navigation_obs = np.array([self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])
            time.sleep(0.5)
            self.collision_history.clear()
            self.episode_start_time = time.time()
            return [self.image_obs, self.navigation_obs]
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    def step(self, action_idx):
        try:
            self.timesteps += 1
            self.fresh_start = False
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = float((action_idx[1] + 1.0) / 2)
                throttle = max(min(throttle, 1.0), 0.0)
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1, throttle=self.throttle * 0.9 + throttle * 0.1))
                self.previous_steer = steer
                self.throttle = throttle
            else:
                steer = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1))
                self.previous_steer = steer
                self.throttle = 1.0
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)
            self.collision_history = self.collision_obj.collision_data
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.location = self.vehicle.get_location()
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2], self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break
            self.current_waypoint_index = waypoint_index
            self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location), self.vector(self.next_waypoint.transform.location), self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center
            fwd = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle = self.angle_diff(fwd, wp_fwd)
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = self.current_waypoint_index // self.checkpoint_frequency * self.checkpoint_frequency
            done = False
            reward = 0
            if len(self.collision_history) != 0:
                done = True
                reward = -10
            elif self.distance_from_center > self.max_distance_from_center:
                done = True
                reward = -10
            elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
                reward = -10
                done = True
            elif self.velocity > self.max_speed:
                reward = -10
                done = True
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)
            if not done:
                if self.continous_action_space:
                    if self.velocity < self.min_speed:
                        reward = self.velocity / self.min_speed * centering_factor * angle_factor
                    elif self.velocity > self.target_speed:
                        reward = (1.0 - (self.velocity - self.target_speed) / (self.max_speed - self.target_speed)) * centering_factor * angle_factor
                    else:
                        reward = 1.0 * centering_factor * angle_factor
                else:
                    reward = 1.0 * centering_factor * angle_factor
            if self.timesteps >= 7500:
                done = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance // 2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0
            while len(self.camera_obj.front_camera) == 0:
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            normalized_velocity = self.velocity / self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                for sensor in self.sensor_list:
                    sensor.destroy()
                self.remove_sensors()
                for actor in self.actor_list:
                    actor.destroy()
            return ([self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation])
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    def create_pedestrians(self):
        try:
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc != None:
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute('speed', walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)
            for i in range(0, len(self.walker_list), 2):
                all_actors[i].start()
                all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])

    def set_other_vehicles(self):
        try:
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print('NPC vehicles have been generated in autopilot mode.')
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)

    def get_world(self) -> object:
        return self.world

    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()

    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle <= -np.pi:
            angle += 2 * np.pi
        return angle

    def distance_to_line(self, A, B, p):
        num = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom

    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

    def get_discrete_action_space(self):
        action_space = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
        return action_space

    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint

    def set_vehicle(self, vehicle_bp, spawn_points):
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None

class CameraSensor:

    def __init__(self, vehicle):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    def _set_camera_sensor(self, world):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', f'160')
        front_camera_bp.set_attribute('image_size_y', f'80')
        front_camera_bp.set_attribute('fov', f'125')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(carla.Location(x=2.4, z=1.5), carla.Rotation(pitch=-10)), attach_to=self.parent)
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        placeholder1 = placeholder.reshape((image.width, image.height, 4))
        target = placeholder1[:, :, :3]
        self.front_camera.append(target)

class CameraSensorEnv:

    def __init__(self, vehicle):
        pygame.init()
        self.display = pygame.display.set_mode((720, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface = None
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

    def _set_camera_sensor(self, world):
        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', f'720')
        thrid_person_camera_bp.set_attribute('image_size_y', f'720')
        third_camera = world.spawn_actor(thrid_person_camera_bp, carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=-12.0)), attach_to=self.parent)
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        placeholder1 = array.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        placeholder2 = placeholder2[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(placeholder2.swapaxes(0, 1))
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()

class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)

