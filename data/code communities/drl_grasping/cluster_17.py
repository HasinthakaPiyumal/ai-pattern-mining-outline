# Cluster 17

def get_model_pose(world: World, model: Union[ModelWrapper, str], link: Union[Link, str, None]=None, xyzw: bool=False) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Return pose of model's link. Orientation is represented as wxyz quaternion or xyzw based on the passed argument `xyzw`.
    """
    if isinstance(model, str):
        model = world.to_gazebo().get_model(model).to_gazebo()
    if link is None:
        link = model.get_link(link_name=model.link_names()[0])
    elif isinstance(link, str):
        link = model.get_link(link_name=link)
    position = link.position()
    quat = link.orientation()
    if xyzw:
        quat = quat_to_xyzw(quat)
    return (position, quat)

def quat_to_xyzw(wxyz: Union[numpy.ndarray, Tuple[float, float, float, float]]) -> numpy.ndarray:
    if isinstance(wxyz, tuple):
        return (wxyz[1], wxyz[2], wxyz[3], wxyz[0])
    return wxyz[[1, 2, 3, 0]]

def transform_move_to_model_position(world: World, position: Tuple[float, float, float], target_model: Union[ModelWrapper, str], target_link: Union[Link, str, None]=None) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    target_frame_position, target_frame_quat_xyzw = get_model_pose(world, model=target_model, link=target_link, xyzw=True)
    transformed_position = Rotation.from_quat(target_frame_quat_xyzw).apply(position)
    transformed_position = (target_frame_position[0] + transformed_position[0], target_frame_position[1] + transformed_position[1], target_frame_position[2] + transformed_position[2])
    return transformed_position

def transform_change_reference_frame_pose(world: World, position: Tuple[float, float, float], quat: Tuple[float, float, float, float], target_model: Union[ModelWrapper, str], target_link: Union[Link, str, None]=None, xyzw: bool=False) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Change reference frame of original `position` and `quat` from world coordinate system to `target_model::target_link` coordinate system.
    """
    target_frame_position, target_frame_quat = get_model_pose(world, model=target_model, link=target_link, xyzw=True)
    transformed_position = (position[0] - target_frame_position[0], position[1] - target_frame_position[1], position[2] - target_frame_position[2])
    transformed_position = Rotation.from_quat(target_frame_quat).apply(transformed_position, inverse=True)
    if not xyzw:
        target_frame_quat = quat_to_wxyz(target_frame_quat)
    transformed_quat = quat_mul(target_frame_quat, quat, xyzw=xyzw)
    return (tuple(transformed_position), transformed_quat)

class ManipulationGazeboEnvRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer, randomizers.abc.PhysicsRandomizer, randomizers.abc.TaskRandomizer, abc.ABC):
    """
    Basic randomizer of environments for robotic manipulation inside Ignition Gazebo. This randomizer
    also populates the simulated world with robot, terrain, lighting and other entities.
    """
    POST_RANDOMIZATION_MAX_STEPS = 50

    def __init__(self, env: MakeEnvCallable, physics_rollouts_num: int=0, gravity: Tuple[float, float, float]=(0.0, 0.0, -9.80665), gravity_std: Tuple[float, float, float]=(0.0, 0.0, 0.0232), plugin_scene_broadcaster: bool=False, plugin_user_commands: bool=False, plugin_sensors_render_engine: str='ogre2', robot_spawn_position: Tuple[float, float, float]=(0.0, 0.0, 0.0), robot_spawn_quat_xyzw: Tuple[float, float, float, float]=(0.0, 0.0, 0.0, 1.0), robot_random_pose: bool=False, robot_random_spawn_volume: Tuple[float, float, float]=(1.0, 1.0, 0.0), robot_random_joint_positions: bool=False, robot_random_joint_positions_std: float=0.1, robot_random_joint_positions_above_object_spawn: bool=False, robot_random_joint_positions_above_object_spawn_elevation: float=0.2, robot_random_joint_positions_above_object_spawn_xy_randomness: float=0.2, camera_enable: bool=True, camera_type: str='rgbd_camera', camera_relative_to: str='base_link', camera_width: int=128, camera_height: int=128, camera_image_format: str='R8G8B8', camera_update_rate: int=10, camera_horizontal_fov: float=np.pi / 3.0, camera_vertical_fov: float=np.pi / 3.0, camera_clip_color: Tuple[float, float]=(0.01, 1000.0), camera_clip_depth: Tuple[float, float]=(0.05, 10.0), camera_noise_mean: float=None, camera_noise_stddev: float=None, camera_publish_color: bool=False, camera_publish_depth: bool=False, camera_publish_points: bool=False, camera_spawn_position: Tuple[float, float, float]=(0, 0, 1), camera_spawn_quat_xyzw: Tuple[float, float, float, float]=(0, 0.70710678118, 0, 0.70710678118), camera_random_pose_rollouts_num: int=1, camera_random_pose_mode: str='orbit', camera_random_pose_orbit_distance: float=1.0, camera_random_pose_orbit_height_range: Tuple[float, float]=(0.1, 0.7), camera_random_pose_orbit_ignore_arc_behind_robot: float=np.pi / 8, camera_random_pose_select_position_options: List[Tuple[float, float, float]]=[], camera_random_pose_focal_point_z_offset: float=0.0, terrain_enable: bool=True, terrain_type: str='flat', terrain_spawn_position: Tuple[float, float, float]=(0, 0, 0), terrain_spawn_quat_xyzw: Tuple[float, float, float, float]=(0, 0, 0, 1), terrain_size: Tuple[float, float]=(1.0, 1.0), terrain_model_rollouts_num: int=1, light_enable: bool=True, light_type: str='sun', light_direction: Tuple[float, float, float]=(0.5, -0.25, -0.75), light_random_minmax_elevation: Tuple[float, float]=(-0.15, -0.65), light_color: Tuple[float, float, float, float]=(1.0, 1.0, 1.0, 1.0), light_distance: float=1000.0, light_visual: bool=True, light_radius: float=25.0, light_model_rollouts_num: int=1, object_enable: bool=True, object_type: str='box', objects_relative_to: str='base_link', object_static: bool=False, object_collision: bool=True, object_visual: bool=True, object_color: Tuple[float, float, float, float]=(0.8, 0.8, 0.8, 1.0), object_dimensions: List[float]=[0.05, 0.05, 0.05], object_mass: float=0.1, object_count: int=1, object_randomize_count: bool=False, object_spawn_position: Tuple[float, float, float]=(0.0, 0.0, 0.0), object_random_pose: bool=True, object_random_spawn_position_segments: List[Tuple[float, float, float]]=[], object_random_spawn_position_update_workspace_centre: bool=False, object_random_spawn_volume: Tuple[float, float, float]=(0.5, 0.5, 0.5), object_models_rollouts_num: int=1, underworld_collision_plane: bool=True, boundary_collision_walls: bool=False, collision_plane_offset: float=1.0, visualise_workspace: bool=False, visualise_spawn_volume: bool=False, **kwargs):
        if physics_rollouts_num != 0:
            raise TypeError('Proper physics randomization at each reset is not yet implemented. Please set `physics_rollouts_num=0`.')
        kwargs.update({'camera_type': camera_type, 'camera_width': camera_width, 'camera_height': camera_height})
        randomizers.abc.TaskRandomizer.__init__(self)
        randomizers.abc.PhysicsRandomizer.__init__(self, randomize_after_rollouts_num=physics_rollouts_num)
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(self, env=env, physics_randomizer=self, **kwargs)
        self._gravity = gravity
        self._gravity_std = gravity_std
        self._plugin_scene_broadcaster = plugin_scene_broadcaster
        self._plugin_user_commands = plugin_user_commands
        self._plugin_sensors_render_engine = plugin_sensors_render_engine
        self._robot_spawn_position = robot_spawn_position
        self._robot_spawn_quat_xyzw = robot_spawn_quat_xyzw
        self._robot_random_pose = robot_random_pose
        self._robot_random_spawn_volume = robot_random_spawn_volume
        self._robot_random_joint_positions = robot_random_joint_positions
        self._robot_random_joint_positions_std = robot_random_joint_positions_std
        self._robot_random_joint_positions_above_object_spawn = robot_random_joint_positions_above_object_spawn
        self._robot_random_joint_positions_above_object_spawn_elevation = robot_random_joint_positions_above_object_spawn_elevation
        self._robot_random_joint_positions_above_object_spawn_xy_randomness = robot_random_joint_positions_above_object_spawn_xy_randomness
        self._camera_enable = camera_enable
        self._camera_type = camera_type
        self._camera_relative_to = camera_relative_to
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_image_format = camera_image_format
        self._camera_update_rate = camera_update_rate
        self._camera_horizontal_fov = camera_horizontal_fov
        self._camera_vertical_fov = camera_vertical_fov
        self._camera_clip_color = camera_clip_color
        self._camera_clip_depth = camera_clip_depth
        self._camera_noise_mean = camera_noise_mean
        self._camera_noise_stddev = camera_noise_stddev
        self._camera_publish_color = camera_publish_color
        self._camera_publish_depth = camera_publish_depth
        self._camera_publish_points = camera_publish_points
        self._camera_spawn_position = camera_spawn_position
        self._camera_spawn_quat_xyzw = camera_spawn_quat_xyzw
        self._camera_random_pose_rollouts_num = camera_random_pose_rollouts_num
        self._camera_random_pose_mode = camera_random_pose_mode
        self._camera_random_pose_orbit_distance = camera_random_pose_orbit_distance
        self._camera_random_pose_orbit_height_range = camera_random_pose_orbit_height_range
        self._camera_random_pose_orbit_ignore_arc_behind_robot = camera_random_pose_orbit_ignore_arc_behind_robot
        self._camera_random_pose_select_position_options = camera_random_pose_select_position_options
        self._camera_random_pose_focal_point_z_offset = camera_random_pose_focal_point_z_offset
        self._terrain_enable = terrain_enable
        self._terrain_spawn_position = terrain_spawn_position
        self._terrain_spawn_quat_xyzw = terrain_spawn_quat_xyzw
        self._terrain_size = terrain_size
        self._terrain_model_rollouts_num = terrain_model_rollouts_num
        self._light_enable = light_enable
        self._light_direction = light_direction
        self._light_random_minmax_elevation = light_random_minmax_elevation
        self._light_color = light_color
        self._light_distance = light_distance
        self._light_visual = light_visual
        self._light_radius = light_radius
        self._light_model_rollouts_num = light_model_rollouts_num
        self._object_enable = object_enable
        self._objects_relative_to = objects_relative_to
        self._object_static = object_static
        self._object_collision = object_collision
        self._object_visual = object_visual
        self._object_color = object_color
        self._object_dimensions = object_dimensions
        self._object_mass = object_mass
        self._object_count = object_count
        self._object_randomize_count = object_randomize_count
        self._object_spawn_position = object_spawn_position
        self._object_random_pose = object_random_pose
        self._object_random_spawn_position_segments = object_random_spawn_position_segments
        self._object_random_spawn_position_update_workspace_centre = object_random_spawn_position_update_workspace_centre
        self._object_random_spawn_volume = object_random_spawn_volume
        self._object_models_rollouts_num = object_models_rollouts_num
        self._underworld_collision_plane = underworld_collision_plane
        self._boundary_collision_walls = boundary_collision_walls
        self._collision_plane_offset = collision_plane_offset
        if self._collision_plane_offset < 0.0:
            self._collision_plane_offset *= -1.0
        self._visualise_workspace = visualise_workspace
        self._visualise_spawn_volume = visualise_spawn_volume
        self.__terrain_model_class = models.get_terrain_model_class(terrain_type)
        self.__is_terrain_type_randomizable = models.is_terrain_type_randomizable(terrain_type)
        self.__light_model_class = models.get_light_model_class(light_type)
        self.__is_light_type_randomizable = models.is_light_type_randomizable(light_type)
        self.__object_model_class = models.get_object_model_class(object_type)
        self.__is_object_type_randomizable = models.is_object_type_randomizable(object_type)
        if self._object_randomize_count:
            self.__object_max_count = self._object_count
        self.__camera_pose_rollout_counter = camera_random_pose_rollouts_num
        self.__terrain_model_rollout_counter = terrain_model_rollouts_num
        self.__light_model_rollout_counter = light_model_rollouts_num
        self.__object_models_rollout_counter = object_models_rollouts_num
        self.__is_camera_attached = False
        self.__env_initialised = False
        self.__object_positions = {}

    def init_physics_preset(self, task: SupportedTasks):
        self.set_gravity(task=task)

    def randomize_physics(self, task: SupportedTasks, **kwargs):
        self.set_gravity(task=task)

    def set_gravity(self, task: SupportedTasks):
        if not task.world.to_gazebo().set_gravity((task.np_random.normal(loc=self._gravity[0], scale=self._gravity_std[0]), task.np_random.normal(loc=self._gravity[1], scale=self._gravity_std[1]), task.np_random.normal(loc=self._gravity[2], scale=self._gravity_std[2]))):
            raise RuntimeError('Failed to set the gravity')

    def get_engine(self):
        return scenario.PhysicsEngine_dart

    def randomize_task(self, task: SupportedTasks, **kwargs):
        """
        Randomization of the task, which is called on each reset of the environment.
        Note that this randomizer reset is called before `reset_task()`.
        """
        if 'gazebo' not in kwargs:
            raise ValueError('Randomizer does not have access to the gazebo interface')
        gazebo = kwargs['gazebo']
        self.internal_overrides(task=task)
        self.external_overrides(task=task)
        if not self.__env_initialised:
            self.init_env(task=task, gazebo=gazebo)
            self.__env_initialised = True
        self.pre_randomization(task=task)
        self.randomize_models(task=task, gazebo=gazebo)
        self.post_randomization(task, gazebo)

    def init_env(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Initialise an instance of the environment before the very first iteration
        """
        set_log_level(log_level=task.get_logger().get_effective_level().name)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')
        self._object_spawn_position = (self._object_spawn_position[0], self._object_spawn_position[1], self._object_spawn_position[2] + task.robot_model_class.BASE_LINK_Z_OFFSET)
        self._camera_random_pose_focal_point_z_offset += task.robot_model_class.BASE_LINK_Z_OFFSET
        self._camera_relative_to = task.substitute_special_frame(self._camera_relative_to)
        self._objects_relative_to = task.substitute_special_frame(self._objects_relative_to)
        self.init_physics_preset(task=task)
        self.init_world_plugins(task=task, gazebo=gazebo)
        self.init_models(task=task, gazebo=gazebo)

    def init_world_plugins(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        if self._plugin_scene_broadcaster:
            if not gazebo.scene_broadcaster_active(task.substitute_special_frame('world')):
                task.get_logger().info('Inserting world plugins for broadcasting scene to GUI clients...')
                task.world.to_gazebo().insert_world_plugin('ignition-gazebo-scene-broadcaster-system', 'ignition::gazebo::systems::SceneBroadcaster')
                if not gazebo.run(paused=True):
                    raise RuntimeError('Failed to execute a paused Gazebo run')
        if self._plugin_user_commands:
            task.get_logger().info('Inserting world plugins to enable user commands...')
            task.world.to_gazebo().insert_world_plugin('ignition-gazebo-user-commands-system', 'ignition::gazebo::systems::UserCommands')
            if not gazebo.run(paused=True):
                raise RuntimeError('Failed to execute a paused Gazebo run')
        if self._camera_enable:
            task.get_logger().info(f'Inserting world plugins for sensors with {self._plugin_sensors_render_engine} rendering engine...')
            task.world.to_gazebo().insert_world_plugin('libignition-gazebo-sensors-system.so', 'ignition::gazebo::systems::Sensors', f"<sdf version='1.9'><render_engine>{self._plugin_sensors_render_engine}</render_engine></sdf>")
            if not gazebo.run(paused=True):
                raise RuntimeError('Failed to execute a paused Gazebo run')

    def init_models(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Initialise all models that are persistent throughout the entire training (they do not need to be re-spawned).
        All other models that need to be re-spawned on each reset are ignored here
        """
        model_names = task.world.to_gazebo().model_names()
        if len(model_names) > 0:
            task.get_logger().warn(f'Before initialisation, the world already contains the following models:\n\t{model_names}')
        if self._light_enable and (not self.__light_model_randomizer_enabled()):
            task.get_logger().info('Inserting default light into the environment...')
            self.add_default_light(task=task, gazebo=gazebo)
        if self._terrain_enable and (not self.__terrain_model_randomizer_enabled()):
            task.get_logger().info('Inserting default terrain into the environment...')
            self.add_default_terrain(task=task, gazebo=gazebo)
        task.get_logger().info('Inserting robot into the environment...')
        self.add_robot(task=task, gazebo=gazebo)
        if self._camera_enable:
            task.get_logger().info('Inserting camera into the environment...')
            self.add_camera(task=task, gazebo=gazebo)
        if self._object_enable and (not self.__object_models_randomizer_enabled()):
            task.get_logger().info('Inserting default objects into the environment...')
            self.add_default_objects(task=task, gazebo=gazebo)
        if self._underworld_collision_plane:
            task.get_logger().info('Inserting invisible plane below the terrain into the environment...')
            self.add_underworld_collision_plane(task=task, gazebo=gazebo)
        if self._boundary_collision_walls:
            task.get_logger().info('Inserting invisible planes around the terrain into the environment...')
            self.add_boundary_collision_walls(task=task, gazebo=gazebo)
        if self._visualise_workspace:
            self.visualise_workspace(task=task, gazebo=gazebo)
        if self._visualise_spawn_volume:
            self.visualise_spawn_volume(task=task, gazebo=gazebo)

    def add_robot(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Configure and insert robot into the simulation
        """
        self.robot = task.robot_model_class(world=task.world, name=task.robot_name, prefix=task.robot_prefix, position=self._robot_spawn_position, orientation=quat_to_wxyz(self._robot_spawn_quat_xyzw), initial_arm_joint_positions=task.initial_arm_joint_positions, initial_gripper_joint_positions=task.initial_gripper_joint_positions)
        task.robot_name = self.robot.name()
        robot_gazebo = self.robot.to_gazebo()
        for gripper_link_name in self.robot.gripper_link_names:
            finger = robot_gazebo.get_link(link_name=gripper_link_name)
            finger.enable_contact_detection(True)
        if self.robot.is_mobile:
            for wheel_link_name in self.robot.wheel_link_names:
                wheel = robot_gazebo.get_link(link_name=wheel_link_name)
                wheel.enable_contact_detection(True)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')
        self.reset_robot_joint_positions(task=task, gazebo=gazebo, above_object_spawn=False, randomize=False)

    def add_camera(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Configure and insert camera into the simulation. Camera is places with respect to the robot
        """
        if task.world.to_gazebo().name() == self._camera_relative_to:
            camera_position = self._camera_spawn_position
            camera_quat_wxyz = quat_to_wxyz(self._camera_spawn_quat_xyzw)
        else:
            camera_position, camera_quat_wxyz = transform_move_to_model_pose(world=task.world, position=self._camera_spawn_position, quat=quat_to_wxyz(self._camera_spawn_quat_xyzw), target_model=self.robot, target_link=self._camera_relative_to, xyzw=False)
        self.camera = models.Camera(world=task.world, position=camera_position, orientation=camera_quat_wxyz, camera_type=self._camera_type, width=self._camera_width, height=self._camera_height, image_format=self._camera_image_format, update_rate=self._camera_update_rate, horizontal_fov=self._camera_horizontal_fov, vertical_fov=self._camera_vertical_fov, clip_color=self._camera_clip_color, clip_depth=self._camera_clip_depth, noise_mean=self._camera_noise_mean, noise_stddev=self._camera_noise_stddev, ros2_bridge_color=self._camera_publish_color, ros2_bridge_depth=self._camera_publish_depth, ros2_bridge_points=self._camera_publish_points)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')
        if task.world.to_gazebo().name() != self._camera_relative_to:
            if not self.robot.to_gazebo().attach_link(self._camera_relative_to, self.camera.name(), self.camera.link_name):
                raise Exception('Cannot attach camera link to robot')
            self.__is_camera_attached = True
            if not gazebo.run(paused=True):
                raise RuntimeError('Failed to execute a paused Gazebo run')
        task.tf2_broadcaster.broadcast_tf(parent_frame_id=self._camera_relative_to, child_frame_id=self.camera.frame_id, translation=self._camera_spawn_position, rotation=self._camera_spawn_quat_xyzw, xyzw=True)

    def add_default_terrain(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Configure and insert default terrain into the simulation
        """
        self.terrain = self.__terrain_model_class(world=task.world, position=self._terrain_spawn_position, orientation=quat_to_wxyz(self._terrain_spawn_quat_xyzw), size=self._terrain_size, np_random=task.np_random)
        task.terrain_name = self.terrain.name()
        for link_name in self.terrain.link_names():
            link = self.terrain.to_gazebo().get_link(link_name=link_name)
            link.enable_contact_detection(True)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def add_default_light(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Configure and insert default light into the simulation
        """
        self.light = self.__light_model_class(world=task.world, direction=self._light_direction, minmax_elevation=self._light_random_minmax_elevation, color=self._light_color, distance=self._light_distance, visual=self._light_visual, radius=self._light_radius, np_random=task.np_random)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def add_default_objects(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Configure and insert default object into the simulation
        """
        while len(self.task.object_names) < self._object_count:
            if self._object_count > 1:
                object_position, object_quat_wxyz = self.get_random_object_pose(task=task, centre=self._object_spawn_position, volume=self._object_random_spawn_volume)
            else:
                object_position = self._object_spawn_position
                object_quat_wxyz = (1.0, 0.0, 0.0, 0.0)
                if task.world.to_gazebo().name() != self._objects_relative_to:
                    object_position, object_quat_wxyz = transform_move_to_model_pose(world=task.world, position=object_position, quat=object_quat_wxyz, target_model=self.robot, target_link=self._objects_relative_to, xyzw=False)
            try:
                object_model = self.__object_model_class(world=task.world, position=object_position, orientation=object_quat_wxyz, size=self._object_dimensions, radius=self._object_dimensions[0], length=self._object_dimensions[1], mass=self._object_mass, collision=self._object_collision, visual=self._object_visual, static=self._object_static, color=self._object_color)
                model_name = object_model.name()
                task.object_names.append(model_name)
                for link_name in object_model.link_names():
                    link = object_model.to_gazebo().get_link(link_name=link_name)
                    link.enable_contact_detection(True)
            except Exception as ex:
                task.get_logger().warn(f'Model could not be inserted. Reason: {ex}')
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def add_underworld_collision_plane(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Add an infinitely large collision plane below the terrain in order to prevent object from falling into the abyss forever
        """
        models.Plane(name='_collision_plane_B', world=task.world, position=(0.0, 0.0, self._terrain_spawn_position[2] - self._collision_plane_offset), orientation=(1.0, 0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), visual=False, collision=True, friction=1000.0)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def add_boundary_collision_walls(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Add an infinitely large collision planes around the terrain in order to prevent object from going into the abyss forever
        """
        models.Plane(name='_collision_plane_N', world=task.world, position=(self._terrain_spawn_position[0] + self._terrain_size[0] / 2 + self._collision_plane_offset, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), direction=(-1.0, 0.0, 0.0), visual=False, collision=True, friction=1000.0)
        models.Plane(name='_collision_plane_S', world=task.world, position=(self._terrain_spawn_position[0] - self._terrain_size[0] / 2 - self._collision_plane_offset, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), visual=False, collision=True, friction=1000.0)
        models.Plane(name='_collision_plane_E', world=task.world, position=(0.0, self._terrain_spawn_position[1] + self._terrain_size[1] / 2 + self._collision_plane_offset, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), direction=(0.0, -1.0, 0.0), visual=False, collision=True, friction=1000.0)
        models.Plane(name='_collision_plane_W', world=task.world, position=(0.0, self._terrain_spawn_position[1] - self._terrain_size[1] / 2 - self._collision_plane_offset, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), direction=(0.0, 1.0, 0.0), visual=False, collision=True, friction=1000.0)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def randomize_models(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Randomize models if needed
        """
        if self._light_enable and self._light_model_expired():
            self.randomize_light(task=task, gazebo=gazebo)
        if self.robot.is_mobile:
            self.reset_robot_pose(task=task, gazebo=gazebo, randomize=self._robot_random_pose)
        self.reset_robot_joint_positions(task=task, gazebo=gazebo, above_object_spawn=self._robot_random_joint_positions_above_object_spawn, randomize=self._robot_random_joint_positions)
        if self._camera_enable and self._camera_pose_expired():
            self.randomize_camera_pose(task=task, gazebo=gazebo, mode=self._camera_random_pose_mode)
        if self._object_enable:
            self.__object_positions.clear()
            if self._object_models_expired():
                self.randomize_object_models(task=task, gazebo=gazebo)
            elif self._object_random_pose:
                self.object_random_pose(task=task, gazebo=gazebo)
            else:
                self.reset_default_object_pose(task=task, gazebo=gazebo)
        if self._terrain_enable and self._terrain_model_expired():
            self.randomize_terrain(task=task, gazebo=gazebo)

    def reset_robot_pose(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator, randomize: bool=False):
        if randomize:
            position = [self._robot_spawn_position[0] + task.np_random.uniform(-self._robot_random_spawn_volume[0] / 2, self._robot_random_spawn_volume[0] / 2), self._robot_spawn_position[1] + task.np_random.uniform(-self._robot_random_spawn_volume[1] / 2, self._robot_random_spawn_volume[1] / 2), self._robot_spawn_position[2] + task.np_random.uniform(-self._robot_random_spawn_volume[2] / 2, self._robot_random_spawn_volume[2] / 2)]
            quat_xyzw = Rotation.from_euler('xyz', (0, 0, task.np_random.uniform(-np.pi, np.pi))).as_quat()
        else:
            position = self._robot_spawn_position
            quat_xyzw = self._robot_spawn_quat_xyzw
        gazebo_robot = self.robot.to_gazebo()
        gazebo_robot.reset_base_pose(position, quat_to_wxyz(quat_xyzw))
        gazebo_robot.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def reset_robot_joint_positions(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator, above_object_spawn: bool=False, randomize: bool=False):
        if task._use_servo:
            if task.servo.is_enabled:
                task.servo.servo()
                task.servo.disable(sync=True)
        gazebo_robot = self.robot.to_gazebo()
        if above_object_spawn:
            if randomize:
                rnd_displacement = self._robot_random_joint_positions_above_object_spawn_xy_randomness * task.np_random.uniform((-self._object_random_spawn_volume[0], -self._object_random_spawn_volume[1]), self._object_random_spawn_volume[:2])
                position = (self._object_spawn_position[0] + rnd_displacement[0], self._object_spawn_position[1] + rnd_displacement[1], self._object_spawn_position[2] + self._robot_random_joint_positions_above_object_spawn_elevation)
                quat_xyzw = Rotation.from_euler('xyz', (0, np.pi, task.np_random.uniform(-np.pi, np.pi))).as_quat()
            else:
                position = (self._object_spawn_position[0], self._object_spawn_position[1], self._object_spawn_position[2] + self._robot_random_joint_positions_above_object_spawn_elevation)
                quat_xyzw = (1.0, 0.0, 0.0, 0.0)
            joint_configuration = task.moveit2.compute_ik(position=position, quat_xyzw=quat_xyzw, start_joint_state=task.initial_arm_joint_positions)
            if joint_configuration is not None:
                arm_joint_positions = joint_configuration.position[:len(task.initial_arm_joint_positions)]
            else:
                task.get_logger().warn('Robot configuration could not be reset above the object spawn. Using initial arm joint positions instead.')
                arm_joint_positions = task.initial_arm_joint_positions
        else:
            arm_joint_positions = task.initial_arm_joint_positions
        if randomize:
            for joint_position in arm_joint_positions:
                joint_position += task.np_random.normal(loc=0.0, scale=self._robot_random_joint_positions_std)
        if not gazebo_robot.reset_joint_positions(arm_joint_positions, self.robot.arm_joint_names):
            raise RuntimeError('Failed to reset robot joint positions')
        if not gazebo_robot.reset_joint_velocities([0.0] * len(self.robot.arm_joint_names), self.robot.arm_joint_names):
            raise RuntimeError('Failed to reset robot joint velocities')
        if task._enable_gripper and self.robot.gripper_joint_names:
            if not gazebo_robot.reset_joint_positions(task.initial_gripper_joint_positions, self.robot.gripper_joint_names):
                raise RuntimeError('Failed to reset gripper joint positions')
            if not gazebo_robot.reset_joint_velocities([0.0] * len(self.robot.gripper_joint_names), self.robot.gripper_joint_names):
                raise RuntimeError('Failed to reset gripper joint velocities')
        if self.robot.passive_joint_names:
            if not gazebo_robot.reset_joint_velocities([0.0] * len(self.robot.passive_joint_names), self.robot.passive_joint_names):
                raise RuntimeError('Failed to reset passive joint velocities')
        if not gazebo.step():
            raise RuntimeError('Failed to execute an unpaused Gazebo step')
        task.moveit2.force_reset_executing_state()
        task.moveit2.reset_controller(joint_state=arm_joint_positions)
        if task._enable_gripper:
            if self.robot.CLOSED_GRIPPER_JOINT_POSITIONS == task.initial_gripper_joint_positions:
                task.gripper.close()
            else:
                task.gripper.open()

    def randomize_camera_pose(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator, mode: str):
        if 'orbit' == mode:
            camera_position, camera_quat_xyzw = self.get_random_camera_pose_orbit(task=task, centre=self._object_spawn_position, distance=self._camera_random_pose_orbit_distance, height=self._camera_random_pose_orbit_height_range, ignore_arc_behind_robot=self._camera_random_pose_orbit_ignore_arc_behind_robot, focal_point_z_offset=self._camera_random_pose_focal_point_z_offset)
        elif 'select_random' == mode:
            camera_position, camera_quat_xyzw = self.get_random_camera_pose_sample_random(task=task, centre=self._object_spawn_position, options=self._camera_random_pose_select_position_options)
        elif 'select_nearest' == mode:
            camera_position, camera_quat_xyzw = self.get_random_camera_pose_sample_nearest(centre=self._object_spawn_position, options=self._camera_random_pose_select_position_options)
        else:
            raise TypeError('Invalid mode for camera pose randomization.')
        if task.world.to_gazebo().name() == self._camera_relative_to:
            transformed_camera_position = camera_position
            transformed_camera_quat_wxyz = quat_to_wxyz(camera_quat_xyzw)
        else:
            transformed_camera_position, transformed_camera_quat_wxyz = transform_move_to_model_pose(world=task.world, position=camera_position, quat=quat_to_wxyz(camera_quat_xyzw), target_model=self.robot, target_link=self._camera_relative_to, xyzw=False)
        if self.__is_camera_attached:
            if not self.robot.to_gazebo().detach_link(self._camera_relative_to, self.camera.name(), self.camera.link_name):
                raise Exception('Cannot detach camera link from robot')
            if not gazebo.run(paused=True):
                raise RuntimeError('Failed to execute a paused Gazebo run')
        camera_gazebo = self.camera.to_gazebo()
        camera_gazebo.reset_base_pose(transformed_camera_position, transformed_camera_quat_wxyz)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')
        if self.__is_camera_attached:
            if not self.robot.to_gazebo().attach_link(self._camera_relative_to, self.camera.name(), self.camera.link_name):
                raise Exception('Cannot attach camera link to robot')
            if not gazebo.run(paused=True):
                raise RuntimeError('Failed to execute a paused Gazebo run')
        task.tf2_broadcaster.broadcast_tf(parent_frame_id=self._camera_relative_to, child_frame_id=self.camera.frame_id, translation=camera_position, rotation=camera_quat_xyzw, xyzw=True)

    def get_random_camera_pose_orbit(self, task: SupportedTasks, centre: Tuple[float, float, float], distance: float, height: Tuple[float, float], ignore_arc_behind_robot: float, focal_point_z_offset: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        while True:
            position = task.np_random.uniform(low=(-1.0, -1.0, height[0]), high=(1.0, 1.0, height[1]))
            position /= np.linalg.norm(position)
            if abs(np.arctan2(position[0], position[1]) + np.pi / 2) > ignore_arc_behind_robot:
                break
        rpy = [0.0, np.arctan2(position[2] - focal_point_z_offset, np.linalg.norm(position[:2], 2)), np.arctan2(position[1], position[0]) + np.pi]
        quat_xyzw = Rotation.from_euler('xyz', rpy).as_quat()
        position *= distance
        position[:2] += centre[:2]
        return (position, quat_xyzw)

    def get_random_camera_pose_sample_random(self, task: SupportedTasks, centre: Tuple[float, float, float], options: List[Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        selection = options[task.np_random.randint(len(options))]
        return self.get_random_camera_pose_sample_process(centre=centre, position=selection, focal_point_z_offset=self._camera_random_pose_focal_point_z_offset)

    def get_random_camera_pose_sample_nearest(self, centre: Tuple[float, float, float], options: List[Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        dist_sqr = np.sum((np.array(options) - np.array(centre)) ** 2, axis=1)
        nearest = options[np.argmin(dist_sqr)]
        return self.get_random_camera_pose_sample_process(centre=centre, position=nearest, focal_point_z_offset=self._camera_random_pose_focal_point_z_offset)

    def get_random_camera_pose_sample_process(self, centre: Tuple[float, float, float], position: Tuple[float, float, float], focal_point_z_offset: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        rpy = [0.0, np.arctan2(position[2] - focal_point_z_offset, np.linalg.norm((position[0] - centre[0], position[1] - centre[1]), 2)), np.arctan2(position[1] - centre[1], position[0] - centre[0]) + np.pi]
        quat_xyzw = Rotation.from_euler('xyz', rpy).as_quat()
        return (position, quat_xyzw)

    def randomize_terrain(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        if hasattr(self, 'terrain'):
            if not task.world.to_gazebo().remove_model(self.terrain.name()):
                raise RuntimeError(f'Failed to remove {self.terrain.name()}')
        orientation = [(1, 0, 0, 0), (0, 0, 0, 1), (0.70710678118, 0, 0, 0.70710678118), (0.70710678118, 0, 0, -0.70710678118)][task.np_random.randint(4)]
        self.terrain = self.__terrain_model_class(world=task.world, position=self._terrain_spawn_position, orientation=orientation, size=self._terrain_size, np_random=task.np_random)
        task.terrain_name = self.terrain.name()
        for link_name in self.terrain.link_names():
            link = self.terrain.to_gazebo().get_link(link_name=link_name)
            link.enable_contact_detection(True)
        if not gazebo.step():
            raise RuntimeError('Failed to execute an unpaused Gazebo run')

    def randomize_light(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        if hasattr(self, 'light'):
            if not task.world.to_gazebo().remove_model(self.light.name()):
                raise RuntimeError(f'Failed to remove {self.light.name()}')
        self.light = self.__light_model_class(world=task.world, direction=self._light_direction, minmax_elevation=self._light_random_minmax_elevation, color=self._light_color, distance=self._light_distance, visual=self._light_visual, radius=self._light_radius, np_random=task.np_random)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def reset_default_object_pose(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        assert len(task.object_names) == 1
        obj = task.world.to_gazebo().get_model(task.object_names[0]).to_gazebo()
        obj.reset_base_pose(self._object_spawn_position, (1.0, 0.0, 0.0, 0.0))
        obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def randomize_object_models(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        if len(self.task.object_names) > 0:
            for object_name in self.task.object_names:
                if not task.world.to_gazebo().remove_model(object_name):
                    raise RuntimeError(f'Failed to remove {object_name}')
            self.task.object_names.clear()
        while len(self.task.object_names) < self._object_count:
            position, quat_random = self.get_random_object_pose(task=task, centre=self._object_spawn_position, volume=self._object_random_spawn_volume)
            try:
                model = self.__object_model_class(world=task.world, position=position, orientation=quat_random, np_random=task.np_random)
                model_name = model.name()
                self.task.object_names.append(model_name)
                self.__object_positions[model_name] = position
                for link_name in model.link_names():
                    link = model.to_gazebo().get_link(link_name=link_name)
                    link.enable_contact_detection(True)
            except Exception as ex:
                task.get_logger().warn(f'Model could not be inserted: {ex}')
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def object_random_pose(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        for object_name in self.task.object_names:
            position, quat_random = self.get_random_object_pose(task=task, centre=self._object_spawn_position, volume=self._object_random_spawn_volume)
            obj = task.world.to_gazebo().get_model(object_name).to_gazebo()
            obj.reset_base_pose(position, quat_random)
            obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self.__object_positions[object_name] = position
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def get_random_object_pose(self, task: SupportedTasks, centre: Tuple[float, float, float], volume: Tuple[float, float, float], name: str='', min_distance_to_other_objects: float=0.2, min_distance_decay_factor: float=0.95):
        is_too_close = True
        while is_too_close:
            object_position = [centre[0] + task.np_random.uniform(-volume[0] / 2, volume[0] / 2), centre[1] + task.np_random.uniform(-volume[1] / 2, volume[1] / 2), centre[2] + task.np_random.uniform(-volume[2] / 2, volume[2] / 2)]
            if task.world.to_gazebo().name() != self._objects_relative_to:
                object_position = transform_move_to_model_position(world=task.world, position=object_position, target_model=self.robot, target_link=self._objects_relative_to)
            is_too_close = False
            for existing_object_name, existing_object_position in self.__object_positions.items():
                if existing_object_name == name:
                    continue
                if distance.euclidean(object_position, existing_object_position) < min_distance_to_other_objects:
                    min_distance_to_other_objects *= min_distance_decay_factor
                    is_too_close = True
                    break
        quat = task.np_random.uniform(-1, 1, 4)
        quat /= np.linalg.norm(quat)
        return (object_position, quat)

    def internal_overrides(self, task: SupportedTasks):
        """
        Perform internal overrides if parameters
        """
        if self._object_randomize_count:
            self._object_count = task.np_random.randint(low=1, high=self.__object_max_count + 1)

    def external_overrides(self, task: SupportedTasks):
        """
        Perform external overrides from either task level or environment before initialising/randomising the task.
        """
        self.__consume_parameter_overrides(task=task)

    def pre_randomization(self, task: SupportedTasks):
        """
        Perform steps that are required before randomization is performed.
        """
        segments_len = len(self._object_random_spawn_position_segments)
        if segments_len > 1:
            start_index = task.np_random.randint(segments_len - 1)
            segment = (self._object_random_spawn_position_segments[start_index], self._object_random_spawn_position_segments[start_index + 1])
            intersect = task.np_random.random()
            direction = (segment[1][0] - segment[0][0], segment[1][1] - segment[0][1], segment[1][2] - segment[0][2])
            self._object_spawn_position = (segment[0][0] + intersect * direction[0], segment[0][1] + intersect * direction[1], segment[0][2] + intersect * direction[2])
            if self._object_random_spawn_position_update_workspace_centre:
                task.workspace_centre = (self._object_spawn_position[0], self._object_spawn_position[1], task.workspace_centre[2])
                workspace_volume_half = (task.workspace_volume[0] / 2, task.workspace_volume[1] / 2, task.workspace_volume[2] / 2)
                task.workspace_min_bound = (task.workspace_centre[0] - workspace_volume_half[0], task.workspace_centre[1] - workspace_volume_half[1], task.workspace_centre[2] - workspace_volume_half[2])
                task.workspace_max_bound = (task.workspace_centre[0] + workspace_volume_half[0], task.workspace_centre[1] + workspace_volume_half[1], task.workspace_centre[2] + workspace_volume_half[2])

    def post_randomization(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator):
        """
        Perform steps that are required once randomization is complete and the simulation can be stepped a few times unpaused.
        """
        attempts = 0
        object_overlapping_ok = False
        if self.robot.is_mobile:
            try:
                robot_gazebo = self.robot.to_gazebo()
                wheel_links = [robot_gazebo.get_link(link_name=wheel_link_name) for wheel_link_name in self.robot.wheel_link_names]
                is_robot_in_contact_with_terrain = False
                while not is_robot_in_contact_with_terrain and attempts < self.POST_RANDOMIZATION_MAX_STEPS:
                    for wheel_link in wheel_links:
                        wheel_contacts = wheel_link.contacts()
                        if wheel_contacts:
                            break
                    for contact in wheel_contacts:
                        if f'{task.terrain_name}::' in contact.body_b:
                            is_robot_in_contact_with_terrain = True
                            break
                        elif '_collision_plane_B::' in contact.body_b:
                            attempts += 1
                            if self._terrain_enable:
                                self.randomize_terrain(task=task, gazebo=gazebo)
                            self.reset_robot_pose(task=task, gazebo=gazebo, randomize=self._robot_random_pose)
                            if self._object_enable:
                                if self._object_random_pose:
                                    self.object_random_pose(task=task, gazebo=gazebo)
                                else:
                                    self.reset_default_object_pose(task=task, gazebo=gazebo)
                            break
                    object_overlapping_ok = self.check_object_overlapping(task=task)
                    if not gazebo.step():
                        raise RuntimeError('Failed to execute an unpaused Gazebo step')
            except Exception as e:
                task.get_logger().error(f'Wheel contacts could not be checked due to an unexpected error: {e}')
        if self.POST_RANDOMIZATION_MAX_STEPS == attempts:
            task.get_logger().error('Robot keeps falling through the terrain. There is something wrong...')
            return
        while not object_overlapping_ok and attempts < self.POST_RANDOMIZATION_MAX_STEPS:
            attempts += 1
            task.get_logger().info('Objects overlapping, trying new positions')
            object_overlapping_ok = self.check_object_overlapping(task=task)
            if not gazebo.step():
                raise RuntimeError('Failed to execute an unpaused Gazebo step')
        if self.POST_RANDOMIZATION_MAX_STEPS == attempts:
            task.get_logger().warn('Objects could not be spawned without any overlapping. The workspace might be too crowded!')
            return
        observations_ready = False
        task.moveit2.reset_new_joint_state_checker()
        if task._enable_gripper:
            task.gripper.reset_new_joint_state_checker()
        if hasattr(task, 'camera_sub'):
            task.camera_sub.reset_new_observation_checker()
        while not observations_ready:
            attempts += 1
            if 0 == attempts % self.POST_RANDOMIZATION_MAX_STEPS:
                task.get_logger().warn(f'Waiting for new joint state after reset. Iteration #{attempts}...')
            else:
                task.get_logger().debug('Waiting for new joint state after reset.')
            if not gazebo.step():
                raise RuntimeError('Failed to execute an unpaused Gazebo step')
            if not task.moveit2.new_joint_state_available:
                continue
            if task._enable_gripper:
                if not task.gripper.new_joint_state_available:
                    continue
            if hasattr(task, 'camera_sub'):
                if not task.camera_sub.new_observation_available:
                    continue
            observations_ready = True
        if self.POST_RANDOMIZATION_MAX_STEPS == attempts:
            task.get_logger().error('Cannot obtain new observation.')
            return

    def check_object_overlapping(self, task: SupportedTasks, allowed_penetration_depth: float=0.001, terrain_allowed_penetration_depth: float=0.002) -> bool:
        """
        Go through all objects and make sure that none of them are overlapping.
        If an object is overlapping, reset its position.
        Positions are reset also if object is in collision with robot right after reset.
        Collisions/overlaps with terrain are ignored.
        Returns True if all objects are okay, false if they had to be reset
        """
        for object_name in self.task.object_names:
            model = task.world.get_model(object_name).to_gazebo()
            self.__object_positions[object_name] = model.get_link(link_name=model.link_names()[0]).position()
        for object_name in self.task.object_names:
            obj = task.world.get_model(object_name).to_gazebo()
            if task.check_object_outside_workspace(self.__object_positions[object_name]):
                position, quat_random = self.get_random_object_pose(task=task, centre=self._object_spawn_position, volume=self._object_random_spawn_volume, name=object_name)
                obj.reset_base_pose(position, quat_random)
                obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                return False
            try:
                for contact in obj.contacts():
                    depth = np.mean([point.depth for point in contact.points])
                    if self.terrain.name() in contact.body_b and depth < terrain_allowed_penetration_depth:
                        continue
                    if task.robot_name in contact.body_b or depth > allowed_penetration_depth:
                        position, quat_random = self.get_random_object_pose(task=task, centre=self._object_spawn_position, volume=self._object_random_spawn_volume, name=object_name)
                        obj.reset_base_pose(position, quat_random)
                        obj.reset_base_world_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                        return False
            except Exception as e:
                task.get_logger().error(f'Runtime error encountered while checking objects intersections: {e}')
        return True

    def __camera_pose_randomizer_enabled(self) -> bool:
        """
        Checks if camera pose randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """
        if self._camera_random_pose_rollouts_num == 0:
            return False
        else:
            return True

    def _camera_pose_expired(self) -> bool:
        """
        Checks if camera pose needs to be randomized.

        Return:
            True if expired, false otherwise
        """
        if not self.__camera_pose_randomizer_enabled():
            return False
        self.__camera_pose_rollout_counter += 1
        if self.__camera_pose_rollout_counter >= self._camera_random_pose_rollouts_num:
            self.__camera_pose_rollout_counter = 0
            return True
        return False

    def __terrain_model_randomizer_enabled(self) -> bool:
        """
        Checks if terrain randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """
        if self._terrain_model_rollouts_num == 0:
            return False
        else:
            return self.__is_terrain_type_randomizable

    def _terrain_model_expired(self) -> bool:
        """
        Checks if terrain model needs to be randomized.

        Return:
            True if expired, false otherwise
        """
        if not self.__terrain_model_randomizer_enabled():
            return False
        self.__terrain_model_rollout_counter += 1
        if self.__terrain_model_rollout_counter >= self._terrain_model_rollouts_num:
            self.__terrain_model_rollout_counter = 0
            return True
        return False

    def __light_model_randomizer_enabled(self) -> bool:
        """
        Checks if light model randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """
        if self._light_model_rollouts_num == 0:
            return False
        else:
            return self.__is_light_type_randomizable

    def _light_model_expired(self) -> bool:
        """
        Checks if light models need to be randomized.

        Return:
            True if expired, false otherwise
        """
        if not self.__light_model_randomizer_enabled():
            return False
        self.__light_model_rollout_counter += 1
        if self.__light_model_rollout_counter >= self._light_model_rollouts_num:
            self.__light_model_rollout_counter = 0
            return True
        return False

    def __object_models_randomizer_enabled(self) -> bool:
        """
        Checks if object model randomizer is enabled.

        Return:
            True if enabled, false otherwise
        """
        if self._object_models_rollouts_num == 0:
            return False
        else:
            return self.__is_object_type_randomizable

    def _object_models_expired(self) -> bool:
        """
        Checks if object models need to be randomized.

        Return:
            True if expired, false otherwise
        """
        if not self.__object_models_randomizer_enabled():
            return False
        self.__object_models_rollout_counter += 1
        if self.__object_models_rollout_counter >= self._object_models_rollouts_num:
            self.__object_models_rollout_counter = 0
            return True
        return False

    def __consume_parameter_overrides(self, task: SupportedTasks):
        for key, value in task._randomizer_parameter_overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self, f'_{key}'):
                setattr(self, f'_{key}', value)
            elif hasattr(self, f'__{key}'):
                setattr(self, f'__{key}', value)
            else:
                task.get_logger().error(f"Override '{key}' is not supperted by the randomizer.")
        task._randomizer_parameter_overrides.clear()

    def visualise_workspace(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator, color: Tuple[float, float, float, float]=(0, 1, 0, 0.8)):
        models.Box(world=task.world, name='_workspace_volume', position=self._object_spawn_position, orientation=(0, 0, 0, 1), size=task.workspace_volume, collision=False, visual=True, gui_only=True, static=True, color=color)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

    def visualise_spawn_volume(self, task: SupportedTasks, gazebo: scenario.GazeboSimulator, color: Tuple[float, float, float, float]=(0, 0, 1, 0.8), color_with_height: Tuple[float, float, float, float]=(1, 0, 1, 0.7)):
        models.Box(world=task.world, name='_object_random_spawn_volume', position=self._object_spawn_position, orientation=(0, 0, 0, 1), size=self._object_random_spawn_volume, collision=False, visual=True, gui_only=True, static=True, color=color)
        models.Box(world=task.world, name='_object_random_spawn_volume_with_height', position=self._object_spawn_position, orientation=(0, 0, 0, 1), size=self._object_random_spawn_volume, collision=False, visual=True, gui_only=True, static=True, color=color_with_height)
        if not gazebo.run(paused=True):
            raise RuntimeError('Failed to execute a paused Gazebo run')

class Manipulation(Task, Node, abc.ABC):
    _ids = count(0)

    def __init__(self, agent_rate: float, robot_model: str, workspace_frame_id: str, workspace_centre: Tuple[float, float, float], workspace_volume: Tuple[float, float, float], ignore_new_actions_while_executing: bool, use_servo: bool, scaling_factor_translation: float, scaling_factor_rotation: float, restrict_position_goal_to_workspace: bool, enable_gripper: bool, num_threads: int, **kwargs):
        self.id = next(self._ids)
        Task.__init__(self, agent_rate=agent_rate)
        try:
            rclpy.init()
        except Exception as e:
            if not rclpy.ok():
                sys.exit(f'ROS 2 context could not be initialised: {e}')
        Node.__init__(self, f'drl_grasping_{self.id}')
        self._callback_group = ReentrantCallbackGroup()
        if num_threads == 1:
            executor = SingleThreadedExecutor()
        elif num_threads > 1:
            executor = MultiThreadedExecutor(num_threads=num_threads)
        else:
            executor = MultiThreadedExecutor(num_threads=multiprocessing.cpu_count())
        executor.add_node(self)
        self._executor_thread = Thread(target=executor.spin, daemon=True, args=())
        self._executor_thread.start()
        self.robot_model_class = get_robot_model_class(robot_model)
        self.workspace_centre = (workspace_centre[0], workspace_centre[1], workspace_centre[2] + self.robot_model_class.BASE_LINK_Z_OFFSET)
        self.workspace_volume = workspace_volume
        self._restrict_position_goal_to_workspace = restrict_position_goal_to_workspace
        self._use_servo = use_servo
        self.__scaling_factor_translation = scaling_factor_translation
        self.__scaling_factor_rotation = scaling_factor_rotation
        self._enable_gripper = enable_gripper
        workspace_volume_half = (workspace_volume[0] / 2, workspace_volume[1] / 2, workspace_volume[2] / 2)
        self.workspace_min_bound = (self.workspace_centre[0] - workspace_volume_half[0], self.workspace_centre[1] - workspace_volume_half[1], self.workspace_centre[2] - workspace_volume_half[2])
        self.workspace_max_bound = (self.workspace_centre[0] + workspace_volume_half[0], self.workspace_centre[1] + workspace_volume_half[1], self.workspace_centre[2] + workspace_volume_half[2])
        self.robot_prefix = self.robot_model_class.DEFAULT_PREFIX
        if 0 == self.id:
            self.robot_name = self.robot_model_class.ROBOT_MODEL_NAME
        else:
            self.robot_name = f'{self.robot_model_class.ROBOT_MODEL_NAME}{self.id}'
            if self.robot_prefix.endswith('_'):
                self.robot_prefix = f'{self.robot_prefix[:-1]}{self.id}_'
            elif self.robot_prefix.empty():
                self.robot_prefix = f'robot{self.id}_'
        self.robot_base_link_name = self.robot_model_class.get_robot_base_link_name(self.robot_prefix)
        self.robot_arm_base_link_name = self.robot_model_class.get_arm_base_link_name(self.robot_prefix)
        self.robot_ee_link_name = self.robot_model_class.get_ee_link_name(self.robot_prefix)
        self.robot_arm_link_names = self.robot_model_class.get_arm_link_names(self.robot_prefix)
        self.robot_gripper_link_names = self.robot_model_class.get_gripper_link_names(self.robot_prefix)
        self.robot_arm_joint_names = self.robot_model_class.get_arm_joint_names(self.robot_prefix)
        self.robot_gripper_joint_names = self.robot_model_class.get_gripper_joint_names(self.robot_prefix)
        self.workspace_frame_id = self.substitute_special_frame(workspace_frame_id)
        self.initial_arm_joint_positions = self.robot_model_class.DEFAULT_ARM_JOINT_POSITIONS
        self.initial_gripper_joint_positions = self.robot_model_class.DEFAULT_GRIPPER_JOINT_POSITIONS
        self.terrain_name = 'terrain'
        self.object_names = []
        self.tf2_listener = Tf2Listener(node=self)
        self.tf2_broadcaster = Tf2Broadcaster(node=self)
        self.moveit2 = MoveIt2(node=self, joint_names=self.robot_arm_joint_names, base_link_name=self.robot_arm_base_link_name, end_effector_name=self.robot_ee_link_name, execute_via_moveit=False, ignore_new_calls_while_executing=ignore_new_actions_while_executing, callback_group=self._callback_group)
        if self._use_servo:
            self.servo = MoveIt2Servo(node=self, frame_id=self.robot_arm_base_link_name, linear_speed=scaling_factor_translation, angular_speed=scaling_factor_rotation, callback_group=self._callback_group)
        self.gripper = MoveIt2Gripper(node=self, gripper_joint_names=self.robot_gripper_joint_names, open_gripper_joint_positions=self.robot_model_class.OPEN_GRIPPER_JOINT_POSITIONS, closed_gripper_joint_positions=self.robot_model_class.CLOSED_GRIPPER_JOINT_POSITIONS, skip_planning=True, ignore_new_calls_while_executing=ignore_new_actions_while_executing, callback_group=self._callback_group)
        self.__task_parameter_overrides: Dict[str, any] = {}
        self._randomizer_parameter_overrides: Dict[str, any] = {}

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:
        action_space = self.create_action_space()
        observation_space = self.create_observation_space()
        return (action_space, observation_space)

    def create_action_space(self) -> ActionSpace:
        raise NotImplementedError()

    def create_observation_space(self) -> ObservationSpace:
        raise NotImplementedError()

    def set_action(self, action: Action):
        raise NotImplementedError()

    def get_observation(self) -> Observation:
        raise NotImplementedError()

    def get_reward(self) -> Reward:
        raise NotImplementedError()

    def is_done(self) -> bool:
        raise NotImplementedError()

    def reset_task(self):
        self.__consume_parameter_overrides()

    def get_relative_ee_position(self, translation: Tuple[float, float, float]) -> Tuple[float, float, float]:
        translation = self.scale_relative_translation(translation)
        current_position = self.get_ee_position()
        target_position = (current_position[0] + translation[0], current_position[1] + translation[1], current_position[2] + translation[2])
        if self._restrict_position_goal_to_workspace:
            target_position = self.restrict_position_goal_to_workspace(target_position)
        return target_position

    def get_relative_ee_orientation(self, rotation: Union[float, Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]], representation: str='quat') -> Tuple[float, float, float, float]:
        current_quat_xyzw = self.get_ee_orientation()
        if 'z' == representation:
            current_yaw = Rotation.from_quat(current_quat_xyzw).as_euler('xyz')[2]
            current_quat_xyzw = Rotation.from_euler('xyz', [np.pi, 0, current_yaw]).as_quat()
        relative_quat_xyzw = None
        if 'quat' == representation:
            relative_quat_xyzw = rotation
        elif '6d' == representation:
            vectors = tuple((rotation[x:x + 3] for x, _ in enumerate(rotation) if x % 3 == 0))
            relative_quat_xyzw = orientation_6d_to_quat(vectors[0], vectors[1])
        elif 'z' == representation:
            rotation = self.scale_relative_rotation(rotation)
            relative_quat_xyzw = Rotation.from_euler('xyz', [0, 0, rotation]).as_quat()
        target_quat_xyzw = quat_mul(current_quat_xyzw, relative_quat_xyzw)
        target_quat_xyzw /= np.linalg.norm(target_quat_xyzw)
        return target_quat_xyzw

    def scale_relative_translation(self, translation: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (self.__scaling_factor_translation * translation[0], self.__scaling_factor_translation * translation[1], self.__scaling_factor_translation * translation[2])

    def scale_relative_rotation(self, rotation: Union[float, Tuple[float, float, float], np.floating, np.ndarray]) -> float:
        if not hasattr(rotation, '__len__'):
            return self.__scaling_factor_rotation * rotation
        else:
            return (self.__scaling_factor_rotation * rotation[0], self.__scaling_factor_rotation * rotation[1], self.__scaling_factor_rotation * rotation[2])

    def restrict_position_goal_to_workspace(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (min(self.workspace_max_bound[0], max(self.workspace_min_bound[0], position[0])), min(self.workspace_max_bound[1], max(self.workspace_min_bound[1], position[1])), min(self.workspace_max_bound[2], max(self.workspace_min_bound[2], position[2])))

    def restrict_servo_translation_to_workspace(self, translation: Tuple[float, float, float]) -> Tuple[float, float, float]:
        current_ee_position = self.get_ee_position()
        translation = tuple((0.0 if current_ee_position[i] > self.workspace_max_bound[i] and translation[i] > 0.0 or (current_ee_position[i] < self.workspace_min_bound[i] and translation[i] < 0.0) else translation[i] for i in range(3)))
        return translation

    def get_ee_pose(self) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        """
        Return the current pose of the end effector with respect to arm base link.
        """
        try:
            robot_model = self.world.to_gazebo().get_model(self.robot_name).to_gazebo()
            ee_position, ee_quat_xyzw = get_model_pose(world=self.world, model=robot_model, link=self.robot_ee_link_name, xyzw=True)
            return transform_change_reference_frame_pose(world=self.world, position=ee_position, quat=ee_quat_xyzw, target_model=robot_model, target_link=self.robot_arm_base_link_name, xyzw=True)
        except Exception as e:
            self.get_logger().warn(f'Cannot get end effector pose from Gazebo ({e}), using tf2...')
            transform = self.tf2_listener.lookup_transform_sync(source_frame=self.robot_ee_link_name, target_frame=self.robot_arm_base_link_name, retry=False)
            if transform is not None:
                return ((transform.translation.x, transform.translation.y, transform.translation.z), (transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w))
            else:
                self.get_logger().error('Cannot get pose of the end effector (default values are returned)')
                return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def get_ee_position(self) -> Tuple[float, float, float]:
        """
        Return the current position of the end effector with respect to arm base link.
        """
        try:
            robot_model = self.world.to_gazebo().get_model(self.robot_name).to_gazebo()
            ee_position = get_model_position(world=self.world, model=robot_model, link=self.robot_ee_link_name)
            return transform_change_reference_frame_position(world=self.world, position=ee_position, target_model=robot_model, target_link=self.robot_arm_base_link_name)
        except Exception as e:
            self.get_logger().warn(f'Cannot get end effector position from Gazebo ({e}), using tf2...')
            transform = self.tf2_listener.lookup_transform_sync(source_frame=self.robot_ee_link_name, target_frame=self.robot_arm_base_link_name, retry=False)
            if transform is not None:
                return (transform.translation.x, transform.translation.y, transform.translation.z)
            else:
                self.get_logger().error('Cannot get position of the end effector (default values are returned)')
                return (0.0, 0.0, 0.0)

    def get_ee_orientation(self) -> Tuple[float, float, float, float]:
        """
        Return the current xyzw quaternion of the end effector with respect to arm base link.
        """
        try:
            robot_model = self.world.to_gazebo().get_model(self.robot_name).to_gazebo()
            ee_quat_xyzw = get_model_orientation(world=self.world, model=robot_model, link=self.robot_ee_link_name, xyzw=True)
            return transform_change_reference_frame_orientation(world=self.world, quat=ee_quat_xyzw, target_model=robot_model, target_link=self.robot_arm_base_link_name, xyzw=True)
        except Exception as e:
            self.get_logger().warn(f'Cannot get end effector orientation from Gazebo ({e}), using tf2...')
            transform = self.tf2_listener.lookup_transform_sync(source_frame=self.robot_ee_link_name, target_frame=self.robot_arm_base_link_name, retry=False)
            if transform is not None:
                return (transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
            else:
                self.get_logger().error('Cannot get orientation of the end effector (default values are returned)')
                return (0.0, 0.0, 0.0, 1.0)

    def get_object_position(self, object_model: Union[ModelWrapper, str]) -> Tuple[float, float, float]:
        """
        Return the current position of an object with respect to arm base link.
        Note: Only simulated objects are currently supported.
        """
        try:
            object_position = get_model_position(world=self.world, model=object_model)
            return transform_change_reference_frame_position(world=self.world, position=object_position, target_model=self.robot_name, target_link=self.robot_arm_base_link_name)
        except Exception as e:
            self.get_logger().error(f'Cannot get position of {object_model} object (default values are returned): {e}')
            return (0.0, 0.0, 0.0)

    def get_object_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Return the current position of all objects with respect to arm base link.
        Note: Only simulated objects are currently supported.
        """
        object_positions = {}
        try:
            robot_model = self.world.to_gazebo().get_model(self.robot_name).to_gazebo()
            robot_arm_base_link = robot_model.get_link(link_name=self.robot_arm_base_link_name)
            for object_name in self.object_names:
                object_position = get_model_position(world=self.world, model=object_name)
                object_positions[object_name] = transform_change_reference_frame_position(world=self.world, position=object_position, target_model=robot_model, target_link=robot_arm_base_link)
        except Exception as e:
            self.get_logger().error(f'Cannot get positions of all objects (empty Dict is returned): {e}')
        return object_positions

    def substitute_special_frame(self, frame_id: str) -> str:
        if 'arm_base_link' == frame_id:
            return self.robot_arm_base_link_name
        elif 'base_link' == frame_id:
            return self.robot_base_link_name
        elif 'end_effector' == frame_id:
            return self.robot_ee_link_name
        elif 'world' == frame_id:
            try:
                return self.world.to_gazebo().name()
            except Exception as e:
                self.get_logger().warn(f'')
                return 'drl_grasping_world'
        else:
            return frame_id

    def wait_until_action_executed(self):
        if self._use_servo:
            rate = self.create_rate(self.agent_rate)
            try:
                if rclpy.ok():
                    rate.sleep()
            except KeyboardInterrupt:
                pass
        self.moveit2.wait_until_executed()
        if self._enable_gripper:
            self.gripper.wait_until_executed()

    def move_to_initial_joint_configuration(self):
        self.moveit2.move_to_configuration(self.initial_arm_joint_positions)
        if self.robot_model_class.CLOSED_GRIPPER_JOINT_POSITIONS == self.initial_gripper_joint_positions:
            self.gripper.reset_close()
        else:
            self.gripper.reset_open()

    def check_terrain_collision(self) -> bool:
        """
        Returns true if robot links are in collision with the ground.
        """
        robot_name_len = len(self.robot_name)
        for contact in self.world.get_model(self.terrain_name).contacts():
            if len(contact.body_b) > robot_name_len:
                if contact.body_b[:robot_name_len] == self.robot_name:
                    link = contact.body_b[len(self.robot_name) + 2:]
                    if not self.robot_base_link_name == link and (link in self.robot_arm_link_names or link in self.robot_gripper_link_names):
                        return True
        return False

    def check_all_objects_outside_workspace(self, object_positions: Dict[str, Tuple[float, float, float]]) -> bool:
        """
        Returns true if all objects are outside the workspace
        """
        return all([self.check_object_outside_workspace(object_position) for object_position in object_positions.values()])

    def check_object_outside_workspace(self, object_position: Tuple[float, float, float]) -> bool:
        """
        Returns true if the object is outside the workspace
        """
        return object_position[0] < self.workspace_min_bound[0] or object_position[1] < self.workspace_min_bound[1] or object_position[2] < self.workspace_min_bound[2] or (object_position[0] > self.workspace_max_bound[0]) or (object_position[1] > self.workspace_max_bound[1]) or (object_position[2] > self.workspace_max_bound[2])

    def add_parameter_overrides(self, parameter_overrides: Dict[str, any]):
        self.add_task_parameter_overrides(parameter_overrides)
        self.add_randomizer_parameter_overrides(parameter_overrides)

    def add_task_parameter_overrides(self, parameter_overrides: Dict[str, any]):
        self.__task_parameter_overrides.update(parameter_overrides)

    def add_randomizer_parameter_overrides(self, parameter_overrides: Dict[str, any]):
        self._randomizer_parameter_overrides.update(parameter_overrides)

    def __consume_parameter_overrides(self):
        for key, value in self.__task_parameter_overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self, f'_{key}'):
                setattr(self, f'_{key}', value)
            elif hasattr(self, f'__{key}'):
                setattr(self, f'__{key}', value)
            else:
                self.get_logger().error(f"Override '{key}' is not supperted by the task.")
        self.__task_parameter_overrides.clear()

class Grasp(Manipulation, abc.ABC):

    def __init__(self, gripper_dead_zone: float, full_3d_orientation: bool, obs_n_stacked: int=1, preload_replay_buffer: bool=False, **kwargs):
        Manipulation.__init__(self, **kwargs)
        self.curriculum = GraspCurriculum(task=self, **kwargs)
        self.__gripper_dead_zone = gripper_dead_zone
        self.__full_3d_orientation = full_3d_orientation
        self.__preload_replay_buffer = preload_replay_buffer
        self._obs_n_stacked = obs_n_stacked
        self.__stacked_obs = deque([], maxlen=self._obs_n_stacked)

    def create_action_space(self) -> ActionSpace:
        if self.__full_3d_orientation:
            if self._use_servo:
                return gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
            else:
                return gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        else:
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def create_observation_space(self) -> ObservationSpace:
        return gym.spaces.Box(low=np.array((-1.0, -np.inf, -np.inf, -np.inf, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf, -np.inf) * self._obs_n_stacked), high=np.array((1.0, np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf) * self._obs_n_stacked), shape=(13 * self._obs_n_stacked,), dtype=np.float32)

    def set_action(self, action: Action):
        if self.__preload_replay_buffer:
            action = self._demonstrate_action()
        self.get_logger().debug(f'action: {action}')
        gripper_action = action[0]
        if gripper_action < -self.__gripper_dead_zone:
            self.gripper.close()
        elif gripper_action > self.__gripper_dead_zone:
            self.gripper.open()
        else:
            pass
        if self._use_servo:
            linear = action[1:4]
            if self._restrict_position_goal_to_workspace:
                linear = self.restrict_servo_translation_to_workspace(linear)
            if self.__full_3d_orientation:
                angular = action[4:7]
            else:
                angular = [0.0, 0.0, action[4]]
            self.servo(linear=linear, angular=angular)
        else:
            position = self.get_relative_ee_position(action[1:4])
            if self.__full_3d_orientation:
                quat_xyzw = self.get_relative_ee_orientation(rotation=action[4:10], representation='6d')
            else:
                quat_xyzw = self.get_relative_ee_orientation(rotation=action[4], representation='z')
            self.moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw)

    def get_observation(self) -> Observation:
        ee_position, ee_orientation = self.get_ee_pose()
        ee_position = np.array(ee_position, dtype=np.float32)
        ee_orientation = np.array(orientation_quat_to_6d(quat_xyzw=ee_orientation), dtype=np.float32)
        object_positions = np.array(tuple(self.get_object_positions().values()), dtype=np.float32)
        nearest_object_position = get_nearest_point(origin=ee_position, points=object_positions)
        obs = np.concatenate([(1.0 if self.gripper.is_open else -1.0,), ee_position, ee_orientation[0], ee_orientation[1], nearest_object_position], dtype=np.float32)
        if self._obs_n_stacked > 1:
            self.__stacked_obs.append(obs)
            while not self._obs_n_stacked == len(self.__stacked_obs):
                self.__stacked_obs.append(obs)
            observation = Observation(np.concatenate(self.__stacked_obs, dtype=np.float32))
        else:
            observation = Observation(obs)
        self.get_logger().debug(f'\nobservation: {observation}')
        return observation

    def get_reward(self) -> Reward:
        return self.curriculum.get_reward()

    def is_done(self) -> bool:
        return self.curriculum.is_done()

    def get_info(self) -> Dict:
        info = self.curriculum.get_info()
        if self.__preload_replay_buffer:
            info.update({'actual_actions': self.__actual_actions})
        return info

    def reset_task(self):
        Manipulation.reset_task(self)
        self.curriculum.reset_task()

    def get_touched_objects(self) -> List[str]:
        """
        Returns list of all objects that are in contact with any finger.
        """
        robot = self.world.get_model(self.robot_name).to_gazebo()
        touched_objects = []
        for gripper_link_name in self.robot_gripper_link_names:
            finger = robot.get_link(link_name=gripper_link_name)
            finger_contacts = finger.contacts()
            for contact in finger_contacts:
                model_name = contact.body_b.split('::', 1)[0]
                if model_name not in touched_objects and any((object_name in model_name for object_name in self.object_names)):
                    touched_objects.append(model_name)
        return touched_objects

    def get_grasped_objects(self, min_angle_between_two_contact: float=np.pi / 8) -> List[str]:
        """
        Returns list of all currently grasped objects.
        Grasped object must be in contact with all gripper links (fingers) and their contact normals must be dissimilar.
        """
        if self.gripper.is_open:
            return []
        robot = self.world.get_model(self.robot_name)
        grasp_candidates = {}
        for gripper_link_name in self.robot_gripper_link_names:
            finger = robot.to_gazebo().get_link(link_name=gripper_link_name)
            finger_contacts = finger.contacts()
            if 0 == len(finger_contacts):
                continue
            for contact in finger_contacts:
                model_name = contact.body_b.split('::', 1)[0]
                if any((object_name in model_name for object_name in self.object_names)):
                    if model_name not in grasp_candidates:
                        grasp_candidates[model_name] = []
                    grasp_candidates[model_name].append(contact.points)
        grasped_objects = []
        for model_name, contact_points_list in grasp_candidates.items():
            if len(contact_points_list) < 2:
                continue
            average_normals = []
            for contact_points in contact_points_list:
                average_normal = np.array([0.0, 0.0, 0.0])
                for point in contact_points:
                    average_normal += point.normal
                average_normal /= np.linalg.norm(average_normal)
                average_normals.append(average_normal)
            normal_angles = []
            for n1, n2 in itertools.combinations(average_normals, 2):
                normal_angles.append(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
            sufficient_angle = min_angle_between_two_contact
            for angle in normal_angles:
                if angle > sufficient_angle:
                    grasped_objects.append(model_name)
                    break
        return grasped_objects

    def _demonstrate_action(self) -> np.ndarray:
        self.__actual_actions = np.zeros(self.action_space.shape)
        ee_position, ee_orientation = self.get_ee_pose()
        ee_position = np.array(ee_position)
        ee_orientation = np.array(ee_orientation)
        object_position = np.array(self.get_object_position(self.object_names[0]))
        distance = object_position - ee_position
        distance_mag = np.linalg.norm(distance)
        if distance_mag < 0.02:
            if self.gripper.is_open:
                self.__actual_actions[0] = -1.0
                self.__actual_actions[1:4] = np.zeros((3,))
            else:
                self.__actual_actions[0] = -1.0
                self.__actual_actions[1:4] = np.array((0.0, 0.0, 1.0))
            if self.__full_3d_orientation:
                pass
            else:
                self.__actual_actions[4] = 0.0
        else:
            self.__actual_actions[0] = 1.0
            if distance_mag > self._relative_position_scaling_factor:
                relative_position = distance / distance_mag
            else:
                relative_position = distance / self._relative_position_scaling_factor
            self.__actual_actions[1:4] = relative_position
            distance_mag_xy = np.linalg.norm(distance[:2])
            if distance_mag_xy > 0.01 and ee_position[2] < 0.1:
                self.__actual_actions[3] = max(0.0, self.__actual_actions[3])
            object_orientation = quat_to_xyzw(np.array(self.get_object_orientation(self.object_names[0])))
            if self.__full_3d_orientation:
                pass
            else:
                current_ee_yaw = Rotation.from_quat(ee_orientation).as_euler('xyz')[2]
                current_object_yaw = Rotation.from_quat(object_orientation).as_euler('xyz')[2]
                yaw_diff = current_object_yaw - current_ee_yaw
                if yaw_diff > np.pi:
                    yaw_diff -= np.pi / 2
                elif yaw_diff < -np.pi:
                    yaw_diff += np.pi / 2
                yaw_diff = min(1.0, 1.0 / (self._z_relative_orientation_scaling_factor / yaw_diff))
                self.__actual_actions[4] = yaw_diff
        if ee_position[2] < 0.025:
            self.__actual_actions[3] = max(0.0, self.__actual_actions[3])
        return self.__actual_actions

