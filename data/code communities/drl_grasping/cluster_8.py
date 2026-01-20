# Cluster 8

class GraspPlanetaryDepthImage(GraspPlanetary, abc.ABC):

    def __init__(self, depth_max_distance: float, image_include_color: bool, image_include_intensity: bool, image_n_stacked: int, proprioceptive_observations: bool, camera_type: str='rgbd_camera', camera_width: int=128, camera_height: int=128, **kwargs):
        GraspPlanetary.__init__(self, **kwargs)
        self.camera_sub = CameraSubscriber(node=self, topic=Camera.get_depth_topic(camera_type), is_point_cloud=False, callback_group=self._callback_group)
        if image_include_color or image_include_intensity:
            assert camera_type == 'rgbd_camera'
            self.camera_sub_color = CameraSubscriber(node=self, topic=Camera.get_color_topic(camera_type), is_point_cloud=False, callback_group=self._callback_group)
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._depth_max_distance = depth_max_distance
        self._image_n_stacked = image_n_stacked
        self._image_include_color = image_include_color
        self._image_include_intensity = image_include_intensity
        self._proprioceptive_observations = proprioceptive_observations
        self._num_pixels = camera_height * camera_width
        self.__stacked_images = deque([], maxlen=self._image_n_stacked)

    def create_observation_space(self) -> ObservationSpace:
        size = self._num_pixels
        if self._image_include_color:
            size += 3 * self._num_pixels
        elif self._image_include_intensity:
            size += self._num_pixels
        if self._proprioceptive_observations:
            size += 11
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self._image_n_stacked, size), dtype=np.float32)

    def get_observation(self) -> Observation:
        depth_image_msg = self.camera_sub.get_observation()
        img_res = depth_image_msg.height * depth_image_msg.width
        if 2 * img_res == len(depth_image_msg.data):
            depth_data_type = np.float16
        else:
            depth_data_type = np.float32
        if depth_image_msg.height != self._camera_width or depth_image_msg.width != self._camera_height:
            import cv2
            depth_image = np.ndarray(buffer=depth_image_msg.data, dtype=depth_data_type, shape=(depth_image_msg.height, depth_image_msg.width)).astype(dtype=np.float32)
            if depth_image_msg.height > depth_image_msg.width:
                diff = depth_image_msg.height - depth_image_msg.width
                diff_2 = diff // 2
                depth_image = depth_image[diff_2:-diff_2, :]
            elif depth_image_msg.height < depth_image_msg.width:
                diff = depth_image_msg.width - depth_image_msg.height
                diff_2 = diff // 2
                depth_image = depth_image[:, diff_2:-diff_2]
            depth_image = cv2.resize(depth_image, dsize=(self._camera_height, self._camera_width), interpolation=cv2.INTER_CUBIC).reshape(self._num_pixels)
        else:
            depth_image = np.ndarray(buffer=depth_image_msg.data, dtype=depth_data_type, shape=(self._num_pixels,)).astype(dtype=np.float32)
        np.nan_to_num(depth_image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        depth_image[depth_image > self._depth_max_distance] = self._depth_max_distance
        depth_image = depth_image / self._depth_max_distance
        if self._image_include_color or self._image_include_intensity:
            color_image_msg = self.camera_sub_color.get_observation()
            if color_image_msg.height != self._camera_width or color_image_msg.width != self._camera_height:
                import cv2
                color_image = np.ndarray(buffer=color_image_msg.data, dtype=np.uint8, shape=(color_image_msg.height, color_image_msg.width, 3))
                if color_image_msg.height > color_image_msg.width:
                    diff = color_image_msg.height - color_image_msg.width
                    diff_2 = diff // 2
                    color_image = color_image[diff_2:-diff_2, :, :]
                elif color_image_msg.height < color_image_msg.width:
                    diff = color_image_msg.width - color_image_msg.height
                    diff_2 = diff // 2
                    color_image = color_image[:, diff_2:-diff_2, :]
                color_image = cv2.resize(color_image, dsize=(self._camera_width, self._camera_height), interpolation=cv2.INTER_CUBIC).reshape(3 * self._num_pixels)
            else:
                color_image = np.ndarray(buffer=color_image_msg.data, dtype=np.uint8, shape=(3 * self._num_pixels,))
            if self._image_include_intensity:
                color_image = color_image.reshape(self._camera_width, self._camera_height, 3)[:, :, 0].reshape(-1)
            color_image.astype(dtype=np.float32)
            color_image = color_image / 255.0
            depth_image = np.concatenate((depth_image, color_image))
        if self._proprioceptive_observations:
            depth_image = np.pad(depth_image, (0, 11), 'constant', constant_values=0)
            depth_image[-1] = np.array(10, dtype=np.float32)
            ee_position, ee_orientation = self.get_ee_pose()
            ee_orientation = orientation_quat_to_6d(quat_xyzw=ee_orientation)
            aux_obs = (1.0 if self.gripper.is_open else -1.0,) + ee_position + ee_orientation[0] + ee_orientation[1]
            depth_image[-11:-1] = np.array(aux_obs, dtype=np.float32)
        self.__stacked_images.append(depth_image)
        while not self._image_n_stacked == len(self.__stacked_images):
            self.__stacked_images.append(depth_image)
        observation = Observation(np.array(self.__stacked_images, dtype=np.uint8))
        self.get_logger().debug(f'\nobservation: {observation}')
        return observation

    def reset_task(self):
        self.__stacked_images.clear()
        GraspPlanetary.reset_task(self)

def orientation_quat_to_6d(quat_xyzw: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    rot_mat = Rotation.from_quat(quat_xyzw).as_matrix()
    return (tuple(rot_mat[:, 0]), tuple(rot_mat[:, 1]))

class GraspPlanetaryOctree(GraspPlanetary, abc.ABC):

    def __init__(self, octree_reference_frame_id: str, octree_min_bound: Tuple[float, float, float], octree_max_bound: Tuple[float, float, float], octree_depth: int, octree_full_depth: int, octree_include_color: bool, octree_include_intensity: bool, octree_n_stacked: int, octree_max_size: int, proprioceptive_observations: bool, camera_type: str='rgbd_camera', **kwargs):
        GraspPlanetary.__init__(self, **kwargs)
        self.camera_sub = CameraSubscriber(node=self, topic=Camera.get_points_topic(camera_type), is_point_cloud=True, callback_group=self._callback_group)
        octree_min_bound = (octree_min_bound[0], octree_min_bound[1], octree_min_bound[2] + self.robot_model_class.BASE_LINK_Z_OFFSET)
        octree_max_bound = (octree_max_bound[0], octree_max_bound[1], octree_max_bound[2] + self.robot_model_class.BASE_LINK_Z_OFFSET)
        self.octree_creator = OctreeCreator(node=self, tf2_listener=self.tf2_listener, reference_frame_id=self.substitute_special_frame(octree_reference_frame_id), min_bound=octree_min_bound, max_bound=octree_max_bound, include_color=octree_include_color, include_intensity=octree_include_intensity, depth=octree_depth, full_depth=octree_full_depth)
        self._octree_n_stacked = octree_n_stacked
        self._octree_max_size = octree_max_size
        self._proprioceptive_observations = proprioceptive_observations
        self.__stacked_octrees = deque([], maxlen=self._octree_n_stacked)

    def create_observation_space(self) -> ObservationSpace:
        return gym.spaces.Box(low=0, high=255, shape=(self._octree_n_stacked, self._octree_max_size), dtype=np.uint8)

    def get_observation(self) -> Observation:
        point_cloud = self.camera_sub.get_observation()
        octree = self.octree_creator(point_cloud).numpy()
        octree_size = octree.shape[0]
        if octree_size > self._octree_max_size:
            self.get_logger().error(f'Octree is larger than the maximum allowed size of {self._octree_max_size} (exceeded with {octree_size})')
        octree = np.pad(octree, (0, self._octree_max_size - octree_size), 'constant', constant_values=0)
        octree[-4:] = np.ndarray(buffer=np.array([octree_size], dtype=np.uint32).tobytes(), shape=(4,), dtype=np.uint8)
        if self._proprioceptive_observations:
            octree[-8:-4] = np.ndarray(buffer=np.array([10], dtype=np.uint32).tobytes(), shape=(4,), dtype=np.uint8)
            ee_position, ee_orientation = self.get_ee_pose()
            ee_orientation = orientation_quat_to_6d(quat_xyzw=ee_orientation)
            aux_obs = (1.0 if self.gripper.is_open else -1.0,) + ee_position + ee_orientation[0] + ee_orientation[1]
            octree[-48:-8] = np.ndarray(buffer=np.array(aux_obs, dtype=np.float32).tobytes(), shape=(40,), dtype=np.uint8)
        self.__stacked_octrees.append(octree)
        while not self._octree_n_stacked == len(self.__stacked_octrees):
            self.__stacked_octrees.append(octree)
        observation = Observation(np.array(self.__stacked_octrees, dtype=np.uint8))
        self.get_logger().debug(f'\nobservation: {observation}')
        return observation

    def reset_task(self):
        self.__stacked_octrees.clear()
        GraspPlanetary.reset_task(self)

class GraspOctree(Grasp, abc.ABC):

    def __init__(self, octree_reference_frame_id: str, octree_min_bound: Tuple[float, float, float], octree_max_bound: Tuple[float, float, float], octree_depth: int, octree_full_depth: int, octree_include_color: bool, octree_include_intensity: bool, octree_n_stacked: int, octree_max_size: int, proprioceptive_observations: bool, camera_type: str='rgbd_camera', **kwargs):
        Grasp.__init__(self, **kwargs)
        self.camera_sub = CameraSubscriber(node=self, topic=Camera.get_points_topic(camera_type), is_point_cloud=True, callback_group=self._callback_group)
        octree_min_bound = (octree_min_bound[0], octree_min_bound[1], octree_min_bound[2] + self.robot_model_class.BASE_LINK_Z_OFFSET)
        octree_max_bound = (octree_max_bound[0], octree_max_bound[1], octree_max_bound[2] + self.robot_model_class.BASE_LINK_Z_OFFSET)
        self.octree_creator = OctreeCreator(node=self, tf2_listener=self.tf2_listener, reference_frame_id=self.substitute_special_frame(octree_reference_frame_id), min_bound=octree_min_bound, max_bound=octree_max_bound, include_color=octree_include_color, include_intensity=octree_include_intensity, depth=octree_depth, full_depth=octree_full_depth)
        self._octree_n_stacked = octree_n_stacked
        self._octree_max_size = octree_max_size
        self._proprioceptive_observations = proprioceptive_observations
        self.__stacked_octrees = deque([], maxlen=self._octree_n_stacked)

    def create_observation_space(self) -> ObservationSpace:
        return gym.spaces.Box(low=0, high=255, shape=(self._octree_n_stacked, self._octree_max_size), dtype=np.uint8)

    def get_observation(self) -> Observation:
        point_cloud = self.camera_sub.get_observation()
        octree = self.octree_creator(point_cloud).numpy()
        octree_size = octree.shape[0]
        if octree_size > self._octree_max_size:
            self.get_logger().error(f'Octree is larger than the maximum allowed size of {self._octree_max_size} (exceeded with {octree_size})')
        octree = np.pad(octree, (0, self._octree_max_size - octree_size), 'constant', constant_values=0)
        octree[-4:] = np.ndarray(buffer=np.array([octree_size], dtype=np.uint32).tobytes(), shape=(4,), dtype=np.uint8)
        if self._proprioceptive_observations:
            octree[-8:-4] = np.ndarray(buffer=np.array([10], dtype=np.uint32).tobytes(), shape=(4,), dtype=np.uint8)
            ee_position, ee_orientation = self.get_ee_pose()
            ee_orientation = orientation_quat_to_6d(quat_xyzw=ee_orientation)
            aux_obs = (1.0 if self.gripper.is_open else -1.0,) + ee_position + ee_orientation[0] + ee_orientation[1]
            octree[-48:-8] = np.ndarray(buffer=np.array(aux_obs, dtype=np.float32).tobytes(), shape=(40,), dtype=np.uint8)
        self.__stacked_octrees.append(octree)
        while not self._octree_n_stacked == len(self.__stacked_octrees):
            self.__stacked_octrees.append(octree)
        observation = Observation(np.array(self.__stacked_octrees, dtype=np.uint8))
        self.get_logger().debug(f'\nobservation: {observation}')
        return observation

    def reset_task(self):
        self.__stacked_octrees.clear()
        Grasp.reset_task(self)

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

def get_nearest_point(origin: Tuple[float, float, float], points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    target_distances = np.linalg.norm(np.array(points) - np.array(origin), axis=1)
    return points[target_distances.argmin()]

