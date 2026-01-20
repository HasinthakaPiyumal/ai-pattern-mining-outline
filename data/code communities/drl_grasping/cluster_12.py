# Cluster 12

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

class Reach(Manipulation, abc.ABC):

    def __init__(self, sparse_reward: bool, act_quick_reward: float, required_accuracy: float, **kwargs):
        Manipulation.__init__(self, **kwargs)
        self._sparse_reward: bool = sparse_reward
        self._act_quick_reward = act_quick_reward if act_quick_reward >= 0.0 else -act_quick_reward
        self._required_accuracy: float = required_accuracy
        self._is_done: bool = False
        self._previous_distance: float = None
        self.initial_gripper_joint_positions = self.robot_model_class.CLOSED_GRIPPER_JOINT_POSITIONS

    def create_action_space(self) -> ActionSpace:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def create_observation_space(self) -> ObservationSpace:
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def set_action(self, action: Action):
        self.get_logger().debug(f'action: {action}')
        if self._use_servo:
            linear = action[0:3]
            self.servo(linear=linear)
        else:
            position = self.get_relative_ee_position(action[0:3])
            quat_xyzw = (1.0, 0.0, 0.0, 0.0)
            self.moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw)

    def get_observation(self) -> Observation:
        ee_position = self.get_ee_position()
        target_position = self.get_object_position(object_model=self.object_names[0])
        observation = Observation(np.concatenate([ee_position, target_position], dtype=np.float32))
        self.get_logger().debug(f'\nobservation: {observation}')
        return observation

    def get_reward(self) -> Reward:
        reward = 0.0
        current_distance = self.get_distance_to_target()
        if current_distance < self._required_accuracy:
            self._is_done = True
            if self._sparse_reward:
                reward += 1.0
        if not self._sparse_reward:
            reward += self._previous_distance - current_distance
            self._previous_distance = current_distance
        reward -= self._act_quick_reward
        self.get_logger().debug(f'reward: {reward}')
        return Reward(reward)

    def is_done(self) -> bool:
        done = self._is_done
        self.get_logger().debug(f'done: {done}')
        return done

    def reset_task(self):
        Manipulation.reset_task(self)
        self._is_done = False
        if not self._sparse_reward:
            self._previous_distance = self.get_distance_to_target()
        self.get_logger().debug(f'\ntask reset')

    def get_distance_to_target(self) -> Tuple[float, float, float]:
        ee_position = self.get_ee_position()
        object_position = self.get_object_position(object_model=self.object_names[0])
        return distance_to_nearest_point(origin=ee_position, points=[object_position])

