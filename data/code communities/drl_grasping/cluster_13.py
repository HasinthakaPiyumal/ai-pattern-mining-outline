# Cluster 13

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

def distance_to_nearest_point(origin: Tuple[float, float, float], points: List[Tuple[float, float, float]]) -> float:
    return np.linalg.norm(np.array(points) - np.array(origin), axis=1).min()

class GraspCurriculum(StageRewardCurriculum, SuccessRateImpl, WorkspaceScaleCurriculum, ObjectSpawnVolumeScaleCurriculum, ObjectCountCurriculum, ArmStuckChecker):
    """
    Curriculum learning implementation for grasp task that provides termination (success/fail) and reward for each stage of the task.
    """

    def __init__(self, task: Task, stages_base_reward: float, reach_required_distance: float, lift_required_height: float, persistent_reward_each_step: float, persistent_reward_terrain_collision: float, persistent_reward_all_objects_outside_workspace: float, persistent_reward_arm_stuck: float, enable_stage_reward_curriculum: bool, enable_workspace_scale_curriculum: bool, enable_object_spawn_volume_scale_curriculum: bool, enable_object_count_curriculum: bool, reach_required_distance_min: Optional[float]=None, reach_required_distance_max: Optional[float]=None, reach_required_distance_max_threshold: Optional[float]=None, lift_required_height_min: Optional[float]=None, lift_required_height_max: Optional[float]=None, lift_required_height_max_threshold: Optional[float]=None, **kwargs):
        StageRewardCurriculum.__init__(self, curriculum_stage=GraspStage, **kwargs)
        SuccessRateImpl.__init__(self, **kwargs)
        WorkspaceScaleCurriculum.__init__(self, task=task, success_rate_impl=self, **kwargs)
        ObjectSpawnVolumeScaleCurriculum.__init__(self, task=task, success_rate_impl=self, **kwargs)
        ObjectCountCurriculum.__init__(self, task=task, success_rate_impl=self, **kwargs)
        ArmStuckChecker.__init__(self, task=task, **kwargs)
        self.__task = task
        self.__stages_base_reward = stages_base_reward
        self.reach_required_distance = reach_required_distance
        self.lift_required_height = lift_required_height
        self.__persistent_reward_each_step = persistent_reward_each_step
        self.__persistent_reward_terrain_collision = persistent_reward_terrain_collision
        self.__persistent_reward_all_objects_outside_workspace = persistent_reward_all_objects_outside_workspace
        self.__persistent_reward_arm_stuck = persistent_reward_arm_stuck
        self.__enable_stage_reward_curriculum = enable_stage_reward_curriculum
        self.__enable_workspace_scale_curriculum = enable_workspace_scale_curriculum
        self.__enable_object_spawn_volume_scale_curriculum = enable_object_spawn_volume_scale_curriculum
        self.__enable_object_count_curriculum = enable_object_count_curriculum
        if self.__persistent_reward_each_step > 0.0:
            self.__persistent_reward_each_step *= -1.0
        if self.__persistent_reward_terrain_collision > 0.0:
            self.__persistent_reward_terrain_collision *= -1.0
        if self.__persistent_reward_all_objects_outside_workspace > 0.0:
            self.__persistent_reward_all_objects_outside_workspace *= -1.0
        if self.__persistent_reward_arm_stuck > 0.0:
            self.__persistent_reward_arm_stuck *= -1.0
        reach_required_distance_min = reach_required_distance_min if reach_required_distance_min is not None else reach_required_distance
        reach_required_distance_max = reach_required_distance_max if reach_required_distance_max is not None else reach_required_distance
        reach_required_distance_max_threshold = reach_required_distance_max_threshold if reach_required_distance_max_threshold is not None else 0.5
        self.__reach_required_distance_curriculum_enabled = not reach_required_distance_min == reach_required_distance_max
        if self.__reach_required_distance_curriculum_enabled:
            self.__reach_required_distance_curriculum = AttributeCurriculum(success_rate_impl=self, attribute_owner=self, attribute_name='reach_required_distance', initial_value=reach_required_distance_min, target_value=reach_required_distance_max, target_value_threshold=reach_required_distance_max_threshold)
        lift_required_height_min = lift_required_height_min if lift_required_height_min is not None else lift_required_height
        lift_required_height_max = lift_required_height_max if lift_required_height_max is not None else lift_required_height
        lift_required_height_max_threshold = lift_required_height_max_threshold if lift_required_height_max_threshold is not None else 0.5
        lift_required_height += task.robot_model_class.BASE_LINK_Z_OFFSET
        lift_required_height_min += task.robot_model_class.BASE_LINK_Z_OFFSET
        lift_required_height_max += task.robot_model_class.BASE_LINK_Z_OFFSET
        lift_required_height_max_threshold += task.robot_model_class.BASE_LINK_Z_OFFSET
        self.__lift_required_height_curriculum_enabled = not lift_required_height_min == lift_required_height_max
        if self.__lift_required_height_curriculum_enabled:
            self.__lift_required_height_curriculum = AttributeCurriculum(success_rate_impl=self, attribute_owner=self, attribute_name='lift_required_height', initial_value=lift_required_height_min, target_value=lift_required_height_max, target_value_threshold=lift_required_height_max_threshold)

    def get_reward(self) -> Reward:
        if self.__enable_stage_reward_curriculum:
            return StageRewardCurriculum.get_reward(self, ee_position=self.__task.get_ee_position(), object_positions=self.__task.get_object_positions(), touched_objects=self.__task.get_touched_objects(), grasped_objects=self.__task.get_grasped_objects())
        else:
            return StageRewardCurriculum.get_reward(self, only_last_stage=True, object_positions=self.__task.get_object_positions(), grasped_objects=self.__task.get_grasped_objects())

    def is_done(self) -> bool:
        return StageRewardCurriculum.is_done(self)

    def get_info(self) -> Dict:
        info = StageRewardCurriculum.get_info(self)
        info.update(SuccessRateImpl.get_info(self))
        if self.__enable_workspace_scale_curriculum:
            info.update(WorkspaceScaleCurriculum.get_info(self))
        if self.__enable_object_spawn_volume_scale_curriculum:
            info.update(ObjectSpawnVolumeScaleCurriculum.get_info(self))
        if self.__enable_object_count_curriculum:
            info.update(ObjectCountCurriculum.get_info(self))
        if self.__persistent_reward_arm_stuck:
            info.update(ArmStuckChecker.get_info(self))
        if self.__reach_required_distance_curriculum_enabled:
            info.update(self.__reach_required_distance_curriculum.get_info())
        if self.__lift_required_height_curriculum_enabled:
            info.update(self.__lift_required_height_curriculum.get_info())
        return info

    def reset_task(self):
        StageRewardCurriculum.reset_task(self)
        if self.__enable_workspace_scale_curriculum:
            WorkspaceScaleCurriculum.reset_task(self)
        if self.__enable_object_spawn_volume_scale_curriculum:
            ObjectSpawnVolumeScaleCurriculum.reset_task(self)
        if self.__enable_object_count_curriculum:
            ObjectCountCurriculum.reset_task(self)
        if self.__persistent_reward_arm_stuck:
            ArmStuckChecker.reset_task(self)
        if self.__reach_required_distance_curriculum_enabled:
            self.__reach_required_distance_curriculum.reset_task()
        if self.__lift_required_height_curriculum_enabled:
            self.__lift_required_height_curriculum.reset_task()

    def on_episode_success(self):
        self.update_success_rate(is_success=True)

    def on_episode_failure(self):
        self.update_success_rate(is_success=False)

    def on_episode_timeout(self):
        self.update_success_rate(is_success=False)

    def get_reward_REACH(self, ee_position: Tuple[float, float, float], object_positions: Dict[str, Tuple[float, float, float]], **kwargs) -> float:
        if not object_positions:
            return 0.0
        nearest_object_distance = distance_to_nearest_point(origin=ee_position, points=list(object_positions.values()))
        self.__task.get_logger().debug(f'[Curriculum] Distance to nearest object: {nearest_object_distance}')
        if nearest_object_distance < self.reach_required_distance:
            self.__task.get_logger().info(f'[Curriculum] An object is now closer than the required distance of {self.reach_required_distance}')
            self.stages_completed_this_episode[GraspStage.REACH] = True
            return self.__stages_base_reward
        else:
            return 0.0

    def get_reward_TOUCH(self, touched_objects: List[str], **kwargs) -> float:
        if touched_objects:
            self.__task.get_logger().info(f'[Curriculum] Touched objects: {touched_objects}')
            self.stages_completed_this_episode[GraspStage.TOUCH] = True
            return self.__stages_base_reward
        else:
            return 0.0

    def get_reward_GRASP(self, grasped_objects: List[str], **kwargs) -> float:
        if grasped_objects:
            self.__task.get_logger().info(f'[Curriculum] Grasped objects: {grasped_objects}')
            self.stages_completed_this_episode[GraspStage.GRASP] = True
            return self.__stages_base_reward
        else:
            return 0.0

    def get_reward_LIFT(self, object_positions: Dict[str, Tuple[float, float, float]], grasped_objects: List[str], **kwargs) -> float:
        if not (grasped_objects or object_positions):
            return 0.0
        for grasped_object in grasped_objects:
            grasped_object_height = object_positions[grasped_object][2]
            self.__task.get_logger().debug(f"[Curriculum] Height of grasped object '{grasped_objects}': {grasped_object_height}")
            if grasped_object_height > self.lift_required_height:
                self.__task.get_logger().info(f'[Curriculum] Lifted object: {grasped_object}')
                self.stages_completed_this_episode[GraspStage.LIFT] = True
                return self.__stages_base_reward
        return 0.0

    def get_persistent_reward(self, object_positions: Dict[str, Tuple[float, float, float]], **kwargs) -> float:
        reward = self.__persistent_reward_each_step
        if self.__persistent_reward_terrain_collision:
            if self.__task.check_terrain_collision():
                self.__task.get_logger().info('[Curriculum] Robot collided with the terrain')
                reward += self.__persistent_reward_terrain_collision
        if self.__persistent_reward_all_objects_outside_workspace:
            if self.__task.check_all_objects_outside_workspace(object_positions=object_positions):
                self.__task.get_logger().warn('[Curriculum] All objects are outside of the workspace')
                reward += self.__persistent_reward_all_objects_outside_workspace
                self.episode_failed = True
        if self.__persistent_reward_arm_stuck:
            if ArmStuckChecker.is_robot_stuck(self):
                self.__task.get_logger().error(f'[Curriculum] Robot appears to be stuck, resetting...')
                reward += self.__persistent_reward_arm_stuck
                self.episode_failed = True
        return reward

