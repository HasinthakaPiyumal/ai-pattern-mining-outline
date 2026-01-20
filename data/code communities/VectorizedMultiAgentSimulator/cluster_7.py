# Cluster 7

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.u_range = kwargs.pop('u_range', 0.5)
        self.a_range = kwargs.pop('a_range', 1)
        self.obs_noise = kwargs.pop('obs_noise', 0)
        self.box_agents = kwargs.pop('box_agents', False)
        self.linear_friction = kwargs.pop('linear_friction', 0.1)
        self.min_input_norm = kwargs.pop('min_input_norm', 0.08)
        self.comms_range = kwargs.pop('comms_range', 5)
        self.shared_rew = kwargs.pop('shared_rew', True)
        self.n_agents = kwargs.pop('n_agents', 4)
        self.pos_shaping_factor = kwargs.pop('pos_shaping_factor', 1)
        self.final_reward = kwargs.pop('final_reward', 0.01)
        self.agent_collision_penalty = kwargs.pop('agent_collision_penalty', -0.1)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.viewer_zoom = 1.7
        controller_params = [2, 6, 0.002]
        self.n_agents = 4
        self.f_range = self.a_range + self.linear_friction
        world = World(batch_dim, device, drag=0, dt=0.1, linear_friction=self.linear_friction, substeps=16 if self.box_agents else 5, collision_force=10000 if self.box_agents else 500)
        self.agent_radius = 0.16
        self.agent_box_length = 0.32
        self.agent_box_width = 0.24
        self.min_collision_distance = 0.005
        self.colors = [Color.GREEN, Color.BLUE, Color.RED, Color.GRAY]
        for i in range(self.n_agents):
            agent = Agent(name=f'agent_{i}', rotatable=False, linear_friction=self.linear_friction, shape=Sphere(radius=self.agent_radius) if not self.box_agents else Box(length=self.agent_box_length, width=self.agent_box_width), u_range=self.u_range, f_range=self.f_range, render_action=True, color=self.colors[i])
            agent.controller = VelocityController(agent, world, controller_params, 'standard')
            goal = Landmark(name=f'goal {i}', collide=False, shape=Sphere(radius=self.agent_radius / 2), color=self.colors[i])
            agent.goal = goal
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)
            world.add_landmark(goal)
        self.spawn_map(world)
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        return world

    def reset_world_at(self, env_index: int=None):
        for i, agent in enumerate(self.world.agents):
            agent.controller.reset(env_index)
            next_i = (i + 1) % self.n_agents
            if i in [0, 2]:
                agent.set_pos(torch.tensor([(self.scenario_length / 2 - self.agent_dist_from_wall) * (-1 if i == 0 else 1), 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                self.world.agents[next_i].goal.set_pos(torch.tensor([(self.scenario_length / 2 - self.goal_dist_from_wall) * (-1 if i == 0 else 1), 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            else:
                agent.set_pos(torch.tensor([0.0, (self.scenario_length / 2 - self.agent_dist_from_wall) * (1 if i == 1 else -1)], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                self.world.agents[next_i].goal.set_pos(torch.tensor([0.0, (self.scenario_length / 2 - self.goal_dist_from_wall) * (1 if i == 1 else -1)], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        for agent in self.world.agents:
            if env_index is None:
                agent.shaping = torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1) * self.pos_shaping_factor
            else:
                agent.shaping[env_index] = torch.linalg.vector_norm(agent.state.pos[env_index] - agent.goal.state.pos[env_index]) * self.pos_shaping_factor
        self.reset_map(env_index)
        if env_index is None:
            self.reached_goal = torch.full((self.world.batch_dim,), False, device=self.world.device)
        else:
            self.reached_goal[env_index] = False

    def process_action(self, agent: Agent):
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, self.u_range)
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < self.min_input_norm] = 0
        agent.vel_action = agent.action.u.clone()
        vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 0.001
        agent.controller.reset(vel_is_zero)
        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0
            for a in self.world.agents:
                a.distance_to_goal = torch.linalg.vector_norm(a.state.pos - a.goal.state.pos, dim=-1)
                a.on_goal = a.distance_to_goal < a.goal.shape.radius
                pos_shaping = a.distance_to_goal * self.pos_shaping_factor
                a.pos_rew = a.shaping - pos_shaping if self.pos_shaping_factor != 0 else -a.distance_to_goal * 0.0001
                a.shaping = pos_shaping
                self.pos_rew += a.pos_rew
            self.all_goal_reached = torch.all(torch.stack([a.on_goal for a in self.world.agents], dim=-1), dim=-1)
            self.final_rew[self.all_goal_reached] = self.final_reward
            self.reached_goal += self.all_goal_reached
        agent.agent_collision_rew[:] = 0
        for a in self.world.agents:
            if a != agent:
                agent.agent_collision_rew[self.world.get_distance(agent, a) <= self.min_collision_distance] += self.agent_collision_penalty
        return (self.pos_rew if self.shared_rew else agent.pos_rew) + agent.agent_collision_rew + self.final_rew

    def observation(self, agent: Agent):
        observations = [agent.state.pos, agent.state.vel, agent.state.pos - agent.goal.state.pos, torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=-1).unsqueeze(-1)]
        if self.obs_noise > 0:
            for i, obs in enumerate(observations):
                noise = torch.zeros(*obs.shape, device=self.world.device).uniform_(-self.obs_noise, self.obs_noise)
                observations[i] = obs + noise
        return torch.cat(observations, dim=-1)

    def info(self, agent: Agent):
        return {'pos_rew': self.pos_rew if self.shared_rew else agent.pos_rew, 'final_rew': self.final_rew, 'agent_collision_rew': agent.agent_collision_rew}

    def extra_render(self, env_index: int=0) -> 'List[Geom]':
        from vmas.simulator import rendering
        geoms: List[Geom] = []
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(agent1.state.pos - agent2.state.pos, dim=-1)
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(agent1.state.pos[env_index], agent2.state.pos[env_index], width=1)
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)
        return geoms

    def spawn_map(self, world: World):
        self.scenario_length = 5
        self.scenario_width = 0.4
        self.long_wall_length = self.scenario_length / 2 - self.scenario_width / 2
        self.short_wall_length = self.scenario_width
        self.goal_dist_from_wall = self.agent_radius + 0.05
        self.agent_dist_from_wall = 0.5
        self.long_walls = []
        for i in range(8):
            landmark = Landmark(name=f'wall {i}', collide=True, shape=Line(length=self.long_wall_length), color=Color.BLACK)
            self.long_walls.append(landmark)
            world.add_landmark(landmark)
        self.short_walls = []
        for i in range(4):
            landmark = Landmark(name=f'short wall {i}', collide=True, shape=Line(length=self.short_wall_length), color=Color.BLACK)
            self.short_walls.append(landmark)
            world.add_landmark(landmark)

    def reset_map(self, env_index):
        for i, landmark in enumerate(self.short_walls):
            if i < 2:
                landmark.set_pos(torch.tensor([-self.scenario_length / 2 if i % 2 == 0 else self.scenario_length / 2, 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            else:
                landmark.set_pos(torch.tensor([0.0, -self.scenario_length / 2 if i % 2 == 0 else self.scenario_length / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        long_wall_pos = self.long_wall_length / 2 - self.scenario_length / 2
        for i, landmark in enumerate(self.long_walls):
            if i < 4:
                landmark.set_pos(torch.tensor([long_wall_pos * (1 if i < 2 else -1), self.scenario_width / 2 * (-1 if i % 2 == 0 else 1)], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            else:
                landmark.set_pos(torch.tensor([self.scenario_width / 2 * (-1 if i % 2 == 0 else 1), long_wall_pos * (1 if i < 6 else -1)], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)

class Scenario(BaseScenario):
    """
    This scenario originally comes from the paper "Xu et al. - 2024 - A Sample Efficient and Generalizable Multi-Agent Reinforcement Learning Framework
    for Motion Planning" (https://arxiv.org/abs/2408.07644, see also its GitHub repo https://github.com/bassamlab/SigmaRL),
    which aims to design an MARL framework with efficient observation design to enable fast training and to empower agents the ability to generalize
    to unseen scenarios.

    Six observation design strategies are proposed in the paper. They correspond to six parameters in this file, and their default
    values are True. Setting them to False will impair the observation efficiency in the evaluation conducted in the paper.
        - is_ego_view: Whether to use ego view (otherwise bird view)
        - is_apply_mask: Whether to mask distant agents
        - is_observe_distance_to_agents: Whether to observe the distance to other agents
        - is_observe_distance_to_boundaries: Whether to observe the distance to labelet boundaries (otherwise the points on lanelet boundaries)
        - is_observe_distance_to_center_line: Whether to observe the distance to reference path (otherwise None)
        - is_observe_vertices: Whether to observe the vertices of other agents (otherwise center points)

    In addition, there are some commonly used parameters you may want to adjust to suit your case:
        - n_agents: Number of agents
        - dt: Sample time in seconds
        - map_type: One of {'1', '2', '3'}:
                         1: the entire map will be used
                         2: the entire map will be used ; besides, challenging initial state buffer will be recorded and used when resetting the envs (inspired
                         by Kaufmann et al. - Nature 2023 - Champion-level drone racing using deep reinforcement learning)
                         3: a specific part of the map (intersection, merge-in, or merge-out) will be used for each env when making or resetting it. You can control the probability of using each of them by the parameter `scenario_probabilities`. It is an array with three values. The first value corresponds to the probability of using intersection. The second and the third values correspond to merge-in and merge-out, respectively. If you only want to use one specific part of the map for all parallel envs, you can set the other two values to zero. For example, if you want to train a RL policy only for intersection, they can set `scenario_probabilities` to [1.0, 0.0, 0.0].
        - is_partial_observation: Whether to enable partial observation (to model partially observable MDP)
        - n_nearing_agents_observed: Number of nearing agents to be observed (consider limited sensor range)

        is_testing_mode: Testing mode is designed to test the learned policy.
                         In non-testing mode, once a collision occurs, all agents will be reset with random initial states.
                         To ensure these initial states are feasible, the initial positions are conservatively large (1.2*diagonalLengthOfAgent).
                         This ensures agents are initially safe and avoids putting agents in an immediate dangerous situation at the beginning of a new scenario.
                         During testing, only colliding agents will be reset, without changing the states of other agents, who are possibly interacting with other agents.
                         This may allow for more effective testing.

    For other parameters, see the class Parameter defined in this file.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(batch_dim, device, **kwargs)
        self.visualize_semidims = False
        world = self.init_world(batch_dim, device)
        self.init_agents(world)
        return world

    def init_params(self, batch_dim, device, **kwargs):
        self.world_x_dim = kwargs.pop('world_x_dim', 4.5)
        self.world_y_dim = kwargs.pop('world_y_dim', 4.0)
        self.agent_width = kwargs.pop('agent_width', 0.08)
        self.agent_length = kwargs.pop('agent_length', 0.16)
        self.l_f = kwargs.pop('l_f', self.agent_length / 2)
        self.l_r = kwargs.pop('l_r', self.agent_length - self.l_f)
        lane_width = kwargs.pop('lane_width', 0.15)
        r_p_normalizer = 100
        reward_progress = kwargs.pop('reward_progress', 10) / r_p_normalizer
        reward_vel = kwargs.pop('reward_vel', 5) / r_p_normalizer
        reward_reach_goal = kwargs.pop('reward_reach_goal', 0) / r_p_normalizer
        threshold_deviate_from_ref_path = kwargs.pop('threshold_deviate_from_ref_path', (lane_width - self.agent_width) / 2)
        threshold_reach_goal = kwargs.pop('threshold_reach_goal', self.agent_width / 2)
        threshold_change_steering = kwargs.pop('threshold_change_steering', 10)
        threshold_near_boundary_high = kwargs.pop('threshold_near_boundary_high', (lane_width - self.agent_width) / 2 * 0.9)
        threshold_near_boundary_low = kwargs.pop('threshold_near_boundary_low', 0)
        threshold_near_other_agents_c2c_high = kwargs.pop('threshold_near_other_agents_c2c_high', self.agent_length + self.agent_width)
        threshold_near_other_agents_c2c_low = kwargs.pop('threshold_near_other_agents_c2c_low', (self.agent_length + self.agent_width) / 2)
        threshold_no_reward_if_too_close_to_boundaries = kwargs.pop('threshold_no_reward_if_too_close_to_boundaries', self.agent_width / 10)
        threshold_no_reward_if_too_close_to_other_agents = kwargs.pop('threshold_no_reward_if_too_close_to_other_agents', self.agent_width / 6)
        self.resolution_factor = kwargs.pop('resolution_factor', 200)
        sample_interval_ref_path = kwargs.pop('sample_interval_ref_path', 2)
        max_ref_path_points = kwargs.pop('max_ref_path_points', 200)
        noise_level = kwargs.pop('noise_level', 0.2 * self.agent_width)
        n_stored_steps = kwargs.pop('n_stored_steps', 5)
        n_observed_steps = kwargs.pop('n_observed_steps', 1)
        self.render_origin = kwargs.pop('render_origin', [self.world_x_dim / 2, self.world_y_dim / 2])
        self.viewer_size = kwargs.pop('viewer_size', (int(self.world_x_dim * self.resolution_factor), int(self.world_y_dim * self.resolution_factor)))
        self.max_steering_angle = kwargs.pop('max_steering_angle', torch.deg2rad(torch.tensor(35, device=device, dtype=torch.float32)))
        self.max_speed = kwargs.pop('max_speed', 1.0)
        self.viewer_zoom = kwargs.pop('viewer_zoom', 1.44)
        parameters = Parameters(n_agents=kwargs.pop('n_agents', 20), is_partial_observation=kwargs.pop('is_partial_observation', True), is_testing_mode=kwargs.pop('is_testing_mode', False), is_visualize_short_term_path=kwargs.pop('is_visualize_short_term_path', True), map_type=kwargs.pop('map_type', '1'), n_nearing_agents_observed=kwargs.pop('n_nearing_agents_observed', 2), is_real_time_rendering=kwargs.pop('is_real_time_rendering', False), n_points_short_term=kwargs.pop('n_points_short_term', 3), dt=kwargs.pop('dt', 0.05), is_ego_view=kwargs.pop('is_ego_view', True), is_apply_mask=kwargs.pop('is_apply_mask', True), is_observe_vertices=kwargs.pop('is_observe_vertices', True), is_observe_distance_to_agents=kwargs.pop('is_observe_distance_to_agents', True), is_observe_distance_to_boundaries=kwargs.pop('is_observe_distance_to_boundaries', True), is_observe_distance_to_center_line=kwargs.pop('is_observe_distance_to_center_line', True), scenario_probabilities=kwargs.pop('scenario_probabilities', [1.0, 0.0, 0.0]), is_add_noise=kwargs.pop('is_add_noise', True), is_observe_ref_path_other_agents=kwargs.pop('is_observe_ref_path_other_agents', False), is_visualize_extra_info=kwargs.pop('is_visualize_extra_info', False), render_title=kwargs.pop('render_title', 'Multi-Agent Reinforcement Learning for Road Traffic (CPM Lab Scenario)'), n_steps_stored=kwargs.pop('n_steps_stored', 10), n_steps_before_recording=kwargs.pop('n_steps_before_recording', 10), n_points_nearing_boundary=kwargs.pop('n_points_nearing_boundary', 5))
        self.parameters = kwargs.pop('parameters', parameters)
        if self.parameters.map_type == '3':
            if self.parameters.scenario_probabilities[1] != 0 or self.parameters.scenario_probabilities[2] != 0:
                if self.parameters.n_agents > 5:
                    raise ValueError("For map_type '3', if the second or third value of scenario_probabilities is not zero, a maximum of 5 agents are allowed, as only a merge-in or a merge-out will be used.")
            elif self.parameters.n_agents > 10:
                raise ValueError("For map_type '3', if only the first value of scenario_probabilities is not zero, a maximum of 10 agents are allowed, as only an intersection will be used.")
        if self.parameters.n_nearing_agents_observed >= self.parameters.n_agents:
            raise ValueError('n_nearing_agents_observed must be less than n_agents')
        self.n_agents = self.parameters.n_agents
        self.timer = Timer(start=time.time(), end=0, step=torch.zeros(batch_dim, device=device, dtype=torch.int32), step_begin=time.time(), render_begin=0)
        map_file_path = kwargs.pop('map_file_path', None)
        if map_file_path is None:
            map_file_path = str(pathlib.Path(__file__).parent.parent / 'scenarios_data' / 'road_traffic' / 'road_traffic_cpm_lab.xml')
        self.map_data = get_map_data(map_file_path, device=device)
        reference_paths_all, reference_paths_intersection, reference_paths_merge_in, reference_paths_merge_out = get_reference_paths(self.map_data)
        if self.parameters.map_type in ('1', '2'):
            max_ref_path_points = max([ref_p['center_line'].shape[0] for ref_p in reference_paths_all]) + self.parameters.n_points_short_term * sample_interval_ref_path + 2
        else:
            max_ref_path_points = max([ref_p['center_line'].shape[0] for ref_p in reference_paths_intersection + reference_paths_merge_in + reference_paths_merge_out]) + self.parameters.n_points_short_term * sample_interval_ref_path + 2
        self.ref_paths_map_related = ReferencePathsMapRelated(long_term_all=reference_paths_all, long_term_intersection=reference_paths_intersection, long_term_merge_in=reference_paths_merge_in, long_term_merge_out=reference_paths_merge_out, point_extended_all=torch.zeros((len(reference_paths_all), self.parameters.n_points_short_term * sample_interval_ref_path, 2), device=device, dtype=torch.float32), point_extended_intersection=torch.zeros((len(reference_paths_intersection), self.parameters.n_points_short_term * sample_interval_ref_path, 2), device=device, dtype=torch.float32), point_extended_merge_in=torch.zeros((len(reference_paths_merge_in), self.parameters.n_points_short_term * sample_interval_ref_path, 2), device=device, dtype=torch.float32), point_extended_merge_out=torch.zeros((len(reference_paths_merge_out), self.parameters.n_points_short_term * sample_interval_ref_path, 2), device=device, dtype=torch.float32), sample_interval=torch.tensor(sample_interval_ref_path, device=device, dtype=torch.int32))
        idx_broadcasting_entend = torch.arange(1, self.parameters.n_points_short_term * sample_interval_ref_path + 1, device=device, dtype=torch.int32).unsqueeze(1)
        for idx, i_path in enumerate(reference_paths_all):
            center_line_i = i_path['center_line']
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_all[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction
        for idx, i_path in enumerate(reference_paths_intersection):
            center_line_i = i_path['center_line']
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_intersection[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction
        for idx, i_path in enumerate(reference_paths_merge_in):
            center_line_i = i_path['center_line']
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_in[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction
        for idx, i_path in enumerate(reference_paths_merge_out):
            center_line_i = i_path['center_line']
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_out[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction
        self.ref_paths_agent_related = ReferencePathsAgentRelated(long_term=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32), long_term_vec_normalized=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32), left_boundary=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32), right_boundary=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32), entry=torch.zeros((batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32), exit=torch.zeros((batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32), is_loop=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.bool), n_points_long_term=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), n_points_left_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), n_points_right_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), short_term=torch.zeros((batch_dim, self.n_agents, self.parameters.n_points_short_term, 2), device=device, dtype=torch.float32), short_term_indices=torch.zeros((batch_dim, self.n_agents, self.parameters.n_points_short_term), device=device, dtype=torch.int32), n_points_nearing_boundary=torch.tensor(self.parameters.n_points_nearing_boundary, device=device, dtype=torch.int32), nearing_points_left_boundary=torch.zeros((batch_dim, self.n_agents, self.parameters.n_points_nearing_boundary, 2), device=device, dtype=torch.float32), nearing_points_right_boundary=torch.zeros((batch_dim, self.n_agents, self.parameters.n_points_nearing_boundary, 2), device=device, dtype=torch.float32), scenario_id=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), path_id=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), point_id=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32))
        self.vertices = torch.zeros((batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32)
        weighting_ref_directions = torch.linspace(1, 0.2, steps=self.parameters.n_points_short_term, device=device, dtype=torch.float32)
        weighting_ref_directions /= weighting_ref_directions.sum()
        self.rewards = Rewards(progress=torch.tensor(reward_progress, device=device, dtype=torch.float32), weighting_ref_directions=weighting_ref_directions, higth_v=torch.tensor(reward_vel, device=device, dtype=torch.float32), reach_goal=torch.tensor(reward_reach_goal, device=device, dtype=torch.float32))
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.penalties = Penalties(deviate_from_ref_path=torch.tensor(-2 / 100, device=device, dtype=torch.float32), weighting_deviate_from_ref_path=self.map_data['mean_lane_width'] / 2, near_boundary=torch.tensor(-20 / 100, device=device, dtype=torch.float32), near_other_agents=torch.tensor(-20 / 100, device=device, dtype=torch.float32), collide_with_agents=torch.tensor(-100 / 100, device=device, dtype=torch.float32), collide_with_boundaries=torch.tensor(-100 / 100, device=device, dtype=torch.float32), change_steering=torch.tensor(-2 / 100, device=device, dtype=torch.float32), time=torch.tensor(5 / 100, device=device, dtype=torch.float32))
        self.observations = Observations(is_partial=torch.tensor(self.parameters.is_partial_observation, device=device, dtype=torch.bool), n_nearing_agents=torch.tensor(self.parameters.n_nearing_agents_observed, device=device, dtype=torch.int32), noise_level=torch.tensor(noise_level, device=device, dtype=torch.float32), n_stored_steps=torch.tensor(n_stored_steps, device=device, dtype=torch.int32), n_observed_steps=torch.tensor(n_observed_steps, device=device, dtype=torch.int32), nearing_agents_indices=torch.zeros((batch_dim, self.n_agents, self.parameters.n_nearing_agents_observed), device=device, dtype=torch.int32))
        if self.parameters.is_ego_view:
            self.observations.past_pos = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_rot = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.float32))
            self.observations.past_vertices = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, 4, 2), device=device, dtype=torch.float32))
            self.observations.past_vel = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_short_term_ref_points = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, self.parameters.n_points_short_term, 2), device=device, dtype=torch.float32))
            self.observations.past_left_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, self.parameters.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
            self.observations.past_right_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, self.parameters.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
        else:
            self.observations.past_pos = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_rot = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
            self.observations.past_vertices = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, 4, 2), device=device, dtype=torch.float32))
            self.observations.past_vel = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_short_term_ref_points = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.parameters.n_points_short_term, 2), device=device, dtype=torch.float32))
            self.observations.past_left_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.parameters.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
            self.observations.past_right_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.parameters.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
        self.observations.past_action_vel = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_action_steering = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_ref_path = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_boundaries = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_left_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_right_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_agents = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.float32))
        self.normalizers = Normalizers(pos=torch.tensor([self.agent_length * 10, self.agent_length * 10], device=device, dtype=torch.float32), pos_world=torch.tensor([self.world_x_dim, self.world_y_dim], device=device, dtype=torch.float32), v=torch.tensor(self.max_speed, device=device, dtype=torch.float32), rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32), action_steering=self.max_steering_angle, action_vel=torch.tensor(self.max_speed, device=device, dtype=torch.float32), distance_lanelet=torch.tensor(lane_width * 3, device=device, dtype=torch.float32), distance_ref=torch.tensor(lane_width * 3, device=device, dtype=torch.float32), distance_agent=torch.tensor(self.agent_length * 10, device=device, dtype=torch.float32))
        self.distances = Distances(agents=torch.zeros(batch_dim, self.n_agents, self.n_agents, device=device, dtype=torch.float32), left_boundaries=torch.zeros((batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32), right_boundaries=torch.zeros((batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32), boundaries=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32), ref_paths=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32), closest_point_on_ref_path=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), closest_point_on_left_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), closest_point_on_right_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32))
        self.thresholds = Thresholds(reach_goal=torch.tensor(threshold_reach_goal, device=device, dtype=torch.float32), deviate_from_ref_path=torch.tensor(threshold_deviate_from_ref_path, device=device, dtype=torch.float32), near_boundary_low=torch.tensor(threshold_near_boundary_low, device=device, dtype=torch.float32), near_boundary_high=torch.tensor(threshold_near_boundary_high, device=device, dtype=torch.float32), near_other_agents_low=torch.tensor(threshold_near_other_agents_c2c_low, device=device, dtype=torch.float32), near_other_agents_high=torch.tensor(threshold_near_other_agents_c2c_high, device=device, dtype=torch.float32), change_steering=torch.tensor(threshold_change_steering, device=device, dtype=torch.float32).deg2rad(), no_reward_if_too_close_to_boundaries=torch.tensor(threshold_no_reward_if_too_close_to_boundaries, device=device, dtype=torch.float32), no_reward_if_too_close_to_other_agents=torch.tensor(threshold_no_reward_if_too_close_to_other_agents, device=device, dtype=torch.float32), distance_mask_agents=self.normalizers.pos[0])
        self.constants = Constants(env_idx_broadcasting=torch.arange(batch_dim, device=device, dtype=torch.int32).unsqueeze(-1), empty_action_vel=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32), empty_action_steering=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32), mask_pos=torch.tensor(1, device=device, dtype=torch.float32), mask_zero=torch.tensor(0, device=device, dtype=torch.float32), mask_one=torch.tensor(1, device=device, dtype=torch.float32), reset_agent_min_distance=torch.tensor((self.l_f + self.l_r) ** 2 + self.agent_width ** 2, device=device, dtype=torch.float32).sqrt() * 1.2)
        self.collisions = Collisions(with_agents=torch.zeros((batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.bool), with_lanelets=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.bool), with_entry_segments=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.bool), with_exit_segments=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.bool))
        self.initial_state_buffer = InitialStateBuffer(probability_record=torch.tensor(1.0, device=device, dtype=torch.float32), probability_use_recording=torch.tensor(kwargs.pop('probability_use_recording', 0.2), device=device, dtype=torch.float32), buffer=torch.zeros((100, self.n_agents, 8), device=device, dtype=torch.float32))
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.state_buffer = StateBuffer(buffer=torch.zeros((self.parameters.n_steps_before_recording, batch_dim, self.n_agents, 8), device=device, dtype=torch.float32))

    def init_world(self, batch_dim: int, device: torch.device):
        world = World(batch_dim, device, x_semidim=self.world_x_dim, y_semidim=self.world_y_dim, dt=self.parameters.dt)
        return world

    def init_agents(self, world, *kwargs):
        for i in range(self.n_agents):
            agent = Agent(name=f'agent_{i}', shape=Box(length=self.l_f + self.l_r, width=self.agent_width), color=tuple(torch.rand(3, device=world.device, dtype=torch.float32).tolist()), collide=False, render_action=False, u_range=[self.max_speed, self.max_steering_angle], u_multiplier=[1, 1], max_speed=self.max_speed, dynamics=KinematicBicycle(world, width=self.agent_width, l_f=self.l_f, l_r=self.l_r, max_steering_angle=self.max_steering_angle, integration='rk4'))
            world.add_agent(agent)

    def reset_world_at(self, env_index: int=None, agent_index: int=None):
        """
        This function resets the world at the specified env_index and the specified agent_index.
        If env_index is given as None, the majority part of computation will be done in a vectorized manner.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed
        :param agent_index: index of the agent to reset. If None all agents in the specified environment will be reset.
        """
        agents = self.world.agents
        is_reset_single_agent = agent_index is not None
        for env_i in [env_index] if env_index is not None else range(self.world.batch_dim):
            if env_i == 0:
                self.timer.start = time.time()
                self.timer.step_begin = time.time()
                self.timer.end = 0
            if not is_reset_single_agent:
                self.timer.step[env_i] = 0
            ref_paths_scenario, extended_points = self.reset_scenario_related_ref_paths(env_i, is_reset_single_agent, agent_index)
            if self.parameters.map_type == '2' and torch.rand(1) < self.initial_state_buffer.probability_use_recording and (self.initial_state_buffer.valid_size >= 1):
                is_use_state_buffer = True
                initial_state = self.initial_state_buffer.get_random()
                self.ref_paths_agent_related.scenario_id[env_i] = initial_state[:, self.initial_state_buffer.idx_scenario]
                self.ref_paths_agent_related.path_id[env_i] = initial_state[:, self.initial_state_buffer.idx_path]
                self.ref_paths_agent_related.point_id[env_i] = initial_state[:, self.initial_state_buffer.idx_point]
            else:
                is_use_state_buffer = False
                initial_state = None
            for i_agent in range(self.n_agents) if not is_reset_single_agent else agent_index.unsqueeze(0):
                ref_path, path_id = self.reset_init_state(env_i, i_agent, is_reset_single_agent, is_use_state_buffer, initial_state, ref_paths_scenario, agents)
                self.reset_agent_related_ref_path(env_i, i_agent, ref_path, path_id, extended_points)
            if env_index is None:
                if env_i == self.world.batch_dim - 1:
                    env_j = slice(None)
                else:
                    continue
            else:
                env_j = env_i
            for i_agent in range(self.n_agents) if not is_reset_single_agent else agent_index.unsqueeze(0):
                self.reset_init_distances_and_short_term_ref_path(env_j, i_agent, agents)
            mutual_distances = get_distances_between_agents(self=self, is_set_diagonal=True)
            self.distances.agents[env_j, :, :] = mutual_distances[env_j, :, :]
            self.collisions.with_agents[env_j, :, :] = False
            self.collisions.with_lanelets[env_j, :] = False
            self.collisions.with_entry_segments[env_j, :] = False
            self.collisions.with_exit_segments[env_j, :] = False
        self.state_buffer.reset()
        state_add = torch.cat((torch.stack([a.state.pos for a in agents], dim=1), torch.stack([a.state.rot for a in agents], dim=1), torch.stack([a.state.vel for a in agents], dim=1), self.ref_paths_agent_related.scenario_id[:].unsqueeze(-1), self.ref_paths_agent_related.path_id[:].unsqueeze(-1), self.ref_paths_agent_related.point_id[:].unsqueeze(-1)), dim=-1)
        self.state_buffer.add(state_add)

    def reset_scenario_related_ref_paths(self, env_i, is_reset_single_agent, agent_index):
        """
        Resets scenario-related reference paths and scenario IDs for the specified environment and agents.

        This function determines and sets the long-term reference paths based on the current map_type.
        If `is_reset_single_agent` is true, the current paths for the specified agent will be kept.

        Args:
            env_i (int): The index of the environment to reset.
            is_reset_single_agent (bool): Flag indicating whether only a single agent is being reset.
            agent_index (int or None): The index of the agent to reset. If None, all agents in
                                    the specified environment are reset.

        Returns:
            - ref_paths_scenario (list): The list of reference paths for the current scenario.
            - extended_points (tensor): [numOfRefPaths, numExtendedPoints, 2] The extended points for the current scenario.
        """
        if self.parameters.map_type in {'1', '2'}:
            ref_paths_scenario = self.ref_paths_map_related.long_term_all
            extended_points = self.ref_paths_map_related.point_extended_all
            self.ref_paths_agent_related.scenario_id[env_i, :] = 0
        else:
            if is_reset_single_agent:
                scenario_id = self.ref_paths_agent_related.scenario_id[env_i, agent_index]
            else:
                scenario_id = torch.multinomial(torch.tensor(self.parameters.scenario_probabilities, device=self.world.device, dtype=torch.float32), 1, replacement=True).item() + 1
                self.ref_paths_agent_related.scenario_id[env_i, :] = scenario_id
            if scenario_id == 1:
                ref_paths_scenario = self.ref_paths_map_related.long_term_intersection
                extended_points = self.ref_paths_map_related.point_extended_intersection
            elif scenario_id == 2:
                ref_paths_scenario = self.ref_paths_map_related.long_term_merge_in
                extended_points = self.ref_paths_map_related.point_extended_merge_in
            elif scenario_id == 3:
                ref_paths_scenario = self.ref_paths_map_related.long_term_merge_out
                extended_points = self.ref_paths_map_related.point_extended_merge_out
        return (ref_paths_scenario, extended_points)

    def reset_init_state(self, env_i, i_agent, is_reset_single_agent, is_use_state_buffer, initial_state, ref_paths_scenario, agents):
        """
        This function resets the initial position, rotation, and velocity for an agent based on the provided
        initial state buffer if it is used. Otherwise, it randomly generates initial states ensuring they
        are feasible and do not collide with other agents.
        """
        if is_use_state_buffer:
            path_id = initial_state[i_agent, self.initial_state_buffer.idx_path].int()
            ref_path = ref_paths_scenario[path_id]
            agents[i_agent].set_pos(initial_state[i_agent, 0:2], batch_index=env_i)
            agents[i_agent].set_rot(initial_state[i_agent, 2], batch_index=env_i)
            agents[i_agent].set_vel(initial_state[i_agent, 3:5], batch_index=env_i)
        else:
            is_feasible_initial_position_found = False
            while not is_feasible_initial_position_found:
                path_id = torch.randint(0, len(ref_paths_scenario), (1,)).item()
                self.ref_paths_agent_related.path_id[env_i, i_agent] = path_id
                ref_path = ref_paths_scenario[path_id]
                num_points = ref_path['center_line'].shape[0]
                if (self.parameters.scenario_probabilities[1] == 0) & (self.parameters.scenario_probabilities[2] == 0):
                    random_point_id = torch.randint(6, int(num_points / 2), (1,)).item()
                else:
                    random_point_id = torch.randint(3, num_points - 5, (1,)).item()
                self.ref_paths_agent_related.point_id[env_i, i_agent] = random_point_id
                position_start = ref_path['center_line'][random_point_id]
                agents[i_agent].set_pos(position_start, batch_index=env_i)
                if not is_reset_single_agent:
                    if i_agent == 0:
                        is_feasible_initial_position_found = True
                        continue
                    else:
                        positions = torch.stack([self.world.agents[i].state.pos[env_i] for i in range(i_agent + 1)])
                else:
                    positions = torch.stack([self.world.agents[i].state.pos[env_i] for i in range(self.n_agents)])
                diff_sq = (positions[i_agent, :] - positions) ** 2
                initial_mutual_distances_sq = torch.sum(diff_sq, dim=-1)
                initial_mutual_distances_sq[i_agent] = torch.max(initial_mutual_distances_sq) + 1
                min_distance_sq = torch.min(initial_mutual_distances_sq)
                is_feasible_initial_position_found = min_distance_sq >= self.constants.reset_agent_min_distance ** 2
            rot_start = ref_path['center_line_yaw'][random_point_id]
            vel_start_abs = torch.rand(1, dtype=torch.float32, device=self.world.device) * agents[i_agent].max_speed
            vel_start = torch.hstack([vel_start_abs * torch.cos(rot_start), vel_start_abs * torch.sin(rot_start)])
            agents[i_agent].set_rot(rot_start, batch_index=env_i)
            agents[i_agent].set_vel(vel_start, batch_index=env_i)
            return (ref_path, path_id)

    def reset_agent_related_ref_path(self, env_i, i_agent, ref_path, path_id, extended_points):
        """
        This function resets the agent-related reference paths and updates various related attributes
        for a specified agent in an environment.
        """
        n_points_long_term = ref_path['center_line'].shape[0]
        self.ref_paths_agent_related.long_term[env_i, i_agent, 0:n_points_long_term, :] = ref_path['center_line']
        self.ref_paths_agent_related.long_term[env_i, i_agent, n_points_long_term:n_points_long_term + self.parameters.n_points_short_term * self.ref_paths_map_related.sample_interval, :] = extended_points[path_id, :, :]
        self.ref_paths_agent_related.long_term[env_i, i_agent, n_points_long_term + self.parameters.n_points_short_term * self.ref_paths_map_related.sample_interval:, :] = extended_points[path_id, -1, :]
        self.ref_paths_agent_related.n_points_long_term[env_i, i_agent] = n_points_long_term
        self.ref_paths_agent_related.long_term_vec_normalized[env_i, i_agent, 0:n_points_long_term - 1, :] = ref_path['center_line_vec_normalized']
        self.ref_paths_agent_related.long_term_vec_normalized[env_i, i_agent, n_points_long_term - 1:n_points_long_term - 1 + self.parameters.n_points_short_term * self.ref_paths_map_related.sample_interval, :] = ref_path['center_line_vec_normalized'][-1, :]
        n_points_left_b = ref_path['left_boundary_shared'].shape[0]
        self.ref_paths_agent_related.left_boundary[env_i, i_agent, 0:n_points_left_b, :] = ref_path['left_boundary_shared']
        self.ref_paths_agent_related.left_boundary[env_i, i_agent, n_points_left_b:, :] = ref_path['left_boundary_shared'][-1, :]
        self.ref_paths_agent_related.n_points_left_b[env_i, i_agent] = n_points_left_b
        n_points_right_b = ref_path['right_boundary_shared'].shape[0]
        self.ref_paths_agent_related.right_boundary[env_i, i_agent, 0:n_points_right_b, :] = ref_path['right_boundary_shared']
        self.ref_paths_agent_related.right_boundary[env_i, i_agent, n_points_right_b:, :] = ref_path['right_boundary_shared'][-1, :]
        self.ref_paths_agent_related.n_points_right_b[env_i, i_agent] = n_points_right_b
        self.ref_paths_agent_related.entry[env_i, i_agent, 0, :] = ref_path['left_boundary_shared'][0, :]
        self.ref_paths_agent_related.entry[env_i, i_agent, 1, :] = ref_path['right_boundary_shared'][0, :]
        self.ref_paths_agent_related.exit[env_i, i_agent, 0, :] = ref_path['left_boundary_shared'][-1, :]
        self.ref_paths_agent_related.exit[env_i, i_agent, 1, :] = ref_path['right_boundary_shared'][-1, :]
        self.ref_paths_agent_related.is_loop[env_i, i_agent] = ref_path['is_loop']

    def reset_init_distances_and_short_term_ref_path(self, env_j, i_agent, agents):
        """
        This function calculates the distances from the agent's center of gravity (CG) to its reference path and boundaries,
        and computes the positions of the four vertices of the agent. It also determines the short-term reference paths
        for the agent based on the long-term reference paths and the agent's current position.
        """
        self.distances.ref_paths[env_j, i_agent], self.distances.closest_point_on_ref_path[env_j, i_agent] = get_perpendicular_distances(point=agents[i_agent].state.pos[env_j, :], polyline=self.ref_paths_agent_related.long_term[env_j, i_agent], n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent])
        center_2_left_b, self.distances.closest_point_on_left_b[env_j, i_agent] = get_perpendicular_distances(point=agents[i_agent].state.pos[env_j, :], polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent], n_points_long_term=self.ref_paths_agent_related.n_points_left_b[env_j, i_agent])
        self.distances.left_boundaries[env_j, i_agent, 0] = center_2_left_b - agents[i_agent].shape.width / 2
        center_2_right_b, self.distances.closest_point_on_right_b[env_j, i_agent] = get_perpendicular_distances(point=agents[i_agent].state.pos[env_j, :], polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent], n_points_long_term=self.ref_paths_agent_related.n_points_right_b[env_j, i_agent])
        self.distances.right_boundaries[env_j, i_agent, 0] = center_2_right_b - agents[i_agent].shape.width / 2
        self.vertices[env_j, i_agent] = get_rectangle_vertices(center=agents[i_agent].state.pos[env_j, :], yaw=agents[i_agent].state.rot[env_j, :], width=agents[i_agent].shape.width, length=agents[i_agent].shape.length, is_close_shape=True)
        for c_i in range(4):
            self.distances.left_boundaries[env_j, i_agent, c_i + 1], _ = get_perpendicular_distances(point=self.vertices[env_j, i_agent, c_i, :], polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent], n_points_long_term=self.ref_paths_agent_related.n_points_left_b[env_j, i_agent])
            self.distances.right_boundaries[env_j, i_agent, c_i + 1], _ = get_perpendicular_distances(point=self.vertices[env_j, i_agent, c_i, :], polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent], n_points_long_term=self.ref_paths_agent_related.n_points_right_b[env_j, i_agent])
        self.distances.boundaries[env_j, i_agent], _ = torch.min(torch.hstack((self.distances.left_boundaries[env_j, i_agent], self.distances.right_boundaries[env_j, i_agent])), dim=-1)
        self.ref_paths_agent_related.short_term[env_j, i_agent], _ = get_short_term_reference_path(polyline=self.ref_paths_agent_related.long_term[env_j, i_agent], index_closest_point=self.distances.closest_point_on_ref_path[env_j, i_agent], n_points_to_return=self.parameters.n_points_short_term, device=self.world.device, is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent], n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent], sample_interval=self.ref_paths_map_related.sample_interval, n_points_shift=1)
        if not self.parameters.is_observe_distance_to_boundaries:
            self.ref_paths_agent_related.nearing_points_left_boundary[env_j, i_agent], _ = get_short_term_reference_path(polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent], index_closest_point=self.distances.closest_point_on_left_b[env_j, i_agent], n_points_to_return=self.parameters.n_points_nearing_boundary, device=self.world.device, is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent], n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent], sample_interval=1, n_points_shift=1)
            self.ref_paths_agent_related.nearing_points_right_boundary[env_j, i_agent], _ = get_short_term_reference_path(polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent], index_closest_point=self.distances.closest_point_on_right_b[env_j, i_agent], n_points_to_return=self.parameters.n_points_nearing_boundary, device=self.world.device, is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent], n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent], sample_interval=1, n_points_shift=1)

    def reward(self, agent: Agent):
        """
        Issue rewards for the given agent in all envs.
            Positive Rewards:
                Moving forward (become negative if the projection of the moving direction to its reference path is negative)
                Moving forward with high speed (become negative if the projection of the moving direction to its reference path is negative)
                Reaching goal (optional)

            Negative Rewards (penalties):
                Too close to lane boundaries
                Too close to other agents
                Deviating from reference paths
                Changing steering too quick
                Colliding with other agents
                Colliding with lane boundaries

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            A tensor with shape [batch_dim].
        """
        self.rew[:] = 0
        agent_index = self.world.agents.index(agent)
        self.update_state_before_rewarding(agent, agent_index)
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(1)
        ref_points_vecs = self.ref_paths_agent_related.short_term[:, agent_index] - latest_state[:, agent_index, 0:2].unsqueeze(1)
        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(move_projected, self.rewards.weighting_ref_directions)
        reward_movement = move_projected_weighted / (agent.max_speed * self.world.dt) * self.rewards.progress
        self.rew += reward_movement
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(-1)
        factor_moving_direction = torch.where(v_proj > 0, 1, 2)
        reward_vel = factor_moving_direction * v_proj / agent.max_speed * self.rewards.higth_v
        self.rew += reward_vel
        reward_goal = self.collisions.with_exit_segments[:, agent_index] * self.rewards.reach_goal
        self.rew += reward_goal
        penalty_close_to_lanelets = exponential_decreasing_fcn(x=self.distances.boundaries[:, agent_index], x0=self.thresholds.near_boundary_low, x1=self.thresholds.near_boundary_high) * self.penalties.near_boundary
        self.rew += penalty_close_to_lanelets
        mutual_distance_exp_fcn = exponential_decreasing_fcn(x=self.distances.agents[:, agent_index, :], x0=self.thresholds.near_other_agents_low, x1=self.thresholds.near_other_agents_high)
        penalty_close_to_agents = torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        self.rew += penalty_close_to_agents
        self.rew += self.distances.ref_paths[:, agent_index] / self.penalties.weighting_deviate_from_ref_path * self.penalties.deviate_from_ref_path
        steering_current = self.observations.past_action_steering.get_latest(n=1)[:, agent_index]
        steering_past = self.observations.past_action_steering.get_latest(n=2)[:, agent_index]
        steering_change = torch.clamp((steering_current - steering_past).abs() * self.normalizers.action_steering - self.thresholds.change_steering, min=0)
        steering_change_reward_factor = steering_change / (2 * agent.u_range[1] - 2 * self.thresholds.change_steering)
        penalty_change_steering = steering_change_reward_factor * self.penalties.change_steering
        self.rew += penalty_change_steering
        is_collide_with_agents = self.collisions.with_agents[:, agent_index]
        penalty_collide_other_agents = is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
        self.rew += penalty_collide_other_agents
        is_collide_with_lanelets = self.collisions.with_lanelets[:, agent_index]
        penalty_collide_lanelet = is_collide_with_lanelets * self.penalties.collide_with_boundaries
        self.rew += penalty_collide_lanelet
        time_reward = torch.where(v_proj > 0, 1, -1) * agent.state.vel.norm(dim=-1) / agent.max_speed * self.penalties.time
        self.rew += time_reward
        self.update_state_after_rewarding(agent_index)
        return self.rew

    def update_state_before_rewarding(self, agent, agent_index):
        """Update some states (such as mutual distances between agents, vertices of each agent, and
        collision matrices) that will be used before rewarding agents.
        """
        if agent_index == 0:
            self.timer.step_begin = time.time()
            self.timer.step += 1
            self.distances.agents = get_distances_between_agents(self=self, is_set_diagonal=True)
            self.collisions.with_agents[:] = False
            self.collisions.with_lanelets[:] = False
            self.collisions.with_entry_segments[:] = False
            self.collisions.with_exit_segments[:] = False
            for a_i in range(self.n_agents):
                self.vertices[:, a_i] = get_rectangle_vertices(center=self.world.agents[a_i].state.pos, yaw=self.world.agents[a_i].state.rot, width=self.world.agents[a_i].shape.width, length=self.world.agents[a_i].shape.length, is_close_shape=True)
                for a_j in range(a_i + 1, self.n_agents):
                    collision_batch_index = interX(self.vertices[:, a_i], self.vertices[:, a_j], False)
                    self.collisions.with_agents[torch.nonzero(collision_batch_index), a_i, a_j] = True
                    self.collisions.with_agents[torch.nonzero(collision_batch_index), a_j, a_i] = True
                collision_with_left_boundary = interX(L1=self.vertices[:, a_i], L2=self.ref_paths_agent_related.left_boundary[:, a_i], is_return_points=False)
                collision_with_right_boundary = interX(L1=self.vertices[:, a_i], L2=self.ref_paths_agent_related.right_boundary[:, a_i], is_return_points=False)
                self.collisions.with_lanelets[collision_with_left_boundary | collision_with_right_boundary, a_i] = True
                if not self.ref_paths_agent_related.is_loop[:, a_i].any():
                    self.collisions.with_entry_segments[:, a_i] = interX(L1=self.vertices[:, a_i], L2=self.ref_paths_agent_related.entry[:, a_i], is_return_points=False)
                    self.collisions.with_exit_segments[:, a_i] = interX(L1=self.vertices[:, a_i], L2=self.ref_paths_agent_related.exit[:, a_i], is_return_points=False)
        self.distances.ref_paths[:, agent_index], self.distances.closest_point_on_ref_path[:, agent_index] = get_perpendicular_distances(point=agent.state.pos, polyline=self.ref_paths_agent_related.long_term[:, agent_index], n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index])
        center_2_left_b, self.distances.closest_point_on_left_b[:, agent_index] = get_perpendicular_distances(point=agent.state.pos[:, :], polyline=self.ref_paths_agent_related.left_boundary[:, agent_index], n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, agent_index])
        self.distances.left_boundaries[:, agent_index, 0] = center_2_left_b - agent.shape.width / 2
        center_2_right_b, self.distances.closest_point_on_right_b[:, agent_index] = get_perpendicular_distances(point=agent.state.pos[:, :], polyline=self.ref_paths_agent_related.right_boundary[:, agent_index], n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, agent_index])
        self.distances.right_boundaries[:, agent_index, 0] = center_2_right_b - agent.shape.width / 2
        for c_i in range(4):
            self.distances.left_boundaries[:, agent_index, c_i + 1], _ = get_perpendicular_distances(point=self.vertices[:, agent_index, c_i, :], polyline=self.ref_paths_agent_related.left_boundary[:, agent_index], n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, agent_index])
            self.distances.right_boundaries[:, agent_index, c_i + 1], _ = get_perpendicular_distances(point=self.vertices[:, agent_index, c_i, :], polyline=self.ref_paths_agent_related.right_boundary[:, agent_index], n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, agent_index])
        self.distances.boundaries[:, agent_index], _ = torch.min(torch.hstack((self.distances.left_boundaries[:, agent_index], self.distances.right_boundaries[:, agent_index])), dim=-1)

    def update_state_after_rewarding(self, agent_index):
        """Update some states (such as previous positions and short-term reference paths) after rewarding agents."""
        if agent_index == self.n_agents - 1:
            state_add = torch.cat((torch.stack([a.state.pos for a in self.world.agents], dim=1), torch.stack([a.state.rot for a in self.world.agents], dim=1), torch.stack([a.state.vel for a in self.world.agents], dim=1), self.ref_paths_agent_related.scenario_id[:].unsqueeze(-1), self.ref_paths_agent_related.path_id[:].unsqueeze(-1), self.ref_paths_agent_related.point_id[:].unsqueeze(-1)), dim=-1)
            self.state_buffer.add(state_add)
        self.ref_paths_agent_related.short_term[:, agent_index], _ = get_short_term_reference_path(polyline=self.ref_paths_agent_related.long_term[:, agent_index], index_closest_point=self.distances.closest_point_on_ref_path[:, agent_index], n_points_to_return=self.parameters.n_points_short_term, device=self.world.device, is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index], n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index], sample_interval=self.ref_paths_map_related.sample_interval)
        if not self.parameters.is_observe_distance_to_boundaries:
            self.ref_paths_agent_related.nearing_points_left_boundary[:, agent_index], _ = get_short_term_reference_path(polyline=self.ref_paths_agent_related.left_boundary[:, agent_index], index_closest_point=self.distances.closest_point_on_left_b[:, agent_index], n_points_to_return=self.parameters.n_points_nearing_boundary, device=self.world.device, is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index], n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index], sample_interval=1, n_points_shift=-2)
            self.ref_paths_agent_related.nearing_points_right_boundary[:, agent_index], _ = get_short_term_reference_path(polyline=self.ref_paths_agent_related.right_boundary[:, agent_index], index_closest_point=self.distances.closest_point_on_right_b[:, agent_index], n_points_to_return=self.parameters.n_points_nearing_boundary, device=self.world.device, is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index], n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index], sample_interval=1, n_points_shift=-2)

    def observation(self, agent: Agent):
        """
        Generate an observation for the given agent in all envs.

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            The observation for the given agent in all envs, which consists of the observation of this agent itself and possibly the observation of its surrounding agents.
                The observation of this agent itself includes
                    position (in case of using bird view),
                    rotation (in case of using bird view),
                    velocity,
                    short-term reference path,
                    distance to its reference path (optional), and
                    lane boundaries (or distances to them).
                The observation of its surrounding agents includes their
                    vertices (or positions and rotations),
                    velocities,
                    distances to them (optional), and
                    reference paths (optional).
        """
        agent_index = self.world.agents.index(agent)
        self.update_observation_and_normalize(agent, agent_index)
        obs_other_agents = self.observe_other_agents(agent_index)
        obs_self = self.observe_self(agent_index)
        obs_self.append(obs_other_agents)
        obs_all = [o for o in obs_self if o is not None]
        obs = torch.hstack(obs_all)
        if self.parameters.is_add_noise:
            return obs + self.observations.noise_level * torch.rand_like(obs, device=self.world.device, dtype=torch.float32)
        else:
            return obs

    def update_observation_and_normalize(self, agent, agent_index):
        """Update observation and normalize them."""
        if agent_index == 0:
            positions_global = torch.stack([a.state.pos for a in self.world.agents], dim=0).transpose(0, 1)
            rotations_global = torch.stack([a.state.rot for a in self.world.agents], dim=0).transpose(0, 1).squeeze(-1)
            self.observations.past_distance_to_agents.add(self.distances.agents / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_ref_path.add(self.distances.ref_paths / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_left_boundary.add(torch.min(self.distances.left_boundaries, dim=-1)[0] / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_right_boundary.add(torch.min(self.distances.right_boundaries, dim=-1)[0] / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_boundaries.add(self.distances.boundaries / self.normalizers.distance_lanelet)
            if self.parameters.is_ego_view:
                pos_i_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents, 2), device=self.world.device, dtype=torch.float32)
                rot_i_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents), device=self.world.device, dtype=torch.float32)
                vel_i_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents, 2), device=self.world.device, dtype=torch.float32)
                ref_i_others = torch.zeros_like(self.observations.past_short_term_ref_points.get_latest())
                l_b_i_others = torch.zeros_like(self.observations.past_left_boundary.get_latest())
                r_b_i_others = torch.zeros_like(self.observations.past_right_boundary.get_latest())
                ver_i_others = torch.zeros_like(self.observations.past_vertices.get_latest())
                for a_i in range(self.n_agents):
                    pos_i = self.world.agents[a_i].state.pos
                    rot_i = self.world.agents[a_i].state.rot
                    pos_i_others[:, a_i] = transform_from_global_to_local_coordinate(pos_i=pos_i, pos_j=positions_global, rot_i=rot_i)
                    rot_i_others[:, a_i] = rotations_global - rot_i
                    for a_j in range(self.n_agents):
                        rot_rel = rot_i_others[:, a_i, a_j].unsqueeze(1)
                        vel_abs = torch.norm(self.world.agents[a_j].state.vel, dim=1).unsqueeze(1)
                        vel_i_others[:, a_i, a_j] = torch.hstack((vel_abs * torch.cos(rot_rel), vel_abs * torch.sin(rot_rel)))
                        ref_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(pos_i=pos_i, pos_j=self.ref_paths_agent_related.short_term[:, a_j], rot_i=rot_i)
                        if not self.parameters.is_observe_distance_to_boundaries:
                            l_b_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(pos_i=pos_i, pos_j=self.ref_paths_agent_related.nearing_points_left_boundary[:, a_j], rot_i=rot_i)
                            r_b_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(pos_i=pos_i, pos_j=self.ref_paths_agent_related.nearing_points_right_boundary[:, a_j], rot_i=rot_i)
                        ver_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(pos_i=pos_i, pos_j=self.vertices[:, a_j, 0:4, :], rot_i=rot_i)
                self.observations.past_pos.add(pos_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_rot.add(rot_i_others / self.normalizers.rot)
                self.observations.past_vel.add(vel_i_others / self.normalizers.v)
                self.observations.past_short_term_ref_points.add(ref_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_left_boundary.add(l_b_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_right_boundary.add(r_b_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_vertices.add(ver_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
            else:
                self.observations.past_pos.add(positions_global / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_vel.add(torch.stack([a.state.vel for a in self.world.agents], dim=1) / self.normalizers.v)
                self.observations.past_rot.add(rotations_global[:] / self.normalizers.rot)
                self.observations.past_vertices.add(self.vertices[:, :, 0:4, :] / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_short_term_ref_points.add(self.ref_paths_agent_related.short_term[:] / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_left_boundary.add(self.ref_paths_agent_related.nearing_points_left_boundary / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_right_boundary.add(self.ref_paths_agent_related.nearing_points_right_boundary / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
            if agent.action.u is None:
                self.observations.past_action_vel.add(self.constants.empty_action_vel)
                self.observations.past_action_steering.add(self.constants.empty_action_steering)
            else:
                self.observations.past_action_vel.add(torch.stack([a.action.u[:, 0] for a in self.world.agents], dim=1) / self.normalizers.action_vel)
                self.observations.past_action_steering.add(torch.stack([a.action.u[:, 1] for a in self.world.agents], dim=1) / self.normalizers.action_steering)

    def observe_other_agents(self, agent_index):
        """Observe surrounding agents."""
        if self.observations.is_partial:
            nearing_agents_distances, nearing_agents_indices = torch.topk(self.distances.agents[:, agent_index], k=self.observations.n_nearing_agents, largest=False)
            if self.parameters.is_apply_mask:
                mask_nearing_agents_too_far = nearing_agents_distances >= self.thresholds.distance_mask_agents
            else:
                mask_nearing_agents_too_far = torch.zeros((self.world.batch_dim, self.parameters.n_nearing_agents_observed), device=self.world.device, dtype=torch.bool)
            indexing_tuple_1 = (self.constants.env_idx_broadcasting,) + ((agent_index,) if self.parameters.is_ego_view else ()) + (nearing_agents_indices,)
            obs_pos_other_agents = self.observations.past_pos.get_latest()[indexing_tuple_1]
            obs_pos_other_agents[mask_nearing_agents_too_far] = self.constants.mask_one
            obs_rot_other_agents = self.observations.past_rot.get_latest()[indexing_tuple_1]
            obs_rot_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero
            obs_vel_other_agents = self.observations.past_vel.get_latest()[indexing_tuple_1]
            obs_vel_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero
            obs_ref_path_other_agents = self.observations.past_short_term_ref_points.get_latest()[indexing_tuple_1]
            obs_ref_path_other_agents[mask_nearing_agents_too_far] = self.constants.mask_one
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[indexing_tuple_1]
            obs_vertices_other_agents[mask_nearing_agents_too_far] = self.constants.mask_one
            obs_distance_other_agents = self.observations.past_distance_to_agents.get_latest()[self.constants.env_idx_broadcasting, agent_index, nearing_agents_indices]
            obs_distance_other_agents[mask_nearing_agents_too_far] = self.constants.mask_one
        else:
            obs_pos_other_agents = self.observations.past_pos.get_latest()[:, agent_index]
            obs_rot_other_agents = self.observations.past_rot.get_latest()[:, agent_index]
            obs_vel_other_agents = self.observations.past_vel.get_latest()[:, agent_index]
            obs_ref_path_other_agents = self.observations.past_short_term_ref_points.get_latest()[:, agent_index]
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[:, agent_index]
            obs_distance_other_agents = self.observations.past_distance_to_agents.get_latest()[:, agent_index]
            obs_distance_other_agents[:, agent_index] = 0
        obs_pos_other_agents_flat = obs_pos_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_rot_other_agents_flat = obs_rot_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_vel_other_agents_flat = obs_vel_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_ref_path_other_agents_flat = obs_ref_path_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_vertices_other_agents_flat = obs_vertices_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_distance_other_agents_flat = obs_distance_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_others_list = [obs_vertices_other_agents_flat if self.parameters.is_observe_vertices else torch.cat([obs_pos_other_agents_flat, obs_rot_other_agents_flat], dim=-1), obs_vel_other_agents_flat, obs_distance_other_agents_flat if self.parameters.is_observe_distance_to_agents else None, obs_ref_path_other_agents_flat if self.parameters.is_observe_ref_path_other_agents else None]
        obs_others_list = [o for o in obs_others_list if o is not None]
        obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(self.world.batch_dim, -1)
        return obs_other_agents

    def observe_self(self, agent_index):
        """Observe the given agent itself."""
        indexing_tuple_3 = (self.constants.env_idx_broadcasting,) + (agent_index,) + ((agent_index,) if self.parameters.is_ego_view else ())
        indexing_tuple_vel = (self.constants.env_idx_broadcasting,) + (agent_index,) + ((agent_index, 0) if self.parameters.is_ego_view else ())
        obs_self = [None if self.parameters.is_ego_view else self.observations.past_pos.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1), None if self.parameters.is_ego_view else self.observations.past_rot.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1), self.observations.past_vel.get_latest()[indexing_tuple_vel].reshape(self.world.batch_dim, -1), self.observations.past_short_term_ref_points.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1), self.observations.past_distance_to_ref_path.get_latest()[:, agent_index].reshape(self.world.batch_dim, -1) if self.parameters.is_observe_distance_to_center_line else None, self.observations.past_distance_to_left_boundary.get_latest()[:, agent_index].reshape(self.world.batch_dim, -1) if self.parameters.is_observe_distance_to_boundaries else self.observations.past_left_boundary.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1), self.observations.past_distance_to_right_boundary.get_latest()[:, agent_index].reshape(self.world.batch_dim, -1) if self.parameters.is_observe_distance_to_boundaries else self.observations.past_right_boundary.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1)]
        return obs_self

    def done(self):
        """
        This function computes the done flag for each env in a vectorized way.

        Testing mode is designed to test the learned policy. In testing mode, collisions do
        not terminate the current simulation; instead, the colliding agents (not all agents)
        will be reset. Besides, if `map_type` is "3", those agents who leave their entries
        or exits will be reset.
        """
        is_collision_with_agents = self.collisions.with_agents.view(self.world.batch_dim, -1).any(dim=-1)
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        if self.parameters.map_type == '2':
            if torch.rand(1) > 1 - self.initial_state_buffer.probability_record:
                for env_collide in torch.where(is_collision_with_agents)[0]:
                    self.initial_state_buffer.add(self.state_buffer.get_latest(n=self.parameters.n_steps_stored)[env_collide])
        if self.parameters.is_testing_mode:
            is_done = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
            agents_reset = self.collisions.with_agents.any(dim=-1) | self.collisions.with_lanelets | self.collisions.with_entry_segments | self.collisions.with_exit_segments
            agents_reset_indices = torch.where(agents_reset)
            for env_idx, agent_idx in zip(agents_reset_indices[0], agents_reset_indices[1]):
                self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
        elif self.parameters.map_type == '3':
            is_done = is_collision_with_agents | is_collision_with_lanelets
            agents_reset = self.collisions.with_entry_segments | self.collisions.with_exit_segments
            agents_reset_indices = torch.where(agents_reset)
            for env_idx, agent_idx in zip(agents_reset_indices[0], agents_reset_indices[1]):
                if not is_done[env_idx]:
                    self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
        else:
            is_done = is_collision_with_agents | is_collision_with_lanelets
        return is_done

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        This function computes the info dict for "agent" in a vectorized way
        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape (n_envs, info_size)

        Implementors can access the world at "self.world"

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute info of
        :return: info: A dict with a key for each info of interest, and a tensor value  of shape (n_envs, info_size)
        """
        agent_index = self.world.agents.index(agent)
        is_action_empty = agent.action.u is None
        is_collision_with_agents = self.collisions.with_agents[:, agent_index].any(dim=-1)
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        info = {'pos': agent.state.pos / self.normalizers.pos_world, 'rot': angle_eliminate_two_pi(agent.state.rot) / self.normalizers.rot, 'vel': agent.state.vel / self.normalizers.v, 'act_vel': agent.action.u[:, 0] / self.normalizers.action_vel if not is_action_empty else self.constants.empty_action_vel[:, agent_index], 'act_steer': agent.action.u[:, 1] / self.normalizers.action_steering if not is_action_empty else self.constants.empty_action_steering[:, agent_index], 'ref': (self.ref_paths_agent_related.short_term[:, agent_index] / self.normalizers.pos_world).reshape(self.world.batch_dim, -1), 'distance_ref': self.distances.ref_paths[:, agent_index] / self.normalizers.distance_ref, 'distance_left_b': self.distances.left_boundaries[:, agent_index].min(dim=-1)[0] / self.normalizers.distance_lanelet, 'distance_right_b': self.distances.right_boundaries[:, agent_index].min(dim=-1)[0] / self.normalizers.distance_lanelet, 'is_collision_with_agents': is_collision_with_agents, 'is_collision_with_lanelets': is_collision_with_lanelets}
        return info

    def extra_render(self, env_index: int=0):
        from vmas.simulator import rendering
        if self.parameters.is_real_time_rendering:
            if self.timer.step[0] == 0:
                pause_duration = 0
            else:
                pause_duration = self.world.dt - (time.time() - self.timer.render_begin)
            if pause_duration > 0:
                time.sleep(pause_duration)
            self.timer.render_begin = time.time()
        geoms = []
        for i in range(len(self.map_data['lanelets'])):
            lanelet = self.map_data['lanelets'][i]
            geom = rendering.PolyLine(v=lanelet['left_boundary'], close=False)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.set_color(*Color.BLACK.value)
            geoms.append(geom)
            geom = rendering.PolyLine(v=lanelet['right_boundary'], close=False)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.set_color(*Color.BLACK.value)
            geoms.append(geom)
        if self.parameters.is_visualize_extra_info:
            hight_a = -0.1
            hight_b = -0.2
            hight_c = -0.3
            geom = rendering.TextLine(text=self.parameters.render_title, x=0.05 * self.resolution_factor, y=(self.world.y_semidim + hight_a) * self.resolution_factor, font_size=14)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)
            geom = rendering.TextLine(text=f't: {self.timer.step[0] * self.parameters.dt:.2f} sec', x=0.05 * self.resolution_factor, y=(self.world.y_semidim + hight_b) * self.resolution_factor, font_size=14)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)
            geom = rendering.TextLine(text=f'n: {self.timer.step[0]}', x=0.05 * self.resolution_factor, y=(self.world.y_semidim + hight_c) * self.resolution_factor, font_size=14)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)
        for agent_i in range(self.n_agents):
            if self.parameters.is_visualize_short_term_path:
                geom = rendering.PolyLine(v=self.ref_paths_agent_related.short_term[env_index, agent_i], close=False)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)
                for i_p in self.ref_paths_agent_related.short_term[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)
            if not self.parameters.is_observe_distance_to_boundaries:
                geom = rendering.PolyLine(v=self.ref_paths_agent_related.nearing_points_left_boundary[env_index, agent_i], close=False)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)
                for i_p in self.ref_paths_agent_related.nearing_points_left_boundary[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)
                geom = rendering.PolyLine(v=self.ref_paths_agent_related.nearing_points_right_boundary[env_index, agent_i], close=False)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)
                for i_p in self.ref_paths_agent_related.nearing_points_right_boundary[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)
            geom = rendering.TextLine(text=f'{agent_i}', x=self.world.agents[agent_i].state.pos[env_index, 0] / self.world.x_semidim * self.viewer_size[0], y=self.world.agents[agent_i].state.pos[env_index, 1] / self.world.y_semidim * self.viewer_size[1], font_size=14)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)
            if self.parameters.is_visualize_lane_boundary:
                if agent_i == 0:
                    geom = rendering.PolyLine(v=self.ref_paths_agent_related.left_boundary[env_index, agent_i], close=False)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.world.agents[agent_i].color)
                    geoms.append(geom)
                    geom = rendering.PolyLine(v=self.ref_paths_agent_related.right_boundary[env_index, agent_i], close=False)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.world.agents[agent_i].color)
                    geoms.append(geom)
                    geom = rendering.PolyLine(v=self.ref_paths_agent_related.entry[env_index, agent_i], close=False)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.world.agents[agent_i].color)
                    geoms.append(geom)
                    geom = rendering.PolyLine(v=self.ref_paths_agent_related.exit[env_index, agent_i], close=False)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.world.agents[agent_i].color)
                    geoms.append(geom)
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.v_range = kwargs.pop('v_range', 0.5)
        self.a_range = kwargs.pop('a_range', 1)
        self.obs_noise = kwargs.pop('obs_noise', 0)
        self.box_agents = kwargs.pop('box_agents', False)
        self.linear_friction = kwargs.pop('linear_friction', 0.1)
        self.mirror_passage = kwargs.pop('mirror_passage', False)
        self.done_on_completion = kwargs.pop('done_on_completion', False)
        self.observe_rel_pos = kwargs.pop('observe_rel_pos', False)
        self.pos_shaping_factor = kwargs.pop('pos_shaping_factor', 1.0)
        self.final_reward = kwargs.pop('final_reward', 0.01)
        self.energy_reward_coeff = kwargs.pop('energy_rew_coeff', 0)
        self.agent_collision_penalty = kwargs.pop('agent_collision_penalty', 0)
        self.passage_collision_penalty = kwargs.pop('passage_collision_penalty', 0)
        self.obstacle_collision_penalty = kwargs.pop('obstacle_collision_penalty', 0)
        self.use_velocity_controller = kwargs.pop('use_velocity_controller', True)
        self.min_input_norm = kwargs.pop('min_input_norm', 0.08)
        self.dt_delay = kwargs.pop('dt_delay', 0)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.viewer_size = (1600, 700)
        controller_params = [2, 6, 0.002]
        self.f_range = self.a_range + self.linear_friction
        self.u_range = self.v_range if self.use_velocity_controller else self.f_range
        world = World(batch_dim, device, drag=0, dt=0.05, linear_friction=self.linear_friction, substeps=16 if self.box_agents else 5, collision_force=10000 if self.box_agents else 500)
        self.agent_radius = 0.16
        self.agent_box_length = 0.32
        self.agent_box_width = 0.24
        self.spawn_pos_noise = 0.02
        self.min_collision_distance = 0.005
        blue_agent = Agent(name='agent_0', rotatable=False, linear_friction=self.linear_friction, shape=Sphere(radius=self.agent_radius) if not self.box_agents else Box(length=self.agent_box_length, width=self.agent_box_width), u_range=self.u_range, f_range=self.f_range, v_range=self.v_range, render_action=True)
        if self.use_velocity_controller:
            blue_agent.controller = VelocityController(blue_agent, world, controller_params, 'standard')
        blue_goal = Landmark(name='goal_0', collide=False, shape=Sphere(radius=self.agent_radius / 2), color=Color.BLUE)
        blue_agent.goal = blue_goal
        world.add_agent(blue_agent)
        world.add_landmark(blue_goal)
        green_agent = Agent(name='agent_1', color=Color.GREEN, linear_friction=self.linear_friction, shape=Sphere(radius=self.agent_radius) if not self.box_agents else Box(length=self.agent_box_length, width=self.agent_box_width), rotatable=False, u_range=self.u_range, f_range=self.f_range, v_range=self.v_range, render_action=True)
        if self.use_velocity_controller:
            green_agent.controller = VelocityController(green_agent, world, controller_params, 'standard')
        green_goal = Landmark(name='goal_1', collide=False, shape=Sphere(radius=self.agent_radius / 2), color=Color.GREEN)
        green_agent.goal = green_goal
        world.add_agent(green_agent)
        world.add_landmark(green_goal)
        null_action = torch.zeros(world.batch_dim, world.dim_p, device=world.device)
        blue_agent.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        green_agent.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        self.spawn_map(world)
        for agent in world.agents:
            agent.energy_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.energy_rew.clone()
            agent.obstacle_collision_rew = agent.agent_collision_rew.clone()
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        return world

    def reset_world_at(self, env_index: int=None):
        self.world.agents[0].set_pos(torch.tensor([-(self.scenario_length / 2 - self.agent_dist_from_wall), 0.0], dtype=torch.float32, device=self.world.device) + torch.zeros(self.world.dim_p, device=self.world.device).uniform_(-self.spawn_pos_noise, self.spawn_pos_noise), batch_index=env_index)
        if self.use_velocity_controller:
            self.world.agents[0].controller.reset(env_index)
        self.world.landmarks[0].set_pos(torch.tensor([self.scenario_length / 2 - self.goal_dist_from_wall, 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        self.world.agents[1].set_pos(torch.tensor([self.scenario_length / 2 - self.agent_dist_from_wall, 0.0], dtype=torch.float32, device=self.world.device) + torch.zeros(self.world.dim_p, device=self.world.device).uniform_(-self.spawn_pos_noise, self.spawn_pos_noise), batch_index=env_index)
        if self.use_velocity_controller:
            self.world.agents[1].controller.reset(env_index)
        self.world.landmarks[1].set_pos(torch.tensor([-(self.scenario_length / 2 - self.goal_dist_from_wall), 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        self.reset_map(env_index)
        for agent in self.world.agents:
            if env_index is None:
                agent.shaping = torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1) * self.pos_shaping_factor
            else:
                agent.shaping[env_index] = torch.linalg.vector_norm(agent.state.pos[env_index] - agent.goal.state.pos[env_index]) * self.pos_shaping_factor
        if env_index is None:
            self.goal_reached = torch.full((self.world.batch_dim,), False, device=self.world.device)
        else:
            self.goal_reached[env_index] = False

    def process_action(self, agent: Agent):
        if self.use_velocity_controller:
            agent.input_queue.append(agent.action.u.clone())
            agent.action.u = agent.input_queue.pop(0)
            agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, self.u_range)
            action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
            agent.action.u[action_norm < self.min_input_norm] = 0
            agent.vel_action = agent.action.u.clone()
            vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 0.001
            agent.controller.reset(vel_is_zero)
            agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        blue_agent = self.world.agents[0]
        green_agent = self.world.agents[-1]
        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0
            self.blue_distance = torch.linalg.vector_norm(blue_agent.state.pos - blue_agent.goal.state.pos, dim=1)
            self.green_distance = torch.linalg.vector_norm(green_agent.state.pos - green_agent.goal.state.pos, dim=1)
            self.blue_on_goal = self.blue_distance < blue_agent.goal.shape.radius
            self.green_on_goal = self.green_distance < green_agent.goal.shape.radius
            self.goal_reached = self.green_on_goal * self.blue_on_goal
            green_shaping = self.green_distance * self.pos_shaping_factor
            self.green_rew = green_agent.shaping - green_shaping
            green_agent.shaping = green_shaping
            blue_shaping = self.blue_distance * self.pos_shaping_factor
            self.blue_rew = blue_agent.shaping - blue_shaping
            blue_agent.shaping = blue_shaping
            self.pos_rew += self.blue_rew
            self.pos_rew += self.green_rew
            self.final_rew[self.goal_reached] = self.final_reward
        agent.agent_collision_rew[:] = 0
        agent.obstacle_collision_rew[:] = 0
        for a in self.world.agents:
            if a != agent:
                agent.agent_collision_rew[self.world.get_distance(agent, a) <= self.min_collision_distance] += self.agent_collision_penalty
        for landmark in self.world.landmarks:
            if self.world.collides(agent, landmark):
                if landmark in ([*self.passage_1, *self.passage_2] if self.mirror_passage is True else [*self.passage_1]):
                    penalty = self.passage_collision_penalty
                else:
                    penalty = self.obstacle_collision_penalty
                agent.obstacle_collision_rew[self.world.get_distance(agent, landmark) <= self.min_collision_distance] += penalty
        agent.energy_expenditure = torch.linalg.vector_norm(agent.action.u, dim=-1) / math.sqrt(self.world.dim_p * agent.f_range ** 2)
        agent.energy_rew = -agent.energy_expenditure * self.energy_reward_coeff
        return self.pos_rew + agent.obstacle_collision_rew + agent.agent_collision_rew + agent.energy_rew + self.final_rew

    def observation(self, agent: Agent):
        rel = []
        for a in self.world.agents:
            if a != agent:
                rel.append(agent.state.pos - a.state.pos)
        observations = [agent.state.pos, agent.state.vel]
        if self.observe_rel_pos:
            observations += rel
        if self.obs_noise > 0:
            for i, obs in enumerate(observations):
                noise = torch.zeros(*obs.shape, device=self.world.device).uniform_(-self.obs_noise, self.obs_noise)
                observations[i] = obs + noise
        return torch.cat(observations, dim=-1)

    def info(self, agent: Agent):
        return {'pos_rew': self.pos_rew, 'final_rew': self.final_rew, 'energy_rew': agent.energy_rew, 'agent_collision_rew': agent.agent_collision_rew, 'obstacle_collision_rew': agent.obstacle_collision_rew}

    def spawn_map(self, world: World):
        self.scenario_length = 5
        self.passage_length = 0.4
        self.passage_width = 0.48
        self.corridor_width = self.passage_length
        self.small_ceiling_length = self.scenario_length / 2 - self.passage_length / 2
        self.goal_dist_from_wall = self.agent_radius + 0.05
        self.agent_dist_from_wall = 0.5
        self.walls = []
        for i in range(2):
            landmark = Landmark(name=f'wall {i}', collide=True, shape=Line(length=self.corridor_width), color=Color.BLACK)
            self.walls.append(landmark)
            world.add_landmark(landmark)
        self.small_ceilings_1 = []
        for i in range(2):
            landmark = Landmark(name=f'ceil 1 {i}', collide=True, shape=Line(length=self.small_ceiling_length), color=Color.BLACK)
            self.small_ceilings_1.append(landmark)
            world.add_landmark(landmark)
        self.passage_1 = []
        for i in range(3):
            landmark = Landmark(name=f'ceil 2 {i}', collide=True, shape=Line(length=self.passage_length if i == 2 else self.passage_width), color=Color.BLACK)
            self.passage_1.append(landmark)
            world.add_landmark(landmark)
        if self.mirror_passage:
            self.small_ceilings_2 = []
            for i in range(2):
                landmark = Landmark(name=f'ceil 12 {i}', collide=True, shape=Line(length=self.small_ceiling_length), color=Color.BLACK)
                self.small_ceilings_2.append(landmark)
                world.add_landmark(landmark)
            self.passage_2 = []
            for i in range(3):
                landmark = Landmark(name=f'ceil 22 {i}', collide=True, shape=Line(length=self.passage_length if i == 2 else self.passage_width), color=Color.BLACK)
                self.passage_2.append(landmark)
                world.add_landmark(landmark)
        else:
            landmark = Landmark(name='floor', collide=True, shape=Line(length=self.scenario_length), color=Color.BLACK)
            self.floor = landmark
            world.add_landmark(landmark)

    def reset_map(self, env_index):
        for i, landmark in enumerate(self.walls):
            landmark.set_pos(torch.tensor([-self.scenario_length / 2 if i == 0 else self.scenario_length / 2, 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        small_ceiling_pos = self.small_ceiling_length / 2 - self.scenario_length / 2
        for i, landmark in enumerate(self.small_ceilings_1):
            landmark.set_pos(torch.tensor([-small_ceiling_pos if i == 0 else small_ceiling_pos, self.passage_length / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        for i, landmark in enumerate(self.passage_1[:-1]):
            landmark.set_pos(torch.tensor([-self.passage_length / 2 if i == 0 else self.passage_length / 2, self.passage_length / 2 + self.passage_width / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        self.passage_1[-1].set_pos(torch.tensor([0, self.passage_length / 2 + self.passage_width], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        if self.mirror_passage:
            for i, landmark in enumerate(self.small_ceilings_2):
                landmark.set_pos(torch.tensor([-small_ceiling_pos if i == 0 else small_ceiling_pos, -self.passage_length / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            for i, landmark in enumerate(self.passage_2[:-1]):
                landmark.set_pos(torch.tensor([-self.passage_length / 2 if i == 0 else self.passage_length / 2, -self.passage_length / 2 - self.passage_width / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            self.passage_2[-1].set_pos(torch.tensor([0, -self.passage_length / 2 - self.passage_width], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        else:
            self.floor.set_pos(torch.tensor([0, -self.passage_length / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)

    def done(self):
        if self.done_on_completion:
            return self.goal_reached
        else:
            return torch.zeros_like(self.goal_reached)

class Scenario(BaseScenario):

    def init_params(self, **kwargs):
        self.viewer_size = kwargs.pop('viewer_size', (1200, 800))
        self.n_blue_agents = kwargs.pop('n_blue_agents', 3)
        self.n_red_agents = kwargs.pop('n_red_agents', 3)
        self.ai_red_agents = kwargs.pop('ai_red_agents', True)
        self.ai_blue_agents = kwargs.pop('ai_blue_agents', False)
        self.physically_different = kwargs.pop('physically_different', False)
        self.spawn_in_formation = kwargs.pop('spawn_in_formation', False)
        self.only_blue_formation = kwargs.pop('only_blue_formation', True)
        self.formation_agents_per_column = kwargs.pop('formation_agents_per_column', 2)
        self.randomise_formation_indices = kwargs.pop('randomise_formation_indices', False)
        self.formation_noise = kwargs.pop('formation_noise', 0.2)
        self.n_traj_points = kwargs.pop('n_traj_points', 0)
        self.ai_speed_strength = kwargs.pop('ai_strength', 1.0)
        self.ai_decision_strength = kwargs.pop('ai_decision_strength', 1.0)
        self.ai_precision_strength = kwargs.pop('ai_precision_strength', 1.0)
        self.disable_ai_red = kwargs.pop('disable_ai_red', False)
        self.agent_size = kwargs.pop('agent_size', 0.025)
        self.goal_size = kwargs.pop('goal_size', 0.35)
        self.goal_depth = kwargs.pop('goal_depth', 0.1)
        self.pitch_length = kwargs.pop('pitch_length', 3.0)
        self.pitch_width = kwargs.pop('pitch_width', 1.5)
        self.ball_mass = kwargs.pop('ball_mass', 0.25)
        self.ball_size = kwargs.pop('ball_size', 0.02)
        self.u_multiplier = kwargs.pop('u_multiplier', 0.1)
        self.enable_shooting = kwargs.pop('enable_shooting', False)
        self.u_rot_multiplier = kwargs.pop('u_rot_multiplier', 0.0003)
        self.u_shoot_multiplier = kwargs.pop('u_shoot_multiplier', 0.6)
        self.shooting_radius = kwargs.pop('shooting_radius', 0.08)
        self.shooting_angle = kwargs.pop('shooting_angle', torch.pi / 2)
        self.max_speed = kwargs.pop('max_speed', 0.15)
        self.ball_max_speed = kwargs.pop('ball_max_speed', 0.3)
        self.dense_reward = kwargs.pop('dense_reward', True)
        self.pos_shaping_factor_ball_goal = kwargs.pop('pos_shaping_factor_ball_goal', 10.0)
        self.pos_shaping_factor_agent_ball = kwargs.pop('pos_shaping_factor_agent_ball', 0.1)
        self.distance_to_ball_trigger = kwargs.pop('distance_to_ball_trigger', 0.4)
        self.scoring_reward = kwargs.pop('scoring_reward', 100.0)
        self.observe_teammates = kwargs.pop('observe_teammates', True)
        self.observe_adversaries = kwargs.pop('observe_adversaries', True)
        self.dict_obs = kwargs.pop('dict_obs', False)
        if kwargs.pop('dense_reward_ratio', None) is not None:
            raise ValueError('dense_reward_ratio in football is deprecated, please use `dense_reward` which is a bool that turns on/off the dense reward')
        ScenarioUtils.check_kwargs_consumed(kwargs)

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(**kwargs)
        self.visualize_semidims = False
        world = self.init_world(batch_dim, device)
        self.init_agents(world)
        self.init_ball(world)
        self.init_background()
        self.init_walls(world)
        self.init_goals(world)
        self.init_traj_pts(world)
        self.left_goal_pos = torch.tensor([-self.pitch_length / 2 - self.ball_size / 2, 0], device=device, dtype=torch.float)
        self.right_goal_pos = -self.left_goal_pos
        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self._sparse_reward_blue = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self._sparse_reward_red = self._sparse_reward_blue.clone()
        self._render_field = True
        self.min_agent_dist_to_ball_blue = None
        self.min_agent_dist_to_ball_red = None
        self._reset_agent_range = torch.tensor([self.pitch_length / 2, self.pitch_width], device=device)
        self._reset_agent_offset_blue = torch.tensor([-self.pitch_length / 2 + self.agent_size, -self.pitch_width / 2], device=device)
        self._reset_agent_offset_red = torch.tensor([-self.agent_size, -self.pitch_width / 2], device=device)
        self._agents_rel_pos_to_ball = None
        return world

    def reset_world_at(self, env_index: int=None):
        self.reset_agents(env_index)
        self.reset_ball(env_index)
        self.reset_walls(env_index)
        self.reset_goals(env_index)
        self.reset_controllers(env_index)
        if env_index is None:
            self._done[:] = False
        else:
            self._done[env_index] = False

    def init_world(self, batch_dim: int, device: torch.device):
        world = World(batch_dim, device, dt=0.1, drag=0.05, x_semidim=self.pitch_length / 2 + self.goal_depth - self.agent_size, y_semidim=self.pitch_width / 2 - self.agent_size, substeps=2)
        world.agent_size = self.agent_size
        world.pitch_width = self.pitch_width
        world.pitch_length = self.pitch_length
        world.goal_size = self.goal_size
        world.goal_depth = self.goal_depth
        return world

    def init_agents(self, world):
        self.blue_color = (0.22, 0.49, 0.72)
        self.red_color = (0.89, 0.1, 0.11)
        self.red_controller = AgentPolicy(team='Red', disabled=self.disable_ai_red, speed_strength=self.ai_speed_strength[1] if isinstance(self.ai_speed_strength, tuple) else self.ai_speed_strength, precision_strength=self.ai_precision_strength[1] if isinstance(self.ai_precision_strength, tuple) else self.ai_precision_strength, decision_strength=self.ai_decision_strength[1] if isinstance(self.ai_decision_strength, tuple) else self.ai_decision_strength) if self.ai_red_agents else None
        self.blue_controller = AgentPolicy(team='Blue', speed_strength=self.ai_speed_strength[0] if isinstance(self.ai_speed_strength, tuple) else self.ai_speed_strength, precision_strength=self.ai_precision_strength[0] if isinstance(self.ai_precision_strength, tuple) else self.ai_precision_strength, decision_strength=self.ai_decision_strength[0] if isinstance(self.ai_decision_strength, tuple) else self.ai_decision_strength) if self.ai_blue_agents else None
        blue_agents = []
        if self.physically_different:
            blue_agents = self.get_physically_different_agents()
            for agent in blue_agents:
                world.add_agent(agent)
        else:
            for i in range(self.n_blue_agents):
                agent = Agent(name=f'agent_blue_{i}', shape=Sphere(radius=self.agent_size), action_script=self.blue_controller.run if self.ai_blue_agents else None, u_multiplier=[self.u_multiplier, self.u_multiplier] if not self.enable_shooting else [self.u_multiplier, self.u_multiplier, self.u_rot_multiplier, self.u_shoot_multiplier], max_speed=self.max_speed, dynamics=Holonomic() if not self.enable_shooting else HolonomicWithRotation(), action_size=2 if not self.enable_shooting else 4, color=self.blue_color, alpha=1)
                world.add_agent(agent)
                blue_agents.append(agent)
        self.blue_agents = blue_agents
        world.blue_agents = blue_agents
        red_agents = []
        for i in range(self.n_red_agents):
            agent = Agent(name=f'agent_red_{i}', shape=Sphere(radius=self.agent_size), action_script=self.red_controller.run if self.ai_red_agents else None, u_multiplier=[self.u_multiplier, self.u_multiplier] if not self.enable_shooting or self.ai_red_agents else [self.u_multiplier, self.u_multiplier, self.u_rot_multiplier, self.u_shoot_multiplier], max_speed=self.max_speed, dynamics=Holonomic() if not self.enable_shooting or self.ai_red_agents else HolonomicWithRotation(), action_size=2 if not self.enable_shooting or self.ai_red_agents else 4, color=self.red_color, alpha=1)
            world.add_agent(agent)
            red_agents.append(agent)
        self.red_agents = red_agents
        world.red_agents = red_agents
        for agent in self.blue_agents + self.red_agents:
            agent.ball_within_angle = torch.zeros(world.batch_dim, device=agent.device, dtype=torch.bool)
            agent.ball_within_range = torch.zeros(world.batch_dim, device=agent.device, dtype=torch.bool)
            agent.shoot_force = torch.zeros(world.batch_dim, 2, device=agent.device, dtype=torch.float32)

    def get_physically_different_agents(self):
        assert self.n_blue_agents == 5, 'Physical differences only for 5 agents'

        def attacker(i):
            attacker_shoot_multiplier_decrease = -0.2
            attacker_multiplier_increase = 0.1
            attacker_speed_increase = 0.05
            attacker_radius_decrease = -0.005
            return Agent(name=f'agent_blue_{i}', shape=Sphere(radius=self.agent_size + attacker_radius_decrease), action_script=self.blue_controller.run if self.ai_blue_agents else None, u_multiplier=[self.u_multiplier + attacker_multiplier_increase, self.u_multiplier + attacker_multiplier_increase] if not self.enable_shooting else [self.u_multiplier + attacker_multiplier_increase, self.u_multiplier + attacker_multiplier_increase, self.u_rot_multiplier, self.u_shoot_multiplier + attacker_shoot_multiplier_decrease], max_speed=self.max_speed + attacker_speed_increase, dynamics=Holonomic() if not self.enable_shooting else HolonomicWithRotation(), action_size=2 if not self.enable_shooting else 4, color=self.blue_color, alpha=1)

        def defender(i):
            return Agent(name=f'agent_blue_{i}', shape=Sphere(radius=self.agent_size), action_script=self.blue_controller.run if self.ai_blue_agents else None, u_multiplier=[self.u_multiplier, self.u_multiplier] if not self.enable_shooting else [self.u_multiplier, self.u_multiplier, self.u_rot_multiplier, self.u_shoot_multiplier], max_speed=self.max_speed, dynamics=Holonomic() if not self.enable_shooting else HolonomicWithRotation(), action_size=2 if not self.enable_shooting else 4, color=self.blue_color, alpha=1)

        def goal_keeper(i):
            goalie_shoot_multiplier_increase = 0.2
            goalie_radius_increase = 0.01
            goalie_speed_decrease = -0.1
            goalie_multiplier_decrease = -0.05
            return Agent(name=f'agent_blue_{i}', shape=Sphere(radius=self.agent_size + goalie_radius_increase), action_script=self.blue_controller.run if self.ai_blue_agents else None, u_multiplier=[self.u_multiplier + goalie_multiplier_decrease, self.u_multiplier + goalie_multiplier_decrease] if not self.enable_shooting else [self.u_multiplier + goalie_multiplier_decrease, self.u_multiplier + goalie_multiplier_decrease, self.u_rot_multiplier + goalie_shoot_multiplier_increase, self.u_shoot_multiplier], max_speed=self.max_speed + goalie_speed_decrease, dynamics=Holonomic() if not self.enable_shooting else HolonomicWithRotation(), action_size=2 if not self.enable_shooting else 4, color=self.blue_color, alpha=1)
        agents = [attacker(0), attacker(1), defender(2), defender(3), goal_keeper(4)]
        return agents

    def reset_agents(self, env_index: int=None):
        if self.spawn_in_formation:
            self._spawn_formation(self.blue_agents, True, env_index)
            if not self.only_blue_formation:
                self._spawn_formation(self.red_agents, False, env_index)
        else:
            for agent in self.blue_agents:
                pos = self._get_random_spawn_position(blue=True, env_index=env_index)
                agent.set_pos(pos, batch_index=env_index)
        if self.spawn_in_formation and self.only_blue_formation or not self.spawn_in_formation:
            for agent in self.red_agents:
                pos = self._get_random_spawn_position(blue=False, env_index=env_index)
                agent.set_pos(pos, batch_index=env_index)
                agent.set_rot(torch.tensor([torch.pi], device=self.world.device, dtype=torch.float32), batch_index=env_index)

    def _spawn_formation(self, agents, blue, env_index):
        if self.randomise_formation_indices:
            order = torch.randperm(len(agents)).tolist()
            agents = [agents[i] for i in order]
        agent_index = 0
        endpoint = -(self.pitch_length / 2 + self.goal_depth) * (1 if blue else -1)
        for x in torch.linspace(0, endpoint, len(agents) // self.formation_agents_per_column + 3):
            if agent_index >= len(agents):
                break
            if x == 0 or x == endpoint:
                continue
            agents_this_column = agents[agent_index:agent_index + self.formation_agents_per_column]
            n_agents_this_column = len(agents_this_column)
            for y in torch.linspace(self.pitch_width / 2, -self.pitch_width / 2, n_agents_this_column + 2):
                if y == -self.pitch_width / 2 or y == self.pitch_width / 2:
                    continue
                pos = torch.tensor([x, y], device=self.world.device, dtype=torch.float32)
                if env_index is None:
                    pos = pos.expand(self.world.batch_dim, self.world.dim_p)
                agents[agent_index].set_pos(pos + (torch.rand((self.world.dim_p,) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device) - 0.5) * self.formation_noise, batch_index=env_index)
                agent_index += 1

    def _get_random_spawn_position(self, blue, env_index):
        return torch.rand((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device) * self._reset_agent_range + (self._reset_agent_offset_blue if blue else self._reset_agent_offset_red)

    def reset_controllers(self, env_index: int=None):
        if self.red_controller is not None:
            if not self.red_controller.initialised:
                self.red_controller.init(self.world)
            self.red_controller.reset(env_index)
        if self.blue_controller is not None:
            if not self.blue_controller.initialised:
                self.blue_controller.init(self.world)
            self.blue_controller.reset(env_index)

    def init_ball(self, world):
        ball = Agent(name='Ball', shape=Sphere(radius=self.ball_size), action_script=ball_action_script, max_speed=self.ball_max_speed, mass=self.ball_mass, alpha=1, color=Color.BLACK)
        ball.pos_rew_blue = torch.zeros(world.batch_dim, device=world.device, dtype=torch.float32)
        ball.pos_rew_red = ball.pos_rew_blue.clone()
        ball.pos_rew_agent_blue = ball.pos_rew_blue.clone()
        ball.pos_rew_agent_red = ball.pos_rew_red.clone()
        ball.kicking_action = torch.zeros(world.batch_dim, world.dim_p, device=world.device, dtype=torch.float32)
        world.add_agent(ball)
        world.ball = ball
        self.ball = ball

    def reset_ball(self, env_index: int=None):
        if not self.ai_blue_agents:
            min_agent_dist_to_ball_blue = self.get_closest_agent_to_ball(self.blue_agents, env_index)
            if env_index is None:
                self.min_agent_dist_to_ball_blue = min_agent_dist_to_ball_blue
            else:
                self.min_agent_dist_to_ball_blue[env_index] = min_agent_dist_to_ball_blue
        if not self.ai_red_agents:
            min_agent_dist_to_ball_red = self.get_closest_agent_to_ball(self.red_agents, env_index)
            if env_index is None:
                self.min_agent_dist_to_ball_red = min_agent_dist_to_ball_red
            else:
                self.min_agent_dist_to_ball_red[env_index] = min_agent_dist_to_ball_red
        if env_index is None:
            if not self.ai_blue_agents:
                self.ball.pos_shaping_blue = torch.linalg.vector_norm(self.ball.state.pos - self.right_goal_pos, dim=-1) * self.pos_shaping_factor_ball_goal
                self.ball.pos_shaping_agent_blue = self.min_agent_dist_to_ball_blue * self.pos_shaping_factor_agent_ball
            if not self.ai_red_agents:
                self.ball.pos_shaping_red = torch.linalg.vector_norm(self.ball.state.pos - self.left_goal_pos, dim=-1) * self.pos_shaping_factor_ball_goal
                self.ball.pos_shaping_agent_red = self.min_agent_dist_to_ball_red * self.pos_shaping_factor_agent_ball
            if self.enable_shooting:
                self.ball.kicking_action[:] = 0.0
        else:
            if not self.ai_blue_agents:
                self.ball.pos_shaping_blue[env_index] = torch.linalg.vector_norm(self.ball.state.pos[env_index] - self.right_goal_pos) * self.pos_shaping_factor_ball_goal
                self.ball.pos_shaping_agent_blue[env_index] = self.min_agent_dist_to_ball_blue[env_index] * self.pos_shaping_factor_agent_ball
            if not self.ai_red_agents:
                self.ball.pos_shaping_red[env_index] = torch.linalg.vector_norm(self.ball.state.pos[env_index] - self.left_goal_pos) * self.pos_shaping_factor_ball_goal
                self.ball.pos_shaping_agent_red[env_index] = self.min_agent_dist_to_ball_red[env_index] * self.pos_shaping_factor_agent_ball
            if self.enable_shooting:
                self.ball.kicking_action[env_index] = 0.0

    def get_closest_agent_to_ball(self, team, env_index):
        pos = torch.stack([a.state.pos for a in team], dim=-2)
        ball_pos = self.ball.state.pos.unsqueeze(-2)
        if isinstance(env_index, int):
            pos = pos[env_index].unsqueeze(0)
            ball_pos = ball_pos[env_index].unsqueeze(0)
        dist = torch.cdist(pos, ball_pos)
        dist = dist.squeeze(-1)
        min_dist = dist.min(dim=-1)[0]
        if isinstance(env_index, int):
            min_dist = min_dist.squeeze(0)
        return min_dist

    def init_background(self):
        self.background = Landmark(name='Background', collide=False, movable=False, shape=Box(length=self.pitch_length, width=self.pitch_width), color=Color.GREEN)
        self.centre_circle_outer = Landmark(name='Centre Circle Outer', collide=False, movable=False, shape=Sphere(radius=self.goal_size / 2), color=Color.WHITE)
        self.centre_circle_inner = Landmark(name='Centre Circle Inner', collide=False, movable=False, shape=Sphere(self.goal_size / 2 - 0.02), color=Color.GREEN)
        centre_line = Landmark(name='Centre Line', collide=False, movable=False, shape=Line(length=self.pitch_width - 2 * self.agent_size), color=Color.WHITE)
        right_line = Landmark(name='Right Line', collide=False, movable=False, shape=Line(length=self.pitch_width - 2 * self.agent_size), color=Color.WHITE)
        left_line = Landmark(name='Left Line', collide=False, movable=False, shape=Line(length=self.pitch_width - 2 * self.agent_size), color=Color.WHITE)
        top_line = Landmark(name='Top Line', collide=False, movable=False, shape=Line(length=self.pitch_length - 2 * self.agent_size), color=Color.WHITE)
        bottom_line = Landmark(name='Bottom Line', collide=False, movable=False, shape=Line(length=self.pitch_length - 2 * self.agent_size), color=Color.WHITE)
        self.background_entities = [self.background, self.centre_circle_outer, self.centre_circle_inner, centre_line, right_line, left_line, top_line, bottom_line]

    def render_field(self, render: bool):
        self._render_field = render
        self.left_top_wall.is_rendering[:] = render
        self.left_bottom_wall.is_rendering[:] = render
        self.right_top_wall.is_rendering[:] = render
        self.right_bottom_wall.is_rendering[:] = render

    def init_walls(self, world):
        self.right_top_wall = Landmark(name='Right Top Wall', collide=True, movable=False, shape=Line(length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2), color=Color.WHITE)
        world.add_landmark(self.right_top_wall)
        self.left_top_wall = Landmark(name='Left Top Wall', collide=True, movable=False, shape=Line(length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2), color=Color.WHITE)
        world.add_landmark(self.left_top_wall)
        self.right_bottom_wall = Landmark(name='Right Bottom Wall', collide=True, movable=False, shape=Line(length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2), color=Color.WHITE)
        world.add_landmark(self.right_bottom_wall)
        self.left_bottom_wall = Landmark(name='Left Bottom Wall', collide=True, movable=False, shape=Line(length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2), color=Color.WHITE)
        world.add_landmark(self.left_bottom_wall)

    def reset_walls(self, env_index: int=None):
        for landmark in self.world.landmarks:
            if landmark.name == 'Left Top Wall':
                landmark.set_pos(torch.tensor([-self.pitch_length / 2, self.pitch_width / 4 + self.goal_size / 4], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Left Bottom Wall':
                landmark.set_pos(torch.tensor([-self.pitch_length / 2, -self.pitch_width / 4 - self.goal_size / 4], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Right Top Wall':
                landmark.set_pos(torch.tensor([self.pitch_length / 2, self.pitch_width / 4 + self.goal_size / 4], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Right Bottom Wall':
                landmark.set_pos(torch.tensor([self.pitch_length / 2, -self.pitch_width / 4 - self.goal_size / 4], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)

    def init_goals(self, world):
        right_goal_back = Landmark(name='Right Goal Back', collide=True, movable=False, shape=Line(length=self.goal_size), color=Color.WHITE)
        world.add_landmark(right_goal_back)
        left_goal_back = Landmark(name='Left Goal Back', collide=True, movable=False, shape=Line(length=self.goal_size), color=Color.WHITE)
        world.add_landmark(left_goal_back)
        right_goal_top = Landmark(name='Right Goal Top', collide=True, movable=False, shape=Line(length=self.goal_depth), color=Color.WHITE)
        world.add_landmark(right_goal_top)
        left_goal_top = Landmark(name='Left Goal Top', collide=True, movable=False, shape=Line(length=self.goal_depth), color=Color.WHITE)
        world.add_landmark(left_goal_top)
        right_goal_bottom = Landmark(name='Right Goal Bottom', collide=True, movable=False, shape=Line(length=self.goal_depth), color=Color.WHITE)
        world.add_landmark(right_goal_bottom)
        left_goal_bottom = Landmark(name='Left Goal Bottom', collide=True, movable=False, shape=Line(length=self.goal_depth), color=Color.WHITE)
        world.add_landmark(left_goal_bottom)
        blue_net = Landmark(name='Blue Net', collide=False, movable=False, shape=Box(length=self.goal_depth, width=self.goal_size), color=(0.5, 0.5, 0.5, 0.5))
        world.add_landmark(blue_net)
        red_net = Landmark(name='Red Net', collide=False, movable=False, shape=Box(length=self.goal_depth, width=self.goal_size), color=(0.5, 0.5, 0.5, 0.5))
        world.add_landmark(red_net)
        self.blue_net = blue_net
        self.red_net = red_net
        world.blue_net = blue_net
        world.red_net = red_net

    def reset_goals(self, env_index: int=None):
        for landmark in self.world.landmarks:
            if landmark.name == 'Left Goal Back':
                landmark.set_pos(torch.tensor([-self.pitch_length / 2 - self.goal_depth + self.agent_size, 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Right Goal Back':
                landmark.set_pos(torch.tensor([self.pitch_length / 2 + self.goal_depth - self.agent_size, 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
                landmark.set_rot(torch.tensor([torch.pi / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Left Goal Top':
                landmark.set_pos(torch.tensor([-self.pitch_length / 2 - self.goal_depth / 2 + self.agent_size, self.goal_size / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Left Goal Bottom':
                landmark.set_pos(torch.tensor([-self.pitch_length / 2 - self.goal_depth / 2 + self.agent_size, -self.goal_size / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Right Goal Top':
                landmark.set_pos(torch.tensor([self.pitch_length / 2 + self.goal_depth / 2 - self.agent_size, self.goal_size / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Right Goal Bottom':
                landmark.set_pos(torch.tensor([self.pitch_length / 2 + self.goal_depth / 2 - self.agent_size, -self.goal_size / 2], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Red Net':
                landmark.set_pos(torch.tensor([self.pitch_length / 2 + self.goal_depth / 2 - self.agent_size / 2, 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            elif landmark.name == 'Blue Net':
                landmark.set_pos(torch.tensor([-self.pitch_length / 2 - self.goal_depth / 2 + self.agent_size / 2, 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)

    def init_traj_pts(self, world):
        world.traj_points = {'Red': {}, 'Blue': {}}
        if self.ai_red_agents:
            for i, agent in enumerate(world.red_agents):
                world.traj_points['Red'][agent] = []
                for j in range(self.n_traj_points):
                    pointj = Landmark(name='Red {agent} Trajectory {pt}'.format(agent=i, pt=j), collide=False, movable=False, shape=Sphere(radius=0.01), color=Color.GRAY)
                    world.add_landmark(pointj)
                    world.traj_points['Red'][agent].append(pointj)
        if self.ai_blue_agents:
            for i, agent in enumerate(world.blue_agents):
                world.traj_points['Blue'][agent] = []
                for j in range(self.n_traj_points):
                    pointj = Landmark(name='Blue {agent} Trajectory {pt}'.format(agent=i, pt=j), collide=False, movable=False, shape=Sphere(radius=0.01), color=Color.GRAY)
                    world.add_landmark(pointj)
                    world.traj_points['Blue'][agent].append(pointj)

    def process_action(self, agent: Agent):
        if agent is self.ball:
            return
        blue = agent in self.blue_agents
        if agent.action_script is None and (not blue):
            agent.action.u[..., X] = -agent.action.u[..., X]
            if self.enable_shooting:
                agent.action.u[..., 2] = -agent.action.u[..., 2]
        if self.enable_shooting and agent.action_script is None:
            agents_exclude_ball = [a for a in self.world.agents if a is not self.ball]
            if self._agents_rel_pos_to_ball is None:
                self._agents_rel_pos_to_ball = torch.stack([self.ball.state.pos - a.state.pos for a in agents_exclude_ball], dim=1)
                self._agent_dist_to_ball = torch.linalg.vector_norm(self._agents_rel_pos_to_ball, dim=-1)
                self._agents_closest_to_ball = self._agent_dist_to_ball == self._agent_dist_to_ball.min(dim=-1, keepdim=True)[0]
            agent_index = agents_exclude_ball.index(agent)
            rel_pos = self._agents_rel_pos_to_ball[:, agent_index]
            agent.ball_within_range = self._agent_dist_to_ball[:, agent_index] <= self.shooting_radius
            rel_pos_angle = torch.atan2(rel_pos[:, Y], rel_pos[:, X])
            a = (agent.state.rot.squeeze(-1) - rel_pos_angle + torch.pi) % (2 * torch.pi) - torch.pi
            agent.ball_within_angle = (-self.shooting_angle / 2 <= a) * (a <= self.shooting_angle / 2)
            shoot_force = torch.zeros(self.world.batch_dim, 2, device=self.world.device, dtype=torch.float32)
            shoot_force[..., X] = agent.action.u[..., -1] * 2.67 * self.u_shoot_multiplier
            shoot_force = TorchUtils.rotate_vector(shoot_force, agent.state.rot)
            agent.shoot_force = shoot_force
            shoot_force = torch.where((agent.ball_within_angle * agent.ball_within_range * self._agents_closest_to_ball[:, agent_index]).unsqueeze(-1), shoot_force, 0.0)
            self.ball.kicking_action += shoot_force
            agent.action.u = agent.action.u[:, :-1]

    def pre_step(self):
        if self.enable_shooting:
            self._agents_rel_pos_to_ball = None
            self.ball.action.u += self.ball.kicking_action
            self.ball.kicking_action[:] = 0

    def reward(self, agent: Agent):
        if agent is None or agent == self.world.agents[0]:
            over_right_line = self.ball.state.pos[:, X] > self.pitch_length / 2 + self.ball_size / 2
            over_left_line = self.ball.state.pos[:, X] < -self.pitch_length / 2 - self.ball_size / 2
            goal_mask = (self.ball.state.pos[:, Y] <= self.goal_size / 2) * (self.ball.state.pos[:, Y] >= -self.goal_size / 2)
            blue_score = over_right_line * goal_mask
            red_score = over_left_line * goal_mask
            self._sparse_reward_blue = self.scoring_reward * blue_score - self.scoring_reward * red_score
            self._sparse_reward_red = -self._sparse_reward_blue
            self._done = blue_score | red_score
            self._dense_reward_blue = 0
            self._dense_reward_red = 0
            if self.dense_reward and agent is not None:
                if not self.ai_blue_agents:
                    self._dense_reward_blue = self.reward_ball_to_goal(blue=True) + self.reward_all_agent_to_ball(blue=True)
                if not self.ai_red_agents:
                    self._dense_reward_red = self.reward_ball_to_goal(blue=False) + self.reward_all_agent_to_ball(blue=False)
        blue = agent in self.blue_agents
        if blue:
            reward = self._sparse_reward_blue + self._dense_reward_blue
        else:
            reward = self._sparse_reward_red + self._dense_reward_red
        return reward

    def reward_ball_to_goal(self, blue: bool):
        if blue:
            self.ball.distance_to_goal_blue = torch.linalg.vector_norm(self.ball.state.pos - self.right_goal_pos, dim=-1)
            distance_to_goal = self.ball.distance_to_goal_blue
        else:
            self.ball.distance_to_goal_red = torch.linalg.vector_norm(self.ball.state.pos - self.left_goal_pos, dim=-1)
            distance_to_goal = self.ball.distance_to_goal_red
        pos_shaping = distance_to_goal * self.pos_shaping_factor_ball_goal
        if blue:
            self.ball.pos_rew_blue = self.ball.pos_shaping_blue - pos_shaping
            self.ball.pos_shaping_blue = pos_shaping
            pos_rew = self.ball.pos_rew_blue
        else:
            self.ball.pos_rew_red = self.ball.pos_shaping_red - pos_shaping
            self.ball.pos_shaping_red = pos_shaping
            pos_rew = self.ball.pos_rew_red
        return pos_rew

    def reward_all_agent_to_ball(self, blue: bool):
        min_dist_to_ball = self.get_closest_agent_to_ball(team=self.blue_agents if blue else self.red_agents, env_index=None)
        if blue:
            self.min_agent_dist_to_ball_blue = min_dist_to_ball
        else:
            self.min_agent_dist_to_ball_red = min_dist_to_ball
        pos_shaping = min_dist_to_ball * self.pos_shaping_factor_agent_ball
        ball_moving = torch.linalg.vector_norm(self.ball.state.vel, dim=-1) > 1e-06
        agent_close_to_goal = min_dist_to_ball < self.distance_to_ball_trigger
        if blue:
            self.ball.pos_rew_agent_blue = torch.where(agent_close_to_goal + ball_moving, 0.0, self.ball.pos_shaping_agent_blue - pos_shaping)
            self.ball.pos_shaping_agent_blue = pos_shaping
            pos_rew_agent = self.ball.pos_rew_agent_blue
        else:
            self.ball.pos_rew_agent_red = torch.where(agent_close_to_goal + ball_moving, 0.0, self.ball.pos_shaping_agent_red - pos_shaping)
            self.ball.pos_shaping_agent_red = pos_shaping
            pos_rew_agent = self.ball.pos_rew_agent_red
        return pos_rew_agent

    def observation(self, agent: Agent, agent_pos=None, agent_rot=None, agent_vel=None, agent_force=None, teammate_poses=None, teammate_forces=None, teammate_vels=None, adversary_poses=None, adversary_forces=None, adversary_vels=None, ball_pos=None, ball_vel=None, ball_force=None, blue=None, env_index=Ellipsis):
        if blue:
            assert agent in self.blue_agents
        else:
            blue = agent in self.blue_agents
        if not blue:
            my_team, other_team = (self.red_agents, self.blue_agents)
            goal_pos = self.left_goal_pos
        else:
            my_team, other_team = (self.blue_agents, self.red_agents)
            goal_pos = self.right_goal_pos
        actual_adversary_poses = []
        actual_adversary_forces = []
        actual_adversary_vels = []
        if self.observe_adversaries:
            for a in other_team:
                actual_adversary_poses.append(a.state.pos[env_index])
                actual_adversary_vels.append(a.state.vel[env_index])
                actual_adversary_forces.append(a.state.force[env_index])
        actual_teammate_poses = []
        actual_teammate_forces = []
        actual_teammate_vels = []
        if self.observe_teammates:
            for a in my_team:
                if a != agent:
                    actual_teammate_poses.append(a.state.pos[env_index])
                    actual_teammate_vels.append(a.state.vel[env_index])
                    actual_teammate_forces.append(a.state.force[env_index])
        obs = self.observation_base(agent.state.pos[env_index] if agent_pos is None else agent_pos, agent.state.rot[env_index] if agent_rot is None else agent_rot, agent.state.vel[env_index] if agent_vel is None else agent_vel, agent.state.force[env_index] if agent_force is None else agent_force, goal_pos=goal_pos, ball_pos=self.ball.state.pos[env_index] if ball_pos is None else ball_pos, ball_vel=self.ball.state.vel[env_index] if ball_vel is None else ball_vel, ball_force=self.ball.state.force[env_index] if ball_force is None else ball_force, adversary_poses=actual_adversary_poses if adversary_poses is None else adversary_poses, adversary_forces=actual_adversary_forces if adversary_forces is None else adversary_forces, adversary_vels=actual_adversary_vels if adversary_vels is None else adversary_vels, teammate_poses=actual_teammate_poses if teammate_poses is None else teammate_poses, teammate_forces=actual_teammate_forces if teammate_forces is None else teammate_forces, teammate_vels=actual_teammate_vels if teammate_vels is None else teammate_vels, blue=blue)
        return obs

    def observation_base(self, agent_pos, agent_rot, agent_vel, agent_force, teammate_poses, teammate_forces, teammate_vels, adversary_poses, adversary_forces, adversary_vels, ball_pos, ball_vel, ball_force, goal_pos, blue: bool):
        input = [agent_pos, agent_rot, agent_vel, agent_force, ball_pos, ball_vel, ball_force, goal_pos, teammate_poses, teammate_forces, teammate_vels, adversary_poses, adversary_forces, adversary_vels]
        for o in input:
            if isinstance(o, Tensor) and len(o.shape) > 1:
                batch_dim = o.shape[0]
                break
        for j in range(len(input)):
            if isinstance(input[j], Tensor):
                if len(input[j].shape) == 1:
                    input[j] = input[j].unsqueeze(0).expand(batch_dim, *input[j].shape)
                input[j] = input[j].clone()
            else:
                o = input[j]
                for i in range(len(o)):
                    if len(o[i].shape) == 1:
                        o[i] = o[i].unsqueeze(0).expand(batch_dim, *o[i].shape)
                    o[i] = o[i].clone()
        agent_pos, agent_rot, agent_vel, agent_force, ball_pos, ball_vel, ball_force, goal_pos, teammate_poses, teammate_forces, teammate_vels, adversary_poses, adversary_forces, adversary_vels = input
        if not blue:
            for tensor in [agent_pos, agent_vel, agent_force, ball_pos, ball_vel, ball_force, goal_pos] + teammate_poses + teammate_forces + teammate_vels + adversary_poses + adversary_forces + adversary_vels:
                tensor[..., X] = -tensor[..., X]
            agent_rot = agent_rot - torch.pi
        obs = {'obs': [agent_force, agent_pos - ball_pos, agent_vel - ball_vel, ball_pos - goal_pos, ball_vel, ball_force], 'pos': [agent_pos - goal_pos], 'vel': [agent_vel]}
        if self.enable_shooting:
            obs['obs'].append(agent_rot)
        if self.observe_adversaries and len(adversary_poses):
            obs['adversaries'] = []
            for adversary_pos, adversary_force, adversary_vel in zip(adversary_poses, adversary_forces, adversary_vels):
                obs['adversaries'].append(torch.cat([agent_pos - adversary_pos, agent_vel - adversary_vel, adversary_vel, adversary_force], dim=-1))
            obs['adversaries'] = [torch.stack(obs['adversaries'], dim=-2) if self.dict_obs else torch.cat(obs['adversaries'], dim=-1)]
        if self.observe_teammates:
            obs['teammates'] = []
            for teammate_pos, teammate_force, teammate_vel in zip(teammate_poses, teammate_forces, teammate_vels):
                obs['teammates'].append(torch.cat([agent_pos - teammate_pos, agent_vel - teammate_vel, teammate_vel, teammate_force], dim=-1))
            obs['teammates'] = [torch.stack(obs['teammates'], dim=-2) if self.dict_obs else torch.cat(obs['teammates'], dim=-1)]
        for key, value in obs.items():
            obs[key] = torch.cat(value, dim=-1)
        if self.dict_obs:
            return obs
        else:
            return torch.cat(list(obs.values()), dim=-1)

    def done(self):
        if self.ai_blue_agents and self.ai_red_agents:
            self.reward(None)
        return self._done

    def _compute_coverage(self, blue: bool, env_index=None):
        team = self.blue_agents if blue else self.red_agents
        pos = torch.stack([a.state.pos for a in team], dim=-2)
        avg_point = pos.mean(-2).unsqueeze(-2)
        if isinstance(env_index, int):
            pos = pos[env_index].unsqueeze(0)
            avg_point = avg_point[env_index].unsqueeze(0)
        dist = torch.cdist(pos, avg_point)
        dist = dist.squeeze(-1)
        max_dist = dist.max(dim=-1)[0]
        if isinstance(env_index, int):
            max_dist = max_dist.squeeze(0)
        return max_dist

    def info(self, agent: Agent):
        blue = agent in self.blue_agents
        info = {'sparse_reward': self._sparse_reward_blue if blue else self._sparse_reward_red, 'ball_goal_pos_rew': self.ball.pos_rew_blue if blue else self.ball.pos_rew_red, 'all_agent_ball_pos_rew': self.ball.pos_rew_agent_blue if blue else self.ball.pos_rew_agent_red, 'ball_pos': self.ball.state.pos, 'dist_ball_to_goal': (self.ball.pos_shaping_blue if blue else self.ball.pos_shaping_red) / self.pos_shaping_factor_ball_goal}
        if blue and self.min_agent_dist_to_ball_blue is not None:
            info['min_agent_dist_to_ball'] = self.min_agent_dist_to_ball_blue
            info['touching_ball'] = self.min_agent_dist_to_ball_blue <= self.agent_size + self.ball_size + 0.01
        elif not blue and self.min_agent_dist_to_ball_red is not None:
            info['min_agent_dist_to_ball'] = self.min_agent_dist_to_ball_red
            info['touching_ball'] = self.min_agent_dist_to_ball_red <= self.agent_size + self.ball_size + 0.01
        return info

    def extra_render(self, env_index: int=0) -> 'List[Geom]':
        from vmas.simulator import rendering
        from vmas.simulator.rendering import Geom
        geoms: List[Geom] = self._get_background_geoms(self.background_entities) if self._render_field else self._get_background_geoms(self.background_entities[3:])
        geoms += ScenarioUtils.render_agent_indices(self, env_index, start_from=1, exclude=self.red_agents + [self.ball])
        if self.enable_shooting:
            for agent in self.blue_agents:
                color = agent.color
                if agent.ball_within_angle[env_index] and agent.ball_within_range[env_index]:
                    color = Color.PINK.value
                sector = rendering.make_circle(radius=self.shooting_radius, angle=self.shooting_angle, filled=True)
                xform = rendering.Transform()
                xform.set_rotation(agent.state.rot[env_index])
                xform.set_translation(*agent.state.pos[env_index])
                sector.add_attr(xform)
                sector.set_color(*color, alpha=agent._alpha / 2)
                geoms.append(sector)
                shoot_intensity = torch.linalg.vector_norm(agent.shoot_force[env_index]) / (self.u_shoot_multiplier * 2)
                l, r, t, b = (0, self.shooting_radius * shoot_intensity, self.agent_size / 2, -self.agent_size / 2)
                line = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
                xform = rendering.Transform()
                xform.set_rotation(agent.state.rot[env_index])
                xform.set_translation(*agent.state.pos[env_index])
                line.add_attr(xform)
                line.set_color(*color, alpha=agent._alpha)
                geoms.append(line)
        return geoms

    def _get_background_geoms(self, objects):

        def _get_geom(entity, pos, rot=0.0):
            from vmas.simulator import rendering
            geom = entity.shape.get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(*pos)
            xform.set_rotation(rot)
            color = entity.color
            geom.set_color(*color)
            return geom
        geoms = []
        for landmark in objects:
            if landmark.name == 'Centre Line':
                geoms.append(_get_geom(landmark, [0.0, 0.0], torch.pi / 2))
            elif landmark.name == 'Right Line':
                geoms.append(_get_geom(landmark, [self.pitch_length / 2 - self.agent_size, 0.0], torch.pi / 2))
            elif landmark.name == 'Left Line':
                geoms.append(_get_geom(landmark, [-self.pitch_length / 2 + self.agent_size, 0.0], torch.pi / 2))
            elif landmark.name == 'Top Line':
                geoms.append(_get_geom(landmark, [0.0, self.pitch_width / 2 - self.agent_size]))
            elif landmark.name == 'Bottom Line':
                geoms.append(_get_geom(landmark, [0.0, -self.pitch_width / 2 + self.agent_size]))
            else:
                geoms.append(_get_geom(landmark, [0, 0]))
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_passages = kwargs.pop('n_passages', 1)
        self.fixed_passage = kwargs.pop('fixed_passage', True)
        self.joint_length = kwargs.pop('joint_length', 0.5)
        self.random_start_angle = kwargs.pop('random_start_angle', True)
        self.random_goal_angle = kwargs.pop('random_goal_angle', True)
        self.observe_joint_angle = kwargs.pop('observe_joint_angle', False)
        self.joint_angle_obs_noise = kwargs.pop('joint_angle_obs_noise', 0.0)
        self.asym_package = kwargs.pop('asym_package', True)
        self.mass_ratio = kwargs.pop('mass_ratio', 5)
        self.mass_position = kwargs.pop('mass_position', 0.75)
        self.max_speed_1 = kwargs.pop('max_speed_1', None)
        self.pos_shaping_factor = kwargs.pop('pos_shaping_factor', 1)
        self.rot_shaping_factor = kwargs.pop('rot_shaping_factor', 1)
        self.collision_reward = kwargs.pop('collision_reward', 0)
        self.energy_reward_coeff = kwargs.pop('energy_reward_coeff', 0)
        self.all_passed_rot = kwargs.pop('all_passed_rot', True)
        self.obs_noise = kwargs.pop('obs_noise', 0.0)
        self.use_controller = kwargs.pop('use_controller', False)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.plot_grid = True
        self.visualize_semidims = False
        world = World(batch_dim, device, x_semidim=1, y_semidim=1, substeps=7 if not self.asym_package else 10, joint_force=900 if self.asym_package else 400, collision_force=2500 if self.asym_package else 1500, drag=0.25 if not self.asym_package else 0.15)
        if not self.observe_joint_angle:
            assert self.joint_angle_obs_noise == 0
        self.middle_angle = torch.pi / 2
        self.n_agents = 2
        self.agent_radius = 0.03333
        self.mass_radius = self.agent_radius * (2 / 3)
        self.passage_width = 0.2
        self.passage_length = 0.1476
        self.scenario_length = 2 * world.x_semidim + 2 * self.agent_radius
        self.n_boxes = int(self.scenario_length // self.passage_length)
        self.min_collision_distance = 0.005
        assert 1 <= self.n_passages <= int(self.scenario_length // self.passage_length)
        cotnroller_params = [2.0, 10, 1e-05]
        agent = Agent(name='agent_0', shape=Sphere(self.agent_radius), obs_noise=self.obs_noise, render_action=True, u_multiplier=0.8, f_range=0.8)
        agent.controller = VelocityController(agent, world, cotnroller_params, 'standard')
        world.add_agent(agent)
        agent = Agent(name='agent_1', shape=Sphere(self.agent_radius), mass=1 if self.asym_package else self.mass_ratio, color=Color.BLUE, max_speed=self.max_speed_1, obs_noise=self.obs_noise, render_action=True, u_multiplier=0.8, f_range=0.8)
        agent.controller = VelocityController(agent, world, cotnroller_params, 'standard')
        world.add_agent(agent)
        self.joint = Joint(world.agents[0], world.agents[1], anchor_a=(0, 0), anchor_b=(0, 0), dist=self.joint_length, rotate_a=True, rotate_b=True, collidable=True, width=0, mass=1)
        world.add_joint(self.joint)
        if self.asym_package:

            def mass_collision_filter(e):
                return not isinstance(e.shape, Sphere)
            self.mass = Landmark(name='mass', shape=Sphere(radius=self.mass_radius), collide=True, movable=True, color=Color.BLACK, mass=self.mass_ratio, collision_filter=mass_collision_filter)
            world.add_landmark(self.mass)
            joint = Joint(self.mass, self.joint.landmark, anchor_a=(0, 0), anchor_b=(self.mass_position, 0), dist=0, rotate_a=True, rotate_b=True)
            world.add_joint(joint)
        self.goal = Landmark(name='joint_goal', shape=Line(length=self.joint_length), collide=False, color=Color.GREEN)
        world.add_landmark(self.goal)
        self.walls = []
        for i in range(4):
            wall = Landmark(name=f'wall {i}', collide=True, shape=Line(length=2 + self.agent_radius * 2), color=Color.BLACK)
            world.add_landmark(wall)
            self.walls.append(wall)
        self.create_passage_map(world)
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.rot_rew = self.pos_rew.clone()
        self.collision_rew = self.pos_rew.clone()
        self.energy_rew = self.pos_rew.clone()
        self.all_passed = torch.full((batch_dim,), False, device=device)
        return world

    def reset_world_at(self, env_index: int=None):
        start_angle = torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-torch.pi / 2 if self.random_start_angle else 0, torch.pi / 2 if self.random_start_angle else 0)
        goal_angle = torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-torch.pi / 2 if self.random_goal_angle else 0, torch.pi / 2 if self.random_goal_angle else 0)
        start_delta_x = self.joint_length / 2 * torch.cos(start_angle)
        start_delta_x_abs = start_delta_x.abs()
        min_x_start = -self.world.x_semidim + (self.agent_radius + start_delta_x_abs)
        max_x_start = self.world.x_semidim - (self.agent_radius + start_delta_x_abs)
        start_delta_y = self.joint_length / 2 * torch.sin(start_angle)
        start_delta_y_abs = start_delta_y.abs()
        min_y_start = -self.world.y_semidim + (self.agent_radius + start_delta_y_abs)
        max_y_start = -2 * self.agent_radius - self.passage_width / 2 - start_delta_y_abs
        goal_delta_x = self.joint_length / 2 * torch.cos(goal_angle)
        goal_delta_x_abs = goal_delta_x.abs()
        min_x_goal = -self.world.x_semidim + (self.agent_radius + goal_delta_x_abs)
        max_x_goal = self.world.x_semidim - (self.agent_radius + goal_delta_x_abs)
        goal_delta_y = self.joint_length / 2 * torch.sin(goal_angle)
        goal_delta_y_abs = goal_delta_y.abs()
        min_y_goal = 2 * self.agent_radius + self.passage_width / 2 + goal_delta_y_abs
        max_y_goal = self.world.y_semidim - (self.agent_radius + goal_delta_y_abs)
        joint_pos = torch.cat([(min_x_start - max_x_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_x_start, (min_y_start - max_y_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_y_start], dim=1)
        goal_pos = torch.cat([(min_x_goal - max_x_goal) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_x_goal, (min_y_goal - max_y_goal) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_y_goal], dim=1)
        self.goal.set_pos(goal_pos, batch_index=env_index)
        self.goal.set_rot(goal_angle, batch_index=env_index)
        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        for i, agent in enumerate(agents):
            agent.controller.reset(env_index)
            if i == 0:
                agent.set_pos(joint_pos - torch.cat([start_delta_x, start_delta_y], dim=1), batch_index=env_index)
            else:
                agent.set_pos(joint_pos + torch.cat([start_delta_x, start_delta_y], dim=1), batch_index=env_index)
        if self.asym_package:
            self.mass.set_pos(joint_pos + self.mass_position * torch.cat([start_delta_x, start_delta_y], dim=1) * (1 if agents[0] == self.world.agents[0] else -1), batch_index=env_index)
        self.spawn_passage_map(env_index)
        self.spawn_walls(env_index)
        if env_index is None:
            self.passed = torch.zeros((self.world.batch_dim,), device=self.world.device)
            self.joint.pos_shaping_pre = torch.stack([torch.linalg.vector_norm(self.joint.landmark.state.pos - p.state.pos, dim=1) for p in self.passages if not p.collide], dim=1).min(dim=1)[0] * self.pos_shaping_factor
            self.joint.pos_shaping_post = torch.linalg.vector_norm(self.joint.landmark.state.pos - self.goal.state.pos, dim=1) * self.pos_shaping_factor
            self.joint.rot_shaping_pre = get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.middle_angle) * self.rot_shaping_factor
            self.joint.rot_shaping_post = get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.goal.state.rot) * self.rot_shaping_factor
        else:
            self.passed[env_index] = 0
            self.joint.pos_shaping_pre[env_index] = torch.stack([torch.linalg.vector_norm(self.joint.landmark.state.pos[env_index] - p.state.pos[env_index]).unsqueeze(-1) for p in self.passages if not p.collide], dim=1).min(dim=1)[0] * self.pos_shaping_factor
            self.joint.pos_shaping_post[env_index] = torch.linalg.vector_norm(self.joint.landmark.state.pos[env_index] - self.goal.state.pos[env_index]) * self.pos_shaping_factor
            self.joint.rot_shaping_pre[env_index] = get_line_angle_dist_0_180(self.joint.landmark.state.rot[env_index], self.middle_angle) * self.rot_shaping_factor
            self.joint.rot_shaping_post[env_index] = get_line_angle_dist_0_180(self.joint.landmark.state.rot[env_index], self.goal.state.rot[env_index]) * self.rot_shaping_factor

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            self.pos_rew[:] = 0
            self.rot_rew[:] = 0
            self.collision_rew[:] = 0
            joint_passed = self.joint.landmark.state.pos[:, Y] > 0
            self.all_passed = (torch.stack([a.state.pos[:, Y] for a in self.world.agents], dim=1) > self.passage_width / 2).all(dim=1)
            joint_dist_to_closest_pass = torch.stack([torch.linalg.vector_norm(self.joint.landmark.state.pos - p.state.pos, dim=1) for p in self.passages if not p.collide], dim=1).min(dim=1)[0]
            joint_shaping = joint_dist_to_closest_pass * self.pos_shaping_factor
            self.pos_rew[~joint_passed] += (self.joint.pos_shaping_pre - joint_shaping)[~joint_passed]
            self.joint.pos_shaping_pre = joint_shaping
            joint_dist_to_goal = torch.linalg.vector_norm(self.joint.landmark.state.pos - self.goal.state.pos, dim=1)
            joint_shaping = joint_dist_to_goal * self.pos_shaping_factor
            self.pos_rew[joint_passed] += (self.joint.pos_shaping_post - joint_shaping)[joint_passed]
            self.joint.pos_shaping_post = joint_shaping
            rot_passed = self.all_passed if self.all_passed_rot else joint_passed
            joint_dist_to_90_rot = get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.middle_angle)
            joint_shaping = joint_dist_to_90_rot * self.rot_shaping_factor
            self.rot_rew[~rot_passed] += (self.joint.rot_shaping_pre - joint_shaping)[~rot_passed]
            self.joint.rot_shaping_pre = joint_shaping
            joint_dist_to_goal_rot = get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.goal.state.rot)
            joint_shaping = joint_dist_to_goal_rot * self.rot_shaping_factor
            self.rot_rew[rot_passed] += (self.joint.rot_shaping_post - joint_shaping)[rot_passed]
            self.joint.rot_shaping_post = joint_shaping
            for a in self.world.agents + ([self.mass] if self.asym_package else []):
                for passage in self.passages:
                    if passage.collide:
                        self.collision_rew[self.world.get_distance(a, passage) <= self.min_collision_distance] += self.collision_reward
                    for wall in self.walls:
                        self.collision_rew[self.world.get_distance(a, wall) <= self.min_collision_distance] += self.collision_reward
            for p in self.passages:
                if p.collide:
                    self.collision_rew[self.world.get_distance(p, self.joint.landmark) <= self.min_collision_distance] += self.collision_reward
            self.energy_expenditure = torch.stack([torch.linalg.vector_norm(a.action.u, dim=-1) / math.sqrt(self.world.dim_p * a.f_range ** 2) for a in self.world.agents], dim=1).sum(-1)
            self.energy_rew = -self.energy_expenditure * self.energy_reward_coeff
            self.rew = self.pos_rew + self.rot_rew + self.collision_rew + self.energy_rew
        return self.rew

    def is_out_or_touching_perimeter(self, agent: Agent):
        is_out_or_touching_perimeter = torch.full((self.world.batch_dim,), False, device=self.world.device)
        is_out_or_touching_perimeter += agent.state.pos[:, X] >= self.world.x_semidim
        is_out_or_touching_perimeter += agent.state.pos[:, X] <= -self.world.x_semidim
        is_out_or_touching_perimeter += agent.state.pos[:, Y] >= self.world.y_semidim
        is_out_or_touching_perimeter += agent.state.pos[:, Y] <= -self.world.y_semidim
        return is_out_or_touching_perimeter

    def observation(self, agent: Agent):
        if self.observe_joint_angle:
            joint_angle = self.joint.landmark.state.rot
            angle_noise = torch.randn(*joint_angle.shape, device=self.world.device, dtype=torch.float32) * self.joint_angle_obs_noise if self.joint_angle_obs_noise else 0.0
            joint_angle += angle_noise
        passage_obs = []
        for passage in self.passages:
            if not passage.collide:
                passage_obs.append(agent.state.pos - passage.state.pos)
        observations = [agent.state.pos, agent.state.vel, agent.state.pos - self.goal.state.pos, *passage_obs, angle_to_vector(self.goal.state.rot)] + ([angle_to_vector(joint_angle)] if self.observe_joint_angle else [])
        for i, obs in enumerate(observations):
            noise = torch.zeros(*obs.shape, device=self.world.device).uniform_(-self.obs_noise, self.obs_noise)
            observations[i] = obs + noise
        return torch.cat(observations, dim=-1)

    def done(self):
        return torch.all((torch.linalg.vector_norm(self.joint.landmark.state.pos - self.goal.state.pos, dim=1) <= 0.01) * (get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.goal.state.rot).unsqueeze(-1) <= 0.01), dim=1)

    def process_action(self, agent: Agent):
        if self.use_controller:
            vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 0.001
            agent.controller.reset(vel_is_zero)
            agent.controller.process_force()

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        is_first = self.world.agents[0] == agent
        if is_first:
            just_passed = self.all_passed * (self.passed == 0)
            self.passed[just_passed] = 100
            self.info_stored = {'pos_rew': self.pos_rew, 'rot_rew': self.rot_rew, 'collision_rew': self.collision_rew, 'energy_rew': self.energy_rew, 'passed': just_passed.to(torch.int)}
        return self.info_stored

    def create_passage_map(self, world: World):
        self.passages = []
        self.collide_passages = []
        self.non_collide_passages = []

        def removed(i):
            return self.n_boxes // 2 - self.n_passages / 2 <= i < self.n_boxes // 2 + self.n_passages / 2
        for i in range(self.n_boxes):
            passage = Landmark(name=f'passage {i}', collide=not removed(i), movable=False, shape=Box(length=self.passage_length, width=self.passage_width), color=Color.RED, collision_filter=lambda e: not isinstance(e.shape, Box))
            if not passage.collide:
                self.non_collide_passages.append(passage)
            else:
                self.collide_passages.append(passage)
            self.passages.append(passage)
            world.add_landmark(passage)

        def joint_collides(e):
            if e in self.collide_passages and self.fixed_passage:
                return e.neighbour
            elif e in self.collide_passages:
                return True
            return False
        self.joint.landmark.collision_filter = joint_collides

    def spawn_passage_map(self, env_index):
        passage_indexes = []
        j = self.n_boxes // 2
        for i in range(self.n_passages):
            if self.fixed_passage:
                j += i * (-1 if i % 2 == 0 else 1)
                passage_index = torch.full((self.world.batch_dim,) if env_index is None else (1,), j, device=self.world.device)
            else:
                passage_index = torch.randint(0, self.n_boxes - 1, (self.world.batch_dim,) if env_index is None else (1,), device=self.world.device)
            passage_indexes.append(passage_index)

        def is_passage(i):
            is_pass = torch.full(i.shape, False, device=self.world.device)
            for index in passage_indexes:
                is_pass += i == index
            return is_pass

        def get_pos(i):
            pos = torch.tensor([-1 - self.agent_radius + self.passage_length / 2, 0.0], dtype=torch.float32, device=self.world.device).repeat(i.shape[0], 1)
            pos[:, X] += self.passage_length * i
            return pos
        for index, i in enumerate(passage_indexes):
            self.non_collide_passages[index].is_rendering[:] = False
            self.non_collide_passages[index].set_pos(get_pos(i), batch_index=env_index)
        i = torch.zeros((self.world.batch_dim,) if env_index is None else (1,), dtype=torch.int, device=self.world.device)
        for passage in self.collide_passages:
            is_pass = is_passage(i)
            while is_pass.any():
                i[is_pass] += 1
                is_pass = is_passage(i)
            passage.set_pos(get_pos(i), batch_index=env_index)
            if self.fixed_passage:
                passage.neighbour = (is_passage(i - 1) + is_passage(i + 1)).all()
            elif env_index is None:
                passage.neighbour = is_passage(i - 1) + is_passage(i + 1)
            else:
                passage.neighbour[env_index] = is_passage(i - 1) + is_passage(i + 1)
            i += 1

    def spawn_walls(self, env_index):
        for i, wall in enumerate(self.walls):
            wall.set_pos(torch.tensor([0.0 if i % 2 else self.world.x_semidim + self.agent_radius if i == 0 else -self.world.x_semidim - self.agent_radius, 0.0 if not i % 2 else self.world.y_semidim + self.agent_radius if i == 1 else -self.world.y_semidim - self.agent_radius], device=self.world.device), batch_index=env_index)
            wall.set_rot(torch.tensor([torch.pi / 2 if not i % 2 else 0.0], device=self.world.device), batch_index=env_index)

    def extra_render(self, env_index: int=0):
        from vmas.simulator import rendering
        geoms = []
        color = self.goal.color
        if isinstance(color, torch.Tensor) and len(color.shape) > 1:
            color = color[env_index]
        goal_agent_1 = rendering.make_circle(self.agent_radius)
        xform = rendering.Transform()
        goal_agent_1.add_attr(xform)
        xform.set_translation(self.goal.state.pos[env_index][X] - self.joint_length / 2 * math.cos(self.goal.state.rot[env_index]), self.goal.state.pos[env_index][Y] - self.joint_length / 2 * math.sin(self.goal.state.rot[env_index]))
        goal_agent_1.set_color(*color)
        geoms.append(goal_agent_1)
        goal_agent_2 = rendering.make_circle(self.agent_radius)
        xform = rendering.Transform()
        goal_agent_2.add_attr(xform)
        xform.set_translation(self.goal.state.pos[env_index][X] + self.joint_length / 2 * math.cos(self.goal.state.rot[env_index]), self.goal.state.pos[env_index][Y] + self.joint_length / 2 * math.sin(self.goal.state.rot[env_index]))
        goal_agent_2.set_color(*color)
        geoms.append(goal_agent_2)
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True
        self.viewer_zoom = 2
        self.vel_shaping_factor = kwargs.pop('vel_shaping_factor', 1)
        self.dist_shaping_factor = kwargs.pop('dist_shaping_factor', 1)
        self.wind_shaping_factor = kwargs.pop('wind_shaping_factor', 1)
        self.pos_shaping_factor = kwargs.pop('pos_shaping_factor', 0)
        self.rot_shaping_factor = kwargs.pop('rot_shaping_factor', 0)
        self.energy_shaping_factor = kwargs.pop('energy_shaping_factor', 0)
        self.observe_rel_pos = kwargs.pop('observe_rel_pos', False)
        self.observe_rel_vel = kwargs.pop('observe_rel_vel', False)
        self.observe_pos = kwargs.pop('observe_pos', True)
        self.use_controller = kwargs.pop('use_controller', True)
        self.wind = torch.tensor([0, -kwargs.pop('wind', 2)], device=device, dtype=torch.float32).expand(batch_dim, 2)
        self.v_range = kwargs.pop('v_range', 0.5)
        self.desired_vel = kwargs.pop('desired_vel', self.v_range)
        self.f_range = kwargs.pop('f_range', 100)
        controller_params = [1.5, 0.6, 0.002]
        self.u_range = self.v_range if self.use_controller else self.f_range
        self.cover_angle_tolerance = kwargs.pop('cover_angle_tolerance', 1)
        self.horizon = kwargs.pop('horizon', 200)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.desired_distance = 1
        self.grid_spacing = self.desired_distance
        world = World(batch_dim, device, drag=0, linear_friction=0.1)
        self.desired_vel = torch.tensor([0.0, self.desired_vel], device=device, dtype=torch.float32)
        self.max_pos = self.horizon * world.dt * self.desired_vel[Y]
        self.desired_pos = 10.0
        self.n_agents = 2
        self.big_agent = Agent(name='agent_0', render_action=True, shape=Sphere(radius=0.05), u_range=self.u_range, v_range=self.v_range, f_range=self.f_range, gravity=self.wind)
        self.big_agent.controller = VelocityController(self.big_agent, world, controller_params, 'standard')
        world.add_agent(self.big_agent)
        self.small_agent = Agent(name='agent_1', render_action=True, shape=Sphere(radius=0.03), u_range=self.u_range, v_range=self.v_range, f_range=self.f_range, gravity=self.wind)
        self.small_agent.controller = VelocityController(self.small_agent, world, controller_params, 'standard')
        world.add_agent(self.small_agent)
        for agent in world.agents:
            agent.wind_rew = torch.zeros(batch_dim, device=device)
            agent.vel_rew = agent.wind_rew.clone()
            agent.energy_rew = agent.wind_rew.clone()
        self.dist_rew = torch.zeros(batch_dim, device=device)
        self.rot_rew = self.dist_rew.clone()
        self.vel_reward = self.dist_rew.clone()
        self.pos_rew = self.dist_rew.clone()
        self.t = self.dist_rew.clone()
        return world

    def set_wind(self, wind):
        self.wind = torch.tensor([0, -wind], device=self.world.device, dtype=torch.float32).expand(self.world.batch_dim, self.world.dim_p)
        self.big_agent.gravity = self.wind
        self.small_agent.gravity = self.wind

    def reset_world_at(self, env_index: int=None):
        start_angle = torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-torch.pi / 8, torch.pi / 8)
        start_delta_x = self.desired_distance / 2 * torch.cos(start_angle)
        start_delta_y = self.desired_distance / 2 * torch.sin(start_angle)
        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        for i, agent in enumerate(agents):
            agent.controller.reset(env_index)
            if i == 0:
                agent.set_pos(-torch.cat([start_delta_x, start_delta_y], dim=1), batch_index=env_index)
            else:
                agent.set_pos(torch.cat([start_delta_x, start_delta_y], dim=1), batch_index=env_index)
            if env_index is None:
                agent.vel_shaping = torch.linalg.vector_norm(agent.state.vel - self.desired_vel, dim=-1) * self.vel_shaping_factor
                agent.energy_shaping = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
                agent.wind_shaping = torch.linalg.vector_norm(agent.gravity, dim=-1) * self.wind_shaping_factor
            else:
                agent.vel_shaping[env_index] = torch.linalg.vector_norm(agent.state.vel[env_index] - self.desired_vel) * self.vel_shaping_factor
                agent.energy_shaping[env_index] = 0
                agent.wind_shaping[env_index] = torch.linalg.vector_norm(agent.gravity[env_index]) * self.wind_shaping_factor
        if env_index is None:
            self.t = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.int)
            self.distance_shaping = (torch.linalg.vector_norm(self.small_agent.state.pos - self.big_agent.state.pos, dim=-1) - self.desired_distance).abs() * self.dist_shaping_factor
            self.pos_shaping = (torch.maximum(self.big_agent.state.pos[:, Y], self.small_agent.state.pos[:, Y]) - self.desired_pos).abs() * self.pos_shaping_factor
            self.rot_shaping = get_line_angle_dist_0_180(self.get_agents_angle(), 0) * self.rot_shaping_factor
        else:
            self.t[env_index] = 0
            self.distance_shaping[env_index] = (torch.linalg.vector_norm(self.small_agent.state.pos[env_index] - self.big_agent.state.pos[env_index]) - self.desired_distance).abs() * self.dist_shaping_factor
            self.pos_shaping[env_index] = (torch.maximum(self.big_agent.state.pos[env_index, Y], self.small_agent.state.pos[env_index, Y]) - self.desired_pos).abs() * self.pos_shaping_factor
            self.rot_shaping[env_index] = get_line_angle_dist_0_180(self.get_agents_angle()[env_index], 0) * self.rot_shaping_factor

    def process_action(self, agent: Agent):
        if self.use_controller:
            agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.t += 1
            self.set_friction()
            distance_shaping = (torch.linalg.vector_norm(self.small_agent.state.pos - self.big_agent.state.pos, dim=-1) - self.desired_distance).abs() * self.dist_shaping_factor
            self.dist_rew = self.distance_shaping - distance_shaping
            self.distance_shaping = distance_shaping
            rot_shaping = get_line_angle_dist_0_180(self.get_agents_angle(), 0) * self.rot_shaping_factor
            self.rot_rew = self.rot_shaping - rot_shaping
            self.rot_shaping = rot_shaping
            pos_shaping = (torch.maximum(self.big_agent.state.pos[:, Y], self.small_agent.state.pos[:, Y]) - self.desired_pos).abs() * self.pos_shaping_factor
            self.pos_rew = self.pos_shaping - pos_shaping
            self.pos_shaping = pos_shaping
            for a in self.world.agents:
                vel_shaping = torch.linalg.vector_norm(a.state.vel - self.desired_vel, dim=-1) * self.vel_shaping_factor
                a.vel_rew = a.vel_shaping - vel_shaping
                a.vel_shaping = vel_shaping
            self.vel_reward = torch.stack([a.vel_rew for a in self.world.agents], dim=1).mean(-1)
            for a in self.world.agents:
                energy_shaping = torch.linalg.vector_norm(a.action.u, dim=-1) * self.energy_shaping_factor
                a.energy_rew = a.energy_shaping - energy_shaping
                a.energy_rew[self.t < 10] = 0
                a.energy_shaping = energy_shaping
            self.energy_rew = torch.stack([a.energy_rew for a in self.world.agents], dim=1).mean(-1)
            for a in self.world.agents:
                wind_shaping = torch.linalg.vector_norm(a.gravity, dim=-1) * self.wind_shaping_factor
                a.wind_rew = a.wind_shaping - wind_shaping
                a.wind_rew[self.t < 5] = 0
                a.wind_shaping = wind_shaping
            self.wind_rew = torch.stack([a.wind_rew for a in self.world.agents], dim=1).mean(-1)
        return self.dist_rew + self.vel_reward + self.rot_rew + self.energy_rew + self.wind_rew + self.pos_rew

    def set_friction(self):
        dist_to_goal_angle = (get_line_angle_dist_0_360(self.get_agents_angle(), torch.tensor([-torch.pi / 2], device=self.world.device).expand(self.world.batch_dim, 1)) + 1).clamp(max=self.cover_angle_tolerance).unsqueeze(-1) + (1 - self.cover_angle_tolerance)
        dist_to_goal_angle = (dist_to_goal_angle - 1 + self.cover_angle_tolerance) / self.cover_angle_tolerance
        self.big_agent.gravity = self.wind * dist_to_goal_angle

    def observation(self, agent: Agent):
        observations = []
        if self.observe_pos:
            observations.append(agent.state.pos)
        observations.append(agent.state.vel)
        if self.observe_rel_pos:
            for a in self.world.agents:
                if a != agent:
                    observations.append(a.state.pos - agent.state.pos)
        if self.observe_rel_vel:
            for a in self.world.agents:
                if a != agent:
                    observations.append(a.state.vel - agent.state.vel)
        return torch.cat(observations, dim=-1)

    def get_agents_angle(self):
        return torch.atan2(self.big_agent.state.pos[:, Y] - self.small_agent.state.pos[:, Y], self.big_agent.state.pos[:, X] - self.small_agent.state.pos[:, X]).unsqueeze(-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'dist_rew': self.dist_rew, 'rot_rew': self.rot_rew, 'pos_rew': self.pos_rew, 'agent_wind_rew': agent.wind_rew, 'agent_vel_rew': agent.vel_rew, 'agent_energy_rew': agent.energy_rew, 'delta_vel_to_goal': torch.linalg.vector_norm(agent.state.vel - self.desired_vel, dim=-1)}

    def extra_render(self, env_index: int=0) -> 'List[Geom]':
        from vmas.simulator import rendering
        geoms = []
        color = Color.BLACK.value
        circle = rendering.Line((-self.desired_distance / 2, 0), (self.desired_distance / 2, 0), width=1)
        xform = rendering.Transform()
        xform.set_translation(*(self.big_agent.state.pos[env_index] + self.small_agent.state.pos[env_index]) / 2)
        xform.set_rotation(self.get_agents_angle()[env_index])
        circle.add_attr(xform)
        circle.set_color(*color)
        geoms.append(circle)
        color = Color.RED.value
        circle = rendering.Line((-self.desired_distance / 2, 0), (self.desired_distance / 2, 0), width=1)
        xform = rendering.Transform()
        xform.set_translation(0.0, self.max_pos)
        circle.add_attr(xform)
        circle.set_color(*color)
        geoms.append(circle)
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.obs_noise = kwargs.pop('obs_noise', 0)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.agent_radius = 0.03
        self.line_length = 3
        world = World(batch_dim, device, drag=0.1)
        self.agent = Agent(name='agent_0', shape=Sphere(self.agent_radius), mass=2, f_range=0.5, u_range=1, render_action=True)
        self.agent.controller = VelocityController(self.agent, world, [4, 1.25, 0.001], 'standard')
        world.add_agent(self.agent)
        self.tangent = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        self.tangent[:, Y] = 1
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.dot_product = self.pos_rew.clone()
        self.steady_rew = self.pos_rew.clone()
        return world

    def process_action(self, agent: Agent):
        self.vel_action = agent.action.u.clone()
        agent.controller.process_force()

    def reset_world_at(self, env_index: int=None):
        self.agent.controller.reset(env_index)
        self.agent.set_pos(torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-1, 1), torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-1, 0)], dim=1), batch_index=env_index)

    def reward(self, agent: Agent):
        closest_point = agent.state.pos.clone()
        closest_point[:, X] = 0
        self.pos_rew = -torch.linalg.vector_norm(agent.state.pos - closest_point, dim=1) ** 0.5 * 1
        self.dot_product = torch.einsum('bs,bs->b', self.tangent, agent.state.vel) * 0.5
        normalized_vel = agent.state.vel / torch.linalg.vector_norm(agent.state.vel, dim=1).unsqueeze(-1)
        normalized_vel = torch.nan_to_num(normalized_vel)
        normalized_vel_action = self.vel_action / torch.linalg.vector_norm(self.vel_action, dim=1).unsqueeze(-1)
        normalized_vel_action = torch.nan_to_num(normalized_vel_action)
        self.steady_rew = torch.einsum('bs,bs->b', normalized_vel, normalized_vel_action) * 0.2
        return self.pos_rew + self.dot_product + self.steady_rew

    def observation(self, agent: Agent):
        observations = [agent.state.pos, agent.state.vel, agent.state.pos]
        for i, obs in enumerate(observations):
            noise = torch.zeros(*obs.shape, device=self.world.device).uniform_(-self.obs_noise, self.obs_noise)
            observations[i] = obs + noise
        return torch.cat(observations, dim=-1)

    def done(self):
        return self.world.agents[0].state.pos[:, Y] > self.line_length - 1

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'pos_rew': self.pos_rew, 'dot_product': self.dot_product, 'steady_rew': self.steady_rew}

    def extra_render(self, env_index: int=0):
        from vmas.simulator import rendering
        geoms = []
        color = Color.BLACK.value
        line = rendering.Line((0, -1), (0, -1 + self.line_length), width=1)
        line.set_color(*color)
        geoms.append(line)
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.u_range = kwargs.pop('u_range', 1)
        self.a_range = kwargs.pop('a_range', 1)
        self.obs_noise = kwargs.pop('obs_noise', 0.0)
        self.dt_delay = kwargs.pop('dt_delay', 0)
        self.min_input_norm = kwargs.pop('min_input_norm', 0.08)
        self.linear_friction = kwargs.pop('linear_friction', 0.1)
        self.pos_shaping_factor = kwargs.pop('pos_shaping_factor', 1.0)
        self.time_rew_coeff = kwargs.pop('time_rew_coeff', -0.01)
        self.energy_reward_coeff = kwargs.pop('energy_rew_coeff', 0.0)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.viewer_size = (1600, 700)
        self.viewer_zoom = 2
        self.plot_grid = True
        self.agent_radius = 0.16
        self.lab_length = 6
        self.lab_width = 3
        controller_params = [2, 6, 0.002]
        self.f_range = self.a_range + self.linear_friction
        world = World(batch_dim, device, drag=0, dt=0.05, substeps=5)
        null_action = torch.zeros(world.batch_dim, world.dim_p, device=world.device)
        self.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        self.goal = Landmark('goal', collide=False, movable=False, shape=Sphere(radius=0.06))
        world.add_landmark(self.goal)
        agent = Agent(name='agent 0', collide=True, color=Color.GREEN, render_action=True, linear_friction=self.linear_friction, shape=Sphere(radius=self.agent_radius), f_range=self.f_range, u_range=self.u_range)
        agent.controller = VelocityController(agent, world, controller_params, 'standard')
        agent.goal = self.goal
        agent.energy_rew = torch.zeros(batch_dim, device=device)
        world.add_agent(agent)
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.time_rew = self.pos_rew.clone()
        return world

    def reset_world_at(self, env_index: int=None):
        for agent in self.world.agents:
            agent.controller.reset(env_index)
            agent.set_pos(torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.lab_length / 2, self.lab_length / 2), torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.lab_width / 2, self.lab_width / 2)], dim=1), batch_index=env_index)
        for landmark in self.world.landmarks:
            landmark.set_pos(torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.lab_length / 2, self.lab_length / 2), torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.lab_width / 2, self.lab_width / 2)], dim=1), batch_index=env_index)
            if env_index is None:
                landmark.pos_shaping = torch.stack([torch.linalg.vector_norm(landmark.state.pos - a.state.pos, dim=1) for a in self.world.agents], dim=1).min(dim=1)[0] * self.pos_shaping_factor
            else:
                landmark.pos_shaping[env_index] = torch.stack([torch.linalg.vector_norm(landmark.state.pos[env_index] - a.state.pos[env_index]).unsqueeze(-1) for a in self.world.agents], dim=1).min(dim=1)[0] * self.pos_shaping_factor

    def process_action(self, agent: Agent):
        self.input_queue.append(agent.action.u.clone())
        agent.action.u = self.input_queue.pop(0)
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, self.u_range)
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < self.min_input_norm] = 0
        agent.vel_action = agent.action.u.clone()
        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.pos_rew[:] = 0
            self.time_rew[:] = 0
            goal_dist = torch.stack([torch.linalg.vector_norm(self.goal.state.pos - a.state.pos, dim=1) for a in self.world.agents], dim=1).min(dim=1)[0]
            self.goal_reached = goal_dist < self.goal.shape.radius
            pos_shaping = goal_dist * self.pos_shaping_factor
            self.pos_rew[~self.goal_reached] = (self.goal.pos_shaping - pos_shaping)[~self.goal_reached]
            self.goal.pos_shaping = pos_shaping
            self.time_rew[~self.goal_reached] += self.time_rew_coeff
        agent.energy_expenditure = torch.stack([torch.linalg.vector_norm(a.action.u, dim=-1) / math.sqrt(self.world.dim_p * a.f_range ** 2) for a in self.world.agents], dim=1).sum(-1)
        agent.energy_rew = -agent.energy_expenditure * self.energy_reward_coeff
        return self.pos_rew + agent.energy_rew + self.time_rew

    def observation(self, agent: Agent):
        observations = [agent.state.pos, agent.state.vel, agent.state.pos - self.goal.state.pos]
        if self.obs_noise > 0:
            for i, obs in enumerate(observations):
                noise = torch.zeros(*obs.shape, device=self.world.device).uniform_(-self.obs_noise, self.obs_noise)
                observations[i] = obs + noise
        return torch.cat(observations, dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'pos_rew': self.pos_rew, 'energy_rew': agent.energy_rew, 'time_rew': self.time_rew}

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.green_mass = kwargs.pop('green_mass', 1)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.plot_grid = True
        self.agent_radius = 0.16
        controller_params = [2, 6, 0.002]
        linear_friction = 0.1
        v_range = 1
        a_range = 1
        f_range = linear_friction + a_range
        u_range = v_range
        world = World(batch_dim, device, linear_friction=linear_friction, drag=0, dt=0.05, substeps=4)
        null_action = torch.zeros(world.batch_dim, world.dim_p, device=world.device)
        self.input_queue = [null_action.clone() for _ in range(2)]
        agent = Agent(name='agent 0', collide=False, color=Color.GREEN, render_action=True, mass=self.green_mass, f_range=f_range, u_range=u_range)
        agent.controller = VelocityController(agent, world, controller_params, 'standard')
        world.add_agent(agent)
        agent = Agent(name='agent 1', collide=False, render_action=True, u_range=u_range)
        agent.controller = VelocityController(agent, world, controller_params, 'standard')
        world.add_agent(agent)
        agent = Agent(name='agent 2', collide=False, render_action=True, f_range=30, u_range=u_range)
        agent.controller = VelocityController(agent, world, controller_params, 'standard')
        world.add_agent(agent)
        self.landmark = Landmark('landmark 0', collide=False, movable=True)
        world.add_landmark(self.landmark)
        self.energy_expenditure = torch.zeros(batch_dim, device=device)
        return world

    def reset_world_at(self, env_index: int=None):
        for agent in self.world.agents:
            agent.controller.reset(env_index)
            agent.set_pos(torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-1, -1), torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(0, 0)], dim=1), batch_index=env_index)

    def process_action(self, agent: Agent):
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < 0.08] = 0
        if agent == self.world.agents[1]:
            max_a = 1
            agent.vel_goal = agent.action.u[:, X]
            requested_a = (agent.vel_goal - agent.state.vel[:, X]) / self.world.dt
            achievable_a = torch.clamp(requested_a, -max_a, max_a)
            agent.action.u[:, X] = achievable_a * self.world.dt + agent.state.vel[:, X]
        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.energy_expenditure = -torch.stack([torch.linalg.vector_norm(a.action.u, dim=-1) for a in self.world.agents], dim=1).sum(-1) * 3
        return self.energy_expenditure

    def observation(self, agent: Agent):
        return torch.cat([agent.state.pos, agent.state.vel], dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'energy_expenditure': self.energy_expenditure}

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.u_range = kwargs.pop('u_range', 1)
        self.a_range = kwargs.pop('a_range', 1)
        self.obs_noise = kwargs.pop('obs_noise', 0.0)
        self.dt_delay = kwargs.pop('dt_delay', 0)
        self.min_input_norm = kwargs.pop('min_input_norm', 0.08)
        self.linear_friction = kwargs.pop('linear_friction', 0.1)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.agent_radius = 0.16
        self.desired_radius = 1.5
        self.viewer_zoom = 2
        world = World(batch_dim, device, linear_friction=self.linear_friction, dt=0.05, drag=0)
        controller_params = [2, 6, 0.002]
        self.f_range = self.a_range + self.linear_friction
        null_action = torch.zeros(world.batch_dim, world.dim_p, device=world.device)
        self.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        self.agent = Agent(name='agent_0', shape=Sphere(self.agent_radius), f_range=self.f_range, u_range=self.u_range, render_action=True)
        self.agent.controller = VelocityController(self.agent, world, controller_params, 'standard')
        world.add_agent(self.agent)
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.dot_product = self.pos_rew.clone()
        return world

    def process_action(self, agent: Agent):
        self.input_queue.append(agent.action.u.clone())
        agent.action.u = self.input_queue.pop(0)
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, self.u_range)
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < self.min_input_norm] = 0
        agent.vel_action = agent.action.u.clone()
        agent.controller.process_force()

    def reset_world_at(self, env_index: int=None):
        self.agent.controller.reset(env_index)
        self.agent.set_pos(torch.zeros((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32).uniform_(-self.desired_radius, self.desired_radius), batch_index=env_index)

    def reward(self, agent: Agent):
        closest_point = self.get_closest_point_circle(agent)
        self.pos_rew = -torch.linalg.vector_norm(agent.state.pos - closest_point, dim=1) ** 0.5 * 1
        tangent = self.get_tangent_to_circle(agent, closest_point)
        self.dot_product = torch.einsum('bs,bs->b', tangent, agent.state.vel) * 0.5
        return self.pos_rew + self.dot_product

    def get_closest_point_circle(self, agent: Agent):
        pos_norm = torch.linalg.vector_norm(agent.state.pos, dim=1)
        agent_pos_normalized = agent.state.pos / pos_norm.unsqueeze(-1)
        agent_pos_normalized *= self.desired_radius
        return torch.nan_to_num(agent_pos_normalized)

    def get_next_closest_point_circle(self, agent: Agent):
        closest_point = self.get_closest_point_circle(agent)
        angle = torch.atan2(closest_point[:, Y], closest_point[:, X])
        angle += torch.pi / 24
        new_point = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1) * self.desired_radius
        return new_point

    def get_tangent_to_circle(self, agent: Agent, closest_point=None):
        if closest_point is None:
            closest_point = self.get_closest_point_circle(agent)
        distance_to_circle = agent.state.pos - closest_point
        inside_circle = (torch.linalg.vector_norm(agent.state.pos, dim=1) < self.desired_radius,)
        angle_90 = torch.tensor(torch.pi / 2, device=self.world.device).expand(self.world.batch_dim)
        rotated_vector_90 = TorchUtils.rotate_vector(distance_to_circle, angle_90)
        rotated_vector_neg_90 = TorchUtils.rotate_vector(distance_to_circle, -angle_90)
        rotated_vector = rotated_vector_90
        rotated_vector[inside_circle] = rotated_vector_neg_90[inside_circle]
        angle = rotated_vector / torch.linalg.vector_norm(rotated_vector, dim=1).unsqueeze(-1)
        angle = torch.nan_to_num(angle)
        return angle

    def observation(self, agent: Agent):
        observations = [agent.state.pos, agent.state.vel, agent.state.pos]
        for i, obs in enumerate(observations):
            noise = torch.zeros(*obs.shape, device=self.world.device).uniform_(-self.obs_noise, self.obs_noise)
            observations[i] = obs + noise
        return torch.cat(observations, dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'pos_rew': self.pos_rew, 'dot_product': self.dot_product}

    def extra_render(self, env_index: int=0):
        from vmas.simulator import rendering
        geoms = []
        color = Color.BLACK.value
        circle = rendering.make_circle(self.desired_radius, filled=False)
        xform = rendering.Transform()
        circle.add_attr(xform)
        xform.set_translation(0, 0)
        circle.set_color(*color)
        geoms.append(circle)
        tangent = self.get_tangent_to_circle(self.agent)
        color = Color.BLACK.value
        circle = rendering.Line((0, 0), tangent[env_index], width=1)
        xform = rendering.Transform()
        circle.add_attr(xform)
        circle.set_color(*color)
        geoms.append(circle)
        return geoms

