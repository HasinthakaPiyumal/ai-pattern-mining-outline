# Cluster 6

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.random_start_angle = kwargs.pop('random_start_angle', True)
        self.pos_shaping_factor = kwargs.pop('pos_shaping_factor', 1)
        self.collision_reward = kwargs.pop('collision_reward', -10)
        self.max_speed_1 = kwargs.pop('max_speed_1', None)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.n_agents = 2
        self.wall_length = 2
        self.agent_spacing = 0.5
        self.agent_radius = 0.03
        self.ball_radius = self.agent_radius
        world = World(batch_dim, device, substeps=15, joint_force=900, collision_force=1500)
        agent = Agent(name='agent_0', shape=Sphere(self.agent_radius), u_multiplier=1, mass=1)
        world.add_agent(agent)
        agent = Agent(name='agent_1', shape=Sphere(self.agent_radius), u_multiplier=1, mass=1, max_speed=self.max_speed_1)
        world.add_agent(agent)
        self.goal = Landmark(name='goal', shape=Sphere(radius=self.ball_radius), collide=False, color=Color.GREEN)
        world.add_landmark(self.goal)
        self.ball = Landmark(name='ball', shape=Sphere(radius=self.ball_radius), collide=True, movable=True)
        world.add_landmark(self.ball)
        self.joints = []
        for i in range(2):
            self.joints.append(Joint(world.agents[i], self.ball, anchor_a=(0, 0), anchor_b=(0, 0), dist=self.agent_spacing / 2, rotate_a=True, rotate_b=True, collidable=False, width=0, mass=1))
            world.add_joint(self.joints[i])
        self.build_path_line(world)
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.collision_rew = self.pos_rew.clone()
        self.collided = torch.full((world.batch_dim,), False, device=device)
        return world

    def reset_world_at(self, env_index: int=None):
        start_angle = torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-torch.pi / 2 + torch.pi / 3 if self.random_start_angle else 0, torch.pi / 2 - torch.pi / 3 if self.random_start_angle else 0)
        start_delta_x = self.agent_spacing / 2 * torch.cos(start_angle)
        min_x_start = -self.agent_radius
        max_x_start = self.agent_radius
        start_delta_y = self.agent_spacing / 2 * torch.sin(start_angle)
        min_y_start = -self.wall_length / 2 + 2 * self.agent_radius
        max_y_start = -self.agent_radius
        min_x_goal = min_x_start
        max_x_goal = max_x_start
        min_y_goal = -min_y_start
        max_y_goal = -max_x_start
        ball_position = torch.cat([(min_x_start - max_x_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_x_start, (min_y_start - max_y_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_y_start], dim=1)
        goal_pos = torch.cat([(min_x_goal - max_x_goal) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_x_goal, (min_y_goal - max_y_goal) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_y_goal], dim=1)
        self.goal.set_pos(goal_pos, batch_index=env_index)
        self.ball.set_pos(ball_position, batch_index=env_index)
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(ball_position + torch.cat([start_delta_x, start_delta_y], dim=1) * (-1 if i == 0 else 1), batch_index=env_index)
        for i, joint in enumerate(self.joints):
            joint.landmark.set_pos(ball_position + torch.cat([start_delta_x, start_delta_y], dim=1) / 2 * (-1 if i == 0 else 1), batch_index=env_index)
            joint.landmark.set_rot(start_angle + (torch.pi if i == 1 else 0), batch_index=env_index)
        self.spawn_path_line(env_index)
        if env_index is None:
            self.pos_shaping = torch.linalg.vector_norm(self.ball.state.pos - self.goal.state.pos, dim=1) * self.pos_shaping_factor
            self.collided[:] = False
        else:
            self.pos_shaping[env_index] = torch.linalg.vector_norm(self.ball.state.pos[env_index] - self.goal.state.pos[env_index]) * self.pos_shaping_factor
            self.collided[env_index] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            self.pos_rew[:] = 0
            self.collision_rew[:] = 0
            self.collided[:] = False
            dist_to_goal = torch.linalg.vector_norm(self.ball.state.pos - self.goal.state.pos, dim=1)
            pos_shaping = dist_to_goal * self.pos_shaping_factor
            self.pos_rew += self.pos_shaping - pos_shaping
            self.pos_shaping = pos_shaping
            for collidable in self.world.agents + [self.ball]:
                for entity in self.walls + self.floors:
                    is_overlap = self.world.is_overlapping(collidable, entity)
                    self.collision_rew[is_overlap] += self.collision_reward
                    self.collided += is_overlap
            self.rew = self.pos_rew + self.collision_rew
        return self.rew

    def observation(self, agent: Agent):
        return torch.cat([agent.state.pos, agent.state.vel, agent.state.pos - self.goal.state.pos], dim=-1)

    def done(self):
        return (torch.linalg.vector_norm(self.ball.state.pos - self.goal.state.pos, dim=1) <= 0.01) + self.collided

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'pos_rew': self.pos_rew, 'collision_rew': self.collision_rew}

    def build_path_line(self, world: World):
        self.walls = []
        for i in range(2):
            self.walls.append(Landmark(name=f'wall {i}', collide=True, shape=Line(length=self.wall_length), color=Color.BLACK))
            world.add_landmark(self.walls[i])
        self.floors = []
        for i in range(2):
            self.floors.append(Landmark(name=f'floor {i}', collide=True, shape=Line(length=self.agent_spacing / 2), color=Color.BLACK))
            world.add_landmark(self.floors[i])

    def spawn_path_line(self, env_index):
        for i, wall in enumerate(self.walls):
            wall.set_pos(torch.tensor([self.agent_spacing / 4 * (-1 if i == 0 else 1), 0.0], device=self.world.device), batch_index=env_index)
            wall.set_rot(torch.tensor(torch.pi / 2, device=self.world.device), batch_index=env_index)
        for i, floor in enumerate(self.floors):
            floor.set_pos(torch.tensor([0, self.wall_length / 2 * (-1 if i == 0 else 1)], device=self.world.device), batch_index=env_index)

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.pos_shaping_factor = kwargs.pop('pos_shaping_factor', 0)
        self.speed_shaping_factor = kwargs.pop('speed_shaping_factor', 1)
        self.dist_shaping_factor = kwargs.pop('dist_shaping_factor', 0)
        self.joints = kwargs.pop('joints', True)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.n_agents = 2
        self.desired_speed = 1
        self.desired_radius = 0.5
        self.agent_spacing = 0.4
        self.agent_radius = 0.03
        self.ball_radius = 2 * self.agent_radius
        world = World(batch_dim, device, substeps=15 if self.joints else 5, joint_force=900 if self.joints else JOINT_FORCE, collision_force=1500 if self.joints else 400, drag=0)
        agent = Agent(name='agent_0', shape=Sphere(self.agent_radius), drag=0.25)
        world.add_agent(agent)
        agent = Agent(name='agent_1', shape=Sphere(self.agent_radius), drag=0.25)
        world.add_agent(agent)
        self.ball = Landmark(name='ball', shape=Sphere(radius=self.ball_radius), collide=True, movable=True, linear_friction=0.04)
        world.add_landmark(self.ball)
        if self.joints:
            self.joints = []
            for i in range(self.n_agents):
                self.joints.append(Joint(world.agents[i], self.ball, anchor_a=(0, 0), anchor_b=(0, 0), dist=self.agent_spacing / 2, rotate_a=True, rotate_b=True, collidable=False, width=0, mass=1))
                world.add_joint(self.joints[i])
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.speed_rew = self.pos_rew.clone()
        self.dist_rew = self.pos_rew.clone()
        return world

    def reset_world_at(self, env_index: int=None):
        ball_pos = torch.zeros((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32).uniform_(-self.desired_radius, self.desired_radius)
        self.ball.set_pos(ball_pos, batch_index=env_index)
        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        for i, agent in enumerate(agents):
            agent_pos = ball_pos.clone()
            agent_pos[:, X] += self.agent_spacing / 2 * (-1 if i == 0 else 1)
            agent.set_pos(agent_pos, batch_index=env_index)
        if env_index is None:
            self.pos_shaping = torch.linalg.vector_norm(self.ball.state.pos - self.get_closest_point_circle(self.ball.state.pos), dim=1) ** 0.5 * self.pos_shaping_factor
            self.speed_shaping = (self.desired_speed - torch.linalg.vector_norm(self.ball.state.vel, dim=1)).abs() * self.speed_shaping_factor
            self.dist_shaping = torch.stack([torch.linalg.vector_norm(a.state.pos - self.ball.state.pos, dim=1) for a in self.world.agents], dim=1).sum(dim=1) * self.dist_shaping_factor
        else:
            self.pos_shaping = torch.linalg.vector_norm(self.ball.state.pos[env_index] - self.get_closest_point_circle(self.ball.state.pos)[env_index]) ** 0.5 * self.pos_shaping_factor
            self.speed_shaping[env_index] = (self.desired_speed - torch.linalg.vector_norm(self.ball.state.vel[env_index])).abs() * self.speed_shaping_factor
            self.dist_shaping[env_index] = torch.stack([torch.linalg.vector_norm(a.state.pos[env_index] - self.ball.state.pos[env_index]).unsqueeze(-1) for a in self.world.agents], dim=1).sum(dim=1) * self.dist_shaping_factor

    def reward(self, agent: Agent):
        pos_shaping = torch.linalg.vector_norm(self.ball.state.pos - self.get_closest_point_circle(self.ball.state.pos), dim=1) ** 0.5 * self.pos_shaping_factor
        self.pos_rew = self.pos_shaping - pos_shaping
        self.pos_shaping = pos_shaping
        speed = torch.linalg.vector_norm(self.ball.state.vel, dim=1)
        speed_shaping = (self.desired_speed - speed).abs() * self.speed_shaping_factor
        self.speed_rew = self.speed_shaping - speed_shaping
        self.speed_shaping = speed_shaping
        dist_shaping = torch.stack([torch.linalg.vector_norm(a.state.pos - self.ball.state.pos, dim=1) for a in self.world.agents], dim=1).sum(dim=1) * self.dist_shaping_factor
        self.dist_rew = self.dist_shaping - dist_shaping
        self.dist_shaping = dist_shaping
        return self.pos_rew + self.speed_rew + self.dist_rew

    def observation(self, agent: Agent):
        return torch.cat([agent.state.pos, agent.state.vel, agent.state.pos - self.ball.state.pos, agent.state.pos], dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'pos_rew': self.pos_rew, 'speed_rew': self.speed_rew, 'dist_rew': self.dist_rew}

    def get_closest_point_circle(self, pos: Tensor):
        pos_norm = torch.linalg.vector_norm(pos, dim=1)
        agent_pos_normalized = pos / pos_norm.unsqueeze(-1)
        agent_pos_normalized *= self.desired_radius
        return torch.nan_to_num(agent_pos_normalized)

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
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.fixed_passage = kwargs.pop('fixed_passage', False)
        self.joint_length = kwargs.pop('joint_length', 0.52)
        self.random_start_angle = kwargs.pop('random_start_angle', False)
        self.random_goal_angle = kwargs.pop('random_goal_angle', False)
        self.observe_joint_angle = kwargs.pop('observe_joint_angle', False)
        self.joint_angle_obs_noise = kwargs.pop('joint_angle_obs_noise', 0.0)
        self.asym_package = kwargs.pop('asym_package', False)
        self.mass_ratio = kwargs.pop('mass_ratio', 1)
        self.mass_position = kwargs.pop('mass_position', 0.75)
        self.max_speed_1 = kwargs.pop('max_speed_1', None)
        self.pos_shaping_factor = kwargs.pop('pos_shaping_factor', 1)
        self.rot_shaping_factor = kwargs.pop('rot_shaping_factor', 1)
        self.collision_reward = kwargs.pop('collision_reward', 0)
        self.energy_reward_coeff = kwargs.pop('energy_reward_coeff', 0)
        self.obs_noise = kwargs.pop('obs_noise', 0.0)
        self.n_passages = kwargs.pop('n_passages', 3)
        self.middle_angle_180 = kwargs.pop('middle_angle_180', False)
        self.use_vel_controller = kwargs.pop('use_vel_controller', False)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        assert self.n_passages == 3 or self.n_passages == 4
        self.plot_grid = False
        self.visualize_semidims = False
        world = World(batch_dim, device, x_semidim=1, y_semidim=1, substeps=5 if not self.asym_package else 10, joint_force=700 if self.asym_package else 400, collision_force=2500 if self.asym_package else 1500, drag=0.25 if not self.asym_package else 0.15)
        if not self.observe_joint_angle:
            assert self.joint_angle_obs_noise == 0
        self.n_agents = 2
        self.middle_angle = torch.zeros((world.batch_dim, 1), device=world.device)
        self.agent_radius = 0.03333
        self.agent_radius_2 = 3 * self.agent_radius
        self.mass_radius = self.agent_radius * (2 / 3)
        self.passage_width = 0.2
        self.passage_length = 0.1476
        self.scenario_length = 2 + 2 * self.agent_radius
        self.n_boxes = int(self.scenario_length // self.passage_length)
        self.min_collision_distance = 0.005
        cotnroller_params = [2.0, 10, 1e-05]
        agent = Agent(name='agent_0', shape=Sphere(self.agent_radius), u_range=1, obs_noise=self.obs_noise, render_action=True, f_range=10)
        agent.controller = VelocityController(agent, world, cotnroller_params, 'standard')
        world.add_agent(agent)
        agent = Agent(name='agent_1', shape=Sphere(self.agent_radius_2), u_range=1, mass=1 if self.asym_package else self.mass_ratio, max_speed=self.max_speed_1, obs_noise=self.obs_noise, render_action=True, f_range=10)
        agent.controller = VelocityController(agent, world, cotnroller_params, 'standard')
        world.add_agent(agent)
        self.joint = Joint(world.agents[0], world.agents[1], anchor_a=(0, 0), anchor_b=(0, 0), dist=self.joint_length, rotate_a=True, rotate_b=True, collidable=False, width=0, mass=1)
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

    def set_n_passages(self, val):
        if val == 4:
            self.middle_angle_180 = True
        elif val == 3:
            self.middle_angle_180 = False
        else:
            raise AssertionError()
        self.n_passages = val
        del self.world._landmarks[-self.n_boxes:]
        self.create_passage_map(self.world)
        self.reset_world_at()

    def reset_world_at(self, env_index: int=None):
        start_angle = torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device)
        start_angle[start_angle >= 0.5] = torch.pi / 2
        start_angle[start_angle < 0.5] = -torch.pi / 2
        goal_angle = torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-torch.pi / 2 if self.random_goal_angle else torch.pi, torch.pi / 2 if self.random_goal_angle else torch.pi)
        bigger_radius = max(self.agent_radius, self.agent_radius_2)
        start_delta_x = self.joint_length / 2 * torch.cos(start_angle)
        start_delta_x_abs = start_delta_x.abs()
        min_x_start = -self.world.x_semidim + (bigger_radius + start_delta_x_abs)
        max_x_start = self.world.x_semidim - (bigger_radius + start_delta_x_abs)
        start_delta_y = self.joint_length / 2 * torch.sin(start_angle)
        start_delta_y_abs = start_delta_y.abs()
        min_y_start = -self.world.y_semidim + (bigger_radius + start_delta_y_abs)
        max_y_start = -2 * bigger_radius - self.passage_width / 2 - start_delta_y_abs
        goal_delta_x = self.joint_length / 2 * torch.cos(goal_angle)
        goal_delta_x_abs = goal_delta_x.abs()
        min_x_goal = -self.world.x_semidim + (bigger_radius + goal_delta_x_abs)
        max_x_goal = self.world.x_semidim - (bigger_radius + goal_delta_x_abs)
        goal_delta_y = self.joint_length / 2 * torch.sin(goal_angle)
        goal_delta_y_abs = goal_delta_y.abs()
        min_y_goal = 2 * bigger_radius + self.passage_width / 2 + goal_delta_y_abs
        max_y_goal = self.world.y_semidim - (bigger_radius + goal_delta_y_abs)
        joint_pos = torch.cat([(min_x_start - max_x_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_x_start, (min_y_start - max_y_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_y_start], dim=1)
        goal_pos = torch.cat([(min_x_goal - max_x_goal) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_x_goal, (min_y_goal - max_y_goal) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_y_goal], dim=1)
        self.goal.set_pos(goal_pos, batch_index=env_index)
        self.goal.set_rot(goal_angle, batch_index=env_index)
        agents = self.world.agents
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
            self.t = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            self.passed = torch.zeros((self.world.batch_dim,), device=self.world.device)
            self.joint.pos_shaping_pre = torch.linalg.vector_norm(self.joint.landmark.state.pos - self.pass_center, dim=1) * self.pos_shaping_factor
            self.joint.pos_shaping_post = torch.linalg.vector_norm(self.joint.landmark.state.pos - self.goal.state.pos, dim=1) * self.pos_shaping_factor
            self.joint.rot_shaping_pre = (get_line_angle_dist_0_360(self.joint.landmark.state.rot, self.middle_angle) if not self.middle_angle_180 else get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.middle_angle)) * self.rot_shaping_factor
        else:
            self.t[env_index] = 0
            self.passed[env_index] = 0
            self.joint.pos_shaping_pre[env_index] = torch.linalg.vector_norm(self.joint.landmark.state.pos[env_index] - self.pass_center[env_index]) * self.pos_shaping_factor
            self.joint.pos_shaping_post[env_index] = torch.linalg.vector_norm(self.joint.landmark.state.pos[env_index] - self.goal.state.pos[env_index]) * self.pos_shaping_factor
            self.joint.rot_shaping_pre[env_index] = (get_line_angle_dist_0_360(self.joint.landmark.state.rot[env_index].unsqueeze(-1), self.middle_angle[env_index].unsqueeze(-1)) if not self.middle_angle_180 else get_line_angle_dist_0_180(self.joint.landmark.state.rot[env_index].unsqueeze(-1), self.middle_angle[env_index].unsqueeze(-1))) * self.rot_shaping_factor

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.t += 1
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            self.pos_rew[:] = 0
            self.rot_rew[:] = 0
            self.collision_rew[:] = 0
            self.energy_rew[:] = 0
            joint_passed = self.joint.landmark.state.pos[:, Y] > 0
            self.all_passed = (torch.stack([a.state.pos[:, Y] for a in self.world.agents], dim=1) > self.passage_width / 2).all(dim=1)
            joint_dist_to_closest_pass = torch.linalg.vector_norm(self.joint.landmark.state.pos - self.pass_center, dim=1) * self.pos_shaping_factor
            joint_shaping = joint_dist_to_closest_pass * self.pos_shaping_factor
            self.pos_rew[~joint_passed] += (self.joint.pos_shaping_pre - joint_shaping)[~joint_passed]
            self.joint.pos_shaping_pre = joint_shaping
            joint_dist_to_goal = torch.linalg.vector_norm(self.joint.landmark.state.pos - self.goal.state.pos, dim=1)
            joint_shaping = joint_dist_to_goal * self.pos_shaping_factor
            self.pos_rew[joint_passed] += (self.joint.pos_shaping_post - joint_shaping)[joint_passed]
            self.joint.pos_shaping_post = joint_shaping
            joint_dist_to_90_rot = get_line_angle_dist_0_360(self.joint.landmark.state.rot, self.middle_angle) if not self.middle_angle_180 else get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.middle_angle)
            joint_shaping = joint_dist_to_90_rot * self.rot_shaping_factor
            self.rot_rew += self.joint.rot_shaping_pre - joint_shaping
            self.joint.rot_shaping_pre = joint_shaping
            if self.collision_reward != 0:
                for a in self.world.agents + ([self.mass] if self.asym_package else []):
                    for passage in self.passages:
                        if passage.collide:
                            self.collision_rew[self.world.get_distance(a, passage) <= self.min_collision_distance] += self.collision_reward
                    for wall in self.walls:
                        self.collision_rew[self.world.get_distance(a, wall) <= self.min_collision_distance] += self.collision_reward
            if self.energy_reward_coeff != 0:
                self.energy_expenditure = torch.stack([torch.linalg.vector_norm(a.action.u, dim=-1) / math.sqrt(self.world.dim_p * (a.u_range * a.u_multiplier) ** 2) for a in self.world.agents], dim=1).sum(-1)
                self.energy_rew = -self.energy_expenditure * self.energy_reward_coeff
            self.rew = self.pos_rew + self.rot_rew + self.collision_rew + self.energy_rew
        return self.rew

    def process_action(self, agent: Agent):
        if self.use_vel_controller:
            vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 0.001
            agent.controller.reset(vel_is_zero)
            agent.controller.process_force()

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
        observations = [agent.state.pos, agent.state.vel, agent.state.pos - self.goal.state.pos, agent.state.pos - self.big_passage_pos, agent.state.pos - self.small_passage_pos, angle_to_vector(self.goal.state.rot)] + ([angle_to_vector(joint_angle)] if self.observe_joint_angle else [])
        if self.obs_noise > 0:
            for i, obs in enumerate(observations):
                noise = torch.zeros(*obs.shape, device=self.world.device).uniform_(-self.obs_noise, self.obs_noise)
                observations[i] = obs + noise
        return torch.cat(observations, dim=-1)

    def done(self):
        return torch.all((torch.linalg.vector_norm(self.joint.landmark.state.pos - self.goal.state.pos, dim=1) <= 0.01) * (get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.goal.state.rot).unsqueeze(-1) <= 0.01), dim=1)

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

        def is_passage(i):
            return i < self.n_passages
        for i in range(self.n_boxes):
            passage = Landmark(name=f'passage {i}', collide=not is_passage(i), movable=False, shape=Box(length=self.passage_length, width=self.passage_width), color=Color.RED, collision_filter=lambda e: not isinstance(e.shape, Box))
            if not passage.collide:
                self.non_collide_passages.append(passage)
            else:
                self.collide_passages.append(passage)
            self.passages.append(passage)
            world.add_landmark(passage)

    def spawn_passage_map(self, env_index):
        if self.fixed_passage:
            big_passage_start_index = torch.full((self.world.batch_dim,) if env_index is None else (1,), 5, device=self.world.device)
            small_left_or_right = torch.full((self.world.batch_dim,) if env_index is None else (1,), 1, device=self.world.device)
        else:
            big_passage_start_index = torch.randint(0, self.n_boxes - 1, (self.world.batch_dim,) if env_index is None else (1,), device=self.world.device)
            small_left_or_right = torch.randint(0, 2, (self.world.batch_dim,) if env_index is None else (1,), device=self.world.device)
        small_left_or_right[big_passage_start_index > self.n_boxes - 1 - (self.n_passages + 1)] = 0
        small_left_or_right[big_passage_start_index < self.n_passages] = 1
        small_left_or_right[small_left_or_right == 0] -= 3
        small_left_or_right[small_left_or_right == 1] += 3

        def is_passage(i):
            is_pass = big_passage_start_index == i
            is_pass += big_passage_start_index == i - 1
            is_pass += big_passage_start_index + small_left_or_right == i
            if self.n_passages == 4:
                is_pass += big_passage_start_index + small_left_or_right == i - torch.sign(small_left_or_right)
            return is_pass

        def get_pos(i):
            pos = torch.tensor([-1 - self.agent_radius + self.passage_length / 2, 0.0], dtype=torch.float32, device=self.world.device).repeat(i.shape[0], 1)
            pos[:, X] += self.passage_length * i
            return pos
        for index, i in enumerate([big_passage_start_index, big_passage_start_index + 1, big_passage_start_index + small_left_or_right] + ([big_passage_start_index + small_left_or_right + torch.sign(small_left_or_right)] if self.n_passages == 4 else [])):
            self.non_collide_passages[index].is_rendering[:] = False
            self.non_collide_passages[index].set_pos(get_pos(i), batch_index=env_index)
        big_passage_pos = (get_pos(big_passage_start_index) + get_pos(big_passage_start_index + 1)) / 2
        small_passage_pos = get_pos(big_passage_start_index + small_left_or_right)
        pass_center = (big_passage_pos + small_passage_pos) / 2
        if env_index is None:
            self.small_left_or_right = small_left_or_right
            self.pass_center = pass_center
            self.big_passage_pos = big_passage_pos
            self.small_passage_pos = small_passage_pos
            self.middle_angle[small_left_or_right > 0] = torch.pi
            self.middle_angle[small_left_or_right < 0] = 0
        else:
            self.pass_center[env_index] = pass_center
            self.small_left_or_right[env_index] = small_left_or_right
            self.big_passage_pos[env_index] = big_passage_pos
            self.small_passage_pos[env_index] = small_passage_pos
            self.middle_angle[env_index] = 0 if small_left_or_right.item() < 0 else torch.pi
        i = torch.zeros((self.world.batch_dim,) if env_index is None else (1,), dtype=torch.int, device=self.world.device)
        for passage in self.collide_passages:
            is_pass = is_passage(i)
            while is_pass.any():
                i[is_pass] += 1
                is_pass = is_passage(i)
            passage.set_pos(get_pos(i), batch_index=env_index)
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
        self.joint_length = kwargs.pop('joint_length', 0.5)
        self.random_start_angle = kwargs.pop('random_start_angle', False)
        self.observe_joint_angle = kwargs.pop('observe_joint_angle', False)
        self.joint_angle_obs_noise = kwargs.pop('joint_angle_obs_noise', 0.0)
        self.asym_package = kwargs.pop('asym_package', True)
        self.mass_ratio = kwargs.pop('mass_ratio', 5)
        self.mass_position = kwargs.pop('mass_position', 0.75)
        self.max_speed_1 = kwargs.pop('max_speed_1', None)
        self.obs_noise = kwargs.pop('obs_noise', 0.2)
        self.rot_shaping_factor = kwargs.pop('rot_shaping_factor', 1)
        self.energy_reward_coeff = kwargs.pop('energy_reward_coeff', 0.08)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        world = World(batch_dim, device, substeps=7 if not self.asym_package else 10, joint_force=900 if self.asym_package else 400, drag=0.25 if not self.asym_package else 0.15)
        if not self.observe_joint_angle:
            assert self.joint_angle_obs_noise == 0
        self.goal_angle = torch.pi / 2
        self.n_agents = 2
        self.agent_radius = 0.03333
        self.mass_radius = self.agent_radius * (2 / 3)
        agent = Agent(name='agent 0', shape=Sphere(self.agent_radius), u_multiplier=0.8, obs_noise=self.obs_noise, render_action=True)
        world.add_agent(agent)
        agent = Agent(name='agent 1', shape=Sphere(self.agent_radius), u_multiplier=0.8, mass=1 if self.asym_package else self.mass_ratio, max_speed=self.max_speed_1, obs_noise=self.obs_noise, render_action=True)
        world.add_agent(agent)
        self.joint = Joint(world.agents[0], world.agents[1], anchor_a=(0, 0), anchor_b=(0, 0), dist=self.joint_length, rotate_a=True, rotate_b=True, collidable=False, width=0, mass=1)
        world.add_joint(self.joint)
        if self.asym_package:

            def mass_collision_filter(e):
                return not isinstance(e.shape, Sphere)
            self.mass = Landmark(name='mass', shape=Sphere(radius=self.mass_radius), collide=False, movable=True, color=Color.BLACK, mass=self.mass_ratio, collision_filter=mass_collision_filter)
            world.add_landmark(self.mass)
            joint = Joint(self.mass, self.joint.landmark, anchor_a=(0, 0), anchor_b=(self.mass_position, 0), dist=0, rotate_a=True, rotate_b=True)
            world.add_joint(joint)
        self.rot_rew = torch.zeros(batch_dim, device=device)
        self.energy_rew = self.rot_rew.clone()
        return world

    def reset_world_at(self, env_index: int=None):
        start_angle = torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-torch.pi / 2 if self.random_start_angle else 0, torch.pi / 2 if self.random_start_angle else 0)
        start_delta_x = self.joint_length / 2 * torch.cos(start_angle)
        min_x_start = 0
        max_x_start = 0
        start_delta_y = self.joint_length / 2 * torch.sin(start_angle)
        min_y_start = 0
        max_y_start = 0
        joint_pos = torch.cat([(min_x_start - max_x_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_x_start, (min_y_start - max_y_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_y_start], dim=1)
        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        for i, agent in enumerate(agents):
            if i == 0:
                agent.set_pos(joint_pos - torch.cat([start_delta_x, start_delta_y], dim=1), batch_index=env_index)
            else:
                agent.set_pos(joint_pos + torch.cat([start_delta_x, start_delta_y], dim=1), batch_index=env_index)
        if self.asym_package:
            self.mass.set_pos(joint_pos + self.mass_position * torch.cat([start_delta_x, start_delta_y], dim=1) * (1 if agents[0] == self.world.agents[0] else -1), batch_index=env_index)
        if env_index is None:
            self.joint.rot_shaping_pre = get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.goal_angle) * self.rot_shaping_factor
        else:
            self.joint.rot_shaping_pre[env_index] = get_line_angle_dist_0_180(self.joint.landmark.state.rot[env_index], self.goal_angle) * self.rot_shaping_factor

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rot_rew[:] = 0
            joint_dist_to_90_rot = get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.goal_angle)
            joint_shaping = joint_dist_to_90_rot * self.rot_shaping_factor
            self.rot_rew += self.joint.rot_shaping_pre - joint_shaping
            self.joint.rot_shaping_pre = joint_shaping
            self.energy_expenditure = torch.stack([torch.linalg.vector_norm(a.action.u, dim=-1) / math.sqrt(self.world.dim_p * (a.u_range * a.u_multiplier) ** 2) for a in self.world.agents], dim=1).sum(-1)
            self.energy_rew = -self.energy_expenditure * self.energy_reward_coeff
            self.rew = self.rot_rew + self.energy_rew
        return self.rew

    def observation(self, agent: Agent):
        if self.observe_joint_angle:
            joint_angle = self.joint.landmark.state.rot
            angle_noise = torch.randn(*joint_angle.shape, device=self.world.device, dtype=torch.float32) * self.joint_angle_obs_noise if self.joint_angle_obs_noise else 0.0
            joint_angle += angle_noise
        observations = [agent.state.pos, agent.state.vel] + ([angle_to_vector(joint_angle)] if self.observe_joint_angle else [])
        for i, obs in enumerate(observations):
            noise = torch.zeros(*obs.shape, device=self.world.device).uniform_(-self.obs_noise, self.obs_noise)
            observations[i] = obs.clone() + noise
        return torch.cat(observations, dim=-1)

    def done(self):
        return torch.all(get_line_angle_dist_0_180(self.joint.landmark.state.rot, self.goal_angle).unsqueeze(-1) <= 0.01, dim=1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'rot_rew': self.rot_rew, 'energy_rew': self.energy_rew}

    def extra_render(self, env_index: int=0) -> 'List[Geom]':
        from vmas.simulator import rendering
        geoms = []
        color = Color.GREEN.value
        origin = rendering.make_circle(0.01)
        xform = rendering.Transform()
        origin.add_attr(xform)
        xform.set_translation(0, 0)
        origin.set_color(*color)
        geoms.append(origin)
        return geoms

