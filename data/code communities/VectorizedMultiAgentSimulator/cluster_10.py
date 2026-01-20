# Cluster 10

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
        self.n_passages = kwargs.pop('n_passages', 1)
        self.fixed_passage = kwargs.pop('fixed_passage', False)
        self.random_start_angle = kwargs.pop('random_start_angle', True)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        assert 1 <= self.n_passages <= 20
        self.pos_shaping_factor = 1
        self.collision_reward = -0.06
        self.n_agents = 2
        self.agent_spacing = 0.5
        self.agent_radius = 0.03333
        self.ball_radius = self.agent_radius
        self.passage_width = 0.2
        self.passage_length = 0.103
        self.visualize_semidims = False
        world = World(batch_dim, device, x_semidim=1, y_semidim=1, drag=0, linear_friction=0.0)
        for i in range(2):
            agent = Agent(name=f'agent_{i}', shape=Sphere(self.agent_radius), u_multiplier=0.7, mass=2, drag=0.25)
            world.add_agent(agent)
        self.goal = Landmark(name='goal', shape=Sphere(radius=self.ball_radius), collide=False, color=Color.GREEN)
        world.add_landmark(self.goal)
        self.ball = Landmark(name='ball', shape=Sphere(radius=self.ball_radius), collide=True, movable=True, mass=1, color=Color.BLACK, linear_friction=0.02)
        world.add_landmark(self.ball)
        self.create_passage_map(world)
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.collision_rew = self.pos_rew.clone()
        return world

    def reset_world_at(self, env_index: int=None):
        start_angle = torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-torch.pi / 2 if self.random_start_angle else -torch.pi / 2, torch.pi / 2 if self.random_start_angle else -torch.pi / 2)
        start_delta_x = self.agent_spacing / 2 * torch.cos(start_angle)
        start_delta_x_abs = start_delta_x.abs()
        min_x_start = -self.world.x_semidim + (self.agent_radius + start_delta_x_abs)
        max_x_start = self.world.x_semidim - (self.agent_radius + start_delta_x_abs)
        start_delta_y = self.agent_spacing / 2 * torch.sin(start_angle)
        start_delta_y_abs = start_delta_y.abs()
        min_y_start = -self.world.y_semidim + (self.agent_radius + start_delta_y_abs)
        max_y_start = -2 * self.agent_radius - self.passage_width / 2 - start_delta_y_abs
        min_x_goal = -self.world.x_semidim + self.agent_radius
        max_x_goal = self.world.x_semidim - self.agent_radius
        min_y_goal = 2 * self.agent_radius + self.passage_width / 2
        max_y_goal = self.world.y_semidim - self.agent_radius
        ball_pos = torch.cat([(min_x_start - max_x_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_x_start, (min_y_start - max_y_start) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_y_start], dim=1)
        self.ball.set_pos(ball_pos, batch_index=env_index)
        for i, agent in enumerate(self.world.agents):
            if i == 0:
                agent.set_pos(ball_pos - torch.cat([start_delta_x, start_delta_y], dim=1), batch_index=env_index)
            else:
                agent.set_pos(ball_pos + torch.cat([start_delta_x, start_delta_y], dim=1), batch_index=env_index)
        self.goal.set_pos(torch.cat([(min_x_goal - max_x_goal) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_x_goal, (min_y_goal - max_y_goal) * torch.rand((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) + max_y_goal], dim=1), batch_index=env_index)
        self.spawn_passage_map(env_index)
        if env_index is None:
            self.ball.pos_shaping_pre = torch.stack([torch.linalg.vector_norm(self.ball.state.pos - p.state.pos, dim=1) for p in self.passages if not p.collide], dim=1).min(dim=1)[0] * self.pos_shaping_factor
            self.ball.pos_shaping_post = torch.linalg.vector_norm(self.ball.state.pos - self.goal.state.pos, dim=1) * self.pos_shaping_factor
        else:
            self.ball.pos_shaping_pre[env_index] = torch.stack([torch.linalg.vector_norm(self.ball.state.pos[env_index] - p.state.pos[env_index]).unsqueeze(-1) for p in self.passages if not p.collide], dim=1).min(dim=1)[0] * self.pos_shaping_factor
            self.ball.pos_shaping_post[env_index] = torch.linalg.vector_norm(self.ball.state.pos[env_index] - self.goal.state.pos[env_index]) * self.pos_shaping_factor

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            self.pos_rew[:] = 0
            self.collision_rew[:] = 0
            ball_passed = self.ball.state.pos[:, Y] > 0
            ball_dist_to_closest_pass = torch.stack([torch.linalg.vector_norm(self.ball.state.pos - p.state.pos, dim=1) for p in self.passages if not p.collide], dim=1).min(dim=1)[0]
            ball_shaping = ball_dist_to_closest_pass * self.pos_shaping_factor
            self.pos_rew[~ball_passed] += (self.ball.pos_shaping_pre - ball_shaping)[~ball_passed]
            self.ball.pos_shaping_pre = ball_shaping
            ball_dist_to_goal = torch.linalg.vector_norm(self.ball.state.pos - self.goal.state.pos, dim=1)
            ball_shaping = ball_dist_to_goal * self.pos_shaping_factor
            self.pos_rew[ball_passed] += (self.ball.pos_shaping_post - ball_shaping)[ball_passed]
            self.ball.pos_shaping_post = ball_shaping
            for a in self.world.agents:
                for passage in self.passages:
                    if passage.collide:
                        self.collision_rew[self.world.is_overlapping(a, passage)] += self.collision_reward
            for p in self.passages:
                if p.collide:
                    self.collision_rew[self.world.is_overlapping(p, self.ball)] += self.collision_reward
            self.rew = self.pos_rew + self.collision_rew
        return self.rew

    def observation(self, agent: Agent):
        passage_obs = []
        for passage in self.passages:
            if not passage.collide:
                passage_obs.append(agent.state.pos - passage.state.pos)
        return torch.cat([agent.state.pos, agent.state.vel, agent.state.pos - self.goal.state.pos, agent.state.pos - self.ball.state.pos, *passage_obs], dim=-1)

    def done(self):
        return (torch.linalg.vector_norm(self.ball.state.pos - self.goal.state.pos, dim=1) <= 0.01) + (-self.world.x_semidim + self.ball_radius >= self.ball.state.pos[:, X]) + (self.ball.state.pos[:, X] >= self.world.x_semidim - self.ball_radius) + (-self.world.y_semidim + self.ball_radius >= self.ball.state.pos[:, Y]) + (self.ball.state.pos[:, Y] >= self.world.y_semidim - self.ball_radius)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'pos_rew': self.pos_rew, 'collision_rew': self.collision_rew}

    def create_passage_map(self, world: World):
        self.passages = []
        n_boxes = int((2 * world.x_semidim + 2 * self.agent_radius) // self.passage_length)

        def removed(i):
            return n_boxes // 2 - self.n_passages / 2 <= i < n_boxes // 2 + self.n_passages / 2
        for i in range(n_boxes):
            passage = Landmark(name=f'passage {i}', collide=not removed(i), movable=False, shape=Box(length=self.passage_length, width=self.passage_width), color=Color.RED, collision_filter=lambda e: not isinstance(e.shape, Box))
            self.passages.append(passage)
            world.add_landmark(passage)

    def spawn_passage_map(self, env_index):
        if not self.fixed_passage:
            order = torch.randperm(len(self.passages)).tolist()
            self.passages_to_place = [self.passages[i] for i in order]
        else:
            self.passages_to_place = self.passages
        for i, passage in enumerate(self.passages_to_place):
            if not passage.collide:
                passage.is_rendering[:] = False
            passage.neighbour = False
            try:
                passage.neighbour += not self.passages_to_place[i - 1].collide
            except IndexError:
                pass
            try:
                passage.neighbour += not self.passages_to_place[i + 1].collide
            except IndexError:
                pass
            pos = torch.tensor([-1 - self.agent_radius + self.passage_length / 2 + self.passage_length * i, 0.0], dtype=torch.float32, device=self.world.device)
            passage.neighbour *= passage.collide
            passage.set_pos(pos, batch_index=env_index)

    def extra_render(self, env_index: int=0):
        from vmas.simulator import rendering
        geoms = []
        for i in range(4):
            geom = Line(length=2 + self.agent_radius * 2).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(0.0 if i % 2 else self.world.x_semidim + self.agent_radius if i == 0 else -self.world.x_semidim - self.agent_radius, 0.0 if not i % 2 else self.world.x_semidim + self.agent_radius if i == 1 else -self.world.x_semidim - self.agent_radius)
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)
        return geoms

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
        n_agents = kwargs.pop('n_agents', 4)
        self.package_width = kwargs.pop('package_width', 0.6)
        self.package_length = kwargs.pop('package_length', 0.6)
        self.package_mass = kwargs.pop('package_mass', 50)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.shaping_factor = 100
        world = World(batch_dim, device, contact_margin=0.006, substeps=5, collision_force=500)
        for i in range(n_agents):
            agent = Agent(name=f'agent_{i}', shape=Sphere(0.03), u_multiplier=0.5)
            world.add_agent(agent)
        goal = Landmark(name='goal', collide=False, shape=Sphere(radius=0.09), color=Color.LIGHT_GREEN)
        world.add_landmark(goal)
        self.package = Landmark(name=f'package {i}', collide=True, movable=True, mass=self.package_mass, shape=Box(length=self.package_length, width=self.package_width, hollow=True), color=Color.RED)
        self.package.goal = goal
        world.add_landmark(self.package)
        return world

    def reset_world_at(self, env_index: int=None):
        package_pos = torch.zeros((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32).uniform_(-1.0, 1.0)
        self.package.set_pos(package_pos, batch_index=env_index)
        for agent in self.world.agents:
            agent.set_pos(torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.package_length / 2 + agent.shape.radius, self.package_length / 2 - agent.shape.radius), torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.package_width / 2 + agent.shape.radius, self.package_width / 2 - agent.shape.radius)], dim=1) + package_pos, batch_index=env_index)
        self.package.goal.set_pos(torch.zeros((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32).uniform_(-1.0, 1.0), batch_index=env_index)
        if env_index is None:
            self.package.global_shaping = torch.linalg.vector_norm(self.package.state.pos - self.package.goal.state.pos, dim=1) * self.shaping_factor
            self.package.on_goal = torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)
        else:
            self.package.global_shaping[env_index] = torch.linalg.vector_norm(self.package.state.pos[env_index] - self.package.goal.state.pos[env_index]) * self.shaping_factor
            self.package.on_goal[env_index] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            self.package.dist_to_goal = torch.linalg.vector_norm(self.package.state.pos - self.package.goal.state.pos, dim=1)
            self.package.on_goal = self.world.is_overlapping(self.package, self.package.goal)
            self.package.color = torch.tensor(Color.RED.value, device=self.world.device, dtype=torch.float32).repeat(self.world.batch_dim, 1)
            self.package.color[self.package.on_goal] = torch.tensor(Color.GREEN.value, device=self.world.device, dtype=torch.float32)
            package_shaping = self.package.dist_to_goal * self.shaping_factor
            self.rew[~self.package.on_goal] += self.package.global_shaping[~self.package.on_goal] - package_shaping[~self.package.on_goal]
            self.package.global_shaping = package_shaping
            self.rew[~self.package.on_goal] += self.package.global_shaping[~self.package.on_goal] - package_shaping[~self.package.on_goal]
            self.package.global_shaping = package_shaping
        return self.rew

    def observation(self, agent: Agent):
        return torch.cat([agent.state.pos, agent.state.vel, self.package.state.vel, self.package.state.pos - agent.state.pos, self.package.state.pos - self.package.goal.state.pos], dim=-1)

    def done(self):
        return self.package.on_goal

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop('n_agents', 4)
        self.line_length = kwargs.pop('line_length', 2)
        line_mass = kwargs.pop('line_mass', 30)
        self.desired_velocity = kwargs.pop('desired_velocity', 0.05)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        world = World(batch_dim, device)
        for i in range(n_agents):
            agent = Agent(name=f'agent_{i}', u_multiplier=0.6, shape=Sphere(0.03))
            world.add_agent(agent)
        self.line = Landmark(name='line', collide=True, rotatable=True, shape=Line(length=self.line_length), mass=line_mass, color=Color.BLACK)
        world.add_landmark(self.line)
        center = Landmark(name='center', shape=Sphere(radius=0.02), collide=False, color=Color.BLACK)
        world.add_landmark(center)
        return world

    def reset_world_at(self, env_index: int=None):
        for agent in self.world.agents:
            agent.set_pos(torch.zeros((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32).uniform_(-1.0, 1.0), batch_index=env_index)
        self.line.set_rot(torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-torch.pi / 2, torch.pi / 2), batch_index=env_index)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = (self.line.state.ang_vel.abs() - self.desired_velocity).abs()
        return -self.rew

    def observation(self, agent: Agent):
        line_end_1 = torch.cat([self.line_length / 2 * torch.cos(self.line.state.rot), self.line_length / 2 * torch.sin(self.line.state.rot)], dim=1)
        line_end_2 = -line_end_1
        return torch.cat([agent.state.pos, agent.state.vel, self.line.state.pos - agent.state.pos, line_end_1 - agent.state.pos, line_end_2 - agent.state.pos, self.line.state.rot % torch.pi, self.line.state.ang_vel.abs(), (self.line.state.ang_vel.abs() - self.desired_velocity).abs()], dim=-1)

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
        n_agents = kwargs.pop('n_agents', 4)
        self.n_packages = kwargs.pop('n_packages', 1)
        self.package_width = kwargs.pop('package_width', 0.15)
        self.package_length = kwargs.pop('package_length', 0.15)
        self.package_mass = kwargs.pop('package_mass', 50)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.shaping_factor = 100
        self.world_semidim = 1
        self.agent_radius = 0.03
        world = World(batch_dim, device, x_semidim=self.world_semidim + 2 * self.agent_radius + max(self.package_length, self.package_width), y_semidim=self.world_semidim + 2 * self.agent_radius + max(self.package_length, self.package_width))
        for i in range(n_agents):
            agent = Agent(name=f'agent_{i}', shape=Sphere(self.agent_radius), u_multiplier=0.6)
            world.add_agent(agent)
        goal = Landmark(name='goal', collide=False, shape=Sphere(radius=0.15), color=Color.LIGHT_GREEN)
        world.add_landmark(goal)
        self.packages = []
        for i in range(self.n_packages):
            package = Landmark(name=f'package {i}', collide=True, movable=True, mass=self.package_mass, shape=Box(length=self.package_length, width=self.package_width), color=Color.RED)
            package.goal = goal
            self.packages.append(package)
            world.add_landmark(package)
        return world

    def reset_world_at(self, env_index: int=None):
        ScenarioUtils.spawn_entities_randomly(self.world.agents, self.world, env_index, min_dist_between_entities=self.agent_radius * 2, x_bounds=(-self.world_semidim, self.world_semidim), y_bounds=(-self.world_semidim, self.world_semidim))
        agent_occupied_positions = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        if env_index is not None:
            agent_occupied_positions = agent_occupied_positions[env_index].unsqueeze(0)
        goal = self.world.landmarks[0]
        ScenarioUtils.spawn_entities_randomly([goal] + self.packages, self.world, env_index, min_dist_between_entities=max((package.shape.circumscribed_radius() + goal.shape.radius + 0.01 for package in self.packages)), x_bounds=(-self.world_semidim, self.world_semidim), y_bounds=(-self.world_semidim, self.world_semidim), occupied_positions=agent_occupied_positions)
        for package in self.packages:
            package.on_goal = self.world.is_overlapping(package, package.goal)
            if env_index is None:
                package.global_shaping = torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1) * self.shaping_factor
            else:
                package.global_shaping[env_index] = torch.linalg.vector_norm(package.state.pos[env_index] - package.goal.state.pos[env_index]) * self.shaping_factor

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            for package in self.packages:
                package.dist_to_goal = torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1)
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.color = torch.tensor(Color.RED.value, device=self.world.device, dtype=torch.float32).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(Color.GREEN.value, device=self.world.device, dtype=torch.float32)
                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal] += package.global_shaping[~package.on_goal] - package_shaping[~package.on_goal]
                package.global_shaping = package_shaping
        return self.rew

    def observation(self, agent: Agent):
        package_obs = []
        for package in self.packages:
            package_obs.append(package.state.pos - package.goal.state.pos)
            package_obs.append(package.state.pos - agent.state.pos)
            package_obs.append(package.state.vel)
            package_obs.append(package.on_goal.unsqueeze(-1))
        return torch.cat([agent.state.pos, agent.state.vel, *package_obs], dim=-1)

    def done(self):
        return torch.all(torch.stack([package.on_goal for package in self.packages], dim=1), dim=-1)

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_passages = kwargs.pop('n_passages', 1)
        self.shared_reward = kwargs.pop('shared_reward', False)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        assert self.n_passages >= 1 and self.n_passages <= 20
        self.shaping_factor = 100
        self.n_agents = 5
        self.agent_radius = 0.03333
        self.agent_spacing = 0.1
        self.passage_width = 0.2
        self.passage_length = 0.103
        self.visualize_semidims = False
        world = World(batch_dim, device, x_semidim=1, y_semidim=1)
        for i in range(self.n_agents):
            agent = Agent(name=f'agent_{i}', shape=Sphere(self.agent_radius), u_multiplier=0.7)
            world.add_agent(agent)
            goal = Landmark(name=f'goal {i}', collide=False, shape=Sphere(radius=self.agent_radius), color=Color.LIGHT_GREEN)
            agent.goal = goal
            world.add_landmark(goal)
        for i in range(int((2 * world.x_semidim + 2 * self.agent_radius) // self.passage_length)):
            removed = i < self.n_passages
            passage = Landmark(name=f'passage {i}', collide=not removed, movable=False, shape=Box(length=self.passage_length, width=self.passage_width), color=Color.RED, collision_filter=lambda e: not isinstance(e.shape, Box))
            world.add_landmark(passage)
        return world

    def reset_world_at(self, env_index: int=None):
        central_agent_pos = torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-1 + (3 * self.agent_radius + self.agent_spacing), 1 - (3 * self.agent_radius + self.agent_spacing)), torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-1 + (3 * self.agent_radius + self.agent_spacing), -(3 * self.agent_radius + self.agent_spacing) - self.passage_width / 2)], dim=1)
        central_goal_pos = torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-1 + (3 * self.agent_radius + self.agent_spacing), 1 - (3 * self.agent_radius + self.agent_spacing)), torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(3 * self.agent_radius + self.agent_spacing + self.passage_width / 2, 1 - (3 * self.agent_radius + self.agent_spacing))], dim=1)
        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        goals = [self.world.landmarks[i] for i in order]
        for i, goal in enumerate(goals):
            if i == self.n_agents - 1:
                goal.set_pos(central_goal_pos, batch_index=env_index)
            else:
                goal.set_pos(central_goal_pos + torch.tensor([[0.0 if i % 2 else self.agent_spacing if i == 0 else -self.agent_spacing, 0.0 if not i % 2 else self.agent_spacing if i == 1 else -self.agent_spacing]], device=self.world.device), batch_index=env_index)
        for i, agent in enumerate(agents):
            if i == self.n_agents - 1:
                agent.set_pos(central_agent_pos, batch_index=env_index)
            else:
                agent.set_pos(central_agent_pos + torch.tensor([[0.0 if i % 2 else self.agent_spacing if i == 0 else -self.agent_spacing, 0.0 if not i % 2 else self.agent_spacing if i == 1 else -self.agent_spacing]], device=self.world.device), batch_index=env_index)
            if env_index is None:
                agent.global_shaping = torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1) * self.shaping_factor
            else:
                agent.global_shaping[env_index] = torch.linalg.vector_norm(agent.state.pos[env_index] - agent.goal.state.pos[env_index]) * self.shaping_factor
        order = torch.randperm(len(self.world.landmarks[self.n_agents:])).tolist()
        passages = [self.world.landmarks[self.n_agents:][i] for i in order]
        for i, passage in enumerate(passages):
            if not passage.collide:
                passage.is_rendering[:] = False
            passage.set_pos(torch.tensor([-1 - self.agent_radius + self.passage_length / 2 + self.passage_length * i, 0.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if self.shared_reward:
            if is_first:
                self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
                for a in self.world.agents:
                    dist_to_goal = torch.linalg.vector_norm(a.state.pos - a.goal.state.pos, dim=1)
                    agent_shaping = dist_to_goal * self.shaping_factor
                    self.rew += a.global_shaping - agent_shaping
                    a.global_shaping = agent_shaping
        else:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            dist_to_goal = torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1)
            agent_shaping = dist_to_goal * self.shaping_factor
            self.rew += agent.global_shaping - agent_shaping
            agent.global_shaping = agent_shaping
        if agent.collide:
            for a in self.world.agents:
                if a != agent:
                    self.rew[self.world.is_overlapping(a, agent)] -= 10
            for landmark in self.world.landmarks[self.n_agents:]:
                if landmark.collide:
                    self.rew[self.world.is_overlapping(agent, landmark)] -= 10
        return self.rew

    def observation(self, agent: Agent):
        passage_obs = []
        passages = self.world.landmarks[self.n_agents:]
        for passage in passages:
            if not passage.collide:
                passage_obs.append(passage.state.pos - agent.state.pos)
        return torch.cat([agent.state.pos, agent.state.vel, agent.goal.state.pos - agent.state.pos, *passage_obs], dim=-1)

    def done(self):
        return torch.all(torch.stack([torch.linalg.vector_norm(a.state.pos - a.goal.state.pos, dim=1) <= a.shape.radius / 2 for a in self.world.agents], dim=1), dim=1)

    def extra_render(self, env_index: int=0):
        from vmas.simulator import rendering
        geoms = []
        for i in range(4):
            geom = Line(length=2 + self.agent_radius * 2).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(0.0 if i % 2 else self.world.x_semidim + self.agent_radius if i == 0 else -self.world.x_semidim - self.agent_radius, 0.0 if not i % 2 else self.world.x_semidim + self.agent_radius if i == 1 else -self.world.x_semidim - self.agent_radius)
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop('n_agents', 3)
        self.package_mass = kwargs.pop('package_mass', 5)
        self.random_package_pos_on_line = kwargs.pop('random_package_pos_on_line', True)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        assert self.n_agents > 1
        self.line_length = 0.8
        self.agent_radius = 0.03
        self.shaping_factor = 100
        self.fall_reward = -10
        self.visualize_semidims = False
        world = World(batch_dim, device, gravity=(0.0, -0.05), y_semidim=1)
        for i in range(self.n_agents):
            agent = Agent(name=f'agent_{i}', shape=Sphere(self.agent_radius), u_multiplier=0.7)
            world.add_agent(agent)
        goal = Landmark(name='goal', collide=False, shape=Sphere(), color=Color.LIGHT_GREEN)
        world.add_landmark(goal)
        self.package = Landmark(name='package', collide=True, movable=True, shape=Sphere(), mass=self.package_mass, color=Color.RED)
        self.package.goal = goal
        world.add_landmark(self.package)
        self.line = Landmark(name='line', shape=Line(length=self.line_length), collide=True, movable=True, rotatable=True, mass=5, color=Color.BLACK)
        world.add_landmark(self.line)
        self.floor = Landmark(name='floor', collide=True, shape=Box(length=10, width=1), color=Color.WHITE)
        world.add_landmark(self.floor)
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.ground_rew = self.pos_rew.clone()
        return world

    def reset_world_at(self, env_index: int=None):
        goal_pos = torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-1.0, 1.0), torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(0.0, self.world.y_semidim)], dim=1)
        line_pos = torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-1.0 + self.line_length / 2, 1.0 - self.line_length / 2), torch.full((1, 1) if env_index is not None else (self.world.batch_dim, 1), -self.world.y_semidim + self.agent_radius * 2, device=self.world.device, dtype=torch.float32)], dim=1)
        package_rel_pos = torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.line_length / 2 + self.package.shape.radius if self.random_package_pos_on_line else 0.0, self.line_length / 2 - self.package.shape.radius if self.random_package_pos_on_line else 0.0), torch.full((1, 1) if env_index is not None else (self.world.batch_dim, 1), self.package.shape.radius, device=self.world.device, dtype=torch.float32)], dim=1)
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(line_pos + torch.tensor([-(self.line_length - agent.shape.radius) / 2 + i * (self.line_length - agent.shape.radius) / (self.n_agents - 1), -self.agent_radius * 2], device=self.world.device, dtype=torch.float32), batch_index=env_index)
        self.line.set_pos(line_pos, batch_index=env_index)
        self.package.goal.set_pos(goal_pos, batch_index=env_index)
        self.line.set_rot(torch.zeros(1, device=self.world.device, dtype=torch.float32), batch_index=env_index)
        self.package.set_pos(line_pos + package_rel_pos, batch_index=env_index)
        self.floor.set_pos(torch.tensor([0, -self.world.y_semidim - self.floor.shape.width / 2 - self.agent_radius], device=self.world.device), batch_index=env_index)
        self.compute_on_the_ground()
        if env_index is None:
            self.global_shaping = torch.linalg.vector_norm(self.package.state.pos - self.package.goal.state.pos, dim=1) * self.shaping_factor
        else:
            self.global_shaping[env_index] = torch.linalg.vector_norm(self.package.state.pos[env_index] - self.package.goal.state.pos[env_index]) * self.shaping_factor

    def compute_on_the_ground(self):
        self.on_the_ground = self.world.is_overlapping(self.line, self.floor) + self.world.is_overlapping(self.package, self.floor)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.pos_rew[:] = 0
            self.ground_rew[:] = 0
            self.compute_on_the_ground()
            self.package_dist = torch.linalg.vector_norm(self.package.state.pos - self.package.goal.state.pos, dim=1)
            self.ground_rew[self.on_the_ground] = self.fall_reward
            global_shaping = self.package_dist * self.shaping_factor
            self.pos_rew = self.global_shaping - global_shaping
            self.global_shaping = global_shaping
        return self.ground_rew + self.pos_rew

    def observation(self, agent: Agent):
        return torch.cat([agent.state.pos, agent.state.vel, agent.state.pos - self.package.state.pos, agent.state.pos - self.line.state.pos, self.package.state.pos - self.package.goal.state.pos, self.package.state.vel, self.line.state.vel, self.line.state.ang_vel, self.line.state.rot % torch.pi], dim=-1)

    def done(self):
        return self.on_the_ground + self.world.is_overlapping(self.package, self.package.goal)

    def info(self, agent: Agent):
        info = {'pos_rew': self.pos_rew, 'ground_rew': self.ground_rew}
        return info

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop('n_agents', 5)
        self.with_joints = kwargs.pop('joints', True)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.agent_dist = 0.1
        self.agent_radius = 0.04
        world = World(batch_dim, device, dt=0.1, drag=0.25, substeps=5, collision_force=500)
        for i in range(self.n_agents):
            agent = Agent(name=f'agent_{i}', shape=Sphere(radius=self.agent_radius), u_multiplier=0.7, rotatable=True)
            world.add_agent(agent)
        if self.with_joints:
            for i in range(self.n_agents - 1):
                joint = Joint(world.agents[i], world.agents[i + 1], anchor_a=(1, 0), anchor_b=(-1, 0), dist=self.agent_dist, rotate_a=True, rotate_b=True, collidable=True, width=0, mass=1)
                world.add_joint(joint)
            landmark = Landmark(name='joined landmark', collide=True, movable=True, rotatable=True, shape=Box(length=self.agent_radius * 2, width=0.3), color=Color.GREEN)
            world.add_landmark(landmark)
            joint = Joint(world.agents[-1], landmark, anchor_a=(1, 0), anchor_b=(-1, 0), dist=self.agent_dist, rotate_a=False, rotate_b=False, collidable=True, width=0, mass=1)
            world.add_joint(joint)
        for i in range(5):
            landmark = Landmark(name=f'landmark {i}', collide=True, movable=True, rotatable=True, shape=Box(length=0.3, width=0.1), color=Color.RED)
            world.add_landmark(landmark)
        floor = Landmark(name='floor', collide=True, movable=False, shape=Line(length=2), color=Color.BLACK)
        world.add_landmark(floor)
        return world

    def reset_world_at(self, env_index: int=None):
        for i, agent in enumerate(self.world.agents + [self.world.landmarks[self.n_agents - 1]]):
            agent.set_pos(torch.tensor([-0.2 + (self.agent_dist + 2 * self.agent_radius) * i, 1.0], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        for i, landmark in enumerate(self.world.landmarks[self.n_agents + 1 if self.with_joints else 0:-1]):
            landmark.set_pos(torch.tensor([0.2 if i % 2 else -0.2, 0.6 - 0.3 * i], dtype=torch.float32, device=self.world.device), batch_index=env_index)
            landmark.set_rot(torch.tensor([torch.pi / 4 if i % 2 else -torch.pi / 4], dtype=torch.float32, device=self.world.device), batch_index=env_index)
        floor = self.world.landmarks[-1]
        floor.set_pos(torch.tensor([0, -1], dtype=torch.float32, device=self.world.device), batch_index=env_index)

    def reward(self, agent: Agent):
        dist2 = torch.linalg.vector_norm(agent.state.pos - self.world.landmarks[-1].state.pos, dim=1)
        return -dist2

    def observation(self, agent: Agent):
        return torch.cat([agent.state.pos, agent.state.vel] + [landmark.state.pos - agent.state.pos for landmark in self.world.landmarks], dim=-1)

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_good_agents = kwargs.pop('num_good_agents', 1)
        num_adversaries = kwargs.pop('num_adversaries', 3)
        num_landmarks = kwargs.pop('num_landmarks', 2)
        self.shape_agent_rew = kwargs.pop('shape_agent_rew', False)
        self.shape_adversary_rew = kwargs.pop('shape_adversary_rew', False)
        self.agents_share_rew = kwargs.pop('agents_share_rew', False)
        self.adversaries_share_rew = kwargs.pop('adversaries_share_rew', True)
        self.observe_same_team = kwargs.pop('observe_same_team', True)
        self.observe_pos = kwargs.pop('observe_pos', True)
        self.observe_vel = kwargs.pop('observe_vel', True)
        self.bound = kwargs.pop('bound', 1.0)
        self.respawn_at_catch = kwargs.pop('respawn_at_catch', False)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.visualize_semidims = False
        world = World(batch_dim=batch_dim, device=device, x_semidim=self.bound, y_semidim=self.bound, substeps=10, collision_force=500)
        num_agents = num_adversaries + num_good_agents
        self.adversary_radius = 0.075
        for i in range(num_agents):
            adversary = True if i < num_adversaries else False
            name = f'adversary_{i}' if adversary else f'agent_{i - num_adversaries}'
            agent = Agent(name=name, collide=True, shape=Sphere(radius=self.adversary_radius if adversary else 0.05), u_multiplier=3.0 if adversary else 4.0, max_speed=1.0 if adversary else 1.3, color=Color.RED if adversary else Color.GREEN, adversary=adversary)
            world.add_agent(agent)
        for i in range(num_landmarks):
            landmark = Landmark(name=f'landmark {i}', collide=True, shape=Sphere(radius=0.2), color=Color.BLACK)
            world.add_landmark(landmark)
        return world

    def reset_world_at(self, env_index: int=None):
        for agent in self.world.agents:
            agent.set_pos(torch.zeros((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32).uniform_(-self.bound, self.bound), batch_index=env_index)
        for landmark in self.world.landmarks:
            landmark.set_pos(torch.zeros((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32).uniform_(-(self.bound - 0.1), self.bound - 0.1), batch_index=env_index)

    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.pos - agent2.state.pos
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        dist_min = agent1.shape.radius + agent2.shape.radius
        return dist < dist_min

    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            for a in self.world.agents:
                a.rew = self.adversary_reward(a) if a.adversary else self.agent_reward(a)
            self.agents_rew = torch.stack([a.rew for a in self.good_agents()], dim=-1).sum(-1)
            self.adverary_rew = torch.stack([a.rew for a in self.adversaries()], dim=-1).sum(-1)
            if self.respawn_at_catch:
                for a in self.good_agents():
                    for adv in self.adversaries():
                        coll = self.is_collision(a, adv)
                        a.state.pos[coll] = torch.zeros((self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32).uniform_(-self.bound, self.bound)[coll]
                        a.state.vel[coll] = 0.0
        if agent.adversary:
            if self.adversaries_share_rew:
                return self.adverary_rew
            else:
                return agent.rew
        elif self.agents_share_rew:
            return self.agents_rew
        else:
            return agent.rew

    def agent_reward(self, agent: Agent):
        rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
        adversaries = self.adversaries()
        if self.shape_agent_rew:
            for adv in adversaries:
                rew += 0.1 * torch.linalg.vector_norm(agent.state.pos - adv.state.pos, dim=-1)
        if agent.collide:
            for a in adversaries:
                rew[self.is_collision(a, agent)] -= 10
        return rew

    def adversary_reward(self, agent: Agent):
        rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
        agents = self.good_agents()
        if self.shape_adversary_rew:
            rew -= 0.1 * torch.min(torch.stack([torch.linalg.vector_norm(a.state.pos - agent.state.pos, dim=-1) for a in agents], dim=-1), dim=-1)[0]
        if agent.collide:
            for ag in agents:
                rew[self.is_collision(ag, agent)] += 10
        return rew

    def observation(self, agent: Agent):
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)
        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            if agent.adversary and (not other.adversary):
                other_pos.append(other.state.pos - agent.state.pos)
                other_vel.append(other.state.vel)
            elif not agent.adversary and (not other.adversary) and self.observe_same_team:
                other_pos.append(other.state.pos - agent.state.pos)
                other_vel.append(other.state.vel)
            elif not agent.adversary and other.adversary:
                other_pos.append(other.state.pos - agent.state.pos)
            elif agent.adversary and other.adversary and self.observe_same_team:
                other_pos.append(other.state.pos - agent.state.pos)
        return torch.cat([*([agent.state.vel] if self.observe_vel else []), *([agent.state.pos] if self.observe_pos else []), *entity_pos, *other_pos, *other_vel], dim=-1)

    def extra_render(self, env_index: int=0):
        from vmas.simulator import rendering
        geoms = []
        for i in range(4):
            geom = Line(length=2 * (self.bound - self.adversary_radius + self.adversary_radius * 2)).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(0.0 if i % 2 else self.bound + self.adversary_radius if i == 0 else -self.bound - self.adversary_radius, 0.0 if not i % 2 else self.bound + self.adversary_radius if i == 1 else -self.bound - self.adversary_radius)
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)
        return geoms

