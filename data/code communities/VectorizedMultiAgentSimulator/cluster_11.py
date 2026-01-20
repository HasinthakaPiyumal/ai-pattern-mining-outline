# Cluster 11

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop('n_agents', 5)
        self.n_targets = kwargs.pop('n_targets', 7)
        self.x_semidim = kwargs.pop('x_semidim', 1)
        self.y_semidim = kwargs.pop('y_semidim', 1)
        self._min_dist_between_entities = kwargs.pop('min_dist_between_entities', 0.2)
        self._lidar_range = kwargs.pop('lidar_range', 0.35)
        self._covering_range = kwargs.pop('covering_range', 0.25)
        self.use_agent_lidar = kwargs.pop('use_agent_lidar', False)
        self.n_lidar_rays_entities = kwargs.pop('n_lidar_rays_entities', 15)
        self.n_lidar_rays_agents = kwargs.pop('n_lidar_rays_agents', 12)
        self._agents_per_target = kwargs.pop('agents_per_target', 2)
        self.targets_respawn = kwargs.pop('targets_respawn', True)
        self.shared_reward = kwargs.pop('shared_reward', False)
        self.agent_collision_penalty = kwargs.pop('agent_collision_penalty', 0)
        self.covering_rew_coeff = kwargs.pop('covering_rew_coeff', 1.0)
        self.time_penalty = kwargs.pop('time_penalty', 0)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self._comms_range = self._lidar_range
        self.min_collision_distance = 0.005
        self.agent_radius = 0.05
        self.target_radius = self.agent_radius
        self.viewer_zoom = 1
        self.target_color = Color.GREEN
        world = World(batch_dim, device, x_semidim=self.x_semidim, y_semidim=self.y_semidim, collision_force=500, substeps=2, drag=0.25)
        entity_filter_agents: Callable[[Entity], bool] = lambda e: e.name.startswith('agent')
        entity_filter_targets: Callable[[Entity], bool] = lambda e: e.name.startswith('target')
        for i in range(self.n_agents):
            agent = Agent(name=f'agent_{i}', collide=True, shape=Sphere(radius=self.agent_radius), sensors=[Lidar(world, n_rays=self.n_lidar_rays_entities, max_range=self._lidar_range, entity_filter=entity_filter_targets, render_color=Color.GREEN)] + ([Lidar(world, angle_start=0.05, angle_end=2 * torch.pi + 0.05, n_rays=self.n_lidar_rays_agents, max_range=self._lidar_range, entity_filter=entity_filter_agents, render_color=Color.BLUE)] if self.use_agent_lidar else []))
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            agent.covering_reward = agent.collision_rew.clone()
            world.add_agent(agent)
        self._targets = []
        for i in range(self.n_targets):
            target = Landmark(name=f'target_{i}', collide=True, movable=False, shape=Sphere(radius=self.target_radius), color=self.target_color)
            world.add_landmark(target)
            self._targets.append(target)
        self.covered_targets = torch.zeros(batch_dim, self.n_targets, device=device)
        self.shared_covering_rew = torch.zeros(batch_dim, device=device)
        return world

    def reset_world_at(self, env_index: int=None):
        placable_entities = self._targets[:self.n_targets] + self.world.agents
        if env_index is None:
            self.all_time_covered_targets = torch.full((self.world.batch_dim, self.n_targets), False, device=self.world.device)
        else:
            self.all_time_covered_targets[env_index] = False
        ScenarioUtils.spawn_entities_randomly(entities=placable_entities, world=self.world, env_index=env_index, min_dist_between_entities=self._min_dist_between_entities, x_bounds=(-self.world.x_semidim, self.world.x_semidim), y_bounds=(-self.world.y_semidim, self.world.y_semidim))
        for target in self._targets[self.n_targets:]:
            target.set_pos(self.get_outside_pos(env_index), batch_index=env_index)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]
        if is_first:
            self.time_rew = torch.full((self.world.batch_dim,), self.time_penalty, device=self.world.device)
            self.agents_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
            self.targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)
            self.agents_targets_dists = torch.cdist(self.agents_pos, self.targets_pos)
            self.agents_per_target = torch.sum((self.agents_targets_dists < self._covering_range).type(torch.int), dim=1)
            self.covered_targets = self.agents_per_target >= self._agents_per_target
            self.shared_covering_rew[:] = 0
            for a in self.world.agents:
                self.shared_covering_rew += self.agent_reward(a)
            self.shared_covering_rew[self.shared_covering_rew != 0] /= 2
        agent.collision_rew[:] = 0
        for a in self.world.agents:
            if a != agent:
                agent.collision_rew[self.world.get_distance(a, agent) < self.min_collision_distance] += self.agent_collision_penalty
        if is_last:
            if self.targets_respawn:
                occupied_positions_agents = [self.agents_pos]
                for i, target in enumerate(self._targets):
                    occupied_positions_targets = [o.state.pos.unsqueeze(1) for o in self._targets if o is not target]
                    occupied_positions = torch.cat(occupied_positions_agents + occupied_positions_targets, dim=1)
                    pos = ScenarioUtils.find_random_pos_for_entity(occupied_positions, env_index=None, world=self.world, min_dist_between_entities=self._min_dist_between_entities, x_bounds=(-self.world.x_semidim, self.world.x_semidim), y_bounds=(-self.world.y_semidim, self.world.y_semidim))
                    target.state.pos[self.covered_targets[:, i]] = pos[self.covered_targets[:, i]].squeeze(1)
            else:
                self.all_time_covered_targets += self.covered_targets
                for i, target in enumerate(self._targets):
                    target.state.pos[self.covered_targets[:, i]] = self.get_outside_pos(None)[self.covered_targets[:, i]]
        covering_rew = agent.covering_reward if not self.shared_reward else self.shared_covering_rew
        return agent.collision_rew + covering_rew + self.time_rew

    def get_outside_pos(self, env_index):
        return torch.empty((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)

    def agent_reward(self, agent):
        agent_index = self.world.agents.index(agent)
        agent.covering_reward[:] = 0
        targets_covered_by_agent = self.agents_targets_dists[:, agent_index] < self._covering_range
        num_covered_targets_covered_by_agent = (targets_covered_by_agent * self.covered_targets).sum(dim=-1)
        agent.covering_reward += num_covered_targets_covered_by_agent * self.covering_rew_coeff
        return agent.covering_reward

    def observation(self, agent: Agent):
        lidar_1_measures = agent.sensors[0].measure()
        return torch.cat([agent.state.pos, agent.state.vel, lidar_1_measures] + ([agent.sensors[1].measure()] if self.use_agent_lidar else []), dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {'covering_reward': agent.covering_reward if not self.shared_reward else self.shared_covering_rew, 'collision_rew': agent.collision_rew, 'targets_covered': self.covered_targets.sum(-1)}
        return info

    def done(self):
        return self.all_time_covered_targets.all(dim=-1)

    def extra_render(self, env_index: int=0) -> 'List[Geom]':
        from vmas.simulator import rendering
        geoms: List[Geom] = []
        for target in self._targets:
            range_circle = rendering.make_circle(self._covering_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*self.target_color.value)
            geoms.append(range_circle)
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(agent1.state.pos - agent2.state.pos, dim=-1)
                if agent_dist[env_index] <= self._comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(agent1.state.pos[env_index], agent2.state.pos[env_index], width=1)
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop('n_agents', 4)
        n_obstacles = kwargs.pop('n_obstacles', 5)
        self._min_dist_between_entities = kwargs.pop('min_dist_between_entities', 0.15)
        self.n_lidar_rays = kwargs.pop('n_lidar_rays', 12)
        self.collision_reward = kwargs.pop('collision_reward', -0.1)
        self.dist_shaping_factor = kwargs.pop('dist_shaping_factor', 1)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.plot_grid = True
        self.desired_distance = 0.1
        self.min_collision_distance = 0.005
        self.x_dim = 1
        self.y_dim = 1
        world = World(batch_dim, device, collision_force=400, substeps=5)
        self._target = Agent(name='target', collide=True, color=Color.GREEN, render_action=True, action_script=self.action_script_creator())
        world.add_agent(self._target)
        goal_entity_filter: Callable[[Entity], bool] = lambda e: not isinstance(e, Agent)
        for i in range(n_agents):
            agent = Agent(name=f'agent_{i}', collide=True, sensors=[Lidar(world, n_rays=self.n_lidar_rays, max_range=0.2, entity_filter=goal_entity_filter)], render_action=True)
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            agent.dist_rew = agent.collision_rew.clone()
            world.add_agent(agent)
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(name=f'obstacle_{i}', collide=True, movable=False, shape=Sphere(radius=0.1), color=Color.RED)
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)
        return world

    def action_script_creator(self):

        def action_script(agent, world):
            t = self.t / 30
            agent.action.u = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        return action_script

    def reset_world_at(self, env_index: int=None):
        target_pos = torch.zeros((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32)
        target_pos[:, Y] = -self.y_dim
        self._target.set_pos(target_pos, batch_index=env_index)
        ScenarioUtils.spawn_entities_randomly(self.obstacles + self.world.policy_agents, self.world, env_index, self._min_dist_between_entities, x_bounds=(-self.x_dim, self.x_dim), y_bounds=(-self.y_dim, self.y_dim), occupied_positions=target_pos.unsqueeze(1))
        for agent in self.world.policy_agents:
            if env_index is None:
                agent.distance_shaping = (torch.stack([torch.linalg.vector_norm(agent.state.pos - a.state.pos, dim=-1) for a in self.world.agents if a != agent], dim=1) - self.desired_distance).pow(2).mean(-1) * self.dist_shaping_factor
            else:
                agent.distance_shaping[env_index] = (torch.stack([torch.linalg.vector_norm(agent.state.pos[env_index] - a.state.pos[env_index]) for a in self.world.agents if a != agent], dim=0) - self.desired_distance).pow(2).mean(-1) * self.dist_shaping_factor
        if env_index is None:
            self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
        else:
            self.t[env_index] = 0

    def reward(self, agent: Agent):
        is_first = self.world.policy_agents.index(agent) == 0
        if is_first:
            self.t += 1
            if self.collision_reward != 0:
                for a in self.world.policy_agents:
                    a.collision_rew[:] = 0
                for i, a in enumerate(self.world.agents):
                    for j, b in enumerate(self.world.agents):
                        if j <= i:
                            continue
                        collision = self.world.get_distance(a, b) <= self.min_collision_distance
                        if a.action_script is None:
                            a.collision_rew[collision] += self.collision_reward
                        if b.action_script is None:
                            b.collision_rew[collision] += self.collision_reward
        agents_dist_shaping = (torch.stack([torch.linalg.vector_norm(agent.state.pos - a.state.pos, dim=-1) for a in self.world.agents if a != agent], dim=1) - self.desired_distance).pow(2).mean(-1) * self.dist_shaping_factor
        agent.dist_rew = agent.distance_shaping - agents_dist_shaping
        agent.distance_shaping = agents_dist_shaping
        return agent.collision_rew + agent.dist_rew

    def observation(self, agent: Agent):
        return torch.cat([agent.state.pos, agent.state.vel, agent.state.pos - self._target.state.pos, agent.sensors[0].measure()], dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {'agent_collision_rew': agent.collision_rew, 'agent_distance_rew': agent.dist_rew}
        return info

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop('n_agents', 4)
        self.collisions = kwargs.pop('collisions', True)
        self.world_spawning_x = kwargs.pop('world_spawning_x', 1)
        self.world_spawning_y = kwargs.pop('world_spawning_y', 1)
        self.enforce_bounds = kwargs.pop('enforce_bounds', False)
        self.agents_with_same_goal = kwargs.pop('agents_with_same_goal', 1)
        self.split_goals = kwargs.pop('split_goals', False)
        self.observe_all_goals = kwargs.pop('observe_all_goals', False)
        self.lidar_range = kwargs.pop('lidar_range', 0.35)
        self.agent_radius = kwargs.pop('agent_radius', 0.1)
        self.comms_range = kwargs.pop('comms_range', 0)
        self.n_lidar_rays = kwargs.pop('n_lidar_rays', 12)
        self.shared_rew = kwargs.pop('shared_rew', True)
        self.pos_shaping_factor = kwargs.pop('pos_shaping_factor', 1)
        self.final_reward = kwargs.pop('final_reward', 0.01)
        self.agent_collision_penalty = kwargs.pop('agent_collision_penalty', -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005
        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
        else:
            self.x_semidim = None
            self.y_semidim = None
        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert not self.collisions, 'If agents share goals they cannot be collidables'
        if self.split_goals:
            assert self.n_agents % 2 == 0 and self.agents_with_same_goal == self.n_agents // 2, 'Splitting the goals is allowed when the agents are even and half the team has the same goal'
        world = World(batch_dim, device, substeps=2, x_semidim=self.x_semidim, y_semidim=self.y_semidim)
        known_colors = [(0.22, 0.49, 0.72), (1.0, 0.5, 0), (0.3, 0.69, 0.29), (0.97, 0.51, 0.75), (0.6, 0.31, 0.64), (0.89, 0.1, 0.11), (0.87, 0.87, 0)]
        colors = torch.randn((max(self.n_agents - len(known_colors), 0), 3), device=device)
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)
        for i in range(self.n_agents):
            color = known_colors[i] if i < len(known_colors) else colors[i - len(known_colors)]
            agent = Agent(name=f'agent_{i}', collide=self.collisions, color=color, shape=Sphere(radius=self.agent_radius), render_action=True, sensors=[Lidar(world, n_rays=self.n_lidar_rays, max_range=self.lidar_range, entity_filter=entity_filter_agents)] if self.collisions else None)
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)
            goal = Landmark(name=f'goal {i}', collide=False, color=color)
            world.add_landmark(goal)
            agent.goal = goal
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        return world

    def reset_world_at(self, env_index: int=None):
        ScenarioUtils.spawn_entities_randomly(self.world.agents, self.world, env_index, self.min_distance_between_entities, (-self.world_spawning_x, self.world_spawning_x), (-self.world_spawning_y, self.world_spawning_y))
        occupied_positions = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)
        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(occupied_positions=occupied_positions, env_index=env_index, world=self.world, min_dist_between_entities=self.min_distance_between_entities, x_bounds=(-self.world_spawning_x, self.world_spawning_x), y_bounds=(-self.world_spawning_y, self.world_spawning_y))
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)
        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i
            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)
            if env_index is None:
                agent.pos_shaping = torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1) * self.pos_shaping_factor
            else:
                agent.pos_shaping[env_index] = torch.linalg.vector_norm(agent.state.pos[env_index] - agent.goal.state.pos[env_index]) * self.pos_shaping_factor

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0
            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0
            self.all_goal_reached = torch.all(torch.stack([a.on_goal for a in self.world.agents], dim=-1), dim=-1)
            self.final_rew[self.all_goal_reached] = self.final_reward
            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[distance <= self.min_collision_distance] += self.agent_collision_penalty
                        b.agent_collision_rew[distance <= self.min_collision_distance] += self.agent_collision_penalty
        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=-1)
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius
        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        return torch.cat([agent.state.pos, agent.state.vel] + goal_poses + ([agent.sensors[0]._max_range - agent.sensors[0].measure()] if self.collisions else []), dim=-1)

    def done(self):
        return torch.stack([torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=-1) < agent.shape.radius for agent in self.world.agents], dim=-1).all(-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'pos_rew': self.pos_rew if self.shared_rew else agent.pos_rew, 'final_rew': self.final_rew, 'agent_collisions': agent.agent_collision_rew}

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
        n_agents = kwargs.pop('n_agents', 4)
        self.energy_coeff = kwargs.pop('energy_coeff', DEFAULT_ENERGY_COEFF)
        self.start_same_point = kwargs.pop('start_same_point', False)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.agent_radius = 0.05
        self.goal_radius = 0.03
        world = World(batch_dim, device)
        for i in range(n_agents):
            agent = Agent(name=f'agent_{i}', collide=False, shape=Sphere(radius=self.agent_radius))
            world.add_agent(agent)
        goal = Landmark(name='goal', collide=False, shape=Sphere(radius=self.goal_radius), color=Color.GREEN)
        world.add_landmark(goal)
        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.energy_rew = self.pos_rew.clone()
        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        return world

    def reset_world_at(self, env_index: int=None):
        if self.start_same_point:
            for agent in self.world.agents:
                agent.set_pos(torch.zeros((1, 2) if env_index is not None else (self.world.batch_dim, 2), device=self.world.device, dtype=torch.float), batch_index=env_index)
            ScenarioUtils.spawn_entities_randomly(self.world.landmarks, self.world, env_index, self.goal_radius + self.agent_radius + 0.01, x_bounds=(-1, 1), y_bounds=(-1, 1), occupied_positions=torch.zeros(1 if env_index is not None else self.world.batch_dim, 1, 2, device=self.world.device, dtype=torch.float))
        else:
            ScenarioUtils.spawn_entities_randomly(self.world.policy_agents + self.world.landmarks, self.world, env_index, self.goal_radius + self.agent_radius + 0.01, x_bounds=(-1, 1), y_bounds=(-1, 1))
        for landmark in self.world.landmarks:
            if env_index is None:
                landmark.eaten = torch.full((self.world.batch_dim,), False, device=self.world.device)
                landmark.reset_render()
                self._done[:] = False
            else:
                landmark.eaten[env_index] = False
                landmark.is_rendering[env_index] = True
                self._done[env_index] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]
        if is_first:
            self.any_eaten = self._done = torch.any(torch.stack([torch.linalg.vector_norm(a.state.pos - self.world.landmarks[0].state.pos, dim=1) < a.shape.radius + self.world.landmarks[0].shape.radius for a in self.world.agents], dim=1), dim=-1)
        self.pos_rew[:] = 0
        self.pos_rew[self.any_eaten * ~self.world.landmarks[0].eaten] = 1
        if is_last:
            self.world.landmarks[0].eaten[self.any_eaten] = True
            self.world.landmarks[0].is_rendering[self.any_eaten] = False
        if is_first:
            self.energy_rew = self.energy_coeff * -torch.stack([torch.linalg.vector_norm(a.action.u, dim=-1) / math.sqrt(self.world.dim_p * (a.u_range * a.u_multiplier) ** 2) for a in self.world.agents], dim=1).sum(-1)
        rew = self.pos_rew + self.energy_rew
        return rew

    def observation(self, agent: Agent):
        return torch.cat([agent.state.pos, agent.state.vel, self.world.landmarks[0].state.pos - agent.state.pos, self.world.landmarks[0].eaten.unsqueeze(-1)], dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {'pos_rew': self.pos_rew, 'energy_rew': self.energy_rew}
        return info

    def done(self):
        return self._done

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Drone example scenario
        Run this file to try it out.

        You can control the three input torques using left/right arrows, up/down arrows, and m/n.
        """
        self.plot_grid = True
        self.n_agents = kwargs.pop('n_agents', 2)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        world = World(batch_dim, device, substeps=10)
        for i in range(self.n_agents):
            agent = Agent(name=f'drone_{i}', collide=True, render_action=True, u_range=[1e-05, 1e-05, 1e-05], u_multiplier=[1, 1, 1], action_size=3, dynamics=Drone(world, integration='rk4'))
            world.add_agent(agent)
        return world

    def reset_world_at(self, env_index: int=None):
        ScenarioUtils.spawn_entities_randomly(self.world.agents, self.world, env_index, min_dist_between_entities=0.1, x_bounds=(-1, 1), y_bounds=(-1, 1))

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim, device=self.world.device)

    def process_action(self, agent: Agent):
        torque = agent.action.u
        thrust = torch.full((self.world.batch_dim, 1), agent.mass * agent.dynamics.g, device=self.world.device)
        agent.action.u = torch.cat([thrust, torque], dim=-1)

    def observation(self, agent: Agent):
        observations = [agent.state.pos, agent.state.vel]
        return torch.cat(observations, dim=-1)

    def done(self):
        return torch.any(torch.stack([agent.dynamics.needs_reset() for agent in self.world.agents], dim=-1), dim=-1)

    def extra_render(self, env_index: int=0) -> 'List[Geom]':
        geoms: List[Geom] = []
        for agent in self.world.agents:
            geoms.append(ScenarioUtils.plot_entity_rotation(agent, env_index, length=0.1))
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop('n_agents', 15)
        self.n_lines = kwargs.pop('n_lines', 15)
        self.n_boxes = kwargs.pop('n_boxes', 15)
        self.lidar = kwargs.pop('lidar', False)
        self.vectorized_lidar = kwargs.pop('vectorized_lidar', True)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.agent_radius = 0.05
        self.line_length = 0.3
        self.box_length = 0.2
        self.box_width = 0.1
        self.world_semidim = 1
        self.min_dist_between_entities = 0.1
        world = World(batch_dim, device, dt=0.1, drag=0.25, substeps=5, collision_force=500, x_semidim=self.world_semidim, y_semidim=self.world_semidim)
        for i in range(self.n_agents):
            agent = Agent(name=f'agent_{i}', shape=Sphere(radius=self.agent_radius), u_multiplier=0.7, rotatable=True, sensors=[Lidar(world, n_rays=16, max_range=0.5)] if self.lidar else [])
            world.add_agent(agent)
        for i in range(self.n_lines):
            landmark = Landmark(name=f'line {i}', collide=True, movable=True, rotatable=True, shape=Line(length=self.line_length), color=Color.BLACK)
            world.add_landmark(landmark)
        for i in range(self.n_boxes):
            landmark = Landmark(name=f'box {i}', collide=True, movable=True, rotatable=True, shape=Box(length=self.box_length, width=self.box_width), color=Color.RED)
            world.add_landmark(landmark)
        return world

    def reset_world_at(self, env_index: int=None):
        ScenarioUtils.spawn_entities_randomly(self.world.agents + self.world.landmarks, self.world, env_index, self.min_dist_between_entities, (-self.world_semidim, self.world_semidim), (-self.world_semidim, self.world_semidim))

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim, device=self.world.device)

    def observation(self, agent: Agent):
        return torch.zeros(self.world.batch_dim, 1, device=self.world.device) if not self.lidar else agent.sensors[0].measure(vectorized=self.vectorized_lidar)

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Kinematic bicycle model example scenario
        """
        self.n_agents = kwargs.pop('n_agents', 2)
        width = kwargs.pop('width', 0.1)
        l_f = kwargs.pop('l_f', 0.1)
        l_r = kwargs.pop('l_r', 0.1)
        max_steering_angle = kwargs.pop('max_steering_angle', torch.deg2rad(torch.tensor(30.0)))
        max_speed = kwargs.pop('max_speed', 1.0)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        world = World(batch_dim, device, substeps=10, collision_force=500)
        for i in range(self.n_agents):
            if i == 0:
                agent = Agent(name=f'bicycle_{i}', shape=Box(length=l_f + l_r, width=width), collide=True, render_action=True, u_range=[max_speed, max_steering_angle], u_multiplier=[1, 1], max_speed=max_speed, dynamics=KinematicBicycle(world, width=width, l_f=l_f, l_r=l_r, max_steering_angle=max_steering_angle, integration='euler'))
            else:
                agent = Agent(name=f'holo_rot_{i}', shape=Box(length=l_f + l_r, width=width), collide=True, render_action=True, u_range=[1, 1, 1], u_multiplier=[1, 1, 0.001], dynamics=HolonomicWithRotation())
            world.add_agent(agent)
        return world

    def reset_world_at(self, env_index: int=None):
        ScenarioUtils.spawn_entities_randomly(self.world.agents, self.world, env_index, min_dist_between_entities=0.1, x_bounds=(-1, 1), y_bounds=(-1, 1))

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim)

    def observation(self, agent: Agent):
        observations = [agent.state.pos, agent.state.vel]
        return torch.cat(observations, dim=-1)

    def extra_render(self, env_index: int=0) -> 'List[Geom]':
        geoms: List[Geom] = []
        for agent in self.world.agents:
            geoms.append(ScenarioUtils.plot_entity_rotation(agent, env_index, length=0.1))
        return geoms

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Differential drive example scenario
        Run this file to try it out

        The first agent has differential drive dynamics.
        You can control its forward input with the LEFT and RIGHT arrows.
        You can control its rotation with UP and DOWN.

        The second agent has standard vmas holonomic dynamics.
        You can control it with WASD
        You can control its rotation with Q and E.

        """
        self.plot_grid = True
        self.n_agents = kwargs.pop('n_agents', 2)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        world = World(batch_dim, device, substeps=10)
        for i in range(self.n_agents):
            if i == 0:
                agent = Agent(name=f'diff_drive_{i}', collide=True, render_action=True, u_range=[1, 1], u_multiplier=[1, 1], dynamics=DiffDrive(world, integration='rk4'))
            else:
                agent = Agent(name=f'holo_rot_{i}', collide=True, render_action=True, u_range=[1, 1, 1], u_multiplier=[1, 1, 0.001], dynamics=HolonomicWithRotation())
            world.add_agent(agent)
        return world

    def reset_world_at(self, env_index: int=None):
        ScenarioUtils.spawn_entities_randomly(self.world.agents, self.world, env_index, min_dist_between_entities=0.1, x_bounds=(-1, 1), y_bounds=(-1, 1))

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim)

    def observation(self, agent: Agent):
        observations = [agent.state.pos, agent.state.vel]
        return torch.cat(observations, dim=-1)

    def extra_render(self, env_index: int=0) -> 'List[Geom]':
        geoms: List[Geom] = []
        for agent in self.world.agents:
            geoms.append(ScenarioUtils.plot_entity_rotation(agent, env_index, length=0.1))
        return geoms

