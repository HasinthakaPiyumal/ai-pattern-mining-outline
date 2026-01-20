# Cluster 19

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

