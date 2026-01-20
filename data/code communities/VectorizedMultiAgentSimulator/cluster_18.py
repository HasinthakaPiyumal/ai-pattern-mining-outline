# Cluster 18

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

class HeuristicPolicy(BaseHeuristicPolicy):

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions is True, 'Heuristic for continuous actions only'
        index_line_extrema = 6
        pos_agent = observation[:, :2]
        pos_end2_agent = observation[:, index_line_extrema + 2:index_line_extrema + 4]
        pos_end2 = pos_end2_agent + pos_agent
        pos_end2_shifted = TorchUtils.rotate_vector(pos_end2, torch.tensor(torch.pi / 4, device=observation.device).expand(pos_end2.shape[0]))
        pos_end2_shifted_agent = pos_end2_shifted - pos_agent
        action_agent = torch.clamp(pos_end2_shifted_agent, min=-u_range, max=u_range)
        return action_agent

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

class World(TorchVectorizedObject):

    def __init__(self, batch_dim: int, device: torch.device, dt: float=0.1, substeps: int=1, drag: float=DRAG, linear_friction: float=LINEAR_FRICTION, angular_friction: float=ANGULAR_FRICTION, x_semidim: float=None, y_semidim: float=None, dim_c: int=0, collision_force: float=COLLISION_FORCE, joint_force: float=JOINT_FORCE, torque_constraint_force: float=TORQUE_CONSTRAINT_FORCE, contact_margin: float=0.001, gravity: Tuple[float, float]=(0.0, 0.0)):
        assert batch_dim > 0, f'Batch dim must be greater than 0, got {batch_dim}'
        super().__init__(batch_dim, device)
        self._agents = []
        self._landmarks = []
        self._x_semidim = x_semidim
        self._y_semidim = y_semidim
        self._dim_p = 2
        self._dim_c = dim_c
        self._dt = dt
        self._substeps = substeps
        self._sub_dt = self._dt / self._substeps
        self._drag = drag
        self._gravity = torch.tensor(gravity, device=self.device, dtype=torch.float32)
        self._linear_friction = linear_friction
        self._angular_friction = angular_friction
        self._collision_force = collision_force
        self._joint_force = joint_force
        self._contact_margin = contact_margin
        self._torque_constraint_force = torque_constraint_force
        self._joints = {}
        self._collidable_pairs = [{Sphere, Sphere}, {Sphere, Box}, {Sphere, Line}, {Line, Line}, {Line, Box}, {Box, Box}]
        self.entity_index_map = {}

    def add_agent(self, agent: Agent):
        """Only way to add agents to the world"""
        agent.batch_dim = self._batch_dim
        agent.to(self._device)
        agent._spawn(dim_c=self._dim_c, dim_p=self.dim_p)
        self._agents.append(agent)

    def add_landmark(self, landmark: Landmark):
        """Only way to add landmarks to the world"""
        landmark.batch_dim = self._batch_dim
        landmark.to(self._device)
        landmark._spawn(dim_c=self.dim_c, dim_p=self.dim_p)
        self._landmarks.append(landmark)

    def add_joint(self, joint: Joint):
        assert self._substeps > 1, 'For joints, world substeps needs to be more than 1'
        if joint.landmark is not None:
            self.add_landmark(joint.landmark)
        for constraint in joint.joint_constraints:
            self._joints.update({frozenset({constraint.entity_a.name, constraint.entity_b.name}): constraint})

    def reset(self, env_index: int):
        for e in self.entities:
            e._reset(env_index)

    def zero_grad(self):
        for e in self.entities:
            e.zero_grad()

    @property
    def agents(self) -> List[Agent]:
        return self._agents

    @property
    def landmarks(self) -> List[Landmark]:
        return self._landmarks

    @property
    def x_semidim(self):
        return self._x_semidim

    @property
    def dt(self):
        return self._dt

    @property
    def y_semidim(self):
        return self._y_semidim

    @property
    def dim_p(self):
        return self._dim_p

    @property
    def dim_c(self):
        return self._dim_c

    @property
    def joints(self):
        return self._joints.values()

    @property
    def entities(self) -> List[Entity]:
        return self._landmarks + self._agents

    @property
    def policy_agents(self) -> List[Agent]:
        return [agent for agent in self._agents if agent.action_script is None]

    @property
    def scripted_agents(self) -> List[Agent]:
        return [agent for agent in self._agents if agent.action_script is not None]

    def _cast_ray_to_box(self, box: Entity, ray_origin: Tensor, ray_direction: Tensor, max_range: float):
        """
        Inspired from https://tavianator.com/2011/ray_box.html
        Computes distance of ray originating from pos at angle to a box and sets distance to
        max_range if there is no intersection.
        """
        assert ray_origin.ndim == 2 and ray_direction.ndim == 1
        assert ray_origin.shape[0] == ray_direction.shape[0]
        assert isinstance(box.shape, Box)
        pos_origin = ray_origin - box.state.pos
        pos_aabb = TorchUtils.rotate_vector(pos_origin, -box.state.rot)
        ray_dir_world = torch.stack([torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1)
        ray_dir_aabb = TorchUtils.rotate_vector(ray_dir_world, -box.state.rot)
        tx1 = (-box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
        tx2 = (box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
        tx = torch.stack([tx1, tx2], dim=-1)
        tmin, _ = torch.min(tx, dim=-1)
        tmax, _ = torch.max(tx, dim=-1)
        ty1 = (-box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
        ty2 = (box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
        ty = torch.stack([ty1, ty2], dim=-1)
        tymin, _ = torch.min(ty, dim=-1)
        tymax, _ = torch.max(ty, dim=-1)
        tmin, _ = torch.max(torch.stack([tmin, tymin], dim=-1), dim=-1)
        tmax, _ = torch.min(torch.stack([tmax, tymax], dim=-1), dim=-1)
        intersect_aabb = tmin.unsqueeze(1) * ray_dir_aabb + pos_aabb
        intersect_world = TorchUtils.rotate_vector(intersect_aabb, box.state.rot) + box.state.pos
        collision = (tmax >= tmin) & (tmin > 0.0)
        dist = torch.linalg.norm(ray_origin - intersect_world, dim=1)
        dist[~collision] = max_range
        return dist

    def _cast_rays_to_box(self, box_pos, box_rot, box_length, box_width, ray_origin: Tensor, ray_direction: Tensor, max_range: float):
        """
        Inspired from https://tavianator.com/2011/ray_box.html
        Computes distance of ray originating from pos at angle to a box and sets distance to
        max_range if there is no intersection.
        """
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2
        assert ray_direction.shape[:-1] == batch_size
        assert box_pos.shape[:-2] == batch_size
        assert box_pos.shape[-1] == 2
        assert box_rot.shape[:-1] == batch_size
        assert box_width.shape[:-1] == batch_size
        assert box_length.shape[:-1] == batch_size
        num_angles = ray_direction.shape[-1]
        n_boxes = box_pos.shape[-2]
        ray_origin = ray_origin.unsqueeze(-2).unsqueeze(-2).expand(*batch_size, n_boxes, num_angles, 2)
        box_pos_expanded = box_pos.unsqueeze(-2).expand(*batch_size, n_boxes, num_angles, 2)
        ray_direction = ray_direction.unsqueeze(-2).expand(*batch_size, n_boxes, num_angles)
        box_rot_expanded = box_rot.unsqueeze(-1).expand(*batch_size, n_boxes, num_angles)
        box_width_expanded = box_width.unsqueeze(-1).expand(*batch_size, n_boxes, num_angles)
        box_length_expanded = box_length.unsqueeze(-1).expand(*batch_size, n_boxes, num_angles)
        pos_origin = ray_origin - box_pos_expanded
        pos_aabb = TorchUtils.rotate_vector(pos_origin, -box_rot_expanded)
        ray_dir_world = torch.stack([torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1)
        ray_dir_aabb = TorchUtils.rotate_vector(ray_dir_world, -box_rot_expanded)
        tx1 = (-box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
        tx2 = (box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
        tx = torch.stack([tx1, tx2], dim=-1)
        tmin, _ = torch.min(tx, dim=-1)
        tmax, _ = torch.max(tx, dim=-1)
        ty1 = (-box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
        ty2 = (box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
        ty = torch.stack([ty1, ty2], dim=-1)
        tymin, _ = torch.min(ty, dim=-1)
        tymax, _ = torch.max(ty, dim=-1)
        tmin, _ = torch.max(torch.stack([tmin, tymin], dim=-1), dim=-1)
        tmax, _ = torch.min(torch.stack([tmax, tymax], dim=-1), dim=-1)
        intersect_aabb = tmin.unsqueeze(-1) * ray_dir_aabb + pos_aabb
        intersect_world = TorchUtils.rotate_vector(intersect_aabb, box_rot_expanded) + box_pos_expanded
        collision = (tmax >= tmin) & (tmin > 0.0)
        dist = torch.linalg.norm(ray_origin - intersect_world, dim=-1)
        dist[~collision] = max_range
        return dist

    def _cast_ray_to_sphere(self, sphere: Entity, ray_origin: Tensor, ray_direction: Tensor, max_range: float):
        ray_dir_world = torch.stack([torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1)
        test_point_pos = sphere.state.pos
        line_rot = ray_direction
        line_length = max_range
        line_pos = ray_origin + ray_dir_world * (line_length / 2)
        closest_point = _get_closest_point_line(line_pos, line_rot.unsqueeze(-1), line_length, test_point_pos, limit_to_line_length=False)
        d = test_point_pos - closest_point
        d_norm = torch.linalg.vector_norm(d, dim=1)
        ray_intersects = d_norm < sphere.shape.radius
        a = sphere.shape.radius ** 2 - d_norm ** 2
        m = torch.sqrt(torch.where(a > 0, a, 1e-08))
        u = test_point_pos - ray_origin
        u1 = closest_point - ray_origin
        u_dot_ray = (u * ray_dir_world).sum(-1)
        sphere_is_in_front = u_dot_ray > 0.0
        dist = torch.linalg.vector_norm(u1, dim=1) - m
        dist[~(ray_intersects & sphere_is_in_front)] = max_range
        return dist

    def _cast_rays_to_sphere(self, sphere_pos, sphere_radius, ray_origin: Tensor, ray_direction: Tensor, max_range: float):
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2
        assert ray_direction.shape[:-1] == batch_size
        assert sphere_pos.shape[:-2] == batch_size
        assert sphere_pos.shape[-1] == 2
        assert sphere_radius.shape[:-1] == batch_size
        num_angles = ray_direction.shape[-1]
        n_spheres = sphere_pos.shape[-2]
        ray_origin = ray_origin.unsqueeze(-2).unsqueeze(-2).expand(*batch_size, n_spheres, num_angles, 2)
        sphere_pos_expanded = sphere_pos.unsqueeze(-2).expand(*batch_size, n_spheres, num_angles, 2)
        ray_direction = ray_direction.unsqueeze(-2).expand(*batch_size, n_spheres, num_angles)
        sphere_radius_expanded = sphere_radius.unsqueeze(-1).expand(*batch_size, n_spheres, num_angles)
        ray_dir_world = torch.stack([torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1)
        line_rot = ray_direction.unsqueeze(-1)
        line_length = max_range
        line_pos = ray_origin + ray_dir_world * (line_length / 2)
        closest_point = _get_closest_point_line(line_pos, line_rot, line_length, sphere_pos_expanded, limit_to_line_length=False)
        d = sphere_pos_expanded - closest_point
        d_norm = torch.linalg.vector_norm(d, dim=-1)
        ray_intersects = d_norm < sphere_radius_expanded
        a = sphere_radius_expanded ** 2 - d_norm ** 2
        m = torch.sqrt(torch.where(a > 0, a, 1e-08))
        u = sphere_pos_expanded - ray_origin
        u1 = closest_point - ray_origin
        u_dot_ray = (u * ray_dir_world).sum(-1)
        sphere_is_in_front = u_dot_ray > 0.0
        dist = torch.linalg.vector_norm(u1, dim=-1) - m
        dist[~(ray_intersects & sphere_is_in_front)] = max_range
        return dist

    def _cast_ray_to_line(self, line: Entity, ray_origin: Tensor, ray_direction: Tensor, max_range: float):
        """
        Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
        Computes distance of ray originating from pos at angle to a line and sets distance to
        max_range if there is no intersection.
        """
        assert ray_origin.ndim == 2 and ray_direction.ndim == 1
        assert ray_origin.shape[0] == ray_direction.shape[0]
        assert isinstance(line.shape, Line)
        p = line.state.pos
        r = torch.stack([torch.cos(line.state.rot.squeeze(1)), torch.sin(line.state.rot.squeeze(1))], dim=-1) * line.shape.length
        q = ray_origin
        s = torch.stack([torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1)
        rxs = TorchUtils.cross(r, s)
        t = TorchUtils.cross(q - p, s / rxs)
        u = TorchUtils.cross(q - p, r / rxs)
        d = torch.linalg.norm(u * s, dim=-1)
        perpendicular = rxs == 0.0
        above_line = t > 0.5
        below_line = t < -0.5
        behind_line = u < 0.0
        d[perpendicular.squeeze(-1)] = max_range
        d[above_line.squeeze(-1)] = max_range
        d[below_line.squeeze(-1)] = max_range
        d[behind_line.squeeze(-1)] = max_range
        return d

    def _cast_rays_to_line(self, line_pos, line_rot, line_length, ray_origin: Tensor, ray_direction: Tensor, max_range: float):
        """
        Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
        Computes distance of ray originating from pos at angle to a line and sets distance to
        max_range if there is no intersection.
        """
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2
        assert ray_direction.shape[:-1] == batch_size
        assert line_pos.shape[:-2] == batch_size
        assert line_pos.shape[-1] == 2
        assert line_rot.shape[:-1] == batch_size
        assert line_length.shape[:-1] == batch_size
        num_angles = ray_direction.shape[-1]
        n_lines = line_pos.shape[-2]
        ray_origin = ray_origin.unsqueeze(-2).unsqueeze(-2).expand(*batch_size, n_lines, num_angles, 2)
        line_pos_expanded = line_pos.unsqueeze(-2).expand(*batch_size, n_lines, num_angles, 2)
        ray_direction = ray_direction.unsqueeze(-2).expand(*batch_size, n_lines, num_angles)
        line_rot_expanded = line_rot.unsqueeze(-1).expand(*batch_size, n_lines, num_angles)
        line_length_expanded = line_length.unsqueeze(-1).expand(*batch_size, n_lines, num_angles)
        r = torch.stack([torch.cos(line_rot_expanded), torch.sin(line_rot_expanded)], dim=-1) * line_length_expanded.unsqueeze(-1)
        q = ray_origin
        s = torch.stack([torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1)
        rxs = TorchUtils.cross(r, s)
        t = TorchUtils.cross(q - line_pos_expanded, s / rxs)
        u = TorchUtils.cross(q - line_pos_expanded, r / rxs)
        d = torch.linalg.norm(u * s, dim=-1)
        perpendicular = rxs == 0.0
        above_line = t > 0.5
        below_line = t < -0.5
        behind_line = u < 0.0
        d[perpendicular.squeeze(-1)] = max_range
        d[above_line.squeeze(-1)] = max_range
        d[below_line.squeeze(-1)] = max_range
        d[behind_line.squeeze(-1)] = max_range
        return d

    def cast_ray(self, entity: Entity, angles: Tensor, max_range: float, entity_filter: Callable[[Entity], bool]=lambda _: False):
        pos = entity.state.pos
        assert pos.ndim == 2 and angles.ndim == 1
        assert pos.shape[0] == angles.shape[0]
        dists = [torch.full((self.batch_dim,), fill_value=max_range, device=self.device)]
        for e in self.entities:
            if entity is e or not entity_filter(e):
                continue
            assert e.collides(entity) and entity.collides(e), 'Rays are only casted among collidables'
            if isinstance(e.shape, Box):
                d = self._cast_ray_to_box(e, pos, angles, max_range)
            elif isinstance(e.shape, Sphere):
                d = self._cast_ray_to_sphere(e, pos, angles, max_range)
            elif isinstance(e.shape, Line):
                d = self._cast_ray_to_line(e, pos, angles, max_range)
            else:
                raise RuntimeError(f'Shape {e.shape} currently not handled by cast_ray')
            dists.append(d)
        dist, _ = torch.min(torch.stack(dists, dim=-1), dim=-1)
        return dist

    def cast_rays(self, entity: Entity, angles: Tensor, max_range: float, entity_filter: Callable[[Entity], bool]=lambda _: False):
        pos = entity.state.pos
        dists = torch.full_like(angles, fill_value=max_range, device=self.device).unsqueeze(-1)
        boxes = []
        spheres = []
        lines = []
        for e in self.entities:
            if entity is e or not entity_filter(e):
                continue
            assert e.collides(entity) and entity.collides(e), 'Rays are only casted among collidables'
            if isinstance(e.shape, Box):
                boxes.append(e)
            elif isinstance(e.shape, Sphere):
                spheres.append(e)
            elif isinstance(e.shape, Line):
                lines.append(e)
            else:
                raise RuntimeError(f'Shape {e.shape} currently not handled by cast_ray')
        if len(boxes):
            pos_box = []
            rot_box = []
            length_box = []
            width_box = []
            for box in boxes:
                pos_box.append(box.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
            pos_box = torch.stack(pos_box, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            length_box = torch.stack(length_box, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            width_box = torch.stack(width_box, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            dist_boxes = self._cast_rays_to_box(pos_box, rot_box.squeeze(-1), length_box, width_box, pos, angles, max_range)
            dists = torch.cat([dists, dist_boxes.transpose(-1, -2)], dim=-1)
        if len(spheres):
            pos_s = []
            radius_s = []
            for s in spheres:
                pos_s.append(s.state.pos)
                radius_s.append(torch.tensor(s.shape.radius, device=self.device))
            pos_s = torch.stack(pos_s, dim=-2)
            radius_s = torch.stack(radius_s, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            dist_spheres = self._cast_rays_to_sphere(pos_s, radius_s, pos, angles, max_range)
            dists = torch.cat([dists, dist_spheres.transpose(-1, -2)], dim=-1)
        if len(lines):
            pos_l = []
            rot_l = []
            length_l = []
            for line in lines:
                pos_l.append(line.state.pos)
                rot_l.append(line.state.rot)
                length_l.append(torch.tensor(line.shape.length, device=self.device))
            pos_l = torch.stack(pos_l, dim=-2)
            rot_l = torch.stack(rot_l, dim=-2)
            length_l = torch.stack(length_l, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            dist_lines = self._cast_rays_to_line(pos_l, rot_l.squeeze(-1), length_l, pos, angles, max_range)
            dists = torch.cat([dists, dist_lines.transpose(-1, -2)], dim=-1)
        dist, _ = torch.min(dists, dim=-1)
        return dist

    def get_distance_from_point(self, entity: Entity, test_point_pos, env_index: int=None):
        self._check_batch_index(env_index)
        if isinstance(entity.shape, Sphere):
            delta_pos = entity.state.pos - test_point_pos
            dist = torch.linalg.vector_norm(delta_pos, dim=-1)
            return_value = dist - entity.shape.radius
        elif isinstance(entity.shape, Box):
            closest_point = _get_closest_point_box(entity.state.pos, entity.state.rot, entity.shape.width, entity.shape.length, test_point_pos)
            distance = torch.linalg.vector_norm(test_point_pos - closest_point, dim=-1)
            return_value = distance - LINE_MIN_DIST
        elif isinstance(entity.shape, Line):
            closest_point = _get_closest_point_line(entity.state.pos, entity.state.rot, entity.shape.length, test_point_pos)
            distance = torch.linalg.vector_norm(test_point_pos - closest_point, dim=-1)
            return_value = distance - LINE_MIN_DIST
        else:
            raise RuntimeError('Distance not computable for given entity')
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    def get_distance(self, entity_a: Entity, entity_b: Entity, env_index: int=None):
        a_shape = entity_a.shape
        b_shape = entity_b.shape
        if isinstance(a_shape, Sphere) and isinstance(b_shape, Sphere):
            dist = self.get_distance_from_point(entity_a, entity_b.state.pos, env_index)
            return_value = dist - b_shape.radius
        elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Sphere) or (isinstance(entity_b.shape, Box) and isinstance(entity_a.shape, Sphere)):
            box, sphere = (entity_a, entity_b) if isinstance(entity_b.shape, Sphere) else (entity_b, entity_a)
            dist = self.get_distance_from_point(box, sphere.state.pos, env_index)
            return_value = dist - sphere.shape.radius
            is_overlapping = self.is_overlapping(entity_a, entity_b)
            return_value[is_overlapping] = -1
        elif isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Sphere) or (isinstance(entity_b.shape, Line) and isinstance(entity_a.shape, Sphere)):
            line, sphere = (entity_a, entity_b) if isinstance(entity_b.shape, Sphere) else (entity_b, entity_a)
            dist = self.get_distance_from_point(line, sphere.state.pos, env_index)
            return_value = dist - sphere.shape.radius
        elif isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line):
            point_a, point_b = _get_closest_points_line_line(entity_a.state.pos, entity_a.state.rot, entity_a.shape.length, entity_b.state.pos, entity_b.state.rot, entity_b.shape.length)
            dist = torch.linalg.vector_norm(point_a - point_b, dim=1)
            return_value = dist - LINE_MIN_DIST
        elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Line) or (isinstance(entity_b.shape, Box) and isinstance(entity_a.shape, Line)):
            box, line = (entity_a, entity_b) if isinstance(entity_b.shape, Line) else (entity_b, entity_a)
            point_box, point_line = _get_closest_line_box(box.state.pos, box.state.rot, box.shape.width, box.shape.length, line.state.pos, line.state.rot, line.shape.length)
            dist = torch.linalg.vector_norm(point_box - point_line, dim=1)
            return_value = dist - LINE_MIN_DIST
        elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box):
            point_a, point_b = _get_closest_box_box(entity_a.state.pos, entity_a.state.rot, entity_a.shape.width, entity_a.shape.length, entity_b.state.pos, entity_b.state.rot, entity_b.shape.width, entity_b.shape.length)
            dist = torch.linalg.vector_norm(point_a - point_b, dim=-1)
            return_value = dist - LINE_MIN_DIST
        else:
            raise RuntimeError('Distance not computable for given entities')
        return return_value

    def is_overlapping(self, entity_a: Entity, entity_b: Entity, env_index: int=None):
        a_shape = entity_a.shape
        b_shape = entity_b.shape
        self._check_batch_index(env_index)
        if isinstance(a_shape, Sphere) and isinstance(b_shape, Sphere) or (isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Sphere) or (isinstance(entity_b.shape, Line) and isinstance(entity_a.shape, Sphere))) or (isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line)) or (isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Line) or (isinstance(entity_b.shape, Box) and isinstance(entity_a.shape, Line))) or (isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box)):
            return self.get_distance(entity_a, entity_b, env_index) < 0
        elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Sphere) or (isinstance(entity_b.shape, Box) and isinstance(entity_a.shape, Sphere)):
            box, sphere = (entity_a, entity_b) if isinstance(entity_b.shape, Sphere) else (entity_b, entity_a)
            closest_point = _get_closest_point_box(box.state.pos, box.state.rot, box.shape.width, box.shape.length, sphere.state.pos)
            distance_sphere_closest_point = torch.linalg.vector_norm(sphere.state.pos - closest_point, dim=-1)
            distance_sphere_box = torch.linalg.vector_norm(sphere.state.pos - box.state.pos, dim=-1)
            distance_closest_point_box = torch.linalg.vector_norm(box.state.pos - closest_point, dim=-1)
            dist_min = sphere.shape.radius + LINE_MIN_DIST
            return_value = (distance_sphere_box < distance_closest_point_box) + (distance_sphere_closest_point < dist_min)
        else:
            raise RuntimeError('Overlap not computable for give entities')
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    def step(self):
        self.entity_index_map = {e: i for i, e in enumerate(self.entities)}
        for substep in range(self._substeps):
            self.forces_dict = {e: torch.zeros(self._batch_dim, self._dim_p, device=self.device, dtype=torch.float32) for e in self.entities}
            self.torques_dict = {e: torch.zeros(self._batch_dim, 1, device=self.device, dtype=torch.float32) for e in self.entities}
            for entity in self.entities:
                if isinstance(entity, Agent):
                    self._apply_action_force(entity)
                    self._apply_action_torque(entity)
                self._apply_friction_force(entity)
                self._apply_gravity(entity)
            self._apply_vectorized_enviornment_force()
            for entity in self.entities:
                self._integrate_state(entity, substep)
        if self._dim_c > 0:
            for agent in self._agents:
                self._update_comm_state(agent)

    def _apply_action_force(self, agent: Agent):
        if agent.movable:
            if agent.max_f is not None:
                agent.state.force = TorchUtils.clamp_with_norm(agent.state.force, agent.max_f)
            if agent.f_range is not None:
                agent.state.force = torch.clamp(agent.state.force, -agent.f_range, agent.f_range)
            self.forces_dict[agent] = self.forces_dict[agent] + agent.state.force

    def _apply_action_torque(self, agent: Agent):
        if agent.rotatable:
            if agent.max_t is not None:
                agent.state.torque = TorchUtils.clamp_with_norm(agent.state.torque, agent.max_t)
            if agent.t_range is not None:
                agent.state.torque = torch.clamp(agent.state.torque, -agent.t_range, agent.t_range)
            self.torques_dict[agent] = self.torques_dict[agent] + agent.state.torque

    def _apply_gravity(self, entity: Entity):
        if entity.movable:
            if not (self._gravity == 0.0).all():
                self.forces_dict[entity] = self.forces_dict[entity] + entity.mass * self._gravity
            if entity.gravity is not None:
                self.forces_dict[entity] = self.forces_dict[entity] + entity.mass * entity.gravity

    def _apply_friction_force(self, entity: Entity):

        def get_friction_force(vel, coeff, force, mass):
            speed = torch.linalg.vector_norm(vel, dim=-1)
            static = speed == 0
            static_exp = static.unsqueeze(-1).expand(vel.shape)
            if not isinstance(coeff, Tensor):
                coeff = torch.full_like(force, coeff, device=self.device)
            coeff = coeff.expand(force.shape)
            friction_force_constant = coeff * mass
            friction_force = -(vel / torch.where(static, 1e-08, speed).unsqueeze(-1)) * torch.minimum(friction_force_constant, vel.abs() / self._sub_dt * mass)
            friction_force = torch.where(static_exp, 0.0, friction_force)
            return friction_force
        if entity.linear_friction is not None:
            self.forces_dict[entity] = self.forces_dict[entity] + get_friction_force(entity.state.vel, entity.linear_friction, self.forces_dict[entity], entity.mass)
        elif self._linear_friction > 0:
            self.forces_dict[entity] = self.forces_dict[entity] + get_friction_force(entity.state.vel, self._linear_friction, self.forces_dict[entity], entity.mass)
        if entity.angular_friction is not None:
            self.torques_dict[entity] = self.torques_dict[entity] + get_friction_force(entity.state.ang_vel, entity.angular_friction, self.torques_dict[entity], entity.moment_of_inertia)
        elif self._angular_friction > 0:
            self.torques_dict[entity] = self.torques_dict[entity] + get_friction_force(entity.state.ang_vel, self._angular_friction, self.torques_dict[entity], entity.moment_of_inertia)

    def _apply_vectorized_enviornment_force(self):
        s_s = []
        l_s = []
        b_s = []
        l_l = []
        b_l = []
        b_b = []
        joints = []
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                joint = self._joints.get(frozenset({entity_a.name, entity_b.name}), None)
                if joint is not None:
                    joints.append(joint)
                    if joint.dist == 0:
                        continue
                if not self.collides(entity_a, entity_b):
                    continue
                if isinstance(entity_a.shape, Sphere) and isinstance(entity_b.shape, Sphere):
                    s_s.append((entity_a, entity_b))
                elif isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Sphere) or (isinstance(entity_b.shape, Line) and isinstance(entity_a.shape, Sphere)):
                    line, sphere = (entity_a, entity_b) if isinstance(entity_b.shape, Sphere) else (entity_b, entity_a)
                    l_s.append((line, sphere))
                elif isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line):
                    l_l.append((entity_a, entity_b))
                elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Sphere) or (isinstance(entity_b.shape, Box) and isinstance(entity_a.shape, Sphere)):
                    box, sphere = (entity_a, entity_b) if isinstance(entity_b.shape, Sphere) else (entity_b, entity_a)
                    b_s.append((box, sphere))
                elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Line) or (isinstance(entity_b.shape, Box) and isinstance(entity_a.shape, Line)):
                    box, line = (entity_a, entity_b) if isinstance(entity_b.shape, Line) else (entity_b, entity_a)
                    b_l.append((box, line))
                elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box):
                    b_b.append((entity_a, entity_b))
                else:
                    raise AssertionError()
        self._vectorized_joint_constraints(joints)
        self._sphere_sphere_vectorized_collision(s_s)
        self._sphere_line_vectorized_collision(l_s)
        self._line_line_vectorized_collision(l_l)
        self._box_sphere_vectorized_collision(b_s)
        self._box_line_vectorized_collision(b_l)
        self._box_box_vectorized_collision(b_b)

    def update_env_forces(self, entity_a, f_a, t_a, entity_b, f_b, t_b):
        if entity_a.movable:
            self.forces_dict[entity_a] = self.forces_dict[entity_a] + f_a
        if entity_a.rotatable:
            self.torques_dict[entity_a] = self.torques_dict[entity_a] + t_a
        if entity_b.movable:
            self.forces_dict[entity_b] = self.forces_dict[entity_b] + f_b
        if entity_b.rotatable:
            self.torques_dict[entity_b] = self.torques_dict[entity_b] + t_b

    def _vectorized_joint_constraints(self, joints):
        if len(joints):
            pos_a = []
            pos_b = []
            pos_joint_a = []
            pos_joint_b = []
            dist = []
            rotate = []
            rot_a = []
            rot_b = []
            joint_rot = []
            for joint in joints:
                entity_a = joint.entity_a
                entity_b = joint.entity_b
                pos_joint_a.append(joint.pos_point(entity_a))
                pos_joint_b.append(joint.pos_point(entity_b))
                pos_a.append(entity_a.state.pos)
                pos_b.append(entity_b.state.pos)
                dist.append(torch.tensor(joint.dist, device=self.device))
                rotate.append(torch.tensor(joint.rotate, device=self.device))
                rot_a.append(entity_a.state.rot)
                rot_b.append(entity_b.state.rot)
                joint_rot.append(torch.tensor(joint.fixed_rotation, device=self.device).unsqueeze(-1).expand(self.batch_dim, 1) if isinstance(joint.fixed_rotation, float) else joint.fixed_rotation)
            pos_a = torch.stack(pos_a, dim=-2)
            pos_b = torch.stack(pos_b, dim=-2)
            pos_joint_a = torch.stack(pos_joint_a, dim=-2)
            pos_joint_b = torch.stack(pos_joint_b, dim=-2)
            rot_a = torch.stack(rot_a, dim=-2)
            rot_b = torch.stack(rot_b, dim=-2)
            dist = torch.stack(dist, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            rotate_prior = torch.stack(rotate, dim=-1)
            rotate = rotate_prior.unsqueeze(0).expand(self.batch_dim, -1).unsqueeze(-1)
            joint_rot = torch.stack(joint_rot, dim=-2)
            force_a_attractive, force_b_attractive = self._get_constraint_forces(pos_joint_a, pos_joint_b, dist_min=dist, attractive=True, force_multiplier=self._joint_force)
            force_a_repulsive, force_b_repulsive = self._get_constraint_forces(pos_joint_a, pos_joint_b, dist_min=dist, attractive=False, force_multiplier=self._joint_force)
            force_a = force_a_attractive + force_a_repulsive
            force_b = force_b_attractive + force_b_repulsive
            r_a = pos_joint_a - pos_a
            r_b = pos_joint_b - pos_b
            torque_a_rotate = TorchUtils.compute_torque(force_a, r_a)
            torque_b_rotate = TorchUtils.compute_torque(force_b, r_b)
            torque_a_fixed, torque_b_fixed = self._get_constraint_torques(rot_a, rot_b + joint_rot, force_multiplier=self._torque_constraint_force)
            torque_a = torch.where(rotate, torque_a_rotate, torque_a_rotate + torque_a_fixed)
            torque_b = torch.where(rotate, torque_b_rotate, torque_b_rotate + torque_b_fixed)
            for i, joint in enumerate(joints):
                self.update_env_forces(joint.entity_a, force_a[:, i], torque_a[:, i], joint.entity_b, force_b[:, i], torque_b[:, i])

    def _sphere_sphere_vectorized_collision(self, s_s):
        if len(s_s):
            pos_s_a = []
            pos_s_b = []
            radius_s_a = []
            radius_s_b = []
            for s_a, s_b in s_s:
                pos_s_a.append(s_a.state.pos)
                pos_s_b.append(s_b.state.pos)
                radius_s_a.append(torch.tensor(s_a.shape.radius, device=self.device))
                radius_s_b.append(torch.tensor(s_b.shape.radius, device=self.device))
            pos_s_a = torch.stack(pos_s_a, dim=-2)
            pos_s_b = torch.stack(pos_s_b, dim=-2)
            radius_s_a = torch.stack(radius_s_a, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            radius_s_b = torch.stack(radius_s_b, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            force_a, force_b = self._get_constraint_forces(pos_s_a, pos_s_b, dist_min=radius_s_a + radius_s_b, force_multiplier=self._collision_force)
            for i, (entity_a, entity_b) in enumerate(s_s):
                self.update_env_forces(entity_a, force_a[:, i], 0, entity_b, force_b[:, i], 0)

    def _sphere_line_vectorized_collision(self, l_s):
        if len(l_s):
            pos_l = []
            pos_s = []
            rot_l = []
            radius_s = []
            length_l = []
            for line, sphere in l_s:
                pos_l.append(line.state.pos)
                pos_s.append(sphere.state.pos)
                rot_l.append(line.state.rot)
                radius_s.append(torch.tensor(sphere.shape.radius, device=self.device))
                length_l.append(torch.tensor(line.shape.length, device=self.device))
            pos_l = torch.stack(pos_l, dim=-2)
            pos_s = torch.stack(pos_s, dim=-2)
            rot_l = torch.stack(rot_l, dim=-2)
            radius_s = torch.stack(radius_s, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            length_l = torch.stack(length_l, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            closest_point = _get_closest_point_line(pos_l, rot_l, length_l, pos_s)
            force_sphere, force_line = self._get_constraint_forces(pos_s, closest_point, dist_min=radius_s + LINE_MIN_DIST, force_multiplier=self._collision_force)
            r = closest_point - pos_l
            torque_line = TorchUtils.compute_torque(force_line, r)
            for i, (entity_a, entity_b) in enumerate(l_s):
                self.update_env_forces(entity_a, force_line[:, i], torque_line[:, i], entity_b, force_sphere[:, i], 0)

    def _line_line_vectorized_collision(self, l_l):
        if len(l_l):
            pos_l_a = []
            pos_l_b = []
            rot_l_a = []
            rot_l_b = []
            length_l_a = []
            length_l_b = []
            for l_a, l_b in l_l:
                pos_l_a.append(l_a.state.pos)
                pos_l_b.append(l_b.state.pos)
                rot_l_a.append(l_a.state.rot)
                rot_l_b.append(l_b.state.rot)
                length_l_a.append(torch.tensor(l_a.shape.length, device=self.device))
                length_l_b.append(torch.tensor(l_b.shape.length, device=self.device))
            pos_l_a = torch.stack(pos_l_a, dim=-2)
            pos_l_b = torch.stack(pos_l_b, dim=-2)
            rot_l_a = torch.stack(rot_l_a, dim=-2)
            rot_l_b = torch.stack(rot_l_b, dim=-2)
            length_l_a = torch.stack(length_l_a, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            length_l_b = torch.stack(length_l_b, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            point_a, point_b = _get_closest_points_line_line(pos_l_a, rot_l_a, length_l_a, pos_l_b, rot_l_b, length_l_b)
            force_a, force_b = self._get_constraint_forces(point_a, point_b, dist_min=LINE_MIN_DIST, force_multiplier=self._collision_force)
            r_a = point_a - pos_l_a
            r_b = point_b - pos_l_b
            torque_a = TorchUtils.compute_torque(force_a, r_a)
            torque_b = TorchUtils.compute_torque(force_b, r_b)
            for i, (entity_a, entity_b) in enumerate(l_l):
                self.update_env_forces(entity_a, force_a[:, i], torque_a[:, i], entity_b, force_b[:, i], torque_b[:, i])

    def _box_sphere_vectorized_collision(self, b_s):
        if len(b_s):
            pos_box = []
            pos_sphere = []
            rot_box = []
            length_box = []
            width_box = []
            not_hollow_box = []
            radius_sphere = []
            for box, sphere in b_s:
                pos_box.append(box.state.pos)
                pos_sphere.append(sphere.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
                not_hollow_box.append(torch.tensor(not box.shape.hollow, device=self.device))
                radius_sphere.append(torch.tensor(sphere.shape.radius, device=self.device))
            pos_box = torch.stack(pos_box, dim=-2)
            pos_sphere = torch.stack(pos_sphere, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            length_box = torch.stack(length_box, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            width_box = torch.stack(width_box, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            not_hollow_box_prior = torch.stack(not_hollow_box, dim=-1)
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(self.batch_dim, -1)
            radius_sphere = torch.stack(radius_sphere, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            closest_point_box = _get_closest_point_box(pos_box, rot_box, width_box, length_box, pos_sphere)
            inner_point_box = closest_point_box
            d = torch.zeros_like(radius_sphere, device=self.device, dtype=torch.float)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(pos_sphere, closest_point_box, pos_box)
                cond = not_hollow_box.unsqueeze(-1).expand(inner_point_box.shape)
                inner_point_box = torch.where(cond, inner_point_box_hollow, inner_point_box)
                d = torch.where(not_hollow_box, d_hollow, d)
            force_sphere, force_box = self._get_constraint_forces(pos_sphere, inner_point_box, dist_min=radius_sphere + LINE_MIN_DIST + d, force_multiplier=self._collision_force)
            r = closest_point_box - pos_box
            torque_box = TorchUtils.compute_torque(force_box, r)
            for i, (entity_a, entity_b) in enumerate(b_s):
                self.update_env_forces(entity_a, force_box[:, i], torque_box[:, i], entity_b, force_sphere[:, i], 0)

    def _box_line_vectorized_collision(self, b_l):
        if len(b_l):
            pos_box = []
            pos_line = []
            rot_box = []
            rot_line = []
            length_box = []
            width_box = []
            not_hollow_box = []
            length_line = []
            for box, line in b_l:
                pos_box.append(box.state.pos)
                pos_line.append(line.state.pos)
                rot_box.append(box.state.rot)
                rot_line.append(line.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
                not_hollow_box.append(torch.tensor(not box.shape.hollow, device=self.device))
                length_line.append(torch.tensor(line.shape.length, device=self.device))
            pos_box = torch.stack(pos_box, dim=-2)
            pos_line = torch.stack(pos_line, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            rot_line = torch.stack(rot_line, dim=-2)
            length_box = torch.stack(length_box, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            width_box = torch.stack(width_box, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            not_hollow_box_prior = torch.stack(not_hollow_box, dim=-1)
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(self.batch_dim, -1)
            length_line = torch.stack(length_line, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            point_box, point_line = _get_closest_line_box(pos_box, rot_box, width_box, length_box, pos_line, rot_line, length_line)
            inner_point_box = point_box
            d = torch.zeros_like(length_line, device=self.device, dtype=torch.float)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(point_line, point_box, pos_box)
                cond = not_hollow_box.unsqueeze(-1).expand(inner_point_box.shape)
                inner_point_box = torch.where(cond, inner_point_box_hollow, inner_point_box)
                d = torch.where(not_hollow_box, d_hollow, d)
            force_box, force_line = self._get_constraint_forces(inner_point_box, point_line, dist_min=LINE_MIN_DIST + d, force_multiplier=self._collision_force)
            r_box = point_box - pos_box
            r_line = point_line - pos_line
            torque_box = TorchUtils.compute_torque(force_box, r_box)
            torque_line = TorchUtils.compute_torque(force_line, r_line)
            for i, (entity_a, entity_b) in enumerate(b_l):
                self.update_env_forces(entity_a, force_box[:, i], torque_box[:, i], entity_b, force_line[:, i], torque_line[:, i])

    def _box_box_vectorized_collision(self, b_b):
        if len(b_b):
            pos_box = []
            pos_box2 = []
            rot_box = []
            rot_box2 = []
            length_box = []
            width_box = []
            not_hollow_box = []
            length_box2 = []
            width_box2 = []
            not_hollow_box2 = []
            for box, box2 in b_b:
                pos_box.append(box.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
                not_hollow_box.append(torch.tensor(not box.shape.hollow, device=self.device))
                pos_box2.append(box2.state.pos)
                rot_box2.append(box2.state.rot)
                length_box2.append(torch.tensor(box2.shape.length, device=self.device))
                width_box2.append(torch.tensor(box2.shape.width, device=self.device))
                not_hollow_box2.append(torch.tensor(not box2.shape.hollow, device=self.device))
            pos_box = torch.stack(pos_box, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            length_box = torch.stack(length_box, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            width_box = torch.stack(width_box, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            not_hollow_box_prior = torch.stack(not_hollow_box, dim=-1)
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(self.batch_dim, -1)
            pos_box2 = torch.stack(pos_box2, dim=-2)
            rot_box2 = torch.stack(rot_box2, dim=-2)
            length_box2 = torch.stack(length_box2, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            width_box2 = torch.stack(width_box2, dim=-1).unsqueeze(0).expand(self.batch_dim, -1)
            not_hollow_box2_prior = torch.stack(not_hollow_box2, dim=-1)
            not_hollow_box2 = not_hollow_box2_prior.unsqueeze(0).expand(self.batch_dim, -1)
            point_a, point_b = _get_closest_box_box(pos_box, rot_box, width_box, length_box, pos_box2, rot_box2, width_box2, length_box2)
            inner_point_a = point_a
            d_a = torch.zeros_like(length_box, device=self.device, dtype=torch.float)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(point_b, point_a, pos_box)
                cond = not_hollow_box.unsqueeze(-1).expand(inner_point_a.shape)
                inner_point_a = torch.where(cond, inner_point_box_hollow, inner_point_a)
                d_a = torch.where(not_hollow_box, d_hollow, d_a)
            inner_point_b = point_b
            d_b = torch.zeros_like(length_box2, device=self.device, dtype=torch.float)
            if not_hollow_box2_prior.any():
                inner_point_box2_hollow, d_hollow2 = _get_inner_point_box(point_a, point_b, pos_box2)
                cond = not_hollow_box2.unsqueeze(-1).expand(inner_point_b.shape)
                inner_point_b = torch.where(cond, inner_point_box2_hollow, inner_point_b)
                d_b = torch.where(not_hollow_box2, d_hollow2, d_b)
            force_a, force_b = self._get_constraint_forces(inner_point_a, inner_point_b, dist_min=d_a + d_b + LINE_MIN_DIST, force_multiplier=self._collision_force)
            r_a = point_a - pos_box
            r_b = point_b - pos_box2
            torque_a = TorchUtils.compute_torque(force_a, r_a)
            torque_b = TorchUtils.compute_torque(force_b, r_b)
            for i, (entity_a, entity_b) in enumerate(b_b):
                self.update_env_forces(entity_a, force_a[:, i], torque_a[:, i], entity_b, force_b[:, i], torque_b[:, i])

    def collides(self, a: Entity, b: Entity) -> bool:
        if not a.collides(b) or not b.collides(a) or a is b:
            return False
        a_shape = a.shape
        b_shape = b.shape
        if not a.movable and (not a.rotatable) and (not b.movable) and (not b.rotatable):
            return False
        if not {a_shape.__class__, b_shape.__class__} in self._collidable_pairs:
            return False
        if not (torch.linalg.vector_norm(a.state.pos - b.state.pos, dim=-1) <= a.shape.circumscribed_radius() + b.shape.circumscribed_radius()).any():
            return False
        return True

    def _get_constraint_forces(self, pos_a: Tensor, pos_b: Tensor, dist_min, force_multiplier: float, attractive: bool=False) -> Tensor:
        min_dist = 1e-06
        delta_pos = pos_a - pos_b
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        sign = -1 if attractive else 1
        k = self._contact_margin
        penetration = torch.logaddexp(torch.tensor(0.0, dtype=torch.float32, device=self.device), (dist_min - dist) * sign / k) * k
        force = sign * force_multiplier * delta_pos / torch.where(dist > 0, dist, 1e-08).unsqueeze(-1) * penetration.unsqueeze(-1)
        force = torch.where((dist < min_dist).unsqueeze(-1), 0.0, force)
        if not attractive:
            force = torch.where((dist > dist_min).unsqueeze(-1), 0.0, force)
        else:
            force = torch.where((dist < dist_min).unsqueeze(-1), 0.0, force)
        return (force, -force)

    def _get_constraint_torques(self, rot_a: Tensor, rot_b: Tensor, force_multiplier: float=TORQUE_CONSTRAINT_FORCE) -> Tensor:
        min_delta_rot = 1e-09
        delta_rot = rot_a - rot_b
        abs_delta_rot = torch.linalg.vector_norm(delta_rot, dim=-1).unsqueeze(-1)
        k = 1
        penetration = k * (torch.exp(abs_delta_rot / k) - 1)
        torque = force_multiplier * delta_rot.sign() * penetration
        torque = torch.where(abs_delta_rot < min_delta_rot, 0.0, torque)
        return (-torque, torque)

    def _integrate_state(self, entity: Entity, substep: int):
        if entity.movable:
            if substep == 0:
                if entity.drag is not None:
                    entity.state.vel = entity.state.vel * (1 - entity.drag)
                else:
                    entity.state.vel = entity.state.vel * (1 - self._drag)
            accel = self.forces_dict[entity] / entity.mass
            entity.state.vel = entity.state.vel + accel * self._sub_dt
            if entity.max_speed is not None:
                entity.state.vel = TorchUtils.clamp_with_norm(entity.state.vel, entity.max_speed)
            if entity.v_range is not None:
                entity.state.vel = entity.state.vel.clamp(-entity.v_range, entity.v_range)
            new_pos = entity.state.pos + entity.state.vel * self._sub_dt
            entity.state.pos = torch.stack([new_pos[..., X].clamp(-self._x_semidim, self._x_semidim) if self._x_semidim is not None else new_pos[..., X], new_pos[..., Y].clamp(-self._y_semidim, self._y_semidim) if self._y_semidim is not None else new_pos[..., Y]], dim=-1)
        if entity.rotatable:
            if substep == 0:
                if entity.drag is not None:
                    entity.state.ang_vel = entity.state.ang_vel * (1 - entity.drag)
                else:
                    entity.state.ang_vel = entity.state.ang_vel * (1 - self._drag)
            entity.state.ang_vel = entity.state.ang_vel + self.torques_dict[entity] / entity.moment_of_inertia * self._sub_dt
            entity.state.rot = entity.state.rot + entity.state.ang_vel * self._sub_dt

    def _update_comm_state(self, agent):
        if not agent.silent:
            agent.state.c = agent.action.c

    @override(TorchVectorizedObject)
    def to(self, device: torch.device):
        super().to(device)
        for e in self.entities:
            e.to(device)

class Forward(Dynamics):

    @property
    def needed_action_size(self) -> int:
        return 1

    def process_action(self):
        force = torch.zeros(self.agent.batch_dim, 2, device=self.agent.device, dtype=torch.float)
        force[:, X] = self.agent.action.u[:, 0]
        self.agent.state.force = TorchUtils.rotate_vector(force, self.agent.state.rot)

