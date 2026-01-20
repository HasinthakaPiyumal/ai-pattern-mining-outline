# Cluster 20

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

