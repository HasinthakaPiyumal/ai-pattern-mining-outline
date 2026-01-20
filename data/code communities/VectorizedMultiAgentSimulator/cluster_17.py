# Cluster 17

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop('n_agents', 3)
        self.shared_rew = kwargs.pop('shared_rew', True)
        self.comms_range = kwargs.pop('comms_range', 0.0)
        self.lidar_range = kwargs.pop('lidar_range', 0.2)
        self.agent_radius = kwargs.pop('agent_radius', 0.025)
        self.xdim = kwargs.pop('xdim', 1)
        self.ydim = kwargs.pop('ydim', 1)
        self.grid_spacing = kwargs.pop('grid_spacing', 0.05)
        self.n_gaussians = kwargs.pop('n_gaussians', 3)
        self.cov = kwargs.pop('cov', 0.05)
        self.collisions = kwargs.pop('collisions', True)
        self.spawn_same_pos = kwargs.pop('spawn_same_pos', False)
        self.norm = kwargs.pop('norm', True)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        assert not (self.spawn_same_pos and self.collisions)
        assert self.xdim / self.grid_spacing % 1 == 0 and self.ydim / self.grid_spacing % 1 == 0
        self.covs = [self.cov] * self.n_gaussians if isinstance(self.cov, float) else self.cov
        assert len(self.covs) == self.n_gaussians
        self.plot_grid = False
        self.visualize_semidims = False
        self.n_x_cells = int(2 * self.xdim / self.grid_spacing)
        self.n_y_cells = int(2 * self.ydim / self.grid_spacing)
        self.max_pdf = torch.zeros((batch_dim,), device=device, dtype=torch.float32)
        self.alpha_plot: float = 0.5
        self.agent_xspawn_range = 0 if self.spawn_same_pos else self.xdim
        self.agent_yspawn_range = 0 if self.spawn_same_pos else self.ydim
        self.x_semidim = self.xdim - self.agent_radius
        self.y_semidim = self.ydim - self.agent_radius
        world = World(batch_dim, device, x_semidim=self.x_semidim, y_semidim=self.y_semidim)
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)
        for i in range(self.n_agents):
            agent = Agent(name=f'agent_{i}', render_action=True, collide=self.collisions, shape=Sphere(radius=self.agent_radius), sensors=[Lidar(world, angle_start=0.05, angle_end=2 * torch.pi + 0.05, n_rays=12, max_range=self.lidar_range, entity_filter=entity_filter_agents)] if self.collisions else None)
            world.add_agent(agent)
        self.sampled = torch.zeros((batch_dim, self.n_x_cells, self.n_y_cells), device=device, dtype=torch.bool)
        self.locs = [torch.zeros((batch_dim, world.dim_p), device=device, dtype=torch.float32) for _ in range(self.n_gaussians)]
        self.cov_matrices = [torch.tensor([[cov, 0], [0, cov]], dtype=torch.float32, device=device).expand(batch_dim, world.dim_p, world.dim_p) for cov in self.covs]
        return world

    def reset_world_at(self, env_index: int=None):
        for i in range(len(self.locs)):
            x = torch.zeros((1,) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.xdim, self.xdim)
            y = torch.zeros((1,) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.ydim, self.ydim)
            new_loc = torch.cat([x, y], dim=-1)
            if env_index is None:
                self.locs[i] = new_loc
            else:
                self.locs[i][env_index] = new_loc
        self.gaussians = [MultivariateNormal(loc=loc, covariance_matrix=cov_matrix) for loc, cov_matrix in zip(self.locs, self.cov_matrices)]
        if env_index is None:
            self.max_pdf[:] = 0
            self.sampled[:] = False
        else:
            self.max_pdf[env_index] = 0
            self.sampled[env_index] = False
        self.nomrlize_pdf(env_index=env_index)
        for agent in self.world.agents:
            agent.set_pos(torch.cat([torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.agent_xspawn_range, self.agent_xspawn_range), torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32).uniform_(-self.agent_yspawn_range, self.agent_yspawn_range)], dim=-1), batch_index=env_index)
            agent.sample = self.sample(agent.state.pos, norm=self.norm)

    def sample(self, pos, update_sampled_flag: bool=False, norm: bool=True):
        out_of_bounds = (pos[:, X] < -self.xdim) + (pos[:, X] > self.xdim) + (pos[:, Y] < -self.ydim) + (pos[:, Y] > self.ydim)
        pos[:, X].clamp_(-self.world.x_semidim, self.world.x_semidim)
        pos[:, Y].clamp_(-self.world.y_semidim, self.world.y_semidim)
        index = pos / self.grid_spacing
        index[:, X] += self.n_x_cells / 2
        index[:, Y] += self.n_y_cells / 2
        index = index.to(torch.long)
        v = torch.stack([gaussian.log_prob(pos).exp() for gaussian in self.gaussians], dim=-1).sum(-1)
        if norm:
            v = v / self.max_pdf
        sampled = self.sampled[torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]]
        v[sampled + out_of_bounds] = 0
        if update_sampled_flag:
            self.sampled[torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]] = True
        return v

    def sample_single_env(self, pos, env_index, norm: bool=True):
        pos = pos.view(-1, self.world.dim_p)
        out_of_bounds = (pos[:, X] < -self.xdim) + (pos[:, X] > self.xdim) + (pos[:, Y] < -self.ydim) + (pos[:, Y] > self.ydim)
        pos[:, X].clamp_(-self.x_semidim, self.x_semidim)
        pos[:, Y].clamp_(-self.y_semidim, self.y_semidim)
        index = pos / self.grid_spacing
        index[:, X] += self.n_x_cells / 2
        index[:, Y] += self.n_y_cells / 2
        index = index.to(torch.long)
        pos = pos.unsqueeze(1).expand(pos.shape[0], self.world.batch_dim, 2)
        v = torch.stack([gaussian.log_prob(pos).exp() for gaussian in self.gaussians], dim=-1).sum(-1)[:, env_index]
        if norm:
            v = v / self.max_pdf[env_index]
        sampled = self.sampled[env_index, index[:, 0], index[:, 1]]
        v[sampled + out_of_bounds] = 0
        return v

    def nomrlize_pdf(self, env_index: int=None):
        xpoints = torch.arange(-self.xdim, self.xdim, self.grid_spacing, device=self.world.device)
        ypoints = torch.arange(-self.ydim, self.ydim, self.grid_spacing, device=self.world.device)
        if env_index is not None:
            ygrid, xgrid = torch.meshgrid(ypoints, xpoints, indexing='ij')
            pos = torch.stack((xgrid, ygrid), dim=-1).reshape(-1, 2)
            sample = self.sample_single_env(pos, env_index, norm=False)
            self.max_pdf[env_index] = sample.max()
        else:
            for x in xpoints:
                for y in ypoints:
                    pos = torch.tensor([x, y], device=self.world.device, dtype=torch.float32).repeat(self.world.batch_dim, 1)
                    sample = self.sample(pos, norm=False)
                    self.max_pdf = torch.maximum(self.max_pdf, sample)

    def reward(self, agent: Agent) -> Tensor:
        is_first = self.world.agents.index(agent) == 0
        if is_first:
            for a in self.world.agents:
                a.sample = self.sample(a.state.pos, update_sampled_flag=True, norm=self.norm)
            self.sampling_rew = torch.stack([a.sample for a in self.world.agents], dim=-1).sum(-1)
        return self.sampling_rew if self.shared_rew else agent.sample

    def observation(self, agent: Agent) -> Tensor:
        observations = [agent.state.pos, agent.state.vel, agent.sensors[0].measure()]
        for delta in [[self.grid_spacing, 0], [-self.grid_spacing, 0], [0, self.grid_spacing], [0, -self.grid_spacing], [-self.grid_spacing, -self.grid_spacing], [self.grid_spacing, -self.grid_spacing], [-self.grid_spacing, self.grid_spacing], [self.grid_spacing, self.grid_spacing]]:
            pos = agent.state.pos + torch.tensor(delta, device=self.world.device, dtype=torch.float32)
            sample = self.sample(pos, update_sampled_flag=False).unsqueeze(-1)
            observations.append(sample)
        return torch.cat(observations, dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {'agent_sample': agent.sample}

    def density_for_plot(self, env_index):

        def f(x):
            sample = self.sample_single_env(torch.tensor(x, dtype=torch.float32, device=self.world.device), env_index=env_index)
            return sample
        return f

    def extra_render(self, env_index: int=0):
        from vmas.simulator import rendering
        from vmas.simulator.rendering import render_function_util
        geoms = [render_function_util(f=self.density_for_plot(env_index=env_index), plot_range=(self.xdim, self.ydim), cmap_alpha=self.alpha_plot)]
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
        for i in range(4):
            geom = Line(length=2 * ((self.ydim if i % 2 == 0 else self.xdim) - self.agent_radius) + self.agent_radius * 2).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(0.0 if i % 2 else self.x_semidim + self.agent_radius if i == 0 else -self.x_semidim - self.agent_radius, 0.0 if not i % 2 else self.y_semidim + self.agent_radius if i == 1 else -self.y_semidim - self.agent_radius)
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)
        return geoms

def render_function_util(f: Callable, plot_range: Union[float, Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]], precision: float=0.01, cmap_range: Optional[Tuple[float, float]]=None, cmap_alpha: float=1.0, cmap_name: str='viridis'):
    if isinstance(plot_range, int) or isinstance(plot_range, float):
        x_min = -plot_range
        y_min = -plot_range
        x_max = plot_range
        y_max = plot_range
    elif len(plot_range) == 2:
        if isinstance(plot_range[0], int) or isinstance(plot_range[0], float):
            x_min = -plot_range[0]
            y_min = -plot_range[1]
            x_max = plot_range[0]
            y_max = plot_range[1]
        else:
            x_min = plot_range[0][0]
            y_min = plot_range[1][0]
            x_max = plot_range[0][1]
            y_max = plot_range[1][1]
    xpoints = np.arange(x_min, x_max, precision)
    ypoints = np.arange(y_min, y_max, precision)
    ygrid, xgrid = np.meshgrid(ypoints, xpoints)
    pos = np.stack((xgrid, ygrid), axis=-1).reshape(-1, 2)
    pos_shape = pos.shape
    outputs = f(pos)
    if isinstance(outputs, torch.Tensor):
        outputs = TorchUtils.to_numpy(outputs)
    assert isinstance(outputs, np.ndarray)
    assert outputs.shape[0] == pos_shape[0]
    assert outputs.ndim <= 2
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        outputs = outputs.squeeze(-1)
    elif outputs.ndim == 2:
        assert outputs.shape[1] == 4
    if outputs.ndim == 1:
        if cmap_range is None:
            cmap_range = [None, None]
        outputs = x_to_rgb_colormap(outputs, low=cmap_range[0], high=cmap_range[1], alpha=cmap_alpha, cmap_name=cmap_name)
    img = outputs.reshape(xgrid.shape[0], xgrid.shape[1], outputs.shape[-1])
    img = img * 255
    img = np.transpose(img, (1, 0, 2))
    geom = Image(img, x=x_min, y=y_min, scale=precision)
    return geom

def x_to_rgb_colormap(x: np.ndarray, low: float=None, high: float=None, alpha: float=1.0, cmap_name: str='viridis', cmap_res: int=10):
    from matplotlib import cm
    colormap = cm.get_cmap(cmap_name, cmap_res)(range(cmap_res))[:, :-1]
    if low is None:
        low = np.min(x)
    if high is None:
        high = np.max(x)
    x = np.clip(x, low, high)
    if high - low > 1e-05:
        x = (x - low) / (high - low) * (cmap_res - 1)
    x_c0_idx = np.floor(x).astype(int)
    x_c1_idx = np.ceil(x).astype(int)
    x_c0 = colormap[x_c0_idx, :]
    x_c1 = colormap[x_c1_idx, :]
    t = x - x_c0_idx
    rgb = t[:, None] * x_c1 + (1 - t)[:, None] * x_c0
    colors = np.concatenate([rgb, alpha * np.ones((rgb.shape[0], 1))], axis=-1)
    return colors

class Environment(TorchVectorizedObject):
    """
    The VMAS environment
    """
    metadata = {'render.modes': ['human', 'rgb_array'], 'runtime.vectorized': True}
    vmas_random_state = [torch.random.get_rng_state(), np.random.get_state(), random.getstate()]

    @local_seed(vmas_random_state)
    def __init__(self, scenario: BaseScenario, num_envs: int=32, device: DEVICE_TYPING='cpu', max_steps: Optional[int]=None, continuous_actions: bool=True, seed: Optional[int]=None, dict_spaces: bool=False, multidiscrete_actions: bool=False, clamp_actions: bool=False, grad_enabled: bool=False, terminated_truncated: bool=False, **kwargs):
        if multidiscrete_actions:
            assert not continuous_actions, 'When asking for multidiscrete_actions, make sure continuous_actions=False'
        self.scenario = scenario
        self.num_envs = num_envs
        TorchVectorizedObject.__init__(self, num_envs, torch.device(device))
        self.world = self.scenario.env_make_world(self.num_envs, self.device, **kwargs)
        self.agents = self.world.policy_agents
        self.n_agents = len(self.agents)
        self.max_steps = max_steps
        self.continuous_actions = continuous_actions
        self.dict_spaces = dict_spaces
        self.clamp_action = clamp_actions
        self.grad_enabled = grad_enabled
        self.terminated_truncated = terminated_truncated
        observations = self._reset(seed=seed)
        self.multidiscrete_actions = multidiscrete_actions
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space(observations)
        self.viewer = None
        self.headless = None
        self.visible_display = None
        self.text_lines = None

    @local_seed(vmas_random_state)
    def reset(self, seed: Optional[int]=None, return_observations: bool=True, return_info: bool=False, return_dones: bool=False):
        """
        Resets the environment in a vectorized way
        Returns observations for all envs and agents
        """
        return self._reset(seed=seed, return_observations=return_observations, return_info=return_info, return_dones=return_dones)

    @local_seed(vmas_random_state)
    def reset_at(self, index: int, return_observations: bool=True, return_info: bool=False, return_dones: bool=False):
        """
        Resets the environment at index
        Returns observations for all agents in that environment
        """
        return self._reset_at(index=index, return_observations=return_observations, return_info=return_info, return_dones=return_dones)

    @local_seed(vmas_random_state)
    def get_from_scenario(self, get_observations: bool, get_rewards: bool, get_infos: bool, get_dones: bool, dict_agent_names: Optional[bool]=None):
        """
        Get the environment data from the scenario

        Args:
            get_observations (bool): whether to return the observations
            get_rewards (bool): whether to return the rewards
            get_infos (bool): whether to return the infos
            get_dones (bool): whether to return the dones
            dict_agent_names (bool, optional): whether to return the information in a dictionary with agent names as keys
                or in a list

        Returns:
            The agents' data

        """
        return self._get_from_scenario(get_observations=get_observations, get_rewards=get_rewards, get_infos=get_infos, get_dones=get_dones, dict_agent_names=dict_agent_names)

    @local_seed(vmas_random_state)
    def seed(self, seed=None):
        """
        Sets the seed for the environment
        Args:
            seed (int, optional): Seed for the environment. Defaults to None.

        """
        return self._seed(seed=seed)

    @local_seed(vmas_random_state)
    def done(self):
        """
        Get the done flags for the scenario.

        Returns:
            Either terminated, truncated (if self.terminated_truncated==True) or terminated + truncated (if self.terminated_truncated==False)

        """
        return self._done()

    def _reset(self, seed: Optional[int]=None, return_observations: bool=True, return_info: bool=False, return_dones: bool=False):
        """
        Resets the environment in a vectorized way
        Returns observations for all envs and agents
        """
        if seed is not None:
            self._seed(seed)
        self.scenario.env_reset_world_at(env_index=None)
        self.steps = torch.zeros(self.num_envs, device=self.device)
        result = self._get_from_scenario(get_observations=return_observations, get_infos=return_info, get_rewards=False, get_dones=return_dones)
        return result[0] if result and len(result) == 1 else result

    def _reset_at(self, index: int, return_observations: bool=True, return_info: bool=False, return_dones: bool=False):
        """
        Resets the environment at index
        Returns observations for all agents in that environment
        """
        self._check_batch_index(index)
        self.scenario.env_reset_world_at(index)
        self.steps[index] = 0
        result = self._get_from_scenario(get_observations=return_observations, get_infos=return_info, get_rewards=False, get_dones=return_dones)
        return result[0] if result and len(result) == 1 else result

    def _get_from_scenario(self, get_observations: bool, get_rewards: bool, get_infos: bool, get_dones: bool, dict_agent_names: Optional[bool]=None):
        if not get_infos and (not get_dones) and (not get_rewards) and (not get_observations):
            return
        if dict_agent_names is None:
            dict_agent_names = self.dict_spaces
        obs = rewards = infos = terminated = truncated = dones = None
        if get_observations:
            obs = {} if dict_agent_names else []
        if get_rewards:
            rewards = {} if dict_agent_names else []
        if get_infos:
            infos = {} if dict_agent_names else []
        if get_rewards:
            for agent in self.agents:
                reward = self.scenario.reward(agent).clone()
                if dict_agent_names:
                    rewards.update({agent.name: reward})
                else:
                    rewards.append(reward)
        if get_observations:
            for agent in self.agents:
                observation = TorchUtils.recursive_clone(self.scenario.observation(agent))
                if dict_agent_names:
                    obs.update({agent.name: observation})
                else:
                    obs.append(observation)
        if get_infos:
            for agent in self.agents:
                info = TorchUtils.recursive_clone(self.scenario.info(agent))
                if dict_agent_names:
                    infos.update({agent.name: info})
                else:
                    infos.append(info)
        if self.terminated_truncated:
            if get_dones:
                terminated, truncated = self._done()
            result = [obs, rewards, terminated, truncated, infos]
        else:
            if get_dones:
                dones = self._done()
            result = [obs, rewards, dones, infos]
        return [data for data in result if data is not None]

    def _seed(self, seed=None):
        """
        Sets the seed for the environment
        Args:
            seed (int, optional): Seed for the environment. Defaults to None.

        """
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    @local_seed(vmas_random_state)
    def step(self, actions: Union[List, Dict]):
        """Performs a vectorized step on all sub environments using `actions`.

        Args:
            actions: Is a list on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs, action_size_of_agent)'.

        Returns:
            obs: List on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs, obs_size_of_agent)'
            rewards: List on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs)'
            dones: Tensor of len 'self.num_envs' of which each element is a bool
            infos: List on len 'self.n_agents' of which each element is a dictionary for which each key is a metric and the value is a tensor of shape '(self.num_envs, metric_size_per_agent)'

        Examples:
            >>> import vmas
            >>> env = vmas.make_env(
            ...     scenario="waterfall",  # can be scenario name or BaseScenario class
            ...     num_envs=32,
            ...     device="cpu",  # Or "cuda" for GPU
            ...     continuous_actions=True,
            ...     max_steps=None,  # Defines the horizon. None is infinite horizon.
            ...     seed=None,  # Seed of the environment
            ...     n_agents=3,  # Additional arguments you want to pass to the scenario
            ... )
            >>> obs = env.reset()
            >>> for _ in range(10):
            ...     obs, rews, dones, info = env.step(env.get_random_actions())

        """
        if isinstance(actions, Dict):
            actions_dict = actions
            actions = []
            for agent in self.agents:
                try:
                    actions.append(actions_dict[agent.name])
                except KeyError:
                    raise AssertionError(f"Agent '{agent.name}' not contained in action dict")
            assert len(actions_dict) == self.n_agents, f'Expecting actions for {self.n_agents}, got {len(actions_dict)} actions'
        assert len(actions) == self.n_agents, f'Expecting actions for {self.n_agents}, got {len(actions)} actions'
        for i in range(len(actions)):
            if not isinstance(actions[i], Tensor):
                actions[i] = torch.tensor(actions[i], dtype=torch.float32, device=self.device)
            if len(actions[i].shape) == 1:
                actions[i].unsqueeze_(-1)
            assert actions[i].shape[0] == self.num_envs, f'Actions used in input of env must be of len {self.num_envs}, got {actions[i].shape[0]}'
            assert actions[i].shape[1] == self.get_agent_action_size(self.agents[i]), f'Action for agent {self.agents[i].name} has shape {actions[i].shape[1]}, but should have shape {self.get_agent_action_size(self.agents[i])}'
        for i, agent in enumerate(self.agents):
            self._set_action(actions[i], agent)
        for agent in self.world.agents:
            self.scenario.env_process_action(agent)
        self.scenario.pre_step()
        self.world.step()
        self.scenario.post_step()
        self.steps += 1
        return self._get_from_scenario(get_observations=True, get_infos=True, get_rewards=True, get_dones=True)

    def _done(self):
        """
        Get the done flags for the scenario.

        Returns:
            Either terminated, truncated (if self.terminated_truncated==True) or terminated + truncated (if self.terminated_truncated==False)

        """
        terminated = self.scenario.done().clone()
        if self.max_steps is not None:
            truncated = self.steps >= self.max_steps
        else:
            truncated = None
        if self.terminated_truncated:
            if truncated is None:
                truncated = torch.zeros_like(terminated)
            return (terminated, truncated)
        else:
            if truncated is None:
                return terminated
            return terminated + truncated

    def get_action_space(self):
        if not self.dict_spaces:
            return spaces.Tuple([self.get_agent_action_space(agent) for agent in self.agents])
        else:
            return spaces.Dict({agent.name: self.get_agent_action_space(agent) for agent in self.agents})

    def get_observation_space(self, observations: Union[List, Dict]):
        if not self.dict_spaces:
            return spaces.Tuple([self.get_agent_observation_space(agent, observations[i]) for i, agent in enumerate(self.agents)])
        else:
            return spaces.Dict({agent.name: self.get_agent_observation_space(agent, observations[agent.name]) for agent in self.agents})

    def get_agent_action_size(self, agent: Agent):
        if self.continuous_actions:
            return agent.action.action_size + (self.world.dim_c if not agent.silent else 0)
        elif self.multidiscrete_actions:
            return agent.action_size + (1 if not agent.silent and self.world.dim_c != 0 else 0)
        else:
            return 1

    def get_agent_action_space(self, agent: Agent):
        if self.continuous_actions:
            return spaces.Box(low=np.array((-agent.action.u_range_tensor).tolist() + [0] * (self.world.dim_c if not agent.silent else 0), dtype=np.float32), high=np.array(agent.action.u_range_tensor.tolist() + [1] * (self.world.dim_c if not agent.silent else 0), dtype=np.float32), shape=(self.get_agent_action_size(agent),), dtype=np.float32)
        elif self.multidiscrete_actions:
            actions = agent.discrete_action_nvec + ([self.world.dim_c] if not agent.silent and self.world.dim_c != 0 else [])
            return spaces.MultiDiscrete(actions)
        else:
            return spaces.Discrete(math.prod(agent.discrete_action_nvec) * (self.world.dim_c if not agent.silent and self.world.dim_c != 0 else 1))

    def get_agent_observation_space(self, agent: Agent, obs: AGENT_OBS_TYPE):
        if isinstance(obs, Tensor):
            return spaces.Box(low=-np.float32('inf'), high=np.float32('inf'), shape=obs.shape[1:], dtype=np.float32)
        elif isinstance(obs, Dict):
            return spaces.Dict({key: self.get_agent_observation_space(agent, value) for key, value in obs.items()})
        else:
            raise NotImplementedError(f'Invalid type of observation {obs} for agent {agent.name}')

    @local_seed(vmas_random_state)
    def get_random_action(self, agent: Agent) -> torch.Tensor:
        """Returns a random action for the given agent.

        Args:
            agent (Agent): The agent to get the action for

        Returns:
            torch.tensor: the random actions tensor with shape ``(agent.batch_dim, agent.action_size)``

        """
        if self.continuous_actions:
            actions = []
            for action_index in range(agent.action_size):
                actions.append(torch.zeros(agent.batch_dim, device=agent.device, dtype=torch.float32).uniform_(-agent.action.u_range_tensor[action_index], agent.action.u_range_tensor[action_index]))
            if self.world.dim_c != 0 and (not agent.silent):
                for _ in range(self.world.dim_c):
                    actions.append(torch.zeros(agent.batch_dim, device=agent.device, dtype=torch.float32).uniform_(0, 1))
            action = torch.stack(actions, dim=-1)
        else:
            action_space = self.get_agent_action_space(agent)
            if self.multidiscrete_actions:
                actions = [torch.randint(low=0, high=action_space.nvec[action_index], size=(agent.batch_dim,), device=agent.device) for action_index in range(action_space.shape[0])]
                action = torch.stack(actions, dim=-1)
            else:
                action = torch.randint(low=0, high=action_space.n, size=(agent.batch_dim,), device=agent.device)
        return action

    def get_random_actions(self) -> Sequence[torch.Tensor]:
        """Returns random actions for all agents that you can feed to :meth:`step`

        Returns:
            Sequence[torch.tensor]: the random actions for the agents

        Examples:
            >>> import vmas
            >>> env = vmas.make_env(
            ...     scenario="waterfall",  # can be scenario name or BaseScenario class
            ...     num_envs=32,
            ...     device="cpu",  # Or "cuda" for GPU
            ...     continuous_actions=True,
            ...     max_steps=None,  # Defines the horizon. None is infinite horizon.
            ...     seed=None,  # Seed of the environment
            ...     n_agents=3,  # Additional arguments you want to pass to the scenario
            ... )
            >>> obs = env.reset()
            >>> for _ in range(10):
            ...     obs, rews, dones, info = env.step(env.get_random_actions())

        """
        return [self.get_random_action(agent) for agent in self.agents]

    def _check_discrete_action(self, action: Tensor, low: int, high: int, type: str):
        assert torch.all((action >= torch.tensor(low, device=self.device)) * (action < torch.tensor(high, device=self.device))), f'Discrete {type} actions are out of bounds, allowed int range [{low},{high})'

    def _set_action(self, action, agent):
        action = action.clone()
        if not self.grad_enabled:
            action = action.detach()
        action = action.to(self.device)
        assert not action.isnan().any()
        agent.action.u = torch.zeros(self.batch_dim, agent.action_size, device=self.device, dtype=torch.float32)
        assert action.shape[1] == self.get_agent_action_size(agent), f'Agent {agent.name} has wrong action size, got {action.shape[1]}, expected {self.get_agent_action_size(agent)}'
        if self.clamp_action and self.continuous_actions:
            physical_action = action[..., :agent.action_size]
            a_range = agent.action.u_range_tensor.unsqueeze(0).expand(physical_action.shape)
            physical_action = physical_action.clamp(-a_range, a_range)
            if self.world.dim_c > 0 and (not agent.silent):
                comm_action = action[..., agent.action_size:]
                action = torch.cat([physical_action, comm_action.clamp(0, 1)], dim=-1)
            else:
                action = physical_action
        action_index = 0
        if self.continuous_actions:
            physical_action = action[:, action_index:action_index + agent.action_size]
            action_index += self.world.dim_p
            assert not torch.any(torch.abs(physical_action) > agent.action.u_range_tensor), f'Physical actions of agent {agent.name} are out of its range {agent.u_range}'
            agent.action.u = physical_action.to(torch.float32)
        else:
            if not self.multidiscrete_actions:
                flat_action = action.squeeze(-1)
                actions = []
                nvec = list(agent.discrete_action_nvec) + ([self.world.dim_c] if not agent.silent and self.world.dim_c != 0 else [])
                for i in range(len(nvec)):
                    n = math.prod(nvec[i + 1:])
                    actions.append(flat_action // n)
                    flat_action = flat_action % n
                action = torch.stack(actions, dim=-1)
            for n in agent.discrete_action_nvec:
                physical_action = action[:, action_index]
                self._check_discrete_action(physical_action.unsqueeze(-1), low=0, high=n, type='physical')
                u_max = agent.action.u_range_tensor[action_index]
                if n % 2 != 0:
                    stay = physical_action == 0
                    decrement = (physical_action > 0) & (physical_action <= n // 2)
                    physical_action[stay] = n // 2
                    physical_action[decrement] -= 1
                agent.action.u[:, action_index] = physical_action / (n - 1) * (2 * u_max) - u_max
                action_index += 1
        agent.action.u *= agent.action.u_multiplier_tensor
        if agent.action.u_noise > 0:
            noise = torch.randn(*agent.action.u.shape, device=self.device, dtype=torch.float32) * agent.u_noise
            agent.action.u += noise
        if self.world.dim_c > 0 and (not agent.silent):
            if not self.continuous_actions:
                comm_action = action[:, action_index:]
                self._check_discrete_action(comm_action, 0, self.world.dim_c, 'communication')
                comm_action = comm_action.long()
                agent.action.c = torch.zeros(self.num_envs, self.world.dim_c, device=self.device, dtype=torch.float32)
                agent.action.c.scatter_(1, comm_action, 1)
            else:
                comm_action = action[:, action_index:]
                assert not torch.any(comm_action > 1) and (not torch.any(comm_action < 0)), 'Comm actions are out of range [0,1]'
                agent.action.c = comm_action
            if agent.c_noise > 0:
                noise = torch.randn(*agent.action.c.shape, device=self.device, dtype=torch.float32) * agent.c_noise
                agent.action.c += noise

    @local_seed(vmas_random_state)
    def render(self, mode='human', env_index=0, agent_index_focus: int=None, visualize_when_rgb: bool=False, plot_position_function: Callable=None, plot_position_function_precision: float=0.01, plot_position_function_range: Optional[Union[float, Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]]=None, plot_position_function_cmap_range: Optional[Tuple[float, float]]=None, plot_position_function_cmap_alpha: Optional[float]=1.0, plot_position_function_cmap_name: Optional[str]='viridis'):
        """
        Render function for environment using pyglet

        On servers use mode="rgb_array" and set

        ```
        export DISPLAY=':99.0'
        Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
        ```

        :param mode: One of human or rgb_array
        :param env_index: Index of the environment to render
        :param agent_index_focus: If specified the camera will stay on the agent with this index. If None, the camera will stay in the center and zoom out to contain all agents
        :param visualize_when_rgb: Also run human visualization when mode=="rgb_array"
        :param plot_position_function: A function to plot under the rendering.
        The function takes a numpy array with shape (n_points, 2), which represents a set of x,y values to evaluate f over and plot it
        It should output either an array with shape (n_points, 1) which will be plotted as a colormap
        or an array with shape (n_points, 4), which will be plotted as RGBA values
        :param plot_position_function_precision: The precision to use for plotting the function
        :param plot_position_function_range: The position range to plot the function in.
        If float, the range for x and y is (-function_range, function_range)
        If Tuple[float, float], the range for x is (-function_range[0], function_range[0]) and y is (-function_range[1], function_range[1])
        If Tuple[Tuple[float, float], Tuple[float, float]], the first tuple is the x range and the second tuple is the y range
        :param plot_position_function_cmap_range: The range of the cmap in case plot_position_function outputs a single value
        :param plot_position_function_cmap_alpha: The alpha of the cmap in case plot_position_function outputs a single value
        :return: Rgb array or None, depending on the mode

        """
        self._check_batch_index(env_index)
        assert mode in self.metadata['render.modes'], f'Invalid mode {mode} received, allowed modes: {self.metadata['render.modes']}'
        if agent_index_focus is not None:
            assert 0 <= agent_index_focus < self.n_agents, f'Agent focus in rendering should be a valid agent index between 0 and {self.n_agents}, got {agent_index_focus}'
        shared_viewer = agent_index_focus is None
        aspect_ratio = self.scenario.viewer_size[X] / self.scenario.viewer_size[Y]
        headless = mode == 'rgb_array' and (not visualize_when_rgb)
        if self.visible_display is None:
            self.visible_display = not headless
            self.headless = headless
        else:
            assert self.visible_display is not headless
        if self.viewer is None:
            try:
                import pyglet
            except ImportError:
                raise ImportError("Cannot import pyg;et: you can install pyglet directly via 'pip install pyglet'.")
            try:
                pyglet.lib.load_library('EGL')
                from pyglet.libs.egl import egl, eglext
                num_devices = egl.EGLint()
                eglext.eglQueryDevicesEXT(0, None, byref(num_devices))
                assert num_devices.value > 0
            except (ImportError, AssertionError):
                self.headless = False
            pyglet.options['headless'] = self.headless
            self._init_rendering()
        if self.scenario.viewer_zoom <= 0:
            raise ValueError('Scenario viewer zoom must be > 0')
        zoom = self.scenario.viewer_zoom
        if aspect_ratio < 1:
            cam_range = torch.tensor([zoom, zoom / aspect_ratio], device=self.device)
        else:
            cam_range = torch.tensor([zoom * aspect_ratio, zoom], device=self.device)
        if shared_viewer:
            all_poses = torch.stack([agent.state.pos[env_index] for agent in self.world.agents], dim=0)
            max_agent_radius = max([agent.shape.circumscribed_radius() for agent in self.world.agents])
            viewer_size_fit = torch.stack([torch.max(torch.abs(all_poses[:, X] - self.scenario.render_origin[X])), torch.max(torch.abs(all_poses[:, Y] - self.scenario.render_origin[Y]))]) + 2 * max_agent_radius
            viewer_size = torch.maximum(viewer_size_fit / cam_range, torch.tensor(zoom, device=self.device))
            cam_range *= torch.max(viewer_size)
            self.viewer.set_bounds(-cam_range[X] + self.scenario.render_origin[X], cam_range[X] + self.scenario.render_origin[X], -cam_range[Y] + self.scenario.render_origin[Y], cam_range[Y] + self.scenario.render_origin[Y])
        else:
            pos = self.agents[agent_index_focus].state.pos[env_index]
            self.viewer.set_bounds(pos[X] - cam_range[X], pos[X] + cam_range[X], pos[Y] - cam_range[Y], pos[Y] + cam_range[Y])
        if self.scenario.visualize_semidims:
            self.plot_boundary()
        self._set_agent_comm_messages(env_index)
        if plot_position_function is not None:
            self.viewer.add_onetime(self.plot_function(plot_position_function, precision=plot_position_function_precision, plot_range=plot_position_function_range, cmap_range=plot_position_function_cmap_range, cmap_alpha=plot_position_function_cmap_alpha, cmap_name=plot_position_function_cmap_name))
        from vmas.simulator.rendering import Grid
        if self.scenario.plot_grid:
            grid = Grid(spacing=self.scenario.grid_spacing)
            grid.set_color(*vmas.simulator.utils.Color.BLACK.value, alpha=0.3)
            self.viewer.add_onetime(grid)
        self.viewer.add_onetime_list(self.scenario.extra_render(env_index))
        for entity in self.world.entities:
            self.viewer.add_onetime_list(entity.render(env_index=env_index))
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def plot_boundary(self):
        if self.world.x_semidim is not None or self.world.y_semidim is not None:
            from vmas.simulator.rendering import Line
            from vmas.simulator.utils import Color
            infinite_value = 100
            x_semi = self.world.x_semidim if self.world.x_semidim is not None else infinite_value
            y_semi = self.world.y_semidim if self.world.y_semidim is not None else infinite_value
            color = Color.GRAY.value
            if self.world.x_semidim is not None and self.world.y_semidim is not None or self.world.y_semidim is not None:
                boundary_points = [(-x_semi, y_semi), (x_semi, y_semi), (x_semi, -y_semi), (-x_semi, -y_semi)]
            else:
                boundary_points = [(-x_semi, y_semi), (-x_semi, -y_semi), (x_semi, y_semi), (x_semi, -y_semi)]
            for i in range(0, len(boundary_points), 1 if self.world.x_semidim is not None and self.world.y_semidim is not None else 2):
                start = boundary_points[i]
                end = boundary_points[(i + 1) % len(boundary_points)]
                line = Line(start, end, width=0.7)
                line.set_color(*color)
                self.viewer.add_onetime(line)

    def plot_function(self, f, precision, plot_range, cmap_range, cmap_alpha, cmap_name):
        from vmas.simulator.rendering import render_function_util
        if plot_range is None:
            assert self.viewer.bounds is not None, 'Set viewer bounds before plotting'
            x_min, x_max, y_min, y_max = self.viewer.bounds.tolist()
            plot_range = ([x_min - precision, x_max - precision], [y_min - precision, y_max + precision])
        geom = render_function_util(f=f, precision=precision, plot_range=plot_range, cmap_range=cmap_range, cmap_alpha=cmap_alpha, cmap_name=cmap_name)
        return geom

    def _init_rendering(self):
        from vmas.simulator import rendering
        self.viewer = rendering.Viewer(*self.scenario.viewer_size, visible=self.visible_display)
        self.text_lines = []
        idx = 0
        if self.world.dim_c > 0:
            for agent in self.world.agents:
                if not agent.silent:
                    text_line = rendering.TextLine(y=idx * 40)
                    self.viewer.geoms.append(text_line)
                    self.text_lines.append(text_line)
                    idx += 1

    def _set_agent_comm_messages(self, env_index: int):
        if self.world.dim_c > 0:
            idx = 0
            for agent in self.world.agents:
                if not agent.silent:
                    assert agent.state.c is not None, 'Agent has no comm state but it should'
                    if self.continuous_actions:
                        word = '[' + ','.join([f'{comm:.2f}' for comm in agent.state.c[env_index]]) + ']'
                    else:
                        word = ALPHABET[torch.argmax(agent.state.c[env_index]).item()]
                    message = agent.name + ' sends ' + word + '   '
                    self.text_lines[idx].set_text(message)
                    idx += 1

    @override(TorchVectorizedObject)
    def to(self, device: DEVICE_TYPING):
        device = torch.device(device)
        self.scenario.to(device)
        super().to(device)

class VectorEnvWrapper(rllib.VectorEnv):
    """
    Vector environment wrapper for rllib
    """

    def __init__(self, env: Environment):
        assert not env.terminated_truncated, 'Rllib wrapper is not compatible with termination and truncation flags. Please set `terminated_truncated=False` in the VMAS environment.'
        self._env = env
        super().__init__(observation_space=self._env.observation_space, action_space=self._env.action_space, num_envs=self._env.num_envs)

    @property
    def env(self):
        return self._env

    def vector_reset(self) -> List[EnvObsType]:
        obs = TorchUtils.to_numpy(self._env.reset())
        return self._read_data(obs)[0]

    def reset_at(self, index: Optional[int]=None) -> EnvObsType:
        assert index is not None
        obs = self._env.reset_at(index)
        return self._read_data(obs, env_index=index)[0]

    def vector_step(self, actions: List[EnvActionType]) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        actions = self._action_list_to_tensor(actions)
        obs, rews, dones, infos = TorchUtils.to_numpy(self._env.step(actions))
        obs, infos, rews = self._read_data(obs, infos, rews)
        return (obs, rews, dones, infos)

    def seed(self, seed=None):
        return self._env.seed(seed)

    def try_render_at(self, index: Optional[int]=None, mode='human', agent_index_focus: Optional[int]=None, visualize_when_rgb: bool=False, **kwargs) -> Optional[np.ndarray]:
        """
        Render function for environment using pyglet

        On servers use mode="rgb_array" and set
        ```
        export DISPLAY=':99.0'
        Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
        ```

        :param mode: One of human or rgb_array
        :param index: Index of the environment to render
        :param agent_index_focus: If specified the camera will stay on the agent with this index.
                                  If None, the camera will stay in the center and zoom out to contain all agents
        :param visualize_when_rgb: Also run human visualization when mode=="rgb_array"
        :return: Rgb array or None, depending on the mode
        """
        if index is None:
            index = 0
        return self._env.render(mode=mode, env_index=index, agent_index_focus=agent_index_focus, visualize_when_rgb=visualize_when_rgb, **kwargs)

    def get_sub_environments(self) -> List[Environment]:
        return [self._env]

    def _action_list_to_tensor(self, list_in: List) -> List:
        if len(list_in) == self.num_envs:
            actions = []
            for agent in self._env.agents:
                actions.append(torch.zeros(self.num_envs, self._env.get_agent_action_size(agent), device=self._env.device, dtype=torch.float32))
            for j in range(self.num_envs):
                assert len(list_in[j]) == self._env.n_agents, f'Expecting actions for {self._env.n_agents} agents, got {len(list_in[j])} actions'
                for i in range(self._env.n_agents):
                    act = torch.tensor(list_in[j][i], dtype=torch.float32, device=self._env.device)
                    if len(act.shape) == 0:
                        assert self._env.get_agent_action_size(self._env.agents[i]) == 1, f'Action of agent {i} in env {j} is supposed to be an scalar int'
                    else:
                        assert len(act.shape) == 1 and act.shape[0] == self._env.get_agent_action_size(self._env.agents[i]), f'Action of agent {i} in env {j} hase wrong shape: expected {self._env.get_agent_action_size(self._env.agents[i])}, got {act.shape[0]}'
                    actions[i][j] = act
            return actions
        else:
            raise TypeError('Input action is not in correct format')

    def _read_data(self, obs: Optional[OBS_TYPE], info: Optional[INFO_TYPE]=None, reward: Optional[REWARD_TYPE]=None, env_index: Optional[int]=None):
        if env_index is None:
            obs_list = []
            if info:
                info_list = []
            if reward:
                rew_list = []
            for env_index in range(self.num_envs):
                observations_processed, info_processed, reward_processed = self._get_data_at_env_index(env_index, obs, info, reward)
                obs_list.append(observations_processed)
                if info:
                    info_list.append(info_processed)
                if reward:
                    rew_list.append(reward_processed)
            return (obs_list, info_list if info else None, rew_list if reward else None)
        else:
            return self._get_data_at_env_index(env_index, obs, info, reward)

    def _get_data_at_env_index(self, env_index: int, obs: Optional[OBS_TYPE], info: Optional[INFO_TYPE]=None, reward: Optional[REWARD_TYPE]=None):
        assert len(obs) == self._env.n_agents
        total_rew = 0.0
        if info:
            new_info = {'rewards': {}}
        if isinstance(obs, Dict):
            new_obs = {}
            for agent_index, agent in enumerate(self._env.agents):
                new_obs[agent.name] = self._get_agent_data_at_env_index(env_index, obs[agent.name])
                if info:
                    new_info[agent.name] = self._get_agent_data_at_env_index(env_index, info[agent.name])
                if reward:
                    agent_rew = self._get_agent_data_at_env_index(env_index, reward[agent.name])
                    new_info['rewards'].update({agent_index: agent_rew})
                    total_rew += agent_rew
        elif isinstance(obs, List):
            new_obs = []
            for agent_index, agent in enumerate(self._env.agents):
                new_obs.append(self._get_agent_data_at_env_index(env_index, obs[agent_index]))
                if info:
                    new_info[agent.name] = self._get_agent_data_at_env_index(env_index, info[agent_index])
                if reward:
                    agent_rew = self._get_agent_data_at_env_index(env_index, reward[agent_index])
                    new_info['rewards'].update({agent_index: agent_rew})
                    total_rew += agent_rew
        else:
            raise ValueError(f'Unsupported obs type {obs}')
        return (new_obs, new_info if info else None, total_rew / self._env.n_agents if reward else None)

    def _get_agent_data_at_env_index(self, env_index: int, agent_data):
        if isinstance(agent_data, (ndarray, Tensor)):
            assert agent_data.shape[0] == self._env.num_envs
            if len(agent_data.shape) == 1 or (len(agent_data.shape) == 2 and agent_data.shape[1] == 1):
                return agent_data[env_index].item()
            elif isinstance(agent_data, Tensor):
                return agent_data[env_index].cpu().detach().numpy()
            else:
                return agent_data[env_index]
        elif isinstance(agent_data, Dict):
            return {key: self._get_agent_data_at_env_index(env_index, value) for key, value in agent_data.items()}
        else:
            raise ValueError(f'Unsupported data type {agent_data}')

class BaseGymWrapper(ABC):

    def __init__(self, env: Environment, return_numpy: bool, vectorized: bool):
        self._env = env
        self.return_numpy = return_numpy
        self.dict_spaces = env.dict_spaces
        self.vectorized = vectorized

    @property
    def env(self):
        return self._env

    def _maybe_to_numpy(self, tensor):
        return TorchUtils.to_numpy(tensor) if self.return_numpy else tensor

    def _convert_output(self, data, item: bool=False):
        if not self.vectorized:
            data = extract_nested_with_index(data, index=0)
            if item:
                return data.item()
        return self._maybe_to_numpy(data)

    def _compress_infos(self, infos):
        if isinstance(infos, dict):
            return infos
        elif isinstance(infos, list):
            return {self._env.agents[i].name: info for i, info in enumerate(infos)}
        else:
            raise ValueError(f'Expected list or dictionary for infos but got {type(infos)}')

    def _convert_env_data(self, obs=None, rews=None, info=None, terminated=None, truncated=None, done=None):
        if self.dict_spaces:
            for agent in obs.keys():
                if obs is not None:
                    obs[agent] = self._convert_output(obs[agent])
                if info is not None:
                    info[agent] = self._convert_output(info[agent])
                if rews is not None:
                    rews[agent] = self._convert_output(rews[agent], item=True)
        else:
            for i in range(self._env.n_agents):
                if obs is not None:
                    obs[i] = self._convert_output(obs[i])
                if info is not None:
                    info[i] = self._convert_output(info[i])
                if rews is not None:
                    rews[i] = self._convert_output(rews[i], item=True)
        terminated = self._convert_output(terminated, item=True) if terminated is not None else None
        truncated = self._convert_output(truncated, item=True) if truncated is not None else None
        done = self._convert_output(done, item=True) if done is not None else None
        info = self._compress_infos(info) if info is not None else None
        return EnvData(obs=obs, rews=rews, terminated=terminated, truncated=truncated, done=done, info=info)

    def _action_list_to_tensor(self, list_in: List) -> List:
        assert len(list_in) == self._env.n_agents, f'Expecting actions for {self._env.n_agents} agents, got {len(list_in)} actions'
        dtype = torch.float32 if self._env.continuous_actions else torch.long
        return [torch.tensor(act, device=self._env.device, dtype=dtype).reshape(self._env.num_envs, self._env.get_agent_action_size(agent)) if not isinstance(act, torch.Tensor) else act.to(dtype=dtype, device=self._env.device).reshape(self._env.num_envs, self._env.get_agent_action_size(agent)) for agent, act in zip(self._env.agents, list_in)]

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        raise NotImplementedError

    @abstractmethod
    def render(self, agent_index_focus: Optional[int]=None, visualize_when_rgb: bool=False, **kwargs) -> Optional[np.ndarray]:
        raise NotImplementedError

