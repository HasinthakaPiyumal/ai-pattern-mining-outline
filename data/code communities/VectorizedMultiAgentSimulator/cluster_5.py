# Cluster 5

class InteractiveEnv:
    """
    Use this script to interactively play with scenarios

    You can change agent by pressing TAB
    You can reset the environment by pressing R
    You can control agent actions with the arrow keys and M/N (left/right control the first action, up/down control the second, M/N controls the third)
    If you have more than 1 agent, you can control another one with W,A,S,D and Q,E in the same way.
    and switch the agent with these controls using LSHIFT
    """

    def __init__(self, env: GymWrapper, control_two_agents: bool=False, display_info: bool=True, save_render: bool=False, render_name: str='interactive'):
        self.env = env
        self.control_two_agents = control_two_agents
        self.current_agent_index = 0
        self.current_agent_index2 = 1
        self.n_agents = self.env.unwrapped.n_agents
        self.agents = self.env.unwrapped.agents
        self.continuous = self.env.unwrapped.continuous_actions
        self.reset = False
        self.keys = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.keys2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.u = [0] * (3 if self.continuous else 2)
        self.u2 = [0] * (3 if self.continuous else 2)
        self.frame_list = []
        self.display_info = display_info
        self.save_render = save_render
        self.render_name = render_name
        if self.control_two_agents:
            assert self.n_agents >= 2, 'Control_two_agents is true but not enough agents in scenario'
        self.text_lines = []
        self.font_size = 15
        self.env.render()
        self.text_idx = len(self.env.unwrapped.text_lines)
        self._init_text()
        self.env.unwrapped.viewer.window.on_key_press = self._key_press
        self.env.unwrapped.viewer.window.on_key_release = self._key_release
        self._cycle()

    def _increment_selected_agent_index(self, index: int):
        index += 1
        if index == self.n_agents:
            index = 0
        return index

    def _cycle(self):
        total_rew = [0] * self.n_agents
        while True:
            if self.reset:
                if self.save_render:
                    save_video(self.render_name, self.frame_list, fps=1 / self.env.unwrapped.world.dt)
                self.env.reset()
                self.reset = False
                total_rew = [0] * self.n_agents
            if self.n_agents > 0:
                action_list = [[0.0] * agent.action_size for agent in self.agents]
                action_list[self.current_agent_index][:self.agents[self.current_agent_index].dynamics.needed_action_size] = self.u[:self.agents[self.current_agent_index].dynamics.needed_action_size]
            else:
                action_list = []
            if self.n_agents > 1 and self.control_two_agents:
                action_list[self.current_agent_index2][:self.agents[self.current_agent_index2].dynamics.needed_action_size] = self.u2[:self.agents[self.current_agent_index2].dynamics.needed_action_size]
            obs, rew, done, info = self.env.step(action_list)
            if self.display_info and self.n_agents > 0:
                obs_str = str(InteractiveEnv.format_obs(obs[self.current_agent_index]))
                message = f'\t\t{obs_str[len(obs_str) // 2:]}'
                self._write_values(0, message)
                message = f'Obs: {obs_str[:len(obs_str) // 2]}'
                self._write_values(1, message)
                message = f'Rew: {round(rew[self.current_agent_index], 3)}'
                self._write_values(2, message)
                total_rew = list(map(add, total_rew, rew))
                message = f'Total rew: {round(total_rew[self.current_agent_index], 3)}'
                self._write_values(3, message)
                message = f'Done: {done}'
                self._write_values(4, message)
                message = f'Selected: {self.env.unwrapped.agents[self.current_agent_index].name}'
                self._write_values(5, message)
            frame = self.env.render(mode='rgb_array' if self.save_render else 'human', visualize_when_rgb=True)
            if self.save_render:
                self.frame_list.append(frame)
            if done:
                self.reset = True

    def _init_text(self):
        from vmas.simulator import rendering
        for i in range(N_TEXT_LINES_INTERACTIVE):
            text_line = rendering.TextLine(y=(self.text_idx + i) * 40, font_size=self.font_size)
            self.env.unwrapped.viewer.add_geom(text_line)
            self.text_lines.append(text_line)

    def _write_values(self, index: int, message: str):
        self.text_lines[index].set_text(message)

    def _key_press(self, k, mod):
        from pyglet.window import key
        agent_range = self.agents[self.current_agent_index].action.u_range_tensor
        try:
            if k == key.LEFT:
                self.keys[0] = agent_range[0]
            elif k == key.RIGHT:
                self.keys[1] = agent_range[0]
            elif k == key.DOWN:
                self.keys[2] = agent_range[1]
            elif k == key.UP:
                self.keys[3] = agent_range[1]
            elif k == key.M:
                self.keys[4] = agent_range[2]
            elif k == key.N:
                self.keys[5] = agent_range[2]
            elif k == key.TAB:
                self.current_agent_index = self._increment_selected_agent_index(self.current_agent_index)
                if self.control_two_agents:
                    while self.current_agent_index == self.current_agent_index2:
                        self.current_agent_index = self._increment_selected_agent_index(self.current_agent_index)
            if self.control_two_agents:
                agent2_range = self.agents[self.current_agent_index2].action.u_range_tensor
                if k == key.A:
                    self.keys2[0] = agent2_range[0]
                elif k == key.D:
                    self.keys2[1] = agent2_range[0]
                elif k == key.S:
                    self.keys2[2] = agent2_range[1]
                elif k == key.W:
                    self.keys2[3] = agent2_range[1]
                elif k == key.E:
                    self.keys2[4] = agent2_range[2]
                elif k == key.Q:
                    self.keys2[5] = agent2_range[2]
                elif k == key.LSHIFT:
                    self.current_agent_index2 = self._increment_selected_agent_index(self.current_agent_index2)
                    while self.current_agent_index == self.current_agent_index2:
                        self.current_agent_index2 = self._increment_selected_agent_index(self.current_agent_index2)
        except IndexError:
            print('Action not available')
        if k == key.R:
            self.reset = True
        self.set_u()

    def _key_release(self, k, mod):
        from pyglet.window import key
        if k == key.LEFT:
            self.keys[0] = 0
        elif k == key.RIGHT:
            self.keys[1] = 0
        elif k == key.DOWN:
            self.keys[2] = 0
        elif k == key.UP:
            self.keys[3] = 0
        elif k == key.M:
            self.keys[4] = 0
        elif k == key.N:
            self.keys[5] = 0
        if self.control_two_agents:
            if k == key.A:
                self.keys2[0] = 0
            elif k == key.D:
                self.keys2[1] = 0
            elif k == key.S:
                self.keys2[2] = 0
            elif k == key.W:
                self.keys2[3] = 0
            elif k == key.E:
                self.keys2[4] = 0
            elif k == key.Q:
                self.keys2[5] = 0
        self.set_u()

    def set_u(self):
        if self.continuous:
            self.u = [self.keys[1] - self.keys[0], self.keys[3] - self.keys[2], self.keys[4] - self.keys[5]]
            self.u2 = [self.keys2[1] - self.keys2[0], self.keys2[3] - self.keys2[2], self.keys2[4] - self.keys2[5]]
        else:
            if np.sum(self.keys[:4]) >= 1:
                self.u[0] = np.argmax(self.keys[:4]) + 1
            else:
                self.u[0] = 0
            if np.sum(self.keys[4:]) >= 1:
                self.u[1] = np.argmax(self.keys[4:]) + 1
            else:
                self.u[1] = 0
            if np.sum(self.keys2[:4]) >= 1:
                self.u2[0] = np.argmax(self.keys2[:4]) + 1
            else:
                self.u2[0] = 0
            if np.sum(self.keys2[4:]) >= 1:
                self.u2[1] = np.argmax(self.keys2[4:]) + 1
            else:
                self.u2[1] = 0

    @staticmethod
    def format_obs(obs):
        if isinstance(obs, (Tensor, np.ndarray)):
            return list(np.around(obs.tolist(), decimals=2))
        elif isinstance(obs, Dict):
            return {key: InteractiveEnv.format_obs(value) for key, value in obs.items()}
        else:
            raise NotImplementedError(f'Invalid type of observation {obs}')

def save_video(name: str, frame_list: List[np.array], fps: int):
    """Requres cv2"""
    import cv2
    video_name = name + '.mp4'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_list[0].shape[1], frame_list[0].shape[0]))
    for img in frame_list:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)
    video.release()

def make_env(scenario: Union[str, BaseScenario], num_envs: int, device: DEVICE_TYPING='cpu', continuous_actions: bool=True, wrapper: Optional[Union[Wrapper, str]]=None, max_steps: Optional[int]=None, seed: Optional[int]=None, dict_spaces: bool=False, multidiscrete_actions: bool=False, clamp_actions: bool=False, grad_enabled: bool=False, terminated_truncated: bool=False, wrapper_kwargs: Optional[dict]=None, **kwargs):
    """Create a vmas environment.

    Args:
        scenario (Union[str, BaseScenario]): Scenario to load.
            Can be the name of a file in `vmas.scenarios` folder or a :class:`~vmas.simulator.scenario.BaseScenario` class,
        num_envs (int): Number of vectorized simulation environments. VMAS performs vectorized simulations using PyTorch.
            This argument indicates the number of vectorized environments that should be simulated in a batch. It will also
            determine the batch size of the environment.
        device (Union[str, int, torch.device], optional): Device for simulation. All the tensors created by VMAS
            will be placed on this device. Default is ``"cpu"``,
        continuous_actions (bool, optional): Whether to use continuous actions. If ``False``, actions
            will be discrete. The number of actions and their size will depend on the chosen scenario. Default is ``True``,
        wrapper (Union[Wrapper, str], optional): Wrapper class to use. For example, it can be
            ``"rllib"``, ``"gym"``, ``"gymnasium"``, ``"gymnasium_vec"``. Default is ``None``.
        max_steps (int, optional): Horizon of the task. Defaults to ``None`` (infinite horizon). Each VMAS scenario can
            be terminating or not. If ``max_steps`` is specified,
            the scenario is also terminated whenever this horizon is reached,
        seed (int, optional): Seed for the environment. Defaults to ``None``,
        dict_spaces (bool, optional):  Weather to use dictionaries spaces with format ``{"agent_name": tensor, ...}``
            for obs, rewards, and info instead of tuples. Defaults to ``False``: obs, rewards, info are tuples with length number of agents,
        multidiscrete_actions (bool, optional): Whether to use multidiscrete action spaces when ``continuous_actions=False``.
            Default is ``False``: the action space will be ``Discrete``, and it will be the cartesian product of the
            discrete action spaces available to an agent,
        clamp_actions (bool, optional): Weather to clamp input actions to their range instead of throwing
            an error when ``continuous_actions==True`` and actions are out of bounds,
        grad_enabled (bool, optional): If ``True`` the simulator will not call ``detach()`` on input actions and gradients can
            be taken from the simulator output. Default is ``False``.
        terminated_truncated (bool, optional): Weather to use terminated and truncated flags in the output of the step method (or single done).
            Default is ``False``.
        wrapper_kwargs (dict, optional): Keyword arguments to pass to the wrapper class. Default is ``{}``.
        **kwargs (dict, optional): Keyword arguments to pass to the :class:`~vmas.simulator.scenario.BaseScenario` class.

    Examples:
        >>> from vmas import make_env
        >>> env = make_env(
        ...     "waterfall",
        ...     num_envs=3,
        ...     num_agents=2,
        ... )
        >>> print(env.reset())


    """
    if isinstance(scenario, str):
        if not scenario.endswith('.py'):
            scenario += '.py'
        scenario = scenarios.load(scenario).Scenario()
    env = Environment(scenario, num_envs=num_envs, device=device, continuous_actions=continuous_actions, max_steps=max_steps, seed=seed, dict_spaces=dict_spaces, multidiscrete_actions=multidiscrete_actions, clamp_actions=clamp_actions, grad_enabled=grad_enabled, terminated_truncated=terminated_truncated, **kwargs)
    if wrapper is not None and isinstance(wrapper, str):
        wrapper = Wrapper[wrapper.upper()]
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    return wrapper.get_env(env, **wrapper_kwargs) if wrapper is not None else env

def use_vmas_env(render: bool=False, save_render: bool=False, num_envs: int=32, n_steps: int=100, random_action: bool=False, device: str='cpu', scenario_name: str='waterfall', continuous_actions: bool=True, visualize_render: bool=True, dict_spaces: bool=True, **kwargs):
    """Example function to use a vmas environment

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario_name (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action
        visualize_render (bool, optional): Whether to visualize the render. Defaults to ``True``.
        dict_spaces (bool, optional): Weather to return obs, rewards, and infos as dictionaries with agent names.
            By default, they are lists of len # of agents
        kwargs (dict, optional): Keyword arguments to pass to the scenario

    Returns:

    """
    assert not (save_render and (not render)), 'To save the video you have to render it'
    env = make_env(scenario=scenario_name, num_envs=num_envs, device=device, continuous_actions=continuous_actions, dict_spaces=dict_spaces, wrapper=None, seed=None, **kwargs)
    frame_list = []
    init_time = time.time()
    step = 0
    for _ in range(n_steps):
        step += 1
        print(f'Step {step}')
        dict_actions = random.choice([True, False])
        actions = {} if dict_actions else []
        for agent in env.agents:
            if not random_action:
                action = _get_deterministic_action(agent, continuous_actions, env)
            else:
                action = env.get_random_action(agent)
            if dict_actions:
                actions.update({agent.name: action})
            else:
                actions.append(action)
        obs, rews, dones, info = env.step(actions)
        if render:
            frame = env.render(mode='rgb_array', agent_index_focus=None, visualize_when_rgb=visualize_render)
            if save_render:
                frame_list.append(frame)
    total_time = time.time() - init_time
    print(f'It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} for {scenario_name} scenario.')
    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)

def _get_deterministic_action(agent: Agent, continuous: bool, env):
    if continuous:
        action = -agent.action.u_range_tensor.expand(env.batch_dim, agent.action_size)
    else:
        action = torch.tensor([1], device=env.device, dtype=torch.long).unsqueeze(-1).expand(env.batch_dim, 1)
    return action.clone()

def run_heuristic(scenario_name: str, heuristic: Type[BaseHeuristicPolicy]=RandomPolicy, n_steps: int=200, n_envs: int=32, env_kwargs: dict=None, render: bool=False, save_render: bool=False, device: str='cpu'):
    assert not (save_render and (not render)), 'To save the video you have to render it'
    if env_kwargs is None:
        env_kwargs = {}
    policy = heuristic(continuous_action=True)
    env = make_env(scenario=scenario_name, num_envs=n_envs, device=device, continuous_actions=True, wrapper=None, **env_kwargs)
    frame_list = []
    init_time = time.time()
    step = 0
    obs = env.reset()
    total_reward = 0
    for _ in range(n_steps):
        step += 1
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
        obs, rews, dones, info = env.step(actions)
        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward
        if render:
            frame_list.append(env.render(mode='rgb_array', agent_index_focus=None, visualize_when_rgb=True))
    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)
    print(f'It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\nThe average total reward was {total_reward}')

def test_all_scenarios_included():
    from vmas import debug_scenarios, mpe_scenarios, scenarios
    assert sorted(scenario_names()) == sorted(scenarios + mpe_scenarios + debug_scenarios)

def scenario_names():
    scenarios = []
    scenarios_folder = Path(__file__).parent.parent / 'vmas' / 'scenarios'
    for path in scenarios_folder.glob('**/*.py'):
        if path.is_file() and (not path.name.startswith('__')):
            scenarios.append(path.stem)
    return scenarios

@pytest.mark.parametrize('scenario', scenario_names())
@pytest.mark.parametrize('continuous_actions', [True, False])
def test_use_vmas_env(scenario, continuous_actions, dict_spaces=True, num_envs=10, n_steps=10):
    render = True
    if sys.platform.startswith('win32'):
        render = False
    use_vmas_env(render=render, save_render=False, visualize_render=False, random_action=True, device='cpu', scenario_name=scenario, continuous_actions=continuous_actions, num_envs=num_envs, n_steps=n_steps, dict_spaces=dict_spaces)

@pytest.mark.parametrize('scenario', scenario_names())
def test_multi_discrete_actions(scenario, num_envs=10, n_steps=10):
    env = make_env(scenario=scenario, num_envs=num_envs, seed=0, multidiscrete_actions=True, continuous_actions=False)
    for _ in range(n_steps):
        env.step(env.get_random_actions())

@pytest.mark.parametrize('scenario', scenario_names())
@pytest.mark.parametrize('multidiscrete_actions', [True, False])
def test_discrete_action_nvec(scenario, multidiscrete_actions, num_envs=10, n_steps=5):
    env = make_env(scenario=scenario, num_envs=num_envs, seed=0, multidiscrete_actions=multidiscrete_actions, continuous_actions=False)
    if type(env.scenario).process_action is not vmas.simulator.scenario.BaseScenario.process_action:
        pytest.skip('Scenario uses a custom process_action method.')
    random.seed(0)
    for agent in env.world.agents:
        agent.discrete_action_nvec = [random.randint(2, 6) for _ in range(agent.action_size)]
    env.action_space = env.get_action_space()

    def to_multidiscrete(action, nvec):
        action_multi = []
        for i in range(len(nvec)):
            n = math.prod(nvec[i + 1:])
            action_multi.append(action // n)
            action = action % n
        return torch.stack(action_multi, dim=-1)

    def full_nvec(agent, world):
        return list(agent.discrete_action_nvec) + ([world.dim_c] if not agent.silent and world.dim_c != 0 else [])
    for _ in range(n_steps):
        actions = env.get_random_actions()
        for a_batch, s in zip(actions, env.action_space.spaces):
            for a in a_batch:
                assert a.numpy() in s
        env.step(actions)
        if not multidiscrete_actions:
            actions = [to_multidiscrete(a.squeeze(-1), full_nvec(agent, env.world)) for a, agent in zip(actions, env.world.policy_agents)]
        for i_a, agent in enumerate(env.world.policy_agents):
            for i, n in enumerate(agent.discrete_action_nvec):
                a = actions[i_a][:, i]
                u = agent.action.u[:, i]
                U = agent.action.u_range_tensor[i]
                k = agent.action.u_multiplier_tensor[i]
                for aj, uj in zip(a, u):
                    assert aj in range(n), f'discrete action {aj} not in [0,{n - 1}] (n={n}, U={U}, k={k})'
                    if n % 2 != 0:
                        assert aj != 0 or uj == 0, f'discrete action {aj} maps to control {uj} (n={n}), U={U}, k={k})'
                        assert (aj < 1 or aj > n // 2) or torch.isclose(uj / k, 2 * U * (aj - 1) / (n - 1) - U), f'discrete action {aj} maps to control {uj} (n={n}, U={U}, k={k})'
                        assert aj <= n // 2 or torch.isclose(uj / k, 2 * U * (aj / (n - 1)) - U), f'discrete action {aj} maps to control {uj} (n={n}), U={U}, k={k})'
                    else:
                        assert torch.isclose(uj / k, 2 * U * (aj / (n - 1)) - U), f'discrete action {aj} maps to control {uj} (n={n}), U={U}, k={k})'

def to_multidiscrete(action, nvec):
    action_multi = []
    for i in range(len(nvec)):
        n = math.prod(nvec[i + 1:])
        action_multi.append(action // n)
        action = action % n
    return torch.stack(action_multi, dim=-1)

@pytest.mark.parametrize('scenario', scenario_names())
def test_non_dict_spaces_actions(scenario, num_envs=10, n_steps=10):
    env = make_env(scenario=scenario, num_envs=num_envs, seed=0, continuous_actions=True, dict_spaces=False)
    for _ in range(n_steps):
        env.step(env.get_random_actions())

@pytest.mark.parametrize('scenario', scenario_names())
def test_partial_reset(scenario, num_envs=10, n_steps=10):
    env = make_env(scenario=scenario, num_envs=num_envs, seed=0)
    env_index = 0
    for _ in range(n_steps):
        env.step(env.get_random_actions())
        env.reset_at(env_index)
        env_index += 1
        if env_index >= num_envs:
            env_index = 0

@pytest.mark.parametrize('scenario', scenario_names())
def test_global_reset(scenario, num_envs=10, n_steps=10):
    env = make_env(scenario=scenario, num_envs=num_envs, seed=0)
    for step in range(n_steps):
        env.step(env.get_random_actions())
        if step == n_steps // 2:
            env.reset()

@pytest.mark.parametrize('scenario', vmas.scenarios + vmas.mpe_scenarios)
def test_vmas_differentiable(scenario, n_steps=10, n_envs=10):
    if scenario == 'football' or scenario == 'simple_crypto' or scenario == 'road_traffic':
        pytest.skip()
    env = make_env(scenario=scenario, num_envs=n_envs, continuous_actions=True, seed=0, grad_enabled=True)
    for step in range(n_steps):
        actions = []
        for agent in env.agents:
            action = env.get_random_action(agent)
            action.requires_grad_(True)
            if step == 0:
                first_action = action
            actions.append(action)
        obs, rews, dones, info = env.step(actions)
    loss = obs[-1].mean() + rews[-1].mean()
    grad = torch.autograd.grad(loss, first_action)

def test_seeding():
    env = make_env(scenario='balance', num_envs=2, seed=0)
    env.seed(0)
    random_obs = env.reset()[0][0, 0]
    env.seed(0)
    assert random_obs == env.reset()[0][0, 0]
    env.seed(0)
    torch.manual_seed(1)
    assert random_obs == env.reset()[0][0, 0]
    torch.manual_seed(0)
    random_obs = torch.randn(1)
    torch.manual_seed(0)
    env.seed(1)
    env.reset()
    assert random_obs == torch.randn(1)

def test_vectorized_lidar(n_envs=12, n_steps=15):

    def get_obs(env):
        rollout_obs = []
        for _ in range(n_steps):
            obs, _, _, _ = env.step(env.get_random_actions())
            obs = torch.stack(obs, dim=-1)
            rollout_obs.append(obs)
        return torch.stack(rollout_obs, dim=-1)
    env_vec_lidar = make_env(scenario='pollock', num_envs=n_envs, seed=0, lidar=True, vectorized_lidar=True)
    obs_vec_lidar = get_obs(env_vec_lidar)
    env_non_vec_lidar = make_env(scenario='pollock', num_envs=n_envs, seed=0, lidar=True, vectorized_lidar=False)
    obs_non_vec_lidar = get_obs(env_non_vec_lidar)
    assert torch.allclose(obs_vec_lidar, obs_non_vec_lidar)

def get_obs(env):
    rollout_obs = []
    for _ in range(n_steps):
        obs, _, _, _ = env.step(env.get_random_actions())
        obs = torch.stack(obs, dim=-1)
        rollout_obs.append(obs)
    return torch.stack(rollout_obs, dim=-1)

class TestNavigation:

    def setUp(self, n_envs, n_agents) -> None:
        self.continuous_actions = True
        self.env = make_env(scenario='navigation', num_envs=n_envs, device='cpu', continuous_actions=self.continuous_actions, n_agents=n_agents)
        self.env.seed(0)

    @pytest.mark.parametrize('n_agents', [1])
    def test_heuristic(self, n_agents, n_envs=5):
        self.setUp(n_envs=n_envs, n_agents=n_agents)
        policy = HeuristicPolicy(continuous_action=self.continuous_actions, clf_epsilon=0.4, clf_slack=100.0)
        obs = self.env.reset()
        all_done = torch.zeros(n_envs, dtype=torch.bool)
        while not all_done.all():
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                action_agent = policy.compute_action(obs_agent, self.env.agents[i].action.u_range_tensor)
                actions.append(action_agent)
            obs, new_rews, dones, _ = self.env.step(actions)
            if dones.any():
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)

class TestTransport:

    def setup_env(self, n_envs, **kwargs) -> None:
        self.n_agents = kwargs.get('n_agents', 4)
        self.n_packages = kwargs.get('n_packages', 1)
        self.package_width = kwargs.get('package_width', 0.15)
        self.package_length = kwargs.get('package_length', 0.15)
        self.package_mass = kwargs.get('package_mass', 50)
        self.continuous_actions = True
        self.env = make_env(scenario='transport', num_envs=n_envs, device='cpu', continuous_actions=self.continuous_actions, **kwargs)
        self.env.seed(0)

    def test_not_passing_through_packages(self, n_agents=1, n_envs=4):
        self.setup_env(n_agents=n_agents, n_envs=n_envs)
        for _ in range(10):
            obs = self.env.reset()
            for _ in range(100):
                obs_agent = obs[0]
                assert (torch.linalg.vector_norm(obs_agent[:, 6:8], dim=1) > self.env.agents[0].shape.radius).all()
                action_agent = torch.clamp(obs_agent[:, 6:8], min=-self.env.agents[0].u_range, max=self.env.agents[0].u_range)
                action_agent /= torch.linalg.vector_norm(action_agent, dim=1).unsqueeze(-1)
                action_agent *= self.env.agents[0].u_range
                obs, rews, dones, _ = self.env.step([action_agent])

    @pytest.mark.parametrize('n_agents', [6])
    def test_heuristic(self, n_agents, n_envs=4):
        self.setup_env(n_agents=n_agents, n_envs=n_envs)
        policy = transport.HeuristicPolicy(self.continuous_actions)
        obs = self.env.reset()
        all_done = torch.zeros(n_envs, dtype=torch.bool)
        while not all_done.all():
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                action_agent = policy.compute_action(obs_agent, self.env.agents[i].u_range)
                actions.append(action_agent)
            obs, new_rews, dones, _ = self.env.step(actions)
            if dones.any():
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)

class TestWheel:

    def setup_env(self, n_envs, n_agents, **kwargs) -> None:
        self.desired_velocity = kwargs.get('desired_velocity', 0.1)
        self.continuous_actions = True
        self.n_envs = 15
        self.env = make_env(scenario='wheel', num_envs=n_envs, device='cpu', continuous_actions=self.continuous_actions, n_agents=n_agents, **kwargs)
        self.env.seed(0)

    @pytest.mark.parametrize('n_agents', [2, 10])
    def test_heuristic(self, n_agents, n_steps=50, n_envs=4):
        line_length = 2
        self.setup_env(n_agents=n_agents, line_length=line_length, n_envs=n_envs)
        policy = wheel.HeuristicPolicy(self.continuous_actions)
        obs = self.env.reset()
        for _ in range(n_steps):
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                action_agent = policy.compute_action(obs_agent, self.env.agents[i].u_range)
                actions.append(action_agent)
            obs, new_rews, dones, _ = self.env.step(actions)

class TestDispersion:

    def setup_env(self, n_agents: int, share_reward: bool, penalise_by_time: bool, n_envs) -> None:
        self.n_agents = n_agents
        self.share_reward = share_reward
        self.penalise_by_time = penalise_by_time
        self.continuous_actions = True
        self.env = make_env(scenario='dispersion', num_envs=n_envs, device='cpu', continuous_actions=self.continuous_actions, n_agents=self.n_agents, share_reward=self.share_reward, penalise_by_time=self.penalise_by_time)
        self.env.seed(0)

    @pytest.mark.parametrize('n_agents', [1, 5, 10])
    def test_heuristic(self, n_agents, n_envs=4):
        self.setup_env(n_agents=n_agents, share_reward=False, penalise_by_time=False, n_envs=n_envs)
        all_done = torch.full((n_envs,), False)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs, n_agents)
        while not all_done.all():
            actions = []
            idx = 0
            for i in range(n_agents):
                obs_agent = obs[i]
                obs_idx = 4 + idx
                action_agent = torch.clamp(obs_agent[:, obs_idx:obs_idx + 2], min=-self.env.agents[i].u_range, max=self.env.agents[i].u_range)
                idx += 3
                actions.append(action_agent)
            obs, rews, dones, _ = self.env.step(actions)
            for i in range(n_agents):
                total_rew[:, i] += rews[i]
            if dones.any():
                assert torch.equal(total_rew[dones].sum(-1).to(torch.long), torch.full((dones.sum(),), n_agents))
                total_rew[dones] = 0
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)

    @pytest.mark.parametrize('n_agents', [1, 5, 10, 20])
    def test_heuristic_share_reward(self, n_agents, n_envs=4):
        self.setup_env(n_agents=n_agents, share_reward=True, penalise_by_time=False, n_envs=n_envs)
        all_done = torch.full((n_envs,), False)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs, n_agents)
        while not all_done.all():
            actions = []
            idx = 0
            for i in range(n_agents):
                obs_agent = obs[i]
                obs_idx = 4 + idx
                action_agent = torch.clamp(obs_agent[:, obs_idx:obs_idx + 2], min=-self.env.agents[i].u_range, max=self.env.agents[i].u_range)
                idx += 3
                actions.append(action_agent)
            obs, rews, dones, _ = self.env.step(actions)
            for i in range(n_agents):
                total_rew[:, i] += rews[i]
            if dones.any():
                assert torch.equal(total_rew[dones], torch.full((dones.sum(), n_agents), n_agents).to(torch.float))
                total_rew[dones] = 0
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)

class TestDiscovery:

    def setup_env(self, n_envs, **kwargs) -> None:
        self.env = make_env(scenario='discovery', num_envs=n_envs, device='cpu', **kwargs)
        self.env.seed(0)

    @pytest.mark.parametrize('n_agents', [1, 4])
    @pytest.mark.parametrize('agent_lidar', [True, False])
    def test_heuristic(self, n_agents, agent_lidar, n_steps=50, n_envs=4):
        self.setup_env(n_agents=n_agents, n_envs=n_envs, use_agent_lidar=agent_lidar)
        policy = discovery.HeuristicPolicy(True)
        obs = self.env.reset()
        for _ in range(n_steps):
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                action_agent = policy.compute_action(obs_agent, self.env.agents[i].u_range)
                actions.append(action_agent)
            obs, new_rews, dones, _ = self.env.step(actions)

class TestDropout:

    def setup_env(self, n_agents: int, num_envs: int, energy_coeff: float=DEFAULT_ENERGY_COEFF) -> None:
        self.n_agents = n_agents
        self.energy_coeff = energy_coeff
        self.continuous_actions = True
        self.n_envs = num_envs
        self.env = make_env(scenario='dropout', num_envs=num_envs, device='cpu', continuous_actions=self.continuous_actions, n_agents=self.n_agents, energy_coeff=self.energy_coeff)
        self.env.seed(0)

    @pytest.mark.parametrize('n_agents', [1, 5])
    def test_heuristic(self, n_agents, n_envs=4):
        self.setup_env(n_agents=n_agents, num_envs=n_envs)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs)
        current_min = float('inf')
        best_i = None
        for i in range(n_agents):
            obs_agent = obs[i]
            if torch.linalg.vector_norm(obs_agent[:, -3:-1], dim=1)[0] < current_min:
                current_min = torch.linalg.vector_norm(obs_agent[:, -3:-1], dim=1)[0]
                best_i = i
        done = False
        while not done:
            obs_agent = obs[best_i]
            action_agent = torch.clamp(obs_agent[:, -3:-1], min=-self.env.agents[best_i].u_range, max=self.env.agents[best_i].u_range)
            actions = []
            other_agents_action = torch.zeros(self.env.num_envs, self.env.world.dim_p)
            for j in range(self.n_agents):
                if best_i != j:
                    actions.append(other_agents_action)
                else:
                    actions.append(action_agent)
            obs, new_rews, dones, _ = self.env.step(actions)
            for j in range(self.n_agents):
                assert torch.equal(new_rews[0], new_rews[j])
            total_rew += new_rews[0]
            assert (total_rew[dones] > 0).all()
            done = dones.any()

    @pytest.mark.parametrize('n_agents', [1, 5])
    def test_one_random_agent_can_do_it(self, n_agents, n_steps=50, n_envs=4):
        self.setup_env(n_agents=n_agents, num_envs=n_envs)
        for i in range(self.n_agents):
            obs = self.env.reset()
            total_rew = torch.zeros(self.env.num_envs)
            for _ in range(n_steps):
                obs_agent = obs[i]
                action_agent = torch.clamp(obs_agent[:, -3:-1], min=-self.env.agents[i].u_range, max=self.env.agents[i].u_range)
                actions = []
                other_agents_action = torch.zeros(self.env.num_envs, self.env.world.dim_p)
                for j in range(self.n_agents):
                    if i != j:
                        actions.append(other_agents_action)
                    else:
                        actions.append(action_agent)
                obs, new_rews, dones, _ = self.env.step(actions)
                for j in range(self.n_agents):
                    assert torch.equal(new_rews[0], new_rews[j])
                total_rew += new_rews[0]
                assert (total_rew[dones] > 0).all()
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)
                total_rew[dones] = 0

    @pytest.mark.parametrize('n_agents', [5, 10])
    def test_all_agents_cannot_do_it(self, n_agents):
        assert self.all_agents(DEFAULT_ENERGY_COEFF, n_agents) < 0
        assert self.all_agents(0, n_agents) > 0

    def all_agents(self, energy_coeff: float, n_agents: int, n_steps=100, n_envs=4):
        rewards = []
        self.setup_env(n_agents=n_agents, energy_coeff=energy_coeff, num_envs=n_envs)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs)
        for _ in range(n_steps):
            actions = []
            for i in range(self.n_agents):
                obs_i = obs[i]
                action_i = torch.clamp(obs_i[:, -3:-1], min=-self.env.agents[i].u_range, max=self.env.agents[i].u_range)
                actions.append(action_i)
            obs, new_rews, dones, _ = self.env.step(actions)
            for j in range(self.n_agents):
                assert torch.equal(new_rews[0], new_rews[j])
            total_rew += new_rews[0]
            for env_index, done in enumerate(dones):
                if done:
                    self.env.reset_at(env_index)
            if dones.any():
                rewards.append(total_rew[dones].clone())
            total_rew[dones] = 0
        return sum([rew.mean().item() for rew in rewards]) / len(rewards)

class TestBalance:

    def setup_env(self, n_envs, **kwargs) -> None:
        self.n_agents = kwargs.get('n_agents', 4)
        self.continuous_actions = True
        self.env = make_env(scenario='balance', num_envs=n_envs, device='cpu', continuous_actions=self.continuous_actions, **kwargs)
        self.env.seed(0)

    @pytest.mark.parametrize('n_agents', [2, 5])
    def test_heuristic(self, n_agents, n_steps=50, n_envs=4):
        self.setup_env(n_agents=n_agents, random_package_pos_on_line=False, n_envs=n_envs)
        policy = balance.HeuristicPolicy(self.continuous_actions)
        obs = self.env.reset()
        prev_package_dist_to_goal = obs[0][:, 8:10]
        for _ in range(n_steps):
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                package_dist_to_goal = obs_agent[:, 8:10]
                action_agent = policy.compute_action(obs_agent, self.env.agents[i].u_range)
                actions.append(action_agent)
            obs, new_rews, dones, _ = self.env.step(actions)
            assert (torch.linalg.vector_norm(package_dist_to_goal, dim=-1) <= torch.linalg.vector_norm(prev_package_dist_to_goal, dim=-1)).all()
            prev_package_dist_to_goal = package_dist_to_goal

class TestWaterfall:

    def setUp(self, n_envs, n_agents) -> None:
        self.continuous_actions = True
        self.env = make_env(scenario='waterfall', num_envs=n_envs, device='cpu', continuous_actions=self.continuous_actions, n_agents=n_agents)
        self.env.seed(0)

    def test_heuristic(self, n_agents=5, n_envs=4, n_steps=50):
        self.setUp(n_envs=n_envs, n_agents=n_agents)
        obs = self.env.reset()
        for _ in range(n_steps):
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                action_agent = torch.clamp(obs_agent[:, -2:], min=-self.env.agents[i].u_range, max=self.env.agents[i].u_range)
                actions.append(action_agent)
            obs, new_rews, _, _ = self.env.step(actions)

class TestGiveWay:

    def setup_env(self, n_envs, **kwargs) -> None:
        self.continuous_actions = True
        self.env = make_env(scenario='give_way', num_envs=n_envs, device='cpu', continuous_actions=self.continuous_actions, **kwargs)
        self.env.seed(0)

    def test_heuristic(self, n_envs=4):
        self.setup_env(mirror_passage=False, n_envs=n_envs)
        all_done = torch.full((n_envs,), False)
        obs = self.env.reset()
        u_range = self.env.agents[0].u_range
        total_rew = torch.zeros((n_envs,))
        while not (total_rew > 17).all():
            obs_agent = obs[0]
            if (obs[1][:, :1] < 0).all():
                action_1 = torch.tensor([u_range / 2, -u_range]).repeat(n_envs, 1)
            else:
                action_1 = torch.tensor([u_range / 2, u_range]).repeat(n_envs, 1)
            action_2 = torch.tensor([-u_range / 3, 0]).repeat(n_envs, 1)
            obs, rews, dones, _ = self.env.step([action_1, action_2])
            for rew in rews:
                total_rew += rew
            if dones.any():
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)

class TestReverseTransport:

    def setup_env(self, n_envs, **kwargs) -> None:
        self.n_agents = kwargs.get('n_agents', 4)
        self.package_width = kwargs.get('package_width', 0.6)
        self.package_length = kwargs.get('package_length', 0.6)
        self.package_mass = kwargs.get('package_mass', 50)
        self.continuous_actions = True
        self.env = make_env(scenario='reverse_transport', num_envs=n_envs, device='cpu', continuous_actions=self.continuous_actions, **kwargs)
        self.env.seed(0)

    @pytest.mark.parametrize('n_agents', [5])
    def test_heuristic(self, n_agents, n_envs=4):
        self.setup_env(n_agents=n_agents, n_envs=n_envs)
        obs = self.env.reset()
        all_done = torch.full((n_envs,), False)
        while not all_done.all():
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                action_agent = torch.clamp(-obs_agent[:, -2:], min=-self.env.agents[i].u_range, max=self.env.agents[i].u_range)
                actions.append(action_agent)
            obs, new_rews, dones, _ = self.env.step(actions)
            if dones.any():
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)

class TestFootball:

    def setup_env(self, n_envs, **kwargs) -> None:
        self.continuous_actions = True
        self.env = make_env(scenario='football', num_envs=n_envs, device='cpu', continuous_actions=True, **kwargs)
        self.env.seed(0)

    @pytest.mark.skipif(sys.platform.startswith('win32'), reason='Test does not work on windows')
    def test_ai_vs_random(self, n_envs=4, n_agents=3, scoring_reward=1):
        self.setup_env(n_red_agents=n_agents, n_blue_agents=n_agents, ai_red_agents=True, ai_blue_agents=False, dense_reward=False, n_envs=n_envs, scoring_reward=scoring_reward)
        all_done = torch.full((n_envs,), False)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs, n_agents)
        with tqdm(total=n_envs) as pbar:
            while not all_done.all():
                pbar.update(all_done.sum().item() - pbar.n)
                actions = []
                for _ in range(n_agents):
                    actions.append(torch.rand(n_envs, 2))
                obs, rews, dones, _ = self.env.step(actions)
                for i in range(n_agents):
                    total_rew[:, i] += rews[i]
                if dones.any():
                    actual_rew = -scoring_reward * n_agents
                    assert torch.equal(total_rew[dones].sum(-1).to(torch.long), torch.full((dones.sum(),), actual_rew, dtype=torch.long))
                    total_rew[dones] = 0
                    all_done += dones
                    for env_index, done in enumerate(dones):
                        if done:
                            self.env.reset_at(env_index)

class TestPassage:

    def setup_env(self, n_envs, **kwargs) -> None:
        self.n_passages = kwargs.get('n_passages', 4)
        self.continuous_actions = True
        self.env = make_env(scenario='passage', num_envs=n_envs, device='cpu', continuous_actions=self.continuous_actions, **kwargs)
        self.env.seed(0)

    def test_heuristic(self, n_envs=4):
        self.setup_env(n_passages=1, shared_reward=True, n_envs=4)
        obs = self.env.reset()
        agent_switched = torch.full((5, n_envs), False)
        all_done = torch.full((n_envs,), False)
        while not all_done.all():
            actions = []
            for i in range(5):
                obs_agent = obs[i]
                dist_to_passage = obs_agent[:, 6:8]
                dist_to_goal = obs_agent[:, 4:6]
                dist_to_passage_is_close = torch.linalg.vector_norm(dist_to_passage, dim=1) <= 0.025
                action_agent = torch.clamp(2 * dist_to_passage, min=-self.env.agents[i].u_range, max=self.env.agents[i].u_range)
                agent_switched[i] += dist_to_passage_is_close
                action_agent[agent_switched[i]] = torch.clamp(2 * dist_to_goal, min=-self.env.agents[i].u_range, max=self.env.agents[i].u_range)[agent_switched[i]]
                actions.append(action_agent)
            obs, new_rews, dones, _ = self.env.step(actions)
            if dones.any():
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        agent_switched[:, env_index] = False
                        self.env.reset_at(env_index)

class TestFlocking:

    def setup_env(self, n_envs, **kwargs) -> None:
        self.env = make_env(scenario='flocking', num_envs=n_envs, device='cpu', **kwargs)
        self.env.seed(0)

    @pytest.mark.parametrize('n_agents', [1, 5])
    def test_heuristic(self, n_agents, n_steps=50, n_envs=4):
        self.setup_env(n_agents=n_agents, n_envs=n_envs)
        policy = flocking.HeuristicPolicy(True)
        obs = self.env.reset()
        for _ in range(n_steps):
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                action_agent = policy.compute_action(obs_agent, self.env.agents[i].u_range)
                actions.append(action_agent)
            obs, new_rews, dones, _ = self.env.step(actions)

@pytest.mark.parametrize('scenario', TEST_SCENARIOS)
@pytest.mark.parametrize('return_numpy', [True, False])
@pytest.mark.parametrize('continuous_actions', [True, False])
@pytest.mark.parametrize('dict_space', [True, False])
def test_gymnasium_wrapper(scenario, return_numpy, continuous_actions, dict_space, max_steps=10):
    env = make_env(scenario=scenario, num_envs=1, device='cpu', continuous_actions=continuous_actions, dict_spaces=dict_space, wrapper='gymnasium', terminated_truncated=True, wrapper_kwargs={'return_numpy': return_numpy}, max_steps=max_steps)
    assert len(env.observation_space) == env.unwrapped.n_agents, 'Expected one observation per agent'
    assert len(env.action_space) == env.unwrapped.n_agents, 'Expected one action per agent'
    if dict_space:
        assert isinstance(env.observation_space, gym.spaces.Dict), 'Expected Dict observation space'
        assert isinstance(env.action_space, gym.spaces.Dict), 'Expected Dict action space'
        obs_shapes = {k: obs_space.shape for k, obs_space in env.observation_space.spaces.items()}
    else:
        assert isinstance(env.observation_space, gym.spaces.Tuple), 'Expected Tuple observation space'
        assert isinstance(env.action_space, gym.spaces.Tuple), 'Expected Tuple action space'
        obs_shapes = [obs_space.shape for obs_space in env.observation_space.spaces]
    assert isinstance(env.unwrapped, Environment), 'The unwrapped attribute of the Gym wrapper should be a VMAS Environment'
    obss, info = env.reset()
    _check_obs_type(obss, obs_shapes, dict_space, return_numpy=return_numpy)
    assert isinstance(info, dict), f'Expected info to be a dictionary but got {type(info)}'
    for _ in range(max_steps):
        actions = [env.unwrapped.get_random_action(agent).numpy() for agent in env.unwrapped.agents]
        obss, rews, terminated, truncated, info = env.step(actions)
        _check_obs_type(obss, obs_shapes, dict_space, return_numpy=return_numpy)
        assert len(rews) == env.unwrapped.n_agents, 'Expected one reward per agent'
        if not dict_space:
            assert isinstance(rews, list), f'Expected list of rewards but got {type(rews)}'
            rew_values = rews
        else:
            assert isinstance(rews, dict), f'Expected dictionary of rewards but got {type(rews)}'
            rew_values = list(rews.values())
        assert all((isinstance(rew, float) for rew in rew_values)), f'Expected float rewards but got {type(rew_values[0])}'
        assert isinstance(terminated, bool), f'Expected bool for terminated but got {type(terminated)}'
        assert isinstance(truncated, bool), f'Expected bool for truncated but got {type(truncated)}'
        assert isinstance(info, dict), f'Expected info to be a dictionary but got {type(info)}'
    assert truncated, 'Expected done to be True after 100 steps'

@pytest.mark.parametrize('scenario', TEST_SCENARIOS)
@pytest.mark.parametrize('return_numpy', [True, False])
@pytest.mark.parametrize('continuous_actions', [True, False])
@pytest.mark.parametrize('dict_space', [True, False])
@pytest.mark.parametrize('num_envs', [1, 10])
def test_gymnasium_wrapper(scenario, return_numpy, continuous_actions, dict_space, num_envs, max_steps=10):
    env = make_env(scenario=scenario, num_envs=num_envs, device='cpu', continuous_actions=continuous_actions, dict_spaces=dict_space, wrapper='gymnasium_vec', terminated_truncated=True, wrapper_kwargs={'return_numpy': return_numpy}, max_steps=max_steps)
    assert isinstance(env.unwrapped, Environment), 'The unwrapped attribute of the Gym wrapper should be a VMAS Environment'
    assert len(env.observation_space) == env.unwrapped.n_agents, 'Expected one observation per agent'
    assert len(env.action_space) == env.unwrapped.n_agents, 'Expected one action per agent'
    if dict_space:
        assert isinstance(env.observation_space, gym.spaces.Dict), 'Expected Dict observation space'
        assert isinstance(env.action_space, gym.spaces.Dict), 'Expected Dict action space'
        obs_shapes = {k: obs_space.shape for k, obs_space in env.observation_space.spaces.items()}
    else:
        assert isinstance(env.observation_space, gym.spaces.Tuple), 'Expected Tuple observation space'
        assert isinstance(env.action_space, gym.spaces.Tuple), 'Expected Tuple action space'
        obs_shapes = [obs_space.shape for obs_space in env.observation_space.spaces]
    obss, info = env.reset()
    _check_obs_type(obss, obs_shapes, dict_space, return_numpy=return_numpy)
    assert isinstance(info, dict), f'Expected info to be a dictionary but got {type(info)}'
    for _ in range(max_steps):
        actions = [env.unwrapped.get_random_action(agent).numpy() for agent in env.unwrapped.agents]
        obss, rews, terminated, truncated, info = env.step(actions)
        _check_obs_type(obss, obs_shapes, dict_space, return_numpy=return_numpy)
        assert len(rews) == env.unwrapped.n_agents, 'Expected one reward per agent'
        if not dict_space:
            assert isinstance(rews, list), f'Expected list of rewards but got {type(rews)}'
            rew_values = rews
        else:
            assert isinstance(rews, dict), f'Expected dictionary of rewards but got {type(rews)}'
            rew_values = list(rews.values())
        if return_numpy:
            assert all((isinstance(rew, np.ndarray) for rew in rew_values)), f'Expected np.array rewards but got {type(rew_values[0])}'
        else:
            assert all((isinstance(rew, torch.Tensor) for rew in rew_values)), f'Expected torch tensor rewards but got {type(rew_values[0])}'
        if return_numpy:
            assert isinstance(terminated, np.ndarray), f'Expected np.array for terminated but got {type(terminated)}'
            assert isinstance(truncated, np.ndarray), f'Expected np.array for truncated but got {type(truncated)}'
        else:
            assert isinstance(terminated, torch.Tensor), f'Expected torch tensor for terminated but got {type(terminated)}'
            assert isinstance(truncated, torch.Tensor), f'Expected torch tensor for truncated but got {type(truncated)}'
        assert isinstance(info, dict), f'Expected info to be a dictionary but got {type(info)}'
    assert all(truncated), 'Expected done to be True after 100 steps'

@pytest.mark.parametrize('scenario', TEST_SCENARIOS)
@pytest.mark.parametrize('return_numpy', [True, False])
@pytest.mark.parametrize('continuous_actions', [True, False])
@pytest.mark.parametrize('dict_space', [True, False])
def test_gym_wrapper(scenario, return_numpy, continuous_actions, dict_space, max_steps=10):
    env = make_env(scenario=scenario, num_envs=1, device='cpu', continuous_actions=continuous_actions, dict_spaces=dict_space, wrapper='gym', wrapper_kwargs={'return_numpy': return_numpy}, max_steps=max_steps)
    assert len(env.observation_space) == env.unwrapped.n_agents, 'Expected one observation per agent'
    assert len(env.action_space) == env.unwrapped.n_agents, 'Expected one action per agent'
    if dict_space:
        assert isinstance(env.observation_space, gym.spaces.Dict), 'Expected Dict observation space'
        assert isinstance(env.action_space, gym.spaces.Dict), 'Expected Dict action space'
        obs_shapes = {k: obs_space.shape for k, obs_space in env.observation_space.spaces.items()}
    else:
        assert isinstance(env.observation_space, gym.spaces.Tuple), 'Expected Tuple observation space'
        assert isinstance(env.action_space, gym.spaces.Tuple), 'Expected Tuple action space'
        obs_shapes = [obs_space.shape for obs_space in env.observation_space.spaces]
    assert isinstance(env.unwrapped, Environment), 'The unwrapped attribute of the Gym wrapper should be a VMAS Environment'
    obss = env.reset()
    _check_obs_type(obss, obs_shapes, dict_space, return_numpy=return_numpy)
    for _ in range(max_steps):
        actions = [env.unwrapped.get_random_action(agent).numpy() for agent in env.unwrapped.agents]
        obss, rews, done, info = env.step(actions)
        _check_obs_type(obss, obs_shapes, dict_space, return_numpy=return_numpy)
        assert len(rews) == env.unwrapped.n_agents, 'Expected one reward per agent'
        if not dict_space:
            assert isinstance(rews, list), f'Expected list of rewards but got {type(rews)}'
            rew_values = rews
        else:
            assert isinstance(rews, dict), f'Expected dictionary of rewards but got {type(rews)}'
            rew_values = list(rews.values())
        assert all((isinstance(rew, float) for rew in rew_values)), f'Expected float rewards but got {type(rew_values[0])}'
        assert isinstance(done, bool), f'Expected bool for done but got {type(done)}'
        assert isinstance(info, dict), f'Expected info to be a dictionary but got {type(info)}'
    assert done, 'Expected done to be True after 100 steps'

def _check_obs_type(obss, obs_shapes, dict_space, return_numpy):
    if dict_space:
        assert isinstance(obss, dict), f'Expected dictionary of observations, got {type(obss)}'
        for k, obs in obss.items():
            obs_shape = obs_shapes[k]
            assert obs.shape == obs_shape, f'Expected shape {obs_shape}, got {obs.shape}'
            if return_numpy:
                assert isinstance(obs, np.ndarray), f'Expected numpy array, got {type(obs)}'
            else:
                assert isinstance(obs, Tensor), f'Expected torch tensor, got {type(obs)}'
    else:
        assert isinstance(obss, list), f'Expected list of observations, got {type(obss)}'
        for obs, shape in zip(obss, obs_shapes):
            assert obs.shape == shape, f'Expected shape {shape}, got {obs.shape}'
            if return_numpy:
                assert isinstance(obs, np.ndarray), f'Expected numpy array, got {type(obs)}'
            else:
                assert isinstance(obs, Tensor), f'Expected torch tensor, got {type(obs)}'

def use_vmas_env(render: bool, num_envs: int, n_steps: int, device: str, scenario: Union[str, BaseScenario], continuous_actions: bool, random_action: bool, **kwargs):
    """Example function to use a vmas environment.

    This is a simplification of the function in `vmas.examples.use_vmas_env.py`.

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action

    """
    scenario_name = scenario if isinstance(scenario, str) else scenario.__class__.__name__
    env = make_env(scenario=scenario, num_envs=num_envs, device=device, continuous_actions=continuous_actions, seed=0, **kwargs)
    frame_list = []
    init_time = time.time()
    step = 0
    for s in range(n_steps):
        step += 1
        print(f'Step {step}')
        actions = []
        for i, agent in enumerate(env.agents):
            if not random_action:
                action = _get_deterministic_action(agent, continuous_actions, env)
            else:
                action = env.get_random_action(agent)
            actions.append(action)
        obs, rews, dones, info = env.step(actions)
        if render:
            frame = env.render(mode='rgb_array', agent_index_focus=None)
            frame_list.append(frame)
    total_time = time.time() - init_time
    print(f'It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} for {scenario_name} scenario.')
    if render:
        from moviepy.editor import ImageSequenceClip
        fps = 30
        clip = ImageSequenceClip(frame_list, fps=fps)
        clip.write_gif(f'{scenario_name}.gif', fps=fps)

def env_creator(config: Dict):
    env = make_env(scenario=config['scenario_name'], num_envs=config['num_envs'], device=config['device'], continuous_actions=config['continuous_actions'], wrapper=Wrapper.RLLIB, max_steps=config['max_steps'], n_agents=config['n_agents'])
    return env

