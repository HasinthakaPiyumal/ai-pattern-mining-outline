# Cluster 22

def full_nvec(agent, world):
    return list(agent.discrete_action_nvec) + ([world.dim_c] if not agent.silent and world.dim_c != 0 else [])

@pytest.mark.parametrize('nvecs', list(zip(random_nvecs(10, seed=0), random_nvecs(10, seed=42))))
def test_discrete_action_nvec_discrete_to_multi(nvecs, scenario='transport', num_envs=10, n_steps=5):
    kwargs = {'scenario': scenario, 'num_envs': num_envs, 'seed': 0, 'continuous_actions': False}
    env = make_env(**kwargs, multidiscrete_actions=False)
    env_multi = make_env(**kwargs, multidiscrete_actions=True)
    if type(env.scenario).process_action is not vmas.simulator.scenario.BaseScenario.process_action:
        pytest.skip('Scenario uses a custom process_action method.')

    def set_nvec(agent, nvec):
        agent.action_size = len(nvec)
        agent.discrete_action_nvec = nvec
        agent.action.action_size = agent.action_size
    random.seed(0)
    for agent, agent_multi, nvec in zip(env.world.policy_agents, env_multi.world.policy_agents, nvecs):
        set_nvec(agent, nvec)
        set_nvec(agent_multi, nvec)
    env.action_space = env.get_action_space()
    env_multi.action_space = env.get_action_space()

    def full_nvec(agent, world):
        return list(agent.discrete_action_nvec) + ([world.dim_c] if not agent.silent and world.dim_c != 0 else [])

    def full_action_size(agent, world):
        return len(full_nvec(agent, world))
    for _ in range(n_steps):
        actions_multi = env_multi.get_random_actions()
        prodss = [[math.prod(full_nvec(agent, env.world)[i + 1:]) for i in range(full_action_size(agent, env.world))] for agent in env.world.policy_agents]
        actions = [(a_multi * torch.tensor(prods)).sum(dim=1) for a_multi, prods in zip(actions_multi, prodss)]
        env_multi.step(actions_multi)
        env.step(actions)
        for agent, agent_multi, action, action_multi in zip(env.world.policy_agents, env_multi.world.policy_agents, actions, actions_multi):
            U = agent.action.u_range_tensor
            k = agent.action.u_multiplier_tensor
            for u, u_multi, a, a_multi in zip(agent.action.u, agent_multi.action.u, action, action_multi):
                assert torch.allclose(u, u_multi), f'{u} != {u_multi} (nvec={agent.discrete_action_nvec}, a={a}, a_multi={a_multi}, U={U}, k={k})'

def set_nvec(agent, nvec):
    agent.action_size = len(nvec)
    agent.discrete_action_nvec = nvec
    agent.action.action_size = agent.action_size

def full_action_size(agent, world):
    return len(full_nvec(agent, world))

def random_nvecs(count, l_min=2, l_max=6, n_min=2, n_max=6, seed=0):
    random.seed(seed)
    return [[random.randint(n_min, n_max) for _ in range(random.randint(l_min, l_max))] for _ in range(count)]

