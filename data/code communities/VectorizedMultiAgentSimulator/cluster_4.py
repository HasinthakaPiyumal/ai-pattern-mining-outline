# Cluster 4

def render_interactively(scenario: Union[str, BaseScenario], control_two_agents: bool=False, display_info: bool=True, save_render: bool=False, **kwargs):
    """Executes a scenario and renders it so that you can debug and control agents interactively.

    You can change the agent to control by pressing TAB.
    You can reset the environment by pressing R.
    You can control agent actions with the arrow keys and M/N (left/right control the first action, up/down control the second, M/N controls the third)

    If you have more than 1 agent, you can control another one with W,A,S,D and Q,E in the same way.
    and switch the agent using LSHIFT.

    Args:
        scenario (Union[str, BaseScenario]): Scenario to load.
            Can be the name of a file in `vmas.scenarios` folder or a :class:`~vmas.simulator.scenario.BaseScenario` class
        control_two_agents (bool, optional): Whether to control two agents or just one. Defaults to ``False``.
        display_info (bool, optional): Whether to display on the screen the following info from the first controlled agent:
            name, reward, total reward, done, and observation. Defaults to ``True``.
        save_render (bool, optional): Whether to save a video of the render up to the first reset.
            The video will be saved in the directory of this file with the name ``{scenario}_interactive``.
            Defaults to ``False``.

    Examples:
        >>> from vmas import render_interactively
        >>> render_interactively(
        ...     "waterfall",
        ...     control_two_agents=True,
        ...     save_render=False,
        ...     display_info=True,
        ... )

    """
    InteractiveEnv(make_env(scenario=scenario, num_envs=1, device='cpu', continuous_actions=True, wrapper='gym', seed=0, wrapper_kwargs={'return_numpy': False}, **kwargs), control_two_agents=control_two_agents, display_info=display_info, save_render=save_render, render_name=f'{scenario}_interactive' if isinstance(scenario, str) else 'interactive')

def parse_args():
    parser = ArgumentParser(description='Interactive rendering')
    parser.add_argument('--scenario', type=str, default='waterfall', help='Scenario to load. Can be the name of a file in `vmas.scenarios` folder or a :class:`~vmas.simulator.scenario.BaseScenario` class')
    parser.add_argument('--control_two_agents', action=BooleanOptionalAction, default=True, help='Whether to control two agents or just one')
    parser.add_argument('--display_info', action=BooleanOptionalAction, default=True, help='Whether to display on the screen the following info from the first controlled agent: name, reward, total reward, done, and observation')
    parser.add_argument('--save_render', action='store_true', help='Whether to save a video of the render up to the first reset')
    return parser.parse_args()

