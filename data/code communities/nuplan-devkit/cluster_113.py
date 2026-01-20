# Cluster 113

def agent_baselines_to_scene(scene: Dict[str, Any], agents: UniqueIDMAgents) -> None:
    """
    Renders all agents' reference baseline paths as series of poses
    :param scene: scene dictionary
    :param agents: all agents for which the baseline path should be rendered for
    """
    for agent in agents.values():
        baseline_path_to_scene(scene, agent.get_path_to_go())

def baseline_path_to_scene(scene: Dict[str, Any], baseline_path: List[StateSE2]) -> None:
    """
    Renders an agent's reference baseline path as a series of poses
    :param scene: scene dictionary
    :param baseline_path: baseline path represented by a list fo StateSE2
    """
    if 'path_info' not in scene.keys():
        scene['path_info'] = []
    scene['path_info'] += [[pose.x, pose.y, pose.heading] for pose in baseline_path]

