# Cluster 114

def agents_pose_to_scene(scene: Dict[str, Any], agents: UniqueIDMAgents) -> None:
    """
    Renders all agents' pose as markers
    :param scene: scene dictionary
    :param agents: all agents for which the pose should be rendered for
    """
    for agent_id, agent in agents.items():
        marker_to_scene(scene, agent_id, agent.to_se2())

def marker_to_scene(scene: Dict[str, Any], marker_id: str, pose: StateSE2) -> None:
    """
    Renders a pose as an arrow marker
    :param scene: scene dictionary
    :param marker_id: marker id as a string
    :param pose: the pose that defines the markers location
    """
    if 'markers' not in scene.keys():
        scene['markers'] = []
    scene['markers'].append({'id': 0, 'name': marker_id, 'pose': pose.serialize(), 'shape': 'arrow'})

