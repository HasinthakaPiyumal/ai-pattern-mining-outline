# Cluster 159

def calculate_relative_progress_to_goal(ego_states: List[EgoState], expert_states: List[EgoState], goal: StateSE2, tolerance: float=0.1) -> float:
    """
    Ratio of ego's to the expert's progress towards goal rounded up
    :param ego_states: A list of ego states
    :param expert_states: A list of expert states
    :param goal: goal
    :param tolerance: tolerance used for round up
    :return Ratio of progress towards goal.
    """
    ego_progress_value = calculate_ego_progress_to_goal(ego_states, goal)
    expert_progress_value = calculate_ego_progress_to_goal(expert_states, goal)
    relative_progress: float = max(tolerance, ego_progress_value) / max(tolerance, expert_progress_value)
    return relative_progress

def calculate_ego_progress_to_goal(ego_states: List[EgoState], goal: StateSE2) -> Any:
    """
    Progress (m) towards goal using euclidean distance assuming the goal
    does not change along the trajectory (suitable for open loop only)
    A positive number means progress to goal
    :param ego_states: A list of ego states
    :param goal: goal
    :return Progress towards goal.
    """
    if len(ego_states) > 1:
        start_distance = ego_states[0].center.distance_to(goal)
        end_distance = ego_states[-1].center.distance_to(goal)
        return start_distance - end_distance
    elif len(ego_states) == 1:
        return 0.0
    else:
        return np.nan

