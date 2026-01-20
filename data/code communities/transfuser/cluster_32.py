# Cluster 32

def get_scenario_type(scenario, match_position, trajectory):
    """
    Some scenarios have different types depending on the route.
    :param scenario: the scenario name
    :param match_position: the matching position for the scenarion
    :param trajectory: the route trajectory the ego is following
    :return: tag representing this subtype

    Also used to check which are not viable (Such as an scenario
    that triggers when turning but the route doesnt')
    WARNING: These tags are used at:
        - VehicleTurningRoute
        - SignalJunctionCrossingRoute
    and changes to these tags will affect them
    """

    def check_this_waypoint(tuple_wp_turn):
        """
        Decides whether or not the waypoint will define the scenario behavior
        """
        if RoadOption.LANEFOLLOW == tuple_wp_turn[1]:
            return False
        elif RoadOption.CHANGELANELEFT == tuple_wp_turn[1]:
            return False
        elif RoadOption.CHANGELANERIGHT == tuple_wp_turn[1]:
            return False
        return True
    subtype = 'valid'
    if scenario == 'Scenario4':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.LEFT == tuple_wp_turn[1]:
                    subtype = 'S4left'
                elif RoadOption.RIGHT == tuple_wp_turn[1]:
                    subtype = 'S4right'
                else:
                    subtype = None
                break
            subtype = None
    if scenario == 'Scenario7':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.LEFT == tuple_wp_turn[1]:
                    subtype = 'S7left'
                elif RoadOption.RIGHT == tuple_wp_turn[1]:
                    subtype = 'S7right'
                elif RoadOption.STRAIGHT == tuple_wp_turn[1]:
                    subtype = 'S7opposite'
                else:
                    subtype = None
                break
            subtype = None
    if scenario == 'Scenario8':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.LEFT == tuple_wp_turn[1]:
                    subtype = 'S8left'
                else:
                    subtype = None
                break
            subtype = None
    if scenario == 'Scenario9':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.RIGHT == tuple_wp_turn[1]:
                    subtype = 'S9right'
                else:
                    subtype = None
                break
            subtype = None
    return subtype

def check_this_waypoint(tuple_wp_turn):
    """
        Decides whether or not the waypoint will define the scenario behavior
        """
    if RoadOption.LANEFOLLOW == tuple_wp_turn[1]:
        return False
    elif RoadOption.CHANGELANELEFT == tuple_wp_turn[1]:
        return False
    elif RoadOption.CHANGELANERIGHT == tuple_wp_turn[1]:
        return False
    return True

