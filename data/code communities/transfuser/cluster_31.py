# Cluster 31

def match_world_location_to_route(world_location, route_description):
    """
    We match this location to a given route.
        world_location:
        route_description:
    """

    def match_waypoints(waypoint1, wtransform):
        """
        Check if waypoint1 and wtransform are similar
        """
        dx = float(waypoint1['x']) - wtransform.location.x
        dy = float(waypoint1['y']) - wtransform.location.y
        dz = float(waypoint1['z']) - wtransform.location.z
        dpos = math.sqrt(dx * dx + dy * dy + dz * dz)
        dyaw = (float(waypoint1['yaw']) - wtransform.rotation.yaw) % 360
        return dpos < TRIGGER_THRESHOLD and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > 360 - TRIGGER_ANGLE_THRESHOLD)
    match_position = 0
    for route_waypoint in route_description:
        if match_waypoints(world_location, route_waypoint[0]):
            return match_position
        match_position += 1
    return None

def match_waypoints(waypoint1, wtransform):
    """
        Check if waypoint1 and wtransform are similar
        """
    dx = float(waypoint1['x']) - wtransform.location.x
    dy = float(waypoint1['y']) - wtransform.location.y
    dz = float(waypoint1['z']) - wtransform.location.z
    dpos = math.sqrt(dx * dx + dy * dy + dz * dz)
    dyaw = (float(waypoint1['yaw']) - wtransform.rotation.yaw) % 360
    return dpos < TRIGGER_THRESHOLD and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > 360 - TRIGGER_ANGLE_THRESHOLD)

