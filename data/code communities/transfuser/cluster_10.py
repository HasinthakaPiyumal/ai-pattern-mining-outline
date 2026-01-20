# Cluster 10

def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
    :param route:
    :param lat_ref:
    :param lon_ref:
    :return:
    """
    gps_route = []
    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))
    return gps_route

def _location_to_gps(lat_ref, lon_ref, location):
    """
    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """
    EARTH_RADIUS_EQUA = 6378137.0
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y
    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z
    return {'lat': lat, 'lon': lon, 'z': z}

def _get_latlon_ref(world):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))
    lat_ref = 42.0
    lon_ref = 2.0
    for opendrive in tree.iter('OpenDRIVE'):
        for header in opendrive.iter('header'):
            for georef in header.iter('geoReference'):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return (lat_ref, lon_ref)

def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
    :param route:
    :param lat_ref:
    :param lon_ref:
    :return:
    """
    gps_route = []
    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))
    return gps_route

def interpolate_trajectory(world, waypoints_trajectory, hop_resolution=1.0):
    """
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    :param world: an reference to the CARLA world so we can use the planner
    :param waypoints_trajectory: the current coarse trajectory
    :param hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    :return: the full interpolated route both in GPS coordinates and also in its original form.
    """
    dao = GlobalRoutePlannerDAO(world.get_map(), hop_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    route = []
    for i in range(len(waypoints_trajectory) - 1):
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp_tuple in interpolated_trace:
            route.append((wp_tuple[0].transform, wp_tuple[1]))
    lat_ref, lon_ref = _get_latlon_ref(world)
    return (location_route_to_gps(route, lat_ref, lon_ref), route)

def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0, max_len=100):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    returns the full interpolated route both in GPS coordinates and also in its original form.
    
    Args:
        - world: an reference to the CARLA world so we can use the planner
        - waypoints_trajectory: the current coarse trajectory
        - hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    """
    dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    route = []
    for i in range(len(waypoints_trajectory) - 1):
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        if waypoint.x != waypoint_next.x or waypoint.y != waypoint_next.y:
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            if len(interpolated_trace) > max_len:
                waypoints_trajectory[i + 1] = waypoints_trajectory[i]
            else:
                for wp_tuple in interpolated_trace:
                    route.append((wp_tuple[0].transform, wp_tuple[1]))
    lat_ref, lon_ref = _get_latlon_ref(world_map)
    return (location_route_to_gps(route, lat_ref, lon_ref), route)

def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
    :param route:
    :param lat_ref:
    :param lon_ref:
    :return:
    """
    gps_route = []
    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))
    return gps_route

