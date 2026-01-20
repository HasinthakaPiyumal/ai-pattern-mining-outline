# Cluster 30

def interpolate_trajectory(world, waypoints_trajectory, hop_resolution=1.0):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    returns the full interpolated route both in GPS coordinates and also in its original form.
    
    Args:
        - world: an reference to the CARLA world so we can use the planner
        - waypoints_trajectory: the current coarse trajectory
        - hop_resolution: is the resolution, how dense is the provided trajectory going to be made
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

def sample_junctions(world_map, route, scenarios_list, town, start_dist=20, end_dist=20, min_len=50, max_len=MAX_LEN):
    """
    Sample individual junctions from the interpolated routes
    Args:
        world_map: town map
        route: interpolated route
    Return:
        custom_routes: list of (start wp, end wp) each representing an individual junction
    """
    custom_routes = []
    start_id = -1
    end_id = -1
    for index in range(start_dist, len(route) - end_dist):
        if route[index - 1][1] == RoadOption.LANEFOLLOW and route[index][1] != RoadOption.LANEFOLLOW:
            start_id = index - start_dist
        elif start_id != -1 and route[index][1] == RoadOption.LANEFOLLOW:
            end_id = index + end_dist
            if end_id > start_id + min_len:
                start_wp = carla.Location(x=route[start_id][0].location.x, y=route[start_id][0].location.y, z=route[start_id][0].location.z)
                end_wp = carla.Location(x=route[end_id][0].location.x, y=route[end_id][0].location.y, z=route[end_id][0].location.z)
                waypoint_list = [start_wp, end_wp]
                extended_route = interpolate_trajectory(world_map, waypoint_list)
                potential_scenarios_definitions, _ = scan_route_for_scenarios(town, extended_route, scenarios_list)
                if len(extended_route) > max_len or len(extended_route) == 1 or len(potential_scenarios_definitions) == 0:
                    start_id = -1
                    end_id = -1
                    continue
                downsampled_route = downsample_route(extended_route, 50)
                custom_route = []
                for element in downsampled_route:
                    custom_transform = (route[start_id + element][0].location.x, route[start_id + element][0].location.y, route[start_id + element][0].location.z, route[start_id + element][0].rotation.yaw)
                    custom_route.append(custom_transform)
                custom_routes.append(custom_route)
            start_id = -1
            end_id = -1
    return custom_routes

def process_route(world_map, route, scenarios_list, return_dict):
    interpolated_route = interpolate_trajectory(world_map, route['trajectory'])
    wp_list = sample_junctions(world_map, interpolated_route, scenarios_list, route['town_name'])
    print('got {} junctions in route {} (interpolated {} waypoints to {} waypoints)'.format(len(wp_list), route['id'], len(route['trajectory']), len(interpolated_route)))
    return_dict[route['id']] = {'wp_list': wp_list, 'town_name': route['town_name'], 'length': len(interpolated_route)}

def main(args):
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    routes_list = parse_routes_file(args.routes_file)
    scenarios_list = parse_annotations_file(args.scenarios_file)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    st = time.time()
    for index, route in enumerate(routes_list):
        if index == 0 or routes_list[index]['town_name'] != routes_list[index - 1]['town_name']:
            world = client.load_world(route['town_name'])
            world_map = world.get_map()
        p = multiprocessing.Process(target=process_route, args=(world_map, route, scenarios_list, return_dict))
        jobs.append(p)
        p.start()
    for process in jobs:
        process.join()
    print('{} routes processed in {} seconds'.format(len(return_dict), time.time() - st))
    route_id = 0
    total_junctions = 0
    route_lengths = []
    root = ET.Element('routes')
    for curr_route in return_dict.keys():
        wp_list = return_dict[curr_route]['wp_list']
        town_name = return_dict[curr_route]['town_name']
        total_junctions += len(wp_list)
        route_lengths.append(return_dict[curr_route]['length'])
        for wps in wp_list:
            add_route = ET.SubElement(root, 'route', id='%d' % route_id, town=town_name)
            for node in wps:
                ET.SubElement(add_route, 'waypoint', x='%f' % node[0], y='%f' % node[1], z='%f' % node[2], pitch='0.0', roll='0.0', yaw='%f' % node[3])
            route_id += 1
    print('\nSource File:')
    print('mean distance: ', np.array(route_lengths).mean())
    print('median distance: ', np.median(np.array(route_lengths)))
    if args.save_file is not None:
        tree = ET.ElementTree(root)
        tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)
        new_index = 0
        outliers = 0
        route_lengths = []
        duplicate_list = []
        new_routes_list = parse_routes_file(args.save_file)
        if args.duplicate_file:
            duplicate_file_list = parse_routes_file(args.duplicate_file)
            for index, route in enumerate(duplicate_file_list):
                if index == 0 or duplicate_file_list[index]['town_name'] != duplicate_file_list[index - 1]['town_name']:
                    world = client.load_world(route['town_name'])
                    world_map = world.get_map()
                new_interpolated_route = interpolate_trajectory(world_map, route['trajectory'])
                locations = (new_interpolated_route[0][0].location.x, new_interpolated_route[0][0].location.y, new_interpolated_route[-1][0].location.x, new_interpolated_route[-1][0].location.y)
                duplicate_list.append(locations)
        for index, route in enumerate(new_routes_list):
            if index == 0 or new_routes_list[index]['town_name'] != new_routes_list[index - 1]['town_name']:
                world = client.load_world(route['town_name'])
                world_map = world.get_map()
            new_interpolated_route = interpolate_trajectory(world_map, route['trajectory'])
            current_node = root.getchildren()[index - outliers]
            locations = (new_interpolated_route[0][0].location.x, new_interpolated_route[0][0].location.y, new_interpolated_route[-1][0].location.x, new_interpolated_route[-1][0].location.y)
            if len(new_interpolated_route) > MAX_LEN or locations in duplicate_list:
                root.remove(current_node)
                outliers += 1
            else:
                duplicate_list.append(locations)
                route_lengths.append(len(new_interpolated_route))
                current_node.set('id', '%d' % (ID_START + new_index))
                new_index += 1
        tree = ET.ElementTree(root)
        tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)
        new_routes_list = parse_routes_file(args.save_file)
        print('\nTarget File:')
        print('saved junctions: ', len(route_lengths))
        print('outliers/duplicates: ', outliers)
        print('file num junctions: ', len(new_routes_list))
        print('mean distance: ', np.array(route_lengths).mean())
        print('median distance: ', np.median(np.array(route_lengths)))

def generate_scenario_7_8_9(carla_map, world=None):
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    other_actors = []
    count_all_routes = 0
    trigger_points = {'Scenario7': [], 'Scenario8': [], 'Scenario9': []}
    actors = world.get_actors()
    traffic_lights_list = actors.filter('*traffic_light')
    print('Got %d traffic lights' % len(traffic_lights_list))
    junction_bbs_centers = []
    junctions_ = []
    duplicate_list = []
    count_all_routes = 0
    for wp_it, waypoint in enumerate(topology):
        if waypoint.is_junction:
            junc_ = waypoint.get_junction()
            jbb_ = junc_.bounding_box
            jbb_center = [round(jbb_.location.x, 2), round(jbb_.location.y, 2)]
            if jbb_center not in junction_bbs_centers:
                junction_bbs_centers.append(jbb_center)
                junctions_.append(junc_)
    pole_ind = []
    grps_ = []
    for tl_ in traffic_lights_list:
        grp_tl = tl_.get_group_traffic_lights()
        grp_tl_locs = [(round(gtl_.get_transform().location.x, 2), round(gtl_.get_transform().location.y, 2)) for gtl_ in grp_tl]
        pole_ind.append(tl_.get_pole_index())
        if grp_tl_locs not in grps_:
            grps_.append(grp_tl_locs)
    bb_flags = []
    for grp_ in grps_:
        midpt_grp = [sum(i) / len(grp_) for i in zip(*grp_)]
        grp_bb_dist = [np.sqrt((bb_c[0] - midpt_grp[0]) ** 2 + (bb_c[1] - midpt_grp[1]) ** 2) for bb_c in junction_bbs_centers]
        bb_dist, bb_idx = min(((val, idx) for idx, val in enumerate(grp_bb_dist)))
        bb_flags.append(bb_idx)
    signalized_junctions = [junc_ for i, junc_ in enumerate(junctions_) if i in bb_flags]
    signalized_junctions_bbs = [(junc_.bounding_box.location.x, junc_.bounding_box.location.y) for i, junc_ in enumerate(junctions_) if i in bb_flags]
    if len(signalized_junctions):
        for wp_it, waypoint in enumerate(topology):
            if waypoint.is_junction:
                junc_ = waypoint.get_junction()
                jbb_ = (junc_.bounding_box.location.x, junc_.bounding_box.location.y)
                if jbb_ in signalized_junctions_bbs:
                    j_wps = junc_.get_waypoints(carla.LaneType.Driving)
                    for it_, j_wp in enumerate(j_wps):
                        wp_p = j_wp[0]
                        dist_prev = 0
                        wp_list_prev = []
                        while True:
                            wp_list_prev.append(wp_p)
                            try:
                                wp_p = wp_p.previous(PRECISION)[0]
                            except:
                                break
                            dist_prev += PRECISION
                            if dist_prev > SAMPLING_DISTANCE:
                                break
                        dist_nxt = 0
                        wp_n = j_wp[1]
                        wp_list_nxt = []
                        while True:
                            wp_list_nxt.append(wp_n)
                            try:
                                wp_n = wp_n.next(PRECISION)[0]
                            except:
                                break
                            dist_nxt += PRECISION
                            if dist_nxt > SAMPLING_DISTANCE:
                                break
                        final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
                        cur_wp = final_wps_list[int(len(final_wps_list) / 2)]
                        prev_wp = final_wps_list[0]
                        nxt_wp = final_wps_list[-1]
                        vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                        vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location
                        dot_ = vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y
                        det_ = vec_wp_nxt.x * vec_wp_prev.y - vec_wp_prev.x * vec_wp_nxt.y
                        angle_bet = math.atan2(det_, dot_)
                        if angle_bet < 0:
                            angle_bet += 2 * math.pi
                        angle_deg = angle_bet * 180 / math.pi
                        if 160 < angle_deg < 195:
                            trig_key = 'Scenario7'
                        elif 10 < angle_deg < 160:
                            trig_key = 'Scenario9'
                        elif 195 < angle_deg < 350:
                            trig_key = 'Scenario8'
                        truncated_wp_lst = [final_wps_list]
                        locations = []
                        for wps_sub in truncated_wp_lst:
                            locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))
                        are_loc_dups = []
                        for location_ in locations:
                            flag_cum_ctr = []
                            for loc_dp in duplicate_list:
                                flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc + PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                                flag_AND_ctr = all(flag_ctrs)
                                flag_cum_ctr.append(flag_AND_ctr)
                            is_loc_dup = any(flag_cum_ctr)
                            are_loc_dups.append(is_loc_dup)
                        for j_, wps_ in enumerate(truncated_wp_lst):
                            if not are_loc_dups[j_]:
                                count_all_routes += 1
                                duplicate_list.append(locations[j_])
                                wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                                extended_route = interpolate_trajectory(carla_map, wps_tmp)
                                if len(extended_route) < PRUNE_ROUTES_MIN_LEN:
                                    continue
                                else:
                                    trigger_point = extended_route[2][0]
                                    trigger_points[trig_key] += [trigger_point]
    other_actors = {'Scenario7': [None] * len(trigger_points['Scenario7']), 'Scenario8': [None] * len(trigger_points['Scenario8']), 'Scenario9': [None] * len(trigger_points['Scenario9'])}
    return (trigger_points, other_actors)

def generate_scenario_4(carla_map):
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    other_actors = []
    count_all_routes = 0
    trigger_points = []
    duplicate_list = []
    for wp_it, waypoint in enumerate(topology):
        if waypoint.is_junction:
            junc_ = waypoint.get_junction()
            j_wps = junc_.get_waypoints(carla.LaneType.Driving)
            for it_, j_wp in enumerate(j_wps):
                count_all_routes += 1
                wp_p = j_wp[0]
                dist_prev = 0
                wp_list_prev = []
                while True:
                    wp_list_prev.append(wp_p)
                    try:
                        wp_p = wp_p.previous(PRECISION)[0]
                    except:
                        break
                    dist_prev += PRECISION
                    if dist_prev > SAMPLING_DISTANCE:
                        break
                dist_nxt = 0
                wp_n = j_wp[1]
                wp_list_nxt = []
                while True:
                    wp_list_nxt.append(wp_n)
                    try:
                        wp_n = wp_n.next(PRECISION)[0]
                    except:
                        break
                    dist_nxt += PRECISION
                    if dist_nxt > SAMPLING_DISTANCE:
                        break
                final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
                truncated_wp_lst = [final_wps_list]
                locations = []
                for wps_sub in truncated_wp_lst:
                    locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))
                are_loc_dups = []
                for location_ in locations:
                    flag_cum_ctr = []
                    for loc_dp in duplicate_list:
                        flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc + PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                        flag_AND_ctr = all(flag_ctrs)
                        flag_cum_ctr.append(flag_AND_ctr)
                    is_loc_dup = any(flag_cum_ctr)
                    are_loc_dups.append(is_loc_dup)
                for j_, wps_ in enumerate(truncated_wp_lst):
                    if not are_loc_dups[j_]:
                        duplicate_list.append(locations[j_])
                        wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                        extended_route = interpolate_trajectory(carla_map, wps_tmp)
                        if len(extended_route) < PRUNE_ROUTES_MIN_LEN:
                            continue
                        else:
                            trigger_point = extended_route[5][0]
                            trigger_points += [trigger_point]
    other_actors = [{}] * len(trigger_points)
    return (trigger_points, other_actors)

def generate_scenario_3(carla_map):
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    other_actors = []
    count_all_routes = 0
    trigger_points = []
    duplicate_list = []
    for wp_it, waypoint in enumerate(topology):
        cur_wp = waypoint
        wp_list_nxt = [cur_wp]
        if not cur_wp.is_junction:
            while True:
                cur_wp_ = wp_list_nxt[-1]
                try:
                    nxt_wp = cur_wp_.next(PRECISION)[0]
                except:
                    break
                if not nxt_wp.is_junction:
                    wp_list_nxt.append(nxt_wp)
                else:
                    break
        wp_list_prev = [cur_wp]
        if not cur_wp.is_junction:
            while True:
                cur_wp_ = wp_list_prev[-1]
                try:
                    nxt_wp = cur_wp_.previous(PRECISION)[0]
                except:
                    break
                if not nxt_wp.is_junction:
                    wp_list_prev.append(nxt_wp)
                else:
                    break
        if len(wp_list_prev) + len(wp_list_nxt) > MIN_ROUTE_LENGTH:
            final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
            cur_wp = final_wps_list[int(len(final_wps_list) / 2)]
            prev_wp = final_wps_list[0]
            nxt_wp = final_wps_list[-1]
            vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
            vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location
            norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x + vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(vec_wp_prev.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_prev.y)
            try:
                dot_ = (vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
            except:
                dot_ = -1
            if dot_ > -1 - DOT_PROD_SLACK and dot_ < -1 + DOT_PROD_SLACK:
                continue
            else:
                truncated_wp_lst = []
                count_all_routes += 1
                for i_ in range(len(final_wps_list)):
                    tmp_wps = final_wps_list[i_ * NUM_WAYPOINTS_DISTANCE:i_ * NUM_WAYPOINTS_DISTANCE + NUM_WAYPOINTS_DISTANCE]
                    if len(tmp_wps) > 1:
                        cur_wp = tmp_wps[int(len(tmp_wps) / 2)]
                        prev_wp = tmp_wps[0]
                        nxt_wp = tmp_wps[-1]
                        vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                        vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location
                        norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x + vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(vec_wp_prev.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_prev.y)
                        try:
                            dot_ = (vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
                        except:
                            dot_ = -1
                        if not dot_ < -1 + DOT_PROD_SLACK:
                            truncated_wp_lst.append(tmp_wps)
                    locations = []
                    for wps_sub in truncated_wp_lst:
                        locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))
                    are_loc_dups = []
                    for location_ in locations:
                        flag_cum_ctr = []
                        for loc_dp in duplicate_list:
                            flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc + PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                            flag_AND_ctr = all(flag_ctrs)
                            flag_cum_ctr.append(flag_AND_ctr)
                        is_loc_dup = any(flag_cum_ctr)
                        are_loc_dups.append(is_loc_dup)
                    for j_, wps_ in enumerate(truncated_wp_lst):
                        if not are_loc_dups[j_]:
                            duplicate_list.append(locations[j_])
                            wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                            extended_route = interpolate_trajectory(carla_map, wps_tmp)
                            if len(extended_route) < PRUNE_ROUTES_MIN_LEN:
                                continue
                            else:
                                trigger_point = extended_route[int(len(extended_route) / 2)][0]
                                trigger_points += [trigger_point]
    other_actors = [None] * len(trigger_points)
    return (trigger_points, other_actors)

def generate_scenario_1(carla_map):
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    other_actors = []
    count_all_routes = 0
    trigger_points = []
    duplicate_list = []
    for wp_it, waypoint in enumerate(topology):
        cur_wp = waypoint
        wp_list_nxt = [cur_wp]
        if not cur_wp.is_junction:
            while True:
                cur_wp_ = wp_list_nxt[-1]
                try:
                    nxt_wp = cur_wp_.next(PRECISION)[0]
                except:
                    break
                if not nxt_wp.is_junction:
                    wp_list_nxt.append(nxt_wp)
                else:
                    break
        wp_list_prev = [cur_wp]
        if not cur_wp.is_junction:
            while True:
                cur_wp_ = wp_list_prev[-1]
                try:
                    nxt_wp = cur_wp_.previous(PRECISION)[0]
                except:
                    break
                if not nxt_wp.is_junction:
                    wp_list_prev.append(nxt_wp)
                else:
                    break
        if len(wp_list_prev) + len(wp_list_nxt) > MIN_ROUTE_LENGTH:
            final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
            cur_wp = final_wps_list[int(len(final_wps_list) / 2)]
            prev_wp = final_wps_list[0]
            nxt_wp = final_wps_list[-1]
            vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
            vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location
            norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x + vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(vec_wp_prev.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_prev.y)
            try:
                dot_ = (vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
            except:
                dot_ = -1
            if dot_ > -1 - DOT_PROD_SLACK and dot_ < -1 + DOT_PROD_SLACK:
                continue
            else:
                truncated_wp_lst = []
                count_all_routes += 1
                for i_ in range(len(final_wps_list)):
                    tmp_wps = final_wps_list[i_ * NUM_WAYPOINTS_DISTANCE:i_ * NUM_WAYPOINTS_DISTANCE + NUM_WAYPOINTS_DISTANCE]
                    if len(tmp_wps) > 1:
                        cur_wp = tmp_wps[int(len(tmp_wps) / 2)]
                        prev_wp = tmp_wps[0]
                        nxt_wp = tmp_wps[-1]
                        vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                        vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location
                        norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x + vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(vec_wp_prev.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_prev.y)
                        try:
                            dot_ = (vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
                        except:
                            dot_ = -1
                        if not dot_ < -1 + DOT_PROD_SLACK:
                            truncated_wp_lst.append(tmp_wps)
                    locations = []
                    for wps_sub in truncated_wp_lst:
                        locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))
                    are_loc_dups = []
                    for location_ in locations:
                        flag_cum_ctr = []
                        for loc_dp in duplicate_list:
                            flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc + PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                            flag_AND_ctr = all(flag_ctrs)
                            flag_cum_ctr.append(flag_AND_ctr)
                        is_loc_dup = any(flag_cum_ctr)
                        are_loc_dups.append(is_loc_dup)
                    for j_, wps_ in enumerate(truncated_wp_lst):
                        if not are_loc_dups[j_]:
                            duplicate_list.append(locations[j_])
                            wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                            extended_route = interpolate_trajectory(carla_map, wps_tmp)
                            if len(extended_route) < PRUNE_ROUTES_MIN_LEN:
                                continue
                            else:
                                trigger_point = extended_route[int(len(extended_route) / 2)][0]
                                trigger_points += [trigger_point]
    other_actors = [None] * len(trigger_points)
    return (trigger_points, other_actors)

def generate_scenario_10(carla_map, world=None):
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    other_actors = []
    count_all_routes = 0
    duplicates = 0
    trigger_points = []
    actors = world.get_actors()
    traffic_lights_list = actors.filter('*traffic_light')
    print('got %d traffic lights' % len(traffic_lights_list))
    junction_bbs_centers = []
    junctions_ = []
    duplicate_list = []
    for wp_it, waypoint in enumerate(topology):
        if waypoint.is_junction:
            junc_ = waypoint.get_junction()
            jbb_ = junc_.bounding_box
            jbb_center = [round(jbb_.location.x, 2), round(jbb_.location.y, 2)]
            if jbb_center not in junction_bbs_centers:
                junction_bbs_centers.append(jbb_center)
                junctions_.append(junc_)
    pole_ind = []
    grps_ = []
    for tl_ in traffic_lights_list:
        grp_tl = tl_.get_group_traffic_lights()
        grp_tl_locs = [(round(gtl_.get_transform().location.x, 2), round(gtl_.get_transform().location.y, 2)) for gtl_ in grp_tl]
        pole_ind.append(tl_.get_pole_index())
        if grp_tl_locs not in grps_:
            grps_.append(grp_tl_locs)
    bb_flags = []
    for grp_ in grps_:
        midpt_grp = [sum(i) / len(grp_) for i in zip(*grp_)]
        grp_bb_dist = [np.sqrt((bb_c[0] - midpt_grp[0]) ** 2 + (bb_c[1] - midpt_grp[1]) ** 2) for bb_c in junction_bbs_centers]
        bb_dist, bb_idx = min(((val, idx) for idx, val in enumerate(grp_bb_dist)))
        bb_flags.append(bb_idx)
    unsignalized_junctions = [junc_ for i, junc_ in enumerate(junctions_) if i not in bb_flags]
    unsignalized_junctions_bbs = [(junc_.bounding_box.location.x, junc_.bounding_box.location.y) for i, junc_ in enumerate(junctions_) if i not in bb_flags]
    if len(unsignalized_junctions):
        count_all_routes = 0
        duplicates = 0
        PRECISION = 2
        SAMPLING_DISTANCE = 30
        duplicate_list = []
        for wp_it, waypoint in enumerate(topology):
            if waypoint.is_junction:
                junc_ = waypoint.get_junction()
                jbb_ = (junc_.bounding_box.location.x, junc_.bounding_box.location.y)
                if jbb_ in unsignalized_junctions_bbs:
                    j_wps = junc_.get_waypoints(carla.LaneType.Driving)
                    for it_, j_wp in enumerate(j_wps):
                        wp_p = j_wp[0]
                        dist_prev = 0
                        wp_list_prev = []
                        while True:
                            wp_list_prev.append(wp_p)
                            try:
                                wp_p = wp_p.previous(PRECISION)[0]
                            except:
                                break
                            dist_prev += PRECISION
                            if dist_prev > SAMPLING_DISTANCE:
                                break
                        dist_nxt = 0
                        wp_n = j_wp[1]
                        wp_list_nxt = []
                        while True:
                            wp_list_nxt.append(wp_n)
                            try:
                                wp_n = wp_n.next(PRECISION)[0]
                            except:
                                break
                            dist_nxt += PRECISION
                            if dist_nxt > SAMPLING_DISTANCE:
                                break
                        final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
                        truncated_wp_lst = [final_wps_list]
                        locations = []
                        for wps_sub in truncated_wp_lst:
                            locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))
                        are_loc_dups = []
                        for location_ in locations:
                            flag_cum_ctr = []
                            for loc_dp in duplicate_list:
                                flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc + PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                                flag_AND_ctr = all(flag_ctrs)
                                flag_cum_ctr.append(flag_AND_ctr)
                            is_loc_dup = any(flag_cum_ctr)
                            are_loc_dups.append(is_loc_dup)
                        for j_, wps_ in enumerate(truncated_wp_lst):
                            if not are_loc_dups[j_]:
                                count_all_routes += 1
                                duplicate_list.append(locations[j_])
                                wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                                extended_route = interpolate_trajectory(carla_map, wps_tmp)
                                trigger_points += [extended_route[0][0]]
                            else:
                                duplicates += 1
            other_actors = [None] * len(trigger_points)
    return (trigger_points, other_actors)

def main(args):
    scenarios_list = parse_annotations_file(args.scenarios_file)
    route_id = ID_START
    root = ET.Element('routes')
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    carla_map = world.get_map()
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    count_all_routes = 0
    duplicates = 0
    actors = world.get_actors()
    traffic_lights_list = actors.filter('*traffic_light')
    print('got %d traffic lights' % len(traffic_lights_list))
    junction_bbs_centers = []
    junctions_ = []
    duplicate_list = []
    for wp_it, waypoint in enumerate(topology):
        if waypoint.is_junction:
            junc_ = waypoint.get_junction()
            jbb_ = junc_.bounding_box
            jbb_center = [round(jbb_.location.x, 2), round(jbb_.location.y, 2)]
            if jbb_center not in junction_bbs_centers:
                junction_bbs_centers.append(jbb_center)
                junctions_.append(junc_)
    pole_ind = []
    grps_ = []
    for tl_ in traffic_lights_list:
        grp_tl = tl_.get_group_traffic_lights()
        grp_tl_locs = [(round(gtl_.get_transform().location.x, 2), round(gtl_.get_transform().location.y, 2)) for gtl_ in grp_tl]
        pole_ind.append(tl_.get_pole_index())
        if grp_tl_locs not in grps_:
            grps_.append(grp_tl_locs)
    bb_flags = []
    for grp_ in grps_:
        midpt_grp = [sum(i) / len(grp_) for i in zip(*grp_)]
        grp_bb_dist = [np.sqrt((bb_c[0] - midpt_grp[0]) ** 2 + (bb_c[1] - midpt_grp[1]) ** 2) for bb_c in junction_bbs_centers]
        bb_dist, bb_idx = min(((val, idx) for idx, val in enumerate(grp_bb_dist)))
        bb_flags.append(bb_idx)
    unsignalized_junctions = [junc_ for i, junc_ in enumerate(junctions_) if i not in bb_flags]
    unsignalized_junctions_bbs = [(junc_.bounding_box.location.x, junc_.bounding_box.location.y) for i, junc_ in enumerate(junctions_) if i not in bb_flags]
    if len(unsignalized_junctions):
        count_all_routes = 0
        duplicates = 0
        PRECISION = 2
        SAMPLING_DISTANCE = 30
        duplicate_list = []
        for wp_it, waypoint in enumerate(topology):
            if waypoint.is_junction:
                junc_ = waypoint.get_junction()
                jbb_ = (junc_.bounding_box.location.x, junc_.bounding_box.location.y)
                if jbb_ in unsignalized_junctions_bbs:
                    j_wps = junc_.get_waypoints(carla.LaneType.Driving)
                    for it_, j_wp in enumerate(j_wps):
                        wp_p = j_wp[0]
                        dist_prev = 0
                        wp_list_prev = []
                        while True:
                            wp_list_prev.append(wp_p)
                            try:
                                wp_p = wp_p.previous(PRECISION)[0]
                            except:
                                break
                            dist_prev += PRECISION
                            if dist_prev > SAMPLING_DISTANCE:
                                break
                        dist_nxt = 0
                        wp_n = j_wp[1]
                        wp_list_nxt = []
                        while True:
                            wp_list_nxt.append(wp_n)
                            try:
                                wp_n = wp_n.next(PRECISION)[0]
                            except:
                                break
                            dist_nxt += PRECISION
                            if dist_nxt > SAMPLING_DISTANCE:
                                break
                        final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
                        truncated_wp_lst = [final_wps_list]
                        locations = []
                        for wps_sub in truncated_wp_lst:
                            locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))
                        are_loc_dups = []
                        for location_ in locations:
                            flag_cum_ctr = []
                            for loc_dp in duplicate_list:
                                flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc + PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                                flag_AND_ctr = all(flag_ctrs)
                                flag_cum_ctr.append(flag_AND_ctr)
                            is_loc_dup = any(flag_cum_ctr)
                            are_loc_dups.append(is_loc_dup)
                        for j_, wps_ in enumerate(truncated_wp_lst):
                            if not are_loc_dups[j_]:
                                count_all_routes += 1
                                duplicate_list.append(locations[j_])
                                wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                                extended_route = interpolate_trajectory(carla_map, wps_tmp)
                                potential_scenarios_definitions, _ = scan_route_for_scenarios(carla_map.name, extended_route, scenarios_list)
                                wps_ = [wps_[0], wps_[-1]]
                                if len(potential_scenarios_definitions) > 0:
                                    route = ET.SubElement(root, 'route', id='%d' % route_id, town=args.town)
                                    for k_, wp_sub in enumerate(wps_):
                                        ET.SubElement(route, 'waypoint', x='%f' % wp_sub.transform.location.x, y='%f' % wp_sub.transform.location.y, z='0.0', pitch='0.0', roll='0.0', yaw='%f' % wp_sub.transform.rotation.yaw)
                                    route_id += 1
                            else:
                                duplicates += 1
        tree = ET.ElementTree(root)
        len_tree = 0
        for _ in tree.iter('route'):
            len_tree += 1
        print(f'Num routes for {args.town}: {len_tree}')
        if args.save_dir is not None and len_tree > 0:
            tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)

def parse_annotations_file(annotation_filename):
    """
    Return the annotations of which positions where the scenarios are going to happen.
    :param annotation_filename: the filename for the anotations file
    :return:
    """
    with open(annotation_filename, 'r') as f:
        annotation_dict = json.loads(f.read(), object_pairs_hook=OrderedDict)
    final_dict = OrderedDict()
    for town_dict in annotation_dict['available_scenarios']:
        final_dict.update(town_dict)
    return final_dict

def scan_route_for_scenarios(route_name, trajectory, world_annotations):
    """
        Just returns a plain list of possible scenarios that can happen in this route by matching
        the locations from the scenario into the route description

        :return:  A list of scenario definitions with their correspondent parameters
        """
    existent_triggers = OrderedDict()
    possible_scenarios = OrderedDict()
    latest_trigger_id = 0
    for town_name in world_annotations.keys():
        if town_name != route_name:
            continue
        scenarios = world_annotations[town_name]
        for scenario in scenarios:
            scenario_name = scenario['scenario_type']
            for event in scenario['available_event_configurations']:
                waypoint = event['transform']
                convert_waypoint_float(waypoint)
                match_position = match_world_location_to_route(waypoint, trajectory)
                if match_position is not None:
                    if 'other_actors' in event:
                        other_vehicles = event['other_actors']
                    else:
                        other_vehicles = None
                    scenario_subtype = get_scenario_type(scenario_name, match_position, trajectory)
                    if scenario_subtype is None:
                        continue
                    scenario_description = {'name': scenario_name, 'other_actors': other_vehicles, 'trigger_position': waypoint, 'scenario_type': scenario_subtype}
                    trigger_id = check_trigger_position(waypoint, existent_triggers)
                    if trigger_id is None:
                        existent_triggers.update({latest_trigger_id: waypoint})
                        possible_scenarios.update({latest_trigger_id: []})
                        trigger_id = latest_trigger_id
                        latest_trigger_id += 1
                    possible_scenarios[trigger_id].append(scenario_description)
    return (possible_scenarios, existent_triggers)

def main(args):
    route_id = 0
    duplicate_list = []
    count_all_routes = 0
    duplicates = 0
    distance_ = 100
    wp_length = 9
    PRECISION_small = 1
    WP_extended = 150
    root = {}
    root['lr'] = ET.Element('routes')
    root['ll'] = ET.Element('routes')
    root['rr'] = ET.Element('routes')
    root['rl'] = ET.Element('routes')
    final_save_dirs = {}
    for key_, _ in root.items():
        sub_path = os.path.join(args.save_dir, key_)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        final_save_dirs[key_] = sub_path
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    carla_map = world.get_map()
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    print(f'Num waypoints for {args.town}: {len(topology)}')
    for wp_it, cur_wp in enumerate(topology):
        wp_list_nxt = [cur_wp]
        if not cur_wp.is_junction:
            tmp_distance_ = 0
            while True:
                cur_wp_ = wp_list_nxt[-1]
                try:
                    nxt_wp = cur_wp_.next(PRECISION)[0]
                except:
                    break
                if not nxt_wp.is_junction and tmp_distance_ < distance_:
                    wp_list_nxt.append(nxt_wp)
                    tmp_distance_ += PRECISION
                else:
                    break
        if len(wp_list_nxt) > wp_length:
            final_wps_list = wp_list_nxt
            end_point = final_wps_list[-1]
            mid_point = final_wps_list[int(len(final_wps_list) / 2)]
            try:
                all_choices_ep = get_possible_lane_changes(end_point)
                all_choices_mp = get_possible_lane_changes(mid_point)
                if not len(all_choices_ep) > 1 and (not len(all_choices_mp) > 1):
                    continue
            except:
                continue
            all_combs_split = {'lr': [], 'll': [], 'rr': [], 'rl': []}
            all_combs = []
            for key_ep, ep_ in all_choices_ep.items():
                for key_mp, mp_ in all_choices_mp.items():
                    if key_ep != key_mp:
                        if key_mp != 'n':
                            mp_direction_ = set(key_mp)
                            mp_cnt_ = len(key_mp)
                            ep_direction_ = set(key_ep)
                            ep_cnt_ = len(key_ep)
                            if mp_direction_ == {'l'}:
                                if ep_direction_ == {'r'} or ep_direction_ == {'n'}:
                                    lane_change_key = 'lr'
                                elif ep_direction_ == {'l'}:
                                    if mp_cnt_ > ep_cnt_:
                                        lane_change_key = 'lr'
                                    else:
                                        lane_change_key = 'll'
                            elif mp_direction_ == {'r'}:
                                if ep_direction_ == {'l'} or ep_direction_ == {'n'}:
                                    lane_change_key = 'rl'
                                elif ep_direction_ == {'r'}:
                                    if mp_cnt_ > ep_cnt_:
                                        lane_change_key = 'rl'
                                    else:
                                        lane_change_key = 'rr'
                            final_wps = [final_wps_list[0], mp_, ep_]
                            all_combs_split[lane_change_key].append(final_wps)
            truncated_wp_lst = [final_wps_list]
            locations = []
            for wps_sub in truncated_wp_lst:
                locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y, wps_sub[0].transform.rotation.yaw))
            for location_ in locations:
                flag_cum_ctr = []
                for loc_dp in duplicate_list:
                    flag_ctrs = [True if prev_loc - PRECISION_small <= curr_loc <= prev_loc + PRECISION_small else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                    flag_AND_ctr = all(flag_ctrs)
                    flag_cum_ctr.append(flag_AND_ctr)
                is_loc_dup = any(flag_cum_ctr)
                if not is_loc_dup:
                    duplicate_list.append(locations[0])
                    for all_combs_key, all_combs in all_combs_split.items():
                        for j_, wps_ in enumerate(all_combs):
                            count_all_routes += 1
                            wps_tmp = [wps_[0].transform.location, wps_[1].transform.location, wps_[-1].transform.location]
                            try:
                                extended_route = interpolate_trajectory(carla_map, wps_tmp)
                                if len(extended_route) > WP_extended or len(extended_route) < PRUNE_ROUTES_MIN_LEN:
                                    continue
                            except:
                                continue
                            wps_tmp2 = wps_
                            route = ET.SubElement(root[all_combs_key], 'route', id='%d' % route_id, town=args.town)
                            for k_, wp_sub in enumerate(wps_tmp2):
                                ET.SubElement(route, 'waypoint', x='%f' % wp_sub.transform.location.x, y='%f' % wp_sub.transform.location.y, z='%f' % wp_sub.transform.location.z, pitch='0.0', roll='0.0', yaw='%f' % wp_sub.transform.rotation.yaw)
                                route_id += 1
                else:
                    duplicates += 1
    tree = {}
    root_pruned = {}
    for key_ in ['rr', 'lr', 'll', 'rl']:
        root_pruned[key_] = ET.Element('routes')
        index_list = list(range(len(root[key_])))
        random.shuffle(index_list)
        index_list_pruned = index_list[:LIMIT_FINAL_ROUTES]
        route_id_pruned = 0
        for ind_, child_ in enumerate(root[key_]):
            if ind_ in index_list_pruned:
                route_new = ET.SubElement(root_pruned[key_], 'route', id='%d' % route_id_pruned, town=args.town)
                for subelement_ in child_.findall('waypoint'):
                    ET.SubElement(route_new, subelement_.tag, subelement_.attrib)
                route_id_pruned += 1
        tree = ET.ElementTree(root_pruned[key_])
        len_tree = 0
        for _ in tree.iter('route'):
            len_tree += 1
        print(f'Num routes for {args.town}: {len_tree}')
        if args.save_dir is not None and len_tree > 0:
            filename_ = os.path.join(final_save_dirs[key_], town_ + '_' + key_ + '.xml')
            tree.write(filename_, xml_declaration=True, encoding='utf-8', pretty_print=True)

def get_possible_lane_changes(current_waypoint):
    all_lefts = {}
    all_rights = {}
    tmp_wp = current_waypoint
    lane_side = 'l'
    while True:
        left_w = tmp_wp.get_left_lane()
        if left_w and left_w.lane_type == carla.LaneType.Driving and (0 <= abs(left_w.transform.rotation.yaw - tmp_wp.transform.rotation.yaw) <= 10):
            all_lefts[lane_side] = left_w
            tmp_wp = left_w
            lane_side += 'l'
        else:
            break
    tmp_wp = current_waypoint
    lane_side = 'r'
    while True:
        right_w = tmp_wp.get_right_lane()
        if right_w and right_w.lane_type == carla.LaneType.Driving and (0 <= abs(right_w.transform.rotation.yaw - tmp_wp.transform.rotation.yaw) <= 10):
            all_rights[lane_side] = right_w
            tmp_wp = right_w
            lane_side += 'r'
        else:
            break
    current_dict = {'n': current_waypoint}
    all_choices = {**all_lefts, **all_rights, **current_dict}
    return all_choices

def main(args):
    route_id = ID_START
    road_type = args.road_type
    root = ET.Element('routes')
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    carla_map = world.get_map()
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    scenarios_list = parse_annotations_file(args.scenarios_file)
    count_all_routes = 0
    duplicates = 0
    if road_type == 'curved':
        DOT_PROD_SLACK = 0.02
        PRECISION = 2
        DISTANCE = 380
        PRUNE_ROUTES_MIN_LEN = 20
        NUM_WAYPOINTS_DISTANCE = int(DISTANCE / PRECISION)
        MIN_ROUTE_LENGTH = 4
        duplicate_list = []
        for wp_it, waypoint in enumerate(topology):
            cur_wp = waypoint
            wp_list_nxt = [cur_wp]
            if not cur_wp.is_junction:
                while True:
                    cur_wp_ = wp_list_nxt[-1]
                    try:
                        nxt_wp = cur_wp_.next(PRECISION)[0]
                    except:
                        break
                    if not nxt_wp.is_junction:
                        wp_list_nxt.append(nxt_wp)
                    else:
                        break
            wp_list_prev = [cur_wp]
            if not cur_wp.is_junction:
                while True:
                    cur_wp_ = wp_list_prev[-1]
                    try:
                        nxt_wp = cur_wp_.previous(PRECISION)[0]
                    except:
                        break
                    if not nxt_wp.is_junction:
                        wp_list_prev.append(nxt_wp)
                    else:
                        break
            if len(wp_list_prev) + len(wp_list_nxt) > MIN_ROUTE_LENGTH:
                final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
                cur_wp = final_wps_list[int(len(final_wps_list) / 2)]
                prev_wp = final_wps_list[0]
                nxt_wp = final_wps_list[-1]
                vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location
                norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x + vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(vec_wp_prev.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_prev.y)
                try:
                    dot_ = (vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
                except:
                    dot_ = -1
                if dot_ > -1 - DOT_PROD_SLACK and dot_ < -1 + DOT_PROD_SLACK:
                    continue
                else:
                    truncated_wp_lst = []
                    for i_ in range(len(final_wps_list)):
                        tmp_wps = final_wps_list[i_ * NUM_WAYPOINTS_DISTANCE:i_ * NUM_WAYPOINTS_DISTANCE + NUM_WAYPOINTS_DISTANCE]
                        if len(tmp_wps) > 1:
                            cur_wp = tmp_wps[int(len(tmp_wps) / 2)]
                            prev_wp = tmp_wps[0]
                            nxt_wp = tmp_wps[-1]
                            vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                            vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location
                            norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x + vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(vec_wp_prev.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_prev.y)
                            try:
                                dot_ = (vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
                            except:
                                dot_ = -1
                            if not dot_ < -1 + DOT_PROD_SLACK:
                                truncated_wp_lst.append(tmp_wps)
                        locations = []
                        for wps_sub in truncated_wp_lst:
                            locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))
                        are_loc_dups = []
                        for location_ in locations:
                            flag_cum_ctr = []
                            for loc_dp in duplicate_list:
                                flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc + PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                                flag_AND_ctr = all(flag_ctrs)
                                flag_cum_ctr.append(flag_AND_ctr)
                            is_loc_dup = any(flag_cum_ctr)
                            are_loc_dups.append(is_loc_dup)
                        for j_, wps_ in enumerate(truncated_wp_lst):
                            if not are_loc_dups[j_]:
                                count_all_routes += 1
                                duplicate_list.append(locations[j_])
                                wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                                extended_route = interpolate_trajectory(carla_map, wps_tmp)
                                potential_scenarios_definitions, _ = scan_route_for_scenarios(args.town, extended_route, scenarios_list)
                                wps_ = [wps_[0], wps_[-1]]
                                if (len(extended_route) <= MAX_LEN and len(potential_scenarios_definitions) > 0) and len(extended_route) > PRUNE_ROUTES_MIN_LEN:
                                    route = ET.SubElement(root, 'route', id='%d' % route_id, town=args.town)
                                    for k_, wp_sub in enumerate(wps_):
                                        ET.SubElement(route, 'waypoint', x='%f' % wp_sub.transform.location.x, y='%f' % wp_sub.transform.location.y, z='0.0', pitch='0.0', roll='0.0', yaw='%f' % wp_sub.transform.rotation.yaw)
                                    route_id += 1
                            else:
                                duplicates += 1
    else:
        PRECISION = 2
        SAMPLING_DISTANCE = 30
        duplicate_list = []
        for wp_it, waypoint in enumerate(topology):
            if waypoint.is_junction:
                junc_ = waypoint.get_junction()
                jbb_ = junc_.bounding_box
                j_wps = junc_.get_waypoints(carla.LaneType.Driving)
                for it_, j_wp in enumerate(j_wps):
                    wp_p = j_wp[0]
                    dist_prev = 0
                    wp_list_prev = []
                    while True:
                        wp_list_prev.append(wp_p)
                        try:
                            wp_p = wp_p.previous(PRECISION)[0]
                        except:
                            break
                        dist_prev += PRECISION
                        if dist_prev > SAMPLING_DISTANCE:
                            break
                    dist_nxt = 0
                    wp_n = j_wp[1]
                    wp_list_nxt = []
                    while True:
                        wp_list_nxt.append(wp_n)
                        try:
                            wp_n = wp_n.next(PRECISION)[0]
                        except:
                            break
                        dist_nxt += PRECISION
                        if dist_nxt > SAMPLING_DISTANCE:
                            break
                    final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
                    truncated_wp_lst = [final_wps_list]
                    locations = []
                    for wps_sub in truncated_wp_lst:
                        locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))
                    are_loc_dups = []
                    for location_ in locations:
                        flag_cum_ctr = []
                        for loc_dp in duplicate_list:
                            flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc + PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                            flag_AND_ctr = all(flag_ctrs)
                            flag_cum_ctr.append(flag_AND_ctr)
                        is_loc_dup = any(flag_cum_ctr)
                        are_loc_dups.append(is_loc_dup)
                    for j_, wps_ in enumerate(truncated_wp_lst):
                        if not are_loc_dups[j_]:
                            count_all_routes += 1
                            duplicate_list.append(locations[j_])
                            wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                            extended_route = interpolate_trajectory(carla_map, wps_tmp)
                            potential_scenarios_definitions, _ = scan_route_for_scenarios(args.town, extended_route, scenarios_list)
                            if not len(potential_scenarios_definitions):
                                continue
                            wps_ = [wps_[0], wps_[-1]]
                            if len(extended_route) < MAX_LEN and len(potential_scenarios_definitions) > 0:
                                route = ET.SubElement(root, 'route', id='%d' % route_id, town=args.town)
                                for k_, wp_sub in enumerate(wps_):
                                    ET.SubElement(route, 'waypoint', x='%f' % wp_sub.transform.location.x, y='%f' % wp_sub.transform.location.y, z='0.0', pitch='0.0', roll='0.0', yaw='%f' % wp_sub.transform.rotation.yaw)
                                route_id += 1
                        else:
                            duplicates += 1
    tree = ET.ElementTree(root)
    len_tree = 0
    for _ in tree.iter('route'):
        len_tree += 1
    print(f'Num routes for {args.town}: {len_tree}')
    if args.save_dir is not None and len_tree > 0:
        tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)

def main(args):
    route_id = ID_START
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    carla_map = world.get_map()
    actors = world.get_actors()
    traffic_lights_list = actors.filter('*traffic_light')
    print('got %d traffic lights' % len(traffic_lights_list))
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    root_7 = ET.Element('routes')
    root_8 = ET.Element('routes')
    root_9 = ET.Element('routes')
    roots_ = {'Scenario7': root_7, 'Scenario8': root_8, 'Scenario9': root_9}
    count_all_routes = 0
    duplicates = 0
    scenarios_list = {}
    scenarios_list['Scenario7'] = parse_annotations_file(args.scenarios_file['Scenario7'])
    scenarios_list['Scenario8'] = parse_annotations_file(args.scenarios_file['Scenario8'])
    scenarios_list['Scenario9'] = parse_annotations_file(args.scenarios_file['Scenario9'])
    junction_bbs_centers = []
    junctions_ = []
    duplicate_list = []
    for wp_it, waypoint in enumerate(topology):
        if waypoint.is_junction:
            junc_ = waypoint.get_junction()
            jbb_ = junc_.bounding_box
            jbb_center = [round(jbb_.location.x, 2), round(jbb_.location.y, 2)]
            if jbb_center not in junction_bbs_centers:
                junction_bbs_centers.append(jbb_center)
                junctions_.append(junc_)
    pole_ind = []
    grps_ = []
    for tl_ in traffic_lights_list:
        grp_tl = tl_.get_group_traffic_lights()
        grp_tl_locs = [(round(gtl_.get_transform().location.x, 2), round(gtl_.get_transform().location.y, 2)) for gtl_ in grp_tl]
        pole_ind.append(tl_.get_pole_index())
        if grp_tl_locs not in grps_:
            grps_.append(grp_tl_locs)
    bb_flags = []
    for grp_ in grps_:
        midpt_grp = [sum(i) / len(grp_) for i in zip(*grp_)]
        grp_bb_dist = [np.sqrt((bb_c[0] - midpt_grp[0]) ** 2 + (bb_c[1] - midpt_grp[1]) ** 2) for bb_c in junction_bbs_centers]
        bb_dist, bb_idx = min(((val, idx) for idx, val in enumerate(grp_bb_dist)))
        bb_flags.append(bb_idx)
    signalized_junctions = [junc_ for i, junc_ in enumerate(junctions_) if i in bb_flags]
    signalized_junctions_bbs = [(junc_.bounding_box.location.x, junc_.bounding_box.location.y) for i, junc_ in enumerate(junctions_) if i in bb_flags]
    if len(signalized_junctions):
        count_all_routes = 0
        duplicates = 0
        duplicate_list = []
        for wp_it, waypoint in enumerate(topology):
            if waypoint.is_junction:
                junc_ = waypoint.get_junction()
                jbb_ = (junc_.bounding_box.location.x, junc_.bounding_box.location.y)
                if jbb_ in signalized_junctions_bbs:
                    j_wps = junc_.get_waypoints(carla.LaneType.Driving)
                    for it_, j_wp in enumerate(j_wps):
                        wp_p = j_wp[0]
                        dist_prev = 0
                        wp_list_prev = []
                        while True:
                            wp_list_prev.append(wp_p)
                            try:
                                wp_p = wp_p.previous(PRECISION)[0]
                            except:
                                break
                            dist_prev += PRECISION
                            if dist_prev > SAMPLING_DISTANCE:
                                break
                        dist_nxt = 0
                        wp_n = j_wp[1]
                        wp_list_nxt = []
                        while True:
                            wp_list_nxt.append(wp_n)
                            try:
                                wp_n = wp_n.next(PRECISION)[0]
                            except:
                                break
                            dist_nxt += PRECISION
                            if dist_nxt > SAMPLING_DISTANCE:
                                break
                        final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
                        cur_wp = final_wps_list[int(len(final_wps_list) / 2)]
                        prev_wp = final_wps_list[0]
                        nxt_wp = final_wps_list[-1]
                        vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                        vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location
                        dot_ = vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y
                        det_ = vec_wp_nxt.x * vec_wp_prev.y - vec_wp_prev.x * vec_wp_nxt.y
                        angle_bet = math.atan2(det_, dot_)
                        if angle_bet < 0:
                            angle_bet += 2 * math.pi
                        angle_deg = angle_bet * 180 / math.pi
                        if 160 < angle_deg < 195:
                            key_ = 'Scenario7'
                            root = roots_['Scenario7']
                        elif 10 < angle_deg < 160:
                            key_ = 'Scenario9'
                            root = roots_['Scenario9']
                        elif 195 < angle_deg < 350:
                            key_ = 'Scenario8'
                            root = roots_['Scenario8']
                        truncated_wp_lst = [final_wps_list]
                        locations = []
                        for wps_sub in truncated_wp_lst:
                            locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))
                        are_loc_dups = []
                        for location_ in locations:
                            flag_cum_ctr = []
                            for loc_dp in duplicate_list:
                                flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc + PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp)]
                                flag_AND_ctr = all(flag_ctrs)
                                flag_cum_ctr.append(flag_AND_ctr)
                            is_loc_dup = any(flag_cum_ctr)
                            are_loc_dups.append(is_loc_dup)
                        for j_, wps_ in enumerate(truncated_wp_lst):
                            if not are_loc_dups[j_]:
                                count_all_routes += 1
                                duplicate_list.append(locations[j_])
                                wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                                extended_route = interpolate_trajectory(carla_map, wps_tmp)
                                potential_scenarios_definitions, _ = scan_route_for_scenarios(carla_map.name, extended_route, scenarios_list[key_])
                                potential_scenarios_definitions = []
                                wps_ = [wps_[0], wps_[-1]]
                                if True or len(potential_scenarios_definitions) > 0:
                                    route = ET.SubElement(root, 'route', id='%d' % route_id, town=args.town)
                                    for k_, wp_sub in enumerate(wps_):
                                        ET.SubElement(route, 'waypoint', x='%f' % wp_sub.transform.location.x, y='%f' % wp_sub.transform.location.y, z='0.0', pitch='0.0', roll='0.0', yaw='%f' % wp_sub.transform.rotation.yaw)
                                    route_id += 1
                            else:
                                duplicates += 1
        tree_7 = ET.ElementTree(root_7)
        tree_8 = ET.ElementTree(root_8)
        tree_9 = ET.ElementTree(root_9)
        len_tree = 0
        for _ in tree_7.iter('route'):
            len_tree += 1
        print(f'Num routes for Scenario 7 for {args.town}: {len_tree}')
        if args.save_dir is not None and len_tree > 0:
            tree_7.write(args.save_file['Scenario7'], xml_declaration=True, encoding='utf-8', pretty_print=True)
        len_tree = 0
        for _ in tree_8.iter('route'):
            len_tree += 1
        print(f'Num routes for Scenario 8 for {args.town}: {len_tree}')
        if args.save_dir is not None and len_tree > 0:
            tree_8.write(args.save_file['Scenario8'], xml_declaration=True, encoding='utf-8', pretty_print=True)
        len_tree = 0
        for _ in tree_9.iter('route'):
            len_tree += 1
        print(f'Num routes for Scenario 9 for {args.town}: {len_tree}')
        if args.save_dir is not None and len_tree > 0:
            tree_9.write(args.save_file['Scenario9'], xml_declaration=True, encoding='utf-8', pretty_print=True)

class AutoPilot(autonomous_agent_local.AutonomousAgent):

    def setup(self, path_to_conf_file, route_index=None):
        self.track = autonomous_agent.Track.MAP
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.save_path = None
        self.render_bev = False
        self.route_index = route_index
        self.frame_rate_sim = 20
        self.gps_buffer = deque(maxlen=100)
        self.frame_rate = 20
        self.ego_model = EgoModel(dt=1.0 / self.frame_rate)
        self.ego_model_gps = EgoModel(dt=1.0 / self.frame_rate_sim)
        self.vehicle_model = EgoModel(dt=1.0 / self.frame_rate)
        self.visualize = int(os.environ['DEBUG_CHALLENGE'])
        self.save_freq = self.frame_rate_sim // 2
        self.steer_buffer_size = 1
        self.target_speed_slow = 3.0
        self.target_speed_fast = 4.0
        self.clip_delta = 0.25
        self.clip_throttle = 0.75
        self.steer_damping = 0.5
        self.slope_pitch = 10.0
        self.slope_throttle = 0.4
        self.angle_search_range = 0
        self.steer_noise = 0.001
        self.steer_buffer = deque(maxlen=self.steer_buffer_size)
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._turn_controller_extrapolation = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self._speed_controller_extrapolation = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self.center_bb_light_x = -2.0
        self.center_bb_light_y = 0.0
        self.center_bb_light_z = 0.0
        self.extent_bb_light_x = 4.5
        self.extent_bb_light_y = 1.5
        self.extent_bb_light_z = 2.0
        self.extrapolation_seconds_no_junction = 1.0
        self.extrapolation_seconds = 4.0
        self.waypoint_seconds = 4.0
        self.detection_radius = 30.0
        self.light_radius = 15.0
        self.vehicle_speed_buffer = defaultdict(lambda: {'velocity': [], 'throttle': [], 'brake': []})
        self.stuck_buffer_size = 30
        self.stuck_vel_threshold = 0.1
        self.stuck_throttle_threshold = 0.1
        self.stuck_brake_threshold = 0.1
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.far_node_prev = [100000.0, 100000.0]
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.target_speed = 4.0
        self.angle = 0.0
        self.stop_sign_hazard = False
        self.traffic_light_hazard = False
        self.walker_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        self.vehicle_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        self.junction = False
        self.ignore_stop_signs = True
        self.cleared_stop_signs = []
        self.future_states = {}
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += f'route{self.route_index}_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
            print(string)
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'measurements').mkdir()

    def _init(self, hd_map):
        self.world_map = carla.Map('RouteMap', hd_map[1]['opendrive'])
        trajectory = [item[0].location for item in self._global_plan_world_coord]
        self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)
        print('Sparse Waypoints:', len(self._global_plan))
        print('Dense Waypoints:', len(self.dense_route))
        self._waypoint_planner = RoutePlanner(3.5, 50)
        self._waypoint_planner.set_route(self.dense_route, True)
        self._waypoint_planner.save()
        self._waypoint_planner_extrapolation = RoutePlanner(3.5, 50)
        self._waypoint_planner_extrapolation.set_route(self.dense_route, True)
        self._waypoint_planner_extrapolation.save()
        self._command_planner = RoutePlanner(7.5, 50)
        self._command_planner.set_route(self._global_plan, True)
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self.initialized = True

    def sensors(self):
        return [{'type': 'sensor.opendrive_map', 'reading_frequency': 1e-06, 'id': 'hd_map'}, {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.05, 'id': 'imu'}, {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.01, 'id': 'gps'}, {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'}]

    def tick(self, input_data):
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if math.isnan(compass) == True:
            compass = 0.0
        result = {'gps': gps, 'speed': speed, 'compass': compass}
        return result

    def run_step(self, input_data, timestamp):
        self.step += 1
        if not self.initialized:
            if 'hd_map' in input_data.keys():
                self._init(input_data['hd_map'])
            else:
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                return control
        control = self._get_control(input_data)
        tick_data_tmp = self.tick(input_data)
        self.update_gps_buffer(control, tick_data_tmp['compass'], tick_data_tmp['speed'])
        return control

    def update_gps_buffer(self, control, theta, speed):
        yaw = np.array([theta - np.pi / 2.0])
        speed = np.array([speed])
        action = np.array(np.stack([control.steer, control.throttle, control.brake], axis=-1))
        for i in range(len(self.gps_buffer)):
            loc = self.gps_buffer[i]
            loc_temp = np.array([loc[1], -loc[0]])
            next_loc_tmp, _, _ = self.ego_model_gps.forward(loc_temp, yaw, speed, action)
            next_loc = np.array([-next_loc_tmp[1], next_loc_tmp[0]])
            self.gps_buffer[i] = next_loc
        return None

    def get_future_states(self):
        return self.future_states

    def _get_control(self, input_data, steer=None, throttle=None, vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None):
        if vehicle_hazard is None or light_hazard is None or walker_hazard is None or (stop_sign_hazard is None):
            brake = self._get_brake(vehicle_hazard, light_hazard, walker_hazard, stop_sign_hazard)
        else:
            brake = vehicle_hazard or light_hazard or walker_hazard or stop_sign_hazard
        ego_vehicle_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
        self.junction = ego_vehicle_waypoint.is_junction
        speed = input_data['speed'][1]['speed']
        target_speed = self.target_speed_slow if self.junction else self.target_speed_fast
        pos = self._get_position(input_data['gps'][1][:2])
        self.gps_buffer.append(pos)
        pos = np.average(self.gps_buffer, axis=0)
        self._waypoint_planner.load()
        waypoint_route = self._waypoint_planner.run_step(pos)
        near_node, near_command = waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0]
        self._waypoint_planner.save()
        self._waypoint_planner_extrapolation.load()
        self.waypoint_route_extrapolation = self._waypoint_planner_extrapolation.run_step(pos)
        self._waypoint_planner_extrapolation.save()
        if throttle is None:
            throttle = self._get_throttle(brake, target_speed, speed)
            if self._vehicle.get_transform().rotation.pitch > self.slope_pitch:
                throttle += self.slope_throttle
        if steer is None:
            theta = input_data['imu'][1][-1]
            if math.isnan(theta):
                theta = 0.0
            steer = self._get_steer(brake, waypoint_route, pos, theta, speed)
            steer_extrapolation = self._get_steer_extrapolation(waypoint_route, pos, theta, speed)
        self.steer_buffer.append(steer)
        control = carla.VehicleControl()
        control.steer = np.mean(self.steer_buffer) + self.steer_noise * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)
        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        self.target_speed = target_speed
        self._save_waypoints()
        if self.step % self.save_freq == 0 and self.save_path is not None:
            command_route = self._command_planner.run_step(pos)
            far_node, far_command = command_route[1] if len(command_route) > 1 else command_route[0]
            if (far_node != self.far_node_prev).all():
                self.far_node_prev = far_node
                self.commands.append(far_command.value)
            if self.render_bev == False:
                tick_data = self.tick(input_data)
            else:
                tick_data = self.tick(input_data, self.future_states)
            self.save(far_node, steer, throttle, brake, target_speed, tick_data)
        return control

    def save(self, far_node, steer, throttle, brake, target_speed, tick_data):
        frame = self.step // self.save_freq
        pos = self._get_position(tick_data['gps'])
        theta = tick_data['compass']
        speed = tick_data['speed']
        waypoints = []
        for i, box in enumerate(self.future_states['ego']):
            if (i + 1) % (self.frame_rate / 2) == 0:
                box_x = -box.location.y
                box_y = box.location.x
                box_theta = box.rotation.yaw * np.pi / 180.0 + np.pi / 2
                if box_theta < 0:
                    box_theta += 2 * np.pi
                waypoints.append((box_x, box_y, box_theta))
        data = {'x': pos[0], 'y': pos[1], 'theta': theta, 'speed': speed, 'target_speed': target_speed, 'x_command': far_node[0], 'y_command': far_node[1], 'command': self.commands[-2], 'waypoints': waypoints, 'steer': steer, 'throttle': throttle, 'brake': brake, 'junction': self.junction, 'vehicle_hazard': self.vehicle_hazard, 'light_hazard': self.traffic_light_hazard, 'walker_hazard': self.walker_hazard, 'stop_sign_hazard': self.stop_sign_hazard, 'angle': self.angle, 'ego_matrix': self._vehicle.get_transform().get_matrix()}
        measurements_file = self.save_path / 'measurements' / ('%04d.json' % frame)
        with open(measurements_file, 'w') as f:
            json.dump(data, f, indent=4)

    def destroy(self):
        pass

    def _get_steer(self, brake, route, pos, theta, speed, restore=True):
        if self._waypoint_planner.is_last:
            angle = 0.0
        if speed < 0.01 and brake == True:
            angle = 0.0
        if len(route) == 1:
            target = route[0][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        elif self.angle_search_range <= 2:
            target = route[1][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        else:
            search_range = min([len(route), self.angle_search_range])
            for i in range(1, search_range):
                target = route[i][0]
                angle_unnorm = self._get_angle_to(pos, theta, target)
                angle_new = angle_unnorm / 90
                if i == 1:
                    angle = angle_new
                if np.abs(angle_new) < np.abs(angle):
                    angle = angle_new
        self.angle = angle
        if restore:
            self._turn_controller.load()
        steer = self._turn_controller.step(angle)
        if restore:
            self._turn_controller.save()
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)
        if brake:
            steer *= self.steer_damping
        return steer

    def _get_steer_extrapolation(self, route, pos, theta, speed, restore=True):
        if self._waypoint_planner_extrapolation.is_last:
            angle = 0.0
        if len(route) == 1:
            target = route[0][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        elif self.angle_search_range <= 2:
            target = route[1][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        else:
            search_range = min([len(route), self.angle_search_range])
            for i in range(1, search_range):
                target = route[i][0]
                angle_unnorm = self._get_angle_to(pos, theta, target)
                angle_new = angle_unnorm / 90
                if i == 1:
                    angle = angle_new
                if np.abs(angle_new) < np.abs(angle):
                    angle = angle_new
        self.angle = angle
        if restore:
            self._turn_controller_extrapolation.load()
        steer = self._turn_controller_extrapolation.step(angle)
        if restore:
            self._turn_controller_extrapolation.save()
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)
        return steer

    def _get_throttle(self, brake, target_speed, speed, restore=True):
        target_speed = target_speed if not brake else 0.0
        if self._waypoint_planner.is_last:
            target_speed = 0.0
        delta = np.clip(target_speed - speed, 0.0, self.clip_delta)
        if restore:
            self._speed_controller.load()
        throttle = self._speed_controller.step(delta)
        if restore:
            self._speed_controller.save()
        throttle = np.clip(throttle, 0.0, self.clip_throttle)
        if brake:
            throttle = 0.0
        return throttle

    def _get_throttle_extrapolation(self, target_speed, speed, restore=True):
        if self._waypoint_planner_extrapolation.is_last:
            target_speed = 0.0
        delta = np.clip(target_speed - speed, 0.0, self.clip_delta)
        if restore:
            self._speed_controller_extrapolation.load()
        throttle = self._speed_controller_extrapolation.step(delta)
        if restore:
            self._speed_controller_extrapolation.save()
        throttle = np.clip(throttle, 0.0, self.clip_throttle)
        return throttle

    def _get_brake(self, vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None):
        actors = self._world.get_actors()
        speed = self._get_forward_speed()
        vehicle_location = self._vehicle.get_location()
        vehicle_transform = self._vehicle.get_transform()
        vehicles = actors.filter('*vehicle*')
        if light_hazard is None:
            light_hazard = False
            self._active_traffic_light = None
            _traffic_lights = self.get_nearby_object(vehicle_location, actors.filter('*traffic_light*'), self.light_radius)
            center_light_detector_bb = vehicle_transform.transform(carla.Location(x=self.center_bb_light_x, y=self.center_bb_light_y, z=self.center_bb_light_z))
            extent_light_detector_bb = carla.Vector3D(x=self.extent_bb_light_x, y=self.extent_bb_light_y, z=self.extent_bb_light_z)
            light_detector_bb = carla.BoundingBox(center_light_detector_bb, extent_light_detector_bb)
            light_detector_bb.rotation = vehicle_transform.rotation
            color2 = carla.Color(255, 255, 255, 255)
            for light in _traffic_lights:
                if light.state == carla.libcarla.TrafficLightState.Red:
                    color = carla.Color(255, 0, 0, 255)
                elif light.state == carla.libcarla.TrafficLightState.Yellow:
                    color = carla.Color(255, 255, 0, 255)
                elif light.state == carla.libcarla.TrafficLightState.Green:
                    color = carla.Color(0, 255, 0, 255)
                elif light.state == carla.libcarla.TrafficLightState.Off:
                    color = carla.Color(0, 0, 0, 255)
                else:
                    color = carla.Color(0, 0, 255, 255)
                size = 0.1
                center_bounding_box = light.get_transform().transform(light.trigger_volume.location)
                center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y, center_bounding_box.z)
                length_bounding_box = carla.Vector3D(light.trigger_volume.extent.x, light.trigger_volume.extent.y, light.trigger_volume.extent.z)
                transform = carla.Transform(center_bounding_box)
                bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(pitch=light.trigger_volume.rotation.pitch + gloabl_rot.pitch, yaw=light.trigger_volume.rotation.yaw + gloabl_rot.yaw, roll=light.trigger_volume.rotation.roll + gloabl_rot.roll)
                if self.visualize == 1:
                    self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
                if self.check_obb_intersection(light_detector_bb, bounding_box) == True:
                    if light.state == carla.libcarla.TrafficLightState.Red or light.state == carla.libcarla.TrafficLightState.Yellow:
                        self._active_traffic_light = light
                        light_hazard = True
                        color2 = carla.Color(255, 0, 0, 255)
            if self.visualize == 1:
                self._world.debug.draw_box(box=light_detector_bb, rotation=light_detector_bb.rotation, thickness=0.1, color=color2, life_time=1.0 / self.frame_rate_sim)
        if stop_sign_hazard is None:
            stop_sign_hazard = False
            if not self.ignore_stop_signs:
                stop_signs = self.get_nearby_object(vehicle_location, actors.filter('*stop*'), self.light_radius)
                center_vehicle_stop_sign_detector_bb = vehicle_transform.transform(self._vehicle.bounding_box.location)
                extent_vehicle_stop_sign_detector_bb = self._vehicle.bounding_box.extent
                vehicle_stop_sign_detector_bb = carla.BoundingBox(center_vehicle_stop_sign_detector_bb, extent_vehicle_stop_sign_detector_bb)
                vehicle_stop_sign_detector_bb.rotation = vehicle_transform.rotation
                for stop_sign in stop_signs:
                    center_bb_stop_sign = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
                    length_bb_stop_sign = carla.Vector3D(stop_sign.trigger_volume.extent.x, stop_sign.trigger_volume.extent.y, stop_sign.trigger_volume.extent.z)
                    transform_stop_sign = carla.Transform(center_bb_stop_sign)
                    bounding_box_stop_sign = carla.BoundingBox(transform_stop_sign.location, length_bb_stop_sign)
                    rotation_stop_sign = stop_sign.get_transform().rotation
                    bounding_box_stop_sign.rotation = carla.Rotation(pitch=stop_sign.trigger_volume.rotation.pitch + rotation_stop_sign.pitch, yaw=stop_sign.trigger_volume.rotation.yaw + rotation_stop_sign.yaw, roll=stop_sign.trigger_volume.rotation.roll + rotation_stop_sign.roll)
                    color = carla.Color(0, 255, 0, 255)
                    if self.check_obb_intersection(vehicle_stop_sign_detector_bb, bounding_box_stop_sign) == True:
                        if not stop_sign.id in self.cleared_stop_signs:
                            if speed * 3.6 > 0.0:
                                stop_sign_hazard = True
                                color = carla.Color(255, 0, 0, 255)
                            else:
                                self.cleared_stop_signs.append(stop_sign.id)
                    if self.visualize == 1:
                        self._world.debug.draw_box(box=bounding_box_stop_sign, rotation=bounding_box_stop_sign.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
                for cleared_stop_sign in self.cleared_stop_signs:
                    remove_stop_sign = True
                    for stop_sign in stop_signs:
                        if stop_sign.id == cleared_stop_sign:
                            remove_stop_sign = False
                    if remove_stop_sign == True:
                        self.cleared_stop_signs.remove(cleared_stop_sign)
        if vehicle_hazard is None or walker_hazard is None:
            vehicle_hazard = False
            self.vehicle_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
            extrapolation_seconds = self.extrapolation_seconds
            detection_radius = self.detection_radius
            number_of_future_frames = int(extrapolation_seconds * self.frame_rate)
            number_of_future_frames_no_junction = int(self.extrapolation_seconds_no_junction * self.frame_rate)
            color = carla.Color(0, 255, 0, 255)
            walkers = actors.filter('*walker*')
            walker_hazard = False
            self.walker_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
            nearby_walkers = []
            for walker in walkers:
                if walker.get_location().distance(vehicle_location) < detection_radius:
                    walker_future_bbs = []
                    walker_transform = walker.get_transform()
                    walker_velocity = walker.get_velocity()
                    walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)
                    walker_location = walker_transform.location
                    walker_direction = walker.get_control().direction
                    for i in range(number_of_future_frames):
                        if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                            break
                        new_x = walker_location.x + walker_direction.x * walker_speed * (1.0 / self.frame_rate)
                        new_y = walker_location.y + walker_direction.y * walker_speed * (1.0 / self.frame_rate)
                        new_z = walker_location.z + walker_direction.z * walker_speed * (1.0 / self.frame_rate)
                        walker_location = carla.Location(new_x, new_y, new_z)
                        transform = carla.Transform(walker_location)
                        bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
                        bounding_box.rotation = carla.Rotation(pitch=walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch, yaw=walker.bounding_box.rotation.yaw + walker_transform.rotation.yaw, roll=walker.bounding_box.rotation.roll + walker_transform.rotation.roll)
                        color = carla.Color(0, 0, 255, 255)
                        if self.visualize == 1:
                            self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
                        walker_future_bbs.append(bounding_box)
                    nearby_walkers.append(walker_future_bbs)
            nearby_vehicles = {}
            tmp_near_vehicle_id = []
            tmp_stucked_vehicle_id = []
            for vehicle in vehicles:
                if vehicle.id == self._vehicle.id:
                    continue
                if vehicle.get_location().distance(vehicle_location) < detection_radius:
                    tmp_near_vehicle_id.append(vehicle.id)
                    veh_future_bbs = []
                    traffic_transform = vehicle.get_transform()
                    traffic_control = vehicle.get_control()
                    traffic_velocity = vehicle.get_velocity()
                    traffic_speed = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity)
                    self.vehicle_speed_buffer[vehicle.id]['velocity'].append(traffic_speed)
                    self.vehicle_speed_buffer[vehicle.id]['throttle'].append(traffic_control.throttle)
                    self.vehicle_speed_buffer[vehicle.id]['brake'].append(traffic_control.brake)
                    if len(self.vehicle_speed_buffer[vehicle.id]['velocity']) > self.stuck_buffer_size:
                        self.vehicle_speed_buffer[vehicle.id]['velocity'] = self.vehicle_speed_buffer[vehicle.id]['velocity'][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[vehicle.id]['throttle'] = self.vehicle_speed_buffer[vehicle.id]['throttle'][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[vehicle.id]['brake'] = self.vehicle_speed_buffer[vehicle.id]['brake'][-self.stuck_buffer_size:]
                    next_loc = np.array([traffic_transform.location.x, traffic_transform.location.y])
                    action = np.array(np.stack([traffic_control.steer, traffic_control.throttle, traffic_control.brake], axis=-1))
                    next_yaw = np.array([traffic_transform.rotation.yaw / 180.0 * np.pi])
                    next_speed = np.array([traffic_speed])
                    for i in range(number_of_future_frames):
                        if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                            break
                        next_loc, next_yaw, next_speed = self.vehicle_model.forward(next_loc, next_yaw, next_speed, action)
                        delta_yaws = next_yaw.item() * 180.0 / np.pi
                        transform = carla.Transform(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z))
                        bounding_box = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
                        bounding_box.rotation = carla.Rotation(pitch=float(traffic_transform.rotation.pitch), yaw=float(delta_yaws), roll=float(traffic_transform.rotation.roll))
                        color = carla.Color(0, 0, 255, 255)
                        if self.visualize == 1:
                            self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
                        veh_future_bbs.append(bounding_box)
                    if statistics.mean(self.vehicle_speed_buffer[vehicle.id]['velocity']) < self.stuck_vel_threshold and statistics.mean(self.vehicle_speed_buffer[vehicle.id]['throttle']) > self.stuck_throttle_threshold and (statistics.mean(self.vehicle_speed_buffer[vehicle.id]['brake']) < self.stuck_brake_threshold):
                        tmp_stucked_vehicle_id.append(vehicle.id)
                    nearby_vehicles[vehicle.id] = veh_future_bbs
            to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
            for d in to_delete:
                del self.vehicle_speed_buffer[d]
            next_loc_no_brake = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
            next_yaw_no_brake = np.array([vehicle_transform.rotation.yaw / 180.0 * np.pi])
            next_speed_no_brake = np.array([speed])
            throttle_extrapolation = self._get_throttle_extrapolation(self.target_speed, speed)
            action_no_brake = np.array(np.stack([self.steer, throttle_extrapolation, 0.0], axis=-1))
            back_only_vehicle_id = []
            ego_future = []
            for i in range(number_of_future_frames):
                if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                    alpha = 255
                    color_value = 50
                    break
                else:
                    alpha = 50
                    color_value = 255
                next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake = self.ego_model.forward(next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake, action_no_brake)
                next_loc_no_brake_temp = np.array([-next_loc_no_brake[1], next_loc_no_brake[0]])
                next_yaw_no_brake_temp = next_yaw_no_brake.item() + np.pi / 2
                waypoint_route_extrapolation_temp = self._waypoint_planner_extrapolation.run_step(next_loc_no_brake_temp)
                steer_extrapolation_temp = self._get_steer_extrapolation(waypoint_route_extrapolation_temp, next_loc_no_brake_temp, next_yaw_no_brake_temp, next_speed_no_brake, restore=False)
                throttle_extrapolation_temp = self._get_throttle_extrapolation(self.target_speed, next_speed_no_brake, restore=False)
                brake_extrapolation_temp = 1.0 if self._waypoint_planner_extrapolation.is_last else 0.0
                action_no_brake = np.array(np.stack([steer_extrapolation_temp, float(throttle_extrapolation_temp), brake_extrapolation_temp], axis=-1))
                delta_yaws_no_brake = next_yaw_no_brake.item() * 180.0 / np.pi
                cosine = np.cos(next_yaw_no_brake.item())
                sine = np.sin(next_yaw_no_brake.item())
                extent = self._vehicle.bounding_box.extent
                extent_org = self._vehicle.bounding_box.extent
                extent.x = extent.x / 2.0
                transform = carla.Transform(carla.Location(x=next_loc_no_brake[0].item() + extent.x * cosine, y=next_loc_no_brake[1].item() + extent.y * sine, z=vehicle_transform.location.z))
                bounding_box = carla.BoundingBox(transform.location, extent)
                bounding_box.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws_no_brake), roll=float(vehicle_transform.rotation.roll))
                transform_back = carla.Transform(carla.Location(x=next_loc_no_brake[0].item() - extent.x * cosine, y=next_loc_no_brake[1].item() - extent.y * sine, z=vehicle_transform.location.z))
                bounding_box_back = carla.BoundingBox(transform_back.location, extent)
                bounding_box_back.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws_no_brake), roll=float(vehicle_transform.rotation.roll))
                color = carla.Color(0, color_value, 0, alpha)
                color2 = carla.Color(0, color_value, color_value, alpha)
                for id, traffic_participant in nearby_vehicles.items():
                    i_stuck = i
                    if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                        break
                    if id in tmp_stucked_vehicle_id:
                        i_stuck = 0
                    back_intersect = self.check_obb_intersection(bounding_box_back, traffic_participant[i_stuck]) == True
                    front_intersect = self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]) == True
                    if id in back_only_vehicle_id:
                        back_only_vehicle_id.remove(id)
                        if back_intersect:
                            back_only_vehicle_id.append(id)
                        continue
                    if back_intersect and (not front_intersect):
                        back_only_vehicle_id.append(id)
                    if front_intersect:
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction == True or i <= number_of_future_frames_no_junction:
                            vehicle_hazard = True
                        self.vehicle_hazard[i] = True
                for walker in nearby_walkers:
                    if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                        break
                    if self.check_obb_intersection(bounding_box, walker[i]) == True:
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction == True or i <= number_of_future_frames_no_junction:
                            walker_hazard = True
                        self.walker_hazard[i] = True
                if self.visualize == 1:
                    self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
                    self._world.debug.draw_box(box=bounding_box_back, rotation=bounding_box.rotation, thickness=0.1, color=color2, life_time=1.0 / self.frame_rate_sim)
            color = carla.Color(0, 255, 0, 255)
            bremsweg = (speed * 3.6 / 10.0) ** 2 / 2.0
            safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)
            center_safety_box = vehicle_transform.transform(carla.Location(x=safety_x, y=0.0, z=0.0))
            bounding_box = carla.BoundingBox(center_safety_box, self._vehicle.bounding_box.extent)
            bounding_box.rotation = vehicle_transform.rotation
            for _, traffic_participant in nearby_vehicles.items():
                if self.check_obb_intersection(bounding_box, traffic_participant[0]) == True:
                    color = carla.Color(255, 0, 0, 255)
                    vehicle_hazard = True
                    self.vehicle_hazard[0] = True
            for walker in nearby_walkers:
                if self.check_obb_intersection(bounding_box, walker[0]) == True:
                    color = carla.Color(255, 0, 0, 255)
                    walker_hazard = True
                    self.walker_hazard[0] = True
            if self.visualize == 1:
                self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
            self.future_states = {'walker': nearby_walkers, 'vehicle': nearby_vehicles}
        else:
            self.vehicle_hazard = vehicle_hazard
            self.walker_hazard = walker_hazard
        self.stop_sign_hazard = stop_sign_hazard
        self.traffic_light_hazard = light_hazard
        return vehicle_hazard or light_hazard or walker_hazard or stop_sign_hazard

    def _intersection_check(self, ego_wps):
        actors = self._world.get_actors()
        speed = self._get_forward_speed()
        vehicle_location = self._vehicle.get_location()
        vehicle_transform = self._vehicle.get_transform()
        vehicles = actors.filter('*vehicle*')
        vehicle_hazard = False
        self.vehicle_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        extrapolation_seconds = self.extrapolation_seconds
        detection_radius = self.detection_radius
        number_of_future_frames = int(extrapolation_seconds * self.frame_rate)
        number_of_future_frames_no_junction = int(self.extrapolation_seconds_no_junction * self.frame_rate)
        color = carla.Color(0, 255, 0, 255)
        walkers = actors.filter('*walker*')
        walker_hazard = False
        self.walker_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        nearby_walkers = []
        for walker in walkers:
            if walker.get_location().distance(vehicle_location) < detection_radius:
                walker_future_bbs = []
                walker_transform = walker.get_transform()
                walker_velocity = walker.get_velocity()
                walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)
                walker_location = walker_transform.location
                walker_direction = walker.get_control().direction
                for i in range(number_of_future_frames):
                    if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                        break
                    new_x = walker_location.x + walker_direction.x * walker_speed * (1.0 / self.frame_rate)
                    new_y = walker_location.y + walker_direction.y * walker_speed * (1.0 / self.frame_rate)
                    new_z = walker_location.z + walker_direction.z * walker_speed * (1.0 / self.frame_rate)
                    walker_location = carla.Location(new_x, new_y, new_z)
                    transform = carla.Transform(walker_location)
                    bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
                    bounding_box.rotation = carla.Rotation(pitch=walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch, yaw=walker.bounding_box.rotation.yaw + walker_transform.rotation.yaw, roll=walker.bounding_box.rotation.roll + walker_transform.rotation.roll)
                    color = carla.Color(0, 0, 255, 255)
                    if self.visualize == 1:
                        self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
                    walker_future_bbs.append(bounding_box)
                nearby_walkers.append(walker_future_bbs)
        nearby_vehicles = {}
        tmp_near_vehicle_id = []
        tmp_stucked_vehicle_id = []
        for vehicle in vehicles:
            if vehicle.id == self._vehicle.id:
                continue
            if vehicle.get_location().distance(vehicle_location) < detection_radius:
                tmp_near_vehicle_id.append(vehicle.id)
                veh_future_bbs = []
                traffic_transform = vehicle.get_transform()
                traffic_control = vehicle.get_control()
                traffic_velocity = vehicle.get_velocity()
                traffic_speed = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity)
                self.vehicle_speed_buffer[vehicle.id]['velocity'].append(traffic_speed)
                self.vehicle_speed_buffer[vehicle.id]['throttle'].append(traffic_control.throttle)
                self.vehicle_speed_buffer[vehicle.id]['brake'].append(traffic_control.brake)
                if len(self.vehicle_speed_buffer[vehicle.id]['velocity']) > self.stuck_buffer_size:
                    self.vehicle_speed_buffer[vehicle.id]['velocity'] = self.vehicle_speed_buffer[vehicle.id]['velocity'][-self.stuck_buffer_size:]
                    self.vehicle_speed_buffer[vehicle.id]['throttle'] = self.vehicle_speed_buffer[vehicle.id]['throttle'][-self.stuck_buffer_size:]
                    self.vehicle_speed_buffer[vehicle.id]['brake'] = self.vehicle_speed_buffer[vehicle.id]['brake'][-self.stuck_buffer_size:]
                next_loc = np.array([traffic_transform.location.x, traffic_transform.location.y])
                action = np.array(np.stack([traffic_control.steer, traffic_control.throttle, traffic_control.brake], axis=-1))
                next_yaw = np.array([traffic_transform.rotation.yaw / 180.0 * np.pi])
                next_speed = np.array([traffic_speed])
                for i in range(number_of_future_frames):
                    if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                        break
                    next_loc, next_yaw, next_speed = self.vehicle_model.forward(next_loc, next_yaw, next_speed, action)
                    delta_yaws = next_yaw.item() * 180.0 / np.pi
                    transform = carla.Transform(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z))
                    bounding_box = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
                    bounding_box.rotation = carla.Rotation(pitch=float(traffic_transform.rotation.pitch), yaw=float(delta_yaws), roll=float(traffic_transform.rotation.roll))
                    color = carla.Color(0, 0, 255, 255)
                    if self.visualize == 1:
                        self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
                    veh_future_bbs.append(bounding_box)
                if statistics.mean(self.vehicle_speed_buffer[vehicle.id]['velocity']) < self.stuck_vel_threshold and statistics.mean(self.vehicle_speed_buffer[vehicle.id]['throttle']) > self.stuck_throttle_threshold and (statistics.mean(self.vehicle_speed_buffer[vehicle.id]['brake']) < self.stuck_brake_threshold):
                    tmp_stucked_vehicle_id.append(vehicle.id)
                nearby_vehicles[vehicle.id] = veh_future_bbs
        to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
        for d in to_delete:
            del self.vehicle_speed_buffer[d]
        number_of_interpolation_frames = self.frame_rate // 2
        cur_loc = np.array([[vehicle_transform.location.x, vehicle_transform.location.y]])
        cur_loc_ego = np.array([[0, 0]])
        vehicl_yaw = (vehicle_transform.rotation.yaw + 90) * np.pi / 180
        rotation = np.array([[np.cos(vehicl_yaw), np.sin(vehicl_yaw)], [-np.sin(vehicl_yaw), np.cos(vehicl_yaw)]])
        future_loc = cur_loc + ego_wps[0] @ rotation
        all_locs = np.append(cur_loc, future_loc, axis=0)
        all_locs_ego = np.append(cur_loc_ego, ego_wps[0], axis=0)
        cur_yaw = np.array([vehicle_transform.rotation.yaw / 180.0 * np.pi])
        prev_yaw = cur_yaw
        back_only_vehicle_id = []
        for i in range(1, 1 + ego_wps.shape[1]):
            if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                alpha = 255
                color_value = 50
                break
            else:
                alpha = 50
                color_value = 255
            delta_yaw = math.atan2(all_locs_ego[i][0] - all_locs_ego[i - 1][0], all_locs_ego[i][1] - all_locs_ego[i - 1][1])
            next_yaw = cur_yaw - delta_yaw
            for k in range(number_of_interpolation_frames):
                tmp_loc = all_locs[i - 1] + (all_locs[i] - all_locs[i - 1]) / number_of_interpolation_frames * k
                tmp_yaw = prev_yaw + (next_yaw - prev_yaw) / number_of_interpolation_frames * k
                next_yaw_deg = tmp_yaw.item() * 180.0 / np.pi
                cosine = np.cos(tmp_yaw.item())
                sine = np.sin(tmp_yaw.item())
                extent = self._vehicle.bounding_box.extent
                extent.x = extent.x / 2.0
                transform = carla.Transform(carla.Location(x=tmp_loc[0].item() + extent.x * cosine, y=tmp_loc[1].item() + extent.y * sine, z=vehicle_transform.location.z))
                bounding_box = carla.BoundingBox(transform.location, extent)
                bounding_box.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(next_yaw_deg), roll=float(vehicle_transform.rotation.roll))
                transform_back = carla.Transform(carla.Location(x=tmp_loc[0].item() - extent.x * cosine, y=tmp_loc[1].item() - extent.y * sine, z=vehicle_transform.location.z))
                bounding_box_back = carla.BoundingBox(transform_back.location, extent)
                bounding_box_back.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(next_yaw_deg), roll=float(vehicle_transform.rotation.roll))
                color = carla.Color(0, color_value, 0, alpha)
                color2 = carla.Color(0, color_value, color_value, alpha)
                index = k + (i - 1) * number_of_interpolation_frames
                i_stuck = index
                for id, traffic_participant in nearby_vehicles.items():
                    if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                        break
                    if id in tmp_stucked_vehicle_id:
                        i_stuck = 0
                    back_intersect = self.check_obb_intersection(bounding_box_back, traffic_participant[i_stuck]) == True
                    front_intersect = self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]) == True
                    if id in back_only_vehicle_id:
                        back_only_vehicle_id.remove(id)
                        if back_intersect:
                            back_only_vehicle_id.append(id)
                        continue
                    if back_intersect and (not front_intersect):
                        back_only_vehicle_id.append(id)
                    if front_intersect:
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction == True or i <= number_of_future_frames_no_junction:
                            vehicle_hazard = True
                        self.vehicle_hazard[i] = True
                for walker in nearby_walkers:
                    if self.render_bev == False and self.junction == False and (i > number_of_future_frames_no_junction):
                        break
                    if self.check_obb_intersection(bounding_box, walker[i]) == True:
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction == True or i <= number_of_future_frames_no_junction:
                            walker_hazard = True
                        self.walker_hazard[i] = True
                if self.visualize == 1:
                    self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
                    self._world.debug.draw_box(box=bounding_box_back, rotation=bounding_box.rotation, thickness=0.1, color=color2, life_time=1.0 / self.frame_rate_sim)
                prev_yaw = next_yaw
        if self.visualize == 1:
            self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=1.0 / self.frame_rate_sim)
        return vehicle_hazard or walker_hazard

    def _save_waypoints(self):
        speed = self._get_forward_speed()
        number_of_future_frames = int(self.waypoint_seconds * self.frame_rate)
        vehicle_transform = self._vehicle.get_transform()
        next_loc = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
        next_yaw = np.array([vehicle_transform.rotation.yaw / 180.0 * np.pi])
        next_speed = np.array([speed])
        action = np.array(np.stack([self.steer, self.throttle, self.brake], axis=-1))
        ego_future = []
        for i in range(number_of_future_frames):
            next_loc, next_yaw, next_speed = self.ego_model.forward(next_loc, next_yaw, next_speed, action)
            next_loc_temp = np.array([-next_loc[1], next_loc[0]])
            next_yaw_temp = next_yaw.item() + np.pi / 2
            waypoint_route_temp = self._waypoint_planner.run_step(next_loc_temp)
            steer_temp = self._get_steer(self.brake, waypoint_route_temp, next_loc_temp, next_yaw_temp, next_speed, restore=False)
            throttle_temp = self._get_throttle(self.brake, self.target_speed, next_speed, restore=False)
            brake_temp = 1.0 if self._waypoint_planner.is_last else self.brake
            action = np.array(np.stack([steer_temp, float(throttle_temp), brake_temp], axis=-1))
            delta_yaws = next_yaw.item() * 180.0 / np.pi
            extent = self._vehicle.bounding_box.extent
            extent_org = self._vehicle.bounding_box.extent
            extent.x = extent.x / 2.0
            transform_whole = carla.Transform(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=vehicle_transform.location.z))
            bounding_box_whole = carla.BoundingBox(transform_whole.location, extent_org)
            bounding_box_whole.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws), roll=float(vehicle_transform.rotation.roll))
            ego_future.append(bounding_box_whole)
        self.future_states['ego'] = ego_future

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()
        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def _get_position(self, gps):
        gps = (gps - self._command_planner.mean) * self._command_planner.scale
        return gps

    def dot_product(self, vector1, vector2):
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

    def cross_product(self, vector1, vector2):
        return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y, y=vector1.z * vector2.x - vector1.x * vector2.z, z=vector1.x * vector2.y - vector1.y * vector2.x)

    def get_separating_plane(self, rPos, plane, obb1, obb2):
        """ Checks if there is a seperating plane
        rPos Vec3
        plane Vec3
        obb1  Bounding Box
        obb2 Bounding Box
        """
        return abs(self.dot_product(rPos, plane)) > abs(self.dot_product(obb1.rotation.get_forward_vector() * obb1.extent.x, plane)) + abs(self.dot_product(obb1.rotation.get_right_vector() * obb1.extent.y, plane)) + abs(self.dot_product(obb1.rotation.get_up_vector() * obb1.extent.z, plane)) + abs(self.dot_product(obb2.rotation.get_forward_vector() * obb2.extent.x, plane)) + abs(self.dot_product(obb2.rotation.get_right_vector() * obb2.extent.y, plane)) + abs(self.dot_product(obb2.rotation.get_up_vector() * obb2.extent.z, plane))

    def check_obb_intersection(self, obb1, obb2):
        RPos = obb2.location - obb1.location
        return not (self.get_separating_plane(RPos, obb1.rotation.get_forward_vector(), obb1, obb2) or self.get_separating_plane(RPos, obb1.rotation.get_right_vector(), obb1, obb2) or self.get_separating_plane(RPos, obb1.rotation.get_up_vector(), obb1, obb2) or self.get_separating_plane(RPos, obb2.rotation.get_forward_vector(), obb1, obb2) or self.get_separating_plane(RPos, obb2.rotation.get_right_vector(), obb1, obb2) or self.get_separating_plane(RPos, obb2.rotation.get_up_vector(), obb1, obb2) or self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_up_vector()), obb1, obb2))

    def _get_angle_to_slow(self, pos, theta, target):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle
        return angle

    def _get_angle_to(self, pos, theta, target):
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        diff = target - pos
        aim_0 = cos_theta * diff[0] + sin_theta * diff[1]
        aim_1 = -sin_theta * diff[0] + cos_theta * diff[1]
        angle = -math.degrees(math.atan2(-aim_1, aim_0))
        angle = np.float_(angle)
        return angle

    def get_nearby_object(self, vehicle_position, actor_list, radius):
        nearby_objects = []
        for actor in actor_list:
            trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
            if trigger_box_global_pos.distance(vehicle_position) < radius:
                nearby_objects.append(actor)
        return nearby_objects

