# Cluster 145

def vehicle_motion(map_data, all_current_vertices, placement_result=[], high_level_action_direction=[], high_level_action_speed=[], dt=0.4, total_len=10):
    if placement_result[0] is None:
        return (None, 'no placement')
    if high_level_action_direction == 'static':
        return np.array(placement_result[0:2])[None, ...].repeat(total_len, axis=0)
    slow_speed_threshold = (1.5, 2)
    fast_speed_threshold = (10, 25)
    random_speed_threshold = (3, 25)
    current_position = placement_result
    transformed_map_data = rot_and_trans(map_data, current_position)
    transformed_all_current_vertices = rot_and_trans_bbox(all_current_vertices, current_position)
    if high_level_action_speed == 'slow':
        v_init = random.uniform(slow_speed_threshold[0], slow_speed_threshold[1])
    elif high_level_action_speed == 'fast':
        v_init = random.randint(fast_speed_threshold[0], fast_speed_threshold[1])
    else:
        v_init = random.randint(random_speed_threshold[0], random_speed_threshold[1])
    transformed_map_data = filter_forward_lane(transformed_map_data)
    if high_level_action_direction == 'turn left':
        transformed_map_data_dest = filter_left_lane(transformed_map_data)
    elif high_level_action_direction == 'turn right':
        transformed_map_data_dest = filter_right_lane(transformed_map_data)
    if high_level_action_direction == 'turn left' or high_level_action_direction == 'turn right':
        destination_anchor = transformed_map_data_dest['centerline'][::5]
        print(destination_anchor)
        sorted_destination = destination_anchor[random.randint(0, len(destination_anchor) - 1)]
        sorted_destination_direction = sorted_destination[2:4] - sorted_destination[0:2]
        sorted_destination = sorted_destination[:2]
    elif high_level_action_direction == 'straight':
        sorted_destination_init = np.array([v_init * dt * total_len, 0])
        _, sorted_destination = find_closest_centerline(transformed_map_data, sorted_destination_init)
        sorted_destination_direction = sorted_destination[2:4] - sorted_destination[0:2]
        sorted_destination = (sorted_destination[0:2] + sorted_destination[2:4]) / 2
    start = np.array([0, 0])
    end = np.array([sorted_destination[0], sorted_destination[1]])
    Vs = np.array([v_init, 0])
    Ve = v_init * sorted_destination_direction / np.linalg.norm(sorted_destination_direction)
    Ve = np.abs(Ve)
    coordinates = hermite_spline_once(start, end, Vs, Ve)
    current_midpoint = coordinates[-int(len(coordinates) / 2)]
    midpoint_check_flag, closest_centerline = find_closest_centerline(transformed_map_data, current_midpoint)
    midpoint = (closest_centerline[0:2] + closest_centerline[2:4]) / 2
    midpoint_direction = closest_centerline[2:4] - closest_centerline[0:2]
    Vm = v_init * midpoint_direction / np.linalg.norm(midpoint_direction)
    Vm = np.abs(Vm)
    coordinates = hermite_spline_twice(start, end, midpoint, Vs, Ve, Vm)
    generated_trajectory = np.array(coordinates[::int(len(coordinates) / total_len)])
    generated_trajectory = check_collision_and_revise_static(generated_trajectory, transformed_all_current_vertices)
    generated_trajectory = inverse_rot_and_trans(generated_trajectory, current_position)
    return generated_trajectory

def transform_points_directly(points, source_vector, target_vector):
    R_source = rotation_matrix_from_vector(source_vector)
    R_target = rotation_matrix_from_vector(target_vector)
    R_direct = np.dot(R_target, np.linalg.inv(R_source))
    transformed_points = np.dot(points, R_direct.T)
    return transformed_points

def rotation_matrix_from_vector(v):
    angle = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    return R

def rot_and_trans_node(input_raw_map, current_pose):
    current_pose = np.array(current_pose)
    coordinate = current_pose[0:2]
    current_vec = current_pose[5:7] - current_pose[3:5]
    ego_vec = np.array([1.0, 0.0])
    output_centerline = []
    output_boundary = []
    centerline = input_raw_map['centerline']
    boundary = input_raw_map['boundary']
    for line in centerline:
        line[:, :2] -= coordinate[None, ...]
        line = transform_points_directly(line[:, :2], ego_vec, current_vec)
        output_centerline.append(line)
    for line in boundary:
        line[:, :2] -= coordinate[None, ...]
        line = transform_points_directly(line[:, :2], ego_vec, current_vec)
        output_boundary.append(line)
    output = {}
    output['centerline'] = output_centerline
    output['boundary'] = output_boundary
    return output

def rot_and_trans(input_map, current_pose):
    centerline = input_map['centerline'].copy()
    boundary = input_map['boundary'].copy()
    output = {}
    current_pose = np.array(current_pose)
    coordinate = current_pose[0:2]
    current_vec = current_pose[5:7] - current_pose[3:5]
    ego_vec = np.array([1.0, 0.0])
    centerline[:, 0:2] -= coordinate
    centerline[:, 2:4] -= coordinate
    boundary[:, 0:2] -= coordinate
    boundary[:, 2:4] -= coordinate
    centerline[:, 0:2] = transform_points_directly(centerline[:, 0:2], ego_vec, current_vec)
    centerline[:, 2:4] = transform_points_directly(centerline[:, 2:4], ego_vec, current_vec)
    boundary[:, 0:2] = transform_points_directly(boundary[:, 0:2], ego_vec, current_vec)
    boundary[:, 2:4] = transform_points_directly(boundary[:, 2:4], ego_vec, current_vec)
    output['centerline'] = centerline
    output['boundary'] = boundary
    return output

def rot_and_trans_bbox(input_bbox, current_pose):
    output_bbox = input_bbox.copy()
    if input_bbox.shape[0] == 0:
        return output_bbox
    current_pose = np.array(current_pose)
    coordinate = current_pose[0:2]
    current_vec = current_pose[5:7] - current_pose[3:5]
    ego_vec = np.array([1.0, 0.0])
    output_bbox = output_bbox.reshape((-1, 2))
    output_bbox[:, 0:2] -= coordinate
    output_bbox[:, 0:2] = transform_points_directly(output_bbox[:, 0:2], ego_vec, current_vec)
    output_bbox = output_bbox.reshape((-1, 4, 2))
    return output_bbox

def inverse_rot_and_trans(input, current_pose):
    current_pose = np.array(current_pose).copy()
    input = np.array(input)
    coordinate = current_pose[0:2]
    current_vec = current_pose[5:7] - current_pose[3:5]
    ego_vec = np.array([1.0, 0.0])
    output = transform_points_directly(input, current_vec, ego_vec)
    output += coordinate
    return output

def hermite_spline_once(P0, P1, T0, T1, num_points=100):
    spline_points = hermite_spline(P0, P1, T0, T1, num_points)
    return spline_points

def hermite_spline(P0, P1, T0, T1, num_points=100):
    t = np.linspace(0, 1, num_points)
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    spline_points = h00[:, None] * P0 + h10[:, None] * T0 + h01[:, None] * P1 + h11[:, None] * T1
    return spline_points

def hermite_spline_twice(P0, P1, PM, T0, T1, TM, num_points=100):
    spline_points_1 = hermite_spline(P0, PM, T0, TM, num_points)
    spline_points_2 = hermite_spline(PM, P1, TM, T1, num_points)
    spline_points = np.vstack((spline_points_1, spline_points_2))
    return spline_points

def hermite_spline_third(P0, P1, PM, PMM1, PMM2, T0, T1, TM, TMM1, TMM2, num_points=100, time=1, input_map=None, post_transform=(False, None), obj=None):
    spline_points_1 = hermite_spline(P0, PMM1, T0, TMM1, num_points)
    spline_points_2 = hermite_spline(PMM1, PM, TMM1, TM, num_points)
    spline_points_3 = hermite_spline(PM, PMM2, TM, TMM2, num_points)
    spline_points_4 = hermite_spline(PMM2, P1, TMM2, T1, num_points)
    spline_points = np.vstack((spline_points_1, spline_points_2, spline_points_3, spline_points_4))
    return spline_points

def filter_forward_lane(input_map):
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    forward_index = (centerline[:, 2:4] - centerline[:, 0:2])[:, 0] > 0
    filtered_centerline = centerline[forward_index, :]
    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output

def filter_left_lane(input_map, v=8):
    thres_min = 5
    thres_max = 30
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    right_index = (centerline[:, 1] > thres_min) & (centerline[:, 1] < thres_max) & (centerline[:, 3] - centerline[:, 1] >= 0)
    filtered_centerline = centerline[right_index, :]
    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output

def filter_right_lane(input_map, v=8):
    thres_min = -30
    thres_max = -5
    output = {}
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    right_index = (centerline[:, 1] > thres_min) & (centerline[:, 1] < thres_max) & (centerline[:, 3] - centerline[:, 1] <= 0)
    filtered_centerline = centerline[right_index, :]
    output['centerline'] = filtered_centerline
    output['boundary'] = boundary
    return output

def find_closest_centerline(transformed_map_data, current_destination):
    thres = 0.3
    current_destination = np.array(current_destination)
    centerlines = transformed_map_data['centerline']
    centernodes = (centerlines[:, 0:2] + centerlines[:, 2:4]) / 2
    distances = np.linalg.norm(current_destination[None] - centernodes, axis=-1, ord=2)
    closest_centerline_index = np.argmin(distances)
    if distances[closest_centerline_index] < thres:
        return (True, centerlines[closest_centerline_index])
    else:
        return (False, centerlines[closest_centerline_index])

