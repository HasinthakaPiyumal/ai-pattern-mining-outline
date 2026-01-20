# Cluster 143

def generate_vertices(car):
    """Generates the vertices of a 3D box."""
    x = car['cx']
    y = car['cy']
    z = car['cz']
    length = car['length']
    width = car['width']
    height = car['height']
    heading = car['heading']
    box_center = np.array([x, y, z])
    half_dims = np.array([length / 2, width / 2, height / 2])
    relative_positions = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) * half_dims
    vertices = np.asarray([rotate(pos, heading) + box_center for pos in relative_positions])
    return vertices

def rotate(point, angle):
    """Rotates a point around the origin by the specified angle in radians."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return np.dot(rotation_matrix, point)

class LearnableSpatialTransformWrapper(nn.Module):

    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple((self.transform(elem) for elem in x))
            y_trans = self.impl(x_trans)
            return tuple((self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x)))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = (int(height * self.pad_coef), int(width * self.pad_coef))
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = (int(height * self.pad_coef), int(width * self.pad_coef))
        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h:y_height - pad_h, pad_w:y_width - pad_w]
        return y

class LearnableSpatialTransformWrapper(nn.Module):

    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple((self.transform(elem) for elem in x))
            y_trans = self.impl(x_trans)
            return tuple((self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x)))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = (int(height * self.pad_coef), int(width * self.pad_coef))
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = (int(height * self.pad_coef), int(width * self.pad_coef))
        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h:y_height - pad_h, pad_w:y_width - pad_w]
        return y

class MotionAgent:

    def __init__(self, config):
        self.config = config
        self.motion_tracking = config.get('motion_tracking', False)

    def llm_reasoning_dependency(self, scene, message):
        """ LLM reasoning of Motion Agent, determine if the vehicle placement is depend on scene elements.
        Input:
            scene : Scene
                scene object.
            message : str
                language prompt to ChatSim.
        """
        try:
            q0 = 'I will provide an operation statement to add a vehicle, and you need to determine whether the position of the added car has any spatial dependency with other cars in my statement'
            q1 = "Only return a JSON format dictionary as your response, which contains a key 'dependency'. If the added car's position depends on other objects, set it to 1; otherwise, set it to 0."
            q2 = "An Example: Given statement 'add an Audi in the back which drives ahead', you should return {'dependency': 0}. This is because I only mention the added Audi."
            q3 = "An Example: Given statement 'add a Porsche at 2m to the right of the red Audi.', you should return {'dependency': 1}. This is because Porsche's position depends on Audi."
            q4 = "An Example: Given statement 'add a car in front of me.', you should return {'dependency': 0}. This is because 'me' is not other car in the scene."
            q5 = 'The statement is:' + message
            prompt_list = [q0, q1, q2, q3, q4, q5]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to extract information from the operations.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Motion Agent LLM] analyzing insertion scene dependency ', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            placement_mode = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {placement_mode} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Motion Agent LLM] reasoning object dependency fails.'
        return placement_mode

    def llm_placement_wo_dependency(self, scene, message):
        try:
            q0 = 'I will provide you with an operation statement to add and place a vehicle, and I need you to extract 3 specific placement information from the statement, including: '
            q1 = " (1) 'mode', one of ['front', 'left front', 'left', 'right front', 'right', 'random'], representing approximate initial positions of the vehicle. If not specified, it defaults to 'random'."
            q2 = " (2) 'distance_constraint' indicates whether there's a constraint on the distance of the added vehicle. 0 means no constraint, 1 means there is a constraint." + " If there's no relevant information mentioned, it defaults to 0."
            q3 = " (3) 'distance_min_max' represents the range of constraints when 'distance_constraint' applicable. It should be a tuple in the format (min, max), for example, (9, 11) means the minimum distance is 9, and the maximum is 11." + " When there's 'distance_constraint' is 0, the default value is (4, 45). If distance is specified as a specific value 'x', 'distance_min_max' is (x, x+5)"
            q4 = "Just return the json dict with keys:'mode', 'distance_constraint', 'distance_min_max'. Do not return any code or discription."
            q5 = "An Example: Given operation statement: 'Add an Audi 7-10 meters ahead', you should return " + "{'mode':'front', 'distance_constraint': 1, 'distance_min_max':(7,10)}"
            q6 = "An Example: Given operation statement: 'Add an Porsche in the right front.', you should return " + "{'mode':'right front', 'distance_constraint': 0, 'distance_min_max':(4, 45)}"
            q7 = 'Note that you should not return any code or explanations, only provide a JSON dictionary.'
            q8 = 'The operation statement:' + message
            prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7, q8]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to determine how to place a car.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Motion Agent LLM] deciding scene-independent object placement', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            placement_prior = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {placement_prior} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Motion Agent LLM] deciding placement fails.'
        return placement_prior

    def llm_placement_w_dependency(self, scene, message, scene_object_description):
        try:
            q0 = 'I will provide you with an operation statement to add and place a vehicle, as well as information of other cars in the scene.'
            q1 = 'I need you to determine a specific position (x, y) for placement of the added car in my statement. '
            q2 = 'Information of other cars in the scene is a two-level dictionary, with the first level representing the different car id in the scene, ' + 'and the second level containing various information about that car, including the (x, y) of its world 3D coordinate, ' + 'its image coordinate (u, v) in an image frame, depth, and rgb color representation.'
            q3 = 'The dictionary is' + str(scene_object_description)
            q4 = 'I will also further inform you about the operations that have been previously performed on this scene. ' + 'You can use these past operations, along with the dictionary I provide, to generate the final position.'
            q5 = 'The previously performed operation is : ' + str(scene.past_operations)
            q6 = "If the car with key 'direction', and direction is close, 'behind' means keep the same 'y' and increase 'x' 10 meters. If direction is away, 'behind' means keep the same 'y' and decrease 'x' 10 meters." + "If the car with key 'direction', and direction is close, 'front' means keep the same 'y' and decrease 'x' 10 meters. If direction is away, 'front' means keep the same 'y' and increase 'x' 10 meters."
            q7 = "'left' means keep the same 'x' and increase 'y' 5m, 'right' means keep the same 'x' and decrease 'y' 5m."
            q8 = "You should return a placemenet positon in JSON dictionary with 2 keys: 'x', 'y'. Do not provide any code or explanations, only return the final JSON dictionary."
            q9 = 'The requirement is:' + message
            prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to determine how to place a car.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Motion Agent LLM] deciding scene-dependent object placement', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            placement_prior = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {placement_prior} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Motion Agent LLM] deciding placement fails.'
        return placement_prior

    def llm_motion_planning(self, scene, message):
        try:
            q0 = 'I will provide you with an operation statement to add and place a vehicle, and I need you to determine the its motion situation from my statement, including: '
            q1 = "(1) 'action', one of ['static', 'random', 'straight', 'turn left', 'turn right']. If action not mentioned in the statement, it defaults to 'straight'." + "For example, the statement is 'add a black car in front of me', then the action is 'straight'."
            q2 = "(2) 'speed', the approximate speed of the vehicle, one of ['random', 'fast', 'slow']. If speed is not mentioned in the statement, it defaults to 'slow'."
            q3 = "(3) 'direction', one of ['away', 'close', 'random']. 'away' represents the direction away from oneself, and 'close' represents the direction toward oneself." + "For example, moving forward is 'away' from oneself, while moving towards oneself is 'close'. If direction is not mentioned in the statement, just return 'random'."
            q4 = "(4) 'wrong_way', if the vehicle drives in the wrong way, one of ['true'. 'false']. If the information is not mentioned in the statement, it defaults to 'false'."
            q4 = "An Example: Given the statement 'add a Tesla that is racing straight ahead in the right front of the scene', you should return {'action': 'straight', 'speed': 'fast', 'direction': 'away', 'wrong_way': 'false'}"
            q5 = "An Example: Given the statement 'add a yellow Audi in front of the scene', you should return {'action': 'static', 'speed': 'random', 'direction': 'away', 'wrong_way': 'false'}"
            q6 = "An Example: Given the statement 'add a Tesla coming from the front and driving in the wrong way', you should return {'action': 'straight', 'speed': 'random', 'direction': 'close', 'wrong_way': 'true'}"
            q7 = 'Note that there is no need to return any code or explanations; only provide a JSON dictionary. Do not include any additional statements.'
            q8 = 'The operation statement is:' + message
            prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7, q8]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to assess the motion situation for adding vehicles.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Motion Agent LLM] finding motion prior', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            motion_prior = eval(answer)
            if not motion_prior.get('wrong_way'):
                motion_prior['wrong_way'] = False
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {motion_prior} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Motion Agent LLM] finding motion prior fails.'
        return motion_prior

    def func_placement_and_motion_single_vehicle(self, scene, added_car_name):
        added_car_id = added_car_name.lstrip('added_car_')
        transformed_map_data_ = transform_node_to_lane(scene.map_data)
        all_current_vertices_coord = scene.all_current_vertices_coord
        for added_traj in scene.all_trajectories:
            all_current_vertices_coord = np.vstack([all_current_vertices_coord, added_traj[0:1, 0:2]])
        one_added_car = scene.added_cars_dict[added_car_name]
        if one_added_car['need_placement_and_motion'] is True:
            scene.added_cars_dict[added_car_name]['need_placement_and_motion'] = False
            one_added_car = scene.added_cars_dict[added_car_name]
            transformed_map_data = deepcopy(transformed_map_data_)
            if one_added_car['wrong_way'] is True:
                transformed_map_data['centerline'][:, -1] = (transformed_map_data['centerline'][:, -1] + 1) % 2
                transformed_map_data['centerline'] = np.concatenate((transformed_map_data['centerline'][:, 2:4], transformed_map_data['centerline'][:, 0:2], transformed_map_data['centerline'][:, 4:]), axis=1)
                transformed_map_data['centerline'] = np.flip(transformed_map_data['centerline'], axis=0)
            if one_added_car.get('x') is None:
                placement_result = vehicle_placement(transformed_map_data, all_current_vertices_coord, one_added_car['direction'] if one_added_car['direction'] != 'random' else random.choice(['away', 'close']), one_added_car['mode'], one_added_car['distance_constraint'], one_added_car['distance_min_max'], 'default')
            else:
                placement_result = vehicle_placement_specific(transformed_map_data, all_current_vertices_coord, np.array([one_added_car['x'], one_added_car['y']]))
            if placement_result[0] is None:
                del scene.added_cars_dict[added_car_name]
                return
            one_added_car['placement_result'] = placement_result
            try:
                motion_result = vehicle_motion(transformed_map_data, scene.all_current_vertices[:, ::2, :2] if scene.all_current_vertices.shape[0] != 0 else scene.all_current_vertices, placement_result=one_added_car['placement_result'], high_level_action_direction=one_added_car['action'], high_level_action_speed=one_added_car['speed'], dt=1 / scene.fps, total_len=scene.frames)
            except ValueError as e:
                print(f'{colored('[Motion Agent] Error: Potentially no feasible destination can be found.', color='red', attrs=['bold'])} {e}')
                raise ValueError('No feasible destination can be found.')
            if motion_result[0] is None:
                del scene.added_cars_dict[added_car_name]
                return
            one_added_car['motion'] = motion_result
            scene.added_cars_dict[added_car_name] = one_added_car
            all_trajectories = []
            for one_car_name in scene.added_cars_dict.keys():
                all_trajectories.append(scene.added_cars_dict[one_car_name]['motion'][:, :2])
            all_trajectories_after_check_collision = check_collision_and_revise_dynamic(all_trajectories)
            all_trajectories_after_check_collision = all_trajectories
            scene.all_trajectories = all_trajectories_after_check_collision
            for idx, one_car_name in enumerate(scene.added_cars_dict.keys()):
                motion_result = all_trajectories_after_check_collision[idx]
                placement_result = scene.added_cars_dict[one_car_name]['placement_result']
                direction = np.zeros((motion_result.shape[0], 1))
                angle = np.arctan2(placement_result[-1] - placement_result[-3], placement_result[-2] - placement_result[-4])
                for i in range(motion_result.shape[0] - 1):
                    if motion_result[i, 0] == motion_result[i + 1, 0] and motion_result[i, 1] == motion_result[i + 1, 1]:
                        direction[i, 0] = angle
                    else:
                        direction[i, 0] = np.arctan2(motion_result[i + 1, 1] - motion_result[i, 1], motion_result[i + 1, 0] - motion_result[i, 0])
                direction[-1, 0] = direction[-2, 0]
                motion_result = np.concatenate((motion_result, direction), axis=1)
                if self.motion_tracking:
                    try:
                        from simulator import TrajectoryTracker
                    except ModuleNotFoundError:
                        error_msg1 = f'{colored('[ERROR]', color='red', attrs=['bold'])} Trajectory Tracking Module is not installed.\n'
                        error_msg2 = "\nYou can 1) Install Installation README's Step 5: Setup Trajectory Tracking Module"
                        error_msg3 = "\n     Or 2) set ['motion_agent']['motion_tracking'] to False in config.\n"
                        raise ModuleNotFoundError(error_msg1 + error_msg2 + error_msg3)
                    reference_line = interpolate_uniformly(motion_result, int(scene.frames * scene.fps / 10))
                    reference_line = [(reference_line[i, 0], reference_line[i, 1]) for i in range(reference_line.shape[0])]
                    init_state = (motion_result[0, 0], motion_result[0, 1], motion_result[0, 2], np.linalg.norm(np.array(reference_line[1]) - np.array(reference_line[0])) * 10)
                    pretrained_checkpoint_dir = './chatsim/foreground/drl-based-trajectory-tracking/submodules/drltt-assets/checkpoints/track/checkpoint'
                    trajectory_tracker = TrajectoryTracker(checkpoint_dir=pretrained_checkpoint_dir)
                    states, actions = trajectory_tracker.track_reference_line(reference_line=reference_line, init_state=init_state)
                    motion_result = np.stack(states)[:, :-1]
                    motion_result = interpolate_uniformly(motion_result, scene.frames)
                scene.added_cars_dict[one_car_name]['motion'] = motion_result

def transform_node_to_lane(input_map, pre_transform=True):
    output_lane_map = {}
    edge_lanes = []
    for edge in input_map['boundary']:
        N = edge.shape[0]
        edge_lane = np.zeros((N, 6))
        edge_lane[:, :2] = edge[:, :2]
        edge_lane[:-1, 2:4] = edge[1:, :2]
        edge_lane[:, -2] = 0
        edge_lane = edge_lane[:-1]
        edge_lanes.append(edge_lane)
    centerline_lanes = []
    for i, centerline in enumerate(input_map['centerline']):
        if pre_transform:
            centerline = centerline[centerline[:, 0] > 0]
        N = centerline.shape[0]
        if N > 0:
            centerline_lane = np.zeros((N, 6))
            centerline_lane[:, :2] = centerline[:, :2]
            centerline_lane[:-1, 2:4] = centerline[1:, :2]
            centerline_lane[:, -2] = 1
            if np.linalg.norm(centerline[-1]) - np.linalg.norm(centerline[0]) > 0:
                centerline_lane[:, -1] = 1
            else:
                centerline_lane[:, -1] = 0
            centerline_lane = centerline_lane[:-1]
            centerline_lanes.append(centerline_lane)
    output_lane_map['boundary'] = np.concatenate(edge_lanes, axis=0)
    output_lane_map['centerline'] = np.concatenate(centerline_lanes, axis=0)
    if pre_transform:
        output_lane_map = crop_map(output_lane_map)
    return output_lane_map

def vehicle_placement(input_map, current_vertices, direction, vehicle_mode, distance_constraint, distance_min_max, vehicle_size):
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    plt.cla()
    for i in range(len(centerline)):
        lane_vec = centerline[i]
        plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='green', linewidth=1)
        plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black', s=1)
    for i in range(len(boundary)):
        lane_vec = boundary[i]
        plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='red', linewidth=1)
        plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black', s=1)
    valid_lane_list = []
    ego_index = 0
    ego_dist = 999
    for i in range(centerline.shape[0]):
        valid_lane_list.append(i)
        center_coord = (centerline[i, 0:2] + centerline[i, 2:4]) / 2
        if np.linalg.norm(center_coord, ord=2) < ego_dist:
            ego_index = i
            ego_dist = np.linalg.norm(center_coord, ord=2)
    ego_lane_vec = centerline[ego_index]
    input_map = centerline
    vehicle_size_x = 2
    vehicle_size_y = 4.5
    distance_min_default = 4
    distance_max_default = 45
    front_placement_distance_threshold = 8
    left_front_placement_distance_threshold = (1.5, 10)
    left_front_placement_theta_threshold = (3, 60)
    right_front_placement_distance_threshold = (1.5, 10)
    right_front_placement_theta_threshold = (300, 357)
    left_placement_distance_threshold = (1.5, 10)
    left_placement_theta_threshold = (75, 105)
    right_placement_distance_threshold = (1.5, 10)
    right_placement_theta_threshold = (255, 285)
    mode = vehicle_mode
    if distance_constraint:
        distance_min = float(distance_min_max[0]) + 4
        distance_max = float(distance_min_max[1]) + 4
    l, w = (vehicle_size_y / 2, vehicle_size_x / 2)
    if mode == 'random':
        while True:
            cur_valid_lane_index_list = []
            for i in range(len(valid_lane_list)):
                center_coord = (input_map[valid_lane_list[i], 0:2] + input_map[valid_lane_list[i], 2:4]) / 2
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) >= distance_min and np.linalg.norm(center_coord, ord=2) <= distance_max:
                    if direction == 'close' and input_map[valid_lane_list[i], -1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i], -1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0, len(cur_valid_lane_index_list) - 1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                return (None, 'No place to put cars')
                index = -1
                break
            del valid_lane_list[random_lane_index]
            if conflict_check(centerline, index, current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break
    elif mode == 'front':
        while True:
            cur_valid_lane_index_list = []
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2] + cur_lane_vec[2:4]) / 2
                dist_to_lane_vec = abs(center_coord[1])
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and (dist_to_lane_vec < front_placement_distance_threshold) and (center_coord[0] > 0):
                    if direction == 'close' and input_map[valid_lane_list[i], -1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i], -1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0, len(cur_valid_lane_index_list) - 1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                return (None, 'No place to put front cars')
                index = -1
                break
            del valid_lane_list[random_lane_index]
            if conflict_check(centerline, index, current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break
    elif mode == 'left front':
        while True:
            cur_valid_lane_index_list = []
            ego_lane_vec_heading = np.array([1.0, 0.0])
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2] + cur_lane_vec[2:4]) / 2
                dist_to_lane_vec = abs(center_coord[1])
                cur_lane_vec_heading = center_coord - ego_lane_vec[0:2]
                theta = get_angle_from_line_to_line(ego_lane_vec_heading, cur_lane_vec_heading)
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and (dist_to_lane_vec >= left_front_placement_distance_threshold[0]) and (dist_to_lane_vec <= left_front_placement_distance_threshold[1]) and (theta >= left_front_placement_theta_threshold[0]) and (theta <= left_front_placement_theta_threshold[1]):
                    if direction == 'close' and input_map[valid_lane_list[i], -1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i], -1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0, len(cur_valid_lane_index_list) - 1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                index = -1
                return (None, 'No place to put left front cars')
                break
            del valid_lane_list[random_lane_index]
            if conflict_check(centerline, index, current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break
    elif mode == 'right front':
        while True:
            cur_valid_lane_index_list = []
            ego_lane_vec_heading = ego_lane_vec_heading = np.array([1.0, 0.0])
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2] + cur_lane_vec[2:4]) / 2
                dist_to_lane_vec = abs(center_coord[1])
                cur_lane_vec_heading = center_coord - ego_lane_vec[0:2]
                theta = get_angle_from_line_to_line(ego_lane_vec_heading, cur_lane_vec_heading)
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and (dist_to_lane_vec >= right_front_placement_distance_threshold[0]) and (dist_to_lane_vec <= right_front_placement_distance_threshold[1]) and (theta >= right_front_placement_theta_threshold[0]) and (theta <= right_front_placement_theta_threshold[1]):
                    if direction == 'close' and input_map[valid_lane_list[i], -1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i], -1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0, len(cur_valid_lane_index_list) - 1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                index = -1
                return (None, 'No place to put right front cars')
                break
            del valid_lane_list[random_lane_index]
            if conflict_check(centerline, index, current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break
    elif mode == 'left':
        while True:
            cur_valid_lane_index_list = []
            ego_lane_vec_heading = ego_lane_vec_heading = np.array([1.0, 0.0])
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2] + cur_lane_vec[2:4]) / 2
                dist_to_lane_vec = abs(center_coord[1])
                cur_lane_vec_heading = center_coord - ego_lane_vec[0:2]
                theta = get_angle_from_line_to_line(ego_lane_vec_heading, cur_lane_vec_heading)
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and (dist_to_lane_vec >= left_placement_distance_threshold[0]) and (dist_to_lane_vec <= left_placement_distance_threshold[1]) and (theta > left_placement_theta_threshold[0]) and (theta <= left_placement_theta_threshold[1]):
                    if direction == 'close' and input_map[valid_lane_list[i], -1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i], -1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0, len(cur_valid_lane_index_list) - 1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                index = -1
                return (None, 'No place to put cars on the left')
                break
            del valid_lane_list[random_lane_index]
            if conflict_check(centerline, index, current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break
    elif mode == 'right':
        while True:
            cur_valid_lane_index_list = []
            ego_lane_vec_heading = ego_lane_vec_heading = np.array([1.0, 0.0])
            for i in range(len(valid_lane_list)):
                cur_lane_vec = input_map[valid_lane_list[i]]
                center_coord = (cur_lane_vec[0:2] + cur_lane_vec[2:4]) / 2
                dist_to_lane_vec = abs(center_coord[1])
                cur_lane_vec_heading = center_coord - ego_lane_vec[0:2]
                theta = get_angle_from_line_to_line(ego_lane_vec_heading, cur_lane_vec_heading)
                if not distance_constraint:
                    distance_min = distance_min_default
                    distance_max = distance_max_default
                if np.linalg.norm(center_coord, ord=2) <= distance_max and np.linalg.norm(center_coord, ord=2) >= distance_min and (dist_to_lane_vec >= right_placement_distance_threshold[0]) and (dist_to_lane_vec <= right_placement_distance_threshold[1]) and (theta > right_placement_theta_threshold[0]) and (theta < right_placement_theta_threshold[1]):
                    if direction == 'close' and input_map[valid_lane_list[i], -1] == 0:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'away' and input_map[valid_lane_list[i], -1] == 1:
                        cur_valid_lane_index_list.append(valid_lane_list[i])
                    elif direction == 'random':
                        cur_valid_lane_index_list.append(valid_lane_list[i])
            try:
                random_lane_index = random.randint(0, len(cur_valid_lane_index_list) - 1)
                index = cur_valid_lane_index_list[random_lane_index]
            except:
                index = -1
                return (None, 'No place to put cars on the right')
                break
            del valid_lane_list[random_lane_index]
            if conflict_check(centerline, index, current_vertices):
                break
            if len(valid_lane_list) <= 0:
                print('exceed the maximum number of vehicle')
                break
    if index < 0:
        return (None, 'No place to put cars')
    lane_vec = input_map[index]
    xs, ys, xe, ye = (lane_vec[0], lane_vec[1], lane_vec[2], lane_vec[3])
    xc, yc = ((xs + xe) / 2, (ys + ye) / 2)
    theta = np.arctan2(xe - xs, ye - ys)
    x1, y1 = (xc - w * np.cos(theta) + l * np.sin(theta), yc + l * np.cos(theta) + w * np.sin(theta))
    x2, y2 = (xc + w * np.cos(theta) + l * np.sin(theta), yc + l * np.cos(theta) - w * np.sin(theta))
    x3, y3 = (xc + w * np.cos(theta) - l * np.sin(theta), yc - l * np.cos(theta) - w * np.sin(theta))
    x4, y4 = (xc - w * np.cos(theta) - l * np.sin(theta), yc - l * np.cos(theta) + w * np.sin(theta))
    return (xc, yc, theta, xs, ys, xe, ye)

def vehicle_placement_specific(input_map, current_vertices, coord):
    centerline = input_map['centerline']
    boundary = input_map['boundary']
    plt.cla()
    print(coord)
    print(centerline[:, 0:2])
    for i in range(len(centerline)):
        lane_vec = centerline[i]
        plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='green', linewidth=1)
        plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black', s=1)
    for i in range(len(boundary)):
        lane_vec = boundary[i]
        plt.plot([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='red', linewidth=1)
        plt.scatter([lane_vec[0], lane_vec[2]], [lane_vec[1], lane_vec[3]], color='black', s=1)
    center_coord = (centerline[:, 0:2] + centerline[:, 2:4]) / 2
    distance = np.linalg.norm(coord[None] - center_coord, ord=2, axis=-1)
    closest_distance_index = np.argmin(distance)
    lane_vec = centerline[closest_distance_index]
    x, y = (coord[0], coord[1])
    xs, ys, xe, ye = (lane_vec[0], lane_vec[1], lane_vec[2], lane_vec[3])
    theta = np.arctan2(xe - xs, ye - ys)
    return (x, y, theta, xs, ys, xe, ye)

def check_collision_and_revise_dynamic(input_trajectory):

    def judge_priority(traj1, traj2):
        T = traj1.shape[0]
        traj2_new = traj2[0:1].repeat(T, axis=0)
        for t in range(T):
            if is_rectangles_overlap(traj1[t], traj2_new[t]):
                return 1
        return 2

    def interpolate_uniformly(track, num_points):
        """
        Interpolates a given track to a specified number of points, distributing them uniformly.

        :param track: A numpy array of shape (n, d) where n is the number of points and d is the dimension.
        :param num_points: The number of points in the output interpolated track.
        :return: A numpy array of shape (num_points, d) representing the uniformly interpolated track.
        """
        distances = np.cumsum(np.sqrt(np.sum(np.diff(track, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)
        max_distance = distances[-1]
        uniform_distances = np.linspace(0, max_distance, num_points)
        uniform_track = np.array([np.interp(uniform_distances, distances, track[:, dim]) for dim in range(track.shape[1])])
        return uniform_track.T

    def add_wait_timesteps(traj, t, wait_timesteps):
        traj_out = traj.copy()
        T = traj_out.shape[0]
        if t + wait_timesteps > T:
            traj_out = interpolate_uniformly(traj[:t], T)
        else:
            traj_out[:t + wait_timesteps] = interpolate_uniformly(traj[:t], t + wait_timesteps)
            traj_out[t + wait_timesteps:] = traj[t:-wait_timesteps]
        return traj_out
    valid_traj = []
    valid_record = []
    for item in input_trajectory:
        if item[0] is not None:
            valid_record.append(1)
            valid_traj.append(item)
        else:
            valid_record.append(0)
    curr_trajectory = np.array(valid_traj)
    car_length = 5
    car_width = 2.2
    safe_distance = 8
    N, T = (curr_trajectory.shape[0], curr_trajectory.shape[1])
    all_corners_trajectory = np.zeros((N, T, 4, 2))
    for n in range(N):
        all_corners_trajectory[n] = calculate_car_corners(curr_trajectory[n], car_length, car_width)
    revised_trajectory = curr_trajectory.copy()
    for i in range(N):
        for j in range(i + 1, N):
            for t in range(1, T):
                if is_rectangles_overlap(all_corners_trajectory[i, t], all_corners_trajectory[j, t]):
                    modify_idx = judge_priority(all_corners_trajectory[i], all_corners_trajectory[j])
                    collision_point = (curr_trajectory[i, t] + curr_trajectory[j, t]) / 2
                    if modify_idx == 1:
                        collision_speed = np.linalg.norm(curr_trajectory[j, t] - curr_trajectory[j, t - 1])
                        wait_timesteps = int(np.ceil(safe_distance / (collision_speed + 0.0001)))
                        curr_trajectory[i] = add_wait_timesteps(curr_trajectory[i], t, wait_timesteps)
                    if modify_idx == 2:
                        collision_speed = np.linalg.norm(curr_trajectory[i, t] - curr_trajectory[i, t - 1])
                        wait_timesteps = int(np.ceil(safe_distance / (collision_speed + 0.0001)))
                        curr_trajectory[j] = add_wait_timesteps(curr_trajectory[j], t, wait_timesteps)
                    break
    output = []
    num = 0
    for i in range(len(valid_record)):
        if valid_record[i] == 1:
            output.append(curr_trajectory[num])
            num += 1
        else:
            output.append(input_trajectory[i])
    return output

def interpolate_uniformly(track, num_points):
    """
    Interpolates a given track to a specified number of points, distributing them uniformly.

    :param track: A numpy array of shape (n, d) where n is the number of points and d is the dimension.
    :param num_points: The number of points in the output interpolated track.
    :return: A numpy array of shape (num_points, d) representing the uniformly interpolated track.
    """
    distances = np.cumsum(np.sqrt(np.sum(np.diff(track, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)
    max_distance = distances[-1]
    uniform_distances = np.linspace(0, max_distance, num_points)
    uniform_track = np.array([np.interp(uniform_distances, distances, track[:, dim]) for dim in range(track.shape[1])])
    return uniform_track.T

def is_rectangles_overlap(rect1, rect2):
    for i in range(4):
        edge = rect1[i] - rect1[(i + 1) % 4]
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)
        proj1 = project_polygon_onto_axis(Polygon(rect1), axis)
        proj2 = project_polygon_onto_axis(Polygon(rect2), axis)
        if not is_projection_overlap(proj1, proj2):
            return False
    for i in range(4):
        edge = rect2[i] - rect2[(i + 1) % 4]
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)
        proj1 = project_polygon_onto_axis(Polygon(rect1), axis)
        proj2 = project_polygon_onto_axis(Polygon(rect2), axis)
        if not is_projection_overlap(proj1, proj2):
            return False
    return True

def is_projection_overlap(proj1, proj2):
    return max(proj1[0], proj2[0]) <= min(proj1[1], proj2[1])

def calculate_translation_vector(rect1, rect2, direction):
    axes = []
    for rect in [rect1, rect2]:
        for i in range(len(rect.exterior.coords) - 1):
            edge = np.subtract(rect.exterior.coords[i + 1], rect.exterior.coords[i])
            normal = [-edge[1], edge[0]]
            axes.append(normal / np.linalg.norm(normal))
    min_translation_vector = None
    min_translation_distance = float('inf')
    for axis in axes:
        min_proj_rect1, max_proj_rect1 = project_polygon_onto_axis(rect1, axis)
        min_proj_rect2, max_proj_rect2 = project_polygon_onto_axis(rect2, axis)
        if max_proj_rect1 < min_proj_rect2 or max_proj_rect2 < min_proj_rect1:
            return [0, 0]
        overlap = min(max_proj_rect1, max_proj_rect2) - max(min_proj_rect1, min_proj_rect2)
        translation_axis = np.multiply(axis, overlap)
        if np.dot(translation_axis, direction) < 0:
            translation_axis = np.multiply(translation_axis, -1)
        translation_distance = np.linalg.norm(translation_axis)
        if translation_distance < min_translation_distance:
            min_translation_distance = translation_distance
            min_translation_vector = translation_axis
    return min_translation_vector

def project_polygon_onto_axis(polygon, axis):
    min_projection = float('inf')
    max_projection = float('-inf')
    for point in polygon.exterior.coords:
        projection = (point[0] * axis[0] + point[1] * axis[1]) / np.linalg.norm(axis)
        min_projection = min(min_projection, projection)
        max_projection = max(max_projection, projection)
    return (min_projection, max_projection)

def check_collision_and_revise_static(curr_trajectory, objects):
    N, T = (objects.shape[0], curr_trajectory.shape[0])
    if N == 0:
        return curr_trajectory
    car_length = 5
    car_width = 2.2
    safe_distance = 7
    obj_corners_trajectory = objects[:, None, :, :].repeat(T, axis=1)
    curr_corner_trajectory = calculate_car_corners(curr_trajectory, car_length, car_width)
    for j in range(N):
        for t in range(T):
            if is_rectangles_overlap(curr_corner_trajectory[t], obj_corners_trajectory[j, t]):
                direction = curr_trajectory[t] - curr_trajectory[t - 1]
                direction /= np.linalg.norm(direction)
                perpendicular = np.array([-direction[1], direction[0]])
                delta = calculate_translation_vector(Polygon(curr_corner_trajectory[t]), Polygon(obj_corners_trajectory[j, t]), perpendicular)
                curr_trajectory[t] += delta
    return curr_trajectory

def calculate_car_corners(trajectory, car_length=4.5, car_width=2):
    T = trajectory.shape[0]
    corners_trajectory = np.zeros((T, 4, 2))
    for i in range(1, T):
        direction = trajectory[i] - trajectory[i - 1]
        if np.linalg.norm(direction) != 0:
            direction /= np.linalg.norm(direction)
        else:
            direction = np.array([1, 0])
        perpendicular = np.array([-direction[1], direction[0]])
        front = 0.5 * car_length * direction
        back = -0.5 * car_length * direction
        left = 0.5 * car_width * perpendicular
        right = -0.5 * car_width * perpendicular
        corners_trajectory[i, 0] = trajectory[i] + front + left
        corners_trajectory[i, 1] = trajectory[i] + front + right
        corners_trajectory[i, 2] = trajectory[i] + back + right
        corners_trajectory[i, 3] = trajectory[i] + back + left
    corners_trajectory[0] = corners_trajectory[1]
    return corners_trajectory

def judge_priority(traj1, traj2):
    T = traj1.shape[0]
    traj2_new = traj2[0:1].repeat(T, axis=0)
    for t in range(T):
        if is_rectangles_overlap(traj1[t], traj2_new[t]):
            return 1
    return 2

def add_wait_timesteps(traj, t, wait_timesteps):
    traj_out = traj.copy()
    T = traj_out.shape[0]
    if t + wait_timesteps > T:
        traj_out = interpolate_uniformly(traj[:t], T)
    else:
        traj_out[:t + wait_timesteps] = interpolate_uniformly(traj[:t], t + wait_timesteps)
        traj_out[t + wait_timesteps:] = traj[t:-wait_timesteps]
    return traj_out

def crop_map(input_map):
    center_point = (input_map['boundary'][:, :2] + input_map['boundary'][:, 2:4]) / 2
    mask = center_point[:, 0] > 0
    input_map['boundary'] = input_map['boundary'][mask]
    center_point = (input_map['centerline'][:, :2] + input_map['centerline'][:, 2:4]) / 2
    mask = center_point[:, 0] > 0
    input_map['centerline'] = input_map['centerline'][mask]
    return input_map

def generate_vertices(car):
    """Generates the vertices of a 3D box."""
    x = car['cx']
    y = car['cy']
    z = car['cz']
    length = car['length']
    width = car['width']
    height = car['height']
    heading = car['heading']
    box_center = np.array([x, y, z])
    half_dims = np.array([length / 2, width / 2, height / 2])
    relative_positions = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) * half_dims
    vertices = np.asarray([rotate(pos, heading) + box_center for pos in relative_positions])
    return vertices

def is_rectangles_overlap(rect1, rect2):
    for i in range(4):
        edge = rect1[i] - rect1[(i + 1) % 4]
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)
        proj1 = project_polygon_onto_axis(rect1, axis)
        proj2 = project_polygon_onto_axis(rect2, axis)
        if not is_projection_overlap(proj1, proj2):
            return False
    for i in range(4):
        edge = rect2[i] - rect2[(i + 1) % 4]
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)
        proj1 = project_polygon_onto_axis(rect1, axis)
        proj2 = project_polygon_onto_axis(rect2, axis)
        if not is_projection_overlap(proj1, proj2):
            return False
    return True

def check_collision_and_revise_waste(all_trajectory):
    all_trajectory = np.array(all_trajectory)
    N, T = (all_trajectory.shape[0], all_trajectory.shape[1])
    car_length = 4.5
    car_width = 2.0
    safe_distance = 7
    all_corners_trajectory = np.zeros((N, T, 4, 2))
    for n in range(N):
        all_corners_trajectory[n] = calculate_car_corners(all_trajectory[n], car_length, car_width)
    for j in range(1, N):
        for t in range(T):
            if is_rectangles_overlap(all_corners_trajectory[0, t], all_corners_trajectory[j, t]):
                trajectory1 = all_trajectory[0]
                trajectory2 = all_trajectory[j]
                speed1 = np.linalg.norm(np.diff(trajectory1, axis=0))
                speed2 = np.linalg.norm(np.diff(trajectory2, axis=0))
                if is_tailgating(trajectory1, trajectory2) and speed1[t - 1] < speed2[t - 1]:
                    all_trajectory[0] = accerlate(all_trajectory[0], calculate_speed_increase(trajectory1, trajectory2))
                    break
                else:
                    collision_point = trajectory1[t]
                    for t_safe in range(t + 1, T):
                        if np.linalg.norm(trajectory2[t_safe] - collision_point) > safe_distance:
                            break
                    if t == T - 1:
                        all_trajectory[0] = deaccerlate_to_zero(all_trajectory[0])
                    else:
                        all_trajectory[0] = deaccerlate(all_trajectory[0], t / t_safe)
    return all_trajectory

def is_tailgating(trajectory1, trajectory2):
    threshold = 0.2
    speed1 = np.diff(trajectory1, axis=0)
    speed2 = np.diff(trajectory2, axis=0)
    direction = trajectory2[t] - trajectory1[t]
    speed_direction1 = speed1[t - 1] / np.linalg.norm(speed1[t - 1])
    speed_direction2 = speed2[t - 1] / np.linalg.norm(speed2[t - 1])
    angle1 = np.arccos(np.clip(np.dot(direction, speed_direction1), -1.0, 1.0))
    angle2 = np.arccos(np.clip(np.dot(-direction, speed_direction2), -1.0, 1.0))
    if angle1 < threshold and angle2 < threshold:
        return True
    else:
        return False

def accerlate(trajectory, speed_increase=1.1):
    speeds = np.diff(trajectory, axis=0)
    speeds *= speed_increase
    new_trajectory = np.cumsum(np.vstack([trajectory[0], speeds]), axis=0)
    return new_trajectory

def calculate_speed_increase(front_car_traj, rear_car_traj, safe_distance=7):
    distances = np.linalg.norm(rear_car_traj - front_car_traj, axis=1)
    front_car_speeds = np.linalg.norm(np.diff(front_car_traj, axis=0), axis=1)
    rear_car_speeds = np.linalg.norm(np.diff(rear_car_traj, axis=0), axis=1)
    relative_speeds = rear_car_speeds - front_car_speeds
    time_to_collision = (distances[1:] - safe_distance) / relative_speeds
    min_time_to_collision = np.min(time_to_collision[relative_speeds > 0])
    if np.any(distances > safe_distance) or min_time_to_collision > 0:
        return 1.0
    speed_increase = 1 + (safe_distance - distances[1:]) / (relative_speeds * min_time_to_collision)
    return np.max(speed_increase)

def deaccerlate(trajectory, speed_decrease=1.1):
    speeds = np.diff(trajectory, axis=0)
    speeds *= speed_decrease
    new_trajectory = np.cumsum(np.vstack([trajectory[0], speeds]), axis=0)
    return new_trajectory

def check_collision_and_revise(all_trajectory):
    all_trajectory = np.array(all_trajectory)
    N, T = (all_trajectory.shape[0], all_trajectory.shape[1])
    car_length = 4.5
    car_width = 2.0
    safe_distance = 7
    all_corners_trajectory = np.zeros((N, T, 4, 2))
    for n in range(N):
        all_corners_trajectory[n] = calculate_car_corners(all_trajectory[n], car_length, car_width)
    for j in range(1, N):
        for t in range(T):
            if is_rectangles_overlap(all_corners_trajectory[0, t], all_corners_trajectory[j, t]):
                trajectory1 = all_trajectory[0]
                trajectory2 = all_trajectory[j]
                speed1 = np.linalg.norm(np.diff(trajectory1, axis=0))
                speed2 = np.linalg.norm(np.diff(trajectory2, axis=0))
                if is_tailgating(trajectory1, trajectory2) and speed1[t - 1] < speed2[t - 1]:
                    return 'accerlate'
                else:
                    return 'deaccerlate'
    return 'no revise'

def generate_vertices(car):
    """Generates the vertices of a 3D box."""
    x = car['cx']
    y = car['cy']
    z = car['cz']
    length = car['length']
    width = car['width']
    height = car['height']
    heading = car['heading']
    box_center = np.array([x, y, z])
    half_dims = np.array([length / 2, width / 2, height / 2])
    relative_positions = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) * half_dims
    vertices = np.asarray([rotate(pos, heading) + box_center for pos in relative_positions])
    return vertices

def conflict_check(centerline, index, current_vertices):
    thres = 4
    for i in range(current_vertices.shape[0]):
        point1 = (centerline[index][0:2] + centerline[index][2:4]) / 2
        point2 = current_vertices[i][:2]
        if np.sqrt(np.sum((np.array(point2) - np.array(point1)) ** 2)) < thres:
            return False
    return True

def get_angle_from_line_to_line(ego_lane_vec_heading, cur_lane_vec_heading):
    cosangle = ego_lane_vec_heading.dot(cur_lane_vec_heading) / (np.linalg.norm(ego_lane_vec_heading) * np.linalg.norm(cur_lane_vec_heading))
    angle = np.arccos(cosangle) * 180 / np.pi
    a1 = np.array([*ego_lane_vec_heading, 0])
    a2 = np.array([*cur_lane_vec_heading, 0])
    a3 = np.cross(a1, a2)
    if np.sign(a3[2]) < 0:
        angle = 360 - angle
    return angle

