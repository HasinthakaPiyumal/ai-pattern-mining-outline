# Cluster 29

def gen_skeleton_dict(towns_, scenarios_):

    def scenarios_list():
        scen_type_dict_lst = []
        for scenario_ in scenarios_:
            scen_type_dict = {}
            scen_type_dict['available_event_configurations'] = []
            scen_type_dict['scenario_type'] = scenario_
            scen_type_dict_lst.append(scen_type_dict)
        return scen_type_dict_lst
    skeleton = {'available_scenarios': []}
    for town_ in towns_:
        skeleton['available_scenarios'].append({town_: scenarios_list()})
    return skeleton

def scenarios_list():
    scen_type_dict_lst = []
    for scenario_ in scenarios_:
        scen_type_dict = {}
        scen_type_dict['available_event_configurations'] = []
        scen_type_dict['scenario_type'] = scenario_
        scen_type_dict_lst.append(scen_type_dict)
    return scen_type_dict_lst

def town_scenario_tp_gen(town_, carla_map, scen_type, save_dir, world):
    ego_triggers_789, other_triggers_789 = FUNC_SCENARIO_TYPE[scen_type](carla_map, world=world)
    for key_ in ego_triggers_789.keys():
        ego_triggers = ego_triggers_789[key_]
        other_triggers = other_triggers_789[key_]
        scen_type = key_
        town_x_scen_y_dict = gen_skeleton_dict([town_], [key_])
        trigger_point_dict_lst = []
        for ego_trig, oa_trig in zip(ego_triggers, other_triggers):
            trigger_pts_dict = {}
            if ego_trig:
                ego_trig_dict = {'pitch': ego_trig.rotation.pitch, 'yaw': ego_trig.rotation.yaw, 'x': ego_trig.location.x, 'y': ego_trig.location.y, 'z': ego_trig.location.z}
                trigger_pts_dict['transform'] = ego_trig_dict
            trigger_pts_dict['other_actors'] = oa_trig
            if oa_trig:
                oa_trig_dict = {'pitch': ego_trig.rotation.pitch, 'yaw': ego_trig.rotation.yaw, 'x': ego_trig.location.x, 'y': ego_trig.location.y, 'z': ego_trig.location.z}
                trigger_pts_dict['other_actors'] = oa_trig_dict
            trigger_point_dict_lst.append(trigger_pts_dict)
        for i, town_dict_ in enumerate(town_x_scen_y_dict['available_scenarios']):
            if town_ in town_dict_.keys():
                for j, scens_dict in enumerate(town_dict_[town_]):
                    if scens_dict['scenario_type'] == scen_type:
                        scens_dict['available_event_configurations'] = trigger_point_dict_lst
        print(f'Num trigger points for {town_} {scen_type}: {len(trigger_point_dict_lst)}')
        with open(os.path.join(save_dir[key_], f'{town_}_{scen_type}.json'), 'w') as f:
            json.dump(town_x_scen_y_dict, f, indent=2, sort_keys=True)

def town_scenario_tp_gen(town_, carla_map, scen_type, save_dir, world):
    scen_save_dir = os.path.join(save_dir, scen_type)
    if not os.path.exists(scen_save_dir):
        os.makedirs(scen_save_dir)
    ego_triggers, other_triggers = FUNC_SCENARIO_TYPE[scen_type](carla_map)
    town_x_scen_y_dict = []
    town_x_scen_y_dict = gen_skeleton_dict([town_], [scen_type]).copy()
    trigger_point_dict_lst = []
    for ego_trig, oa_trig in zip(ego_triggers, other_triggers):
        trigger_pts_dict = {}
        if ego_trig:
            ego_trig_dict = {'pitch': ego_trig.rotation.pitch, 'yaw': ego_trig.rotation.yaw, 'x': ego_trig.location.x, 'y': ego_trig.location.y, 'z': ego_trig.location.z}
            trigger_pts_dict['transform'] = ego_trig_dict
        trigger_pts_dict['other_actors'] = oa_trig
        if oa_trig:
            oa_trig_dict = {'pitch': ego_trig.rotation.pitch, 'yaw': ego_trig.rotation.yaw, 'x': ego_trig.location.x, 'y': ego_trig.location.y, 'z': ego_trig.location.z}
            trigger_pts_dict['other_actors'] = oa_trig_dict
        trigger_point_dict_lst.append(trigger_pts_dict)
    for i, town_dict_ in enumerate(town_x_scen_y_dict['available_scenarios']):
        if town_ in town_dict_.keys():
            for j, scens_dict in enumerate(town_dict_[town_]):
                if scens_dict['scenario_type'] == scen_type:
                    scens_dict['available_event_configurations'] = trigger_point_dict_lst
    print(f'Num trigger points for {town_} {scen_type}: {len(trigger_point_dict_lst)}')
    with open(os.path.join(scen_save_dir, f'{town_}_{scen_type}.json'), 'w') as f:
        json.dump(town_x_scen_y_dict, f, indent=2, sort_keys=True)

def town_scenario_tp_gen(town_, carla_map, scen_type, save_dir, world):
    scen_save_dir = os.path.join(save_dir, scen_type)
    if not os.path.exists(scen_save_dir):
        os.makedirs(scen_save_dir)
    ego_triggers, other_triggers = FUNC_SCENARIO_TYPE[scen_type](carla_map)
    town_x_scen_y_dict = gen_skeleton_dict([town_], [scen_type])
    trigger_point_dict_lst = []
    for ego_trig, oa_trig in zip(ego_triggers, other_triggers):
        trigger_pts_dict = {}
        if ego_trig:
            ego_trig_dict = {'pitch': ego_trig.rotation.pitch, 'yaw': ego_trig.rotation.yaw, 'x': ego_trig.location.x, 'y': ego_trig.location.y, 'z': ego_trig.location.z}
            trigger_pts_dict['transform'] = ego_trig_dict
        if oa_trig:
            oa_trig_dict = {'pitch': ego_trig.rotation.pitch, 'yaw': ego_trig.rotation.yaw, 'x': ego_trig.location.x, 'y': ego_trig.location.y, 'z': ego_trig.location.z}
            trigger_pts_dict['other_actors'] = oa_trig_dict
        trigger_point_dict_lst.append(trigger_pts_dict)
    for i, town_dict_ in enumerate(town_x_scen_y_dict['available_scenarios']):
        if town_ in town_dict_.keys():
            for j, scens_dict in enumerate(town_dict_[town_]):
                if scens_dict['scenario_type'] == scen_type:
                    scens_dict['available_event_configurations'] = trigger_point_dict_lst
    print(f'Num trigger points for {town_} {scen_type}: {len(trigger_point_dict_lst)}')
    with open(os.path.join(scen_save_dir, f'{town_}_{scen_type}.json'), 'w') as f:
        json.dump(town_x_scen_y_dict, f, indent=2, sort_keys=True)

def town_scenario_tp_gen(town_, carla_map, scen_type, save_dir, world):
    scen_save_dir = os.path.join(save_dir, scen_type)
    if not os.path.exists(scen_save_dir):
        os.makedirs(scen_save_dir)
    ego_triggers, other_triggers = FUNC_SCENARIO_TYPE[scen_type](carla_map, world)
    town_x_scen_y_dict = gen_skeleton_dict([town_], [scen_type])
    trigger_point_dict_lst = []
    for ego_trig, oa_trig in zip(ego_triggers, other_triggers):
        trigger_pts_dict = {}
        if ego_trig:
            ego_trig_dict = {'pitch': ego_trig.rotation.pitch, 'yaw': ego_trig.rotation.yaw, 'x': ego_trig.location.x, 'y': ego_trig.location.y, 'z': ego_trig.location.z}
            trigger_pts_dict['transform'] = ego_trig_dict
        trigger_pts_dict['other_actors'] = oa_trig
        if oa_trig:
            oa_trig_dict = {'pitch': ego_trig.rotation.pitch, 'yaw': ego_trig.rotation.yaw, 'x': ego_trig.location.x, 'y': ego_trig.location.y, 'z': ego_trig.location.z}
            trigger_pts_dict['other_actors'] = oa_trig_dict
        trigger_point_dict_lst.append(trigger_pts_dict)
    for i, town_dict_ in enumerate(town_x_scen_y_dict['available_scenarios']):
        if town_ in town_dict_.keys():
            for j, scens_dict in enumerate(town_dict_[town_]):
                if scens_dict['scenario_type'] == scen_type:
                    scens_dict['available_event_configurations'] = trigger_point_dict_lst
    print(f'Num trigger points for {town_} {scen_type}: {len(trigger_point_dict_lst)}')
    with open(os.path.join(scen_save_dir, f'{town_}_{scen_type}.json'), 'w') as f:
        json.dump(town_x_scen_y_dict, f, indent=2, sort_keys=True)

