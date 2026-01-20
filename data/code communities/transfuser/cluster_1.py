# Cluster 1

def gen_scenarios(args, scenario_type_dict, town_scenario_tp_gen):
    if args.towns == 'all':
        towns = ALL_TOWNS
    else:
        towns = [args.towns]
    for town_ in towns:
        client = carla.Client('localhost', 2000)
        client.set_timeout(200.0)
        world = client.load_world(town_)
        carla_map = world.get_map()
        save_dir = args.save_dir
        for scen_type, _ in scenario_type_dict.items():
            town_scenario_tp_gen(town_, carla_map, scen_type, save_dir, world)

