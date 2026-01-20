# Cluster 21

class RouteScenario(BasicScenario):
    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """
    category = 'RouteScenario'

    def __init__(self, world, config, debug_mode=0, criteria_enable=True):
        """
        Setup all relevant parameters and create scenarios along route
        """
        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None
        self._update_route(world, config, debug_mode > 0)
        ego_vehicle = self._update_ego_vehicle()
        self.list_scenarios = self._build_scenario_instances(world, ego_vehicle, self.sampled_scenarios_definitions, scenarios_per_tick=10, timeout=self.timeout, debug_mode=debug_mode > 1)
        super(RouteScenario, self).__init__(name=config.name, ego_vehicles=[ego_vehicle], config=config, world=world, debug_mode=debug_mode > 1, terminate_on_failure=False, criteria_enable=criteria_enable)

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)
        gps_route, route = interpolate_trajectory(world, config.trajectory)
        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(config.town, route, world_annotations)
        self.route = route
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
        config.agent.set_global_plan(gps_route, self.route)
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)
        self.timeout = self._estimate_route_timeout()
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5
        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', elevate_transform, rolename='hero')
        spectator = CarlaDataProvider.get_world().get_spectator()
        ego_trans = ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        return ego_vehicle

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0
        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length + INITIAL_SECONDS_DELAY)

    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)
            size = 0.2
            if w[1] == RoadOption.LEFT:
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:
                color = carla.Color(128, 128, 128)
            else:
                color = carla.Color(0, 255, 0)
                size = 0.1
            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)
        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2, color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2, color=carla.Color(255, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """
        rgn = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True
            return False

        def select_scenario(list_scenarios):
            higher_id = -1
            selected_scenario = None
            for scenario in list_scenarios:
                try:
                    scenario_number = int(scenario['name'].split('Scenario')[1])
                except:
                    scenario_number = -1
                if scenario_number >= higher_id:
                    higher_id = scenario_number
                    selected_scenario = scenario
            return selected_scenario
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]
            scenario_choice = select_scenario(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rgn.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]
            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)
        return sampled_scenarios

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions, scenarios_per_tick=5, timeout=300, debug_mode=False):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []
        if debug_mode:
            for scenario in scenario_definitions:
                loc = carla.Location(scenario['trigger_position']['x'], scenario['trigger_position']['y'], scenario['trigger_position']['z']) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False, color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)
        for scenario_number, definition in enumerate(scenario_definitions):
            scenario_class = NUMBER_CLASS_TRANSLATION[definition['name']]
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017', ego_vehicle.get_transform(), 'hero')]
            route_var_name = 'ScenarioRouteNumber{}'.format(scenario_number)
            scenario_configuration.route_var_name = route_var_name
            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration, criteria_enable=False, timeout=timeout)
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()
            except Exception as e:
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue
            scenario_instance_vec.append(scenario_instance)
        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))
            return sublist_of_actors
        list_of_actors = []
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])
        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])
        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])
        return list_of_actors

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        if int(os.environ.get('DATAGEN')) == 1:
            town_amount = {'Town01': 130, 'Town02': 60, 'Town03': 135, 'Town04': 190, 'Town05': 120, 'Town06': 155, 'Town07': 60, 'Town08': 180, 'Town09': 300, 'Town10HD': 80}
            amount = town_amount[config.town] if config.town in town_amount else 0
            amount = random.randint(amount, 2 * amount)
        else:
            amount = 500
        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*', amount, carla.Transform(), autopilot=True, random_location=True, rolename='background')
        if new_actors is None:
            raise Exception('Error: Unable to add the background activity, all spawn points were occupied')
        for _actor in new_actors:
            self.other_actors.append(_actor)
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        scenario_trigger_distance = 1.5
        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        subbehavior = py_trees.composites.Parallel(name='Behavior', policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        scenario_behaviors = []
        blackboard_list = []
        for i, scenario in enumerate(self.list_scenarios):
            if scenario.scenario.behavior is not None:
                route_var_name = scenario.config.route_var_name
                if route_var_name is not None:
                    scenario_behaviors.append(scenario.scenario.behavior)
                    blackboard_list.append([scenario.config.route_var_name, scenario.config.trigger_points[0].location])
                else:
                    name = '{} - {}'.format(i, scenario.scenario.behavior.name)
                    oneshot_idiom = oneshot_behavior(name=name, variable_name=name, behaviour=scenario.scenario.behavior)
                    scenario_behaviors.append(oneshot_idiom)
        scenario_triggerer = ScenarioTriggerer(self.ego_vehicles[0], self.route, blackboard_list, scenario_trigger_distance, repeat_scenarios=False)
        subbehavior.add_child(scenario_triggerer)
        subbehavior.add_children(scenario_behaviors)
        subbehavior.add_child(Idle())
        behavior.add_child(subbehavior)
        return behavior

    def _create_test_criteria(self):
        """
        """
        criteria = []
        route = convert_transform_to_location(self.route)
        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=False)
        route_criterion = InRouteTest(self.ego_vehicles[0], route=route, offroad_max=30, terminate_on_failure=True)
        completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route)
        outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], route=route)
        red_light_criterion = RunningRedLightTest(self.ego_vehicles[0])
        stop_criterion = RunningStopTest(self.ego_vehicles[0])
        blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[0], speed_threshold=0.1, below_threshold_max_time=180.0, terminate_on_failure=True, name='AgentBlockedTest')
        criteria.append(completion_criterion)
        criteria.append(outsidelane_criterion)
        criteria.append(collision_criterion)
        criteria.append(red_light_criterion)
        criteria.append(stop_criterion)
        criteria.append(route_criterion)
        criteria.append(blocked_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

def oneshot_behavior(name, variable_name, behaviour):
    """
    This is taken from py_trees.idiom.oneshot.
    """
    blackboard = py_trees.blackboard.Blackboard()
    _ = blackboard.set(variable_name, False)
    subtree_root = py_trees.composites.Selector(name=name)
    check_flag = py_trees.blackboard.CheckBlackboardVariable(name=variable_name + ' Done?', variable_name=variable_name, expected_value=True, clearing_policy=py_trees.common.ClearingPolicy.ON_INITIALISE)
    set_flag = py_trees.blackboard.SetBlackboardVariable(name='Mark Done', variable_name=variable_name, variable_value=True)
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(set_flag)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name='OneShot')
        sequence.add_children([behaviour, set_flag])
    subtree_root.add_children([check_flag, sequence])
    return subtree_root

class RouteScenario(BasicScenario):
    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """
    category = 'RouteScenario'

    def __init__(self, world, config, debug_mode=0, criteria_enable=True):
        """
        Setup all relevant parameters and create scenarios along route
        """
        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None
        self._update_route(world, config, debug_mode > 0)
        ego_vehicle = self._update_ego_vehicle()
        self.list_scenarios = self._build_scenario_instances(world, ego_vehicle, self.sampled_scenarios_definitions, scenarios_per_tick=10, timeout=self.timeout, debug_mode=debug_mode > 1)
        super(RouteScenario, self).__init__(name=config.name, ego_vehicles=[ego_vehicle], config=config, world=world, debug_mode=debug_mode > 1, terminate_on_failure=False, criteria_enable=criteria_enable)

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)
        gps_route, route = interpolate_trajectory(world, config.trajectory)
        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(config.town, route, world_annotations)
        self.route = route
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
        config.agent.set_global_plan(gps_route, self.route)
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)
        self.timeout = self._estimate_route_timeout()
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5
        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', elevate_transform, rolename='hero')
        spectator = CarlaDataProvider.get_world().get_spectator()
        ego_trans = ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        return ego_vehicle

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0
        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length + INITIAL_SECONDS_DELAY)

    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)
            size = 0.2
            if w[1] == RoadOption.LEFT:
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:
                color = carla.Color(128, 128, 128)
            else:
                color = carla.Color(0, 255, 0)
                size = 0.1
            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)
        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2, color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2, color=carla.Color(255, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """
        rgn = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True
            return False

        def select_scenario(list_scenarios):
            higher_id = -1
            selected_scenario = None
            for scenario in list_scenarios:
                try:
                    scenario_number = int(scenario['name'].split('Scenario')[1])
                except:
                    scenario_number = -1
                if scenario_number >= higher_id:
                    higher_id = scenario_number
                    selected_scenario = scenario
            return selected_scenario
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]
            scenario_choice = select_scenario(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rgn.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]
            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)
        return sampled_scenarios

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions, scenarios_per_tick=5, timeout=300, debug_mode=False):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []
        if debug_mode:
            for scenario in scenario_definitions:
                loc = carla.Location(scenario['trigger_position']['x'], scenario['trigger_position']['y'], scenario['trigger_position']['z']) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False, color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)
        for scenario_number, definition in enumerate(scenario_definitions):
            scenario_class = NUMBER_CLASS_TRANSLATION[definition['name']]
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017', ego_vehicle.get_transform(), 'hero')]
            route_var_name = 'ScenarioRouteNumber{}'.format(scenario_number)
            scenario_configuration.route_var_name = route_var_name
            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration, criteria_enable=False, timeout=timeout)
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()
            except Exception as e:
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue
            scenario_instance_vec.append(scenario_instance)
        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))
            return sublist_of_actors
        list_of_actors = []
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])
        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])
        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])
        return list_of_actors

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        town_amount = {'Town01': 120, 'Town02': 100, 'Town03': 120, 'Town04': 200, 'Town05': 120, 'Town06': 150, 'Town07': 110, 'Town08': 180, 'Town09': 300, 'Town10HD': 120}
        amount = town_amount[config.town] if config.town in town_amount else 0
        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*', amount, carla.Transform(), autopilot=True, random_location=True, rolename='background')
        if new_actors is None:
            raise Exception('Error: Unable to add the background activity, all spawn points were occupied')
        for _actor in new_actors:
            self.other_actors.append(_actor)
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        scenario_trigger_distance = 1.5
        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        subbehavior = py_trees.composites.Parallel(name='Behavior', policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        scenario_behaviors = []
        blackboard_list = []
        for i, scenario in enumerate(self.list_scenarios):
            if scenario.scenario.behavior is not None:
                route_var_name = scenario.config.route_var_name
                if route_var_name is not None:
                    scenario_behaviors.append(scenario.scenario.behavior)
                    blackboard_list.append([scenario.config.route_var_name, scenario.config.trigger_points[0].location])
                else:
                    name = '{} - {}'.format(i, scenario.scenario.behavior.name)
                    oneshot_idiom = oneshot_behavior(name=name, variable_name=name, behaviour=scenario.scenario.behavior)
                    scenario_behaviors.append(oneshot_idiom)
        scenario_triggerer = ScenarioTriggerer(self.ego_vehicles[0], self.route, blackboard_list, scenario_trigger_distance, repeat_scenarios=False)
        subbehavior.add_child(scenario_triggerer)
        subbehavior.add_children(scenario_behaviors)
        subbehavior.add_child(Idle())
        behavior.add_child(subbehavior)
        return behavior

    def _create_test_criteria(self):
        """
        """
        criteria = []
        route = convert_transform_to_location(self.route)
        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=False)
        route_criterion = InRouteTest(self.ego_vehicles[0], route=route, offroad_max=30, terminate_on_failure=True)
        completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route)
        outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], route=route)
        red_light_criterion = RunningRedLightTest(self.ego_vehicles[0])
        stop_criterion = RunningStopTest(self.ego_vehicles[0])
        blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[0], speed_threshold=0.1, below_threshold_max_time=180.0, terminate_on_failure=True, name='AgentBlockedTest')
        criteria.append(completion_criterion)
        criteria.append(outsidelane_criterion)
        criteria.append(collision_criterion)
        criteria.append(red_light_criterion)
        criteria.append(stop_criterion)
        criteria.append(route_criterion)
        criteria.append(blocked_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

class OpenScenario(BasicScenario):
    """
    Implementation of the OpenSCENARIO scenario
    """

    def __init__(self, world, ego_vehicles, config, config_file, debug_mode=False, criteria_enable=True, timeout=300):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self.route = None
        self.config_file = config_file
        self.timeout = timeout
        super(OpenScenario, self).__init__('OpenScenario', ego_vehicles=ego_vehicles, config=config, world=world, debug_mode=debug_mode, terminate_on_failure=False, criteria_enable=criteria_enable)

    def _initialize_environment(self, world):
        """
        Initialization of weather and road friction.
        """
        pass

    def _create_environment_behavior(self):
        env_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name='EnvironmentBehavior')
        weather_update = ChangeWeather(OpenScenarioParser.get_weather_from_env_action(self.config.init, self.config.catalogs))
        road_friction = ChangeRoadFriction(OpenScenarioParser.get_friction_from_env_action(self.config.init, self.config.catalogs))
        env_behavior.add_child(oneshot_behavior(variable_name='InitialWeather', behaviour=weather_update))
        env_behavior.add_child(oneshot_behavior(variable_name='InitRoadFriction', behaviour=road_friction))
        return env_behavior

    def _create_init_behavior(self):
        init_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name='InitBehaviour')
        for actor in self.config.other_actors + self.config.ego_vehicles:
            for carla_actor in self.other_actors + self.ego_vehicles:
                if 'role_name' in carla_actor.attributes and carla_actor.attributes['role_name'] == actor.rolename:
                    actor_init_behavior = py_trees.composites.Sequence(name='InitActor{}'.format(actor.rolename))
                    controller_atomic = None
                    for private in self.config.init.iter('Private'):
                        if private.attrib.get('entityRef', None) == actor.rolename:
                            for private_action in private.iter('PrivateAction'):
                                for controller_action in private_action.iter('ControllerAction'):
                                    module, args = OpenScenarioParser.get_controller(controller_action, self.config.catalogs)
                                    controller_atomic = ChangeActorControl(carla_actor, control_py_module=module, args=args)
                    if controller_atomic is None:
                        controller_atomic = ChangeActorControl(carla_actor, control_py_module=None, args={})
                    actor_init_behavior.add_child(controller_atomic)
                    if actor.speed > 0:
                        actor_init_behavior.add_child(ChangeActorTargetSpeed(carla_actor, actor.speed, init_speed=True))
                    init_behavior.add_child(actor_init_behavior)
                    break
        return init_behavior

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        story_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name='Story')
        joint_actor_list = self.other_actors + self.ego_vehicles + [None]
        for act in self.config.story.iter('Act'):
            act_sequence = py_trees.composites.Sequence(name='Act StartConditions and behaviours')
            start_conditions = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name='StartConditions Group')
            parallel_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name='Maneuver + EndConditions Group')
            parallel_sequences = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name='Maneuvers')
            for sequence in act.iter('ManeuverGroup'):
                sequence_behavior = py_trees.composites.Sequence(name=sequence.attrib.get('name'))
                repetitions = sequence.attrib.get('maximumExecutionCount', 1)
                for _ in range(int(repetitions)):
                    actor_ids = []
                    for actor in sequence.iter('Actors'):
                        for entity in actor.iter('EntityRef'):
                            entity_name = entity.attrib.get('entityRef', None)
                            for k, _ in enumerate(joint_actor_list):
                                if joint_actor_list[k] and entity_name == joint_actor_list[k].attributes['role_name']:
                                    actor_ids.append(k)
                                    break
                    if not actor_ids:
                        print('Warning: Maneuvergroup {} does not use reference actors!'.format(sequence.attrib.get('name')))
                        actor_ids.append(len(joint_actor_list) - 1)
                    catalog_maneuver_list = []
                    for catalog_reference in sequence.iter('CatalogReference'):
                        catalog_maneuver = OpenScenarioParser.get_catalog_entry(self.config.catalogs, catalog_reference)
                        catalog_maneuver_list.append(catalog_maneuver)
                    all_maneuvers = itertools.chain(iter(catalog_maneuver_list), sequence.iter('Maneuver'))
                    single_sequence_iteration = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name=sequence_behavior.name)
                    for maneuver in all_maneuvers:
                        maneuver_parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name='Maneuver ' + maneuver.attrib.get('name'))
                        for event in maneuver.iter('Event'):
                            event_sequence = py_trees.composites.Sequence(name='Event ' + event.attrib.get('name'))
                            parallel_actions = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name='Actions')
                            for child in event.iter():
                                if child.tag == 'Action':
                                    for actor_id in actor_ids:
                                        maneuver_behavior = OpenScenarioParser.convert_maneuver_to_atomic(child, joint_actor_list[actor_id], self.config.catalogs)
                                        maneuver_behavior = StoryElementStatusToBlackboard(maneuver_behavior, 'ACTION', child.attrib.get('name'))
                                        parallel_actions.add_child(oneshot_behavior(variable_name=get_xml_path(self.config.story, sequence) + '>' + get_xml_path(maneuver, child), behaviour=maneuver_behavior))
                                if child.tag == 'StartTrigger':
                                    parallel_condition_groups = self._create_condition_container(child, 'Parallel Condition Groups', sequence, maneuver)
                                    event_sequence.add_child(parallel_condition_groups)
                            parallel_actions = StoryElementStatusToBlackboard(parallel_actions, 'EVENT', event.attrib.get('name'))
                            event_sequence.add_child(parallel_actions)
                            maneuver_parallel.add_child(oneshot_behavior(variable_name=get_xml_path(self.config.story, sequence) + '>' + get_xml_path(maneuver, event), behaviour=event_sequence))
                        maneuver_parallel = StoryElementStatusToBlackboard(maneuver_parallel, 'MANEUVER', maneuver.attrib.get('name'))
                        single_sequence_iteration.add_child(oneshot_behavior(variable_name=get_xml_path(self.config.story, sequence) + '>' + maneuver.attrib.get('name'), behaviour=maneuver_parallel))
                    single_sequence_iteration = StoryElementStatusToBlackboard(single_sequence_iteration, 'SCENE', sequence.attrib.get('name'))
                    single_sequence_iteration = repeatable_behavior(single_sequence_iteration, get_xml_path(self.config.story, sequence))
                    sequence_behavior.add_child(single_sequence_iteration)
                if sequence_behavior.children:
                    parallel_sequences.add_child(oneshot_behavior(variable_name=get_xml_path(self.config.story, sequence), behaviour=sequence_behavior))
            if parallel_sequences.children:
                parallel_sequences = StoryElementStatusToBlackboard(parallel_sequences, 'ACT', act.attrib.get('name'))
                parallel_behavior.add_child(parallel_sequences)
            start_triggers = act.find('StartTrigger')
            if list(start_triggers) is not None:
                for start_condition in start_triggers:
                    parallel_start_criteria = self._create_condition_container(start_condition, 'StartConditions')
                    if parallel_start_criteria.children:
                        start_conditions.add_child(parallel_start_criteria)
            end_triggers = act.find('StopTrigger')
            if end_triggers is not None and list(end_triggers) is not None:
                for end_condition in end_triggers:
                    parallel_end_criteria = self._create_condition_container(end_condition, 'EndConditions', success_on_all=False)
                    if parallel_end_criteria.children:
                        parallel_behavior.add_child(parallel_end_criteria)
            if start_conditions.children:
                act_sequence.add_child(start_conditions)
            if parallel_behavior.children:
                act_sequence.add_child(parallel_behavior)
            if act_sequence.children:
                story_behavior.add_child(act_sequence)
        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name='behavior')
        env_behavior = self._create_environment_behavior()
        if env_behavior is not None:
            behavior.add_child(oneshot_behavior(variable_name='InitialEnvironmentSettings', behaviour=env_behavior))
        init_behavior = self._create_init_behavior()
        if init_behavior is not None:
            behavior.add_child(oneshot_behavior(variable_name='InitialActorSettings', behaviour=init_behavior))
        behavior.add_child(story_behavior)
        return behavior

    def _create_condition_container(self, node, name='Conditions Group', sequence=None, maneuver=None, success_on_all=True):
        """
        This is a generic function to handle conditions utilising ConditionGroups
        Each ConditionGroup is represented as a Sequence of Conditions
        The ConditionGroups are grouped under a SUCCESS_ON_ONE Parallel
        """
        parallel_condition_groups = py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        for condition_group in node.iter('ConditionGroup'):
            if success_on_all:
                condition_group_sequence = py_trees.composites.Parallel(name='Condition Group', policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
            else:
                condition_group_sequence = py_trees.composites.Parallel(name='Condition Group', policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
            for condition in condition_group.iter('Condition'):
                criterion = OpenScenarioParser.convert_condition_to_atomic(condition, self.other_actors + self.ego_vehicles)
                if sequence is not None and maneuver is not None:
                    xml_path = get_xml_path(self.config.story, sequence) + '>' + get_xml_path(maneuver, condition)
                else:
                    xml_path = get_xml_path(self.config.story, condition)
                criterion = oneshot_behavior(variable_name=xml_path, behaviour=criterion)
                condition_group_sequence.add_child(criterion)
            if condition_group_sequence.children:
                parallel_condition_groups.add_child(condition_group_sequence)
        return parallel_condition_groups

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        parallel_criteria = py_trees.composites.Parallel('EndConditions (Criteria Group)', policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        criteria = []
        for endcondition in self.config.storyboard.iter('StopTrigger'):
            for condition in endcondition.iter('Condition'):
                if condition.attrib.get('name').startswith('criteria_'):
                    condition.set('name', condition.attrib.get('name')[9:])
                    criteria.append(condition)
        for condition in criteria:
            criterion = OpenScenarioParser.convert_condition_to_atomic(condition, self.ego_vehicles)
            parallel_criteria.add_child(criterion)
        return parallel_criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

def get_xml_path(tree, node):
    """
    Extract the full path of a node within an XML tree

    Note: Catalogs are pulled from a separate file so the XML tree is split.
          This means that in order to get the XML path, it must be done in 2 steps.
          Some places in this python script do that by concatenating the results
          of 2 get_xml_path calls with another ">".
          Example: "Behavior>AutopilotSequence" + ">" + "StartAutopilot>StartAutopilot>StartAutopilot"
    """
    path = ''
    parent_map = {c: p for p in tree.iter() for c in p}
    cur_node = node
    while cur_node != tree:
        path = '{}>{}'.format(cur_node.attrib.get('name'), path)
        cur_node = parent_map[cur_node]
    path = path[:-1]
    return path

def repeatable_behavior(behaviour, name=None):
    """
    This behaviour allows a composite with oneshot ancestors to run multiple
    times, resetting the oneshot variables after each execution
    """
    if not name:
        name = behaviour.name
    clear_descendant_variables = ClearBlackboardVariablesStartingWith(name='Clear Descendant Variables of {}'.format(name), variable_name_beginning=name + '>')
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(clear_descendant_variables)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name='RepeatableBehaviour of {}'.format(name))
        sequence.add_children([behaviour, clear_descendant_variables])
    return sequence

class RouteScenario(BasicScenario):
    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, debug_mode=False, criteria_enable=True, timeout=300):
        """
        Setup all relevant parameters and create scenarios along route
        """
        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None
        self._update_route(world, config, debug_mode)
        ego_vehicle = self._update_ego_vehicle()
        self.list_scenarios = self._build_scenario_instances(world, ego_vehicle, self.sampled_scenarios_definitions, scenarios_per_tick=5, timeout=self.timeout, debug_mode=debug_mode)
        super(RouteScenario, self).__init__(name=config.name, ego_vehicles=[ego_vehicle], config=config, world=world, debug_mode=False, terminate_on_failure=False, criteria_enable=criteria_enable)

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)
        gps_route, route = interpolate_trajectory(world, config.trajectory)
        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(config.town, route, world_annotations)
        self.route = route
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
        config.agent.set_global_plan(gps_route, self.route)
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)
        self.timeout = self._estimate_route_timeout()
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5
        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', elevate_transform, rolename='hero')
        return ego_vehicle

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0
        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)
            size = 0.2
            if w[1] == RoadOption.LEFT:
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:
                color = carla.Color(128, 128, 128)
            else:
                color = carla.Color(0, 255, 0)
                size = 0.1
            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)
        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2, color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2, color=carla.Color(255, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """
        rng = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True
            return False
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]
            scenario_choice = rng.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rng.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]
            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)
        return sampled_scenarios

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions, scenarios_per_tick=5, timeout=300, debug_mode=False):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []
        if debug_mode:
            for scenario in scenario_definitions:
                loc = carla.Location(scenario['trigger_position']['x'], scenario['trigger_position']['y'], scenario['trigger_position']['z']) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False, color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)
        for scenario_number, definition in enumerate(scenario_definitions):
            scenario_class = NUMBER_CLASS_TRANSLATION[definition['name']]
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017', ego_vehicle.get_transform(), 'hero')]
            route_var_name = 'ScenarioRouteNumber{}'.format(scenario_number)
            scenario_configuration.route_var_name = route_var_name
            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration, criteria_enable=False, timeout=timeout)
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()
                scenario_number += 1
            except Exception as e:
                if debug_mode:
                    traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue
            scenario_instance_vec.append(scenario_instance)
        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))
            return sublist_of_actors
        list_of_actors = []
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])
        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])
        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])
        return list_of_actors

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        town_amount = {'Town01': 120, 'Town02': 100, 'Town03': 120, 'Town04': 200, 'Town05': 120, 'Town06': 150, 'Town07': 110, 'Town08': 180, 'Town09': 300, 'Town10': 120}
        amount = town_amount[config.town] if config.town in town_amount else 0
        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*', amount, carla.Transform(), autopilot=True, random_location=True, rolename='background')
        if new_actors is None:
            raise Exception('Error: Unable to add the background activity, all spawn points were occupied')
        for _actor in new_actors:
            self.other_actors.append(_actor)
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        scenario_trigger_distance = 1.5
        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        subbehavior = py_trees.composites.Parallel(name='Behavior', policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        scenario_behaviors = []
        blackboard_list = []
        for i, scenario in enumerate(self.list_scenarios):
            if scenario.scenario.behavior is not None:
                route_var_name = scenario.config.route_var_name
                if route_var_name is not None:
                    scenario_behaviors.append(scenario.scenario.behavior)
                    blackboard_list.append([scenario.config.route_var_name, scenario.config.trigger_points[0].location])
                else:
                    name = '{} - {}'.format(i, scenario.scenario.behavior.name)
                    oneshot_idiom = oneshot_behavior(name, behaviour=scenario.scenario.behavior, name=name)
                    scenario_behaviors.append(oneshot_idiom)
        scenario_triggerer = ScenarioTriggerer(self.ego_vehicles[0], self.route, blackboard_list, scenario_trigger_distance, repeat_scenarios=False)
        subbehavior.add_child(scenario_triggerer)
        subbehavior.add_children(scenario_behaviors)
        subbehavior.add_child(Idle())
        behavior.add_child(subbehavior)
        return behavior

    def _create_test_criteria(self):
        """
        """
        criteria = []
        route = convert_transform_to_location(self.route)
        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=False)
        route_criterion = InRouteTest(self.ego_vehicles[0], route=route, offroad_max=30, terminate_on_failure=True)
        completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route)
        outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], route=route)
        red_light_criterion = RunningRedLightTest(self.ego_vehicles[0])
        stop_criterion = RunningStopTest(self.ego_vehicles[0])
        blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[0], speed_threshold=0.1, below_threshold_max_time=90.0, terminate_on_failure=True)
        criteria.append(completion_criterion)
        criteria.append(collision_criterion)
        criteria.append(route_criterion)
        criteria.append(outsidelane_criterion)
        criteria.append(red_light_criterion)
        criteria.append(stop_criterion)
        criteria.append(blocked_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

class OpenScenarioParser(object):
    """
    Pure static class providing conversions from OpenSCENARIO elements to ScenarioRunner elements
    """
    operators = {'greaterThan': operator.gt, 'lessThan': operator.lt, 'equalTo': operator.eq}
    actor_types = {'pedestrian': 'walker', 'vehicle': 'vehicle', 'miscellaneous': 'miscellaneous'}
    tl_states = {'GREEN': carla.TrafficLightState.Green, 'YELLOW': carla.TrafficLightState.Yellow, 'RED': carla.TrafficLightState.Red, 'OFF': carla.TrafficLightState.Off}
    global_osc_parameters = dict()
    use_carla_coordinate_system = False
    osc_filepath = None

    @staticmethod
    def get_traffic_light_from_osc_name(name):
        """
        Returns a carla.TrafficLight instance that matches the name given
        """
        traffic_light = None
        if name.startswith('id='):
            tl_id = name[3:]
            for carla_tl in CarlaDataProvider.get_world().get_actors().filter('traffic.traffic_light'):
                if carla_tl.id == tl_id:
                    traffic_light = carla_tl
                    break
        elif name.startswith('pos='):
            tl_pos = name[4:]
            pos = tl_pos.split(',')
            for carla_tl in CarlaDataProvider.get_world().get_actors().filter('traffic.traffic_light'):
                carla_tl_location = carla_tl.get_transform().location
                distance = carla_tl_location.distance(carla.Location(float(pos[0]), float(pos[1]), carla_tl_location.z))
                if distance < 2.0:
                    traffic_light = carla_tl
                    break
        if traffic_light is None:
            raise AttributeError('Unknown  traffic light {}'.format(name))
        return traffic_light

    @staticmethod
    def set_osc_filepath(filepath):
        """
        Set path of OSC file. This is required if for example custom commands are provided with
        relative paths.
        """
        OpenScenarioParser.osc_filepath = filepath

    @staticmethod
    def set_use_carla_coordinate_system():
        """
        CARLA internally uses a left-hand coordinate system (Unreal), but OpenSCENARIO and OpenDRIVE
        are intended for right-hand coordinate system. Hence, we need to invert the coordinates, if
        the scenario does not use CARLA coordinates, but instead right-hand coordinates.
        """
        OpenScenarioParser.use_carla_coordinate_system = True

    @staticmethod
    def set_parameters(xml_tree, additional_parameter_dict=None):
        """
        Parse the xml_tree, and replace all parameter references
        with the actual values.

        Note: Parameter names must not start with "$", however when referencing a parameter the
              reference has to start with "$".
              https://releases.asam.net/OpenSCENARIO/1.0.0/ASAM_OpenSCENARIO_BS-1-2_User-Guide_V1-0-0.html#_re_use_mechanisms

        Args:
            xml_tree: Containing all nodes that should be updated
            additional_parameter_dict (dictionary): Additional parameters as dict (key, value). Optional.

        returns:
            updated xml_tree, dictonary containing all parameters and their values
        """
        parameter_dict = dict()
        if additional_parameter_dict is not None:
            parameter_dict = additional_parameter_dict
        parameters = xml_tree.find('ParameterDeclarations')
        if parameters is None and (not parameter_dict):
            return (xml_tree, parameter_dict)
        if parameters is None:
            parameters = []
        for parameter in parameters:
            name = parameter.attrib.get('name')
            value = parameter.attrib.get('value')
            parameter_dict[name] = value
        for node in xml_tree.iter():
            for key in node.attrib:
                for param in sorted(parameter_dict, key=len, reverse=True):
                    if '$' + param in node.attrib[key]:
                        node.attrib[key] = node.attrib[key].replace('$' + param, parameter_dict[param])
        return (xml_tree, parameter_dict)

    @staticmethod
    def set_global_parameters(parameter_dict):
        """
        Set global_osc_parameter dictionary

        Args:
            parameter_dict (Dictionary): Input for global_osc_parameter
        """
        OpenScenarioParser.global_osc_parameters = parameter_dict

    @staticmethod
    def get_catalog_entry(catalogs, catalog_reference):
        """
        Get catalog entry referenced by catalog_reference included correct parameter settings

        Args:
            catalogs (Dictionary of dictionaries): List of all catalogs and their entries
            catalog_reference (XML ElementTree): Reference containing the exact catalog to be used

        returns:
            Catalog entry (XML ElementTree)
        """
        entry = catalogs[catalog_reference.attrib.get('catalogName')][catalog_reference.attrib.get('entryName')]
        entry_copy = copy.deepcopy(entry)
        catalog_copy = copy.deepcopy(catalog_reference)
        entry = OpenScenarioParser.assign_catalog_parameters(entry_copy, catalog_copy)
        return entry

    @staticmethod
    def assign_catalog_parameters(entry_instance, catalog_reference):
        """
        Parse catalog_reference, and replace all parameter references
        in entry_instance by the values provided in catalog_reference.

        Not to be used from outside this class.

        Args:
            entry_instance (XML ElementTree): Entry to be updated
            catalog_reference (XML ElementTree): Reference containing the exact parameter values

        returns:
            updated entry_instance with updated parameter values
        """
        parameter_dict = dict()
        for elem in entry_instance.iter():
            if elem.find('ParameterDeclarations') is not None:
                parameters = elem.find('ParameterDeclarations')
                for parameter in parameters:
                    name = parameter.attrib.get('name')
                    value = parameter.attrib.get('value')
                    parameter_dict[name] = value
        for parameter_assignments in catalog_reference.iter('ParameterAssignments'):
            for parameter_assignment in parameter_assignments.iter('ParameterAssignment'):
                parameter = parameter_assignment.attrib.get('parameterRef')
                value = parameter_assignment.attrib.get('value')
                parameter_dict[parameter] = value
        for node in entry_instance.iter():
            for key in node.attrib:
                for param in sorted(parameter_dict, key=len, reverse=True):
                    if '$' + param in node.attrib[key]:
                        node.attrib[key] = node.attrib[key].replace('$' + param, parameter_dict[param])
        OpenScenarioParser.set_parameters(entry_instance, OpenScenarioParser.global_osc_parameters)
        return entry_instance

    @staticmethod
    def get_friction_from_env_action(xml_tree, catalogs):
        """
        Extract the CARLA road friction coefficient from an OSC EnvironmentAction

        Args:
            xml_tree: Containing the EnvironmentAction,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the EnvironmentAction

        returns:
           friction (float)
        """
        set_environment = next(xml_tree.iter('EnvironmentAction'))
        if sum((1 for _ in set_environment.iter('Weather'))) != 0:
            environment = set_environment.find('Environment')
        elif set_environment.find('CatalogReference') is not None:
            catalog_reference = set_environment.find('CatalogReference')
            environment = OpenScenarioParser.get_catalog_entry(catalogs, catalog_reference)
        friction = 1.0
        road_condition = environment.iter('RoadCondition')
        for condition in road_condition:
            friction = condition.attrib.get('frictionScaleFactor')
        return friction

    @staticmethod
    def get_weather_from_env_action(xml_tree, catalogs):
        """
        Extract the CARLA weather parameters from an OSC EnvironmentAction

        Args:
            xml_tree: Containing the EnvironmentAction,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the EnvironmentAction

        returns:
           Weather (srunner.scenariomanager.weather_sim.Weather)
        """
        set_environment = next(xml_tree.iter('EnvironmentAction'))
        if sum((1 for _ in set_environment.iter('Weather'))) != 0:
            environment = set_environment.find('Environment')
        elif set_environment.find('CatalogReference') is not None:
            catalog_reference = set_environment.find('CatalogReference')
            environment = OpenScenarioParser.get_catalog_entry(catalogs, catalog_reference)
        weather = environment.find('Weather')
        sun = weather.find('Sun')
        carla_weather = carla.WeatherParameters()
        carla_weather.sun_azimuth_angle = math.degrees(float(sun.attrib.get('azimuth', 0)))
        carla_weather.sun_altitude_angle = math.degrees(float(sun.attrib.get('elevation', 0)))
        carla_weather.cloudiness = 100 - float(sun.attrib.get('intensity', 0)) * 100
        fog = weather.find('Fog')
        carla_weather.fog_distance = float(fog.attrib.get('visualRange', 'inf'))
        if carla_weather.fog_distance < 1000:
            carla_weather.fog_density = 100
        carla_weather.precipitation = 0
        carla_weather.precipitation_deposits = 0
        carla_weather.wetness = 0
        carla_weather.wind_intensity = 0
        precepitation = weather.find('Precipitation')
        if precepitation.attrib.get('precipitationType') == 'rain':
            carla_weather.precipitation = float(precepitation.attrib.get('intensity')) * 100
            carla_weather.precipitation_deposits = 100
            carla_weather.wetness = carla_weather.precipitation
        elif precepitation.attrib.get('type') == 'snow':
            raise AttributeError('CARLA does not support snow precipitation')
        time_of_day = environment.find('TimeOfDay')
        weather_animation = strtobool(time_of_day.attrib.get('animation'))
        time = time_of_day.attrib.get('dateTime')
        dtime = datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%S')
        return Weather(carla_weather, dtime, weather_animation)

    @staticmethod
    def get_controller(xml_tree, catalogs):
        """
        Extract the object controller from the OSC XML or a catalog

        Args:
            xml_tree: Containing the controller information,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the EnvironmentAction

        returns:
           module: Python module containing the controller implementation
           args: Dictonary with (key, value) parameters for the controller
        """
        assign_action = next(xml_tree.iter('AssignControllerAction'))
        properties = None
        if assign_action.find('Controller') is not None:
            properties = assign_action.find('Controller').find('Properties')
        elif assign_action.find('CatalogReference') is not None:
            catalog_reference = assign_action.find('CatalogReference')
            properties = OpenScenarioParser.get_catalog_entry(catalogs, catalog_reference).find('Properties')
        module = None
        args = {}
        for prop in properties:
            if prop.attrib.get('name') == 'module':
                module = prop.attrib.get('value')
            else:
                args[prop.attrib.get('name')] = prop.attrib.get('value')
        override_action = xml_tree.find('OverrideControllerValueAction')
        for child in override_action:
            if strtobool(child.attrib.get('active')):
                raise NotImplementedError('Controller override actions are not yet supported')
        return (module, args)

    @staticmethod
    def get_route(xml_tree, catalogs):
        """
        Extract the route from the OSC XML or a catalog

        Args:
            xml_tree: Containing the route information,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the Route

        returns:
           waypoints: List of route waypoints
        """
        route = None
        if xml_tree.find('Route') is not None:
            route = xml_tree.find('Route')
        elif xml_tree.find('CatalogReference') is not None:
            catalog_reference = xml_tree.find('CatalogReference')
            route = OpenScenarioParser.get_catalog_entry(catalogs, catalog_reference)
        else:
            raise AttributeError('Unknown private FollowRoute action')
        waypoints = []
        if route is not None:
            for waypoint in route.iter('Waypoint'):
                position = waypoint.find('Position')
                transform = OpenScenarioParser.convert_position_to_transform(position)
                waypoints.append(transform)
        return waypoints

    @staticmethod
    def convert_position_to_transform(position, actor_list=None):
        """
        Convert an OpenScenario position into a CARLA transform

        Not supported: Road, RelativeRoad, Lane, RelativeLane as the PythonAPI currently
                       does not provide sufficient access to OpenDrive information
                       Also not supported is Route. This can be added by checking additional
                       route information
        """
        if position.find('WorldPosition') is not None:
            world_pos = position.find('WorldPosition')
            x = float(world_pos.attrib.get('x', 0))
            y = float(world_pos.attrib.get('y', 0))
            z = float(world_pos.attrib.get('z', 0))
            yaw = math.degrees(float(world_pos.attrib.get('h', 0)))
            pitch = math.degrees(float(world_pos.attrib.get('p', 0)))
            roll = math.degrees(float(world_pos.attrib.get('r', 0)))
            if not OpenScenarioParser.use_carla_coordinate_system:
                y = y * -1.0
                yaw = yaw * -1.0
            return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))
        elif position.find('RelativeWorldPosition') is not None or position.find('RelativeObjectPosition') is not None or position.find('RelativeLanePosition') is not None:
            if position.find('RelativeWorldPosition') is not None:
                rel_pos = position.find('RelativeWorldPosition')
            if position.find('RelativeObjectPosition') is not None:
                rel_pos = position.find('RelativeObjectPosition')
            if position.find('RelativeLanePosition') is not None:
                rel_pos = position.find('RelativeLanePosition')
            obj = rel_pos.attrib.get('entityRef')
            obj_actor = None
            actor_transform = None
            if actor_list is not None:
                for actor in actor_list:
                    if actor.rolename == obj:
                        obj_actor = actor
                        actor_transform = actor.transform
            else:
                for actor in CarlaDataProvider.get_world().get_actors():
                    if 'role_name' in actor.attributes and actor.attributes['role_name'] == obj:
                        obj_actor = actor
                        actor_transform = obj_actor.get_transform()
                        break
            if obj_actor is None:
                raise AttributeError("Object '{}' provided as position reference is not known".format(obj))
            is_absolute = False
            dyaw = 0
            dpitch = 0
            droll = 0
            if rel_pos.find('Orientation') is not None:
                orientation = rel_pos.find('Orientation')
                is_absolute = orientation.attrib.get('type') == 'absolute'
                dyaw = math.degrees(float(orientation.attrib.get('h', 0)))
                dpitch = math.degrees(float(orientation.attrib.get('p', 0)))
                droll = math.degrees(float(orientation.attrib.get('r', 0)))
            if not OpenScenarioParser.use_carla_coordinate_system:
                dyaw = dyaw * -1.0
            yaw = actor_transform.rotation.yaw
            pitch = actor_transform.rotation.pitch
            roll = actor_transform.rotation.roll
            if not is_absolute:
                yaw = yaw + dyaw
                pitch = pitch + dpitch
                roll = roll + droll
            else:
                yaw = dyaw
                pitch = dpitch
                roll = droll
            if position.find('RelativeWorldPosition') is not None or position.find('RelativeObjectPosition') is not None:
                dx = float(rel_pos.attrib.get('dx', 0))
                dy = float(rel_pos.attrib.get('dy', 0))
                dz = float(rel_pos.attrib.get('dz', 0))
                if not OpenScenarioParser.use_carla_coordinate_system:
                    dy = dy * -1.0
                x = actor_transform.location.x + dx
                y = actor_transform.location.y + dy
                z = actor_transform.location.z + dz
            elif position.find('RelativeLanePosition') is not None:
                dlane = float(rel_pos.attrib.get('dLane'))
                ds = float(rel_pos.attrib.get('ds'))
                offset = float(rel_pos.attrib.get('offset', 0.0))
                carla_map = CarlaDataProvider.get_map()
                relative_waypoint = carla_map.get_waypoint(actor_transform.location)
                if dlane == 0:
                    wp = relative_waypoint
                elif dlane == -1:
                    wp = relative_waypoint.get_left_lane()
                elif dlane == 1:
                    wp = relative_waypoint.get_right_lane()
                if wp is None:
                    raise AttributeError("Object '{}' position with dLane={} is not valid".format(obj, dlane))
                if ds < 0:
                    ds = -1.0 * ds
                    wp = wp.previous(ds)[-1]
                else:
                    wp = wp.next(ds)[-1]
                h = math.radians(wp.transform.rotation.yaw)
                x_offset = math.sin(h) * offset
                y_offset = math.cos(h) * offset
                if OpenScenarioParser.use_carla_coordinate_system:
                    x_offset = x_offset * -1.0
                    y_offset = y_offset * -1.0
                x = wp.transform.location.x + x_offset
                y = wp.transform.location.y + y_offset
                z = wp.transform.location.z
            return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))
        elif position.find('RoadPosition') is not None:
            raise NotImplementedError('Road positions are not yet supported')
        elif position.find('RelativeRoadPosition') is not None:
            raise NotImplementedError('RelativeRoad positions are not yet supported')
        elif position.find('LanePosition') is not None:
            lane_pos = position.find('LanePosition')
            road_id = int(lane_pos.attrib.get('roadId', 0))
            lane_id = int(lane_pos.attrib.get('laneId', 0))
            offset = float(lane_pos.attrib.get('offset', 0))
            s = float(lane_pos.attrib.get('s', 0))
            is_absolute = True
            waypoint = CarlaDataProvider.get_map().get_waypoint_xodr(road_id, lane_id, s)
            if waypoint is None:
                raise AttributeError('Lane position cannot be found')
            transform = waypoint.transform
            if lane_pos.find('Orientation') is not None:
                orientation = lane_pos.find('Orientation')
                dyaw = math.degrees(float(orientation.attrib.get('h', 0)))
                dpitch = math.degrees(float(orientation.attrib.get('p', 0)))
                droll = math.degrees(float(orientation.attrib.get('r', 0)))
                if not OpenScenarioParser.use_carla_coordinate_system:
                    dyaw = dyaw * -1.0
                transform.rotation.yaw = transform.rotation.yaw + dyaw
                transform.rotation.pitch = transform.rotation.pitch + dpitch
                transform.rotation.roll = transform.rotation.roll + droll
            if offset != 0:
                forward_vector = transform.rotation.get_forward_vector()
                orthogonal_vector = carla.Vector3D(x=-forward_vector.y, y=forward_vector.x, z=forward_vector.z)
                transform.location.x = transform.location.x + offset * orthogonal_vector.x
                transform.location.y = transform.location.y + offset * orthogonal_vector.y
            return transform
        elif position.find('RoutePosition') is not None:
            raise NotImplementedError('Route positions are not yet supported')
        else:
            raise AttributeError('Unknown position')

    @staticmethod
    def convert_condition_to_atomic(condition, actor_list):
        """
        Convert an OpenSCENARIO condition into a Behavior/Criterion atomic

        If there is a delay defined in the condition, then the condition is checked after the delay time
        passed by, e.g. <Condition name="" delay="5">.

        Note: Not all conditions are currently supported.
        """
        atomic = None
        delay_atomic = None
        condition_name = condition.attrib.get('name')
        if condition.attrib.get('delay') is not None and str(condition.attrib.get('delay')) != '0':
            delay = float(condition.attrib.get('delay'))
            delay_atomic = TimeOut(delay)
        if condition.find('ByEntityCondition') is not None:
            trigger_actor = None
            triggered_actor = None
            for triggering_entities in condition.find('ByEntityCondition').iter('TriggeringEntities'):
                for entity in triggering_entities.iter('EntityRef'):
                    for actor in actor_list:
                        if entity.attrib.get('entityRef', None) == actor.attributes['role_name']:
                            trigger_actor = actor
                            break
            for entity_condition in condition.find('ByEntityCondition').iter('EntityCondition'):
                if entity_condition.find('EndOfRoadCondition') is not None:
                    end_road_condition = entity_condition.find('EndOfRoadCondition')
                    condition_duration = float(end_road_condition.attrib.get('duration'))
                    atomic_cls = py_trees.meta.inverter(EndofRoadTest)
                    atomic = atomic_cls(trigger_actor, condition_duration, terminate_on_failure=True, name=condition_name)
                elif entity_condition.find('CollisionCondition') is not None:
                    collision_condition = entity_condition.find('CollisionCondition')
                    if collision_condition.find('EntityRef') is not None:
                        collision_entity = collision_condition.find('EntityRef')
                        for actor in actor_list:
                            if collision_entity.attrib.get('entityRef', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break
                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(collision_condition.attrib.get('entityRef', None)))
                        atomic_cls = py_trees.meta.inverter(CollisionTest)
                        atomic = atomic_cls(trigger_actor, other_actor=triggered_actor, terminate_on_failure=True, name=condition_name)
                    elif collision_condition.find('ByType') is not None:
                        collision_type = collision_condition.find('ByType').attrib.get('type', None)
                        triggered_type = OpenScenarioParser.actor_types[collision_type]
                        atomic_cls = py_trees.meta.inverter(CollisionTest)
                        atomic = atomic_cls(trigger_actor, other_actor_type=triggered_type, terminate_on_failure=True, name=condition_name)
                    else:
                        atomic_cls = py_trees.meta.inverter(CollisionTest)
                        atomic = atomic_cls(trigger_actor, terminate_on_failure=True, name=condition_name)
                elif entity_condition.find('OffroadCondition') is not None:
                    off_condition = entity_condition.find('OffroadCondition')
                    condition_duration = float(off_condition.attrib.get('duration'))
                    atomic_cls = py_trees.meta.inverter(OffRoadTest)
                    atomic = atomic_cls(trigger_actor, condition_duration, terminate_on_failure=True, name=condition_name)
                elif entity_condition.find('TimeHeadwayCondition') is not None:
                    headtime_condition = entity_condition.find('TimeHeadwayCondition')
                    condition_value = float(headtime_condition.attrib.get('value'))
                    condition_rule = headtime_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]
                    condition_freespace = strtobool(headtime_condition.attrib.get('freespace', False))
                    if condition_freespace:
                        raise NotImplementedError('TimeHeadwayCondition: freespace attribute is currently not implemented')
                    condition_along_route = strtobool(headtime_condition.attrib.get('alongRoute', False))
                    for actor in actor_list:
                        if headtime_condition.attrib.get('entityRef', None) == actor.attributes['role_name']:
                            triggered_actor = actor
                            break
                    if triggered_actor is None:
                        raise AttributeError("Cannot find actor '{}' for condition".format(headtime_condition.attrib.get('entityRef', None)))
                    atomic = InTimeToArrivalToVehicle(trigger_actor, triggered_actor, condition_value, condition_along_route, condition_operator, condition_name)
                elif entity_condition.find('TimeToCollisionCondition') is not None:
                    ttc_condition = entity_condition.find('TimeToCollisionCondition')
                    condition_rule = ttc_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]
                    condition_value = ttc_condition.attrib.get('value')
                    condition_target = ttc_condition.find('TimeToCollisionConditionTarget')
                    condition_freespace = strtobool(ttc_condition.attrib.get('freespace', False))
                    if condition_freespace:
                        raise NotImplementedError('TimeToCollisionCondition: freespace attribute is currently not implemented')
                    condition_along_route = strtobool(ttc_condition.attrib.get('alongRoute', False))
                    if condition_target.find('Position') is not None:
                        position = condition_target.find('Position')
                        atomic = InTimeToArrivalToOSCPosition(trigger_actor, position, condition_value, condition_along_route, condition_operator)
                    else:
                        for actor in actor_list:
                            if ttc_condition.attrib.get('EntityRef', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break
                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(ttc_condition.attrib.get('EntityRef', None)))
                        atomic = InTimeToArrivalToVehicle(trigger_actor, triggered_actor, condition_value, condition_along_route, condition_operator, condition_name)
                elif entity_condition.find('AccelerationCondition') is not None:
                    accel_condition = entity_condition.find('AccelerationCondition')
                    condition_value = float(accel_condition.attrib.get('value'))
                    condition_rule = accel_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]
                    atomic = TriggerAcceleration(trigger_actor, condition_value, condition_operator, condition_name)
                elif entity_condition.find('StandStillCondition') is not None:
                    ss_condition = entity_condition.find('StandStillCondition')
                    duration = float(ss_condition.attrib.get('duration'))
                    atomic = StandStill(trigger_actor, condition_name, duration)
                elif entity_condition.find('SpeedCondition') is not None:
                    spd_condition = entity_condition.find('SpeedCondition')
                    condition_value = float(spd_condition.attrib.get('value'))
                    condition_rule = spd_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]
                    atomic = TriggerVelocity(trigger_actor, condition_value, condition_operator, condition_name)
                elif entity_condition.find('RelativeSpeedCondition') is not None:
                    relspd_condition = entity_condition.find('RelativeSpeedCondition')
                    condition_value = float(relspd_condition.attrib.get('value'))
                    condition_rule = relspd_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]
                    for actor in actor_list:
                        if relspd_condition.attrib.get('entityRef', None) == actor.attributes['role_name']:
                            triggered_actor = actor
                            break
                    if triggered_actor is None:
                        raise AttributeError("Cannot find actor '{}' for condition".format(relspd_condition.attrib.get('entityRef', None)))
                    atomic = RelativeVelocityToOtherActor(trigger_actor, triggered_actor, condition_value, condition_operator, condition_name)
                elif entity_condition.find('TraveledDistanceCondition') is not None:
                    distance_condition = entity_condition.find('TraveledDistanceCondition')
                    distance_value = float(distance_condition.attrib.get('value'))
                    atomic = DriveDistance(trigger_actor, distance_value, name=condition_name)
                elif entity_condition.find('ReachPositionCondition') is not None:
                    rp_condition = entity_condition.find('ReachPositionCondition')
                    distance_value = float(rp_condition.attrib.get('tolerance'))
                    position = rp_condition.find('Position')
                    atomic = InTriggerDistanceToOSCPosition(trigger_actor, position, distance_value, name=condition_name)
                elif entity_condition.find('DistanceCondition') is not None:
                    distance_condition = entity_condition.find('DistanceCondition')
                    distance_value = float(distance_condition.attrib.get('value'))
                    distance_rule = distance_condition.attrib.get('rule')
                    distance_operator = OpenScenarioParser.operators[distance_rule]
                    distance_freespace = strtobool(distance_condition.attrib.get('freespace', False))
                    if distance_freespace:
                        raise NotImplementedError('DistanceCondition: freespace attribute is currently not implemented')
                    distance_along_route = strtobool(distance_condition.attrib.get('alongRoute', False))
                    if distance_condition.find('Position') is not None:
                        position = distance_condition.find('Position')
                        atomic = InTriggerDistanceToOSCPosition(trigger_actor, position, distance_value, distance_along_route, distance_operator, name=condition_name)
                elif entity_condition.find('RelativeDistanceCondition') is not None:
                    distance_condition = entity_condition.find('RelativeDistanceCondition')
                    distance_value = float(distance_condition.attrib.get('value'))
                    distance_freespace = strtobool(distance_condition.attrib.get('freespace', False))
                    if distance_freespace:
                        raise NotImplementedError('RelativeDistanceCondition: freespace attribute is currently not implemented')
                    if distance_condition.attrib.get('relativeDistanceType') == 'cartesianDistance':
                        for actor in actor_list:
                            if distance_condition.attrib.get('entityRef', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break
                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(distance_condition.attrib.get('entityRef', None)))
                        condition_rule = distance_condition.attrib.get('rule')
                        condition_operator = OpenScenarioParser.operators[condition_rule]
                        atomic = InTriggerDistanceToVehicle(triggered_actor, trigger_actor, distance_value, condition_operator, name=condition_name)
                    else:
                        raise NotImplementedError('RelativeDistance condition with the given specification is not yet supported')
        elif condition.find('ByValueCondition') is not None:
            value_condition = condition.find('ByValueCondition')
            if value_condition.find('ParameterCondition') is not None:
                parameter_condition = value_condition.find('ParameterCondition')
                arg_name = parameter_condition.attrib.get('parameterRef')
                value = parameter_condition.attrib.get('value')
                if value != '':
                    arg_value = float(value)
                else:
                    arg_value = 0
                parameter_condition.attrib.get('rule')
                if condition_name in globals():
                    criterion_instance = globals()[condition_name]
                else:
                    raise AttributeError('The condition {} cannot be mapped to a criterion atomic'.format(condition_name))
                atomic = py_trees.composites.Parallel('Evaluation Criteria for multiple ego vehicles')
                for triggered_actor in actor_list:
                    if arg_name != '':
                        atomic.add_child(criterion_instance(triggered_actor, arg_value))
                    else:
                        atomic.add_child(criterion_instance(triggered_actor))
            elif value_condition.find('SimulationTimeCondition') is not None:
                simtime_condition = value_condition.find('SimulationTimeCondition')
                value = float(simtime_condition.attrib.get('value'))
                rule = simtime_condition.attrib.get('rule')
                atomic = SimulationTimeCondition(value, success_rule=rule)
            elif value_condition.find('TimeOfDayCondition') is not None:
                tod_condition = value_condition.find('TimeOfDayCondition')
                condition_date = tod_condition.attrib.get('dateTime')
                condition_rule = tod_condition.attrib.get('rule')
                condition_operator = OpenScenarioParser.operators[condition_rule]
                atomic = TimeOfDayComparison(condition_date, condition_operator, condition_name)
            elif value_condition.find('StoryboardElementStateCondition') is not None:
                state_condition = value_condition.find('StoryboardElementStateCondition')
                element_name = state_condition.attrib.get('storyboardElementRef')
                element_type = state_condition.attrib.get('storyboardElementType')
                state = state_condition.attrib.get('state')
                if state == 'startTransition':
                    atomic = OSCStartEndCondition(element_type, element_name, rule='START', name=state + 'Condition')
                elif state == 'stopTransition' or state == 'endTransition' or state == 'completeState':
                    atomic = OSCStartEndCondition(element_type, element_name, rule='END', name=state + 'Condition')
                else:
                    raise NotImplementedError('Only start, stop, endTransitions and completeState are currently supported')
            elif value_condition.find('UserDefinedValueCondition') is not None:
                raise NotImplementedError('ByValue UserDefinedValue conditions are not yet supported')
            elif value_condition.find('TrafficSignalCondition') is not None:
                tl_condition = value_condition.find('TrafficSignalCondition')
                name_condition = tl_condition.attrib.get('name')
                traffic_light = OpenScenarioParser.get_traffic_light_from_osc_name(name_condition)
                tl_state = tl_condition.attrib.get('state').upper()
                if tl_state not in OpenScenarioParser.tl_states:
                    raise KeyError('CARLA only supports Green, Red, Yellow or Off')
                state_condition = OpenScenarioParser.tl_states[tl_state]
                atomic = WaitForTrafficLightState(traffic_light, state_condition, name=condition_name)
            elif value_condition.find('TrafficSignalControllerCondition') is not None:
                raise NotImplementedError('ByValue TrafficSignalController conditions are not yet supported')
            else:
                raise AttributeError('Unknown ByValue condition')
        else:
            raise AttributeError('Unknown condition')
        if delay_atomic is not None and atomic is not None:
            new_atomic = py_trees.composites.Sequence('delayed sequence')
            new_atomic.add_child(delay_atomic)
            new_atomic.add_child(atomic)
        else:
            new_atomic = atomic
        return new_atomic

    @staticmethod
    def convert_maneuver_to_atomic(action, actor, catalogs):
        """
        Convert an OpenSCENARIO maneuver action into a Behavior atomic

        Note not all OpenSCENARIO actions are currently supported
        """
        maneuver_name = action.attrib.get('name', 'unknown')
        if action.find('GlobalAction') is not None:
            global_action = action.find('GlobalAction')
            if global_action.find('InfrastructureAction') is not None:
                infrastructure_action = global_action.find('InfrastructureAction').find('TrafficSignalAction')
                if infrastructure_action.find('TrafficSignalStateAction') is not None:
                    traffic_light_action = infrastructure_action.find('TrafficSignalStateAction')
                    name_condition = traffic_light_action.attrib.get('name')
                    traffic_light = OpenScenarioParser.get_traffic_light_from_osc_name(name_condition)
                    tl_state = traffic_light_action.attrib.get('state').upper()
                    if tl_state not in OpenScenarioParser.tl_states:
                        raise KeyError('CARLA only supports Green, Red, Yellow or Off')
                    traffic_light_state = OpenScenarioParser.tl_states[tl_state]
                    atomic = TrafficLightStateSetter(traffic_light, traffic_light_state, name=maneuver_name + '_' + str(traffic_light.id))
                else:
                    raise NotImplementedError('TrafficLights can only be influenced via TrafficSignalStateAction')
            elif global_action.find('EnvironmentAction') is not None:
                weather_behavior = ChangeWeather(OpenScenarioParser.get_weather_from_env_action(global_action, catalogs))
                friction_behavior = ChangeRoadFriction(OpenScenarioParser.get_friction_from_env_action(global_action, catalogs))
                env_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name=maneuver_name)
                env_behavior.add_child(oneshot_behavior(variable_name=maneuver_name + '>WeatherUpdate', behaviour=weather_behavior))
                env_behavior.add_child(oneshot_behavior(variable_name=maneuver_name + '>FrictionUpdate', behaviour=friction_behavior))
                return env_behavior
            else:
                raise NotImplementedError('Global actions are not yet supported')
        elif action.find('UserDefinedAction') is not None:
            user_defined_action = action.find('UserDefinedAction')
            if user_defined_action.find('CustomCommandAction') is not None:
                command = user_defined_action.find('CustomCommandAction').attrib.get('type')
                atomic = RunScript(command, base_path=OpenScenarioParser.osc_filepath, name=maneuver_name)
        elif action.find('PrivateAction') is not None:
            private_action = action.find('PrivateAction')
            if private_action.find('LongitudinalAction') is not None:
                private_action = private_action.find('LongitudinalAction')
                if private_action.find('SpeedAction') is not None:
                    long_maneuver = private_action.find('SpeedAction')
                    distance = float('inf')
                    duration = float('inf')
                    dimension = long_maneuver.find('SpeedActionDynamics').attrib.get('dynamicsDimension')
                    if dimension == 'distance':
                        distance = float(long_maneuver.find('SpeedActionDynamics').attrib.get('value', float('inf')))
                    else:
                        duration = float(long_maneuver.find('SpeedActionDynamics').attrib.get('value', float('inf')))
                    if long_maneuver.find('SpeedActionTarget').find('AbsoluteTargetSpeed') is not None:
                        target_speed = float(long_maneuver.find('SpeedActionTarget').find('AbsoluteTargetSpeed').attrib.get('value', 0))
                        atomic = ChangeActorTargetSpeed(actor, target_speed, distance=distance, duration=duration, name=maneuver_name)
                    if long_maneuver.find('SpeedActionTarget').find('RelativeTargetSpeed') is not None:
                        relative_speed = long_maneuver.find('SpeedActionTarget').find('RelativeTargetSpeed')
                        obj = relative_speed.attrib.get('entityRef')
                        value = float(relative_speed.attrib.get('value', 0))
                        value_type = relative_speed.attrib.get('speedTargetValueType')
                        continuous = relative_speed.attrib.get('continuous')
                        for traffic_actor in CarlaDataProvider.get_world().get_actors():
                            if 'role_name' in traffic_actor.attributes and traffic_actor.attributes['role_name'] == obj:
                                obj_actor = traffic_actor
                        atomic = ChangeActorTargetSpeed(actor, target_speed=0.0, relative_actor=obj_actor, value=value, value_type=value_type, continuous=continuous, distance=distance, duration=duration, name=maneuver_name)
                elif private_action.find('LongitudinalDistanceAction') is not None:
                    raise NotImplementedError('Longitudinal distance actions are not yet supported')
                else:
                    raise AttributeError('Unknown longitudinal action')
            elif private_action.find('LateralAction') is not None:
                private_action = private_action.find('LateralAction')
                if private_action.find('LaneChangeAction') is not None:
                    lat_maneuver = private_action.find('LaneChangeAction')
                    target_lane_rel = float(lat_maneuver.find('LaneChangeTarget').find('RelativeTargetLane').attrib.get('value', 0))
                    distance = float('inf')
                    duration = float('inf')
                    dimension = lat_maneuver.find('LaneChangeActionDynamics').attrib.get('dynamicsDimension')
                    if dimension == 'distance':
                        distance = float(lat_maneuver.find('LaneChangeActionDynamics').attrib.get('value', float('inf')))
                    else:
                        duration = float(lat_maneuver.find('LaneChangeActionDynamics').attrib.get('value', float('inf')))
                    atomic = ChangeActorLateralMotion(actor, direction='left' if target_lane_rel < 0 else 'right', distance_lane_change=distance, distance_other_lane=1000, name=maneuver_name)
                else:
                    raise AttributeError('Unknown lateral action')
            elif private_action.find('VisibilityAction') is not None:
                raise NotImplementedError('Visibility actions are not yet supported')
            elif private_action.find('SynchronizeAction') is not None:
                raise NotImplementedError('Synchronization actions are not yet supported')
            elif private_action.find('ActivateControllerAction') is not None:
                private_action = private_action.find('ActivateControllerAction')
                activate = strtobool(private_action.attrib.get('longitudinal'))
                atomic = ChangeAutoPilot(actor, activate, name=maneuver_name)
            elif private_action.find('ControllerAction') is not None:
                controller_action = private_action.find('ControllerAction')
                module, args = OpenScenarioParser.get_controller(controller_action, catalogs)
                atomic = ChangeActorControl(actor, control_py_module=module, args=args)
            elif private_action.find('TeleportAction') is not None:
                position = private_action.find('TeleportAction')
                atomic = ActorTransformSetterToOSCPosition(actor, position, name=maneuver_name)
            elif private_action.find('RoutingAction') is not None:
                private_action = private_action.find('RoutingAction')
                if private_action.find('AssignRouteAction') is not None:
                    route_action = private_action.find('AssignRouteAction')
                    waypoints = OpenScenarioParser.get_route(route_action, catalogs)
                    atomic = ChangeActorWaypoints(actor, waypoints=waypoints, name=maneuver_name)
                elif private_action.find('FollowTrajectoryAction') is not None:
                    raise NotImplementedError('Private FollowTrajectory actions are not yet supported')
                elif private_action.find('AcquirePositionAction') is not None:
                    route_action = private_action.find('AcquirePositionAction')
                    osc_position = route_action.find('Position')
                    position = OpenScenarioParser.convert_position_to_transform(osc_position)
                    atomic = ChangeActorWaypointsToReachPosition(actor, position=position, name=maneuver_name)
                else:
                    raise AttributeError('Unknown private routing action')
            else:
                raise AttributeError('Unknown private action')
        elif list(action):
            raise AttributeError('Unknown action: {}'.format(maneuver_name))
        else:
            return Idle(duration=0, name=maneuver_name)
        return atomic

