# Cluster 9

def fetch_dict(endpoint):
    data = None
    if endpoint.startswith(('http:', 'https:', 'ftp:')):
        proxies = autodetect_proxy()
        if proxies:
            response = requests.get(url=endpoint, proxies=proxies)
        else:
            response = requests.get(url=endpoint)
        try:
            data = response.json()
        except json.decoder.JSONDecodeError:
            data = {}
    else:
        data = {}
        if os.path.exists(endpoint):
            with open(endpoint) as fd:
                try:
                    data = json.load(fd)
                except json.JSONDecodeError:
                    data = {}
    return data

def autodetect_proxy():
    proxies = {}
    proxy_https = os.getenv('HTTPS_PROXY', os.getenv('https_proxy', None))
    proxy_http = os.getenv('HTTP_PROXY', os.getenv('http_proxy', None))
    if proxy_https:
        proxies['https'] = proxy_https
    if proxy_http:
        proxies['http'] = proxy_http
    return proxies

def save_dict(endpoint, data):
    if endpoint.startswith(('http:', 'https:', 'ftp:')):
        proxies = autodetect_proxy()
        if proxies:
            _ = requests.patch(url=endpoint, headers={'content-type': 'application/json'}, data=json.dumps(data, indent=4, sort_keys=True), proxies=proxies)
        else:
            _ = requests.patch(url=endpoint, headers={'content-type': 'application/json'}, data=json.dumps(data, indent=4, sort_keys=True))
    else:
        with open(endpoint, 'w') as fd:
            json.dump(data, fd, indent=4, sort_keys=True)

class RouteIndexer:

    def __init__(self, routes_file, scenarios_file, repetitions):
        self._routes_file = routes_file
        self._scenarios_file = scenarios_file
        self._repetitions = repetitions
        self._configs_dict = OrderedDict()
        self._configs_list = []
        self.routes_length = []
        self._index = 0
        route_configurations = RouteParser.parse_routes_file(self._routes_file, self._scenarios_file, False)
        self.n_routes = len(route_configurations)
        self.total = self.n_routes * self._repetitions
        for i, config in enumerate(route_configurations):
            for repetition in range(repetitions):
                config.index = i * self._repetitions + repetition
                config.repetition_index = repetition
                self._configs_dict['{}.{}'.format(config.name, repetition)] = copy.copy(config)
        self._configs_list = list(self._configs_dict.items())

    def peek(self):
        return not self._index >= len(self._configs_list)

    def next(self):
        if self._index >= len(self._configs_list):
            return None
        key, config = self._configs_list[self._index]
        self._index += 1
        return config

    def resume(self, endpoint):
        data = fetch_dict(endpoint)
        if data:
            checkpoint_dict = dictor(data, '_checkpoint')
            if checkpoint_dict and 'progress' in checkpoint_dict:
                progress = checkpoint_dict['progress']
                if not progress:
                    current_route = 0
                else:
                    current_route, total_routes = progress
                if current_route <= self.total:
                    self._index = current_route
                else:
                    print('Problem reading checkpoint. Route id {} larger than maximum number of routes {}'.format(current_route, self.total))

    def save_state(self, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()
        data['_checkpoint']['progress'] = [self._index, self.total]
        save_dict(endpoint, data)

def create_default_json_msg():
    msg = {'sensors': [], 'values': [], 'labels': [], 'entry_status': '', 'eligible': '', '_checkpoint': {'progress': [], 'records': [], 'global_record': {}}}
    return msg

class StatisticsManager(object):
    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """

    def __init__(self):
        self._master_scenario = None
        self._registry_route_records = []

    def resume(self, endpoint):
        data = fetch_dict(endpoint)
        if data and dictor(data, '_checkpoint.records'):
            records = data['_checkpoint']['records']
            for record in records:
                self._registry_route_records.append(to_route_record(record))

    def set_route(self, route_id, index):
        self._master_scenario = None
        route_record = RouteRecord()
        route_record.route_id = route_id
        route_record.index = index
        if index < len(self._registry_route_records):
            self._registry_route_records[index] = route_record
        else:
            self._registry_route_records.append(route_record)

    def set_scenario(self, scenario):
        """
        Sets the scenario from which the statistics willb e taken
        """
        self._master_scenario = scenario

    def compute_route_statistics(self, config, duration_time_system=-1, duration_time_game=-1, failure=''):
        """
        Compute the current statistics by evaluating all relevant scenario criteria
        """
        index = config.index
        if not self._registry_route_records or index >= len(self._registry_route_records):
            raise Exception('Critical error with the route registry.')
        route_record = self._registry_route_records[index]
        target_reached = False
        score_penalty = 1.0
        score_route = 0.0
        route_record.meta['duration_system'] = duration_time_system
        route_record.meta['duration_game'] = duration_time_game
        route_record.meta['route_length'] = compute_route_length(config)
        if self._master_scenario:
            if self._master_scenario.timeout_node.timeout:
                route_record.infractions['route_timeout'].append('Route timeout.')
                failure = 'Agent timed out'
            for node in self._master_scenario.get_criteria():
                if node.list_traffic_events:
                    for event in node.list_traffic_events:
                        if event.get_type() == TrafficEventType.COLLISION_STATIC:
                            score_penalty *= PENALTY_COLLISION_STATIC
                            route_record.infractions['collisions_layout'].append(event.get_message())
                        elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                            score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                            route_record.infractions['collisions_pedestrian'].append(event.get_message())
                        elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                            score_penalty *= PENALTY_COLLISION_VEHICLE
                            route_record.infractions['collisions_vehicle'].append(event.get_message())
                        elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                            score_penalty *= 1 - event.get_dict()['percentage'] / 100
                            route_record.infractions['outside_route_lanes'].append(event.get_message())
                        elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                            score_penalty *= PENALTY_TRAFFIC_LIGHT
                            route_record.infractions['red_light'].append(event.get_message())
                        elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                            route_record.infractions['route_dev'].append(event.get_message())
                            failure = 'Agent deviated from the route'
                        elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                            score_penalty *= PENALTY_STOP
                            route_record.infractions['stop_infraction'].append(event.get_message())
                        elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                            route_record.infractions['vehicle_blocked'].append(event.get_message())
                            failure = 'Agent got blocked'
                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                            score_route = 100.0
                            target_reached = True
                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                            if not target_reached:
                                if event.get_dict():
                                    score_route = event.get_dict()['route_completed']
                                else:
                                    score_route = 0
        route_record.scores['score_route'] = score_route
        route_record.scores['score_penalty'] = score_penalty
        route_record.scores['score_composed'] = max(score_route * score_penalty, 0.0)
        if target_reached:
            route_record.status = 'Completed'
        else:
            route_record.status = 'Failed'
            if failure:
                route_record.status += ' - ' + failure
        return route_record

    def compute_global_statistics(self, total_routes):
        global_record = RouteRecord()
        global_record.route_id = -1
        global_record.index = -1
        global_record.status = 'Completed'
        if self._registry_route_records:
            for route_record in self._registry_route_records:
                global_record.scores['score_route'] += route_record.scores['score_route']
                global_record.scores['score_penalty'] += route_record.scores['score_penalty']
                global_record.scores['score_composed'] += route_record.scores['score_composed']
                for key in global_record.infractions.keys():
                    route_length_kms = max(route_record.scores['score_route'] * route_record.meta['route_length'] / 1000.0, 0.001)
                    if isinstance(global_record.infractions[key], list):
                        global_record.infractions[key] = len(route_record.infractions[key]) / route_length_kms
                    else:
                        global_record.infractions[key] += len(route_record.infractions[key]) / route_length_kms
                if route_record.status is not 'Completed':
                    global_record.status = 'Failed'
                    if 'exceptions' not in global_record.meta:
                        global_record.meta['exceptions'] = []
                    global_record.meta['exceptions'].append((route_record.route_id, route_record.index, route_record.status))
        global_record.scores['score_route'] /= float(total_routes)
        global_record.scores['score_penalty'] /= float(total_routes)
        global_record.scores['score_composed'] /= float(total_routes)
        return global_record

    @staticmethod
    def save_record(route_record, index, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()
        stats_dict = route_record.__dict__
        record_list = data['_checkpoint']['records']
        if index > len(record_list):
            print('Error! No enough entries in the list')
            sys.exit(-1)
        elif index == len(record_list):
            record_list.append(stats_dict)
        else:
            record_list[index] = stats_dict
        save_dict(endpoint, data)

    @staticmethod
    def save_global_record(route_record, sensors, total_routes, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()
        stats_dict = route_record.__dict__
        data['_checkpoint']['global_record'] = stats_dict
        data['values'] = ['{:.3f}'.format(stats_dict['scores']['score_composed']), '{:.3f}'.format(stats_dict['scores']['score_route']), '{:.3f}'.format(stats_dict['scores']['score_penalty']), '{:.3f}'.format(stats_dict['infractions']['collisions_pedestrian']), '{:.3f}'.format(stats_dict['infractions']['collisions_vehicle']), '{:.3f}'.format(stats_dict['infractions']['collisions_layout']), '{:.3f}'.format(stats_dict['infractions']['red_light']), '{:.3f}'.format(stats_dict['infractions']['stop_infraction']), '{:.3f}'.format(stats_dict['infractions']['outside_route_lanes']), '{:.3f}'.format(stats_dict['infractions']['route_dev']), '{:.3f}'.format(stats_dict['infractions']['route_timeout']), '{:.3f}'.format(stats_dict['infractions']['vehicle_blocked'])]
        data['labels'] = ['Avg. driving score', 'Avg. route completion', 'Avg. infraction penalty', 'Collisions with pedestrians', 'Collisions with vehicles', 'Collisions with layout', 'Red lights infractions', 'Stop sign infractions', 'Off-road infractions', 'Route deviations', 'Route timeouts', 'Agent blocked']
        entry_status = 'Finished'
        eligible = True
        route_records = data['_checkpoint']['records']
        progress = data['_checkpoint']['progress']
        if progress[1] != total_routes:
            raise Exception('Critical error with the route registry.')
        if len(route_records) != total_routes or progress[0] != progress[1]:
            entry_status = 'Finished with missing data'
            eligible = False
        else:
            for route in route_records:
                route_status = route['status']
                if 'Agent' in route_status:
                    entry_status = 'Finished with agent errors'
                    break
        data['entry_status'] = entry_status
        data['eligible'] = eligible
        save_dict(endpoint, data)

    @staticmethod
    def save_sensors(sensors, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()
        if not data['sensors']:
            data['sensors'] = sensors
            save_dict(endpoint, data)

    @staticmethod
    def save_entry_status(entry_status, eligible, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()
        data['entry_status'] = entry_status
        data['eligible'] = eligible
        save_dict(endpoint, data)

    @staticmethod
    def clear_record(endpoint):
        if not endpoint.startswith(('http:', 'https:', 'ftp:')):
            with open(endpoint, 'w') as fd:
                fd.truncate(0)

