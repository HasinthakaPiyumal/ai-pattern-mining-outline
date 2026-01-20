# Cluster 16

def test_should_return_any_action_when_everything_works_correctly(mocker):
    mocker.patch.object(ClaiClient, 'send', return_value=ANY_NO_ACTION, autospec=True)
    action = send_command(ANY_ID, ANY_USER, ANY_NO_ACTION.origin_command)
    assert action == ANY_NO_ACTION

def send_command(command_id: str, user_name: str, command_to_check: str, host: str=LOCALHOST, port: int=DEFAULT_PORT) -> Action:
    StateDTO.update_forward_refs()
    command_to_execute = StateDTO(command_id=command_id, user_name=user_name, command=command_to_check)
    clai_client = ClaiClient(host=host, port=port)
    return clai_client.send(command_to_execute)

def test_should_return_a_valid_action_when_socket_crash(mocker):
    mocker.patch.object(SocketClientConnector, 'send', side_effect=socket.error(), autospec=True)
    action = send_command(ANY_ID, ANY_USER, ANY_NO_ACTION.origin_command)
    assert action == ANY_NO_ACTION

def map_processes(processes) -> List[Process]:
    return list(map(lambda _: Process(name=_['name']), processes))

def obtain_last_processes(user_name):
    process_changes = []
    if PLATFORM not in ('zos', 'os390'):
        for process in PSUTIL.process_iter(attrs=['pid', 'name', 'username', 'create_time']):
            process_changes.append(process.info)
    else:
        pass
    porcess_changes = list(filter(lambda _: _['username'] == user_name, process_changes))
    porcess_changes.sort(key=lambda _: _['create_time'], reverse=True)
    return map_processes(process_changes[EXCLUDE_OWN_PROCESS:SIZE_PROCESS])

def post_process_command(command_id: str, user_name: str, cmd_result: str, stderr: str):
    process_changes = obtain_last_processes(user_name)
    post_command_action = send_command_post_execute(command_id=command_id, user_name=user_name, result_code=cmd_result, stderr=stderr, processes=ProcessesValues(last_processes=process_changes))
    if stderr:
        print(stderr)
    if post_command_action and post_command_action.description:
        print(post_command_action.description)

def send_command_post_execute(command_id: str, user_name: str, result_code: str, stderr: str, processes: ProcessesValues, host: str=LOCALHOST, port: int=DEFAULT_PORT) -> Action:
    post_execute_state = StateDTO(command_id=command_id, user_name=user_name, result_code=result_code, stderr=stderr, processes=processes)
    clai_client = ClaiClient(host=host, port=port)
    return clai_client.send(post_execute_state)

class ClaiClient:

    def __init__(self, host: str=LOCALHOST, port: int=DEFAULT_PORT, connector: ClientConnector=None):
        self.connector = connector
        if not connector:
            self.connector = SocketClientConnector(host=host, port=port)
        self.port = port
        self.host = host

    def send(self, message: StateDTO) -> Action:
        try:
            return self.connector.send(message)
        except Exception as exception:
            logger.info(f'error: {exception}')
            return Action(origin_command=message.command, suggested_command=message.command)

class ClaiServer:

    def __init__(self, server_status_datasource: ServerStatusDatasource=current_status_datasource, connector: ServerConnector=SocketServerConnector(current_status_datasource), agent_datasource=AgentDatasource()):
        self.connector = connector
        self.agent_datasource = agent_datasource
        self.server_status_datasource = server_status_datasource
        self.remote_storage = ActionRemoteStorage()
        self.message_handler = MessageHandler(server_status_datasource, agent_datasource=agent_datasource)
        self.stats_tracker = StatsTracker()

    def init_server(self):
        self.message_handler.init_server()
        self.server_status_datasource.running = True
        self.remote_storage.start(self.agent_datasource)
        self.stats_tracker.start(self.agent_datasource)

    @staticmethod
    def serialize_message(data) -> State:
        StateDTO.update_forward_refs()
        dto = StateDTO(**json.loads(data))
        return State(command_id=dto.command_id, user_name=dto.user_name, command=dto.command, root=dto.root, processes=dto.processes, file_changes=dto.file_changes, network=dto.network, result_code=dto.result_code, stderr=dto.stderr)

    def create_socket(self, host, port):
        self.connector.create_socket(host, port)

    def listen_client_sockets(self):
        self.connector.loop(self.process_message)
        self.remote_storage.wait()
        self.stats_tracker.wait()

    def process_message(self, received_data: bytes) -> Action:
        message = self.serialize_message(received_data)
        return self.message_handler.process_message(message)

