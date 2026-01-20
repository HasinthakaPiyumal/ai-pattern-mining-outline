# Cluster 22

class EmulatorDockerBridge:

    def __init__(self):
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        self.queue_out = self.manager.Queue()
        self.pool = mp.Pool(1)
        self.consumer_messages = None
        self.emulator_docker_log_conector = EmulatorDockerLogConnector(mp.Pool(1), self.manager.Queue(), self.queue_out)

    def start(self):
        print(f'-Start docker bridge-')
        self.emulator_docker_log_conector.start()
        self.consumer_messages = self.pool.map_async(__consumer__, ((self.queue, self.emulator_docker_log_conector.log_queue, self.queue_out),))
        self.__internal_send__(DockerMessage(docker_command='start'))

    def stop_server(self):
        self.__internal_send__(DockerMessage(docker_command='request_stop'))
        self.consumer_messages.wait(timeout=3)

    def request_skills(self):
        self.__internal_send__(DockerMessage(docker_command='request_skills', message='clai skills'))

    def select_skill(self, skill_name):
        self.__internal_send__(DockerMessage(docker_command='select_skill', message=f'clai activate {skill_name}'))

    def unselect_skill(self, skill_name):
        self.__internal_send__(DockerMessage(docker_command='unselect_skill', message=f'clai deactivate {skill_name}'))

    def send_message(self, message: str):
        self.__internal_send__(DockerMessage(docker_command='send_message', message=message))

    def refresh_files(self):
        self.__internal_send__(DockerMessage(docker_command='refresh', message='clai reload'))

    def __internal_send__(self, event: DockerMessage):
        try:
            print(f'sending to queue -> {event}')
            self.queue.put(event)
        except Exception as err:
            print(f'error sending: {err}')

    def retrieve_message(self) -> Optional[DockerReply]:
        try:
            message = self.queue_out.get(block=False)
            return message
        except:
            return None

