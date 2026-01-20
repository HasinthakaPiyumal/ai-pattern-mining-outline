# Cluster 24

class ContractSkills:

    def is_auto_mode(self):
        return True

    def get_skill_name(self):
        raise NotImplementedError('You should provide the commands to execute.')

    def get_commands_to_execute(self):
        raise NotImplementedError('You should provide the commands to execute.')

    def get_commands_expected(self):
        raise NotImplementedError('You should provide the commands expected.')

    @pytest.mark.dependency()
    def test_install(self, my_clai_module):
        if self.is_auto_mode():
            execute_cmd(my_clai_module, 'clai auto')
        skill_name = self.get_skill_name()
        execute_cmd(my_clai_module, 'clai deactivate gpt3')
        command_select = f'clai activate {skill_name}'
        command_executed = execute_cmd(my_clai_module, command_select)
        assert f'\x1b[32m {skill_name} (Installed)' in command_executed, f'Skill {skill_name} not found installed. Output: {command_executed}'

    @pytest.mark.dependency(depends=['test_install'])
    def test_skill_values(self, my_clai_module, command, command_expected):
        command_executed = execute_cmd(my_clai_module, command)
        assert command_expected in command_executed, f'Expected: {command_expected}, Received: {command_executed}'

def execute_cmd(container, command):
    socket = container.exec_run(cmd='bash -l', stdin=True, tty=True, privileged=True, socket=True)
    wait_server_is_started()
    command_to_exec = command + '\n'
    socket.output._sock.send(command_to_exec.encode())
    data = read(socket)
    sleep(1)
    socket.output._sock.send(b'exit\n')
    return str(data)

def get_base_path():
    root_path = os.getcwd()
    print(root_path)
    if 'test_integration' in root_path:
        return '../'
    return '.'

def test_install_should_finish_correctly(my_clai):
    install_output = execute_cmd(my_clai, 'sudo ./install.sh --unassisted --demo')
    assert INSTALL_CORRECTLY_MESSAGE in install_output

def test_install_should_modify_correct_startup_files(my_clai):
    execute_cmd(my_clai, 'sudo ./install.sh --unassisted --demo')
    files = my_clai.get_files('/root')
    bashrc_output = str(files['root/.bashrc'])
    bash_profile_output = str(files['root/.bash_profile'])
    assert '# CLAI setup' in bashrc_output
    assert '# CLAI setup' in bash_profile_output
    assert '# End CLAI setup' in bashrc_output
    assert '# End CLAI setup' in bash_profile_output

def test_uninstall_should_return_the_correct_uninstall_message(my_clai):
    execute_cmd(my_clai, 'sudo ./install.sh --unassisted --demo')
    uninstall_output = execute_cmd(my_clai, 'sudo ./uninstall.sh')
    sleep(2)
    print(uninstall_output)
    assert UNINSTALL_CORRECTLY_MESSAGE in uninstall_output

def test_uninstall_should_return_bash_files_to_previous_state(my_clai):
    files = my_clai.get_files('/root')
    bashrc_original = str(files['root/.bashrc'])
    bash_profile_original = str(files['root/.bash_profile'])
    execute_cmd(my_clai, 'sudo ./install.sh --unassisted --demo')
    execute_cmd(my_clai, 'sudo ./uninstall.sh')
    sleep(2)
    files = my_clai.get_files('/root')
    bashrc_after_uninstall = str(files['root/.bashrc'])
    bash_profile_after_uninstall = str(files['root/.bash_profile'])
    assert bashrc_after_uninstall == bashrc_original
    assert bash_profile_original == bash_profile_after_uninstall

def wait_server_is_started():
    sleep(2)

def read(socket, chunk_readed=None):
    data = ''
    try:
        socket.output._sock.recv(1)
        while True:
            data_bytes = socket.output._sock.recv(4096)
            if not data_bytes:
                break
            chunk = data_bytes.decode('utf8', errors='ignore')
            if chunk.endswith(']# '):
                if data:
                    break
            else:
                data += chunk
            if chunk_readed:
                chunk_readed(chunk)
            data = data[-MAX_SIZE_STDOUT:]
    except Exception as exception:
        print(f'error: {exception}')
    return data

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

def __get_image(docker_client):
    path = get_base_path()
    print(f'Building {path}')
    try:
        image, logs = docker_client.images.build(path=path, dockerfile='./clai/emulator/docker/centos/Dockerfile', rm=True)
        for log in logs:
            print(log)
    except:
        traceback.print_exc()
    return image

def __start_docker():
    docker_client = docker.from_env()
    image = __get_image(docker_client)
    docker_container = docker_client.containers.run(image=image.id, detach=True)
    my_clai = Container(docker_container)
    wait_for_callable('Waiting for container to be ready', my_clai.ready)
    print(f'container run {my_clai.status} {my_clai.name}')
    return my_clai

def copy_files(my_clai):
    old_path = os.getcwd()
    print(f'Building {old_path}')
    srcpath = os.path.join(get_base_path(), 'clai', 'server', 'plugins')
    os.chdir(srcpath)
    tar = tarfile.open('temp.tar', mode='w')
    try:
        tar.add('.', recursive=True)
    finally:
        tar.close()
    data = open('temp.tar', 'rb').read()
    destdir = os.path.join(os.path.expanduser('/opt/local/share'), 'clai', 'bin', 'clai', 'server', 'plugins')
    my_clai._container.put_archive(destdir, data)
    os.chdir(old_path)
    print('Done the refresh')

def __consumer__(args):
    queue, log_queue, queue_out = args
    my_clai = None
    socket = None
    print('starting reading from the queue')
    while True:
        docker_message: DockerMessage = queue.get()
        if docker_message is None:
            break
        print(f'message_received: {docker_message.docker_command}:{docker_message.message}')
        if docker_message.docker_command == 'start':
            my_clai = __start_docker()
            log_queue.put(DockerMessage(docker_command='start_logger', message=my_clai.name))
        elif docker_message.docker_command == 'send_message' or docker_message.docker_command == 'request_skills' or docker_message.docker_command == 'unselect_skill' or (docker_message.docker_command == 'refresh') or (docker_message.docker_command == 'select_skill'):
            if my_clai:
                print(f'socket {socket}')
                if not socket:
                    socket = my_clai.exec_run(cmd='bash -l', stdin=True, tty=True, privileged=True, socket=True)
                    wait_server_is_started()
                if docker_message.docker_command == 'refresh':
                    copy_files(my_clai)
                command_to_exec = docker_message.message + '\n'
                socket.output._sock.send(command_to_exec.encode())
                stdout = read(socket)
                reply = None
                if docker_message.docker_command == 'request_skills' or docker_message.docker_command == 'unselect_skill':
                    reply = DockerReply(docker_reply='skills', message=stdout)
                elif docker_message.docker_command == 'send_message':
                    socket.output._sock.send('clai last-info\n'.encode())
                    info = read(socket)
                    reply = DockerReply(docker_reply='reply_message', message=stdout, info=info)
                if reply:
                    queue_out.put(reply)
        elif docker_message == 'request_stop':
            if my_clai:
                my_clai.kill()
                break
        queue.task_done()
    print('----STOP CONSUMING-----')

def __log_consumer__(args):
    queue, queue_out = args
    my_clai: Container = None
    socket = None
    print('starting reading the log queue')
    while True:
        docker_message: DockerMessage = queue.get()
        if docker_message.docker_command == 'start_logger':
            docker_client = docker.from_env()
            docker_container = docker_client.containers.get(docker_message.message)
            my_clai = Container(docker_container)
        if my_clai:
            if not socket:
                socket = my_clai.exec_run(cmd='bash -l', stdin=True, tty=True, privileged=True, socket=True)
                wait_server_is_started()
            socket.output._sock.send('clai "none" tail -f /var/tmp/app.log\n'.encode())
            read(socket, lambda chunk: queue_out.put(DockerReply(docker_reply='log', message=chunk)))
        queue.task_done()
        queue.put(DockerMessage(docker_command='log'))

