# Cluster 8

def test_should_return_the_start_server_command_when_received_clai_start(mocker):
    with pytest.raises(SystemExit):
        mock_override_last_command = mocker.patch('clai.process_command.override_last_command')
        process_command(ANY_ID, ANY_USER, COMMAND_START_SERVER)
        mock_override_last_command.assert_called_once_with(f'\n{START_SERVER_COMMAND_TO_EXECUTE}\n')

def process_command(command_id, user_name, command_to_check):
    pending_commands = False
    if command_to_check == COMMAND_START_SERVER:
        command_accepted_by_the_user = START_SERVER_COMMAND_TO_EXECUTE
        print(Colorize().info().append('CLAI Starting. CLAI could take a while to start replying').to_console())
    elif command_to_check.strip() == COMMAND_BASE.strip():
        command_accepted_by_the_user = NOOP_COMMAND
        print(create_message_server_runing())
        print(create_message_help().description)
    else:
        command_accepted_by_the_user, pending_commands = process_command_from_user(command_id, user_name, command_to_check)
    override_last_command('\n' + command_accepted_by_the_user + '\n')
    if not pending_commands:
        sys.exit(1)

class DemoAgent(Agent):

    def get_next_action(self, state: State) -> Union[Action, List[Action]]:
        logger.info('This is my agent')
        if state.command == 'ls':
            return Action(suggested_command='ls -la', description='This is a demo sample that helps to execute the command in better way.', confidence=1)
        if state.command == 'pwd':
            return [Action(suggested_command='ls -la', description='This is a demo sample that helps to execute the command in better way.', confidence=1), Action(suggested_command='pwd -P', description='This is a demo sample that helps to execute the command in better way.', confidence=1)]
        if state.previous_execution and state.previous_execution.command == 'ls -4':
            return Action(suggested_command='ls -a', execute=True, confidence=1)
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        if state.command.startswith('ls') and state.result_code != '0':
            return Action(description=Colorize().append(f'Are you sure that this command is correct?({state.result_code})\n').warning().append(f'Try man ls for more info ').to_console(), confidence=1)
        return Action(suggested_command=state.command)

def create_message_server_runing() -> str:
    colorize = Colorize()
    if check_if_process_running():
        colorize.complete().append(f'The server is running')
    else:
        colorize.warning().append(f'The server is not running')
    return colorize.to_console()

def create_message_help() -> Action:
    text = Colorize().info().append('CLAI usage:\nclai [help] [skills [-v]] [orchestrate [name]] [activate [skill_name]] [deactivate [skill_name]] [manual | automatic] [install [name | url]] \n\nhelp           Print help and usage of clai.\nskills         List available skills. Use -v For a verbose description of each skill.\norchestrate    Activate the orchestrator by name. If name is empty, list available orchestrators.\nactivate       Activate the named skill.\ndeactivate     Deactivate the named skill.\nmanual         Disables automatic execution of commands without operator confirmation.\nauto           Enables automatic execution of commands without operator confirmation.\ninstall        Installs a new skill. The required argument may be a local file path\n               to a skill plugin folder, or it may be a URL to install a skill plugin \n               over a network connection.\n').to_console()
    return Action(suggested_command=':', description=text, execute=True)

def check_if_process_running() -> bool:
    return os.system('ps -Ao args | grep "[c]lai-run" > /dev/null 2>&1') == 0

class ClaiHelpCommandRunner(CommandRunner):

    def execute(self, state: State) -> Action:
        return create_message_help()

class ClaiReloadCommandRunner(CommandRunner):

    def __init__(self, agent_datasource: AgentDatasource):
        self.agent_datasource = agent_datasource

    def execute(self, state: State) -> Action:
        self.agent_datasource.reload()
        text = Colorize().complete().append('Plugins reloaded.\n').to_console()
        return Action(suggested_command=':', execute=True, description=text, origin_command=state.command)

class NLC2CMD(Agent):

    def __init__(self):
        super(NLC2CMD, self).__init__()
        self.service = Service()

    def get_next_action(self, state: State) -> Action:
        command = state.command
        data, confidence = self.service(command)
        response = data['text']
        return Action(suggested_command=NOOP_COMMAND, execute=True, description=Colorize().info().append(response).to_console(), confidence=confidence)

class HowDoIAgent(Agent):

    def __init__(self):
        super(HowDoIAgent, self).__init__()
        inifile_path = os.path.join(str(Path(__file__).parent.absolute()), 'config.ini')
        self.store = Datastore(inifile_path)
        self.questionIdentifier = QuestionDetection()

    def compute_simple_token_similarity(self, src_sequence, tgt_sequence):
        src_tokens = set([x.lower().strip() for x in src_sequence.split()])
        tgt_tokens = set([x.lower().strip() for x in tgt_sequence.split()])
        return len(src_tokens & tgt_tokens) / len(src_tokens)

    def get_next_action(self, state: State) -> Action:
        logger.info('================== In HowDoI Bot:get_next_action ========================')
        logger.info('State:\n\tCommand: {}\n\tError Code: {}\n\tStderr: {}'.format(state.command, state.result_code, state.stderr))
        logger.info('============================================================================')
        is_question = self.questionIdentifier.is_question(state.command)
        if not is_question:
            return Action(suggested_command=state.command, confidence=0.0)
        apis: OrderedDict = self.store.get_apis()
        helpWasFound = False
        for provider in apis:
            if provider == 'manpages':
                logger.info(f"Skipping search provider 'manpages'")
                continue
            thisAPI: Provider = apis[provider]
            if not thisAPI.can_run_on_this_os():
                logger.info(f"Skipping search provider '{provider}'")
                logger.info(f'==> Excluded on platforms: {str(thisAPI.get_excludes())}')
                continue
            logger.info(f"Processing search provider '{provider}'")
            if thisAPI.has_variants():
                logger.info(f'==> Has search variants: {str(thisAPI.get_variants())}')
                variants: List = thisAPI.get_variants()
            else:
                logger.info(f'==> Has no search variants')
                variants: List = [None]
            for variant in variants:
                if variant is not None:
                    logger.info(f"==> Searching variant '{variant}'")
                    data = self.store.search(state.command, service=provider, size=1, searchType=variant)
                else:
                    data = self.store.search(state.command, service=provider, size=1)
                if data:
                    logger.info(f'==> Success!!! Found a result in the {thisAPI}')
                    searchResult = thisAPI.extract_search_result(data)
                    manpages = self.store.search(searchResult, service='manpages', size=5)
                    if manpages:
                        logger.info('==> Success!!! found relevant manpages.')
                        command = manpages['commands'][-1]
                        confidence = manpages['dists'][-1]
                        logger.info('==> Command: {} \t Confidence:{}'.format(command, confidence))
                        suggested_command = 'man {}'.format(command)
                        description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I did little bit of Internet searching for you, ').append(f'and found this in the {thisAPI}:\n').info().append(thisAPI.get_printable_output(data)).warning().append('Do you want to try: man {}'.format(command)).to_console()
                        helpWasFound = True
                        break
            if helpWasFound:
                break
        if not helpWasFound:
            logger.info('Failure: Unable to be helpful')
            logger.info('============================================================================')
            suggested_command = NOOP_COMMAND
            description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f"Sorry. It looks like you have stumbled across a problem that even the Internet doesn't have answer to.\n").info().append(f'Have you tried turning it OFF and ON again. ;)').to_console()
            confidence = 0.0
        return Action(suggested_command=suggested_command, description=description, confidence=confidence)

class MsgCodeAgent(Agent):

    def __init__(self):
        super(MsgCodeAgent, self).__init__()
        inifile_path = os.path.join(str(Path(__file__).parent.absolute()), 'config.ini')
        self.store = Datastore(inifile_path)

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        logger.info('==================== In zMsgCode Bot:post_execute ============================')
        logger.info('State:\n\tCommand: {}\n\tError Code: {}\n\tStderr: {}'.format(state.command, state.result_code, state.stderr))
        logger.info('============================================================================')
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        stderr = state.stderr.strip()
        matches = re.compile(REGEX_ZMSG).match(stderr)
        if matches is None:
            logger.info(f"No Z message ID found in '{stderr}'")
            return Action(suggested_command=state.command)
        logger.info(f"Analyzing error message '{matches[0]}'")
        msgid: str = matches[2]
        helpWasFound = False
        bpx_matches: List[str] = self.__search(matches[0], REGEX_BPX)
        if bpx_matches is not None:
            reason_code: str = bpx_matches[1]
            logger.info(f'==> Reason Code: {reason_code}')
            result: CompletedProcess = subprocess.run(['bpxmtext', reason_code], stdout=subprocess.PIPE)
            if result.returncode == 0:
                messageText = result.stdout.decode('UTF8')
                if self.__search(messageText, REGEX_BPX_BADANSWER) is None:
                    suggested_command = state.command
                    description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I asked bpxmtext about that message:\n').info().append(messageText).warning().to_console()
                    helpWasFound = True
        if not helpWasFound:
            kc_api: Provider = self.store.get_apis()['ibm_kc']
            if kc_api is not None and kc_api.can_run_on_this_os():
                data = self.store.search(msgid, service='ibm_kc', size=1)
                if data:
                    logger.info(f'==> Success!!! Found information for msgid {msgid}')
                    suggested_command = state.command
                    description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I looked up {msgid} in the IBM KnowledgeCenter for you:\n').info().append(kc_api.get_printable_output(data)).warning().to_console()
                    helpWasFound = True
        if not helpWasFound:
            logger.info('Failure: Unable to be helpful')
            logger.info('============================================================================')
            suggested_command = NOOP_COMMAND
            description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f"I couldn't find any help for message code '{msgid}'\n").info().to_console()
        return Action(suggested_command=suggested_command, description=description, confidence=1.0)

    def __search(self, target: str, regex_list: List[str]) -> List[str]:
        """Check all possible regexes in a list, return the first match encountered"""
        for regex in regex_list:
            this_match = re.compile(regex).match(target)
            if this_match is not None:
                return this_match
        return None

class TELLINA(Agent):

    def __init__(self):
        super(TELLINA, self).__init__()

    def get_next_action(self, state: State) -> Action:
        command = state.command
        try:
            endpoint_comeback = requests.post(tellina_endpoint, json={'command': command}).json()
            response = 'Try >> ' + endpoint_comeback['response']
            confidence = float(endpoint_comeback['confidence'])
            return Action(suggested_command=NOOP_COMMAND, execute=True, description=Colorize().info().append(response).to_console(), confidence=confidence)
        except Exception as ex:
            return [{'text': 'Method failed with status ' + str(ex)}, 0.0]

class Service:

    def __init__(self):
        self.state = {'ready_flag': False, 'threshold': 0.9, 'current_intent': None, 'current_branch': None, 'commit_details': None, 'has_done_add': False, 'has_done_commit': False}
        self.gh_session = requests.Session()
        self.gh_session.auth = (_github_username, _github_access_token)

    def __call__(self, *args, **kwargs):
        command = args[0]
        confidence = 0.0
        try:
            response = requests.post(_rasa_service, json={'text': command}).json()
            confidence = response['intent']['confidence']
            if confidence > self.state['threshold']:
                self.state['current_intent'] = response['intent']['name']
        except Exception as ex:
            print('Method failed with status ' + str(ex))
        if self.state['current_intent'] == 'commit':
            if not self.state['ready_flag']:
                self.state['ready_flag'] = True
                temp_description = 'Ready to {}. Press execute to continue.'.format(self.state['current_intent'])
                return [Action(suggested_command='git status | tee {}'.format(_path_to_log_file), execute=True, description=None, confidence=1.0), Action(suggested_command=NOOP_COMMAND, execute=True, description=Colorize().info().append(temp_description).to_console(), confidence=1.0)]
        if command == 'execute':
            now = datetime.now()
            commit_message = now.strftime('%d/%m/%Y %H:%M:%S')
            stdout = open(_path_to_log_file).read().split('\n')
            commit_description = ''
            current_branch = None
            for line in stdout:
                if line.startswith('On branch'):
                    current_branch = line.split()[-1]
                if line.strip().startswith('modified:'):
                    commit_description += line.strip() + ' + '
            if commit_description:
                self.state['commit_details'] = commit_description
            if current_branch:
                self.state['current_branch'] = current_branch
            commit_description = self.state['commit_details']
            respond = Action(suggested_command='git commit -m "{}" -m "{}";'.format(commit_message, commit_description), execute=False, description=None, confidence=1.0)
            if self.state['has_done_add']:
                return respond
            else:
                return [Action(suggested_command='git add -A', execute=True, description='Adding untracked files...', confidence=1.0), respond]
        if self.state['current_intent'] == 'push':
            return Action(suggested_command='git push', execute=False, description=None, confidence=confidence)
        if self.state['current_intent'] == 'merge':
            merge_url = '{}/pulls'.format(_github_url)
            source = None
            target = 'master'
            for entity in response['entities']:
                if entity['entity'] == 'source':
                    source = entity['value']
                if entity['entity'] == 'target':
                    target = entity['value']
            if not source:
                source = self.state['current_branch']
            payload = {'title': 'Dummy PR from gitbot', 'body': 'Example from the `gitbot` screencast. This will not be merged.', 'head': source, 'base': target}
            ping_github = json.loads(self.gh_session.post(merge_url, json=payload).text)
            return Action(suggested_command=NOOP_COMMAND, execute=True, description='Success. Created PR {}'.format(ping_github['number']), confidence=confidence)
        if self.state['current_intent'] == 'comment':
            idx = 0
            comment = None
            for entity in response['entities']:
                if entity['entity'] == 'id':
                    idx = entity['value']
                if entity['entity'] == 'comment':
                    comment = entity['value']
            idx = command.split('<')[-1].split('>')[0]
            comment_url = '{}/issues/{}/comments'.format(_github_url, idx)
            payload = {'body': comment}
            ping_github = json.loads(self.gh_session.post(comment_url, json=payload).text)
            return Action(suggested_command=NOOP_COMMAND, execute=True, description='Success', confidence=confidence)

    def parse_command(self, command: str, stdout: str):
        if command.startswith('git checkout'):
            self.state['current_branch'] = None
        if command.startswith('git add'):
            self.state['has_done_add'] = True
        if command.startswith('commit'):
            self.state['has_done_add'] = False
            self.state['has_done_commit'] = True

class DATAXPLORE(Agent):

    def __init__(self):
        super(DATAXPLORE, self).__init__()

    def get_next_action(self, state: State) -> Action:
        command = state.command
        try:
            logger.info('Command passed in dataxplore: ' + command)
            commandStr = str(command)
            commandTokenized = commandStr.split(' ')
            if len(commandTokenized) == 2:
                if commandTokenized[0] == 'summarize':
                    fileName = commandTokenized[1]
                    csvFile = fileName.split('.')
                    if len(csvFile) == 2:
                        if csvFile[1] == 'csv':
                            path = os.path.abspath(fileName)
                            data = pd.read_csv(path)
                            df = pd.DataFrame(data)
                            response = df.describe().to_string()
                        else:
                            response = 'We currently support only csv files. Please, Try >> clai dataxplore summarize csvFileLocation '
                    else:
                        response = 'Not a supported file format. Please, Try >> clai dataxplore summarize csvFileLocation '
                elif commandTokenized[0] == 'plot':
                    fileName = commandTokenized[1]
                    csvFile = fileName.split('.')
                    if len(csvFile) == 2:
                        if csvFile[1] == 'csv':
                            plt.close('all')
                            path = os.path.abspath(fileName)
                            data = pd.read_csv(path, index_col=0, parse_dates=True)
                            data.plot()
                            plt.savefig('/tmp/claifigure.png')
                            im = Image.open('/tmp/claifigure.png')
                            im.show()
                            response = 'Please, check the popup for figure.'
                        else:
                            response = 'We currently support only csv files. Please, Try >> clai dataxplore plot csvFileLocation '
                    else:
                        response = 'Not a supported file format. Please, Try >> clai dataxplore plot csvFileLocation '
                else:
                    response = 'Try >> clai dataxplore function fileLocation '
            else:
                response = 'Few parts missing. Please, Try >> clai dataxplore function fileLocation '
            confidence = 0.0
            return Action(suggested_command=NOOP_COMMAND, execute=True, description=Colorize().info().append(response).to_console(), confidence=confidence)
        except Exception as ex:
            return [{'text': 'Method failed with status ' + str(ex)}, 0.0]

class IBMCloud(Agent):

    def __init__(self):
        super(IBMCloud, self).__init__()
        self.exe = KubeExe()
        self.intents = ['deploy to kube', 'build yaml', 'run Dockerfile']

    def get_next_action(self, state: State) -> Action:
        if state.command in self.intents:
            self.exe.set_goal(state.command)
            plan = self.exe.get_plan()
            if plan:
                logger.info('####### log plan inside ibmcloud ########')
                logger.info(plan)
                action_list = []
                for action in plan:
                    action_object = self.exe.execute_action(action)
                    if action_object:
                        action_list.append(action_object)
                return action_list
            else:
                return Action(suggested_command=NOOP_COMMAND, execute=True, description=Colorize().info().append('Sorry could not find a plan to help! :-(').to_console(), confidence=1.0)
        else:
            return Action(suggested_command=NOOP_COMMAND)

    def post_execute(self, state: State) -> Action:
        if state.result_code == '0':
            self.exe.parse_command(state.command, stdout='')
        return Action(suggested_command=NOOP_COMMAND)

class HelpMeAgent(Agent):

    def __init__(self):
        super(HelpMeAgent, self).__init__()
        inifile_path = os.path.join(str(Path(__file__).parent.absolute()), 'config.ini')
        self.store = Datastore(inifile_path)

    def compute_simple_token_similarity(self, src_sequence, tgt_sequence):
        src_tokens = set([x.lower().strip() for x in src_sequence.split()])
        tgt_tokens = set([x.lower().strip() for x in tgt_sequence.split()])
        return len(src_tokens & tgt_tokens) / len(src_tokens)

    def compute_confidence(self, query, forum, manpage):
        """
        Computes the confidence based on query, stack-exchange post answer and manpage

        Algorithm:
            1. Compute token-wise similarity b/w query and forum text
            2. Compute token-wise similarity b/w forum text and manpage description
            3. Return product of two similarities


        Args:
            query (str): standard error captured in state variable
            forum (str): answer text from most relevant stack exchange post w.r.t query
            manpage (str): manpage description for most relevant manpage w.r.t. forum

        Returns:
             confidence (float): confidence on the returned manpage w.r.t. query
        """
        query_forum_similarity = self.compute_simple_token_similarity(query, forum[0]['Content'])
        forum_manpage_similarity = self.compute_simple_token_similarity(forum[0]['Answer'], manpage)
        confidence = query_forum_similarity * forum_manpage_similarity
        return confidence

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        logger.info('==================== In Helpme Bot:post_execute ============================')
        logger.info('State:\n\tCommand: {}\n\tError Code: {}\n\tStderr: {}'.format(state.command, state.result_code, state.stderr))
        logger.info('============================================================================')
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        apis: OrderedDict = self.store.get_apis()
        helpWasFound = False
        for provider in apis:
            if provider == 'manpages':
                logger.info(f"Skipping search provider 'manpages'")
                continue
            thisAPI: Provider = apis[provider]
            if not thisAPI.can_run_on_this_os():
                logger.info(f"Skipping search provider '{provider}'")
                logger.info(f'==> Excluded on platforms: {str(thisAPI.get_excludes())}')
                continue
            logger.info(f"Processing search provider '{provider}'")
            if thisAPI.has_variants():
                logger.info(f'==> Has search variants: {str(thisAPI.get_variants())}')
                variants: List = thisAPI.get_variants()
            else:
                logger.info(f'==> Has no search variants')
                variants: List = [None]
            for variant in variants:
                if variant is not None:
                    logger.info(f"==> Searching variant '{variant}'")
                    data = self.store.search(state.stderr, service=provider, size=1, searchType=variant)
                else:
                    data = self.store.search(state.stderr, service=provider, size=1)
                if data:
                    apiString = str(thisAPI)
                    if variant is not None:
                        apiString = f"{apiString} '{variant}' variant"
                    logger.info(f'==> Success!!! Found a result in the {apiString}')
                    searchResult = thisAPI.extract_search_result(data)
                    manpages = self.store.search(searchResult, service='manpages', size=5)
                    if manpages:
                        logger.info('==> Success!!! found relevant manpages.')
                        command = manpages['commands'][-1]
                        confidence = manpages['dists'][-1]
                        confidence = 1.0
                        logger.info('==> Command: {} \t Confidence:{}'.format(command, confidence))
                        suggested_command = 'man {}'.format(command)
                        description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I did little bit of Internet searching for you, ').append(f'and found this in the {thisAPI}:\n').info().append(thisAPI.get_printable_output(data)).warning().append('Do you want to try: man {}'.format(command)).to_console()
                        helpWasFound = True
                        break
            if helpWasFound:
                break
        if not helpWasFound:
            logger.info('Failure: Unable to be helpful')
            logger.info('============================================================================')
            suggested_command = NOOP_COMMAND
            description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f"Sorry. It looks like you have stumbled across a problem that even the Internet doesn't have answer to.\n").info().append(f'Have you tried turning it OFF and ON again. ;)').to_console()
            confidence = 0.0
        return Action(suggested_command=suggested_command, description=description, confidence=confidence)

class FixBot(Agent):
    """
    Fixes the last executed command by running it through the `thefuck` plugin
    """

    def __init__(self):
        super(FixBot, self).__init__()
        pass

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        cmd = str(state.command)
        stderr = str(state.stderr)
        try:
            settings.init()
            cmd = Command(cmd, stderr)
            cmd_corrected = get_corrected_commands(cmd)
            cmd_to_run = next(cmd_corrected).script
        except Exception:
            return Action(suggested_command=state.command, confidence=0.1)
        else:
            return Action(description=Colorize().info().append('Maybe you want to try: {}'.format(cmd_to_run)).to_console(), confidence=0.8)

def get_command_tldr(cmd):
    if not TLDR_AVAILABLE:
        return ''
    cmd_tldr = tldr.get_page(cmd)
    if cmd_tldr is None:
        return ''
    description = Colorize()
    for i, line in enumerate(cmd_tldr):
        line = line.rstrip().decode('utf-8')
        if i == 0:
            description.append('-' * 50 + '\n')
        if len(line) < 1:
            description.append('\n')
        elif line[0] == '#':
            line = line[1:]
            description.warning().append(line.strip() + '\n')
        elif line[0] == '>':
            line = ' ' + line[1:]
            description.normal().append(line.strip() + '\n')
        elif line[0] == '-':
            description.normal().append(line.strip() + '\n')
        elif line[0] == '`':
            line = ' ' + line[1:-1]
            description.info().append(line.strip() + '\n')
    description.normal().append('summary provided by tldr package\n')
    description.normal().append('-' * 50 + '\n')
    return description.to_console()

