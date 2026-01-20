# Cluster 12

@unittest.skip('Only for local testing')
class SearchAgentTest(unittest.TestCase):

    @classmethod
    def set_up_class(cls):
        _agent = HowDoIAgent()
        cls.agent = _agent

    def print_and_verify(self, question, answer):
        state = State(user_name='tester', command_id='0', command=question)
        action = self.agent.get_next_action(state=state)
        print(f'Input: {state.command}')
        print('===========================')
        print(f'Response: {action.suggested_command}')
        print('===========================')
        print(f'Explanation: {action.description}')
        self.assertEqual(answer, action.suggested_command)

    @unittest.skip('Only for local testing')
    def test_get_next_action_pwd_without_question(self):
        self.agent.init_agent()
        if OS_NAME in ('OS/390', 'Z/OS'):
            self.print_and_verify('pds', 'pds')
        else:
            self.print_and_verify('pds', None)

    @unittest.skip('Only for local testing')
    def test_get_next_action_pwd_with_question(self):
        self.agent.init_agent()
        if OS_NAME in ('OS/390', 'Z/OS'):
            self.print_and_verify('What is a pds?', 'man readlink')
        else:
            self.print_and_verify('What is pwd?', 'man pwd')

    @unittest.skip('Only for local testing')
    def test_get_next_action_sudo(self):
        self.agent.init_agent()
        self.print_and_verify('when to use sudo vs su?', 'man su')

    @unittest.skip('Only for local testing')
    def test_get_next_action_disk(self):
        self.agent.init_agent()
        question: str = 'find out disk usage per user?'
        if OS_NAME in ('OS/390', 'Z/OS'):
            self.print_and_verify(question, 'man du')
        else:
            self.print_and_verify(question, 'man df')

    @unittest.skip('Only for local testing')
    def test_get_next_action_zip(self):
        self.agent.init_agent()
        question: str = 'How to process gz files?'
        if OS_NAME in ('OS/390', 'Z/OS'):
            self.print_and_verify(question, 'man dnctl')
        else:
            self.print_and_verify(question, 'man gzip')

    @unittest.skip('Only for local testing')
    def test_get_next_action_pds(self):
        self.agent.init_agent()
        question: str = 'copy a PDS member?'
        if OS_NAME in ('OS/390', 'Z/OS'):
            self.print_and_verify(question, 'man tcsh')
        else:
            self.print_and_verify(question, 'man cmp')

class NLC2CMDCloudTest(unittest.TestCase):

    @classmethod
    def set_up_class(cls):
        cls.state = State(user_name='tester', command_id='0', command='show me the list of cloud tags', result_code='0')
        cls.agent = NLC2CMD()

    @unittest.skip('Local dev testing only')
    def test_get_next_action_cloud_login(self):
        self.state.command = 'how do i login'
        action = self.agent.get_next_action(state=self.state)
        print('Input: {}'.format(self.state.command))
        print('---------------------------')
        print('Explanation: {}'.format(action.description))
        self.assertEqual(NOOP_COMMAND, action.suggested_command)
        self.assertEqual('\x1b[95mTry >> ibmcloud login\x1b[0m', action.description)
        print('===========================')

    @unittest.skip('Local dev testing only')
    def test_get_next_action_cloud_help(self):
        self.state.command = 'help me'
        action = self.agent.get_next_action(state=self.state)
        print('Input: {}'.format(self.state.command))
        print('---------------------------')
        print('Explanation: {}'.format(action.description))
        self.assertEqual(NOOP_COMMAND, action.suggested_command)
        self.assertEqual('\x1b[95mTry >> ibmcloud help COMMAND\x1b[0m', action.description)
        print('===========================')

    @unittest.skip('Local dev testing only')
    def test_get_next_action_cloud_invite(self):
        self.state.command = 'I want to invite someone to my cloud'
        action = self.agent.get_next_action(state=self.state)
        print('Input: {}'.format(self.state.command))
        print('---------------------------')
        print('Explanation: {}'.format(action.description))
        self.assertEqual(NOOP_COMMAND, action.suggested_command)
        self.assertEqual('\x1b[95mTry >> ibmcloud account user-invite USER_EMAIL\x1b[0m', action.description)
        print('===========================')

def clai_power_disabled_state():
    return State(command_id=ANY_ID, user_name=ANY_NAME, command='clai manual')

class RetrievalAgentTest(unittest.TestCase):

    @classmethod
    def set_up_class(cls):
        cls.state = State(user_name='tester', command_id='0', command='./tmp_file.sh', result_code='1', stderr='Permission denied')
        cls.agent = HelpMeAgent()

    @unittest.skip('Need internet connection')
    def test_get_post_execute(self):
        action = self.agent.post_execute(state=self.state)
        print('Input: {}'.format(self.state.command))
        print('===========================')
        print('Response: {}'.format(action.suggested_command))
        print('===========================')
        print('Explanation: {}'.format(action.description))
        self.assertNotEqual(NOOP_COMMAND, action.suggested_command)

    @unittest.skip('Local testing')
    def test_get_forum(self):
        forum = self.agent.store.search('Permission denied', service='stack_exchange', size=1)
        self.assertEqual(1, len(forum))

    @unittest.skip('Local testing')
    def test_get_kc(self):
        kc_hits = self.agent.store.search('Permission denied', service='ibm_kc', size=1)
        print('Got this result from the KnowledgeCenter: ' + str(kc_hits))
        self.assertEqual(1, len(kc_hits))
        man_hits = self.agent.store.search(kc_hits[0]['summary'], service='manpages', size=10)
        print('Got this result from the Manpages service: ' + str(man_hits))
        self.assertEqual('connect', man_hits['commands'][-1])

    @unittest.skip('Local testing')
    def test_get_command(self):
        hits = self.agent.store.search("sudo allows a permitted user to execute a commandas the superuser or another user, as specified by the security policy.  The invoking user's real (not effective) user ID is used to determine the user name with which to query the security policy.", service='manpages', size=10)
        self.assertEqual('sudo', hits['commands'][-1])

def clai_unselect_state(plugin_to_select):
    return State(command_id=ANY_ID, user_name=ANY_NAME, command=f'clai unselect {plugin_to_select}')

@unittest.skipIf(OS_NAME not in ('OS/390', 'Z/OS'), 'Test only valid on z/OS')
class MsgCodeAgentTest(unittest.TestCase):

    @classmethod
    def set_up_class(cls):
        _agent = MsgCodeAgent()
        cls.agent = _agent

    @classmethod
    def try_command(cls, command_and_args: List[str]) -> Action:
        result: CompletedProcess = subprocess.run(command_and_args, encoding='UTF8', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command: str = None
        if isinstance(result.args, list):
            command = ' '.join(result.args)
        else:
            command = result.args
        result_code: str = str(result.returncode)
        return cls.scaffold_command_response(command=command, result_code=result_code, stdout=result.stdout, stderr=result.stderr)

    @classmethod
    def scaffold_command_response(cls, **kwargs) -> Action:
        state = State(command_id='0', user_name='tester', command=kwargs['command'], result_code=kwargs['result_code'], stderr=kwargs['stderr'])
        print(f'Command: {kwargs['command']}')
        print(f'RetCode: {kwargs['result_code']}')
        print(f"stdout: '{kwargs['stdout']}'")
        print(f"stderr: '{kwargs['stderr']}'")
        print('===========================')
        action = cls.agent.post_execute(state=state)
        print('Input: {}'.format(state.command))
        print('===========================')
        print('Response: {}'.format(action.suggested_command))
        print('===========================')
        print('Explanation: {}'.format(action.description))
        return action

    @unittest.skip('Only for local testing')
    def test_bpxmtextable_message_bad_cd(self):
        self.agent.init_agent()
        with tempfile.NamedTemporaryFile() as our_file:
            action = self.try_command(['cd', our_file.name])
        lines = action.description.strip().split('\n')
        self.assertEqual(f'{Colorize.EMOJI_ROBOT}I asked bpxmtext about that message:', lines[0])
        self.assertEqual('Action: Reissue the service specifying a directory file.', lines[-2])

    @unittest.skip('Only for local testing')
    def test_bpxmtextable_message_bad_cat(self):
        self.agent.init_agent()
        with tempfile.NamedTemporaryFile() as our_file:
            file_that_isnt_there: str = our_file.name
        action = self.try_command(['cat', file_that_isnt_there])
        lines = action.description.strip().split('\n')
        self.assertEqual(f'{Colorize.EMOJI_ROBOT}I asked bpxmtext about that message:', lines[0])
        self.assertEqual('Action: The open service request cannot be processed. Correct the name or the', lines[-3])
        self.assertEqual('open flags and retry the operation.', lines[-2])

    @unittest.skip('Only for local testing')
    def test_bpxmtextable_unhelpful_message_1(self):
        self.agent.init_agent()
        action = self.scaffold_command_response(command=['not', 'a', 'real', 'command'], result_code=str(1), stdout='', stderr='IEB4223I 0xDFDFDFDF')
        lines = action.description.strip().split('\n')
        self.assertEqual(f"{Colorize.EMOJI_ROBOT}I couldn't find any help for message code 'IEB4223I'", lines[0])

    @unittest.skip('Only for local testing')
    def test_bpxmtextable_unhelpful_message_2(self):
        self.agent.init_agent()
        action = self.scaffold_command_response(command=['not', 'a', 'real', 'command'], result_code=str(1), stdout='', stderr='IEB4223I 0xDFDF')
        lines = action.description.strip().split('\n')
        self.assertEqual(f"{Colorize.EMOJI_ROBOT}I couldn't find any help for message code 'IEB4223I'", lines[0])

    @unittest.skip('Only for local testing')
    def test_bad_cp(self):
        self.agent.init_agent()
        with tempfile.NamedTemporaryFile() as our_file:
            action = self.try_command(['cp', our_file.name, our_file.name])
        lines = action.description.strip().split('\n')
        self.assertEqual(f'{Colorize.EMOJI_ROBOT}I looked up FSUM8977 in the IBM KnowledgeCenter for you:', lines[0])
        self.assertEqual(f'{Colorize.INFO}Product: z/OS', lines[1])
        self.assertEqual('Topic: FSUM8977 ...', lines[2])

