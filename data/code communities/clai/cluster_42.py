# Cluster 42

class GPT3(Agent):

    def __init__(self):
        super(GPT3, self).__init__()
        self._gpt3_api = self.__init_gpt3_api__()

    def __init_gpt3_api__(self):
        current_directory = str(Path(__file__).parent.absolute())
        path_to_gpt3_key = os.path.join(current_directory, 'openai_api.key')
        path_to_gpt3_prompts = os.path.join(current_directory, 'prompt.json')
        gpt3_key = open(path_to_gpt3_key, 'r').read()
        gpt3_prompts = json.load(open(path_to_gpt3_prompts, 'r'))
        gpt3_api = GPT(temperature=0)
        gpt3_api.set_api_key(gpt3_key)
        for prompt in gpt3_prompts:
            ip, op = (prompt['input'], prompt['output'])
            example = Example(ip, op)
            gpt3_api.add_example(example)
        return gpt3_api

    def get_next_action(self, state: State) -> Action:
        command = state.command
        command = command[:1000]
        try:
            response = self._gpt3_api.get_top_reply(command, strip_output_suffix=True)
            response = response.strip()
            return Action(suggested_command=response, execute=False, description='Currently the GPT-3 skill does not provide an explanation. Got an idea? Contribute to CLAI!', confidence=0.0)
        except Exception as ex:
            return [{'text': 'Method failed with status ' + str(ex)}, 0.0]

class Voice(Agent):

    def __init__(self):
        super(Voice, self).__init__()
        self._api_filename = 'openai_api.key'
        self._priming_filename = 'priming.json'
        self._tmp_filepath = os.path.join(tempfile.gettempdir(), 'tts.mp3')
        self._gpt_api = self.__init_gpt_api__()
        self.__prime_gpt_model__()

    def __init_gpt_api__(self):
        curdir = str(Path(__file__).parent.absolute())
        key_filepath = os.path.join(curdir, self._api_filename)
        with open(key_filepath, 'r') as f:
            key = f.read()
        gpt_api = GPT()
        gpt_api.set_api_key(key)
        return gpt_api

    def __prime_gpt_model__(self):
        curdir = str(Path(__file__).parent.absolute())
        priming_filepath = os.path.join(curdir, self._priming_filename)
        with open(priming_filepath, 'r') as f:
            priming_examples = json.load(f)
        for priming_set in priming_examples:
            ip, op = (priming_set['input'], priming_set['output'])
            example = Example(ip, op)
            self._gpt_api.add_example(example)

    def summarize_output(self, state):
        stderr = str(state.stderr)
        prompt = stderr.split('\n')[0]
        gpt_summary = self._gpt_api.get_top_reply(prompt, strip_output_suffix=True)
        summary = f'error. {gpt_summary}'
        return summary

    def synthesize(self, text):
        """ Converts text to audio and saves to temp file """
        tts = gTTS(text, lang='en', lang_check=False)
        tts.save(self._tmp_filepath)

    def speak(self):
        subprocess.Popen(['nohup', 'ffplay', '-nodisp', '-autoexit', '-nostats', '-hide_banner', '-loglevel', 'warning', self._tmp_filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def post_execute(self, state: State) -> Action:
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        text_to_speak = self.summarize_output(state)
        self.synthesize(text_to_speak)
        self.speak()
        return Action(suggested_command=NOOP_COMMAND, confidence=0.01)

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

