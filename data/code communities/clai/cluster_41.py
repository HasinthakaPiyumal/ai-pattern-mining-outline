# Cluster 41

class GITBOT(Agent):

    def __init__(self):
        super(GITBOT, self).__init__()
        self.service = Service()
    ' pre execution processing '

    def get_next_action(self, state: State) -> Action:
        command = state.command
        return self.service(command)
    ' pre execution processing '

    def post_execute(self, state: State) -> Action:
        if state.result_code == '0':
            self.service.parse_command(state.command, stdout='')
        return Action(suggested_command=NOOP_COMMAND)

    def save_agent(self) -> bool:
        os.system('lsof -t -i tcp:{} | xargs kill'.format(_rasa_port_number))
        super().save_agent()

