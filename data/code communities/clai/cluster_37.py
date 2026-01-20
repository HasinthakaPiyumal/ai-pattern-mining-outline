# Cluster 37

class NLC2CMD(Agent):

    def __init__(self):
        super(NLC2CMD, self).__init__()
        self.service = Service()

    def get_next_action(self, state: State) -> Action:
        command = state.command
        data, confidence = self.service(command)
        response = data['text']
        return Action(suggested_command=NOOP_COMMAND, execute=True, description=Colorize().info().append(response).to_console(), confidence=confidence)

