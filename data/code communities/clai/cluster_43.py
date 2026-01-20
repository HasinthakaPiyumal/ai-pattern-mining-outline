# Cluster 43

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

