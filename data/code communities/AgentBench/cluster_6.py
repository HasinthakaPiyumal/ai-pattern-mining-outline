# Cluster 6

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/agents/api_agents.yaml')
    parser.add_argument('--agent', type=str, default='gpt-3.5-turbo-0613')
    return parser.parse_args()

def interaction(agent: AgentClient):
    try:
        history = []
        while True:
            print('================= USER  ===================')
            user = input('>>> ')
            history.append({'role': 'user', 'content': user})
            try:
                agent_response = agent.inference(history)
                print('================ AGENT ====================')
                print(agent_response)
                history.append({'role': 'agent', 'content': agent_response})
            except Exception as e:
                print(e)
                exit(0)
    except KeyboardInterrupt:
        print('\n[Exit] KeyboardInterrupt')
        exit(0)

