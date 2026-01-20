# Cluster 69

def create_optimized_agent(role_name, role_description, model_config, temperature_adjustment=0.0):
    """Create optimized agent, adjust model parameters based on role characteristics"""
    role_prompt = f'\nYou are debater #{{agent_id}} (role: {{role}}). This is round {{round_index}} of {{total_rounds}}.\n\nProblem:\n{{problem}}\n\nConversation so far:\n{{transcript_text}}\n\nInstructions:\n- You are a {role_name.upper()} who {role_description}\n- Think briefly (<= 120 words), then present your {role_name.lower()} argument or rebuttal\n- Focus on your unique perspective and expertise\n- If confident, provide your current answer for this round\n- Your output MUST follow this XML template:\n\n<response>\n  <thought>Your brief {role_name.lower()} reasoning</thought>\n  <argument>Your {role_name.lower()} argument or rebuttal</argument>\n  <answer>Optional current answer; leave empty if uncertain</answer>\n</response>\n'
    adjusted_config = model_config.model_copy()
    if hasattr(adjusted_config, 'temperature'):
        adjusted_config.temperature = max(0.0, min(1.0, adjusted_config.temperature + temperature_adjustment))
    inputs = [{'name': 'problem', 'type': 'str', 'description': 'Problem statement'}, {'name': 'transcript_text', 'type': 'str', 'description': 'Formatted debate transcript so far'}, {'name': 'role', 'type': 'str', 'description': 'Debater role/persona'}, {'name': 'agent_id', 'type': 'str', 'description': 'Debater id (string)'}, {'name': 'round_index', 'type': 'str', 'description': '1-based round index'}, {'name': 'total_rounds', 'type': 'str', 'description': 'Total rounds'}]
    outputs = [{'name': 'thought', 'type': 'str', 'description': 'Brief reasoning', 'required': True}, {'name': 'argument', 'type': 'str', 'description': 'Argument or rebuttal', 'required': True}, {'name': 'answer', 'type': 'str', 'description': 'Optional current answer', 'required': False}]
    return CustomizeAgent(name=role_name, description=f'{role_name} debater: {role_description}', prompt=role_prompt, llm_config=adjusted_config, inputs=inputs, outputs=outputs, parse_mode='xml')

def create_role_model_mapping():
    """Create role-model mapping strategy"""
    load_dotenv()
    openai_key = os.getenv('OPENAI_API_KEY')
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    roles = {'Optimist': 'always sees the bright side and positive opportunities', 'Pessimist': 'focuses on risks, problems, and potential downsides', 'Analyst': 'provides data-driven, balanced analysis', 'Innovator': 'thinks outside the box and suggests creative solutions', 'Conservative': 'values tradition, stability, and proven approaches', 'Skeptic': 'questions assumptions and demands evidence', 'Advocate': 'passionately defends a particular viewpoint', 'Mediator': 'seeks common ground and compromise', 'Expert': 'provides specialized knowledge and technical insights', 'Critic': 'identifies flaws and suggests improvements'}
    models = {'gpt4o_mini': OpenAILLMConfig(model='gpt-4o-mini', openai_key=openai_key, temperature=0.3), 'gpt4o': OpenAILLMConfig(model='gpt-4o', openai_key=openai_key, temperature=0.2), 'llama': OpenRouterConfig(model='meta-llama/llama-3.1-70b-instruct', openrouter_key=openrouter_key, temperature=0.3)}
    role_model_mapping = {'Innovator': ('gpt4o', 0.3), 'Advocate': ('gpt4o', 0.2), 'Analyst': ('llama', -0.1), 'Expert': ('llama', 0.0), 'Skeptic': ('llama', 0.0), 'Optimist': ('gpt4o_mini', 0.1), 'Pessimist': ('gpt4o_mini', 0.0), 'Conservative': ('gpt4o_mini', -0.1), 'Critic': ('gpt4o_mini', 0.0), 'Mediator': ('gpt4o_mini', 0.1)}
    return (roles, models, role_model_mapping)

def run_optimized_debate():
    """Run optimized debate: select most suitable model based on role characteristics"""
    print('=== Optimized Debate: Intelligent Role-Model Matching ===')
    roles, models, mapping = create_role_model_mapping()
    selected_roles = ['Analyst', 'Innovator', 'Skeptic', 'Advocate', 'Mediator']
    agents = []
    for role in selected_roles:
        model_name, temp_adjust = mapping[role]
        model_config = models[model_name]
        agent = create_optimized_agent(role, roles[role], model_config, temp_adjust)
        agents.append(agent)
    graph = MultiAgentDebateActionGraph(debater_agents=agents, llm_config=agents[0].llm_config if agents else None)
    result = graph.execute(problem='Should we invest heavily in AI research? Give a final Yes/No with reasons.', num_agents=5, num_rounds=3, judge_mode='llm_judge', return_transcript=True)
    print('Final Answer:', result.get('final_answer'))
    print('Winner:', result.get('winner'))
    if result.get('winner_answer'):
        print('Winner Answer:', result.get('winner_answer'))
    print('\nRole-Model Matching Strategy:')
    for i, agent in enumerate(agents):
        model_name = agent.llm_config.model if hasattr(agent.llm_config, 'model') else 'Unknown'
        temp = agent.llm_config.temperature if hasattr(agent.llm_config, 'temperature') else 'Unknown'
        print(f'  {agent.name}: {model_name} (Temperature: {temp}) - {roles[agent.name]}')

def main():
    """Main function"""
    print('MultiAgentDebate Advanced Example - Dynamic Role-Model Mapping')
    print('=' * 60)
    if not os.getenv('OPENAI_API_KEY'):
        print('Warning: OPENAI_API_KEY environment variable not set')
    if not os.getenv('OPENROUTER_API_KEY'):
        print('Warning: OPENROUTER_API_KEY environment variable not set')
    run_optimized_debate()

def run_self_consistency_example():
    llm_config = get_llm_config()
    debate = MultiAgentDebateActionGraph(name='MAD Minimal', description='Minimal runnable example for multi-agent debate', llm_config=llm_config)
    fixed_problem = 'How many labeled trees on 10 vertices are there such that vertex 1 has degree exactly 4? Return only the final integer.'
    result = debate.execute(problem=fixed_problem, num_agents=3, num_rounds=5, judge_mode='self_consistency', return_transcript=True)
    print('=== Example: Self-Consistency (Fixed Answer) ===')
    print('Final Answer:', result.get('final_answer'))
    print('Winner:', result.get('winner'))
    print('\nTranscript:')
    for turn in result.get('transcript', []):
        print(f'[Round {turn['round']}] Agent#{turn['agent_id']} ({turn['role']})\nArgument: {turn.get('argument', '').strip()}\nAnswer: {str(turn.get('answer') or '').strip()}\n')

def get_llm_config():
    load_dotenv()
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        raise RuntimeError('Set OPENAI_API_KEY for OpenAI access.')
    model = os.getenv('OPENAI_MODEL', 'gpt-4o')
    return OpenAILLMConfig(model=model, openai_key=openai_key, max_completion_tokens=800, temperature=0.5, output_response=True)

def run_llm_judge_example():
    llm_config = get_llm_config()
    debate = MultiAgentDebateActionGraph(name='MAD Minimal', description='Minimal runnable example for multi-agent debate', llm_config=llm_config)
    open_problem = 'Should AI agent service engineers be required to take an algorithms exam to validate their competencies? Return a final Yes/No and up to five concise reasons, assuming responsibilities include tool/function orchestration, workflow design, RAG integration, evaluation/telemetry, reliability/safety, and rapid delivery.'
    result = debate.execute(problem=open_problem, num_agents=5, num_rounds=5, judge_mode='llm_judge', return_transcript=True)
    print('=== Example: LLM Judge (Open Question) ===')
    print('Final Answer:', result.get('final_answer'))
    print('Winner:', result.get('winner'))
    print('\nTranscript:')
    for turn in result.get('transcript', []):
        print(f'[Round {turn['round']}] Agent#{turn['agent_id']} ({turn['role']})\nArgument: {turn.get('argument', '').strip()}\nAnswer: {str(turn.get('answer') or '').strip()}\n')

def main():
    run_llm_judge_example()

class GroupOfManyGraph(ActionGraph):
    name: str = 'GroupOfManyGraph'
    description: str = 'Group with variable number of inner debaters'
    llm_config: OpenAILLMConfig
    num_inner: int = 3
    _inner_debaters: List[CustomizeAgent] = None

    def init_module(self):
        super().init_module()
        prompt = '\nYou are a sub-team debater (role: {role}), agent {agent_id}, round {round_index}/{total_rounds}.\nProblem:\n{problem}\n\nTranscript so far:\n{transcript_text}\n\nReturn XML:\n<response>\n  <thought>...</thought>\n  <argument>...</argument>\n  <answer>optional</answer>\n</response>\n\t\t\t'.strip()
        self._inner_debaters = []
        for i in range(self.num_inner):
            debater = CustomizeAgent(name=f'GroupDebater#{i + 1}', description='Inner debater of a group (variable size)', prompt=prompt, llm_config=self.llm_config, inputs=[{'name': 'problem', 'type': 'str', 'description': 'The problem to debate'}, {'name': 'transcript_text', 'type': 'str', 'description': 'Transcript of previous debate rounds'}, {'name': 'role', 'type': 'str', 'description': 'Role of the debater'}, {'name': 'agent_id', 'type': 'str', 'description': 'Unique identifier for the agent'}, {'name': 'round_index', 'type': 'str', 'description': 'Current round number'}, {'name': 'total_rounds', 'type': 'str', 'description': 'Total number of debate rounds'}], outputs=[{'name': 'thought', 'type': 'str', 'required': True, 'description': "The agent's reasoning process"}, {'name': 'argument', 'type': 'str', 'required': True, 'description': "The agent's main argument"}, {'name': 'answer', 'type': 'str', 'required': False, 'description': 'Optional final answer from the agent'}], parse_mode='xml')
            self._inner_debaters.append(debater)

    def execute(self, problem: str, transcript_text: str, role: str, agent_id: str, round_index: str, total_rounds: str, **kwargs) -> dict:
        arguments: List[str] = []
        thoughts: List[str] = []
        answers: List[str] = []
        local_transcript = transcript_text
        for i, debater in enumerate(self._inner_debaters):
            msg = debater(inputs=dict(problem=problem, transcript_text=local_transcript, role=f'{role} - #{i + 1}', agent_id=f'{agent_id}_#{i + 1}', round_index=round_index, total_rounds=total_rounds))
            data = msg.content.get_structured_data()
            arg = (data.get('argument', '') or '').strip()
            th = (data.get('thought', '') or '').strip()
            ans = (data.get('answer') or '').strip()
            arguments.append(f'[#{i + 1}] {arg}')
            thoughts.append(f'[#{i + 1}] {th}')
            if ans:
                answers.append(ans)
            local_transcript = local_transcript + '\n' + f'[GroupInner#{i + 1} argument]: {arg}'
        answer = answers[-1] if answers else None
        argument_joined = '\n'.join(arguments)
        thought_joined = ' | '.join(thoughts)
        return {'argument': argument_joined, 'answer': answer, 'thought': thought_joined}

def create_sample_agents():
    """Create sample agents"""
    agents = []
    agent1 = CustomizeAgent(name='OptimistAgent', description='Optimistic debater who always sees the positive side of problems', prompt='You are an optimistic debater. Please analyze the problem from a positive perspective: {problem}', llm_config=OpenAILLMConfig(model='gpt-4o-mini', openai_key=os.getenv('OPENAI_API_KEY')), inputs=[{'name': 'problem', 'type': 'str', 'description': 'Problem'}], outputs=[{'name': 'argument', 'type': 'str', 'description': 'Argument'}], parse_mode='title')
    agents.append(agent1)
    agent2 = CustomizeAgent(name='PessimistAgent', description='Pessimistic debater who always sees the negative side of problems', prompt='You are a pessimistic debater. Please analyze the problem from a negative perspective: {problem}', llm_config=OpenAILLMConfig(model='gpt-4o-mini', openai_key=os.getenv('OPENAI_API_KEY')), inputs=[{'name': 'problem', 'type': 'str', 'description': 'Problem'}], outputs=[{'name': 'argument', 'type': 'str', 'description': 'Argument'}], parse_mode='title')
    agents.append(agent2)
    return agents

def demo_save_and_load():
    """Demonstrate save and load functionality"""
    print('=== Demonstrate Save and Load Functionality ===')
    agents = create_sample_agents()
    graph = MultiAgentDebateActionGraph(name='Demo Debate', description='Demo debate graph', debater_agents=agents, llm_config=agents[0].llm_config if agents else None)
    print('\n1. Get current configuration...')
    config = graph.get_config()
    print(f'Configuration contains {len(config)} fields')
    print(f'Number of agents: {len(config.get('debater_agents', []))}')
    print('\n2. Save configuration to file...')
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        save_path = graph.save_module(temp_path)
        print(f'Configuration saved to temporary file: {save_path}')
    except Exception as e:
        print(f'Failed to save to temp file: {e}')
        files_dir = 'examples/multi_agent_debate/files'
        os.makedirs(files_dir, exist_ok=True)
        save_path = graph.save_module(os.path.join(files_dir, 'demo_debate_config.json'))
        print(f'Configuration saved to files directory: {save_path}')
    print('\n3. Create new instance from configuration dictionary...')
    new_graph_from_dict = MultiAgentDebateActionGraph.from_dict(config)
    print(f'New instance name: {new_graph_from_dict.name}')
    print(f'New instance agent count: {len(new_graph_from_dict.debater_agents or [])}')
    print('\n4. Load new instance from file...')
    new_graph_from_file = MultiAgentDebateActionGraph.load_module(save_path)
    print(f'Loaded instance name: {new_graph_from_file.name}')
    print(f'Loaded instance agent count: {len(new_graph_from_file.debater_agents or [])}')
    print('\n5. Load configuration to existing instance...')
    empty_graph = MultiAgentDebateActionGraph()
    empty_graph.load_module(save_path)
    print(f'Loaded instance name: {empty_graph.name}')
    print(f'Loaded agent count: {len(empty_graph.debater_agents or [])}')
    return save_path

def demo_error_handling():
    """Demonstrate error handling"""
    print('\n=== Demonstrate Error Handling ===')
    print('\n1. Try to load non-existent file...')
    try:
        MultiAgentDebateActionGraph.load_module('nonexistent_file.json')
    except FileNotFoundError as e:
        print(f'Expected error: {e}')
    print('\n2. Try to create instance from invalid dictionary...')
    try:
        invalid_config = {'invalid_field': 'invalid_value'}
        MultiAgentDebateActionGraph.from_dict(invalid_config)
        print('Successfully created instance (using default values)')
    except Exception as e:
        print(f'Error: {e}')

class TestModule(unittest.TestCase):

    def setUp(self):
        self.save_file = 'tests/agents/saved_agent.json'

    def test_initialization(self):
        agent_data = {'name': 'test_agent', 'description': 'test_agent_description', 'llm_config': {'class_name': 'LiteLLMConfig', 'model': 'gpt-4o-mini', 'openai_key': 'xxxxx'}, 'actions': [{'class_name': 'Action', 'name': 'test_action_name', 'description': 'test_action_desc', 'prompt': 'test_action_prompt'}]}
        agent = Agent.from_dict(agent_data)
        self.assertEqual(agent.llm_config.model, 'gpt-4o-mini')
        self.assertTrue(isinstance(agent.llm, LiteLLM))
        self.assertTrue(isinstance(agent.actions[0], Action))
        self.assertTrue(len(agent.get_all_actions()) == 1)
        action = agent.get_action('test_action_name')
        self.assertEqual(action.name, 'test_action_name')
        self.assertEqual(action.description, 'test_action_desc')
        prompts = agent.get_prompts()
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts['test_action_name']['system_prompt'], None)
        self.assertEqual(prompts['test_action_name']['prompt'], 'test_action_prompt')
        agent.set_prompt('test_action_name', 'new_test_action_prompt', 'new_system_prompt')
        self.assertTrue(agent.system_prompt, 'new_system_prompt')
        self.assertEqual(agent.get_action('test_action_name').prompt, 'new_test_action_prompt')
        agent.set_prompts({'test_action_name': {'system_prompt': 'new_system_prompt_v2', 'prompt': 'new_test_action_prompt_v2'}})
        self.assertTrue(agent.system_prompt, 'new_system_prompt_v2')
        self.assertEqual(agent.get_action('test_action_name').prompt, 'new_test_action_prompt_v2')
        agent2 = Agent.from_dict(agent_data)
        agent_list = [agent]
        self.assertTrue(agent2 not in agent_list)
        self.assertTrue(agent2 != agent)
        agent2_id = agent2.agent_id
        agent2.agent_id = agent.agent_id
        self.assertTrue(agent2 in agent_list)
        self.assertTrue(agent2 == agent)
        agent2.agent_id = agent2_id

    def test_save_agent(self):
        llm_config = LiteLLMConfig(model='gpt-4o-mini', openai_key='xxxxx')
        agent = Agent(name='Bob', description='Bob is an engineer. He excels in writing and reviewing codes for different projects.', system_prompt='You are an excellent engineer and you can solve diverse coding tasks.', llm_config=llm_config, actions=[{'name': 'WriteFileToDisk', 'description': 'save several files to local storage.', 'tools': [{'name': 'FileToolKit', 'tools': [{'name': 'WriteFile', 'description': 'Write file to disk', 'inputs': {}}]}]}])
        agent.save_module(path=self.save_file)
        loaded_agent = Agent.from_file(path=self.save_file, llm_config=llm_config)
        self.assertEqual(agent, loaded_agent)

    def tearDown(self):
        if os.path.exists(self.save_file):
            os.remove(self.save_file)

class TestModule(unittest.TestCase):

    def test_agent_manager(self):
        OPENAI_API_KEY = 'xxxxx'
        llm_config = LiteLLMConfig(model='gpt-4o-mini', openai_key=OPENAI_API_KEY)
        agent = Agent(name='Bob', description='Bob is an engineer. He excels in writing and reviewing codes for different projects.', system_prompt='You are an excellent engineer and you can solve diverse coding tasks.', llm_config=llm_config, actions=[{'name': 'WriteFileToDisk', 'description': 'save several files to local storage.', 'tools': [{'name': 'FileToolKit', 'tools': [{'name': 'WriteFile', 'description': 'Write file to disk', 'inputs': {}}]}]}])
        agent_manager = AgentManager()
        agent_manager.add_agents(agents=[agent, {'class_name': 'Agent', 'name': 'test_agent', 'description': 'test_agent_description', 'llm_config': llm_config}])
        self.assertEqual(agent_manager.size, 2)
        self.assertTrue(agent_manager.has_agent(agent_name='Bob'))
        num_agents = agent_manager.size
        agent_manager.add_agents(agents=[agent])
        self.assertEqual(agent_manager.size, num_agents)
        self.assertTrue(isinstance(agent_manager.get_agent('test_agent'), Agent))
        self.assertEqual(agent_manager.size, 2)
        agent_manager.add_agent({'name': 'custom_agent', 'description': 'custom_agent_desc', 'prompt': 'customize prompt', 'is_human': True})
        self.assertEqual(agent_manager.size, 3)
        self.assertTrue(isinstance(agent_manager.get_agent('custom_agent'), CustomizeAgent))
        agent_manager.remove_agent(agent_name='test_agent')
        self.assertEqual(agent_manager.size, 2)
        self.assertTrue(agent_manager.has_agent('Bob'))
        self.assertTrue(agent_manager.has_agent('custom_agent'))
        self.assertEqual(agent_manager.get_agent_state('Bob'), AgentState.AVAILABLE)
        agent_manager.set_agent_state(agent_name='Bob', new_state=AgentState.RUNNING)
        self.assertEqual(agent_manager.get_agent_state('Bob'), AgentState.RUNNING)
        agent_manager.clear_agents()
        self.assertEqual(agent_manager.size, 0)

class TestModule(unittest.TestCase):

    def setUp(self):
        self.save_files = ['tests/agents/saved_customize_agent.json', 'tests/agents/saved_customize_agent_with_inputs.json', 'tests/agents/saved_customize_agent_with_outputs.json', 'tests/agents/saved_customize_agent_with_inputs_outputs.json', 'tests/agents/saved_customize_agent_with_parser.json']

    @patch('evoagentx.models.litellm_model.LiteLLM.single_generate')
    def test_simple_agent(self, mock_generate):
        mock_generate.return_value = 'Hello, world!'
        llm_config = LiteLLMConfig(model='gpt-4o-mini', openai_key='xxxxx')
        simple_agent = CustomizeAgent(name='Simple Agent', description='A simple agent that prints hello world', prompt='You are a simple agent that prints hello world.', llm_config=llm_config)
        self.assertEqual(simple_agent.name, 'Simple Agent')
        self.assertEqual(simple_agent.prompt, 'You are a simple agent that prints hello world.')
        self.assertEqual(simple_agent.customize_action_name, 'SimpleAgentAction')
        self.assertEqual(simple_agent.get_prompts()['SimpleAgentAction']['prompt'], 'You are a simple agent that prints hello world.')
        self.assertEqual(len(simple_agent.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(simple_agent.action.outputs_format.get_attrs()), 0)
        simple_agent.save_module(self.save_files[0])
        new_agent: CustomizeAgent = CustomizeAgent.from_file(self.save_files[0], llm_config=llm_config)
        self.assertEqual(new_agent.name, 'Simple Agent')
        self.assertEqual(len(new_agent.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(new_agent.action.outputs_format.get_attrs()), 0)
        msg = new_agent()
        self.assertTrue(isinstance(msg, Message))
        self.assertEqual(msg.msg_type, MessageType.UNKNOWN)
        self.assertEqual(msg.content.content, 'Hello, world!')

    @patch('evoagentx.models.litellm_model.LiteLLM.single_generate')
    def test_agent_with_inputs_and_outputs(self, mock_generate):
        mock_generate.return_value = "```python\nprint('Hello, world!')```"
        llm_config = LiteLLMConfig(model='gpt-4o-mini', openai_key='xxxxx')
        agent_with_inputs = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt='Write Python code that implements the following requirement: {requirement}', llm_config=llm_config, inputs=[{'name': 'requirement', 'type': 'string', 'description': 'The coding requirement', 'required': True}])
        self.assertEqual(len(agent_with_inputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(agent_with_inputs.action.outputs_format.get_attrs()), 0)
        agent_with_inputs.save_module(self.save_files[1])
        new_agent_with_inputs: CustomizeAgent = CustomizeAgent.from_file(self.save_files[1], llm_config=llm_config)
        self.assertEqual(len(new_agent_with_inputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(new_agent_with_inputs.action.outputs_format.get_attrs()), 0)
        msg = new_agent_with_inputs(inputs={'requirement': 'Write Python code that prints hello world'}, return_msg_type=MessageType.RESPONSE)
        self.assertEqual(msg.msg_type, MessageType.RESPONSE)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        agent_with_outputs = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt='Write Python code that implements the following requirement: Write Python code that prints hello world', llm_config=llm_config, outputs=[{'name': 'code', 'type': 'string', 'description': 'The generated Python code', 'required': True}], parse_mode='custom', parse_func=customize_parse_func, title_format='## {title}')
        self.assertEqual(len(agent_with_outputs.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(agent_with_outputs.action.outputs_format.get_attrs()), 1)
        agent_with_outputs.save_module(self.save_files[2])
        new_agent_with_outputs: CustomizeAgent = CustomizeAgent.from_file(self.save_files[2], llm_config=llm_config)
        self.assertEqual(len(new_agent_with_outputs.action.inputs_format.get_attrs()), 0)
        self.assertEqual(len(new_agent_with_outputs.action.outputs_format.get_attrs()), 1)
        self.assertEqual(new_agent_with_outputs.parse_func.__name__, 'customize_parse_func')
        msg = new_agent_with_outputs(return_msg_type=MessageType.RESPONSE)
        self.assertEqual(msg.msg_type, MessageType.RESPONSE)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        self.assertEqual(msg.content.code, "print('Hello, world!')")
        agent_with_inputs_outputs = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt='Write Python code that implements the following requirement: {requirement}', llm_config=llm_config, inputs=[{'name': 'requirement', 'type': 'string', 'description': 'The coding requirement', 'required': True}], outputs=[{'name': 'code', 'type': 'string', 'description': 'The generated Python code', 'required': True}], parse_mode='custom', parse_func=customize_parse_func)
        self.assertEqual(len(agent_with_inputs_outputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(agent_with_inputs_outputs.action.outputs_format.get_attrs()), 1)
        agent_with_inputs_outputs.save_module(self.save_files[3])
        new_agent_with_inputs_outputs: CustomizeAgent = CustomizeAgent.from_file(self.save_files[3], llm_config=llm_config)
        self.assertEqual(len(new_agent_with_inputs_outputs.action.inputs_format.get_attrs()), 1)
        self.assertEqual(len(new_agent_with_inputs_outputs.action.outputs_format.get_attrs()), 1)
        msg = new_agent_with_inputs_outputs(inputs={'requirement': 'Write Python code that prints hello world'}, return_msg_type=MessageType.RESPONSE)
        self.assertEqual(msg.msg_type, MessageType.RESPONSE)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        self.assertEqual(msg.content.code, "print('Hello, world!')")
        agent_with_parser = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt='Write Python code that implements the following requirement: {requirement}', llm_config=llm_config, inputs=[{'name': 'requirement', 'type': 'string', 'description': 'The coding requirement', 'required': True}], outputs=[{'name': 'code', 'type': 'string', 'description': 'The generated Python code', 'required': True}, {'name': 'explanation', 'type': 'string', 'description': 'The explanation of the generated Python code', 'required': True}], output_parser=CodeWriterActionOutput, parse_mode='custom', parse_func=customize_parse_func)
        self.assertEqual(agent_with_parser.action.outputs_format.__name__, 'CodeWriterActionOutput')
        agent_with_parser.save_module(self.save_files[4])
        new_agent_with_parser: CustomizeAgent = CustomizeAgent.from_file(self.save_files[4], llm_config=llm_config)
        self.assertEqual(new_agent_with_parser.action.outputs_format.__name__, 'CodeWriterActionOutput')
        msg = new_agent_with_parser(inputs={'requirement': 'Write Python code that prints hello world'}, return_msg_type=MessageType.RESPONSE)
        self.assertEqual(msg.msg_type, MessageType.RESPONSE)
        self.assertEqual(msg.content.content, "```python\nprint('Hello, world!')```")
        self.assertEqual(msg.content.code, "print('Hello, world!')")

    def tearDown(self):
        for file in self.save_files:
            if os.path.exists(file):
                os.remove(file)

