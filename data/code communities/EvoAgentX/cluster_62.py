# Cluster 62

def build_sequential_workflow():
    llm_config = OpenAILLMConfig(model='gpt-4o-mini', openai_key=OPENAI_API_KEY, stream=True, output_response=True)
    llm = OpenAILLM(llm_config)
    tasks = [{'name': 'Planning', 'description': 'Create a detailed plan for code generation', 'inputs': [{'name': 'problem', 'type': 'str', 'required': True, 'description': 'Description of the problem to be solved'}], 'outputs': [{'name': 'plan', 'type': 'str', 'required': True, 'description': 'Detailed plan with steps, components, and architecture'}], 'prompt': 'You are a software architect. Your task is to create a detailed implementation plan for the given problem.\n\nProblem: {problem}\n\nPlease provide a comprehensive implementation plan including:\n1. Problem breakdown\n2. Algorithm or approach selection\n3. Implementation steps\n4. Potential edge cases and solutions', 'parse_mode': 'str'}, {'name': 'Coding', 'description': 'Implement the code based on the implementation plan', 'inputs': [{'name': 'problem', 'type': 'str', 'required': True, 'description': 'Description of the problem to be solved'}, {'name': 'plan', 'type': 'str', 'required': True, 'description': 'Detailed implementation plan from the Planning phase'}], 'outputs': [{'name': 'code', 'type': 'str', 'required': True, 'description': 'Implemented code with explanations'}], 'prompt': 'You are a software developer. Your task is to implement the code based on the provided problem and implementation plan.\n\nProblem: {problem}\nImplementation Plan: {plan}\n\nPlease provide the implementation code with appropriate comments.', 'parse_mode': 'custom', 'parse_func': custom_parse_func, 'tool_names': ['FileToolkit']}]
    graph = SequentialWorkFlowGraph(goal='Generate code to solve programming problems', tasks=tasks)
    graph.save_module('debug/tool/sequential_workflow.json')
    graph = SequentialWorkFlowGraph.from_file('debug/tool/sequential_workflow.json')
    agent_manager = AgentManager(tools=[FileToolkit()])
    agent_manager.add_agents_from_workflow(graph, llm_config=llm_config)
    workflow = WorkFlow(graph=graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute(inputs={'problem': 'Write a function to find the longest palindromic substring in a given string. Save the code to local file: ./debug/test.py'})
    print('Workflow completed!')
    print('Workflow output:\n', output)

