# Cluster 60

class CodeVerification(Action):

    def __init__(self, **kwargs):
        name = kwargs.pop('name') if 'name' in kwargs else CODE_VERIFICATION_ACTION['name']
        description = kwargs.pop('description') if 'description' in kwargs else CODE_VERIFICATION_ACTION['description']
        prompt = kwargs.pop('prompt') if 'prompt' in kwargs else CODE_VERIFICATION_ACTION['prompt']
        inputs_format = kwargs.pop('inputs_format', None) or CodeVerificationInput
        outputs_format = kwargs.pop('outputs_format', None) or CodeVerificationOutput
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)

    def execute(self, llm: Optional[BaseLLM]=None, inputs: Optional[dict]=None, sys_msg: Optional[str]=None, return_prompt: bool=False, **kwargs) -> CodeVerificationOutput:
        if not inputs:
            logger.error('CodeVerification action received invalid `inputs`: None or empty.')
            raise ValueError('The `inputs` to CodeVerification action is None or empty.')
        prompt_params_names = ['code', 'requirements']
        prompt_params_values = {param: inputs.get(param, 'Not Provided') for param in prompt_params_names}
        prompt = self.prompt.format(**prompt_params_values)
        response = llm.generate(prompt=prompt, system_message=sys_msg)
        try:
            verification_result = self.outputs_format.parse(response.content, parse_mode='title')
        except Exception:
            try:
                code_blocks = extract_code_blocks(response.content, return_type=True)
                code = '\n\n'.join([f'```{code_type}\n{code}\n```' for code_type, code in code_blocks])
                verification_result = self.outputs_format(verified_code=code)
            except Exception:
                raise ValueError(f'Failed to extract code blocks from the response: {response.content}')
        if return_prompt:
            return (verification_result, prompt)
        return verification_result

def extract_code_blocks(text: str, return_type: bool=False) -> Union[List[str], List[tuple]]:
    """
    Extract code blocks from text enclosed in triple backticks.
    
    Args:
        text (str): The text containing code blocks
        return_type (bool): If True, returns tuples of (language, code), otherwise just code
        
    Returns:
        Union[List[str], List[tuple]]: Either list of code blocks or list of (language, code) tuples
    """
    code_block_pattern = '```((?:[a-zA-Z]*)?)\\n*(.*?)\\n*```'
    matches = regex.findall(code_block_pattern, text, regex.DOTALL)
    if not matches:
        return [(None, text.strip())] if return_type else [text.strip()]
    if return_type:
        return [(lang.strip() or None, code.strip()) for lang, code in matches]
    else:
        return [code.strip() for _, code in matches]

def execute_workflow(stock_code, data_dir, report_dir, timestamp):
    """Execute the workflow with the given parameters"""
    try:
        workflow_file = 'workflow.json'
        if platform.system() == 'Windows':
            workflow_file = 'workflow_windows.json'
        workflow_graph = WorkFlowGraph.from_file(workflow_file, llm_config=llm.config, tools=tools)
        agent_manager = AgentManager(tools=tools)
        agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm.config)
        workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
        workflow.init_module()
        output_file = report_dir / f'text_report_{stock_code}_{timestamp}.md'
        past_report = report_dir / f'text_report_{stock_code}_{timestamp}_previous.md'
        goal = f'I need a daily trading decision for stock {stock_code}.\nAvailable funds: {available_funds} RMB\nCurrent positions: {current_positions} shares of {stock_code} at average price {average_price} RMB\nDate: {report_date}\nType of position: {position_type}\nData folder: {data_dir}\nPast report folder: {past_report}\n\nPlease read ALL files in the data folder and generate a comprehensive trading decision report in Chinese based on real data. Return the complete content.\n'
        output = workflow.execute({'goal': goal})
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f'Trading decision report saved to: {output_file}')
        except Exception as e:
            print(f'Error saving report: {e}')
    except Exception as e:
        print(f'Error executing workflow: {e}')
        import traceback
        traceback.print_exc()

@register_parse_function
def custom_parse_func(content: str) -> str:
    return {'code': extract_code_blocks(content)[0]}

def main():
    openai_config = OpenAILLMConfig(model='gpt-4o-mini', openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    llm = OpenAILLM(config=openai_config)
    goal = 'Generate html code for the Tetris game that can be played in the browser.'
    target_directory = 'examples/output/tetris_game'
    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
    workflow_graph.display()
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()
    verification_llm_config = LiteLLMConfig(model='anthropic/claude-3-7-sonnet-20250219', anthropic_key=ANTHROPIC_API_KEY, stream=True, output_response=True, max_tokens=20000)
    verification_llm = LiteLLM(config=verification_llm_config)
    code_verifier = CodeVerification()
    output = code_verifier.execute(llm=verification_llm, inputs={'requirements': goal, 'code': output}).verified_code
    os.makedirs(target_directory, exist_ok=True)
    code_blocks = extract_code_blocks(output)
    if len(code_blocks) == 1:
        file_path = os.path.join(target_directory, 'index.html')
        with open(file_path, 'w') as f:
            f.write(code_blocks[0])
        print(f'You can open this HTML file in a browser to play the Tetris game: {file_path}')
        return
    code_extractor = CodeExtraction()
    results = code_extractor.execute(llm=llm, inputs={'code_string': output, 'target_directory': target_directory})
    print(f'Extracted {len(results.extracted_files)} files:')
    for filename, path in results.extracted_files.items():
        print(f'  - {filename}: {path}')
    if results.main_file:
        print(f'\nMain file: {results.main_file}')
        file_type = os.path.splitext(results.main_file)[1].lower()
        if file_type == '.html':
            print(f'You can open this HTML file in a browser to play the Tetris game')
        else:
            print(f'This is the main entry point for your application')

def generate_plan(llm: LiteLLM, goal: str, output_dir: str):
    """2.1 Generate task planning"""
    wait_for_user_confirmation('task planning generation')
    print('Starting task planning generation...')
    wf = WorkFlowGenerator(llm=llm)
    plan = wf.generate_plan(goal=goal)
    save_intermediate_result(plan, 'plan', output_dir)
    print(f'Task planning completed, containing {len(plan.sub_tasks)} sub-tasks')
    return plan

def wait_for_user_confirmation(step_name: str):
    """Wait for user confirmation before proceeding"""
    while True:
        user_input = input(f'\nReady to proceed with {step_name}? (yes/no): ').strip().lower()
        if user_input == 'yes':
            return True
        elif user_input == 'no':
            print('Stopping execution.')
            exit(0)
        else:
            print("Please enter 'yes' or 'no'")

def save_intermediate_result(data, stage: str, output_dir: str):
    """Save intermediate results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{stage}_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    if hasattr(data, 'to_dict'):
        data_dict = data.to_dict()
    elif hasattr(data, '__dict__'):
        data_dict = data.__dict__.copy()
        for key, value in data_dict.items():
            if hasattr(value, '__dict__') and (not isinstance(value, (str, int, float, bool, list, dict, type(None)))):
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    data_dict[key] = str(value)
    else:
        data_dict = data
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
        print(f'Saved {stage} results to: {filepath}')
    except (TypeError, ValueError) as e:
        print(f'Warning: Cannot fully serialize {stage} results ({e}), saving string representation')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(data))
    return filepath

def build_workflow_from_plan(llm: LiteLLM, goal: str, plan, output_dir: str):
    """2.2 Build workflow from plan"""
    wait_for_user_confirmation('workflow graph construction')
    print('Starting workflow graph construction...')
    wf = WorkFlowGenerator(llm=llm)
    workflow = wf.build_workflow_from_plan(goal=goal, plan=plan)
    save_intermediate_result(workflow, 'workflow_structure', output_dir)
    print(f'Workflow graph construction completed, containing {len(workflow.nodes)} nodes and {len(workflow.edges)} edges')
    return workflow

def generate_agents_for_workflow(llm: LiteLLM, goal: str, workflow: WorkFlowGraph, output_dir: str):
    """2.3 Generate agents for workflow"""
    wait_for_user_confirmation('agent generation for workflow')
    print('Starting agent generation for workflow...')
    wf = WorkFlowGenerator(llm=llm)
    workflow_with_agents = wf.generate_agents(goal=goal, workflow=workflow)
    save_intermediate_result(workflow_with_agents, 'workflow_with_agents', output_dir)
    print('Agent generation completed')
    return workflow_with_agents

def generate_workflow_step_by_step(llm: LiteLLM, goal: str, output_dir: str) -> WorkFlowGraph:
    """2. Generate and display workflow step by step"""
    print(f'Starting step-by-step workflow generation, goal: {goal}')
    print(f'Intermediate results will be saved to: {output_dir}')
    plan = generate_plan(llm, goal, output_dir)
    workflow = build_workflow_from_plan(llm, goal, plan, output_dir)
    workflow_with_agents = generate_agents_for_workflow(llm, goal, workflow, output_dir)
    workflow_with_agents.display()
    save_intermediate_result(workflow_with_agents, 'final_workflow', output_dir)
    print('Workflow generation completed!')
    return workflow_with_agents

def execute_workflow(llm: LiteLLM, graph: WorkFlowGraph, goal: str, target_dir: str):
    """3. Register Agents and execute workflow"""
    wait_for_user_confirmation('workflow execution')
    print('Starting workflow execution...')
    cfg = llm.config
    mgr = AgentManager()
    mgr.add_agents_from_workflow(graph, llm_config=cfg)
    workflow = WorkFlow(graph=graph, agent_manager=mgr, llm=llm)
    output = workflow.execute()
    print('Workflow execution completed')
    return output

def verify_and_extract_code(llm: LiteLLM, goal: str, output: str, target_dir: str):
    """4. Verify code and extract to files"""
    wait_for_user_confirmation('code verification and extraction')
    print('Starting code verification and extraction...')
    verifier = CodeVerification()
    verified = verifier.execute(llm=llm, inputs={'requirements': goal, 'code': output}).verified_code
    os.makedirs(target_dir, exist_ok=True)
    blocks = extract_code_blocks(verified)
    if len(blocks) == 1:
        path = os.path.join(target_dir, 'index.html')
        with open(path, 'w') as f:
            f.write(blocks[0])
        print(f'HTML file generated: {path}')
        return
    extractor = CodeExtraction()
    res = extractor.execute(llm=llm, inputs={'code_string': verified, 'target_directory': target_dir})
    print(f'Extracted {len(res.extracted_files)} files:')
    for name, p in res.extracted_files.items():
        print(f' - {name}: {p}')
    if res.main_file:
        ext = os.path.splitext(res.main_file)[1].lower()
        tip = 'can be opened in browser' if ext == '.html' else 'main entry file'
        print(f'\nMain file: {res.main_file}, {tip}')

def main():
    goal = 'Generate html code for the Tetris game that can be played in the browser.'
    target_dir = 'examples/output/tetris_game'
    output_dir = 'examples/output/workflow_intermediates'
    wait_for_user_confirmation('LLM configuration')
    llm = configure_llm()
    graph = generate_workflow_step_by_step(llm, goal, output_dir)
    output = execute_workflow(llm, graph, goal, target_dir)
    verify_and_extract_code(llm, goal, output, target_dir)
    print(f'\nComplete Tetris game has been generated to directory: {target_dir}')

def configure_llm() -> LiteLLM:
    """1. LLM Configuration - Using LiteLLM with Azure OpenAI"""
    cfg = LiteLLMConfig(model='azure/' + os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'), azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'), azure_key=os.getenv('AZURE_OPENAI_KEY'), api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview'), stream=True, output_response=True, max_tokens=16000, temperature=0.7)
    return LiteLLM(config=cfg)

def build_customize_agent_with_custom_parse_func():
    code_writer = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt='Write Python code that implements the following requirement: {requirement}', llm_config=openrouter_config, inputs=[{'name': 'requirement', 'type': 'string', 'description': 'The coding requirement'}], outputs=[{'name': 'code', 'type': 'string', 'description': 'The generated Python code'}], parse_mode='custom', parse_func=lambda content: {'code': extract_code_blocks(content)[0]})
    message = code_writer(inputs={'requirement': 'Write a function that returns the sum of two numbers'})
    print(f'Response from {code_writer.name}:')
    print(message.content.code)

def build_customize_agent_with_inputs_and_outputs_and_prompt_template():
    code_writer = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt_template=StringTemplate(instruction='Write Python code that implements the provided `requirement`'), llm_config=openrouter_config, inputs=[{'name': 'requirement', 'type': 'string', 'description': 'The coding requirement'}], outputs=[{'name': 'code', 'type': 'string', 'description': 'The generated Python code'}], parse_mode='custom', parse_func=lambda content: {'code': extract_code_blocks(content)[0]})
    message = code_writer(inputs={'requirement': 'Write a function that returns the sum of two numbers'})
    print(f'Response from {code_writer.name}:')
    print(message.content.code)

@register_parse_function
def customize_parse_func(content: str) -> dict:
    return {'code': extract_code_blocks(content)[0]}

@register_parse_function
def custom_parse_func(content: str) -> str:
    return {'code': extract_code_blocks(content)[0]}

@register_parse_function
def custom_parse_func(content: str) -> str:
    return {'code': extract_code_blocks(content)[0]}

