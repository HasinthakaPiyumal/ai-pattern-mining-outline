# Cluster 65

def test_MCP_server():
    mcp_Toolkit = MCPToolkit(config_path='examples/output/mcp_agent/mcp.config')
    tools = mcp_Toolkit.get_toolkits()
    mcp_agent = CustomizeAgent(name='MCPAgent', description='A MCP agent that can use the tools provided by the MCP server', prompt_template=StringTemplate(instruction="Do some operations based on the user's instruction."), llm_config=openai_config, inputs=[{'name': 'instruction', 'type': 'string', 'description': 'The goal you need to achieve'}], outputs=[{'name': 'result', 'type': 'string', 'description': 'The result of the operation'}], tools=tools)
    mcp_agent.save_module('examples/output/mcp_agent/mcp_agent.json')
    mcp_agent.load_module('examples/output/mcp_agent/mcp_agent.json', llm_config=openai_config, tools=tools)
    message = mcp_agent(inputs={'instruction': 'Summarize all the tools.'})
    print(f'Response from {mcp_agent.name}:')
    print(message.content.result)

@register_parse_function
def extract_code_blocks(content: str) -> dict:
    return {'code': util_extract_code_blocks(content)[0]}

def build_customize_agent_with_custom_parse_func():
    code_writer = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt='Write Python code that implements the following requirement: {requirement}', llm_config=model_config, inputs=[{'name': 'requirement', 'type': 'string', 'description': 'The coding requirement'}], outputs=[{'name': 'code', 'type': 'string', 'description': 'The generated Python code'}], parse_mode='custom', parse_func=lambda content: {'code': util_extract_code_blocks(content)[0]})
    message = code_writer(inputs={'requirement': 'Write a function that returns the sum of two numbers'})
    print(f'Response from {code_writer.name}:')
    print(message.content.code)

def build_customize_agent_with_inputs_and_outputs_and_prompt_template():
    code_writer = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt_template=StringTemplate(instruction='Write Python code that implements the provided `requirement`'), llm_config=model_config, inputs=[{'name': 'requirement', 'type': 'string', 'description': 'The coding requirement'}], outputs=[{'name': 'code', 'type': 'string', 'description': 'The generated Python code'}], parse_mode='custom', parse_func=lambda content: {'code': util_extract_code_blocks(content)[0]})
    message = code_writer(inputs={'requirement': 'Write a function that returns the sum of two numbers'})
    print(f'Response from {code_writer.name}:')
    print(message.content.code)

def build_customize_agent_with_tools():
    code_writer = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt_template=StringTemplate(instruction='Write Python code that implements the provided `requirement` and save the code to the provided `file_path`'), llm_config=model_config, inputs=[{'name': 'requirement', 'type': 'string', 'description': 'The coding requirement'}, {'name': 'file_path', 'type': 'string', 'description': 'The path to save the code'}], tools=[FileToolkit()])
    message = code_writer(inputs={'requirement': 'Write a function that returns the sum of two numbers', 'file_path': 'examples/output/test_code.py'})
    print(f'Response from {code_writer.name}:')
    print(message.content.content)

def build_customize_agent_with_MCP(config_path):
    mcp_Toolkit = MCPToolkit(config_path=config_path)
    tools = mcp_Toolkit.get_toolkits()
    customize_agent = CustomizeAgent(name='MCPToolUser', description='Do some tasks using the tools', prompt_template=StringTemplate(instruction='Do some tasks using the tools'), llm_config=model_config, inputs=[{'name': 'instruction', 'type': 'string', 'description': 'The instruction to the tool user'}], outputs=[{'name': 'result', 'type': 'string', 'description': 'The result of the task'}, {'name': 'tool_calls', 'type': 'string', 'description': 'The tool calls used to get the result (if any)'}], tools=tools)
    message = customize_agent(inputs={'instruction': 'Summarize all your tools.'})
    print(f'Response from {customize_agent.name}:')
    print(message.content)

def build_customize_agent():
    agent_data = {'name': 'FirstAgent', 'description': 'A simple agent that prints hello world', 'prompt': "Print 'hello world'", 'llm_config': model_config}
    agent = CustomizeAgent.from_dict(agent_data)
    message: Message = agent()
    print(f'Response from {agent.name}:')
    print(message.content.content)

def build_customize_agent_with_inputs():
    simple_agent = CustomizeAgent(name='SimpleAgent', description='A basic agent that responds to queries', prompt='Answer the following question: {question}', llm_config=model_config, inputs=[{'name': 'question', 'type': 'string', 'description': 'The question to answer'}])
    response = simple_agent(inputs={'question': 'What is a language model?'})
    print(f'Response from {simple_agent.name}:')
    print(response.content.content)

def build_customize_agent_with_inputs_and_outputs():
    code_writer = CustomizeAgent(name='CodeWriter', description='Writes Python code based on requirements', prompt='Write Python code that implements the following requirement: {requirement}', llm_config=model_config, inputs=[{'name': 'requirement', 'type': 'string', 'description': 'The coding requirement'}], outputs=[{'name': 'code', 'type': 'string', 'description': 'The generated Python code'}], parse_mode='str')
    message = code_writer(inputs={'requirement': 'Write a function that returns the sum of two numbers'})
    print(f'Response from {code_writer.name}:')
    print(message.content.code)

def build_customize_agent_with_json_parse():
    """Test case demonstrating JSON parse mode for structured data extraction."""
    print('Test case: build_customize_agent_with_json_parse')
    recipe_analyzer = CustomizeAgent(name='RecipeAnalyzer', description='Analyzes recipe information and returns structured data', prompt='Analyze the following recipe and extract key information.\nRecipe: {recipe_text}\n\nPlease format your response as a JSON object with the following structure (all on one line):\n{{\'name\': \'Recipe name\', \'prep_time_minutes\': "12", \'ingredients\': [\'ingredient1\', \'ingredient2\', ...], \'difficulty\': \'easy|medium|hard\'}}', llm_config=model_config, inputs=[{'name': 'recipe_text', 'type': 'string', 'description': 'The recipe text to analyze'}], outputs=[{'name': 'name', 'type': 'string', 'description': 'Name of the recipe'}, {'name': 'prep_time_minutes', 'type': 'string', 'description': 'Preparation time in minutes'}, {'name': 'ingredients', 'type': 'list', 'description': 'List of ingredients'}, {'name': 'difficulty', 'type': 'string', 'description': 'Difficulty level of the recipe'}], parse_mode='json')
    sample_recipe = '\n    Classic Chocolate Chip Cookies\n    \n    Mix 2 1/4 cups flour, 1 cup butter, 3/4 cup sugar, 2 eggs, \n    1 tsp vanilla extract, and 2 cups chocolate chips. \n    Bake at 375°F for 10-12 minutes.\n    Total prep time: 25 minutes.\n    '
    message = recipe_analyzer(inputs={'recipe_text': sample_recipe})
    print(f'\nResponse from {recipe_analyzer.name}:')
    print('Recipe Name:', message.content.name)
    print('Prep Time:', message.content.prep_time_minutes, 'minutes')
    print('Ingredients:', ', '.join(message.content.ingredients))
    print('Difficulty:', message.content.difficulty)

def test_str_parse_mode():
    """Test case demonstrating string parse mode."""
    print('\nTest case: test_str_parse_mode')
    simple_agent = CustomizeAgent(name='SimpleGreeter', description='A simple agent that generates greetings', prompt='Generate a greeting for {name}', llm_config=model_config, inputs=[{'name': 'name', 'type': 'string', 'description': 'The name to greet'}], outputs=[{'name': 'greeting', 'type': 'string', 'description': 'The generated greeting'}], parse_mode='str')
    message = simple_agent(inputs={'name': 'Alice'})
    print(f'Response from {simple_agent.name}:')
    print('Raw content:', message.content.content)
    print('Greeting field:', message.content.greeting)

def test_title_parse_mode():
    """Test case demonstrating title parse mode."""
    print('\nTest case: test_title_parse_mode')
    report_agent = CustomizeAgent(name='ReportGenerator', description='Generates a structured report', prompt='Create a report about {topic} with summary and analysis sections, less than 200 words, section title format: ### title', llm_config=model_config, inputs=[{'name': 'topic', 'type': 'string', 'description': 'The topic to analyze'}], outputs=[{'name': 'summary', 'type': 'string', 'description': 'Brief summary'}, {'name': 'analysis', 'type': 'string', 'description': 'Detailed analysis'}], parse_mode='title', title_format='### {title}')
    message = report_agent(inputs={'topic': 'Artificial Intelligence'})
    print(f'Response from {report_agent.name}:')
    print('Summary:', message.content.summary)
    print('Analysis:', message.content.analysis)

def test_xml_parse_mode():
    """Test case demonstrating XML parse mode."""
    print('\nTest case: test_xml_parse_mode')
    extractor_agent = CustomizeAgent(name='DataExtractor', description='Extracts structured data', prompt='Extract key information from this text: {text}\n        Format your response using XML tags for each field.\n        Example format:\n        The people mentioned are: <people>John and Jane</people>\n        The places mentioned are: <places>New York and London</places>', llm_config=model_config, inputs=[{'name': 'text', 'type': 'string', 'description': 'The text to extract information from'}], outputs=[{'name': 'people', 'type': 'string', 'description': 'Names of people mentioned'}, {'name': 'places', 'type': 'string', 'description': 'Locations mentioned'}], parse_mode='xml')
    sample_text = 'John and Jane visited New York and London last summer.'
    message = extractor_agent(inputs={'text': sample_text})
    print(f'Response from {extractor_agent.name}:')
    print('People:', message.content.people)
    print('Places:', message.content.places)

def build_customize_agent_with_prompt_template():
    agent = CustomizeAgent(name='FirstAgent', description='A simple agent that prints hello world', prompt_template=StringTemplate(instruction="Print 'hello world'"), llm_config=model_config)
    message = agent()
    print(f'Response from {agent.name}:')
    print(message.content.content)

def build_customize_agent_with_chat_prompt_template():
    agent = CustomizeAgent(name='FirstAgent', description='A simple agent that prints hello world', prompt_template=ChatTemplate(instruction="Print 'hello world'"), llm_config=model_config)
    message = agent()
    print(f'Response from {agent.name}:')
    print(message.content.content)

def build_customize_agent_with_custom_parse_and_format():
    """Test case demonstrating custom parse function and output format with XML."""

    def custom_xml_parser(content: str) -> dict:
        """Custom parser that extracts data from XML-like format."""
        result = {}
        for field in ['name', 'age', 'occupation']:
            start_tag = f'<{field}>'
            end_tag = f'</{field}>'
            try:
                start_idx = content.index(start_tag) + len(start_tag)
                end_idx = content.index(end_tag)
                result[field] = content[start_idx:end_idx].strip()
            except ValueError:
                result[field] = ''
        return result
    person_info_agent = CustomizeAgent(name='PersonInfoExtractor', description='Extracts structured person information in XML format', prompt_template=StringTemplate(instruction='Extract information about the following person: `person_description`'), llm_config=model_config, inputs=[{'name': 'person_description', 'type': 'string', 'description': 'Description of the person'}], outputs=[{'name': 'name', 'type': 'string', 'description': "Person's name"}, {'name': 'age', 'type': 'string', 'description': "Person's age"}, {'name': 'occupation', 'type': 'string', 'description': "Person's occupation"}], parse_mode='custom', parse_func=custom_xml_parser, custom_output_format="Please format your response in XML tags:\n<name>person's name</name>\n<age>person's age</age>\n<occupation>person's occupation</occupation>")
    message = person_info_agent(inputs={'person_description': 'John is a 35-year-old software engineer who loves coding.'})
    print(f'Response from {person_info_agent.name}:')
    print('Name:', message.content.name)
    print('Age:', message.content.age)
    print('Occupation:', message.content.occupation)

def test_str_parse_mode_with_template():
    """Test case demonstrating string parse mode with PromptTemplate."""
    print('\nTest case: test_str_parse_mode_with_template')
    simple_agent = CustomizeAgent(name='SimpleGreeter', description='A simple agent that generates greetings', prompt_template=StringTemplate(instruction='Generate a friendly greeting for the provided `name`', constraints=['Keep the greeting concise and friendly', 'Use proper capitalization']), llm_config=model_config, inputs=[{'name': 'name', 'type': 'string', 'description': 'The name to greet'}], outputs=[{'name': 'greeting', 'type': 'string', 'description': 'The generated greeting'}], parse_mode='str')
    message = simple_agent(inputs={'name': 'Alice'})
    print(f'Response from {simple_agent.name}:')
    print('Raw content:', message.content.content)
    print('Greeting field:', message.content.greeting)

def test_title_parse_mode_with_template():
    """Test case demonstrating title parse mode with PromptTemplate."""
    print('\nTest case: test_title_parse_mode_with_template')
    report_agent = CustomizeAgent(name='ReportGenerator', description='Generates a structured report', prompt_template=StringTemplate(instruction='Create a comprehensive report about the provided `topic`', constraints=['Keep each section under 100 words', 'Use professional language', 'Be specific and factual'], context='You are a professional report writer with expertise in creating concise, informative reports.'), llm_config=model_config, inputs=[{'name': 'topic', 'type': 'string', 'description': 'The topic to analyze'}], outputs=[{'name': 'summary', 'type': 'string', 'description': 'Brief summary of key points'}, {'name': 'analysis', 'type': 'string', 'description': 'Detailed analysis and implications'}], parse_mode='title', title_format='### {title}')
    message = report_agent(inputs={'topic': 'Artificial Intelligence'})
    print(f'Response from {report_agent.name}:')
    print('Summary:', message.content.summary)
    print('Analysis:', message.content.analysis)

def test_xml_parse_mode_with_template():
    """Test case demonstrating XML parse mode with PromptTemplate."""
    print('\nTest case: test_xml_parse_mode_with_template')
    extractor_agent = CustomizeAgent(name='DataExtractor', description='Extracts structured data', prompt_template=StringTemplate(instruction='Extract key information from the provided `text`', context='You are an expert at extracting structured information from text.', constraints=['Use XML tags to structure the output', 'Extract all relevant people and places', 'Maintain original spelling of names'], demonstrations=[{'text': 'Sarah and Mike went to Paris.', 'output': 'Found the following information:\n                    <people>Sarah and Mike</people>\n                    <places>Paris</places>'}]), llm_config=model_config, inputs=[{'name': 'text', 'type': 'string', 'description': 'The text to extract information from'}], outputs=[{'name': 'people', 'type': 'string', 'description': 'Names of people mentioned'}, {'name': 'places', 'type': 'string', 'description': 'Locations mentioned'}], parse_mode='xml')
    sample_text = 'John and Jane visited New York and London last summer.'
    message = extractor_agent(inputs={'text': sample_text})
    print(f'Response from {extractor_agent.name}:')
    print('People:', message.content.people)
    print('Places:', message.content.places)

def main(goal=None):
    openai_config = OpenAILLMConfig(model='gpt-4o-mini', openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    llm = OpenAILLM(config=openai_config)
    goal = "Read and analyze the candidate's pdf resume at examples/output/direction/test_pdf.pdf, and recommend one future PHD directions based on the resume. You should provide a list of 5 review papers about the topic for the candidate to learn more about this direction as well."
    helper_prompt = 'The input is one parameter called "goal", and the output is a markdown report. \n    You should firstly read the pdf resume and summarize the background and recommend one future PHD direction based on the resume.\n    Then you should find 3 trending Review Papers about the topic by searching the keyword on arxiv (by searching web instead of using your out-dated training data) and provide the link of the papers.\n    Lastly you should summarize all the information and provide a detailed markdown report.\n    If you cannot find the papers, you should say "I cannot find the papers".\n    '
    goal += helper_prompt
    mcp_Toolkit = MCPToolkit(config_path=mcp_config_path)
    tools = mcp_Toolkit.get_toolkits()
    tools.append(FileToolkit())
    workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(module_save_path)
    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f'Direction recommendations have been saved to {output_file}')
    except Exception as e:
        print(f'Error saving direction recommendations: {e}')
    print(output)

def main():
    openai_config = OpenAILLMConfig(model='gpt-4o', openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    llm = OpenAILLM(config=openai_config)
    keywords = 'medical, multiagent'
    max_results = 10
    date_from = '2024-01-01'
    categories = ['cs.AI', 'cs.LG']
    search_constraints = f'\n    Search constraints:\n    - Query keywords: {keywords}\n    - Max results: {max_results}\n    - Date from: {date_from}\n    - Categories: {', '.join(categories)}\n    '
    goal = f'Create a daily research paper recommendation assistant that takes user keywords and pushes new relevant papers with summaries.\n\n    The assistant should:\n    1. Use the ArxivToolkit to search for the latest papers using the given keywords.\n    2. Apply the following search constraints:\n    {search_constraints}\n    3. Summarize the search results.\n    4. Compile the summaries into a well-formatted Markdown digest.\n\n    ### Output\n    daily_paper_digest\n    '
    target_directory = 'EvoAgentX/examples/output/paper_push'
    module_save_path = os.path.join(target_directory, 'paper_push_workflow.json')
    result_path = os.path.join(target_directory, 'daily_paper_digest.md')
    os.makedirs(target_directory, exist_ok=True)
    arxiv_toolkit = ArxivToolkit()
    tools = [arxiv_toolkit, FileToolkit()]
    wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
    workflow_graph.save_module(module_save_path)
    workflow_graph.display()
    agent_manager = AgentManager(tools=tools)
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f'✅ Your file has been saved to：{result_path}')
    print('📬 You can run this script everyday to obtain daily recommendation')

