# Cluster 0

def load_agents_from_folder(user_guid=None):
    agents_directory = os.path.join(os.path.dirname(__file__), 'agents')
    files_in_agents_directory = os.listdir(agents_directory)
    agent_files = [f for f in files_in_agents_directory if f.endswith('.py') and f not in ['__init__.py', 'basic_agent.py']]
    declared_agents = {}
    for file in agent_files:
        try:
            module_name = file[:-3]
            module = importlib.import_module(f'agents.{module_name}')
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BasicAgent) and (obj is not BasicAgent):
                    agent_instance = obj()
                    declared_agents[agent_instance.name] = agent_instance
        except Exception as e:
            logging.error(f'Error loading agent {file}: {str(e)}')
            continue
    storage_manager = AzureFileStorageManager()
    enabled_agents = None
    if user_guid:
        try:
            agent_config_path = f'agent_config/{user_guid}'
            agent_config_content = storage_manager.read_file(agent_config_path, 'enabled_agents.json')
            if agent_config_content:
                enabled_agents = json.loads(agent_config_content)
        except Exception as e:
            logging.info(f'No agent config found for GUID {user_guid}, loading all agents: {str(e)}')
    try:
        agent_files = storage_manager.list_files('agents')
        for file in agent_files:
            if not file.name.endswith('_agent.py'):
                continue
            if enabled_agents is not None:
                if file.name not in enabled_agents:
                    continue
            try:
                file_content = storage_manager.read_file('agents', file.name)
                if file_content is None:
                    continue
                temp_dir = '/tmp/agents'
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = f'{temp_dir}/{file.name}'
                with open(temp_file, 'w') as f:
                    f.write(file_content)
                if temp_dir not in sys.path:
                    sys.path.append(temp_dir)
                module_name = file.name[:-3]
                spec = importlib.util.spec_from_file_location(module_name, temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BasicAgent) and (obj is not BasicAgent):
                        agent_instance = obj()
                        declared_agents[agent_instance.name] = agent_instance
                os.remove(temp_file)
            except Exception as e:
                logging.error(f'Error loading agent {file.name} from Azure File Share: {str(e)}')
                continue
    except Exception as e:
        logging.error(f'Error loading agents from Azure File Share: {str(e)}')
    try:
        multi_agent_files = storage_manager.list_files('multi_agents')
        for file in multi_agent_files:
            if not file.name.endswith('_agent.py'):
                continue
            if enabled_agents is not None:
                if file.name not in enabled_agents:
                    continue
            try:
                file_content = storage_manager.read_file('multi_agents', file.name)
                if file_content is None:
                    continue
                temp_dir = '/tmp/multi_agents'
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = f'{temp_dir}/{file.name}'
                with open(temp_file, 'w') as f:
                    f.write(file_content)
                if temp_dir not in sys.path:
                    sys.path.append(temp_dir)
                parent_dir = '/tmp'
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)
                module_name = file.name[:-3]
                spec = importlib.util.spec_from_file_location(f'multi_agents.{module_name}', temp_file)
                module = importlib.util.module_from_spec(spec)
                import types
                if 'multi_agents' not in sys.modules:
                    multi_agents_module = types.ModuleType('multi_agents')
                    sys.modules['multi_agents'] = multi_agents_module
                sys.modules[f'multi_agents.{module_name}'] = module
                spec.loader.exec_module(module)
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BasicAgent) and (obj is not BasicAgent):
                        agent_instance = obj()
                        declared_agents[agent_instance.name] = agent_instance
                        logging.info(f'Loaded multi-agent: {agent_instance.name}')
                os.remove(temp_file)
            except Exception as e:
                logging.error(f'Error loading multi-agent {file.name} from Azure File Share: {str(e)}')
                continue
    except Exception as e:
        logging.error(f'Error loading multi-agents from Azure File Share: {str(e)}')
    return declared_agents

@app.route(route='businessinsightbot_function', auth_level=func.AuthLevel.FUNCTION)
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    origin = req.headers.get('origin')
    cors_headers = build_cors_response(origin)
    if req.method == 'OPTIONS':
        return func.HttpResponse(status_code=200, headers=cors_headers)
    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse('Invalid JSON in request body', status_code=400, headers=cors_headers)
    if not req_body:
        return func.HttpResponse('Missing JSON payload in request body', status_code=400, headers=cors_headers)
    user_input = req_body.get('user_input')
    if user_input is None:
        user_input = ''
    else:
        user_input = str(user_input)
    conversation_history = req_body.get('conversation_history', [])
    if not isinstance(conversation_history, list):
        conversation_history = []
    user_guid = req_body.get('user_guid')
    is_guid_only = re.match('^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', user_input.strip(), re.IGNORECASE)
    if not is_guid_only and (not user_input.strip()):
        return func.HttpResponse(json.dumps({'error': 'Missing or empty user_input in JSON payload'}), status_code=400, mimetype='application/json', headers=cors_headers)
    try:
        agents = load_agents_from_folder(user_guid)
        assistant = Assistant(agents)
        if user_guid:
            assistant.user_guid = user_guid
            assistant._initialize_context_memory(user_guid)
        elif is_guid_only:
            assistant.user_guid = user_input.strip()
            assistant._initialize_context_memory(user_input.strip())
        assistant_response, voice_response, agent_logs = assistant.get_response(user_input, conversation_history)
        response = {'assistant_response': str(assistant_response), 'voice_response': str(voice_response), 'agent_logs': str(agent_logs), 'user_guid': assistant.user_guid}
        return func.HttpResponse(json.dumps(response), mimetype='application/json', headers=cors_headers)
    except Exception as e:
        error_response = {'error': 'Internal server error', 'details': str(e)}
        return func.HttpResponse(json.dumps(error_response), status_code=500, mimetype='application/json', headers=cors_headers)

def build_cors_response(origin):
    """
    Builds CORS response headers.
    Safely handles None origin.
    """
    return {'Access-Control-Allow-Origin': str(origin) if origin else '*', 'Access-Control-Allow-Methods': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Credentials': 'true', 'Access-Control-Max-Age': '86400'}

