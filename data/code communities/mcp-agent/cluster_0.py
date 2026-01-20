# Cluster 0

def load_settings_class(file_path: Path) -> Tuple[type[BaseSettings], Dict[str, Dict[str, str]]]:
    """Load Settings class from a Python file."""
    src_dir = file_path.parent.parent.parent / 'src'
    sys.path.insert(0, str(src_dir))
    create_mock_modules()
    namespace = {'BaseModel': BaseModel, 'BaseSettings': BaseSettings, 'Path': Path, 'Dict': dict, 'List': list, 'Literal': str}
    with open(file_path, mode='r', encoding='utf-8') as f:
        content = f.read()
    model_info = extract_model_info(content)
    exec(content, namespace)
    return (namespace['Settings'], model_info)

def create_mock_modules() -> None:
    """Create mock modules for imports we want to ignore."""
    mocked_modules = ['opentelemetry', 'opentelemetry.sdk', 'opentelemetry.sdk.trace', 'opentelemetry.sdk.resources', 'opentelemetry.exporter.otlp.proto.http', 'opentelemetry.trace', 'mcp_agent.logging', 'mcp_agent.logging.logger', 'yaml']
    for module_name in mocked_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = MockModule()

def extract_model_info(content: str) -> Dict[str, Dict[str, str]]:
    """
    Extract docstrings for all models and their fields.
    Returns a dict mapping model names to their field descriptions.
    """
    models = {}
    current_model = None
    lines = content.splitlines()
    for i, line in enumerate(lines):
        class_match = re.match('\\s*class\\s+(\\w+)(?:\\([^)]+\\))?\\s*:', line.strip())
        if class_match:
            current_model = class_match.group(1)
            models[current_model] = {'__doc__': ''}
            for j in range(i + 1, min(i + 4, len(lines))):
                doc_match = re.match('\\s*"""(.+?)"""', lines[j], re.DOTALL)
                if doc_match:
                    models[current_model]['__doc__'] = doc_match.group(1).strip()
                    break
            continue
        if current_model:
            if line and (not line.startswith(' ')) and (not line.startswith('#')):
                current_model = None
                continue
            field_match = re.match('\\s+(\\w+)\\s*:', line)
            if field_match:
                field_name = field_match.group(1)
                if field_name in ('model_config', 'Config'):
                    continue
                description = None
                field_desc_match = re.search('Field\\([^)]*description="([^"]+)"', line)
                if field_desc_match:
                    description = field_desc_match.group(1).strip()
                else:
                    for j in range(i + 1, min(i + 4, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and (not next_line.startswith('"""')):
                            break
                        doc_match = re.match('\\s*"""(.+?)"""', lines[j], re.DOTALL)
                        if doc_match:
                            description = doc_match.group(1).strip()
                            break
                if description:
                    models[current_model][field_name] = description
    console.print('\nFound models and their field descriptions:')
    for model, fields in models.items():
        console.print(f'\n[bold]{model}[/bold]: {fields.get('__doc__', '')}')
        for field, desc in fields.items():
            if field != '__doc__':
                console.print(f'  {field}: {desc}')
    return models

@app.command()
def generate(config_py: Path=typer.Option(Path('src/mcp_agent/config.py'), '--config', '-c', help='Path to the config.py file'), output: Path=typer.Option(Path('schema/mcp-agent.config.schema.json'), '--output', '-o', help='Output path for the schema file')):
    """Generate JSON schema from Pydantic models in config.py"""
    if not config_py.exists():
        console.print(f'[red]Error:[/] File not found: {config_py}')
        raise typer.Exit(1)
    try:
        Settings, model_info = load_settings_class(config_py)
        schema = Settings.model_json_schema()
        console.print('\nSchema structure:')
        if '$defs' in schema:
            console.print('Found models in $defs:', list(schema['$defs'].keys()))
        schema.update({'$schema': 'http://json-schema.org/draft-07/schema#', 'title': 'MCP Agent Configuration Schema', 'description': 'Configuration schema for MCP Agent applications'})
        apply_descriptions_to_schema(schema, model_info)
        output.parent.mkdir(parents=True, exist_ok=True)
        output = output.absolute()
        with open(output, 'w') as f:
            json.dump(schema, f, indent=2)
        console.print(f'[green]✓[/] Schema written to: {output}')
        try:
            rel_path = f'./{output.relative_to(Path.cwd())}'
        except ValueError:
            rel_path = str(output)
        vscode_settings = {'yaml.schemas': {rel_path: ['mcp-agent.config.yaml', 'mcp_agent.config.yaml', 'mcp-agent.secrets.yaml', 'mcp_agent.secrets.yaml']}}
        console.print('\n[yellow]VS Code Integration:[/]')
        console.print('Add this to .vscode/settings.json:')
        console.print(json.dumps(vscode_settings, indent=2))
    except Exception as e:
        console.print(f'[red]Error generating schema:[/] {str(e)}')
        raise typer.Exit(1)

def apply_descriptions_to_schema(schema: Dict[str, Any], model_info: Dict[str, Dict[str, str]]) -> None:
    """Recursively apply descriptions to schema and all its nested models."""
    if not isinstance(schema, dict):
        return
    if '$defs' in schema:
        for model_name, model_schema in schema['$defs'].items():
            if model_name in model_info:
                doc = model_info[model_name].get('__doc__', '').strip()
                if doc:
                    model_schema['description'] = doc
                if 'properties' in model_schema:
                    for field_name, field_schema in model_schema['properties'].items():
                        if field_name in model_info[model_name]:
                            field_schema['description'] = model_info[model_name][field_name].strip()
    if 'properties' in schema:
        for field_name, field_schema in schema['properties'].items():
            if 'Settings' in model_info and field_name in model_info['Settings']:
                field_schema['description'] = model_info['Settings'][field_name].strip()

