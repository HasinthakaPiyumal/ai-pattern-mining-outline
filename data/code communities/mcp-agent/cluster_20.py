# Cluster 20

@app.command('show')
def show(secrets: bool=typer.Option(False, '--secrets', '-s', help='Show secrets file'), path: Optional[Path]=typer.Option(None, '--path', '-p', help='Explicit path'), raw: bool=typer.Option(False, '--raw', '-r', help='Show raw YAML without validation')) -> None:
    """Display the current config or secrets file with YAML validation."""
    file_path = path
    if file_path is None:
        file_path = _find_secrets_file() if secrets else _find_config_file()
    if not file_path or not file_path.exists():
        typer.secho('Config file not found', fg=typer.colors.RED, err=True)
        console.print('\n[dim]Hint: Run [cyan]mcp-agent config builder[/cyan] to create one[/dim]')
        raise typer.Exit(2)
    try:
        text = file_path.read_text(encoding='utf-8')
        if raw:
            console.print(text)
            return
        parsed = yaml.safe_load(text)
        console.print(Panel(f'[bold cyan]{file_path}[/bold cyan]\nSize: {file_path.stat().st_size} bytes\nModified: {Path(file_path).stat().st_mtime}', title=f'[bold]{('Secrets' if secrets else 'Config')} File[/bold]', border_style='cyan'))
        if parsed is None:
            console.print('\n[yellow]⚠️  File is empty[/yellow]')
        else:
            console.print('\n[green]✅ YAML syntax is valid[/green]')
            console.print('\n[bold]Structure:[/bold]')
            for key in parsed.keys():
                if isinstance(parsed[key], dict):
                    console.print(f'  • {key}: {len(parsed[key])} items')
                else:
                    console.print(f'  • {key}: {type(parsed[key]).__name__}')
        console.print('\n[bold]Content:[/bold]')
        from rich.syntax import Syntax
        syntax = Syntax(text, 'yaml', theme='monokai', line_numbers=True)
        console.print(syntax)
    except yaml.YAMLError as e:
        console.print(f'[red]❌ YAML syntax error: {e}[/red]')
        console.print('\n[yellow]Raw content:[/yellow]')
        console.print(text)
        raise typer.Exit(5)
    except Exception as e:
        typer.secho(f'Error reading file: {e}', fg=typer.colors.RED, err=True)
        raise typer.Exit(5)

def _find_secrets_file() -> Optional[Path]:
    return Settings.find_secrets()

def _find_config_file() -> Optional[Path]:
    return Settings.find_config()

@app.command('check')
def check(verbose: bool=typer.Option(False, '--verbose', '-v', help='Show detailed information')) -> None:
    """Check and summarize configuration status."""
    if verbose:
        LOG_VERBOSE.set(True)
    verbose = LOG_VERBOSE.get()
    cfg = _find_config_file()
    sec = _find_secrets_file()
    table = Table(show_header=False, box=None)
    table.add_column('Key', style='cyan', width=20)
    table.add_column('Value')
    table.add_row('Config file', str(cfg) if cfg else '[red]Not found[/red]')
    table.add_row('Secrets file', str(sec) if sec else '[yellow]Not found[/yellow]')
    if not cfg:
        console.print(Panel(table, title='[bold]Configuration Status[/bold]', border_style='red'))
        console.print('\n[dim]Run [cyan]mcp-agent config builder[/cyan] to create configuration[/dim]')
        raise typer.Exit(1)
    try:
        settings = get_settings()
        table.add_row('', '')
        table.add_row('[bold]Engine[/bold]', '')
        table.add_row('Execution', settings.execution_engine or 'asyncio')
        if settings.logger:
            table.add_row('', '')
            table.add_row('[bold]Logger[/bold]', '')
            table.add_row('Type', settings.logger.type or 'none')
            table.add_row('Level', settings.logger.level or 'info')
            if settings.logger.type == 'file':
                table.add_row('Path', str(settings.logger.path_settings.path_pattern if settings.logger.path_settings else 'Not set'))
        if settings.otel and settings.otel.enabled:
            table.add_row('', '')
            table.add_row('[bold]OpenTelemetry[/bold]', '')
            table.add_row('Enabled', '[green]Yes[/green]')
            table.add_row('Sample rate', str(settings.otel.sample_rate))
            if settings.otel.exporters:
                table.add_row('Exporters', ', '.join((str(e) for e in settings.otel.exporters)))
        table.add_row('', '')
        table.add_row('[bold]MCP Servers[/bold]', '')
        if settings.mcp and settings.mcp.servers:
            servers = list(settings.mcp.servers.keys())
            table.add_row('Count', str(len(servers)))
            if verbose:
                for name in servers[:5]:
                    server = settings.mcp.servers[name]
                    status = '✅' if server.transport == 'stdio' else '🌐'
                    table.add_row(f'  {status} {name}', server.transport)
                if len(servers) > 5:
                    table.add_row('  ...', f'and {len(servers) - 5} more')
            else:
                table.add_row('Names', ', '.join(servers[:3]) + ('...' if len(servers) > 3 else ''))
        else:
            table.add_row('Count', '[yellow]0[/yellow]')
        table.add_row('', '')
        table.add_row('[bold]Providers[/bold]', '')
        providers = [('OpenAI', settings.openai, 'api_key'), ('Anthropic', settings.anthropic, 'api_key'), ('Google', settings.google, 'api_key'), ('Azure', settings.azure, 'api_key')]
        configured = []
        for name, obj, field in providers:
            if obj and getattr(obj, field, None):
                configured.append(name)
            elif os.getenv(f'{name.upper()}_API_KEY'):
                configured.append(f'{name} (env)')
        if configured:
            table.add_row('Configured', ', '.join(configured))
        else:
            table.add_row('Configured', '[yellow]None[/yellow]')
        status_color = 'green' if configured else 'yellow'
        console.print(Panel(table, title='[bold]Configuration Status[/bold]', border_style=status_color))
        warnings = []
        if not sec or not sec.exists():
            warnings.append('No secrets file found - API keys should be in environment variables')
        if not configured:
            warnings.append('No AI providers configured - add API keys to use agents')
        if settings.mcp and (not settings.mcp.servers):
            warnings.append("No MCP servers configured - agents won't have tool access")
        if warnings:
            console.print('\n[yellow]⚠️  Warnings:[/yellow]')
            for warning in warnings:
                console.print(f'  • {warning}')
        if verbose:
            console.print('\n[dim]Run [cyan]mcp-agent doctor[/cyan] for detailed diagnostics[/dim]')
    except Exception as e:
        table.add_row('', '')
        table.add_row('Error', f'[red]{e}[/red]')
        console.print(Panel(table, title='[bold]Configuration Status[/bold]', border_style='red'))
        raise typer.Exit(5)

@app.command('edit')
def edit(secrets: bool=typer.Option(False, '--secrets', '-s', help='Edit secrets file'), editor: Optional[str]=typer.Option(None, '--editor', '-e', help='Editor to use')) -> None:
    """Open config or secrets in an editor."""
    target = _find_secrets_file() if secrets else _find_config_file()
    if not target:
        console.print(f'[red]No {('secrets' if secrets else 'config')} file found[/red]')
        if Confirm.ask('Create one now?', default=True):
            builder()
            return
        raise typer.Exit(2)
    import subprocess
    if editor:
        editors = [editor]
    else:
        editor = os.environ.get('EDITOR') or os.environ.get('VISUAL')
        editors = [editor] if editor else []
        editors += ['code --wait', 'nano', 'vim', 'vi', 'emacs']
    for cmd in editors:
        if not cmd:
            continue
        try:
            console.print(f'\n[cyan]Opening {target.name} in editor...[/cyan]')
            console.print('[dim]Save and close the editor to continue.[/dim]\n')
            if ' ' in cmd:
                parts = cmd.split()
                subprocess.run(parts + [str(target)], check=True)
            else:
                subprocess.run([cmd, str(target)], check=True)
            console.print('\n[bold]Validating edited file...[/bold]')
            try:
                yaml.safe_load(target.read_text())
                console.print('[green]✅ File is valid YAML[/green]')
            except yaml.YAMLError as e:
                console.print(f'[red]⚠️  YAML syntax error: {e}[/red]')
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    console.print('[yellow]No editor found. File location:[/yellow]')
    console.print(str(target))

@app.command('builder')
def builder(expert: bool=typer.Option(False, '--expert', help='Expert mode with all options'), template: Optional[str]=typer.Option(None, '--template', '-t', help='Start from template')) -> None:
    """Interactive configuration builder."""
    console.print('\n[bold cyan]🔧 MCP-Agent Configuration Builder[/bold cyan]\n')
    existing_config = _find_config_file()
    existing_secrets = _find_secrets_file()
    if existing_config and existing_config.exists():
        console.print(f'[yellow]⚠️  Config file exists: {existing_config}[/yellow]')
        if not Confirm.ask('Overwrite?', default=False):
            raise typer.Exit(0)
    config: Dict[str, Any] = {}
    secrets: Dict[str, Any] = {}
    if template:
        template_map = {'basic': 'mcp_agent.config.yaml', 'claude': 'config_claude.yaml', 'server': 'config_server.yaml'}
        template_file = template_map.get(template, template)
        template_content = _load_template(template_file)
        if template_content:
            try:
                config = yaml.safe_load(template_content) or {}
                console.print(f'[green]Loaded template: {template}[/green]')
            except Exception as e:
                console.print(f'[red]Failed to load template: {e}[/red]')
    console.print('\n[bold]Basic Configuration[/bold]')
    config['execution_engine'] = Prompt.ask('Execution engine', default=config.get('execution_engine', 'asyncio'), choices=['asyncio', 'temporal'])
    console.print('\n[bold]Logger Configuration[/bold]')
    logger_type = Prompt.ask('Logger type', default='console', choices=['none', 'console', 'file', 'http'])
    config.setdefault('logger', {})
    config['logger']['type'] = logger_type
    if logger_type != 'none':
        config['logger']['level'] = Prompt.ask('Log level', default='info', choices=['debug', 'info', 'warning', 'error'])
        if logger_type == 'console':
            config['logger']['transports'] = ['console']
        elif logger_type == 'file':
            config['logger']['transports'] = ['file']
            config['logger']['path_settings'] = {'path_pattern': Prompt.ask('Log file pattern', default='logs/mcp-agent-{unique_id}.jsonl'), 'unique_id': Prompt.ask('Unique ID type', default='timestamp', choices=['timestamp', 'session_id'])}
    if expert:
        console.print('\n[bold]OpenTelemetry Configuration[/bold]')
        if Confirm.ask('Enable OpenTelemetry?', default=False):
            config.setdefault('otel', {})
            config['otel']['enabled'] = True
            config['otel']['service_name'] = Prompt.ask('Service name', default='mcp-agent')
            config['otel']['endpoint'] = Prompt.ask('OTLP endpoint', default='http://localhost:4317')
            config['otel']['sample_rate'] = float(Prompt.ask('Sample rate (0.0-1.0)', default='1.0'))
    console.print('\n[bold]MCP Server Configuration[/bold]')
    config.setdefault('mcp', {})
    config['mcp'].setdefault('servers', {})
    if Confirm.ask('Add filesystem server?', default=True):
        config['mcp']['servers']['filesystem'] = {'transport': 'stdio', 'command': 'npx', 'args': ['-y', '@modelcontextprotocol/server-filesystem', '.']}
    if Confirm.ask('Add web fetch server?', default=True):
        config['mcp']['servers']['fetch'] = {'transport': 'stdio', 'command': 'uvx', 'args': ['mcp-server-fetch']}
    if Confirm.ask('Add more servers?', default=False):
        from mcp_agent.cli.commands.server import SERVER_RECIPES
        categories = {}
        for name, recipe in SERVER_RECIPES.items():
            cat = recipe.get('category', 'other')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        console.print('\n[bold]Available server recipes:[/bold]')
        for cat, names in sorted(categories.items()):
            console.print(f'  [cyan]{cat}:[/cyan] {', '.join(names[:5])}')
        while True:
            server_name = Prompt.ask("\nServer recipe name (or 'done')")
            if server_name.lower() == 'done':
                break
            if server_name in SERVER_RECIPES:
                recipe = SERVER_RECIPES[server_name]
                config['mcp']['servers'][server_name] = {'transport': recipe['transport'], 'command': recipe.get('command'), 'args': recipe.get('args', [])}
                console.print(f'[green]Added: {server_name}[/green]')
                if recipe.get('env_required'):
                    console.print(f'[yellow]Note: Requires {', '.join(recipe['env_required'])}[/yellow]')
            else:
                console.print(f'[red]Unknown recipe: {server_name}[/red]')
    console.print('\n[bold]AI Provider Configuration[/bold]')
    providers = [('openai', 'OpenAI', 'gpt-4o-mini'), ('anthropic', 'Anthropic', 'claude-3-5-sonnet-20241022'), ('google', 'Google', 'gemini-1.5-pro')]
    for key, name, default_model in providers:
        if Confirm.ask(f'Configure {name}?', default=key in ['openai', 'anthropic']):
            config.setdefault(key, {})
            config[key]['default_model'] = Prompt.ask(f'{name} default model', default=default_model)
            if Confirm.ask(f'Add {name} API key to secrets?', default=True):
                api_key = Prompt.ask(f'{name} API key', password=True)
                if api_key and api_key != 'skip':
                    secrets.setdefault(key, {})
                    secrets[key]['api_key'] = api_key
    config['$schema'] = 'https://raw.githubusercontent.com/lastmile-ai/mcp-agent/refs/heads/main/schema/mcp-agent.config.schema.json'
    config_path = existing_config or Path.cwd() / 'mcp_agent.config.yaml'
    with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'), console=console) as progress:
        progress.add_task('Writing configuration files...', total=None)
        try:
            config_yaml = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
            config_path.write_text(config_yaml, encoding='utf-8')
            console.print(f'[green]✅ Created:[/green] {config_path}')
            if secrets:
                secrets_path = existing_secrets or Path.cwd() / 'mcp_agent.secrets.yaml'
                template_secrets = _load_template('mcp_agent.secrets.yaml')
                if template_secrets:
                    base_secrets = yaml.safe_load(template_secrets) or {}
                    for key, value in secrets.items():
                        if key in base_secrets and isinstance(base_secrets[key], dict):
                            base_secrets[key].update(value)
                        else:
                            base_secrets[key] = value
                    secrets = base_secrets
                secrets_yaml = yaml.safe_dump(secrets, sort_keys=False, default_flow_style=False)
                secrets_path.write_text(secrets_yaml, encoding='utf-8')
                console.print(f'[green]✅ Created:[/green] {secrets_path}')
                try:
                    import stat
                    os.chmod(secrets_path, stat.S_IRUSR | stat.S_IWUSR)
                    console.print('[dim]Set secure permissions on secrets file[/dim]')
                except Exception:
                    pass
            gitignore = Path.cwd() / '.gitignore'
            if not gitignore.exists() or 'mcp_agent.secrets.yaml' not in gitignore.read_text():
                if Confirm.ask('Add secrets to .gitignore?', default=True):
                    with open(gitignore, 'a') as f:
                        f.write('\n# MCP-Agent\nmcp_agent.secrets.yaml\n*.secrets.yaml\n')
                    console.print('[green]✅ Updated .gitignore[/green]')
        except Exception as e:
            console.print(f'[red]Error writing files: {e}[/red]')
            raise typer.Exit(5)
    console.print('\n[bold green]✅ Configuration complete![/bold green]\n')
    table = Table(show_header=False, box=None)
    table.add_column('Item', style='cyan')
    table.add_column('Status')
    table.add_row('Config file', str(config_path))
    table.add_row('MCP servers', str(len(config.get('mcp', {}).get('servers', {}))))
    table.add_row('Providers', ', '.join((k for k in ['openai', 'anthropic', 'google'] if k in config)))
    console.print(Panel(table, title='[bold]Summary[/bold]', border_style='green'))
    console.print('\n[bold]Next steps:[/bold]')
    console.print('1. Review configuration: [cyan]mcp-agent config show[/cyan]')
    console.print('2. Test configuration: [cyan]mcp-agent doctor[/cyan]')
    console.print('3. Test servers: [cyan]mcp-agent server test <name>[/cyan]')
    console.print('4. Start chatting: [cyan]mcp-agent chat[/cyan]')

def _load_template(template_name: str) -> str:
    """Load a template file from the data/templates directory."""
    try:
        from importlib import resources
        with resources.files('mcp_agent.data.templates').joinpath(template_name).open() as file:
            return file.read()
    except Exception as e:
        console.print(f'[red]Error loading template {template_name}: {e}[/red]')
        return ''

@app.command('validate')
def validate(config_file: Optional[Path]=typer.Option(None, '--config', '-c', help='Config file path'), secrets_file: Optional[Path]=typer.Option(None, '--secrets', '-s', help='Secrets file path'), schema: Optional[str]=typer.Option(None, '--schema', help='Schema URL or path')) -> None:
    """Validate configuration files against schema."""
    config_path = config_file or _find_config_file()
    secrets_path = secrets_file or _find_secrets_file()
    if not config_path or not config_path.exists():
        console.print('[red]Config file not found[/red]')
        raise typer.Exit(1)
    console.print('[bold]Validating configuration files...[/bold]\n')
    errors = []
    warnings = []
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        console.print('[green]✅[/green] Config YAML syntax valid')
    except yaml.YAMLError as e:
        errors.append(f'Config YAML error: {e}')
        config = None
    if secrets_path and secrets_path.exists():
        try:
            with open(secrets_path) as f:
                yaml.safe_load(f)
            console.print('[green]✅[/green] Secrets YAML syntax valid')
        except yaml.YAMLError as e:
            errors.append(f'Secrets YAML error: {e}')
    else:
        warnings.append('No secrets file found')
    if schema:
        try:
            import jsonschema
            import requests
            if schema.startswith('http'):
                response = requests.get(schema)
                schema_data = response.json()
            else:
                with open(schema) as f:
                    schema_data = json.load(f)
            jsonschema.validate(config, schema_data)
            console.print('[green]✅[/green] Config validates against schema')
        except ImportError:
            warnings.append('jsonschema not installed - skipping schema validation')
        except Exception as e:
            errors.append(f'Schema validation error: {e}')
    try:
        settings = get_settings()
        console.print('[green]✅[/green] Settings load successfully')
        if settings.mcp and settings.mcp.servers:
            for name, server in settings.mcp.servers.items():
                if server.transport == 'stdio' and (not server.command):
                    warnings.append(f"Server '{name}' missing command")
                elif server.transport in ['http', 'sse'] and (not server.url):
                    warnings.append(f"Server '{name}' missing URL")
    except Exception as e:
        errors.append(f'Settings load error: {e}')
    console.print()
    if errors:
        console.print('[bold red]Errors:[/bold red]')
        for error in errors:
            console.print(f'  ❌ {error}')
    if warnings:
        console.print('\n[bold yellow]Warnings:[/bold yellow]')
        for warning in warnings:
            console.print(f'  ⚠️  {warning}')
    if not errors:
        console.print('\n[bold green]✅ Configuration is valid![/bold green]')
    else:
        raise typer.Exit(1)

def _write_readme(dir_path: Path, content: str, force: bool) -> str | None:
    """Create a README file with fallback naming if a README already exists.

    Returns the filename created, or None if it could not be written (in which case
    the content is printed to console as a fallback).
    """
    candidates = ['README.md', 'README.mcp-agent.md', 'README.mcp.md']
    candidates += [f'README.{i}.md' for i in range(1, 6)]
    for name in candidates:
        path = dir_path / name
        if not path.exists() or force:
            ok = _write(path, content, force)
            if ok:
                return name
    console.print('\n[yellow]A README already exists and could not be overwritten.[/yellow]')
    console.print('[bold]Suggested README contents:[/bold]\n')
    console.print(content)
    return None

def _write(path: Path, content: str, force: bool) -> bool:
    """Write content to a file with optional overwrite confirmation."""
    if path.exists() and (not force):
        if not Confirm.ask(f'{path} exists. Overwrite?', default=False):
            return False
    try:
        path.write_text(content, encoding='utf-8')
        console.print(f'[green]Created[/green] {path}')
        return True
    except Exception as e:
        console.print(f'[red]Error writing {path}: {e}[/red]')
        return False

def _copy_any(node, target: Path):
    if node.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in node.iterdir():
            _copy_any(child, target / child.name)
    else:
        if target.exists() and (not force):
            return
        with node.open('rb') as rf:
            data = rf.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'wb') as wf:
            wf.write(data)

def _copy_pkg_tree(pkg_rel: str, dst: Path, force: bool) -> int:
    """Copy packaged examples from mcp_agent.data/examples/<pkg_rel> into dst.

    Uses importlib.resources to locate files installed with the package.
    Returns 1 on success, 0 on failure.
    """
    try:
        root = resources.files('mcp_agent.data').joinpath('examples').joinpath(pkg_rel)
    except Exception:
        return 0
    if not root.exists():
        return 0

    def _copy_any(node, target: Path):
        if node.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            for child in node.iterdir():
                _copy_any(child, target / child.name)
        else:
            if target.exists() and (not force):
                return
            with node.open('rb') as rf:
                data = rf.read()
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, 'wb') as wf:
                wf.write(data)
    _copy_any(root, dst)
    return 1

@app.callback(invoke_without_command=True)
def init(ctx: typer.Context, dir: Path=typer.Option(Path('.'), '--dir', '-d', help='Target directory'), template: str=typer.Option('basic', '--template', '-t', help='Template to use'), quickstart: str=typer.Option(None, '--quickstart', help='Quickstart mode: copy example without config files'), force: bool=typer.Option(False, '--force', '-f', help='Overwrite existing files'), no_gitignore: bool=typer.Option(False, '--no-gitignore', help='Skip creating .gitignore'), list_templates: bool=typer.Option(False, '--list', '-l', help='List available templates')) -> None:
    """Initialize a new MCP-Agent project with configuration and example files.

    Use --template for full project initialization with config files.
    Use --quickstart for copying examples only."""
    scaffolding_templates = {'basic': 'Simple agent with filesystem and fetch capabilities', 'server': 'MCP server with workflow and parallel agents', 'factory': 'Agent factory with router-based selection', 'minimal': 'Minimal configuration files only'}
    example_templates = {'workflow': 'Workflow examples (from examples/workflows)', 'researcher': 'MCP researcher use case (from examples/usecases/mcp_researcher)', 'data-analysis': 'Financial data analysis example', 'state-transfer': 'Workflow router with state transfer', 'mcp-basic-agent': 'Basic MCP agent example', 'token-counter': 'Token counting with monitoring', 'agent-factory': 'Agent factory pattern', 'basic-agent-server': 'Basic agent server (asyncio)', 'reference-agent-server': 'Reference agent server implementation', 'elicitation': 'Elicitation server example', 'sampling': 'Sampling server example', 'notifications': 'Notifications server example', 'hello-world': 'Basic hello world cloud example', 'mcp': 'Comprehensive MCP server example with tools, sampling, elicitation', 'temporal': 'Temporal integration with durable workflows', 'chatgpt-app': 'ChatGPT App with interactive UI widgets'}
    templates = {**scaffolding_templates, **example_templates}
    example_map = {'workflow': ('workflow', 'workflows'), 'researcher': ('researcher', 'usecases/mcp_researcher'), 'data-analysis': ('data-analysis', 'usecases/mcp_financial_analyzer'), 'state-transfer': ('state-transfer', 'workflows/workflow_router'), 'basic-agent-server': ('basic_agent_server', 'mcp_agent_server/asyncio'), 'mcp-basic-agent': ('mcp_basic_agent', 'basic/mcp_basic_agent'), 'token-counter': ('token_counter', 'basic/token_counter'), 'agent-factory': ('agent_factory', 'basic/agent_factory'), 'reference-agent-server': ('reference_agent_server', 'mcp_agent_server/reference'), 'elicitation': ('elicitation', 'mcp_agent_server/elicitation'), 'sampling': ('sampling', 'mcp_agent_server/sampling'), 'notifications': ('notifications', 'mcp_agent_server/notifications'), 'hello-world': ('hello_world', 'cloud/hello_world'), 'mcp': ('mcp', 'cloud/mcp'), 'temporal': ('temporal', 'cloud/temporal'), 'chatgpt-app': ('chatgpt_app', 'cloud/chatgpt_app')}
    if list_templates:
        console.print('\n[bold]Available Templates:[/bold]\n')
        console.print('[bold cyan]Templates:[/bold cyan]')
        console.print('[dim]Creates minimal project structure with config files[/dim]\n')
        table1 = Table(show_header=True, header_style='cyan')
        table1.add_column('Template', style='green')
        table1.add_column('Description')
        for name, desc in scaffolding_templates.items():
            table1.add_row(name, desc)
        console.print(table1)
        console.print('\n[bold cyan]Quickstart Templates:[/bold cyan]')
        console.print('[dim]Copies complete example projects[/dim]\n')
        table2 = Table(show_header=True, header_style='cyan')
        table2.add_column('Template', style='green')
        table2.add_column('Description')
        for name, desc in example_templates.items():
            table2.add_row(name, desc)
        console.print(table2)
        console.print('\n[dim]Use: mcp-agent init --template <name>[/dim]')
        return
    if ctx.invoked_subcommand:
        return
    if quickstart:
        if quickstart not in example_templates:
            console.print(f'[red]Unknown quickstart example: {quickstart}[/red]')
            console.print(f'Available examples: {', '.join(example_templates.keys())}')
            console.print('[dim]Use --list to see all available templates[/dim]')
            raise typer.Exit(1)
        mapping = example_map.get(quickstart)
        if not mapping:
            console.print(f"[red]Quickstart example '{quickstart}' not found[/red]")
            raise typer.Exit(1)
        base_dir = dir.resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        dst_name, pkg_rel = mapping
        dst = base_dir / dst_name
        copied = _copy_pkg_tree(pkg_rel, dst, force)
        if copied:
            console.print(f'Copied {copied} set(s) to {dst}')
        else:
            console.print(f"[yellow]Could not copy '{quickstart}' - destination may already exist[/yellow]")
            console.print('Use --force to overwrite')
        return
    if template not in templates:
        console.print(f'[red]Unknown template: {template}[/red]')
        console.print(f'Available templates: {', '.join(templates.keys())}')
        console.print('[dim]Use --list to see template descriptions[/dim]')
        raise typer.Exit(1)
    dir = dir.resolve()
    dir.mkdir(parents=True, exist_ok=True)
    console.print('\n[bold]Initializing MCP-Agent project[/bold]')
    console.print(f'Directory: [cyan]{dir}[/cyan]')
    console.print(f'Template: [cyan]{template}[/cyan] - {templates[template]}\n')
    files_created = []
    entry_script_name: str | None = None
    config_path = dir / 'mcp_agent.config.yaml'
    config_content = _load_template('mcp_agent.config.yaml')
    if config_content and _write(config_path, config_content, force):
        files_created.append('mcp_agent.config.yaml')
    secrets_path = dir / 'mcp_agent.secrets.yaml'
    secrets_content = _load_template('secrets.yaml')
    if secrets_content and _write(secrets_path, secrets_content, force):
        files_created.append('mcp_agent.secrets.yaml')
    if not no_gitignore:
        gitignore_path = dir / '.gitignore'
        gitignore_content = _load_template('gitignore.template')
        if gitignore_content and _write(gitignore_path, gitignore_content, force):
            files_created.append('.gitignore')
    if template in example_templates:
        mapping = example_map.get(template)
        if not mapping:
            console.print(f"[red]Example template '{template}' not found[/red]")
            raise typer.Exit(1)
        dst_name, pkg_rel = mapping
        dst = dir / dst_name
        copied = _copy_pkg_tree(pkg_rel, dst, force)
        if copied:
            console.print(f"\n[green]✅ Successfully copied example '{template}'![/green]")
            console.print(f'Created: [cyan]{dst}[/cyan]\n')
            console.print('[bold]Next steps:[/bold]')
            console.print(f'1. cd [cyan]{dst}[/cyan]')
            console.print('2. Review the README for instructions')
            console.print('3. Add your API keys to config/secrets files if needed')
        else:
            console.print(f"[yellow]Example '{template}' could not be copied[/yellow]")
            console.print('The destination may already exist. Use --force to overwrite.')
        return
    if template == 'basic':
        script_name = 'main.py'
        script_path = dir / script_name
        agent_content = _load_template('basic_agent.py')
        if agent_content:
            write_force_flag = force
            if script_path.exists() and (not force):
                if Confirm.ask(f'{script_path} exists. Overwrite?', default=False):
                    write_force_flag = True
                else:
                    alt_name = Prompt.ask('Enter a filename to save the agent', default='main.py')
                    if not alt_name.endswith('.py'):
                        alt_name += '.py'
                    script_name = alt_name
                    script_path = dir / script_name
            if _write(script_path, agent_content, write_force_flag):
                files_created.append(script_name)
                entry_script_name = script_name
                try:
                    script_path.chmod(script_path.stat().st_mode | 73)
                except Exception:
                    pass
        readme_content = _load_template('README_basic.md')
        if readme_content:
            created = _write_readme(dir, readme_content, force)
            if created:
                files_created.append(created)
    elif template == 'server':
        server_path = dir / 'main.py'
        server_content = _load_template('basic_agent_server.py')
        if server_content and _write(server_path, server_content, force):
            files_created.append('main.py')
            try:
                server_path.chmod(server_path.stat().st_mode | 73)
            except Exception:
                pass
        readme_content = _load_template('README_server.md')
        if readme_content:
            created = _write_readme(dir, readme_content, force)
            if created:
                files_created.append(created)
    elif template == 'factory':
        factory_path = dir / 'main.py'
        factory_content = _load_template('agent_factory.py')
        if factory_content and _write(factory_path, factory_content, force):
            files_created.append('main.py')
            try:
                factory_path.chmod(factory_path.stat().st_mode | 73)
            except Exception:
                pass
        agents_path = dir / 'agents.yaml'
        agents_content = _load_template('agents.yaml')
        if agents_content and _write(agents_path, agents_content, force):
            files_created.append('agents.yaml')
        run_worker_path = dir / 'run_worker.py'
        run_worker_content = _load_template('agent_factory_run_worker.py')
        if run_worker_content and _write(run_worker_path, run_worker_content, force):
            files_created.append('run_worker.py')
            try:
                run_worker_path.chmod(run_worker_path.stat().st_mode | 73)
            except Exception:
                pass
        readme_content = _load_template('README_factory.md')
        if readme_content:
            created = _write_readme(dir, readme_content, force)
            if created:
                files_created.append(created)
    if files_created:
        console.print('\n[green]✅ Successfully initialized project![/green]')
        console.print(f'Created {len(files_created)} file(s)\n')
        console.print('[bold]Next steps:[/bold]')
        console.print('1. Add your API keys to [cyan]mcp_agent.secrets.yaml[/cyan]')
        console.print('   Or set environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY')
        console.print('2. Review and customize [cyan]mcp_agent.config.yaml[/cyan]')
        if template == 'basic':
            run_file = entry_script_name or 'main.py'
            console.print(f'3. Run your agent: [cyan]uv run {run_file}[/cyan]')
        elif template == 'server':
            console.print('3. Run the server: [cyan]uv run main.py[/cyan]')
            console.print('   Or serve: [cyan]mcp-agent dev serve --script main.py[/cyan]')
        elif template == 'factory':
            console.print('3. Customize agents in [cyan]agents.yaml[/cyan]')
            console.print('4. Run the factory: [cyan]uv run main.py[/cyan]')
            console.print('   Optional: to exercise Temporal locally, run [cyan]temporal server start-dev[/cyan]')
            console.print('             in another terminal and start the worker with [cyan]uv run run_worker.py[/cyan].')
    elif template == 'minimal':
        console.print('3. Create your agent script')
        console.print('   See examples: [cyan]mcp-agent init --list[/cyan]')
        console.print('\n[dim]Run [cyan]mcp-agent doctor[/cyan] to check your configuration[/dim]')
        console.print('[dim]Run [cyan]mcp-agent init --list[/cyan] to see all available templates[/dim]')
    else:
        console.print('\n[yellow]No files were created[/yellow]')

@app.command()
def interactive(dir: Path=typer.Option(Path('.'), '--dir', '-d', help='Target directory')) -> None:
    """Interactive project initialization with prompts."""
    console.print('\n[bold cyan]🚀 MCP-Agent Interactive Setup[/bold cyan]\n')
    project_name = Prompt.ask('Project name', default=dir.name)
    templates = {'1': ('basic', 'Simple agent with filesystem and fetch'), '2': ('server', 'MCP server with workflows'), '3': ('factory', 'Agent factory with routing'), '4': ('minimal', 'Config files only')}
    console.print('\n[bold]Choose a template:[/bold]')
    for key, (name, desc) in templates.items():
        console.print(f'  {key}. [green]{name}[/green] - {desc}')
    choice = Prompt.ask('\nTemplate', choices=list(templates.keys()), default='1')
    template_name, _ = templates[choice]
    console.print('\n[bold]Select AI providers to configure:[/bold]')
    providers = []
    if Confirm.ask('Configure OpenAI?', default=True):
        providers.append('openai')
    if Confirm.ask('Configure Anthropic?', default=True):
        providers.append('anthropic')
    if Confirm.ask('Configure Google?', default=False):
        providers.append('google')
    console.print('\n[bold]Select MCP servers to enable:[/bold]')
    servers = []
    if Confirm.ask('Enable filesystem access?', default=True):
        servers.append('filesystem')
    if Confirm.ask('Enable web fetch?', default=True):
        servers.append('fetch')
    if Confirm.ask('Enable GitHub integration?', default=False):
        servers.append('github')
    console.print(f"\n[bold]Creating project '{project_name}'...[/bold]")
    ctx = typer.Context(init)
    init(ctx=ctx, dir=dir, template=template_name, quickstart=None, force=False, no_gitignore=False, list_templates=False)
    if 'github' in servers:
        console.print('\n[yellow]Note:[/yellow] GitHub server requires GITHUB_PERSONAL_ACCESS_TOKEN')
        console.print('Add it to mcp_agent.secrets.yaml or set as environment variable')
    console.print('\n[green bold]✨ Project setup complete![/green bold]')

def configure_logger(endpoint: Optional[str]=typer.Argument(None, help='OTEL endpoint URL for log collection'), headers: Optional[str]=typer.Option(None, '--headers', '-h', help='Additional headers in key=value,key2=value2 format'), test: bool=typer.Option(False, '--test', help='Test the connection without saving configuration')) -> None:
    """Configure OTEL endpoint and headers for log collection.

    This command allows you to configure the OpenTelemetry endpoint and headers
    that will be used for collecting logs from your deployed MCP apps.

    Examples:
        mcp-agent cloud logger configure https://otel.example.com:4318/v1/logs
        mcp-agent cloud logger configure https://otel.example.com --headers "Authorization=Bearer token,X-Custom=value"
        mcp-agent cloud logger configure --test  # Test current configuration
    """
    if not endpoint and (not test):
        print_error('Must specify endpoint or use --test')
        raise typer.Exit(1)
    config_path = _find_config_file()
    if test:
        if config_path and config_path.exists():
            config = _load_config(config_path)
            otel_config = config.get('otel', {})
            endpoint = otel_config.get('endpoint')
            headers_dict = otel_config.get('headers', {})
        else:
            console.print('[yellow]No configuration file found. Use --endpoint to set up OTEL configuration.[/yellow]')
            raise typer.Exit(1)
    else:
        headers_dict = {}
        if headers:
            try:
                for header_pair in headers.split(','):
                    key, value = header_pair.strip().split('=', 1)
                    headers_dict[key.strip()] = value.strip()
            except ValueError:
                print_error("Headers must be in format 'key=value,key2=value2'")
                raise typer.Exit(1)
    if endpoint:
        console.print(f'[blue]Testing connection to {endpoint}...[/blue]')
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(endpoint.replace('/v1/logs', '/health') if '/v1/logs' in endpoint else f'{endpoint}/health', headers=headers_dict)
                if response.status_code in [200, 404]:
                    console.print('[green]✓ Connection successful[/green]')
                else:
                    console.print(f'[yellow]⚠ Got status {response.status_code}, but endpoint is reachable[/yellow]')
        except httpx.RequestError as e:
            print_error(f'✗ Connection failed: {e}')
            if not test:
                console.print('[yellow]Configuration will be saved anyway. Check your endpoint URL and network connection.[/yellow]')
    if not test:
        if not config_path:
            config_path = Path.cwd() / 'mcp_agent.config.yaml'
        config = _load_config(config_path) if config_path.exists() else {}
        if 'otel' not in config:
            config['otel'] = {}
        config['otel']['endpoint'] = endpoint
        config['otel']['headers'] = headers_dict
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            console.print(Panel(f'[green]✓ OTEL configuration saved to {config_path}[/green]\n\nEndpoint: {endpoint}\nHeaders: {len(headers_dict)} configured' + (f' ({', '.join(headers_dict.keys())})' if headers_dict else ''), title='Configuration Saved', border_style='green'))
        except Exception as e:
            raise CLIError(f'Error saving configuration: {e}')

def create_transport(settings: LoggerSettings, event_filter: EventFilter | None=None, session_id: str | None=None) -> EventTransport:
    """Create event transport based on settings."""
    transports: List[EventTransport] = []
    transport_types = []
    if hasattr(settings, 'transports') and settings.transports:
        transport_types = settings.transports
    else:
        transport_types = [settings.type]
    for transport_type in transport_types:
        if transport_type == 'none':
            continue
        elif transport_type == 'console':
            transports.append(ConsoleTransport(event_filter=event_filter))
        elif transport_type == 'file':
            filepath = get_log_filename(settings, session_id)
            if not filepath:
                raise ValueError("File path required for file transport. Either specify 'path' or configure 'path_settings'")
            transports.append(FileTransport(filepath=filepath, event_filter=event_filter))
        elif transport_type == 'http':
            if not settings.http_endpoint:
                raise ValueError('HTTP endpoint required for HTTP transport')
            transports.append(HTTPTransport(endpoint=settings.http_endpoint, headers=settings.http_headers, batch_size=settings.batch_size, timeout=settings.http_timeout, event_filter=event_filter))
        else:
            raise ValueError(f'Unsupported transport type: {transport_type}')
    if not transports:
        return NoOpTransport(event_filter=event_filter)
    elif len(transports) == 1:
        return transports[0]
    else:
        return MultiTransport(transports)

def get_log_filename(settings: LoggerSettings, session_id: str | None=None) -> str:
    """Generate a log filename based on the configuration.

    Args:
        settings: Logger settings containing path configuration
        session_id: Optional session ID to use in the filename

    Returns:
        String path for the log file
    """
    if settings.path and (not settings.path_settings):
        return settings.path
    if settings.path_settings:
        path_pattern = settings.path_settings.path_pattern
        unique_id_type = settings.path_settings.unique_id
        if unique_id_type == 'session_id':
            unique_id = session_id if session_id else str(uuid.uuid4())
        else:
            now = datetime.datetime.now()
            time_format = settings.path_settings.timestamp_format
            unique_id = now.strftime(time_format)
        return path_pattern.replace('{unique_id}', unique_id)
    raise ValueError('No path settings provided')

def register_asyncio_decorators(decorator_registry: DecoratorRegistry):
    """Registers default asyncio decorators."""
    executor_name = 'asyncio'
    decorator_registry.register_workflow_defn_decorator(executor_name, default_workflow_defn)
    decorator_registry.register_workflow_run_decorator(executor_name, default_workflow_run)
    decorator_registry.register_workflow_signal_decorator(executor_name, default_workflow_signal)

def register_temporal_decorators(decorator_registry: DecoratorRegistry):
    """Registers Temporal decorators if Temporal SDK is available."""
    try:
        import temporalio.workflow as temporal_workflow
        import temporalio.activity as temporal_activity
        TEMPORAL_AVAILABLE = True
    except ImportError:
        TEMPORAL_AVAILABLE = False
    if not TEMPORAL_AVAILABLE:
        return
    executor_name = 'temporal'
    decorator_registry.register_workflow_defn_decorator(executor_name, temporal_workflow.defn)
    decorator_registry.register_workflow_run_decorator(executor_name, temporal_workflow.run)
    decorator_registry.register_workflow_task_decorator(executor_name, temporal_activity.defn)
    decorator_registry.register_workflow_signal_decorator(executor_name, temporal_workflow.signal)

