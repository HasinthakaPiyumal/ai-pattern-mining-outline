# Cluster 41

def print_error(message: str, *args: Any, log: bool=True, console_output: bool=True, **kwargs: Any) -> None:
    """Print an error message."""
    if console_output:
        label = _create_label('', 'error')
        console.print(f'{label}{message}', *args, **kwargs)
    if log:
        logger.error(message, exc_info=True)

def maybe_warn_newer_version() -> None:
    """Best-effort version check kicked off exactly once per process."""
    if os.environ.get('MCP_AGENT_DISABLE_VERSION_CHECK', '').lower() in {'1', 'true', 'yes'}:
        return
    if os.environ.get('MCP_AGENT_VERSION_CHECKED'):
        return
    with _version_check_lock:
        global _version_check_started, _version_check_message
        if _version_check_started:
            return
        _version_check_started = True
        _version_check_message = None
        _version_check_event.clear()
        try:
            _spawn_version_check_thread()
        except Exception:
            _version_check_started = False
            return
        os.environ['MCP_AGENT_VERSION_CHECKED'] = '1'
        atexit.register(_flush_version_check_message)

def _spawn_version_check_thread() -> None:
    thread = threading.Thread(target=_run_version_check, name='mcp-agent-version-check', daemon=True)
    thread.start()

class HelpfulTyperGroup(TyperGroup):
    """Typer group that shows help before usage errors for better UX."""

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            click.echo(ctx.get_help())
            console = Console(stderr=True)
            error_panel = Panel(str(e), title='Error', title_align='left', border_style='red', expand=True)
            console.print(error_panel)
            ctx.exit(2)

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except CLIError as e:
            logging.error(f'CLI error: {str(e)}')
            print_error(str(e))
            ctx.exit(e.exit_code)

def retrieve_secrets_from_config(config_path: str, required_secrets: List[str]) -> Dict[str, str]:
    """Retrieve dot-notated user secrets from a YAML configuration file.

    This function reads a YAML configuration file and extracts user secrets
    based on the provided required secret keys.

    Args:
        config_path: Path to the configuration file
        required_secrets: List of required user secret keys to retrieve

    Returns:
        Dict with secret keys and their corresponding values
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = load_yaml_with_secrets(f.read())
    except Exception as e:
        print_error(f'Failed to read or parse config file: {str(e)}')
        raise
    secrets = {}
    for secret_key in required_secrets:
        value = get_nested_key_value(config, secret_key)
        if not SECRET_ID_PATTERN.match(value):
            raise ValueError(f"Secret '{secret_key}' in config does not match expected secret ID pattern")
        secrets[secret_key] = value
    return secrets

def get_nested_key_value(config: dict, dotted_key: str) -> Any:
    parts = dotted_key.split('.')
    value = config
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Required secret '{dotted_key}' not found in config.")
        value = value[part]
    return value

def run() -> None:
    """Run the CLI application."""
    try:
        try:
            maybe_warn_newer_version()
        except Exception:
            pass
        app()
    except Exception as e:
        logging.exception('Unhandled exception in CLI')
        print_error(f'An unexpected error occurred: {str(e)}')
        raise typer.Exit(1) from e

@handle_server_api_errors
def list_servers(limit: Optional[int]=typer.Option(None, '--limit', help='Maximum number of results to return'), filter: Optional[str]=typer.Option(None, '--filter', help='Filter by name, description, or status (case-insensitive)'), sort_by: Optional[str]=typer.Option(None, '--sort-by', help='Sort by field: name, created, status (prefix with - for reverse)'), format: Optional[str]=typer.Option('text', '--format', help='Output format (text|json|yaml)')) -> None:
    """List MCP Servers with optional filtering and sorting.

    Examples:

        mcp-agent cloud servers list --filter api

        mcp-agent cloud servers list --sort-by -created

        mcp-agent cloud servers list --filter active --sort-by name

        mcp-agent cloud servers list --filter production --format json
    """
    validate_output_format(format)
    client = setup_authenticated_client()
    max_results = limit or 100

    async def parallel_requests():
        return await asyncio.gather(client.list_apps(max_results=max_results), client.list_app_configurations(max_results=max_results))
    list_apps_res, list_app_configs_res = run_async(parallel_requests())
    filtered_deployed = _apply_filter(list_apps_res.apps, filter) if filter else list_apps_res.apps
    filtered_configured = _apply_filter(list_app_configs_res.appConfigurations, filter) if filter else list_app_configs_res.appConfigurations
    sorted_deployed = _apply_sort(filtered_deployed, sort_by) if sort_by else filtered_deployed
    sorted_configured = _apply_sort(filtered_configured, sort_by) if sort_by else filtered_configured
    if format == 'json':
        _print_servers_json(sorted_deployed, sorted_configured)
    elif format == 'yaml':
        _print_servers_yaml(sorted_deployed, sorted_configured)
    else:
        _print_servers_text(sorted_deployed, sorted_configured, filter, sort_by)

def validate_output_format(format: str) -> None:
    """Validate output format parameter.

    Args:
        format: Output format to validate

    Raises:
        CLIError: If format is invalid
    """
    valid_formats = ['text', 'json', 'yaml']
    if format not in valid_formats:
        raise CLIError(f"Invalid format '{format}'. Valid options are: {', '.join(valid_formats)}", retriable=False)

def setup_authenticated_client() -> MCPAppClient:
    """Setup authenticated MCP App client.

    Returns:
        Configured MCPAppClient instance

    Raises:
        CLIError: If authentication fails
    """
    effective_api_key = settings.API_KEY or load_api_key_credentials()
    if not effective_api_key:
        raise CLIError("Must be authenticated. Set MCP_API_KEY or run 'mcp-agent login'.", retriable=False)
    return MCPAppClient(api_url=settings.API_BASE_URL, api_key=effective_api_key)

@handle_server_api_errors
def describe_server(id_or_url: str=typer.Argument(..., help='App ID, server URL, or app name to describe'), format: Optional[str]=typer.Option('text', '--format', help='Output format (text|json|yaml)')) -> None:
    """Describe a specific MCP Server."""
    validate_output_format(format)
    client = setup_authenticated_client()
    server = resolve_server(client, id_or_url)
    print_server_description(server, format)

def print_workflows(workflows: list[Workflow]) -> None:
    """Print workflows in text format."""
    if not workflows:
        console.print(Panel('[yellow]No workflows found[/yellow]', title='Workflows', border_style='blue'))
        return
    panels = []
    for workflow in workflows:
        header = Text(workflow.name, style='bold cyan')
        desc = textwrap.dedent(workflow.description or 'No description available').strip()
        body_parts: list = [Text(desc, style='white')]
        capabilities = getattr(workflow, 'capabilities', [])
        cap_text = Text('\nCapabilities:\n', style='bold green')
        cap_text.append_text(Text(', '.join(capabilities) or 'None', style='white'))
        body_parts.append(cap_text)
        tool_endpoints = getattr(workflow, 'tool_endpoints', [])
        endpoints_text = Text('\nTool Endpoints:\n', style='bold green')
        endpoints_text.append_text(Text('\n'.join(tool_endpoints) or 'None', style='white'))
        body_parts.append(endpoints_text)
        if workflow.run_parameters:
            run_params = clean_run_parameters(workflow.run_parameters)
            properties = run_params.get('properties', {})
            if len(properties) > 0:
                schema_str = json.dumps(run_params, indent=2)
                schema_syntax = Syntax(schema_str, 'json', theme='monokai', word_wrap=True)
                body_parts.append(Text('\nRun Parameters:', style='bold magenta'))
                body_parts.append(schema_syntax)
        body = Group(*body_parts)
        panels.append(Panel(body, title=header, border_style='green', expand=False))
    console.print(Panel(Group(*panels), title='Workflows', border_style='blue'))

def clean_run_parameters(schema: dict) -> dict:
    """Clean the run parameters schema by removing 'self' references."""
    schema = schema.copy()
    if 'properties' in schema and 'self' in schema['properties']:
        schema['properties'].pop('self')
    if 'required' in schema and 'self' in schema['required']:
        schema['required'] = [r for r in schema['required'] if r != 'self']
    return schema

def print_workflow_runs(runs: list[WorkflowRun], status_filter: Optional[str]=None) -> None:
    """Print workflows in text format."""
    console.print(f'\n[bold blue] Workflow Runs ({len(runs)})[/bold blue]')
    if not runs:
        print_info('No workflow runs found.')
        return
    for i, workflow in enumerate(runs):
        if i > 0:
            console.print()
        workflow_id = getattr(workflow.temporal, 'workflow_id', 'Unknown') if workflow.temporal else 'Unknown'
        name = getattr(workflow, 'name', 'Unknown')
        execution_status = getattr(workflow, 'status', 'Unknown')
        run_id = getattr(workflow, 'id', 'Unknown')
        started_at = getattr(workflow.temporal, 'start_time', 'Unknown') if workflow.temporal else 'Unknown'
        status_display = format_workflow_status(execution_status)
        if started_at and started_at != 'Unknown':
            if hasattr(started_at, 'strftime'):
                started_display = started_at.strftime('%Y-%m-%d %H:%M:%S')
            else:
                try:
                    if isinstance(started_at, (int, float)):
                        dt = datetime.fromtimestamp(started_at)
                    else:
                        dt = datetime.fromisoformat(str(started_at).replace('Z', '+00:00'))
                    started_display = dt.strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    started_display = str(started_at)
        else:
            started_display = 'Unknown'
        console.print(f'[bold cyan]{name or 'Unnamed'}[/bold cyan] {status_display}')
        console.print(f'  Workflow ID: {workflow_id}')
        console.print(f'  Run ID: {run_id}')
        console.print(f'  Started: {started_display}')
    if status_filter:
        console.print(f'\n[dim]Filtered by status: {status_filter}[/dim]')

def format_workflow_status(status: Optional[str]=None) -> str:
    """Format the execution status text."""
    if not status:
        return '❓ Unknown'
    status_lower = str(status).lower()
    if 'running' in status_lower:
        return '[green]🔄 Running[/green]'
    elif 'failed' in status_lower or 'error' in status_lower:
        return '[red]❌ Failed[/red]'
    elif 'timeout' in status_lower or 'timed_out' in status_lower:
        return '[red]⌛ Timed Out[/red]'
    elif 'cancel' in status_lower:
        return '[yellow]🚫 Cancelled[/yellow]'
    elif 'terminat' in status_lower:
        return '[red]🛑 Terminated[/red]'
    elif 'complet' in status_lower:
        return '[green]✅ Completed[/green]'
    elif 'continued' in status_lower:
        return '[blue]🔁 Continued as New[/blue]'
    else:
        return f'❓ {status}'

@handle_server_api_errors
def list_workflows(server_id_or_url_or_name: str=typer.Argument(..., help='App ID, server URL, or app name to list workflows for'), format: Optional[str]=typer.Option('text', '--format', help='Output format (text|json|yaml)')) -> None:
    """List available workflow definitions for an MCP Server.

    This command lists the workflow definitions that a server provides,
    showing what workflows can be executed.

    Examples:

        mcp-agent cloud workflows list app_abc123

        mcp-agent cloud workflows list https://server.example.com --format json
    """
    validate_output_format(format)
    run_async(_list_workflows_async(server_id_or_url_or_name, format))

def print_workflow_status(workflow_status: WorkflowRun, format: str='text') -> None:
    """Print workflow status information in requested format"""
    if format == 'json':
        print(json.dumps(workflow_status.model_dump(), indent=2))
    elif format == 'yaml':
        print(yaml.dump(workflow_status.model_dump(), default_flow_style=False))
    else:
        name = getattr(workflow_status, 'name', 'Unknown')
        workflow_id = getattr(workflow_status.temporal, 'workflow_id', 'Unknown') if workflow_status.temporal else 'Unknown'
        run_id = getattr(workflow_status, 'id', 'Unknown')
        status = getattr(workflow_status, 'status', 'Unknown')
        created_at = getattr(workflow_status.temporal, 'start_time', None) if workflow_status.temporal else None
        if created_at is not None:
            try:
                created_dt = datetime.fromtimestamp(created_at)
                created_at = created_dt.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                created_at = str(created_at)
        else:
            created_at = 'Unknown'
        console.print('\n[bold blue]🔍 Workflow Details[/bold blue]')
        console.print()
        console.print(f'[bold cyan]{name}[/bold cyan] {format_workflow_status(status)}')
        console.print(f'  Workflow ID: {workflow_id}')
        console.print(f'  Run ID: {run_id}')
        console.print(f'  Created: {created_at}')
        if workflow_status.result:
            console.print('\n[bold green]📄 Result[/bold green]')
            console.print(f'  Kind: {getattr(workflow_status.result, 'kind', 'Unknown')}')
            result_value = getattr(workflow_status.result, 'value', None)
            if result_value:
                if len(str(result_value)) > 10000:
                    truncated_value = str(result_value)[:10000] + '...'
                    console.print(f'  Value: {truncated_value}')
                else:
                    console.print(f'  Value: {result_value}')
            start_time = getattr(workflow_status.result, 'start_time', None)
            end_time = getattr(workflow_status.result, 'end_time', None)
            if start_time:
                start_dt = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
                console.print(f'  Started: {start_dt}')
            if end_time:
                end_dt = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
                console.print(f'  Ended: {end_dt}')
        if workflow_status.error:
            console.print('\n[bold red]❌ Error[/bold red]')
            console.print(f'  {workflow_status.error}')
        if workflow_status.state and workflow_status.state.error and (workflow_status.state.error != workflow_status.error):
            console.print('\n[bold red]⚠️  State Error[/bold red]')
            if isinstance(workflow_status.state.error, dict):
                error_type = workflow_status.state.error.get('type', 'Unknown')
                error_message = workflow_status.state.error.get('message', 'Unknown error')
                console.print(f'  Type: {error_type}')
                console.print(f'  Message: {error_message}')
            else:
                console.print(f'  {workflow_status.state.error}')

def list_workflow_runs(server_id_or_url: str=typer.Argument(..., help='App ID, server URL, or app name to list workflow runs for'), limit: Optional[int]=typer.Option(None, '--limit', help='Maximum number of results to return'), status: Optional[str]=typer.Option(None, '--status', help='Filter by status: running|failed|timed_out|timeout|canceled|terminated|completed|continued', callback=lambda value: _get_status_filter(value) if value else None), format: Optional[str]=typer.Option('text', '--format', help='Output format (text|json|yaml)')) -> None:
    """List workflow runs for an MCP Server.

    Examples:

        mcp-agent cloud workflows runs app_abc123

        mcp-agent cloud workflows runs https://server.example.com --status running

        mcp-agent cloud workflows runs apcnf_xyz789 --limit 10 --format json
    """
    validate_output_format(format)
    run_async(_list_workflow_runs_async(server_id_or_url, limit, status, format))

