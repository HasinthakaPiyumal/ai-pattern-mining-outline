# Cluster 16

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, verbose: bool=typer.Option(False, '--verbose', '-v', help='Enable verbose output'), color: bool=typer.Option(True, '--color/--no-color', help='Enable/disable color output'), version: bool=typer.Option(False, '--version', help='Show version and exit'), format: str=typer.Option('text', '--format', help='Output format for list/describe commands', show_default=True, case_sensitive=False)) -> None:
    """mcp-agent command line interface."""
    if verbose:
        LOG_VERBOSE.set(True)
    ctx.obj = {'color': color, 'format': format.lower()}
    if not color:
        console.no_color = True
        err_console.no_color = True
    if version:
        _print_version()
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        console.print('mcp-agent - Model Context Protocol agent CLI\n')
        console.print("Run 'mcp-agent --help' to see all commands.")

def _print_version() -> None:
    try:
        import importlib.metadata as _im
        ver = _im.version('mcp-agent')
    except Exception:
        ver = 'unknown'
    console.print(f'mcp-agent {ver}')

