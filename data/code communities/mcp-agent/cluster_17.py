# Cluster 17

def run() -> None:
    """Display a spinner only during terminal bootstrap , then hand off to main.run()."""
    console = Console(stderr=True)
    if console.is_terminal:
        with console.status('[dim]Loading mcp-agent CLI...[/dim]', spinner='dots'):
            from mcp_agent.cli.main import run as main_run
    else:
        from mcp_agent.cli.main import run as main_run
    main_run()

def generate_step():
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run())
        return result
    except Exception as e:
        logger.exception('Error during script generation', exc_info=e)
        return ''
    finally:
        loop.close()
        asyncio.set_event_loop(None)

def main():

    async def run():
        try:
            await app.initialize()
            with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'), console=console) as progress:
                task = progress.add_task(description='Loading model selector...', total=None)
                model_selector = ModelSelector()
                progress.update(task, description='Model selector loaded!')
            await interactive_model_selection(model_selector)
        finally:
            await app.cleanup()
    typer.run(lambda: asyncio.run(run()))

