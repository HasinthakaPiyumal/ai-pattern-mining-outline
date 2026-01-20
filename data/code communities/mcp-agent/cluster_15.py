# Cluster 15

def _preload_workflow_task_modules(app: 'MCPApp') -> None:
    """
    Import modules that define @workflow_task activities so they register with the app
    before we hand the activity list to the Temporal worker.
    """
    module_names = set(DEFAULT_TEMPORAL_WORKFLOW_TASK_MODULES)
    try:
        global_modules = getattr(getattr(app.context, 'config', None), 'workflow_task_modules', None)
        if global_modules:
            module_names.update((module for module in global_modules if module))
    except Exception:
        pass
    try:
        temporal_settings = getattr(getattr(app.context, 'config', None), 'temporal', None)
        if temporal_settings and getattr(temporal_settings, 'workflow_task_modules', None):
            module_names.update((module for module in temporal_settings.workflow_task_modules if module))
    except Exception:
        pass
    for module_name in sorted(module_names):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_dep = exc.name or module_name
            extra_hint = MODULE_OPTIONAL_EXTRAS.get(module_name)
            logger.warning('Workflow task module import skipped; install optional dependency', data={'module': module_name, 'missing_dependency': missing_dep, 'install_hint': f'pip install "mcp-agent[{extra_hint}]"' if extra_hint else 'Install the matching optional extras for your provider'})
        except Exception as exc:
            logger.warning('Failed to import workflow task module', data={'module': module_name, 'error': str(exc)})

