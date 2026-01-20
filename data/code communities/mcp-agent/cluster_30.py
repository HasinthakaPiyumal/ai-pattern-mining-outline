# Cluster 30

def print_verbose(message: str, *args: Any, log: bool=True, console_output: bool=True, **kwargs: Any):
    """
    Print debug-like verbose content as info only if configured for verbose logging,
    i.e. replaces "if verbose then print_info"
    """
    if LOG_VERBOSE.get():
        print_info(message, *args, log=log, console_output=console_output, **kwargs)

def get_git_metadata(project_dir: Path) -> Optional[GitMetadata]:
    """Return GitMetadata for the repo containing project_dir, if any.

    Returns None if git is unavailable or project_dir is not inside a repo.
    """
    try:
        inside = _run_git(['rev-parse', '--is-inside-work-tree'], project_dir)
        if inside is None or inside != 'true':
            return None
        commit_sha = _run_git(['rev-parse', 'HEAD'], project_dir)
        if not commit_sha:
            return None
        short_sha = _run_git(['rev-parse', '--short', 'HEAD'], project_dir) or commit_sha[:7]
        branch = _run_git(['rev-parse', '--abbrev-ref', 'HEAD'], project_dir)
        status = _run_git(['status', '--porcelain'], project_dir)
        dirty = bool(status)
        tag = _run_git(['describe', '--tags', '--exact-match'], project_dir)
        commit_message = _run_git(['log', '-1', '--pretty=%s'], project_dir)
        return GitMetadata(commit_sha=commit_sha, short_sha=short_sha, branch=branch, dirty=dirty, tag=tag, commit_message=commit_message)
    except Exception:
        return None

def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    """Run a git command and return stdout, suppressing all stderr noise.

    Returns None on any error or non-zero exit to avoid leaking git messages
    like "fatal: no tag exactly matches" to the console.
    """
    try:
        proc = subprocess.run(['git', *args], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        if proc.returncode != 0:
            return None
        return proc.stdout.decode('utf-8', errors='replace').strip()
    except Exception:
        return None

def create_git_tag(project_dir: Path, tag_name: str, message: str) -> bool:
    """Create an annotated git tag at HEAD. Returns True on success.

    Does nothing and returns False if not a repo or git fails.
    """
    inside = _run_git(['rev-parse', '--is-inside-work-tree'], project_dir)
    if inside is None or inside != 'true':
        return False
    try:
        subprocess.check_call(['git', 'tag', '-a', tag_name, '-m', message], cwd=str(project_dir))
        return True
    except Exception:
        return False

def wrangler_deploy(app_id: str, api_key: str, project_dir: Path, ignore_file: Path | None=None) -> None:
    """Bundle the MCP Agent using Wrangler.

    A thin wrapper around the Wrangler CLI to bundle the MCP Agent application code
    and upload it our internal cf storage.

    Some key details here:
    - We copy the user's project to a temporary directory and perform all operations there
    - Secrets file must be excluded from the bundle
    - We must add a temporary `wrangler.toml` to the project directory to set python_workers
      compatibility flag (CLI arg is not sufficient).
    - Python workers with a `requirements.txt` file cannot be published by Wrangler, so we must
      rename any `requirements.txt` file to `requirements.txt.mcpac.py` before bundling
    - Non-python files (e.g. `uv.lock`, `poetry.lock`, `pyproject.toml`) would be excluded by default
    due to no py extension, so they are renamed with a `.mcpac.py` extension.
    - We exclude .venv directories from the copy to avoid bundling issues.

    Args:
        app_id (str): The application ID.
        api_key (str): User MCP Agent Cloud API key.
        project_dir (Path): The directory of the project to deploy.
        ignore_file (Path | None): Optional path to a gitignore-style file for excluding files from the bundle.
    """
    env = os.environ.copy()
    env_updates = {'CLOUDFLARE_ACCOUNT_ID': CLOUDFLARE_ACCOUNT_ID, 'CLOUDFLARE_API_TOKEN': api_key, 'CLOUDFLARE_EMAIL': CLOUDFLARE_EMAIL, 'WRANGLER_AUTH_DOMAIN': deployment_settings.wrangler_auth_domain, 'WRANGLER_AUTH_URL': deployment_settings.wrangler_auth_url, 'WRANGLER_SEND_METRICS': str(WRANGLER_SEND_METRICS).lower(), 'CLOUDFLARE_API_BASE_URL': deployment_settings.cloudflare_api_base_url, 'HOME': os.path.expanduser(settings.DEPLOYMENT_CACHE_DIR), 'XDG_HOME_DIR': os.path.expanduser(settings.DEPLOYMENT_CACHE_DIR)}
    if os.name == 'nt':
        npm_prefix = Path(os.path.expanduser(settings.DEPLOYMENT_CACHE_DIR)) / 'npm-global'
        npm_prefix.mkdir(parents=True, exist_ok=True)
        env_updates['npm_config_prefix'] = str(npm_prefix)
    if os.environ.get('__MCP_DISABLE_TLS_VALIDATION', '').lower() in ('1', 'true', 'yes'):
        if deployment_settings.DEPLOYMENTS_UPLOAD_API_BASE_URL == DEFAULT_DEPLOYMENTS_UPLOAD_API_BASE_URL:
            print_error(f'Cannot disable TLS validation when using {DEFAULT_DEPLOYMENTS_UPLOAD_API_BASE_URL}. Set MCP_DEPLOYMENTS_UPLOAD_API_BASE_URL to a custom endpoint.')
            raise ValueError(f'TLS validation cannot be disabled with {DEFAULT_DEPLOYMENTS_UPLOAD_API_BASE_URL}')
        env_updates['NODE_TLS_REJECT_UNAUTHORIZED'] = '0'
        print_warning('TLS certificate validation disabled (__MCP_DISABLE_TLS_VALIDATION is set).')
        if settings.VERBOSE:
            print_info(f'Deployment endpoint: {deployment_settings.DEPLOYMENTS_UPLOAD_API_BASE_URL}')
    env.update(env_updates)
    validate_project(project_dir)
    main_py = 'main.py'
    with tempfile.TemporaryDirectory(prefix='mcp-deploy-') as temp_dir_str:
        temp_project_dir = Path(temp_dir_str) / 'project'
        ignore_spec = create_pathspec_from_gitignore(ignore_file) if ignore_file else None
        if ignore_file:
            if ignore_spec is None:
                print_warning(f"Ignore file '{ignore_file}' not found; applying default excludes only")
            else:
                print_info(f'Using ignore patterns from {ignore_file}')
        else:
            print_verbose('No ignore file provided; applying default excludes only')

        def ignore_patterns(path_str, names):
            ignored = set()
            for name in names:
                if name.startswith('.') and name not in {'.env'} or name in {'logs', '__pycache__', 'node_modules', 'venv', MCP_SECRETS_FILENAME}:
                    ignored.add(name)
            spec_ignored = should_ignore_by_gitignore(path_str, names, project_dir, ignore_spec)
            ignored.update(spec_ignored)
            return ignored
        shutil.copytree(project_dir, temp_project_dir, ignore=ignore_patterns)
        requirements_path = temp_project_dir / 'requirements.txt'
        if _needs_requirements_modification(requirements_path):
            _modify_requirements_txt(requirements_path)
        for root, _dirs, files in os.walk(temp_project_dir):
            for filename in files:
                file_path = Path(root) / filename
                if filename.startswith('.') or filename.endswith(('.bak', '.tmp')):
                    continue
                if filename == 'wrangler.toml':
                    continue
                if filename.endswith('.py'):
                    continue
                py_path = file_path.with_suffix(file_path.suffix + '.mcpac.py')
                file_path.rename(py_path)
        bundled_original_files: list[str] = []
        internal_bundle_files = {'wrangler.toml', 'mcp_deploy_breadcrumb.py'}
        for root, _dirs, files in os.walk(temp_project_dir):
            for filename in files:
                rel = Path(root).relative_to(temp_project_dir) / filename
                if filename in internal_bundle_files:
                    continue
                if filename.endswith('.mcpac.py'):
                    orig_rel = str(rel)[:-len('.mcpac.py')]
                    bundled_original_files.append(orig_rel)
                else:
                    bundled_original_files.append(str(rel))
        bundled_original_files.sort()
        if bundled_original_files:
            print_verbose('\n'.join([f'Bundling {len(bundled_original_files)} project file(s):'] + [f' - {p}' for p in bundled_original_files]))
        git_meta = get_git_metadata(project_dir)
        deploy_source = 'git' if git_meta else 'workspace'
        meta_vars = {'MCP_DEPLOY_SOURCE': deploy_source, 'MCP_DEPLOY_TIME_UTC': utc_iso_now()}
        if git_meta:
            meta_vars.update({'MCP_DEPLOY_GIT_COMMIT': git_meta.commit_sha, 'MCP_DEPLOY_GIT_SHORT': git_meta.short_sha, 'MCP_DEPLOY_GIT_BRANCH': git_meta.branch or '', 'MCP_DEPLOY_GIT_DIRTY': 'true' if git_meta.dirty else 'false'})
            dirty_mark = '*' if git_meta.dirty else ''
            print_info(f'Deploying from git commit {git_meta.short_sha}{dirty_mark} on branch {git_meta.branch or '?'}')
        else:
            bundle_hash = compute_directory_fingerprint(temp_project_dir, ignore_names={'.git', 'logs', '__pycache__', 'node_modules', 'venv', MCP_SECRETS_FILENAME})
            meta_vars.update({'MCP_DEPLOY_WORKSPACE_HASH': bundle_hash})
            print_verbose(f'Deploying from non-git workspace (hash {bundle_hash[:12]}…)')
        breadcrumb = {'version': 1, 'app_id': app_id, 'deploy_time_utc': meta_vars['MCP_DEPLOY_TIME_UTC'], 'source': meta_vars['MCP_DEPLOY_SOURCE']}
        if git_meta:
            breadcrumb.update({'git': {'commit': git_meta.commit_sha, 'short': git_meta.short_sha, 'branch': git_meta.branch, 'dirty': git_meta.dirty, 'tag': git_meta.tag, 'message': git_meta.commit_message}})
        else:
            breadcrumb.update({'workspace_fingerprint': meta_vars['MCP_DEPLOY_WORKSPACE_HASH']})
        breadcrumb_py = textwrap.dedent('\n            # Auto-generated by mcp-agent deploy. Do not edit.\n            # Contains deployment metadata for traceability.\n            import json as _json\n            BREADCRUMB = %s\n            BREADCRUMB_JSON = _json.dumps(BREADCRUMB, separators=(",", ":"))\n            __all__ = ["BREADCRUMB", "BREADCRUMB_JSON"]\n            ').strip() % json.dumps(breadcrumb, indent=2)
        (temp_project_dir / 'mcp_deploy_breadcrumb.py').write_text(breadcrumb_py)
        meta_json = json.dumps(meta_vars, separators=(',', ':'))
        vars_lines = ['[vars]'] + [f'{k} = "{v}"' for k, v in meta_vars.items()]
        vars_lines.append(f'MCP_DEPLOY_META = """{meta_json}"""')
        wrangler_toml_content = textwrap.dedent(f'\n            name = "{app_id}"\n            main = "{main_py}"\n            compatibility_flags = ["python_workers"]\n            compatibility_date = "2025-06-26"\n\n            {os.linesep.join(vars_lines)}\n        ').strip()
        wrangler_toml_path = temp_project_dir / 'wrangler.toml'
        wrangler_toml_path.write_text(wrangler_toml_content)
        spinner_column = SpinnerColumn(spinner_name='aesthetic')
        with Progress('', spinner_column, TextColumn(' [progress.description]{task.description}')) as progress:
            task = progress.add_task('Bundling MCP Agent...', total=None)
            try:
                cmd = ['npx', '--yes', 'wrangler@4.22.0', 'deploy', main_py, '--name', app_id, '--no-bundle']
                subprocess.run(cmd, check=True, env=env, cwd=str(temp_project_dir), capture_output=True, text=True, shell=os.name == 'nt', encoding='utf-8', errors='replace')
                spinner_column.spinner.frames = spinner_column.spinner.frames[-2:-1]
                progress.update(task, description='Bundled successfully')
            except subprocess.CalledProcessError as e:
                progress.update(task, description='❌ Bundling failed')
                _handle_wrangler_error(e)
                raise

def _needs_requirements_modification(requirements_path: Path) -> bool:
    """Check if requirements.txt contains relative mcp-agent imports that need modification."""
    if not requirements_path.exists():
        return False
    content = requirements_path.read_text()
    return bool(RELATIVE_MCP_AGENT_PATTERN.search(content))

def _modify_requirements_txt(requirements_path: Path) -> None:
    """Modify requirements.txt in place to replace relative mcp-agent imports with absolute ones."""
    content = requirements_path.read_text()
    modified_content = RELATIVE_MCP_AGENT_PATTERN.sub('mcp-agent', content)
    requirements_path.write_text(modified_content)

def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def compute_directory_fingerprint(root: Path, *, ignore_names: set[str] | None=None) -> str:
    """Compute a cheap, stable SHA256 over file metadata under root.

    This avoids reading file contents. The hash includes the relative path,
    file size and modification time for each included file. Hidden files/dirs
    and any names in `ignore_names` are skipped, as are symlinks.
    """
    if ignore_names is None:
        ignore_names = set()
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_names and (not d.startswith('.'))]
        for fname in sorted(filenames):
            if fname in ignore_names or (fname.startswith('.') and fname != '.env'):
                continue
            fpath = Path(dirpath) / fname
            if fpath.is_symlink():
                continue
            rel = fpath.relative_to(root).as_posix()
            try:
                st = fpath.stat()
                size = st.st_size
                mtime = int(st.st_mtime)
            except Exception:
                size = -1
                mtime = 0
            h.update(rel.encode('utf-8'))
            h.update(b'\x00')
            h.update(str(size).encode('utf-8'))
            h.update(b'\x00')
            h.update(str(mtime).encode('utf-8'))
            h.update(b'\n')
    return h.hexdigest()

def deploy_config(ctx: typer.Context, app_name: Optional[str]=typer.Argument(None, help='Name of the MCP App to deploy.'), app_description: Optional[str]=typer.Option(None, '--app-description', '-d', help='Description of the MCP App being deployed.'), config_dir: Optional[Path]=typer.Option(None, '--config-dir', '-c', help='Path to the directory containing the app config and app files. If relative, it is resolved against --working-dir.', readable=True, dir_okay=True, file_okay=False, resolve_path=False), working_dir: Path=typer.Option(Path('.'), '--working-dir', '-w', help='Working directory to resolve config and bundle files from. Defaults to the current directory.', exists=True, readable=True, dir_okay=True, file_okay=False, resolve_path=True), non_interactive: bool=typer.Option(False, '--non-interactive', help='Use existing secrets and update existing app where applicable, without prompting.'), unauthenticated_access: Optional[bool]=typer.Option(None, '--no-auth/--auth', help='Allow unauthenticated access to the deployed server. Defaults to preserving the existing setting.'), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.', envvar=ENV_API_BASE_URL), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.', envvar=ENV_API_KEY), git_tag: bool=typer.Option(False, '--git-tag/--no-git-tag', help='Create a local git tag for this deploy (if in a git repo)', envvar='MCP_DEPLOY_GIT_TAG'), retry_count: int=typer.Option(3, '--retry-count', help='Number of retries on deployment failure.', min=1, max=10), ignore_file: Optional[Path]=typer.Option(None, '--ignore-file', help='Path to ignore file (gitignore syntax). Precedence: 1) --ignore-file <path>, 2) .mcpacignore in --config-dir, 3) .mcpacignore in working directory.', exists=False, readable=True, dir_okay=False, file_okay=True, resolve_path=True), verbose: bool=typer.Option(False, '--verbose', '-v', help='Enable verbose output for this command')) -> Optional[str]:
    """Deploy an mcp-agent using the specified configuration.

    An MCP App is deployed from bundling the code at the specified config directory.
    This directory must contain an 'mcp_agent.config.yaml' at its root. The process will look for an existing
    'mcp_agent.deployed.secrets.yaml' in the config directory or create one by processing the 'mcp_agent.secrets.yaml'
    in the config directory (if it exists) and prompting for desired secrets usage.
    The 'deployed' secrets file is processed to replace raw secrets with secret handles before deployment and
    that file is included in the deployment bundle in place of the original secrets file.

    Args:
        ctx: Typer context.
        app_name: Name of the MCP App to deploy
        app_description: Description of the MCP App being deployed
        config_dir: Path to the directory containing the app configuration files
        working_dir: Working directory from which to resolve config and bundle files.
        non_interactive: Never prompt for reusing or updating secrets or existing apps; reuse existing where possible
        unauthenticated_access: Whether to allow unauthenticated access to the deployed server. Defaults to preserving
        the existing setting.
        api_url: API base URL
        api_key: API key for authentication
        git_tag: Create a local git tag for this deploy (if in a git repo)
        retry_count: Number of retries on deployment failure
        ignore_file: Path to ignore file (gitignore syntax)
        verbose: Whether to enable verbose output

    Returns:
        Newly-deployed MCP App ID, or None if declined without creating
    """
    if verbose:
        LOG_VERBOSE.set(True)
    try:
        if config_dir is None:
            resolved_config_dir = working_dir
        elif config_dir.is_absolute():
            resolved_config_dir = config_dir
        else:
            resolved_config_dir = working_dir / config_dir
        if not resolved_config_dir.exists() or not resolved_config_dir.is_dir():
            raise CLIError(f"Configuration directory '{resolved_config_dir}' does not exist or is not a directory.", retriable=False)
        config_dir = resolved_config_dir
        config_file, secrets_file, deployed_secrets_file = get_config_files(config_dir)
        default_app_name, default_app_description = get_app_defaults_from_config(config_file)
        if app_name is None:
            if default_app_name:
                print_verbose(f"Using app name from config.yaml: '{default_app_name}'")
                app_name = default_app_name
            else:
                app_name = 'default'
                print_verbose("Using app name: 'default'")
        effective_api_url = api_url or settings.API_BASE_URL
        effective_api_key = api_key or settings.API_KEY or load_api_key_credentials()
        if not effective_api_url:
            raise CLIError('MCP_API_BASE_URL environment variable or --api-url option must be set.', retriable=False)
        if not effective_api_key:
            raise CLIError('You need to be logged in to deploy.\n\nTo continue, do one of the following:\n  • Run: mcp-agent login\n  • Or set the MCP_API_KEY environment variable\n  • Or use the --api-key flag with your key', retriable=False)
        print_verbose(f'Using API at {effective_api_url}')
        mcp_app_client = MCPAppClient(api_url=effective_api_url, api_key=effective_api_key)
        print_verbose(f"Checking for existing app ID for '{app_name}'...")
        configurable_fields = (('description', 'Description'), ('unauthenticated_access', 'Allow unauthenticated access'))
        existing_properties: dict[str, Optional[str | bool]] = {}
        update_payload: dict[str, Optional[str | bool]] = {'description': app_description, 'unauthenticated_access': unauthenticated_access}
        create_new_app = False
        app_id = None
        try:
            existing_app: Optional[MCPApp] = run_async(mcp_app_client.get_app_by_name(app_name))
            if existing_app:
                app_id = existing_app.appId
                print_verbose(f"Found existing app '{app_name}' (ID: {app_id})")
                print_verbose(f'Will deploy an update to app ID: {app_id}')
                existing_properties['description'] = existing_app.description
                existing_properties['unauthenticated_access'] = existing_app.unauthenticatedAccess
            else:
                create_new_app = True
        except UnauthenticatedError as e:
            raise CLIError("Invalid API key for deployment. Run 'mcp-agent login' or set MCP_API_KEY environment variable with new API key.", retriable=False) from e
        except Exception as e:
            raise CLIError(f'Error checking for existing app: {str(e)}') from e
        if app_description is None:
            if default_app_description:
                app_description = default_app_description
        if deployed_secrets_file:
            if secrets_file:
                print_verbose(f"Both '{MCP_SECRETS_FILENAME}' and '{MCP_DEPLOYED_SECRETS_FILENAME}' found in {config_dir}.")
                if non_interactive:
                    print_info('Running in non-interactive mode — reusing previously-deployed secrets.')
                else:
                    reuse = typer.confirm('Reuse previously-deployed secrets?', default=True)
                    if not reuse:
                        deployed_secrets_file = None
            else:
                print_verbose(f"Found '{MCP_DEPLOYED_SECRETS_FILENAME}' in {config_dir}, but no '{MCP_SECRETS_FILENAME}' to re-process. Using existing deployed secrets file.")
        existing_properties = {k: v for k, v in existing_properties.items() if v is not None}
        update_payload = {k: v for k, v in update_payload.items() if v is not None}
        deployment_properties_display_info: List[Tuple[str, any, bool]] = [(lambda u, s: (name, u if u is not None else s, u is not None and u != s))(update_payload.get(k), existing_properties.get(k)) for k, name in configurable_fields if k in existing_properties or k in update_payload]
        print_deployment_header(app_name, app_id, config_file, secrets_file, deployed_secrets_file, deployment_properties_display_info)
        if non_interactive:
            start_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            print_info(f'[{start_time}] Running in non-interactive mode — proceeding with deployment.', highlight=False)
        else:
            proceed = typer.confirm('Proceed with deployment?', default=True)
            if not proceed:
                print_info('Deployment cancelled.')
                return None if create_new_app else app_id
            start_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            print_info(f'[{start_time}] Beginning deployment...', highlight=False)
        secrets_client = SecretsClient(api_url=effective_api_url, api_key=effective_api_key)
        if create_new_app:
            app = run_async(mcp_app_client.create_app(name=app_name, description=app_description, unauthenticated_access=unauthenticated_access))
            app_id = app.appId
            print_success(f"Created new app '{app_name}'")
            print_verbose(f'New app id: `{app_id}`')
        elif update_payload:
            print_verbose('Updating app settings before deployment...')
            run_async(mcp_app_client.update_app(app_id=app_id, **update_payload))
        if secrets_file and (not deployed_secrets_file):
            secrets_transformed_path = config_dir / MCP_DEPLOYED_SECRETS_FILENAME
            run_async(secrets_processor.process_config_secrets(input_path=secrets_file, output_path=secrets_transformed_path, client=secrets_client, api_url=effective_api_url, api_key=effective_api_key, non_interactive=non_interactive))
            print_success('Secrets file processed successfully')
            print_verbose(f'Transformed secrets file written to {secrets_transformed_path}')
            deployed_secrets_file = secrets_transformed_path
        else:
            print_verbose('Skipping secrets processing...')
        deployed_config_path, deployed_secrets_path = materialize_deployment_artifacts(config_dir=config_dir, app_id=app_id, config_file=config_file, deployed_secrets_path=config_dir / MCP_DEPLOYED_SECRETS_FILENAME, secrets_client=secrets_client, non_interactive=non_interactive)
        print_verbose(f'Materialized deployment config at {deployed_config_path} and secrets at {deployed_secrets_path}')
        if git_tag:
            git_meta = get_git_metadata(config_dir)
            if git_meta:
                safe_name = sanitize_git_ref_component(app_name)
                ts = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
                tag_name = f'mcp-deploy/{safe_name}/{ts}-{git_meta.short_sha}'
                msg = f"mcp-agent deploy for app '{app_name}' (ID: `{app_id}`)\nCommit: {git_meta.commit_sha}\nBranch: {git_meta.branch or ''}\nDirty: {git_meta.dirty}"
                if create_git_tag(config_dir, tag_name, msg):
                    print_success(f'Created local git tag: {tag_name}')
                else:
                    print_info('Skipping git tag (not a repo or tag failed)')
            else:
                print_info('Skipping git tag (not a git repository)')
        ignore_path: Optional[Path] = None
        if ignore_file is not None:
            ignore_path = ignore_file
        else:
            candidate = config_dir / '.mcpacignore'
            if not candidate.exists():
                candidate = Path.cwd() / '.mcpacignore'
            ignore_path = candidate if candidate.exists() else None
        app = run_async(_deploy_with_retry(app_id=app_id, api_key=effective_api_key, project_dir=config_dir, mcp_app_client=mcp_app_client, retry_count=retry_count, ignore=ignore_path))
        end_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        if create_new_app:
            print_info(f'[{end_time}] Deployment of {app_name} succeeded. ID: {app.appId}', highlight=False)
        else:
            print_info(f'[{end_time}] Deployment of {app_name} succeeded.', highlight=False)
        if app.appServerInfo:
            status = 'ONLINE' if app.appServerInfo.status == 'APP_SERVER_STATUS_ONLINE' else 'OFFLINE'
            server_url = app.appServerInfo.serverUrl
            print_info(f'App URL: [link={server_url}]{server_url}[/link]')
            print_info(f'App Status: {status}')
            if app.appServerInfo.unauthenticatedAccess is not None:
                auth_text = 'Not required (unauthenticated access allowed)' if app.appServerInfo.unauthenticatedAccess else 'Required'
                print_info(f'Authentication: {auth_text}')
            print_info(f'Use this app as an MCP server at {server_url}/sse\n\nMCP configuration example:')
            mcp_config = {'mcpServers': {app_name: {'url': f'{server_url}/sse', 'transport': 'sse', 'headers': {'Authorization': f'Bearer {effective_api_key}'}}}}
            console.print(f'[bright_black]{json.dumps(mcp_config, indent=2)}[/bright_black]', soft_wrap=True)
        return app_id
    except Exception as e:
        end_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        if LOG_VERBOSE.get():
            import traceback
            typer.echo(traceback.format_exc())
        raise CLIError(f'[{end_time}] Deployment failed: {str(e)}') from e

def print_deployment_header(app_name: str, existing_app_id: Optional[str], config_file: Path, secrets_file: Optional[Path], deployed_secrets_file: Optional[Path], deployment_properties_display_info: List[Tuple[str, any, bool]]) -> None:
    """Print a styled header for the deployment process."""
    deployed_secrets_file_message = '[bright_black]N/A[/bright_black]'
    if deployed_secrets_file:
        deployed_secrets_file_message = f'[cyan]{str(deployed_secrets_file)}[/cyan]'
    elif secrets_file:
        deployed_secrets_file_message = '[cyan]Pending creation[/cyan]'
    secrets_file_message = f'[cyan]{secrets_file}[/cyan]' if secrets_file else '[bright_black]N/A[/bright_black]'
    app_id_display = f'[ID: {existing_app_id}]' if existing_app_id else '[bright_yellow][NEW][/bright_yellow]'
    console.print(Panel('\n'.join([f'App: [cyan]{app_name}[/cyan] {app_id_display}', f'Configuration: [cyan]{config_file}[/cyan]', f'Secrets file: {secrets_file_message}', f'Deployed secrets file: {deployed_secrets_file_message}'] + [f'{name}: [{('bright_yellow' if is_changed else 'bright_black')}]{value}[/{('bright_yellow' if is_changed else 'bright_black')}]' for name, value, is_changed in deployment_properties_display_info]), title='mcp-agent deployment', subtitle='LastMile AI', border_style='blue', expand=False))
    logger.info(f'Starting deployment with configuration: {config_file}')
    logger.info(f'Using secrets file: {secrets_file or 'N/A'}, deployed secrets file: {deployed_secrets_file_message}')

def sanitize_git_ref_component(name: str) -> str:
    """Sanitize a string to be safe as a single refname component.

    Rules (aligned with `git check-ref-format` constraints and our usage):
    - Disallow spaces and special characters: ~ ^ : ? * [ \\ (replace with '-')
    - Replace '/' to avoid creating nested namespaces from user input
    - Collapse consecutive dots '..' into '-'
    - Remove leading dots '.' (cannot start with '.')
    - Remove trailing '.lock' and trailing dots
    - Disallow '@{' sequence
    - Ensure non-empty; fallback to 'unnamed'
    """
    s = name.strip()
    s = _INVALID_REF_CHARS.sub('-', s)
    s = s.replace('/', '-')
    s = re.sub('\\.{2,}', '-', s)
    s = s.replace('@{', '-{')
    s = re.sub('^[\\.-]+', '', s)
    s = re.sub('\\.lock$', '', s, flags=re.IGNORECASE)
    s = re.sub('\\.+$', '', s)
    if not s:
        s = 'unnamed'
    return s

def configure_app(ctx: typer.Context, app_server_url: str=typer.Option(None, '--id', '-i', help='Server URL of the app to configure.'), secrets_file: Optional[Path]=typer.Option(None, '--secrets-file', '-s', help='Path to a secrets.yaml file containing user secret IDs to use for configuring the app. If not provided, secrets will be prompted interactively.', exists=True, readable=True, dir_okay=False, resolve_path=True), secrets_output_file: Optional[Path]=typer.Option(None, '--secrets-output-file', '-o', help='Path to write prompted and tranformed secrets to. Defaults to mcp_agent.configured.secrets.yaml', resolve_path=True), dry_run: bool=typer.Option(False, '--dry-run', help="Validate the configuration but don't store secrets."), params: bool=typer.Option(False, '--params', help='Show required parameters (user secrets) for the configuration process and exit.'), api_url: Optional[str]=typer.Option(settings.API_BASE_URL, '--api-url', help='API base URL. Defaults to MCP_API_BASE_URL environment variable.', envvar=ENV_API_BASE_URL), api_key: Optional[str]=typer.Option(settings.API_KEY, '--api-key', help='API key for authentication. Defaults to MCP_API_KEY environment variable.', envvar=ENV_API_KEY), verbose: bool=typer.Option(False, '--verbose', '-v', help='Enable verbose output for this command')) -> str:
    """Configure an MCP app with the required params (e.g. user secrets).

    Args:
        app_server_url: Server URL of the MCP App to configure
        secrets_file: Path to an existing secrets file containing processed user secrets to use for configuring the app
        secrets_output_file: Path to write processed secrets to, if secrets are prompted. Defaults to mcp-agent.configured.secrets.yaml
        dry_run: Don't actually store secrets, just validate
        api_url: API base URL
        api_key: API key for authentication

    Returns:
        Configured app ID.
    """
    if verbose:
        LOG_VERBOSE.set(True)
    if not app_server_url:
        raise CLIError('You must provide a server URL to configure.')
    effective_api_key = api_key or settings.API_KEY or load_api_key_credentials()
    if not effective_api_key:
        raise CLIError("Must be logged in to configure. Run 'mcp-agent login', set MCP_API_KEY environment variable or specify --api-key option.")
    client: Union[MockMCPAppClient, MCPAppClient]
    if dry_run:
        print_verbose('Using MOCK API client for dry run')
        client = MockMCPAppClient(api_url=api_url or DEFAULT_API_BASE_URL, api_key=effective_api_key)
    else:
        client = MCPAppClient(api_url=api_url or DEFAULT_API_BASE_URL, api_key=effective_api_key)
    if secrets_file and secrets_output_file:
        raise CLIError('Cannot provide both --secrets-file and --secrets-output-file options. Please specify only one.')
    elif secrets_file and (not secrets_file.suffix == '.yaml'):
        raise CLIError('The --secrets-file must be a YAML file. Please provide a valid path.')
    elif secrets_output_file and (not secrets_output_file.suffix == '.yaml'):
        raise CLIError('The --secrets-output-file must be a YAML file. Please provide a valid path.')
    required_params = []
    try:
        required_params = run_async(client.list_config_params(app_server_url=app_server_url))
    except UnauthenticatedError as e:
        raise CLIError("Invalid API key. Run 'mcp-agent login' or set MCP_API_KEY environment variable with new API key.") from e
    except Exception as e:
        raise CLIError(f'Failed to retrieve required secrets for app {app_server_url}: {e}') from e
    requires_secrets = len(required_params) > 0
    configured_secrets = {}
    if params:
        if requires_secrets:
            print_info(f'App {app_server_url} requires the following ({len(required_params)}) user secrets: {', '.join(required_params)}')
        else:
            print_info(f'App {app_server_url} does not require any user secrets.')
        raise typer.Exit(0)
    if requires_secrets:
        if not secrets_file and secrets_output_file is None:
            secrets_output_file = Path(MCP_CONFIGURED_SECRETS_FILENAME)
            print_verbose(f'Using default output path: {secrets_output_file}')
        print_verbose(f'App {app_server_url} requires the following ({len(required_params)}) user secrets: {', '.join(required_params)}')
        try:
            print_verbose('Processing user secrets...')
            if dry_run:
                print_verbose('Using MOCK Secrets API client for dry run')
                mock_client = MockSecretsClient(api_url=api_url or DEFAULT_API_BASE_URL, api_key=effective_api_key)
                try:
                    configured_secrets = run_async(configure_user_secrets(required_secrets=required_params, config_path=secrets_file, output_path=secrets_output_file, client=mock_client))
                except Exception as e:
                    raise CLIError(f'Error during secrets processing with mock client: {str(e)}') from e
            else:
                configured_secrets = run_async(configure_user_secrets(required_secrets=required_params, config_path=secrets_file, output_path=secrets_output_file, api_url=api_url, api_key=effective_api_key))
            print_verbose('User secrets processed successfully')
        except Exception as e:
            if LOG_VERBOSE.get():
                import traceback
                typer.echo(traceback.format_exc())
            raise CLIError(f'{str(e)}') from e
    else:
        print_info(f'App {app_server_url} does not require any parameters.')
        if secrets_file:
            raise CLIError(f'App {app_server_url} does not require any parameters, but a secrets file was provided: {secrets_file}')
    print_configuration_header(app_server_url, required_params if requires_secrets else [], secrets_file, secrets_output_file, dry_run)
    if not dry_run:
        proceed = typer.confirm('Proceed with configuration?', default=True)
        if not proceed:
            print_info('Configuration cancelled.')
            return None
    else:
        print_info('Running in dry run mode.')
    start_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    print_info(f'[{start_time}] Starting configuration process...', highlight=False)
    if dry_run:
        print_success('Configuration completed in dry run mode.')
        return 'dry-run-app-configuration-id'
    config = None
    spinner_column = SpinnerColumn(spinner_name='aesthetic')
    with Progress('', spinner_column, TextColumn(' [progress.description]{task.description}')) as progress:
        task = progress.add_task('Configuring MCP App...', total=None)
        try:
            config = run_async(client.configure_app(app_server_url=app_server_url, config_params=configured_secrets))
            spinner_column.spinner.frames = spinner_column.spinner.frames[-2:-1]
            progress.update(task, description='MCP App configured successfully!')
        except Exception as e:
            progress.update(task, description='❌ MCP App configuration failed')
            end_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            raise CLIError(f'[{end_time}] Failed to configure app {app_server_url}: {str(e)}') from e
    end_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    if config.app:
        print_info(f"[{end_time}] Configuration of '{config.app.name}' succeeded. ID: {config.appConfigurationId}", highlight=False)
    else:
        print_info(f'[{end_time}] Configuration succeeded. ID: {config.appConfigurationId}', highlight=False)
    if config.appServerInfo:
        server_url = config.appServerInfo.serverUrl
        print_info(f'App Server URL: [link={server_url}]{server_url}[/link]')
        print_info(f'Use this configured app as an MCP server at {server_url}/sse\n\nMCP configuration example:')
        app_name = config.app.name if config.app else 'configured-app'
        mcp_config = {'mcpServers': {app_name: {'url': f'{server_url}/sse', 'transport': 'sse', 'headers': {'Authorization': f'Bearer {effective_api_key}'}}}}
        console.print(f'[bright_black]{json.dumps(mcp_config, indent=2)}[/bright_black]', soft_wrap=True)
    return config.appConfigurationId

def print_configuration_header(app_server_url: str, required_params: List[str], secrets_file: Optional[Path], output_file: Optional[Path], dry_run: bool) -> None:
    """Print a styled header for the configuration process."""
    sections = [f'App Server URL: [cyan]{app_server_url}[/cyan]']
    if required_params:
        sections.append(f'Required secrets: [cyan]{', '.join(required_params)}[/cyan]')
        sections.append(f'Secrets file: [cyan]{secrets_file or 'Will prompt for values'}[/cyan]')
        if output_file:
            sections.append(f'Output file: [cyan]{output_file}[/cyan]')
    else:
        sections.append('Required secrets: [bright_black]None[/bright_black]')
    if dry_run:
        sections.append('Mode: [yellow]DRY RUN[/yellow]')
    console.print(Panel('\n'.join(sections), title='mcp-agent configuration', subtitle='LastMile AI', border_style='blue', expand=False))
    logger.info(f'Starting configuration for app: {app_server_url}')
    logger.info(f'Required params: {required_params}')
    logger.info(f'Secrets file: {secrets_file}')
    logger.info(f'Output file: {output_file}')
    logger.info(f'Dry Run: {dry_run}')

def test_wrangler_deploy_file_copying(complex_project_structure):
    """Test that wrangler_deploy correctly copies project to temp directory and processes files."""
    temp_project_dir = None

    def check_files_during_subprocess(*args, **kwargs):
        nonlocal temp_project_dir
        temp_project_dir = Path(kwargs['cwd'])
        assert (temp_project_dir / 'README.md.mcpac.py').exists()
        assert (temp_project_dir / 'config.json.mcpac.py').exists()
        assert (temp_project_dir / 'data.txt.mcpac.py').exists()
        assert (temp_project_dir / 'requirements.txt.mcpac.py').exists()
        assert (temp_project_dir / 'nested/nested_config.yaml.mcpac.py').exists()
        assert (temp_project_dir / 'nested/nested_data.csv.mcpac.py').exists()
        assert (temp_project_dir / 'nested/deep/deep_file.txt.mcpac.py').exists()
        assert (temp_project_dir / 'main.py').exists()
        assert (temp_project_dir / 'nested/nested_script.py').exists()
        assert not (temp_project_dir / 'nested/nested_script.py.mcpac.py').exists()
        assert not (temp_project_dir / 'logs').exists()
        assert not (temp_project_dir / '.git').exists()
        assert not (temp_project_dir / '.venv').exists()
        assert not (temp_project_dir / '.hidden').exists()
        assert not (temp_project_dir / 'README.md').exists()
        assert not (temp_project_dir / 'config.json').exists()
        return MagicMock(returncode=0)
    with patch('subprocess.run', side_effect=check_files_during_subprocess):
        wrangler_deploy('test-app', 'test-api-key', complex_project_structure)
        assert (complex_project_structure / 'README.md').exists()
        assert (complex_project_structure / 'config.json').exists()
        assert not (complex_project_structure / 'README.md.mcpac.py').exists()

def test_wrangler_deploy_file_content_preservation(complex_project_structure):
    """Test that file content is preserved when copying to temp directory and renaming."""
    original_content = '# Test Project Content'
    (complex_project_structure / 'README.md').write_text(original_content)

    def check_content_during_subprocess(*args, **kwargs):
        temp_project_dir = Path(kwargs['cwd'])
        mcpac_file = temp_project_dir / 'README.md.mcpac.py'
        assert mcpac_file.exists()
        assert mcpac_file.read_text() == original_content
        return MagicMock(returncode=0)
    with patch('subprocess.run', side_effect=check_content_during_subprocess):
        wrangler_deploy('test-app', 'test-api-key', complex_project_structure)
        assert (complex_project_structure / 'README.md').exists()
        assert (complex_project_structure / 'README.md').read_text() == original_content
        assert not (complex_project_structure / 'README.md.mcpac.py').exists()

def test_wrangler_deploy_temp_directory_isolation(complex_project_structure):
    """Test that operations happen in temp directory without affecting original files."""
    original_files = ['README.md', 'config.json', 'data.txt', 'requirements.txt', 'nested/nested_config.yaml', 'nested/nested_data.csv']

    def check_files_during_subprocess(*args, **kwargs):
        temp_project_dir = Path(kwargs['cwd'])
        for file_path in original_files:
            original_file = complex_project_structure / file_path
            temp_mcpac_file = temp_project_dir / f'{file_path}.mcpac.py'
            temp_original_file = temp_project_dir / file_path
            assert original_file.exists(), f'Original {file_path} should still exist'
            assert temp_mcpac_file.exists(), f'Temp {file_path}.mcpac.py should exist'
            assert not temp_original_file.exists(), f'Temp {file_path} should be renamed'
        return MagicMock(returncode=0)
    with patch('subprocess.run', side_effect=check_files_during_subprocess):
        wrangler_deploy('test-app', 'test-api-key', complex_project_structure)
    for file_path in original_files:
        original_file = complex_project_structure / file_path
        assert original_file.exists(), f'Original {file_path} should be unchanged'

def test_wrangler_deploy_cleanup_on_success(complex_project_structure):
    """Test that original project files are untouched after successful deployment."""
    with patch('subprocess.run') as mock_subprocess:
        mock_subprocess.return_value = MagicMock(returncode=0)
        wrangler_deploy('test-app', 'test-api-key', complex_project_structure)
        assert not (complex_project_structure / 'README.md.mcpac.py').exists()
        assert not (complex_project_structure / 'config.json.mcpac.py').exists()
        assert not (complex_project_structure / 'nested/nested_config.yaml.mcpac.py').exists()
        assert (complex_project_structure / 'README.md').exists()
        assert (complex_project_structure / 'config.json').exists()
        assert (complex_project_structure / 'nested/nested_config.yaml').exists()
        assert not (complex_project_structure / 'wrangler.toml').exists()

def test_wrangler_deploy_cleanup_on_failure(complex_project_structure):
    """Test that original project files are untouched even when deployment fails."""
    with patch('subprocess.run') as mock_subprocess:
        mock_subprocess.side_effect = subprocess.CalledProcessError(returncode=1, cmd=['wrangler'], stderr='Deployment failed')
        with pytest.raises(subprocess.CalledProcessError):
            wrangler_deploy('test-app', 'test-api-key', complex_project_structure)
        assert not (complex_project_structure / 'README.md.mcpac.py').exists()
        assert not (complex_project_structure / 'config.json.mcpac.py').exists()
        assert (complex_project_structure / 'README.md').exists()
        assert (complex_project_structure / 'config.json').exists()
        assert not (complex_project_structure / 'wrangler.toml').exists()

def test_wrangler_deploy_venv_exclusion(complex_project_structure):
    """Test that .venv directory is excluded from temp directory copy."""
    venv_dir = complex_project_structure / '.venv'
    assert venv_dir.exists()
    (venv_dir / 'test_file').write_text('venv content')

    def check_venv_during_subprocess(*args, **kwargs):
        temp_project_dir = Path(kwargs['cwd'])
        assert not (temp_project_dir / '.venv').exists(), '.venv should not be copied to temp dir'
        assert venv_dir.exists(), 'Original .venv should still exist'
        return MagicMock(returncode=0)
    with patch('subprocess.run', side_effect=check_venv_during_subprocess):
        wrangler_deploy('test-app', 'test-api-key', complex_project_structure)
    assert venv_dir.exists(), '.venv should still exist'
    assert (venv_dir / 'test_file').exists(), '.venv content should be preserved'
    assert (venv_dir / 'test_file').read_text() == 'venv content'

def test_wrangler_deploy_nested_directory_creation(complex_project_structure):
    """Test that nested directory structure is preserved when creating .mcpac.py files in temp directory."""

    def check_nested_files_during_subprocess(*args, **kwargs):
        temp_project_dir = Path(kwargs['cwd'])
        nested_mcpac = temp_project_dir / 'nested/nested_config.yaml.mcpac.py'
        deep_mcpac = temp_project_dir / 'nested/deep/deep_file.txt.mcpac.py'
        assert nested_mcpac.exists(), 'Nested .mcpac.py file should exist during subprocess'
        assert deep_mcpac.exists(), 'Deep nested .mcpac.py file should exist during subprocess'
        assert nested_mcpac.parent == temp_project_dir / 'nested'
        assert deep_mcpac.parent == temp_project_dir / 'nested/deep'
        return MagicMock(returncode=0)
    with patch('subprocess.run', side_effect=check_nested_files_during_subprocess):
        wrangler_deploy('test-app', 'test-api-key', complex_project_structure)
        assert (complex_project_structure / 'nested/nested_config.yaml').exists()
        assert (complex_project_structure / 'nested/deep/deep_file.txt').exists()
        assert not (complex_project_structure / 'nested/nested_config.yaml.mcpac.py').exists()
        assert not (complex_project_structure / 'nested/deep/deep_file.txt.mcpac.py').exists()

def test_wrangler_deploy_file_permissions_preserved(complex_project_structure):
    """Test that file permissions are preserved when copying files."""
    test_file = complex_project_structure / 'executable.sh'
    test_file.write_text("#!/bin/bash\necho 'test'")
    if hasattr(os, 'chmod'):
        os.chmod(test_file, 493)

    def check_file_permissions_during_subprocess(*args, **kwargs):
        temp_project_dir = Path(kwargs['cwd'])
        assert oct((temp_project_dir / 'executable.sh.mcpac.py').stat().st_mode)[-3:] == '755'
        return MagicMock(returncode=0)
    with patch('subprocess.run', side_effect=check_file_permissions_during_subprocess):
        wrangler_deploy('test-app', 'test-api-key', complex_project_structure)

def test_wrangler_deploy_complex_file_extensions():
    """Test handling of files with complex extensions (e.g., .tar.gz, .config.json) in temp directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        (project_path / 'main.py').write_text('\nfrom mcp_agent_cloud import MCPApp\napp = MCPApp(name="test-app")\n')
        (project_path / 'requirements.txt').write_text('mcp-agent')
        complex_files = {'archive.tar.gz': 'archive content', 'config.json.template': 'template content', 'data.csv.backup': 'backup data', 'script.sh.orig': 'original script', 'file.name.with.multiple.dots.txt': 'multi-dot content'}
        for filename, content in complex_files.items():
            (project_path / filename).write_text(content)

        def check_complex_extensions_during_subprocess(*args, **kwargs):
            temp_project_dir = Path(kwargs['cwd'])
            for filename in complex_files.keys():
                mcpac_file = temp_project_dir / f'{filename}.mcpac.py'
                original_temp_file = temp_project_dir / filename
                original_project_file = project_path / filename
                assert mcpac_file.exists(), f'Temp {filename}.mcpac.py should exist during subprocess'
                assert not original_temp_file.exists(), f'Temp {filename} should be renamed during subprocess'
                assert original_project_file.exists(), f'Original {filename} should be unchanged'
            return MagicMock(returncode=0)
        with patch('subprocess.run', side_effect=check_complex_extensions_during_subprocess):
            wrangler_deploy('test-app', 'test-api-key', project_path)
            for filename, expected_content in complex_files.items():
                original_file = project_path / filename
                mcpac_file = project_path / f'{filename}.mcpac.py'
                assert original_file.exists(), f'Original {filename} should be unchanged'
                assert original_file.read_text() == expected_content, f'{filename} content should be preserved'
                assert not mcpac_file.exists(), f'No {filename}.mcpac.py should exist in original directory'

def test_needs_requirements_modification_no_file():
    """Test _needs_requirements_modification when requirements.txt doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        requirements_path = Path(temp_dir) / 'requirements.txt'
        assert not _needs_requirements_modification(requirements_path)

def test_needs_requirements_modification_no_relative_imports():
    """Test _needs_requirements_modification with no relative mcp-agent imports."""
    with tempfile.TemporaryDirectory() as temp_dir:
        requirements_path = Path(temp_dir) / 'requirements.txt'
        requirements_path.write_text('requests==2.31.0\nnumpy==1.24.0\nmcp-agent==1.0.0\npandas>=1.0.0')
        assert not _needs_requirements_modification(requirements_path)

def test_needs_requirements_modification_with_relative_imports():
    """Test _needs_requirements_modification with relative mcp-agent imports."""
    with tempfile.TemporaryDirectory() as temp_dir:
        requirements_path = Path(temp_dir) / 'requirements.txt'
        test_cases = ['mcp-agent @ file://../../', 'mcp-agent@file://../../', 'mcp-agent  @  file://../../some/path', 'mcp-agent @ file:///absolute/path']
        for relative_import in test_cases:
            requirements_content = f'requests==2.31.0\n{relative_import}\nnumpy==1.24.0'
            requirements_path.write_text(requirements_content)
            assert _needs_requirements_modification(requirements_path), f'Should detect relative import: {relative_import}'

def test_needs_requirements_modification_mixed_content():
    """Test _needs_requirements_modification with mixed content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        requirements_path = Path(temp_dir) / 'requirements.txt'
        requirements_content = '# This is a requirements file\nrequests==2.31.0\nnumpy==1.24.0\nmcp-agent @ file://../../\npandas>=1.0.0\n# Comment line\nfastapi==0.68.0'
        requirements_path.write_text(requirements_content)
        assert _needs_requirements_modification(requirements_path)

def test_modify_requirements_txt_relative_import():
    """Test _modify_requirements_txt with relative import."""
    with tempfile.TemporaryDirectory() as temp_dir:
        requirements_path = Path(temp_dir) / 'requirements.txt'
        original_content = 'requests==2.31.0\nmcp-agent @ file://../../\nnumpy==1.24.0'
        requirements_path.write_text(original_content)
        _modify_requirements_txt(requirements_path)
        modified_content = requirements_path.read_text()
        expected_content = 'requests==2.31.0\nmcp-agent\nnumpy==1.24.0'
        assert modified_content == expected_content

def test_modify_requirements_txt_preserves_formatting():
    """Test _modify_requirements_txt preserves comments and formatting."""
    with tempfile.TemporaryDirectory() as temp_dir:
        requirements_path = Path(temp_dir) / 'requirements.txt'
        original_content = '# Project dependencies\nrequests==2.31.0\n# Development version of mcp-agent\nmcp-agent @ file://../../\n\n# Data processing\nnumpy==1.24.0\npandas>=1.0.0\n'
        requirements_path.write_text(original_content)
        _modify_requirements_txt(requirements_path)
        modified_content = requirements_path.read_text()
        expected_content = '# Project dependencies\nrequests==2.31.0\n# Development version of mcp-agent\nmcp-agent\n\n# Data processing\nnumpy==1.24.0\npandas>=1.0.0\n'
        assert modified_content == expected_content

def test_wrangler_deploy_requirements_txt_modification_in_temp_dir(project_with_relative_mcp_agent):
    """Test that requirements.txt is modified in temp directory while original is untouched."""
    requirements_path = project_with_relative_mcp_agent / 'requirements.txt'
    original_content = requirements_path.read_text()

    def check_requirements_during_subprocess(*args, **kwargs):
        temp_project_dir = Path(kwargs['cwd'])
        temp_requirements = temp_project_dir / 'requirements.txt'
        temp_deployed_path = temp_project_dir / 'requirements.txt.mcpac.py'
        if temp_requirements.exists():
            modified_content = temp_requirements.read_text()
            assert 'mcp-agent @ file://' not in modified_content
            assert 'mcp-agent\n' in modified_content
        assert temp_deployed_path.exists()
        deployed_content = temp_deployed_path.read_text()
        assert 'mcp-agent @ file://' not in deployed_content
        assert 'mcp-agent\n' in deployed_content
        assert requirements_path.exists(), 'Original requirements.txt should be unchanged'
        assert requirements_path.read_text() == original_content
        return MagicMock(returncode=0)
    with patch('subprocess.run', side_effect=check_requirements_during_subprocess):
        wrangler_deploy('test-app', 'test-api-key', project_with_relative_mcp_agent)
    final_content = requirements_path.read_text()
    assert final_content == original_content
    assert 'mcp-agent @ file://../../' in final_content

def test_wrangler_deploy_requirements_txt_no_modification_needed(project_with_requirements):
    """Test that requirements.txt without relative imports is copied and renamed normally in temp directory."""
    requirements_path = project_with_requirements / 'requirements.txt'
    original_content = requirements_path.read_text()

    def check_requirements_during_subprocess(*args, **kwargs):
        temp_project_dir = Path(kwargs['cwd'])
        temp_mcpac_path = temp_project_dir / 'requirements.txt.mcpac.py'
        temp_requirements_path = temp_project_dir / 'requirements.txt'
        assert temp_mcpac_path.exists(), 'Temp requirements.txt.mcpac.py should exist'
        assert not temp_requirements_path.exists(), 'Temp requirements.txt should be renamed'
        assert temp_mcpac_path.read_text() == original_content
        assert requirements_path.exists(), 'Original requirements.txt should be unchanged'
        assert requirements_path.read_text() == original_content
        return MagicMock(returncode=0)
    with patch('subprocess.run', side_effect=check_requirements_during_subprocess):
        wrangler_deploy('test-app', 'test-api-key', project_with_requirements)
    final_content = requirements_path.read_text()
    assert final_content == original_content

def test_wrangler_deploy_no_requirements_txt():
    """Test that deployment works normally when no requirements.txt exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        (project_path / 'main.py').write_text('\nfrom mcp_agent_cloud import MCPApp\napp = MCPApp(name="test-app")\n')
        (project_path / 'pyproject.toml').write_text('[project]\nname = "test-app"\nversion = "0.1.0"\ndependencies = ["mcp-agent"]\n')
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = MagicMock(returncode=0)
            wrangler_deploy('test-app', 'test-api-key', project_path)
        assert not (project_path / 'requirements.txt').exists()

def test_wrangler_deploy_secrets_file_exclusion():
    """Test that mcp_agent.secrets.yaml is excluded from the bundle and not processed as mcpac.py."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        (project_path / 'main.py').write_text('\nfrom mcp_agent_cloud import MCPApp\napp = MCPApp(name="test-app")\n')
        (project_path / 'requirements.txt').write_text('mcp-agent')
        secrets_content = '\napi_key: !developer_secret\ndb_password: !developer_secret\n'
        secrets_file = project_path / MCP_SECRETS_FILENAME
        secrets_file.write_text(secrets_content)
        secrets_example_file = project_path / 'mcp_agent.secrets.yaml.example'
        secrets_example_file.write_text('\n# Example secrets file\napi_key: your_api_key_here\ndb_password: your_password_here\n')
        config_file = project_path / 'config.yaml'
        config_file.write_text('name: test-app')
        mcp_config_file = project_path / 'mcp_agent.config.yaml'
        mcp_config_file.write_text('config: value')
        mcp_deployed_secrets_file = project_path / 'mcp_agent.deployed.secrets.yaml'
        mcp_deployed_secrets_file.write_text('secret: mcpac_sc_tst')

        def check_secrets_exclusion_during_subprocess(*args, **kwargs):
            temp_project_dir = Path(kwargs['cwd'])
            assert not (temp_project_dir / MCP_SECRETS_FILENAME).exists(), 'Secrets file should be excluded from temp directory'
            assert not (temp_project_dir / f'{MCP_SECRETS_FILENAME}.mcpac.py').exists(), 'Secrets file should not be processed as .mcpac.py'
            assert (temp_project_dir / 'mcp_agent.secrets.yaml.example.mcpac.py').exists()
            assert (temp_project_dir / 'config.yaml.mcpac.py').exists(), 'Other YAML files should be processed as .mcpac.py'
            assert (temp_project_dir / 'mcp_agent.config.yaml.mcpac.py').exists(), 'mcp_agent.config.yaml should be processed as .mcpac.py'
            assert (temp_project_dir / 'mcp_agent.deployed.secrets.yaml.mcpac.py').exists(), 'mcp_agent.deployed.secrets.yaml should be processed as .mcpac.py'
            assert not (temp_project_dir / 'config.yaml').exists(), 'Other YAML files should be renamed in temp directory'
            assert secrets_file.exists(), 'Original secrets file should remain untouched'
            assert config_file.exists(), 'Original config file should remain untouched'
            assert secrets_file.read_text() == secrets_content, 'Secrets file content should be unchanged'
            return MagicMock(returncode=0)
        with patch('subprocess.run', side_effect=check_secrets_exclusion_during_subprocess):
            wrangler_deploy('test-app', 'test-api-key', project_path)
        assert secrets_file.exists(), 'Secrets file should still exist'
        assert secrets_file.read_text() == secrets_content, 'Secrets file content should be preserved'
        assert secrets_example_file.exists()
        assert config_file.exists(), 'Config file should still exist'
        assert not (project_path / f'{MCP_SECRETS_FILENAME}.mcpac.py').exists(), 'No secrets .mcpac.py file should exist in original directory'

def test_wrangler_deploy_with_ignore_file():
    """Bundling honours explicit ignore file patterns end to end.

    Creates a project containing included and excluded files, supplies a real
    `.mcpacignore`, and checks the temp bundle only contains files that should
    survive, proving the ignore spec is wired into `copytree` correctly.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        (project_path / 'main.py').write_text('\nfrom mcp_agent_cloud import MCPApp\napp = MCPApp(name="test-app")\n')
        (project_path / 'requirements.txt').write_text('mcp-agent')
        ignore_content = '*.log\n*.tmp\nbuild/\ndist/\n*.pyc\n'
        (project_path / '.mcpacignore').write_text(ignore_content)
        (project_path / 'debug.log').write_text('log content')
        (project_path / 'temp.tmp').write_text('temp content')
        (project_path / 'cache.pyc').write_text('pyc content')
        build_dir = project_path / 'build'
        build_dir.mkdir()
        (build_dir / 'output.txt').write_text('build output')
        (project_path / 'config.yaml').write_text('config: value')
        (project_path / 'data.txt').write_text('data content')

        def check_gitignore_respected(*args, **kwargs):
            temp_project_dir = Path(kwargs['cwd'])
            assert not (temp_project_dir / 'debug.log').exists()
            assert not (temp_project_dir / 'temp.tmp').exists()
            assert not (temp_project_dir / 'cache.pyc').exists()
            assert not (temp_project_dir / 'build').exists()
            assert (temp_project_dir / 'main.py').exists()
            assert (temp_project_dir / 'config.yaml.mcpac.py').exists()
            assert (temp_project_dir / 'data.txt.mcpac.py').exists()
            return MagicMock(returncode=0)
        with patch('subprocess.run', side_effect=check_gitignore_respected):
            wrangler_deploy('test-app', 'test-api-key', project_path, project_path / '.mcpacignore')

def test_wrangler_deploy_warns_when_ignore_file_missing():
    """Missing ignore files should warn but still bundle everything.

    Passes a nonexistent ignore path, asserts `print_warning` reports the issue,
    and that the temporary bundle still includes files that would only be
    skipped by an actual ignore spec.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        (project_path / 'main.py').write_text('\nfrom mcp_agent_cloud import MCPApp\n\napp = MCPApp(name="test-app")\n')
        (project_path / 'requirements.txt').write_text('mcp-agent')
        (project_path / 'config.yaml').write_text('name: test-app\n')
        (project_path / 'artifact.txt').write_text('artifact\n')
        missing_ignore = project_path / '.customignore'

        def check_missing_ignore_behavior(*args, **kwargs):
            temp_project_dir = Path(kwargs['cwd'])
            assert (temp_project_dir / 'artifact.txt.mcpac.py').exists()
            assert (temp_project_dir / 'config.yaml.mcpac.py').exists()
            return MagicMock(returncode=0)
        with patch('mcp_agent.cli.cloud.commands.deploy.wrangler_wrapper.print_warning') as mock_warning, patch('subprocess.run', side_effect=check_missing_ignore_behavior):
            wrangler_deploy('test-app', 'test-api-key', project_path, missing_ignore)
        mock_warning.assert_called_once()
        warning_message = mock_warning.call_args[0][0]
        assert str(missing_ignore) in warning_message
        assert 'not found' in warning_message

def test_deploy_with_secrets_file():
    """Test the deploy command with a secrets file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_content = '\nserver:\n  host: example.com\n  port: 443\n'
        config_path = temp_path / MCP_CONFIG_FILENAME
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        secrets_content = '\nserver:\n  api_key: mock-server-api-key\n  user_token: mock-server-user-token\n'
        secrets_path = temp_path / MCP_SECRETS_FILENAME
        with open(secrets_path, 'w', encoding='utf-8') as f:
            f.write(secrets_content)
        mock_client = AsyncMock()
        mock_client.get_app_id_by_name = AsyncMock(return_value=None)
        mock_existing_app = MagicMock()
        mock_existing_app.appId = MOCK_APP_ID
        mock_existing_app.description = 'Test app description'
        mock_existing_app.unauthenticatedAccess = False
        mock_client.get_app_by_name = AsyncMock(return_value=mock_existing_app)
        mock_app = MagicMock()
        mock_app.appId = MOCK_APP_ID
        mock_client.create_app = AsyncMock(return_value=mock_app)
        mock_client.update_app = AsyncMock(return_value=mock_app)
        with patch('mcp_agent.cli.cloud.commands.deploy.main.wrangler_deploy', return_value=MOCK_APP_ID), patch('mcp_agent.cli.cloud.commands.deploy.main.MCPAppClient', return_value=mock_client):
            result = deploy_config(ctx=MagicMock(), app_name=MOCK_APP_NAME, app_description='A test MCP Agent app', config_dir=temp_path, api_url='http://test.api/', api_key='test-token', non_interactive=True, retry_count=3, verbose=False)
            secrets_output = temp_path / MCP_DEPLOYED_SECRETS_FILENAME
            assert os.path.exists(secrets_output), 'Output file should exist'
            with open(secrets_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert content == secrets_content, 'Output file content should match original secrets'
            assert result == MOCK_APP_ID

def direct_configure_app(**kwargs):
    kwargs.setdefault('api_url', 'http://test-api')
    kwargs.setdefault('api_key', 'test-token')
    kwargs.setdefault('verbose', False)
    mock_ctx = MagicMock()
    return configure_app(mock_ctx, **kwargs)

def test_output_secrets_file_creation(tmp_path):
    """Test that the output secrets file is created with valid content."""
    required_secrets = ['server.bedrock.api_key', 'server.openai.api_key']
    processed_secrets = {'server.bedrock.api_key': 'mcpac_sc_12345678-1234-1234-1234-123456789012', 'server.openai.api_key': 'mcpac_sc_87654321-4321-4321-4321-210987654321'}
    mock_client = MagicMock()
    mock_client.list_config_params = AsyncMock(return_value=required_secrets)
    mock_app = MagicMock()
    mock_app.appId = MOCK_APP_ID
    mock_client.get_app = AsyncMock(return_value=mock_app)
    mock_config = MagicMock()
    mock_config.appConfigurationId = MOCK_APP_CONFIG_ID
    mock_config.appServerInfo = MagicMock()
    mock_config.appServerInfo.serverUrl = 'https://test-server.example.com'
    mock_config.app = MagicMock()
    mock_config.app.name = 'Test App'
    mock_client.configure_app = AsyncMock(return_value=mock_config)
    secrets_output_file = tmp_path / 'test_output_secrets.yaml'
    _create_test_secrets_file(secrets_output_file, processed_secrets)
    with patch('mcp_agent.cli.cloud.commands.configure.main.MCPAppClient', return_value=mock_client), patch('mcp_agent.cli.cloud.commands.configure.main.MockMCPAppClient', return_value=mock_client), patch('mcp_agent.cli.cloud.commands.configure.main.configure_user_secrets', AsyncMock(return_value=processed_secrets)), patch('mcp_agent.cli.cloud.commands.configure.main.typer.Exit', side_effect=RuntimeError), patch('mcp_agent.cli.cloud.commands.configure.main.typer.confirm', return_value=True):
        try:

            def direct_configure_app(**kwargs):
                kwargs.setdefault('api_url', 'http://test-api')
                kwargs.setdefault('api_key', 'test-token')
                kwargs.setdefault('verbose', False)
                mock_ctx = MagicMock()
                return configure_app(mock_ctx, **kwargs)
            result = direct_configure_app(app_server_url=MOCK_APP_SERVER_URL, secrets_file=None, secrets_output_file=secrets_output_file, dry_run=False, params=False)
            assert result == MOCK_APP_CONFIG_ID
            assert secrets_output_file.exists()
            with open(secrets_output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            assert 'mcpac_sc_12345678-1234-1234-1234-123456789012' in content
            assert 'mcpac_sc_87654321-4321-4321-4321-210987654321' in content
            yaml_content = yaml.safe_load(content)
            assert yaml_content['server']['bedrock']['api_key'] == 'mcpac_sc_12345678-1234-1234-1234-123456789012'
            assert yaml_content['server']['openai']['api_key'] == 'mcpac_sc_87654321-4321-4321-4321-210987654321'
        except RuntimeError as e:
            if 'Typer exit with code' not in str(e):
                raise

