# Cluster 35

def dump_yaml_with_secrets(data):
    """
    Dump Python objects to YAML string, properly handling secret tags.

    Args:
        data: Python object that may contain UserSecret or DeveloperSecret objects

    Returns:
        YAML string with proper secret tags
    """
    yaml_str = yaml.dump(data, Dumper=SecretYamlDumper, default_flow_style=False)
    return re.sub("(!user_secret|!developer_secret) \\'\\'", '\\1', yaml_str)

def print_secret_summary(secrets_context: Dict[str, Any]) -> None:
    """Print a summary of processed secrets from context.

    Args:
        secrets_context: Dictionary containing info about processed secrets
    """
    deployment_secrets = secrets_context.get('deployment_secrets', [])
    user_secrets = secrets_context.get('user_secrets', [])
    reused_secrets = secrets_context.get('reused_secrets', [])
    skipped_secrets = secrets_context.get('skipped_secrets', [])
    return print_secrets_summary(deployment_secrets, user_secrets, reused_secrets, skipped_secrets)

def load_yaml_with_secrets(yaml_str):
    """
    Load YAML string containing secret tags into Python objects.

    Args:
        yaml_str: YAML string that may contain !user_secret or !developer_secret tags

    Returns:
        Parsed Python object with UserSecret and DeveloperSecret objects
    """
    return yaml.load(yaml_str, Loader=SecretYamlLoader)

def get_at_path(config_dict, path_str):
    if not config_dict or not path_str:
        return None
    parts = path_str.split('.')
    curr = config_dict
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        elif '[' in part and ']' in part:
            base_part = part.split('[')[0]
            idx_str = part.split('[')[1].split(']')[0]
            try:
                idx = int(idx_str)
                if base_part in curr and isinstance(curr[base_part], list) and (idx < len(curr[base_part])):
                    curr = curr[base_part][idx]
                else:
                    return None
            except (ValueError, IndexError):
                return None
        else:
            return None
    return curr

def _load_deployed_secrets(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = path.read_text(encoding='utf-8')
    loaded = load_yaml_with_secrets(raw)
    return loaded or {}

def _persist_deployed_secrets(path: Path, data: dict) -> None:
    content = dump_yaml_with_secrets(data)
    path.write_text(content, encoding='utf-8')

def _redact_config_values(current: object, secrets_overlay: object, raw_config: object) -> object:
    """Return `current` with any nodes present in `secrets_overlay` removed or replaced with `raw_config` values."""
    if secrets_overlay is None:
        return current
    if isinstance(secrets_overlay, dict) and isinstance(current, dict):
        result: dict = copy.deepcopy(current)
        raw_dict = raw_config if isinstance(raw_config, dict) else {}
        for key, overlay_value in secrets_overlay.items():
            if key not in result:
                continue
            base_value = raw_dict.get(key)
            replacement = _redact_config_values(result[key], overlay_value, base_value)
            if replacement is _REMOVE:
                if base_value is not None:
                    result[key] = copy.deepcopy(base_value)
                else:
                    result.pop(key, None)
            else:
                result[key] = replacement
        if not result:
            if raw_dict:
                return copy.deepcopy(raw_dict)
            return _REMOVE
        return result
    if isinstance(secrets_overlay, list) and isinstance(current, list):
        raw_list = raw_config if isinstance(raw_config, list) else []
        result_list = []
        max_len = len(current)
        for idx in range(max_len):
            item = current[idx]
            overlay_item = secrets_overlay[idx] if idx < len(secrets_overlay) else None
            base_item = raw_list[idx] if idx < len(raw_list) else None
            if overlay_item is None:
                result_list.append(item)
                continue
            replacement = _redact_config_values(item, overlay_item, base_item)
            if replacement is _REMOVE:
                if base_item is not None:
                    result_list.append(copy.deepcopy(base_item))
            else:
                result_list.append(replacement)
        return result_list
    if raw_config is not None:
        return copy.deepcopy(raw_config)
    return _REMOVE

def materialize_deployment_artifacts(*, config_dir: Path, app_id: str, config_file: Path, deployed_secrets_path: Path, secrets_client: SecretsClient, non_interactive: bool) -> tuple[Path, Path]:
    """Generate deployment-ready config and secrets files.

    Returns the paths to the deployed config and secrets files.
    """
    if not config_file.exists():
        raise CLIError(f'Configuration file not found: {config_file}')
    settings = _load_settings_from_app(config_dir)
    settings_source = 'main.py MCPApp'
    if settings is None:
        settings_source = str(config_file)
        try:
            settings = get_settings(config_path=str(config_file), set_global=False)
        except Exception as exc:
            typer.secho(f'Skipping deployment materialization due to config error: {exc}', fg=typer.colors.YELLOW)
            if not deployed_secrets_path.exists():
                deployed_secrets_path.write_text(yaml.safe_dump({}, default_flow_style=False, sort_keys=False), encoding='utf-8')
            return (config_file, deployed_secrets_path)
    typer.secho(f'Materializing config from {settings_source}', fg=typer.colors.BLUE)
    env_specs = _normalize_env_specs(settings)
    secrets_data = _load_deployed_secrets(deployed_secrets_path)
    materialized_config = settings.model_dump(mode='json', exclude_none=True, exclude_unset=True, exclude_defaults=True)
    raw_config = _load_raw_config(config_file)
    sanitized_config = _redact_config_values(copy.deepcopy(materialized_config), copy.deepcopy(secrets_data), raw_config)
    deployed_config_path = config_dir / MCP_DEPLOYED_CONFIG_FILENAME
    _write_deployed_config(deployed_config_path, sanitized_config or {})
    if not env_specs:
        if not deployed_secrets_path.exists():
            deployed_secrets_path.write_text(yaml.safe_dump({}, default_flow_style=False, sort_keys=False), encoding='utf-8')
        return (deployed_config_path, deployed_secrets_path)
    secrets_path_parent = deployed_secrets_path.parent
    secrets_path_parent.mkdir(parents=True, exist_ok=True)
    existing_env_handles = _extract_existing_env_handles(secrets_data)
    normalized_env_entries: list[dict[str, str]] = []
    for spec in env_specs:
        value = os.environ.get(spec.key)
        fallback_used = False
        if value is None:
            if spec.fallback is not None:
                value = str(spec.fallback)
                fallback_used = True
            elif non_interactive:
                raise CLIError(f"Environment variable '{spec.key}' is required but not set. Provide it via the environment, configure a fallback, or rerun without --non-interactive.")
            else:
                prompt_text = f"Enter value for environment variable '{spec.key}'"
                value = typer.prompt(prompt_text, hide_input=True)
                fallback_used = True
        if value is None or value == '':
            raise CLIError(f"Environment variable '{spec.key}' resolved to an empty value. Provide a non-empty value via the environment or configuration.")
        handle = existing_env_handles.get(spec.key)
        secret_name = _secret_name_for_env(app_id, spec.key)
        handle_reused = False
        if handle:
            try:
                success = run_async(secrets_client.set_secret_value(handle, value))
                if success:
                    handle_reused = True
                else:
                    typer.secho(f"Existing secret handle for '{spec.key}' is invalid; creating a new secret.", fg=typer.colors.YELLOW)
                    handle = None
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    typer.secho(f"Secret handle for '{spec.key}' no longer exists; creating a new secret.", fg=typer.colors.YELLOW)
                    handle = None
                else:
                    raise
            except Exception as exc:
                typer.secho(f"Failed to reuse secret handle for '{spec.key}': {exc}. Creating a new secret.", fg=typer.colors.YELLOW)
                handle = None
        if not handle:
            handle = run_async(secrets_client.create_secret(name=secret_name, secret_type=SecretType.DEVELOPER, value=value))
            handle_reused = False
        if not handle_reused:
            existing_env_handles[spec.key] = handle
        normalized_env_entries.append({spec.key: handle})
        if fallback_used and spec.fallback is None:
            typer.secho(f"Captured value for '{spec.key}' during deployment; it will be stored as a secret.", fg=typer.colors.BLUE)
    secrets_data['env'] = normalized_env_entries
    _persist_deployed_secrets(deployed_secrets_path, secrets_data)
    return (deployed_config_path, deployed_secrets_path)

def _normalize_env_specs(settings: Settings) -> list[EnvSpec]:
    """Coerce the flexible env syntax into ordered EnvSpec rows."""
    specs: list[EnvSpec] = []
    for key, fallback in settings.iter_env_specs():
        specs.append(EnvSpec(key=key, fallback=fallback))
    return specs

def _load_raw_config(config_file: Path) -> dict:
    if not config_file.exists():
        return {}
    try:
        return yaml.safe_load(config_file.read_text(encoding='utf-8')) or {}
    except Exception:
        return {}

def _write_deployed_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)

def _extract_existing_env_handles(data: dict) -> dict[str, str]:
    env_section = data.get('env')
    handles: dict[str, str] = {}
    if isinstance(env_section, list):
        for item in env_section:
            if isinstance(item, dict) and len(item) == 1:
                key, value = next(iter(item.items()))
                if isinstance(key, str) and isinstance(value, str):
                    handles[key] = value
    return handles

def _secret_name_for_env(app_id: str, key: str) -> str:
    return f'apps/{app_id}/env/{key}'

class TestYamlSecretTags(unittest.TestCase):
    """Test case for YAML secret tag handling."""

    def test_basic_round_trip(self):
        """Test basic round-trip serialization and deserialization."""
        config = {'server': {'api_key': DeveloperSecret('some-value'), 'empty_dev_secret': DeveloperSecret(), 'user_token': UserSecret('user-value'), 'empty_user_secret': UserSecret()}}
        yaml_str = dump_yaml_with_secrets(config)
        self.assertIn("api_key: !developer_secret 'some-value'", yaml_str)
        self.assertIn('empty_dev_secret: !developer_secret', yaml_str)
        self.assertIn("user_token: !user_secret 'user-value'", yaml_str)
        self.assertIn('empty_user_secret: !user_secret', yaml_str)
        loaded = load_yaml_with_secrets(yaml_str)
        self.assertIsInstance(loaded, dict)
        self.assertIn('server', loaded)
        server = loaded['server']
        self.assertIsInstance(server['api_key'], DeveloperSecret)
        self.assertEqual(server['api_key'].value, 'some-value')
        self.assertIsInstance(server['empty_dev_secret'], DeveloperSecret)
        self.assertIsNone(server['empty_dev_secret'].value)
        self.assertIsInstance(server['user_token'], UserSecret)
        self.assertEqual(server['user_token'].value, 'user-value')
        self.assertIsInstance(server['empty_user_secret'], UserSecret)
        self.assertIsNone(server['empty_user_secret'].value)

    def test_direct_yaml_format(self):
        """Test loading YAML string with empty tags directly."""
        yaml_with_empty_tags = "\nserver:\n  api_key: !developer_secret 'key123'\n  empty_dev_secret: !developer_secret\n  user_token: !user_secret 'token456'\n  empty_user_secret: !user_secret\n"
        loaded = load_yaml_with_secrets(yaml_with_empty_tags)
        server = loaded['server']
        self.assertEqual(server['api_key'].value, 'key123')
        self.assertIsNone(server['empty_dev_secret'].value)
        self.assertEqual(server['user_token'].value, 'token456')
        self.assertIsNone(server['empty_user_secret'].value)

    def test_nested_structure(self):
        """Test handling of secrets in nested structures."""
        config = {'server': {'providers': {'bedrock': {'api_key': DeveloperSecret('bedrock-key')}, 'openai': {'api_key': UserSecret('openai-key')}}}}
        yaml_str = dump_yaml_with_secrets(config)
        loaded = load_yaml_with_secrets(yaml_str)
        self.assertEqual(loaded['server']['providers']['bedrock']['api_key'].value, 'bedrock-key')
        self.assertEqual(loaded['server']['providers']['openai']['api_key'].value, 'openai-key')

    def test_integration_with_standard_yaml(self):
        """Test that our custom tags work with standard YAML functions."""
        config = {'server': {'api_key': DeveloperSecret('api-key'), 'port': 8080, 'debug': True}}
        yaml_str = yaml.dump(config, Dumper=SecretYamlDumper, default_flow_style=False)
        processed_yaml = yaml_str.replace(" ''", '')
        loaded = yaml.load(processed_yaml, Loader=SecretYamlLoader)
        self.assertEqual(loaded['server']['port'], 8080)
        self.assertEqual(loaded['server']['debug'], True)
        self.assertIsInstance(loaded['server']['api_key'], DeveloperSecret)
        self.assertEqual(loaded['server']['api_key'].value, 'api-key')

class TestYamlSecretTags(TestCase):
    """Test handling of YAML tags for secrets."""

    def test_round_trip_serialization(self):
        """Test that secrets can be round-tripped through YAML."""
        test_cases = [{'server': {'api_key': DeveloperSecret('dev-api-key'), 'user_token': UserSecret('user-token')}}, {'server': {'api_key': DeveloperSecret(), 'user_token': UserSecret()}}, {'server': {'providers': {'bedrock': {'api_key': DeveloperSecret('bedrock-key'), 'region': 'us-west-2'}, 'openai': {'api_key': UserSecret('openai-key'), 'org_id': 'org-123'}}, 'database': {'password': DeveloperSecret('db-password'), 'user_password': UserSecret('user-db-password')}}}, {'server': {'api_key': DeveloperSecret('dev-api-key'), 'port': 8080, 'debug': True, 'tags': ['prod', 'us-west'], 'metadata': {'created_at': '2023-01-01', 'created_by': UserSecret('user-123')}}}]
        for config in test_cases:
            yaml_str = dump_yaml_with_secrets(config)
            loaded = load_yaml_with_secrets(yaml_str)
            self._verify_config_structure(config, loaded)

    def _verify_config_structure(self, original, loaded):
        """Helper to verify config structure is preserved."""
        if isinstance(original, dict):
            assert isinstance(loaded, dict)
            for key, value in original.items():
                assert key in loaded
                self._verify_config_structure(value, loaded[key])
        elif isinstance(original, list):
            assert isinstance(loaded, list)
            assert len(original) == len(loaded)
            for orig_item, loaded_item in zip(original, loaded):
                self._verify_config_structure(orig_item, loaded_item)
        elif isinstance(original, DeveloperSecret):
            assert isinstance(loaded, DeveloperSecret)
            assert loaded.value == original.value
        elif isinstance(original, UserSecret):
            assert isinstance(loaded, UserSecret)
            assert loaded.value == original.value
        else:
            assert loaded == original

    def test_empty_tags_handling(self):
        """Test handling of empty tags."""
        yaml_str = '\n        server:\n          empty_dev_secret: !developer_secret\n          empty_user_secret: !user_secret\n        '
        loaded = load_yaml_with_secrets(yaml_str)
        assert isinstance(loaded['server']['empty_dev_secret'], DeveloperSecret)
        assert loaded['server']['empty_dev_secret'].value is None
        assert isinstance(loaded['server']['empty_user_secret'], UserSecret)
        assert loaded['server']['empty_user_secret'].value is None
        dumped = dump_yaml_with_secrets(loaded)
        assert '!developer_secret ""' not in dumped
        assert '!user_secret ""' not in dumped
        assert 'empty_dev_secret: !developer_secret' in dumped
        assert 'empty_user_secret: !user_secret' in dumped

    def test_uuid_handle_handling(self):
        """Test handling of UUID handles."""
        yaml_str = f'\n        server:\n          bedrock:\n            # Deployed secret with UUID handle\n            api_key: "{UUID_PREFIX}12345678-abcd-1234-a123-123456789abc"\n            # User secret that will be collected during configure\n            user_access_key: !user_secret USER_KEY\n        database:\n          # Another deployed secret with UUID handle\n          password: "{UUID_PREFIX}87654321-dcba-4321-b321-987654321cba"\n        '
        loaded = load_yaml_with_secrets(yaml_str)
        assert isinstance(loaded['server']['bedrock']['api_key'], str)
        assert loaded['server']['bedrock']['api_key'].startswith(UUID_PREFIX)
        assert loaded['server']['bedrock']['api_key'] == f'{UUID_PREFIX}12345678-abcd-1234-a123-123456789abc'
        assert SECRET_ID_PATTERN.match(loaded['server']['bedrock']['api_key']) is not None
        assert SECRET_ID_PATTERN.match(loaded['database']['password']) is not None
        assert isinstance(loaded['server']['bedrock']['user_access_key'], UserSecret)
        assert loaded['server']['bedrock']['user_access_key'].value == 'USER_KEY'
        dumped = dump_yaml_with_secrets(loaded)
        reloaded = load_yaml_with_secrets(dumped)
        assert reloaded['server']['bedrock']['api_key'] == f'{UUID_PREFIX}12345678-abcd-1234-a123-123456789abc'
        assert reloaded['database']['password'] == f'{UUID_PREFIX}87654321-dcba-4321-b321-987654321cba'
        assert isinstance(reloaded['server']['bedrock']['user_access_key'], UserSecret)
        assert reloaded['server']['bedrock']['user_access_key'].value == 'USER_KEY'

    def test_uuid_pattern_validation(self):
        """Test UUID pattern validation for handles."""
        valid_handles = [f'{UUID_PREFIX}12345678-abcd-1234-a123-123456789abc', f'{UUID_PREFIX}00000000-0000-0000-0000-000000000000', f'{UUID_PREFIX}ffffffff-ffff-ffff-ffff-ffffffffffff']
        invalid_handles = ['12345678-abcd-1234-a123-123456789abc', 'wrong_prefix_12345678-abcd-1234-a123-123456789abc', f'{UUID_PREFIX}12345678abcd1234a123123456789abc', f'{UUID_PREFIX}12345678-abcd-1234-a123', f'{UUID_PREFIX}1234567g-abcd-1234-a123-123456789abc', '']
        for handle in valid_handles:
            assert SECRET_ID_PATTERN.match(handle) is not None, f"Valid handle {handle} didn't match pattern"
        for handle in invalid_handles:
            assert SECRET_ID_PATTERN.match(handle) is None, f'Invalid handle {handle} matched pattern'

def test_realistic_yaml_examples():
    """Test handling of realistic YAML examples."""
    yaml_str = '\n    # Example deployment configuration with secrets\n    server:\n      bedrock:\n        # Value comes from env var BEDROCK_KEY\n        api_key: !developer_secret BEDROCK_KEY\n        # Value collected during configure, env var USER_KEY is an override\n        user_access_key: !user_secret USER_KEY \n      openai:\n        api_key: !developer_secret\n        org_id: "org-123456"\n    database:\n      # Must be prompted for during deploy\n      password: !developer_secret \n      host: "localhost"\n      port: 5432\n    '
    loaded = load_yaml_with_secrets(yaml_str)
    assert isinstance(loaded['server']['bedrock']['api_key'], DeveloperSecret)
    assert loaded['server']['bedrock']['api_key'].value == 'BEDROCK_KEY'
    assert isinstance(loaded['server']['bedrock']['user_access_key'], UserSecret)
    assert loaded['server']['bedrock']['user_access_key'].value == 'USER_KEY'
    assert isinstance(loaded['server']['openai']['api_key'], DeveloperSecret)
    assert loaded['server']['openai']['api_key'].value is None
    assert loaded['server']['openai']['org_id'] == 'org-123456'
    assert isinstance(loaded['database']['password'], DeveloperSecret)
    assert loaded['database']['password'].value is None
    assert loaded['database']['host'] == 'localhost'
    assert loaded['database']['port'] == 5432
    dumped = dump_yaml_with_secrets(loaded)
    reloaded = load_yaml_with_secrets(dumped)
    assert isinstance(reloaded['server']['bedrock']['api_key'], DeveloperSecret)
    assert reloaded['server']['bedrock']['api_key'].value == 'BEDROCK_KEY'
    assert isinstance(reloaded['server']['bedrock']['user_access_key'], UserSecret)
    assert reloaded['server']['bedrock']['user_access_key'].value == 'USER_KEY'
    assert isinstance(reloaded['server']['openai']['api_key'], DeveloperSecret)
    assert reloaded['server']['openai']['api_key'].value is None
    assert isinstance(reloaded['database']['password'], DeveloperSecret)
    assert reloaded['database']['password'].value is None

def test_deployed_secrets_example():
    """Test handling of post-deployment YAML with UUID handles."""
    yaml_str = f'\n    # Post-deployment configuration\n    server:\n      bedrock:\n        api_key: "{UUID_PREFIX}12345678-abcd-1234-a123-123456789abc"\n        # User secret tag remains for configure phase\n        user_access_key: !user_secret USER_KEY \n      openai:\n        api_key: "{UUID_PREFIX}23456789-bcde-2345-b234-234567890bcd"\n    database:\n      password: "{UUID_PREFIX}87654321-dcba-4321-b321-987654321cba"\n    '
    loaded = load_yaml_with_secrets(yaml_str)
    assert loaded['server']['bedrock']['api_key'] == f'{UUID_PREFIX}12345678-abcd-1234-a123-123456789abc'
    assert isinstance(loaded['server']['bedrock']['user_access_key'], UserSecret)
    assert loaded['server']['bedrock']['user_access_key'].value == 'USER_KEY'
    assert loaded['server']['openai']['api_key'] == f'{UUID_PREFIX}23456789-bcde-2345-b234-234567890bcd'
    assert loaded['database']['password'] == f'{UUID_PREFIX}87654321-dcba-4321-b321-987654321cba'

def test_fully_configured_secrets_example():
    """Test handling of fully configured secrets with all UUIDs."""
    yaml_str = f'\n    # Fully configured with all secrets as UUID handles\n    server:\n      bedrock:\n        api_key: "{UUID_PREFIX}12345678-abcd-1234-a123-123456789abc"\n        # User secret now has a UUID handle too\n        user_access_key: "{UUID_PREFIX}98765432-edcb-5432-c432-567890123def"\n      openai:\n        api_key: "{UUID_PREFIX}23456789-bcde-2345-b234-234567890bcd"\n    database:\n      password: "{UUID_PREFIX}87654321-dcba-4321-b321-987654321cba"\n    '
    loaded = load_yaml_with_secrets(yaml_str)
    assert loaded['server']['bedrock']['api_key'] == f'{UUID_PREFIX}12345678-abcd-1234-a123-123456789abc'
    assert loaded['server']['bedrock']['user_access_key'] == f'{UUID_PREFIX}98765432-edcb-5432-c432-567890123def'
    assert loaded['server']['openai']['api_key'] == f'{UUID_PREFIX}23456789-bcde-2345-b234-234567890bcd'
    assert loaded['database']['password'] == f'{UUID_PREFIX}87654321-dcba-4321-b321-987654321cba'
    for path in ['server.bedrock.api_key', 'server.bedrock.user_access_key', 'server.openai.api_key', 'database.password']:
        parts = path.split('.')
        value = loaded
        for part in parts:
            value = value[part]
        assert SECRET_ID_PATTERN.match(value) is not None

def test_materialize_creates_deployed_files(tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'super-secret')
    secrets_client = FakeSecretsClient()
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_config, deployed_secrets_path = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_123', config_file=config_file, deployed_secrets_path=deployed_secrets, secrets_client=secrets_client, non_interactive=True)
    assert deployed_config.exists()
    assert deployed_secrets_path.exists()
    saved = yaml.safe_load(deployed_secrets_path.read_text(encoding='utf-8'))
    assert 'env' in saved
    assert saved['env'][0]['OPENAI_API_KEY'].startswith('mcpac_sc_')
    assert secrets_client.created

def test_materialize_uses_fallback_value(tmp_path: Path):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text('env:\n  - {SUPABASE_URL: "https://example.com"}\n', encoding='utf-8')
    secrets_client = FakeSecretsClient()
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_456', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=secrets_client, non_interactive=True)
    saved = yaml.safe_load(deployed_secrets.read_text(encoding='utf-8'))
    assert saved['env'][0]['SUPABASE_URL'].startswith('mcpac_sc_')
    assert secrets_client.created['apps/app_456/env/SUPABASE_URL'] == 'https://example.com'

def test_materialize_reuses_existing_handles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text('env:\n  - OPENAI_API_KEY\n', encoding='utf-8')
    existing_handle = 'mcpac_sc_existing_handle'
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_secrets.write_text(yaml.safe_dump({'env': [{'OPENAI_API_KEY': existing_handle}]}), encoding='utf-8')

    class TrackingSecretsClient(FakeSecretsClient):

        async def create_secret(self, name, secret_type, value):
            raise AssertionError('Should reuse existing handle')
    client = TrackingSecretsClient()
    monkeypatch.setenv('OPENAI_API_KEY', 'fresh-secret')
    materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_789', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=client, non_interactive=True)
    assert client.updated[existing_handle] == 'fresh-secret'

def test_materialize_recovers_from_deleted_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text('env:\n  - OPENAI_API_KEY\n', encoding='utf-8')
    existing_handle = 'mcpac_sc_existing_handle'
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_secrets.write_text(yaml.safe_dump({'env': [{'OPENAI_API_KEY': existing_handle}]}), encoding='utf-8')

    class DeletedHandleClient(FakeSecretsClient):

        async def set_secret_value(self, handle, value):
            response = httpx.Response(status_code=404, request=httpx.Request('POST', 'https://example.com'), text='not found')
            raise httpx.HTTPStatusError('secret missing', request=response.request, response=response)
    client = DeletedHandleClient()
    monkeypatch.setenv('OPENAI_API_KEY', 'fresh-secret')
    _, secrets_path = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_recover', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=client, non_interactive=True)
    saved = yaml.safe_load(secrets_path.read_text(encoding='utf-8'))
    handle = saved['env'][0]['OPENAI_API_KEY']
    assert handle != existing_handle

def test_materialize_skips_invalid_config(tmp_path: Path):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text('invalid: [\n', encoding='utf-8')
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    client = FakeSecretsClient()
    deployed_config_path, secrets_out = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_invalid', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=client, non_interactive=True)
    assert deployed_config_path == cfg
    assert secrets_out.exists()
    assert yaml.safe_load(secrets_out.read_text(encoding='utf-8')) == {}

def test_materialize_prefers_app_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text('name: from-config\n', encoding='utf-8')
    module_name = 'main'
    main_path = tmp_path / f'{module_name}.py'
    main_path.write_text(textwrap.dedent('\n            from mcp_agent.app import MCPApp\n\n\n            app = MCPApp()\n            app.config.name = "from-app"\n            '), encoding='utf-8')
    secrets_client = FakeSecretsClient()
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_config_path, _ = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_appconfig', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=secrets_client, non_interactive=True)
    realized = yaml.safe_load(deployed_config_path.read_text(encoding='utf-8'))
    assert realized['name'] == 'from-app'

def test_deployed_config_redacts_secrets(tmp_path: Path):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text(textwrap.dedent('\n            openai:\n              api_key: "${oc.env:OPENAI_API_KEY}"\n              default_model: gpt-4o\n            '), encoding='utf-8')
    raw_secrets = tmp_path / 'mcp_agent.secrets.yaml'
    raw_secrets.write_text('openai:\n  api_key: sk-live\n', encoding='utf-8')
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_secrets.write_text(yaml.safe_dump({'openai': {'api_key': 'mcpac_sc_handle'}}), encoding='utf-8')
    secrets_client = FakeSecretsClient()
    deployed_config_path, _ = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_redact', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=secrets_client, non_interactive=True)
    realized = yaml.safe_load(deployed_config_path.read_text(encoding='utf-8'))
    assert realized['openai']['api_key'] == '${oc.env:OPENAI_API_KEY}'
    assert realized['openai']['default_model'] == 'gpt-4o'
    assert 'sk-live' not in deployed_config_path.read_text(encoding='utf-8')

def test_deployed_config_omits_secret_only_nodes(tmp_path: Path):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text('name: sample-app\n', encoding='utf-8')
    raw_secrets = tmp_path / 'mcp_agent.secrets.yaml'
    raw_secrets.write_text('notion:\n  api_key: top-secret\n', encoding='utf-8')
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_secrets.write_text(yaml.safe_dump({'notion': {'api_key': 'mcpac_sc_handle'}}), encoding='utf-8')
    secrets_client = FakeSecretsClient()
    deployed_config_path, _ = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_secret_nodes', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=secrets_client, non_interactive=True)
    realized = yaml.safe_load(deployed_config_path.read_text(encoding='utf-8'))
    assert 'notion' not in realized
    assert realized['name'] == 'sample-app'

def test_deployed_config_omits_secret_only_nested_env(tmp_path: Path):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text(textwrap.dedent('\n            name: sample-app\n            mcp:\n              servers:\n                fetch:\n                  command: uvx\n                  args: ["mcp-server-fetch"]\n            '), encoding='utf-8')
    raw_secrets = tmp_path / 'mcp_agent.secrets.yaml'
    raw_secrets.write_text(textwrap.dedent('\n            mcp:\n              servers:\n                slack:\n                  env:\n                    SLACK_BOT_TOKEN: token\n            '), encoding='utf-8')
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_secrets.write_text(yaml.safe_dump({'mcp': {'servers': {'slack': {'env': {'SLACK_BOT_TOKEN': 'mcpac_sc_handle'}}}}}), encoding='utf-8')
    secrets_client = FakeSecretsClient()
    deployed_config_path, _ = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_nested_env', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=secrets_client, non_interactive=True)
    realized = yaml.safe_load(deployed_config_path.read_text(encoding='utf-8'))
    servers = realized['mcp']['servers']
    assert 'slack' not in servers
    assert 'fetch' in servers

def test_deployed_config_preserves_env_declarations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text(textwrap.dedent('\n            env:\n              - OPENAI_API_KEY\n              - {SUPABASE_URL: "https://db.example.com"}\n            '), encoding='utf-8')
    monkeypatch.setenv('OPENAI_API_KEY', 'secret')
    monkeypatch.delenv('SUPABASE_URL', raising=False)
    secrets_client = FakeSecretsClient()
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_config_path, _ = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_env_preserve', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=secrets_client, non_interactive=True)
    realized = yaml.safe_load(deployed_config_path.read_text(encoding='utf-8'))
    assert realized['env'] == ['OPENAI_API_KEY', {'SUPABASE_URL': 'https://db.example.com'}]

def test_deployed_config_handles_anyhttpurl_fields(tmp_path: Path):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text(textwrap.dedent('\n            authorization:\n              enabled: true\n              issuer_url: https://idp.example.com/\n              resource_server_url: https://api.example.com/resource\n            '), encoding='utf-8')
    secrets_client = FakeSecretsClient()
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_config_path, _ = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_oauth', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=secrets_client, non_interactive=True)
    realized = yaml.safe_load(deployed_config_path.read_text(encoding='utf-8'))
    assert realized['authorization']['issuer_url'] == 'https://idp.example.com/'
    assert realized['authorization']['resource_server_url'] == 'https://api.example.com/resource'

def test_materialize_uses_app_config_when_available(tmp_path: Path, monkeypatch):
    cfg = tmp_path / 'mcp_agent.config.yaml'
    cfg.write_text('name: from-config\n', encoding='utf-8')
    main_py = tmp_path / 'main.py'
    main_py.write_text(textwrap.dedent('\n            from mcp_agent.app import MCPApp\n\n            app = MCPApp()\n            from mcp_agent.config import MCPAuthorizationServerSettings\n\n            app.config.authorization = MCPAuthorizationServerSettings(\n                enabled=True,\n                issuer_url="https://issuer.example.com",\n                resource_server_url="https://api.example.com",\n                expected_audiences=["example"],\n            )\n            '), encoding='utf-8')
    secrets_client = FakeSecretsClient()
    deployed_secrets = tmp_path / 'mcp_agent.deployed.secrets.yaml'
    deployed_config_path, _ = materialize_deployment_artifacts(config_dir=tmp_path, app_id='app_programmatic', config_file=cfg, deployed_secrets_path=deployed_secrets, secrets_client=secrets_client, non_interactive=True)
    realized = yaml.safe_load(deployed_config_path.read_text(encoding='utf-8'))
    assert realized['authorization']['issuer_url'] == 'https://issuer.example.com/'

