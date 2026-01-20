# Cluster 66

def _agent_spec_from_dict(obj: dict, context: Context | None=None, *, default_instruction: str | None=None) -> AgentSpec:
    name = obj.get('name')
    if not name:
        raise ValueError("AgentSpec requires a 'name'")
    instruction = obj.get('instruction')
    if not instruction:
        desc = obj.get('description')
        if default_instruction and desc:
            instruction = f'{desc}\n\n{default_instruction}'.strip()
        else:
            instruction = default_instruction or desc
    server_names = obj.get('server_names') or obj.get('servers') or []
    connection_persistence = obj.get('connection_persistence', True)
    functions = obj.get('functions', [])
    if not server_names and 'tools' in obj:
        tools_val = obj.get('tools')
        if isinstance(tools_val, str):
            server_names = [t.strip() for t in tools_val.split(',') if t.strip()]
        elif isinstance(tools_val, list):
            server_names = [str(t).strip() for t in tools_val if str(t).strip()]
    resolved_functions: list[Callable] = []
    for f in functions:
        if callable(f):
            resolved_functions.append(f)
        elif isinstance(f, str):
            resolved_functions.append(_resolve_callable(f))
        else:
            raise ValueError(f'Unsupported function entry: {f}')
    human_cb = obj.get('human_input_callback')
    if isinstance(human_cb, str):
        human_cb = _resolve_callable(human_cb)
    return AgentSpec(name=name, instruction=instruction, server_names=list(server_names), functions=resolved_functions, connection_persistence=connection_persistence, human_input_callback=human_cb)

def _resolve_callable(ref: str) -> Callable:
    """Resolve a dotted reference 'package.module:attr' to a callable.
    Raises ValueError if not found or not callable.
    """
    if not isinstance(ref, str) or (':' not in ref and '.' not in ref):
        raise ValueError(f'Invalid callable reference: {ref}')
    module_name, attr = ref.split(':', 1) if ':' in ref else ref.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    obj = getattr(mod, attr)
    if not callable(obj):
        raise ValueError(f'Referenced object is not callable: {ref}')
    return obj

def load_agent_specs_from_text(text: str, *, fmt: str | None=None, context: Context | None=None) -> List[AgentSpec]:
    """Load AgentSpec list from text in yaml/json/md.

    - YAML: either a list or {'agents': [...]}
    - JSON: same as YAML
    - Markdown: supports YAML front-matter or fenced code blocks with yaml/json containing agents
    """
    specs: list[AgentSpec] = []
    fmt_lower = (fmt or '').lower()
    try_parsers = []
    if fmt_lower in ('yaml', 'yml'):
        try_parsers = [lambda t: _load_yaml(t)]
    elif fmt_lower == 'json':
        try_parsers = [lambda t: json.loads(t)]
    elif fmt_lower == 'md':
        fm, body = _extract_front_matter_and_body_md(text)
        if fm is not None:
            try_parsers.append(lambda _t, fm=fm: ('__FM__', _load_yaml(fm), body))
        for lang, code in _extract_code_blocks_md(text):
            lang = (lang or '').lower()
            if lang in ('yaml', 'yml'):
                try_parsers.append(lambda _t, code=code: ('__YAML__', _load_yaml(code), ''))
            elif lang == 'json':
                try_parsers.append(lambda _t, code=code: ('__JSON__', json.loads(code), ''))
    else:
        try_parsers = [lambda t: _load_yaml(t), lambda t: json.loads(t)]
    for parser in try_parsers:
        try:
            data = parser(text)
        except Exception:
            continue
        body_text: str | None = None
        if isinstance(data, tuple) and len(data) == 3 and isinstance(data[1], (dict, list)):
            _, parsed, body_text = data
            data = parsed
        agents_data = _normalize_agents_data(data)
        for obj in agents_data:
            try:
                specs.append(_agent_spec_from_dict(obj, context=context, default_instruction=body_text))
            except Exception:
                continue
        if specs:
            break
    return specs

def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except Exception as e:
        raise ImportError('PyYAML is required to load YAML agent specs') from e
    return yaml.safe_load(text)

def _extract_front_matter_and_body_md(text: str) -> tuple[str | None, str]:
    """Return (front_matter_yaml, body_text).

    Allows leading whitespace/BOM before front matter.
    """
    s = text.lstrip('\ufeff\r\n \t')
    if s.startswith('---\n'):
        end = s.find('\n---', 4)
        if end != -1:
            fm = s[4:end]
            body = s[end + len('\n---'):].lstrip('\n')
            return (fm, body)
    return (None, text)

def _extract_code_blocks_md(text: str) -> list[tuple[str, str]]:
    """Return list of (lang, code) for fenced code blocks.

    Relaxed to allow attributes after language, e.g. ```yaml title="...".
    """
    pattern = re.compile('```\\s*([A-Za-z0-9_-]+)(?:[^\\n]*)?\\n([\\s\\S]*?)```', re.MULTILINE)
    return [(m.group(1) or '', m.group(2)) for m in pattern.finditer(text)]

def _normalize_agents_data(data: Any) -> list[dict]:
    """Normalize arbitrary parsed data into a list of agent dicts.

    Accepts:
      - {'agents': [...]} or {'agent': {...}} or a list of agents or a single agent dict
    """
    if data is None:
        return []
    if isinstance(data, dict):
        if 'agents' in data and isinstance(data['agents'], list):
            return data['agents']
        if 'agent' in data and isinstance(data['agent'], dict):
            return [data['agent']]
        if 'name' in data:
            return [data]
        return []
    if isinstance(data, list):
        return data
    return []

def test_yaml_agents_list_parses_agentspecs(tmp_path):
    yaml_text = dedent('\n        agents:\n          - name: finder\n            instruction: You can read files\n            server_names: [filesystem]\n          - name: fetcher\n            servers: [fetch]\n            instruction: You can fetch URLs\n        ')
    specs = load_agent_specs_from_text(yaml_text, fmt='yaml')
    assert len(specs) == 2
    assert isinstance(specs[0], AgentSpec)
    assert specs[0].name == 'finder'
    assert specs[0].instruction == 'You can read files'
    assert specs[0].server_names == ['filesystem']
    assert specs[1].name == 'fetcher'
    assert specs[1].server_names == ['fetch']

def test_json_single_agent_object(tmp_path):
    json_text = dedent('\n        {"agent": {"name": "coder", "instruction": "Modify code", "servers": ["filesystem"]}}\n        ')
    specs = load_agent_specs_from_text(json_text, fmt='json')
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == 'coder'
    assert spec.instruction == 'Modify code'
    assert spec.server_names == ['filesystem']

def test_markdown_front_matter_and_body_merges_instruction():
    md_text = dedent('\n        ---\n        name: code-reviewer\n        description: Expert code reviewer, use proactively\n        tools: filesystem, fetch\n        ---\n\n        You are a senior code reviewer ensuring high standards.\n\n        Provide feedback organized by priority.\n        ')
    specs = load_agent_specs_from_text(md_text, fmt='md')
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == 'code-reviewer'
    assert 'Expert code reviewer' in (spec.instruction or '')
    assert 'senior code reviewer' in (spec.instruction or '')
    assert spec.server_names == ['filesystem', 'fetch']

def test_markdown_code_blocks_yaml_and_json():
    md_text = dedent('\n        Here are some agents:\n\n        ```yaml\n        agents:\n          - name: a\n            servers: [filesystem]\n        ```\n\n        And some JSON:\n\n        ```json\n        {"agent": {"name": "b", "servers": ["fetch"]}}\n        ```\n        ')
    specs = load_agent_specs_from_text(md_text, fmt='md')
    assert any((s.name == 'a' for s in specs)) or any((s.name == 'b' for s in specs))

def test_functions_resolution_with_dotted_ref(tmp_path, monkeypatch):
    yaml_text = dedent('\n        agents:\n          - name: tools-agent\n            servers: [filesystem]\n            functions:\n              - "tests.workflows.test_agentspec_loader:sample_fn"\n        ')
    specs = load_agent_specs_from_text(yaml_text, fmt='yaml')
    assert len(specs) == 1
    spec = specs[0]
    assert len(spec.functions) == 1
    assert spec.functions[0]() == 'ok'

