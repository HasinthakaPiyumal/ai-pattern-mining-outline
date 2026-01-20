# Cluster 11

def add_log_file_handler():
    logger.add('sdgx-{time}.log', rotation='10 MB')

@click.option('--torchrun', type=bool, default=False, help='Use `torchrun` to run cli.')
@click.option('--torchrun_kwargs', type=str, default='{}', help='[Json String] torchrun kwargs.')
@wraps(func)
def wrapper(torchrun, torchrun_kwargs, *args, **kwargs):
    if not torchrun:
        func(*args, **kwargs)
    else:
        torchrun_kwargs = json.loads(torchrun_kwargs)
        torchrun_kwargs.setdefault('master_port', find_free_port())
        origin_args = copy.deepcopy(sys.argv)
        sys.argv = [re.sub('(-script\\.pyw|\\.exe)?$', '', sys.argv[0])]
        for k, v in torchrun_kwargs.items():
            sys.argv.extend([f'--{k}', str(v)])
        if '--torchrun' in origin_args:
            i = origin_args.index('--torchrun')
            if i + 1 < len(origin_args) and origin_args[i + 1] == 'true':
                origin_args.pop(i)
                origin_args.pop(i)
        if '--torchrun=true' in origin_args:
            origin_args.remove('--torchrun=true')
        sys.argv.extend(origin_args)
        sys.exit(load_entry_point('torch', 'console_scripts', 'torchrun')())

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def load_entry_point(distribution, group, name):
    dist_obj = importlib_metadata.distribution(distribution)
    eps = [ep for ep in dist_obj.entry_points if ep.group == group and ep.name == name]
    if not eps:
        raise ImportError('Entry point %r not found' % ((group, name),))
    return eps[0].load()

@pytest.mark.parametrize('json_output', [True, False])
@pytest.mark.parametrize('command', [list_cachers, list_data_connectors, list_data_processors, list_data_exporters, list_models])
def test_list_extension_api(command, json_output):
    runner = CliRunner()
    result = runner.invoke(command, ['--json_output', json_output])
    assert result.exit_code == 0
    if json_output:
        assert NormalMessage()._dump_json() in result.output
        assert NormalMessage()._dump_json() == result.output.strip().split('\n')[-1]
    else:
        assert NormalMessage()._dump_json() not in result.output

@pytest.mark.parametrize('exception_caller', [unknown_exception, sdgx_exception])
def test_exception_message(exception_caller):
    try:
        exception_caller()
    except Exception as e:
        msg = ExceptionMessage.from_exception(e)
        assert msg._dump_json()
        assert msg.code != 0
        assert msg.payload
        assert 'details' in msg.payload

