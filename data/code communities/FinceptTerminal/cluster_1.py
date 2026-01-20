# Cluster 1

def determine_pip_install_arguments():
    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument('--no-setuptools', action='store_true')
    pre_parser.add_argument('--no-wheel', action='store_true')
    pre, args = pre_parser.parse_known_args()
    args.append('pip')
    if include_setuptools(pre):
        args.append('setuptools')
    if include_wheel(pre):
        args.append('wheel')
    return ['install', '--upgrade', '--force-reinstall'] + args

def include_setuptools(args):
    """
    Install setuptools only if absent, not excluded and when using Python <3.12.
    """
    cli = not args.no_setuptools
    env = not os.environ.get('PIP_NO_SETUPTOOLS')
    absent = not importlib.util.find_spec('setuptools')
    python_lt_3_12 = this_python < (3, 12)
    return cli and env and absent and python_lt_3_12

def include_wheel(args):
    """
    Install wheel only if absent, not excluded and when using Python <3.12.
    """
    cli = not args.no_wheel
    env = not os.environ.get('PIP_NO_WHEEL')
    absent = not importlib.util.find_spec('wheel')
    python_lt_3_12 = this_python < (3, 12)
    return cli and env and absent and python_lt_3_12

def bootstrap(tmpdir):
    monkeypatch_for_cert(tmpdir)
    from pip._internal.cli.main import main as pip_entry_point
    args = determine_pip_install_arguments()
    sys.exit(pip_entry_point(args))

def monkeypatch_for_cert(tmpdir):
    """Patches `pip install` to provide default certificate with the lowest priority.

    This ensures that the bundled certificates are used unless the user specifies a
    custom cert via any of pip's option passing mechanisms (config, env-var, CLI).

    A monkeypatch is the easiest way to achieve this, without messing too much with
    the rest of pip's internals.
    """
    from pip._internal.commands.install import InstallCommand
    cert_path = os.path.join(tmpdir, 'cacert.pem')
    with open(cert_path, 'wb') as cert:
        cert.write(pkgutil.get_data('pip._vendor.certifi', 'cacert.pem'))
    install_parse_args = InstallCommand.parse_args

    def cert_parse_args(self, args):
        if not self.parser.get_default_values().cert:
            self.parser.defaults['cert'] = cert_path
        return install_parse_args(self, args)
    InstallCommand.parse_args = cert_parse_args

def main():
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp()
        pip_zip = os.path.join(tmpdir, 'pip.zip')
        with open(pip_zip, 'wb') as fp:
            fp.write(b85decode(DATA.replace(b'\n', b'')))
        sys.path.insert(0, pip_zip)
        bootstrap(tmpdir=tmpdir)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

def normalize_country(country: str) -> Optional[str]:
    """Normalize country name to 2-letter ISO code"""
    country_lower = country.lower().strip()
    return COUNTRIES.get(country_lower, country.upper() if len(country) == 2 else None)

def normalize_indicator(indicator: str) -> Optional[str]:
    """Normalize indicator name to EconDB symbol"""
    indicator_lower = indicator.lower().strip()
    return INDICATORS.get(indicator_lower, indicator.upper())

def start_terminal():
    """OPTIMIZED: Console script entry point"""
    try:
        main()
    except Exception as e:
        print(f'[CRITICAL] Terminal startup failed: {e}')
        sys.exit(1)

def signal_handler(orchestrator):

    def handler(signum, frame):
        logging.info(f'Received signal {signum}')
        orchestrator.stop()
    return handler

