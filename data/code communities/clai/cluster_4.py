# Cluster 4

def remove_system_folder(path: str=None):
    if path is not None:
        default_system_destdir = '/'.join(path.split('/')[0:-1])
    else:
        default_system_destdir = os.path.join(os.path.expanduser('/opt/local/share'), 'clai')
    remove(default_system_destdir)

def remove(path):
    print('cleaning %s' % path)
    try:
        remove_tree(path)
    except OSError:
        print('folder not found')

def unregister_the_user(bin_path):
    users = read_users(bin_path)
    user_path = os.path.expanduser(get_rc_file())
    if user_path in users:
        users.remove(user_path)
    with open(bin_path + '/usersInstalled.json', 'w') as json_file:
        json.dump(users, json_file)
    return users

def read_users(bin_path):
    try:
        with open(bin_path + '/usersInstalled.json') as file:
            users = json.load(file)
            return users
    except:
        return []

def get_rc_file(system=False):
    if system:
        return '/etc/profile'
    return '~/.bashrc'

def remove_between(rc_file_path, start, end):
    path = os.path.expanduser(rc_file_path)
    if os.path.isfile(path):
        codeset = 'utf-8'
        lines = []
        if is_rw_with_EBCDIC(path):
            codeset = 'cp1047'
            newline = '\x85'
            for err_line in open(path, 'r', encoding=codeset, errors='ignore').readlines():
                right_line_list = err_line.split(newline)
                length = len(right_line_list)
                i = 0
                for right_line in right_line_list:
                    i = i + 1
                    if right_line or length != i:
                        lines.append(right_line + newline)
        else:
            lines = open(path, 'r', encoding=codeset, errors='ignore').readlines()
        remove_line = False
        lines_after_remove = []
        for line in lines:
            if line.strip() == start.strip():
                remove_line = True
            if not remove_line:
                lines_after_remove.append(line)
            if line.strip() == end.strip():
                remove_line = False
        io.open(path, 'w', encoding=codeset).writelines(lines_after_remove)

def is_rw_with_EBCDIC(file):
    is_EBCDIC = False
    file_path_complete = os.path.expanduser(file)
    if is_zos():
        if not os.path.exists(file_path_complete):
            is_EBCDIC = True
        else:
            cmd = 'chtag -p ' + file_path_complete + " | cut -d ' ' -f 2,6 "
            pfile = os.popen(cmd, 'r')
            output = pfile.read().strip()
            if output in ('untagged T=off', 'IBM-1047 T=off'):
                is_EBCDIC = True
            pfile.close()
    return is_EBCDIC

def remove_lines_setup(rc_file_path):
    remove_between(rc_file_path, '# CLAI setup\n', '# End CLAI setup\n')

def remove_setup_register():
    rc_files = get_rc_files()
    for file in rc_files:
        remove_lines_setup(file)

def get_rc_files(system=False):
    if system:
        return ['/etc/profile']
    return ['~/.bash_profile', '~/.bashrc']

def is_user_install(bin_path):
    plugins_config = ConfigStorage(alternate_path=f'{bin_path}/configPlugins.json').read_config(None).user_install
    return plugins_config

def execute(args):
    bin_path = os.getenv('CLAI_PATH', None)
    if '-h' in args or '--help' in args:
        print('usage: uninstall.py [-h] [--help] [--user]\n             \nUninstall CLAI.\n             \noptional arguments:             \n-h, --help            show this help message and exit             \n--user                Uninstalls a local installation of clai for the current user')
        sys.exit(0)
    path = clai_installed(get_setup_file())
    print(f'path= {path}')
    print(f'bin path= {bin_path}')
    if not path and bin_path is None:
        print_error('CLAI is not installed.')
        sys.exit(1)
    if not bin_path:
        bin_path = path
    stat_uninstall(path)
    users = unregister_the_user(path)
    if '--user' in args or is_user_install(bin_path):
        remove_system_folder(bin_path)
    elif not users:
        remove_system_folder()
    remove_setup_file(get_setup_file())
    remove_setup_register()
    print_complete('CLAI has been uninstalled correctly, you will need to restart your shell.')
    return 0

def clai_installed(rc_file_path):
    path = os.path.expanduser(rc_file_path)
    if os.path.isfile(path):
        line_to_search = 'export CLAI_PATH='
        print('searching %s' % line_to_search)
        lines = io.open(path, 'r', encoding='utf-8', errors='ignore').readlines()
        lines_found = list(filter(lambda line: line_to_search in line, lines))
        if lines_found:
            my_path = lines_found[0].replace(line_to_search, '').replace('\n', '').strip()
            return my_path
    return None

def get_setup_file():
    return '~/.clairc'

def print_error(text):
    print(Colorize().warning().append(text).to_console())

def remove_setup_file(rc_file_path):
    path = os.path.expanduser(rc_file_path)
    os.remove(path)

def print_complete(text):
    print(Colorize().complete().append(text).to_console())

def parse_args():
    default_user_destdir = os.path.join(os.path.expanduser('/opt/local/share'), 'clai')
    parser = argparse.ArgumentParser(description='Install CLAI for all users.')
    parser.add_argument('--shell', help='if you like to install for different shell', dest='shell', action='store')
    parser.add_argument('--demo', help='if you like to jump installation restrictions', dest='demo_mode', action='store_true')
    parser.add_argument('--system', help='if you like install it for all users.', dest='system', action='store_true', default=False)
    parser.add_argument('--unassisted', help="Don't ask to he user for questions or inputs in the install process", dest='unassisted', action='store_true', default=False)
    parser.add_argument('--no-skills', help="Don't install the default skills", dest='no_skills', action='store_true', default=False)
    parser.add_argument('-d', '--destdir', metavar='DIR', default=default_user_destdir, help='set destination to DIR')
    parser.add_argument('--user', help='Installs clai in the users own bin directory', dest='user_install', action='store_true', default=False)
    parser.add_argument('--port', help='port listen server', type=int, default=8010, dest='port')
    args = parser.parse_args()
    if not valid_python_version():
        print_error('You need install python 3.6 or upper is required.')
        sys.exit(1)
    if not is_root_user(args):
        if not args.user_install:
            print_error('You need root privileges for the system wide installation process.')
            sys.exit(1)
    if args.user_install:
        if args.destdir == default_user_destdir:
            args.destdir = os.path.join(os.path.expanduser('~/.bin'), 'clai')
    if is_windows():
        print_error('CLAI is not supported on Windows.')
        sys.exit(1)
    shell = get_shell(args)
    if shell not in SUPPORTED_SHELLS:
        print_error('%s is not supported yet.' % shell)
        sys.exit(1)
    if args.system:
        if args.destdir != default_user_destdir:
            print_error('Custom paths incompatible with --system option.')
            sys.exit(1)
    return args

def valid_python_version():
    return sys.version_info[0] == 3 and sys.version_info[1] >= 6

def is_root_user():
    return os.geteuid() == 0

def is_windows():
    return platform.system() == 'Windows'

def get_shell(args):
    if args.shell is None:
        return os.path.basename(os.getenv('SHELL', ''))
    return args.shell

def cp_tree(from_path, to_path):
    print('copying folder from %s to %s' % (from_path, to_path))
    copy_tree(from_path, to_path)
    if is_zos():
        recursively_tag_untagged_files_as_ebcdic(to_path)

def is_zos():
    return platform.system().lower() in ('z/os', 'os/390')

def install_plugins(install_path, user_install):
    agent_datasource = AgentDatasource(config_storage=ConfigStorage(alternate_path=f'{install_path}/configPlugins.json'))
    plugins = agent_datasource.all_plugins()
    for plugin in plugins:
        default = z_default = False
        if PLATFORM in ('zos', 'os390'):
            z_default = plugin.z_default
        else:
            default = plugin.default
        if default or z_default:
            installed = install_plugins_dependencies(install_path, plugin.pkg_name, user_install)
            if installed:
                agent_datasource.mark_plugins_as_installed(plugin.name, None)
    return agent_datasource

def register_the_user(bin_path, system):
    users = read_users(bin_path)
    user_path = os.path.expanduser(get_rc_file(system))
    if user_path not in users:
        users.append(user_path)
    with open(bin_path + '/usersInstalled.json', 'w') as json_file:
        json.dump(users, json_file)

def create_rc_file_if_not_exist(system):
    rc_file_path = os.path.expanduser(get_rc_file(system))
    if not os.path.isfile(rc_file_path):
        open(rc_file_path, 'a').close()

def mark_user_flag(bin_path: str, value: bool):
    config_storage = ConfigStorage(alternate_path=f'{bin_path}/configPlugins.json')
    plugins_config = config_storage.read_config(None)
    plugins_config.user_install = value
    config_storage.store_config(plugins_config, None)

def execute(args):
    unassisted = args.unassisted
    no_skills = args.no_skills
    demo_mode = args.demo_mode
    user_install = args.user_install
    bin_path = os.path.join(args.destdir, 'bin')
    code_path = os.path.join(bin_path, 'clai')
    cli_path = os.path.join(bin_path, 'bin')
    temp_path = './tmp'
    mkdir(f'{temp_path}/')
    create_rc_file_if_not_exist(args.system)
    if clai_installed(get_setup_file()):
        print_error('CLAI is already in you system. You should execute uninstall first')
        sys.exit(1)
    if not binary_installed(bin_path):
        mkdir(bin_path)
        mkdir(code_path)
        cp_tree('./clai', code_path)
        cp_tree('./bin', cli_path)
        copy('./scripts/clai.sh', bin_path)
        copy('./scripts/saveFilesChanges.sh', bin_path)
        copy('./configPlugins.json', bin_path)
        copy('./usersInstalled.json', bin_path)
        copy('./anonymize.json', bin_path)
        copy('./scripts/fileExist.sh', bin_path)
        copy('./scripts/installOrchestrator.sh', bin_path)
        os.system(f'chmod 775 {bin_path}/saveFilesChanges.sh')
        os.system(f'chmod 775 {bin_path}/fileExist.sh')
        os.system(f'chmod 775 {bin_path}/installOrchestrator.sh')
        os.system(f'chmod -R 777 {code_path}/server/plugins')
        os.system(f'chmod 777 {bin_path}/clai.sh')
        os.system(f'chmod 666 {bin_path}/configPlugins.json')
        os.system(f'chmod 666 {bin_path}/anonymize.json')
        os.system(f'chmod -R 777 {bin_path}')
        cli_executable(cli_path)
        download_file(URL_BASH_PREEXEC, filename='%s/%s' % (temp_path, BASH_PREEXEC))
        copy('%s/%s' % (temp_path, BASH_PREEXEC), bin_path)
        if is_zos():
            os.system(f'chtag -tc 819 {bin_path}/{BASH_PREEXEC}')
    if user_install:
        mark_user_flag(bin_path, True)
    else:
        mark_user_flag(bin_path, False)
    register_the_user(bin_path, args.system)
    append_setup_to_file(get_setup_file(), bin_path, args.port)
    register_file(args.system)
    install_orchestration(bin_path)
    if not no_skills:
        save_report_info(unassisted, install_plugins(bin_path, user_install), bin_path, demo_mode)
    remove(f'{temp_path}')
    if not user_install:
        os.system(f'chmod -R 777 /var/tmp')
    print_complete('CLAI has been installed correctly, you will need to restart your shell.')

def mkdir(path):
    print('creating directory: %s' % path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def binary_installed(path):
    return os.path.exists(path)

def copy(file, to_path):
    print('copying file from %s to %s' % (file, to_path))
    copy_file(file, to_path, update=1)

def cli_executable(cli_path):
    os.system(f'chmod 777 {cli_path}/clai-run')
    os.system(f'chmod 777 {cli_path}/fswatchlog')
    os.system(f'chmod 777 {cli_path}/obtain-command-id')
    os.system(f'chmod 777 {cli_path}/post-execution')
    os.system(f'chmod 777 {cli_path}/process-command')
    os.system(f'chmod 777 {cli_path}/restore-history')

def download_file(file_url, filename):
    print('Download %s' % file_url)
    from urllib import request
    ssl._create_default_https_context = ssl._create_unverified_context
    request.urlretrieve(file_url, filename=filename)

def register_file(system):
    rc_files = get_rc_files(system)
    for file in rc_files:
        encoding = 'utf-8'
        newline = '\n'
        left_bracket = '['
        right_bracket = ']'
        if is_rw_with_EBCDIC(file):
            encoding = 'cp1047'
            left_bracket = 'Ý'
            right_bracket = '¨'
        print(f'registering {file}')
        append_to_file(file, '# CLAI setup' + newline, encoding)
        append_to_file(file, 'if ! ' + left_bracket + ' ${#preexec_functions' + left_bracket + '@' + right_bracket + '} -eq 0 ' + right_bracket + '; then' + newline, encoding)
        append_to_file(file, '  if ! ' + left_bracket + left_bracket + ' " ${preexec_functions' + left_bracket + '@' + right_bracket + '} " =~ " preexec_override_invoke " ' + right_bracket + right_bracket + '; then' + newline, encoding)
        append_to_file(file, f'     source {get_setup_file()} ' + newline, encoding)
        append_to_file(file, '  fi' + newline, encoding)
        append_to_file(file, 'else' + newline, encoding)
        append_to_file(file, f' source {get_setup_file()} ' + newline, encoding)
        append_to_file(file, 'fi' + newline, encoding)
        append_to_file(file, '# End CLAI setup' + newline, encoding)

def append_to_file(file_path, value_to_append, codeset='utf-8'):
    file_path_complete = os.path.expanduser(file_path)
    print('append to file %s' % file_path_complete)
    with open(os.path.expanduser(file_path_complete), 'a+', encoding=codeset) as file:
        file.write(value_to_append)

def append_setup_to_file(rc_path, bin_path, port):
    append_to_file(rc_path, '\n export CLAI_PATH=%s' % bin_path)
    append_to_file(rc_path, '\n export CLAI_PORT=%s' % port)
    append_to_file(rc_path, '\n export PYTHONPATH=%s' % bin_path)
    append_to_file(rc_path, '\n[[ -f %s/bash-preexec.sh ]] && source %s/bash-preexec.sh' % (bin_path, bin_path))
    append_to_file(rc_path, f'\n[[ -f {bin_path}/clai.sh ]] && source {bin_path}/clai.sh --port {port}')

def parse_args():
    parser = argparse.ArgumentParser(description='Setup Clai in a development enviorment to make live changes')
    parser.add_argument('action', action='store', type=str, help=f'action for script to perform one of: {' '.join(ACTIONS)}')
    parser.add_argument('-p', '--path', help='path to source directory', dest='path', action='store')
    parser.add_argument('-i', '--install-directory', dest='install_path', action='store', type=str, help='The location that clai is installed in', default=f'{os.getenv('HOME', '/home/root')}/.bin/clai/bin')
    args = parser.parse_args()
    if args.action not in ACTIONS:
        print_error(f"Not a valid action: '{args.action}' Valid actions: [{', '.join(ACTIONS)}]")
        sys.exit(1)
    if args.path is None:
        print_error('The path flag is required')
        sys.exit(1)
    return args

def install(repo_path: str, install_path: str):
    createInstallDir(install_path)
    required_scripts = os.listdir(os.path.join(repo_path, 'scripts'))
    required_dirs = ['bin', 'clai']
    required_files = [file for file in os.listdir(repo_path) if file.endswith('.json')]
    print('Linking all needed files to install directory')
    try:
        for script in required_scripts:
            link(os.path.join(repo_path, f'scripts/{script}'), os.path.join(install_path, script))
            os.system(f'chmod 775 {os.path.join(install_path, script)}')
        for directory in required_dirs:
            link(os.path.join(repo_path, directory), os.path.join(install_path, directory))
            if directory == 'bin':
                os.system(f'chmod -R 777 {os.path.join(install_path, directory)}')
        for file in required_files:
            link(os.path.join(repo_path, file), os.path.join(install_path, file))
            os.system(f'chmod 666 {os.path.join(install_path, file)}')
    except Exception as e:
        print(e)
        sys.exit(1)
    download_file(URL_BASH_PREEXEC, filename='%s/%s' % (install_path, BASH_PREEXEC))
    register_the_user(install_path, False)
    append_setup_to_file(get_setup_file(), install_path, DEFAULT_PORT)
    register_file(False)
    install_orchestration(install_path)
    install_plugins(install_path, False)
    print_complete('CLAI has been installed correctly, you need restart your shell.')

def createInstallDir(directory):
    try:
        if not os.path.exists(directory):
            print(f'creating install directory: {directory}')
            os.makedirs(directory, exist_ok=True)
        else:
            print(f'Install directory already exists')
    except Exception as e:
        print(e)
        sys.exit(1)

def link(src, dest):
    try:
        if not os.path.exists(dest):
            os.symlink(src, dest)
    except Exception as e:
        print(e)
        sys.exit(1)

def main(args: list):
    action = args.action
    repo_path = args.path
    install_path = args.install_path
    if action == 'install':
        install(repo_path, install_path)
    elif action == 'uninstall':
        path = clai_installed(get_setup_file())
        if not path:
            print_error('CLAI is not installed.')
            sys.exit(1)
        revert(install_path)
        uninstall(['--user'])
        with open(f'{repo_path}/configPlugins.json', 'w') as file:
            file.write(json.dumps({'selected': {'user': ['']}, 'default': [''], 'default_orchestrator': 'max_orchestrator', 'installed': [], 'report_enable': False}))
    return 0

def clai_installed(path):
    expand_path = os.path.expanduser(path)
    return os.path.isfile(expand_path)

def revert(install_path):
    print('Reverting file permissions to original state')
    scripts = [file for file in os.listdir(install_path) if file.endswith('.sh')]
    json_files = [file for file in os.listdir(install_path) if file.endswith('.json')]
    try:
        for script in scripts:
            os.system(f'chmod 644 {os.path.join(install_path, script)}')
        os.system(f'chmod 755 {install_path}/bin')
        for file in os.listdir(f'{install_path}/bin'):
            if file in ['emulator.py', 'process-command']:
                os.system(f'chmod 755 {os.path.join(install_path, f'bin/{file}')}')
            else:
                os.system(f'chmod 644 {os.path.join(install_path, f'bin/{file}')}')
        for file in json_files:
            os.system(f'chmod 666 {os.path.join(install_path, file)}')
    except Exception as e:
        print(e)
        sys.exit(1)

