# Cluster 0

def get_git_hash():

    def _minimal_ext_cmd(cmd):
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'
    return sha

def _minimal_ext_cmd(cmd):
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out

def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from basicsr.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'
    return sha

def write_version_py():
    content = "# GENERATED VERSION FILE\n# TIME: {}\n__version__ = '{}'\nshort_version = '{}'\nversion_info = ({})\n"
    sha = get_hash()
    with open('VERSION', 'r') as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ', '.join([x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')])
    VERSION = SHORT_VERSION + '+' + sha
    version_file_str = content.format(time.asctime(), VERSION, SHORT_VERSION, VERSION_INFO)
    with open(version_file, 'w') as f:
        f.write(version_file_str)

def make_cuda_ext(name, module, sources, sources_cuda=None):
    if sources_cuda is None:
        sources_cuda = []
    define_macros = []
    extra_compile_args = {'cxx': []}
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = ['-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__']
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
    return extension(name=f'{module}.{name}', sources=[os.path.join(*module.split('.'), p) for p in sources], define_macros=define_macros, extra_compile_args=extra_compile_args)

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def readme():
    return ''

def get_requirements(filename='requirements.txt'):
    return []
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

