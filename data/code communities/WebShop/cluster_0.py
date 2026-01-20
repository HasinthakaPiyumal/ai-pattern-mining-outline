# Cluster 0

def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for k, v in d.items():
        logkv(k, v)

def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    Logger.CURRENT.logkv(key, val)

def debug(*args):
    log(*args, level=DEBUG)

def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    Logger.CURRENT.log(*args, level=level)

def info(*args):
    log(*args, level=INFO)

def warn(*args):
    log(*args, level=WARN)

def error(*args):
    log(*args, level=ERROR)

def configure(dir=None, format_strs=None):
    if dir is None:
        dir = os.getenv('OPENAI_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(), datetime.datetime.now().strftime('openai-%Y-%m-%d-%H-%M-%S-%f'))
    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)
    log_suffix = ''
    rank = 0
    for varname in ['PMI_RANK', 'OMPI_COMM_WORLD_RANK']:
        if varname in os.environ:
            rank = int(os.environ[varname])
    if rank > 0:
        log_suffix = '-rank%03i' % rank
    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv('OPENAI_LOG_FORMAT', 'stdout,log,csv').split(',')
        else:
            format_strs = os.getenv('OPENAI_LOG_FORMAT_MPI', 'log').split(',')
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]
    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)
    log('Logging to %s' % dir)

def make_output_format(format, ev_dir, log_suffix='', args=None):
    os.makedirs(ev_dir, exist_ok=True)
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        return HumanOutputFormat(osp.join(ev_dir, 'log%s.txt' % log_suffix))
    elif format == 'json':
        return JSONOutputFormat(osp.join(ev_dir, 'progress%s.json' % log_suffix))
    elif format == 'csv':
        return CSVOutputFormat(osp.join(ev_dir, 'progress%s.csv' % log_suffix))
    elif format == 'tensorboard':
        return TensorBoardOutputFormat(osp.join(ev_dir, 'tb%s' % log_suffix))
    elif format == 'wandb':
        return WandBOutputFormat(ev_dir)
    else:
        raise ValueError('Unknown format specified: %s' % (format,))

def _configure_default_logger():
    format_strs = None
    if 'OPENAI_LOG_FORMAT' not in os.environ:
        format_strs = ['stdout']
    configure(format_strs=format_strs)
    Logger.DEFAULT = Logger.CURRENT

def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')

class scoped_configure(object):

    def __init__(self, dir=None, format_strs=None):
        self.dir = dir
        self.format_strs = format_strs
        self.prevlogger = None

    def __enter__(self):
        self.prevlogger = Logger.CURRENT
        configure(dir=self.dir, format_strs=self.format_strs)

    def __exit__(self, *args):
        Logger.CURRENT.close()
        Logger.CURRENT = self.prevlogger

def _demo():
    info('hi')
    debug("shouldn't appear")
    set_level(DEBUG)
    debug('should appear')
    dir = '/tmp/testlogging'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    configure(dir=dir)
    logkv('a', 3)
    logkv('b', 2.5)
    dumpkvs()
    logkv('b', -2.5)
    logkv('a', 5.5)
    dumpkvs()
    info('^^^ should see a = 5.5')
    logkv_mean('b', -22.5)
    logkv_mean('b', -44.4)
    logkv('a', 5.5)
    dumpkvs()
    info('^^^ should see b = 33.3')
    logkv('b', -2.5)
    dumpkvs()
    logkv('a', 'longasslongasslongasslongasslongasslongassvalue')
    dumpkvs()

def set_level(level):
    """
    Set logging threshold on current logger.
    """
    Logger.CURRENT.set_level(level)

def dumpkvs():
    """
    Write all of the diagnostics from the current iteration

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.dumpkvs()

def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    Logger.CURRENT.logkv_mean(key, val)

