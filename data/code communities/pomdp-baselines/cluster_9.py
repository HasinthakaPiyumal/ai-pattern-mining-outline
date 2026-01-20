# Cluster 9

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

def configure(dir=None, format_strs=None, log_suffix='', precision=None):
    if dir is None:
        dir = os.getenv('OPENAI_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(), datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)
    if format_strs is None:
        strs = os.getenv('OPENAI_LOG_FORMAT')
        format_strs = strs.split(',') if strs else LOG_OUTPUT_FORMATS
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]
    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, precision=precision)
    log('Logging to %s' % dir)

def make_output_format(format, ev_dir, log_suffix=''):
    os.makedirs(ev_dir, exist_ok=True)
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        return HumanOutputFormat(osp.join(ev_dir, 'experiment%s.log' % log_suffix))
    elif format == 'json':
        return JSONOutputFormat(osp.join(ev_dir, 'progress.json'))
    elif format == 'csv':
        return CSVOutputFormat(osp.join(ev_dir, 'progress.csv'))
    elif format == 'tensorboard':
        return TensorBoardOutputFormat(ev_dir)
    else:
        raise ValueError('Unknown format specified: %s' % (format,))

