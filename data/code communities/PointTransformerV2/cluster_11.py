# Cluster 11

def launch(main_func, num_gpus_per_machine, num_machines=1, machine_rank=0, dist_url=None, cfg=(), timeout=DEFAULT_TIMEOUT):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url == 'auto':
            assert num_machines == 1, 'dist_url=auto not supported in multi-machine jobs.'
            port = _find_free_port()
            dist_url = f'tcp://127.0.0.1:{port}'
        if num_machines > 1 and dist_url.startswith('file://'):
            logger = logging.getLogger(__name__)
            logger.warning('file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://')
        mp.spawn(_distributed_worker, nprocs=num_gpus_per_machine, args=(main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, cfg, timeout), daemon=False)
    else:
        main_func(*cfg)

def _find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def default_config_parser(file_path, options):
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find('-')
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1:]))
    if options is not None:
        cfg.merge_from_dict(options)
    if cfg.seed is None:
        cfg.seed = get_random_seed()
    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch
    os.makedirs(os.path.join(cfg.save_path, 'model'), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, 'config.py'))
    return cfg

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    launch(main_worker, num_gpus_per_machine=args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url, cfg=(cfg,))

def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(epilog=epilog or f'\n    Examples:\n    Run on single machine:\n        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml\n    Change some config options:\n        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001\n    Run on multiple machines:\n        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]\n        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]\n    ', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-file', default='', metavar='FILE', help='path to config file')
    parser.add_argument('--num-gpus', type=int, default=1, help='number of gpus *per machine*')
    parser.add_argument('--num-machines', type=int, default=1, help='total number of machines')
    parser.add_argument('--machine-rank', type=int, default=0, help='the rank of this machine (unique per machine)')
    parser.add_argument('--dist-url', default='auto', help='initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    return parser

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    launch(main_worker, num_gpus_per_machine=args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url, cfg=(cfg,))

