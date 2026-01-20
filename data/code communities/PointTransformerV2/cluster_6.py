# Cluster 6

def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_random_seed():
    seed = os.getpid() + int(datetime.now().strftime('%S%f')) + int.from_bytes(os.urandom(2), 'big')
    return seed

def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    worker_seed = num_workers * rank + worker_id + seed
    set_seed(worker_seed)

def main():
    args = get_parser()
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if cfg.seed is None:
        cfg.seed = get_random_seed()
    os.makedirs(cfg.save_path, exist_ok=True)
    set_seed(cfg.seed)
    cfg.batch_size_val_per_gpu = cfg.batch_size_test
    cfg.num_worker_per_gpu = cfg.num_worker
    weight_name = os.path.basename(cfg.weight).split('.')[0]
    logger = get_root_logger(log_file=os.path.join(cfg.save_path, 'test-{}.log'.format(weight_name)))
    logger.info('=> Loading config ...')
    logger.info(f'Save path: {cfg.save_path}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info('=> Building model ...')
    model = build_model(cfg.model).cuda()
    n_parameters = sum((p.numel() for p in model.parameters() if p.requires_grad))
    logger.info(f'Num params: {n_parameters}')
    logger.info('=> Building test dataset & dataloader ...')
    test_dataset = build_dataset(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size_val_per_gpu, shuffle=False, num_workers=cfg.num_worker_per_gpu, pin_memory=True, collate_fn=collate_fn)
    if os.path.isfile(cfg.weight):
        checkpoint = torch.load(cfg.weight)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded weight '{}' (epoch {})".format(cfg.weight, checkpoint['epoch']))
        cfg.epochs = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(cfg.weight))
    TEST.build(cfg.test)(cfg, test_loader, model)

def get_parser():
    parser = argparse.ArgumentParser(description='PCR Test Process')
    parser.add_argument('--config-file', default='', metavar='FILE', help='path to config file')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    args = parser.parse_args()
    return args

def build_model(cfg):
    """Build test_datasets."""
    return MODELS.build(cfg)

def build_dataset(cfg):
    """Build test_datasets."""
    return DATASETS.build(cfg)

