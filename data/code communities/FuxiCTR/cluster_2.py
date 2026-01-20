# Cluster 2

def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s P%(process)d %(levelname)s %(message)s', handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
    logging.info('FuxiCTR version: ' + fuxictr.__version__)

