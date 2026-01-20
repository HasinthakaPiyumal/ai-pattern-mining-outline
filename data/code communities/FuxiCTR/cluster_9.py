# Cluster 9

def seed_everything(seed=2019):
    logging.info('Setting random seed={}'.format(seed))
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

