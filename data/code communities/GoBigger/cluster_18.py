# Cluster 18

def read_config(path: str) -> EasyDict:
    """
    Overview:
        read configuration from path
    Arguments:
        - path (:obj:`str`): Path of source yaml
    Returns:
        - (:obj:`EasyDict`): Config data from this file with dict type
    """
    if path:
        assert os.path.exists(path), path
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return EasyDict(config)

