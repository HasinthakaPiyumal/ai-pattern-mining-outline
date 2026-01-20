# Cluster 10

def str2bool(value: Union[str, bool]) -> bool:
    """
    Convert logical string to boolean. Can be used as argparse type.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

