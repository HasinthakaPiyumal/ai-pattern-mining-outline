# Cluster 20

def load_bev_params(param):
    """
    Load bev related geometry parameters s.t. boundary, resolutions, input
    shape, target shape etc.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `geometry_param`.

    """
    res = param['preprocess']['args']['res']
    L1, W1, H1, L2, W2, H2 = param['preprocess']['cav_lidar_range']
    downsample_rate = param['preprocess']['args']['downsample_rate']

    def f(low, high, r):
        return int((high - low) / r)
    input_shape = (int(f(L1, L2, res)), int(f(W1, W2, res)), int(f(H1, H2, res) + 1))
    label_shape = (int(input_shape[0] / downsample_rate), int(input_shape[1] / downsample_rate), 7)
    geometry_param = {'L1': L1, 'L2': L2, 'W1': W1, 'W2': W2, 'H1': H1, 'H2': H2, 'downsample_rate': downsample_rate, 'input_shape': input_shape, 'label_shape': label_shape, 'res': res}
    param['preprocess']['geometry_param'] = geometry_param
    param['postprocess']['geometry_param'] = geometry_param
    param['model']['args']['geometry_param'] = geometry_param
    return param

def f(low, high, r):
    return int((high - low) / r)

