# Cluster 6

def vis_parser():
    parser = argparse.ArgumentParser(description='data visualization')
    parser.add_argument('--color_mode', type=str, default='intensity', help='lidar color rendering mode, e.g. intensity,z-value or constant.')
    opt = parser.parse_args()
    return opt

