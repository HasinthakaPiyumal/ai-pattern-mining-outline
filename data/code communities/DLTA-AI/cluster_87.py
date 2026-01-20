# Cluster 87

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args

def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser('plot_curve', help='parser for plotting curves')
    parser_plt.add_argument('json_logs', type=str, nargs='+', help='path of train log in json format')
    parser_plt.add_argument('--keys', type=str, nargs='+', default=['bbox_mAP'], help='the metric that you want to plot')
    parser_plt.add_argument('--start-epoch', type=str, default='1', help='the epoch that you want to start')
    parser_plt.add_argument('--eval-interval', type=str, default='1', help='the eval interval when training')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument('--legend', type=str, nargs='+', default=None, help='legend of each plot')
    parser_plt.add_argument('--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument('--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)

def add_time_parser(subparsers):
    parser_time = subparsers.add_parser('cal_train_time', help='parser for computing the average time per training iteration')
    parser_time.add_argument('json_logs', type=str, nargs='+', help='path of train log in json format')
    parser_time.add_argument('--include-outliers', action='store_true', help='include the first value of every epoch when computing the average time')

