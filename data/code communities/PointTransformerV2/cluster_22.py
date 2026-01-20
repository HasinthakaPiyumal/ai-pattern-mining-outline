# Cluster 22

def main():
    overlaps = glob.glob(os.path.join(opt.target_dir, '*/pcd/overlap.txt'))
    with open(os.path.join(opt.target_dir, 'overlap30.txt'), 'w') as f:
        for fo in overlaps:
            for line in open(fo):
                pcd0, pcd1, op = line.strip().split()
                if float(op) >= 0.3:
                    print('{} {} {}'.format(pcd0, pcd1, op), file=f)
    print('done')

