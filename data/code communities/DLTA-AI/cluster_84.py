# Cluster 84

def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    parser.add_argument('--selfsup', type=str, choices=['moco', 'swav'], help='save path')
    args = parser.parse_args()
    if args.selfsup == 'moco':
        moco_convert(args.src, args.dst)
    elif args.selfsup == 'swav':
        print('SWAV does not need to convert the keys')

def moco_convert(src, dst):
    """Convert keys in pycls pretrained moco models to mmdet style."""
    moco_model = torch.load(src)
    blobs = moco_model['state_dict']
    state_dict = OrderedDict()
    for k, v in blobs.items():
        if not k.startswith('module.encoder_q.'):
            continue
        old_k = k
        k = k.replace('module.encoder_q.', '')
        state_dict[k] = v
        print(old_k, '->', k)
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)

