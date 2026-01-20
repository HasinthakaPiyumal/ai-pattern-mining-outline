# Cluster 82

def convert(src, dst, depth):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    if depth not in arch_settings:
        raise ValueError('Only support ResNet-50 and ResNet-101 currently')
    block_nums = arch_settings[depth]
    caffe_model = mmcv.load(src, encoding='latin1')
    blobs = caffe_model['blobs'] if 'blobs' in caffe_model else caffe_model
    state_dict = OrderedDict()
    converted_names = set()
    convert_conv_fc(blobs, state_dict, 'conv1', 'conv1', converted_names)
    convert_bn(blobs, state_dict, 'res_conv1_bn', 'bn1', converted_names)
    for i in range(1, len(block_nums) + 1):
        for j in range(block_nums[i - 1]):
            if j == 0:
                convert_conv_fc(blobs, state_dict, f'res{i + 1}_{j}_branch1', f'layer{i}.{j}.downsample.0', converted_names)
                convert_bn(blobs, state_dict, f'res{i + 1}_{j}_branch1_bn', f'layer{i}.{j}.downsample.1', converted_names)
            for k, letter in enumerate(['a', 'b', 'c']):
                convert_conv_fc(blobs, state_dict, f'res{i + 1}_{j}_branch2{letter}', f'layer{i}.{j}.conv{k + 1}', converted_names)
                convert_bn(blobs, state_dict, f'res{i + 1}_{j}_branch2{letter}_bn', f'layer{i}.{j}.bn{k + 1}', converted_names)
    for key in blobs:
        if key not in converted_names:
            print(f'Not Convert: {key}')
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)

def convert_conv_fc(blobs, state_dict, caffe_name, torch_name, converted_names):
    state_dict[torch_name + '.weight'] = torch.from_numpy(blobs[caffe_name + '_w'])
    converted_names.add(caffe_name + '_w')
    if caffe_name + '_b' in blobs:
        state_dict[torch_name + '.bias'] = torch.from_numpy(blobs[caffe_name + '_b'])
        converted_names.add(caffe_name + '_b')

def convert_bn(blobs, state_dict, caffe_name, torch_name, converted_names):
    state_dict[torch_name + '.bias'] = torch.from_numpy(blobs[caffe_name + '_b'])
    state_dict[torch_name + '.weight'] = torch.from_numpy(blobs[caffe_name + '_s'])
    bn_size = state_dict[torch_name + '.weight'].size()
    state_dict[torch_name + '.running_mean'] = torch.zeros(bn_size)
    state_dict[torch_name + '.running_var'] = torch.ones(bn_size)
    converted_names.add(caffe_name + '_b')
    converted_names.add(caffe_name + '_s')

def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    parser.add_argument('depth', type=int, help='ResNet model depth')
    args = parser.parse_args()
    convert(args.src, args.dst, args.depth)

def main():
    parser = argparse.ArgumentParser(description='Upgrade model version')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    parser.add_argument('--num-classes', type=int, default=81, help='number of classes of the original model')
    args = parser.parse_args()
    convert(args.in_file, args.out_file, args.num_classes)

def main():
    parser = argparse.ArgumentParser(description='Upgrade SSD version')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)

def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)

