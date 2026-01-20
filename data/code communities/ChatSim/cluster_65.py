# Cluster 65

def get_checkpoint_files(s):
    s = s.strip()
    if ',' in s:
        return [get_checkpoint_files(chunk) for chunk in s.split(',')]
    return 'last.ckpt' if s == 'last' else f'{s}.ckpt'

def main(args):
    checkpoint_fnames = get_checkpoint_files(args.epochs)
    if isinstance(checkpoint_fnames, str):
        checkpoint_fnames = [checkpoint_fnames]
    assert len(checkpoint_fnames) >= 1
    checkpoint_path = os.path.join(args.indir, 'models', checkpoint_fnames[0])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    del checkpoint['optimizer_states']
    if len(checkpoint_fnames) > 1:
        for fname in checkpoint_fnames[1:]:
            print('sum', fname)
            sum_tensors_cnt = 0
            other_cp = torch.load(os.path.join(args.indir, 'models', fname), map_location='cpu')
            for k in checkpoint['state_dict'].keys():
                if checkpoint['state_dict'][k].dtype is torch.float:
                    checkpoint['state_dict'][k].data.add_(other_cp['state_dict'][k].data)
                    sum_tensors_cnt += 1
            print('summed', sum_tensors_cnt, 'tensors')
        for k in checkpoint['state_dict'].keys():
            if checkpoint['state_dict'][k].dtype is torch.float:
                checkpoint['state_dict'][k].data.mul_(1 / float(len(checkpoint_fnames)))
    state_dict = checkpoint['state_dict']
    if not args.leave_discriminators:
        for k in list(state_dict.keys()):
            if k.startswith('discriminator.'):
                del state_dict[k]
    if not args.leave_losses:
        for k in list(state_dict.keys()):
            if k.startswith('loss_'):
                del state_dict[k]
    out_checkpoint_path = os.path.join(args.outdir, 'models', 'best.ckpt')
    os.makedirs(os.path.dirname(out_checkpoint_path), exist_ok=True)
    torch.save(checkpoint, out_checkpoint_path)
    shutil.copy2(os.path.join(args.indir, 'config.yaml'), os.path.join(args.outdir, 'config.yaml'))

