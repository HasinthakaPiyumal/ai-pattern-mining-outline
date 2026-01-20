# Cluster 4

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', default=20, help='default: 20', type=int)
    parser.add_argument('--batch_size', default=128, help='default: 128', type=int)
    parser.add_argument('--num_threads', default=4, help='# of data read threads (default: 4)', type=int)
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True)
    parser.add_argument('--name', help='default: name=model')
    parser.add_argument('--dataset', '-D', help='CelebA / LSUN', required=True)
    parser.add_argument('--ckpt_step', default=5000, help='# of steps for saving checkpoint (default: 5000)', type=int)
    parser.add_argument('--renew', action='store_true', help='train model from scratch -         clean saved checkpoints and summaries', default=False)
    return parser

def pprint_args(FLAGS):
    print('\nParameters:')
    for attr, value in sorted(vars(FLAGS).items()):
        print('{}={}'.format(attr.upper(), value))
    print('')

def get_dataset(dataset_name):
    celebA_64 = './data/celebA_tfrecords/*.tfrecord'
    celebA_128 = './data/celebA_128_tfrecords/*.tfrecord'
    lsun_bedroom_128 = './data/lsun/bedroom_128_tfrecords/*.tfrecord'
    if dataset_name == 'celeba':
        path = celebA_128
        n_examples = 202599
    elif dataset_name == 'lsun':
        path = lsun_bedroom_128
        n_examples = 3033042
    else:
        raise ValueError('{} is does not supported. dataset must be celeba or lsun.'.format(dataset_name))
    return (path, n_examples)

def get_model(mtype, name, training):
    model = None
    if mtype == 'DCGAN':
        model = dcgan.DCGAN
    elif mtype == 'LSGAN':
        model = lsgan.LSGAN
    elif mtype == 'WGAN':
        model = wgan.WGAN
    elif mtype == 'WGAN-GP':
        model = wgan_gp.WGAN_GP
    elif mtype == 'EBGAN':
        model = ebgan.EBGAN
    elif mtype == 'BEGAN':
        model = began.BEGAN
    elif mtype == 'DRAGAN':
        model = dragan.DRAGAN
    elif mtype == 'COULOMBGAN':
        model = coulombgan.CoulombGAN
    else:
        assert False, mtype + ' is not in the model zoo'
    assert model, mtype + ' is work in progress'
    return model(name=name, training=training)

