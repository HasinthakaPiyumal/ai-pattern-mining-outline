# Cluster 1

def train(epochs=10, batch_size=32, alpha=0.6, w=0.4, num_workers=2, lr=0.0001, save_epoch=10, train_path=ROOT / 'dataset/KITTI/training', model_path=ROOT / 'weights/', select_model='resnet18', api_key=''):
    train_path = str(train_path)
    model_path = str(model_path)
    print('[INFO] Loading dataset...')
    dataset = Dataset(train_path)
    hyper_params = {'epochs': epochs, 'batch_size': batch_size, 'w': w, 'num_workers': num_workers, 'lr': lr, 'shuffle': True}
    experiment = Experiment(api_key, project_name='YOLO3D')
    experiment.log_parameters(hyper_params)
    data_gen = data.DataLoader(dataset, batch_size=hyper_params['batch_size'], shuffle=hyper_params['shuffle'], num_workers=hyper_params['num_workers'])
    base_model = model_factory[select_model]
    model = regressor_factory[select_model](model=base_model).cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=hyper_params['lr'], momentum=0.9)
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss
    latest_model = None
    first_epoch = 1
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass
    if latest_model is not None:
        checkpoint = torch.load(model_path + latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'[INFO] Using previous model {latest_model} at {first_epoch} epochs')
        print('[INFO] Resuming training...')
    total_num_batches = int(len(dataset) / hyper_params['batch_size'])
    with experiment.train():
        for epoch in range(first_epoch, int(hyper_params['epochs']) + 1):
            curr_batch = 0
            passes = 0
            with tqdm(data_gen, unit='batch') as tepoch:
                for local_batch, local_labels in tepoch:
                    tepoch.set_description(f'Epoch {epoch}')
                    truth_orient = local_labels['Orientation'].float().cuda()
                    truth_conf = local_labels['Confidence'].float().cuda()
                    truth_dim = local_labels['Dimensions'].float().cuda()
                    local_batch = local_batch.float().cuda()
                    [orient, conf, dim] = model(local_batch)
                    orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
                    dim_loss = dim_loss_func(dim, truth_dim)
                    truth_conf = torch.max(truth_conf, dim=1)[1]
                    conf_loss = conf_loss_func(conf, truth_conf)
                    loss_theta = conf_loss + w * orient_loss
                    loss = alpha * dim_loss + loss_theta
                    writer.add_scalar('Loss/train', loss, epoch)
                    experiment.log_metric('Loss/train', loss, epoch=epoch)
                    opt_SGD.zero_grad()
                    loss.backward()
                    opt_SGD.step()
                    tepoch.set_postfix(loss=loss.item())
            if epoch % save_epoch == 0:
                model_name = os.path.join(model_path, f'{select_model}_epoch_{epoch}.pkl')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt_SGD.state_dict(), 'loss': loss}, model_name)
                print(f'[INFO] Saving weights as {model_name}')
    writer.flush()
    writer.close()

def main(opt):
    train(**vars(opt))

def parse_opt():
    parser = argparse.ArgumentParser(description='Regressor Model Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size')
    parser.add_argument('--alpha', type=float, default=0.6, help='Aplha default=0.6 DONT CHANGE')
    parser.add_argument('--w', type=float, default=0.4, help='w DONT CHANGE')
    parser.add_argument('--num_workers', type=int, default=2, help='Total # workers, for colab & kaggle use 2')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_epoch', type=int, default=10, help='Save model every # epochs')
    parser.add_argument('--train_path', type=str, default=ROOT / 'dataset/KITTI/training', help='Training path KITTI')
    parser.add_argument('--model_path', type=str, default=ROOT / 'weights', help='Weights path, for load and save model')
    parser.add_argument('--select_model', type=str, default='resnet18', help='Model selection: {resnet18, vgg11}')
    parser.add_argument('--api_key', type=str, default='', help='API key for comet.ml')
    opt = parser.parse_args()
    return opt

def main(opt):
    train(**vars(opt))

def sweep():
    wandb.init()
    hyp_dict = vars(wandb.config).get('_items')
    opt = parse_opt(known=True)
    opt.batch_size = hyp_dict.get('batch_size')
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    opt.epochs = hyp_dict.get('epochs')
    opt.nosave = True
    opt.data = hyp_dict.get('data')
    opt.weights = str(opt.weights)
    opt.cfg = str(opt.cfg)
    opt.data = str(opt.data)
    opt.hyp = str(opt.hyp)
    opt.project = str(opt.project)
    device = select_device(opt.device, batch_size=opt.batch_size)
    train(hyp_dict, opt, device, callbacks=Callbacks())

