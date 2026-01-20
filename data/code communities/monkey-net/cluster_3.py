# Cluster 3

def prediction(config, generator, kp_detector, checkpoint, log_dir):
    dataset = FramesDataset(is_train=True, transform=VideoToTensor(), **config['dataset_params'])
    log_dir = os.path.join(log_dir, 'prediction')
    png_dir = os.path.join(log_dir, 'png')
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='prediction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    print('Extracting keypoints...')
    kp_detector.eval()
    generator.eval()
    keypoints_array = []
    prediction_params = config['prediction_params']
    for it, x in tqdm(enumerate(dataloader)):
        if prediction_params['train_size'] is not None:
            if it > prediction_params['train_size']:
                break
        with torch.no_grad():
            keypoints = []
            for i in range(x['video'].shape[2]):
                kp = kp_detector(x['video'][:, :, i:i + 1])
                kp = {k: v.data.cpu().numpy() for k, v in kp.items()}
                keypoints.append(kp)
            keypoints_array.append(keypoints)
    predictor = PredictionModule(num_kp=config['model_params']['common_params']['num_kp'], kp_variance=config['model_params']['common_params']['kp_variance'], **prediction_params['rnn_params']).cuda()
    num_epochs = prediction_params['num_epochs']
    lr = prediction_params['lr']
    bs = prediction_params['batch_size']
    num_frames = prediction_params['num_frames']
    init_frames = prediction_params['init_frames']
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=50)
    kp_dataset = KPDataset(keypoints_array, num_frames=num_frames)
    kp_dataloader = DataLoader(kp_dataset, batch_size=bs)
    print('Training prediction...')
    for _ in trange(num_epochs):
        loss_list = []
        for x in kp_dataloader:
            x = {k: v.cuda() for k, v in x.items()}
            gt = {k: v.clone() for k, v in x.items()}
            for k in x:
                x[k][:, init_frames:] = 0
            prediction = predictor(x)
            loss = sum([torch.abs(gt[k][:, init_frames:] - prediction[k][:, init_frames:]).mean() for k in x])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.detach().data.cpu().numpy())
        loss = np.mean(loss_list)
        scheduler.step(loss)
    dataset = FramesDataset(is_train=False, transform=VideoToTensor(), **config['dataset_params'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    print('Make predictions...')
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x['video'] = x['video'][:, :, :num_frames]
            kp_init = kp_detector(x['video'])
            for k in kp_init:
                kp_init[k][:, init_frames:] = 0
            kp_source = kp_detector(x['video'][:, :, :1])
            kp_video = predictor(kp_init)
            for k in kp_video:
                kp_video[k][:, :init_frames] = kp_init[k][:, :init_frames]
            if 'var' in kp_video and prediction_params['predict_variance']:
                kp_video['var'] = kp_init['var'][:, init_frames - 1:init_frames].repeat(1, kp_video['var'].shape[1], 1, 1, 1)
            out = generate(generator, appearance_image=x['video'][:, :, :1], kp_appearance=kp_source, kp_video=kp_video)
            x['source'] = x['video'][:, :, :1]
            out_video_batch = out['video_prediction'].data.cpu().numpy()
            out_video_batch = np.concatenate(np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0], axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * out_video_batch).astype(np.uint8))
            image = Visualizer(**config['visualizer_params']).visualize_reconstruction(x, out)
            image_name = x['name'][0] + prediction_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), image)
            del x, kp_video, kp_source, out

def generate(generator, appearance_image, kp_appearance, kp_video):
    out = {'video_prediction': [], 'video_deformed': []}
    for i in range(kp_video['mean'].shape[1]):
        kp_target = {k: v[:, i:i + 1] for k, v in kp_video.items()}
        kp_dict_part = {'kp_driving': kp_target, 'kp_source': kp_appearance}
        out_part = generator(appearance_image, **kp_dict_part)
        out['video_prediction'].append(out_part['video_prediction'])
        out['video_deformed'].append(out_part['video_deformed'])
    out['video_prediction'] = torch.cat(out['video_prediction'], dim=2)
    out['video_deformed'] = torch.cat(out['video_deformed'], dim=2)
    out['kp_driving'] = kp_video
    out['kp_source'] = kp_appearance
    return out

def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='test'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    loss_list = []
    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()
    cat_dict = lambda l, dim: {k: torch.cat([v[k] for v in l], dim=dim) for k in l[0]}
    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            kp_appearance = kp_detector(x['video'][:, :, :1])
            d = x['video'].shape[2]
            kp_video = cat_dict([kp_detector(x['video'][:, :, i:i + 1]) for i in range(d)], dim=1)
            out = generate(generator, appearance_image=x['video'][:, :, :1], kp_appearance=kp_appearance, kp_video=kp_video)
            x['source'] = x['video'][:, :, :1]
            out_video_batch = out['video_prediction'].data.cpu().numpy()
            out_video_batch = np.concatenate(np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0], axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * out_video_batch).astype(np.uint8))
            image = Visualizer(**config['visualizer_params']).visualize_reconstruction(x, out)
            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), image)
            loss = reconstruction_loss(out['video_prediction'].cpu(), x['video'].cpu(), 1)
            loss_list.append(loss.data.cpu().numpy())
            del x, kp_video, kp_appearance, out, loss
    print('Reconstruction loss: %s' % np.mean(loss_list))

def transfer(config, generator, kp_detector, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'transfer')
    png_dir = os.path.join(log_dir, 'png')
    transfer_params = config['transfer_params']
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=transfer_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='transfer'.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x = {key: value if not hasattr(value, 'cuda') else value.cuda() for key, value in x.items()}
            driving_video = x['driving_video']
            source_image = x['source_video'][:, :, :1, :, :]
            out = transfer_one(generator, kp_detector, source_image, driving_video, transfer_params)
            img_name = '-'.join([x['driving_name'][0], x['source_name'][0]])
            out_video_batch = out['video_prediction'].data.cpu().numpy()
            out_video_batch = np.concatenate(np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0], axis=1)
            imageio.imsave(os.path.join(png_dir, img_name + '.png'), (255 * out_video_batch).astype(np.uint8))
            image = Visualizer(**config['visualizer_params']).visualize_transfer(driving_video=driving_video, source_image=source_image, out=out)
            imageio.mimsave(os.path.join(log_dir, img_name + transfer_params['format']), image)

