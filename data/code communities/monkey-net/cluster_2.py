# Cluster 2

def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    if checkpoint is not None:
        start_epoch, it = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector, optimizer_generator, optimizer_discriminator, optimizer_kp_detector)
    else:
        start_epoch = 0
        it = 0
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    generator_full_par = DataParallelWithCallback(generator_full, device_ids=device_ids)
    discriminator_full_par = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], **train_params['log_params']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                out = generator_full_par(x)
                loss_values = out[:-2]
                generated = out[-2]
                kp_joined = out[-1]
                loss_values = [val.mean() for val in loss_values]
                loss = sum(loss_values)
                loss.backward(retain_graph=not train_params['detach_kp_discriminator'])
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_discriminator.zero_grad()
                if train_params['detach_kp_discriminator']:
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()
                generator_loss_values = [val.detach().cpu().numpy() for val in loss_values]
                loss_values = discriminator_full_par(x, kp_joined, generated)
                loss_values = [val.mean() for val in loss_values]
                loss = sum(loss_values)
                loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
                if not train_params['detach_kp_discriminator']:
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()
                discriminator_loss_values = [val.detach().cpu().numpy() for val in loss_values]
                logger.log_iter(it, names=generator_loss_names(train_params['loss_weights']) + discriminator_loss_names(), values=generator_loss_values + discriminator_loss_values, inp=x, out=generated)
                it += 1
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            logger.log_epoch(epoch, {'generator': generator, 'discriminator': discriminator, 'kp_detector': kp_detector, 'optimizer_generator': optimizer_generator, 'optimizer_discriminator': optimizer_discriminator, 'optimizer_kp_detector': optimizer_kp_detector})

def generator_loss_names(loss_weights):
    loss_names = []
    if loss_weights['reconstruction_deformed'] != 0:
        loss_names.append('rec_def')
    if loss_weights['reconstruction'] is not None:
        for i, _ in enumerate(loss_weights['reconstruction']):
            if loss_weights['reconstruction'][i] == 0:
                continue
            loss_names.append('layer-%s_rec' % i)
    loss_names.append('gen_gan')
    return loss_names

def discriminator_loss_names():
    return ['disc_gan']

