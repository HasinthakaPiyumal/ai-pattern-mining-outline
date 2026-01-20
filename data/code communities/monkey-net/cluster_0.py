# Cluster 0

class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

    def forward(self, x):
        kp_joined = self.kp_extractor(torch.cat([x['source'], x['video']], dim=2))
        generated = self.generator(x['source'], **split_kp(kp_joined, self.train_params['detach_kp_generator']))
        video_prediction = generated['video_prediction']
        video_deformed = generated['video_deformed']
        kp_dict = split_kp(kp_joined, False)
        discriminator_maps_generated = self.discriminator(video_prediction, **kp_dict)
        discriminator_maps_real = self.discriminator(x['video'], **kp_dict)
        generated.update(kp_dict)
        losses = generator_loss(discriminator_maps_generated=discriminator_maps_generated, discriminator_maps_real=discriminator_maps_real, video_deformed=video_deformed, loss_weights=self.train_params['loss_weights'])
        return tuple(losses) + (generated, kp_joined)

def split_kp(kp_joined, detach=False):
    if detach:
        kp_video = {k: v[:, 1:].detach() for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1].detach() for k, v in kp_joined.items()}
    else:
        kp_video = {k: v[:, 1:] for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1] for k, v in kp_joined.items()}
    return {'kp_driving': kp_video, 'kp_source': kp_appearance}

def generator_loss(discriminator_maps_generated, discriminator_maps_real, video_deformed, loss_weights):
    loss_values = []
    if loss_weights['reconstruction_deformed'] != 0:
        loss_values.append(reconstruction_loss(discriminator_maps_real[0], video_deformed, loss_weights['reconstruction_deformed']))
    if loss_weights['reconstruction'] != 0:
        for i, (a, b) in enumerate(zip(discriminator_maps_real[:-1], discriminator_maps_generated[:-1])):
            if loss_weights['reconstruction'][i] == 0:
                continue
            loss_values.append(reconstruction_loss(b, a, weight=loss_weights['reconstruction'][i]))
    loss_values.append(generator_gan_loss(discriminator_maps_generated, weight=loss_weights['generator_gan']))
    return loss_values

class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

    def forward(self, x, kp_joined, generated):
        kp_dict = split_kp(kp_joined, self.train_params['detach_kp_discriminator'])
        discriminator_maps_generated = self.discriminator(generated['video_prediction'].detach(), **kp_dict)
        discriminator_maps_real = self.discriminator(x['video'], **kp_dict)
        loss = discriminator_loss(discriminator_maps_generated=discriminator_maps_generated, discriminator_maps_real=discriminator_maps_real, loss_weights=self.train_params['loss_weights'])
        return loss

def discriminator_loss(discriminator_maps_generated, discriminator_maps_real, loss_weights):
    loss_values = [discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real, loss_weights['discriminator_gan'])]
    return loss_values

def reconstruction_loss(prediction, target, weight):
    if weight == 0:
        return 0
    return weight * mean_batch(torch.abs(prediction - target))

def mean_batch(val):
    return val.view(val.shape[0], -1).mean(-1)

def generator_gan_loss(discriminator_maps_generated, weight):
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_generated) ** 2
    return weight * mean_batch(score)

def discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real, weight):
    scores_real = discriminator_maps_real[-1]
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_real) ** 2 + scores_generated ** 2
    return weight * mean_batch(score)

