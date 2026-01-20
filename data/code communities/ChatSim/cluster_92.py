# Cluster 92

class PairwiseScore(EvaluatorScore, ABC):

    def __init__(self):
        super().__init__()
        self.individual_values = None

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        individual_values = torch.cat(states, dim=-1).reshape(-1).cpu().numpy() if states is not None else self.individual_values
        total_results = {'mean': individual_values.mean(), 'std': individual_values.std()}
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_scores = individual_values[index]
            group_results[label] = {'mean': group_scores.mean(), 'std': group_scores.std()}
        return (total_results, group_results)

    def reset(self):
        self.individual_values = []

def get_groupings(groups):
    """
    :param groups: group numbers for respective elements
    :return: dict of kind {group_idx: indices of the corresponding group elements}
    """
    label_groups, count_groups = np.unique(groups, return_counts=True)
    indices = np.argsort(groups)
    grouping = dict()
    cur_start = 0
    for label, count in zip(label_groups, count_groups):
        cur_end = cur_start + count
        cur_indices = indices[cur_start:cur_end]
        grouping[label] = cur_indices
        cur_start = cur_end
    return grouping

def calculate_frechet_distance(activations_pred, activations_target, eps=1e-06):
    mu1, sigma1 = fid_calculate_activation_statistics(activations_pred)
    mu2, sigma2 = fid_calculate_activation_statistics(activations_target)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = 'fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps
        LOGGER.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.01):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def fid_calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return (mu, sigma)

class FIDScore(EvaluatorScore):

    def __init__(self, dims=2048, eps=1e-06):
        LOGGER.info('FIDscore init called')
        super().__init__()
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.reset()
        LOGGER.info('FIDscore init done')

    def forward(self, pred_batch, target_batch, mask=None):
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)
        self.activations_pred.append(activations_pred.detach().cpu())
        self.activations_target.append(activations_target.detach().cpu())
        return (activations_pred, activations_target)

    def get_value(self, groups=None, states=None):
        LOGGER.info('FIDscore get_value called')
        activations_pred, activations_target = zip(*states) if states is not None else (self.activations_pred, self.activations_target)
        activations_pred = torch.cat(activations_pred).cpu().numpy()
        activations_target = torch.cat(activations_target).cpu().numpy()
        total_distance = calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)
        total_results = dict(mean=total_distance)
        if groups is None:
            group_results = None
        else:
            group_results = dict()
            grouping = get_groupings(groups)
            for label, index in grouping.items():
                if len(index) > 1:
                    group_distance = calculate_frechet_distance(activations_pred[index], activations_target[index], eps=self.eps)
                    group_results[label] = dict(mean=group_distance)
                else:
                    group_results[label] = dict(mean=float('nan'))
        self.reset()
        LOGGER.info('FIDscore get_value done')
        return (total_results, group_results)

    def reset(self):
        self.activations_pred = []
        self.activations_target = []

    def _get_activations(self, batch):
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            assert False, 'We should not have got here, because Inception always scales inputs to 299x299'
        activations = activations.squeeze(-1).squeeze(-1)
        return activations

class SegmentationAwarePairwiseScore(SegmentationAwareScore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.individual_values = []
        self.segm_idx2name = get_segmentation_idx2name()

    def forward(self, pred_batch, target_batch, mask):
        cur_class_stats = super().forward(pred_batch, target_batch, mask)
        score_values = self.calc_score(pred_batch, target_batch, mask)
        self.individual_values.append(score_values)
        return cur_class_stats + (score_values,)

    @abstractmethod
    def calc_score(self, pred_batch, target_batch, mask):
        raise NotImplementedError()

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, individual_values = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            individual_values = self.individual_values
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        individual_values = np.concatenate(individual_values, axis=0)
        total_results = {'mean': individual_values.mean(), 'std': individual_values.std(), **distribute_values_to_classes(target_class_freq_by_image_mask, individual_values, self.segm_idx2name)}
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_class_freq = target_class_freq_by_image_mask[index]
            group_scores = individual_values[index]
            group_results[label] = {'mean': group_scores.mean(), 'std': group_scores.std(), **distribute_values_to_classes(group_class_freq, group_scores, self.segm_idx2name)}
        return (total_results, group_results)

    def reset(self):
        super().reset()
        self.individual_values = []

def distribute_values_to_classes(target_class_freq_by_image_mask, values, idx2name):
    assert target_class_freq_by_image_mask.ndim == 2 and target_class_freq_by_image_mask.shape[0] == values.shape[0]
    total_class_freq = target_class_freq_by_image_mask.sum(0)
    distr_values = (target_class_freq_by_image_mask * values[..., None]).sum(0)
    result = distr_values / (total_class_freq + 0.001)
    return {idx2name[i]: val for i, val in enumerate(result) if total_class_freq[i] > 0}

class SegmentationClassStats(SegmentationAwarePairwiseScore):

    def calc_score(self, pred_batch, target_batch, mask):
        return 0

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, _ = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        target_class_freq_by_image_total_marginal = target_class_freq_by_image_total.sum(0).astype('float32')
        target_class_freq_by_image_total_marginal /= target_class_freq_by_image_total_marginal.sum()
        target_class_freq_by_image_mask_marginal = target_class_freq_by_image_mask.sum(0).astype('float32')
        target_class_freq_by_image_mask_marginal /= target_class_freq_by_image_mask_marginal.sum()
        pred_class_freq_diff = (pred_class_freq_by_image_mask - target_class_freq_by_image_mask).sum(0) / (target_class_freq_by_image_mask.sum(0) + 0.001)
        total_results = dict()
        total_results.update({f'total_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(target_class_freq_by_image_total_marginal) if v > 0})
        total_results.update({f'mask_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(target_class_freq_by_image_mask_marginal) if v > 0})
        total_results.update({f'mask_freq_diff/{self.segm_idx2name[i]}': v for i, v in enumerate(pred_class_freq_diff) if target_class_freq_by_image_total_marginal[i] > 0})
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_target_class_freq_by_image_total = target_class_freq_by_image_total[index]
            group_target_class_freq_by_image_mask = target_class_freq_by_image_mask[index]
            group_pred_class_freq_by_image_mask = pred_class_freq_by_image_mask[index]
            group_target_class_freq_by_image_total_marginal = group_target_class_freq_by_image_total.sum(0).astype('float32')
            group_target_class_freq_by_image_total_marginal /= group_target_class_freq_by_image_total_marginal.sum()
            group_target_class_freq_by_image_mask_marginal = group_target_class_freq_by_image_mask.sum(0).astype('float32')
            group_target_class_freq_by_image_mask_marginal /= group_target_class_freq_by_image_mask_marginal.sum()
            group_pred_class_freq_diff = (group_pred_class_freq_by_image_mask - group_target_class_freq_by_image_mask).sum(0) / (group_target_class_freq_by_image_mask.sum(0) + 0.001)
            cur_group_results = dict()
            cur_group_results.update({f'total_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(group_target_class_freq_by_image_total_marginal) if v > 0})
            cur_group_results.update({f'mask_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(group_target_class_freq_by_image_mask_marginal) if v > 0})
            cur_group_results.update({f'mask_freq_diff/{self.segm_idx2name[i]}': v for i, v in enumerate(group_pred_class_freq_diff) if group_target_class_freq_by_image_total_marginal[i] > 0})
            group_results[label] = cur_group_results
        return (total_results, group_results)

def calculade_fid_no_img(img_i, activations_pred, activations_target, eps=1e-06):
    activations_pred = activations_pred.copy()
    activations_pred[img_i] = activations_target[img_i]
    return calculate_frechet_distance(activations_pred, activations_target, eps=eps)

class SegmentationAwareFID(SegmentationAwarePairwiseScore):

    def __init__(self, *args, dims=2048, eps=1e-06, n_jobs=-1, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.n_jobs = n_jobs

    def calc_score(self, pred_batch, target_batch, mask):
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)
        return (activations_pred, activations_target)

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, activation_pairs = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            activation_pairs = self.individual_values
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        activations_pred, activations_target = zip(*activation_pairs)
        activations_pred = np.concatenate(activations_pred, axis=0)
        activations_target = np.concatenate(activations_target, axis=0)
        total_results = {'mean': calculate_frechet_distance(activations_pred, activations_target, eps=self.eps), 'std': 0, **self.distribute_fid_to_classes(target_class_freq_by_image_mask, activations_pred, activations_target)}
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            if len(index) > 1:
                group_activations_pred = activations_pred[index]
                group_activations_target = activations_target[index]
                group_class_freq = target_class_freq_by_image_mask[index]
                group_results[label] = {'mean': calculate_frechet_distance(group_activations_pred, group_activations_target, eps=self.eps), 'std': 0, **self.distribute_fid_to_classes(group_class_freq, group_activations_pred, group_activations_target)}
            else:
                group_results[label] = dict(mean=float('nan'), std=0)
        return (total_results, group_results)

    def distribute_fid_to_classes(self, class_freq, activations_pred, activations_target):
        real_fid = calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)
        fid_no_images = Parallel(n_jobs=self.n_jobs)((delayed(calculade_fid_no_img)(img_i, activations_pred, activations_target, eps=self.eps) for img_i in range(activations_pred.shape[0])))
        errors = real_fid - fid_no_images
        return distribute_values_to_classes(class_freq, errors, self.segm_idx2name)

    def _get_activations(self, batch):
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            activations = F.adaptive_avg_pool2d(activations, output_size=(1, 1))
        activations = activations.squeeze(-1).squeeze(-1).detach().cpu().numpy()
        return activations

def get_activations(files, model, batch_size=50, dims=2048, cuda=False, verbose=False, keep_size=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    if len(files) % batch_size != 0:
        print('Warning: number of images is not a multiple of the batch size. Some samples are going to be ignored.')
    if batch_size > len(files):
        print('Warning: batch size is bigger than the data size. Setting batch size to data size')
        batch_size = len(files)
    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, dims))
    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size
        t = transform if not keep_size else ToTensor()
        if isinstance(files[0], pathlib.PosixPath):
            images = [t(Image.open(str(f))) for f in files[start:end]]
        elif isinstance(files[0], Image.Image):
            images = [t(f) for f in files[start:end]]
        else:
            raise ValueError(f'Unknown data type for image: {type(files[0])}')
        batch = torch.stack(images)
        if cuda:
            batch = batch.cuda()
        pred = model(batch)[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
    if verbose:
        print(' done')
    return pred_arr

def calculate_activation_statistics(files, model, batch_size=50, dims=2048, cuda=False, verbose=False, keep_size=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose, keep_size=keep_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return (mu, sigma)

def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = (f['mu'][:], f['sigma'][:])
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size, dims, cuda)
    return (m, s)

def _compute_statistics_of_images(images, model, batch_size, dims, cuda, keep_size=False):
    if isinstance(images, list):
        m, s = calculate_activation_statistics(images, model, batch_size, dims, cuda, keep_size=keep_size)
        return (m, s)
    else:
        raise ValueError

def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def calculate_fid_given_images(images, batch_size, cuda, dims, use_globals=False, keep_size=False):
    if use_globals:
        global FID_MODEL
    for imgs in images:
        if isinstance(imgs, list) and isinstance(imgs[0], (Image.Image, JpegImagePlugin.JpegImageFile)):
            pass
        else:
            raise RuntimeError('Invalid images')
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    if 'FID_MODEL' not in globals() or not use_globals:
        model = InceptionV3([block_idx])
        if cuda:
            model.cuda()
        if use_globals:
            FID_MODEL = model
    else:
        model = FID_MODEL
    m1, s1 = _compute_statistics_of_images(images[0], model, batch_size, dims, cuda, keep_size=False)
    m2, s2 = _compute_statistics_of_images(images[1], model, batch_size, dims, cuda, keep_size=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

class PairwiseScore(EvaluatorScore, ABC):

    def __init__(self):
        super().__init__()
        self.individual_values = None

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        individual_values = torch.cat(states, dim=-1).reshape(-1).cpu().numpy() if states is not None else self.individual_values
        total_results = {'mean': individual_values.mean(), 'std': individual_values.std()}
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_scores = individual_values[index]
            group_results[label] = {'mean': group_scores.mean(), 'std': group_scores.std()}
        return (total_results, group_results)

    def reset(self):
        self.individual_values = []

def calculate_frechet_distance(activations_pred, activations_target, eps=1e-06):
    mu1, sigma1 = fid_calculate_activation_statistics(activations_pred)
    mu2, sigma2 = fid_calculate_activation_statistics(activations_target)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = 'fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps
        LOGGER.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.01):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class FIDScore(EvaluatorScore):

    def __init__(self, dims=2048, eps=1e-06):
        LOGGER.info('FIDscore init called')
        super().__init__()
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.reset()
        LOGGER.info('FIDscore init done')

    def forward(self, pred_batch, target_batch, mask=None):
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)
        self.activations_pred.append(activations_pred.detach().cpu())
        self.activations_target.append(activations_target.detach().cpu())
        return (activations_pred, activations_target)

    def get_value(self, groups=None, states=None):
        LOGGER.info('FIDscore get_value called')
        activations_pred, activations_target = zip(*states) if states is not None else (self.activations_pred, self.activations_target)
        activations_pred = torch.cat(activations_pred).cpu().numpy()
        activations_target = torch.cat(activations_target).cpu().numpy()
        total_distance = calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)
        total_results = dict(mean=total_distance)
        if groups is None:
            group_results = None
        else:
            group_results = dict()
            grouping = get_groupings(groups)
            for label, index in grouping.items():
                if len(index) > 1:
                    group_distance = calculate_frechet_distance(activations_pred[index], activations_target[index], eps=self.eps)
                    group_results[label] = dict(mean=group_distance)
                else:
                    group_results[label] = dict(mean=float('nan'))
        self.reset()
        LOGGER.info('FIDscore get_value done')
        return (total_results, group_results)

    def reset(self):
        self.activations_pred = []
        self.activations_target = []

    def _get_activations(self, batch):
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            assert False, 'We should not have got here, because Inception always scales inputs to 299x299'
        activations = activations.squeeze(-1).squeeze(-1)
        return activations

class SegmentationAwarePairwiseScore(SegmentationAwareScore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.individual_values = []
        self.segm_idx2name = get_segmentation_idx2name()

    def forward(self, pred_batch, target_batch, mask):
        cur_class_stats = super().forward(pred_batch, target_batch, mask)
        score_values = self.calc_score(pred_batch, target_batch, mask)
        self.individual_values.append(score_values)
        return cur_class_stats + (score_values,)

    @abstractmethod
    def calc_score(self, pred_batch, target_batch, mask):
        raise NotImplementedError()

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, individual_values = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            individual_values = self.individual_values
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        individual_values = np.concatenate(individual_values, axis=0)
        total_results = {'mean': individual_values.mean(), 'std': individual_values.std(), **distribute_values_to_classes(target_class_freq_by_image_mask, individual_values, self.segm_idx2name)}
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_class_freq = target_class_freq_by_image_mask[index]
            group_scores = individual_values[index]
            group_results[label] = {'mean': group_scores.mean(), 'std': group_scores.std(), **distribute_values_to_classes(group_class_freq, group_scores, self.segm_idx2name)}
        return (total_results, group_results)

    def reset(self):
        super().reset()
        self.individual_values = []

class SegmentationClassStats(SegmentationAwarePairwiseScore):

    def calc_score(self, pred_batch, target_batch, mask):
        return 0

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, _ = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        target_class_freq_by_image_total_marginal = target_class_freq_by_image_total.sum(0).astype('float32')
        target_class_freq_by_image_total_marginal /= target_class_freq_by_image_total_marginal.sum()
        target_class_freq_by_image_mask_marginal = target_class_freq_by_image_mask.sum(0).astype('float32')
        target_class_freq_by_image_mask_marginal /= target_class_freq_by_image_mask_marginal.sum()
        pred_class_freq_diff = (pred_class_freq_by_image_mask - target_class_freq_by_image_mask).sum(0) / (target_class_freq_by_image_mask.sum(0) + 0.001)
        total_results = dict()
        total_results.update({f'total_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(target_class_freq_by_image_total_marginal) if v > 0})
        total_results.update({f'mask_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(target_class_freq_by_image_mask_marginal) if v > 0})
        total_results.update({f'mask_freq_diff/{self.segm_idx2name[i]}': v for i, v in enumerate(pred_class_freq_diff) if target_class_freq_by_image_total_marginal[i] > 0})
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_target_class_freq_by_image_total = target_class_freq_by_image_total[index]
            group_target_class_freq_by_image_mask = target_class_freq_by_image_mask[index]
            group_pred_class_freq_by_image_mask = pred_class_freq_by_image_mask[index]
            group_target_class_freq_by_image_total_marginal = group_target_class_freq_by_image_total.sum(0).astype('float32')
            group_target_class_freq_by_image_total_marginal /= group_target_class_freq_by_image_total_marginal.sum()
            group_target_class_freq_by_image_mask_marginal = group_target_class_freq_by_image_mask.sum(0).astype('float32')
            group_target_class_freq_by_image_mask_marginal /= group_target_class_freq_by_image_mask_marginal.sum()
            group_pred_class_freq_diff = (group_pred_class_freq_by_image_mask - group_target_class_freq_by_image_mask).sum(0) / (group_target_class_freq_by_image_mask.sum(0) + 0.001)
            cur_group_results = dict()
            cur_group_results.update({f'total_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(group_target_class_freq_by_image_total_marginal) if v > 0})
            cur_group_results.update({f'mask_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(group_target_class_freq_by_image_mask_marginal) if v > 0})
            cur_group_results.update({f'mask_freq_diff/{self.segm_idx2name[i]}': v for i, v in enumerate(group_pred_class_freq_diff) if group_target_class_freq_by_image_total_marginal[i] > 0})
            group_results[label] = cur_group_results
        return (total_results, group_results)

def calculade_fid_no_img(img_i, activations_pred, activations_target, eps=1e-06):
    activations_pred = activations_pred.copy()
    activations_pred[img_i] = activations_target[img_i]
    return calculate_frechet_distance(activations_pred, activations_target, eps=eps)

class SegmentationAwareFID(SegmentationAwarePairwiseScore):

    def __init__(self, *args, dims=2048, eps=1e-06, n_jobs=-1, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.n_jobs = n_jobs

    def calc_score(self, pred_batch, target_batch, mask):
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)
        return (activations_pred, activations_target)

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, activation_pairs = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            activation_pairs = self.individual_values
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        activations_pred, activations_target = zip(*activation_pairs)
        activations_pred = np.concatenate(activations_pred, axis=0)
        activations_target = np.concatenate(activations_target, axis=0)
        total_results = {'mean': calculate_frechet_distance(activations_pred, activations_target, eps=self.eps), 'std': 0, **self.distribute_fid_to_classes(target_class_freq_by_image_mask, activations_pred, activations_target)}
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            if len(index) > 1:
                group_activations_pred = activations_pred[index]
                group_activations_target = activations_target[index]
                group_class_freq = target_class_freq_by_image_mask[index]
                group_results[label] = {'mean': calculate_frechet_distance(group_activations_pred, group_activations_target, eps=self.eps), 'std': 0, **self.distribute_fid_to_classes(group_class_freq, group_activations_pred, group_activations_target)}
            else:
                group_results[label] = dict(mean=float('nan'), std=0)
        return (total_results, group_results)

    def distribute_fid_to_classes(self, class_freq, activations_pred, activations_target):
        real_fid = calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)
        fid_no_images = Parallel(n_jobs=self.n_jobs)((delayed(calculade_fid_no_img)(img_i, activations_pred, activations_target, eps=self.eps) for img_i in range(activations_pred.shape[0])))
        errors = real_fid - fid_no_images
        return distribute_values_to_classes(class_freq, errors, self.segm_idx2name)

    def _get_activations(self, batch):
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            activations = F.adaptive_avg_pool2d(activations, output_size=(1, 1))
        activations = activations.squeeze(-1).squeeze(-1).detach().cpu().numpy()
        return activations

def calculate_activation_statistics(files, model, batch_size=50, dims=2048, cuda=False, verbose=False, keep_size=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose, keep_size=keep_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return (mu, sigma)

def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = (f['mu'][:], f['sigma'][:])
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size, dims, cuda)
    return (m, s)

def _compute_statistics_of_images(images, model, batch_size, dims, cuda, keep_size=False):
    if isinstance(images, list):
        m, s = calculate_activation_statistics(images, model, batch_size, dims, cuda, keep_size=keep_size)
        return (m, s)
    else:
        raise ValueError

def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def calculate_fid_given_images(images, batch_size, cuda, dims, use_globals=False, keep_size=False):
    if use_globals:
        global FID_MODEL
    for imgs in images:
        if isinstance(imgs, list) and isinstance(imgs[0], (Image.Image, JpegImagePlugin.JpegImageFile)):
            pass
        else:
            raise RuntimeError('Invalid images')
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    if 'FID_MODEL' not in globals() or not use_globals:
        model = InceptionV3([block_idx])
        if cuda:
            model.cuda()
        if use_globals:
            FID_MODEL = model
    else:
        model = FID_MODEL
    m1, s1 = _compute_statistics_of_images(images[0], model, batch_size, dims, cuda, keep_size=False)
    m2, s2 = _compute_statistics_of_images(images[1], model, batch_size, dims, cuda, keep_size=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

