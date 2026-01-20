# Cluster 6

def transfer_one(generator, kp_detector, source_image, driving_video, transfer_params):
    cat_dict = lambda l, dim: {k: torch.cat([v[k] for v in l], dim=dim) for k in l[0]}
    d = driving_video.shape[2]
    kp_driving = cat_dict([kp_detector(driving_video[:, :, i:i + 1]) for i in range(d)], dim=1)
    kp_source = kp_detector(source_image)
    kp_driving_norm = normalize_kp(kp_driving, kp_source, **transfer_params['normalization_params'])
    kp_video_list = [{k: v[:, i:i + 1] for k, v in kp_driving_norm.items()} for i in range(d)]
    out = cat_dict([generator(source_image=source_image, kp_driving=kp, kp_source=kp_source) for kp in kp_video_list], dim=2)
    out['kp_driving'] = kp_driving
    out['kp_source'] = kp_source
    out['kp_norm'] = kp_driving_norm
    return out

def normalize_kp(kp_video, kp_appearance, movement_mult=False, move_location=False, adapt_variance=False, clip_mean=False):
    if movement_mult:
        appearance_area = ConvexHull(kp_appearance['mean'][0, 0].data.cpu().numpy()).volume
        video_area = ConvexHull(kp_video['mean'][0, 0].data.cpu().numpy()).volume
        movement_mult = np.sqrt(appearance_area) / np.sqrt(video_area)
    else:
        movement_mult = 1
    kp_video = {k: v for k, v in kp_video.items()}
    if move_location:
        kp_video_diff = kp_video['mean'] - kp_video['mean'][:, 0:1]
        kp_video_diff *= movement_mult
        kp_video['mean'] = kp_video_diff + kp_appearance['mean']
    if clip_mean:
        one = torch.ones(1).type(kp_video_diff.type())
        kp_video['mean'] = torch.max(kp_video['mean'], -one)
        kp_video['mean'] = torch.min(kp_video['mean'], one)
    if 'var' in kp_video and adapt_variance:
        var_first = kp_video['var'][:, 0:1].repeat(1, kp_video['var'].shape[1], 1, 1, 1)
        kp_var, _ = torch.gesv(var_first, kp_video['var'])
        kp_var = torch.matmul(kp_video['var'], matrix_inverse(kp_video['var'][:, 0:1], eps=0))
        kp_var = torch.matmul(kp_var, kp_appearance['var'])
        kp_var = make_symetric_matrix(kp_var)
        kp_video['var'] = kp_var
    return kp_video

def make_symetric_matrix(torch_matrix):
    a = torch_matrix.cpu().numpy()
    c = (a + np.transpose(a, (0, 1, 2, 4, 3))) / 2
    d, u = np.linalg.eig(c)
    d[d <= 0] = 1e-06
    d_matrix = np.zeros_like(a)
    d_matrix[..., 0, 0] = d[..., 0]
    d_matrix[..., 1, 1] = d[..., 1]
    res = np.matmul(np.matmul(u, d_matrix), np.transpose(u, (0, 1, 2, 4, 3)))
    res = torch.from_numpy(res).type(torch_matrix.type())
    return res

