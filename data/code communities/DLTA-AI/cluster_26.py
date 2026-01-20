# Cluster 26

def build_anchor_generator(cfg, default_args=None):
    warnings.warn('``build_anchor_generator`` would be deprecated soon, please use ``build_prior_generator`` ')
    return build_prior_generator(cfg, default_args=default_args)

def test_standard_anchor_generator():
    from mmdet.core.anchor import build_anchor_generator
    anchor_generator_cfg = dict(type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8])
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    assert anchor_generator.num_base_priors == anchor_generator.num_base_anchors
    assert anchor_generator.num_base_priors == [3, 3]
    assert anchor_generator is not None

def test_ssd_anchor_generator():
    from mmdet.core.anchor import build_anchor_generator
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    with pytest.raises(AssertionError):
        anchor_generator_cfg = dict(type='SSDAnchorGenerator', scale_major=False, min_sizes=[48, 100, 150, 202, 253, 300], max_sizes=None, strides=[8, 16, 32, 64, 100, 300], ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])
        build_anchor_generator(anchor_generator_cfg)
    with pytest.raises(AssertionError):
        anchor_generator_cfg = dict(type='SSDAnchorGenerator', scale_major=False, min_sizes=[48, 100, 150, 202, 253, 300], max_sizes=[100, 150, 202, 253], strides=[8, 16, 32, 64, 100, 300], ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])
        build_anchor_generator(anchor_generator_cfg)
    anchor_generator_cfg = dict(type='SSDAnchorGenerator', scale_major=False, min_sizes=[48, 100, 150, 202, 253, 304], max_sizes=[100, 150, 202, 253, 304, 320], strides=[16, 32, 64, 107, 160, 320], ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]])
    featmap_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    expected_base_anchors = [torch.Tensor([[-16.0, -16.0, 32.0, 32.0], [-26.641, -26.641, 42.641, 42.641], [-25.9411, -8.9706, 41.9411, 24.9706], [-8.9706, -25.9411, 24.9706, 41.9411], [-33.5692, -5.8564, 49.5692, 21.8564], [-5.8564, -33.5692, 21.8564, 49.5692]]), torch.Tensor([[-34.0, -34.0, 66.0, 66.0], [-45.2372, -45.2372, 77.2372, 77.2372], [-54.7107, -19.3553, 86.7107, 51.3553], [-19.3553, -54.7107, 51.3553, 86.7107], [-70.6025, -12.8675, 102.6025, 44.8675], [-12.8675, -70.6025, 44.8675, 102.6025]]), torch.Tensor([[-43.0, -43.0, 107.0, 107.0], [-55.0345, -55.0345, 119.0345, 119.0345], [-74.066, -21.033, 138.066, 85.033], [-21.033, -74.066, 85.033, 138.066], [-97.9038, -11.3013, 161.9038, 75.3013], [-11.3013, -97.9038, 75.3013, 161.9038]]), torch.Tensor([[-47.5, -47.5, 154.5, 154.5], [-59.5332, -59.5332, 166.5332, 166.5332], [-89.3356, -17.9178, 196.3356, 124.9178], [-17.9178, -89.3356, 124.9178, 196.3356], [-121.4371, -4.8124, 228.4371, 111.8124], [-4.8124, -121.4371, 111.8124, 228.4371]]), torch.Tensor([[-46.5, -46.5, 206.5, 206.5], [-58.6651, -58.6651, 218.6651, 218.6651], [-98.898, -9.449, 258.898, 169.449], [-9.449, -98.898, 169.449, 258.898], [-139.1044, 6.9652, 299.1044, 153.0348], [6.9652, -139.1044, 153.0348, 299.1044]]), torch.Tensor([[8.0, 8.0, 312.0, 312.0], [4.0513, 4.0513, 315.9487, 315.9487], [-54.9605, 52.5198, 374.9604, 267.4802], [52.5198, -54.9605, 267.4802, 374.9604], [-103.2717, 72.2428, 423.2717, 247.7572], [72.2428, -103.2717, 247.7572, 423.2717]])]
    base_anchors = anchor_generator.base_anchors
    for i, base_anchor in enumerate(base_anchors):
        assert base_anchor.allclose(expected_base_anchors[i])
    expected_valid_pixels = [2400, 600, 150, 54, 24, 6]
    multi_level_valid_flags = anchor_generator.valid_flags(featmap_sizes, (320, 320), device)
    for i, single_level_valid_flag in enumerate(multi_level_valid_flags):
        assert single_level_valid_flag.sum() == expected_valid_pixels[i]
    assert anchor_generator.num_base_anchors == [6, 6, 6, 6, 6, 6]
    anchors = anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(anchors) == 6
    anchor_generator_cfg = dict(type='SSDAnchorGenerator', scale_major=False, input_size=300, basesize_ratio_range=(0.15, 0.9), strides=[8, 16, 32, 64, 100, 300], ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])
    featmap_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    expected_base_anchors = [torch.Tensor([[-6.5, -6.5, 14.5, 14.5], [-11.3704, -11.3704, 19.3704, 19.3704], [-10.8492, -3.4246, 18.8492, 11.4246], [-3.4246, -10.8492, 11.4246, 18.8492]]), torch.Tensor([[-14.5, -14.5, 30.5, 30.5], [-25.3729, -25.3729, 41.3729, 41.3729], [-23.8198, -7.9099, 39.8198, 23.9099], [-7.9099, -23.8198, 23.9099, 39.8198], [-30.9711, -4.9904, 46.9711, 20.9904], [-4.9904, -30.9711, 20.9904, 46.9711]]), torch.Tensor([[-33.5, -33.5, 65.5, 65.5], [-45.5366, -45.5366, 77.5366, 77.5366], [-54.0036, -19.0018, 86.0036, 51.0018], [-19.0018, -54.0036, 51.0018, 86.0036], [-69.7365, -12.5788, 101.7365, 44.5788], [-12.5788, -69.7365, 44.5788, 101.7365]]), torch.Tensor([[-44.5, -44.5, 108.5, 108.5], [-56.9817, -56.9817, 120.9817, 120.9817], [-76.1873, -22.0937, 140.1873, 86.0937], [-22.0937, -76.1873, 86.0937, 140.1873], [-100.5019, -12.1673, 164.5019, 76.1673], [-12.1673, -100.5019, 76.1673, 164.5019]]), torch.Tensor([[-53.5, -53.5, 153.5, 153.5], [-66.2185, -66.2185, 166.2185, 166.2185], [-96.3711, -23.1855, 196.3711, 123.1855], [-23.1855, -96.3711, 123.1855, 196.3711]]), torch.Tensor([[19.5, 19.5, 280.5, 280.5], [6.6342, 6.6342, 293.3658, 293.3658], [-34.5549, 57.7226, 334.5549, 242.2774], [57.7226, -34.5549, 242.2774, 334.5549]])]
    base_anchors = anchor_generator.base_anchors
    for i, base_anchor in enumerate(base_anchors):
        assert base_anchor.allclose(expected_base_anchors[i])
    expected_valid_pixels = [5776, 2166, 600, 150, 36, 4]
    multi_level_valid_flags = anchor_generator.valid_flags(featmap_sizes, (300, 300), device)
    for i, single_level_valid_flag in enumerate(multi_level_valid_flags):
        assert single_level_valid_flag.sum() == expected_valid_pixels[i]
    assert anchor_generator.num_base_anchors == [4, 6, 6, 6, 4, 4]
    anchors = anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(anchors) == 6

def test_anchor_generator_with_tuples():
    from mmdet.core.anchor import build_anchor_generator
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    anchor_generator_cfg = dict(type='SSDAnchorGenerator', scale_major=False, input_size=300, basesize_ratio_range=(0.15, 0.9), strides=[8, 16, 32, 64, 100, 300], ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])
    featmap_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    anchors = anchor_generator.grid_anchors(featmap_sizes, device)
    anchor_generator_cfg_tuples = dict(type='SSDAnchorGenerator', scale_major=False, input_size=300, basesize_ratio_range=(0.15, 0.9), strides=[(8, 8), (16, 16), (32, 32), (64, 64), (100, 100), (300, 300)], ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]])
    anchor_generator_tuples = build_anchor_generator(anchor_generator_cfg_tuples)
    anchors_tuples = anchor_generator_tuples.grid_anchors(featmap_sizes, device)
    for anchor, anchor_tuples in zip(anchors, anchors_tuples):
        assert torch.equal(anchor, anchor_tuples)

def test_yolo_anchor_generator():
    from mmdet.core.anchor import build_anchor_generator
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    anchor_generator_cfg = dict(type='YOLOAnchorGenerator', strides=[32, 16, 8], base_sizes=[[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]])
    featmap_sizes = [(14, 18), (28, 36), (56, 72)]
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    expected_base_anchors = [torch.Tensor([[-42.0, -29.0, 74.0, 61.0], [-62.0, -83.0, 94.0, 115.0], [-170.5, -147.0, 202.5, 179.0]]), torch.Tensor([[-7.0, -22.5, 23.0, 38.5], [-23.0, -14.5, 39.0, 30.5], [-21.5, -51.5, 37.5, 67.5]]), torch.Tensor([[-1.0, -2.5, 9.0, 10.5], [-4.0, -11.0, 12.0, 19.0], [-12.5, -7.5, 20.5, 15.5]])]
    base_anchors = anchor_generator.base_anchors
    for i, base_anchor in enumerate(base_anchors):
        assert base_anchor.allclose(expected_base_anchors[i])
    assert anchor_generator.num_base_anchors == [3, 3, 3]
    anchors = anchor_generator.grid_anchors(featmap_sizes, device)
    assert len(anchors) == 3

