# Cluster 156

def split_imageset(original_dataroot, taget_dataroot, n_channels=3, p_size=800, p_overlap=96, p_max=1000):
    """
    split the large images from original_dataroot into small overlapped images with size (p_size)x(p_size),
    and save them into taget_dataroot; only the images with larger size than (p_max)x(p_max)
    will be splitted.
    Args:
        original_dataroot:
        taget_dataroot:
        p_size: size of small images
        p_overlap: patch size in training is a good choice
        p_max: images with smaller size than (p_max)x(p_max) keep unchanged.
    """
    paths = get_image_paths(original_dataroot)
    for img_path in paths:
        img = imread_uint(img_path, n_channels=n_channels)
        patches = patches_from_image(img, p_size, p_overlap, p_max)
        imssave(patches, os.path.join(taget_dataroot, os.path.basename(img_path)))

def get_image_paths(dataroot):
    paths = None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths

def patches_from_image(img, p_size=512, p_overlap=64, p_max=800):
    w, h = img.shape[:2]
    patches = []
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w - p_size, p_size - p_overlap, dtype=np.int))
        h1 = list(np.arange(0, h - p_size, p_size - p_overlap, dtype=np.int))
        w1.append(w - p_size)
        h1.append(h - p_size)
        for i in w1:
            for j in h1:
                patches.append(img[i:i + p_size, j:j + p_size, :])
    else:
        patches.append(img)
    return patches

def imssave(imgs, img_path):
    """
    imgs: list, N images of size WxHxC
    """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    for i, img in enumerate(imgs):
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]
        new_path = os.path.join(os.path.dirname(img_path), img_name + str('_s{:04d}'.format(i)) + '.png')
        cv2.imwrite(new_path, img)

