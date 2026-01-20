# Cluster 46

def show_img_with_mask(img, mask):
    if np.max(mask) == 1:
        mask = np.uint8(mask * 255)
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_mask(plt.gca(), mask, random_color=False)
    tmp_p = mkstemp('.png')
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f'{suffix}', dir=dir)
    os.close(fd)
    return Path(path)

def show_img_with_point(img, point_coords, point_labels):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_points(plt.gca(), point_coords, point_labels, size=(width * 0.04) ** 2)
    tmp_p = mkstemp('.png')
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_box(img, box):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(width / dpi / 0.77, height / dpi / 0.77))
    ax.imshow(img)
    ax.axis('off')
    x1, y1, w, h = box
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    tmp_p = mkstemp('.png')
    fig.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_mask(img, mask):
    if np.max(mask) == 1:
        mask = np.uint8(mask * 255)
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_mask(plt.gca(), mask, random_color=False)
    tmp_p = mkstemp('.png')
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_point(img, point_coords, point_labels):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_points(plt.gca(), point_coords, point_labels, size=(width * 0.04) ** 2)
    tmp_p = mkstemp('.png')
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_box(img, box):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(width / dpi / 0.77, height / dpi / 0.77))
    ax.imshow(img)
    ax.axis('off')
    x1, y1, w, h = box
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    tmp_p = mkstemp('.png')
    fig.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size):
    point_coords = [w, h]
    point_labels = [1]
    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w
    print(point_coords)
    masks, _, _ = model['sam'].predict(point_coords=np.array([point_coords]), point_labels=np.array(point_labels), multimask_output=True)
    masks = masks.astype(np.uint8) * 255
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    else:
        masks = [mask for mask in masks]
    figs = []
    for idx, mask in enumerate(masks):
        tmp_p = mkstemp('.png')
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [point_coords], point_labels, size=(width * 0.04) ** 2)
        show_mask(plt.gca(), mask, random_color=False)
        plt.tight_layout()
        plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
        figs.append(fig)
        plt.close()
    return (*figs, *masks)

