# Cluster 91

def compose(rendered_output_dir, background_RGB, background_depth, render_downsample, motion_blur_degree, depth_and_occlusion):
    """
    Args:
        rendered_output_dir: str
            the folder that saves RGB/depth/mask for vehicle w/wo plane
        background_RGB: np.ndarray
            background image
        background_depth: np.ndarray
            background depth image
    """
    if depth_and_occlusion == False:
        RGB_over_background = read_from_render(rendered_output_dir, 'RGB', 'vehicle_and_shadow_over_background')
        H, W = RGB_over_background.shape[:2]
        RGB_over_background = cv2.resize(RGB_over_background, (render_downsample * W, render_downsample * H))
        RGB_over_background = RGB_over_background[:, :, :3]
        RGB_over_background = motion_blur(RGB_over_background, degree=motion_blur_degree, angle=45)
        imageio.imsave(os.path.join(rendered_output_dir, 'RGB_composite.png'), RGB_over_background)
        return
    depth_vp = read_from_render(rendered_output_dir, 'depth', 'vehicle_and_plane')
    depth_vp = expand_depth(depth_vp).astype(np.float32)
    RGB_over_background = read_from_render(rendered_output_dir, 'RGB', 'vehicle_and_shadow_over_background')
    mask_vs = read_from_render(rendered_output_dir, 'mask', 'vehicle_and_shadow')
    H, W = depth_vp.shape[:2]
    depth_vp = cv2.resize(depth_vp, (render_downsample * W, render_downsample * H))
    RGB_over_background = cv2.resize(RGB_over_background, (render_downsample * W, render_downsample * H))
    RGB_over_background = RGB_over_background[:, :, :3]
    RGB_over_background = motion_blur(RGB_over_background, degree=motion_blur_degree, angle=45)
    mask_vs = cv2.resize(mask_vs, (render_downsample * W, render_downsample * H))
    depth_vp = depth_vp[:, :, 0:1]
    mask_vs = mask_vs[:, :, 0:1]
    noise_thres = 0.02
    depth_vs = np.where(mask_vs > noise_thres, depth_vp, np.inf)
    RGB_composite = np.where(depth_vs <= background_depth[..., np.newaxis], RGB_over_background, background_RGB).astype(np.uint8)
    imageio.imsave(os.path.join(rendered_output_dir, 'RGB_composite.png'), RGB_composite)
    imageio.imsave(os.path.join(rendered_output_dir, 'RGB_over_background.png'), RGB_over_background)

def read_from_render(rendered_output_dir, image_type, image_prefix):
    """
    Args:
        image_type: str,
            RGB/depth/mask
        image_prefix: str,
            vehicle_only/vehicle_and_plane/plane_only
    """
    files = glob.glob(os.path.join(rendered_output_dir, image_type) + f'/{image_prefix}*')
    assert len(files) == 1
    return imageio.imread(files[0])

def motion_blur(image, degree=12, angle=45):
    """
    degree : intensity of blur
    angle : direction of blur. 
        angle = 0 is +u, angle = 90 is -v
    """
    image = np.array(image)
    angle -= 135
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def expand_depth(depth_vp, neighbor_size=1, bg_value=10000000000.0):
    """
        depth_vp: np.ndarray
            shape: [H, W, 1]
        neighbor_size: int
            kernel size, but not used currently
    """
    bg_pos = depth_vp == bg_value
    neighbor_depth = np.zeros((depth_vp.shape[0], depth_vp.shape[1], 5))
    neighbor_depth[:, :, 0] = depth_vp[:, :, 0]
    for idx, axis_shift in enumerate([(0, 1), (0, -1), (1, 1), (1, -1)]):
        axis, shift = axis_shift
        depth_vp_new = np.roll(depth_vp[:, :, 0], axis=axis, shift=shift)
        if axis == 0:
            target = 0 if shift == 1 else -1
            depth_vp_new[target] = bg_value
        if axis == 1:
            target = 0 if shift == 1 else -1
            depth_vp_new[:, target] = bg_value
        neighbor_depth[:, :, idx + 1] = depth_vp_new
    bg_neighbor_depth = np.min(neighbor_depth, axis=2)[..., np.newaxis]
    return bg_pos * bg_neighbor_depth + (1 - bg_pos) * depth_vp

