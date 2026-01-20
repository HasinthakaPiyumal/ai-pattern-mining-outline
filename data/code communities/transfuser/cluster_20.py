# Cluster 20

def lidar_bev_cam_correspondences(world, lidar_vis=None, image_vis=None, step=None, debug=False):
    """
    Convert LiDAR point cloud to camera co-ordinates

    world: Expects the point cloud from CARLA in the CARLA coordinate system: x left, y forward, z up (LiDAR rotated by 90 degree)
    lidar_vis: lidar prjected to BEV
    image_vis: RGB input image to the network
    step: current timestep
    debug: Whether to save the debug images. If false only world is required
    """
    pixels_per_meter = 8
    lidar_width = 256
    lidar_height = 256
    lidar_meters_x = lidar_width / pixels_per_meter / 2
    lidar_meters_y = lidar_height / pixels_per_meter
    downscale_factor = 32
    img_width = 352
    img_height = 160
    fov_width = 60
    left_camera_rotation = -60.0
    right_camera_rotation = 60.0
    fov_height = 2.0 * np.arctan(img_height / img_width * np.tan(0.5 * np.radians(fov_width)))
    fov_height = np.rad2deg(fov_height)
    focal_x = img_width / (2.0 * np.tan(np.deg2rad(fov_width) / 2.0))
    focal_y = img_height / (2.0 * np.tan(np.deg2rad(fov_height) / 2.0))
    cam_z = 2.3
    lidar_z = 2.5
    world[:, 0] *= -1
    lidar = world[abs(world[:, 0]) < lidar_meters_x]
    lidar = lidar[lidar[:, 1] < lidar_meters_y]
    lidar = lidar[lidar[:, 1] > 0]
    lidar[..., 2] = lidar[..., 2] + (lidar_z - cam_z)
    lidar_for_left_camera = deepcopy(lidar)
    lidar_for_right_camera = deepcopy(lidar)
    lidar_indices = np.arange(0, lidar.shape[0], 1)
    z = lidar[..., 1]
    x = focal_x * lidar[..., 0] / z + img_width / 2.0
    y = focal_y * lidar[..., 2] / z + img_height / 2.0
    result_center = np.stack([x, y, lidar_indices], 1)
    result_center = result_center[np.logical_and(result_center[..., 0] > 0, result_center[..., 0] < img_width)]
    result_center = result_center[np.logical_and(result_center[..., 1] > 0, result_center[..., 1] < img_height)]
    result_center_shifted = result_center
    result_center_shifted[..., 0] = result_center_shifted[..., 0] + img_width / 2.0
    theta = np.radians(left_camera_rotation)
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    lidar_for_left_camera = R.dot(lidar_for_left_camera.T).T
    z = lidar_for_left_camera[..., 1]
    x = focal_x * lidar_for_left_camera[..., 0] / z + img_width / 2.0
    y = focal_y * lidar_for_left_camera[..., 2] / z + img_height / 2.0
    result_left = np.stack([x, y, lidar_indices], 1)
    result_left = result_left[np.logical_and(result_left[..., 0] > 0, result_left[..., 0] < img_width)]
    result_left = result_left[np.logical_and(result_left[..., 1] > 0, result_left[..., 1] < img_height)]
    result_left_shifted = result_left[result_left[..., 0] >= img_width / 2.0]
    result_left_shifted[..., 0] = result_left_shifted[..., 0] - img_width / 2.0
    theta = np.radians(right_camera_rotation)
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    lidar_for_right_camera = R.dot(lidar_for_right_camera.T).T
    z = lidar_for_right_camera[..., 1]
    x = focal_x * lidar_for_right_camera[..., 0] / z + img_width / 2.0
    y = focal_y * lidar_for_right_camera[..., 2] / z + img_height / 2.0
    result_right = np.stack([x, y, lidar_indices], 1)
    result_right = result_right[np.logical_and(result_right[..., 0] > 0, result_right[..., 0] < img_width)]
    result_right = result_right[np.logical_and(result_right[..., 1] > 0, result_right[..., 1] < img_height)]
    result_right_shifted = result_right[result_right[..., 0] < img_width / 2.0]
    result_right_shifted[..., 0] = result_right_shifted[..., 0] + img_width / 2.0 + img_width
    results_total = np.concatenate((result_left_shifted, result_center_shifted, result_right_shifted), axis=0)
    if debug == True:
        vis = np.zeros([img_height, 2 * img_width])
        vis_bev = np.zeros([lidar_height, lidar_width])
        vis_original_image = image_vis[0].detach().cpu().numpy()
        vis_original_image = np.transpose(vis_original_image, (1, 2, 0)) / 255.0
        vis_original_lidar = np.zeros([lidar_height, lidar_width])
        lidar_vis = lidar_vis.detach().cpu().numpy()
        vis_original_lidar[np.greater(lidar_vis[0, 0], 0)] = 255
        vis_original_lidar[np.greater(lidar_vis[0, 1], 0)] = 255
    valid_bev_points = []
    valid_cam_points = []
    for i in range(results_total.shape[0]):
        lidar_index = int(results_total[i, 2])
        bev_x = int((lidar[lidar_index][0] + lidar_meters_x) * pixels_per_meter)
        bev_y = (int(lidar[lidar_index][1] * pixels_per_meter) - (lidar_height - 1)) * -1
        valid_bev_points.append([bev_x, bev_y])
        img_x = int(results_total[i][0])
        img_y = (int(results_total[i][1]) - (img_height - 1)) * -1
        valid_cam_points.append([img_x, img_y])
        if debug == True:
            vis_original_image[img_y, img_x] = np.array([0.0, 1.0, 0.0])
            vis_bev[bev_y, bev_x] = 255
            vis[img_y, img_x] = 255
    if debug == True:
        from matplotlib import pyplot as plt
        plt.ion()
        plt.imshow(vis_bev)
        plt.savefig('/home/hiwi/save folder/Visualizations/2/bev_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.imshow(vis_original_image)
        plt.savefig('/home/hiwi/save folder/Visualizations/2/image_with_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.ioff()
    valid_bev_points = np.array(valid_bev_points)
    valid_cam_points = np.array(valid_cam_points)
    bev_points, cam_points = correspondences_at_one_scale(valid_bev_points, valid_cam_points, lidar_width // downscale_factor, lidar_height // downscale_factor, img_width // downscale_factor * 2, img_height // downscale_factor, downscale_factor)
    return (bev_points, cam_points)

def correspondences_at_one_scale(valid_bev_points, valid_cam_points, lidar_x, lidar_y, camera_x, camera_y, scale):
    """
    Compute projections between LiDAR BEV and image space
    """
    cam_to_bev_proj_locs = np.zeros((lidar_x, lidar_y, 5, 2))
    bev_to_cam_proj_locs = np.zeros((camera_x, camera_y, 5, 2))
    tmp_bev = np.empty((lidar_x, lidar_y), dtype=object)
    tmp_cam = np.empty((camera_x, camera_y), dtype=object)
    for i in range(lidar_x):
        for j in range(lidar_y):
            tmp_bev[i, j] = []
    for i in range(camera_x):
        for j in range(camera_y):
            tmp_cam[i, j] = []
    for i in range(valid_bev_points.shape[0]):
        tmp_bev[valid_bev_points[i][0] // scale, valid_bev_points[i][1] // scale].append(valid_cam_points[i] // scale)
        tmp_cam[valid_cam_points[i][0] // scale, valid_cam_points[i][1] // scale].append(valid_bev_points[i] // scale)
    for i in range(lidar_x):
        for j in range(lidar_y):
            cam_to_bev_points = tmp_bev[i, j]
            if len(cam_to_bev_points) > 5:
                cam_to_bev_proj_locs[i, j] = np.array(random.sample(cam_to_bev_points, 5))
            elif len(cam_to_bev_points) > 0:
                num_points = len(cam_to_bev_points)
                cam_to_bev_proj_locs[i, j, :num_points] = np.array(cam_to_bev_points)
    for i in range(camera_x):
        for j in range(camera_y):
            bev_to_cam_points = tmp_cam[i, j]
            if len(bev_to_cam_points) > 5:
                bev_to_cam_proj_locs[i, j] = np.array(random.sample(bev_to_cam_points, 5))
            elif len(bev_to_cam_points) > 0:
                num_points = len(bev_to_cam_points)
                bev_to_cam_proj_locs[i, j, :num_points] = np.array(bev_to_cam_points)
    return (cam_to_bev_proj_locs, bev_to_cam_proj_locs)

