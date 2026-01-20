# Cluster 21

def compute_full_overlapping(data_root, scene_id, voxel_size=0.05):
    _points = [(pcd_name, make_open3d_point_cloud(torch.load(pcd_name)['coord'], voxel_size=voxel_size)) for pcd_name in glob.glob(os.path.join(data_root, scene_id, 'pcd', '*.pth'))]
    points = [(pcd_name, pcd) for pcd_name, pcd in _points if pcd is not None]
    print('load {} point clouds ({} invalid has been filtered), computing matching/overlapping'.format(len(points), len(_points) - len(points)))
    matching_matrix = np.zeros((len(points), len(points)))
    for i, (pcd0_name, pcd0) in enumerate(points):
        print('matching to...{}'.format(pcd0_name))
        pcd0_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd0))
        for j, (pcd1_name, pcd1) in enumerate(points):
            if i == j:
                continue
            matching_matrix[i, j] = float(len(get_matching_indices(pcd1, pcd0_tree, 1.5 * voxel_size, 1))) / float(len(pcd1.points))
    with open(os.path.join(data_root, scene_id, 'pcd', 'overlap.txt'), 'w') as f:
        for i, (pcd0_name, pcd0) in enumerate(points):
            for j, (pcd1_name, pcd1) in enumerate(points):
                if i < j:
                    overlap = max(matching_matrix[i, j], matching_matrix[j, i])
                    f.write('{} {} {}\n'.format(pcd0_name.replace(data_root, ''), pcd1_name.replace(data_root, ''), overlap))

def make_open3d_point_cloud(xyz, color=None, voxel_size=None):
    if np.isnan(xyz).any():
        return None
    xyz = xyz[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd

def get_matching_indices(source, pcd_tree, search_voxel_size, K=None):
    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds

def parse_sens(sens_dir, output_dir):
    scene_id = os.path.basename(os.path.dirname(sens_dir))
    print(f'Parsing sens data{sens_dir}')
    reader(sens_dir, os.path.join(output_dir, scene_id), frame_skip, export_color_images=True, export_depth_images=True, export_poses=True, export_intrinsics=True)
    extractor(os.path.join(output_dir, scene_id), os.path.join(output_dir, scene_id, 'pcd'))
    compute_full_overlapping(output_dir, scene_id)

def reader(filename, output_path, frame_skip, export_color_images=False, export_depth_images=False, export_poses=False, export_intrinsics=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('loading %s...' % filename)
    sd = SensorData(filename)
    if export_depth_images:
        sd.export_depth_images(os.path.join(output_path, 'depth'), frame_skip=frame_skip)
    if export_color_images:
        sd.export_color_images(os.path.join(output_path, 'color'), frame_skip=frame_skip)
    if export_poses:
        sd.export_poses(os.path.join(output_path, 'pose'), frame_skip=frame_skip)
    if export_intrinsics:
        sd.export_intrinsics(os.path.join(output_path, 'intrinsic'))

def extractor(input_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    depth_intrinsic = np.loadtxt(input_path + '/intrinsic/intrinsic_depth.txt')
    print('Depth intrinsic: ')
    print(depth_intrinsic)
    poses = sorted(glob.glob(input_path + '/pose/*.txt'), key=lambda a: int(os.path.basename(a).split('.')[0]))
    depths = sorted(glob.glob(input_path + '/depth/*.png'), key=lambda a: int(os.path.basename(a).split('.')[0]))
    colors = sorted(glob.glob(input_path + '/color/*.png'), key=lambda a: int(os.path.basename(a).split('.')[0]))
    for ind, (pose, depth, color) in enumerate(zip(poses, depths, colors)):
        name = os.path.basename(pose).split('.')[0]
        if os.path.exists(output_path + '/{}.npz'.format(name)):
            continue
        try:
            print('=' * 50, ': {}'.format(pose))
            depth_img = cv2.imread(depth, -1)
            mask = depth_img != 0
            color_image = cv2.imread(color)
            color_image = cv2.resize(color_image, (640, 480))
            color_image = np.reshape(color_image[mask], [-1, 3])
            colors = np.zeros_like(color_image)
            colors[:, 0] = color_image[:, 2]
            colors[:, 1] = color_image[:, 1]
            colors[:, 2] = color_image[:, 0]
            pose = np.loadtxt(poses[ind])
            print('Camera pose: ')
            print(pose)
            depth_shift = 1000.0
            x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]), np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
            uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
            uv_depth[:, :, 0] = x
            uv_depth[:, :, 1] = y
            uv_depth[:, :, 2] = depth_img / depth_shift
            uv_depth = np.reshape(uv_depth, [-1, 3])
            uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()
            intrinsic_inv = np.linalg.inv(depth_intrinsic)
            fx = depth_intrinsic[0, 0]
            fy = depth_intrinsic[1, 1]
            cx = depth_intrinsic[0, 2]
            cy = depth_intrinsic[1, 2]
            bx = depth_intrinsic[0, 3]
            by = depth_intrinsic[1, 3]
            point_list = []
            n = uv_depth.shape[0]
            points = np.ones((n, 4))
            X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
            Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
            points[:, 0] = X
            points[:, 1] = Y
            points[:, 2] = uv_depth[:, 2]
            points_world = np.dot(points, np.transpose(pose))
            print(points_world.shape)
            pcd = dict(coord=points_world[:, :3], color=colors)
            torch.save(pcd, output_path + '/{}.pth'.format(name))
        except:
            continue

