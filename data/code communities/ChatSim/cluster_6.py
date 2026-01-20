# Cluster 6

def load_colmap_sparse_points(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)

    """

    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character='<'):
        """Read and unpack the next bytes from a binary file.
        :param fid:
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)
    with open(path_to_model_file, 'rb') as fid:
        num_points = read_next_bytes(fid, 8, 'Q')[0]
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence='QdddBBBd')
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence='ii' * track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    points = OrderedDict()
    points['xyz'] = xyzs
    points['rgb'] = rgbs
    points['error'] = errors
    return points

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character='<'):
    """Read and unpack the next bytes from a binary file.
        :param fid:
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_xml_save_npy(data_dir):
    """
    We will save `cams_meta_metashape.npy` (RUB convention)
    not `poses_bounds_metashape.npy` (DRB) now.
    """
    print('Parsing Metashape results')
    intrinsic, (width, height), dist_params = intrinsics_from_xml(os.path.join(data_dir, 'camera.xml'))
    poses_RDF, labels_sort = extrinsics_from_xml(os.path.join(data_dir, 'camera.xml'))
    poses_RDF = np.stack(poses_RDF, axis=0)
    poses_RUB = np.concatenate((poses_RDF[:, :, 0:1], -poses_RDF[:, :, 1:2], -poses_RDF[:, :, 2:3], poses_RDF[:, :, 3:]), axis=-1)
    poses_RUB = poses_RUB[:, :3, :]
    N = poses_RUB.shape[0]
    intrinsic = intrinsic.reshape(1, 3, 3).repeat(N, axis=0)
    dist_params = np.array(dist_params).reshape(1, 4).repeat(N, axis=0)
    bounds = np.array([0.1, 999]).reshape(1, 2).repeat(N, axis=0)
    cams_meta = np.concatenate([poses_RUB.reshape(N, -1), intrinsic.reshape(N, -1), dist_params.reshape(N, -1), bounds.reshape(N, -1)], axis=1)
    np.save(os.path.join(data_dir, 'cams_meta_metashape.npy'), cams_meta)

def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    f = float(calibration.find('f').text)
    cx = width / 2
    cy = height / 2
    dist_params = (0.0, 0.0, 0.0, 0.0)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    return (K, (width, height), dist_params)

def extrinsics_from_xml(xml_file, verbose=False):
    """
    Metashape return RDF convention camera poses
    """
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):
        label = e.get('label')
        try:
            transforms[label] = e.find('transform').text
        except:
            if verbose:
                print('failed to align camera', label)
    view_matrices = []
    labels_sort = list(transforms)
    for label in labels_sort:
        extrinsic = np.array([float(x) for x in transforms[label].split()]).reshape(4, 4)
        view_matrices.append(extrinsic)
    return (view_matrices, labels_sort)

def align(data_dir, src_cams_meta='cams_meta_metashape.npy', dst_cams_meta='cams_meta_waymo.npy'):
    if src_cams_meta == 'cams_meta_metashape.npy':
        print("Aligning Metashape's coordinates with Waymo's coordinates")
    elif src_cams_meta == 'cams_meta_colmap.npy':
        print("Aligning Colmap's coordinates with Waymo's coordinates")
    cams_meta_data_source = np.load(os.path.join(data_dir, src_cams_meta))
    cams_meta_data_target = np.load(os.path.join(data_dir, dst_cams_meta))
    extrinsic_source = cams_meta_data_source[:, :12].reshape(-1, 3, 4)
    last_row = np.zeros((extrinsic_source.shape[0], 1, 4))
    last_row[:, :, -1] = 1
    extrinsic_source = np.concatenate((extrinsic_source, last_row), axis=1)
    extrinsic_target = cams_meta_data_target[:, :12].reshape(-1, 3, 4)
    last_row = np.zeros((extrinsic_target.shape[0], 1, 4))
    last_row[:, :, -1] = 1
    extrinsic_target = np.concatenate((extrinsic_target, last_row), axis=1)
    scale = np.linalg.norm(extrinsic_source[3, :3, -1] - extrinsic_source[0, :3, -1]) / np.linalg.norm(extrinsic_target[3, :3, -1] - extrinsic_target[0, :3, -1])
    rotate_0_target = extrinsic_target[0, :3, :3]
    rotate_0_source = extrinsic_source[0, :3, :3]
    rotate_source_world_to_target_world = rotate_0_target @ np.linalg.inv(rotate_0_source)
    rotate_source_world_to_target_world = rotate_source_world_to_target_world[None, ...]
    extrinsic_results = np.zeros_like(extrinsic_source)
    extrinsic_results[:, :3, :3] = rotate_source_world_to_target_world @ extrinsic_source[:, :3, :3]
    delta_translation_in_source_world = extrinsic_source[:, :3, -1:] - extrinsic_source[0:1, :3, -1:]
    delta_translation_in_target_world = rotate_source_world_to_target_world @ delta_translation_in_source_world / scale
    extrinsic_results[:, :3, -1:] = delta_translation_in_target_world + extrinsic_target[0:1, :3, -1:]
    extrinsic_results[:, -1, -1] = 1
    cams_meta_data_source[:, :12] = extrinsic_results[:, :3, :].reshape(-1, 12)
    data = np.ascontiguousarray(np.array(cams_meta_data_source).astype(np.float64))
    if src_cams_meta == 'cams_meta_metashape.npy':
        print(f'\n{colored('[Imporant]', 'green', attrs=['bold'])} save to cams_meta.npy')
        np.save(os.path.join(data_dir, 'cams_meta.npy'), data)
    if src_cams_meta == 'cams_meta_colmap.npy':
        print(f'\n{colored('[Imporant]', 'green', attrs=['bold'])} Save to colmap/sparse_undistorted/cams_meta.npy')
        print(f'cams_meta.npy from metashape (in the root folder) will not be overwritten.')
        np.save(os.path.join(data_dir, 'colmap/sparse_undistorted/cams_meta.npy'), data)
        src_point3D_path = os.path.join(data_dir, 'colmap/sparse_undistorted/sparse/points3D.bin')
        dst_point3D_path = os.path.join(data_dir, 'colmap/sparse_undistorted/points3D_waymo.ply')
        points = load_colmap_sparse_points(src_point3D_path)
        points3D = points['xyz']
        points3D_colors = points['rgb']
        delta_translation_in_source_world = points3D - np.expand_dims(extrinsic_source[0, :3, -1], axis=0)
        delta_translation_in_source_world = delta_translation_in_source_world[..., np.newaxis]
        delta_translation_in_target_world = rotate_source_world_to_target_world @ delta_translation_in_source_world / scale
        translation_0_target = extrinsic_target[0:1, :3, -1:]
        points3D_in_target_world = delta_translation_in_target_world + translation_0_target
        sfm_points = np.squeeze(points3D_in_target_world)
        sfm_colors = points3D_colors / 255.0
        lidar_open3d = o3d.io.read_point_cloud(os.path.join(data_dir, 'point_cloud/000_TOP.ply'))
        lidar_points = np.array(lidar_open3d.points)
        lidar_colors = np.full(lidar_points.shape, 0.3)
        mask = lidar_points[:, 0] > 0
        lidar_points = lidar_points[mask]
        lidar_colors = lidar_colors[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate([sfm_points, lidar_points], axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate([sfm_colors, lidar_colors], axis=0))
        o3d.io.write_point_cloud(dst_point3D_path, pcd)

def read_sparse_save_npy(data_dir):
    print('Parsing Colmap results to cams_meta_colmap.npy')
    dataset = Colamp_Dataset(data_dir)
    dataset.export(data_dir)

def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, 'rb') as fid:
        num_cameras = read_next_bytes(fid, 8, 'Q')[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence='iiQQ')
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence='d' * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, 'rb') as fid:
        num_reg_images = read_next_bytes(fid, 8, 'Q')[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ''
            current_char = read_next_bytes(fid, 1, 'c')[0]
            while current_char != b'\x00':
                image_name += current_char.decode('utf-8')
                current_char = read_next_bytes(fid, 1, 'c')[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence='ddq' * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, 'rb') as fid:
        num_points = read_next_bytes(fid, 8, 'Q')[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence='QdddBBBd')
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence='ii' * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs)
    return points3D

def read_model(path, ext):
    if ext == '.txt':
        cameras = read_cameras_text(os.path.join(path, 'cameras' + ext))
        images = read_images_text(os.path.join(path, 'images' + ext))
        points3D = read_points3D_text(os.path.join(path, 'points3D') + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, 'cameras' + ext))
        images = read_images_binary(os.path.join(path, 'images' + ext))
        points3D = read_points3d_binary(os.path.join(path, 'points3D') + ext)
    return (cameras, images, points3D)

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, 'r') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, 'r') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def main():
    if len(sys.argv) != 3:
        print('Usage: python read_model.py path/to/model/folder [.txt,.bin]')
        return
    cameras, images, points3D = read_model(path=sys.argv[1], ext=sys.argv[2])
    print('num_cameras:', len(cameras))
    print('num_images:', len(images))
    print('num_points3D:', len(points3D))

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, 'rb') as fid:
        num_points = read_next_bytes(fid, 8, 'Q')[0]
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence='QdddBBBd')
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence='ii' * track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return (xyzs, rgbs, errors)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, 'rb') as fid:
        num_reg_images = read_next_bytes(fid, 8, 'Q')[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ''
            current_char = read_next_bytes(fid, 1, 'c')[0]
            while current_char != b'\x00':
                image_name += current_char.decode('utf-8')
                current_char = read_next_bytes(fid, 1, 'c')[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence='ddq' * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, 'rb') as fid:
        num_cameras = read_next_bytes(fid, 8, 'Q')[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence='iiQQ')
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence='d' * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

