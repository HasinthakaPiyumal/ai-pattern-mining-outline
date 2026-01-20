# Cluster 9

def calibread(file_path):
    out = dict()
    for line in open(file_path, 'r'):
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        val = line.split(':')
        assert len(val) == 2, 'Wrong file format, only one : per line!'
        key_name = val[0].strip()
        val = np.asarray(val[-1].strip().split(' '), dtype='f8')
        assert len(val) in [12, 9], 'Wrong file format, wrong number of numbers!'
        if len(val) == 12:
            out[key_name] = val.reshape(3, 4)
        elif len(val) == 9:
            out[key_name] = val.reshape(3, 3)
    return out

class DatasetKittiTest(torch.utils.data.Dataset):

    def __init__(self, kitti_data_path, kitti_meta_path, NH):
        self.img_dir = kitti_data_path + '/object/testing/image_2/'
        self.calib_dir = kitti_data_path + '/object/testing/calib/'
        self.lidar_dir = kitti_data_path + '/object/testing/velodyne/'
        self.detections_2d_dir = kitti_meta_path + '/object/testing/2d_detections/'
        self.NH = NH
        with open(kitti_meta_path + '/kitti_train_mean_car_size.pkl', 'rb') as file:
            self.mean_car_size = pickle.load(file)
        with open(kitti_meta_path + '/kitti_centered_frustum_mean_xyz.pkl', 'rb') as file:
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)
        img_ids = []
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            img_id = img_name.split('.png')[0]
            img_ids.append(img_id)
        self.examples = []
        for img_id in img_ids:
            detections_file_path = self.detections_2d_dir + img_id + '.txt'
            with open(detections_file_path) as file:
                for line in file:
                    values = line.split()
                    object_class = float(values[3])
                    if object_class == 1:
                        u_min = float(values[4])
                        v_min = float(values[5])
                        u_max = float(values[6])
                        v_max = float(values[7])
                        score_2d = float(values[8])
                        detection_2d = {}
                        detection_2d['u_min'] = u_min
                        detection_2d['v_min'] = v_min
                        detection_2d['u_max'] = u_max
                        detection_2d['v_max'] = v_max
                        detection_2d['score_2d'] = score_2d
                        detection_2d['img_id'] = img_id
                        self.examples.append(detection_2d)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        lidar_path = self.lidar_dir + img_id + '.bin'
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        orig_point_cloud = point_cloud
        point_cloud = point_cloud[point_cloud[:, 0] > -5, :]
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]
        calib = calibread(self.calib_dir + img_id + '.txt')
        P2 = calib['P2']
        Tr_velo_to_cam_orig = calib['Tr_velo_to_cam']
        R0_rect_orig = calib['R0_rect']
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig
        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3]
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0] / img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1] / img_points_hom[:, 2]
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera
        u_min = example['u_min']
        u_max = example['u_max']
        v_min = example['v_min']
        v_max = example['v_max']
        score_2d = example['score_2d']
        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])
        row_mask = np.logical_and(np.logical_and(img_points[:, 0] >= u_min, img_points[:, 0] <= u_max), np.logical_and(img_points[:, 1] >= v_min, img_points[:, 1] <= v_max))
        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :]
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]
        empty_frustum_flag = 0
        if frustum_point_cloud.shape[0] == 0:
            empty_frustum_flag = 1
            frustum_point_cloud = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz = np.zeros((1024, 3), dtype=np.float32)
            frustum_point_cloud_camera = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz_camera = np.zeros((1024, 3), dtype=np.float32)
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        u_center = u_min + (u_max - u_min) / 2.0
        v_center = v_min + (v_max - v_min) / 2.0
        center_img_point_hom = np.array([u_center, v_center, 1])
        P2_pseudo_inverse = np.linalg.pinv(P2)
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)
        point = np.array([point_hom[0] / point_hom[3], point_hom[1] / point_hom[3], point_hom[2] / point_hom[3]])
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]
        frustum_angle = np.arctan2(point[0], point[2])
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)], [0, 1, 0], [np.sin(frustum_angle), 0, np.cos(frustum_angle)]], dtype='float32')
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz
        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera
        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera)
        return (centered_frustum_point_cloud_camera, img_id, input_2Dbbox, frustum_R, frustum_angle, empty_frustum_flag, self.centered_frustum_mean_xyz, self.mean_car_size, score_2d)

    def __len__(self):
        return self.num_examples

class DatasetKittiTestSequence(torch.utils.data.Dataset):

    def __init__(self, kitti_data_path, kitti_meta_path, NH, sequence):
        self.img_dir = kitti_data_path + '/tracking/testing/image_02/' + sequence + '/'
        self.lidar_dir = kitti_data_path + '/tracking/testing/velodyne/' + sequence + '/'
        self.calib_path = kitti_meta_path + '/tracking/testing/calib/' + sequence + '.txt'
        self.detections_2d_path = kitti_meta_path + '/tracking/testing/2d_detections/' + sequence + '/inferResult_1.txt'
        self.NH = NH
        with open(kitti_meta_path + '/kitti_train_mean_car_size.pkl', 'rb') as file:
            self.mean_car_size = pickle.load(file)
        with open(kitti_meta_path + '/kitti_centered_frustum_mean_xyz.pkl', 'rb') as file:
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)
        self.examples = []
        with open(self.detections_2d_path) as file:
            for line in file:
                values = line.split()
                object_class = float(values[3])
                if object_class == 1:
                    img_id = values[0]
                    u_min = float(values[4])
                    v_min = float(values[5])
                    u_max = float(values[6])
                    v_max = float(values[7])
                    score_2d = float(values[8])
                    detection_2d = {}
                    detection_2d['u_min'] = u_min
                    detection_2d['v_min'] = v_min
                    detection_2d['u_max'] = u_max
                    detection_2d['v_max'] = v_max
                    detection_2d['score_2d'] = score_2d
                    detection_2d['img_id'] = img_id
                    self.examples.append(detection_2d)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        lidar_path = self.lidar_dir + img_id + '.bin'
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        orig_point_cloud = point_cloud
        point_cloud = point_cloud[point_cloud[:, 0] > -5, :]
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]
        calib = calibread(self.calib_path)
        P2 = calib['P2']
        Tr_velo_to_cam_orig = calib['Tr_velo_to_cam']
        R0_rect_orig = calib['R0_rect']
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig
        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3]
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0] / img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1] / img_points_hom[:, 2]
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera
        u_min = example['u_min']
        u_max = example['u_max']
        v_min = example['v_min']
        v_max = example['v_max']
        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])
        row_mask = np.logical_and(np.logical_and(img_points[:, 0] >= u_min, img_points[:, 0] <= u_max), np.logical_and(img_points[:, 1] >= v_min, img_points[:, 1] <= v_max))
        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :]
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]
        empty_frustum_flag = 0
        if frustum_point_cloud.shape[0] == 0:
            empty_frustum_flag = 1
            frustum_point_cloud = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz = np.zeros((1024, 3), dtype=np.float32)
            frustum_point_cloud_camera = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz_camera = np.zeros((1024, 3), dtype=np.float32)
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        u_center = u_min + (u_max - u_min) / 2.0
        v_center = v_min + (v_max - v_min) / 2.0
        center_img_point_hom = np.array([u_center, v_center, 1])
        P2_pseudo_inverse = np.linalg.pinv(P2)
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)
        point = np.array([point_hom[0] / point_hom[3], point_hom[1] / point_hom[3], point_hom[2] / point_hom[3]])
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]
        frustum_angle = np.arctan2(point[0], point[2])
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)], [0, 1, 0], [np.sin(frustum_angle), 0, np.cos(frustum_angle)]], dtype='float32')
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz
        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera
        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera)
        return (centered_frustum_point_cloud_camera, img_id, input_2Dbbox, frustum_R, frustum_angle, empty_frustum_flag, self.centered_frustum_mean_xyz, self.mean_car_size)

    def __len__(self):
        return self.num_examples

class DatasetKittiVal2ddetections(torch.utils.data.Dataset):

    def __init__(self, kitti_data_path, kitti_meta_path, NH):
        self.img_dir = kitti_data_path + '/object/training/image_2/'
        self.calib_dir = kitti_data_path + '/object/training/calib/'
        self.lidar_dir = kitti_data_path + '/object/training/velodyne/'
        self.detections_2d_path = kitti_meta_path + '/rgb_detection_val.txt'
        self.NH = NH
        with open(kitti_meta_path + '/kitti_train_mean_car_size.pkl', 'rb') as file:
            self.mean_car_size = pickle.load(file)
        with open(kitti_meta_path + '/kitti_centered_frustum_mean_xyz.pkl', 'rb') as file:
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)
        self.examples = []
        with open(self.detections_2d_path) as file:
            for line in file:
                values = line.split()
                object_class = float(values[1])
                if object_class == 2:
                    score_2d = float(values[2])
                    u_min = float(values[3])
                    v_min = float(values[4])
                    u_max = float(values[5])
                    v_max = float(values[6])
                    img_id = values[0].split('image_2/')[1]
                    img_id = img_id.split('.')[0]
                    detection_2d = {}
                    detection_2d['u_min'] = u_min
                    detection_2d['v_min'] = v_min
                    detection_2d['u_max'] = u_max
                    detection_2d['v_max'] = v_max
                    detection_2d['score_2d'] = score_2d
                    detection_2d['img_id'] = img_id
                    self.examples.append(detection_2d)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        lidar_path = self.lidar_dir + img_id + '.bin'
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        orig_point_cloud = point_cloud
        point_cloud = point_cloud[point_cloud[:, 0] > -5, :]
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]
        calib = calibread(self.calib_dir + img_id + '.txt')
        P2 = calib['P2']
        Tr_velo_to_cam_orig = calib['Tr_velo_to_cam']
        R0_rect_orig = calib['R0_rect']
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig
        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3]
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0] / img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1] / img_points_hom[:, 2]
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera
        u_min = example['u_min']
        u_max = example['u_max']
        v_min = example['v_min']
        v_max = example['v_max']
        score_2d = example['score_2d']
        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])
        row_mask = np.logical_and(np.logical_and(img_points[:, 0] >= u_min, img_points[:, 0] <= u_max), np.logical_and(img_points[:, 1] >= v_min, img_points[:, 1] <= v_max))
        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :]
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]
        empty_frustum_flag = 0
        if frustum_point_cloud.shape[0] == 0:
            empty_frustum_flag = 1
            frustum_point_cloud = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz = np.zeros((1024, 3), dtype=np.float32)
            frustum_point_cloud_camera = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz_camera = np.zeros((1024, 3), dtype=np.float32)
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        u_center = u_min + (u_max - u_min) / 2.0
        v_center = v_min + (v_max - v_min) / 2.0
        center_img_point_hom = np.array([u_center, v_center, 1])
        P2_pseudo_inverse = np.linalg.pinv(P2)
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)
        point = np.array([point_hom[0] / point_hom[3], point_hom[1] / point_hom[3], point_hom[2] / point_hom[3]])
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]
        frustum_angle = np.arctan2(point[0], point[2])
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)], [0, 1, 0], [np.sin(frustum_angle), 0, np.cos(frustum_angle)]], dtype='float32')
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz
        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera
        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera)
        return (centered_frustum_point_cloud_camera, img_id, input_2Dbbox, frustum_R, frustum_angle, empty_frustum_flag, self.centered_frustum_mean_xyz, self.mean_car_size, score_2d)

    def __len__(self):
        return self.num_examples

class DatasetImgNetEvalTestSeq(torch.utils.data.Dataset):

    def __init__(self, kitti_data_path, kitti_meta_path, sequence='0000'):
        self.img_dir = kitti_data_path + '/tracking/testing/image_02/' + sequence + '/'
        self.lidar_dir = kitti_data_path + '/tracking/testing/velodyne/' + sequence + '/'
        self.calib_path = kitti_meta_path + '/tracking/testing/calib/' + sequence + '.txt'
        self.detections_2d_path = kitti_meta_path + '/tracking/testing/2d_detections/' + sequence + '/inferResult_1.txt'
        with open(kitti_meta_path + '/kitti_train_mean_car_size.pkl', 'rb') as file:
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)
        with open(kitti_meta_path + '/kitti_train_mean_distance.pkl', 'rb') as file:
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]
        self.examples = []
        with open(self.detections_2d_path) as file:
            for line in file:
                values = line.split()
                object_class = float(values[3])
                if object_class == 1:
                    img_id = values[0]
                    u_min = float(values[4])
                    v_min = float(values[5])
                    u_max = float(values[6])
                    v_max = float(values[7])
                    score_2d = float(values[8])
                    detection_2d = {}
                    detection_2d['u_min'] = u_min
                    detection_2d['v_min'] = v_min
                    detection_2d['u_max'] = u_max
                    detection_2d['v_max'] = v_max
                    detection_2d['score_2d'] = score_2d
                    detection_2d['img_id'] = img_id
                    self.examples.append(detection_2d)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        calib = calibread(self.calib_path)
        camera_matrix = calib['P2']
        u_min = example['u_min']
        u_max = example['u_max']
        v_min = example['v_min']
        v_max = example['v_max']
        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])
        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w / 2.0
        v_center = v_min + h / 2.0
        img_path = self.img_dir + img_id + '.png'
        img = cv2.imread(img_path, -1)
        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))
        bbox_2d_img = bbox_2d_img / 255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img / np.array([0.229, 0.224, 0.225])
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1))
        bbox_2d_img = bbox_2d_img.astype(np.float32)
        bbox_2d_img = torch.from_numpy(bbox_2d_img)
        return (bbox_2d_img, img_id, self.mean_car_size, w, h, u_center, v_center, input_2Dbbox, camera_matrix, self.mean_distance)

    def __len__(self):
        return self.num_examples

class DatasetKittiTest(torch.utils.data.Dataset):

    def __init__(self, kitti_data_path, kitti_meta_path):
        self.img_dir = kitti_data_path + '/object/testing/image_2/'
        self.calib_dir = kitti_data_path + '/object/testing/calib/'
        self.lidar_dir = kitti_data_path + '/object/testing/velodyne/'
        self.detections_2d_dir = kitti_meta_path + '/object/testing/2d_detections/'
        with open(kitti_meta_path + '/kitti_train_mean_car_size.pkl', 'rb') as file:
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)
        with open(kitti_meta_path + '/kitti_train_mean_distance.pkl', 'rb') as file:
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]
        img_ids = []
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            img_id = img_name.split('.png')[0]
            img_ids.append(img_id)
        self.examples = []
        for img_id in img_ids:
            detections_file_path = self.detections_2d_dir + img_id + '.txt'
            with open(detections_file_path) as file:
                for line in file:
                    values = line.split()
                    object_class = float(values[3])
                    if object_class == 1:
                        u_min = float(values[4])
                        v_min = float(values[5])
                        u_max = float(values[6])
                        v_max = float(values[7])
                        score_2d = float(values[8])
                        detection_2d = {}
                        detection_2d['u_min'] = u_min
                        detection_2d['v_min'] = v_min
                        detection_2d['u_max'] = u_max
                        detection_2d['v_max'] = v_max
                        detection_2d['score_2d'] = score_2d
                        detection_2d['img_id'] = img_id
                        self.examples.append(detection_2d)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        calib_path = self.calib_dir + img_id + '.txt'
        calib = calibread(calib_path)
        camera_matrix = calib['P2']
        u_min = example['u_min']
        u_max = example['u_max']
        v_min = example['v_min']
        v_max = example['v_max']
        score_2d = example['score_2d']
        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])
        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w / 2.0
        v_center = v_min + h / 2.0
        img_path = self.img_dir + img_id + '.png'
        img = cv2.imread(img_path, -1)
        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))
        bbox_2d_img = bbox_2d_img / 255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img / np.array([0.229, 0.224, 0.225])
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1))
        bbox_2d_img = bbox_2d_img.astype(np.float32)
        bbox_2d_img = torch.from_numpy(bbox_2d_img)
        return (bbox_2d_img, img_id, self.mean_car_size, w, h, u_center, v_center, input_2Dbbox, camera_matrix, self.mean_distance, score_2d)

    def __len__(self):
        return self.num_examples

class DatasetImgNetVal2ddetections(torch.utils.data.Dataset):

    def __init__(self, kitti_data_path, kitti_meta_path):
        self.img_dir = kitti_data_path + '/object/training/image_2/'
        self.calib_dir = kitti_data_path + '/object/training/calib/'
        self.lidar_dir = kitti_data_path + '/object/training/velodyne/'
        self.detections_2d_path = kitti_meta_path + '/rgb_detection_val.txt'
        with open(kitti_meta_path + '/kitti_train_mean_car_size.pkl', 'rb') as file:
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)
        with open(kitti_meta_path + '/kitti_train_mean_distance.pkl', 'rb') as file:
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]
        self.examples = []
        with open(self.detections_2d_path) as file:
            for line in file:
                values = line.split()
                object_class = float(values[1])
                if object_class == 2:
                    score_2d = float(values[2])
                    u_min = float(values[3])
                    v_min = float(values[4])
                    u_max = float(values[5])
                    v_max = float(values[6])
                    img_id = values[0].split('image_2/')[1]
                    img_id = img_id.split('.')[0]
                    detection_2d = {}
                    detection_2d['u_min'] = u_min
                    detection_2d['v_min'] = v_min
                    detection_2d['u_max'] = u_max
                    detection_2d['v_max'] = v_max
                    detection_2d['score_2d'] = score_2d
                    detection_2d['img_id'] = img_id
                    self.examples.append(detection_2d)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        calib_path = self.calib_dir + img_id + '.txt'
        calib = calibread(calib_path)
        camera_matrix = calib['P2']
        u_min = example['u_min']
        u_max = example['u_max']
        v_min = example['v_min']
        v_max = example['v_max']
        score_2d = example['score_2d']
        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])
        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w / 2.0
        v_center = v_min + h / 2.0
        img_path = self.img_dir + img_id + '.png'
        img = cv2.imread(img_path, -1)
        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))
        bbox_2d_img = bbox_2d_img / 255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img / np.array([0.229, 0.224, 0.225])
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1))
        bbox_2d_img = bbox_2d_img.astype(np.float32)
        bbox_2d_img = torch.from_numpy(bbox_2d_img)
        return (bbox_2d_img, img_id, self.mean_car_size, w, h, u_center, v_center, input_2Dbbox, camera_matrix, self.mean_distance, score_2d)

    def __len__(self):
        return self.num_examples

