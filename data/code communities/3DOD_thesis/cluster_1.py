# Cluster 1

class BoxRegressor(object):

    def __init__(self, camera_matrix, pred_size, pred_keypoints, pred_distance):
        super(BoxRegressor, self).__init__()
        self.P = camera_matrix
        self.P_pseudo_inverse = np.linalg.pinv(self.P)
        self.pred_keypoints = pred_keypoints
        self.pred_size = pred_size
        self.pred_distance = pred_distance

    def _residuals(self, params):
        [h, w, l, x, y, z, rot_y] = params
        projected_keypoints = get_keypoints(np.array([x, y, z]), h, w, l, rot_y, self.P)
        resids_keypoints = projected_keypoints - self.pred_keypoints
        resids_keypoints = resids_keypoints.flatten()
        resids_size_regularization = np.array([h - self.pred_size[0], w - self.pred_size[1], l - self.pred_size[2]])
        resids_distance_regularization = np.array([np.linalg.norm(params[3:6]) - self.pred_distance])
        resids = np.append(resids_keypoints, 100 * resids_size_regularization)
        resids = np.append(resids, 10 * resids_distance_regularization)
        return resids

    def _initial_guess(self):
        h, w, l = self.pred_size
        img_keypoints_center_hom = [np.mean(self.pred_keypoints[:, 0]), np.mean(self.pred_keypoints[:, 1]), 1]
        l0 = np.dot(self.P_pseudo_inverse, img_keypoints_center_hom)
        l0 = l0[:3] / l0[3]
        if l0[2] < 0:
            l0[0] = -l0[0]
            l0[2] = -l0[2]
        [x0, y0, z0] = l0 / np.linalg.norm(l0) * self.pred_distance
        rot_y = -np.pi / 2
        return [h, w, l, x0, y0, z0, rot_y]

    def solve(self):
        x0 = self._initial_guess()
        ls_results = []
        costs = []
        for rot_y in [-2, -1, 0, 1]:
            x0[6] = rot_y * np.pi / 2
            ls_result = least_squares(self._residuals, x0, jac='3-point')
            ls_results.append(ls_result)
            costs.append(ls_result.cost)
        self.result = ls_results[np.argmin(costs)]
        params = self.result.x
        return params

def get_keypoints(center, h, w, l, r_y, P2_mat):
    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)], [0, 1, 0], [-math.sin(r_y), 0, math.cos(r_y)]], dtype='float32')
    p0 = center + np.dot(Rmat, np.asarray([l / 2.0, 0, w / 2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l / 2.0, 0, w / 2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l / 2.0, 0, -w / 2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l / 2.0, 0, -w / 2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l / 2.0, -h, w / 2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l / 2.0, -h, w / 2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l / 2.0, -h, -w / 2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l / 2.0, -h, -w / 2.0], dtype='float32').flatten())
    keypoints_3d = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    keypoints_3d_hom = np.ones((8, 4), dtype=np.float32)
    keypoints_3d_hom[:, 0:3] = keypoints_3d
    keypoints_hom = np.dot(P2_mat, keypoints_3d_hom.T).T
    keypoints = np.zeros((8, 2), dtype=np.float32)
    keypoints[:, 0] = keypoints_hom[:, 0] / keypoints_hom[:, 2]
    keypoints[:, 1] = keypoints_hom[:, 1] / keypoints_hom[:, 2]
    return keypoints

class DatasetImgNetAugmentation(torch.utils.data.Dataset):

    def __init__(self, kitti_data_path, kitti_meta_path, type):
        self.img_dir = kitti_data_path + '/object/training/image_2/'
        self.label_dir = kitti_data_path + '/object/training/label_2/'
        self.calib_dir = kitti_data_path + '/object/training/calib/'
        self.lidar_dir = kitti_data_path + '/object/training/velodyne/'
        with open(kitti_meta_path + '/%s_img_ids_random.pkl' % type, 'rb') as file:
            img_ids = pickle.load(file)
        with open(kitti_meta_path + '/kitti_train_mean_car_size.pkl', 'rb') as file:
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)
        with open(kitti_meta_path + '/kitti_train_mean_distance.pkl', 'rb') as file:
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]
        self.examples = []
        for img_id in img_ids:
            labels = LabelLoader2D3D(img_id, self.label_dir, '.txt', self.calib_dir, '.txt')
            for label in labels:
                label_2d = label['label_2D']
                if label_2d['truncated'] < 0.5 and label_2d['class'] == 'Car':
                    label['img_id'] = img_id
                    self.examples.append(label)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        label_2D = example['label_2D']
        label_3D = example['label_3D']
        bbox = label_2D['poly']
        u_min = bbox[0, 0]
        u_max = bbox[1, 0]
        v_min = bbox[0, 1]
        v_max = bbox[2, 1]
        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w / 2.0
        v_center = v_min + h / 2.0
        u_center = u_center + np.random.uniform(low=-0.1 * w, high=0.1 * w)
        v_center = v_center + np.random.uniform(low=-0.1 * h, high=0.1 * h)
        w = w * np.random.uniform(low=0.9, high=1.1)
        h = h * np.random.uniform(low=0.9, high=1.1)
        u_min = u_center - w / 2.0
        u_max = u_center + w / 2.0
        v_min = v_center - h / 2.0
        v_max = v_center + h / 2.0
        img_path = self.img_dir + img_id + '.png'
        img = cv2.imread(img_path, -1)
        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            bbox_2d_img = cv2.flip(bbox_2d_img, 1)
        bbox_2d_img = bbox_2d_img / 255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img / np.array([0.229, 0.224, 0.225])
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1))
        bbox_2d_img = bbox_2d_img.astype(np.float32)
        label_size = np.zeros((3,), dtype=np.float32)
        label_size[0] = label_3D['h']
        label_size[1] = label_3D['w']
        label_size[2] = label_3D['l']
        label_size = label_size - self.mean_car_size
        label_keypoints = get_keypoints(label_3D['center'], label_3D['h'], label_3D['w'], label_3D['l'], label_3D['r_y'], label_3D['P0_mat'])
        if flip == 1:
            img = cv2.imread(self.img_dir + img_id + '.png', -1)
            img_w = img.shape[1]
            u_center = img_w - u_center
            label_keypoints[:, 0] = img_w - label_keypoints[:, 0]
            temp = np.copy(label_keypoints[7, :])
            label_keypoints[7, :] = label_keypoints[4, :]
            label_keypoints[4, :] = temp
            temp = np.copy(label_keypoints[3, :])
            label_keypoints[3, :] = label_keypoints[0, :]
            label_keypoints[0, :] = temp
            temp = np.copy(label_keypoints[6, :])
            label_keypoints[6, :] = label_keypoints[5, :]
            label_keypoints[5, :] = temp
            temp = np.copy(label_keypoints[2, :])
            label_keypoints[2, :] = label_keypoints[1, :]
            label_keypoints[1, :] = temp
        label_keypoints = label_keypoints - np.array([u_center, v_center])
        label_keypoints = label_keypoints / np.array([w, h])
        label_keypoints = label_keypoints.flatten()
        label_keypoints = label_keypoints.astype(np.float32)
        label_distance = np.array([np.linalg.norm(label_3D['center'])], dtype=np.float32)
        label_distance = label_distance - self.mean_distance
        bbox_2d_img = torch.from_numpy(bbox_2d_img)
        label_size = torch.from_numpy(label_size)
        label_keypoints = torch.from_numpy(label_keypoints)
        label_distance = torch.from_numpy(label_distance)
        return (bbox_2d_img, label_size, label_keypoints, label_distance)

    def __len__(self):
        return self.num_examples

class DatasetImgNetEval(torch.utils.data.Dataset):

    def __init__(self, kitti_data_path, kitti_meta_path, type):
        self.img_dir = kitti_data_path + '/object/training/image_2/'
        self.label_dir = kitti_data_path + '/object/training/label_2/'
        self.calib_dir = kitti_data_path + '/object/training/calib/'
        self.lidar_dir = kitti_data_path + '/object/training/velodyne/'
        with open(kitti_meta_path + '/%s_img_ids.pkl' % type, 'rb') as file:
            img_ids = pickle.load(file)
        with open(kitti_meta_path + '/kitti_train_mean_car_size.pkl', 'rb') as file:
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)
        with open(kitti_meta_path + '/kitti_train_mean_distance.pkl', 'rb') as file:
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]
        self.examples = []
        for img_id in img_ids:
            labels = LabelLoader2D3D(img_id, self.label_dir, '.txt', self.calib_dir, '.txt')
            for label in labels:
                label_2d = label['label_2D']
                if label_2d['truncated'] < 0.5 and label_2d['class'] == 'Car':
                    label['img_id'] = img_id
                    self.examples.append(label)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        label_2D = example['label_2D']
        label_3D = example['label_3D']
        bbox = label_2D['poly']
        u_min = bbox[0, 0]
        u_max = bbox[1, 0]
        v_min = bbox[0, 1]
        v_max = bbox[2, 1]
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
        label_size = np.zeros((3,), dtype=np.float32)
        label_size[0] = label_3D['h']
        label_size[1] = label_3D['w']
        label_size[2] = label_3D['l']
        label_size = label_size - self.mean_car_size
        label_keypoints = get_keypoints(label_3D['center'], label_3D['h'], label_3D['w'], label_3D['l'], label_3D['r_y'], label_3D['P0_mat'])
        label_keypoints = label_keypoints - np.array([u_center, v_center])
        label_keypoints = label_keypoints / np.array([w, h])
        label_keypoints = label_keypoints.flatten()
        label_keypoints = label_keypoints.astype(np.float32)
        label_distance = np.array([np.linalg.norm(label_3D['center'])], dtype=np.float32)
        label_distance = label_distance - self.mean_distance
        bbox_2d_img = torch.from_numpy(bbox_2d_img)
        label_size = torch.from_numpy(label_size)
        label_keypoints = torch.from_numpy(label_keypoints)
        label_distance = torch.from_numpy(label_distance)
        camera_matrix = label_3D['P0_mat']
        gt_center = label_3D['center']
        gt_center = gt_center.astype(np.float32)
        gt_r_y = np.float32(label_3D['r_y'])
        return (bbox_2d_img, label_size, label_keypoints, label_distance, img_id, self.mean_car_size, w, h, u_center, v_center, camera_matrix, gt_center, gt_r_y, self.mean_distance)

    def __len__(self):
        return self.num_examples

class DatasetImgNetEvalValSeq(torch.utils.data.Dataset):

    def __init__(self, kitti_data_path, kitti_meta_path, sequence='0000'):
        self.img_dir = kitti_data_path + '/tracking/training/image_02/' + sequence + '/'
        self.lidar_dir = kitti_data_path + '/tracking/training/velodyne/' + sequence + '/'
        self.label_path = kitti_data_path + '/tracking/training/label_02/' + sequence + '.txt'
        self.calib_path = kitti_meta_path + '/tracking/training/calib/' + sequence + '.txt'
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
            if img_id.lstrip('0') == '':
                img_id_float = 0.0
            else:
                img_id_float = float(img_id.lstrip('0'))
            labels = LabelLoader2D3D_sequence(img_id, img_id_float, self.label_path, self.calib_path)
            for label in labels:
                label_2d = label['label_2D']
                if label_2d['truncated'] < 0.5 and label_2d['class'] == 'Car':
                    label['img_id'] = img_id
                    self.examples.append(label)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        label_2D = example['label_2D']
        label_3D = example['label_3D']
        bbox = label_2D['poly']
        u_min = bbox[0, 0]
        u_max = bbox[1, 0]
        v_min = bbox[0, 1]
        v_max = bbox[2, 1]
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
        label_size = np.zeros((3,), dtype=np.float32)
        label_size[0] = label_3D['h']
        label_size[1] = label_3D['w']
        label_size[2] = label_3D['l']
        label_size = label_size - self.mean_car_size
        label_keypoints = get_keypoints(label_3D['center'], label_3D['h'], label_3D['w'], label_3D['l'], label_3D['r_y'], label_3D['P0_mat'])
        label_keypoints = label_keypoints - np.array([u_center, v_center])
        label_keypoints = label_keypoints / np.array([w, h])
        label_keypoints = label_keypoints.flatten()
        label_keypoints = label_keypoints.astype(np.float32)
        label_distance = np.array([np.linalg.norm(label_3D['center'])], dtype=np.float32)
        label_distance = label_distance - self.mean_distance
        bbox_2d_img = torch.from_numpy(bbox_2d_img)
        label_size = torch.from_numpy(label_size)
        label_keypoints = torch.from_numpy(label_keypoints)
        label_distance = torch.from_numpy(label_distance)
        camera_matrix = label_3D['P0_mat']
        gt_center = label_3D['center']
        gt_center = gt_center.astype(np.float32)
        gt_r_y = np.float32(label_3D['r_y'])
        return (bbox_2d_img, label_size, label_keypoints, label_distance, img_id, self.mean_car_size, w, h, u_center, v_center, camera_matrix, gt_center, gt_r_y, self.mean_distance)

    def __len__(self):
        return self.num_examples

