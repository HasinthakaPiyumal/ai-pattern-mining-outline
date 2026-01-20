# Cluster 15

class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to assign correct
    index and add noise.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the dataset is used for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    """

    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(params['data_augment'], train)
        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            self.async_flag = params['wild_setting']['async']
            self.async_mode = 'sim' if 'async_mode' not in params['wild_setting'] else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']
            self.data_size = params['wild_setting']['data_size'] if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = params['wild_setting']['transmission_speed'] if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = params['wild_setting']['backbone_delay'] if 'backbone_delay' in params['wild_setting'] else 0
        else:
            self.async_flag = False
            self.async_overhead = 0
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0
            self.transmission_speed = 27
            self.backbone_delay = 0
        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        if 'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']
        scenario_folders = sorted([os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_database = OrderedDict()
        self.len_record = []
        for i, scenario_folder in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})
            cav_list = sorted([x for x in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]
            for j, cav_id in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()
                cav_path = os.path.join(scenario_folder, cav_id)
                yaml_files = sorted([os.path.join(cav_path, x) for x in os.listdir(cav_path) if x.endswith('.yaml')])
                timestamps = self.extract_timestamps(yaml_files)
                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = OrderedDict()
                    yaml_file = os.path.join(cav_path, timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path, timestamp + '.pcd')
                    camera_files = self.load_camera_files(cav_path, timestamp)
                    self.scenario_database[i][cav_id][timestamp]['yaml'] = yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = lidar_file
                    self.scenario_database[i][cav_id][timestamp]['camera0'] = camera_files
                if j == 0:
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]
        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)
        data = OrderedDict()
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            timestamp_delay = self.time_delay_calculation(cav_content['ego'])
            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database, timestamp_index_delay)
            data[cav_id]['time_delay'] = timestamp_delay
            data[cav_id]['params'] = self.reform_param(cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag)
            data[cav_id]['lidar_np'] = pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
        return data

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []
        for file in yaml_files:
            res = file.split('/')[-1]
            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)
        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        timestamp_keys = list(scenario_database.items())[0][1]
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]
        return timestamp_key

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break
        assert ego_lidar_pose is not None
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            distance = math.sqrt((cur_lidar_pose[0] - ego_lidar_pose[0]) ** 2 + (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2)
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})
        return ego_cav_content

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        if ego_flag:
            return 0
        if self.async_mode == 'real':
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            time_delay = np.abs(self.async_overhead)
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0], pose[1] + xyz_noise[1], pose[2] + xyz_noise[2], pose[3], pose[4] + ryp_std[1], pose[5]]
        return noise_pose

    def reform_param(self, cav_content, ego_content, timestamp_cur, timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])
        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params['lidar_pose']
        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']
        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)
        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose, cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose, delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose, cur_ego_lidar_pose)
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose, cur_ego_lidar_pose)
        delay_params['vehicles'] = cur_params['vehicles']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix
        return delay_params

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path, timestamp + '_camera0.png')
        camera1_file = os.path.join(cav_path, timestamp + '_camera1.png')
        camera2_file = os.path.join(cav_path, timestamp + '_camera2.png')
        camera3_file = os.path.join(cav_path, timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Data augmentation operation.
        """
        tmp_dict = {'lidar_np': lidar_np, 'object_bbx_center': object_bbx_center, 'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)
        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']
        return (lidar_np, object_bbx_center, object_bbx_mask)

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        output_dict = {'ego': {}}
        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []
        if self.visualize:
            origin_lidar = []
        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])
            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        processed_lidar_torch_dict = self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': object_bbx_mask, 'processed_lidar': processed_lidar_torch_dict, 'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})
        return output_dict

    def visualize_result(self, pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        self.post_processor.visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=dataset)

def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)
    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix

def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

def mask_boxes_outside_range_numpy(boxes, limit_range, order, min_num_corners=8):
    """
    Parameters
    ----------
    boxes: np.ndarray
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    limit_range: list
        [minx, miny, minz, maxx, maxy, maxz]

    min_num_corners: int
        The required minimum number of corners to be considered as in range.

    order : str
        'lwh' or 'hwl'

    Returns
    -------
    boxes: np.ndarray
        The filtered boxes.
    """
    assert boxes.shape[1] == 8 or boxes.shape[1] == 7
    new_boxes = boxes.copy()
    if boxes.shape[1] == 7:
        new_boxes = boxes_to_corners_3d(new_boxes, order)
    mask = ((new_boxes >= limit_range[0:3]) & (new_boxes <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners
    return boxes[mask]

def project_world_objects(object_dict, output_dict, lidar_pose, lidar_range, order):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'
    """
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']
        object_pose = [location[0] + center[0], location[1] + center[1], location[2] + center[2], rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, lidar_pose)
        bbx = create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar, lidar_range, order)
        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: bbx_lidar})

def create_bbx(extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        Width, height, length of the bbx.

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """
    bbx = np.array([[extent[0], -extent[1], -extent[2]], [extent[0], extent[1], -extent[2]], [-extent[0], extent[1], -extent[2]], [-extent[0], -extent[1], -extent[2]], [extent[0], -extent[1], extent[2]], [extent[0], extent[1], extent[2]], [-extent[0], extent[1], extent[2]], [-extent[0], -extent[1], extent[2]]])
    return bbx

def corner_to_center(corner3d, order='lwh'):
    """
    Convert 8 corners to x, y, z, dx, dy, dz, yaw.

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)

    order : str
        'lwh' or 'hwl'

    Returns
    -------
    box3d : np.ndarray
        (N, 7)
    """
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]
    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)
    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1, keepdims=True))
    l = (np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) + np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) + np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) + np.sqrt(np.sum((corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4
    w = (np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) + np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) + np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) + np.sqrt(np.sum((corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4
    theta = (np.arctan2(corner3d[:, 1, 1] - corner3d[:, 2, 1], corner3d[:, 1, 0] - corner3d[:, 2, 0]) + np.arctan2(corner3d[:, 0, 1] - corner3d[:, 3, 1], corner3d[:, 0, 0] - corner3d[:, 3, 0]) + np.arctan2(corner3d[:, 5, 1] - corner3d[:, 6, 1], corner3d[:, 5, 0] - corner3d[:, 6, 0]) + np.arctan2(corner3d[:, 4, 1] - corner3d[:, 7, 1], corner3d[:, 4, 0] - corner3d[:, 7, 0]))[:, np.newaxis] / 4
    if order == 'lwh':
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(batch_size, 7)
    elif order == 'hwl':
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(batch_size, 7)
    else:
        sys.exit('Unknown order')

