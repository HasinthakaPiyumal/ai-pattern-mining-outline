# Cluster 12

class EarlyFusionVisDataset(basedataset.BaseDataset):

    def __init__(self, params, visualize, train=True):
        super(EarlyFusionVisDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'], train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}
        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0
        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []
        for cav_id, selected_cav_base in base_data_dict.items():
            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)
            projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1
        projected_lidar_stack = np.vstack(projected_lidar_stack)
        projected_lidar_stack, object_bbx_center, mask = self.augment(projected_lidar_stack, object_bbx_center, mask)
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack, self.params['preprocess']['cav_lidar_range'])
        object_bbx_center_valid = object_bbx_center[mask == 1]
        object_bbx_center_valid = box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid, self.params['preprocess']['cav_lidar_range'], self.params['postprocess']['order'])
        mask[object_bbx_center_valid.shape[0]:] = 0
        object_bbx_center[:object_bbx_center_valid.shape[0]] = object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
        processed_data_dict['ego'].update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': mask, 'object_ids': [object_id_stack[i] for i in unique_indices], 'origin_lidar': projected_lidar_stack})
        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}
        transformation_matrix = selected_cav_base['params']['transformation_matrix']
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center([selected_cav_base], ego_pose)
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        selected_cav_processed.update({'object_bbx_center': object_bbx_center[object_bbx_mask == 1], 'object_ids': object_ids, 'projected_lidar': lidar_np})
        return selected_cav_processed

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
        origin_lidar = []
        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            origin_lidar.append(ego_dict['origin_lidar'])
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        output_dict['ego'].update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': object_bbx_mask})
        origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
        origin_lidar = torch.from_numpy(origin_lidar)
        output_dict['ego'].update({'origin_lidar': origin_lidar})
        return output_dict

def downsample_lidar_minimum(pcd_np_list):
    """
    Given a list of pcd, find the minimum number and downsample all
    point clouds to the minimum number.

    Parameters
    ----------
    pcd_np_list : list
        A list of pcd numpy array(n, 4).
    Returns
    -------
    pcd_np_list : list
        Downsampled point clouds.
    """
    minimum = np.Inf
    for i in range(len(pcd_np_list)):
        num = pcd_np_list[i].shape[0]
        minimum = num if minimum > num else minimum
    for i, pcd_np in enumerate(pcd_np_list):
        pcd_np_list[i] = downsample_lidar(pcd_np, minimum)
    return pcd_np_list

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

class IntermediateFusionDataset(basedataset.BaseDataset):

    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDataset, self).__init__(params, visualize, train)
        self.cur_ego_pose_flag = params['fusion']['args']['cur_ego_pose_flag']
        self.pre_processor = build_preprocessor(params['preprocess'], train)
        self.post_processor = post_processor.build_postprocessor(params['postprocess'], train)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)
        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}
        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict.keys())[0], 'The first element in the OrderedDict must be ego'
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0
        pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.params['train_params']['max_cav'])
        processed_features = []
        object_stack = []
        object_id_stack = []
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []
        if self.visualize:
            projected_lidar_stack = []
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            if distance > v2xvit.data_utils.datasets.COM_RANGE:
                continue
            selected_cav_processed, void_lidar = self.get_item_single_car(selected_cav_base, ego_lidar_pose)
            if void_lidar:
                continue
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(selected_cav_processed['processed_features'])
            velocity.append(selected_cav_processed['velocity'])
            time_delay.append(float(selected_cav_base['time_delay']))
            spatial_correction_matrix.append(selected_cav_base['params']['spatial_correction_matrix'])
            infra.append(1 if int(cav_id) < 0 else 0)
            if self.visualize:
                projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1
        cav_num = len(processed_features)
        merged_feature_dict = self.merge_features_to_dict(processed_features)
        anchor_box = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask)
        velocity = velocity + (self.max_cav - len(velocity)) * [0.0]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.0]
        infra = infra + (self.max_cav - len(infra)) * [0.0]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(spatial_correction_matrix), 1, 1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix, padding_eye], axis=0)
        processed_data_dict['ego'].update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': mask, 'object_ids': [object_id_stack[i] for i in unique_indices], 'anchor_box': anchor_box, 'processed_lidar': merged_feature_dict, 'label_dict': label_dict, 'cav_num': cav_num, 'velocity': velocity, 'time_delay': time_delay, 'infra': infra, 'spatial_correction_matrix': spatial_correction_matrix, 'pairwise_t_matrix': pairwise_t_matrix})
        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar': np.vstack(projected_lidar_stack)})
        return processed_data_dict

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix across different agents.
        This is only used for v2vnet and disconet. Currently we set
        this as identity matrix as the pointcloud is projected to
        ego vehicle first.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        pairwise_t_matrix[:, :] = np.identity(4)
        return pairwise_t_matrix

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}
        transformation_matrix = selected_cav_base['params']['transformation_matrix']
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center([selected_cav_base], ego_pose)
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
        void_lidar = True if lidar_np.shape[0] < 1 else False
        processed_lidar = self.pre_processor.preprocess(lidar_np)
        velocity = selected_cav_base['params']['ego_speed']
        velocity = velocity / 30
        selected_cav_processed.update({'object_bbx_center': object_bbx_center[object_bbx_mask == 1], 'object_ids': object_ids, 'projected_lidar': lidar_np, 'processed_features': processed_lidar, 'velocity': velocity})
        return (selected_cav_processed, void_lidar)

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """
        merged_feature_dict = OrderedDict()
        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)
        return merged_feature_dict

    def collate_batch_train(self, batch):
        output_dict = {'ego': {}}
        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        record_len = []
        label_dict_list = []
        velocity = []
        time_delay = []
        infra = []
        pairwise_t_matrix_list = []
        spatial_correction_matrix_list = []
        if self.visualize:
            origin_lidar = []
        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])
            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(ego_dict['spatial_correction_matrix'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = torch.from_numpy(np.array(spatial_correction_matrix_list))
        prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
        output_dict['ego'].update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': object_bbx_mask, 'processed_lidar': processed_lidar_torch_dict, 'record_len': record_len, 'label_dict': label_torch_dict, 'object_ids': object_ids[0], 'prior_encoding': prior_encoding, 'spatial_correction_matrix': spatial_correction_matrix_list, 'pairwise_t_matrix': pairwise_t_matrix})
        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})
        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, 'Batch size 1 is required during testing!'
        output_dict = self.collate_batch_train(batch)
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box': torch.from_numpy(np.array(batch[0]['ego']['anchor_box']))})
        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix': transformation_matrix_torch})
        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)
        return (pred_box_tensor, pred_score, gt_box_tensor)

class EarlyFusionDataset(basedataset.BaseDataset):

    def __init__(self, params, visualize, train=True):
        super(EarlyFusionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'], train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=True)
        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}
        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0
        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            if distance > v2xvit.data_utils.datasets.COM_RANGE:
                continue
            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)
            projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1
        projected_lidar_stack = np.vstack(projected_lidar_stack)
        projected_lidar_stack, object_bbx_center, mask = self.augment(projected_lidar_stack, object_bbx_center, mask)
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack, self.params['preprocess']['cav_lidar_range'])
        object_bbx_center_valid = object_bbx_center[mask == 1]
        object_bbx_center_valid = box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid, self.params['preprocess']['cav_lidar_range'], self.params['postprocess']['order'])
        mask[object_bbx_center_valid.shape[0]:] = 0
        object_bbx_center[:object_bbx_center_valid.shape[0]] = object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
        lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)
        anchor_box = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask)
        processed_data_dict['ego'].update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': mask, 'object_ids': [object_id_stack[i] for i in unique_indices], 'anchor_box': anchor_box, 'processed_lidar': lidar_dict, 'label_dict': label_dict})
        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar': projected_lidar_stack})
        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}
        transformation_matrix = selected_cav_base['params']['transformation_matrix']
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center([selected_cav_base], ego_pose)
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        selected_cav_processed.update({'object_bbx_center': object_bbx_center[object_bbx_mask == 1], 'object_ids': object_ids, 'projected_lidar': lidar_np})
        return selected_cav_processed

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        assert len(batch) <= 1, 'Batch size 1 is required during testing!'
        batch = batch[0]
        output_dict = {}
        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            object_bbx_center = torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box': torch.from_numpy(np.array(cav_content['anchor_box']))})
            if self.visualize:
                origin_lidar = [cav_content['origin_lidar']]
            processed_lidar_torch_dict = self.pre_processor.collate_batch([cav_content['processed_lidar']])
            label_torch_dict = self.post_processor.collate_batch([cav_content['label_dict']])
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
            output_dict[cav_id].update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': object_bbx_mask, 'processed_lidar': processed_lidar_torch_dict, 'label_dict': label_torch_dict, 'object_ids': object_ids, 'transformation_matrix': transformation_matrix_torch})
            if self.visualize:
                origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})
        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)
        return (pred_box_tensor, pred_score, gt_box_tensor)

class LateFusionDataset(basedataset.BaseDataset):

    def __init__(self, params, visualize, train=True):
        super(LateFusionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'], train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=True)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict)
        return reformat_data_dict

    def get_item_single_car(self, selected_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
        lidar_np = mask_ego_points(lidar_np)
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center([selected_cav_base], selected_cav_base['params']['lidar_pose'])
        lidar_np, object_bbx_center, object_bbx_mask = self.augment(lidar_np, object_bbx_center, object_bbx_mask)
        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update({'processed_lidar': lidar_dict})
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({'anchor_box': anchor_box})
        selected_cav_processed.update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': object_bbx_mask, 'object_ids': object_ids})
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=object_bbx_mask)
        selected_cav_processed.update({'label_dict': label_dict})
        return selected_cav_processed

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()
        if not self.visualize:
            selected_cav_id, selected_cav_base = random.choice(list(base_data_dict.items()))
        else:
            selected_cav_id, selected_cav_base = list(base_data_dict.items())[0]
        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({'ego': selected_cav_processed})
        return processed_data_dict

    def get_item_test(self, base_data_dict):
        processed_data_dict = OrderedDict()
        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            if distance > v2xvit.data_utils.datasets.COM_RANGE:
                continue
            transformation_matrix = selected_cav_base['params']['transformation_matrix']
            gt_transformation_matrix = selected_cav_base['params']['gt_transformation_matrix']
            selected_cav_processed = self.get_item_single_car(selected_cav_base)
            selected_cav_processed.update({'transformation_matrix': transformation_matrix})
            selected_cav_processed.update({'gt_transformation_matrix': gt_transformation_matrix})
            update_cav = 'ego' if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})
        return processed_data_dict

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        assert len(batch) <= 1, 'Batch size 1 is required during testing!'
        batch = batch[0]
        output_dict = {}
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []
        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            object_bbx_center = torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box': torch.from_numpy(np.array(cav_content['anchor_box']))})
            if self.visualize:
                transformation_matrix = cav_content['transformation_matrix']
                origin_lidar = [cav_content['origin_lidar']]
                projected_lidar = cav_content['origin_lidar']
                projected_lidar[:, :3] = box_utils.project_points_by_matrix_torch(projected_lidar[:, :3], transformation_matrix)
                projected_lidar_list.append(projected_lidar)
            processed_lidar_torch_dict = self.pre_processor.collate_batch([cav_content['processed_lidar']])
            label_torch_dict = self.post_processor.collate_batch([cav_content['label_dict']])
            transformation_matrix_torch = torch.from_numpy(np.array(cav_content['transformation_matrix'])).float()
            gt_transformation_matrix_torch = torch.from_numpy(np.array(cav_content['gt_transformation_matrix'])).float()
            output_dict[cav_id].update({'object_bbx_center': object_bbx_center, 'object_bbx_mask': object_bbx_mask, 'processed_lidar': processed_lidar_torch_dict, 'label_dict': label_torch_dict, 'object_ids': object_ids, 'transformation_matrix': transformation_matrix_torch, 'gt_transformation_matrix': gt_transformation_matrix_torch})
            if self.visualize:
                origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})
        if self.visualize:
            projected_lidar_stack = [torch.from_numpy(np.vstack(projected_lidar_list))]
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})
        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)
        return (pred_box_tensor, pred_score, gt_box_tensor)

def downsample_lidar(pcd_np, num):
    """
    Downsample the lidar points to a certain number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points, (n, 4).

    num : int
        The downsample target number.

    Returns
    -------
    pcd_np : np.ndarray
        The downsampled lidar points.
    """
    assert pcd_np.shape[0] >= num
    selected_index = np.random.choice(pcd_np.shape[0], num, replace=False)
    pcd_np = pcd_np[selected_index]
    return pcd_np

