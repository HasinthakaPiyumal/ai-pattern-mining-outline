# Cluster 9

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

def build_postprocessor(anchor_cfg, train):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in ['VoxelPostprocessor', 'BevPostprocessor']
    anchor_generator = __all__[process_method_name](anchor_params=anchor_cfg, train=train)
    return anchor_generator

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

