# Cluster 2

def visualize_single_sample_output_gt(pred_tensor, gt_tensor, pcd, show_vis=True, save_path='', mode='constant'):
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.

    mode : str
        Color rendering mode.
    """

    def custom_draw_geometry(pcd, pred, gt):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1.0
        vis.add_geometry(pcd)
        for ele in pred:
            vis.add_geometry(ele)
        for ele in gt:
            vis.add_geometry(ele)
        vis.run()
        vis.destroy_window()
    origin_lidar = pcd
    if not isinstance(pcd, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(pcd)
    origin_lidar_intcolor = color_encoding(origin_lidar[:, -1] if mode == 'intensity' else origin_lidar[:, 2], mode=mode)
    origin_lidar[:, :1] = -origin_lidar[:, :1]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)
    oabbs_pred = bbx2oabb(pred_tensor, color=(1, 0, 0))
    oabbs_gt = bbx2oabb(gt_tensor, color=(0, 1, 0))
    visualize_elements = [o3d_pcd] + oabbs_pred + oabbs_gt
    if show_vis:
        custom_draw_geometry(o3d_pcd, oabbs_pred, oabbs_gt)
    if save_path:
        save_o3d_visualization(visualize_elements, save_path)

def color_encoding(intensity, mode='intensity'):
    """
    Encode the single-channel intensity to 3 channels rgb color.

    Parameters
    ----------
    intensity : np.ndarray
        Lidar intensity, shape (n,)

    mode : str
        The color rendering mode. intensity, z-value and constant are
        supported.

    Returns
    -------
    color : np.ndarray
        Encoded Lidar color, shape (n, 3)
    """
    assert mode in ['intensity', 'z-value', 'constant']
    if mode == 'intensity':
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]), np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]), np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]
    elif mode == 'z-value':
        min_value = -1.5
        max_value = 0.5
        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(intensity)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5
        int_color = colors[:, :3]
    elif mode == 'constant':
        int_color = np.ones((intensity.shape[0], 3))
        int_color[:, 0] *= 247 / 255
        int_color[:, 1] *= 244 / 255
        int_color[:, 2] *= 237 / 255
    return int_color

def bbx2oabb(bbx_corner, order='hwl', color=(0, 0, 1)):
    """
    Convert the torch tensor bounding box to o3d oabb for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    oabbs : list
        The list containing all oriented bounding boxes.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)
    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner, order)
    oabbs = []
    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        bbx[:, :1] = -bbx[:, :1]
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)
        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)
    return oabbs

def custom_draw_geometry(pcd, pred, gt):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0
    vis.add_geometry(pcd)
    for ele in pred:
        vis.add_geometry(ele)
    for ele in gt:
        vis.add_geometry(ele)
    vis.run()
    vis.destroy_window()

def visualize_sequence_sample_output(pred_tensor_list, gt_tensor_list, pcd_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().show_coordinate_frame = True
    vis_pcd = o3d.geometry.PointCloud()
    while True:
        for i, (pred_tensor, gt_tensor, pcd) in enumerate(zip(pred_tensor_list, gt_tensor_list, pcd_list)):
            pred_tensor = pred_tensor.copy()
            gt_tensor = gt_tensor.copy()
            pcd = pcd.copy()
            pcd_intcolor = color_encoding(pcd[:, -1])
            pcd[:, :1] = -pcd[:, :1]
            vis_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
            vis_pcd.colors = o3d.utility.Vector3dVector(pcd_intcolor)
            oabbs_pred = bbx2oabb(pred_tensor, 'hwl')
            oabbs_gt = bbx2oabb(gt_tensor, 'hwl', color=(0, 1, 0))
            oabbs = oabbs_pred + oabbs_gt
            if i == 0:
                vis.add_geometry(vis_pcd)
            for oabb in oabbs:
                vis.add_geometry(oabb)
            vis.update_geometry(vis_pcd)
            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters('pinhole_param.json')
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            for oabb in oabbs:
                vis.remove_geometry(oabb)
            time.sleep(0.01)
    vis.destroy_window()

def visualize_single_sample_dataloader(batch_data, o3d_pcd, order, key='origin_lidar', visualize=False, save_path='', oabb=False, mode='constant'):
    """
    Visualize a single frame of a single CAV for validation of data pipeline.

    Parameters
    ----------
    o3d_pcd : o3d.PointCloud
        Open3d PointCloud.

    order : str
        The bounding box order.

    key : str
        origin_lidar for late fusion and stacked_lidar for early fusion.
        todo: consider intermediate fusion in the future.

    visualize : bool
        Whether to visualize the sample.

    batch_data : dict
        The dictionary that contains current timestamp's data.

    save_path : str
        If set, save the visualization image to the path.

    oabb : bool
        If oriented bounding box is used.
    """
    origin_lidar = batch_data[key]
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar_intcolor = color_encoding(origin_lidar[:, -1] if mode == 'intensity' else origin_lidar[:, 2], mode=mode)
    origin_lidar[:, :1] = -origin_lidar[:, :1]
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)
    object_bbx_center = batch_data['object_bbx_center']
    object_bbx_mask = batch_data['object_bbx_mask']
    object_bbx_center = object_bbx_center[object_bbx_mask == 1]
    aabbs = bbx2linset(object_bbx_center, order) if not oabb else bbx2oabb(object_bbx_center, order)
    visualize_elements = [o3d_pcd] + aabbs
    if visualize:
        o3d.visualization.draw_geometries(visualize_elements)
    if save_path:
        save_o3d_visualization(visualize_elements, save_path)
    return (o3d_pcd, aabbs)

def bbx2linset(bbx_corner, order='hwl', color=(0, 1, 0)):
    """
    Convert the torch tensor bounding box to o3d lineset for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    line_set : list
        The list containing linsets.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)
    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner, order)
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [list(color) for _ in range(len(lines))]
    bbx_linset = []
    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        bbx[:, :1] = -bbx[:, :1]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbx_linset.append(line_set)
    return bbx_linset

def visualize_inference_sample_dataloader(pred_box_tensor, gt_box_tensor, origin_lidar, o3d_pcd, mode='constant'):
    """
    Visualize a frame during inference for video stream.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_box_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    origin_lidar : torch.Tensor
        PointCloud, (N, 4).

    o3d_pcd : open3d.PointCloud
        Used to visualize the pcd.

    mode : str
        lidar point rendering mode.
    """
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar_intcolor = color_encoding(origin_lidar[:, -1] if mode == 'intensity' else origin_lidar[:, 2], mode=mode)
    if not isinstance(pred_box_tensor, np.ndarray):
        pred_box_tensor = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    if not isinstance(gt_box_tensor, np.ndarray):
        gt_box_tensor = common_utils.torch_tensor_to_numpy(gt_box_tensor)
    origin_lidar[:, :1] = -origin_lidar[:, :1]
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)
    gt_o3d_box = bbx2linset(gt_box_tensor, order='hwl', color=(0, 1, 0))
    pred_o3d_box = bbx2linset(pred_box_tensor, color=(1, 0, 0))
    return (o3d_pcd, pred_o3d_box, gt_o3d_box)

def visualize_sequence_dataloader(dataloader, order, color_mode='constant'):
    """
    Visualize the batch data in animation.

    Parameters
    ----------
    dataloader : torch.Dataloader
        Pytorch dataloader

    order : str
        Bounding box order(N, 7).

    color_mode : str
        Color rendering mode.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().show_coordinate_frame = True
    vis_pcd = o3d.geometry.PointCloud()
    vis_aabbs = []
    for _ in range(50):
        vis_aabbs.append(o3d.geometry.LineSet())
    while True:
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch)
            pcd, aabbs = visualize_single_sample_dataloader(sample_batched['ego'], vis_pcd, order, mode=color_mode)
            if i_batch == 0:
                vis.add_geometry(pcd)
                for i in range(len(vis_aabbs)):
                    index = i if i < len(aabbs) else -1
                    vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                    vis.add_geometry(vis_aabbs[i])
            for i in range(len(vis_aabbs)):
                index = i if i < len(aabbs) else -1
                vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                vis.update_geometry(vis_aabbs[i])
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)
    vis.destroy_window()

def lineset_assign(lineset1, lineset2):
    """
    Assign the attributes of lineset2 to lineset1.

    Parameters
    ----------
    lineset1 : open3d.LineSet
    lineset2 : open3d.LineSet

    Returns
    -------
    The lineset1 object with 2's attributes.
    """
    lineset1.points = lineset2.points
    lineset1.lines = lineset2.lines
    lineset1.colors = lineset2.colors
    return lineset1

