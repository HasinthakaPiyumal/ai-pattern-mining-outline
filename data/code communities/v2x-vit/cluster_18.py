# Cluster 18

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), radians, angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return (torch.from_numpy(x).float(), True)
    return (x, False)

def rotate_points_along_z_2d(points, angle):
    """
    Rorate the points along z-axis.
    Parameters
    ----------
    points : torch.Tensor / np.ndarray
        (N, 2).
    angle : torch.Tensor / np.ndarray
        (N,)

    Returns
    -------
    points_rot : torch.Tensor / np.ndarray
        Rorated points with shape (N, 2)

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2, 2).float()
    points_rot = torch.einsum('ik, ikj->ij', points.float(), rot_matrix)
    return points_rot.numpy() if is_numpy else points_rot

def boxes2d_to_corners2d(boxes2d, order='lwh'):
    """
      0 -------- 1
      |          |
      |          |
      |          |
      3 -------- 2
    Parameters
    __________
    boxes2d: np.ndarray or torch.Tensor
        (..., 5) [x, y, dx, dy, heading], (x, y) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners2d: np.ndarray or torch.Tensor
        (..., 4, 2), the 4 corners of the bounding box.

    """
    assert order == 'lwh', 'boxes2d_to_corners_2d only supports lwh order for now.'
    boxes2d, is_numpy = common_utils.check_numpy_to_torch(boxes2d)
    template = boxes2d.new_tensor(([1, -1], [1, 1], [-1, 1], [-1, -1])) / 2
    input_shape = boxes2d.shape
    boxes2d = boxes2d.view(-1, 5)
    corners2d = boxes2d[:, None, 2:4].repeat(1, 4, 1) * template[None, :, :]
    corners2d = common_utils.rotate_points_along_z_2d(corners2d.view(-1, 2), boxes2d[:, 4].repeat_interleave(4)).view(-1, 4, 2)
    corners2d += boxes2d[:, None, 0:2]
    corners2d = corners2d.view(*input_shape[:-1], 4, 2)
    return corners2d

def boxes_to_corners_3d(boxes3d, order):
    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    if order == 'hwl':
        boxes3d[:, 3:6] = boxes3d[:, [5, 4, 3]]
    template = boxes3d.new_tensor(([1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1])) / 2
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d.numpy() if is_numpy else corners3d

def project_box3d(box3d, transformation_matrix):
    """
    Project the 3d bounding box to another coordinate system based on the
    transfomration matrix.

    Parameters
    ----------
    box3d : torch.Tensor or np.ndarray
        3D bounding box, (N, 8, 3)

    transformation_matrix : torch.Tensor or np.ndarray
        Transformation matrix, (4, 4)

    Returns
    -------
    projected_box3d : torch.Tensor
        The projected bounding box, (N, 8, 3)
    """
    assert transformation_matrix.shape == (4, 4)
    box3d, is_numpy = common_utils.check_numpy_to_torch(box3d)
    transformation_matrix, _ = common_utils.check_numpy_to_torch(transformation_matrix)
    box3d_corner = box3d.transpose(1, 2)
    torch_ones = torch.ones((box3d_corner.shape[0], 1, 8))
    torch_ones = torch_ones.to(box3d_corner.device)
    box3d_corner = torch.cat((box3d_corner, torch_ones), dim=1)
    projected_box3d = torch.matmul(transformation_matrix, box3d_corner)
    projected_box3d = projected_box3d[:, :3, :].transpose(1, 2)
    return projected_box3d if not is_numpy else projected_box3d.numpy()

def project_points_by_matrix_torch(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transformation matrix.

    Parameters
    ----------
    points : torch.Tensor
        3D points, (N, 3)
    transformation_matrix : torch.Tensor
        Transformation matrix, (4, 4)
    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 3)
    """
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    transformation_matrix, _ = common_utils.check_numpy_to_torch(transformation_matrix)
    points_homogeneous = F.pad(points, (0, 1), mode='constant', value=1)
    projected_points = torch.einsum('ik, jk->ij', points_homogeneous, transformation_matrix)
    return projected_points[:, :3] if not is_numpy else projected_points[:, :3].numpy()

