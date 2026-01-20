# Cluster 14

def axis_angle_to_rotation_matrix(axis, angle=None):
    if angle is None:
        angle = np.linalg.norm(axis)
        if np.abs(angle) > np.finfo('float').eps:
            axis = axis / angle
    cp_axis = cross_prod_matrix(axis)
    return np.eye(3) + (np.sin(angle) * cp_axis + (1.0 - np.cos(angle)) * cp_axis.dot(cp_axis))

def cross_prod_matrix(v):
    return np.array(((0.0, -v[2], v[1]), (v[2], 0.0, -v[0]), (-v[1], v[0], 0.0)))

def axis_angle_to_rotation_matrix(axis, angle=None):
    if angle is None:
        angle = np.linalg.norm(axis)
        if np.abs(angle) > np.finfo('float').eps:
            axis = axis / angle
    cp_axis = cross_prod_matrix(axis)
    return np.eye(3) + (np.sin(angle) * cp_axis + (1.0 - np.cos(angle)) * cp_axis.dot(cp_axis))

