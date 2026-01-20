# Cluster 11

def get_vehicle_to_lidar_transform():
    return np.linalg.inv(get_lidar_to_vehicle_transform())

def get_lidar_to_vehicle_transform():
    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    T = np.eye(4)
    T[:3, :3] = rot
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T

def align(lidar_0, measurements_0, measurements_1, degree=0):
    matrix_0 = measurements_0['ego_matrix']
    matrix_1 = measurements_1['ego_matrix']
    matrix_0 = np.array(matrix_0)
    matrix_1 = np.array(matrix_1)
    Tr_lidar_to_vehicle = get_lidar_to_vehicle_transform()
    Tr_vehicle_to_lidar = get_vehicle_to_lidar_transform()
    transform_0_to_1 = Tr_vehicle_to_lidar @ np.linalg.inv(matrix_1) @ matrix_0 @ Tr_lidar_to_vehicle
    rad = np.deg2rad(degree)
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0, 0], [-np.sin(rad), np.cos(rad), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    transform_0_to_1 = degree_matrix @ transform_0_to_1
    lidar = lidar_0.copy()
    lidar[:, -1] = 1.0
    lidar[:, 1] *= -1.0
    lidar = transform_0_to_1 @ lidar.T
    lidar = lidar.T
    lidar[:, -1] = lidar_0[:, -1]
    lidar[:, 1] *= -1.0
    return lidar

