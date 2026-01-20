# Cluster 6

def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())

def get_virtual_lidar_to_vehicle_transform():
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T

def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""
    T = get_vehicle_to_virtual_lidar_transform()
    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix
    return waypoints

