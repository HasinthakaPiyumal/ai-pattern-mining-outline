# Cluster 8

def calc_location(dimension, proj_matrix, box_2d, alpha, theta_ray):
    orient = alpha + theta_ray
    R = rotation_matrix(orient)
    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]
    box_corners = [xmin, ymin, xmax, ymax]
    constraints = []
    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2
    left_mult = 1
    right_mult = -1
    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88):
        left_mult = 1
        right_mult = 1
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92):
        left_mult = -1
        right_mult = -1
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90):
        left_mult = -1
        right_mult = 1
    switch_mult = -1
    if alpha > 0:
        switch_mult = 1
    for i in (-1, 1):
        left_constraints.append([left_mult * dx, i * dy, -switch_mult * dz])
    for i in (-1, 1):
        right_constraints.append([right_mult * dx, i * dy, switch_mult * dz])
    for i in (-1, 1):
        for j in (-1, 1):
            top_constraints.append([i * dx, -dy, j * dz])
    for i in (-1, 1):
        for j in (-1, 1):
            bottom_constraints.append([i * dx, dy, j * dz])
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])
    constraints = filter(lambda x: len(x) == len(set((tuple(i) for i in x))), constraints)
    pre_M = np.zeros([4, 4])
    for i in range(0, 4):
        pre_M[i][i] = 1
    best_loc = None
    best_error = [1000000000.0]
    best_X = None
    count = 0
    for constraint in constraints:
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]
        X_array = [Xa, Xb, Xc, Xd]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)
        M_array = [Ma, Mb, Mc, Md]
        A = np.zeros([4, 3], dtype=np.float)
        b = np.zeros([4, 1])
        indicies = [0, 1, 0, 1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]
            RX = np.dot(R, X)
            M[:3, 3] = RX.reshape(3)
            M = np.dot(proj_matrix, M)
            A[row, :] = M[index, :3] - box_corners[row] * M[2, :3]
            b[row] = box_corners[row] * M[2, 3] - M[index, 3]
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)
        if error < best_error:
            count += 1
            best_loc = loc
            best_error = error
            best_X = X_array
    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return (best_loc, best_X)

def rotation_matrix(yaw, pitch=0, roll=0):
    tx = roll
    ty = yaw
    tz = pitch
    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
    return Ry.reshape([3, 3])

def project_3d_pt(pt, cam_to_img, calib_file=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
        R0_rect = get_R0(calib_file)
        Tr_velo_to_cam = get_tr_to_velo(calib_file)
    point = np.array(pt)
    point = np.append(point, 1)
    point = np.dot(cam_to_img, point)
    point = point[:2] / point[2]
    point = point.astype(np.int16)
    return point

def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img
    file_not_found(cab_f)

def get_R0(cab_f):
    for line in open(cab_f):
        if 'R0_rect:' in line:
            R0 = line.strip().split(' ')
            R0 = np.asarray([float(number) for number in R0[1:]])
            R0 = np.reshape(R0, (3, 3))
            R0_rect = np.zeros([4, 4])
            R0_rect[3, 3] = 1
            R0_rect[:3, :3] = R0
            return R0_rect

def get_tr_to_velo(cab_f):
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line:
            Tr = line.strip().split(' ')
            Tr = np.asarray([float(number) for number in Tr[1:]])
            Tr = np.reshape(Tr, (3, 4))
            Tr_to_velo = np.zeros([4, 4])
            Tr_to_velo[3, 3] = 1
            Tr_to_velo[:3, :4] = Tr
            return Tr_to_velo

def plot_3d_pts(img, pts, center, calib_file=None, cam_to_img=None, relative=False, constraint_idx=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
    for pt in pts:
        if relative:
            pt = [i + center[j] for j, i in enumerate(pt)]
        point = project_3d_pt(pt, cam_to_img)
        color = cv_colors.RED.value
        if constraint_idx is not None:
            color = constraint_to_color(constraint_idx)
        cv2.circle(img, (point[0], point[1]), 3, color, thickness=-1)

def constraint_to_color(constraint_idx):
    return {0: cv_colors.PURPLE.value, 1: cv_colors.ORANGE.value, 2: cv_colors.MINT.value, 3: cv_colors.YELLOW.value}[constraint_idx]

def plot_3d_box(img, cam_to_img, ry, dimension, center):
    R = rotation_matrix(ry)
    corners = create_corners(dimension, location=center, R=R)
    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        point[0] = int(point[0] * 1242 / 640)
        point[1] = int(point[1] * 375 / 224)
        box_3d.append(point)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0], box_3d[2][1]), cv_colors.GREEN.value, 2)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0], box_3d[6][1]), cv_colors.GREEN.value, 2)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0], box_3d[4][1]), cv_colors.GREEN.value, 2)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0], box_3d[6][1]), cv_colors.GREEN.value, 2)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0], box_3d[3][1]), cv_colors.GREEN.value, 2)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0], box_3d[5][1]), cv_colors.GREEN.value, 2)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0], box_3d[3][1]), cv_colors.GREEN.value, 2)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0], box_3d[5][1]), cv_colors.GREEN.value, 2)
    for i in range(0, 7, 2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i + 1][0], box_3d[i + 1][1]), cv_colors.GREEN.value, 2)
    frame = np.zeros_like(img, np.uint8)
    cv2.fillPoly(frame, np.array([[[box_3d[0]], [box_3d[1]], [box_3d[3]], [box_3d[2]]]], dtype=np.int32), cv_colors.BLUE.value)
    alpha = 0.5
    mask = frame.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, frame, 1 - alpha, 0)[mask]

def create_corners(dimension, location=None, R=None):
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2
    x_corners = []
    y_corners = []
    z_corners = []
    for i in [1, -1]:
        for j in [1, -1]:
            for k in [1, -1]:
                x_corners.append(dx * i)
                y_corners.append(dy * j)
                z_corners.append(dz * k)
    corners = [x_corners, y_corners, z_corners]
    if R is not None:
        corners = np.dot(R, corners)
    if location is not None:
        for i, loc in enumerate(location):
            corners[i, :] = corners[i, :] + loc
    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])
    return final_corners

