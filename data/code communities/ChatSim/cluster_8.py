# Cluster 8

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute('UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?', (model, width, height, array_to_blob(params), camera_id))
        return cursor.lastrowid

    def update_image(self, IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID):
        cursor = self.execute('UPDATE images SET prior_qw=?,  prior_qx=?, prior_qy=?, prior_qz=?, prior_tx=?, prior_ty=?, prior_tz=? ,camera_id=? WHERE image_id=?', (QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_ID))
        return cursor.lastrowid

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def camTodatabase(txtfile, dbfile):
    camModelDict = {'SIMPLE_PINHOLE': 0, 'PINHOLE': 1, 'SIMPLE_RADIAL': 2, 'RADIAL': 3, 'OPENCV': 4, 'FULL_OPENCV': 5, 'SIMPLE_RADIAL_FISHEYE': 6, 'RADIAL_FISHEYE': 7, 'OPENCV_FISHEYE': 8, 'FOV': 9, 'THIN_PRISM_FISHEYE': 10}
    db = COLMAPDatabase.connect(dbfile)
    idList = list()
    modelList = list()
    widthList = list()
    heightList = list()
    paramsList = list()
    with open(txtfile, 'r') as cam:
        lines = cam.readlines()
        for i in range(0, len(lines), 1):
            if lines[i][0] != '#':
                strLists = lines[i].split()
                cameraId = int(strLists[0])
                cameraModel = camModelDict[strLists[1]]
                width = int(strLists[2])
                height = int(strLists[3])
                paramstr = np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)
    db.commit()
    rows = db.execute('SELECT * FROM cameras')
    for i in range(0, len(idList), 1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and (height == heightList[i])
        assert np.allclose(params, paramsList[i])
    db.close()

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def imgTodatabase(txtfile, dbfile):
    db = COLMAPDatabase.connect(dbfile)
    with open(txtfile, 'r') as images:
        lines = images.readlines()
        for i in range(0, len(lines)):
            image_metas = lines[i].split()
            if len(image_metas) > 0:
                db.update_image(IMAGE_ID=int(image_metas[0]), QW=float(image_metas[1]), QX=float(image_metas[2]), QY=float(image_metas[3]), QZ=float(image_metas[4]), TX=float(image_metas[5]), TY=float(image_metas[6]), TZ=float(image_metas[7]), CAMERA_ID=int(image_metas[8]))
    db.commit()
    db.close()

def add_camera(db, model, width, height, params, prior_focal_length=False, camera_id=None):
    params = np.asarray(params, np.float64)
    db.execute('INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)', (camera_id, model, width, height, array_to_blob(params), prior_focal_length))

def add_descriptors(db, image_id, descriptors):
    descriptors = np.ascontiguousarray(descriptors, np.uint8)
    db.execute('INSERT INTO descriptors VALUES (?, ?, ?, ?)', (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

def add_inlier_matches(db, image_id1, image_id2, matches, config=2, F=None, E=None, H=None):
    assert len(matches.shape) == 2
    assert matches.shape[1] == 2
    if image_id1 > image_id2:
        matches = matches[:, ::-1]
    if F is not None:
        F = np.asarray(F, np.float64)
    if E is not None:
        E = np.asarray(E, np.float64)
    if H is not None:
        H = np.asarray(H, np.float64)
    pair_id = get_pair_id(image_id1, image_id2)
    matches = np.asarray(matches, np.uint32)
    db.execute('INSERT INTO inlier_matches VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (pair_id,) + matches.shape + (array_to_blob(matches), config, F, E, H))

def get_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = (image_id2, image_id1)
    return image_id1 * MAX_IMAGE_ID + image_id2

def add_keypoints(db, image_id, keypoints):
    assert len(keypoints.shape) == 2
    assert keypoints.shape[1] in [2, 4, 6]
    keypoints = np.asarray(keypoints, np.float32)
    db.execute('INSERT INTO keypoints VALUES (?, ?, ?, ?)', (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

def add_matches(db, image_id1, image_id2, matches):
    assert len(matches.shape) == 2
    assert matches.shape[1] == 2
    if image_id1 > image_id2:
        matches = matches[:, ::-1]
    pair_id = get_pair_id(image_id1, image_id2)
    matches = np.asarray(matches, np.uint32)
    db.execute('INSERT INTO matches VALUES (?, ?, ?, ?)', (pair_id,) + matches.shape + (array_to_blob(matches),))

def main(args):
    import os
    if os.path.exists(args.database_path):
        print('Error: database path already exists -- will not modify it.')
        exit()
    db = COLMAPDatabase.connect(args.database_path)
    db.initialize_tables()
    model1, w1, h1, params1 = (0, 1024, 768, np.array((1024.0, 512.0, 384.0)))
    model2, w2, h2, params2 = (2, 1024, 768, np.array((1024.0, 512.0, 384.0, 0.1)))
    db.add_camera(model1, w1, h1, params1)
    db.add_camera(model2, w2, h2, params2)
    db.add_image('image1.png', 0)
    db.add_image('image2.png', 0)
    db.add_image('image3.png', 2)
    db.add_image('image4.png', 2)
    N = 1000
    kp1 = np.random.rand(N, 2) * (1024.0, 768.0)
    kp2 = np.random.rand(N, 2) * (1024.0, 768.0)
    kp3 = np.random.rand(N, 2) * (1024.0, 768.0)
    kp4 = np.random.rand(N, 2) * (1024.0, 768.0)
    db.add_keypoints(1, kp1)
    db.add_keypoints(2, kp2)
    db.add_keypoints(3, kp3)
    db.add_keypoints(4, kp4)
    M = 50
    m12 = np.random.randint(N, size=(M, 2))
    m23 = np.random.randint(N, size=(M, 2))
    m34 = np.random.randint(N, size=(M, 2))
    db.add_matches(1, 2, m12)
    db.add_matches(2, 3, m23)
    db.add_matches(3, 4, m34)
    rows = db.execute('SELECT * FROM cameras')
    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float32)
    assert model == model1 and width == w1 and (height == h1)
    assert np.allclose(params, params1)
    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float32)
    assert model == model2 and width == w2 and (height == h2)
    assert np.allclose(params, params2)
    kps = dict(((image_id, blob_to_array(data, np.float32, (-1, 2))) for image_id, data in db.execute('SELECT image_id, data FROM keypoints')))
    assert np.allclose(kps[1], kp1)
    assert np.allclose(kps[2], kp2)
    assert np.allclose(kps[3], kp3)
    assert np.allclose(kps[4], kp4)
    pair_ids = [get_pair_id(*pair) for pair in [(1, 2), (2, 3), (3, 4)]]
    matches = dict(((get_image_ids_from_pair_id(pair_id), blob_to_array(data, np.uint32, (-1, 2))) for pair_id, data in db.execute('SELECT pair_id, data FROM matches')))
    assert np.all(matches[1, 2] == m12)
    assert np.all(matches[2, 3] == m23)
    assert np.all(matches[3, 4] == m34)
    db.close()
    os.remove(args.database_path)

def get_image_ids_from_pair_id(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    return ((pair_id - image_id2) / MAX_IMAGE_ID, image_id2)

def add_camera(db, model, width, height, params, prior_focal_length=False, camera_id=None):
    params = np.asarray(params, np.float64)
    db.execute('INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)', (camera_id, model, width, height, array_to_blob(params), prior_focal_length))

def add_descriptors(db, image_id, descriptors):
    descriptors = np.ascontiguousarray(descriptors, np.uint8)
    db.execute('INSERT INTO descriptors VALUES (?, ?, ?, ?)', (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

def add_inlier_matches(db, image_id1, image_id2, matches, config=2, F=None, E=None, H=None):
    assert len(matches.shape) == 2
    assert matches.shape[1] == 2
    if image_id1 > image_id2:
        matches = matches[:, ::-1]
    if F is not None:
        F = np.asarray(F, np.float64)
    if E is not None:
        E = np.asarray(E, np.float64)
    if H is not None:
        H = np.asarray(H, np.float64)
    pair_id = get_pair_id(image_id1, image_id2)
    matches = np.asarray(matches, np.uint32)
    db.execute('INSERT INTO inlier_matches VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (pair_id,) + matches.shape + (array_to_blob(matches), config, F, E, H))

def add_keypoints(db, image_id, keypoints):
    assert len(keypoints.shape) == 2
    assert keypoints.shape[1] in [2, 4, 6]
    keypoints = np.asarray(keypoints, np.float32)
    db.execute('INSERT INTO keypoints VALUES (?, ?, ?, ?)', (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

def add_matches(db, image_id1, image_id2, matches):
    assert len(matches.shape) == 2
    assert matches.shape[1] == 2
    if image_id1 > image_id2:
        matches = matches[:, ::-1]
    pair_id = get_pair_id(image_id1, image_id2)
    matches = np.asarray(matches, np.uint32)
    db.execute('INSERT INTO matches VALUES (?, ?, ?, ?)', (pair_id,) + matches.shape + (array_to_blob(matches),))

def main(args):
    import os
    if os.path.exists(args.database_path):
        print('Error: database path already exists -- will not modify it.')
        exit()
    db = COLMAPDatabase.connect(args.database_path)
    db.initialize_tables()
    model1, w1, h1, params1 = (0, 1024, 768, np.array((1024.0, 512.0, 384.0)))
    model2, w2, h2, params2 = (2, 1024, 768, np.array((1024.0, 512.0, 384.0, 0.1)))
    db.add_camera(model1, w1, h1, params1)
    db.add_camera(model2, w2, h2, params2)
    db.add_image('image1.png', 0)
    db.add_image('image2.png', 0)
    db.add_image('image3.png', 2)
    db.add_image('image4.png', 2)
    N = 1000
    kp1 = np.random.rand(N, 2) * (1024.0, 768.0)
    kp2 = np.random.rand(N, 2) * (1024.0, 768.0)
    kp3 = np.random.rand(N, 2) * (1024.0, 768.0)
    kp4 = np.random.rand(N, 2) * (1024.0, 768.0)
    db.add_keypoints(1, kp1)
    db.add_keypoints(2, kp2)
    db.add_keypoints(3, kp3)
    db.add_keypoints(4, kp4)
    M = 50
    m12 = np.random.randint(N, size=(M, 2))
    m23 = np.random.randint(N, size=(M, 2))
    m34 = np.random.randint(N, size=(M, 2))
    db.add_matches(1, 2, m12)
    db.add_matches(2, 3, m23)
    db.add_matches(3, 4, m34)
    rows = db.execute('SELECT * FROM cameras')
    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float32)
    assert model == model1 and width == w1 and (height == h1)
    assert np.allclose(params, params1)
    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float32)
    assert model == model2 and width == w2 and (height == h2)
    assert np.allclose(params, params2)
    kps = dict(((image_id, blob_to_array(data, np.float32, (-1, 2))) for image_id, data in db.execute('SELECT image_id, data FROM keypoints')))
    assert np.allclose(kps[1], kp1)
    assert np.allclose(kps[2], kp2)
    assert np.allclose(kps[3], kp3)
    assert np.allclose(kps[4], kp4)
    pair_ids = [get_pair_id(*pair) for pair in [(1, 2), (2, 3), (3, 4)]]
    matches = dict(((get_image_ids_from_pair_id(pair_id), blob_to_array(data, np.uint32, (-1, 2))) for pair_id, data in db.execute('SELECT pair_id, data FROM matches')))
    assert np.all(matches[1, 2] == m12)
    assert np.all(matches[2, 3] == m23)
    assert np.all(matches[3, 4] == m34)
    db.close()
    os.remove(args.database_path)

