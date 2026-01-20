# Cluster 2

def detect3d(reg_weights, model_select, source, calib_file, show_result, save_result, output_path):
    imgs_path = sorted(glob.glob(str(source) + '/*'))
    calib = str(calib_file)
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()
    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)
    for i, img_path in enumerate(imgs_path):
        img = cv2.imread(img_path)
        dets = detect2d(weights='yolov5s.pt', source=img_path, data='data/coco128.yaml', imgsz=[640, 640], device=0, classes=[0, 2, 3, 5])
        for det in dets:
            if not averages.recognized_class(det.detected_class):
                continue
            try:
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib)
            except:
                continue
            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = det.box_2d
            detected_class = det.detected_class
            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = input_img
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)
            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi
            plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray)
        if show_result:
            cv2.imshow('3d detection', img)
            cv2.waitKey(0)
        if save_result and output_path is not None:
            try:
                os.mkdir(output_path)
            except:
                pass
            cv2.imwrite(f'{output_path}/{i:03d}.png', img)

def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2
    return angle_bins

def main(opt):
    detect3d(reg_weights=opt.reg_weights, model_select=opt.model_select, source=opt.source, calib_file=opt.calib_file, show_result=opt.show_result, save_result=opt.save_result, output_path=opt.output_path)

@app.route('/upload', methods=['POST'])
def upload_file():
    FILENAME = {}
    image = request.files['image']
    image.save('static/image_eval.png')
    if 'image' in request.files:
        detect = True
        detect3d(reg_weights='weights/epoch_10.pkl', model_select='resnet', source='static', calib_file='eval/camera_cal/calib_cam_to_cam.txt', save_result=True, show_result=False, output_path='static/')
        with open('static/000.png', 'rb') as image_file:
            img_encode = base64.b64encode(image_file.read())
            to_send = 'data:image/png;base64, ' + str(img_encode, 'utf-8')
    else:
        detect = False
    return render_template('index.html', init=True, detect=detect, image_to_show=to_send)

class Dataset(data.Dataset):

    def __init__(self, path, bins=2, overlap=0.1):
        self.top_img_path = path + '/image_2/'
        self.top_label_path = path + '/label_2/'
        self.top_calib_path = path + '/calib/'
        self.global_calib = path + '/calib_cam_to_cam.txt'
        self.proj_matrix = get_P(self.global_calib)
        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_calib_path))]
        self.num_images = len(self.ids)
        self.bins = bins
        self.angle_bins = generate_bins(self.bins)
        self.interval = 2 * np.pi / self.bins
        self.overlap = overlap
        self.bin_ranges = []
        for i in range(0, bins):
            self.bin_ranges.append(((i * self.interval - overlap) % (2 * np.pi), (i * self.interval + self.interval + overlap) % (2 * np.pi)))
        class_list = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)
        self.object_list = self.get_objects(self.ids)
        self.labels = {}
        last_id = ''
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id
            self.labels[id][str(line_num)] = label
        self.curr_id = ''
        self.curr_img = None

    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]
        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = cv2.imread(self.top_img_path + f'{id}.png')
        label = self.labels[id][str(line_num)]
        obj = DetectedObject(self.curr_img, label['Class'], label['Box_2D'], self.proj_matrix, label=label)
        return (obj.img, label)

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        """
        Get objects parameter from labels, like dimension and class name
        """
        objects = []
        for id in ids:
            with open(self.top_label_path + f'{id}.txt') as file:
                for line_num, line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == 'DontCare':
                        continue
                    dimension = np.array([float(line[8]), float(line[9]), float(line[10])], dtype=np.double)
                    self.averages.add_item(obj_class, dimension)
                    objects.append((id, line_num))
        self.averages.dump_to_file()
        return objects

    def get_label(self, id, line_num):
        lines = open(self.top_label_path + f'{id}.txt').read().splitlines()
        label = self.format_label(lines[line_num])
        return label

    def get_bin(self, angle):
        bin_idxs = []

        def is_between(min, max, angle):
            max = max - min if max - min > 0 else max - min + 2 * np.pi
            angle = angle - min if angle - min > 0 else angle - min + 2 * np.pi
            return angle < max
        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)
        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(' ')
        Class = line[0]
        for i in range(1, len(line)):
            line[i] = float(line[i])
        Alpha = line[3]
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]
        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double)
        Dimension -= self.averages.get_item(Class)
        Location = [line[11], line[12], line[13]]
        Location[1] -= Dimension[0] / 2
        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)
        angle = Alpha + np.pi
        bin_idxs = self.get_bin(angle)
        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]
            Orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1
        label = {'Class': Class, 'Box_2D': Box_2D, 'Dimensions': Dimension, 'Alpha': Alpha, 'Orientation': Orientation, 'Confidence': Confidence}
        return label

def get_P(calib_file):
    """
    Get matrix P_rect_02 (camera 2 RGB)
    and transform to 3 x 4 matrix
    """
    for line in open(calib_file):
        if 'P_rect_02' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            matrix = np.zeros((3, 4))
            matrix = cam_P.reshape((3, 4))
            return matrix

class DetectedObject:
    """
    Processing image for NN input
    """

    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):
        if isinstance(proj_matrix, str):
            proj_matrix = get_P(proj_matrix)
        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        """
        Calculate global angle of object, see paper
        """
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - width / 2
        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan(2 * dx * np.tan(fovx / 2) / width)
        angle = angle * mult
        return angle

    def format_img(self, img, box_2d):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        process = transforms.Compose([transforms.ToTensor(), normalize])
        pt1, pt2 = (box_2d[0], box_2d[1])
        crop = img[pt1[1]:pt2[1] + 1, pt1[0]:pt2[0] + 1]
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        batch = process(crop)
        return batch

class Dataset(data.Dataset):

    def __init__(self, path, bins=2, overlap=0.1):
        self.top_img_path = path + '/image_2/'
        self.top_label_path = path + '/label_2/'
        self.top_calib_path = path + '/calib/'
        self.global_calib = path + '/calib_cam_to_cam.txt'
        self.proj_matrix = get_P(self.global_calib)
        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_calib_path))]
        self.num_images = len(self.ids)
        self.bins = bins
        self.angle_bins = generate_bins(self.bins)
        self.interval = 2 * np.pi / bins
        self.overlap = overlap
        self.bin_ranges = []
        for i in range(0, bins):
            self.bin_ranges.append(((i * self.interval - overlap) % (2 * np.pi), (i * self.interval + self.interval + overlap) % (2 * np.pi)))
        class_list = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)
        self.object_list = self.get_objects(self.ids)
        self.labels = {}
        last_id = ''
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id
            self.labels[id][str(line_num)] = label
        self.curr_id = ''
        self.curr_img = None

    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]
        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = cv2.imread(self.top_img_path + f'{id}.png')
        label = self.labels[id][str(line_num)]
        obj = DetectedObject(self.curr_img, label['Class'], label['Box_2D'], self.proj_matrix, label=label)
        return (obj.img, label)

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        """
        Get objects parameter from labels, like dimension and class name
        """
        objects = []
        for id in ids:
            with open(self.top_label_path + f'{id}.txt') as file:
                for line_num, line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == 'DontCare':
                        continue
                    dimension = np.array([float(line[8]), float(line[9]), float(line[10])], dtype=np.double)
                    self.averages.add_item(obj_class, dimension)
                    objects.append((id, line_num))
        self.averages.dump_to_file()
        return objects

    def get_label(self, id, line_num):
        lines = open(self.top_label_path + f'{id}.txt').read().splitlines()
        label = self.format_label(lines[line_num])
        return label

    def get_bin(self, angle):
        bin_idxs = []

        def is_between(min, max, angle):
            max = max - min if max - min > 0 else max - min + 2 * np.pi
            angle = angle - min if angle - min > 0 else angle - min + 2 * np.pi
            return angle < max
        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)
        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(' ')
        Class = line[0]
        for i in range(1, len(line)):
            line[i] = float(line[i])
        Alpha = line[3]
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]
        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double)
        Dimension -= self.averages.get_item(Class)
        Location = [line[11], line[12], line[13]]
        Location[1] -= Dimension[0] / 2
        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)
        angle = Alpha + np.pi
        bin_idxs = self.get_bin(angle)
        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]
            Orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1
        label = {'Class': Class, 'Box_2D': Box_2D, 'Dimensions': Dimension, 'Alpha': Alpha, 'Orientation': Orientation, 'Confidence': Confidence}
        return label

class DetectedObject:
    """
    Processing image for NN input
    """

    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):
        if isinstance(proj_matrix, str):
            proj_matrix = get_P(proj_matrix)
        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        """
        Calculate global angle of object, see paper
        """
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - width / 2
        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan(2 * dx * np.tan(fovx / 2) / width)
        angle = angle * mult
        return angle

    def format_img(self, img, box_2d):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        process = transforms.Compose([transforms.ToTensor(), normalize])
        pt1, pt2 = (box_2d[0], box_2d[1])
        crop = img[pt1[1]:pt2[1] + 1, pt1[0]:pt2[0] + 1]
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        batch = process(crop)
        return batch

