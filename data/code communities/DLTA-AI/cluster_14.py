# Cluster 14

class LabelFile(object):
    suffix = '.json'

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error('Failed opening image file: {}'.format(filename))
            return
        image_pil = utils.apply_exif_orientation(image_pil)
        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = 'PNG'
            elif ext in ['.jpg', '.jpeg']:
                format = 'JPEG'
            else:
                format = 'PNG'
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = ['version', 'imageData', 'imagePath', 'shapes', 'flags', 'imageHeight', 'imageWidth']
        shape_keys = ['label', 'points', 'bbox', 'group_id', 'shape_type', 'flags', 'content']
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            version = data.get('version')
            if version is None:
                logger.warn('Loading JSON file ({}) of unknown version'.format(filename))
            elif version.split('.')[0] != __version__.split('.')[0]:
                logger.warn('This JSON file ({}) may be incompatible with current labelme. version in file: {}, current version: {}'.format(filename, version, __version__))
            if data['imageData'] is not None:
                imageData = base64.b64decode(data['imageData'])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                imagePath = osp.join(osp.dirname(filename), data['imagePath'])
                imageData = self.load_image_file(imagePath)
            flags = data.get('flags') or {}
            imagePath = data['imagePath']
            self._check_image_height_and_width(base64.b64encode(imageData).decode('utf-8'), data.get('imageHeight'), data.get('imageWidth'))
            shapes = [dict(label=s['label'], points=s['points'], bbox=s['bbox'], shape_type=s.get('shape_type', 'polygon'), flags=s.get('flags', {}), content=s.get('content'), group_id=s.get('group_id'), other_data={k: v for k, v in s.items() if k not in shape_keys}) for s in data['shapes']]
        except Exception as e:
            raise LabelFileError(e)
        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error('imageHeight does not match with imageData or imagePath, so getting imageHeight from actual image.')
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error('imageWidth does not match with imageData or imagePath, so getting imageWidth from actual image.')
            imageWidth = img_arr.shape[1]
        return (imageHeight, imageWidth)

    def save(self, filename, shapes, imagePath, imageHeight, imageWidth, imageData=None, otherData=None, flags=None):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode('utf-8')
            imageHeight, imageWidth = self._check_image_height_and_width(imageData, imageHeight, imageWidth)
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(version=__version__, flags=flags, shapes=shapes, imagePath=imagePath, imageData=imageData, imageHeight=imageHeight, imageWidth=imageWidth)
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix

@contextlib.contextmanager
def open(name, mode):
    assert mode in ['r', 'w']
    if PY2:
        mode += 'b'
        encoding = None
    else:
        encoding = 'utf-8'
    yield io.open(name, mode, encoding=encoding)
    return

def main():
    logger.warning('This script is aimed to demonstrate how to convert the JSON file to a single image dataset.')
    logger.warning("It won't handle multiple JSON files to generate a real-use dataset.")
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()
    json_file = args.json_file
    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    data = json.load(open(json_file))
    imageData = data.get('imageData')
    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)
    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    lbl_viz = imgviz.label2rgb(label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb')
    PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')
    logger.info('Saved to: {}'.format(out_dir))

def labelme_on_docker(in_file, out_file):
    ip = get_ip()
    cmd = 'xhost + %s' % ip
    subprocess.check_output(shlex.split(cmd))
    if out_file:
        out_file = osp.abspath(out_file)
        if osp.exists(out_file):
            raise RuntimeError('File exists: %s' % out_file)
        else:
            open(osp.abspath(out_file), 'w')
    cmd = 'docker run -it --rm -e DISPLAY={0}:0 -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix -v {1}:{2} -w /home/developer'
    in_file_a = osp.abspath(in_file)
    in_file_b = osp.join('/home/developer', osp.basename(in_file))
    cmd = cmd.format(ip, in_file_a, in_file_b)
    if out_file:
        out_file_a = osp.abspath(out_file)
        out_file_b = osp.join('/home/developer', osp.basename(out_file))
        cmd += ' -v {0}:{1}'.format(out_file_a, out_file_b)
    cmd += ' wkentaro/labelme labelme {0}'.format(in_file_b)
    if out_file:
        cmd += ' -O {0}'.format(out_file_b)
    subprocess.call(shlex.split(cmd))
    if out_file:
        try:
            json.load(open(out_file))
            return out_file
        except Exception:
            if open(out_file).read() == '':
                os.remove(out_file)
            raise RuntimeError('Annotation is cancelled.')

def get_ip():
    dist = platform.platform().split('-')[0]
    if dist == 'Linux':
        return ''
    elif dist == 'Darwin':
        cmd = 'ifconfig en0'
        output = subprocess.check_output(shlex.split(cmd))
        if str != bytes:
            output = output.decode('utf-8')
        for row in output.splitlines():
            cols = row.strip().split(' ')
            if cols[0] == 'inet':
                ip = cols[1]
                return ip
        else:
            raise RuntimeError('No ip is found.')
    else:
        raise RuntimeError('Unsupported platform.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', help='Input file or directory.')
    parser.add_argument('-O', '--output')
    args = parser.parse_args()
    if not distutils.spawn.find_executable('docker'):
        print('Please install docker', file=sys.stderr)
        sys.exit(1)
    try:
        out_file = labelme_on_docker(args.in_file, args.output)
        if out_file:
            print('Saved to: %s' % out_file)
    except RuntimeError as e:
        sys.stderr.write(e.__str__() + '\n')
        sys.exit(1)

def PopUp():
    """
    Displays a dialog box for selecting and editing keyboard shortcuts for the application.

    Parameters:
    None

    Returns:
    None
    """
    shortcuts = {}
    with open('labelme/config/default_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        shortcuts = config.get('shortcuts', {})
    shortcuts_names_encode = {name: name.lower().capitalize().replace('_', ' ').replace('Sam', 'SAM').replace('sam', 'SAM') for name in shortcuts.keys()}
    shortcuts_names_decode = {value: key for key, value in shortcuts_names_encode.items()}
    shortcuts = {shortcuts_names_encode[key]: value for key, value in shortcuts.items()}
    shortcut_table = QtWidgets.QTableWidget()
    shortcut_table.setColumnCount(2)
    shortcut_table.setHorizontalHeaderLabels(['Function', 'Shortcut'])
    shortcut_table.setRowCount(len(shortcuts))
    shortcut_table.verticalHeader().setVisible(False)
    row = 0
    for name, key in shortcuts.items():
        name_item = QtWidgets.QTableWidgetItem(name)
        shortcut_item = QtWidgets.QTableWidgetItem(key)
        shortcut_table.setItem(row, 0, name_item)
        shortcut_table.setItem(row, 1, shortcut_item)
        row += 1

    def on_shortcut_table_clicked(item):
        row = item.row()
        name_item = shortcut_table.item(row, 0)
        name = name_item.text()
        current_key = shortcuts[name]
        key_edit = QtWidgets.QKeySequenceEdit(QtGui.QKeySequence(current_key))
        key_edit.setWindowTitle(f'Edit Shortcut for {name}')
        key_edit_label = QtWidgets.QLabel('Enter new shortcut for ' + name)
        dialog = QtWidgets.QDialog()
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        dialog.setWindowTitle('Shortcut Selector')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(key_edit_label)
        layout.addWidget(key_edit)
        ok_button = QtWidgets.QPushButton('OK')
        ok_button.clicked.connect(dialog.accept)
        null_hint_label = QtWidgets.QLabel("to remove shortcut, press 'Ctrl' only then click 'OK")
        layout.addWidget(ok_button)
        layout.addWidget(null_hint_label)
        dialog.setLayout(layout)
        if dialog.exec():
            key = key_edit.keySequence().toString(QtGui.QKeySequence.SequenceFormat.NativeText)
            if key in shortcuts.values() and list(shortcuts.keys())[list(shortcuts.values()).index(key)] != name:
                conflicting_shortcut = list(shortcuts.keys())[list(shortcuts.values()).index(key)]
                QtWidgets.QMessageBox.warning(None, 'Error', f'{key} is already assigned to {conflicting_shortcut}.')
            else:
                if key == '':
                    key = None
                shortcuts[name] = key
                shortcut_table.item(row, 1).setText(key)

    def write_shortcuts_to_ui(config):
        shortcuts = config.get('shortcuts', {})
        shortcuts_names_encode = {name: name.lower().capitalize().replace('_', ' ').replace('Sam', 'SAM').replace('sam', 'SAM') for name in shortcuts.keys()}
        shortcuts = {shortcuts_names_encode[key]: value for key, value in shortcuts.items()}
        row = 0
        for name, key in shortcuts.items():
            name_item = QtWidgets.QTableWidgetItem(name)
            shortcut_item = QtWidgets.QTableWidgetItem(key)
            shortcut_table.setItem(row, 0, name_item)
            shortcut_table.setItem(row, 1, shortcut_item)
            row += 1

    def on_reset_button_clicked():
        with open('labelme/config/default_config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        write_shortcuts_to_ui(config)

    def on_restore_button_clicked():
        with open('labelme/config/default_config_base.yaml', 'r') as f:
            configBase = yaml.load(f, Loader=yaml.FullLoader)
        with open('labelme/config/default_config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['shortcuts'] = configBase['shortcuts']
        write_shortcuts_to_ui(config)
    shortcut_table.itemClicked.connect(on_shortcut_table_clicked)
    dialog = QtWidgets.QDialog()
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
    dialog.setWindowTitle('Shortcuts')
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(shortcut_table)
    ok_button = QtWidgets.QPushButton('OK')
    ok_button.clicked.connect(dialog.accept)
    layout.addWidget(ok_button)
    reset_button = QtWidgets.QPushButton('Reset')
    reset_button.clicked.connect(on_reset_button_clicked)
    layout.addWidget(reset_button)
    restore_button = QtWidgets.QPushButton('Restore Default Shortcuts')
    restore_button.clicked.connect(on_restore_button_clicked)
    layout.addWidget(restore_button)
    note_label = QtWidgets.QLabel('Shortcuts will be updated after restarting the app.')
    layout.addWidget(note_label)
    dialog.setLayout(layout)
    dialog.setMinimumWidth(shortcut_table.sizeHintForColumn(0) + shortcut_table.sizeHintForColumn(1) + 55)
    dialog.setMinimumHeight(shortcut_table.rowHeight(0) * 10 + 50)
    dialog.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
    dialog.exec()
    shortcuts = {}
    for row in range(shortcut_table.rowCount()):
        name_item = shortcut_table.item(row, 0)
        name = name_item.text()
        shortcut_item = shortcut_table.item(row, 1)
        shortcut = shortcut_item.text()
        shortcuts[name] = shortcut if shortcut != '' else None
    shortcuts = {shortcuts_names_decode[key]: value for key, value in shortcuts.items()}
    with open('labelme/config/default_config.yaml', 'w') as f:
        config['shortcuts'] = shortcuts
        yaml.dump(config, f)

def on_reset_button_clicked():
    with open('labelme/config/default_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    write_shortcuts_to_ui(config)

def write_shortcuts_to_ui(config):
    shortcuts = config.get('shortcuts', {})
    shortcuts_names_encode = {name: name.lower().capitalize().replace('_', ' ').replace('Sam', 'SAM').replace('sam', 'SAM') for name in shortcuts.keys()}
    shortcuts = {shortcuts_names_encode[key]: value for key, value in shortcuts.items()}
    row = 0
    for name, key in shortcuts.items():
        name_item = QtWidgets.QTableWidgetItem(name)
        shortcut_item = QtWidgets.QTableWidgetItem(key)
        shortcut_table.setItem(row, 0, name_item)
        shortcut_table.setItem(row, 1, shortcut_item)
        row += 1

def on_restore_button_clicked():
    with open('labelme/config/default_config_base.yaml', 'r') as f:
        configBase = yaml.load(f, Loader=yaml.FullLoader)
    with open('labelme/config/default_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['shortcuts'] = configBase['shortcuts']
    write_shortcuts_to_ui(config)

class SegmentationOptionsUI:

    def __init__(self, parent):
        self.parent = parent
        self.conf_threshold = 0.3
        self.iou_threshold = 0.5
        with open('labelme/config/default_config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.default_classes = self.config['default_classes']
        try:
            self.selectedclasses = {}
            for class_ in self.default_classes:
                if class_ in coco_classes:
                    index = coco_classes.index(class_)
                    self.selectedclasses[index] = class_
        except:
            self.selectedclasses = {i: class_ for i, class_ in enumerate(coco_classes)}
            print('error in loading the default classes from the config file, so we will use all the coco classes')

    def setConfThreshold(self, prev_threshold=0.3):
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Threshold Selector')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel('Enter Confidence Threshold')
        layout.addWidget(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        slider.setValue(int(prev_threshold * 100))
        text_input = QtWidgets.QLineEdit(str(prev_threshold))

        def on_slider_change(value):
            text_input.setText(str(value / 100))

        def on_text_change(text):
            try:
                value = float(text)
                slider.setValue(int(value * 100))
            except ValueError:
                pass
        slider.valueChanged.connect(on_slider_change)
        text_input.textChanged.connect(on_text_change)
        layout.addWidget(slider)
        layout.addWidget(text_input)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(button_box)

        def on_ok():
            threshold = float(text_input.text())
            dialog.accept()
            return threshold

        def on_cancel():
            dialog.reject()
            return prev_threshold
        button_box.accepted.connect(on_ok)
        button_box.rejected.connect(on_cancel)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return slider.value() / 100
        else:
            return prev_threshold

    def setIOUThreshold(self, prev_threshold=0.5):
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Threshold Selector')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel('Enter IOU Threshold')
        layout.addWidget(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        slider.setValue(int(prev_threshold * 100))
        text_input = QtWidgets.QLineEdit(str(prev_threshold))

        def on_slider_change(value):
            text_input.setText(str(value / 100))

        def on_text_change(text):
            try:
                value = float(text)
                slider.setValue(int(value * 100))
            except ValueError:
                pass
        slider.valueChanged.connect(on_slider_change)
        text_input.textChanged.connect(on_text_change)
        layout.addWidget(slider)
        layout.addWidget(text_input)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(button_box)

        def on_ok():
            threshold = float(text_input.text())
            dialog.accept()
            return threshold

        def on_cancel():
            dialog.reject()
            return prev_threshold
        button_box.accepted.connect(on_ok)
        button_box.rejected.connect(on_cancel)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return slider.value() / 100
        else:
            return prev_threshold

    def selectClasses(self):
        """
        Display a dialog box that allows the user to select which classes to annotate.

        The function creates a QDialog object and adds various widgets to it, including a QScrollArea that contains QCheckBox
        widgets for each class. The function sets the state of each QCheckBox based on whether the class is in the
        self.selectedclasses dictionary. The function also adds "Select All", "Deselect All", "Select Classes", "Set as Default",
        and "Cancel" buttons to the dialog box. When the user clicks the "Select Classes" button, the function saves the selected
        classes to the self.selectedclasses dictionary and returns it.

        :return: A dictionary that maps class indices to class names for the selected classes.
        """
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Select Classes')
        dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dialog.resize(500, 500)
        dialog.setMinimumSize(QtCore.QSize(500, 500))
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        verticalLayout = QtWidgets.QVBoxLayout(dialog)
        verticalLayout.setObjectName('verticalLayout')
        horizontalLayout = QtWidgets.QHBoxLayout()
        selectAllButton = QtWidgets.QPushButton('Select All', dialog)
        deselectAllButton = QtWidgets.QPushButton('Deselect All', dialog)
        horizontalLayout.addWidget(selectAllButton)
        horizontalLayout.addWidget(deselectAllButton)
        verticalLayout.addLayout(horizontalLayout)
        scrollArea = QtWidgets.QScrollArea(dialog)
        scrollArea.setWidgetResizable(True)
        scrollArea.setObjectName('scrollArea')
        scrollAreaWidgetContents = QtWidgets.QWidget()
        scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 478, 478))
        scrollAreaWidgetContents.setObjectName('scrollAreaWidgetContents')
        gridLayout = QtWidgets.QGridLayout(scrollAreaWidgetContents)
        gridLayout.setObjectName('gridLayout')
        self.scrollAreaWidgetContents = scrollAreaWidgetContents
        scrollArea.setWidget(scrollAreaWidgetContents)
        verticalLayout.addWidget(scrollArea)
        buttonBox = QtWidgets.QDialogButtonBox(dialog)
        buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttonBox.setObjectName('buttonBox')
        buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText('Select Classes')
        defaultButton = QtWidgets.QPushButton('Set as Default', dialog)
        buttonBox.addButton(defaultButton, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok))
        buttonLayout.addWidget(defaultButton)
        buttonLayout.addWidget(buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel))
        verticalLayout.addLayout(buttonLayout)
        buttonBox.accepted.connect(lambda: self.saveClasses(dialog))
        buttonBox.rejected.connect(dialog.reject)
        defaultButton.clicked.connect(lambda: self.saveClasses(dialog, True))
        self.classes = []
        for i in range(len(coco_classes)):
            self.classes.append(QtWidgets.QCheckBox(coco_classes[i], dialog))
            row = i // 3
            col = i % 3
            gridLayout.addWidget(self.classes[i], row, col)
        for value in self.selectedclasses.values():
            if value != None:
                indx = coco_classes.index(value)
                self.classes[indx].setChecked(True)
        selectAllButton.clicked.connect(lambda: self.selectAll())
        deselectAllButton.clicked.connect(lambda: self.deselectAll())
        dialog.show()
        dialog.exec()
        self.selectedclasses.clear()
        for i in range(len(self.classes)):
            if self.classes[i].isChecked():
                indx = coco_classes.index(self.classes[i].text())
                self.selectedclasses[indx] = self.classes[i].text()
        return self.selectedclasses

    def saveClasses(self, dialog, is_default=False):
        """
        Save the selected classes to the self.selectedclasses dictionary.

        The function clears the self.selectedclasses dictionary and then iterates over the QCheckBox widgets for each class.
        If a QCheckBox is checked, the function adds the corresponding class name to the self.selectedclasses dictionary. If the
        is_default parameter is True, the function also updates the default_config.yaml file with the selected classes.

        :param dialog: The QDialog object that contains the class selection dialog.
        :param is_default: A boolean that indicates whether to update the default_config.yaml file with the selected classes.
        """
        self.selectedclasses.clear()
        for i in range(len(self.classes)):
            if self.classes[i].isChecked():
                indx = coco_classes.index(self.classes[i].text())
                self.selectedclasses[indx] = self.classes[i].text()
        if is_default:
            with open('labelme/config/default_config.yaml', 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            config['default_classes'] = list(self.selectedclasses.values())
            with open('labelme/config/default_config.yaml', 'w') as f:
                yaml.dump(config, f)
        dialog.accept()

    def selectAll(self):
        """
        Select all classes in the class selection dialog.

        The function iterates over the QCheckBox widgets for each class and sets their checked state to True.
        """
        for checkbox in self.classes:
            checkbox.setChecked(True)

    def deselectAll(self):
        """
        Deselect all classes in the class selection dialog.

        The function iterates over the QCheckBox widgets for each class and sets their checked state to False.
        """
        for checkbox in self.classes:
            checkbox.setChecked(False)

class MergeFeatureUI:

    def __init__(self, parent):
        self.parent = parent
        self.selectedmodels = []

    def mergeSegModels(self):
        models = []
        with open('saved_models.json') as json_file:
            data = json.load(json_file)
            for model in data.keys():
                if 'YOLOv8' not in model:
                    models.append(model)
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Select Models')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dialog.resize(200, 250)
        dialog.setMinimumSize(QtCore.QSize(200, 200))
        verticalLayout = QtWidgets.QVBoxLayout(dialog)
        verticalLayout.setObjectName('verticalLayout')
        scrollArea = QtWidgets.QScrollArea(dialog)
        scrollArea.setWidgetResizable(True)
        scrollArea.setObjectName('scrollArea')
        scrollAreaWidgetContents = QtWidgets.QWidget()
        scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 478, 478))
        scrollAreaWidgetContents.setObjectName('scrollAreaWidgetContents')
        verticalLayout_2 = QtWidgets.QVBoxLayout(scrollAreaWidgetContents)
        verticalLayout_2.setObjectName('verticalLayout_2')
        self.scrollAreaWidgetContents = scrollAreaWidgetContents
        scrollArea.setWidget(scrollAreaWidgetContents)
        verticalLayout.addWidget(scrollArea)
        buttonBox = QtWidgets.QDialogButtonBox(dialog)
        buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttonBox.setObjectName('buttonBox')
        verticalLayout.addWidget(buttonBox)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        self.models = []
        for i in range(len(models)):
            self.models.append(QtWidgets.QCheckBox(models[i], dialog))
            verticalLayout_2.addWidget(self.models[i])
        dialog.show()
        dialog.exec()
        self.selectedmodels.clear()
        for i in range(len(self.models)):
            if self.models[i].isChecked():
                self.selectedmodels.append(self.models[i].text())
        print(self.selectedmodels)
        return self.selectedmodels

def PopUp():
    """

    Description:
    This function displays a dialog box with preferences for the LabelMe application, including theme and notification settings.

    Parameters:
    This function takes no parameters.

    Returns:
    If the user clicks the OK button, this function writes the new theme and notification settings to the config file and returns `QtWidgets.QDialog.DialogCode.Accepted`. If the user clicks the Cancel button, this function does not write any changes to the config file and returns `QtWidgets.QDialog.Rejected`.

    Libraries:
    This function requires the following libraries to be installed:
    - yaml
    - PyQt6.QtWidgets
    - PyQt6.QtGui
    - PyQt6.QtCore
    """
    with open('labelme/config/default_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle('Preferences')
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
    themeLabel = QtWidgets.QLabel('Theme Settings 🌓')
    themeLabel.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Weight.Bold))
    theme_note_label = QtWidgets.QLabel('Requires app restart to take effect')
    notificationLabel = QtWidgets.QLabel('Notifications Settings 🔔')
    notificationLabel.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Weight.Bold))
    notification_note_label = QtWidgets.QLabel("Notifications works only for long tasks and if the app isn't focused")
    current_theme = config['theme']
    current_mute = config['mute']
    autoButton = QtWidgets.QRadioButton('OS Default')
    lightButton = QtWidgets.QRadioButton('Light')
    darkButton = QtWidgets.QRadioButton('Dark')
    if current_theme == 'auto':
        autoButton.setChecked(True)
    elif current_theme == 'light':
        lightButton.setChecked(True)
    elif current_theme == 'dark':
        darkButton.setChecked(True)
    autoImage = QtGui.QPixmap('labelme/icons/auto-img.png').scaledToWidth(128)
    lightImage = QtGui.QPixmap('labelme/icons/light-img.png').scaledToWidth(128)
    darkImage = QtGui.QPixmap('labelme/icons/dark-img.png').scaledToWidth(128)
    autoLabel = QtWidgets.QLabel()
    autoLabel.setPixmap(autoImage)
    lightLabel = QtWidgets.QLabel()
    lightLabel.setPixmap(lightImage)
    darkLabel = QtWidgets.QLabel()
    darkLabel.setPixmap(darkImage)
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(themeLabel)
    layout.addWidget(theme_note_label)
    buttonLayout = QtWidgets.QHBoxLayout()
    buttonLayout.addWidget(autoButton)
    buttonLayout.addWidget(lightButton)
    buttonLayout.addWidget(darkButton)
    layout.addLayout(buttonLayout)
    imageLayout = QtWidgets.QHBoxLayout()
    imageLayout.addWidget(autoLabel)
    imageLayout.addWidget(lightLabel)
    imageLayout.addWidget(darkLabel)
    layout.addLayout(imageLayout)
    notificationCheckbox = QtWidgets.QCheckBox('Mute Notifications')
    notificationCheckbox.setChecked(current_mute)
    layout.addWidget(notificationLabel)
    layout.addWidget(notification_note_label)
    layout.addWidget(notificationCheckbox)
    dialog.setLayout(layout)
    okButton = QtWidgets.QPushButton('OK')
    cancelButton = QtWidgets.QPushButton('Cancel')
    buttonLayout = QtWidgets.QHBoxLayout()
    buttonLayout.addWidget(okButton)
    buttonLayout.addWidget(cancelButton)
    layout.addLayout(buttonLayout)
    okButton.clicked.connect(dialog.accept)
    cancelButton.clicked.connect(dialog.reject)
    if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        if autoButton.isChecked():
            theme = 'auto'
        elif lightButton.isChecked():
            theme = 'light'
        elif darkButton.isChecked():
            theme = 'dark'
        mute = notificationCheckbox.isChecked()
        with open('labelme/config/default_config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['theme'] = theme
        config['mute'] = mute
        with open('labelme/config/default_config.yaml', 'w') as f:
            yaml.dump(config, f)

class ModelExplorerDialog(QDialog):
    """
    A dialog window for exploring available models and downloading them.

    Attributes:
        main_window (QMainWindow): The main window of the application.
        mute (bool): Whether to mute notifications or not.
        notification (function): A function for displaying notifications.
    """

    def __init__(self, main_window=None, mute=None, notification=None):
        """
        Initializes the ModelExplorerDialog.

        Args:
            main_window (QMainWindow): The main window of the application.
            mute (bool): Whether to mute notifications.
            notification (function): A function for displaying notifications.
        """
        super().__init__()
        self.main_window = main_window
        self.mute = mute
        self.notification = notification
        self.setWindowTitle('Model Explorer')
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        self.cols_labels = ['id', 'Model Name', 'Backbone', 'Lr schd', 'Memory (GB)', 'Inference Time (fps)', 'box AP', 'mask AP', 'Checkpoint Size (MB)']
        self.model_keys = sorted(list(set([model['Model'] for model in models_json])))
        layout = QVBoxLayout()
        self.setLayout(layout)
        toolbar = QToolBar()
        layout.addWidget(toolbar)
        self.model_type_dropdown = QComboBox()
        self.model_type_dropdown.addItems(['All'] + self.model_keys)
        self.model_type_dropdown.currentIndexChanged.connect(self.search)
        toolbar.addWidget(self.model_type_dropdown)
        self.available_checkbox = QCheckBox('Downloaded')
        self.available_checkbox.clicked.connect(self.search)
        toolbar.addWidget(self.available_checkbox)
        self.not_available_checkbox = QCheckBox('Not Downloaded')
        self.not_available_checkbox.clicked.connect(self.search)
        toolbar.addWidget(self.not_available_checkbox)
        open_checkpoints_dir_button = QPushButton('Open Checkpoints Dir')
        open_checkpoints_dir_button.setIcon(QtGui.QIcon(cwd + '/labelme/icons/downloads.png'))
        open_checkpoints_dir_button.setIconSize(QtCore.QSize(20, 20))
        open_checkpoints_dir_button.clicked.connect(self.open_checkpoints_dir)
        toolbar.addWidget(open_checkpoints_dir_button)
        layout.setSpacing(10)
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.num_rows = len(models_json)
        self.num_cols = 9
        self.check_availability()
        self.populate_table()
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        close_button = QPushButton('Ok')
        close_button.clicked.connect(self.close)
        close_button.setFixedWidth(100)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        layout.setSpacing(10)

    def populate_table(self):
        """
        Populates the table with data from models_json.

        Returns:
            None
        """
        self.table.clearContents()
        self.table.setRowCount(self.num_rows)
        self.table.setColumnCount(self.num_cols + 2)
        header = self.table.horizontalHeader()
        self.table.setHorizontalHeaderLabels(self.cols_labels + ['Status', 'Select Model'])
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        row_count = 0
        for model in models_json:
            col_count = 0
            for key in self.cols_labels:
                item = QTableWidgetItem(f'{model[key]}')
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_count, col_count, item)
                col_count += 1
            self.selected_model = (-1, -1, -1)
            select_row_button = QPushButton('Select Model')
            select_row_button.clicked.connect(self.select_model)
            self.table.setContentsMargins(10, 10, 10, 10)
            self.table.setCellWidget(row_count, 10, select_row_button)
            if model['Downloaded']:
                available_item = QTableWidgetItem('Downloaded')
                available_item.setForeground(QtCore.Qt.GlobalColor.darkGreen)
                self.table.setItem(row_count, 9, available_item)
                available_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            else:
                available_item = QPushButton('Requires Download')
                available_item.clicked.connect(self.create_download_callback(model['id']))
                available_item.setContentsMargins(10, 10, 10, 10)
                available_item.setStyleSheet('color: red')
                self.table.setCellWidget(row_count, 9, available_item)
                select_row_button.setEnabled(False)
            if model['Model'] == 'SAM':
                select_row_button.setEnabled(False)
                select_row_button.setText('Select from SAM Toolbar')
            row_count += 1

    def search(self):
        """
        Filters the table based on the selected model type and availability.

        Returns:
            None
        """
        model_type = self.model_type_dropdown.currentText()
        available = self.available_checkbox.isChecked()
        not_available = self.not_available_checkbox.isChecked()
        for row in range(self.num_rows):
            show_row = True
            if model_type != 'All':
                id = int(self.table.item(row, 0).text())
                if models_json[id]['Model'] != model_type:
                    show_row = False
            if available or not_available:
                available_text = self.table.item(row, 9)
                try:
                    available_text = available_text.text()
                except AttributeError:
                    pass
                if available and available_text != 'Downloaded':
                    show_row = False
                if not_available and available_text == 'Downloaded':
                    show_row = False
            self.table.setRowHidden(row, not show_row)

    def select_model(self):
        """
        Gets the selected model from the table and sets it as the selected model.

        Returns:
            None
        """
        sender = self.sender()
        index = self.table.indexAt(sender.pos())
        row = index.row()
        model_id = int(self.table.item(row, 0).text())
        self.selected_model = (models_json[model_id]['Model Name'], models_json[model_id]['Config'], models_json[model_id]['Checkpoint'])
        self.accept()

    def download_model(self, id):
        """
        Downloads the model with the given id and updates the progress dialog.

        Args:
            id (int): The id of the model to download.

        Returns:
            None
        """
        checkpoint_link = models_json[id]['Checkpoint_link']
        model_name = models_json[id]['Model Name']
        self.progress_dialog = QProgressDialog(f'Downloading {model_name}...', 'Cancel', 0, 100, self)
        self.progress_dialog.setWindowTitle('Downloading Model')
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_download)
        self.progress_dialog.show()
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_downloaded = 0
        self.download_canceled = False

        def handle_progress(block_num, block_size, total_size):
            """
            Updates the progress dialog with the current download progress.

            Args:
                block_num (int): The number of blocks downloaded.
                block_size (int): The size of each block.
                total_size (int): The total size of the file being downloaded.

            Returns:
                None
            """
            read_data = block_num * block_size
            if total_size > 0:
                download_percentage = read_data * 100 / total_size
                self.progress_dialog.setValue(download_percentage)
                self.progress_dialog.setLabelText(f'Downloading {model_name}... ')
                QApplication.processEvents()
        failed = False
        try:
            response = requests.get(checkpoint_link, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            block_num = 0
            file_path = f'{cwd}/mmdetection/checkpoints/{checkpoint_link.split('/')[-1]}'
            with open(file_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    if self.download_canceled:
                        break
                    f.write(data)
                    block_num += 1
                    handle_progress(block_num, block_size, total_size)
            if self.download_canceled:
                os.remove(file_path)
                print('Download canceled by user')
                failed = True
        except Exception as e:
            os.remove(file_path)
            print(f'Download error: {e}')
            failed = True
        self.progress_dialog.close()
        self.check_availability()
        self.populate_table()
        print('Download finished')
        try:
            if not self.mute:
                if not self.isActiveWindow():
                    if not failed:
                        self.notification(f'{model_name} has been downloaded successfully')
                    else:
                        self.notification(f'Failed to download {model_name}')
        except:
            pass

    def cancel_download(self):
        """
        Sets the download_canceled flag to True to cancel the download.

        Returns:
            None
        """
        self.download_canceled = True

    def create_download_callback(self, model_id):
        """
        Returns a lambda function that downloads the model with the given id.

        Args:
            model_id (int): The id of the model to download.

        Returns:
            function: A lambda function that downloads the model with the given id.
        """
        return lambda: self.download_model(model_id)

    def check_availability(self):
        """
        Checks the availability of each model in the table and updates the "Downloaded" column.

        Returns:
            None
        """
        checkpoints_dir = cwd + '/mmdetection/checkpoints/'
        for model in models_json:
            if model['Checkpoint'].split('/')[-1] in os.listdir(checkpoints_dir):
                model['Downloaded'] = True
            else:
                model['Downloaded'] = False

    def open_checkpoints_dir(self):
        """
        Opens the directory containing the downloaded checkpoints in the file explorer.

        Returns:
            None
        """
        url = QtCore.QUrl.fromLocalFile(cwd + '/mmdetection/checkpoints/')
        if not QtGui.QDesktopServices.openUrl(url):
            print('Failed to open checkpoints directory')

def exportCOCO(json_paths, annotation_path):
    """
    Export annotations in COCO format from a directory of JSON files for image and dir modes

    Args:
        target_directory (str): The directory containing the JSON files (dir)
        save_path (str): The path to save the output file (image mode)
        annotation_path (str): The path to the output file.

    Returns:
        str: The path to the output file.

    Raises:
        ValueError: If no JSON files are found in the directory.

    """
    file = {}
    file['info'] = {'description': 'Exported from DLTA-AI', 'year': datetime.datetime.now().year, 'date_created': datetime.date.today().strftime('%Y/%m/%d')}
    used_classes = set()
    annotations = []
    images = []
    for i in range(len(json_paths)):
        try:
            with open(json_paths[i]) as f:
                data = json.load(f)
                images.append({'id': i, 'width': data['imageWidth'], 'height': data['imageHeight'], 'file_name': json_paths[i].split('/')[-1].replace('.json', '.jpg')})
                for j in range(len(data['shapes'])):
                    if len(data['shapes'][j]['points']) == 0:
                        continue
                    if data['shapes'][j]['label'].lower() not in coco_classes:
                        print(f'{data['shapes'][j]['label']} is not a valid COCO class.. Adding it to the list.')
                        coco_classes.append(data['shapes'][j]['label'].lower())
                    annotations.append({'id': len(annotations), 'image_id': i, 'category_id': coco_classes.index(data['shapes'][j]['label'].lower()) + 1, 'bbox': get_bbox(data['shapes'][j]['points']), 'iscrowd': 0})
                    try:
                        annotations[-1]['segmentation'] = [data['shapes'][j]['points']]
                        annotations[-1]['area'] = get_area_from_polygon(annotations[-1]['segmentation'][0], mode='segmentation')
                    except:
                        annotations[-1]['area'] = get_area_from_polygon(annotations[-1]['bbox'], mode='bbox')
                    try:
                        annotations[-1]['score'] = float(data['shapes'][j]['content'])
                    except:
                        pass
                    used_classes.add(coco_classes.index(data['shapes'][j]['label'].lower()) + 1)
        except Exception as e:
            print(f'Error with {json_paths[i]}')
            print(e)
            continue
    used_classes = sorted(used_classes)
    file['categories'] = [{'id': i, 'name': coco_classes[i - 1]} for i in used_classes]
    file['images'] = images
    file['annotations'] = annotations
    with open(annotation_path, 'w') as outfile:
        json.dump(file, outfile, indent=4)
    return annotation_path

def get_bbox(segmentation):
    """
    Calculates the bounding box of a polygon defined by a list of consecutive pairs of x-y coordinates.

    Args:
        segmentation (list): A list of consecutive pairs of x-y coordinates that define a polygon.

    Returns:
        list: A list of four values: the minimum x and y values, and the width and height of the bounding box that encloses the polygon.
    """
    try:
        x = []
        y = []
        for i in range(len(segmentation)):
            if i % 2 == 0:
                x.append(segmentation[i])
            else:
                y.append(segmentation[i])
        return [min(x), min(y), max(x) - min(x), max(y) - min(y)]
    except:
        segmentation = [item for sublist in segmentation for item in sublist]
        x = []
        y = []
        for i in range(len(segmentation)):
            if i % 2 == 0:
                x.append(segmentation[i])
            else:
                y.append(segmentation[i])
        return [min(x), min(y), max(x) - min(x), max(y) - min(y)]

def get_area_from_polygon(polygon, mode='segmentation'):
    """
    Calculates the area of a polygon defined by a list of consecutive pairs of x-y coordinates.

    Args:
        polygon (list): A list of consecutive pairs of x-y coordinates that define a polygon.
        mode (str): The mode to use for calculating the area. Can be "segmentation" (default) or "bbox".

    Returns:
        float: The area of the polygon.
    """
    if mode == 'segmentation':
        polygon = np.array(polygon).reshape(-1, 2)
        area = 0.5 * np.abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
        return area
    elif mode == 'bbox':
        x_min, y_min, width, height = polygon
        area = width * height
        return area
    else:
        raise ValueError("mode must be either 'segmentation' or 'bbox'")

def exportCOCOvid(results_file, vid_width, vid_height, annotation_path):
    """
    Export object detection results in COCO format for a video.

    Args:
        results_file (str): Path to the JSON file containing the object detection results.
        vid_width (int): Width of the video frames.
        vid_height (int): Height of the video frames.
        annotation_path (str): Path to the output COCO annotation file.

    Returns:
        str: Path to the output COCO annotation file.

    Raises:
        ValueError: If no object detection results are found in the JSON file.

    """
    file = {}
    file['info'] = {'description': 'Exported from DLTA-AI', 'year': datetime.datetime.now().year, 'date_created': datetime.date.today().strftime('%Y/%m/%d')}
    annotations = []
    images = []
    used_classes = set()
    with open(results_file) as f:
        data = json.load(f)
        for frame in data:
            if len(frame['frame_data']) == 0:
                continue
            images.append({'id': frame['frame_idx'], 'width': vid_width, 'height': vid_height, 'file_name': f'frame {frame['frame_idx']}'})
            for object in frame['frame_data']:
                annotations.append({'id': len(annotations), 'image_id': frame['frame_idx'], 'category_id': object['class_id'] + 1, 'iscrowd': 0})
                if annotations[-1]['category_id'] == 0:
                    coco_classes.append(object['class_name'].lower())
                    annotations[-1]['category_id'] = coco_classes.index(object['class_name'].lower()) + 1
                try:
                    annotations[-1]['bbox'] = get_bbox(object['segment'])
                    annotations[-1]['segmentation'] = [[val for sublist in object['segment'] for val in sublist]]
                    annotations[-1]['area'] = get_area_from_polygon(annotations[-1]['segmentation'][0], mode='segmentation')
                except:
                    annotations[-1]['bbox'] = object['bbox']
                    annotations[-1]['area'] = get_area_from_polygon(annotations[-1]['bbox'], mode='bbox')
                try:
                    annotations[-1]['score'] = float(object['confidence'])
                except:
                    pass
                used_classes.add(annotations[-1]['category_id'])
    used_classes = sorted(list(used_classes))
    file['categories'] = [{'id': i, 'name': coco_classes[i - 1]} for i in used_classes]
    file['images'] = images
    file['annotations'] = annotations
    with open(annotation_path, 'w') as outfile:
        json.dump(file, outfile, indent=4)
    return annotation_path

def exportMOT(results_file, annotation_path):
    """
    Export object tracking results in MOT format.

    Args:
        results_file (str): Path to the JSON file containing the object tracking results.
        annotation_path (str): Path to the output MOT annotation file.

    Returns:
        str: Path to the output MOT annotation file.

    """
    with open(results_file) as f, open(annotation_path, 'w') as outfile:
        for frame in json.load(f):
            for object in frame['frame_data']:
                outfile.write(f'{frame['frame_idx']}, {object['tracker_id']},  {object['bbox'][0]},  {object['bbox'][1]},  {object['bbox'][2]},  {object['bbox'][3]},  {object['confidence']}, {object['class_id'] + 1}, 1\n')
    return annotation_path

def count_objects(json_paths, annotation_path):
    import matplotlib.pyplot as plt
    import json
    labels = []
    counts = []
    for i in range(len(json_paths)):
        with open(json_paths[i]) as f:
            labels.append(json_paths[i].split('time_')[-1].split('.')[0].replace('_', ':')[-5:])
            data = json.load(f)
            inner_count = 0
            for j in range(len(data['shapes'])):
                inner_count += 1
            counts.append(inner_count)
        plt.figure(figsize=(20, 12))
        plt.plot(counts)
        plt.title('Number of Objects Over Time')
        plt.tight_layout(pad=3)
        plt.grid()
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(max(counts) + 1))
        plt.xlabel('Time')
        plt.ylabel('Number of Objects')
        plt.savefig(annotation_path)
        plt.close()
    return annotation_path

def load_objects_from_json__json(json_file_name, nTotalFrames):
    """
    Summary:
        Load objects from a json file using json library.
        
    Args:
        json_file_name: the name of the json file
        nTotalFrames: the total number of frames
        
    Returns:
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
    """
    listObj = [{'frame_idx': i + 1, 'frame_data': []} for i in range(nTotalFrames)]
    if not os.path.exists(json_file_name):
        with open(json_file_name, 'w') as jf:
            json.dump(listObj, jf, indent=4, separators=(',', ': '))
        jf.close()
    with open(json_file_name, 'r') as jf:
        listObj = json.load(jf)
    jf.close()
    return listObj

def load_objects_to_json__json(json_file_name, listObj):
    """
    Summary:
        Load objects to a json file using json library.
        
    Args:
        json_file_name: the name of the json file
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
        
    Returns:
        None
    """
    with open(json_file_name, 'w') as json_file:
        json.dump(listObj, json_file, indent=4, separators=(',', ': '))
    json_file.close()

def load_objects_from_json__orjson(json_file_name, nTotalFrames):
    """
    Summary:
        Load objects from a json file using orjson library.
        
    Args:
        json_file_name: the name of the json file
        nTotalFrames: the total number of frames
        
    Returns:
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
    """
    listObj = [{'frame_idx': i + 1, 'frame_data': []} for i in range(nTotalFrames)]
    if not os.path.exists(json_file_name):
        with open(json_file_name, 'wb') as jf:
            jf.write(orjson.dumps(listObj))
        jf.close()
    with open(json_file_name, 'rb') as jf:
        listObj = orjson.loads(jf.read())
    jf.close()
    return listObj

def load_objects_to_json__orjson(json_file_name, listObj):
    """
    Summary:
        Load objects to a json file using orjson library.
        
    Args:
        json_file_name: the name of the json file
        listObj: a list of objects (each object is a dictionary of a frame with keys (frame_idx, frame_data))
        
    Returns:
        None
    """
    with open(json_file_name, 'wb') as jf:
        jf.write(orjson.dumps(listObj, option=orjson.OPT_INDENT_2))
    jf.close()

def update_saved_models_json(cwd):
    """
    Summary:
        Update the saved models json file.
    """
    checkpoints_dir = cwd + '/mmdetection/checkpoints/'
    try:
        files = os.listdir(checkpoints_dir)
    except:
        os.mkdir(checkpoints_dir)
    with open(cwd + '/models_menu/models_json.json') as f:
        models_json = json.load(f)
    saved_models = {}
    for model in models_json:
        if model['Model'] != 'SAM':
            if model['Checkpoint'].split('/')[-1] in os.listdir(checkpoints_dir):
                saved_models[model['Model Name']] = {'id': model['id'], 'checkpoint': model['Checkpoint'], 'config': model['Config']}
    with open(cwd + '/saved_models.json', 'w') as f:
        json.dump(saved_models, f, indent=4)

def update_dict(target_dict, new_dict, validate_item=None):
    for key, value in new_dict.items():
        if validate_item:
            validate_item(key, value)
        if key not in target_dict:
            logger.warn('Skipping unexpected key in config: {}'.format(key))
            continue
        if isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict(target_dict[key], value, validate_item=validate_item)
        else:
            target_dict[key] = value

def get_default_config():
    config_file = osp.join(here, 'default_config.yaml')
    with open(config_file) as f:
        config = yaml.safe_load(f)
    user_config_file = osp.join(osp.expanduser('~'), '.labelmerc')
    if not osp.exists(user_config_file):
        try:
            shutil.copy(config_file, user_config_file)
        except Exception:
            logger.warn('Failed to save config: {}'.format(user_config_file))
    return config

def get_config(config_file_or_yaml=None, config_from_args=None):
    config = get_default_config()
    if config_file_or_yaml is not None:
        config_from_yaml = yaml.safe_load(config_file_or_yaml)
        if not isinstance(config_from_yaml, dict):
            with open(config_from_yaml) as f:
                logger.info('Loading config file from: {}'.format(config_from_yaml))
                config_from_yaml = yaml.safe_load(f)
        update_dict(config, config_from_yaml, validate_item=validate_config_item)
    if config_from_args is not None:
        update_dict(config, config_from_args, validate_item=validate_config_item)
    return config

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def parse_require_file(fpath):
    with open(fpath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line and (not line.startswith('#')):
                for info in parse_line(line):
                    yield info

def parse_line(line):
    """Parse information from a line in a requirements text file."""
    if line.startswith('-r '):
        target = line.split(' ')[1]
        for info in parse_require_file(target):
            yield info
    else:
        info = {'line': line}
        if line.startswith('-e '):
            info['package'] = line.split('#egg=')[1]
        elif '@git+' in line:
            info['package'] = line
        else:
            pat = '(' + '|'.join(['>=', '==', '>']) + ')'
            parts = re.split(pat, line, maxsplit=1)
            parts = [p.strip() for p in parts]
            info['package'] = parts[0]
            if len(parts) > 1:
                op, rest = parts[1:]
                if ';' in rest:
                    version, platform_deps = map(str.strip, rest.split(';'))
                    info['platform_deps'] = platform_deps
                else:
                    version = rest
                info['version'] = (op, version)
        yield info

def gen_packages_items():
    if exists(require_fpath):
        for info in parse_require_file(require_fpath):
            parts = [info['package']]
            if with_version and 'version' in info:
                parts.extend(info['version'])
            if not sys.version.startswith('3.4'):
                platform_deps = info.get('platform_deps')
                if platform_deps is not None:
                    parts.append(';' + platform_deps)
            item = ''.join(parts)
            yield item

def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]
                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and (not line.startswith('#')):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item
    packages = list(gen_packages_items())
    return packages

def add_mim_extension():
    """Add extra files that are required to support MIM into the package.

    These files will be added by creating a symlink to the originals if the
    package is installed in `editable` mode (e.g. pip install -e .), or by
    copying from the originals otherwise.
    """
    if 'develop' in sys.argv:
        if platform.system() == 'Windows':
            mode = 'copy'
        else:
            mode = 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        mode = 'copy'
    else:
        return
    filenames = ['tools', 'configs', 'demo', 'model-index.yml']
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'mmdet', '.mim')
    os.makedirs(mim_path, exist_ok=True)
    for filename in filenames:
        if osp.exists(filename):
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(mim_path, filename)
            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)
            if mode == 'symlink':
                src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                os.symlink(src_relpath, tar_path)
            elif mode == 'copy':
                if osp.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)
                else:
                    warnings.warn(f'Cannot copy file {src_path}.')
            else:
                raise ValueError(f'Invalid mode {mode}')

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def parse_config(config_strings):
    temp_file = tempfile.NamedTemporaryFile()
    config_path = f'{temp_file.name}.py'
    with open(config_path, 'w') as f:
        f.write(config_strings)
    config = Config.fromfile(config_path)
    if config.model.bbox_head.type != 'SSDHead':
        raise AssertionError('This is not a SSD model.')

def get_metas_from_csv_style_ann_file(ann_file):
    data_infos = []
    cp_filename = None
    with open(ann_file, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            img_id = line[0]
            filename = f'{img_id}.jpg'
            if filename != cp_filename:
                data_infos.append(dict(filename=filename))
                cp_filename = filename
    return data_infos

def get_metas_from_txt_style_ann_file(ann_file):
    with open(ann_file) as f:
        lines = f.readlines()
    i = 0
    data_infos = []
    while i < len(lines):
        filename = lines[i].rstrip()
        data_infos.append(dict(filename=filename))
        skip_lines = int(lines[i + 2]) + 3
        i += skip_lines
    return data_infos

def main():
    args = parse_args()
    assert args.out.endswith('pkl'), 'The output file name must be pkl suffix'
    cfg = Config.fromfile(args.config)
    ann_file = cfg.data.test.ann_file
    img_prefix = cfg.data.test.img_prefix
    print(f'{'-' * 5} Start Processing {'-' * 5}')
    if ann_file.endswith('csv'):
        data_infos = get_metas_from_csv_style_ann_file(ann_file)
    elif ann_file.endswith('txt'):
        data_infos = get_metas_from_txt_style_ann_file(ann_file)
    else:
        shuffix = ann_file.split('.')[-1]
        raise NotImplementedError(f'File name must be csv or txt suffix but get {shuffix}')
    print(f'Successfully load annotation file from {ann_file}')
    print(f'Processing {len(data_infos)} images...')
    pool = Pool(args.nproc)
    image_metas = pool.starmap(get_image_metas, zip(data_infos, [img_prefix for _ in range(len(data_infos))]))
    pool.close()
    root_path = cfg.data.test.ann_file.rsplit('/', 1)[0]
    save_path = osp.join(root_path, args.out)
    mmcv.dump(image_metas, save_path)
    print(f'Image meta file save to: {save_path}')

def test_dataset_evaluation():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_dummy_coco_json(fake_json_file)
    coco_dataset = CocoDataset(ann_file=fake_json_file, classes=('car',), pipeline=[])
    fake_results = _create_dummy_results()
    eval_results = coco_dataset.evaluate(fake_results, classwise=True)
    assert eval_results['bbox_mAP'] == 1
    assert eval_results['bbox_mAP_50'] == 1
    assert eval_results['bbox_mAP_75'] == 1
    fake_concat_results = _create_dummy_results() + _create_dummy_results()
    coco_cfg = dict(type='CocoDataset', ann_file=fake_json_file, classes=('car',), pipeline=[])
    concat_cfgs = [coco_cfg, coco_cfg]
    concat_dataset = build_dataset(concat_cfgs)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_bbox_mAP'] == 1
    assert eval_results['0_bbox_mAP_50'] == 1
    assert eval_results['0_bbox_mAP_75'] == 1
    assert eval_results['1_bbox_mAP'] == 1
    assert eval_results['1_bbox_mAP_50'] == 1
    assert eval_results['1_bbox_mAP_75'] == 1
    coco_cfg = dict(type='CocoDataset', ann_file=[fake_json_file, fake_json_file], classes=('car',), pipeline=[])
    concat_dataset = build_dataset(coco_cfg)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_bbox_mAP'] == 1
    assert eval_results['0_bbox_mAP_50'] == 1
    assert eval_results['0_bbox_mAP_75'] == 1
    assert eval_results['1_bbox_mAP'] == 1
    assert eval_results['1_bbox_mAP_50'] == 1
    assert eval_results['1_bbox_mAP_75'] == 1
    fake_pkl_file = osp.join(tmp_dir.name, 'fake_data.pkl')
    _create_dummy_custom_pkl(fake_pkl_file)
    custom_dataset = CustomDataset(ann_file=fake_pkl_file, classes=('car',), pipeline=[])
    fake_results = _create_dummy_results()
    eval_results = custom_dataset.evaluate(fake_results)
    assert eval_results['mAP'] == 1
    fake_concat_results = _create_dummy_results() + _create_dummy_results()
    custom_cfg = dict(type='CustomDataset', ann_file=fake_pkl_file, classes=('car',), pipeline=[])
    concat_cfgs = [custom_cfg, custom_cfg]
    concat_dataset = build_dataset(concat_cfgs)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_mAP'] == 1
    assert eval_results['1_mAP'] == 1
    concat_cfg = dict(type='CustomDataset', ann_file=[fake_pkl_file, fake_pkl_file], classes=('car',), pipeline=[])
    concat_dataset = build_dataset(concat_cfg)
    eval_results = concat_dataset.evaluate(fake_concat_results)
    assert eval_results['0_mAP'] == 1
    assert eval_results['1_mAP'] == 1
    concat_cfg = dict(type='ConcatDataset', datasets=[custom_cfg, custom_cfg], separate_eval=False)
    concat_dataset = build_dataset(concat_cfg)
    eval_results = concat_dataset.evaluate(fake_concat_results, metric='mAP')
    assert eval_results['mAP'] == 1
    assert len(concat_dataset.datasets[0].data_infos) == len(concat_dataset.datasets[1].data_infos)
    assert len(concat_dataset.datasets[0].data_infos) == 1
    tmp_dir.cleanup()

def _create_dummy_coco_json(json_name):
    image = {'id': 0, 'width': 640, 'height': 640, 'file_name': 'fake_name.jpg'}
    annotation_1 = {'id': 1, 'image_id': 0, 'category_id': 0, 'area': 400, 'bbox': [50, 60, 20, 20], 'iscrowd': 0}
    annotation_2 = {'id': 2, 'image_id': 0, 'category_id': 0, 'area': 900, 'bbox': [100, 120, 30, 30], 'iscrowd': 0}
    annotation_3 = {'id': 3, 'image_id': 0, 'category_id': 0, 'area': 1600, 'bbox': [150, 160, 40, 40], 'iscrowd': 0}
    annotation_4 = {'id': 4, 'image_id': 0, 'category_id': 0, 'area': 10000, 'bbox': [250, 260, 100, 100], 'iscrowd': 0}
    categories = [{'id': 0, 'name': 'car', 'supercategory': 'car'}]
    fake_json = {'images': [image], 'annotations': [annotation_1, annotation_2, annotation_3, annotation_4], 'categories': categories}
    mmcv.dump(fake_json, json_name)

def _create_dummy_results():
    boxes = [np.array([[50, 60, 70, 80, 1.0], [100, 120, 130, 150, 0.98], [150, 160, 190, 200, 0.96], [250, 260, 350, 360, 0.95]])]
    return [boxes]

def _create_dummy_custom_pkl(pkl_name):
    fake_pkl = [{'filename': 'fake_name.jpg', 'width': 640, 'height': 640, 'ann': {'bboxes': np.array([[50, 60, 70, 80], [100, 120, 130, 150], [150, 160, 190, 200], [250, 260, 350, 360]]), 'labels': np.array([0, 0, 0, 0])}}]
    mmcv.dump(fake_pkl, pkl_name)

@patch('mmdet.datasets.CocoDataset.load_annotations', MagicMock())
@patch('mmdet.datasets.CustomDataset.load_annotations', MagicMock())
@patch('mmdet.datasets.XMLDataset.load_annotations', MagicMock())
@patch('mmdet.datasets.CityscapesDataset.load_annotations', MagicMock())
@patch('mmdet.datasets.CocoDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.CustomDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.XMLDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.CityscapesDataset._filter_imgs', MagicMock)
@pytest.mark.parametrize('dataset', ['CocoDataset', 'VOCDataset', 'CityscapesDataset'])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)
    if dataset in ['CocoDataset', 'CityscapesDataset']:
        dataset_class.coco = MagicMock()
        dataset_class.cat_ids = MagicMock()
    original_classes = dataset_class.CLASSES
    custom_dataset = dataset_class(ann_file=MagicMock(), pipeline=[], classes=('bus', 'car'), test_mode=True, img_prefix='VOC2007' if dataset == 'VOCDataset' else '')
    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ('bus', 'car')
    print(custom_dataset)
    custom_dataset = dataset_class(ann_file=MagicMock(), pipeline=[], classes=['bus', 'car'], test_mode=True, img_prefix='VOC2007' if dataset == 'VOCDataset' else '')
    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
    print(custom_dataset)
    custom_dataset = dataset_class(ann_file=MagicMock(), pipeline=[], classes=['foo'], test_mode=True, img_prefix='VOC2007' if dataset == 'VOCDataset' else '')
    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['foo']
    print(custom_dataset)
    custom_dataset = dataset_class(ann_file=MagicMock(), pipeline=[], classes=None, test_mode=True, img_prefix='VOC2007' if dataset == 'VOCDataset' else '')
    assert custom_dataset.CLASSES == original_classes
    print(custom_dataset)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + 'classes.txt'
        with open(path, 'w') as f:
            f.write('bus\ncar\n')
    custom_dataset = dataset_class(ann_file=MagicMock(), pipeline=[], classes=path, test_mode=True, img_prefix='VOC2007' if dataset == 'VOCDataset' else '')
    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
    print(custom_dataset)

def _create_ids_error_oid_csv(label_file, fake_csv_file):
    label_description = ['/m/000002', 'Football']
    with open(label_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(label_description)
    header = ['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
    annotations = [['color', 'xclick', '/m/000002', '1', '0.022673031', '0.9642005', '0.07103825', '0.80054647', '0', '0', '0', '0', '0'], ['000595fe6fee6369', 'xclick', '/m/000000', '1', '0', '1', '0', '1', '0', '0', '1', '0', '0']]
    with open(fake_csv_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(annotations)

def _create_oid_style_ann(label_file, csv_file, label_level_file):
    label_description = [['/m/000000', 'Sports equipment'], ['/m/000001', 'Ball'], ['/m/000002', 'Football'], ['/m/000004', 'Bicycle']]
    with open(label_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(label_description)
    header = ['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
    annotations = [['color', 'xclick', '/m/000002', 1, 0.0333333, 0.1, 0.0333333, 0.1, 0, 0, 1, 0, 0], ['color', 'xclick', '/m/000002', 1, 0.1, 0.166667, 0.1, 0.166667, 0, 0, 0, 0, 0]]
    with open(csv_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(annotations)
    header = ['ImageID', 'Source', 'LabelName', 'Confidence']
    annotations = [['color', 'xclick', '/m/000002', '1'], ['color', 'xclick', '/m/000004', '0']]
    with open(label_level_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(annotations)

def _create_hierarchy_np(hierarchy_name):
    fake_hierarchy = np.array([[0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 0, 0, 1], [0, 0, 0, 0, 0]])
    with open(hierarchy_name, 'wb') as f:
        np.save(f, fake_hierarchy)

def _creat_oid_challenge_style_ann(txt_file, label_file, label_level_file):
    bboxes = ['validation/color.jpg\n', '4 29\n', '2\n', '1 0.0333333 0.1 0.0333333 0.1 1\n', '1 0.1 0.166667 0.1 0.166667 0\n']
    with open(txt_file, 'w', newline='') as f:
        f.writelines(bboxes)
        f.close()
    label_description = [['/m/000000', 'Sports equipment', 1], ['/m/000001', 'Ball', 2], ['/m/000002', 'Football', 3], ['/m/000004', 'Bicycle', 4]]
    with open(label_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(label_description)
    header = ['ImageID', 'LabelName', 'Confidence']
    annotations = [['color', '/m/000001', '1'], ['color', '/m/000000', '0']]
    with open(label_level_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(annotations)

def test_oid_annotation_ids_unique():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_label_file = osp.join(tmp_dir.name, 'fake_label.csv')
    fake_ann_file = osp.join(tmp_dir.name, 'fake_ann.csv')
    _create_ids_error_oid_csv(fake_label_file, fake_ann_file)
    with pytest.raises(AssertionError):
        OpenImagesDataset(ann_file=fake_ann_file, label_file=fake_label_file, pipeline=[])
    tmp_dir.cleanup()

def test_openimages_dataset():
    tmp_dir = tempfile.TemporaryDirectory()
    label_file = osp.join(tmp_dir.name, 'label_file.csv')
    ann_file = osp.join(tmp_dir.name, 'ann_file.csv')
    label_level_file = osp.join(tmp_dir.name, 'label_level_file.csv')
    _create_oid_style_ann(label_file, ann_file, label_level_file)
    hierarchy_json = osp.join(tmp_dir.name, 'hierarchy.json')
    _create_hierarchy_json(hierarchy_json)
    with pytest.raises(AssertionError):
        OpenImagesDataset(ann_file=ann_file, label_file=label_file, image_level_ann_file=label_level_file, pipeline=[])
    dataset = OpenImagesDataset(ann_file=ann_file, label_file=label_file, image_level_ann_file=label_level_file, hierarchy_file=hierarchy_json, pipeline=[])
    ann = dataset.get_ann_info(0)
    assert ann['bboxes'].shape[0] == ann['labels'].shape[0] == ann['gt_is_group_ofs'].shape[0] == 2
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    test_pipeline = [dict(type='LoadImageFromFile'), dict(type='MultiScaleFlipAug', img_scale=(128, 128), flip=False, transforms=[dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'), dict(type='Normalize', **img_norm_cfg), dict(type='Pad', size_divisor=32), dict(type='ImageToTensor', keys=['img']), dict(type='Collect', keys=['img'])])]
    dataset = OpenImagesDataset(ann_file=ann_file, img_prefix='tests/data', label_file=label_file, image_level_ann_file=label_level_file, load_from_file=False, hierarchy_file=hierarchy_json, pipeline=test_pipeline)
    dataset.prepare_test_img(0)
    assert len(dataset.test_img_metas) == 1
    result = _create_dummy_results()
    dataset.evaluate(result)
    hierarchy_json = osp.join(tmp_dir.name, 'hierarchy.json')
    _create_hierarchy_json(hierarchy_json)
    with pytest.raises(AssertionError):
        fake_path = osp.join(tmp_dir.name, 'hierarchy.csv')
        OpenImagesDataset(ann_file=ann_file, img_prefix='tests/data', label_file=label_file, image_level_ann_file=label_level_file, load_from_file=False, hierarchy_file=fake_path, pipeline=test_pipeline)
    hierarchy = dataset.get_relation_matrix(hierarchy_json)
    hierarchy_gt = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1]])
    assert np.equal(hierarchy, hierarchy_gt).all()
    meta_file = osp.join(tmp_dir.name, 'meta.pkl')
    _create_metas(meta_file)
    dataset = OpenImagesDataset(ann_file=ann_file, label_file=label_file, image_level_ann_file=label_level_file, hierarchy_file=hierarchy_json, meta_file=meta_file, pipeline=[])
    result = _create_dummy_results()
    parsed_results = dataset.evaluate(result)
    assert np.isclose(parsed_results['mAP'], 0.8333, 0.0001)
    dataset = OpenImagesDataset(ann_file=ann_file, label_file=label_file, load_image_level_labels=False, image_level_ann_file=label_level_file, hierarchy_file=hierarchy_json, meta_file=meta_file, pipeline=[])
    result = _create_dummy_results()
    parsed_results = dataset.evaluate(result)
    assert np.isclose(parsed_results['mAP'], 0.8333, 0.0001)
    tmp_dir.cleanup()

def _create_hierarchy_json(hierarchy_name):
    fake_hierarchy = {'LabelName': '/m/0bl9f', 'Subcategory': [{'LabelName': '/m/000000', 'Subcategory': [{'LabelName': '/m/000001', 'Subcategory': [{'LabelName': '/m/000002'}]}, {'LabelName': '/m/000004'}]}]}
    mmcv.dump(fake_hierarchy, hierarchy_name)

def _create_metas(meta_file):
    fake_meta = [{'filename': 'data/OpenImages/OpenImages/validation/color.jpg', 'ori_shape': (300, 300, 3)}]
    mmcv.dump(fake_meta, meta_file)

def test_openimages_challenge_dataset():
    tmp_dir = tempfile.TemporaryDirectory()
    ann_file = osp.join(tmp_dir.name, 'ann_file.txt')
    label_file = osp.join(tmp_dir.name, 'label_file.csv')
    label_level_file = osp.join(tmp_dir.name, 'label_level_file.csv')
    _creat_oid_challenge_style_ann(ann_file, label_file, label_level_file)
    dataset = OpenImagesChallengeDataset(ann_file=ann_file, label_file=label_file, load_image_level_labels=False, get_supercategory=False, pipeline=[])
    ann = dataset.get_ann_info(0)
    assert ann['bboxes'].shape[0] == ann['labels'].shape[0] == ann['gt_is_group_ofs'].shape[0] == 2
    dataset.prepare_train_img(0)
    dataset.prepare_test_img(0)
    meta_file = osp.join(tmp_dir.name, 'meta.pkl')
    _create_metas(meta_file)
    result = _create_dummy_results()
    with pytest.raises(AssertionError):
        fake_json = osp.join(tmp_dir.name, 'hierarchy.json')
        OpenImagesChallengeDataset(ann_file=ann_file, label_file=label_file, image_level_ann_file=label_level_file, hierarchy_file=fake_json, meta_file=meta_file, pipeline=[])
    hierarchy_file = osp.join(tmp_dir.name, 'hierarchy.np')
    _create_hierarchy_np(hierarchy_file)
    dataset = OpenImagesChallengeDataset(ann_file=ann_file, label_file=label_file, image_level_ann_file=label_level_file, hierarchy_file=hierarchy_file, meta_file=meta_file, pipeline=[])
    dataset.evaluate(result)
    tmp_dir.cleanup()

def create_tracker(tracker_type, tracker_config, reid_weights, device, half):
    cfg = get_config()
    cfg.merge_from_file(tracker_config)
    if tracker_type == 'strongsort':
        from trackers.strongsort.strong_sort import StrongSORT
        strongsort = StrongSORT(reid_weights, device, half, max_dist=cfg.strongsort.max_dist, max_iou_dist=cfg.strongsort.max_iou_dist, max_age=cfg.strongsort.max_age, max_unmatched_preds=cfg.strongsort.max_unmatched_preds, n_init=cfg.strongsort.n_init, nn_budget=cfg.strongsort.nn_budget, mc_lambda=cfg.strongsort.mc_lambda, ema_alpha=cfg.strongsort.ema_alpha)
        return strongsort
    elif tracker_type == 'ocsort':
        from trackers.ocsort.ocsort import OCSort
        ocsort = OCSort(det_thresh=cfg.ocsort.det_thresh, max_age=cfg.ocsort.max_age, min_hits=cfg.ocsort.min_hits, iou_threshold=cfg.ocsort.iou_thresh, delta_t=cfg.ocsort.delta_t, asso_func=cfg.ocsort.asso_func, inertia=cfg.ocsort.inertia, use_byte=cfg.ocsort.use_byte)
        return ocsort
    elif tracker_type == 'bytetrack':
        from trackers.bytetrack.byte_tracker import BYTETracker
        bytetracker = BYTETracker(track_thresh=cfg.bytetrack.track_thresh, match_thresh=cfg.bytetrack.match_thresh, track_buffer=cfg.bytetrack.track_buffer, frame_rate=cfg.bytetrack.frame_rate)
        return bytetracker
    elif tracker_type == 'botsort':
        from trackers.botsort.bot_sort import BoTSORT
        botsort = BoTSORT(reid_weights, device, half, track_high_thresh=cfg.botsort.track_high_thresh, new_track_thresh=cfg.botsort.new_track_thresh, track_buffer=cfg.botsort.track_buffer, match_thresh=cfg.botsort.match_thresh, proximity_thresh=cfg.botsort.proximity_thresh, appearance_thresh=cfg.botsort.appearance_thresh, cmc_method=cfg.botsort.cmc_method, frame_rate=cfg.botsort.frame_rate, lambda_=cfg.botsort.lambda_)
        return botsort
    elif tracker_type == 'deepocsort':
        from trackers.deepocsort.ocsort import OCSort
        botsort = OCSort(reid_weights, device, half, det_thresh=cfg.deepocsort.det_thresh, max_age=cfg.deepocsort.max_age, min_hits=cfg.deepocsort.min_hits, iou_threshold=cfg.deepocsort.iou_thresh, delta_t=cfg.deepocsort.delta_t, asso_func=cfg.deepocsort.asso_func, inertia=cfg.deepocsort.inertia)
        return botsort
    else:
        print('No such tracker')
        exit()

class ReIDDetectMultiBackend(nn.Module):

    def __init__(self, weights='osnet_x0_25_msmt17.pt', device=torch.device('cpu'), fp16=False):
        super().__init__()
        w = weights[0] if isinstance(weights, list) else weights
        self.pt, self.jit, self.onnx, self.xml, self.engine, self.tflite = self.model_type(w)
        self.fp16 = fp16
        self.fp16 &= self.pt or self.jit or self.engine
        self.device = device
        self.image_size = (256, 128)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.transforms = []
        self.transforms += [T.Resize(self.image_size)]
        self.transforms += [T.ToTensor()]
        self.transforms += [T.Normalize(mean=self.pixel_mean, std=self.pixel_std)]
        self.preprocess = T.Compose(self.transforms)
        self.to_pil = T.ToPILImage()
        model_name = get_model_name(w)
        if w.suffix == '.pt':
            model_url = get_model_url(w)
            if not file_exists(w) and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif file_exists(w):
                pass
            else:
                print(f'No URL associated to the chosen StrongSORT weights ({w}). Choose between:')
                show_downloadeable_models()
                exit()
        self.model = build_model(model_name, num_classes=1, pretrained=not (w and w.is_file()), use_gpu=device)
        if self.pt:
            if w and w.is_file() and (w.suffix == '.pt'):
                load_pretrained_weights(self.model, w)
            self.model.to(device).eval()
            self.model.half() if self.fp16 else self.model.float()
        elif self.jit:
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            self.model = torch.jit.load(w)
            self.model.half() if self.fp16 else self.model.float()
        elif self.onnx:
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available() and device.type != 'cpu'
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(str(w), providers=providers)
        elif self.engine:
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt
            check_version(trt.__version__, '7.0.0', hard=True)
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                self.model_ = runtime.deserialize_cuda_engine(f.read())
            self.context = self.model_.create_execution_context()
            self.bindings = OrderedDict()
            self.fp16 = False
            dynamic = False
            for index in range(self.model_.num_bindings):
                name = self.model_.get_binding_name(index)
                dtype = trt.nptype(self.model_.get_binding_dtype(index))
                if self.model_.binding_is_input(index):
                    if -1 in tuple(self.model_.get_binding_shape(index)):
                        dynamic = True
                        self.context.set_binding_shape(index, tuple(self.model_.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        self.fp16 = True
                shape = tuple(self.context.get_binding_shape(index))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict(((n, d.ptr) for n, d in self.bindings.items()))
            batch_size = self.bindings['images'].shape[0]
        elif self.xml:
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino',))
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():
                w = next(Path(w).glob('*.xml'))
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout('NCWH'))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            self.executable_network = ie.compile_model(network, device_name='CPU')
            self.output_layer = next(iter(self.executable_network.outputs))
        elif self.tflite:
            LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = (tf.lite.Interpreter, tf.lite.experimental.load_delegate)
            self.interpreter = tf.lite.Interpreter(model_path=w)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            input_data = np.array(np.random.random_sample((1, 256, 128, 3)), dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            print('This model framework is not supported yet!')
            exit()

    @staticmethod
    def model_type(p='path/to/model.pt'):
        from trackers.reid_export import export_formats
        sf = list(export_formats().Suffix)
        check_suffix(p, sf)
        types = [s in Path(p).name for s in sf]
        return types

    def _preprocess(self, im_batch):
        images = []
        for element in im_batch:
            image = self.to_pil(element)
            image = self.preprocess(image)
            images.append(image)
        images = torch.stack(images, dim=0)
        images = images.to(self.device)
        return images

    def forward(self, im_batch):
        im_batch = self._preprocess(im_batch)
        if self.fp16 and im_batch.dtype != torch.float16:
            im_batch = im_batch.half()
        features = []
        if self.pt:
            features = self.model(im_batch)
        elif self.jit:
            features = self.model(im_batch)
        elif self.onnx:
            im_batch = im_batch.cpu().numpy()
            features = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im_batch})[0]
        elif self.engine:
            if True and im_batch.shape != self.bindings['images'].shape:
                i_in, i_out = (self.model_.get_binding_index(x) for x in ('images', 'output'))
                self.context.set_binding_shape(i_in, im_batch.shape)
                self.bindings['images'] = self.bindings['images']._replace(shape=im_batch.shape)
                self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings['images'].shape
            assert im_batch.shape == s, f'input size {im_batch.shape} {('>' if self.dynamic else 'not equal to')} max model size {s}'
            self.binding_addrs['images'] = int(im_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings['output'].data
        elif self.xml:
            im_batch = im_batch.cpu().numpy()
            features = self.executable_network([im_batch])[self.output_layer]
        else:
            print('Framework not supported at the moment, we are working on it...')
            exit()
        if isinstance(features, (list, tuple)):
            return self.from_numpy(features[0]) if len(features) == 1 else [self.from_numpy(x) for x in features]
        else:
            return self.from_numpy(features)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=[(256, 128, 3)]):
        warmup_types = (self.pt, self.jit, self.onnx, self.engine, self.tflite)
        if any(warmup_types) and self.device.type != 'cpu':
            im = [np.empty(*imgsz).astype(np.uint8)]
            for _ in range(2 if self.jit else 1):
                self.forward(im)

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = (x1 + w, y1 + h)
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)

def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())
                if is_gt:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])
                results_dict[fid].append((tlwh, target_id, score))
    return results_dict

class BboxToJsonLogger(BaseJsonLogger):
    """
    ُ This module is designed to automate the task of logging jsons. An example json is used
    to show the contents of json file shortly
    Example:
          {
          "video_details": {
            "frame_width": 1920,
            "frame_height": 1080,
            "frame_rate": 20,
            "video_name": "/home/gpu/codes/MSD/pedestrian_2/project/public/camera1.avi"
          },
          "frames": [
            {
              "frame_id": 329,
              "timestamp": 3365.1254
              "bboxes": [
                {
                  "labels": [
                    {
                      "category": "pedestrian",
                      "confidence": 0.9
                    }
                  ],
                  "bbox_id": 0,
                  "top": 1257,
                  "left": 138,
                  "width": 68,
                  "height": 109
                }
              ]
            }],

    Attributes:
        frames (dict): It's a dictionary that maps each frame_id to json attributes.
        video_details (dict): information about video file.
        top_k_labels (int): shows the allowed number of labels
        start_time (datetime object): we use it to automate the json output by time.

    Args:
        top_k_labels (int): shows the allowed number of labels

    """

    def __init__(self, top_k_labels: int=1):
        self.frames = {}
        self.video_details = self.video_details = dict(frame_width=None, frame_height=None, frame_rate=None, video_name=None)
        self.top_k_labels = top_k_labels
        self.start_time = datetime.now()

    def set_top_k(self, value):
        self.top_k_labels = value

    def frame_exists(self, frame_id: int) -> bool:
        """
        Args:
            frame_id (int):

        Returns:
            bool: true if frame_id is recognized
        """
        return frame_id in self.frames.keys()

    def add_frame(self, frame_id: int, timestamp: float=None) -> None:
        """
        Args:
            frame_id (int):
            timestamp (float): opencv captured frame time property

        Raises:
             ValueError: if frame_id would not exist in class frames attribute

        Returns:
            None

        """
        if not self.frame_exists(frame_id):
            self.frames[frame_id] = Frame(frame_id, timestamp)
        else:
            raise ValueError('Frame id: {} already exists'.format(frame_id))

    def bbox_exists(self, frame_id: int, bbox_id: int) -> bool:
        """
        Args:
            frame_id:
            bbox_id:

        Returns:
            bool: if bbox exists in frame bboxes list
        """
        bboxes = []
        if self.frame_exists(frame_id=frame_id):
            bboxes = [bbox.bbox_id for bbox in self.frames[frame_id].bboxes]
        return bbox_id in bboxes

    def find_bbox(self, frame_id: int, bbox_id: int):
        """

        Args:
            frame_id:
            bbox_id:

        Returns:
            bbox_id (int):

        Raises:
            ValueError: if bbox_id does not exist in the bbox list of specific frame.
        """
        if not self.bbox_exists(frame_id, bbox_id):
            raise ValueError('frame with id: {} does not contain bbox with id: {}'.format(frame_id, bbox_id))
        bboxes = {bbox.bbox_id: bbox for bbox in self.frames[frame_id].bboxes}
        return bboxes.get(bbox_id)

    def add_bbox_to_frame(self, frame_id: int, bbox_id: int, top: int, left: int, width: int, height: int) -> None:
        """

        Args:
            frame_id (int):
            bbox_id (int):
            top (int):
            left (int):
            width (int):
            height (int):

        Returns:
            None

        Raises:
            ValueError: if bbox_id already exist in frame information with frame_id
            ValueError: if frame_id does not exist in frames attribute
        """
        if self.frame_exists(frame_id):
            frame = self.frames[frame_id]
            if not self.bbox_exists(frame_id, bbox_id):
                frame.add_bbox(bbox_id, top, left, width, height)
            else:
                raise ValueError('frame with frame_id: {} already contains the bbox with id: {} '.format(frame_id, bbox_id))
        else:
            raise ValueError('frame with frame_id: {} does not exist'.format(frame_id))

    def add_label_to_bbox(self, frame_id: int, bbox_id: int, category: str, confidence: float):
        """
        Args:
            frame_id:
            bbox_id:
            category:
            confidence: the confidence value returned from yolo detection

        Returns:
            None

        Raises:
            ValueError: if labels quota (top_k_labels) exceeds.
        """
        bbox = self.find_bbox(frame_id, bbox_id)
        if not bbox.labels_full(self.top_k_labels):
            bbox.add_label(category, confidence)
        else:
            raise ValueError('labels in frame_id: {}, bbox_id: {} is fulled'.format(frame_id, bbox_id))

    def add_video_details(self, frame_width: int=None, frame_height: int=None, frame_rate: int=None, video_name: str=None):
        self.video_details['frame_width'] = frame_width
        self.video_details['frame_height'] = frame_height
        self.video_details['frame_rate'] = frame_rate
        self.video_details['video_name'] = video_name

    def output(self):
        output = {'video_details': self.video_details}
        result = list(self.frames.values())
        output['frames'] = [item.dic() for item in result]
        return output

    def json_output(self, output_name):
        """
        Args:
            output_name:

        Returns:
            None

        Notes:
            It creates the json output with `output_name` name.
        """
        if not output_name.endswith('.json'):
            output_name += '.json'
        with open(output_name, 'w') as file:
            json.dump(self.output(), file)
        file.close()

    def set_start(self):
        self.start_time = datetime.now()

    def schedule_output_by_time(self, output_dir=JsonMeta.PATH_TO_SAVE, hours: int=0, minutes: int=0, seconds: int=60) -> None:
        """
        Notes:
            Creates folder and then periodically stores the jsons on that address.

        Args:
            output_dir (str): the directory where output files will be stored
            hours (int):
            minutes (int):
            seconds (int):

        Returns:
            None

        """
        end = datetime.now()
        interval = 0
        interval += abs(min([hours, JsonMeta.HOURS]) * 3600)
        interval += abs(min([minutes, JsonMeta.MINUTES]) * 60)
        interval += abs(min([seconds, JsonMeta.SECONDS]))
        diff = (end - self.start_time).seconds
        if diff > interval:
            output_name = self.start_time.strftime('%Y-%m-%d %H-%M-%S') + '.json'
            if not exists(output_dir):
                makedirs(output_dir)
            output = join(output_dir, output_name)
            self.json_output(output_name=output)
            self.frames = {}
            self.start_time = datetime.now()

    def schedule_output_by_frames(self, frames_quota, frame_counter, output_dir=JsonMeta.PATH_TO_SAVE):
        """
        saves as the number of frames quota increases higher.
        :param frames_quota:
        :param frame_counter:
        :param output_dir:
        :return:
        """
        pass

    def flush(self, output_dir):
        """
        Notes:
            We use this function to output jsons whenever possible.
            like the time that we exit the while loop of opencv.

        Args:
            output_dir:

        Returns:
            None

        """
        filename = self.start_time.strftime('%Y-%m-%d %H-%M-%S') + '-remaining.json'
        output = join(output_dir, filename)
        self.json_output(output_name=output)

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}
        if config_file is not None:
            assert os.path.isfile(config_file)
            with open(config_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)
        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)

class CMCComputer:

    def __init__(self, minimum_features=10, method='sparse'):
        assert method in ['file', 'sparse', 'sift']
        os.makedirs('./cache', exist_ok=True)
        self.cache_path = './cache/affine_ocsort.pkl'
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as fp:
                self.cache = pickle.load(fp)
        self.minimum_features = minimum_features
        self.prev_img = None
        self.prev_desc = None
        self.sparse_flow_param = dict(maxCorners=3000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)
        self.file_computed = {}
        self.comp_function = None
        if method == 'sparse':
            self.comp_function = self._affine_sparse_flow
        elif method == 'sift':
            self.comp_function = self._affine_sift
        elif method == 'file':
            self.comp_function = self._affine_file
            self.file_affines = {}
            self.file_names = {}
            for f_name in os.listdir('./cache/cmc_files/MOT17_ablation/'):
                tag = f_name.replace('GMC-', '').replace('.txt', '') + '-FRCNN'
                f_name = os.path.join('./cache/cmc_files/MOT17_ablation/', f_name)
                self.file_names[tag] = f_name
            for f_name in os.listdir('./cache/cmc_files/MOT20_ablation/'):
                tag = f_name.replace('GMC-', '').replace('.txt', '')
                f_name = os.path.join('./cache/cmc_files/MOT20_ablation/', f_name)
                self.file_names[tag] = f_name
            for f_name in os.listdir('./cache/cmc_files/MOTChallenge/'):
                tag = f_name.replace('GMC-', '').replace('.txt', '')
                if 'MOT17' in tag:
                    tag = tag + '-FRCNN'
                if tag in self.file_names:
                    continue
                f_name = os.path.join('./cache/cmc_files/MOTChallenge/', f_name)
                self.file_names[tag] = f_name

    def compute_affine(self, img, bbox, tag):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if tag in self.cache:
            A = self.cache[tag]
            return A
        mask = np.ones_like(img, dtype=np.uint8)
        if bbox.shape[0] > 0:
            bbox = np.round(bbox).astype(np.int32)
            bbox[bbox < 0] = 0
            for bb in bbox:
                mask[bb[1]:bb[3], bb[0]:bb[2]] = 0
        A = self.comp_function(img, mask, tag)
        self.cache[tag] = A
        return A

    def _load_file(self, name):
        affines = []
        with open(self.file_names[name], 'r') as fp:
            for line in fp:
                tokens = [float(f) for f in line.split('\t')[1:7]]
                A = np.eye(2, 3)
                A[0, 0] = tokens[0]
                A[0, 1] = tokens[1]
                A[0, 2] = tokens[2]
                A[1, 0] = tokens[3]
                A[1, 1] = tokens[4]
                A[1, 2] = tokens[5]
                affines.append(A)
        self.file_affines[name] = affines

    def _affine_file(self, frame, mask, tag):
        name, num = tag.split(':')
        if name not in self.file_affines:
            self._load_file(name)
        if name not in self.file_affines:
            raise RuntimeError('Error loading file affines for CMC.')
        return self.file_affines[name][int(num) - 1]

    def _affine_sift(self, frame, mask, tag):
        A = np.eye(2, 3)
        detector = cv2.SIFT_create()
        kp, desc = detector.detectAndCompute(frame, mask)
        if self.prev_desc is None:
            self.prev_desc = [kp, desc]
            return A
        if desc.shape[0] < self.minimum_features or self.prev_desc[1].shape[0] < self.minimum_features:
            return A
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(self.prev_desc[1], desc, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > self.minimum_features:
            src_pts = np.float32([self.prev_desc[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            A, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        else:
            print('Warning: not enough matching points')
        if A is None:
            A = np.eye(2, 3)
        self.prev_desc = [kp, desc]
        return A

    def _affine_sparse_flow(self, frame, mask, tag):
        A = np.eye(2, 3)
        keypoints = cv2.goodFeaturesToTrack(frame, mask=mask, **self.sparse_flow_param)
        if self.prev_img is None:
            self.prev_img = frame
            self.prev_desc = keypoints
            return A
        matched_kp, status, err = cv2.calcOpticalFlowPyrLK(self.prev_img, frame, self.prev_desc, None)
        matched_kp = matched_kp.reshape(-1, 2)
        status = status.reshape(-1)
        prev_points = self.prev_desc.reshape(-1, 2)
        prev_points = prev_points[status]
        curr_points = matched_kp[status]
        if prev_points.shape[0] > self.minimum_features:
            A, _ = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC)
        else:
            print('Warning: not enough matching points')
        if A is None:
            A = np.eye(2, 3)
        self.prev_img = frame
        self.prev_desc = keypoints
        return A

    def dump_cache(self):
        with open(self.cache_path, 'wb') as fp:
            pickle.dump(self.cache, fp)

class EmbeddingComputer:

    def __init__(self, dataset):
        self.model = None
        self.dataset = dataset
        self.crop_size = (128, 384)
        os.makedirs('./cache/embeddings/', exist_ok=True)
        self.cache_path = './cache/embeddings/{}_embedding.pkl'
        self.cache = {}
        self.cache_name = ''

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                self.cache = pickle.load(fp)

    def compute_embedding(self, img, bbox, tag, is_numpy=True):
        if self.cache_name != tag.split(':')[0]:
            self.load_cache(tag.split(':')[0])
        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError("ERROR: The number of cached embeddings don't match the number of detections.\nWas the detector model changed? Delete cache if so.")
            return embs
        if self.model is None:
            self.initialize_model()
        if is_numpy:
            h, w = img.shape[:2]
        else:
            h, w = img.shape[2:]
        results = np.round(bbox).astype(np.int32)
        results[:, 0] = results[:, 0].clip(0, w)
        results[:, 1] = results[:, 1].clip(0, h)
        results[:, 2] = results[:, 2].clip(0, w)
        results[:, 3] = results[:, 3].clip(0, h)
        crops = []
        for p in results:
            if is_numpy:
                crop = img[p[1]:p[3], p[0]:p[2]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
                crop = torch.as_tensor(crop.astype('float32').transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
            else:
                crop = img[:, :, p[1]:p[3], p[0]:p[2]]
                crop = torchvision.transforms.functional.resize(crop, self.crop_size)
            crops.append(crop)
        crops = torch.cat(crops, dim=0)
        with torch.no_grad():
            crops = crops.cuda()
            crops = crops.half()
            embs = self.model(crops)
        embs = torch.nn.functional.normalize(embs)
        embs = embs.cpu().numpy()
        self.cache[tag] = embs
        return embs

    def initialize_model(self):
        """
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)
        sd = torch.load("external/weights/osnet_ain_ms_d_c.pth.tar")["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        """
        if self.dataset == 'mot17':
            path = 'external/weights/mot17_sbs_S50.pth'
        elif self.dataset == 'mot20':
            path = 'external/weights/mot20_sbs_S50.pth'
        elif self.dataset == 'dance':
            path = None
        else:
            raise RuntimeError('Need the path for a new ReID model.')
        model = FastReID(path)
        model.eval()
        model.cuda()
        model.half()
        self.model = model

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), 'wb') as fp:
                pickle.dump(self.cache, fp)

class GMC:

    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        super(GMC, self).__init__()
        self.method = method
        self.downscale = max(1, int(downscale))
        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-06
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)
        elif self.method == 'file' or self.method == 'files':
            seqName = verbose[0]
            ablation = verbose[1]
            if ablation:
                filePath = 'tracker/GMC_files/MOT17_ablation'
            else:
                filePath = 'tracker/GMC_files/MOTChallenge'
            if '-FRCNN' in seqName:
                seqName = seqName[:-6]
            elif '-DPM' in seqName:
                seqName = seqName[:-4]
            elif '-SDP' in seqName:
                seqName = seqName[:-4]
            self.gmcFile = open(filePath + '/GMC-' + seqName + '.txt', 'r')
            if self.gmcFile is None:
                raise ValueError('Error: Unable to open GMC file in directory:' + filePath)
        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'
        else:
            raise ValueError('Error: Unknown CMC method:' + method)
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame, detections=None):
        if self.method == 'orb' or self.method == 'sift':
            return self.applyFeaures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)
        elif self.method == 'file':
            return self.applyFile(raw_frame, detections)
        elif self.method == 'none':
            return np.eye(2, 3)
        else:
            return np.eye(2, 3)

    def applyEcc(self, raw_frame, detections=None):
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H
        try:
            cc, H = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except:
            print('Warning: find transform failed. Set warp as identity')
        return H

    def applyFeaures(self, raw_frame, detections=None):
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale
        mask = np.zeros_like(frame)
        mask[int(0.02 * height):int(0.98 * height), int(0.02 * width):int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0
        keypoints = self.detector.detect(frame, mask)
        keypoints, descriptors = self.extractor.compute(frame, keypoints)
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            self.initializedFirstFrame = True
            return H
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)
        matches = []
        spatialDistances = []
        maxSpatialDistance = 0.25 * np.array([width, height])
        if len(knnMatches) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H
        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt
                spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0], prevKeyPointLocation[1] - currKeyPointLocation[1])
                if np.abs(spatialDistance[0]) < maxSpatialDistance[0] and np.abs(spatialDistance[1]) < maxSpatialDistance[1]:
                    spatialDistances.append(spatialDistance)
                    matches.append(m)
        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)
        inliesrs = spatialDistances - meanSpatialDistances < 2.5 * stdSpatialDistances
        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)
        if 0:
            matches_img = np.hstack((self.prevFrame, frame))
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            W = np.size(self.prevFrame, 1)
            for m in goodMatches:
                prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))
                matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
                matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)
            plt.figure()
            plt.imshow(matches_img)
            plt.show()
        if np.size(prevPoints, 0) > 4 and np.size(prevPoints, 0) == np.size(prevPoints, 0):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)
        return H

    def applySparseOptFlow(self, raw_frame, detections=None):
        t0 = time.time()
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)
        prevPoints = []
        currPoints = []
        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)
        if np.size(prevPoints, 0) > 4 and np.size(prevPoints, 0) == np.size(prevPoints, 0):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        t1 = time.time()
        return H

    def applyFile(self, raw_frame, detections=None):
        line = self.gmcFile.readline()
        tokens = line.split('\t')
        H = np.eye(2, 3, dtype=np.float_)
        H[0, 0] = float(tokens[1])
        H[0, 1] = float(tokens[2])
        H[0, 2] = float(tokens[3])
        H[1, 0] = float(tokens[4])
        H[1, 1] = float(tokens[5])
        H[1, 2] = float(tokens[6])
        return H

class HiddenPrints:
    """
    A context manager to suppress print statements.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def evaluate_coco(gt_file: str, pred_file: str, task: str='bbox', evaluation_type: str='full') -> None:
    """
    Evaluates the performance of a COCO object detection model.

    Args:
        gt_file (str): Path to the ground truth file.
        pred_file (str): Path to the prediction file.
        task (str, optional): The type of task to evaluate (bbox or segm). Defaults to "bbox".
        evaluation_type (str, optional): The type of evaluation to perform (full or mAP). Defaults to "full".
    """
    with HiddenPrints():
        coco_gt = COCO(gt_file)
        with open(pred_file, 'r') as f:
            pred_file = json.load(f)
            pred_file = pred_file[0]['annotations']
        coco_dt = coco_gt.loadRes(pred_file)
        coco_eval = COCOeval(coco_gt, coco_dt, task)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    if evaluation_type == 'full':
        coco_eval.summarize()
    elif evaluation_type == 'mAP':
        print(f'{task} mAP: {coco_eval.stats[0]:.3f}')

