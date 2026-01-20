# Cluster 9

def _convert_to_example(image_data, shape, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id):
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={'image/height': int64_feature(shape[0]), 'image/width': int64_feature(shape[1]), 'image/channels': int64_feature(shape[2]), 'image/classid': int64_feature(class_id), 'image/keypoint/x': int64_feature(keypoint_x), 'image/keypoint/y': int64_feature(keypoint_y), 'image/keypoint/v': int64_feature(keypoint_v), 'image/keypoint/id': int64_feature(keypoint_id), 'image/keypoint/gid': int64_feature(keypoint_global_id), 'image/format': bytes_feature(image_format), 'image/filename': bytes_feature(image_file.encode('utf8')), 'image/encoded': bytes_feature(image_data)}))
    return example

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _add_to_tfrecord(tfrecord_writer, image_path, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id):
    image_data, shape = _process_image(image_path)
    example = _convert_to_example(image_data, shape, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id)
    tfrecord_writer.write(example.SerializeToString())

def _process_image(filename):
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    return (image_data, misc.imread(filename).shape)

def _test_add_to_tfrecord(tfrecord_writer, image_path, image_file, class_id):
    image_data, shape = _process_image(image_path)
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={'image/height': int64_feature(shape[0]), 'image/width': int64_feature(shape[1]), 'image/channels': int64_feature(shape[2]), 'image/classid': int64_feature(class_id), 'image/format': bytes_feature(image_format), 'image/filename': bytes_feature(image_file.encode('utf8')), 'image/encoded': bytes_feature(image_data)}))
    tfrecord_writer.write(example.SerializeToString())

def convert_train(output_dir, val_per=0.015, all_splits=config.SPLITS, file_idx_start=0):
    class_hist = {'blouse': 0, 'dress': 0, 'outwear': 0, 'skirt': 0, 'trousers': 0}
    start_file_idx = {'blouse': 5, 'dress': 3, 'outwear': 4, 'skirt': 4, 'trousers': 4}
    for cat in config.CATEGORIES:
        total_examples = 0
        sys.stdout.write('\nprocessing category: {}...'.format(cat))
        sys.stdout.flush()
        file_idx = file_idx_start
        record_idx = 0
        tf_filename = os.path.join(output_dir, '%s_%04d.tfrecord' % (cat, file_idx))
        tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
        tf_val_filename = os.path.join(output_dir, '%s_%04d_val.tfrecord' % (cat, 0))
        val_tfrecord_writer = tf.python_io.TFRecordWriter(tf_val_filename)
        this_key_map = keymap_factory[cat]
        for split in all_splits:
            if 'test' in split:
                continue
            sys.stdout.write('\nprocessing split: {}...\n'.format(split))
            sys.stdout.flush()
            split_path = os.path.join(config.DATA_DIR, split)
            anna_root = os.path.join(split_path, 'Annotations')
            anna_file = os.path.join(anna_root, os.listdir(anna_root)[0])
            anna_pd = pd.read_csv(anna_file)
            anna_pd = anna_pd.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
            this_nums = len(anna_pd.index)
            total_examples += this_nums
            all_columns_name = list(anna_pd.columns)
            all_columns_name = sorted([s.strip() for s in all_columns_name[2:]])
            for index, row in anna_pd.iterrows():
                sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, this_nums))
                sys.stdout.flush()
                category = row['image_category']
                if not cat in category:
                    continue
                class_hist[category] += 1
                image_file = row['image_id']
                full_file_path = os.path.join(split_path, image_file)
                class_id = config.category2ind[category]
                keypoint_x = []
                keypoint_y = []
                keypoint_v = []
                keypoint_id = []
                keypoint_global_id = []
                for keys in config.all_keys:
                    if keys in this_key_map:
                        keypoint_id.append(this_key_map[keys])
                    else:
                        keypoint_id.append(-1)
                    keypoint_global_id.append(config.key2ind[keys] - 1)
                    keypoint_info = row[keys].strip().split('_')
                    keypoint_x.append(int(keypoint_info[0]))
                    keypoint_y.append(int(keypoint_info[1]))
                    keypoint_v.append(int(keypoint_info[2]))
                if np.random.random_sample() > val_per:
                    _add_to_tfrecord(tfrecord_writer, full_file_path, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id)
                else:
                    _add_to_tfrecord(val_tfrecord_writer, full_file_path, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id)
                record_idx += 1
                if record_idx > SAMPLES_PER_FILES:
                    record_idx = 0
                    file_idx += 1
                    tf_filename = os.path.join(output_dir, '%s_%04d.tfrecord' % (cat, file_idx))
                    tfrecord_writer.flush()
                    tfrecord_writer.close()
                    tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
        val_tfrecord_writer.flush()
        val_tfrecord_writer.close()
    print('\nFinished converting the whole dataset!')
    print(class_hist, total_examples)
    return (class_hist, total_examples)

def convert_test(output_dir, splits=config.SPLITS):
    class_hist = {'blouse': 0, 'dress': 0, 'outwear': 0, 'skirt': 0, 'trousers': 0}
    for cat in config.CATEGORIES:
        total_examples = 0
        sys.stdout.write('\nprocessing category: {}...'.format(cat))
        sys.stdout.flush()
        file_idx = 0
        record_idx = 0
        tf_filename = os.path.join(output_dir, '%s_%04d.tfrecord' % (cat, file_idx))
        tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
        this_key_map = keymap_factory[cat]
        for split in splits:
            if 'train' in split:
                continue
            sys.stdout.write('\nprocessing split: {}...\n'.format(split))
            sys.stdout.flush()
            split_path = os.path.join(config.DATA_DIR, split)
            anna_file = os.path.join(split_path, 'test.csv')
            anna_pd = pd.read_csv(anna_file)
            this_nums = len(anna_pd.index)
            total_examples += this_nums
            for index, row in anna_pd.iterrows():
                sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, this_nums))
                sys.stdout.flush()
                category = row['image_category']
                if not cat in category:
                    continue
                class_hist[category] += 1
                image_file = row['image_id']
                full_file_path = os.path.join(split_path, image_file)
                class_id = config.category2ind[category]
                _test_add_to_tfrecord(tfrecord_writer, full_file_path, image_file, class_id)
                record_idx += 1
                if record_idx > SAMPLES_PER_FILES:
                    record_idx = 0
                    file_idx += 1
                    tf_filename = os.path.join(output_dir, '%s_%04d.tfrecord' % (cat, file_idx))
                    tfrecord_writer.flush()
                    tfrecord_writer.close()
                    tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
    print('\nFinished converting the whole test dataset!')
    print(class_hist, total_examples)
    return (class_hist, total_examples)

def count_split_examples(split_path, file_pattern=''):
    num_samples = 0
    tfrecords_to_count = [os.path.join(split_path, file) for file in os.listdir(split_path) if file_pattern in file]
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    return num_samples

