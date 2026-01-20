# Cluster 0

def convert(source_dir, target_dir, crop_size, out_size, exts=[''], num_shards=128, tfrecords_prefix=''):
    if not tf.gfile.Exists(source_dir):
        print('source_dir does not exists')
        return
    if tfrecords_prefix and (not tfrecords_prefix.endswith('-')):
        tfrecords_prefix += '-'
    if tf.gfile.Exists(target_dir):
        print('{} is Already exists'.format(target_dir))
        return
    else:
        tf.gfile.MakeDirs(target_dir)
    path_list = []
    for ext in exts:
        pattern = '*.' + ext if ext != '' else '*'
        path = os.path.join(source_dir, pattern)
        path_list.extend(glob.glob(path))
    np.random.shuffle(path_list)
    num_files = len(path_list)
    num_per_shard = num_files // num_shards
    print('# of files: {}'.format(num_files))
    print('# of shards: {}'.format(num_shards))
    print('# files per shards: {}'.format(num_per_shard))
    shard_idx = 0
    writer = None
    for i, path in enumerate(path_list):
        if i % num_per_shard == 0 and shard_idx < num_shards:
            shard_idx += 1
            tfrecord_fn = '{}{:0>4d}-of-{:0>4d}.tfrecord'.format(tfrecords_prefix, shard_idx, num_shards)
            tfrecord_path = os.path.join(target_dir, tfrecord_fn)
            print('Writing {} ...'.format(tfrecord_path))
            if shard_idx > 1:
                writer.close()
            writer = tf.python_io.TFRecordWriter(tfrecord_path)
        im = scipy.misc.imread(path, mode='RGB')
        try:
            im = center_crop(im, crop_size)
        except Exception as e:
            print('[Exception] {}'.format(e))
            continue
        im = scipy.misc.imresize(im, out_size)
        example = tf.train.Example(features=tf.train.Features(feature={'image': _bytes_features([im.tostring()])}))
        writer.write(example.SerializeToString())
    writer.close()

def center_crop(im, output_size):
    output_height, output_width = output_size
    h, w = im.shape[:2]
    if h < output_height and w < output_width:
        raise ValueError('image is small')
    offset_h = int((h - output_height) / 2)
    offset_w = int((w - output_width) / 2)
    return im[offset_h:offset_h + output_height, offset_w:offset_w + output_width, :]

def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

