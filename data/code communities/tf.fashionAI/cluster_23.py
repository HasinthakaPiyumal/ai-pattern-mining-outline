# Cluster 23

def slim_get_split(dataset_dir, image_preprocessing_fn, batch_size, num_readers, num_preprocessing_threads, num_epochs=None, is_training=True, category='blouse', file_pattern='{}_????', reader=None, return_keypoints=False):
    if reader is None:
        reader = tf.TFRecordReader
    num_joints = config.class_num_joints[category]
    suffix = '.tfrecord' if is_training else '_val.tfrecord'
    file_pattern = file_pattern.format(category) + suffix
    keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''), 'image/filename': tf.FixedLenFeature((), tf.string, default_value=''), 'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'), 'image/height': tf.FixedLenFeature([1], tf.int64), 'image/width': tf.FixedLenFeature([1], tf.int64), 'image/channels': tf.FixedLenFeature([1], tf.int64), 'image/classid': tf.FixedLenFeature([1], tf.int64), 'image/keypoint/x': tf.VarLenFeature(dtype=tf.int64), 'image/keypoint/y': tf.VarLenFeature(dtype=tf.int64), 'image/keypoint/v': tf.VarLenFeature(dtype=tf.int64), 'image/keypoint/id': tf.VarLenFeature(dtype=tf.int64), 'image/keypoint/gid': tf.VarLenFeature(dtype=tf.int64)}
    items_to_handlers = {'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'), 'height': slim.tfexample_decoder.Tensor('image/height'), 'width': slim.tfexample_decoder.Tensor('image/width'), 'channels': slim.tfexample_decoder.Tensor('image/channels'), 'classid': slim.tfexample_decoder.Tensor('image/classid'), 'keypoint/x': slim.tfexample_decoder.Tensor('image/keypoint/x'), 'keypoint/y': slim.tfexample_decoder.Tensor('image/keypoint/y'), 'keypoint/v': slim.tfexample_decoder.Tensor('image/keypoint/v'), 'keypoint/id': slim.tfexample_decoder.Tensor('image/keypoint/id'), 'keypoint/gid': slim.tfexample_decoder.Tensor('image/keypoint/gid')}
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    input_source = os.path.join(dataset_dir, file_pattern)
    dataset = slim.dataset.Dataset(data_sources=input_source, reader=reader, decoder=decoder, num_samples=config.split_size[category]['train' if is_training else 'val'], items_to_descriptions=None, num_classes=num_joints, labels_to_names=None)
    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=num_readers, common_queue_capacity=32 * batch_size, common_queue_min=8 * batch_size, shuffle=True, num_epochs=num_epochs)
    [org_image, height, width, channels, classid, key_x, key_y, key_v, key_id, key_gid] = provider.get(['image', 'height', 'width', 'channels', 'classid', 'keypoint/x', 'keypoint/y', 'keypoint/v', 'keypoint/id', 'keypoint/gid'])
    gather_ind = config.class2global_ind_map[category]
    key_x, key_y, key_v, key_id, key_gid = (tf.gather(key_x, gather_ind), tf.gather(key_y, gather_ind), tf.gather(key_v, gather_ind), tf.gather(key_id, gather_ind), tf.gather(key_gid, gather_ind))
    shape = tf.stack([height, width, channels], axis=0)
    if not return_keypoints:
        image, targets, new_key_v, isvalid, norm_value = image_preprocessing_fn(org_image, classid, shape, key_x, key_y, key_v)
        batch_list = [image, shape, classid, targets, new_key_v, isvalid, norm_value]
    else:
        image, targets, new_key_x, new_key_y, new_key_v, isvalid, norm_value = image_preprocessing_fn(org_image, classid, shape, key_x, key_y, key_v)
        batch_list = [image, shape, classid, targets, new_key_x, new_key_y, new_key_v, isvalid, norm_value]
    batch_input = tf.train.batch(batch_list, dynamic_pad=False, batch_size=batch_size, allow_smaller_final_batch=True, num_threads=num_preprocessing_threads, capacity=64 * batch_size)
    return batch_input

