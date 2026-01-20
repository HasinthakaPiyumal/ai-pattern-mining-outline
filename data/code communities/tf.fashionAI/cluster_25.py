# Cluster 25

def rotate_all(images, angles, x, y, interpolation='NEAREST'):
    """Rotate image(s) by the passed angle(s) in radians.
  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW).
    angles: A scalar angle to rotate all images by, or (if images has rank 4)
       a vector of length num_images, with an angle for each image in the batch.
    interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
  Returns:
    Image(s) with the same type and shape as `images`, rotated by the given
    angle(s). Empty space due to the rotation will be filled with zeros.
  Raises:
    TypeError: If `image` is an invalid type.
  """
    image_or_images = tf.convert_to_tensor(images, name='images')
    if len(image_or_images.get_shape()) == 2:
        images = image_or_images[None, :, :, None]
    elif len(image_or_images.get_shape()) == 3:
        images = image_or_images[None, :, :, :]
    elif len(image_or_images.get_shape()) == 4:
        images = image_or_images
    else:
        raise TypeError('Images should have rank between 2 and 4.')
    image_height = tf.cast(tf.shape(images)[1], tf.float32)[None]
    image_width = tf.cast(tf.shape(images)[2], tf.float32)[None]
    rotate_matrix = get_projective_transforms(angles, image_height, image_width, x, y)
    flaten_rotate_matrix = tf.squeeze(rotate_matrix)
    a0, a1, a2, b0, b1, b2 = (flaten_rotate_matrix[0], flaten_rotate_matrix[1], flaten_rotate_matrix[2], flaten_rotate_matrix[3], flaten_rotate_matrix[4], flaten_rotate_matrix[5])
    normalizor = a1 * b0 - a0 * b1 + 1e-08
    new_x = -(b1 * x - a1 * y - b1 * a2 + a1 * b2) / normalizor
    new_y = (b0 * x - a0 * y - a2 * b0 + a0 * b2) / normalizor
    output = tf.contrib.image.transform(images, rotate_matrix, interpolation=interpolation)
    if len(image_or_images.get_shape()) == 2:
        return (output[0, :, :, 0], new_x, new_y)
    elif len(image_or_images.get_shape()) == 3:
        return (output[0, :, :, :], new_x, new_y)
    else:
        return (output, new_x, new_y)

def get_projective_transforms(angles, image_height, image_width, x, y, name=None):
    """Returns projective transform(s) for the given angle(s).
  Args:
    angles: A scalar angle to rotate all images by, or (for batches of images)
        a vector with an angle to rotate each image in the batch. The rank must
        be statically known (the shape is not `TensorShape(None)`.
    image_height: Height of the image(s) to be transformed.
    image_width: Width of the image(s) to be transformed.
  Returns:
    A tensor of shape (num_images, 8). Projective transforms which can be given
      to `tf.contrib.image.transform`.
  """
    with tf.name_scope(name, 'get_projective_transforms'):
        angle_or_angles = tf.convert_to_tensor(angles, name='angles', dtype=tf.float32)
        if len(angle_or_angles.get_shape()) == 0:
            angles = angle_or_angles[None]
        elif len(angle_or_angles.get_shape()) == 1:
            angles = angle_or_angles
        else:
            raise TypeError('Angles should have rank 0 or 1.')
        valid_x = tf.boolean_mask(x, x > 0.0)
        valid_y = tf.boolean_mask(y, y > 0.0)
        min_x = tf.reduce_min(valid_x, axis=-1)
        max_x = tf.reduce_max(valid_x, axis=-1)
        min_y = tf.reduce_min(valid_y, axis=-1)
        max_y = tf.reduce_max(valid_y, axis=-1)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        x_offset = center_x - (tf.cos(angles) * image_width / 2.0 - tf.sin(angles) * image_height / 2.0)
        y_offset = center_y - (tf.sin(angles) * image_width / 2.0 + tf.cos(angles) * image_height / 2.0)
        num_angles = tf.shape(angles)[0]
        return tf.concat(values=[tf.cos(angles)[:, None], -tf.sin(angles)[:, None], x_offset[:, None], tf.sin(angles)[:, None], tf.cos(angles)[:, None], y_offset[:, None], tf.zeros((num_angles, 2), tf.float32)], axis=1)

def rotate_augum(image, shape, fkey_x, fkey_y, bbox_border):
    x_mask = fkey_x > 0.0
    y_mask = fkey_y > 0.0
    bak_fkey_x, bak_fkey_y, bak_image = (fkey_x / tf.cast(shape[1], tf.float32), fkey_y / tf.cast(shape[0], tf.float32), image)
    image, fkey_x, fkey_y = tf.cond(tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32)[0] < 0.4, lambda: rotate_all(image, tf.random_uniform([1], minval=-3.14 / 6.0, maxval=3.14 / 6.0, dtype=tf.float32)[0], fkey_x, fkey_y), lambda: (image, fkey_x, fkey_y))
    fkey_x, fkey_y = (fkey_x / tf.cast(shape[1], tf.float32), fkey_y / tf.cast(shape[0], tf.float32))
    x_mask_ = tf.logical_and(fkey_x > 0.0, fkey_x < 1.0)
    y_mask_ = tf.logical_and(fkey_y > 0.0, fkey_y < 1.0)
    x_mask = tf.logical_and(x_mask, x_mask_)
    y_mask = tf.logical_and(y_mask, y_mask_)
    fkey_x = fkey_x * tf.cast(x_mask, tf.float32) + (tf.cast(x_mask, tf.float32) - 1.0)
    fkey_y = fkey_y * tf.cast(y_mask, tf.float32) + (tf.cast(y_mask, tf.float32) - 1.0)
    new_image, new_fkey_x, new_fkey_y = tf.cond(tf.count_nonzero(tf.logical_and(x_mask, y_mask)) > 0, lambda: (image, fkey_x, fkey_y), lambda: (bak_image, bak_fkey_x, bak_fkey_y))
    valid_x = tf.boolean_mask(new_fkey_x, new_fkey_x > 0.0)
    valid_y = tf.boolean_mask(new_fkey_y, new_fkey_y > 0.0)
    min_x = tf.maximum(tf.reduce_min(valid_x, axis=-1) - bbox_border / tf.cast(shape[0], tf.float32), 0.0)
    max_x = tf.minimum(tf.reduce_max(valid_x, axis=-1) + bbox_border / tf.cast(shape[0], tf.float32), 1.0)
    min_y = tf.maximum(tf.reduce_min(valid_y, axis=-1) - bbox_border / tf.cast(shape[1], tf.float32), 0.0)
    max_y = tf.minimum(tf.reduce_max(valid_y, axis=-1) + bbox_border / tf.cast(shape[1], tf.float32), 1.0)
    return (new_image, new_fkey_x, new_fkey_y, tf.reshape(tf.stack([min_y, min_x, max_y, max_x], axis=-1), [1, 1, 4]))

def preprocess_for_train(image, classid, shape, output_height, output_width, key_x, key_y, key_v, norm_table, data_format, category, bbox_border, heatmap_sigma, heatmap_size, return_keypoints=False, resize_side_min=_RESIZE_SIDE_MIN, resize_side_max=_RESIZE_SIDE_MAX, fast_mode=False, scope=None, add_image_summaries=True):
    """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
    with tf.name_scope(scope, 'vgg_distort_image', [image, output_height, output_width]):
        orig_dtype = image.dtype
        if orig_dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        num_distort_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(image, lambda x, ordering: distort_color(x, ordering, fast_mode), num_cases=num_distort_cases)
        distorted_image = tf.to_float(tf.image.convert_image_dtype(distorted_image, orig_dtype, saturate=True))
        if add_image_summaries:
            tf.summary.image('color_distorted_image', tf.cast(tf.expand_dims(distorted_image, 0), tf.uint8))
        normarlized_image = _mean_image_subtraction(distorted_image, [_R_MEAN, _G_MEAN, _B_MEAN])
        fkey_x, fkey_y = (tf.cast(key_x, tf.float32), tf.cast(key_y, tf.float32))
        image, fkey_x, fkey_y, bbox = rotate_augum(normarlized_image, shape, fkey_x, fkey_y, bbox_border)
        distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
        distorted_bbox = tf.squeeze(distorted_bbox)
        fkey_x = fkey_x - distorted_bbox[1]
        fkey_y = fkey_y - distorted_bbox[0]
        outside_x = fkey_x >= distorted_bbox[3]
        outside_y = fkey_y >= distorted_bbox[2]
        fkey_x = fkey_x - tf.cast(outside_x, tf.float32)
        fkey_y = fkey_y - tf.cast(outside_y, tf.float32)
        fkey_x = fkey_x / (distorted_bbox[3] - distorted_bbox[1])
        fkey_y = fkey_y / (distorted_bbox[2] - distorted_bbox[0])
        distorted_image.set_shape([None, None, 3])
        if add_image_summaries:
            tf.summary.image('cropped_image', tf.expand_dims(distorted_image, 0))
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(distorted_image, lambda x, method: tf.image.resize_images(x, [output_height, output_width], method), num_cases=num_resize_cases)
        distorted_image.set_shape([output_height, output_width, 3])
        ikey_x = tf.cast(tf.round(fkey_x * heatmap_size), tf.int64)
        ikey_y = tf.cast(tf.round(fkey_y * heatmap_size), tf.int64)
        gather_ind = config.left_right_remap[category]
        if add_image_summaries:
            tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))
        distorted_image, new_key_x, new_key_y, new_key_v = tf.cond(tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32)[0] < 0.5, lambda: (tf.image.flip_left_right(distorted_image), heatmap_size - tf.gather(ikey_x, gather_ind), tf.gather(ikey_y, gather_ind), tf.gather(key_v, gather_ind)), lambda: (distorted_image, ikey_x, ikey_y, key_v))
        targets, isvalid = draw_labelmap(new_key_x, new_key_y, heatmap_sigma, heatmap_size)
        norm_gather_ind = tf.stack([norm_table[0].lookup(classid), norm_table[1].lookup(classid)], axis=-1)
        scale_x_ = tf.cast(output_width, tf.float32) / tf.cast(shape[1], tf.float32)
        scale_y_ = tf.cast(output_height, tf.float32) / tf.cast(shape[0], tf.float32)
        scale_x = tf.cast(output_width, tf.float32) / tf.cast(heatmap_size, tf.float32)
        scale_y = tf.cast(output_height, tf.float32) / tf.cast(heatmap_size, tf.float32)
        norm_x, norm_y = tf.cond(tf.reduce_sum(tf.gather(isvalid, norm_gather_ind)) < 2, lambda: (tf.cast(tf.gather(key_x, norm_gather_ind), tf.float32) * scale_x_, tf.cast(tf.gather(key_y, norm_gather_ind), tf.float32) * scale_y_), lambda: (tf.cast(tf.gather(new_key_x, norm_gather_ind), tf.float32) * scale_x, tf.cast(tf.gather(new_key_y, norm_gather_ind), tf.float32) * scale_y))
        norm_x, norm_y = (tf.squeeze(norm_x), tf.squeeze(norm_y))
        norm_value = tf.pow(tf.pow(norm_x[0] - norm_x[1], 2.0) + tf.pow(norm_y[0] - norm_y[1], 2.0), 0.5)
        if config.DEBUG:
            save_image_op = tf.py_func(save_image_with_heatmap, [unwhiten_image(distorted_image), targets, config.left_right_group_map[category][0], config.left_right_group_map[category][1], config.left_right_group_map[category][2], [output_height, output_width], heatmap_size], tf.int64, stateful=True)
            with tf.control_dependencies([save_image_op]):
                distorted_image = distorted_image / 255.0
        else:
            distorted_image = distorted_image / 255.0
        if data_format == 'NCHW':
            distorted_image = tf.transpose(distorted_image, perm=(2, 0, 1))
        if not return_keypoints:
            return (distorted_image, targets, new_key_v, isvalid, norm_value)
        else:
            return (distorted_image, targets, new_key_x, new_key_y, new_key_v, isvalid, norm_value)

def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        elif color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        else:
            raise ValueError('color_ordering must be in [0, 3]')
        return tf.clip_by_value(image, 0.0, 1.0)

def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

def distorted_bounding_box_crop(image, bbox, min_object_covered=1.0, aspect_ratio_range=(0.75, 1.33), area_range=(0.45, 1.0), max_attempts=100, scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox, min_object_covered=min_object_covered, aspect_ratio_range=aspect_ratio_range, area_range=area_range, max_attempts=max_attempts, use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return (cropped_image, distort_bbox)

def draw_labelmap(x, y, heatmap_sigma, heatmap_size):
    heatmap, isvalid = tf.map_fn(lambda pt: tf.py_func(np_draw_labelmap, [pt, heatmap_sigma, heatmap_size], [tf.float32, tf.int64], stateful=True), tf.stack([x, y], axis=-1), dtype=[tf.float32, tf.int64], parallel_iterations=10, back_prop=False, swap_memory=False, infer_shape=True)
    heatmap.set_shape([x.get_shape().as_list()[0], heatmap_size, heatmap_size])
    isvalid.set_shape([x.get_shape().as_list()[0]])
    return (heatmap, isvalid)

def unwhiten_image(image):
    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(axis=2, values=channels)

def preprocess_for_train_v0(image, classid, shape, output_height, output_width, key_x, key_y, key_v, norm_table, data_format, category, bbox_border, heatmap_sigma, heatmap_size, return_keypoints=False, resize_side_min=_RESIZE_SIDE_MIN, resize_side_max=_RESIZE_SIDE_MAX, fast_mode=True, scope=None, add_image_summaries=True):
    """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
    with tf.name_scope(scope, 'vgg_distort_image', [image, output_height, output_width]):
        fkey_x, fkey_y = (tf.cast(key_x, tf.float32), tf.cast(key_y, tf.float32))
        image, fkey_x, fkey_y, bbox = rotate_augum(image, shape, fkey_x, fkey_y, bbox_border)
        distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
        distorted_bbox = tf.squeeze(distorted_bbox)
        fkey_x = fkey_x - distorted_bbox[1]
        fkey_y = fkey_y - distorted_bbox[0]
        outside_x = fkey_x >= distorted_bbox[3]
        outside_y = fkey_y >= distorted_bbox[2]
        fkey_x = fkey_x - tf.cast(outside_x, tf.float32)
        fkey_y = fkey_y - tf.cast(outside_y, tf.float32)
        fkey_x = fkey_x / (distorted_bbox[3] - distorted_bbox[1])
        fkey_y = fkey_y / (distorted_bbox[2] - distorted_bbox[0])
        distorted_image.set_shape([None, None, 3])
        if add_image_summaries:
            tf.summary.image('cropped_image', tf.expand_dims(distorted_image, 0))
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(distorted_image, lambda x, method: tf.image.resize_images(x, [output_height, output_width], method), num_cases=num_resize_cases)
        distorted_image.set_shape([output_height, output_width, 3])
        ikey_x = tf.cast(tf.round(fkey_x * heatmap_size), tf.int64)
        ikey_y = tf.cast(tf.round(fkey_y * heatmap_size), tf.int64)
        gather_ind = config.left_right_remap[category]
        if add_image_summaries:
            tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))
        distorted_image, new_key_x, new_key_y, new_key_v = tf.cond(tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32)[0] < 0.5, lambda: (tf.image.flip_left_right(distorted_image), heatmap_size - tf.gather(ikey_x, gather_ind), tf.gather(ikey_y, gather_ind), tf.gather(key_v, gather_ind)), lambda: (distorted_image, ikey_x, ikey_y, key_v))
        distorted_image = tf.to_float(distorted_image)
        num_distort_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(distorted_image, lambda x, ordering: distort_color_v0(x, ordering, fast_mode), num_cases=num_distort_cases)
        if add_image_summaries:
            tf.summary.image('final_distorted_image', tf.cast(tf.expand_dims(distorted_image, 0), tf.uint8))
        targets, isvalid = draw_labelmap(new_key_x, new_key_y, heatmap_sigma, heatmap_size)
        norm_gather_ind = tf.stack([norm_table[0].lookup(classid), norm_table[1].lookup(classid)], axis=-1)
        scale_x_ = tf.cast(output_width, tf.float32) / tf.cast(shape[1], tf.float32)
        scale_y_ = tf.cast(output_height, tf.float32) / tf.cast(shape[0], tf.float32)
        scale_x = tf.cast(output_width, tf.float32) / tf.cast(heatmap_size, tf.float32)
        scale_y = tf.cast(output_height, tf.float32) / tf.cast(heatmap_size, tf.float32)
        norm_x, norm_y = tf.cond(tf.reduce_sum(tf.gather(isvalid, norm_gather_ind)) < 2, lambda: (tf.cast(tf.gather(key_x, norm_gather_ind), tf.float32) * scale_x_, tf.cast(tf.gather(key_y, norm_gather_ind), tf.float32) * scale_y_), lambda: (tf.cast(tf.gather(new_key_x, norm_gather_ind), tf.float32) * scale_x, tf.cast(tf.gather(new_key_y, norm_gather_ind), tf.float32) * scale_y))
        norm_x, norm_y = (tf.squeeze(norm_x), tf.squeeze(norm_y))
        norm_value = tf.pow(tf.pow(norm_x[0] - norm_x[1], 2.0) + tf.pow(norm_y[0] - norm_y[1], 2.0), 0.5)
        if config.DEBUG:
            save_image_op = tf.py_func(save_image_with_heatmap, [distorted_image, targets, config.left_right_group_map[category][0], config.left_right_group_map[category][1], config.left_right_group_map[category][2], [output_height, output_width], heatmap_size], tf.int64, stateful=True)
            with tf.control_dependencies([save_image_op]):
                normarlized_image = _mean_image_subtraction(distorted_image, [_R_MEAN, _G_MEAN, _B_MEAN])
        else:
            normarlized_image = _mean_image_subtraction(distorted_image, [_R_MEAN, _G_MEAN, _B_MEAN])
        if data_format == 'NCHW':
            normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))
        return (normarlized_image / 255.0, targets, new_key_v, isvalid, norm_value)

def distort_color_v0(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32.0)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32.0)
        elif color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.0)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.0)
        else:
            raise ValueError('color_ordering must be in [0, 3]')
        return tf.clip_by_value(image, 0.0, 255.0)

def preprocess_for_eval(image, classid, shape, output_height, output_width, key_x, key_y, key_v, norm_table, data_format, category, bbox_border, heatmap_sigma, heatmap_size, resize_side, scope=None):
    """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
    with tf.name_scope(scope, 'vgg_eval_image', [image, output_height, output_width]):
        fkey_x, fkey_y = (tf.cast(key_x, tf.float32) / tf.cast(shape[1], tf.float32), tf.cast(key_y, tf.float32) / tf.cast(shape[0], tf.float32))
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
        image = tf.squeeze(image, [0])
        image.set_shape([output_height, output_width, 3])
        image = tf.to_float(image)
        ikey_x = tf.cast(tf.round(fkey_x * heatmap_size), tf.int64)
        ikey_y = tf.cast(tf.round(fkey_y * heatmap_size), tf.int64)
        targets, isvalid = draw_labelmap(ikey_x, ikey_y, heatmap_sigma, heatmap_size)
        norm_gather_ind = tf.stack([norm_table[0].lookup(classid), norm_table[1].lookup(classid)], axis=-1)
        key_x = tf.cast(tf.round(fkey_x * output_width), tf.int64)
        key_y = tf.cast(tf.round(fkey_y * output_height), tf.int64)
        norm_x, norm_y = (tf.cast(tf.gather(key_x, norm_gather_ind), tf.float32), tf.cast(tf.gather(key_y, norm_gather_ind), tf.float32))
        norm_x, norm_y = (tf.squeeze(norm_x), tf.squeeze(norm_y))
        norm_value = tf.pow(tf.pow(norm_x[0] - norm_x[1], 2.0) + tf.pow(norm_y[0] - norm_y[1], 2.0), 0.5)
        if config.DEBUG:
            save_image_op = tf.py_func(save_image_with_heatmap, [image, targets, config.left_right_group_map[category][0], config.left_right_group_map[category][1], config.left_right_group_map[category][2], [output_height, output_width], heatmap_size], tf.int64, stateful=True)
            with tf.control_dependencies([save_image_op]):
                normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        else:
            normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        if data_format == 'NCHW':
            normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))
        return (normarlized_image / 255.0, targets, key_v, isvalid, norm_value)

def preprocess_for_test_v0(image, shape, output_height, output_width, data_format='NCHW', bbox_border=25.0, heatmap_sigma=1.0, heatmap_size=64, scope=None):
    """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
    with tf.name_scope(scope, 'vgg_test_image', [image, output_height, output_width]):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
        image = tf.squeeze(image, [0])
        image.set_shape([output_height, output_width, 3])
        image = tf.to_float(image)
        normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        if data_format == 'NCHW':
            normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))
        return normarlized_image / 255.0

def preprocess_for_test(image, file_name, shape, output_height, output_width, data_format='NCHW', bbox_border=25.0, heatmap_sigma=1.0, heatmap_size=64, pred_df=None, scope=None):
    """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
    with tf.name_scope(scope, 'vgg_test_image', [image, output_height, output_width]):
        if pred_df is not None:
            xmin, ymin, xmax, ymax = [table_.lookup(file_name) for table_ in pred_df]
            height, width, channals = tf.unstack(shape, axis=0)
            xmin, ymin, xmax, ymax = (xmin - 100, ymin - 80, xmax + 100, ymax + 80)
            xmin, ymin, xmax, ymax = (tf.clip_by_value(xmin, 0, width[0] - 1), tf.clip_by_value(ymin, 0, height[0] - 1), tf.clip_by_value(xmax, 0, width[0] - 1), tf.clip_by_value(ymax, 0, height[0] - 1))
            bbox_h = ymax - ymin
            bbox_w = xmax - xmin
            areas = bbox_h * bbox_w
            offsets = tf.stack([xmin, ymin], axis=0)
            crop_shape = tf.stack([bbox_h, bbox_w, channals[0]], axis=0)
            ymin, xmin, bbox_h, bbox_w = (tf.cast(ymin, tf.int32), tf.cast(xmin, tf.int32), tf.cast(bbox_h, tf.int32), tf.cast(bbox_w, tf.int32))
            crop_image = tf.image.crop_to_bounding_box(image, ymin, xmin, bbox_h, bbox_w)
            image, shape, offsets = tf.cond(areas > 0, lambda: (crop_image, crop_shape, offsets), lambda: (image, shape, tf.constant([0, 0], tf.int64)))
            offsets.set_shape([2])
            shape.set_shape([3])
        else:
            offsets = tf.constant([0, 0], tf.int64)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
        image = tf.squeeze(image, [0])
        image.set_shape([output_height, output_width, 3])
        if config.DEBUG:
            save_image_op = tf.py_func(_save_image, [image], tf.int64, stateful=True)
            image = tf.Print(image, [save_image_op])
        image = tf.to_float(image)
        normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        if data_format == 'NCHW':
            normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))
        return (normarlized_image / 255.0, shape, offsets)

def preprocess_for_test_raw_output(image, output_height, output_width, data_format='NCHW', scope=None):
    """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
    with tf.name_scope(scope, 'vgg_test_image_raw_output', [image, output_height, output_width]):
        image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
        image = tf.squeeze(image, [0])
        image.set_shape([output_height, output_width, 3])
        if config.DEBUG:
            save_image_op = tf.py_func(_save_image, [image], tf.int64, stateful=True)
            image = tf.Print(image, [save_image_op])
        image = tf.to_float(image)
        normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        if data_format == 'NCHW':
            normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))
        return tf.expand_dims(normarlized_image / 255.0, 0)

def preprocess_image(image, classid, shape, output_height, output_width, key_x, key_y, key_v, norm_table, is_training=False, data_format='NCHW', category='*', bbox_border=25.0, heatmap_sigma=1.0, heatmap_size=64, return_keypoints=False, resize_side_min=_RESIZE_SIDE_MIN, resize_side_max=_RESIZE_SIDE_MAX):
    """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].

  Returns:
    A preprocessed image.
  """
    if is_training:
        return preprocess_for_train(image, classid, shape, output_height, output_width, key_x, key_y, key_v, norm_table, data_format, category, bbox_border, heatmap_sigma, heatmap_size, return_keypoints, resize_side_min, resize_side_max)
    else:
        return preprocess_for_eval(image, classid, shape, output_height, output_width, key_x, key_y, key_v, norm_table, data_format, category, bbox_border, heatmap_sigma, heatmap_size, min(output_height, output_width))

