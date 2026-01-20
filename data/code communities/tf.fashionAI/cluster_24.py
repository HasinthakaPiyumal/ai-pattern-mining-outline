# Cluster 24

def _random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
    if not image_list:
        raise ValueError('Empty image_list.')
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(tf.equal(image_rank, 3), ['Wrong rank for tensor  %s [expected] [actual]', image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)
    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(tf.logical_and(tf.greater_equal(image_height, crop_height), tf.greater_equal(image_width, crop_width)), ['Crop size greater than the image size.'])
    asserts = [rank_assertions[0], crop_size_assert]
    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        height_assert = tf.Assert(tf.equal(height, image_height), ['Wrong height for tensor %s [expected][actual]', image.name, height, image_height])
        width_assert = tf.Assert(tf.equal(width, image_width), ['Wrong width for tensor %s [expected][actual]', image.name, width, image_width])
        asserts.extend([height_assert, width_assert])
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)
    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]

def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
    original_shape = tf.shape(image)
    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
    size_assertion = tf.Assert(tf.logical_and(tf.greater_equal(original_shape[0], crop_height), tf.greater_equal(original_shape[1], crop_width)), ['Crop size greater than the image size.'])
    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)

def _central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2
        outputs.append(_crop(image, offset_height, offset_width, crop_height, crop_width))
    return outputs

def _aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)
    scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    return (new_height, new_width)

def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
    shape = tf.shape(image)
    height, width = (shape[0], shape[1])
    new_height, new_width = _smallest_size_at_least(height, width, resize_min)
    return _resize_image(image, new_height, new_width)

def preprocess_image(image_buffer, bbox, output_height, output_width, num_channels, is_training=False):
    """Preprocesses the given image.

  Preprocessing includes decoding, cropping, and resizing for both training
  and eval images. Training preprocessing, however, introduces some random
  distortion of the image to improve accuracy.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    num_channels: Integer depth of the image buffer for decoding.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
    if is_training:
        image = _decode_crop_and_flip(image_buffer, bbox, num_channels)
        image = _resize_image(image, output_height, output_width)
    else:
        image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, output_height, output_width)
    image.set_shape([output_height, output_width, num_channels])
    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)

def _decode_crop_and_flip(image_buffer, bbox, num_channels):
    """Crops the given image to a random part of the image, and randomly flips.

  We use the fused decode_and_crop op, which performs better than the two ops
  used separately in series, but note that this requires that the image be
  passed in as an un-decoded string Tensor.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    num_channels: Integer depth of the image buffer for decoding.

  Returns:
    3-D tensor with cropped image.

  """
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(tf.image.extract_jpeg_shape(image_buffer), bounding_boxes=bbox, min_object_covered=0.1, aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0], max_attempts=100, use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    cropped = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=num_channels)
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped

def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.

  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.

  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.

  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
    return tf.image.resize_images(image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

