# Cluster 8

class RandomCrop(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' + 'but got list of {0}'.format(type(clip[0])))
        clip = pad_clip(clip, h, w)
        im_h, im_w = clip.shape[1:3]
        x1 = 0 if h == im_h else random.randint(0, im_w - w)
        y1 = 0 if w == im_w else random.randint(0, im_h - h)
        cropped = crop_clip(clip, y1, x1, h, w)
        return cropped

def pad_clip(clip, h, w):
    im_h, im_w = clip[0].shape[:2]
    pad_h = (0, 0) if h < im_h else ((h - im_h) // 2, (h - im_h + 1) // 2)
    pad_w = (0, 0) if w < im_w else ((w - im_w) // 2, (w - im_w + 1) // 2)
    return pad(clip, ((0, 0), pad_h, pad_w, (0, 0)), mode='edge')

def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' + 'but got list of {0}'.format(type(clip[0])))
    return cropped

