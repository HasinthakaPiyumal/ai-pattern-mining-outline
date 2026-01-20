# Cluster 159

def augment_imgs(img_list, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img
    return [_augment(img) for img in img_list]

def _augment(img):
    if hflip:
        img = img[:, ::-1, :]
    if vflip:
        img = img[::-1, :, :]
    if rot90:
        img = img.transpose(1, 0, 2)
    return img

