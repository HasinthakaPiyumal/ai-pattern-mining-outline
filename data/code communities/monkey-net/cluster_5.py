# Cluster 5

def read_video(name, image_shape):
    if name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)
        if image.shape[2] == 4:
            image = image[..., :3]
        image = img_as_float32(image)
        video_array = np.moveaxis(image, 1, 0)
        video_array = video_array.reshape((-1,) + image_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception('Unknown file extensions  %s' % name)
    return video_array

class FramesDataset(Dataset):
    """Dataset of videos, videos can be represented as an image of concatenated frames, or in '.mp4','.gif' format"""

    def __init__(self, root_dir, augmentation_params, image_shape=(64, 64, 3), is_train=True, random_seed=0, pairs_list=None, transform=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.image_shape = tuple(image_shape)
        self.pairs_list = pairs_list
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print('Use predefined train-test split.')
            train_images = os.listdir(os.path.join(root_dir, 'train'))
            test_images = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print('Use random train-test split.')
            train_images, test_images = train_test_split(self.images, random_state=random_seed, test_size=0.2)
        if is_train:
            self.images = train_images
        else:
            self.images = test_images
        if transform is None:
            if is_train:
                self.transform = AllAugmentationTransform(**augmentation_params)
            else:
                self.transform = VideoToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        video_array = read_video(img_name, image_shape=self.image_shape)
        out = self.transform(video_array)
        out['name'] = os.path.basename(img_name)
        return out

