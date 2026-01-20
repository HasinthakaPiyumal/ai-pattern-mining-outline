# Cluster 66

class SimpleImageSquareMaskDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.mask = torch.FloatTensor(create_rectangle_mask(*self.dataset.image_size))
        self.model = Model()

    def __getitem__(self, index):
        img = self.dataset[index]
        mask = self.mask.clone()
        inpainted = self.model(img[None, ...], mask[None, ...])
        return dict(image=img, mask=mask, inpainted=inpainted)

    def __len__(self):
        return len(self.dataset)

def create_rectangle_mask(height, width):
    mask = np.ones((height, width))
    up_left_corner = (width // 4, height // 4)
    down_right_corner = (width - up_left_corner[0] - 1, height - up_left_corner[1] - 1)
    cv2.rectangle(mask, up_left_corner, down_right_corner, (0, 0, 0), thickness=cv2.FILLED)
    return mask

