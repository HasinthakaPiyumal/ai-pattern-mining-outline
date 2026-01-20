# Cluster 4

class KPDataset(Dataset):
    """Dataset of detected keypoints"""

    def __init__(self, keypoints_array, num_frames):
        self.keypoints_array = keypoints_array
        self.transform = SelectRandomFrames(consequent=True, number_of_frames=num_frames)

    def __len__(self):
        return len(self.keypoints_array)

    def __getitem__(self, idx):
        keypoints = self.keypoints_array[idx]
        selected = self.transform(keypoints)
        selected = {k: np.concatenate([v[k][0] for v in selected], axis=0) for k in selected[0].keys()}
        return selected

