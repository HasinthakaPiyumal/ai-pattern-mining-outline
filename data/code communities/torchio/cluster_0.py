# Cluster 0

def plot_batch(sampler):
    queue = tio.Queue(dataset, max_queue_length, patches_per_volume, sampler)
    loader = tio.SubjectsLoader(queue, batch_size=16)
    batch = tio.utils.get_first_item(loader)
    _, axes = plt.subplots(4, 4, figsize=(12, 10))
    for ax, im in zip(axes.flatten(), batch['t1']['data']):
        ax.imshow(im.squeeze(), cmap='gray')
    plt.suptitle(sampler.__class__.__name__)
    plt.tight_layout()

