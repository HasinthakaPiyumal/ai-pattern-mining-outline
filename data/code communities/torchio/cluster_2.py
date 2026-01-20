# Cluster 2

def plot_gif(image):

    def _update_frame(num):
        frame = get_frame(image, num)
        im.set_data(frame)
        return

    def get_frame(image, i):
        return image.data[..., i].permute(1, 2, 0).byte()
    plt.rcParams['animation.embed_limit'] = 25
    fig, ax = plt.subplots()
    im = ax.imshow(get_frame(image, 0))
    return animation.FuncAnimation(fig, _update_frame, repeat_delay=image['delay'], frames=image.shape[-1])

def get_frame(image, i):
    return image.data[..., i].permute(1, 2, 0).byte()

def read_clip(path, undersample=4):
    """Read a GIF a return an array of shape (C, W, H, T)."""
    gif = Image.open(path)
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frames.append(np.array(gif.convert('RGB')))
    frames = frames[::undersample]
    array = np.stack(frames).transpose(3, 1, 2, 0)
    delay = gif.info['duration']
    return (array, delay)

