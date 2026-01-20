# Cluster 1

def process(video, filename, outdir, skipdir):
    global mask, value, drawing
    video2 = video.copy()
    current_it = 0
    cv.namedWindow('input')
    cv.setMouseCallback('input', onmouse)
    cv.moveWindow('input', video.shape[2] + 10, 90)
    video_as_num = color_as_num(video)
    original_fillmask = video_as_num != 256 ** 3 - 1
    paused = False
    while 1:
        current_frame = current_it // 25
        cv.imshow('input', video[current_frame % video.shape[0]])
        k = cv.waitKey(1)
        if not paused:
            current_it += 1
        if k == 27:
            break
        elif k == ord('0'):
            print(' Mark region to fill with left mouse button \n')
            value = FILL_AREA
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif k == ord('1'):
            print('Mark colors to fill with left mouse button \n')
            value = FILL_COLOR
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif k == ord('2'):
            print('Mark area to fill (in  all frames) with left mouse button \n')
            value = FILL_ALL_COLOR
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif k == ord('3'):
            print('Mark connected area to fill with left mouse button \n')
            value = FILL_CONNECTED
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif k == ord('f'):
            color = np.random.randint(255, size=3)
            video_as_num = color_as_num(video)
            fillmask = video_as_num != 256 ** 3 - 1
            for i in range(len(video)):
                video[i, scipy.ndimage.morphology.binary_fill_holes(fillmask[i])] = color
            video[original_fillmask] = (0, 0, 0)
        elif k == ord('d'):
            video_as_num = color_as_num(video)
            fillmask = video_as_num != 256 ** 3 - 1
            for i in range(len(video)):
                video[i, binary_dilation(fillmask[i])] = (0, 0, 0)
        elif k == ord('e'):
            video_as_num = color_as_num(video)
            fillmask = video_as_num != 256 ** 3 - 1
            for i in range(len(video)):
                video[i, np.logical_not(binary_erosion(fillmask[i]))] = (255, 255, 255)
        elif k == ord('i'):
            video = 255 - video
        elif k == ord('p'):
            video = np.array([img_as_ubyte(np.concatenate([median(frame[..., i], disk(1))[..., np.newaxis] for i in range(3)], axis=-1)) for frame in video])
        elif k == ord('l'):
            paused = not paused
        elif k == ord('n'):
            mimsave(os.path.join(outdir, filename), video[..., ::-1])
            break
        elif k == ord('s'):
            mimsave(os.path.join(skipdir, filename), video2[..., ::-1])
            break
        elif k == ord('r'):
            print('resetting \n')
            drawing = False
            video = video2.copy()
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        if mask.sum() == 0:
            continue
        if value == FILL_AREA:
            video[:, mask.astype(bool)] = (255, 255, 255)
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif value == FILL_COLOR:
            colors = video[current_frame % video.shape[0]][mask.astype(bool)]
            colors = color_as_num(val=colors).reshape((-1,))
            colors = np.unique(colors)
            video_as_num = color_as_num(video)
            for color in colors:
                video[video_as_num == color] = (255, 255, 255)
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif value == FILL_ALL_COLOR:
            colors = video[:, mask.astype(bool)]
            colors = color_as_num(val=colors).reshape((-1,))
            colors = np.unique(colors)
            video_as_num = color_as_num(video)
            for color in colors:
                video[video_as_num == color] = (255, 255, 255)
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
        elif value == FILL_CONNECTED:
            color = np.random.randint(255, size=3)
            video_as_num = color_as_num(video)
            fillmask = video_as_num != 256 ** 3 - 1
            for i in range(len(video)):
                labels = label(fillmask[i])
                index = labels[mask]
                video[i, labels == np.unique(index)] = color
            video[original_fillmask] = (0, 0, 0)
            mask = np.zeros(video.shape[1:3], dtype=np.uint8)
    cv.destroyAllWindows()

def color_as_num(val):
    val = val.astype(np.uint64)
    return val[..., 0] + 256 * val[..., 1] + 256 * 256 * val[..., 2]

def get_files_by_file_size(filepaths, dir, reverse=False):
    """ Return list of file paths  sorted by file size """
    for i in range(len(filepaths)):
        filepaths[i] = (filepaths[i], os.path.getsize(os.path.join(dir, filepaths[i])))
    filepaths.sort(key=lambda filename: filename[1], reverse=reverse)
    for i in range(len(filepaths)):
        filepaths[i] = filepaths[i][0]
    return filepaths

def convert_gif_to_frames(gif):
    frame_num = 0
    frame_list = []
    while True:
        try:
            okay, frame = gif.read()
            if not okay:
                break
            frame = cv.resize(frame, image_shape, interpolation=cv.INTER_NEAREST)
            frame_list.append(frame)
            frame_num += 1
        except KeyboardInterrupt:
            break
    return frame_list

