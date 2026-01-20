# Cluster 17

def frame2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    im_list = sorted([os.path.join(im_dir, img) for img in os.listdir(im_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for i in im_list:
        im_name = os.path.join(im_dir, i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
    videoWriter.release()
    print('Done')

