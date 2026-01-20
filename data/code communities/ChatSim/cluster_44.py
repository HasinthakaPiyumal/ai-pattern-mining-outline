# Cluster 44

def merge_videos(video_path1, video_path2, output_path):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, min(fps1, fps2), (width1 + width2, max(height1, height2)))
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 and (not ret2):
            break
        if not ret1:
            frame1 = 255 * np.ones((height1, width1, 3), dtype=np.uint8)
        if not ret2:
            frame2 = 255 * np.ones((height2, width2, 3), dtype=np.uint8)
        if height1 != height2:
            frame1 = cv2.resize(frame1, (width1, height2))
            frame2 = cv2.resize(frame2, (width2, height2))
        merged_frame = cv2.hconcat([frame1, frame2])
        out.write(merged_frame)
    cap1.release()
    cap2.release()
    out.release()

