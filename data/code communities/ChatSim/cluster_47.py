# Cluster 47

def vis_traj(seq, output_boxes):
    frames_list = []
    for frame, box in zip(seq.frames, output_boxes):
        frame = cv2.imread(frame)
        x, y, w, h = box
        x1, y1, x2, y2 = map(lambda x: int(x), [x, y, x + w, y + h])
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        frames_list.append(frame)
    return frames_list

