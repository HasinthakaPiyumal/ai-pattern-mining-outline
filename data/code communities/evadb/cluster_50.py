# Cluster 50

def annotate_video(detections, input_video_path, output_video_path):
    color1 = (207, 248, 64)
    color2 = (255, 49, 49)
    thickness = 4
    vcap = cv2.VideoCapture(input_video_path)
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    fps = vcap.get(5)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_id = 0
    label = detections['labels'][0]
    ret, frame = vcap.read()
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = (350, 100, 900, 500)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color1, thickness)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color1, thickness)
        cv2.putText(frame, 'Frame ID: ' + str(frame_id), (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color2, thickness)
        video.write(frame)
        if frame_id % 40 == 0:
            plt.imshow(frame)
            plt.show()
        frame_id += 1
        ret, frame = vcap.read()
    video.release()
    vcap.release()

