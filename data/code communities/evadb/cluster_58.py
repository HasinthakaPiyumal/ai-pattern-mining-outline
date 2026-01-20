# Cluster 58

def show_video_frame(input_video_path, show_frame_number=100):
    vcap = cv2.VideoCapture(input_video_path)
    vcap.set(1, show_frame_number)
    ret, frame = vcap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    plt.imshow(frame)
    plt.show()
    vcap.release()

