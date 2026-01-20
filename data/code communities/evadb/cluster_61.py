# Cluster 61

def annotate_video(segmentations, input_video_path, output_video_path, model_name='hfsegmentation'):
    all_segments = segmentations[f'mask']
    all_labels = segmentations[f'label']
    color_mapping = get_color_mapping(all_labels)
    vcap = cv2.VideoCapture(input_video_path)
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    fps = vcap.get(5)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_id = 0
    ret, frame = vcap.read()
    while ret and frame_id < len(all_segments):
        segments = all_segments[frame_id]
        labels = all_labels[frame_id]
        new_frame = annotate_single_frame(frame, segments, labels, color_mapping)
        video.write(np.array(new_frame))
        if frame_id % 5 == 0:
            legend_patches = [mpatches.Patch(color=np.array(color_mapping[label]) / 255, label=label) for label in set(labels)]
            plt.imshow(new_frame)
            plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        frame_id += 1
        ret, frame = vcap.read()
    video.release()
    vcap.release()

def get_color_mapping(all_labels):
    unique_labels = set((label for labels in all_labels for label in labels))
    num_colors = len(unique_labels)
    colormap = plt.colormaps['tab20']
    colors = [colormap(i % 20)[:3] for i in range(num_colors)]
    colors = [tuple((int(c * 255) for c in color)) for color in colors]
    color_mapping = {label: color for label, color in zip(unique_labels, colors)}
    return color_mapping

def annotate_single_frame(frame, segments, labels, color_mapping):
    overlay = np.zeros_like(frame)
    for mask, label in zip(segments, labels):
        mask_np = np.array(mask).astype(bool)
        overlay[mask_np] = color_mapping[label]
    new_frame = Image.blend(Image.fromarray(frame.astype(np.uint8)), Image.fromarray(overlay.astype(np.uint8)), alpha=0.5)
    return new_frame

