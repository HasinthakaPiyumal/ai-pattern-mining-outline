# Cluster 12

def match_detections_with_tracks(detections, tracks, iou_threshold=0.5):
    """
    Summary:
        Match detections with tracks based on their bounding boxes using IOU threshold.

    Args:
        detections (list): List of detections, each detection is a dictionary with keys (bbox, confidence, class_id)
        tracks (list): List of tracks, each track is a tuple of (bboxes, track_id, class, conf)
        iou_threshold (float): IOU threshold for matching detections with tracks.

    Returns:
        matched_detections (list): List of detections that are matched with tracks, each detection is a dictionary with keys (bbox, confidence, class_id)
        unmatched_detections (list): List of detections that are not matched with any tracks, each detection is a dictionary with keys (bbox, confidence, class_id)
    """
    matched_detections = []
    unmatched_detections = []
    for detection in detections:
        detection_bbox = detection['bbox']
        max_iou = 0
        matched_track = None
        for track in tracks:
            track_bbox = track[0:4]
            iou = compute_iou(detection_bbox, track_bbox)
            if iou > iou_threshold and iou > max_iou:
                matched_track = track
                max_iou = iou
        if matched_track is not None:
            detection['group_id'] = int(matched_track[4])
            matched_detections.append(detection)
            tracks.remove(matched_track)
        else:
            unmatched_detections.append(detection)
    return (matched_detections, unmatched_detections)

def compute_iou(box1, box2):
    """
    Summary:
        Computes IOU between two bounding boxes.

    Args:
        box1 (list): List of 4 coordinates (xmin, ymin, xmax, ymax) of the first box.
        box2 (list): List of 4 coordinates (xmin, ymin, xmax, ymax) of the second box.

    Returns:
        iou (float): IOU between the two boxes.
    """
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    if xmin < xmax and ymin < ymax:
        intersection_area = (xmax - xmin) * (ymax - ymin)
    else:
        intersection_area = 0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def get_boxes_conf_classids_segments(shapes):
    """
    Summary:
        Get bounding boxes, confidences, class ids, and segments from shapes (NOT QT).
        
    Args:
        shapes: a list of shapes
        
    Returns:
        boxes: a list of bounding boxes 
        confidences: a list of confidences
        class_ids: a list of class ids
        segments: a list of segments 
    """
    boxes = []
    confidences = []
    class_ids = []
    segments = []
    for s in shapes:
        label = s['label']
        points = s['points']
        segment = []
        for j in range(0, len(points), 2):
            segment.append([int(points[j]), int(points[j + 1])])
        segments.append(segment)
        boxes.append(get_bbox_xyxy(segment))
        confidences.append(float(s['content']))
        class_ids.append(coco_classes.index(label) if label in coco_classes else -1)
    return (boxes, confidences, class_ids, segments)

def get_bbox_xyxy(segment):
    """
    Summary:
        Get the bounding box of a polygon in format of [xmin, ymin, xmax, ymax].
        
    Args:
        segment: a list of points
        
    Returns:
        bbox: [x, y, w, h]
    """
    segment = np.array(segment)
    x0 = np.min(segment[:, 0])
    y0 = np.min(segment[:, 1])
    x1 = np.max(segment[:, 0])
    y1 = np.max(segment[:, 1])
    return [x0, y0, x1, y1]

def convert_qt_shapes_to_shapes(qt_shapes):
    """
    Summary:
        Convert QT shapes to shapes.
        
    Args:
        qt_shapes: a list of QT shapes
        
    Returns:
        shapes: a list of shapes
    """
    shapes = []
    for s in qt_shapes:
        shapes.append(dict(label=s.label.encode('utf-8') if PY2 else s.label, points=flattener(s.points), bbox=get_bbox_xyxy([(p.x(), p.y()) for p in s.points]), group_id=s.group_id, content=s.content, shape_type=s.shape_type, flags=s.flags))
    return shapes

def flattener(list_2d):
    """
    Summary:
        Flatten a list of QTpoints.
        
    Args:
        list_2d: a list of QTpoints
        
    Returns:
        points: a list of points
    """
    points = [(p.x(), p.y()) for p in list_2d]
    points = np.array(points, np.int16).flatten().tolist()
    return points

def track_area_adjustedBboex(area_points, dims, ratio=0.1):
    [x1, y1, x2, y2] = get_bbox_xyxy(area_points)
    [w, h] = [x2 - x1, y2 - y1]
    x1 = int(max(0, x1 - w * ratio))
    y1 = int(max(0, y1 - h * ratio))
    x2 = int(min(dims[1], x2 + w * ratio))
    y2 = int(min(dims[0], y2 + h * ratio))
    return [x1, y1, x2, y2]

def polygon_to_shape(polygon, score, className='SAM instance'):
    shape = {}
    shape['label'] = className
    shape['content'] = str(round(score, 2))
    shape['group_id'] = None
    shape['shape_type'] = 'polygon'
    shape['bbox'] = get_bbox_xyxy(polygon)
    shape['flags'] = {}
    shape['other_data'] = {}
    shape['points'] = [item for sublist in polygon for item in sublist]
    return shape

def OURnms_confidenceBased(shapes, iou_threshold=0.5):
    """
    Perform non-maximum suppression on a list of shapes based on their bounding boxes using IOU threshold.

    Args:
        shapes (list): List of shapes, each shape is a dictionary with keys (bbox, confidence, class_id)
        iou_threshold (float): IOU threshold for non-maximum suppression.

    Returns:
        list: List of shapes after performing non-maximum suppression, each shape is a dictionary with keys (bbox, confidence, class_id)
    """
    iou_threshold = float(iou_threshold)
    for shape in shapes:
        if shape['content'] is None:
            shape['content'] = 1.0
    shapes.sort(key=lambda x: x['content'], reverse=True)
    boxes, confidences, class_ids, segments = get_boxes_conf_classids_segments(shapes)
    toBeRemoved = []
    for i in range(len(shapes)):
        shape_bbox = boxes[i]
        for j in range(i + 1, len(shapes)):
            remaining_shape_bbox = boxes[j]
            iou = compute_iou(shape_bbox, remaining_shape_bbox)
            if iou > iou_threshold:
                toBeRemoved.append(j)
    shapesFinal = []
    boxesFinal = []
    confidencesFinal = []
    class_idsFinal = []
    segmentsFinal = []
    for i in range(len(shapes)):
        if i in toBeRemoved:
            continue
        shapesFinal.append(shapes[i])
    boxesFinal, confidencesFinal, class_idsFinal, segmentsFinal = get_boxes_conf_classids_segments(shapesFinal)
    return (shapesFinal, boxesFinal, confidencesFinal, class_idsFinal, segmentsFinal)

