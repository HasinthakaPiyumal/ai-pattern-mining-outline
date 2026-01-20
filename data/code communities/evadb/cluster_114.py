# Cluster 114

class NorFairTracker(EvaDBTrackerAbstractFunction):

    @property
    def name(self) -> str:
        return 'NorFairTracker'

    def setup(self, distance_threshold=DISTANCE_THRESHOLD_CENTROID) -> None:
        try_to_import_norfair()
        from norfair import Tracker
        self.tracker = Tracker(distance_function='euclidean', distance_threshold=distance_threshold)
        self.prev_frame_id = None

    def forward(self, frame_id, frame, labels, bboxes, scores):
        from norfair import Detection
        norfair_detections = [Detection(points=get_centroid(bbox), scores=np.array([score]), label=hash(label) % 10 ** 8, data=(label, bbox, score)) for bbox, score, label in zip(bboxes, scores, labels)]
        period = frame_id - self.prev_frame_id if self.prev_frame_id else 1
        self.prev_frame_id = frame_id
        tracked_objects = self.tracker.update(detections=norfair_detections, period=period)
        bboxes_xyxy = []
        labels = []
        scores = []
        ids = []
        for obj in tracked_objects:
            det = obj.last_detection.data
            labels.append(det[0])
            bboxes_xyxy.append(det[1])
            scores.append(det[2])
            ids.append(obj.id)
        return (np.array(ids), np.array(labels), np.array(bboxes_xyxy), np.array(scores))

def get_centroid(bboxes):
    """
    https://adipandas.github.io/multi-object-tracker/_modules/motrackers/utils/misc.html
    Calculate centroids for multiple bounding boxes.

    Args:
        bboxes (numpy.ndarray): Array of shape `(n, 4)` or of shape `(4,)` where
            each row contains `(xmin, ymin, width, height)`.

    Returns:
        numpy.ndarray: Centroid (x, y) coordinates of shape `(n, 2)` or `(2,)`.

    """
    one_bbox = False
    if len(bboxes.shape) == 1:
        one_bbox = True
        bboxes = bboxes[None, :]
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    w, h = (bboxes[:, 2], bboxes[:, 3])
    xc = xmin + 0.5 * w
    yc = ymin + 0.5 * h
    x = np.hstack([xc[:, None], yc[:, None]])
    if one_bbox:
        x = x.flatten()
    return x

