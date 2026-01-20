# Cluster 21

def nms_rotated(boxes, scores, threshold):
    """Performs rorated non-maximum suppression and returns indices of kept
    boxes.

    Parameters
    ----------
    boxes : torch.tensor
        The location preds with shape (N, 4, 2).

    scores : torch.tensor
        The predicted confidence score with shape (N,)

    threshold: float
        IoU threshold to use for filtering.

    Returns
    -------
        An array of index
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    boxes = boxes.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()
    polygons = common_utils.convert_format(boxes)
    top = 1000
    ixs = scores.argsort()[::-1][:top]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = common_utils.compute_iou(polygons[i], polygons[ixs[1:]])
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

def convert_format(boxes_array):
    """
    Convert boxes array to shapely.geometry.Polygon format.
    Parameters
    ----------
    boxes_array : np.ndarray
        (N, 4, 2) or (N, 8, 3).

    Returns
    -------
        list of converted shapely.geometry.Polygon object.

    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """
    Compute iou between box and boxes list
    Parameters
    ----------
    box : shapely.geometry.Polygon
        Bounding box Polygon.

    boxes : list
        List of shapely.geometry.Polygon.

    Returns
    -------
    iou : np.ndarray
        Array of iou between box and boxes.

    """
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]
    return np.array(iou, dtype=np.float32)

