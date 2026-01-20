# Cluster 13

def OURnms_areaBased_fromSAM(self, sam_result, iou_threshold=0.5):
    iou_threshold = float(iou_threshold)
    sortedResult = sorted(sam_result, key=lambda x: x['area'], reverse=True)
    masks = [mask['segmentation'] for mask in sortedResult]
    scores = [mask['stability_score'] for mask in sortedResult]
    polygons = [mask_to_polygons(mask) for mask in masks]
    toBeRemoved = []
    if iou_threshold > 0.99:
        for i in range(len(polygons)):
            shape1 = polygons[i]
            for j in range(i + 1, len(sortedResult)):
                shape2 = polygons[j]
                iou = compute_iou_exact(shape1, shape2)
                if iou > iou_threshold:
                    toBeRemoved.append(j)
    shapes = []
    for i in range(len(polygons)):
        if i in toBeRemoved:
            continue
        shapes.append(self.polygon_to_shape(polygons[i], scores[i], f'X{i}'))
    return shapes

def mask_to_polygons(mask, n_points=25, resize_factors=[1.0, 1.0]):
    mask = mask > 0.0
    contours = skimage.measure.find_contours(mask)
    if len(contours) == 0:
        return []
    contour = max(contours, key=get_contour_length)
    coords = skimage.measure.approximate_polygon(coords=contour, tolerance=np.ptp(contour, axis=0).max() / 100)
    coords = coords * resize_factors
    coords = np.fliplr(coords)
    segment_points = coords.astype(int)
    polygon = segment_points
    return polygon

def compute_iou_exact(shape1, shape2):
    """
    Summary:
        Computes IOU between two polygons.
    
    Args:
        shape1 (list): List of 2D coordinates(also list) of the first polygon.
        shape2 (list): List of 2D coordinates(also list) of the second polygon.
        
    Returns:
        iou (float): IOU between the two polygons.
    """
    shape1 = [tuple(x) for x in shape1]
    shape2 = [tuple(x) for x in shape2]
    polygon1 = Polygon(shape1)
    polygon2 = Polygon(shape2)
    if polygon1.intersects(polygon2) is False:
        return 0
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersection / union if union > 0 else 0
    return iou

