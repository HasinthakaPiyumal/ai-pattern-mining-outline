# Cluster 11

def addPoints(shape, n):
    """
    Summary:
        Add points to a polygon.
        
    Args:
        shape: a list of points
        n: number of points to add
        
    Returns:
        res: a list of points
    """
    sub = 1.0 * n / (len(shape) - 1)
    if sub == 0:
        return shape
    if sub < 1:
        res = []
        res.append(shape[0])
        for i in range(len(shape) - 1):
            newPoint = [(shape[i][0] + shape[i + 1][0]) / 2, (shape[i][1] + shape[i + 1][1]) / 2]
            res.append(newPoint)
            res.append(shape[i + 1])
        return handlePoints(res, n + len(shape))
    else:
        toBeAdded = int(sub) + 1
        res = []
        res.append(shape[0])
        for i in range(len(shape) - 1):
            dif = [shape[i + 1][0] - shape[i][0], shape[i + 1][1] - shape[i][1]]
            for j in range(1, toBeAdded):
                newPoint = [shape[i][0] + dif[0] * j / toBeAdded, shape[i][1] + dif[1] * j / toBeAdded]
                res.append(newPoint)
            res.append(shape[i + 1])
        return addPoints(res, n + len(shape) - len(res))

def reducePoints(polygon, n):
    """
    Summary:
        Remove points from a polygon.
        
    Args:
        polygon: a list of points
        n: number of points to reduce to
        
    Returns:
        polygon: a list of points
    """
    if n >= len(polygon):
        return polygon
    distances = polygon.copy()
    for i in range(len(polygon)):
        x1, y1, x2, y2 = (polygon[i - 1][0], polygon[i - 1][1], polygon[(i + 1) % len(polygon)][0], polygon[(i + 1) % len(polygon)][1])
        x, y = (polygon[i][0], polygon[i][1])
        if x1 == x2:
            dist_perp = abs(x - x1)
        elif y1 == y2:
            dist_perp = abs(y - y1)
        else:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            dist_perp = abs(m * x - y + c) / np.sqrt(m * m + 1)
        dif_right = np.array(polygon[(i + 1) % len(polygon)]) - np.array(polygon[i])
        dist_right = np.sqrt(dif_right[0] * dif_right[0] + dif_right[1] * dif_right[1])
        dif_left = np.array(polygon[i - 1]) - np.array(polygon[i])
        dist_left = np.sqrt(dif_left[0] * dif_left[0] + dif_left[1] * dif_left[1])
        distances[i] = min(dist_perp, dist_right, dist_left)
    distances = [distances[i] + random.random() for i in range(len(distances))]
    ratio = 1.0 * n / len(polygon)
    threshold = np.percentile(distances, 100 - ratio * 100)
    i = 0
    while i < len(polygon):
        if distances[i] < threshold:
            polygon[i] = None
            i += 1
        i += 1
    res = [x for x in polygon if x is not None]
    return reducePoints(res, n)

def handlePoints(polygon, n):
    """
    Summary:
        Add or remove points from a polygon.
        
    Args:
        polygon: a list of points
        n: number of points that the polygon should have
        
    Returns:
        polygon: a list of points
    """
    if n == len(polygon):
        return polygon
    elif n > len(polygon):
        return addPoints(polygon, n - len(polygon))
    else:
        return reducePoints(polygon, n)

def handleTwoSegments(segment1, segment2):
    """
    Summary:
        Add or remove points from two polygons to make them have the same number of points.
        
    Args:
        segment1: a list of points
        segment2: a list of points
        
    Returns:
        segment1: a list of points
        segment2: a list of points
    """
    if len(segment1) != len(segment2):
        biglen = max(len(segment1), len(segment2))
        segment1 = handlePoints(segment1, biglen)
        segment2 = handlePoints(segment2, biglen)
    segment1, segment2 = allign(segment1, segment2)
    return (segment1, segment2)

def getInterpolated(baseObject, baseObjectFrame, nextObject, nextObjectFrame, curFrame):
    """
    Summary:
        Interpolate a shape between two frames using linear interpolation.
        
    Args:
        baseObject: the base object
        baseObjectFrame: the base object frame
        nextObject: the next object
        nextObjectFrame: the next object frame
        curFrame: the frame to interpolate
        
    Returns:
        cur: the interpolated shape
    """
    prvR = (nextObjectFrame - curFrame) / (nextObjectFrame - baseObjectFrame)
    nxtR = (curFrame - baseObjectFrame) / (nextObjectFrame - baseObjectFrame)
    cur_bbox = prvR * np.array(baseObject['bbox']) + nxtR * np.array(nextObject['bbox'])
    cur_bbox = [int(cur_bbox[i]) for i in range(len(cur_bbox))]
    baseObject['segment'], nextObject['segment'] = handleTwoSegments(baseObject['segment'], nextObject['segment'])
    cur_segment = prvR * np.array(baseObject['segment']) + nxtR * np.array(nextObject['segment'])
    cur_segment = [[int(sublist[0]), int(sublist[1])] for sublist in cur_segment]
    cur = copy.deepcopy(baseObject)
    cur['bbox'] = cur_bbox
    cur['segment'] = cur_segment
    return cur

