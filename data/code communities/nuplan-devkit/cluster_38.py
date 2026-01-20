# Cluster 38

def union(a: Rectangle, b: Rectangle) -> float:
    """
    Union of two rectangles.
    :param a: Rectangle 1.
    :param b: Rectangle 2.
    :return: Area of union between a and b.
    """
    return (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection(a, b)

def intersection(a: Rectangle, b: Rectangle) -> float:
    """
    Intersection between rectangles.
    :param a: Rectangle 1.
    :param b: Rectangle 2.
    :return: Area of intersection between a and b.
    """
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx >= 0 and dy >= 0:
        return dx * dy
    else:
        return 0

