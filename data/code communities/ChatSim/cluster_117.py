# Cluster 117

def compute_bezier_points(p0, p1, p2, p3, num_points=100):
    return np.array([cubic_bezier(p0, p1, p2, p3, t) for t in np.linspace(0, 1, num_points)])

def cubic_bezier(p0, p1, p2, p3, t):
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

