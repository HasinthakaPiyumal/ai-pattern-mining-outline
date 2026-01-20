# Cluster 40

class TestMinimumBoundingRectangle(unittest.TestCase):
    """Tests for the minimum_bounding_rectangle() methods."""

    def check_minimum_bounding_rectangle(self, rect_points: npt.NDArray[np.float64], points_to_check: List[List[int]]) -> None:
        """
        Given the points of the minimum rectangle and the points to check, this function checks whether each point
        in points_to_check lies in rect_points.
        :param rect_points: The points of the minimum rectangle.
        :param points_to_check: Points to check if they lie in the minimum rectangle.
        """
        self.assertTrue(rect_points.shape == (4, 2))
        rect_points = np.around(rect_points, decimals=3)
        for point in points_to_check:
            self.assertTrue(np.equal(rect_points, np.around(point, decimals=3)).all(1).any())

    def test_all_square_vertices(self) -> None:
        """
        Use the vertices of a square as the input points. The minimum bounding rectangle for them would be the same
        square.
        """
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_all_rectangle_vertices(self) -> None:
        """
        Use the vertices of a rectangle as the input points. The minimum bounding rectangle for them would be the
        complete rectangle.
        """
        points = np.array([[0, 0], [2, 1], [2, 0], [0, 1]])
        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [2, 0], [0, 1], [2, 1]])

    def test_three_square_vertices(self) -> None:
        """
        Use the three vertices of a square as the input points. The minimum bounding rectangle for them would be the
        complete square.
        """
        points = np.array([[0, 0], [1, 1], [0, 1]])
        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.array([[1, 0], [1, 1], [0, 1]])
        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.array([[0, 0], [1, 1], [1, 0]])
        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_lots_of_random_points_in_a_square(self) -> None:
        """
        Use the three vertices of a square as the input points. Then concatenate a bunch of random points inside the
        square to those points. The minimum bounding rectangle for them would be the original square.
        """
        points = np.array([[0, 0], [1, 1], [0, 1]])
        pts_inside_square = np.random.rand(30, 2)
        points = np.concatenate([points, pts_inside_square])
        rect_points = minimum_bounding_rectangle(points)
        self.check_minimum_bounding_rectangle(rect_points, [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_lots_of_random_points_in_a_rotated_square(self) -> None:
        """
        Use the four vertices of a square as the input points. Then concatenate a bunch of random points inside the
        square to those points. Finally rotate all the points by a fixed angle. The minimum bounding rectangle for them
        would be the original square rotated by the same angle chosen in the last step.
        """
        points = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        pts_inside_square = np.random.rand(30, 2)
        points = np.concatenate([points, pts_inside_square])
        rand_angle = np.random.randn()
        rot_mat = np.array([[np.cos(rand_angle), np.sin(rand_angle)], [-np.sin(rand_angle), np.cos(rand_angle)]])
        rect_points = minimum_bounding_rectangle(np.dot(rot_mat, points.T).T)
        self.check_minimum_bounding_rectangle(rect_points, np.dot(rot_mat, np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T).T)

def minimum_bounding_rectangle(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Finds the smallest bounding rectangle for a set of points in two dimensional space.
    Returns a set of points (in clockwise order) representing the corners of the bounding box.

    Algorithm high level idea:
        One edge of the minimum bounding rectangle for a set of points will be the same as one of the edges of the
        convex hull of those points.

    Algorithm:
     1. Create a convex hull (https://en.wikipedia.org/wiki/Convex_hull) of the input points.
     2. Calculate the angles that all the edges of the convex hull make with the x-axis. Assume that there are N unique
        angles calculated in this step.
     3. Create rotation matrices for all the N unique angles computed in step 2.
     4. Create N set of convex hull points by rotating the original convex hull points using all the N rotation matrices
        computed in the last step.
     5. For each of the N set of convex hull points computed in the last step, calculate the bounding rectangle by
        calculating (min_x, max_x, min_y, max_y).
     6. For the N bounding rectangles computed in the last step, find the rectangle with the minimum area. This will
        give the minimum bounding rectangle for our rotated set of convex hull points (see Step 4).
     7. Undo the rotation of the convex hull by multiplying the points with the inverse of the rotation matrix. And
        remember that the inverse of a rotation matrix is equal to the transpose of the rotation matrix. The returned
        points are in a clockwise order.

    To visualize what this function does, you can use the following snippet:

    for n in range(10):
        points = np.random.rand(8,2)
        plt.scatter(points[:,0], points[:,1])
        bbox = minimum_bounding_rectangle(points)
        plt.fill(bbox[:,0], bbox[:,1], alpha=0.2)
        plt.axis('equal')
        plt.show()

    :param points: <nbr_points, 2>. A nx2 matrix of coordinates where n >= 3.
    :return: A 4x2 matrix of coordinates of the minimum bounding rectangle (in clockwise order).
    """
    assert points.ndim == 2, 'Points ndim should be 2.'
    assert points.shape[1] == 2, 'Points shape: n x 2 where n>= 3.'
    assert points.shape[0] >= 3, 'Points shape: n x 2 where n>= 3.'
    pi2 = np.pi / 2.0
    hull_points = points[ConvexHull(points).vertices]
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)
    rotations = np.vstack([np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))
    rot_points = np.dot(rotations, hull_points.T)
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]
    pts_clockwise_order = np.zeros((4, 2))
    pts_clockwise_order[0] = np.dot([x1, y2], r)
    pts_clockwise_order[1] = np.dot([x2, y2], r)
    pts_clockwise_order[2] = np.dot([x2, y1], r)
    pts_clockwise_order[3] = np.dot([x1, y1], r)
    return pts_clockwise_order

