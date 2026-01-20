# Cluster 65

def estimate_curvature_along_path(path: geom.LineString, arc_length: float, distance_for_curvature_estimation: float) -> float:
    """
    Estimate curvature along a path at arc_length from origin.
    :param path: LineString creating a continuous path.
    :param arc_length: [m] distance from origin of the path.
    :param distance_for_curvature_estimation: [m] the distance used to construct 3 points.
    :return estimated curvature at point arc_length.
    """
    assert 0 <= arc_length <= path.length
    if path.length < 2.0 * distance_for_curvature_estimation:
        first_arch_length = 0.0
        second_arc_length = path.length / 2.0
        third_arc_length = path.length
    elif arc_length - distance_for_curvature_estimation < 0.0:
        first_arch_length = 0.0
        second_arc_length = distance_for_curvature_estimation
        third_arc_length = 2.0 * distance_for_curvature_estimation
    elif arc_length + distance_for_curvature_estimation > path.length:
        first_arch_length = path.length - 2.0 * distance_for_curvature_estimation
        second_arc_length = path.length - distance_for_curvature_estimation
        third_arc_length = path.length
    else:
        first_arch_length = arc_length - distance_for_curvature_estimation
        second_arc_length = arc_length
        third_arc_length = arc_length + distance_for_curvature_estimation
    first_arch_position = path.interpolate(first_arch_length)
    second_arch_position = path.interpolate(second_arc_length)
    third_arch_position = path.interpolate(third_arc_length)
    return compute_curvature(first_arch_position, second_arch_position, third_arch_position)

def compute_curvature(point1: geom.Point, point2: geom.Point, point3: geom.Point) -> float:
    """
    Estimate signed curvature along the three points.
    :param point1: First point of a circle.
    :param point2: Second point of a circle.
    :param point3: Third point of a circle.
    :return signed curvature of the three points.
    """
    a = point1.distance(point2)
    b = point2.distance(point3)
    c = point3.distance(point1)
    surface_2 = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    if surface_2 < 1e-06:
        return 0.0
    assert surface_2 >= 0
    k = np.sqrt(surface_2) / 4
    den = a * b * c
    curvature = 4 * k / den if not np.isclose(den, 0.0) else 0.0
    position = np.sign((point2.x - point1.x) * (point3.y - point1.y) - (point2.y - point1.y) * (point3.x - point1.x))
    return float(position * curvature)

class NuPlanPolylineMapObject(PolylineMapObject):
    """
    NuPlanMap implementation of Polyline Map Object.
    """

    def __init__(self, polyline: Series, distance_for_curvature_estimation: float=2.0, distance_for_heading_estimation: float=0.5):
        """
        Constructor of polyline map layer.
        :param polyline: a pandas series representing the polyline.
        :param distance_for_curvature_estimation: [m] distance of the split between 3-points curvature estimation.
        :param distance_for_heading_estimation: [m] distance between two points on the polyline to calculate
                                                    the relative heading.
        """
        super().__init__(polyline['fid'])
        self._polyline: LineString = polyline.geometry
        assert self._polyline.length > 0.0, 'The length of the polyline has to be greater than 0!'
        self._distance_for_curvature_estimation = distance_for_curvature_estimation
        self._distance_for_heading_estimation = distance_for_heading_estimation

    @property
    def linestring(self) -> LineString:
        """Inherited from superclass."""
        return self._polyline

    @property
    def length(self) -> float:
        """Inherited from superclass."""
        return float(self._polyline.length)

    @cached_property
    def discrete_path(self) -> List[StateSE2]:
        """Inherited from superclass."""
        return cast(List[StateSE2], extract_discrete_polyline(self._polyline))

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """Inherited from superclass."""
        return self._polyline.project(Point(point.x, point.y))

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """Inherited from superclass."""
        arc_length = self.get_nearest_arc_length_from_position(point)
        state1 = self._polyline.interpolate(arc_length)
        state2 = self._polyline.interpolate(arc_length + self._distance_for_heading_estimation)
        if state1 == state2:
            state2 = self._polyline.interpolate(arc_length - self._distance_for_heading_estimation)
            heading = _get_heading(state2, state1)
        else:
            heading = _get_heading(state1, state2)
        return StateSE2(state1.x, state1.y, heading)

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """Inherited from superclass."""
        curvature = estimate_curvature_along_path(self._polyline, arc_length, self._distance_for_curvature_estimation)
        return float(curvature)

