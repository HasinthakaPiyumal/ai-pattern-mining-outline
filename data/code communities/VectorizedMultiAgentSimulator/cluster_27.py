# Cluster 27

def make_capsule(length, width):
    l, r, t, b = (0, length, width / 2, -width / 2)
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom

def make_polygon(v, filled=True, draw_border: float=True):
    if filled:
        return FilledPolygon(v, draw_border=draw_border)
    else:
        return PolyLine(v, True)

def make_circle(radius=10, res=30, filled=True, angle=2 * math.pi):
    return make_ellipse(radius_x=radius, radius_y=radius, res=res, filled=filled, angle=angle)

