# Cluster 10

def plot_2d_box(img, box_2d):
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)

def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]
    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])
    return (pt1, pt2, pt3, pt4)

