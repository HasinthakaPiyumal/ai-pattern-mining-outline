# Cluster 0

def isclose(p1, p2):
    c_d = dist(p1[2], p2[2])
    if p1[1] < p2[1]:
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]
    T = 0
    try:
        T = (p2[2][1] - p1[2][1]) / (p2[2][0] - p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C * c_d
    d_ver = S * c_d
    vc_calib_hor = a_w * 1.3
    vc_calib_ver = a_h * 0.4 * angle_factor
    c_calib_hor = a_w * 1.7
    c_calib_ver = a_h * 0.2 * angle_factor
    if 0 < d_hor < vc_calib_hor and 0 < d_ver < vc_calib_ver:
        return 1
    elif 0 < d_hor < c_calib_hor and 0 < d_ver < c_calib_ver:
        return 2
    else:
        return 0

