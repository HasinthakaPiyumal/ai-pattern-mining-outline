# Cluster 7

def wrapToPi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def getBinCenters(bin_numbers, NH):
    if NH == 4:
        bin_centers = wrapToPi(bin_numbers * (np.pi / 2))
    else:
        raise Exception('getBinCenters: NH is not 4')
    return bin_centers

def getBinCenter(bin_number, NH):
    if NH == 4:
        bin_center = wrapToPi(bin_number * (np.pi / 2))
    else:
        raise Exception('getBinCenter: NH is not 4')
    return bin_center

def getBinCenters(bin_numbers, NH):
    if NH == 4:
        bin_centers = wrapToPi(bin_numbers * (np.pi / 2))
    else:
        raise Exception('getBinCenters: NH is not 4')
    return bin_centers

