# Cluster 6

def getBinCenter(bin_number, NH):
    if NH == 4:
        bin_center = wrapToPi(bin_number * (np.pi / 2))
    else:
        raise Exception('getBinCenter: NH is not 4')
    return bin_center

