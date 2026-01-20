# Cluster 13

def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    def splat_points(point_cloud):
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-x_meters_max, x_meters_max, 32 * pixels_per_meter + 1)
        ybins = np.linspace(-y_meters_max, 0, 32 * pixels_per_meter + 1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat
    below = lidar[lidar[..., 2] <= -2.3]
    above = lidar[lidar[..., 2] > -2.3]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    features = np.rot90(features, -1, axes=(1, 2)).copy()
    return features

def splat_points(point_cloud):
    pixels_per_meter = 8
    hist_max_per_pixel = 5
    x_meters_max = 16
    y_meters_max = 32
    xbins = np.linspace(-x_meters_max, x_meters_max, 32 * pixels_per_meter + 1)
    ybins = np.linspace(-y_meters_max, 0, 32 * pixels_per_meter + 1)
    hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    overhead_splat = hist / hist_max_per_pixel
    return overhead_splat

