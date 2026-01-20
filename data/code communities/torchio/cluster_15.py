# Cluster 15

def _normalize(tensor: torch.Tensor, landmarks: np.ndarray, mask: np.ndarray | None, cutoff: tuple[float, float] | None=None, epsilon: float=1e-05) -> torch.Tensor:
    cutoff_ = DEFAULT_CUTOFF if cutoff is None else cutoff
    array = tensor.numpy()
    mapping = landmarks
    data = array
    shape = data.shape
    data = data.reshape(-1).astype(np.float32)
    if mask is None:
        mask = np.ones_like(data, bool)
    mask = mask.reshape(-1)
    range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]
    quantiles_cutoff = _standardize_cutoff(cutoff_)
    percentiles_cutoff = 100 * np.array(quantiles_cutoff)
    a, b = percentiles_cutoff
    percentiles = _get_percentiles((a, b))
    percentile_values = np.percentile(data[mask], percentiles)
    range_mapping = mapping[range_to_use]
    range_perc = percentile_values[range_to_use]
    diff_mapping = np.diff(range_mapping)
    diff_perc = np.diff(range_perc)
    diff_perc[diff_perc < epsilon] = np.inf
    affine_map = np.zeros([2, len(range_to_use) - 1])
    affine_map[0] = diff_mapping / diff_perc
    affine_map[1] = range_mapping[:-1] - affine_map[0] * range_perc[:-1]
    bin_id = np.digitize(data, range_perc[1:-1], right=False)
    lin_img = affine_map[0, bin_id]
    aff_img = affine_map[1, bin_id]
    new_img = lin_img * data + aff_img
    new_img = new_img.reshape(shape)
    new_img = new_img.astype(np.float32)
    new_img = torch.as_tensor(new_img)
    return new_img

def _standardize_cutoff(cutoff: Sequence[float]) -> np.ndarray:
    """Standardize the cutoff values given in the configuration.

    Computes percentile landmark normalization by default.
    """
    cutoff_array = np.asarray(cutoff)
    cutoff_array[0] = max(0, cutoff_array[0])
    cutoff_array[1] = min(1, cutoff_array[1])
    cutoff_array[0] = np.min([cutoff_array[0], 0.09])
    cutoff_array[1] = np.max([cutoff_array[1], 0.91])
    return cutoff_array

def _get_percentiles(percentiles_cutoff: tuple[float, float]) -> np.ndarray:
    quartiles = np.arange(25, 100, 25).tolist()
    deciles = np.arange(10, 100, 10).tolist()
    all_percentiles = list(percentiles_cutoff) + quartiles + deciles
    percentiles = sorted(set(all_percentiles))
    return np.array(percentiles)

