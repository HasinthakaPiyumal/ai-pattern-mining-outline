# Cluster 19

def _matrix_to_itk_transform(matrix: TypeData, dimensions: int=3) -> sitk.AffineTransform:
    matrix = _to_itk_convention(matrix)
    rotation = matrix[:dimensions, :dimensions].ravel().tolist()
    translation = matrix[:dimensions, 3].tolist()
    transform = sitk.AffineTransform(rotation, translation)
    return transform

def _to_itk_convention(matrix: TypeData) -> np.ndarray:
    """RAS to LPS."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
    matrix = np.dot(FLIPXY_44, matrix)
    matrix = np.dot(matrix, FLIPXY_44)
    matrix = np.linalg.inv(matrix)
    return matrix

