# Cluster 18

def _read_itk_matrix(path: TypePath) -> torch.Tensor:
    """Read an affine transform in ITK's .tfm format."""
    transform = sitk.ReadTransform(str(path))
    parameters = transform.GetParameters()
    rotation_parameters = parameters[:9]
    rotation_matrix = np.array(rotation_parameters).reshape(3, 3)
    translation_parameters = parameters[9:]
    translation_vector = np.array(translation_parameters).reshape(3, 1)
    matrix = np.hstack([rotation_matrix, translation_vector])
    homogeneous_matrix_lps = np.vstack([matrix, [0, 0, 0, 1]])
    homogeneous_matrix_ras = _from_itk_convention(homogeneous_matrix_lps)
    return torch.as_tensor(homogeneous_matrix_ras)

def _from_itk_convention(matrix: TypeData) -> np.ndarray:
    """LPS to RAS."""
    matrix = np.dot(matrix, FLIPXY_44)
    matrix = np.dot(FLIPXY_44, matrix)
    matrix = np.linalg.inv(matrix)
    return matrix

