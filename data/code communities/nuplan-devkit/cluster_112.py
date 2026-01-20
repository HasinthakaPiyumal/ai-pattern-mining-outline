# Cluster 112

def transform_vector_global_to_local_frame(vector: Tuple[float, float, float], theta: float) -> Tuple[float, float, float]:
    """
    Transform a vector from global frame to local frame.

    :param vector: the vector to be rotated
    :param theta: the amount to rotate by
    :return: the transformed vector.
    """
    return rotate_vector(vector, theta)

def rotate_vector(vector: Tuple[float, float, float], theta: float, inverse: bool=False) -> Tuple[float, float, float]:
    """
    Apply a 2D rotation around the z axis.

    :param vector: the vector to be rotated
    :param theta: the amount to rotate by
    :param inverse: direction of rotation
    :return: the transformed vector.
    """
    assert len(vector) == 3, 'vector to be transformed must have length 3'
    rotation_matrix = R.from_rotvec([0, 0, theta])
    if inverse:
        rotation_matrix = rotation_matrix.inv()
    local_vector = rotation_matrix.apply(vector)
    return cast(Tuple[float, float, float], local_vector.tolist())

def transform_vector_local_to_global_frame(vector: Tuple[float, float, float], theta: float) -> Tuple[float, float, float]:
    """
    Transform a vector from local frame to global frame.

    :param vector: the vector to be rotated
    :param theta: the amount to rotate by
    :return: the transformed vector.
    """
    return rotate_vector(vector, theta, inverse=True)

class IDMPolicyTests(unittest.TestCase):
    """
    Tests IDM utils.
    """

    def setUp(self) -> None:
        """Test setup."""
        self.test_vector = [1, 0, 0]

    def test_convert_global_to_local_frame(self):
        """
        Tests transform_vector_global_to_local_frame.
        """
        result = transform_vector_global_to_local_frame(self.test_vector, np.pi / 2)
        expect: npt.NDArray[np.int_] = np.array([0, 1, 0])
        actual: npt.NDArray[np.float_] = np.array(result)
        self.assertTrue(np.allclose(expect, actual))
        result = transform_vector_global_to_local_frame(self.test_vector, -np.pi / 2)
        expect = np.array([0, -1, 0])
        actual = np.array(result)
        self.assertTrue(np.allclose(expect, actual))

    def test_convert_local_to_global_frame(self):
        """
        Tests transform_vector_local_to_global_frame.
        """
        result = transform_vector_local_to_global_frame(self.test_vector, np.pi / 2)
        expect: npt.NDArray[np.int_] = np.array([0, -1, 0])
        actual: npt.NDArray[np.float_] = np.array(result)
        self.assertTrue(np.allclose(expect, actual))
        result = transform_vector_local_to_global_frame(self.test_vector, -np.pi / 2)
        expect = np.array([0, 1, 0])
        actual = np.array(result)
        self.assertTrue(np.allclose(expect, actual))
    if __name__ == '__main__':
        unittest.main()

