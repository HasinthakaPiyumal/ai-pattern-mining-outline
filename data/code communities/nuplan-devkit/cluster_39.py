# Cluster 39

class TestTransformMatrix(unittest.TestCase):
    """Test TransformMatrix."""

    def test_transform_matrix(self) -> None:
        """Test transform matrix using translation and rotation."""
        zero_rotation = Quaternion(axis=(0.0, 0.0, 1.0), angle=0.0)
        for _ in range(100):
            x_trans = random.uniform(-100.0, 100.0)
            y_trans = random.uniform(-100.0, 100.0)
            z_trans = random.uniform(-100.0, 100.0)
            translation = np.array([x_trans, y_trans, z_trans])
            tm = transform_matrix(translation, zero_rotation, False)
            tm_test = np.eye(4)
            tm_test[0:3, 3] = translation
            assert_array_almost_equal(tm, tm_test)
        zero_translation = np.array([0.0, 0.0, 0.0])
        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        z_axis = (0.0, 0.0, 1.0)
        for axis_idx, axis in enumerate([x_axis, y_axis, z_axis]):
            for theta in np.linspace(-4.0 * np.pi, 4.0 * np.pi, 100):
                rotation = Quaternion(axis=axis, angle=theta)
                tm = transform_matrix(zero_translation, rotation, False)
                tm_test = np.eye(4)
                tm_test[(axis_idx + 1) % 3, (axis_idx + 1) % 3] = np.cos(theta)
                tm_test[(axis_idx + 1) % 3, (axis_idx + 2) % 3] = -np.sin(theta)
                tm_test[(axis_idx + 2) % 3, (axis_idx + 1) % 3] = np.sin(theta)
                tm_test[(axis_idx + 2) % 3, (axis_idx + 2) % 3] = np.cos(theta)
                assert_array_almost_equal(tm, tm_test)
        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        z_axis = (0.0, 0.0, 1.0)
        for axis_idx, axis in enumerate([x_axis, y_axis, z_axis]):
            for theta in np.linspace(-4.0 * np.pi, 4.0 * np.pi, 100):
                x_trans = random.uniform(-100.0, 100.0)
                y_trans = random.uniform(-100.0, 100.0)
                z_trans = random.uniform(-100.0, 100.0)
                translation = np.array([x_trans, y_trans, z_trans])
                rotation = Quaternion(axis=axis, angle=theta)
                tm = transform_matrix(translation, rotation, False)
                tm_test = np.eye(4)
                tm_test[(axis_idx + 1) % 3, (axis_idx + 1) % 3] = np.cos(theta)
                tm_test[(axis_idx + 1) % 3, (axis_idx + 2) % 3] = -np.sin(theta)
                tm_test[(axis_idx + 2) % 3, (axis_idx + 1) % 3] = np.sin(theta)
                tm_test[(axis_idx + 2) % 3, (axis_idx + 2) % 3] = np.cos(theta)
                tm_test[0:3, 3] = translation
                assert_array_almost_equal(tm, tm_test)
        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        z_axis = (0.0, 0.0, 1.0)
        for axis_idx, axis in enumerate([x_axis, y_axis, z_axis]):
            for theta in np.linspace(-4.0 * np.pi, 4.0 * np.pi, 100):
                x_trans = random.uniform(-100.0, 100.0)
                y_trans = random.uniform(-100.0, 100.0)
                z_trans = random.uniform(-100.0, 100.0)
                translation = np.array([x_trans, y_trans, z_trans])
                rotation = Quaternion(axis=axis, angle=theta)
                tm = transform_matrix(translation, rotation, False)
                inverse_tm = transform_matrix(translation, rotation, True)
                assert_array_almost_equal(inverse_tm, np.linalg.inv(tm))
        zero_rotation = Quaternion(axis=(0.0, 0.0, 1.0), angle=0.0)
        for _ in range(100):
            x_trans1 = random.uniform(-100.0, 100.0)
            y_trans1 = random.uniform(-100.0, 100.0)
            z_trans1 = random.uniform(-100.0, 100.0)
            translation1 = np.array([x_trans1, y_trans1, z_trans1])
            tm1 = transform_matrix(translation1, zero_rotation, False)
            x_trans2 = random.uniform(-100.0, 100.0)
            y_trans2 = random.uniform(-100.0, 100.0)
            z_trans2 = random.uniform(-100.0, 100.0)
            translation2 = np.array([x_trans2, y_trans2, z_trans2])
            tm2 = transform_matrix(translation2, zero_rotation, False)
            assert_array_almost_equal(tm1 * tm2, tm2 * tm1)
        zero_translation = np.array([0.0, 0.0, 0.0])
        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        z_axis = (0.0, 0.0, 1.0)
        for _ in range(100):
            axis1 = random.choice([x_axis, y_axis, z_axis])
            theta1 = random.uniform(-4.0 * np.pi, 4.0 * np.pi)
            rotation1 = Quaternion(axis=axis1, angle=theta1)
            tm1 = transform_matrix(zero_translation, rotation1, False)
            axis2 = random.choice([x_axis, y_axis, z_axis])
            theta2 = random.uniform(-4.0 * np.pi, 4.0 * np.pi)
            rotation2 = Quaternion(axis=axis2, angle=theta2)
            tm2 = transform_matrix(zero_translation, rotation2, False)
            assert_array_almost_equal(tm1 * tm2, tm2 * tm1)

def transform_matrix(translation: npt.NDArray[np.float64]=np.array([0, 0, 0]), rotation: Quaternion=Quaternion([1, 0, 0, 0]), inverse: bool=False) -> npt.NDArray[np.float64]:
    """
    Converts pose to transform matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w, ri, rj, rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)
    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm

