# Cluster 78

class TestAngularInterpolator(unittest.TestCase):
    """Tests AngularInterpolator class"""

    @patch('nuplan.common.geometry.compute.interp1d', autospec=True)
    def setUp(self, mock_interp: Mock) -> None:
        """Sets up variables for testing"""
        interpolator = Mock(return_value='interpolated')
        mock_interp.return_value = interpolator
        self.states: npt.NDArray[np.float64] = np.array([1, 2, 3, 4, 5])
        self.angular_states = [[11], [22], [33], [44]]
        self.interpolator = AngularInterpolator(self.states, self.angular_states)

    @patch('nuplan.common.geometry.compute.np.unwrap', autospec=True)
    @patch('nuplan.common.geometry.compute.interp1d', autospec=True)
    def test_initialization(self, mock_interp: Mock, unwrap: Mock) -> None:
        """Tests interpolation for angular states."""
        interpolator = AngularInterpolator(self.states, self.angular_states)
        unwrap.assert_called_with(self.angular_states, axis=0)
        self.assertEqual(mock_interp.return_value, interpolator.interpolator)

    @patch('nuplan.common.geometry.compute.principal_value')
    def test_interpolate(self, principal_value: Mock) -> None:
        """Interpolates single state"""
        state = 1.5
        principal_value.return_value = 1.23
        result = self.interpolator.interpolate(state)
        self.interpolator.interpolator.assert_called_once_with(state)
        principal_value.assert_called_once_with('interpolated')
        self.assertEqual(1.23, result)

    def test_interpolate_real_value(self) -> None:
        """Interpolates multiple state"""
        states: npt.NDArray[np.float64] = np.array([1, 3])
        angular_states = [[3.0, -2.0], [-3.0, 2.0]]
        interpolator = AngularInterpolator(states, angular_states)
        np.testing.assert_allclose(np.array([-np.pi, -np.pi]), interpolator.interpolate(2))

