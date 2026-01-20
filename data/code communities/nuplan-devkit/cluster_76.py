# Cluster 76

class AngularInterpolator:
    """Creates an angular linear interpolator."""

    def __init__(self, states: npt.NDArray[np.float64], angular_states: npt.NDArray[np.float64]):
        """
        :param states: x values for interpolation
        :param angular_states: y values for interpolation
        """
        _angular_states = np.unwrap(angular_states, axis=0)
        self.interpolator = interp1d(states, _angular_states, axis=0)

    def interpolate(self, sampled_state: Union[float, List[float]]) -> npt.NDArray[np.float64]:
        """
        Interpolates a single state
        :param sampled_state: The state at which to perform interpolation
        :return: The value of the state interpolating linearly at the given state
        """
        return principal_value(self.interpolator(sampled_state))

def principal_value(angle: Union[float, int, npt.NDArray[np.float64]], min_: float=-np.pi) -> Union[float, npt.NDArray[np.float64]]:
    """
    Wrap heading angle in to specified domain (multiples of 2 pi alias),
    ensuring that the angle is between min_ and min_ + 2 pi. This function raises an error if the angle is infinite
    :param angle: rad
    :param min_: minimum domain for angle (rad)
    :return angle wrapped to [min_, min_ + 2 pi).
    """
    assert np.all(np.isfinite(angle)), 'angle is not finite'
    lhs = (angle - min_) % (2 * np.pi) + min_
    return lhs

class TestCompute(unittest.TestCase):
    """Tests for compute functions"""

    @patch('nuplan.common.geometry.compute.get_pacifica_parameters', autospec=True)
    def test_signed_lateral_distance(self, mock_pacifica: Mock) -> None:
        """Tests signed lateral distance of ego to polygon"""
        mock_pacifica.return_value = Mock(half_width=1)
        result_0 = signed_lateral_distance(StateSE2(1, 1, -np.pi / 2), Polygon(((3, 2), (4, 3), (6, 1), (5, 0))))
        result_1 = signed_lateral_distance(StateSE2(1, 1, np.pi / 2), Polygon(((3, 2), (4, 3), (6, 1), (5, 0))))
        self.assertAlmostEqual(result_0, 1)
        self.assertAlmostEqual(result_1, -1)

    @patch('nuplan.common.geometry.compute.get_pacifica_parameters', autospec=True)
    def test_signed_longitudinal_distance(self, mock_pacifica: Mock) -> None:
        """Tests signed longitudinal distance of ego to polygon"""
        mock_pacifica.return_value = Mock(half_length=1)
        result_0 = signed_longitudinal_distance(StateSE2(1, 1, 0), Polygon(((3, 2), (4, 3), (6, 1), (5, 0))))
        result_1 = signed_longitudinal_distance(StateSE2(1, 1, np.pi), Polygon(((3, 2), (4, 3), (6, 1), (5, 0))))
        self.assertAlmostEqual(result_0, 1)
        self.assertAlmostEqual(result_1, -1)

    def test_compute_distance(self) -> None:
        """Tests distance between two points"""
        point_0 = StateSE2(8, 8, np.pi)
        point_1 = StateSE2(4, 5, 0)
        result_0 = compute_distance(point_0, point_1)
        result_1 = compute_distance(point_1, point_0)
        self.assertEqual(result_0, 5)
        self.assertEqual(result_1, 5)

    def test_compute_lateral_displacements(self) -> None:
        """Tests lateral distance between a list of points"""
        state_0 = StateSE2(0, 0, 0)
        state_1 = StateSE2(0, 1, 0)
        state_2 = StateSE2(0, 2, 0)
        state_3 = StateSE2(0, 3, 0)
        result = compute_lateral_displacements([state_0, state_1, state_2, state_3])
        for i in range(3):
            self.assertEqual(result[i], 1)

    def test_principal_value(self) -> None:
        """Tests principal angle calculation"""
        values: npt.NDArray[np.float64] = np.array([0, np.pi, 2 * np.pi, 3 * np.pi, -4 * np.pi, -3 * np.pi])
        expected_wrapped_0_to_pi: npt.NDArray[np.float64] = np.array([0, np.pi, 0, np.pi, 0, np.pi])
        expected_wrapped_neg_pi_to_pi: npt.NDArray[np.float64] = np.array([0, -np.pi, 0, -np.pi, 0, -np.pi])
        actual_wrapped_0_to_pi = principal_value(values, min_=0)
        actual_wrapped_neg_pi_to_pi = principal_value(values)
        np.testing.assert_allclose(expected_wrapped_0_to_pi, actual_wrapped_0_to_pi)
        np.testing.assert_allclose(expected_wrapped_neg_pi_to_pi, actual_wrapped_neg_pi_to_pi)

    def test_l2_euclidean_corners_distance(self) -> None:
        """Tests computation of distances between"""
        box_dimension = Dimension(4, 3, 1)
        box1 = OrientedBox(StateSE2(0, 0, 0), box_dimension.length, box_dimension.width, box_dimension.height)
        box2 = OrientedBox(StateSE2(2, 0, 0), box_dimension.length, box_dimension.width, box_dimension.height)
        box3 = OrientedBox(StateSE2(0, 2, 0), box_dimension.length, box_dimension.width, box_dimension.height)
        box4 = OrientedBox(StateSE2(3, 4, 0), box_dimension.length, box_dimension.width, box_dimension.height)
        box1_rot = OrientedBox(StateSE2(0, 0, np.pi), box_dimension.length, box_dimension.width, box_dimension.height)
        box5 = OrientedBox(StateSE2(1, 2, 3), box_dimension.length, box_dimension.width, box_dimension.height)
        self.assertEqual(0, l2_euclidean_corners_distance(box1, box1))
        self.assertEqual(4.0, l2_euclidean_corners_distance(box1, box2))
        self.assertEqual(l2_euclidean_corners_distance(box1, box2), l2_euclidean_corners_distance(box1, box3))
        self.assertEqual(10.0, l2_euclidean_corners_distance(box1, box4))
        self.assertEqual(10.0, l2_euclidean_corners_distance(box1, box1_rot))
        self.assertTrue(math.isclose(10.931588394648887, l2_euclidean_corners_distance(box1, box5)))

    def test_se2_box_distances(self) -> None:
        """Tests computation of distances between SE2 poses using OrientedBox"""
        box_dimension = Dimension(4, 3, 1)
        query = StateSE2(0, 0, 0)
        targets = [StateSE2(0, 0, 0), StateSE2(0, 0, np.pi), StateSE2(2, 0, 0)]
        self.assertEqual([0, 0, 4.0], se2_box_distances(query, targets, box_dimension))
        self.assertEqual([0, 10.0, 4.0], se2_box_distances(query, targets, box_dimension, consider_flipped=False))

class TestUtils(unittest.TestCase):
    """
    Tests utils library.
    """

    def setUp(self) -> None:
        """Sets sample parameters for testing."""
        np.random.seed(0)
        self.inits = np.random.rand(100)
        self.deltas = np.random.rand(100)
        self.sampling_times = np.random.randint(1000000, size=100)

    def test_forward_integrate(self) -> None:
        """
        Test forward_integrate.
        """
        for init, delta, sampling_time in zip(self.inits, self.deltas, self.sampling_times):
            result = forward_integrate(init, delta, TimePoint(sampling_time))
            expect = init + delta * sampling_time * 1e-06
            self.assertAlmostEqual(result, expect)

def forward_integrate(init: float, delta: float, sampling_time: TimePoint) -> float:
    """
    Performs a simple euler integration.
    :param init: Initial state
    :param delta: The rate of chance of the state.
    :param sampling_time: The time duration to propagate for.
    :return: The result of integration
    """
    return float(init + delta * sampling_time.time_s)

def _get_xy_heading_displacements_from_poses(poses: DoubleMatrix) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Returns position and heading displacements given a pose trajectory.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of poses (x, y, heading).
    :return: Tuple of xy displacements with shape (num_poses-1, 2) and heading displacements with shape (num_poses-1,).
    """
    assert len(poses.shape) == 2, 'Expect a 2D matrix representing a trajectory of poses.'
    assert poses.shape[0] > 1, 'Cannot get displacements given an empty or single element pose trajectory.'
    assert poses.shape[1] == 3, 'Expect pose to have three elements (x, y, heading).'
    pose_differences = np.diff(poses, axis=0)
    xy_displacements = pose_differences[:, :2]
    heading_displacements = principal_value(pose_differences[:, 2])
    return (xy_displacements, heading_displacements)

def _fit_initial_velocity_and_acceleration_profile(xy_displacements: DoubleMatrix, heading_profile: DoubleMatrix, discretization_time: float, jerk_penalty: float) -> Tuple[float, DoubleMatrix]:
    """
    Estimates initial velocity (v_0) and acceleration ({a_0, ...}) using least squares with jerk penalty regularization.
    :param xy_displacements: [m] Deviations in x and y occurring between M+1 poses, a M by 2 matrix.
    :param heading_profile: [rad] Headings associated to the starting timestamp for xy_displacements, a M-length vector.
    :param discretization_time: [s] Time discretization used for integration.
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :return: Least squares solution for initial velocity (v_0) and acceleration profile ({a_0, ..., a_M-1})
             for M displacement values.
    """
    assert discretization_time > 0.0, 'Discretization time must be positive.'
    assert jerk_penalty > 0, 'Should have a positive jerk_penalty.'
    assert len(xy_displacements.shape) == 2, 'Expect xy_displacements to be a matrix.'
    assert xy_displacements.shape[1] == 2, 'Expect xy_displacements to have 2 columns.'
    num_displacements = len(xy_displacements)
    assert heading_profile.shape == (num_displacements,), 'Expect the length of heading_profile to match that of xy_displacements.'
    y = xy_displacements.flatten()
    A: DoubleMatrix = np.zeros((2 * num_displacements, num_displacements), dtype=np.float64)
    for idx_timestep, heading in enumerate(heading_profile):
        start_row = 2 * idx_timestep
        A[start_row:start_row + 2, 0] = np.array([np.cos(heading) * discretization_time, np.sin(heading) * discretization_time], dtype=np.float64)
        if idx_timestep > 0:
            A[start_row:start_row + 2, 1:1 + idx_timestep] = np.array([[np.cos(heading) * discretization_time ** 2], [np.sin(heading) * discretization_time ** 2]], dtype=np.float64)
    banded_matrix = _make_banded_difference_matrix(num_displacements - 2)
    R: DoubleMatrix = np.block([np.zeros((len(banded_matrix), 1)), banded_matrix])
    x = np.linalg.pinv(A.T @ A + jerk_penalty * R.T @ R) @ A.T @ y
    initial_velocity = x[0]
    acceleration_profile = x[1:]
    return (initial_velocity, acceleration_profile)

def _make_banded_difference_matrix(number_rows: int) -> DoubleMatrix:
    """
    Returns a banded difference matrix with specified number_rows.
    When applied to a vector [x_1, ..., x_N], it returns [x_2 - x_1, ..., x_N - x_{N-1}].
    :param number_rows: The row dimension of the banded difference matrix (e.g. N-1 in the example above).
    :return: A banded difference matrix with shape (number_rows, number_rows+1).
    """
    banded_matrix: DoubleMatrix = -1.0 * np.eye(number_rows + 1, dtype=np.float64)[:-1, :]
    for ind in range(len(banded_matrix)):
        banded_matrix[ind, ind + 1] = 1.0
    return banded_matrix

def compute_steering_angle_feedback(pose_reference: DoubleMatrix, pose_current: DoubleMatrix, lookahead_distance: float, k_lateral_error: float) -> float:
    """
    Given pose information, determines the steering angle feedback value to address initial tracking error.
    This is based on the feedback controller developed in Section 2.2 of the following paper:
    https://ddl.stanford.edu/publications/design-feedback-feedforward-steering-controller-accurate-path-tracking-and-stability
    :param pose_reference: <np.ndarray: 3,> Contains the reference pose at the current timestep.
    :param pose_current: <np.ndarray: 3,> Contains the actual pose at the current timestep.
    :param lookahead_distance: [m] Distance ahead for which we should estimate lateral error based on a linear fit.
    :param k_lateral_error: Feedback gain for lateral error used to determine steering angle feedback.
    :return: [rad] The steering angle feedback to apply.
    """
    assert pose_reference.shape == (3,), 'We expect a single reference pose.'
    assert pose_current.shape == (3,), 'We expect a single current pose.'
    assert lookahead_distance > 0.0, 'Lookahead distance should be positive.'
    assert k_lateral_error > 0.0, 'Feedback gain for lateral error should be positive.'
    x_reference, y_reference, heading_reference = pose_reference
    x_current, y_current, heading_current = pose_current
    x_error = x_current - x_reference
    y_error = y_current - y_reference
    heading_error = principal_value(heading_current - heading_reference)
    lateral_error = -x_error * np.sin(heading_reference) + y_error * np.cos(heading_reference)
    return float(-k_lateral_error * (lateral_error + lookahead_distance * heading_error))

def get_velocity_curvature_profiles_with_derivatives_from_poses(discretization_time: float, poses: DoubleMatrix, jerk_penalty: float, curvature_rate_penalty: float) -> Tuple[DoubleMatrix, DoubleMatrix, DoubleMatrix, DoubleMatrix]:
    """
    Main function for joint estimation of velocity, acceleration, curvature, and curvature rate given N poses
    sampled at discretization_time.  This is done by solving two least squares problems with the given penalty weights.
    :param discretization_time: [s] Time discretization used for integration.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of N poses (x, y, heading).
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :return: Profiles for velocity (N-1), acceleration (N-2), curvature (N-1), and curvature rate (N-2).
    """
    xy_displacements, heading_displacements = _get_xy_heading_displacements_from_poses(poses)
    initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(xy_displacements=xy_displacements, heading_profile=poses[:-1, 2], discretization_time=discretization_time, jerk_penalty=jerk_penalty)
    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(initial_condition=initial_velocity, derivatives=acceleration_profile, discretization_time=discretization_time)
    initial_curvature, curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(heading_displacements=heading_displacements, velocity_profile=velocity_profile, discretization_time=discretization_time, curvature_rate_penalty=curvature_rate_penalty)
    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(initial_condition=initial_curvature, derivatives=curvature_rate_profile, discretization_time=discretization_time)
    return (velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile)

def _generate_profile_from_initial_condition_and_derivatives(initial_condition: float, derivatives: DoubleMatrix, discretization_time: float) -> DoubleMatrix:
    """
    Returns the corresponding profile (i.e. trajectory) given an initial condition and derivatives at
    multiple timesteps by integration.
    :param initial_condition: The value of the variable at the initial timestep.
    :param derivatives: The trajectory of time derivatives of the variable at timesteps 0,..., N-1.
    :param discretization_time: [s] Time discretization used for integration.
    :return: The trajectory of the variable at timesteps 0,..., N.
    """
    assert discretization_time > 0.0, 'Discretization time must be positive.'
    profile = initial_condition + np.insert(np.cumsum(derivatives * discretization_time), 0, 0.0)
    return profile

def _fit_initial_curvature_and_curvature_rate_profile(heading_displacements: DoubleMatrix, velocity_profile: DoubleMatrix, discretization_time: float, curvature_rate_penalty: float, initial_curvature_penalty: float=INITIAL_CURVATURE_PENALTY) -> Tuple[float, DoubleMatrix]:
    """
    Estimates initial curvature (curvature_0) and curvature rate ({curvature_rate_0, ...})
    using least squares with curvature rate regularization.
    :param heading_displacements: [rad] Angular deviations in heading occuring between timesteps.
    :param velocity_profile: [m/s] Estimated or actual velocities at the timesteps matching displacements.
    :param discretization_time: [s] Time discretization used for integration.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :param initial_curvature_penalty: A regularization parameter to handle zero initial speed.  Should be positive and small.
    :return: Least squares solution for initial curvature (curvature_0) and curvature rate profile
             (curvature_rate_0, ..., curvature_rate_{M-1}) for M heading displacement values.
    """
    assert discretization_time > 0.0, 'Discretization time must be positive.'
    assert curvature_rate_penalty > 0.0, 'Should have a positive curvature_rate_penalty.'
    assert initial_curvature_penalty > 0.0, 'Should have a positive initial_curvature_penalty.'
    y = heading_displacements
    A: DoubleMatrix = np.tri(len(y), dtype=np.float64)
    A[:, 0] = velocity_profile * discretization_time
    for idx, velocity in enumerate(velocity_profile):
        if idx == 0:
            continue
        A[idx, 1:] *= velocity * discretization_time ** 2
    Q: DoubleMatrix = curvature_rate_penalty * np.eye(len(y))
    Q[0, 0] = initial_curvature_penalty
    x = np.linalg.pinv(A.T @ A + Q) @ A.T @ y
    initial_curvature = x[0]
    curvature_rate_profile = x[1:]
    return (initial_curvature, curvature_rate_profile)

def complete_kinematic_state_and_inputs_from_poses(discretization_time: float, wheel_base: float, poses: DoubleMatrix, jerk_penalty: float, curvature_rate_penalty: float) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Main function for joint estimation of velocity, acceleration, steering angle, and steering rate given poses
    sampled at discretization_time and the vehicle wheelbase parameter for curvature -> steering angle conversion.
    One caveat is that we can only determine the first N-1 kinematic states and N-2 kinematic inputs given
    N-1 displacement/difference values, so we need to extrapolate to match the length of poses provided.
    This is handled by repeating the last input and extrapolating the motion model for the last state.
    :param discretization_time: [s] Time discretization used for integration.
    :param wheel_base: [m] The wheelbase length for the kinematic bicycle model being used.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of poses (x, y, heading).
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :return: kinematic_states (x, y, heading, velocity, steering_angle) and corresponding
            kinematic_inputs (acceleration, steering_rate).
    """
    velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile = get_velocity_curvature_profiles_with_derivatives_from_poses(discretization_time=discretization_time, poses=poses, jerk_penalty=jerk_penalty, curvature_rate_penalty=curvature_rate_penalty)
    steering_angle_profile, steering_rate_profile = _convert_curvature_profile_to_steering_profile(curvature_profile=curvature_profile, discretization_time=discretization_time, wheel_base=wheel_base)
    acceleration_profile = np.append(acceleration_profile, acceleration_profile[-1])
    steering_rate_profile = np.append(steering_rate_profile, steering_rate_profile[-1])
    velocity_profile = np.append(velocity_profile, velocity_profile[-1] + acceleration_profile[-1] * discretization_time)
    steering_angle_profile = np.append(steering_angle_profile, steering_angle_profile[-1] + steering_rate_profile[-1] * discretization_time)
    kinematic_states: DoubleMatrix = np.column_stack((poses, velocity_profile, steering_angle_profile))
    kinematic_inputs: DoubleMatrix = np.column_stack((acceleration_profile, steering_rate_profile))
    return (kinematic_states, kinematic_inputs)

def _convert_curvature_profile_to_steering_profile(curvature_profile: DoubleMatrix, discretization_time: float, wheel_base: float) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Converts from a curvature profile to the corresponding steering profile.
    We assume a kinematic bicycle model where curvature = tan(steering_angle) / wheel_base.
    For simplicity, we just use finite differences to determine steering rate.
    :param curvature_profile: [rad] Curvature trajectory to convert.
    :param discretization_time: [s] Time discretization used for integration.
    :param wheel_base: [m] The vehicle wheelbase parameter required for conversion.
    :return: The [rad] steering angle and [rad/s] steering rate (derivative) profiles.
    """
    assert discretization_time > 0.0, 'Discretization time must be positive.'
    assert wheel_base > 0.0, "The vehicle's wheelbase length must be positive."
    steering_angle_profile = np.arctan(wheel_base * curvature_profile)
    steering_rate_profile = np.diff(steering_angle_profile) / discretization_time
    return (steering_angle_profile, steering_rate_profile)

def get_interpolated_reference_trajectory_poses(trajectory: AbstractTrajectory, discretization_time: float) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Resamples the reference trajectory at discretization_time resolution.
    It will return N times and poses, where N is a function of the trajectory duration and the discretization time.
    :param trajectory: The full trajectory from which we perform pose interpolation.
    :param discretization_time: [s] The discretization time for resampling the trajectory.
    :return An array of times in seconds (N) and an array of associated poses (N,3), sampled at the discretization time.
    """
    start_time_point = trajectory.start_time
    end_time_point = trajectory.end_time
    delta_time_point = TimePoint(int(discretization_time * 1000000.0))
    interpolation_times_us = np.arange(start_time_point.time_us, end_time_point.time_us, delta_time_point.time_us)
    if interpolation_times_us[-1] + delta_time_point.time_us <= end_time_point.time_us:
        interpolation_times_us = np.append(interpolation_times_us, interpolation_times_us[-1] + delta_time_point.time_us)
    interpolation_time_points = [TimePoint(t_us) for t_us in interpolation_times_us]
    states = trajectory.get_state_at_times(interpolation_time_points)
    poses_interp = [[*state.rear_axle] for state in states]
    return (interpolation_times_us / 1000000.0, np.array(poses_interp))

class LQRTracker(AbstractTracker):
    """
    Implements an LQR tracker for a kinematic bicycle model.

    We decouple into two subsystems, longitudinal and lateral, with small angle approximations for linearization.
    We then solve two sequential LQR subproblems to find acceleration and steering rate inputs.

    Longitudinal Subsystem:
        States: [velocity]
        Inputs: [acceleration]
        Dynamics (continuous time):
            velocity_dot = acceleration

    Lateral Subsystem (After Linearization/Small Angle Approximation):
        States: [lateral_error, heading_error, steering_angle]
        Inputs: [steering_rate]
        Parameters: [velocity, curvature]
        Dynamics (continuous time):
            lateral_error_dot  = velocity * heading_error
            heading_error_dot  = velocity * (steering_angle / wheelbase_length - curvature)
            steering_angle_dot = steering_rate

    The continuous time dynamics are discretized using Euler integration and zero-order-hold on the input.
    In case of a stopping reference, we use a simplified stopping P controller instead of LQR.

    The final control inputs passed on to the motion model are:
        - acceleration
        - steering_rate
    """

    def __init__(self, q_longitudinal: npt.NDArray[np.float64], r_longitudinal: npt.NDArray[np.float64], q_lateral: npt.NDArray[np.float64], r_lateral: npt.NDArray[np.float64], discretization_time: float, tracking_horizon: int, jerk_penalty: float, curvature_rate_penalty: float, stopping_proportional_gain: float, stopping_velocity: float, vehicle: VehicleParameters=get_pacifica_parameters()):
        """
        Constructor for LQR controller
        :param q_longitudinal: The weights for the Q matrix for the longitudinal subystem.
        :param r_longitudinal: The weights for the R matrix for the longitudinal subystem.
        :param q_lateral: The weights for the Q matrix for the lateral subystem.
        :param r_lateral: The weights for the R matrix for the lateral subystem.
        :param discretization_time: [s] The time interval used for discretizing the continuous time dynamics.
        :param tracking_horizon: How many discrete time steps ahead to consider for the LQR objective.
        :param stopping_proportional_gain: The proportional_gain term for the P controller when coming to a stop.
        :param stopping_velocity: [m/s] The velocity below which we are deemed to be stopping and we don't use LQR.
        :param vehicle: Vehicle parameters
        """
        assert len(q_longitudinal) == 1, 'q_longitudinal should have 1 element (velocity).'
        assert len(r_longitudinal) == 1, 'r_longitudinal should have 1 element (acceleration).'
        self._q_longitudinal: npt.NDArray[np.float64] = np.diag(q_longitudinal)
        self._r_longitudinal: npt.NDArray[np.float64] = np.diag(r_longitudinal)
        assert len(q_lateral) == 3, 'q_lateral should have 3 elements (lateral_error, heading_error, steering_angle).'
        assert len(r_lateral) == 1, 'r_lateral should have 1 element (steering_rate).'
        self._q_lateral: npt.NDArray[np.float64] = np.diag(q_lateral)
        self._r_lateral: npt.NDArray[np.float64] = np.diag(r_lateral)
        for attr in ['_q_lateral', '_q_longitudinal']:
            assert np.all(np.diag(getattr(self, attr)) >= 0.0), f'self.{attr} must be positive semidefinite.'
        for attr in ['_r_lateral', '_r_longitudinal']:
            assert np.all(np.diag(getattr(self, attr)) > 0.0), f'self.{attr} must be positive definite.'
        assert discretization_time > 0.0, 'The discretization_time should be positive.'
        assert tracking_horizon > 1, 'We expect the horizon to be greater than 1 - else steering_rate has no impact with Euler integration.'
        self._discretization_time = discretization_time
        self._tracking_horizon = tracking_horizon
        self._wheel_base = vehicle.wheel_base
        assert jerk_penalty > 0.0, 'The jerk penalty must be positive.'
        assert curvature_rate_penalty > 0.0, 'The curvature rate penalty must be positive.'
        self._jerk_penalty = jerk_penalty
        self._curvature_rate_penalty = curvature_rate_penalty
        assert stopping_proportional_gain > 0, 'stopping_proportional_gain has to be greater than 0.'
        assert stopping_velocity > 0, 'stopping_velocity has to be greater than 0.'
        self._stopping_proportional_gain = stopping_proportional_gain
        self._stopping_velocity = stopping_velocity

    def track_trajectory(self, current_iteration: SimulationIteration, next_iteration: SimulationIteration, initial_state: EgoState, trajectory: AbstractTrajectory) -> DynamicCarState:
        """Inherited, see superclass."""
        initial_velocity, initial_lateral_state_vector = self._compute_initial_velocity_and_lateral_state(current_iteration, initial_state, trajectory)
        reference_velocity, curvature_profile = self._compute_reference_velocity_and_curvature_profile(current_iteration, trajectory)
        should_stop = reference_velocity <= self._stopping_velocity and initial_velocity <= self._stopping_velocity
        if should_stop:
            accel_cmd, steering_rate_cmd = self._stopping_controller(initial_velocity, reference_velocity)
        else:
            accel_cmd = self._longitudinal_lqr_controller(initial_velocity, reference_velocity)
            velocity_profile = _generate_profile_from_initial_condition_and_derivatives(initial_condition=initial_velocity, derivatives=np.ones(self._tracking_horizon, dtype=np.float64) * accel_cmd, discretization_time=self._discretization_time)[:self._tracking_horizon]
            steering_rate_cmd = self._lateral_lqr_controller(initial_lateral_state_vector, velocity_profile, curvature_profile)
        return DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist, rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=StateVector2D(accel_cmd, 0), tire_steering_rate=steering_rate_cmd)

    def _compute_initial_velocity_and_lateral_state(self, current_iteration: SimulationIteration, initial_state: EgoState, trajectory: AbstractTrajectory) -> Tuple[float, npt.NDArray[np.float64]]:
        """
        This method projects the initial tracking error into vehicle/Frenet frame.  It also extracts initial velocity.
        :param current_iteration: Used to get the current time.
        :param initial_state: The current state for ego.
        :param trajectory: The reference trajectory we are tracking.
        :return: Initial velocity [m/s] and initial lateral state.
        """
        initial_trajectory_state = trajectory.get_state_at_time(current_iteration.time_point)
        x_error = initial_state.rear_axle.x - initial_trajectory_state.rear_axle.x
        y_error = initial_state.rear_axle.y - initial_trajectory_state.rear_axle.y
        heading_reference = initial_trajectory_state.rear_axle.heading
        lateral_error = -x_error * np.sin(heading_reference) + y_error * np.cos(heading_reference)
        heading_error = angle_diff(initial_state.rear_axle.heading, heading_reference, 2 * np.pi)
        initial_velocity = initial_state.dynamic_car_state.rear_axle_velocity_2d.x
        initial_lateral_state_vector: npt.NDArray[np.float64] = np.array([lateral_error, heading_error, initial_state.tire_steering_angle], dtype=np.float64)
        return (initial_velocity, initial_lateral_state_vector)

    def _compute_reference_velocity_and_curvature_profile(self, current_iteration: SimulationIteration, trajectory: AbstractTrajectory) -> Tuple[float, npt.NDArray[np.float64]]:
        """
        This method computes reference velocity and curvature profile based on the reference trajectory.
        We use a lookahead time equal to self._tracking_horizon * self._discretization_time.
        :param current_iteration: Used to get the current time.
        :param trajectory: The reference trajectory we are tracking.
        :return: The reference velocity [m/s] and curvature profile [rad] to track.
        """
        times_s, poses = get_interpolated_reference_trajectory_poses(trajectory, self._discretization_time)
        velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile = get_velocity_curvature_profiles_with_derivatives_from_poses(discretization_time=self._discretization_time, poses=poses, jerk_penalty=self._jerk_penalty, curvature_rate_penalty=self._curvature_rate_penalty)
        reference_time = current_iteration.time_point.time_s + self._tracking_horizon * self._discretization_time
        reference_velocity = np.interp(reference_time, times_s[:-1], velocity_profile)
        profile_times = [current_iteration.time_point.time_s + x * self._discretization_time for x in range(self._tracking_horizon)]
        reference_curvature_profile = np.interp(profile_times, times_s[:-1], curvature_profile)
        return (float(reference_velocity), reference_curvature_profile)

    def _stopping_controller(self, initial_velocity: float, reference_velocity: float) -> Tuple[float, float]:
        """
        Apply proportional controller when at near-stop conditions.
        :param initial_velocity: [m/s] The current velocity of ego.
        :param reference_velocity: [m/s] The reference velocity to track.
        :return: Acceleration [m/s^2] and zero steering_rate [rad/s] command.
        """
        accel = -self._stopping_proportional_gain * (initial_velocity - reference_velocity)
        return (accel, 0.0)

    def _longitudinal_lqr_controller(self, initial_velocity: float, reference_velocity: float) -> float:
        """
        This longitudinal controller determines an acceleration input to minimize velocity error at a lookahead time.
        :param initial_velocity: [m/s] The current velocity of ego.
        :param reference_velocity: [m/s] The reference_velocity to track at a lookahead time.
        :return: Acceleration [m/s^2] command based on LQR.
        """
        A: npt.NDArray[np.float64] = np.array([1.0], dtype=np.float64)
        B: npt.NDArray[np.float64] = np.array([self._tracking_horizon * self._discretization_time], dtype=np.float64)
        accel_cmd = self._solve_one_step_lqr(initial_state=np.array([initial_velocity], dtype=np.float64), reference_state=np.array([reference_velocity], dtype=np.float64), Q=self._q_longitudinal, R=self._r_longitudinal, A=A, B=B, g=np.zeros(1, dtype=np.float64), angle_diff_indices=[])
        return float(accel_cmd)

    def _lateral_lqr_controller(self, initial_lateral_state_vector: npt.NDArray[np.float64], velocity_profile: npt.NDArray[np.float64], curvature_profile: npt.NDArray[np.float64]) -> float:
        """
        This lateral controller determines a steering_rate input to minimize lateral errors at a lookahead time.
        It requires a velocity sequence as a parameter to ensure linear time-varying lateral dynamics.
        :param initial_lateral_state_vector: The current lateral state of ego.
        :param velocity_profile: [m/s] The velocity over the entire self._tracking_horizon-step lookahead.
        :param curvature_profile: [rad] The curvature over the entire self._tracking_horizon-step lookahead..
        :return: Steering rate [rad/s] command based on LQR.
        """
        assert len(velocity_profile) == self._tracking_horizon, f'The linearization velocity sequence should have length {self._tracking_horizon} but is {len(velocity_profile)}.'
        assert len(curvature_profile) == self._tracking_horizon, f'The linearization curvature sequence should have length {self._tracking_horizon} but is {len(curvature_profile)}.'
        n_lateral_states = len(LateralStateIndex)
        I: npt.NDArray[np.float64] = np.eye(n_lateral_states, dtype=np.float64)
        A: npt.NDArray[np.float64] = I
        B: npt.NDArray[np.float64] = np.zeros((n_lateral_states, 1), dtype=np.float64)
        g: npt.NDArray[np.float64] = np.zeros(n_lateral_states, dtype=np.float64)
        idx_lateral_error = LateralStateIndex.LATERAL_ERROR
        idx_heading_error = LateralStateIndex.HEADING_ERROR
        idx_steering_angle = LateralStateIndex.STEERING_ANGLE
        input_matrix: npt.NDArray[np.float64] = np.zeros((n_lateral_states, 1), np.float64)
        input_matrix[idx_steering_angle] = self._discretization_time
        for index_step, (velocity, curvature) in enumerate(zip(velocity_profile, curvature_profile)):
            state_matrix_at_step: npt.NDArray[np.float64] = np.eye(n_lateral_states, dtype=np.float64)
            state_matrix_at_step[idx_lateral_error, idx_heading_error] = velocity * self._discretization_time
            state_matrix_at_step[idx_heading_error, idx_steering_angle] = velocity * self._discretization_time / self._wheel_base
            affine_term: npt.NDArray[np.float64] = np.zeros(n_lateral_states, dtype=np.float64)
            affine_term[idx_heading_error] = -velocity * curvature * self._discretization_time
            A = state_matrix_at_step @ A
            B = state_matrix_at_step @ B + input_matrix
            g = state_matrix_at_step @ g + affine_term
        steering_rate_cmd = self._solve_one_step_lqr(initial_state=initial_lateral_state_vector, reference_state=np.zeros(n_lateral_states, dtype=np.float64), Q=self._q_lateral, R=self._r_lateral, A=A, B=B, g=g, angle_diff_indices=[idx_heading_error, idx_steering_angle])
        return float(steering_rate_cmd)

    @staticmethod
    def _solve_one_step_lqr(initial_state: npt.NDArray[np.float64], reference_state: npt.NDArray[np.float64], Q: npt.NDArray[np.float64], R: npt.NDArray[np.float64], A: npt.NDArray[np.float64], B: npt.NDArray[np.float64], g: npt.NDArray[np.float64], angle_diff_indices: List[int]=[]) -> npt.NDArray[np.float64]:
        """
        This function uses LQR to find an optimal input to minimize tracking error in one step of dynamics.
        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.
        :param initial_state: The current state.
        :param reference_state: The desired state in 1 step (according to A,B,g dynamics).
        :param Q: The state tracking 2-norm cost matrix.
        :param R: The input 2-norm cost matrix.
        :param A: The state dynamics matrix.
        :param B: The input dynamics matrix.
        :param g: The offset/affine dynamics term.
        :param angle_diff_indices: The set of state indices for which we need to apply angle differences, if defined.
        :return: LQR optimal input for the 1-step problem.
        """
        state_error_zero_input = A @ initial_state + g - reference_state
        for angle_diff_index in angle_diff_indices:
            state_error_zero_input[angle_diff_index] = angle_diff(state_error_zero_input[angle_diff_index], 0.0, 2 * np.pi)
        lqr_input = -np.linalg.inv(B.T @ Q @ B + R) @ B.T @ Q @ state_error_zero_input
        return lqr_input

def _integrate_acceleration_and_curvature_profile(initial_pose: DoubleMatrix, initial_velocity: DoubleMatrix, initial_curvature: DoubleMatrix, acceleration_profile: DoubleMatrix, curvature_rate_profile: DoubleMatrix, discretization_time: float) -> Tuple[DoubleMatrix, DoubleMatrix, DoubleMatrix]:
    """
    This test helper function takes in an initial state and input profile to generate the associated state trajectory.
    We use curvature for simplicity (the relationship with steering angle is 1-1 for the achievable range).
    :param initial_pose: Initial (x, y, heading) pose state.
    :param initial_velocity: [m/s] The initial velocity state.
    :param initial_curvature: [rad] The initial curvature state.
    :param acceleration_profile: [m/s^2] The acceleration input sequence to apply.
    :param curvature_rate_profile: [rad/s] The curvature rate input to apply.
    :param discretization_time: [s] Time discretization used for integration.
    :return Pose, velocity, and curvature state trajectories after integration.
    """
    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(initial_condition=initial_velocity, derivatives=acceleration_profile, discretization_time=discretization_time)
    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(initial_condition=initial_curvature, derivatives=curvature_rate_profile, discretization_time=discretization_time)
    pose_trajectory = [initial_pose]
    for velocity, curvature in zip(velocity_profile, curvature_profile):
        x, y, heading = pose_trajectory[-1]
        next_pose = [x + velocity * np.cos(heading) * discretization_time, y + velocity * np.sin(heading) * discretization_time, principal_value(heading + velocity * curvature * discretization_time)]
        pose_trajectory.append(next_pose)
    return (np.array(pose_trajectory), velocity_profile, curvature_profile)

class TestTrackerUtils(unittest.TestCase):
    """
    Tests tracker utils, including least squares fit of kinematic states given poses.
    Throughout, we assume a kinematic bicycle model as the base dynamics model.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.test_discretization_time = 0.2
        self.least_squares_penalty = 1e-10
        self.proximity_rtol = 1e-06
        self.proximity_atol = 1e-08
        self.moving_velocity_threshold = 0.1
        self.assert_allclose = partial(np_test.assert_allclose, rtol=self.proximity_rtol, atol=self.proximity_atol)
        self.test_wheel_base = 3.0
        self.initial_pose: DoubleMatrix = np.array([5.0, 1.0, 0.1], dtype=np.float64)
        self.initial_velocity = 3.0
        self.initial_curvature = 0.0
        max_acceleration = 3.0
        max_curvature_rate = 0.05
        input_length = 10
        self.input_profiles = {}
        acceleration_profile_dict = _make_input_profiles(key_prefix='accel', magnitude=max_acceleration, length=input_length)
        curvature_rate_profile_dict = _make_input_profiles(key_prefix='curv_rate', magnitude=max_curvature_rate, length=input_length)
        for acceleration_profile_name, acceleration_profile in acceleration_profile_dict.items():
            for curvature_rate_profile_name, curvature_rate_profile in curvature_rate_profile_dict.items():
                poses, velocities, curvatures = _integrate_acceleration_and_curvature_profile(initial_pose=self.initial_pose, initial_velocity=self.initial_velocity, initial_curvature=self.initial_curvature, acceleration_profile=acceleration_profile, curvature_rate_profile=curvature_rate_profile, discretization_time=self.test_discretization_time)
                self.input_profiles[f'{acceleration_profile_name}_{curvature_rate_profile_name}'] = {'acceleration': acceleration_profile, 'curvature_rate': curvature_rate_profile, 'poses': poses, 'velocity': velocities, 'curvature': curvatures}

    def test__generate_profile_from_initial_condition_and_derivatives(self) -> None:
        """
        Check that we can correctly integrate derivative profiles.
        We use a loop here to compare against the vectorized implementation.
        """
        for input_profile in self.input_profiles.values():
            velocity_profile = [self.initial_velocity]
            for acceleration in input_profile['acceleration']:
                velocity_profile.append(velocity_profile[-1] + acceleration * self.test_discretization_time)
            self.assert_allclose(velocity_profile, input_profile['velocity'])
            curvature_profile = [self.initial_curvature]
            for curvature_rate in input_profile['curvature_rate']:
                curvature_profile.append(curvature_profile[-1] + curvature_rate * self.test_discretization_time)
            self.assert_allclose(curvature_profile, input_profile['curvature'])

    def test__get_xy_heading_displacements_from_poses(self) -> None:
        """Get displacements and check consistency with original pose trajectory."""
        for input_profile in self.input_profiles.values():
            poses = input_profile['poses']
            xy_displacements, heading_displacements = _get_xy_heading_displacements_from_poses(poses)
            self.assertEqual(len(xy_displacements), len(poses) - 1)
            self.assertEqual(len(heading_displacements), len(poses) - 1)
            x_integrated = _generate_profile_from_initial_condition_and_derivatives(initial_condition=self.initial_pose[0], derivatives=xy_displacements[:, 0], discretization_time=1.0)
            y_integrated = _generate_profile_from_initial_condition_and_derivatives(initial_condition=self.initial_pose[1], derivatives=xy_displacements[:, 1], discretization_time=1.0)
            heading_integrated = _generate_profile_from_initial_condition_and_derivatives(initial_condition=self.initial_pose[2], derivatives=heading_displacements, discretization_time=1.0)
            heading_integrated = principal_value(heading_integrated)
            self.assert_allclose(np.column_stack((x_integrated, y_integrated, heading_integrated)), poses)

    def test__make_banded_difference_matrix(self) -> None:
        """Test that the banded difference matrix has expected structure for different sizes."""
        for test_number_rows in [1, 5, 10]:
            banded_difference_matrix = _make_banded_difference_matrix(test_number_rows)
            self.assertEqual(banded_difference_matrix.shape, (test_number_rows, test_number_rows + 1))
            self.assert_allclose(np.diag(banded_difference_matrix, k=0), -1.0)
            self.assert_allclose(np.diag(banded_difference_matrix, k=1), 1.0)
            removal_mask = np.ones_like(banded_difference_matrix)
            for idx in range(len(removal_mask)):
                removal_mask[idx, idx:idx + 2] = 0.0
            banded_difference_matrix_masked = np.multiply(banded_difference_matrix, removal_mask)
            self.assert_allclose(banded_difference_matrix_masked, 0.0)

    def test__convert_curvature_profile_to_steering_profile(self) -> None:
        """Check consistency of converted steering angle/rate with curvature and pose information."""
        for input_profile in self.input_profiles.values():
            curvature_profile = input_profile['curvature']
            velocity_profile = input_profile['velocity']
            heading_profile = input_profile['poses'][:, 2]
            steering_angle_profile, steering_rate_profile = _convert_curvature_profile_to_steering_profile(curvature_profile=curvature_profile, discretization_time=self.test_discretization_time, wheel_base=self.test_wheel_base)
            self.assertEqual(len(steering_angle_profile), len(curvature_profile))
            self.assertEqual(len(steering_rate_profile), len(curvature_profile) - 1)
            steering_angle_integrated = _generate_profile_from_initial_condition_and_derivatives(initial_condition=steering_angle_profile[0], derivatives=steering_rate_profile, discretization_time=self.test_discretization_time)
            self.assert_allclose(steering_angle_integrated, steering_angle_profile)
            yawrate_profile = velocity_profile * np.tan(steering_angle_profile) / self.test_wheel_base
            heading_integrated = _generate_profile_from_initial_condition_and_derivatives(initial_condition=self.initial_pose[2], derivatives=yawrate_profile, discretization_time=self.test_discretization_time)
            heading_integrated = principal_value(heading_integrated)
            self.assert_allclose(heading_integrated, heading_profile)

    def test__fit_initial_velocity_and_acceleration_profile(self) -> None:
        """
        Test given noiseless data and a small jerk penalty, the least squares speed/acceleration match expected values.
        """
        for input_profile in self.input_profiles.values():
            poses = input_profile['poses']
            xy_displacements, _ = _get_xy_heading_displacements_from_poses(poses)
            heading_profile = poses[:-1, 2]
            initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(xy_displacements=xy_displacements, heading_profile=heading_profile, discretization_time=self.test_discretization_time, jerk_penalty=self.least_squares_penalty)
            velocity_profile = _generate_profile_from_initial_condition_and_derivatives(initial_condition=initial_velocity, derivatives=acceleration_profile, discretization_time=self.test_discretization_time)
            self.assert_allclose(velocity_profile, input_profile['velocity'])
            self.assert_allclose(acceleration_profile, input_profile['acceleration'])

    def test__fit_initial_curvature_and_curvature_rate_profile(self) -> None:
        """
        Test given noiseless data and a small curvature_rate penalty, the least squares curvature/curvature rate match
        expected values.  A caveat is we exclude cases where ego is stopped and thus curvature estimation is unreliable.
        """
        for input_profile in self.input_profiles.values():
            poses = input_profile['poses']
            velocity_profile = input_profile['velocity']
            _, heading_displacements = _get_xy_heading_displacements_from_poses(poses)
            initial_curvature, curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(heading_displacements=heading_displacements, velocity_profile=velocity_profile, discretization_time=self.test_discretization_time, curvature_rate_penalty=self.least_squares_penalty)
            curvature_profile = _generate_profile_from_initial_condition_and_derivatives(initial_condition=initial_curvature, derivatives=curvature_rate_profile, discretization_time=self.test_discretization_time)
            moving_mask = (np.abs(velocity_profile) > self.moving_velocity_threshold).astype(np.float64)
            self.assert_allclose(moving_mask * curvature_profile, moving_mask * input_profile['curvature'])
            if np.all(moving_mask > 0.0):
                self.assert_allclose(curvature_rate_profile, input_profile['curvature_rate'])

    def test_compute_steering_angle_feedback(self) -> None:
        """Check that sign of the steering angle feedback makes sense for various initial tracking errors."""
        pose_reference: DoubleMatrix = self.initial_pose
        heading_reference = pose_reference[2]
        lookahead_distance = 10.0
        k_lateral_error = 0.1
        steering_angle_zero_lateral_error = compute_steering_angle_feedback(pose_reference=pose_reference, pose_current=pose_reference, lookahead_distance=lookahead_distance, k_lateral_error=k_lateral_error)
        self.assertEqual(steering_angle_zero_lateral_error, 0.0)
        for lateral_error in [-1.0, 1.0]:
            pose_lateral_error: DoubleMatrix = pose_reference + lateral_error * np.array([-np.sin(heading_reference), np.cos(heading_reference), 0.0])
            steering_angle_lateral_error = compute_steering_angle_feedback(pose_reference=pose_reference, pose_current=pose_lateral_error, lookahead_distance=lookahead_distance, k_lateral_error=k_lateral_error)
            self.assertEqual(-np.sign(lateral_error), np.sign(steering_angle_lateral_error))
        for heading_error in [-0.05, 0.05]:
            steering_angle_heading_error = compute_steering_angle_feedback(pose_reference=pose_reference, pose_current=pose_reference + [0.0, 0.0, heading_error], lookahead_distance=lookahead_distance, k_lateral_error=k_lateral_error)
            self.assertEqual(-np.sign(heading_error), np.sign(steering_angle_heading_error))

    def test_get_velocity_curvature_profiles_with_derivatives_from_poses(self) -> None:
        """
        Test the joint estimation of velocity and curvature, along with their derivatives.
        Since there is overlap with complete_kinematic_state_and_inputs_from_poses,
        we just test for one given input profile and leave the extensive testing for that function.
        """
        test_input_profile = self.input_profiles['accel_cosine_curv_rate_cosine']
        velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile = get_velocity_curvature_profiles_with_derivatives_from_poses(discretization_time=self.test_discretization_time, poses=test_input_profile['poses'], jerk_penalty=self.least_squares_penalty, curvature_rate_penalty=self.least_squares_penalty)
        self.assert_allclose(velocity_profile, test_input_profile['velocity'])
        self.assert_allclose(acceleration_profile, test_input_profile['acceleration'])
        self.assert_allclose(curvature_profile, test_input_profile['curvature'])
        self.assert_allclose(curvature_rate_profile, test_input_profile['curvature_rate'])
        self.assert_allclose(np.diff(velocity_profile) / self.test_discretization_time, acceleration_profile)
        self.assert_allclose(np.diff(curvature_profile) / self.test_discretization_time, curvature_rate_profile)

    def test_complete_kinematic_state_and_inputs_from_poses(self) -> None:
        """
        Test that the joint estimation of kinematic states and inputs are consistent with expectations.
        Since there is extrapolation involved, we only compare the non-extrapolated values.
        """
        for input_profile in self.input_profiles.values():
            poses = input_profile['poses']
            velocity_profile = input_profile['velocity']
            acceleration_profile = input_profile['acceleration']
            curvature_profile = input_profile['curvature']
            kinematic_states, kinematic_inputs = complete_kinematic_state_and_inputs_from_poses(discretization_time=self.test_discretization_time, wheel_base=self.test_wheel_base, poses=poses, jerk_penalty=self.least_squares_penalty, curvature_rate_penalty=self.least_squares_penalty)
            velocity_fit = kinematic_states[:-1, 3]
            self.assert_allclose(velocity_fit, velocity_profile)
            acceleration_fit = kinematic_inputs[:-1, 0]
            self.assert_allclose(acceleration_fit, acceleration_profile)
            steering_angle_expected, steering_rate_expected = _convert_curvature_profile_to_steering_profile(curvature_profile=curvature_profile, discretization_time=self.test_discretization_time, wheel_base=self.test_wheel_base)
            moving_mask = (np.abs(velocity_profile) > self.moving_velocity_threshold).astype(np.float64)
            steering_angle_fit = kinematic_states[:-1, 4]
            self.assert_allclose(moving_mask * steering_angle_fit, moving_mask * steering_angle_expected)
            if np.all(moving_mask > 0.0):
                steering_rate_fit = kinematic_inputs[:-1, 1]
                self.assert_allclose(steering_rate_fit, steering_rate_expected)

    def test_get_interpolated_reference_trajectory_poses(self) -> None:
        """
        Test that we can interpolate a trajectory with constant discretization time and extract poses.
        """
        scenario = MockAbstractScenario()
        trajectory = InterpolatedTrajectory(list(scenario.get_expert_ego_trajectory()))
        expected_num_steps = 1 + int((trajectory.end_time.time_s - trajectory.start_time.time_s) / self.test_discretization_time)
        times_s, poses = get_interpolated_reference_trajectory_poses(trajectory, self.test_discretization_time)
        self.assertEqual(times_s.shape, (expected_num_steps,))
        self.assertEqual(poses.shape, (expected_num_steps, 3))
        self.assertTrue(np.all(times_s >= trajectory.start_time.time_s))
        self.assertTrue(np.all(times_s <= trajectory.end_time.time_s))
        self.assert_allclose(np.diff(times_s), self.test_discretization_time)

def _make_input_profiles(key_prefix: str, magnitude: float, length: int) -> Dict[str, DoubleMatrix]:
    """
    This test helper function adds input profiles to a dictionary to enable parametrized testing of the tracker utils.
    :param key_prefix: A prefix for keys in the dictionary, e.g. "curv_rate" or "acceleration".
    :param magnitude: A maximum absolute value bound for the input profile.
    :param length: How many elements (timesteps) we should have within the input profile.
    :return: A dictionary containing multiple input profiles we can apply.
    """
    acceleration_dict: Dict[str, DoubleMatrix] = {}
    acceleration_dict[f'{key_prefix}_positive'] = magnitude * np.ones(length, dtype=np.float64)
    acceleration_dict[f'{key_prefix}_zero'] = np.zeros(length, dtype=np.float64)
    acceleration_dict[f'{key_prefix}_negative'] = -magnitude * np.ones(length, dtype=np.float64)
    acceleration_dict[f'{key_prefix}_cosine'] = magnitude * np.cos(np.arange(length, dtype=np.float64))
    return acceleration_dict

class ILQRSolver:
    """iLQR solver implementation, see module docstring for details."""

    def __init__(self, solver_params: ILQRSolverParameters, warm_start_params: ILQRWarmStartParameters) -> None:
        """
        Initialize solver parameters.
        :param solver_params: Contains solver parameters for iLQR.
        :param warm_start_params: Contains warm start parameters for iLQR.
        """
        self._solver_params = solver_params
        self._warm_start_params = warm_start_params
        self._n_states = 5
        self._n_inputs = 2
        state_cost_diagonal_entries = self._solver_params.state_cost_diagonal_entries
        assert len(state_cost_diagonal_entries) == self._n_states, f'State cost matrix should have diagonal length {self._n_states}.'
        self._state_cost_matrix: DoubleMatrix = np.diag(state_cost_diagonal_entries)
        input_cost_diagonal_entries = self._solver_params.input_cost_diagonal_entries
        assert len(input_cost_diagonal_entries) == self._n_inputs, f'Input cost matrix should have diagonal length {self._n_inputs}.'
        self._input_cost_matrix: DoubleMatrix = np.diag(input_cost_diagonal_entries)
        state_trust_region_entries = self._solver_params.state_trust_region_entries
        assert len(state_trust_region_entries) == self._n_states, f'State trust region cost matrix should have diagonal length {self._n_states}.'
        self._state_trust_region_cost_matrix: DoubleMatrix = np.diag(state_trust_region_entries)
        input_trust_region_entries = self._solver_params.input_trust_region_entries
        assert len(input_trust_region_entries) == self._n_inputs, f'Input trust region cost matrix should have diagonal length {self._n_inputs}.'
        self._input_trust_region_cost_matrix: DoubleMatrix = np.diag(input_trust_region_entries)
        max_acceleration = self._solver_params.max_acceleration
        max_steering_angle_rate = self._solver_params.max_steering_angle_rate
        self._input_clip_min = (-max_acceleration, -max_steering_angle_rate)
        self._input_clip_max = (max_acceleration, max_steering_angle_rate)

    def solve(self, current_state: DoubleMatrix, reference_trajectory: DoubleMatrix) -> List[ILQRSolution]:
        """
        Run the main iLQR loop used to try to find (locally) optimal inputs to track the reference trajectory.
        :param current_state: The initial state from which we apply inputs, z_0.
        :param reference_trajectory: The state reference we'd like to track, inclusive of the initial timestep,
                                     z_{r,k} for k in {0, ..., N}.
        :return: A list of solution iterates after running the iLQR algorithm where the index is the iteration number.
        """
        assert current_state.shape == (self._n_states,), 'Incorrect state shape.'
        assert len(reference_trajectory.shape) == 2, 'Reference trajectory should be a 2D matrix.'
        reference_trajectory_length, reference_trajectory_state_dimension = reference_trajectory.shape
        assert reference_trajectory_length > 1, 'The reference trajectory should be at least two timesteps long.'
        assert reference_trajectory_state_dimension == self._n_states, 'The reference trajectory should have a matching state dimension.'
        solution_list: List[ILQRSolution] = []
        current_iterate = self._input_warm_start(current_state, reference_trajectory)
        solve_start_time = time.perf_counter()
        for _ in range(self._solver_params.max_ilqr_iterations):
            tracking_cost = self._compute_tracking_cost(iterate=current_iterate, reference_trajectory=reference_trajectory)
            solution_list.append(ILQRSolution(input_trajectory=current_iterate.input_trajectory, state_trajectory=current_iterate.state_trajectory, tracking_cost=tracking_cost))
            lqr_input_policy = self._run_lqr_backward_recursion(current_iterate=current_iterate, reference_trajectory=reference_trajectory)
            input_trajectory_next = self._update_inputs_with_policy(current_iterate=current_iterate, lqr_input_policy=lqr_input_policy)
            input_trajectory_norm_difference = np.linalg.norm(input_trajectory_next - current_iterate.input_trajectory)
            current_iterate = self._run_forward_dynamics(current_state, input_trajectory_next)
            if input_trajectory_norm_difference < self._solver_params.convergence_threshold:
                break
            elapsed_time = time.perf_counter() - solve_start_time
            if isinstance(self._solver_params.max_solve_time, float) and elapsed_time >= self._solver_params.max_solve_time:
                break
        tracking_cost = self._compute_tracking_cost(iterate=current_iterate, reference_trajectory=reference_trajectory)
        solution_list.append(ILQRSolution(input_trajectory=current_iterate.input_trajectory, state_trajectory=current_iterate.state_trajectory, tracking_cost=tracking_cost))
        return solution_list

    def _compute_tracking_cost(self, iterate: ILQRIterate, reference_trajectory: DoubleMatrix) -> float:
        """
        Compute the trajectory tracking cost given a candidate solution.
        :param iterate: Contains the candidate state and input trajectory to evaluate.
        :param reference_trajectory: The desired state reference trajectory with same length as state_trajectory.
        :return: The tracking cost of the candidate state/input trajectory.
        """
        input_trajectory = iterate.input_trajectory
        state_trajectory = iterate.state_trajectory
        assert len(state_trajectory) == len(reference_trajectory), 'The state and reference trajectory should have the same length.'
        error_state_trajectory = state_trajectory - reference_trajectory
        error_state_trajectory[:, 2] = principal_value(error_state_trajectory[:, 2])
        cost = np.sum([u.T @ self._input_cost_matrix @ u for u in input_trajectory]) + np.sum([e.T @ self._state_cost_matrix @ e for e in error_state_trajectory])
        return float(cost)

    def _clip_inputs(self, inputs: DoubleMatrix) -> DoubleMatrix:
        """
        Used to clip control inputs within constraints.
        :param: inputs: The control inputs with shape (self._n_inputs,) to clip.
        :return: Clipped version of the control inputs, unmodified if already within constraints.
        """
        assert inputs.shape == (self._n_inputs,), f'The inputs should be a 1D vector with {self._n_inputs} elements.'
        return np.clip(inputs, self._input_clip_min, self._input_clip_max)

    def _clip_steering_angle(self, steering_angle: float) -> float:
        """
        Used to clip the steering angle state within bounds.
        :param steering_angle: [rad] A steering angle (scalar) to clip.
        :return: [rad] The clipped steering angle.
        """
        steering_angle_sign = 1.0 if steering_angle >= 0 else -1.0
        steering_angle = steering_angle_sign * min(abs(steering_angle), self._solver_params.max_steering_angle)
        return steering_angle

    def _input_warm_start(self, current_state: DoubleMatrix, reference_trajectory: DoubleMatrix) -> ILQRIterate:
        """
        Given a reference trajectory, we generate the warm start (initial guess) by inferring the inputs applied based
        on poses in the reference trajectory.
        :param current_state: The initial state from which we apply inputs.
        :param reference_trajectory: The reference trajectory we are trying to follow.
        :return: The warm start iterate from which to start iLQR.
        """
        reference_states_completed, reference_inputs_completed = complete_kinematic_state_and_inputs_from_poses(discretization_time=self._solver_params.discretization_time, wheel_base=self._solver_params.wheelbase, poses=reference_trajectory[:, :3], jerk_penalty=self._warm_start_params.jerk_penalty_warm_start_fit, curvature_rate_penalty=self._warm_start_params.curvature_rate_penalty_warm_start_fit)
        _, _, _, velocity_current, steering_angle_current = current_state
        _, _, _, velocity_reference, steering_angle_reference = reference_states_completed[0, :]
        acceleration_feedback = -self._warm_start_params.k_velocity_error_feedback * (velocity_current - velocity_reference)
        steering_angle_feedback = compute_steering_angle_feedback(pose_reference=current_state[:3], pose_current=reference_states_completed[0, :3], lookahead_distance=self._warm_start_params.lookahead_distance_lateral_error, k_lateral_error=self._warm_start_params.k_lateral_error)
        steering_angle_desired = steering_angle_feedback + steering_angle_reference
        steering_rate_feedback = -self._warm_start_params.k_steering_angle_error_feedback * (steering_angle_current - steering_angle_desired)
        reference_inputs_completed[0, 0] += acceleration_feedback
        reference_inputs_completed[0, 1] += steering_rate_feedback
        return self._run_forward_dynamics(current_state, reference_inputs_completed)

    def _run_forward_dynamics(self, current_state: DoubleMatrix, input_trajectory: DoubleMatrix) -> ILQRIterate:
        """
        Compute states and corresponding state/input Jacobian matrices using forward dynamics.
        We additionally return the input since the dynamics may modify the input to ensure constraint satisfaction.
        :param current_state: The initial state from which we apply inputs.  Must be feasible given constraints.
        :param input_trajectory: The input trajectory applied to the model.  May be modified to ensure feasibility.
        :return: A feasible iterate after applying dynamics with state/input trajectories and Jacobian matrices.
        """
        N = len(input_trajectory)
        state_trajectory = np.nan * np.ones((N + 1, self._n_states), dtype=np.float64)
        final_input_trajectory = np.nan * np.ones_like(input_trajectory, dtype=np.float64)
        state_jacobian_trajectory = np.nan * np.ones((N, self._n_states, self._n_states), dtype=np.float64)
        final_input_jacobian_trajectory = np.nan * np.ones((N, self._n_states, self._n_inputs), dtype=np.float64)
        state_trajectory[0] = current_state
        for idx_u, u in enumerate(input_trajectory):
            state_next, final_input, state_jacobian, final_input_jacobian = self._dynamics_and_jacobian(state_trajectory[idx_u], u)
            state_trajectory[idx_u + 1] = state_next
            final_input_trajectory[idx_u] = final_input
            state_jacobian_trajectory[idx_u] = state_jacobian
            final_input_jacobian_trajectory[idx_u] = final_input_jacobian
        iterate = ILQRIterate(state_trajectory=state_trajectory, input_trajectory=final_input_trajectory, state_jacobian_trajectory=state_jacobian_trajectory, input_jacobian_trajectory=final_input_jacobian_trajectory)
        return iterate

    def _dynamics_and_jacobian(self, current_state: DoubleMatrix, current_input: DoubleMatrix) -> Tuple[DoubleMatrix, DoubleMatrix, DoubleMatrix, DoubleMatrix]:
        """
        Propagates the state forward by one step and computes the corresponding state and input Jacobian matrices.
        We also impose all constraints here to ensure the current input and next state are always feasible.
        :param current_state: The current state z_k.
        :param current_input: The applied input u_k.
        :return: The next state z_{k+1}, (possibly modified) input u_k, and state (df/dz) and input (df/du) Jacobians.
        """
        x, y, heading, velocity, steering_angle = current_state
        assert np.abs(steering_angle) < np.pi / 2.0, f'The steering angle {steering_angle} is outside expected limits.  There is a singularity at delta = np.pi/2.'
        current_input = self._clip_inputs(current_input)
        acceleration, steering_rate = current_input
        discretization_time = self._solver_params.discretization_time
        wheelbase = self._solver_params.wheelbase
        next_state: DoubleMatrix = np.copy(current_state)
        next_state[0] += velocity * np.cos(heading) * discretization_time
        next_state[1] += velocity * np.sin(heading) * discretization_time
        next_state[2] += velocity * np.tan(steering_angle) / wheelbase * discretization_time
        next_state[3] += acceleration * discretization_time
        next_state[4] += steering_rate * discretization_time
        next_state[2] = principal_value(next_state[2])
        next_steering_angle = self._clip_steering_angle(next_state[4])
        applied_steering_rate = (next_steering_angle - steering_angle) / discretization_time
        next_state[4] = next_steering_angle
        current_input[1] = applied_steering_rate
        state_jacobian: DoubleMatrix = np.eye(self._n_states, dtype=np.float64)
        input_jacobian: DoubleMatrix = np.zeros((self._n_states, self._n_inputs), dtype=np.float64)
        min_velocity_linearization = self._solver_params.min_velocity_linearization
        if -min_velocity_linearization <= velocity and velocity <= min_velocity_linearization:
            sign_velocity = 1.0 if velocity >= 0.0 else -1.0
            velocity = sign_velocity * min_velocity_linearization
        state_jacobian[0, 2] = -velocity * np.sin(heading) * discretization_time
        state_jacobian[0, 3] = np.cos(heading) * discretization_time
        state_jacobian[1, 2] = velocity * np.cos(heading) * discretization_time
        state_jacobian[1, 3] = np.sin(heading) * discretization_time
        state_jacobian[2, 3] = np.tan(steering_angle) / wheelbase * discretization_time
        state_jacobian[2, 4] = velocity * discretization_time / (wheelbase * np.cos(steering_angle) ** 2)
        input_jacobian[3, 0] = discretization_time
        input_jacobian[4, 1] = discretization_time
        return (next_state, current_input, state_jacobian, input_jacobian)

    def _run_lqr_backward_recursion(self, current_iterate: ILQRIterate, reference_trajectory: DoubleMatrix) -> ILQRInputPolicy:
        """
        Computes the locally optimal affine state feedback policy by applying dynamic programming to linear perturbation
        dynamics about a specified linearization trajectory.  We include a trust region penalty as part of the cost.
        :param current_iterate: Contains all relevant linearization information needed to compute LQR policy.
        :param reference_trajectory: The desired state trajectory we are tracking.
        :return: An affine state feedback policy - state feedback matrices and feedforward inputs found using LQR.
        """
        state_trajectory = current_iterate.state_trajectory
        input_trajectory = current_iterate.input_trajectory
        state_jacobian_trajectory = current_iterate.state_jacobian_trajectory
        input_jacobian_trajectory = current_iterate.input_jacobian_trajectory
        assert reference_trajectory.shape == state_trajectory.shape, 'The reference trajectory has incorrect shape.'
        error_state_trajectory = state_trajectory - reference_trajectory
        error_state_trajectory[:, 2] = principal_value(error_state_trajectory[:, 2])
        p_current = self._state_cost_matrix + self._state_trust_region_cost_matrix
        rho_current = self._state_cost_matrix @ error_state_trajectory[-1]
        N = len(input_trajectory)
        state_feedback_matrices = np.nan * np.ones((N, self._n_inputs, self._n_states), dtype=np.float64)
        feedforward_inputs = np.nan * np.ones((N, self._n_inputs), dtype=np.float64)
        for i in reversed(range(N)):
            A = state_jacobian_trajectory[i]
            B = input_jacobian_trajectory[i]
            u = input_trajectory[i]
            error = error_state_trajectory[i]
            inverse_matrix_term = np.linalg.inv(self._input_cost_matrix + self._input_trust_region_cost_matrix + B.T @ p_current @ B)
            state_feedback_matrix = -inverse_matrix_term @ B.T @ p_current @ A
            feedforward_input = -inverse_matrix_term @ (self._input_cost_matrix @ u + B.T @ rho_current)
            a_closed_loop = A + B @ state_feedback_matrix
            p_prior = self._state_cost_matrix + self._state_trust_region_cost_matrix + state_feedback_matrix.T @ self._input_cost_matrix @ state_feedback_matrix + state_feedback_matrix.T @ self._input_trust_region_cost_matrix @ state_feedback_matrix + a_closed_loop.T @ p_current @ a_closed_loop
            rho_prior = self._state_cost_matrix @ error + state_feedback_matrix.T @ self._input_cost_matrix @ (feedforward_input + u) + state_feedback_matrix.T @ self._input_trust_region_cost_matrix @ feedforward_input + a_closed_loop.T @ p_current @ B @ feedforward_input + a_closed_loop.T @ rho_current
            p_current = p_prior
            rho_current = rho_prior
            state_feedback_matrices[i] = state_feedback_matrix
            feedforward_inputs[i] = feedforward_input
        lqr_input_policy = ILQRInputPolicy(state_feedback_matrices=state_feedback_matrices, feedforward_inputs=feedforward_inputs)
        return lqr_input_policy

    def _update_inputs_with_policy(self, current_iterate: ILQRIterate, lqr_input_policy: ILQRInputPolicy) -> DoubleMatrix:
        """
        Used to update an iterate of iLQR by applying a perturbation input policy for local cost improvement.
        :param current_iterate: Contains the state and input trajectory about which we linearized.
        :param lqr_input_policy: Contains the LQR policy to apply.
        :return: The next input trajectory found by applying the LQR policy.
        """
        state_trajectory = current_iterate.state_trajectory
        input_trajectory = current_iterate.input_trajectory
        delta_state_trajectory = np.nan * np.ones((len(input_trajectory) + 1, self._n_states), dtype=np.float64)
        delta_state_trajectory[0] = [0.0] * self._n_states
        input_next_trajectory = np.nan * np.ones_like(input_trajectory, dtype=np.float64)
        zip_object = zip(input_trajectory, state_trajectory[:-1], state_trajectory[1:], lqr_input_policy.state_feedback_matrices, lqr_input_policy.feedforward_inputs)
        for input_idx, (input_lin, state_lin, state_lin_next, state_feedback_matrix, feedforward_input) in enumerate(zip_object):
            delta_state = delta_state_trajectory[input_idx]
            delta_input = state_feedback_matrix @ delta_state + feedforward_input
            input_perturbed = input_lin + delta_input
            state_perturbed = state_lin + delta_state
            state_perturbed[2] = principal_value(state_perturbed[2])
            state_perturbed_next, input_perturbed, _, _ = self._dynamics_and_jacobian(state_perturbed, input_perturbed)
            delta_state_next = state_perturbed_next - state_lin_next
            delta_state_next[2] = principal_value(delta_state_next[2])
            delta_state_trajectory[input_idx + 1] = delta_state_next
            input_next_trajectory[input_idx] = input_perturbed
        assert ~np.any(np.isnan(input_next_trajectory)), 'All next inputs should be valid float values.'
        return input_next_trajectory

class KinematicBicycleModel(AbstractMotionModel):
    """
    A class describing the kinematic motion model where the rear axle is the point of reference.
    """

    def __init__(self, vehicle: VehicleParameters, max_steering_angle: float=np.pi / 3, accel_time_constant: float=0.2, steering_angle_time_constant: float=0.05):
        """
        Construct KinematicBicycleModel.

        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        :param accel_time_constant: low pass filter time constant for acceleration in s
        :param steering_angle_time_constant: low pass filter time constant for steering angle in s
        """
        self._vehicle = vehicle
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """Inherited, see super class."""
        longitudinal_speed = state.dynamic_car_state.rear_axle_velocity_2d.x
        x_dot = longitudinal_speed * np.cos(state.rear_axle.heading)
        y_dot = longitudinal_speed * np.sin(state.rear_axle.heading)
        yaw_dot = longitudinal_speed * np.tan(state.tire_steering_angle) / self._vehicle.wheel_base
        return EgoStateDot.build_from_rear_axle(rear_axle_pose=StateSE2(x=x_dot, y=y_dot, heading=yaw_dot), rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_acceleration_2d, rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=state.dynamic_car_state.tire_steering_rate, time_point=state.time_point, is_in_auto_mode=True, vehicle_parameters=self._vehicle)

    def _update_commands(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """
        This function applies some first order control delay/a low pass filter to acceleration/steering.

        :param state: Ego state
        :param ideal_dynamic_state: The desired dynamic state for propagation
        :param sampling_time: The time duration to propagate for
        :return: propagating_state including updated dynamic_state
        """
        dt_control = sampling_time.time_s
        accel = state.dynamic_car_state.rear_axle_acceleration_2d.x
        steering_angle = state.tire_steering_angle
        ideal_accel_x = ideal_dynamic_state.rear_axle_acceleration_2d.x
        ideal_steering_angle = dt_control * ideal_dynamic_state.tire_steering_rate + steering_angle
        updated_accel_x = dt_control / (dt_control + self._accel_time_constant) * (ideal_accel_x - accel) + accel
        updated_steering_angle = dt_control / (dt_control + self._steering_angle_time_constant) * (ideal_steering_angle - steering_angle) + steering_angle
        updated_steering_rate = (updated_steering_angle - steering_angle) / dt_control
        dynamic_state = DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist=state.car_footprint.rear_axle_to_center_dist, rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d, rear_axle_acceleration_2d=StateVector2D(updated_accel_x, 0), tire_steering_rate=updated_steering_rate)
        propagating_state = EgoState(car_footprint=state.car_footprint, dynamic_car_state=dynamic_state, tire_steering_angle=state.tire_steering_angle, is_in_auto_mode=True, time_point=state.time_point)
        return propagating_state

    def propagate_state(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        """Inherited, see super class."""
        propagating_state = self._update_commands(state, ideal_dynamic_state, sampling_time)
        state_dot = self.get_state_dot(propagating_state)
        next_x = forward_integrate(propagating_state.rear_axle.x, state_dot.rear_axle.x, sampling_time)
        next_y = forward_integrate(propagating_state.rear_axle.y, state_dot.rear_axle.y, sampling_time)
        next_heading = forward_integrate(propagating_state.rear_axle.heading, state_dot.rear_axle.heading, sampling_time)
        next_heading = principal_value(next_heading)
        next_point_velocity_x = forward_integrate(propagating_state.dynamic_car_state.rear_axle_velocity_2d.x, state_dot.dynamic_car_state.rear_axle_velocity_2d.x, sampling_time)
        next_point_velocity_y = 0.0
        next_point_tire_steering_angle = np.clip(forward_integrate(propagating_state.tire_steering_angle, state_dot.tire_steering_angle, sampling_time), -self._max_steering_angle, self._max_steering_angle)
        next_point_angular_velocity = next_point_velocity_x * np.tan(next_point_tire_steering_angle) / self._vehicle.wheel_base
        rear_axle_accel = [state_dot.dynamic_car_state.rear_axle_velocity_2d.x, state_dot.dynamic_car_state.rear_axle_velocity_2d.y]
        angular_accel = (next_point_angular_velocity - state.dynamic_car_state.angular_velocity) / sampling_time.time_s
        return EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(next_x, next_y, next_heading), rear_axle_velocity_2d=StateVector2D(next_point_velocity_x, next_point_velocity_y), rear_axle_acceleration_2d=StateVector2D(rear_axle_accel[0], rear_axle_accel[1]), tire_steering_angle=float(next_point_tire_steering_angle), time_point=propagating_state.time_point + sampling_time, vehicle_parameters=self._vehicle, is_in_auto_mode=True, angular_vel=next_point_angular_velocity, angular_accel=angular_accel, tire_steering_rate=state_dot.tire_steering_angle)

