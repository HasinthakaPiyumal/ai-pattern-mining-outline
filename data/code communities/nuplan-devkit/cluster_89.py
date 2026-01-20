# Cluster 89

class TestLogFuturePredictor(unittest.TestCase):
    """
    Test LogFuturePredictor class.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.scenario = MockAbstractScenario()
        self.future_trajectory_sampling = TrajectorySampling(num_poses=1, time_horizon=1.0)
        self.predictor = LogFuturePredictor(self.scenario, self.future_trajectory_sampling)

    def test_compute_predicted_trajectories(self) -> None:
        """Test compute_predicted_trajectories."""
        predictor_input = get_mock_predictor_input()
        start_time = time.perf_counter()
        detections = self.predictor.compute_predictions(predictor_input)
        compute_predictions_time = time.perf_counter() - start_time
        _, input_detections = predictor_input.history.current_state
        self.assertEqual(len(detections.tracked_objects), len(input_detections.tracked_objects))
        for agent in detections.tracked_objects.get_agents():
            self.assertTrue(agent.predictions is not None)
            for prediction in agent.predictions:
                self.assertEqual(len(prediction.valid_waypoints), self.future_trajectory_sampling.num_poses)
        predictor_report = self.predictor.generate_predictor_report()
        self.assertEqual(len(predictor_report.compute_predictions_runtimes), 1)
        self.assertNotIsInstance(predictor_report, MLPredictorReport)
        self.assertAlmostEqual(predictor_report.compute_predictions_runtimes[0], compute_predictions_time, delta=0.1)

class TestILQRTracker(unittest.TestCase):
    """
    Tests the functionality of the ILQRTracker class.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.initial_time_point = TimePoint(1000000)
        self.scenario = MockAbstractScenario(initial_time_us=self.initial_time_point)
        self.trajectory = InterpolatedTrajectory(list(self.scenario.get_expert_ego_trajectory()))
        solver_params = ILQRSolverParameters(discretization_time=0.2, state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0], input_cost_diagonal_entries=[1.0, 10.0], state_trust_region_entries=[1.0] * 5, input_trust_region_entries=[1.0] * 2, max_ilqr_iterations=100, convergence_threshold=1e-06, max_solve_time=0.05, max_acceleration=3.0, max_steering_angle=np.pi / 3.0, max_steering_angle_rate=0.5, min_velocity_linearization=0.01)
        warm_start_params = ILQRWarmStartParameters(k_velocity_error_feedback=0.5, k_steering_angle_error_feedback=0.05, lookahead_distance_lateral_error=15.0, k_lateral_error=0.1, jerk_penalty_warm_start_fit=0.0001, curvature_rate_penalty_warm_start_fit=0.01)
        self.tracker = ILQRTracker(n_horizon=40, ilqr_solver=ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params))
        self.discretization_time_us = int(1000000.0 * self.tracker._ilqr_solver._solver_params.discretization_time)

    def test_track_trajectory(self) -> None:
        """Ensure that we can run a single solver call to track a trajectory."""
        current_iteration = SimulationIteration(time_point=self.initial_time_point, index=0)
        time_point_delta = TimePoint(self.discretization_time_us)
        next_iteration = SimulationIteration(time_point=self.initial_time_point + time_point_delta, index=1)
        self.tracker.track_trajectory(current_iteration=current_iteration, next_iteration=next_iteration, initial_state=self.scenario.initial_ego_state, trajectory=self.trajectory)

    def test__get_reference_trajectory(self) -> None:
        """Test reference trajectory extraction for the solver."""
        current_iteration_before_trajectory_start = SimulationIteration(time_point=self.trajectory.start_time - TimePoint(1), index=0)
        with self.assertRaises(AssertionError):
            self.tracker._get_reference_trajectory(current_iteration_before_trajectory_start, self.trajectory)
        current_iteration_after_trajectory_end = SimulationIteration(time_point=self.trajectory.end_time + TimePoint(1), index=0)
        with self.assertRaises(AssertionError):
            self.tracker._get_reference_trajectory(current_iteration_after_trajectory_end, self.trajectory)
        start_time_us = self.trajectory.start_time.time_us
        end_time_us = self.trajectory.end_time.time_us
        mid_time_us = int((start_time_us + end_time_us) / 2)
        for test_time_us in [start_time_us, mid_time_us, end_time_us]:
            expected_trajectory_length = min((end_time_us - test_time_us) // self.discretization_time_us + 1, self.tracker._n_horizon + 1)
            current_iteration = SimulationIteration(time_point=TimePoint(test_time_us), index=0)
            reference_trajectory = self.tracker._get_reference_trajectory(current_iteration, self.trajectory)
            self.assertEqual(len(reference_trajectory), expected_trajectory_length)
            first_state_reference_trajectory = reference_trajectory[0]
            first_ego_state_expected = self.trajectory.get_state_at_time(current_iteration.time_point)
            np_test.assert_allclose(first_state_reference_trajectory[0], first_ego_state_expected.rear_axle.x)
            np_test.assert_allclose(first_state_reference_trajectory[1], first_ego_state_expected.rear_axle.y)
            np_test.assert_allclose(first_state_reference_trajectory[2], first_ego_state_expected.rear_axle.heading)
            np_test.assert_allclose(first_state_reference_trajectory[3], first_ego_state_expected.dynamic_car_state.rear_axle_velocity_2d.x)
            np_test.assert_allclose(first_state_reference_trajectory[4], first_ego_state_expected.tire_steering_angle)

class TestLQRTracker(unittest.TestCase):
    """
    Tests LQR Tracker.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.initial_time_point = TimePoint(0)
        self.scenario = MockAbstractScenario(initial_time_us=self.initial_time_point)
        self.trajectory = InterpolatedTrajectory(list(self.scenario.get_expert_ego_trajectory()))
        self.sampling_time = 0.5
        self.tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, jerk_penalty=0.0001, curvature_rate_penalty=0.01, stopping_proportional_gain=0.5, stopping_velocity=0.2)

    def test_track_trajectory(self) -> None:
        """Ensure we are able to run track trajectory using LQR."""
        dynamic_state = self.tracker.track_trajectory(current_iteration=SimulationIteration(self.initial_time_point, 0), next_iteration=SimulationIteration(TimePoint(int(self.sampling_time * 1000000.0)), 1), initial_state=self.scenario.initial_ego_state, trajectory=self.trajectory)
        self.assertIsInstance(dynamic_state._rear_axle_to_center_dist, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_velocity_2d.x, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_velocity_2d.y, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_acceleration_2d.x, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_acceleration_2d.y, (int, float))
        self.assertIsInstance(dynamic_state.tire_steering_rate, (int, float))
        self.assertGreater(dynamic_state._rear_axle_to_center_dist, 0.0)
        self.assertEqual(dynamic_state.rear_axle_acceleration_2d.y, 0.0)

    def test__compute_initial_velocity_and_lateral_state(self) -> None:
        """
        This essentially checks that our projection to vehicle/Frenet frame works by reconstructing specified errors.
        """
        current_iteration = SimulationIteration(self.initial_time_point, 0)
        base_initial_state = self.trajectory.get_state_at_time(self.initial_time_point)
        base_pose_rear_axle = base_initial_state.car_footprint.rear_axle
        test_lateral_errors = [-3.0, 3.0]
        test_heading_errors = [-0.1, 0.1]
        test_longitudinal_errors = [-3.0, 3.0]
        error_product = itertools.product(test_lateral_errors, test_heading_errors, test_longitudinal_errors)
        for lateral_error, heading_error, longitudinal_error in error_product:
            theta = base_pose_rear_axle.heading
            delta_x = longitudinal_error * np.cos(theta) - lateral_error * np.sin(theta)
            delta_y = longitudinal_error * np.sin(theta) + lateral_error * np.cos(theta)
            perturbed_pose_rear_axle = StateSE2(x=base_pose_rear_axle.x + delta_x, y=base_pose_rear_axle.y + delta_y, heading=theta + heading_error)
            perturbed_car_footprint = CarFootprint.build_from_rear_axle(rear_axle_pose=perturbed_pose_rear_axle, vehicle_parameters=base_initial_state.car_footprint.vehicle_parameters)
            perturbed_initial_state = EgoState(car_footprint=perturbed_car_footprint, dynamic_car_state=base_initial_state.dynamic_car_state, tire_steering_angle=base_initial_state.tire_steering_angle, is_in_auto_mode=base_initial_state.is_in_auto_mode, time_point=base_initial_state.time_point)
            initial_velocity, initial_lateral_state_vector = self.tracker._compute_initial_velocity_and_lateral_state(current_iteration=current_iteration, initial_state=perturbed_initial_state, trajectory=self.trajectory)
            self.assertEqual(initial_velocity, base_initial_state.dynamic_car_state.rear_axle_velocity_2d.x)
            np_test.assert_allclose(initial_lateral_state_vector, [lateral_error, heading_error, base_initial_state.tire_steering_angle])

    def test__compute_reference_velocity_and_curvature_profile(self) -> None:
        """
        This test just checks functionality of computing a reference velocity / curvature profile.
        Detailed evaluation of the result is handled in test_tracker_utils and omitted here.
        """
        current_iteration = SimulationIteration(self.initial_time_point, 0)
        reference_velocity, curvature_profile = self.tracker._compute_reference_velocity_and_curvature_profile(current_iteration=current_iteration, trajectory=self.trajectory)
        tracking_horizon = self.tracker._tracking_horizon
        discretization_time = self.tracker._discretization_time
        lookahead_time_point = TimePoint(current_iteration.time_point.time_us + int(1000000.0 * tracking_horizon * discretization_time))
        expected_lookahead_ego_state = self.trajectory.get_state_at_time(lookahead_time_point)
        np_test.assert_allclose(np.sign(reference_velocity), np.sign(expected_lookahead_ego_state.dynamic_car_state.rear_axle_velocity_2d.x))
        self.assertEqual(curvature_profile.shape, (tracking_horizon,))

    def test__stopping_controller(self) -> None:
        """Test P controller for when we are coming to a stop."""
        initial_velocity = 5.0
        accel, steering_rate_cmd = self.tracker._stopping_controller(initial_velocity=initial_velocity, reference_velocity=0.5 * initial_velocity)
        self.assertLess(accel, 0.0)
        self.assertEqual(steering_rate_cmd, 0.0)
        accel, steering_rate_cmd = self.tracker._stopping_controller(initial_velocity=-initial_velocity, reference_velocity=0.0)
        self.assertGreater(accel, 0.0)
        self.assertEqual(steering_rate_cmd, 0.0)

    def test__longitudinal_lqr_controller(self) -> None:
        """Test longitudinal control for simple cases of speed above or below the reference velocity."""
        test_initial_velocities = [2.0, 6.0]
        reference_velocity = float(np.mean(test_initial_velocities))
        for initial_velocity in test_initial_velocities:
            accel_cmd = self.tracker._longitudinal_lqr_controller(initial_velocity=initial_velocity, reference_velocity=reference_velocity)
            np_test.assert_allclose(np.sign(accel_cmd), -np.sign(initial_velocity - reference_velocity))

    def test__lateral_lqr_controller_straight_road(self) -> None:
        """Test how the controller handles non-zero initial tracking error on a straight road."""
        test_velocity_profile = 5.0 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)
        test_curvature_profile = 0.0 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)
        test_lateral_errors = [-3.0, 3.0]
        for lateral_error in test_lateral_errors:
            initial_lateral_state_vector_lateral_only: npt.NDArray[np.float64] = np.array([lateral_error, 0.0, 0.0], dtype=np.float64)
            steering_rate_cmd = self.tracker._lateral_lqr_controller(initial_lateral_state_vector=initial_lateral_state_vector_lateral_only, velocity_profile=test_velocity_profile, curvature_profile=test_curvature_profile)
            np_test.assert_allclose(np.sign(steering_rate_cmd), -np.sign(lateral_error))
        test_heading_errors = [-0.1, 0.1]
        for heading_error in test_heading_errors:
            initial_lateral_state_vector_heading_only: npt.NDArray[np.float64] = np.array([0.0, heading_error, 0.0], dtype=np.float64)
            steering_rate_cmd = self.tracker._lateral_lqr_controller(initial_lateral_state_vector=initial_lateral_state_vector_heading_only, velocity_profile=test_velocity_profile, curvature_profile=test_curvature_profile)
            np_test.assert_allclose(np.sign(steering_rate_cmd), -np.sign(heading_error))

    def test__lateral_lqr_controller_curved_road(self) -> None:
        """Test how the controller handles a curved road with zero initial tracking error and zero steering angle."""
        test_velocity_profile = 5.0 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)
        test_curvature_profile = 0.1 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)
        test_initial_lateral_state_vector: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        steering_rate_cmd = self.tracker._lateral_lqr_controller(initial_lateral_state_vector=test_initial_lateral_state_vector, velocity_profile=test_velocity_profile, curvature_profile=test_curvature_profile)
        np_test.assert_allclose(np.sign(steering_rate_cmd), np.sign(test_curvature_profile[0]))

    def test__solve_one_step_lqr(self) -> None:
        """Test LQR on a simple linear system."""
        A: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        B: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        g: npt.NDArray[np.float64] = np.zeros(A.shape[0], dtype=np.float64)
        Q: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        R: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        for component_1, component_2 in itertools.product([-5.0, 5.0], [-10.0, 10.0]):
            initial_state: npt.NDArray[np.float64] = np.array([component_1, component_2], dtype=np.float64)
            solution = self.tracker._solve_one_step_lqr(initial_state=initial_state, reference_state=np.zeros_like(initial_state), Q=Q, R=R, A=A, B=B, g=g, angle_diff_indices=[])
            np_test.assert_allclose(np.sign(solution), -np.sign(initial_state))

class TestILQRSolver(unittest.TestCase):
    """
    Tests for ILQRSolver class.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        solver_params = ILQRSolverParameters(discretization_time=0.2, state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0], input_cost_diagonal_entries=[1.0, 10.0], state_trust_region_entries=[1.0] * 5, input_trust_region_entries=[1.0] * 2, max_ilqr_iterations=100, convergence_threshold=1e-06, max_solve_time=0.05, max_acceleration=3.0, max_steering_angle=np.pi / 3.0, max_steering_angle_rate=0.5, min_velocity_linearization=0.01)
        warm_start_params = ILQRWarmStartParameters(k_velocity_error_feedback=0.5, k_steering_angle_error_feedback=0.05, lookahead_distance_lateral_error=15.0, k_lateral_error=0.1, jerk_penalty_warm_start_fit=0.0001, curvature_rate_penalty_warm_start_fit=0.01)
        self.solver = ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params)
        self.discretization_time = self.solver._solver_params.discretization_time
        self.n_states = self.solver._n_states
        self.n_inputs = self.solver._n_inputs
        self.n_horizon = 40
        self.min_velocity_linearization = self.solver._solver_params.min_velocity_linearization
        self.max_acceleration = self.solver._solver_params.max_acceleration
        self.max_steering_angle = self.solver._solver_params.max_steering_angle
        self.max_steering_angle_rate = self.solver._solver_params.max_steering_angle_rate
        self.rtol = 1e-08
        self.atol = 1e-10
        self.check_if_allclose = partial(np.allclose, rtol=self.rtol, atol=self.atol)
        self.assert_allclose = partial(np_test.assert_allclose, rtol=self.rtol, atol=self.atol)
        constant_speed = 1.0
        self.reference_trajectory: DoubleMatrix = np.zeros((self.n_horizon + 1, self.n_states), dtype=np.float64)
        self.reference_trajectory[:, 0] = constant_speed * np.arange(self.n_horizon + 1) * self.discretization_time
        self.reference_trajectory[:, 3] = constant_speed
        reference_acceleration = np.diff(self.reference_trajectory[:, 3]) / self.discretization_time
        reference_steering_rate = np.diff(self.reference_trajectory[:, 4]) / self.discretization_time
        self.reference_inputs: DoubleMatrix = np.column_stack((reference_acceleration, reference_steering_rate))

    def test_solve(self) -> None:
        """
        Check that, for reasonable tuning parameters and small initial error, the tracking cost is non-increasing.
        """
        perturbed_current_state: DoubleMatrix = np.array([1.0, -1.0, 0.01, 1.0, -0.1], dtype=np.float64)
        current_state = self.reference_trajectory[0, :] + perturbed_current_state
        start_time = time.perf_counter()
        ilqr_solutions = self.solver.solve(current_state, self.reference_trajectory)
        end_time = time.perf_counter()
        print(f'Solve took {end_time - start_time} seconds for {len(ilqr_solutions)} iterations.')
        tracking_cost_history = [isol.tracking_cost for isol in ilqr_solutions]
        self.assertTrue(np.all(np.diff(tracking_cost_history) <= 0.0))

    def test__compute_tracking_cost(self) -> None:
        """Check tracking cost computation."""
        zero_input_trajectory: DoubleMatrix = np.zeros((self.n_horizon, self.n_inputs), dtype=np.float64)
        zero_state_jacobian_trajectory: DoubleMatrix = np.zeros((self.n_horizon, self.n_states, self.n_states), dtype=np.float64)
        zero_input_jacobian_trajectory: DoubleMatrix = np.zeros((self.n_horizon, self.n_states, self.n_inputs), dtype=np.float64)
        test_iterate = ILQRIterate(input_trajectory=np.zeros((self.n_horizon, self.n_inputs), dtype=np.float64), state_trajectory=self.reference_trajectory, state_jacobian_trajectory=zero_state_jacobian_trajectory, input_jacobian_trajectory=zero_input_jacobian_trajectory)
        cost_zero_error_zero_inputs = self.solver._compute_tracking_cost(iterate=test_iterate, reference_trajectory=self.reference_trajectory)
        self.assert_allclose(0.0, cost_zero_error_zero_inputs)
        for input_sign in [-1.0, 1.0]:
            costs = []
            for input_magnitude in [1.0, 5.0, 10.0, 100.0]:
                test_input_trajectory = input_sign * input_magnitude * np.ones((self.n_horizon, self.n_inputs), dtype=np.float64)
                test_iterate = ILQRIterate(input_trajectory=test_input_trajectory, state_trajectory=self.reference_trajectory, state_jacobian_trajectory=zero_state_jacobian_trajectory, input_jacobian_trajectory=zero_input_jacobian_trajectory)
                cost = self.solver._compute_tracking_cost(iterate=test_iterate, reference_trajectory=self.reference_trajectory)
                costs.append(cost)
            self.assertTrue(np.all(np.diff(costs) > 0.0))
        for error_sign in [-1.0, 1.0]:
            costs = []
            for error_magnitude in [1.0, 5.0, 10.0, 100.0]:
                test_state_trajectory = self.reference_trajectory + error_sign * error_magnitude * np.ones_like(self.reference_trajectory)
                test_iterate = ILQRIterate(input_trajectory=zero_input_trajectory, state_trajectory=test_state_trajectory, state_jacobian_trajectory=zero_state_jacobian_trajectory, input_jacobian_trajectory=zero_input_jacobian_trajectory)
                cost = self.solver._compute_tracking_cost(iterate=test_iterate, reference_trajectory=self.reference_trajectory)
                costs.append(cost)
            self.assertTrue(np.all(np.diff(costs) > 0.0))

    def test__clip_inputs(self) -> None:
        """Check that input clipping works."""
        zero_inputs: DoubleMatrix = np.zeros(self.n_inputs, dtype=np.float64)
        inputs_at_bounds: DoubleMatrix = np.array([self.max_acceleration, self.max_steering_angle_rate], dtype=np.float64)
        inputs_within_bounds = 0.5 * inputs_at_bounds
        inputs_outside_bounds = 2.0 * inputs_at_bounds
        for test_inputs, expect_same in zip([zero_inputs, inputs_within_bounds, inputs_at_bounds, inputs_outside_bounds], [True, True, True, False]):
            for input_sign in [-1.0, 1.0]:
                signed_test_inputs = input_sign * test_inputs
                clipped_inputs = self.solver._clip_inputs(signed_test_inputs)
                are_same = self.check_if_allclose(signed_test_inputs, clipped_inputs)
                self.assertEqual(are_same, expect_same)
                self.assertTrue(np.all(np.sign(signed_test_inputs) == np.sign(clipped_inputs)))

    def test__clip_steering_angle(self) -> None:
        """Check that steering angle clipping works."""
        zero_steering_angle = 0.0
        steering_angle_at_bounds = self.max_steering_angle
        steering_angle_within_bounds = 0.5 * steering_angle_at_bounds
        steering_angle_outside_bounds = 2.0 * steering_angle_at_bounds
        for test_steering_angle, expect_same in zip([zero_steering_angle, steering_angle_within_bounds, steering_angle_at_bounds, steering_angle_outside_bounds], [True, True, True, False]):
            for sign in [-1.0, 1.0]:
                signed_steering_angle = sign * test_steering_angle
                clipped_steering_angle = self.solver._clip_steering_angle(signed_steering_angle)
                are_same = self.check_if_allclose(signed_steering_angle, clipped_steering_angle)
                self.assertEqual(are_same, expect_same)
                self.assertTrue(np.all(np.sign(signed_steering_angle) == np.sign(clipped_steering_angle)))

    def test__input_warm_start(self) -> None:
        """Check first warm start generation under zero and nonzero initial tracking error."""
        test_current_state = self.reference_trajectory[0, :]
        warm_start_iterate = self.solver._input_warm_start(test_current_state, self.reference_trajectory)
        self.assert_allclose(warm_start_iterate.input_trajectory, self.reference_inputs)
        perturbed_current_state = self.reference_trajectory[0, :] + np.array([1.0, -1.0, 0.01, 1.0, -0.1], dtype=np.float64)
        perturbed_warm_start_iterate = self.solver._input_warm_start(perturbed_current_state, self.reference_trajectory)
        first_input_close = self.check_if_allclose(perturbed_warm_start_iterate.input_trajectory[0], self.reference_inputs[0])
        self.assertFalse(first_input_close)
        self.assert_allclose(perturbed_warm_start_iterate.input_trajectory[1, :], self.reference_inputs[1, :])

    def test__run_forward_dynamics_no_saturation(self) -> None:
        """Check generation of a state trajectory from current state and inputs without steering angle saturation."""
        test_current_state = self.reference_trajectory[0, :]
        current_steering_angle = test_current_state[4]
        time_to_saturation_s = (self.max_steering_angle - abs(current_steering_angle)) / self.max_steering_angle_rate
        timesteps_to_saturation = np.ceil(time_to_saturation_s / self.discretization_time).astype(int)
        assert timesteps_to_saturation >= 2, "We'd like at least two timesteps_to_saturation for the subsequent test."
        steering_rate_input = self.max_steering_angle_rate
        acceleration_input = self.max_acceleration
        test_input_trajectory: DoubleMatrix = np.ones((timesteps_to_saturation - 1, 2), dtype=np.float64)
        test_input_trajectory[:, 0] = acceleration_input
        test_input_trajectory[:, 1] = steering_rate_input
        ilqr_iterate = self.solver._run_forward_dynamics(test_current_state, test_input_trajectory)
        self.assertLess(np.amax(np.abs(ilqr_iterate.state_trajectory[:, 4])), self.max_steering_angle)
        steering_rate_unmodified = self.check_if_allclose(ilqr_iterate.input_trajectory[:, 1], test_input_trajectory[:, 1])
        self.assertTrue(steering_rate_unmodified)
        acceleration_finite_differences = np.diff(ilqr_iterate.state_trajectory[:, 3]) / self.discretization_time
        self.assert_allclose(acceleration_finite_differences, ilqr_iterate.input_trajectory[:, 0])
        steering_rate_finite_differences = np.diff(ilqr_iterate.state_trajectory[:, 4]) / self.discretization_time
        self.assert_allclose(steering_rate_finite_differences, ilqr_iterate.input_trajectory[:, 1])

    def test__run_forward_dynamics_saturation(self) -> None:
        """Check generation of a state trajectory from current state and inputs with steering angle saturation."""
        test_current_state = self.reference_trajectory[0, :]
        current_steering_angle = test_current_state[4]
        time_to_saturation_s = (self.max_steering_angle - abs(current_steering_angle)) / self.max_steering_angle_rate
        timesteps_to_saturation = np.ceil(time_to_saturation_s / self.discretization_time).astype(int)
        assert timesteps_to_saturation >= 2, "We'd like at least two timesteps_to_saturation for the subsequent test."
        if np.abs(current_steering_angle) > 0.0:
            steering_rate_input = self.max_steering_angle_rate * np.sign(current_steering_angle)
        else:
            steering_rate_input = self.max_steering_angle_rate
        acceleration_input = self.max_acceleration
        test_input_trajectory: DoubleMatrix = np.ones((timesteps_to_saturation + 1, 2), dtype=np.float64)
        test_input_trajectory[:, 0] = acceleration_input
        test_input_trajectory[:, 1] = steering_rate_input
        ilqr_iterate = self.solver._run_forward_dynamics(test_current_state, test_input_trajectory)
        self.assertEqual(np.abs(ilqr_iterate.state_trajectory[-2, 4]), self.max_steering_angle)
        self.assertEqual(np.amax(np.abs(ilqr_iterate.state_trajectory[:, 4])), self.max_steering_angle)
        steering_rate_unmodified = self.check_if_allclose(ilqr_iterate.input_trajectory[:, 1], test_input_trajectory[:, 1])
        self.assertFalse(steering_rate_unmodified)
        acceleration_finite_differences = np.diff(ilqr_iterate.state_trajectory[:, 3]) / self.discretization_time
        steering_rate_finite_differences = np.diff(ilqr_iterate.state_trajectory[:, 4]) / self.discretization_time
        self.assert_allclose(acceleration_finite_differences, ilqr_iterate.input_trajectory[:, 0])
        self.assert_allclose(steering_rate_finite_differences, ilqr_iterate.input_trajectory[:, 1])

    def test_dynamics_and_jacobian_constraints(self) -> None:
        """Check application of constraints in dynamics."""
        test_state_in_bounds: DoubleMatrix = np.array([0.0, 0.0, 0.1, 1.0, -0.01], dtype=np.float64)
        test_state_at_bounds: DoubleMatrix = np.copy(test_state_in_bounds)
        test_state_at_bounds[4] = self.max_steering_angle
        input_at_bounds: DoubleMatrix = np.array([self.max_acceleration, self.max_steering_angle_rate], dtype=np.float64)
        test_input_in_bounds = 0.5 * input_at_bounds
        test_input_outside_bounds = 2.0 * input_at_bounds
        test_cases_dict = {}
        test_cases_dict['state_in_bounds_input_in_bounds'] = {'state': test_state_in_bounds, 'input': test_input_in_bounds, 'expect_acceleration_modified': False, 'expect_steering_rate_modified': False}
        test_cases_dict['state_at_bounds_input_in_bounds'] = {'state': test_state_at_bounds, 'input': test_input_in_bounds, 'expect_acceleration_modified': False, 'expect_steering_rate_modified': True}
        test_cases_dict['state_in_bounds_input_outside_bounds'] = {'state': test_state_in_bounds, 'input': test_input_outside_bounds, 'expect_acceleration_modified': True, 'expect_steering_rate_modified': True}
        test_cases_dict['state_at_bounds_input_outside_bounds'] = {'state': test_state_at_bounds, 'input': test_input_outside_bounds, 'expect_acceleration_modified': True, 'expect_steering_rate_modified': True}
        for test_name, test_config in test_cases_dict.items():
            next_state, applied_input, _, _ = self.solver._dynamics_and_jacobian(test_config['state'], test_config['input'])
            self.assertEqual(test_config['expect_acceleration_modified'], not self.check_if_allclose(applied_input[0], test_config['input'][0]))
            self.assertEqual(test_config['expect_steering_rate_modified'], not self.check_if_allclose(applied_input[1], test_config['input'][1]))
            self.assertLessEqual(np.abs(next_state[4]), self.max_steering_angle)

    def test_dynamics_and_jacobian_linearization(self) -> None:
        """
        Check that Jacobian computation makes sense by comparison to finite difference estimate.
        Also check that the minimum velocity linearization is triggered for the Jacobian computation.
        """
        test_state: DoubleMatrix = np.array([0.0, 0.0, 0.1, 1.0, -0.01], dtype=np.float64)
        test_input: DoubleMatrix = np.array([1.0, 0.01], dtype=np.float64)
        epsilon = 1e-06
        _, applied_input, state_jacobian, input_jacobian = self.solver._dynamics_and_jacobian(test_state, test_input)
        self.assert_allclose(test_input, applied_input)
        state_jacobian_finite_differencing = np.zeros_like(state_jacobian)
        for state_idx in range(self.n_states):
            epsilon_array = epsilon * np.array([x == state_idx for x in range(self.n_states)], dtype=np.float64)
            next_state_plus, _, _, _ = self.solver._dynamics_and_jacobian(test_state + epsilon_array, test_input)
            next_state_minus, _, _, _ = self.solver._dynamics_and_jacobian(test_state - epsilon_array, test_input)
            state_jacobian_finite_differencing[:, state_idx] = (next_state_plus - next_state_minus) / (2.0 * epsilon)
        self.assert_allclose(state_jacobian, state_jacobian_finite_differencing)
        input_jacobian_finite_differencing = np.zeros_like(input_jacobian)
        for input_idx in range(self.n_inputs):
            epsilon_array = epsilon * np.array([x == input_idx for x in range(self.n_inputs)], dtype=np.float64)
            next_state_plus, _, _, _ = self.solver._dynamics_and_jacobian(test_state, test_input + epsilon_array)
            next_state_minus, _, _, _ = self.solver._dynamics_and_jacobian(test_state, test_input - epsilon_array)
            input_jacobian_finite_differencing[:, input_idx] = (next_state_plus - next_state_minus) / (2.0 * epsilon)
        self.assert_allclose(input_jacobian, input_jacobian_finite_differencing)
        test_state_stopped: DoubleMatrix = np.copy(test_state)
        test_state_stopped[3] = 0.0
        _, _, state_jacobian_stopped, _ = self.solver._dynamics_and_jacobian(test_state_stopped, test_input)
        velocity_inferred_stopped = np.hypot(state_jacobian_stopped[0, 2], state_jacobian_stopped[1, 2]) / self.discretization_time
        with np_test.assert_raises(AssertionError):
            self.assert_allclose(velocity_inferred_stopped, test_state_stopped[3])
        self.assert_allclose(velocity_inferred_stopped, self.min_velocity_linearization)

    def test__run_lqr_backward_recursion(self) -> None:
        """Check some properties of the LQR input policy."""
        test_current_state = self.reference_trajectory[0]
        test_input_trajectory = self.reference_inputs
        input_perturbation: DoubleMatrix = np.array([-0.1 * self.max_acceleration, 0.1 * self.max_steering_angle], dtype=np.float64)
        test_input_trajectory[0] += input_perturbation
        ilqr_iterate = self.solver._run_forward_dynamics(test_current_state, test_input_trajectory)
        ilqr_input_policy = self.solver._run_lqr_backward_recursion(current_iterate=ilqr_iterate, reference_trajectory=self.reference_trajectory)
        state_feedback_matrices = ilqr_input_policy.state_feedback_matrices
        feedforward_inputs = ilqr_input_policy.feedforward_inputs
        self.assertTrue(np.all(np.sign(feedforward_inputs[0]) == -np.sign(input_perturbation)))
        self.assertEqual(state_feedback_matrices.shape, (self.n_horizon, self.n_inputs, self.n_states))

    def test__update_inputs_with_policy(self) -> None:
        """Check how application of a specified input policy affects the next input trajectory."""
        ilqr_iterate = self.solver._run_forward_dynamics(self.reference_trajectory[0], self.reference_inputs)
        input_trajectory = ilqr_iterate.input_trajectory
        state_trajectory = ilqr_iterate.state_trajectory
        angle_distance_to_saturation = self.max_steering_angle - np.amax(np.abs(state_trajectory[:, 4]))
        test_feedforward_steering_rate = min(0.5 * angle_distance_to_saturation / (self.n_horizon * self.discretization_time), self.max_steering_angle_rate)
        feedforward_inputs = np.ones_like(input_trajectory)
        feedforward_inputs[:, 0] = self.max_acceleration
        feedforward_inputs[:, 1] = test_feedforward_steering_rate
        feedforward_inputs[1::2] *= -1.0
        state_feedback_matrices = np.zeros((len(feedforward_inputs), self.n_inputs, self.n_states))
        lqr_input_policy = ILQRInputPolicy(state_feedback_matrices=state_feedback_matrices, feedforward_inputs=feedforward_inputs)
        input_next_trajectory = self.solver._update_inputs_with_policy(current_iterate=ilqr_iterate, lqr_input_policy=lqr_input_policy)
        self.assert_allclose(feedforward_inputs, input_next_trajectory)
        test_input_perturbation = [self.max_acceleration, test_feedforward_steering_rate]
        feedforward_inputs = np.zeros_like(input_trajectory)
        feedforward_inputs[0, :] = test_input_perturbation
        state_feedback_matrices = np.zeros((len(feedforward_inputs), self.n_inputs, self.n_states))
        state_feedback_matrices[:, 0, 3] = -1.0
        state_feedback_matrices[:, 1, 4] = -1.0
        lqr_input_policy = ILQRInputPolicy(state_feedback_matrices=state_feedback_matrices, feedforward_inputs=feedforward_inputs)
        input_next_trajectory = self.solver._update_inputs_with_policy(current_iterate=ilqr_iterate, lqr_input_policy=lqr_input_policy)
        first_delta_input = input_next_trajectory[0, :] - input_trajectory[0, :]
        second_delta_input = input_next_trajectory[1, :] - input_trajectory[1, :]
        self.assertTrue(np.all(np.sign(first_delta_input) == -np.sign(second_delta_input)))

class TestLogFuturePlanner(unittest.TestCase):
    """
    Test LogFuturePlanner class
    """

    def _get_mock_planner_input(self) -> PlannerInput:
        """
        Returns a mock PlannerInput for testing.
        :return: PlannerInput.
        """
        buffer = SimulationHistoryBuffer.initialize_from_list(1, [self.scenario.initial_ego_state], [self.scenario.initial_tracked_objects])
        return PlannerInput(iteration=SimulationIteration(TimePoint(0), 0), history=buffer, traffic_light_data=None)

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.scenario = MockAbstractScenario(number_of_future_iterations=20)
        self.num_poses = 10
        self.future_time_horizon = 5
        self.planner = LogFuturePlanner(self.scenario, self.num_poses, self.future_time_horizon)

    def test_name(self) -> None:
        """Tests planner name is set correctly."""
        result = self.planner.name()
        self.assertEqual(result, 'LogFuturePlanner')

    @patch('nuplan.planning.simulation.planner.log_future_planner.DetectionsTracks')
    def test_observation_type(self, mock_detection_tracks: Mock) -> None:
        """Tests observation type is set correctly."""
        result = self.planner.observation_type()
        self.assertEqual(result, mock_detection_tracks)

    def test_compute_trajectory(self) -> None:
        """Test compute_trajectory"""
        planner_input = self._get_mock_planner_input()
        start_time = time.perf_counter()
        result = self.planner.compute_trajectory(planner_input)
        compute_trajectory_time = time.perf_counter() - start_time
        self.assertEqual(len(result.get_sampled_trajectory()), self.num_poses + 1)
        planner_report = self.planner.generate_planner_report()
        self.assertEqual(len(planner_report.compute_trajectory_runtimes), 1)
        self.assertNotIsInstance(planner_report, MLPlannerReport)
        self.assertAlmostEqual(planner_report.compute_trajectory_runtimes[0], compute_trajectory_time, delta=0.1)

    def test_compute_trajectory_fail_extraction_previous_available(self) -> None:
        """
        Test compute_trajectory when future ego extraction from scenario fails and planner should fall back on previous
        trajectory.
        """
        previous_trajectory = Mock()
        self.planner._trajectory = previous_trajectory
        planner_input = self._get_mock_planner_input()
        with patch.object(self.scenario, 'get_ego_future_trajectory', side_effect=AssertionError):
            result = self.planner.compute_trajectory(planner_input)
        self.assertEqual(result, previous_trajectory)

    def test_compute_trajectory_fail_extraction_no_previous(self) -> None:
        """
        Test compute_trajectory when future ego extraction from scenario fails and there is no prior trajectory
        to fall back on.
        """
        self.planner._trajectory = None
        planner_input = self._get_mock_planner_input()
        with patch.object(self.scenario, 'get_ego_future_trajectory', side_effect=AssertionError):
            with self.assertRaises(RuntimeError):
                _ = self.planner.compute_trajectory(planner_input)

class TestSimulationHistoryBuffer(unittest.TestCase):
    """Test suite for SimulationHistoryBuffer"""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = MockAbstractScenario(number_of_past_iterations=20)
        self.buffer_size = 10

    def test_initialize_with_box(self) -> None:
        """Test the initialize function"""
        tracks_observation = TracksObservation(self.scenario)
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(buffer_size=self.buffer_size, scenario=self.scenario, observation_type=tracks_observation.observation_type())
        self.assertEqual(len(history_buffer), self.buffer_size)

    def test_initialize_with_lidar_pc(self) -> None:
        """Test the initialize function"""
        lidar_pc_observation = LidarPcObservation(self.scenario)
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(buffer_size=self.buffer_size, scenario=self.scenario, observation_type=lidar_pc_observation.observation_type())
        self.assertEqual(len(history_buffer), self.buffer_size)

    def test_initialize_from_list(self) -> None:
        """Test the initialization from lists"""
        history_buffer = SimulationHistoryBuffer.initialize_from_list(buffer_size=self.buffer_size, ego_states=[self.scenario.initial_ego_state], observations=[self.scenario.initial_tracked_objects], sample_interval=0.05)
        self.assertEqual(len(history_buffer), 1)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state])
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects])

    def test_append(self) -> None:
        """Test the append function"""
        history_buffer = SimulationHistoryBuffer(ego_state_buffer=deque([Mock()], maxlen=1), observations_buffer=deque([Mock()], maxlen=1))
        history_buffer.append(self.scenario.initial_ego_state, self.scenario.initial_tracked_objects)
        self.assertEqual(len(history_buffer), 1)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state])
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects])

    def test_extend(self) -> None:
        """Test the extend function"""
        history_buffer = SimulationHistoryBuffer(ego_state_buffer=deque([Mock()], maxlen=2), observations_buffer=deque([Mock()], maxlen=2))
        history_buffer.extend([self.scenario.initial_ego_state] * 2, [self.scenario.initial_tracked_objects] * 2)
        self.assertEqual(len(history_buffer), 2)
        self.assertEqual(history_buffer.ego_states, [self.scenario.initial_ego_state] * 2)
        self.assertEqual(history_buffer.observations, [self.scenario.initial_tracked_objects] * 2)

class UrbanDriverOpenLoopModel(TorchModuleWrapper):
    """
    Vector-based model that uses PointNet-based subgraph layers for collating loose collections of vectorized inputs
    into local feature descriptors to be used as input to a global Transformer.

    Adapted from L5Kit's implementation of "Urban Driver: Learning to Drive from Real-world Demonstrations
    Using Policy Gradients":
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py
    Only the open-loop  version of the model is here represented, with slight modifications to fit the nuPlan framework.
    Changes:
        1. Use nuPlan features from NuPlanScenario
        2. Format model for using pytorch_lightning
    """

    def __init__(self, model_params: UrbanDriverOpenLoopModelParams, feature_params: UrbanDriverOpenLoopModelFeatureParams, target_params: UrbanDriverOpenLoopModelTargetParams):
        """
        Initialize UrbanDriverOpenLoop model.
        :param model_params: internal model parameters.
        :param feature_params: agent and map feature parameters.
        :param target_params: target parameters.
        """
        super().__init__(feature_builders=[VectorSetMapFeatureBuilder(map_features=feature_params.map_features, max_elements=feature_params.max_elements, max_points=feature_params.max_points, radius=feature_params.vector_set_map_feature_radius, interpolation_method=feature_params.interpolation_method), GenericAgentsFeatureBuilder(feature_params.agent_features, feature_params.past_trajectory_sampling)], target_builders=[EgoTrajectoryTargetBuilder(target_params.future_trajectory_sampling)], future_trajectory_sampling=target_params.future_trajectory_sampling)
        self._model_params = model_params
        self._feature_params = feature_params
        self._target_params = target_params
        self.feature_embedding = nn.Linear(self._feature_params.feature_dimension, self._model_params.local_embedding_size)
        self.positional_embedding = SinusoidalPositionalEmbedding(self._model_params.local_embedding_size)
        self.type_embedding = TypeEmbedding(self._model_params.global_embedding_size, self._feature_params.feature_types)
        self.local_subgraph = LocalSubGraph(num_layers=self._model_params.num_subgraph_layers, dim_in=self._model_params.local_embedding_size)
        if self._model_params.global_embedding_size != self._model_params.local_embedding_size:
            self.global_from_local = nn.Linear(self._model_params.local_embedding_size, self._model_params.global_embedding_size)
        num_timesteps = self.future_trajectory_sampling.num_poses
        self.global_head = MultiheadAttentionGlobalHead(self._model_params.global_embedding_size, num_timesteps, self._target_params.num_output_features // num_timesteps, dropout=self._model_params.global_head_dropout)

    def extract_agent_features(self, ego_agent_features: GenericAgents, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ego and agent features into format expected by network and build accompanying availability matrix.
        :param ego_agent_features: agent features to be extracted (ego + other agents)
        :param batch_size: number of samples in batch to extract
        :return:
            agent_features: <torch.FloatTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element, feature_dimension>. Stacked ego, agent, and map features.
            agent_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        agent_features = []
        agent_avails = []
        for sample_idx in range(batch_size):
            sample_ego_feature = ego_agent_features.ego[sample_idx][..., :min(self._feature_params.ego_dimension, self._feature_params.feature_dimension)].unsqueeze(0)
            if min(self._feature_params.ego_dimension, GenericAgents.ego_state_dim()) < self._feature_params.feature_dimension:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.feature_dimension, dim=2)
            sample_ego_avails = torch.ones(sample_ego_feature.shape[0], sample_ego_feature.shape[1], dtype=torch.bool, device=sample_ego_feature.device)
            sample_ego_feature = torch.flip(sample_ego_feature, dims=[1])
            sample_ego_feature = sample_ego_feature[:, :self._feature_params.total_max_points, ...]
            sample_ego_avails = sample_ego_avails[:, :self._feature_params.total_max_points, ...]
            if sample_ego_feature.shape[1] < self._feature_params.total_max_points:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.total_max_points, dim=1)
                sample_ego_avails = pad_avails(sample_ego_avails, self._feature_params.total_max_points, dim=1)
            sample_features = [sample_ego_feature]
            sample_avails = [sample_ego_avails]
            for feature_name in self._feature_params.agent_features:
                if ego_agent_features.has_agents(feature_name, sample_idx):
                    sample_agent_features = torch.permute(ego_agent_features.agents[feature_name][sample_idx], (1, 0, 2))
                    sample_agent_features = sample_agent_features[..., :min(self._feature_params.agent_dimension, self._feature_params.feature_dimension)]
                    if min(self._feature_params.agent_dimension, GenericAgents.agents_states_dim()) < self._feature_params.feature_dimension:
                        sample_agent_features = pad_polylines(sample_agent_features, self._feature_params.feature_dimension, dim=2)
                    sample_agent_avails = torch.ones(sample_agent_features.shape[0], sample_agent_features.shape[1], dtype=torch.bool, device=sample_agent_features.device)
                    sample_agent_features = torch.flip(sample_agent_features, dims=[1])
                    sample_agent_features = sample_agent_features[:, :self._feature_params.total_max_points, ...]
                    sample_agent_avails = sample_agent_avails[:, :self._feature_params.total_max_points, ...]
                    if sample_agent_features.shape[1] < self._feature_params.total_max_points:
                        sample_agent_features = pad_polylines(sample_agent_features, self._feature_params.total_max_points, dim=1)
                        sample_agent_avails = pad_avails(sample_agent_avails, self._feature_params.total_max_points, dim=1)
                    sample_agent_features = sample_agent_features[:self._feature_params.max_agents, ...]
                    sample_agent_avails = sample_agent_avails[:self._feature_params.max_agents, ...]
                    if sample_agent_features.shape[0] < self._feature_params.max_agents:
                        sample_agent_features = pad_polylines(sample_agent_features, self._feature_params.max_agents, dim=0)
                        sample_agent_avails = pad_avails(sample_agent_avails, self._feature_params.max_agents, dim=0)
                else:
                    sample_agent_features = torch.zeros(self._feature_params.max_agents, self._feature_params.total_max_points, self._feature_params.feature_dimension, dtype=torch.float32, device=sample_ego_feature.device)
                    sample_agent_avails = torch.zeros(self._feature_params.max_agents, self._feature_params.total_max_points, dtype=torch.bool, device=sample_agent_features.device)
                sample_features.append(sample_agent_features)
                sample_avails.append(sample_agent_avails)
            sample_features = torch.cat(sample_features, dim=0)
            sample_avails = torch.cat(sample_avails, dim=0)
            agent_features.append(sample_features)
            agent_avails.append(sample_avails)
        agent_features = torch.stack(agent_features)
        agent_avails = torch.stack(agent_avails)
        return (agent_features, agent_avails)

    def extract_map_features(self, vector_set_map_data: VectorSetMap, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract map features into format expected by network and build accompanying availability matrix.
        :param vector_set_map_data: VectorSetMap features to be extracted
        :param batch_size: number of samples in batch to extract
        :return:
            map_features: <torch.FloatTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element, feature_dimension>. Stacked map features.
            map_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        map_features = []
        map_avails = []
        for sample_idx in range(batch_size):
            sample_map_features = []
            sample_map_avails = []
            for feature_name in self._feature_params.map_features:
                coords = vector_set_map_data.coords[feature_name][sample_idx]
                tl_data = vector_set_map_data.traffic_light_data[feature_name][sample_idx] if feature_name in vector_set_map_data.traffic_light_data else None
                avails = vector_set_map_data.availabilities[feature_name][sample_idx]
                if tl_data is not None:
                    coords = torch.cat((coords, tl_data), dim=2)
                coords = coords[:, :self._feature_params.total_max_points, ...]
                avails = avails[:, :self._feature_params.total_max_points]
                if coords.shape[1] < self._feature_params.total_max_points:
                    coords = pad_polylines(coords, self._feature_params.total_max_points, dim=1)
                    avails = pad_avails(avails, self._feature_params.total_max_points, dim=1)
                coords = coords[..., :self._feature_params.feature_dimension]
                if coords.shape[2] < self._feature_params.feature_dimension:
                    coords = pad_polylines(coords, self._feature_params.feature_dimension, dim=2)
                sample_map_features.append(coords)
                sample_map_avails.append(avails)
            map_features.append(torch.cat(sample_map_features))
            map_avails.append(torch.cat(sample_map_avails))
        map_features = torch.stack(map_features)
        map_avails = torch.stack(map_avails)
        return (map_features, map_avails)

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        vector_set_map_data = cast(VectorSetMap, features['vector_set_map'])
        ego_agent_features = cast(GenericAgents, features['generic_agents'])
        batch_size = ego_agent_features.batch_size
        agent_features, agent_avails = self.extract_agent_features(ego_agent_features, batch_size)
        map_features, map_avails = self.extract_map_features(vector_set_map_data, batch_size)
        features = torch.cat([agent_features, map_features], dim=1)
        avails = torch.cat([agent_avails, map_avails], dim=1)
        feature_embedding = self.feature_embedding(features)
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2)
        invalid_mask = ~avails
        invalid_polys = invalid_mask.all(-1)
        embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding)
        if hasattr(self, 'global_from_local'):
            embeddings = self.global_from_local(embeddings)
        embeddings = F.normalize(embeddings, dim=-1) * self._model_params.global_embedding_size ** 0.5
        embeddings = embeddings.transpose(0, 1)
        type_embedding = self.type_embedding(batch_size, self._feature_params.max_agents, self._feature_params.agent_features, self._feature_params.map_features, self._feature_params.max_elements, device=features.device).transpose(0, 1)
        if self._feature_params.disable_agents:
            invalid_polys[:, 1:1 + self._feature_params.max_agents * len(self._feature_params.agent_features)] = 1
        if self._feature_params.disable_map:
            invalid_polys[:, 1 + self._feature_params.max_agents * len(self._feature_params.agent_features):] = 1
        invalid_polys[:, 0] = 0
        outputs, attns = self.global_head(embeddings, type_embedding, invalid_polys)
        return {'trajectory': Trajectory(data=convert_predictions_to_trajectory(outputs))}

def pad_polylines(polylines: torch.Tensor, pad_to: int, dim: int) -> torch.Tensor:
    """
    Copied from L5Kit's implementation `pad_points`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/common.py.
    Changes:
        1. Change function name `pad_points` to `pad_polylines`
        2. Add dimension checking and adjust output dimension

    Pad vectors to 'pad_to' size. Dimensions are:
    N: number of elements (polylines)
    P: number of points
    F: number of features
    :param polylines: Polylines to be padded, should be (N,P,F) and we're padding dim.
    :param pad_to: Number of elements, points, or features.
    :param dim: Dimension at which to apply padding.
    :return: The padded polylines (N,P,F).
    """
    num_els, num_points, num_feats = polylines.shape
    if dim == 0 or dim == -3:
        num_els = pad_to - num_els
    elif dim == 1 or dim == -2:
        num_points = pad_to - num_points
    elif dim == 2 or dim == -1:
        num_feats = pad_to - num_feats
    else:
        raise ValueError(dim)
    pad = torch.zeros(num_els, num_points, num_feats, dtype=polylines.dtype, device=polylines.device)
    return torch.cat([polylines, pad], dim=dim)

def pad_avails(avails: torch.Tensor, pad_to: int, dim: int) -> torch.Tensor:
    """
    Copied from L5Kit's implementation `pad_avail`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/common.py.
    Changes:
        1. Change function name `pad_avail` to `pad_avails`
        2. Add dimension checking and adjust output dimension

    Pad vectors to 'pad_to' size. Dimensions are:
    N: number of elements (polylines)
    P: number of points
    :param avails: Availabilities to be padded, should be (N,P) and we're padding dim.
    :param pad_to: Number of elements or points.
    :param dim: Dimension at which to apply padding.
    :return: The padded polyline availabilities (N,P).
    """
    num_els, num_points = avails.shape
    if dim == 0 or dim == -2:
        num_els = pad_to - num_els
    elif dim == 1 or dim == -1:
        num_points = pad_to - num_points
    else:
        raise ValueError(dim)
    pad = torch.zeros(num_els, num_points, dtype=avails.dtype, device=avails.device)
    return torch.cat([avails, pad], dim=dim)

class TestUrbanDriverOpenLoopUtils(unittest.TestCase):
    """Test UrbanDriverOpenLoop utils functions."""

    def test_pad_avails(self) -> None:
        """Test padding availability masks."""
        num_elements = 10
        num_points = 20
        input = torch.ones((num_elements, num_points), dtype=torch.bool)
        elements_padded = pad_avails(input, 20, 0)
        self.assertEqual(elements_padded.dtype, torch.bool)
        self.assertEqual(elements_padded.shape, (20, num_points))
        torch.testing.assert_allclose(elements_padded[10:, :], torch.zeros((10, num_points), dtype=torch.bool))
        points_padded = pad_avails(input, 30, 1)
        self.assertEqual(points_padded.dtype, torch.bool)
        self.assertEqual(points_padded.shape, (num_elements, 30))
        torch.testing.assert_allclose(points_padded[:, 20:], torch.zeros((num_elements, 10), dtype=torch.bool))

    def test_pad_polylines(self) -> None:
        """Test padding polyline features."""
        num_elements = 10
        num_points = 20
        num_features = 2
        input = torch.ones((num_elements, num_points, num_features), dtype=torch.float32)
        elements_padded = pad_polylines(input, 20, 0)
        self.assertEqual(elements_padded.dtype, torch.float32)
        self.assertEqual(elements_padded.shape, (20, num_points, num_features))
        torch.testing.assert_allclose(elements_padded[10:, :, :], torch.zeros((10, num_points, num_features), dtype=torch.float32))
        points_padded = pad_polylines(input, 30, 1)
        self.assertEqual(points_padded.dtype, torch.float32)
        self.assertEqual(points_padded.shape, (num_elements, 30, num_features))
        torch.testing.assert_allclose(points_padded[:, 20:, :], torch.zeros((num_elements, 10, num_features), dtype=torch.float32))
        features_padded = pad_polylines(input, 3, 2)
        self.assertEqual(features_padded.dtype, torch.float32)
        self.assertEqual(features_padded.shape, (num_elements, num_points, 3))
        torch.testing.assert_allclose(features_padded[:, :, 2:], torch.zeros((num_elements, num_points, 1), dtype=torch.float32))

class TestUrbanDriverOpenLoop(unittest.TestCase):
    """Test UrbanDriverOpenLoopModel model."""

    def setUp(self) -> None:
        """Set up the test."""
        self.model_params = UrbanDriverOpenLoopModelParams(local_embedding_size=256, global_embedding_size=256, num_subgraph_layers=3, global_head_dropout=0.0)
        self.feature_params = UrbanDriverOpenLoopModelFeatureParams(feature_types={'NONE': -1, 'EGO': 0, 'VEHICLE': 1, 'BICYCLE': 2, 'PEDESTRIAN': 3, 'LANE': 4, 'STOP_LINE': 5, 'CROSSWALK': 6, 'LEFT_BOUNDARY': 7, 'RIGHT_BOUNDARY': 8, 'ROUTE_LANES': 9}, total_max_points=20, feature_dimension=8, agent_features=['VEHICLE', 'BICYCLE', 'PEDESTRIAN'], ego_dimension=3, agent_dimension=8, max_agents=30, past_trajectory_sampling=TrajectorySampling(time_horizon=2.0, num_poses=4), map_features=['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES'], max_elements={'LANE': 30, 'LEFT_BOUNDARY': 30, 'RIGHT_BOUNDARY': 30, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 30}, max_points={'LANE': 20, 'LEFT_BOUNDARY': 20, 'RIGHT_BOUNDARY': 20, 'STOP_LINE': 20, 'CROSSWALK': 20, 'ROUTE_LANES': 20}, vector_set_map_feature_radius=35, interpolation_method='linear', disable_map=False, disable_agents=False)
        self.target_params = UrbanDriverOpenLoopModelTargetParams(num_output_features=36, future_trajectory_sampling=TrajectorySampling(time_horizon=6.0, num_poses=12))

    def _build_model(self) -> UrbanDriverOpenLoopModel:
        """
        Creates a new instance of a UrbanDriverOpenLoop with some default parameters.
        """
        model = UrbanDriverOpenLoopModel(self.model_params, self.feature_params, self.target_params)
        return model

    def _build_input_features(self, device: torch.device, include_agents: bool) -> FeaturesType:
        """
        Creates a set of input features for use with unit testing.
        :param device: The device on which to create the tensors.
        :param include_agents: If true, the generated input features will have agents.
            If not, then there will be no agents in the agents feature.
        :return: FeaturesType to be consumed by the model
        """
        num_frames = 5
        num_agents = num_frames if include_agents else 0
        coords: Dict[str, List[torch.Tensor]] = dict()
        traffic_light_data: Dict[str, List[torch.Tensor]] = dict()
        availabilities: Dict[str, List[torch.BoolTensor]] = dict()
        for feature_name in self.feature_params.map_features:
            coords[feature_name] = [torch.zeros((self.feature_params.max_elements[feature_name], self.feature_params.max_points[feature_name], VectorSetMap.coord_dim()), dtype=torch.float32, device=device)]
            availabilities[feature_name] = [torch.ones((self.feature_params.max_elements[feature_name], self.feature_params.max_points[feature_name]), dtype=torch.bool, device=device)]
        traffic_light_data['LANE'] = [torch.zeros((self.feature_params.max_elements['LANE'], self.feature_params.max_points['LANE'], VectorSetMap.traffic_light_status_dim()), dtype=torch.float32, device=device)]
        vector_set_map_feature = VectorSetMap(coords=coords, traffic_light_data=traffic_light_data, availabilities=availabilities)
        ego_agents = [torch.zeros((num_frames, GenericAgents.ego_state_dim()), dtype=torch.float32, device=device)]
        agent_agents = {feature_name: [torch.zeros((num_frames, num_agents, GenericAgents.agents_states_dim()), dtype=torch.float32, device=device)] for feature_name in self.feature_params.agent_features}
        generic_agents_feature = GenericAgents(ego=ego_agents, agents=agent_agents)
        return {'vector_set_map': vector_set_map_feature, 'generic_agents': generic_agents_feature}

    def _find_free_port(self) -> int:
        """
        Finds a free port to use for gloo server.
        :return: A port not in use.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            address, port = s.getsockname()
            return int(port)

    def _init_distributed_process_group(self) -> None:
        """
        Sets up the torch distributed processing server.
        :param port: The port to use for the gloo server.
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self._find_free_port())
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        torch.distributed.init_process_group(backend='gloo')

    def _assert_valid_output(self, model_output: TargetsType) -> None:
        """
        Validates that the output from the model has the correct keys and that the tensor is of the correct type.
        :param model_output: The output from the model.
        """
        self.assertTrue('trajectory' in model_output)
        self.assertTrue(isinstance(model_output['trajectory'], Trajectory))
        predicted_trajectory: Trajectory = model_output['trajectory']
        self.assertIsNotNone(predicted_trajectory.data)

    def _perform_backprop_step(self, optimizer: torch.optim.Optimizer, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], predictions: TargetsType) -> None:
        """
        Performs a backpropagation step.
        :param optimizer: The optimizer to use for training.
        :param loss_function: The loss function to use.
        :param predictions: The output from the model.
        """
        loss = loss_function(predictions['trajectory'].data, torch.zeros_like(predictions['trajectory'].data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_backprop(self) -> None:
        """
        Tests that the UrbanDriverOpenLoop model can train with DDP.
        This test was developed in response to an error related to zero agent input
        """
        self._init_distributed_process_group()
        device = torch.device('cpu')
        model = self._build_model().to(device)
        ddp_model = DDP(model, device_ids=None, output_device=None)
        optimizer = torch.optim.RMSprop(ddp_model.parameters())
        loss_function = torch.nn.MSELoss()
        num_epochs = 3
        for _ in range(num_epochs):
            for include_agents in [True, False]:
                input_features = self._build_input_features(device, include_agents=include_agents)
                predictions = ddp_model.forward(input_features)
                self._assert_valid_output(predictions)
                self._perform_backprop_step(optimizer, loss_function, predictions)

class TestScenarioScoringCallback(unittest.TestCase):
    """Test scenario scoring callback"""

    def setUp(self) -> None:
        """Set up test case."""
        self.output_dir = tempfile.TemporaryDirectory()
        preprocessor = Mock()
        preprocessor.compute_features.side_effect = mock_compute_features
        self.mock_scenarios = [MockAbstractScenario(mission_goal=StateSE2(x=1.0, y=0.0, heading=0.0)), MockAbstractScenario(mission_goal=StateSE2(x=0.0, y=0.0, heading=0.0))]
        self.scenario_time_stamp = self.mock_scenarios[0]._initial_time_us
        mock_scenario_dataset = ScenarioDataset(scenarios=self.mock_scenarios, feature_preprocessor=preprocessor)
        mock_datamodule = Mock()
        mock_datamodule.val_dataloader().dataset = mock_scenario_dataset
        self.trainer = Mock()
        self.trainer.datamodule = mock_datamodule
        self.trainer.current_epoch = 1
        mock_objective = Mock()
        mock_objective.compute.side_effect = mock_compute_objective
        self.pl_module = Mock()
        self.pl_module.device = 'cpu'
        self.pl_module.side_effect = mock_predict
        self.pl_module.objectives = [mock_objective]
        scenario_converter = ScenarioSceneConverter(ego_trajectory_horizon=1, ego_trajectory_poses=2)
        self.callback = ScenarioScoringCallback(scene_converter=scenario_converter, num_store=1, frequency=1, output_dir=self.output_dir.name)
        self.callback._initialize_dataloaders(self.trainer.datamodule)

    def test_initialize_dataloaders(self) -> None:
        """
        Test callback dataloader initialization.
        """
        invalid_datamodule = Mock()
        invalid_datamodule.val_dataloader().dataset = None
        with self.assertRaises(AssertionError):
            self.callback._initialize_dataloaders(invalid_datamodule)
        self.callback._initialize_dataloaders(self.trainer.datamodule)
        self.assertIsInstance(self.callback._val_dataloader, torch.utils.data.DataLoader)

    def test_score_model(self) -> None:
        """
        Test scoring of the model with mock features.
        """
        data1 = torch.tensor(1)
        data2 = torch.tensor(2)
        data3 = torch.tensor(3)
        mock_feature = DummyVectorMapFeature(data1=[data1], data2=[data2], data3=[{'test': data3}])
        mock_input = {'mock_feature': mock_feature}
        score, prediction = _score_model(self.pl_module, mock_input, mock_input)
        self.assertEqual(score, mock_feature.data1[0])
        self.assertEqual(prediction, mock_input)

    def test_on_validation_epoch_end(self) -> None:
        """
        Test on validation callback.
        """
        BEST_INDEX = 1
        WORST_INDEX = 0
        self.callback._initialize_dataloaders(self.trainer.datamodule)
        self.callback.on_validation_epoch_end(self.trainer, self.pl_module)
        best_score_path = pathlib.Path(self.output_dir.name + f'/scenes/epoch={self.trainer.current_epoch}' + f'/best/{self.mock_scenarios[BEST_INDEX].token}/{self.scenario_time_stamp.time_us}.json')
        self.assertTrue(best_score_path.exists())
        worst_score_path = pathlib.Path(self.output_dir.name + f'/scenes/epoch={self.trainer.current_epoch}' + f'/worst/{self.mock_scenarios[WORST_INDEX].token}/{self.scenario_time_stamp.time_us}.json')
        self.assertTrue(worst_score_path.exists())
        random_score_dir = pathlib.Path(self.output_dir.name + f'/scenes/epoch={self.trainer.current_epoch}/random/')
        random_score_paths = list(random_score_dir.glob(f'*/{self.scenario_time_stamp.time_us}.json'))
        self.assertEqual(len(random_score_paths), 1)
        with open(str(best_score_path), 'r') as f:
            best_data = json.load(f)
        with open(str(worst_score_path), 'r') as f:
            worst_data = json.load(f)
        self.assertEqual(worst_data['goal']['pose'][0], self.mock_scenarios[WORST_INDEX].get_mission_goal().x)
        self.assertEqual(best_data['goal']['pose'][0], self.mock_scenarios[BEST_INDEX].get_mission_goal().x)

class GenericAgentsFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, agent_features: List[str], trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__()
        self.agent_features = agent_features
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon
        self._agents_states_dim = GenericAgents.agents_states_dim()
        if 'EGO' in self.agent_features:
            raise AssertionError('EGO not valid agents feature type!')
        for feature_name in self.agent_features:
            if feature_name not in TrackedObjectType._member_names_:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'generic_agents'

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return GenericAgents

    @torch.jit.unused
    def get_scriptable_input_from_scenario(self, scenario: AbstractScenario) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        anchor_ego_state = scenario.initial_ego_state
        past_ego_states = scenario.get_ego_past_trajectory(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)
        sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
        time_stamps = list(scenario.get_past_timestamps(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)) + [scenario.start_time]
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [tracked_objects.tracked_objects for tracked_objects in scenario.get_past_tracked_objects(iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses)]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        assert len(sampled_past_ego_states) == len(sampled_past_observations), f'Expected the trajectory length of ego and agent to be equal. Got ego: {len(sampled_past_ego_states)} and agent: {len(sampled_past_observations)}'
        assert len(sampled_past_observations) > 2, f'Trajectory of length of {len(sampled_past_observations)} needs to be at least 3'
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_scriptable_input_from_simulation(self, current_input: PlannerInput) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the simulation input
        :param current_input: planner input from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        history = current_input.history
        assert isinstance(history.observations[0], DetectionsTracks), f'Expected observation of type DetectionTracks, got {type(history.observations[0])}'
        present_ego_state, present_observation = history.current_state
        past_observations = history.observations[:-1]
        past_ego_states = history.ego_states[:-1]
        assert history.sample_interval, 'SimulationHistoryBuffer sample interval is None'
        indices = sample_indices_with_time_horizon(self.num_past_poses, self.past_time_horizon, history.sample_interval)
        try:
            sampled_past_observations = [cast(DetectionsTracks, past_observations[-idx]).tracked_objects for idx in reversed(indices)]
            sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
        except IndexError:
            raise RuntimeError(f'SimulationHistoryBuffer duration: {history.duration} is too short for requested past_time_horizon: {self.past_time_horizon}. Please increase the simulation_buffer_duration in default_simulation.yaml')
        sampled_past_observations = sampled_past_observations + [cast(DetectionsTracks, present_observation).tracked_objects]
        sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]
        time_stamps = [state.time_point for state in sampled_past_ego_states]
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> GenericAgents:
        """Inherited, see superclass."""
        with torch.no_grad():
            tensors, list_tensors, list_list_tensors = self.get_scriptable_input_from_scenario(scenario)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: GenericAgents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> GenericAgents:
        """Inherited, see superclass."""
        with torch.no_grad():
            tensors, list_tensors, list_list_tensors = self.get_scriptable_input_from_simulation(current_input)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: GenericAgents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(self, past_ego_states: List[EgoState], past_time_stamps: List[TimePoint], past_tracked_objects: List[TrackedObjects]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param past_ego_states: The past states of the ego vehicle.
        :param past_time_stamps: The past time stamps of the input data.
        :param past_tracked_objects: The past tracked objects.
        :return: The packed tensors.
        """
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(past_ego_states)
        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)
        for feature_name in self.agent_features:
            past_tracked_objects_tensor_list = sampled_tracked_objects_to_tensor_list(past_tracked_objects, TrackedObjectType[feature_name])
            list_tensor_data[f'past_tracked_objects.{feature_name}'] = past_tracked_objects_tensor_list
        return ({'past_ego_states': past_ego_states_tensor, 'past_time_stamps': past_time_stamps_tensor}, list_tensor_data, {})

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> GenericAgents:
        """
        Unpacks the data returned from the scriptable core into an GenericAgents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed GenericAgents object.
        """
        ego_features = [list_tensor_data['generic_agents.ego'][0].detach().numpy()]
        agent_features = {}
        for key in list_tensor_data:
            if key.startswith('generic_agents.agents.'):
                feature_name = key[len('generic_agents.agents.'):]
                agent_features[feature_name] = [list_tensor_data[key][0].detach().numpy()]
        return GenericAgents(ego=ego_features, agents=agent_features)

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        output_dict: Dict[str, torch.Tensor] = {}
        output_list_dict: Dict[str, List[torch.Tensor]] = {}
        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}
        ego_history: torch.Tensor = tensor_data['past_ego_states']
        time_stamps: torch.Tensor = tensor_data['past_time_stamps']
        anchor_ego_state = ego_history[-1, :].squeeze()
        ego_tensor = build_generic_ego_features_from_tensor(ego_history, reverse=True)
        output_list_dict['generic_agents.ego'] = [ego_tensor]
        for feature_name in self.agent_features:
            if f'past_tracked_objects.{feature_name}' in list_tensor_data:
                agents: List[torch.Tensor] = list_tensor_data[f'past_tracked_objects.{feature_name}']
                agent_history = filter_agents_tensor(agents, reverse=True)
                if agent_history[-1].shape[0] == 0:
                    agents_tensor: torch.Tensor = torch.zeros((len(agent_history), 0, self._agents_states_dim)).float()
                else:
                    padded_agent_states = pad_agent_states(agent_history, reverse=True)
                    local_coords_agent_states = convert_absolute_quantities_to_relative(padded_agent_states, anchor_ego_state)
                    yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
                    agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)
                output_list_dict[f'generic_agents.agents.{feature_name}'] = [agents_tensor]
        return (output_dict, output_list_dict, output_list_list_dict)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {'past_ego_states': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_time_stamps': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_tracked_objects': {'iteration': '0', 'time_horizon': str(self.past_time_horizon), 'num_samples': str(self.num_past_poses), 'agent_features': ','.join(self.agent_features)}}

class TestGenericAgentsFeatureBuilder(unittest.TestCase):
    """Test builder that constructs agent features during training and simulation."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.batch_size = 1
        self.past_time_horizon = 4.0
        self.num_agents = 10
        self.num_past_poses = 4
        self.num_total_past_poses = self.num_past_poses + 1
        self.agent_features = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'TRAFFIC_CONE', 'BARRIER', 'CZONE_SIGN', 'GENERIC_OBJECT']
        self.tracked_object_types: List[TrackedObjectType] = []
        for feature_name in self.agent_features:
            try:
                self.tracked_object_types.append(TrackedObjectType[feature_name])
            except KeyError:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')
        self.feature_builder = GenericAgentsFeatureBuilder(self.agent_features, TrajectorySampling(num_poses=self.num_past_poses, time_horizon=self.past_time_horizon))

    def test_generic_agent_feature_builder(self) -> None:
        """
        Test GenericAgentFeatureBuilder
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents, tracked_object_types=self.tracked_object_types)
        feature = self.feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_no_agents(self) -> None:
        """
        Test when there are no agents
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=0, tracked_object_types=self.tracked_object_types)
        feature = self.feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), 0)
            self.assertEqual(feature.agents[feature_name][0].shape[1], 0)
            self.assertEqual(feature.agents[feature_name][0].shape[2], GenericAgents.agents_states_dim())

    def test_get_feature_from_simulation(self) -> None:
        """
        Test get feature from simulation
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents, tracked_object_types=self.tracked_object_types)
        mock_meta_data = PlannerInitialization(map_api=MockAbstractMap(), route_roadblock_ids=None, mission_goal=StateSE2(0, 0, 0))
        ego_past_states = list(scenario.get_ego_past_trajectory(iteration=0, num_samples=10, time_horizon=5))
        ego_initial_state = scenario.initial_ego_state
        ego_history = ego_past_states + [ego_initial_state]
        past_observations = list(scenario.get_past_tracked_objects(iteration=0, num_samples=10, time_horizon=5))
        initial_observation = scenario.initial_tracked_objects
        observation_history = past_observations + [initial_observation]
        history = SimulationHistoryBuffer.initialize_from_list(len(ego_history), ego_history, observation_history, scenario.database_interval)
        current_input = PlannerInput(iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history, traffic_light_data=scenario.get_traffic_light_status_at_iteration(0))
        feature = self.feature_builder.get_features_from_simulation(current_input=current_input, initialization=mock_meta_data)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_agents_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the Generic Agents Feature Builder scripts properly
        """
        config = self.feature_builder.precomputed_feature_config()
        for expected_key in ['past_ego_states', 'past_time_stamps']:
            self.assertTrue(expected_key in config)
            config_dict = config[expected_key]
            self.assertTrue(len(config_dict) == 3)
            self.assertEqual(0, int(config_dict['iteration']))
            self.assertEqual(self.num_past_poses, int(config_dict['num_samples']))
            self.assertEqual(self.past_time_horizon, int(float(config_dict['time_horizon'])))
        tracked_objects_config_dict = config['past_tracked_objects']
        self.assertTrue(len(tracked_objects_config_dict) == 4)
        self.assertEqual(0, int(tracked_objects_config_dict['iteration']))
        self.assertEqual(self.num_past_poses, int(tracked_objects_config_dict['num_samples']))
        self.assertEqual(self.past_time_horizon, int(float(tracked_objects_config_dict['time_horizon'])))
        self.assertTrue('agent_features' in tracked_objects_config_dict)
        self.assertEqual(','.join(self.agent_features), tracked_objects_config_dict['agent_features'])
        num_frames = 5
        num_agents = 3
        ego_dim = EgoInternalIndex.dim()
        agent_dim = AgentInternalIndex.dim()
        past_ego_states = torch.zeros((num_frames, ego_dim), dtype=torch.float32)
        past_timestamps = torch.tensor([i * 50 for i in range(num_frames)], dtype=torch.int64)
        past_tracked_objects = [torch.ones((num_agents, agent_dim), dtype=torch.float32) for _ in range(num_frames)]
        for i in range(num_frames):
            for j in range(num_agents):
                past_tracked_objects[i][j, :] *= j + 1
        tensor_data = {'past_ego_states': past_ego_states, 'past_time_stamps': past_timestamps}
        list_tensor_data = {f'past_tracked_objects.{feature_name}': past_tracked_objects for feature_name in self.agent_features}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_builder = torch.jit.script(self.feature_builder)
        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        py_tensors, py_list_tensors, py_list_list_tensors = self.feature_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        self.assertEqual(0, len(scripted_tensors))
        self.assertEqual(0, len(py_tensors))
        self.assertEqual(len(scripted_list_tensors), len(py_list_tensors))
        for key in py_list_tensors:
            scripted_list = scripted_list_tensors[key]
            py_list = py_list_tensors[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                scripted = scripted_list[i]
                py = py_list[i]
                torch.testing.assert_allclose(py, scripted, atol=0.05, rtol=0.05)
        self.assertEqual(0, len(scripted_list_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))

def setup_history(scene: Dict[str, Any], scenario: MockAbstractScenario) -> SimulationHistory:
    """
    Mock the history with a mock scenario. The scenario contains the map api, and markers present in the scene are
    used to build a list of ego poses.
    :param scene: The json scene.
    :param scenario: Scenario object.
    :return The mock history.
    """
    if 'expert_ego_states' in scene:
        expert_ego_states = scene['expert_ego_states']
        expert_egos = []
        for expert_ego_state in expert_ego_states:
            ego_state = EgoState.build_from_rear_axle(time_point=TimePoint(expert_ego_state['time_us']), rear_axle_pose=StateSE2(x=expert_ego_state['pose'][0], y=expert_ego_state['pose'][1], heading=expert_ego_state['pose'][2]), rear_axle_velocity_2d=StateVector2D(x=expert_ego_state['velocity'][0], y=expert_ego_state['velocity'][1]), rear_axle_acceleration_2d=StateVector2D(x=expert_ego_state['acceleration'][0], y=expert_ego_state['acceleration'][1]), tire_steering_angle=0, vehicle_parameters=scenario.ego_vehicle_parameters)
            expert_egos.append(ego_state)
        if len(expert_egos):
            scenario.get_expert_ego_trajectory = lambda: expert_egos
            scenario.get_ego_future_trajectory = lambda iteration, time_horizon, num_samples: expert_egos[iteration:iteration + time_horizon + 1:time_horizon // num_samples][1:num_samples + 1]
    map_name = scene['map']['area']
    map_api = get_maps_api(NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, map_name)
    tracked_objects = from_scene_to_tracked_objects(scene['world'])
    for tracked_object in tracked_objects:
        tracked_object._track_token = tracked_object.token
    ego_pose = scene['ego']['pose']
    ego_x = ego_pose[0]
    ego_y = ego_pose[1]
    ego_heading = ego_pose[2]
    ego_states = []
    observations = []
    ego_state = EgoState.build_from_rear_axle(time_point=TimePoint(scene['ego']['time_us']), rear_axle_pose=StateSE2(x=ego_x, y=ego_y, heading=ego_heading), rear_axle_velocity_2d=StateVector2D(x=scene['ego']['velocity'][0], y=scene['ego']['velocity'][1]), rear_axle_acceleration_2d=StateVector2D(x=scene['ego']['acceleration'][0], y=scene['ego']['acceleration'][1]), tire_steering_angle=0, vehicle_parameters=scenario.ego_vehicle_parameters)
    ego_states.append(ego_state)
    observations.append(DetectionsTracks(tracked_objects))
    ego_future_states: List[Dict[str, Any]] = scene['ego_future_states'] if 'ego_future_states' in scene else []
    world_future_states: List[Dict[str, Any]] = scene['world_future_states'] if 'world_future_states' in scene else []
    assert len(ego_future_states) == len(world_future_states), f'Length of world world_future_states: {len(world_future_states)} and length of ego_future_states: {len(ego_future_states)} not same'
    for index, (ego_future_state, future_world_state) in enumerate(zip(ego_future_states, world_future_states)):
        pose = ego_future_state['pose']
        time_us = ego_future_state['time_us']
        ego_state = EgoState.build_from_rear_axle(time_point=TimePoint(time_us), rear_axle_pose=StateSE2(x=pose[0], y=pose[1], heading=pose[2]), rear_axle_velocity_2d=StateVector2D(x=ego_future_state['velocity'][0], y=ego_future_state['velocity'][1]), rear_axle_acceleration_2d=StateVector2D(x=ego_future_state['acceleration'][0], y=ego_future_state['acceleration'][1]), vehicle_parameters=scenario.ego_vehicle_parameters, tire_steering_angle=0)
        future_tracked_objects = from_scene_to_tracked_objects(future_world_state)
        for future_tracked_object in future_tracked_objects:
            future_tracked_object._track_token = future_tracked_object.token
        ego_states.append(ego_state)
        observations.append(DetectionsTracks(future_tracked_objects))
    if ego_states:
        scenario.get_number_of_iterations = lambda: len(ego_states)
    simulation_iterations = []
    trajectories = []
    for index, ego_state in enumerate(ego_states):
        simulation_iterations.append(SimulationIteration(ego_state.time_point, index))
        history_buffer = SimulationHistoryBuffer.initialize_from_list(buffer_size=10, ego_states=[ego_states[index]], observations=[observations[index]], sample_interval=1)
        planner_input = PlannerInput(iteration=SimulationIteration(ego_states[index].time_point, 0), history=history_buffer)
        planner = SimplePlanner(horizon_seconds=10.0, sampling_time=1, acceleration=[0.0, 0.0])
        trajectories.append(planner.compute_planner_trajectory(planner_input))
    history = SimulationHistory(map_api, scenario.get_mission_goal())
    for ego_state, simulation_iteration, trajectory, observation in zip(ego_states, simulation_iterations, trajectories, observations):
        history.add_sample(SimulationHistorySample(iteration=simulation_iteration, ego_state=ego_state, trajectory=trajectory, observation=observation, traffic_light_status=scenario.get_traffic_light_status_at_iteration(simulation_iteration.index)))
    return history

class TestViolationMetricBase(unittest.TestCase):
    """Creates mock violations for testing."""

    def setUp(self) -> None:
        """Set up mock violations."""
        self.violation_metric_base = ViolationMetricBase(name='metric_1', category='Dynamics', max_violation_threshold=1)
        self.mock_abstract_scenario = MockAbstractScenario()
        self.violation_metric_1 = [self._create_mock_violation('metric_1', duration=3, extremum=12.23, mean=8.9), self._create_mock_violation('metric_1', duration=1, extremum=123.23, mean=111.1), self._create_mock_violation('metric_1', duration=10, extremum=12.23, mean=4.92)]
        self.violation_metric_2 = [self._create_mock_violation('metric_2', duration=13, extremum=1.2, mean=0.0)]

    def _create_mock_violation(self, metric_name: str, duration: int, extremum: float, mean: float) -> MetricViolation:
        """Creates a simple violation
        :param metric_name: name of the metric
        :param duration: duration of the violation
        :param extremum: maximally violating value
        :param mean: mean value of violation depth
        :return: a MetricViolation with the given parameters.
        """
        return MetricViolation(metric_computator=self.violation_metric_base.name, name=metric_name, metric_category=self.violation_metric_base.category, unit='unit', start_timestamp=0, duration=duration, extremum=extremum, mean=mean)

    def test_successful_aggregation(self) -> None:
        """Checks that the aggregation of MetricViolations works as intended."""
        aggregated_metrics = self.violation_metric_base.aggregate_metric_violations(metric_violations=self.violation_metric_1, scenario=self.mock_abstract_scenario)[0]
        self.assertEqual(aggregated_metrics.metric_computator, self.violation_metric_base.name)
        self.assertEqual(aggregated_metrics.metric_category, self.violation_metric_base.category)
        statistics = aggregated_metrics.statistics
        self.assertEqual(len(self.violation_metric_1), statistics[0].value)
        self.assertAlmostEqual(statistics[1].value, 123.23, 2)
        self.assertAlmostEqual(statistics[2].value, 12.23, 3)
        self.assertAlmostEqual(statistics[3].value, 13.357, 3)

    def test_failure_on_mixed_metrics(self) -> None:
        """Checks that the aggregation fails when called on MetricViolations from different metrics."""
        with self.assertRaises(AssertionError):
            self.violation_metric_base.aggregate_metric_violations(self.violation_metric_1 + self.violation_metric_2, scenario=self.mock_abstract_scenario)

    def test_empty_statistics_on_empty_violations(self) -> None:
        """Checks that for an empty list of MetricViolations we get a MetricStatistics with zero violations."""
        empty_statistics = self.violation_metric_base.aggregate_metric_violations([], self.mock_abstract_scenario)[0]
        self.assertTrue(empty_statistics.statistics[0].value)

@nuplan_test(path='json/ego_stop_at_stop_line/ego_stop_at_stop_line.json')
def test_stop_polygons_in_lanes(scene: Dict[str, Any]) -> None:
    """
    Check if verification of stop polygons in lanes works as expected
    :param scene: the json scene.
    """
    mock_abstract_scenario = MockAbstractScenario()
    history = setup_history(scene, scenario=mock_abstract_scenario)
    ego_stop_at_stop_line_metric = EgoStopAtStopLineStatistics(name='ego_stop_at_stop_line', category='scenario_dependent', distance_threshold=5.0, velocity_threshold=0.1, max_violation_threshold=1)
    map_api: AbstractMap = history.map_api
    valid_stop_polygons = []
    for data in history.data:
        ego_corners = data.ego_state.car_footprint.oriented_box.geometry.exterior.coords
        ego_pose_front: LineString = LineString([ego_corners[0], ego_corners[3]])
        stop_polygon_info = ego_stop_at_stop_line_metric.get_nearest_stop_line(map_api=map_api, ego_pose_front=ego_pose_front)
        if stop_polygon_info is not None:
            valid_stop_polygons.append(stop_polygon_info)
    assert len(history.data) == 6
    assert len(valid_stop_polygons) == 6

@nuplan_test(path='json/ego_stop_at_stop_line/ego_stop_at_stop_line.json')
def test_check_leading_agent(scene: Dict[str, Any]) -> None:
    """
    Check if check_leading_agent work as expected
    :param scene: the json scene.
    """
    mock_abstract_scenario = MockAbstractScenario()
    history = setup_history(scene, scenario=mock_abstract_scenario)
    ego_stop_at_stop_line_metric = EgoStopAtStopLineStatistics(name='ego_stop_at_stop_line', category='scenario_dependent', distance_threshold=5.0, velocity_threshold=0.1, max_violation_threshold=1)
    map_api: AbstractMap = history.map_api
    remove_agents = [False, False, False, True, True, False]
    expected_results = [True, True, True, False, False, False]
    results = []
    for data, remove_agent in zip(history.data, remove_agents):
        detections = data.observation
        if remove_agent:
            detections.boxes = []
        has_leading_agent = ego_stop_at_stop_line_metric.check_for_leading_agents(detections=detections, ego_state=data.ego_state, map_api=map_api)
        results.append(has_leading_agent)
    assert expected_results == results

@nuplan_test(path='json/ego_stop_at_stop_line/ego_stop_at_stop_line.json')
def test_egos_stop_at_stop_line(scene: Dict[str, Any]) -> None:
    """
    Check if egos stop at stop line as expected
    :param scene: the json scene.
    """
    scene['world']['vehicles'] = []
    mock_abstract_scenario = MockAbstractScenario()
    history = setup_history(scene, scenario=mock_abstract_scenario)
    ego_stop_at_stop_line_metric = EgoStopAtStopLineStatistics(name='ego_stop_at_stop_line', category='scenario_dependent', distance_threshold=5.0, velocity_threshold=0.1, max_violation_threshold=1)
    results = ego_stop_at_stop_line_metric.compute(history=history, scenario=mock_abstract_scenario)
    assert len(results) == 1
    result = results[0]
    metric_statistics = result.statistics
    time_series: Optional[TimeSeries] = result.time_series
    assert metric_statistics[0].value == 1
    assert metric_statistics[1].value == 1
    assert metric_statistics[2].value == 0.06016734670118855
    assert metric_statistics[3].value == 0.05
    expected_velocity = [0.5, 0.05]
    assert time_series.values if time_series is not None else [] == expected_velocity

class MockAbstractScenarioBuilder(AbstractScenarioBuilder):
    """Mock abstract scenario builder class used for testing."""

    def __init__(self, num_scenarios: int=0):
        """
        The init method
        :param num_scenarios: The number of scenarios to return from get_scenarios()
        """
        self.num_scenarios = num_scenarios

    @classmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """Inherited. See superclass."""
        return cast(Type[AbstractScenario], MockAbstractScenario)

    def get_scenarios(self, scenario_filter: ScenarioFilter, worker: WorkerPool) -> List[AbstractScenario]:
        """Implemented. See interface."""
        return [MockAbstractScenario() for _ in range(self.num_scenarios)]

    def get_map_factory(self) -> AbstractMapFactory:
        """Implemented. See interface."""
        return MockMapFactory()

    @property
    def repartition_strategy(self) -> RepartitionStrategy:
        """Implemented. See interface."""
        return RepartitionStrategy.INLINE

def filter_non_stationary_ego(scenario_dict: ScenarioDict, minimum_threshold: float) -> ScenarioDict:
    """
    Filters a ScenarioDict, leaving only scenarios (of any type) in which the ego center travels at least
        minimum_threshold meters cumulatively. These are "non-stationary ego scenarios"
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type. Modified by function
    :param minimum_threshold: minimum distance in meters (inclusive, cumulative) the ego center has to travel in a given
        scenario for the scenario to be called a non-stationary ego scenario
    :return: Filtered scenario dictionary where the cumulative frame-to-frame displacement of the ego center in the
        scenario is >= the minimum threshold
    """
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = list(filter(lambda scenario: _is_non_stationary(scenario, minimum_threshold), scenario_dict[scenario_type]))
    return scenario_dict

def _is_non_stationary(scenario: NuPlanScenario, minimum_threshold: float) -> bool:
    """
    Determines whether the ego cumulatively moves at least minimum_threshold meters over the course of a given scenario
    :param scenario: a NuPlan expert scenario
    :param minimum_threshold: minimum distance in meters (inclusive) the ego center has to travel in the scenario
        for the ego to be determined non-stationary
    :return: True if the cumulative frame-to-frame displacement of the ego center in the scenario
        is >= the minimum threshold
    """
    trajectory = scenario.get_ego_future_trajectory(iteration=0, time_horizon=scenario.duration_s.time_s)
    trajectory_xy_matrix = np.array([[state.center.x, state.center.y] for state in trajectory])
    current_state = trajectory_xy_matrix[:-1]
    next_state = trajectory_xy_matrix[1:]
    total_ego_displacement = np.sum(np.linalg.norm(next_state - current_state, axis=1))
    return bool(total_ego_displacement >= minimum_threshold)

def filter_ego_starts(scenario_dict: ScenarioDict, speed_threshold: float, speed_noise_tolerance: float) -> ScenarioDict:
    """
    Filters a ScenarioDict, leaving only scenarios where the ego has started from a static position at some point

    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type. Modified by function
    :param speed_threshold: exclusive minimum velocity in meters per second that the ego rear axle must reach to be
        considered started
    :return: Filtered scenario dictionary where the ego reaches a speed greater than speed_threshold m/s from below
        at some point in all scenarios
    """
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = list(filter(lambda scenario: _check_for_speed_edge(scenario, speed_threshold, speed_noise_tolerance, EdgeType.RISING), scenario_dict[scenario_type]))
    return scenario_dict

def _check_for_speed_edge(scenario: NuPlanScenario, speed_threshold: float, speed_noise_tolerance: float, edge_type: EdgeType) -> bool:
    """
    For a given scenario, determine whether there is a sub-scenario in which the ego's speed either
        rises above or falls below the speed_threshold.

    :param scenario: a NuPlan scenario
    :speed_threshold: what rear axle speed does the ego have to pass above (exclusive) to have "started moving?"
        likewise, what rear axle speed does the ego have to fall below (inclusive) to have "stopped moving?"
    :param speed_noise_tolerance: a value at or below which a speed change be ignored as noise.
    :param edge_type: are we filtering for speed RISING above the threshold or FALLING below the threshold?
    :return: a boolean, revealing whether a RISING/FALLING ego speed edge is present in the given scenario.
        or equal to the speed threshold and a subsequent frame in which the ego's speed is above the speed threshold.
        The second tells whether the scenario contains one frame in which the ego's speed is above the speed
        threshold and a subsequent frame in which the ego's speed is below or equal to the speed threshold.
    """
    if speed_noise_tolerance is None:
        speed_noise_tolerance = 0.1
    initial_ego_state = scenario.get_ego_state_at_iteration(0)
    current_speed, start_detector, stop_detector = (initial_ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude(),) * 3
    edge_type_presence = [False, False]
    for next_ego_state in scenario.get_ego_future_trajectory(iteration=0, time_horizon=scenario.duration_s.time_s):
        next_speed = next_ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        if next_speed > start_detector:
            start_detector = next_speed
            if start_detector > speed_threshold >= stop_detector and start_detector - stop_detector > speed_noise_tolerance:
                edge_type_presence[EdgeType.RISING] = True
                stop_detector = start_detector
        if next_speed < stop_detector:
            stop_detector = next_speed
            if start_detector > speed_threshold >= stop_detector and start_detector - stop_detector > speed_noise_tolerance:
                edge_type_presence[EdgeType.FALLING] = True
                start_detector = stop_detector
    return edge_type_presence[edge_type]

def filter_ego_stops(scenario_dict: ScenarioDict, speed_threshold: float, speed_noise_tolerance: float) -> ScenarioDict:
    """
    Filters a ScenarioDict, leaving only scenarios where the ego has stopped from a moving position at some point

    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type. Modified by function
    :param speed_threshold: inclusive maximum velocity in meters per second that the ego rear axle must reach to be
        considered stopped
    :return: Filtered scenario dictionary where the ego reaches a speed less than or equal to speed_threshold m/s
        from above at some point in all scenarios
    """
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = list(filter(lambda scenario: _check_for_speed_edge(scenario, speed_threshold, speed_noise_tolerance, EdgeType.FALLING), scenario_dict[scenario_type]))
    return scenario_dict

class TestNuPlanScenarioFilterUtils(unittest.TestCase):
    """
    Tests scenario filter utils for NuPlan
    """

    def _get_mock_scenario_dict(self) -> Dict[str, List[CachedScenario]]:
        """Gets mock scenario dict."""
        return {DEFAULT_SCENARIO_NAME: [CachedScenario(log_name='log/name', token=DEFAULT_SCENARIO_NAME, scenario_type=DEFAULT_SCENARIO_NAME) for i in range(500)], 'lane_following_with_lead': [CachedScenario(log_name='log/name', token='lane_following_with_lead', scenario_type='lane_following_with_lead') for i in range(80)], 'unprotected_left_turn': [CachedScenario(log_name='log/name', token='unprotected_left_turn', scenario_type='unprotected_left_turn') for i in range(120)]}

    def _get_mock_nuplan_scenario_dict_for_timestamp_filtering(self) -> Dict[str, List[CachedScenario]]:
        """Gets mock scenario dict."""
        mock_scenario_dict = {DEFAULT_SCENARIO_NAME: [Mock(NuPlanScenario) for _ in range(0, 100, 3)], 'lane_following_with_lead': [Mock(NuPlanScenario) for _ in range(0, 100, 6)], 'lane_following_without_lead': [Mock(NuPlanScenario) for _ in range(3)]}
        for i in range(0, len(mock_scenario_dict[DEFAULT_SCENARIO_NAME]) * int(1000000.0), int(1000000.0)):
            mock_scenario_dict[DEFAULT_SCENARIO_NAME][int(i / 1000000.0)]._initial_lidar_timestamp = i * 3
        for i in range(0, len(mock_scenario_dict['lane_following_with_lead']) * int(1000000.0), int(1000000.0)):
            mock_scenario_dict['lane_following_with_lead'][int(i / 1000000.0)]._initial_lidar_timestamp = i * 6
        mock_scenario_dict['lane_following_without_lead'][0]._initial_lidar_timestamp = 5.0 * int(1000000.0)
        mock_scenario_dict['lane_following_without_lead'][1]._initial_lidar_timestamp = 100.0 * int(1000000.0)
        mock_scenario_dict['lane_following_without_lead'][2]._initial_lidar_timestamp = 6.0 * int(1000000.0)
        return mock_scenario_dict

    def _get_mock_worker_map(self) -> Callable[..., List[Any]]:
        """
        Gets mock worker_map function.
        """

        def mock_worker_map(worker: WorkerPool, fn: Callable[..., List[Any]], input_objects: List[Any]) -> List[Any]:
            """
            Mock function for worker_map
            :param worker: Worker pool
            :param fn: Callable function
            :param input_objects: List of objects to be used as input
            :return: List of output objects
            """
            return fn(input_objects)
        return mock_worker_map

    def test_filter_total_num_scenarios_int_max_scenarios_requires_removing_known_scenario_types(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 100
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)
        self.assertTrue(len(final_scenario_dict['lane_following_with_lead']) < len(mock_scenario_dict['lane_following_with_lead']))
        self.assertTrue(len(final_scenario_dict['unprotected_left_turn']) < len(mock_scenario_dict['unprotected_left_turn']))
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), limit_total_scenarios)

    def test_filter_total_num_scenarios_int_max_scenarios_less_than_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 300
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertNotEqual(final_scenario_dict[DEFAULT_SCENARIO_NAME], mock_scenario_dict[DEFAULT_SCENARIO_NAME])
        self.assertEqual(final_scenario_dict['lane_following_with_lead'], mock_scenario_dict['lane_following_with_lead'])
        self.assertEqual(final_scenario_dict['unprotected_left_turn'], mock_scenario_dict['unprotected_left_turn'])
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), limit_total_scenarios)

    def test_filter_total_num_scenarios_int_max_scenarios_more_than_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 800
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertDictEqual(final_scenario_dict, mock_scenario_dict)

    def test_filter_total_num_scenarios_float_requires_removing_known_scenario_types(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an float, the actual number of scenarios,
        where the number of scenarios required is requires reomving known scenario types.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.2
        randomize = True
        final_num_of_scenarios = int(limit_total_scenarios * sum((len(scenarios) for scenarios in mock_scenario_dict.values())))
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)
        self.assertTrue(len(final_scenario_dict['lane_following_with_lead']) < len(mock_scenario_dict['lane_following_with_lead']))
        self.assertTrue(len(final_scenario_dict['unprotected_left_turn']) < len(mock_scenario_dict['unprotected_left_turn']))
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), final_num_of_scenarios)

    def test_filter_total_num_scenarios_float_removes_only_default_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an float, the actual number of scenarios,
        where the number of scenarios required is requires reomving known scenario types.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.5
        randomize = True
        final_num_of_scenarios = int(limit_total_scenarios * sum((len(scenarios) for scenarios in mock_scenario_dict.values())))
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertNotEqual(final_scenario_dict[DEFAULT_SCENARIO_NAME], mock_scenario_dict[DEFAULT_SCENARIO_NAME])
        self.assertEqual(final_scenario_dict['lane_following_with_lead'], mock_scenario_dict['lane_following_with_lead'])
        self.assertEqual(final_scenario_dict['unprotected_left_turn'], mock_scenario_dict['unprotected_left_turn'])
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), final_num_of_scenarios)

    def test_remove_all_scenarios_int_limit_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to 0. This should raise an assertion error.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0
        randomize = True
        with self.assertRaises(AssertionError):
            filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)

    def test_remove_all_scenarios_float_limit_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to 0. This should raise an assertion error.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.0
        randomize = True
        with self.assertRaises(AssertionError):
            filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)

    def test_remove_exactly_all_default_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to number of known scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 200
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)
        self.assertEqual(len(final_scenario_dict['lane_following_with_lead']), len(mock_scenario_dict['lane_following_with_lead']))
        self.assertEqual(len(final_scenario_dict['unprotected_left_turn']), len(mock_scenario_dict['unprotected_left_turn']))
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), limit_total_scenarios)

    def test_filter_scenarios_by_timestamp(self) -> None:
        """
        Tests filter_scenarios_by_timestamp with default threshold
        """
        mock_worker_map = self._get_mock_worker_map()
        mock_nuplan_scenario_dict = self._get_mock_nuplan_scenario_dict_for_timestamp_filtering()
        with patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.worker_map', mock_worker_map):
            final_scenario_dict = filter_scenarios_by_timestamp(mock_nuplan_scenario_dict.copy())
            self.assertEqual(len(final_scenario_dict['lane_following_with_lead']), len(mock_nuplan_scenario_dict['lane_following_with_lead']))
            self.assertEqual(len(final_scenario_dict[DEFAULT_SCENARIO_NAME]), len(mock_nuplan_scenario_dict[DEFAULT_SCENARIO_NAME]) * 0.5)
            self.assertEqual(len(final_scenario_dict['lane_following_without_lead']), len(mock_nuplan_scenario_dict['lane_following_without_lead']) - 1)

    def test_filter_fraction_lidarpc_tokens_in_set(self) -> None:
        """
        Test filter_fraction_lidarpc_tokens_in_set with fractional thresholds {0, 0.5, 1}.
        """
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
        mock_nuplan_scenarios = []
        for start_letter in range(4):
            mock_nuplan_scenario = Mock(NuPlanScenario)
            mock_nuplan_scenario.get_scenario_tokens.return_value = set(alphabet[start_letter:start_letter + 3])
            mock_nuplan_scenarios.append(mock_nuplan_scenario)
        full_intersection_scenario, two_intersection_scenario, one_intersection_scenario, no_intersection_scenario = mock_nuplan_scenarios
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json_path = Path(tmp_dir) / 'tmp_token_set.json'
            json.dump(['a', 'b', 'c'], open(tmp_json_path, 'w'))
            scenario_dict = {'on_pickup_dropoff': [no_intersection_scenario, one_intersection_scenario]}
            self.assertEqual(filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 0), {'on_pickup_dropoff': [one_intersection_scenario]})
            scenario_dict['on_pickup_dropoff'] = [one_intersection_scenario, two_intersection_scenario]
            self.assertEqual(filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 0.5), {'on_pickup_dropoff': [two_intersection_scenario]})
            scenario_dict['on_pickup_dropoff'] = [two_intersection_scenario, full_intersection_scenario]
            self.assertEqual(filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 1), {'on_pickup_dropoff': [full_intersection_scenario]})

    def test_filter_non_stationary_ego(self) -> None:
        """Test filter_non_stationary_ego with 0.5m displacement threshold"""
        stationary_ego_pudo_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.01, y=0.0))
        mobile_ego_pudo_scenario = MockAbstractScenario()
        scenario_dict = {'on_pickup_dropoff': [stationary_ego_pudo_scenario, mobile_ego_pudo_scenario]}
        filtered_scenario_dict = filter_non_stationary_ego(scenario_dict, minimum_threshold=0.5)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [mobile_ego_pudo_scenario])

    def test_filter_ego_starts(self) -> None:
        """Test filter_ego_starts with 0.1 m/s speed threshold"""
        slow_acceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=0.01, y=0.0), time_step=1)
        fast_acceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=1, y=0.0), time_step=1)
        scenario_dict = {'on_pickup_dropoff': [slow_acceleration_scenario, fast_acceleration_scenario]}
        filtered_scenario_dict = filter_ego_starts(scenario_dict, speed_threshold=0.1, speed_noise_tolerance=0.1)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [fast_acceleration_scenario])

    def test_filter_ego_stops(self) -> None:
        """Test filter_ego_stops with 0.1 m/s speed threshold"""
        slow_deceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=1.0, y=0.0), fixed_acceleration=StateVector2D(x=-0.01, y=0.0), time_step=1)
        fast_deceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=1.0, y=0.0), fixed_acceleration=StateVector2D(x=-1 / 9, y=0.0), time_step=1)
        scenario_dict = {'on_pickup_dropoff': [slow_deceleration_scenario, fast_deceleration_scenario]}
        filtered_scenario_dict = filter_ego_stops(scenario_dict, speed_threshold=0.1, speed_noise_tolerance=0.1)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [fast_deceleration_scenario])

    def test_ego_startstop_noise_tolerance(self) -> None:
        """Test filter_ego_starts with ego barely crossing speed threshold and noise tolerance higher than threshold"""
        fast_enough_acceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=0.11, y=0.0), time_step=1)
        scenario_dict = {'on_pickup_dropoff': [fast_enough_acceleration_scenario]}
        filtered_scenario_dict = filter_ego_starts(scenario_dict, speed_threshold=1, speed_noise_tolerance=2)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [])

    def test_filter_ego_has_route(self) -> None:
        """
        Test filter_ego_has_route with one route roadblock in the VectorMap (True case),
        and with no route-intersecting roadblocks (False case).
        """
        map_radius = 35
        scenario = MockAbstractScenario()
        scenario_dict = {'on_pickup_dropoff': [scenario]}
        with patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_neighbor_vector_map') as get_neighbor_vector_map:
            get_neighbor_vector_map.return_value = (None, None, None, None, LaneSegmentRoadBlockIDs(['a', 'b', 'c']))
            with patch.object(scenario, 'get_route_roadblock_ids') as get_route_roadblock_ids:
                get_route_roadblock_ids.return_value = ['d', 'e', 'a']
                self.assertEqual(filter_ego_has_route(scenario_dict, map_radius)['on_pickup_dropoff'], [scenario])
                get_route_roadblock_ids.return_value = ['d', 'e', 'f']
                self.assertEqual(filter_ego_has_route(scenario_dict, map_radius)['on_pickup_dropoff'], [])

