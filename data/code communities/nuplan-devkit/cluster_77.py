# Cluster 77

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

class InterpolatedTrajectory(AbstractTrajectory):
    """Class representing a trajectory that can be interpolated from a list of points."""

    def __init__(self, trajectory: List[InterpolatableState]):
        """
        :param trajectory: List of states creating a trajectory.
            The trajectory has to have at least 2 elements, otherwise it is considered invalid and the class will raise.
        """
        assert trajectory, "Trajectory can't be empty!"
        assert isinstance(trajectory[0], InterpolatableState)
        self._trajectory_class = trajectory[0].__class__
        assert all((isinstance(point, self._trajectory_class) for point in trajectory))
        if len(trajectory) <= 1:
            raise ValueError(f'There is not enough states in trajectory: {len(trajectory)}!')
        self._trajectory = trajectory
        time_series = [point.time_us for point in trajectory]
        linear_states = []
        angular_states = []
        for point in trajectory:
            split_state = point.to_split_state()
            linear_states.append(split_state.linear_states)
            angular_states.append(split_state.angular_states)
        self._fixed_state = trajectory[0].to_split_state().fixed_states
        linear_states = np.array(linear_states, dtype='float64')
        angular_states = np.array(angular_states, dtype='float64')
        self._function_interp_linear = sp_interp.interp1d(time_series, linear_states, axis=0)
        self._angular_interpolator = AngularInterpolator(time_series, angular_states)

    def __reduce__(self) -> Tuple[Type[InterpolatedTrajectory], Tuple[Any, ...]]:
        """
        Helper for pickling.
        """
        return (self.__class__, (self._trajectory,))

    @property
    def start_time(self) -> TimePoint:
        """Inherited, see superclass."""
        return self._trajectory[0].time_point

    @property
    def end_time(self) -> TimePoint:
        """Inherited, see superclass."""
        return self._trajectory[-1].time_point

    def get_state_at_time(self, time_point: TimePoint) -> InterpolatableState:
        """Inherited, see superclass."""
        start_time = self.start_time
        end_time = self.end_time
        assert start_time <= time_point <= end_time, f'Interpolation time time_point={time_point!r} not in trajectory time window! \nstart_time.time_us={start_time.time_us!r} <= time_point.time_us={time_point.time_us!r} <= end_time.time_us={end_time.time_us!r}'
        linear_states = list(self._function_interp_linear(time_point.time_us))
        angular_states = list(self._angular_interpolator.interpolate(time_point.time_us))
        return self._trajectory_class.from_split_state(SplitState(linear_states, angular_states, self._fixed_state))

    def get_state_at_times(self, time_points: List[TimePoint]) -> List[InterpolatableState]:
        """Inherited, see superclass."""
        start_time = self.start_time
        end_time = self.end_time
        assert start_time <= min(time_points), f'Interpolation time not in trajectory time window! The following is not satisfied:Trajectory start time: ({start_time.time_s}) <= Earliest interpolation time ({min(time_points).time_s}) {max(time_points).time_s} <= {end_time.time_s} '
        assert max(time_points) <= end_time, f'Interpolation time not in trajectory time window! The following is not satisfied:Trajectory end time: ({end_time.time_s}) >= Latest interpolation time ({max(time_points).time_s}) '
        interpolation_times = [t.time_us for t in time_points]
        linear_states = list(self._function_interp_linear(interpolation_times))
        angular_states = list(self._angular_interpolator.interpolate(interpolation_times))
        return [self._trajectory_class.from_split_state(SplitState(lin_state, ang_state, self._fixed_state)) for lin_state, ang_state in zip(linear_states, angular_states)]

    def get_sampled_trajectory(self) -> List[InterpolatableState]:
        """Inherited, see superclass."""
        return self._trajectory

class InterpolatedPath(AbstractPath):
    """A path that is interpolated from a list of points."""

    def __init__(self, path: List[ProgressStateSE2]):
        """
        Constructor of InterpolatedPath.

        :param path: List of states creating a path.
            The path has to have at least 2 elements, otherwise it is considered invalid and the class will raise.
        """
        assert len(path) > 1, 'Path has to has more than 1 element!'
        self._path = path
        progress = [point.progress for point in self._path]
        linear_states = []
        angular_states = []
        for point in path:
            linear_states.append([point.progress, point.x, point.y])
            angular_states.append([point.heading])
        linear_states = np.array(linear_states, dtype='float64')
        angular_states = np.array(angular_states, dtype='float64')
        self._function_interp_linear = sp_interp.interp1d(progress, linear_states, axis=0)
        self._angular_interpolator = AngularInterpolator(progress, angular_states)

    def get_start_progress(self) -> float:
        """Inherited, see superclass."""
        return self._path[0].progress

    def get_end_progress(self) -> float:
        """Inherited, see superclass."""
        return self._path[-1].progress

    def get_state_at_progress(self, progress: float) -> ProgressStateSE2:
        """Inherited, see superclass."""
        self._assert_progress(progress)
        linear_states = list(self._function_interp_linear(progress))
        angular_states = list(self._angular_interpolator.interpolate(progress))
        return ProgressStateSE2.deserialize(linear_states + angular_states)

    def get_state_at_progresses(self, progresses: List[float]) -> List[ProgressStateSE2]:
        """Inherited, see superclass."""
        self._assert_progress(min(progresses))
        self._assert_progress(max(progresses))
        linear_states_batch = self._function_interp_linear(progresses)
        angular_states_batch = self._angular_interpolator.interpolate(progresses)
        return [ProgressStateSE2.deserialize(list(linear_states) + list(angular_states)) for linear_states, angular_states in zip(linear_states_batch, angular_states_batch)]

    def get_sampled_path(self) -> List[ProgressStateSE2]:
        """Inherited, see superclass."""
        return self._path

    def _assert_progress(self, progress: float) -> None:
        """Check if queried progress is within bounds"""
        start_progress = self.get_start_progress()
        end_progress = self.get_end_progress()
        assert start_progress <= progress <= end_progress, f'Progress exceeds path! {start_progress} <= {progress} <= {end_progress}'

