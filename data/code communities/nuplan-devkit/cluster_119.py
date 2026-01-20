# Cluster 119

class TestTimingCallback(TestCase):
    """
    Tests the simulation TimingCallback.
    """

    def setUp(self) -> None:
        """
        Setup mocks for the tests
        """
        self.writer = Mock(spec=SummaryWriter)
        self.setup = Mock(spec=SimulationSetup)
        self.planner = Mock(spec=AbstractPlanner)
        self.trajectory = Mock(spec=AbstractTrajectory)
        self.history = Mock(spec=SimulationHistory)
        self.history_sample = Mock(spec=SimulationHistorySample)
        self.setup.scenario = Mock()
        self.setup.scenario.token = TOKEN
        self.tc = TimingCallback(self.writer)
        return super().setUp()

    def test_constructor(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        self.assertEqual(self.tc._writer, self.writer)
        self.assertFalse(self.tc._scenarios_captured)
        self.assertIsNone(self.tc._step_start)
        self.assertIsNone(self.tc._simulation_start)
        self.assertIsNone(self.tc._planner_start)
        self.assertFalse(self.tc._step_duration)
        self.assertFalse(self.tc._planner_step_duration)
        self.assertEqual(self.tc._tensorboard_global_step, 0)

    @patch.object(TimingCallback, '_get_time', autospec=True)
    def test_on_planner_start(self, get_time: MagicMock) -> None:
        """
        Tests if the get_time method is called and the start time is set accordingly.
        """
        get_time.return_value = START_TIME
        self.tc.on_planner_start(self.setup, self.planner)
        get_time.assert_called_once()
        self.assertEqual(self.tc._planner_start, START_TIME)

    def test_on_planner_end_throws_if_no_start_time_set(self) -> None:
        """
        Tests if on_planner_end throws an exception if the planner_start time is not set.
        """
        with self.assertRaises(AssertionError):
            self.tc.on_planner_end(self.setup, self.planner, self.trajectory)

    @patch.object(TimingCallback, '_get_time', autospec=True)
    def test_on_planner_end(self, get_time: MagicMock) -> None:
        """
        Tests if the get_time method is called and the duration is set accordingly.
        """
        get_time.return_value = END_TIME
        self.tc._planner_start = START_TIME
        with patch.object(self.tc, '_planner_step_duration') as planner_step_duration:
            self.tc.on_planner_end(self.setup, self.planner, self.trajectory)
            planner_step_duration.append.assert_called_once_with(END_TIME - START_TIME)
            get_time.assert_called_once()

    @patch.object(TimingCallback, '_get_time', autospec=True)
    def test_on_simulation_start(self, get_time: MagicMock) -> None:
        """
        Tests if the captured scenarios for token passed with setup is set to None.
        Tests if the get_time method is called and the simulation_start is set accordingly.
        """
        get_time.return_value = START_TIME
        self.tc.on_simulation_start(self.setup)
        get_time.assert_called_once()
        self.assertEqual(self.tc._scenarios_captured[TOKEN], None)
        self.assertEqual(self.tc._simulation_start, START_TIME)

    def test_on_simulation_end_throws_if_no_start_time_set(self) -> None:
        """
        Tests if on_simulation_end throws an exception if the simulation_start time is not set.
        """
        with self.assertRaises(AssertionError):
            self.tc.on_simulation_end(self.setup, self.planner, self.history)

    @patch.object(TimingCallback, '_get_time', autospec=True)
    def test_on_simulation_end(self, get_time: MagicMock) -> None:
        """
        Tests if the get_time method is called and the elapsed time is set accordingly.
        Tests if the timings are calculated properly and writer is called with the correct values.
        Tests if the timings are stored in the scenarios_captured under the right token.
        Tests if the step_duration and planner_step_duration are cleared.
        """
        get_time.return_value = END_TIME
        self.writer.add_scalar = Mock()
        self.tc._tensorboard_global_step = GLOBAL_STEP
        self.tc._simulation_start = START_TIME
        self.tc._step_duration = [123, 444, 789]
        self.tc._planner_step_duration = [456, 555, 1011]
        self.tc.on_simulation_end(self.setup, self.planner, self.history)
        get_time.assert_called_once()
        self.writer.add_scalar.assert_has_calls([call('simulation_elapsed_time', END_TIME - START_TIME, 7), call('mean_step_time', 452, 7), call('max_step_time', 789, 7), call('max_planner_step_time', 1011, 7), call('mean_planner_step_time', 674, 7)])
        self.assertEqual(self.tc._scenarios_captured[TOKEN], {'simulation_elapsed_time': END_TIME - START_TIME, 'mean_step_time': 452, 'max_step_time': 789, 'max_planner_step_time': 1011, 'mean_planner_step_time': 674})
        self.assertEqual(self.tc._tensorboard_global_step, GLOBAL_STEP + 1)
        self.assertFalse(self.tc._step_duration)
        self.assertFalse(self.tc._planner_step_duration)

    @patch.object(TimingCallback, '_get_time', autospec=True)
    def test_on_step_start(self, get_time: MagicMock) -> None:
        """
        Tests if the get_time method is called and the step_start is set accordingly.
        """
        get_time.return_value = START_TIME
        self.tc.on_step_start(self.setup, self.planner)
        self.assertEqual(self.tc._step_start, START_TIME)

    def test_on_step_end_throws_if_no_start_time_set(self) -> None:
        """
        Tests if on_step_end throws an exception if the step_start time is not set.
        """
        with self.assertRaises(AssertionError):
            self.tc.on_step_end(self.setup, self.planner, self.history_sample)

    @patch.object(TimingCallback, '_step_start', create=True, new_callable=PropertyMock)
    @patch.object(TimingCallback, '_get_time', autospec=True)
    def test_on_step_end(self, get_time: MagicMock, step_start: MagicMock) -> None:
        """
        Tests if the get_time method is called and the duration since start is appended to the step_duration.
        """
        get_time.return_value = END_TIME
        step_start.return_value = START_TIME
        with patch.object(self.tc, '_step_duration') as step_duration:
            self.tc.on_step_end(self.setup, self.planner, self.history_sample)
            step_duration.append.assert_called_once_with(END_TIME - START_TIME)
            get_time.assert_called_once()

    @patch('nuplan.planning.simulation.callback.timing_callback.time.perf_counter')
    def test_get_time(self, perf_counter: MagicMock) -> None:
        """
        Tests if the perf_counter method is called and the result is returned.
        """
        perf_counter.return_value = START_TIME
        result = self.tc._get_time()
        self.assertEqual(result, START_TIME)
        perf_counter.assert_called_once()

