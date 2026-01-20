# Cluster 120

class TestSimulationLogCallback(unittest.TestCase):
    """Tests simulation_log_callback."""

    def setUp(self) -> None:
        """Setup Mocked classes."""
        self.output_folder = tempfile.TemporaryDirectory()
        self.callback = SimulationLogCallback(output_directory=self.output_folder.name, simulation_log_dir='simulation_log', serialization_type='msgpack')
        self.sim_manager = Mock(spec=AbstractSimulationTimeController)
        self.observation = Mock(spec=AbstractObservation)
        self.controller = Mock(spec=AbstractEgoController)

    def tearDown(self) -> None:
        """Clean up folder."""
        self.output_folder.cleanup()

    def test_callback(self) -> None:
        """
        Tests whether a scene can be dumped into a simulation log, checks that the keys are correct,
        and checks that the log contains the expected data after being re-loaded from disk.
        """
        scenario = MockAbstractScenario()
        self.setup = SimulationSetup(observations=self.observation, scenario=scenario, time_controller=self.sim_manager, ego_controller=self.controller)
        planner = SimplePlanner(2, 0.5, [0, 0])
        directory = self.callback._get_scenario_folder(planner.name(), scenario)
        self.assertEqual(str(directory), self.output_folder.name + '/simulation_log/SimplePlanner/mock_scenario_type/mock_log_name/mock_scenario_name')
        self.callback.on_initialization_start(self.setup, planner)
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
        state_0 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(0))
        state_1 = EgoState.build_from_rear_axle(StateSE2(0, 0, 0), vehicle_parameters=scenario.ego_vehicle_parameters, rear_axle_velocity_2d=StateVector2D(x=0, y=0), rear_axle_acceleration_2d=StateVector2D(x=0, y=0), tire_steering_angle=0, time_point=TimePoint(1000))
        history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_0, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0))))
        history.add_sample(SimulationHistorySample(iteration=SimulationIteration(time_point=TimePoint(0), index=0), ego_state=state_1, trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]), observation=DetectionsTracks(TrackedObjects()), traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0))))
        for data in history.data:
            self.callback.on_step_end(self.setup, planner, data)
        self.callback.on_simulation_end(self.setup, planner, history)
        path = pathlib.Path(self.output_folder.name + '/simulation_log/SimplePlanner/mock_scenario_type/mock_log_name/mock_scenario_name/mock_scenario_name.msgpack.xz')
        self.assertTrue(path.exists())
        simulation_log = SimulationLog.load_data(file_path=path)
        self.assertEqual(simulation_log.file_path, path)
        self.assertTrue(objects_are_equal(simulation_log.simulation_history, history))

