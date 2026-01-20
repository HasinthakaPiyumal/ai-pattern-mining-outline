# Cluster 56

class LeaderBoardWriter:
    """Class to write to EvalAI leaderboard."""

    def __init__(self, cfg: DictConfig, submission_path: str) -> None:
        """
        :param cfg: Hydra configuration
        :param submission_path: Path to the directory where the submission files are stored.
        """
        self.contestant_id = cfg.contestant_id
        self.submission_id = cfg.submission_id
        self.output_dir = cfg.output_dir
        self.aggregator_save_path = cfg.aggregator_save_path
        self.challenges = cfg.challenges
        with open(f'{submission_path}/submission_metadata.json', 'r') as file:
            self.submission_metadata = json.load(file)
        try:
            with open(f'{submission_path}/stout.log', 'r') as stdout:
                self.stdout = stdout.read()
        except FileNotFoundError:
            logger.info('No STDOUT log file found')
            self.stdout = ''
        try:
            with open(f'{submission_path}/stderr.log', 'r') as stderr:
                self.stderr = stderr.read()
        except FileNotFoundError:
            logger.info('No STDERR log file found')
            self.stderr = ''
        self.interface = EvalaiInterface()

    def write_to_leaderboard(self, simulation_successful: bool) -> None:
        """
        Writes to the leaderboard
        :param simulation_successful: Whether the simulation was successful or not.
        """
        if simulation_successful:
            logger.info('Writing to leaderboard SUCCESSFUL simulation...')
            data = self._on_successful_submission()
        else:
            logger.info('Writing to leaderboard FAILED simulation...')
            data = self._on_failed_submission()
        self.interface.update_submission_data(data)

    def _on_failed_submission(self) -> Dict[str, str]:
        """
        Builds leaderboard message for failed simulations.
        :return: Message to mark submission as failed
        """
        submission_data = {'challenge_phase': self.submission_metadata.get('challenge_phase'), 'submission': self.submission_metadata.get('submission_id'), 'stdout': self.stdout, 'stderr': self.stderr, 'submission_status': 'FAILED', 'metadata': ''}
        return submission_data

    def _on_successful_submission(self) -> Dict[str, str]:
        """
        Builds leaderboard message for successful simulations.
        :return: Message to mark submission as successful, and to add metric values to leaderboard.
        """
        results: Dict[str, pd.DataFrame] = {}
        for challenge in self.challenges:
            challenge_result_files = Path(self.aggregator_save_path).glob('*.parquet')
            challenge_parquets = [pd.read_parquet(file) for file in challenge_result_files if challenge in str(file)]
            results[challenge] = challenge_parquets[0] if challenge_parquets else []
        result = json.dumps([{'split': 'data_split', 'show_to_participant': True, 'accuracies': read_metrics_from_results(results)}])
        submission_data = {'challenge_phase': self.submission_metadata.get('challenge_phase'), 'submission': self.submission_metadata.get('submission_id'), 'stdout': self.stdout, 'stderr': self.stderr, 'result': result, 'submission_status': 'FINISHED', 'metadata': {'status': 'finished'}}
        return submission_data

def read_metrics_from_results(results: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Transforms a pandas dataframe containing metric results to a string understandable by EvalAI leaderboard.
    :param results: The dataframes of metric results.
    :return: Dict holding the metric names and values.
    """
    ch1_df = results['open_loop_boxes']
    ch2_df = results['closed_loop_nonreactive_agents']
    ch3_df = results['closed_loop_reactive_agents']
    ch1, ch2, ch3 = [df.loc[df['scenario'] == 'final_score'] for df in [ch1_df, ch2_df, ch3_df]]
    metrics = {'ch1_overall_score': ch1['score'].values[0], 'ch1_avg_displacement_error_within_bound': ch1['planner_expert_average_l2_error_within_bound'].values[0], 'ch1_final_displacement_error_within_bound': ch1['planner_expert_final_l2_error_within_bound'].values[0], 'ch1_miss_rate_within_bound': ch1['planner_miss_rate_within_bound'].values[0], 'ch1_avg_heading_error_within_bound': ch1['planner_expert_average_heading_error_within_bound'].values[0], 'ch1_final_heading_error_within_bound': ch1['planner_expert_final_heading_error_within_bound'].values[0], 'ch2_overall_score': ch2['score'].values[0], 'ch2_ego_is_making_progress': ch2['ego_is_making_progress'].values[0], 'ch2_no_ego_at_fault_collisions': ch2['no_ego_at_fault_collisions'].values[0], 'ch2_drivable_area_compliance': ch2['drivable_area_compliance'].values[0], 'ch2_driving_direction_compliance': ch2['driving_direction_compliance'].values[0], 'ch2_ego_is_comfortable': ch2['ego_is_comfortable'].values[0], 'ch2_ego_progress_along_expert_route': ch2['ego_progress_along_expert_route'].values[0], 'ch2_time_to_collision_within_bound': ch2['time_to_collision_within_bound'].values[0], 'ch2_speed_limit_compliance': ch2['speed_limit_compliance'].values[0], 'ch3_overall_score': ch3['score'].values[0], 'ch3_ego_is_making_progress': ch3['ego_is_making_progress'].values[0], 'ch3_no_ego_at_fault_collisions': ch3['no_ego_at_fault_collisions'].values[0], 'ch3_drivable_area_compliance': ch3['drivable_area_compliance'].values[0], 'ch3_driving_direction_compliance': ch3['driving_direction_compliance'].values[0], 'ch3_ego_is_comfortable': ch3['ego_is_comfortable'].values[0], 'ch3_ego_progress_along_expert_route': ch3['ego_progress_along_expert_route'].values[0], 'ch3_time_to_collision_within_bound': ch3['time_to_collision_within_bound'].values[0], 'ch3_speed_limit_compliance': ch3['speed_limit_compliance'].values[0], 'combined_overall_score': np.mean([ch1['score'].values[0], ch2['score'].values[0], ch3['score'].values[0]])}
    return metrics

class TestLeaderboardWriter(unittest.TestCase):
    """Tests for the LeaderboardWriter class."""

    @patch(f'{TEST_FILE}.EvalaiInterface')
    def setUp(self, mock_interface: Mock) -> None:
        """Sets up variables for testing."""
        self.mock_interface = mock_interface
        main_path = os.path.dirname(os.path.realpath(__file__))
        common_dir = 'file://' + os.path.join(main_path, '../../../planning/script/config/common')
        self.search_path = f'hydra.searchpath=[{common_dir}]'
        with initialize_config_dir(config_dir=CONFIG_PATH):
            cfg = compose(config_name=CONFIG_NAME, overrides=[self.search_path, 'contestant_id=contestant', 'submission_id=submission'])
            self.tmpdir = tempfile.TemporaryDirectory()
            self.addCleanup(self.tmpdir.cleanup)
            metadata = {'challenge_phase': 'phase', 'submission_id': 'my_sub'}
            with open(f'{self.tmpdir.name}/submission_metadata.json', 'w') as fp:
                json.dump(metadata, fp)
            self.leaderboard_writer = LeaderBoardWriter(cfg, self.tmpdir.name)

    def test_write_to_leaderboard(self) -> None:
        """Tests that writing to leaderboard calls the correct callbacks an api."""
        with patch.object(self.leaderboard_writer, '_on_successful_submission'):
            self.leaderboard_writer.write_to_leaderboard(simulation_successful=True)
            self.leaderboard_writer._on_successful_submission.assert_called_once()
            self.leaderboard_writer.interface.update_submission_data.assert_called_once_with(self.leaderboard_writer._on_successful_submission.return_value)
        self.mock_interface.reset_mock()
        with patch.object(self.leaderboard_writer, '_on_failed_submission'):
            self.leaderboard_writer.write_to_leaderboard(simulation_successful=False)
            self.leaderboard_writer._on_failed_submission.assert_called_once()
            self.leaderboard_writer.interface.update_submission_data.assert_called_once_with(self.leaderboard_writer._on_failed_submission.return_value)

    def test__on_failed_submission(self) -> None:
        """Tests message creation on failes submission callback."""
        expected_data = {'challenge_phase': 'phase', 'submission': 'my_sub', 'stdout': '', 'stderr': '', 'submission_status': 'FAILED', 'metadata': ''}
        data = self.leaderboard_writer._on_failed_submission()
        self.assertEqual(expected_data, data)

    def test__on_successful_submission(self) -> None:
        """Tests message creation on successful submission callback."""
        expected_data = {'challenge_phase': 'phase', 'submission': 'my_sub', 'stdout': '', 'stderr': '', 'result': '[{"split": "data_split", "show_to_participant": true, "accuracies": "results"}]', 'submission_status': 'FINISHED', 'metadata': {'status': 'finished'}}
        with patch(f'{TEST_FILE}.read_metrics_from_results') as reader:
            reader.return_value = 'results'
            data = self.leaderboard_writer._on_successful_submission()
            self.assertEqual(expected_data, data)

    def test_read_metrics_from_results(self) -> None:
        """Tests parsing of dataframes."""
        dataframes = {'open_loop_boxes': pd.DataFrame.from_dict({'scenario': 'final_score', 'score': [0], 'planner_expert_average_l2_error_within_bound': [1], 'planner_expert_final_l2_error_within_bound': [2], 'planner_miss_rate_within_bound': [3], 'planner_expert_average_heading_error_within_bound': [4], 'planner_expert_final_heading_error_within_bound': [5]}), 'closed_loop_nonreactive_agents': pd.DataFrame.from_dict({'scenario': 'final_score', 'score': [10], 'ego_is_making_progress': [11], 'no_ego_at_fault_collisions': [12], 'drivable_area_compliance': [13], 'driving_direction_compliance': [14], 'ego_is_comfortable': [15], 'ego_progress_along_expert_route': [16], 'time_to_collision_within_bound': [17], 'speed_limit_compliance': [18]}), 'closed_loop_reactive_agents': pd.DataFrame.from_dict({'scenario': 'final_score', 'score': [110], 'ego_is_making_progress': [111], 'no_ego_at_fault_collisions': [112], 'drivable_area_compliance': [113], 'driving_direction_compliance': [114], 'ego_is_comfortable': [115], 'ego_progress_along_expert_route': [116], 'time_to_collision_within_bound': [117], 'speed_limit_compliance': [118]})}
        metrics = read_metrics_from_results(dataframes)
        expected_metrics = {'ch1_overall_score': 0, 'ch1_avg_displacement_error_within_bound': 1, 'ch1_final_displacement_error_within_bound': 2, 'ch1_miss_rate_within_bound': 3, 'ch1_avg_heading_error_within_bound': 4, 'ch1_final_heading_error_within_bound': 5, 'ch2_overall_score': 10, 'ch2_ego_is_making_progress': 11, 'ch2_no_ego_at_fault_collisions': 12, 'ch2_drivable_area_compliance': 13, 'ch2_driving_direction_compliance': 14, 'ch2_ego_is_comfortable': 15, 'ch2_ego_progress_along_expert_route': 16, 'ch2_time_to_collision_within_bound': 17, 'ch2_speed_limit_compliance': 18, 'ch3_overall_score': 110, 'ch3_ego_is_making_progress': 111, 'ch3_no_ego_at_fault_collisions': 112, 'ch3_drivable_area_compliance': 113, 'ch3_driving_direction_compliance': 114, 'ch3_ego_is_comfortable': 115, 'ch3_ego_progress_along_expert_route': 116, 'ch3_time_to_collision_within_bound': 117, 'ch3_speed_limit_compliance': 118, 'combined_overall_score': 40.0}
        self.assertEqual(metrics, expected_metrics)

