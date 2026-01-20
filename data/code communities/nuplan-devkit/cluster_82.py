# Cluster 82

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

class TestEvalaiInterface(unittest.TestCase):
    """Tests interface class to EvalAI api."""

    @patch.dict(os.environ, {'EVALAI_CHALLENGE_PK': '1234', 'EVALAI_PERSONAL_AUTH_TOKEN': 'authorization_token'})
    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.evalai = EvalaiInterface('bounce_server')

    def test_initialization(self) -> None:
        """Checks that initialization works and fails as expected."""
        self.assertEqual(self.evalai.EVALAI_AUTH_TOKEN, 'authorization_token')
        self.assertEqual(self.evalai.CHALLENGE_PK, '1234')
        self.assertEqual(self.evalai.EVALAI_API_SERVER, 'bounce_server')
        with patch.dict(os.environ, {'EVALAI_CHALLENGE_PK': ''}):
            with self.assertRaises(AssertionError):
                _ = EvalaiInterface('server')
        with patch.dict(os.environ, {'EVALAI_PERSONAL_AUTH_TOKEN': ''}):
            with self.assertRaises(AssertionError):
                _ = EvalaiInterface('server')

    @patch('requests.request', side_effect=mocked_put_request)
    def test_update_submission_data(self, mock_put: Mock) -> None:
        """Tests update submission with mock server."""
        test_payload = {'test': 'payload'}
        response = self.evalai.update_submission_data(test_payload)
        self.assertEqual(response, test_payload)
        expected_call = call(method='PUT', url='bounce_server/api/jobs/challenge/1234/update_submission/', headers={'Authorization': 'Bearer authorization_token'}, data=test_payload)
        self.assertEqual(response, test_payload)
        self.assertIn(expected_call, mock_put.call_args_list)
        self.assertEqual(len(mock_put.call_args_list), 1)

    def test_fail_on_missing_api(self) -> None:
        """Test failure of url generation on missing api."""
        with self.assertRaises(AssertionError):
            _ = self.evalai._format_url('missing_api')

