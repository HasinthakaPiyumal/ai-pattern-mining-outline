# Cluster 80

def validate_submission(image: str, validator: BaseSubmissionValidator) -> tuple[bool, Optional[Type[AbstractSubmissionValidator]]]:
    """
    Calls the chain of validators on one image.
    :param image: The query docker image
    :param validator: The chain of validators
    :return: A tuple with two possible values:
        (True, None) If the image is valid
        (False, Failing validator type) if image is deemed invalid by a validator on the chain
    """
    image_is_valid = validator.validate(image)
    return (bool(image_is_valid), validator.failing_validator)

class TestImageIsRunnableValidator(unittest.TestCase):
    """Tests for the ImageIsRunnableValidator class"""

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.validator = ImageIsRunnableValidator()

    def test_construction(self) -> None:
        """Tests that the variables are initialized correctly."""
        self.assertTrue(isinstance(self.validator, BaseSubmissionValidator))

    @patch('nuplan.submission.validators.image_is_runnable_validator.SubmissionContainer')
    def test_validate_runnable(self, mock_submission_container: Mock) -> None:
        """Tests that validator calls the next validator when the image is runnable."""
        submission = 'foo'
        with patch.object(BaseSubmissionValidator, 'validate') as mock_validate:
            self.validator.validate(submission)
            mock_submission_container.return_value.start.assert_called_once()
            mock_validate.assert_called_with(submission)

    @patch('nuplan.submission.validators.image_is_runnable_validator.SubmissionContainer')
    def test_validate_not_runnable(self, mock_submission_container: Mock) -> None:
        """Tests that validator returns False when image is not runnable."""
        mock_submission_container.return_value.wait_until_running.side_effect = TimeoutError
        result = self.validator.validate('foo')
        self.assertFalse(result)

class TestSubmissionValidator(unittest.TestCase):
    """Tests for the BaseSubmissionValidator class"""

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.validator = BaseSubmissionValidator()

    def test_construction(self) -> None:
        """Tests that the variables are initialized correctly."""
        self.assertEqual(None, self.validator._next_validator)
        self.assertEqual(None, self.validator.failing_validator)

    def test_set_next(self) -> None:
        """Tests that assigning the next validator works."""
        next_validator = Mock()
        self.validator.set_next(next_validator)
        self.assertEqual(next_validator, self.validator._next_validator)

    def test_validate(self) -> None:
        """Tests that base validator works, and that validators are called in chain."""
        self.assertTrue(self.validator.validate(Mock()))
        validate = Mock(return_value=False)
        next_validator = Mock(validate=validate)
        self.validator._next_validator = next_validator
        result = self.validator.validate(Mock())
        validate.assert_called_once()
        self.assertEqual(validate.return_value, result)

class TestImageExistsValidator(unittest.TestCase):
    """Tests for the ImageExistsValidator"""

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.validator = ImageExistsValidator()

    def test_construction(self) -> None:
        """Tests that the variables are initialized correctly."""
        self.assertTrue(isinstance(self.validator, BaseSubmissionValidator))

    @patch('docker.from_env')
    def test_validate(self, mock_env: Mock) -> None:
        """Tests that the validator behaves as intended"""
        missing_submission = 'foo'
        present_submission = 'bar'
        mock_env.return_value.images.list.return_value = ['bar', 'b']
        self.assertEqual(False, self.validator.validate(missing_submission))
        with patch.object(BaseSubmissionValidator, 'validate') as mock_validate:
            self.validator.validate(present_submission)
            mock_validate.assert_called_with(present_submission)

class TestSubmissionComputesTrajectoryValidator(unittest.TestCase):
    """Tests for SubmissionComputesTrajectoryValidator class"""

    def setUp(self) -> None:
        """Sets variables for testing"""
        self.submission_computes_trajectory_validator = SubmissionComputesTrajectoryValidator()

    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.get_test_nuplan_scenario', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SimulationIteration', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SimulationHistoryBuffer', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.PlannerInput', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SubmissionContainerFactory', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.RemotePlanner')
    def test_report_invalid_case(self, mock_remote_planner: Mock) -> None:
        """Tests that if a planner can't provide a trajectory, the validator fails"""
        submission = 'foo'
        mock_remote_planner().compute_trajectory.return_value = []
        valid = self.submission_computes_trajectory_validator.validate(submission)
        mock_remote_planner().compute_trajectory.assert_called()
        self.assertFalse(valid)
        self.assertEqual(self.submission_computes_trajectory_validator.failing_validator, SubmissionComputesTrajectoryValidator)

    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.get_test_nuplan_scenario', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SimulationIteration', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SimulationHistoryBuffer', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.PlannerInput', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.SubmissionContainerFactory', Mock())
    @patch('nuplan.submission.validators.submission_computes_trajectory_validator.RemotePlanner')
    def test_report_valid_case(self, mock_remote_planner: Mock) -> None:
        """Checks that the validator succeeds when the planner computes a trajectory"""
        submission = 'foo'
        mock_remote_planner().compute_trajectory.return_value = ['my', 'wonderful', 'trajectory']
        valid = self.submission_computes_trajectory_validator.validate(submission)
        mock_remote_planner().compute_trajectory.assert_called()
        self.assertTrue(valid)
        self.assertEqual(self.submission_computes_trajectory_validator.failing_validator, None)

