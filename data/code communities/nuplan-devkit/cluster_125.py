# Cluster 125

class TestImitationObjective(unittest.TestCase):
    """Test weight decay imitation objective."""

    def setUp(self) -> None:
        """Set up test case."""
        self.target_data: npt.NDArray[np.float32] = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
        self.prediction_data: npt.NDArray[np.float32] = np.array([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])
        self.objective = ImitationObjective(scenario_type_loss_weighting={'unknown': 1.0, 'lane_following_with_lead': 2.0})

    def test_compute_loss(self) -> None:
        """
        Test loss computation
        """
        prediction = Trajectory(data=self.prediction_data)
        target = Trajectory(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='', scenario_type='lane_following_with_lead'), CachedScenario(log_name='', token='', scenario_type='unknown')]
        loss = self.objective.compute({'trajectory': prediction.to_feature_tensor()}, {'trajectory': target.to_feature_tensor()}, scenarios)
        self.assertEqual(loss, torch.tensor(1.5))

    def test_zero_loss(self) -> None:
        """
        Test perfect prediction. The loss should be zero
        """
        target = Trajectory(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='', scenario_type='lane_following_with_lead'), CachedScenario(log_name='', token='', scenario_type='unknown')]
        loss = self.objective.compute({'trajectory': target.to_feature_tensor()}, {'trajectory': target.to_feature_tensor()}, scenarios)
        self.assertEqual(loss, torch.tensor(0.0))

