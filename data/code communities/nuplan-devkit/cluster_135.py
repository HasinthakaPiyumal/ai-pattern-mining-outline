# Cluster 135

class TestKinematicUnicycleLayersUtils(unittest.TestCase):
    """
    Test Kinematic Unicycle Layers utils.
    """

    def setUp(self) -> None:
        """Sets variables for testing"""
        pass

    def test_enums_equal(self) -> None:
        """
        Ensure our internal indexing matches the
        one from Ego's trajectory
        """
        self.assertEqual(StateIndex.X_POS, GenericEgoFeatureIndex.x())
        self.assertEqual(StateIndex.Y_POS, GenericEgoFeatureIndex.y())
        self.assertEqual(StateIndex.YAW, GenericEgoFeatureIndex.heading())
        self.assertEqual(StateIndex.X_VELOCITY, GenericEgoFeatureIndex.vx())
        self.assertEqual(StateIndex.Y_VELOCITY, GenericEgoFeatureIndex.vy())
        self.assertEqual(StateIndex.X_ACCEL, GenericEgoFeatureIndex.ax())
        self.assertEqual(StateIndex.Y_ACCEL, GenericEgoFeatureIndex.ay())

