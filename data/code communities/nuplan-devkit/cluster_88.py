# Cluster 88

class TestAbstractPredictor(unittest.TestCase):
    """Test the AbstractPredictor interface"""

    def setUp(self) -> None:
        """Inherited, see superclass"""
        self.predictor = MockAbstractPredictor()

    def test_initialize(self) -> None:
        """Test initialization"""
        mock_initialization = get_mock_predictor_initialization()
        self.predictor.initialize(mock_initialization)
        self.assertEqual(self.predictor._map_api, mock_initialization.map_api)

    def test_name(self) -> None:
        """Test name"""
        self.assertEqual(self.predictor.name(), 'MockAbstractPredictor')

    def test_observation_type(self) -> None:
        """Test observation_type"""
        self.assertEqual(self.predictor.observation_type(), DetectionsTracks)

    def test_compute_predictions(self) -> None:
        """Test compute_predictions"""
        predictor_input = get_mock_predictor_input()
        start_time = time.perf_counter()
        detections = self.predictor.compute_predictions(predictor_input)
        compute_predictions_time = time.perf_counter() - start_time
        self.assertEqual(type(detections), DetectionsTracks)
        predictor_report = self.predictor.generate_predictor_report()
        self.assertEqual(len(predictor_report.compute_predictions_runtimes), 1)
        self.assertNotIsInstance(predictor_report, MLPredictorReport)
        self.assertAlmostEqual(predictor_report.compute_predictions_runtimes[0], compute_predictions_time, delta=0.1)

