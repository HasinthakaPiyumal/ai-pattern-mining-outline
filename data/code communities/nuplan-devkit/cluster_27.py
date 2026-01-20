# Cluster 27

class TestTrafficLightStatus(unittest.TestCase):
    """Tests the TrafficLightStatus class"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.traffic_light_status = TrafficLightStatus()

    @patch('nuplan.database.nuplan_db_orm.traffic_light_status.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the repr method"""
        result = self.traffic_light_status.__repr__()
        simple_repr_mock.assert_called_once_with(self.traffic_light_status)
        self.assertEqual(result, simple_repr_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.traffic_light_status.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the session property"""
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock
        result = self.traffic_light_status._session()
        inspect_mock.assert_called_once_with(self.traffic_light_status)
        self.assertEqual(result, session_mock.return_value)

