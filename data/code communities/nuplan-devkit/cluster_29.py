# Cluster 29

class TestScenarioTag(unittest.TestCase):
    """Tests class ScenarioTag"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.scenario_tag = ScenarioTag()

    @patch('nuplan.database.nuplan_db_orm.scenario_tag.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the session property"""
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock
        result = self.scenario_tag._session
        inspect_mock.assert_called_once_with(self.scenario_tag)
        self.assertEqual(result, session_mock)

    @patch('nuplan.database.nuplan_db_orm.scenario_tag.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the __repr__ method"""
        result = self.scenario_tag.__repr__()
        self.assertEqual(result, simple_repr_mock.return_value)

