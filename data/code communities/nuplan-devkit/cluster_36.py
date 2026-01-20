# Cluster 36

class TestScene(unittest.TestCase):
    """Test class Scene"""

    def setUp(self) -> None:
        """Sets up for the test cases"""
        self.scene = Scene()

    @patch('nuplan.database.nuplan_db_orm.scene.simple_repr', autospec=True)
    def test_repr(self, simple_repr_mock: Mock) -> None:
        """Tests the repr method"""
        result = self.scene.__repr__()
        simple_repr_mock.assert_called_once_with(self.scene)
        self.assertEqual(result, simple_repr_mock.return_value)

    @patch('nuplan.database.nuplan_db_orm.scene.inspect', autospec=True)
    def test_session(self, inspect_mock: Mock) -> None:
        """Tests the session property"""
        session_mock = PropertyMock()
        inspect_mock.return_value = Mock()
        inspect_mock.return_value.session = session_mock
        result = self.scene._session()
        inspect_mock.assert_called_once_with(self.scene)
        self.assertEqual(result, session_mock.return_value)

