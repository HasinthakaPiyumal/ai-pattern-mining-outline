# Cluster 127

class TestLocalMLP(unittest.TestCase):
    """Test LocalMLP layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.dim_in = 256
        self.model = LocalMLP(self.dim_in)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_inputs = 10
        inputs = torch.zeros((num_inputs, self.dim_in))
        output = self.model.forward(inputs)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_inputs, self.dim_in))

