# Cluster 128

class TestMLP(unittest.TestCase):
    """Test MLP layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.input_dim = 256
        self.hidden_dim = 256 * 4
        self.output_dim = 12 * 3
        self.num_layers = 3
        self.model = MLP(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_inputs = 10
        inputs = torch.zeros((num_inputs, self.input_dim))
        output = self.model.forward(inputs)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_inputs, self.output_dim))

