# Cluster 129

class TestLocalSubGraphLayer(unittest.TestCase):
    """Test LocalSubGraphLayer layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.dim_in = 256
        self.dim_out = 256
        self.model = LocalSubGraphLayer(self.dim_in, self.dim_out)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works.
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_elements = 10
        num_points = 20
        inputs = torch.zeros((num_elements, num_points, self.dim_in))
        invalid_mask = torch.zeros((num_elements, num_points), dtype=torch.bool)
        output = self.model.forward(inputs, invalid_mask)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_elements, num_points, self.dim_out))

