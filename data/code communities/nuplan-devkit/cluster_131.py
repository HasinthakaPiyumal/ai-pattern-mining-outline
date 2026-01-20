# Cluster 131

class TestGraphAttention(unittest.TestCase):
    """Test graph attention layer."""

    def setUp(self) -> None:
        """Set up test case."""
        self.src_feature_len = 4
        self.dst_feature_len = 4
        self.dist_threshold = 6.0
        self.model = GraphAttention(self.src_feature_len, self.dst_feature_len, self.dist_threshold)

    def test_instantiate(self) -> None:
        """
        Dummy test to check that instantiation works
        """
        self.assertNotEqual(self.model, None)

    def test_forward(self) -> None:
        """Test forward()."""
        num_src_nodes = 2
        num_dst_nodes = 3
        src_node_features = torch.zeros((num_src_nodes, self.src_feature_len))
        src_node_pos = torch.zeros((num_src_nodes, 2))
        dst_node_features = torch.zeros((num_dst_nodes, self.dst_feature_len))
        dst_node_pos = torch.zeros((num_dst_nodes, 2))
        output = self.model.forward(src_node_features, src_node_pos, dst_node_features, dst_node_pos)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (num_dst_nodes, self.dst_feature_len))

