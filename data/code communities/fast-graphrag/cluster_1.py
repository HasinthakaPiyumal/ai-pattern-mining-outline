# Cluster 1

class TestRankingPolicyWithThreshold(unittest.TestCase):

    def test_threshold(self):
        policy = RankingPolicy_WithThreshold(RankingPolicy_WithThreshold.Config(0.1))
        scores = csr_matrix([0.05, 0.2, 0.15, 0.05])
        result = policy(scores)
        expected = csr_matrix([0, 0.2, 0.15, 0])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_all_below_threshold(self):
        policy = RankingPolicy_WithThreshold(RankingPolicy_WithThreshold.Config(0.1))
        scores = csr_matrix([0.05, 0.05, 0.05, 0.05])
        result = policy(scores)
        expected = csr_matrix([], shape=(1, 4))
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_explicit_batch_size_1(self):
        policy = RankingPolicy_WithThreshold(RankingPolicy_WithThreshold.Config(0.1))
        scores = csr_matrix([[0.05, 0.2, 0.15, 0.05]])
        result = policy(scores)
        expected = csr_matrix([[0, 0.2, 0.15, 0]])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_all_above_threshold(self):
        policy = RankingPolicy_WithThreshold(RankingPolicy_WithThreshold.Config(0.1))
        scores = csr_matrix([0.15, 0.2, 0.25, 0.35])
        result = policy(scores)
        expected = csr_matrix([0.15, 0.2, 0.25, 0.35])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

class TestRankingPolicyTopK(unittest.TestCase):

    def test_top_k(self):
        policy = RankingPolicy_TopK(RankingPolicy_TopK.Config(2))
        scores = csr_matrix([0.05, 0.05, 0.2, 0.15, 0.25])
        result = policy(scores)
        expected = csr_matrix([0, 0, 0.2, 0.0, 0.25])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())
        policy = RankingPolicy_TopK(RankingPolicy_TopK.Config(1))
        result = policy(scores)
        expected = csr_matrix([0, 0, 0.0, 0.0, 0.25])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_top_k_less_than_k(self):
        policy = RankingPolicy_TopK(RankingPolicy_TopK.Config(5))
        scores = csr_matrix([0.05, 0.2, 0.0, 0.15])
        result = policy(scores)
        expected = csr_matrix([0.05, 0.2, 0.0, 0.15])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_top_k_is_zero(self):
        policy = RankingPolicy_TopK(RankingPolicy_TopK.Config(0))
        scores = csr_matrix([0.05, 0.2, 0.15, 0.25])
        result = policy(scores)
        expected = csr_matrix([0.05, 0.2, 0.15, 0.25])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_top_k_all_zero(self):
        policy = RankingPolicy_TopK(RankingPolicy_TopK.Config(2))
        scores = csr_matrix([0, 0, 0, 0, 0])
        result = policy(scores)
        expected = csr_matrix([0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

