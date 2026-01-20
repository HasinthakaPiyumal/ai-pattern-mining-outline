# Cluster 23

class TestRankingPolicyElbow(unittest.TestCase):

    def test_elbow(self):
        policy = RankingPolicy_Elbow(config=None)
        scores = csr_matrix([0.05, 0.2, 0.1, 0.25, 0.1])
        result = policy(scores)
        expected = csr_matrix([0, 0.2, 0.0, 0.25, 0])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_elbow_all_zero(self):
        policy = RankingPolicy_Elbow(config=None)
        scores = csr_matrix([0, 0, 0, 0, 0])
        result = policy(scores)
        expected = csr_matrix([0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def test_elbow_all_same(self):
        policy = RankingPolicy_Elbow(config=None)
        scores = csr_matrix([0.05, 0.05, 0.05, 0.05, 0.05])
        result = policy(scores)
        expected = csr_matrix([0, 0.05, 0.05, 0.05, 0.05])
        np.testing.assert_array_equal(result.toarray(), expected.toarray())

