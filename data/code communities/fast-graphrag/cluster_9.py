# Cluster 9

def extract_sorted_scores(row_vector: csr_matrix) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Take a sparse row vector and return a list of non-zero (index, score) pairs sorted by score."""
    assert row_vector.shape[0] <= 1, 'The input matrix must be a row vector.'
    if row_vector.shape[0] == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    non_zero_indices = row_vector.nonzero()[1]
    probabilities = row_vector.data
    indices_array = np.array(non_zero_indices)
    probabilities_array = np.array(probabilities)
    sorted_indices = np.argsort(probabilities_array)[::-1]
    sorted_indices_array = indices_array[sorted_indices]
    sorted_probabilities_array = probabilities_array[sorted_indices]
    return (sorted_indices_array, sorted_probabilities_array)

class TestExtractSortedScores(unittest.TestCase):

    def test_non_zero_elements(self):
        row_vector = csr_matrix([[0, 0.1, 0, 0.7, 0.5, 0]])
        indices, scores = extract_sorted_scores(row_vector)
        np.testing.assert_array_equal(indices, np.array([3, 4, 1]))
        np.testing.assert_array_equal(scores, np.array([0.7, 0.5, 0.1]))

    def test_empty(self):
        row_vector = csr_matrix((0, 0))
        indices, scores = extract_sorted_scores(row_vector)
        np.testing.assert_array_equal(indices, np.array([], dtype=np.int64))
        np.testing.assert_array_equal(scores, np.array([], dtype=np.float32))

    def test_empty_row_vector(self):
        row_vector = csr_matrix([[]])
        indices, scores = extract_sorted_scores(row_vector)
        np.testing.assert_array_equal(indices, np.array([], dtype=np.int64))
        np.testing.assert_array_equal(scores, np.array([], dtype=np.float32))

    def test_single_element(self):
        row_vector = csr_matrix([[0.5]])
        indices, scores = extract_sorted_scores(row_vector)
        np.testing.assert_array_equal(indices, np.array([0]))
        np.testing.assert_array_equal(scores, np.array([0.5]))

    def test_all_zero_elements(self):
        row_vector = csr_matrix([[0, 0, 0, 0, 0]])
        indices, scores = extract_sorted_scores(row_vector)
        np.testing.assert_array_equal(indices, np.array([], dtype=np.int64))
        np.testing.assert_array_equal(scores, np.array([], dtype=np.float32))

    def test_duplicate_elements(self):
        row_vector = csr_matrix([[0, 0.1, 0, 0.7, 0.5, 0.7]])
        indices, scores = extract_sorted_scores(row_vector)
        expected_indices_1 = np.array([5, 3, 4, 1])
        expected_indices_2 = np.array([3, 5, 4, 1])
        self.assertTrue(np.array_equal(indices, expected_indices_1) or np.array_equal(indices, expected_indices_2), f'indices {indices} do not match either {expected_indices_1} or {expected_indices_2}')
        np.testing.assert_array_equal(scores, np.array([0.7, 0.7, 0.5, 0.1]))

