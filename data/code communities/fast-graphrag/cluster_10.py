# Cluster 10

def csr_from_indices_list(data: List[List[Union[int, TIndex]]], shape: Tuple[int, int]) -> csr_matrix:
    """Create a CSR matrix from a list of lists."""
    num_rows = len(data)
    row_indices = np.repeat(np.arange(num_rows), [len(row) for row in data])
    col_indices = np.concatenate(data) if num_rows > 0 else np.array([], dtype=np.int64)
    values = np.broadcast_to(1, len(row_indices))
    return csr_matrix((values, (row_indices, col_indices)), shape=shape)

class TestCsrFromListOfLists(unittest.TestCase):

    def test_repeated_elements(self):
        data: List[List[int]] = [[0, 0], [], []]
        expected_matrix = csr_matrix(([1, 1, 0], ([0, 0, 0], [0, 0, 0])), shape=(3, 3))
        result_matrix = csr_from_indices_list(data, shape=(3, 3))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())

    def test_non_zero_elements(self):
        data = [[0, 1, 2], [2, 3], [0, 3]]
        expected_matrix = csr_matrix([[1, 1, 1, 0, 0], [0, 0, 1, 1, 0], [1, 0, 0, 1, 0]], shape=(3, 5))
        result_matrix = csr_from_indices_list(data, shape=(3, 5))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())

    def test_empty_list_of_lists(self):
        data: List[List[int]] = []
        expected_matrix = csr_matrix((0, 0))
        result_matrix = csr_from_indices_list(data, shape=(0, 0))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())

    def test_empty_list_of_lists_with_unempty_shape(self):
        data: List[List[int]] = []
        expected_matrix = csr_matrix((1, 1))
        result_matrix = csr_from_indices_list(data, shape=(1, 1))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())

    def test_list_with_empty_sublists(self):
        data: List[List[int]] = [[], [], []]
        expected_matrix = csr_matrix((3, 0))
        result_matrix = csr_from_indices_list(data, shape=(3, 0))
        np.testing.assert_array_equal(result_matrix.toarray(), expected_matrix.toarray())

