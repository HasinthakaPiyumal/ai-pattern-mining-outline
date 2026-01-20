# Cluster 65

class ColumnTypeTests(unittest.TestCase):

    def test_ndarray_type_to_numpy_type(self):
        expected_type = [np.int8, np.uint8, np.int16, np.int32, np.int64, np.unicode_, np.bool_, np.float32, np.float64, Decimal, np.str_, np.datetime64]
        for ndarray_type, np_type in zip(NdArrayType, expected_type):
            self.assertEqual(NdArrayType.to_numpy_type(ndarray_type), np_type)

    def test_raise_exception_uknown_ndarray_type(self):
        self.assertRaises(ValueError, NdArrayType.to_numpy_type, ColumnType.TEXT)

