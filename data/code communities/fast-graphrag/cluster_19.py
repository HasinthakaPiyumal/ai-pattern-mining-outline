# Cluster 19

def dump_to_reference_list(data: Iterable[object], separator: str='\n=====\n\n'):
    return [f'[{i + 1}]  {d}{separator}' for i, d in enumerate(data)]

class TestDumpToReferenceList(unittest.TestCase):

    def test_empty_list(self):
        self.assertEqual(dump_to_reference_list([]), [])

    def test_single_element(self):
        self.assertEqual(dump_to_reference_list(['item']), ['[1]  item\n=====\n\n'])

    def test_multiple_elements(self):
        data = ['item1', 'item2', 'item3']
        expected = ['[1]  item1\n=====\n\n', '[2]  item2\n=====\n\n', '[3]  item3\n=====\n\n']
        self.assertEqual(dump_to_reference_list(data), expected)

    def test_custom_separator(self):
        data = ['item1', 'item2']
        separator = ' | '
        expected = ['[1]  item1 | ', '[2]  item2 | ']
        self.assertEqual(dump_to_reference_list(data, separator), expected)

