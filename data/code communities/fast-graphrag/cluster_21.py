# Cluster 21

def dump_to_csv(data: Iterable[object], fields: List[str], separator: str='\t', with_header: bool=False, **values: Dict[str, List[Any]]) -> List[str]:
    rows = list(chain((separator.join(chain(fields, values.keys())),) if with_header else (), chain((separator.join(chain((str(getattr(d, field)).replace('\n', '  ').replace('\t', ' ') for field in fields), (str(v).replace('\n', '  ').replace('\t', ' ') for v in vs))) for d, *vs in zip(data, *values.values())))))
    return rows

class TestDumpToCsv(unittest.TestCase):

    def test_empty_data(self):
        self.assertEqual(dump_to_csv([], ['field1', 'field2']), [])

    def test_single_element(self):

        class Data:

            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2
        data = [Data('value1', 'value2')]
        expected = ['value1\tvalue2']
        self.assertEqual(dump_to_csv(data, ['field1', 'field2']), expected)

    def test_multiple_elements(self):

        class Data:

            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2
        data = [Data('value1', 'value2'), Data('value3', 'value4')]
        expected = ['value1\tvalue2', 'value3\tvalue4']
        self.assertEqual(dump_to_csv(data, ['field1', 'field2']), expected)

    def test_with_header(self):

        class Data:

            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2
        data = [Data('value1', 'value2')]
        expected = ['field1\tfield2', 'value1\tvalue2']
        self.assertEqual(dump_to_csv(data, ['field1', 'field2'], with_header=True), expected)

    def test_custom_separator(self):

        class Data:

            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2
        data = [Data('value1', 'value2')]
        expected = ['value1 | value2']
        self.assertEqual(dump_to_csv(data, ['field1', 'field2'], separator=' | '), expected)

    def test_additional_values(self):

        class Data:

            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2
        data = [Data('value1', 'value2')]
        expected = ['value1\tvalue2\tvalue3']
        self.assertEqual(dump_to_csv(data, ['field1', 'field2'], value3=['value3']), expected)

