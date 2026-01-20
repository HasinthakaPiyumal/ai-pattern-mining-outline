# Cluster 41

class TestParseLabelmap(unittest.TestCase):
    """Test Parsing LabMap."""

    def setUp(self) -> None:
        """Setup function."""
        self.label1 = Label('label1', (1, 1, 1, 1))
        self.label2 = Label('label2', (2, 2, 2, 2))

    def test_empty(self) -> None:
        """Tests empty label map case."""
        id2name, id2color = parse_labelmap_dataclass({})
        self.assertIsInstance(id2name, OrderedDict)
        self.assertIsInstance(id2color, OrderedDict)
        self.assertEqual(len(id2name), 0)
        self.assertEqual(len(id2color), 0)

    def test_one(self) -> None:
        """Tests one label case."""
        num = 1
        mapping = {num: self.label1}
        id2name, id2color = parse_labelmap_dataclass(mapping)
        self.assertEqual(len(id2name), len(mapping))
        self.assertEqual(id2name[num], self.label1.name)
        self.assertEqual(len(id2color), len(mapping))
        self.assertEqual(id2color[num], self.label1.color)

    def test_multiple(self) -> None:
        """Tests multiple labels case."""
        num1, num2 = (1, 2)
        mapping = {num1: self.label1, num2: self.label2}
        id2name, id2color = parse_labelmap_dataclass(mapping)
        self.assertEqual(len(id2name), len(mapping))
        self.assertEqual(len(id2color), len(mapping))
        self.assertEqual(id2name[num1], self.label1.name)
        self.assertEqual(id2name[num2], self.label2.name)
        self.assertEqual(id2color[num1], self.label1.color)
        self.assertEqual(id2color[num2], self.label2.color)
        self.assertEqual(list(id2name.keys())[0], min(num1, num2))
        self.assertEqual(list(id2name.keys())[1], max(num1, num2))
        self.assertEqual(list(id2color.keys())[0], min(num1, num2))
        self.assertEqual(list(id2color.keys())[1], max(num1, num2))

def parse_labelmap_dataclass(labelmap: Dict[int, Label]) -> Tuple[OrderedDict[int, Any], OrderedDict[int, Tuple[Any, ...]]]:
    """
    A labelmap provides a map from integer ids to text and color labels. After loading a label map from json, this
    will parse the labelmap into commonly utilized mappings and fix the formatting issues caused by json.
    :param labelmap: Dictionary of label id and its corresponding Label class information.
    :return: (id2name {id <int>: name <str>}, id2color {id <int>: color (R <int>, G <int>, B <int>, A <int>)}.
        Label id to name and label id to color mappings tuple.
    """
    id2name = OrderedDict()
    id2color = OrderedDict()
    ids = [int(_id) for _id in labelmap.keys()]
    ids.sort()
    for _id in ids:
        id2name[_id] = labelmap[_id].name
        id2color[_id] = tuple(labelmap[_id].color)
    return (id2name, id2color)

