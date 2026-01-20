# Cluster 64

class TestVizDataAdapter(TestCase):

    def test_to_javascript(self):
        js_str = make_viz_data_adapter().to_javascript()
        self.assertEqual(js_str[:34], 'function getDataAndInfo() { return')
        self.assertEqual(js_str[-3:], '; }')
        json_str = js_str[34:-3]
        self.assertEqual(PAYLOAD, json.loads(json_str))

    def test_to_json(self):
        json_str = make_viz_data_adapter().to_json()
        self.assertEqual(PAYLOAD, json.loads(json_str))

def make_viz_data_adapter():
    return VizDataAdapter(PAYLOAD)

