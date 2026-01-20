# Cluster 38

class TestModule(unittest.TestCase):

    def setUp(self):
        self.model = OpenAILLM(config=OpenAILLMConfig(model='gpt-4o-mini', openai_key='XXX'))
        self.graph = SEWWorkFlowGraph(llm=self.model)
        self.scheme = SEWWorkFlowScheme(self.graph)
    '\n    def test_python_scheme(self):\n\n        repr = self.scheme.convert_to_scheme(scheme="python")\n        new_graph = self.scheme.parse_workflow_python_repr("```python\n" + repr + "\n```")\n        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))\n        self.assertEqual(len(new_graph.edges), len(self.graph.edges))\n        self.assertFalse(new_graph == self.graph)\n\n        # test empty repr \n        new_graph = self.scheme.parse_workflow_python_repr("")\n        self.assertEqual(new_graph, self.graph)\n\n        # test invalid repr \n        new_graph = self.scheme.parse_workflow_python_repr("invalid repr")\n        self.assertEqual(new_graph, self.graph)\n\n        # test create new graph  \n        steps = eval(repr.replace("steps = ", "").strip())\n        new_steps = steps + [{"name": "test", "args": ["test_input", "code"], "outputs": ["test_output"]}]\n        new_repr = "steps = " + str(new_steps)\n        new_graph = self.scheme.parse_workflow_python_repr("```python\n" + new_repr + "\n```")\n        new_graph_info = new_graph.get_graph_info() \n        self.assertEqual(len(new_graph_info["tasks"]), 3) \n        self.assertEqual(new_graph_info["tasks"][-1]["name"], "test") \n        new_task_inputs = [input_info["name"] for input_info in new_graph_info["tasks"][-1]["inputs"]]\n        self.assertEqual(new_task_inputs, ["test_input", "code"])\n        new_task_outputs = [output_info["name"] for output_info in new_graph_info["tasks"][-1]["outputs"]]\n        self.assertEqual(new_task_outputs, ["test_output"])\n        self.assertFalse(new_graph == self.graph)\n    '

    def test_yaml_scheme(self):
        repr = self.scheme.convert_to_scheme(scheme='yaml')
        new_graph = self.scheme.parse_workflow_yaml_repr('```yaml\n' + repr + '\n```')
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertFalse(new_graph == self.graph)
        new_graph = self.scheme.parse_workflow_yaml_repr('')
        self.assertEqual(new_graph, self.graph)
        new_graph = self.scheme.parse_workflow_yaml_repr('invalid repr')
        self.assertEqual(new_graph, self.graph)

    def test_code_scheme(self):
        repr = self.scheme.convert_to_scheme(scheme='code')
        new_graph = self.scheme.parse_workflow_code_repr('```code\n' + repr + '\n```')
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertFalse(new_graph == self.graph)
        new_graph = self.scheme.parse_workflow_code_repr('')
        self.assertEqual(new_graph, self.graph)
        new_graph = self.scheme.parse_workflow_code_repr('invalid repr')
        self.assertEqual(new_graph, self.graph)

    def test_bpmn_scheme(self):
        repr = self.scheme.convert_to_scheme(scheme='bpmn')
        new_graph = self.scheme.parse_workflow_bpmn_repr('```bpmn\n' + repr + '\n```')
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertFalse(new_graph == self.graph)
        new_graph = self.scheme.parse_workflow_bpmn_repr('')
        self.assertEqual(new_graph, self.graph)
        new_graph = self.scheme.parse_workflow_bpmn_repr('invalid repr')
        self.assertEqual(new_graph, self.graph)

    def test_core_scheme(self):
        repr = self.scheme.convert_to_scheme(scheme='core')
        new_graph = self.scheme.parse_workflow_core_repr('```core\n' + repr + '\n```')
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertFalse(new_graph == self.graph)
        new_graph = self.scheme.parse_workflow_core_repr('')
        self.assertEqual(new_graph, self.graph)
        new_graph = self.scheme.parse_workflow_core_repr('invalid repr')
        self.assertEqual(new_graph, self.graph)

