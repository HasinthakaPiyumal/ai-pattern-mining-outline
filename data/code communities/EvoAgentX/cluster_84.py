# Cluster 84

class TestModule(unittest.TestCase):

    def setUp(self):
        self.llm_config = OpenAILLMConfig(model='gpt-4o-mini', openai_key='XXX')
        self.sample_tasks = [{'name': 'Task1', 'description': 'First task in the sequence', 'inputs': [{'name': 'input1', 'type': 'string', 'required': True, 'description': 'Input for Task1'}], 'outputs': [{'name': 'output1', 'type': 'string', 'required': True, 'description': 'Output from Task1'}], 'prompt': 'Execute Task1'}, {'name': 'Task2', 'description': 'Second task in the sequence', 'inputs': [{'name': 'output1', 'type': 'string', 'required': True, 'description': 'Input from Task1'}], 'outputs': [{'name': 'output2', 'type': 'string', 'required': True, 'description': 'Output from Task2'}], 'prompt': 'Execute Task2'}, {'name': 'Task3', 'description': 'Third task in the sequence', 'inputs': [{'name': 'output2', 'type': 'string', 'required': True, 'description': 'Input from Task2'}], 'outputs': [{'name': 'final_output', 'type': 'string', 'required': True, 'description': 'Final output'}], 'prompt': 'Execute Task3', 'parse_mode': 'custom', 'parse_func': custom_parse_func}]

    def tearDown(self):
        if os.path.exists('tests/workflow/test_workflow.json'):
            os.remove('tests/workflow/test_workflow.json')

    def test_sequential_workflow_graph_creation(self):
        """Test that a sequential workflow graph is created correctly."""
        graph = SequentialWorkFlowGraph(goal='Test Workflow', tasks=self.sample_tasks)
        self.assertEqual('Test Workflow', graph.goal)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual(2, len(graph.edges))
        node_names = [node.name for node in graph.nodes]
        self.assertListEqual(['Task1', 'Task2', 'Task3'], node_names)
        edge_connections = [(edge.source, edge.target) for edge in graph.edges]
        self.assertIn(('Task1', 'Task2'), edge_connections)
        self.assertIn(('Task2', 'Task3'), edge_connections)

    def test_sequential_workflow_node_properties(self):
        """Test that nodes in the workflow have correct properties."""
        graph = SequentialWorkFlowGraph(goal='Test Workflow', tasks=self.sample_tasks)
        node1 = graph.get_node('Task1')
        self.assertEqual('Task1', node1.name)
        self.assertEqual('First task in the sequence', node1.description)
        self.assertEqual(1, len(node1.inputs))
        self.assertEqual(1, len(node1.outputs))
        self.assertEqual('input1', node1.inputs[0].name)
        self.assertEqual('output1', node1.outputs[0].name)
        self.assertTrue(len(node1.agents) > 0)
        agent = node1.agents[0]
        self.assertEqual('Execute Task1', agent.get('prompt'))
        node3 = graph.get_node('Task3')
        self.assertEqual('custom', node3.agents[0]['parse_mode'])
        self.assertEqual(custom_parse_func, node3.agents[0]['parse_func'])

    def test_sequential_workflow_execution_flow(self):
        """Test the execution flow of a sequential workflow."""
        graph = SequentialWorkFlowGraph(goal='Test Workflow', tasks=self.sample_tasks)
        for node in graph.nodes:
            self.assertEqual(WorkFlowNodeState.PENDING, node.status)
        next_nodes = graph.next()
        self.assertEqual(1, len(next_nodes))
        self.assertEqual('Task1', next_nodes[0].name)
        graph.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        next_nodes = graph.next()
        self.assertEqual(1, len(next_nodes))
        self.assertEqual('Task2', next_nodes[0].name)
        graph.set_node_status('Task2', WorkFlowNodeState.COMPLETED)
        next_nodes = graph.next()
        self.assertEqual(1, len(next_nodes))
        self.assertEqual('Task3', next_nodes[0].name)
        graph.set_node_status('Task3', WorkFlowNodeState.COMPLETED)
        self.assertTrue(graph.is_complete)
        next_nodes = graph.next()
        self.assertEqual(0, len(next_nodes))

    def test_sequential_workflow_save_and_load(self):
        """Test saving and loading a sequential workflow."""
        graph = SequentialWorkFlowGraph(goal='Test Workflow', tasks=self.sample_tasks)
        save_path = 'tests/workflow/test_workflow.json'
        graph.save_module(save_path)
        self.assertTrue(os.path.exists(save_path))
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        self.assertEqual('SequentialWorkFlowGraph', saved_data['class_name'])
        self.assertEqual('Test Workflow', saved_data['goal'])
        self.assertEqual(3, len(saved_data['tasks']))
        self.assertEqual('custom_parse_func', saved_data['tasks'][2]['parse_func'])
        task_names = [task['name'] for task in saved_data['tasks']]
        self.assertListEqual(['Task1', 'Task2', 'Task3'], task_names)
        loaded_graph: SequentialWorkFlowGraph = SequentialWorkFlowGraph.from_file(save_path)
        self.assertEqual('Test Workflow', loaded_graph.goal)
        self.assertEqual(3, len(loaded_graph.nodes))
        self.assertEqual('custom_parse_func', loaded_graph.get_node('Task3').agents[0]['parse_func'])

    def test_node_status_management(self):
        """Test that node status can be properly managed."""
        graph = SequentialWorkFlowGraph(goal='Test Workflow', tasks=self.sample_tasks)
        self.assertEqual(WorkFlowNodeState.PENDING, graph.get_node_status('Task1'))
        graph.set_node_status('Task1', WorkFlowNodeState.RUNNING)
        self.assertEqual(WorkFlowNodeState.RUNNING, graph.get_node_status('Task1'))
        self.assertTrue(graph.running('Task1'))
        graph.set_node_status('Task1', WorkFlowNodeState.COMPLETED)
        self.assertEqual(WorkFlowNodeState.COMPLETED, graph.get_node_status('Task1'))
        self.assertTrue(graph.completed('Task1'))
        graph.set_node_status('Task2', WorkFlowNodeState.FAILED)
        self.assertEqual(WorkFlowNodeState.FAILED, graph.get_node_status('Task2'))
        self.assertTrue(graph.failed('Task2'))

    def test_graph_reset(self):
        """Test that the graph can be reset to initial state."""
        graph = SequentialWorkFlowGraph(goal='Test Workflow', tasks=self.sample_tasks)
        for node in graph.nodes:
            graph.set_node_status(node.name, WorkFlowNodeState.COMPLETED)
        for node in graph.nodes:
            self.assertEqual(WorkFlowNodeState.COMPLETED, node.status)
        graph.reset_graph()
        for node in graph.nodes:
            self.assertEqual(WorkFlowNodeState.PENDING, node.status)

