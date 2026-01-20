# Cluster 83

class TestWorkFlowEditor(unittest.IsolatedAsyncioTestCase):
    """Test the WorkFlowEditor class"""

    def setUp(self):
        """Test preparation"""
        os.environ['PYTEST_CURRENT_TEST'] = 'test_workflow_editor.py::TestWorkFlowEditor'
        self.temp_dir = tempfile.mkdtemp()
        self.test_workflow_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'output', 'tetris_game', 'workflow_demo_4o_mini.json')
        if not os.path.exists(self.test_workflow_file):
            self.test_workflow_file = os.path.join(self.temp_dir, 'test_workflow.json')
            test_workflow = {'nodes': [{'name': 'node1', 'type': 'start'}, {'name': 'node2', 'type': 'process'}, {'name': 'node3', 'type': 'end'}], 'edges': [{'source': 'node1', 'target': 'node2'}, {'source': 'node2', 'target': 'node3'}]}
            with open(self.test_workflow_file, 'w', encoding='utf-8') as f:
                json.dump(test_workflow, f, indent=2, ensure_ascii=False)
        self.test_instruction = 'delete the last node which is not useful in our case'
        with open(self.test_workflow_file, 'r', encoding='utf-8') as f:
            original_workflow = json.load(f)
        self.expected_optimized_workflow = original_workflow.copy()
        if self.expected_optimized_workflow['nodes']:
            last_node = self.expected_optimized_workflow['nodes'][-1]
            self.expected_optimized_workflow['nodes'] = self.expected_optimized_workflow['nodes'][:-1]
            self.expected_optimized_workflow['edges'] = [edge for edge in self.expected_optimized_workflow['edges'] if edge['target'] != last_node['name'] and edge['source'] != last_node['name']]

    def tearDown(self):
        """Test cleanup"""
        if 'PYTEST_CURRENT_TEST' in os.environ:
            del os.environ['PYTEST_CURRENT_TEST']
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_workflow_editor_instantiation(self):
        """Test the instantiation of the WorkFlowEditor class"""
        editor = WorkFlowEditor(save_dir=self.temp_dir)
        self.assertIsInstance(editor, WorkFlowEditor)
        self.assertEqual(editor.save_dir, self.temp_dir)
        self.assertEqual(editor.max_retries, 3)
        self.assertIsNotNone(editor.llm)
        custom_mock_llm = MockLLM()
        editor_custom = WorkFlowEditor(save_dir=self.temp_dir, llm=custom_mock_llm, max_retries=5)
        self.assertIsInstance(editor_custom, WorkFlowEditor)
        self.assertEqual(editor_custom.save_dir, self.temp_dir)
        self.assertEqual(editor_custom.max_retries, 5)
        self.assertEqual(editor_custom.llm, custom_mock_llm)

    @patch('evoagentx.workflow.workflow.WorkFlow')
    @patch('evoagentx.workflow.workflow_graph.WorkFlowGraph')
    async def test_edit_workflow_without_new_file_path(self, mock_workflow_graph, mock_workflow):
        """Test the edit_workflow method (without providing the new_file_path parameter)"""
        mock_workflow_graph.from_dict.return_value = MagicMock()
        mock_workflow.return_value = MagicMock()
        editor = WorkFlowEditor(save_dir=self.temp_dir)
        custom_mock_llm = MockLLM()

        async def mock_generate_async(messages, **kwargs):
            return json.dumps(self.expected_optimized_workflow)
        custom_mock_llm.single_generate_async = mock_generate_async
        editor.llm = custom_mock_llm
        result = await editor.edit_workflow(file_path=self.test_workflow_file, instruction=self.test_instruction)
        self.assertIsInstance(result, WorkFlowEditorReturn)
        self.assertEqual(result.status, 'success')
        self.assertIsNotNone(result.workflow_json)
        self.assertIsNotNone(result.workflow_json_path)
        self.assertIsNone(result.error_message)
        self.assertTrue(os.path.exists(result.workflow_json_path))
        self.assertIn('new_json_for__', os.path.basename(result.workflow_json_path))
        self.assertTrue(result.workflow_json_path.endswith('.json'))
        with open(result.workflow_json_path, 'r', encoding='utf-8') as f:
            saved_json = json.load(f)
        self.assertEqual(saved_json, self.expected_optimized_workflow)
        if os.path.exists(result.workflow_json_path):
            os.remove(result.workflow_json_path)

    @patch('evoagentx.workflow.workflow.WorkFlow')
    @patch('evoagentx.workflow.workflow_graph.WorkFlowGraph')
    async def test_edit_workflow_with_new_file_path(self, mock_workflow_graph, mock_workflow):
        """Test the edit_workflow method (with the new_file_path parameter)"""
        mock_workflow_graph.from_dict.return_value = MagicMock()
        mock_workflow.return_value = MagicMock()
        editor = WorkFlowEditor(save_dir=self.temp_dir)
        custom_mock_llm = MockLLM()

        async def mock_generate_async(messages, **kwargs):
            return json.dumps(self.expected_optimized_workflow)
        custom_mock_llm.single_generate_async = mock_generate_async
        editor.llm = custom_mock_llm
        temp_file_name = 'test_optimized_workflow.json'
        temp_file_path = os.path.join(self.temp_dir, temp_file_name)
        result = await editor.edit_workflow(file_path=self.test_workflow_file, instruction=self.test_instruction, new_file_path=temp_file_name)
        self.assertIsInstance(result, WorkFlowEditorReturn)
        self.assertEqual(result.status, 'success')
        self.assertIsNotNone(result.workflow_json)
        self.assertEqual(result.workflow_json_path, temp_file_path)
        self.assertIsNone(result.error_message)
        self.assertTrue(os.path.exists(result.workflow_json_path))
        with open(result.workflow_json_path, 'r', encoding='utf-8') as f:
            saved_json = json.load(f)
        self.assertEqual(saved_json, self.expected_optimized_workflow)
        if os.path.exists(result.workflow_json_path):
            os.remove(result.workflow_json_path)

    @patch('evoagentx.workflow.workflow.WorkFlow')
    @patch('evoagentx.workflow.workflow_graph.WorkFlowGraph')
    async def test_edit_workflow_llm_failure(self, mock_workflow_graph, mock_workflow):
        """Test the edit_workflow method when the LLM fails"""
        editor = WorkFlowEditor(save_dir=self.temp_dir)
        custom_mock_llm = MockLLM()

        async def mock_generate_async_failure(messages, **kwargs):
            raise Exception('LLM failure')
        custom_mock_llm.single_generate_async = mock_generate_async_failure
        editor.llm = custom_mock_llm
        result = await editor.edit_workflow(file_path=self.test_workflow_file, instruction=self.test_instruction)
        self.assertIsInstance(result, WorkFlowEditorReturn)
        self.assertEqual(result.status, 'failed')
        self.assertIsNone(result.workflow_json)
        self.assertIsNone(result.workflow_json_path)
        self.assertEqual(result.error_message, 'LLM optimization failed')

    @patch('evoagentx.workflow.workflow.WorkFlow')
    @patch('evoagentx.workflow.workflow_graph.WorkFlowGraph')
    async def test_edit_workflow_invalid_json_structure(self, mock_workflow_graph, mock_workflow):
        """Test the edit_workflow method when the workflow JSON structure validation fails"""
        editor = WorkFlowEditor(save_dir=self.temp_dir)
        custom_mock_llm = MockLLM()

        async def mock_generate_async_invalid(messages, **kwargs):
            return json.dumps({'invalid': 'structure'})
        custom_mock_llm.single_generate_async = mock_generate_async_invalid
        editor.llm = custom_mock_llm
        mock_workflow_graph.from_dict.side_effect = Exception('Invalid structure')
        result = await editor.edit_workflow(file_path=self.test_workflow_file, instruction=self.test_instruction)
        self.assertIsInstance(result, WorkFlowEditorReturn)
        self.assertEqual(result.status, 'failed')
        self.assertIsNone(result.workflow_json)
        self.assertIsNone(result.workflow_json_path)
        self.assertEqual(result.error_message, 'Workflow json structure check failed')

    async def test_edit_workflow_invalid_file_path(self):
        """Test the edit_workflow method when providing an invalid file path"""
        editor = WorkFlowEditor(save_dir=self.temp_dir)
        invalid_path = '/non_existent_directory/test.json'
        with self.assertRaises(FileNotFoundError):
            await editor.edit_workflow(file_path=self.test_workflow_file, instruction=self.test_instruction, new_file_path=invalid_path)

