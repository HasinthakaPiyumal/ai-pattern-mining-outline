# Cluster 35

def normalize_text(s: str) -> str:

    def remove_articles(text):
        return regex.sub('\\b(a|an|the)\\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return text.replace('_', ' ')

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def white_space_fix(text):
    return ' '.join(text.split())

def remove_articles(text):
    return regex.sub('\\b(a|an|the)\\b', ' ', text)

def remove_punc(text):
    return text.replace('_', ' ')

def lower(text):
    return text.lower()

def normalize_answer(s: str) -> str:

    def remove_articles(text: str) -> str:
        return regex.sub('\\b(a|an|the)\\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join((ch for ch in text if ch not in exclude))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    assert isinstance(ground_truth, str), f'ground_truth must be a string, but got {type(ground_truth)}'
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def ems(prediction: str, ground_truths: List[str]) -> float:
    assert isinstance(ground_truths, list), f'ground_truths must be a list, but got {type(ground_truths)}'
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def f1_score(prediction: str, ground_truth: str) -> float:
    assert isinstance(ground_truth, str), f'ground_truth must be a string, but got {type(ground_truth)}'
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC[0]
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC[0]
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC[0]
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def has_answer(answers, text, match_type='string') -> bool:
    """Check if the text contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)
    tokenizer = SimpleTokenizer()
    if match_type == 'string':
        text = tokenizer.tokenize(text).words(uncased=True)
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i:i + len(single_answer)]:
                    return True
    elif match_type == 'regex':
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = regex.compile(pattern, flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None

def acc_score(prediction: str, ground_truths: List[str]) -> float:
    assert isinstance(ground_truths, list), f'ground_truths must be a list, but got {type(ground_truths)}'
    return float(has_answer(answers=ground_truths, text=prediction, match_type='string'))

class GeneratedAgent(BaseModule):
    """
    Representation of a generated agent with validation capabilities.
    """
    name: str
    description: str
    inputs: List[Parameter]
    outputs: List[Parameter]
    prompt: str
    tool_names: Optional[List[str]] = None

    @classmethod
    def find_output_name(cls, text: str, outputs: List[str]):

        def sim(t1: str, t2: str):
            t1_words = normalize_text(t1).split()
            t2_words = normalize_text(t2).split()
            return len(set(t1_words) & set(t2_words))
        similarities = [sim(text, output) for output in outputs]
        max_sim = max(similarities)
        return outputs[similarities.index(max_sim)]

    @model_validator(mode='after')
    @classmethod
    def validate_prompt(cls, agent: 'GeneratedAgent'):
        """Validate and fix the agent's prompt template.
        
        This validator ensures that:
        1. All input parameters are properly referenced in the prompt
        2. Input references use the correct format with braces
        3. All output sections match the defined output parameters
        
        If there are mismatches in the output sections, it attempts to
        fix them by finding the most similar output name.
        
        Args:
            agent: The GeneratedAgent instance to validate.
            
        Returns:
            The validated and potentially modified GeneratedAgent.
            
        Raises:
            ValueError: If inputs are missing from the prompt or output sections don't match the defined outputs.
        """
        input_names = [inp.name for inp in agent.inputs]
        prompt_has_inputs = [name in agent.prompt for name in input_names]
        if not all(prompt_has_inputs):
            missing_input_names = [name for name, has_input in zip(input_names, prompt_has_inputs) if not has_input]
            raise ValueError(f'The prompt miss inputs: {missing_input_names}')
        pattern = '### Instructions(.*?)### Output Format'
        prompt = agent.prompt

        def replace_with_braces(match):
            instructions = match.group(1)
            for name in input_names:
                instructions = re.sub(f'<input>{{*\\b{re.escape(name)}\\b}}*</input>', f'<input>{{{name}}}</input>', instructions)
            return '### Instructions' + instructions + '### Output Format'
        modified_prompt = re.sub(pattern, replace_with_braces, prompt, flags=re.DOTALL)
        agent.prompt = modified_prompt
        prompt = agent.prompt
        pattern = '### Output Format(.*)'
        outputs_names = [out.name for out in agent.outputs]

        def fix_output_names(match):
            output_format = match.group(1)
            matches = re.findall('## ([^\\n#]+)', output_format, flags=re.DOTALL)
            generated_outputs = [m.strip() for m in matches if m.strip() != 'Thought']
            if len(generated_outputs) != len(outputs_names):
                raise ValueError(f"The number of outputs in the prompt is different from that defined in the `outputs` field of the agent. The outputs in the prompt are: {generated_outputs}, while the outputs from the agent's `outputs` field are: {outputs_names}")
            for generated_output in generated_outputs:
                if generated_output not in outputs_names:
                    most_similar_output_name = cls.find_output_name(text=generated_output, outputs=outputs_names)
                    output_format = output_format.replace(generated_output, most_similar_output_name)
                    logger.warning(f"Couldn't find output name in prompt ('{generated_output}') in agent's outputs. Replace it with the most similar agent output: '{most_similar_output_name}'")
            return '### Output Format' + output_format
        modified_prompt = re.sub(pattern, fix_output_names, prompt, flags=re.DOTALL)
        agent.prompt = modified_prompt
        return agent

class MessageBenchmark(Benchmark):
    """
    Adapt dataset in messages format, automatically extract last user/assistant round.
    """

    def __init__(self, path: str, mode: str='train'):
        super().__init__(name='MessageBenchmark', path=path, mode=mode)

    def _load_data(self):
        import json
        file_path = os.path.join(self.path, 'worfbench_train.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            self._train_data = json.load(f)

    def _get_label(self, example):
        return [m['content'] for m in example['messages'] if m['role'] == 'assistant'][-1]

    def _get_id(self, example):
        user_msg = [m['content'] for m in example['messages'] if m['role'] == 'user'][-1]
        return example.get('source', '') + '_' + user_msg[:20]

    def evaluate(self, prediction, label):
        from evoagentx.benchmark.measures import exact_match_score, f1_score, acc_score
        em = exact_match_score(prediction, label)
        f1 = f1_score(prediction, label)
        acc = acc_score(prediction, [label])
        return {'em': em, 'f1': f1, 'acc': acc}

def f1_chain(prediction: str, label: str) -> float:
    from evoagentx.benchmark.measures import f1_score
    return f1_score(prediction, label)

def semantic_similarity(text1, text2):
    """Calculate semantic similarity"""
    text1_norm = normalize_text(text1)
    text2_norm = normalize_text(text2)
    words1 = set(text1_norm.split())
    words2 = set(text2_norm.split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0

def build_graph_structure(nodes, edges):
    """Build graph structure and calculate topological features"""
    import re
    from collections import defaultdict, deque
    node_ids = set()
    node_content_map = {}
    for node in nodes:
        match = re.match('^(\\d+):\\s*(.+)$', node)
        if match:
            node_id = match.group(1)
            content = match.group(2)
            node_ids.add(node_id)
            node_content_map[node_id] = content
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    for edge in edges:
        match = re.match('\\(([^,]+),\\s*([^)]+)\\)', edge)
        if match:
            from_node = match.group(1).strip()
            to_node = match.group(2).strip()
            if from_node == 'START':
                from_node = '0'
            if to_node == 'END':
                to_node = str(max((int(n) for n in node_ids)) + 1 if node_ids else 1)
            if from_node in node_ids and to_node in node_ids:
                graph[from_node].append(to_node)
                in_degree[to_node] += 1

    def get_topological_features():
        """Calculate topological sorting and features"""
        queue = deque([node for node in node_ids if in_degree[node] == 0])
        topo_order = []
        visited = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            topo_order.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        features = {'node_count': len(node_ids), 'edge_count': len(edges), 'max_depth': len(topo_order), 'avg_branching': sum((len(graph[node]) for node in node_ids)) / len(node_ids) if node_ids else 0, 'parallel_paths': sum((1 for node in node_ids if len(graph[node]) > 1)), 'sequential_paths': sum((1 for node in node_ids if len(graph[node]) == 1))}
        return (features, topo_order)
    return get_topological_features()

def get_topological_features():
    """Calculate topological sorting and features"""
    queue = deque([node for node in node_ids if in_degree[node] == 0])
    topo_order = []
    visited = set()
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        topo_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    features = {'node_count': len(node_ids), 'edge_count': len(edges), 'max_depth': len(topo_order), 'avg_branching': sum((len(graph[node]) for node in node_ids)) / len(node_ids) if node_ids else 0, 'parallel_paths': sum((1 for node in node_ids if len(graph[node]) > 1)), 'sequential_paths': sum((1 for node in node_ids if len(graph[node]) == 1))}
    return (features, topo_order)

def structural_similarity(pred_nodes, pred_edges, label_nodes, label_edges):
    """Calculate graph structure similarity"""
    try:
        pred_features, pred_topo = build_graph_structure(pred_nodes, pred_edges)
        label_features, label_topo = build_graph_structure(label_nodes, label_edges)
        feature_similarity = 0
        total_features = 0
        for key in pred_features:
            if key in label_features:
                pred_val = pred_features[key]
                label_val = label_features[key]
                if pred_val == 0 and label_val == 0:
                    similarity = 1.0
                elif pred_val == 0 or label_val == 0:
                    similarity = 0.0
                else:
                    similarity = min(pred_val, label_val) / max(pred_val, label_val)
                feature_similarity += similarity
                total_features += 1
        avg_feature_similarity = feature_similarity / total_features if total_features > 0 else 0.0
        topo_similarity = 0
        if pred_topo and label_topo:
            common_nodes = set(pred_topo) & set(label_topo)
            if common_nodes:
                pred_positions = {node: i for i, node in enumerate(pred_topo)}
                label_positions = {node: i for i, node in enumerate(label_topo)}
                position_diffs = []
                for node in common_nodes:
                    diff = abs(pred_positions[node] - label_positions[node])
                    position_diffs.append(diff)
                if position_diffs:
                    avg_diff = sum(position_diffs) / len(position_diffs)
                    max_possible_diff = max(len(pred_topo), len(label_topo))
                    topo_similarity = 1.0 - avg_diff / max_possible_diff
        return 0.6 * avg_feature_similarity + 0.4 * topo_similarity
    except Exception as e:
        print(f'Error calculating structural similarity: {e}')
        return 0.0

def improved_f1(set_pred, set_label, similarity_threshold=0.7):
    """Improved F1 calculation considering semantic similarity"""
    if not set_pred or not set_label:
        return 0.0
    exact_matches = len(set_pred & set_label)
    semantic_matches = 0
    for pred_item in set_pred:
        if pred_item in set_label:
            continue
        for label_item in set_label:
            if label_item in set_pred:
                continue
            if semantic_similarity(pred_item, label_item) >= similarity_threshold:
                semantic_matches += 1
                break
    total_matches = exact_matches + semantic_matches
    precision = total_matches / len(set_pred) if set_pred else 0.0
    recall = total_matches / len(set_label) if set_label else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

def f1_graph(prediction: str, label: str) -> float:

    def parse_graph_improved(text):
        """Improved graph parsing function for accurate node and edge extraction"""
        import re
        nodes = []
        edges = []
        text = text.strip()
        lines = text.splitlines()
        node_section = False
        edge_section = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('node:'):
                node_section = True
                edge_section = False
                continue
            if line.lower().startswith('edge:'):
                edge_section = True
                node_section = False
                continue
            if node_section:
                node_match = re.match('^(\\d+):\\s*(.+)$', line)
                if node_match:
                    node_num = node_match.group(1)
                    node_content = node_match.group(2).strip()
                    nodes.append(f'{node_num}: {node_content}')
            if edge_section:
                edge_matches = re.findall('\\(([^)]+)\\)', line)
                for edge_match in edge_matches:
                    edge = edge_match.strip()
                    if edge and ',' in edge:
                        edges.append(f'({edge})')
        return (set(nodes), set(edges))

    def normalize_text(text):
        """Normalize text for better matching accuracy"""
        import re
        text = re.sub('\\s+', ' ', text)
        text = text.replace(',', ',').replace('.', '.').replace(':', ':')
        return text.strip().lower()

    def semantic_similarity(text1, text2):
        """Calculate semantic similarity"""
        text1_norm = normalize_text(text1)
        text2_norm = normalize_text(text2)
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def build_graph_structure(nodes, edges):
        """Build graph structure and calculate topological features"""
        import re
        from collections import defaultdict, deque
        node_ids = set()
        node_content_map = {}
        for node in nodes:
            match = re.match('^(\\d+):\\s*(.+)$', node)
            if match:
                node_id = match.group(1)
                content = match.group(2)
                node_ids.add(node_id)
                node_content_map[node_id] = content
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        for edge in edges:
            match = re.match('\\(([^,]+),\\s*([^)]+)\\)', edge)
            if match:
                from_node = match.group(1).strip()
                to_node = match.group(2).strip()
                if from_node == 'START':
                    from_node = '0'
                if to_node == 'END':
                    to_node = str(max((int(n) for n in node_ids)) + 1 if node_ids else 1)
                if from_node in node_ids and to_node in node_ids:
                    graph[from_node].append(to_node)
                    in_degree[to_node] += 1

        def get_topological_features():
            """Calculate topological sorting and features"""
            queue = deque([node for node in node_ids if in_degree[node] == 0])
            topo_order = []
            visited = set()
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                topo_order.append(node)
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            features = {'node_count': len(node_ids), 'edge_count': len(edges), 'max_depth': len(topo_order), 'avg_branching': sum((len(graph[node]) for node in node_ids)) / len(node_ids) if node_ids else 0, 'parallel_paths': sum((1 for node in node_ids if len(graph[node]) > 1)), 'sequential_paths': sum((1 for node in node_ids if len(graph[node]) == 1))}
            return (features, topo_order)
        return get_topological_features()

    def structural_similarity(pred_nodes, pred_edges, label_nodes, label_edges):
        """Calculate graph structure similarity"""
        try:
            pred_features, pred_topo = build_graph_structure(pred_nodes, pred_edges)
            label_features, label_topo = build_graph_structure(label_nodes, label_edges)
            feature_similarity = 0
            total_features = 0
            for key in pred_features:
                if key in label_features:
                    pred_val = pred_features[key]
                    label_val = label_features[key]
                    if pred_val == 0 and label_val == 0:
                        similarity = 1.0
                    elif pred_val == 0 or label_val == 0:
                        similarity = 0.0
                    else:
                        similarity = min(pred_val, label_val) / max(pred_val, label_val)
                    feature_similarity += similarity
                    total_features += 1
            avg_feature_similarity = feature_similarity / total_features if total_features > 0 else 0.0
            topo_similarity = 0
            if pred_topo and label_topo:
                common_nodes = set(pred_topo) & set(label_topo)
                if common_nodes:
                    pred_positions = {node: i for i, node in enumerate(pred_topo)}
                    label_positions = {node: i for i, node in enumerate(label_topo)}
                    position_diffs = []
                    for node in common_nodes:
                        diff = abs(pred_positions[node] - label_positions[node])
                        position_diffs.append(diff)
                    if position_diffs:
                        avg_diff = sum(position_diffs) / len(position_diffs)
                        max_possible_diff = max(len(pred_topo), len(label_topo))
                        topo_similarity = 1.0 - avg_diff / max_possible_diff
            return 0.6 * avg_feature_similarity + 0.4 * topo_similarity
        except Exception as e:
            print(f'Error calculating structural similarity: {e}')
            return 0.0

    def improved_f1(set_pred, set_label, similarity_threshold=0.7):
        """Improved F1 calculation considering semantic similarity"""
        if not set_pred or not set_label:
            return 0.0
        exact_matches = len(set_pred & set_label)
        semantic_matches = 0
        for pred_item in set_pred:
            if pred_item in set_label:
                continue
            for label_item in set_label:
                if label_item in set_pred:
                    continue
                if semantic_similarity(pred_item, label_item) >= similarity_threshold:
                    semantic_matches += 1
                    break
        total_matches = exact_matches + semantic_matches
        precision = total_matches / len(set_pred) if set_pred else 0.0
        recall = total_matches / len(set_label) if set_label else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    pred_nodes, pred_edges = parse_graph_improved(prediction)
    label_nodes, label_edges = parse_graph_improved(label)
    node_f1 = improved_f1(pred_nodes, label_nodes)
    edge_f1 = improved_f1(pred_edges, label_edges)
    structural_sim = structural_similarity(pred_nodes, pred_edges, label_nodes, label_edges)
    semantic_score = 0.6 * node_f1 + 0.4 * edge_f1
    final_score = 0.7 * semantic_score + 0.3 * structural_sim
    return final_score

def parse_graph_improved(text):
    """Improved graph parsing function for accurate node and edge extraction"""
    import re
    nodes = []
    edges = []
    text = text.strip()
    lines = text.splitlines()
    node_section = False
    edge_section = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith('node:'):
            node_section = True
            edge_section = False
            continue
        if line.lower().startswith('edge:'):
            edge_section = True
            node_section = False
            continue
        if node_section:
            node_match = re.match('^(\\d+):\\s*(.+)$', line)
            if node_match:
                node_num = node_match.group(1)
                node_content = node_match.group(2).strip()
                nodes.append(f'{node_num}: {node_content}')
        if edge_section:
            edge_matches = re.findall('\\(([^)]+)\\)', line)
            for edge_match in edge_matches:
                edge = edge_match.strip()
                if edge and ',' in edge:
                    edges.append(f'({edge})')
    return (set(nodes), set(edges))

def collate_func(example: dict) -> dict:
    return {'question': example['prompt']}

def output_postprocess_func(output: dict) -> dict:
    """
        Args:
            output (dict): The output from the workflow.

        Returns: 
            The processed output that can be used to compute the metrics. The output will be directly passed to the benchmark's `evaluate` method. 
        """
    return output['answer']

def parse_workflow(llm_output):
    import re
    nodes = []
    edges = []
    node_section = False
    edge_section = False
    for line in llm_output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith('node:'):
            node_section = True
            edge_section = False
            continue
        if line.lower().startswith('edge:'):
            edge_section = True
            node_section = False
            for em in re.findall('\\(([^)]+)\\)', line):
                if ',' in em:
                    from_idx, to_idx = [x.strip() for x in em.split(',', 1)]
                    edges.append([from_idx, to_idx])
            continue
        if node_section:
            m = re.match('^\\s*(\\d+)[\\.:]\\s*(.+)$', gode=line)
            if m:
                nodes.append(m.group(2).strip())
        if edge_section:
            for em in re.findall('\\(([^)]+)\\)', line):
                if ',' in em:
                    from_idx, to_idx = [x.strip() for x in em.split(',', 1)]
                    edges.append([from_idx, to_idx])
    return (nodes, edges)

