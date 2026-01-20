# Cluster 12

def code_extract(text: str) -> str:
    lines = text.split('\n')
    longest_line_pair = (0, 0)
    longest_so_far = 0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = '\n'.join(lines[i:j + 1])
            if syntax_check(current_lines):
                current_length = sum((1 for line in lines[i:j + 1] if line.strip()))
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)
    return '\n'.join(lines[longest_line_pair[0]:longest_line_pair[1] + 1])

def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False

def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == NodeType.RETURN.value:
            return True
    return False

def traverse_tree(node: Node) -> Generator[Node, None, None]:
    """
    Traverse the tree structure starting from the given node.

    :param node: The root node to start the traversal from.
    :return: A generator object that yields nodes in the tree.
    """
    cursor = node.walk()
    depth = 0
    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1

def dfs_get_deps(node: Node, deps: Set[str]) -> None:
    for child in node.children:
        if child.type == NodeType.IDENTIFIER.value:
            deps.add(child.text.decode('utf8'))
        else:
            dfs_get_deps(child, deps)

def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:

    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == NodeType.IDENTIFIER.value:
                deps.add(child.text.decode('utf8'))
            else:
                dfs_get_deps(child, deps)
    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps

def sanitize(code: str, entrypoint: Optional[str]=None) -> str:
    """
    Sanitize and extract relevant parts of the given Python code.
    This function parses the input code, extracts import statements, class and function definitions,
    and variable assignments. If an entrypoint is provided, it only includes definitions that are
    reachable from the entrypoint in the call graph.

    :param code: The input Python code as a string.
    :param entrypoint: Optional name of a function to use as the entrypoint for dependency analysis.
    :return: A sanitized version of the input code, containing only relevant parts.
    """
    code = code_extract(code)
    code_bytes = bytes(code, 'utf8')
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)
    class_names = set()
    function_names = set()
    variable_names = set()
    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []
    for child in root_node.children:
        if child.type in NodeType.IMPORT.value:
            import_nodes.append(child)
        elif child.type == NodeType.CLASS.value:
            name = get_definition_name(child)
            if not (name in class_names or name in variable_names or name in function_names):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == NodeType.FUNCTION.value:
            name = get_definition_name(child)
            if not (name in function_names or name in variable_names or name in class_names) and has_return_statement(child):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif child.type == NodeType.EXPRESSION.value and child.children[0].type == NodeType.ASSIGNMENT.value:
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (name in variable_names or name in function_names or name in class_names):
                definition_nodes.append((name, subchild))
                variable_names.add(name)
    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reacheable = get_function_dependency(entrypoint, name2deps)
    sanitized_output = b''
    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte:node.end_byte] + b'\n'
    for pair in definition_nodes:
        name, node = pair
        if entrypoint and name not in reacheable:
            continue
        sanitized_output += code_bytes[node.start_byte:node.end_byte] + b'\n'
    return sanitized_output[:-1].decode('utf8')

def get_definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == NodeType.IDENTIFIER.value:
            return child.text.decode('utf8')

def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited

class CodingBenchmark(Benchmark):
    """
    Abstract base class for defining coding benchmarks. This class provides methods to check the solution code.
    """

    def __init__(self, name: str, path: str, mode: str='all', timeout: int=60, **kwargs):
        self.SUCCESS = 0
        self.FAILED = 1
        self.TIMEOUT = 2
        self.timeout = timeout
        super().__init__(name=name, path=path, mode=mode, **kwargs)

    def handle_special_cases(self, task_id: str, solution: str, test: str) -> bool:
        return (solution, test)

    def _check_evaluation_inputs(self, prediction: Any, label: Any) -> bool:
        """
        Check if the inputs are valid for evaluation.
        """
        assert isinstance(prediction, str) or isinstance(prediction, list), 'prediction must be a string or a list of strings, but got {}'.format(type(prediction))
        assert isinstance(label, dict) or isinstance(label, list), 'label must be a string or a list of strings, but got {}'.format(type(label))
        prediction = [prediction] if isinstance(prediction, str) else prediction
        label = [label] if isinstance(label, dict) else label
        return (prediction, label)

    def check_solution(self, task_id: str, solution: str, test: str, entry_point: Optional[str]=None, use_entrypoint_as_input: bool=True) -> Tuple[int, str]:
        """
        Execute the solution code and check if it passes the unit test.

        Args:
            task_id (str): The task id.
            solution (str): The solution code.
            test (str): The unit test code in HumanEval format. 
            entry_point (str): The entry point of the solution code.
        Returns:
            Tuple[int, str]: A tuple containing an integer indicating whether the solution passes the unit test (0: success, 1: failed, 2: timeout) and a string containing the success/error message.
        """
        solution = sanitize(solution, entrypoint=entry_point)
        try:
            global_dict = {'math': __import__('math'), 'hashlib': __import__('hashlib'), 're': __import__('re'), 'List': List, 'Dict': Dict, 'Tuple': Tuple, 'Optional': Optional, 'Any': Any}
            solution, test = self.handle_special_cases(task_id=task_id, solution=solution, test=test)
            exec(solution, global_dict)
            if entry_point not in global_dict:
                raise ValueError(f'Function {entry_point} not found in the solution code.')
            exec(test, global_dict)
            unit_test_func = global_dict['check']
            with timeout(seconds=self.timeout):
                if use_entrypoint_as_input:
                    unit_test_func(global_dict[entry_point])
                else:
                    unit_test_func()
            result = (self.SUCCESS, 'The solution passed the unit test.')
        except TimeoutException:
            result = (self.TIMEOUT, 'Execution timed out.')
        except Exception as e:
            error_msg = f'An error occurred: {e}\nSolution:\n{solution}\nTest:\n{test}'
            result = (self.FAILED, error_msg)
        return result

    def compute_pass_at_k(self, results: List[bool], k_list: List[int]) -> Dict[str, float]:
        """
        Compute the pass@k for the given results.
        """
        pass_at_k = {}
        n = len(results)
        c = sum(results)
        for k in k_list:
            if n >= k:
                pass_at_k[f'pass@{k}'] = float(estimate_pass_at_k(np.array([n]), np.array([c]), k)[0])
        return pass_at_k

