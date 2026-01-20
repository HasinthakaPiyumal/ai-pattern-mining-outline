# Cluster 5

def expression_to_lisp(expression) -> str:
    rtn = '('
    for i, e in enumerate(expression):
        if isinstance(e, list):
            rtn += expression_to_lisp(e)
        else:
            rtn += e
        if i != len(expression) - 1:
            rtn += ' '
    rtn += ')'
    return rtn

def get_nesting_level(expression) -> int:
    max_sub = 0
    for item in expression:
        if isinstance(item, list):
            level = get_nesting_level(item)
            if level > max_sub:
                max_sub = level
    return 1 + max_sub

def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]

def get_answer_type(query: str):
    try:
        expression = lisp_to_nested_expression(query)
        G = logical_form_to_graph(expression)
        for node in G.nodes.items():
            if 'question_node' in node[1] and node[1]['question_node'] == 1:
                return node[1]['id']
    except Exception:
        return None

def same_logical_form(form1: str, form2: str) -> bool:
    if form1.__contains__('@@UNKNOWN@@') or form2.__contains__('@@UNKNOWN@@'):
        return False
    try:
        G1 = logical_form_to_graph(lisp_to_nested_expression(form1))
    except Exception:
        return False
    try:
        G2 = logical_form_to_graph(lisp_to_nested_expression(form2))
    except Exception:
        return False

    def node_match(n1, n2):
        if n1['id'] == n2['id'] and n1['type'] == n2['type']:
            func1 = n1.pop('function', 'none')
            func2 = n2.pop('function', 'none')
            tc1 = n1.pop('tc', 'none')
            tc2 = n2.pop('tc', 'none')
            if func1 == func2 and tc1 == tc2:
                return True
            else:
                return False
        else:
            return False

    def multi_edge_match(e1, e2):
        if len(e1) != len(e2):
            return False
        values1 = []
        values2 = []
        for v in e1.values():
            values1.append(v['relation'])
        for v in e2.values():
            values2.append(v['relation'])
        return sorted(values1) == sorted(values2)
    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=multi_edge_match)

def binary_nesting(function: str, elements: List[str], types_along_path=None) -> str:
    if len(elements) < 2:
        print('error: binary function should have 2 parameters!')
    if not types_along_path:
        if len(elements) == 2:
            return '(' + function + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + elements[0] + ' ' + binary_nesting(function, elements[1:]) + ')'
    elif len(elements) == 2:
        return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' + elements[1] + ')'
    else:
        return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' + binary_nesting(function, elements[1:], types_along_path[1:]) + ')'

def get_canonical_lisp(logical_form: str):
    expression = lisp_to_nested_expression(logical_form)
    new_expression = _anonymize_entities(expression)
    new_logical_form = expression_to_lisp(new_expression)
    return new_logical_form

def postprocess_raw_code(raw_lisp):
    expression = lisp_to_nested_expression(raw_lisp)
    if expression[0] in ['ARGMAX', 'ARGMIN'] and len(expression) > 3:
        expression[2] = binary_nesting('JOIN', expression[2:])
        expression = expression[:3]
        raw_lisp = expression_to_lisp(expression)
    splits = raw_lisp.split(' ')
    for i, s in enumerate(splits):
        if len(s) > 4 and s[-4:] == '_inv':
            splits[i] = f'(R {s[:-4]})'
        if len(s) > 5 and s[-5:] == '_inv)':
            splits[i] = f'(R {s[:-5]}))'
    processed_lisp = ' '.join(splits)
    return processed_lisp

def max_count_relations(program: str):
    expression = lisp_to_nested_expression(program)
    relations_count = count_relations_expression(expression)
    max = 0
    for r in relations_count:
        if relations_count[r] > max:
            max = relations_count[r]
    return max

