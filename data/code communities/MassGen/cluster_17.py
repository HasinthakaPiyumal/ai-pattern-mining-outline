# Cluster 17

def _safe_eval(node):
    """Safely evaluate an AST node"""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        if node.id in safe_functions:
            return safe_functions[node.id]
        else:
            raise ValueError(f'Unknown variable: {node.id}')
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if type(node.op) in safe_operators:
            return safe_operators[type(node.op)](left, right)
        else:
            raise ValueError(f'Unsupported operation: {type(node.op)}')
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        if type(node.op) in safe_operators:
            return safe_operators[type(node.op)](operand)
        else:
            raise ValueError(f'Unsupported unary operation: {type(node.op)}')
    elif isinstance(node, ast.Call):
        func = _safe_eval(node.func)
        args = [_safe_eval(arg) for arg in node.args]
        return func(*args)
    else:
        raise ValueError(f'Unsupported node type: {type(node)}')

def calculator(expression: str) -> float:
    """
    Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)')
    """
    safe_operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}
    safe_functions = {'abs': abs, 'round': round, 'max': max, 'min': min, 'sum': sum, 'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'log': math.log, 'log10': math.log10, 'exp': math.exp, 'pi': math.pi, 'e': math.e}

    def _safe_eval(node):
        """Safely evaluate an AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in safe_functions:
                return safe_functions[node.id]
            else:
                raise ValueError(f'Unknown variable: {node.id}')
        elif isinstance(node, ast.BinOp):
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            if type(node.op) in safe_operators:
                return safe_operators[type(node.op)](left, right)
            else:
                raise ValueError(f'Unsupported operation: {type(node.op)}')
        elif isinstance(node, ast.UnaryOp):
            operand = _safe_eval(node.operand)
            if type(node.op) in safe_operators:
                return safe_operators[type(node.op)](operand)
            else:
                raise ValueError(f'Unsupported unary operation: {type(node.op)}')
        elif isinstance(node, ast.Call):
            func = _safe_eval(node.func)
            args = [_safe_eval(arg) for arg in node.args]
            return func(*args)
        else:
            raise ValueError(f'Unsupported node type: {type(node)}')
    try:
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree.body)
        return {'expression': expression, 'result': result, 'success': True}
    except Exception as e:
        return {'expression': expression, 'error': str(e), 'success': False}

