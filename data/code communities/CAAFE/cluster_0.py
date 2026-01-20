# Cluster 0

def check_ast(node: ast.AST) -> None:
    """
    Checks if the given AST node is allowed.

    Parameters:
    node (ast.AST): The AST node to check.

    Raises:
    ValueError: If the AST node is not allowed.
    """
    allowed_nodes = {ast.Module, ast.Expr, ast.Load, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Num, ast.Str, ast.Bytes, ast.List, ast.Tuple, ast.Dict, ast.Name, ast.Call, ast.Attribute, ast.keyword, ast.Subscript, ast.Index, ast.Slice, ast.ExtSlice, ast.Assign, ast.AugAssign, ast.NameConstant, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.And, ast.Or, ast.BitOr, ast.BitAnd, ast.BitXor, ast.Invert, ast.Not, ast.Constant, ast.Store, ast.If, ast.IfExp, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Assert, ast.Return, ast.FunctionDef, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom, ast.Lambda, ast.BoolOp, ast.FormattedValue, ast.JoinedStr, ast.Set, ast.Ellipsis, ast.expr, ast.stmt, ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop, ast.comprehension, ast.arguments, ast.arg, ast.Import, ast.ImportFrom, ast.alias}
    allowed_packages = {'numpy', 'pandas', 'sklearn'}
    allowed_funcs = {'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round, 'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'enumerate': enumerate, 'zip': zip, 'range': range, 'sorted': sorted, 'reversed': reversed}
    allowed_attrs = {'array', 'arange', 'values', 'linspace', 'mean', 'sum', 'contains', 'where', 'min', 'max', 'median', 'std', 'sqrt', 'pow', 'iloc', 'cut', 'qcut', 'inf', 'nan', 'isna', 'map', 'reshape', 'shape', 'split', 'var', 'codes', 'abs', 'cumsum', 'cumprod', 'cummax', 'cummin', 'diff', 'repeat', 'index', 'log', 'log10', 'log1p', 'slice', 'exp', 'expm1', 'pow', 'pct_change', 'corr', 'cov', 'round', 'clip', 'dot', 'transpose', 'T', 'astype', 'copy', 'drop', 'dropna', 'fillna', 'replace', 'merge', 'append', 'join', 'groupby', 'resample', 'rolling', 'expanding', 'ewm', 'agg', 'aggregate', 'filter', 'transform', 'apply', 'pivot', 'melt', 'sort_values', 'sort_index', 'reset_index', 'set_index', 'reindex', 'shift', 'extract', 'rename', 'tail', 'head', 'describe', 'count', 'value_counts', 'unique', 'nunique', 'idxmin', 'idxmax', 'isin', 'between', 'duplicated', 'rank', 'to_numpy', 'to_dict', 'to_list', 'to_frame', 'squeeze', 'add', 'sub', 'mul', 'div', 'mod', 'columns', 'loc', 'lt', 'le', 'eq', 'ne', 'ge', 'gt', 'all', 'any', 'clip', 'conj', 'conjugate', 'round', 'trace', 'cumprod', 'cumsum', 'prod', 'dot', 'flatten', 'ravel', 'T', 'transpose', 'swapaxes', 'clip', 'item', 'tolist', 'argmax', 'argmin', 'argsort', 'max', 'mean', 'min', 'nonzero', 'ptp', 'sort', 'std', 'var', 'str', 'dt', 'cat', 'sparse', 'plot'}
    if type(node) not in allowed_nodes:
        raise ValueError(f'Disallowed code: {ast.unparse(node)} is {type(node)}')
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in allowed_funcs:
            raise ValueError(f'Disallowed function: {node.func.id}')
    if isinstance(node, ast.Attribute) and node.attr not in allowed_attrs:
        raise ValueError(f'Disallowed attribute: {node.attr}')
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name not in allowed_packages:
                raise ValueError(f'Disallowed package import: {alias.name}')
    for child in ast.iter_child_nodes(node):
        check_ast(child)

