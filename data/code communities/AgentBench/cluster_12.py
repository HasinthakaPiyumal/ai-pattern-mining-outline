# Cluster 12

def get_derivations_from_lisp(expression: List):
    if expression[0] == 'AND':
        assert len(expression) == 3
        if isinstance(expression[1], str):
            return get_derivations_from_lisp(expression[2])
        else:
            rtn = get_derivations_from_lisp(expression[1])
            rtn.update(get_derivations_from_lisp(expression[2]))
            return rtn
    elif expression[0] in ['ARGMIN', 'ARGMAX']:
        return None
    elif expression[0] == 'COUNT':
        return get_derivations_from_lisp(expression[1])
    elif expression[0] == 'JOIN':
        assert isinstance(expression[1], str)
        if isinstance(expression[2], str):
            rtn = {expression[2]: [':' + expression[1][:-4] if expression[1][-4:] == '_inv' else '^:' + expression[1]]}
            return rtn
        else:
            previous = get_derivations_from_lisp(expression[2])
            for k in previous:
                relation = expression[1]
                if isinstance(previous[k], list):
                    previous[k].extend([':' + relation[:-4] if relation[-4:] == '_inv' else '^:' + relation])
                elif isinstance(previous[k], tuple):
                    previous[k][0].extend([':' + relation[:-4] if relation[-4:] == '_inv' else '^:' + relation])
            return previous
    elif expression[0] in ['le', 'ge', 'lt', 'gt']:
        assert len(expression) == 3 and isinstance(expression[1], str) and isinstance(expression[2], str)
        rtn = {expression[2]: (['^:' + expression[1]], expression[0])}
        return rtn
    elif expression[0] == 'TC':
        assert len(expression) == 4
        return get_derivations_from_lisp(expression[1])

