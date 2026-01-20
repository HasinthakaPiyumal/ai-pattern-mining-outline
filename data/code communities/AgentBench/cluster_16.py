# Cluster 16

def lisp_to_lambda(expressions: Union[List[str], str]):
    if not isinstance(expressions, list):
        return expressions
    if expressions[0] == 'AND':
        return lisp_to_lambda(expressions[1]) + ' AND ' + lisp_to_lambda(expressions[2])
    elif expressions[0] == 'JOIN':
        return lisp_to_lambda(expressions[1]) + '*' + lisp_to_lambda(expressions[2])

