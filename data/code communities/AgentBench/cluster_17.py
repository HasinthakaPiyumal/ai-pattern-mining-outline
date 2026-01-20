# Cluster 17

@DeprecationWarning
def lisp_to_sparql_and(expressions: Union[List[str], str], variable=1):
    assert len(expressions) == 2
    clauses = []
    if not isinstance(expressions[0], list):
        pass
    elif expressions[0][0] == 'JOIN':
        clauses.extend(lisp_to_sparql_join(expressions[0][1:], variable))
    elif expressions[0][0] == 'AND':
        clauses.extend(lisp_to_sparql_and(expressions[0][1:], variable))
    if not isinstance(expressions[1], list):
        clauses.append('?x' + str(variable) + ' :type.object.type ' + expressions[0] + ' .')
    elif expressions[1][0] == 'JOIN':
        clauses.extend(lisp_to_sparql_join(expressions[1][1:], variable))
    elif expressions[1][0] == 'AND':
        clauses.extend(lisp_to_sparql_and(expressions[1][1:], variable))
    return clauses

@DeprecationWarning
def lisp_to_sparql_join(expressions: Union[List[str], str], variable=1):
    assert len(expressions) == 2
    clauses = []
    if not isinstance(expressions[1], list):
        if not isinstance(expressions[0], list):
            clauses.append('?x' + str(variable) + ' :' + expressions[0] + ' :' + expressions[1] + ' .')
        else:
            clauses.append(':' + expressions[1] + ' :' + expressions[0][1] + ' ' + '?x' + str(variable) + ' .')
    elif not isinstance(expressions[0], list):
        if expressions[1][0] == 'JOIN':
            clauses.append('?x' + str(variable) + ' :' + expressions[0] + ' ' + '?x' + str(variable + 1) + ' .')
            clauses.extend(lisp_to_sparql_join(expressions[1][1:], variable + 1))
        elif expressions[1][0] == 'AND':
            clauses.append('?x' + str(variable) + ' :' + expressions[0] + ' ' + '?x' + str(variable + 1) + ' .')
            clauses.extend(lisp_to_sparql_and(expressions[1][1:], variable + 1))
    elif expressions[1][0] == 'JOIN':
        clauses.append('?x' + str(variable + 1) + ' :' + expressions[0][1] + ' ' + '?x' + str(variable) + ' .')
        clauses.extend(lisp_to_sparql_join(expressions[1][1:], variable + 1))
    elif expressions[1][0] == 'AND':
        clauses.append('?x' + str(variable + 1) + ' :' + expressions[0][1] + ' ' + '?x' + str(variable) + ' .')
        clauses.extend(lisp_to_sparql_and(expressions[1][1:], variable + 1))
    return clauses

