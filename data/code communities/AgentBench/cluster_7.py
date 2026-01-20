# Cluster 7

def lisp_to_sparql(lisp_program: str):
    clauses = []
    order_clauses = []
    entities = set()
    identical_variables_r = {}
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False
    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        if isinstance(expression[2], list):

            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn
            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)
    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    count = False

    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]
        return var
    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):
                if subp[2][:2] in ['m.', 'g.']:
                    clauses.append('ns:' + subp[2] + ' ns:' + subp[1][1] + ' ?x' + i + ' .')
                    entities.add(subp[2])
                elif subp[2][0] == '#':
                    clauses.append('?x' + subp[2][1:] + ' ns:' + subp[1][1] + ' ?x' + i + ' .')
                else:
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split('^^')[1].split('#')[1]
                        if data_type not in ['integer', 'float', 'dateTime', 'double']:
                            subp[2] = f'"{subp[2].split('^^')[0] + '-08:00'}"^^<{subp[2].split('^^')[1]}>'
                        else:
                            subp[2] = f'"{subp[2].split('^^')[0]}"^^<{subp[2].split('^^')[1]}>'
                    clauses.append(subp[2] + ' ns:' + subp[1][1] + ' ?x' + i + ' .')
            elif subp[2][:2] in ['m.', 'g.']:
                clauses.append('?x' + i + ' ns:' + subp[1] + ' ns:' + subp[2] + ' .')
                entities.add(subp[2])
            elif subp[2][0] == '#':
                clauses.append('?x' + i + ' ns:' + subp[1] + ' ?x' + subp[2][1:] + ' .')
            elif subp[2].__contains__('^^'):
                data_type = subp[2].split('^^')[1].split('#')[1]
                if data_type not in ['integer', 'float', 'dateTime', 'double']:
                    subp[2] = f'"{subp[2].split('^^')[0] + '-08:00'}"^^<{subp[2].split('^^')[1]}>'
                else:
                    subp[2] = f'"{subp[2].split('^^')[0]}"^^<{subp[2].split('^^')[1]}>'
                clauses.append('?x' + i + ' ns:' + subp[1] + ' ' + subp[2] + ' .')
            else:
                clauses.append(f'?x ns:{subp[1]} ?obj .')
                clauses.append(f'FILTER (str(?obj) = "{subp[2]}") .')
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            if subp[1][0] == '#':
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                if root1 > root2:
                    identical_variables_r[root1] = root2
                else:
                    identical_variables_r[root2] = root1
            else:
                clauses.append('?x' + i + ' ns:type.object.type ns:' + subp[1] + ' .')
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:
            clauses.append('?x' + i + ' ns:' + subp[1] + ' ?y' + i + ' .')
            if subp[0] == 'le':
                op = '<='
            elif subp[0] == 'lt':
                op = '<'
            elif subp[0] == 'ge':
                op = '>='
            else:
                op = '>'
            if subp[2].__contains__('^^'):
                data_type = subp[2].split('^^')[1].split('#')[1]
                if data_type not in ['integer', 'float', 'dateTime', 'double']:
                    subp[2] = f'"{subp[2].split('^^')[0] + '-08:00'}"^^<{subp[2].split('^^')[1]}>'
                else:
                    subp[2] = f'"{subp[2].split('^^')[0]}"^^<{subp[2].split('^^')[1]}>'
            clauses.append(f'FILTER (?y{i} {op} {subp[2]})')
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti
            year = subp[3]
            if year == 'NOW':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'
            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            if subp[2][-4:] == 'from':
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-4] + 'to'} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-4] + 'to'} ?sk3 . ')
            else:
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-9] + 'to_date'} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-9] + 'to_date'} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')
        elif subp[0] in ['ARGMIN', 'ARGMAX']:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')
            if len(subp) == 3:
                clauses.append(f'?x{i} ns:{subp[2]} ?sk0 .')
            elif len(subp) > 3:
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')
                clauses.append(f'?c{j} ns:{subp[-1]} ?sk0 .')
            if subp[0] == 'ARGMIN':
                order_clauses.append('ORDER BY ?sk0')
            elif subp[0] == 'ARGMAX':
                order_clauses.append('ORDER BY DESC(?sk0)')
            order_clauses.append('LIMIT 1')
        elif subp[0] == 'COUNT':
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var
            count = True
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
    question_var = get_root(question_var)
    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')
    if superlative:
        arg_clauses = clauses[:]
    for entity in entities:
        clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0, f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, 'WHERE {')
    if count:
        clauses.insert(0, f'SELECT COUNT DISTINCT ?x')
    elif superlative:
        clauses.insert(0, '{SELECT ?sk0')
        clauses = arg_clauses + clauses
        clauses.insert(0, 'WHERE {')
        clauses.insert(0, f'SELECT DISTINCT ?x')
    else:
        clauses.insert(0, f'SELECT DISTINCT ?x')
    clauses.insert(0, 'PREFIX ns: <http://rdf.freebase.com/ns/>')
    clauses.append('}')
    clauses.extend(order_clauses)
    if superlative:
        clauses.append('}')
        clauses.append('}')
    return '\n'.join(clauses)

def retrieve_relations(exp: list):
    rtn = []
    for element in exp:
        if element == 'JOIN':
            continue
        elif isinstance(element, str):
            rtn.append(element)
        elif isinstance(element, list) and element[0] == 'R':
            rtn.append(element)
        elif isinstance(element, list) and element[0] == 'JOIN':
            rtn.extend(retrieve_relations(element))
    return rtn

def get_root(var: int):
    while var in identical_variables_r:
        var = identical_variables_r[var]
    return var

