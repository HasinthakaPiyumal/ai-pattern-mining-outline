# Cluster 18

def process_inv_function(expression: List):
    for i, item in enumerate(expression):
        if isinstance(item, list):
            if item[0] == 'R':
                expression[i] = item[1] + '_inv'
            else:
                process_inv_function(item)

def preprocess_relation_path_for_superlatives(expression):
    relations = []
    for element in expression:
        if element == 'JOIN':
            continue
        if isinstance(element, list) and element[0] != 'R':
            assert element[0] == 'JOIN'
            relations.extend(preprocess_relation_path_for_superlatives(element))
            continue
        relations.append(element)
    return relations

def linearize_lisp_expression_for_bottom_up(expression: list, sub_formula_id):
    sub_formulas = []
    level = {}
    max_sub_level = -1
    for i, e in enumerate(expression):
        if isinstance(e, list):
            sf, lvl = linearize_lisp_expression_for_bottom_up(e, sub_formula_id)
            sub_formulas.extend(sf)
            level.update(lvl)
            expression[i] = '#' + str(sub_formula_id[0] - 1)
            if lvl[sub_formula_id[0] - 1] > max_sub_level:
                max_sub_level = lvl[sub_formula_id[0] - 1]
    current_level = max_sub_level + 1
    sub_formulas.append(expression)
    level[sub_formula_id[0]] = current_level
    sub_formula_id[0] += 1
    return (sub_formulas, level)

def get_sub_programs(formula: str):
    expression = lisp_to_nested_expression(formula)
    process_inv_function(expression)
    if expression[0] in ['ARGMIN', 'ARGMAX']:
        if isinstance(expression[2], list) and expression[2][0] == 'JOIN':
            arg_path = preprocess_relation_path_for_superlatives(expression[2])
            expression = expression[:2]
            expression.extend(arg_path)
    sub_formulas, level_mapping = linearize_lisp_expression_for_bottom_up(expression, [0])
    if sub_formulas[-1][0] in ['ARGMAX', 'ARGMIN'] and len(sub_formulas[-1]) > 3:
        last_id = len(level_mapping) - 1
        last_level = level_mapping[last_id]
        new_sub_formulas = sub_formulas[:-1]
        for i in range(len(sub_formulas[-1]) - 2):
            new_sub_formulas.append(sub_formulas[-1][:3 + i])
            level_mapping[last_id] = last_level
            last_id += 1
            last_level += 1
        sub_formulas = new_sub_formulas
    new_level_mapping = defaultdict(lambda: [])
    for k, v in level_mapping.items():
        new_level_mapping[v].append(k)
    return (sub_formulas, new_level_mapping)

