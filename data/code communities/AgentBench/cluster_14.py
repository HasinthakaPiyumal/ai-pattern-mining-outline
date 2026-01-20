# Cluster 14

def _linearize_lisp_expression(expression: list, sub_formula_id):
    sub_formulas = []
    for i, e in enumerate(expression):
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
            expression[i] = '#' + str(sub_formula_id[0] - 1)
    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    return sub_formulas

