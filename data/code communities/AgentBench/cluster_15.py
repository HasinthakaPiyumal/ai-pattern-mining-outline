# Cluster 15

def _anonymize_entities(expression: list):
    if isinstance(expression, list):
        for i in range(len(expression)):
            if isinstance(expression[i], str):
                if expression[i].__contains__('^^') or expression[i][:2] in ['m.', 'g.']:
                    expression[i] = '[ENT]'
            else:
                _anonymize_entities(expression[i])
    return expression

