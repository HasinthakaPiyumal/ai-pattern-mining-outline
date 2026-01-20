# Cluster 8

def _get_graph(expression: List) -> nx.MultiGraph:
    if isinstance(expression, str):
        G = nx.MultiDiGraph()
        if get_symbol_type(expression) == 1:
            G.add_node(1, id=expression, type='entity')
        elif get_symbol_type(expression) == 2:
            G.add_node(1, id=expression, type='literal')
        elif get_symbol_type(expression) == 3:
            G.add_node(1, id=expression, type='class')
        elif get_symbol_type(expression) == 4:
            domain, rang = relation_dr[expression]
            G.add_node(1, id=rang, type='class')
            G.add_node(2, id=domain, type='class')
            G.add_edge(2, 1, relation=expression)
            if REVERSE:
                if expression in reverse_properties:
                    G.add_edge(1, 2, relation=reverse_properties[expression])
        return G
    if expression[0] == 'R':
        if get_symbol_type(expression[1]) != 4:
            pass
        G = _get_graph(expression[1])
        size = len(G.nodes())
        mapping = {}
        for n in G.nodes():
            mapping[n] = size - n + 1
        G = nx.relabel_nodes(G, mapping)
        return G
    elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
        if isinstance(expression[1], str) and get_symbol_type(expression[1]) != 4 or (not isinstance(expression[2], list) and get_symbol_type(expression[2]) not in [1, 2]) or (isinstance(expression[1], list) and expression[1][0] != 'R'):
            pass
        G1 = _get_graph(expression=expression[1])
        G2 = _get_graph(expression=expression[2])
        size = len(G2.nodes())
        qn_id = size
        if G1.nodes[1]['type'] == G2.nodes[qn_id]['type'] == 'class':
            if G2.nodes[qn_id]['id'] in upper_types[G1.nodes[1]['id']]:
                G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
        if G1.nodes[1]['type'] == 'entity':
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size
            G1 = nx.relabel_nodes(G1, mapping)
        else:
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size - 1
            G1 = nx.relabel_nodes(G1, mapping)
        G = nx.compose(G1, G2)
        if expression[0] != 'JOIN':
            G.nodes[1]['function'] = function_map[expression[0]]
        return G
    elif expression[0] == 'AND':
        if not isinstance(expression[1], list) and get_symbol_type(expression[1]) != 3 or not isinstance(expression[2], list):
            pass
        G1 = _get_graph(expression[1])
        G2 = _get_graph(expression[2])
        size1 = len(G1.nodes())
        size2 = len(G2.nodes())
        if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
            G2.nodes[size2]['id'] = G1.nodes[size1]['id']
        if G1.nodes[1]['type'] == 'entity':
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2
            G1 = nx.relabel_nodes(G1, mapping)
        else:
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)
        return G
    elif expression[0] == 'COUNT':
        if len(expression) != 2 or not isinstance(expression[1], list):
            pass
        G = _get_graph(expression[1])
        size = len(G.nodes())
        G.nodes[size]['function'] = 'count'
        return G
    elif expression[0].__contains__('ARG'):
        if not isinstance(expression[1], list) and get_symbol_type(expression[1]) != 3 or (not isinstance(expression[2], list) and get_symbol_type(expression[2]) != 4):
            pass
        G1 = _get_graph(expression[1])
        size1 = len(G1.nodes())
        G2 = _get_graph(expression[2])
        size2 = len(G2.nodes())
        G2.nodes[1]['id'] = 0
        G2.nodes[1]['type'] = 'literal'
        G2.nodes[1]['function'] = expression[0].lower()
        if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
            G2.nodes[size2]['id'] = G1.nodes[size1]['id']
        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size2 - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)
        return G
    elif expression[0] == 'TC':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        G.nodes[size]['tc'] = (expression[2], expression[3])
        return G

def get_symbol_type(symbol: str) -> int:
    if symbol.__contains__('^^'):
        return 2
    elif symbol in types:
        return 3
    elif symbol in relations:
        return 4
    elif symbol:
        return 1

def count_relations_expression(expression: List):
    rtn = defaultdict(lambda: 0)
    for item in expression:
        if isinstance(item, str):
            if get_symbol_type(item) == 4:
                rtn[item] += 1
                if item in reverse_properties:
                    rtn[reverse_properties[item]] += 1
        else:
            item_rtn = count_relations_expression(item)
            for r in item_rtn:
                if r in rtn:
                    rtn[r] = rtn[r] + item_rtn[r]
                else:
                    rtn[r] = item_rtn[r]
    return rtn

