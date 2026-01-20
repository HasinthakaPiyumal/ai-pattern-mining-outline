# Cluster 9

def set_visited(G, s, e, relation):
    end_num = get_end_num(G, s)
    for i in range(0, end_num[e]):
        if G.edges[s, e, i]['relation'] == relation:
            G.edges[s, e, i]['visited'] = True

def get_end_num(G, s):
    end_num = defaultdict(lambda: 0)
    for edge in list(G.edges(s)):
        end_num[list(edge)[1]] += 1
    return end_num

def none_function(G, start, arg_node=None, type_constraint=True):
    if arg_node is not None:
        arg = G.nodes[arg_node]['function']
        path = list(nx.all_simple_paths(G, start, arg_node))
        assert len(path) == 1
        arg_clause = []
        for i in range(0, len(path[0]) - 1):
            edge = G.edges[path[0][i], path[0][i + 1], 0]
            if edge['reverse']:
                relation = '(R ' + edge['relation'] + ')'
            else:
                relation = edge['relation']
            arg_clause.append(relation)
        while i >= 0:
            flag = False
            if G.out_degree[path[0][i]] > 2:
                flag = True
            G.remove_edge(path[0][i], path[0][i + 1], 0)
            i -= 1
            if flag:
                break
        if len(arg_clause) > 1:
            arg_clause = binary_nesting(function='JOIN', elements=arg_clause)
        else:
            arg_clause = arg_clause[0]
        return '(' + arg.upper() + ' ' + none_function(G, start) + ' ' + arg_clause + ')'
    if G.nodes[start]['type'] != 'class':
        return G.nodes[start]['id']
    end_num = get_end_num(G, start)
    clauses = []
    if G.nodes[start]['question'] and type_constraint:
        clauses.append(G.nodes[start]['id'])
    for key in end_num.keys():
        for i in range(0, end_num[key]):
            if not G.edges[start, key, i]['visited']:
                relation = G.edges[start, key, i]['relation']
                G.edges[start, key, i]['visited'] = True
                set_visited(G, key, start, relation)
                if G.edges[start, key, i]['reverse']:
                    relation = '(R ' + relation + ')'
                if G.nodes[key]['function'].__contains__('<') or G.nodes[key]['function'].__contains__('>'):
                    if G.nodes[key]['function'] == '>':
                        clauses.append('(gt ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '>=':
                        clauses.append('(ge ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '<':
                        clauses.append('(lt ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '<=':
                        clauses.append('(le ' + relation + ' ' + none_function(G, key) + ')')
                else:
                    clauses.append('(JOIN ' + relation + ' ' + none_function(G, key) + ')')
    if len(clauses) == 0:
        return G.nodes[start]['id']
    if len(clauses) == 1:
        return clauses[0]
    else:
        return binary_nesting(function='AND', elements=clauses)

def get_lisp_from_graph_query(graph_query):
    G = nx.MultiDiGraph()
    aggregation = 'none'
    arg_node = None
    for node in graph_query['nodes']:
        G.add_node(node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'], function=node['function'], cla=node['class'])
        if node['question_node'] == 1:
            qid = node['nid']
        if node['function'] != 'none':
            aggregation = node['function']
            if node['function'].__contains__('arg'):
                arg_node = node['nid']
    for edge in graph_query['edges']:
        G.add_edge(edge['start'], edge['end'], relation=edge['relation'], reverse=False, visited=False)
        G.add_edge(edge['end'], edge['start'], relation=edge['relation'], reverse=True, visited=False)
    if 'count' == aggregation:
        return count_function(G, qid)
    else:
        return none_function(G, qid, arg_node=arg_node)

def count_function(G, start):
    return '(COUNT ' + none_function(G, start) + ')'

