# Cluster 1

def write_route_file(file, end, step):
    with open(file, 'w+') as f:
        f.write('<routes>\n                <route id="route_ns" edges="n_t t_s"/>\n                <route id="route_nw" edges="n_t t_w"/>\n                <route id="route_ne" edges="n_t t_e"/>\n                <route id="route_we" edges="w_t t_e"/>\n                <route id="route_wn" edges="w_t t_n"/>\n                <route id="route_ws" edges="w_t t_s"/>\n                <route id="route_ew" edges="e_t t_w"/>\n                <route id="route_en" edges="e_t t_n"/>\n                <route id="route_es" edges="e_t t_s"/>\n                <route id="route_sn" edges="s_t t_n"/>\n                <route id="route_se" edges="s_t t_e"/>\n                <route id="route_sw" edges="s_t t_w"/>')
        c = 0
        for i in range(0, end, step):
            f.write(get_context(i, i + step, c))
            c += 1
        f.write('</routes>')

def get_context(begin, end, c):
    if c % 2 == 0:
        s = v
    else:
        s = h
    s = s.replace('c', str(c)).replace('bb', str(begin)).replace('ee', str(end))
    return s

