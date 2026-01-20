# Cluster 0

def _filter_out(flow_exec_list, name):
    return list(filter(lambda exc: exc['name'] != name, flow_exec_list))

