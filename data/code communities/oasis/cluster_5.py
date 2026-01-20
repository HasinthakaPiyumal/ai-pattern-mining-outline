# Cluster 5

def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    if parsed is None:
        parsed = {root}
    else:
        parsed.add(root)
    neighbors = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        neighbors.remove(parent)
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos = _hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    return pos

