# Cluster 17

def replace_grid_symbols(grid, old_to_new_map):
    """Replaces symbols in the grid.

    If mapping is not defined the symbol is not updated.

    Args:
      grid: Represented as a list of strings.
      old_to_new_map: Mapping between symbols.

    Returns:
      Updated grid.
    """

    def symbol_map(x):
        if x in old_to_new_map:
            return old_to_new_map[x]
        return x
    new_grid = []
    for row in grid:
        new_grid.append(''.join((symbol_map(i) for i in row)))
    return new_grid

def symbol_map(x):
    if x in old_to_new_map:
        return old_to_new_map[x]
    return x

