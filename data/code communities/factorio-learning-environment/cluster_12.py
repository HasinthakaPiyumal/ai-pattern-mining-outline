# Cluster 12

def lua_table_to_dict(lua_table):
    if not isinstance(lua_table, (dict, list)):
        return lua_table
    if isinstance(lua_table, list):
        return [lua_table_to_dict(v) for v in lua_table]
    return {k: lua_table_to_dict(v) for k, v in lua_table.items()}

