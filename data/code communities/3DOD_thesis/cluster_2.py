# Cluster 2

def vals_to_dict(vals, keys, vals_n=0):
    out = dict()
    for key in keys:
        if isinstance(key, str):
            try:
                val = float(vals[vals_n])
            except:
                val = vals[vals_n]
            data = val
            key_name = key
            vals_n += 1
        else:
            data, vals_n = vals_to_dict(vals, key[1], vals_n)
            key_name = key[0]
        out[key_name] = data
        if vals_n >= len(vals):
            break
    return (out, vals_n)

