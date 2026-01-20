# Cluster 1

@partial(jax.pmap, axis_name='batch')
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

