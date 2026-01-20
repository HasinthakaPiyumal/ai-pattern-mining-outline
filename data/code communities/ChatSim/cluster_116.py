# Cluster 116

def zero_corrected_countless(data):
    """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
    data, upgraded = upgrade_type(data)
    data += 1
    sections = []
    factor = (2, 2)
    for offset in np.ndindex(factor):
        part = data[tuple((np.s_[o::f] for o, f in zip(offset, factor)))]
        sections.append(part)
    a, b, c, d = sections
    ab = a * (a == b)
    ac = a * (a == c)
    bc = b * (b == c)
    a = ab | ac | bc
    result = a + (a == 0) * d - 1
    if upgraded:
        return downgrade_type(result)
    data -= 1
    return result

def upgrade_type(arr):
    dtype = arr.dtype
    if dtype == np.uint8:
        return (arr.astype(np.uint16), True)
    elif dtype == np.uint16:
        return (arr.astype(np.uint32), True)
    elif dtype == np.uint32:
        return (arr.astype(np.uint64), True)
    return (arr, False)

def downgrade_type(arr):
    dtype = arr.dtype
    if dtype == np.uint64:
        return arr.astype(np.uint32)
    elif dtype == np.uint32:
        return arr.astype(np.uint16)
    elif dtype == np.uint16:
        return arr.astype(np.uint8)
    return arr

def countless(data):
    """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
    data, upgraded = upgrade_type(data)
    data += 1
    sections = []
    factor = (2, 2)
    for offset in np.ndindex(factor):
        part = data[tuple((np.s_[o::f] for o, f in zip(offset, factor)))]
        sections.append(part)
    a, b, c, d = sections
    ab_ac = a * ((a == b) | (a == c))
    ab_ac |= b * (b == c)
    result = ab_ac + (ab_ac == 0) * d - 1
    if upgraded:
        return downgrade_type(result)
    data -= 1
    return result

