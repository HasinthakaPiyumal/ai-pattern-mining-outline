# Cluster 93

def countless_extreme(data):
    nonzeros = np.count_nonzero(data)
    N = reduce(operator.mul, data.shape)
    if nonzeros == N:
        print('quick')
        return quick_countless(data)
    elif np.count_nonzero(data + 1) == N:
        print('quick')
        return quick_countless(data)
    else:
        return countless(data)

def quick_countless(data):
    """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
    sections = []
    factor = (2, 2)
    for offset in np.ndindex(factor):
        part = data[tuple((np.s_[o::f] for o, f in zip(offset, factor)))]
        sections.append(part)
    a, b, c, d = sections
    ab_ac = a * ((a == b) | (a == c))
    bc = b * (b == c)
    a = ab_ac | bc
    return a + (a == 0) * d

def countless_extreme(data):
    nonzeros = np.count_nonzero(data)
    N = reduce(operator.mul, data.shape)
    if nonzeros == N:
        print('quick')
        return quick_countless(data)
    elif np.count_nonzero(data + 1) == N:
        print('quick')
        return quick_countless(data)
    else:
        return countless(data)

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

