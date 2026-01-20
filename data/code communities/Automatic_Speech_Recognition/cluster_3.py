# Cluster 3

def list_to_sparse_tensor(targetList, mode='train'):
    """ turn 2-D List to SparseTensor
    """
    indices = []
    vals = []
    group_phn = group_phoneme(phn, mapping)
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            if mode == 'train':
                indices.append([tI, seqI])
                vals.append(val)
            elif mode == 'test':
                if phn[val] in mapping.keys():
                    val = group_phn.index(mapping[phn[val]])
                indices.append([tI, seqI])
                vals.append(val)
            else:
                raise ValueError('Invalid mode.', mode)
    shape = [len(targetList), np.asarray(indices).max(0)[1] + 1]
    return (np.array(indices), np.array(vals), np.array(shape))

def group_phoneme(orig_phn, mapping):
    group_phn = []
    for val in orig_phn:
        group_phn.append(val)
    group_phn.append('sil')
    for key in mapping.keys():
        if key in orig_phn:
            group_phn.remove(key)
    group_phn.sort()
    return group_phn

