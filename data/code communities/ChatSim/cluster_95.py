# Cluster 95

def countless_generalized(data, factor):
    assert len(data.shape) == len(factor)
    sections = []
    mode_of = reduce(lambda x, y: x * y, factor)
    majority = int(math.ceil(float(mode_of) / 2))
    data += 1
    for offset in np.ndindex(factor):
        part = data[tuple((np.s_[o::f] for o, f in zip(offset, factor)))]
        sections.append(part)

    def pick(elements):
        eq = (elements[i] == elements[i + 1] for i in range(len(elements) - 1))
        anded = reduce(lambda p, q: p & q, eq)
        return elements[0] * anded

    def logical_or(x, y):
        return x + (x == 0) * y
    result = (pick(combo) for combo in combinations(sections, majority))
    result = reduce(logical_or, result)
    for i in range(majority - 1, 3 - 1, -1):
        partial_result = (pick(combo) for combo in combinations(sections, i))
        partial_result = reduce(logical_or, partial_result)
        result = logical_or(result, partial_result)
    partial_result = (pick(combo) for combo in combinations(sections[:-1], 2))
    partial_result = reduce(logical_or, partial_result)
    result = logical_or(result, partial_result)
    result = logical_or(result, sections[-1]) - 1
    data -= 1
    return result

def pick(elements):
    eq = (elements[i] == elements[i + 1] for i in range(len(elements) - 1))
    anded = reduce(lambda p, q: p & q, eq)
    return elements[0] * anded

def logical_or(x, y):
    return x + (x == 0) * y

def dynamic_countless_generalized(data, factor):
    assert len(data.shape) == len(factor)
    sections = []
    mode_of = reduce(lambda x, y: x * y, factor)
    majority = int(math.ceil(float(mode_of) / 2))
    data += 1
    for offset in np.ndindex(factor):
        part = data[tuple((np.s_[o::f] for o, f in zip(offset, factor)))]
        sections.append(part)
    pick = lambda a, b: a * (a == b)
    lor = lambda x, y: x + (x == 0) * y
    subproblems = [{}, {}]
    results2 = None
    for x, y in combinations(range(len(sections) - 1), 2):
        res = pick(sections[x], sections[y])
        subproblems[0][x, y] = res
        if results2 is not None:
            results2 = lor(results2, res)
        else:
            results2 = res
    results = [results2]
    for r in range(3, majority + 1):
        r_results = None
        for combo in combinations(range(len(sections)), r):
            res = pick(subproblems[0][combo[:-1]], sections[combo[-1]])
            if combo[-1] != len(sections) - 1:
                subproblems[1][combo] = res
            if r_results is not None:
                r_results = lor(r_results, res)
            else:
                r_results = res
        results.append(r_results)
        subproblems[0] = subproblems[1]
        subproblems[1] = {}
    results.reverse()
    final_result = lor(reduce(lor, results), sections[-1]) - 1
    data -= 1
    return final_result

def countless3d_generalized(img):
    return countless_generalized(img, (2, 8, 1))

def countless3d_dynamic_generalized(img):
    return dynamic_countless_generalized(img, (8, 8, 1))

def dynamic_countless3d(data):
    """countless8 + dynamic programming. ~2x faster"""
    sections = []
    data += 1
    factor = (2, 2, 2)
    for offset in np.ndindex(factor):
        part = data[tuple((np.s_[o::f] for o, f in zip(offset, factor)))]
        sections.append(part)
    pick = lambda a, b: a * (a == b)
    lor = lambda x, y: x + (x == 0) * y
    subproblems2 = {}
    results2 = None
    for x, y in combinations(range(7), 2):
        res = pick(sections[x], sections[y])
        subproblems2[x, y] = res
        if results2 is not None:
            results2 += (results2 == 0) * res
        else:
            results2 = res
    subproblems3 = {}
    results3 = None
    for x, y, z in combinations(range(8), 3):
        res = pick(subproblems2[x, y], sections[z])
        if z != 7:
            subproblems3[x, y, z] = res
        if results3 is not None:
            results3 += (results3 == 0) * res
        else:
            results3 = res
    results3 = reduce(lor, (results3, results2, sections[-1]))
    results2 = None
    subproblems2 = None
    res = None
    results4 = (pick(subproblems3[x, y, z], sections[w]) for x, y, z, w in combinations(range(8), 4))
    results4 = reduce(lor, results4)
    subproblems3 = None
    final_result = lor(results4, results3) - 1
    data -= 1
    return final_result

def countless_generalized(data, factor):
    assert len(data.shape) == len(factor)
    sections = []
    mode_of = reduce(lambda x, y: x * y, factor)
    majority = int(math.ceil(float(mode_of) / 2))
    data += 1
    for offset in np.ndindex(factor):
        part = data[tuple((np.s_[o::f] for o, f in zip(offset, factor)))]
        sections.append(part)

    def pick(elements):
        eq = (elements[i] == elements[i + 1] for i in range(len(elements) - 1))
        anded = reduce(lambda p, q: p & q, eq)
        return elements[0] * anded

    def logical_or(x, y):
        return x + (x == 0) * y
    result = (pick(combo) for combo in combinations(sections, majority))
    result = reduce(logical_or, result)
    for i in range(majority - 1, 3 - 1, -1):
        partial_result = (pick(combo) for combo in combinations(sections, i))
        partial_result = reduce(logical_or, partial_result)
        result = logical_or(result, partial_result)
    partial_result = (pick(combo) for combo in combinations(sections[:-1], 2))
    partial_result = reduce(logical_or, partial_result)
    result = logical_or(result, partial_result)
    result = logical_or(result, sections[-1]) - 1
    data -= 1
    return result

def dynamic_countless_generalized(data, factor):
    assert len(data.shape) == len(factor)
    sections = []
    mode_of = reduce(lambda x, y: x * y, factor)
    majority = int(math.ceil(float(mode_of) / 2))
    data += 1
    for offset in np.ndindex(factor):
        part = data[tuple((np.s_[o::f] for o, f in zip(offset, factor)))]
        sections.append(part)
    pick = lambda a, b: a * (a == b)
    lor = lambda x, y: x + (x == 0) * y
    subproblems = [{}, {}]
    results2 = None
    for x, y in combinations(range(len(sections) - 1), 2):
        res = pick(sections[x], sections[y])
        subproblems[0][x, y] = res
        if results2 is not None:
            results2 = lor(results2, res)
        else:
            results2 = res
    results = [results2]
    for r in range(3, majority + 1):
        r_results = None
        for combo in combinations(range(len(sections)), r):
            res = pick(subproblems[0][combo[:-1]], sections[combo[-1]])
            if combo[-1] != len(sections) - 1:
                subproblems[1][combo] = res
            if r_results is not None:
                r_results = lor(r_results, res)
            else:
                r_results = res
        results.append(r_results)
        subproblems[0] = subproblems[1]
        subproblems[1] = {}
    results.reverse()
    final_result = lor(reduce(lor, results), sections[-1]) - 1
    data -= 1
    return final_result

def countless3d_generalized(img):
    return countless_generalized(img, (2, 8, 1))

def countless3d_dynamic_generalized(img):
    return dynamic_countless_generalized(img, (8, 8, 1))

