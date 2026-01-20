# Cluster 3

def test_normalize_color_size():
    product_prices = {(1, 'black forest', '3 meter'): 10.29, (2, 'violet lavender', 'xx-large'): 23.42, (3, 'steelivy fuchsia', 'random value'): 193.87, (4, '123alabaster', '40cm plus'): 67.23, (5, 'webshop', '142'): 1.02, (6, 'webshopsteel', '2 petite'): 57.99, (7, 'leather black', '91ft walnut feet'): 6.2}
    color_mapping_expected = {'N.A.': 'not_matched', 'black forest': 'black', 'violet lavender': 'lavender', 'steelivy fuchsia': 'fuchsia', '123alabaster': 'alabaster', 'webshop': 'not_matched', 'webshopsteel': 'steel', 'leather black': 'black'}
    size_mapping_expected = {'N.A.': 'not_matched', '3 meter': '(.*)meter', 'xx-large': 'xx-large', 'random value': 'not_matched', '40cm plus': '(.*)plus', '142': 'numeric_size', '2 petite': '(.*)petite', '91ft walnut feet': '(.*)ft'}
    color_mapping, size_mapping = normalize_color_size(product_prices)
    assert type(color_mapping) == dict
    assert type(size_mapping) == dict
    assert color_mapping == color_mapping_expected
    assert size_mapping == size_mapping_expected

def normalize_color_size(product_prices: dict) -> Tuple[dict, dict]:
    """Get mappings of all colors, sizes to corresponding values in COLOR_SET, SIZE_PATTERNS"""
    all_colors, all_sizes = (set(), set())
    for (_, color, size), _ in product_prices.items():
        all_colors.add(color.lower())
        all_sizes.add(size.lower())
    color_mapping = {'N.A.': 'not_matched'}
    for c in all_colors:
        matched = False
        for base in COLOR_SET:
            if base in c:
                color_mapping[c] = base
                matched = True
                break
        if not matched:
            color_mapping[c] = 'not_matched'
    size_mapping = {'N.A.': 'not_matched'}
    for s in all_sizes:
        matched = False
        for pattern in SIZE_PATTERNS:
            m = re.search(pattern, s)
            if m is not None:
                matched = True
                size_mapping[s] = pattern.pattern
                break
        if not matched:
            if s.replace('.', '', 1).isdigit():
                size_mapping[s] = 'numeric_size'
                matched = True
        if not matched:
            size_mapping[s] = 'not_matched'
    return (color_mapping, size_mapping)

