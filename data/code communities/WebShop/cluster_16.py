# Cluster 16

def get_option_reward(purchased_options, goal_options):
    """Calculate reward for purchased product's options w.r.t. goal options"""
    purchased_options = [normalize_color(o) for o in purchased_options]
    goal_options = [normalize_color(o) for o in goal_options]
    num_option_matches = 0
    for g_option in goal_options:
        for p_option in purchased_options:
            score = fuzz.token_set_ratio(p_option, g_option)
            if score > 85:
                num_option_matches += 1
                break
    r_option = num_option_matches / len(goal_options) if len(goal_options) > 0 else None
    return (r_option, num_option_matches)

def normalize_color(color_string: str) -> str:
    """Extracts the first color found if exists"""
    for norm_color in COLOR_SET:
        if norm_color in color_string:
            return norm_color
    return color_string

def test_normalize_color():
    suite = [('', ''), ('black forest', 'black'), ('violet lavender', 'lavender'), ('steelivy fuchsia', 'fuchsia'), ('123alabaster', 'alabaster'), ('webshop', 'webshop')]
    for color_string, expected in suite:
        output = normalize_color(color_string)
        assert type(output) is str
        assert output == expected

def test_get_option_reward():
    goal = ['grey', 'XL', 'pack of 12']
    purchased = ['pack of 12', 'grey', 'XL']
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == len(goal)
    assert r_option == 1
    goal = ['grey', 'XL', 'pack of 12']
    purchased = ['pack of 12', 'blue', 'XL']
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == len(goal) - 1
    assert r_option == 2.0 / 3.0
    goal = ['cool powder snow', 'XL', 'pack of 12']
    purchased = ['pack of 12', 'powder snow', 'XL']
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == len(goal)
    assert r_option == 1
    goal = []
    purchased = ['goal 1', 'goal 2']
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == 0
    assert r_option == None
    goal = ['goal 1', 'goal 2']
    purchased = []
    r_option, matches = get_option_reward(purchased, goal)
    assert matches == 0
    assert r_option == 0

