# Cluster 90

def test_fixed_combination_inspector(test_fixed_combination_data: pd.DataFrame):
    inspector = FixedCombinationInspector()
    inspector.fit(test_fixed_combination_data)
    assert inspector.ready
    assert inspector.fixed_combinations
    expected_combinations = {'A': {'categorical_3', 'D', 'E', 'B'}, 'B': {'categorical_3', 'D', 'E', 'A'}, 'D': {'categorical_3', 'E', 'A', 'B'}, 'E': {'categorical_3', 'D', 'A', 'B'}, 'categorical_3': {'categorical_4', 'D', 'E', 'A', 'B'}, 'categorical_1': {'categorical_2'}, 'categorical_5': {'categorical_6'}}
    assert inspector.fixed_combinations == expected_combinations
    assert inspector.inspect_level == 70

