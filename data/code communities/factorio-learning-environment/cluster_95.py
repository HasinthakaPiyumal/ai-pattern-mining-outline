# Cluster 95

class TestProductionDivergence(unittest.TestCase):

    def test_achievements(self):
        instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
        instance.set_speed(10)
        _, _, _, achievements = eval_program_with_achievements(instance, test_string_1)
        ground_truth_achievement = {'static': {'stone-furnace': 1, 'coal': 10, 'stone': 10, 'iron-ore': 10}, 'dynamic': {'iron-plate': 5}}
        assert achievements == ground_truth_achievement
        _, _, _, achievements = eval_program_with_achievements(instance, test_string)
        ground_truth_achievement = {'static': {'stone-furnace': 1, 'coal': 10, 'stone': 10, 'copper-ore': 10}, 'dynamic': {'copper-plate': 5}}
        assert achievements == ground_truth_achievement

def eval_program_with_achievements(instance, program):
    pre_production_flows = instance.get_production_stats()
    try:
        score, goal, result = instance.eval_with_error(program, timeout=300)
        error = False
    except Exception as e:
        result = e
        result = str(e)
        error = True
    output_list = result.splitlines()
    post_production_flows = instance.get_production_stats()
    achievements = get_achievements(pre_production_flows, post_production_flows)
    return (output_list, result, error, achievements)

