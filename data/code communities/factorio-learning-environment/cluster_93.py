# Cluster 93

def test_profits():
    instance = FactorioInstance(address='localhost', bounding_box=200, tcp_port=27000, fast=True, inventory={})
    instance.set_speed(10)
    profit_config = {'max_static_unit_profit_cap': 5, 'dynamic_profit_multiplier': 10}
    test_string_1 = 'pos = nearest(Resource.Stone)\nmove_to(pos)\nharvest_resource(pos, 10)\ncraft_item(Prototype.StoneFurnace, 2)\npos = nearest(Resource.Coal)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = nearest(Resource.IronOre)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = Position(x = 0, y = 0)\nmove_to(pos)\nfurnace = place_entity(Prototype.StoneFurnace, position = pos)\ninsert_item(Prototype.IronOre, furnace, 5)\ninsert_item(Prototype.Coal, furnace, 5)\nsleep(25)\nextract_item(Prototype.IronPlate, furnace.position, 10)'
    _, _, _, profits = eval_program_with_profits(instance, test_string_1, profit_config)
    assert math.isclose(profits['static'], 2.8, rel_tol=1)
    assert math.isclose(profits['dynamic'], 96, rel_tol=1)
    assert math.isclose(profits['total'], 98.8, rel_tol=1)
    test_string = 'pos = nearest(Resource.Stone)\nmove_to(pos)\nharvest_resource(pos, 10)\ncraft_item(Prototype.StoneFurnace, 1)\npos = nearest(Resource.Coal)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = nearest(Resource.CopperOre)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = Position(x = 0, y = 0)\nmove_to(pos)\nfurnace = place_entity(Prototype.StoneFurnace, position = pos)\ninsert_item(Prototype.CopperOre, furnace, 5)\ninsert_item(Prototype.Coal, furnace, 5)\nsleep(25)'
    _, _, _, profits = eval_program_with_profits(instance, test_string, profit_config)
    assert math.isclose(profits['static'], 1.4, rel_tol=1)
    assert math.isclose(profits['dynamic'], 136, rel_tol=1)
    assert math.isclose(profits['total'], 137.4, rel_tol=1)
    test_string = 'pos = nearest(Resource.Stone)\nmove_to(pos)\nharvest_resource(pos, 10)\ncraft_item(Prototype.StoneFurnace, 1)\npos = nearest(Resource.Coal)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = nearest(Resource.CopperOre)\nmove_to(pos)\nharvest_resource(pos, 10)\npos = Position(x = 0, y = 0)\nmove_to(pos)\nfurnace = place_entity(Prototype.StoneFurnace, position = pos)\ninsert_item(Prototype.CopperOre, furnace, 5)\ninsert_item(Prototype.Coal, furnace, 5)\nsleep(25)'
    _, _, _, profits = eval_program_with_profits(instance, test_string, profit_config)
    assert math.isclose(profits['static'], 1.4, rel_tol=1)
    assert math.isclose(profits['dynamic'], 106, rel_tol=1)
    assert math.isclose(profits['total'], 107.4, rel_tol=1)

def eval_program_with_profits(instance, program, profit_config):
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
    profits = get_profits(pre_production_flows, post_production_flows, profit_config=profit_config)
    return (output_list, result, error, profits)

