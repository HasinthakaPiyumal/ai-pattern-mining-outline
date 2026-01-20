# Cluster 87

def test_long_handed_inserter(game):
    """Test long-handed inserter's ability to move items between chests"""
    input_chest, inserter, output_chest = setup_power_and_chests(game, Prototype.LongHandedInserter)
    game.insert_item(Prototype.IronPlate, input_chest, quantity=50)
    game.sleep(20)
    output_inventory = game.inspect_inventory(output_chest)
    assert output_inventory.get(Prototype.IronPlate, 0) > 0, 'Long-handed inserter failed to move items'

def setup_power_and_chests(game, inserter_type, origin_pos=Position(x=0, y=0)):
    """Helper function to set up power and chest configuration for inserter tests"""
    solar_panel = game.place_entity(Prototype.SolarPanel, position=origin_pos)
    pole = game.place_entity_next_to(Prototype.SmallElectricPole, solar_panel.position, Direction.RIGHT)
    input_chest = game.place_entity_next_to(Prototype.SteelChest, pole.position, Direction.DOWN)
    if inserter_type != Prototype.LongHandedInserter:
        inserter = game.place_entity_next_to(inserter_type, input_chest.position, Direction.RIGHT, spacing=0)
        output_chest = game.place_entity_next_to(Prototype.SteelChest, inserter.position, Direction.RIGHT)
    else:
        inserter = game.place_entity_next_to(inserter_type, input_chest.position, Direction.RIGHT, spacing=1)
        output_chest = game.place_entity_next_to(Prototype.SteelChest, inserter.position, Direction.RIGHT, spacing=1)
    game.connect_entities(pole, inserter, Prototype.SmallElectricPole)
    return (input_chest, inserter, output_chest)

def test_filter_inserter(game):
    """Test filter inserter's ability to selectively move items"""
    input_chest, inserter, output_chest = setup_power_and_chests(game, Prototype.FilterInserter)
    game.set_entity_recipe(inserter, Prototype.IronPlate)
    game.insert_item(Prototype.IronPlate, input_chest, quantity=50)
    game.insert_item(Prototype.CopperPlate, input_chest, quantity=50)
    game.sleep(20)
    output_inventory = game.inspect_inventory(output_chest)
    assert output_inventory.get(Prototype.IronPlate, 0) > 0, 'Filter inserter failed to move iron plates'
    assert output_inventory.get(Prototype.CopperPlate, 0) == 0, 'Filter inserter incorrectly moved copper plates'

def test_stack_inserter(game):
    """Test stack inserter's ability to move multiple items at once"""
    input_chest, inserter, output_chest = setup_power_and_chests(game, Prototype.StackInserter)
    game.insert_item(Prototype.ElectronicCircuit, input_chest, quantity=100)
    first_transfer = game.inspect_inventory(output_chest).get(Prototype.ElectronicCircuit, 0)
    game.sleep(5)
    second_transfer = game.inspect_inventory(output_chest).get(Prototype.ElectronicCircuit, 0)
    assert second_transfer > first_transfer, 'Stack inserter failed to move multiple items at once'
    assert second_transfer >= 12, 'Stack inserter not moving expected quantity of items'

def test_filter_stack_inserter(game):
    """Test stack inserter's ability to move multiple items at once"""
    input_chest, inserter, output_chest = setup_power_and_chests(game, Prototype.StackFilterInserter)
    game.set_entity_recipe(inserter, Prototype.ElectronicCircuit)
    game.insert_item(Prototype.ElectronicCircuit, input_chest, quantity=100)
    first_transfer = game.inspect_inventory(output_chest).get(Prototype.ElectronicCircuit, 0)
    game.sleep(5)
    second_transfer = game.inspect_inventory(output_chest).get(Prototype.ElectronicCircuit, 0)
    assert second_transfer > first_transfer, 'Stack inserter failed to move multiple items at once'
    assert second_transfer >= 12, 'Stack inserter not moving expected quantity of items'

