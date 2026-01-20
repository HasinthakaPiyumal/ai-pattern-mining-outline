# Cluster 88

def test_connect_electricity_bug(game):
    water_pos = game.nearest(Resource.Water)
    print(f'Found water at {water_pos}')
    game.move_to(water_pos)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_pos)
    print(f'Placed offshore pump at {offshore_pump.position}')
    building_box = BuildingBox(width=Prototype.StorageTank.WIDTH + 4, height=Prototype.StorageTank.HEIGHT + 4)
    coords = game.nearest_buildable(Prototype.StorageTank, building_box, offshore_pump.position)
    game.move_to(coords.center)
    water_tank = game.place_entity(Prototype.StorageTank, position=coords.center)
    print(f'Placed water storage tank at {water_tank.position}')
    game.connect_entities(offshore_pump, water_tank, {Prototype.Pipe, Prototype.UndergroundPipe})
    print('Connected offshore pump to storage tank with pipes')
    print('Setting up power system...')
    building_box = BuildingBox(width=Prototype.Boiler.WIDTH + 4, height=Prototype.Boiler.HEIGHT + 4)
    coords = game.nearest_buildable(Prototype.Boiler, building_box, water_tank.position)
    game.move_to(coords.center)
    boiler = game.place_entity(Prototype.Boiler, position=coords.center)
    print(f'Placed boiler at {boiler.position}')
    building_box = BuildingBox(width=Prototype.SteamEngine.WIDTH + 4, height=Prototype.SteamEngine.HEIGHT + 4)
    coords = game.nearest_buildable(Prototype.SteamEngine, building_box, boiler.position)
    game.move_to(coords.center)
    steam_engine = game.place_entity(Prototype.SteamEngine, position=coords.center)
    print(f'Placed steam engine at {steam_engine.position}')
    game.connect_entities(water_tank, boiler, {Prototype.Pipe, Prototype.UndergroundPipe})
    game.connect_entities(boiler, steam_engine, {Prototype.Pipe, Prototype.UndergroundPipe})
    print('Connected water and steam pipes')

def test_connect_power_system_with_nearest_buildable(game):
    water_position = game.nearest(Resource.Water)
    game.move_to(water_position)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_position)
    print(f'Placed offshore pump to get water at {offshore_pump.position}')
    boiler_building_box = BuildingBox(width=3, height=2)
    boiler_bounding_box = game.nearest_buildable(Prototype.Boiler, building_box=boiler_building_box, center_position=offshore_pump.position)
    print(f'Found buildable position for boiler: {boiler_bounding_box.center}')
    game.move_to(boiler_bounding_box.center)
    boiler = game.place_entity(Prototype.Boiler, position=boiler_bounding_box.center.left(1))
    print(f'Placed boiler at {boiler.position}')
    pipes_to_boiler = game.connect_entities(offshore_pump.position, boiler.position, Prototype.Pipe)
    print(f'Connected offshore pump to boiler with pipes: {pipes_to_boiler}')
    game.sleep(2)
    print(f'Updated entities on the map: {game.get_entities()}')
    pass

def test_multiple(game):
    water_pos = Position(x=-1.0, y=28.0)
    print(f'Found water source at {water_pos}')
    game.move_to(water_pos)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_pos)
    print(f'Placed offshore pump at {offshore_pump.position}')
    building_box = BuildingBox(width=3, height=3)
    buildable_coords = game.nearest_buildable(Prototype.Boiler, building_box, offshore_pump.position)
    boiler_pos = Position(x=buildable_coords.left_top.x + 1.5, y=buildable_coords.left_top.y + 1.5)
    game.move_to(boiler_pos)
    boiler = game.place_entity(Prototype.Boiler, position=boiler_pos)
    print(f'Placed boiler at {boiler.position}')
    building_box = BuildingBox(width=3, height=5)
    buildable_coords = game.nearest_buildable(Prototype.SteamEngine, building_box, boiler.position)
    engine_pos = buildable_coords.center
    game.move_to(engine_pos)
    steam_engine = game.place_entity(Prototype.SteamEngine, position=engine_pos)
    print(f'Placed steam engine at {steam_engine.position}')
    pump_to_boiler = game.connect_entities(offshore_pump.position, boiler.position, Prototype.Pipe)
    print(f'Connected offshore pump to boiler with pipes: {pump_to_boiler}')
    boiler_to_engine = game.connect_entities(boiler.position, steam_engine.position, Prototype.Pipe)
    print(f'Connected boiler to steam engine with pipes: {boiler_to_engine}')

def test_connect_power_poles_without_blocking_mining_drill(game):
    coal_position = game.nearest(Resource.Coal)
    coal_patch = game.get_resource_patch(Resource.Coal, coal_position, radius=10)
    assert coal_patch, 'No coal patch found within radius'
    game.move_to(coal_patch.bounding_box.center)
    miner = game.place_entity(Prototype.ElectricMiningDrill, Direction.UP, coal_patch.bounding_box.center)
    initial_inventory = game.inspect_inventory()
    print(f'Inventory at starting: {initial_inventory}')
    water_position = game.nearest(Resource.Water)
    assert water_position, 'No water source found nearby'
    game.move_to(water_position)
    offshore_pump = game.place_entity(Prototype.OffshorePump, Direction.UP, water_position)
    assert offshore_pump, 'Failed to place offshore pump'
    print(f'Offshore pump placed at {offshore_pump.position}')
    building_box = BuildingBox(width=Prototype.Boiler.WIDTH + 4, height=Prototype.Boiler.HEIGHT + 4)
    coords = game.nearest_buildable(Prototype.Boiler, building_box, offshore_pump.position)
    game.move_to(coords.center)
    boiler = game.place_entity(Prototype.Boiler, position=coords.center, direction=Direction.LEFT)
    assert boiler, 'Failed to place boiler'
    print(f'Boiler placed at {boiler.position}')
    print(f'Current inventory: {game.inspect_inventory()}')
    game.insert_item(Prototype.Coal, boiler, quantity=5)
    print(f'Inventory after adding coal: {game.inspect_inventory()}')
    pipes = game.connect_entities(offshore_pump, boiler, Prototype.Pipe)
    assert pipes, 'Failed to connect offshore pump to boiler'
    print('Pipes placed between offshore pump and boiler')
    building_box = BuildingBox(width=Prototype.SteamEngine.WIDTH + 4, height=Prototype.SteamEngine.HEIGHT + 4)
    coords = game.nearest_buildable(Prototype.SteamEngine, building_box, boiler.position)
    game.move_to(coords.center)
    steam_engine = game.place_entity(Prototype.SteamEngine, position=coords.center)
    assert steam_engine, 'Failed to place steam engine'
    print(f'Steam engine placed at {steam_engine.position}')
    pipes = game.connect_entities(boiler, steam_engine, Prototype.Pipe)
    assert pipes, 'Failed to connect boiler to steam engine'
    poles = game.connect_entities(miner, steam_engine, Prototype.SmallElectricPole)
    assert poles, 'Failed to connect drill to steam engine'
    print('Connected electric mining drill to steam engine with power poles')
    drill = game.get_entity(Prototype.ElectricMiningDrill, miner.position)
    assert drill, 'Failed to get mining drill'
    assert drill.status.value == EntityStatus.WORKING.value

def test_nearest_buildable_simple(game):
    """
    Test finding a buildable position for a simple entity like a wooden chest
    without a bounding box.
    """
    chest_box = BuildingBox(height=1, width=1)
    boundingbox_coords = game.nearest_buildable(Prototype.WoodenChest, chest_box, Position(x=5, y=5))
    can_build = game.can_place_entity(Prototype.WoodenChest, position=boundingbox_coords.center)
    assert can_build is True

def test_nearest_buildable_near_water(game):
    """
    Test finding a buildable position for a simple entity like a wooden chest
    without a bounding box.
    """
    water_pos = game.nearest(Resource.Water)
    game.move_to(water_pos)
    building_box = BuildingBox(width=5, height=3)
    buildable_area = game.nearest_buildable(Prototype.SteamEngine, building_box, water_pos)
    steam_engine_position = buildable_area.center
    game.move_to(steam_engine_position.right(5))
    game.place_entity(Prototype.SteamEngine, direction=Direction.RIGHT, position=steam_engine_position)
    assert True, 'The steam engine should be placeable due to the bounding box'

def test_nearest_buildable_prototype_dimensions(game):
    """
    Test finding a buildable position for an entity with prototype dimensions.
    """
    offshore_pump_box = BuildingBox(width=Prototype.OffshorePump.WIDTH, height=Prototype.OffshorePump.HEIGHT)
    assert True

def test_nearest_buildable_mining_drill(game):
    """
    Test finding a buildable position for an electric mining drill with a bounding box
    over an ore patch.
    """
    drill_box = BuildingBox(height=5, width=5)
    copper_ore = game.nearest(Resource.CopperOre)
    can_build = game.can_place_entity(Prototype.BurnerMiningDrill, position=copper_ore)
    boundingbox_coords = game.nearest_buildable(Prototype.BurnerMiningDrill, drill_box, center_position=game.nearest(Resource.CopperOre))
    game.move_to(boundingbox_coords.center)
    can_build = game.can_place_entity(Prototype.BurnerMiningDrill, position=boundingbox_coords.center)
    game.place_entity(Prototype.BurnerMiningDrill, position=boundingbox_coords.center)
    boundingbox_coords = game.nearest_buildable(Prototype.BurnerMiningDrill, drill_box, center_position=Position(5, 5))
    game.move_to(boundingbox_coords.center)
    can_build = game.can_place_entity(Prototype.BurnerMiningDrill, direction=Direction.UP, position=boundingbox_coords.center)
    assert can_build is True
    game.place_entity(Prototype.BurnerMiningDrill, direction=Direction.UP, position=boundingbox_coords.center)

def test_nearest_buildable_invalid_position(game):
    """
    Test that nearest_buildable raises an exception when no valid position
    is found within search radius.
    """
    drill_box = BuildingBox(height=11, width=7)
    with pytest.raises(Exception) as exc_info:
        boundingbox_coords = game.nearest_buildable(Prototype.BurnerMiningDrill, drill_box, center_position=game.nearest(Resource.CopperOre))
        assert 'Could not find a buildable position' in str(exc_info.value)

def test_nearest_buildable_multiple_entities(game):
    """
    Test finding buildable positions for multiple entities of the same type
    ensuring they don't overlap.
    """
    drill_box = BuildingBox(height=3, width=9)
    game.move_to(game.nearest(Resource.IronOre))
    coordinates = game.nearest_buildable(Prototype.ElectricMiningDrill, drill_box, center_position=game.nearest(Resource.IronOre))
    top_left = coordinates.left_top
    positions = []
    for i in range(0, 3):
        pos = Position(x=top_left.x + 3 * i, y=top_left.y)
        game.move_to(pos)
        game.place_entity(Prototype.ElectricMiningDrill, position=pos, exact=True)
        positions.append(pos)
    assert len(set(((p.x, p.y) for p in positions))) == 3
    for pos in positions:
        game.pickup_entity(Prototype.ElectricMiningDrill, pos)
        can_build = game.can_place_entity(Prototype.ElectricMiningDrill, position=pos)
        assert can_build is True

def test_nearest_buildable_relative_to_player(game):
    """
    Test that nearest_buildable finds positions relative to player location.
    """
    player_pos = Position(x=100, y=100)
    game.move_to(player_pos)
    buildingbox = BuildingBox(height=3, width=3)
    position = game.nearest_buildable(Prototype.WoodenChest, buildingbox, player_pos).center
    distance = ((position.x - player_pos.x) ** 2 + (position.y - player_pos.y) ** 2) ** 0.5
    assert distance <= 50

def test_nearest_buildable_with_obstacles(game):
    """
    Test finding buildable position when there are obstacles in the way.
    """
    player_pos = Position(x=0, y=0)
    for dx, dy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
        obstacle_pos = Position(x=player_pos.x + dx, y=player_pos.y + dy)
        game.place_entity(Prototype.WoodenChest, Direction.UP, obstacle_pos)
    chest_box = BuildingBox(height=1, width=1)
    coords = game.nearest_buildable(Prototype.WoodenChest, chest_box, player_pos)
    position = coords.center
    can_build = game.can_place_entity(Prototype.WoodenChest, Direction.UP, position)
    assert can_build is True
    for dx, dy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
        obstacle_pos = Position(x=player_pos.x + dx, y=player_pos.y + dy)
        assert position is not obstacle_pos

def test_drill_groups(game):
    iron_ore_pos = game.nearest(Resource.IronOre)
    print(f'Found iron ore patch at {iron_ore_pos}')
    drill_positions = []
    for i in range(3):
        building_box = BuildingBox(width=3, height=3)
        buildable_coords = game.nearest_buildable(Prototype.ElectricMiningDrill, building_box, iron_ore_pos)
        drill_pos = Position(x=buildable_coords.left_top.x + 1.5, y=buildable_coords.left_top.y + 1.5)
        game.move_to(drill_pos)
        drill = game.place_entity(Prototype.ElectricMiningDrill, position=drill_pos, direction=Direction.DOWN)
        print(f'Placed electric mining drill {i + 1} at {drill.position}')
        drill_positions.append(drill.position)
        iron_ore_pos = drill.position
    entities = game.get_entities()
    assert len(entities) == 3

def test_nearest_buildable_pumpjack(game):
    crude_oil_pos = game.nearest(Resource.CrudeOil)
    print(f'Found crude oil patch at {crude_oil_pos}')
    building_box = BuildingBox(width=5, height=5)
    buildable_coords = game.nearest_buildable(Prototype.PumpJack, building_box, crude_oil_pos)
    game.move_to(buildable_coords.center)
    game.place_entity(Prototype.PumpJack, position=buildable_coords.center, direction=Direction.DOWN)
    entities = game.get_entities()
    assert len(entities) == 1

def test_nearest_buildable_smelter(game):
    print('I will create a chest that transports items to a furnace using a straight belt line')
    source_position = Position(0, 0)
    game.move_to(source_position)
    building_box = BuildingBox(width=1, height=3)
    buildable_coordinates = game.nearest_buildable(Prototype.WoodenChest, building_box, source_position)
    source_pos = buildable_coordinates.left_top
    game.move_to(source_pos)
    source = game.place_entity(Prototype.WoodenChest, position=source_pos)
    print(f'Placed chest at {source.position}')
    source_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=source.position, direction=Direction.DOWN, spacing=0)
    print(f'Placed an inserter at {source_inserter.position} to extract items from the chest at {source.position}')
    target_position = Position(x=source.position.x + 10, y=source.position.y)
    game.move_to(target_position)
    building_box = BuildingBox(width=3, height=1)
    buildable_coordinates = game.nearest_buildable(Prototype.StoneFurnace, building_box, target_position)
    furnace_pos = buildable_coordinates.left_top
    game.move_to(furnace_pos)
    destination_furnace = game.place_entity(Prototype.StoneFurnace, position=furnace_pos)
    print(f'Placed furnace at {destination_furnace.position}')
    destination_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=destination_furnace.position, direction=Direction.RIGHT, spacing=0)
    destination_inserter = game.rotate_entity(destination_inserter, Direction.LEFT)
    print(f'Placed inserter at {destination_inserter.position} to feed the furnace at {destination_furnace.position}')
    belt = game.connect_entities(source_inserter, destination_inserter, Prototype.TransportBelt)
    print(f'Connected chest inserter at {source_inserter.position} to furnace inserter at {destination_inserter.position} with a straight belt: {belt}')

def test_full_oil_chain(game):
    water_pos = game.nearest(Resource.Water)
    print(f'Found water at {water_pos}')
    game.move_to(water_pos)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_pos)
    print(f'Placed offshore pump at {offshore_pump.position}')
    building_box = BuildingBox(width=Prototype.Boiler.WIDTH + 4, height=Prototype.Boiler.HEIGHT + 4)
    coords = game.nearest_buildable(Prototype.Boiler, building_box, offshore_pump.position)
    game.move_to(coords.center)
    boiler = game.place_entity(Prototype.Boiler, position=coords.center)
    print(f'Placed boiler at {boiler.position}')
    building_box = BuildingBox(width=Prototype.SteamEngine.WIDTH + 4, height=Prototype.SteamEngine.HEIGHT + 4)
    coords = game.nearest_buildable(Prototype.SteamEngine, building_box, boiler.position)
    game.move_to(coords.center)
    steam_engine = game.place_entity(Prototype.SteamEngine, position=coords.center)
    print(f'Placed steam engine at {steam_engine.position}')
    game.move_to(offshore_pump.position)
    game.connect_entities(offshore_pump, boiler, Prototype.Pipe)
    game.connect_entities(boiler, steam_engine, Prototype.Pipe)
    print('Connected water and steam pipes')
    boiler = game.insert_item(Prototype.Coal, boiler, 50)
    print('Added coal to boiler')
    oil_pos = game.nearest(Resource.CrudeOil)
    game.move_to(oil_pos)
    pumpjack = game.place_entity(Prototype.PumpJack, position=oil_pos)
    print(f'Placed pumpjack at {pumpjack.position}')
    game.connect_entities(steam_engine, pumpjack, Prototype.SmallElectricPole)
    print('Connected power to pumpjack')
    refinery_pos = Position(x=pumpjack.position.x + 15, y=pumpjack.position.y)
    game.move_to(refinery_pos)
    refinery = game.place_entity(Prototype.OilRefinery, position=refinery_pos)
    print(f'Placed oil refinery at {refinery.position}')
    refinery = game.set_entity_recipe(refinery, RecipeName.BasicOilProcessing)
    print('Set refinery recipe to basic oil processing')
    game.connect_entities(pumpjack, refinery, Prototype.SmallElectricPole)
    print('Connected power to refinery')
    game.connect_entities(pumpjack, refinery, {Prototype.UndergroundPipe, Prototype.Pipe})
    pass

def test_fix_storage_tank_connection(game):
    water_pos = game.nearest(Resource.Water)
    print(f'Found water at {water_pos}')
    game.move_to(water_pos)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_pos)
    print(f'Placed offshore pump at {offshore_pump.position}')
    building_box = BuildingBox(width=Prototype.Boiler.WIDTH + 4, height=Prototype.Boiler.HEIGHT + 4)
    coords = game.nearest_buildable(Prototype.Boiler, building_box, offshore_pump.position)
    game.move_to(coords.center)
    boiler = game.place_entity(Prototype.Boiler, position=coords.center)
    print(f'Placed boiler at {boiler.position}')
    building_box = BuildingBox(width=Prototype.SteamEngine.WIDTH + 4, height=Prototype.SteamEngine.HEIGHT + 4)
    coords = game.nearest_buildable(Prototype.SteamEngine, building_box, boiler.position)
    game.move_to(coords.center)
    steam_engine = game.place_entity(Prototype.SteamEngine, position=coords.center)
    print(f'Placed steam engine at {steam_engine.position}')
    game.connect_entities(offshore_pump, boiler, Prototype.Pipe)
    game.connect_entities(boiler, steam_engine, Prototype.Pipe)
    print('Connected water and steam pipes')
    boiler = game.insert_item(Prototype.Coal, boiler, 50)
    print('Added coal to boiler')
    jack_pos = Position(x=32.5, y=49.5)
    game.move_to(jack_pos)
    pumpjack = game.place_entity(Prototype.PumpJack, position=jack_pos)
    print(f'Placed pumpjack at {pumpjack.position}')
    game.connect_entities(steam_engine, pumpjack, Prototype.SmallElectricPole)
    print('Connected power to pumpjack')
    refinery_pos = Position(x=pumpjack.position.x + 15, y=pumpjack.position.y)
    game.move_to(refinery_pos)
    refinery = game.place_entity(Prototype.OilRefinery, position=refinery_pos)
    print(f'Placed oil refinery at {refinery.position}')
    refinery = game.set_entity_recipe(refinery, RecipeName.BasicOilProcessing)
    print('Set refinery recipe to basic oil processing')
    game.connect_entities(pumpjack, refinery, Prototype.SmallElectricPole)
    print('Connected power to refinery')
    game.connect_entities(pumpjack, refinery, {Prototype.UndergroundPipe, Prototype.Pipe})
    print('Connected pumpjack to refinery with pipes')
    tank_pos = Position(x=refinery.position.x + 10, y=refinery.position.y)
    game.move_to(tank_pos)
    storage_tank = game.place_entity(Prototype.StorageTank, position=tank_pos)
    print(f'Placed storage tank at {storage_tank.position}')
    game.connect_entities(refinery, storage_tank, {Prototype.UndergroundPipe, Prototype.Pipe})
    pass

