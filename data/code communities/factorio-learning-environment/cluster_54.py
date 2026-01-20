# Cluster 54

def test_build_iron_gear_factory(game):
    """
    Build a factory that produces iron gears from iron plates.
    :param game:
    :return:
    """
    iron_ore_patch = game.get_resource_patch(Resource.IronOre, game.nearest(Resource.IronOre))
    game.move_to(iron_ore_patch.bounding_box.left_top + Position(x=1, y=1))
    while game.inspect_inventory()[Prototype.IronOre] < 80:
        game.harvest_resource(iron_ore_patch.bounding_box.left_top, quantity=10)
    stone_patch = game.get_resource_patch(Resource.Stone, game.nearest(Resource.Stone))
    game.move_to(stone_patch.bounding_box.left_top + Position(x=1, y=1))
    game.harvest_resource(stone_patch.bounding_box.left_top, quantity=10)
    coal_patch: ResourcePatch = game.get_resource_patch(Resource.Coal, game.nearest(Resource.Coal))
    game.move_to(coal_patch.bounding_box.left_top + Position(x=1, y=1))
    while game.inspect_inventory()[Prototype.Coal] < 30:
        game.harvest_resource(coal_patch.bounding_box.left_top, quantity=10)
    copper_patch: ResourcePatch = game.get_resource_patch(Resource.CopperOre, game.nearest(Resource.CopperOre))
    game.move_to(copper_patch.bounding_box.left_top + Position(x=1, y=1))
    while game.inspect_inventory()[Prototype.CopperOre] < 30:
        game.harvest_resource(copper_patch.bounding_box.left_top, quantity=10)
    game.move_to(Position(x=0, y=0))
    stone_furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=0, y=0))
    game.insert_item(Prototype.Coal, stone_furnace, quantity=20)
    game.insert_item(Prototype.IronOre, stone_furnace, quantity=50)
    while game.inspect_inventory(stone_furnace)[Prototype.IronPlate] < 50:
        sleep(1)
    game.extract_item(Prototype.IronPlate, stone_furnace, quantity=50)
    game.insert_item(Prototype.IronOre, stone_furnace, quantity=30)
    while game.inspect_inventory(stone_furnace)[Prototype.IronPlate] < 30:
        sleep(1)
    game.extract_item(Prototype.IronPlate, stone_furnace, quantity=30)
    game.insert_item(Prototype.CopperOre, stone_furnace, quantity=20)
    while game.inspect_inventory(stone_furnace)[Prototype.CopperPlate] < 20:
        sleep(5)
    game.extract_item(Prototype.CopperPlate, stone_furnace, quantity=20)
    game.pickup_entity(stone_furnace)
    recipe: Recipe = game.get_prototype_recipe(Prototype.BurnerMiningDrill)
    for ingredient in recipe.ingredients:
        if game.inspect_inventory()[ingredient.name] < ingredient.count:
            game.craft_item(ingredient.name, quantity=ingredient.count)
    game.craft_item(Prototype.BurnerMiningDrill)
    game.move_to(iron_ore_patch.bounding_box.left_top + Position(x=1, y=1))
    burner_mining_drill: BurnerMiningDrill = game.place_entity(Prototype.BurnerMiningDrill, position=iron_ore_patch.bounding_box.left_top)
    game.insert_item(Prototype.Coal, burner_mining_drill, quantity=5)
    stone_furnace = game.place_entity_next_to(Prototype.StoneFurnace, reference_position=burner_mining_drill.drop_position, direction=Direction.UP, spacing=0)
    burner_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=stone_furnace.position, direction=Direction.UP, spacing=0)

    def ensure_ingredients(game, recipe, quantity=1):
        for ingredient in recipe.ingredients:
            required = ingredient.count * quantity
            available = game.inspect_inventory()[ingredient.name]
            if available < required:
                craft_recursive(game, ingredient.name, required - available)

    def craft_recursive(game, item_name, quantity):
        if item_name in ['copper-ore', 'iron-ore', 'wood', 'copper-plate', 'iron-plate']:
            return
        recipe = game.get_prototype_recipe(item_name)
        ensure_ingredients(game, recipe, quantity)
        game.craft_item(item_name, quantity=quantity)
    recipe = game.get_prototype_recipe(Prototype.AssemblingMachine1)
    ensure_ingredients(game, recipe)
    game.craft_item(Prototype.AssemblingMachine1)
    assembly_machine = game.place_entity_next_to(Prototype.AssemblingMachine1, reference_position=burner_inserter.drop_position, direction=Direction.UP, spacing=0)
    game.set_entity_recipe(assembly_machine, Prototype.IronGearWheel)
    recipe = game.get_prototype_recipe(Prototype.OffshorePump)
    ensure_ingredients(game, recipe)
    game.craft_item(Prototype.OffshorePump)
    game.move_to(game.nearest(Resource.Water))
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=game.nearest(Resource.Water))
    recipe = game.get_prototype_recipe(Prototype.Boiler)
    ensure_ingredients(game, recipe)
    game.craft_item(Prototype.Boiler)
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=Direction.UP, spacing=2)
    recipe = game.get_prototype_recipe(Prototype.SteamEngine)
    ensure_ingredients(game, recipe)
    game.craft_item(Prototype.SteamEngine)
    steam_engine = game.place_entity_next_to(Prototype.SteamEngine, reference_position=boiler.position, direction=Direction.RIGHT, spacing=2)
    tree_patch = game.get_resource_patch(Resource.Wood, game.nearest(Resource.Wood))
    game.move_to(tree_patch.bounding_box.left_top + Position(x=1, y=1))
    game.harvest_resource(tree_patch.bounding_box.left_top, quantity=30)
    recipe = game.get_prototype_recipe(Prototype.SmallElectricPole)
    ensure_ingredients(game, recipe, quantity=12)
    game.craft_item(Prototype.SmallElectricPole, quantity=12)
    game.connect_entities(steam_engine, assembly_machine, connection_type=Prototype.SmallElectricPole)

def craft_recursive(game, item_name, quantity):
    if item_name in ['copper-ore', 'iron-ore', 'wood', 'copper-plate', 'iron-plate']:
        return
    recipe = game.get_prototype_recipe(item_name)
    ensure_ingredients(game, recipe, quantity)
    game.craft_item(item_name, quantity=quantity)

def ensure_ingredients(game, recipe, quantity=1):
    for ingredient in recipe.ingredients:
        required = ingredient.count * quantity
        available = game.inspect_inventory()[ingredient.name]
        if available < required:
            craft_recursive(game, ingredient.name, required - available)

def test_build_iron_gear_factory_from_scratch(game):
    """
    Build a factory that produces iron gears from iron plates.
    :param game:
    :return:
    """
    iron_ore_patch = game.get_resource_patch(Resource.IronOre, game.nearest(Resource.IronOre))
    game.move_to(iron_ore_patch.bounding_box.left_top + Position(x=1, y=1))
    while game.inspect_inventory()[Prototype.IronOre] < 80:
        game.harvest_resource(iron_ore_patch.bounding_box.left_top, quantity=10)
    stone_patch = game.get_resource_patch(Resource.Stone, game.nearest(Resource.Stone))
    game.move_to(stone_patch.bounding_box.left_top + Position(x=1, y=1))
    game.harvest_resource(stone_patch.bounding_box.left_top, quantity=10)
    coal_patch: ResourcePatch = game.get_resource_patch(Resource.Coal, game.nearest(Resource.Coal))
    game.move_to(coal_patch.bounding_box.left_top + Position(x=1, y=1))
    while game.inspect_inventory()[Prototype.Coal] < 30:
        game.harvest_resource(coal_patch.bounding_box.left_top, quantity=10)
    copper_patch: ResourcePatch = game.get_resource_patch(Resource.CopperOre, game.nearest(Resource.CopperOre))
    game.move_to(copper_patch.bounding_box.left_top + Position(x=1, y=1))
    while game.inspect_inventory()[Prototype.CopperOre] < 30:
        game.harvest_resource(copper_patch.bounding_box.left_top, quantity=10)
    game.move_to(Position(x=0, y=0))
    stone_furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=0, y=0))
    game.insert_item(Prototype.Coal, stone_furnace, quantity=20)
    game.insert_item(Prototype.IronOre, stone_furnace, quantity=50)
    while game.inspect_inventory(stone_furnace)[Prototype.IronPlate] < 50:
        sleep(1)
    game.extract_item(Prototype.IronPlate, stone_furnace, quantity=50)
    game.insert_item(Prototype.IronOre, stone_furnace, quantity=30)
    while game.inspect_inventory(stone_furnace)[Prototype.IronPlate] < 30:
        sleep(1)
    game.extract_item(Prototype.IronPlate, stone_furnace, quantity=30)
    game.insert_item(Prototype.CopperOre, stone_furnace, quantity=20)
    while game.inspect_inventory(stone_furnace)[Prototype.CopperPlate] < 20:
        sleep(5)
    game.extract_item(Prototype.CopperPlate, stone_furnace, quantity=20)
    game.pickup_entity(stone_furnace)
    recipe: Recipe = game.get_prototype_recipe(Prototype.BurnerMiningDrill)
    for ingredient in recipe.ingredients:
        if game.inspect_inventory()[ingredient.name] < ingredient.count:
            game.craft_item(ingredient.name, quantity=ingredient.count)
    game.craft_item(Prototype.BurnerMiningDrill)
    game.move_to(iron_ore_patch.bounding_box.left_top + Position(x=1, y=1))
    burner_mining_drill: BurnerMiningDrill = game.place_entity(Prototype.BurnerMiningDrill, position=iron_ore_patch.bounding_box.left_top)
    game.insert_item(Prototype.Coal, burner_mining_drill, quantity=5)
    stone_furnace = game.place_entity_next_to(Prototype.StoneFurnace, reference_position=burner_mining_drill.position, direction=Direction.UP, spacing=0)
    burner_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=stone_furnace.position, direction=Direction.UP, spacing=0)

    def ensure_ingredients(game, recipe, quantity=1):
        for ingredient in recipe.ingredients:
            required = ingredient.count * quantity
            available = game.inspect_inventory()[ingredient.name]
            if available < required:
                craft_recursive(game, ingredient.name, required - available)

    def craft_recursive(game, item_name, quantity):
        recipe = game.get_prototype_recipe(item_name)
        ensure_ingredients(game, recipe, quantity)
        game.craft_item(item_name, quantity=quantity)
    recipe = game.get_prototype_recipe(Prototype.AssemblingMachine1)
    ensure_ingredients(game, recipe)
    game.craft_item(Prototype.AssemblingMachine1)
    assembly_machine = game.place_entity_next_to(Prototype.AssemblingMachine1, reference_position=burner_inserter.position, direction=Direction.UP, spacing=0)
    game.set_entity_recipe(assembly_machine, Prototype.IronGearWheel)
    recipe = game.get_prototype_recipe(Prototype.OffshorePump)
    ensure_ingredients(game, recipe)
    game.craft_item(Prototype.OffshorePump)
    game.move_to(game.nearest(Resource.Water))
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=game.nearest(Resource.Water), direction=Direction.LEFT)
    recipe = game.get_prototype_recipe(Prototype.Boiler)
    ensure_ingredients(game, recipe)
    game.craft_item(Prototype.Boiler)
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=Direction.UP, spacing=2)
    recipe = game.get_prototype_recipe(Prototype.SteamEngine)
    ensure_ingredients(game, recipe)
    game.craft_item(Prototype.SteamEngine)
    steam_engine = game.place_entity_next_to(Prototype.SteamEngine, reference_position=boiler.position, direction=Direction.LEFT, spacing=5)
    nearest_wood = game.nearest(Resource.Wood)
    game.move_to(nearest_wood)
    game.harvest_resource(nearest_wood, quantity=40)
    recipe = game.get_prototype_recipe(Prototype.SmallElectricPole)
    ensure_ingredients(game, recipe, quantity=10)
    game.craft_item(Prototype.SmallElectricPole, quantity=10)
    game.connect_entities(steam_engine, assembly_machine, connection_type=Prototype.SmallElectricPole)
    game.connect_entities(boiler, steam_engine, connection_type=Prototype.Pipe)
    game.connect_entities(boiler, offshore_pump, connection_type=Prototype.Pipe)
    game.move_to(boiler.position)
    game.insert_item(Prototype.Coal, boiler, quantity=10)
    game.insert_item(Prototype.Coal, burner_inserter, quantity=10)
    game.insert_item(Prototype.Coal, stone_furnace, quantity=10)
    game.move_to(burner_mining_drill.position)
    game.insert_item(Prototype.Coal, burner_mining_drill, quantity=10)
    game.sleep(30)
    game.move_to(assembly_machine.position)
    extracted = game.extract_item(Prototype.IronGearWheel, assembly_machine, quantity=3)
    inventory = game.inspect_inventory(entity=assembly_machine)
    assert inventory.get(Prototype.IronGearWheel) == 0 and extracted == 3

def ensure_ingredients(game, recipe, quantity=1):
    for ingredient in recipe.ingredients:
        required = ingredient.count * quantity
        available = game.inspect_inventory()[ingredient.name]
        if available < required:
            craft_recursive(game, ingredient.name, required - available)

def craft_recursive(game, item_name, quantity):
    recipe = game.get_prototype_recipe(item_name)
    ensure_ingredients(game, recipe, quantity)
    game.craft_item(item_name, quantity=quantity)

