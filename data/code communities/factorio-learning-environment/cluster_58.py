# Cluster 58

class GetFactoryCentroid(Tool):

    def __init__(self, lua_script_manager, game_state):
        self.state = {'input': {}, 'output': {}}
        super().__init__(lua_script_manager, game_state)

    def __call__(self) -> Camera:
        """
        Gets the bounding box of the enti factory.
        """
        result, _ = self.execute(self.player_index)
        if isinstance(result, str):
            raise Exception(result)
        result = self.clean_response(result)
        try:
            if 'bounds' in result:
                bounds = BoundingBox(left_top=Position(x=result['bounds']['left_top']['x'], y=result['bounds']['left_top']['y']), right_bottom=Position(x=result['bounds']['right_bottom']['x'], y=result['bounds']['right_bottom']['y']), left_bottom=Position(x=result['bounds']['left_top']['x'], y=result['bounds']['right_bottom']['y']), right_top=Position(x=result['bounds']['right_bottom']['x'], y=result['bounds']['left_top']['y']))
            else:
                bounds = BoundingBox(left_top=Position(x=-10, y=-10), right_bottom=Position(x=10, y=10), left_bottom=Position(x=-10, y=10), right_top=Position(x=10, y=-10))
            return Camera(bounds=bounds, zoom=result['camera']['zoom'], centroid=result['centroid'], raw_centroid=result['raw_centroid'], entity_count=result['entity_count'], position=result['camera']['position'])
        except Exception:
            return None

class GetResourcePatch(Tool):

    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(self, resource: Resource, position: Position, radius: int=30) -> Optional[ResourcePatch]:
        """
        Get the resource patch at position (x, y) if it exists in the radius.
        if radius is set to 0, it will only check the exact position for this resource patch.
        :param resource: Resource to get, e.g Resource.Coal
        :param position: Position to get resource patch
        :param radius: Radius to search for resource patch
        :example coal_patch_at_origin = get_resource_patch(Resource.Coal, Position(x=0, y=0))
        :return: ResourcePatch if found, else None
        """
        response, time_elapsed = self.execute(self.player_index, resource[0], position.x, position.y, radius)
        if not isinstance(response, dict) or response == {}:
            top_level_message = str(response).split(':')[-1].strip()
            raise Exception(f'Could not get {resource[0]} at {position}: {top_level_message}')
        left_top = Position(x=response['bounding_box']['left_top']['x'], y=response['bounding_box']['left_top']['y'])
        right_bottom = Position(x=response['bounding_box']['right_bottom']['x'], y=response['bounding_box']['right_bottom']['y'])
        left_bottom = Position(x=response['bounding_box']['left_top']['x'], y=response['bounding_box']['right_bottom']['y'])
        right_top = Position(x=response['bounding_box']['right_bottom']['x'], y=response['bounding_box']['left_top']['y'])
        bounding_box = BoundingBox(left_top=left_top, right_bottom=right_bottom, left_bottom=left_bottom, right_top=right_top)
        resource_patch = ResourcePatch(name=resource[0], size=response['size'], bounding_box=bounding_box)
        return resource_patch

def test_mining_blueprint_1(game):
    left_top = Position(x=0, y=0)
    right_bottom = Position(x=2, y=9)
    left_bottom = Position(x=left_top.x, y=right_bottom.y)
    right_top = Position(x=right_bottom.x, y=left_top.y)
    miner_box = BoundingBox(left_top=left_top, right_bottom=right_bottom, left_bottom=left_bottom, right_top=right_top)
    origin = game.nearest_buildable(Prototype.BurnerMiningDrill, miner_box, center_position=Position(x=0, y=0))
    assert origin, 'Could not find valid position'
    origin = origin + left_top + Position(x=0.5, y=0.5)
    game.move_to(origin)
    for i in range(10):
        world_y = 0.0 + 1.0 * i + origin.y
        world_x = 0.0 + origin.x
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.TransportBelt, position=Position(x=world_x, y=world_y), direction=Direction.UP)
    for i in range(3):
        world_y = 1.0 + 3.0 * i + origin.y
        world_x = 2.0 + origin.x
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.BurnerInserter, position=Position(x=world_x, y=world_y), direction=Direction.UP)
    for i in range(3):
        world_y = 2.5 + 3.0 * i + origin.y
        world_x = 1.5 + origin.x
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.BurnerMiningDrill, position=Position(x=world_x, y=world_y), direction=Direction.LEFT)
    game.move_to(Position(x=origin.x + 1.0, y=origin.y + 1.0))
    game.place_entity(Prototype.BurnerInserter, position=Position(x=origin.x + 1.0, y=origin.y + 1.0), direction=Direction.DOWN)
    game.move_to(Position(x=origin.x + 2.0, y=origin.y + 0.0))
    game.place_entity(Prototype.WoodenChest, position=Position(x=origin.x + 2.0, y=origin.y + 0.0), direction=Direction.UP)
    game.move_to(Position(x=origin.x + 1.0, y=origin.y + 0.0))
    game.place_entity(Prototype.BurnerInserter, position=Position(x=origin.x + 1.0, y=origin.y + 0.0), direction=Direction.LEFT)

def test_mining_blueprint_2(game):
    left_top = Position(x=0, y=0)
    right_bottom = Position(x=29.0, y=4.0)
    left_bottom = Position(x=left_top.x, y=right_bottom.y)
    right_top = Position(x=right_bottom.x, y=left_top.y)
    miner_box = BoundingBox(left_top=left_top, right_bottom=right_bottom, left_bottom=left_bottom, right_top=right_top)
    origin = game.nearest_buildable(Prototype.ElectricMiningDrill, miner_box, center_position=Position(x=0, y=0))
    assert origin, 'Could not find valid position'
    origin = origin + left_top + Position(x=0.5, y=0.5)
    game.move_to(origin)
    for i in range(4):
        world_x = 6.0 + 6.0 * i + origin.x
        world_y = 0.0 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.ElectricMiningDrill, position=Position(x=world_x, y=world_y), direction=Direction.DOWN)
    for i in range(5):
        world_x = 3.0 + 6.0 * i + origin.x
        world_y = 2.0 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.ElectricMiningDrill, position=Position(x=world_x, y=world_y), direction=Direction.RIGHT)
    for i in range(5):
        world_x = 1.0 + 6.0 * i + origin.x
        world_y = 2.0 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.SmallElectricPole, position=Position(x=world_x, y=world_y), direction=Direction.UP)
    for i in range(4):
        world_x = 6.0 + 6.0 * i + origin.x
        world_y = 4.0 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.ElectricMiningDrill, position=Position(x=world_x, y=world_y), direction=Direction.UP)
    game.move_to(Position(x=origin.x + 0.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 0.0, y=origin.y + 2.0), direction=Direction.RIGHT)
    game.move_to(Position(x=origin.x + 5.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 5.0, y=origin.y + 2.0), direction=Direction.RIGHT)
    game.move_to(Position(x=origin.x + 6.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 6.0, y=origin.y + 2.0), direction=Direction.RIGHT)
    game.move_to(Position(x=origin.x + 11.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 11.0, y=origin.y + 2.0), direction=Direction.RIGHT)
    game.move_to(Position(x=origin.x + 12.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 12.0, y=origin.y + 2.0), direction=Direction.RIGHT)
    game.move_to(Position(x=origin.x + 17.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 17.0, y=origin.y + 2.0), direction=Direction.RIGHT)
    game.move_to(Position(x=origin.x + 18.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 18.0, y=origin.y + 2.0), direction=Direction.RIGHT)
    game.move_to(Position(x=origin.x + 23.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 23.0, y=origin.y + 2.0), direction=Direction.RIGHT)
    game.move_to(Position(x=origin.x + 24.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 24.0, y=origin.y + 2.0), direction=Direction.RIGHT)
    game.move_to(Position(x=origin.x + 29.0, y=origin.y + 2.0))
    game.place_entity(Prototype.UndergroundBelt, position=Position(x=origin.x + 29.0, y=origin.y + 2.0), direction=Direction.RIGHT)

def test_mining_blueprint_3(game):
    left_top = Position(x=-6.0, y=0.0)
    right_bottom = Position(x=23.0, y=4.0)
    left_bottom = Position(x=left_top.x, y=right_bottom.y)
    right_top = Position(x=right_bottom.x, y=left_top.y)
    miner_box = BoundingBox(left_top=left_top, right_bottom=right_bottom, left_bottom=left_bottom, right_top=right_top)
    origin = game.nearest_buildable(Prototype.ElectricMiningDrill, miner_box, center_position=Position(x=0, y=0))
    assert origin, 'Could not find valid position for miners'
    game.move_to(origin)
    for x in range(-9, 10, 6):
        world_x = x + 14.5 + origin.x + left_top.x
        world_y = -0.5 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.ElectricMiningDrill, position=Position(x=world_x, y=world_y), direction=Direction.DOWN)
    for x in range(-12, 13, 6):
        world_x = x + 14.5 + origin.x + left_top.x
        world_y = 1.5 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.ElectricMiningDrill, position=Position(x=world_x, y=world_y), direction=Direction.RIGHT)
    for x in range(-9, 10, 6):
        world_x = x + 14.5 + origin.x + left_top.x
        world_y = 3.5 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.ElectricMiningDrill, position=Position(x=world_x, y=world_y), direction=Direction.UP)
    for x in range(-15, -9, 5):
        world_x = x + 14.5 + origin.x + left_top.x
        world_y = 1.5 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.UndergroundBelt, position=Position(x=world_x, y=world_y), direction=Direction.RIGHT)
    for x in range(-9, -3, 5):
        world_x = x + 14.5 + origin.x + left_top.x
        world_y = 1.5 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.UndergroundBelt, position=Position(x=world_x, y=world_y), direction=Direction.RIGHT)
    for x in range(-3, 3, 5):
        world_x = x + 14.5 + origin.x + left_top.x
        world_y = 1.5 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.UndergroundBelt, position=Position(x=world_x, y=world_y), direction=Direction.RIGHT)
    for x in range(3, 9, 5):
        world_x = x + 14.5 + origin.x + left_top.x
        world_y = 1.5 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.UndergroundBelt, position=Position(x=world_x, y=world_y), direction=Direction.RIGHT)
    for x in range(9, 15, 5):
        world_x = x + 14.5 + origin.x + left_top.x
        world_y = 1.5 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.UndergroundBelt, position=Position(x=world_x, y=world_y), direction=Direction.RIGHT)
    for x in range(-14, 11, 6):
        world_x = x + 14.5 + origin.x + left_top.x
        world_y = 1.5 + origin.y
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.SmallElectricPole, position=Position(x=world_x, y=world_y), direction=Direction.UP)

def test_minig_blueprint_4(game):
    left_top = Position(x=0, y=0)
    right_bottom = Position(x=5, y=5.5)
    left_bottom = Position(x=left_top.x, y=right_bottom.y)
    right_top = Position(x=right_bottom.x, y=left_top.y)
    miner_box = BoundingBox(left_top=left_top, right_bottom=right_bottom, left_bottom=left_bottom, right_top=right_top)
    origin = game.nearest_buildable(Prototype.BurnerMiningDrill, miner_box, center_position=Position(x=0, y=0))
    assert origin, 'Could not find valid position'
    origin = origin + left_top + Position(x=0.5, y=0.5)
    game.move_to(origin)
    game.move_to(Position(x=origin.x + 2.0, y=origin.y + 0.0))
    game.place_entity(Prototype.BurnerMiningDrill, position=Position(x=origin.x + 2.0, y=origin.y + 0.0), direction=Direction.DOWN, exact=True)
    game.move_to(Position(x=origin.x + 0.0, y=origin.y + 3.0))
    game.place_entity(Prototype.BurnerMiningDrill, position=Position(x=origin.x + 0.0, y=origin.y + 3.0), direction=Direction.DOWN, exact=True)
    game.move_to(Position(x=origin.x + 2.5, y=origin.y + 2.5))
    assembling_machine_1_1 = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=origin.x + 2.5, y=origin.y + 2.5), direction=Direction.UP, exact=True)
    game.set_entity_recipe(assembling_machine_1_1, Prototype.StoneFurnace)
    game.move_to(Position(x=origin.x + 5.0, y=origin.y + 3.0))
    game.place_entity(Prototype.BurnerMiningDrill, position=Position(x=origin.x + 5.0, y=origin.y + 3.0), direction=Direction.DOWN, exact=True)
    game.move_to(Position(x=origin.x + 0.0, y=origin.y + 5.0))
    game.place_entity(Prototype.StoneFurnace, position=Position(x=origin.x + 0.0, y=origin.y + 5.0), direction=Direction.UP, exact=True)
    game.move_to(Position(x=origin.x + 2.5, y=origin.y + 4.5))
    game.place_entity(Prototype.Inserter, position=Position(x=origin.x + 2.5, y=origin.y + 4.5), direction=Direction.UP, exact=True)
    game.move_to(Position(x=origin.x + 5.0, y=origin.y + 5.0))
    game.place_entity(Prototype.StoneFurnace, position=Position(x=origin.x + 5.0, y=origin.y + 5.0), direction=Direction.UP, exact=True)
    game.move_to(Position(x=origin.x + 3.5, y=origin.y + 4.5))
    game.place_entity(Prototype.SmallElectricPole, position=Position(x=origin.x + 3.5, y=origin.y + 4.5), direction=Direction.UP, exact=True)
    game.move_to(Position(x=origin.x + 2.5, y=origin.y + 5.5))
    game.place_entity(Prototype.WoodenChest, position=Position(x=origin.x + 2.5, y=origin.y + 5.5), direction=Direction.UP, exact=True)
    game.move_to(Position(x=origin.x + 1.5, y=origin.y + 5.5))
    game.place_entity(Prototype.Inserter, position=Position(x=origin.x + 1.5, y=origin.y + 5.5), direction=Direction.LEFT, exact=True)
    game.move_to(Position(x=origin.x + 3.5, y=origin.y + 5.5))
    game.place_entity(Prototype.Inserter, position=Position(x=origin.x + 3.5, y=origin.y + 5.5), direction=Direction.RIGHT, exact=True)
    game.get_entities()

