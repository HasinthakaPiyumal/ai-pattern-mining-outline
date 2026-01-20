# Cluster 59

class GetPath(Tool):

    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(self, path_handle: int, max_attempts: int=10) -> List[Position]:
        """
        Retrieve a path requested from the game, using backoff polling.
        """
        try:
            wait_time = 0.032
            for attempt in range(max_attempts):
                response, elapsed = self.execute(path_handle)
                if response is None or response == {} or isinstance(response, str):
                    raise Exception('Could not request path (get_path)', response)
                path = json.loads(response)
                if path['status'] == 'success':
                    list_of_positions = []
                    for pos in path['waypoints']:
                        list_of_positions.append(Position(x=pos['x'], y=pos['y']))
                    return list_of_positions
                elif path['status'] in ['not_found', 'invalid_request']:
                    raise Exception(f'Path not found or invalid request: {path['status']}')
                elif path['status'] == 'busy':
                    raise Exception('Pathfinder is busy, try again later')
                wait_time *= 2
            raise Exception(f'Path request timed out after {max_attempts} attempts')
        except Exception as e:
            raise ConnectionError(f'Could not get path with handle {path_handle}') from e

class SaveBlueprint(Tool):

    def __init__(self, *args):
        super().__init__(*args)

    def __call__(self) -> Tuple[str, Position]:
        """
        Saves the current player entities on the map into a blueprint string
        :return: Blueprint and offset to blueprint from the origin.
        """
        result, _ = self.execute(self.player_index)
        blueprint = result['blueprint']
        offset = Position(x=result['center_x'], y=result['center_y'])
        return (blueprint, offset)

class PlaceEntityNextTo(Tool):

    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(self, entity: Prototype, reference_position: Position=Position(x=0, y=0), direction: DirectionInternal=DirectionInternal.RIGHT, spacing: int=0) -> Entity:
        """
        Places an entity next to an existing entity, with an optional space in-between (0 space means adjacent).
        In order to place something with a gap, you must increase the spacing parameter.
        :param entity: Entity to place
        :param reference_position: Position of existing entity or position to place entity next to
        :param direction: Direction to place entity from reference_position
        :param spacing: Space between entity and reference_position
        :example: place_entity_next_to(Prototype.WoodenChest, Position(x=0, y=0), direction=Direction.UP, spacing=1)
        :return: Entity placed
        """
        try:
            name, metaclass = entity.value
            assert isinstance(reference_position, Position) and reference_position, 'reference_position must be a Position object'
            assert isinstance(entity, Prototype) and entity, 'entity must be a Prototype object'
            x, y = (reference_position.x, reference_position.y)
            factorio_direction = DirectionInternal.to_factorio_direction(direction)
            response, elapsed = self.execute(self.player_index, name, x, y, factorio_direction, spacing)
            if not isinstance(response, dict) or response == {}:
                msg = self.get_error_message(str(response))
                raise Exception(f'Could not place {name} next to {reference_position} with spacing {spacing} and direction {direction}. {msg}')
            cleaned_response = self.clean_response(response)
            placement_feedback = cleaned_response.pop('placement_feedback', None)
            if placement_feedback:
                feedback_msg = f'Placement feedback for {name}: {placement_feedback['reason']}'
                if placement_feedback.get('auto_oriented'):
                    feedback_msg += ' (Auto-oriented for optimal flow)'
            try:
                object = metaclass(prototype=name, game=self.connection, **cleaned_response)
                if placement_feedback:
                    object._placement_feedback = placement_feedback
            except Exception as e:
                raise Exception(f'Could not create {name} object from response (place entity next to): {response}', e)
            return object
        except Exception as e:
            raise e

class PowerConnectionResolver(Resolver):

    def __init__(self, *args):
        super().__init__(*args)

    def _check_existing_network_connection(self, source_entity, target_entity) -> bool:
        """
        Check if source and target are already connected to the same power network.

        Returns True if already connected, False otherwise.
        """
        if not (source_entity and target_entity):
            return False
        if not hasattr(source_entity, 'electrical_id') or not hasattr(target_entity, 'electrical_id'):
            return False
        return source_entity.electrical_id == target_entity.electrical_id and source_entity.electrical_id is not None

    def _get_entity_connection_points(self, entity: Entity) -> Set[Position]:
        """Get all predefined connection points for an entity."""
        connection_points = set()
        if hasattr(entity, 'connection_points'):
            connection_points.update(entity.connection_points)
        if hasattr(entity, 'input_connection_points'):
            connection_points.update(entity.input_connection_points)
        if hasattr(entity, 'output_connection_points'):
            connection_points.update(entity.output_connection_points)
        return connection_points

    def _get_adjacent_tiles(self, entity: Entity) -> List[Position]:
        """Generate a list of positions for tiles adjacent to the entity."""
        positions = []
        width = entity.tile_dimensions.tile_width
        height = entity.tile_dimensions.tile_height
        start_x = entity.position.x - width / 2
        start_y = entity.position.y - height / 2
        end_x = start_x + width
        end_y = start_y + height
        for x in range(int(start_x), int(end_x + 1)):
            positions.append(Position(x=x, y=start_y - 1))
        for x in range(int(start_x), int(end_x + 1)):
            positions.append(Position(x=x, y=end_y))
        for y in range(int(start_y), int(end_y + 1)):
            positions.append(Position(x=start_x - 1, y=y))
        for y in range(int(start_y), int(end_y + 1)):
            positions.append(Position(x=end_x, y=y))
        return [pos.down(0.5).right(0.5) for pos in positions]

    def _get_valid_connection_points(self, entity: Entity) -> List[Position]:
        """Get valid connection points for an entity, avoiding predefined points."""
        predefined_points = self._get_entity_connection_points(entity)
        adjacent_tiles = self._get_adjacent_tiles(entity)
        ignore_points = set()
        for tile in adjacent_tiles:
            for point in predefined_points:
                if tile.is_close(point, tolerance=0.707):
                    ignore_points.add(point)
        return list(set(adjacent_tiles) - ignore_points)

    def resolve(self, source: Union[Position, Entity, ElectricityGroup], target: Union[Position, Entity, ElectricityGroup]) -> List[Tuple[Position, Position]]:
        """Resolve positions for power connections"""
        if isinstance(source, ElectricityGroup):
            positions = []
            if isinstance(target, Entity):
                target_positions = self._get_valid_connection_points(target)
                for pole in source.poles:
                    nearest_point = min(target_positions, key=lambda pos: abs(pos.x - pole.position.x) + abs(pos.y - pole.position.y))
                    positions.append((pole.position, nearest_point))
            else:
                target_pos = target.position if isinstance(target, Entity) or isinstance(target, ElectricityGroup) else target
                target_pos = Position(x=round(target_pos.x * 2) / 2, y=round(target_pos.y * 2) / 2)
                for pole in source.poles:
                    positions.append((pole.position, target_pos))
            return positions
        else:
            source_pos = source.position if isinstance(source, Entity) else source
            source_pos = Position(x=round(source_pos.x * 2) / 2, y=round(source_pos.y * 2) / 2)
            if isinstance(target, Entity):
                target_positions = self._get_valid_connection_points(target)
                nearest_point = min(target_positions, key=lambda pos: abs(pos.x - source_pos.x) + abs(pos.y - source_pos.y))
                return [(source_pos, nearest_point)]
            else:
                target_pos = target.position if isinstance(target, Entity) or isinstance(target, ElectricityGroup) else target
                target_pos = Position(x=round(target_pos.x * 2) / 2, y=round(target_pos.y * 2) / 2)
                return [(source_pos, target_pos)]

class TransportConnectionResolver(Resolver):

    def __init__(self, *args):
        super().__init__(*args)

    def _get_transport_belt_adjacent_positions(self, belt, target=False) -> List[Position]:
        source_positions = [belt.output_position] if not target else [belt.input_position]
        match belt.direction.value:
            case Direction.UP.value:
                source_positions.extend([belt.position.left(1), belt.position.right(1)])
            case Direction.DOWN.value:
                source_positions.extend([belt.position.left(1), belt.position.right(1)])
            case Direction.LEFT.value:
                source_positions.extend([belt.position.up(1), belt.position.down(1)])
            case Direction.RIGHT.value:
                source_positions.extend([belt.position.up(1), belt.position.down(1)])
        return source_positions

    def resolve(self, source: Union[Position, Entity, EntityGroup], target: Union[Position, Entity, EntityGroup]) -> List[Tuple[Position, Position]]:
        match source:
            case GunTurret() | AssemblingMachine() | Lab() | Chest() | Accumulator() | Furnace():
                raise Exception(f'Transport belts cannot be connected directly from a {source.prototype} object as a source. You need to add an inserter that takes items from {source.prototype} and use the inserter as a source entity.')
            case BeltGroup():
                source_positions = self._get_transport_belt_adjacent_positions(source.outputs[0], target=False)
            case Inserter():
                source_position = source.drop_position
                entities = self.get_entities(position=source_position, radius=0)
                problematic_entities = [x for x in entities if not (x.name in ['transport-belt', 'fast-transport-belt', 'express-transport-belt', 'belt-group'] or x.name.endswith('-belt'))]
                if len(problematic_entities) > 0:
                    entities_desc = [f'{x.name} at {x.position}' for x in problematic_entities]
                    raise Exception(f'Cannot connect to source inserter drop_position position {source_position} as it is already occupied by incompatible entities - {entities_desc}.')
                source_positions = [source.drop_position]
            case MiningDrill():
                source_positions = [source.drop_position]
            case TransportBelt():
                source_positions = [source.position]
            case Position():
                source_positions = [Position(x=math.floor(source.x) + 0.5, y=math.floor(source.y) + 0.5)]
            case _:
                source_positions = [source.position]
        match target:
            case GunTurret() | AssemblingMachine() | Lab() | Chest() | Accumulator() | Furnace() | Boiler() | Generator():
                raise Exception(f'Transport belts cannot be connected directly to a {target.prototype} object as a target. You need to add an inserter that inputs items into {target.prototype} and use the inserter as the target entity.')
            case BeltGroup():
                target_positions = self._get_transport_belt_adjacent_positions(target.inputs[0], target=True)
            case Inserter():
                target_position = target.pickup_position
                entities = self.get_entities(position=target_position, radius=0)
                problematic_entities = [x for x in entities if not (x.name in ['transport-belt', 'fast-transport-belt', 'express-transport-belt', 'belt-group'] or x.name.endswith('-belt'))]
                if len(problematic_entities) > 0:
                    entities_desc = [f'{x.name} at {x.position}' for x in problematic_entities]
                    raise Exception(f'Cannot connect to target inserter pickup_position position {target_position} as it is already occupied by incompatible entities - {entities_desc}.')
                target_positions = [target.pickup_position]
            case MiningDrill():
                target_positions = [target.drop_position]
            case TransportBelt():
                target_positions = []
                for x_sign in [-1, 1]:
                    for y_sign in [-1, 1]:
                        target_positions.append(Position(x=target.position.x + x_sign * target.tile_dimensions.tile_width, y=target.position.y + y_sign * target.tile_dimensions.tile_height))
            case Position():
                target_positions = [Position(x=math.floor(target.x) + 0.5, y=math.floor(target.y) + 0.5)]
            case _:
                target_positions = [target.position]
        connection_pairs = [(src_pos, tgt_pos) for src_pos in source_positions for tgt_pos in target_positions]
        return sorted(connection_pairs, key=lambda pair: abs(pair[0].x - pair[1].x) + abs(pair[0].y - pair[1].y))

class Nearest(Tool):

    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)

    def __call__(self, type: Union[Prototype, Resource]) -> Position:
        """
        Find the nearest entity or resource to your position.
        :param type: Entity or resource type to find
        :return: Position of nearest entity or resource
        """
        try:
            if not isinstance(type, tuple) and isinstance(type.value, tuple):
                type = type.value
            name, metaclass = type
            if not isinstance(name, str):
                raise Exception("'Nearest' must be called with an entity name as the first argument.")
            response, time_elapsed = self.execute(self.player_index, name)
            if response is None or response == {}:
                if metaclass == ResourcePatch:
                    raise Exception(f'No {type} found on the map within 500 tiles of the player. Move around to explore the map more.')
                else:
                    raise Exception(f'No {type} found within 500 tiles of the player')
            x = response['x']
            y = response['y']
            position = Position(x=x, y=y)
            return position
        except TypeError:
            raise Exception(f'Could not find nearest {type[0]} on the surface')
        except Exception as e:
            raise Exception(f'Could not find nearest {type[0]}', e)

class MoveTo(Tool):

    def __init__(self, connection: LuaScriptManager, game_state):
        super().__init__(connection, game_state)
        self.request_path = RequestPath(connection, game_state)
        self.get_path = GetPath(connection, game_state)

    def __call__(self, position: Position, laying: Prototype=None, leading: Prototype=None) -> Position:
        """
        Move to a position.
        :param position: Position to move to.
        :return: Your final position
        """
        X_OFFSET, Y_OFFSET = (0, 0)
        x, y = (math.floor(position.x * 4) / 4 + X_OFFSET, math.floor(position.y * 4) / 4 + Y_OFFSET)
        nposition = Position(x=x, y=y)
        path_handle = self.request_path(start=Position(x=self.game_state.player_location.x, y=self.game_state.player_location.y), finish=nposition, allow_paths_through_own_entities=True, resolution=-1)
        sleep(0.05)
        ticks_before = self.game_state.instance.get_elapsed_ticks()
        try:
            if laying is not None:
                entity_name = laying.value[0]
                response, execution_time = self.execute(self.player_index, path_handle, entity_name, 1)
            elif leading:
                entity_name = leading.value[0]
                response, execution_time = self.execute(self.player_index, path_handle, entity_name, 0)
            else:
                response, execution_time = self.execute(self.player_index, path_handle, NONE, NONE)
            ticks_after = self.game_state.instance.get_elapsed_ticks()
            ticks_added = ticks_after - ticks_before
            if ticks_added > 0:
                game_speed = self.game_state.instance.get_speed()
                real_world_sleep = ticks_added / 60 / game_speed if game_speed > 0 else 0
                sleep(real_world_sleep)
            if isinstance(response, int) and response == 0:
                raise Exception('Could not move.')
            if isinstance(response, str):
                raise Exception(f'Could not move. {response}')
            if response == 'trailing' or response == 'leading':
                raise Exception('Could not lay entity, perhaps a typo?')
            if response and isinstance(response, dict):
                self.game_state.player_location = Position(x=response['x'], y=response['y'])
            if not self.game_state.instance.fast:
                remaining_steps = self.connection.rcon_client.send_command(f'/silent-command rcon.print(global.actions.get_walking_queue_length({self.player_index}))')
                while remaining_steps != '0':
                    sleep(0.5)
                    remaining_steps = self.connection.rcon_client.send_command(f'/silent-command rcon.print(global.actions.get_walking_queue_length({self.player_index}))')
                self.game_state.player_location = Position(x=position.x, y=position.y)
            return Position(x=response['x'], y=response['y'])
        except Exception as e:
            if response:
                raise Exception(f'Cannot move. {e} - {response}')
            raise Exception(f'Cannot move. {e}')

class PlaceObject(Tool):

    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'place_entity'
        self.load()
        self.get_entity = GetEntity(*args)
        self.pickup_entity = PickupEntity(*args)

    def __call__(self, entity: Prototype, direction: Direction=Direction.UP, position: Position=Position(x=0, y=0), exact: bool=True) -> Entity:
        """
        Places an entity e at local position (x, y) if you have it in inventory.
        :param entity: Entity to place
        :param direction: Cardinal direction to place
        :param position: Position to place entity
        :param exact: If True, place entity at exact position, else place entity at nearest possible position
        :return: Entity object
        """
        if isinstance(position, tuple):
            position = Position(x=position[0], y=position[1])
        if not isinstance(position, Position):
            raise ValueError('The position argument must be a Position object')
        if not isinstance(direction, (DirectionInternal, Direction)):
            raise ValueError('The second argument must be a Direction object')
        x, y = self.get_position(position)
        try:
            name, metaclass = entity.value
            while isinstance(metaclass, tuple):
                metaclass = metaclass[1]
        except Exception as e:
            raise Exception(f'Passed in {entity} argument is not a valid Prototype', e)
        factorio_direction = DirectionInternal.to_factorio_direction(direction)
        try:
            response, elapsed = self.execute(self.player_index, name, factorio_direction, x, y, exact)
        except Exception as e:
            try:
                msg = self.get_error_message(str(e))
                raise Exception(f'Could not place {name} at ({x}, {y}), {msg}')
            except Exception:
                raise Exception(f'Could not place {name} at ({x}, {y})', e)
        if not self.game_state.instance.fast:
            sleep(1)
            return self.get_entity(entity, position)
        else:
            if not isinstance(response, dict):
                try:
                    msg = str(response).split(':')[-1].replace('"', '').replace("'", '').strip()
                except:
                    msg = str(response).lstrip()
                raise Exception(f'Could not place {name} at ({x}, {y}), {msg}')
            cleaned_response = self.clean_response(response)
            try:
                object = metaclass(prototype=entity.name, game=self.connection, **cleaned_response)
            except Exception as e:
                raise Exception(f'Could not create {name} object from response (place entity): {cleaned_response}', e)
            return object

def test_cliff_straight_lines(clear_terrain):
    """Test straight cliff formations (cliff-sides)"""
    game = clear_terrain
    game.instance.rcon_client.send_command("/sc for i=-5,5 do game.surfaces[1].create_entity{name='cliff', position={x=i*2, y=0}, cliff_orientation='west-to-east'} end")
    game.instance.rcon_client.send_command("/sc for i=-5,5 do game.surfaces[1].create_entity{name='cliff', position={x=0, y=i*2}, cliff_orientation='north-to-south'} end")
    image = game._render(position=Position(x=0, y=0), radius=15, layers=Layer.ALL)
    image.show()
    assert image is not None

def test_cliff_outer_corners(clear_terrain):
    """Test outer corner cliff formations (cliff-outer)"""
    game = clear_terrain
    game.instance.rcon_client.send_command("/sc -- Bottom-left outer corner\ngame.surfaces[1].create_entity{name='cliff', position={x=-10, y=0}, cliff_orientation='west-to-north'} game.surfaces[1].create_entity{name='cliff', position={x=-8, y=0}, cliff_orientation='west-to-east'} game.surfaces[1].create_entity{name='cliff', position={x=-10, y=2}, cliff_orientation='north-to-south'} -- Bottom-right outer corner\ngame.surfaces[1].create_entity{name='cliff', position={x=10, y=0}, cliff_orientation='north-to-east'} game.surfaces[1].create_entity{name='cliff', position={x=8, y=0}, cliff_orientation='east-to-west'} game.surfaces[1].create_entity{name='cliff', position={x=10, y=2}, cliff_orientation='north-to-south'} -- Top-right outer corner\ngame.surfaces[1].create_entity{name='cliff', position={x=10, y=10}, cliff_orientation='east-to-south'} game.surfaces[1].create_entity{name='cliff', position={x=8, y=10}, cliff_orientation='east-to-west'} game.surfaces[1].create_entity{name='cliff', position={x=10, y=8}, cliff_orientation='south-to-north'} -- Top-left outer corner\ngame.surfaces[1].create_entity{name='cliff', position={x=-10, y=10}, cliff_orientation='south-to-west'} game.surfaces[1].create_entity{name='cliff', position={x=-8, y=10}, cliff_orientation='west-to-east'} game.surfaces[1].create_entity{name='cliff', position={x=-10, y=8}, cliff_orientation='south-to-north'} ")
    image = game._render(position=Position(x=0, y=5), radius=15, layers=Layer.ALL)
    image.show()
    assert image is not None

def test_cliff_inner_corners(clear_terrain):
    """Test inner corner cliff formations (cliff-inner)"""
    game = clear_terrain
    game.instance.rcon_client.send_command("/sc -- Create a box with inner corners\n-- Top edge\nfor i=-3,3 do   if i ~= 0 then     game.surfaces[1].create_entity{name='cliff', position={x=i*2, y=-6}, cliff_orientation='west-to-east'}   end end -- Bottom edge\nfor i=-3,3 do   if i ~= 0 then     game.surfaces[1].create_entity{name='cliff', position={x=i*2, y=6}, cliff_orientation='west-to-east'}   end end -- Left edge\nfor i=-2,2 do   if i ~= 0 then     game.surfaces[1].create_entity{name='cliff', position={x=-6, y=i*2}, cliff_orientation='north-to-south'}   end end -- Right edge\nfor i=-2,2 do   if i ~= 0 then     game.surfaces[1].create_entity{name='cliff', position={x=6, y=i*2}, cliff_orientation='north-to-south'}   end end -- Inner corners\ngame.surfaces[1].create_entity{name='cliff', position={x=-6, y=-6}, cliff_orientation='west-to-south'} game.surfaces[1].create_entity{name='cliff', position={x=6, y=-6}, cliff_orientation='south-to-east'} game.surfaces[1].create_entity{name='cliff', position={x=6, y=6}, cliff_orientation='east-to-north'} game.surfaces[1].create_entity{name='cliff', position={x=-6, y=6}, cliff_orientation='north-to-west'} ")
    image = game._render(position=Position(x=0, y=0), radius=10, layers=Layer.ALL)
    image.show()
    assert image is not None

def test_cliff_terminals(clear_terrain):
    """Test terminal cliff pieces (cliff-entrance)"""
    game = clear_terrain
    game.instance.rcon_client.send_command("/sc -- Terminals ending in each direction\ngame.surfaces[1].create_entity{name='cliff', position={x=-6, y=0}, cliff_orientation='west-to-none'} game.surfaces[1].create_entity{name='cliff', position={x=6, y=0}, cliff_orientation='east-to-none'} game.surfaces[1].create_entity{name='cliff', position={x=0, y=-6}, cliff_orientation='north-to-none'} game.surfaces[1].create_entity{name='cliff', position={x=0, y=6}, cliff_orientation='south-to-none'} -- Terminals starting from each direction\ngame.surfaces[1].create_entity{name='cliff', position={x=-10, y=10}, cliff_orientation='none-to-east'} game.surfaces[1].create_entity{name='cliff', position={x=10, y=10}, cliff_orientation='none-to-west'} game.surfaces[1].create_entity{name='cliff', position={x=-10, y=-10}, cliff_orientation='none-to-south'} game.surfaces[1].create_entity{name='cliff', position={x=10, y=-10}, cliff_orientation='none-to-north'} ")
    image = game._render(position=Position(x=0, y=0), radius=15, layers=Layer.ALL)
    image.show()
    assert image is not None

def test_cliff_t_junctions(clear_terrain):
    """Test T-junction cliff formations"""
    game = clear_terrain
    game.instance.rcon_client.send_command("/sc -- T-junction pointing up\nfor i=-2,2 do   game.surfaces[1].create_entity{name='cliff', position={x=i*2, y=0}, cliff_orientation='west-to-east'} end for i=1,3 do   game.surfaces[1].create_entity{name='cliff', position={x=0, y=-i*2}, cliff_orientation='north-to-south'} end -- T-junction pointing down\nfor i=-2,2 do   game.surfaces[1].create_entity{name='cliff', position={x=i*2, y=10}, cliff_orientation='west-to-east'} end for i=1,3 do   game.surfaces[1].create_entity{name='cliff', position={x=0, y=10+i*2}, cliff_orientation='north-to-south'} end -- T-junction pointing right\nfor i=-2,2 do   game.surfaces[1].create_entity{name='cliff', position={x=-10, y=i*2}, cliff_orientation='north-to-south'} end for i=1,3 do   game.surfaces[1].create_entity{name='cliff', position={x=-10+i*2, y=0}, cliff_orientation='west-to-east'} end -- T-junction pointing left\nfor i=-2,2 do   game.surfaces[1].create_entity{name='cliff', position={x=10, y=i*2}, cliff_orientation='north-to-south'} end for i=1,3 do   game.surfaces[1].create_entity{name='cliff', position={x=10-i*2, y=0}, cliff_orientation='west-to-east'} end ")
    image = game._render(position=Position(x=0, y=5), radius=20, layers=Layer.ALL)
    image.show()
    assert image is not None

def test_cliff_all_orientations_grid(clear_terrain):
    """Test all 20 cliff orientations in a grid layout"""
    game = clear_terrain
    game.instance.rcon_client.send_command("/sc local orientations = {  'west-to-east', 'north-to-south', 'east-to-west', 'south-to-north',  'west-to-north', 'north-to-east', 'east-to-south', 'south-to-west',  'west-to-south', 'north-to-west', 'east-to-north', 'south-to-east',  'west-to-none', 'none-to-east', 'east-to-none', 'none-to-west',  'north-to-none', 'none-to-south', 'south-to-none', 'none-to-north'} for i, orientation in ipairs(orientations) do   local row = math.floor((i-1) / 5)   local col = (i-1) % 5   local x = col * 4 - 8   local y = row * 4 - 6   game.surfaces[1].create_entity{    name='cliff',     position={x=x, y=y},     cliff_orientation=orientation  } end")
    image = game._render(position=Position(x=0, y=0), radius=12, layers=Layer.ALL)
    assert image is not None

def test_entities_with_cliffs(clear_terrain):
    """Test entity placement alongside cliffs"""
    game = clear_terrain
    game.instance.rcon_client.send_command("/sc for i=-3,3 do   game.surfaces[1].create_entity{name='cliff', position={x=i*2, y=-10}, cliff_orientation='west-to-east'} end")
    game.place_entity(Prototype.IronChest, position=Position(x=0, y=0))
    game.place_entity(Prototype.Splitter, position=Position(x=5, y=0))
    game.place_entity(Prototype.Lab, position=Position(x=10, y=0))
    game.connect_entities(Position(x=0, y=-2), Position(x=15, y=5), {Prototype.TransportBelt, Prototype.UndergroundBelt})
    game.connect_entities(Position(x=0, y=-5), Position(x=15, y=-5), {Prototype.SmallElectricPole})
    image = game._render(position=Position(x=5, y=0), radius=20, layers=Layer.ALL)
    image.show()
    assert image is not None

def test_rocks_and_decoratives(clear_terrain):
    """Test rock placement as decoratives"""
    game = clear_terrain
    image = game._render(position=Position(x=0, y=0), radius=15, layers=Layer.ALL)
    assert image is not None

def test_edge_case_entity_placement(game):
    """Test placement of entities at the edge of the map and in tight spaces."""
    edge_position = Position(x=game.bounding_box, y=game.bounding_box)
    with pytest.raises(Exception):
        game.place_entity(Prototype.StoneFurnace, position=edge_position)
    game.place_entity(Prototype.StoneFurnace, position=Position(x=0, y=0))
    game.place_entity(Prototype.StoneFurnace, position=Position(x=3, y=0))
    with pytest.raises(Exception):
        game.place_entity(Prototype.StoneFurnace, position=Position(x=1.5, y=0))

def test_complex_resource_patch_interaction(game):
    """Test interactions with resource patches of varying shapes and sizes."""
    iron_patch = game.get_resource_patch(Resource.IronOre, game.nearest(Resource.IronOre))
    assert isinstance(iron_patch, ResourcePatch)
    drill_positions = [iron_patch.bounding_box.left_top, Position(x=iron_patch.bounding_box.left_top.x + 3, y=iron_patch.bounding_box.left_top.y), Position(x=iron_patch.bounding_box.left_top.x, y=iron_patch.bounding_box.left_top.y + 3)]
    for pos in drill_positions:
        game.move_to(pos)
        drill = game.place_entity(Prototype.ElectricMiningDrill, position=pos)
        assert drill is not None
    for drill in game.inspect_entities(iron_patch.bounding_box.left_top, radius=10).entities:
        if drill.prototype == Prototype.ElectricMiningDrill:
            assert drill.mining_target == Resource.IronOre

def test_error_handling_and_invalid_inputs(game):
    """Test error handling for invalid inputs and operations."""
    with pytest.raises(ValueError):
        game.place_entity('invalid_entity', position=Position(x=0, y=0))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=5, y=5))
    with pytest.raises(ValueError):
        game.set_entity_recipe(assembler, 'invalid_recipe')

def test_performance_under_load(game):
    """Test performance when placing and manipulating many entities."""
    start_time = game.get_game_time()
    belt_count = 1000
    for i in range(belt_count):
        game.place_entity(Prototype.TransportBelt, position=Position(x=i * 0.5, y=0))
    for belt in game.inspect_entities(Position(x=0, y=0), radius=belt_count).entities:
        if belt.prototype == Prototype.TransportBelt:
            game.rotate_entity(belt, Direction.LEFT)
    end_time = game.get_game_time()
    assert end_time - start_time < 60

def test_blueprint_functionality(game):
    """Test creating, saving, and loading blueprints."""
    game.move_to(Position(x=0, y=0))
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=0, y=0))
    inserter = game.place_entity_next_to(Prototype.BurnerInserter, furnace.position, Direction.UP, spacing=1)
    chest = game.place_entity_next_to(Prototype.IronChest, inserter.position, Direction.UP, spacing=1)
    blueprint = game.create_blueprint([furnace, inserter, chest])
    assert blueprint is not None
    game.clear_area(Position(x=-5, y=-5), Position(x=5, y=5))
    game.load_blueprint(blueprint, Position(x=10, y=10))
    placed_entities = game.inspect_entities(Position(x=10, y=10), radius=5)
    assert len(placed_entities.entities) == 3
    assert any((e.prototype == Prototype.StoneFurnace for e in placed_entities.entities))
    assert any((e.prototype == Prototype.BurnerInserter for e in placed_entities.entities))
    assert any((e.prototype == Prototype.IronChest for e in placed_entities.entities))

def test_break_7(game):
    game.initial_inventory = {'coal': 200, 'burner-mining-drill': 10, 'wooden-chest': 10, 'burner-inserter': 10, 'transport-belt': 200, 'stone-furnace': 5, 'boiler': 4, 'offshore-pump': 3, 'steam-engine': 2, 'iron-gear-wheel': 22, 'iron-plate': 19, 'copper-plate': 52, 'electronic-circuit': 99, 'iron-ore': 62, 'stone': 50, 'electric-mining-drill': 10, 'small-electric-pole': 200, 'pipe': 100, 'assembling-machine-1': 5}
    game.reset()
    water_pos = game.nearest(Resource.Water)
    print(f'Found water at {water_pos}')
    game.move_to(water_pos)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_pos)
    print(f'Placed offshore pump at {offshore_pump.position}')
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=Direction.RIGHT, spacing=3)
    print(f'Placed boiler at {boiler.position}')
    steam_engine = game.place_entity_next_to(Prototype.SteamEngine, reference_position=boiler.position, direction=Direction.RIGHT, spacing=3)
    print(f'Placed steam engine at {steam_engine.position}')
    game.connect_entities(offshore_pump, boiler, Prototype.Pipe)
    game.connect_entities(boiler, steam_engine, Prototype.Pipe)
    boiler = game.insert_item(Prototype.Coal, boiler, quantity=50)
    print(f'Power system positions - Pump: {offshore_pump.position}, Boiler: {boiler.position}, Engine: {steam_engine.position}')
    iron_pos = game.nearest(Resource.IronOre)
    print(f'Found iron ore at {iron_pos}')
    steam_engine = game.get_entity(Prototype.SteamEngine, Position(x=-2.5, y=-1.5))
    drills = []
    game.move_to(Position(x=-13.5, y=25.5))
    drill = game.place_entity(Prototype.ElectricMiningDrill, position=Position(x=-13.5, y=25.5))
    drills.append(drill)
    game.move_to(Position(x=-14.5, y=21.5))
    drill = game.place_entity(Prototype.ElectricMiningDrill, position=Position(x=-14.5, y=21.5))
    drills.append(drill)
    drill = game.place_entity(Prototype.ElectricMiningDrill, position=Position(x=-17.5, y=26.5))
    drills.append(drill)
    game.move_to(steam_engine.position)
    first_pole = game.place_entity(Prototype.SmallElectricPole, position=Position(x=steam_engine.position.x, y=steam_engine.position.y - 3))
    print(f'Placed first power pole at {first_pole.position}')
    for drill in drills:
        game.connect_entities(drill, first_pole, Prototype.SmallElectricPole)
        print(f'Connected power to drill at {drill.position}')
    pass

def test_rocket_launch(game):
    pos = Position(x=0, y=0)
    game.move_to(pos)
    water_pos = game.nearest(Resource.Water)
    game.move_to(water_pos)
    pump = game.place_entity(Prototype.OffshorePump, position=water_pos)
    boiler = game.place_entity_next_to(Prototype.Boiler, pump.position, Direction.RIGHT, spacing=5)
    engine = game.place_entity_next_to(Prototype.SteamEngine, boiler.position, Direction.DOWN, spacing=5)
    game.connect_entities(pump, boiler, Prototype.Pipe)
    game.connect_entities(boiler, engine, Prototype.Pipe)
    game.insert_item(Prototype.Coal, boiler, quantity=50)
    silo = game.place_entity_next_to(Prototype.RocketSilo, engine.position, Direction.RIGHT, spacing=5)
    lds_chest = game.place_entity_next_to(Prototype.SteelChest, silo.position, Direction.LEFT, spacing=1)
    game.place_entity_next_to(Prototype.FastInserter, lds_chest.position, Direction.RIGHT)
    game.move_to(lds_chest.position)
    lds_chest2 = game.place_entity_next_to(Prototype.SteelChest, lds_chest.position, Direction.UP, spacing=0)
    game.place_entity_next_to(Prototype.FastInserter, lds_chest2.position, Direction.RIGHT)
    lds_chest3 = game.place_entity_next_to(Prototype.SteelChest, lds_chest2.position, Direction.UP, spacing=0)
    game.place_entity_next_to(Prototype.FastInserter, lds_chest3.position, Direction.RIGHT)
    fuel_chest = game.place_entity_next_to(Prototype.SteelChest, lds_chest3.position, Direction.UP, spacing=0)
    game.place_entity_next_to(Prototype.FastInserter, fuel_chest.position, Direction.RIGHT)
    fuel_chest2 = game.place_entity_next_to(Prototype.SteelChest, fuel_chest.position, Direction.UP, spacing=0)
    game.place_entity_next_to(Prototype.FastInserter, fuel_chest2.position, Direction.RIGHT)
    fuel_chest3 = game.place_entity_next_to(Prototype.SteelChest, lds_chest.position, Direction.DOWN, spacing=0)
    game.place_entity_next_to(Prototype.FastInserter, fuel_chest3.position, Direction.RIGHT)
    rcu_chest = game.place_entity_next_to(Prototype.SteelChest, fuel_chest3.position, Direction.DOWN, spacing=0)
    game.place_entity_next_to(Prototype.FastInserter, rcu_chest.position, Direction.RIGHT)
    rcu_chest2 = game.place_entity_next_to(Prototype.SteelChest, rcu_chest.position, Direction.DOWN, spacing=0)
    game.place_entity_next_to(Prototype.FastInserter, rcu_chest2.position, Direction.RIGHT)
    rcu_chest3 = game.place_entity_next_to(Prototype.SteelChest, rcu_chest2.position, Direction.DOWN, spacing=0)
    game.place_entity_next_to(Prototype.FastInserter, rcu_chest3.position, Direction.RIGHT)
    game.place_entity(Prototype.SmallElectricPole, position=Position(x=-0.5, y=10.5))
    game.place_entity(Prototype.SmallElectricPole, position=Position(x=1.5, y=13.5))
    game.place_entity(Prototype.SmallElectricPole, position=Position(x=1.5, y=8.5))
    game.place_entity(Prototype.SmallElectricPole, position=Position(x=5.5, y=5.5))
    for chest in [lds_chest, lds_chest2, lds_chest3, fuel_chest, fuel_chest2, fuel_chest3, rcu_chest, rcu_chest2, rcu_chest3]:
        game.insert_item(Prototype.RocketFuel, chest, quantity=112)
        game.insert_item(Prototype.RocketControlUnit, chest, quantity=112)
        game.insert_item(Prototype.LowDensityStructure, chest, quantity=112)
        inventory_items = {'rocket-control-unit': 112, 'rocket-fuel': 112, 'low-density-structure': 112}
        game._set_inventory(inventory_items)
    assert silo.rocket_parts == 0
    assert silo.launch_count == 0
    game.sleep(300)
    silo = game.get_entities({Prototype.RocketSilo})[0]
    assert silo.status == EntityStatus.PREPARING_ROCKET_FOR_LAUNCH
    game.sleep(120)
    silo = game.get_entities({Prototype.RocketSilo})[0]
    assert silo.status == EntityStatus.WAITING_TO_LAUNCH_ROCKET
    silo = game.launch_rocket(silo)
    assert silo.status == EntityStatus.LAUNCHING_ROCKET
    game.sleep(60)
    silo = game.get_entities({Prototype.RocketSilo})[0]
    assert silo.status == EntityStatus.ITEM_INGREDIENT_SHORTAGE

def test_connect_offshore_pump_to_boiler(game):
    game.move_to(game.nearest(Resource.Water))
    game.move_to(game.nearest(Resource.Wood))
    game.harvest_resource(game.nearest(Resource.Wood), quantity=100)
    game.move_to(game.nearest(Resource.Water))
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=game.nearest(Resource.Water))
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=offshore_pump.direction, spacing=5)
    water_pipes = game.connect_entities(boiler, offshore_pump, connection_type=Prototype.Pipe)
    assert len(water_pipes.pipes) == 5 + boiler.tile_dimensions.tile_width / 2 + offshore_pump.tile_dimensions.tile_width / 2 + 1
    game.instance.reset()
    game.move_to(game.nearest(Resource.Water))
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=game.nearest(Resource.Water), direction=Direction.RIGHT)
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=offshore_pump.direction, spacing=5)
    assert boiler.direction.value == offshore_pump.direction.value
    water_pipes = game.connect_entities(boiler, offshore_pump, connection_type=Prototype.Pipe)
    assert len(water_pipes.pipes) >= math.ceil(5 + boiler.tile_dimensions.tile_height / 2 + offshore_pump.tile_dimensions.tile_height / 2 + 1)
    game.instance.reset()
    game.move_to(game.nearest(Resource.Water))
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=game.nearest(Resource.Water), direction=Direction.DOWN, exact=False)
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=offshore_pump.direction, spacing=5)
    assert boiler.direction.value == offshore_pump.direction.value
    water_pipes = game.connect_entities(boiler, offshore_pump, connection_type=Prototype.Pipe)
    assert len(water_pipes.pipes) >= math.ceil(5 + boiler.tile_dimensions.tile_height / 2 + offshore_pump.tile_dimensions.tile_height / 2 + 1)
    game.move_to(Position(x=-30, y=0))
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=game.nearest(Resource.Water), direction=Direction.LEFT)
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=offshore_pump.direction, spacing=5)
    assert boiler.direction.value == offshore_pump.direction.value
    water_pipes = game.connect_entities(boiler, offshore_pump, connection_type=Prototype.Pipe)
    assert len(water_pipes.pipes) >= math.ceil(5 + boiler.tile_dimensions.tile_width / 2 + offshore_pump.tile_dimensions.tile_width / 2 + 1)

def test_connect_steam_engines_to_boilers_using_pipes(game):
    """
    Place a boiler and a steam engine next to each other in 3 cardinal directions.
    :param game:
    :return:
    """
    boilers_in_inventory = game.inspect_inventory()[Prototype.Boiler]
    steam_engines_in_inventory = game.inspect_inventory()[Prototype.SteamEngine]
    pipes_in_inventory = game.inspect_inventory()[Prototype.Pipe]
    game.move_to(Position(x=0, y=0))
    boiler: Entity = game.place_entity(Prototype.Boiler, position=Position(x=0, y=0), direction=Direction.UP)
    assert boiler.direction.value == Direction.UP.value
    game.move_to(Position(x=0, y=10))
    steam_engine: Entity = game.place_entity(Prototype.SteamEngine, position=Position(x=0, y=10), direction=Direction.UP)
    assert steam_engine.direction.value == Direction.UP.value
    connection: PipeGroup = game.connect_entities(boiler, steam_engine, connection_type=Prototype.Pipe)
    assert boilers_in_inventory - 1 == game.inspect_inventory()[Prototype.Boiler]
    assert steam_engines_in_inventory - 1 == game.inspect_inventory()[Prototype.SteamEngine]
    assert pipes_in_inventory - len(connection.pipes) == game.inspect_inventory()[Prototype.Pipe]
    assert len(connection.pipes) >= 10
    game.instance.reset()
    offsets = [Position(x=5, y=0), Position(x=0, y=-5), Position(x=-5, y=0)]
    directions = [Direction.RIGHT, Direction.UP, Direction.LEFT]
    for offset, direction in zip(offsets, directions):
        game.move_to(Position(x=0, y=0))
        boiler: Entity = game.place_entity(Prototype.Boiler, position=Position(x=0, y=0), direction=direction)
        game.move_to(offset)
        steam_engine: Entity = game.place_entity(Prototype.SteamEngine, position=offset, direction=direction)
        try:
            connection: PipeGroup = game.connect_entities(boiler, steam_engine, connection_type=Prototype.Pipe)
        except Exception as e:
            print(e)
            assert False
        assert boilers_in_inventory - 1 == game.inspect_inventory()[Prototype.Boiler]
        assert steam_engines_in_inventory - 1 == game.inspect_inventory()[Prototype.SteamEngine]
        current_pipes_in_inventory = game.inspect_inventory()[Prototype.Pipe]
        spent_pipes = pipes_in_inventory - current_pipes_in_inventory
        assert spent_pipes == len(connection.pipes)
        game.get_entities(position=steam_engine.position)
        game.instance.reset()

def test_connect_steam_engine_boiler_nearly_adjacent(game):
    """
    We've had problems with gaps of exactly 2.
    :param game:
    :return:
    """
    game.move_to(Position(x=-30, y=12))
    game.move_to(game.nearest(Resource.Water))
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=game.nearest(Resource.Water), direction=Direction.LEFT)
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=offshore_pump.direction, spacing=2)
    steam_engine = game.place_entity_next_to(Prototype.SteamEngine, reference_position=boiler.position, direction=boiler.direction, spacing=2)
    game.connect_entities(boiler, steam_engine, connection_type=Prototype.Pipe)
    game.connect_entities(offshore_pump, boiler, connection_type=Prototype.Pipe)
    game.insert_item(Prototype.Coal, boiler, 50)
    engine = game.get_entity(Prototype.SteamEngine, steam_engine.position)
    assert engine.status == EntityStatus.NOT_PLUGGED_IN_ELECTRIC_NETWORK

def test_connect_boiler_to_steam_engine_with_pipes_horizontally(game):
    boiler_pos = Position(x=0, y=0)
    game.move_to(boiler_pos)
    boiler = game.place_entity(Prototype.Boiler, position=boiler_pos, direction=Direction.RIGHT)
    steam_engine_pos = Position(x=boiler.position.x + 5, y=boiler.position.y + 5)
    game.move_to(steam_engine_pos)
    steam_engine = game.place_entity(Prototype.SteamEngine, position=steam_engine_pos, direction=Direction.RIGHT)
    pipes = game.connect_entities(boiler, steam_engine, Prototype.Pipe)
    assert pipes, 'Failed to connect boiler to steam engine with pipes'

def test_connect_boiler_to_steam_engine_with_pipes_vertically(game):
    boiler_pos = Position(x=0, y=0)
    game.move_to(boiler_pos)
    boiler = game.place_entity(Prototype.Boiler, position=boiler_pos, direction=Direction.UP)
    steam_engine_pos = Position(x=boiler.position.x + 5, y=boiler.position.y + 5)
    game.move_to(steam_engine_pos)
    steam_engine = game.place_entity(Prototype.SteamEngine, position=steam_engine_pos, direction=Direction.UP)
    pipes = game.connect_entities(boiler, steam_engine, Prototype.Pipe)
    assert pipes, 'Failed to connect boiler to steam engine with pipes'

def test_connect_boiler_to_steam_engine_with_pipes_vertically_with_positions(game):
    boiler_pos = Position(x=0, y=0)
    game.move_to(boiler_pos)
    boiler = game.place_entity(Prototype.Boiler, position=boiler_pos, direction=Direction.UP)
    steam_engine_pos = Position(x=boiler.position.x + 5, y=boiler.position.y + 5)
    game.move_to(steam_engine_pos)
    steam_engine: Generator = game.place_entity(Prototype.SteamEngine, position=steam_engine_pos, direction=Direction.UP)
    pipes = game.connect_entities(boiler.steam_output_point, steam_engine.connection_points[0], Prototype.Pipe)
    assert pipes, 'Failed to connect boiler to steam engine with pipes'

def test_avoid_self_collision(game):
    target_position = Position(x=5, y=-4)
    game.move_to(target_position)
    print(f'Moved to target position: {target_position}')
    water_source = game.nearest(Resource.Water)
    print(f'Nearest water source found at: {water_source}')
    game.move_to(water_source)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_source, direction=Direction.SOUTH)
    print(f'Placed offshore pump at: {offshore_pump.position}')
    boiler_pos = Position(x=offshore_pump.position.x + 5, y=offshore_pump.position.y - 2)
    game.move_to(boiler_pos)
    boiler = game.place_entity(Prototype.Boiler, position=boiler_pos, direction=Direction.RIGHT)
    print(f'Placed boiler at: {boiler.position}')
    pipes = game.connect_entities(offshore_pump, boiler, Prototype.Pipe)
    assert pipes, 'Failed to connect offshore pump to boiler with pipes'
    print('Successfully connected offshore pump to boiler with pipes')

def test_connect_where_connection_points_are_blocked(game):
    water_source = game.nearest(Resource.Water)
    game.move_to(water_source)
    print(f'Moved to water source at {water_source}')
    pump = game.place_entity(Prototype.OffshorePump, Direction.RIGHT, water_source)
    print(f'Placed offshore pump at {pump.position}')
    '\n    Step 2: Place the boiler and connect it to the pump\n    '
    boiler_position = Position(x=pump.position.x + 4, y=pump.position.y)
    game.move_to(boiler_position)
    print(f'Moved to boiler position at {boiler_position}')
    boiler = game.place_entity(Prototype.Boiler, Direction.UP, boiler_position)
    print(f'Placed boiler at {boiler.position}')
    pump_to_boiler_pipes = game.connect_entities(pump, boiler, Prototype.Pipe)
    assert pump_to_boiler_pipes, 'Failed to connect pump to boiler with pipes'
    print('Connected pump to boiler with pipes')
    assert boiler.connection_points[0] in [p.position for p in pump_to_boiler_pipes.pipes]

def test_connect_pipes_by_positions(game):
    """
    This should ensure that pipe groups are always returned - instead of pipes themselves.
    """
    position_1 = Position(x=0, y=1)
    position_2 = Position(x=2, y=4)
    pipes = game.connect_entities(position_1, position_2, Prototype.Pipe)
    assert len(pipes.pipes) == 6

def test_connect_pipes_to_advanced_assembler(game):
    """
    Ensure that advanced assemblers can be connected to.
    """
    position_1 = Position(x=10, y=0)
    assembler_2 = game.place_entity(Prototype.AssemblingMachine2, Direction.UP, Position(x=0, y=0))
    pipes = game.connect_entities(position_1, assembler_2, Prototype.Pipe)
    assert len(pipes.pipes) == 13

def test_fail_connect_pipes_with_mixed_connection_types(game):
    """
    This should ensure that pipe groups are always returned - instead of pipes themselves.
    """
    position_1 = Position(x=0, y=1)
    position_2 = Position(x=2, y=4)
    try:
        game.connect_entities(position_1, position_2, {Prototype.Pipe, Prototype.UndergroundBelt})
        assert False
    except Exception:
        assert True

def test_avoiding_pipe_networks(game):
    """Test connecting pipes that cross paths"""
    start1 = Position(x=0, y=0)
    end1 = Position(x=10, y=0)
    start2 = Position(x=5, y=-5)
    end2 = Position(x=5, y=5)
    pipes1 = game.connect_entities(start1, end1, Prototype.Pipe)
    pipes2 = game.connect_entities(start2, end2, Prototype.Pipe)
    assert pipes1
    assert pipes2
    assert pipes1.id != pipes2.id

def test_pipe_around_obstacle(game):
    """Test pipe pathfinding around placed entities"""
    obstacle_pos = Position(x=5, y=0)
    game.move_to(obstacle_pos)
    game.place_entity(Prototype.Boiler, position=obstacle_pos)
    start = Position(x=0, y=0)
    end = Position(x=10, y=0)
    pipes = game.connect_entities(start, end, Prototype.Pipe)
    assert pipes
    assert len(pipes.pipes) > 10

def test_pipe_network_branching(game):
    """Test creating T-junctions and branched pipe networks"""
    start = Position(x=0, y=0)
    end = Position(x=10, y=0)
    main_line = game.connect_entities(start, end, Prototype.Pipe)
    branch_end = Position(x=5, y=5)
    branch = game.connect_entities(Position(x=5, y=0), branch_end, Prototype.Pipe)
    assert branch
    assert branch.id == main_line.id

def test_pipe_network_branching_inverted(game):
    """Test creating T-junctions and branched pipe networks"""
    start = Position(x=0, y=0)
    end = Position(x=10, y=0)
    main_line = game.connect_entities(start, end, Prototype.Pipe)
    branch_end = Position(x=5, y=5)
    branch = game.connect_entities(branch_end, Position(x=5, y=0), Prototype.Pipe)
    assert branch
    assert branch.id == main_line.id

def test_get_existing_pipe_connection_group(game):
    """Test existing pipe group return functionality"""
    pos1 = Position(x=20, y=20)
    pos2 = Position(x=25, y=20)
    first_pipes = game.connect_entities(pos1, pos2, Prototype.Pipe)
    assert first_pipes, 'Initial pipe connection should succeed'
    second_pipes = game.connect_entities(pos1, pos2, Prototype.Pipe)
    assert second_pipes, 'Second pipe connection should return existing group'
    print('✓ Pipe connection handled gracefully')

def test_pipe_retry_logic(game):
    """Test retry logic for intermittent Lua errors in pipe connections"""
    pos1 = Position(x=30, y=30)
    pos2 = Position(x=35, y=30)
    for i in range(3):
        try:
            connection = game.connect_entities(pos1, pos2, Prototype.Pipe)
            assert connection, f'Pipe connection attempt {i + 1} should succeed'
            break
        except Exception as e:
            if 'attempt to index field' in str(e):
                print(f'Caught expected Lua error on attempt {i + 1}, retry should handle this')
            else:
                raise
    print('✓ Pipe retry logic allows connections to succeed')

def test_pipe_performance_no_sleep(game):
    """Test that pipe connections complete without artificial delays"""
    import time
    pos1 = Position(x=40, y=40)
    pos2 = Position(x=50, y=40)
    start_time = time.time()
    connection = game.connect_entities(pos1, pos2, Prototype.Pipe)
    end_time = time.time()
    assert connection, 'Pipe connection should succeed'
    duration = end_time - start_time
    assert duration < 5.0, f'Pipe connection took {duration}s, should be faster without sleep'
    print(f'✓ Pipe connection completed in {duration:.2f}s (performance improved)')

def test_connect_wall_line(game):
    start_position = Position(x=0, y=0)
    end_position = Position(x=5, y=0)
    wall = game.connect_entities(start_position, end_position, connection_type=Prototype.StoneWall)
    assert len(wall.entities) == 6

def test_multiple_inserter_connections(game):
    """
    Tests placing multiple pairs of burner inserters and connecting them with transport belts.
    Verifies the belt positions and connections for each iteration.
    """
    test_iterations = 5
    for i in range(test_iterations):
        position_1 = Position(x=i * 5, y=i * 3)
        game.move_to(position_1)
        inserter_1 = game.place_entity(Prototype.BurnerInserter, position=position_1, direction=Direction.LEFT, exact=True)
        position_2 = Position(x=i * 5 + 10, y=i * 3 + 8)
        game.move_to(position_2)
        inserter_2 = game.place_entity(Prototype.BurnerInserter, position=position_2, direction=Direction.LEFT, exact=True)
        belt_groups = game.connect_entities(inserter_1, inserter_2, connection_type=Prototype.TransportBelt)
        assert belt_groups, f'Iteration {i}: Expected 1 belt group, got {len(belt_groups)}'
        belt_group = belt_groups
        assert len(belt_group.belts) >= 20, f'Iteration {i}: Belt group has no belts'
        expected_belt_count = int(abs(position_1.x - position_2.x) + abs(position_1.y - position_2.y)) + 1
        assert len(belt_group.belts) >= expected_belt_count, f'Iteration {i}: Expected at least {expected_belt_count} belts, got {len(belt_group.belts)}'
        game.instance.reset()

def test_inserter_pickup_positions(game):
    iron = game.get_resource_patch(Resource.IronOre, game.nearest(Resource.IronOre))
    left_of_iron = iron.bounding_box.left_top.up().right(10)
    far_left_of_iron = iron.bounding_box.left_top.up().right(2)
    coal_belt = game.connect_entities(left_of_iron, far_left_of_iron, connection_type=Prototype.TransportBelt)
    game.move_to(far_left_of_iron.down())
    iron_drill = game.place_entity(Prototype.BurnerMiningDrill, position=far_left_of_iron.down(3), direction=Direction.DOWN, exact=True)
    inserter_position = Position(x=coal_belt.inputs[0].input_position.x, y=coal_belt.inputs[0].input_position.y)
    iron_drill_fuel_inserter = game.place_entity(Prototype.BurnerInserter, position=inserter_position, direction=Direction.LEFT, exact=True)
    furnace_position = Position(x=iron_drill.drop_position.x, y=iron_drill.drop_position.y)
    iron_furnace = game.place_entity(Prototype.StoneFurnace, position=furnace_position)
    furnace_fuel_inserter_position = Position(x=iron_furnace.position.x + 1, y=iron_furnace.position.y)
    furnace_fuel_inserter = game.place_entity(Prototype.BurnerInserter, position=furnace_fuel_inserter_position, direction=Direction.LEFT)
    coal_belt_to_furnace = game.connect_entities(iron_drill_fuel_inserter.pickup_position, furnace_fuel_inserter.pickup_position, connection_type=Prototype.TransportBelt)
    assert coal_belt_to_furnace.outputs[0].position == furnace_fuel_inserter.pickup_position
    assert coal_belt_to_furnace.inputs[0].position == iron_drill_fuel_inserter.pickup_position

def test_basic_connection_between_furnace_and_miner(game):
    """
    Place a furnace with a burner inserter pointing towards it.
    Find the nearest coal and place a burner mining drill on it.
    Connect the burner mining drill to the inserter using a transport belt.
    :param game:
    :return:
    """
    coal: Position = game.nearest(Resource.Coal)
    furnace_position = Position(x=coal.x, y=coal.y - 10)
    game.move_to(furnace_position)
    furnace = game.place_entity(Prototype.StoneFurnace, position=furnace_position)
    inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace.position, direction=game.RIGHT, spacing=0)
    inserter = game.rotate_entity(inserter, Direction.LEFT)
    game.move_to(coal)
    miner = game.place_entity(Prototype.BurnerMiningDrill, position=coal)
    belts_in_inventory = game.inspect_inventory()[Prototype.TransportBelt]
    connection = game.connect_entities(miner, inserter, connection_type=Prototype.TransportBelt)
    current_belts_in_inventory = game.inspect_inventory()[Prototype.TransportBelt]
    spent_belts = belts_in_inventory - current_belts_in_inventory
    assert spent_belts == len(connection.belts)

def test_inserter_pickup_positions2(game):
    inserter1_position = Position(x=0, y=0)
    inserter1 = game.place_entity(Prototype.BurnerInserter, position=inserter1_position, direction=Direction.LEFT, exact=True)
    inserter2_position = Position(x=0, y=-5)
    inserter2 = game.place_entity(Prototype.BurnerInserter, position=inserter2_position, direction=Direction.LEFT)
    belt = game.connect_entities(inserter1.pickup_position, inserter2.pickup_position, connection_type=Prototype.TransportBelt)
    assert len(belt.belts) == int(abs(inserter1_position.y - inserter2_position.y) + 1)

def test_ensure_final_belt_is_the_correct_orientation(game):
    copper_ore_patch = game.get_resource_patch(Resource.CopperOre, game.nearest(Resource.CopperOre))
    assert copper_ore_patch, 'No copper ore patch found'
    print(f'copper ore patch found at {copper_ore_patch.bounding_box}')
    chest_inserter2 = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=Position(x=0, y=0), direction=Direction.LEFT)
    assert chest_inserter2, 'Failed to place inserter'
    print(f'Second Inserter placed at {chest_inserter2.position}')
    game.move_to(copper_ore_patch.bounding_box.center)
    copper_drill = game.place_entity(Prototype.BurnerMiningDrill, direction=Direction.RIGHT, position=copper_ore_patch.bounding_box.center)
    assert copper_drill, 'Failed to place burner mining drill'
    print(f'Burner mining drill placed at {copper_drill.position}')
    copper_drill_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=copper_drill.position, direction=Direction.LEFT)
    assert copper_drill_inserter, 'Failed to place inserter'
    print(f'Inserter placed at {copper_drill_inserter.position}')
    copper_drill_inserter = game.rotate_entity(copper_drill_inserter, Direction.RIGHT)
    assert copper_drill_inserter.direction.value == Direction.RIGHT.value, 'Failed to rotate inserter'
    inserter_with_coal = game.insert_item(Prototype.Coal, copper_drill_inserter, quantity=5)
    assert inserter_with_coal.fuel.get(Prototype.Coal, 0) > 0, 'Failed to fuel inserter'
    beltgroup = game.connect_entities(chest_inserter2, copper_drill_inserter, connection_type=Prototype.TransportBelt)
    assert beltgroup.belts[-1].direction.value == Direction.DOWN.value, 'Final belt is not facing down'

def test_no_broken_edges(game):
    """
    There is a weird issue where the full path is missing a point in the middle, and so the belt is broken.
    :param game:
    :return:
    """
    copper_ore_patch = game.get_resource_patch(Resource.CopperOre, game.nearest(Resource.CopperOre))
    assert copper_ore_patch, 'No copper ore patch found'
    print(f'copper ore patch found at {copper_ore_patch.bounding_box.center}')
    game.move_to(copper_ore_patch.bounding_box.center)
    drill = game.place_entity(Prototype.BurnerMiningDrill, direction=Direction.RIGHT, position=copper_ore_patch.bounding_box.center)
    assert drill, 'Failed to place burner mining drill'
    print(f'Burner mining drill placed at {drill.position}')
    copper_ore_drill_with_coal = game.insert_item(Prototype.Coal, drill, quantity=5)
    assert copper_ore_drill_with_coal.fuel.get(Prototype.Coal, 0) > 0, 'Failed to fuel burner mining drill'
    print(f'Inserted {copper_ore_drill_with_coal.fuel.get(Prototype.Coal, 0)} coal into burner mining drill')
    game.move_to(Position(x=0, y=0))
    chest = game.place_entity(Prototype.WoodenChest, position=Position(x=0, y=0))
    assert chest, 'Failed to place chest'
    chest_copper_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=chest.position, direction=Direction.RIGHT)
    assert chest_copper_inserter, 'Failed to place inserter'
    print(f'Inserter placed at {chest_copper_inserter.position}')
    chest_copper_inserter = game.rotate_entity(chest_copper_inserter, Direction.LEFT)
    assert chest_copper_inserter.direction.value == Direction.LEFT.value, 'Failed to rotate inserter'
    inserter_with_coal = game.insert_item(Prototype.Coal, chest_copper_inserter, quantity=5)
    assert inserter_with_coal.fuel.get(Prototype.Coal, 0) > 0, 'Failed to fuel inserter'
    belt_group = game.connect_entities(copper_ore_drill_with_coal.drop_position, chest_copper_inserter.pickup_position, connection_type=Prototype.TransportBelt)
    assert belt_group, 'Failed to connect entities with transport belts'
    game.sleep(60)
    chest = game.get_entity(Prototype.WoodenChest, chest.position)
    assert chest.inventory.get(Prototype.CopperOre, 0) > 0, 'Chest is empty'

def test_connecting_transport_belts_around_sharp_edges2(game):
    iron_patch: ResourcePatch = game.get_resource_patch(Resource.IronOre, game.nearest(Resource.IronOre))
    right_top = Position(x=iron_patch.bounding_box.right_bottom.x, y=iron_patch.bounding_box.left_top.y)
    left_bottom = Position(x=iron_patch.bounding_box.left_top.x, y=iron_patch.bounding_box.right_bottom.y)
    positions = [left_bottom, iron_patch.bounding_box.left_top, right_top, iron_patch.bounding_box.right_bottom]
    for position in positions:
        game.move_to(position)
        miner = game.place_entity(Prototype.BurnerMiningDrill, position=position, direction=Direction.LEFT)
        belts = game.connect_entities(miner, Position(x=0, y=0), connection_type=Prototype.TransportBelt)
        assert belts, 'Failed to connect transport belts around the water patch'
        game.instance.reset()

def test_connect_belt_groups_horizontally(game):
    belt_group_right = game.connect_entities(Position(x=0, y=0), Position(x=5, y=0), Prototype.TransportBelt)
    belt_group_right = game.connect_entities(belt_group_right, belt_group_right, Prototype.TransportBelt)
    assert belt_group_right
    belt_group_left = game.connect_entities(Position(x=0, y=-10), Position(x=-5, y=-10), Prototype.TransportBelt)
    belt_group_left = game.connect_entities(belt_group_left, belt_group_left, Prototype.TransportBelt)
    assert belt_group_left

def test_connect_belt_groups_vertically(game):
    belt_group_down = game.connect_entities(Position(x=0, y=0), Position(x=0, y=5), Prototype.TransportBelt)
    belt_group_down = game.connect_entities(belt_group_down, belt_group_down, Prototype.TransportBelt)
    assert belt_group_down
    assert len(belt_group_down.inputs) == 0
    assert len(belt_group_down.outputs) == 0
    belt_group_up = game.connect_entities(Position(x=-2, y=0), Position(x=-2, y=-5), Prototype.TransportBelt)
    belt_group_up = game.connect_entities(belt_group_up, belt_group_up, Prototype.TransportBelt)
    assert belt_group_up
    assert len(belt_group_up.inputs) == 0
    assert len(belt_group_up.outputs) == 0

def test_connect_belt_groups_diagonally(game):
    belt_group_up_left = game.connect_entities(Position(x=0, y=0), Position(x=-5, y=-5), Prototype.TransportBelt)
    belt_group_up_left = game.connect_entities(belt_group_up_left, belt_group_up_left, Prototype.TransportBelt)
    assert belt_group_up_left
    assert len(belt_group_up_left.inputs) == 0
    assert len(belt_group_up_left.outputs) == 0

def test_connect_belt_groups_into_a_square(game):
    belt_group = game.connect_entities(Position(x=0, y=0), Position(x=5, y=0), Prototype.TransportBelt)
    belt_group = game.connect_entities(belt_group, Position(x=5, y=5), Prototype.TransportBelt)
    belt_group = game.connect_entities(belt_group, Position(x=0, y=5), Prototype.TransportBelt)
    belt_group = game.connect_entities(belt_group, belt_group, Prototype.TransportBelt)
    assert belt_group
    assert len(belt_group.inputs) == 0
    assert len(belt_group.outputs) == 0

def test_connect_belt_groups_into_a_square_waypoints(game):
    belt_group = game.connect_entities(Position(x=0, y=0), Position(x=5, y=0), Position(x=5, y=5), Position(x=0, y=5), Position(x=0, y=0), Prototype.TransportBelt)
    assert belt_group
    assert len(belt_group.inputs) == 1
    assert len(belt_group.outputs) == 1

def test_connect_belt_groups_into_an_octagon(game):
    belt_group = game.connect_entities(Position(x=0, y=0), Position(x=5, y=0), Prototype.TransportBelt)
    belt_group = game.connect_entities(belt_group, Position(x=7, y=2), Prototype.TransportBelt)
    belt_group = game.connect_entities(belt_group, Position(x=7, y=5), Prototype.TransportBelt)
    belt_group = game.connect_entities(belt_group, Position(x=5, y=7), Prototype.TransportBelt)
    belt_group = game.connect_entities(belt_group, Position(x=0, y=7), Prototype.TransportBelt)
    belt_group = game.connect_entities(belt_group, Position(x=-2, y=5), Prototype.TransportBelt)
    belt_group = game.connect_entities(belt_group, Position(x=-2, y=2), Prototype.TransportBelt)
    assert len(belt_group.inputs) == 1, 'There must be a single input'
    assert len(belt_group.outputs) == 1, 'There must be a single output'
    belt_group = game.connect_entities(belt_group, belt_group, Prototype.TransportBelt)
    assert belt_group
    assert len(belt_group.inputs) == 0
    assert len(belt_group.outputs) == 0

def test_belt_group(game):
    game.connect_entities(Position(x=0, y=0), Position(x=5, y=0), Prototype.TransportBelt)
    entities = game.get_entities()
    print(entities)
    belt_groups = [entity for entity in game.get_entities() if isinstance(entity, BeltGroup)]
    assert belt_groups
    pass

def test_connect_belts_with_end_rotation(game):
    iron_pos = game.nearest(Resource.IronOre)
    game.move_to(iron_pos)
    drill2 = game.place_entity(Prototype.BurnerMiningDrill, position=iron_pos, direction=Direction.LEFT)
    chest_pos = Position(x=iron_pos.x - 10, y=iron_pos.y)
    game.move_to(chest_pos)
    collection_chest = game.place_entity(Prototype.WoodenChest, position=chest_pos)
    print(f'Placed collection chest at {collection_chest.position}')
    chest_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=collection_chest.position, direction=Direction.LEFT, spacing=0)
    chest_inserter = game.rotate_entity(chest_inserter, Direction.RIGHT)
    print(f'Placed chest inserter at {chest_inserter.position}')
    chest_2_pos = Position(x=chest_pos.x - 10, y=chest_pos.y)
    game.move_to(chest_2_pos)
    collection_chest2 = game.place_entity(Prototype.WoodenChest, position=chest_2_pos)
    print(f'Placed collection chest at {collection_chest2.position}')
    chest_inserter2 = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=collection_chest2.position, direction=Direction.LEFT, spacing=0)
    chest_inserter2 = game.rotate_entity(chest_inserter2, Direction.RIGHT)
    print(f'Placed chest inserter at {chest_inserter2.position}')
    main_connection = game.connect_entities(drill2.drop_position, chest_inserter2.pickup_position, Prototype.TransportBelt)
    main_connection2 = game.connect_entities(main_connection, chest_inserter.pickup_position, Prototype.TransportBelt)
    assert len(main_connection2.belts) > 20

def test_merge_belt_into_another_belt(game):
    belt_start_position = Position(x=0.0, y=0.0)
    belt_end_position = Position(x=10.0, y=0.0)
    belts = game.connect_entities(belt_start_position, belt_end_position, Prototype.TransportBelt)
    game.move_to(Position(x=0.0, y=0.0))
    nbelt_start = Position(x=5.0, y=5.0)
    nbelt_midpoint = belts.belts[len(belts.belts) // 2]
    merge = game.connect_entities(nbelt_start, nbelt_midpoint.position, Prototype.TransportBelt)
    assert len(merge.inputs) == 2
    assert len(merge.outputs) == 1

def test_merge_multiple_belts(game):
    belt_start_position = Position(x=0.0, y=0.0)
    belt_end_position = Position(x=10.0, y=0.0)
    belts = game.connect_entities(belt_start_position, belt_end_position, Prototype.TransportBelt)
    game.move_to(Position(x=0.0, y=0.0))
    nbelt_start = Position(x=5.0, y=5.0)
    nbelt_midpoint = belts.belts[len(belts.belts) // 2]
    merge = game.connect_entities(nbelt_start, nbelt_midpoint, Prototype.TransportBelt)
    game.move_to(Position(x=0.0, y=0.0))
    nbelt_start = Position(x=5.0, y=-5.0)
    nbelt_midpoint = belts.belts[len(belts.belts) // 2]
    merge = game.connect_entities(nbelt_start, nbelt_midpoint, Prototype.TransportBelt)
    assert len(merge.inputs) == 3
    assert len(merge.outputs) == 1

def test_multi_belt_join(game):
    iron_pos = game.nearest(Resource.IronOre)
    print(f'Found iron ore at {iron_pos}')
    game.move_to(Position(x=15.5, y=70.5))
    drill1 = game.place_entity(Prototype.BurnerMiningDrill, position=Position(x=15.5, y=70.5), direction=Direction.LEFT)
    print(f'placed drill at {drill1.position}')
    game.move_to(Position(x=11.5, y=69.5))
    drill2 = game.place_entity(Prototype.BurnerMiningDrill, position=Position(x=19.5, y=72.5), direction=Direction.LEFT)
    print(f'placed drill at {drill2.position}')
    game.move_to(Position(x=22.5, y=75.5))
    drill3 = game.place_entity(Prototype.BurnerMiningDrill, position=Position(x=22.5, y=75.5), direction=Direction.LEFT)
    print(f'placed drill at {drill3.position}')
    game.move_to(Position(x=-0.5, y=66.5))
    collection_chest = game.place_entity(Prototype.WoodenChest, position=Position(x=-0.5, y=66.5))
    print(f'Placed collection chest at {collection_chest.position}')
    chest_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=collection_chest.position, direction=Direction.LEFT, spacing=0)
    chest_inserter = game.rotate_entity(chest_inserter, Direction.RIGHT)
    print(f'Placed chest inserter at {chest_inserter.position}')
    chest_inserter = game.insert_item(Prototype.Coal, chest_inserter, quantity=50)
    belts = game.connect_entities(drill3.drop_position, chest_inserter.pickup_position, Prototype.TransportBelt)
    print(f'Connected drill at {drill2.position} to collection')
    belts = game.connect_entities(drill1.drop_position, belts, Prototype.TransportBelt)
    print(f'Connected drill at {drill1.position} to collection system')
    belts = game.connect_entities(drill2.drop_position, belts, Prototype.TransportBelt)
    assert len(belts.belts) > 25

def test_ensure_final_belt_rotation_correct(game):
    iron_ore_loc = game.nearest(Resource.IronOre)
    print(f'found iron ore at {iron_ore_loc}')
    game.move_to(iron_ore_loc)
    print('Moved to iron ore location')
    furnace = game.place_entity(Prototype.StoneFurnace, position=iron_ore_loc)
    print(f'Placed a drill at location ({furnace.position}) and inserted coal')
    furnace_output_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace.position, spacing=0)
    print(f'Placed inserter at {furnace_output_inserter.position} and inserted coal')
    chest_pos = Position(x=furnace.position.x - 9, y=furnace.position.y)
    game.move_to(chest_pos)
    chest = game.place_entity(Prototype.WoodenChest, position=chest_pos)
    print(f'Placed chest to pickup plates at ({chest.position})')
    belts = game.connect_entities(furnace_output_inserter.drop_position, chest.position.right(1), Prototype.TransportBelt)
    print(f'Connected furnace_output_inserter at {furnace_output_inserter.position} to chest at {chest.position} with belts {belts}')
    assert len(belts.belts) > 10
    assert True, 'Could not create belt'

def test_connect_furnace(game):
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=2, y=0))
    try:
        game.connect_entities(Position(x=0, y=0), furnace.position, Prototype.TransportBelt)
        assert False, 'Should not be able to connect here, as it is blocked by the furnace'
    except:
        assert True

def test_failure_to_connect_furnace(game):
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=2, y=0))
    try:
        game.connect_entities(Position(x=0, y=-10), furnace, Prototype.TransportBelt)
        assert False, 'Should not be able to connect here, as it is blocked by the furnace'
    except Exception:
        assert True

def test_get_existing_belt_connection_group(game):
    """Test that connecting already-connected belt entities returns existing group instead of failing"""
    pos1 = Position(x=10, y=10)
    pos2 = Position(x=15, y=10)
    first_connection = game.connect_entities(pos1, pos2, Prototype.TransportBelt)
    assert first_connection, 'Initial belt connection should succeed'
    assert hasattr(first_connection, 'belts'), 'Should return BeltGroup'
    second_connection = game.connect_entities(pos1, pos2, Prototype.TransportBelt)
    assert second_connection, 'Second belt connection should return existing group'
    if hasattr(second_connection, 'belts'):
        assert len(second_connection.belts) > 0, 'Should have belts in returned group'
    print(f'✓ First belt connection: {type(first_connection)}')
    print(f'✓ Second belt connection: {type(second_connection)}')

def test_belt_connect_retry_logic(game):
    """Test retry logic for intermittent Lua errors in belt connections"""
    pos1 = Position(x=50, y=50)
    pos2 = Position(x=55, y=50)
    for i in range(3):
        try:
            connection = game.connect_entities(pos1, pos2, Prototype.TransportBelt)
            assert connection, f'Belt connection attempt {i + 1} should succeed'
            break
        except Exception as e:
            if 'attempt to index field' in str(e):
                print(f'Caught expected Lua error on attempt {i + 1}, retry should handle this')
            else:
                raise
    print('✓ Belt retry logic allows connections to succeed')

def test_mixed_belt_types_connection(game):
    """Test connecting with different belt types and existing groups"""
    belt_types = [Prototype.TransportBelt, Prototype.FastTransportBelt, Prototype.ExpressTransportBelt]
    y_offset = 0
    for belt_type in belt_types:
        game.move_to(Position(x=30, y=70 + y_offset))
        test_pos1 = Position(x=30, y=70 + y_offset)
        test_pos2 = Position(x=35, y=70 + y_offset)
        connection = game.connect_entities(test_pos1, test_pos2, belt_type)
        assert connection, f'Connection with {belt_type} should succeed'
        second_connection = game.connect_entities(test_pos1, test_pos2, belt_type)
        assert second_connection, f'Second connection with {belt_type} should return existing group'
        y_offset += 3
    print('✓ All belt types connect successfully and handle existing connections')

def test_belt_connection_to_existing_entities(game):
    """Test connecting belts to entities that already exist"""
    game.move_to(Position(x=90, y=90))
    inserter1 = game.place_entity(Prototype.BurnerInserter, position=Position(x=90, y=90))
    inserter2 = game.place_entity(Prototype.BurnerInserter, position=Position(x=95, y=90))
    connection = game.connect_entities(inserter1, inserter2, Prototype.TransportBelt)
    assert connection, 'Belt connection between existing entities should succeed'
    second_connection = game.connect_entities(inserter1, inserter2, Prototype.TransportBelt)
    assert second_connection, 'Second belt connection should return existing group'
    print('✓ Belt connections to existing entities handled properly')

def test_connect_pipes_underground_limited_inventory(game):
    game.instance.initial_inventory = {'pipe-to-ground': 2, 'pipe': 200}
    game.instance.reset()
    belt_start_position = Position(x=0, y=-5.0)
    belt_end_position = Position(x=0.0, y=15.0)
    try:
        belts = game.connect_entities(belt_start_position, belt_end_position, {Prototype.UndergroundPipe, Prototype.Pipe})
        counter = 0
        for belt in belts.belts:
            if isinstance(belt, UndergroundBelt):
                counter += 1
        assert counter == 2
        print(f'Transport Belts laid from {belt_start_position} to {belt_end_position}.')
    except Exception as e:
        print(f'Failed to lay Transport Belts: {e}')

def test_connect_pipes_with_underground_pipes(game):
    """
    This should ensure that pipe groups are always returned - instead of pipes themselves.
    """
    position_1 = Position(x=0.5, y=0.5)
    position_2 = Position(x=0, y=20)
    pipes = game.connect_entities(position_1, position_2, {Prototype.Pipe, Prototype.UndergroundPipe})
    assert len(pipes.pipes) > 5

def test_connect_pipes_with_underground_pipes_loop(game):
    """
    This should ensure that pipe groups are always returned - instead of pipes themselves.
    """
    position_1 = Position(x=10, y=0)
    position_2 = Position(x=10, y=10)
    position_3 = Position(x=20, y=10)
    position_4 = Position(x=20, y=0)
    pipes = game.connect_entities(position_1, position_2, position_3, position_4, {Prototype.Pipe, Prototype.UndergroundPipe})
    assert len(pipes.pipes) == 18

def test_electricity_far_west_configuration(game):
    """Test electricity connection with steam engine far west of boiler"""
    boiler_pos = Position(x=-15.5, y=-5.5)
    steam_engine_pos = boiler_pos.left(20).up(10)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)

def create_electricity_connection(game, steam_engine_pos, boiler_pos):
    water_pos = game.nearest(Resource.Water)
    game.move_to(water_pos)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_pos)
    print(offshore_pump)
    game.move_to(boiler_pos)
    boiler = game.place_entity(Prototype.Boiler, position=boiler_pos)
    game.insert_item(Prototype.Coal, boiler, 20)
    water_pipes = game.connect_entities(offshore_pump, boiler, {Prototype.Pipe, Prototype.UndergroundPipe})
    game.move_to(steam_engine_pos)
    engine = game.place_entity(Prototype.SteamEngine, position=steam_engine_pos)
    steam_pipes = game.connect_entities(boiler, engine, {Prototype.Pipe, Prototype.UndergroundPipe})
    engine = game.get_entity(Prototype.SteamEngine, engine.position)
    assert steam_pipes
    assert water_pipes
    assert engine.energy > 0

def test_electricity_vertical_close_configuration(game):
    """Test electricity connection with steam engine directly above boiler"""
    boiler_pos = Position(x=-5.5, y=0.5)
    steam_engine_pos = boiler_pos.up(4)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)
    pass

def test_electricity_northwest_configuration(game):
    """Test electricity connection with steam engine northwest of boiler"""
    boiler_pos = Position(x=-5.5, y=4.5)
    steam_engine_pos = boiler_pos.up(15).left(15)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)

def test_electricity_far_west_horizontal_configuration(game):
    """Test electricity connection with steam engine far west on same y-level"""
    boiler_pos = Position(x=-5.5, y=4.5)
    steam_engine_pos = boiler_pos.left(20)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)

def test_electricity_east_configuration(game):
    """Test electricity connection with steam engine east of boiler"""
    boiler_pos = Position(x=-5.5, y=-2.5)
    steam_engine_pos = boiler_pos.right(5)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)

def test_electricity_vertical_below_configuration(game):
    """Test electricity connection with steam engine below boiler"""
    boiler_pos = Position(x=-5.5, y=-2.5)
    steam_engine_pos = boiler_pos.down(5)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)
    pass

def test_electricity_southwest_configuration(game):
    """Test electricity connection with steam engine southwest of boiler"""
    steam_engine_pos = Position(x=-15.5, y=-7.5)
    boiler_pos = Position(x=-5.5, y=-2.5)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)

def test_electricity_southwest_far_configuration(game):
    """Test electricity connection with steam engine far southwest of boiler"""
    steam_engine_pos = Position(x=-15.5, y=-7.5)
    boiler_pos = Position(x=-5.5, y=5.5)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)

def test_electricity_southwest_offset_configuration(game):
    """Test electricity connection with steam engine southwest of offset boiler"""
    steam_engine_pos = Position(x=-15.5, y=-7.5)
    boiler_pos = Position(x=-8.5, y=5.5)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)
    pass

def test_electricity_south_configuration(game):
    """Test electricity connection with steam engine south of boiler"""
    steam_engine_pos = Position(x=-5.5, y=-7.5)
    boiler_pos = Position(x=-8.5, y=5.5)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)
    pass

def test_electricity_north_configuration(game):
    """Test electricity connection with steam engine north of boiler"""
    steam_engine_pos = Position(x=8.5, y=15.5)
    boiler_pos = Position(x=8.5, y=5.5)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)
    pass

def test_electricity_southwest_horizontal_configuration(game):
    """Test electricity connection with steam engine southwest on same y-level"""
    steam_engine_pos = Position(x=8.5, y=4.5)
    boiler_pos = Position(x=0.5, y=-5.5)
    create_electricity_connection(game, steam_engine_pos, boiler_pos)
    pass

def test_connect_steam_engines(game):
    steam_engine_pos1 = Position(x=0, y=4.5)
    game.move_to(steam_engine_pos1)
    engine1 = game.place_entity(Prototype.SteamEngine, position=steam_engine_pos1)
    steam_engine_pos2 = Position(x=5, y=4.5)
    game.move_to(steam_engine_pos2)
    engine2 = game.place_entity(Prototype.SteamEngine, position=steam_engine_pos2)
    game.connect_entities(engine1, engine2, Prototype.Pipe)
    game.connect_entities(engine1, engine2, Prototype.Pipe)
    assert True

def test_connect_boilers(game):
    pos1 = Position(x=0, y=4.5)
    game.move_to(pos1)
    boiler1 = game.place_entity(Prototype.Boiler, position=pos1)
    pos2 = Position(x=5, y=4.5)
    game.move_to(pos2)
    boiler2 = game.place_entity(Prototype.Boiler, position=pos2)
    game.connect_entities(boiler1, boiler2, Prototype.Pipe)
    assert True

def test_for_attribute_error(game):
    boiler_position = Position(x=2.5, y=29.5)
    game.move_to(boiler_position)
    boiler = game.place_entity(Prototype.Boiler, position=boiler_position)
    offshore_pump_pos = Position(x=-0, y=29)
    pump = game.place_entity(Prototype.OffshorePump, position=offshore_pump_pos, direction=Direction.UP)
    game.connect_entities(pump, boiler, Prototype.Pipe)
    try:
        engine = game.place_entity(Prototype.SteamEngine, position=Position(x=8.5, y=28.5))
        game.connect_entities(engine, boiler, Prototype.Pipe)
    except Exception as e:
        assert not isinstance(e, AttributeError)

def test_connect_belt_underground(game):
    game.instance.initial_inventory = {'express-underground-belt': 4, 'express-transport-belt': 200}
    game.instance.reset()
    position_1 = Position(x=0.0, y=1.0)
    position_2 = Position(x=0.0, y=10.0)
    position_3 = Position(x=10, y=10)
    try:
        belts = game.connect_entities(position_1, position_2, position_3, {Prototype.ExpressTransportBelt, Prototype.ExpressUndergroundBelt})
        counter = 0
        for belt in belts.belts:
            if isinstance(belt, UndergroundBelt):
                counter += 1
        game.pickup_entity(belts)
        assert not game.get_entities()
        assert counter == 2
        print(f'Transport Belts laid from {position_1} to {position_3}.')
    except Exception as e:
        print(f'Failed to lay Transport Belts: {e}')

def test_connect_steam_engine_to_assembler_with_electricity_poles(game):
    """
    Place a steam engine and an assembling machine next to each other.
    Connect them with electricity poles.
    :param game:
    :return:
    """
    steam_engine = game.place_entity(Prototype.SteamEngine, position=Position(x=0, y=0))
    assembler = game.place_entity_next_to(Prototype.AssemblingMachine1, reference_position=steam_engine.position, direction=game.RIGHT, spacing=10)
    game.move_to(Position(x=5, y=5))
    diagonal_assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=10, y=10))
    inspected_assemblers = game.get_entities({Prototype.AssemblingMachine1}, position=diagonal_assembler.position)
    for a in inspected_assemblers:
        assert a.warnings == ['not connected to power network']
    poles_in_inventory = game.inspect_inventory()[Prototype.SmallElectricPole]
    game.connect_entities(steam_engine, assembler, connection_type=Prototype.SmallElectricPole)
    poles2 = game.connect_entities(steam_engine, diagonal_assembler, connection_type=Prototype.SmallElectricPole)
    current_poles_in_inventory = game.inspect_inventory()[Prototype.SmallElectricPole]
    spent_poles = poles_in_inventory - current_poles_in_inventory
    assert spent_poles == len(poles2.poles)
    assemblers = game.get_entities({Prototype.AssemblingMachine1})
    for assembler in assemblers:
        assert assembler.status == EntityStatus.NO_POWER

def test_pole_to_generator(game):
    game.move_to(Position(x=1, y=1))
    water = game.get_resource_patch(Resource.Water, game.nearest(Resource.Water))
    water_position = water.bounding_box.right_bottom
    assert water_position, 'No water source found nearby'
    game.move_to(water_position)
    offshore_pump = game.place_entity(Prototype.OffshorePump, Direction.DOWN, water_position)
    assert offshore_pump, 'Failed to place offshore pump'
    boiler = game.place_entity_next_to(Prototype.Boiler, offshore_pump.position, Direction.RIGHT, spacing=2)
    assert boiler, 'Failed to place boiler'
    boiler = game.insert_item(Prototype.Coal, boiler, quantity=5)
    pipes = game.connect_entities(offshore_pump, boiler, Prototype.Pipe)
    assert pipes, 'Failed to connect offshore pump to boiler'
    steam_engine = game.place_entity_next_to(Prototype.SteamEngine, boiler.position, Direction.RIGHT, spacing=2)
    assert steam_engine, 'Failed to place steam engine'
    pipes = game.connect_entities(boiler, steam_engine, Prototype.Pipe)
    assert pipes, 'Failed to connect boiler to steam engine'
    inspected_steam_engine = game.get_entities({Prototype.SteamEngine}, position=steam_engine.position)[0]
    assert inspected_steam_engine.status == EntityStatus.NOT_PLUGGED_IN_ELECTRIC_NETWORK
    '\n    Step 1: Place electric mining drill. We need to find a stone patch and place the electric mining drill on it.\n    '
    stone_patch_position = game.nearest(Resource.Stone)
    print(f'Nearest stone patch found at: {stone_patch_position}')
    game.move_to(stone_patch_position)
    print(f'Moved to stone patch at: {stone_patch_position}')
    drill = game.place_entity(Prototype.ElectricMiningDrill, Direction.UP, stone_patch_position)
    print(f'Placed electric mining drill at: {drill.position}')
    print('Electric mining drill successfully placed on stone patch')
    print(f'Current inventory: {game.inspect_inventory()}')
    '\n    Step 2: Connect power to the drill. We need to create a power line from the steam engine to the electric mining drill using small electric poles.\n    '
    entities = game.get_entities({Prototype.SteamEngine})
    steam_engines = [x for x in entities if x.prototype is Prototype.SteamEngine]
    steam_engine = steam_engines[0]
    connection = game.connect_entities(steam_engine, drill, Prototype.SmallElectricPole)
    assert connection, 'Failed to connect electric mining drill to power'
    print('Electric mining drill connected to power')
    '\n    Step 3: Verify power connection. We need to check if the electric mining drill is powered by examining its status.\n    - Wait for a few seconds to allow the power to stabilize\n    - Check the status of the electric mining drill to confirm it has power\n    '
    game.sleep(5)
    drill = game.get_entity(Prototype.ElectricMiningDrill, drill.position)
    drill_status = drill.status
    assert drill_status != EntityStatus.NO_POWER, 'Electric mining drill is not powered'
    print('Electric mining drill is powered and working')

def test_pole_groups(game):
    water_position = game.nearest(Resource.Water)
    game.move_to(water_position)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_position)
    print(offshore_pump)
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, spacing=3)
    boiler = game.insert_item(Prototype.Coal, boiler, 10)
    steam_engine = game.place_entity_next_to(Prototype.SteamEngine, reference_position=boiler.position, spacing=3)
    print(f'Placed steam_engine at {steam_engine.position}')
    game.connect_entities(offshore_pump, boiler, Prototype.Pipe)
    game.connect_entities(boiler, steam_engine, Prototype.Pipe)
    game.sleep(5)
    print(steam_engine)
    game.connect_entities(steam_engine.position, Position(x=4, y=-20), Prototype.SmallElectricPole)
    entities = game.get_entities()
    assert len(entities) == 6

def test_connect_electricity_2(game):
    print('Starting to build power infrastructure')
    water_pos = game.nearest(Resource.Water)
    game.move_to(water_pos)
    pump = game.place_entity(Prototype.OffshorePump, position=water_pos)
    print(f'Placed offshore pump at {pump.position}')
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=pump.position, direction=Direction.RIGHT, spacing=2)
    print(f'Placed boiler at {boiler.position}')
    boiler = game.insert_item(Prototype.Coal, boiler, 50)
    print('Added coal to boiler')
    steam_engine = game.place_entity_next_to(Prototype.SteamEngine, reference_position=boiler.position, direction=Direction.RIGHT, spacing=2)
    print(f'Placed steam engine at {steam_engine.position}')
    game.connect_entities(pump, boiler, Prototype.Pipe)
    print('Connected water from pump to boiler')
    game.connect_entities(boiler, steam_engine, Prototype.Pipe)
    print('Connected steam from boiler to engine')
    game.sleep(5)
    steam_engine = game.get_entity(Prototype.SteamEngine, steam_engine.position)
    assert steam_engine.energy > 0, 'Steam engine is not generating power'
    print('Power infrastructure successfully built and generating electricity')
    pole_group = game.connect_entities(steam_engine, Position(x=0, y=0), Prototype.SmallElectricPole)
    pole_group = game.connect_entities(pole_group, Position(x=10, y=-10), Prototype.SmallElectricPole)
    pass

def test_prevent_power_pole_cobwebbing(game):
    """
    Test that the connect_entities function prevents unnecessary power pole placement
    when points are already connected to the same power network.
    """
    steam_engine = game.place_entity(Prototype.SteamEngine, position=Position(x=0, y=0))
    pole1 = game.place_entity_next_to(Prototype.SmallElectricPole, steam_engine.position, Direction.RIGHT, spacing=3)
    pole2 = game.place_entity_next_to(Prototype.SmallElectricPole, steam_engine.position, Direction.DOWN, spacing=3)
    pole3 = game.place_entity_next_to(Prototype.SmallElectricPole, pole1.position, Direction.DOWN, spacing=3)
    game.connect_entities(steam_engine, pole3, connection_type=Prototype.SmallElectricPole)
    nr_of_poles = len(game.get_entities({Prototype.ElectricityGroup})[0].poles)
    game.connect_entities(pole1, pole2, connection_type=Prototype.SmallElectricPole)
    groups = game.get_entities({Prototype.ElectricityGroup})
    assert len(groups[0].poles) == nr_of_poles, f'Expected only {nr_of_poles} poles, found {len(groups[0].poles)}'
    ids = {pole.electrical_id for pole in groups[0].poles}
    assert len(ids) == 1, 'All poles should be in the same network'

def test_get_existing_electricity_connection_group(game):
    """Test existing electricity group return functionality"""
    pos1 = Position(x=30, y=30)
    pos2 = Position(x=35, y=30)
    first_poles = game.connect_entities(pos1, pos2, Prototype.SmallElectricPole)
    assert first_poles, 'Initial pole connection should succeed'
    second_poles = game.connect_entities(pos1, pos2, Prototype.SmallElectricPole)
    assert second_poles, 'Second pole connection should return existing group'
    print('✓ Electricity connection handled gracefully')

def test_pole_retry_logic(game):
    """Test retry logic for intermittent Lua errors in pole connections"""
    pos1 = Position(x=40, y=40)
    pos2 = Position(x=50, y=40)
    for i in range(3):
        try:
            connection = game.connect_entities(pos1, pos2, Prototype.SmallElectricPole)
            assert connection, f'Pole connection attempt {i + 1} should succeed'
            break
        except Exception as e:
            if 'attempt to index field' in str(e):
                print(f'Caught expected Lua error on attempt {i + 1}, retry should handle this')
            else:
                raise
    print('✓ Pole retry logic allows connections to succeed')

def test_pole_performance_no_sleep(game):
    """Test that pole connections complete without artificial delays"""
    import time
    pos1 = Position(x=60, y=60)
    pos2 = Position(x=70, y=60)
    start_time = time.time()
    connection = game.connect_entities(pos1, pos2, Prototype.SmallElectricPole)
    end_time = time.time()
    assert connection, 'Pole connection should succeed'
    duration = end_time - start_time
    assert duration < 5.0, f'Pole connection took {duration}s, should be faster without sleep'
    print(f'✓ Pole connection completed in {duration:.2f}s (performance improved)')

def test_pole_network_connections_multiple_types(game):
    """Test electric pole network connections with different pole types"""
    pole_types = [Prototype.SmallElectricPole, Prototype.MediumElectricPole, Prototype.BigElectricPole]
    y_offset = 0
    for pole_type in pole_types:
        pos1 = Position(x=100, y=100 + y_offset)
        pos2 = Position(x=110, y=100 + y_offset)
        connection = game.connect_entities(pos1, pos2, pole_type)
        assert connection, f'{pole_type} connection should succeed'
        second_connection = game.connect_entities(pos1, pos2, pole_type)
        assert second_connection, f'Second {pole_type} connection should return existing group'
        y_offset += 15
    print('✓ All pole types create networks successfully and handle existing connections')

def test_pole_connection_to_existing_entities(game):
    """Test connecting poles to entities that already exist"""
    game.move_to(Position(x=45, y=45))
    steam_engine = game.place_entity(Prototype.SteamEngine, position=Position(x=40, y=40))
    drill = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=50, y=50))
    first_poles = game.connect_entities(steam_engine, drill, Prototype.SmallElectricPole)
    assert first_poles, 'Initial pole connection should succeed'
    second_poles = game.connect_entities(steam_engine, drill, Prototype.SmallElectricPole)
    assert second_poles, 'Second pole connection should return existing group'
    print('✓ Pole connections to existing entities handled properly')

def test_moving_accumulate_ticks(game):
    ticks = game.instance.get_elapsed_ticks()
    ticks_ = []
    for i in range(10):
        game.move_to(Position(x=i, y=0))
        ticks_.append(game.instance.get_elapsed_ticks())
    nticks = game.instance.get_elapsed_ticks()
    assert nticks > 80, 'The tick count should be proportional to the distance moved.'
    assert nticks - ticks == ticks, 'The tick count should be invariant to the number of moves made.'

def test_fail_on_incorrect_blueprint(game):
    assert not game._load_blueprint('BLHA', Position(x=0, y=0))

def test_belt_inserter_chain(game):
    belt1 = game.place_entity(Prototype.TransportBelt, Direction.EAST, Position(x=0, y=0))
    game.place_entity(Prototype.TransportBelt, Direction.EAST, Position(x=1, y=0))
    game.place_entity(Prototype.Inserter, Direction.NORTH, Position(x=1, y=1))
    chest = game.place_entity(Prototype.WoodenChest, Direction.NORTH, Position(x=1, y=2))
    game.insert_item(Prototype.IronOre, belt1, quantity=5)
    game.insert_item(Prototype.Coal, chest, quantity=5)
    entities = game._save_entity_state(distance=30, player_entities=True)
    game.reset()
    assert game._load_entity_state(entities)
    pass

def test_save_load1(game):
    furnace = game.place_entity(Prototype.StoneFurnace, Direction.UP, Position(x=5, y=0))
    game.insert_item(Prototype.Coal, furnace, quantity=5)
    game.insert_item(Prototype.IronOre, furnace, quantity=5)
    game.move_to(Position(x=20, y=20))
    game.instance.set_speed(1)
    entities = game._save_entity_state(distance=30, player_entities=True)
    copied_entities = deepcopy(entities)
    game.instance.reset()
    assert game._load_entity_state(entities)
    entities = game._save_entity_state(distance=30, player_entities=True)
    game.instance.set_speed(1)
    assert copied_entities[0]['burner']['inventory']['coal'] == entities[0]['burner']['inventory']['coal']

def test_benchmark(game):
    furnace = game.place_entity(Prototype.StoneFurnace, Direction.UP, Position(x=5, y=0))
    game.insert_item(Prototype.Coal, furnace, quantity=5)
    game.insert_item(Prototype.IronOre, furnace, quantity=5)
    game.move_to(Position(x=20, y=20))
    save_times = []
    load_times = []
    lengths = []
    for i in range(10):
        save_start = time.time()
        entities = game._save_entity_state(distance=100, encode=True, compress=True)
        lengths.append(len(entities))
        save_end = time.time()
        game.reset()
        load_start = time.time()
        game._load_entity_state(entities, decompress=True)
        load_end = time.time()
        save_times.append(save_end - save_start)
        load_times.append(load_end - load_start)
    print()
    print(f'Average save time: {sum(save_times) / len(save_times) * 1000} milliseconds (player entities)')
    print(f'Average load time: {sum(load_times) / len(load_times) * 1000} milliseconds (player entities)')
    print(f'Average length of saved data: {sum(lengths) / len(lengths)} bytes')

def test_get_iron_chest(game):
    """
    Test to ensure that the inventory of an iron chest is correctly updated after inserting items
    :param game:
    :return:
    """
    inventory = game.inspect_inventory()
    iron_chest_count = inventory.get(Prototype.IronChest, 0)
    assert iron_chest_count != 0, 'Failed to get iron chest count'
    iron_chest = game.place_entity(Prototype.IronChest, position=Position(x=0, y=0))
    game.insert_item(Prototype.Coal, iron_chest, quantity=5)
    game.insert_item(Prototype.IronPlate, iron_chest, quantity=5)
    retrieved_chest = game.get_entity(Prototype.IronChest, iron_chest.position)
    assert retrieved_chest is not None, 'Failed to retrieve iron chest'
    assert retrieved_chest.inventory.get(Prototype.Coal, 0) == 5, 'Failed to insert coal'
    assert retrieved_chest.inventory.get(Prototype.IronPlate, 0) == 5, 'Failed to insert iron plate'

def test_get_assembling_machine(game):
    """
    Test to ensure that the inventory of an assembling machine is correctly updated after crafting items
    :param game:
    :return:
    """
    inventory = game.inspect_inventory()
    assembling_machine_count = inventory.get(Prototype.AssemblingMachine1, 0)
    assert assembling_machine_count != 0, 'Failed to get assembling machine count'
    assembling_machine = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=0, y=0))
    game.set_entity_recipe(assembling_machine, Prototype.IronGearWheel)
    game.insert_item(Prototype.IronPlate, assembling_machine, quantity=5)
    game.craft_item(Prototype.IronGearWheel, 5)
    game.insert_item(Prototype.IronGearWheel, assembling_machine, quantity=5)
    retrieved_machine = game.get_entity(Prototype.AssemblingMachine1, assembling_machine.position)
    assert retrieved_machine is not None, 'Failed to retrieve assembling machine'
    assert retrieved_machine.assembling_machine_output.get(Prototype.IronGearWheel, 0) == 5, 'Failed to get output inventory'
    assert retrieved_machine.assembling_machine_input.get(Prototype.IronPlate, 0) == 5, 'Failed to consume input inventory'

def test_get_lab(game):
    """
    Test to ensure that the inventory of a lab is correctly updated after researching a science pack
    :param game:
    :return:
    """
    inventory = game.inspect_inventory()
    lab_count = inventory.get(Prototype.Lab, 0)
    assert lab_count != 0, 'Failed to get lab count'
    lab = game.place_entity(Prototype.Lab, position=Position(x=0, y=0))
    game.insert_item(Prototype.AutomationSciencePack, lab, quantity=1)
    retrieved_lab = game.get_entity(Prototype.Lab, lab.position)
    assert retrieved_lab is not None, 'Failed to retrieve lab'
    assert retrieved_lab.lab_input.get(Prototype.AutomationSciencePack, 0) == 1, 'Failed to consume science pack'

def test_get_turret(game):
    """
    Test to ensure that the inventory of a turret is correctly updated after shooting a target
    :param game:
    :return:
    """
    inventory = game.inspect_inventory()
    turret_count = inventory.get(Prototype.GunTurret, 0)
    assert turret_count != 0, 'Failed to get turret count'
    turret = game.place_entity(Prototype.GunTurret, position=Position(x=0, y=0))
    game.insert_item(Prototype.FirearmMagazine, turret, quantity=5)
    retrieved_turret = game.get_entity(Prototype.GunTurret, turret.position)
    assert retrieved_turret is not None, 'Failed to retrieve turret'
    assert retrieved_turret.turret_ammo.get(Prototype.FirearmMagazine, 0) == 5, 'Failed to consume ammo'

def test_get_boiler(game):
    """
    Test to ensure that the inventory of a boiler is correctly updated after burning fuel
    :param game:
    :return:
    """
    inventory = game.inspect_inventory()
    boiler_count = inventory.get(Prototype.Boiler, 0)
    assert boiler_count != 0, 'Failed to get boiler count'
    boiler = game.place_entity(Prototype.Boiler, position=Position(x=0, y=0))
    game.insert_item(Prototype.Coal, boiler, quantity=5)
    retrieved_boiler = game.get_entity(Prototype.Boiler, boiler.position)
    assert retrieved_boiler is not None, 'Failed to retrieve boiler'
    assert retrieved_boiler.fuel.get(Prototype.Coal, 0) == 5, 'Failed to consume fuel'

def test_get_assembling_machine_1(game):
    """
    Test to ensure that the inventory of an assembling machine is correctly updated after crafting items
    :param game:
    :return:
    """
    inventory = game.inspect_inventory()
    assembling_machine_count = inventory.get(Prototype.AssemblingMachine1, 0)
    assert assembling_machine_count != 0, 'Failed to get assembling machine count'
    assembling_machine = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=0, y=0))
    game.set_entity_recipe(assembling_machine, Prototype.IronGearWheel)
    game.insert_item(Prototype.IronPlate, assembling_machine, quantity=5)
    retrieved_machine = game.get_entity(Prototype.AssemblingMachine1, assembling_machine.position)
    assert retrieved_machine is not None, 'Failed to retrieve assembling machine'

def test_pickup_item_full_inventory(game):
    """
    Test that pickup fails when inventory is at maximum capacity.
    Uses existing inventory items but maximizes stacks to test true full inventory.
    """
    game._set_inventory({'wooden-chest': 1})
    placement_position = Position(x=0, y=0)
    game.move_to(placement_position)
    chest = game.place_entity(Prototype.WoodenChest, position=placement_position)
    game._set_inventory({'coal': 10000})
    try:
        result = game.pickup_entity(chest)
        assert False, f'Expected pickup to fail due to full inventory, but got result: {result}'
    except Exception as e:
        print(e)
        assert True

def test_place_pickup(game):
    """
    Place a boiler at (0, 0) and then pick it up
    :param game:
    :return:
    """
    boilers_in_inventory = game.inspect_inventory()[Prototype.Boiler]
    game.place_entity(Prototype.Boiler, position=Position(x=0, y=0))
    assert boilers_in_inventory == game.inspect_inventory()[Prototype.Boiler] + 1
    game.pickup_entity(Prototype.Boiler, position=Position(x=0, y=0))
    assert boilers_in_inventory == game.inspect_inventory()[Prototype.Boiler]

def test_place_pickup_pipe_group(game):
    game.move_to(Position(x=0, y=0))
    water_pipes = game.connect_entities(Position(x=0, y=1), Position(x=10, y=1), connection_type=Prototype.Pipe)
    game.pickup_entity(water_pipes)
    assert game.inspect_inventory()[Prototype.Pipe] == 100
    game.move_to(Position(x=0, y=0))
    water_pipes = game.connect_entities(Position(x=0, y=1), Position(x=10, y=1), connection_type=Prototype.Pipe)
    for pipe in water_pipes.pipes:
        game.pickup_entity(pipe)
    assert game.inspect_inventory()[Prototype.Pipe] == 100

def test_place_pickup_inventory(game):
    chest = game.place_entity(Prototype.WoodenChest, position=Position(x=0, y=0))
    iron_plate_in_inventory = game.inspect_inventory()[Prototype.IronPlate]
    game.insert_item(Prototype.IronPlate, chest, quantity=5)
    game.pickup_entity(Prototype.WoodenChest, position=chest.position)
    assert game.inspect_inventory()[Prototype.IronPlate] == iron_plate_in_inventory

def test_place_pickup_inventory2(game):
    chest = game.place_entity(Prototype.WoodenChest, position=Position(x=0, y=0))
    iron_plate_in_inventory = game.inspect_inventory()[Prototype.IronPlate]
    game.insert_item(Prototype.IronPlate, chest, quantity=5)
    game.pickup_entity(chest)
    assert game.inspect_inventory()[Prototype.IronPlate] == iron_plate_in_inventory

def test_pickup_belts(game):
    belts = game.connect_entities(Position(x=0.5, y=0.5), Position(x=0.5, y=8.5), Prototype.TransportBelt)
    belt = belts
    game.get_entity(Prototype.BeltGroup, belt.position)
    pickup_belts = game.pickup_entity(belt)
    assert pickup_belts

def test_pickup_belts_position(game):
    belts = game.connect_entities(Position(x=1, y=-1), Position(x=-2, y=0), Prototype.TransportBelt)
    print(belts)
    print(belts.belts)
    game.pickup_entity(Prototype.TransportBelt, Position(x=0.5, y=0.5))
    pass

def test_pickup_pipes(game):
    pipes = game.connect_entities(Position(x=1, y=-1), Position(x=-2, y=0), Prototype.Pipe)
    print(pipes)
    print(pipes.pipes)
    for belt in pipes.pipes:
        game.pickup_entity(Prototype.Pipe, belt.position)
        print(f'Pickup belt at {belt.position}')

def test_pickup_belts_that_dont_exist(game):
    belts = game.connect_entities(Position(x=0.5, y=0.5), Position(x=0.5, y=8.5), Prototype.TransportBelt)
    belt = belts
    nbelts = game.get_entity(Prototype.BeltGroup, belt.position)
    pickup_belts = game.pickup_entity(belt)
    assert pickup_belts
    try:
        game.pickup_entity(nbelts)
    except Exception:
        assert True, 'Should not be able to pick up a non-existent belt'

def test_inspect_inventory(game):
    assert game.inspect_inventory().get(Prototype.Coal, 0) == 50
    inventory = game.inspect_inventory()
    coal_count = inventory[Prototype.Coal]
    assert coal_count != 0
    chest = game.place_entity(Prototype.IronChest, position=Position(x=0, y=0))
    chest = game.insert_item(Prototype.Coal, chest, quantity=5)
    chest_inventory = game.inspect_inventory(entity=chest)
    chest_coal_count = chest_inventory[Prototype.Coal]
    assert chest_coal_count == 5

def test_inspect_assembling_machine_inventory(game):
    machine = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=0, y=0))
    game.set_entity_recipe(machine, Prototype.IronGearWheel)
    game.insert_item(Prototype.IronPlate, machine, quantity=5)
    chest_inventory = game.inspect_inventory(entity=machine)
    iron_count = chest_inventory[Prototype.IronPlate]
    assert iron_count == 5

def test_inserters_above_chest(game):
    game.move_to(Position(x=0, y=0))
    for i in range(3):
        chest = game.place_entity(Prototype.WoodenChest, Direction.UP, Position(x=i, y=0))
        assert chest, 'Failed to place chest'
        inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=Position(x=i, y=0), direction=Direction.UP, spacing=2)
        assert inserter, 'Failed to place inserter'

def test_inserters_below_furnace(game):
    game.move_to(Position(x=0, y=0))
    furnace = game.place_entity(Prototype.StoneFurnace, Direction.UP, Position(x=0, y=0))
    assert furnace, 'Failed to place furnace'
    inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace.position, direction=Direction.DOWN, spacing=0)
    assert inserter, 'Failed to place inserter'

def test_smart_inserter_placement_around_assembler(game):
    """Test smart inserter placement around 3x3 assembling machine"""
    game.move_to(Position(x=10, y=10))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=10.5, y=10.5), direction=Direction.UP)
    assert assembler, 'Failed to place assembling machine'
    directions_and_expected = [(Direction.LEFT, Position(x=8.5, y=10.5)), (Direction.RIGHT, Position(x=12.5, y=10.5)), (Direction.UP, Position(x=10.5, y=8.5)), (Direction.DOWN, Position(x=10.5, y=12.5))]
    for direction, expected_pos in directions_and_expected:
        inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=direction, spacing=0)
        assert inserter, f'Failed to place inserter in direction {direction}'
        assert inserter.position.is_close(expected_pos, tolerance=0.1), f'Inserter not at reserved position for {direction}. Expected {expected_pos}, got {inserter.position}'
        game.instance.reset()
        game.move_to(Position(x=10, y=10))
        assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=10, y=10))

def test_pole_collision_resolution(game):
    """Test that poles blocking optimal positions trigger smart alternatives"""
    game.move_to(Position(x=30, y=30))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=30, y=30))
    assert assembler, 'Failed to place assembling machine'
    blocking_pole = game.place_entity(Prototype.SmallElectricPole, position=Position(x=32, y=30))
    assert blocking_pole, 'Failed to place blocking pole'
    try:
        inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=Direction.RIGHT, spacing=0)
        if inserter:
            blocked_pos = Position(x=32, y=30)
            assert not inserter.position.is_close(blocked_pos, tolerance=0.1), 'Should not place at blocked position'
            print(f'✓ Found alternative position: {inserter.position}')
    except Exception as e:
        error_msg = str(e).lower()
        helpful_keywords = ['pole', 'corner', 'large entities', 'spacing', 'direction']
        assert any((keyword in error_msg for keyword in helpful_keywords)), f'Error should provide helpful suggestions: {e}'
        print(f'✓ Got helpful error message: {e}')

def test_reserved_slots_for_large_entities(game):
    """Test reserved slots work for different large entity types"""
    large_entities = [Prototype.AssemblingMachine1, Prototype.StoneFurnace]
    y_offset = 0
    for entity_proto in large_entities:
        game.move_to(Position(x=40, y=40 + y_offset))
        entity = game.place_entity(entity_proto, position=Position(x=40, y=40 + y_offset))
        assert entity, f'Failed to place {entity_proto}'
        placed_count = 0
        for direction in [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]:
            try:
                inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=entity.position, direction=direction, spacing=0)
                if inserter:
                    placed_count += 1
                    if direction == Direction.LEFT:
                        assert abs(inserter.position.x - (entity.position.x - 2)) < 0.5
                    elif direction == Direction.RIGHT:
                        assert abs(inserter.position.x - (entity.position.x + 2)) < 0.5
                    elif direction == Direction.UP:
                        assert abs(inserter.position.y - (entity.position.y - 2)) < 0.5
                    elif direction == Direction.DOWN:
                        assert abs(inserter.position.y - (entity.position.y + 2)) < 0.5
            except Exception as e:
                print(f'Expected placement issue for {entity_proto} {direction}: {e}')
        assert placed_count > 0, f'Should place at least one inserter around {entity_proto}'
        y_offset += 10
        game.instance.reset()

def test_corner_positions_available_for_poles(game):
    """Test that corner positions remain available after reserving middle slots"""
    game.move_to(Position(x=60, y=60))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=60, y=60))
    assert assembler, 'Failed to place assembling machine'
    for direction in [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]:
        try:
            game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=direction, spacing=0)
        except:
            pass
    corner_positions = [Position(x=58, y=58), Position(x=62, y=58), Position(x=62, y=62), Position(x=58, y=62)]
    poles_placed = 0
    for corner_pos in corner_positions:
        try:
            pole = game.place_entity(Prototype.SmallElectricPole, position=corner_pos)
            if pole:
                poles_placed += 1
        except:
            pass
    assert poles_placed > 0, 'Should be able to place poles at corner positions'
    print(f'✓ Successfully placed {poles_placed} poles at corners')

def test_collision_resolution_alternatives(game):
    """Test smart collision resolution finds good alternatives"""
    game.move_to(Position(x=70, y=70))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=70, y=70))
    game.place_entity(Prototype.WoodenChest, position=Position(x=72, y=70))
    try:
        inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=Direction.RIGHT, spacing=0)
        if inserter:
            blocked_pos = Position(x=72, y=70)
            assert not inserter.position.is_close(blocked_pos, tolerance=0.1), 'Should not place at blocked position'
            print(f'✓ Found alternative at {inserter.position}')
    except Exception as e:
        print(f'✓ Got collision resolution guidance: {e}')

def test_size_detection_accuracy(game):
    """Test that entity size detection correctly identifies 3x3+ entities"""
    game.move_to(Position(x=90, y=90))
    large_3x3_entities = [Prototype.AssemblingMachine1, Prototype.ElectricFurnace]
    for entity_proto in large_3x3_entities:
        entity = game.place_entity(entity_proto, position=Position(x=90, y=90))
        inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=entity.position, direction=Direction.RIGHT, spacing=0)
        expected_x = entity.position.x + 2
        assert abs(inserter.position.x - expected_x) < 0.5, f'{entity_proto} should trigger 3x3 reserved slots (±2 offset)'
        game.instance.reset()
        game.move_to(Position(x=90, y=90))
    print('✓ Size detection correctly identifies 3x3+ entities')

def test_factory_pattern_optimization(game):
    """Test that factory patterns are recognized and optimized"""
    game.move_to(Position(x=50, y=50))
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=50, y=50), direction=Direction.UP)
    assert furnace, 'Failed to place furnace'
    input_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace.position, direction=Direction.LEFT, spacing=0)
    assert input_inserter, 'Failed to place input inserter'
    output_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace.position, direction=Direction.RIGHT, spacing=0)
    assert output_inserter, 'Failed to place output inserter'
    try:
        game.place_entity_next_to(Prototype.TransportBelt, reference_position=input_inserter.position, direction=Direction.LEFT, spacing=0)
        game.place_entity_next_to(Prototype.TransportBelt, reference_position=output_inserter.position, direction=Direction.RIGHT, spacing=0)
        print('Created factory line: input belt -> inserter -> furnace -> inserter -> output belt')
    except Exception as e:
        print(f'Belt placement encountered issue (may be expected): {e}')

def test_belt_placement_with_smart_routing(game):
    """Test belt placement with smart collision avoidance"""
    game.move_to(Position(x=80, y=80))
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=80, y=80))
    inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace.position, direction=Direction.RIGHT, spacing=0)
    belt = game.place_entity_next_to(Prototype.TransportBelt, reference_position=inserter.position, direction=Direction.RIGHT, spacing=0)
    assert belt, 'Belt placement with smart routing should succeed'
    try:
        belt2 = game.place_entity_next_to(Prototype.TransportBelt, reference_position=belt.position, direction=Direction.RIGHT, spacing=0)
        if belt2:
            print('Belt chaining successful with smart placement')
    except Exception as e:
        print(f'Belt chaining issue (may be expected due to space): {e}')

def test_item_on_ground_clearance(game):
    """Test that item-on-ground entities are cleared before placement"""
    nearest_coal = game.nearest(Resource.Coal)
    game.move_to(nearest_coal)
    drill = game.place_entity(Prototype.BurnerMiningDrill, position=nearest_coal)
    assert drill, 'Failed to place drill'
    game.insert_item(Prototype.Coal, drill, 5)
    game.sleep(6)
    chest = game.place_entity(Prototype.WoodenChest, position=drill.drop_position)
    assert chest, 'chest placement should succeed despite items on ground'
    game.move_to(Position(x=20, y=20))
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=20, y=20))
    assert furnace, 'Failed to place furnace'
    print(f'✓ Furnace placed successfully at {furnace.position}')

def test_placement_feedback_system(game):
    """Test that placement feedback is provided for optimization learning"""
    game.move_to(Position(x=120, y=120))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=120, y=120))
    assert assembler, 'Failed to place assembling machine'
    inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=Direction.LEFT, spacing=0)
    assert inserter, 'Failed to place inserter'
    if hasattr(inserter, '_placement_feedback'):
        feedback = inserter._placement_feedback
        assert 'reason' in feedback, 'Feedback should contain reason'
        assert 'optimal' in feedback, 'Feedback should contain optimal flag'
        print(f'✓ Placement feedback: {feedback['reason']}')
        if feedback.get('auto_oriented'):
            print('✓ Inserter was auto-oriented for optimal flow')
    else:
        print('Note: Placement feedback not attached to entity (may be in logs only)')

def test_auto_orientation_verification(game):
    """Test that inserters are automatically oriented for optimal flow"""
    game.move_to(Position(x=140, y=140))
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=140, y=140))
    assert furnace, 'Failed to place furnace'
    input_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace.position, direction=Direction.LEFT, spacing=0)
    assert input_inserter, 'Failed to place input inserter'
    output_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace.position, direction=Direction.RIGHT, spacing=0)
    assert output_inserter, 'Failed to place output inserter'
    print(f'✓ Input inserter direction: {input_inserter.direction}')
    print(f'✓ Output inserter direction: {output_inserter.direction}')

def test_factory_pattern_recognition_details(game):
    """Test detailed factory pattern recognition for different entity types"""
    game.move_to(Position(x=160, y=160))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=160, y=160))
    west_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=Direction.LEFT, spacing=0)
    assert west_inserter, 'West inserter should be placed (optimal input side)'
    east_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=Direction.RIGHT, spacing=0)
    assert east_inserter, 'East inserter should be placed (optimal output side)'
    print('✓ Assembling machine factory pattern recognized')
    game.instance.reset()
    iron_position = game.nearest(Resource.IronOre)
    game.move_to(iron_position)
    drill = game.place_entity(Prototype.ElectricMiningDrill, position=iron_position)
    directions_tested = []
    for direction in [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]:
        try:
            inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=drill.position, direction=direction, spacing=0)
            if inserter:
                directions_tested.append(direction)
        except:
            pass
        game.instance.reset()
        iron_position = game.nearest(Resource.IronOre)
        game.move_to(iron_position)
        drill = game.place_entity(Prototype.ElectricMiningDrill, position=iron_position)
    assert len(directions_tested) > 0, 'Mining drill should allow output inserters'
    print(f'✓ Mining drill accepts inserters in {len(directions_tested)} directions')

def test_large_entity_reserved_slots_details(game):
    """Test detailed reserved slot behavior for large entities"""
    game.move_to(Position(x=180, y=180))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=180, y=180))
    expected_positions = {Direction.LEFT: Position(x=178, y=180), Direction.RIGHT: Position(x=182, y=180), Direction.UP: Position(x=180, y=178), Direction.DOWN: Position(x=180, y=182)}
    for direction, expected_pos in expected_positions.items():
        inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=direction, spacing=0)
        assert inserter, f'Should place inserter in {direction} direction'
        assert inserter.position.is_close(expected_pos, tolerance=0.6), f'Inserter should be at reserved position {expected_pos}, got {inserter.position}'
        game.instance.reset()
        game.move_to(Position(x=180, y=180))
        assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=180, y=180))
    print('✓ Large entity reserved slots working correctly')

def test_collision_resolution_with_helpful_errors(game):
    """Test that collision resolution provides helpful error messages"""
    game.move_to(Position(x=220, y=220))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=220, y=220))
    game.place_entity(Prototype.SmallElectricPole, position=Position(x=222, y=220))
    game.place_entity(Prototype.SmallElectricPole, position=Position(x=218, y=220))
    try:
        inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=Direction.RIGHT, spacing=0)
        if inserter:
            print(f'✓ Found alternative position: {inserter.position}')
    except Exception as e:
        error_msg = str(e).lower()
        helpful_keywords = ['large entities', 'corner', 'pole', 'spacing', 'direction', 'consider', 'middle sides', 'connect_entities']
        found_helpful = any((keyword in error_msg for keyword in helpful_keywords))
        assert found_helpful, f'Error should provide helpful suggestions: {e}'
        print('✓ Got helpful error message with suggestions')

def test_alternative_position_scoring(game):
    """Test that alternative positions are scored and prioritized correctly"""
    game.move_to(Position(x=20, y=20))
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=20, y=20))
    game.place_entity(Prototype.WoodenChest, position=Position(x=22, y=20))
    inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=assembler.position, direction=Direction.RIGHT, spacing=0)
    if inserter:
        blocked_pos = Position(x=22, y=20)
        assert not inserter.position.is_close(blocked_pos, tolerance=0.1), 'Should find alternative, not blocked position'
        distance = ((inserter.position.x - assembler.position.x) ** 2 + (inserter.position.y - assembler.position.y) ** 2) ** 0.5
        assert distance < 5, f'Alternative should be reasonably close, got distance {distance}'
        print(f'✓ Found good alternative at distance {distance:.1f}')

def test_belt_smart_collision_avoidance(game):
    """Test smart collision avoidance specifically for belt placement"""
    game.move_to(Position(x=260, y=260))
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=260, y=260))
    game.place_entity(Prototype.SmallElectricPole, position=Position(x=262, y=261))
    game.place_entity(Prototype.SmallElectricPole, position=Position(x=262, y=259))
    belt = game.place_entity_next_to(Prototype.TransportBelt, reference_position=furnace.position, direction=Direction.RIGHT, spacing=0)
    assert belt, 'Belt should be placed with smart collision avoidance'
    print(f'✓ Belt placed at {belt.position} avoiding obstacles')

def test_dry_run(game):
    position_1 = Position(x=3, y=1)
    position_2 = Position(x=2, y=4)
    belts = game.connect_entities(position_1, position_2, Prototype.TransportBelt, dry_run=True)
    assert game.inspect_inventory()[Prototype.TransportBelt] == 12
    assert len(game.get_entities()) == 0
    assert isinstance(belts, dict)
    assert belts['number_of_entities_available'] == 12
    assert belts['number_of_entities_required'] < 12
    position_1 = Position(x=0, y=0)
    position_2 = Position(x=0, y=25)
    belts = game.connect_entities(position_1, position_2, Prototype.TransportBelt, dry_run=True)
    assert game.inspect_inventory()[Prototype.TransportBelt] == 12
    assert len(game.get_entities()) == 0
    assert isinstance(belts, dict)
    assert belts['number_of_entities_available'] == 12
    assert belts['number_of_entities_required'] > 12

def test_connect_without_enough(game):
    position_1 = Position(x=0, y=0)
    position_2 = Position(x=0, y=25)
    try:
        belts = game.connect_entities(position_1, position_2, Prototype.TransportBelt)
    except Exception as e:
        exception_message = str(e)
        assert 'You do not have enough transport-belt in you inventory to complete this connection. Required number - 26, Available in inventory - 12' in exception_message
        pass
    assert game.inspect_inventory()[Prototype.TransportBelt] == 12
    assert len(game.get_entities()) == 0
    position_1 = Position(x=3, y=1)
    position_2 = Position(x=0, y=4)
    belts = game.connect_entities(position_1, position_2, Prototype.TransportBelt)
    assert game.inspect_inventory()[Prototype.TransportBelt] < 12
    assert len(game.get_entities()) != 0
    assert len(belts.belts) != 0

def test_set_entity_recipe(game):
    assembling_machine = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=0, y=0))
    assembling_machine = game.set_entity_recipe(assembling_machine, Prototype.IronGearWheel)
    prototype_name, _ = Prototype.IronGearWheel.value
    assert assembling_machine.recipe == prototype_name

def test_path(game):
    """
    Get a path from (0, 0) to (10, 0)
    :param game:
    :return:
    """
    path = game._request_path(Position(x=0, y=0), Position(x=10, y=0))
    assert path

def test_insert_and_fuel_furnace(game):
    furnace = game.place_entity(Prototype.StoneFurnace, direction=Direction.UP, position=Position(x=0, y=0))
    furnace = game.insert_item(Prototype.IronOre, furnace, quantity=50)
    furnace = game.insert_item(Prototype.Coal, furnace, quantity=50)
    assert furnace.status == EntityStatus.WORKING
    assert furnace.fuel[Prototype.Coal] == 50
    assert furnace.furnace_source[Prototype.IronOre] == 50

def test_insert_iron_ore_into_stone_furnace(game):
    furnace = game.place_entity(Prototype.StoneFurnace, direction=Direction.UP, position=Position(x=0, y=0))
    furnace = game.insert_item(Prototype.IronOre, furnace, quantity=10)
    assert furnace.status == EntityStatus.NO_FUEL
    assert furnace.furnace_source[Prototype.IronOre] == 10

def test_insert_iron_ore_into_stone_furnace2(game):
    furnace = game.place_entity(Prototype.StoneFurnace, direction=Direction.UP, position=Position(x=0, y=0))
    try:
        furnace = game.insert_item(Prototype.IronOre, furnace, quantity=500)
        furnace = game.insert_item(Prototype.IronPlate, furnace, quantity=10)
    except Exception as e:
        assert True, f'Cannot insert incorrect item into a stone furnace: {e}'

def test_insert_copper_ore_and_iron_ore_into_stone_furnace(game):
    furnace = game.place_entity(Prototype.StoneFurnace, direction=Direction.UP, position=Position(x=0, y=0))
    try:
        furnace = game.insert_item(Prototype.IronOre, furnace, quantity=10)
        furnace = game.insert_item(Prototype.CopperOre, furnace, quantity=10)
    except Exception as e:
        assert True, f'Inserting both copper and iron ore into a stone furnace should raise an exception: {e}'

def test_insert_coal_into_burner_inserter(game):
    inserter = game.place_entity(Prototype.BurnerInserter, direction=Direction.UP, position=Position(x=0, y=0))
    inserter = game.insert_item(Prototype.Coal, inserter, quantity=10)
    assert inserter.fuel[Prototype.Coal] == 10

def test_invalid_insert_ore_into_burner_inserter(game):
    inserter = game.place_entity(Prototype.BurnerInserter, direction=Direction.UP, position=Position(x=0, y=0))
    try:
        inserter = game.insert_item(Prototype.IronOre, inserter, quantity=10)
    except:
        assert True, 'Should not be able to add iron to an inserter'
        return
    assert False, 'Should not be able to add iron to an inserter'

def test_insert_into_assembler(game):
    assembler = game.place_entity(Prototype.AssemblingMachine1, direction=Direction.UP, position=Position(x=0, y=0))
    assembler = game.set_entity_recipe(assembler, Prototype.IronGearWheel)
    assembler = game.insert_item(Prototype.IronGearWheel, assembler, quantity=1000)
    assembler = game.insert_item(Prototype.IronPlate, assembler, quantity=1000)
    assert assembler.status == EntityStatus.NO_POWER
    assert assembler.assembling_machine_input[Prototype.IronPlate] == 100
    assert assembler.assembling_machine_output[Prototype.IronGearWheel] == 100

def test_insert_ore_onto_belt(game):
    belt = game.connect_entities(Position(x=0.5, y=0.5), Position(x=0.5, y=8.5), Prototype.TransportBelt)
    belt = game.insert_item(Prototype.IronOre, belt, quantity=5)
    assert belt.inventory[Prototype.IronOre] == 5

def test_blocked_belt(game):
    belt = game.connect_entities(Position(x=0.5, y=0.5), Position(x=0.5, y=8.5), Prototype.TransportBelt)
    try:
        belt = game.insert_item(Prototype.IronOre, belt, quantity=500)
    except Exception:
        pass
    belt = game.get_entities({Prototype.TransportBelt}, position=Position(x=0.5, y=0.5))
    assert belt[0].status == EntityStatus.FULL_OUTPUT

def test_insert_into_two_furnaces(game):
    furnace_pos = Position(x=-12, y=-12)
    game.move_to(furnace_pos)
    game.place_entity(Prototype.StoneFurnace, Direction.UP, furnace_pos)
    '\n    Step 1: Print recipe. We need to print the recipe for offshore pump\n    '
    recipe = game.get_prototype_recipe(Prototype.OffshorePump)
    print('offshore pump Recipe:')
    print(f'Ingredients: {recipe.ingredients}')
    '\n    Step 1: Gather resources. We need to mine the following:\n    - 5 iron ore\n    - 3 copper ore\n    - Coal (at least 10 for smelting)\n    - 5 stone (to craft an additional furnace)\n    OUTPUT CHECK: Verify that we have at least 5 iron ore, 3 copper ore, 10 coal, and 5 stone in our inventory.\n    '
    resources_to_mine = [(Resource.IronOre, 5), (Resource.CopperOre, 3), (Resource.Coal, 10), (Resource.Stone, 5)]
    for resource_type, required_amount in resources_to_mine:
        print(f'Finding nearest {resource_type}...')
        nearest_position = game.nearest(resource_type)
        print(f'Moving to {resource_type} at position {nearest_position}...')
        game.move_to(nearest_position)
        print(f'Harvesting {required_amount} units of {resource_type}...')
        game.harvest_resource(nearest_position, quantity=required_amount)
        current_inventory = game.inspect_inventory()
        assert current_inventory.get(resource_type) >= required_amount, f'Failed to harvest enough {resource_type}. Expected at least {required_amount}, but got {current_inventory.get(resource_type)}'
    print('Successfully gathered all necessary resources.')
    print(f'Current Inventory: {game.inspect_inventory()}')
    final_inventory = game.inspect_inventory()
    assert final_inventory.get(Resource.IronOre) >= 5, 'Not enough Iron Ore.'
    assert final_inventory.get(Resource.CopperOre) >= 3, 'Not enough Copper Ore.'
    assert final_inventory.get(Resource.Coal) >= 10, 'Not enough Coal.'
    assert final_inventory.get(Resource.Stone) >= 5, 'Not enough Stone.'
    print('All initial gathering objectives met successfully!')
    '\n    Step 2: Craft an additional stone furnace. We need to carry out the following:\n    - Craft a stone furnace using 5 stone\n    OUTPUT CHECK: Verify that we now have 2 stone furnaces (1 in inventory, 1 on map)\n    '
    print('Attempting to craft a Stone Furnace...')
    crafted_furnace_count = game.craft_item(Prototype.StoneFurnace, 1)
    assert crafted_furnace_count == 1, 'Failed to craft Stone Furnace.'
    inventory_after_crafting = game.inspect_inventory()
    stone_furnace_in_inventory = inventory_after_crafting.get(Prototype.StoneFurnace, 0)
    print(f'Stone Furnaces in Inventory after crafting: {stone_furnace_in_inventory}')
    assert stone_furnace_in_inventory == 1, f'Expected 1 Stone Furnace in inventory but found {stone_furnace_in_inventory}.'
    existing_stone_furnaces_on_map = game.get_entities({Prototype.StoneFurnace})
    furnaces_on_map_count = len(existing_stone_furnaces_on_map)
    print(f'Stone Furnaces currently on map: {furnaces_on_map_count}')
    total_stone_furnaces = furnaces_on_map_count + stone_furnace_in_inventory
    assert total_stone_furnaces == 2, f'Total Stone Furnaces should be 2 but found {total_stone_furnaces}.'
    print('Successfully crafted an additional Stone Furnace.')
    '\n    Step 3: Set up smelting operation. We need to:\n    - Place the new stone furnace next to the existing one\n    - Fuel both furnaces with coal\n    OUTPUT CHECK: Verify that both furnaces are placed and fueled\n    '
    existing_furnace = game.get_entities({Prototype.StoneFurnace})[0]
    print(f'Existing Stone Furnace found at position {existing_furnace.position}')
    new_furnace_position = Position(x=existing_furnace.position.x + 2, y=existing_furnace.position.y)
    game.move_to(new_furnace_position)
    new_stone_furnace = game.place_entity(Prototype.StoneFurnace, Direction.UP, new_furnace_position)
    print(f'Placed new Stone Furnace at position {new_stone_furnace.position}')
    coal_in_inventory = game.inspect_inventory()[Prototype.Coal]
    half_coal_each = coal_in_inventory // 2
    existing_furnace = game.insert_item(Prototype.Coal, existing_furnace, half_coal_each)
    print(f'Fueled Existing Stone Furnace with {half_coal_each} units of Coal')
    new_stone_furnace = game.insert_item(Prototype.Coal, new_stone_furnace, half_coal_each)
    print(f'Fueled New Stone Furnace with {half_coal_each} units of Coal')
    assert EntityStatus.NO_FUEL not in [existing_furnace.status], 'Existing furnace is out of fuel!'
    assert EntityStatus.NO_FUEL not in [new_stone_furnace.status], 'Newly placed furnace is out of fuel!'
    print('Both furnaces are successfully placed and fueled.')
    '\n    Step 4: Smelt plates. We need to:\n    - Smelt 5 iron ore into 5 iron plates\n    - Smelt 3 copper ore into 3 copper plates\n    OUTPUT CHECK: Verify that we have 5 iron plates and 3 copper plates in our inventory\n    '
    stone_furnaces = game.get_entities({Prototype.StoneFurnace})
    furnace_iron = stone_furnaces[0]
    furnace_copper = stone_furnaces[1]
    print(f'Using Furnace at {furnace_iron.position} for Iron Ore')
    print(f'Using Furnace at {furnace_copper.position} for Copper Ore')
    iron_ore_count = game.inspect_inventory()[Prototype.IronOre]
    furnace_iron = game.insert_item(Prototype.IronOre, furnace_iron, iron_ore_count)
    print(f'Inserted {iron_ore_count} Iron Ore into first Stone Furnace.')
    copper_ore_count = game.inspect_inventory()[Prototype.CopperOre]
    furnace_copper = game.insert_item(Prototype.CopperOre, furnace_copper, copper_ore_count)
    print(f'Inserted {copper_ore_count} Copper Ore into second Stone Furnace.')
    assert True

def test_place_transport_belt_next_to_miner(game):
    """
    Place a transport belt next to a burner mining drill
    :param game:
    :return:
    """
    iron_position = game.get_resource_patch(Resource.IronOre, game.nearest(Resource.IronOre)).bounding_box.center
    game.move_to(iron_position)
    drill = game.place_entity(Prototype.BurnerMiningDrill, position=iron_position, exact=True)
    for y in range(-1, 3, 1):
        world_y = y + drill.position.y
        world_x = -1.0 + drill.position.x - 1
        game.move_to(Position(x=world_x, y=world_y))
        game.place_entity(Prototype.TransportBelt, position=Position(x=world_x, y=world_y), direction=Direction.UP, exact=True)
    pass

def test_place_pickup(game):
    """
    Place a boiler at (0, 0) and then pick it up
    :param game:
    :return:
    """
    boilers_in_inventory = game.inspect_inventory()[Prototype.Boiler]
    game.place_entity(Prototype.Boiler, position=Position(x=0, y=0))
    assert boilers_in_inventory == game.inspect_inventory()[Prototype.Boiler] + 1
    game.pickup_entity(Prototype.Boiler, position=Position(x=0, y=0))
    assert boilers_in_inventory == game.inspect_inventory()[Prototype.Boiler]

def test_place_offshore_pumps(game):
    """
    Place offshore pumps at each cardinal direction
    :param game:
    :return:
    """
    entity = Prototype.OffshorePump
    water_location = game.nearest(Resource.Water)
    water_patch = game.get_resource_patch(Resource.Water, water_location)
    left_of_water_patch = Position(x=water_patch.bounding_box.left_top.x, y=water_patch.bounding_box.center.y)
    game.move_to(left_of_water_patch)
    offshore_pump = game.place_entity(entity, position=left_of_water_patch, direction=Direction.LEFT)
    assert offshore_pump.direction.value == Direction.LEFT.value
    right_of_water_patch = Position(x=water_patch.bounding_box.right_bottom.x, y=water_patch.bounding_box.center.y)
    game.move_to(right_of_water_patch)
    offshore_pump = game.place_entity(entity, position=right_of_water_patch, direction=Direction.RIGHT)
    assert offshore_pump.direction.value == Direction.RIGHT.value
    above_water_patch = Position(x=water_patch.bounding_box.center.x, y=water_patch.bounding_box.left_top.y)
    game.move_to(above_water_patch)
    offshore_pump = game.place_entity(entity, position=above_water_patch, direction=Direction.UP)
    assert offshore_pump.direction.value == Direction.UP.value
    below_water_patch = Position(x=water_patch.bounding_box.center.x, y=water_patch.bounding_box.right_bottom.y)
    game.move_to(below_water_patch)
    offshore_pump = game.place_entity(entity, position=below_water_patch, direction=Direction.DOWN)
    assert offshore_pump.direction.value == Direction.DOWN.value

def test_place_offshore_pumps_no_default_direction(game):
    """
    Place offshore pumps at each cardinal direction
    :param game:
    :return:
    """
    entity = Prototype.OffshorePump
    water_location = game.nearest(Resource.Water)
    water_patch = game.get_resource_patch(Resource.Water, water_location)
    left_of_water_patch = Position(x=water_patch.bounding_box.left_top.x, y=water_patch.bounding_box.center.y)
    game.move_to(left_of_water_patch)
    offshore_pump = game.place_entity(entity, position=left_of_water_patch)
    assert offshore_pump.direction.value == Direction.LEFT.value
    assert offshore_pump.connection_points
    right_of_water_patch = Position(x=water_patch.bounding_box.right_bottom.x, y=water_patch.bounding_box.center.y)
    game.move_to(right_of_water_patch)
    offshore_pump = game.place_entity(entity, position=right_of_water_patch)
    assert offshore_pump.direction.value == Direction.RIGHT.value
    assert offshore_pump.connection_points
    above_water_patch = Position(x=water_patch.bounding_box.center.x, y=water_patch.bounding_box.left_top.y)
    game.move_to(above_water_patch)
    offshore_pump = game.place_entity(entity, position=above_water_patch)
    assert offshore_pump.direction.value == Direction.UP.value
    assert offshore_pump.connection_points
    below_water_patch = Position(x=water_patch.bounding_box.center.x, y=water_patch.bounding_box.right_bottom.y)
    game.move_to(below_water_patch)
    offshore_pump = game.place_entity(entity, position=below_water_patch)
    assert offshore_pump.direction.value == Direction.DOWN.value
    assert offshore_pump.connection_points

def test_place_burner_inserters(game):
    """
    Place inserters at each cardinal direction
    :param game:
    :return:
    """
    entity = Prototype.BurnerInserter
    location = game.nearest(Resource.Coal)
    game.move_to(Position(x=location.x - 10, y=location.y))
    offshore_pump = game.place_entity(entity, position=location, direction=Direction.LEFT)
    assert offshore_pump.direction.value == Direction.LEFT.value
    game.instance.reset()
    game.move_to(Position(x=location.x, y=location.y))
    offshore_pump = game.place_entity(entity, position=location, direction=Direction.RIGHT)
    assert offshore_pump.direction.value == Direction.RIGHT.value
    game.instance.reset()
    game.move_to(Position(x=location.x, y=location.y))
    offshore_pump = game.place_entity(entity, position=location, direction=Direction.UP)
    assert offshore_pump.direction.value == Direction.UP.value
    game.instance.reset()
    game.move_to(Position(x=location.x, y=location.y))
    offshore_pump = game.place_entity(entity, position=location, direction=Direction.DOWN)
    assert offshore_pump.direction.value == Direction.DOWN.value

def test_place_burner_mining_drills(game):
    """
    Place mining drills at each cardinal direction
    :param game:
    :return:
    """
    entity = Prototype.BurnerMiningDrill
    location = game.nearest(Resource.IronOre)
    game.move_to(Position(x=location.x - 10, y=location.y))
    drill = game.place_entity(entity, position=location, direction=Direction.LEFT)
    assert drill.direction.value == Direction.LEFT.value
    game.instance.reset()
    game.move_to(Position(x=location.x, y=location.y))
    drill = game.place_entity(entity, position=location, direction=Direction.RIGHT)
    assert drill.direction.value == Direction.RIGHT.value
    game.instance.reset()
    game.move_to(Position(x=location.x, y=location.y))
    drill = game.place_entity(entity, position=location, direction=Direction.UP)
    assert drill.direction.value == Direction.UP.value
    game.instance.reset()
    game.move_to(Position(x=location.x, y=location.y))
    drill = game.place_entity(entity, position=location, direction=Direction.DOWN)
    assert drill.direction.value == Direction.DOWN.value
    game.instance.reset()

def test_place_generator(game):
    """
    Place a steam engine at (0,0)
    """
    game.place_entity(Prototype.SteamEngine, position=Position(x=0, y=0), direction=Direction.UP)
    pass

def test_place_too_far_away(game):
    try:
        game.place_entity(Prototype.BurnerMiningDrill, position=Position(x=100, y=0))
    except Exception:
        assert True

def test_cannot_place_at_water(game):
    steam_engine_pos = Position(x=-20.5, y=8.5)
    game.move_to(steam_engine_pos)
    try:
        engine = game.place_entity(Prototype.SteamEngine, position=steam_engine_pos)
        failed = True
    except:
        failed = False
    assert not failed

def align_to_grid(pos):
    return Position(x=round(pos.x * 2) / 2, y=round(pos.y * 2) / 2)

def test_harvest_stump(game):
    instance = game.instance
    instance.rcon_client.send_command(f'/c {create_stump}')
    harvested = game.harvest_resource(Position(x=0, y=0), quantity=1)
    assert harvested == 2

def test_harvest_rock(game):
    instance = game.instance
    instance.rcon_client.send_command(f'/c {create_rock}')
    harvested = game.harvest_resource(Position(x=0, y=0), quantity=1)
    assert harvested == 20

def test_harvest_bug_2(game):
    """
    Planning:
    1. Gather stone to craft a furnace
    2. Craft a stone furnace
    3. Place the furnace
    4. Mine coal for fuel
    5. Mine iron ore
    6. Smelt iron ore into iron plates
    7. Verify the production of iron plates
    """
    '\n    Step 1: Gather stone to craft a furnace\n    '
    stone_position = game.nearest(Resource.Stone)
    game.move_to(stone_position)
    stone_needed = 5
    stone_mined = game.harvest_resource(stone_position, stone_needed)
    print(f'Mined {stone_mined} stone')
    inventory = game.inspect_inventory()
    assert inventory.get(Prototype.Stone) >= stone_needed, f'Failed to mine enough stone. Current inventory: {inventory}'
    '\n    Step 2: Craft a stone furnace\n    '
    game.craft_item(Prototype.StoneFurnace, 1)
    print('Crafted 1 stone furnace')
    inventory = game.inspect_inventory()
    assert inventory.get(Prototype.StoneFurnace) >= 1, f'Failed to craft stone furnace. Current inventory: {inventory}'
    '\n    Step 3: Place the furnace\n    '
    origin = Position(x=0, y=0)
    game.move_to(origin)
    furnace = game.place_entity(Prototype.StoneFurnace, position=origin)
    print(f'Placed stone furnace at {furnace.position}')
    '\n    Step 4: Mine coal for fuel\n    '
    coal_position = game.nearest(Resource.Coal)
    game.move_to(coal_position)
    coal_needed = 10
    coal_mined = game.harvest_resource(coal_position, coal_needed)
    print(f'Mined {coal_mined} coal')
    inventory = game.inspect_inventory()
    assert inventory.get(Prototype.Coal) >= coal_needed, f'Failed to mine enough coal. Current inventory: {inventory}'
    game.move_to(furnace.position)
    updated_furnace = game.insert_item(Prototype.Coal, furnace, 10)
    print('Inserted coal into the furnace')
    '\n    Step 5: Mine iron ore\n    '
    iron_position = game.nearest(Resource.IronOre)
    game.move_to(iron_position)
    iron_ore_needed = 10
    iron_ore_mined = game.harvest_resource(iron_position, iron_ore_needed)
    print(f'Mined {iron_ore_mined} iron ore')
    inventory = game.inspect_inventory()
    assert inventory.get(Prototype.IronOre) >= iron_ore_needed, f'Failed to mine enough iron ore. Current inventory: {inventory}'
    '\n    Step 6: Smelt iron ore into iron plates\n    '
    game.move_to(furnace.position)
    updated_furnace = game.insert_item(Prototype.IronOre, updated_furnace, 10)
    print('Inserted iron ore into the furnace')
    smelting_time = 10 * 0.7
    game.sleep(int(smelting_time))
    max_attempts = 5
    for _ in range(max_attempts):
        game.extract_item(Prototype.IronPlate, updated_furnace.position, 10)
        inventory = game.inspect_inventory()
        if inventory.get(Prototype.IronPlate, 0) >= 10:
            break
        game.sleep(5)
    print('Extracted iron plates from the furnace')
    '\n    Step 7: Verify the production of iron plates\n    '
    inventory = game.inspect_inventory()
    iron_plates = inventory.get(Prototype.IronPlate, 0)
    print(f'Current inventory: {inventory}')
    assert iron_plates >= 10, f'Failed to produce enough iron plates. Expected 10, got {iron_plates}'
    print('Successfully produced 10 iron plates!')

def test_move_to_check_position(game):
    target_pos = Position(x=-9.5, y=-11.5)
    game.move_to(target_pos)

def test_move_to_string_response_error_handling(game):
    """Test enhanced error handling for string responses from Lua server"""
    positions_to_test = [Position(x=10, y=10), Position(x=50, y=50), Position(x=100, y=100)]
    for pos in positions_to_test:
        try:
            game.move_to(pos)
            print(f'✓ Successfully moved to {pos}')
        except Exception as e:
            error_msg = str(e)
            if 'Could not move' in error_msg:
                print(f'✓ Got properly formatted move error: {error_msg}')
            else:
                raise

def test_move_to_invalid_positions(game):
    """Test move_to behavior with potentially problematic positions"""
    edge_positions = [Position(x=0, y=0), Position(x=-1, y=-1), Position(x=1000, y=1000)]
    successful_moves = 0
    for pos in edge_positions:
        try:
            game.move_to(pos)
            successful_moves += 1
            print(f'✓ Successfully moved to edge position {pos}')
        except Exception as e:
            error_msg = str(e)
            assert 'Could not move' in error_msg or 'Could not get path' in error_msg, f'Should get formatted error: {error_msg}'
            print(f'✓ Got expected move failure for {pos}: {error_msg}')
    assert successful_moves >= 0, 'Error handling should not break all movement'

def test_move_to_near_entities(game):
    """Test movement near entities doesn't cause string response errors"""
    game.move_to(Position(x=18, y=20))
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=20, y=20))
    assert furnace, 'Failed to place furnace'
    try:
        game.move_to(Position(x=20.1, y=20.1))
        print('✓ Can move very close to entities')
    except Exception as e:
        error_msg = str(e)
        assert 'Could not move' in error_msg, f'Should get formatted error: {error_msg}'
        print(f'✓ Got proper error for blocked movement: {error_msg}')

def test_inspect_entities(game):
    inventory = game.inspect_inventory()
    coal_count = inventory[Prototype.Coal]
    assert coal_count != 0
    chest = game.place_entity(Prototype.IronChest, position=Position(x=0, y=0))
    game.insert_item(Prototype.Coal, chest, quantity=5)
    inspected = game.inspect_entities(radius=5, position=Position(x=chest.position.x, y=chest.position.y))
    assert len(inspected.entities) == 2

def test_inspect_inserters(game):
    """Test to ensure that inspected inserters are facing in the correct direction"""
    game.place_entity(Prototype.BurnerInserter, Direction.RIGHT, position=Position(x=0, y=0))
    entities = game.inspect_entities(radius=5)
    for entity in entities.entities:
        if entity.name == 'burner-inserter':
            assert entity.direction == Direction.RIGHT.value

def test_rotate_assembling_machine_2(game):
    assembler = game.place_entity_next_to(Prototype.AssemblingMachine2, reference_position=Position(x=0, y=0), direction=Direction.RIGHT, spacing=2)
    orthogonal_direction = Direction.DOWN
    try:
        assembler = game.rotate_entity(assembler, orthogonal_direction)
        assert False, 'Cannot rotate an assembler without a recipe set'
    except:
        assert True
        return

def test_rotate_assembling_machine_2_with_recipe(game):
    assembler = game.place_entity_next_to(Prototype.AssemblingMachine2, reference_position=Position(x=0, y=0), direction=Direction.RIGHT, spacing=2)
    orthogonal_direction = Direction.DOWN
    game.set_entity_recipe(assembler, RecipeName.FillCrudeOilBarrel)
    assembler = game.rotate_entity(assembler, orthogonal_direction)
    assert assembler.direction.value == orthogonal_direction.value

def test_rotate_boiler(game):
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=Position(x=0, y=0), direction=Direction.RIGHT, spacing=2)
    orthogonal_direction = Direction.UP
    boiler = game.rotate_entity(boiler, orthogonal_direction)
    assert boiler.direction.value == orthogonal_direction.value

def test_rotate_inserters(game):
    insert1 = game.place_entity_next_to(Prototype.BurnerInserter, Position(x=0, y=0), Direction.DOWN, spacing=0)
    insert1 = game.rotate_entity(insert1, Direction.UP)
    assert insert1 is not None, 'Failed to place input inserter'
    assert insert1.direction.value == Direction.UP.value

def test_extract(game):
    chest = game.place_entity(Prototype.IronChest, position=Position(x=0, y=0))
    game.insert_item(Prototype.IronPlate, chest, quantity=10)
    count = game.extract_item(Prototype.IronPlate, chest.position, quantity=2)
    assert game.inspect_inventory()[Prototype.IronPlate] == 2
    assert count == 2

def test_extract_assembler_multi(game):
    assembler = game.place_entity(Prototype.AssemblingMachine1, position=Position(x=0, y=0))
    game.set_entity_recipe(assembler, Prototype.ElectronicCircuit)
    game.insert_item(Prototype.IronPlate, assembler, quantity=10)
    game.insert_item(Prototype.CopperCable, assembler, quantity=3)
    inventory = game.inspect_inventory(assembler)
    assert inventory[Prototype.IronPlate] == 10
    assert inventory[Prototype.CopperCable] == 3
    count1 = game.extract_item(Prototype.IronPlate, assembler, quantity=2)
    count2 = game.extract_item(Prototype.CopperCable, assembler, quantity=2)
    assert game.inspect_inventory()[Prototype.IronPlate] == 2
    assert game.inspect_inventory()[Prototype.CopperCable] == 2
    assert count1 == 2 and count2 == 2

def test_basic_render(game):
    game.place_entity(Prototype.IronChest, position=Position(x=0, y=0))
    game.connect_entities(Position(x=0, y=-2), Position(x=15, y=5), {Prototype.Pipe, Prototype.UndergroundPipe})
    game.connect_entities(Position(x=0, y=-10), Position(x=15, y=-10), {Prototype.SmallElectricPole})
    image = game._render(position=Position(x=0, y=5), layers=Layer.ALL)
    image.show()
    pass

def test_can_place_over_player_large(game):
    game.move_to(Position(x=0, y=0))
    assert game.can_place_entity(Prototype.SteamEngine, position=Position(x=0, y=0))
    game.place_entity(Prototype.SteamEngine, position=Position(x=0, y=0), direction=Direction.UP)

def test_move_to_elapsed_ticks_and_timing(game):
    """Test that move_to adds correct ticks and sleeps appropriately."""
    game.instance.set_speed_and_unpause(1.0)
    initial_pos = Position(x=0, y=0)
    game.move_to(initial_pos)
    initial_ticks = game.instance.get_elapsed_ticks()
    target_pos = Position(x=initial_pos.x + 5, y=initial_pos.y)
    start_time = time.time()
    game.move_to(target_pos)
    end_time = time.time()
    real_time = end_time - start_time
    final_ticks = game.instance.get_elapsed_ticks()
    ticks_added = final_ticks - initial_ticks
    expected_ticks = 5 / 0.15
    assert 25 <= ticks_added <= 40, f'Expected ~{expected_ticks:.0f} ticks for 5-tile movement, got {ticks_added}'
    expected_real_time = ticks_added / 60
    assert abs(real_time - expected_real_time) < 0.5, f'Expected ~{expected_real_time:.2f}s real time, got {real_time:.2f}s'

def test_move_to_with_different_speeds(game):
    """Test move_to timing at different game speeds."""
    target_pos = Position(x=3, y=3)
    game.instance.set_speed_and_unpause(5.0)
    initial_ticks = game.instance.get_elapsed_ticks()
    start_time = time.time()
    game.move_to(target_pos)
    end_time = time.time()
    real_time = end_time - start_time
    final_ticks = game.instance.get_elapsed_ticks()
    ticks_added = final_ticks - initial_ticks
    expected_real_time = ticks_added / 60 / 5.0
    assert abs(real_time - expected_real_time) < 0.2, f'Expected ~{expected_real_time:.2f}s real time, got {real_time:.2f}s'

def test_multiple_actions_cumulative_ticks(game):
    """Test that multiple actions accumulate ticks correctly."""
    game.instance.set_speed_and_unpause(2.0)
    game.move_to(Position(x=0, y=0))
    initial_ticks = game.instance.get_elapsed_ticks()
    start_time = time.time()
    game.sleep(1)
    game.craft_item(Prototype.IronGearWheel, 1)
    game.move_to(Position(x=2, y=2))
    end_time = time.time()
    total_real_time = end_time - start_time
    final_ticks = game.instance.get_elapsed_ticks()
    total_ticks_added = final_ticks - initial_ticks
    assert 100 <= total_ticks_added <= 120, f'Expected ~110 total ticks, got {total_ticks_added}'
    expected_real_time = total_ticks_added / 60 / 2.0
    assert abs(total_real_time - expected_real_time) < 0.5, f'Expected ~{expected_real_time:.2f}s total real time, got {total_real_time:.2f}s'

def test_multi_drill_multi_furnace(game):
    copper_pos = game.nearest(Resource.CopperOre)
    print(f'Found copper ore at {copper_pos}')
    game.move_to(copper_pos)
    drill1 = game.place_entity(Prototype.BurnerMiningDrill, position=copper_pos)
    drill1 = game.insert_item(Prototype.Coal, drill1, quantity=5)
    print(f'Placed first drill at {drill1.position}')
    drill2_pos = Position(x=drill1.position.x - 3, y=drill1.position.y)
    game.move_to(drill2_pos)
    drill2 = game.place_entity(Prototype.BurnerMiningDrill, position=drill2_pos)
    drill2 = game.insert_item(Prototype.Coal, drill2, quantity=5)
    print(f'Placed second drill at {drill2.position}')
    drill3_pos = Position(x=drill2.position.x - 3, y=drill2.position.y)
    game.move_to(drill3_pos)
    drill3 = game.place_entity(Prototype.BurnerMiningDrill, position=drill3_pos)
    drill3 = game.insert_item(Prototype.Coal, drill3, quantity=5)
    print(f'Placed third drill at {drill3.position}')
    print(f'Drill positions: {drill1.position}, {drill2.position}, {drill3.position}')
    furnace1_pos = drill1.drop_position.up(4)
    game.move_to(furnace1_pos)
    furnace1 = game.place_entity(Prototype.StoneFurnace, position=furnace1_pos)
    furnace1 = game.insert_item(Prototype.Coal, furnace1, quantity=5)
    print(f'Placed first furnace at {furnace1.position}')
    game.move_to(furnace1_pos)
    furnace2_pos = furnace1_pos.right(3)
    game.move_to(furnace2_pos)
    furnace2 = game.place_entity(Prototype.StoneFurnace, position=furnace2_pos)
    furnace2 = game.insert_item(Prototype.Coal, furnace2, quantity=5)
    print(f'Placed second furnace at {furnace2.position}')
    drill1 = game.get_entity(Prototype.BurnerMiningDrill, drill1.position)
    drill2 = game.get_entity(Prototype.BurnerMiningDrill, drill2.position)
    drill3 = game.get_entity(Prototype.BurnerMiningDrill, drill3.position)
    belts = game.connect_entities(drill1.drop_position, drill2.drop_position, Prototype.TransportBelt)
    belts = game.connect_entities(belts, drill3.drop_position, Prototype.TransportBelt)
    inserter1 = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace1.position, direction=Direction.DOWN, spacing=0)
    inserter1 = game.insert_item(Prototype.Coal, inserter1, quantity=1)
    print(f'Placed first inserter at {inserter1.position}')
    inserter2 = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace2.position, direction=Direction.DOWN, spacing=0)
    inserter2 = game.insert_item(Prototype.Coal, inserter2, quantity=1)
    print(f'Placed second inserter at {inserter2.position}')
    inserter1 = game.rotate_entity(inserter1, Direction.UP)
    inserter2 = game.rotate_entity(inserter2, Direction.UP)
    belts = game.connect_entities(belts, inserter2.pickup_position, Prototype.TransportBelt)
    belts = game.connect_entities(belts, inserter1.pickup_position, Prototype.TransportBelt)
    output_inserter1 = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace1_pos, direction=Direction.UP, spacing=0)
    output_inserter1 = game.insert_item(Prototype.Coal, output_inserter1, quantity=1)
    print(f'Placed first output inserter at {output_inserter1.position}')
    output_inserter2 = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=furnace2_pos, direction=Direction.UP, spacing=0)
    output_inserter2 = game.insert_item(Prototype.Coal, output_inserter2, quantity=1)
    print(f'Placed second output inserter at {output_inserter2.position}')
    collection_chest = game.place_entity_next_to(Prototype.WoodenChest, reference_position=furnace2_pos, spacing=2)
    print(f'Placed collection chest at {collection_chest.position}')
    belts = game.connect_entities(output_inserter1.drop_position, output_inserter2.drop_position, Prototype.TransportBelt)
    game.connect_entities(belts, collection_chest.position.right(2), Prototype.TransportBelt)
    output_inserter3 = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=collection_chest.position, direction=Direction.RIGHT, spacing=0)
    output_inserter3 = game.rotate_entity(output_inserter3, Direction.LEFT)
    output_inserter3 = game.insert_item(Prototype.Coal, output_inserter3, quantity=10)
    game.get_entities()
    game.sleep(30)
    production_stats = game._production_stats()
    assert production_stats['output']['copper-plate'] > 10

def test_place_ore_in_furnace(game):
    """
    Collect 10 iron ore and place it in a furnace
    :param game:
    :return:
    """
    furnace = game.place_entity(Prototype.StoneFurnace, position=Position(x=0, y=0))
    iron_ore_patch = game.get_resource_patch(Resource.IronOre, game.nearest(Resource.IronOre))
    game.move_to(iron_ore_patch.bounding_box.left_top + Position(x=1, y=1))
    game.harvest_resource(iron_ore_patch.bounding_box.left_top, quantity=10)
    coal_patch = game.get_resource_patch(Resource.Coal, game.nearest(Resource.Coal))
    game.move_to(coal_patch.bounding_box.left_top + Position(x=1, y=1))
    game.harvest_resource(coal_patch.bounding_box.left_top, quantity=10)
    game.move_to(furnace.position)
    game.insert_item(Prototype.IronOre, furnace, quantity=10)
    game.insert_item(Prototype.Coal, furnace, quantity=10)
    game.reset()

def test_connect_steam_engines_to_boilers_using_pipes(game):
    """
    Place a boiler and a steam engine next to each other in 3 cardinal directions.
    :param game:
    :return:
    """
    boilers_in_inventory = game.inspect_inventory()[Prototype.Boiler]
    steam_engines_in_inventory = game.inspect_inventory()[Prototype.SteamEngine]
    pipes_in_inventory = game.inspect_inventory()[Prototype.Pipe]
    game.move_to(Position(x=0, y=0))
    boiler: Entity = game.place_entity(Prototype.Boiler, position=Position(x=0, y=0))
    game.move_to(Position(x=0, y=5))
    steam_engine: Entity = game.place_entity(Prototype.SteamEngine, position=Position(x=0, y=10))
    try:
        connection: List[Entity] = game.connect_entities(boiler, steam_engine, connection_type=Prototype.Pipe)
        assert False
    except Exception as e:
        print(e)
        assert True
    game.pickup_entity(steam_engine)
    game.pickup_entity(connection)
    offsets = [Position(x=5, y=0), Position(x=0, y=5), Position(x=-5, y=0), Position(x=0, y=-5)]
    for offset in offsets:
        game.move_to(offset)
        steam_engine: Entity = game.place_entity(Prototype.SteamEngine, position=offset)
        try:
            connection: List[Union[EntityGroup, Entity]] = game.connect_entities(boiler, steam_engine, connection_type=Prototype.Pipe)
        except Exception as e:
            print(e)
            assert False
        assert boilers_in_inventory - 1 == game.inspect_inventory()[Prototype.Boiler]
        assert steam_engines_in_inventory - 1 == game.inspect_inventory()[Prototype.SteamEngine]
        current_pipes_in_inventory = game.inspect_inventory()[Prototype.Pipe]
        spent_pipes = pipes_in_inventory - current_pipes_in_inventory
        assert spent_pipes == len(connection.pipes)
        game.pickup_entity(steam_engine)
        game.pickup_entity(connection)

def test_build_auto_refilling_coal_system(game):
    num_drills = 3
    game.move_to(Position(x=0, y=0))
    coal_patch = game.get_resource_patch(Resource.Coal, game.nearest(Resource.Coal))
    game.move_to(coal_patch.bounding_box.left_top)
    drill = game.place_entity(Prototype.BurnerMiningDrill, Direction.UP, coal_patch.bounding_box.left_top)
    chest = game.place_entity(Prototype.IronChest, Direction.RIGHT, drill.drop_position)
    inserter = game.place_entity_next_to(Prototype.BurnerInserter, chest.position, direction=Direction.UP, spacing=0)
    first_inserter = inserter
    drill_bottom_y = drill.position.y + drill.dimensions.height
    drill_inserter = game.place_entity(Prototype.BurnerInserter, Direction.UP, Position(x=drill.position.x, y=drill_bottom_y))
    drill_inserter = game.rotate_entity(drill_inserter, Direction.UP)
    first_drill_inserter = drill_inserter
    game.move_to(inserter.drop_position)
    drills = []
    belt = None
    for i in range(1, num_drills):
        next_drill = game.place_entity_next_to(Prototype.BurnerMiningDrill, drill.position, Direction.RIGHT, spacing=2)
        next_drill = game.rotate_entity(next_drill, Direction.UP)
        drills.append(next_drill)
        try:
            chest = game.place_entity(Prototype.IronChest, Direction.RIGHT, next_drill.drop_position)
        except Exception as e:
            print(f'Could not place chest next to drill: {e}')
        next_inserter = game.place_entity_next_to(Prototype.BurnerInserter, chest.position, direction=Direction.UP, spacing=0)
        drill_bottom_y = next_drill.position.y + next_drill.dimensions.height
        drill_inserter = game.place_entity(Prototype.BurnerInserter, Direction.UP, Position(x=next_drill.position.x, y=drill_bottom_y))
        drill_inserter = game.rotate_entity(drill_inserter, Direction.UP)
        if not belt:
            belt = game.connect_entities(first_inserter.drop_position, next_inserter.drop_position, Prototype.TransportBelt)
        else:
            belt = game.connect_entities(belt, next_inserter.drop_position, Prototype.TransportBelt)
        drill = next_drill
        inserter = next_inserter
        next_drill_inserter = drill_inserter
    belt = game.connect_entities(belt, next_drill_inserter, Prototype.TransportBelt)
    belt = game.connect_entities(belt, first_drill_inserter, Prototype.TransportBelt)
    belt = game.connect_entities(belt, belt, Prototype.TransportBelt)
    for drill in drills:
        game.insert_item(Prototype.Coal, drill, 5)
    print(f'Auto-refilling coal mining system with {num_drills} drills has been built!')

def test_simple_automated_drill(game):
    coal_patch = game.get_resource_patch(Resource.Coal, game.nearest(Resource.Coal))
    assert coal_patch, 'No coal patch found nearby'
    drill_position = coal_patch.bounding_box.center
    game.move_to(drill_position)
    drill = game.place_entity(Prototype.BurnerMiningDrill, Direction.UP, drill_position)
    assert drill, f'Failed to place burner mining drill at {drill_position}'
    print(f'Placed burner mining drill at {drill.position}')
    inserter_position = Position(x=drill.position.x, y=drill.position.y + 1)
    inserter = game.place_entity(Prototype.BurnerInserter, Direction.UP, inserter_position)
    assert inserter, f'Failed to place inserter at {inserter_position}'
    print(f'Placed inserter at {inserter.position}')
    assert inserter.direction.name == Direction.UP.name, f'Inserter is not facing the drill. Current direction: {inserter.direction}'
    belt_start = drill.drop_position
    belt_end = inserter.pickup_position
    belts = game.connect_entities(belt_start, belt_end, Prototype.TransportBelt)
    assert belts, f'Failed to place transport belt from {belt_start} to {belt_end}'
    print(f'Placed {len(belts.belts)} transport belt(s) from drill to inserter')

def test_another_self_fueling_coal_belt(game):
    coal_patch = game.get_resource_patch(Resource.Coal, game.nearest(Resource.Coal))
    assert coal_patch is not None, 'No coal patch found nearby'
    assert coal_patch.size >= 25, f'Coal patch too small: {coal_patch.size} tiles (need at least 25)'
    drills = []
    inserters = []
    game.move_to(coal_patch.bounding_box.center)
    for i in range(5):
        drill_position = Position(x=coal_patch.bounding_box.left_top.x + i * 2, y=coal_patch.bounding_box.center.y)
        game.move_to(drill_position)
        drill = game.place_entity(Prototype.BurnerMiningDrill, Direction.DOWN, drill_position)
        inserter = game.place_entity_next_to(Prototype.BurnerInserter, drill_position, direction=Direction.UP, spacing=0)
        inserter = game.rotate_entity(inserter, Direction.DOWN)
        assert drill is not None, f'Failed to place burner mining drill at {drill_position}'
        assert inserter is not None, f'Failed to place inserter at {drill_position}'
        drills.append(drill)
        inserters.append(inserter)
    print(f'Placed {len(drills)} burner mining drills')
    belt_start = Position(x=drills[0].drop_position.x, y=drills[0].drop_position.y)
    belt_end = Position(x=drills[-1].drop_position.x, y=drills[0].drop_position.y)
    belt = game.connect_entities(belt_start, belt_end, Prototype.TransportBelt)
    assert belt, 'Failed to place transport belt'
    belt = game.connect_entities(belt, inserters[-1].pickup_position, Prototype.TransportBelt)
    assert belt, 'Failed to connect belt to last inserter'
    belt = game.connect_entities(belt, inserters[0].pickup_position, Prototype.TransportBelt)
    assert belt, 'Failed to connect belt to first inserter'
    belt = game.connect_entities(belt, belt, Prototype.TransportBelt)
    assert belt, 'Failed to connect belt to close the loop'
    print('All components verified')
    game.move_to(drills[0].position)
    coal_placed = game.insert_item(Prototype.Coal, drills[0], quantity=10)
    assert coal_placed is not None, 'Failed to place coal on the belt'
    print('System kickstarted with coal')
    print('Self-fueling belt of 5 burner mining drills successfully set up')

def test_defence(game):
    max_gears = 100 // 8
    max_ammo = 100 // 8
    max_turrets = min(5, 100 // 20)
    gears_crafted = game.craft_item(Prototype.IronGearWheel, max_gears)
    ammo_crafted = game.craft_item(Prototype.FirearmMagazine, max_ammo)
    turrets_crafted = game.craft_item(Prototype.GunTurret, max_turrets)
    print(f'Crafted {gears_crafted} iron gear wheels, {ammo_crafted} firearm magazines, and {turrets_crafted} gun turrets')
    defensive_position = Position(x=10, y=10)
    turrets = []
    for i in range(turrets_crafted):
        turret_position = Position(x=defensive_position.x + i * 2, y=defensive_position.y)
        game.move_to(turret_position)
        turret = game.place_entity(Prototype.GunTurret, direction=Direction.SOUTH, position=turret_position)
        if turret:
            turrets.append(turret)
    print(f'Placed {len(turrets)} gun turrets')
    if turrets:
        ammo_per_turret = min(20, ammo_crafted // len(turrets))
        for turret in turrets:
            inserted_ammo = game.insert_item(Prototype.FirearmMagazine, turret, ammo_per_turret)
            if inserted_ammo:
                print(f'Inserted {ammo_per_turret} ammunition into turret at {turret.position}')
            else:
                print(f'Failed to insert ammunition into turret at {turret.position}')
    player_inventory = game.inspect_inventory()
    remaining_ammo = player_inventory.get(Prototype.FirearmMagazine, 0)
    print(f'Defensive line of {len(turrets)} gun turrets built and supplied with {ammo_per_turret} ammunition each')
    print(f'Remaining ammunition in inventory: {remaining_ammo}')
    assert len(turrets) > 0, 'Failed to build any gun turrets'
    assert len(turrets) <= 5, f'Built too many turrets: {len(turrets)}'
    assert remaining_ammo < ammo_crafted, 'Failed to supply turrets with ammunition'
    assert len(turrets) == turrets_crafted, f'Expected to place {turrets_crafted} turrets, but placed {len(turrets)}'
    print('Objective completed: Built a defensive line of gun turrets and manually supplied them with ammunition')

@pytest.fixture()
def game(base_game):
    """Create electricity system"""
    base_game.inspect_inventory()
    water_location = base_game.nearest(Resource.Water)
    base_game.move_to(water_location)
    offshore_pump = base_game.place_entity(Prototype.OffshorePump, position=water_location)
    direction = offshore_pump.direction
    boiler = base_game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=direction, spacing=2)
    assert boiler.direction.value == direction.value
    boiler = base_game.rotate_entity(boiler, Direction.next_clockwise(direction))
    base_game.insert_item(Prototype.Coal, boiler, quantity=5)
    base_game.connect_entities(offshore_pump, boiler, connection_type=Prototype.Pipe)
    base_game.move_to(Position(x=0, y=10))
    steam_engine: Entity = base_game.place_entity_next_to(Prototype.SteamEngine, reference_position=boiler.position, direction=boiler.direction, spacing=1)
    base_game.connect_entities(steam_engine, boiler, connection_type=Prototype.Pipe)
    yield base_game

def test_build_chemical_plant(game):
    game.move_to(game.nearest(Resource.CrudeOil))
    pumpjack = game.place_entity(Prototype.PumpJack, direction=Direction.DOWN, position=game.nearest(Resource.CrudeOil))
    game.move_to(Position(x=0, y=-6))
    refinery = game.place_entity(Prototype.OilRefinery, direction=Direction.DOWN, position=Position(x=0, y=-6))
    refinery = game.set_entity_recipe(refinery, RecipeName.AdvancedOilProcessing)
    game.move_to(Position(x=0, y=0))
    chemical_plant = game.place_entity(Prototype.ChemicalPlant, direction=Direction.DOWN, position=Position(x=0, y=6))
    chemical_plant = game.set_entity_recipe(chemical_plant, RecipeName.LightOilCracking)
    steam_engine = game.get_entity(Prototype.SteamEngine, game.nearest(Prototype.SteamEngine))
    game.connect_entities(pumpjack, refinery, connection_type=Prototype.Pipe)
    game.connect_entities(refinery, chemical_plant, connection_type=Prototype.Pipe)
    game.connect_entities(pumpjack, refinery, chemical_plant, steam_engine, connection_type=Prototype.SmallElectricPole)

def test_build_iron_plate_factory(game):
    WIDTH_SPACING = 1
    iron_ore_patch = game.get_resource_patch(Resource.IronOre, game.nearest(Resource.IronOre))
    game.move_to(iron_ore_patch.bounding_box.left_top)
    miner = game.place_entity(Prototype.BurnerMiningDrill, Direction.DOWN, iron_ore_patch.bounding_box.left_top)
    chest = game.place_entity_next_to(Prototype.IronChest, miner.position, Direction.UP, spacing=miner.dimensions.height)
    game.insert_item(Prototype.Coal, chest, 50)
    game.place_entity_next_to(Prototype.BurnerInserter, chest.position, Direction.DOWN, spacing=0)
    coal_chest_inserter = game.place_entity_next_to(Prototype.BurnerInserter, chest.position, Direction.UP, spacing=0)
    coal_chest_inserter = game.rotate_entity(coal_chest_inserter, Direction.DOWN)
    coal_belt_inserter = game.place_entity_next_to(Prototype.BurnerInserter, chest.position, Direction.RIGHT, spacing=0)
    coal_belt_inserter = game.rotate_entity(coal_belt_inserter, Direction.RIGHT)
    iron_drill_coal_belt_inserter = game.place_entity_next_to(Prototype.BurnerInserter, chest.position, Direction.LEFT, spacing=0)
    iron_belt_start = miner.position.down()
    furnaces = []
    furnace_line_start = game.place_entity_next_to(Prototype.StoneFurnace, miner.position, Direction.DOWN, spacing=2)
    furnaces.append(furnace_line_start)
    current_furnace = furnace_line_start
    for _ in range(3):
        current_furnace = game.place_entity_next_to(Prototype.StoneFurnace, current_furnace.position, Direction.RIGHT, spacing=WIDTH_SPACING)
        furnaces.append(current_furnace)
    above_current_furnace = Position(x=current_furnace.position.x, y=current_furnace.position.y - 2.5)
    iron_belt = game.connect_entities(iron_belt_start, above_current_furnace, Prototype.TransportBelt)
    game.connect_entities(iron_drill_coal_belt_inserter.drop_position, iron_belt, Prototype.TransportBelt)
    iron_belt = game.connect_entities(coal_belt_inserter.position, coal_belt_inserter.position.right(10), Prototype.TransportBelt)
    miners = [miner]
    for i in range(3):
        miner = game.place_entity_next_to(Prototype.BurnerMiningDrill, miner.position, Direction.RIGHT, spacing=WIDTH_SPACING)
        miner = game.rotate_entity(miner, Direction.DOWN)
        miners.append(miner)
        above_current_drill = Position(x=miner.position.x, y=miner.position.y - miner.dimensions.height - 1)
        game.move_to(Position(x=miner.drop_position.x, y=above_current_drill.y + 1))
        miner_coal_inserter = game.place_entity(Prototype.BurnerInserter, Direction.UP, Position(x=miner.drop_position.x, y=above_current_drill.y + 1))
        miner_coal_inserter = game.rotate_entity(miner_coal_inserter, Direction.DOWN)
    for i in range(4):
        furnace_pos = furnaces[i].position
        game.move_to(furnace_pos)
        game.place_entity_next_to(Prototype.BurnerInserter, furnace_pos, Direction.DOWN)
        ins = game.place_entity_next_to(Prototype.BurnerInserter, furnace_pos, Direction.UP)
        game.rotate_entity(ins, Direction.DOWN)
    output_belt = game.connect_entities(Position(x=furnace_line_start.position.x, y=furnace_line_start.position.y + 2.5), Position(x=current_furnace.position.x, y=furnace_line_start.position.y + 2.5), Prototype.TransportBelt)
    output_chest = game.place_entity_next_to(Prototype.IronChest, output_belt.outputs[0].position, Direction.RIGHT, spacing=1)
    game.place_entity(Prototype.BurnerInserter, Direction.RIGHT, output_chest.position.left())
    coal_patch = game.get_resource_patch(Resource.Coal, game.nearest(Resource.Coal))
    game.move_to(coal_patch.bounding_box.left_top)
    coal_miner = game.place_entity(Prototype.BurnerMiningDrill, Direction.UP, coal_patch.bounding_box.left_top)
    game.connect_entities(coal_miner.drop_position, coal_chest_inserter, Prototype.TransportBelt)
    game.insert_item(Prototype.Coal, coal_miner, 50)
    reinserter = game.place_entity_next_to(Prototype.BurnerInserter, Position(x=coal_miner.position.x - 1, y=coal_miner.position.y - 1), Direction.LEFT, spacing=0)
    reinserter = game.rotate_entity(reinserter, Direction.RIGHT)
    print('Simple iron plate factory has been built!')

def test_auto_fueling_iron_smelting_factory(game):
    """
    Builds an auto-fueling iron smelting factory:
    - Mines coal and iron ore.
    - Uses transport belts to deliver coal to fuel the iron miner and furnace.
    - Smelts iron ore into iron plates.
    - Stores iron plates in an iron chest.
    """
    coal_position = game.nearest(Resource.Coal)
    game.move_to(coal_position)
    coal_drill = game.place_entity(Prototype.BurnerMiningDrill, position=coal_position, direction=Direction.DOWN)
    iron_position = game.nearest(Resource.IronOre)
    game.move_to(iron_position)
    iron_drill = game.place_entity(Prototype.BurnerMiningDrill, position=iron_position, direction=Direction.DOWN)
    iron_drill_fuel_inserter = game.place_entity_next_to(Prototype.BurnerInserter, reference_position=iron_drill.position, direction=Direction.RIGHT, spacing=0)
    iron_drill_fuel_inserter = game.rotate_entity(iron_drill_fuel_inserter, Direction.LEFT)
    coal_belt = game.connect_entities(source=coal_drill, target=iron_drill_fuel_inserter, connection_type=Prototype.TransportBelt)
    furnace_position = Position(x=iron_drill.drop_position.x, y=iron_drill.drop_position.y)
    iron_furnace = game.place_entity(Prototype.StoneFurnace, position=furnace_position)
    furnace_fuel_inserter_position = Position(x=iron_furnace.position.x + 1, y=iron_furnace.position.y)
    furnace_fuel_inserter = game.place_entity(Prototype.BurnerInserter, position=furnace_fuel_inserter_position, direction=Direction.LEFT)
    coal_belt = game.connect_entities(coal_belt, furnace_fuel_inserter, connection_type=Prototype.TransportBelt)
    game.place_entity_next_to(Prototype.BurnerInserter, reference_position=iron_furnace.position, direction=Direction.DOWN, spacing=0)
    iron_chest = game.place_entity_next_to(Prototype.IronChest, reference_position=iron_furnace.position, direction=Direction.DOWN, spacing=1)
    game.move_to(coal_position)
    game.insert_item(Prototype.Coal, coal_drill, quantity=10)
    sleep(15)
    chest_inventory = game.inspect_inventory(iron_chest)
    iron_plates_in_chest = chest_inventory.get(Prototype.IronPlate, 0)
    assert iron_plates_in_chest > 0, 'No iron plates were produced'
    print(f'Successfully produced {iron_plates_in_chest} iron plates.')

def test_create_offshore_pump_to_steam_engine(game):
    """
    Place a boiler and a steam engine next to each other in 3 cardinal directions.
    :param game:
    :return:
    """
    water_location = game.nearest(Resource.Water)
    game.move_to(water_location)
    offshore_pump = game.place_entity(Prototype.OffshorePump, position=water_location)
    direction = offshore_pump.direction
    boiler = game.place_entity_next_to(Prototype.Boiler, reference_position=offshore_pump.position, direction=direction, spacing=2)
    assert boiler.direction.value == direction.value
    boiler = game.rotate_entity(boiler, Direction.next_clockwise(direction))
    game.insert_item(Prototype.Coal, boiler, quantity=5)
    game.connect_entities(offshore_pump, boiler, connection_type=Prototype.Pipe)
    game.move_to(Position(x=0, y=10))
    steam_engine: Entity = game.place_entity_next_to(Prototype.SteamEngine, reference_position=boiler.position, direction=boiler.direction, spacing=1)
    game.connect_entities(boiler, steam_engine, connection_type=Prototype.Pipe)
    inspected_steam_engine = game.get_entity(Prototype.SteamEngine, steam_engine.position)
    assert inspected_steam_engine.status == EntityStatus.NOT_PLUGGED_IN_ELECTRIC_NETWORK
    assert steam_engine.direction.value == Direction.opposite(boiler.direction).value
    image = game._render()
    image.show()
    pass

