# Cluster 64

class ConnectEntities(Tool):

    def __init__(self, connection, game_state):
        super().__init__(connection, game_state)
        self._setup_actions()
        self._setup_resolvers()

    def _setup_actions(self):
        self.request_path = RequestPath(self.connection, self.game_state)
        self.get_path = GetPath(self.connection, self.game_state)
        self.rotate_entity = RotateEntity(self.connection, self.game_state)
        self.pickup_entity = PickupEntity(self.connection, self.game_state)
        self.inspect_inventory = InspectInventory(self.connection, self.game_state)
        self.get_entities = GetEntities(self.connection, self.game_state)
        self.get_entity = GetEntity(self.connection, self.game_state)
        self._extend_collision_boxes = ExtendCollisionBoxes(self.connection, self.game_state)
        self._clear_collision_boxes = ClearCollisionBoxes(self.connection, self.game_state)

    def _setup_resolvers(self):
        self.resolvers = {ConnectionType.FLUID: FluidConnectionResolver(self.get_entities), ConnectionType.TRANSPORT: TransportConnectionResolver(self.get_entities), ConnectionType.POWER: PowerConnectionResolver(self.get_entities), ConnectionType.WALL: Resolver(self.get_entities)}

    def _get_connection_type(self, prototype: Prototype) -> ConnectionType:
        match prototype:
            case Prototype.Pipe | Prototype.UndergroundPipe:
                return ConnectionType.FLUID
            case Prototype.TransportBelt | Prototype.ExpressUndergroundBelt | Prototype.FastTransportBelt | Prototype.UndergroundBelt | Prototype.ExpressTransportBelt | Prototype.FastUndergroundBelt:
                return ConnectionType.TRANSPORT
            case Prototype.SmallElectricPole | Prototype.MediumElectricPole | Prototype.BigElectricPole:
                return ConnectionType.POWER
            case Prototype.StoneWall:
                return ConnectionType.WALL
            case _:
                raise ValueError(f'Unsupported connection type: {prototype}')

    def is_set_of_prototype(self, arg) -> bool:
        return isinstance(arg, AbstractSet) and all((isinstance(item, Prototype) for item in arg))

    def __call__(self, *args, **kwargs):
        """Wrapper method with retry logic for specific Lua errors."""
        max_retries = 2
        retry_count = 0
        while retry_count <= max_retries:
            try:
                return self.__call_impl__(*args, **kwargs)
            except Exception as e:
                error_message = str(e)
                if 'attempt to index field ? (a nil value)' in error_message:
                    retry_count += 1
                    if retry_count <= max_retries:
                        print(f'ConnectEntities retry {retry_count}/{max_retries} due to Lua indexing error: {error_message}')
                        continue
                    else:
                        print(f'ConnectEntities failed after {max_retries} retries: {error_message}')
                        raise
                else:
                    raise

    def __call_impl__(self, *args, **kwargs):
        connection_types = set()
        waypoints = []
        if 'connection_type' in kwargs:
            waypoints = args
            connection_types = kwargs['connection_type']
            if isinstance(connection_types, Prototype):
                connection_types = {connection_types}
            elif self.is_set_of_prototype(connection_types):
                connection_types = connection_types
        if 'target' in kwargs and 'source' in kwargs:
            waypoints = []
            waypoints.append(kwargs['source'])
            waypoints.append(kwargs['target'])
        if not waypoints:
            for arg in args:
                if isinstance(arg, Prototype):
                    connection_types = {arg}
                elif self.is_set_of_prototype(arg):
                    connection_types = arg
                else:
                    waypoints.append(arg)
        assert len(waypoints) > 1, 'Need more than one waypoint'
        connection = waypoints[0]
        dry_run = kwargs.get('dry_run', False)
        ticks_before = self.game_state.instance.get_elapsed_ticks() if not dry_run else 0
        if dry_run:
            total_required_entities = 0
        for _, target in zip(waypoints[:-1], waypoints[1:]):
            connection = self._connect_pair_of_waypoints(connection, target, connection_types=connection_types, dry_run=dry_run)
            if dry_run:
                total_required_entities += connection['number_of_entities_required']
                entities_available = connection['number_of_entities_available']
        if not dry_run:
            ticks_after = self.game_state.instance.get_elapsed_ticks()
            ticks_added = ticks_after - ticks_before
            if ticks_added > 0:
                game_speed = self.game_state.instance.get_speed()
                real_world_sleep = ticks_added / 60 / game_speed if game_speed > 0 else 0
        if dry_run:
            return {'number_of_entities_required': total_required_entities, 'number_of_entities_available': entities_available}
        return connection

    def _validate_connection_types(self, connection_types: Set[Prototype]):
        """
        Ensure that all connection_types handle the same contents - either FLUID, TRANSPORT or POWER
        """
        types = [self._get_connection_type(connection_type) for connection_type in connection_types]
        return len(set(types)) == 1

    def _connect_pair_of_waypoints(self, source: Union[Position, Entity, EntityGroup], target: Union[Position, Entity, EntityGroup], connection_types: Set[Prototype]={}, dry_run: bool=False) -> Union[Entity, EntityGroup]:
        """Connect two entities or positions."""
        valid = self._validate_connection_types(connection_types)
        if not valid:
            raise Exception(f'All connection types must handle the sort of contents: either fluid, power or items. Your types are incompatible {set(['Prototype.' + type.name for type in connection_types])}')
        if isinstance(source, Position):
            source = self._resolve_position_into_entity(source)
        else:
            source = self._refresh_entity(source)
        if isinstance(target, Position):
            target = self._resolve_position_into_entity(target)
        else:
            target = self._refresh_entity(target)
        resolver = self.resolvers[self._get_connection_type(list(connection_types)[0])]
        prioritised_list_of_position_pairs = resolver.resolve(source, target)
        last_exception = None
        for source_pos, target_pos in prioritised_list_of_position_pairs:
            try:
                self._last_source_pos = source_pos
                connection = self._create_connection(source_pos, target_pos, connection_types, dry_run, source_entity=source if isinstance(source, (Entity, EntityGroup)) else None, target_entity=target if isinstance(target, (Entity, EntityGroup)) else None)
                if not dry_run:
                    if connection and len(connection) > 0:
                        return connection[0]
                    else:
                        return self._get_existing_connection_group(target_pos, list(connection_types)[0], target)
                else:
                    return connection
            except Exception as e:
                error_str = str(e)
                if 'Failed to find a path' in error_str:
                    if any((ct in (Prototype.TransportBelt, Prototype.FastTransportBelt, Prototype.ExpressTransportBelt) for ct in connection_types)):
                        existing_group = self._get_existing_connection_group(target_pos, list(connection_types)[0], target)
                        if existing_group:
                            return existing_group
                    elif any((ct in (Prototype.SmallElectricPole, Prototype.MediumElectricPole, Prototype.BigElectricPole) for ct in connection_types)):
                        existing_group = self._get_existing_connection_group(target_pos, list(connection_types)[0], target)
                        if existing_group:
                            return existing_group
                    elif any((ct in (Prototype.Pipe, Prototype.UndergroundPipe) for ct in connection_types)):
                        existing_group = self._get_existing_connection_group(target_pos, list(connection_types)[0], target)
                        if existing_group:
                            return existing_group
                last_exception = e
                pass
        if (Prototype.UndergroundPipe in connection_types and Prototype.Pipe in connection_types) and (isinstance(target, (ChemicalPlant, OilRefinery)) or isinstance(source, (ChemicalPlant, OilRefinery))) and (not dry_run):
            for source_pos, target_pos in prioritised_list_of_position_pairs:
                try:
                    connection = self._create_modified_straight_connection(source_pos, target_pos, connection_types, source_entity=source if isinstance(source, (Entity, EntityGroup)) else None, target_entity=target if isinstance(target, (Entity, EntityGroup)) else None)
                    if connection and len(connection) > 0:
                        return connection[0]
                    else:
                        return self._get_existing_connection_group(target_pos, list(connection_types)[0], target)
                except Exception:
                    continue
        source_pos = source.position if not isinstance(source, Position) else source
        target_pos = target.position if not isinstance(target, Position) else target
        source_error_message_addition = f'{source}' if isinstance(source, Position) else f'{source.name} at {source.position}'
        target_error_message_addition = f'{target}' if isinstance(target, Position) else f'{target.name} at {target.position}'
        exception_message = str(last_exception)
        if exception_message == 'nil,':
            exception_message = 'Failed to connect entities. Please reposition entities or clear potential blockages'
        raise Exception(f'Failed to connect {set([type.name for type in connection_types])} from {source_error_message_addition} to {target_error_message_addition}. {self.get_error_message(exception_message)}')

    def _refresh_entity(self, entity: Entity) -> Entity:
        if isinstance(entity, (BeltGroup, ElectricityGroup, PipeGroup)):
            return entity
        updated_entities = self.get_entities(position=entity.position, radius=0)
        if len(updated_entities) == 1:
            return updated_entities[0]
        return entity

    def _resolve_position_into_entity(self, position: Position):
        entity = self._check_for_fluidhandlers(position)
        if entity:
            return entity
        entities = self.get_entities(position=position, radius=0)
        if not entities:
            entities = self.get_entities(position=position, radius=0.5)
            if not entities or len(entities) > 1:
                return position
        if isinstance(entities[0], EntityGroup):
            if isinstance(entities[0], PipeGroup):
                for pipe in entities[0].pipes:
                    if pipe.position.is_close(position, tolerance=0.707):
                        return pipe
            elif isinstance(entities[0], ElectricityGroup):
                for pole in entities[0].poles:
                    if pole.position.is_close(position, tolerance=0.707):
                        return pole
            elif isinstance(entities[0], BeltGroup):
                for belt in entities[0].belts:
                    if belt.position.is_close(position, tolerance=0.707):
                        return belt
        return entities[0]

    def _check_for_fluidhandlers(self, position: Position):
        """
        A very hacky way for now to check if the agent sent a multifluid or fluid handler input/output point
        We then use that entity but use strictly that position for connection
        """
        entities = self.get_entities(position=position, radius=1)
        for entity in entities:
            if isinstance(entity, MultiFluidHandler):
                for connection_point in entity.input_connection_points:
                    if connection_point.is_close(position, tolerance=0.01):
                        entity.input_connection_points = [connection_point]
                        return entity
                for connection_point in entity.output_connection_points:
                    if connection_point.is_close(position, tolerance=0.01):
                        entity.output_connection_points = [connection_point]
                        return entity
            elif isinstance(entity, FluidHandler):
                for connection_point in entity.connection_points:
                    if connection_point.is_close(position, tolerance=0.005):
                        entity.connection_points = [connection_point]
                        return entity
        return None

    def _attempt_path_finding(self, source_pos: Position, target_pos: Position, connection_prototypes: List[str], num_available: int, pathing_radius: float=1, dry_run: bool=False, allow_paths_through_own: bool=False) -> PathResult:
        """Attempt to find a path between two positions"""
        entity_sizes = [1.5, 1, 0.5, 0.25]
        for size in entity_sizes:
            path_handle = self.request_path(finish=target_pos, start=source_pos, allow_paths_through_own_entities=allow_paths_through_own, radius=pathing_radius, entity_size=size)
            sleep(0.05)
            response, _ = self.execute(self.player_index, source_pos.x, source_pos.y, target_pos.x, target_pos.y, path_handle, ','.join(connection_prototypes), dry_run, num_available)
            result = PathResult(response)
            if result.is_success:
                return result
        return result

    def _create_connection(self, source_pos: Position, target_pos: Position, connection_types: Set[Prototype], dry_run: bool=False, source_entity: Optional[Entity]=None, target_entity: Optional[Entity]=None) -> List[Union[Entity, EntityGroup]]:
        """Create a connection between two positions"""
        connection_info = self._get_connection_info(connection_types)
        connection_prototype = connection_info['last_connection_prototype']
        connection_type = connection_info['last_connection_type']
        connection_type_names = connection_info['connection_names']
        names_to_type = connection_info['names_to_types']
        metaclasses = connection_info['metaclasses']
        inventory = self.inspect_inventory()
        num_available = inventory.get(connection_prototype, 0)
        connection_type_names_values = list(connection_type_names.values())
        match connection_types:
            case _ if connection_types & {Prototype.Pipe, Prototype.UndergroundPipe}:
                pathing_radius = 0.5
                self._extend_collision_boxes(source_pos, target_pos)
                num_available = inventory.get(Prototype.Pipe, 0)
                try:
                    result = self._attempt_path_finding(source_pos, target_pos, connection_type_names_values, num_available, pathing_radius, dry_run)
                finally:
                    self._clear_collision_boxes()
            case _ if connection_types & {Prototype.TransportBelt, Prototype.UndergroundBelt} or connection_types & {Prototype.FastTransportBelt, Prototype.FastUndergroundBelt} or connection_types & {Prototype.ExpressTransportBelt, Prototype.ExpressUndergroundBelt}:
                pathing_radius = 0.5
                result = self._attempt_path_finding(source_pos, target_pos, connection_type_names_values, num_available, pathing_radius, dry_run)
                if not result.is_success:
                    source_pos_adjusted = self._adjust_belt_position(source_pos, source_entity)
                    target_pos_adjusted = self._adjust_belt_position(target_pos, target_entity)
                    adjusted_result = self._attempt_path_finding(source_pos_adjusted, target_pos_adjusted, connection_type_names_values, num_available, 2, dry_run, False)
                    if adjusted_result.is_success:
                        result = adjusted_result
                    pass
            case _:
                pathing_radius = 4
                self._extend_collision_boxes(source_pos, target_pos)
                try:
                    result = self._attempt_path_finding(source_pos, target_pos, connection_type_names_values, num_available, pathing_radius, dry_run, True)
                finally:
                    self._clear_collision_boxes()
        if not result.is_success:
            raise Exception(f'{self.get_error_message(result.error_message.lstrip())}')
        if dry_run:
            return {'number_of_entities_required': result.required_entities, 'number_of_entities_available': num_available}
        groupable_entities, path = self._get_groupable_entities(result, metaclasses, names_to_type)
        entity_groups = self._process_entity_groups(connection_type, groupable_entities, source_entity, target_entity, source_pos)
        return _deduplicate_entities(path) + entity_groups

    def _get_connection_info(self, connection_types: Set[Prototype]):
        connection_type_names = {}
        names_to_type = {}
        metaclasses = {}
        for connection_type in connection_types:
            connection_prototype, metaclass = connection_type.value
            metaclasses[connection_prototype] = metaclass
            connection_type_names[connection_type] = connection_prototype
            names_to_type[connection_prototype] = connection_type
        return {'connection_names': connection_type_names, 'names_to_types': names_to_type, 'metaclasses': metaclasses, 'last_connection_prototype': connection_prototype, 'last_connection_type': connection_type}

    def _get_groupable_entities(self, result, metaclasses, names_to_type):
        path = []
        groupable_entities = []
        for entity_data in result.entities.values():
            if not isinstance(entity_data, dict):
                continue
            try:
                self._process_warnings(entity_data)
                entity = metaclasses[entity_data['name']](prototype=names_to_type[entity_data['name']], **entity_data)
                if entity.prototype in (Prototype.TransportBelt, Prototype.UndergroundBelt, Prototype.FastTransportBelt, Prototype.FastUndergroundBelt, Prototype.ExpressTransportBelt, Prototype.ExpressUndergroundBelt, Prototype.StoneWall, Prototype.Pipe, Prototype.UndergroundPipe, Prototype.SmallElectricPole, Prototype.BigElectricPole, Prototype.MediumElectricPole):
                    groupable_entities.append(entity)
                else:
                    path.append(entity)
            except Exception as e:
                if entity_data:
                    raise Exception(f'Failed to create {entity_data['name']} object from response: {result.raw_response}') from e
        return (groupable_entities, path)

    def _create_modified_straight_connection(self, source_pos: Position, target_pos: Position, connection_types: Set[Prototype], source_entity: Optional[Entity]=None, target_entity: Optional[Entity]=None) -> List[Union[Entity, EntityGroup]]:
        """Create a connection between two positions"""
        connection_info = self._get_connection_info(connection_types)
        connection_prototype = connection_info['last_connection_prototype']
        connection_type = connection_info['last_connection_type']
        connection_type_names = connection_info['connection_names']
        names_to_type = connection_info['names_to_types']
        metaclasses = connection_info['metaclasses']
        inventory = self.inspect_inventory()
        num_available = inventory.get(connection_prototype, 0)
        pathing_radius = 0.5
        connection_type_names_values = list(connection_type_names.values())
        result = None
        try:
            result = self.try_straight_line_pathing(source_pos, target_pos, connection_type_names_values, num_available, pathing_radius, target_entity, source_entity, names_to_type, metaclasses)
        finally:
            pass
        if result is None:
            return None
        groupable_entities, path = self._get_groupable_entities(result, metaclasses, names_to_type)
        entity_groups = self._process_entity_groups(connection_type, groupable_entities, source_entity, target_entity, source_pos)
        return _deduplicate_entities(path) + entity_groups

    def _process_warnings(self, entity_data: Dict):
        """Process warnings in entity data"""
        if not entity_data.get('warnings'):
            entity_data['warnings'] = []
        else:
            warnings = entity_data['warnings']
            entity_data['warnings'] = list(warnings.values()) if isinstance(warnings, dict) else [warnings]

    def _process_entity_groups(self, connection_type: Prototype, groupable_entities: List[Entity], source_entity: Optional[Entity], target_entity: Optional[Entity], source_pos: Position) -> List[EntityGroup]:
        """Process and create entity groups based on connection type"""
        match connection_type:
            case Prototype.ExpressTransportBelt | Prototype.FastTransportBelt | Prototype.TransportBelt | Prototype.UndergroundBelt | Prototype.FastUndergroundBelt | Prototype.ExpressUndergroundBelt:
                return self._process_belt_groups(groupable_entities, source_entity, target_entity, source_pos)
            case Prototype.Pipe | Prototype.UndergroundPipe:
                return self._process_pipe_groups(groupable_entities, source_pos)
            case Prototype.StoneWall:
                return self._process_groups(Prototype.StoneWall, groupable_entities, source_pos)
            case _:
                return self._process_power_groups(groupable_entities, source_pos)

    def _process_belt_groups(self, groupable_entities: List[Entity], source_entity: Optional[Union[Entity, EntityGroup]], target_entity: Optional[Union[Entity, EntityGroup]], source_pos: Position) -> List[BeltGroup]:
        """Process transport belt groups"""
        if isinstance(source_entity, BeltGroup):
            entity_groups = agglomerate_groupable_entities(groupable_entities)
        elif isinstance(target_entity, BeltGroup):
            entity_groups = agglomerate_groupable_entities(groupable_entities + target_entity.belts)
        else:
            entity_groups = agglomerate_groupable_entities(groupable_entities)
        for group in entity_groups:
            if hasattr(group, 'belts'):
                group.belts = _deduplicate_entities(group.belts)
        if isinstance(source_entity, BeltGroup) and entity_groups:
            self.rotate_end_belt_to_face(source_entity, entity_groups[0])
        if isinstance(target_entity, BeltGroup) and entity_groups:
            self.rotate_end_belt_to_face(entity_groups[0], target_entity)
        entity_groups = self.get_entities({Prototype.TransportBelt, Prototype.ExpressTransportBelt, Prototype.FastTransportBelt, Prototype.UndergroundBelt, Prototype.FastUndergroundBelt, Prototype.ExpressUndergroundBelt}, source_pos)
        for group in entity_groups:
            if source_pos in [entity.position for entity in group.belts]:
                return cast(List[BeltGroup], [group])
        return cast(List[BeltGroup], entity_groups)

    def _update_belt_group(self, new_belt: BeltGroup, source_belt: TransportBelt, target_belt: TransportBelt):
        new_belt.outputs[0] = source_belt
        for belt in new_belt.belts:
            if belt.position == source_belt.position:
                belt.input_position = source_belt.input_position
                belt.output_position = source_belt.output_position
                belt.direction = source_belt.direction
                belt.is_source = source_belt.is_source
                belt.is_terminus = source_belt.is_terminus
                if not belt.is_terminus and belt in new_belt.outputs:
                    new_belt.outputs.remove(belt)
                if not belt.is_source and belt in new_belt.inputs:
                    new_belt.inputs.remove(belt)
            if belt.position == target_belt.position:
                belt.is_source = target_belt.is_source
                belt.is_terminus = target_belt.is_terminus
                if not belt.is_terminus and belt in new_belt.outputs:
                    new_belt.outputs.remove(belt)
                if not belt.is_source and belt in new_belt.inputs:
                    new_belt.inputs.remove(belt)

    def rotate_end_belt_to_face(self, source_belt_group: BeltGroup, target: BeltGroup) -> BeltGroup:
        if not source_belt_group.outputs:
            return source_belt_group
        source_belt = source_belt_group.outputs[0]
        target_belt = target.inputs[0]
        source_pos = source_belt.position
        target_pos = target_belt.position
        if not source_pos.is_close(target_pos, 1.001):
            return source_belt_group
        relative_pos = (numpy.sign(source_pos.x - target_pos.x), numpy.sign(source_pos.y - target_pos.y))
        match relative_pos:
            case [1, 1]:
                pass
            case [-1, -1]:
                pass
            case [1, _] if source_belt.direction.value not in (DirectionInternal.LEFT.value, DirectionInternal.RIGHT.value):
                source_belt = self.rotate_entity(source_belt, DirectionInternal.LEFT)
            case [-1, _] if source_belt.direction.value not in (DirectionInternal.LEFT.value, DirectionInternal.RIGHT.value):
                source_belt = self.rotate_entity(source_belt, DirectionInternal.RIGHT)
            case [_, 1] if source_belt.direction.value not in (DirectionInternal.UP.value, DirectionInternal.DOWN.value):
                source_belt = self.rotate_entity(source_belt, DirectionInternal.UP)
            case [_, -1] if source_belt.direction.value not in (DirectionInternal.UP.value, DirectionInternal.DOWN.value):
                source_belt = self.rotate_entity(source_belt, DirectionInternal.DOWN)
        target_belt = self.get_entity(target_belt.prototype, target_belt.position)
        self._update_belt_group(source_belt_group, source_belt, target_belt)
        return source_belt

    def _process_pipe_groups(self, groupable_entities: List[Entity], source_pos: Position) -> List[PipeGroup]:
        """Process pipe groups"""
        entity_groups = self.get_entities({Prototype.Pipe, Prototype.UndergroundPipe}, source_pos)
        for group in entity_groups:
            group.pipes = _deduplicate_entities(group.pipes)
            if source_pos in [entity.position for entity in group.pipes]:
                return [group]
        return entity_groups

    def _process_groups(self, prototype: Prototype, groupable_entities: List[Entity], source_pos: Position) -> List[PipeGroup]:
        """Process other groups"""
        entity_groups = self.get_entities(prototype, source_pos)
        for group in entity_groups:
            group.entities = _deduplicate_entities(group.entities)
            if source_pos in [entity.position for entity in group.entities]:
                return [group]
        return entity_groups

    def _attempt_to_get_entity(self, position: Position, get_connectors: bool=False) -> Union[Position, Entity, EntityGroup]:
        """
        Attempts to find an entity at the given position.

        Args:
            position: The position to check
            get_connectors: If True, returns connector entities (belts, pipes) instead of treating them as positions

        Returns:
            - The original position if no entity is found
            - The position if a connector entity is found and get_connectors is False
            - The entity or entity group if found and either get_connectors is True or it's not a connector
        """
        entities = self.get_entities(position=position, radius=0.1)
        if not entities:
            return position
        entity = entities[0]
        if not get_connectors and isinstance(entity, (BeltGroup, TransportBelt, PipeGroup, Pipe)):
            return position
        return entity

    def _process_power_groups(self, groupable_entities: List[Entity], source_pos: Position) -> List[ElectricityGroup]:
        """Process power pole groups"""
        return cast(List[ElectricityGroup], self.get_entities({Prototype.SmallElectricPole, Prototype.BigElectricPole, Prototype.MediumElectricPole}, source_pos))

    def _adjust_belt_position(self, pos: Position, entity: Optional[Entity]) -> Position:
        """Adjust belt position for better path finding"""
        if not entity or isinstance(entity, Position):
            entity = self._attempt_to_get_entity(pos, get_connectors=True)
            if entity and isinstance(entity, BeltGroup):
                return entity.outputs[0].output_position
        return pos

    def try_straight_line_pathing(self, source_pos: Position, target_pos: Position, connection_type_names_values: List[str], num_available: int, pathing_radius: int, target_entity: Entity, source_entity: Entity, names_to_type: Dict[str, Prototype], metaclasses: Dict[str, Entity]) -> PathResult:
        """
        Try to create a path between source and target with a straight line extension
        This helps to avoid blockages using underground pipes
        First try to find unblocked areas for source and target and then connect them
        """
        required_pipes = 0
        if isinstance(target_entity, (ChemicalPlant, OilRefinery)):
            required_pipes += 2
        if isinstance(source_entity, (ChemicalPlant, OilRefinery)):
            required_pipes += 2
        inventory = self.inspect_inventory()
        underground_pipes = inventory.get('pipe-to-ground', 0)
        if underground_pipes < required_pipes:
            return None
        margin = 3
        max_distance = 10 - margin
        pathing_radius = 0.5
        offset_dict = {'UP': {'x': 0, 'y': -1}, 'DOWN': {'x': 0, 'y': 1}, 'LEFT': {'x': -1, 'y': 0}, 'RIGHT': {'x': 1, 'y': 0}}
        for target_run_idx in range(1, max_distance + 1):
            target_straight_line_path_dict = self.create_straight_line_dict(target_entity, target_pos, offset_dict, margin, target_run_idx, num_available, offset_sign=-1)
            if not target_straight_line_path_dict:
                continue
            for source_run_idx in range(1, max_distance + 1):
                source_straight_line_path_dict = self.create_straight_line_dict(source_entity, source_pos, offset_dict, margin, source_run_idx, num_available)
                if not source_straight_line_path_dict:
                    continue
                if source_straight_line_path_dict['success'] and target_straight_line_path_dict['success']:
                    source_pos = self.get_final_connection_pos(source_straight_line_path_dict, offset_dict, source_entity)
                    target_pos = self.get_final_connection_pos(target_straight_line_path_dict, offset_dict, target_entity)
                    try:
                        self._extend_collision_boxes(source_pos, target_pos)
                        inbetween_path = self._attempt_path_finding(source_pos, target_pos, connection_type_names_values, num_available, pathing_radius, False, False)
                    finally:
                        self._clear_collision_boxes()
                    if inbetween_path.is_success:
                        if source_straight_line_path_dict['path']:
                            for value in source_straight_line_path_dict['path'].entities.values():
                                inbetween_path.entities[len(inbetween_path.entities) + 1] = value
                        if target_straight_line_path_dict['path']:
                            for value in target_straight_line_path_dict['path'].entities.values():
                                inbetween_path.entities[len(inbetween_path.entities) + 1] = value
                        return inbetween_path
                if source_straight_line_path_dict['path']:
                    self.pickup_entities(source_straight_line_path_dict['path'], metaclasses, names_to_type)
            if target_straight_line_path_dict['path']:
                self.pickup_entities(target_straight_line_path_dict['path'], metaclasses, names_to_type)
        return None

    def get_final_connection_pos(self, input_path_dict: Dict, offset_dict: Dict[str, int], input_entity: Entity):
        """
        Get the final connection point for the input path dict
        if it has a path parameter, it means it's a straight line extension and hence need to do one more offset and make that as the connection point
        this prevents the pathing to do any weird paths where the underground exit won't be merged with
        othersiwe we can just return the original connection position
        """
        if input_path_dict['path']:
            connection_pos = input_path_dict['connection_position']
            offset = offset_dict[input_entity.direction.name]
            connection_pos = Position(connection_pos.x + offset['x'], connection_pos.y + offset['y'])
        else:
            connection_pos = input_path_dict['connection_position']
        return connection_pos

    def create_straight_line_dict(self, input_entity: Entity, input_pos: Position, offset_dict: Dict[str, int], margin: int, run_idx: int, num_available: int, offset_sign: int=1):
        """
        Create a straight line extension for the input entity if it is a chemical plant or oil refinery
        The offset sign is used to specify which direction to extend the straight line
        By default, we use calibrated offsets for source directions
        Hence when connecting target entity, we need to flip the offsets
        """
        entity_direction = input_entity.direction
        if isinstance(input_entity, ChemicalPlant) or isinstance(input_entity, OilRefinery):
            if entity_direction.name not in offset_dict:
                raise Exception(f'Invalid direction for {input_entity.name}')
            offset = offset_dict[entity_direction.name]
            offset['x'] = offset['x'] * offset_sign
            offset['y'] = offset['y'] * offset_sign
            try:
                output_dict = self.get_straight_line_path(input_pos, offset, margin, run_idx, num_available)
            except:
                return None
        else:
            output_dict = {'success': True, 'connection_position': input_pos, 'path': None}
        return output_dict

    def get_straight_line_path(self, starting_pos: Position, offset: Dict[str, int], margin: int, distance: int, num_available: int):
        """
        Create the straight line path for the input starting position
        Offset it by the given direction and distance and use pipe-to-ground to connect
        """
        new_straigth_line_end_pos = Position(starting_pos.x + offset['x'] * (distance + margin), starting_pos.y + offset['y'] * (distance + margin))
        if self._is_blocked(new_straigth_line_end_pos, 1.5):
            raise Exception('Point to be connected to is blocked')
        try:
            self._extend_collision_boxes(starting_pos, new_straigth_line_end_pos)
            target_entity_straight_path = self._attempt_path_finding(starting_pos, new_straigth_line_end_pos, ['pipe-to-ground'], num_available, 0.5, False, True)
        finally:
            self._clear_collision_boxes()
        if target_entity_straight_path.is_success:
            return {'success': True, 'connection_position': new_straigth_line_end_pos, 'path': target_entity_straight_path}
        raise Exception('Failed to connect straight line')

    def _is_blocked(self, pos: Position, radius=0.5) -> bool:
        """
        Check if the position is blocked by entities"""
        entities = self.get_entities(position=pos, radius=radius)
        return bool(entities)

    def _get_existing_connection_group(self, target_pos: Position, connection_type: Prototype, target_entity) -> Union[Entity, EntityGroup, Position]:
        """
        Get existing connection group when no new entities were created.
        This handles cases where entities are already connected.
        """
        try:
            if connection_type in (Prototype.SmallElectricPole, Prototype.MediumElectricPole, Prototype.BigElectricPole):
                groups = self.get_entities({Prototype.SmallElectricPole, Prototype.MediumElectricPole, Prototype.BigElectricPole}, target_pos, radius=10)
            elif connection_type in (Prototype.TransportBelt, Prototype.FastTransportBelt, Prototype.ExpressTransportBelt):
                groups = self.get_entities({Prototype.TransportBelt, Prototype.FastTransportBelt, Prototype.ExpressTransportBelt, Prototype.UndergroundBelt, Prototype.FastUndergroundBelt, Prototype.ExpressUndergroundBelt}, target_pos, radius=10)
                if not groups and hasattr(self, '_last_source_pos'):
                    groups = self.get_entities({Prototype.TransportBelt, Prototype.FastTransportBelt, Prototype.ExpressTransportBelt, Prototype.UndergroundBelt, Prototype.FastUndergroundBelt, Prototype.ExpressUndergroundBelt}, self._last_source_pos, radius=10)
            elif connection_type in (Prototype.Pipe, Prototype.UndergroundPipe):
                groups = self.get_entities({Prototype.Pipe, Prototype.UndergroundPipe}, target_pos, radius=5)
            else:
                groups = []
            if groups:
                return groups[0]
            if isinstance(target_entity, (Entity, EntityGroup)):
                return target_entity
            return target_pos
        except Exception:
            if isinstance(target_entity, (Entity, EntityGroup)):
                return target_entity
            return target_pos

    def pickup_entities(self, path_data: dict, metaclasses: Dict[str, Entity], names_to_type: Dict[str, Prototype]):
        """
        Pickup the entities in the path data
        """
        for entity_data in path_data.entities.values():
            if not isinstance(entity_data, dict):
                continue
            self._process_warnings(entity_data)
            entity = metaclasses[entity_data['name']](prototype=names_to_type[entity_data['name']], **entity_data)
            self.pickup_entity(entity)

