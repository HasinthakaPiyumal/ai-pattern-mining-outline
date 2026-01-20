# Cluster 68

def test_reset_observation(instance):
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()

def test_inventory_observation(instance):
    """Test that inventory changes are reflected in observations."""
    instance.initial_inventory = {'coal': 50, 'iron-chest': 1, 'iron-plate': 5}
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    inventory_items = {item['type']: item['quantity'] for item in observation['inventory']}
    assert inventory_items['coal'] == 50
    assert inventory_items['iron-chest'] == 1
    assert inventory_items['iron-plate'] == 5
    chest = instance.namespace.place_entity(Prototype.IronChest, position=Position(x=2.5, y=2.5))
    chest = instance.namespace.insert_item(Prototype.Coal, chest, quantity=10)
    action = Action(agent_idx=0, code='pass', game_state=None)
    observation, reward, terminated, truncated, info = env.step(action)
    chest_entities = [e for e in observation['entities'] if 'iron-chest' in e]
    assert len(chest_entities) == 1
    chest_str = chest_entities[0]
    assert 'iron-chest' in chest_str
    assert 'x=2.5, y=2.5' in chest_str

def test_entity_placement_observation(instance):
    """Test that entity placement is reflected in observations."""
    instance.initial_inventory = {'stone-furnace': 1, 'coal': 50, 'iron-ore': 10}
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    assert len(observation['entities']) == 0
    instance.namespace.place_entity(Prototype.StoneFurnace, direction=Direction.UP, position=Position(x=2.5, y=2.5))
    action = Action(agent_idx=0, code='pass', game_state=None)
    observation, reward, terminated, truncated, info = env.step(action)
    furnace_entities = [e for e in observation['entities'] if 'stone-furnace' in e]
    assert len(furnace_entities) == 1
    furnace_str = furnace_entities[0]
    assert 'stone-furnace' in furnace_str
    assert 'x=3.0, y=3.0' in furnace_str
    assert 'Direction.UP' in furnace_str

def test_research_observation(instance):
    """Test that research state changes are reflected in observations."""
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    action = Action(agent_idx=0, code='Technology = Prototype.Automation; self.research(Technology)', game_state=None)
    observation, reward, terminated, truncated, info = env.step(action)
    research = observation['research']
    assert 'technologies' in research
    if isinstance(research['technologies'], list):
        assert len(research['technologies']) > 0
        tech = research['technologies'][0]
        assert 'name' in tech
    elif isinstance(research['technologies'], dict):
        assert len(research['technologies']) > 0
        tech = next(iter(research['technologies'].values()))
        assert 'name' in tech.__dict__ or hasattr(tech, 'name')

def test_flows_observation(instance):
    """Test that production flows change after crafting or smelting."""
    instance.initial_inventory = {'iron-ore': 10, 'stone-furnace': 1, 'coal': 10}
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    env.instance.namespace.place_entity(Prototype.StoneFurnace, position=Position(x=1.5, y=1.5))
    action = Action(agent_idx=0, code='for i in range(5): pass', game_state=None)
    observation, reward, terminated, truncated, info = env.step(action)
    flows = observation['flows']
    assert 'input' in flows
    assert 'output' in flows
    assert isinstance(flows['input'], list)
    assert isinstance(flows['output'], list)

def test_raw_text_observation(instance):
    """Test that raw_text is updated after an action that prints output."""
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False)
    env.reset()
    action = Action(agent_idx=0, code='print("Hello world!")', game_state=None)
    observation, reward, terminated, truncated, info = env.step(action)
    assert 'raw_text' in observation
    assert 'Hello world' in observation['raw_text']

def test_serialized_functions_observation(instance):
    """Test that defining a function via action adds it to serialized_functions in observation."""
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False)
    env.reset()
    code = 'def my_test_func():\n    return 42'
    action = Action(agent_idx=0, code=code, game_state=None)
    observation, reward, terminated, truncated, info = env.step(action)
    assert 'serialized_functions' in observation
    assert any((f['name'] == 'my_test_func' for f in observation['serialized_functions']))

def test_messages_observation(instance):
    """Test that sending a message is reflected in the observation."""
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False)
    env.reset()
    if hasattr(instance.namespace, 'load_messages'):
        msg = {'sender': 'test_agent', 'message': 'Test message', 'timestamp': 1234567890}
        instance.namespace.load_messages([msg])
    action = Action(agent_idx=0, code='pass', game_state=None)
    observation, reward, terminated, truncated, info = env.step(action)
    assert 'messages' in observation
    if observation['messages']:
        assert any(('Test message' in m.get('content', '') or 'Test message' in m.get('message', '') for m in observation['messages']))

def test_game_info_elapsed_ticks_sleep(instance):
    """Test that game_info.tick reflects elapsed ticks correctly after sleep actions."""
    instance.initial_inventory = {'coal': 100}
    instance.reset()
    instance.set_speed(10.0)
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    initial_ticks = observation['game_info']['tick']
    sleep_action = Action(agent_idx=0, code='sleep(2)', game_state=None)
    observation, reward, terminated, truncated, info = env.step(sleep_action)
    final_ticks = observation['game_info']['tick']
    ticks_added = final_ticks - initial_ticks
    assert ticks_added == 120, f'Expected 120 ticks for 2-second sleep, got {ticks_added}'
    assert observation['game_info']['speed'] == 10.0, 'Game speed should be 10.0'

def test_game_info_elapsed_ticks_craft_item(instance):
    """Test that game_info.tick reflects elapsed ticks correctly after crafting."""
    instance.initial_inventory = {'iron-plate': 100}
    instance.reset()
    instance.set_speed(10.0)
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    initial_ticks = observation['game_info']['tick']
    craft_action = Action(agent_idx=0, code='craft_item(Prototype.IronGearWheel, 3)', game_state=None)
    observation, reward, terminated, truncated, info = env.step(craft_action)
    final_ticks = observation['game_info']['tick']
    ticks_added = final_ticks - initial_ticks
    assert ticks_added == 90, f'Expected 90 ticks for crafting 3 iron gear wheels, got {ticks_added}'

def test_game_info_elapsed_ticks_move_to(instance):
    """Test that game_info.tick reflects elapsed ticks correctly after movement."""
    instance.initial_inventory = {'coal': 100}
    instance.reset()
    instance.set_speed(10.0)
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    initial_ticks = observation['game_info']['tick']
    move_action = Action(agent_idx=0, code='move_to(Position(x=3, y=3))', game_state=None)
    observation, reward, terminated, truncated, info = env.step(move_action)
    final_ticks = observation['game_info']['tick']
    ticks_added = final_ticks - initial_ticks
    assert 30 <= ticks_added <= 40, f'Expected ~35 ticks for 3-tile movement, got {ticks_added}'

def test_game_info_elapsed_ticks_harvest_resource(instance):
    """Test that game_info.tick reflects elapsed ticks correctly after harvesting."""
    instance.initial_inventory = {'coal': 100}
    instance.reset()
    instance.set_speed(10.0)
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    instance.rcon_client.send_command('/sc game.player.character.reach_distance = 1000; game.player.character.mining_reach_distance = 1000')
    instance.rcon_client.send_command('/sc game.tick_paused = false')
    namespace = instance.namespace
    nearest_iron_ore = namespace.nearest(Resource.IronOre)
    print(f'nearest iron ore: {nearest_iron_ore}')
    namespace.move_to(Position(x=nearest_iron_ore.x, y=nearest_iron_ore.y))
    current_ticks = instance.get_elapsed_ticks()
    namespace.harvest_resource(nearest_iron_ore, 1, radius=1000)
    end_ticks = instance.get_elapsed_ticks()
    ticks_added = end_ticks - current_ticks
    print(f'inventory: {observation['inventory']}')
    print(f'raw_text: {observation['raw_text']}')
    assert 50 <= ticks_added <= 80, f'Expected ~60 ticks for harvesting iron ore, got {ticks_added}'

def test_game_info_elapsed_ticks_multiple_actions(instance):
    """Test that game_info.tick correctly accumulates ticks across multiple actions."""
    instance.initial_inventory = {'iron-plate': 100, 'coal': 100}
    instance.reset()
    instance.set_speed(10.0)
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    initial_ticks = observation['game_info']['tick']
    actions = ['sleep(1)', 'craft_item(Prototype.IronGearWheel, 1)', 'move_to(Position(x=2, y=2))']
    for i, code in enumerate(actions):
        action = Action(agent_idx=0, code=code, game_state=None)
        observation, reward, terminated, truncated, info = env.step(action)
        current_ticks = observation['game_info']['tick']
        ticks_since_start = current_ticks - initial_ticks
        if i == 0:
            assert 55 <= ticks_since_start <= 65, f'Expected ~60 ticks after sleep, got {ticks_since_start}'
        elif i == 1:
            assert 85 <= ticks_since_start <= 95, f'Expected ~90 ticks after sleep+craft, got {ticks_since_start}'
        elif i == 2:
            assert 110 <= ticks_since_start <= 120, f'Expected ~115 ticks after all actions, got {ticks_since_start}'

def test_game_info_elapsed_ticks_with_game_speed(instance):
    """Test that game_info.tick is independent of game speed (always standard ticks)."""
    instance.initial_inventory = {'coal': 100}
    instance.reset()
    instance.set_speed(3.0)
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    initial_ticks = observation['game_info']['tick']
    sleep_action = Action(agent_idx=0, code='sleep(1)', game_state=None)
    observation, reward, terminated, truncated, info = env.step(sleep_action)
    final_ticks = observation['game_info']['tick']
    ticks_added = final_ticks - initial_ticks
    assert ticks_added == 60, f'Expected 60 ticks regardless of speed, got {ticks_added}'
    assert observation['game_info']['speed'] == 3.0, 'Game speed should be 3.0'

def test_game_info_tick_persistence(instance):
    """Test that game_info.tick persists and accumulates correctly across observations."""
    instance.initial_inventory = {'iron-plate': 100}
    instance.reset()
    instance.set_speed(1.0)
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    tick_history = [observation['game_info']['tick']]
    action1 = Action(agent_idx=0, code='sleep(0.5)', game_state=None)
    observation, reward, terminated, truncated, info = env.step(action1)
    tick_history.append(observation['game_info']['tick'])
    action2 = Action(agent_idx=0, code='craft_item(Prototype.IronGearWheel, 1)', game_state=None)
    observation, reward, terminated, truncated, info = env.step(action2)
    tick_history.append(observation['game_info']['tick'])
    action3 = Action(agent_idx=0, code='pass', game_state=None)
    observation, reward, terminated, truncated, info = env.step(action3)
    tick_history.append(observation['game_info']['tick'])
    assert tick_history[1] - tick_history[0] == 30, 'First action should add 30 ticks'
    assert tick_history[2] - tick_history[1] == 30, 'Second action should add 30 ticks'
    assert tick_history[3] - tick_history[2] == 0, 'No-op should add 0 ticks'
    assert tick_history[3] - tick_history[0] == 60, 'Total should be 60 ticks'

def test_game_info_structure(instance):
    """Test that game_info contains all expected fields with correct types."""
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation, info = env.reset()
    game_info = observation['game_info']
    assert 'tick' in game_info, "game_info should contain 'tick' field"
    assert 'time' in game_info, "game_info should contain 'time' field"
    assert 'speed' in game_info, "game_info should contain 'speed' field"
    assert isinstance(game_info['tick'], int), 'tick should be an integer'
    assert isinstance(game_info['time'], (int, float)), 'time should be numeric'
    assert isinstance(game_info['speed'], (int, float)), 'speed should be numeric'
    assert game_info['tick'] >= 0, 'tick should be non-negative'
    assert game_info['speed'] > 0, 'speed should be positive'

def test_vision_disabled_by_default(instance):
    """Test that vision is disabled by default and map_image is empty."""
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False)
    observation = env.reset()
    assert 'map_image' in observation, 'Observation should contain map_image field'
    assert observation['map_image'] == '', 'map_image should be empty when vision is disabled'

def test_vision_enabled_produces_base64_image(instance):
    """Test that enabling vision produces a base64 encoded image."""
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False, enable_vision=True)
    observation = env.reset()
    assert 'map_image' in observation, 'Observation should contain map_image field'
    assert observation['map_image'] != '', 'map_image should not be empty when vision is enabled'
    assert isinstance(observation['map_image'], str), 'map_image should be a string'
    try:
        image_data = base64.b64decode(observation['map_image'])
        img = Image.open(io.BytesIO(image_data))
        assert img.width > 0, 'Image should have positive width'
        assert img.height > 0, 'Image should have positive height'
        assert img.mode in ['RGB', 'RGBA'], f'Image should be RGB or RGBA, got {img.mode}'
        assert 600 <= img.width <= 1000, f'Image width {img.width} should be roughly 800px'
        assert 600 <= img.height <= 1000, f'Image height {img.height} should be roughly 800px'
        print(f'Image size: {img.width}x{img.height}')
    except Exception as e:
        raise AssertionError(f'Failed to decode base64 image: {e}')

def test_vision_multimodal_api_format(instance):
    """Test that the base64 image is in the correct format for multimodal APIs."""
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False, enable_vision=True)
    observation = env.reset()
    map_image_b64 = observation['map_image']
    message_content = [{'type': 'text', 'text': 'What do you see in this Factorio factory?'}, {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{map_image_b64}'}}]
    assert len(message_content) == 2
    assert message_content[1]['type'] == 'image_url'
    assert message_content[1]['image_url']['url'].startswith('data:image/png;base64,')
    base64_part = message_content[1]['image_url']['url'].split(',')[1]
    assert base64_part == map_image_b64
    try:
        decoded = base64.b64decode(base64_part)
        assert len(decoded) > 0, 'Decoded image should have content'
    except Exception as e:
        raise AssertionError(f'Failed to decode base64 for API: {e}')
    print('✓ Image is correctly formatted for multimodal APIs')
    print(f'  Base64 length: {len(map_image_b64)} characters')
    print(f'  Decoded size: {len(decoded)} bytes')

def test_vision_persists_across_steps(instance):
    """Test that vision rendering works across multiple steps."""
    instance.initial_inventory = {'stone-furnace': 1}
    instance.reset()
    env = FactorioGymEnv(instance, pause_after_action=False, enable_vision=True)
    observation1 = env.reset()
    assert observation1['map_image'] != ''
    initial_image = observation1['map_image']
    player_pos = instance.namespaces[0].player_location
    action = Action(agent_idx=0, code=f'place_entity(Prototype.StoneFurnace, position=Position(x={player_pos.x + 5}, y={player_pos.y + 5}))', game_state=None)
    observation2, _, _, _, _ = env.step(action)
    assert observation2['map_image'] != ''
    assert len(observation2['map_image']) > 100
    try:
        base64.b64decode(initial_image)
        base64.b64decode(observation2['map_image'])
    except Exception as e:
        raise AssertionError(f'Failed to decode images: {e}')

@pytest.fixture
def env(instance):
    env = FactorioGymEnv(instance, pause_after_action=False)
    yield env
    env.close()

def test_gym_env_interface(env):
    assert isinstance(env.action_space, spaces.Dict)
    assert isinstance(env.observation_space, spaces.Dict)
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    action = Action(agent_idx=0, code='pass', game_state=None)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_gym_env_rejects_bad_action(env):
    with pytest.raises(Exception):
        env.step({'bad': 'action'})
    with pytest.raises(Exception):
        env.step(123)
    with pytest.raises(Exception):
        env.step(Action(agent_idx=99, code='pass', game_state=None))

