# Cluster 97

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'coal': 5, 'iron-chest': 1, 'iron-plate': 50, 'iron-ore': 10, 'stone-furnace': 1, 'assembling-machine-1': 1, 'burner-mining-drill': 1, 'lab': 1, 'automation-science-pack': 1, 'gun-turret': 1, 'firearm-magazine': 5, 'boiler': 1, 'offshore-pump': 1}, merge=True)

@pytest.fixture()
def configure_game(instance):

    def _configure_game(inventory: dict | None=None, merge: bool=False, persist_inventory: bool=False, *, reset_position: bool=True, all_technologies_researched: bool=True):
        if hasattr(instance, 'default_initial_inventory'):
            try:
                instance.initial_inventory = dict(instance.default_initial_inventory)
            except Exception:
                instance.initial_inventory = instance.default_initial_inventory
        instance.reset(reset_position=reset_position, all_technologies_researched=all_technologies_researched)
        if inventory is not None:
            print(f'Setting inventory: {inventory}')
            if merge:
                try:
                    updated = {**instance.initial_inventory, **inventory}
                except Exception:
                    updated = dict(instance.initial_inventory)
                    updated.update(inventory)
            else:
                updated = dict(inventory)
            if persist_inventory:
                instance.initial_inventory = updated
            instance.first_namespace._set_inventory(updated)
        return instance.namespace
    return _configure_game

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'stone-furnace': 1, 'boiler': 1, 'steam-engine': 1, 'offshore-pump': 4, 'pipe': 100, 'iron-plate': 50, 'copper-plate': 20, 'coal': 50, 'burner-inserter': 50, 'burner-mining-drill': 50, 'transport-belt': 50, 'stone-wall': 100, 'splitter': 4, 'wooden-chest': 1})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'iron-chest': 1})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'coal': 50, 'iron-chest': 1, 'iron-plate': 5}, merge=True, all_technologies_researched=False)

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'boiler': 5, 'transport-belt': 20, 'stone-furnace': 10, 'burner-mining-drill': 5, 'electric-furnace': 5, 'burner-inserter': 30, 'electric-mining-drill': 10, 'assembling-machine-1': 10, 'steam-engine': 5, 'pipe': 20, 'coal': 50, 'offshore-pump': 5, 'wooden-chest': 20, 'small-electric-pole': 30, 'medium-electric-pole': 10}, persist_inventory=True)

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'coal': 10, 'iron-chest': 1, 'iron-plate': 50, 'iron-ore': 10, 'stone-furnace': 1, 'offshore-pump': 1, 'assembly-machine-1': 1, 'burner-mining-drill': 1, 'lab': 1, 'automation-science-pack': 1, 'gun-turret': 1, 'firearm-magazine': 5, 'transport-belt': 200, 'boiler': 1, 'pipe': 20}, merge=True)

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'transport-belt': 12})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'assembling-machine-1': 1})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'assembling-machine-1': 1}, merge=True, all_technologies_researched=False)

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'iron-chest': 1, 'iron-ore': 500, 'copper-ore': 10, 'iron-plate': 1000, 'iron-gear-wheel': 1000, 'coal': 100, 'stone-furnace': 1, 'transport-belt': 10, 'burner-inserter': 1, 'assembling-machine-1': 1, 'solar-panel': 2})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'stone-furnace': 1, 'boiler': 1, 'steam-engine': 1, 'offshore-pump': 4, 'pipe': 100, 'iron-plate': 50, 'copper-plate': 20, 'coal': 50, 'burner-inserter': 50, 'burner-mining-drill': 50, 'transport-belt': 50, 'stone-wall': 100, 'splitter': 4, 'wooden-chest': 1})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'boiler': 1, 'transport-belt': 1, 'stone-furnace': 1, 'burner-mining-drill': 1, 'burner-inserter': 2, 'electric-mining-drill': 1, 'assembling-machine-1': 1, 'steam-engine': 1, 'pipe': 1, 'offshore-pump': 1})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'coal': 50, 'iron-chest': 1, 'iron-plate': 5, 'stone-furnace': 1})

@pytest.fixture()
def game(configure_game):
    return configure_game(all_technologies_researched=False)

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'wooden-chest': 100, 'electric-mining-drill': 10, 'steam-engine': 1, 'burner-mining-drill': 5, 'pumpjack': 1})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'iron-chest': 1, 'pipe': 10, 'assembling-machine-2': 2, 'transport-belt': 10, 'burner-inserter': 10, 'iron-plate': 10, 'assembling-machine-1': 1, 'copper-cable': 3})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'iron-chest': 1, 'iron-plate': 10, 'assembling-machine-1': 1, 'copper-cable': 3})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'iron-plate': 40, 'iron-gear-wheel': 1, 'electronic-circuit': 3, 'pipe': 1, 'copper-plate': 10})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'assembling-machine-1': 1})

@pytest.fixture()
def game(configure_game):
    return configure_game(inventory={'iron-chest': 1, 'small-electric-pole': 20, 'iron-plate': 10, 'assembling-machine-1': 1, 'pipe-to-ground': 10, 'pipe': 30, 'transport-belt': 50, 'underground-belt': 30})

