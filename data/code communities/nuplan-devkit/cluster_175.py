# Cluster 175

def _filter_scenarios(scenario_dict: ScenarioDict, total_num_scenarios: int, required_num_scenarios: int, randomize: bool) -> ScenarioDict:
    """
    Filters scenarios until we reach the user specified number of scenarios. Scenarios with scenario_type DEFAULT_SCENARIO_NAME are removed first either randomly or with equisampling, and subsequently
    the other scenarios are sampled randomly or with equisampling if necessary.
    :param scenario_dict: Dictionary containining a mapping of scenario_type to a list of the AbstractScenario objects.
    :param total_num_scenarios: Total number of scenarios in the scenario dictionary.
    :param required_num_scenarios: Number of scenarios desired.
    :param randomize: boolean to decide whether to randomize the sampling of scenarios.
    :return: Scenario dictionary with the required number of scenarios.
    """

    def _filter_scenarios_from_scenario_list(scenario_list: List[NuPlanScenario], num_scenarios_to_keep: int, randomize: bool) -> List[NuPlanScenario]:
        """
        Removes scenarios randomly or does equisampling of the scenarios.
        :param scenario_list: List of scenarios.
        :param num_scenarios_to_keep: Number of scenarios that should be in the final list.
        :param randomize: Boolean for whether to randomly sample from scenario_list or carry out equisampling of scenarios.
        """
        total_num_scenarios = len(scenario_list)
        step = max(total_num_scenarios // num_scenarios_to_keep, 1)
        scenario_list = random.sample(scenario_list, num_scenarios_to_keep) if randomize else scenario_list[::step]
        scenario_list = scenario_list[:num_scenarios_to_keep]
        return scenario_list
    if total_num_scenarios == 0 or required_num_scenarios == 0 or len(scenario_dict) == 0:
        return {}
    if DEFAULT_SCENARIO_NAME in scenario_dict:
        num_default_scenarios = len(scenario_dict[DEFAULT_SCENARIO_NAME])
        if total_num_scenarios - required_num_scenarios < num_default_scenarios:
            num_default_scenarios_to_keep = num_default_scenarios - (total_num_scenarios - required_num_scenarios)
            scenario_dict[DEFAULT_SCENARIO_NAME] = _filter_scenarios_from_scenario_list(scenario_dict[DEFAULT_SCENARIO_NAME], num_default_scenarios_to_keep, randomize)
            return scenario_dict
        else:
            scenario_dict.pop(DEFAULT_SCENARIO_NAME)
    scenario_list = scenario_dict_to_list(scenario_dict)
    scenario_list = _filter_scenarios_from_scenario_list(scenario_list, required_num_scenarios, randomize)
    scenario_dict = scenario_list_to_dict(scenario_list)
    return scenario_dict

def _filter_scenarios_from_scenario_list(scenario_list: List[NuPlanScenario], num_scenarios_to_keep: int, randomize: bool) -> List[NuPlanScenario]:
    """
        Removes scenarios randomly or does equisampling of the scenarios.
        :param scenario_list: List of scenarios.
        :param num_scenarios_to_keep: Number of scenarios that should be in the final list.
        :param randomize: Boolean for whether to randomly sample from scenario_list or carry out equisampling of scenarios.
        """
    total_num_scenarios = len(scenario_list)
    step = max(total_num_scenarios // num_scenarios_to_keep, 1)
    scenario_list = random.sample(scenario_list, num_scenarios_to_keep) if randomize else scenario_list[::step]
    scenario_list = scenario_list[:num_scenarios_to_keep]
    return scenario_list

