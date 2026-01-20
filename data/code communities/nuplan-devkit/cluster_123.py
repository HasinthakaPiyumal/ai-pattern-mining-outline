# Cluster 123

class TrajectoryWeightDecayImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    When comparing model's predictions to expert's trajectory, it assigns more weight to earlier timestamps than later ones.
    Formula: mean(loss * exp(-index / poses))
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float=1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'trajectory_weight_decay_imitation_objective'
        self._weight = weight
        self._decay = 1.0
        self._fn_xy = torch.nn.modules.loss.L1Loss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ['trajectory']

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(Trajectory, predictions['trajectory'])
        targets_trajectory = cast(Trajectory, targets['trajectory'])
        loss_weights = extract_scenario_type_weight(scenarios, self._scenario_type_loss_weighting, device=predicted_trajectory.xy.device)
        planner_output_steps = predicted_trajectory.xy.shape[1]
        decay_weight = torch.ones_like(predicted_trajectory.xy)
        decay_value = torch.exp(-torch.Tensor(range(planner_output_steps)) / (planner_output_steps * self._decay))
        decay_weight[:, :] = decay_value.unsqueeze(1)
        broadcast_shape_xy = tuple([-1] + [1 for _ in range(predicted_trajectory.xy.dim() - 1)])
        broadcasted_loss_weights_xy = loss_weights.view(broadcast_shape_xy)
        broadcast_shape_heading = tuple([-1] + [1 for _ in range(predicted_trajectory.heading.dim() - 1)])
        broadcasted_loss_weights_heading = loss_weights.view(broadcast_shape_heading)
        weighted_xy_loss = self._fn_xy(predicted_trajectory.xy, targets_trajectory.xy) * broadcasted_loss_weights_xy
        weighted_heading_loss = self._fn_heading(predicted_trajectory.heading, targets_trajectory.heading) * broadcasted_loss_weights_heading
        assert weighted_xy_loss.size() == predicted_trajectory.xy.size()
        assert weighted_heading_loss.size() == predicted_trajectory.heading.size()
        return self._weight * (torch.mean(weighted_xy_loss * decay_weight) + torch.mean(weighted_heading_loss * decay_weight[:, :, 0]))

def extract_scenario_type_weight(scenarios: List[AbstractScenario], scenario_type_loss_weights: Dict[str, float], device: torch.device) -> torch.Tensor:
    """
    Gets the scenario loss weights.
    :param scenarios: List of scenario objects
    :return: Tensor with scenario_weights
    """
    default_scenario_weight = 1.0
    scenario_weights = [scenario_type_loss_weights.get(s.scenario_type, default_scenario_weight) for s in scenarios]
    return torch.FloatTensor(scenario_weights).to(device)

class ImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float=1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'imitation_objective'
        self._weight = weight
        self._fn_xy = torch.nn.modules.loss.MSELoss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ['trajectory']

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(Trajectory, predictions['trajectory'])
        targets_trajectory = cast(Trajectory, targets['trajectory'])
        loss_weights = extract_scenario_type_weight(scenarios, self._scenario_type_loss_weighting, device=predicted_trajectory.xy.device)
        broadcast_shape_xy = tuple([-1] + [1 for _ in range(predicted_trajectory.xy.dim() - 1)])
        broadcasted_loss_weights_xy = loss_weights.view(broadcast_shape_xy)
        broadcast_shape_heading = tuple([-1] + [1 for _ in range(predicted_trajectory.heading.dim() - 1)])
        broadcasted_loss_weights_heading = loss_weights.view(broadcast_shape_heading)
        weighted_xy_loss = self._fn_xy(predicted_trajectory.xy, targets_trajectory.xy) * broadcasted_loss_weights_xy
        weighted_heading_loss = self._fn_heading(predicted_trajectory.heading, targets_trajectory.heading) * broadcasted_loss_weights_heading
        assert weighted_xy_loss.size() == predicted_trajectory.xy.size()
        assert weighted_heading_loss.size() == predicted_trajectory.heading.size()
        return self._weight * (torch.mean(weighted_xy_loss) + torch.mean(weighted_heading_loss))

