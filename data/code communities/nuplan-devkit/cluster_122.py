# Cluster 122

class LightningModuleWrapper(pl.LightningModule):
    """
    Lightning module that wraps the training/validation/testing procedure and handles the objective/metric computation.
    """

    def __init__(self, model: TorchModuleWrapper, objectives: List[AbstractObjective], metrics: List[AbstractTrainingMetric], batch_size: int, optimizer: Optional[DictConfig]=None, lr_scheduler: Optional[DictConfig]=None, warm_up_lr_scheduler: Optional[DictConfig]=None, objective_aggregate_mode: str='mean') -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.objectives = objectives
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warm_up_lr_scheduler = warm_up_lr_scheduler
        self.objective_aggregate_mode = objective_aggregate_mode
        model_targets = {builder.get_feature_unique_name() for builder in model.get_list_of_computed_target()}
        for objective in self.objectives:
            for feature in objective.get_list_of_required_target_types():
                assert feature in model_targets, f'Objective target: "{feature}" is not in model computed targets!'
        for metric in self.metrics:
            for feature in metric.get_list_of_required_target_types():
                assert feature in model_targets, f'Metric target: "{feature}" is not in model computed targets!'

    def _step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch
        predictions = self.forward(features)
        objectives = self._compute_objectives(predictions, targets, scenarios)
        metrics = self._compute_metrics(predictions, targets)
        loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)
        self._log_step(loss, objectives, metrics, prefix)
        return loss

    def _compute_objectives(self, predictions: TargetsType, targets: TargetsType, scenarios: ScenarioListType) -> Dict[str, torch.Tensor]:
        """
        Computes a set of learning objectives used for supervision given the model's predictions and targets.

        :param predictions: model's output signal
        :param targets: supervisory signal
        :return: dictionary of objective names and values
        """
        return {objective.name(): objective.compute(predictions, targets, scenarios) for objective in self.objectives}

    def _compute_metrics(self, predictions: TargetsType, targets: TargetsType) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        return {metric.name(): metric.compute(predictions, targets) for metric in self.metrics}

    def _log_step(self, loss: torch.Tensor, objectives: Dict[str, torch.Tensor], metrics: Dict[str, torch.Tensor], prefix: str, loss_name: str='loss') -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(f'loss/{prefix}_{loss_name}', loss)
        for key, value in objectives.items():
            self.log(f'objectives/{prefix}_{key}', value)
        for key, value in metrics.items():
            self.log(f'metrics/{prefix}_{key}', value)

    def training_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'train')

    def validation_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'val')

    def test_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'test')

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        if self.optimizer is None:
            raise RuntimeError('To train, optimizer must not be None.')
        optimizer: Optimizer = instantiate(config=self.optimizer, params=self.parameters(), lr=self.optimizer.lr)
        logger.info(f'Using optimizer: {self.optimizer._target_}')
        lr_scheduler_params: Dict[str, Union[_LRScheduler, str, int]] = build_lr_scheduler(optimizer=optimizer, lr=self.optimizer.lr, warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler, lr_scheduler_cfg=self.lr_scheduler)
        optimizer_dict: Dict[str, Any] = {}
        optimizer_dict['optimizer'] = optimizer
        if lr_scheduler_params:
            logger.info(f'Using lr_schedulers {lr_scheduler_params}')
            optimizer_dict['lr_scheduler'] = lr_scheduler_params
        return optimizer_dict if 'lr_scheduler' in optimizer_dict else optimizer_dict['optimizer']

def build_lr_scheduler(optimizer: Optimizer, lr: float, warm_up_lr_scheduler_cfg: Optional[DictConfig], lr_scheduler_cfg: Optional[DictConfig]) -> _LRScheduler:
    """
    :param optimizer: Optimizer object
    :param lr: Initial learning rate to be used. If using OneCycleLR, this will be the max learning rate to be used instead.
    :param lr_warm_up: DictConfig for warm up scheduler
    :param lr_scheduler: DictConfig for actual scheduler
    :return: aggregate lr_scheduler
    """
    lr_scheduler_params: Dict[str, Any] = {}
    if lr_scheduler_cfg is not None:
        lr_scheduler_params = _instantiate_main_lr_scheduler(optimizer=optimizer, lr_scheduler_cfg=lr_scheduler_cfg, lr=lr)
        logger.info(f'Using lr_scheduler provided: {lr_scheduler_cfg._target_}')
    if warm_up_lr_scheduler_cfg is not None:
        initial_lr = _get_lr_from_optimizer(optimizer)
        lr_scheduler_params = _instantiate_warm_up_lr_scheduler(optimizer=optimizer, warm_up_lr_scheduler_cfg=warm_up_lr_scheduler_cfg, initial_lr=initial_lr, lr_scheduler_params=lr_scheduler_params)
    else:
        logger.info('Not using any lr_schedulers.')
    return lr_scheduler_params

class TestLRSchedulerBuilder(unittest.TestCase):
    """Test update_distributed_optimizer_config function."""
    world_size = 4

    def setUp(self) -> None:
        """Setup test attributes."""
        self.lr = 5e-05
        self.weight_decay = 0.0005
        self.betas = [0.9, 0.999]
        self.mock_params = [torch.rand(1)]
        self.warm_up_lr_scheduler_cfg = DictConfig({'_target_': 'torch.optim.lr_scheduler.LambdaLR', '_convert_': 'all', 'optimizer': '', 'lr_lambda': {'_target_': 'nuplan.planning.script.builders.lr_scheduler_builder.get_warm_up_lr_scheduler_func', 'warm_up_steps': 100, 'warm_up_strategy': 'linear'}})
        self.lr_scheduler_cfg = DictConfig({'_target_': 'torch.optim.lr_scheduler.OneCycleLR', '_convert_': 'all', 'optimizer': '', 'max_lr': 5e-05, 'epochs': 1, 'steps_per_epoch': 100, 'pct_start': 0.25, 'anneal_strategy': 'cos', 'cycle_momentum': True, 'base_momentum': 0.85, 'max_momentum': 0.95, 'div_factor': 10, 'final_div_factor': 10, 'last_epoch': -1})
        self.initial_lr = self.lr / self.lr_scheduler_cfg.div_factor

    def _get_lr_from_optimizer(self, optimizer: torch.optim.Optimizer) -> float:
        lr: float
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def test_build_lr_scheduler_with_warm_up_scheduler_and_one_cycle_lr_scheduler(self) -> None:
        """Test that lr_scheduler with warm up scheduler works as expected"""
        optimizer = torch.optim.AdamW(lr=self.lr, weight_decay=self.weight_decay, betas=self.betas, params=self.mock_params)
        sequential_scheduler = build_lr_scheduler(optimizer=optimizer, lr=self.lr, warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler_cfg, lr_scheduler_cfg=self.lr_scheduler_cfg)['scheduler']
        self.assertAlmostEqual(self._get_lr_from_optimizer(sequential_scheduler.optimizer), 0.0)
        total_steps = self.lr_scheduler_cfg.steps_per_epoch * self.lr_scheduler_cfg.epochs + self.warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps
        lrs = []
        for _ in range(total_steps):
            sequential_scheduler.step()
            lrs.append(self._get_lr_from_optimizer(sequential_scheduler.optimizer))
        lr_at_end_of_warm_up = lrs[self.warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps - 1]
        lr_at_end_of_training = lrs[-1]
        max_lr = max(lrs)
        self.assertAlmostEqual(max_lr, self.lr)
        self.assertAlmostEqual(lr_at_end_of_warm_up, self.initial_lr)
        self.assertAlmostEqual(lr_at_end_of_training, self.initial_lr / self.lr_scheduler_cfg.final_div_factor)

    def test_build_lr_scheduler_with_warm_up_scheduler_and_no_main_scheduler(self) -> None:
        """Test that lr_scheduler with warm up scheduler works as expected"""
        optimizer = torch.optim.AdamW(lr=self.lr, weight_decay=self.weight_decay, betas=self.betas, params=self.mock_params)
        sequential_scheduler = build_lr_scheduler(optimizer=optimizer, lr=self.lr, warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler_cfg, lr_scheduler_cfg=None)['scheduler']
        self.assertAlmostEqual(self._get_lr_from_optimizer(sequential_scheduler.optimizer), 0.0)
        total_steps = self.lr_scheduler_cfg.steps_per_epoch * self.lr_scheduler_cfg.epochs + self.warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps
        lrs = []
        for _ in range(total_steps):
            sequential_scheduler.step()
            lrs.append(self._get_lr_from_optimizer(sequential_scheduler.optimizer))
        lr_at_end_of_warm_up = lrs[self.warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps - 1]
        lr_at_end_of_training = lrs[-1]
        self.assertAlmostEqual(lr_at_end_of_warm_up, self.lr)
        self.assertAlmostEqual(lr_at_end_of_training, self.lr)

