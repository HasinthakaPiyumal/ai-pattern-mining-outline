# Cluster 85

class TestSequentialLRScheduler(unittest.TestCase):
    """Test update_distributed_optimizer_config function."""
    world_size = 4

    def setUp(self) -> None:
        """Set up schedulers and optimizer"""
        self.optimizer = Adam(params=[torch.tensor([1.0])], lr=0.0001)
        self.total_steps = 300
        self.milestones = [100, 200]
        self.scheduler3 = LambdaLR(self.optimizer, lambda step: 1)
        self.scheduler2 = LambdaLR(self.optimizer, self._get_mock_linear_func())
        self.scheduler1 = LambdaLR(self.optimizer, self._get_mock_linear_func())
        self.schedulers = [self.scheduler1, self.scheduler2, self.scheduler3]
        self.expected_lrs = [1e-06, 0.0001, 0.0001, 0.0001]

    def _get_mock_linear_func(self) -> Callable[..., float]:
        """Gets mock linear function"""

        def _linear_func(step: int) -> float:
            num_steps = self.milestones[1] - self.milestones[0]
            return step / num_steps if step <= num_steps else 1.0
        return _linear_func

    def _get_lr_from_optimizer(self, optimizer: torch.optim.Optimizer) -> float:
        lr: float
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def test_sequential_lr_with_multiple_schedulers(self) -> None:
        """Tests that SequentialLR Scheduler works with multiple schedulers."""
        sequential_lr_scheduler = SequentialLR(self.optimizer, self.schedulers, self.milestones)
        lr = []
        self.assertAlmostEqual(self._get_lr_from_optimizer(sequential_lr_scheduler.optimizer), 0.0)
        for i in range(self.total_steps):
            sequential_lr_scheduler.step()
            print(i + 1, self._get_lr_from_optimizer(sequential_lr_scheduler.optimizer))
            lr.append(self._get_lr_from_optimizer(sequential_lr_scheduler.optimizer))
        self.assertAlmostEqual(lr[0], self.expected_lrs[0])
        self.assertAlmostEqual(lr[self.milestones[0] - 1], self.expected_lrs[1])
        self.assertAlmostEqual(lr[self.milestones[1] - 1], self.expected_lrs[2])
        self.assertAlmostEqual(lr[-1], self.expected_lrs[-1])

def _instantiate_warm_up_lr_scheduler(optimizer: Optimizer, warm_up_lr_scheduler_cfg: DictConfig, initial_lr: float, lr_scheduler_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Instantiates the warm up learning rate scheduler to be used during training.
    :param optimizer: Optimizer used for training.
    :param warm_up_lr_scheduler_cfg: Learning rate scheduler config for warm_up phase
    :param initial_lr: Initial learning rate. To be scaled down further during warm_up phase.
    :param lr: Learning rate to be used. If using OneCycleLR, then this is the maximum learning rate to be reached during training.
    :return: Learning rate scheduler and associated parameters
    """
    using_main_lr_scheduler = 'scheduler' in lr_scheduler_params
    if using_main_lr_scheduler:
        warm_up_lr_scheduler = instantiate(config=warm_up_lr_scheduler_cfg, optimizer=optimizer, lr_lambda=instantiate(config=warm_up_lr_scheduler_cfg.lr_lambda, final_div_factor=1.0))
        lr_schedulers = [warm_up_lr_scheduler, lr_scheduler_params['scheduler']]
        sequential_lr_scheduler = SequentialLR(optimizer=optimizer, schedulers=lr_schedulers, milestones=[warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps])
        lr_scheduler_params['scheduler'] = sequential_lr_scheduler
        logger.info(f'Added Warm up learning rate scheduler before main scheduler with {warm_up_lr_scheduler_cfg.lr_lambda.warm_up_strategy} strategy.')
    else:
        warm_up_lr_scheduler = instantiate(config=warm_up_lr_scheduler_cfg, optimizer=optimizer, lr_lambda=instantiate(config=warm_up_lr_scheduler_cfg.lr_lambda, final_div_factor=1.0))
        warm_up_phase_initial_lr = initial_lr / warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps
        for group in optimizer.param_groups:
            group['initial_lr'] = warm_up_phase_initial_lr
        lr_scheduler_params['scheduler'] = warm_up_lr_scheduler
        logger.info(f'Using Warm up learning rate scheduler with {warm_up_lr_scheduler_cfg.lr_lambda.warm_up_strategy} strategy.')
    return lr_scheduler_params

