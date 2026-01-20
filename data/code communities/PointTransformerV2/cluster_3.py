# Cluster 3

def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD

def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)
    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []

def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]

class JSONWriter(EventWriter):
    """
    Write scalars to a json file.
    It saves scalars as one json per line (instead of a big json) for easy parsing.
    Examples parsing such a json file:
    ::
        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 19,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 39,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]
        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._file_handle = open(json_file, 'a')
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        to_save = defaultdict(dict)
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)
        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter['iteration'] = itr
            self._file_handle.write(json.dumps(scalars_per_iter, sort_keys=True) + '\n')
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self._file_handle.close()

def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(_CURRENT_STORAGE_STACK), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]

class TensorboardXWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, window_size: int=20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir, **kwargs)
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                self._writer.add_scalar(k, v, iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self._writer.add_image(img_name, img, step_num)
            storage.clear_images()
        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self._writer.add_histogram_raw(**params)
            storage.clear_histograms()

    def close(self):
        if hasattr(self, '_writer'):
            self._writer.close()

class CommonMetricPrinter(EventWriter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.
    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter: Optional[int]=None, window_size: int=20):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter
        self._window_size = window_size
        self._last_write = None

    def _get_eta(self, storage) -> Optional[str]:
        if self._max_iter is None:
            return ''
        iteration = storage.iter
        try:
            eta_seconds = storage.history('time').median(1000) * (self._max_iter - iteration - 1)
            storage.put_scalar('eta_seconds', eta_seconds, smoothing_hint=False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            eta_string = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (iteration - self._last_write[0])
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())
            return eta_string

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            return
        try:
            data_time = storage.history('data_time').avg(20)
        except KeyError:
            data_time = None
        try:
            iter_time = storage.history('time').global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = '{:.5g}'.format(storage.history('lr').latest())
        except KeyError:
            lr = 'N/A'
        eta_string = self._get_eta(storage)
        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None
        self.logger.info(' {eta}iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}'.format(eta=f'eta: {eta_string}  ' if eta_string else '', iter=iteration, losses='  '.join(['{}: {:.4g}'.format(k, v.median(self._window_size)) for k, v in storage.histories().items() if 'loss' in k]), time='time: {:.4f}  '.format(iter_time) if iter_time is not None else '', data_time='data_time: {:.4f}  '.format(data_time) if data_time is not None else '', lr=lr, memory='max_mem: {:.0f}M'.format(max_mem_mb) if max_mem_mb is not None else ''))

class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:
    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.
    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.
    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()
        "\n        We set the model to training mode in the trainer.\n        However it's valid to train a model that's in eval mode.\n        If you want your model (or a submodule of it) to behave\n        like evaluation during training, you can overwrite its train() method.\n        "
        model.train()
        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter_obj = None
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
        start = time.perf_counter()
        '\n        If you want to do something with the data, you can wrap the dataloader.\n        '
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        '\n        If you want to do something with the losses, you can wrap the model.\n        '
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {'total_loss': loss_dict}
        else:
            losses = sum(loss_dict.values())
        '\n        If you need to accumulate gradients or do something similar, you can\n        wrap the optimizer with your custom `zero_grad()` method.\n        '
        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        '\n        If you need gradient clipping/scaling or other processing, you can\n        wrap the optimizer with your custom `step()` method. But it is\n        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4\n        '
        self.optimizer.step()

    @property
    def _data_loader_iter(self):
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj

    def reset_data_loader(self, data_loader_builder):
        """
        Delete and replace the current data loader with a new one, which will be created
        by calling `data_loader_builder` (without argument).
        """
        del self.data_loader
        data_loader = data_loader_builder()
        self.data_loader = data_loader
        self._data_loader_iter_obj = None

    def _write_metrics(self, loss_dict: Mapping[str, torch.Tensor], data_time: float, prefix: str='') -> None:
        SimpleTrainer.write_metrics(loss_dict, data_time, prefix)

    @staticmethod
    def write_metrics(loss_dict: Mapping[str, torch.Tensor], data_time: float, prefix: str='') -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict['data_time'] = data_time
        all_metrics_dict = comm.gather(metrics_dict)
        if comm.is_main_process():
            storage = get_event_storage()
            data_time = np.max([x.pop('data_time') for x in all_metrics_dict])
            storage.put_scalar('data_time', data_time)
            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(f'Loss became infinite or NaN at iteration={storage.iter}!\nloss_dict = {metrics_dict}')
            storage.put_scalar('{}total_loss'.format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict['optimizer'])

