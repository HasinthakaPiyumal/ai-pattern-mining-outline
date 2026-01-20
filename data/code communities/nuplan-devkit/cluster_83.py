# Cluster 83

class RayDistributed(WorkerPool):
    """
    This worker uses ray to distribute work across all available threads.
    """

    def __init__(self, master_node_ip: Optional[str]=None, threads_per_node: Optional[int]=None, debug_mode: bool=False, log_to_driver: bool=True, output_dir: Optional[Union[str, Path]]=None, logs_subdir: Optional[str]='logs', use_distributed: bool=False):
        """
        Initialize ray worker.
        :param master_node_ip: if available, ray will connect to remote cluster.
        :param threads_per_node: Number of threads to use per node.
        :param debug_mode: If true, the code will be executed serially. This
            is useful for debugging.
        :param log_to_driver: If true, the output from all of the worker
                processes on all nodes will be directed to the driver.
        :param output_dir: Experiment output directory.
        :param logs_subdir: Subdirectory inside experiment dir to store worker logs.
        :param use_distributed: Boolean flag to explicitly enable/disable distributed computation
        """
        self._master_node_ip = master_node_ip
        self._threads_per_node = threads_per_node
        self._local_mode = debug_mode
        self._log_to_driver = log_to_driver
        self._log_dir: Optional[Path] = Path(output_dir) / (logs_subdir or '') if output_dir is not None else None
        self._use_distributed = use_distributed
        super().__init__(self.initialize())

    def initialize(self) -> WorkerResources:
        """
        Initialize ray.
        :return: created WorkerResources.
        """
        if ray.is_initialized():
            logger.warning('Ray is running, we will shut it down before starting again!')
            ray.shutdown()
        return initialize_ray(master_node_ip=self._master_node_ip, threads_per_node=self._threads_per_node, local_mode=self._local_mode, log_to_driver=self._log_to_driver, use_distributed=self._use_distributed)

    def shutdown(self) -> None:
        """
        Shutdown the worker and clear memory.
        """
        ray.shutdown()

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool=False) -> List[Any]:
        """Inherited, see superclass."""
        del verbose
        return ray_map(task, *item_lists, log_dir=self._log_dir)

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        remote_fn = ray.remote(task.fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids: ray._raylet.ObjectRef = remote_fn.remote(*args, **kwargs)
        return object_ids.future()

def ray_map(task: Task, *item_lists: Iterable[List[Any]], log_dir: Optional[Path]=None) -> List[Any]:
    """
    Initialize ray, align item lists and map each item of a list of arguments to a callable and executes in parallel.
    :param task: callable to be run
    :param item_lists: items to be parallelized
    :param log_dir: directory to store worker logs
    :return: list of outputs
    """
    try:
        results = _ray_map_items(task, *item_lists, log_dir=log_dir)
        return results
    except (RayTaskError, Exception) as exc:
        ray.shutdown()
        traceback.print_exc()
        raise RuntimeError(exc)

def _ray_map_items(task: Task, *item_lists: Iterable[List[Any]], log_dir: Optional[Path]=None) -> List[Any]:
    """
    Map each item of a list of arguments to a callable and executes in parallel.
    :param fn: callable to be run
    :param item_list: items to be parallelized
    :param log_dir: directory to store worker logs
    :return: list of outputs
    """
    assert len(item_lists) > 0, 'No map arguments received for mapping'
    assert all((isinstance(items, list) for items in item_lists)), 'All map arguments must be lists'
    assert all((len(cast(List, items)) == len(item_lists[0]) for items in item_lists)), 'All lists must have equal size'
    fn = task.fn
    if isinstance(fn, partial):
        _, _, pack = fn.__reduce__()
        fn, _, args, _ = pack
        fn = wrap_function(fn, log_dir=log_dir)
        remote_fn: RemoteFunction = ray.remote(fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids = [remote_fn.remote(*items, **args) for items in zip(*item_lists)]
    else:
        fn = wrap_function(fn, log_dir=log_dir)
        remote_fn = ray.remote(fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids = [remote_fn.remote(*items) for items in zip(*item_lists)]
    object_result_map = dict.fromkeys(object_ids, None)
    for object_id, output in tqdm(_ray_object_iterator(object_ids), total=len(object_ids), desc='Ray objects'):
        object_result_map[object_id] = output
    results = list(object_result_map.values())
    return results

def wrap_function(fn: Callable[..., Any], log_dir: Optional[Path]=None) -> Callable[..., Any]:
    """
    Wraps a function to save its logs to a unique file inside the log directory.
    :param fn: function to be wrapped.
    :param log_dir: directory to store logs (wrapper function does nothing if it's not set).
    :return: wrapped function which changes logging settings while it runs.
    """

    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if log_dir is None:
            return fn(*args, **kwargs)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{uuid1().hex}__{fn.__name__}.log'
        logging.basicConfig()
        logger = logging.getLogger()
        fh = logging.FileHandler(log_path, delay=True)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        result = fn(*args, **kwargs)
        fh.flush()
        fh.close()
        logger.removeHandler(fh)
        return result
    return wrapped_fn

def _ray_object_iterator(initial_ids: List[ray.ObjectRef]) -> Iterator[Tuple[ray.ObjectRef, Any]]:
    """
    Iterator that waits for each ray object in the input object list to be completed and fetches the result.
    :param initial_ids: list of ray object ids
    :yield: result of worker
    """
    next_ids = initial_ids
    while next_ids:
        ready_ids, not_ready_ids = ray.wait(next_ids)
        next_id = ready_ids[0]
        yield (next_id, ray.get(next_id))
        next_ids = not_ready_ids

