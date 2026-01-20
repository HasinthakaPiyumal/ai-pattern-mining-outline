# Cluster 81

def align_size_of_arguments(*item_lists: Iterable[List[Any]]) -> Tuple[int, Iterable[List[Any]]]:
    """
    Align item lists by repeating elements in order to achieve the same size.
        eg. [db, [arg1, arg2] -> [[db, db], [arg1, arg2]].
    :param item_lists: multiple arguments which will be used to call a function.
    :return: arguments with same dimension, e.g., [[db, db], [arg1, arg1]].
    """
    max_size = get_max_size_of_arguments(*item_lists)
    aligned_item_lists = [items if isinstance(items, list) else [items] * max_size for items in item_lists]
    return (max_size, aligned_item_lists)

def get_max_size_of_arguments(*item_lists: Iterable[List[Any]]) -> int:
    """
    Find the argument with most elements.
        e.g. [db, [arg1, arg2] -> 2.
    :param item_lists: arguments where some of the arguments is a list.
    :return: size of largest list.
    """
    lengths = [len(items) for items in item_lists if isinstance(items, list)]
    if len(list(set(lengths))) > 1:
        raise RuntimeError(f'There exists lists with different element size = {lengths}!')
    return max(lengths) if len(lengths) != 0 else 1

class WorkerPool(abc.ABC):
    """
    This class executed function on list of arguments. This can either be distributed/parallel or sequential worker.
    """

    def __init__(self, config: WorkerResources):
        """
        Initialize worker with resource description.
        :param config: setup of this worker.
        """
        self.config = config
        if self.config.number_of_threads < 1:
            raise RuntimeError(f'Number of threads can not be 0, and it is {self.config.number_of_threads}!')
        logger.info(f'Worker: {self.__class__.__name__}')
        logger.info(f'{self}')

    def map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool=False) -> List[Any]:
        """
        Run function with arguments from item_lists, this function will make sure all arguments have the same
        number of elements.
        :param task: function to be run.
        :param item_lists: arguments to the function.
        :param verbose: Whether to increase logger verbosity.
        :return: type from the fn.
        """
        max_size, aligned_item_lists = align_size_of_arguments(*item_lists)
        if verbose:
            logger.info(f'Submitting {max_size} tasks!')
        return self._map(task, *aligned_item_lists, verbose=verbose)

    @abc.abstractmethod
    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool=False) -> List[Any]:
        """
        Run function with arguments from item_lists. This function can assume that all the args in item_lists have
        the same number of elements.
        :param fn: function to be run.
        :param item_lists: arguments to the function.
        :param number_of_elements: number of calls to the function.
        :return: type from the fn.
        """

    @abc.abstractmethod
    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """
        Submit a task to the worker.
        :param task: to be submitted.
        :param args: arguments for the task.
        :param kwargs: keyword arguments for the task.
        :return: future.
        """
        pass

    @property
    def number_of_threads(self) -> int:
        """
        :return: the number of available threads across all nodes.
        """
        return self.config.number_of_threads

    def __str__(self) -> str:
        """
        :return: string with information about this worker.
        """
        return f'Number of nodes: {self.config.number_of_nodes}\nNumber of CPUs per node: {self.config.number_of_cpus_per_node}\nNumber of GPUs per node: {self.config.number_of_gpus_per_node}\nNumber of threads across all nodes: {self.config.number_of_threads}'

class SingleMachineParallelExecutor(WorkerPool):
    """
    This worker distributes all tasks across multiple threads on this machine.
    """

    def __init__(self, use_process_pool: bool=False, max_workers: Optional[int]=None):
        """
        Create worker with limited threads.
        :param use_process_pool: if true, ProcessPoolExecutor will be used as executor, otherwise ThreadPoolExecutor.
        :param max_workers: if available, use this number as used number of threads.
        """
        number_of_cpus_per_node = max_workers if max_workers else WorkerResources.current_node_cpu_count()
        super().__init__(WorkerResources(number_of_nodes=1, number_of_cpus_per_node=number_of_cpus_per_node, number_of_gpus_per_node=0))
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) if use_process_pool else concurrent.futures.ThreadPoolExecutor(max_workers=number_of_cpus_per_node)

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool=False) -> List[Any]:
        """Inherited, see superclass."""
        return list(tqdm(self._executor.map(task.fn, *item_lists), leave=False, total=get_max_size_of_arguments(*item_lists), desc='SingleMachineParallelExecutor', disable=not verbose))

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        return self._executor.submit(task.fn, *args, **kwargs)

class Sequential(WorkerPool):
    """
    This function does execute all functions sequentially.
    """

    def __init__(self) -> None:
        """
        Initialize simple sequential worker.
        """
        super().__init__(WorkerResources(number_of_nodes=1, number_of_cpus_per_node=1, number_of_gpus_per_node=0))

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool=False) -> List[Any]:
        """Inherited, see superclass."""
        if task.num_cpus not in [None, 1]:
            raise ValueError(f'Expected num_cpus to be 1 or unset for Sequential worker, got {task.num_cpus}')
        output = [task.fn(*args) for args in tqdm(zip(*item_lists), leave=False, total=get_max_size_of_arguments(*item_lists), desc='Sequential', disable=not verbose)]
        return output

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        raise NotImplementedError

