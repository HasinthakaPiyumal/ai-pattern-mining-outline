# Cluster 93

class Context:
    """
    Stores the context information of the executor, i.e.,
    if using spark, name of the application, current spark executors,
    if using horovod: current rank etc.
    """

    def __init__(self, user_provided_gpu_conf=[]):
        self._user_provided_gpu_conf = user_provided_gpu_conf
        self._gpus = self._populate_gpu_ids()

    @property
    def gpus(self):
        return self._gpus

    def _populate_gpu_from_config(self) -> List:
        available_gpus = [i for i in range(get_gpu_count())]
        return list(set(available_gpus) & set(self._user_provided_gpu_conf))

    def _populate_gpu_from_env(self) -> List:
        gpu_conf = map(lambda x: x.strip(), os.environ.get('CUDA_VISIBLE_DEVICES', '').strip().split(','))
        gpu_conf = list(filter(lambda x: x, gpu_conf))
        gpu_conf = [int(gpu_id) for gpu_id in gpu_conf]
        available_gpus = [i for i in range(get_gpu_count())]
        return list(set(available_gpus) & set(gpu_conf))

    def _populate_gpu_ids(self) -> List:
        if not is_gpu_available():
            return []
        gpus = self._populate_gpu_from_config()
        if len(gpus) == 0:
            gpus = self._populate_gpu_from_env()
        return gpus

    def _select_random_gpu(self) -> str:
        """
        A random GPU selection strategy
        Returns:
            (str): GPU device ID
        """
        return random.choice(self.gpus)

    def gpu_device(self) -> str:
        """
        Selects a GPU on which the task can be executed
        Returns:
             (str): GPU device ID
        """
        if self.gpus:
            return self._select_random_gpu()
        return NO_GPU

def get_gpu_count() -> int:
    """
    Check number of GPUs through Torch.
    """
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0

