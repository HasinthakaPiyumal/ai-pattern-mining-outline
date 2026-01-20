# Cluster 23

def cast_tensor_type(inputs, src_type=None, dst_type=None):
    """Recursively convert Tensor in inputs from ``src_type`` to ``dst_type``.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype | torch.device): Source type.
        src_type (torch.dtype | torch.device): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    assert dst_type is not None
    if isinstance(inputs, torch.Tensor):
        if isinstance(dst_type, torch.device):
            if hasattr(inputs, 'to') and hasattr(inputs, 'device') and (inputs.device == src_type or src_type is None):
                return inputs.to(dst_type)
            else:
                return inputs
        elif hasattr(inputs, 'to') and hasattr(inputs, 'dtype') and (inputs.dtype == src_type or src_type is None):
            return inputs.to(dst_type)
        else:
            return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({k: cast_tensor_type(v, src_type=src_type, dst_type=dst_type) for k, v in inputs.items()})
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)((cast_tensor_type(item, src_type=src_type, dst_type=dst_type) for item in inputs))
    else:
        return inputs

class AvoidOOM:
    """Try to convert inputs to FP16 and CPU if got a PyTorch's CUDA Out of
    Memory error. It will do the following steps:

        1. First retry after calling `torch.cuda.empty_cache()`.
        2. If that still fails, it will then retry by converting inputs
          to FP16.
        3. If that still fails trying to convert inputs to CPUs.
          In this case, it expects the function to dispatch to
          CPU implementation.

    Args:
        to_cpu (bool): Whether to convert outputs to CPU if get an OOM
            error. This will slow down the code significantly.
            Defaults to True.
        test (bool): Skip `_ignore_torch_cuda_oom` operate that can use
            lightweight data in unit test, only used in
            test unit. Defaults to False.

    Examples:
        >>> from mmdet.utils.memory import AvoidOOM
        >>> AvoidCUDAOOM = AvoidOOM()
        >>> output = AvoidOOM.retry_if_cuda_oom(
        >>>     some_torch_function)(input1, input2)
        >>> # To use as a decorator
        >>> # from mmdet.utils import AvoidCUDAOOM
        >>> @AvoidCUDAOOM.retry_if_cuda_oom
        >>> def function(*args, **kwargs):
        >>>     return None
    ```

    Note:
        1. The output may be on CPU even if inputs are on GPU. Processing
            on CPU will slow down the code significantly.
        2. When converting inputs to CPU, it will only look at each argument
            and check if it has `.device` and `.to` for conversion. Nested
            structures of tensors are not supported.
        3. Since the function might be called more than once, it has to be
            stateless.
    """

    def __init__(self, to_cpu=True, test=False):
        self.to_cpu = to_cpu
        self.test = test

    def retry_if_cuda_oom(self, func):
        """Makes a function retry itself after encountering pytorch's CUDA OOM
        error.

        The implementation logic is referred to
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py

        Args:
            func: a stateless callable that takes tensor-like objects
                as arguments.
        Returns:
            func: a callable which retries `func` if OOM is encountered.
        """

        @wraps(func)
        def wrapped(*args, **kwargs):
            if not self.test:
                with _ignore_torch_cuda_oom():
                    return func(*args, **kwargs)
                torch.cuda.empty_cache()
                with _ignore_torch_cuda_oom():
                    return func(*args, **kwargs)
            dtype, device = (None, None)
            values = args + tuple(kwargs.values())
            for value in values:
                if isinstance(value, torch.Tensor):
                    dtype = value.dtype
                    device = value.device
                    break
            if dtype is None or device is None:
                raise ValueError('There is no tensor in the inputs, cannot get dtype and device.')
            fp16_args = cast_tensor_type(args, dst_type=torch.half)
            fp16_kwargs = cast_tensor_type(kwargs, dst_type=torch.half)
            logger = get_root_logger()
            logger.warning(f'Attempting to copy inputs of {str(func)} to FP16 due to CUDA OOM')
            with _ignore_torch_cuda_oom():
                output = func(*fp16_args, **fp16_kwargs)
                output = cast_tensor_type(output, src_type=torch.half, dst_type=dtype)
                if not self.test:
                    return output
            logger.warning('Using FP16 still meet CUDA OOM')
            if self.to_cpu:
                logger.warning(f'Attempting to copy inputs of {str(func)} to CPU due to CUDA OOM')
                cpu_device = torch.empty(0).device
                cpu_args = cast_tensor_type(args, dst_type=cpu_device)
                cpu_kwargs = cast_tensor_type(kwargs, dst_type=cpu_device)
                with _ignore_torch_cuda_oom():
                    logger.warning(f'Convert outputs to GPU (device={device})')
                    output = func(*cpu_args, **cpu_kwargs)
                    output = cast_tensor_type(output, src_type=cpu_device, dst_type=device)
                    return output
                warnings.warn('Cannot convert output to GPU due to CUDA OOM, the output is now on CPU, which might cause errors if the output need to interact with GPU data in subsequent operations')
                logger.warning('Cannot convert output to GPU due to CUDA OOM, the output is on CPU now.')
                return func(*cpu_args, **cpu_kwargs)
            else:
                return func(*args, **kwargs)
        return wrapped

@contextmanager
def _ignore_torch_cuda_oom():
    """A context which ignores CUDA OOM exception from pytorch.

    Code is modified from
    <https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py>  # noqa: E501
    """
    try:
        yield
    except RuntimeError as e:
        if 'CUDA out of memory. ' in str(e):
            pass
        else:
            raise

@IOU_CALCULATORS.register_module()
class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (n, 4) in <x1, y1, x2, y2>
                format, shape (n, 5) in <x1, y1, x2, y2, score> format, or be
                empty.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        if self.dtype == 'fp16':
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                overlaps = overlaps.float()
            return overlaps
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(scale={self.scale}, dtype={self.dtype})'
        return repr_str

def test_cast_tensor_type():
    inputs = torch.rand(10)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    with pytest.raises(AssertionError):
        cast_tensor_type(inputs, src_type=None, dst_type=None)
    out = cast_tensor_type(10.0, dst_type=torch.half)
    assert out == 10.0 and isinstance(out, float)
    fp16_out = cast_tensor_type(inputs, dst_type=torch.half)
    assert fp16_out.dtype == torch.half
    fp32_out = cast_tensor_type(fp16_out, dst_type=torch.float32)
    assert fp32_out.dtype == torch.float32
    list_input = [inputs, inputs]
    list_outs = cast_tensor_type(list_input, dst_type=torch.half)
    assert len(list_outs) == len(list_input) and isinstance(list_outs, list)
    for out in list_outs:
        assert out.dtype == torch.half
    dict_input = {'test1': inputs, 'test2': inputs}
    dict_outs = cast_tensor_type(dict_input, dst_type=torch.half)
    assert len(dict_outs) == len(dict_input) and isinstance(dict_outs, dict)
    if torch.cuda.is_available():
        cpu_device = torch.empty(0).device
        gpu_device = inputs.device
        cpu_out = cast_tensor_type(inputs, dst_type=cpu_device)
        assert cpu_out.device == cpu_device
        gpu_out = cast_tensor_type(inputs, dst_type=gpu_device)
        assert gpu_out.device == gpu_device

def test_cast_tensor_type():
    inputs = torch.FloatTensor([5.0])
    src_type = torch.float32
    dst_type = torch.int32
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.dtype == dst_type
    inputs = 'tensor'
    src_type = str
    dst_type = str
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, str)
    inputs = np.array([5.0])
    src_type = np.ndarray
    dst_type = np.ndarray
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, np.ndarray)
    inputs = dict(tensor_a=torch.FloatTensor([1.0]), tensor_b=torch.FloatTensor([2.0]))
    src_type = torch.float32
    dst_type = torch.int32
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, dict)
    assert outputs['tensor_a'].dtype == dst_type
    assert outputs['tensor_b'].dtype == dst_type
    inputs = [torch.FloatTensor([1.0]), torch.FloatTensor([2.0])]
    src_type = torch.float32
    dst_type = torch.int32
    outputs = cast_tensor_type(inputs, src_type, dst_type)
    assert isinstance(outputs, list)
    assert outputs[0].dtype == dst_type
    assert outputs[1].dtype == dst_type
    inputs = 5
    outputs = cast_tensor_type(inputs, None, None)
    assert isinstance(outputs, int)

