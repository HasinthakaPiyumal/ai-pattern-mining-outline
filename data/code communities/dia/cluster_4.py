# Cluster 4

class DenseGeneral(nn.Module):
    """
    PyTorch equivalent of flax.linen.DenseGeneral with shapes defined at init.
    Stores weights (`kernel`) in the same layout as Jax and uses torch.tensordot
    for the generalized matrix multiplication. Weight/bias shapes are calculated
    and parameters created during initialization based on config.
    `load_weights` validates shapes and copies data.
    Attributes:
        axis (Tuple[int, ...]): Input axis or axes to contract.
        in_shapes (Tuple[int, ...]): Sizes of the input dimensions specified by `axis`.
        out_features (Tuple[int, ...]): Shape of the output features (non-contracted dims).
        use_bias (bool): Whether to add a bias term.
        weight (nn.Parameter): The kernel parameter.
        bias (Optional[nn.Parameter]): The bias parameter (if use_bias=True).
    """

    def __init__(self, in_shapes: tuple[int, ...], out_features: tuple[int, ...], axis: tuple[int, ...]=(-1,), weight_dtype: torch.dtype | None=None, device: torch.device | None=None):
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis
        self.kernel_shape = self.in_shapes + self.out_features
        factory_kwargs = {'device': device, 'dtype': weight_dtype}
        self.weight = nn.Parameter(torch.empty(self.kernel_shape, **factory_kwargs))

    def forward(self, inputs: Tensor) -> Tensor:
        norm_axis = _normalize_axes(self.axis, inputs.ndim)
        kernel_contract_axes = tuple(range(len(norm_axis)))
        output = torch.tensordot(inputs.to(self.weight.dtype), self.weight, dims=(norm_axis, kernel_contract_axes)).to(inputs.dtype)
        return output

def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple((ax if ax >= 0 else ndim + ax for ax in axes))

