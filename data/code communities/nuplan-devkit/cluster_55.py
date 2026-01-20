# Cluster 55

def approximate_derivatives_tensor(y: torch.Tensor, x: torch.Tensor, window_length: int=5, poly_order: int=2, deriv_order: int=1) -> torch.Tensor:
    """
    Given a time series [y], and [x], approximate [dy/dx].
    :param y: Input tensor to filter.
    :param x: Time dimension for tensor to filter.
    :param window_length: The size of the window to use.
    :param poly_order: The order of polymonial to use when filtering.
    :deriv_order: The order of derivitave to use when filtering.
    :return: The differentiated tensor.
    """
    _validate_approximate_derivatives_shapes(y, x)
    window_length = min(window_length, x.shape[0])
    if not poly_order < window_length:
        raise ValueError(f'{poly_order} < {window_length} does not hold!')
    dx = torch.diff(x)
    min_increase = float(torch.min(dx).item())
    if min_increase <= 0:
        raise RuntimeError('dx is not monotonically increasing!')
    dx = dx.mean()
    derivative: torch.Tensor = _torch_savgol_filter(y, poly_order=poly_order, window_length=window_length, deriv_order=deriv_order, delta=dx)
    return derivative

def _validate_approximate_derivatives_shapes(y: torch.Tensor, x: torch.Tensor) -> None:
    """
    Validates that the shapes for approximate_derivatives_tensor are correct.
    :param y: The Y input.
    :param x: The X input.
    """
    if len(y.shape) == 2 and len(x.shape) == 1 and (y.shape[1] == x.shape[0]):
        return
    raise ValueError(f'Unexpected tensor shapes in approximate_derivatives_tensor: y.shape = {y.shape}, x.shape = {x.shape}')

def _torch_savgol_filter(y: torch.Tensor, window_length: int, poly_order: int, deriv_order: int, delta: float) -> torch.Tensor:
    """
    Perform Savinsky Golay filtering on the given tensor.
    This is adapted from the scipy method `scipy.signal.savgol_filter`
        However, it currently only works with window_length of 3.
    :param y: The tensor to filter. Should be of dimension 2.
    :param window_length: The window length to use.
        Currently provided as a parameter, but for now must be 3.
    :param poly_order: The polynomial order to use.
    :param deriv_order: The order of derivitave to use.
    :coefficients: The Savinsky Golay coefficients to use.
    :return: The filtered tensor.
    """
    if window_length != 3:
        raise ValueError('This method has unexpected edge behavior for window_length != 3.')
    if len(y.shape) != 2:
        raise ValueError(f'Unexpected input tensor shape to _torch_savgol_filter(): {y.shape}')
    halflen, rem = divmod(window_length, 2)
    if rem == 0:
        pos = halflen - 0.5
    else:
        pos = float(halflen)
    x = torch.arange(-pos, window_length - pos, dtype=torch.float64)
    order = torch.arange(poly_order + 1).reshape(-1, 1)
    yy = torch.zeros(poly_order + 1, dtype=torch.float64)
    A = x ** order
    yy[deriv_order] = math.factorial(deriv_order) / delta ** deriv_order
    coeffs, _, _, _ = torch.linalg.lstsq(A, yy)
    y_in = y.unsqueeze(1)
    coeffs_in = coeffs.reshape(1, 1, -1)
    result = torch.nn.functional.conv1d(y_in, coeffs_in, padding='same').reshape(y.shape)
    n = result.shape[1]
    result[:, 0] = y[:, 1] - y[:, 0]
    result[:, n - 1] = y[:, n - 1] - y[:, n - 2]
    return result

class TestTorchMath(unittest.TestCase):
    """
    A class to test the functionality of the scriptable torch math library.
    """

    def test_approximate_derivatives_tensor_functionality(self) -> None:
        """
        Tests the numerical accuracy of approximate_derivatives_tensor.
        """
        input_y = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float64)
        input_x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)
        window_length = 3
        poly_order = 2
        deriv_order = 1
        expected_output = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.float64)
        actual_output = approximate_derivatives_tensor(input_y, input_x, window_length, poly_order, deriv_order)
        torch.testing.assert_allclose(expected_output, actual_output)

    def test_approximate_derivatives_tensor_scripts_properly(self) -> None:
        """
        Tests that approximate_derivatives_tensor scripts properly.
        """

        class tmp_module(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                output = approximate_derivatives_tensor(y, x, window_length=3, poly_order=2, deriv_order=1)
                return output
        to_script = tmp_module()
        scripted = torch.jit.script(to_script)
        test_y = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float64)
        test_x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)
        py_result = to_script.forward(test_x, test_y)
        script_result = scripted.forward(test_x, test_y)
        torch.testing.assert_allclose(py_result, script_result)

    def test_unwrap_functionality(self) -> None:
        """
        Tests that the unwrap function behaves in the same way as np.unwrap
        """
        threshold = 1e-06
        signal = torch.tensor([-math.pi - threshold, -math.pi + threshold, -threshold, 0, threshold, math.pi - threshold, math.pi + threshold], dtype=torch.float64)
        signal_np = torch.from_numpy(np.unwrap(signal.numpy(), axis=-1))
        signal_torch = unwrap(signal, dim=-1)
        self.assertTrue(torch.allclose(signal_np, signal_torch, atol=threshold, rtol=0))

    def test_unwrap_scripts_properly(self) -> None:
        """
        Tests that unwrap scripts properly.
        """

        class tmp_module(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                output = unwrap(x, dim=-1)
                return output
        to_script = tmp_module()
        scripted = torch.jit.script(to_script)
        test_x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)
        py_result = to_script.forward(test_x)
        script_result = scripted.forward(test_x)
        torch.testing.assert_allclose(py_result, script_result)

class tmp_module(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = unwrap(x, dim=-1)
        return output

def unwrap(angles: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """
    This unwraps a signal p by changing elements which have an absolute difference from their
    predecessor of more than Pi to their period-complementary values.
    It is meant to mimic numpy.unwrap (https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html)
    :param angles: The tensor to unwrap.
    :param dim: Axis where the unwrap operation is performed.
    :return: Unwrapped tensor.
    """
    pi = torch.tensor(math.pi, dtype=torch.float64)
    angle_diff = torch.diff(angles, dim=dim)
    nn_functional_pad_args = [(0, 0) for _ in range(len(angles.shape))]
    nn_functional_pad_args[dim] = (1, 0)
    pad_arg: List[int] = []
    for value in nn_functional_pad_args[::-1]:
        pad_arg.append(value[0])
        pad_arg.append(value[1])
    dphi = torch.nn.functional.pad(angle_diff, pad_arg)
    dphi_m = (dphi + pi) % (2.0 * pi) - pi
    dphi_m[(dphi_m == -pi) & (dphi > 0)] = pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < pi] = 0
    return angles + phi_adj.cumsum(dim)

def state_se2_tensor_to_transform_matrix(input_data: torch.Tensor, precision: Optional[torch.dtype]=None) -> torch.Tensor:
    """
    Transforms a state of the form [x, y, heading] into a 3x3 transform matrix.
    :param input_data: the input data as a 3-d tensor.
    :return: The output 3x3 transformation matrix.
    """
    _validate_state_se2_tensor_shape(input_data, expected_first_dim=1)
    if precision is None:
        precision = input_data.dtype
    x: float = float(input_data[0].item())
    y: float = float(input_data[1].item())
    h: float = float(input_data[2].item())
    cosine: float = math.cos(h)
    sine: float = math.sin(h)
    return torch.tensor([[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]], dtype=precision, device=input_data.device)

def _validate_state_se2_tensor_shape(tensor: torch.Tensor, expected_first_dim: Optional[int]=None) -> None:
    """
    Validates that a tensor is of the proper shape for a tensorized StateSE2.
    :param tensor: The tensor to validate.
    :param expected_first_dim: The expected first dimension. Can be one of three values:
        * 1: Tensor is expected to be of shape (3,)
        * 2: Tensor is expected to be of shape (N, 3)
        * None: Either shape is acceptable
    """
    expected_feature_dim = 3
    if len(tensor.shape) == 2 and tensor.shape[1] == expected_feature_dim:
        if expected_first_dim is None or expected_first_dim == 2:
            return
    if len(tensor.shape) == 1 and tensor.shape[0] == expected_feature_dim:
        if expected_first_dim is None or expected_first_dim == 1:
            return
    raise ValueError(f'Improper se2 tensor shape: {tensor.shape}')

def state_se2_tensor_to_transform_matrix_batch(input_data: torch.Tensor, precision: Optional[torch.dtype]=None) -> torch.Tensor:
    """
    Transforms a tensor of states of the form Nx3 (x, y, heading) into a Nx3x3 transform tensor.
    :param input_data: the input data as a Nx3 tensor.
    :param precision: The precision with which to create the output tensor. If None, then it will be inferred from the input tensor.
    :return: The output Nx3x3 batch transformation tensor.
    """
    _validate_state_se2_tensor_batch_shape(input_data)
    if precision is None:
        precision = input_data.dtype
    processed_input = torch.column_stack((input_data[:, 0], input_data[:, 1], torch.cos(input_data[:, 2]), torch.sin(input_data[:, 2]), torch.ones_like(input_data[:, 0], dtype=precision)))
    reshaping_tensor = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=precision, device=input_data.device)
    return (processed_input @ reshaping_tensor).reshape(-1, 3, 3)

def _validate_state_se2_tensor_batch_shape(tensor: torch.Tensor) -> None:
    """
    Validates that a tensor is of the proper shape for a batch of tensorized StateSE2.
    :param tensor: The tensor to validate.
    """
    expected_feature_dim = 3
    if len(tensor.shape) == 2 and tensor.shape[1] == expected_feature_dim:
        return
    raise ValueError(f'Improper se2 tensor batch shape: {tensor.shape}')

def transform_matrix_to_state_se2_tensor(input_data: torch.Tensor, precision: Optional[torch.dtype]=None) -> torch.Tensor:
    """
    Converts a Nx3x3 transformation tensor into a Nx3 tensor of [x, y, heading] rows.
    :param input_data: The Nx3x3 transformation matrix.
    :param precision: The precision with which to create the output tensor. If None, then it will be inferred from the input tensor.
    :return: The converted tensor.
    """
    _validate_transform_matrix_shape(input_data)
    if precision is None:
        precision = input_data.dtype
    return torch.tensor([float(input_data[0, 2].item()), float(input_data[1, 2].item()), float(math.atan2(float(input_data[1, 0].item()), float(input_data[0, 0].item())))], dtype=precision)

def _validate_transform_matrix_shape(tensor: torch.Tensor) -> None:
    """
    Validates that a tensor has the proper shape for a 3x3 transform matrix.
    :param tensor: the tensor to validate.
    """
    if len(tensor.shape) == 2 and tensor.shape[0] == 3 and (tensor.shape[1] == 3):
        return
    raise ValueError(f'Improper transform matrix shape: {tensor.shape}')

def transform_matrix_to_state_se2_tensor_batch(input_data: torch.Tensor) -> torch.Tensor:
    """
    Converts a Nx3x3 batch transformation matrix into a Nx3 tensor of [x, y, heading] rows.
    :param input_data: The 3x3 transformation matrix.
    :return: The converted tensor.
    """
    _validate_transform_matrix_batch_shape(input_data)
    first_columns = input_data[:, :, 0].reshape(-1, 3)
    angles = torch.atan2(first_columns[:, 1], first_columns[:, 0])
    result = input_data[:, :, 2]
    result[:, 2] = angles
    return result

def _validate_transform_matrix_batch_shape(tensor: torch.Tensor) -> None:
    """
    Validates that a tensor has the proper shape for a 3x3 transform matrix.
    :param tensor: the tensor to validate.
    """
    if len(tensor.shape) == 3 and tensor.shape[1] == 3 and (tensor.shape[2] == 3):
        return
    raise ValueError(f'Improper transform matrix shape: {tensor.shape}')

def global_state_se2_tensor_to_local(global_states: torch.Tensor, local_state: torch.Tensor, precision: Optional[torch.dtype]=None) -> torch.Tensor:
    """
    Transforms the StateSE2 in tensor from to the frame of reference in local_frame.

    :param global_states: A tensor of Nx3, where the columns are [x, y, heading].
    :param local_state: A tensor of [x, y, h] of the frame to which to transform.
    :param precision: The precision with which to allocate the intermediate tensors. If None, then it will be inferred from the input precisions.
    :return: The transformed coordinates.
    """
    _validate_state_se2_tensor_shape(global_states, expected_first_dim=2)
    _validate_state_se2_tensor_shape(local_state, expected_first_dim=1)
    if precision is None:
        if global_states.dtype != local_state.dtype:
            raise ValueError('Mixed datatypes provided to coordinates_to_local_frame without precision specifier.')
        precision = global_states.dtype
    local_xform = state_se2_tensor_to_transform_matrix(local_state, precision=precision)
    local_xform_inv = torch.linalg.inv(local_xform)
    transforms = state_se2_tensor_to_transform_matrix_batch(global_states, precision=precision)
    transforms = torch.matmul(local_xform_inv, transforms)
    output = transform_matrix_to_state_se2_tensor_batch(transforms)
    return output

def coordinates_to_local_frame(coords: torch.Tensor, anchor_state: torch.Tensor, precision: Optional[torch.dtype]=None) -> torch.Tensor:
    """
    Transform a set of [x, y] coordinates without heading to the the given frame.
    :param coords: <torch.Tensor: num_coords, 2> Coordinates to be transformed, in the form [x, y].
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param precision: The precision with which to allocate the intermediate tensors. If None, then it will be inferred from the input precisions.
    :return: <torch.Tensor: num_coords, 2> Transformed coordinates.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f'Unexpected coords shape: {coords.shape}')
    if precision is None:
        if coords.dtype != anchor_state.dtype:
            raise ValueError('Mixed datatypes provided to coordinates_to_local_frame without precision specifier.')
        precision = coords.dtype
    if coords.shape[0] == 0:
        return coords
    transform = state_se2_tensor_to_transform_matrix(anchor_state, precision=precision)
    transform = torch.linalg.inv(transform)
    coords = torch.nn.functional.pad(coords, (0, 1, 0, 0), 'constant', value=1.0)
    coords = torch.matmul(transform, coords.transpose(0, 1))
    result = coords.transpose(0, 1)
    result = result[:, :2]
    return result

def vector_set_coordinates_to_local_frame(coords: torch.Tensor, avails: torch.Tensor, anchor_state: torch.Tensor, output_precision: Optional[torch.dtype]=torch.float32) -> torch.Tensor:
    """
    Transform the vector set map element coordinates from global frame to ego vehicle frame, as specified by
        anchor_state.
    :param coords: Coordinates to transform. <torch.Tensor: num_elements, num_points, 2>.
    :param avails: Availabilities mask identifying real vs zero-padded data in coords.
        <torch.Tensor: num_elements, num_points>.
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param output_precision: The precision with which to allocate output tensors.
    :return: Transformed coordinates.
    :raise ValueError: If coordinates dimensions are not valid or don't match availabilities.
    """
    if len(coords.shape) != 3 or coords.shape[2] != 2:
        raise ValueError(f'Unexpected coords shape: {coords.shape}. Expected shape: (*, *, 2)')
    if coords.shape[:2] != avails.shape:
        raise ValueError(f'Mismatching shape between coords and availabilities: {coords.shape[:2]}, {avails.shape}')
    num_map_elements, num_points_per_element, _ = coords.size()
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)
    coords = coordinates_to_local_frame(coords.double(), anchor_state.double(), precision=torch.float64)
    coords = coords.reshape(num_map_elements, num_points_per_element, 2)
    coords = coords.to(output_precision)
    coords[~avails] = 0.0
    return coords

class TestTorchGeometry(unittest.TestCase):
    """
    A class for testing the functionality of the torch geometry library.
    """

    def test_transform_matrix_conversion_functionality(self) -> None:
        """
        Test the numerical accuracy of the transform matrix conversion utilities.
        """
        initial_state = torch.tensor([5, 6, math.pi / 2], dtype=torch.float32)
        expected_xform_matrix = torch.tensor([[0, -1, 5], [1, 0, 6], [0, 0, 1]], dtype=torch.float32)
        xform_matrix = state_se2_tensor_to_transform_matrix(initial_state, precision=torch.float32)
        torch.testing.assert_allclose(expected_xform_matrix, xform_matrix)
        reverted = transform_matrix_to_state_se2_tensor(xform_matrix, precision=torch.float32)
        torch.testing.assert_allclose(initial_state, reverted)

    def test_transform_matrix_scriptability(self) -> None:
        """
        Tests that the transform matrix conversion utilities script properly.
        """

        class tmp_module(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                xform = state_se2_tensor_to_transform_matrix(x)
                result = transform_matrix_to_state_se2_tensor(xform)
                return result
        to_script = tmp_module()
        scripted = torch.jit.script(to_script)
        test_input = torch.tensor([1, 2, 3], dtype=torch.float32)
        py_result = to_script.forward(test_input)
        script_result = scripted.forward(test_input)
        torch.testing.assert_allclose(py_result, script_result)

    def test_transform_matrix_batch_conversion_functionality(self) -> None:
        """
        Test the numerical accuracy of the transform matrix conversion utilities.
        """
        initial_state = torch.tensor([[5, 6, math.pi / 2]], dtype=torch.float32)
        expected_xform_matrix = torch.tensor([[[0, -1, 5], [1, 0, 6], [0, 0, 1]]], dtype=torch.float32)
        xform_matrix = state_se2_tensor_to_transform_matrix_batch(initial_state, precision=torch.float32)
        torch.testing.assert_allclose(expected_xform_matrix, xform_matrix)
        reverted = transform_matrix_to_state_se2_tensor_batch(xform_matrix)
        torch.testing.assert_allclose(initial_state, reverted)
        with self.assertRaises(ValueError):
            misshaped_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
            state_se2_tensor_to_transform_matrix_batch(misshaped_tensor)

    def test_transform_matrix_batch_scriptability(self) -> None:
        """
        Tests that the transform matrix conversion utilities script properly.
        """

        class tmp_module(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                xform = state_se2_tensor_to_transform_matrix_batch(x)
                result = transform_matrix_to_state_se2_tensor_batch(xform)
                return result
        to_script = tmp_module()
        scripted = torch.jit.script(to_script)
        test_input = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        py_result = to_script.forward(test_input)
        script_result = scripted.forward(test_input)
        torch.testing.assert_allclose(py_result, script_result)

    def test_global_state_se2_tensor_to_local_functionality(self) -> None:
        """
        Tests the numerical accuracy of global_state_se2_tensor_to_local.
        """
        '\n           o = coordinates to transform, facing direction >\n           # = local reference frame\n           y_world\n            ^              x_local\n            |               ^\n            |               |\n            |   y_local <---#\n            |\n            |\n            |\n         o  |  o>\n         V  |\n            |\n            *---------------------> x_world\n               ^\n        <o     o\n\n        '
        global_states = torch.tensor([[1, 1, 0], [1, -1, math.pi / 2], [-1, -1, math.pi], [-1, 1, -math.pi / 2]], dtype=torch.float32)
        local_state = torch.tensor([5, 5, math.pi / 2], dtype=torch.float32)
        expected_transformed_states = torch.tensor([[-4, 4, -math.pi / 2], [-6, 4, 0], [-6, 6, math.pi / 2], [-4, 6, math.pi]], dtype=torch.float32)
        actual_transformed_states = global_state_se2_tensor_to_local(global_states, local_state, precision=torch.float32)
        torch.testing.assert_allclose(expected_transformed_states, actual_transformed_states)

    def test_global_state_se2_tensor_to_local_scriptability(self) -> None:
        """
        Tests that global_state_se2_tensor_to_local scripts properly.
        """

        class tmp_module(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, states: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
                result = global_state_se2_tensor_to_local(states, pose)
                return result
        to_script = tmp_module()
        scripted = torch.jit.script(to_script)
        test_states = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        test_pose = torch.tensor([1, 2, 3], dtype=torch.float32)
        py_result = to_script.forward(test_states, test_pose)
        script_result = scripted.forward(test_states, test_pose)
        torch.testing.assert_allclose(py_result, script_result)

    def test_coordinates_to_local_frame_functionality(self) -> None:
        """
        Tests the numerical accuracy of coordinates_to_local_frame.
        """
        '\n           o = coordinates to transform\n           # = local reference frame\n           y_world\n            ^              x_local\n            |               ^\n            |               |\n            |   y_local <---#\n            |\n            |\n            |\n         o  |  o\n            |\n            |\n            *---------------------> x_world\n\n         o     o\n\n        '
        coordinates = torch.tensor([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=torch.float32)
        local_state = torch.tensor([5, 5, math.pi / 2], dtype=torch.float32)
        expected_coordinates = torch.tensor([[-4, 4], [-6, 4], [-6, 6], [-4, 6]], dtype=torch.float32)
        actual_coordinates = coordinates_to_local_frame(coordinates, local_state, precision=torch.float32)
        torch.testing.assert_allclose(expected_coordinates, actual_coordinates)

    def test_coordinates_to_local_frame_scriptability(self) -> None:
        """
        Tests that the function coordinates_to_local_frame scripts properly.
        """

        class tmp_module(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, states: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
                result = coordinates_to_local_frame(states, pose, precision=torch.float32)
                return result
        to_script = tmp_module()
        scripted = torch.jit.script(to_script)
        test_states = torch.tensor([[1, 2], [4, 5]], dtype=torch.float32)
        test_pose = torch.tensor([1, 2, 3], dtype=torch.float32)
        py_result = to_script.forward(test_states, test_pose)
        script_result = scripted.forward(test_states, test_pose)
        torch.testing.assert_allclose(py_result, script_result)

    def test_vector_set_coordinates_to_local_frame_functionality(self) -> None:
        """
        Test converting vector set map coordinates from global to local ego frame.
        """
        coords = torch.tensor([[[1, 1], [3, 1], [5, 1]]], dtype=torch.float64)
        avails = torch.ones(coords.shape[:2], dtype=torch.bool)
        anchor_state = torch.tensor([0, 0, 0], dtype=torch.float64)
        result_coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)
        self.assertIsInstance(result_coords, torch.FloatTensor)
        self.assertEqual(result_coords.shape, coords.shape)
        torch.testing.assert_allclose(coords.float(), result_coords)
        with self.assertRaises(ValueError):
            vector_set_coordinates_to_local_frame(coords[0], avails[0], anchor_state)
        with self.assertRaises(ValueError):
            vector_set_coordinates_to_local_frame(coords, avails[0], anchor_state)

    def test_vector_set_coordinates_to_local_frame_scriptability(self) -> None:
        """
        Tests that the function vector_set_coordinates_to_local_frame scripts properly.
        """

        class tmp_module(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, coords: torch.Tensor, avails: torch.Tensor, anchor_state: torch.Tensor) -> torch.Tensor:
                result = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)
                return result
        to_script = tmp_module()
        scripted = torch.jit.script(to_script)
        test_coords = torch.tensor([[[1, 1], [3, 1], [5, 1]]], dtype=torch.float64)
        test_avails = torch.ones(test_coords.shape[:2], dtype=torch.bool)
        test_anchor_state = torch.tensor([0, 0, 0], dtype=torch.float64)
        py_result = to_script.forward(test_coords, test_avails, test_anchor_state)
        script_result = scripted.forward(test_coords, test_avails, test_anchor_state)
        torch.testing.assert_allclose(py_result, script_result)

class tmp_module(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, coords: torch.Tensor, avails: torch.Tensor, anchor_state: torch.Tensor) -> torch.Tensor:
        result = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)
        return result

class TestKinematicBicycleLayersUtils(unittest.TestCase):
    """
    Test Kinematic Bicycle Layers utils.
    """

    def setUp(self) -> None:
        """Sets variables for testing"""
        pass

    def test_enums_equal(self) -> None:
        """
        Ensure our internal indexing matches the
        one from Agents' trajectories
        """
        self.assertEqual(StateIndex.X_POS, AgentFeatureIndex.x())
        self.assertEqual(StateIndex.Y_POS, AgentFeatureIndex.y())
        self.assertEqual(StateIndex.YAW, AgentFeatureIndex.heading())
        self.assertEqual(StateIndex.X_VELOCITY, AgentFeatureIndex.vx())
        self.assertEqual(StateIndex.Y_VELOCITY, AgentFeatureIndex.vy())
        self.assertEqual(StateIndex.YAW_RATE, AgentFeatureIndex.yaw_rate())

def _validate_ego_feature_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be an EgoFeature.
    :param feature: The tensor to validate.
    """
    if len(feature.shape) == 2 and feature.shape[1] == EgoFeatureIndex.dim():
        return
    if len(feature.shape) == 1 and feature.shape[0] == EgoFeatureIndex.dim():
        return
    raise ValueError(f'Improper ego feature shape: {feature.shape}.')

def _validate_agent_feature_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be an AgentFeature.
    :param feature: The tensor to validate.
    """
    if len(feature.shape) != 3 or feature.shape[2] != AgentFeatureIndex.dim():
        raise ValueError(f'Improper agent feature shape: {feature.shape}.')

def convert_absolute_quantities_to_relative(agent_states: List[torch.Tensor], ego_state: torch.Tensor) -> List[torch.Tensor]:
    """
    Converts the agents' poses and relative velocities from absolute to ego-relative coordinates.
    :param agent_states: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """
    _validate_ego_internal_shape(ego_state, expected_first_dim=1)
    ego_pose = torch.tensor([float(ego_state[EgoInternalIndex.x()].item()), float(ego_state[EgoInternalIndex.y()].item()), float(ego_state[EgoInternalIndex.heading()].item())], dtype=torch.float64)
    ego_velocity = torch.tensor([float(ego_state[EgoInternalIndex.vx()].item()), float(ego_state[EgoInternalIndex.vy()].item()), float(ego_state[EgoInternalIndex.heading()].item())], dtype=torch.float64)
    for agent_state in agent_states:
        _validate_agent_internal_shape(agent_state)
        agent_global_poses = agent_state[:, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]].double()
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy(), AgentInternalIndex.heading()]].double()
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        transformed_velocities = global_state_se2_tensor_to_local(agent_global_velocities, ego_velocity, precision=torch.float64)
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()
    return agent_states

def _validate_ego_internal_shape(feature: torch.Tensor, expected_first_dim: Optional[int]=None) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be an EgoInternal.
    :param feature: The tensor to validate.
    :param expected_first_dim: If None, accept either [N, EgoInternalIndex.dim()] or [EgoInternalIndex.dim()]
                                If 1, only accept [EgoInternalIndex.dim()]
                                If 2, only accept [N, EgoInternalIndex.dim()]
    """
    if len(feature.shape) == 2 and feature.shape[1] == EgoInternalIndex.dim():
        if expected_first_dim is None or expected_first_dim == 2:
            return
    if len(feature.shape) == 1 and feature.shape[0] == EgoInternalIndex.dim():
        if expected_first_dim is None or expected_first_dim == 1:
            return
    raise ValueError(f'Improper ego internal shape: {feature.shape}')

def _validate_agent_internal_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be an AgentInternal.
    :param feature: the tensor to validate.
    """
    if len(feature.shape) != 2 or feature.shape[1] != AgentInternalIndex.dim():
        raise ValueError(f'Improper agent internal shape: {feature.shape}')

def pad_agent_states(agent_trajectories: List[torch.Tensor], reverse: bool) -> List[torch.Tensor]:
    """
    Pads the agent states with the most recent available states. The order of the agents is also
    preserved. Note: only agents that appear in the current time step will be computed for. Agents appearing in the
    future or past will be discarded.

     t1      t2           t1      t2
    |a1,t1| |a1,t2|  pad |a1,t1| |a1,t2|
    |a2,t1| |a3,t2|  ->  |a2,t1| |a2,t1| (padded with agent 2 state at t1)
    |a3,t1| |     |      |a3,t1| |a3,t2|


    If reverse is True, the padding direction will start from the end of the trajectory towards the start

     tN-1    tN             tN-1    tN
    |a1,tN-1| |a1,tN|  pad |a1,tN-1| |a1,tN|
    |a2,tN  | |a2,tN|  <-  |a3,tN-1| |a2,tN| (padded with agent 2 state at tN)
    |a3,tN-1| |a3,tN|      |       | |a3,tN|

    :param agent_trajectories: agent trajectories [num_frames, num_agents, AgentInternalIndex.dim()], corresponding to the AgentInternalIndex schema.
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states
    """
    for traj in agent_trajectories:
        _validate_agent_internal_shape(traj)
    track_id_idx = AgentInternalIndex.track_token()
    if reverse:
        agent_trajectories = agent_trajectories[::-1]
    key_frame = agent_trajectories[0]
    id_row_mapping: Dict[int, int] = {}
    for idx, val in enumerate(key_frame[:, track_id_idx]):
        id_row_mapping[int(val.item())] = idx
    current_state = torch.zeros((key_frame.shape[0], key_frame.shape[1]), dtype=torch.float32)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        for row_idx in range(frame.shape[0]):
            mapped_row: int = id_row_mapping[int(frame[row_idx, track_id_idx].item())]
            current_state[mapped_row, :] = frame[row_idx, :]
        agent_trajectories[idx] = torch.clone(current_state)
    if reverse:
        agent_trajectories = agent_trajectories[::-1]
    return agent_trajectories

def build_ego_features_from_tensor(ego_trajectory: torch.Tensor, reverse: bool=False) -> torch.Tensor:
    """
    Build agent features from the ego states
    :param ego_trajectory: ego states at past times. Tensors complying with the EgoInternalIndex schema.
    :param reverse: if True, the last element in the list will be considered as the present ego state
    :return: Tensor complying with the EgoFeatureIndex schema.
    """
    _validate_ego_internal_shape(ego_trajectory, expected_first_dim=2)
    if reverse:
        anchor_ego_state = ego_trajectory[ego_trajectory.shape[0] - 1, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]].squeeze().double()
    else:
        anchor_ego_state = ego_trajectory[0, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]].squeeze().double()
    global_ego_trajectory = ego_trajectory[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]].double()
    local_ego_trajectory = global_state_se2_tensor_to_local(global_ego_trajectory, anchor_ego_state, precision=torch.float64)
    return local_ego_trajectory.float()

def build_generic_ego_features_from_tensor(ego_trajectory: torch.Tensor, reverse: bool=False) -> torch.Tensor:
    """
    Build generic agent features from the ego states
    :param ego_trajectory: ego states at past times. Tensors complying with the EgoInternalIndex schema.
    :param reverse: if True, the last element in the list will be considered as the present ego state
    :return: Tensor complying with the GenericEgoFeatureIndex schema.
    """
    _validate_ego_internal_shape(ego_trajectory, expected_first_dim=2)
    if reverse:
        anchor_ego_pose = ego_trajectory[ego_trajectory.shape[0] - 1, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]].squeeze().double()
        anchor_ego_velocity = ego_trajectory[ego_trajectory.shape[0] - 1, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.heading()]].squeeze().double()
        anchor_ego_acceleration = ego_trajectory[ego_trajectory.shape[0] - 1, [EgoInternalIndex.ax(), EgoInternalIndex.ay(), EgoInternalIndex.heading()]].squeeze().double()
    else:
        anchor_ego_pose = ego_trajectory[0, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]].squeeze().double()
        anchor_ego_velocity = ego_trajectory[0, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.heading()]].squeeze().double()
        anchor_ego_acceleration = ego_trajectory[0, [EgoInternalIndex.ax(), EgoInternalIndex.ay(), EgoInternalIndex.heading()]].squeeze().double()
    global_ego_poses = ego_trajectory[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]].double()
    global_ego_velocities = ego_trajectory[:, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.heading()]].double()
    global_ego_accelerations = ego_trajectory[:, [EgoInternalIndex.ax(), EgoInternalIndex.ay(), EgoInternalIndex.heading()]].double()
    local_ego_poses = global_state_se2_tensor_to_local(global_ego_poses, anchor_ego_pose, precision=torch.float64)
    local_ego_velocities = global_state_se2_tensor_to_local(global_ego_velocities, anchor_ego_velocity, precision=torch.float64)
    local_ego_accelerations = global_state_se2_tensor_to_local(global_ego_accelerations, anchor_ego_acceleration, precision=torch.float64)
    local_ego_trajectory: torch.Tensor = torch.empty(ego_trajectory.size(), dtype=torch.float32, device=ego_trajectory.device)
    local_ego_trajectory[:, EgoInternalIndex.x()] = local_ego_poses[:, 0].float()
    local_ego_trajectory[:, EgoInternalIndex.y()] = local_ego_poses[:, 1].float()
    local_ego_trajectory[:, EgoInternalIndex.heading()] = local_ego_poses[:, 2].float()
    local_ego_trajectory[:, EgoInternalIndex.vx()] = local_ego_velocities[:, 0].float()
    local_ego_trajectory[:, EgoInternalIndex.vy()] = local_ego_velocities[:, 1].float()
    local_ego_trajectory[:, EgoInternalIndex.ax()] = local_ego_accelerations[:, 0].float()
    local_ego_trajectory[:, EgoInternalIndex.ay()] = local_ego_accelerations[:, 1].float()
    return local_ego_trajectory

def filter_agents_tensor(agents: List[torch.Tensor], reverse: bool=False) -> List[torch.Tensor]:
    """
    Filter detections to keep only agents which appear in the first frame (or last frame if reverse=True)
    :param agents: The past agents in the scene. A list of [num_frames] tensors, each complying with the AgentInternalIndex schema
    :param reverse: if True, the last element in the list will be used as the filter
    :return: filtered agents in the same format as the input `agents` parameter
    """
    target_tensor = agents[-1] if reverse else agents[0]
    for i in range(len(agents)):
        _validate_agent_internal_shape(agents[i])
        rows: List[torch.Tensor] = []
        for j in range(agents[i].shape[0]):
            if target_tensor.shape[0] > 0:
                agent_id: float = float(agents[i][j, int(AgentInternalIndex.track_token())].item())
                is_in_target_frame: bool = bool((agent_id == target_tensor[:, AgentInternalIndex.track_token()]).max().item())
                if is_in_target_frame:
                    rows.append(agents[i][j, :].squeeze())
        if len(rows) > 0:
            agents[i] = torch.stack(rows)
        else:
            agents[i] = torch.empty((0, agents[i].shape[1]), dtype=torch.float32)
    return agents

def compute_yaw_rate_from_state_tensors(agent_states: List[torch.Tensor], time_stamps: torch.Tensor) -> torch.Tensor:
    """
    Computes the yaw rate of all agents over the trajectory from heading
    :param agent_states_horizon: Agent trajectories [num_frames, num_agent, AgentsInternalBuffer.dim()]
    :param time_stamps: The time stamps of each frame.
    :return: <torch.Tensor: num_frames, num_agents> of yaw rates
    """
    if len(time_stamps.shape) != 1:
        raise ValueError(f'Unexpected timestamps shape: {time_stamps.shape}')
    time_stamps_s = (time_stamps - int(torch.min(time_stamps).item())).double() * 1e-06
    yaws: List[torch.Tensor] = []
    for agent_state in agent_states:
        _validate_agent_internal_shape(agent_state)
        yaws.append(agent_state[:, AgentInternalIndex.heading()].squeeze().double())
    yaws_tensor = torch.vstack(yaws)
    yaws_tensor = yaws_tensor.transpose(0, 1)
    yaws_tensor = unwrap(yaws_tensor, dim=-1)
    yaw_rate_horizon = approximate_derivatives_tensor(yaws_tensor, time_stamps_s, window_length=3)
    return yaw_rate_horizon.transpose(0, 1)

def pack_agents_tensor(padded_agents_tensors: List[torch.Tensor], yaw_rates: torch.Tensor) -> torch.Tensor:
    """
    Combines the local padded agents states and the computed yaw rates into the final output feature tensor.
    :param padded_agents_tensors: The padded agent states for each timestamp.
        Each tensor is of shape <num_agents, len(AgentInternalIndex)> and conforms to the AgentInternalIndex schema.
    :param yaw_rates: The computed yaw rates. The tensor is of shape <num_timestamps, agent>
    :return: The final feature, a tensor of shape [timestamp, num_agents, len(AgentsFeatureIndex)] conforming to the AgentFeatureIndex Schema
    """
    if yaw_rates.shape != (len(padded_agents_tensors), padded_agents_tensors[0].shape[0]):
        raise ValueError(f'Unexpected yaw_rates tensor shape: {yaw_rates.shape}')
    agents_tensor = torch.zeros((len(padded_agents_tensors), padded_agents_tensors[0].shape[0], AgentFeatureIndex.dim()))
    for i in range(len(padded_agents_tensors)):
        _validate_agent_internal_shape(padded_agents_tensors[i])
        agents_tensor[i, :, AgentFeatureIndex.x()] = padded_agents_tensors[i][:, AgentInternalIndex.x()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.y()] = padded_agents_tensors[i][:, AgentInternalIndex.y()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.heading()] = padded_agents_tensors[i][:, AgentInternalIndex.heading()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.vx()] = padded_agents_tensors[i][:, AgentInternalIndex.vx()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.vy()] = padded_agents_tensors[i][:, AgentInternalIndex.vy()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.yaw_rate()] = yaw_rates[i, :].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.width()] = padded_agents_tensors[i][:, AgentInternalIndex.width()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.length()] = padded_agents_tensors[i][:, AgentInternalIndex.length()].squeeze()
    return agents_tensor

def _create_ego_trajectory_tensor(num_frames: int) -> torch.Tensor:
    """
    Generate a dummy ego trajectory
    :param num_frames: length of the trajectory to be generate
    :return: The generated trajectory
    """
    output = torch.ones((num_frames, EgoInternalIndex.dim()))
    for i in range(num_frames):
        output[i, :] *= i
    return output

def _create_tracked_object_agent_tensor(num_agents: int) -> torch.Tensor:
    """
    Generates a dummy tracked object input tensor.
    :param num_agents: The number of agents in the tensor.
    :return: The generated tensor.
    """
    output = torch.ones((num_agents, AgentInternalIndex.dim()))
    for i in range(num_agents):
        output[i, :] *= i
    return output

class TestAgentsFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs features with vectorized agent information."""

    def setUp(self) -> None:
        """Set up test case."""
        self.num_frames = 8
        self.num_agents = 10
        self.num_missing_agents = 2
        self.agent_trajectories = [*_create_tracked_objects(5, self.num_agents), *_create_tracked_objects(3, self.num_agents - self.num_missing_agents)]
        self.time_stamps = [TimePoint(step) for step in range(self.num_frames)]

    def test_build_ego_features(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        ego_trajectory = _create_ego_trajectory(num_frames)
        ego_features = build_ego_features(ego_trajectory)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(np.allclose(ego_features[0], np.array([0, 0, 0])))
        ego_features_reversed = build_ego_features(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(np.allclose(ego_features_reversed[-1], np.array([0, 0, 0])))

    def test_extract_and_pad_agent_poses(self) -> None:
        """
        Test when there is agent pose trajectory is incomplete
        """
        padded_poses, availability = extract_and_pad_agent_poses(self.agent_trajectories)
        availability = np.asarray(availability)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses])
        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)
        self.assertEqual(len(availability.shape), 2)
        self.assertEqual(availability.shape[0], self.num_frames)
        self.assertEqual(availability.shape[1], self.num_agents)
        self.assertTrue(availability[:5, :].all())
        self.assertTrue(availability[:, :self.num_agents - self.num_missing_agents].all())
        self.assertTrue((~availability[5:, -self.num_missing_agents:]).all())
        padded_poses_reversed, availability_reversed = extract_and_pad_agent_poses(self.agent_trajectories[::-1], reverse=True)
        availability_reversed = np.asarray(availability_reversed)
        stacked_poses = np.stack([[agent.serialize() for agent in frame] for frame in padded_poses_reversed])
        self.assertEqual(stacked_poses.shape[0], self.num_frames)
        self.assertEqual(stacked_poses.shape[1], self.num_agents)
        self.assertEqual(stacked_poses.shape[2], 3)
        self.assertEqual(len(availability_reversed.shape), 2)
        self.assertEqual(availability_reversed.shape[0], self.num_frames)
        self.assertEqual(availability_reversed.shape[1], self.num_agents)
        self.assertTrue(availability_reversed[-5:, :].all())
        self.assertTrue(availability_reversed[:, :self.num_agents - self.num_missing_agents].all())
        self.assertTrue((~availability_reversed[:3, -self.num_missing_agents:]).all())

    def test_extract_and_pad_agent_sizes(self) -> None:
        """
        Test when there is agent size trajectory is incomplete
        """
        padded_sizes, _ = extract_and_pad_agent_sizes(self.agent_trajectories)
        stacked_sizes = np.stack(padded_sizes)
        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)
        padded_sizes_reversed, _ = extract_and_pad_agent_sizes(self.agent_trajectories[::-1], reverse=True)
        stacked_sizes = np.stack(padded_sizes_reversed)
        self.assertEqual(stacked_sizes.shape[0], self.num_frames)
        self.assertEqual(stacked_sizes.shape[1], self.num_agents)
        self.assertEqual(stacked_sizes.shape[2], 2)

    def test_extract_and_pad_agent_velocities(self) -> None:
        """
        Test when there is agent velocity trajectory is incomplete
        """
        padded_velocities, _ = extract_and_pad_agent_velocities(self.agent_trajectories)
        stacked_velocities = np.stack([[agent.serialize() for agent in frame] for frame in padded_velocities])
        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)
        padded_velocities_reversed, _ = extract_and_pad_agent_velocities(self.agent_trajectories[::-1], reverse=True)
        stacked_velocities = np.stack([[agent.serialize() for agent in frame] for frame in padded_velocities_reversed])
        self.assertEqual(stacked_velocities.shape[0], self.num_frames)
        self.assertEqual(stacked_velocities.shape[1], self.num_agents)
        self.assertEqual(stacked_velocities.shape[2], 3)

    def test_compute_yaw_rate_from_states(self) -> None:
        """
        Test computing yaw from the agent pose trajectory
        """
        padded_poses, _ = extract_and_pad_agent_poses(self.agent_trajectories)
        yaw_rates = compute_yaw_rate_from_states(padded_poses, self.time_stamps)
        self.assertEqual(yaw_rates.transpose().shape[0], self.num_frames)
        self.assertEqual(yaw_rates.transpose().shape[1], self.num_agents)

    def test_filter_agents(self) -> None:
        """
        Test agent filtering
        """
        num_frames = 8
        num_agents = 5
        missing_agents = 2
        tracked_objects_history = [*_create_tracked_objects(num_frames=5, num_agents=num_agents, object_type=TrackedObjectType.VEHICLE), *_create_tracked_objects(num_frames=2, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.BICYCLE), *_create_tracked_objects(num_frames=1, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.VEHICLE)]
        filtered_agents = filter_agents(tracked_objects_history)
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[0].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)
        filtered_agents = filter_agents(tracked_objects_history, reverse=True)
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[-1].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)
        tracked_objects_history = [*_create_tracked_objects(num_frames=5, num_agents=num_agents, object_type=TrackedObjectType.BICYCLE), *_create_tracked_objects(num_frames=2, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.VEHICLE), *_create_tracked_objects(num_frames=1, num_agents=num_agents - missing_agents, object_type=TrackedObjectType.BICYCLE)]
        filtered_agents = filter_agents(tracked_objects_history, allowable_types=[TrackedObjectType.BICYCLE])
        self.assertEqual(len(filtered_agents), num_frames)
        self.assertEqual(len(filtered_agents[0].tracked_objects), len(tracked_objects_history[0].tracked_objects))
        self.assertEqual(len(filtered_agents[5].tracked_objects), 0)
        self.assertEqual(len(filtered_agents[7].tracked_objects), num_agents - missing_agents)

    def test_build_ego_features_from_tensor(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        zeros = torch.tensor([0, 0, 0], dtype=torch.float32)
        ego_trajectory = _create_ego_trajectory_tensor(num_frames)
        ego_features = build_ego_features_from_tensor(ego_trajectory)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(torch.allclose(ego_features[0], zeros, atol=1e-07))
        ego_features_reversed = build_ego_features_from_tensor(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, EgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(torch.allclose(ego_features_reversed[-1], zeros, atol=1e-07))

    def test_build_generic_ego_features_from_tensor(self) -> None:
        """
        Test the ego feature building
        """
        num_frames = 5
        zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        ego_trajectory = _create_ego_trajectory_tensor(num_frames)
        ego_features = build_generic_ego_features_from_tensor(ego_trajectory)
        self.assertEqual((num_frames, GenericEgoFeatureIndex.dim()), ego_features.shape)
        self.assertTrue(torch.allclose(ego_features[0], zeros, atol=1e-07))
        ego_features_reversed = build_generic_ego_features_from_tensor(ego_trajectory, reverse=True)
        self.assertEqual((num_frames, GenericEgoFeatureIndex.dim()), ego_features_reversed.shape)
        self.assertTrue(torch.allclose(ego_features_reversed[-1], zeros, atol=1e-07))

    def test_convert_absolute_quantities_to_relative(self) -> None:
        """
        Test the conversion routine between absolute and relative quantities
        """

        def get_dummy_states() -> List[torch.Tensor]:
            """
            Create a series of dummy agent tensors
            """
            dummy_agent_state = _create_tracked_object_agent_tensor(7)
            dummy_states = [dummy_agent_state + i for i in range(5)]
            return dummy_states
        zeros = torch.tensor([0, 0, 0], dtype=torch.float32)
        dummy_states = get_dummy_states()
        ego_pose = torch.tensor([4, 4, 4, 2, 2, 2, 2], dtype=torch.float32)
        transformed = convert_absolute_quantities_to_relative(dummy_states, ego_pose)
        for i in range(0, len(transformed), 1):
            should_be_zero_row = 4 - i
            check_tensor = torch.tensor([transformed[i][should_be_zero_row, AgentInternalIndex.x()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.y()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.heading()].item()], dtype=torch.float32)
            self.assertTrue(torch.allclose(check_tensor, zeros, atol=1e-07))
        dummy_states = get_dummy_states()
        ego_pose = torch.tensor([2, 2, 4, 4, 4, 4, 4], dtype=torch.float32)
        transformed = convert_absolute_quantities_to_relative(dummy_states, ego_pose)
        for i in range(0, len(transformed), 1):
            should_be_zero_row = 4 - i
            check_tensor = torch.tensor([transformed[i][should_be_zero_row, AgentInternalIndex.vx()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.vy()].item(), transformed[i][should_be_zero_row, AgentInternalIndex.heading()].item()], dtype=torch.float32)
            self.assertTrue(torch.allclose(check_tensor, zeros, atol=1e-07))

    def test_pad_agent_states(self) -> None:
        """
        Test the pad agent states functionality
        """
        forward_dummy_states = [_create_tracked_object_agent_tensor(7), _create_tracked_object_agent_tensor(5), _create_tracked_object_agent_tensor(6)]
        padded = pad_agent_states(forward_dummy_states, reverse=False)
        self.assertTrue(len(padded) == 3)
        self.assertEqual((7, AgentInternalIndex.dim()), padded[0].shape)
        for i in range(1, len(padded)):
            self.assertTrue(torch.allclose(padded[0], padded[i]))
        backward_dummy_states = [_create_tracked_object_agent_tensor(6), _create_tracked_object_agent_tensor(5), _create_tracked_object_agent_tensor(7)]
        padded_reverse = pad_agent_states(backward_dummy_states, reverse=True)
        self.assertTrue(len(padded_reverse) == 3)
        self.assertEqual((7, AgentInternalIndex.dim()), padded_reverse[2].shape)
        for i in range(0, len(padded_reverse) - 1):
            self.assertTrue(torch.allclose(padded_reverse[2], padded_reverse[i]))

    def test_compute_yaw_rate_from_state_tensors(self) -> None:
        """
        Test compute yaw rate functionality
        """
        num_frames = 6
        num_agents = 5
        agent_states = [_create_tracked_object_agent_tensor(num_agents) + i for i in range(num_frames)]
        time_stamps = torch.tensor([int(i * 1000000.0) for i in range(num_frames)], dtype=torch.int64)
        yaw_rate = compute_yaw_rate_from_state_tensors(agent_states, time_stamps)
        self.assertEqual((num_frames, num_agents), yaw_rate.shape)
        self.assertTrue(torch.allclose(torch.ones((num_frames, num_agents), dtype=torch.float64), yaw_rate))

    def test_filter_agents_tensor(self) -> None:
        """
        Test filter agents
        """
        dummy_states = [_create_tracked_object_agent_tensor(7), _create_tracked_object_agent_tensor(8), _create_tracked_object_agent_tensor(6)]
        filtered = filter_agents_tensor(dummy_states, reverse=False)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered[0].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered[1].shape)
        self.assertEqual((6, AgentInternalIndex.dim()), filtered[2].shape)
        dummy_states = [_create_tracked_object_agent_tensor(6), _create_tracked_object_agent_tensor(8), _create_tracked_object_agent_tensor(7)]
        filtered_reverse = filter_agents_tensor(dummy_states, reverse=True)
        self.assertEqual((6, AgentInternalIndex.dim()), filtered_reverse[0].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered_reverse[1].shape)
        self.assertEqual((7, AgentInternalIndex.dim()), filtered_reverse[2].shape)

    def test_sampled_past_ego_states_to_tensor(self) -> None:
        """
        Test the conversion routine to convert ego states to tensors.
        """
        num_egos = 6
        test_egos = []
        for i in range(num_egos):
            footprint = CarFootprint(center=StateSE2(x=i, y=i, heading=i), vehicle_parameters=VehicleParameters(vehicle_name='vehicle_name', vehicle_type='vehicle_type', width=i, front_length=i, rear_length=i, cog_position_from_rear_axle=i, wheel_base=i, height=i))
            dynamic_car_state = DynamicCarState(rear_axle_to_center_dist=i, rear_axle_velocity_2d=StateVector2D(x=i + 5, y=i + 5), rear_axle_acceleration_2d=StateVector2D(x=i, y=i), angular_velocity=i, angular_acceleration=i, tire_steering_rate=i)
            test_ego = EgoState(car_footprint=footprint, dynamic_car_state=dynamic_car_state, tire_steering_angle=i, is_in_auto_mode=i, time_point=TimePoint(time_us=i))
            test_egos.append(test_ego)
        tensor = sampled_past_ego_states_to_tensor(test_egos)
        self.assertEqual((6, EgoInternalIndex.dim()), tensor.shape)
        for i in range(0, tensor.shape[0], 1):
            ego = test_egos[i]
            self.assertEqual(ego.rear_axle.x, tensor[i, EgoInternalIndex.x()].item())
            self.assertEqual(ego.rear_axle.y, tensor[i, EgoInternalIndex.y()].item())
            self.assertEqual(ego.rear_axle.heading, tensor[i, EgoInternalIndex.heading()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_velocity_2d.x, tensor[i, EgoInternalIndex.vx()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_velocity_2d.y, tensor[i, EgoInternalIndex.vy()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_acceleration_2d.x, tensor[i, EgoInternalIndex.ax()].item())
            self.assertEqual(ego.dynamic_car_state.rear_axle_acceleration_2d.y, tensor[i, EgoInternalIndex.ay()].item())

    def test_sampled_past_timestamps_to_tensor(self) -> None:
        """
        Test the conversion routine to convert timestamps to tensors.
        """
        points = [TimePoint(time_us=i) for i in range(10)]
        tensor = sampled_past_timestamps_to_tensor(points)
        self.assertEqual((10,), tensor.shape)
        for i in range(tensor.shape[0]):
            self.assertEqual(i, int(tensor[i].item()))

    def test_tracked_objects_to_tensor_list(self) -> None:
        """
        Test the conversion routine to convert tracked objects to tensors.
        """
        num_frames = 5
        test_tracked_objects = _create_dummy_tracked_objects_tensor(num_frames)
        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)
            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))
        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects, object_type=TrackedObjectType.BICYCLE)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)
            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))
        tensors = sampled_tracked_objects_to_tensor_list(test_tracked_objects, object_type=TrackedObjectType.PEDESTRIAN)
        self.assertEqual(num_frames, len(tensors))
        for idx, generated_tensor in enumerate(tensors):
            expected_num_agents = idx + 1
            self.assertEqual((expected_num_agents, AgentInternalIndex.dim()), generated_tensor.shape)
            for row in range(generated_tensor.shape[0]):
                for col in range(generated_tensor.shape[1]):
                    self.assertEqual(row + col, int(generated_tensor[row, col].item()))

    def test_pack_agents_tensor(self) -> None:
        """
        Test the routine used to convert local buffers into the final feature.
        """
        num_agents = 4
        num_timestamps = 3
        agents_tensors = [_create_tracked_object_agent_tensor(num_agents) for _ in range(num_timestamps)]
        yaw_rates = torch.ones((num_timestamps, num_agents)) * 100
        packed = pack_agents_tensor(agents_tensors, yaw_rates)
        self.assertEqual((num_timestamps, num_agents, AgentFeatureIndex.dim()), packed.shape)
        for ts in range(num_timestamps):
            for agent in range(num_agents):
                for col in range(AgentFeatureIndex.dim()):
                    if col == AgentFeatureIndex.yaw_rate():
                        self.assertEqual(100, packed[ts, agent, col])
                    else:
                        self.assertEqual(agent, packed[ts, agent, col])

class GenericAgentsFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, agent_features: List[str], trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__()
        self.agent_features = agent_features
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon
        self._agents_states_dim = GenericAgents.agents_states_dim()
        if 'EGO' in self.agent_features:
            raise AssertionError('EGO not valid agents feature type!')
        for feature_name in self.agent_features:
            if feature_name not in TrackedObjectType._member_names_:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'generic_agents'

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return GenericAgents

    @torch.jit.unused
    def get_scriptable_input_from_scenario(self, scenario: AbstractScenario) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        anchor_ego_state = scenario.initial_ego_state
        past_ego_states = scenario.get_ego_past_trajectory(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)
        sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
        time_stamps = list(scenario.get_past_timestamps(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)) + [scenario.start_time]
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [tracked_objects.tracked_objects for tracked_objects in scenario.get_past_tracked_objects(iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses)]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        assert len(sampled_past_ego_states) == len(sampled_past_observations), f'Expected the trajectory length of ego and agent to be equal. Got ego: {len(sampled_past_ego_states)} and agent: {len(sampled_past_observations)}'
        assert len(sampled_past_observations) > 2, f'Trajectory of length of {len(sampled_past_observations)} needs to be at least 3'
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_scriptable_input_from_simulation(self, current_input: PlannerInput) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the simulation input
        :param current_input: planner input from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        history = current_input.history
        assert isinstance(history.observations[0], DetectionsTracks), f'Expected observation of type DetectionTracks, got {type(history.observations[0])}'
        present_ego_state, present_observation = history.current_state
        past_observations = history.observations[:-1]
        past_ego_states = history.ego_states[:-1]
        assert history.sample_interval, 'SimulationHistoryBuffer sample interval is None'
        indices = sample_indices_with_time_horizon(self.num_past_poses, self.past_time_horizon, history.sample_interval)
        try:
            sampled_past_observations = [cast(DetectionsTracks, past_observations[-idx]).tracked_objects for idx in reversed(indices)]
            sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
        except IndexError:
            raise RuntimeError(f'SimulationHistoryBuffer duration: {history.duration} is too short for requested past_time_horizon: {self.past_time_horizon}. Please increase the simulation_buffer_duration in default_simulation.yaml')
        sampled_past_observations = sampled_past_observations + [cast(DetectionsTracks, present_observation).tracked_objects]
        sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]
        time_stamps = [state.time_point for state in sampled_past_ego_states]
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> GenericAgents:
        """Inherited, see superclass."""
        with torch.no_grad():
            tensors, list_tensors, list_list_tensors = self.get_scriptable_input_from_scenario(scenario)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: GenericAgents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> GenericAgents:
        """Inherited, see superclass."""
        with torch.no_grad():
            tensors, list_tensors, list_list_tensors = self.get_scriptable_input_from_simulation(current_input)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: GenericAgents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(self, past_ego_states: List[EgoState], past_time_stamps: List[TimePoint], past_tracked_objects: List[TrackedObjects]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param past_ego_states: The past states of the ego vehicle.
        :param past_time_stamps: The past time stamps of the input data.
        :param past_tracked_objects: The past tracked objects.
        :return: The packed tensors.
        """
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(past_ego_states)
        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)
        for feature_name in self.agent_features:
            past_tracked_objects_tensor_list = sampled_tracked_objects_to_tensor_list(past_tracked_objects, TrackedObjectType[feature_name])
            list_tensor_data[f'past_tracked_objects.{feature_name}'] = past_tracked_objects_tensor_list
        return ({'past_ego_states': past_ego_states_tensor, 'past_time_stamps': past_time_stamps_tensor}, list_tensor_data, {})

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> GenericAgents:
        """
        Unpacks the data returned from the scriptable core into an GenericAgents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed GenericAgents object.
        """
        ego_features = [list_tensor_data['generic_agents.ego'][0].detach().numpy()]
        agent_features = {}
        for key in list_tensor_data:
            if key.startswith('generic_agents.agents.'):
                feature_name = key[len('generic_agents.agents.'):]
                agent_features[feature_name] = [list_tensor_data[key][0].detach().numpy()]
        return GenericAgents(ego=ego_features, agents=agent_features)

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        output_dict: Dict[str, torch.Tensor] = {}
        output_list_dict: Dict[str, List[torch.Tensor]] = {}
        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}
        ego_history: torch.Tensor = tensor_data['past_ego_states']
        time_stamps: torch.Tensor = tensor_data['past_time_stamps']
        anchor_ego_state = ego_history[-1, :].squeeze()
        ego_tensor = build_generic_ego_features_from_tensor(ego_history, reverse=True)
        output_list_dict['generic_agents.ego'] = [ego_tensor]
        for feature_name in self.agent_features:
            if f'past_tracked_objects.{feature_name}' in list_tensor_data:
                agents: List[torch.Tensor] = list_tensor_data[f'past_tracked_objects.{feature_name}']
                agent_history = filter_agents_tensor(agents, reverse=True)
                if agent_history[-1].shape[0] == 0:
                    agents_tensor: torch.Tensor = torch.zeros((len(agent_history), 0, self._agents_states_dim)).float()
                else:
                    padded_agent_states = pad_agent_states(agent_history, reverse=True)
                    local_coords_agent_states = convert_absolute_quantities_to_relative(padded_agent_states, anchor_ego_state)
                    yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
                    agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)
                output_list_dict[f'generic_agents.agents.{feature_name}'] = [agents_tensor]
        return (output_dict, output_list_dict, output_list_list_dict)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {'past_ego_states': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_time_stamps': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_tracked_objects': {'iteration': '0', 'time_horizon': str(self.past_time_horizon), 'num_samples': str(self.num_past_poses), 'agent_features': ','.join(self.agent_features)}}

def _generate_multi_scale_connections(connections: torch.Tensor, scales: List[int]) -> Dict[int, torch.Tensor]:
    """
    Generate multi-scale connections by finding the neighbors up to max(scales) hops away for each node.
    :param connections: <torch.Tensor: num_connections, 2>. A 1-hop connection is represented by [start_idx, end_idx]
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
             Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
    """
    if len(connections.shape) != 2 or connections.shape[1] != 2:
        raise ValueError(f'Unexpected connections shape: {connections.shape}')
    node_idx_to_neighbor_dict: Dict[int, Dict[str, Dict[int, int]]] = {}
    dummy_value: int = 0
    for connection in connections:
        start_idx, end_idx = (connection[0].item(), connection[1].item())
        if start_idx not in node_idx_to_neighbor_dict:
            start_empty: Dict[int, int] = {}
            node_idx_to_neighbor_dict[start_idx] = {'1_hop_neighbors': start_empty}
        if end_idx not in node_idx_to_neighbor_dict:
            end_empty: Dict[int, int] = {}
            node_idx_to_neighbor_dict[end_idx] = {'1_hop_neighbors': end_empty}
        node_idx_to_neighbor_dict[start_idx]['1_hop_neighbors'][end_idx] = dummy_value
    for scale in range(2, max(scales) + 1):
        scale_hop_neighbors = f'{scale}_hop_neighbors'
        prev_scale_hop_neighbors = f'{scale - 1}_hop_neighbors'
        for neighbor_dict in node_idx_to_neighbor_dict.values():
            empty: Dict[int, int] = {}
            neighbor_dict[scale_hop_neighbors] = empty
            for n_hop_neighbor in neighbor_dict[prev_scale_hop_neighbors]:
                for n_plus_1_hop_neighbor in node_idx_to_neighbor_dict[n_hop_neighbor]['1_hop_neighbors']:
                    neighbor_dict[scale_hop_neighbors][n_plus_1_hop_neighbor] = dummy_value
    return _accumulate_connections(node_idx_to_neighbor_dict, scales)

def _accumulate_connections(node_idx_to_neighbor_dict: Dict[int, Dict[str, Dict[int, int]]], scales: List[int]) -> Dict[int, torch.Tensor]:
    """
    Accumulate the connections over multiple scales
    :param node_idx_to_neighbor_dict: {node_idx: neighbor_dict} where each neighbor_dict
                                      will have format {'i_hop_neighbors': set_of_i_hop_neighbors}
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
    """
    multi_scale_connections: Dict[int, torch.Tensor] = {}
    for scale in scales:
        scale_hop_neighbors = f'{scale}_hop_neighbors'
        scale_connections: List[List[int]] = []
        for node_idx, neighbor_dict in node_idx_to_neighbor_dict.items():
            for n_hop_neighbor in neighbor_dict[scale_hop_neighbors]:
                scale_connections.append([node_idx, n_hop_neighbor])
        if len(scale_connections) == 0:
            multi_scale_connections[scale] = torch.empty((0, 2), dtype=torch.int64)
        else:
            multi_scale_connections[scale] = torch.tensor(scale_connections, dtype=torch.int64)
    return multi_scale_connections

class VectorMapFeatureBuilder(ScriptableFeatureBuilder):
    """
    Feature builder for constructing map features in a vector-representation.
    """

    def __init__(self, radius: float, connection_scales: Optional[List[int]]=None) -> None:
        """
        Initialize vector map builder with configuration parameters.
        :param radius:  The query radius scope relative to the current ego-pose.
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        super().__init__()
        self._radius = radius
        self._connection_scales = connection_scales

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return VectorMap

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'vector_map'

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            lane_seg_coords, lane_seg_conns, lane_seg_groupings, lane_seg_lane_ids, lane_seg_roadblock_ids = get_neighbor_vector_map(scenario.map_api, ego_coords, self._radius)
            on_route_status = get_on_route_status(scenario.get_route_roadblock_ids(), lane_seg_roadblock_ids)
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(lane_seg_coords, lane_seg_conns, lane_seg_groupings, on_route_status, traffic_light_data, ego_state.rear_axle)
            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            ego_state = current_input.history.ego_states[-1]
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            lane_seg_coords, lane_seg_conns, lane_seg_groupings, lane_seg_lane_ids, lane_seg_roadblock_ids = get_neighbor_vector_map(initialization.map_api, ego_coords, self._radius)
            on_route_status = get_on_route_status(initialization.route_roadblock_ids, lane_seg_roadblock_ids)
            if current_input.traffic_light_data is None:
                raise ValueError('Cannot build VectorMap feature. PlannerInput.traffic_light_data is None')
            traffic_light_data = current_input.traffic_light_data
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(lane_seg_coords, lane_seg_conns, lane_seg_groupings, on_route_status, traffic_light_data, ego_state.rear_axle)
            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.ignore
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> VectorMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorMap.
        """
        multi_scale_connections: Dict[int, torch.Tensor] = {}
        for key in list_tensor_data:
            if key.startswith('vector_map.multi_scale_connections_'):
                multi_scale_connections[int(key[len('vector_map.multi_scale_connections_'):])] = list_tensor_data[key][0].detach().numpy()
        lane_groupings = [t.detach().numpy() for t in list_list_tensor_data['vector_map.lane_groupings'][0]]
        return VectorMap(coords=[list_tensor_data['vector_map.coords'][0].detach().numpy()], lane_groupings=[lane_groupings], multi_scale_connections=[multi_scale_connections], on_route_status=[list_tensor_data['vector_map.on_route_status'][0].detach().numpy()], traffic_light_data=[list_tensor_data['vector_map.traffic_light_data'][0].detach().numpy()])

    @torch.jit.ignore
    def _pack_to_feature_tensor_dict(self, lane_coords: LaneSegmentCoords, lane_conns: LaneSegmentConnections, lane_groupings: LaneSegmentGroupings, lane_on_route_status: LaneOnRouteStatusData, traffic_light_data: LaneSegmentTrafficLightData, anchor_state: StateSE2) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature tranform.
        :param lane_coords: The LaneSegmentCoords returned from `get_neighbor_vector_map` to transform.
        :param lane_conns: The LaneSegmentConnections returned from `get_neighbor_vector_map` to transform.
        :param lane_groupings: The LaneSegmentGroupings returned from `get_neighbor_vector_map` to transform.
        :param lane_on_route_status: The LaneOnRouteStatusData returned from `get_neighbor_vector_map` to transform.
        :param traffic_light_data: The LaneSegmentTrafficLightData returned from `get_neighbor_vector_map` to transform.
        :param anchor_state: The ego state to transform to vector.
        """
        lane_segment_coords: torch.tensor = torch.tensor(lane_coords.to_vector(), dtype=torch.float64)
        lane_segment_conns: torch.tensor = torch.tensor(lane_conns.to_vector(), dtype=torch.int64)
        on_route_status: torch.tensor = torch.tensor(lane_on_route_status.to_vector(), dtype=torch.float32)
        traffic_light_array: torch.tensor = torch.tensor(traffic_light_data.to_vector(), dtype=torch.float32)
        lane_segment_groupings: List[torch.tensor] = []
        for lane_grouping in lane_groupings.to_vector():
            lane_segment_groupings.append(torch.tensor(lane_grouping, dtype=torch.int64))
        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)
        return ({'lane_segment_coords': lane_segment_coords, 'lane_segment_conns': lane_segment_conns, 'on_route_status': on_route_status, 'traffic_light_array': traffic_light_array, 'anchor_state': anchor_state_tensor}, {'lane_segment_groupings': lane_segment_groupings}, {})

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        lane_segment_coords = tensor_data['lane_segment_coords']
        anchor_state = tensor_data['anchor_state']
        lane_segment_conns = tensor_data['lane_segment_conns']
        if len(lane_segment_conns.shape) == 1:
            if lane_segment_conns.shape[0] == 0:
                lane_segment_conns = torch.zeros((0, 2), device=lane_segment_coords.device, layout=lane_segment_coords.layout, dtype=torch.int64)
            else:
                raise ValueError(f'Unexpected shape for lane_segment_conns: {lane_segment_conns.shape}')
        lane_segment_coords = lane_segment_coords.reshape(-1, 2)
        lane_segment_coords = coordinates_to_local_frame(lane_segment_coords, anchor_state, precision=torch.float64)
        lane_segment_coords = lane_segment_coords.reshape(-1, 2, 2).float()
        if self._connection_scales is not None:
            multi_scale_connections = _generate_multi_scale_connections(lane_segment_conns, self._connection_scales)
        else:
            multi_scale_connections = {1: lane_segment_conns}
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {'vector_map.lane_groupings': [list_tensor_data['lane_segment_groupings']]}
        list_tensor_output: Dict[str, List[torch.Tensor]] = {'vector_map.coords': [lane_segment_coords], 'vector_map.on_route_status': [tensor_data['on_route_status']], 'vector_map.traffic_light_data': [tensor_data['traffic_light_array']]}
        for key in multi_scale_connections:
            list_tensor_output[f'vector_map.multi_scale_connections_{key}'] = [multi_scale_connections[key]]
        tensor_output: Dict[str, torch.Tensor] = {}
        return (tensor_output, list_tensor_output, list_list_tensor_output)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        return {'neighbor_vector_map': {'radius': str(self._radius)}, 'initial_ego_state': empty}

class VectorSetMapFeatureBuilder(ScriptableFeatureBuilder):
    """
    Feature builder for constructing map features in a vector set representation, similar to that of
        VectorNet ("VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation").
    """

    def __init__(self, map_features: List[str], max_elements: Dict[str, int], max_points: Dict[str, int], radius: float, interpolation_method: str) -> None:
        """
        Initialize vector set map builder with configuration parameters.
        :param map_features: name of map features to be extracted.
        :param max_elements: maximum number of elements to extract per feature layer.
        :param max_points: maximum number of points per feature to extract per feature layer.
        :param radius:  [m ]The query radius scope relative to the current ego-pose.
        :param interpolation_method: Interpolation method to apply when interpolating to maintain fixed size
            map elements.
        :return: Vector set map data including map element coordinates and traffic light status info.
        """
        super().__init__()
        self.map_features = map_features
        self.max_elements = max_elements
        self.max_points = max_points
        self.radius = radius
        self.interpolation_method = interpolation_method
        self._traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()
        for feature_name in self.map_features:
            try:
                VectorFeatureLayer[feature_name]
            except KeyError:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')
            if feature_name not in self.max_elements:
                raise RuntimeError(f'Max elements unavailable for {feature_name} feature layer!')
            if feature_name not in self.max_points:
                raise RuntimeError(f'Max points unavailable for {feature_name} feature layer!')

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return VectorSetMap

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'vector_set_map'

    @torch.jit.unused
    def get_scriptable_input_from_scenario(self, scenario: AbstractScenario) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
        coords, traffic_light_data = get_neighbor_vector_set_map(scenario.map_api, self.map_features, ego_coords, self.radius, route_roadblock_ids, [TrafficLightStatuses(traffic_light_data)])
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(coords, traffic_light_data[0], ego_state.rear_axle)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_scriptable_input_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the simulation objects
        :param current_input: planner input from sim
        :param initialization: planner initialization from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = current_input.history.ego_states[-1]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = initialization.route_roadblock_ids
        if current_input.traffic_light_data is None:
            raise ValueError('Cannot build VectorSetMap feature. PlannerInput.traffic_light_data is None')
        traffic_light_data = current_input.traffic_light_data
        coords, traffic_light_data = get_neighbor_vector_set_map(initialization.map_api, self.map_features, ego_coords, self.radius, route_roadblock_ids, [TrafficLightStatuses(traffic_light_data)])
        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(coords, traffic_light_data[0], ego_state.rear_axle)
        return (tensor, list_tensor, list_list_tensor)

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_scenario(scenario)
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_simulation(current_input, initialization)
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(tensor_data, list_tensor_data, list_list_tensor_data)
        return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> VectorSetMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorSetMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorSetMap.
        """
        coords: Dict[str, List[FeatureDataType]] = {}
        traffic_light_data: Dict[str, List[FeatureDataType]] = {}
        availabilities: Dict[str, List[FeatureDataType]] = {}
        for key in list_tensor_data:
            if key.startswith('vector_set_map.coords.'):
                feature_name = key[len('vector_set_map.coords.'):]
                coords[feature_name] = [list_tensor_data[key][0].detach().numpy()]
            if key.startswith('vector_set_map.traffic_light_data.'):
                feature_name = key[len('vector_set_map.traffic_light_data.'):]
                traffic_light_data[feature_name] = [list_tensor_data[key][0].detach().numpy()]
            if key.startswith('vector_set_map.availabilities.'):
                feature_name = key[len('vector_set_map.availabilities.'):]
                availabilities[feature_name] = [list_tensor_data[key][0].detach().numpy()]
        return VectorSetMap(coords=coords, traffic_light_data=traffic_light_data, availabilities=availabilities)

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(self, coords: Dict[str, MapObjectPolylines], traffic_light_data: Dict[str, LaneSegmentTrafficLightData], anchor_state: StateSE2) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature transform.
        :param coords: Dictionary mapping feature name to polyline vector sets.
        :param traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
        :param anchor_state: The ego state to transform to vector.
        :return
           tensor_data: Packed tensor data.
           list_tensor_data: Packed List[tensor] data.
           list_list_tensor_data: Packed List[List[tensor]] data.
        """
        tensor_data: Dict[str, torch.Tensor] = {}
        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)
        tensor_data['anchor_state'] = anchor_state_tensor
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        for feature_name, feature_coords in coords.items():
            list_feature_coords: List[torch.Tensor] = []
            for element_coords in feature_coords.to_vector():
                list_feature_coords.append(torch.tensor(element_coords, dtype=torch.float64))
            list_tensor_data[f'coords.{feature_name}'] = list_feature_coords
            if feature_name in traffic_light_data:
                list_feature_tl_data: List[torch.Tensor] = []
                for element_tl_data in traffic_light_data[feature_name].to_vector():
                    list_feature_tl_data.append(torch.tensor(element_tl_data, dtype=torch.float32))
                list_tensor_data[f'traffic_light_data.{feature_name}'] = list_feature_tl_data
        return (tensor_data, list_tensor_data, {})

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        tensor_output: Dict[str, torch.Tensor] = {}
        list_tensor_output: Dict[str, List[torch.Tensor]] = {}
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {}
        anchor_state = tensor_data['anchor_state']
        for feature_name in self.map_features:
            if f'coords.{feature_name}' in list_tensor_data:
                feature_coords = list_tensor_data[f'coords.{feature_name}']
                feature_tl_data = [list_tensor_data[f'traffic_light_data.{feature_name}']] if f'traffic_light_data.{feature_name}' in list_tensor_data else None
                coords, tl_data, avails = convert_feature_layer_to_fixed_size(feature_coords, feature_tl_data, self.max_elements[feature_name], self.max_points[feature_name], self._traffic_light_encoding_dim, interpolation=self.interpolation_method if feature_name in [VectorFeatureLayer.LANE.name, VectorFeatureLayer.LEFT_BOUNDARY.name, VectorFeatureLayer.RIGHT_BOUNDARY.name, VectorFeatureLayer.ROUTE_LANES.name] else None)
                coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)
                list_tensor_output[f'vector_set_map.coords.{feature_name}'] = [coords]
                list_tensor_output[f'vector_set_map.availabilities.{feature_name}'] = [avails]
                if tl_data is not None:
                    list_tensor_output[f'vector_set_map.traffic_light_data.{feature_name}'] = [tl_data[0]]
        return (tensor_output, list_tensor_output, list_list_tensor_output)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        max_elements: List[str] = [f'{feature_name}.{feature_max_elements}' for feature_name, feature_max_elements in self.max_elements.items()]
        max_points: List[str] = [f'{feature_name}.{feature_max_points}' for feature_name, feature_max_points in self.max_points.items()]
        return {'neighbor_vector_set_map': {'radius': str(self.radius), 'interpolation_method': self.interpolation_method, 'map_features': ','.join(self.map_features), 'max_elements': ','.join(max_elements), 'max_points': ','.join(max_points)}, 'initial_ego_state': empty}

class AgentsFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, trajectory_sampling: TrajectorySampling, object_type: TrackedObjectType=TrackedObjectType.VEHICLE) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        :param object_type: Type of agents (TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN) set to TrackedObjectType.VEHICLE by default
        """
        super().__init__()
        if object_type not in [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]:
            raise ValueError(f"The model's been tested just for vehicles and pedestrians types, but the provided object_type is {object_type}.")
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon
        self.object_type = object_type
        self._agents_states_dim = Agents.agents_states_dim()

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return 'agents'

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Agents

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> Agents:
        """Inherited, see superclass."""
        with torch.no_grad():
            anchor_ego_state = scenario.initial_ego_state
            past_ego_states = scenario.get_ego_past_trajectory(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)
            sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
            time_stamps = list(scenario.get_past_timestamps(iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon)) + [scenario.start_time]
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [tracked_objects.tracked_objects for tracked_objects in scenario.get_past_tracked_objects(iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses)]
            sampled_past_observations = past_tracked_objects + [present_tracked_objects]
            assert len(sampled_past_ego_states) == len(sampled_past_observations), f'Expected the trajectory length of ego and agent to be equal. Got ego: {len(sampled_past_ego_states)} and agent: {len(sampled_past_observations)}'
            assert len(sampled_past_observations) > 2, f'Trajectory of length of {len(sampled_past_observations)} needs to be at least 3'
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: Agents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> Agents:
        """Inherited, see superclass."""
        with torch.no_grad():
            history = current_input.history
            assert isinstance(history.observations[0], DetectionsTracks), f'Expected observation of type DetectionTracks, got {type(history.observations[0])}'
            present_ego_state, present_observation = history.current_state
            past_observations = history.observations[:-1]
            past_ego_states = history.ego_states[:-1]
            assert history.sample_interval, 'SimulationHistoryBuffer sample interval is None'
            indices = sample_indices_with_time_horizon(self.num_past_poses, self.past_time_horizon, history.sample_interval)
            try:
                sampled_past_observations = [cast(DetectionsTracks, past_observations[-idx]).tracked_objects for idx in reversed(indices)]
                sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
            except IndexError:
                raise RuntimeError(f'SimulationHistoryBuffer duration: {history.duration} is too short for requested past_time_horizon: {self.past_time_horizon}. Please increase the simulation_buffer_duration in default_simulation.yaml')
            sampled_past_observations = sampled_past_observations + [cast(DetectionsTracks, present_observation).tracked_objects]
            sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]
            time_stamps = [state.time_point for state in sampled_past_ego_states]
            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(sampled_past_ego_states, time_stamps, sampled_past_observations)
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
            output: Agents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(self, past_ego_states: List[EgoState], past_time_stamps: List[TimePoint], past_tracked_objects: List[TrackedObjects]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param past_ego_states: The past states of the ego vehicle.
        :param past_time_stamps: The past time stamps of the input data.
        :param past_tracked_objects: The past tracked objects.
        :return: The packed tensors.
        """
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(past_ego_states)
        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)
        past_tracked_objects_tensor_list = sampled_tracked_objects_to_tensor_list(past_tracked_objects=past_tracked_objects, object_type=self.object_type)
        return ({'past_ego_states': past_ego_states_tensor, 'past_time_stamps': past_time_stamps_tensor}, {'past_tracked_objects': past_tracked_objects_tensor_list}, {})

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Agents:
        """
        Unpacks the data returned from the scriptable core into an Agents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed Agents object.
        """
        ego_features = [list_tensor_data['agents.ego'][0].detach().numpy()]
        agent_features = [list_tensor_data['agents.agents'][0].detach().numpy()]
        return Agents(ego=ego_features, agents=agent_features)

    @torch.jit.export
    def scriptable_forward(self, tensor_data: Dict[str, torch.Tensor], list_tensor_data: Dict[str, List[torch.Tensor]], list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        ego_history: torch.Tensor = tensor_data['past_ego_states']
        time_stamps: torch.Tensor = tensor_data['past_time_stamps']
        agents: List[torch.Tensor] = list_tensor_data['past_tracked_objects']
        anchor_ego_state = ego_history[-1, :].squeeze()
        agent_history = filter_agents_tensor(agents, reverse=True)
        if agent_history[-1].shape[0] == 0:
            agents_tensor: torch.Tensor = torch.zeros((len(agent_history), 0, self._agents_states_dim)).float()
        else:
            padded_agent_states = pad_agent_states(agent_history, reverse=True)
            local_coords_agent_states = convert_absolute_quantities_to_relative(padded_agent_states, anchor_ego_state)
            yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
            agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)
        ego_tensor = build_ego_features_from_tensor(ego_history, reverse=True)
        output_dict: Dict[str, torch.Tensor] = {}
        output_list_dict: Dict[str, List[torch.Tensor]] = {'agents.ego': [ego_tensor], 'agents.agents': [agents_tensor]}
        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}
        return (output_dict, output_list_dict, output_list_list_dict)

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {'past_ego_states': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_time_stamps': {'iteration': '0', 'num_samples': str(self.num_past_poses), 'time_horizon': str(self.past_time_horizon)}, 'past_tracked_objects': {'iteration': '0', 'time_horizon': str(self.past_time_horizon), 'num_samples': str(self.num_past_poses)}}

class TestGenericAgentsFeatureBuilder(unittest.TestCase):
    """Test builder that constructs agent features during training and simulation."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.batch_size = 1
        self.past_time_horizon = 4.0
        self.num_agents = 10
        self.num_past_poses = 4
        self.num_total_past_poses = self.num_past_poses + 1
        self.agent_features = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'TRAFFIC_CONE', 'BARRIER', 'CZONE_SIGN', 'GENERIC_OBJECT']
        self.tracked_object_types: List[TrackedObjectType] = []
        for feature_name in self.agent_features:
            try:
                self.tracked_object_types.append(TrackedObjectType[feature_name])
            except KeyError:
                raise ValueError(f'Object representation for layer: {feature_name} is unavailable!')
        self.feature_builder = GenericAgentsFeatureBuilder(self.agent_features, TrajectorySampling(num_poses=self.num_past_poses, time_horizon=self.past_time_horizon))

    def test_generic_agent_feature_builder(self) -> None:
        """
        Test GenericAgentFeatureBuilder
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents, tracked_object_types=self.tracked_object_types)
        feature = self.feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_no_agents(self) -> None:
        """
        Test when there are no agents
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=0, tracked_object_types=self.tracked_object_types)
        feature = self.feature_builder.get_features_from_scenario(scenario)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), 0)
            self.assertEqual(feature.agents[feature_name][0].shape[1], 0)
            self.assertEqual(feature.agents[feature_name][0].shape[2], GenericAgents.agents_states_dim())

    def test_get_feature_from_simulation(self) -> None:
        """
        Test get feature from simulation
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents, tracked_object_types=self.tracked_object_types)
        mock_meta_data = PlannerInitialization(map_api=MockAbstractMap(), route_roadblock_ids=None, mission_goal=StateSE2(0, 0, 0))
        ego_past_states = list(scenario.get_ego_past_trajectory(iteration=0, num_samples=10, time_horizon=5))
        ego_initial_state = scenario.initial_ego_state
        ego_history = ego_past_states + [ego_initial_state]
        past_observations = list(scenario.get_past_tracked_objects(iteration=0, num_samples=10, time_horizon=5))
        initial_observation = scenario.initial_tracked_objects
        observation_history = past_observations + [initial_observation]
        history = SimulationHistoryBuffer.initialize_from_list(len(ego_history), ego_history, observation_history, scenario.database_interval)
        current_input = PlannerInput(iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history, traffic_light_data=scenario.get_traffic_light_status_at_iteration(0))
        feature = self.feature_builder.get_features_from_simulation(current_input=current_input, initialization=mock_meta_data)
        self.assertEqual(type(feature), GenericAgents)
        self.assertEqual(feature.batch_size, self.batch_size)
        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())
        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_agents_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the Generic Agents Feature Builder scripts properly
        """
        config = self.feature_builder.precomputed_feature_config()
        for expected_key in ['past_ego_states', 'past_time_stamps']:
            self.assertTrue(expected_key in config)
            config_dict = config[expected_key]
            self.assertTrue(len(config_dict) == 3)
            self.assertEqual(0, int(config_dict['iteration']))
            self.assertEqual(self.num_past_poses, int(config_dict['num_samples']))
            self.assertEqual(self.past_time_horizon, int(float(config_dict['time_horizon'])))
        tracked_objects_config_dict = config['past_tracked_objects']
        self.assertTrue(len(tracked_objects_config_dict) == 4)
        self.assertEqual(0, int(tracked_objects_config_dict['iteration']))
        self.assertEqual(self.num_past_poses, int(tracked_objects_config_dict['num_samples']))
        self.assertEqual(self.past_time_horizon, int(float(tracked_objects_config_dict['time_horizon'])))
        self.assertTrue('agent_features' in tracked_objects_config_dict)
        self.assertEqual(','.join(self.agent_features), tracked_objects_config_dict['agent_features'])
        num_frames = 5
        num_agents = 3
        ego_dim = EgoInternalIndex.dim()
        agent_dim = AgentInternalIndex.dim()
        past_ego_states = torch.zeros((num_frames, ego_dim), dtype=torch.float32)
        past_timestamps = torch.tensor([i * 50 for i in range(num_frames)], dtype=torch.int64)
        past_tracked_objects = [torch.ones((num_agents, agent_dim), dtype=torch.float32) for _ in range(num_frames)]
        for i in range(num_frames):
            for j in range(num_agents):
                past_tracked_objects[i][j, :] *= j + 1
        tensor_data = {'past_ego_states': past_ego_states, 'past_time_stamps': past_timestamps}
        list_tensor_data = {f'past_tracked_objects.{feature_name}': past_tracked_objects for feature_name in self.agent_features}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        scripted_builder = torch.jit.script(self.feature_builder)
        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        py_tensors, py_list_tensors, py_list_list_tensors = self.feature_builder.scriptable_forward(copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data))
        self.assertEqual(0, len(scripted_tensors))
        self.assertEqual(0, len(py_tensors))
        self.assertEqual(len(scripted_list_tensors), len(py_list_tensors))
        for key in py_list_tensors:
            scripted_list = scripted_list_tensors[key]
            py_list = py_list_tensors[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                scripted = scripted_list[i]
                py = py_list[i]
                torch.testing.assert_allclose(py, scripted, atol=0.05, rtol=0.05)
        self.assertEqual(0, len(scripted_list_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))

