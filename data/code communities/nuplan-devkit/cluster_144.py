# Cluster 144

def convert_feature_layer_to_fixed_size(feature_coords: List[torch.Tensor], feature_tl_data_over_time: Optional[List[List[torch.Tensor]]], max_elements: int, max_points: int, traffic_light_encoding_dim: int, interpolation: Optional[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Converts variable sized map features to fixed size tensors. Map elements are padded/trimmed to max_elements size.
        Points per feature are interpolated to maintain max_points size.
    :param feature_coords: Vector set of coordinates for collection of elements in map layer.
        [num_elements, num_points_in_element (variable size), 2]
    :param feature_tl_data_over_time: Optional traffic light status corresponding to map elements at given index in coords.
        [num_frames, num_elements, traffic_light_encoding_dim (4)]
    :param max_elements: Number of elements to pad/trim to.
    :param max_points: Number of points to interpolate or pad/trim to.
    :param traffic_light_encoding_dim: Dimensionality of traffic light data.
    :param interpolation: Optional interpolation mode for maintaining fixed number of points per element.
        None indicates trimming and zero-padding to take place in lieu of interpolation. Interpolation options:
        'linear' and 'area'.
    :return
        coords_tensor: The converted coords tensor.
        tl_data_tensor: The converted traffic light data tensor (if available).
        avails_tensor: Availabilities tensor identifying real vs zero-padded data in coords_tensor and tl_data_tensor.
    :raise ValueError: If coordinates and traffic light data size do not match.
    """
    coords_tensor = torch.zeros((max_elements, max_points, 2), dtype=torch.float64)
    avails_tensor = torch.zeros((max_elements, max_points), dtype=torch.bool)
    tl_data_tensor = torch.zeros((len(feature_tl_data_over_time), max_elements, max_points, traffic_light_encoding_dim), dtype=torch.float32) if feature_tl_data_over_time is not None else None
    for element_idx in range(min(len(feature_coords), max_elements)):
        element_coords = feature_coords[element_idx]
        if interpolation is not None:
            num_points = max_points
            element_coords = interpolate_points(element_coords, max_points, interpolation=interpolation)
        else:
            num_points = min(len(element_coords), max_points)
            element_coords = element_coords[:num_points]
        coords_tensor[element_idx, :num_points] = element_coords
        avails_tensor[element_idx, :num_points] = True
        if feature_tl_data_over_time is not None and tl_data_tensor is not None:
            for time_ind in range(len(feature_tl_data_over_time)):
                if len(feature_coords) != len(feature_tl_data_over_time[time_ind]):
                    raise ValueError(f'num_elements between feature_coords and feature_tl_data_over_time inconsistent: {len(feature_coords)}, {len(feature_tl_data_over_time[time_ind])}')
                tl_data_tensor[time_ind, element_idx, :num_points] = feature_tl_data_over_time[time_ind][element_idx]
    return (coords_tensor, tl_data_tensor, avails_tensor)

def interpolate_points(coords: torch.Tensor, max_points: int, interpolation: str) -> torch.Tensor:
    """
    Interpolate points within map element to maintain fixed size.
    :param coords: Sequence of coordinate points representing map element. <torch.Tensor: num_points, 2>
    :param max_points: Desired size to interpolate to.
    :param interpolation: Torch interpolation mode. Available options: 'linear' and 'area'.
    :return: Coordinate points interpolated to max_points size.
    :raise ValueError: If coordinates dimensions are not valid.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f'Unexpected coords shape: {coords.shape}. Expected shape: (*, 2)')
    x_coords = coords[:, 0].unsqueeze(0).unsqueeze(0)
    y_coords = coords[:, 1].unsqueeze(0).unsqueeze(0)
    align_corners = True if interpolation == 'linear' else None
    x_coords = torch.nn.functional.interpolate(x_coords, max_points, mode=interpolation, align_corners=align_corners)
    y_coords = torch.nn.functional.interpolate(y_coords, max_points, mode=interpolation, align_corners=align_corners)
    coords = torch.stack((x_coords, y_coords), dim=-1).squeeze()
    return coords

class TestVectorPreprocessing(unittest.TestCase):
    """Test preprocessing utility functions to assist with builders for vectorized map features."""

    def setUp(self) -> None:
        """Set up test case."""
        self.max_elements = 30
        self.max_points = 20
        self.interpolation = None
        self.traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    def test_interpolate_points_functionality(self) -> None:
        """
        Test interpolating coordinate points.
        """
        coords = torch.tensor([[1, 1], [3, 1], [5, 1]], dtype=torch.float64)
        interpolated_coords = interpolate_points(coords, 5, interpolation='linear')
        self.assertEqual(interpolated_coords.shape, (5, 2))
        torch.testing.assert_allclose(coords, interpolated_coords[::2])
        torch.testing.assert_allclose(interpolated_coords[:, 1], torch.ones(5, dtype=torch.float64))
        self.assertTrue(interpolated_coords[1][0].item() > interpolated_coords[0][0].item())
        self.assertTrue(interpolated_coords[1][0].item() < interpolated_coords[2][0].item())
        self.assertTrue(interpolated_coords[3][0].item() > interpolated_coords[2][0].item())
        self.assertTrue(interpolated_coords[3][0].item() < interpolated_coords[4][0].item())
        interpolated_coords = interpolate_points(coords, 5, interpolation='area')
        self.assertEqual(interpolated_coords.shape, (5, 2))
        torch.testing.assert_allclose(coords, interpolated_coords[::2])
        torch.testing.assert_allclose(interpolated_coords[:, 1], torch.ones(5, dtype=torch.float64))
        self.assertTrue(interpolated_coords[1][0].item() > interpolated_coords[0][0].item())
        self.assertTrue(interpolated_coords[1][0].item() < interpolated_coords[2][0].item())
        self.assertTrue(interpolated_coords[3][0].item() > interpolated_coords[2][0].item())
        self.assertTrue(interpolated_coords[3][0].item() < interpolated_coords[4][0].item())

    def test_interpolate_points_scriptability(self) -> None:
        """
        Tests that the function interpolate_points scripts properly.
        """

        class tmp_module(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, coords: torch.Tensor, max_points: int, interpolation: str) -> torch.Tensor:
                result = interpolate_points(coords, max_points, interpolation)
                return result
        to_script = tmp_module()
        scripted = torch.jit.script(to_script)
        test_coords = torch.tensor([[1, 1], [3, 1], [5, 1]], dtype=torch.float64)
        py_result = to_script.forward(test_coords, 5, 'linear')
        script_result = scripted.forward(test_coords, 5, 'linear')
        torch.testing.assert_allclose(py_result, script_result)

    def test_convert_feature_layer_to_fixed_size_functionality(self) -> None:
        """
        Test converting variable size data to fixed size tensors.
        """
        coords: List[torch.Tensor] = [torch.tensor([[0.0, 0.0]])]
        traffic_light_data: List[torch.Tensor] = [[torch.tensor([LaneSegmentTrafficLightData.encode(TrafficLightStatusType.UNKNOWN)])]]
        coords_tensor, tl_data_tensor, avails_tensor = convert_feature_layer_to_fixed_size(coords, traffic_light_data, self.max_elements, self.max_points, self.traffic_light_encoding_dim, self.interpolation)
        self.assertIsInstance(coords_tensor, torch.DoubleTensor)
        self.assertIsInstance(tl_data_tensor, torch.FloatTensor)
        self.assertIsInstance(avails_tensor, torch.BoolTensor)
        self.assertEqual(coords_tensor.shape, (self.max_elements, self.max_points, 2))
        self.assertEqual(tl_data_tensor[0].shape, (self.max_elements, self.max_points, LaneSegmentTrafficLightData.encoding_dim()))
        self.assertEqual(avails_tensor.shape, (self.max_elements, self.max_points))
        expected_avails = torch.zeros(avails_tensor.shape, dtype=torch.bool)
        expected_avails[0][0] = True
        torch.testing.assert_equal(expected_avails, avails_tensor)
        coords_tensor, tl_data_tensor, avails_tensor = convert_feature_layer_to_fixed_size(coords, traffic_light_data, self.max_elements, self.max_points, self.traffic_light_encoding_dim, interpolation='linear')
        expected_avails = torch.zeros(avails_tensor.shape, dtype=torch.bool)
        expected_avails[0][:] = True
        torch.testing.assert_equal(expected_avails, avails_tensor)

    def test_convert_feature_layer_to_fixed_size_scriptability(self) -> None:
        """
        Tests that the function convert_feature_layer_to_fixed_size scripts properly.
        """

        class tmp_module(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, coords: List[torch.Tensor], traffic_light_data: Optional[List[List[torch.Tensor]]], max_elements: int, max_points: int, traffic_light_encoding_dim: int, interpolation: Optional[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
                result_coords, result_tl_data, result_avails = convert_feature_layer_to_fixed_size(coords, traffic_light_data, max_elements, max_points, traffic_light_encoding_dim, interpolation)
                return (result_coords, result_tl_data, result_avails)
        to_script = tmp_module()
        scripted = torch.jit.script(to_script)
        test_coords: List[torch.Tensor] = [torch.tensor([[0.0, 0.0]])]
        test_traffic_light_data: List[torch.Tensor] = [[torch.tensor([LaneSegmentTrafficLightData.encode(TrafficLightStatusType.UNKNOWN)])]]
        py_result_coords, py_script_result_tl_data, py_script_result_avails = to_script.forward(test_coords, test_traffic_light_data, self.max_elements, self.max_points, self.traffic_light_encoding_dim, self.interpolation)
        script_result_coords, script_result_tl_data, script_result_avails = scripted.forward(test_coords, test_traffic_light_data, self.max_elements, self.max_points, self.traffic_light_encoding_dim, self.interpolation)
        torch.testing.assert_allclose(py_result_coords, script_result_coords)
        torch.testing.assert_allclose(py_script_result_tl_data, script_result_tl_data)
        torch.testing.assert_allclose(py_script_result_avails, script_result_avails)

class tmp_module(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, coords: List[torch.Tensor], traffic_light_data: Optional[List[List[torch.Tensor]]], max_elements: int, max_points: int, traffic_light_encoding_dim: int, interpolation: Optional[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        result_coords, result_tl_data, result_avails = convert_feature_layer_to_fixed_size(coords, traffic_light_data, max_elements, max_points, traffic_light_encoding_dim, interpolation)
        return (result_coords, result_tl_data, result_avails)

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

class TestVectorUtils(unittest.TestCase):
    """Test vector building utility functions."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        scenario = MockAbstractScenario()
        ego_state = scenario.initial_ego_state
        self.ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        self.map_api = scenario.map_api
        self.route_roadblock_ids = scenario.get_route_roadblock_ids()
        self.traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
        self.radius = 35
        self.map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        self._num_past_poses = 1
        self._past_time_horizon = 1.0
        self._num_future_poses = 5
        self._future_time_horizon = 5.0
        current_tl = [TrafficLightStatuses(list(scenario.get_traffic_light_status_at_iteration(iteration=0)))]
        past_tl = scenario.get_past_traffic_light_status_history(iteration=0, num_samples=self._num_past_poses, time_horizon=self._past_time_horizon)
        future_tl = scenario.get_future_traffic_light_status_history(iteration=0, num_samples=self._num_future_poses, time_horizon=self._future_time_horizon)
        past_tl_list = list(past_tl)
        future_tl_list = list(future_tl)
        self.traffic_light_data_over_time = past_tl_list + current_tl + future_tl_list

    def test_prune_route_by_connectivity(self) -> None:
        """
        Test pruning route roadblock ids by those within query radius (specified in roadblock_ids)
        maintaining connectivity.
        """
        route_roadblock_ids = ['-1', '0', '1', '2', '3']
        roadblock_ids = {'0', '1', '3'}
        pruned_route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids, roadblock_ids)
        self.assertEqual(pruned_route_roadblock_ids, ['0', '1'])

    def test_get_lane_polylines(self) -> None:
        """
        Test extracting lane/lane connector baseline path and boundary polylines from given map api.
        """
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(self.map_api, self.ego_coords, self.radius)
        assert type(lanes_mid) == MapObjectPolylines
        assert type(lanes_left) == MapObjectPolylines
        assert type(lanes_right) == MapObjectPolylines
        assert type(lane_ids) == LaneSegmentLaneIDs

    def test_get_map_object_polygons(self) -> None:
        """
        Test extracting map object polygons from map.
        """
        for layer in [SemanticMapLayer.CROSSWALK, SemanticMapLayer.STOP_LINE]:
            polygons = get_map_object_polygons(self.map_api, self.ego_coords, self.radius, layer)
            assert type(polygons) == MapObjectPolylines

    def test_get_route_polygon_from_roadblock_ids(self) -> None:
        """
        Test extracting route polygon from map given list of roadblock ids.
        """
        route = get_route_polygon_from_roadblock_ids(self.map_api, self.ego_coords, self.radius, self.route_roadblock_ids)
        assert type(route) == MapObjectPolylines

    def test_get_route_lane_polylines_from_roadblock_ids(self) -> None:
        """
        Test extracting route lane polylines from map given list of roadblock ids.
        """
        route = get_route_lane_polylines_from_roadblock_ids(self.map_api, self.ego_coords, self.radius, self.route_roadblock_ids)
        assert type(route) == MapObjectPolylines

    def test_get_on_route_status(self) -> None:
        """
        Test identifying whether given roadblock lie within goal route.
        """
        route_roadblock_ids = ['0']
        roadblock_ids = LaneSegmentRoadBlockIDs(['0', '1'])
        on_route_status = get_on_route_status(route_roadblock_ids, roadblock_ids)
        assert type(on_route_status) == LaneOnRouteStatusData
        assert len(on_route_status.on_route_status) == LaneOnRouteStatusData.encoding_dim()
        assert on_route_status.on_route_status[0] == on_route_status.encode(OnRouteStatusType.ON_ROUTE)
        assert on_route_status.on_route_status[1] == on_route_status.encode(OnRouteStatusType.OFF_ROUTE)

    def test_get_neighbor_vector_map(self) -> None:
        """
        Test extracting neighbor vector map information from map api.
        """
        lane_seg_coords, lane_seg_conns, lane_seg_groupings, lane_seg_lane_ids, lane_seg_roadblock_ids = get_neighbor_vector_map(self.map_api, self.ego_coords, self.radius)
        assert type(lane_seg_coords) == LaneSegmentCoords
        assert type(lane_seg_conns) == LaneSegmentConnections
        assert type(lane_seg_groupings) == LaneSegmentGroupings
        assert type(lane_seg_lane_ids) == LaneSegmentLaneIDs
        assert type(lane_seg_roadblock_ids) == LaneSegmentRoadBlockIDs

    def test_get_neighbor_vector_set_map(self) -> None:
        """
        Test extracting neighbor vector set map information from map api.
        """
        coords, traffic_light_data = get_neighbor_vector_set_map(self.map_api, self.map_features, self.ego_coords, self.radius, self.route_roadblock_ids, [TrafficLightStatuses(self.traffic_light_data)])
        for feature_name in self.map_features:
            assert feature_name in coords
            assert type(coords[feature_name]) == MapObjectPolylines
        assert len(traffic_light_data) == 1
        assert 'LANE' in traffic_light_data[0]
        assert type(traffic_light_data[0]['LANE']) == LaneSegmentTrafficLightData

    def test_get_neighbor_vector_set_map_for_time_horizon(self) -> None:
        """
        Test extracting neighbor vector set map information from map api.
        """
        coords, traffic_light_data_list = get_neighbor_vector_set_map(self.map_api, self.map_features, self.ego_coords, self.radius, self.route_roadblock_ids, self.traffic_light_data_over_time)
        for feature_name in self.map_features:
            assert feature_name in coords
            assert type(coords[feature_name]) == MapObjectPolylines
        for traffic_light_data in traffic_light_data_list:
            assert 'LANE' in traffic_light_data
            assert type(traffic_light_data['LANE']) == LaneSegmentTrafficLightData

def _form_lane_segment_coords_connections_from_points(points: List[Point2D], start_lane_segment_index: int) -> Tuple[LaneSegmentCoords, LaneSegmentConnections]:
    """
    Helper function to take in a set of points and convert into an example set of lane segments and lane connections.
    We assume that points i and (i+1) form lane segments l_i.
    We assume lane_segment l_i connects to segment l_{i+1}.
    :param points: The list of points to form lane segments + connections from.
    :param start_lane_segment_index: This is used to label the lane segments by setting the starting value of i above.
    :return: The lane segments coordinates (start + end point) and connectivity (lane_segment_from, lane_segment_to).
    """
    segments = [(p_prev, p_next) for p_prev, p_next in zip(points[:-1], points[1:])]
    connections = [(start_lane_segment_index + idx, start_lane_segment_index + idx + 1) for idx in range(len(segments) - 1)]
    return (LaneSegmentCoords(segments), LaneSegmentConnections(connections))

def _get_neighbor_vector_map_patch(map_api: AbstractMap, point: Point2D, radius: float) -> Tuple[LaneSegmentCoords, LaneSegmentConnections, LaneSegmentGroupings, LaneSegmentLaneIDs, LaneSegmentRoadBlockIDs]:
    """
    A patch for get_neighbor_vector_map that uses the following dummy map for testing.
    Original function docstring:
    Extract neighbor vector map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :return
        lane_seg_coords: lane_segment coords in shape of [num_lane_segment, 2, 2].
        lane_seg_conns: lane_segment connections [start_idx, end_idx] in shape of [num_connection, 2].
        lane_seg_groupings: collection of lane_segment indices in each lane in shape of
            [num_lane, num_lane_segment_in_lane].
        lane_seg_lane_ids: lane ids of segments at given index in coords in shape of [num_lane_segment 1].
        lane_seg_roadblock_ids: roadblock ids of segments at given index in coords in shape of [num_lane_segment 1].

    Dummy map setup where ls = lane_segment, lc = lane/lane connector, rb = roadblock.  Origin at the center of the map.

    ls_id    0  1  2  3  4  5
    lc_id   5000    5001   5002
            ______|_____|______

            x--x--x--x--x--x--x
    origin           O
            x--x--x--x--x--x--x
            ______|_____|______
    ls_id    6  7  8  9  10 11
    lc_id   5003    5004   5005
    rb_id   60000  70000  80000
    """
    top_line_points = [Point2D(x=x, y=1) for x in range(-3, 4)]
    top_line_segments_coords, top_line_segment_connections = _form_lane_segment_coords_connections_from_points(points=top_line_points, start_lane_segment_index=0)
    bottom_line_points = [Point2D(x=x, y=-1) for x in range(-3, 4)]
    bottom_line_segments_coords, bottom_line_segment_connections = _form_lane_segment_coords_connections_from_points(points=bottom_line_points, start_lane_segment_index=len(top_line_segments_coords.coords))
    combined_coords = LaneSegmentCoords(coords=top_line_segments_coords.coords + bottom_line_segments_coords.coords)
    combined_connections = LaneSegmentConnections(connections=top_line_segment_connections.connections + bottom_line_segment_connections.connections)
    if len(combined_coords.coords) != 12:
        raise ValueError(f'Expected 12 lane segments to match dummy map.  Got {combined_coords} instead.')
    if len(combined_connections.connections) != 10:
        raise ValueError(f'Expected 10 lane segment connections to match dummy map.  Got {combined_connections} instead.')
    combined_lane_seg_groupings = LaneSegmentGroupings([[x, x + 1] for x in range(0, 12, 2)])
    lane_id_list = [str(x) for x in range(5000, 5006)]
    combined_lane_seg_lane_ids = LaneSegmentLaneIDs([doubled_entry for doubled_entry in itertools.chain.from_iterable(((entry, entry) for entry in lane_id_list))])
    roadblock_id_list = ['60000', '70000', '80000'] * 2
    combined_lane_seg_roadblock_ids = LaneSegmentRoadBlockIDs([doubled_entry for doubled_entry in itertools.chain.from_iterable(((entry, entry) for entry in roadblock_id_list))])
    return (combined_coords, combined_connections, combined_lane_seg_groupings, combined_lane_seg_lane_ids, combined_lane_seg_roadblock_ids)

def _ego_has_route(scenario: NuPlanScenario, map_radius: float) -> bool:
    """
    Determines the presence of an on-route lane segment in a VectorMap built from
    the given scenario within map_radius meters of the ego.
    :param scenario: A NuPlan scenario.
    :param map_radius: the radius of the VectorMap built around the ego's position
    to check for on-route lane segments.
    :return: True if there is at least one on-route lane segment in the VectorMap.
    """
    ego_state = scenario.initial_ego_state
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
    _, _, _, _, lane_seg_roadblock_ids = get_neighbor_vector_map(scenario.map_api, ego_coords, map_radius)
    map_lane_roadblock_ids = set(lane_seg_roadblock_ids.roadblock_ids)
    return len(map_lane_roadblock_ids.intersection(scenario.get_route_roadblock_ids())) > 0

def filter_ego_has_route(scenario_dict: ScenarioDict, map_radius: float) -> ScenarioDict:
    """
    Rid a scenario dictionary of the scenarios that don't have an on-route lane segment within map_radius meters of the ego.
    Uses a VectorMap to gather lane segments.
    :param scenario_dict: Dictionary that holds a list of scenarios for each scenario type.
    :param map_radius: How far out from ego to check for on-route lane segments.
    :return: Filtered scenario dictionary.
    """
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = list(filter(lambda scenario: _ego_has_route(scenario, map_radius), scenario_dict[scenario_type]))
    return scenario_dict

class TestNuPlanScenarioFilterUtils(unittest.TestCase):
    """
    Tests scenario filter utils for NuPlan
    """

    def _get_mock_scenario_dict(self) -> Dict[str, List[CachedScenario]]:
        """Gets mock scenario dict."""
        return {DEFAULT_SCENARIO_NAME: [CachedScenario(log_name='log/name', token=DEFAULT_SCENARIO_NAME, scenario_type=DEFAULT_SCENARIO_NAME) for i in range(500)], 'lane_following_with_lead': [CachedScenario(log_name='log/name', token='lane_following_with_lead', scenario_type='lane_following_with_lead') for i in range(80)], 'unprotected_left_turn': [CachedScenario(log_name='log/name', token='unprotected_left_turn', scenario_type='unprotected_left_turn') for i in range(120)]}

    def _get_mock_nuplan_scenario_dict_for_timestamp_filtering(self) -> Dict[str, List[CachedScenario]]:
        """Gets mock scenario dict."""
        mock_scenario_dict = {DEFAULT_SCENARIO_NAME: [Mock(NuPlanScenario) for _ in range(0, 100, 3)], 'lane_following_with_lead': [Mock(NuPlanScenario) for _ in range(0, 100, 6)], 'lane_following_without_lead': [Mock(NuPlanScenario) for _ in range(3)]}
        for i in range(0, len(mock_scenario_dict[DEFAULT_SCENARIO_NAME]) * int(1000000.0), int(1000000.0)):
            mock_scenario_dict[DEFAULT_SCENARIO_NAME][int(i / 1000000.0)]._initial_lidar_timestamp = i * 3
        for i in range(0, len(mock_scenario_dict['lane_following_with_lead']) * int(1000000.0), int(1000000.0)):
            mock_scenario_dict['lane_following_with_lead'][int(i / 1000000.0)]._initial_lidar_timestamp = i * 6
        mock_scenario_dict['lane_following_without_lead'][0]._initial_lidar_timestamp = 5.0 * int(1000000.0)
        mock_scenario_dict['lane_following_without_lead'][1]._initial_lidar_timestamp = 100.0 * int(1000000.0)
        mock_scenario_dict['lane_following_without_lead'][2]._initial_lidar_timestamp = 6.0 * int(1000000.0)
        return mock_scenario_dict

    def _get_mock_worker_map(self) -> Callable[..., List[Any]]:
        """
        Gets mock worker_map function.
        """

        def mock_worker_map(worker: WorkerPool, fn: Callable[..., List[Any]], input_objects: List[Any]) -> List[Any]:
            """
            Mock function for worker_map
            :param worker: Worker pool
            :param fn: Callable function
            :param input_objects: List of objects to be used as input
            :return: List of output objects
            """
            return fn(input_objects)
        return mock_worker_map

    def test_filter_total_num_scenarios_int_max_scenarios_requires_removing_known_scenario_types(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 100
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)
        self.assertTrue(len(final_scenario_dict['lane_following_with_lead']) < len(mock_scenario_dict['lane_following_with_lead']))
        self.assertTrue(len(final_scenario_dict['unprotected_left_turn']) < len(mock_scenario_dict['unprotected_left_turn']))
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), limit_total_scenarios)

    def test_filter_total_num_scenarios_int_max_scenarios_less_than_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 300
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertNotEqual(final_scenario_dict[DEFAULT_SCENARIO_NAME], mock_scenario_dict[DEFAULT_SCENARIO_NAME])
        self.assertEqual(final_scenario_dict['lane_following_with_lead'], mock_scenario_dict['lane_following_with_lead'])
        self.assertEqual(final_scenario_dict['unprotected_left_turn'], mock_scenario_dict['unprotected_left_turn'])
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), limit_total_scenarios)

    def test_filter_total_num_scenarios_int_max_scenarios_more_than_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 800
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertDictEqual(final_scenario_dict, mock_scenario_dict)

    def test_filter_total_num_scenarios_float_requires_removing_known_scenario_types(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an float, the actual number of scenarios,
        where the number of scenarios required is requires reomving known scenario types.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.2
        randomize = True
        final_num_of_scenarios = int(limit_total_scenarios * sum((len(scenarios) for scenarios in mock_scenario_dict.values())))
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)
        self.assertTrue(len(final_scenario_dict['lane_following_with_lead']) < len(mock_scenario_dict['lane_following_with_lead']))
        self.assertTrue(len(final_scenario_dict['unprotected_left_turn']) < len(mock_scenario_dict['unprotected_left_turn']))
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), final_num_of_scenarios)

    def test_filter_total_num_scenarios_float_removes_only_default_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an float, the actual number of scenarios,
        where the number of scenarios required is requires reomving known scenario types.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.5
        randomize = True
        final_num_of_scenarios = int(limit_total_scenarios * sum((len(scenarios) for scenarios in mock_scenario_dict.values())))
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertNotEqual(final_scenario_dict[DEFAULT_SCENARIO_NAME], mock_scenario_dict[DEFAULT_SCENARIO_NAME])
        self.assertEqual(final_scenario_dict['lane_following_with_lead'], mock_scenario_dict['lane_following_with_lead'])
        self.assertEqual(final_scenario_dict['unprotected_left_turn'], mock_scenario_dict['unprotected_left_turn'])
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), final_num_of_scenarios)

    def test_remove_all_scenarios_int_limit_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to 0. This should raise an assertion error.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0
        randomize = True
        with self.assertRaises(AssertionError):
            filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)

    def test_remove_all_scenarios_float_limit_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to 0. This should raise an assertion error.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.0
        randomize = True
        with self.assertRaises(AssertionError):
            filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)

    def test_remove_exactly_all_default_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to number of known scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 200
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize)
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)
        self.assertEqual(len(final_scenario_dict['lane_following_with_lead']), len(mock_scenario_dict['lane_following_with_lead']))
        self.assertEqual(len(final_scenario_dict['unprotected_left_turn']), len(mock_scenario_dict['unprotected_left_turn']))
        self.assertEqual(sum((len(scenarios) for scenarios in final_scenario_dict.values())), limit_total_scenarios)

    def test_filter_scenarios_by_timestamp(self) -> None:
        """
        Tests filter_scenarios_by_timestamp with default threshold
        """
        mock_worker_map = self._get_mock_worker_map()
        mock_nuplan_scenario_dict = self._get_mock_nuplan_scenario_dict_for_timestamp_filtering()
        with patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.worker_map', mock_worker_map):
            final_scenario_dict = filter_scenarios_by_timestamp(mock_nuplan_scenario_dict.copy())
            self.assertEqual(len(final_scenario_dict['lane_following_with_lead']), len(mock_nuplan_scenario_dict['lane_following_with_lead']))
            self.assertEqual(len(final_scenario_dict[DEFAULT_SCENARIO_NAME]), len(mock_nuplan_scenario_dict[DEFAULT_SCENARIO_NAME]) * 0.5)
            self.assertEqual(len(final_scenario_dict['lane_following_without_lead']), len(mock_nuplan_scenario_dict['lane_following_without_lead']) - 1)

    def test_filter_fraction_lidarpc_tokens_in_set(self) -> None:
        """
        Test filter_fraction_lidarpc_tokens_in_set with fractional thresholds {0, 0.5, 1}.
        """
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
        mock_nuplan_scenarios = []
        for start_letter in range(4):
            mock_nuplan_scenario = Mock(NuPlanScenario)
            mock_nuplan_scenario.get_scenario_tokens.return_value = set(alphabet[start_letter:start_letter + 3])
            mock_nuplan_scenarios.append(mock_nuplan_scenario)
        full_intersection_scenario, two_intersection_scenario, one_intersection_scenario, no_intersection_scenario = mock_nuplan_scenarios
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json_path = Path(tmp_dir) / 'tmp_token_set.json'
            json.dump(['a', 'b', 'c'], open(tmp_json_path, 'w'))
            scenario_dict = {'on_pickup_dropoff': [no_intersection_scenario, one_intersection_scenario]}
            self.assertEqual(filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 0), {'on_pickup_dropoff': [one_intersection_scenario]})
            scenario_dict['on_pickup_dropoff'] = [one_intersection_scenario, two_intersection_scenario]
            self.assertEqual(filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 0.5), {'on_pickup_dropoff': [two_intersection_scenario]})
            scenario_dict['on_pickup_dropoff'] = [two_intersection_scenario, full_intersection_scenario]
            self.assertEqual(filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 1), {'on_pickup_dropoff': [full_intersection_scenario]})

    def test_filter_non_stationary_ego(self) -> None:
        """Test filter_non_stationary_ego with 0.5m displacement threshold"""
        stationary_ego_pudo_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.01, y=0.0))
        mobile_ego_pudo_scenario = MockAbstractScenario()
        scenario_dict = {'on_pickup_dropoff': [stationary_ego_pudo_scenario, mobile_ego_pudo_scenario]}
        filtered_scenario_dict = filter_non_stationary_ego(scenario_dict, minimum_threshold=0.5)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [mobile_ego_pudo_scenario])

    def test_filter_ego_starts(self) -> None:
        """Test filter_ego_starts with 0.1 m/s speed threshold"""
        slow_acceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=0.01, y=0.0), time_step=1)
        fast_acceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=1, y=0.0), time_step=1)
        scenario_dict = {'on_pickup_dropoff': [slow_acceleration_scenario, fast_acceleration_scenario]}
        filtered_scenario_dict = filter_ego_starts(scenario_dict, speed_threshold=0.1, speed_noise_tolerance=0.1)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [fast_acceleration_scenario])

    def test_filter_ego_stops(self) -> None:
        """Test filter_ego_stops with 0.1 m/s speed threshold"""
        slow_deceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=1.0, y=0.0), fixed_acceleration=StateVector2D(x=-0.01, y=0.0), time_step=1)
        fast_deceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=1.0, y=0.0), fixed_acceleration=StateVector2D(x=-1 / 9, y=0.0), time_step=1)
        scenario_dict = {'on_pickup_dropoff': [slow_deceleration_scenario, fast_deceleration_scenario]}
        filtered_scenario_dict = filter_ego_stops(scenario_dict, speed_threshold=0.1, speed_noise_tolerance=0.1)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [fast_deceleration_scenario])

    def test_ego_startstop_noise_tolerance(self) -> None:
        """Test filter_ego_starts with ego barely crossing speed threshold and noise tolerance higher than threshold"""
        fast_enough_acceleration_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=0.11, y=0.0), time_step=1)
        scenario_dict = {'on_pickup_dropoff': [fast_enough_acceleration_scenario]}
        filtered_scenario_dict = filter_ego_starts(scenario_dict, speed_threshold=1, speed_noise_tolerance=2)
        self.assertEqual(filtered_scenario_dict['on_pickup_dropoff'], [])

    def test_filter_ego_has_route(self) -> None:
        """
        Test filter_ego_has_route with one route roadblock in the VectorMap (True case),
        and with no route-intersecting roadblocks (False case).
        """
        map_radius = 35
        scenario = MockAbstractScenario()
        scenario_dict = {'on_pickup_dropoff': [scenario]}
        with patch('nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_neighbor_vector_map') as get_neighbor_vector_map:
            get_neighbor_vector_map.return_value = (None, None, None, None, LaneSegmentRoadBlockIDs(['a', 'b', 'c']))
            with patch.object(scenario, 'get_route_roadblock_ids') as get_route_roadblock_ids:
                get_route_roadblock_ids.return_value = ['d', 'e', 'a']
                self.assertEqual(filter_ego_has_route(scenario_dict, map_radius)['on_pickup_dropoff'], [scenario])
                get_route_roadblock_ids.return_value = ['d', 'e', 'f']
                self.assertEqual(filter_ego_has_route(scenario_dict, map_radius)['on_pickup_dropoff'], [])

