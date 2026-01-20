# Cluster 142

def _validate_generic_ego_feature_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be a GenericEgoFeature.
    :param feature: The tensor to validate.
    """
    if len(feature.shape) == 2 and feature.shape[1] == GenericEgoFeatureIndex.dim():
        return
    if len(feature.shape) == 1 and feature.shape[0] == GenericEgoFeatureIndex.dim():
        return
    raise ValueError(f'Improper ego feature shape: {feature.shape}.')

def _validate_generic_agent_feature_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be a GenericAgentFeature.
    :param feature: The tensor to validate.
    """
    if len(feature.shape) != 3 or feature.shape[2] != GenericAgentFeatureIndex.dim():
        raise ValueError(f'Improper agent feature shape: {feature.shape}.')

