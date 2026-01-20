# Cluster 26

def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)

def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
    return (pred_box_tensor, pred_score, gt_box_tensor)

