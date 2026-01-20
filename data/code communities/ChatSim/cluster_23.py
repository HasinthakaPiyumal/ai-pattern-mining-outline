# Cluster 23

def merge(last_trans, sky_hdri, sur_hdri):
    """
        Merge the sky hdri and the surrounding env hdri to the final hdri

        last_trans : torch.Tensor
            shape [H, W, 1]

        sky_hdri : torch.Tensor
            shape [H, W, 3]

        sur_hdri: torch.Tensor
            shape [H, W, 3]
    """
    return sur_hdri * (1 - last_trans) + sky_hdri * last_trans

