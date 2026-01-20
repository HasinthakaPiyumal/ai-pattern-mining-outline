# Cluster 2

def warping_module(dataset: Literal['dresscode', 'vitonhd']):
    tps = ConvNet_TPS(256, 192, 21, 3)
    refinement = UNetVanilla(n_channels=24, n_classes=3, bilinear=True)
    checkpoint_url = f'https://github.com/miccunifi/ladi-vton/releases/download/weights/warping_{dataset}.pth'
    tps.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['tps'])
    refinement.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['refinement'])
    return (tps, refinement)

