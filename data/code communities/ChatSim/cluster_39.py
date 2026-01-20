# Cluster 39

def flatten(cubemap):
    """
    flatten the cube map for visualization
    Args:
        cubemap : torch.tensor
            shape [6, N, N, C], C usually 3
    Returns:
        cubemap_flatten : torch.tensor
            shape [3*N, 4*N, C]
    """
    view_names = ['right', 'left', 'top', 'bottom', 'back', 'front']
    cubemap_vis = torch.clone(cubemap).detach()
    for i, view_name in enumerate(view_names):
        cubemap_vis[i] = convert_cubemap_torch(cubemap_vis[i], view_name)
    _, N, N, C = cubemap_vis.shape
    cubemap_flatten = torch.zeros(N * 3, N * 4, C).to(cubemap)
    cubemap_flatten[:N, N:2 * N] = cubemap_vis[2]
    cubemap_flatten[N:2 * N, :N] = cubemap_vis[1]
    cubemap_flatten[N:2 * N, N:2 * N] = cubemap_vis[5]
    cubemap_flatten[N:2 * N, 2 * N:3 * N] = cubemap_vis[0]
    cubemap_flatten[N:2 * N, 3 * N:4 * N] = cubemap_vis[4]
    cubemap_flatten[2 * N:3 * N, N:2 * N] = cubemap_vis[3]
    return cubemap_flatten

def convert_cubemap_torch(img, view_name):
    """
    Args:
        img : torch.Tensor
            [res, res, 3]
        view_name : str
            one of ['front', 'back', 'left', 'right', 'top', 'bottom']

    if view name is in ['front', 'back', 'left', 'right'], horizontal flip the image!
    if view name is in ['top', 'bottom'], vertical flip the image!
    """
    if view_name in ['front', 'back', 'left', 'right']:
        img = torch.flip(img, [1])
    elif view_name in ['top', 'bottom']:
        img = torch.flip(img, [0])
    return img

