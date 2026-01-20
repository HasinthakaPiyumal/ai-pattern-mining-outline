# Cluster 15

def interX(L1, L2, is_return_points=False):
    """Calculate the intersections of batches of curves.
        Adapted from https://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections
    Args:
        L1: [batch_size, num_points, 2]
        L2: [batch_size, num_points, 2]
        is_return_points: bool. Whether to return the intersecting points.
    """
    batch_dim = L1.shape[0]
    collision_index = torch.zeros(batch_dim, dtype=torch.bool)
    if L1.numel() == 0 or L2.numel() == 0:
        return torch.empty((0, 2), device=L1.device) if is_return_points else False
    x1, y1 = (L1[..., 0], L1[..., 1])
    x2, y2 = (L2[..., 0], L2[..., 1])
    dx1, dy1 = (torch.diff(x1, dim=1), torch.diff(y1, dim=1))
    dx2, dy2 = (torch.diff(x2, dim=1), torch.diff(y2, dim=1))
    S1 = dx1 * y1[..., :-1] - dy1 * x1[..., :-1]
    S2 = dx2 * y2[..., :-1] - dy2 * x2[..., :-1]

    def D(x, y):
        return (x[..., :-1] - y) * (x[..., 1:] - y)
    C1 = D(dx1.unsqueeze(2) * y2.unsqueeze(1) - dy1.unsqueeze(2) * x2.unsqueeze(1), S1.unsqueeze(2)) < 0
    C2 = (D((y1.unsqueeze(2) * dx2.unsqueeze(1) - x1.unsqueeze(2) * dy2.unsqueeze(1)).transpose(1, 2), S2.unsqueeze(2)) < 0).transpose(1, 2)
    batch_indices, i, j = torch.where(C1 & C2)
    batch_indices_pruned = torch.sort(torch.unique(batch_indices))[0]
    collision_index[batch_indices_pruned] = True
    if is_return_points:
        if batch_indices.numel() == 0:
            return torch.empty((0, 2), device=L1.device)
        else:
            intersections = []
            for b in batch_indices.unique():
                L = dy2[b, j] * dx1[b, i] - dy1[b, i] * dx2[b, j]
                nonzero = L != 0
                i_nz, j_nz, L_nz = (i[nonzero], j[nonzero], L[nonzero])
                P = torch.stack(((dx2[b, j_nz] * S1[b, i_nz] - dx1[b, i_nz] * S2[b, j_nz]) / L_nz, (dy2[b, j_nz] * S1[b, i_nz] - dy1[b, i_nz] * S2[b, j_nz]) / L_nz), dim=-1)
                intersections.append(P)
            return torch.cat(intersections, dim=0)
    else:
        return collision_index

def D(x, y):
    return (x[..., :-1] - y) * (x[..., 1:] - y)

