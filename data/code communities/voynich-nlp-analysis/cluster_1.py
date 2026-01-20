# Cluster 1

def get_reducer(reducer: str) -> Union[PCA, umap.UMAP, pacmap.PaCMAP, pacmap.LocalMAP]:
    """
    Returns PCA, UMAP, or PaCMAP reducer based on argument.
    """
    if reducer == 'umap':
        return umap.UMAP()
    elif reducer == 'pacmap':
        return pacmap.PaCMAP(n_components=2, n_neighbors=5, MN_ratio=0.3, FP_ratio=1.5, random_state=42)
    elif reducer == 'localmap':
        return pacmap.LocalMAP(n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
    else:
        return PCA(n_components=2)

