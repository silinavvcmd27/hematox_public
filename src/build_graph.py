# пространственный граф патчей: узлы = патчи, рёбра = соседство (kNN или радиус)
# граф строится отдельно для каждого слайда
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_edge_index(coords, mode="knn", k=6, radius=None):
    n = len(coords)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    if mode == "knn":
        kk = min(k + 1, n)   # +1 потому что первый сосед — сам узел
        _, idx = NearestNeighbors(n_neighbors=kk).fit(coords).kneighbors(coords)
        idx = idx[:, 1:]
        src = np.repeat(np.arange(n), idx.shape[1])
        dst = idx.reshape(-1)
    elif mode == "radius":
        neigh = NearestNeighbors(radius=radius).fit(coords).radius_neighbors(
            coords, return_distance=False)
        src = np.repeat(np.arange(n), [len(v) for v in neigh])
        dst = np.concatenate(neigh).astype(np.int64) if len(neigh) else np.empty(0, np.int64)
        keep = src != dst        
        src, dst = src[keep], dst[keep]
    else:
        raise ValueError(mode)

    edge = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge = torch.cat([edge, edge.flip(0)], dim=1)   # делаем неориентированным
    return torch.unique(edge, dim=1)


def slide_to_data(npz_path, cfg_graph):
    from torch_geometric.data import Data

    d = np.load(npz_path, allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.long)
    coords = d["coords"].astype(float)
    if not len(X) == len(y) == len(coords):
        raise SystemExit(f"{npz_path}: рассинхрон, X={len(X)} y={len(y)} coords={len(coords)}")

    radius = None
    if cfg_graph["mode"] == "radius":
        if len(coords) > 1:
            dists, _ = NearestNeighbors(n_neighbors=2).fit(coords).kneighbors(coords)
            step = np.median(dists[:, 1])
        else:
            step = 1.0
        radius = cfg_graph["radius_patches"] * step

    edge_index = build_edge_index(coords, cfg_graph["mode"], cfg_graph["k"], radius)
    data = Data(x=X, y=y, edge_index=edge_index)
    data.coords = torch.tensor(coords, dtype=torch.float32)
    return data