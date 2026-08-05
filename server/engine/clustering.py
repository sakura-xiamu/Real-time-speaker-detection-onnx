"""类似 sherpa-onnx FastClustering：L2 归一化 + 余弦距离 + complete linkage。"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def fast_clustering(
    embeddings: np.ndarray,
    num_clusters: int = -1,
    threshold: float = 0.78,
) -> np.ndarray:
    """
    Args:
        embeddings: (N, D)
        num_clusters: >0 时按指定簇数切割；否则按距离阈值
        threshold: 余弦距离阈值（1 - cosine_similarity），与 sherpa 默认量级一致

    Returns:
        labels: (N,) int32，从 0 开始
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings 须为二维，实际 shape={x.shape}")
    n = x.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if n == 1:
        return np.zeros((1,), dtype=np.int32)

    # L2 归一化
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms

    # 余弦距离 = 1 - cosine_similarity
    condensed = pdist(x, metric="cosine")
    condensed = np.clip(condensed, 0.0, None)

    z = linkage(condensed, method="complete")
    if num_clusters is not None and int(num_clusters) > 0:
        k = min(int(num_clusters), n)
        labels = fcluster(z, t=k, criterion="maxclust")
    else:
        labels = fcluster(z, t=float(threshold), criterion="distance")

    # fcluster 从 1 开始，转为从 0
    labels = labels.astype(np.int32) - 1
    return labels
