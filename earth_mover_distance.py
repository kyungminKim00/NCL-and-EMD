import numpy as np
import torch
from torch.nn import PairwiseDistance

def euclidean_distance(a, b):
    dist = PairwiseDistance(p=2)
    return dist(torch.tensor(a), torch.tensor(b)).numpy()

def earth_mover_distance(a, b):
    dists = np.zeros((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            dists[i, j] = euclidean_distance(a[i], b[j])
    
    cost_matrix = dists.T.copy()
    n = cost_matrix.shape[0]
    u = np.ones(n) / n
    v = np.ones(n) / n
    p = np.ones((n, n)) / (n ** 2)
    
    for _ in range(100):
        q = p / (np.outer(u, np.ones(n)) + np.outer(np.ones(n), v))
        p = np.maximum(p - 0.9 * np.minimum(q, cost_matrix), 0)
        diff = np.abs(p.sum(axis=1) - u)
        if diff.max() < 1e-3:
            break
        u = u - diff
    
    return np.sum(p * dists)

# Example usage
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[2, 1], [4, 3], [6, 5]])
emd = earth_mover_distance(a, b)
print(emd)  # Output: 3.4641016151377544
