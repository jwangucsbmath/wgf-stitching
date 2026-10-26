"""Step 5 — initial particle paths (analysis/00 sec. 4).

1. Match the first and last snapshots by exact OT on the squared Euclidean cost (Hungarian).
2. Pick N matched pairs (u_k, z_k) at random, with replacement if there are fewer than N pairs.
3. Interpolate linearly across the grid: x_{t_j,k} = (1 - eta_j) u_k + eta_j z_k, eta_j = (t_j - t_0)/(t_last - t_0)
   (the W2 geodesic between the two matched points).
4. Add one Gaussian offset per particle, std = jitter * data range per dimension, the same at every time.
The interior snapshots play no part here; they enter only through the NLL.
"""

import torch
from scipy.optimize import linear_sum_assignment


def ot_straight_lines(first, last, t_grid, N, gen=None, jitter=0.01):
    """first: (M0, D), last: (M1, D) snapshots; t_grid: (K,). Returns X (K, N, D)."""
    cost = torch.cdist(first, last).pow(2).detach().cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    rows, cols = torch.as_tensor(rows), torch.as_tensor(cols)
    n_pairs = len(rows)
    if N <= n_pairs:
        pick = torch.randperm(n_pairs, generator=gen)[:N]
    else:
        pick = torch.randint(n_pairs, (N,), generator=gen)
    u, z = first[rows[pick]], last[cols[pick]]  # (N, D)
    eta = ((t_grid - t_grid[0]) / (t_grid[-1] - t_grid[0]))[:, None, None]
    both = torch.cat([first, last])
    span = both.amax(0) - both.amin(0)
    offset = jitter * span * torch.randn(N, first.shape[1], generator=gen, dtype=first.dtype)
    return (1 - eta) * u + eta * z + offset
