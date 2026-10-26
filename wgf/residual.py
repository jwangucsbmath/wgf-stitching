"""Step 4 — the discretised velocity residual (paper Eq. 13 / App. C).

For a path X[K, N, D] on t_grid, with dx_jk = x_{j+1,k} - x_{j,k} and dt_j = t_{j+1} - t_j:

    R_vel = sum_{j=0}^{K-2} (1 / dt_j) sum_k w_k || dx_jk + dt_j gbar_jk ||^2,
    gbar_j = g_j                   (euler)
    gbar_j = (g_j + g_{j+1}) / 2   (trapezoid, default)

This is the Riemann sum of int sum_k w_k ||xdot_k + g(x_k)||^2 dt, i.e. the velocity residual
with rho replaced by the particle measure (Eqs. 10-12) and v(x_k) = xdot_k (Claim 2); the 1/dt
comes from dt ||a/dt + b||^2 = ||a + dt b||^2 / dt.

R_vel scores a whole path that already exists; it never steps forward. Do not feed SDE sample
paths into it: Brownian increments are O(sqrt(dt)), so R_vel ~ 2 beta D (K - 1) grows as dt -> 0.
"""

import torch


def residual(X, t_grid, w, force_fn=None, scheme="trapezoid", g=None):
    """R_vel of the path X (K, N, D) on t_grid (K,), weights w (N,).

    force_fn(X_slice, j) -> g at the centres of slice j, shape (N, D). Alternatively pass the
    forces of all slices at once as g (K, N, D), e.g. from a batched force evaluation.
    """
    if g is None:
        if force_fn is None:
            raise ValueError("pass force_fn or g")
        g = torch.stack([force_fn(X[j], j) for j in range(X.shape[0])])
    if scheme == "euler":
        g_bar = g[:-1]
    elif scheme == "trapezoid":
        g_bar = 0.5 * (g[:-1] + g[1:])
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    dt = (t_grid[1:] - t_grid[:-1])[:, None, None]
    dX = X[1:] - X[:-1]
    per_particle = ((dX + dt * g_bar) ** 2).sum(-1)  # (K-1, N)
    return ((per_particle @ w) / dt[:, 0, 0]).sum()
