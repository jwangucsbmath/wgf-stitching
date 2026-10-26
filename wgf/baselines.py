"""[EVAL] No-physics reference: OT (displacement) interpolation between training snapshots.

In 1-D the W2 geodesic between two distributions averages their quantile functions:
    Q_t(u) = (1 - a) Q_A(u) + a Q_B(u),   a = (t - t_A) / (t_B - t_A).
A test time is predicted from its two neighbouring training snapshots. If a learned V_theta does
not beat this, the test has not shown any learned dynamics.
"""

import torch


def ot_interpolate_1d(A, B, a, M):
    """M points of the 1-D OT interpolant between samples A (n, 1) and B (m, 1) at fraction a;
    the sizes may differ (quantiles at levels (i + 1/2) / M)."""
    u = (torch.arange(M, dtype=A.dtype) + 0.5) / M
    return ((1 - a) * torch.quantile(A[:, 0], u) + a * torch.quantile(B[:, 0], u))[:, None]


def ot_interpolate_snapshots(train, t, M):
    """Predict time t from the training snapshots that bracket it."""
    times = sorted(train)
    if not times[0] <= t <= times[-1]:
        raise ValueError(f"time {t} is outside the training times")
    tb = next(s for s in times if s >= t)
    ta = max(s for s in times if s <= t)
    a = 0.0 if tb == ta else (t - ta) / (tb - ta)
    return ot_interpolate_1d(train[ta], train[tb], a, M)
