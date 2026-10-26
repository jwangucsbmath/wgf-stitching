"""[EVAL] Metrics comparing a learned flow with data or with the true system."""

import torch


def w1_1d(a, b):
    """Exact W1 between two 1-D samples of equal size: mean |a_(i) - b_(i)| of sorted values."""
    if a.shape != b.shape or a.shape[-1] != 1:
        raise ValueError("w1_1d needs two (M, 1) samples of equal size")
    return (a.sort(0).values - b.sort(0).values).abs().mean().item()


def w2_1d(a, b):
    """Exact W2 between two 1-D samples of equal size: sqrt(mean (a_(i) - b_(i))^2) of sorted values
    (the monotone coupling is optimal in 1-D)."""
    if a.shape != b.shape or a.shape[-1] != 1:
        raise ValueError("w2_1d needs two (M, 1) samples of equal size")
    return ((a.sort(0).values - b.sort(0).values) ** 2).mean().sqrt().item()


@torch.no_grad()
def potential_r2(V_learned, V_true, Y):
    """R^2 of centred potentials on the points Y (V is defined up to a constant).

        R^2 = 1 - sum (Vt~ - Vs~)^2 / sum Vs~^2,   V~ = V - mean_Y V
    Pass V_learned already scaled by c_V, so the metric uses the same F as the model.
    """
    vs, vt = V_true(Y), V_learned(Y)
    vs, vt = vs - vs.mean(), vt - vt.mean()
    return (1 - ((vt - vs) ** 2).sum() / (vs**2).sum()).item()
