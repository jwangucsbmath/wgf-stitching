"""Step 3 — the force g = grad dF/drho at the centres of one time slice (paper Eqs. 7, 14):

    g_k = c_H grad log rho(x_k) + c_V grad V(x_k) + c_W sum_j w_j grad W_s(x_k - x_j),
    W_s(r) = (W(r) + W(-r)) / 2.

W must be even, otherwise the force is not the gradient of an energy and the self-term
grad W(0) != 0 pushes every particle; symmetrising makes grad W_s(0) = 0 automatically.
X may be one slice (N, D) or a stack of slices (K, N, D); each slice only sees itself.
"""

import torch

from .kde import score_at_centres


def interaction_force(X, w, W_net):
    """sum_j w_j grad W_s(x_i - x_j) at every particle, W_s the even part of W_net   (Eq. 14).

    W_net maps differences (..., D) -> (...). The gradient is taken with create_graph=True,
    so a loss can backpropagate through grad W_theta (steps 6-7). Works on plain data and
    under torch.no_grad() as well.
    """
    track = torch.is_grad_enabled() and X.requires_grad
    with torch.enable_grad():
        diff = X[..., :, None, :] - X[..., None, :, :]  # r_ij = x_i - x_j, (..., N, N, D)
        if not track:  # simulator / tests / evaluation: plain data
            diff = diff.detach().requires_grad_(True)
        W_sym = 0.5 * (W_net(diff) + W_net(-diff))  # even by construction
        grad_W = torch.autograd.grad(W_sym.sum(), diff, create_graph=True)[0]
    return (w[:, None] * grad_W).sum(dim=-2)  # sum over j


def force(X, h, w, grad_V, c_H, c_V=1.0, c_W=0.0, W_net=None, loo=False):
    """g = c_H grad log rho + c_V grad V + c_W sum_j w_j grad W_s(x_k - x_j) at every centre   (Eq. 7).

    grad_V: callable (..., D) -> (..., D), or None for V = 0. c_H, c_V, c_W: floats or 0-d tensors.
    loo: leave-one-out score (Eq. 37) instead of the exact KDE score (Eq. 36).
    Returns the same shape as X.
    """
    g = c_H * score_at_centres(X, h, w, loo=loo)
    if grad_V is not None:
        g = g + c_V * grad_V(X)
    if W_net is not None and not (isinstance(c_W, (int, float)) and c_W == 0):
        g = g + c_W * interaction_force(X, w, W_net)
    return g


def grad_x(f, x):
    """grad_x f(x) for an energy f acting row-wise on (..., D), differentiable in f's parameters.

    Uses create_graph=True (grad V sits inside a loss that is differentiated again) and keeps the
    graph to x only when the caller tracks gradients; under no_grad (e.g. inside the simulator,
    step 6 prediction) it differentiates a detached copy.
    """
    track = torch.is_grad_enabled() and x.requires_grad
    with torch.enable_grad():
        x_ = x if track else x.detach().requires_grad_(True)
        return torch.autograd.grad(f(x_).sum(), x_, create_graph=True)[0]
