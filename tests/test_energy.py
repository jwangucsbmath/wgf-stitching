"""Step 3 — the force g: exact answers for known V and W."""

import torch

from wgf import potentials as P
from wgf.energy import force, interaction_force
from wgf.kde import score_at_centres

torch.set_default_dtype(torch.float64)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_quadratic_interaction_is_exact():
    """W(r) = |r|^2 / 2: sum_j w_j (x_i - x_j) = x_i - sum_j w_j x_j."""
    X = torch.randn(30, 2, generator=gen())
    w = torch.softmax(torch.randn(30, generator=gen(1)), 0)
    out = interaction_force(X, w, P.quadratic_interaction)
    assert torch.allclose(out, X - w @ X, atol=1e-12)


def test_non_even_w_is_symmetrised():
    """W(r) = r_1 + r_1^2: the odd part r_1 drops out, W_s = r_1^2, and the self-force is zero."""
    W_odd = lambda r: r[..., 0] + r[..., 0] ** 2
    W_even = lambda r: r[..., 0] ** 2
    X = torch.randn(25, 2, generator=gen())
    w = torch.full((25,), 1 / 25)
    assert torch.allclose(interaction_force(X, w, W_odd), interaction_force(X, w, W_even), atol=1e-12)
    single = torch.randn(1, 2, generator=gen(2))
    assert torch.all(interaction_force(single, torch.ones(1), W_odd) == 0)


def test_quadratic_v_plus_entropy_slope():
    """V = x^2/2, c_H = beta on a Gaussian cloud N(m, s^2): g = x + beta * score, and the KDE score
    has slope -1/(s^2 + h^2), so g has slope 1 - beta/(s^2 + h^2) in x - m."""
    m, s, h, beta, N = 1.0, 0.7, 0.3, 0.3, 5000
    X = m + s * torch.randn(N, 1, generator=gen())
    g = force(X, torch.tensor([h]), torch.full((N,), 1 / N), P.grad_quadratic, c_H=beta)
    bulk = (X - m).abs()[:, 0] < 1.5 * s
    u, v = (X - m)[bulk, 0], g[bulk, 0]
    slope = (((u - u.mean()) * (v - v.mean())).sum() / ((u - u.mean()) ** 2).sum()).item()
    assert abs(slope - (1 - beta / (s**2 + h**2))) < 0.03  # 0.483; unsmoothed would be 0.388


def test_force_gradcheck_double_backprop():
    """gradcheck of g w.r.t. X: second derivatives through grad V and grad W are correct."""
    V = lambda x: (x**2).sum(-1) / 2 + torch.sin(x).sum(-1)
    W = lambda r: torch.exp(-(r**2).sum(-1))
    h, w = torch.tensor([0.5, 0.7]), torch.full((5,), 0.2)
    gV = P.autograd_grad(V, create_graph=True)
    X = torch.randn(5, 2, generator=gen()).requires_grad_(True)
    for loo in (False, True):
        f = lambda X_: force(X_, h, w, gV, c_H=0.4, c_V=1.0, c_W=0.8, W_net=W, loo=loo)
        assert torch.autograd.gradcheck(f, (X,))


def test_force_batched_over_slices_and_without_grad():
    W = lambda r: torch.exp(-(r**2).sum(-1))
    h, w = torch.tensor([0.5, 0.7]), torch.full((10,), 0.1)
    X = torch.randn(3, 10, 2, generator=gen())
    batched = force(X, h, w, P.grad_quadratic, c_H=0.5, c_W=1.0, W_net=W)
    sliced = torch.stack([force(X[j], h, w, P.grad_quadratic, c_H=0.5, c_W=1.0, W_net=W) for j in range(3)])
    assert torch.allclose(batched, sliced, atol=1e-12)
    with torch.no_grad():
        assert torch.allclose(force(X, h, w, P.grad_quadratic, c_H=0.5, c_W=1.0, W_net=W), batched, atol=1e-12)


def test_force_parts():
    X = torch.randn(20, 2, generator=gen())
    h, w = torch.tensor([0.5, 0.5]), torch.full((20,), 0.05)
    assert torch.allclose(force(X, h, w, None, c_H=1.0), score_at_centres(X, h, w))
    assert torch.allclose(force(X, h, w, P.grad_quadratic, c_H=0.0, c_V=2.0), 2 * X)
