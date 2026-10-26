"""Step 2 — KDE, score, NW velocity: checks with known answers."""

import math

import pytest
import torch

from wgf.kde import log_density, nw_velocity, sample, score, score_at_centres, silverman

torch.set_default_dtype(torch.float64)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def uniform(N):
    return torch.full((N,), 1.0 / N)


def test_density_integrates_to_one():
    centres = torch.tensor([[-1.0], [0.5], [2.0]])
    w = torch.tensor([0.2, 0.5, 0.3])
    y = torch.linspace(-10, 12, 40001)[:, None]
    integral = torch.trapezoid(log_density(y, centres, torch.tensor([0.4]), w).exp(), y[:, 0])
    assert abs(integral.item() - 1) < 1e-6


def test_score_matches_autograd():
    centres = torch.randn(40, 2, generator=gen())
    h, w = torch.tensor([0.3, 0.5]), torch.softmax(torch.randn(40, generator=gen(1)), 0)
    y = (torch.randn(25, 2, generator=gen(2)) * 1.5).requires_grad_(True)
    auto = torch.autograd.grad(log_density(y, centres, h, w).sum(), y)[0]
    assert torch.allclose(score(y, centres, h, w), auto, atol=1e-10)
    # at the centres (full score)
    c = centres.clone().requires_grad_(True)
    auto_c = torch.autograd.grad(log_density(c, centres, h, w).sum(), c)[0]
    assert torch.allclose(score_at_centres(centres, h, w, loo=False), auto_c, atol=1e-10)


def test_loo_score_matches_autograd_of_leave_one_out_density():
    centres = torch.randn(15, 2, generator=gen())
    h, w = torch.tensor([0.4, 0.6]), uniform(15)
    loo = score_at_centres(centres, h, w, loo=True)
    for j in range(15):
        keep = torch.arange(15) != j
        y = centres[j : j + 1].clone().requires_grad_(True)
        auto = torch.autograd.grad(log_density(y, centres[keep], h, w[keep]).sum(), y)[0]
        assert torch.allclose(loo[j], auto[0], atol=1e-10)


def test_score_worked_example_from_notes():
    """Notes sec. 9.2: particles 0, 0.5, 1.4, h = 0.4, at 0.5: full -0.64, LOO -1.83."""
    c, h, w = torch.tensor([[0.0], [0.5], [1.4]]), torch.tensor([0.4]), uniform(3)
    assert round(score_at_centres(c, h, w)[1, 0].item(), 2) == -0.64
    assert round(score_at_centres(c, h, w, loo=True)[1, 0].item(), 2) == -1.83


def test_gaussian_score_slope():
    """x ~ N(1, 0.7^2), h = 0.3: the KDE estimates the smoothed density, variance s^2 + h^2,
    so the score slope in x - m is -1/(s^2 + h^2) = -1.724, not -1/s^2 = -2.041.
    """
    m, s, h, N = 1.0, 0.7, 0.3, 20_000
    x = m + s * torch.randn(N, 1, generator=gen())
    w, hh = uniform(N), torch.tensor([h])
    sc = torch.cat([score(x[i : i + 2000], x, hh, w) for i in range(0, N, 2000)])  # chunks: N^2 memory
    bulk = (x - m).abs()[:, 0] < 1.5 * s
    u, v = (x - m)[bulk, 0], sc[bulk, 0]
    slope = ((u - u.mean()) * (v - v.mean())).sum() / ((u - u.mean()) ** 2).sum()
    target = -1 / (s**2 + h**2)
    assert abs(slope.item() / target - 1) < 0.05
    assert abs(slope.item() - (-1 / s**2)) > 0.15


def test_nw_constant_velocity_is_exact():
    centres = torch.randn(30, 2, generator=gen())
    v0 = torch.tensor([0.7, -1.3])
    y = torch.randn(50, 2, generator=gen(1)) * 3
    out = nw_velocity(y, centres, v0.expand(30, 2), torch.tensor([0.4, 0.4]), uniform(30))
    assert torch.allclose(out, v0.expand(50, 2), atol=1e-12)


def test_nw_velocity_at_centres_tends_to_particle_velocity():
    centres = torch.randn(30, 2, generator=gen())
    vel = torch.randn(30, 2, generator=gen(1))
    errs = [(nw_velocity(centres, centres, vel, torch.full((2,), h), uniform(30)) - vel).abs().max().item()
            for h in (0.3, 0.1, 0.03)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-4


def test_kde_continuity_equation():
    """Claim 1: d rho/dt + div(rho v_NW) = 0 for moving centres (1-D, finite differences)."""
    x0, xdot = torch.tensor([[-1.0], [0.3], [1.2]]), torch.tensor([[0.5], [-1.0], [2.0]])
    h, w = torch.tensor([0.5]), uniform(3)
    y = torch.linspace(-4, 5, 2001)[:, None]
    eps = 1e-5
    rho = lambda t: log_density(y, x0 + t * xdot, h, w).exp()
    drho_dt = (rho(eps) - rho(-eps)) / (2 * eps)
    div = torch.gradient(rho(0) * nw_velocity(y, x0, xdot, h, w)[:, 0], spacing=(y[1, 0] - y[0, 0]).item())[0]
    assert (drho_dt + div)[5:-5].abs().max().item() < 1e-4


def test_score_at_centres_batched_over_slices():
    X = torch.randn(4, 20, 2, generator=gen())
    h, w = torch.tensor([0.3, 0.5]), uniform(20)
    for loo in (False, True):
        batched = score_at_centres(X, h, w, loo=loo)
        assert torch.allclose(batched, torch.stack([score_at_centres(X[j], h, w, loo=loo) for j in range(4)]))


def test_silverman():
    x = torch.randn(500, 1, generator=gen())
    assert silverman(x).item() == pytest.approx(x.std().item() * (4 / (3 * 500)) ** 0.2)
    assert (4 / 3) ** 0.2 == pytest.approx(1.0592, abs=1e-4)  # the familiar 1.06 sigma n^{-1/5}
    assert silverman(torch.randn(500, 3, generator=gen())).shape == (3,)


def test_sample_respects_weights_and_dtype():
    c = torch.tensor([[-7.0], [7.0]])
    S = sample(c, torch.tensor([0.1]), torch.tensor([0.3, 0.7]), 1000, gen())
    assert abs((S > 0).double().mean().item() - 0.7) <= 1e-3
    c32 = torch.zeros(3, 1, dtype=torch.float32)
    assert sample(c32, torch.ones(1, dtype=torch.float32), torch.full((3,), 1 / 3, dtype=torch.float32), 4).dtype == torch.float32


def test_loo_needs_two_particles():
    with pytest.raises(ValueError):
        score_at_centres(torch.zeros(1, 1), torch.ones(1), torch.ones(1), loo=True)
