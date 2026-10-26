"""Step 5 — initialisation, model, loss and training with V_a = a x^2/2 (no network)."""

import math

import pytest
import torch

from wgf.datasets import quadratic_snapshots
from wgf.init import ot_straight_lines
from wgf.kde import silverman
from wgf.model import QuadraticV, Stitching, inv_softplus
from wgf.train import fit, loss, nll

torch.set_default_dtype(torch.float64)
BETA, M0, S0 = 0.5, 2.0, 0.5


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.fixture(scope="module")
def data():
    return quadratic_snapshots(gen(0))


def build(train, K=41, N=100, seed=0, **kw):
    t_grid = torch.linspace(0.0, 2.0, K)
    X = ot_straight_lines(train[0.0], train[2.0], t_grid, N, gen(seed))
    return Stitching(t_grid, X, silverman(train[0.0]), QuadraticV(), c_H=BETA, **kw)


# ── init ────────────────────────────────────────────────────────────────────────────


def test_init_straight_lines_and_monotone_ot():
    first = torch.randn(200, 1, generator=gen())
    last = 3 + 2 * torch.randn(200, 1, generator=gen(1))
    t = torch.linspace(0, 1, 11)
    X = ot_straight_lines(first, last, t, 50, gen(2), jitter=0.0)
    inc = X[1:] - X[:-1]
    assert torch.allclose(inc, inc[:1].expand_as(inc), atol=1e-12)  # straight, constant speed
    # 1-D OT is the monotone matching: ordering at t0 is preserved at the end
    order = X[0, :, 0].argsort()
    assert torch.all(X[-1, order, 0].diff() >= 0)
    assert set(X[0, :, 0].tolist()) <= set(first[:, 0].tolist())


def test_init_with_replacement_and_jitter():
    first, last = torch.randn(10, 2, generator=gen()), torch.randn(10, 2, generator=gen(1))
    X = ot_straight_lines(first, last, torch.linspace(0, 1, 5), 30, gen(2))
    assert X.shape == (5, 30, 2)  # 30 > 10 pairs: drawn with replacement
    plain = ot_straight_lines(first, last, torch.linspace(0, 1, 5), 30, gen(2), jitter=0.0)
    offset = X - plain
    assert offset.abs().max() > 0
    assert torch.allclose(offset, offset[:1].expand_as(offset), atol=1e-12)  # same offset at every time
    assert torch.equal(X, ot_straight_lines(first, last, torch.linspace(0, 1, 5), 30, gen(2)))  # seeded


# ── model ───────────────────────────────────────────────────────────────────────────


def test_inv_softplus_round_trip():
    for y in (1e-6, 0.05, 0.5, 3.0, 40.0):
        assert torch.nn.functional.softplus(inv_softplus(y)).item() == pytest.approx(y, rel=1e-10)


def test_centres_at_grid_nodes_and_outside(data):
    m = build(data, K=41)
    assert m.centres_at(0.5) is not None and torch.equal(m.centres_at(0.5), m.X[10])  # 0.5 is node 10
    assert torch.equal(m.centres_at(2.0), m.X[-1])
    with pytest.raises(ValueError):
        m.centres_at(2.5)


def test_moments_at_adds_h_squared(data):
    m = build(data)
    mean, var = m.moments_at(1.0)
    c = m.centres_at(1.0).detach()
    assert mean.item() == pytest.approx(c.mean().item())
    assert var.item() == pytest.approx(c.var(unbiased=False).item() + m.h.item() ** 2)


def test_every_trainable_parameter_gets_gradient(data):
    m = build(data, learn_c_H=True)
    with torch.no_grad():
        m.V.a.fill_(0.3)
    loss(m, data)[0].backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_nll_per_snapshot_vs_pooled(data):
    m = build(data)
    per = nll(m, data).item()
    pooled = nll(m, data, pooled=True).item()
    assert per == pytest.approx(len(data) * pooled, rel=1e-12)  # equal snapshot sizes


# ── training ────────────────────────────────────────────────────────────────────────


def test_lambda_zero_leaves_a_exactly_zero(data):
    """a enters only through R_vel; with lambda = 0 it gets zero gradient and must stay 0."""
    train = {t: Y for t, Y in data.items() if t != 1.0}
    m = build(train)
    fit(m, train, epochs=50, lam=0.0, log=None)
    assert m.V.a.item() == 0.0


def test_fit_recovers_a_and_held_out_moments(data):
    """Plan targets: |a - 1| < 0.1; at held-out t = 1, KDE mean within 0.05 of m(1) = 0.7358 and
    variance within 10% of s(1)^2 = 0.4662."""
    train = {t: Y for t, Y in data.items() if t != 1.0}
    m = build(train)
    fit(m, train, epochs=2000, log=None)
    mean, var = m.moments_at(1.0)
    assert abs(m.V.a.item() - 1) < 0.1
    assert abs(mean.item() - M0 * math.exp(-1)) < 0.05
    assert abs(var.item() / (BETA + (S0**2 - BETA) * math.exp(-2)) - 1) < 0.1
