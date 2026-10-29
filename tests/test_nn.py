"""Step 6 — the network V_theta, the metrics, and 6a: V_theta recovers x^2/2."""

import math
import sys
from pathlib import Path

import pytest
import torch

from wgf import potentials as P
from wgf.datasets import quadratic_snapshots
from wgf.energy import grad_x
from wgf.metrics import potential_r2, w1_1d, w2_1d
from wgf.nn import MLP
from wgf.train import fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "validation" / "ou_quadratic"))
from fit_nn import build_nn_model  # noqa: E402

torch.set_default_dtype(torch.float64)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_mlp_starts_at_zero_with_residual_skip():
    torch.manual_seed(0)
    net = MLP(2)
    x = torch.randn(10, 2, generator=gen())
    assert torch.all(net(x) == 0) and net(x).shape == (10,)
    assert net.head.bias is None
    # second hidden layer (64 -> 64) uses the skip: its output is input + SiLU(L input)
    h1 = torch.nn.functional.silu(net.layers[0](x))
    h2 = h1 + torch.nn.functional.silu(net.layers[1](h1))
    with torch.no_grad():
        net.head.weight.normal_(generator=gen(1))
    assert torch.allclose(net(x), net.head(h2).squeeze(-1))


def test_grad_x_is_differentiable_in_the_weights():
    torch.manual_seed(0)
    net = MLP(2)
    with torch.no_grad():
        net.head.weight.normal_(generator=gen(1))
    x = torch.randn(10, 2, generator=gen())
    (grad_x(net, x) ** 2).sum().backward()  # double backprop through create_graph=True
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    with torch.no_grad():  # also usable inside the no_grad simulator
        assert grad_x(net, x).shape == (10, 2)


def test_metrics():
    a = torch.randn(1000, 1, generator=gen())
    assert w1_1d(a, a + 0.7) == pytest.approx(0.7) and w2_1d(a, a + 0.7) == pytest.approx(0.7)
    assert w2_1d(a, 2 * a) == pytest.approx((a**2).mean().sqrt().item())
    Y = torch.randn(500, 1, generator=gen(2))
    assert potential_r2(lambda x: 3 + P.quadratic(x), P.quadratic, Y) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        w2_1d(a, a[:10])


def test_6a_network_recovers_quadratic():
    """Plan 6a: centred V_theta vs centred x^2/2 on the data range, R^2 > 0.95; held-out t = 1 as step 5."""
    data = quadratic_snapshots(gen(0))
    train = {t: Y for t, Y in data.items() if t != 1.0}
    m = build_nn_model(train)
    fit(m, train, epochs=2000, log=None)
    Y = torch.cat(list(train.values()))
    assert potential_r2(lambda x: m.c_V * m.V(x), P.quadratic, Y) > 0.95
    mean, var = m.moments_at(1.0)
    assert abs(mean.item() - 2 * math.exp(-1)) < 0.05
    assert abs(var.item() / (0.5 - 0.25 * math.exp(-2)) - 1) < 0.1
