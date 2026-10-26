"""Step 6 — the potential network V_theta.

MLP D -> 64 -> 64 -> 1: Linear -> SiLU, with a residual skip y <- y + SiLU(L y) whenever input and
output widths match, then a final Linear initialised to zero, so V_theta = 0 at the start (exactly
like V_a with a = 0 in step 5). SiLU is smooth, which matters because V_theta is differentiated in
x inside the loss and then again in theta. grad_x V_theta comes from wgf.energy.grad_x
(torch.autograd.grad(V.sum(), x, create_graph=True)). c_V stays fixed at 1: the network carries the scale.
"""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in, hidden=(64, 64)):
        super().__init__()
        widths = (d_in,) + tuple(hidden)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        # no bias in the head: a constant shifts V but never the force, so it would get no gradient
        self.head = nn.Linear(widths[-1], 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, x):
        """(..., D) -> (...)."""
        for L in self.layers:
            y = F.silu(L(x))
            x = x + y if y.shape == x.shape else y
        return self.head(x).squeeze(-1)
