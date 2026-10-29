"""Steps 5-6 — loss and optimiser.

Loss (Eq. 6, Eqs. 27-28):
    L = sum_s NLL_s + lambda R_vel,   NLL_s = -(1/M_s) sum_{i in s} log rho_{t_s}(y_i)
per snapshot then summed, as the paper's sum_t; pooled=True instead takes one mean over all cells
(each snapshot weighted by its cell count, as the reference code). R_vel is Eq. 13 (wgf.residual).

Optimiser: Adam, lr with cosine decay to 0.05 lr, full batch, one step per epoch:
    m <- b1 m + (1-b1) grad,  v <- b2 v + (1-b2) grad^2,  theta <- theta - lr_n mhat/(sqrt(vhat)+eps)
    lr_n = lr_min + (lr - lr_min)(1 + cos(pi n / epochs))/2

Resuming: fit can hand out its full training state every few epochs (Adam moments m, v and step
counts, the scheduler position, the completed epoch count, the history, the RNG states) and can
start from such a state, so an interrupted fit continues exactly where it stopped.
"""

import copy
import random
import time

import numpy as np
import torch

from .kde import log_density
from .residual import residual


def nll(model, data, pooled=False):
    """data: dict {t: (M_t, D)}. Per-snapshot mean NLL summed over snapshots (or one pooled mean)."""
    terms = [-log_density(Y, model.centres_at(t), model.h, model.w) for t, Y in data.items()]
    if pooled:
        return torch.cat(terms).mean()
    return sum(term.mean() for term in terms)


def loss(model, data, lam=1.0, scheme="trapezoid", pooled=False):
    """Returns (L, NLL, R_vel)."""
    data_term = nll(model, data, pooled)
    R = residual(model.X, model.t_grid, model.w, g=model.force(), scheme=scheme)
    return data_term + lam * R, data_term, R


class TrainingInterrupted(Exception):
    """fit stopped at `stop_at` before the planned last epoch, after handing out a checkpoint."""


def rng_state():
    """Global torch, numpy and python RNG states. fit draws no random numbers, so these only make a
    resumed process continue with the same streams for anything that runs after training."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": {"name": name, "keys": keys.tolist(), "pos": pos, "has_gauss": has_gauss, "cached": cached}}


def set_rng_state(st):
    torch.set_rng_state(st["torch"])
    random.setstate(st["python"])
    n = st["numpy"]
    np.random.set_state((n["name"], np.array(n["keys"], dtype=np.uint32), n["pos"], n["has_gauss"], n["cached"]))


def fit(model, data, epochs=2000, lr=5e-3, lam=1.0, scheme="trapezoid", pooled=False,
        log_every=100, log=print, track=None, state=None, checkpoint_every=0, on_checkpoint=None, stop_at=None):
    """Train in place. track: dict name -> callable(model) -> float, logged every epoch.

    Returns a history dict with one value per epoch of: loss, nll, R and lamR (= lam * R) at the
    parameters before that epoch's step; lr used by that step; h (mean over dimensions), c_H and the
    tracked quantities after it. Plus the run time in seconds, and `resumed_at_epoch` when resumed.

    Checkpoints and resuming (all optional; without them fit behaves exactly as before):
      on_checkpoint(train_state) is called after every `checkpoint_every` completed epochs, after
        the last epoch and at `stop_at`. train_state holds the optimiser and scheduler state dicts,
        the completed epoch count, the planned epochs, the hyperparameters, the history and the
        RNG states; the caller saves it together with the model's parameters.
      state: such a train_state, with `model` already holding the parameters saved with it. Training
        continues at state["epoch"] on the same schedule (`epochs` must be the planned total and
        lr, lam, scheme, pooled the same) and appends to its history.
      stop_at: stop after that many completed epochs (counted from 0, not from the resume point),
        checkpoint, and raise TrainingInterrupted, as if the process had been interrupted there.
    """
    track = dict(track or {})
    hyper = {"lr": lr, "lam": lam, "scheme": scheme, "pooled": pooled, "track": sorted(track)}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.05 * lr)
    keys = ["loss", "nll", "R", "lamR", "lr", "h", "c_H", *track]
    hist, start, seconds_before = {k: [] for k in keys}, 0, 0.0
    if state is not None:
        if state["epochs"] != epochs or state["hyper"] != hyper:
            raise ValueError(f"cannot continue this checkpoint exactly: it was made with epochs={state['epochs']}, "
                             f"{state['hyper']}, not epochs={epochs}, {hyper}; the learning-rate schedule "
                             "(cosine over the planned epochs) or the loss would differ")
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        hist = copy.deepcopy(state["history"])
        start, seconds_before = state["epoch"], hist.pop("seconds", 0.0)
        hist["resumed_at_epoch"] = hist.get("resumed_at_epoch", []) + [start]
        set_rng_state(state["rng"])
    t0 = time.time()

    def train_state(done):
        hist["seconds"] = seconds_before + time.time() - t0
        return {"format": 1, "epoch": done, "epochs": epochs, "hyper": hyper,
                "optimizer": copy.deepcopy(opt.state_dict()), "scheduler": sched.state_dict(),
                "history": copy.deepcopy(hist), "rng": rng_state()}

    for ep in range(start, epochs):
        opt.zero_grad()
        L, d, R = loss(model, data, lam, scheme, pooled)
        L.backward()
        lr_now = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
        values = [L.item(), d.item(), R.item(), lam * R.item(), lr_now, model.h.mean().item(), model.c_H.item()]
        values += [fn(model) for fn in track.values()]
        for k, v in zip(keys, values):
            hist[k].append(v)
        if log is not None and log_every and (ep % log_every == 0 or ep == epochs - 1):
            extra = "  ".join(f"{k} {hist[k][-1]:.4f}" for k in track)
            log(f"epoch {ep:5d}  NLL {d.item():8.4f}  R {R.item():9.5f}  h {hist['h'][-1]:.4f}  "
                f"c_H {hist['c_H'][-1]:.4f}  {extra}")
        done = ep + 1
        stopping = stop_at is not None and done >= stop_at and done < epochs
        if on_checkpoint is not None and ((checkpoint_every and done % checkpoint_every == 0)
                                          or done == epochs or stopping):
            on_checkpoint(train_state(done))
        if stopping:
            raise TrainingInterrupted(f"stopped after epoch {done} of {epochs}")
    hist["seconds"] = seconds_before + time.time() - t0
    return hist
