"""Run records (wgf/runs.py): checkpoints rebuild in a fresh process, folders never collide,
data are hash-checked, and the eval / plot commands never train and fail clearly on missing files."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from wgf import potentials as P
from wgf.model import FixedV, Stitching
from wgf.nn import MLP
from wgf.runs import MissingRunFile, Run, RunMismatch, load_model, open_eval, save_model
from wgf.train import fit

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]


def small_model(V, W=None, learn_c_H=True, loo=False):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(6, 8, 1, generator=g)
    torch.manual_seed(0)
    m = Stitching(torch.linspace(0, 1, 6), X, torch.tensor([0.3]), V, c_H=0.7, learn_c_H=learn_c_H, W=W, loo=loo)
    with torch.no_grad():  # make every parameter non-trivial
        for p in m.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    return m


MODELS = {
    "mlp": lambda: small_model(MLP(1)),
    "mlp_loo_fixed_c_H": lambda: small_model(MLP(1, (16, 16)), learn_c_H=False, loo=True),
    "fixed_V": lambda: small_model(FixedV(P.double_well_1d), learn_c_H=False),
    "with_W": lambda: small_model(MLP(1), W=MLP(1)),
}


@pytest.mark.parametrize("kind", list(MODELS))
def test_checkpoint_rebuilds_in_fresh_process(tmp_path, kind):
    m = MODELS[kind]()
    path = save_model(tmp_path / "m.pt", m)
    want = m.force().detach()
    torch.save(want, tmp_path / "force.pt")
    # a new interpreter in float32 by default: load_model must restore float64 and every tensor
    code = (f"import torch; from wgf.runs import load_model; m = load_model({str(path)!r}); "
            f"want = torch.load({str(tmp_path / 'force.pt')!r}); g = m.force().detach(); "
            f"assert m.X.dtype == torch.float64, m.X.dtype; assert torch.equal(g, want), (g - want).abs().max(); "
            f"print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    m2 = load_model(path)
    for (k, a), (k2, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert k == k2 and torch.equal(a, b)
    assert type(m2.raw_c_H) is type(m.raw_c_H) and m2.loo == m.loo
    ck = torch.load(path, weights_only=True)
    assert torch.equal(ck["readable"]["X"], m.X) and torch.allclose(ck["readable"]["h"], m.h)


def test_fixed_V_needs_a_named_potential(tmp_path):
    m = small_model(FixedV(lambda x: (x**2).sum(-1)))
    with pytest.raises(ValueError, match="wgf.potentials"):
        save_model(tmp_path / "m.pt", m)


def test_run_folders_never_collide_and_record_status(tmp_path):
    with Run.create("same", "exp", {"a": 1}, tmp_path) as r1:
        pass
    r2 = Run.create("same", "exp", {"a": 2}, tmp_path)
    assert r1.path != r2.path and r1.config == {"a": 1} and r2.config == {"a": 2}
    m = r1.manifest
    assert m["status"] == "finished" and m["seconds"] >= 0
    assert {"commit", "dirty", "packages", "python", "argv"} <= set(m["provenance"])
    with pytest.raises(ZeroDivisionError):
        with Run.create("bad", "exp", {}, tmp_path) as r3:
            1 / 0
    assert r3.manifest["status"].startswith("failed: ZeroDivisionError")
    with pytest.raises(RunMismatch, match="did not finish"):
        Run.open(r3.path)
    with pytest.raises(RunMismatch, match="'exp' run, not 'other'"):
        Run.open(r1.path, "other")
    with pytest.raises(MissingRunFile, match="not a run folder"):
        Run.open(tmp_path / "nothing")


def test_data_hash_and_missing_files(tmp_path):
    with Run.create("r", "exp", {}, tmp_path) as run:
        run.save_data({"test": {0.0: torch.ones(3, 1), 0.5: torch.zeros(3, 1)}})
        run.save_model("a", small_model(MLP(1)))
    assert torch.equal(run.load_data()["test"][0.0], torch.ones(3, 1))
    torch.save({"test": {0.0: torch.ones(3, 1), 0.5: torch.ones(3, 1)}}, run.path / "data.pt")
    with pytest.raises(RunMismatch, match="hash"):
        run.load_data()
    with pytest.raises(MissingRunFile, match=r"this run has: \['a'\]"):
        run.load_model("b")
    with pytest.raises(MissingRunFile, match="no evaluations"):
        open_eval(run.path)


def test_fit_history_records_lr_and_lambda_R():
    m = small_model(MLP(1))
    data = {0.0: torch.randn(20, 1), 1.0: torch.randn(20, 1)}
    hist = fit(m, data, epochs=10, lr=1e-2, lam=0.5, log=None)
    for k in ("loss", "nll", "R", "lamR", "lr", "h", "c_H"):
        assert len(hist[k]) == 10
    assert hist["lr"][0] == 1e-2 and hist["lr"][-1] < hist["lr"][0]
    assert all(abs(lr_ - 0.5 * R) < 1e-12 for lr_, R in zip(hist["lamR"], hist["R"]))
    assert all(abs(L - (n + lr_)) < 1e-9 for L, n, lr_ in zip(hist["loss"], hist["nll"], hist["lamR"]))


def _cmd(*args):
    return subprocess.run([sys.executable, "scripts/demo_split.py", *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True)


def _snapshot(folder):
    return {p: p.stat().st_mtime_ns for p in folder.rglob("*") if p.is_file()}


def test_split_commands_train_eval_plot(tmp_path):
    r = _cmd("train", "--epochs", 2, "--M", 200, "--N", 10, "--K", 6, "--runs-dir", tmp_path)
    assert r.returncode == 0, r.stderr
    (run,) = tmp_path.iterdir()
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["config"]["train"]["epochs"] == 2 and cfg["config"]["model"]["K"] == 6
    assert len(list((run / "models").glob("*.pt"))) == 6
    trained = _snapshot(run)

    r = _cmd("eval", run)
    assert r.returncode == 0, r.stderr
    (ev,) = (run / "evals").iterdir()
    ej = json.loads((ev / "eval.json").read_text())
    assert ej["status"] == "finished" and ej["config"]["data_regenerated_from_seed_matches"]
    metrics = json.loads((ev / "metrics.json").read_text())
    assert set(metrics["regimes"]["dense"]["scores"]) == {"ot_interp", "theta_kde", "theta_sde", "true_kde", "true_sde"}
    assert len(metrics["regimes"]["dense"]["scores"]["theta_sde"]["W2"]) == len(metrics["times"])

    (ev / "predictions.pt").unlink()  # plot needs metrics.json only
    r = _cmd("plot", run)
    assert r.returncode == 0, r.stderr
    assert (ev / "tables.md").exists() and (ev / "training_history.png").exists()
    after = _snapshot(run)
    assert all(after[p] == t for p, t in trained.items())  # eval and plot left the trained run untouched
    assert len(list(tmp_path.iterdir())) == 1  # and started no new run

    (run / "models" / "dense_V_star.pt").unlink()
    r = _cmd("eval", run)
    assert r.returncode == 1 and "missing checkpoint" in r.stderr and "Traceback" not in r.stderr
    assert len(list((run / "evals").iterdir())) == 1  # failed before making an eval folder
    (ev / "metrics.json").unlink()
    r = _cmd("plot", run)
    assert r.returncode == 1 and "metrics.json" in r.stderr
