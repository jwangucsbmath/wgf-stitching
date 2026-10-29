"""Checkpoints and resume (wgf/train.py, wgf/runs.py): an interrupted run continued in a fresh process
ends exactly where an uninterrupted run with the same planned epochs ends; model-only files cannot be
resumed and say why; checkpoint writes survive interruption; eval never regenerates data unless
asked with --verify-data."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from wgf import datastore
from wgf.model import Stitching
from wgf.nn import MLP
from wgf.runs import NoTrainingState, RunMismatch, load_checkpoint, load_model, save_model
from wgf.train import TrainingInterrupted, fit

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
DW = ROOT / "experiments" / "validation" / "double_well"
OU = ROOT / "experiments" / "validation" / "ou_quadratic"
sys.path[:0] = [str(DW), str(OU)]
import data as dw_data  # noqa: E402
import rollout as ou  # noqa: E402
import run as dw  # noqa: E402

TINY = ["--split", "coarse", "--epochs", 12, "--N", 8, "--K", 5, "--checkpoint-every", 5]
TAGS = [tag for tag, _, _ in dw.PLAN]
PER_EPOCH = ("loss", "nll", "R", "lamR", "lr", "h", "c_H")


def cli(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def only_run(folder):
    (run,) = [p for p in Path(folder).iterdir() if (p / "config.json").exists()]
    return run


def assert_same_training(a, b, tags):
    """Final parameters, Adam state, scheduler and per-epoch history of two run folders agree."""
    for tag in tags:
        ma, mb = load_model(a / "models" / f"{tag}.pt"), load_model(b / "models" / f"{tag}.pt")
        for (k, x), (_, y) in zip(ma.state_dict().items(), mb.state_dict().items()):
            torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-14, msg=f"{tag} {k}")
        sa = load_checkpoint(a / "checkpoints" / f"{tag}.pt")[1]
        sb = load_checkpoint(b / "checkpoints" / f"{tag}.pt")[1]
        assert sa["epoch"] == sb["epoch"] == sa["epochs"]
        assert sa["scheduler"]["last_epoch"] == sb["scheduler"]["last_epoch"] == sa["epochs"]
        for i, st in sa["optimizer"]["state"].items():
            for k in ("exp_avg", "exp_avg_sq", "step"):  # Adam m, v and step count
                torch.testing.assert_close(st[k], sb["optimizer"]["state"][i][k], rtol=1e-12, atol=1e-14)
        ha, hb = (json.loads((r / "history" / f"{tag}.json").read_text()) for r in (a, b))
        assert hb["lr"] == ha["lr"]  # the learning-rate schedule is the original one, epoch by epoch
        for k in PER_EPOCH:
            assert len(ha[k]) == len(hb[k]) == sa["epochs"]
            torch.testing.assert_close(torch.tensor(hb[k]), torch.tensor(ha[k]), rtol=1e-12, atol=1e-14)


# ── fit itself ──────────────────────────────────────────────────────────────────────────────────


def small():
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    return Stitching(torch.linspace(0, 1, 6), torch.randn(6, 8, 1, generator=g), torch.tensor([0.3]),
                     MLP(1, (16, 16)), c_H=0.7, learn_c_H=True)


DATA = {0.0: torch.randn(30, 1, generator=torch.Generator().manual_seed(1)),
        1.0: 1 + torch.randn(30, 1, generator=torch.Generator().manual_seed(2))}


def test_fit_stops_checkpoints_and_resumes_exactly(tmp_path):
    a = small()
    ha = fit(a, DATA, epochs=20, log=None)
    b, saved = small(), []
    with pytest.raises(TrainingInterrupted, match="after epoch 7 of 20"):
        fit(b, DATA, epochs=20, log=None, checkpoint_every=5,
            on_checkpoint=lambda st: saved.append(save_model(tmp_path / "ck.pt", b, st) and st["epoch"]), stop_at=7)
    assert saved == [5, 7]
    c, state = load_checkpoint(tmp_path / "ck.pt")
    hc = fit(c.train(), DATA, epochs=20, log=None, state=state)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values()))
    assert all(ha[k] == hc[k] for k in PER_EPOCH) and hc["resumed_at_epoch"] == [7]
    with pytest.raises(ValueError, match="cannot continue this checkpoint exactly"):
        fit(load_checkpoint(tmp_path / "ck.pt")[0], DATA, epochs=30, log=None, state=state)


def test_model_only_checkpoint_explains_why_it_cannot_resume(tmp_path):
    save_model(tmp_path / "model.pt", small())
    with pytest.raises(NoTrainingState, match="exact continuation is unavailable") as e:
        load_checkpoint(tmp_path / "model.pt")
    assert "Adam moments" in str(e.value) and "scheduler" in str(e.value)
    assert load_model(tmp_path / "model.pt").X.shape == (6, 8, 1)  # still fine for evaluation


def test_interrupted_checkpoint_write_keeps_previous_version(tmp_path, monkeypatch):
    m = small()
    save_model(tmp_path / "ck.pt", m, {"epoch": 5})
    real = torch.save

    def dies_halfway(obj, f, *a, **k):
        f.write(b"partial bytes")
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", dies_halfway)
    with torch.no_grad():
        m.X.add_(1.0)
    with pytest.raises(KeyboardInterrupt):
        save_model(tmp_path / "ck.pt", m, {"epoch": 10})
    monkeypatch.setattr(torch, "save", real)
    ck = torch.load(tmp_path / "ck.pt", weights_only=True)
    assert ck["train_state"]["epoch"] == 5 and torch.equal(ck["state_dict"]["X"], small().X)
    assert [p.name for p in tmp_path.iterdir()] == ["ck.pt"]  # no temporary file left behind


# ── whole runs, interrupted and resumed in fresh processes ─────────────────────────────────────


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Uninterrupted: six models, 12 planned epochs each."""
    out = tmp_path_factory.mktemp("uninterrupted")
    r = cli(DW / "run.py", "train", *TINY, "--runs-dir", out)
    assert r.returncode == 0, r.stderr
    return only_run(out)


def test_resume_in_fresh_processes_matches_uninterrupted(reference, tmp_path):
    r = cli(DW / "run.py", "train", *TINY, "--runs-dir", tmp_path, "--stop-after", 30)
    assert r.returncode == 3 and "resume" in r.stderr, r.stderr
    run = only_run(tmp_path)
    m = json.loads((run / "config.json").read_text())
    assert m["status"].startswith("interrupted")
    assert {t: p["status"] for t, p in m["progress"].items()} == {
        "dense_V_theta": "done", "dense_V_star": "done", "terminal_V_theta": "partial"}
    assert m["progress"]["terminal_V_theta"]["epoch"] == 6  # 30 = 12 + 12 + 6
    kept = {t: (run / "models" / f"{t}.pt").stat().st_mtime_ns for t in TAGS[:2]}

    r = cli(DW / "run.py", "resume", run, "--stop-after", 15)  # a second interruption: 6 -> 12, then 9 more
    assert r.returncode == 3, r.stderr
    m = json.loads((run / "config.json").read_text())
    assert m["progress"]["terminal_V_star"]["status"] == "partial" and m["progress"]["terminal_V_star"]["epoch"] == 9

    r = cli(DW / "run.py", "resume", run)
    assert r.returncode == 0, r.stderr
    m = json.loads((run / "config.json").read_text())
    assert m["status"] == "finished" and [s["command"] for s in m["sessions"]] == ["train", "resume", "resume"]
    assert all(p["status"] == "done" for p in m["progress"].values()) and set(m["progress"]) == set(TAGS)
    assert {t: (run / "models" / f"{t}.pt").stat().st_mtime_ns for t in TAGS[:2]} == kept  # retained, not retrained
    assert json.loads((run / "history" / "terminal_V_theta.json").read_text())["resumed_at_epoch"] == [6]
    assert json.loads((run / "history" / "terminal_V_star.json").read_text())["resumed_at_epoch"] == [9]
    assert_same_training(reference, run, TAGS)

    r = cli(DW / "run.py", "resume", run)
    assert r.returncode == 0 and "nothing to resume" in r.stdout


def test_resume_after_the_process_is_killed(tmp_path):
    """SIGKILL mid-training (no clean-up runs), then resume: same result as never stopping."""
    args = ["--split", "coarse", "--epochs", 400, "--N", 8, "--K", 5, "--checkpoint-every", 7]
    ref = tmp_path / "ref"
    assert cli(DW / "run.py", "train", *args, "--runs-dir", ref).returncode == 0
    killed = tmp_path / "killed"
    p = subprocess.Popen([sys.executable, str(DW / "run.py"), "train", *map(str, args), "--runs-dir", str(killed)],
                         cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 60
    while time.time() < deadline and p.poll() is None and not list(killed.glob("*/checkpoints/dense_V_theta.pt")):
        time.sleep(0.005)
    if p.poll() is not None:
        pytest.skip("training finished before it could be killed")
    p.send_signal(signal.SIGKILL)
    p.wait()
    run = only_run(killed)
    assert json.loads((run / "config.json").read_text())["status"] == "running"  # killed: no status written
    r = cli(DW / "run.py", "resume", run)
    assert r.returncode == 0, r.stderr
    assert_same_training(only_run(ref), run, TAGS)


def test_resume_refuses_a_model_only_file_for_an_unfinished_model(tmp_path):
    r = cli(DW / "run.py", "train", *TINY, "--runs-dir", tmp_path, "--stop-after", 30)
    assert r.returncode == 3
    run = only_run(tmp_path)
    ck = run / "checkpoints" / "terminal_V_theta.pt"
    save_model(run / "models" / "terminal_V_theta.pt", load_model(ck))  # parameters only
    ck.unlink()
    r = cli(DW / "run.py", "resume", run)
    assert r.returncode == 1 and "exact continuation is unavailable" in r.stderr and "Traceback" not in r.stderr


# ── evaluation: no data regeneration unless --verify-data ───────────────────────────────────────


def no_simulation(*a, **k):
    raise AssertionError("the dataset was regenerated")


def test_double_well_eval_regenerates_data_only_with_verify(reference, monkeypatch):
    monkeypatch.setattr(dw_data, "double_well_dataset", no_simulation)
    ev = dw.evaluate(argparse.Namespace(run=reference, eval_seed=0, verify_data=False))
    assert json.loads((ev.path / "eval.json").read_text())["config"]["data_check"]["regenerated"] is False
    with pytest.raises(AssertionError, match="regenerated"):
        dw.evaluate(argparse.Namespace(run=reference, eval_seed=0, verify_data=True))


def test_double_well_verify_data_reports_a_mismatch(reference, monkeypatch):
    real = dw_data.verify_dataset

    def with_wrong_seed(ds):
        meta = {**ds.meta, "seeds": {**ds.meta["seeds"], "data_generation": 1}}
        return real(datastore.Dataset(**{**ds.__dict__, "meta": meta}))

    monkeypatch.setattr(dw, "verify_dataset", with_wrong_seed)
    before = len(list((reference / "evals").glob("*")))
    with pytest.raises(RunMismatch, match="does not reproduce") as e:
        dw.evaluate(argparse.Namespace(run=reference, eval_seed=0, verify_data=True))
    assert "X: values differ" in str(e.value) and "no evaluation was run" in str(e.value)
    assert len(list((reference / "evals").glob("*"))) == before
    ev = dw.evaluate(argparse.Namespace(run=reference, eval_seed=0, verify_data=False))  # saved data still fine
    assert (ev.path / "metrics.json").exists()


def test_ou_eval_regenerates_data_only_with_verify(tmp_path, monkeypatch):
    assert cli(OU / "rollout.py", "train", "--epochs", 3, "--runs-dir", tmp_path).returncode == 0
    run = only_run(tmp_path)
    r = cli(OU / "rollout.py", "eval", run, "--verify-data")
    assert r.returncode == 0, r.stderr
    (ev,) = (run / "evals").iterdir()
    assert json.loads((ev / "eval.json").read_text())["config"]["data_check"]["match"] is True
    monkeypatch.setattr(ou, "make_data", no_simulation)
    ev = ou.evaluate(argparse.Namespace(run=run, eval_seed=0, verify_data=False))
    assert json.loads((ev.path / "eval.json").read_text())["config"]["data_check"]["regenerated"] is False
    monkeypatch.undo()
    cfg = json.loads((run / "config.json").read_text())
    cfg["config"]["seeds"]["data"] = 5  # the recorded seed no longer produces data.pt
    (run / "config.json").write_text(json.dumps(cfg))
    r = cli(OU / "rollout.py", "eval", run, "--verify-data")
    assert r.returncode == 1 and "does not reproduce" in r.stderr and "values differ" in r.stderr
