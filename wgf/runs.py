"""Run records: one folder per experiment run, so no result ever overwrites another.

    <experiment>/runs/<YYYYmmdd-HHMMSS>_<name>/
        config.json              config, seeds, data hash, source commit, package versions, runtime, status
        source.diff              uncommitted changes against that commit (only when the tree was dirty)
        data.pt                  the run's data as tensors, when it has no saved dataset (wgf/datastore.py);
                                 runs on a saved dataset record its ID, split and hashes in config.json instead
        models/<tag>.pt          one finished model: spec + state_dict + readable X, h, c_H, w, t_grid
                                 (model-only: enough for eval, not for continuing training)
        history/<tag>.json       fit() history of a finished model, one value per epoch
        checkpoints/<tag>.pt     training checkpoint, rewritten every `checkpoint_every` epochs: the model
                                 file's contents plus train_state (Adam m, v and step counts, scheduler,
                                 completed epochs, history so far, RNG states); `resume` continues from it
        evals/<YYYYmmdd-HHMMSS>_seed<k>/
            eval.json            eval settings and seeds, source commit, runtime, status
            predictions.pt       every predicted cloud, by method and time
            metrics.json         W1/W2 per method and time, force errors, ...
            tables.md, *.png     written by `plot` from metrics.json alone

Each experiment script has four commands (see `cli`), and none of them calls another:
    train   creates a run: data, models, histories, checkpoints
    resume  continues an interrupted run: keeps finished models, continues the unfinished one from its
            checkpoint on the same learning-rate schedule, starts the ones not begun
    eval    loads a finished run's data and models and writes a new evals/ folder; never trains
    plot    reads one evals/ folder's metrics.json and writes tables and figures; never trains or evaluates

Every file is written to a temporary name and then renamed over the target, so an interrupted write
leaves the previous version intact. config.json records each train / resume session (command,
provenance, time, status) and each model's progress.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn as nn

from . import potentials as P
from .model import FixedV, QuadraticV, Stitching
from .nn import MLP
from .train import TrainingInterrupted, fit

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FORMAT = 1


# ── provenance ──────────────────────────────────────────────────────────────────────────────────


def _git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode in (0, 1) else ""  # diff --no-index exits 1 when files differ


def provenance():
    """Source commit, whether the tree was dirty, interpreter, packages, torch settings, command line."""
    status = _git("status", "--porcelain").strip()
    packages = {}
    for p in ("torch", "numpy", "scipy", "matplotlib"):
        try:
            packages[p] = version(p)
        except PackageNotFoundError:
            packages[p] = None
    return {
        "commit": _git("rev-parse", "HEAD").strip() or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip() or None,
        "dirty": bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "argv": sys.argv,
    }


def _source_diff():
    """`git diff HEAD` plus every untracked, non-ignored file as an added-file patch."""
    parts = [_git("diff", "HEAD")]
    for f in _git("ls-files", "--others", "--exclude-standard").split():
        parts.append(_git("diff", "--no-index", "--", "/dev/null", f))
    return "".join(parts)


# ── files ───────────────────────────────────────────────────────────────────────────────────────


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write(path, write):
    """write(f) into a temporary file next to `path`, fsync it, then rename it over `path`: a reader
    (or a process killed mid-write) sees either the old file or the complete new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def write_json(path, obj):
    text = (json.dumps(obj, indent=2, default=str) + "\n").encode()
    return atomic_write(path, lambda f: f.write(text))


def read_json(path):
    return json.loads(require(path).read_text())


class MissingRunFile(FileNotFoundError):
    """A file a command needs is not in the run folder."""


class RunMismatch(ValueError):
    """The run folder exists but does not fit the command (other experiment, unfinished, altered data)."""


class NoTrainingState(RunMismatch):
    """A model-only file was asked to continue training."""


def require(path, hint=""):
    """The path if it exists, else a MissingRunFile naming it (and what to run, if hint is given)."""
    path = Path(path)
    if not path.exists():
        raise MissingRunFile(f"missing required file: {path}" + (f"\n  {hint}" if hint else ""))
    return path


def data_hash(obj):
    """sha256 over a nested dict of tensors (keys, dtypes, shapes and values), order-independent."""
    h = hashlib.sha256()

    def walk(o, key):
        if isinstance(o, dict):
            for k in sorted(o, key=repr):
                walk(o[k], f"{key}/{k!r}")
        elif torch.is_tensor(o):
            h.update(f"{key}:{o.dtype}:{tuple(o.shape)}".encode())
            h.update(o.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(f"{key}={o!r}".encode())

    walk(obj, "")
    return h.hexdigest()


def _new_dir(parent, name):
    """parent/<timestamp>_<name>, with -1, -2, ... appended if that folder already exists."""
    base = Path(parent) / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    path, i = base, 0
    while True:
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            i += 1
            path = base.with_name(f"{base.name}-{i}")


class _Record:
    """A folder with a JSON manifest that records status and runtime. Use as a context manager:
    the manifest says "running" while inside, then "finished", "interrupted: ..." or "failed: <error>".
    Runtime adds up over sessions (a session killed outright records none)."""

    MANIFEST = None

    def __init__(self, path):
        self.path = Path(path)

    @property
    def manifest(self):
        return read_json(self.path / self.MANIFEST)

    def _update(self, **fields):
        m = self.manifest
        m.update(fields)
        write_json(self.path / self.MANIFEST, m)

    def __enter__(self):
        self._t0 = time.time()
        self._prior = self.manifest.get("seconds") or 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            status = "finished"
        elif isinstance(exc, (TrainingInterrupted, KeyboardInterrupt)):
            status = f"interrupted: {exc or exc_type.__name__}"
            exc.run_path = self.path  # so the command line can name the folder to resume
        else:
            status = f"failed: {exc_type.__name__}: {exc}"
        took = round(time.time() - self._t0, 3)
        m = self.manifest
        m.update(status=status, ended=_now(), seconds=round(self._prior + took, 3))
        if m.get("sessions"):
            m["sessions"][-1].update(status=status, ended=m["ended"], seconds=took)
        write_json(self.path / self.MANIFEST, m)
        return False

    def save_json(self, rel, obj):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, obj)
        return p

    def load_json(self, rel, hint=""):
        return read_json(require(self.path / rel, hint))

    def save_tensors(self, rel, obj):
        return atomic_write(self.path / rel, lambda f: torch.save(obj, f))

    def load_tensors(self, rel, hint=""):
        return torch.load(require(self.path / rel, hint), weights_only=True)


class Run(_Record):
    MANIFEST = "config.json"

    @classmethod
    def create(cls, name, experiment, config, root=RUNS):
        path = _new_dir(root, name)
        prov = provenance()
        write_json(path / cls.MANIFEST, {"format": FORMAT, "experiment": experiment, "name": name,
                                          "config": config, "provenance": prov, "status": "running",
                                          "started": _now(), "progress": {},
                                          "sessions": [{"command": "train", "started": _now(), "provenance": prov}]})
        if prov["dirty"]:
            (path / "source.diff").write_text(_source_diff())
        return cls(path)

    def begin_resume(self):
        """Record a resume session (with this process's provenance) and mark the run running again."""
        prov, m = provenance(), self.manifest
        first = m["provenance"]
        if (prov["commit"], prov["dirty"]) != (first["commit"], first["dirty"]) or prov["dirty"]:
            print(f"note: training started at commit {first['commit']} (dirty {first['dirty']}), resuming at "
                  f"{prov['commit']} (dirty {prov['dirty']}); exact continuation assumes the same training code")
        sessions = m.get("sessions") or [{"command": "train", "started": m["started"], "provenance": first}]
        sessions.append({"command": "resume", "started": _now(), "provenance": prov})
        if prov["dirty"]:
            (self.path / f"source.session{len(sessions) - 1}.diff").write_text(_source_diff())
        self._update(sessions=sessions, status="running")
        return self

    @classmethod
    def open(cls, path, experiment=None, finished=True):
        """An existing run. Checks it is a run folder, made by `experiment`, whose training finished."""
        path = Path(path)
        if not (path / cls.MANIFEST).exists():
            raise MissingRunFile(f"{path} is not a run folder (no {cls.MANIFEST}); "
                                    f"create one with the script's `train` command")
        run = cls(path)
        m = run.manifest
        if experiment is not None and m["experiment"] != experiment:
            raise RunMismatch(f"{path} is a {m['experiment']!r} run, not {experiment!r}")
        if finished and m.get("status") != "finished":
            raise RunMismatch(f"{path}: training did not finish (status {m.get('status')!r}); "
                              f"continue it with the script's `resume` command")
        return run

    @property
    def config(self):
        return self.manifest["config"]

    # data: saved once, its hash goes into config.json and is checked on every load
    def save_data(self, data):
        self.save_tensors("data.pt", data)
        self._update(data_sha256=data_hash(data))

    def load_data(self):
        data = self.load_tensors("data.pt", "the run folder is incomplete; retrain with `train`")
        if data_hash(data) != self.manifest["data_sha256"]:
            raise RunMismatch(f"{self.path / 'data.pt'} does not match the hash in config.json")
        return data

    def save_model(self, tag, model):
        return save_model(self.path / "models" / f"{tag}.pt", model)

    def load_model(self, tag):
        p = self.path / "models" / f"{tag}.pt"
        if not p.exists():
            raise MissingRunFile(f"missing checkpoint {p}; this run has: {self.model_tags() or 'none'}")
        return load_model(p)

    def model_tags(self):
        return sorted(p.stem for p in (self.path / "models").glob("*.pt"))

    def _progress(self, tag, epoch, epochs):
        m = self.manifest
        m.setdefault("progress", {})[tag] = {"epoch": epoch, "epochs": epochs,
                                             "status": "done" if epoch >= epochs else "partial"}
        write_json(self.path / self.MANIFEST, m)

    def model_done(self, tag, epochs):
        """True when models/<tag>.pt holds the finished model: marked done in config.json, or (runs from
        before checkpoints existed) its history covers all planned epochs."""
        if not (self.path / "models" / f"{tag}.pt").exists():
            return False
        prog = self.manifest.get("progress", {}).get(tag)
        if prog is not None:
            return prog["status"] == "done"
        h = self.path / "history" / f"{tag}.json"
        return h.exists() and len(read_json(h)["loss"]) == epochs

    def train_or_resume(self, tag, build, data, epochs, fit_kw, checkpoint_every, budget=None):
        """Train model `tag` with periodic checkpoints, or continue it from checkpoints/<tag>.pt, or
        reuse it when it is already finished. build() makes the fresh model (seeded, so a model that
        never started is rebuilt identically). budget: epochs this call may train before stopping as if
        interrupted (None: no limit). Returns (model, history, epochs trained by this call)."""
        model_p, ck_p = self.path / "models" / f"{tag}.pt", self.path / "checkpoints" / f"{tag}.pt"
        if self.model_done(tag, epochs):
            return load_model(model_p), self.load_history(tag), 0
        if ck_p.exists():
            model, state = load_checkpoint(ck_p)
            model.train()
        elif model_p.exists():
            raise NoTrainingState(model_only_message(model_p))
        else:
            model, state = build(), None
        start = state["epoch"] if state else 0
        if budget is not None and budget <= 0:
            raise TrainingInterrupted(f"stopped before {tag} (epoch {start} of {epochs})")

        def checkpoint(train_state):
            save_model(ck_p, model, train_state)
            self._progress(tag, train_state["epoch"], epochs)

        hist = fit(model, data, epochs=epochs, log_every=0, state=state, checkpoint_every=checkpoint_every,
                   on_checkpoint=checkpoint, stop_at=None if budget is None else start + budget, **fit_kw)
        save_model(model_p, model)
        self.save_history(tag, hist)
        self._progress(tag, epochs, epochs)
        return model, hist, epochs - start

    def save_history(self, tag, hist):
        return self.save_json(f"history/{tag}.json", hist)

    def load_history(self, tag):
        return self.load_json(f"history/{tag}.json")

    def new_eval(self, seed, config):
        path = _new_dir(self.path / "evals", f"seed{seed}")
        prov = provenance()
        write_json(path / Eval.MANIFEST, {"format": FORMAT, "run": self.path.name, "eval_seed": seed,
                                          "config": config, "provenance": prov, "status": "running",
                                          "started": _now()})
        if prov["dirty"]:
            (path / "source.diff").write_text(_source_diff())
        return Eval(path)

    def evals(self):
        return sorted(p for p in (self.path / "evals").glob("*") if (p / Eval.MANIFEST).exists())


class Eval(_Record):
    MANIFEST = "eval.json"


def open_eval(run_path, eval_name=None):
    """A run's evaluation folder: the one named, else the newest. Errors clearly if there is none."""
    run = Run.open(run_path, finished=False)
    if eval_name:
        path = Path(eval_name) if Path(eval_name).is_absolute() else run.path / "evals" / eval_name
        require(path / Eval.MANIFEST)
        return Eval(path)
    found = run.evals()
    if not found:
        raise MissingRunFile(f"{run.path} has no evaluations; run the script's `eval` command first")
    return Eval(found[-1])


# ── models ──────────────────────────────────────────────────────────────────────────────────────


def _energy_spec(E):
    if E is None:
        return None
    if isinstance(E, MLP):
        return {"type": "mlp", "d_in": E.d_in, "hidden": list(E.hidden)}
    if isinstance(E, QuadraticV):
        return {"type": "quadratic"}
    if isinstance(E, FixedV):
        name = getattr(E.fn, "__name__", None)
        if getattr(P, str(name), None) is not E.fn:
            raise ValueError("FixedV can only be saved when its fn is a function of wgf.potentials")
        return {"type": "fixed", "fn": name}
    raise TypeError(f"no saved form for energy {type(E).__name__}")


def _build_energy(spec):
    if spec is None:
        return None
    if spec["type"] == "mlp":
        return MLP(spec["d_in"], tuple(spec["hidden"]))
    if spec["type"] == "quadratic":
        return QuadraticV()
    if spec["type"] == "fixed":
        return FixedV(getattr(P, spec["fn"]))
    raise ValueError(f"unknown energy type {spec['type']!r}")


def model_spec(model):
    """Everything besides the state_dict needed to rebuild a Stitching model."""
    K, N, D = model.X.shape
    return {"class": "Stitching", "K": K, "N": N, "D": D, "dtype": str(model.X.dtype).split(".")[-1],
            "learn_c_H": isinstance(model.raw_c_H, nn.Parameter), "c_V": float(model.c_V),
            "loo": bool(model.loo), "V": _energy_spec(model.V), "W": _energy_spec(model.W)}


@torch.no_grad()
def save_model(path, model, train_state=None):
    """Spec + state_dict (X, raw h, raw c_H, V_theta, W_theta and raw c_W when present, w, t_grid),
    plus the same quantities in their natural units for reading without rebuilding. With train_state
    (from wgf.train.fit) it is a training checkpoint that `load_checkpoint` can continue from.
    Written atomically."""
    readable = {"X": model.X.detach().clone(), "h": model.h.detach().clone(),
                "c_H": model.c_H.detach().clone(), "c_V": float(model.c_V),
                "c_W": model.c_W.detach().clone() if model.W is not None else None,
                "w": model.w.clone(), "t_grid": model.t_grid.clone()}
    payload = {"format": FORMAT, "spec": model_spec(model),
               "state_dict": {k: v.clone() for k, v in model.state_dict().items()}, "readable": readable}
    if train_state is not None:
        payload["train_state"] = train_state
    return atomic_write(path, lambda f: torch.save(payload, f))


def model_only_message(path):
    return (f"{path} is a model-only checkpoint: it holds the parameters but no Adam moments (m, v), no Adam "
            "step counts, no learning-rate scheduler position and no completed-epoch count. Continuing from it "
            "would restart Adam from zero moments on a fresh cosine schedule, a different optimisation from the "
            "one that was interrupted, so exact continuation is unavailable. The file still works for `eval`; to "
            "train this model again, delete it and run `resume`, which restarts it from its seeded initialisation.")


def load_checkpoint(path):
    """(model, train_state) from a training checkpoint; NoTrainingState for a model-only file."""
    ck = torch.load(require(path), weights_only=True)
    if "train_state" not in ck:
        raise NoTrainingState(model_only_message(path))
    return load_model(path), ck["train_state"]


def load_model(path):
    """Rebuild a model saved by save_model, in any Python process (its dtype is restored)."""
    ck = torch.load(require(path), weights_only=True)
    s = ck["spec"]
    dtype = getattr(torch, s["dtype"])
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # inv_softplus and nn.Linear build in the default dtype
    try:
        model = Stitching(torch.zeros(s["K"]), torch.zeros(s["K"], s["N"], s["D"]), torch.ones(s["D"]),
                          _build_energy(s["V"]), learn_c_H=s["learn_c_H"], c_V=s["c_V"],
                          W=_build_energy(s["W"]), loo=s["loo"])
    finally:
        torch.set_default_dtype(previous)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


# ── command line shared by the experiment scripts ───────────────────────────────────────────────


def cli(description, add_train_args, train, evaluate, plot, resume, runs_dir=RUNS, argv=None):
    """`<script> train [...] | resume RUN | eval RUN [--eval-seed S] [--verify-data] | plot RUN [--eval NAME]`.

    train(args) -> Run; resume(args) -> Run; evaluate(args) -> Eval; plot(eval, args) -> None.
    """
    stop_help = "stop after N training epochs in this session, as if interrupted (to try `resume`)"
    ap = argparse.ArgumentParser(description=description)
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train and save a new run (writes a new folder under runs/)")
    tr.add_argument("--name", default=None, help="folder name suffix")
    tr.add_argument("--runs-dir", type=Path, default=runs_dir)
    tr.add_argument("--checkpoint-every", type=int, default=100, help="epochs between training checkpoints")
    tr.add_argument("--stop-after", type=int, default=None, metavar="N", help=stop_help)
    add_train_args(tr)
    rs = sub.add_parser("resume", help="continue an interrupted run from its checkpoints (same schedule)")
    rs.add_argument("run", type=Path)
    rs.add_argument("--stop-after", type=int, default=None, metavar="N", help=stop_help)
    ev = sub.add_parser("eval", help="load a run's checkpoints and evaluate them (never trains)")
    ev.add_argument("run", type=Path)
    ev.add_argument("--eval-seed", type=int, default=0)
    ev.add_argument("--verify-data", action="store_true",
                    help="also regenerate the dataset from its recorded seed and settings and compare it with "
                         "the saved one (by default the saved data are only checked against their hashes)")
    pl = sub.add_parser("plot", help="tables and figures from saved metrics only (never trains or evaluates)")
    pl.add_argument("run", type=Path)
    pl.add_argument("--eval", default=None, help="evals/ folder name (default: the newest)")
    args = ap.parse_args(argv)

    script = Path(sys.argv[0]).resolve()
    script = script.relative_to(ROOT) if ROOT in script.parents else script
    try:
        if args.command in ("train", "resume"):
            run = train(args) if args.command == "train" else resume(args)
            print(f"\nsaved run {run.path}\nnext:  .venv/bin/python {script} eval {run.path}")
        elif args.command == "eval":
            ev = evaluate(args)
            print(f"\nsaved evaluation {ev.path}\nnext:  .venv/bin/python {script} plot {args.run}")
        else:
            plot(open_eval(args.run, args.eval), args)
    except TrainingInterrupted as e:
        run = getattr(e, "run_path", None) or getattr(args, "run", None) or "<run folder>"
        print(f"\ninterrupted: {e}\ncontinue with:  .venv/bin/python {script} resume {run}", file=sys.stderr)
        sys.exit(3)
    except (MissingRunFile, RunMismatch) as e:
        sys.exit(f"error: {e}")
