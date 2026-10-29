"""[EVAL] Show saved runs without loading data or training anything.

  .venv/bin/python scripts/show_run.py              list every run (experiments/**/runs/ and runs/)
  .venv/bin/python scripts/show_run.py <run folder> its full configuration, seeds, provenance, models, evaluations
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wgf.runs import Eval, Run, read_json  # noqa: E402


def flat(d, prefix=""):
    """Nested dict -> lines `a.b.c = value`."""
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            yield from flat(v, key + ".")
        else:
            yield f"{key} = {json.dumps(v) if isinstance(v, (list, dict)) else v}"


def list_runs(roots):
    runs = sorted((p for r in roots for p in r.glob("*") if (p / Run.MANIFEST).exists()), key=lambda p: p.name)
    if not runs:
        print(f"no runs under {', '.join(map(str, roots))}")
        return
    print(f"{'run':46} {'experiment':24} {'dataset':20} {'status':9} {'commit':8} {'dirty':5} {'s':>5} evals  folder")
    for p in runs:
        m = read_json(p / Run.MANIFEST)
        pv = m["provenance"]
        ds = m["config"].get("dataset", {}).get("split_id", "-")
        print(f"{p.name:46} {m['experiment']:24} {ds:20} {m['status'][:9]:9} {(pv['commit'] or '-')[:8]:8} "
              f"{str(pv['dirty']):5} {m.get('seconds', float('nan')):5.0f} {len(Run(p).evals()):5}  "
              f"{p.parent.relative_to(ROOT) if ROOT in p.parents else p.parent}")


def show(path):
    run = Run.open(path, finished=False)
    m = run.manifest
    print(f"# {run.path}\n")
    for k in ("experiment", "status", "started", "ended", "seconds", "data_sha256"):
        print(f"{k} = {m.get(k)}")
    print("\n## config")
    print("\n".join(flat(m["config"])))
    print("\n## provenance")
    print("\n".join(flat(m["provenance"])))
    if (run.path / "source.diff").exists():
        print("source.diff = uncommitted changes at train time (saved in the run folder)")
    if m.get("sessions"):
        print("\n## sessions")
        for i, se in enumerate(m["sessions"]):
            pv = se.get("provenance", {})
            print(f"{i}: {se['command']:7} {se['started']} -> {se.get('ended', '?')}  {se.get('status', 'running?')}  "
                  f"commit {(pv.get('commit') or '-')[:8]} dirty {pv.get('dirty')}")
    if m.get("progress"):
        print("\n## progress")
        for tag, p in m["progress"].items():
            print(f"{tag:18} {p['epoch']:6d} / {p['epochs']} epochs  {p['status']}")
    print("\n## models")
    for tag in run.model_tags():
        ck = torch.load(run.path / "models" / f"{tag}.pt", weights_only=True)
        s, r = ck["spec"], ck["readable"]
        V = s["V"]["type"] + (f" {s['V']['hidden']}" if s["V"]["type"] == "mlp" else f" {s['V'].get('fn', '')}")
        epochs = len(run.load_history(tag)["loss"]) if (run.path / "history" / f"{tag}.json").exists() else "?"
        print(f"{tag:18} K={s['K']} N={s['N']} D={s['D']}  V={V}  W={s['W'] and s['W']['type']}  "
              f"score={'loo' if s['loo'] else 'full'}  c_H={r['c_H'].item():.4f} "
              f"({'learned' if s['learn_c_H'] else 'fixed'})  h={[round(x, 4) for x in r['h'].tolist()]}  epochs={epochs}")
    cks = sorted((run.path / "checkpoints").glob("*.pt"))
    if cks:
        print("\n## training checkpoints (resume continues from these)")
        for p in cks:
            st = torch.load(p, weights_only=True)["train_state"]
            print(f"{p.stem:18} epoch {st['epoch']} / {st['epochs']}")
    print("\n## evaluations")
    for p in run.evals():
        e = read_json(p / Eval.MANIFEST)
        files = ", ".join(sorted(f.name for f in p.iterdir() if f.name != Eval.MANIFEST))
        print(f"{p.name:28} {e['status'][:9]:9} {e.get('seconds', float('nan')):6.1f} s   {files}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", type=Path)
    ap.add_argument("--runs-dir", type=Path, action="append", help="where to look (default: every runs/ folder)")
    a = ap.parse_args()
    roots = a.runs_dir or [ROOT / "runs", *sorted((ROOT / "experiments").glob("**/runs"))]
    show(a.run) if a.run else list_runs(roots)
