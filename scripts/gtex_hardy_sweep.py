"""
Two-stage sweep over GTEx tissues x death-type groups (Hardy scale) x seeds.

Stage 1  learn a clock without supervision on the donors of one death-type group:
         gtex_train.py --keep-hardy <group> (mouse clock start, decoder released at
         epoch 10, grid phase prior).
Stage 2  freeze that clock and infer the phases of ALL donors of the tissue:
         gtex_train.py --reference-from-model <stage-1 model.pt> (grid phase prior,
         'hardy' as covariate); errors vs CHIRAL are also reported per death type.

Groups: all (no filter), 0 (ventilator), 1-2 (fast deaths), 3-4 (slow deaths).
A stage-2 job starts as soon as its stage-1 job has finished. Each job runs
gtex_train.py in its own process and writes train.log next to its outputs:
  <outdir>/stage{1,2}/<tissue>/<group>_seed<s>/<tissue>/{summary.json, model.pt, ...}
At the end: <outdir>/summary_stage1.csv, summary_stage2.csv (one row per job).
Progress: <outdir>/sweep.log.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"all": None, "0": ["0"], "1-2": ["1", "2"], "3-4": ["3", "4"]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tissues", nargs="*", default=None, help="file stems (default: every tissue file)")
    p.add_argument("--groups", nargs="*", default=list(GROUPS), choices=list(GROUPS))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--jobs", type=int, default=10, help="concurrent gtex_train.py processes")
    p.add_argument("--threads", type=int, default=6, help="torch threads per process")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "gtex" / "exp10_hardy")
    p.add_argument("--extra", default="", help="extra arguments passed to both stages")
    return p.parse_args()


class Sweep:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=args.jobs)
        self.pending = 0
        self.done = threading.Event()
        self.log_path = args.outdir / "sweep.log"

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self.lock:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        print(line, flush=True)

    def common(self):
        return ["--epochs", str(self.args.epochs), "--auto-likelihood-weights", "--grid-phase-prior",
                "--threads", str(self.args.threads)] + self.args.extra.split()

    def submit(self, fn, *a):
        with self.lock:
            self.pending += 1
        fut = self.pool.submit(fn, *a)
        fut.add_done_callback(self._finished)

    def _finished(self, fut):
        if fut.exception() is not None:
            self.log(f"driver error: {fut.exception()!r}")
        with self.lock:
            self.pending -= 1
            if self.pending == 0:
                self.done.set()

    def run(self, tag, cmd, outdir):
        outdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(outdir / "train.log", "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        self.log(f"{tag} finished rc={rc} in {time.time() - t0:.0f} s")
        return rc

    def stage1(self, tissue, group, seed):
        out = self.args.outdir / "stage1" / tissue / f"{group}_seed{seed}"
        cmd = [sys.executable, "scripts/gtex_train.py", "--tissues", tissue, "--seed", str(seed),
               "--outdir", str(out)] + self.common()
        if GROUPS[group]:
            cmd += ["--keep-hardy"] + GROUPS[group]
        rc = self.run(f"stage1 {tissue} {group} seed{seed}", cmd, out)
        model = out / tissue / "model.pt"
        if rc == 0 and model.exists():
            self.submit(self.stage2, tissue, group, seed, model)
        else:
            self.log(f"stage1 {tissue} {group} seed{seed}: no model, stage 2 skipped")

    def stage2(self, tissue, group, seed, model):
        out = self.args.outdir / "stage2" / tissue / f"{group}_seed{seed}"
        cmd = [sys.executable, "scripts/gtex_train.py", "--tissues", tissue, "--seed", str(seed),
               "--reference-from-model", str(model), "--outdir", str(out)] + self.common()
        self.run(f"stage2 {tissue} {group} seed{seed}", cmd, out)


def collect(outdir, stage):
    rows = []
    for f in sorted((outdir / f"stage{stage}").glob("*/*_seed*/*/summary.json")):
        s = json.loads(f.read_text())
        group, seed = f.parents[1].name.rsplit("_seed", 1)
        row = {"tissue_file": f.parent.name, "group": group, "seed": int(seed)}
        row.update({k: v for k, v in s.items() if not isinstance(v, (list, dict))})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"summary_stage{stage}.csv", index=False)
    return df


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "sweep_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    tissues = args.tissues or sorted(p.stem for p in (REPO_ROOT / "data" / "gtex_v8" / "tissues").glob("*.h5ad"))
    sweep = Sweep(args)
    jobs = [(t, g, s) for g in args.groups for t in tissues for s in args.seeds]  # biggest groups first
    sweep.log(f"start: {len(tissues)} tissues x {len(args.groups)} groups x {len(args.seeds)} seeds "
              f"= {len(jobs)} stage-1 jobs (+ as many stage-2), {args.jobs} at a time")
    for j in jobs:
        sweep.submit(sweep.stage1, *j)
    sweep.done.wait()
    sweep.pool.shutdown()
    for stage in (1, 2):
        df = collect(args.outdir, stage)
        sweep.log(f"stage {stage}: {len(df)} summaries -> summary_stage{stage}.csv")
    sweep.log("All done.")


if __name__ == "__main__":
    main()
