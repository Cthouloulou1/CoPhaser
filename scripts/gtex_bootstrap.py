"""
Step 7: donor bootstrap of the step-6 GTEx fit (exp. 12 settings), for confidence intervals
of the covariate effects (mesor, log-amplitude, phase delay) of the clock genes.

For every tissue and replicate b = 0 .. B-1, one gtex_train.py process with
  --bootstrap-seed b --seed b          (donors resampled with replacement; own training seed)
  --covariate-amplitude --covariate-phase --auto-likelihood-weights --grid-phase-prior
  --decoder-phase-from-grid --phase-prior-anneal-epochs 100000   (fixed grid phases)
  --reference-from-model <learned clock of exp. 10>  (--clock mouse: --freeze-decoder;
  --clock human: --reference reference/human_chiral/<tissue>.csv --freeze-decoder)
Each replicate recomputes its grid phases (once, before training), so the intervals include
the phase uncertainty and the optimisation noise. --phase-table (step 9): phases shared by the
tissues of a donor, read from a table instead (same for every replicate: frozen clock).

Outputs: <outdir>/<tissue>/boot<b>/<tissue>/..., and at the end effects_long.csv (tissue,
replicate, gene, covariate, kind, value, amp, age_sd) for the clock genes and summary.csv.
Then run scripts/gtex_bootstrap_report.py.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gtex_amplitude_sweep import CLOCK_GENES, SUBSETS, clock_args, learned_clocks  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tissues", nargs="*", default=None)
    p.add_argument("--replicates", type=int, default=40)
    p.add_argument("--clock", default="learned", choices=["mouse", "learned", "human"])
    p.add_argument("--human-dir", type=Path, default=REPO_ROOT / "reference" / "human_chiral")
    p.add_argument("--phase-table", type=Path, default=None,
                   help="donor phases shared across tissues (scripts/gtex_shared_phase.py), passed to "
                        "gtex_train.py; with a frozen clock a donor's phase does not depend on which donors "
                        "are drawn, so one table serves every replicate")
    p.add_argument("--covariate-encoder-detach", action="store_true",
                   help="passed to gtex_train.py: only the generative model trains the covariate effects")
    p.add_argument("--subset", default="all", choices=list(SUBSETS))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--exp10", type=Path, default=REPO_ROOT / "results" / "gtex" / "exp10_hardy")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "gtex" / "exp13_bootstrap")
    return p.parse_args()


def collect(outdir):
    rows, effects = [], []
    for f in sorted(outdir.glob("*/boot*/*/summary.json")):
        run_dir = f.parent
        key = {"tissue_file": run_dir.name, "replicate": int(run_dir.parent.name[4:])}
        s = json.loads(f.read_text())
        rows.append({**key, **{k: v for k, v in s.items() if not isinstance(v, (list, dict))}})
        e = pd.read_csv(run_dir / "covariate_effects.csv")
        e = e[e["gene"].isin(CLOCK_GENES)]
        amp = pd.read_csv(run_dir / "clock_genes.csv", index_col=0)["amp"]
        spec = {c["key"]: c for c in torch.load(run_dir / "model.pt", weights_only=False)["covariate_spec"]}
        effects.append(e.assign(amp=e["gene"].map(amp), age_sd=float(spec["age"]["sd"]) if "age" in spec else float("nan"),
                                **key))
    pd.DataFrame(rows).to_csv(outdir / "summary.csv", index=False)
    if effects:
        pd.concat(effects, ignore_index=True).to_csv(outdir / "effects_long.csv", index=False)
    return len(rows)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tissues = args.tissues or sorted(p.stem for p in (REPO_ROOT / "data" / "gtex_v8" / "tissues").glob("*.h5ad"))
    learned = learned_clocks(args.exp10) if args.clock == "learned" else {}
    (args.outdir / "bootstrap_args.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "learned_clocks": {k: str(v) for k, v in learned.items()}},
        indent=2))
    log_path = args.outdir / "bootstrap.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    jobs = []
    for b in range(args.replicates):  # replicate-major: partial results cover every tissue
        for tissue in tissues:
            out = args.outdir / tissue / f"boot{b}"
            if (out / tissue / "summary.json").exists():
                continue  # resume
            cmd = [sys.executable, "scripts/gtex_train.py", "--tissues", tissue, "--seed", str(b),
                   "--bootstrap-seed", str(b), "--outdir", str(out), "--epochs", str(args.epochs),
                   "--threads", str(args.threads), "--covariate-amplitude", "--covariate-phase",
                   "--auto-likelihood-weights", "--grid-phase-prior", "--decoder-phase-from-grid",
                   "--phase-prior-anneal-epochs", "100000"]
            cmd += clock_args(args.clock, tissue, learned, args.human_dir)
            if args.phase_table is not None:
                cmd += ["--phase-table", str(args.phase_table.resolve())]
            if args.covariate_encoder_detach:
                cmd += ["--covariate-encoder-detach"]
            if SUBSETS[args.subset]:
                cmd += ["--keep-hardy"] + SUBSETS[args.subset]
            jobs.append((f"{tissue} boot{b}", cmd, out))

    def run(job):
        tag, cmd, out = job
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(out / "train.log", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        log(f"{tag} finished rc={rc} in {time.time() - t0:.0f} s")

    log(f"start: {len(jobs)} jobs, {args.jobs} at a time")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, jobs))
    n = collect(args.outdir)
    log(f"collected {n} replicates. All done.")


if __name__ == "__main__":
    main()
