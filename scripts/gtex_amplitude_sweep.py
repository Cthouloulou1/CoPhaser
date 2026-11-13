"""
Step 5 on GTEx: covariates (age, sex, death type) modulate the mesor AND the amplitude of
each gene's rhythm, with a fixed clock and grid phases (step 6: --covariate-phase adds
per-gene phase delays).

Jobs: tissues x clocks x donor subsets x seeds, each one gtex_train.py process with
  --covariate-amplitude --auto-likelihood-weights --grid-phase-prior
  --decoder-phase-from-grid --grid-refresh-every 10 --phase-prior-anneal-epochs 100000
Clocks:
  mouse    the example mouse clock, frozen (--freeze-decoder)
  human    the human clock of the tissue fitted on the CHIRAL phases (reference/human_chiral,
           scripts/gtex_human_clock.py), frozen
  learned  the clock learned without supervision in exp. 10 (stage 1) whose stage-2 grid
           was closest to CHIRAL on all donors of the tissue (--reference-from-model);
           selected on CHIRAL, hence slightly optimistic for the phase error
Subsets: all donors, fast deaths (Hardy 1-2).

Outputs: <outdir>/<tissue>/<clock>_<subset>_seed<s>/<tissue>/... and, at the end,
  summary.csv (one row per job) and effects_long.csv (tissue, clock, subset, seed, gene,
  covariate, kind, value) for the clock genes.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOCK_GENES = ["ARNTL", "NPAS2", "CRY1", "CRY2", "PER1", "PER2", "PER3", "NR1D1", "NR1D2", "CLOCK", "RORA",
               "TEF", "DBP", "CIART"]
SUBSETS = {"all": None, "fast": ["1", "2"]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tissues", nargs="*", default=None)
    p.add_argument("--clocks", nargs="*", default=["mouse", "learned"], choices=["mouse", "learned", "human"])
    p.add_argument("--human-dir", type=Path, default=REPO_ROOT / "reference" / "human_chiral",
                   help="human clocks from CHIRAL (scripts/gtex_human_clock.py), <tissue>.csv")
    p.add_argument("--subsets", nargs="*", default=list(SUBSETS), choices=list(SUBSETS))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--exp10", type=Path, default=REPO_ROOT / "results" / "gtex" / "exp10_hardy")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "gtex" / "exp11_amplitude")
    p.add_argument("--fixed-phases", action="store_true",
                   help="grid phases computed once before training and never re-estimated "
                        "(no --grid-refresh-every): the covariate effects cannot feed back on the phases")
    p.add_argument("--covariate-phase", action="store_true",
                   help="also fit per-gene phase delays (step 6); use another --outdir")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="extra arguments passed to gtex_train.py (must come last)")
    return p.parse_args()


def learned_clocks(exp10):
    s2 = pd.read_csv(exp10 / "summary_stage2.csv")
    best = s2.loc[s2.groupby("tissue_file")["grid_final_noz_aligned_h"].idxmin()]
    return {r.tissue_file: exp10 / "stage1" / r.tissue_file / f"{r.group}_seed{r.seed}" / r.tissue_file / "model.pt"
            for r in best.itertuples()}


def clock_args(clock, tissue, learned, human_dir):
    """gtex_train.py arguments for a frozen clock: mouse example, learned (exp. 10) or human
    (cosinor on the CHIRAL phases of all donors of the tissue)."""
    if clock == "mouse":
        return ["--freeze-decoder"]
    if clock == "learned":
        return ["--reference-from-model", str(learned[tissue])]
    return ["--reference", str(human_dir / f"{tissue}.csv"), "--freeze-decoder"]


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tissues = args.tissues or sorted(p.stem for p in (REPO_ROOT / "data" / "gtex_v8" / "tissues").glob("*.h5ad"))
    learned = learned_clocks(args.exp10) if "learned" in args.clocks else {}
    (args.outdir / "sweep_args.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "learned_clocks": {k: str(v) for k, v in learned.items()}},
        indent=2))
    log_path = args.outdir / "sweep.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    jobs = []
    for subset in args.subsets:
        for tissue in tissues:
            for clock in args.clocks:
                for seed in args.seeds:
                    out = args.outdir / tissue / f"{clock}_{subset}_seed{seed}"
                    cmd = [sys.executable, "scripts/gtex_train.py", "--tissues", tissue, "--seed", str(seed),
                           "--outdir", str(out), "--epochs", str(args.epochs), "--threads", str(args.threads),
                           "--covariate-amplitude", "--auto-likelihood-weights", "--grid-phase-prior",
                           "--decoder-phase-from-grid", "--phase-prior-anneal-epochs", "100000"]
                    cmd += [] if args.fixed_phases else ["--grid-refresh-every", "10"]
                    cmd += clock_args(clock, tissue, learned, args.human_dir)
                    cmd += ["--covariate-phase"] if args.covariate_phase else []
                    cmd += list(args.extra)
                    if SUBSETS[subset]:
                        cmd += ["--keep-hardy"] + SUBSETS[subset]
                    jobs.append((f"{tissue} {clock} {subset} seed{seed}", cmd, out))

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

    rows, effects = [], []
    for f in sorted(args.outdir.glob("*/*_seed*/*/summary.json")):
        clock, subset, seed = f.parents[1].name.rsplit("_seed", 1)[0].split("_", 1) + [f.parents[1].name.rsplit("_seed", 1)[1]]
        key = {"tissue_file": f.parent.name, "clock": clock, "subset": subset, "seed": int(seed)}
        s = json.loads(f.read_text())
        rows.append({**key, **{k: v for k, v in s.items() if not isinstance(v, (list, dict))}})
        e = pd.read_csv(f.parent / "covariate_effects.csv")
        e = e[e["gene"].isin(CLOCK_GENES)]
        effects.append(e.assign(**key))
    pd.DataFrame(rows).to_csv(args.outdir / "summary.csv", index=False)
    if effects:
        pd.concat(effects, ignore_index=True).to_csv(args.outdir / "effects_long.csv", index=False)
    log(f"collected {len(rows)} summaries. All done.")


if __name__ == "__main__":
    main()
