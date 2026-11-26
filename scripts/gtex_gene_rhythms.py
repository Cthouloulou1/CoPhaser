"""
Step 10 on GTEx: rhythm of every expressed gene and its modulation by age, sex and death type,
on phases fixed beforehand (default: donor phases shared across tissues, exp. 14, frozen human
clocks). Per tissue, one negative binomial GLM per gene (cophaser.gene_rhythms):

    log mu = log(library) + a + b cos(theta) + c sin(theta) + beta . x + nu . z
             + (b' cos(theta) + c' sin(theta)) . x

x: age ((age - 50) / 10, i.e. per 10 years), sex (female = 1, reference male), death type
(Hardy; default levels 1 and 2 merged as the reference 'fast deaths'); z: nuisance covariates
(ischemic time, RIN; standardised; mesor only). Samples without a phase or a death type are
dropped. Genes: median CPM >= --min-median-cpm, plus the clock genes; 'in_hvg' flags the
genes of the CoPhaser context (scripts/gtex_train.py: --n-hvg most variable + clock genes).

Caveats: the phase error (~2 h) attenuates amplitudes; it is larger for ventilator cases
(Hardy 0), so their rhythms look flatter even without a true change (see the step 10
simulation, scripts/step10_sim.py). The clock genes gave the phases: their own results are
optimistic (circular).

Outputs in --outdir: <tissue>.csv.gz (one row per gene; q-values within the tissue),
summary.csv (one row per tissue), args.json.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tissue-dir", type=Path, default=REPO_ROOT / "data" / "gtex_v8" / "tissues")
    p.add_argument("--tissues", nargs="*", default=None, help="file stems (default: every .h5ad)")
    p.add_argument("--phase-table", type=Path,
                   default=REPO_ROOT / "results" / "gtex" / "exp14_shared_phase" / "donor_phases.csv",
                   help="CSV with columns donor, theta (rad), optionally kappa")
    p.add_argument("--min-kappa", type=float, default=0.0, help="drop donors whose phase concentration is lower")
    p.add_argument("--permute-phases", type=int, default=None,
                   help="negative control: shuffle the phases across the samples of a tissue with this seed "
                        "(covariates stay with their sample); rhythm p-values should then be uniform")
    p.add_argument("--min-median-cpm", type=float, default=1.0)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--min-mean-count", type=float, default=5.0, help="as gtex_train.py, for the HVG flag")
    p.add_argument("--covariates", nargs="*", default=["age", "sex", "hardy"])
    p.add_argument("--merge-hardy", nargs="*", default=["1", "2"])
    p.add_argument("--hardy-reference", default="1-2")
    p.add_argument("--nuisance", nargs="*", default=["ischemic_time", "rin"])
    p.add_argument("--age-center", type=float, default=50.0)
    p.add_argument("--age-scale", type=float, default=10.0)
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=2, help="tissues fitted in parallel")
    p.add_argument("--threads", type=int, default=8, help="torch threads per job")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "step10" / "gtex" / "shared_ref12")
    return p.parse_args(argv)


def encode(obs, args):
    """Covariates of interest (centred / 0-1 coded) and nuisance covariates (standardised)."""
    from gtex_train import merge_levels

    cov = pd.DataFrame(index=obs.index)
    if "age" in args.covariates:
        cov["age"] = (obs["age"].to_numpy(float) - args.age_center) / args.age_scale
    if "sex" in args.covariates:
        cov["sex"] = (obs["sex"].astype(str).to_numpy() == "female").astype(float)
    if "hardy" in args.covariates:
        h = merge_levels(obs["hardy"], args.merge_hardy) if args.merge_hardy else obs["hardy"].astype(str)
        ref = str(args.hardy_reference)
        levels = sorted(set(h.dropna()) - {ref}, key=str)
        if ref not in set(h.dropna()):
            raise ValueError(f"Hardy reference {ref!r} not among the levels {sorted(set(h.dropna()))}")
        for lv in levels:
            cov[f"hardy[{lv}]"] = (h.to_numpy() == lv).astype(float)
    nui = pd.DataFrame(index=obs.index)
    for m in args.nuisance:
        x = obs[m].to_numpy(float)
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        nui[m] = (x - x.mean()) / x.std()
    return cov, nui


def run_tissue(path, args):
    import torch

    from cophaser.gene_rhythms import add_qvalues, fit_gene_rhythms
    from cophaser.gene_sets import HUMAN_CIRCADIAN_GENE_SET
    from gtex_train import select_genes

    torch.set_num_threads(args.threads)
    t0 = time.time()
    name = path.stem
    ad = anndata.read_h5ad(path)
    phases = pd.read_csv(args.phase_table).set_index("donor")
    keep = ad.obs["donor"].astype(str).isin(phases.index).to_numpy().copy()
    if args.min_kappa > 0 and "kappa" in phases:
        ok = phases.index[phases["kappa"] >= args.min_kappa]
        keep &= ad.obs["donor"].astype(str).isin(ok).to_numpy()
    if "hardy" in args.covariates:
        keep &= ad.obs["hardy"].notna().to_numpy()
    n_all = ad.n_obs
    ad = ad[keep].copy()
    theta = phases.loc[ad.obs["donor"].astype(str), "theta"].to_numpy(float)
    if args.permute_phases is not None:
        theta = np.random.default_rng(args.permute_phases).permutation(theta)

    lib = ad.obs["library_size"].to_numpy(float)
    X = ad.layers["counts"]
    med_cpm = np.median((X.toarray() if hasattr(X, "toarray") else np.asarray(X)) / lib[:, None] * 1e6, axis=0)
    clock = [g for g in HUMAN_CIRCADIAN_GENE_SET if g in ad.var_names]
    genes = sorted(set(ad.var_names[med_cpm >= args.min_median_cpm]) | set(clock))
    _, context = select_genes(ad, args.n_hvg, args.min_mean_count)
    Y = ad[:, genes].layers["counts"]
    Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)

    cov, nui = encode(ad.obs, args)
    res = fit_gene_rhythms(Y, lib, theta, cov, nui, genes=genes, chunk=args.chunk)
    res = add_qvalues(res)
    res.insert(1, "tissue", name)
    res["is_clock"] = res["gene"].isin(clock)
    res["in_hvg"] = res["gene"].isin(context)
    args.outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.outdir / f"{name}.csv.gz", index=False)

    s = dict(tissue=name, permuted=args.permute_phases is not None,
             n_samples_all=n_all, n_samples=int(ad.n_obs), n_genes=len(genes),
             frac_p_rhythm_05=float((res["p_rhythm"] < 0.05).mean()),
             n_hvg=int(res["in_hvg"].sum()), not_converged=int((~res["converged"]).sum()),
             rhythmic_q05=int((res["q_rhythm"] < 0.05).sum()),
             rhythmic_q05_hvg=int(((res["q_rhythm"] < 0.05) & res["in_hvg"]).sum()),
             clock_rhythmic_q05=int(((res["q_rhythm"] < 0.05) & res["is_clock"]).sum()), n_clock=len(clock),
             seconds=round(time.time() - t0, 1))
    for k in cov.columns:
        s[f"diffrhythm_q05_{k}"] = int((res[f"q_diffrhythm_{k}"] < 0.05).sum())
        s[f"mesor_q05_{k}"] = int((res[f"q_mesor_{k}"] < 0.05).sum())
        clk = res[res["is_clock"] & (res["amp"] >= 0.15)]
        s[f"clock_median_log_amp_ratio_{k}"] = float(np.median(clk[f"log_amp_ratio_{k}"])) if len(clk) else np.nan
    for k, v in cov.items():
        if set(np.unique(v)) <= {0.0, 1.0}:
            s[f"n_{k}"] = int(v.sum())
    print(f"[{name}] {s['n_samples']} samples, {s['n_genes']} genes, rhythmic (q < 0.05) {s['rhythmic_q05']} "
          f"(clock {s['clock_rhythmic_q05']}/{s['n_clock']}), {s['seconds']} s", flush=True)
    return s


def main(argv=None):
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    paths = [args.tissue_dir / f"{t}.h5ad" for t in args.tissues] if args.tissues else sorted(args.tissue_dir.glob("*.h5ad"))
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            rows = list(ex.map(run_tissue, paths, [args] * len(paths)))
    else:
        rows = [run_tissue(p, args) for p in paths]
    summ = pd.DataFrame(rows)
    summ.to_csv(args.outdir / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(summ.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
