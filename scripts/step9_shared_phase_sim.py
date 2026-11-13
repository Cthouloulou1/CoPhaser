"""
Step 9 validation: per-tissue vs shared donor phases, on a multi-tissue simulation with known
effects. Do phases shared by the tissues of a donor bias or widen the covariate effects that
the GTEx pipeline estimates, compared with one phase per tissue?

Simulation (cophaser.simulation_covariates): GTEx-like donors (age, sex, condition, ONE phase
per donor), several tissues per donor with known clock offsets (TissueSpec.phase_shift) and
amplitude scales, incomplete sampling (p_sampled). The base clock is a human clock fitted on
CHIRAL (reference/human_chiral/<tissue>.csv: realistic amplitudes); each tissue's frozen clock
is that clock rotated by its offset and scaled. The covariate effects (mesor, log-amplitude,
phase delay) are drawn once (--effects-seed) and identical for every tissue and data seed, so
the spread of the estimates across data seeds is their true sampling variability.
Optional per-sample phase jitter (--jitter-hours): a tissue's sample departs from the donor
phase, i.e. a shared phase is then misspecified.

For every data seed x jitter, the GTEx pipeline (scripts/gtex_train.py, exp. 13h/15 settings:
frozen clock, fixed phases, mesor + amplitude + delay, --covariate-encoder-detach) is run per
tissue with three phase modes:
  tissue  grid phase of each sample with its own tissue's clock (exp. 13h)
  shared  grid log-likelihoods summed over the tissues of a donor (exp. 15,
          gtex_shared_phase.shared_phases)
  oracle  true donor phase (floor; kappa 50)

Simulated tissues are written in the GTEx format read by gtex_train.py (obs donor, tissue,
library_size, age, sex = male/female, hardy = condition 0/1, chiral_ct = true sample phase in
hours), so 'hardy[1]' in the outputs is the simulated condition.

Outputs in --outdir:
  data/s<seed>_j<jitter>/      simulated tissues (.h5ad), phase tables, truth.csv
  clocks/<tissue>.csv          frozen tissue clocks
  runs/s<seed>_j<jitter>/<mode>/<tissue>/...   gtex_train.py outputs
  phase_errors.csv             median |error| (h) of the phases given to the decoder
  effects_long.csv             every estimate with its truth
  bias_variance.csv            per jitter x mode x kind x covariate: slope, intercept,
                               bias (mean of seed-averaged errors), sd across seeds, RMSE
  bias_variance.png
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gtex_shared_phase import shared_phases, tissue_loglik  # noqa: E402

from cophaser.reference_clock import load_reference_clock  # noqa: E402
from cophaser.simulation_covariates import CovariateEffects, TissueSpec, sample_donors, simulate_with_covariates  # noqa: E402

H = 24 / (2 * np.pi)
SIM_COVS = ["age_std", "sex", "condition"]
MODEL_COV = {"age_std": "age", "sex": "sex[female]", "condition": "hardy[1]"}
MODES = ("tissue", "shared", "oracle")
# known tissue offsets (h, positive = clock later) and amplitude scales
TISSUES = [("tA", 0.0, 1.0), ("tB", 2.0, 0.8), ("tC", -1.5, 1.2), ("tD", 4.0, 0.7), ("tE", -3.0, 1.0)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-clock", type=Path, default=REPO_ROOT / "reference" / "human_chiral" / "artery_tibial.csv")
    p.add_argument("--n-donors", type=int, default=900)
    p.add_argument("--n-tissues", type=int, default=5, help=f"first n of {len(TISSUES)} tissues")
    p.add_argument("--p-sampled", type=float, default=0.8)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--jitter-hours", type=float, nargs="*", default=[0.0, 1.0])
    p.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    p.add_argument("--effects-seed", type=int, default=1000)
    p.add_argument("--n-background-genes", type=int, default=500)
    p.add_argument("--frac-affected", type=float, default=0.5)
    p.add_argument("--phase-sd-hours", type=float, default=1.5)
    p.add_argument("--log-amp-sd", type=float, default=0.3)
    p.add_argument("--mesor-sd", type=float, default=0.3)
    p.add_argument("--library-size", type=float, default=5e7)
    p.add_argument("--dispersion", type=float, default=0.5, help="NB dispersion of the simulation")
    p.add_argument("--grid-dispersion", type=float, default=0.1, help="as gtex_train.py --grid-dispersion")
    p.add_argument("--grid-points", type=int, default=96)
    p.add_argument("--min-amp", type=float, default=0.15, help="report phase/amplitude effects of genes above")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--prepare-only", action="store_true", help="simulate and report phase errors, no training")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "step9" / "sim")
    return p.parse_args()


def tissue_specs(args):
    return [TissueSpec(n, phase_shift=s / H, amp_scale=a, mean_sd=0.5, p_sampled=args.p_sampled)
            for n, s, a in TISSUES[: args.n_tissues]]


def tissue_clock(base, spec):
    """Frozen clock of a tissue in the donor-phase frame (n_harm = 1): the simulator draws
    amp * amp_scale * cos(theta - phase_shift - phase), i.e. phase + phase_shift."""
    c = base[["gene", "a_0", "amp", "phase"]].copy()
    c["amp"] = c["amp"] * spec.amp_scale
    c["phase"] = np.mod(c["phase"] + spec.phase_shift, 2 * np.pi)
    c["phase_hours"] = c["phase"] * H
    return c


def draw_effects(args, clock):
    rng = np.random.default_rng(args.effects_seed)
    genes = clock + [f"BG_{i:04d}" for i in range(args.n_background_genes)]
    eff = CovariateEffects.zeros(genes, SIM_COVS)
    shape, cshape = (len(genes), len(SIM_COVS)), (len(clock), len(SIM_COVS))
    eff.mesor.loc[:, :] = rng.normal(0, args.mesor_sd, size=shape) * (rng.random(shape) < args.frac_affected)
    eff.log_amp.loc[clock, :] = rng.normal(0, args.log_amp_sd, size=cshape) * (rng.random(cshape) < args.frac_affected)
    eff.phase.loc[clock, :] = (rng.normal(0, args.phase_sd_hours / H, size=cshape)
                               * (rng.random(cshape) < args.frac_affected))
    return eff


def to_gtex_format(ad):
    o = ad.obs
    ad.obs = pd.DataFrame({
        "donor": o["donor"].astype(str).to_numpy(), "tissue": o["tissue"].astype(str).to_numpy(),
        "library_size": o["library_size_sim"].to_numpy(float), "age": o["age"].to_numpy(float),
        "sex": np.where(o["sex"].to_numpy() == 1, "female", "male"),
        "hardy": o["condition"].astype(int).astype(str).to_numpy(),
        "chiral_ct": np.mod(o["true_phase"].to_numpy(float), 2 * np.pi) * H,
        "donor_phase": o["donor_phase"].to_numpy(float), "true_phase": o["true_phase"].to_numpy(float),
    }, index=ad.obs_names)
    ad.X = ad.layers["counts"].copy()
    return ad


def abs_err_h(a, b):
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))) * H


def prepare(args, seed, jitter, base, effects):
    """Simulate one data set, write the tissues and the phase tables, return phase errors."""
    d = args.outdir / "data" / f"s{seed}_j{jitter:g}"
    d.mkdir(parents=True, exist_ok=True)
    specs = tissue_specs(args)
    clocks = {s.name: tissue_clock(base, s) for s in specs}
    fourier = load_reference_clock(base, n_harm=1).loc[list(base["gene"])]
    rng = np.random.default_rng(seed)
    donors = sample_donors(args.n_donors, rng=rng)
    sim = simulate_with_covariates(fourier, donors, effects, covariate_columns=SIM_COVS, tissues=specs,
                                   n_background_genes=args.n_background_genes, library_size=args.library_size,
                                   dispersion=args.dispersion, phase_jitter_sd=jitter / H, rng=rng)
    ad_all = to_gtex_format(sim.adata)
    clock_genes = list(base["gene"])
    lls, dons, names, per_tissue = [], [], [], {}
    for s in specs:
        ad = ad_all[(ad_all.obs["tissue"] == s.name).to_numpy()].copy()
        ad.write_h5ad(d / f"{s.name}.h5ad")
        fourier_t = load_reference_clock(clocks[s.name], n_harm=1)
        ll, grid = tissue_loglik(ad, fourier_t, clock_genes, args.grid_dispersion, args.grid_points)
        lls.append(ll), dons.append(ad.obs["donor"].to_numpy()), names.append(s.name)
        idx = ll.argmax(1)
        per_tissue[s.name] = pd.DataFrame({"donor": ad.obs["donor"].to_numpy(), "theta": grid[idx],
                                           "true_phase": ad.obs["true_phase"].to_numpy(),
                                           "donor_phase": ad.obs["donor_phase"].to_numpy()})
    shared = shared_phases(dons, lls, grid, names).reset_index()
    shared[["donor", "theta", "kappa", "resultant", "n_tissues"]].to_csv(d / "phases_shared.csv", index=False)
    oracle = pd.DataFrame({"donor": donors.index, "theta": donors["true_phase"].to_numpy(), "kappa": 50.0})
    oracle.to_csv(d / "phases_oracle.csv", index=False)
    effects.long().to_csv(d / "truth.csv", index=False)

    rows = []
    sh = shared.set_index("donor")
    for name, t in per_tissue.items():
        th_sh = sh.loc[t["donor"], "theta"].to_numpy()
        nt = sh.loc[t["donor"], "n_tissues"].to_numpy()
        for mode, th in (("tissue", t["theta"].to_numpy()), ("shared", th_sh), ("oracle", t["donor_phase"].to_numpy())):
            rows.append(dict(seed=seed, jitter_h=jitter, tissue=name, mode=mode, n=len(t),
                             err_vs_sample_h=float(np.median(abs_err_h(th, t["true_phase"]))),
                             err_vs_donor_h=float(np.median(abs_err_h(th, t["donor_phase"]))),
                             median_n_tissues=float(np.median(nt))))
    return rows


def train_cmd(args, seed, jitter, mode, tissue):
    d = args.outdir / "data" / f"s{seed}_j{jitter:g}"
    out = args.outdir / "runs" / f"s{seed}_j{jitter:g}" / mode
    cmd = [sys.executable, "scripts/gtex_train.py", "--tissue-dir", str(d), "--tissues", tissue,
           "--reference", str(args.outdir / "clocks" / f"{tissue}.csv"), "--freeze-decoder",
           "--covariates", "age", "sex", "hardy", "--covariate-amplitude", "--covariate-phase",
           "--auto-likelihood-weights", "--grid-phase-prior", "--decoder-phase-from-grid",
           "--phase-prior-anneal-epochs", "100000", "--covariate-encoder-detach",
           "--grid-dispersion", str(args.grid_dispersion), "--grid-points", str(args.grid_points),
           "--seed", str(seed), "--epochs", str(args.epochs), "--threads", str(args.threads),
           "--outdir", str(out)]
    if mode != "tissue":
        cmd += ["--phase-table", str(d / f"phases_{mode}.csv")]
    return cmd, out / tissue


def truth_in_model_units(truth, run_dir, clock_amp):
    """Simulator effects per unit of the model's covariate columns; delays centred on the
    amplitude-weighted mean of the clock genes (the model's convention)."""
    import torch

    spec = {c["key"]: c for c in torch.load(run_dir / "model.pt", weights_only=False)["covariate_spec"]}
    t = truth[truth["covariate"].isin(SIM_COVS)].copy()
    t["value"] = np.where(t["covariate"] == "age_std", t["value"] * spec["age"]["sd"] / 15.0, t["value"])
    t["covariate"] = t["covariate"].map(MODEL_COV)
    ph = t["kind"] == "phase"
    w = t.loc[ph, "gene"].map(clock_amp / clock_amp.sum()).fillna(0.0)
    mean = (t.loc[ph].assign(w=w, wv=w * t.loc[ph, "value"]).groupby("covariate")["wv"].sum())
    t.loc[ph, "value"] = t.loc[ph, "value"] - t.loc[ph, "covariate"].map(mean)
    return t


def collect(args):
    rows = []
    for f in sorted((args.outdir / "runs").glob("s*_j*/*/*/summary.json")):
        run_dir, mode, sj = f.parent, f.parent.parent.name, f.parent.parent.parent.name
        seed, jitter = int(sj.split("_j")[0][1:]), float(sj.split("_j")[1])
        truth = pd.read_csv(args.outdir / "data" / sj / "truth.csv")
        amp = pd.read_csv(run_dir / "clock_genes.csv", index_col=0)["amp"]
        t = truth_in_model_units(truth, run_dir, amp)
        e = pd.read_csv(run_dir / "covariate_effects.csv")
        m = e.merge(t, on=["gene", "covariate", "kind"], suffixes=("_est", "_true"))
        m = m[m["gene"].isin(amp.index)]
        m["amp"] = m["gene"].map(amp)
        rows.append(m.assign(seed=seed, jitter_h=jitter, mode=mode, tissue=run_dir.name))
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(args.outdir / "effects_long.csv", index=False)
    return df


def bias_variance(df, min_amp):
    keep = (df["kind"] == "mesor") | (df["amp"] >= min_amp)
    df = df[keep].copy()
    scale = np.where(df["kind"] == "phase", H, 1.0)  # phase in hours
    df["est"], df["true"] = df["value_est"] * scale, df["value_true"] * scale
    df["err"] = df["est"] - df["true"]
    out = []
    for (j, mode, kind, cov), g in df.groupby(["jitter_h", "mode", "kind", "covariate"]):
        cell = g.groupby(["tissue", "gene"])
        mean_err = cell["err"].mean()  # bias of each estimate, averaged over seeds
        sd = cell["est"].std(ddof=1)
        slope, intercept = np.polyfit(g["true"], g["est"], 1) if g["true"].std() > 0 else (np.nan, np.nan)
        nz = g["true"].abs() > 1e-9
        out.append(dict(jitter_h=j, mode=mode, kind=kind, covariate=cov, n_estimates=len(g),
                        n_seeds=g["seed"].nunique(), slope=slope, intercept=intercept,
                        slope_nonzero=(np.polyfit(g.loc[nz, "true"], g.loc[nz, "est"], 1)[0]
                                       if nz.sum() > 2 else np.nan),
                        corr=float(np.corrcoef(g["true"], g["est"])[0, 1]) if g["true"].std() > 0 else np.nan,
                        bias_rms=float(np.sqrt((mean_err ** 2).mean())),
                        sd_across_seeds_median=float(sd.median()),
                        rmse=float(np.sqrt((g["err"] ** 2).mean()))))
    return pd.DataFrame(out)


def plot(bv, outdir):
    kinds = [k for k in ("phase", "log_amp", "mesor") if k in set(bv["kind"])]
    metrics = [("slope", "slope (1 = unbiased)"), ("bias_rms", "RMS bias"), ("sd_across_seeds_median", "SD across seeds")]
    jit = sorted(bv["jitter_h"].unique())
    colors = {"tissue": "#2a6f97", "shared": "#e07a1f", "oracle": "#6c757d"}
    fig, axes = plt.subplots(len(kinds), len(metrics), figsize=(4.2 * len(metrics), 3.0 * len(kinds)), squeeze=False)
    for r, kind in enumerate(kinds):
        for c, (m, label) in enumerate(metrics):
            ax = axes[r, c]
            sub = bv[bv["kind"] == kind]
            covs = sorted(sub["covariate"].unique())
            labels = [f"{cv}\nj={j:g}h" for j in jit for cv in covs]
            x = np.arange(len(labels))
            modes = [mo for mo in MODES if mo in set(sub["mode"])]
            for i, mo in enumerate(modes):
                v = [sub[(sub["mode"] == mo) & (sub["covariate"] == cv) & (sub["jitter_h"] == j)][m].mean()
                     for j in jit for cv in covs]
                ax.bar(x + (i - (len(modes) - 1) / 2) * 0.27, v, width=0.27, color=colors[mo], label=mo)
            if m == "slope":
                ax.axhline(1, color="#333", lw=0.8, ls="--")
            ax.set_xticks(x, labels, fontsize=7)
            ax.set_title(f"{kind}: {label}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "bias_variance.png", dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    base = pd.read_csv(args.base_clock)
    clock = list(base["gene"])
    effects = draw_effects(args, clock)
    if not args.report_only:
        (args.outdir / "clocks").mkdir(exist_ok=True)
        for s in tissue_specs(args):
            tissue_clock(base, s).to_csv(args.outdir / "clocks" / f"{s.name}.csv", index=False)
        rows = []
        for seed in args.seeds:
            for j in args.jitter_hours:
                t0 = time.time()
                rows += prepare(args, seed, j, base, effects)
                print(f"prepared seed {seed} jitter {j:g} h in {time.time() - t0:.0f} s", flush=True)
        pe = pd.DataFrame(rows)
        pe.to_csv(args.outdir / "phase_errors.csv", index=False)
        print(pe.groupby(["jitter_h", "mode"])[["err_vs_sample_h", "err_vs_donor_h"]].median().round(2), flush=True)
        if args.prepare_only:
            return
        jobs = []
        for seed in args.seeds:  # seed-major: partial results cover every condition
            for j in args.jitter_hours:
                for mode in args.modes:
                    for s in tissue_specs(args):
                        cmd, done = train_cmd(args, seed, j, mode, s.name)
                        if not (done / "summary.json").exists():
                            jobs.append((f"s{seed} j{j:g} {mode} {s.name}", cmd, done))
        log_path = args.outdir / "train.log"

        def run(job):
            tag, cmd, done = job
            done.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(done / "train.log", "w") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
            with open(log_path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {tag} rc={rc} {time.time() - t0:.0f} s\n")

        print(f"{len(jobs)} training jobs, {args.jobs} at a time", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(run, jobs))
    df = collect(args)
    if df is None:
        print("no finished runs")
        return
    bv = bias_variance(df, args.min_amp)
    bv.to_csv(args.outdir / "bias_variance.csv", index=False)
    plot(bv, args.outdir)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(bv.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
