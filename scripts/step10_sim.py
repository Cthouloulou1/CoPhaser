"""
Step 10 validation: rhythm of non-clock genes and its modulation by covariates, estimated by
the per-gene NB GLM (cophaser.gene_rhythms) on phases inferred from the clock genes only.

Simulation (cophaser.simulation_covariates, multi-tissue design of step 9): GTEx-like donors
(age, sex, condition, one phase per donor), several tissues with known clock offsets. Genes:
  - clock genes: frozen human clock (reference/human_chiral/<tissue>.csv), no covariate
    effects; they alone give the phases (grid per tissue, or summed over a donor's tissues);
  - output genes OUT_xxxx: a fraction is rhythmic (random amplitude and peak time), the others
    flat; sparse known effects of each covariate on the mesor (all output genes), and on the
    log-amplitude and peak time (rhythmic ones). Effects and output genes are drawn once
    (--effects-seed); data seeds redraw donors and counts.

For each data seed, the GLM is fitted per tissue with four phase modes:
  oracle            true donor phase
  shared            grid log-likelihoods summed over the tissues of a donor (step 9 / exp. 14)
  tissue            grid phase of each sample with its own tissue's clock
  shared_noisycond  shared phase + extra noise (--extra-noise-hours) for the condition = 1
                    donors only, mimicking the worse phases of ventilator cases in GTEx. A flat
                    attenuation of their rhythms is then expected; 'cal' estimates subtract the
                    median log-amplitude ratio of the clock genes (which have no true effect
                    here) as a correction.

Outputs in --outdir: glm/<seed>_<mode>_<tissue>.csv.gz (every gene), phase_errors.csv,
estimates_long.csv (estimate vs truth), metrics.csv (detection power / false discovery
proportion at q < 0.05; slope, correlation, median under the null and RMSE of the estimates),
scatter_log_amp.png, scatter_phase.png.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gtex_shared_phase import shared_phases, tissue_loglik  # noqa: E402
from step9_shared_phase_sim import SIM_COVS, abs_err_h, tissue_clock, tissue_specs, to_gtex_format  # noqa: E402

from cophaser.gene_rhythms import H, add_qvalues, fit_gene_rhythms  # noqa: E402
from cophaser.reference_clock import load_reference_clock  # noqa: E402
from cophaser.simulation_covariates import CovariateEffects, sample_donors, simulate_with_covariates  # noqa: E402

MODES = ("oracle", "shared", "tissue", "shared_noisycond")
GLM_COV = {"age_std": "age", "sex": "sex", "condition": "condition"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-clock", type=Path, default=REPO_ROOT / "reference" / "human_chiral" / "artery_tibial.csv")
    p.add_argument("--n-donors", type=int, default=900)
    p.add_argument("--n-tissues", type=int, default=5)
    p.add_argument("--p-sampled", type=float, default=0.8)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    p.add_argument("--effects-seed", type=int, default=2000)
    p.add_argument("--n-out", type=int, default=1000, help="output (non-clock) genes")
    p.add_argument("--frac-rhythmic", type=float, default=0.5)
    p.add_argument("--out-amp", type=float, nargs=2, default=[0.1, 0.6], help="amplitude range (log units)")
    p.add_argument("--out-log-frac", type=float, nargs=2, default=[-13.0, -9.0], help="log count-fraction range")
    p.add_argument("--frac-mesor", type=float, default=0.3)
    p.add_argument("--mesor-sd", type=float, default=0.3)
    p.add_argument("--frac-diff", type=float, default=0.2, help="P(effect) on log-amplitude, and on the phase")
    p.add_argument("--log-amp-sd", type=float, default=0.4)
    p.add_argument("--phase-sd-hours", type=float, default=2.0)
    p.add_argument("--dispersion", type=float, default=0.5, help="NB dispersion (as step 9: realistic phase errors)")
    p.add_argument("--library-size", type=float, default=5e7)
    p.add_argument("--grid-dispersion", type=float, default=0.1)
    p.add_argument("--grid-points", type=int, default=96)
    p.add_argument("--extra-noise-hours", type=float, default=2.5)
    p.add_argument("--min-amp", type=float, default=0.15, help="amplitude/phase estimates scored above this true amp")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "step10" / "sim")
    return p.parse_args()


def output_genes(args, clock_fourier, rng):
    names = [f"OUT_{i:04d}" for i in range(args.n_out)]
    rhythmic = rng.random(args.n_out) < args.frac_rhythmic
    amp = np.where(rhythmic, rng.uniform(*args.out_amp, args.n_out), 0.0)
    ph = rng.uniform(-np.pi, np.pi, args.n_out)
    vals = {"A_0": rng.uniform(*args.out_log_frac, args.n_out), "A_1": amp * np.cos(ph), "B_1": amp * np.sin(ph)}
    if set(clock_fourier.columns) != set(vals):
        raise ValueError(f"expected a one-harmonic clock, got columns {list(clock_fourier.columns)}")
    out = pd.DataFrame({c: vals[c] for c in clock_fourier.columns}, index=pd.Index(names))
    return out, pd.Series(amp, index=names)


def draw_effects(args, clock, out_genes, out_amp, rng):
    eff = CovariateEffects.zeros(clock + out_genes, SIM_COVS)
    shape = (len(out_genes), len(SIM_COVS))
    rh = (out_amp.to_numpy() > 0)[:, None]
    eff.mesor.loc[out_genes, :] = rng.normal(0, args.mesor_sd, shape) * (rng.random(shape) < args.frac_mesor)
    eff.log_amp.loc[out_genes, :] = rng.normal(0, args.log_amp_sd, shape) * (rng.random(shape) < args.frac_diff) * rh
    eff.phase.loc[out_genes, :] = (rng.normal(0, args.phase_sd_hours / H, shape)
                                   * (rng.random(shape) < args.frac_diff) * rh)
    return eff


def truth_table(base, clock_fourier, out_amp, effects, specs):
    """One row per (tissue, gene, covariate): true reference amplitude and effects in GLM units
    (age per 15 years = age_std; phase in hours)."""
    amp = pd.concat([pd.Series(np.hypot(clock_fourier["A_1"], clock_fourier["B_1"]).to_numpy(),
                               index=clock_fourier.index), out_amp])
    rows = []
    for s in specs:
        for cov in SIM_COVS:
            rows.append(pd.DataFrame({
                "tissue": s.name, "gene": amp.index, "covariate": GLM_COV[cov],
                "amp_true": amp.to_numpy() * s.amp_scale,
                "mesor_true": effects.mesor[cov].reindex(amp.index).fillna(0).to_numpy(),
                "log_amp_true": effects.log_amp[cov].reindex(amp.index).fillna(0).to_numpy(),
                "phase_true_h": effects.phase[cov].reindex(amp.index).fillna(0).to_numpy() * H,
                "is_clock": amp.index.isin(base["gene"]),
            }))
    return pd.concat(rows, ignore_index=True)


def simulate_seed(args, seed, base, fourier, effects):
    specs = tissue_specs(args)
    rng = np.random.default_rng(seed)
    donors = sample_donors(args.n_donors, rng=rng)
    sim = simulate_with_covariates(fourier, donors, effects, covariate_columns=SIM_COVS, tissues=specs,
                                   n_background_genes=0, library_size=args.library_size,
                                   dispersion=args.dispersion, rng=rng)
    ad_all = to_gtex_format(sim.adata)
    clock = list(base["gene"])
    tissues, lls, dons, per_tissue = {}, [], [], {}
    for s in specs:
        ad = ad_all[(ad_all.obs["tissue"] == s.name).to_numpy()].copy()
        tissues[s.name] = ad
        ll, grid = tissue_loglik(ad, load_reference_clock(tissue_clock(base, s), n_harm=1), clock,
                                 args.grid_dispersion, args.grid_points)
        ll, grid = np.asarray(ll), np.asarray(grid)
        lls.append(ll), dons.append(ad.obs["donor"].astype(str).to_numpy())
        per_tissue[s.name] = grid[ll.argmax(1)]
    shared = shared_phases(dons, lls, grid, [s.name for s in specs])
    noise = np.random.default_rng(10_000 + seed).normal(0, args.extra_noise_hours / H, len(donors))
    extra = pd.Series(noise * donors["condition"].to_numpy(), index=donors.index.astype(str))
    return donors, tissues, per_tissue, shared, extra


def mode_phase(mode, ad, donors, per_tissue, shared, extra):
    d = ad.obs["donor"].astype(str).to_numpy()
    if mode == "oracle":
        return donors["true_phase"].reindex(d).to_numpy(float)
    if mode == "tissue":
        return per_tissue
    th = shared["theta"].reindex(d).to_numpy(float)
    if mode == "shared_noisycond":
        th = th + extra.reindex(d).to_numpy(float)
    return th


def glm_covariates(obs):
    return pd.DataFrame({
        "age": (obs["age"].to_numpy(float) - 50.0) / 15.0,  # = age_std of the simulator
        "sex": (obs["sex"].astype(str).to_numpy() == "female").astype(float),
        "condition": (obs["hardy"].astype(str).to_numpy() == "1").astype(float),
    })


def long_estimates(res, seed, mode, tissue, min_amp_clock=0.15):
    """Estimates in long format, with the clock-calibrated log-amplitude ratio."""
    rows = []
    for k in GLM_COV.values():
        clk = res[res["gene"].str.startswith("OUT_") == False]  # noqa: E712
        clk = clk[clk["amp"] >= min_amp_clock]
        offset = float(np.median(clk[f"log_amp_ratio_{k}"])) if len(clk) else 0.0
        rows.append(pd.DataFrame({
            "seed": seed, "mode": mode, "tissue": tissue, "gene": res["gene"], "covariate": k,
            "amp_est": res["amp"], "q_rhythm": res["q_rhythm"], "q_diffrhythm": res[f"q_diffrhythm_{k}"],
            "mesor_est": res[f"mesor_{k}"], "log_amp_est": res[f"log_amp_ratio_{k}"],
            "log_amp_cal_est": res[f"log_amp_ratio_{k}"] - offset, "clock_log_amp_offset": offset,
            "phase_est_h": res[f"phase_shift_h_{k}"],
        }))
    return pd.concat(rows, ignore_index=True)


def score(est, truth, min_amp):
    """Detection and estimation metrics per mode x covariate (output genes only)."""
    m = est.merge(truth, on=["tissue", "gene", "covariate"])
    m = m[~m["is_clock"]]
    rows = []
    for (mode, cov), g in m.groupby(["mode", "covariate"]):
        r = dict(mode=mode, covariate=cov)
        # detection, per seed x tissue then averaged
        det = []
        for _, gg in g.groupby(["seed", "tissue"]):
            rh = gg["amp_true"] > 0
            call = gg["q_rhythm"] < 0.05
            diff = rh & ((gg["log_amp_true"] != 0) | (gg["phase_true_h"] != 0))
            dcall = gg["q_diffrhythm"] < 0.05
            det.append(dict(rhythm_power=call[rh].mean(), rhythm_fdp=(~rh[call]).mean() if call.any() else 0.0,
                            diff_power=dcall[diff].mean() if diff.any() else np.nan,
                            diff_fdp=(~diff[dcall]).mean() if dcall.any() else 0.0))
        r.update(pd.DataFrame(det).mean().to_dict())
        sel = g[g["amp_true"] >= min_amp]
        for kind, e, t in (("mesor", g["mesor_est"], g["mesor_true"]),
                           ("log_amp", sel["log_amp_est"], sel["log_amp_true"]),
                           ("log_amp_cal", sel["log_amp_cal_est"], sel["log_amp_true"]),
                           ("phase_h", sel["phase_est_h"], sel["phase_true_h"])):
            ok = np.isfinite(e) & np.isfinite(t)
            e, t = e[ok].to_numpy(), t[ok].to_numpy()
            nz = t != 0
            r[f"{kind}_slope"] = float(np.polyfit(t[nz], e[nz], 1)[0]) if nz.sum() > 2 else np.nan
            r[f"{kind}_corr"] = float(np.corrcoef(t, e)[0, 1]) if nz.any() else np.nan
            r[f"{kind}_null_median"] = float(np.median(e[~nz])) if (~nz).any() else np.nan
            r[f"{kind}_rmse"] = float(np.sqrt(np.mean((e - t) ** 2)))
        rows.append(r)
    return pd.DataFrame(rows)


def scatter(est, truth, kind, true_col, path, min_amp):
    m = est.merge(truth, on=["tissue", "gene", "covariate"])
    m = m[(~m["is_clock"]) & (m["amp_true"] >= min_amp)]
    covs, modes = list(GLM_COV.values()), [mo for mo in MODES if mo in set(m["mode"])]
    fig, axes = plt.subplots(len(covs), len(modes), figsize=(3.2 * len(modes), 3 * len(covs)), squeeze=False)
    for i, cov in enumerate(covs):
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            g = m[(m["covariate"] == cov) & (m["mode"] == mode)]
            ax.scatter(g[true_col], g[kind], s=3, alpha=0.3)
            lim = np.nanpercentile(np.abs(np.r_[g[true_col], g[kind]]), 99)
            ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.8)
            ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
            ax.set_title(f"{cov} - {mode}", fontsize=9)
            if i == len(covs) - 1:
                ax.set_xlabel("true")
            if j == 0:
                ax.set_ylabel("estimated")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "glm").mkdir(exist_ok=True)
    (args.outdir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    base = pd.read_csv(args.base_clock)
    clock = list(base["gene"])
    clock_fourier = load_reference_clock(base, n_harm=1).loc[clock]
    rng = np.random.default_rng(args.effects_seed)
    out_fourier, out_amp = output_genes(args, clock_fourier, rng)
    effects = draw_effects(args, clock, list(out_fourier.index), out_amp, rng)
    fourier = pd.concat([clock_fourier, out_fourier])
    specs = tissue_specs(args)
    truth = truth_table(base, clock_fourier, out_amp, effects, specs)
    truth.to_csv(args.outdir / "truth.csv.gz", index=False)

    est, perr = [], []
    for seed in args.seeds:
        t0 = time.time()
        donors, tissues, per_tissue, shared, extra = simulate_seed(args, seed, base, fourier, effects)
        print(f"seed {seed}: simulated and phased in {time.time() - t0:.0f} s", flush=True)
        for mode in args.modes:
            for name, ad in tissues.items():
                theta = mode_phase(mode, ad, donors, per_tissue[name], shared, extra)
                true = donors["true_phase"].reindex(ad.obs["donor"].astype(str)).to_numpy(float)
                cond = (ad.obs["hardy"].astype(str) == "1").to_numpy()
                err = abs_err_h(theta, true)
                perr.append(dict(seed=seed, mode=mode, tissue=name, err_h=float(np.median(err)),
                                 err_cond0_h=float(np.median(err[~cond])), err_cond1_h=float(np.median(err[cond]))))
                Y = ad.layers["counts"]
                Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
                t1 = time.time()
                res = fit_gene_rhythms(Y, ad.obs["library_size"].to_numpy(float), theta, glm_covariates(ad.obs),
                                       genes=list(ad.var_names))
                res = add_qvalues(res)
                res.to_csv(args.outdir / "glm" / f"s{seed}_{mode}_{name}.csv.gz", index=False)
                est.append(long_estimates(res, seed, mode, name, args.min_amp))
                print(f"  {mode} {name}: n={ad.n_obs}, phase error {perr[-1]['err_h']:.2f} h, "
                      f"GLM {time.time() - t1:.0f} s, not converged {(~res['converged']).sum()}", flush=True)
    est = pd.concat(est, ignore_index=True)
    est.to_csv(args.outdir / "estimates_long.csv.gz", index=False)
    pe = pd.DataFrame(perr)
    pe.to_csv(args.outdir / "phase_errors.csv", index=False)
    print(pe.groupby("mode")[["err_h", "err_cond0_h", "err_cond1_h"]].median().round(2).to_string(), flush=True)
    met = score(est, truth, args.min_amp)
    met.to_csv(args.outdir / "metrics.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(met.round(3).to_string(index=False), flush=True)
    scatter(est, truth, "log_amp_est", "log_amp_true", args.outdir / "scatter_log_amp.png", args.min_amp)
    scatter(est, truth, "phase_est_h", "phase_true_h", args.outdir / "scatter_phase.png", args.min_amp)


if __name__ == "__main__":
    main()
