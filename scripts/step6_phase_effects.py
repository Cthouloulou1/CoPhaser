"""
Step 6 validation: covariates shift the phase of each gene's rhythm, F_g(theta - delta_g . c).

Simulates a GTEx-like single-tissue dataset (one sample per donor) with KNOWN effects of
age, sex and condition on the mesor, the log-amplitude and the phase (peak delay) of the
clock genes, then trains CoPhaser with the reference clock frozen:

  amp_only        mesor + amplitude effects modelled, phase effects not
  amp_phase       mesor + amplitude + phase effects
  amp_phase_grid  same, the decoder sees the grid phase, re-estimated every 10 epochs
                  (the configuration of exp. 11)
  amp_phase_fixed same, but the grid phase is computed once before training and never
                  re-estimated (the effects cannot feed back on the phases; exp. 11b/12)

Only relative delays are identifiable (a delay common to all clock genes is a shift of
theta). The model centres its delays on the amplitude-weighted mean of the clock genes; the
truth is centred the same way (weights = amplitudes of the frozen clock) before comparison.

Reported per run: phase error (raw and after the best rotation/reflection, encoder and
grid), recovery of the phase effects (est = a + b * truth per covariate: slope, correlation,
intercept; in hours), recovery of the log-amplitude and mesor effects.

Outputs in --outdir: summary.csv, effects_<run>.csv, phase_recovery_<run>.png,
learning_curves_<run>.png.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cophaser import report_plots  # noqa: E402
from cophaser.cycle_configs import CYCLE_TRAINER_DEFAULTS  # noqa: E402
from cophaser.loss import Loss  # noqa: E402
from cophaser.model.CoPhaser import CoPhaser  # noqa: E402
from cophaser.model.decoder_prior import DecoderPrior  # noqa: E402
from cophaser.reference_clock import load_reference_clock  # noqa: E402
from cophaser.simulation_covariates import CovariateEffects, TissueSpec, sample_donors, simulate_with_covariates  # noqa: E402
from cophaser.trainer import Trainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "reference" / "clock_reference_learned.csv"
SIM_COVS = ["age_std", "sex", "condition"]
H = 24 / (2 * np.pi)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--n-harm", type=int, default=2)
    p.add_argument("--n-donors", type=int, default=900)
    p.add_argument("--n-background-genes", type=int, default=500)
    p.add_argument("--frac-affected", type=float, default=0.5)
    p.add_argument("--phase-sd-hours", type=float, default=1.5)
    p.add_argument("--log-amp-sd", type=float, default=0.3)
    p.add_argument("--mesor-sd", type=float, default=0.3)
    p.add_argument("--library-size", type=float, default=2e7)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--adam-beta2", type=float, default=0.99)
    p.add_argument("--runs", default="amp_only,amp_phase,amp_phase_grid")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def simulate(args, fourier, rng):
    donors = sample_donors(args.n_donors, rng=rng)
    clock = list(fourier.index)
    genes = clock + [f"BG_{i:04d}" for i in range(args.n_background_genes)]
    eff = CovariateEffects.zeros(genes, SIM_COVS)
    shape = (len(genes), len(SIM_COVS))
    eff.mesor.loc[:, :] = rng.normal(0, args.mesor_sd, size=shape) * (rng.random(shape) < args.frac_affected)
    cshape = (len(clock), len(SIM_COVS))
    eff.log_amp.loc[clock, :] = rng.normal(0, args.log_amp_sd, size=cshape) * (rng.random(cshape) < args.frac_affected)
    eff.phase.loc[clock, :] = (rng.normal(0, args.phase_sd_hours / H, size=cshape)
                               * (rng.random(cshape) < args.frac_affected))
    return simulate_with_covariates(
        fourier, donors, eff, covariate_columns=SIM_COVS, tissues=[TissueSpec("tissue_A", mean_sd=0.0)],
        n_background_genes=args.n_background_genes, library_size=args.library_size, rng=rng,
    )


def truth_in_model_units(table, model, genes):
    """Simulator effects per unit of the model's encoded covariate columns (long table)."""
    spec = {s["key"]: s for s in model.covariate_spec}
    t = table.loc[genes]
    out = pd.DataFrame({"age": t["age_std"] * spec["age"]["sd"] / 15.0, "sex[1]": t["sex"],
                        "condition[1]": t["condition"]})
    out.index.name = "gene"
    return out.reset_index().melt(id_vars="gene", var_name="covariate")


def centred_phase_truth(sim, model, clock):
    """True delays relative to the amplitude-weighted mean of the clock genes (the model's
    convention with the frozen reference clock)."""
    truth = truth_in_model_units(sim.effects.phase, model, clock)
    idx = [model.context_genes.index(g) for g in clock]
    amp = model.rhythmic_decoder.fourier_coefficients.weight.detach()[idx].norm(dim=1).numpy()
    w = pd.Series(amp / amp.sum(), index=clock)
    mean = truth.assign(w=truth["gene"].map(w)).groupby("covariate").apply(
        lambda d: float((d["w"] * d["value"]).sum()), include_groups=False)
    return truth.assign(value=truth["value"] - truth["covariate"].map(mean))


def calibration(truth, est):
    m = truth.merge(est, on=["gene", "covariate"], suffixes=("_true", "_est"))
    rows = {}
    for cov, g in m.groupby("covariate"):
        if g["value_true"].std() == 0:
            continue
        b, a = np.polyfit(g["value_true"], g["value_est"], 1)
        rows[cov] = dict(slope=float(b), intercept=float(a),
                         corr=float(np.corrcoef(g["value_true"], g["value_est"])[0, 1]))
    return rows, m


def phase_errors(theta_pred, theta_true):
    raw = np.median(np.abs(np.angle(np.exp(1j * (theta_pred - theta_true))))) * H
    best = np.inf
    for sign in (1, -1):
        d = sign * theta_pred - theta_true
        off = np.angle(np.mean(np.exp(1j * d)))
        best = min(best, np.median(np.abs(np.angle(np.exp(1j * (d - off))))) * H)
    return float(raw), float(best)


def run(name, args, fourier, sim, outdir):
    ad = sim.adata
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    clock = list(fourier.index)
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names),
                     **dict(CYCLE_TRAINER_DEFAULTS["circadian"]["model"], n_harm=args.n_harm))
    DecoderPrior.load_fourier_coefficients_prior(fourier, model, freeze_defined_weights=True)
    model.load_anndata(ad, layer_to_use="counts", covariate_keys=["age", "sex", "condition"],
                       categorical_covariates=["sex", "condition"], covariate_amplitude=True,
                       covariate_phase=name != "amp_only")
    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"], adam_betas=(0.9, args.adam_beta2),
              unfreeze_epoch_layer=[])
    if name in ("amp_phase_grid", "amp_phase_fixed"):
        gp = model.grid_phase(dispersion=0.05)
        ad.obs["grid_phase"], ad.obs["grid_kappa"] = gp["theta"], gp["kappa"]
        model.add_phase_prior(ad, "grid_phase", "grid_kappa")
        kw.update(decoder_phase_from_prior=True, phase_prior_anneal_epochs=10**6)
        if name == "amp_phase_grid":
            kw.update(phase_prior_refresh_every=10, phase_prior_refresh_kwargs=dict(dispersion=0.05))
    trainer = Trainer(model, Loss.compute_loss, cycle="circadian", **kw)
    t0 = time.time()
    trainer.train_model(n_epochs=args.epochs, lr=1e-2, subsample=None, device="cpu", silent=True)
    model.to("cpu")
    gen, space = model.get_outputs()
    truth_phase = ad.obs["true_phase"].to_numpy()
    raw, aligned = phase_errors(space["theta"].numpy(), truth_phase)
    graw, galigned = phase_errors(model.grid_phase(use_z=False)["theta"], truth_phase)
    row = dict(run=name, train_seconds=round(time.time() - t0, 1), phase_error_raw_h=raw,
               phase_error_aligned_h=aligned, grid_phase_error_raw_h=graw, grid_phase_error_aligned_h=galigned)

    est = model.covariate_effects()
    est.to_csv(outdir / f"effects_{name}.csv", index=False)
    truths = {"mesor": truth_in_model_units(sim.effects.mesor, model, list(ad.var_names)),
              "log_amp": truth_in_model_units(sim.effects.log_amp, model, clock),
              "phase": centred_phase_truth(sim, model, clock)}
    for kind, truth in truths.items():
        e = est[est["kind"] == kind].drop(columns="kind")
        if e.empty:
            continue
        cal, merged = calibration(truth, e[e["gene"].isin(truth["gene"].unique())])
        for cov, c in cal.items():
            for k, v in c.items():
                row[f"{kind}/{cov}/{k}"] = v * (H if kind == "phase" and k == "intercept" else 1)
        if kind == "phase":
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharex=True, sharey=True)
            for ax, (cov, g) in zip(axes, merged.groupby("covariate")):
                ax.scatter(g["value_true"] * H, g["value_est"] * H, s=18, color="#2a6f97")
                lim = max(0.5, (g[["value_true", "value_est"]].abs().to_numpy() * H).max()) * 1.1
                ax.plot([-lim, lim], [-lim, lim], ls="--", color="#c0392b", lw=1)
                ax.set(title=f"{cov}: slope {cal.get(cov, {}).get('slope', np.nan):.2f}",
                       xlabel="true relative delay (h)")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel("learned delay (h)")
            fig.suptitle(f"{name}: phase effects on the clock genes (relative to the clock)", y=1.02)
            fig.tight_layout()
            fig.savefig(outdir / f"phase_recovery_{name}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
    fig = report_plots.plot_learning_curves(trainer.losses_history)
    fig.savefig(outdir / f"learning_curves_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return row


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    outdir = args.outdir or REPO_ROOT / "results" / "step6" / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fourier = load_reference_clock(args.csv, n_harm=args.n_harm)
    sim = simulate(args, fourier, rng)
    sim.adata.obs["sex"] = sim.adata.obs["sex"].astype(int)
    n_ph = int((sim.effects.phase != 0).values.sum())
    print(f"{sim.adata.n_obs} samples, {sim.adata.n_vars} genes; {n_ph} non-zero phase effects", flush=True)
    rows = [run(name, args, fourier, sim, outdir) for name in args.runs.split(",")]
    pd.DataFrame(rows).set_index("run").to_csv(outdir / "summary.csv")
    print(f"\nOutputs in {outdir}")


if __name__ == "__main__":
    main()
