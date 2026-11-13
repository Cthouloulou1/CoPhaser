"""
Step 4 validation: linear covariate mean shift W_c . c.

Simulates a GTEx-like single-tissue dataset with KNOWN mesor effects of age, sex and
condition (on clock genes and on a fraction of background genes; no amplitude or
phase effects), then trains CoPhaser in four configurations:

  frozen_nocov / frozen_cov   reference decoder frozen, without / with covariates
  default_nocov / default_cov circadian defaults (decoder unfrozen at epoch 10)

and reports, for the runs with covariates:
  - recovery of the effects: est = a + b * truth per covariate. The intercept a is
    expected (normalising by total counts makes effects identifiable only up to a
    constant per covariate); b < 1 means shrinkage, e.g. leakage of c into z.
  - leakage: R^2 of a linear prediction of each covariate column from z.
and, for all runs, the phase error (raw, and after the best rotation/reflection for
the runs where the decoder is free to move).

Outputs in --outdir: summary.csv, effects_<run>.csv, effect_recovery_<run>.png,
learning_curves_<run>.png, phase_error.png.
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
from cophaser.simulation_covariates import (  # noqa: E402
    CovariateEffects,
    TissueSpec,
    sample_donors,
    simulate_with_covariates,
)
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
    p.add_argument("--frac-affected", type=float, default=0.3)
    p.add_argument("--mesor-sd", type=float, default=0.3)
    p.add_argument("--library-size", type=float, default=2e7)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--z-input", choices=["raw", "cp10k"], default="raw", help="CoPhaser z encoder input")
    p.add_argument("--adam-beta2", type=float, default=0.99,
                   help="0.99 for deep (bulk) data; torch default 0.999 trains far too slowly here")
    p.add_argument("--runs", default="frozen_nocov,frozen_cov,default_nocov,default_cov")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def simulate(args, fourier, rng):
    donors = sample_donors(args.n_donors, rng=rng)
    genes = list(fourier.index) + [f"BG_{i:04d}" for i in range(args.n_background_genes)]
    eff = CovariateEffects.zeros(genes, SIM_COVS)
    mask = rng.random((len(genes), len(SIM_COVS))) < args.frac_affected
    eff.mesor.loc[:, :] = rng.normal(0, args.mesor_sd, size=mask.shape) * mask
    res = simulate_with_covariates(
        fourier, donors, eff, covariate_columns=SIM_COVS, tissues=[TissueSpec("tissue_A", mean_sd=0.0)],
        n_background_genes=args.n_background_genes, library_size=args.library_size, rng=rng,
    )
    return res


def truth_in_model_units(res, model) -> pd.DataFrame:
    """Simulator effects expressed per unit of the model's encoded covariate columns."""
    spec = {s["key"]: s for s in model.covariate_spec}
    t = res.effects.mesor
    cols = {
        "age": t["age_std"] * spec["age"]["sd"] / 15.0,  # simulator: per 15 y; model: per sd of age
        "sex[1]": t["sex"],
        "condition[1]": t["condition"],
    }
    out = pd.DataFrame(cols)
    out.index.name = "gene"
    long = out.reset_index().melt(id_vars="gene", var_name="covariate")
    long["kind"] = "mesor"
    return long[["gene", "covariate", "kind", "value"]]


def calibration(truth, est):
    m = truth.merge(est, on=["gene", "covariate", "kind"], suffixes=("_true", "_est"))
    rows = {}
    for cov, g in m.groupby("covariate"):
        b, a = np.polyfit(g["value_true"], g["value_est"], 1)
        rows[cov] = dict(
            intercept=a, slope=b, corr=np.corrcoef(g["value_true"], g["value_est"])[0, 1],
            median_abs_error_centered=np.median(np.abs(g["value_est"] - a - g["value_true"])),
        )
    return rows


def leakage_r2(z, C, names):
    Z = np.column_stack([np.ones(len(z)), z])
    out = {}
    for j, name in enumerate(names):
        y = C[:, j]
        coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
        out[name] = 1 - ((y - Z @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return out


def phase_errors(theta_pred, theta_true):
    """Median |error| in hours, raw and after the best rotation (and reflection)."""
    raw = np.median(np.abs(np.angle(np.exp(1j * (theta_pred - theta_true))))) * H
    best = np.inf
    for sign in (1, -1):
        d = sign * theta_pred - theta_true
        off = np.angle(np.mean(np.exp(1j * d)))
        best = min(best, np.median(np.abs(np.angle(np.exp(1j * (d - off))))) * H)
    return raw, best


def run(name, args, fourier, ad, outdir):
    frozen, with_cov = name.startswith("frozen"), name.endswith("_cov")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    genes = list(fourier.index)
    model_kwargs = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["model"], n_harm=args.n_harm, z_input=args.z_input)
    model = CoPhaser(rhythmic_gene_names=genes, context_genes=list(ad.var_names), **model_kwargs)
    DecoderPrior.load_fourier_coefficients_prior(fourier, model, freeze_defined_weights=True)
    model.load_anndata(
        ad, layer_to_use="counts",
        covariate_keys=["age", "sex", "condition"] if with_cov else None,
        categorical_covariates=["sex", "condition"] if with_cov else None,
    )
    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"], adam_betas=(0.9, args.adam_beta2))
    if frozen:
        kw["unfreeze_epoch_layer"] = []
    trainer = Trainer(model, Loss.compute_loss, cycle="circadian", **kw)
    t0 = time.time()
    trainer.train_model(n_epochs=args.epochs, lr=1e-2, subsample=None, device="cpu", silent=True)
    seconds = time.time() - t0
    model.to("cpu")

    _, space = model.get_outputs()
    theta = space["theta"].numpy()
    raw, aligned = phase_errors(theta, ad.obs["true_phase"].to_numpy())
    row = dict(run=name, train_seconds=round(seconds, 1), phase_median_error_h=raw,
               phase_median_error_aligned_h=aligned)

    fig = report_plots.plot_learning_curves(trainer.losses_history)
    fig.suptitle(name, y=1.01)
    fig.savefig(outdir / f"learning_curves_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    if with_cov:
        est = model.covariate_effects()
        truth = truth_in_model_units(SIM_RESULT, model)
        est.to_csv(outdir / f"effects_{name}.csv", index=False)
        cal = calibration(truth, est)
        leak = leakage_r2(space["z"].numpy(), model.covariates.numpy(), model.covariate_names)
        for cov, c in cal.items():
            for k, v in c.items():
                row[f"{cov}/{k}"] = v
            row[f"{cov}/leak_R2_from_z"] = leak[cov]
        # centre the estimates by the fitted intercept (identifiable up to a constant)
        est_c = est.copy()
        est_c["value"] = est_c["value"] - est_c["covariate"].map({c: v["intercept"] for c, v in cal.items()})
        fig = report_plots.plot_effect_recovery(
            truth, est_c, title=f"{name}: injected vs learned W_c (centred per covariate)"
        )
        fig.savefig(outdir / f"effect_recovery_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
    return row


SIM_RESULT = None


def main():
    global SIM_RESULT
    args = parse_args()
    torch.set_num_threads(args.threads)
    outdir = args.outdir or REPO_ROOT / "results" / "step4" / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fourier = load_reference_clock(args.csv, n_harm=args.n_harm)
    SIM_RESULT = simulate(args, fourier, rng)
    ad = SIM_RESULT.adata
    ad.obs["sex"] = ad.obs["sex"].astype(int)
    print(f"{ad.n_obs} samples, {ad.n_vars} genes; "
          f"{int((SIM_RESULT.effects.mesor != 0).values.sum())} non-zero mesor effects")

    rows = [run(name, args, fourier, ad, outdir) for name in args.runs.split(",")]
    summary = pd.DataFrame(rows).set_index("run")
    summary.to_csv(outdir / "summary.csv")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    summary[["phase_median_error_h", "phase_median_error_aligned_h"]].plot.bar(ax=ax, rot=0, color=["#2a6f97", "#9fb8c8"])
    ax.set(ylabel="median |phase error| (h)", title="Phase recovery with / without covariates")
    ax.legend(["raw", "after best rotation/reflection"], fontsize=8, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "phase_error.png", dpi=130)
    plt.close(fig)
    print(f"\nOutputs in {outdir}")


if __name__ == "__main__":
    main()
