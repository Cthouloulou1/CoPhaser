"""
Step 1 validation: frozen reference decoder -> phase recovery on simulated cells.

Cells are simulated from a reference clock (amp/phase table, see
cophaser.reference_clock) with known ground-truth phase; CoPhaser is trained with
the reference loaded into the rhythmic decoder and kept frozen, and the inferred
phase is compared to the truth.

With a frozen decoder the phase lives in the reference's frame (theta = 2*pi*t/period),
so no origin/direction alignment is done: a systematic offset in the report points
to a convention problem, not to noise.

Outputs (in --outdir): summary.json, losses.csv, phases.csv, gene_fits.csv and
figures (learning curves, clock faces, phase recovery, phase distribution).

Example (quick CPU smoke test):
    python scripts/step1_validate_phase_recovery.py --n-cells 500 --epochs 5 --outdir results/step1/smoke
Full run:
    python scripts/step1_validate_phase_recovery.py --n-cells 4000 --epochs 150
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import anndata
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack

from cophaser import simulation
from cophaser.cycle_configs import CYCLE_TRAINER_DEFAULTS
from cophaser.loss import Loss
from cophaser.model.CoPhaser import CoPhaser
from cophaser.model.decoder_prior import DecoderPrior
from cophaser.reference_clock import fourier_to_amp_phase, load_reference_clock
from cophaser import report_plots
from cophaser.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "reference" / "clock_reference_learned.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="reference clock (amp/phase table)")
    p.add_argument("--n-harm", type=int, default=2, help="harmonics used by the model and the simulation")
    p.add_argument("--n-cells", type=int, default=4000)
    p.add_argument("--n-decoy-genes", type=int, default=50, help="flat genes giving z something to encode")
    p.add_argument("--library-size", type=float, default=1e4, help="mean total counts per cell")
    p.add_argument("--dispersion", type=float, default=0.1, help="NB dispersion of clock genes")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--period", type=float, default=24.0, help="period in hours, for reporting")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z-input", choices=["raw", "cp10k"], default="raw", help="CoPhaser z encoder input")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=8, help="torch CPU threads (shared node: keep it modest)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="output folder (default: results/step1/<timestamp>)")
    return p.parse_args()


def set_seeds(seed):
    # simulation.* draws from the global numpy RNG
    np.random.seed(seed)
    torch.manual_seed(seed)


def simulate_from_reference(fourier, n_cells, n_decoy_genes, library_size, dispersion):
    counts, true_theta, _, _ = simulation.simulate_counts_rhythmic(
        fourier, n_cells=n_cells, alphas=dispersion, lambda_=1.0, library_size=library_size
    )
    blocks, var_names = [counts], list(fourier.index)
    if n_decoy_genes > 0:
        decoy_frac = np.random.uniform(1e-4, 5e-3, size=n_decoy_genes)
        blocks.append(
            simulation.simulate_counts(
                mean_fractions=np.tile(decoy_frac, (n_cells, 1)), alphas=0.2, library_size=library_size
            )
        )
        var_names += [f"DECOY_{i}" for i in range(n_decoy_genes)]
    X = hstack(blocks).tocsr()
    obs = pd.DataFrame({"true_theta": true_theta}, index=[f"cell_{i}" for i in range(n_cells)])
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=var_names), obs=obs)
    adata.layers["counts"] = X.copy()
    return adata


def oracle_phase(adata, fourier, dispersion, library_size, n_grid=240):
    """Maximum-likelihood phase per cell under the TRUE generative model (known Fourier
    coefficients, NB dispersion, library size, lambda=1). Its error is the floor any
    method can reach on these counts."""

    genes = list(fourier.index)
    X = adata[:, genes].layers["counts"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    harm = sorted({int(c.split("_")[1]) for c in fourier.columns if c.startswith("A_") and c != "A_0"})
    log_mu = np.repeat(fourier["A_0"].to_numpy()[:, None], n_grid, axis=1)
    for h in harm:
        log_mu += np.outer(fourier[f"A_{h}"], np.cos(h * grid)) + np.outer(fourier[f"B_{h}"], np.sin(h * grid))
    mu = np.exp(log_mu) * library_size  # (genes, grid)
    r = 1.0 / dispersion
    # NB log-likelihood up to terms constant in theta: x*log(mu) - (x + r)*log(mu + r)
    ll = X @ np.log(mu) - np.einsum("cg,gt->ct", X + r, np.log(mu + r))
    return grid[np.argmax(ll, axis=1)]


def phase_report(theta_pred, theta_true, period):
    """Errors in hours. 'offset' is the circular mean of (pred - true)."""
    diff = np.angle(np.exp(1j * (theta_pred - theta_true)))
    to_h = period / (2 * np.pi)
    offset = np.angle(np.mean(np.exp(1j * diff)))
    diff_centered = np.angle(np.exp(1j * (diff - offset)))
    # same-direction vs reversed-direction agreement (|mean resultant length|)
    r_same = np.abs(np.mean(np.exp(1j * (theta_pred - theta_true))))
    r_reversed = np.abs(np.mean(np.exp(1j * (theta_pred + theta_true))))
    return {
        "median_abs_error_h": float(np.median(np.abs(diff)) * to_h),
        "mean_abs_error_h": float(np.mean(np.abs(diff)) * to_h),
        "systematic_offset_h": float(offset * to_h),
        "median_abs_error_after_offset_h": float(np.median(np.abs(diff_centered)) * to_h),
        "resultant_same_direction": float(r_same),
        "resultant_reversed_direction": float(r_reversed),
    }


def gene_fits(adata, genes, theta_pred, theta_true, fourier):
    """Reference amp/phase vs cosinor fits of log1p(CP10k) on inferred and on true phase."""
    X = adata[:, genes].layers["counts"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    lib = np.asarray(adata.layers["counts"].sum(1)).ravel()
    expr = np.log1p(X / lib[:, None] * 1e4)
    ref = fourier_to_amp_phase(fourier)
    fit_pred = report_plots.cosinor_fit(expr, theta_pred).set_index(pd.Index(genes))
    fit_true = report_plots.cosinor_fit(expr, theta_true).set_index(pd.Index(genes))
    out = pd.concat(
        {"reference": ref[["amp", "phase"]], "on_inferred_phase": fit_pred, "on_true_phase": fit_true}, axis=1
    )
    out.columns = ["_".join(c) for c in out.columns]
    return out


def save_figures(outdir, trainer, fits, genes, theta_pred, theta_true, report, period):
    figs = {}
    figs["learning_curves"] = report_plots.plot_learning_curves(trainer.losses_history)

    fig = plt.figure(figsize=(14, 4.8))
    ax1 = fig.add_subplot(1, 3, 1, projection="polar")
    report_plots.plot_clock_face(fits["reference_amp"], fits["reference_phase"], genes, ax=ax1, period=period,
                                 title="Reference clock (frozen decoder)\nradius = log-amplitude")
    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    report_plots.plot_clock_comparison(
        genes, fits["on_true_phase_amp"], fits["on_true_phase_phase"],
        fits["on_inferred_phase_amp"], fits["on_inferred_phase_phase"],
        "cosinor on true phase", "cosinor on inferred phase", ax=ax2, period=period,
        title="Data re-fitted on inferred phase\nradius = amplitude of log1p(CP10k)",
    )
    ax3 = fig.add_subplot(1, 3, 3)
    d_h = np.angle(np.exp(1j * (fits["on_inferred_phase_phase"] - fits["on_true_phase_phase"]))) * period / (2 * np.pi)
    order = np.argsort(fits["reference_phase"].to_numpy())
    # A gene's phase is only measurable if its counts carry rhythm (amplitude AND depth):
    # fade by the R^2 of the cosinor on the true phase.
    r2 = fits["on_true_phase_r2"].to_numpy()
    alpha = np.clip(r2 / np.nanmax(r2), 0.15, 1)
    for y, i in enumerate(order):
        ax3.barh(y, np.asarray(d_h)[i], color="#2a6f97", alpha=alpha[i])
    ax3.set_yticks(range(len(order)))
    ax3.set_yticklabels([f"{genes[i]}  (R² {r2[i]:.2f})" for i in order], fontsize=8)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set(xlabel="acrophase on inferred − on true phase (h)",
            title="Per-gene acrophase shift\n(faded = little rhythm in the counts, R² on true phase)")
    ax3.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    figs["clock_faces"] = fig

    figs["phase_recovery"] = report_plots.plot_phase_recovery(
        theta_true, theta_pred, period=period, offset=report["systematic_offset_h"]
    )
    figs["phase_distribution"] = report_plots.plot_phase_distribution(theta_pred, theta_true, period=period)
    for name, fig in figs.items():
        fig.savefig(outdir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return list(figs)


def main():
    args = parse_args()
    set_seeds(args.seed)
    torch.set_num_threads(args.threads)
    outdir = args.outdir or REPO_ROOT / "results" / "step1" / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    fourier = load_reference_clock(args.csv, n_harm=args.n_harm)
    genes = list(fourier.index)
    mean_counts = np.exp(fourier["A_0"]) * args.library_size
    print(f"{len(genes)} clock genes: {genes}")
    print(
        f"Mean counts/cell per clock gene at library size {args.library_size:g}: "
        f"min {mean_counts.min():.3f}, median {mean_counts.median():.3f}, max {mean_counts.max():.3f}"
    )

    adata = simulate_from_reference(fourier, args.n_cells, args.n_decoy_genes, args.library_size, args.dispersion)

    model_kwargs = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["model"])
    model_kwargs["n_harm"] = args.n_harm
    model_kwargs["z_input"] = args.z_input
    model = CoPhaser(rhythmic_gene_names=genes, context_genes=list(adata.var_names), **model_kwargs)

    genes_modified = DecoderPrior.load_fourier_coefficients_prior(fourier, model, freeze_defined_weights=True)
    print(f"Prior loaded and frozen for {int(genes_modified.sum())}/{len(genes)} clock genes.")
    frozen_before = model.rhythmic_decoder.fourier_coefficients.weight.detach().clone()[genes_modified]

    model.load_anndata(adata, layer_to_use="counts")

    trainer_kwargs = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"])
    # Keep the reference frozen for the whole run: this tests whether the encoder alone
    # can invert counts to the reference's phase.
    trainer_kwargs["unfreeze_epoch_layer"] = []
    trainer = Trainer(model, Loss.compute_loss, cycle="circadian", **trainer_kwargs)
    t0 = time.time()
    trainer.train_model(n_epochs=args.epochs, lr=args.lr, subsample=None, device=args.device, silent=False)
    train_seconds = time.time() - t0
    plt.close("all")  # Trainer.plot_losses draws its own figure

    frozen_after = model.rhythmic_decoder.fourier_coefficients.weight.detach().cpu()[genes_modified]
    max_drift = float((frozen_after - frozen_before.cpu()).abs().max())
    print(f"Max change of frozen reference weights during training: {max_drift:.2e}")

    _, space = model.get_outputs()
    theta_pred = space["theta"].detach().cpu().numpy()
    theta_true = adata.obs["true_theta"].to_numpy()

    report = phase_report(theta_pred, theta_true, args.period)
    theta_oracle = oracle_phase(adata, fourier, args.dispersion, args.library_size)
    oracle = phase_report(theta_oracle, theta_true, args.period)
    report["oracle_median_abs_error_h"] = oracle["median_abs_error_h"]
    report["oracle_mean_abs_error_h"] = oracle["mean_abs_error_h"]
    fits = gene_fits(adata, genes, theta_pred, theta_true, fourier)
    report["train_seconds"] = round(train_seconds, 1)
    report["frozen_weight_drift"] = max_drift
    report["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    report_plots.losses_history_frame(trainer.losses_history).to_csv(outdir / "losses.csv")
    pd.DataFrame({"true_theta": theta_true, "inferred_theta": theta_pred, "oracle_theta": theta_oracle}, index=adata.obs_names).to_csv(
        outdir / "phases.csv"
    )
    fits.to_csv(outdir / "gene_fits.csv")
    (outdir / "summary.json").write_text(json.dumps(report, indent=2))
    figures = save_figures(outdir, trainer, fits, genes, theta_pred, theta_true, report, args.period)

    print("\n" + "\n".join(f"{k:>34}: {v}" for k, v in report.items() if k != "args"))
    print("(paper's aorta benchmark: median 1.67 h, on real not simulated data)")
    print(f"\nOutputs in {outdir}: {', '.join(f + '.png' for f in figures)}, summary.json, losses.csv, "
          "phases.csv, gene_fits.csv")


if __name__ == "__main__":
    main()
