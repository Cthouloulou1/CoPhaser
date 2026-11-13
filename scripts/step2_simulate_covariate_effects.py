"""
Step 2: GTEx-like simulation with known age / sex / condition effects on the clock.

Generates donors (age, sex, condition, one phase per donor), several tissues per
donor (tissue-specific clock shift and amplitude), and counts whose mesor, amplitude
and timing depend on the covariates gene by gene. Effects = sparse random effects
plus optional systematic trends (age dampening, sex amplitude difference).

Then checks, with a cosinor on the TRUE phase (an oracle that later steps will not
have), that the effects are visible in the counts, and draws:
  clock_faces_by_group.png  clock faces female/male, young/old, condition 1/0
  effect_recovery.png       injected vs group-cosinor estimate, per covariate and kind
  design.png                donors: age, sex, condition, phase; samples per tissue

Outputs: simulated.h5ad, effects_true.csv, effects_cosinor.csv, donors.csv, summary.json.

Example:
    python scripts/step2_simulate_covariate_effects.py --n-donors 900 --outdir results/step2/gtex_like
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

from cophaser import report_plots  # noqa: E402
from cophaser.reference_clock import load_reference_clock  # noqa: E402
from cophaser.simulation_covariates import (  # noqa: E402
    TissueSpec,
    cosinor_by_group,
    random_effects,
    sample_donors,
    simulate_with_covariates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "reference" / "clock_reference_learned.csv"
COVARIATES = ["age_std", "sex", "condition"]
H = 24 / (2 * np.pi)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--n-harm", type=int, default=2)
    p.add_argument("--n-donors", type=int, default=900)
    p.add_argument("--n-background-genes", type=int, default=500)
    p.add_argument("--library-size", type=float, default=2e7, help="bulk-like depth by default")
    p.add_argument("--dispersion", type=float, default=0.05)
    p.add_argument("--frac-affected", type=float, default=0.3, help="fraction of (gene, covariate, kind) with a random effect")
    p.add_argument("--mesor-sd", type=float, default=0.3)
    p.add_argument("--log-amp-sd", type=float, default=0.3)
    p.add_argument("--phase-sd-hours", type=float, default=1.5)
    p.add_argument("--age-dampening", type=float, default=-0.2,
                   help="log-amplitude change per 15 years added to all clock genes (Talamanca 2023: dampening)")
    p.add_argument("--female-amp", type=float, default=0.15,
                   help="log-amplitude change for females added to all clock genes (Talamanca 2023: more sustained)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def default_tissues():
    # three tissues with different clock timing/strength and incomplete sampling, GTEx-like
    return [
        TissueSpec("tissue_A", phase_shift=0.0, amp_scale=1.0, p_sampled=0.9),
        TissueSpec("tissue_B", phase_shift=1.5 / H, amp_scale=0.7, p_sampled=0.6),
        TissueSpec("tissue_C", phase_shift=-1.0 / H, amp_scale=1.2, p_sampled=0.5),
    ]


def estimate_effects_by_groups(adata, genes, tissues):
    """Group-cosinor estimates on the TRUE phase, within tissue then averaged.

    sex / condition: group difference; age: (old - young) / (mean age_std difference).
    Returns the long format of CovariateEffects.long().
    """
    obs = adata.obs
    a = obs["age_std"].to_numpy()
    contrasts = {
        "sex": (obs["sex"].to_numpy(), 1, 0, 1.0),
        "condition": (obs["condition"].to_numpy(), 1, 0, 1.0),
    }
    age_group = np.where(a < -0.7, "young", np.where(a > 0.7, "old", "mid"))
    d_age = a[age_group == "old"].mean() - a[age_group == "young"].mean()
    contrasts["age_std"] = (age_group, "old", "young", d_age)

    rows = []
    for cov, (groups, hi, lo, scale) in contrasts.items():
        per_tissue = []
        for t in tissues:
            m = (obs["tissue"] == t).to_numpy()
            fit = cosinor_by_group(adata[m], genes, groups[m])
            f_hi = fit[fit["group"] == hi].set_index("gene")
            f_lo = fit[fit["group"] == lo].set_index("gene")
            per_tissue.append(
                pd.DataFrame(
                    {
                        "mesor": (f_hi["mesor"] - f_lo["mesor"]) / scale,
                        "log_amp": np.log(f_hi["amp"] / f_lo["amp"]) / scale,
                        "phase_x": np.cos(f_hi["phase"] - f_lo["phase"]),
                        "phase_y": np.sin(f_hi["phase"] - f_lo["phase"]),
                    }
                )
            )
        mean = sum(per_tissue) / len(per_tissue)
        mean["phase"] = np.arctan2(mean["phase_y"], mean["phase_x"]) / scale
        for kind in ("mesor", "log_amp", "phase"):
            rows.append(pd.DataFrame({"gene": mean.index, "covariate": cov, "kind": kind, "value": mean[kind].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def plot_design(res, outdir):
    d = res.donors
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    axes[0].hist(d["age"], bins=np.arange(20, 81, 5), color="#2a6f97")
    axes[0].set(title="Age (years)", xlabel="age")
    counts = d.groupby(["sex", "condition"]).size().unstack(fill_value=0)
    counts.index = ["male" if i == 0 else "female" for i in counts.index]
    counts.plot.bar(ax=axes[1], color=["#9fb8c8", "#c0392b"], rot=0)
    axes[1].set(title="Donors by sex and condition", ylabel="donors")
    axes[1].legend(title="condition", fontsize=8, frameon=False)
    axes[2].hist(np.mod(d["true_phase"], 2 * np.pi) * H, bins=24, color="#2a6f97")
    axes[2].set(title="Donor phase (time of death, h)", xlabel="h")
    res.adata.obs["tissue"].value_counts().sort_index().plot.bar(ax=axes[3], color="#2a6f97", rot=0)
    axes[3].set(title="Samples per tissue")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "design.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_group_clocks(res, genes, outdir):
    ad = res.adata[(res.adata.obs["tissue"] == "tissue_A").to_numpy()]
    a = ad.obs["age_std"].to_numpy()
    panels = [
        ("sex", ad.obs["sex"].to_numpy(), 0, 1, "male", "female"),
        ("age", np.where(a < -0.7, "young", np.where(a > 0.7, "old", "mid")), "young", "old", "< 40 y", "> 60 y"),
        ("condition", ad.obs["condition"].to_numpy(), 0, 1, "condition 0", "condition 1"),
    ]
    fig = plt.figure(figsize=(16, 5.4))
    for k, (name, groups, lo, hi, l_lo, l_hi) in enumerate(panels):
        fit = cosinor_by_group(ad, genes, groups)
        f_lo = fit[fit["group"] == lo].set_index("gene").loc[genes]
        f_hi = fit[fit["group"] == hi].set_index("gene").loc[genes]
        ax = fig.add_subplot(1, 3, k + 1, projection="polar")
        report_plots.plot_clock_comparison(
            genes, f_lo["amp"], f_lo["phase"], f_hi["amp"], f_hi["phase"], l_lo, l_hi, ax=ax,
            title=f"{name}: cosinor on true phase (tissue_A)\nradius = log-amplitude",
        )
    fig.tight_layout()
    fig.savefig(outdir / "clock_faces_by_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = args.outdir or REPO_ROOT / "results" / "step2" / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    fourier = load_reference_clock(args.csv, n_harm=args.n_harm)
    genes = list(fourier.index)
    donors = sample_donors(args.n_donors, rng=rng)

    effects = random_effects(
        genes, COVARIATES, frac_affected=args.frac_affected, mesor_sd=args.mesor_sd,
        log_amp_sd=args.log_amp_sd, phase_sd_hours=args.phase_sd_hours, rng=rng,
    )
    effects.log_amp["age_std"] += args.age_dampening
    effects.log_amp["sex"] += args.female_amp

    tissues = default_tissues()
    res = simulate_with_covariates(
        fourier, donors, effects, covariate_columns=COVARIATES, tissues=tissues,
        n_background_genes=args.n_background_genes, library_size=args.library_size,
        dispersion=args.dispersion, rng=rng,
    )
    ad = res.adata
    print(f"{ad.n_obs} samples ({args.n_donors} donors x {len(tissues)} tissues, incomplete), {ad.n_vars} genes")
    clock_counts = np.asarray(ad[:, genes].layers["counts"].mean(0)).ravel()
    print(f"Mean counts per clock gene and sample: min {clock_counts.min():.0f}, median {np.median(clock_counts):.0f}")

    truth = effects.long()
    estimate = estimate_effects_by_groups(ad, genes, [t.name for t in tissues])

    # agreement summary per (covariate, kind)
    m = truth.merge(estimate, on=["gene", "covariate", "kind"], suffixes=("_true", "_est"))
    ph = m["kind"] == "phase"
    m.loc[ph, ["value_true", "value_est"]] = m.loc[ph, ["value_true", "value_est"]] * H
    summary = {
        f"{c}/{k}": {
            "corr": float(np.corrcoef(g["value_true"], g["value_est"])[0, 1]) if g["value_true"].std() > 0 else None,
            "median_abs_error": float(np.median(np.abs(g["value_true"] - g["value_est"]))),
            "units": "hours" if k == "phase" else "log",
        }
        for (c, k), g in m.groupby(["covariate", "kind"])
    }
    for key, v in summary.items():
        corr = "  n/a" if v["corr"] is None else f"{v['corr']:.2f}"
        print(f"{key:>20}: corr {corr}, median |error| {v['median_abs_error']:.3f} {v['units']}")

    ad.uns["simulation"] = {k: (str(v) if not isinstance(v, (int, float, str)) else v) for k, v in res.params.items()}
    ad.write_h5ad(outdir / "simulated.h5ad")
    truth.to_csv(outdir / "effects_true.csv", index=False)
    estimate.to_csv(outdir / "effects_cosinor.csv", index=False)
    donors.to_csv(outdir / "donors.csv")
    (outdir / "summary.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                    "n_samples": ad.n_obs, "recovery": summary}, indent=2))

    plot_design(res, outdir)
    plot_group_clocks(res, genes, outdir)
    fig = report_plots.plot_effect_recovery(
        truth, estimate, title="Injected effects vs group cosinor on the TRUE phase (oracle check)"
    )
    fig.savefig(outdir / "effect_recovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nOutputs in {outdir}")


if __name__ == "__main__":
    main()
