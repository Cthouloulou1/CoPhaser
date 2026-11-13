"""
Report of the step-5 GTEx sweep (scripts/gtex_amplitude_sweep.py): effects of age, sex and
death type on the mesor and the amplitude of the clock genes, and their consistency across
seeds and clocks.

Effects are point estimates (no confidence intervals before step 7). Units:
  age        per 10 years (the model's per-sd coefficient rescaled with the training sd)
  sex        female vs male
  hardy[k]   death type k vs the reference level (all donors: Hardy 0 = ventilator;
             fast-death subset: Hardy 1)
  mesor      log-mean change; log_amp: log-amplitude change (exp(.) = fold change)
  phase      peak delay in hours (positive = later), relative to the amplitude-weighted
             mean of the clock genes (step 6 sweeps)
Amplitude and phase effects are only reported for genes whose clock amplitude is >= --min-amp
(learned decoder, clock_genes.csv): gamma is not identifiable for a flat gene.

Outputs in the sweep directory: report_effects.csv (one row per tissue, subset,
covariate, kind, gene: mean over the runs, sd, number of runs, fraction of runs with the
sign of the mean), report_phase.csv (phase errors), heatmap_<subset>_<kind>_<covariate>.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=Path, default=REPO_ROOT / "results" / "gtex" / "exp11_amplitude")
    p.add_argument("--min-amp", type=float, default=0.15)
    return p.parse_args()


def age_sd(run_dir):
    ckpt = torch.load(run_dir / "model.pt", weights_only=False)
    spec = {s["key"]: s for s in ckpt["covariate_spec"]}
    return float(spec["age"]["sd"]) if "age" in spec else np.nan


def main():
    args = parse_args()
    eff = pd.read_csv(args.sweep / "effects_long.csv")
    summ = pd.read_csv(args.sweep / "summary.csv")

    # rescale age to "per 10 years" with each run's own training sd
    sds = {}
    for (t, c, s, seed), _ in eff.groupby(["tissue_file", "clock", "subset", "seed"]):
        sds[(t, c, s, seed)] = age_sd(args.sweep / t / f"{c}_{s}_seed{seed}" / t)
    is_age = eff["covariate"] == "age"
    run_sd = np.array([sds[k] for k in zip(eff["tissue_file"], eff["clock"], eff["subset"], eff["seed"])])
    eff.loc[is_age, "value"] = eff.loc[is_age, "value"] / run_sd[is_age.to_numpy()] * 10
    eff["covariate"] = eff["covariate"].replace({"age": "age (per 10 y)", "sex[female]": "sex (F vs M)"})

    # amplitude of each clock gene in each run's decoder
    amps = []
    for (t, c, s, seed), _ in eff.groupby(["tissue_file", "clock", "subset", "seed"]):
        cg = pd.read_csv(args.sweep / t / f"{c}_{s}_seed{seed}" / t / "clock_genes.csv", index_col=0)
        amps.append(pd.DataFrame({"gene": cg.index, "amp": cg["amp"].to_numpy(), "tissue_file": t, "clock": c,
                                  "subset": s, "seed": seed}))
    eff = eff.merge(pd.concat(amps), on=["gene", "tissue_file", "clock", "subset", "seed"], how="left")
    eff = eff[(eff["kind"] == "mesor") | (eff["amp"] >= args.min_amp)]
    is_phase = eff["kind"] == "phase"
    eff.loc[is_phase, "value"] = eff.loc[is_phase, "value"] * 24 / (2 * np.pi)

    g = eff.groupby(["tissue_file", "subset", "covariate", "kind", "gene"])["value"]
    rep = g.agg(mean="mean", sd="std", n_runs="count").reset_index()
    sign = eff.assign(s=np.sign(eff["value"])).merge(rep[["tissue_file", "subset", "covariate", "kind", "gene", "mean"]])
    sign["agree"] = sign["s"] == np.sign(sign["mean"])
    rep = rep.merge(sign.groupby(["tissue_file", "subset", "covariate", "kind", "gene"])["agree"].mean()
                    .rename("sign_agreement").reset_index())
    rep.to_csv(args.sweep / "report_effects.csv", index=False)

    cols = [c for c in summ.columns if c.endswith("aligned_h") or c in ("chiral_median_abs_error_h",)]
    summ.groupby(["tissue_file", "clock", "subset"])[cols].mean().round(2).to_csv(args.sweep / "report_phase.csv")

    for (subset, kind, cov), d in rep.groupby(["subset", "kind", "covariate"]):
        m = d.pivot(index="gene", columns="tissue_file", values="mean")
        a = d.pivot(index="gene", columns="tissue_file", values="sign_agreement").reindex_like(m)
        lim = np.nanmax(np.abs(m.to_numpy())) if np.isfinite(m.to_numpy()).any() else 1
        fig, ax = plt.subplots(figsize=(1.0 * m.shape[1] + 3, 0.38 * m.shape[0] + 1.6))
        im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m.iat[i, j]):
                    ax.text(j, i, f"{m.iat[i, j]:+.2f}" + ("" if a.iat[i, j] == 1 else "*"), ha="center",
                            va="center", fontsize=7)
        ax.set_xticks(range(m.shape[1]), [c.replace("_", " ") for c in m.columns], rotation=40, ha="right")
        ax.set_yticks(range(m.shape[0]), m.index)
        ax.set_title(f"{kind} effect of {cov} ({subset} donors)\nmean of seeds x clocks; * = sign not "
                     f"consistent across runs", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        safe = cov.split(" ")[0].replace("[", "").replace("]", "")
        fig.savefig(args.sweep / f"heatmap_{subset}_{kind}_{safe}.png", dpi=130)
        plt.close(fig)
    print(json.dumps({"effects_rows": len(rep), "outputs": str(args.sweep)}))


if __name__ == "__main__":
    main()
