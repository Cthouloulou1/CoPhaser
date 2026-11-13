"""
Step 7 report: bootstrap confidence intervals of the covariate effects on the clock genes
(output of scripts/gtex_bootstrap.py).

For each tissue x covariate x kind x gene:
  estimate     median of the bootstrap replicates
  ci_low/high  percentile interval (default 95 %)
  p_boot       two-sided percentile p-value: 2 x min(fraction <= 0, fraction >= 0), at least
               1 / (number of replicates), capped at 1 (coarse with few replicates)
  p_norm       two-sided normal p-value from the bootstrap standard error, z = estimate / sd
               (usable with few replicates; assumes a roughly normal bootstrap distribution)
  q_bh         Benjamini-Hochberg adjusted p_norm, within each kind over all tissues, genes and
               the age / sex / death-type columns (missing-value columns excluded)
Units: age per 10 years (each replicate rescaled with its own training sd), sex female vs
male, hardy[k] vs ventilator (Hardy 0); mesor and log_amp in log units; phase in hours
(positive = later peak, relative to the amplitude-weighted mean of the clock genes).
log_amp and phase are only reported for genes whose clock amplitude is >= --min-amp.

Outputs in the bootstrap directory: bootstrap_effects.csv, bootstrap_significant.csv
(q <= --fdr), forest_<kind>_<covariate>.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
H = 24 / (2 * np.pi)
SHORT = {"adipose_subcutaneous": "adipose", "artery_tibial": "artery", "lung": "lung", "muscle_skeletal": "muscle",
         "nerve_tibial": "nerve", "skin_not_sun_exposed_suprapubic": "skin (not exposed)",
         "skin_sun_exposed_lower_leg": "skin (exposed)", "thyroid": "thyroid"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boot", type=Path, default=REPO_ROOT / "results" / "gtex" / "exp13_bootstrap")
    p.add_argument("--min-amp", type=float, default=0.15)
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--forest", nargs="*", default=["phase:sex (F vs M)", "log_amp:sex (F vs M)",
                                                   "phase:age (per 10 y)", "log_amp:age (per 10 y)"])
    return p.parse_args()


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def summarise(eff, level):
    a = (1 - level) / 2

    def one(v):
        v = v.to_numpy()
        n = len(v)
        p = min(1.0, max(2 * min((v <= 0).mean(), (v >= 0).mean()), 1 / n))
        sd = v.std(ddof=1) if n > 1 else np.nan
        p_norm = float(2 * norm.sf(abs(np.median(v)) / sd)) if sd > 0 else np.nan
        return pd.Series({"estimate": np.median(v), "ci_low": np.quantile(v, a), "ci_high": np.quantile(v, 1 - a),
                          "sd": sd, "n_rep": n, "p_boot": p, "p_norm": p_norm})

    keys = ["tissue_file", "covariate", "kind", "gene"]
    return eff.groupby(keys)["value"].apply(one).unstack().reset_index()


def forest(rep, kind, cov, path):
    d = rep[(rep["kind"] == kind) & (rep["covariate"] == cov)]
    if d.empty:
        return
    genes = sorted(d["gene"].unique())
    tissues = sorted(d["tissue_file"].unique())
    ncol = 4
    nrow = int(np.ceil(len(genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 0.32 * len(tissues) * nrow + 1.2 * nrow),
                             sharex=True, squeeze=False)
    for ax, g in zip(axes.flat, genes):
        x = d[d["gene"] == g].set_index("tissue_file").reindex(tissues)
        y = np.arange(len(tissues))
        sig = x["q_bh"] <= 0.05
        ax.errorbar(x["estimate"], y, xerr=[x["estimate"] - x["ci_low"], x["ci_high"] - x["estimate"]], fmt="none",
                    ecolor=np.where(sig, "#1f5f8b", "#9aa5ad").tolist() if len(y) else None, elinewidth=1.4)
        ax.scatter(x["estimate"], y, c=np.where(sig, "#1f5f8b", "#9aa5ad"), s=16, zorder=3)
        ax.axvline(0, color="#c0392b", lw=0.8, ls="--")
        ax.set_yticks(y, [SHORT.get(t, t) for t in tissues], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(g, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    for ax in list(axes.flat)[len(genes):]:
        ax.axis("off")
    unit = "hours (+ = later peak)" if kind == "phase" else "log-amplitude" if kind == "log_amp" else "log-mean"
    fig.suptitle(f"{kind} effect of {cov}: bootstrap median and 95 % interval ({unit});\n"
                 f"blue = significant after Benjamini-Hochberg (q <= 0.05)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    eff = pd.read_csv(args.boot / "effects_long.csv")
    is_age = eff["covariate"] == "age"
    eff.loc[is_age, "value"] = eff.loc[is_age, "value"] / eff.loc[is_age, "age_sd"] * 10
    eff.loc[eff["kind"] == "phase", "value"] *= H
    eff["covariate"] = eff["covariate"].replace({"age": "age (per 10 y)", "sex[female]": "sex (F vs M)"})
    eff = eff[~eff["covariate"].str.endswith("__missing")]
    eff = eff[(eff["kind"] == "mesor") | (eff["amp"] >= args.min_amp)]
    rep = summarise(eff, args.level)
    rep["q_bh"] = np.nan
    for kind, idx in rep.groupby("kind").groups.items():
        rep.loc[idx, "q_bh"] = bh(rep.loc[idx, "p_norm"].fillna(1.0))
    rep["significant"] = rep["q_bh"] <= args.fdr
    rep.to_csv(args.boot / "bootstrap_effects.csv", index=False)
    sig = rep[rep["significant"]].sort_values(["kind", "covariate", "gene", "tissue_file"])
    sig.to_csv(args.boot / "bootstrap_significant.csv", index=False)
    for spec in args.forest:
        kind, cov = spec.split(":", 1)
        forest(rep, kind, cov, args.boot / f"forest_{kind}_{cov.split(' ')[0]}.png")
    print(json.dumps({"effects": len(rep), "replicates": int(rep["n_rep"].max()),
                      "significant": {f"{k} | {c}": int(n) for (k, c), n in
                                      sig.groupby(["kind", "covariate"]).size().items()}}))


if __name__ == "__main__":
    main()
