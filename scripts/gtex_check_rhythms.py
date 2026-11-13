"""
Sanity check of the GTEx tissue files against the CHIRAL phases (no CoPhaser model).

For each tissue: cosinor of log CPM of the clock genes on the CHIRAL phase
(log2(CPM + 1) ~ m + a cos(t) + b sin(t)), with a shuffled-phase control. If the data
and the CHIRAL join are right, the clock genes should be clearly rhythmic (R^2 well
above the shuffled control) with the canonical antiphase between ARNTL/NPAS2/CLOCK
and PER/NR1D/DBP/TEF.

Outputs in --outdir: rhythms.csv (tissue, gene, amp, peak_h, r2, r2_shuffled),
clock_faces.png (one clock face per tissue, CHIRAL time frame).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import anndata  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cophaser import report_plots  # noqa: E402
from cophaser.gene_sets import HUMAN_CIRCADIAN_GENE_SET  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
H = 24 / (2 * np.pi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tissue-dir", type=Path, default=REPO_ROOT / "data" / "gtex_v8" / "tissues")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "gtex" / "check_rhythms_chiral")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    paths = sorted(args.tissue_dir.glob("*.h5ad"))
    rows = []
    ncol = 4
    fig = plt.figure(figsize=(4.6 * ncol, 4.8 * int(np.ceil(len(paths) / ncol))))
    for k, path in enumerate(paths):
        ad = anndata.read_h5ad(path)
        ad = ad[ad.obs["chiral_ct"].notna().to_numpy()]
        genes = [g for g in HUMAN_CIRCADIAN_GENE_SET if g in ad.var_names]
        X = ad[:, genes].layers["counts"].toarray().astype(float)
        cpm = X / ad.obs["library_size"].to_numpy()[:, None] * 1e6
        y = np.log2(cpm + 1)
        t = ad.obs["chiral_ct"].to_numpy(float) / H
        fit = report_plots.cosinor_fit(y, t)
        shuffled = report_plots.cosinor_fit(y, rng.permutation(t))
        tissue = str(ad.obs["tissue"].iloc[0])
        for g, f, s in zip(genes, fit.itertuples(), shuffled.itertuples()):
            rows.append(dict(tissue=tissue, gene=g, n=ad.n_obs, amp_log2=f.amp, peak_h=f.phase * H,
                             r2=f.r2, r2_shuffled=s.r2))
        ax = fig.add_subplot(int(np.ceil(len(paths) / ncol)), ncol, k + 1, projection="polar")
        report_plots.plot_clock_face(fit["amp"], fit["phase"], genes, ax=ax,
                                     title=f"{tissue} (n={ad.n_obs})\ncosinor on CHIRAL phase, radius = log2 amp")
        print(f"{tissue:40s} n={ad.n_obs:4d}  median R2 {fit['r2'].median():.3f} "
              f"(shuffled {shuffled['r2'].median():.3f}), max R2 {fit['r2'].max():.2f} ({genes[int(fit['r2'].argmax())]})")
    fig.tight_layout()
    fig.savefig(args.outdir / "clock_faces.png", dpi=110, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(args.outdir / "rhythms.csv", index=False)
    print(f"Outputs in {args.outdir}")


if __name__ == "__main__":
    main()
