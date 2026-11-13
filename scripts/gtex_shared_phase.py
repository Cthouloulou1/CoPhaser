"""
Step 9: one circadian phase per donor, shared by all its tissues (as in CHIRAL).

With a frozen clock per tissue, the grid log-likelihood of a donor's phase is the sum of
the grid log-likelihoods of its samples (one per tissue; tissues are independent given the
phase). The clocks of all tissues must be in the same time frame: the human clocks fitted on
the CHIRAL phases are (reference/human_chiral/<tissue>.csv), and so is the mouse example
clock (the same table for every tissue, tissue-specific offsets ignored); the clocks learned
per tissue (exp. 10) are not.

Outputs in --outdir:
  donor_phases.csv        donor, theta, kappa, resultant, n_tissues, tissues: shared phases
                          with the full human clocks, all donors (for gtex_train.py
                          --phase-table)
  donor_phases_mouse.csv  same with the mouse example clock (comparison only)
  evaluation_by_tissue.csv, evaluation_by_ntissues.csv, shared_phase_errors.png:
      honest evaluation. Human clocks are refitted (cosinor on CHIRAL) on a random half of
      the donors, drawn once for all tissues (--split-seed); per-tissue grid phases and
      shared phases of the other half are compared with CHIRAL (raw: same frame; aligned:
      after the best rotation / reflection). Still optimistic: CHIRAL estimated its phases
      from these samples (and more tissues).

Example:
  python scripts/gtex_shared_phase.py --outdir results/gtex/exp14_shared_phase
"""
import argparse
import json
import sys
from pathlib import Path

import anndata
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from cophaser.gene_sets import HUMAN_CIRCADIAN_GENE_SET  # noqa: E402
from cophaser.model.CoPhaser import CoPhaser  # noqa: E402
from cophaser.model.decoder_prior import DecoderPrior  # noqa: E402
from cophaser.reference_clock import load_reference_clock  # noqa: E402

H = 24 / (2 * np.pi)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tissues", nargs="*", default=None)
    p.add_argument("--tissue-dir", type=Path, default=REPO_ROOT / "data" / "gtex_v8" / "tissues")
    p.add_argument("--human-dir", type=Path, default=REPO_ROOT / "reference" / "human_chiral")
    p.add_argument("--mouse", type=Path, default=REPO_ROOT / "reference" / "clock_reference_learned.csv")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "gtex" / "exp14_shared_phase")
    p.add_argument("--dispersion", type=float, default=0.1, help="as gtex_train.py --grid-dispersion")
    p.add_argument("--grid-points", type=int, default=96)
    p.add_argument("--kappa-max", type=float, default=50.0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args()


def tissue_loglik(ad, fourier, clock, dispersion, n_points):
    """Grid log-likelihoods (n_samples, n_points) of the samples of one tissue under a frozen
    clock (rhythmic genes only, no covariates: as gtex_train.py before training)."""
    torch.manual_seed(0)
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=clock, n_latent=2, n_harm=1)
    DecoderPrior.load_fourier_coefficients_prior(fourier, model, freeze_defined_weights=True)
    model.load_anndata(ad[:, clock].copy(), layer_to_use="counts", library_size_field="library_size")
    g = model.grid_phase(n_points=n_points, dispersion=dispersion, return_loglik=True)
    return g["loglik"], g["grid"]


def shared_phases(donors, logliks, grid, tissues=None, kappa_max=50.0):
    """Sum the grid log-likelihoods of the samples of each donor and summarise.

    donors: list of per-tissue arrays of donor ids; logliks: matching list of
    (n_samples, n_points) arrays; tissues: optional names, listed per donor.
    Returns a DataFrame indexed by donor: theta, kappa, resultant, n_tissues, tissues."""
    d = np.concatenate([np.asarray(x).astype(str) for x in donors])
    ll = np.concatenate(logliks, axis=0)
    t = np.concatenate([np.full(len(x), name) for x, name in
                        zip(donors, tissues or [str(i) for i in range(len(donors))])])
    codes, uniq = pd.factorize(d)
    summed = np.zeros((len(uniq), ll.shape[1]))
    np.add.at(summed, codes, ll)
    s = CoPhaser.phase_from_loglik(summed, grid, kappa_max=kappa_max)
    out = pd.DataFrame({"donor": np.asarray(uniq), **s})
    tis = pd.Series(t).groupby(codes).agg(lambda x: ";".join(sorted(x)))
    out["n_tissues"] = np.bincount(codes)
    out["tissues"] = tis.sort_index().to_numpy()
    return out.set_index("donor")


def errors(theta, chiral_h):
    from gtex_train import circular_compare

    m = np.isfinite(chiral_h) & np.isfinite(theta)
    raw = float(np.median(np.abs(np.angle(np.exp(1j * (theta[m] - chiral_h[m] / H))))) * H)
    return raw, float(circular_compare(np.where(m, theta, np.nan), np.where(m, chiral_h, np.nan))["median_abs_error_h"]), int(m.sum())


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    from gtex_human_clock import fit_cosinor_clock

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "half_clocks").mkdir(exist_ok=True)
    tissues = args.tissues or sorted(p.stem for p in args.tissue_dir.glob("*.h5ad"))
    mouse = load_reference_clock(args.mouse, rename_map={"Bmal1": "ARNTL"}, n_harm=1)
    mouse.index = mouse.index.str.upper()

    data = {}
    for t in tissues:
        ad = anndata.read_h5ad(args.tissue_dir / f"{t}.h5ad")
        clock = [g for g in HUMAN_CIRCADIAN_GENE_SET if g in ad.var_names]
        data[t] = (ad[:, clock].copy(), clock)
    all_donors = np.unique(np.concatenate([d.obs["donor"].astype(str).to_numpy() for d, _ in data.values()]))
    rng = np.random.default_rng(args.split_seed)
    train_donors = set(all_donors[rng.random(len(all_donors)) < 0.5])  # one split for every tissue

    per = {k: {"donors": [], "ll": [], "names": []} for k in ("human", "mouse", "half")}
    by_tissue, sample_rows, grid = [], [], None
    for t, (ad, clock) in data.items():
        donors = ad.obs["donor"].astype(str).to_numpy()
        chiral = ad.obs["chiral_ct"].to_numpy(float)
        has = np.isfinite(chiral)
        train = has & np.isin(donors, list(train_donors))
        X = ad.layers["counts"]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        half = fit_cosinor_clock(X[train], ad.obs["library_size"].to_numpy(float)[train], chiral[train] / H, clock)
        half.to_csv(args.outdir / "half_clocks" / f"{t}.csv", index=False)
        clocks = {"human": load_reference_clock(args.human_dir / f"{t}.csv", n_harm=1),
                  "mouse": mouse,
                  "half": load_reference_clock(args.outdir / "half_clocks" / f"{t}.csv", n_harm=1)}
        for k, fourier in clocks.items():
            ll, grid = tissue_loglik(ad, fourier, clock, args.dispersion, args.grid_points)
            per[k]["donors"].append(donors)
            per[k]["ll"].append(ll)
            per[k]["names"].append(t)
            theta = CoPhaser.phase_from_loglik(ll, grid, args.kappa_max)["theta"]
            sample_rows.append(pd.DataFrame({"tissue": t, "donor": donors, "clock": k, "theta_tissue": theta,
                                             "chiral_h": chiral, "train_donor": np.isin(donors, list(train_donors))}))
        print(f"[{t}] {ad.n_obs} samples, {len(clock)} clock genes, {int(train.sum())} donors for the half clock",
              flush=True)

    shared = {k: shared_phases(v["donors"], v["ll"], grid, v["names"], args.kappa_max) for k, v in per.items()}
    shared["human"].reset_index().to_csv(args.outdir / "donor_phases.csv", index=False)
    shared["mouse"].reset_index().to_csv(args.outdir / "donor_phases_mouse.csv", index=False)

    s = pd.concat(sample_rows, ignore_index=True)
    s["theta_shared"] = np.nan
    s["n_tissues"] = 0
    for k, tab in shared.items():
        m = s["clock"] == k
        s.loc[m, "theta_shared"] = s.loc[m, "donor"].map(tab["theta"]).to_numpy()
        s.loc[m, "n_tissues"] = s.loc[m, "donor"].map(tab["n_tissues"]).to_numpy()
    s.to_csv(args.outdir / "sample_phases.csv", index=False)

    # evaluation: half clock on held-out donors (honest), full human and mouse on all donors (optimistic)
    rows = []
    for (t, k), g in s.groupby(["tissue", "clock"]):
        ev = g[~g["train_donor"]] if k == "half" else g
        for which in ("tissue", "shared"):
            raw, aligned, n = errors(ev[f"theta_{which}"].to_numpy(float), ev["chiral_h"].to_numpy(float))
            rows.append({"tissue": t, "clock": k, "phase": which, "raw_h": raw, "aligned_h": aligned, "n": n,
                         "donors": "held-out half" if k == "half" else "all"})
    ev = pd.DataFrame(rows)
    ev.to_csv(args.outdir / "evaluation_by_tissue.csv", index=False)

    rows = []
    for k in shared:
        g = s[s["clock"] == k]
        g = g[~g["train_donor"]] if k == "half" else g
        g = g.drop_duplicates("donor")  # one row per donor: the shared phase is the same in every tissue
        for n_t, gg in g.groupby("n_tissues"):
            raw, aligned, n = errors(gg["theta_shared"].to_numpy(float), gg["chiral_h"].to_numpy(float))
            rows.append({"clock": k, "n_tissues": n_t, "raw_h": raw, "aligned_h": aligned, "n_donors": n})
    by_n = pd.DataFrame(rows)
    by_n.to_csv(args.outdir / "evaluation_by_ntissues.csv", index=False)

    piv = ev.pivot_table(index="tissue", columns=["clock", "phase"], values="aligned_h")
    print(piv.round(2).to_string())
    print(by_n.round(2).to_string(index=False))
    summary = {f"{k}_{p}_median_aligned_h": float(ev[(ev.clock == k) & (ev.phase == p)]["aligned_h"].median())
               for k in shared for p in ("tissue", "shared")}
    summary.update(n_donors=int(len(all_donors)), n_train_donors=len(train_donors), split_seed=args.split_seed,
                   dispersion=args.dispersion)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(ev, by_n, args.outdir)


def plot(ev, by_n, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = {"half": "human clock, half (held-out donors)", "human": "human clock, full (all donors)",
              "mouse": "mouse clock (all donors)"}
    colors = {"half": "#1f5f8b", "human": "#7fb3d5", "mouse": "#c0392b"}
    ax = axes[0]
    tissues = sorted(ev["tissue"].unique())
    x = np.arange(len(tissues))
    for i, k in enumerate(["half", "human", "mouse"]):
        for p, marker in (("tissue", "o"), ("shared", "s")):
            d = ev[(ev.clock == k) & (ev.phase == p)].set_index("tissue").reindex(tissues)
            ax.scatter(x + (i - 1) * 0.22, d["aligned_h"], marker=marker, color=colors[k],
                       facecolors="none" if p == "tissue" else colors[k],
                       label=f"{labels[k]}, {'per tissue' if p == 'tissue' else 'shared'}")
    ax.set_xticks(x, [t.replace("_", " ") for t in tissues], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("median |error| vs CHIRAL (h, aligned)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=6.5)
    ax = axes[1]
    for k in ["half", "human", "mouse"]:
        d = by_n[by_n.clock == k]
        ax.plot(d["n_tissues"], d["aligned_h"], "o-", color=colors[k], label=labels[k])
    ax.set_xlabel("number of tissues of the donor")
    ax.set_ylabel("median |error| vs CHIRAL (h, aligned)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle("Shared donor phase (sum of grid log-likelihoods over tissues) vs per-tissue grid phase", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "shared_phase_errors.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
