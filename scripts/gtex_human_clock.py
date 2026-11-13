"""
Human reference clocks derived from the CHIRAL phases (Talamanca et al. 2023), one per GTEx
tissue: a cosinor of each clock gene's log count fraction on the CHIRAL phase, in the units
of CoPhaser's rhythmic decoder (natural log of counts / library size) and in the CHIRAL time
frame (theta = 2 pi CT / 24).

Two tables per tissue, in the reference-clock format read by
cophaser.reference_clock.load_reference_clock (gene, a_0, amp, phase, phase_hours, r2,
n_samples):
  reference/human_chiral/<tissue>.csv        fitted on all donors with a CHIRAL phase
                                             (for estimating covariate effects)
  reference/human_chiral/half/<tissue>.csv   fitted on a random half of the donors
                                             (--split-seed), for evaluation on the other half

Evaluation (no training): grid phase (CoPhaser.grid_phase, frozen clock, dispersion
--dispersion) of the held-out donors vs CHIRAL, for the half human clock (raw error: same
frame; and after the best rotation/reflection), the mouse example clock and the clock
learned without supervision in exp. 10 (aligned errors). Caveat: CHIRAL estimated its
phases from these same samples, so the agreement is optimistic for every clock.

Outputs in --outdir: evaluation.csv, amplitudes.csv (amp per gene x tissue x clock),
clock_faces.png, amplitudes.png.
"""
import argparse
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
    p.add_argument("--refdir", type=Path, default=REPO_ROOT / "reference" / "human_chiral")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "gtex" / "human_clock")
    p.add_argument("--mouse", type=Path, default=REPO_ROOT / "reference" / "clock_reference_learned.csv")
    p.add_argument("--exp10", type=Path, default=REPO_ROOT / "results" / "gtex" / "exp10_hardy")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--dispersion", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args()


def fit_cosinor_clock(counts, library_size, theta, genes):
    """First-harmonic cosinor of log((counts + 0.5) / library_size) on theta (radians), per
    gene. Returns a reference-clock table: a_0, amp, phase (radians, peak of
    amp cos(theta - phase)), phase_hours, r2, n_samples; indexed by gene (column 'gene')."""
    counts = np.asarray(counts, float)
    y = np.log((counts + 0.5) / np.asarray(library_size, float)[:, None])
    D = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ coef
    r2 = 1 - resid.var(0) / y.var(0)
    phase = np.mod(np.arctan2(coef[2], coef[1]), 2 * np.pi)
    return pd.DataFrame({"gene": list(genes), "a_0": coef[0], "amp": np.hypot(coef[1], coef[2]), "phase": phase,
                         "phase_hours": phase * H, "r2": r2, "n_samples": len(theta)})


def grid_errors(ad, clock, fourier, eval_mask, dispersion):
    """Grid phase with a frozen clock, compared with CHIRAL on eval_mask samples."""
    from gtex_train import circular_compare

    torch.manual_seed(0)
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=clock, n_latent=2, n_harm=1)
    DecoderPrior.load_fourier_coefficients_prior(fourier, model, freeze_defined_weights=True)
    model.load_anndata(ad[:, clock].copy(), layer_to_use="counts", library_size_field="library_size")
    theta = model.grid_phase(dispersion=dispersion)["theta"]
    chiral = np.where(eval_mask, ad.obs["chiral_ct"].to_numpy(float), np.nan)
    m = np.isfinite(chiral)
    raw = float(np.median(np.abs(np.angle(np.exp(1j * (theta[m] - chiral[m] / H))))) * H)
    return raw, circular_compare(theta, chiral)["median_abs_error_h"]


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    from gtex_amplitude_sweep import learned_clocks
    from gtex_train import clock_from_model

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.refdir / "half").mkdir(parents=True, exist_ok=True)
    tissues = args.tissues or sorted(p.stem for p in args.tissue_dir.glob("*.h5ad"))
    learned = learned_clocks(args.exp10)
    mouse = load_reference_clock(args.mouse, rename_map={"Bmal1": "ARNTL"}, n_harm=1)
    mouse.index = mouse.index.str.upper()  # DecoderPrior matches case-insensitively; tables do not
    rows, amps = [], []
    for tissue in tissues:
        ad = anndata.read_h5ad(args.tissue_dir / f"{tissue}.h5ad")
        clock = [g for g in HUMAN_CIRCADIAN_GENE_SET if g in ad.var_names]
        has = ad.obs["chiral_ct"].notna().to_numpy()
        donors = ad.obs["donor"].astype(str).to_numpy()
        rng = np.random.default_rng(args.split_seed)
        train_donors = {d for d in np.unique(donors[has]) if rng.random() < 0.5}
        train = has & np.isin(donors, list(train_donors))
        held = has & ~train
        X = ad[:, clock].layers["counts"]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        lib = ad.obs["library_size"].to_numpy(float)
        theta = ad.obs["chiral_ct"].to_numpy(float) / H
        full = fit_cosinor_clock(X[has], lib[has], theta[has], clock)
        half = fit_cosinor_clock(X[train], lib[train], theta[train], clock)
        full.to_csv(args.refdir / f"{tissue}.csv", index=False)
        half.to_csv(args.refdir / "half" / f"{tissue}.csv", index=False)

        row = {"tissue": tissue, "n_train": int(train.sum()), "n_heldout": int(held.sum())}
        row["human_half_raw_h"], row["human_half_aligned_h"] = grid_errors(
            ad, clock, load_reference_clock(args.refdir / "half" / f"{tissue}.csv", n_harm=1), held, args.dispersion)
        row["mouse_aligned_h"] = grid_errors(ad, clock, mouse, held, args.dispersion)[1]
        lc = clock_from_model(learned[tissue], clock)
        row["learned_aligned_h"] = grid_errors(ad, [g for g in clock if g in lc.index], lc, held, args.dispersion)[1]
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

        amps.append(full.assign(tissue=tissue, clock="human (CHIRAL)")[["tissue", "clock", "gene", "amp", "phase_hours", "r2"]])
        m = mouse.reindex(clock)
        amps.append(pd.DataFrame({"tissue": tissue, "clock": "mouse", "gene": clock, "amp": np.hypot(m["A_1"], m["B_1"]).to_numpy(),
                                  "phase_hours": np.mod(np.arctan2(m["B_1"], m["A_1"]), 2 * np.pi).to_numpy() * H}))
        lr = lc.reindex(clock)
        amps.append(pd.DataFrame({"tissue": tissue, "clock": "learned (exp. 10)", "gene": clock,
                                  "amp": np.hypot(lr["A_1"], lr["B_1"]).to_numpy()}))
    ev = pd.DataFrame(rows)
    ev.to_csv(args.outdir / "evaluation.csv", index=False)
    amps = pd.concat(amps, ignore_index=True)
    amps.to_csv(args.outdir / "amplitudes.csv", index=False)
    plot(amps, args.outdir)
    print(ev.round(2).to_string(index=False))


def plot(amps, outdir):
    hum = amps[amps["clock"] == "human (CHIRAL)"]
    genes = sorted(hum["gene"].unique())
    tissues = sorted(hum["tissue"].unique())
    colors = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for i, t in enumerate(tissues):
        d = hum[hum["tissue"] == t]
        ax.scatter(d["phase_hours"] / H, d["amp"], s=18, color=colors(i), label=t.replace("_", " "), alpha=0.85)
    for g in genes:
        d = hum[hum["gene"] == g]
        ang = np.angle(np.mean(np.exp(1j * d["phase_hours"] / H)))
        ax.annotate(g, (ang, d["amp"].median()), fontsize=8, ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(np.arange(0, 24, 3) / H, [f"{h}h" for h in range(0, 24, 3)])
    ax.set_title("Human clock from CHIRAL phases: peak time (CHIRAL frame) and amplitude (log units)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(outdir / "clock_faces.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    piv = amps.groupby(["gene", "clock"])["amp"].median().unstack().reindex(genes)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    piv.plot.bar(ax=ax, width=0.8, color=["#1f5f8b", "#7fb3d5", "#c0392b"][: piv.shape[1]])
    ax.set_ylabel("amplitude (log units), median over tissues")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Clock amplitudes: human (CHIRAL cosinor) vs mouse example vs unsupervised learned (exp. 10)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "amplitudes.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
