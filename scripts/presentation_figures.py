"""
Figures for the progress presentation: phase inference on GTEx, the amplitude bias of the
CoPhaser covariate model, validation of the step 10 gene-rhythm GLM and its first GTEx
results. Reads existing result tables only (nothing is fitted here, except the reading of
one tissue's counts for the gene profiles).

Sources: results/gtex/exp14_shared_phase (phase errors), exp13*/exp17b (bootstrap effects),
results/step9/sim{,_nolambda} (lambda bias), results/step10/sim (GLM validation),
results/step10/gtex/shared_ref12{,_perm0} (GLM on GTEx), data/gtex_v8/tissues (counts).

Usage, from the repository root:
    python scripts/presentation_figures.py [--out results/presentation] [--only fig01 fig07 ...]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
H = 24 / (2 * np.pi)

# light chart surface, ink and categorical slots (validated default palette)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "font.size": 10, "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "savefig.dpi": 200, "figure.titlesize": 12, "figure.titleweight": "bold",
})

TISSUES = ["adipose_subcutaneous", "artery_tibial", "lung", "muscle_skeletal", "nerve_tibial",
           "skin_not_sun_exposed_suprapubic", "skin_sun_exposed_lower_leg", "thyroid"]
SHORT = {"adipose_subcutaneous": "Adipose", "artery_tibial": "Artery (tibial)", "lung": "Lung",
         "muscle_skeletal": "Muscle", "nerve_tibial": "Nerve (tibial)",
         "skin_not_sun_exposed_suprapubic": "Skin (not exposed)", "skin_sun_exposed_lower_leg": "Skin (sun-exposed)",
         "thyroid": "Thyroid"}
GLM_COVS = [("age", "Age (per 10 years)"), ("sex", "Sex (women vs men)"), ("hardy[0]", "Ventilator (Hardy 0) vs fast death")]
BOOT_COVS = {"age": "age (per 10 y)", "sex": "sex (F vs M)", "hardy[0]": "hardy[0]"}


def clock_genes():
    from cophaser.gene_sets import HUMAN_CIRCADIAN_GENE_SET
    return list(HUMAN_CIRCADIAN_GENE_SET)


def wrap_h(x):
    return (np.asarray(x, dtype=float) + 12) % 24 - 12


def circ_mean_h(h):
    return float(np.angle(np.exp(1j * np.asarray(h, dtype=float) / H).mean()) * H) % 24


def load_glm(tissue, cols=None, perm=False):
    sub = "shared_ref12_perm0" if perm else "shared_ref12"
    return pd.read_csv(R / "step10" / "gtex" / sub / f"{tissue}.csv.gz", usecols=cols)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png", flush=True)


def dot(ax, x, y, color, s=46, filled=True, **kw):
    ax.scatter(x, y, s=s, color=color if filled else SURF, edgecolor=color if not filled else SURF,
               linewidth=1.5, zorder=3, **kw)


def fig01_phase_error(out):
    """Median phase error vs CHIRAL: one phase per tissue vs a phase shared across a donor's tissues."""
    ev = pd.read_csv(R / "gtex" / "exp14_shared_phase" / "evaluation_by_tissue.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    y = np.arange(len(TISSUES))[::-1]
    for ax, clock, title in zip(axes, ["half", "mouse"],
                                ["Human clock (from CHIRAL, held-out donors)", "Mouse clock (all donors)"]):
        d = ev[ev.clock == clock].pivot(index="tissue", columns="phase", values="aligned_h").loc[TISSUES]
        ax.hlines(y, d["shared"], d["tissue"], color=AXIS, lw=2, zorder=1)
        dot(ax, d["tissue"], y, C2, label="One phase per tissue")
        dot(ax, d["shared"], y, C1, label="One phase per donor, shared across tissues")
        ax.axvline(6, color=MUTED, lw=1, zorder=1)
        ax.text(5.9, y.max() + 0.45, "random phases", color=MUTED, ha="right", fontsize=8)
        ax.set_xlim(0, 6.4)
        ax.set_ylim(-0.6, len(TISSUES) - 0.2)
        ax.set_yticks(y, [SHORT[t] for t in TISSUES])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Median phase error vs CHIRAL (h, after alignment)")
        ax.set_title(title)
        ax.text(4.6, y.min(), f"median {d['tissue'].median():.1f} h → {d['shared'].median():.1f} h",
                ha="center", va="center", color=INK, fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    save(fig, out, "fig01_phase_error_shared")


def fig02_cry1_sex(out):
    """CRY1 peak shift in women: depends on the reference clock."""
    src = [("Mouse clock", "exp13b_bootstrap_mouse", C2),
           ("Clock learned per tissue", "exp13_bootstrap", C3),
           ("Human clock (from CHIRAL)", "exp13h_bootstrap_human", C1)]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    y0 = np.arange(len(TISSUES))[::-1]
    for i, (lab, d, col) in enumerate(src):
        b = pd.read_csv(R / "gtex" / d / "bootstrap_effects.csv")
        b = b[(b.gene == "CRY1") & (b.kind == "phase") & b.covariate.str.startswith("sex")].set_index("tissue_file")
        off = (1 - i) * 0.24
        for j, t in enumerate(TISSUES):
            if t not in b.index:
                continue
            r = b.loc[t]
            ax.plot([r.ci_low, r.ci_high], [y0[j] + off] * 2, color=col, lw=1.5, zorder=2)
            dot(ax, r.estimate, y0[j] + off, col, s=40, filled=bool(r.q_bh <= 0.05))
        ax.scatter([], [], s=40, color=col, label=lab)
    ax.scatter([], [], s=40, color=SURF, edgecolor=MUTED, linewidth=1.5, label="Not significant (BH q > 0.05)")
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    ax.set_yticks(y0, [SHORT[t] for t in TISSUES])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("CRY1 peak shift, women vs men (h; 95% bootstrap interval, 40 replicates)")
    ax.set_title("CRY1 later in women only with the mouse clock")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, out, "fig02_cry1_sex_by_clock")


def fig03_lambda_bias(out):
    """Simulation (step 9): log-amplitude effects of genes without a true effect, with and without lambda(z)."""
    def load(d):
        e = pd.read_csv(R / "step9" / d / "effects_long.csv")
        e = e[(e.jitter_h == 0) & e.seed.isin([0, 1]) & e["mode"].isin(["oracle", "shared"])
              & (e.kind == "log_amp") & (e.amp >= 0.15) & (e.value_true == 0)]
        return e
    covs = [("age", "Age\n(per SD)"), ("sex[female]", "Sex\n(women vs men)"), ("hardy[1]", "Condition\n(yes vs no)")]
    runs = [("sim", "With λ(z) (before)", C2), ("sim_nolambda", "Without λ(z) (--no-lambda)", C1)]
    data = {r: load(r) for r, _, _ in runs}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mode, title in zip(axes, ["oracle", "shared"], ["True phase", "Shared phase (≈1 h error)"]):
        for k, (r, lab, col) in enumerate(runs):
            e = data[r]
            for i, (c, _) in enumerate(covs):
                v = e[(e["mode"] == mode) & (e.covariate == c)].value_est.to_numpy()
                x = i + (k - 0.5) * 0.36
                ax.scatter(x + rng.uniform(-0.1, 0.1, len(v)), v, s=10, color=col, alpha=0.45, linewidth=0, zorder=2,
                           label=lab if i == 0 else None)
                med = np.median(v)
                ax.plot([x - 0.14, x + 0.14], [med, med], color=INK, lw=2.5, zorder=3)
                ax.text(x, 1.42, f"{med:+.2f}", ha="center", color=INK2, fontsize=8)
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ax.set_xticks(range(len(covs)), [c[1] for c in covs])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(-1.05, 1.55)
        ax.set_title(title)
    axes[0].set_ylabel("Estimated log-amplitude effect\n(genes with true effect 0)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), markerscale=2)
    fig.text(0.5, -0.08, "Black bar and number: median. Simulation, 900 donors, 5 tissues, frozen clock, seeds 0-1.",
             ha="center", color=MUTED, fontsize=8)
    save(fig, out, "fig03_lambda_amplitude_bias")


SIM_MODES = [("oracle", "True phase", INK2), ("shared", "Shared phase (1.0 h error)", C1),
             ("shared_noisycond", "Shared, extra noise in condition group", C3),
             ("tissue", "Phase per tissue (2.3 h error)", C2)]


def fig04_sim_recovery(out):
    """Step 10 simulation: estimated vs true amplitude ratio and peak shift, by phase mode."""
    est = pd.read_csv(R / "step10" / "sim" / "estimates_long.csv.gz")
    tru = pd.read_csv(R / "step10" / "sim" / "truth.csv.gz")
    m = est.merge(tru, on=["tissue", "gene", "covariate"])
    m = m[(m.seed == 0) & (~m.is_clock) & (m.amp_true >= 0.15)]
    modes = [SIM_MODES[0], SIM_MODES[1], SIM_MODES[3]]
    rows = [("log_amp_true", "log_amp_est", 1.5, "log-amplitude ratio"), ("phase_true_h", "phase_est_h", 6.0, "peak shift (h)")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    for j, (mode, lab, col) in enumerate(modes):
        s = m[m["mode"] == mode]
        for i, (xt, ye, lim, name) in enumerate(rows):
            ax = axes[i, j]
            ok = s[[xt, ye]].notna().all(axis=1)
            x, y = s.loc[ok, xt].to_numpy(), s.loc[ok, ye].to_numpy()
            ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=1, zorder=1)
            ax.scatter(x, np.clip(y, -lim, lim), s=7, color=col, alpha=0.3, linewidth=0, zorder=2)
            nz = (x != 0) & (np.abs(y) < 4 * lim)
            slope = np.polyfit(x[nz], y[nz], 1)[0]
            r = np.corrcoef(x[nz], y[nz])[0, 1]
            ax.text(0.04, 0.96, f"slope {slope:.2f}\nr = {r:.2f}", transform=ax.transAxes, va="top", color=INK, fontsize=9)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel(f"True {name}")
            if j == 0:
                ax.set_ylabel(f"Estimated {name}")
            if i == 0:
                ax.set_title(lab)
    fig.text(0.5, -0.01, "Rhythmic non-clock genes (true amplitude ≥ 0.15), age, sex and condition pooled; "
             "slope and r on genes with a non-zero true effect. Seed 0, 5 tissues.", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig04_step10_sim_recovery")


def fig05_sim_power(out):
    """Step 10 simulation: detection power and false discovery proportion by phase mode."""
    mt = pd.read_csv(R / "step10" / "sim" / "metrics.csv")
    covs = [("age", "Age"), ("sex", "Sex"), ("condition", "Condition")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.6, 1.6]})
    w = 0.2
    ax = axes[0]
    for k, (mode, lab, col) in enumerate(SIM_MODES):
        v = mt[mt["mode"] == mode].rhythm_power.iloc[0]
        ax.bar(k, v, width=0.7, color=col, label=lab)
        ax.text(k, v + 0.02, f"{v:.2f}", ha="center", color=INK2, fontsize=8)
    ax.set_xticks([])
    ax.set_ylim(0, 1.08)
    ax.set_title("Rhythm detected")
    ax.set_ylabel("Fraction of rhythmic genes (q < 0.05)")
    for ax, col_name, title, ylab in [(axes[1], "diff_power", "Rhythm change detected", "Power (q < 0.05)"),
                                      (axes[2], "diff_fdp", "False discoveries", "False discovery proportion")]:
        for k, (mode, lab, col) in enumerate(SIM_MODES):
            v = [mt[(mt["mode"] == mode) & (mt.covariate == c)][col_name].iloc[0] for c, _ in covs]
            x = np.arange(len(covs)) + (k - 1.5) * w
            ax.bar(x, v, width=w - 0.02, color=col)
        ax.set_xticks(range(len(covs)), [c[1] for c in covs])
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        ax.set_ylabel(ylab)
    axes[1].set_ylim(0, 0.5)
    axes[2].axhline(0.05, color=INK, lw=1)
    axes[2].set_xlim(-0.5, 2.5)
    axes[2].text(2.53, 0.05, "target\n5%", color=INK, fontsize=8, ha="left", va="center", clip_on=False)
    axes[2].set_ylim(0, 0.22)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, out, "fig05_step10_sim_power")


def fig06_pvalues(out):
    """GTEx: rhythm p-values with the shared phases and with permuted phases (negative control)."""
    tis = ["thyroid", "lung", "artery_tibial"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, t in zip(axes, tis):
        r = load_glm(t, ["p_rhythm"]).p_rhythm.dropna()
        p = load_glm(t, ["p_rhythm"], perm=True).p_rhythm.dropna()
        ax.hist(r, bins, density=True, histtype="step", color=C1, lw=2, label="Shared phases")
        ax.hist(p, bins, density=True, histtype="step", color=C2, lw=2, label="Phases permuted (negative control)")
        ax.axhline(1, color=INK, lw=1)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_title(SHORT[t])
        ax.set_xlabel("p-value (rhythm, LRT 2 df)")
        ax.text(0.97, 0.95, f"p < 0.05: {(r < 0.05).mean():.0%} vs {(p < 0.05).mean():.0%}\n(expected 5%)",
                transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=9)
    axes[0].set_ylabel("Density (uniform = 1)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, out, "fig06_pvalues_real_vs_permuted")


def clock_table():
    clock = clock_genes()
    d = pd.concat([load_glm(t).query("gene in @clock").assign(tissue=t) for t in TISSUES])
    order = d.groupby("gene").phase_avg_h.apply(circ_mean_h).sort_values()
    return d, order


def fig07_clock_peaks(out):
    """GTEx: peak time of each clock gene in the 8 tissues."""
    d, order = clock_table()
    genes = list(order.index)
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for i, g in enumerate(genes):
        y = len(genes) - 1 - i
        s = d[d.gene == g]
        x = order[g] + wrap_h(s.phase_avg_h - order[g])
        sig = (s.q_rhythm < 0.05).to_numpy()
        size = 12 + 220 * s.amp_avg.clip(upper=0.8).to_numpy()
        ax.scatter(x[sig], np.full(sig.sum(), y), s=size[sig], color=MUTED, alpha=0.6, edgecolor=SURF, linewidth=1, zorder=2)
        ax.scatter(x[~sig], np.full((~sig).sum(), y), s=size[~sig], facecolor="none", edgecolor=MUTED, linewidth=1, zorder=2)
        ax.plot([order[g]] * 2, [y - 0.34, y + 0.34], color=C1, lw=2.5, zorder=3)
    ax.set_yticks(range(len(genes)), genes[::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlim(-1.5, 27)
    ax.set_xlabel("Peak time (h, CHIRAL reference frame, not wall-clock time)")
    ax.set_title("Clock genes peak in the same order in all 8 tissues")
    for a, lab in [(0.1, "amp 0.1"), (0.4, "amp 0.4")]:
        ax.scatter([], [], s=12 + 220 * a, color=MUTED, alpha=0.6, label=lab)
    ax.scatter([], [], s=50, facecolor="none", edgecolor=MUTED, label="not rhythmic (q ≥ 0.05)")
    ax.plot([], [], color=C1, lw=2.5, label="circular mean over tissues")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, out, "fig07_clock_peak_times")


def rhythmic(d, amp_min=0.2):
    return d[(d.q_rhythm < 0.05) & (d.amp_avg >= amp_min) & (~d.is_clock)]


def fig08_counts(out):
    """GTEx: number of rhythmic non-clock genes per tissue."""
    cols = ["gene", "q_rhythm", "amp_avg", "phase_avg_h", "is_clock"]
    rows = []
    for t in TISSUES:
        d = load_glm(t, cols)
        rows.append(dict(tissue=t, n=len(d), q05=int((d.q_rhythm < 0.05).sum()), strong=len(rhythmic(d))))
    c = pd.DataFrame(rows).set_index("tissue").loc[TISSUES[::-1]]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(c))
    ax.barh(y, c.strong, height=0.62, color=C1)
    for yi, (_, r) in zip(y, c.iterrows()):
        ax.text(r.strong + 12, yi, f"{r.strong}   (q < 0.05 alone: {r.q05:,} of {r.n:,})", va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, [SHORT[t] for t in c.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, c.strong.max() * 2.1)
    ax.set_xlabel("Rhythmic genes (q < 0.05 and peak/trough ≥ 1.5), clock genes excluded")
    ax.set_title("Rhythmic genes beyond the clock, per tissue")
    save(fig, out, "fig08_rhythmic_counts")


def fig09_peak_distribution(out):
    """GTEx: peak times of the rhythmic non-clock genes, with the DBP and ARNTL peaks as landmarks."""
    cols = ["gene", "q_rhythm", "amp_avg", "phase_avg_h", "is_clock"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True)
    for ax, t in zip(axes.ravel(), TISSUES):
        d = load_glm(t, cols)
        s = rhythmic(d)
        ax.hist(s.phase_avg_h % 24, bins=np.arange(0, 25, 1), color=C1, edgecolor=SURF, linewidth=1)
        top = ax.get_ylim()[1]
        for g in ["DBP", "ARNTL"]:
            ph = d.loc[d.gene == g, "phase_avg_h"].iloc[0] % 24
            ax.axvline(ph, color=INK, lw=1)
            ax.text(ph + 0.3, top * 0.97, g, color=INK, fontsize=7, va="top",
                    bbox=dict(facecolor=SURF, edgecolor="none", pad=0.5))
        ax.set_title(f"{SHORT[t]} (n = {len(s)})", fontsize=10)
        ax.set_xticks(range(0, 25, 6))
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Peak time (h, CHIRAL frame)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Genes")
    fig.suptitle("Peak times of rhythmic non-clock genes (q < 0.05, peak/trough ≥ 1.5)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "fig09_peak_time_distribution")


def glm_clock_effects():
    clock = clock_genes()
    rows = []
    for t in TISSUES:
        g = load_glm(t).query("gene in @clock")
        for k, _ in GLM_COVS:
            for _, r in g.iterrows():
                rows.append(dict(tissue=t, k=k, gene=r.gene, amp=r.amp, lar=r[f"log_amp_ratio_{k}"],
                                 lar_se=r[f"log_amp_ratio_se_{k}"], shift=r[f"phase_shift_h_{k}"],
                                 q=r[f"q_diffrhythm_{k}"]))
    return pd.DataFrame(rows)


def fig10_cophaser_vs_glm(out):
    """Clock genes: CoPhaser bootstrap estimates (exp. 17b) vs the GLM, same phases and reference."""
    glm = glm_clock_effects()
    b = pd.read_csv(R / "gtex" / "exp17b_ref_hardy12" / "bootstrap_effects.csv")
    b = b[b.kind == "log_amp"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))
    lim = 1.6
    for ax, (k, lab) in zip(axes, GLM_COVS):
        bb = b[b.covariate == BOOT_COVS[k]][["tissue_file", "gene", "estimate"]]
        m = glm[(glm.k == k) & (glm.amp >= 0.15)].merge(bb, left_on=["tissue", "gene"], right_on=["tissue_file", "gene"])
        ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=1, zorder=1)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        ax.axvline(0, color=GRID, lw=1, zorder=1)
        ax.scatter(m.lar.clip(-lim, lim), m.estimate.clip(-lim, lim), s=22, color=C1, alpha=0.8, edgecolor=SURF,
                   linewidth=1, zorder=2)
        mx, my = m.lar.median(), m.estimate.median()
        ax.scatter([mx], [my], s=160, marker="+", color=C2, linewidth=2.5, zorder=4)
        ax.text(0.04, 0.96, f"r = {np.corrcoef(m.lar, m.estimate)[0, 1]:.2f}\nmedian: GLM {mx:+.2f}, CoPhaser {my:+.2f}",
                transform=ax.transAxes, va="top", color=INK, fontsize=9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(lab)
        ax.set_xlabel("GLM (step 10): log-amplitude ratio")
    axes[0].set_ylabel("CoPhaser (exp. 17b): log-amplitude effect")
    fig.text(0.5, -0.03, "Clock genes × 8 tissues (reference amplitude ≥ 0.15). Same shared phases, Hardy 1-2 as reference. "
             "Orange cross: medians.", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig10_cophaser_vs_glm")


def fig11_clock_effects(out):
    """GLM: amplitude change of each clock gene by age, sex and death type, across tissues."""
    glm = glm_clock_effects()
    _, order = clock_table()
    tested = set(glm.loc[glm.amp >= 0.15, "gene"])
    genes = [g for g in order.index if g in tested]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharey=True)
    lim = 1.5
    for ax, (k, lab) in zip(axes, GLM_COVS):
        s = glm[(glm.k == k) & (glm.amp >= 0.15)]
        for i, g in enumerate(genes):
            y = len(genes) - 1 - i
            sg = s[s.gene == g]
            if sg.empty:
                continue
            sig = (sg.q < 0.05).to_numpy()
            x = sg.lar.clip(-lim, lim).to_numpy()
            ax.scatter(x[sig], np.full(sig.sum(), y), s=30, color=C1, edgecolor=SURF, linewidth=1, zorder=3)
            ax.scatter(x[~sig], np.full((~sig).sum(), y), s=30, facecolor="none", edgecolor=MUTED, linewidth=1, zorder=2)
            med = np.median(sg.lar)
            ax.plot([med] * 2, [y - 0.36, y + 0.36], color=INK, lw=2, zorder=4)
        ax.axvline(0, color=AXIS, lw=1, zorder=1)
        ax.set_xlim(-lim, lim)
        ax.grid(axis="y", visible=False)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("log-amplitude ratio (0 = no change)")
    axes[0].set_yticks(range(len(genes)), genes[::-1])
    axes[0].scatter([], [], s=30, color=C1, label="rhythm differs (q < 0.05)")
    axes[0].scatter([], [], s=30, facecolor="none", edgecolor=MUTED, label="not significant")
    axes[0].plot([], [], color=INK, lw=2, label="median over tissues")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.text(0.5, -0.02, "One dot per tissue (reference amplitude ≥ 0.15). GLM per gene, shared phases, reference: "
             "men, age 50, fast deaths (Hardy 1-2).", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig11_clock_effects_glm")


def fig12_profiles(out, tissue="thyroid"):
    """Data and GLM curves for three genes of one tissue: HLF (all donors), PER1 and ARNTL by death type."""
    import anndata

    ad = anndata.read_h5ad(REPO / "data" / "gtex_v8" / "tissues" / f"{tissue}.h5ad")
    ph = pd.read_csv(R / "gtex" / "exp14_shared_phase" / "donor_phases.csv").set_index("donor")
    donor = ad.obs["donor"].astype(str)
    hardy = pd.to_numeric(ad.obs["hardy"].astype(str), errors="coerce")
    keep = (donor.isin(ph.index) & hardy.notna()).to_numpy()
    ad = ad[keep]
    theta = ph.loc[ad.obs["donor"].astype(str), "theta"].to_numpy(float)
    t_h = (theta * H) % 24
    hardy = hardy[keep].to_numpy()
    groups = {"Fast death (Hardy 1-2)": np.isin(hardy, [1, 2]), "Ventilator (Hardy 0)": hardy == 0}
    gcol = {"Fast death (Hardy 1-2)": C1, "Ventilator (Hardy 0)": C2}
    lib = ad.obs["library_size"].to_numpy(float)
    res = load_glm(tissue).set_index("gene")
    tt = np.linspace(0, 24, 200)
    bins = np.arange(0, 25, 2)
    mid = bins[:-1] + 1
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, gene, by_group in zip(axes, ["HLF", "PER1", "ARNTL"], [False, True, True]):
        x = ad[:, gene].layers["counts"]
        x = np.asarray(x.toarray() if hasattr(x, "toarray") else x).ravel()
        y = np.log2((x + 0.5) / lib * 1e6)
        r = res.loc[gene]
        sel = {"All donors": np.ones_like(y, dtype=bool)} if not by_group else groups
        for k, (lab, mask) in enumerate(sel.items()):
            col = INK2 if not by_group else gcol[lab]
            yc = y[mask] - y[mask].mean()
            idx = np.digitize(t_h[mask], bins) - 1
            mu = np.array([yc[idx == i].mean() for i in range(len(mid))])
            se = np.array([yc[idx == i].std() / np.sqrt(max((idx == i).sum(), 1)) for i in range(len(mid))])
            off = (k - 0.5) * 0.35 if by_group else 0
            ax.errorbar(mid + off, mu, yerr=se, fmt="o", ms=5, color=col, mec=SURF, mew=1, elinewidth=1.2, zorder=3)
            if not by_group:
                curve = r.amp_avg * np.cos(tt / H - r.phase_avg_h / H)
                note = f"peak/trough ×{np.exp(2 * r.amp_avg):.2f}, peak at {r.phase_avg_h % 24:.1f} h, q = {r.q_rhythm:.1g}"
            else:
                b, c = r.b, r.c
                if lab.startswith("Ventilator"):
                    b, c = b + r["db_hardy[0]"], c + r["dc_hardy[0]"]
                curve = b * np.cos(tt / H) + c * np.sin(tt / H)
            ax.plot(tt, curve / np.log(2), color=col, lw=2, zorder=2, label=lab)
        if by_group:
            lar, sh, q = r["log_amp_ratio_hardy[0]"], r["phase_shift_h_hardy[0]"], r["q_diffrhythm_hardy[0]"]
            note = f"Hardy 0: amplitude ×{np.exp(lar):.2f}, shift {sh:+.1f} h, q = {q:.1g}"
            ax.legend(loc="lower left", fontsize=8)
        ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", color=INK, fontsize=8)
        ax.set_title(f"{gene} ({SHORT[tissue].lower()})")
        ax.set_xticks(range(0, 25, 6))
        ax.set_xlim(0, 24)
        ax.set_xlabel("Shared donor phase (h, CHIRAL frame)")
    axes[0].set_ylabel("log2 CPM, centred")
    fig.text(0.5, -0.03, "Dots: mean ± SE per 2-h bin; lines: GLM fit (reference covariates), centred.",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig12_gene_profiles")


FIGURES = {f.__name__.split("_")[0]: f for f in [
    fig01_phase_error, fig02_cry1_sex, fig03_lambda_bias, fig04_sim_recovery, fig05_sim_power, fig06_pvalues,
    fig07_clock_peaks, fig08_counts, fig09_peak_distribution, fig10_cophaser_vs_glm, fig11_clock_effects,
    fig12_profiles]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=R / "presentation")
    p.add_argument("--only", nargs="*", default=None, help=f"subset of {sorted(FIGURES)}")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only or sorted(FIGURES):
        FIGURES[name](args.out)


if __name__ == "__main__":
    main()
