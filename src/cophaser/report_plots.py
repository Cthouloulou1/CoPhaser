"""
Report figures for rhythmic fits: learning curves, clock-face (polar) gene plots,
and phase-recovery diagnostics.

Clock-face convention (as in the CHIRAL / GTEx figures): time 0 at the top,
running clockwise, radius = amplitude, one point per gene at its acrophase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TWO_PI = 2 * np.pi

# Epochs at which CoPhaser switches parts of the model on (CoPhaser.forward).
MODEL_SWITCH_EPOCHS = {50: "mu_z correction on", 100: "lambda(z) on"}


def _to_hours(phase, period):
    return np.mod(phase, TWO_PI) * period / TWO_PI


def losses_history_frame(history: dict) -> pd.DataFrame:
    """Trainer.losses_history -> DataFrame indexed by epoch (one column per loss term)."""
    longest = max(len(v) for v in history.values())
    padded = {k: list(v) + [np.nan] * (longest - len(v)) for k, v in history.items()}
    df = pd.DataFrame(padded)
    return df.set_index("epoch") if "epoch" in df.columns else df


def plot_learning_curves(history: dict | pd.DataFrame, switch_epochs=MODEL_SWITCH_EPOCHS, ncols=3):
    """One small panel per loss term, with the epochs where model parts switch on."""
    df = history if isinstance(history, pd.DataFrame) else losses_history_frame(history)
    # constant series (terms switched off in this configuration) carry no information
    cols = [c for c in df.columns if df[c].notna().any() and np.nanstd(df[c].to_numpy(float)) > 0]
    dropped = [c for c in df.columns if c not in cols]
    order = ["total_loss", "elbo_loss"]
    cols = [c for c in order if c in cols] + [c for c in cols if c not in order]
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), squeeze=False, sharex=True)
    for ax, col in zip(axes.flat, cols):
        ax.plot(df.index, df[col], lw=1.4, color="#2a6f97")
        ax.set_title(col, fontsize=10)
        for ep, label in (switch_epochs or {}).items():
            if df.index.min() <= ep <= df.index.max():
                ax.axvline(ep, color="0.6", ls=":", lw=1)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    notes = []
    if switch_epochs:
        notes.append("dotted lines — " + "  ·  ".join(f"epoch {e}: {l}" for e, l in switch_epochs.items()))
    if dropped:
        notes.append("constant, not shown: " + ", ".join(dropped))
    if notes:
        fig.text(0.5, 0.0, "\n".join(notes), ha="center", fontsize=8, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def setup_clock_axis(ax, period=24.0, n_ticks=6, top_label=True):
    """Polar axis with time 0 at the top, clockwise, labelled in hours."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    hours = np.linspace(0, period, n_ticks, endpoint=False)
    ax.set_xticks(hours * TWO_PI / period)
    labels = [f"{h:g}" for h in hours]
    if top_label:
        labels[0] = f"0/{period:g}"
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.35)
    return ax


def _finish_radial_axis(ax, rmax):
    """Few, small radial ticks placed between the hour labels so they do not overlap."""
    ax.set_rlim(0, rmax)
    ticks = np.round(np.linspace(0, rmax, 4)[1:-1], 2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=7, color="0.45")
    ax.set_rlabel_position(135)


def plot_clock_face(
    amp,
    phase,
    genes,
    ax=None,
    period=24.0,
    title=None,
    color="black",
    marker="o",
    label=None,
    annotate=True,
    rmax=None,
):
    """Genes on a clock face: angle = acrophase (radians), radius = amplitude."""
    amp, phase = np.asarray(amp, float), np.asarray(phase, float)
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="polar")
    setup_clock_axis(ax, period)
    ax.scatter(np.mod(phase, TWO_PI), amp, s=22, color=color, marker=marker, label=label, zorder=3)
    if annotate:
        for a, p, g in zip(amp, phase, genes):
            ax.annotate(
                str(g).upper(), (np.mod(p, TWO_PI), a), xytext=(4, 4), textcoords="offset points",
                fontsize=7, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="0.3", lw=0.5),
            )
    _finish_radial_axis(ax, rmax if rmax is not None else 1.15 * np.nanmax(amp))
    if title:
        ax.set_title(title, fontsize=10, pad=14)
    return ax


def plot_clock_comparison(
    genes, amp_a, phase_a, amp_b, phase_b, label_a, label_b, ax=None, period=24.0, title=None
):
    """Two fits of the same genes on one clock face, joined gene by gene."""
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="polar")
    setup_clock_axis(ax, period)
    pa, pb = np.mod(phase_a, TWO_PI), np.mod(phase_b, TWO_PI)
    for a1, p1, a2, p2 in zip(amp_a, pa, amp_b, pb):
        d = np.angle(np.exp(1j * (p2 - p1)))  # shortest way round
        t = np.linspace(0, 1, 20)
        ax.plot(p1 + t * d, a1 + t * (a2 - a1), color="0.6", lw=0.8, zorder=1)
    ax.scatter(pa, amp_a, s=30, facecolors="none", edgecolors="black", label=label_a, zorder=3)
    ax.scatter(pb, amp_b, s=22, color="#c0392b", label=label_b, zorder=4)
    for a, p, g in zip(amp_b, pb, genes):
        ax.annotate(str(g).upper(), (p, a), xytext=(4, 4), textcoords="offset points", fontsize=7,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="0.3", lw=0.5))
    _finish_radial_axis(ax, 1.15 * np.nanmax(np.r_[amp_a, amp_b]))
    ax.legend(loc="upper left", bbox_to_anchor=(0.85, 1.12), fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=10, pad=14)
    return ax


def cosinor_fit(expr: np.ndarray, theta: np.ndarray) -> pd.DataFrame:
    """Per-column first-harmonic regression ``y = m + a cos(theta) + b sin(theta)``.

    expr: (n_samples, n_genes). Returns mesor, amp, phase (acrophase, radians) and R^2.
    """
    expr = np.asarray(expr, float)
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    coef, *_ = np.linalg.lstsq(X, expr, rcond=None)
    resid = expr - X @ coef
    ss_tot = ((expr - expr.mean(0)) ** 2).sum(0)
    r2 = 1 - (resid**2).sum(0) / np.where(ss_tot > 0, ss_tot, np.nan)
    m, a, b = coef
    return pd.DataFrame(
        {"mesor": m, "amp": np.sqrt(a**2 + b**2), "phase": np.mod(np.arctan2(b, a), TWO_PI), "r2": r2}
    )


def plot_phase_recovery(theta_true, theta_pred, period=24.0, offset=None):
    """Inferred vs true time (scatter on the torus) and signed error histogram."""
    t_true, t_pred = _to_hours(theta_true, period), _to_hours(theta_pred, period)
    err = np.angle(np.exp(1j * (theta_pred - theta_true))) * period / TWO_PI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.scatter(t_true, t_pred, s=4, alpha=0.35, color="#2a6f97", lw=0)
    for k in (-1, 0, 1):
        ax1.plot([0, period], [k * period, (k + 1) * period], color="#c0392b", lw=1, ls="--")
    ax1.set(xlim=(0, period), ylim=(0, period), xlabel="true time (h)", ylabel="inferred time (h)",
            title="Inferred vs true phase")
    ax1.set_xticks(np.arange(0, period + 1e-9, period / 6))
    ax1.set_yticks(np.arange(0, period + 1e-9, period / 6))
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)
    bins = np.linspace(-period / 2, period / 2, 49)
    ax2.hist(err, bins=bins, color="#2a6f97", alpha=0.85)
    ax2.axvline(0, color="black", lw=0.8)
    med = np.median(np.abs(err))
    ax2.set(xlabel="inferred − true (h)", ylabel="cells",
            title=f"Signed error  ·  median |error| {med:.2f} h")
    if offset is not None:
        ax2.axvline(offset, color="#c0392b", ls="--", lw=1, label=f"circular mean {offset:+.2f} h")
        ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_phase_distribution(theta_pred, theta_true=None, period=24.0, n_bins=24):
    """Rose histogram of inferred (and true) phases on the clock face."""
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(projection="polar")
    setup_clock_axis(ax, period)
    bins = np.linspace(0, TWO_PI, n_bins + 1)
    width = TWO_PI / n_bins
    h_pred, _ = np.histogram(np.mod(theta_pred, TWO_PI), bins=bins)
    ax.bar(bins[:-1] + width / 2, h_pred, width=width, color="#2a6f97", alpha=0.6, label="inferred")
    if theta_true is not None:
        h_true, _ = np.histogram(np.mod(theta_true, TWO_PI), bins=bins)
        ax.step(np.r_[bins[:-1], TWO_PI], np.r_[h_true, h_true[0]], where="post", color="black",
                lw=1, label="true")
    ax.set_title("Phase distribution", fontsize=10, pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0.85, 1.12), fontsize=8, frameon=False)
    return fig


EFFECT_LABELS = {
    "mesor": "mesor shift (log)",
    "log_amp": "log-amplitude change",
    "phase": "peak delay (h)",
}


def plot_effect_recovery(truth: pd.DataFrame, estimate: pd.DataFrame, period=24.0, title=None):
    """Known vs estimated effects, one panel per (covariate, kind).

    Both tables are long: gene, covariate, kind, value (phase in radians).
    """
    m = truth.merge(estimate, on=["gene", "covariate", "kind"], suffixes=("_true", "_est"))
    m = m.copy()
    is_phase = m["kind"] == "phase"
    for col in ("value_true", "value_est"):
        m.loc[is_phase, col] = np.angle(np.exp(1j * m.loc[is_phase, col])) * period / TWO_PI
    covs, kinds = list(pd.unique(m["covariate"])), [k for k in EFFECT_LABELS if k in set(m["kind"])]
    fig, axes = plt.subplots(len(covs), len(kinds), figsize=(3.6 * len(kinds), 3.3 * len(covs)), squeeze=False)
    for i, c in enumerate(covs):
        for j, k in enumerate(kinds):
            ax = axes[i, j]
            d = m[(m["covariate"] == c) & (m["kind"] == k)]
            lim = 1.1 * np.nanmax(np.abs(np.r_[d["value_true"], d["value_est"], 1e-3]))
            ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, ls="--")
            ax.axhline(0, color="0.85", lw=0.6)
            ax.axvline(0, color="0.85", lw=0.6)
            ax.scatter(d["value_true"], d["value_est"], s=16, color="#2a6f97", zorder=3)
            for _, r in d[np.abs(d["value_true"]) > 0].iterrows():
                ax.annotate(str(r["gene"]).upper(), (r["value_true"], r["value_est"]), fontsize=6,
                            xytext=(2, 2), textcoords="offset points", color="0.3")
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
            ax.set_aspect("equal")
            ax.set_title(f"{c} · {EFFECT_LABELS[k]}", fontsize=9)
            if i == len(covs) - 1:
                ax.set_xlabel("injected")
            if j == 0:
                ax.set_ylabel("estimated")
            ax.grid(alpha=0.25)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
