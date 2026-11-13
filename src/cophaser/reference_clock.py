"""
Reference clocks given as per-gene amplitude/phase tables.

A reference clock is a table indexed by gene with, for each harmonic h, an
amplitude and a phase (radians) in log-expression space:

    harmonic 1: ``amp``,  ``phase``
    harmonic h: ``amp{h}``, ``phase{h}``   (h >= 2)

and optionally ``a_0`` (log fraction of counts). Any other column (e.g. QC
columns of a fitted clock) is ignored.

These helpers convert it to the Fourier-coefficient table expected by
``DecoderPrior.load_fourier_coefficients_prior`` and ``simulation.simulate_counts_rhythmic``
(columns ``A_0, A_1, B_1, A_2, B_2, ...``), using the convention of the rhythmic
decoder: ``A_h = amp_h * cos(phase_h)``, ``B_h = amp_h * sin(phase_h)``, so that
the log-mean of a gene is ``A_0 + sum_h amp_h * cos(h * theta - phase_h)``.

Nothing here is species- or dataset-specific: the mouse clock is just one input.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

_AMP_PATTERN = re.compile(r"^amp(\d*)$")


def _harmonic_suffix(h: int) -> str:
    return "" if h == 1 else str(h)


def amp_phase_harmonics(df: pd.DataFrame) -> list[int]:
    """Harmonics h for which both ``amp{h}`` and ``phase{h}`` columns are present."""
    harmonics = []
    for col in df.columns:
        m = _AMP_PATTERN.match(str(col))
        if m is None:
            continue
        h = int(m.group(1)) if m.group(1) else 1
        if h >= 1 and f"phase{_harmonic_suffix(h)}" in df.columns:
            harmonics.append(h)
    return sorted(harmonics)


def amp_phase_to_fourier(df: pd.DataFrame) -> pd.DataFrame:
    """Convert an amp/phase table (index = gene) to ``A_0, A_1, B_1, A_2, B_2, ...``.

    ``A_0`` is copied from ``a_0`` when present. Raises if no harmonic is found.
    """
    harmonics = amp_phase_harmonics(df)
    if not harmonics:
        raise ValueError("No amp/phase columns found (expected at least 'amp' and 'phase').")

    out = pd.DataFrame(index=df.index)
    if "a_0" in df.columns:
        out["A_0"] = df["a_0"].astype(float)
    for h in harmonics:
        s = _harmonic_suffix(h)
        amp = df[f"amp{s}"].astype(float)
        phase = df[f"phase{s}"].astype(float)
        out[f"A_{h}"] = amp * np.cos(phase)
        out[f"B_{h}"] = amp * np.sin(phase)
    return out


def fourier_to_amp_phase(fourier_df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of :func:`amp_phase_to_fourier`. Phases are wrapped to ``[0, 2*pi)``.

    Also works on Fourier coefficients read back from a trained decoder.
    """
    harmonics = sorted(
        {
            int(c.split("_")[1])
            for c in fourier_df.columns
            if re.match(r"^A_\d+$", str(c)) and c != "A_0" and f"B_{c.split('_')[1]}" in fourier_df.columns
        }
    )
    out = pd.DataFrame(index=fourier_df.index)
    if "A_0" in fourier_df.columns:
        out["a_0"] = fourier_df["A_0"]
    for h in harmonics:
        s = _harmonic_suffix(h)
        A, B = fourier_df[f"A_{h}"], fourier_df[f"B_{h}"]
        out[f"amp{s}"] = np.sqrt(A**2 + B**2)
        out[f"phase{s}"] = np.mod(np.arctan2(B, A), 2 * np.pi)
    return out


def phase_to_hours(phase, period: float = 24.0):
    """Phase in radians -> time in ``[0, period)`` (e.g. ZT hours for a circadian clock)."""
    return np.mod(phase, 2 * np.pi) * period / (2 * np.pi)


def read_reference_clock(
    source: str | os.PathLike | pd.DataFrame,
    gene_column: str = "gene",
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read an amp/phase reference clock (CSV path or DataFrame) indexed by gene.

    Columns are read by name, never by position. ``rename_map`` maps gene aliases
    of the reference onto the names used in the dataset (e.g. ``{"Bmal1": "Arntl"}``):
    ``DecoderPrior`` only tolerates case differences, so unmatched aliases are dropped.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)
    if gene_column in df.columns:
        df = df.set_index(gene_column)
    df.index = df.index.astype(str)
    if rename_map:
        df = df.rename(index=rename_map)
    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate genes in reference clock: {dupes}")
    df.index.name = gene_column
    return df


def load_reference_clock(
    source: str | os.PathLike | pd.DataFrame,
    gene_column: str = "gene",
    rename_map: dict[str, str] | None = None,
    n_harm: int | None = None,
) -> pd.DataFrame:
    """Read a reference clock and return its Fourier table (``A_0, A_1, B_1, ...``).

    ``n_harm`` keeps only the first ``n_harm`` harmonics (None keeps all of them).
    """
    fourier = amp_phase_to_fourier(read_reference_clock(source, gene_column, rename_map))
    if n_harm is not None:
        keep = [c for c in fourier.columns if c == "A_0" or int(c.split("_")[1]) <= n_harm]
        fourier = fourier[keep]
    return fourier
