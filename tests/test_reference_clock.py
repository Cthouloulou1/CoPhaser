from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cophaser.reference_clock import (
    amp_phase_to_fourier,
    fourier_to_amp_phase,
    load_reference_clock,
    phase_to_hours,
    read_reference_clock,
)

REFERENCE_CSV = Path(__file__).resolve().parents[1] / "reference" / "clock_reference_learned.csv"


@pytest.fixture
def toy_clock():
    return pd.DataFrame(
        {
            "a_0": [-11.0, -10.0, -9.0],
            "amp": [1.5, 0.5, 0.0],
            "phase": [5.9, 0.0, 1.0],
            "amp2": [0.4, 0.1, 0.2],
            "phase2": [2.3, 4.0, 6.0],
            "tau": [0.3, 0.1, 0.2],  # extra QC column, must be ignored
        },
        index=pd.Index(["Bmal1", "Per2", "Flat"], name="gene"),
    )


def test_convention_matches_decoder(toy_clock):
    """A_h = amp*cos(phase), B_h = amp*sin(phase), columns ordered as the decoder."""
    f = amp_phase_to_fourier(toy_clock)
    assert list(f.columns) == ["A_0", "A_1", "B_1", "A_2", "B_2"]
    np.testing.assert_allclose(f["A_1"], toy_clock["amp"] * np.cos(toy_clock["phase"]))
    np.testing.assert_allclose(f["B_2"], toy_clock["amp2"] * np.sin(toy_clock["phase2"]))
    np.testing.assert_allclose(f["A_0"], toy_clock["a_0"])


def test_log_mean_peaks_at_phase(toy_clock):
    """With f = [cos h*theta, sin h*theta], A*cos + B*sin peaks at theta = phase."""
    f = amp_phase_to_fourier(toy_clock[["amp", "phase"]])
    theta = np.linspace(0, 2 * np.pi, 10_000, endpoint=False)
    gene = "Bmal1"
    curve = f.loc[gene, "A_1"] * np.cos(theta) + f.loc[gene, "B_1"] * np.sin(theta)
    peak = theta[np.argmax(curve)]
    assert abs(np.angle(np.exp(1j * (peak - toy_clock.loc[gene, "phase"])))) < 1e-3


def test_round_trip(toy_clock):
    back = fourier_to_amp_phase(amp_phase_to_fourier(toy_clock))
    for col in ["a_0", "amp", "amp2"]:
        np.testing.assert_allclose(back[col], toy_clock[col], atol=1e-12)
    nonzero = toy_clock["amp"] > 0  # phase undefined when amp == 0
    np.testing.assert_allclose(back.loc[nonzero, "phase"], toy_clock.loc[nonzero, "phase"], atol=1e-12)
    np.testing.assert_allclose(back["phase2"], toy_clock["phase2"], atol=1e-12)


def test_rename_and_harmonic_truncation(toy_clock):
    f = load_reference_clock(toy_clock.reset_index(), rename_map={"Bmal1": "Arntl"}, n_harm=1)
    assert "Arntl" in f.index and "Bmal1" not in f.index
    assert list(f.columns) == ["A_0", "A_1", "B_1"]


def test_no_harmonics_raises():
    with pytest.raises(ValueError):
        amp_phase_to_fourier(pd.DataFrame({"a_0": [1.0]}, index=["g"]))


@pytest.mark.skipif(not REFERENCE_CSV.exists(), reason="reference clock CSV not available")
def test_reference_csv():
    """Read by column name; phase_hours in the CSV must equal phase * 24 / 2pi."""
    raw = read_reference_clock(REFERENCE_CSV)
    assert {"a_0", "amp", "phase", "amp2", "phase2", "phase_hours"} <= set(raw.columns)
    np.testing.assert_allclose(phase_to_hours(raw["phase"]), raw["phase_hours"], atol=1e-6)
    f = load_reference_clock(REFERENCE_CSV)
    assert list(f.columns) == ["A_0", "A_1", "B_1", "A_2", "B_2"]
    back = fourier_to_amp_phase(f)
    np.testing.assert_allclose(back["amp"], raw["amp"], atol=1e-12)
    np.testing.assert_allclose(back["phase"], np.mod(raw["phase"], 2 * np.pi), atol=1e-12)
