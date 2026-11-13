"""Human reference clock from known phases (scripts/gtex_human_clock.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gtex_human_clock import fit_cosinor_clock  # noqa: E402

from cophaser.reference_clock import load_reference_clock  # noqa: E402


def test_cosinor_recovers_amplitude_and_peak_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    n = 2000
    theta = rng.uniform(-np.pi, np.pi, n)
    lib = rng.uniform(2e7, 6e7, n)
    truth = {"ARNTL": (-9.0, 0.4, 5.2), "DBP": (-10.5, 0.8, 1.3), "CRY2": (-9.5, 0.0, 0.0)}
    counts = np.column_stack([
        rng.negative_binomial(20, 20 / (20 + lib * np.exp(a0 + amp * np.cos(theta - ph)))) for a0, amp, ph in truth.values()
    ])
    clock = fit_cosinor_clock(counts, lib, theta, list(truth))
    for (_, r), (a0, amp, ph) in zip(clock.iterrows(), truth.values()):
        assert abs(r["amp"] - amp) < 0.03
        assert abs(r["a_0"] - a0) < 0.05
        if amp > 0:
            assert abs(np.angle(np.exp(1j * (r["phase"] - ph)))) < 0.05
            assert np.isclose(r["phase_hours"], r["phase"] * 24 / (2 * np.pi))
    assert clock.set_index("gene").loc["DBP", "r2"] > clock.set_index("gene").loc["CRY2", "r2"]
    clock.to_csv(tmp_path / "c.csv", index=False)
    f = load_reference_clock(tmp_path / "c.csv", n_harm=1)
    np.testing.assert_allclose(f.loc["DBP", "A_1"], clock.loc[1, "amp"] * np.cos(clock.loc[1, "phase"]))
    np.testing.assert_allclose(f.loc["DBP", "B_1"], clock.loc[1, "amp"] * np.sin(clock.loc[1, "phase"]))
