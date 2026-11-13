"""Step 9: one phase per donor shared by its tissues (scripts/gtex_shared_phase.py,
gtex_train.py --phase-table, CoPhaser.grid_phase(return_loglik=True))."""
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gtex_shared_phase import shared_phases, tissue_loglik  # noqa: E402
from gtex_train import bootstrap_donors, parse_args, phase_table_lookup  # noqa: E402

from cophaser.model.CoPhaser import CoPhaser  # noqa: E402
from cophaser.model.decoder_prior import DecoderPrior  # noqa: E402

GENES = ["ARNTL", "NR1D1", "DBP", "PER3", "CRY1", "TEF"]
H = 24 / (2 * np.pi)


def clock_table(phases, amp=0.3):
    phases = np.asarray(phases, float)
    return pd.DataFrame({"A_1": amp * np.cos(phases), "B_1": amp * np.sin(phases)}, index=GENES)


def tissue(donors, theta, clock, rng, lib=3e7, a0=-9.0, r=10.0):
    """Counts with log-mean a0 + amp cos(theta - phase) (reference-clock convention), NB with
    dispersion 1 / r."""
    amp = np.hypot(clock["A_1"], clock["B_1"]).to_numpy()
    ph = np.arctan2(clock["B_1"], clock["A_1"]).to_numpy()
    mu = lib * np.exp(a0 + amp[None] * np.cos(theta[:, None] - ph[None]))
    counts = rng.negative_binomial(r, r / (r + mu)).astype(np.float32)
    obs = pd.DataFrame({"donor": donors, "library_size": np.full(len(donors), lib)},
                       index=[f"{d}-{i}" for i, d in enumerate(donors)])
    ad = anndata.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=GENES))
    ad.layers["counts"] = counts
    return ad


def err_h(pred, truth):
    return float(np.median(np.abs(np.angle(np.exp(1j * (pred - truth)))))) * H


def test_grid_phase_default_output_unchanged_and_loglik_consistent():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, 50)
    clock = clock_table(np.linspace(0, 5, 6))
    ad = tissue([f"D{i}" for i in range(50)], theta, clock, rng)
    torch.manual_seed(0)
    model = CoPhaser(rhythmic_gene_names=GENES, context_genes=GENES, n_latent=2, n_harm=1)
    DecoderPrior.load_fourier_coefficients_prior(clock, model, freeze_defined_weights=True)
    model.load_anndata(ad, layer_to_use="counts", library_size_field="library_size")
    default = model.grid_phase(dispersion=0.1)
    assert set(default) == {"theta", "resultant", "kappa"}
    full = model.grid_phase(dispersion=0.1, return_loglik=True)
    assert full["loglik"].shape == (50, 96) and full["grid"].shape == (96,)
    np.testing.assert_array_equal(full["theta"], full["grid"][full["loglik"].argmax(1)])
    again = CoPhaser.phase_from_loglik(full["loglik"], full["grid"])
    for k in default:
        np.testing.assert_allclose(again[k], default[k])
    assert err_h(default["theta"], theta) < 4  # same frame as the clock table


def test_shared_phase_is_more_accurate_than_one_tissue():
    rng = np.random.default_rng(1)
    n = 300
    donors = np.array([f"D{i:03d}" for i in range(n)])
    truth = rng.uniform(-np.pi, np.pi, n)
    # three tissues, each with its own clock in the same time frame
    clocks = [clock_table(np.linspace(0, 5, 6) + off, amp) for off, amp in ((0.0, 0.3), (0.6, 0.25), (-0.4, 0.35))]
    keep = [np.ones(n, bool), np.ones(n, bool), rng.random(n) < 0.8]  # some donors lack tissue 3
    lls, ds, single = [], [], []
    for clock, k in zip(clocks, keep):
        ad = tissue(donors[k], truth[k], clock, rng)
        ll, grid = tissue_loglik(ad, clock, GENES, dispersion=0.1, n_points=96)
        lls.append(ll)
        ds.append(donors[k])
        single.append(err_h(CoPhaser.phase_from_loglik(ll, grid)["theta"], truth[k]))
    sh = shared_phases(ds, lls, grid, ["t1", "t2", "t3"])
    assert list(sh.index) == list(donors)
    np.testing.assert_array_equal(sh["n_tissues"].to_numpy(), np.where(keep[2], 3, 2))
    assert sh.loc[donors[~keep[2]][0], "tissues"] == "t1;t2"
    shared = err_h(sh.loc[donors, "theta"].to_numpy(), truth)
    assert shared < 0.75 * min(single), (shared, single)
    single_kappa = np.median(CoPhaser.phase_from_loglik(lls[0], grid)["kappa"])
    assert sh["kappa"].median() > single_kappa
    # one tissue only: the shared phase is the per-sample phase
    one = shared_phases(ds[:1], lls[:1], grid)
    np.testing.assert_array_equal(one["theta"].to_numpy(), CoPhaser.phase_from_loglik(lls[0], grid)["theta"])


def test_phase_table_lookup_maps_donors_also_after_bootstrap():
    obs = pd.DataFrame({"donor": ["A", "B", "C", "A"]}, index=["s1", "s2", "s3", "s4"])
    table = pd.DataFrame({"donor": ["A", "B"], "theta": [0.5, -1.0], "kappa": [10.0, 3.0]})
    theta, kappa, found = phase_table_lookup(obs, table)
    np.testing.assert_array_equal(found, [True, True, False, True])
    np.testing.assert_allclose(theta[found], [0.5, -1.0, 0.5])
    np.testing.assert_allclose(kappa[found], [10.0, 3.0, 10.0])
    ad = anndata.AnnData(X=np.zeros((4, 1), np.float32), obs=obs)
    boot = bootstrap_donors(ad, 3)
    theta, _, found = phase_table_lookup(boot.obs, table)
    expected = boot.obs["donor"].map({"A": 0.5, "B": -1.0}).to_numpy(float)
    np.testing.assert_array_equal(found, np.isfinite(expected))
    np.testing.assert_allclose(theta[found], expected[found])


def test_phase_table_option_defaults_to_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gtex_train.py", "--tissues", "x"])
    assert parse_args().phase_table is None
