"""The simulator must put in the data exactly the effects it claims (checked with a
cosinor on the TRUE phase, at bulk depth, with many donors)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cophaser.reference_clock import load_reference_clock
from cophaser.simulation_covariates import (
    CovariateEffects,
    TissueSpec,
    age_to_gtex_bin,
    cosinor_by_group,
    sample_donors,
    simulate_with_covariates,
)

REFERENCE_CSV = Path(__file__).resolve().parents[1] / "reference" / "clock_reference_learned.csv"
H = 24 / (2 * np.pi)  # radians -> hours


def wrap(x):
    return np.angle(np.exp(1j * np.asarray(x)))


@pytest.fixture(scope="module")
def fourier():
    if REFERENCE_CSV.exists():
        return load_reference_clock(REFERENCE_CSV, n_harm=2)
    # fallback: a few synthetic clock genes
    return pd.DataFrame(
        {"A_0": [-10.0, -10.0, -10.5], "A_1": [1.5, 0.0, -0.8], "B_1": [0.0, 1.5, 0.5],
         "A_2": [0.1, 0.0, 0.0], "B_2": [0.0, 0.1, 0.0]},
        index=["Dbp", "Nr1d1", "Per3"],
    )


def by_gene(table, group):
    return table[table["group"] == group].set_index("gene")


def test_age_bins():
    assert list(age_to_gtex_bin([20, 29.9, 30, 79])) == ["20-29", "20-29", "30-39", "70-79"]


def test_shapes_and_shared_donor_phase(fourier):
    donors = sample_donors(50, rng=0)
    eff = CovariateEffects.zeros(fourier.index, ["age_std", "sex", "condition"])
    tissues = [TissueSpec("A"), TissueSpec("B", phase_shift=1.0, p_sampled=0.6)]
    res = simulate_with_covariates(fourier, donors, eff, tissues=tissues, n_background_genes=20, rng=1)
    ad = res.adata
    assert ad.n_vars == len(fourier) + 20
    assert (ad.obs["tissue"] == "A").sum() == 50
    # all observations of a donor share the same phase when there is no jitter
    assert ad.obs.groupby("donor", observed=True)["true_phase"].nunique().max() == 1
    assert np.all(np.isfinite(ad.layers["counts"].data))


def test_sex_effects_are_recovered(fourier):
    genes = ["Dbp", "Nr1d1", "Per3"]
    donors = sample_donors(3000, p_female=0.5, rng=2)
    eff = CovariateEffects.zeros(fourier.index, ["age_std", "sex", "condition"])
    eff.log_amp.loc["Dbp", "sex"] = np.log(0.5)  # females: half amplitude
    eff.phase.loc["Nr1d1", "sex"] = 2 / H  # females: peak 2 h later
    eff.mesor.loc["Per3", "sex"] = 0.5  # females: higher mean
    res = simulate_with_covariates(fourier, donors, eff, n_background_genes=50, rng=3)
    fit = cosinor_by_group(res.adata, genes, res.adata.obs["sex"].to_numpy())
    m, f = by_gene(fit, 0), by_gene(fit, 1)

    assert np.log(f.loc["Dbp", "amp"] / m.loc["Dbp", "amp"]) == pytest.approx(np.log(0.5), abs=0.1)
    assert wrap(f.loc["Nr1d1", "phase"] - m.loc["Nr1d1", "phase"]) * H == pytest.approx(2.0, abs=0.3)
    assert f.loc["Per3", "mesor"] - m.loc["Per3", "mesor"] == pytest.approx(0.5, abs=0.1)
    # genes without effect do not move
    assert np.log(f.loc["Nr1d1", "amp"] / m.loc["Nr1d1", "amp"]) == pytest.approx(0, abs=0.1)
    assert wrap(f.loc["Dbp", "phase"] - m.loc["Dbp", "phase"]) * H == pytest.approx(0, abs=0.3)


def test_continuous_age_phase_delay(fourier):
    donors = sample_donors(4000, rng=4)
    eff = CovariateEffects.zeros(fourier.index, ["age_std", "sex", "condition"])
    eff.phase.loc["Dbp", "age_std"] = 1 / H  # 1 h later per 15 years
    res = simulate_with_covariates(fourier, donors, eff, n_background_genes=50, rng=5)
    a = res.adata.obs["age_std"].to_numpy()
    groups = np.where(a < -0.7, "young", np.where(a > 0.7, "old", "mid"))
    fit = cosinor_by_group(res.adata, ["Dbp"], groups)
    d_age = a[groups == "old"].mean() - a[groups == "young"].mean()
    d_phase = wrap(by_gene(fit, "old").loc["Dbp", "phase"] - by_gene(fit, "young").loc["Dbp", "phase"]) * H
    assert d_phase == pytest.approx(1.0 * d_age, abs=0.3)


def test_tissue_phase_shift(fourier):
    donors = sample_donors(2000, rng=6)
    eff = CovariateEffects.zeros(fourier.index, ["age_std", "sex", "condition"])
    tissues = [TissueSpec("A"), TissueSpec("B", phase_shift=3 / H)]
    res = simulate_with_covariates(fourier, donors, eff, tissues=tissues, n_background_genes=50, rng=7)
    fit = cosinor_by_group(res.adata, ["Dbp"], res.adata.obs["tissue"].astype(str).to_numpy())
    d = wrap(by_gene(fit, "B").loc["Dbp", "phase"] - by_gene(fit, "A").loc["Dbp", "phase"]) * H
    assert d == pytest.approx(3.0, abs=0.3)
