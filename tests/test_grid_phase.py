"""CoPhaser.grid_phase: maximum-likelihood phase on a grid with the current rhythmic decoder."""
import numpy as np
import pandas as pd
import pytest
import torch

from cophaser.model.CoPhaser import CoPhaser
from cophaser.model.decoder_prior import DecoderPrior
from cophaser.simulation_covariates import CovariateEffects, sample_donors, simulate_with_covariates

FOURIER = pd.DataFrame(
    {"A_0": [-9.0, -9.5, -10.0, -9.0, -9.5], "A_1": [1.0, 0.0, -0.8, 0.3, -0.6],
     "B_1": [0.0, 1.0, 0.5, -0.9, -0.6]},
    index=["Dbp", "Nr1d1", "Per3", "Arntl", "Cry1"],
)
H = 24 / (2 * np.pi)


def simulate(n_donors=300, seed=0):
    donors = sample_donors(n_donors, rng=seed)
    genes = list(FOURIER.index) + [f"BG_{i:04d}" for i in range(20)]
    eff = CovariateEffects.zeros(genes, ["age_std", "sex", "condition"])
    return simulate_with_covariates(FOURIER, donors, eff, n_background_genes=20, library_size=2e6,
                                    dispersion=0.05, rng=seed + 1).adata


def make_model(ad, with_prior=True):
    torch.manual_seed(0)
    clock = list(ad.var_names[ad.var["is_clock_gene"].to_numpy()])
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names), n_latent=3, n_harm=1)
    if with_prior:
        DecoderPrior.load_fourier_coefficients_prior(FOURIER, model, freeze_defined_weights=True)
    model.load_anndata(ad, layer_to_use="counts")
    return model


def circ_err_h(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b)))) * H


def test_recovers_true_phase_with_known_clock_before_training():
    ad = simulate()
    model = make_model(ad)
    out = model.grid_phase(n_points=96, dispersion=0.05)
    assert set(out) == {"theta", "resultant", "kappa"}
    assert all(v.shape == (ad.n_obs,) for v in out.values())
    err = circ_err_h(out["theta"], ad.obs["true_phase"].to_numpy())
    assert np.median(err) < 1.5, np.median(err)
    assert ((out["resultant"] >= 0) & (out["resultant"] <= 1)).all()
    assert ((out["kappa"] >= 0) & (out["kappa"] <= 50)).all()


def test_confidence_drops_without_a_clock():
    """With a decoder carrying no rhythm (zero coefficients) the grid posterior is flat."""
    ad = simulate()
    model = make_model(ad, with_prior=False)
    with torch.no_grad():
        model.rhythmic_decoder.fourier_coefficients.weight.zero_()
    flat = model.grid_phase(dispersion=0.05)
    sharp = make_model(ad).grid_phase(dispersion=0.05)
    assert np.median(flat["resultant"]) < 0.05
    assert np.median(sharp["kappa"]) > 5 * np.median(flat["kappa"]) + 1


def test_does_not_change_model_and_use_z_runs():
    ad = simulate(n_donors=80)
    model = make_model(ad)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    out = model.grid_phase(n_points=24, use_z=True, batch_size=30)  # learned dispersion, several batches
    assert np.isfinite(out["theta"]).all() and np.isfinite(out["kappa"]).all()
    assert model.training
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_errors():
    ad = simulate(n_donors=20)
    clock = list(ad.var_names[ad.var["is_clock_gene"].to_numpy()])
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names), n_latent=3, n_harm=1)
    with pytest.raises(RuntimeError):
        model.grid_phase()
    model.load_anndata(ad, layer_to_use="counts")
    with pytest.raises(ValueError):
        model.grid_phase(dispersion=0)
