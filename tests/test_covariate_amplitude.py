"""Step 5: covariates modulate the amplitude of the rhythm, F_g * exp(gamma_g . c)."""
import numpy as np
import pandas as pd
import pytest
import torch

from cophaser.cycle_configs import CYCLE_TRAINER_DEFAULTS
from cophaser.loss import Loss
from cophaser.model.CoPhaser import CoPhaser
from cophaser.model.decoder_prior import DecoderPrior
from cophaser.simulation_covariates import CovariateEffects, sample_donors, simulate_with_covariates
from cophaser.trainer import Trainer

FOURIER = pd.DataFrame(
    {"A_0": [-9.0, -9.5, -10.0, -9.0, -9.5, -9.0], "A_1": [1.2, 0.0, -0.9, 0.4, -0.7, 0.8],
     "B_1": [0.0, 1.2, 0.5, -1.0, -0.7, 0.8]},
    index=["Dbp", "Nr1d1", "Per3", "Arntl", "Cry1", "Tef"],
)
COVS = ["age_std", "sex", "condition"]


def simulate(amp_genes=(), effect=0.7, n_donors=300, seed=0):
    donors = sample_donors(n_donors, p_female=0.5, rng=seed)
    eff = CovariateEffects.zeros(list(FOURIER.index) + [f"BG_{i:04d}" for i in range(20)], COVS)
    for g in amp_genes:
        eff.log_amp.loc[g, "sex"] = effect
    return simulate_with_covariates(FOURIER, donors, eff, n_background_genes=20, library_size=2e6,
                                    dispersion=0.05, rng=seed + 1).adata


def make_model(ad, amplitude=True, prior=True, seed=0):
    torch.manual_seed(seed)
    clock = list(ad.var_names[ad.var["is_clock_gene"].to_numpy()])
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names), n_latent=3, n_harm=1)
    if prior:
        DecoderPrior.load_fourier_coefficients_prior(FOURIER, model, freeze_defined_weights=True)
    model.load_anndata(ad, layer_to_use="counts", covariate_keys=["sex"], categorical_covariates=["sex"],
                       covariate_amplitude=amplitude)
    return model


def outputs(model):
    model.eval()
    return model.forward(model.variable_genes, model.rhythmic_genes, model.library_size, -1,
                         use_max_posterior=True, covariates=model.covariates)


def test_default_has_no_amplitude_layer():
    model = make_model(simulate(n_donors=40), amplitude=False)
    assert not hasattr(model, "covariate_amp_decoder")
    assert "covariate_amp_decoder.weight" not in model.state_dict()
    assert set(model.covariate_effects()["kind"]) == {"mesor"}
    assert outputs(model)[0]["covariate_log_amp"] is None


def test_zero_init_gives_the_same_model_as_without_amplitude():
    ad = simulate(n_donors=40)
    with_amp, without = make_model(ad, amplitude=True), make_model(ad, amplitude=False)
    assert torch.count_nonzero(with_amp.covariate_amp_decoder.weight) == 0
    g1, s1 = outputs(with_amp)
    g0, s0 = outputs(without)
    torch.testing.assert_close(g1["px_rate"], g0["px_rate"])
    torch.testing.assert_close(s1["theta"], s0["theta"])
    eff = with_amp.covariate_effects()
    assert set(eff["kind"]) == {"mesor", "log_amp"} and (eff["value"] == 0).all()


def test_gamma_scales_the_rhythm_and_the_encoder_input():
    ad = simulate(n_donors=40)
    model = make_model(ad)
    g0, _ = outputs(model)
    with torch.no_grad():
        model.covariate_amp_decoder.weight.fill_(0.5)
    g1, _ = outputs(model)
    c = model.covariates  # (n, 1): sex[female]
    torch.testing.assert_close(g1["F"], g0["F"] * torch.exp(0.5 * c))
    enc = model.rhythmic_encoder
    x, lib = model.rhythmic_genes[:5], model.library_size[:5].view(-1, 1)
    a = enc.normalize_expression(x, lib, use_mu_z=False)
    b = enc.normalize_expression(x, lib, use_mu_z=False, covariate_log_amp=torch.full_like(x, 0.3))
    torch.testing.assert_close(b, a * np.exp(-0.3))


def test_save_load_and_old_checkpoints(tmp_path):
    model = make_model(simulate(n_donors=40))
    with torch.no_grad():
        model.covariate_amp_decoder.weight.normal_()
    model.save(tmp_path / "m.pt")
    loaded = CoPhaser.load(tmp_path / "m.pt")
    assert loaded.covariate_amplitude
    torch.testing.assert_close(loaded.covariate_amp_decoder.weight, model.covariate_amp_decoder.weight)
    pd.testing.assert_frame_equal(loaded.covariate_effects(), model.covariate_effects())
    # a checkpoint saved before step 5 (mesor-only model, no "covariate_amplitude" key)
    old = make_model(simulate(n_donors=40), amplitude=False)
    old.save(tmp_path / "old.pt")
    ckpt = torch.load(tmp_path / "old.pt", weights_only=False)
    del ckpt["covariate_amplitude"]
    torch.save(ckpt, tmp_path / "old.pt")
    assert not CoPhaser.load(tmp_path / "old.pt").covariate_amplitude


def test_amplitude_cannot_be_added_to_a_mesor_only_model():
    ad = simulate(n_donors=40)
    model = make_model(ad, amplitude=False)
    with pytest.raises(ValueError):
        model.add_covariates(ad, ["sex"], ["sex"], amplitude=True)


def test_strong_amplitude_effect_is_recovered_with_frozen_clock():
    genes = ["Dbp", "Per3", "Cry1"]
    ad = simulate(amp_genes=genes, effect=0.7, n_donors=600)
    np.random.seed(0)
    model = make_model(ad)
    # beta2 0.99 as recommended for deep data; gamma needs ~200 epochs to converge here
    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"], unfreeze_epoch_layer=[], adam_betas=(0.9, 0.99))
    Trainer(model, Loss.compute_loss, **kw).train_model(n_epochs=200, lr=1e-2, subsample=None, device="cpu",
                                                        silent=True)
    eff = model.covariate_effects()
    gamma = eff[eff["kind"] == "log_amp"].set_index("gene")["value"]
    clock = list(FOURIER.index)
    others = gamma.loc[[g for g in clock if g not in genes]]
    assert ((gamma.loc[genes] - others.median()).between(0.45, 0.95)).all(), gamma.loc[clock]  # truth 0.7
    assert (others.abs() < 0.25).all(), others
