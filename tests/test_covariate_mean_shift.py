"""Step 4: linear covariate mean shift W_c . c in the decoder and the rhythmic encoder."""
import numpy as np
import pandas as pd
import pytest
import torch

from cophaser.cycle_configs import CYCLE_TRAINER_DEFAULTS
from cophaser.loss import Loss
from cophaser.model.CoPhaser import CoPhaser
from cophaser.simulation_covariates import CovariateEffects, sample_donors, simulate_with_covariates
from cophaser.trainer import Trainer

FOURIER = pd.DataFrame(
    {"A_0": [-9.0, -9.5, -10.0, -9.0], "A_1": [1.5, 0.0, -0.8, 0.3], "B_1": [0.0, 1.5, 0.5, -0.9]},
    index=["Dbp", "Nr1d1", "Per3", "Arntl"],
)
COVS = ["age_std", "sex", "condition"]


def simulate(effect_genes=(), effect=1.0, n_donors=400, seed=0):
    donors = sample_donors(n_donors, p_female=0.5, rng=seed)
    eff = CovariateEffects.zeros(list(FOURIER.index) + [f"BG_{i:04d}" for i in range(20)], COVS)
    for g in effect_genes:
        eff.mesor.loc[g, "sex"] = effect
    return simulate_with_covariates(FOURIER, donors, eff, n_background_genes=20, library_size=2e5, rng=seed + 1)


def make_model(ad, keys=("sex",)):
    clock = list(ad.var_names[ad.var["is_clock_gene"].to_numpy()])
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names), n_latent=3, n_harm=1)
    model.load_anndata(ad, layer_to_use="counts", covariate_keys=list(keys), categorical_covariates=["sex"])
    return model


def test_layer_starts_at_zero_and_learns():
    ad = simulate().adata
    model = make_model(ad)
    W = model.covariate_mean_decoder.weight
    assert W.shape == (len(model.context_genes), 1)
    assert torch.count_nonzero(W) == 0
    gen, space = model.forward(model.variable_genes, model.rhythmic_genes, model.library_size,
                               epoch=0, covariates=model.covariates)
    assert torch.count_nonzero(gen["covariate_mean"]) == 0
    # gradients reach W_c
    gen["px_rate"].log().mean().backward()
    assert W.grad is not None and torch.count_nonzero(W.grad) > 0


def test_encoder_input_is_corrected_by_covariate_shift():
    """Shifting the counts of a gene by exp(d) and the covariate effect by d leaves the
    rhythmic encoder input unchanged."""
    ad = simulate().adata
    model = make_model(ad)
    enc = model.rhythmic_encoder
    x = model.rhythmic_genes[:5]
    lib = model.library_size[:5].view(-1, 1)
    shift = torch.full((5, x.shape[1]), 0.7)
    a = enc.normalize_expression(x, lib, use_mu_z=False)
    b = enc.normalize_expression(x * np.exp(0.7), lib, use_mu_z=False, covariate_shift=shift)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_save_load_keeps_effects(tmp_path):
    ad = simulate().adata
    model = make_model(ad)
    with torch.no_grad():
        model.covariate_mean_decoder.weight.normal_()
    model.save(tmp_path / "m.pt")
    loaded = CoPhaser.load(tmp_path / "m.pt")
    torch.testing.assert_close(loaded.covariate_mean_decoder.weight, model.covariate_mean_decoder.weight)
    pd.testing.assert_frame_equal(loaded.covariate_effects(), model.covariate_effects())


def test_model_without_covariates_has_no_layer():
    ad = simulate().adata
    clock = list(ad.var_names[ad.var["is_clock_gene"].to_numpy()])
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names), n_latent=3, n_harm=1)
    model.load_anndata(ad, layer_to_use="counts")
    assert not hasattr(model, "covariate_mean_decoder")
    assert model.covariate_effects().empty
    assert "covariate_mean_decoder.weight" not in model.state_dict()


def test_strong_mean_effect_is_recovered():
    genes = ["Dbp", "Per3", "BG_0000", "BG_0001", "BG_0002"]
    ad = simulate(effect_genes=genes, effect=1.0, n_donors=600).adata
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(ad)
    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"])
    Trainer(model, Loss.compute_loss, **kw).train_model(n_epochs=40, lr=1e-2, subsample=None, device="cpu", silent=True)
    eff = model.covariate_effects().set_index("gene")["value"]  # column sex[1]: female - male
    # Normalising by total counts makes effects identifiable only up to a constant per
    # covariate (raising 5 of 24 genes lowers the fraction of all the others): compare
    # to the unaffected genes.
    others = eff.drop(genes)
    relative = eff.loc[genes] - np.median(others)
    assert (relative > 0.7).all() and (relative < 1.3).all(), relative
    assert np.median(np.abs(others - np.median(others))) < 0.1, others


def test_adam_betas_option(monkeypatch):
    """Default betas are torch's; adam_betas reaches the optimizer."""
    ad = simulate().adata
    model = make_model(ad)
    assert Trainer(model, Loss.compute_loss).adam_betas == (0.9, 0.999)
    seen = {}
    real_adam = torch.optim.Adam

    def spy(params, lr, **kw):
        seen.setdefault("betas", kw.get("betas"))
        return real_adam(params, lr=lr, **kw)

    monkeypatch.setattr(torch.optim, "Adam", spy)
    Trainer(model, Loss.compute_loss, adam_betas=(0.9, 0.99)).train_model(
        n_epochs=2, subsample=None, device="cpu", silent=True)
    assert seen["betas"] == (0.9, 0.99)


# ---------------------------------------------------------------- z encoder input


def _model(ad, **kw):
    clock = list(ad.var_names[ad.var["is_clock_gene"].to_numpy()])
    return CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names), n_latent=3, n_harm=1, **kw)


def test_z_input_default_is_raw_and_invalid_raises():
    ad = simulate().adata
    assert _model(ad).z_input == "raw"
    with pytest.raises(ValueError):
        _model(ad, z_input="log")


def test_z_input_cp10k_normalises_and_stays_finite_on_huge_counts():
    ad = simulate().adata
    torch.manual_seed(0)
    raw = _model(ad)
    torch.manual_seed(0)
    cp = _model(ad, z_input="cp10k")
    x = torch.full((4, len(raw.context_genes)), 1e7)
    lib = torch.full((4,), 1e9)
    m_raw, _ = raw.latent_encoder_inference(x, lib)
    m_cp, v_cp = cp.latent_encoder_inference(x, lib)
    # cp10k input = log1p(1e7 / 1e9 * 1e4) = log1p(100); raw input = log1p(1e7)
    m_ref, _ = raw.latent_encoder_inference(torch.full_like(x, 100.0), torch.ones(4))
    torch.testing.assert_close(m_cp, m_ref)
    assert not torch.allclose(m_cp, m_raw)
    assert torch.isfinite(v_cp).all()


def test_z_input_saved_and_old_checkpoints_default_to_raw(tmp_path):
    ad = simulate().adata
    m = _model(ad, z_input="cp10k")
    m.save(tmp_path / "m.pt")
    assert CoPhaser.load(tmp_path / "m.pt").z_input == "cp10k"
    ckpt = torch.load(tmp_path / "m.pt", weights_only=False)
    del ckpt["init_params"]["z_input"]
    torch.save(ckpt, tmp_path / "old.pt")
    assert CoPhaser.load(tmp_path / "old.pt").z_input == "raw"
