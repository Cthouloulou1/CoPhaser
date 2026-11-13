"""Opt-in covariate_encoder_detach: the covariate effects removed from the rhythmic encoder
input are detached there, so the losses on the encoder output (phase prior, entropy, KL of
theta, ...) cannot train the effects; only the generative model does."""
import numpy as np
import pandas as pd
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
EFFECT_LAYERS = ("covariate_mean_decoder", "covariate_amp_decoder", "covariate_phase_decoder")


def simulate(n_donors=60, seed=0):
    donors = sample_donors(n_donors, p_female=0.5, rng=seed)
    eff = CovariateEffects.zeros(list(FOURIER.index) + [f"BG_{i:04d}" for i in range(20)], COVS)
    eff.mesor.loc["Dbp", "sex"] = 0.5
    eff.log_amp.loc["Per3", "sex"] = 0.5
    return simulate_with_covariates(FOURIER, donors, eff, n_background_genes=20, library_size=2e6,
                                    dispersion=0.05, rng=seed + 1).adata


def make_model(ad, detach=None, seed=0):
    torch.manual_seed(seed)
    clock = list(ad.var_names[ad.var["is_clock_gene"].to_numpy()])
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names), n_latent=3, n_harm=1)
    DecoderPrior.load_fourier_coefficients_prior(FOURIER, model, freeze_defined_weights=True)
    model.load_anndata(ad, layer_to_use="counts", covariate_keys=["age", "sex"], categorical_covariates=["sex"],
                       covariate_amplitude=True, covariate_phase=True, covariate_encoder_detach=detach)
    return model


def effect_grads(model, loss):
    model.zero_grad()
    loss.backward()
    return {k: getattr(model, k).weight.grad for k in EFFECT_LAYERS}


def encoder_and_generative_outputs(model):
    model.train()
    with torch.no_grad():  # non-zero effects so that every path is active
        for k in EFFECT_LAYERS:
            getattr(model, k).weight.normal_(0, 0.1)
    return model.forward(model.variable_genes, model.rhythmic_genes, model.library_size, -1,
                         use_max_posterior=True, covariates=model.covariates)


def test_default_off_and_values_unchanged():
    ad = simulate()
    assert make_model(ad).covariate_encoder_detach is False
    g0, s0 = encoder_and_generative_outputs(make_model(ad, detach=False))
    g1, s1 = encoder_and_generative_outputs(make_model(ad, detach=True))
    # detaching changes gradients only, never values
    torch.testing.assert_close(s0["mu_theta"], s1["mu_theta"], rtol=0, atol=0)
    torch.testing.assert_close(s0["kappa_theta"], s1["kappa_theta"], rtol=0, atol=0)
    torch.testing.assert_close(g0["px_rate"], g1["px_rate"], rtol=0, atol=0)


def test_encoder_losses_no_longer_reach_the_effects():
    ad = simulate()
    for detach in (False, True):
        model = make_model(ad, detach=detach)
        gen, space = encoder_and_generative_outputs(model)
        enc_loss = space["mu_theta"].sum() + space["kappa_theta"].sum()
        grads = effect_grads(model, enc_loss)
        mean_amp = [grads[k] for k in ("covariate_mean_decoder", "covariate_amp_decoder")]
        if detach:
            assert all(g is None or torch.count_nonzero(g) == 0 for g in mean_amp)
        else:
            assert all(g is not None and torch.count_nonzero(g) > 0 for g in mean_amp)
        # the phase effects never enter the encoder input
        g = grads["covariate_phase_decoder"]
        assert g is None or torch.count_nonzero(g) == 0
        # the generative model still trains all the effects
        gen, space = encoder_and_generative_outputs(model)
        grads = effect_grads(model, gen["px_rate"].log().sum())
        assert all(g is not None and torch.count_nonzero(g) > 0 for g in grads.values())


def train_encoder_losses_only(ad, detach):
    """Fixed phases for the decoder and zero likelihood weights: the only losses left are on
    the encoder output (phase prior, entropy, KL, MI) and on z."""
    model = make_model(ad, detach=detach)
    model.add_phase_prior(ad, "true_phase", 5.0)
    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"], unfreeze_epoch_layer=[],
              rhythmic_likelihood_weight=0.0, non_rhythmic_likelihood_weight=0.0,
              decoder_phase_from_prior=True, phase_prior_anneal_epochs=10**6)
    trainer = Trainer(model, Loss.compute_loss, cycle="circadian", **kw)
    trainer.train_model(n_epochs=3, lr=1e-2, subsample=None, device="cpu", silent=True)
    model.to("cpu")
    return model


def test_training_with_encoder_losses_only_leaves_detached_effects_at_zero():
    ad = simulate()
    leaky = train_encoder_losses_only(ad, detach=False)
    clean = train_encoder_losses_only(ad, detach=True)
    for k in ("covariate_mean_decoder", "covariate_amp_decoder"):
        assert torch.count_nonzero(getattr(leaky, k).weight) > 0, k
        assert torch.count_nonzero(getattr(clean, k).weight) == 0, k


def test_save_load_and_old_checkpoints(tmp_path):
    ad = simulate()
    model = make_model(ad, detach=True)
    model.save(tmp_path / "m.pt")
    loaded = CoPhaser.load(tmp_path / "m.pt")
    assert loaded.covariate_encoder_detach is True
    # load_anndata without the argument keeps the saved setting; an explicit value overrides it
    loaded.load_anndata(ad, layer_to_use="counts")
    assert loaded.covariate_encoder_detach is True
    loaded.load_anndata(ad, layer_to_use="counts", covariate_encoder_detach=False)
    assert loaded.covariate_encoder_detach is False
    # checkpoint saved before the option existed
    ckpt = torch.load(tmp_path / "m.pt", weights_only=False)
    del ckpt["covariate_encoder_detach"]
    torch.save(ckpt, tmp_path / "old.pt")
    assert CoPhaser.load(tmp_path / "old.pt").covariate_encoder_detach is False
    assert np.isfinite(CoPhaser.load(tmp_path / "old.pt").covariate_effects()["value"]).all()


def test_gtex_train_flag(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gtex_train import parse_args

    monkeypatch.setattr(sys, "argv", ["gtex_train.py", "--tissues", "thyroid"])
    assert parse_args().covariate_encoder_detach is False
    monkeypatch.setattr(sys, "argv", ["gtex_train.py", "--tissues", "thyroid", "--covariate-encoder-detach"])
    assert parse_args().covariate_encoder_detach is True
