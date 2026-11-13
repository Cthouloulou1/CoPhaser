"""Trainer options phase_prior_refresh_every (grid re-estimation of the phase prior) and
decoder_phase_from_prior (generative model fed with the prior phase), and
CoPhaser.forward(theta_override=...)."""
import numpy as np
import pytest
import torch

from cophaser.loss import Loss
from cophaser.trainer import Trainer

from test_grid_phase import H, circ_err_h, make_model, simulate


def test_theta_override_feeds_the_decoder_and_nan_keeps_encoder():
    ad = simulate(n_donors=40)
    model = make_model(ad)
    model.eval()
    args = (model.variable_genes, model.rhythmic_genes, model.library_size, -1)
    gen0, sp0 = model.forward(*args, use_max_posterior=True)
    gen_none, _ = model.forward(*args, use_max_posterior=True, theta_override=None)
    torch.testing.assert_close(gen_none["px_rate"], gen0["px_rate"])

    override = torch.full((ad.n_obs,), 0.7)
    override[:10] = float("nan")
    gen, sp = model.forward(*args, use_max_posterior=True, theta_override=override)
    f = model.rhythmic_encoder.fourier_basis_expansion(torch.full((ad.n_obs - 10,), 0.7))
    torch.testing.assert_close(gen["F"][10:], model.rhythmic_decoder(f) * gen["lambda"][10:].view(-1, 1))
    torch.testing.assert_close(gen["px_rate"][:10], gen0["px_rate"][:10])
    torch.testing.assert_close(sp["theta"], sp0["theta"])  # encoder outputs unchanged


def _trainer(model, **kw):
    return Trainer(model, Loss.compute_loss, rhythmic_likelihood_weight=1, non_rhythmic_likelihood_weight=1,
                   phase_prior_anneal_epochs=10_000, **kw)


def test_refresh_replaces_a_wrong_prior_by_the_grid_estimate():
    ad = simulate(n_donors=200)
    model = make_model(ad)
    rng = np.random.default_rng(0)
    ad.obs["wrong"] = rng.uniform(-np.pi, np.pi, ad.n_obs)
    model.add_phase_prior(ad, "wrong", 2.0)
    truth = ad.obs["true_phase"].to_numpy()
    assert np.median(circ_err_h(model.phase_prior.numpy(), truth)) > 4
    torch.manual_seed(0)
    tr = _trainer(model, phase_prior_refresh_every=2, phase_prior_refresh_kwargs=dict(dispersion=0.05),
                  decoder_phase_from_prior=True)
    tr.train_model(n_epochs=5, subsample=None, device="cpu", silent=True)
    assert [e for e, _, _ in tr.phase_prior_refresh_log] == [2, 4]
    assert np.median(circ_err_h(model.phase_prior.numpy(), truth)) < 1.5  # clock frozen at its prior


def test_constant_refresh_kappa_and_errors():
    ad = simulate(n_donors=60)
    model = make_model(ad)
    with pytest.raises(ValueError):
        _trainer(model, decoder_phase_from_prior=True).train_model(n_epochs=2, subsample=None, device="cpu",
                                                                   silent=True)
    with pytest.raises(ValueError):
        _trainer(model, phase_prior_refresh_every=0)
    ad.obs["p"] = 0.0
    model.add_phase_prior(ad, "p", 1.0)
    tr = _trainer(model, phase_prior_refresh_every=1, phase_prior_refresh_kappa=3.0,
                  phase_prior_refresh_kwargs=dict(dispersion=0.05, n_points=24))
    tr.train_model(n_epochs=2, subsample=None, device="cpu", silent=True)
    assert torch.all(model.phase_prior_kappa == 3.0)
