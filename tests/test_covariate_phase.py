"""Step 6: covariates shift the phase of each gene's rhythm, F_g(theta - delta_g . c).

delta is only identifiable relative to the clock: a shift common to all clock genes is the
same as a shift of theta. The model therefore reports centred effects (amplitude-weighted
mean over the rhythmic genes = 0, or 0 for a reference gene)."""
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
H = 24 / (2 * np.pi)


def simulate(phase_genes=(), effect=0.6, n_donors=300, seed=0):
    donors = sample_donors(n_donors, p_female=0.5, rng=seed)
    eff = CovariateEffects.zeros(list(FOURIER.index) + [f"BG_{i:04d}" for i in range(20)], COVS)
    for g in phase_genes:
        eff.phase.loc[g, "sex"] = effect
    return simulate_with_covariates(FOURIER, donors, eff, n_background_genes=20, library_size=2e6,
                                    dispersion=0.05, rng=seed + 1).adata


def make_model(ad, phase=True, amplitude=False, reference=None, n_harm=1, seed=0):
    torch.manual_seed(seed)
    clock = list(ad.var_names[ad.var["is_clock_gene"].to_numpy()])
    model = CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names), n_latent=3, n_harm=n_harm)
    DecoderPrior.load_fourier_coefficients_prior(FOURIER, model, freeze_defined_weights=True)
    model.load_anndata(ad, layer_to_use="counts", covariate_keys=["sex"], categorical_covariates=["sex"],
                       covariate_amplitude=amplitude, covariate_phase=phase,
                       covariate_phase_reference=reference)
    return model


def outputs(model):
    model.eval()
    return model.forward(model.variable_genes, model.rhythmic_genes, model.library_size, -1,
                         use_max_posterior=True, covariates=model.covariates)


def test_default_has_no_phase_layer():
    model = make_model(simulate(n_donors=40), phase=False)
    assert not hasattr(model, "covariate_phase_decoder")
    assert "covariate_phase_decoder.weight" not in model.state_dict()
    assert "phase" not in set(model.covariate_effects()["kind"])
    assert model.covariate_phase_shift(model.covariates) is None


def test_zero_init_gives_the_same_model_as_without_phase():
    ad = simulate(n_donors=40)
    with_phase, without = make_model(ad, phase=True), make_model(ad, phase=False)
    assert torch.count_nonzero(with_phase.covariate_phase_decoder.weight) == 0
    g1, s1 = outputs(with_phase)
    g0, s0 = outputs(without)
    torch.testing.assert_close(g1["px_rate"], g0["px_rate"])
    torch.testing.assert_close(s1["theta"], s0["theta"])
    eff = with_phase.covariate_effects()
    assert set(eff["kind"]) == {"mesor", "phase"} and (eff["value"] == 0).all()


@pytest.mark.parametrize("n_harm", [1, 2])
def test_shift_rotates_each_gene_by_its_own_delay(n_harm):
    ad = simulate(n_donors=40)
    model = make_model(ad, n_harm=n_harm)
    with torch.no_grad():
        model.rhythmic_decoder.fourier_coefficients.weight.normal_()
        model.covariate_phase_decoder.weight.normal_()
    cov = model.covariates
    theta = torch.linspace(-3, 3, cov.shape[0])
    f = model.rhythmic_encoder.fourier_basis_expansion(theta)
    out = model.decode_rhythm(f, cov)
    delta = model.covariate_phase_shift(cov)  # (n, genes), centred
    for g in range(out.shape[1]):
        ref = model.rhythmic_decoder(model.rhythmic_encoder.fourier_basis_expansion(theta - delta[:, g]))[:, g]
        torch.testing.assert_close(out[:, g], ref, rtol=1e-5, atol=1e-5)
    # males (reference level) are not shifted
    male = cov[:, 0] == 0
    torch.testing.assert_close(out[male], model.rhythmic_decoder(f)[male])


def test_common_shift_is_not_identifiable_and_is_removed():
    ad = simulate(n_donors=40)
    model = make_model(ad)
    idx = model.rhythmic_gene_indices
    with torch.no_grad():
        model.covariate_phase_decoder.weight.normal_()
    W = model.covariate_phase_weight()
    amp = model.rhythmic_decoder.fourier_coefficients.weight[idx].norm(dim=1)
    w = amp / amp.sum()
    torch.testing.assert_close(w @ W[idx], torch.zeros(W.shape[1]), atol=1e-6, rtol=0)
    f = model.rhythmic_encoder.fourier_basis_expansion(torch.zeros(model.covariates.shape[0]))
    before = model.decode_rhythm(f, model.covariates)
    with torch.no_grad():
        model.covariate_phase_decoder.weight += 0.4  # same delay for every gene = a shift of theta
    torch.testing.assert_close(model.decode_rhythm(f, model.covariates), before)
    # reference gene: its delay is 0 by definition
    ref = make_model(ad, reference="Dbp")
    with torch.no_grad():
        ref.covariate_phase_decoder.weight.normal_()
    W = ref.covariate_phase_weight()
    assert torch.all(W[ref.context_genes.index("Dbp")] == 0)


def test_frozen_clock_stays_frozen():
    ad = simulate(n_donors=60)
    model = make_model(ad)
    w0 = model.rhythmic_decoder.fourier_coefficients.weight.detach().clone()
    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"], unfreeze_epoch_layer=[], adam_betas=(0.9, 0.99))
    Trainer(model, Loss.compute_loss, **kw).train_model(n_epochs=3, lr=1e-2, subsample=None, device="cpu",
                                                        silent=True)
    clock = [model.context_genes.index(g) for g in FOURIER.index]
    torch.testing.assert_close(model.rhythmic_decoder.fourier_coefficients.weight[clock], w0[clock])
    assert torch.count_nonzero(model.covariate_phase_decoder.weight) > 0


def test_save_load_and_old_checkpoints(tmp_path):
    model = make_model(simulate(n_donors=40), amplitude=True, reference="Per3")
    with torch.no_grad():
        model.covariate_phase_decoder.weight.normal_()
    model.save(tmp_path / "m.pt")
    loaded = CoPhaser.load(tmp_path / "m.pt")
    assert loaded.covariate_phase and loaded.covariate_amplitude
    assert loaded.covariate_phase_reference == "Per3"
    pd.testing.assert_frame_equal(loaded.covariate_effects(), model.covariate_effects())
    # a checkpoint saved before step 6 (no "covariate_phase" key)
    old = make_model(simulate(n_donors=40), phase=False, amplitude=True)
    old.save(tmp_path / "old.pt")
    ckpt = torch.load(tmp_path / "old.pt", weights_only=False)
    del ckpt["covariate_phase"], ckpt["covariate_phase_reference"]
    torch.save(ckpt, tmp_path / "old.pt")
    loaded = CoPhaser.load(tmp_path / "old.pt")
    assert not loaded.covariate_phase and loaded.covariate_amplitude


def test_phase_cannot_be_added_to_a_model_without_it_and_bad_reference():
    ad = simulate(n_donors=40)
    model = make_model(ad, phase=False)
    with pytest.raises(ValueError):
        model.add_covariates(ad, ["sex"], ["sex"], phase=True)
    with pytest.raises(ValueError):
        make_model(ad, reference="BG_0000")  # not a rhythmic gene


def _aligned_error(pred, truth):
    d = np.angle(np.exp(1j * (pred - truth)))
    d = np.angle(np.exp(1j * (d - np.angle(np.mean(np.exp(1j * d))))))
    return np.median(np.abs(d)) * H


def test_grid_phase_uses_the_phase_effects():
    genes = ["Dbp", "Per3", "Cry1"]
    ad = simulate(phase_genes=genes, effect=1.2, n_donors=400)
    model = make_model(ad)
    female = model.covariates[:, 0].numpy() == 1
    truth = ad.obs["true_phase"].to_numpy()
    err0 = _aligned_error(model.grid_phase(dispersion=0.05)["theta"][female], truth[female])
    with torch.no_grad():  # the true delays, in the model's (centred) parametrisation
        for g in genes:
            model.covariate_phase_decoder.weight[model.context_genes.index(g), 0] = 1.2
    err1 = _aligned_error(model.grid_phase(dispersion=0.05)["theta"][female], truth[female])
    assert err1 < 0.8 * err0, (err0, err1)


def test_strong_phase_effect_is_recovered_with_frozen_clock_and_grid_phase():
    # The decoder sees the grid phase computed once before training and kept fixed (the
    # recommended setting, exp. 11b/12). Re-estimating it every 10 epochs let the effects
    # feed back on the phases: Cry1 then varied with the number of threads (0.33-0.51 for
    # 0.6). With the encoder's phase, the delays come out ~20-45 % too small (0.33-0.53,
    # stable from 200 to 400 epochs): the encoder input is not corrected for the delays.
    genes = ["Dbp", "Per3", "Cry1"]
    ad = simulate(phase_genes=genes, effect=0.6, n_donors=600)  # 0.6 rad = 2.3 h later in females
    np.random.seed(0)
    model = make_model(ad)
    gp = model.grid_phase(dispersion=0.05)
    ad.obs["grid_phase"], ad.obs["grid_kappa"] = gp["theta"], gp["kappa"]
    model.add_phase_prior(ad, "grid_phase", "grid_kappa")
    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"], unfreeze_epoch_layer=[], adam_betas=(0.9, 0.99),
              decoder_phase_from_prior=True, phase_prior_anneal_epochs=10**6)
    Trainer(model, Loss.compute_loss, **kw).train_model(n_epochs=200, lr=1e-2, subsample=None, device="cpu",
                                                        silent=True)
    eff = model.covariate_effects()
    delta = eff[eff["kind"] == "phase"].set_index("gene")["value"]
    clock = list(FOURIER.index)
    others = delta.loc[[g for g in clock if g not in genes]]
    # only differences between genes are identifiable
    diff = delta.loc[genes] - others.median()
    # truth 0.6. Dbp and Per3 come out ~0.5, Cry1 only ~0.3: with such a large effect (2.3 h on
    # half of the clock), the grid phases computed before training (without the delays) are
    # biased for the females, and the bias falls mostly on Cry1. Smaller, sparser effects
    # (scripts/step6_phase_effects.py) are recovered with slopes 0.8-1.1.
    assert diff.between(0.25, 0.85).all(), delta.loc[clock]
    assert diff.min() > (others - others.median()).abs().max() + 0.1, delta.loc[clock]
    assert (others - others.median()).abs().max() < 0.2, others
