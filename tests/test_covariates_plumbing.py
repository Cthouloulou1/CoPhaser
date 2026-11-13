"""Step 3: covariates are encoded, carried through dataset / trainer / model / save-load,
and (not used by the generative model yet) leave training strictly unchanged."""
import numpy as np
import pandas as pd
import pytest
import torch

from cophaser.covariates import build_covariate_spec, covariate_column_names, encode_covariates
from cophaser.cycle_configs import CYCLE_TRAINER_DEFAULTS
from cophaser.loss import Loss
from cophaser.model.CoPhaser import CoPhaser
from cophaser.simulation_covariates import CovariateEffects, sample_donors, simulate_with_covariates
from cophaser.single_cell_dataset import SingleCellDataset
from cophaser.trainer import Trainer

KEYS = ["age", "sex", "condition"]


# ---------------------------------------------------------------- encoding (no torch)


@pytest.fixture
def obs():
    return pd.DataFrame(
        {
            "age": [25.0, 35.0, np.nan, 75.0],
            "sex": ["male", "female", "female", "male"],
            "tissue": pd.Categorical(["liver", "lung", "heart", np.nan]),
        }
    )


def test_continuous_is_standardised_with_missing_indicator(obs):
    spec = build_covariate_spec(obs, ["age"])
    X = encode_covariates(obs, spec)
    assert covariate_column_names(spec) == ["age", "age__missing"]
    ages = np.array([25.0, 35.0, 75.0])
    np.testing.assert_allclose(X[[0, 1, 3], 0], (ages - ages.mean()) / ages.std(), rtol=1e-6)
    assert X[2, 0] == 0.0  # missing -> mean
    np.testing.assert_array_equal(X[:, 1], [0, 0, 1, 0])


def test_categorical_one_hot_with_reference(obs):
    spec = build_covariate_spec(obs, ["sex", "tissue"], reference={"sex": "male", "tissue": "liver"})
    assert covariate_column_names(spec) == ["sex[female]", "tissue[heart]", "tissue[lung]", "tissue__missing"]
    X = encode_covariates(obs, spec)
    np.testing.assert_array_equal(X[:, 0], [0, 1, 1, 0])
    np.testing.assert_array_equal(X[:, 1:], [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])


def test_numeric_forced_categorical():
    o = pd.DataFrame({"sex": [0, 1, 1, 0]})
    spec = build_covariate_spec(o, ["sex"], categorical=["sex"])
    assert covariate_column_names(spec) == ["sex[1]"]
    np.testing.assert_array_equal(encode_covariates(o, spec)[:, 0], [0, 1, 1, 0])


def test_spec_reused_on_new_data_and_unknown_level_raises(obs):
    spec = build_covariate_spec(obs, ["age", "sex"])
    new = pd.DataFrame({"age": [55.0, 30.0], "sex": ["female", "male"]})
    X = encode_covariates(new, spec)
    # default reference = first sorted level ("female") -> column sex[male]
    assert covariate_column_names(spec) == ["age", "age__missing", "sex[male]"]
    expected_age = (55.0 - spec[0]["mean"]) / spec[0]["sd"]
    np.testing.assert_allclose(X[0], [expected_age, 0.0, 0.0], rtol=1e-6)
    assert X[1, 2] == 1.0
    with pytest.raises(ValueError, match="Unknown levels"):
        encode_covariates(pd.DataFrame({"age": [40.0], "sex": ["other"]}), spec)
    with pytest.raises(KeyError):
        encode_covariates(pd.DataFrame({"age": [40.0]}), spec)


def test_invalid_specs(obs):
    with pytest.raises(KeyError):
        build_covariate_spec(obs, ["bmi"])
    with pytest.raises(ValueError):
        build_covariate_spec(pd.DataFrame({"x": [1.0, 1.0]}), ["x"])
    with pytest.raises(ValueError):
        build_covariate_spec(obs, ["sex"], reference={"sex": "other"})


# ---------------------------------------------------------------- model plumbing (torch)


@pytest.fixture(scope="module")
def adata():
    fourier = pd.DataFrame(
        {"A_0": [-9.0, -9.5, -10.0, -9.0], "A_1": [1.5, 0.0, -0.8, 0.3], "B_1": [0.0, 1.5, 0.5, -0.9]},
        index=["Dbp", "Nr1d1", "Per3", "Arntl"],
    )
    donors = sample_donors(120, rng=0)
    eff = CovariateEffects.zeros(fourier.index, ["age_std", "sex", "condition"])
    res = simulate_with_covariates(fourier, donors, eff, n_background_genes=20, library_size=2e5, rng=1)
    ad = res.adata
    ad.obs["sex"] = np.where(ad.obs["sex"] == 1, "female", "male")
    return ad


def make_model(ad):
    clock = list(ad.var_names[ad.var["is_clock_gene"].to_numpy()])
    return CoPhaser(rhythmic_gene_names=clock, context_genes=list(ad.var_names), n_latent=3, n_harm=1)


def train(ad, covariate_keys=None, epochs=3, seed=0, freeze_covariate_effects=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(ad)
    model.load_anndata(ad, layer_to_use="counts", covariate_keys=covariate_keys,
                       categorical_covariates=["condition"] if covariate_keys else None)
    if freeze_covariate_effects:
        model.covariate_mean_decoder.weight.requires_grad_(False)
    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"])
    trainer = Trainer(model, Loss.compute_loss, **kw)
    trainer.train_model(n_epochs=epochs, lr=1e-2, subsample=None, device="cpu", silent=True)
    return model, trainer


def test_dataset_returns_covariates_last():
    ds = SingleCellDataset(torch.zeros(5, 2), torch.zeros(5, 3), torch.ones(5), covariates=torch.arange(10.0).view(5, 2))
    item = ds[2]
    assert len(item) == 7
    torch.testing.assert_close(item[-1], torch.tensor([4.0, 5.0]))
    assert SingleCellDataset(torch.zeros(5, 2), torch.zeros(5, 3), torch.ones(5))[0][-1].nelement() == 0


def test_model_encodes_covariates(adata):
    model = make_model(adata)
    model.load_anndata(adata, layer_to_use="counts", covariate_keys=KEYS, categorical_covariates=["condition"])
    assert model.covariate_names == ["age", "sex[male]", "condition[1]"]
    assert model.n_covariates == 3
    assert model.covariates.shape == (adata.n_obs, 3)
    # same rule is reused, keys cannot change
    with pytest.raises(ValueError):
        model.add_covariates(adata, ["age"])


def test_training_unchanged_by_covariates(adata):
    """With the covariate effects held at their initial value 0, carrying covariates
    through dataset / trainer / model must leave training bit-identical."""
    m0, t0 = train(adata, covariate_keys=None)
    m1, t1 = train(adata, covariate_keys=KEYS, freeze_covariate_effects=True)
    assert m0.covariates is None and m1.covariates is not None
    for key in ("elbo_loss", "total_loss"):
        np.testing.assert_allclose(t0.losses_history[key], t1.losses_history[key], rtol=0, atol=0)
    th0 = m0.get_outputs()[1]["theta"]
    th1 = m1.get_outputs()[1]["theta"]
    torch.testing.assert_close(th0, th1, rtol=0, atol=0)


def test_inference_paths_use_covariates(adata):
    model, _ = train(adata, covariate_keys=KEYS, epochs=2)
    th = model.infer_pseudotimes(adata, "counts", isCellCycle=False, plot=False)
    assert th.shape == (adata.n_obs,)
    post, grid = model.get_posterior(adata, "counts", n_points=8, device="cpu")
    assert post.shape == (adata.n_obs, 8)
    with pytest.raises(KeyError):
        model.infer_pseudotimes(adata[:, :].copy().__class__(X=adata.X, obs=adata.obs.drop(columns=["age"]),
                                                              var=adata.var, layers={"counts": adata.layers["counts"]}),
                                "counts", isCellCycle=False, plot=False)


def test_save_load_roundtrip(adata, tmp_path):
    model, _ = train(adata, covariate_keys=KEYS, epochs=2)
    out_before = model.get_outputs()[1]["theta"]
    path = tmp_path / "model.pt"
    model.save(path)

    loaded = CoPhaser.load(path)
    assert loaded.covariate_spec == model.covariate_spec
    loaded.load_anndata(adata, layer_to_use="counts")  # rule reused without passing keys
    assert loaded.covariate_names == model.covariate_names
    torch.testing.assert_close(loaded.get_outputs()[1]["theta"], out_before)


def test_old_checkpoints(adata, tmp_path):
    model, _ = train(adata, covariate_keys=None, epochs=2)  # 1 epoch + silent crashes (pre-existing trainer bug)
    path = tmp_path / "old.pt"
    model.save(path)
    ckpt = torch.load(path, weights_only=False)
    # checkpoints saved before covariates (and before cycling_status_prior) existed
    del ckpt["covariate_spec"]
    del ckpt["init_params"]["cycling_status_prior"]
    torch.save(ckpt, path)
    loaded = CoPhaser.load(path)
    assert loaded.covariate_spec is None and loaded.cycling_status_prior == 1

    del ckpt["init_params"]
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="init_params"):
        CoPhaser.load(path)
