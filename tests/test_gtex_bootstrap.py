"""Step 7: donor bootstrap of the GTEx training script (scripts/gtex_train.py)."""
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gtex_train import bootstrap_donors, parse_args  # noqa: E402


def toy(n_donors=30, per_donor=1):
    donors = np.repeat([f"D{i:02d}" for i in range(n_donors)], per_donor)
    obs = pd.DataFrame({"donor": donors, "age": np.arange(len(donors), dtype=float)},
                       index=[f"S{i:03d}" for i in range(len(donors))])
    x = np.arange(len(donors) * 3, dtype=np.float32).reshape(-1, 3)
    return anndata.AnnData(X=x, obs=obs)


def test_bootstrap_is_reproducible_and_draws_only_known_donors():
    ad = toy()
    a, b, c = bootstrap_donors(ad, 1), bootstrap_donors(ad, 1), bootstrap_donors(ad, 2)
    assert a.n_obs == ad.n_obs
    assert list(a.obs_names) == list(b.obs_names)
    assert list(a.obs_names) != list(c.obs_names)
    assert a.obs_names.is_unique
    assert set(a.obs["donor"]) <= set(ad.obs["donor"])
    assert a.obs["donor"].nunique() < ad.n_obs  # with replacement: some donors are repeated
    # each row is a copy of its source sample (expression and covariates kept together)
    src = [n.split("#")[0] for n in a.obs_names]
    np.testing.assert_array_equal(a.X, ad[src].X)
    np.testing.assert_array_equal(a.obs["age"].to_numpy(), ad.obs.loc[src, "age"].to_numpy())


def test_bootstrap_keeps_all_samples_of_a_drawn_donor():
    ad = toy(n_donors=10, per_donor=3)
    a = bootstrap_donors(ad, 0)
    assert a.n_obs == ad.n_obs  # 10 draws x 3 samples
    sizes = a.obs.groupby("bootstrap_donor").size()
    assert (sizes == 3).all() and len(sizes) == 10
    assert (a.obs.groupby("bootstrap_donor")["donor"].nunique() == 1).all()


def test_no_bootstrap_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gtex_train.py", "--tissues", "thyroid"])
    assert parse_args().bootstrap_seed is None
