"""Step 10: per-gene negative binomial rhythm GLM (cophaser.gene_rhythms)."""
import numpy as np
import pandas as pd
import pytest
import torch

from cophaser.gene_rhythms import (
    H,
    add_qvalues,
    bh,
    build_design,
    fit_gene_rhythms as gene_rhythms,
    fit_nb_glm,
    rhythm_change,
)


def simulate(n=800, G=60, alpha=0.1, rhythmic=True, effects=True, seed=0):
    """Counts from the GLM itself (log-linear interactions), with the true coefficients."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, n)
    cov = pd.DataFrame({"age": rng.normal(0, 1, n), "sex": (rng.random(n) < 0.4).astype(float)})
    nui = pd.DataFrame({"rin": rng.normal(0, 1, n)})
    X, names, blocks = build_design(theta, cov, nui)
    p = X.shape[1]
    beta = np.zeros((G, p))
    beta[:, 0] = rng.uniform(-11, -8, G)
    for j in list(blocks["mesor"].values()) + blocks["nuisance"]:
        beta[:, j] = rng.normal(0, 0.2, G) if effects else 0.0
    if rhythmic:
        amp, ph = rng.uniform(0.2, 0.6, G), rng.uniform(-np.pi, np.pi, G)
        beta[:, blocks["rhythm"][0]], beta[:, blocks["rhythm"][1]] = amp * np.cos(ph), amp * np.sin(ph)
        if effects:
            for jc, js in blocks["interaction"].values():
                beta[:, jc], beta[:, js] = rng.normal(0, 0.1, G), rng.normal(0, 0.1, G)
    lib = 2e7 * np.exp(rng.normal(0, 0.2, n))
    mu = np.exp(X @ beta.T + np.log(lib)[:, None])
    r = 1 / alpha
    Y = rng.negative_binomial(r, r / (r + mu))
    return Y, lib, theta, cov, nui, beta, names


def test_coefficients_and_standard_errors_are_calibrated():
    Y, lib, theta, cov, nui, beta, names = simulate()
    res = gene_rhythms(Y, lib, theta, cov, nui)
    assert res["converged"].all()
    assert 0.8 < np.median(res["alpha"] / 0.1) < 1.2
    z = []
    for k in ["age", "sex"]:
        j = names.index(f"mesor:{k}")
        z.append((res[f"mesor_{k}"] - beta[:, j]) / res[f"mesor_se_{k}"])
    z = np.concatenate(z)
    assert 0.88 < np.mean(np.abs(z) < 1.96) <= 1.0
    assert np.median(np.abs(res["b"] - beta[:, names.index("cos")])) < 0.03
    assert np.median(np.abs(res["c"] - beta[:, names.index("sin")])) < 0.03
    assert np.median(np.abs(res["db_sex"] - beta[:, names.index("cos:sex")])) < 0.05
    assert (res["p_rhythm"] < 1e-6).mean() > 0.95


def test_null_pvalues_are_uniform():
    Y, lib, theta, cov, nui, *_ = simulate(n=300, G=400, rhythmic=False, effects=False, seed=1)
    res = gene_rhythms(Y, lib, theta, cov, nui)
    for col in ["p_rhythm", "p_diffrhythm_sex", "p_diffrhythm_age", "p_mesor_sex",
                "p_rhythm_rob", "p_diffrhythm_rob_sex", "p_diffrhythm_rob_age", "p_mesor_rob_sex"]:
        frac = (res[col] < 0.05).mean()
        assert 0.02 <= frac <= 0.09, (col, frac)
    q = add_qvalues(res)
    assert (q["q_rhythm"] < 0.05).sum() <= 3


def test_sandwich_matches_model_based_errors_when_the_model_is_right():
    Y, lib, theta, cov, nui, *_ = simulate(n=800, G=60, seed=6)
    res = gene_rhythms(Y, lib, theta, cov, nui)
    for k in ["age", "sex"]:
        ratio = res[f"mesor_se_rob_{k}"] / res[f"mesor_se_{k}"]
        assert 0.9 < np.median(ratio) < 1.15, (k, np.median(ratio))
        ratio = res[f"log_amp_ratio_se_rob_{k}"] / res[f"log_amp_ratio_se_{k}"]
        assert 0.9 < np.median(ratio) < 1.2, (k, np.median(ratio))
    assert (res["p_rhythm_rob"] < 1e-6).mean() > 0.95
    assert np.isfinite(res["max_cook"]).all()


def test_robust_tests_resist_outlier_samples():
    """Null genes (no rhythm, no effect) with 3 extreme samples each (as cross-tissue
    contamination): the likelihood ratio test over-rejects, the sandwich Wald test does not."""
    Y, lib, theta, cov, nui, *_ = simulate(n=400, G=400, rhythmic=False, effects=False, seed=5)
    rng = np.random.default_rng(5)
    for g in range(Y.shape[1]):
        i = rng.choice(Y.shape[0], 3, replace=False)
        Y[i, g] = Y[i, g] * 30 + 50
    res = gene_rhythms(Y, lib, theta, cov, nui)
    lrt, rob = (res["p_rhythm"] < 0.05).mean(), (res["p_rhythm_rob"] < 0.05).mean()
    assert lrt > 0.1, lrt
    assert 0.005 <= rob <= 0.08, rob  # 2026-11-26: LRT 0.80, robust 0.012 (HC3 slightly conservative)
    assert res["max_cook"].median() > 1


def test_binary_amplitude_and_phase_change():
    """Group sex = 1 has 1.5x the amplitude and peaks 2 h later (exact multiplicative model)."""
    rng = np.random.default_rng(2)
    n, G = 1500, 30
    theta = rng.uniform(-np.pi, np.pi, n)
    sex = (rng.random(n) < 0.5).astype(float)
    amp, ph = rng.uniform(0.3, 0.6, G), rng.uniform(-np.pi, np.pi, G)
    gain, delay = np.log(1.5), 2 / H
    amp_s = amp[None, :] * np.exp(gain * sex[:, None])
    log_mu = -9 + amp_s * np.cos(theta[:, None] - ph[None, :] - delay * sex[:, None])
    lib = np.full(n, 2e7)
    r = 1 / 0.05
    mu = np.exp(log_mu) * lib[:, None]
    Y = rng.negative_binomial(r, r / (r + mu))
    res = gene_rhythms(Y, lib, theta, pd.DataFrame({"sex": sex}))
    assert abs(res["log_amp_ratio_sex"].median() - gain) < 0.05
    assert abs(res["phase_shift_h_sex"].median() - 2.0) < 0.2
    zr = (res["log_amp_ratio_sex"] - gain) / res["log_amp_ratio_se_sex"]
    assert np.mean(np.abs(zr) < 1.96) > 0.8
    assert np.median(np.abs(np.angle(np.exp(1j * (res["phase"] - ph))))) < 0.1


@pytest.mark.parametrize("binary", [True, False])
def test_delta_method_gradient(binary):
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.3, (5, 4)) + np.array([0.4, -0.2, 0, 0])
    eps = 1e-6
    for i in range(4):
        cov4 = np.zeros((5, 4, 4))
        cov4[:, i, i] = 1.0
        _, se_lr, _, se_ps = rhythm_change(*x.T, cov4, binary)
        xp, xm = x.copy(), x.copy()
        xp[:, i] += eps
        xm[:, i] -= eps
        zero = np.zeros((5, 4, 4))
        lp, _, pp, _ = rhythm_change(*xp.T, zero, binary)
        lm, _, pm, _ = rhythm_change(*xm.T, zero, binary)
        assert np.allclose(se_lr, np.abs((lp - lm) / (2 * eps)), rtol=1e-4, atol=1e-6)
        dps = np.angle(np.exp(1j * (pp - pm))) / (2 * eps)
        assert np.allclose(se_ps, np.abs(dps), rtol=1e-4, atol=1e-6)


def test_matches_statsmodels_with_fixed_dispersion():
    sm = pytest.importorskip("statsmodels.api")
    Y, lib, theta, cov, nui, *_ = simulate(n=200, G=3, seed=4)
    X, _, _ = build_design(theta, cov, nui)
    beta, _, _, conv = fit_nb_glm(torch.tensor(Y, dtype=torch.float64), torch.tensor(X), torch.tensor(np.log(lib)),
                                  torch.full((3,), 0.1, dtype=torch.float64))
    assert conv.all()
    for g in range(3):
        ref = sm.GLM(Y[:, g], X, family=sm.families.NegativeBinomial(alpha=0.1), offset=np.log(lib)).fit(tol=1e-12)
        assert np.allclose(beta[g].numpy(), ref.params, atol=1e-5)


def test_gtex_encoding_fast_deaths_reference():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gtex_gene_rhythms import encode, parse_args

    obs = pd.DataFrame({"age": [24.5, 54.5, 74.5, 44.5], "sex": ["male", "female", "female", "male"],
                        "hardy": ["0", "1", "2", "4"], "ischemic_time": [100.0, 200.0, np.nan, 400.0],
                        "rin": [6.0, 7.0, 8.0, 9.0]})
    cov, nui = encode(obs, parse_args([]))
    assert list(cov.columns) == ["age", "sex", "hardy[0]", "hardy[4]"]
    assert np.allclose(cov["age"], [-2.55, 0.45, 2.45, -0.55])
    assert np.allclose(cov["sex"], [0, 1, 1, 0])
    assert np.allclose(cov[["hardy[0]", "hardy[4]"]].to_numpy(), [[1, 0], [0, 0], [0, 0], [0, 1]])
    assert np.isfinite(nui.to_numpy()).all() and np.allclose(nui.mean(), 0)


def test_gtex_flags_contamination_and_one_group_genes():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gtex_gene_rhythms import flag_genes, group_ratio, parse_args, top_share

    rng = np.random.default_rng(7)
    n = 200
    sex = (np.arange(n) % 2).astype(float)
    cpm = rng.gamma(20, 1, (n, 3))
    cpm[:2, 1] = 5000  # 1 % of the samples carry most of gene 1 (contamination-like): 1e4 / (1e4 + 198 x 20)
    cpm[sex == 1, 2] = 0.01  # gene 2 barely expressed in women (Y-linked-like)
    share = top_share(cpm)
    assert share[0] < 0.05 and 0.6 < share[1] < 0.8
    ratio = group_ratio(cpm, sex)
    assert ratio[0] > 0.8 and ratio[2] < 0.01
    res = pd.DataFrame({"gene": ["a", "b", "c"], "p_diffrhythm_sex": 0.01, "p_diffrhythm_rob_sex": 0.01,
                        "p_diffrhythm_age": 0.01, "p_diffrhythm_any": 0.01, "p_diffrhythm_any_rob": 0.01})
    cov = pd.DataFrame({"age": rng.normal(0, 1, n), "sex": sex})
    out = flag_genes(res, cpm, cov, parse_args([]))
    assert out["spiky"].tolist() == [False, True, False]
    assert out["p_diffrhythm_rob_sex"].isna().tolist() == [False, False, True]
    assert out["p_diffrhythm_any_rob"].isna().tolist() == [False, False, True]
    assert out["p_diffrhythm_age"].notna().all() and "group_ratio_age" not in out


def test_bh():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    q = bh(p)
    assert np.isnan(q[-1])
    assert np.allclose(q[:4], [0.02, 0.04, 0.04, 0.02])
