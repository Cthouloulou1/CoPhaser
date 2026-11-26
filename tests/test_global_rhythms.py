"""Step 10, global test over a gene set (scripts/gtex_global_rhythms.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cophaser.gene_rhythms import H, fit_gene_rhythms, project_rhythm_change

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gtex_global_rhythms as gg  # noqa: E402


def settings():
    from gtex_gene_rhythms import parse_args

    g = parse_args([])
    g.covariates, g.nuisance = ["age", "sex"], []
    return g


def simulate_tissue(n=500, G=60, amp_factor=0.8, delay_h=1.0, seed=0):
    """One sample per donor; women: amplitude x amp_factor and peak delay_h later; no age effect."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, n)
    female = rng.random(n) < 0.4
    obs = pd.DataFrame({"donor": [f"D{i:04d}" for i in range(n)],
                        "age": rng.choice([24.5, 34.5, 44.5, 54.5, 64.5, 74.5], n),
                        "sex": np.where(female, "female", "male")})
    amp, ph = rng.uniform(0.3, 0.6, G), rng.uniform(-np.pi, np.pi, G)
    amp_s = amp[None, :] * np.where(female, amp_factor, 1.0)[:, None]
    shift = np.where(female, delay_h / H, 0.0)[:, None]
    log_mu = rng.uniform(-11, -8, G)[None, :] + amp_s * np.cos(theta[:, None] - ph[None, :] - shift)
    lib = 2e7 * np.exp(rng.normal(0, 0.2, n))
    mu = np.exp(log_mu) * lib[:, None]
    r = 1 / 0.1
    return rng.negative_binomial(r, r / (r + mu)), lib, theta, obs


def test_projection_splits_amplitude_and_phase():
    rng = np.random.default_rng(0)
    amp, ph = rng.uniform(0.2, 0.8, 50), rng.uniform(-np.pi, np.pi, 50)
    b, c = amp * np.cos(ph), amp * np.sin(ph)
    rel, ps = project_rhythm_change(amp, ph, 0.1 * b, 0.1 * c)  # amplitude x 1.1
    assert np.allclose(rel, 0.1) and np.allclose(ps, 0)
    d = 1 / H  # peak 1 h later, same amplitude
    rel, ps = project_rhythm_change(amp, ph, amp * np.cos(ph + d) - b, amp * np.sin(ph + d) - c)
    assert np.allclose(ps, np.sin(d)) and np.allclose(rel, np.cos(d) - 1)


def test_global_test_recovers_a_damping_in_one_group():
    from gtex_gene_rhythms import encode

    Y, lib, theta, obs = simulate_tissue()
    g = settings()
    cov, nui = encode(obs, g)
    res = fit_gene_rhythms(Y, lib, theta, cov, nui, robust=False)
    res["strong"], res["is_clock"] = True, False
    res, sets, masks, direction = gg.gene_sets(res, 0.15)
    donors = obs.set_index("donor")[["age", "sex"]]
    rng = np.random.default_rng(1)
    U = len(donors)
    boot = rng.multinomial(U, np.full(U, 1 / U), size=30)
    perms = np.stack([rng.permutation(U) for _ in range(30)])
    rep, _, _ = gg.replicates(Y, lib, theta, obs, g, res["alpha"].to_numpy(), direction, sets, masks, donors,
                              np.arange(U), boot, perms, ["age", "sex"])
    rep.insert(0, "tissue", "t1")
    assert (rep["frac_converged"] > 0.95).all()
    eff = gg.effects_table(gg.combine(rep)).set_index(["tissue", "covariate", "gene_set", "kind"])
    amp, phase = eff.loc[("t1", "sex", "strong", "amp")], eff.loc[("t1", "sex", "strong", "phase")]
    # projection on the average rhythm: ~ log(0.8) and ~ +1 h (exactly -0.227 and +0.95 h here)
    assert abs(amp["estimate"] - np.log(0.8)) < 0.06, amp["estimate"]
    assert abs(phase["estimate"] - 1.0) < 0.25, phase["estimate"]
    assert amp["ci_lo"] < np.log(0.8) < amp["ci_hi"]
    assert amp["p_perm"] <= 1 / 31 + 1e-9 and phase["p_perm"] <= 1 / 31 + 1e-9
    age = eff.loc[("t1", "age", "strong", "amp")]
    assert abs(age["estimate"]) < 3 * age["se_boot"]
    assert 0.5 < age["null_sd"] / age["se_boot"] < 2
    assert np.isclose(eff.loc[("all", "sex", "strong", "amp"), "estimate"], amp["estimate"])


def test_run_settings_and_combine(tmp_path):
    saved = {"phase_table": "/x/donor_phases.csv", "min_kappa": "0.0", "permute_phases": "None",
             "covariates": "['age', 'sex', 'hardy']", "merge_hardy": "['1', '2']", "hardy_reference": "1-2",
             "nuisance": "['ischemic_time', 'rin']", "age_center": "50.0", "age_scale": "10.0", "jobs": "4"}
    (tmp_path / "args.json").write_text(json.dumps(saved))
    g = gg.run_settings(tmp_path)
    assert g.phase_table == Path("/x/donor_phases.csv") and g.permute_phases is None
    assert g.covariates == ["age", "sex", "hardy"] and g.merge_hardy == ["1", "2"] and g.age_scale == 10.0
    saved["permute_phases"] = "0"
    (tmp_path / "args.json").write_text(json.dumps(saved))
    assert gg.run_settings(tmp_path).permute_phases == 0

    rep = pd.DataFrame({"tissue": ["a", "b", "a"], "replicate": ["observed", "observed", "bootstrap"], "rep": 0,
                        "covariate": "sex", "gene_set": "strong", "n_genes": [10, 20, 10], "amp": [0.1, 0.3, 0.2],
                        "amp_frac_neg": 0.5, "phase_h": [1.0, -1.0, 0.0], "phase_frac_neg": 0.5})
    allt = gg.combine(rep).query("tissue == 'all'")
    assert len(allt) == 1  # the bootstrap replicate is missing in tissue b
    assert np.isclose(allt["amp"].iloc[0], 0.2) and np.isclose(allt["phase_h"].iloc[0], 0.0)
    assert allt["n_genes"].iloc[0] == 30


def test_phase_quality_by_group():
    donors = pd.DataFrame({"age": [24.5, 54.5, 74.5, 74.5], "sex": ["male", "female", "female", "male"],
                           "hardy": ["0", "1", "2", "4"], "chiral_ct": [0.0, 6.0, 12.0, np.nan]}, index=list("abcd"))
    phases = pd.DataFrame({"theta": [1 / H, 6 / H, 12 / H, 0.0], "kappa": [50.0, 10.0, 50.0, 5.0],
                           "resultant": 0.9, "n_tissues": [1, 2, 3, 4]}, index=list("abcd"))
    pq = gg.phase_quality(donors, phases, settings()).set_index(["variable", "level"])
    assert pq.loc[("hardy", "1-2"), "n_donors"] == 2
    assert np.isclose(pq.loc[("sex", "male"), "err_chiral_median_h"], 1.0)  # donor d has no CHIRAL phase
    assert pq.loc[("age", "70-79"), "n_donors"] == 2
    assert np.isclose(pq.loc[("sex", "female"), "frac_kappa_max"], 0.5)
