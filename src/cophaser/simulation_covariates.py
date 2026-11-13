"""
Simulate rhythmic expression with KNOWN covariate effects (age, sex, condition, ...)
on the mesor, the amplitude and the timing of each gene.

Design (GTEx-like by default):
- a donor has covariates (age, sex, condition) and ONE phase theta_d (time of death),
  shared by all its observations;
- an observation is (donor, tissue) for bulk, or a cell of a donor for single-cell
  (``n_obs_per_donor_tissue`` > 1, lower ``library_size``, optional ``phase_jitter_sd``);
- log-mean of gene g in observation s (donor d, tissue t), with covariate vector c_d:

    log mu = A0_g + tissue_mean_{g,t} + beta_g . c_d
             + sum_h amp_{g,h} * tissue_amp_t * exp(gamma_g . c_d) * lambda_s
                     * cos(h * (theta_s - tissue_shift_t - delta_g . c_d) - phase_{g,h})

  i.e. mean shift ``beta`` (log scale), log-amplitude factor ``gamma`` and peak delay
  ``delta`` (radians of the cycle; harmonic h rotates by h*delta, so the whole
  waveform is delayed). Counts ~ NB(mu * library_size_s, dispersion).

Continuous covariates are used as given (standardise them yourself, see ``standardize_age``);
the effects tables give the change per unit of each covariate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

EFFECT_KINDS = ("mesor", "log_amp", "phase")

AGE_BINS = [20, 30, 40, 50, 60, 70, 80]


def standardize_age(age, center=50.0, scale=15.0):
    return (np.asarray(age, float) - center) / scale


def age_to_gtex_bin(age):
    """Public GTEx reports age in 10-year bins ('20-29', ..., '70-79')."""
    age = np.asarray(age, float)
    idx = np.clip(np.digitize(age, AGE_BINS) - 1, 0, len(AGE_BINS) - 2)
    return np.array([f"{AGE_BINS[i]}-{AGE_BINS[i] + 9}" for i in idx])


@dataclass
class CovariateEffects:
    """Per-gene effects, one DataFrame per kind (index = genes, columns = covariates).

    mesor:   additive change of log-mean per unit covariate
    log_amp: additive change of log-amplitude per unit covariate (amp * exp(.))
    phase:   peak delay in radians per unit covariate (positive = later peak)
    """

    mesor: pd.DataFrame
    log_amp: pd.DataFrame
    phase: pd.DataFrame

    @classmethod
    def zeros(cls, genes, covariates):
        z = lambda: pd.DataFrame(0.0, index=pd.Index(genes), columns=list(covariates))  # noqa: E731
        return cls(z(), z(), z())

    def long(self) -> pd.DataFrame:
        """Tidy table: gene, covariate, kind, value."""
        parts = []
        for kind in EFFECT_KINDS:
            df = getattr(self, kind).copy()
            df.index.name = "gene"
            parts.append(df.reset_index().melt(id_vars="gene", var_name="covariate").assign(kind=kind))
        return pd.concat(parts, ignore_index=True)[["gene", "covariate", "kind", "value"]]


def random_effects(
    genes,
    covariates,
    frac_affected=0.3,
    mesor_sd=0.3,
    log_amp_sd=0.3,
    phase_sd_hours=1.5,
    period=24.0,
    rng=None,
) -> CovariateEffects:
    """Sparse random effects: each (gene, covariate, kind) is non-zero with prob. frac_affected."""
    rng = np.random.default_rng(rng)
    eff = CovariateEffects.zeros(genes, covariates)
    shape = (len(genes), len(covariates))
    for kind, sd in (("mesor", mesor_sd), ("log_amp", log_amp_sd), ("phase", phase_sd_hours * 2 * np.pi / period)):
        mask = rng.random(shape) < frac_affected
        getattr(eff, kind).loc[:, :] = rng.normal(0, sd, size=shape) * mask
    return eff


def sample_donors(
    n_donors,
    age_range=(20, 79),
    p_female=0.35,
    p_condition=0.3,
    rng=None,
) -> pd.DataFrame:
    """Donor table: age (years), sex (1 = female), condition (0/1), true phase (rad).

    Defaults are GTEx-like (about 1/3 female). Phases are uniform on the circle.
    """
    rng = np.random.default_rng(rng)
    age = rng.uniform(*age_range, size=n_donors)
    donors = pd.DataFrame(
        {
            "age": age,
            "age_bin": age_to_gtex_bin(age),
            "sex": (rng.random(n_donors) < p_female).astype(int),
            "condition": (rng.random(n_donors) < p_condition).astype(int),
            "true_phase": rng.uniform(-np.pi, np.pi, size=n_donors),
        },
        index=pd.Index([f"donor_{i:04d}" for i in range(n_donors)], name="donor"),
    )
    donors["age_std"] = standardize_age(donors["age"])
    return donors


@dataclass
class TissueSpec:
    name: str
    phase_shift: float = 0.0  # radians, positive = clock runs later in this tissue
    amp_scale: float = 1.0  # multiplies the clock amplitude
    mean_sd: float = 0.5  # sd of tissue-specific gene means (log scale)
    p_sampled: float = 1.0  # probability that a donor has this tissue


@dataclass
class SimulationResult:
    adata: anndata.AnnData
    donors: pd.DataFrame
    effects: CovariateEffects
    fourier: pd.DataFrame  # baseline clock (A_0, A_h, B_h), rhythmic genes only
    params: dict = field(default_factory=dict)


def _harmonics(fourier):
    return sorted({int(c.split("_")[1]) for c in fourier.columns if c.startswith("A_") and c != "A_0"})


def simulate_with_covariates(
    fourier: pd.DataFrame,
    donors: pd.DataFrame,
    effects: CovariateEffects,
    covariate_columns=("age_std", "sex", "condition"),
    tissues=(TissueSpec("tissue_A"),),
    n_obs_per_donor_tissue=1,
    n_background_genes=200,
    background_log_frac_range=(-13.0, -7.0),
    library_size=2e7,
    library_size_sd=0.3,
    dispersion=0.05,
    phase_jitter_sd=0.0,
    amp_noise_sd=0.0,
    rng=None,
) -> SimulationResult:
    """Simulate counts. See module docstring for the model.

    fourier: baseline clock (index = rhythmic genes, columns A_0, A_1, B_1, ...).
    effects: indexed by any subset of genes (rhythmic or background); others get 0.
        Amplitude/phase effects on background genes have no effect (they are flat).
    """
    rng = np.random.default_rng(rng)
    covariate_columns = list(covariate_columns)
    tissues = list(tissues)
    harm = _harmonics(fourier)
    clock_genes = list(fourier.index)

    bg_genes = [f"BG_{i:04d}" for i in range(n_background_genes)]
    genes = clock_genes + bg_genes
    n_genes = len(genes)
    a0 = np.r_[fourier["A_0"].to_numpy(float), rng.uniform(*background_log_frac_range, size=n_background_genes)]

    # baseline amp/phase per harmonic (0 for background genes)
    amp = np.zeros((n_genes, len(harm)))
    ph = np.zeros((n_genes, len(harm)))
    for j, h in enumerate(harm):
        A, B = fourier[f"A_{h}"].to_numpy(float), fourier[f"B_{h}"].to_numpy(float)
        amp[: len(clock_genes), j] = np.hypot(A, B)
        ph[: len(clock_genes), j] = np.arctan2(B, A)

    def effect_matrix(df):
        return df.reindex(index=genes, columns=covariate_columns).fillna(0.0).to_numpy(float)

    beta, gamma, delta = (effect_matrix(getattr(effects, k)) for k in EFFECT_KINDS)
    tissue_means = {t.name: rng.normal(0, t.mean_sd, size=n_genes) for t in tissues}

    rows, blocks = [], []
    C_all = donors[covariate_columns].to_numpy(float)
    for t in tissues:
        has = rng.random(len(donors)) < t.p_sampled
        d_idx = np.repeat(np.flatnonzero(has), n_obs_per_donor_tissue)
        if d_idx.size == 0:
            continue
        n = d_idx.size
        C = C_all[d_idx]  # (n, n_cov)
        theta = donors["true_phase"].to_numpy(float)[d_idx] + rng.normal(0, phase_jitter_sd, size=n)
        lam = np.exp(rng.normal(0, amp_noise_sd, size=n))
        lib = library_size * np.exp(rng.normal(0, library_size_sd, size=n))

        log_mu = a0[None, :] + tissue_means[t.name][None, :] + C @ beta.T  # (n, G)
        amp_factor = np.exp(C @ gamma.T) * t.amp_scale * lam[:, None]  # (n, G)
        shift = (theta[:, None] - t.phase_shift - C @ delta.T)  # (n, G) effective phase
        for j, h in enumerate(harm):
            log_mu += amp[None, :, j] * amp_factor * np.cos(h * shift - ph[None, :, j])

        mean = np.exp(log_mu) * lib[:, None]
        r = 1.0 / dispersion
        counts = rng.negative_binomial(r, r / (r + mean))
        blocks.append(csr_matrix(counts))

        obs = donors.iloc[d_idx].reset_index()
        obs["tissue"] = t.name
        obs["true_phase"] = np.angle(np.exp(1j * theta))
        obs["donor_phase"] = donors["true_phase"].to_numpy(float)[d_idx]
        obs["library_size_sim"] = lib
        rows.append(obs)

    from scipy.sparse import vstack

    X = vstack(blocks).tocsr()
    obs = pd.concat(rows, ignore_index=True)
    obs.index = [f"obs_{i:06d}" for i in range(len(obs))]
    var = pd.DataFrame({"is_clock_gene": [g in set(clock_genes) for g in genes]}, index=pd.Index(genes))
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    for col in ("tissue", "age_bin", "donor"):
        adata.obs[col] = adata.obs[col].astype("category")

    params = dict(
        covariate_columns=covariate_columns,
        tissues=[t.__dict__ for t in tissues],
        n_obs_per_donor_tissue=n_obs_per_donor_tissue,
        n_background_genes=n_background_genes,
        library_size=library_size,
        library_size_sd=library_size_sd,
        dispersion=dispersion,
        phase_jitter_sd=phase_jitter_sd,
        amp_noise_sd=amp_noise_sd,
        harmonics=harm,
    )
    return SimulationResult(adata=adata, donors=donors, effects=effects, fourier=fourier, params=params)


def cosinor_by_group(adata, genes, groups, phase_column="true_phase", layer="counts"):
    """First-harmonic fit of log(count fraction) per group (e.g. sex) on a given phase.

    Uses log((x + 0.5) / library) so that it is unbiased enough at bulk depth.
    Returns a long table: group, gene, mesor, amp, phase, r2.
    """
    from cophaser.report_plots import cosinor_fit

    X = adata[:, genes].layers[layer]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    lib = np.asarray(adata.layers[layer].sum(1)).ravel()
    y = np.log((X + 0.5) / lib[:, None])
    theta = adata.obs[phase_column].to_numpy(float)
    groups = np.asarray(groups)
    out = []
    for g in pd.unique(groups):
        m = groups == g
        fit = cosinor_fit(y[m], theta[m])
        fit.insert(0, "gene", genes)
        fit.insert(0, "group", g)
        out.append(fit)
    return pd.concat(out, ignore_index=True)
