"""
Step 10: rhythm of every gene and its modulation by covariates, given known phases.

The phase of each sample (theta, radians) comes from elsewhere (e.g. donor phases shared
across tissues, inferred with a frozen clock). For every gene, a negative binomial GLM with
log link is fitted on the counts:

    log mu = log(library) + a + b cos(theta) + c sin(theta)
             + sum_k beta_k x_k + sum_m nu_m z_m
             + sum_k (b'_k x_k cos(theta) + c'_k x_k sin(theta))

x_k: covariates of interest (age, sex, condition, ...), z_m: nuisance covariates (mesor only,
e.g. ischemic time, RIN). Var = mu + alpha mu^2, one dispersion alpha per gene.

Tests (likelihood ratio, dispersion fixed at its estimate under the full model, as DESeq2):
  rhythm        b = c = 0                    (model without interactions vs without rhythm)
  diff rhythm k b'_k = c'_k = 0              (full model vs full model without x_k's interactions)
  diff rhythm   all b', c' = 0
and Wald tests of the mesor effects beta_k in the full model.

Rhythm parameters (same convention as the CoPhaser decoder): amp = hypot(b, c) (log units),
peak phase = atan2(c, b), i.e. the log-mean peaks at theta = phase. For covariate k:
  binary (values 0/1): rhythm of the group x_k = 1 (other covariates at 0) vs the reference:
      log_amp_ratio = log |(b + b'_k, c + c'_k)| - log |(b, c)|,
      phase_shift = angle(b + b'_k + i(c + c'_k)) - angle(b + i c)   (> 0: later peak)
  continuous: derivatives at x_k = 0 per unit of x_k (linear interactions are a first-order
      model of a multiplicative amplitude change and a phase rotation):
      d log amp = (b b'_k + c c'_k) / amp^2,   d phase = (b c'_k - c b'_k) / amp^2
with delta-method standard errors. Covariates should be centred (continuous) or 0/1 coded
(categorical, reference level = 0), so that (b, c) is the rhythm of the reference group.

Everything is vectorised over genes (torch, float64): all genes share the design matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, norm

H = 24 / (2 * np.pi)


# ---------------------------------------------------------------------------------------------
# negative binomial GLM, batched over genes
# ---------------------------------------------------------------------------------------------
def nb_loglik(Y, mu, alpha):
    """Per-gene NB log-likelihood summed over samples. Y, mu (n, G); alpha (G,)."""
    r = 1.0 / alpha
    ll = (
        torch.lgamma(Y + r) - torch.lgamma(r) - torch.lgamma(Y + 1)
        + r * (torch.log(r) - torch.log(r + mu))
        + Y * (torch.log(mu) - torch.log(r + mu))
    )
    return ll.sum(0)


def _eta_mu(X, beta, offset):
    eta = X @ beta.T + offset[:, None]
    eta = eta.clamp(-50, 50)
    return eta, torch.exp(eta)


def fit_nb_glm(Y, X, offset, alpha, beta0=None, max_iter=100, tol=1e-7, ridge=1e-8):
    """IRLS (Fisher scoring) with fixed dispersion, all genes at once, with step halving when
    the log-likelihood of a gene decreases.

    Y (n, G), X (n, p), offset (n,), alpha (G,). Returns beta (G, p), the inverse Fisher
    information (G, p, p), the log-likelihood (G,) and a convergence flag (G,).
    """
    n, G = Y.shape
    p = X.shape[1]
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)
    eye = torch.eye(p, dtype=X.dtype)
    if beta0 is None:
        beta = torch.zeros(G, p, dtype=X.dtype)
        # intercept column = the one constant column (first column by construction)
        beta[:, 0] = torch.log((Y / torch.exp(offset)[:, None]).mean(0).clamp_min(1e-12))
    else:
        beta = beta0.clone()
    _, mu = _eta_mu(X, beta, offset)
    ll = nb_loglik(Y, mu, alpha)
    converged = torch.zeros(G, dtype=torch.bool)
    for _ in range(max_iter):
        eta, mu = _eta_mu(X, beta, offset)
        w = mu / (1 + alpha * mu)
        z = eta - offset[:, None] + (Y - mu) / mu
        XtWX = (w.T @ XX).reshape(G, p, p) + ridge * eye
        XtWz = (w * z).T @ X
        step = torch.linalg.solve(XtWX, XtWz) - beta
        t = torch.ones(G, dtype=X.dtype)
        new = beta + step
        for _ in range(30):  # step halving per gene
            _, mu_new = _eta_mu(X, new, offset)
            ll_new = nb_loglik(Y, mu_new, alpha)
            bad = ~(ll_new >= ll - 1e-10) | ~torch.isfinite(ll_new)
            if not bad.any():
                break
            t = torch.where(bad, t / 2, t)
            new = beta + t[:, None] * step
        delta = (new - beta).abs().max(1).values
        beta, ll = new, torch.where(torch.isfinite(ll_new), ll_new, ll)
        converged = delta < tol
        if converged.all():
            break
    _, mu = _eta_mu(X, beta, offset)
    w = mu / (1 + alpha * mu)
    cov = torch.linalg.inv((w.T @ XX).reshape(G, p, p) + ridge * eye)
    return beta, cov, nb_loglik(Y, mu, alpha), converged


def estimate_dispersion(Y, mu, log_alpha_range=(-10.0, 3.0), n_grid=27, n_refine=16):
    """Maximum likelihood dispersion per gene given fitted means: grid search on log alpha,
    then golden-section refinement. Y, mu (n, G). Returns alpha (G,)."""
    G = Y.shape[1]
    grid = torch.linspace(*log_alpha_range, n_grid, dtype=Y.dtype)
    ll = torch.stack([nb_loglik(Y, mu, torch.full((G,), float(np.exp(g)), dtype=Y.dtype)) for g in grid.tolist()])
    i = ll.argmax(0)
    step = float(grid[1] - grid[0])
    lo = grid[i] - step
    hi = grid[i] + step
    gr = (np.sqrt(5) - 1) / 2

    def f(la):
        return nb_loglik(Y, mu, torch.exp(la))

    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(n_refine):
        left = fc > fd  # maximum in [lo, d]
        hi = torch.where(left, d, hi)
        lo = torch.where(left, lo, c)
        c_new = hi - gr * (hi - lo)
        d_new = lo + gr * (hi - lo)
        # golden section: one interior point is reused, only the other one is evaluated
        f_new = f(torch.where(left, c_new, d_new))
        fc_new = torch.where(left, f_new, fd)
        fd_new = torch.where(left, fc, f_new)
        c, d, fc, fd = c_new, d_new, fc_new, fd_new
    return torch.exp((lo + hi) / 2)


# ---------------------------------------------------------------------------------------------
# design and rhythm parameters
# ---------------------------------------------------------------------------------------------
def build_design(theta, covariates: pd.DataFrame, nuisance: pd.DataFrame | None = None, interact=None):
    """Columns of the full model and the index sets of each block.

    Returns (X (n, p) float64 array, names list, blocks dict): blocks 'intercept', 'rhythm'
    ([cos, sin]), 'mesor' {k: idx}, 'nuisance' [idx], 'interaction' {k: [idx_cos, idx_sin]}.
    """
    theta = np.asarray(theta, float)
    n = len(theta)
    covariates = pd.DataFrame(covariates) if covariates is not None else pd.DataFrame(index=range(n))
    nuisance = pd.DataFrame(nuisance) if nuisance is not None else pd.DataFrame(index=range(n))
    interact = list(covariates.columns) if interact is None else list(interact)
    cols, names = [np.ones(n)], ["intercept"]
    blocks = {"intercept": [0], "mesor": {}, "nuisance": [], "interaction": {}}
    for k in covariates.columns:
        blocks["mesor"][k] = len(cols)
        cols.append(covariates[k].to_numpy(float)), names.append(f"mesor:{k}")
    for m in nuisance.columns:
        blocks["nuisance"].append(len(cols))
        cols.append(nuisance[m].to_numpy(float)), names.append(f"nuisance:{m}")
    blocks["rhythm"] = [len(cols), len(cols) + 1]
    cols += [np.cos(theta), np.sin(theta)]
    names += ["cos", "sin"]
    for k in interact:
        x = covariates[k].to_numpy(float)
        blocks["interaction"][k] = [len(cols), len(cols) + 1]
        cols += [x * np.cos(theta), x * np.sin(theta)]
        names += [f"cos:{k}", f"sin:{k}"]
    X = np.column_stack(cols)
    if not np.all(np.isfinite(X)):
        raise ValueError("non-finite values in the design (missing covariates or phases?)")
    return X, names, blocks


def is_binary(x):
    v = np.unique(np.asarray(x, float))
    return set(v.tolist()) <= {0.0, 1.0}


def rhythm_change(b, c, db, dc, cov4, binary):
    """log-amplitude ratio and phase shift (rad) of a covariate, with delta-method SEs.

    b, c, db, dc: arrays (G,); cov4: (G, 4, 4) covariance of (b, c, db, dc).
    """
    S0 = b**2 + c**2
    if binary:
        b1, c1 = b + db, c + dc
        S1 = b1**2 + c1**2
        lr = 0.5 * (np.log(S1) - np.log(S0))
        ps = np.angle(np.exp(1j * (np.arctan2(c1, b1) - np.arctan2(c, b))))
        g_lr = np.stack([b1 / S1 - b / S0, c1 / S1 - c / S0, b1 / S1, c1 / S1], 1)
        g_ps = np.stack([-c1 / S1 + c / S0, b1 / S1 - b / S0, -c1 / S1, b1 / S1], 1)
    else:
        num_lr, num_ps = b * db + c * dc, b * dc - c * db
        lr, ps = num_lr / S0, num_ps / S0
        g_lr = np.stack([db / S0 - 2 * b * num_lr / S0**2, dc / S0 - 2 * c * num_lr / S0**2, b / S0, c / S0], 1)
        g_ps = np.stack([dc / S0 - 2 * b * num_ps / S0**2, -db / S0 - 2 * c * num_ps / S0**2, -c / S0, b / S0], 1)
    se_lr = np.sqrt(np.einsum("gi,gij,gj->g", g_lr, cov4, g_lr))
    se_ps = np.sqrt(np.einsum("gi,gij,gj->g", g_ps, cov4, g_ps))
    return lr, se_lr, ps, se_ps


def bh(p):
    """Benjamini-Hochberg q-values (NaNs kept)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order] * ok.sum() / np.arange(1, ok.sum() + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ps)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# ---------------------------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------------------------
def fit_gene_rhythms(
    Y,
    library_size,
    theta,
    covariates: pd.DataFrame | None = None,
    nuisance: pd.DataFrame | None = None,
    genes=None,
    interact=None,
    chunk=2000,
    n_dispersion_rounds=2,
    max_iter=100,
):
    """Fit the rhythm GLMs gene by gene (vectorised) and return one row per gene.

    Y: counts (n_samples, n_genes), dense array. library_size: (n,). theta: phases (n,) rad.
    covariates: DataFrame (n, k) of covariates of interest, centred / 0-1 coded.
    nuisance: DataFrame (n, m), mesor-only covariates. interact: covariates whose rhythm
    changes are tested (default: all of `covariates`).

    Columns: gene, mean_count, alpha, converged, a, b, c, amp, phase, phase_h, p_rhythm,
    p_diffrhythm_any, and per covariate k: mesor_k, mesor_se_k, p_mesor_k, db_k, dc_k,
    p_diffrhythm_k, log_amp_ratio_k (+ _se), phase_shift_h_k (+ _se). No multiple-testing
    correction here (see bh).
    """
    Y = np.asarray(Y, float)
    n, G = Y.shape
    genes = list(genes) if genes is not None else [f"g{i}" for i in range(G)]
    covariates = pd.DataFrame(covariates).reset_index(drop=True) if covariates is not None else pd.DataFrame(index=range(n))
    nuisance = pd.DataFrame(nuisance).reset_index(drop=True) if nuisance is not None else None
    X, names, blocks = build_design(theta, covariates, nuisance, interact)
    interact = list(blocks["interaction"])
    p = X.shape[1]
    full = list(range(p))
    rh, inter_all = blocks["rhythm"], [i for v in blocks["interaction"].values() for i in v]
    no_inter = [i for i in full if i not in inter_all]
    no_rhythm = [i for i in no_inter if i not in rh]
    reduced = {k: [i for i in full if i not in blocks["interaction"][k]] for k in interact}

    Xt = torch.tensor(X, dtype=torch.float64)
    off = torch.tensor(np.log(np.asarray(library_size, float)), dtype=torch.float64)
    binary = {k: is_binary(covariates[k]) for k in interact}
    out = []
    for s in range(0, G, chunk):
        Yc = torch.tensor(Y[:, s:s + chunk], dtype=torch.float64)
        Gc = Yc.shape[1]
        # dispersion under the full model: start from a moment estimate, alternate with IRLS
        m = Yc.mean(0)
        v = Yc.var(0)
        alpha = ((v - m) / m.clamp_min(1e-8) ** 2).clamp(1e-4, 10.0)
        beta = None
        for _ in range(n_dispersion_rounds + 1):
            beta, cov, ll_full, conv = fit_nb_glm(Yc, Xt, off, alpha, beta0=beta, max_iter=max_iter)
            _, mu = _eta_mu(Xt, beta, off)
            alpha = estimate_dispersion(Yc, mu).clamp(1e-5, 20.0)
        beta, cov, ll_full, conv = fit_nb_glm(Yc, Xt, off, alpha, beta0=beta, max_iter=max_iter)

        def sub(cols):
            b0 = beta[:, cols]
            return fit_nb_glm(Yc, Xt[:, cols], off, alpha, beta0=b0, max_iter=max_iter)

        beta1, cov1, ll1, conv1 = sub(no_inter)
        _, _, ll0, conv0 = sub(no_rhythm)
        ll_red = {k: sub(cols)[2] for k, cols in reduced.items()}

        bn, covn = beta.numpy(), cov.numpy()
        b1n = beta1.numpy()
        i_cos, i_sin = no_inter.index(rh[0]), no_inter.index(rh[1])
        r = pd.DataFrame({"gene": genes[s:s + Gc]})
        r["mean_count"] = m.numpy()
        r["alpha"] = alpha.numpy()
        r["converged"] = (conv & conv1 & conv0).numpy()
        r["a"] = bn[:, 0]
        # base rhythm of the reference group (full model), and the average rhythm (no interactions)
        b, c = bn[:, rh[0]], bn[:, rh[1]]
        r["b"], r["c"] = b, c
        r["amp"] = np.hypot(b, c)
        r["phase"] = np.arctan2(c, b)
        r["phase_h"] = np.mod(r["phase"], 2 * np.pi) * H
        r["amp_avg"] = np.hypot(b1n[:, i_cos], b1n[:, i_sin])
        r["phase_avg_h"] = np.mod(np.arctan2(b1n[:, i_sin], b1n[:, i_cos]), 2 * np.pi) * H
        r["p_rhythm"] = chi2.sf(2 * (ll1 - ll0).clamp_min(0).numpy(), 2)
        if inter_all:
            r["p_diffrhythm_any"] = chi2.sf(2 * (ll_full - ll1).clamp_min(0).numpy(), len(inter_all))
        for k, j in blocks["mesor"].items():
            se = np.sqrt(covn[:, j, j])
            r[f"mesor_{k}"], r[f"mesor_se_{k}"] = bn[:, j], se
            r[f"p_mesor_{k}"] = 2 * norm.sf(np.abs(bn[:, j] / se))
        for k in interact:
            jc, js = blocks["interaction"][k]
            idx = [rh[0], rh[1], jc, js]
            cov4 = covn[:, idx][:, :, idx]
            db, dc = bn[:, jc], bn[:, js]
            r[f"db_{k}"], r[f"dc_{k}"] = db, dc
            r[f"p_diffrhythm_{k}"] = chi2.sf(2 * (ll_full - ll_red[k]).clamp_min(0).numpy(), 2)
            lr, se_lr, ps, se_ps = rhythm_change(b, c, db, dc, cov4, binary[k])
            r[f"log_amp_ratio_{k}"], r[f"log_amp_ratio_se_{k}"] = lr, se_lr
            r[f"phase_shift_h_{k}"], r[f"phase_shift_se_h_{k}"] = ps * H, se_ps * H
        out.append(r)
    res = pd.concat(out, ignore_index=True)
    res.attrs["design_columns"] = names
    res.attrs["binary"] = binary
    return res


def add_qvalues(res, prefix="p_", groupby=None):
    """Add q_<test> columns (BH) for every p_<test> column, optionally within groups
    (e.g. tissue)."""
    res = res.copy()
    pcols = [c for c in res.columns if c.startswith(prefix)]
    for pc in pcols:
        qc = "q_" + pc[len(prefix):]
        if groupby is None:
            res[qc] = bh(res[pc].to_numpy())
        else:
            res[qc] = res.groupby(groupby)[pc].transform(lambda s: pd.Series(bh(s.to_numpy()), index=s.index))
    return res
