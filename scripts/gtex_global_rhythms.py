"""
Step 10, global test: do age, sex and death type change the rhythms of the rhythmic genes as
a whole? Gene by gene, the rhythm-change tests of scripts/gtex_gene_rhythms.py only detect very
large effects (SE of a log-amplitude ratio 0.2-0.7, of a peak shift 0.7-2.7 h); here the
changes are summarised over a gene set, which averages out the per-gene noise.

Per gene and covariate k, the interaction coefficients (b'_k, c'_k) of the full GLM are
projected on the direction of the gene's average rhythm (amp_avg, phase_avg: model without
interactions, all donors; cophaser.gene_rhythms.project_rhythm_change):
  amp    relative amplitude change, per unit of k (~ log amplitude ratio for a 0/1 covariate;
         per 10 years for age)
  phase  peak shift in hours (> 0: later peak)
The average direction is used instead of the reference group's rhythm: it is estimated on all
donors and, when nothing changes, nearly uncorrelated with the interaction estimates (no
regression-to-the-mean bias). Statistic: median over the gene set, per tissue; tissue 'all' =
mean of the tissue medians.

Gene sets (per-gene results of --results-dir, a gtex_gene_rhythms.py run on real phases):
strong (q_rhythm_rob < 0.05, amp_avg >= 0.2, not spiky, not clock) and clock (clock genes with
amp_avg >= --clock-min-amp; they gave the phases, so their own rhythm is optimistic). For a 0/1
covariate, the genes whose rhythm-change test was voided (expressed in one group only, e.g. Y
genes for sex) are left out.

Inference, with the gene sets, the average directions and the dispersions fixed:
  bootstrap    donors drawn with replacement, one draw for all tissues (a donor comes with all
               of its tissues), full GLM refitted -> SE and percentile 95 % interval
  permutation  the covariate (age, sex or death type) shuffled across donors, one shuffle for
               all tissues, full GLM refitted -> two-sided p-value (p_perm >= 1 / (n_perm + 1));
               p_norm: normal approximation with the permutation SD, q_norm: BH within gene
               set x kind over the tissue rows (the 'all' rows separately)
Replicate fits start from the observed fit and stop after --max-iter IRLS iterations: Fisher
scoring converges only linearly for the NB (lung: 34 s per fit to tol 1e-7; 20 iterations,
3.7 s, change the interaction coefficients by at most 3e-3, far below their SE of ~0.05-0.1).
Caveat: if the phase is less accurate in a group, its rhythms look flatter without any true
change (attenuation). phase_quality.csv compares the phase concentration, the number of
tissues and the error to the CHIRAL phases (same frame; optimistic: CHIRAL saw these samples)
across the groups.

Outputs in --outdir (default <results-dir>/global): replicates.csv.gz (every statistic of every
replicate), global_effects.csv, phase_quality.csv, global_forest.png, args.json.
"""
import argparse
import ast
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

H = 24 / (2 * np.pi)
FAMILIES = {"age": ["age"], "sex": ["sex"], "hardy": ["hardy"]}  # family -> obs columns shuffled
RUN_KEYS = ["phase_table", "min_kappa", "permute_phases", "covariates", "merge_hardy", "hardy_reference",
            "nuisance", "age_center", "age_scale"]
KINDS = {"amp": ("amp", "amp_frac_neg"), "phase": ("phase_h", "phase_frac_neg")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "step10" / "gtex" / "robust_ref12",
                   help="gtex_gene_rhythms.py output (real phases)")
    p.add_argument("--tissue-dir", type=Path, default=REPO_ROOT / "data" / "gtex_v8" / "tissues")
    p.add_argument("--tissues", nargs="*", default=None, help="default: every <tissue>.csv.gz of --results-dir")
    p.add_argument("--clock-min-amp", type=float, default=0.15)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-perm", type=int, default=200, help="permutations per covariate family")
    p.add_argument("--families", nargs="*", default=["age", "sex", "hardy"])
    p.add_argument("--max-iter", type=int, default=25, help="IRLS iterations of the replicate fits")
    p.add_argument("--tol", type=float, default=1e-6, help="IRLS tolerance of the replicate fits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=4, help="tissues in parallel")
    p.add_argument("--threads", type=int, default=6, help="torch threads per job")
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args(argv)


def run_settings(results_dir):
    """Settings of a finished gtex_gene_rhythms.py run (its args.json, values saved as str)."""
    from gtex_gene_rhythms import parse_args as rhythm_args

    g = rhythm_args([])
    saved = json.loads((Path(results_dir) / "args.json").read_text())
    for k in RUN_KEYS:
        if k not in saved:
            continue
        v = saved[k]
        if v == "None":
            setattr(g, k, None)
        elif k == "phase_table":
            setattr(g, k, Path(v))
        elif k in ("covariates", "merge_hardy", "nuisance"):
            setattr(g, k, list(ast.literal_eval(v)))
        elif k in ("min_kappa", "age_center", "age_scale"):
            setattr(g, k, float(v))
        elif k == "permute_phases":
            setattr(g, k, int(v))
        else:
            setattr(g, k, v)
    return g


def select_samples(obs, phases, g):
    """Samples kept by gtex_gene_rhythms.run_tissue: donor with a phase (and kappa), known death type."""
    donors = obs["donor"].astype(str)
    keep = donors.isin(phases.index).to_numpy().copy()
    if g.min_kappa and g.min_kappa > 0 and "kappa" in phases:
        keep &= donors.isin(phases.index[phases["kappa"] >= g.min_kappa]).to_numpy()
    if "hardy" in g.covariates:
        keep &= obs["hardy"].notna().to_numpy()
    return keep


def donor_table(tissues, tissue_dir, g, phases):
    """One row per donor kept in at least one tissue: age, sex, hardy (as strings, as in the
    tissue files) and chiral_ct. Donor attributes must agree across tissues."""
    import anndata

    parts = []
    for t in tissues:
        ad = anndata.read_h5ad(Path(tissue_dir) / f"{t}.h5ad", backed="r")
        obs = ad.obs.copy()
        ad.file.close()
        keep = select_samples(obs, phases, g)
        cols = [c for c in ["donor", "age", "sex", "hardy", "chiral_ct"] if c in obs]
        parts.append(obs.loc[keep, cols])
    d = pd.concat(parts, ignore_index=True)
    d["donor"] = d["donor"].astype(str)
    for c in ["sex", "hardy"]:
        if c in d:
            v = d[c].astype(object)
            d[c] = v.where(v.isna(), v.astype(str))
    attrs = [c for c in ["age", "sex", "hardy"] if c in d]
    n_values = d.groupby("donor")[attrs].nunique().max()
    if (n_values > 1).any():
        raise ValueError(f"donor attributes differ across tissues: {n_values.to_dict()}")
    return d.groupby("donor").first().sort_index()


def gene_sets(res, clock_min_amp):
    """Rows of the strong and clock genes, their set masks, the per-covariate masks of genes
    whose rhythm-change test is defined, and their average rhythm direction (amp, phase rad)."""
    strong = res["strong"].astype(bool).to_numpy()
    clock = (res["is_clock"].astype(bool) & (res["amp_avg"] >= clock_min_amp)).to_numpy()
    keep = strong | clock
    res = res[keep].reset_index(drop=True)
    sets = {"strong": strong[keep], "clock": clock[keep]}
    pre = "p_diffrhythm_rob_"
    masks = {c[len(pre):]: res[c].notna().to_numpy() for c in res.columns if c.startswith(pre)}
    direction = (res["amp_avg"].to_numpy(float), res["phase_avg_h"].to_numpy(float) / H)
    return res, sets, masks, direction


def summarise(beta, blocks, direction, sets, masks, keep=None):
    """Median relative amplitude change and peak shift (h) per covariate and gene set."""
    from cophaser.gene_rhythms import project_rhythm_change

    out = []
    for k, (jc, js) in blocks["interaction"].items():
        if keep is not None and k not in keep:
            continue
        rel, ps = project_rhythm_change(direction[0], direction[1], beta[:, jc], beta[:, js])
        for s, m in sets.items():
            sel = m & masks.get(k, np.ones_like(m)) & np.isfinite(rel) & np.isfinite(ps)
            r = dict(covariate=k, gene_set=s, n_genes=int(sel.sum()))
            if sel.any():
                r.update(amp=float(np.median(rel[sel])), amp_frac_neg=float(np.mean(rel[sel] < 0)),
                         phase_h=float(np.median(ps[sel]) * H), phase_frac_neg=float(np.mean(ps[sel] < 0)))
            out.append(r)
    return out


def replicates(Y, lib, theta, obs, g, alpha, direction, sets, masks, donors, pos, boot, perms, families,
               max_iter=25, tol=1e-6):
    """Statistics of the observed data and of every bootstrap / permutation replicate of one tissue.

    Y (n, G) counts, lib (n,), theta (n,) phases, obs: sample table with the columns used by
    gtex_gene_rhythms.encode; alpha (G,) dispersions; direction (amp, phase rad) arrays (G,);
    sets {name: bool (G,)}; masks {covariate: bool (G,)}; donors: donor attributes (rows = the
    columns of boot / perms); pos (n,): row of each sample's donor in donors; boot (B, U) donor
    multiplicities; perms (P, U) permutations of the donors (donor u gets the attributes of
    donor perms[b, u]). max_iter, tol: IRLS settings of the replicate fits (warm start from the
    observed fit, which is run to convergence).
    """
    import torch

    from cophaser.gene_rhythms import build_design, fit_nb_glm
    from gtex_gene_rhythms import encode

    cov, nui = encode(obs, g)
    X, _, blocks = build_design(theta, cov, nui)
    off = np.log(np.asarray(lib, float))
    a = torch.tensor(np.asarray(alpha, float), dtype=torch.float64)
    Y = np.asarray(Y, float)

    def fit(Xm, idx=None, beta0=None):
        i = slice(None) if idx is None else idx
        kw = {} if beta0 is None else dict(max_iter=max_iter, tol=tol)
        beta, _, _, conv = fit_nb_glm(torch.tensor(Y[i], dtype=torch.float64), torch.tensor(Xm[i]),
                                      torch.tensor(off[i]), a, beta0=beta0, **kw)
        return beta, conv

    rows = []

    def add(kind, rep, beta, conv, keep=None):
        for r in summarise(beta.numpy(), blocks, direction, sets, masks, keep):
            rows.append(dict(replicate=kind, rep=rep, frac_converged=float(conv.double().mean()), **r))

    beta_hat, conv = fit(X)
    add("observed", 0, beta_hat, conv)
    for b in range(len(boot)):
        idx = np.repeat(np.arange(len(pos)), boot[b, pos])
        beta, conv = fit(X, idx, beta_hat)
        add("bootstrap", b, beta, conv)
    for fam in families:
        covs = [k for k in blocks["interaction"] if k == fam or k.startswith(f"{fam}[")]
        if not covs:
            continue
        base = obs.copy()
        for c in FAMILIES[fam]:
            base[c] = donors[c].to_numpy()[pos]
        if not np.allclose(encode(base, g)[0].to_numpy(), cov.to_numpy()):
            raise ValueError(f"donor table and sample table disagree for {fam}")
        for b in range(len(perms)):
            o = base.copy()
            for c in FAMILIES[fam]:
                o[c] = donors[c].to_numpy()[perms[b]][pos]
            cp, _ = encode(o, g)
            if list(cp.columns) != list(cov.columns):
                continue
            beta, conv = fit(build_design(theta, cp, nui)[0], None, beta_hat)
            add(f"perm_{fam}", b, beta, conv, keep=covs)
    return pd.DataFrame(rows), beta_hat.numpy(), blocks


def tissue_replicates(name, args, g, donors, boot, perms):
    import anndata
    import torch

    torch.set_num_threads(args.threads)
    t0 = time.time()
    res, sets, masks, direction = gene_sets(pd.read_csv(args.results_dir / f"{name}.csv.gz"), args.clock_min_amp)
    phases = pd.read_csv(g.phase_table).set_index("donor")
    ad = anndata.read_h5ad(args.tissue_dir / f"{name}.h5ad")
    ad = ad[select_samples(ad.obs, phases, g)].copy()
    obs = ad.obs.reset_index(drop=True)
    dn = obs["donor"].astype(str).to_numpy()
    theta = phases.loc[dn, "theta"].to_numpy(float)
    Y = ad[:, res["gene"].tolist()].layers["counts"]
    Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
    pos = donors.index.get_indexer(dn)
    if (pos < 0).any():
        raise ValueError(f"{name}: samples whose donor is not in the donor table")
    out, beta_hat, blocks = replicates(Y, obs["library_size"].to_numpy(float), theta, obs, g, res["alpha"].to_numpy(),
                                       direction, sets, masks, donors, pos, boot, perms, args.families,
                                       args.max_iter, args.tol)
    # the observed fit must reproduce the per-gene results (same model, same dispersions)
    diff = max(float(np.nanmax(np.abs(beta_hat[:, jc] - res[f"db_{k}"].to_numpy()))) for k, (jc, _) in
               blocks["interaction"].items())
    out.insert(0, "tissue", name)
    print(f"[{name}] {len(obs)} samples, {int(sets['strong'].sum())} strong + {int(sets['clock'].sum())} clock genes, "
          f"max |db - per-gene results| {diff:.1e}, {time.time() - t0:.0f} s", flush=True)
    return out


def combine(rep):
    """Add tissue 'all': mean of the tissue statistics, replicate by replicate (the bootstrap
    draws and the permutations are shared by the tissues); only replicates present in every tissue."""
    keys = ["replicate", "rep", "covariate", "gene_set"]
    n_t = rep["tissue"].nunique()
    stats = ["amp", "amp_frac_neg", "phase_h", "phase_frac_neg"]
    agg = rep.groupby(keys)[stats].mean()
    agg["n_genes"] = rep.groupby(keys)["n_genes"].sum()
    agg["n_tissues"] = rep.groupby(keys)["tissue"].nunique()
    agg = agg[agg["n_tissues"] == n_t].drop(columns="n_tissues").reset_index()
    agg.insert(0, "tissue", "all")
    return pd.concat([rep, agg], ignore_index=True)


def effects_table(rep):
    """One row per tissue x covariate x gene set x kind: estimate, bootstrap SE and 95 %
    percentile interval, permutation p-values."""
    from scipy.stats import norm

    from cophaser.gene_rhythms import bh

    key = ["tissue", "covariate", "gene_set"]
    groups = {k: d for k, d in rep.groupby(key + ["replicate"])}
    rows = []
    for _, o in rep[rep["replicate"] == "observed"].iterrows():
        k = tuple(o[key])
        boot = groups.get(k + ("bootstrap",), pd.DataFrame(columns=rep.columns))
        perm = groups.get(k + (f"perm_{o['covariate'].split('[')[0]}",), pd.DataFrame(columns=rep.columns))
        for kind, (col, neg) in KINDS.items():
            est = o[col]
            bt = pd.to_numeric(boot[col], errors="coerce").dropna().to_numpy()
            pm = pd.to_numeric(perm[col], errors="coerce").dropna().to_numpy()
            r = dict(tissue=o["tissue"], covariate=o["covariate"], gene_set=o["gene_set"], kind=kind,
                     n_genes=o["n_genes"], estimate=est, frac_neg=o[neg], n_boot=len(bt), n_perm=len(pm))
            if len(bt) > 1:
                r.update(se_boot=bt.std(ddof=1), ci_lo=np.quantile(bt, 0.025), ci_hi=np.quantile(bt, 0.975))
            if len(pm) > 1:
                sd = pm.std(ddof=1)
                r.update(null_sd=sd, p_perm=(1 + np.sum(np.abs(pm) >= abs(est))) / (1 + len(pm)),
                         p_norm=2 * norm.sf(abs(est) / sd))
            rows.append(r)
    eff = pd.DataFrame(rows)
    if "p_norm" in eff:
        eff["q_norm"] = np.nan
        for _, idx in eff.groupby(["gene_set", "kind", eff["tissue"] == "all"]).groups.items():
            eff.loc[idx, "q_norm"] = bh(eff.loc[idx, "p_norm"].to_numpy(float))
    return eff


def phase_quality(donors, phases, g):
    """Phase precision of the donors by sex, age bracket and death type."""
    from gtex_train import merge_levels

    d = donors.join(phases[[c for c in ["theta", "resultant", "kappa", "n_tissues"] if c in phases]])
    if "chiral_ct" in d:
        d["err_chiral_h"] = np.abs(np.angle(np.exp(1j * (d["theta"] - d["chiral_ct"] / H)))) * H
    groups = {"sex": d["sex"]}
    if "age" in d:
        lo = np.floor(d["age"].astype(float) / 10) * 10
        groups["age"] = lo.map(lambda v: f"{int(v)}-{int(v) + 9}" if np.isfinite(v) else np.nan)
    if "hardy" in d:
        groups["hardy"] = merge_levels(d["hardy"], g.merge_hardy) if g.merge_hardy else d["hardy"]
    kmax = d["kappa"].max() if "kappa" in d else np.nan
    rows = []
    for var, lev in groups.items():
        for v, x in d.groupby(lev.to_numpy()):
            r = dict(variable=var, level=v, n_donors=len(x))
            if "kappa" in x:
                r.update(kappa_median=x["kappa"].median(), frac_kappa_max=float(np.mean(x["kappa"] >= 0.99 * kmax)))
            if "resultant" in x:
                r["resultant_median"] = x["resultant"].median()
            if "n_tissues" in x:
                r["n_tissues_median"] = x["n_tissues"].median()
            if "err_chiral_h" in x:
                e = x["err_chiral_h"].dropna()
                r.update(n_chiral=len(e), err_chiral_median_h=e.median(),
                         mean_cos_err_chiral=float(np.mean(np.cos(e / H))) if len(e) else np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


TISSUE_LABELS = {"adipose_subcutaneous": "Adipose", "artery_tibial": "Artery", "lung": "Lung",
                 "muscle_skeletal": "Muscle", "nerve_tibial": "Nerve", "skin_not_sun_exposed_suprapubic":
                 "Skin (not exposed)", "skin_sun_exposed_lower_leg": "Skin (exposed)", "thyroid": "Thyroid",
                 "all": "All tissues"}
COV_LABELS = {"age": "Age (per 10 years)", "sex": "Female vs male", "hardy[0]": "Ventilator (Hardy 0) vs fast death",
              "hardy[3]": "Hardy 3 vs fast death", "hardy[4]": "Hardy 4 vs fast death"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SET_STYLE = {"strong": ("#2a78d6", "rhythmic genes (strong, non-clock)"), "clock": ("#eb6834", "clock genes")}


def plot_forest(eff, path):
    """Estimates and bootstrap 95 % intervals per tissue, one row of panels per covariate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    covs = list(dict.fromkeys(eff["covariate"]))
    tissues = [t for t in sorted(eff["tissue"].unique()) if t != "all"]
    ypos = {t: len(tissues) - i for i, t in enumerate(tissues)}
    ypos["all"] = -0.4
    fig, axes = plt.subplots(len(covs), 2, figsize=(10, 2.9 * len(covs) + 0.8), squeeze=False)
    for r, k in enumerate(covs):
        for c, (kind, xlabel, scale) in enumerate([("amp", "relative amplitude change (%)", 100),
                                                   ("phase", "peak shift (h, > 0 = later)", 1)]):
            ax = axes[r, c]
            for j, (s, (col, _)) in enumerate(SET_STYLE.items()):
                d = eff[(eff["covariate"] == k) & (eff["kind"] == kind) & (eff["gene_set"] == s)]
                for _, row in d.iterrows():
                    if row["tissue"] not in ypos:
                        continue
                    y = ypos[row["tissue"]] + (0.14 if j == 0 else -0.14)
                    if np.isfinite(row.get("ci_lo", np.nan)):
                        ax.plot([row["ci_lo"] * scale, row["ci_hi"] * scale], [y, y], color=col, lw=1.6,
                                solid_capstyle="round", zorder=2)
                    ax.plot(row["estimate"] * scale, y, "o", ms=6.5 if row["tissue"] == "all" else 5, color=col,
                            mec="white", mew=1, zorder=3)
            ax.axvline(0, color=AXIS, lw=1, zorder=1)
            ax.axhline(0.3, color=GRID, lw=0.8, zorder=0)
            ax.set_yticks(list(ypos.values()))
            ax.set_yticklabels([TISSUE_LABELS.get(t, t) for t in ypos] if c == 0 else [], color=INK2, fontsize=8.5)
            if c == 0:
                ax.get_yticklabels()[-1].set_fontweight("bold")
                ax.set_title(COV_LABELS.get(k, k), loc="left", fontsize=10, color=INK)
            ax.set_ylim(-1.0, len(tissues) + 0.6)
            ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
            ax.tick_params(colors=INK2, labelsize=8, length=0)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ["top", "right", "left"]:
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(AXIS)
    handles = [Line2D([0], [0], color=col, marker="o", lw=1.6, mec="white", label=lab)
               for col, lab in SET_STYLE.values()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Change of the rhythm over the gene set (median over genes, bootstrap 95 % interval)",
                 x=0.01, y=0.975, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    g = run_settings(args.results_dir)
    if g.permute_phases is not None:
        raise SystemExit(f"{args.results_dir} was fitted on permuted phases: use the run on the real phases")
    outdir = args.outdir or args.results_dir / "global"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    tissues = args.tissues or sorted(p.name[: -len(".csv.gz")] for p in args.results_dir.glob("*.csv.gz"))
    phases = pd.read_csv(g.phase_table).set_index("donor")
    donors = donor_table(tissues, args.tissue_dir, g, phases)
    U = len(donors)
    rng = np.random.default_rng(args.seed)
    boot = rng.multinomial(U, np.full(U, 1 / U), size=args.n_boot)
    perms = np.stack([rng.permutation(U) for _ in range(args.n_perm)]) if args.n_perm else np.zeros((0, U), int)
    print(f"{len(tissues)} tissues, {U} donors, {args.n_boot} bootstrap replicates, {args.n_perm} permutations "
          f"x {args.families}", flush=True)
    n = len(tissues)
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            parts = list(ex.map(tissue_replicates, tissues, [args] * n, [g] * n, [donors] * n, [boot] * n, [perms] * n))
    else:
        parts = [tissue_replicates(t, args, g, donors, boot, perms) for t in tissues]
    rep = combine(pd.concat(parts, ignore_index=True))
    rep.to_csv(outdir / "replicates.csv.gz", index=False)
    eff = effects_table(rep)
    eff.to_csv(outdir / "global_effects.csv", index=False)
    pq = phase_quality(donors, phases, g)
    pq.to_csv(outdir / "phase_quality.csv", index=False)
    plot_forest(eff, outdir / "global_forest.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 400):
        cols = ["tissue", "covariate", "gene_set", "kind", "n_genes", "estimate", "ci_lo", "ci_hi", "p_perm", "q_norm"]
        print(eff[eff["tissue"] == "all"][[c for c in cols if c in eff]].round(4).to_string(index=False))
        print(pq.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
