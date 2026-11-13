"""
Train CoPhaser on GTEx tissues (one model per tissue) with observed covariates.

For each tissue AnnData built by gtex_prepare.py:
  - rhythmic genes: cophaser.gene_sets.HUMAN_CIRCADIAN_GENE_SET present in the data
  - context genes: the --n-hvg most variable expressed genes + rhythmic genes
  - decoder initialised from the reference clock (mouse, Bmal1 -> ARNTL), unfrozen at
    epoch 10 (circadian defaults): it only gives a starting orientation
  - covariates: age (continuous, bin midpoint), sex (reference male), hardy (categorical)
    -> linear mean shift W_c (step 4)
  - Adam beta2 = 0.99 (deep bulk data, see CLAUDE.md)

Outputs in <outdir>/<tissue_slug>/: model.pt, phases.csv, covariate_effects.csv,
clock_genes.csv, summary.json, learning_curves.png, clock_face.png, chiral_comparison.png.
A summary over tissues is written to <outdir>/summary.csv.

Example:
  python scripts/gtex_train.py --tissues muscle_skeletal --epochs 300
  python scripts/gtex_train.py --all --epochs 300
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import anndata  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import torch  # noqa: E402

from cophaser import report_plots  # noqa: E402
from cophaser.auto_params import suggest_likelihood_weights  # noqa: E402
from cophaser.cycle_configs import CYCLE_TRAINER_DEFAULTS  # noqa: E402
from cophaser.gene_sets import HUMAN_CIRCADIAN_GENE_SET  # noqa: E402
from cophaser.loss import Loss  # noqa: E402
from cophaser.model.CoPhaser import CoPhaser  # noqa: E402
from cophaser.model.decoder_prior import DecoderPrior  # noqa: E402
from cophaser.reference_clock import load_reference_clock  # noqa: E402
from cophaser.trainer import Trainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
H = 24 / (2 * np.pi)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tissue-dir", type=Path, default=REPO_ROOT / "data" / "gtex_v8" / "tissues")
    p.add_argument("--tissues", nargs="*", default=[], help="file stems in --tissue-dir")
    p.add_argument("--all", action="store_true", help="every .h5ad in --tissue-dir")
    p.add_argument("--reference", type=Path, default=REPO_ROOT / "reference" / "clock_reference_learned.csv")
    p.add_argument("--no-reference", action="store_true", help="random decoder initialisation")
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--min-mean-count", type=float, default=5.0, help="drop genes with lower mean count")
    p.add_argument("--n-harm", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--adam-beta2", type=float, default=0.99)
    p.add_argument("--z-input", choices=["raw", "cp10k"], default="cp10k",
                   help="z encoder input; cp10k needed for GTEx depth (run 1 with raw failed)")
    p.add_argument("--auto-likelihood-weights", action="store_true",
                   help="likelihood weights from cophaser.auto_params.suggest_likelihood_weights (ELBO near "
                        "the package target) instead of the circadian defaults 10 / 2, which make the ELBO "
                        "of deep bulk data swamp the entropy / MI / KL terms")
    p.add_argument("--likelihood-weight", type=float, default=None,
                   help="same weight for rhythmic and non-rhythmic likelihoods (overrides the defaults "
                        "and --auto-likelihood-weights)")
    p.add_argument("--freeze-decoder", action="store_true",
                   help="keep the reference genes of the decoder frozen (no unfreezing at epoch 10)")
    p.add_argument("--exclude-hardy0", action="store_true",
                   help="drop ventilator deaths (DTHHRDY = 0) - diagnostic")
    p.add_argument("--bootstrap-seed", type=int, default=None,
                   help="step 7: resample the donors with replacement (after donor selection and gene "
                        "selection) with this seed; default: no resampling")
    p.add_argument("--keep-hardy", nargs="*", default=None,
                   help="keep only these death types (DTHHRDY values, e.g. 1 2); 'hardy' is dropped from "
                        "the covariates if a single level remains")
    p.add_argument("--reference-from-model", type=Path, default=None,
                   help="decoder clock = A/B coefficients of the clock genes of a trained model.pt (frozen), "
                        "e.g. a clock learned on a subset of donors")
    p.add_argument("--human-reference-from-chiral", action="store_true",
                   help="diagnostic: decoder = cosinor of the clock genes on the CHIRAL phases of a random "
                        "half of the donors (frozen); phases evaluated on the other half only")
    p.add_argument("--grid-phase-prior", action="store_true",
                   help="before training, estimate each sample's phase on a grid with the starting clock "
                        "(CoPhaser.grid_phase, no context shift) and pass it as the per-sample phase prior "
                        "(model.add_phase_prior), annealed by the trainer")
    p.add_argument("--phase-prior-kappa", default="auto",
                   help="prior concentration: a number, or 'auto' = per-sample confidence from the grid")
    p.add_argument("--phase-prior-anneal-epochs", type=int, default=50,
                   help="epochs over which the prior is annealed to flat (> --epochs: never fully removed)")
    p.add_argument("--phase-prior-weight", type=float, default=2.0)
    p.add_argument("--phase-table", type=Path, default=None,
                   help="with --grid-phase-prior: CSV (donor, theta, kappa) of phases shared by the tissues of "
                        "a donor (scripts/gtex_shared_phase.py), used instead of this tissue's grid phases; it "
                        "must be in the frame of the clock used here (e.g. the same frozen human clocks). "
                        "Samples whose donor is missing keep their own grid phase")
    p.add_argument("--grid-points", type=int, default=96)
    p.add_argument("--grid-dispersion", type=float, default=0.1,
                   help="NB dispersion used by the grid before training (learned dispersion is random then)")
    p.add_argument("--grid-refresh-every", type=int, default=None,
                   help="with --grid-phase-prior: re-estimate the prior on the grid with the current decoder "
                        "every k epochs (Trainer phase_prior_refresh_every)")
    p.add_argument("--grid-refresh-use-z", action="store_true",
                   help="the refresh grid includes the fitted context shift and lambda (default: not)")
    p.add_argument("--decoder-phase-from-grid", action="store_true",
                   help="the decoder sees the grid phase instead of the encoder's (Trainer decoder_phase_from_prior)")
    p.add_argument("--mi-weight", type=float, default=None, help="override MI_weight (circadian default 50)")
    p.add_argument("--entropy-weight-factor", type=float, default=None,
                   help="override entropy_weight_factor (circadian default 50)")
    p.add_argument("--covariates", nargs="*", default=["age", "sex", "hardy"])
    p.add_argument("--covariate-amplitude", action="store_true",
                   help="covariates also modulate the amplitude of each gene's rhythm (step 5)")
    p.add_argument("--covariate-phase", action="store_true",
                   help="covariates also delay each gene's rhythm (step 6; relative delays)")
    p.add_argument("--covariate-phase-reference", default=None,
                   help="clock gene with delay 0 (default: amplitude-weighted mean of the clock genes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def select_genes(ad, n_hvg, min_mean_count):
    clock = [g for g in HUMAN_CIRCADIAN_GENE_SET if g in ad.var_names]
    expressed = np.asarray(ad.layers["counts"].mean(0)).ravel() >= min_mean_count
    tmp = ad[:, expressed].copy()
    tmp.X = tmp.layers["counts"].astype(np.float32)
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_hvg, tmp.n_vars), flavor="seurat")
    hvg = list(tmp.var_names[tmp.var["highly_variable"].to_numpy()])
    context = sorted(set(hvg) | set(clock))
    return clock, context


def circular_compare(theta, chiral_hours):
    """Best rotation (and reflection) of our phases onto CHIRAL's; returns errors in hours."""
    m = np.isfinite(chiral_hours)
    a, b = theta[m], chiral_hours[m] / H
    best = None
    for sign in (1, -1):
        d = sign * a - b
        off = np.angle(np.mean(np.exp(1j * d)))
        err = np.abs(np.angle(np.exp(1j * (d - off)))) * H
        r = np.abs(np.mean(np.exp(1j * d)))
        if best is None or np.median(err) < best["median_abs_error_h"]:
            best = dict(sign=sign, offset_h=float(off * H), median_abs_error_h=float(np.median(err)),
                        mean_abs_error_h=float(np.mean(err)), resultant=float(r), n=int(m.sum()))
    best["aligned"] = np.mod(best["sign"] * a - best["offset_h"] / H, 2 * np.pi) * H
    best["chiral"] = chiral_hours[m]
    return best


def human_reference_from_chiral(ad, clock, seed):
    """Fourier table (A_1, B_1) of the clock genes fitted on the CHIRAL phases of a random
    half of the donors, in the decoder's units (natural log of the count fraction) and in
    the CHIRAL time frame. Returns (table, train_mask over samples)."""
    rng = np.random.default_rng(seed)
    has = ad.obs["chiral_ct"].notna().to_numpy()
    donors = ad.obs["donor"].astype(str).to_numpy()
    train_donors = {d for d in np.unique(donors[has]) if rng.random() < 0.5}
    train = has & np.isin(donors, list(train_donors))
    X = ad[:, clock].layers["counts"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    y = np.log((X[train] + 0.5) / ad.obs["library_size"].to_numpy()[train, None])
    t = ad.obs["chiral_ct"].to_numpy(float)[train] / H
    D = np.column_stack([np.ones_like(t), np.cos(t), np.sin(t)])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    fourier = pd.DataFrame({"A_1": coef[1], "B_1": coef[2]}, index=clock)
    print(f"  human reference from CHIRAL: {train.sum()} training samples, "
          f"amplitudes {np.round(np.hypot(coef[1], coef[2]), 2).tolist()}", flush=True)
    return fourier, train


def decoder_amp_phase(model, genes):
    W = model.rhythmic_decoder.fourier_coefficients.weight.detach().cpu().numpy()
    idx = [model.context_genes.index(g) for g in genes]
    A, B = W[idx, 0], W[idx, 1]
    return pd.DataFrame({"amp": np.hypot(A, B), "phase": np.mod(np.arctan2(B, A), 2 * np.pi)}, index=genes)


HARDY_GROUPS = {"0": ["0"], "1-2": ["1", "2"], "3-4": ["3", "4"]}


def clock_from_model(path, clock):
    """Fourier table (A_h, B_h columns) of the clock genes in a trained CoPhaser checkpoint."""
    m = CoPhaser.load(path)
    W = m.rhythmic_decoder.fourier_coefficients.weight.detach().cpu().numpy()
    genes = [g for g in clock if g in m.context_genes]
    idx = [m.context_genes.index(g) for g in genes]
    cols = [f"{ab}_{h}" for h in range(1, W.shape[1] // 2 + 1) for ab in ("A", "B")]
    print(f"  clock from {path}: {len(genes)} genes", flush=True)
    return pd.DataFrame(W[idx], index=genes, columns=cols)


def errors_by_hardy(theta, chiral_eval, hardy):
    """Median |error| (h) vs CHIRAL per death-type group, after the alignment fitted on all samples."""
    cmp = circular_compare(theta, chiral_eval)
    ok = np.isfinite(chiral_eval)
    err = np.full(len(theta), np.nan)
    aligned = cmp["sign"] * theta[ok] - cmp["offset_h"] / H
    err[ok] = np.abs(np.angle(np.exp(1j * (aligned - chiral_eval[ok] / H)))) * H
    out = {}
    for grp, levels in HARDY_GROUPS.items():
        m = ok & np.isin(hardy, levels)
        if m.sum() >= 10:
            out[grp] = dict(n=int(m.sum()), median_abs_error_h=float(np.median(err[m])))
    return out


def phase_errors_vs_chiral(theta, chiral_eval, same_frame):
    """Median |error| (h) vs CHIRAL after the best rotation/reflection, plus the raw error
    when theta is already in the CHIRAL frame (human reference fitted on CHIRAL)."""
    out = {"aligned_h": circular_compare(theta, chiral_eval)["median_abs_error_h"]}
    if same_frame:
        ev = np.isfinite(chiral_eval)
        out["raw_h"] = float(np.median(np.abs(np.angle(np.exp(1j * (theta[ev] - chiral_eval[ev] / H)))) * H))
    return out


def bootstrap_donors(ad, seed, donor_key="donor"):
    """Donor bootstrap: draw as many donors as there are, with replacement, and keep all the
    samples of each drawn donor (one per donor in a single-tissue file). obs_names are made
    unique (<name>#b<k>); obs["bootstrap_donor"] numbers the draws, so a donor drawn twice
    counts as two donors."""
    rng = np.random.default_rng(seed)
    donors = np.asarray(pd.unique(ad.obs[donor_key].astype(str)))
    drawn = rng.choice(donors, size=len(donors), replace=True)
    by_donor = ad.obs.groupby(ad.obs[donor_key].astype(str), observed=True).indices
    idx = np.concatenate([by_donor[d] for d in drawn])
    draw = np.concatenate([np.full(len(by_donor[d]), k) for k, d in enumerate(drawn)])
    out = ad[idx].copy()
    out.obs_names = [f"{n}#b{k}" for k, n in enumerate(ad.obs_names[idx])]
    out.obs["bootstrap_donor"] = draw
    return out


def phase_table_lookup(obs, table, donor_key="donor"):
    """Per-sample (theta, kappa, found) from a donor phase table (columns donor, theta, kappa);
    found is False (theta, kappa NaN) for samples whose donor is not in the table. Works on
    bootstrap replicates (the donor column is kept)."""
    t = table.assign(donor=table["donor"].astype(str)).set_index("donor")
    if t.index.has_duplicates:
        raise ValueError("phase table: duplicated donors")
    donors = obs[donor_key].astype(str)
    theta = donors.map(t["theta"]).to_numpy(float)
    kappa = donors.map(t["kappa"]).to_numpy(float)
    return theta, kappa, np.isfinite(theta) & np.isfinite(kappa)


def train_tissue(path, args, outdir):
    t_start = time.time()
    name = path.stem
    out = outdir / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ad = anndata.read_h5ad(path)
    if args.exclude_hardy0:
        ad = ad[(ad.obs["hardy"].astype(str) != "0").to_numpy()].copy()
    if args.keep_hardy:
        ad = ad[ad.obs["hardy"].astype(str).isin(args.keep_hardy).to_numpy()].copy()
    covs = list(args.covariates)
    if "hardy" in covs and ad.obs["hardy"].astype(str).nunique() < 2:
        covs.remove("hardy")
        print(f"[{name}] single death type left: 'hardy' dropped from the covariates", flush=True)
    clock, context = select_genes(ad, args.n_hvg, args.min_mean_count)
    ad = ad[:, context].copy()
    if args.bootstrap_seed is not None:  # genes are selected on the original donors
        ad = bootstrap_donors(ad, args.bootstrap_seed)
        print(f"[{name}] donor bootstrap (seed {args.bootstrap_seed}): {ad.obs['donor'].nunique()} distinct "
              f"donors in {ad.n_obs} samples", flush=True)
    print(f"[{name}] {ad.n_obs} samples, {len(context)} context genes, clock genes: {clock}", flush=True)

    # CHIRAL phases used for evaluation (all samples, or only held-out donors, see below)
    chiral_eval = ad.obs["chiral_ct"].to_numpy(float) if "chiral_ct" in ad.obs else None
    if args.human_reference_from_chiral:
        fourier, train_mask = human_reference_from_chiral(ad, clock, args.seed)
        chiral_eval = np.where(train_mask, np.nan, chiral_eval)  # evaluate on held-out donors only
    elif args.reference_from_model is not None:
        fourier = clock_from_model(args.reference_from_model, clock)
    elif not args.no_reference:
        fourier = load_reference_clock(args.reference, rename_map={"Bmal1": "ARNTL"}, n_harm=args.n_harm)

    model = CoPhaser(rhythmic_gene_names=clock, context_genes=context,
                     **dict(CYCLE_TRAINER_DEFAULTS["circadian"]["model"], n_harm=args.n_harm, z_input=args.z_input))
    n_prior = 0
    if args.human_reference_from_chiral or args.reference_from_model is not None or not args.no_reference:
        n_prior = int(DecoderPrior.load_fourier_coefficients_prior(fourier, model, freeze_defined_weights=True).sum())
    cat = [c for c in covs if c in ("sex", "hardy")]
    model.load_anndata(ad, layer_to_use="counts", library_size_field="library_size",
                       covariate_keys=covs or None, categorical_covariates=cat or None,
                       covariate_reference={"sex": "male"} if "sex" in covs else None,
                       covariate_amplitude=args.covariate_amplitude and bool(covs),
                       covariate_phase=args.covariate_phase and bool(covs),
                       covariate_phase_reference=args.covariate_phase_reference)

    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"], adam_betas=(0.9, args.adam_beta2))
    if args.no_reference or args.freeze_decoder or args.human_reference_from_chiral or args.reference_from_model:
        kw["unfreeze_epoch_layer"] = []
    if args.auto_likelihood_weights:
        kw.update(suggest_likelihood_weights(model))
    if args.likelihood_weight is not None:
        kw.update(rhythmic_likelihood_weight=args.likelihood_weight,
                  non_rhythmic_likelihood_weight=args.likelihood_weight)
    if args.mi_weight is not None:
        kw["MI_weight"] = args.mi_weight
    if args.entropy_weight_factor is not None:
        kw["entropy_weight_factor"] = args.entropy_weight_factor
    grid_summary = {}
    has_chiral = chiral_eval is not None and np.isfinite(chiral_eval).any()
    if args.grid_phase_prior:
        gp = model.grid_phase(n_points=args.grid_points, use_z=False, dispersion=args.grid_dispersion)
        if args.phase_table is not None:
            if args.grid_refresh_every:
                raise SystemExit("--phase-table is for fixed phases: drop --grid-refresh-every")
            t_theta, t_kappa, found = phase_table_lookup(ad.obs, pd.read_csv(args.phase_table))
            gp["theta"] = np.where(found, t_theta, gp["theta"])
            gp["kappa"] = np.where(found, t_kappa, gp["kappa"])
            grid_summary.update(phase_table=str(args.phase_table), phase_table_found=int(found.sum()))
            print(f"[{name}] shared donor phases from {args.phase_table}: {int(found.sum())}/{ad.n_obs} samples "
                  f"(others keep this tissue's grid phase)", flush=True)
        ad.obs["grid_phase_prior"] = gp["theta"]
        ad.obs["grid_phase_kappa"] = gp["kappa"] if args.phase_prior_kappa == "auto" else float(args.phase_prior_kappa)
        model.add_phase_prior(ad, "grid_phase_prior", "grid_phase_kappa")
        kw.update(phase_prior_anneal_epochs=args.phase_prior_anneal_epochs,
                  phase_prior_weight=args.phase_prior_weight,
                  decoder_phase_from_prior=args.decoder_phase_from_grid)
        if args.grid_refresh_every:
            kw.update(phase_prior_refresh_every=args.grid_refresh_every,
                      phase_prior_refresh_kappa="auto" if args.phase_prior_kappa == "auto" else float(args.phase_prior_kappa),
                      phase_prior_refresh_kwargs=dict(n_points=args.grid_points, use_z=args.grid_refresh_use_z,
                                                      dispersion=args.grid_dispersion))
        grid_summary["grid_prior_kappa_median"] = float(np.median(ad.obs["grid_phase_kappa"]))
        if has_chiral:
            e = phase_errors_vs_chiral(gp["theta"], chiral_eval, args.human_reference_from_chiral)
            grid_summary.update({f"grid_prior_{k}": v for k, v in e.items()})
        print(f"[{name}] grid phase prior: kappa median {grid_summary['grid_prior_kappa_median']:.2f}, "
              f"anneal {args.phase_prior_anneal_epochs} epochs, weight {args.phase_prior_weight}; "
              f"vs CHIRAL {grid_summary.get('grid_prior_aligned_h', float('nan')):.2f} h", flush=True)
    elif args.grid_refresh_every or args.decoder_phase_from_grid or args.phase_table is not None:
        raise SystemExit("--grid-refresh-every / --decoder-phase-from-grid / --phase-table need --grid-phase-prior")
    print(f"[{name}] likelihood weights: rhythmic {kw['rhythmic_likelihood_weight']:.3g}, "
          f"non-rhythmic {kw['non_rhythmic_likelihood_weight']:.3g}; decoder "
          f"{'frozen' if kw['unfreeze_epoch_layer'] == [] else 'unfrozen at epoch 10'}", flush=True)
    trainer = Trainer(model, Loss.compute_loss, cycle="circadian", **kw)
    t0 = time.time()
    trainer.train_model(n_epochs=args.epochs, lr=args.lr, subsample=None, device="cpu", silent=True)
    train_s = time.time() - t0
    model.to("cpu")
    model.save(out / "model.pt")

    _, space = model.get_outputs()
    theta = space["theta"].numpy()
    obs = ad.obs.copy()
    obs["theta"] = theta
    obs["phase_h"] = np.mod(theta, 2 * np.pi) * H

    summary = dict(tissue=str(obs["tissue"].iloc[0]), file=name, n_samples=int(ad.n_obs),
                   n_context_genes=len(context), clock_genes=clock, n_prior_genes=n_prior,
                   train_seconds=round(train_s, 1), epochs=args.epochs,
                   rhythmic_likelihood_weight=kw["rhythmic_likelihood_weight"],
                   non_rhythmic_likelihood_weight=kw["non_rhythmic_likelihood_weight"],
                   decoder_frozen=kw["unfreeze_epoch_layer"] == [],
                   final_elbo=float(trainer.losses_history["elbo_loss"][-1]),
                   covariates=model.covariate_names, **grid_summary)
    summary.update(MI_weight=kw["MI_weight"], entropy_weight_factor=kw["entropy_weight_factor"])
    if args.grid_phase_prior:
        summary.update(grid_refresh_every=args.grid_refresh_every, grid_refresh_use_z=args.grid_refresh_use_z,
                       decoder_phase_from_grid=args.decoder_phase_from_grid,
                       grid_refresh_log=[[e, round(c * H, 3), round(k, 2)] for e, c, k in trainer.phase_prior_refresh_log])
        summary.update(phase_prior_anneal_epochs=args.phase_prior_anneal_epochs,
                       phase_prior_weight=args.phase_prior_weight, phase_prior_kappa=args.phase_prior_kappa)
    # phase by grid with the trained decoder, for comparison with the encoder
    grids = {"grid_final_noz": model.grid_phase(n_points=args.grid_points, use_z=False),
             "grid_final_z": model.grid_phase(n_points=args.grid_points, use_z=True)}
    for k, g in grids.items():
        obs[f"{k}_theta"] = g["theta"]
        if has_chiral:
            e = phase_errors_vs_chiral(g["theta"], chiral_eval, args.human_reference_from_chiral)
            summary.update({f"{k}_{kk}": v for kk, v in e.items()})
    summary["keep_hardy"] = args.keep_hardy
    summary["bootstrap_seed"] = args.bootstrap_seed
    summary["reference_from_model"] = str(args.reference_from_model) if args.reference_from_model else None
    if has_chiral:
        hardy = ad.obs["hardy"].astype(str).to_numpy()
        for k, th in (("encoder", theta), ("grid_final_noz", grids["grid_final_noz"]["theta"])):
            for grp, v in errors_by_hardy(th, chiral_eval, hardy).items():
                summary[f"{k}_hardy{grp}_n"] = v["n"]
                summary[f"{k}_hardy{grp}_median_abs_error_h"] = v["median_abs_error_h"]

    figs = {"learning_curves": report_plots.plot_learning_curves(trainer.losses_history)}
    amp_phase = decoder_amp_phase(model, clock)
    if chiral_eval is not None and np.isfinite(chiral_eval).any():
        cmp = circular_compare(theta, chiral_eval)
        summary.update({f"chiral_{k}": v for k, v in cmp.items() if k not in ("aligned", "chiral")})
        summary["chiral_eval_on"] = "held-out donors" if args.human_reference_from_chiral else "all samples"
        if args.human_reference_from_chiral:  # same time frame as CHIRAL: no alignment needed
            ev = np.isfinite(chiral_eval)
            raw = np.abs(np.angle(np.exp(1j * (theta[ev] - chiral_eval[ev] / H)))) * H
            summary["chiral_raw_median_abs_error_h"] = float(np.median(raw))
        obs.loc[np.isfinite(chiral_eval), "phase_h_aligned_to_chiral"] = cmp["aligned"]
        # express gene peak times in the CHIRAL frame too
        amp_phase["peak_h_chiral_frame"] = np.mod(cmp["sign"] * amp_phase["phase"] - cmp["offset_h"] / H,
                                                  2 * np.pi) * H
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
        ax[0].scatter(cmp["chiral"], cmp["aligned"], s=6, alpha=0.5, color="#2a6f97", lw=0)
        for k in (-1, 0, 1):
            ax[0].plot([0, 24], [24 * k, 24 * (k + 1)], ls="--", color="#c0392b", lw=1)
        ax[0].set(xlim=(0, 24), ylim=(0, 24), xlabel="CHIRAL phase (h)", ylabel="CoPhaser phase, aligned (h)",
                  title=f"{summary['tissue']}: median |diff| {cmp['median_abs_error_h']:.2f} h")
        ax[0].set_aspect("equal")
        ax[0].grid(alpha=0.3)
        d = np.angle(np.exp(1j * (cmp["aligned"] - cmp["chiral"]) / H)) * H
        ax[1].hist(d, bins=np.linspace(-12, 12, 49), color="#2a6f97")
        ax[1].set(xlabel="CoPhaser − CHIRAL (h)", ylabel="samples", title="Difference after best alignment")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        figs["chiral_comparison"] = fig
    amp_phase.to_csv(out / "clock_genes.csv")

    peak = amp_phase["peak_h_chiral_frame"] / H if "peak_h_chiral_frame" in amp_phase else amp_phase["phase"]
    fig = plt.figure(figsize=(5.2, 5.2))
    axp = fig.add_subplot(projection="polar")
    frame = "CHIRAL time frame" if "peak_h_chiral_frame" in amp_phase else "model frame"
    report_plots.plot_clock_face(amp_phase["amp"], peak, clock, ax=axp,
                                 title=f"{summary['tissue']}: learned clock ({frame})\nradius = log-amplitude")
    figs["clock_face"] = fig

    for k, f in figs.items():
        f.savefig(out / f"{k}.png", dpi=140, bbox_inches="tight")
        plt.close(f)
    model.covariate_effects().to_csv(out / "covariate_effects.csv", index=False)
    obs.to_csv(out / "phases.csv")
    summary["total_seconds"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{name}] done in {summary['total_seconds']} s; "
          f"vs CHIRAL: median |diff| {summary.get('chiral_median_abs_error_h', float('nan')):.2f} h "
          f"(grid with trained decoder: {summary.get('grid_final_noz_aligned_h', float('nan')):.2f} h)", flush=True)
    return summary


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    outdir = args.outdir or REPO_ROOT / "results" / "gtex" / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    paths = sorted(args.tissue_dir.glob("*.h5ad")) if args.all else [args.tissue_dir / f"{t}.h5ad" for t in args.tissues]
    if not paths:
        raise SystemExit("No tissue given (--tissues ... or --all).")
    rows = []
    for p in paths:
        try:
            rows.append(train_tissue(p, args, outdir))
        except Exception:  # keep going with the other tissues
            print(f"[{p.stem}] FAILED\n{traceback.format_exc()}", flush=True)
        pd.DataFrame(rows).drop(columns=["clock_genes", "covariates"], errors="ignore").to_csv(outdir / "summary.csv", index=False)
    print(f"\nAll done. Outputs in {outdir}")


if __name__ == "__main__":
    main()
