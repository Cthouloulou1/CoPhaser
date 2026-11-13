"""
Positive control on real single-cell data: mouse whole aorta (Auerbach et al. 2022,
Tempo paper, GEO GSE206583), as in paper/code/figure_5/aorta.ipynb, where the CoPhaser
preprint reports a median absolute phase error of ~1.67 h.

Data: one 10x library per time point (ZT0, ZT6, ZT12, ZT18), runs merged
(AUERBACH_2022_AORTA_MERGED/filtered_GSM625718{3,4,5,6}.h5ad, ZT from the GEO titles).
The notebook's `qc_pass` column and cell types are not available here: a simple QC
(library >= --min-counts, >= 500 detected genes, mitochondrial fraction < 20 %) is used
instead, and all cell types are kept (as in the notebook's training).

Model and training follow the notebook (circadian defaults, amp_phase_circadian prior,
decoder released at epoch 10). Options allow the GTEx settings (Adam beta2, z_input).

Covariates (--covariates, any subset of):
  celltype  marker-based annotation (highest score_genes score; "Other" if all scores <= 0),
            reference level "SMC"; present at every ZT, so not confounded with time
  mito      mitochondrial count fraction (continuous, technical)
  sample    10x library = one ZT each: fully confounded with time (negative control,
            W_c can absorb the rhythm as per-sample mean shifts)
For the celltype levels, the learned W_c is compared to the observed log fold change of
mean cp10k vs SMC (Pearson r and slope; W_c is identifiable up to a constant per level).

Reported: median |phase error| vs ZT, (a) as in the notebook (optimal_shift: best shift
and direction minimising the median), (b) mean-offset alignment; plus diagnostics of the
rhythmic encoder (encoder phase vs grid maximum-likelihood phase under the trained
decoder, centring of the encoder input).
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import anndata  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from cophaser import gene_sets, report_plots, utils  # noqa: E402
from cophaser.cycle_configs import CYCLE_TRAINER_DEFAULTS  # noqa: E402
from cophaser.loss import Loss  # noqa: E402
from cophaser.model.CoPhaser import CoPhaser  # noqa: E402
from cophaser.model.decoder_prior import DecoderPrior  # noqa: E402
from cophaser.trainer import Trainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/data/shared/upnae_shared/2024_Circa_IE/AUERBACH_2022_AORTA_MERGED")
SAMPLES = {"GSM6257183": 0, "GSM6257184": 6, "GSM6257185": 12, "GSM6257186": 18}  # GEO titles
H = 24 / (2 * np.pi)
CELLTYPE_MARKERS = {
    "SMC": ["Acta2", "Myh11", "Cnn1", "Tagln"],
    "Fibroblast": ["Dcn", "Pdgfra", "Lum", "Col1a1"],
    "Endothelial": ["Pecam1", "Cdh5", "Kdr"],
    "Immune": ["Ptprc", "Cd74", "Lyz2"],
    "Adipocyte": ["Adipoq", "Cfd", "Plin1"],
}
COVARIATES = ("celltype", "mito", "sample")
GENES_TEMPO = [
    "Arntl", "Arntl2", "Bhlhe41", "Ciart", "Clock", "Cry1", "Cry2", "Csnk1a1", "Csnk1d",
    "Csnk1e", "Dbp", "Dec1", "Dec2", "Fbxo21", "Fbxo3", "Gm129", "Hlf", "Nampt", "Nfil3",
    "Npas2", "Nr1d1", "Nr1d2", "Per1", "Per2", "Per3", "Rora", "Rorc", "Tef",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--min-counts", type=float, default=1000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--adam-beta2", type=float, default=0.999, help="0.999 = torch default, as in the notebook")
    p.add_argument("--z-input", choices=["raw", "cp10k"], default="raw")
    p.add_argument("--covariates", nargs="*", default=[], choices=COVARIATES)
    p.add_argument("--grid-points", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def load_aorta(data_dir, min_counts):
    parts = []
    for gsm, zt in SAMPLES.items():
        a = anndata.read_h5ad(data_dir / f"filtered_{gsm}.h5ad")
        a = anndata.AnnData(X=sp.csr_matrix(a.X), var=a.var[[]].copy(), obs=pd.DataFrame(index=a.obs_names))
        a.obs_names = [f"{gsm}_{c}" for c in a.obs_names]
        a.obs["sample"], a.obs["zt"] = gsm, zt
        parts.append(a)
    ad = anndata.concat(parts)
    lib = np.asarray(ad.X.sum(1)).ravel()
    n_genes = np.asarray((ad.X > 0).sum(1)).ravel()
    mt = np.asarray(ad[:, ad.var_names.str.startswith("mt-")].X.sum(1)).ravel() / lib
    keep = (lib >= min_counts) & (n_genes >= 500) & (mt < 0.2)
    ad = ad[keep].copy()
    ad.layers["total"] = ad.X.copy()
    ad.obs["library_size"] = np.asarray(ad.X.sum(1)).ravel()
    ad.obs["mito"] = mt[keep]
    annotate_celltypes(ad)
    names = ad.var_names.tolist()  # the notebook's prior uses Bmal1
    if "Arntl" in names and "Bmal1" not in names:
        names[names.index("Arntl")] = "Bmal1"
        ad.var_names = names
    print(f"{ad.n_obs} cells after QC ({keep.mean():.1%} kept); per ZT: "
          f"{ad.obs['zt'].value_counts().sort_index().to_dict()}", flush=True)
    return ad


def annotate_celltypes(ad):
    """Marker-based cell type in ad.obs["celltype"] (score_genes on log1p cp10k)."""
    import scanpy as sc

    tmp = anndata.AnnData(X=ad.X.copy(), obs=pd.DataFrame(index=ad.obs_names), var=pd.DataFrame(index=ad.var_names))
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    scores = {}
    for ct, genes in CELLTYPE_MARKERS.items():
        genes = [g for g in genes if g in tmp.var_names]
        sc.tl.score_genes(tmp, genes, score_name=ct, random_state=0)
        scores[ct] = tmp.obs[ct].to_numpy()
    S = pd.DataFrame(scores, index=ad.obs_names)
    ad.obs["celltype"] = np.where(S.max(1) > 0, S.idxmax(1), "Other")
    print("cell types:", ad.obs["celltype"].value_counts().to_dict(), flush=True)
    print("cells per cell type and ZT:\n", pd.crosstab(ad.obs["celltype"], ad.obs["zt"]).to_string(), flush=True)


def celltype_effect_check(model, ad):
    """Learned W_c (celltype levels) vs observed log fold change of mean cp10k vs SMC."""
    est = model.covariate_effects()
    est = est[est["covariate"].str.startswith("celltype[")]
    X = ad[:, model.context_genes].layers["total"]
    cp = X.multiply(1e4 / ad.obs["library_size"].to_numpy()[:, None]).tocsr()
    ct = ad.obs["celltype"].to_numpy()
    ref = np.log(np.asarray(cp[ct == "SMC"].mean(0)).ravel() + 1e-3)
    rows, long = {}, []
    for cov, g in est.groupby("covariate"):
        level = cov[len("celltype["):-1]
        obs_lfc = np.log(np.asarray(cp[ct == level].mean(0)).ravel() + 1e-3) - ref
        w = g.set_index("gene").loc[model.context_genes, "value"].to_numpy()
        slope, _ = np.polyfit(obs_lfc, w, 1)
        rows[level] = dict(n_cells=int((ct == level).sum()), pearson_r=float(np.corrcoef(obs_lfc, w)[0, 1]),
                           slope=float(slope))
        long.append(pd.DataFrame({"gene": model.context_genes, "level": level, "observed_lfc": obs_lfc, "W_c": w}))
    return rows, pd.concat(long) if long else None


def optimal_shift_mad(p, p0, n_s=200):
    """Notebook metric: median circular deviation after the best of n_s shifts, both directions."""
    shifts = np.linspace(0, 2 * np.pi, n_s)
    best = np.inf
    for sign in (1, -1):
        cand = (sign * p[:, None] - shifts[None, :]) % (2 * np.pi)
        d = np.abs(p0[:, None] % (2 * np.pi) - cand)
        d = np.minimum(d, 2 * np.pi - d)
        best = min(best, np.median(d, axis=0).min())
    return best * H


def mean_offset_error(p, p0):
    """Median |error| (h) after the mean-offset rotation, best direction."""
    best = np.inf
    for sign in (1, -1):
        d = sign * p - p0
        off = np.angle(np.mean(np.exp(1j * d)))
        best = min(best, np.median(np.abs(np.angle(np.exp(1j * (d - off))))) * H)
    return best


def grid_phase(model, space, gen, n_points, covariates=None, batch=2048):
    """Maximum-likelihood phase (rhythmic genes) on a grid, z, lambda, b_z fixed at their fitted values."""
    grid = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    out = np.empty(model.variable_genes.shape[0])
    with torch.no_grad():
        for s in range(0, len(out), batch):
            e = min(s + batch, len(out))
            ll = torch.empty(e - s, n_points)
            for i, t in enumerate(grid):
                f = model.rhythmic_encoder.fourier_basis_expansion(torch.full((e - s,), float(t)))
                p = model.generative(f, space["z"][s:e], space["b_z"][s:e], gen["lambda"][s:e],
                                     model.library_size[s:e], space["f_g1"],
                                     covariates=None if covariates is None else covariates[s:e])
                ll[:, i] = Loss.log_likelihood_NB_weighted(
                    model.variable_genes[s:e], p["px_rate"], p["theta_dispersion"],
                    model.rhythmic_gene_indices, model.non_rhythmic_gene_indices, 1, 0)
            out[s:e] = grid[ll.argmax(1).numpy()]
    return out


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outdir = args.outdir or REPO_ROOT / "results" / "aorta" / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    ad = load_aorta(args.data_dir, args.min_counts)
    context = utils.get_variable_genes(ad, 2000, layer="total")
    rhythmic = sorted((set(gene_sets.SMALL_CIRCADIAN_GENE_SET) | set(GENES_TEMPO)) & set(ad.var_names))
    print(f"{len(rhythmic)} rhythmic genes: {rhythmic}", flush=True)

    model = CoPhaser(rhythmic, context, **dict(CYCLE_TRAINER_DEFAULTS["circadian"]["model"], z_input=args.z_input))
    cov = list(args.covariates)
    cat = [c for c in cov if c in ("celltype", "sample")]
    model.load_anndata(ad, layer_to_use="total", covariate_keys=cov or None, categorical_covariates=cat or None,
                       covariate_reference={"celltype": "SMC"} if "celltype" in cov else None)
    print(f"covariates: {model.covariate_names}", flush=True)
    DecoderPrior.define_decoder_prior(amp_phase_prior=gene_sets.amp_phase_circadian, model=model)
    kw = dict(CYCLE_TRAINER_DEFAULTS["circadian"]["trainer"], adam_betas=(0.9, args.adam_beta2))
    trainer = Trainer(model, Loss.compute_loss, cycle="circadian", **kw)
    t0 = time.time()
    trainer.train_model(n_epochs=args.epochs, lr=args.lr, subsample=None, device="cpu", silent=True)
    train_s = time.time() - t0
    model.to("cpu")
    model.eval()
    model.save(outdir / "model.pt")

    with torch.no_grad():
        gen, space = model.forward(model.variable_genes, model.rhythmic_genes, model.library_size, -1,
                                   use_max_posterior=True, covariates=model.covariates)
        mu_z = gen["Z"][:, model.rhythmic_gene_indices]
        x_in = model.rhythmic_encoder.normalize_expression(
            model.rhythmic_genes, model.library_size.view(-1, 1), mu_z, gen["lambda"], model.use_mu_z_encoder).numpy()
    theta = space["theta"].numpy()
    theta_grid = grid_phase(model, space, gen, args.grid_points, covariates=model.covariates)
    zt = ad.obs["zt"].to_numpy(float) / H

    summary = dict(
        n_cells=int(ad.n_obs), n_rhythmic=len(rhythmic), epochs=args.epochs, adam_beta2=args.adam_beta2,
        z_input=args.z_input, seed=args.seed, covariates=cov, covariate_columns=model.covariate_names, train_seconds=round(train_s, 1),
        final_elbo=float(trainer.losses_history["elbo_loss"][-1]),
        encoder_mad_notebook_h=optimal_shift_mad(theta, zt),
        encoder_median_error_mean_offset_h=mean_offset_error(theta, zt),
        grid_mad_notebook_h=optimal_shift_mad(theta_grid, zt),
        grid_median_error_mean_offset_h=mean_offset_error(theta_grid, zt),
        theta_resultant=float(np.abs(np.mean(np.exp(1j * theta)))),
        lambda_median=float(np.median(gen["lambda"].numpy())),
        encoder_input_abs_mean_per_gene_median=float(np.median(np.abs(x_in.mean(0)))),
        encoder_input_sd_per_gene_median=float(np.median(x_in.std(0))),
    )
    if cov:
        model.covariate_effects().to_csv(outdir / "covariate_effects.csv", index=False)
    if "celltype" in cov:
        ct_rows, ct_long = celltype_effect_check(model, ad)
        summary["celltype_W_c_vs_observed"] = ct_rows
        ct_long.to_csv(outdir / "celltype_W_c_vs_observed.csv", index=False)
        fig, axes = plt.subplots(1, len(ct_rows), figsize=(3.2 * len(ct_rows), 3.2), squeeze=False)
        for ax, (level, g) in zip(axes[0], ct_long.groupby("level")):
            ax.scatter(g["observed_lfc"], g["W_c"], s=3, alpha=0.4)
            ax.set(title=f"{level} vs SMC (r = {ct_rows[level]['pearson_r']:.2f})", xlabel="observed log FC")
        axes[0, 0].set_ylabel("learned W_c")
        fig.tight_layout()
        fig.savefig(outdir / "celltype_W_c_vs_observed.png", dpi=120)
        plt.close(fig)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"zt": ad.obs["zt"].to_numpy(), "theta": theta, "theta_grid": theta_grid},
                 index=ad.obs_names).to_csv(outdir / "phases.csv")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True)

    fig = report_plots.plot_learning_curves(trainer.losses_history)
    fig.savefig(outdir / "learning_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (name, th) in zip(axes, (("encoder", theta), ("grid ML", theta_grid))):
        for z in sorted(SAMPLES.values()):
            m = ad.obs["zt"].to_numpy() == z
            ax.hist(np.mod(th[m], 2 * np.pi) * H, bins=48, range=(0, 24), histtype="step", lw=1.5, label=f"ZT{z}")
        ax.set(xlabel="inferred phase (h, model frame)", title=f"{name} phase by ZT")
    axes[0].set_ylabel("cells")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "phase_by_zt.png", dpi=130)
    plt.close(fig)
    print(f"All done. Outputs in {outdir}", flush=True)


if __name__ == "__main__":
    main()
