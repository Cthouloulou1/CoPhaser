"""
Build one AnnData per tissue from public GTEx v8 files.

Inputs (in --data-dir, see the download commands in CLAUDE.md):
  gene_reads.gct.gz          GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz
  SampleAttributesDS.txt     GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
  SubjectPhenotypesDS.txt    GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt (public: SEX, AGE bin, DTHHRDY)
  chiral_sample_metadata.txt (optional) CHIRAL phases, naef-lab/CHIRAL Paper/paper_data/sample_metadata.txt

Output: <out-dir>/<tissue_slug>.h5ad with
  X / layers["counts"]  raw integer read counts, all genes (var_names = gene symbols, made unique)
  obs: donor, tissue, library_size (sum over ALL genes), age (midpoint of the public
       10-year bin), age_bin, sex ("male"/"female"), hardy (DTHHRDY as a string, NaN if
       missing), ischemic_time (SMTSISCH, minutes), rin (SMRIN), chiral_ct (hours, if available)
Only samples of the RNA-seq analysis freeze (SMAFRZE == "RNASEQ") are kept.

Example:
  python scripts/gtex_prepare.py --tissues "Muscle - Skeletal" "Lung"
  python scripts/gtex_prepare.py --top-tissues 8
"""
import argparse
import gzip
import re
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "gtex_v8")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <data-dir>/tissues")
    p.add_argument("--tissues", nargs="*", default=None, help="SMTSD names")
    p.add_argument("--top-tissues", type=int, default=None, help="the N tissues with most samples")
    return p.parse_args()


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_metadata(data_dir):
    samples = pd.read_csv(data_dir / "SampleAttributesDS.txt", sep="\t", low_memory=False)
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["donor"] = samples["SAMPID"].str.split("-").str[:2].str.join("-")
    subjects = pd.read_csv(data_dir / "SubjectPhenotypesDS.txt", sep="\t")
    subjects["age"] = subjects["AGE"].str.split("-").str[0].astype(float) + 4.5  # bin midpoint
    subjects["sex"] = subjects["SEX"].map({1: "male", 2: "female"})
    subjects["hardy"] = subjects["DTHHRDY"].map(lambda v: np.nan if pd.isna(v) else str(int(v)))
    meta = samples.merge(subjects.rename(columns={"SUBJID": "donor"}), on="donor", how="left")
    meta = meta.rename(columns={"SMTSD": "tissue", "AGE": "age_bin", "SMTSISCH": "ischemic_time", "SMRIN": "rin"})
    meta = meta.set_index("SAMPID")

    chiral = data_dir / "chiral_sample_metadata.txt"
    if chiral.exists():
        c = pd.read_csv(chiral, sep=" ")
        c.index = c["fullID"].str.replace(".", "-", regex=False)
        meta["chiral_ct"] = c["CT"].reindex(meta.index)
    keep = ["donor", "tissue", "age", "age_bin", "sex", "hardy", "ischemic_time", "rin"]
    keep += ["chiral_ct"] if "chiral_ct" in meta.columns else []
    return meta[keep]


def read_counts(gct_path, sample_ids):
    """Read only the wanted sample columns of the GCT (2 header lines, then Name, Description, samples)."""
    with gzip.open(gct_path, "rt") as fh:
        fh.readline()
        fh.readline()
        header = fh.readline().rstrip("\n").split("\t")
    wanted = [s for s in header[2:] if s in set(sample_ids)]
    t0 = time.time()
    df = pd.read_csv(gct_path, sep="\t", skiprows=2, usecols=["Name", "Description"] + wanted,
                     dtype={s: np.float64 for s in wanted})
    print(f"  read {df.shape[0]} genes x {len(wanted)} samples in {time.time() - t0:.0f} s")
    return df, wanted


def main():
    args = parse_args()
    out_dir = args.out_dir or args.data_dir / "tissues"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(args.data_dir)

    with gzip.open(args.data_dir / "gene_reads.gct.gz", "rt") as fh:
        fh.readline()
        fh.readline()
        in_counts = set(fh.readline().rstrip("\n").split("\t")[2:])
    meta = meta[meta.index.isin(in_counts)]
    counts_per_tissue = meta["tissue"].value_counts()

    tissues = args.tissues or []
    if args.top_tissues:
        tissues += [t for t in counts_per_tissue.index[: args.top_tissues] if t not in tissues]
    if not tissues:
        raise SystemExit("Give --tissues or --top-tissues. Available:\n" + counts_per_tissue.to_string())
    unknown = set(tissues) - set(counts_per_tissue.index)
    if unknown:
        raise SystemExit(f"Unknown tissues: {sorted(unknown)}")

    sample_ids = meta.index[meta["tissue"].isin(tissues)]
    print(f"{len(tissues)} tissues, {len(sample_ids)} samples; reading counts ...")
    df, wanted = read_counts(args.data_dir / "gene_reads.gct.gz", sample_ids)
    X_all = df[wanted].to_numpy()
    if not np.allclose(X_all, np.round(X_all)):
        raise ValueError("Counts are not integers.")
    X_all = np.round(X_all).astype(np.int32).T  # samples x genes
    var = pd.DataFrame({"gene_id": df["Name"].to_numpy()}, index=pd.Index(df["Description"].astype(str), name="symbol"))
    col = {s: i for i, s in enumerate(wanted)}

    for tissue in tissues:
        ids = [s for s in meta.index[meta["tissue"] == tissue] if s in col]
        X = X_all[[col[s] for s in ids]]
        obs = meta.loc[ids].copy()
        obs["library_size"] = X.sum(1).astype(float)
        ad = anndata.AnnData(X=csr_matrix(X), obs=obs, var=var.copy())
        ad.var_names_make_unique()
        ad.layers["counts"] = ad.X.copy()
        for c in ("donor", "tissue", "age_bin", "sex", "hardy"):
            ad.obs[c] = ad.obs[c].astype("category")
        path = out_dir / f"{slug(tissue)}.h5ad"
        ad.write_h5ad(path, compression="gzip")
        n_ct = int(obs["chiral_ct"].notna().sum()) if "chiral_ct" in obs else 0
        print(f"  {tissue}: {ad.n_obs} samples, {obs['donor'].nunique()} donors, "
              f"{n_ct} with CHIRAL phase, hardy missing {int(obs['hardy'].isna().sum())} -> {path.name}")


if __name__ == "__main__":
    main()
