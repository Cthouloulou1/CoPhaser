"""
Encoding of observed covariates (age, sex, condition, ...) into a numeric design matrix.

The encoding rule (``spec``) is learned once from the training data and stored, so
that any later AnnData (inference, posterior, a reloaded model) is encoded the same
way. The spec is a plain list of dicts and is saved in model checkpoints.

- continuous: standardised with the training mean / sd; missing -> 0 (the mean)
- categorical: one 0/1 column per non-reference level; missing -> all zeros
- any key with missing values in the training data gets an extra ``<key>__missing``
  0/1 column, so "missing" is not silently confounded with "mean" or "reference".
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

MISSING_SUFFIX = "__missing"


def _is_missing(values: pd.Series) -> np.ndarray:
    return values.isna().to_numpy()


def build_covariate_spec(
    obs: pd.DataFrame,
    keys,
    categorical=None,
    reference: dict | None = None,
) -> list[dict]:
    """Learn the encoding rule from ``obs``.

    keys: obs columns to use, in order.
    categorical: keys forced to categorical (non-numeric columns always are).
    reference: {key: level} reference level of a categorical key (default: first
        level in sorted order). Effects are then relative to that level.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    categorical = set([categorical] if isinstance(categorical, str) else (categorical or []))
    reference = reference or {}
    missing_keys = [k for k in keys if k not in obs.columns]
    if missing_keys:
        raise KeyError(f"Covariates not found in adata.obs: {missing_keys}")
    unknown_ref = set(reference) - set(keys)
    if unknown_ref:
        raise KeyError(f"reference given for keys that are not covariates: {sorted(unknown_ref)}")

    spec = []
    for key in keys:
        col = obs[key]
        has_missing = bool(_is_missing(col).any())
        is_cat = (
            key in categorical
            or not pd.api.types.is_numeric_dtype(col)
            or isinstance(col.dtype, pd.CategoricalDtype)
        )
        if is_cat:
            levels = sorted(pd.unique(col.dropna()).tolist(), key=str)
            if len(levels) < 2:
                raise ValueError(f"Categorical covariate {key!r} has fewer than 2 levels: {levels}")
            ref = reference.get(key, levels[0])
            if ref not in levels:
                raise ValueError(f"Reference level {ref!r} not among the levels of {key!r}: {levels}")
            spec.append(
                dict(
                    key=key,
                    type="categorical",
                    levels=[_to_builtin(v) for v in levels],
                    reference=_to_builtin(ref),
                    has_missing=has_missing,
                )
            )
        else:
            values = col.to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            sd = float(np.nanstd(values))
            if not np.isfinite(sd) or sd == 0:
                raise ValueError(f"Continuous covariate {key!r} is constant or empty.")
            spec.append(dict(key=key, type="continuous", mean=mean, sd=sd, has_missing=has_missing))
    return spec


def _to_builtin(v):
    """numpy scalars -> python scalars, so the spec pickles/JSONs cleanly."""
    return v.item() if hasattr(v, "item") else v


def covariate_column_names(spec: list[dict]) -> list[str]:
    names = []
    for s in spec:
        if s["type"] == "continuous":
            names.append(s["key"])
        else:
            names += [f"{s['key']}[{lvl}]" for lvl in s["levels"] if lvl != s["reference"]]
        if s["has_missing"]:
            names.append(s["key"] + MISSING_SUFFIX)
    return names


def encode_covariates(obs: pd.DataFrame, spec: list[dict]) -> np.ndarray:
    """Apply a spec to ``obs``. Returns a float32 array (n_obs, n_columns).

    Missing values in a key that had none at training time are encoded as the mean /
    reference (there is no indicator column for them) with a warning. Unknown
    categorical levels raise.
    """
    missing_keys = [s["key"] for s in spec if s["key"] not in obs.columns]
    if missing_keys:
        raise KeyError(f"Covariates required by the model are missing from adata.obs: {missing_keys}")
    blocks = []
    for s in spec:
        col = obs[s["key"]]
        miss = _is_missing(col)
        if miss.any() and not s["has_missing"]:
            warnings.warn(
                f"Covariate {s['key']!r} has {int(miss.sum())} missing values but had none during "
                "training: they are encoded as the training mean / reference level."
            )
        if s["type"] == "continuous":
            x = (col.to_numpy(dtype=float) - s["mean"]) / s["sd"]
            blocks.append(np.where(miss, 0.0, x)[:, None])
        else:
            present = col[~miss]
            unknown = sorted(set(present.tolist()) - set(s["levels"]), key=str)
            if unknown:
                raise ValueError(f"Unknown levels for covariate {s['key']!r}: {unknown} (known: {s['levels']})")
            values = col.to_numpy(dtype=object)
            for lvl in s["levels"]:
                if lvl == s["reference"]:
                    continue
                blocks.append(((values == lvl) & ~miss).astype(float)[:, None])
        if s["has_missing"]:
            blocks.append(miss.astype(float)[:, None])
    if not blocks:
        return np.zeros((len(obs), 0), dtype=np.float32)
    return np.hstack(blocks).astype(np.float32)
