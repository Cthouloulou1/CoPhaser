"""Hardy (death type) recoding options of scripts/gtex_train.py: merged levels and reference level."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cophaser.covariates import build_covariate_spec, covariate_column_names, encode_covariates

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gtex_train import covariate_reference, merge_levels, parse_args  # noqa: E402


def test_merge_levels_keeps_missing_and_other_levels():
    s = pd.Series([0, 1, 2, 3, np.nan, 2], dtype="object")
    out = merge_levels(s, ["1", "2"])
    assert list(out[:4]) == ["0", "1-2", "1-2", "3"]
    assert pd.isna(out[4]) and out[5] == "1-2"


def test_covariate_reference():
    assert covariate_reference(["age", "sex", "hardy"]) == {"sex": "male"}
    assert covariate_reference(["age", "sex", "hardy"], "1") == {"sex": "male", "hardy": "1"}
    assert covariate_reference(["age", "hardy"], "1-2") == {"hardy": "1-2"}
    assert covariate_reference(["age"], "1") is None


def test_merged_reference_encodes_as_expected():
    hardy = merge_levels(pd.Series(["0", "1", "2", "3", "4", "0", "2"], dtype="object"), ["1", "2"])
    obs = pd.DataFrame({"hardy": hardy.to_numpy()}, index=[f"S{i}" for i in range(len(hardy))])
    spec = build_covariate_spec(obs, ["hardy"], categorical=["hardy"], reference={"hardy": "1-2"})
    names = covariate_column_names(spec)
    assert names == ["hardy[0]", "hardy[3]", "hardy[4]"]
    X = np.asarray(encode_covariates(obs, spec))
    assert np.all(X[(hardy == "1-2").to_numpy()] == 0)
    assert X[0, names.index("hardy[0]")] == 1


def test_parse_args_hardy_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gtex_train.py", "--merge-hardy", "1", "2", "--hardy-reference", "1-2",
                                      "--no-lambda"])
    a = parse_args()
    assert a.merge_hardy == ["1", "2"] and a.hardy_reference == "1-2" and a.no_lambda
