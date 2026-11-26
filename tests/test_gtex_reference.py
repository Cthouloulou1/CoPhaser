"""GTEx scripts: the human clock of the tissue is the default reference (not the mouse example)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import gtex_amplitude_sweep  # noqa: E402
import gtex_bootstrap  # noqa: E402
import gtex_train  # noqa: E402

TISSUES = ["adipose_subcutaneous", "artery_tibial", "lung", "muscle_skeletal", "nerve_tibial",
           "skin_not_sun_exposed_suprapubic", "skin_sun_exposed_lower_leg", "thyroid"]


def test_human_clocks_are_in_the_repository():
    from cophaser.reference_clock import load_reference_clock

    for t in TISSUES:
        for d in [REPO / "reference" / "human_chiral", REPO / "reference" / "human_chiral" / "half"]:
            clock = load_reference_clock(d / f"{t}.csv", n_harm=1)
            assert {"ARNTL", "PER3", "DBP", "NR1D1"} <= set(clock.index)


def test_gtex_train_defaults_to_the_tissue_human_clock(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["gtex_train.py", "--tissues", "thyroid"])
    a = gtex_train.parse_args()
    assert a.reference is None
    assert gtex_train.reference_path(a, "thyroid") == REPO / "reference" / "human_chiral" / "thyroid.csv"
    a.reference = tmp_path / "my_clock.csv"
    assert gtex_train.reference_path(a, "thyroid") == tmp_path / "my_clock.csv"
    a.reference = None
    with pytest.raises(FileNotFoundError, match="gtex_human_clock"):
        gtex_train.reference_path(a, "liver")


def test_sweep_and_bootstrap_default_to_human(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gtex_bootstrap.py"])
    assert gtex_bootstrap.parse_args().clock == "human"
    monkeypatch.setattr(sys, "argv", ["gtex_amplitude_sweep.py"])
    assert gtex_amplitude_sweep.parse_args().clocks == ["human"]
    human_dir = REPO / "reference" / "human_chiral"
    assert gtex_amplitude_sweep.clock_args("human", "lung", {}, human_dir) == [
        "--reference", str(human_dir / "lung.csv"), "--freeze-decoder"]
    # the mouse clock is still available, but must now be passed explicitly
    mouse = gtex_amplitude_sweep.clock_args("mouse", "lung", {}, human_dir)
    assert mouse[:2] == ["--reference", str(REPO / "reference" / "clock_reference_learned.csv")]
