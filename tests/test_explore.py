from __future__ import annotations

from pathlib import Path

from betatrend.nn.explore import (
    latest_fold_ckpt,
    parse_histogram_field,
    parse_train_log,
    summarize_explorer_hists,
)


def test_parse_train_log_extracts_fold_and_mean_r():
    text = """
2026-08-24 16:52:15 | INFO | walk-forward folds=110 warmup=360 min_train=2160 test_h=504 seeds=3 epochs=80
2026-08-24 16:52:15 | INFO | fold 1/110 train_idx=[360, 2496) n_train=2136 n_test=504 warm_start=False
2026-08-24 16:52:17 | INFO | GRPO seed=7 epoch 1/80 mean_r=-0.0792
2026-08-24 16:52:30 | INFO | GRPO seed=7 epoch 20/80 mean_r=0.3662
2026-08-24 16:53:17 | INFO | saved checkpoint /tmp/eth_decision_fold0_s7_e80.pt
"""
    snap = parse_train_log(text)
    assert snap["n_folds"] == 110
    assert snap["fold"] == 1
    assert snap["n_train"] == 2136
    assert snap["epoch"] == 20
    assert snap["mean_r"] == 0.3662
    assert snap["points"][0]["seed"] == 7
    assert snap["last_ckpt"].endswith("eth_decision_fold0_s7_e80.pt")
    assert snap["status"] == "running"


def test_parse_train_log_flags_nan_crash():
    text = "ValueError: Expected parameter loc to satisfy the constraint Real()"
    assert parse_train_log(text)["status"] == "error"


def test_latest_fold_ckpt_picks_newest(tmp_path: Path):
    (tmp_path / "eth_decision_fold0_s7_e80.pt").write_bytes(b"a")
    later = tmp_path / "eth_decision_fold2_s41_e80.pt"
    later.write_bytes(b"b")
    assert latest_fold_ckpt(tmp_path) == later


def test_parse_histogram_field_collapses_time_bins():
    blob = (
        "input::raw val!!-1::1!!-1.0::1.0!!step!!0::1!!0::1"
        "!!1::2::3;4::5::6"
    )
    hists = parse_histogram_field(blob, "input")
    assert len(hists) == 1
    assert hists[0]["min"] == -1.0
    assert hists[0]["max"] == 1.0
    assert hists[0]["bins"] == [5.0, 7.0, 9.0]


def test_summarize_explorer_hists_keeps_depth1_modules():
    rows = [
        {
            "type": "nodes",
            "nodes:id": 0,
            "nodes:display_name": "GRPOActor",
            "nodes:parent_stack": "GRPOActor::0",
            "nodes:input_histograms": "",
            "nodes:output_histograms": "",
            "nodes:param_histograms": "",
        },
        {
            "type": "nodes",
            "nodes:id": 2,
            "nodes:display_name": "LSTM",
            "nodes:parent_stack": "GRPOActor::0;LSTM::2",
            "nodes:input_histograms": (
                "input::raw val!!-1::1!!-1.0::1.0!!step!!0!!0!!1::2::3"
            ),
            "nodes:output_histograms": "",
            "nodes:param_histograms": "",
        },
        {
            "type": "nodes",
            "nodes:id": 3,
            "nodes:display_name": "Input",
            "nodes:parent_stack": "GRPOActor::0;LSTM::2;Input::3",
            "nodes:input_histograms": (
                "input::raw val!!-1::1!!-1.0::1.0!!step!!0!!0!!9::9::9"
            ),
            "nodes:output_histograms": "",
            "nodes:param_histograms": "",
        },
    ]
    mods = summarize_explorer_hists(rows)["modules"]
    assert [m["id"] for m in mods] == [2]
    assert mods[0]["hists"][0]["bins"] == [1.0, 2.0, 3.0]
