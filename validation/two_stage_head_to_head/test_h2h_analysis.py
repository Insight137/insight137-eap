"""Tests for h2h_analysis.py.

Model checks use the DEVELOPMENT studies only. Statistics checks use made-up numbers
whose only purpose is to exercise the code paths; they are not data.
"""
from pathlib import Path

import numpy as np
import pytest

import h2h_analysis as h

DATA = Path(__file__).with_name("studies.csv")


def synthetic(d_values, triples_per_cluster=1):
    """Rows with a fixed (rival error - M_Q error) per cluster. Not empirical data."""
    rows = []
    for j, d in enumerate(d_values):
        for _ in range(triples_per_cluster):
            rows.append({"cluster_id": f"c{j}", "err": {"Q": 0.10, "C": 0.10 + d,
                                                        "K": 0.10 + d, "O": 0.10 + d}})
    return rows


def test_development_set_reproduces_registered_table():
    report = h.run(DATA, "development", None)
    mae = report["descriptives"]["mean_abs_error"]
    assert round(mae["C"], 3) == 0.166
    assert round(mae["Q"], 3) == 0.105
    assert round(mae["K"], 3) == 0.094
    assert round(mae["O"], 3) == 0.102
    by_id = {row["triple_id"]: row for row in report["per_triple"]}
    expected_q = {"D1": 0.518, "D2": 0.495, "D3": 0.579, "D4": 0.607, "D5": 0.905}
    for triple_id, value in expected_q.items():
        assert round(by_id[triple_id]["pred_Q"], 3) == value
    assert report["evidence_status"] == "development set: not evidence"
    assert report["H1_vs_classical"]["verdict"] == "INCONCLUSIVE"  # 5 clusters < 6


def test_heldout_is_refused_without_registration():
    with pytest.raises(SystemExit, match="Refusing to analyse the held-out set"):
        h.run(DATA, "heldout", None)


def test_frozen_model_hash_is_enforced(monkeypatch):
    monkeypatch.setattr(h, "EXPECTED_MODULE_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="Frozen-model check failed"):
        h.load_frozen_model()


def test_sign_flip_exact_values():
    assert h.sign_flip_p_value(np.array([1.0, 1.0, 1.0]))["p"] == pytest.approx(1 / 8)
    assert h.sign_flip_p_value(np.array([1.0, -1.0]))["p"] == pytest.approx(3 / 4)
    six = h.sign_flip_p_value(np.full(6, 0.2))
    assert six["p"] == pytest.approx(1 / 64) and six["method"] == "exact"


def test_baseline_formulas():
    assert h.predict_classical(0.8, 0.6) == pytest.approx(0.7)
    assert h.predict_offset(0.8, 0.6) == pytest.approx(0.45)
    assert h.predict_offset(0.2, 0.1) == pytest.approx(0.01)       # clipped
    assert h.predict_oracle(0.8, 0.6, 0.9) == pytest.approx(0.8)   # above the interval
    assert h.predict_oracle(0.8, 0.6, 0.7) == pytest.approx(0.7)   # inside the interval
    assert h.predict_oracle(0.6, 0.8, 0.3) == pytest.approx(0.6)   # below, inputs swapped


def test_cluster_mean_uses_all_triples_in_a_cluster():
    rows = [
        {"cluster_id": "a", "err": {"Q": 0.1, "C": 0.3}},
        {"cluster_id": "a", "err": {"Q": 0.1, "C": 0.1}},
        {"cluster_id": "b", "err": {"Q": 0.2, "C": 0.1}},
    ]
    d = h.cluster_differences(rows, "C")
    assert d["a"] == pytest.approx(0.1) and d["b"] == pytest.approx(-0.1)


def test_verdict_supported():
    result = h.compare(synthetic([0.08, 0.10, 0.12, 0.09, 0.11, 0.10, 0.07, 0.13]), "C", 0.05)
    assert result["verdict"] == "SUPPORTED"


def test_verdict_refuted_when_classical_is_better():
    result = h.compare(synthetic([-0.05] * 8), "C", 0.05)
    assert result["verdict"] == "REFUTED" and "not positive" in result["reason"]


def test_verdict_refuted_when_effect_is_too_small():
    result = h.compare(synthetic([0.004, 0.006, 0.005, 0.005, 0.004, 0.006, 0.005, 0.005]), "C", 0.05)
    assert result["verdict"] == "REFUTED" and "below" in result["reason"]


def test_verdict_inconclusive_when_noisy():
    result = h.compare(synthetic([0.30, -0.20, 0.25, -0.15, 0.20, -0.10, 0.10, -0.05]), "C", 0.05)
    assert result["verdict"] == "INCONCLUSIVE"


def test_verdict_inconclusive_below_minimum_clusters():
    result = h.compare(synthetic([0.2] * 5), "C", 0.05)
    assert result["verdict"] == "INCONCLUSIVE" and "at least 6" in result["reason"]


def test_paper_id_groups_samples_from_one_paper():
    assert h.paper_id("BBP2020-E2KU") == "BBP2020"
    assert h.paper_id("XIN2026") == "XIN2026"
    assert h.paper_id("DEV-D1") == "DEV-D1"


def test_paper_level_check_averages_clusters_within_a_paper():
    rows = [
        {"cluster_id": "AAA-1", "err": {"Q": 0.1, "C": 0.3}},
        {"cluster_id": "AAA-2", "err": {"Q": 0.1, "C": 0.1}},
        {"cluster_id": "BBB", "err": {"Q": 0.2, "C": 0.1}},
    ]
    check = h.paper_level_check(rows, "C")
    assert check["papers"] == 2
    assert check["d_by_paper"]["AAA"] == pytest.approx(0.1)
    assert check["d_by_paper"]["BBB"] == pytest.approx(-0.1)
    assert check["mean_d"] == pytest.approx(0.0)
