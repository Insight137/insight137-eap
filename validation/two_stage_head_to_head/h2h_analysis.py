"""Two-stage decision head-to-head: frozen quantum routine vs classical baselines.

Implements the analysis plan in the pre-registration
"Pre-registration - Two-Stage Decision Head-to-Head" (Insight137, drafted 2026-10-04).

Models (none has a fitted parameter; inputs are the two known-condition proportions):
    M_Q   insight137_eap.quantum_probability from insight137-eap==2.0.0, default arguments
    M_C   (p_true + p_false) / 2                      law of total probability, prior 0.5
    M_K   M_C - 0.25, clipped to [0.01, 0.99]         constant-offset rival
    M_O   point in [min(p_true, p_false), max(...)] closest to the observed value
          (oracle: the best any classical mixture could do; written M_C* in the plan)

Usage:
    python h2h_analysis.py --data studies.csv                 # development set only
    python h2h_analysis.py --data studies.csv --set heldout \
        --registration-id <OSF registration URL>              # only after registration

The held-out set cannot be analysed without --registration-id. That guard exists so the
model is never run on held-out studies before the plan is registered.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# --- Frozen constants (do not edit after registration) -----------------------
EXPECTED_VERSION = "2.0.0"
EXPECTED_MODULE_SHA256 = "c9db925a29998ca3e4a66b234bd1ffb9bc845a7f19e46c581397562e056b4b85"
SESOI = 0.03                  # smallest effect that matters, probability points
ALPHA_PRIMARY = 0.05          # H1: M_Q vs M_C
ALPHA_SECONDARY = 0.025       # H2 (vs M_K) and H3 (vs M_O), Bonferroni for two tests
MIN_CLUSTERS = 6
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEEDS = (1, 2, 3, 4, 5)
SEEDS_REQUIRED = 4
OFFSET_K = 0.25
CLIP_LO, CLIP_HI = 0.01, 0.99
EXACT_ENUMERATION_MAX_K = 22  # 2**22 sign patterns; above this use Monte Carlo
MONTE_CARLO_DRAWS = 1_000_000
MONTE_CARLO_SEED = 137


# --- Frozen model -------------------------------------------------------------
def load_frozen_model():
    """Import insight137_eap and refuse to continue unless it is the frozen 2.0.0 file."""
    import insight137_eap as eap

    version = getattr(eap, "__version__", None)
    digest = hashlib.sha256(Path(eap.__file__).read_bytes()).hexdigest()
    if version != EXPECTED_VERSION or digest != EXPECTED_MODULE_SHA256:
        raise SystemExit(
            "Frozen-model check failed.\n"
            f"  expected version {EXPECTED_VERSION}, sha256 {EXPECTED_MODULE_SHA256}\n"
            f"  found    version {version}, sha256 {digest}\n"
            f"  file: {eap.__file__}\n"
            "Install the frozen release in a clean environment: pip install insight137-eap==2.0.0\n"
            "(Run this script from outside the repo root so the working-tree copy is not imported.)"
        )
    return eap


def predict_quantum(eap, p_true: float, p_false: float) -> float:
    conditionals = {
        "target": {"p_given_a_true": p_true, "p_given_a_false": p_false},
        "other": {"p_given_a_true": 1.0 - p_true, "p_given_a_false": 1.0 - p_false},
    }
    return float(eap.quantum_probability(conditionals)["target"])


def predict_classical(p_true: float, p_false: float) -> float:
    return (p_true + p_false) / 2.0


def predict_offset(p_true: float, p_false: float) -> float:
    return min(max(predict_classical(p_true, p_false) - OFFSET_K, CLIP_LO), CLIP_HI)


def predict_oracle(p_true: float, p_false: float, observed: float) -> float:
    lo, hi = min(p_true, p_false), max(p_true, p_false)
    return min(max(observed, lo), hi)


# --- Data ---------------------------------------------------------------------
def load_triples(path: Path, which_set: str) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["set"] == which_set]
    if not rows:
        raise SystemExit(f"No rows with set == {which_set!r} in {path}")
    for row in rows:
        for key in ("p_true", "p_false", "p_unknown"):
            row[key] = float(row[key])
            if not 0.0 <= row[key] <= 1.0:
                raise SystemExit(f"{row['triple_id']}: {key} outside [0, 1]")
        row["n_unknown"] = int(row["n_unknown"]) if row["n_unknown"] else None
    return rows


def score_triples(eap, rows: Sequence[dict]) -> List[dict]:
    scored = []
    for row in rows:
        pt, pf, obs = row["p_true"], row["p_false"], row["p_unknown"]
        preds = {
            "Q": predict_quantum(eap, pt, pf),
            "C": predict_classical(pt, pf),
            "K": predict_offset(pt, pf),
            "O": predict_oracle(pt, pf, obs),
        }
        scored.append({**row, "pred": preds, "err": {m: abs(v - obs) for m, v in preds.items()}})
    return scored


def cluster_differences(scored: Sequence[dict], rival: str) -> "OrderedDict[str, float]":
    """d_j = mean over the cluster's triples of (rival error - M_Q error). Positive favours M_Q."""
    by_cluster: "OrderedDict[str, List[float]]" = OrderedDict()
    for row in scored:
        by_cluster.setdefault(row["cluster_id"], []).append(row["err"][rival] - row["err"]["Q"])
    return OrderedDict((cid, float(np.mean(vals))) for cid, vals in by_cluster.items())


# --- Statistics ---------------------------------------------------------------
def sign_flip_p_value(d: np.ndarray) -> Dict[str, object]:
    """One-sided sign-flip permutation test of mean(d) > 0.

    Exact enumeration of all 2**k sign patterns when k is small enough, otherwise a
    seeded Monte Carlo estimate with the +1 correction.
    """
    k = len(d)
    observed = float(d.mean())
    tolerance = 1e-12
    if k <= EXACT_ENUMERATION_MAX_K:
        count = 0
        total = 2 ** k
        chunk = 1 << 16
        for start in range(0, total, chunk):
            idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
            bits = (idx[:, None] >> np.arange(k)) & 1
            signs = 1.0 - 2.0 * bits
            count += int(((signs * d).mean(axis=1) >= observed - tolerance).sum())
        return {"p": count / total, "method": "exact", "patterns": total}
    rng = np.random.default_rng(MONTE_CARLO_SEED)
    signs = rng.choice((-1.0, 1.0), size=(MONTE_CARLO_DRAWS, k))
    count = int(((signs * d).mean(axis=1) >= observed - tolerance).sum())
    return {"p": (count + 1) / (MONTE_CARLO_DRAWS + 1), "method": "monte_carlo",
            "patterns": MONTE_CARLO_DRAWS}


def bootstrap_interval(d: np.ndarray, seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(BOOTSTRAP_RESAMPLES, len(d)))
    means = d[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def cohens_dz(d: np.ndarray) -> Optional[float]:
    if len(d) < 2:
        return None
    sd = float(d.std(ddof=1))
    return float(d.mean() / sd) if sd > 0 else None


def compare(scored: Sequence[dict], rival: str, alpha: float) -> dict:
    d_by_cluster = cluster_differences(scored, rival)
    d = np.array(list(d_by_cluster.values()), dtype=float)
    k = len(d)
    mean_d = float(d.mean())
    test = sign_flip_p_value(d)
    intervals = {seed: bootstrap_interval(d, seed) for seed in BOOTSTRAP_SEEDS}
    share_closer = float((d > 0).mean())
    seeds_lower_above_zero = sum(lo > 0 for lo, _ in intervals.values())
    seeds_upper_below_sesoi = sum(hi < SESOI for _, hi in intervals.values())

    if k < MIN_CLUSTERS:
        verdict = "INCONCLUSIVE"
        reason = f"only {k} clusters; the plan requires at least {MIN_CLUSTERS}"
    elif (test["p"] < alpha and mean_d >= SESOI and share_closer > 0.5
          and seeds_lower_above_zero >= SEEDS_REQUIRED):
        verdict, reason = "SUPPORTED", "all four conditions of the verdict rule hold"
    elif mean_d <= 0:
        verdict, reason = "REFUTED", "mean difference is not positive"
    elif seeds_upper_below_sesoi >= SEEDS_REQUIRED:
        verdict, reason = "REFUTED", f"upper end of the 95% interval is below {SESOI}"
    else:
        verdict, reason = "INCONCLUSIVE", "neither the Supported nor the Refuted rule is met"

    return {
        "rival": rival, "alpha": alpha, "clusters": k, "mean_d": mean_d,
        "cohens_dz": cohens_dz(d), "p_one_sided": test["p"], "p_method": test["method"],
        "share_clusters_Q_closer": share_closer,
        "bootstrap_95ci_by_seed": {str(s): ci for s, ci in intervals.items()},
        "seeds_lower_above_zero": seeds_lower_above_zero,
        "seeds_upper_below_sesoi": seeds_upper_below_sesoi,
        "verdict": verdict, "reason": reason,
        "d_by_cluster": dict(d_by_cluster),
    }


def paper_id(cluster_id: str) -> str:
    """Clusters are named <PAPER>-<sample>; development rows are one paper each."""
    return cluster_id if cluster_id.startswith("DEV-") else cluster_id.split("-")[0]


def paper_level_check(scored: Sequence[dict], rival: str) -> dict:
    """Robustness check: samples from one paper share a lab and a method, so they are
    not fully independent. Average the cluster differences within each paper and repeat
    the sign-flip test with papers as the unit."""
    by_paper: "OrderedDict[str, List[float]]" = OrderedDict()
    for cid, d in cluster_differences(scored, rival).items():
        by_paper.setdefault(paper_id(cid), []).append(d)
    d = np.array([float(np.mean(v)) for v in by_paper.values()], dtype=float)
    test = sign_flip_p_value(d)
    return {"papers": len(d), "mean_d": float(d.mean()), "p_one_sided": test["p"],
            "share_papers_Q_closer": float((d > 0).mean()),
            "d_by_paper": dict(zip(by_paper.keys(), d.tolist()))}


def binomial_log_likelihood(scored: Sequence[dict], model: str) -> Optional[float]:
    """Descriptive only. Uses rows whose unknown-condition n is known."""
    total, used = 0.0, 0
    for row in scored:
        n = row["n_unknown"]
        if not n:
            continue
        p = min(max(row["pred"][model], CLIP_LO), CLIP_HI)
        successes = row["p_unknown"] * n
        total += successes * math.log(p) + (n - successes) * math.log(1.0 - p)
        used += 1
    return total if used else None


def descriptives(scored: Sequence[dict]) -> dict:
    models = ("C", "Q", "K", "O")
    return {
        "triples": len(scored),
        "mean_abs_error": {m: float(np.mean([r["err"][m] for r in scored])) for m in models},
        "share_triples_Q_equals_C": float(np.mean(
            [abs(r["pred"]["Q"] - r["pred"]["C"]) < 1e-9 for r in scored])),
        "binomial_log_likelihood": {m: binomial_log_likelihood(scored, m) for m in ("C", "Q", "K")},
    }


def subset_reports(scored: Sequence[dict]) -> dict:
    """Pre-specified descriptive subsets. No claims are drawn from these."""
    def keep(predicate):
        rows = [r for r in scored if predicate(r)]
        clusters = {r["cluster_id"] for r in rows}
        if not rows:
            return None
        d = np.array(list(cluster_differences(rows, "C").values()))
        return {"clusters": len(clusters), "triples": len(rows), "mean_d_vs_C": float(d.mean())}

    return {
        "gamble_only": keep(lambda r: r["paradigm"] == "gamble"),
        "pd_only": keep(lambda r: r["paradigm"] == "PD"),
        "within_only": keep(lambda r: r["design"] == "within"),
        "between_only": keep(lambda r: r["design"] == "between"),
        "primary_source_only": keep(lambda r: r["source_type"] == "primary"),
        "published_2020_or_later": keep(lambda r: int(r["year"]) >= 2020),
    }


# --- Entry point --------------------------------------------------------------
def run(data: Path, which_set: str, registration_id: Optional[str]) -> dict:
    if which_set == "heldout" and not registration_id:
        raise SystemExit(
            "Refusing to analyse the held-out set without --registration-id.\n"
            "Register the plan first, then pass the registration URL."
        )
    eap = load_frozen_model()
    scored = score_triples(eap, load_triples(data, which_set))
    h1 = compare(scored, "C", ALPHA_PRIMARY)
    h1_papers = paper_level_check(scored, "C")
    if h1["verdict"] == "SUPPORTED" and h1_papers["mean_d"] <= 0:
        h1["verdict"] = "INCONCLUSIVE"
        h1["reason"] = "cluster-level rule met, but the paper-level mean difference is not positive"
    report = {
        "set": which_set,
        "registration_id": registration_id,
        "evidence_status": ("confirmatory" if which_set == "heldout"
                            else "development set: not evidence"),
        "model": {"package": "insight137-eap", "version": EXPECTED_VERSION,
                  "module_sha256": EXPECTED_MODULE_SHA256},
        "descriptives": descriptives(scored),
        "H1_vs_classical": h1,
        "H1_paper_level_check": h1_papers,
        "H2_vs_constant_offset": compare(scored, "K", ALPHA_SECONDARY),
        "H3_vs_best_classical_mixture": compare(scored, "O", ALPHA_SECONDARY),
        "descriptive_subsets": subset_reports(scored),
        "per_triple": [
            {"triple_id": r["triple_id"], "cluster_id": r["cluster_id"], "observed": r["p_unknown"],
             **{f"pred_{m}": r["pred"][m] for m in ("C", "Q", "K", "O")},
             **{f"err_{m}": r["err"][m] for m in ("C", "Q", "K", "O")}}
            for r in scored
        ],
    }
    return report


def print_summary(report: dict) -> None:
    print(f"Set: {report['set']}  ({report['evidence_status']})")
    desc = report["descriptives"]
    print(f"Triples: {desc['triples']}")
    print("Mean absolute error  " + "  ".join(
        f"M_{m}={v:.3f}" for m, v in desc["mean_abs_error"].items()))
    for key in ("H1_vs_classical", "H2_vs_constant_offset", "H3_vs_best_classical_mixture"):
        r = report[key]
        dz = "n/a" if r["cohens_dz"] is None else f"{r['cohens_dz']:.2f}"
        print(f"{key}: clusters={r['clusters']} mean_d={r['mean_d']:+.4f} dz={dz} "
              f"p={r['p_one_sided']:.4f} ({r['p_method']}) "
              f"Q_closer={r['share_clusters_Q_closer']:.0%} -> {r['verdict']} ({r['reason']})")
    pl = report["H1_paper_level_check"]
    print(f"H1 paper-level check: papers={pl['papers']} mean_d={pl['mean_d']:+.4f} "
          f"p={pl['p_one_sided']:.4f} Q_closer={pl['share_papers_Q_closer']:.0%}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("studies.csv"))
    parser.add_argument("--set", dest="which_set", choices=("development", "heldout"),
                        default="development")
    parser.add_argument("--registration-id", default=None,
                        help="URL of the registered plan; required for --set heldout")
    parser.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)

    report = run(args.data, args.which_set, args.registration_id)
    print_summary(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Full report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
