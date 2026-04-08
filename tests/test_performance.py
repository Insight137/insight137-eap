"""
Performance Benchmark Suite for insight137_eap
===============================================

Comprehensive benchmarks covering:
  1. Single-call latency (microseconds per call)
  2. Throughput (calls per second)
  3. Scaling behavior (time vs input size, complexity class)
  4. Memory usage (footprint and leak detection)
  5. Numpy optimization check (cProfile hotspots)

Uses timeit for latency, time.perf_counter for throughput,
tracemalloc for memory, and cProfile for hotspot analysis.

Run with: python -m pytest tests/test_performance.py -v -s
"""

import cProfile
import gc
import io
import json
import os
import pstats
import sys
import time
import timeit
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from tabulate import tabulate

import insight137_eap as eap

# ═══════════════════════════════════════════════════════════════════════
# FIXTURES AND HELPERS
# ═══════════════════════════════════════════════════════════════════════

RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"
ALL_RESULTS = {}

RNG = np.random.default_rng(137)

# Pre-generate test data at various sizes
DATA = {}
for sz in [10, 50, 100, 500, 1_000, 5_000, 10_000]:
    DATA[sz] = RNG.uniform(10.0, 500.0, size=sz)

AGENTS_11 = list(RNG.uniform(0.01, 0.99, size=11))
AGENTS_100 = list(RNG.uniform(0.01, 0.99, size=100))

CONDS_2 = {
    "defect": {"p_given_a_true": 0.87, "p_given_a_false": 0.74},
    "cooperate": {"p_given_a_true": 0.13, "p_given_a_false": 0.26},
}

CONDS_10 = {}
for i in range(10):
    p = RNG.uniform(0.05, 0.95)
    CONDS_10[f"outcome_{i}"] = {
        "p_given_a_true": float(p),
        "p_given_a_false": float(RNG.uniform(0.05, 0.95)),
    }

OUTCOMES_2 = [
    {"p_given_a_true": 0.87, "p_given_a_false": 0.74},
    {"p_given_a_true": 0.13, "p_given_a_false": 0.26},
]

MIN_ITERATIONS = 100


def _benchmark_fn(fn, n_iter=None):
    """Benchmark a callable. Returns dict with mean, std, min, max in microseconds."""
    # Warm up
    for _ in range(5):
        fn()

    # Auto-scale iterations: aim for at least 0.5s total
    if n_iter is None:
        single = timeit.timeit(fn, number=1)
        n_iter = max(MIN_ITERATIONS, int(0.5 / max(single, 1e-9)))
        n_iter = min(n_iter, 50_000)

    # Collect individual timings
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    arr = np.array(times)
    return {
        "mean_us": float(np.mean(arr)),
        "std_us": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "n_iter": n_iter,
        "calls_per_sec": 1e6 / float(np.mean(arr)) if np.mean(arr) > 0 else float("inf"),
    }


def _save_results():
    """Save accumulated results to JSON."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(ALL_RESULTS, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# 1. SINGLE-CALL LATENCY
# ═══════════════════════════════════════════════════════════════════════

class TestSingleCallLatency:
    """Measure microseconds per call for core operations."""

    def test_psi_from_sequence_10(self):
        """Latency of compute_psi_from_sequence with 10 values."""
        r = _benchmark_fn(lambda: eap.compute_psi_from_sequence(DATA[10]))
        ALL_RESULTS["latency_psi_seq_10"] = r
        assert r["mean_us"] < 50_000  # sanity: under 50ms

    def test_psi_from_sequence_100(self):
        """Latency of compute_psi_from_sequence with 100 values."""
        r = _benchmark_fn(lambda: eap.compute_psi_from_sequence(DATA[100]))
        ALL_RESULTS["latency_psi_seq_100"] = r
        assert r["mean_us"] < 50_000

    def test_psi_from_sequence_1000(self):
        """Latency of compute_psi_from_sequence with 1,000 values."""
        r = _benchmark_fn(lambda: eap.compute_psi_from_sequence(DATA[1_000]))
        ALL_RESULTS["latency_psi_seq_1000"] = r
        assert r["mean_us"] < 100_000

    def test_psi_from_sequence_10000(self):
        """Latency of compute_psi_from_sequence with 10,000 values."""
        r = _benchmark_fn(lambda: eap.compute_psi_from_sequence(DATA[10_000]))
        ALL_RESULTS["latency_psi_seq_10000"] = r
        assert r["mean_us"] < 1_000_000  # under 1s

    def test_psi_from_conditionals_2(self):
        """Latency of compute_psi_from_conditionals with 2 outcomes."""
        r = _benchmark_fn(lambda: eap.compute_psi_from_conditionals(CONDS_2))
        ALL_RESULTS["latency_psi_cond_2"] = r
        assert r["mean_us"] < 10_000

    def test_psi_from_conditionals_10(self):
        """Latency of compute_psi_from_conditionals with 10 outcomes."""
        r = _benchmark_fn(lambda: eap.compute_psi_from_conditionals(CONDS_10))
        ALL_RESULTS["latency_psi_cond_10"] = r
        assert r["mean_us"] < 10_000

    def test_psi4_11_agents(self):
        """Latency of compute_psi4 with 11 agents."""
        r = _benchmark_fn(lambda: eap.compute_psi4(AGENTS_11))
        ALL_RESULTS["latency_psi4_11"] = r
        assert r["mean_us"] < 10_000

    def test_psi4_100_agents(self):
        """Latency of compute_psi4 with 100 agents."""
        r = _benchmark_fn(lambda: eap.compute_psi4(AGENTS_100))
        ALL_RESULTS["latency_psi4_100"] = r
        assert r["mean_us"] < 50_000

    def test_verify_huang_paper(self):
        """Latency of verify_huang_paper() full suite."""
        r = _benchmark_fn(lambda: eap.verify_huang_paper())
        ALL_RESULTS["latency_verify_huang"] = r
        assert r["mean_us"] < 50_000


# ═══════════════════════════════════════════════════════════════════════
# 2. THROUGHPUT
# ═══════════════════════════════════════════════════════════════════════

class TestThroughput:
    """Measure calls per second for production workloads.
    Target: >1000 calls/sec for real-time use."""

    def test_throughput_psi_sequence_100(self):
        """Throughput of compute_psi_from_sequence(100 values)."""
        r = _benchmark_fn(lambda: eap.compute_psi_from_sequence(DATA[100]))
        ALL_RESULTS["throughput_psi_seq_100"] = r
        assert r["calls_per_sec"] > 1000, (
            f"Below production target: {r['calls_per_sec']:.0f} calls/sec < 1000"
        )

    def test_throughput_quantum_probability(self):
        """Throughput of quantum_probability()."""
        r = _benchmark_fn(lambda: eap.quantum_probability(CONDS_2))
        ALL_RESULTS["throughput_quantum_prob"] = r
        assert r["calls_per_sec"] > 1000, (
            f"Below production target: {r['calls_per_sec']:.0f} calls/sec < 1000"
        )

    def test_throughput_belief_degree(self):
        """Throughput of belief_degree_huang()."""
        r = _benchmark_fn(lambda: eap.belief_degree_huang(OUTCOMES_2))
        ALL_RESULTS["throughput_belief_degree"] = r
        assert r["calls_per_sec"] > 1000, (
            f"Below production target: {r['calls_per_sec']:.0f} calls/sec < 1000"
        )


# ═══════════════════════════════════════════════════════════════════════
# 3. SCALING BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════

class TestScalingBehavior:
    """Analyze how compute_psi_from_sequence scales with input size."""

    def test_scaling_curve(self):
        """Measure time vs input length and determine complexity class."""
        sizes = [10, 50, 100, 500, 1_000, 5_000, 10_000]
        scaling_data = {}

        for sz in sizes:
            r = _benchmark_fn(lambda s=sz: eap.compute_psi_from_sequence(DATA[s]))
            scaling_data[sz] = r
            ALL_RESULTS[f"scaling_{sz}"] = r

        # Determine complexity by fitting log-log slope
        log_sizes = np.log10([float(s) for s in sizes])
        log_times = np.log10([scaling_data[s]["mean_us"] for s in sizes])
        slope, intercept = np.polyfit(log_sizes, log_times, 1)

        complexity = "O(n)" if slope < 1.5 else "O(n^2)" if slope < 2.5 else f"O(n^{slope:.1f})"
        ALL_RESULTS["scaling_slope"] = float(slope)
        ALL_RESULTS["scaling_complexity"] = complexity

        # Print scaling table
        table = []
        for sz in sizes:
            d = scaling_data[sz]
            table.append([
                f"{sz:,}",
                f"{d['mean_us']:.1f}",
                f"{d['std_us']:.1f}",
                f"{d['min_us']:.1f}",
                f"{d['max_us']:.1f}",
                f"{d['calls_per_sec']:,.0f}",
            ])
        print("\n" + tabulate(
            table,
            headers=["Input Size", "Mean (µs)", "Std (µs)", "Min (µs)", "Max (µs)", "Calls/sec"],
            tablefmt="github",
        ))
        print(f"\nLog-log slope: {slope:.2f} -> Complexity: {complexity}")

    def test_find_100ms_threshold(self):
        """Find the input size where a single call exceeds 100ms."""
        threshold_us = 100_000
        sizes_to_test = [10_000, 20_000, 50_000, 100_000]
        threshold_size = None

        for sz in sizes_to_test:
            data = RNG.uniform(10.0, 500.0, size=sz)
            t0 = time.perf_counter()
            eap.compute_psi_from_sequence(data)
            elapsed_us = (time.perf_counter() - t0) * 1e6

            if elapsed_us > threshold_us:
                threshold_size = sz
                break

        if threshold_size:
            ALL_RESULTS["threshold_100ms"] = threshold_size
            print(f"\n100ms threshold exceeded at input size: {threshold_size:,}")
        else:
            ALL_RESULTS["threshold_100ms"] = f">{sizes_to_test[-1]:,}"
            print(f"\n100ms threshold NOT exceeded up to input size: {sizes_to_test[-1]:,}")

    def test_find_1s_threshold(self):
        """Find the input size where a single call exceeds 1 second."""
        threshold_us = 1_000_000
        sizes_to_test = [50_000, 100_000, 200_000, 500_000]
        threshold_size = None

        for sz in sizes_to_test:
            data = RNG.uniform(10.0, 500.0, size=sz)
            t0 = time.perf_counter()
            eap.compute_psi_from_sequence(data)
            elapsed_us = (time.perf_counter() - t0) * 1e6

            if elapsed_us > threshold_us:
                threshold_size = sz
                break

        if threshold_size:
            ALL_RESULTS["threshold_1s"] = threshold_size
            print(f"\n1s threshold exceeded at input size: {threshold_size:,}")
        else:
            ALL_RESULTS["threshold_1s"] = f">{sizes_to_test[-1]:,}"
            print(f"\n1s threshold NOT exceeded up to input size: {sizes_to_test[-1]:,}")


# ═══════════════════════════════════════════════════════════════════════
# 4. MEMORY USAGE
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryUsage:
    """Measure memory footprint and check for leaks."""

    def test_memory_10k_sequence(self):
        """Memory footprint of processing a 10K-element sequence."""
        gc.collect()
        tracemalloc.start()

        data = RNG.uniform(10.0, 500.0, size=10_000)
        eap.compute_psi_from_sequence(data)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = {
            "current_kb": current / 1024,
            "peak_kb": peak / 1024,
        }
        ALL_RESULTS["memory_10k_seq"] = result
        print(f"\n10K sequence: current={current/1024:.1f} KB, peak={peak/1024:.1f} KB")
        assert peak < 50 * 1024 * 1024  # under 50 MB

    def test_memory_1000_sequences(self):
        """Memory footprint of processing 1000 sequences of 100 elements."""
        gc.collect()
        tracemalloc.start()

        for _ in range(1000):
            data = RNG.uniform(10.0, 500.0, size=100)
            eap.compute_psi_from_sequence(data)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = {
            "current_kb": current / 1024,
            "peak_kb": peak / 1024,
        }
        ALL_RESULTS["memory_1000x100_seq"] = result
        print(f"\n1000x100 sequences: current={current/1024:.1f} KB, peak={peak/1024:.1f} KB")
        assert peak < 50 * 1024 * 1024

    def test_memory_leak_check(self):
        """Process 10K sequences and verify memory returns to baseline.
        A leak would show monotonically increasing memory."""
        gc.collect()
        tracemalloc.start()
        snapshot_baseline = tracemalloc.take_snapshot()

        # Process 10K sequences
        for _ in range(10_000):
            data = RNG.uniform(10.0, 500.0, size=100)
            eap.compute_psi_from_sequence(data)

        gc.collect()
        snapshot_after = tracemalloc.take_snapshot()

        # Compare top allocations
        stats = snapshot_after.compare_to(snapshot_baseline, "lineno")
        # Sum of size differences for top allocations
        total_diff_kb = sum(s.size_diff for s in stats[:20]) / 1024

        tracemalloc.stop()

        result = {
            "total_diff_kb": total_diff_kb,
            "top_allocations": [
                {"file": str(s.traceback), "size_diff_kb": s.size_diff / 1024}
                for s in stats[:5]
            ],
        }
        ALL_RESULTS["memory_leak_check"] = result
        print(f"\nMemory diff after 10K sequences: {total_diff_kb:.1f} KB")

        # Allow some variance but flag large leaks (>10 MB)
        assert total_diff_kb < 10_000, (
            f"Potential memory leak: {total_diff_kb:.1f} KB retained after 10K sequences"
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. NUMPY OPTIMIZATION CHECK
# ═══════════════════════════════════════════════════════════════════════

class TestNumpyOptimizationCheck:
    """Profile hotspots and verify vectorization effectiveness."""

    def test_vectorized_vs_loop_sliding_window(self):
        """Compare current vectorized _compute_sliding_interferences
        against a naive loop-based reimplementation."""
        data = DATA[500]

        # Current vectorized implementation
        r_vec = _benchmark_fn(
            lambda: eap._compute_sliding_interferences(data, 3),
            n_iter=500,
        )

        # Naive loop-based reimplementation for comparison
        def _loop_sliding(values, window_size=3):
            n = len(values)
            ew = min(window_size, n)
            if ew < 2:
                return []
            results = []
            for start in range(n - ew + 1):
                window = values[start:start + ew]
                ws = window.sum()
                if ws <= 1e-15:
                    continue
                norm = window / ws
                p1 = max(0.01, min(float(norm[0]), 0.99))
                p2 = max(0.01, min(float(norm[1]), 0.99))
                outcomes = [
                    {"p_given_a_true": p1, "p_given_a_false": p2},
                    {"p_given_a_true": 1.0 - p1, "p_given_a_false": 1.0 - p2},
                ]
                db = eap.belief_degree_huang(outcomes)
                results.append(db)
            return results

        r_loop = _benchmark_fn(
            lambda: _loop_sliding(data, 3),
            n_iter=200,
        )

        speedup = r_loop["mean_us"] / r_vec["mean_us"] if r_vec["mean_us"] > 0 else float("inf")

        result = {
            "vectorized_mean_us": r_vec["mean_us"],
            "loop_mean_us": r_loop["mean_us"],
            "speedup": speedup,
        }
        ALL_RESULTS["vec_vs_loop"] = result

        print(f"\nSliding window (500 elements):")
        print(f"  Vectorized: {r_vec['mean_us']:.1f} µs")
        print(f"  Loop-based: {r_loop['mean_us']:.1f} µs")
        print(f"  Speedup:    {speedup:.1f}x")

        assert speedup > 1.0, "Vectorized path should be faster than loop"

    def test_cprofile_hotspots(self):
        """Identify top 3 hotspots using cProfile on the main entry point."""
        data = DATA[100]

        pr = cProfile.Profile()
        pr.enable()
        for _ in range(1000):
            eap.compute_psi_from_sequence(data, agent_decisions=AGENTS_11)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats(15)
        profile_output = s.getvalue()

        # Extract top functions by tottime
        lines = profile_output.strip().split("\n")
        hotspots = []
        for line in lines:
            if "insight137_eap.py" in line or "_methods.py" in line:
                parts = line.strip().split()
                if len(parts) >= 6:
                    hotspots.append({
                        "ncalls": parts[0],
                        "tottime": parts[1],
                        "function": parts[-1],
                    })
        hotspots = hotspots[:3]

        result = {
            "top_3_hotspots": hotspots,
            "full_profile": profile_output[:2000],
        }
        ALL_RESULTS["cprofile_hotspots"] = result

        print(f"\nTop 3 hotspots (compute_psi_from_sequence x 1000, 100 elements):")
        for i, h in enumerate(hotspots, 1):
            print(f"  {i}. {h['function']} — {h['tottime']}s total ({h['ncalls']} calls)")

    def test_optimization_suggestions(self):
        """Generate specific optimization suggestions based on profiling."""
        data = DATA[1_000]

        # Profile the 1K-element case
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(100):
            eap.compute_psi_from_sequence(data, agent_decisions=AGENTS_11)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(10)
        profile_text = s.getvalue()

        # Analyze and suggest
        suggestions = []

        # Check if _belief_distance_vec dominates
        if "_belief_distance_vec" in profile_text:
            suggestions.append(
                "1. _belief_distance_vec: Already vectorized. Further gains possible "
                "via Cython/Numba JIT compilation of the inner np.where logic."
            )

        # Check if validation overhead is significant
        if "_validate_probability" in profile_text:
            suggestions.append(
                "2. _validate_probability: Called per-agent in psi4. For trusted "
                "internal calls, consider a fast-path that skips validation."
            )

        # Check if numpy overhead functions appear
        if "std" in profile_text or "mean" in profile_text:
            suggestions.append(
                "3. np.std/np.mean: numpy dispatch overhead. For small arrays, "
                "manual computation (sum/len) can be 2-5x faster."
            )

        if not suggestions:
            suggestions.append("No significant hotspots detected — already well optimized.")

        ALL_RESULTS["optimization_suggestions"] = suggestions

        print("\nOptimization suggestions:")
        for s_item in suggestions:
            print(f"  {s_item}")


# ═══════════════════════════════════════════════════════════════════════
# REPORT GENERATION (runs after all tests)
# ═══════════════════════════════════════════════════════════════════════

class TestReportGeneration:
    """Generate final summary report and save results. Runs last."""

    def test_zz_final_report(self):
        """Print final summary table and save all results to JSON."""
        # Latency table
        latency_rows = []
        latency_keys = [
            ("latency_psi_seq_10", "psi_from_sequence(10)"),
            ("latency_psi_seq_100", "psi_from_sequence(100)"),
            ("latency_psi_seq_1000", "psi_from_sequence(1K)"),
            ("latency_psi_seq_10000", "psi_from_sequence(10K)"),
            ("latency_psi_cond_2", "psi_from_conditionals(2)"),
            ("latency_psi_cond_10", "psi_from_conditionals(10)"),
            ("latency_psi4_11", "compute_psi4(11)"),
            ("latency_psi4_100", "compute_psi4(100)"),
            ("latency_verify_huang", "verify_huang_paper()"),
        ]
        for key, label in latency_keys:
            if key in ALL_RESULTS:
                d = ALL_RESULTS[key]
                latency_rows.append([
                    label,
                    f"{d['mean_us']:.1f}",
                    f"{d['std_us']:.1f}",
                    f"{d['min_us']:.1f}",
                    f"{d['max_us']:.1f}",
                    f"{d['calls_per_sec']:,.0f}",
                ])

        print("\n" + "=" * 80)
        print("SINGLE-CALL LATENCY")
        print("=" * 80)
        print(tabulate(
            latency_rows,
            headers=["Operation", "Mean (µs)", "Std (µs)", "Min (µs)", "Max (µs)", "Calls/sec"],
            tablefmt="github",
        ))

        # Throughput table
        throughput_rows = []
        throughput_keys = [
            ("throughput_psi_seq_100", "psi_from_sequence(100)"),
            ("throughput_quantum_prob", "quantum_probability()"),
            ("throughput_belief_degree", "belief_degree_huang()"),
        ]
        for key, label in throughput_keys:
            if key in ALL_RESULTS:
                d = ALL_RESULTS[key]
                cps = d["calls_per_sec"]
                status = "PASS" if cps > 1000 else "FAIL"
                throughput_rows.append([
                    label,
                    f"{cps:,.0f}",
                    f"{d['mean_us']:.1f}",
                    status,
                ])

        print("\n" + "=" * 80)
        print("THROUGHPUT (target: >1,000 calls/sec)")
        print("=" * 80)
        print(tabulate(
            throughput_rows,
            headers=["Operation", "Calls/sec", "Mean (µs)", "Status"],
            tablefmt="github",
        ))

        # Scaling table
        scaling_rows = []
        for sz in [10, 50, 100, 500, 1_000, 5_000, 10_000]:
            key = f"scaling_{sz}"
            if key in ALL_RESULTS:
                d = ALL_RESULTS[key]
                scaling_rows.append([
                    f"{sz:,}",
                    f"{d['mean_us']:.1f}",
                    f"{d['calls_per_sec']:,.0f}",
                ])
        if scaling_rows:
            print("\n" + "=" * 80)
            print("SCALING BEHAVIOR")
            print("=" * 80)
            print(tabulate(
                scaling_rows,
                headers=["Input Size", "Mean (µs)", "Calls/sec"],
                tablefmt="github",
            ))
            if "scaling_complexity" in ALL_RESULTS:
                print(f"\nComplexity: {ALL_RESULTS['scaling_complexity']} "
                      f"(slope={ALL_RESULTS.get('scaling_slope', '?'):.2f})")

        # Thresholds
        if "threshold_100ms" in ALL_RESULTS:
            print(f"100ms threshold: {ALL_RESULTS['threshold_100ms']}")
        if "threshold_1s" in ALL_RESULTS:
            print(f"1s threshold: {ALL_RESULTS['threshold_1s']}")

        # Memory table
        memory_rows = []
        if "memory_10k_seq" in ALL_RESULTS:
            d = ALL_RESULTS["memory_10k_seq"]
            memory_rows.append(["10K sequence", f"{d['peak_kb']:.1f}"])
        if "memory_1000x100_seq" in ALL_RESULTS:
            d = ALL_RESULTS["memory_1000x100_seq"]
            memory_rows.append(["1000x100 sequences", f"{d['peak_kb']:.1f}"])
        if "memory_leak_check" in ALL_RESULTS:
            d = ALL_RESULTS["memory_leak_check"]
            memory_rows.append(["Leak check (10K seqs)", f"{d['total_diff_kb']:.1f} diff"])

        if memory_rows:
            print("\n" + "=" * 80)
            print("MEMORY USAGE")
            print("=" * 80)
            print(tabulate(
                memory_rows,
                headers=["Test", "Peak / Diff (KB)"],
                tablefmt="github",
            ))

        # Vectorization
        if "vec_vs_loop" in ALL_RESULTS:
            d = ALL_RESULTS["vec_vs_loop"]
            print("\n" + "=" * 80)
            print("VECTORIZATION COMPARISON (sliding window, 500 elements)")
            print("=" * 80)
            print(f"  Vectorized:  {d['vectorized_mean_us']:.1f} µs")
            print(f"  Loop-based:  {d['loop_mean_us']:.1f} µs")
            print(f"  Speedup:     {d['speedup']:.1f}x")

        # Save results
        _save_results()
        print(f"\nRaw data saved to: {RESULTS_PATH}")
