"""
Adversarial Test Suite for insight137_eap
==========================================

Comprehensive tests designed to break every public function via:
  1. Input validation attacks (type errors, boundary values, NaN/Inf)
  2. Numerical edge cases (zeros, extremes, log-of-zero)
  3. Mathematical correctness (property tests, paper verification)
  4. Adversarial sequences (spikes, monotonic, noise)
  5. Psi4 edge cases (min/max agents, consensus, split)
  6. Frozen dataclass integrity (immutability, round-trip, hash)
  7. Concurrency (thread-safety of stateless computation)
"""

import math
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

import insight137_eap as eap


# ═══════════════════════════════════════════════════════════════════════
# 1. INPUT VALIDATION ATTACKS
# ═══════════════════════════════════════════════════════════════════════

class TestInputValidationAttacks:
    """Tests that every function rejects or safely handles garbage input."""

    # --- _validate_probability ---

    @pytest.mark.parametrize("bad_val", [None, "0.5", [0.5], {"v": 0.5}])
    def test_validate_probability_wrong_types(self, bad_val):
        """Non-numeric types must raise TypeError, not silently coerce."""
        with pytest.raises(TypeError):
            eap._validate_probability(bad_val, "test")

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_validate_probability_nan_inf(self, bad_val):
        """NaN and Inf must raise ValueError — they are numeric but not finite."""
        with pytest.raises(ValueError):
            eap._validate_probability(bad_val, "test")

    def test_validate_probability_negative_clamps(self):
        """Negative probabilities should be clamped to 0.0, not raise."""
        result = eap._validate_probability(-0.5, "test")
        assert result == 0.0

    def test_validate_probability_above_one_clamps(self):
        """Probabilities > 1.0 should be clamped to 1.0, not raise."""
        result = eap._validate_probability(1.5, "test")
        assert result == 1.0

    # --- _validate_sequence ---

    def test_validate_sequence_empty(self):
        """Empty sequences must be rejected when min_length >= 1."""
        with pytest.raises(ValueError):
            eap._validate_sequence([], "test", min_length=1)

    def test_validate_sequence_nan_in_array(self):
        """Arrays containing NaN must be rejected."""
        with pytest.raises(ValueError):
            eap._validate_sequence([1.0, float("nan"), 3.0], "test")

    def test_validate_sequence_inf_in_array(self):
        """Arrays containing Inf must be rejected."""
        with pytest.raises(ValueError):
            eap._validate_sequence([1.0, float("inf"), 3.0], "test")

    def test_validate_sequence_2d_array(self):
        """2D arrays must be rejected — only 1D allowed."""
        with pytest.raises(ValueError):
            eap._validate_sequence([[1.0, 2.0], [3.0, 4.0]], "test")

    def test_validate_sequence_strings_raise(self):
        """String elements should fail during float conversion."""
        with pytest.raises((ValueError, TypeError)):
            eap._validate_sequence(["a", "b", "c"], "test")

    def test_validate_sequence_single_element(self):
        """Single-element sequence is valid when min_length=1."""
        arr = eap._validate_sequence([42.0], "test", min_length=1)
        assert len(arr) == 1

    # --- deng_entropy ---

    def test_deng_entropy_empty_masses(self):
        """Empty mass list must raise ValueError."""
        with pytest.raises(ValueError):
            eap.deng_entropy([])

    def test_deng_entropy_negative_mass(self):
        """Negative mass values must raise ValueError."""
        with pytest.raises(ValueError):
            eap.deng_entropy([(1, -0.5)])

    def test_deng_entropy_zero_cardinality(self):
        """Cardinality < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            eap.deng_entropy([(0, 0.5)])

    def test_deng_entropy_negative_cardinality(self):
        """Negative cardinality must raise ValueError."""
        with pytest.raises(ValueError):
            eap.deng_entropy([(-1, 0.5)])

    # --- shannon_entropy ---

    def test_shannon_entropy_negative_probs(self):
        """Probabilities significantly below zero must raise ValueError."""
        with pytest.raises(ValueError):
            eap.shannon_entropy([-0.5, 1.5])

    def test_shannon_entropy_all_zeros(self):
        """All-zero probabilities should return 0.0 (no information)."""
        assert eap.shannon_entropy([0.0, 0.0, 0.0]) == 0.0

    def test_shannon_entropy_single_one(self):
        """A certain event [1.0] has zero entropy."""
        assert eap.shannon_entropy([1.0]) == 0.0

    # --- belief_degree_huang ---

    def test_belief_degree_empty_outcomes(self):
        """Empty outcomes list must raise ValueError."""
        with pytest.raises(ValueError):
            eap.belief_degree_huang([])

    def test_belief_degree_bad_prior_sum(self):
        """Priors that don't sum to ~1.0 must raise ValueError."""
        outcomes = [{"p_given_a_true": 0.5, "p_given_a_false": 0.5}]
        with pytest.raises(ValueError):
            eap.belief_degree_huang(outcomes, p_a_true=0.3, p_a_false=0.3)

    def test_belief_degree_n_unobserved_zero(self):
        """n_unobserved < 1 must raise ValueError."""
        outcomes = [{"p_given_a_true": 0.5, "p_given_a_false": 0.5}]
        with pytest.raises(ValueError):
            eap.belief_degree_huang(outcomes, n_unobserved=0)

    def test_belief_degree_nan_priors(self):
        """NaN priors must raise ValueError."""
        outcomes = [{"p_given_a_true": 0.5, "p_given_a_false": 0.5}]
        with pytest.raises(ValueError):
            eap.belief_degree_huang(outcomes, p_a_true=float("nan"), p_a_false=0.5)

    # --- quantum_probability ---

    def test_quantum_probability_empty_conditionals(self):
        """Empty conditionals dict must raise ValueError."""
        with pytest.raises(ValueError):
            eap.quantum_probability({})

    def test_quantum_probability_inf_priors(self):
        """Inf priors must raise ValueError."""
        conds = {"x": {"p_given_a_true": 0.5, "p_given_a_false": 0.5}}
        with pytest.raises(ValueError):
            eap.quantum_probability(conds, p_a_true=float("inf"), p_a_false=0.5)

    # --- compute_psi_from_sequence ---

    def test_psi_from_sequence_empty(self):
        """Empty sequence must raise ValueError."""
        with pytest.raises(ValueError):
            eap.compute_psi_from_sequence([])

    def test_psi_from_sequence_nan_values(self):
        """NaN in sequence must raise ValueError."""
        with pytest.raises(ValueError):
            eap.compute_psi_from_sequence([1.0, float("nan"), 3.0])

    def test_psi_from_sequence_dict_input(self):
        """Dict input where list expected must raise."""
        with pytest.raises((TypeError, ValueError)):
            eap.compute_psi_from_sequence({"a": 1, "b": 2})

    # --- compute_psi4 ---

    @pytest.mark.parametrize("bad_val", [None, "0.5"])
    def test_psi4_wrong_types_in_list(self, bad_val):
        """Non-numeric elements in agent_probabilities must raise TypeError."""
        with pytest.raises(TypeError):
            eap.compute_psi4([bad_val, 0.5])

    def test_psi4_nan_in_list(self):
        """NaN in agent probabilities must raise ValueError."""
        with pytest.raises(ValueError):
            eap.compute_psi4([float("nan"), 0.5])

    def test_psi4_inf_in_list(self):
        """Inf in agent probabilities must raise ValueError."""
        with pytest.raises(ValueError):
            eap.compute_psi4([float("inf"), 0.5])

    # --- ConditionalProbability dataclass ---

    def test_conditional_probability_out_of_range(self):
        """ConditionalProbability rejects values outside [0, 1]."""
        with pytest.raises(ValueError):
            eap.ConditionalProbability(p_given_a_true=1.5, p_given_a_false=0.5)
        with pytest.raises(ValueError):
            eap.ConditionalProbability(p_given_a_true=0.5, p_given_a_false=-0.1)

    def test_conditional_probability_valid(self):
        """ConditionalProbability accepts values in [0, 1]."""
        cp = eap.ConditionalProbability(p_given_a_true=0.0, p_given_a_false=1.0)
        assert cp.p_given_a_true == 0.0
        assert cp.p_given_a_false == 1.0

    # --- Unicode / bytes edge cases ---

    def test_validate_probability_bytes_object(self):
        """Bytes objects must raise TypeError."""
        with pytest.raises(TypeError):
            eap._validate_probability(b"0.5", "test")

    def test_validate_probability_unicode_string(self):
        """Unicode strings must raise TypeError."""
        with pytest.raises(TypeError):
            eap._validate_probability("½", "test")


# ═══════════════════════════════════════════════════════════════════════
# 2. NUMERICAL EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestNumericalEdgeCases:
    """Tests for extreme values, boundary conditions, and near-zero denominators."""

    def test_all_zeros_sequence(self):
        """All-zero sequence should produce psi_1=0 (no entropy)."""
        profile = eap.compute_psi_from_sequence([0.0, 0.0, 0.0, 0.0])
        assert profile.psi_1 == 0.0

    def test_all_identical_values(self):
        """All identical values: uniform distribution after normalization."""
        profile = eap.compute_psi_from_sequence([5.0, 5.0, 5.0, 5.0])
        assert profile.psi_1 > 0  # entropy of uniform distribution is positive
        assert np.isfinite(profile.psi_1)

    def test_extremely_small_values(self):
        """Values near floating-point underflow should not produce NaN/Inf."""
        profile = eap.compute_psi_from_sequence([1e-300, 1e-300, 1e-300, 1e-300])
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)
        assert np.isfinite(profile.psi_3)

    def test_extremely_large_values(self):
        """Very large values should not overflow."""
        profile = eap.compute_psi_from_sequence([1e300, 1e300, 1e300, 1e300])
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)

    def test_mixed_extreme_values(self):
        """Mixing very small and very large values should not crash."""
        profile = eap.compute_psi_from_sequence([1e-200, 1e200, 1e-200, 1e200])
        assert np.isfinite(profile.psi_1)

    def test_probability_exact_zero(self):
        """Probability exactly 0.0 should not cause log(0) errors."""
        result = eap.shannon_entropy([0.0, 1.0])
        assert result == 0.0  # -1*log2(1) = 0

    def test_probability_exact_one(self):
        """Probability exactly 1.0: entropy should be 0."""
        result = eap.shannon_entropy([1.0, 0.0])
        assert result == 0.0

    def test_probability_exact_half(self):
        """Probability 0.5/0.5: maximum entropy for binary."""
        result = eap.shannon_entropy([0.5, 0.5])
        assert abs(result - 1.0) < 1e-10

    def test_belief_distance_denominator_near_zero(self):
        """When alpha + beta ≈ 1.0, denominator → 0. Must not crash."""
        # alpha=0.5, beta=0.5 → denominator = |0.5+0.5-1| = 0
        result = eap._belief_distance(0.5, 0.5)
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_belief_distance_both_zero(self):
        """Both amplitudes zero: degenerate case, must not crash."""
        result = eap._belief_distance(0.0, 0.0)
        assert np.isfinite(result)

    def test_belief_distance_both_one(self):
        """Both amplitudes 1.0: extreme case."""
        result = eap._belief_distance(1.0, 1.0)
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_deng_entropy_large_cardinality(self):
        """Large cardinality (2^card subsets) should not overflow for reasonable card."""
        # cardinality=50 → 2^50 subsets ≈ 1e15. Should be fine.
        result = eap.deng_entropy([(50, 1.0)])
        assert np.isfinite(result)
        assert result > 0.0

    def test_deng_entropy_huge_cardinality(self):
        """Very large cardinality (2^1024) is handled via log-space
        approximation to avoid OverflowError."""
        result = eap.deng_entropy([(1024, 1.0)])
        assert np.isfinite(result)
        assert result > 0.0

    def test_shannon_entropy_tiny_probabilities(self):
        """Very small probabilities: -p*log2(p) should stay finite."""
        probs = [1e-15, 1 - 1e-15]
        result = eap.shannon_entropy(probs)
        assert np.isfinite(result)

    def test_compute_psi3_two_element_sequence(self):
        """Minimum-viable sequence for psi3 (needs >= MIN_WINDOW_SIZE+1 = 3)."""
        # 2 elements: too short
        assert eap.compute_psi3([1.0, 2.0]) == 0.0
        # 3 elements: minimum viable
        result = eap.compute_psi3([1.0, 2.0, 3.0])
        assert np.isfinite(result)

    def test_cohens_d_identical_groups(self):
        """Identical groups: pooled std ≈ 0, should return 0.0 not Inf."""
        result = eap.cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert result == 0.0

    def test_cohens_d_single_element_rejected(self):
        """Groups with < 2 elements must raise ValueError."""
        with pytest.raises(ValueError):
            eap.cohens_d([5.0], [5.0, 6.0])


# ═══════════════════════════════════════════════════════════════════════
# 3. MATHEMATICAL CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════

class TestMathematicalCorrectness:
    """Property tests validating mathematical guarantees of the library."""

    def test_deng_reduces_to_shannon_for_singletons(self):
        """When all focal elements are singletons (|A|=1), Deng entropy
        must be exactly Shannon entropy. This is the fundamental property
        from Deng (2016)."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            n = rng.integers(2, 10)
            raw = rng.random(n)
            probs = raw / raw.sum()
            masses = [(1, float(p)) for p in probs]
            deng_val = eap.deng_entropy(masses)
            shannon_val = eap.shannon_entropy(probs)
            assert abs(deng_val - shannon_val) < 1e-10, (
                f"Deng={deng_val}, Shannon={shannon_val} for probs={probs}"
            )

    def test_quantum_probability_sums_to_one(self):
        """Quantum probability must sum to exactly 1.0 for any valid input.
        This is guaranteed by Born rule normalization."""
        rng = np.random.default_rng(137)
        for _ in range(100):
            p1 = rng.uniform(0.01, 0.99)
            p2 = 1.0 - p1
            conds = {
                "A": {"p_given_a_true": p1, "p_given_a_false": rng.uniform(0.01, 0.99)},
                "B": {"p_given_a_true": p2, "p_given_a_false": rng.uniform(0.01, 0.99)},
            }
            q = eap.quantum_probability(conds)
            total = sum(q.values())
            assert abs(total - 1.0) < 1e-10, (
                f"Quantum probs sum to {total}, not 1.0"
            )

    def test_belief_degree_always_in_valid_range(self):
        """Belief degree D_b must be finite for all valid inputs.
        The Huang formulation can produce values outside [-1,1], but they
        must always be finite."""
        rng = np.random.default_rng(2024)
        for _ in range(100):
            p_true = rng.uniform(0.01, 0.99)
            p_false = rng.uniform(0.01, 0.99)
            outcomes = [
                {"p_given_a_true": p_true, "p_given_a_false": p_false},
                {"p_given_a_true": 1 - p_true, "p_given_a_false": 1 - p_false},
            ]
            db = eap.belief_degree_huang(outcomes)
            assert np.isfinite(db), f"D_b is not finite: {db}"

    def test_psi_profile_dimensions_non_negative(self):
        """All four Psi dimensions must be >= 0 for any valid input.
        Psi1=entropy, Psi2=|interference|, Psi3=std, Psi4=std — all non-negative."""
        rng = np.random.default_rng(99)
        for _ in range(50):
            n = rng.integers(5, 20)
            vals = rng.uniform(1.0, 100.0, size=n)
            agents = list(rng.uniform(0.01, 0.99, size=rng.integers(2, 8)))
            profile = eap.compute_psi_from_sequence(vals, agent_decisions=agents)
            assert profile.psi_1 >= 0, f"psi_1 = {profile.psi_1}"
            assert profile.psi_2 >= 0, f"psi_2 = {profile.psi_2}"
            assert profile.psi_3 >= 0, f"psi_3 = {profile.psi_3}"
            assert profile.psi_4 >= 0, f"psi_4 = {profile.psi_4}"

    def test_compute_psi_from_sequence_deterministic(self):
        """Same input must always produce exactly the same PsiProfile."""
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        agents = [0.9, 0.1, 0.5]
        p1 = eap.compute_psi_from_sequence(vals, agent_decisions=agents)
        p2 = eap.compute_psi_from_sequence(vals, agent_decisions=agents)
        assert p1 == p2

    def test_verify_huang_paper_all_pass(self):
        """The built-in verification against Huang et al. (2019) must pass."""
        results = eap.verify_huang_paper()
        for check, passed in results.items():
            assert passed, f"Verification check '{check}' failed"

    def test_deng_exceeds_shannon_for_imprecise(self):
        """Deng entropy must be strictly greater than Shannon entropy
        when focal elements have cardinality > 1."""
        probs = [0.3, 0.7]
        shannon_val = eap.shannon_entropy(probs)
        # Same masses but with |A|=2 for second focal element
        deng_val = eap.deng_entropy([(1, 0.3), (2, 0.7)])
        assert deng_val > shannon_val

    def test_cohens_d_sign_convention(self):
        """Cohen's d should be positive when group_a > group_b."""
        a = [10.0, 11.0, 12.0, 13.0]
        b = [1.0, 2.0, 3.0, 4.0]
        d = eap.cohens_d(a, b)
        assert d > 0
        # Reverse should be negative
        d_rev = eap.cohens_d(b, a)
        assert d_rev < 0
        assert abs(d + d_rev) < 1e-10  # antisymmetric

    def test_bias_phase_symmetry(self):
        """bias_phase should be antisymmetric around the center p0=0.5.
        phase(0.5+d) = -phase(0.5-d)"""
        for delta in [0.1, 0.2, 0.3, 0.4, 0.49]:
            p_high = eap._bias_phase(0.5 + delta)
            p_low = eap._bias_phase(0.5 - delta)
            assert abs(p_high + p_low) < 1e-10, (
                f"Asymmetry at delta={delta}: {p_high} vs {p_low}"
            )

    def test_bias_phase_center_is_zero(self):
        """phase(0.5) should be exactly 0."""
        assert eap._bias_phase(0.5) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 4. ADVERSARIAL SEQUENCES
# ═══════════════════════════════════════════════════════════════════════

class TestAdversarialSequences:
    """Tests with pathological input patterns designed to trigger edge cases."""

    def test_alternating_zero_nonzero(self):
        """Alternating 0.0 and 100.0: windows may have near-zero sums."""
        seq = [0.0, 100.0] * 20
        profile = eap.compute_psi_from_sequence(seq)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)
        assert np.isfinite(profile.psi_3)

    def test_monotonically_increasing(self):
        """Monotonically increasing sequence: 1..1000."""
        seq = list(range(1, 1001))
        profile = eap.compute_psi_from_sequence(seq)
        assert np.isfinite(profile.psi_1)
        assert profile.psi_1 > 0  # non-uniform distribution

    def test_single_spike_in_flat(self):
        """One large value among many small ones: extreme skew."""
        seq = [1.0] * 99 + [10000.0]
        profile = eap.compute_psi_from_sequence(seq)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)
        assert np.isfinite(profile.psi_3)

    def test_all_identical_except_last(self):
        """All 1.0 except last element is 1000.0."""
        seq = [1.0] * 49 + [1000.0]
        profile = eap.compute_psi_from_sequence(seq)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_3)

    def test_random_uniform_noise(self):
        """Uniform random noise should produce finite results."""
        rng = np.random.default_rng(42)
        seq = rng.uniform(0.0, 100.0, size=500)
        profile = eap.compute_psi_from_sequence(seq)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)
        assert np.isfinite(profile.psi_3)

    def test_random_gaussian_noise(self):
        """Gaussian noise (with abs to ensure positive values)."""
        rng = np.random.default_rng(42)
        seq = np.abs(rng.normal(50.0, 25.0, size=500))
        profile = eap.compute_psi_from_sequence(seq)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)

    def test_massive_list_100k_elements(self):
        """100K element sequence: stress test for performance and correctness."""
        seq = np.ones(100_000)
        seq[50_000] = 999.0  # one spike
        profile = eap.compute_psi_from_sequence(seq)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)
        assert np.isfinite(profile.psi_3)

    def test_single_element_sequence(self):
        """Single element: should return profile with psi_2=psi_3=0."""
        profile = eap.compute_psi_from_sequence([42.0])
        assert profile.psi_1 == 0.0  # single element → no entropy
        assert profile.psi_2 == 0.0
        assert profile.psi_3 == 0.0

    def test_two_element_sequence(self):
        """Two elements: should compute psi_1 but psi_3=0 (too short)."""
        profile = eap.compute_psi_from_sequence([10.0, 20.0])
        assert np.isfinite(profile.psi_1)
        assert profile.psi_3 == 0.0  # needs >= 3 elements

    def test_exponentially_growing_sequence(self):
        """Exponentially growing values: tests numerical stability."""
        seq = [2.0 ** i for i in range(50)]
        profile = eap.compute_psi_from_sequence(seq)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)

    def test_near_zero_values_sequence(self):
        """All values near EPSILON: tests log-of-near-zero paths."""
        seq = [1e-14, 1e-14, 1e-14, 1e-14]
        profile = eap.compute_psi_from_sequence(seq)
        assert np.isfinite(profile.psi_1)
        # These values are at the EPSILON threshold — windows may be skipped
        assert np.isfinite(profile.psi_2)


# ═══════════════════════════════════════════════════════════════════════
# 5. PSI4 EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestPsi4EdgeCases:
    """Edge cases for the relational entropy dimension (cross-agent)."""

    def test_single_agent_returns_zero(self):
        """Single agent: below MIN_AGENTS_PSI4, should return 0.0."""
        assert eap.compute_psi4([0.5]) == 0.0

    def test_two_agents_minimum(self):
        """Two agents: minimum valid input, should produce finite result."""
        result = eap.compute_psi4([0.3, 0.7])
        assert np.isfinite(result)
        assert result >= 0.0

    def test_100_agents(self):
        """100 agents: stress test for O(n^2) pairwise computation."""
        rng = np.random.default_rng(42)
        agents = list(rng.uniform(0.01, 0.99, size=100))
        result = eap.compute_psi4(agents)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_all_agents_same_low_decision(self):
        """All agents at minimum probability: complete consensus on comply.
        Floating-point arithmetic in pairwise cos/sqrt may produce noise ~1e-18."""
        agents = [0.01] * 10
        result = eap.compute_psi4(agents)
        assert np.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-14)

    def test_all_agents_same_high_decision(self):
        """All agents at 0.99: complete consensus on bypass.
        Floating-point noise ~1e-16 expected from pairwise computation."""
        agents = [0.99] * 10
        result = eap.compute_psi4(agents)
        assert np.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-14)

    def test_agents_half_and_half_split(self):
        """Half agents bypass (1.0), half comply (0.01): maximum disagreement."""
        agents = [1.0] * 5 + [0.01] * 5
        result = eap.compute_psi4(agents)
        assert np.isfinite(result)
        assert result > 0.0  # should show diversity

    def test_empty_agents_returns_zero(self):
        """Empty agent list: should return 0.0."""
        assert eap.compute_psi4([]) == 0.0

    def test_agents_at_boundary_zero(self):
        """Agent probability exactly 0.0: should be clamped to PROB_FLOOR."""
        result = eap.compute_psi4([0.0, 0.5])
        assert np.isfinite(result)

    def test_agents_at_boundary_one(self):
        """Agent probability exactly 1.0: maximum value."""
        result = eap.compute_psi4([1.0, 0.0])
        assert np.isfinite(result)
        assert result >= 0.0

    def test_psi4_via_compute_psi_from_sequence(self):
        """Psi4 computed through the unified entry point matches direct call."""
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        agents = [0.9, 0.1, 0.5, 0.3]
        profile = eap.compute_psi_from_sequence(vals, agent_decisions=agents)
        direct = eap.compute_psi4(agents)
        assert abs(profile.psi_4 - round(direct, 6)) < 1e-10

    def test_psi4_no_agents_is_zero(self):
        """When agent_decisions=None, psi_4 should be 0.0."""
        profile = eap.compute_psi_from_sequence([10.0, 20.0, 30.0])
        assert profile.psi_4 == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 6. FROZEN DATACLASS INTEGRITY
# ═══════════════════════════════════════════════════════════════════════

class TestFrozenDataclassIntegrity:
    """Verify that frozen dataclasses are truly immutable and serialize correctly."""

    def test_psi_profile_immutable(self):
        """Attempting to set an attribute on a frozen PsiProfile must raise."""
        profile = eap.PsiProfile(psi_1=1.0, psi_2=2.0, psi_3=3.0, psi_4=4.0)
        with pytest.raises(AttributeError):
            profile.psi_1 = 99.0
        with pytest.raises(AttributeError):
            profile.psi_2 = 99.0
        with pytest.raises(AttributeError):
            profile.method = "hacked"

    def test_to_dict_round_trip(self):
        """to_dict() output should faithfully represent the profile.
        Creating a new PsiProfile from to_dict() values should be equivalent."""
        profile = eap.PsiProfile(
            psi_1=0.123456789, psi_2=0.987654321,
            psi_3=0.111111111, psi_4=0.222222222,
            belief_degree=-0.555555555,
            method=eap.PsiMethod.HUANG_2019.value,
        )
        d = profile.to_dict()
        # Check all keys present
        assert set(d.keys()) == {"psi_1", "psi_2", "psi_3", "psi_4", "belief_degree", "method"}
        # Reconstruct (with rounded values as to_dict rounds to 6 decimals)
        reconstructed = eap.PsiProfile(
            psi_1=d["psi_1"], psi_2=d["psi_2"],
            psi_3=d["psi_3"], psi_4=d["psi_4"],
            belief_degree=d["belief_degree"],
            method=d["method"],
        )
        assert reconstructed.to_dict() == d  # round-trip stable

    def test_hash_stability(self):
        """Same PsiProfile values must produce the same hash."""
        p1 = eap.PsiProfile(psi_1=1.0, psi_2=2.0, psi_3=3.0, psi_4=4.0)
        p2 = eap.PsiProfile(psi_1=1.0, psi_2=2.0, psi_3=3.0, psi_4=4.0)
        assert hash(p1) == hash(p2)
        assert p1 == p2

    def test_hash_different_for_different_values(self):
        """Different values should (almost certainly) produce different hashes."""
        p1 = eap.PsiProfile(psi_1=1.0, psi_2=2.0, psi_3=3.0, psi_4=4.0)
        p2 = eap.PsiProfile(psi_1=1.0, psi_2=2.0, psi_3=3.0, psi_4=4.1)
        assert hash(p1) != hash(p2)

    def test_psi_profile_usable_as_dict_key(self):
        """Frozen dataclass should be hashable and usable as dict key."""
        profile = eap.PsiProfile(psi_1=1.0, psi_2=2.0, psi_3=3.0, psi_4=4.0)
        d = {profile: "test"}
        assert d[profile] == "test"

    def test_conditional_probability_immutable(self):
        """ConditionalProbability is also frozen."""
        cp = eap.ConditionalProbability(p_given_a_true=0.5, p_given_a_false=0.5)
        with pytest.raises(AttributeError):
            cp.p_given_a_true = 0.99

    def test_to_dict_rounding(self):
        """to_dict must round to exactly 6 decimal places."""
        profile = eap.PsiProfile(
            psi_1=0.1234567890123, psi_2=0.0, psi_3=0.0, psi_4=0.0
        )
        d = profile.to_dict()
        assert d["psi_1"] == round(0.1234567890123, 6)


# ═══════════════════════════════════════════════════════════════════════
# 7. CONCURRENCY
# ═══════════════════════════════════════════════════════════════════════

class TestConcurrency:
    """Verify thread-safety: stateless functions should produce correct
    results when called from multiple threads simultaneously."""

    def test_concurrent_compute_psi_from_sequence(self):
        """10 threads computing PsiProfile from different sequences.
        Each must get its own correct result — no shared state corruption."""
        rng = np.random.default_rng(42)
        inputs = [rng.uniform(1.0, 100.0, size=50) for _ in range(10)]
        # Compute expected results sequentially first
        expected = [eap.compute_psi_from_sequence(inp) for inp in inputs]

        results = [None] * 10
        errors = []

        def worker(idx):
            try:
                results[idx] = eap.compute_psi_from_sequence(inputs[idx])
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        for i in range(10):
            assert results[i] == expected[i], (
                f"Thread {i} result mismatch: {results[i]} != {expected[i]}"
            )

    def test_concurrent_quantum_probability(self):
        """Multiple threads computing quantum_probability simultaneously."""
        rng = np.random.default_rng(99)
        conds_list = []
        for _ in range(10):
            p1 = rng.uniform(0.1, 0.9)
            conds_list.append({
                "A": {"p_given_a_true": p1, "p_given_a_false": rng.uniform(0.1, 0.9)},
                "B": {"p_given_a_true": 1 - p1, "p_given_a_false": rng.uniform(0.1, 0.9)},
            })

        expected = [eap.quantum_probability(c) for c in conds_list]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(eap.quantum_probability, c): i
                for i, c in enumerate(conds_list)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                for key in expected[idx]:
                    assert abs(result[key] - expected[idx][key]) < 1e-12, (
                        f"Thread {idx} mismatch for {key}"
                    )

    def test_concurrent_psi4(self):
        """Multiple threads computing psi4 simultaneously."""
        rng = np.random.default_rng(137)
        inputs = [list(rng.uniform(0.01, 0.99, size=20)) for _ in range(10)]
        expected = [eap.compute_psi4(inp) for inp in inputs]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(eap.compute_psi4, inp): i
                for i, inp in enumerate(inputs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                assert abs(result - expected[idx]) < 1e-12


# ═══════════════════════════════════════════════════════════════════════
# 8. ADDITIONAL ATTACK VECTORS
# ═══════════════════════════════════════════════════════════════════════

class TestAdditionalAttacks:
    """Extra tests for boundary conditions and unusual but valid inputs."""

    def test_compute_psi_from_conditionals_single_outcome(self):
        """Single outcome: should produce valid profile."""
        conds = {"only": {"p_given_a_true": 0.8, "p_given_a_false": 0.6}}
        profile = eap.compute_psi_from_conditionals(conds)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)

    def test_compute_psi_from_conditionals_many_outcomes(self):
        """Many outcomes: 20 outcome entries."""
        rng = np.random.default_rng(42)
        conds = {}
        for i in range(20):
            conds[f"o{i}"] = {
                "p_given_a_true": float(rng.uniform(0.01, 0.99)),
                "p_given_a_false": float(rng.uniform(0.01, 0.99)),
            }
        profile = eap.compute_psi_from_conditionals(conds)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)

    def test_quantum_probability_extreme_asymmetry(self):
        """Extreme prior asymmetry: P(A)=0.99, P(~A)=0.01."""
        conds = {
            "X": {"p_given_a_true": 0.9, "p_given_a_false": 0.1},
            "Y": {"p_given_a_true": 0.1, "p_given_a_false": 0.9},
        }
        q = eap.quantum_probability(conds, p_a_true=0.99, p_a_false=0.01)
        total = sum(q.values())
        assert abs(total - 1.0) < 1e-10

    def test_quantum_probability_near_identical_conditionals(self):
        """When conditionals are nearly identical, interference should be small."""
        conds = {
            "X": {"p_given_a_true": 0.50, "p_given_a_false": 0.50},
            "Y": {"p_given_a_true": 0.50, "p_given_a_false": 0.50},
        }
        q = eap.quantum_probability(conds)
        # Should be close to classical (0.5, 0.5)
        assert abs(q["X"] - 0.5) < 0.1
        assert abs(q["Y"] - 0.5) < 0.1

    def test_compute_psi_from_sequence_numpy_input(self):
        """numpy array input should work identically to list input."""
        lst = [10.0, 20.0, 30.0, 40.0, 50.0]
        arr = np.array(lst)
        p_list = eap.compute_psi_from_sequence(lst)
        p_arr = eap.compute_psi_from_sequence(arr)
        assert p_list == p_arr

    def test_compute_psi_from_sequence_integer_input(self):
        """Integer sequences should be accepted (auto-cast to float64)."""
        profile = eap.compute_psi_from_sequence([10, 20, 30, 40, 50])
        assert np.isfinite(profile.psi_1)

    def test_psi_method_enum_values(self):
        """PsiMethod enum should have expected string values."""
        assert eap.PsiMethod.HUANG_2019.value == "huang_2019"
        assert eap.PsiMethod.CLASSICAL.value == "classical"

    def test_backward_compat_aliases(self):
        """Backward compatibility aliases should reference the same functions."""
        assert eap.quantum_probability_qlbn is eap.quantum_probability
        assert eap.compute_amplitude_vectors is eap._compute_amplitude_vectors
        assert eap.belief_distance_huang_raw is eap._belief_distance

    def test_belief_degree_missing_keys_uses_defaults(self):
        """Outcomes dicts missing keys should use default 0.5 via .get()."""
        outcomes = [{"p_given_a_true": 0.8}]  # missing p_given_a_false
        db = eap.belief_degree_huang(outcomes)
        assert np.isfinite(db)

    def test_sliding_interferences_all_zero_windows(self):
        """All-zero windows should be skipped (window_sum <= EPSILON)."""
        interferences = eap._compute_sliding_interferences(
            np.array([0.0, 0.0, 0.0, 0.0]), window_size=3
        )
        assert interferences == []

    def test_psi3_returns_zero_for_short_sequence(self):
        """Sequences shorter than MIN_WINDOW_SIZE+1 should return 0.0."""
        assert eap.compute_psi3([1.0]) == 0.0
        assert eap.compute_psi3([1.0, 2.0]) == 0.0

    def test_large_window_size(self):
        """Window larger than sequence: should still work (effective_window clamped)."""
        profile = eap.compute_psi_from_sequence([10.0, 20.0, 30.0], window_size=100)
        assert np.isfinite(profile.psi_1)
        assert np.isfinite(profile.psi_2)
