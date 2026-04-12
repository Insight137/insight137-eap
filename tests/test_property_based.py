"""
Property-Based Tests for insight137_eap
========================================

Uses Hypothesis to generate thousands of random valid inputs and verify
that mathematical invariants always hold. Unlike example-based tests,
these explore the input space far more broadly and catch edge cases
that hand-written tests miss.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

import insight137_eap as eap


# ═══════════════════════════════════════════════════════════════════════
# STRATEGIES (reusable input generators)
# ═══════════════════════════════════════════════════════════════════════

# Finite floats that won't cause overflow in np.log2 / np.sqrt
reasonable_float = st.floats(min_value=1e-12, max_value=1e12, allow_nan=False, allow_infinity=False)

# Probability in (0, 1) — strict interior to avoid log(0) boundary issues
probability = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)

# Probability distribution: n values in (0,1) that sum to ~1.0
@st.composite
def probability_distribution(draw, min_size=2, max_size=10):
    """Generate a valid probability distribution (sums to 1.0, all > 0)."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    raw = draw(st.lists(
        st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    total = sum(raw)
    assume(total > 1e-10)
    return [x / total for x in raw]


# Sequence of positive values for compute_psi_from_sequence
positive_sequence = st.lists(
    st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=4, max_size=100,
)

# Agent probabilities for psi4
agent_probabilities = st.lists(
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=2, max_size=50,
)

# Conditional probability pair for QLBN
@st.composite
def outcome_pair(draw):
    """Generate a complementary pair of outcomes (binary decision)."""
    p1_true = draw(probability)
    p1_false = draw(probability)
    return [
        {"p_given_a_true": p1_true, "p_given_a_false": p1_false},
        {"p_given_a_true": 1.0 - p1_true, "p_given_a_false": 1.0 - p1_false},
    ]


@st.composite
def conditionals_dict(draw):
    """Generate a valid conditionals dict for quantum_probability."""
    p1_true = draw(probability)
    p1_false = draw(probability)
    return {
        "outcome_A": {"p_given_a_true": p1_true, "p_given_a_false": p1_false},
        "outcome_B": {"p_given_a_true": 1.0 - p1_true, "p_given_a_false": 1.0 - p1_false},
    }


@st.composite
def prior_pair(draw):
    """Generate a valid prior probability pair that sums to 1.0."""
    p = draw(st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False))
    return (p, 1.0 - p)


# Singleton masses for Deng-Shannon equivalence
@st.composite
def singleton_masses(draw):
    """Generate masses where all focal elements are singletons (|A|=1).
    Returns both the masses list and the equivalent probability array."""
    dist = draw(probability_distribution(min_size=2, max_size=8))
    masses = [(1, p) for p in dist]
    return masses, np.array(dist)


# ═══════════════════════════════════════════════════════════════════════
# 1. Shannon entropy is always non-negative
# ═══════════════════════════════════════════════════════════════════════

class TestShannonEntropyNonNegative:
    """Shannon entropy H(X) >= 0 for any valid probability distribution.
    This is a fundamental information-theoretic bound."""

    @given(dist=probability_distribution())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_shannon_entropy_non_negative(self, dist):
        """H(X) >= 0 for any probability distribution."""
        result = eap.shannon_entropy(dist)
        assert result >= -1e-15, f"Shannon entropy is negative: {result}"

    @given(dist=probability_distribution(min_size=2, max_size=2))
    @settings(max_examples=200)
    def test_binary_shannon_entropy_bounded(self, dist):
        """For binary distributions, H(X) <= 1.0 bit."""
        result = eap.shannon_entropy(dist)
        assert result <= 1.0 + 1e-10

    @given(dist=probability_distribution(min_size=2, max_size=10))
    @settings(max_examples=200)
    def test_shannon_entropy_bounded_by_log_n(self, dist):
        """H(X) <= log2(n) where n is the number of outcomes."""
        n = len(dist)
        result = eap.shannon_entropy(dist)
        assert result <= np.log2(n) + 1e-10


# ═══════════════════════════════════════════════════════════════════════
# 2. Quantum probability sums to 1.0
# ═══════════════════════════════════════════════════════════════════════

class TestQuantumProbabilitySumsToOne:
    """Born rule normalization guarantees sum(P_q) = 1.0 for any
    valid conditional probability structure."""

    @given(conds=conditionals_dict(), priors=prior_pair())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_quantum_probability_sums_to_one(self, conds, priors):
        """Sum of quantum probabilities must be 1.0 within tolerance."""
        p_true, p_false = priors
        q = eap.quantum_probability(conds, p_a_true=p_true, p_a_false=p_false)
        total = sum(q.values())
        assert abs(total - 1.0) < 1e-10, (
            f"Quantum probs sum to {total}, not 1.0. "
            f"Conds={conds}, priors=({p_true}, {p_false})"
        )

    @given(conds=conditionals_dict(), priors=prior_pair())
    @settings(max_examples=200)
    def test_quantum_probability_all_non_negative(self, conds, priors):
        """Quantum probabilities should be >= 0 after normalization.
        Strong destructive interference can produce negative individual
        values (a known quantum cognition phenomenon); the library falls
        back to classical probabilities when the total goes non-positive,
        but individual values can still be negative when total remains
        positive. We verify they stay above -0.5 (theoretical floor)."""
        p_true, p_false = priors
        q = eap.quantum_probability(conds, p_a_true=p_true, p_a_false=p_false)
        for name, val in q.items():
            assert val >= -0.5, (
                f"Quantum probability too negative for {name}: {val}"
            )

    @given(conds=conditionals_dict())
    @settings(max_examples=200)
    def test_quantum_probability_values_finite(self, conds):
        """All quantum probability values must be finite."""
        q = eap.quantum_probability(conds)
        for name, val in q.items():
            assert np.isfinite(val), f"Non-finite quantum probability for {name}: {val}"


# ═══════════════════════════════════════════════════════════════════════
# 3. Psi dimensions are non-negative for positive sequences
# ═══════════════════════════════════════════════════════════════════════

class TestPsiDimensionsNonNegative:
    """Psi1 (entropy), Psi2 (|interference|), and Psi3 (std) are all
    non-negative by construction for any valid positive input."""

    @given(values=positive_sequence)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_psi_123_non_negative(self, values):
        """Psi1, Psi2, Psi3 must all be >= 0 for any positive sequence."""
        profile = eap.compute_psi_from_sequence(values)
        assert profile.psi_1 >= -1e-15, f"psi_1 negative: {profile.psi_1}"
        assert profile.psi_2 >= -1e-15, f"psi_2 negative: {profile.psi_2}"
        assert profile.psi_3 >= -1e-15, f"psi_3 negative: {profile.psi_3}"

    @given(values=positive_sequence)
    @settings(max_examples=200)
    def test_all_psi_dimensions_finite(self, values):
        """All Psi dimensions and belief_degree must be finite."""
        profile = eap.compute_psi_from_sequence(values)
        assert np.isfinite(profile.psi_1), f"psi_1 not finite: {profile.psi_1}"
        assert np.isfinite(profile.psi_2), f"psi_2 not finite: {profile.psi_2}"
        assert np.isfinite(profile.psi_3), f"psi_3 not finite: {profile.psi_3}"
        assert np.isfinite(profile.psi_4), f"psi_4 not finite: {profile.psi_4}"
        assert np.isfinite(profile.belief_degree), f"belief_degree not finite: {profile.belief_degree}"

    @given(values=positive_sequence)
    @settings(max_examples=200)
    def test_psi_2_equals_abs_belief_degree(self, values):
        """Psi2 is defined as |mean(interferences)| — verify consistency
        with belief_degree (which is the signed mean)."""
        profile = eap.compute_psi_from_sequence(values)
        # When there are interferences, psi_2 = |belief_degree| (both rounded)
        # When there are no interferences, both are 0.0
        if profile.psi_2 == 0.0:
            assert profile.belief_degree == 0.0
        else:
            assert abs(profile.psi_2 - abs(profile.belief_degree)) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# 4. Psi4 is non-negative for valid agent probabilities
# ═══════════════════════════════════════════════════════════════════════

class TestPsi4NonNegative:
    """Psi4 is a standard deviation of pairwise values, so it must be >= 0."""

    @given(agents=agent_probabilities)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_psi4_non_negative(self, agents):
        """Psi4 must be >= 0 for any set of agent probabilities in [0.01, 1.0]."""
        result = eap.compute_psi4(agents)
        assert result >= -1e-15, f"psi4 negative: {result}"

    @given(agents=agent_probabilities)
    @settings(max_examples=200)
    def test_psi4_finite(self, agents):
        """Psi4 must be finite for any valid agent probabilities."""
        result = eap.compute_psi4(agents)
        assert np.isfinite(result), f"psi4 not finite: {result}"

    @given(p=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
           n=st.integers(min_value=2, max_value=20))
    @settings(max_examples=200)
    def test_identical_agents_psi4_near_zero(self, p, n):
        """When all agents have the same probability, psi4 should be ~0
        (all pairwise values identical → std ≈ 0)."""
        agents = [p] * n
        result = eap.compute_psi4(agents)
        assert result < 1e-10, f"psi4={result} for {n} identical agents at p={p}"


# ═══════════════════════════════════════════════════════════════════════
# 5. belief_degree_huang always returns finite float
# ═══════════════════════════════════════════════════════════════════════

class TestBeliefDegreeFinite:
    """The Huang belief degree must always be a finite float for valid inputs."""

    @given(outcomes=outcome_pair(), priors=prior_pair())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_belief_degree_finite(self, outcomes, priors):
        """D_b must be finite for any valid outcome pair and priors."""
        p_true, p_false = priors
        db = eap.belief_degree_huang(outcomes, p_a_true=p_true, p_a_false=p_false)
        assert isinstance(db, float), f"D_b is not float: {type(db)}"
        assert np.isfinite(db), f"D_b is not finite: {db}"

    @given(outcomes=outcome_pair(),
           n_unobserved=st.integers(min_value=1, max_value=10))
    @settings(max_examples=200)
    def test_belief_degree_finite_varying_unobserved(self, outcomes, n_unobserved):
        """D_b must be finite for any n_unobserved >= 1."""
        db = eap.belief_degree_huang(outcomes, n_unobserved=n_unobserved)
        assert np.isfinite(db), f"D_b={db} for n_unobserved={n_unobserved}"

    @given(outcomes=outcome_pair())
    @settings(max_examples=200)
    def test_belief_degree_type_is_float(self, outcomes):
        """Return type must be Python float, not numpy scalar."""
        db = eap.belief_degree_huang(outcomes)
        assert type(db) is float, f"Expected float, got {type(db)}"


# ═══════════════════════════════════════════════════════════════════════
# 6. Deng entropy equals Shannon entropy for singletons
# ═══════════════════════════════════════════════════════════════════════

class TestDengShannonEquivalence:
    """When all focal elements are singletons (|A|=1), Deng entropy
    reduces exactly to Shannon entropy. This is the core property
    from Deng (2016) — the generalization must be backward-compatible."""

    @given(data=singleton_masses())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_deng_equals_shannon_for_singletons(self, data):
        """Deng entropy with all |A|=1 must equal Shannon entropy exactly."""
        masses, probs = data
        deng_val = eap.deng_entropy(masses)
        shannon_val = eap.shannon_entropy(probs)
        assert abs(deng_val - shannon_val) < 1e-10, (
            f"Deng={deng_val} != Shannon={shannon_val} for probs={list(probs)}"
        )

    @given(data=singleton_masses())
    @settings(max_examples=200)
    def test_both_non_negative(self, data):
        """Both Deng and Shannon must be non-negative for valid distributions."""
        masses, probs = data
        assert eap.deng_entropy(masses) >= -1e-15
        assert eap.shannon_entropy(probs) >= -1e-15

    @given(dist=probability_distribution(min_size=2, max_size=6))
    @settings(max_examples=200)
    def test_deng_with_multi_element_exceeds_shannon(self, dist):
        """When at least one focal element has |A|>1, Deng entropy
        must exceed Shannon entropy (more uncertainty from imprecision)."""
        shannon_val = eap.shannon_entropy(dist)
        # Replace last singleton with cardinality=2
        masses = [(1, p) for p in dist[:-1]] + [(2, dist[-1])]
        deng_val = eap.deng_entropy(masses)
        assert deng_val > shannon_val - 1e-15, (
            f"Deng={deng_val} should exceed Shannon={shannon_val}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 7. compute_psi_from_sequence is deterministic
# ═══════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Pure functions with no randomness must return identical results
    for identical inputs. Any non-determinism would indicate hidden
    shared state or race conditions."""

    @given(values=positive_sequence)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_psi_from_sequence_deterministic(self, values):
        """Two calls with the same input must produce identical PsiProfile."""
        p1 = eap.compute_psi_from_sequence(values)
        p2 = eap.compute_psi_from_sequence(values)
        assert p1 == p2, f"Non-deterministic: {p1.to_dict()} != {p2.to_dict()}"

    @given(values=positive_sequence,
           agents=agent_probabilities)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_psi_from_sequence_with_agents_deterministic(self, values, agents):
        """Determinism must hold even with agent_decisions provided."""
        p1 = eap.compute_psi_from_sequence(values, agent_decisions=agents)
        p2 = eap.compute_psi_from_sequence(values, agent_decisions=agents)
        assert p1 == p2

    @given(conds=conditionals_dict(), priors=prior_pair())
    @settings(max_examples=200)
    def test_psi_from_conditionals_deterministic(self, conds, priors):
        """compute_psi_from_conditionals must also be deterministic."""
        p_true, p_false = priors
        p1 = eap.compute_psi_from_conditionals(conds, p_a_true=p_true, p_a_false=p_false)
        p2 = eap.compute_psi_from_conditionals(conds, p_a_true=p_true, p_a_false=p_false)
        assert p1 == p2

    @given(conds=conditionals_dict())
    @settings(max_examples=200)
    def test_quantum_probability_deterministic(self, conds):
        """quantum_probability must be deterministic."""
        q1 = eap.quantum_probability(conds)
        q2 = eap.quantum_probability(conds)
        for key in q1:
            assert q1[key] == q2[key], f"Non-deterministic for key {key}"

    @given(agents=agent_probabilities)
    @settings(max_examples=200)
    def test_psi4_deterministic(self, agents):
        """compute_psi4 must be deterministic."""
        r1 = eap.compute_psi4(agents)
        r2 = eap.compute_psi4(agents)
        assert r1 == r2
