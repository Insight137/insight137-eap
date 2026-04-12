"""
Insight137 EAP — Entropy Attunement Protocol
==============================================
Enterprise-grade library for 4-dimensional entropy profiling.

Integrates formally grounded methods from peer-reviewed research:
  - Deng (2016): Generalized entropy for belief functions
  - Huang, Yang, Jiang (2019): Belief entropy interference for QLBN
  - Moreira & Wichert (2016): Quantum-like Bayesian networks
  - Meghdadi, Akbarzadeh-T, Javidan (2022): BEQBN entanglement
  - Busemeyer & Bruza (2012): Quantum cognition foundations

Validated across 128,675 samples in 4 domains.
Paper verification: Db=-0.9421 matches published -0.9420.

Copyright (c) 2026 Insight137 (insight137.com)
License: CC BY-NC-ND 4.0

Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Roger Yau"
__license__ = "CC BY-NC-ND 4.0"

import math
from functools import lru_cache
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("insight137.eap")


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

EPSILON = 1e-15           # Numerical floor for log computations
PROB_FLOOR = 0.01         # Minimum probability to prevent division by zero
PROB_CEILING = 0.99       # Maximum probability clamp
MIN_WINDOW_SIZE = 2       # Minimum sliding window for Ψ₃
DEFAULT_WINDOW_SIZE = 3   # Default sliding window for interference
MIN_AGENTS_PSI4 = 2       # Minimum agents for Ψ₄ computation
PHASE_BIAS_CENTER = 0.5   # Center point for bias phase mapping


class PsiMethod(str, Enum):
    """Enumeration of supported computation methods."""
    HUANG_2019 = "huang_2019"
    CLASSICAL = "classical"


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PsiProfile:
    """Immutable 4-dimensional entropy profile.

    Attributes:
        psi_1: Informational entropy (Shannon/Deng).
        psi_2: Behavioral entropy (Huang interference magnitude).
        psi_3: Adaptive entropy (interference volatility over time).
        psi_4: Relational entropy (cross-agent decision diversity).
        belief_degree: Raw Huang belief degree (D_b).
        method: Computation method identifier.
    """
    psi_1: float
    psi_2: float
    psi_3: float
    psi_4: float
    belief_degree: float = 0.0
    method: str = PsiMethod.HUANG_2019.value

    def to_dict(self) -> Dict[str, Union[float, str]]:
        """Serialize to dictionary for JSON/API responses."""
        return {
            "psi_1": round(self.psi_1, 6),
            "psi_2": round(self.psi_2, 6),
            "psi_3": round(self.psi_3, 6),
            "psi_4": round(self.psi_4, 6),
            "belief_degree": round(self.belief_degree, 6),
            "method": self.method,
        }


@dataclass(frozen=True)
class ConditionalProbability:
    """Validated conditional probability pair for QLBN computation.

    Attributes:
        p_given_a_true: P(B=outcome | A=True), clamped to [PROB_FLOOR, PROB_CEILING].
        p_given_a_false: P(B=outcome | A=False), clamped to [PROB_FLOOR, PROB_CEILING].
    """
    p_given_a_true: float
    p_given_a_false: float

    def __post_init__(self):
        if not (0.0 <= self.p_given_a_true <= 1.0):
            raise ValueError(
                f"p_given_a_true must be in [0, 1], got {self.p_given_a_true}"
            )
        if not (0.0 <= self.p_given_a_false <= 1.0):
            raise ValueError(
                f"p_given_a_false must be in [0, 1], got {self.p_given_a_false}"
            )


# ═══════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def _validate_probability(value: float, name: str) -> float:
    """Validate and clamp a probability value to [0, 1]."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return max(0.0, min(float(value), 1.0))


def _validate_probability_pair(
    p_true: float, p_false: float
) -> Tuple[float, float]:
    """Validate prior probability pair sums to ~1.0."""
    p_true = _validate_probability(p_true, "p_a_true")
    p_false = _validate_probability(p_false, "p_a_false")
    total = p_true + p_false
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Prior probabilities must sum to ~1.0, got {total:.4f}"
        )
    return p_true, p_false


def _validate_sequence(
    values: Sequence[float], name: str, min_length: int = 1
) -> np.ndarray:
    """Validate and convert a numeric sequence to numpy array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got {arr.ndim}D")
    if len(arr) < min_length:
        raise ValueError(
            f"{name} requires >= {min_length} elements, got {len(arr)}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")
    return arr


def _coerce_list_to_conditionals(
    data: list,
) -> Dict[str, Dict[str, float]]:
    """Convert a list-of-lists to the dict-of-dicts conditionals format.

    Accepts [[p_true_0, p_false_0], [p_true_1, p_false_1], ...] and
    returns {"outcome_0": {"p_given_a_true": ..., "p_given_a_false": ...}, ...}.

    Raises:
        TypeError: If elements are not 2-element sequences of numbers.
    """
    result: Dict[str, Dict[str, float]] = {}
    for i, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError(
                f"conditionals[{i}] must be a [p_true, p_false] pair, "
                f"got {type(item).__name__}. "
                "Pass a dict for named outcomes: "
                "{'cooperate': {'p_given_a_true': 0.8, 'p_given_a_false': 0.3}}"
            )
        result[f"outcome_{i}"] = {
            "p_given_a_true": float(item[0]),
            "p_given_a_false": float(item[1]),
        }
    logger.info(
        "Converted list format to dict. For named outcomes, pass a dict: "
        "{'cooperate': {'p_given_a_true': 0.8, 'p_given_a_false': 0.3}}"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
# Ψ₁: INFORMATIONAL ENTROPY
# Reference: Deng (2016), Chaos Solitons & Fractals, 91, 549-553.
#            Shannon (1948), Bell System Technical Journal, 27(3).
# ═══════════════════════════════════════════════════════════════════════

def deng_entropy(masses: List[Tuple[int, float]]) -> float:
    """Compute Deng entropy for belief function evidence.

    Generalizes Shannon entropy to handle imprecise evidence with
    multi-element focal sets. When all focal elements are singletons
    (|A|=1), reduces exactly to Shannon entropy.

    Args:
        masses: List of (cardinality, mass) tuples where
                cardinality = |A| (size of focal element),
                mass = m(A) (basic probability assignment).

    Returns:
        Deng entropy in bits.

    Raises:
        ValueError: If masses are empty or contain invalid values.
    """
    if not masses:
        raise ValueError("masses must be non-empty")
    entropy = 0.0
    for cardinality, mass in masses:
        if cardinality < 1:
            raise ValueError(f"Cardinality must be >= 1, got {cardinality}")
        if mass < 0.0:
            raise ValueError(f"Mass must be >= 0, got {mass}")
        if mass > EPSILON:
            if cardinality < 1023:
                num_subsets = (2 ** cardinality) - 1
                entropy -= mass * math.log2(mass / num_subsets)
            else:
                # For large cardinalities, log2(2^card - 1) ≈ card
                # to avoid OverflowError when converting bigint to float.
                entropy -= mass * (math.log2(mass) - cardinality)
    return float(entropy)


def shannon_entropy(probs: Union[np.ndarray, Sequence[float]]) -> float:
    """Compute Shannon entropy of a probability distribution.

    Special case of Deng entropy where all focal elements are
    singletons (|A|=1 for all A).

    Args:
        probs: Probability distribution (must sum to ~1.0).

    Returns:
        Shannon entropy in bits.

    Raises:
        ValueError: If probs contains invalid values.
    """
    p = np.asarray(probs, dtype=np.float64)
    if np.any(p < -EPSILON):
        raise ValueError("Probabilities must be non-negative")
    p = p[p > EPSILON]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


# ═══════════════════════════════════════════════════════════════════════
# Ψ₂: BEHAVIORAL ENTROPY (Huang Interference)
# Reference: Huang, Yang, Jiang (2019), Applied Mathematics and
#            Computation, 347, 417-428. arXiv: 1709.02844.
# ═══════════════════════════════════════════════════════════════════════

def _compute_amplitude_vectors(
    p_b_given_a_true: float,
    p_b_given_a_false: float,
    p_a_true: float,
    p_a_false: float,
) -> Tuple[float, float]:
    """Compute amplitude vector [alpha, beta] for a QLBN outcome.

    From Huang et al. (2019) Eq. 17:
        alpha = sqrt(P(B|A=T)) * sqrt(P(A=T))
        beta  = sqrt(P(B|A=F)) * sqrt(P(A=F))

    Returns:
        Tuple of (alpha, beta) amplitude values.
    """
    alpha = math.sqrt(max(p_b_given_a_true, 0.0)) * math.sqrt(max(p_a_true, 0.0))
    beta = math.sqrt(max(p_b_given_a_false, 0.0)) * math.sqrt(max(p_a_false, 0.0))
    return alpha, beta


def _belief_distance(alpha: float, beta: float) -> float:
    """Compute Belief Distance per Huang et al. (2019) Eq. 18.

    Bd = |alpha + (alpha - beta) / |alpha + beta - 1||

    Alpha must be the amplitude closer to 0.5. If not, this
    function swaps them automatically.

    Reference: Huang et al. (2019), Definition 4.2.
    """
    if abs(alpha - 0.5) >= abs(beta - 0.5):
        alpha, beta = beta, alpha
    denominator = abs(alpha + beta - 1.0)
    if denominator < EPSILON:
        return abs(alpha)
    bd = abs(alpha + (alpha - beta) / denominator)
    return max(0.0, min(bd, 1.0))


def belief_degree_huang(
    outcomes: List[Dict[str, float]],
    p_a_true: float = 0.5,
    p_a_false: float = 0.5,
    n_unobserved: int = 1,
    **kwargs,
) -> float:
    """Compute Huang Belief Degree (interference cos theta).

    This is the primary function for computing quantum-like
    interference. Replaces the static QDT value (cos theta = -0.25)
    and the Moreira-Wichert similarity heuristic with a dynamic,
    data-driven computation grounded in Deng entropy.

    Args:
        outcomes: List of dicts, each with keys:
            'p_given_a_true':  P(B=outcome | A=True)
            'p_given_a_false': P(B=outcome | A=False)
        p_a_true: Prior P(A=True), default 0.5.
        p_a_false: Prior P(A=False), default 0.5.
        n_unobserved: Number of unobserved variables |Ai|.

    Returns:
        Belief Degree D_b (negative = destructive interference).

    Raises:
        ValueError: If outcomes is empty or priors invalid.

    Reference: Huang et al. (2019), Lemma 4.1, Eq. 19.
    """
    if "priors" in kwargs:
        raise TypeError(
            "Unknown argument 'priors'. "
            "Use p_a_true=0.5, p_a_false=0.5 instead."
        )
    if kwargs:
        raise TypeError(
            f"Unexpected keyword arguments: {', '.join(kwargs.keys())}"
        )
    if not outcomes:
        raise ValueError("outcomes must be non-empty")
    if not isinstance(outcomes, list):
        raise TypeError(
            f"outcomes must be a list of dicts, got {type(outcomes).__name__}"
        )
    for i, item in enumerate(outcomes):
        if not isinstance(item, dict):
            raise TypeError(
                f"outcomes[{i}] must be a dict, got {type(item).__name__}"
            )
    p_a_true, p_a_false = _validate_probability_pair(p_a_true, p_a_false)
    if n_unobserved < 1:
        raise ValueError(f"n_unobserved must be >= 1, got {n_unobserved}")

    belief_distances = []
    for outcome in outcomes:
        if "priors" in outcome and "p_given_a_true" not in outcome:
            priors = outcome["priors"]
            outcome = {
                "p_given_a_true": float(priors[0]),
                "p_given_a_false": float(priors[1]),
            }
        p_true = float(outcome.get("p_given_a_true", 0.5))
        p_false = float(outcome.get("p_given_a_false", 0.5))
        alpha, beta = _compute_amplitude_vectors(
            p_true, p_false, p_a_true, p_a_false
        )
        bd = _belief_distance(alpha, beta)
        belief_distances.append(bd)

    num_subsets = (2 ** n_unobserved) - 1
    db = 0.0
    for bd in belief_distances:
        if bd > EPSILON:
            db += bd * math.log2(bd / num_subsets)
    return db


# ═══════════════════════════════════════════════════════════════════════
# QUANTUM TOTAL PROBABILITY
# Reference: Moreira & Wichert (2016), Frontiers in Psychology, 7, 11.
# ═══════════════════════════════════════════════════════════════════════

def quantum_probability(
    conditionals: Dict[str, Dict[str, float]],
    p_a_true: float = 0.5,
    p_a_false: float = 0.5,
    n_unobserved: int = 1,
    **kwargs,
) -> Dict[str, float]:
    """Compute quantum-like probabilities with Huang interference.

    Applies the quantum law of total probability:
        P_q(B) = P(B|A)P(A) + P(B|~A)P(~A) +
                 2 * D_b * sqrt(P(B|A)P(A)P(B|~A)P(~A))

    With Born rule normalization ensuring sum(P_q) = 1.

    Args:
        conditionals: Dict mapping outcome names to
            {'p_given_a_true': float, 'p_given_a_false': float}.
        p_a_true: Prior P(A=True).
        p_a_false: Prior P(A=False).
        n_unobserved: Number of unobserved variables.

    Returns:
        Dict mapping outcome names to normalized quantum probabilities.

    Raises:
        ValueError: If conditionals is empty or priors invalid.
    """
    if "priors" in kwargs:
        raise TypeError(
            "Unknown argument 'priors'. "
            "Use p_a_true=0.5, p_a_false=0.5 instead."
        )
    if kwargs:
        raise TypeError(
            f"Unexpected keyword arguments: {', '.join(kwargs.keys())}"
        )
    if isinstance(conditionals, list):
        conditionals = _coerce_list_to_conditionals(conditionals)
    if not conditionals:
        raise ValueError("conditionals must be non-empty")
    p_a_true, p_a_false = _validate_probability_pair(p_a_true, p_a_false)

    outcomes_list = [
        {"p_given_a_true": v["p_given_a_true"],
         "p_given_a_false": v["p_given_a_false"]}
        for v in conditionals.values()
    ]
    db = belief_degree_huang(
        outcomes_list, p_a_true, p_a_false, n_unobserved
    )

    raw_probs: Dict[str, float] = {}
    for name, cond in conditionals.items():
        p_t = float(cond["p_given_a_true"])
        p_f = float(cond["p_given_a_false"])
        classical = p_t * p_a_true + p_f * p_a_false
        interference = 2.0 * db * math.sqrt(
            p_t * p_a_true * p_f * p_a_false
        )
        raw_probs[name] = classical + interference

    total = sum(raw_probs.values())
    if total <= 0:
        logger.warning("Negative total probability; falling back to classical")
        fallback: Dict[str, float] = {}
        for name, cond in conditionals.items():
            fallback[name] = (
                cond["p_given_a_true"] * p_a_true +
                cond["p_given_a_false"] * p_a_false
            )
        fb_total = sum(fallback.values())
        if fb_total <= 0:
            raise ValueError(
                "Total classical probability is zero — cannot compute fallback"
            )
        return {k: v / fb_total for k, v in fallback.items()}

    return {k: v / total for k, v in raw_probs.items()}


# ═══════════════════════════════════════════════════════════════════════
# Ψ₃: ADAPTIVE ENTROPY (Interference Volatility)
# Novel contribution — measures rate of change in behavioral
# interference patterns over temporal sequences.
# ═══════════════════════════════════════════════════════════════════════

# Pre-computed constants to avoid repeated computation per call
_SQRT_HALF = math.sqrt(0.5)
_EMPTY_ARRAY = np.array([], dtype=np.float64)
_HUANG_METHOD = "huang_2019"  # cached string avoids PsiMethod enum lookup per call


def _belief_distance_vec(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Vectorized Belief Distance (Huang et al. 2019, Eq. 18).

    Computes belief distance for arrays of alpha/beta pairs in a
    single numpy pass. Equivalent to calling _belief_distance on
    each pair, but ~100x faster for large arrays.
    """
    # Swap so alpha is the amplitude closer to 0.5
    dist_a = np.abs(alpha - 0.5)
    dist_b = np.abs(beta - 0.5)
    swap = dist_a >= dist_b
    a = np.where(swap, beta, alpha)
    b = np.where(swap, alpha, beta)

    denom = np.abs(a + b - 1.0)
    small = denom < EPSILON

    # Safe division: use 1.0 for small denominators (overwritten by np.where)
    safe_denom = np.where(small, 1.0, denom)
    bd = np.where(small, np.abs(a), np.abs(a + (a - b) / safe_denom))
    return np.clip(bd, 0.0, 1.0)

def _compute_sliding_interferences(
    values: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> np.ndarray:
    """Compute Huang interference over sliding windows of a sequence.

    For each window position, normalizes values to a probability
    distribution, forms a binary conditional, and computes the
    Huang Belief Degree.

    Vectorized: computes all windows in a single numpy pass instead
    of calling belief_degree_huang per window.

    Args:
        values: 1-D array of positive values (e.g., message lengths).
        window_size: Number of elements per window (>= 2).

    Returns:
        numpy array of D_b values, one per valid window position.
        Empty array if no valid windows.
    """
    n = len(values)
    effective_window = min(window_size, n)
    if effective_window < MIN_WINDOW_SIZE:
        return _EMPTY_ARRAY

    num_windows = n - effective_window + 1

    # Vectorized window sums via cumsum
    cumsum = np.empty(n + 1)
    cumsum[0] = 0.0
    np.cumsum(values, out=cumsum[1:])
    window_sums = cumsum[effective_window:] - cumsum[:num_windows]

    # Filter windows with sum > EPSILON
    valid = window_sums > EPSILON
    if not np.any(valid):
        return _EMPTY_ARRAY

    valid_indices = np.nonzero(valid)[0]
    valid_sums = window_sums[valid_indices]

    # Normalized p1, p2 for each valid window, clamped to [PROB_FLOOR, PROB_CEILING]
    p1 = np.clip(values[valid_indices] / valid_sums, PROB_FLOOR, PROB_CEILING)
    p2 = np.clip(values[valid_indices + 1] / valid_sums, PROB_FLOOR, PROB_CEILING)

    # Vectorized belief distance for both outcomes.
    # With default priors (0.5, 0.5), n_unobserved=1:
    #   alpha = sqrt(p_true * 0.5), beta = sqrt(p_false * 0.5)
    #   num_subsets = 1, so db = sum(bd * log2(bd))
    sqrt_half = _SQRT_HALF
    sp1 = np.sqrt(p1)
    sp2 = np.sqrt(p2)
    bd1 = _belief_distance_vec(sp1 * sqrt_half, sp2 * sqrt_half)
    bd2 = _belief_distance_vec(np.sqrt(1.0 - p1) * sqrt_half, np.sqrt(1.0 - p2) * sqrt_half)

    # Belief degree: sum of bd * log2(bd) for each outcome (num_subsets=1)
    db = np.zeros(len(valid_indices))
    mask1 = bd1 > EPSILON
    db[mask1] += bd1[mask1] * np.log2(bd1[mask1])
    mask2 = bd2 > EPSILON
    db[mask2] += bd2[mask2] * np.log2(bd2[mask2])

    return db


def compute_psi3(
    values: Union[np.ndarray, Sequence[float]],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> float:
    """Compute Psi3 (Adaptive Entropy) from a behavioral sequence.

    Psi3 is the standard deviation of Huang interference values
    computed over a sliding temporal window. High Psi3 indicates
    rapid changes in behavioral patterns — the system's interference
    profile is volatile.

    This dimension is novel to this work and is validated with
    Cohen's d = 1.024 (SHADE-Arena) and d = 0.644 (Palisade).

    Args:
        values: Sequence of positive values (message lengths, etc.).
        window_size: Sliding window size (default: 3).

    Returns:
        Psi3 value (standard deviation of interference series).
        Returns 0.0 if sequence is too short.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < MIN_WINDOW_SIZE + 1:
        return 0.0
    interferences = _compute_sliding_interferences(arr, window_size)
    if len(interferences) < 2:
        return 0.0
    return float(interferences.std())


# ═══════════════════════════════════════════════════════════════════════
# Ψ₄: RELATIONAL ENTROPY (BEQBN Entanglement Witness)
# Reference: Meghdadi, Akbarzadeh-T, Javidan (2022), Applied Soft
#            Computing, 118, 108528.
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def _triu_indices_cached(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cached upper-triangle indices for pairwise computation."""
    return np.triu_indices(n, k=1)


def compute_psi4(
    agent_probabilities: Sequence[float],
) -> float:
    """Compute Psi4 (Relational Entropy) from cross-agent decisions.

    Measures decision entanglement across multiple agents or models
    responding to the same scenario. High Psi4 indicates diverse
    decision patterns; low Psi4 indicates consensus.

    Validated on Palisade data: r = 0.9983 correlation with
    cross-model disagreement (100 scenarios, 11 models).

    Args:
        agent_probabilities: Sequence of probability values, one per
            agent/model. For binary decisions (bypass/comply), use
            1.0 for bypass, 0.01 for comply.

    Returns:
        Psi4 value (std of pairwise phase-weighted cosines).
        Returns 0.0 if fewer than 2 agents.

    Raises:
        ValueError: If any probability is outside [0, 1].

    Note:
        There is no ``compute_psi4_from_conditionals`` — use
        ``compute_psi4()`` with a list of per-agent probabilities.
    """
    # Batch validation: type-check then validate with numpy
    for i, p in enumerate(agent_probabilities):
        if not isinstance(p, (int, float, np.integer, np.floating)):
            raise TypeError(f"agent_{i} must be numeric, got {type(p).__name__}")
    arr = np.asarray(agent_probabilities, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        # Find the first bad value for a clear error message
        for i, p in enumerate(agent_probabilities):
            if math.isnan(p) or math.isinf(p):
                raise ValueError(f"agent_{i} must be finite, got {p}")
    n = len(arr)
    if n < MIN_AGENTS_PSI4:
        return 0.0

    # Vectorized: clamp, compute phases, then pairwise via broadcasting
    clamped = np.clip(arr, PROB_FLOOR, 1.0)
    phases = np.arcsin(np.clip((clamped - PHASE_BIAS_CENTER) / PHASE_BIAS_CENTER, -1.0, 1.0))

    i_idx, j_idx = _triu_indices_cached(n)
    cos_diff = np.cos(phases[i_idx] - phases[j_idx])
    weight = np.sqrt(clamped[i_idx] * clamped[j_idx])
    pairwise = cos_diff * weight

    return float(pairwise.std())


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════

def compute_psi_from_sequence(
    values: Union[np.ndarray, Sequence[float]],
    window_size: int = DEFAULT_WINDOW_SIZE,
    agent_decisions: Optional[Sequence[float]] = None,
) -> PsiProfile:
    """Compute full 4D Psi profile from a behavioral sequence.

    Primary entry point for AI behavioral data (message lengths,
    keystroke timings, action sequences). Computes all four
    dimensions from a single temporal sequence.

    Args:
        values: Sequence of positive values representing the
            behavioral signal (e.g., message lengths in a transcript).
        window_size: Sliding window for interference computation.
        agent_decisions: Optional sequence of agent probabilities
            for Psi4 computation (one per agent/model). If None,
            Psi4 is set to 0.0.

    Returns:
        Complete PsiProfile with all four dimensions.

    Raises:
        ValueError: If values is empty or contains invalid data.
    """
    arr = _validate_sequence(values, "values", min_length=1)

    # Ψ₁: Shannon entropy of value distribution
    total = arr.sum()
    if total > EPSILON:
        dist = arr / total
        psi_1 = shannon_entropy(dist)
    else:
        psi_1 = 0.0

    # Ψ₂: Mean Huang interference over sliding windows
    interferences = _compute_sliding_interferences(arr, window_size)
    n_interf = len(interferences)
    if n_interf > 0:
        db = float(interferences.mean())
        psi_2 = abs(db)
    else:
        psi_2 = 0.0
        db = 0.0

    # Ψ₃: Volatility of interference
    psi_3 = float(interferences.std()) if n_interf >= 2 else 0.0

    # Ψ₄: Relational entropy (requires multi-agent data)
    psi_4 = 0.0
    if agent_decisions is not None and len(agent_decisions) >= MIN_AGENTS_PSI4:
        psi_4 = compute_psi4(agent_decisions)

    return PsiProfile(
        psi_1=round(psi_1, 6),
        psi_2=round(psi_2, 6),
        psi_3=round(psi_3, 6),
        psi_4=round(psi_4, 6),
        belief_degree=round(db, 6),
        method=_HUANG_METHOD,
    )


def compute_psi_from_conditionals(
    conditionals: Dict[str, Dict[str, float]],
    p_a_true: float = 0.5,
    p_a_false: float = 0.5,
    **kwargs,
) -> PsiProfile:
    """Compute Psi profile from QLBN conditional probabilities.

    Entry point for quantum cognition data (survey order effects,
    Prisoner's Dilemma, etc.). Computes Psi1 and Psi2 from the
    conditional probability structure.

    Args:
        conditionals: Dict mapping outcome names to
            {'p_given_a_true': float, 'p_given_a_false': float}.
        p_a_true: Prior P(A=True).
        p_a_false: Prior P(A=False).

    Returns:
        PsiProfile with Psi1, Psi2, and belief_degree populated.
        Psi3 and Psi4 are 0.0 (require temporal/multi-agent data).

    Raises:
        ValueError: If conditionals is empty or priors invalid.
    """
    if "priors" in kwargs:
        raise TypeError(
            "Unknown argument 'priors'. "
            "Use p_a_true=0.5, p_a_false=0.5 instead."
        )
    if kwargs:
        raise TypeError(
            f"Unexpected keyword arguments: {', '.join(kwargs.keys())}"
        )
    if isinstance(conditionals, list):
        conditionals = _coerce_list_to_conditionals(conditionals)
    if not conditionals:
        raise ValueError("conditionals must be non-empty")
    p_a_true, p_a_false = _validate_probability_pair(p_a_true, p_a_false)

    outcomes_list = [
        {"p_given_a_true": v["p_given_a_true"],
         "p_given_a_false": v["p_given_a_false"]}
        for v in conditionals.values()
    ]

    # Ψ₁: Shannon entropy of classical marginal
    classical_probs = []
    for cond in conditionals.values():
        p = (cond["p_given_a_true"] * p_a_true +
             cond["p_given_a_false"] * p_a_false)
        classical_probs.append(p)
    psi_1 = shannon_entropy(np.array(classical_probs))

    # Ψ₂: Huang interference magnitude
    db = belief_degree_huang(outcomes_list, p_a_true, p_a_false)
    psi_2 = abs(db)

    return PsiProfile(
        psi_1=round(psi_1, 6),
        psi_2=round(psi_2, 6),
        psi_3=0.0,
        psi_4=0.0,
        belief_degree=round(db, 6),
        method=_HUANG_METHOD,
    )


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def cohens_d(
    group_a: Union[np.ndarray, Sequence[float]],
    group_b: Union[np.ndarray, Sequence[float]],
) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation (Hedges' approach for unequal
    sample sizes).

    Args:
        group_a: First group of observations.
        group_b: Second group of observations.

    Returns:
        Cohen's d (positive = group_a > group_b).

    Raises:
        ValueError: If either group has fewer than 2 elements.
    """
    a = _validate_sequence(group_a, "group_a", min_length=2)
    b = _validate_sequence(group_b, "group_b", min_length=2)
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )
    if pooled_std < EPSILON:
        return 0.0
    return float((mean_a - mean_b) / pooled_std)


# ═══════════════════════════════════════════════════════════════════════
# USER-FRIENDLY API
# Convenience functions for researchers encountering the library for
# the first time. These delegate to the core functions above.
# ═══════════════════════════════════════════════════════════════════════

def _interpret_level(value: float, thresholds: List[Tuple[float, str]]) -> str:
    """Return the interpretation string for the first threshold exceeded."""
    for limit, label in thresholds:
        if value < limit:
            return label
    return thresholds[-1][1]


_PSI1_BANDS = [(1.0, "low uncertainty"), (3.0, "moderate uncertainty"), (float("inf"), "high uncertainty")]
_PSI2_BANDS = [(0.3, "low interference (mechanical/predictable)"), (0.7, "moderate interference (natural)"), (float("inf"), "high interference (chaotic/volatile)")]
_PSI3_BANDS = [(0.1, "stable (rigid/mechanical)"), (0.4, "moderate adaptation (natural decision-making)"), (float("inf"), "high adaptation (active mode transitions)")]
_PSI4_BANDS = [(0.05, "strong consensus"), (0.2, "moderate agreement"), (float("inf"), "high divergence")]


def summary(profile) -> str:
    """Generate a plain-English interpretation of a Psi profile.

    Accepts a PsiProfile or a dict with a 'profile' key (e.g., output
    from quick_analyze).

    Args:
        profile: PsiProfile instance, or dict containing 'profile' key.

    Returns:
        Multi-line string with interpretation of each dimension.

    Raises:
        TypeError: If profile is not a PsiProfile or compatible dict.
    """
    if isinstance(profile, dict):
        profile = profile.get("profile", profile)
    if not isinstance(profile, PsiProfile):
        raise TypeError(
            f"Expected PsiProfile or dict with 'profile' key, "
            f"got {type(profile).__name__}"
        )
    lines = [
        f"Psi1 = {profile.psi_1:.4f}  ({_interpret_level(profile.psi_1, _PSI1_BANDS)})",
        f"Psi2 = {profile.psi_2:.4f}  ({_interpret_level(profile.psi_2, _PSI2_BANDS)})",
        f"Psi3 = {profile.psi_3:.4f}  ({_interpret_level(profile.psi_3, _PSI3_BANDS)})",
        f"Psi4 = {profile.psi_4:.4f}  ({_interpret_level(profile.psi_4, _PSI4_BANDS)})",
    ]
    return "\n".join(lines)


class ExampleData:
    """Built-in sample datasets for immediate experimentation.

    Access via the module-level ``examples`` instance::

        from insight137_eap import examples
        profile = compute_psi_from_conditionals(examples.prisoners_dilemma)
    """

    @property
    def prisoners_dilemma(self) -> Dict[str, Dict[str, float]]:
        """Average Prisoner's Dilemma conditionals from Huang et al. (2019) Table 2.

        P(Defect|opponent defected)=0.87, P(Defect|opponent cooperated)=0.74.
        """
        return {
            "defect": {"p_given_a_true": 0.87, "p_given_a_false": 0.74},
            "cooperate": {"p_given_a_true": 0.13, "p_given_a_false": 0.26},
        }

    @property
    def human_keystrokes(self) -> List[float]:
        """50 realistic inter-keystroke intervals (ms) with natural variability.

        Seeded RNG for reproducibility. Typical human range: 60-200ms.
        """
        rng = np.random.RandomState(42)
        return [float(x) for x in rng.normal(120, 35, 50).clip(40, 250)]

    @property
    def bot_keystrokes(self) -> List[float]:
        """50 near-constant inter-keystroke intervals (ms) simulating a bot.

        Mechanical regularity with minimal jitter (std=1ms).
        """
        rng = np.random.RandomState(99)
        return [float(x) for x in rng.normal(50, 1, 50).clip(45, 55)]

    @property
    def order_effects(self) -> Dict[str, Dict[str, float]]:
        """Survey order-effect conditionals (Clinton-Gore style).

        Demonstrates question-order interference in political surveys.
        Based on Busemeyer & Bruza (2012, Ch. 9) reported patterns.
        """
        return {
            "approve": {"p_given_a_true": 0.69, "p_given_a_false": 0.56},
            "disapprove": {"p_given_a_true": 0.31, "p_given_a_false": 0.44},
        }

    @property
    def multi_agent_diverse(self) -> List[float]:
        """5 agents with diverse bypass decisions (high disagreement).

        3 bypass (1.0), 2 comply (0.01). Psi4 should be high.
        """
        return [1.0, 1.0, 1.0, 0.01, 0.01]

    @property
    def multi_agent_consensus(self) -> List[float]:
        """5 agents with near-identical decisions (consensus).

        All comply with minor variation. Psi4 should be near zero.
        """
        return [0.05, 0.03, 0.04, 0.06, 0.02]


examples = ExampleData()


def quick_analyze(data, **kwargs) -> dict:
    """Auto-detect input type and run the appropriate analysis.

    Accepts:
        - list/array of numbers -> compute_psi_from_sequence
        - dict with outcome keys -> compute_psi_from_conditionals
        - Otherwise raises TypeError with guidance

    Args:
        data: Input data (sequence, conditionals dict, or list-of-lists).
        **kwargs: Passed through to the underlying function
            (e.g., window_size, p_a_true, p_a_false, agent_decisions).

    Returns:
        Dict with keys:
            'profile': PsiProfile,
            'summary': str (plain-English interpretation),
            'input_type': str ('sequence' or 'conditionals'),
            'raw': dict of all computed values

    Raises:
        TypeError: If data format is not recognized.
    """
    # Dict with p_given_a_true values → conditionals
    if isinstance(data, dict):
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, dict) and "p_given_a_true" in first_val:
            profile = compute_psi_from_conditionals(data, **kwargs)
            return {
                "profile": profile,
                "summary": summary(profile),
                "input_type": "conditionals",
                "raw": profile.to_dict(),
            }
        raise TypeError(
            "Dict values must be dicts with 'p_given_a_true' and "
            "'p_given_a_false' keys."
        )

    # List/array of numbers → sequence
    if isinstance(data, (list, tuple, np.ndarray)):
        # Check if it looks like a list-of-lists (conditionals shorthand)
        if (len(data) > 0
                and isinstance(data[0], (list, tuple))
                and len(data[0]) == 2):
            conditionals = _coerce_list_to_conditionals(data)
            profile = compute_psi_from_conditionals(conditionals, **kwargs)
            return {
                "profile": profile,
                "summary": summary(profile),
                "input_type": "conditionals",
                "raw": profile.to_dict(),
            }
        # Extract agent_decisions from kwargs if present
        agent_decisions = kwargs.pop("agent_decisions", None)
        window_size = kwargs.pop("window_size", DEFAULT_WINDOW_SIZE)
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments: {', '.join(kwargs.keys())}"
            )
        profile = compute_psi_from_sequence(
            data, window_size=window_size, agent_decisions=agent_decisions,
        )
        return {
            "profile": profile,
            "summary": summary(profile),
            "input_type": "sequence",
            "raw": profile.to_dict(),
        }

    raise TypeError(
        f"Cannot analyze {type(data).__name__}. Accepted formats:\n"
        "  - list/array of numbers: [10, 20, 30, 40]\n"
        "  - dict of conditionals: {'outcome': {'p_given_a_true': 0.8, "
        "'p_given_a_false': 0.3}}\n"
        "  - list of [p_true, p_false] pairs: [[0.8, 0.2], [0.3, 0.7]]"
    )


def compare(
    a,
    b,
    labels: Optional[Tuple[str, str]] = None,
) -> dict:
    """Compare two datasets and return a complete comparison.

    Accepts two sequences (lists/arrays) or two conditional dicts.
    Auto-detects input type using the same logic as quick_analyze.

    Args:
        a, b: Two sequences or two conditional dicts.
        labels: Optional tuple of names, e.g. ("human", "bot").

    Returns:
        Dict with keys:
            'profile_a': PsiProfile,
            'profile_b': PsiProfile,
            'differences': dict mapping each Ψ dimension to its
                absolute difference,
            'largest_difference': str (which Ψ dimension differs most),
            'verdict': str (plain-English summary of comparison)
    """
    label_a = labels[0] if labels else "A"
    label_b = labels[1] if labels else "B"

    result_a = quick_analyze(a)
    result_b = quick_analyze(b)
    pa = result_a["profile"]
    pb = result_b["profile"]

    diffs = {
        "psi_1": abs(pa.psi_1 - pb.psi_1),
        "psi_2": abs(pa.psi_2 - pb.psi_2),
        "psi_3": abs(pa.psi_3 - pb.psi_3),
        "psi_4": abs(pa.psi_4 - pb.psi_4),
    }

    largest_key = max(diffs, key=diffs.get)
    largest_val = diffs[largest_key]

    if largest_val < 0.2:
        magnitude = "negligible"
    elif largest_val < 0.5:
        magnitude = "small"
    elif largest_val < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    dim_names = {
        "psi_1": "informational entropy",
        "psi_2": "behavioral interference",
        "psi_3": "adaptive volatility",
        "psi_4": "relational divergence",
    }

    verdict = (
        f"{magnitude.capitalize()} difference between {label_a} and "
        f"{label_b}. Largest gap: {largest_key} "
        f"({dim_names[largest_key]}) differs by {largest_val:.4f}."
    )

    return {
        "profile_a": pa,
        "profile_b": pb,
        "differences": diffs,
        "largest_difference": largest_key,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# Optional matplotlib integration. All functions lazy-import so the
# library works without matplotlib installed.
# ═══════════════════════════════════════════════════════════════════════

_TEAL = "#4fc4c0"
_AMBER = "#f59e0b"
_DIM_LABELS = ["Psi1", "Psi2", "Psi3", "Psi4"]


def _resolve_profile(data) -> "PsiProfile":
    """Convert various inputs to a PsiProfile."""
    if isinstance(data, PsiProfile):
        return data
    if isinstance(data, dict):
        p = data.get("profile", data)
        if isinstance(p, PsiProfile):
            return p
    result = quick_analyze(data)
    return result["profile"]


def _get_psi_values(profile: "PsiProfile") -> List[float]:
    """Extract [psi_1, psi_2, psi_3, psi_4] from a profile."""
    return [profile.psi_1, profile.psi_2, profile.psi_3, profile.psi_4]


def plot(profile, title=None, save_path=None, show=True):
    """Plot a 4-axis radar chart of a Psi profile.

    Args:
        profile: PsiProfile, dict from quick_analyze(), or raw data.
        title: Optional chart title.
        save_path: If provided, save as PNG/PDF (detected from extension).
        show: If True, call plt.show().

    Requires matplotlib (optional dependency).

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    p = _resolve_profile(profile)
    values = _get_psi_values(p)
    values_closed = values + [values[0]]

    angles = [n / 4.0 * 2.0 * math.pi for n in range(4)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    labels = [f"{name}\n{val:.4f}" for name, val in zip(_DIM_LABELS, values)]
    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], labels)

    ax.plot(angles, values_closed, color=_TEAL, linewidth=2)
    ax.fill(angles, values_closed, color=_TEAL, alpha=0.25)

    ax.spines["polar"].set_visible(False)
    ax.set_title(title or "Psi Profile", pad=20, fontsize=14)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_compare(a, b, labels=None, title=None, save_path=None, show=True):
    """Overlay two Psi profiles on one radar chart for comparison.

    Args:
        a, b: PsiProfiles, sequences, or conditional dicts.
        labels: Tuple of names, e.g. ("Human", "Bot").
        title: Optional chart title.
        save_path: If provided, save as PNG/PDF.
        show: If True, call plt.show().

    Requires matplotlib (optional dependency).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    pa = _resolve_profile(a)
    pb = _resolve_profile(b)
    label_a = labels[0] if labels else "A"
    label_b = labels[1] if labels else "B"

    vals_a = _get_psi_values(pa)
    vals_b = _get_psi_values(pb)
    vals_a_closed = vals_a + [vals_a[0]]
    vals_b_closed = vals_b + [vals_b[0]]

    angles = [n / 4.0 * 2.0 * math.pi for n in range(4)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        [a_ * 180 / math.pi for a_ in angles[:-1]], _DIM_LABELS,
    )

    ax.plot(angles, vals_a_closed, color=_TEAL, linewidth=2, label=label_a)
    ax.fill(angles, vals_a_closed, color=_TEAL, alpha=0.25)
    ax.plot(angles, vals_b_closed, color=_AMBER, linewidth=2, label=label_b)
    ax.fill(angles, vals_b_closed, color=_AMBER, alpha=0.25)

    ax.spines["polar"].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))

    comp = compare(a, b, labels=labels)
    ax.set_title(title or "Psi Profile Comparison", pad=20, fontsize=14)
    fig.text(
        0.5, 0.02, comp["verdict"],
        ha="center", fontsize=10, style="italic",
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_dimensions(a, b, labels=None, title=None, save_path=None, show=True):
    """Bar chart comparing each Psi dimension side by side.

    More readable than radar for publications. Shows each dimension
    as a grouped bar with values above bars.

    Args:
        a, b: PsiProfiles, sequences, or conditional dicts.
        labels: Tuple of names, e.g. ("Human", "Bot").
        title: Optional chart title.
        save_path: If provided, save as PNG/PDF.
        show: If True, call plt.show().

    Requires matplotlib (optional dependency).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    pa = _resolve_profile(a)
    pb = _resolve_profile(b)
    label_a = labels[0] if labels else "A"
    label_b = labels[1] if labels else "B"

    vals_a = _get_psi_values(pa)
    vals_b = _get_psi_values(pb)

    x = np.arange(4)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a = ax.bar(x - width / 2, vals_a, width, label=label_a, color=_TEAL)
    bars_b = ax.bar(x + width / 2, vals_b, width, label=label_b, color=_AMBER)

    ax.set_xticks(x)
    ax.set_xticklabels(_DIM_LABELS)
    ax.set_ylabel("Value")
    ax.set_title(title or "Psi Dimension Comparison", fontsize=14)
    ax.legend()

    for bar in bars_a:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
        )
    for bar in bars_b:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# EXPORT / INTEROPERABILITY
# CSV, JSON (stdlib), and MATLAB .mat (optional scipy).
# ═══════════════════════════════════════════════════════════════════════

def _profiles_to_rows(
    profiles, labels: Optional[List[str]] = None,
) -> List[dict]:
    """Convert one or more PsiProfiles to a list of row dicts."""
    if isinstance(profiles, PsiProfile):
        profiles = [profiles]
    if labels is None:
        labels = [f"profile_{i}" for i in range(len(profiles))]
    rows = []
    for label, p in zip(labels, profiles):
        d = p.to_dict()
        d["label"] = label
        rows.append(d)
    return rows


def to_csv(
    profiles, path: str, labels: Optional[List[str]] = None,
) -> None:
    """Export PsiProfiles to CSV.

    CSV format compatible with MATLAB readtable(), R read.csv(),
    Excel, and pandas.

    Args:
        profiles: Single PsiProfile or list of PsiProfiles.
        path: Output file path (.csv).
        labels: Optional list of row labels.

    Columns: label, psi_1, psi_2, psi_3, psi_4, belief_degree, method
    """
    import csv

    rows = _profiles_to_rows(profiles, labels)
    fieldnames = [
        "label", "psi_1", "psi_2", "psi_3", "psi_4",
        "belief_degree", "method",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d profiles to %s", len(rows), path)


def to_json(
    profiles,
    path: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> Optional[str]:
    """Export PsiProfiles to JSON.

    JSON format compatible with MATLAB jsondecode(), R jsonlite,
    and JavaScript.

    Args:
        profiles: Single PsiProfile or list of PsiProfiles.
        path: If provided, write to file. If None, return JSON string.
        labels: Optional list of labels.

    Returns:
        JSON string if path is None, otherwise None.
    """
    import json as _json

    rows = _profiles_to_rows(profiles, labels)
    text = _json.dumps(rows, indent=2)
    if path is not None:
        with open(path, "w") as f:
            f.write(text)
        logger.info("Wrote %d profiles to %s", len(rows), path)
        return None
    return text


def to_matlab(
    profiles,
    path: str,
    labels: Optional[List[str]] = None,
) -> None:
    """Export PsiProfiles to MATLAB-compatible .mat file.

    Creates a .mat file loadable with MATLAB's load() command.
    Variables: psi_1, psi_2, psi_3, psi_4, belief_degree (each as
    arrays), plus labels as a cell array of strings.

    Args:
        profiles: Single PsiProfile or list of PsiProfiles.
        path: Output file path (.mat).
        labels: Optional list of labels.

    Requires scipy (optional dependency).

    Raises:
        ImportError: If scipy is not installed.
    """
    try:
        from scipy.io import savemat
    except ImportError:
        raise ImportError(
            "scipy is required for MATLAB export. "
            "Install with: pip install scipy"
        )

    rows = _profiles_to_rows(profiles, labels)
    mat_dict = {
        "psi_1": np.array([r["psi_1"] for r in rows]),
        "psi_2": np.array([r["psi_2"] for r in rows]),
        "psi_3": np.array([r["psi_3"] for r in rows]),
        "psi_4": np.array([r["psi_4"] for r in rows]),
        "belief_degree": np.array([r["belief_degree"] for r in rows]),
        "labels": np.array([r["label"] for r in rows], dtype=object),
    }
    savemat(path, mat_dict)
    logger.info("Wrote %d profiles to %s", len(rows), path)


# ═══════════════════════════════════════════════════════════════════════
# BUILT-IN VERIFICATION
# Validates implementation against published results.
# ═══════════════════════════════════════════════════════════════════════

class VerificationError(Exception):
    """Raised when implementation verification fails."""
    pass


def verify_huang_paper() -> Dict[str, bool]:
    """Verify implementation against Huang et al. (2019) published values.

    Uses the average Prisoner's Dilemma data from Table 2:
        P(Defect|Defect) = 0.87, P(Defect|Cooperate) = 0.74
    Expected: D_b = -0.9420, P(Defect|Unknown) = 0.6926

    Returns:
        Dict with verification results for each checkpoint.

    Raises:
        VerificationError: If any critical check fails.
    """
    results: Dict[str, bool] = {}
    tolerance = 0.005

    conditionals = {
        "defect": {"p_given_a_true": 0.87, "p_given_a_false": 0.74},
        "cooperate": {"p_given_a_true": 0.13, "p_given_a_false": 0.26},
    }

    # Check 1: Belief Degree
    outcomes = [
        {"p_given_a_true": 0.87, "p_given_a_false": 0.74},
        {"p_given_a_true": 0.13, "p_given_a_false": 0.26},
    ]
    db = belief_degree_huang(outcomes)
    results["belief_degree_match"] = abs(db - (-0.9420)) < tolerance
    if not results["belief_degree_match"]:
        raise VerificationError(
            f"D_b = {db:.4f}, expected -0.9420 (tolerance {tolerance})"
        )

    # Check 2: Quantum probability
    q_probs = quantum_probability(conditionals)
    p_defect = q_probs["defect"]
    results["probability_match"] = abs(p_defect - 0.6926) < tolerance
    if not results["probability_match"]:
        raise VerificationError(
            f"P(Defect) = {p_defect:.4f}, expected 0.6926 (tolerance {tolerance})"
        )

    # Check 3: Amplitude vectors
    alpha, beta = _compute_amplitude_vectors(0.26, 0.13, 0.5, 0.5)
    results["amplitude_alpha"] = abs(alpha - 0.3606) < tolerance
    results["amplitude_beta"] = abs(beta - 0.2550) < tolerance

    # Check 4: Deng entropy reduces to Shannon for singletons
    sh = shannon_entropy(np.array([0.3, 0.7]))
    de = deng_entropy([(1, 0.3), (1, 0.7)])
    results["deng_shannon_equivalence"] = abs(sh - de) < 1e-10

    # Check 5: Deng > Shannon for imprecise evidence
    de_imprecise = deng_entropy([(1, 0.3), (2, 0.7)])
    results["deng_exceeds_shannon"] = de_imprecise > sh

    # Check 6: Psi profile computation
    profile = compute_psi_from_conditionals(conditionals)
    results["psi_profile_valid"] = (
        profile.psi_1 > 0 and
        profile.psi_2 > 0 and
        abs(profile.belief_degree - round(db, 6)) < 1e-10
    )

    logger.info(
        "Verification complete: %d/%d checks passed",
        sum(results.values()), len(results),
    )
    return results


# ═══════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# Aliases for code that imports from the old deng_entropy_eap module.
# ═══════════════════════════════════════════════════════════════════════

quantum_probability_qlbn = quantum_probability
compute_amplitude_vectors = _compute_amplitude_vectors
belief_distance_huang_raw = _belief_distance


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    # Data structures
    "PsiProfile",
    "PsiMethod",
    "VerificationError",
    # Ψ₁: Informational Entropy
    "deng_entropy",
    "shannon_entropy",
    # Ψ₂: Behavioral Entropy
    "belief_degree_huang",
    "quantum_probability",
    # Ψ₃: Adaptive Entropy
    "compute_psi3",
    # Ψ₄: Relational Entropy
    "compute_psi4",
    # Unified entry points
    "compute_psi_from_sequence",
    "compute_psi_from_conditionals",
    # Statistics
    "cohens_d",
    # User-friendly API
    "quick_analyze",
    "compare",
    "summary",
    "ExampleData",
    "examples",
    # Visualization (requires matplotlib)
    "plot",
    "plot_compare",
    "plot_dimensions",
    # Export / Interoperability
    "to_csv",
    "to_json",
    "to_matlab",
    # Verification
    "verify_huang_paper",
]


# ═══════════════════════════════════════════════════════════════════════
# MODULE-LEVEL ATTRIBUTE LOOKUP (helpful error for common mistakes)
# ═══════════════════════════════════════════════════════════════════════

def __getattr__(name: str):
    if name == "compute_psi4_from_conditionals":
        raise AttributeError(
            "No function 'compute_psi4_from_conditionals'. "
            "Use compute_psi4(agent_probabilities) instead. "
            "Example: compute_psi4([0.9, 0.1, 0.7])"
        )
    raise AttributeError(f"module 'insight137_eap' has no attribute '{name}'")


# ═══════════════════════════════════════════════════════════════════════
# COMMAND-LINE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print(f"Insight137 EAP v{__version__}")
    print("Entropy Attunement Protocol - Verification Suite")
    print("=" * 60)

    try:
        results = verify_huang_paper()
        print("\nHuang et al. (2019) verification:")
        for check, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check}")

        all_passed = all(results.values())
        print(f"\n{sum(results.values())}/{len(results)} checks passed")

        # Demo: Psi from sequence
        print("\n" + "-" * 60)
        print("Demo: Psi profile from behavioral sequence")
        demo_seq = [150, 200, 180, 350, 120, 400, 90, 250]
        profile = compute_psi_from_sequence(demo_seq)
        print(f"  Input: {demo_seq}")
        print(f"  Psi1 (informational): {profile.psi_1:.4f}")
        print(f"  Psi2 (behavioral):    {profile.psi_2:.4f}")
        print(f"  Psi3 (adaptive):      {profile.psi_3:.4f}")
        print(f"  Psi4 (relational):    {profile.psi_4:.4f}")
        print(f"  Belief Degree:        {profile.belief_degree:.4f}")

        # Demo: Psi4 from multi-agent decisions
        print("\n" + "-" * 60)
        print("Demo: Psi4 from 5 agents (3 bypass, 2 comply)")
        agents = [1.0, 1.0, 1.0, 0.01, 0.01]
        p4 = compute_psi4(agents)
        print(f"  Agent decisions: {agents}")
        print(f"  Psi4: {p4:.4f}")

        print("\n" + "=" * 60)
        if all_passed:
            print("ALL VERIFICATIONS PASSED")
            sys.exit(0)
        else:
            print("SOME VERIFICATIONS FAILED")
            sys.exit(1)

    except VerificationError as e:
        print(f"\nVERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        sys.exit(2)
