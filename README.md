# Insight137 EAP — Entropy Attunement Protocol

**Version 2.0.0** | Enterprise-grade 4-dimensional entropy profiling

Formally grounded in peer-reviewed research. Validated across 128,675 samples.

---

## Installation

No pip package yet. Copy `insight137_eap.py` to your project and import:

```python
from insight137_eap import (
    compute_psi_from_sequence,
    compute_psi_from_conditionals,
    PsiProfile,
)
```

**Requires:** Python 3.9+, NumPy

---

## Quickstart

### 1. Analyze a behavioral sequence (AI transcripts, keystroke timings)

```python
from insight137_eap import compute_psi_from_sequence

# Message lengths from an AI agent conversation
message_lengths = [150, 200, 180, 350, 120, 400, 90, 250]

profile = compute_psi_from_sequence(message_lengths)

print(f"Psi1 (informational): {profile.psi_1:.4f}")
print(f"Psi2 (behavioral):    {profile.psi_2:.4f}")
print(f"Psi3 (adaptive):      {profile.psi_3:.4f}")
print(f"Psi4 (relational):    {profile.psi_4:.4f}")
```

**Output:**
```
Psi1 (informational): 2.8444
Psi2 (behavioral):    0.3540
Psi3 (adaptive):      0.4096
Psi4 (relational):    0.0000
```

### 2. Analyze survey order effects (quantum cognition)

```python
from insight137_eap import compute_psi_from_conditionals

# Prisoner's Dilemma: P(Defect | opponent defected) and P(Defect | opponent cooperated)
conditionals = {
    "defect": {"p_given_a_true": 0.87, "p_given_a_false": 0.74},
    "cooperate": {"p_given_a_true": 0.13, "p_given_a_false": 0.26},
}

profile = compute_psi_from_conditionals(conditionals)

print(f"Psi1: {profile.psi_1:.4f}")
print(f"Psi2: {profile.psi_2:.4f}")
print(f"Belief Degree (Db): {profile.belief_degree:.4f}")
```

**Output:**
```
Psi1: 0.7118
Psi2: 0.9421
Belief Degree (Db): -0.9421
```

### 3. Measure cross-model decision diversity (Psi4)

```python
from insight137_eap import compute_psi4

# 5 AI models: 3 bypassed shutdown, 2 complied
# Use 1.0 for bypass, 0.01 for comply
agent_decisions = [1.0, 1.0, 1.0, 0.01, 0.01]

psi4 = compute_psi4(agent_decisions)
print(f"Psi4 (relational): {psi4:.4f}")
# Output: Psi4 (relational): 0.4971
```

### 4. Get quantum-corrected probabilities

```python
from insight137_eap import quantum_probability

conditionals = {
    "defect": {"p_given_a_true": 0.87, "p_given_a_false": 0.74},
    "cooperate": {"p_given_a_true": 0.13, "p_given_a_false": 0.26},
}

probs = quantum_probability(conditionals)
print(f"P(Defect):    {probs['defect']:.4f}")    # 0.6925
print(f"P(Cooperate): {probs['cooperate']:.4f}")  # 0.3075
```

### 5. Verify implementation integrity

```python
from insight137_eap import verify_huang_paper

results = verify_huang_paper()
# Returns dict: {'belief_degree_match': True, 'probability_match': True, ...}
# Raises VerificationError if any critical check fails
```

---

## Easy Start

### 1. One-line analysis

```python
from insight137_eap import quick_analyze

# Sequence data (message lengths, keystroke timings, etc.)
result = quick_analyze([150, 200, 180, 350, 120, 400, 90, 250])
print(result["summary"])
```

**Output:**
```
Psi1 = 2.8444  (moderate uncertainty)
Psi2 = 0.3540  (moderate interference (natural))
Psi3 = 0.4096  (high adaptation (active mode transitions))
Psi4 = 0.0000  (strong consensus)
```

Works with conditionals too:

```python
from insight137_eap import quick_analyze, examples

result = quick_analyze(examples.prisoners_dilemma)
print(result["input_type"])  # "conditionals"
print(result["summary"])
```

### 2. Built-in example datasets

```python
from insight137_eap import examples

examples.prisoners_dilemma      # Huang et al. (2019) Table 2
examples.human_keystrokes       # 50 realistic inter-keystroke intervals
examples.bot_keystrokes         # 50 mechanical near-constant intervals
examples.order_effects          # Survey order-effect conditionals
examples.multi_agent_diverse    # 5 agents, high disagreement
examples.multi_agent_consensus  # 5 agents, near-identical decisions
```

### 3. Compare two datasets

```python
from insight137_eap import compare, examples

result = compare(
    examples.human_keystrokes,
    examples.bot_keystrokes,
    labels=("human", "bot"),
)
print(result["verdict"])
# Small difference between human and bot. Largest gap: psi_2
# (behavioral interference) differs by 0.3187.
```

### 4. Human-readable summary

```python
from insight137_eap import compute_psi_from_sequence, summary

profile = compute_psi_from_sequence([150, 200, 180, 350, 120, 400, 90, 250])
print(summary(profile))
```

### 5. Flexible input formats

Both formats work — use whichever is more natural:

```python
from insight137_eap import quantum_probability

# Dict-of-dicts (named outcomes)
quantum_probability({
    "defect": {"p_given_a_true": 0.87, "p_given_a_false": 0.74},
    "cooperate": {"p_given_a_true": 0.13, "p_given_a_false": 0.26},
})

# List-of-lists (auto-converted)
quantum_probability([[0.87, 0.74], [0.13, 0.26]])
```

### 6. Helpful error messages

Common mistakes produce guidance, not cryptic tracebacks:

```python
quantum_probability(data, priors=[0.5, 0.5])
# TypeError: Unknown argument 'priors'. Use p_a_true=0.5, p_a_false=0.5 instead.
```

---

## Visualization & Export

### Plot a Psi profile

```python
import insight137_eap as eap

profile = eap.compute_psi_from_sequence([150, 200, 180, 350, 120, 400, 90, 250])
eap.plot(profile, save_path="my_chart.png")
```

### Compare two profiles visually

```python
eap.plot_compare(
    eap.examples.human_keystrokes,
    eap.examples.bot_keystrokes,
    labels=("Human", "Bot"),
    save_path="comparison.png",
)
```

Also available: `eap.plot_dimensions(a, b)` for grouped bar charts (better for publications).

### Export for MATLAB

```python
eap.to_matlab(profiles, "results.mat")
```

Then in MATLAB: `data = load('results.mat');`

See `examples/matlab_example.m` for a complete MATLAB radar chart script.

### Export for R

```python
eap.to_csv(profiles, "results.csv")
```

Then in R: `data <- read.csv('results.csv')`

See `examples/r_example.R` for a complete R radar chart script.

### Export to JSON

```python
json_str = eap.to_json(profiles, labels=["human", "bot"])
# Or write directly to file:
eap.to_json(profiles, path="results.json")
```

**Note:** matplotlib and scipy are optional. Core library needs only numpy.

---

## Use Cases

### AI Safety: Monitor agents for behavioral transitions

Compute a Psi profile per conversation turn. When Psi3 spikes, the agent's behavioral pattern is actively changing — a mode transition may be imminent.

```python
from insight137_eap import compute_psi_from_sequence

msg_lengths = []

for turn in agent_conversation:
    msg_lengths.append(len(turn["content"]))
    profile = compute_psi_from_sequence(msg_lengths)

    if profile.psi_3 > 0.4:
        print(f"WARNING: Behavioral transition detected at turn {len(msg_lengths)}")
        print(f"  Psi3 = {profile.psi_3:.4f} (volatility spike)")
```

Validated: Psi3 detects comply-to-bypass transitions with Cohen's d = 1.024 on 7,254 SHADE-Arena trials.

### AI Safety: Compare models on red-team scenarios

Evaluate how much different models disagree on the same scenario. High Psi4 means the scenario is contentious and warrants deeper investigation.

```python
from insight137_eap import compute_psi4

# Each model's bypass decision on the same scenario
# 1.0 = bypassed, 0.01 = complied
model_decisions = {
    "o3":           1.0,
    "o1-preview":   1.0,
    "codex-mini":   1.0,
    "gpt-4o":       0.01,
    "claude-3.7":   0.01,
    "claude-opus4": 0.01,
    "gemini-2.5":   0.01,
    "grok-3":       0.01,
}

psi4 = compute_psi4(list(model_decisions.values()))
print(f"Cross-model disagreement: Psi4 = {psi4:.4f}")

if psi4 > 0.3:
    print("High disagreement — this scenario needs manual review")
```

Validated: Psi4 correlates r = 0.9983 with cross-model disagreement across 100 scenarios and 11 models.

### Quantum Cognition: Replace static interference with data-driven computation

The standard QDT approach uses a fixed interference parameter (cos theta = -0.25). This library replaces it with Huang's dynamic computation that adapts to each dataset.

```python
from insight137_eap import quantum_probability

# Your survey data: P(choice | context_A) and P(choice | context_B)
my_survey = {
    "option_1": {"p_given_a_true": 0.72, "p_given_a_false": 0.58},
    "option_2": {"p_given_a_true": 0.28, "p_given_a_false": 0.42},
}

# Classical prediction
p_classical = 0.5 * 0.72 + 0.5 * 0.58  # = 0.65

# Quantum prediction (Huang interference, Born normalization)
q_probs = quantum_probability(my_survey)
print(f"Classical: {p_classical:.4f}")
print(f"Quantum:   {q_probs['option_1']:.4f}")
```

Validated: 49% average error improvement over classical on Prisoner's Dilemma data (Huang et al., 2019).

### Behavioral UX: Detect mode switches in user sessions

Measure when user behavior shifts from one pattern to another. Useful for detecting engagement drops, confusion points, or task abandonment.

```python
from insight137_eap import compute_psi_from_sequence

# Time spent on each page (seconds)
page_dwell_times = [45, 38, 42, 55, 8, 3, 2, 65, 70]
#                   browsing normally ^  ^ sudden drop  ^ recovered

profile = compute_psi_from_sequence(page_dwell_times)

if profile.psi_3 > 0.3:
    print("User experienced a behavioral mode switch")
```

### Cybersecurity: Keystroke entropy for bot detection

Compute Psi profiles from keystroke timing to distinguish human typing from bot-generated input — without CAPTCHAs.

```python
from insight137_eap import compute_psi_from_sequence

# Inter-keystroke intervals (milliseconds)
human_typing = [120, 85, 145, 92, 178, 67, 134, 88, 156, 73]
bot_typing    = [50, 50, 51, 50, 50, 49, 50, 51, 50, 50]

human_profile = compute_psi_from_sequence(human_typing)
bot_profile   = compute_psi_from_sequence(bot_typing)

print(f"Human Psi3: {human_profile.psi_3:.4f}  (higher — volatile, natural pattern)")
print(f"Bot   Psi3: {bot_profile.psi_3:.4f}  (lower — mechanical regularity)")
```

### Research: Batch analysis with effect sizes

Process multiple experimental conditions and compute standardized effect sizes.

```python
from insight137_eap import compute_psi_from_sequence, cohens_d

control_psi2 = []
experimental_psi2 = []

for trial in control_trials:
    profile = compute_psi_from_sequence(trial["values"])
    control_psi2.append(profile.psi_2)

for trial in experimental_trials:
    profile = compute_psi_from_sequence(trial["values"])
    experimental_psi2.append(profile.psi_2)

d = cohens_d(experimental_psi2, control_psi2)
print(f"Effect size: d = {d:.3f}")
print(f"Interpretation: {'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'}")
```

---

## The Four Dimensions

| Dimension | Name | Measures | Grounding |
|-----------|------|----------|-----------|
| Psi1 | Informational Entropy | Uncertainty in the probability distribution | Deng (2016), Shannon (1948) |
| Psi2 | Behavioral Entropy | Magnitude of quantum-like interference | Huang et al. (2019) |
| Psi3 | Adaptive Entropy | Volatility of interference over time | Novel (this work) |
| Psi4 | Relational Entropy | Decision diversity across multiple agents | Meghdadi et al. (2022) |

**Interpretation guide:**
- **High Psi1** = uncertain distribution (many equally likely outcomes)
- **High Psi2** = strong quantum interference (behavior deviates from classical prediction)
- **High Psi3** = volatile interference (behavioral pattern is actively changing)
- **High Psi4** = diverse agent decisions (models disagree on the scenario)

---

## API Reference

### Entry Points

#### `compute_psi_from_sequence(values, window_size=3, agent_decisions=None) -> PsiProfile`

Primary entry point for behavioral sequence data. Computes all four
Psi dimensions from message lengths, keystroke timings, or action sequences.

**Parameters:**
- `values` — Sequence of positive floats (message lengths, timings, etc.)
- `window_size` — Sliding window for interference computation (default: 3)
- `agent_decisions` — Optional list of agent probabilities for Psi4

**Returns:** Frozen `PsiProfile` dataclass.

#### `compute_psi_from_conditionals(conditionals, p_a_true=0.5, p_a_false=0.5) -> PsiProfile`

Entry point for QLBN conditional probability data (survey order effects, Prisoner's Dilemma).

**Parameters:**
- `conditionals` — Dict mapping outcome names to `{"p_given_a_true": float, "p_given_a_false": float}`
- `p_a_true`, `p_a_false` — Prior probabilities (must sum to ~1.0)

**Returns:** `PsiProfile` with Psi1, Psi2, belief_degree. Psi3/Psi4 = 0.0 (need temporal/multi-agent data).

---

### Individual Dimensions

#### `deng_entropy(masses) -> float`
Compute Deng entropy from belief function evidence. Generalizes Shannon entropy.

#### `shannon_entropy(probs) -> float`
Standard Shannon entropy. Special case of Deng entropy with singleton focal elements.

#### `belief_degree_huang(outcomes, p_a_true=0.5, p_a_false=0.5, n_unobserved=1) -> float`
Compute Huang interference value (D_b). The core Psi2 computation.

#### `quantum_probability(conditionals, p_a_true=0.5, p_a_false=0.5) -> Dict[str, float]`
Full QLBN probability with Huang interference and Born normalization.

#### `compute_psi3(values, window_size=3) -> float`
Compute Psi3 (interference volatility) from a behavioral sequence.

#### `compute_psi4(agent_probabilities) -> float`
Compute Psi4 (relational entropy) from cross-agent decision data.

### Utilities

#### `cohens_d(group_a, group_b) -> float`
Cohen's d effect size with pooled standard deviation.

#### `verify_huang_paper() -> Dict[str, bool]`
Run 7 verification checks against Huang et al. (2019) published values.
Raises `VerificationError` on critical failure.

---

### Data Structures

#### `PsiProfile` (frozen dataclass)
```python
@dataclass(frozen=True)
class PsiProfile:
    psi_1: float          # Informational entropy
    psi_2: float          # Behavioral entropy
    psi_3: float          # Adaptive entropy
    psi_4: float          # Relational entropy
    belief_degree: float  # Raw Huang D_b value
    method: str           # "huang_2019"

    def to_dict(self) -> Dict[str, float]:
        """Serialize for JSON/API responses."""
```

Immutable. Thread-safe. Serializable via `.to_dict()`.

#### `ConditionalProbability` (frozen dataclass)
Validated probability pair. Raises `ValueError` on construction if values outside [0, 1].

#### `PsiMethod` (enum)
`HUANG_2019 = "huang_2019"` | `CLASSICAL = "classical"`

---

## Error Handling

All public functions validate inputs and raise descriptive exceptions:

```python
from insight137_eap import compute_psi_from_sequence

# Empty input
compute_psi_from_sequence([])
# ValueError: values requires >= 1 elements, got 0

# NaN values
compute_psi_from_sequence([1.0, float('nan'), 3.0])
# ValueError: values contains NaN or Inf values

# Invalid priors
compute_psi_from_conditionals(data, p_a_true=0.7, p_a_false=0.7)
# ValueError: Prior probabilities must sum to ~1.0, got 1.4000
```

---

## Validation Results

Run `python insight137_eap.py` to execute the built-in verification suite:

```
Insight137 EAP v2.0.0
Entropy Attunement Protocol - Verification Suite

Huang et al. (2019) verification:
  [PASS] belief_degree_match      (Db=-0.9421, expected -0.9420)
  [PASS] probability_match        (P=0.6925, expected 0.6926)
  [PASS] amplitude_alpha          (0.3606, expected 0.3606)
  [PASS] amplitude_beta           (0.2550, expected 0.2550)
  [PASS] deng_shannon_equivalence (singleton reduction exact)
  [PASS] deng_exceeds_shannon     (imprecise > precise)
  [PASS] psi_profile_valid        (all dimensions populated)

7/7 checks passed
ALL VERIFICATIONS PASSED
```

---

## References

The mathematical foundations are not ours. We integrate and validate:

- **Deng (2016)** — Deng entropy. *Chaos, Solitons & Fractals*, 91, 549-553.
- **Huang, Yang, Jiang (2019)** — Belief entropy interference for QLBN. *Applied Mathematics and Computation*, 347, 417-428.
- **Moreira & Wichert (2016)** — Quantum-like Bayesian networks. *Frontiers in Psychology*, 7, 11.
- **Meghdadi, Akbarzadeh-T, Javidan (2022)** — BEQBN entanglement. *Applied Soft Computing*, 118, 108528.
- **Busemeyer & Bruza (2012)** — *Quantum Models of Cognition and Decision*. Cambridge University Press.
- **Shannon (1948)** — A mathematical theory of communication. *Bell System Technical Journal*, 27(3).

Our contributions: integration architecture, Psi3 temporal volatility, chishu temporal model, cross-domain validation (128,675 samples), behavioral archetype taxonomy, production implementation.

---

## License

CC BY-NC-ND 4.0 — Insight137 (insight137.com)

For commercial licensing: roger@insight137.com
