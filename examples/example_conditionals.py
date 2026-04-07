"""
Example 2: Quantum-corrected probabilities
============================================
Compute quantum-like probabilities for Prisoner's Dilemma
using the Huang interference method.
"""

from insight137_eap import (
    compute_psi_from_conditionals,
    quantum_probability,
)

# Prisoner's Dilemma data (Busemeyer et al. 2006)
# P(Defect | opponent defected) = 0.91
# P(Defect | opponent cooperated) = 0.84
# Observed P(Defect | unknown) = 0.66

conditionals = {
    "defect": {
        "p_given_a_true": 0.91,
        "p_given_a_false": 0.84,
    },
    "cooperate": {
        "p_given_a_true": 0.09,
        "p_given_a_false": 0.16,
    },
}

# Classical prediction (law of total probability)
p_classical = 0.5 * 0.91 + 0.5 * 0.84
print(f"Classical P(Defect):  {p_classical:.4f}")

# Quantum-corrected prediction (Huang interference)
q_probs = quantum_probability(conditionals)
print(f"Quantum P(Defect):    {q_probs['defect']:.4f}")
print(f"Observed P(Defect):   0.6600")
print(f"Quantum is closer:    {abs(q_probs['defect'] - 0.66) < abs(p_classical - 0.66)}")

# Full Psi profile
profile = compute_psi_from_conditionals(conditionals)
print(f"\nPsi1: {profile.psi_1:.4f}")
print(f"Psi2: {profile.psi_2:.4f}")
print(f"Db:   {profile.belief_degree:.4f}")
