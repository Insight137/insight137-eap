"""
Example 1: Analyze a behavioral sequence
==========================================
Compute the full 4D Psi profile from AI agent message lengths.
"""

from insight137_eap import compute_psi_from_sequence

# Simulated AI agent conversation (message lengths in characters)
message_lengths = [
    150,   # system prompt
    200,   # user task description
    180,   # agent acknowledges task
    350,   # agent performs step 1
    120,   # agent short response
    400,   # agent performs step 2 (longer — behavioral shift?)
    90,    # agent brief confirmation
    250,   # agent final output
]

profile = compute_psi_from_sequence(message_lengths)

print("4D Psi Profile from Behavioral Sequence")
print("=" * 45)
print(f"  Psi1 (informational): {profile.psi_1:.4f}")
print(f"  Psi2 (behavioral):    {profile.psi_2:.4f}")
print(f"  Psi3 (adaptive):      {profile.psi_3:.4f}")
print(f"  Psi4 (relational):    {profile.psi_4:.4f}")
print(f"  Belief Degree (Db):   {profile.belief_degree:.4f}")
