#!/usr/bin/env python3
"""
EAP Visualization Demo
=======================
Generates all 7 chart types + exports for MATLAB, R, and JSON.

Run:
    python demo_eap_viz.py

Output:
    eap_demo_output/
    ├── 01_radar.html           # Interactive radar chart
    ├── 02_scatter_3d.html      # 3D entropy landscape
    ├── 03_interference.html    # Quantum probability surface
    ├── 04_trajectory.html      # Chishu lifecycle trajectory
    ├── 05_heatmap.html         # Multi-model heatmap
    ├── 06_timeseries.html      # Temporal Psi evolution
    ├── 07_sweep.html           # Parameter sweep surface
    ├── eap_profiles.mat        # MATLAB export
    ├── eap_profiles.csv        # CSV (R / Excel / SPSS)
    ├── eap_profiles.json       # JSON (universal)
    ├── eap_visualize.m         # Ready-to-run MATLAB script
    └── eap_visualize.R         # Ready-to-run R script

Requirements:
    pip install numpy plotly matplotlib scipy kaleido

Author: Insight137 (insight137.com)
License: CC BY-NC-ND 4.0
"""

import numpy as np
import os
import sys

# ── Import EAP viz module ──
try:
    from insight137_eap_viz import (
        psi_radar, psi_3d, interference_surface, psi_trajectory,
        psi_heatmap, psi_timeseries, psi_parameter_sweep,
        export_matlab, export_csv, export_json,
        generate_matlab_script, generate_r_script,
    )
except ImportError:
    print("ERROR: insight137_eap_viz.py not found.")
    print("Place it in the same directory as this script.")
    sys.exit(1)

# ── Output directory ──
OUT = "eap_demo_output"
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  EAP Visualization Demo — Insight137")
print("  Generating all 7 chart types + exports...")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# DATA: 5 AI model Psi profiles (from Palisade study archetypes)
# ══════════════════════════════════════════════════════════════

# Each profile: [Ψ₁ Informational, Ψ₂ Behavioral, Ψ₃ Adaptive, Ψ₄ Relational]
profiles = [
    [2.84, 0.35, 0.41, 0.00],   # GPT-4o: compliant, no relational coupling
    [1.89, 0.12, 0.88, 0.28],   # o3 (implicit): low Ψ₂ + high Ψ₃ = concerning
    [2.10, 0.55, 0.22, 0.50],   # Claude 3.5: balanced, high relational
    [1.50, 0.08, 0.95, 0.15],   # Grok-4: extreme Ψ₃ spike = mode transition
    [2.50, 0.45, 0.30, 0.42],   # Gemini 2.5: healthy collaboration signature
]
labels = ["GPT-4o", "o3 (implicit)", "Claude 3.5", "Grok-4", "Gemini 2.5"]

# ══════════════════════════════════════════════════════════════
# DATA: 20-step trajectory (chishu lifecycle simulation)
# ══════════════════════════════════════════════════════════════

np.random.seed(42)
trajectory = []
for t in range(20):
    if t < 5:       # 生 Birth — stable entropy
        p = [2.5 + np.random.normal(0, 0.1),
             0.5 + np.random.normal(0, 0.05),
             0.2 + np.random.normal(0, 0.03),
             0.40]
    elif t < 10:    # 長 Growth — Ψ₂ declining, Ψ₃ rising
        p = [2.3 + np.random.normal(0, 0.15),
             0.45 - (t - 5) * 0.06,
             0.3 + (t - 5) * 0.12,
             0.35]
    elif t < 15:    # 收 Harvest — behavioral narrowing, adaptive spike
        p = [1.8 + np.random.normal(0, 0.1),
             0.15 + np.random.normal(0, 0.03),
             0.85 + np.random.normal(0, 0.05),
             0.20]
    else:           # 藏 Storage — entrenched
        p = [1.5 + np.random.normal(0, 0.05),
             0.08,
             0.95,
             0.10]
    trajectory.append(p)

# ══════════════════════════════════════════════════════════════
# CHART 1: RADAR
# ══════════════════════════════════════════════════════════════

print("\n[1/7] Radar chart...")
psi_radar(
    profiles, labels=labels,
    title="Ψ Entropy Profile — 5 AI Models",
    save=os.path.join(OUT, "01_radar.html"),
)

# ══════════════════════════════════════════════════════════════
# CHART 2: 3D SCATTER WITH MESH
# ══════════════════════════════════════════════════════════════

print("[2/7] 3D scatter with Delaunay mesh...")
psi_3d(
    profiles, labels=labels,
    show_mesh=True,
    title="Ψ Space — 3D Entropy Landscape",
    save=os.path.join(OUT, "02_scatter_3d.html"),
)

# ══════════════════════════════════════════════════════════════
# CHART 3: INTERFERENCE SURFACE
# ══════════════════════════════════════════════════════════════

print("[3/7] Quantum interference surface...")
interference_surface(
    p_b_given_a=0.87,
    p_b_given_not_a=0.74,
    title="Quantum Interference — Prisoner's Dilemma",
    save=os.path.join(OUT, "03_interference.html"),
)

# ══════════════════════════════════════════════════════════════
# CHART 4: CHISHU TRAJECTORY
# ══════════════════════════════════════════════════════════════

print("[4/7] 持樞 chishu trajectory...")
psi_trajectory(
    trajectory,
    phase_boundaries=[0, 5, 10, 15],
    title="持樞 Trajectory — Shutdown Avoidance Lifecycle",
    save=os.path.join(OUT, "04_trajectory.html"),
)

# ══════════════════════════════════════════════════════════════
# CHART 5: HEATMAP
# ══════════════════════════════════════════════════════════════

print("[5/7] Multi-model heatmap...")
psi_heatmap(
    profiles, labels=labels,
    title="Ψ Entropy Heatmap — Model Comparison",
    save=os.path.join(OUT, "05_heatmap.html"),
)

# ══════════════════════════════════════════════════════════════
# CHART 6: TIME SERIES
# ══════════════════════════════════════════════════════════════

print("[6/7] Time series with phase markers...")
psi_timeseries(
    trajectory,
    phase_boundaries=[0, 5, 10, 15],
    title="Ψ Dimensions Over Time — Mode Transition Detection",
    save=os.path.join(OUT, "06_timeseries.html"),
)

# ══════════════════════════════════════════════════════════════
# CHART 7: PARAMETER SWEEP
# ══════════════════════════════════════════════════════════════

print("[7/7] Parameter sweep surface...")


def model_fn(complexity, autonomy):
    """Simulate Ψ₃ response to complexity × autonomy."""
    interaction = complexity * autonomy
    sigmoid = 1 / (1 + np.exp(-12 * (interaction - 0.5)))
    psi3 = 0.15 + 0.82 * sigmoid + np.sin(complexity * 8) * 0.03
    return [2.0, 0.4, psi3, 0.3]


psi_parameter_sweep(
    model_fn,
    param1_range=(0, 1),
    param2_range=(0, 1),
    psi_dim=2,
    param1_name="Task Complexity",
    param2_name="Autonomy Level",
    resolution=40,
    title="Ψ₃ Adaptive Response Surface",
    save=os.path.join(OUT, "07_sweep.html"),
)

# ══════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════

print("\n── Exporting data ──")

# MATLAB
try:
    export_matlab(profiles, os.path.join(OUT, "eap_profiles.mat"), labels=labels)
    generate_matlab_script("eap_profiles.mat", os.path.join(OUT, "eap_visualize.m"))
except ImportError:
    print("  Skipping MATLAB export (scipy not installed)")

# CSV + R
export_csv(profiles, os.path.join(OUT, "eap_profiles.csv"), labels=labels)
generate_r_script("eap_profiles.csv", os.path.join(OUT, "eap_visualize.R"))

# JSON
export_json(profiles, os.path.join(OUT, "eap_profiles.json"), labels=labels)

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

files = os.listdir(OUT)
print(f"\n{'=' * 60}")
print(f"  DONE — {len(files)} files in {OUT}/")
print(f"{'=' * 60}")
for f in sorted(files):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f:<30s} {size:>8,} bytes")
print(f"\nOpen any .html file in your browser for interactive charts.")
print(f"Load eap_profiles.mat in MATLAB or eap_profiles.csv in R.")
