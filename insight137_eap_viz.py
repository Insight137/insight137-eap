"""
Insight137 EAP Visualization Module
====================================
Version: 2.0.0
License: CC BY-NC-ND 4.0
DOI: 10.17605/OSF.IO/H96QD

Sophisticated, cross-platform visualization for Entropy Attunement Protocol
(EAP) Psi profiles. Works with Plotly (interactive), Matplotlib (publication),
and exports to MATLAB (.mat), R (.csv), JSON, and standard image formats.

Requirements
------------
Core:     numpy (already required by insight137_eap)
Interactive: plotly >= 5.0
Static:      matplotlib >= 3.5
MATLAB:      scipy (for .mat export)

Install all optional deps:
    pip install plotly matplotlib scipy kaleido

Quick Start
-----------
    from insight137_eap import compute_psi_from_sequence
    from insight137_eap_viz import psi_radar, psi_3d, psi_trajectory

    profile = compute_psi_from_sequence([150, 200, 180, 350, 120, 400])

    psi_radar(profile)                          # interactive radar
    psi_3d([profile])                           # 3D scatter
    psi_trajectory(sequence_of_profiles)        # animated trajectory

    # Export for MATLAB / R / any tool
    from insight137_eap_viz import export_matlab, export_csv, export_json
    export_matlab(profile, "my_profile.mat")
    export_csv([profile], "profiles.csv")

Cross-Platform
--------------
- Windows / macOS / Linux: all visualizations use standard libraries
- MATLAB: export_matlab() writes .mat files; also provides MATLAB script generator
- R: export_csv() with headers; export_json() for jsonlite::fromJSON()
- Jupyter: Plotly renders inline automatically
- CLI: save to HTML/PNG/SVG/PDF without GUI display

Author: Roger Yau (Jus) — Insight137
ORCID: 0009-0009-0729-6274
"""

from __future__ import annotations

import json
import csv
import os
import warnings
from dataclasses import dataclass, asdict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS & THEMING
# ─────────────────────────────────────────────────────────────────────

# Psi dimension metadata
PSI_LABELS = ["Ψ₁ Informational", "Ψ₂ Behavioral", "Ψ₃ Adaptive", "Ψ₄ Relational"]
PSI_SHORT  = ["Ψ₁", "Ψ₂", "Ψ₃", "Ψ₄"]
PSI_KEYS   = ["psi_1", "psi_2", "psi_3", "psi_4"]

# Chishu (持樞) phase metadata
CHISHU_PHASES = [
    {"key": "sheng", "char": "生", "name": "Birth",   "en": "Emergence"},
    {"key": "zhang", "char": "長", "name": "Growth",   "en": "Expansion"},
    {"key": "shou",  "char": "收", "name": "Harvest",  "en": "Consolidation"},
    {"key": "cang",  "char": "藏", "name": "Storage",  "en": "Entrenchment"},
]

# ── Color palettes ──
# Named palettes users can select by string
_PALETTES = {
    "insight137": {
        "psi_colors": ["#4ecdc4", "#5b8dee", "#c57bdb", "#e85d75"],
        "bg":         "#0d0d1a",
        "surface":    "#14142b",
        "grid":       "#1e1e3a",
        "text":       "#e0e0f0",
        "dim":        "#6b6b8d",
        "accent":     "#4ecdc4",
        "chishu":     ["#4ecdc4", "#5b8dee", "#f5a623", "#e85d75"],
    },
    "publication": {
        "psi_colors": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"],
        "bg":         "#ffffff",
        "surface":    "#f8f8f8",
        "grid":       "#e0e0e0",
        "text":       "#1a1a1a",
        "dim":        "#666666",
        "accent":     "#1b9e77",
        "chishu":     ["#1b9e77", "#d95f02", "#e6ab02", "#e7298a"],
    },
    "matlab": {
        "psi_colors": ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"],
        "bg":         "#ffffff",
        "surface":    "#f5f5f5",
        "grid":       "#d0d0d0",
        "text":       "#000000",
        "dim":        "#555555",
        "accent":     "#0072BD",
        "chishu":     ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"],
    },
}

DEFAULT_PALETTE = "insight137"


def get_palette(name: str = DEFAULT_PALETTE) -> dict:
    """Return a color palette by name. Options: 'insight137', 'publication', 'matlab'."""
    if name not in _PALETTES:
        raise ValueError(
            f"Unknown palette '{name}'. Choose from: {list(_PALETTES.keys())}"
        )
    return _PALETTES[name]


# ─────────────────────────────────────────────────────────────────────
# PROFILE HELPERS
# ─────────────────────────────────────────────────────────────────────

def _extract_psi(profile) -> np.ndarray:
    """Extract [Ψ₁, Ψ₂, Ψ₃, Ψ₄] from a PsiProfile, dict, list, or ndarray."""
    if hasattr(profile, "psi_1"):
        return np.array([profile.psi_1, profile.psi_2, profile.psi_3, profile.psi_4])
    if isinstance(profile, dict):
        return np.array([profile[k] for k in PSI_KEYS])
    if isinstance(profile, (list, tuple, np.ndarray)):
        arr = np.asarray(profile, dtype=float)
        if arr.shape == (4,):
            return arr
    raise TypeError(
        f"Cannot extract Psi values from {type(profile).__name__}. "
        "Pass a PsiProfile, dict with psi_1..psi_4, or length-4 array."
    )


def _profile_to_dict(profile, label: Optional[str] = None) -> dict:
    """Convert any profile representation to a standard dict."""
    psi = _extract_psi(profile)
    d = {PSI_KEYS[i]: float(psi[i]) for i in range(4)}
    if label:
        d["label"] = label
    elif hasattr(profile, "label"):
        d["label"] = profile.label
    # Carry extra fields from PsiProfile dataclass
    if hasattr(profile, "__dataclass_fields__"):
        for key in profile.__dataclass_fields__:
            if key not in d:
                val = getattr(profile, key)
                if val is not None:
                    d[key] = val
    return d


def _ensure_plotly():
    """Import plotly or raise a helpful error."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        return go, pio
    except ImportError:
        raise ImportError(
            "Plotly is required for interactive visualizations.\n"
            "Install with: pip install plotly kaleido"
        )


def _ensure_matplotlib():
    """Import matplotlib or raise a helpful error."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for static/publication figures.\n"
            "Install with: pip install matplotlib"
        )


# ─────────────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────────────

def _save_or_show(fig, save_path: Optional[str], engine: str = "plotly"):
    """Save figure to file or show interactively.

    Parameters
    ----------
    fig : plotly Figure or matplotlib Figure
    save_path : str or None
        If None, show interactively.
        Supported extensions: .html, .png, .svg, .pdf, .jpg, .jpeg, .webp
    engine : str
        'plotly' or 'matplotlib'
    """
    if save_path is None:
        if engine == "plotly":
            fig.show()
        else:
            import matplotlib.pyplot as plt
            plt.show()
        return

    save_path = str(save_path)
    ext = os.path.splitext(save_path)[1].lower()

    if engine == "plotly":
        if ext == ".html":
            fig.write_html(save_path, include_plotlyjs="cdn")
        elif ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".webp"):
            fig.write_image(save_path, scale=3)
        else:
            raise ValueError(f"Unsupported format '{ext}' for Plotly. Use: .html .png .svg .pdf .jpg")
        print(f"Saved: {save_path}")
    else:
        import matplotlib.pyplot as plt
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════
# 1. RADAR / SPIDER CHART
# ═════════════════════════════════════════════════════════════════════

def psi_radar(
    profiles: Union[Any, List[Any]],
    labels: Optional[List[str]] = None,
    title: str = "Ψ Entropy Profile",
    palette: str = DEFAULT_PALETTE,
    fill: bool = True,
    save: Optional[str] = None,
    engine: str = "plotly",
) -> Any:
    """Radar chart of one or more Psi profiles.

    Parameters
    ----------
    profiles : PsiProfile, dict, list, or list of these
        One profile or a list of profiles to overlay.
    labels : list of str, optional
        Legend labels for each profile.
    title : str
        Chart title.
    palette : str
        Color palette name: 'insight137', 'publication', 'matlab'.
    fill : bool
        Whether to fill the radar polygons.
    save : str, optional
        File path to save (.html, .png, .svg, .pdf). None = show interactive.
    engine : str
        'plotly' (interactive, default) or 'matplotlib' (static/publication).

    Returns
    -------
    Figure object (plotly or matplotlib)

    Examples
    --------
    >>> from insight137_eap import compute_psi_from_sequence
    >>> p = compute_psi_from_sequence([150, 200, 180, 350])
    >>> psi_radar(p)                              # single profile
    >>> psi_radar([p1, p2], labels=["Agent A", "Agent B"])  # compare
    >>> psi_radar(p, save="radar.png", engine="matplotlib") # publication
    """
    # Normalize input to list
    if not isinstance(profiles, list) or (
        isinstance(profiles, list) and len(profiles) == 4 and isinstance(profiles[0], (int, float))
    ):
        profiles = [profiles]

    pal = get_palette(palette)
    psi_data = [_extract_psi(p) for p in profiles]

    if labels is None:
        labels = [f"Profile {i+1}" for i in range(len(profiles))]

    categories = PSI_LABELS + [PSI_LABELS[0]]  # close the polygon

    if engine == "plotly":
        go, pio = _ensure_plotly()

        fig = go.Figure()
        for i, (psi, label) in enumerate(zip(psi_data, labels)):
            color = pal["psi_colors"][i % len(pal["psi_colors"])]
            values = list(psi) + [psi[0]]  # close polygon
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                name=label,
                line=dict(color=color, width=2.5),
                fill="toself" if fill else "none",
                fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba")
                    if fill and color.startswith("rgb") else None,
                opacity=0.85 if fill else 1.0,
                hovertemplate="%{theta}: %{r:.4f}<extra>" + label + "</extra>",
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=pal["text"])),
            polar=dict(
                bgcolor=pal["surface"],
                radialaxis=dict(
                    visible=True,
                    gridcolor=pal["grid"],
                    linecolor=pal["grid"],
                    tickfont=dict(color=pal["dim"], size=10),
                ),
                angularaxis=dict(
                    gridcolor=pal["grid"],
                    linecolor=pal["grid"],
                    tickfont=dict(color=pal["text"], size=12),
                ),
            ),
            paper_bgcolor=pal["bg"],
            font=dict(color=pal["text"]),
            showlegend=len(profiles) > 1,
            legend=dict(
                bgcolor=pal["surface"],
                bordercolor=pal["grid"],
                borderwidth=1,
                font=dict(color=pal["text"]),
            ),
            margin=dict(t=80, b=40, l=80, r=80),
        )
        _save_or_show(fig, save, engine="plotly")
        return fig

    elif engine == "matplotlib":
        matplotlib, plt = _ensure_matplotlib()

        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.set_facecolor(pal["bg"])
        ax.set_facecolor(pal["surface"])

        for i, (psi, label) in enumerate(zip(psi_data, labels)):
            color = pal["psi_colors"][i % len(pal["psi_colors"])]
            values = list(psi) + [psi[0]]
            ax.plot(angles, values, "o-", color=color, linewidth=2, label=label, markersize=6)
            if fill:
                ax.fill(angles, values, color=color, alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(PSI_LABELS, fontsize=11, color=pal["text"])
        ax.tick_params(axis="y", colors=pal["dim"])
        ax.spines["polar"].set_color(pal["grid"])
        ax.grid(color=pal["grid"], linewidth=0.5)
        ax.set_title(title, fontsize=16, color=pal["text"], pad=20)

        if len(profiles) > 1:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                      facecolor=pal["surface"], edgecolor=pal["grid"],
                      labelcolor=pal["text"])

        _save_or_show(fig, save, engine="matplotlib")
        return fig

    else:
        raise ValueError(f"engine must be 'plotly' or 'matplotlib', got '{engine}'")


# ═════════════════════════════════════════════════════════════════════
# 2. 3D SCATTER / POINT CLOUD
# ═════════════════════════════════════════════════════════════════════

def psi_3d(
    profiles: List[Any],
    labels: Optional[List[str]] = None,
    axes: Tuple[int, int, int] = (0, 1, 2),
    color_dim: int = 3,
    title: str = "Ψ Space — 3D Entropy Landscape",
    palette: str = DEFAULT_PALETTE,
    marker_size: float = 8.0,
    show_mesh: bool = False,
    save: Optional[str] = None,
    engine: str = "plotly",
) -> Any:
    """3D scatter plot of Psi profiles in entropy space.

    Three Ψ dimensions map to spatial axes; the fourth maps to color.

    Parameters
    ----------
    profiles : list of PsiProfile / dict / array
        Collection of profiles to plot.
    labels : list of str, optional
        Point labels (hover text for plotly, annotations for matplotlib).
    axes : tuple of 3 ints
        Which Ψ dimensions for X, Y, Z (0-indexed). Default: (0, 1, 2).
    color_dim : int
        Which Ψ dimension for color encoding. Default: 3 (Ψ₄).
    title : str
        Chart title.
    palette : str
        Color palette name.
    marker_size : float
        Marker size.
    show_mesh : bool
        If True, render a Delaunay triangulation mesh connecting the points.
    save : str, optional
        File path to save. None = show interactive.
    engine : str
        'plotly' or 'matplotlib'.

    Returns
    -------
    Figure object

    Examples
    --------
    >>> profiles = [compute_psi_from_sequence(data) for data in dataset]
    >>> psi_3d(profiles)  # Ψ₁/Ψ₂/Ψ₃ axes, Ψ₄ color
    >>> psi_3d(profiles, axes=(1,2,3), color_dim=0)  # custom mapping
    >>> psi_3d(profiles, show_mesh=True, save="landscape.html")
    """
    pal = get_palette(palette)
    psi_matrix = np.array([_extract_psi(p) for p in profiles])

    x = psi_matrix[:, axes[0]]
    y = psi_matrix[:, axes[1]]
    z = psi_matrix[:, axes[2]]
    c = psi_matrix[:, color_dim]

    x_label = PSI_SHORT[axes[0]]
    y_label = PSI_SHORT[axes[1]]
    z_label = PSI_SHORT[axes[2]]
    c_label = PSI_SHORT[color_dim]

    if labels is None:
        labels = [f"P{i+1}" for i in range(len(profiles))]

    if engine == "plotly":
        go, pio = _ensure_plotly()

        traces = []

        # Main scatter
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text" if len(profiles) <= 30 else "markers",
            marker=dict(
                size=marker_size,
                color=c,
                colorscale=[
                    [0.0, pal["psi_colors"][0]],
                    [0.33, pal["psi_colors"][1]],
                    [0.66, pal["psi_colors"][2]],
                    [1.0, pal["psi_colors"][3]],
                ],
                colorbar=dict(
                    title=dict(text=c_label, font=dict(color=pal["text"])),
                    tickfont=dict(color=pal["dim"]),
                    bgcolor=pal["surface"],
                    bordercolor=pal["grid"],
                ),
                opacity=0.9,
                line=dict(width=0.5, color=pal["grid"]),
            ),
            text=labels if len(profiles) <= 30 else None,
            textposition="top center",
            textfont=dict(color=pal["text"], size=9),
            hovertemplate=(
                f"<b>%{{text}}</b><br>"
                f"{x_label}: %{{x:.4f}}<br>"
                f"{y_label}: %{{y:.4f}}<br>"
                f"{z_label}: %{{z:.4f}}<br>"
                f"{c_label}: %{{marker.color:.4f}}"
                "<extra></extra>"
            ),
        ))

        # Optional mesh surface
        if show_mesh and len(profiles) >= 4:
            try:
                from scipy.spatial import Delaunay
                tri = Delaunay(np.column_stack([x, y, z]))
                i_idx, j_idx, k_idx = [], [], []
                for simplex in tri.simplices:
                    i_idx.append(simplex[0])
                    j_idx.append(simplex[1])
                    k_idx.append(simplex[2])
                    i_idx.append(simplex[0])
                    j_idx.append(simplex[1])
                    k_idx.append(simplex[3])
                    i_idx.append(simplex[0])
                    j_idx.append(simplex[2])
                    k_idx.append(simplex[3])
                    i_idx.append(simplex[1])
                    j_idx.append(simplex[2])
                    k_idx.append(simplex[3])

                traces.append(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i_idx, j=j_idx, k=k_idx,
                    intensity=c,
                    colorscale=[
                        [0.0, pal["psi_colors"][0]],
                        [1.0, pal["psi_colors"][3]],
                    ],
                    opacity=0.15,
                    showscale=False,
                    hoverinfo="skip",
                ))
            except ImportError:
                warnings.warn("scipy required for mesh. Install: pip install scipy")

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=pal["text"])),
            scene=dict(
                xaxis=dict(title=x_label, gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                yaxis=dict(title=y_label, gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                zaxis=dict(title=z_label, gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                bgcolor=pal["bg"],
            ),
            paper_bgcolor=pal["bg"],
            font=dict(color=pal["text"]),
            margin=dict(t=80, b=20, l=20, r=20),
        )
        _save_or_show(fig, save, engine="plotly")
        return fig

    elif engine == "matplotlib":
        matplotlib, plt = _ensure_matplotlib()

        fig = plt.figure(figsize=(10, 8))
        fig.set_facecolor(pal["bg"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(pal["surface"])

        sc = ax.scatter(x, y, z, c=c, s=marker_size * 15,
                        cmap="viridis", edgecolors=pal["grid"],
                        linewidth=0.3, alpha=0.85)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(c_label, color=pal["text"])
        cbar.ax.yaxis.set_tick_params(color=pal["dim"])
        for label in cbar.ax.get_yticklabels():
            label.set_color(pal["dim"])

        ax.set_xlabel(x_label, color=pal["text"], fontsize=12)
        ax.set_ylabel(y_label, color=pal["text"], fontsize=12)
        ax.set_zlabel(z_label, color=pal["text"], fontsize=12)
        ax.set_title(title, color=pal["text"], fontsize=14, pad=20)
        ax.tick_params(colors=pal["dim"])

        _save_or_show(fig, save, engine="matplotlib")
        return fig

    else:
        raise ValueError(f"engine must be 'plotly' or 'matplotlib', got '{engine}'")


# ═════════════════════════════════════════════════════════════════════
# 3. INTERFERENCE SURFACE MESH
# ═════════════════════════════════════════════════════════════════════

def interference_surface(
    p_b_given_a: float,
    p_b_given_not_a: float,
    p_a_range: Tuple[float, float] = (0.01, 0.99),
    theta_range: Tuple[float, float] = (-np.pi, np.pi),
    resolution: int = 80,
    title: str = "Quantum Interference Surface",
    palette: str = DEFAULT_PALETTE,
    save: Optional[str] = None,
    engine: str = "plotly",
) -> Any:
    """3D surface mesh showing quantum probability as a function of P(A) and
    the interference phase angle θ.

    Renders the quantum total probability:
        P_q(B) = P(A)·P(B|A) + P(¬A)·P(B|¬A) + 2·√(P(A)·P(B|A)·P(¬A)·P(B|¬A))·cos(θ)

    The classical prediction is the slice at θ = π/2 (cos(θ)=0).
    Constructive interference appears at θ = 0; destructive at θ = π.

    Parameters
    ----------
    p_b_given_a : float
        P(B | A) — conditional probability.
    p_b_given_not_a : float
        P(B | ¬A) — conditional probability.
    p_a_range : tuple of float
        Range of P(A) for the x-axis.
    theta_range : tuple of float
        Range of θ for the y-axis (radians).
    resolution : int
        Grid resolution per axis (default 80 = 6,400 surface points).
    title : str
        Chart title.
    palette : str
        Color palette name.
    save : str, optional
        File path to save. None = show interactive.
    engine : str
        'plotly' or 'matplotlib'.

    Returns
    -------
    Figure object

    Examples
    --------
    >>> # Prisoner's Dilemma interference
    >>> interference_surface(0.87, 0.74)
    >>> # With publication styling
    >>> interference_surface(0.87, 0.74, palette="publication", save="fig2.pdf")
    """
    pal = get_palette(palette)

    pa = np.linspace(p_a_range[0], p_a_range[1], resolution)
    theta = np.linspace(theta_range[0], theta_range[1], resolution)
    PA, THETA = np.meshgrid(pa, theta)

    # Quantum total probability (Moreira & Wichert 2016)
    classical = PA * p_b_given_a + (1 - PA) * p_b_given_not_a
    interference = 2 * np.sqrt(PA * p_b_given_a * (1 - PA) * p_b_given_not_a) * np.cos(THETA)
    Z = np.clip(classical + interference, 0, 1)

    if engine == "plotly":
        go, pio = _ensure_plotly()

        fig = go.Figure()

        # Quantum surface
        fig.add_trace(go.Surface(
            x=PA, y=THETA, z=Z,
            colorscale=[
                [0.0, pal["psi_colors"][0]],
                [0.5, pal["psi_colors"][1]],
                [1.0, pal["psi_colors"][3]],
            ],
            opacity=0.85,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor=pal["accent"],
                       project_z=True),
            ),
            hovertemplate=(
                "P(A): %{x:.3f}<br>"
                "θ: %{y:.3f} rad<br>"
                "P_q(B): %{z:.4f}<extra></extra>"
            ),
            name="Quantum P(B)",
        ))

        # Classical plane (θ = π/2 slice extended)
        classical_flat = PA[0] * p_b_given_a + (1 - PA[0]) * p_b_given_not_a
        classical_surface = np.tile(classical_flat, (resolution, 1))
        fig.add_trace(go.Surface(
            x=PA, y=THETA, z=classical_surface,
            colorscale=[[0, pal["dim"]], [1, pal["dim"]]],
            opacity=0.3,
            showscale=False,
            hovertemplate="Classical P(B): %{z:.4f}<extra></extra>",
            name="Classical P(B)",
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=pal["text"])),
            scene=dict(
                xaxis=dict(title="P(A)", gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                yaxis=dict(title="θ (radians)", gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                zaxis=dict(title="P_q(B)", gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"],
                           range=[0, 1]),
                bgcolor=pal["bg"],
                camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            ),
            paper_bgcolor=pal["bg"],
            font=dict(color=pal["text"]),
            margin=dict(t=80, b=20, l=20, r=20),
        )
        _save_or_show(fig, save, engine="plotly")
        return fig

    elif engine == "matplotlib":
        matplotlib, plt = _ensure_matplotlib()

        fig = plt.figure(figsize=(12, 8))
        fig.set_facecolor(pal["bg"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(pal["surface"])

        ax.plot_surface(PA, THETA, Z, cmap="cool", alpha=0.8,
                        edgecolor=pal["grid"], linewidth=0.1)

        # Classical plane
        classical_flat = PA * p_b_given_a + (1 - PA) * p_b_given_not_a
        ax.plot_surface(PA, THETA, classical_flat, color=pal["dim"],
                        alpha=0.2, linewidth=0)

        ax.set_xlabel("P(A)", color=pal["text"], fontsize=12)
        ax.set_ylabel("θ (radians)", color=pal["text"], fontsize=12)
        ax.set_zlabel("P_q(B)", color=pal["text"], fontsize=12)
        ax.set_title(title, color=pal["text"], fontsize=14, pad=20)
        ax.set_zlim(0, 1)
        ax.tick_params(colors=pal["dim"])

        _save_or_show(fig, save, engine="matplotlib")
        return fig

    else:
        raise ValueError(f"engine must be 'plotly' or 'matplotlib', got '{engine}'")


# ═════════════════════════════════════════════════════════════════════
# 4. CHISHU (持樞) TRAJECTORY
# ═════════════════════════════════════════════════════════════════════

def psi_trajectory(
    profiles: List[Any],
    time_labels: Optional[List[str]] = None,
    axes: Tuple[int, int, int] = (0, 1, 2),
    color_dim: int = 3,
    phase_boundaries: Optional[List[int]] = None,
    title: str = "持樞 Trajectory — Ψ Evolution Over Time",
    palette: str = DEFAULT_PALETTE,
    animate: bool = True,
    save: Optional[str] = None,
    engine: str = "plotly",
) -> Any:
    """Animated trajectory through Ψ-space showing temporal evolution.

    Ideal for visualizing chishu (持樞) 4-phase lifecycle transitions:
    生 (sheng) → 長 (zhang) → 收 (shou) → 藏 (cang).

    Parameters
    ----------
    profiles : list of PsiProfile / dict / array
        Ordered sequence of profiles over time.
    time_labels : list of str, optional
        Labels for each time step (e.g., ["t=0", "t=1", ...]).
    axes : tuple of 3 ints
        Which Ψ dimensions for X, Y, Z. Default: (0, 1, 2).
    color_dim : int
        Which Ψ dimension for color. Default: 3.
    phase_boundaries : list of int, optional
        Time-step indices where chishu phases transition.
        E.g., [0, 5, 12, 18] means 生 at t=0, 長 at t=5, etc.
    title : str
        Chart title.
    palette : str
        Color palette name.
    animate : bool
        If True (plotly only), add animation frames showing the trajectory
        building up step by step.
    save : str, optional
        File path to save. None = show interactive.
    engine : str
        'plotly' or 'matplotlib'.

    Returns
    -------
    Figure object

    Examples
    --------
    >>> # Monitor an AI agent over 20 conversation turns
    >>> profiles = [compute_psi_from_sequence(turn) for turn in conversation]
    >>> psi_trajectory(profiles, phase_boundaries=[0, 5, 12, 18])
    """
    pal = get_palette(palette)
    psi_matrix = np.array([_extract_psi(p) for p in profiles])
    n = len(profiles)

    x = psi_matrix[:, axes[0]]
    y = psi_matrix[:, axes[1]]
    z = psi_matrix[:, axes[2]]
    c = psi_matrix[:, color_dim]

    x_label = PSI_SHORT[axes[0]]
    y_label = PSI_SHORT[axes[1]]
    z_label = PSI_SHORT[axes[2]]

    if time_labels is None:
        time_labels = [f"t={i}" for i in range(n)]

    if engine == "plotly":
        go, pio = _ensure_plotly()

        fig = go.Figure()

        # Draw phase-colored trajectory segments
        if phase_boundaries and len(phase_boundaries) >= 2:
            boundaries = sorted(phase_boundaries)
            for pi in range(len(boundaries)):
                start = boundaries[pi]
                end = boundaries[pi + 1] if pi + 1 < len(boundaries) else n
                seg_color = pal["chishu"][min(pi, 3)]
                phase_name = CHISHU_PHASES[min(pi, 3)]

                fig.add_trace(go.Scatter3d(
                    x=x[start:end], y=y[start:end], z=z[start:end],
                    mode="lines",
                    line=dict(color=seg_color, width=5),
                    name=f"{phase_name['char']} {phase_name['en']}",
                    hoverinfo="skip",
                ))
        else:
            # Single gradient trajectory
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(
                    color=c,
                    colorscale=[
                        [0.0, pal["psi_colors"][0]],
                        [1.0, pal["psi_colors"][3]],
                    ],
                    width=4,
                ),
                hoverinfo="skip",
                name="Trajectory",
            ))

        # Scatter points
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=5,
                color=c,
                colorscale=[
                    [0.0, pal["psi_colors"][0]],
                    [1.0, pal["psi_colors"][3]],
                ],
                colorbar=dict(
                    title=dict(text=PSI_SHORT[color_dim], font=dict(color=pal["text"])),
                    tickfont=dict(color=pal["dim"]),
                ),
                opacity=0.9,
                line=dict(width=0.5, color=pal["grid"]),
            ),
            text=time_labels,
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{x_label}: %{{x:.4f}}<br>"
                f"{y_label}: %{{y:.4f}}<br>"
                f"{z_label}: %{{z:.4f}}<extra></extra>"
            ),
            name="Time Steps",
            showlegend=False,
        ))

        # Start / end markers
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers+text",
            marker=dict(size=10, color=pal["psi_colors"][0], symbol="diamond"),
            text=["START"], textposition="top center",
            textfont=dict(color=pal["psi_colors"][0], size=11),
            name="Start", showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode="markers+text",
            marker=dict(size=10, color=pal["psi_colors"][3], symbol="diamond"),
            text=["END"], textposition="top center",
            textfont=dict(color=pal["psi_colors"][3], size=11),
            name="End", showlegend=False,
        ))

        # Animation frames (plotly only)
        if animate and save is None:
            frames = []
            for k in range(2, n + 1):
                frame_data = []
                frame_data.append(go.Scatter3d(
                    x=x[:k], y=y[:k], z=z[:k],
                    mode="lines",
                    line=dict(color=pal["accent"], width=4),
                ))
                frame_data.append(go.Scatter3d(
                    x=x[:k], y=y[:k], z=z[:k],
                    mode="markers",
                    marker=dict(size=5, color=list(c[:k]),
                                colorscale=[[0, pal["psi_colors"][0]],
                                            [1, pal["psi_colors"][3]]]),
                ))
                frames.append(go.Frame(data=frame_data, name=f"t={k-1}"))

            if frames:
                fig.frames = frames
                fig.update_layout(
                    updatemenus=[dict(
                        type="buttons",
                        showactive=False,
                        y=0,
                        x=0.5,
                        xanchor="center",
                        buttons=[
                            dict(label="▶ Play",
                                 method="animate",
                                 args=[None, dict(
                                     frame=dict(duration=200, redraw=True),
                                     fromcurrent=True,
                                     transition=dict(duration=100),
                                 )]),
                            dict(label="⏸ Pause",
                                 method="animate",
                                 args=[[None], dict(
                                     frame=dict(duration=0, redraw=False),
                                     mode="immediate",
                                 )]),
                        ],
                        font=dict(color=pal["text"]),
                        bgcolor=pal["surface"],
                        bordercolor=pal["grid"],
                    )],
                    sliders=[dict(
                        active=0,
                        steps=[dict(args=[[f.name], dict(
                            mode="immediate",
                            frame=dict(duration=200, redraw=True),
                        )], label=f.name) for f in frames],
                        x=0.05, len=0.9,
                        currentvalue=dict(prefix="Step: ", font=dict(color=pal["text"])),
                        font=dict(color=pal["dim"]),
                        bgcolor=pal["surface"],
                        bordercolor=pal["grid"],
                        activebgcolor=pal["accent"],
                    )],
                )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=pal["text"])),
            scene=dict(
                xaxis=dict(title=x_label, gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                yaxis=dict(title=y_label, gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                zaxis=dict(title=z_label, gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                bgcolor=pal["bg"],
            ),
            paper_bgcolor=pal["bg"],
            font=dict(color=pal["text"]),
            margin=dict(t=80, b=60, l=20, r=20),
        )
        _save_or_show(fig, save, engine="plotly")
        return fig

    elif engine == "matplotlib":
        matplotlib, plt = _ensure_matplotlib()

        fig = plt.figure(figsize=(12, 9))
        fig.set_facecolor(pal["bg"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(pal["surface"])

        if phase_boundaries and len(phase_boundaries) >= 2:
            boundaries = sorted(phase_boundaries)
            for pi in range(len(boundaries)):
                start = boundaries[pi]
                end = boundaries[pi + 1] if pi + 1 < len(boundaries) else n
                seg_color = pal["chishu"][min(pi, 3)]
                phase = CHISHU_PHASES[min(pi, 3)]
                ax.plot(x[start:end], y[start:end], z[start:end],
                        color=seg_color, linewidth=2.5,
                        label=f"{phase['char']} {phase['en']}")
        else:
            ax.plot(x, y, z, color=pal["accent"], linewidth=2)

        sc = ax.scatter(x, y, z, c=c, s=40, cmap="viridis",
                        edgecolors=pal["grid"], linewidth=0.3, alpha=0.85, zorder=5)

        # Start / end
        ax.scatter([x[0]], [y[0]], [z[0]], c=pal["psi_colors"][0],
                   s=120, marker="D", zorder=10, edgecolors="white", linewidth=0.8)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], c=pal["psi_colors"][3],
                   s=120, marker="D", zorder=10, edgecolors="white", linewidth=0.8)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label(PSI_SHORT[color_dim], color=pal["text"])
        for label in cbar.ax.get_yticklabels():
            label.set_color(pal["dim"])

        ax.set_xlabel(x_label, color=pal["text"], fontsize=12)
        ax.set_ylabel(y_label, color=pal["text"], fontsize=12)
        ax.set_zlabel(z_label, color=pal["text"], fontsize=12)
        ax.set_title(title, color=pal["text"], fontsize=14, pad=20)
        ax.tick_params(colors=pal["dim"])

        if phase_boundaries:
            ax.legend(facecolor=pal["surface"], edgecolor=pal["grid"],
                      labelcolor=pal["text"])

        _save_or_show(fig, save, engine="matplotlib")
        return fig

    else:
        raise ValueError(f"engine must be 'plotly' or 'matplotlib', got '{engine}'")


# ═════════════════════════════════════════════════════════════════════
# 5. PSI HEATMAP — MULTI-PROFILE COMPARISON
# ═════════════════════════════════════════════════════════════════════

def psi_heatmap(
    profiles: List[Any],
    labels: Optional[List[str]] = None,
    title: str = "Ψ Entropy Heatmap — Profile Comparison",
    palette: str = DEFAULT_PALETTE,
    annotate: bool = True,
    save: Optional[str] = None,
    engine: str = "plotly",
) -> Any:
    """Heatmap comparing Ψ dimensions across multiple profiles.

    Rows = profiles, columns = Ψ₁–Ψ₄. Color intensity = value.

    Parameters
    ----------
    profiles : list of PsiProfile / dict / array
    labels : list of str, optional
        Row labels for each profile.
    title : str
    palette : str
    annotate : bool
        Show numeric values in cells.
    save : str, optional
    engine : str
        'plotly' or 'matplotlib'.

    Returns
    -------
    Figure object

    Examples
    --------
    >>> # Compare 11 AI models from Palisade study
    >>> psi_heatmap(model_profiles, labels=model_names)
    """
    pal = get_palette(palette)
    psi_matrix = np.array([_extract_psi(p) for p in profiles])

    if labels is None:
        labels = [f"Profile {i+1}" for i in range(len(profiles))]

    if engine == "plotly":
        go, pio = _ensure_plotly()

        text_matrix = [[f"{v:.4f}" for v in row] for row in psi_matrix] if annotate else None

        fig = go.Figure(data=go.Heatmap(
            z=psi_matrix,
            x=PSI_LABELS,
            y=labels,
            colorscale=[
                [0.0, pal["bg"]],
                [0.25, pal["psi_colors"][0]],
                [0.5, pal["psi_colors"][1]],
                [0.75, pal["psi_colors"][2]],
                [1.0, pal["psi_colors"][3]],
            ],
            text=text_matrix,
            texttemplate="%{text}" if annotate else None,
            textfont=dict(size=11, color=pal["text"]),
            hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
            colorbar=dict(
                title=dict(text="Entropy", font=dict(color=pal["text"])),
                tickfont=dict(color=pal["dim"]),
            ),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=pal["text"])),
            paper_bgcolor=pal["bg"],
            plot_bgcolor=pal["surface"],
            font=dict(color=pal["text"]),
            xaxis=dict(side="top", tickfont=dict(size=12)),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            margin=dict(t=100, b=40, l=150, r=40),
        )
        _save_or_show(fig, save, engine="plotly")
        return fig

    elif engine == "matplotlib":
        matplotlib, plt = _ensure_matplotlib()

        fig, ax = plt.subplots(figsize=(8, max(4, len(profiles) * 0.5 + 2)))
        fig.set_facecolor(pal["bg"])
        ax.set_facecolor(pal["surface"])

        im = ax.imshow(psi_matrix, aspect="auto", cmap="viridis")

        ax.set_xticks(range(4))
        ax.set_xticklabels(PSI_LABELS, fontsize=11, color=pal["text"])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10, color=pal["text"])
        ax.xaxis.tick_top()

        if annotate:
            for i in range(psi_matrix.shape[0]):
                for j in range(psi_matrix.shape[1]):
                    ax.text(j, i, f"{psi_matrix[i, j]:.3f}",
                            ha="center", va="center", fontsize=9,
                            color="white" if psi_matrix[i, j] > psi_matrix.mean() else "black")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Entropy", color=pal["text"])
        for label in cbar.ax.get_yticklabels():
            label.set_color(pal["dim"])

        ax.set_title(title, color=pal["text"], fontsize=14, pad=20)
        fig.tight_layout()

        _save_or_show(fig, save, engine="matplotlib")
        return fig


# ═════════════════════════════════════════════════════════════════════
# 6. PSI TIME SERIES
# ═════════════════════════════════════════════════════════════════════

def psi_timeseries(
    profiles: List[Any],
    time_labels: Optional[List[str]] = None,
    phase_boundaries: Optional[List[int]] = None,
    dimensions: Tuple[int, ...] = (0, 1, 2, 3),
    title: str = "Ψ Dimensions Over Time",
    palette: str = DEFAULT_PALETTE,
    save: Optional[str] = None,
    engine: str = "plotly",
) -> Any:
    """2D time series of selected Ψ dimensions with optional chishu phase
    annotations.

    Parameters
    ----------
    profiles : list of PsiProfile / dict / array
        Ordered time sequence.
    time_labels : list of str, optional
        X-axis labels.
    phase_boundaries : list of int, optional
        Indices where chishu phases transition (draws vertical lines).
    dimensions : tuple of int
        Which Ψ dimensions to plot (0-indexed). Default: all four.
    title : str
    palette : str
    save : str, optional
    engine : str

    Returns
    -------
    Figure object

    Examples
    --------
    >>> psi_timeseries(profiles, phase_boundaries=[0, 5, 12, 18])
    >>> psi_timeseries(profiles, dimensions=(1, 2))  # just Ψ₂ and Ψ₃
    """
    pal = get_palette(palette)
    psi_matrix = np.array([_extract_psi(p) for p in profiles])
    n = len(profiles)
    t = np.arange(n)

    if time_labels is None:
        time_labels = [str(i) for i in range(n)]

    if engine == "plotly":
        go, pio = _ensure_plotly()

        fig = go.Figure()

        for dim in dimensions:
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=psi_matrix[:, dim],
                mode="lines+markers",
                name=PSI_LABELS[dim],
                line=dict(color=pal["psi_colors"][dim], width=2.5),
                marker=dict(size=5, color=pal["psi_colors"][dim]),
                hovertemplate=f"{PSI_SHORT[dim]}: %{{y:.4f}}<extra></extra>",
            ))

        # Phase boundary annotations
        if phase_boundaries:
            for pi, idx in enumerate(phase_boundaries):
                if idx < n:
                    phase = CHISHU_PHASES[min(pi, 3)]
                    label_text = time_labels[idx] if idx < len(time_labels) else str(idx)
                    fig.add_vline(
                        x=label_text,
                        line=dict(color=pal["chishu"][min(pi, 3)], width=2, dash="dot"),
                    )
                    fig.add_annotation(
                        x=label_text,
                        y=1.05,
                        yref="paper",
                        text=f"{phase['char']} {phase['en']}",
                        showarrow=False,
                        font=dict(color=pal["chishu"][min(pi, 3)], size=11),
                    )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=pal["text"])),
            xaxis=dict(title="Time Step", gridcolor=pal["grid"], color=pal["text"]),
            yaxis=dict(title="Entropy Value", gridcolor=pal["grid"], color=pal["text"]),
            paper_bgcolor=pal["bg"],
            plot_bgcolor=pal["surface"],
            font=dict(color=pal["text"]),
            legend=dict(
                bgcolor=pal["surface"],
                bordercolor=pal["grid"],
                borderwidth=1,
                font=dict(color=pal["text"]),
            ),
            margin=dict(t=80, b=60, l=60, r=40),
            hovermode="x unified",
        )
        _save_or_show(fig, save, engine="plotly")
        return fig

    elif engine == "matplotlib":
        matplotlib, plt = _ensure_matplotlib()

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor(pal["bg"])
        ax.set_facecolor(pal["surface"])

        for dim in dimensions:
            ax.plot(t, psi_matrix[:, dim], "o-",
                    color=pal["psi_colors"][dim], linewidth=2,
                    markersize=4, label=PSI_LABELS[dim])

        if phase_boundaries:
            for pi, idx in enumerate(phase_boundaries):
                if idx < n:
                    phase = CHISHU_PHASES[min(pi, 3)]
                    color = pal["chishu"][min(pi, 3)]
                    ax.axvline(x=idx, color=color, linestyle=":", linewidth=1.5)
                    ax.text(idx + 0.3, ax.get_ylim()[1] * 0.95,
                            f"{phase['char']} {phase['en']}",
                            color=color, fontsize=9, va="top")

        ax.set_xlabel("Time Step", color=pal["text"], fontsize=12)
        ax.set_ylabel("Entropy Value", color=pal["text"], fontsize=12)
        ax.set_title(title, color=pal["text"], fontsize=14, pad=15)
        ax.legend(facecolor=pal["surface"], edgecolor=pal["grid"],
                  labelcolor=pal["text"])
        ax.tick_params(colors=pal["dim"])
        ax.grid(color=pal["grid"], linewidth=0.3)

        _save_or_show(fig, save, engine="matplotlib")
        return fig


# ═════════════════════════════════════════════════════════════════════
# 7. PARAMETER SWEEP SURFACE
# ═════════════════════════════════════════════════════════════════════

def psi_parameter_sweep(
    sweep_fn,
    param1_range: Tuple[float, float],
    param2_range: Tuple[float, float],
    psi_dim: int = 0,
    param1_name: str = "Parameter 1",
    param2_name: str = "Parameter 2",
    resolution: int = 50,
    title: Optional[str] = None,
    palette: str = DEFAULT_PALETTE,
    save: Optional[str] = None,
    engine: str = "plotly",
) -> Any:
    """3D surface showing how a Ψ dimension varies across two parameters.

    Parameters
    ----------
    sweep_fn : callable
        Function(param1, param2) -> PsiProfile or array-like [Ψ₁,Ψ₂,Ψ₃,Ψ₄].
    param1_range, param2_range : tuple of float
        (min, max) for each parameter.
    psi_dim : int
        Which Ψ dimension to plot on Z-axis (0-3).
    param1_name, param2_name : str
        Axis labels.
    resolution : int
        Grid resolution per axis.
    title : str, optional
    palette : str
    save : str, optional
    engine : str

    Returns
    -------
    Figure object

    Examples
    --------
    >>> def my_model(price, quality):
    ...     data = simulate_decisions(price, quality)
    ...     return compute_psi_from_sequence(data)
    >>> psi_parameter_sweep(my_model, (10, 100), (1, 10),
    ...                     psi_dim=2, param1_name="Price", param2_name="Quality")
    """
    pal = get_palette(palette)

    p1 = np.linspace(param1_range[0], param1_range[1], resolution)
    p2 = np.linspace(param2_range[0], param2_range[1], resolution)
    P1, P2 = np.meshgrid(p1, p2)
    Z = np.zeros_like(P1)

    for i in range(resolution):
        for j in range(resolution):
            result = sweep_fn(P1[i, j], P2[i, j])
            psi = _extract_psi(result)
            Z[i, j] = psi[psi_dim]

    if title is None:
        title = f"{PSI_SHORT[psi_dim]} Response Surface"

    if engine == "plotly":
        go, pio = _ensure_plotly()

        fig = go.Figure(data=go.Surface(
            x=P1, y=P2, z=Z,
            colorscale=[
                [0.0, pal["psi_colors"][0]],
                [0.5, pal["psi_colors"][1]],
                [1.0, pal["psi_colors"][3]],
            ],
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=True),
            ),
            hovertemplate=(
                f"{param1_name}: %{{x:.3f}}<br>"
                f"{param2_name}: %{{y:.3f}}<br>"
                f"{PSI_SHORT[psi_dim]}: %{{z:.4f}}<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=pal["text"])),
            scene=dict(
                xaxis=dict(title=param1_name, gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                yaxis=dict(title=param2_name, gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                zaxis=dict(title=PSI_SHORT[psi_dim], gridcolor=pal["grid"],
                           backgroundcolor=pal["surface"], color=pal["text"]),
                bgcolor=pal["bg"],
            ),
            paper_bgcolor=pal["bg"],
            font=dict(color=pal["text"]),
            margin=dict(t=80, b=20, l=20, r=20),
        )
        _save_or_show(fig, save, engine="plotly")
        return fig

    elif engine == "matplotlib":
        matplotlib, plt = _ensure_matplotlib()

        fig = plt.figure(figsize=(12, 8))
        fig.set_facecolor(pal["bg"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(pal["surface"])

        ax.plot_surface(P1, P2, Z, cmap="viridis", alpha=0.85,
                        edgecolor=pal["grid"], linewidth=0.1)

        ax.set_xlabel(param1_name, color=pal["text"], fontsize=12)
        ax.set_ylabel(param2_name, color=pal["text"], fontsize=12)
        ax.set_zlabel(PSI_SHORT[psi_dim], color=pal["text"], fontsize=12)
        ax.set_title(title, color=pal["text"], fontsize=14, pad=20)
        ax.tick_params(colors=pal["dim"])

        _save_or_show(fig, save, engine="matplotlib")
        return fig


# ═════════════════════════════════════════════════════════════════════
# EXPORT: MATLAB
# ═════════════════════════════════════════════════════════════════════

def export_matlab(
    profiles: Union[Any, List[Any]],
    filepath: str = "eap_profiles.mat",
    labels: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Export Psi profiles to a MATLAB .mat file.

    MATLAB usage:
        load('eap_profiles.mat')
        disp(psi_matrix)    % [N x 4] matrix of Ψ values
        disp(psi_labels)    % {'Psi1_Info', 'Psi2_Behav', 'Psi3_Adapt', 'Psi4_Relat'}
        disp(profile_names) % cell array of profile labels

    Parameters
    ----------
    profiles : PsiProfile, dict, array, or list of these
    filepath : str
        Output .mat file path. Default: 'eap_profiles.mat'.
    labels : list of str, optional
        Profile labels.
    extra : dict, optional
        Additional variables to include in the .mat file.

    Returns
    -------
    str : absolute path of saved file

    Examples
    --------
    >>> export_matlab(profile, "single.mat")
    >>> export_matlab([p1, p2, p3], "comparison.mat", labels=["A", "B", "C"])
    """
    try:
        from scipy.io import savemat
    except ImportError:
        raise ImportError(
            "scipy is required for MATLAB export.\n"
            "Install with: pip install scipy"
        )

    if not isinstance(profiles, list):
        profiles = [profiles]

    psi_matrix = np.array([_extract_psi(p) for p in profiles])

    if labels is None:
        labels = [f"Profile_{i+1}" for i in range(len(profiles))]

    mat_dict = {
        "psi_matrix": psi_matrix,
        "psi_labels": np.array(["Psi1_Info", "Psi2_Behav", "Psi3_Adapt", "Psi4_Relat"],
                               dtype=object),
        "profile_names": np.array(labels, dtype=object),
        "psi1": psi_matrix[:, 0],
        "psi2": psi_matrix[:, 1],
        "psi3": psi_matrix[:, 2],
        "psi4": psi_matrix[:, 3],
        "n_profiles": len(profiles),
        "eap_version": "2.0.0",
    }

    # Include extra fields from PsiProfile dataclass
    for i, p in enumerate(profiles):
        d = _profile_to_dict(p)
        for key, val in d.items():
            if key not in PSI_KEYS and key != "label" and val is not None:
                mat_key = f"{key}" if len(profiles) == 1 else f"{key}_{i}"
                if isinstance(val, (int, float)):
                    mat_dict[mat_key] = val

    if extra:
        mat_dict.update(extra)

    savemat(filepath, mat_dict, do_compression=True)
    abs_path = os.path.abspath(filepath)
    print(f"MATLAB export: {abs_path}")
    return abs_path


# ═════════════════════════════════════════════════════════════════════
# EXPORT: CSV
# ═════════════════════════════════════════════════════════════════════

def export_csv(
    profiles: Union[Any, List[Any]],
    filepath: str = "eap_profiles.csv",
    labels: Optional[List[str]] = None,
    include_extras: bool = True,
) -> str:
    """Export Psi profiles to CSV.

    Compatible with R, Excel, SPSS, Stata, pandas, and any tool that reads CSV.

    R usage:
        df <- read.csv("eap_profiles.csv")
        plot(df$psi_1, df$psi_2)

    Parameters
    ----------
    profiles : PsiProfile, dict, array, or list of these
    filepath : str
    labels : list of str, optional
    include_extras : bool
        If True, include additional PsiProfile fields beyond Ψ₁–Ψ₄.

    Returns
    -------
    str : absolute path of saved file
    """
    if not isinstance(profiles, list):
        profiles = [profiles]

    if labels is None:
        labels = [f"Profile_{i+1}" for i in range(len(profiles))]

    rows = []
    for i, p in enumerate(profiles):
        d = _profile_to_dict(p, label=labels[i])
        rows.append(d)

    # Collect all unique keys across rows
    all_keys = ["label"] + PSI_KEYS
    if include_extras:
        extra_keys = set()
        for row in rows:
            extra_keys.update(k for k in row if k not in all_keys)
        all_keys.extend(sorted(extra_keys))

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    abs_path = os.path.abspath(filepath)
    print(f"CSV export: {abs_path}")
    return abs_path


# ═════════════════════════════════════════════════════════════════════
# EXPORT: JSON
# ═════════════════════════════════════════════════════════════════════

def export_json(
    profiles: Union[Any, List[Any]],
    filepath: str = "eap_profiles.json",
    labels: Optional[List[str]] = None,
    indent: int = 2,
) -> str:
    """Export Psi profiles to JSON.

    Compatible with any language: Python json, R jsonlite, MATLAB jsondecode,
    JavaScript JSON.parse, Julia JSON.parse, etc.

    Parameters
    ----------
    profiles : PsiProfile, dict, array, or list of these
    filepath : str
    labels : list of str, optional
    indent : int
        JSON indentation (default 2).

    Returns
    -------
    str : absolute path of saved file
    """
    if not isinstance(profiles, list):
        profiles = [profiles]

    if labels is None:
        labels = [f"Profile_{i+1}" for i in range(len(profiles))]

    data = {
        "eap_version": "2.0.0",
        "n_profiles": len(profiles),
        "dimensions": PSI_LABELS,
        "profiles": [],
    }

    for i, p in enumerate(profiles):
        d = _profile_to_dict(p, label=labels[i])
        # Convert numpy types to native Python for JSON serialization
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, (np.integer,)):
                cleaned[k] = int(v)
            elif isinstance(v, (np.floating,)):
                cleaned[k] = float(v)
            elif isinstance(v, np.ndarray):
                cleaned[k] = v.tolist()
            else:
                cleaned[k] = v
        data["profiles"].append(cleaned)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    abs_path = os.path.abspath(filepath)
    print(f"JSON export: {abs_path}")
    return abs_path


# ═════════════════════════════════════════════════════════════════════
# MATLAB SCRIPT GENERATOR
# ═════════════════════════════════════════════════════════════════════

def generate_matlab_script(
    mat_filepath: str = "eap_profiles.mat",
    output_script: str = "eap_visualize.m",
) -> str:
    """Generate a ready-to-run MATLAB .m script for visualizing EAP data.

    The script loads the .mat file and produces:
    - Radar chart of all profiles
    - 3D scatter plot
    - Bar chart comparison
    - Heatmap

    Parameters
    ----------
    mat_filepath : str
        Path to the .mat file (used in the load() call).
    output_script : str
        Output .m file path.

    Returns
    -------
    str : absolute path of the generated .m script
    """
    script = f"""%% EAP Visualization Script — Auto-generated by Insight137 EAP v2.0.0
%% Load data exported by insight137_eap_viz.export_matlab()
%%
%% Usage: Run this script in MATLAB after placing '{mat_filepath}' in your
%%        working directory.
%%
%% Author: Insight137 (insight137.com)
%% License: CC BY-NC-ND 4.0

clear; clc; close all;

%% ── Load Data ──────────────────────────────────────────────────────
data = load('{mat_filepath}');
psi = data.psi_matrix;          % [N x 4] matrix
labels = data.psi_labels;       % {{'Psi1_Info', 'Psi2_Behav', 'Psi3_Adapt', 'Psi4_Relat'}}
names = data.profile_names;     % cell array of profile names
N = size(psi, 1);

fprintf('Loaded %d profiles from %s\\n', N, '{mat_filepath}');

%% ── Color palette ──────────────────────────────────────────────────
colors = [
    0.306 0.804 0.769;   % Psi1 teal
    0.357 0.553 0.933;   % Psi2 blue
    0.773 0.482 0.859;   % Psi3 purple
    0.910 0.365 0.459;   % Psi4 red
];

%% ── Figure 1: Radar Chart ─────────────────────────────────────────
figure('Name', 'EAP Radar Chart', 'Color', 'w', 'Position', [100 100 600 600]);

theta = linspace(0, 2*pi, 5);  % 4 dimensions + close
dim_labels = {{'\\Psi_1 Info', '\\Psi_2 Behav', '\\Psi_3 Adapt', '\\Psi_4 Relat'}};

for i = 1:N
    vals = [psi(i,:), psi(i,1)];  % close polygon
    polarplot(theta, vals, '-o', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
end

ax = gca;
ax.ThetaTick = rad2deg(theta(1:4));
ax.ThetaTickLabel = dim_labels;
ax.FontSize = 12;
title('\\Psi Entropy Profile — Radar', 'FontSize', 16);
if N > 1
    legend(names, 'Location', 'bestoutside');
end

%% ── Figure 2: 3D Scatter ──────────────────────────────────────────
if N >= 2
    figure('Name', 'EAP 3D Scatter', 'Color', 'w', 'Position', [200 100 700 600]);

    scatter3(psi(:,1), psi(:,2), psi(:,3), 100, psi(:,4), 'filled', ...
        'MarkerEdgeColor', [0.3 0.3 0.3], 'LineWidth', 0.5);
    colorbar('Label', '\\Psi_4 Relational');
    colormap(turbo);

    xlabel('\\Psi_1 Informational', 'FontSize', 12);
    ylabel('\\Psi_2 Behavioral', 'FontSize', 12);
    zlabel('\\Psi_3 Adaptive', 'FontSize', 12);
    title('\\Psi Space — 3D Entropy Landscape', 'FontSize', 16);
    grid on;
    rotate3d on;

    % Label points if few enough
    if N <= 20
        for i = 1:N
            text(psi(i,1), psi(i,2), psi(i,3), ['  ' names{{i}}], ...
                'FontSize', 8, 'Color', [0.4 0.4 0.4]);
        end
    end
end

%% ── Figure 3: Bar Chart ───────────────────────────────────────────
figure('Name', 'EAP Bar Comparison', 'Color', 'w', 'Position', [300 100 800 500]);

bar_handle = bar(psi);
for k = 1:4
    bar_handle(k).FaceColor = colors(k,:);
end

set(gca, 'XTickLabel', names, 'FontSize', 10);
xlabel('Profile', 'FontSize', 12);
ylabel('Entropy Value', 'FontSize', 12);
title('\\Psi Dimensions by Profile', 'FontSize', 16);
legend(dim_labels, 'Location', 'bestoutside');
grid on;

%% ── Figure 4: Heatmap ─────────────────────────────────────────────
figure('Name', 'EAP Heatmap', 'Color', 'w', 'Position', [400 100 600 500]);

imagesc(psi);
colorbar('Label', 'Entropy Value');
colormap(turbo);

set(gca, 'XTick', 1:4, 'XTickLabel', dim_labels, ...
    'YTick', 1:N, 'YTickLabel', names, 'FontSize', 10);
title('\\Psi Entropy Heatmap', 'FontSize', 16);

% Annotate cells
for i = 1:N
    for j = 1:4
        text(j, i, sprintf('%.3f', psi(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9, ...
            'Color', 'w', 'FontWeight', 'bold');
    end
end

fprintf('\\nAll figures generated. Use rotate3d to interact with 3D plots.\\n');
"""

    with open(output_script, "w", encoding="utf-8") as f:
        f.write(script)

    abs_path = os.path.abspath(output_script)
    print(f"MATLAB script: {abs_path}")
    return abs_path


# ═════════════════════════════════════════════════════════════════════
# R SCRIPT GENERATOR
# ═════════════════════════════════════════════════════════════════════

def generate_r_script(
    csv_filepath: str = "eap_profiles.csv",
    output_script: str = "eap_visualize.R",
) -> str:
    """Generate a ready-to-run R script for visualizing EAP data.

    Uses ggplot2 and plotly (if available).

    Parameters
    ----------
    csv_filepath : str
        Path to the CSV file exported by export_csv().
    output_script : str
        Output .R file path.

    Returns
    -------
    str : absolute path of the generated .R script
    """
    script = f"""# EAP Visualization Script — Auto-generated by Insight137 EAP v2.0.0
# Load data exported by insight137_eap_viz.export_csv()
#
# Requirements: install.packages(c("ggplot2", "reshape2", "plotly"))
#
# Author: Insight137 (insight137.com)
# License: CC BY-NC-ND 4.0

library(ggplot2)
library(reshape2)

# ── Load Data ──────────────────────────────────────────────────────
df <- read.csv("{csv_filepath}", stringsAsFactors = FALSE)
cat(sprintf("Loaded %d profiles from {csv_filepath}\\n", nrow(df)))

# ── Reshape for ggplot ─────────────────────────────────────────────
psi_cols <- c("psi_1", "psi_2", "psi_3", "psi_4")
psi_labels <- c("Psi1 Informational", "Psi2 Behavioral", "Psi3 Adaptive", "Psi4 Relational")

df_long <- melt(df, id.vars = "label", measure.vars = psi_cols,
                variable.name = "dimension", value.name = "entropy")
df_long$dimension <- factor(df_long$dimension, levels = psi_cols, labels = psi_labels)

# ── Color palette ──────────────────────────────────────────────────
eap_colors <- c(
  "Psi1 Informational" = "#4ecdc4",
  "Psi2 Behavioral"    = "#5b8dee",
  "Psi3 Adaptive"      = "#c57bdb",
  "Psi4 Relational"    = "#e85d75"
)

# ── Plot 1: Grouped bar chart ─────────────────────────────────────
p1 <- ggplot(df_long, aes(x = label, y = entropy, fill = dimension)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = eap_colors) +
  labs(title = expression(Psi ~ "Entropy Profile Comparison"),
       x = "Profile", y = "Entropy Value", fill = "Dimension") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p1)
ggsave("eap_barplot.png", p1, width = 10, height = 6, dpi = 300)

# ── Plot 2: Heatmap ───────────────────────────────────────────────
p2 <- ggplot(df_long, aes(x = dimension, y = label, fill = entropy)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.3f", entropy)), color = "white", size = 3.5) +
  scale_fill_gradient2(low = "#0d0d1a", mid = "#5b8dee", high = "#e85d75",
                       midpoint = median(df_long$entropy)) +
  labs(title = expression(Psi ~ "Entropy Heatmap"),
       x = "Dimension", y = "Profile", fill = "Entropy") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

print(p2)
ggsave("eap_heatmap.png", p2, width = 8, height = 6, dpi = 300)

# ── Plot 3: Radar chart (base R) ──────────────────────────────────
if (nrow(df) <= 6) {{
  stars(df[, psi_cols], labels = df$label,
        main = expression(Psi ~ "Radar Profiles"),
        col.stars = rainbow(nrow(df)),
        key.loc = c(6, 1.5), draw.segments = TRUE)
}}

# ── Plot 4: 3D scatter (plotly if available) ───────────────────────
if (requireNamespace("plotly", quietly = TRUE)) {{
  library(plotly)
  p3d <- plot_ly(df, x = ~psi_1, y = ~psi_2, z = ~psi_3,
                 color = ~psi_4, text = ~label,
                 type = "scatter3d", mode = "markers+text",
                 marker = list(size = 8)) %>%
    layout(title = list(text = "Psi Space - 3D Entropy Landscape"),
           scene = list(
             xaxis = list(title = "Psi1 Informational"),
             yaxis = list(title = "Psi2 Behavioral"),
             zaxis = list(title = "Psi3 Adaptive")
           ))
  print(p3d)
  htmlwidgets::saveWidget(p3d, "eap_3d_scatter.html")
  cat("3D scatter saved to eap_3d_scatter.html\\n")
}} else {{
  cat("Install plotly for 3D visualization: install.packages('plotly')\\n")
}}

cat("\\nAll figures generated.\\n")
"""

    with open(output_script, "w", encoding="utf-8") as f:
        f.write(script)

    abs_path = os.path.abspath(output_script)
    print(f"R script: {abs_path}")
    return abs_path


# ═════════════════════════════════════════════════════════════════════
# CONVENIENCE: FULL REPORT
# ═════════════════════════════════════════════════════════════════════

def full_report(
    profiles: List[Any],
    labels: Optional[List[str]] = None,
    output_dir: str = "eap_report",
    palette: str = DEFAULT_PALETTE,
    formats: Tuple[str, ...] = ("html", "png", "mat", "csv", "json"),
) -> Dict[str, str]:
    """Generate a complete visualization report with all chart types and exports.

    Creates an output directory with:
    - Interactive HTML visualizations (Plotly)
    - Static PNG images (Plotly + kaleido)
    - MATLAB .mat export + .m script
    - R .csv export + .R script
    - JSON export

    Parameters
    ----------
    profiles : list of PsiProfile / dict / array
    labels : list of str, optional
    output_dir : str
        Directory to create for output files.
    palette : str
    formats : tuple of str
        Which output formats to generate.

    Returns
    -------
    dict : mapping of output type -> file path

    Examples
    --------
    >>> files = full_report(profiles, labels=model_names)
    >>> print(files)  # {'radar_html': 'eap_report/radar.html', ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}

    if labels is None:
        labels = [f"Profile_{i+1}" for i in range(len(profiles))]

    # ── Visualizations ──
    if "html" in formats:
        psi_radar(profiles, labels=labels, palette=palette,
                  save=os.path.join(output_dir, "radar.html"))
        outputs["radar_html"] = os.path.join(output_dir, "radar.html")

        if len(profiles) >= 2:
            psi_3d(profiles, labels=labels, palette=palette, show_mesh=True,
                   save=os.path.join(output_dir, "scatter_3d.html"))
            outputs["scatter_3d_html"] = os.path.join(output_dir, "scatter_3d.html")

        psi_heatmap(profiles, labels=labels, palette=palette,
                    save=os.path.join(output_dir, "heatmap.html"))
        outputs["heatmap_html"] = os.path.join(output_dir, "heatmap.html")

    if "png" in formats:
        psi_radar(profiles, labels=labels, palette=palette,
                  save=os.path.join(output_dir, "radar.png"))
        outputs["radar_png"] = os.path.join(output_dir, "radar.png")

        psi_heatmap(profiles, labels=labels, palette=palette,
                    save=os.path.join(output_dir, "heatmap.png"))
        outputs["heatmap_png"] = os.path.join(output_dir, "heatmap.png")

    # ── Data exports ──
    if "mat" in formats:
        export_matlab(profiles, os.path.join(output_dir, "eap_profiles.mat"), labels=labels)
        generate_matlab_script(
            "eap_profiles.mat",
            os.path.join(output_dir, "eap_visualize.m"),
        )
        outputs["matlab_mat"] = os.path.join(output_dir, "eap_profiles.mat")
        outputs["matlab_script"] = os.path.join(output_dir, "eap_visualize.m")

    if "csv" in formats:
        export_csv(profiles, os.path.join(output_dir, "eap_profiles.csv"), labels=labels)
        generate_r_script(
            "eap_profiles.csv",
            os.path.join(output_dir, "eap_visualize.R"),
        )
        outputs["csv"] = os.path.join(output_dir, "eap_profiles.csv")
        outputs["r_script"] = os.path.join(output_dir, "eap_visualize.R")

    if "json" in formats:
        export_json(profiles, os.path.join(output_dir, "eap_profiles.json"), labels=labels)
        outputs["json"] = os.path.join(output_dir, "eap_profiles.json")

    print(f"\nFull report generated in: {os.path.abspath(output_dir)}/")
    print(f"  {len(outputs)} files created.")
    return outputs


# ═════════════════════════════════════════════════════════════════════
# MODULE INFO
# ═════════════════════════════════════════════════════════════════════

__version__ = "2.0.0"
__author__ = "Roger Yau (Jus) — Insight137"
__license__ = "CC BY-NC-ND 4.0"

__all__ = [
    # Visualizations
    "psi_radar",
    "psi_3d",
    "interference_surface",
    "psi_trajectory",
    "psi_heatmap",
    "psi_timeseries",
    "psi_parameter_sweep",
    # Exports
    "export_matlab",
    "export_csv",
    "export_json",
    # Script generators
    "generate_matlab_script",
    "generate_r_script",
    # Convenience
    "full_report",
    # Config
    "get_palette",
    "PSI_LABELS",
    "PSI_SHORT",
    "CHISHU_PHASES",
]
