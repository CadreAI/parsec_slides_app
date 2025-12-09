"""
Minimal helper_functions.py for nwea.py only
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Global chart style
# ---------------------------------------------------------------------
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"]
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "legend.frameon": True,
    }
)

# ---------------------------------------------------------------------
# Color Palettes
# ---------------------------------------------------------------------
default_quintile_colors = ["#808080", "#c5c5c5", "#78daf4", "#00baeb", "#0381a2"]
default_quartile_colors = ["#808080", "#c5c5c5", "#00baeb", "#0381a2"]

# ---------------------------------------------------------------------
# CAASPP/CERS Level Definitions
# ---------------------------------------------------------------------
CERS_LEVELS = [
    "Level 1 - Standard Not Met",
    "Level 2 - Standard Nearly Met",
    "Level 3 - Standard Met",
    "Level 4 - Standard Exceeded",
]
CERS_LEVEL_COLORS = {
    lvl: default_quartile_colors[i] for i, lvl in enumerate(CERS_LEVELS)
}

# ---------------------------------------------------------------------
# NWEA Definitions
# ---------------------------------------------------------------------
NWEA_ORDER = [
    "Low",
    "LoAvg",
    "Avg",
    "HiAvg",
    "High",
]
NWEA_COLORS = {cat: default_quintile_colors[i] for i, cat in enumerate(NWEA_ORDER)}

# Rollups for insights
NWEA_HIGH_GROUP = ["Avg", "HiAvg", "High"]
NWEA_LOW_GROUP = ["Low"]

# ---------------------------------------------------------------------
# iReady Definitions
# ---------------------------------------------------------------------
IREADY_ORDER = [
    "3+ Below",
    "2 Below",
    "1 Below",
    "Early On",
    "Mid/Above",
]

IREADY_COLORS = {
    band: default_quintile_colors[i] for i, band in enumerate(IREADY_ORDER)
}

IREADY_HIGH_GROUP = ["Early On", "Mid/Above"]
IREADY_LOW_GROUP = ["3+ Below", "2 Below"]

IREADY_LABEL_MAP = {
    "3 or More Grade Levels Below": "3+ Below",
    "2 Grade Levels Below": "2 Below",
    "1 Grade Level Below": "1 Below",
    "Early On Grade Level": "Early On",
    "Mid or Above Grade Level": "Mid/Above",
}

# ---------------------------------------------------------------------
# STAR Definitions
# ---------------------------------------------------------------------
STAR_ORDER = [
    "1 - Standard Not Met",
    "2 - Standard Nearly Met",
    "3 - Standard Met",
    "4 - Standard Exceeded",
]

STAR_COLORS = {
    "1 - Standard Not Met": default_quintile_colors[0],  # darkest gray
    "2 - Standard Nearly Met": default_quintile_colors[1],  # light gray
    "3 - Standard Met": default_quintile_colors[3],  # bright teal
    "4 - Standard Exceeded": default_quintile_colors[4],  # deep teal
}

STAR_HIGH_GROUP = [
    "3 - Standard Met",
    "4 - Standard Exceeded",
]
STAR_LOW_GROUP = ["1 - Standard Not Met"]

# Note: STAR-specific column definitions and helper functions are now in star/star_helper_functions.py

# ---------------------------------------------------------------------
# Global DEV_MODE flag
# ---------------------------------------------------------------------
DEV_MODE = False

# ---------------------------------------------------------------------
# Save and Render Function
# ---------------------------------------------------------------------
def _save_and_render(
    fig: plt.Figure, out_path: Path | None = None, dev_mode: bool | None = None
):
    """
    Save the figure if a path is provided, then either show (DEV_MODE)
    or close to keep batch runs non-interactive.
    Precedence: explicit dev_mode arg > helper_functions.DEV_MODE
    """
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    effective_dev = DEV_MODE if dev_mode is None else bool(dev_mode)
    if effective_dev:
        plt.show(block=False)
        plt.pause(0.001)  # allow nonblocking preview in VSCode
    else:
        plt.close(fig)

# ---------------------------------------------------------------------
# School Name Normalization
# ---------------------------------------------------------------------
def _safe_normalize_school_name(raw_school, cfg_dict):
    """
    Return a cleaned display name using cfg['school_name_map'] when available.
    Accepts dict or list-like configs. Falls back to the original string.
    District (None) falls back to None.
    """
    if raw_school is None:
        return None
    s = str(raw_school)
    raw_map = (cfg_dict or {}).get("school_name_map", {})
    lower_map = {}

    # Normalize to a {lower_from: to} dict if possible
    if isinstance(raw_map, dict):
        lower_map = {str(k).lower(): v for k, v in raw_map.items()}
    elif isinstance(raw_map, (list, tuple)):
        # Support list of 2-tuples/lists or list of small dicts {'from':..., 'to':...}
        for item in raw_map:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
                lower_map[str(k).lower()] = v
            elif isinstance(item, dict):
                if "from" in item and "to" in item:
                    lower_map[str(item["from"]).lower()] = item["to"]
                elif "raw" in item and "display" in item:
                    lower_map[str(item["raw"]).lower()] = item["display"]
                elif len(item) == 1:
                    # single-key dict: {'Old Name': 'New Name'}
                    k, v = next(iter(item.items()))
                    lower_map[str(k).lower()] = v
            else:
                # ignore unsupported shapes
                pass
    else:
        # Not a dict or list; ignore
        lower_map = {}

    return lower_map.get(s.lower(), s)

