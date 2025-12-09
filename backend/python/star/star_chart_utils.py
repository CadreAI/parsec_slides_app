"""
Common chart drawing utilities for STAR charts
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
# Add parent directory to path to import helper_functions
sys.path.insert(0, str(Path(__file__).parent.parent))
import helper_functions as hf

# Global threshold for inline % labels on stacked bars
LABEL_MIN_PCT = 5.0


def draw_stacked_bar(ax, pct_df, score_df, labels):
    """Draw stacked bar chart for STAR benchmark achievement levels"""
    if pct_df.empty or len(pct_df) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return
    
    try:
        stack_df = (
            pct_df.pivot(
                index="time_label", columns="state_benchmark_achievement", values="pct"
            )
            .reindex(columns=hf.STAR_ORDER)
            .fillna(0)
        )
    except Exception as e:
        ax.text(0.5, 0.5, f"Error processing data: {str(e)}", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return
    
    if stack_df.empty or len(stack_df) == 0 or len(stack_df.index) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return
    
    x_labels = stack_df.index.tolist()
    if len(x_labels) == 0:
        ax.text(0.5, 0.5, "No time periods available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return
    
    x = np.arange(len(x_labels))
    cumulative = np.zeros(len(stack_df))
    
    for cat in hf.STAR_ORDER:
        if cat not in stack_df.columns:
            continue
        try:
            band_vals = stack_df[cat].to_numpy()
            if len(band_vals) == 0:
                continue
            bars = ax.bar(
                x,
                band_vals,
                bottom=cumulative,
                color=hf.STAR_COLORS[cat],
                edgecolor="white",
                linewidth=1.2,
            )
            for idx, rect in enumerate(bars):
                if idx >= len(band_vals):
                    break
                h = band_vals[idx]
                if h >= LABEL_MIN_PCT and idx < len(cumulative):
                    bottom_before = cumulative[idx]
                    # Determine label color based on benchmark level
                    if cat == "2 - Standard Nearly Met":
                        label_color = "#434343"
                    else:
                        label_color = "white"
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        bottom_before + h / 2,
                        f"{h:.2f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=label_color,
                    )
            if len(band_vals) == len(cumulative):
                cumulative += band_vals
        except Exception as e:
            # Skip this category if there's an error
            continue
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of Students")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_score_bar(ax, score_df, labels, n_map=None):
    """Draw average unified scale score bar chart"""
    if score_df.empty or len(score_df) == 0:
        ax.text(0.5, 0.5, "No score data available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return
    
    if "time_label" not in score_df.columns or "avg_score" not in score_df.columns:
        ax.text(0.5, 0.5, "Missing required columns", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return
    
    if len(score_df["time_label"]) == 0:
        ax.text(0.5, 0.5, "No time periods available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return
    
    try:
        rit_x = np.arange(len(score_df["time_label"]))
        rit_vals = score_df["avg_score"].to_numpy()
        
        if len(rit_vals) == 0:
            ax.text(0.5, 0.5, "No score values available", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return
        
        if len(hf.default_quintile_colors) < 5:
            bar_color = "#4A90E2"
        else:
            bar_color = hf.default_quintile_colors[4]
        
        rit_bars = ax.bar(rit_x, rit_vals, color=bar_color, edgecolor="white", linewidth=1.2)
        for rect, v in zip(rit_bars, rit_vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#434343",
            )
        
        # Build x-axis labels (optionally with n-counts)
        base_labels = score_df["time_label"].astype(str).tolist()
        if n_map is not None:
            labels_with_n = [
                f"{lbl}\n(n = {int(n_map.get(lbl, 0))})" for lbl in base_labels
            ]
        else:
            labels_with_n = base_labels
        
        ax.set_ylabel("Avg Unified Scale Score")
        ax.set_xticks(rit_x)
        ax.set_xticklabels(labels_with_n)
        ax.tick_params(pad=10)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error drawing chart: {str(e)}", ha="center", va="center", fontsize=10)
        ax.axis("off")


def draw_insight_card(ax, metrics, title):
    """Draw insight text box for STAR charts"""
    ax.axis("off")
    if metrics and metrics.get("t_prev"):
        t_prev, t_curr = metrics["t_prev"], metrics["t_curr"]
        # Prioritize high_delta if available (for "Standard Exceeded" only)
        high_delta = metrics.get("high_delta")
        if high_delta is None:
            high_delta = metrics.get("hi_delta", 0)
        hi_delta = metrics.get("hi_delta", 0)  # Meet or Exceed
        lo_delta = metrics.get("lo_delta", 0)
        score_delta = metrics.get("score_delta", 0)
        
        insight_lines = [
            f"Comparison of current and prior year",
            rf"$\Delta$ Exceed: $\mathbf{{{high_delta:+.1f}}}$ ppts",
            rf"$\Delta$ Meet or Exceed: $\mathbf{{{hi_delta:+.1f}}}$ ppts",
            rf"$\Delta$ Not Met: $\mathbf{{{lo_delta:+.1f}}}$ ppts",
            rf"$\Delta$ Avg Unified Scale Score: $\mathbf{{{score_delta:+.1f}}}$ pts",
        ]
    else:
        insight_lines = ["Not enough history for change insights"]
    
    ax.text(
        0.5,
        0.5,
        "\n".join(insight_lines),
        fontsize=11,
        fontweight="normal",
        color="#434343",
        ha="center",
        va="center",
        wrap=True,
        usetex=False,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8),
    )

