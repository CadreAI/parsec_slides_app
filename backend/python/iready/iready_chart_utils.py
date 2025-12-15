"""
Common chart drawing utilities for iReady charts
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
    """Draw stacked bar chart"""
    if pct_df.empty or len(pct_df) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return
    
    try:
        stack_df = (
            pct_df.pivot(index="time_label", columns="relative_placement", values="pct")
            .reindex(columns=hf.IREADY_ORDER)
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
    
    for cat in hf.IREADY_ORDER:
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
                color=hf.IREADY_COLORS[cat],
                edgecolor="white",
                linewidth=1.2,
            )
            for idx, rect in enumerate(bars):
                if idx >= len(band_vals):
                    break
                h = band_vals[idx]
                if h >= LABEL_MIN_PCT and idx < len(cumulative):
                    bottom_before = cumulative[idx]
                    # Determine label color based on placement category
                    if cat == "Mid/Above" or cat == "Early On":
                        label_color = "white"
                    elif cat == "1 Below" or cat == "2 Below":
                        label_color = "#434343"
                    elif cat == "3+ Below":
                        label_color = "white"
                    else:
                        label_color = "#434343"
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
    """Draw average scale score bar chart"""
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
        
        ax.set_ylabel("Avg Scale Score")
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
    """Draw insight text box (shows current values, not deltas)"""
    ax.axis("off")
    if metrics and metrics.get("t_prev"):
        t_prev, t_curr = metrics["t_prev"], metrics["t_curr"]
        # Show current values, not deltas
        high_now = metrics.get("high_now", metrics.get("hi_now", 0))
        lo_now = metrics.get("lo_now", 0)
        score_now = metrics.get("score_now", 0)
        
        insight_lines = [
            f"Current values ({t_curr}):",
            f"Mid/Above: {high_now:.1f} ppts",
            f"2+ Below: {lo_now:.1f} ppts",
            f"Avg Scale Score: {score_now:.1f} pts",
        ]
    else:
        insight_lines = ["Not enough history for insights"]
    
    ax.text(
        0.5,
        0.5,
        "\n".join(insight_lines),
        fontsize=11,
        fontweight="medium",
        color="#333333",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8),
    )

