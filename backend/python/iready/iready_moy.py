# %% Imports and config
# iready.py — charts and analytics
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib import transforms as mtransforms
from matplotlib import lines as mlines
import helper_functions as hf
import skimpy
from skimpy import skim
import tabula as tb
import yaml
import os
import sys

# Libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb

pd.set_option("display.max_rows", None)

# Global threshold for inline % labels on stacked bars
LABEL_MIN_PCT = 5.0
# Toggle for cohort DQC printouts
COHORT_DEBUG = True

# ---------------------------------------------------------------------
# Load partner-specific config using settings.yaml pointer
# ---------------------------------------------------------------------
SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"

# Step 1: read partner name from settings.yaml
with open(SETTINGS_PATH, "r") as f:
    base_cfg = yaml.safe_load(f)

partner_name = base_cfg.get("partner_name")
if not partner_name:
    raise ValueError("settings.yaml must include a 'partner_name' key")

# Step 2: load the partner config file from /config_files/{partner_name}.yaml
CONFIG_PATH = Path(__file__).resolve().parent / "config_files" / f"{partner_name}.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Partner config not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

print(f"Loaded config for: {cfg.get('partner_name', partner_name)}")

# ---------------------------------------------------------------------
# SINGLE PREVIEW / DEV-MODE TOGGLE
# One source of truth: CLI or ENV override; defaults to False
# Usage:
#   python main.py --preview     → enables preview
#   python main.py --full        → disables preview
#   export PREVIEW=true          → enables preview globally
# ---------------------------------------------------------------------

DEV_MODE = True  # FALSE = Batch Run; TRUE = Preview Mode

# 1. Check environment variable
_env_preview = os.getenv("PREVIEW")
if _env_preview is not None:
    DEV_MODE = str(_env_preview).strip().lower() in ("1", "true", "yes", "on")

# 2. Check CLI arguments (override env)
_argv = [a.lower() for a in sys.argv]
if "--preview" in _argv or "--dev" in _argv:
    DEV_MODE = True
elif "--full" in _argv:
    DEV_MODE = False

# 3. Apply globally
import helper_functions as hf

hf.DEV_MODE = DEV_MODE
print(f"[INFO] Preview mode: {DEV_MODE}")

# %% Load Data
# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
DATA_DIR = Path("../data")
csv_path = DATA_DIR / "iready_data.csv"
LABEL_MIN_PCT = 5.0

if not csv_path.exists():
    raise FileNotFoundError(
        f"Expected CSV not found: {csv_path}. Please run data_ingest.py first."
    )

iready_base = pd.read_csv(csv_path)
iready_base.columns = iready_base.columns.str.strip().str.lower()
print(
    f"IREADY data loaded: {iready_base.shape[0]:,} rows, {iready_base.shape[1]} columns"
)
print(iready_base["year"].value_counts().sort_index())
print(iready_base.columns.tolist())

# Normalize district name for fallback in the title
district_label = cfg.get("district_name", ["Districtwide"])[0]

# Inspect categorical columns (quick QC)
cat_cols = [
    c
    for c in iready_base.columns
    if iready_base[c].dtype == "object" or iready_base[c].dtype.name == "category"
]
print("\n--- Unique values per categorical column ---")
for c in cat_cols:
    uniq = iready_base[c].dropna().unique()
    n = len(uniq)
    sample = uniq[:10]
    print(f"\n{c} ({n} unique): {sample}")

# %% SECTION 0 - i-Ready Pred vs Actual
# =========================================================
# SECTION 0 — i-Ready vs CERS (Faceted ELA + Math)
# =========================================================
# Compares i-Ready Spring placement to CERS (CAASPP) Spring performance bands.
# For each subject (ELA, Math), cross-tabulates i-Ready placement vs CERS band.
# Produces a faceted dashboard with:
#   - Stacked bar: % of students in each CERS band by i-Ready placement
#   - Bar panel: % i-Ready Mid/Above vs % CERS Met/Exceed
#   - Insight panel: difference between i-Ready and CERS rates
# =========================================================


def run_section0_iready(
    df_scope, scope_label="Districtwide", folder="_district", preview=False
):
    print("\n>>> STARTING SECTION 0 <<<")

    # =========================================================
    # DATA PREP
    # =========================================================
    def _prep_section0_iready(df, subject):
        """
        Prepare cross-tabulation and metrics for i-Ready vs CERS comparison for a given subject.

        Args:
            df: DataFrame with student assessment data.
            subject: Subject string ("ELA" or "Math").

        Returns:
            cross_dict: dict mapping (placement, cers_band) to percent of students
            metrics: dict with summary metrics (placement rate, CERS rate, delta, year)
            last_year: int, most recent academic year with valid data
        """
        d = df.copy()

        # Normalize i-Ready placement labels for consistency
        if hasattr(hf, "IREADY_LABEL_MAP"):
            d["relative_placement"] = d["relative_placement"].replace(
                hf.IREADY_LABEL_MAP
            )

        subj = subject.upper()

        # --- Filter for most recent academic year with Spring + CERS data ---
        valid_years = (
            d.loc[
                (d["testwindow"].str.upper() == "WINTER")
                & (d["cers_overall_performanceband"].notna())
            ]["academicyear"]
            .dropna()
            .unique()
        )
        if len(valid_years) == 0:
            print(f"[WARN] No Winter rows with valid CERS data for {subj}")
            return None, None, None

        last_year = max(valid_years)
        d = d[
            (d["academicyear"] == last_year)
            & (d["testwindow"].str.upper() == "WINTER")
            & (d["subject"].str.upper() == subj)
            & (d["cers_overall_performanceband"].notna())
            & (d["domain"] == "Overall")
            & (d["relative_placement"].notna())
            & (d["enrolled"] == "Enrolled")
        ].copy()

        if d.empty:
            print(f"[WARN] No Winter {last_year} data for {subj}")
            return None, None, None

        placement_col = "relative_placement"
        cers_col = "cers_overall_performanceband"

        # --- Build cross-tab of (placement x CERS band) ---
        cross = d.groupby([placement_col, cers_col]).size().reset_index(name="n")
        total = cross.groupby(placement_col)["n"].sum().reset_index(name="N_total")
        cross = cross.merge(total, on=placement_col, how="left")
        cross["pct"] = 100 * cross["n"] / cross["N_total"]
        cross_dict = {
            (r[placement_col], r[cers_col]): r["pct"] for _, r in cross.iterrows()
        }

        # --- Compute summary metrics for insight panel ---
        iready_mid_above = (
            d[placement_col]
            .eq(hf.IREADY_LABEL_MAP.get("Mid or Above Grade Level", "Mid/Above"))
            .mean()
            * 100
        )
        cers_met_exceed = (
            d[cers_col]
            .isin(["Level 3 - Standard Met", "Level 4 - Standard Exceeded"])
            .mean()
            * 100
        )

        metrics = {
            "iready_mid_above": iready_mid_above,
            "cers_met_exceed": cers_met_exceed,
            "delta": iready_mid_above - cers_met_exceed,
            "year": int(last_year),
        }

        return cross_dict, metrics, last_year

    # =========================================================
    # PLOTTING
    # =========================================================
    def _plot_section0_iready(scope_label, folder, data_dict, preview=False):
        """
        Plot faceted dashboard comparing i-Ready placement to CERS performance bands.

        Args:
            scope_label: Display label for the scope (district or site)
            folder: Output folder for charts
            data_dict: dict {subject: (cross_dict, metrics)}
            preview: If True, show chart interactively
        """
        # Define consistent order for CERS bands and i-Ready placements
        cers_levels = [
            "Level 1 - Standard Not Met",
            "Level 2 - Standard Nearly Met",
            "Level 3 - Standard Met",
            "Level 4 - Standard Exceeded",
        ]
        placements = [
            hf.IREADY_LABEL_MAP.get("3 or More Grade Levels Below", "3+ Below"),
            hf.IREADY_LABEL_MAP.get("2 Grade Levels Below", "2 Below"),
            hf.IREADY_LABEL_MAP.get("1 Grade Level Below", "1 Below"),
            hf.IREADY_LABEL_MAP.get("Early On Grade Level", "Early On"),
            hf.IREADY_LABEL_MAP.get("Mid or Above Grade Level", "Mid/Above"),
        ]

        # Chart layout: 3 rows x 2 columns (ELA left, Math right)
        fig = plt.figure(figsize=(16, 9), dpi=300)
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.8, 0.65, 0.5])
        fig.subplots_adjust(hspace=0.35, wspace=0.25)

        # Legend for CERS bands (top center)
        handles = [
            Patch(facecolor=hf.CERS_LEVEL_COLORS.get(l, "#ccc"), label=l)
            for l in cers_levels
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=4,
            frameon=False,
            fontsize=9,
        )

        for i, (subj, (cross_pct, metrics)) in enumerate(data_dict.items()):
            print(
                f"[Plot] Subject: {subj} — {len(cross_pct)} cells  Metrics: {metrics}"
            )

            # --- Stacked bar: % of students in each CERS band by i-Ready placement ---
            ax_top = fig.add_subplot(gs[0, i])
            bottom = np.zeros(len(placements))
            for lvl in cers_levels:
                # Get % for each placement x CERS band
                vals = [cross_pct.get((p, lvl), 0) for p in placements]
                color = hf.CERS_LEVEL_COLORS.get(lvl, "#ccc")
                ax_top.bar(
                    placements,
                    vals,
                    bottom=bottom,
                    color=color,
                    edgecolor="white",
                    linewidth=1,
                )
                # Inline % labels for bars above threshold
                for j, v in enumerate(vals):
                    if v >= 5:
                        ax_top.text(
                            j,
                            bottom[j] + v / 2,
                            f"{v:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                        )
                bottom += np.array(vals)
            ax_top.set_ylim(0, 100)
            ax_top.set_ylabel("% of Students")
            ax_top.set_title(subj, fontsize=14, fontweight="bold")
            ax_top.grid(axis="y", alpha=0.2)
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)

            # --- Bar panel: i-Ready Mid/Above vs CERS Met/Exceed ---
            ax_mid = fig.add_subplot(gs[1, i])
            bars = ax_mid.bar(
                ["i-Ready Mid/Above", "CERS Met/Exceed"],
                [metrics["iready_mid_above"], metrics["cers_met_exceed"]],
                color=["#00baeb", "#0381a2"],
                edgecolor="white",
                width=0.6,
            )
            for rect, val in zip(
                bars, [metrics["iready_mid_above"], metrics["cers_met_exceed"]]
            ):
                ax_mid.text(
                    rect.get_x() + rect.get_width() / 2,
                    val + 1,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#434343",
                )
            ax_mid.set_ylim(0, 100)
            ax_mid.set_ylabel("% of Students")
            ax_mid.grid(axis="y", alpha=0.2)
            ax_mid.spines["top"].set_visible(False)
            ax_mid.spines["right"].set_visible(False)

            # --- Insight panel: difference between i-Ready and CERS rates ---
            ax_bot = fig.add_subplot(gs[2, i])
            ax_bot.axis("off")
            insight_text = (
                f"Winter i-Ready Mid/Above vs CERS Met/Exceed:\n"
                rf"${metrics['iready_mid_above']:.1f}\% - {metrics['cers_met_exceed']:.1f}\% = "
                rf"\mathbf{{{metrics['delta']:+.1f}}}$ pts"
            )
            ax_bot.text(
                0.5,
                0.5,
                insight_text,
                ha="center",
                va="center",
                fontsize=12,
                color="#333",
                bbox=dict(
                    boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbb"
                ),
            )

        # --- Chart title and save ---
        year = next(iter(data_dict.values()))[1].get("year", "")
        fig.suptitle(
            f"{scope_label} • Winter {year} • i-Ready Placement vs CERS Performance",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )

        fig.text(
            0.5,
            0.975,
            "The charts below reflect data for students in Gr 3-8 and 11 with matched CAASPP scores. "
            "The 'CERS Met/Exceed' may not align to official results.",
            ha="center",
            fontsize=10,
            style="italic",
        )

        out_dir = Path("../charts") / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scope_label}_section0_iready_vs_cers_.png"
        hf._save_and_render(fig, out_path)
        print(f"[SAVE] Section 0 → {out_path}")

        if preview:
            plt.show()
        plt.close()

    # =========================================================
    # DRIVER LOGIC
    # =========================================================
    data_dict = {}
    for subj in ["ELA", "Math"]:
        result = _prep_section0_iready(df_scope, subj)
        if result and result[0] is not None:
            cross, metrics, year = result
            data_dict[subj] = (cross, metrics)

    if not data_dict:
        print(f"[WARN] No valid Section 0 data for {scope_label}")
        return

    _plot_section0_iready(scope_label, folder, data_dict, preview=preview)


# ---------------------------------------------------------------------
# DRIVER — RUN SECTION 0 (District + Sites)
# ---------------------------------------------------------------------
print("Running Section 0 batch...")

# District-level
scope_df = iready_base.copy()
scope_label = cfg.get("district_name", ["Districtwide"])[0]
folder = "_district"
run_section0_iready(scope_df, scope_label=scope_label, folder=folder, preview=False)

# Site-level
for raw_school in sorted(iready_base["school"].dropna().unique()):
    scope_df = iready_base[iready_base["school"] == raw_school].copy()
    scope_label = hf._safe_normalize_school_name(raw_school, cfg)
    folder = scope_label.replace(" ", "_")
    run_section0_iready(scope_df, scope_label=scope_label, folder=folder, preview=False)


print("Section 0 batch complete.")


# %% SECTION 0.1 — i-Ready Fall → Winter (Placement + Scale Score)
# =========================================================
# MUCH more basic than Section 0.
# Faceted ELA + Math dashboard (3 rows x 2 cols):
#   Row 1: 100% stacked bar of relative_placement for Fall vs Winter
#   Row 2: Avg scale score for Fall vs Winter
#   Row 3: Insight box showing change in avg scale score (Fall → Winter)
# =========================================================


def run_section0_1_iready_fall_winter(
    df_scope, scope_label="Districtwide", folder="_district", preview=False
):
    print("\n>>> STARTING SECTION 0.1 <<<")

    def _prep_section0_1(df, subject):
        d = df.copy()

        # Normalize i-Ready placement labels for consistency
        if hasattr(hf, "IREADY_LABEL_MAP"):
            d["relative_placement"] = d["relative_placement"].replace(
                hf.IREADY_LABEL_MAP
            )

        subj = str(subject).upper()

        # --- Restrict to current (max) academic year only ---
        d["academicyear"] = pd.to_numeric(d.get("academicyear"), errors="coerce")
        year = int(d["academicyear"].max())

        d = d[
            (d["academicyear"] == year)
            & (d["testwindow"].astype(str).str.upper().isin(["FALL", "WINTER"]))
            & (d["subject"].astype(str).str.upper() == subj)
            & (d["domain"].astype(str) == "Overall")
            & (d["enrolled"].astype(str) == "Enrolled")
            & (d["relative_placement"].notna())
        ].copy()

        if d.empty:
            print(f"[WARN] No qualifying rows for Section 0.1 ({subj}, {year})")
            return None, None

        # Ensure scale_score numeric
        d["scale_score"] = pd.to_numeric(d.get("scale_score"), errors="coerce")

        # --- Placement % by window ---
        win_order = ["Fall", "Winter"]
        counts = (
            d.groupby(["testwindow", "relative_placement"])
            .size()
            .rename("n")
            .reset_index()
        )
        totals = d.groupby("testwindow").size().rename("N_total").reset_index()
        pct_df = counts.merge(totals, on="testwindow", how="left")
        pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]

        pct_df["testwindow"] = pct_df["testwindow"].astype(str).str.title()
        pct_df = pct_df[pct_df["testwindow"].isin(win_order)].copy()

        # Ensure all placements exist for stacking
        all_idx = pd.MultiIndex.from_product(
            [win_order, hf.IREADY_ORDER], names=["testwindow", "relative_placement"]
        )
        pct_df = (
            pct_df.set_index(["testwindow", "relative_placement"])
            .reindex(all_idx)
            .reset_index()
        )
        pct_df["pct"] = pct_df["pct"].fillna(0)
        pct_df["n"] = pct_df["n"].fillna(0)
        pct_df["N_total"] = pct_df.groupby("testwindow")["N_total"].transform(
            lambda s: s.ffill().bfill()
        )

        # --- Avg scale score by window ---
        score_df = (
            d.dropna(subset=["scale_score"])
            .groupby(d["testwindow"].astype(str).str.title())["scale_score"]
            .mean()
            .reindex(win_order)
            .rename("avg_score")
            .reset_index()
            .rename(columns={"testwindow": "window"})
        )

        # Compute diff (Winter - Fall)
        fall_val = score_df.loc[score_df["window"] == "Fall", "avg_score"]
        winter_val = score_df.loc[score_df["window"] == "Winter", "avg_score"]
        if len(fall_val) == 0 or len(winter_val) == 0:
            diff = np.nan
        else:
            diff = float(winter_val.iloc[0]) - float(fall_val.iloc[0])

        metrics = {"year": year, "diff": diff}
        return pct_df, score_df, metrics

    def _plot_section0_1(scope_label, folder, data_dict, preview=False):
        # Layout: 3 rows x 2 columns (ELA left, Math right)
        fig = plt.figure(figsize=(16, 9), dpi=300)
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
        fig.subplots_adjust(hspace=0.35, wspace=0.25)

        # Legend once (top center)
        legend_handles = [
            Patch(facecolor=hf.IREADY_COLORS[q], edgecolor="none", label=q)
            for q in hf.IREADY_ORDER
        ]
        fig.legend(
            handles=legend_handles,
            labels=hf.IREADY_ORDER,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=len(hf.IREADY_ORDER),
            frameon=False,
            fontsize=9,
        )

        win_order = ["Fall", "Winter"]

        for i, (subj, (pct_df, score_df, metrics)) in enumerate(data_dict.items()):
            # --- Row 1: 100% stacked bar (relative_placement) for Fall vs Winter ---
            ax_top = fig.add_subplot(gs[0, i])
            pivot = (
                pct_df.pivot(
                    index="testwindow", columns="relative_placement", values="pct"
                )
                .reindex(index=win_order)
                .reindex(columns=hf.IREADY_ORDER)
                .fillna(0)
            )
            x = np.arange(len(win_order))
            bottom = np.zeros(len(win_order))
            for cat in hf.IREADY_ORDER:
                vals = pivot[cat].to_numpy()
                bars = ax_top.bar(
                    x,
                    vals,
                    bottom=bottom,
                    color=hf.IREADY_COLORS[cat],
                    edgecolor="white",
                    linewidth=1.2,
                    width=0.7,
                )
                for j, v in enumerate(vals):
                    if v >= LABEL_MIN_PCT:
                        ax_top.text(
                            x[j],
                            bottom[j] + v / 2,
                            f"{v:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color=(
                                "white"
                                if cat in ["3+ Below", "Mid/Above", "Early On"]
                                else "#333"
                            ),
                        )
                bottom += vals

            ax_top.set_ylim(0, 100)
            ax_top.set_ylabel("% of Students")
            ax_top.set_xticks(x)
            ax_top.set_xticklabels(win_order)
            ax_top.set_title(subj, fontsize=14, fontweight="bold")
            ax_top.grid(axis="y", alpha=0.2)
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)

            # --- Row 2: Avg Scale Score (Fall vs Winter) ---
            ax_mid = fig.add_subplot(gs[1, i])
            xx = np.arange(len(win_order))
            yvals = score_df["avg_score"].to_numpy()
            bars = ax_mid.bar(
                xx,
                yvals,
                color=hf.default_quintile_colors[4],
                edgecolor="white",
                linewidth=1.2,
                width=0.7,
            )
            for rect, val in zip(bars, yvals):
                if pd.notna(val):
                    ax_mid.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height(),
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                        color="#333",
                    )
            # --- add n-counts under window labels ---
            n_map = (
                pct_df.groupby("testwindow")["N_total"]
                .max()
                .reindex(win_order)
                .to_dict()
            )
            labels_with_n = [
                f"{w}\n(n = {0 if pd.isna(n_map.get(w, np.nan)) else int(n_map.get(w))})"
                for w in win_order
            ]
            ax_mid.set_ylabel("Avg Scale Score")
            ax_mid.set_xticks(xx)
            ax_mid.set_xticklabels(labels_with_n)
            ax_mid.grid(axis="y", alpha=0.2)
            ax_mid.spines["top"].set_visible(False)
            ax_mid.spines["right"].set_visible(False)

            # --- Row 3: Insight box (diff Winter - Fall) ---
            ax_bot = fig.add_subplot(gs[2, i])
            ax_bot.axis("off")
            diff = metrics.get("diff", np.nan)
            diff_str = "NA" if pd.isna(diff) else f"{diff:+.1f}"
            insight_text = "Change in Avg Scale Score Fall to Winter:\n" f"{diff_str}"
            ax_bot.text(
                0.5,
                0.5,
                insight_text,
                ha="center",
                va="center",
                fontsize=12,
                color="#333",
                bbox=dict(
                    boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbb"
                ),
            )

        # Title + save
        year = next(iter(data_dict.values()))[2].get("year", "")
        fig.suptitle(
            f"{scope_label} • {year} • i-Ready Fall vs Winter Trends",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )

        out_dir = Path("../charts") / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scope_label}_section0_1_fall_to_winter_.png"
        hf._save_and_render(fig, out_path)
        print(f"[SAVE] Section 0.1 → {out_path}")

        if preview:
            plt.show()
        plt.close(fig)

    # --- Build and plot ---
    data_dict = {}
    for subj in ["ELA", "Math"]:
        prep = _prep_section0_1(df_scope, subj)
        if prep and prep[0] is not None:
            pct_df, score_df, metrics = prep
            data_dict[subj] = (pct_df, score_df, metrics)

    if not data_dict:
        print(f"[WARN] No valid Section 0.1 data for {scope_label}")
        return

    _plot_section0_1(scope_label, folder, data_dict, preview=preview)


# ---------------------------------------------------------------------
# DRIVER — RUN SECTION 0.1 (District + Sites)
# ---------------------------------------------------------------------
print("Running Section 0.1 batch...")

# District-level
scope_df = iready_base.copy()
scope_label = cfg.get("district_name", ["Districtwide"])[0]
folder = "_district"
run_section0_1_iready_fall_winter(
    scope_df, scope_label=scope_label, folder=folder, preview=False
)

# Site-level
for raw_school in sorted(iready_base["school"].dropna().unique()):
    scope_df = iready_base[iready_base["school"] == raw_school].copy()
    scope_label = hf._safe_normalize_school_name(raw_school, cfg)
    folder = scope_label.replace(" ", "_")
    run_section0_1_iready_fall_winter(
        scope_df, scope_label=scope_label, folder=folder, preview=False
    )


print("Section 0.1 batch complete.")

# ---------------------------------------------------------------------
# DRIVER — RUN SECTION 0.1 BY GRADE (District + Sites)
# ---------------------------------------------------------------------
print("Running Section 0.1 grade-level batch...")

# Use current year only (matches Section 0.1 logic)
_base0_1 = iready_base.copy()
_base0_1["academicyear"] = pd.to_numeric(_base0_1.get("academicyear"), errors="coerce")
_curr_year0_1 = int(_base0_1["academicyear"].max())
_base0_1 = _base0_1[_base0_1["academicyear"] == _curr_year0_1].copy()

# Normalize grade to numeric for looping
_base0_1["student_grade"] = pd.to_numeric(
    _base0_1.get("student_grade"), errors="coerce"
)

# Limit to grades used in reporting (Gr 3-8 and 11)
_grade_whitelist0_1 = {3, 4, 5, 6, 7, 8, 11}

# ---- District-level (by grade) ----
scope_label_district = cfg.get("district_name", ["Districtwide"])[0]
for g in sorted(_base0_1["student_grade"].dropna().unique()):
    if int(g) not in _grade_whitelist0_1:
        continue
    df_g = _base0_1[_base0_1["student_grade"] == g].copy()
    run_section0_1_iready_fall_winter(
        df_g,
        scope_label=f"{scope_label_district} • Grade {int(g)}",
        folder="_district",
        preview=False,
    )

# ---- Site-level (by grade) ----
for raw_school in sorted(_base0_1["school"].dropna().unique()):
    school_df = _base0_1[_base0_1["school"] == raw_school].copy()
    scope_label_school = hf._safe_normalize_school_name(raw_school, cfg)
    folder_school = scope_label_school.replace(" ", "_")

    for g in sorted(school_df["student_grade"].dropna().unique()):
        if int(g) not in _grade_whitelist0_1:
            continue
        df_g = school_df[school_df["student_grade"] == g].copy()
        run_section0_1_iready_fall_winter(
            df_g,
            scope_label=f"{scope_label_school} • Grade {int(g)}",
            folder=folder_school,
            preview=False,
        )

print("Section 0.1 grade-level batch complete.")


# %% SECTION 1 - Winter Performance Trends
# Subject Dashboards by Year/Window
# Example labels: "Winter 22-23", "Winter 23-24", "Winter 24-25", "Winter 25-26"
# Rules:
#   - Window: Winter only (default, configurable)
#   - Subject filtering:
#       - ELA: subject contains "ELA" (case-insensitive)
#       - Math: subject contains "Math"
#   - Only valid relative_placement rows retained
#   - Latest test per student per year used (based on completion_date)
#   - Y-axis: percent of students in each quintile (100% stacked bar)
#   - Second panel: average Scale Score per year
#   - Third panel: insight box with deltas (last 2 years)
#   - District chart includes all; school charts filtered to site
# ---------------------------------------------------------------------


def _prep_iready_for_charts(
    df: pd.DataFrame,
    subject_str: str,
    window_filter: str = "Winter",
):
    """Prepare i-Ready data for dashboard plotting."""
    d = df.copy()

    # --- Normalize i-Ready placement labels ---
    if hasattr(hf, "IREADY_LABEL_MAP"):
        d["relative_placement"] = d["relative_placement"].replace(hf.IREADY_LABEL_MAP)

    # --- Verify expected columns ---
    expected_cols = [
        "uniqueidentifier",
        "academicyear",
        "student_grade",
        "subject",
        "testwindow",
        "scale_score",
        "relative_placement",
        "school",
        "domain",
    ]
    missing = [c for c in expected_cols if c not in d.columns]
    if missing:
        print(f"[WARN] Missing expected columns in i-Ready data: {missing}")

    # --- Ensure academic year numeric ---
    d["year"] = pd.to_numeric(d["academicyear"], errors="coerce")

    # --- Year label helpers ---
    def _short_year(y):
        if pd.isna(y):
            return ""
        ys = str(int(y))
        return f"{str(int(ys)-1)[-2:]}-{str(int(ys))[-2:]}"

    d["year_short"] = d["year"].apply(_short_year)
    d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["year_short"]

    # --- Filter to window + subject ---
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()].copy()
    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[
            d["subject"].astype(str).str.contains("math", case=False, na=False)
        ].copy()
    elif "ela" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("ela", case=False, na=False)].copy()
    d = d[d["domain"] == "Overall"].copy()
    d = d[d["testwindow"].notna()].copy()
    d = d[d["enrolled"] == "Enrolled"].copy()
    d = d[d["relative_placement"].notna()].copy()

    # --- Dedupe to latest completion per student/year ---
    if "completion_date" in d.columns:
        d["completion_date"] = pd.to_datetime(d["completion_date"], errors="coerce")
        sort_col = "completion_date"
    elif "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
        sort_col = "teststartdate"
    else:
        d["completion_date"] = pd.NaT
        sort_col = "completion_date"

    d.sort_values(["uniqueidentifier", "time_label", sort_col], inplace=True)
    d = d.groupby(["uniqueidentifier", "time_label"], as_index=False).tail(1)

    # --- Percent by placement ---
    quint_counts = (
        d.groupby(["time_label", "relative_placement"]).size().rename("n").reset_index()
    )
    total_counts = d.groupby("time_label").size().rename("N_total").reset_index()
    pct_df = quint_counts.merge(total_counts, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]

    # --- Ensure all quintiles exist for stacking ---
    all_idx = pd.MultiIndex.from_product(
        [pct_df["time_label"].unique(), hf.IREADY_ORDER],
        names=["time_label", "relative_placement"],
    )
    pct_df = (
        pct_df.set_index(["time_label", "relative_placement"])
        .reindex(all_idx)
        .reset_index()
    )
    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df.groupby("time_label")["N_total"].transform(
        lambda s: s.ffill().bfill()
    )

    # --- Average scale score ---
    score_df = (
        d[["time_label", "scale_score"]]
        .dropna(subset=["scale_score"])
        .groupby("time_label")["scale_score"]
        .mean()
        .rename("avg_score")
        .reset_index()
    )

    # --- Chronological order ---
    time_order = sorted(pct_df["time_label"].unique().tolist())
    pct_df["time_label"] = pd.Categorical(
        pct_df["time_label"], categories=time_order, ordered=True
    )
    score_df["time_label"] = pd.Categorical(
        score_df["time_label"], categories=time_order, ordered=True
    )
    pct_df.sort_values(["time_label", "relative_placement"], inplace=True)
    score_df.sort_values(["time_label"], inplace=True)

    # --- Year map for filtering ---
    year_map = d.drop_duplicates("time_label")[["time_label", "year"]]
    pct_df = pct_df.merge(year_map, on="time_label", how="left")
    score_df = score_df.merge(year_map, on="time_label", how="left")

    # --- Insight metrics ---
    last_two = time_order[-2:] if len(time_order) >= 2 else time_order
    if len(last_two) == 2:
        t_prev, t_curr = last_two

        def pct_for(buckets, tlabel):
            return pct_df[
                (pct_df["time_label"] == tlabel)
                & (pct_df["relative_placement"].isin(buckets))
            ]["pct"].sum()

        hi_curr = pct_for(hf.IREADY_HIGH_GROUP, t_curr)
        hi_prev = pct_for(hf.IREADY_HIGH_GROUP, t_prev)
        lo_curr = pct_for(hf.IREADY_LOW_GROUP, t_curr)
        lo_prev = pct_for(hf.IREADY_LOW_GROUP, t_prev)
        high_curr = pct_for(["Mid/Above"], t_curr)
        high_prev = pct_for(["Mid/Above"], t_prev)

        metrics = {
            "t_prev": t_prev,
            "t_curr": t_curr,
            "hi_now": hi_curr,
            "hi_delta": hi_curr - hi_prev,
            "lo_now": lo_curr,
            "lo_delta": lo_curr - lo_prev,
            "score_now": float(
                score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
            ),
            "score_delta": float(
                score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
            )
            - float(
                score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0]
            ),
            "high_now": high_curr,
            "high_delta": high_curr - high_prev,
        }
    else:
        metrics = {
            k: None
            for k in [
                "t_prev",
                "t_curr",
                "hi_now",
                "hi_delta",
                "lo_now",
                "lo_delta",
                "score_now",
                "score_delta",
                "high_now",
                "high_delta",
            ]
        }

    return pct_df, score_df, metrics, time_order


# ---------------------------------------------------------------------
# Plot dual-subject dashboard (district or school)
# ---------------------------------------------------------------------
def plot_iready_dual_subject_dashboard(
    df,
    window_filter="Winter",
    figsize=(16, 9),
    school_raw=None,
    scope_label=None,
    preview=False,
):
    """Render dual-subject i-Ready dashboard with scale scores."""
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    subjects = ["ELA", "Math"]
    titles = ["ELA", "Math"]

    def draw_stacked_bar(ax, pct_df):
        stack_df = (
            pct_df.pivot(index="time_label", columns="relative_placement", values="pct")
            .reindex(columns=hf.IREADY_ORDER)
            .fillna(0)
        )
        x_labels = stack_df.index.tolist()
        x = np.arange(len(x_labels))
        cumulative = np.zeros(len(stack_df))
        for cat in hf.IREADY_ORDER:
            vals = stack_df[cat].to_numpy()
            bars = ax.bar(
                x,
                vals,
                bottom=cumulative,
                color=hf.IREADY_COLORS[cat],
                edgecolor="white",
                linewidth=1.2,
            )
            for idx, rect in enumerate(bars):
                h = vals[idx]
                if h >= 5:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        cumulative[idx] + h / 2,
                        f"{h:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="#333",
                    )
            cumulative += vals
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of Students")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.tick_params(pad=5)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_score_bar(ax, score_df, n_map=None):
        rit_x = np.arange(len(score_df["time_label"]))
        rit_vals = score_df["avg_score"].to_numpy()
        bars = ax.bar(
            rit_x,
            rit_vals,
            color=hf.default_quintile_colors[4],
            edgecolor="white",
            linewidth=1.2,
        )
        for rect, v in zip(bars, rit_vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333",
            )
        # Build x-axis labels (optionally with n-counts)
        base_labels = score_df["time_label"].astype(str).tolist()
        if n_map is not None:
            labels = [f"{lbl}\n(n = {int(n_map.get(lbl, 0))})" for lbl in base_labels]
        else:
            labels = base_labels
        ax.set_ylabel("Avg Scale Score")
        ax.set_xticks(rit_x)
        ax.set_xticklabels(labels)
        ax.tick_params(pad=10)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_insight_card(ax, metrics):
        ax.axis("off")
        if metrics.get("t_prev"):
            pct_df = metrics.get("pct_df")
            if pct_df is not None:

                def _bucket_delta(bucket, df):
                    curr = df.loc[df["time_label"] == metrics["t_curr"]]
                    prev = df.loc[df["time_label"] == metrics["t_prev"]]
                    return (
                        curr.loc[curr["relative_placement"] == bucket, "pct"].sum()
                        - prev.loc[prev["relative_placement"] == bucket, "pct"].sum()
                    )

                high_delta = _bucket_delta("Mid/Above", pct_df)
                lo_delta = sum(
                    _bucket_delta(b, pct_df) for b in ["3+ Below", "2 Below"]
                )
                score_delta = metrics["score_delta"]

                lines = [
                    "Comparisons based on the current and previous year:\n\n"
                    rf"$\Delta$ Mid/Above: $\mathbf{{{high_delta:+.1f}}}$ ppts",
                    rf"$\Delta$ 2+ Below: $\mathbf{{{lo_delta:+.1f}}}$ ppts",
                    rf"$\Delta$ Avg Scale Score: $\mathbf{{{score_delta:+.1f}}}$ pts",
                ]
            else:
                lines = ["(Missing data for insights)"]
        else:
            lines = ["Not enough history for change insights"]
        ax.text(
            0.5,
            0.5,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=11,
            color="#333",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbb"),
        )

    # --- Draw panels for both subjects ---
    for i, (subject_filter, title) in enumerate(zip(subjects, titles)):
        pct_df, score_df, metrics, _ = _prep_iready_for_charts(
            df,
            subject_str=subject_filter,
            window_filter=window_filter,
        )
        recent_years = sorted(pct_df["year"].unique())[-4:]
        pct_df = pct_df.query("year in @recent_years")
        score_df = score_df.query("year in @recent_years")

        # Build n-count map for middle panel axis labels
        n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
        n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))

        metrics = dict(metrics)
        metrics["pct_df"] = pct_df

        ax1 = fig.add_subplot(gs[0, i])
        draw_stacked_bar(ax1, pct_df)
        ax1.set_title(title, fontsize=14, fontweight="bold", pad=30)

        ax2 = fig.add_subplot(gs[1, i])
        draw_score_bar(ax2, score_df, n_map)
        ax2.set_title("Avg Scale Score", fontsize=8, fontweight="bold", pad=10)

        ax3 = fig.add_subplot(gs[2, i])
        draw_insight_card(ax3, metrics)

    # --- Shared legend ---
    legend_handles = [
        Patch(facecolor=hf.IREADY_COLORS[q], edgecolor="none", label=q)
        for q in hf.IREADY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        labels=hf.IREADY_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(hf.IREADY_ORDER),
        frameon=False,
        fontsize=9,
    )

    # --- Title and save ---
    fig.suptitle(
        f"{scope_label} • {window_filter} Year-to-Year Trends",
        fontsize=20,
        fontweight="bold",
        y=1,
    )

    charts_dir = Path("../charts")
    folder_name = "_district" if school_raw is None else scope_label.replace(" ", "_")
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section1_iready_{window_filter.lower()}_trends.png"
    out_path = out_dir / out_name

    hf._save_and_render(fig, out_path)
    print(f"Saved: {out_path}")

    if preview:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------
# DRIVER — Dual Subject i-Ready Dashboard (District + Site)
# ---------------------------------------------------------------------

# ---- District-level ----
scope_df = iready_base.copy()
scope_label = cfg.get("district_name", ["Districtwide"])[0]

plot_iready_dual_subject_dashboard(
    scope_df,
    window_filter="Winter",
    figsize=(16, 9),
    school_raw=None,
    scope_label=scope_label,
    preview=True,  # or False for batch
)

# ---- Site-level ----
for raw_school in sorted(iready_base["school"].dropna().unique()):
    scope_df = iready_base[iready_base["school"] == raw_school].copy()
    scope_label = hf._safe_normalize_school_name(raw_school, cfg)

    plot_iready_dual_subject_dashboard(
        scope_df,
        window_filter="Winter",
        figsize=(16, 9),
        school_raw=raw_school,  # keep raw for internal filters
        scope_label=scope_label,  # standardized label for chart titles + filenames
        preview=False,
    )

# %% SECTION 2 - Student Group Performance Trends
# ---------------------------------------------------------------------
# SECTION 2 — Student Group Dashboards
# One dashboard per student group per subject.
# District view = all students in that group across the org.
# Site view = all students in that group within that site.
# Only render if group size >= 12 unique students in that scope.
# ---------------------------------------------------------------------


def _apply_student_group_mask(
    df_in: pd.DataFrame, group_name: str, group_def: dict
) -> pd.Series:
    """
    Returns boolean mask for df_in selecting the student group.
    Uses cfg['student_groups'] spec:
      type: "all"                  -> everyone True
      or {column: <col>, in: [...]} -> membership by value match (case-insensitive str compare)
    """
    if group_def.get("type") == "all":
        return pd.Series(True, index=df_in.index)

    col = group_def["column"]
    allowed_vals = group_def["in"]

    # normalize both sides as lowercase strings
    vals = df_in[col].astype(str).str.strip().str.lower()
    allowed_norm = {str(v).strip().lower() for v in allowed_vals}
    return vals.isin(allowed_norm)


def plot_iready_subject_dashboard_by_group(
    df,
    subject_str=None,
    window_filter="Winter",
    group_name=None,
    group_def=None,
    figsize=(16, 9),
    school_raw=None,
    scope_label=None,
    preview=False,
):
    """
    Same visual layout as the main dashboard and the grade dashboard
    but filtered to one student group.
    We also enforce min n >= 12 unique students in the current scope.
    If facet=True, plot both ELA and Math side by side in a single chart.
    """

    # filter to this student group
    d0 = df.copy()

    # normalize school pretty label
    school_display = (
        hf._safe_normalize_school_name(school_raw, cfg) if school_raw else None
    )
    title_label = (
        cfg.get("district_name", ["District (All Students)"])[0]
        if not school_display
        else school_display
    )

    subjects = ["ELA", "Math"]
    subject_titles = ["ELA", "Math"]
    # Apply group mask first
    mask = _apply_student_group_mask(d0, group_name, group_def)
    d0 = d0[mask].copy()
    if d0.empty:
        print(
            f"[group {group_name}] no rows after group mask ({school_raw or 'district'})"
        )
        return
    # Aggregate for each subject
    pct_dfs = []
    score_dfs = []
    metrics_list = []
    time_orders = []
    min_ns = []
    n_maps = []
    for subj in subjects:
        # Filter for subject
        if subj == "ELA":
            subj_df = d0[
                d0["subject"].astype(str).str.contains("ela", case=False, na=False)
            ].copy()
        elif subj == "Math":
            subj_df = d0[
                d0["subject"].astype(str).str.contains("math", case=False, na=False)
            ].copy()
        else:
            subj_df = d0.copy()
        if subj_df.empty:
            pct_dfs.append(None)
            score_dfs.append(None)
            metrics_list.append(None)
            time_orders.append([])
            min_ns.append(0)
            n_maps.append({})
            continue
        pct_df, score_df, metrics, time_order = _prep_iready_for_charts(
            subj_df,
            subject_str=subj,
            window_filter=window_filter,
        )
        # restrict to most recent 4 timepoints
        if len(time_order) > 4:
            time_order = time_order[-4:]
            pct_df = pct_df[pct_df["time_label"].isin(time_order)].copy()
            score_df = score_df[score_df["time_label"].isin(time_order)].copy()
        pct_dfs.append(pct_df)
        score_dfs.append(score_df)
        metrics_list.append(metrics)
        time_orders.append(time_order)
        # --- minimum n >= 12 check ---
        if pct_df is not None and not pct_df.empty and time_order:
            latest_label = time_order[-1]
            latest_slice = pct_df[pct_df["time_label"] == latest_label]
            if "N_total" in latest_slice.columns:
                latest_n = latest_slice["N_total"].max()
            else:
                latest_n = latest_slice["n"].sum()
            min_ns.append(latest_n if not pd.isna(latest_n) else 0)
        else:
            min_ns.append(0)
        # --- n_map for xticklabels in score panel ---
        if pct_df is not None and not pct_df.empty:
            n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
            n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        else:
            n_map = {}
        n_maps.append(n_map)
    # If either panel fails min_n, skip
    if any((n is None or n < 12) for n in min_ns):
        print(
            f"[group {group_name}] skipped (<12 students in one or both subjects) in {title_label}"
        )
        return
    # If either panel has no data, skip
    if any((df is None or df.empty) for df in pct_dfs) or any(
        (df is None or df.empty) for df in score_dfs
    ):
        print(
            f"[group {group_name}] skipped (empty data in one or both subjects) in {title_label}"
        )
        return

    # Setup subplots: 3 rows x 2 columns (ELA left, Math right)
    # Use a gridspec layout for more flexible spacing
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ]
    # Panel 1: Stacked bar for each subject
    legend_handles = [
        Patch(facecolor=hf.IREADY_COLORS[q], edgecolor="none", label=q)
        for q in hf.IREADY_ORDER
    ]
    for i, subj in enumerate(subjects):
        pct_df = pct_dfs[i]
        score_df = score_dfs[i]
        metrics = metrics_list[i]
        time_order = time_orders[i]
        stack_df = (
            pct_df.pivot(index="time_label", columns="relative_placement", values="pct")
            .reindex(columns=hf.IREADY_ORDER)
            .fillna(0)
        )
        x_labels = stack_df.index.tolist()
        x = np.arange(len(x_labels))
        ax1 = axes[0][i]
        cumulative = np.zeros(len(stack_df))
        for cat in hf.IREADY_ORDER:
            band_vals = stack_df[cat].to_numpy()
            bars = ax1.bar(
                x,
                band_vals,
                bottom=cumulative,
                color=hf.IREADY_COLORS[cat],
                edgecolor="white",
                linewidth=1.2,
            )
            for idx, rect in enumerate(bars):
                h = band_vals[idx]
                if h >= LABEL_MIN_PCT:
                    bottom_before = cumulative[idx]
                    if cat == "Mid/Above" or cat == "Early On":
                        label_color = "white"
                    elif cat == "1 Below" or cat == "2 Below":
                        label_color = "#434343"
                    elif cat == "3+ Below":
                        label_color = "white"
                    else:
                        label_color = "#434343"
                    ax1.text(
                        rect.get_x() + rect.get_width() / 2,
                        bottom_before + h / 2,
                        f"{h:.2f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=label_color,
                    )
            cumulative += band_vals
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("% of Students")
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels)
        ax1.grid(axis="y", alpha=0.2)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        # Title
        ax1.set_title(f"{subject_titles[i]}", fontsize=14, fontweight="bold", y=1.1)
    # Place legend once, centered above both facets
    fig.legend(
        handles=legend_handles,
        labels=hf.IREADY_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(hf.IREADY_ORDER),
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    # Panel 2: Avg score by subject
    for i, subj in enumerate(subjects):
        score_df = score_dfs[i]
        ax2 = axes[1][i]
        rit_x = np.arange(len(score_df["time_label"]))
        rit_vals = score_df["avg_score"].to_numpy()
        rit_bars = ax2.bar(
            rit_x,
            rit_vals,
            color=hf.default_quintile_colors[4],
            edgecolor="white",
            linewidth=1.2,
        )
        for rect, v in zip(rit_bars, rit_vals):
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#434343",
            )
        # --- Add n-counts under xticklabels ---
        n_map = n_maps[i] if i < len(n_maps) else {}
        base_labels = score_df["time_label"].astype(str).tolist()
        labels_with_n = [
            f"{lbl}\n(n = {int(n_map.get(lbl, 0))})" for lbl in base_labels
        ]
        ax2.set_ylabel("Avg Scale Score")
        ax2.set_xticks(rit_x)
        ax2.set_xticklabels(labels_with_n)
        ax2.set_title("Average Scale Score", fontsize=8, fontweight="bold", pad=10)
        ax2.grid(axis="y", alpha=0.2)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    # Panel 3: Insights by subject
    for i, subj in enumerate(subjects):
        metrics = metrics_list[i]
        pct_df = pct_dfs[i]
        time_order = time_orders[i]
        ax3 = axes[2][i]
        ax3.axis("off")
        if metrics and metrics.get("t_prev"):
            t_prev = metrics["t_prev"]
            t_curr = metrics["t_curr"]

            def _pct_for_bucket(bucket_name, tlabel):
                return pct_df[
                    (pct_df["time_label"] == tlabel)
                    & (pct_df["relative_placement"] == bucket_name)
                ]["pct"].sum()

            high_now = _pct_for_bucket("Mid/Above", t_curr)
            high_prev = _pct_for_bucket("Mid/Above", t_prev)
            high_delta = high_now - high_prev
            hi_delta = metrics["hi_delta"]
            lo_delta = metrics["lo_delta"]
            score_delta = metrics["score_delta"]
            title_line = "Comparison based on current and previous year:\n"
            line_high = rf"$\Delta$ Mid or Above: $\mathbf{{{high_delta:+.1f}}}$ ppts"
            line_low = rf"$\Delta$ 2 or More Below: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
            line_rit = rf"$\Delta$ Avg Scale Score: $\mathbf{{{score_delta:+.1f}}}$ pts"
            insight_lines = [title_line, line_high, line_low, line_rit]
        else:
            insight_lines = ["Not enough history for change insights"]
        ax3.text(
            0.5,
            0.5,
            "\n".join(insight_lines),
            fontsize=13,
            fontweight="medium",
            color="#333333",
            ha="center",
            va="center",
            wrap=True,
            usetex=False,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#f5f5f5",
                edgecolor="#ccc",
                linewidth=0.8,
            ),
        )
    # Main title for the whole figure
    fig.suptitle(
        f"{title_label} • {group_name} • Winter Year-to-Year Trends",
        fontsize=20,
        fontweight="bold",
        y=1,
    )
    # --- Save ---
    charts_dir = Path("../charts")
    folder_name = "_district" if school_raw is None else scope_label.replace(" ", "_")
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_scope = scope_label.replace(" ", "_")
    safe_group = (
        group_name.lower().replace(" ", "_").replace("/", "_").replace("+", "plus")
    )

    # Pull order value from YAML config (defaults to 99 if not found)
    order_map = cfg.get("student_group_order", {})
    group_order_val = order_map.get(group_name, 99)

    out_name = f"{safe_scope}_section2_{group_order_val:02d}_{safe_group}_Winter_i-Ready_trends.png"
    out_path = out_dir / out_name

    hf._save_and_render(fig, out_path)
    print(f"Saved: {out_path}")

    if preview:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------
# DRIVER — Faceted Student Group Dashboards (District and Site)
# ---------------------------------------------------------------------
student_groups_cfg = cfg.get("student_groups", {})
group_order = cfg.get("student_group_order", {})

# ---- District-level ----
scope_df = iready_base.copy()
scope_label = cfg.get("district_name", ["Districtwide"])[0]

for group_name, group_def in sorted(
    student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99)
):
    if group_def.get("type") == "all":
        continue
    plot_iready_subject_dashboard_by_group(
        scope_df.copy(),
        subject_str=None,
        window_filter="Winter",
        group_name=group_name,
        group_def=group_def,
        figsize=(16, 9),
        school_raw=None,
        scope_label=scope_label,
    )

# ---- Site-level ----
for raw_school in sorted(iready_base["schoolname"].dropna().unique()):
    scope_df = iready_base[iready_base["schoolname"] == raw_school].copy()
    scope_label = hf._safe_normalize_school_name(raw_school, cfg)

    for group_name, group_def in sorted(
        student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99)
    ):
        if group_def.get("type") == "all":
            continue
        plot_iready_subject_dashboard_by_group(
            scope_df.copy(),
            subject_str=None,
            window_filter="Winter",
            group_name=group_name,
            group_def=group_def,
            figsize=(16, 9),
            school_raw=raw_school,
            scope_label=scope_label,
        )


# %% SECTION 3 — Overall + Cohort Trends (i-Ready)
# ---------------------------------------------------------------------
def _prep_iready_matched_cohort_by_grade(
    df, subject_str, current_grade, window_filter, cohort_year
):
    base = df.copy()
    base["academicyear"] = pd.to_numeric(base.get("academicyear"), errors="coerce")
    base["student_grade"] = pd.to_numeric(base.get("student_grade"), errors="coerce")

    anchor_year = int(cohort_year or base["academicyear"].max())
    cohort_rows, ordered_labels = [], []

    for offset in range(0, 4):
        yr = anchor_year - 3 + offset
        gr = current_grade - 3 + offset
        if gr < 0:
            continue

        tmp = base[
            (base["testwindow"].astype(str).str.upper() == window_filter.upper())
            & (base["student_grade"] == gr)
            & (base["academicyear"] == yr)
        ].copy()

        subj_norm = subject_str.strip().lower()
        if "math" in subj_norm:
            tmp = tmp[
                tmp["subject"].astype(str).str.contains("math", case=False, na=False)
            ]
        elif "ela" in subj_norm:
            tmp = tmp[
                tmp["subject"].astype(str).str.contains("ela", case=False, na=False)
            ]

        tmp = tmp[tmp["domain"] == "Overall"]
        tmp = tmp[tmp["relative_placement"].notna()]
        tmp = tmp[tmp["enrolled"] == "Enrolled"]

        if hasattr(hf, "IREADY_LABEL_MAP"):
            tmp["relative_placement"] = tmp["relative_placement"].replace(
                hf.IREADY_LABEL_MAP
            )

        if tmp.empty:
            continue

        tmp.sort_values(["uniqueidentifier", "completion_date"], inplace=True)
        tmp = tmp.groupby("uniqueidentifier", as_index=False).tail(1)

        y_prev, y_curr = str(yr - 1)[-2:], str(yr)[-2:]
        label = f"Gr {int(gr)} • Winter {y_prev}-{y_curr}"
        tmp["cohort_label"] = label
        cohort_rows.append(tmp)
        ordered_labels.append(label)

    if not cohort_rows:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    cohort_df = pd.concat(cohort_rows, ignore_index=True)
    cohort_df["label"] = cohort_df["cohort_label"]

    counts = (
        cohort_df.groupby(["label", "relative_placement"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals = cohort_df.groupby("label").size().rename("N_total").reset_index()
    pct_df = counts.merge(totals, on="label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]

    all_idx = pd.MultiIndex.from_product(
        [pct_df["label"].unique(), hf.IREADY_ORDER],
        names=["label", "relative_placement"],
    )
    pct_df = (
        pct_df.set_index(["label", "relative_placement"]).reindex(all_idx).reset_index()
    )
    pct_df[["pct", "n"]] = pct_df[["pct", "n"]].fillna(0)
    pct_df["N_total"] = pct_df.groupby("label")["N_total"].transform(
        lambda s: s.ffill().bfill()
    )

    score_df = (
        cohort_df[["label", "scale_score"]]
        .dropna(subset=["scale_score"])
        .groupby("label")["scale_score"]
        .mean()
        .rename("avg_score")
        .reset_index()
    )

    pct_df["label"] = pd.Categorical(
        pct_df["label"], categories=ordered_labels, ordered=True
    )
    score_df["label"] = pd.Categorical(
        score_df["label"], categories=ordered_labels, ordered=True
    )
    pct_df = (
        pct_df.rename(columns={"label": "time_label"})
        .sort_values("time_label")
        .reset_index(drop=True)
    )
    score_df = (
        score_df.rename(columns={"label": "time_label"})
        .sort_values("time_label")
        .reset_index(drop=True)
    )

    labels_order = ordered_labels
    last_two = labels_order[-2:] if len(labels_order) >= 2 else labels_order
    if len(last_two) == 2:
        t_prev, t_curr = last_two

        def pct_for(buckets, tlabel):
            return pct_df[
                (pct_df["time_label"] == tlabel)
                & (pct_df["relative_placement"].isin(buckets))
            ]["pct"].sum()

        hi_now = pct_for(["Mid or Above Grade Level", "Early On Grade Level"], t_curr)
        lo_now = pct_for(
            ["2 Grade Levels Below", "3 or More Grade Levels Below"], t_curr
        )
        hi_delta = hi_now - pct_for(
            ["Mid or Above Grade Level", "Early On Grade Level"], t_prev
        )
        lo_delta = lo_now - pct_for(
            ["2 Grade Levels Below", "3 or More Grade Levels Below"], t_prev
        )
        score_now = float(
            score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
        )
        score_prev = float(
            score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0]
        )

        metrics = dict(
            t_prev=t_prev,
            t_curr=t_curr,
            hi_now=hi_now,
            hi_delta=hi_delta,
            lo_now=lo_now,
            lo_delta=lo_delta,
            score_now=score_now,
            score_delta=score_now - score_prev,
        )
    else:
        metrics = {
            k: None
            for k in [
                "t_prev",
                "t_curr",
                "hi_now",
                "hi_delta",
                "lo_now",
                "lo_delta",
                "score_now",
                "score_delta",
            ]
        }

    return pct_df, score_df, metrics, ordered_labels


# ---------------------------------------------------------------------
def plot_iready_blended_dashboard(
    df,
    subject_str,
    current_grade,
    window_filter="Winter",
    cohort_year=None,
    figsize=(16, 9),
    scope_label=None,
    preview=False,
):
    """Dual-facet dashboard showing Overall vs Cohort Trends with three panels each."""
    scope_label = scope_label or cfg.get("district_name", ["Districtwide"])[0]
    folder_name = (
        "_district"
        if scope_label == cfg.get("district_name", ["Districtwide"])[0]
        else scope_label.replace(" ", "_")
    )

    # Prep left and right
    d = df.copy()
    d["student_grade"] = pd.to_numeric(d["student_grade"], errors="coerce")
    d = d[d["student_grade"] == current_grade]
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()]
    subj_norm = subject_str.strip().lower()
    if "math" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("math", case=False, na=False)]
    elif "ela" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("ela", case=False, na=False)]
    d = d[d["domain"] == "Overall"]
    d = d[d["relative_placement"].notna()]
    d = d[d["enrolled"] == "Enrolled"]

    pct_df_left, score_df_left, metrics_left, _ = _prep_iready_for_charts(
        d, subject_str=subject_str, window_filter=window_filter
    )
    pct_df_right, score_df_right, metrics_right, _ = (
        _prep_iready_matched_cohort_by_grade(
            df, subject_str, current_grade, window_filter, cohort_year
        )
    )

    # --- Normalize cohort time labels to match expected pattern ---
    if "time_label" in pct_df_right.columns:
        pct_df_right["time_label_clean"] = (
            pct_df_right["time_label"]
            .astype(str)
            .str.replace(r"Gr\s*\d+\s*•\s*", "", regex=True)
            .str.strip()
        )
    else:
        pct_df_right["time_label_clean"] = pct_df_right.get("label", "")

    # --- Compute n-maps for left and right panels for n-count x-axis labels ---
    n_map_left_df = pct_df_left.groupby("time_label")["N_total"].max().reset_index()
    n_map_left = dict(
        zip(n_map_left_df["time_label"].astype(str), n_map_left_df["N_total"])
    )
    if "time_label" in pct_df_right.columns and not pct_df_right.empty:
        n_map_right_df = (
            pct_df_right.groupby("time_label")["N_total"].max().reset_index()
        )
        n_map_right = dict(
            zip(n_map_right_df["time_label"].astype(str), n_map_right_df["N_total"])
        )
    else:
        n_map_right = {}

    if (
        pct_df_left.empty
        or score_df_left.empty
        or pct_df_right.empty
        or score_df_right.empty
    ):
        print(f"[Skip] No data for Grade {current_grade} {subject_str} ({scope_label})")
        return

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    def draw_stacked(ax, pdf):
        pivot = (
            pdf.pivot(index="time_label", columns="relative_placement", values="pct")
            .reindex(columns=hf.IREADY_ORDER)
            .fillna(0)
        )
        x = np.arange(len(pivot))
        bottom = np.zeros(len(pivot))
        for cat in hf.IREADY_ORDER:
            vals = pivot[cat].to_numpy()
            bars = ax.bar(
                x,
                vals,
                bottom=bottom,
                color=hf.IREADY_COLORS[cat],
                edgecolor="white",
                linewidth=1.2,
            )
            for i, v in enumerate(vals):
                if v >= 5:
                    ax.text(
                        bars[i].get_x() + bars[i].get_width() / 2,
                        bottom[i] + v / 2,
                        f"{v:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=(
                            "white"
                            if cat in ["3+ Below", "Mid/Above", "Early On"]
                            else "#434343"
                        ),
                    )
            bottom += vals
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of Students")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.tolist(), fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Refactored draw_score to accept n_map and include n-counts in xticklabels
    def draw_score(ax, sdf, n_map=None):
        x = np.arange(len(sdf))
        bars = ax.bar(
            x, sdf["avg_score"], color=hf.default_quintile_colors[4], edgecolor="white"
        )
        for r, v in zip(bars, sdf["avg_score"]):
            ax.text(
                r.get_x() + r.get_width() / 2,
                v,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333",
            )
        base_labels = sdf["time_label"].astype(str).tolist()
        if n_map is not None:
            labels = [f"{lbl}\n(n = {int(n_map.get(lbl, 0))})" for lbl in base_labels]
        else:
            labels = base_labels
        ax.set_ylabel("Avg Scale Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_insight(ax, metrics, pct_df=None, use_clean_time_label=False):
        ax.axis("off")
        if metrics.get("t_prev"):
            if use_clean_time_label and pct_df is not None:
                from difflib import get_close_matches

                t_prev = metrics["t_prev"]
                t_curr = metrics["t_curr"]

                # Use time_label_clean for matching; robust to label differences
                def _bucket_delta(bucket, df):
                    # Find closest matches for t_prev/t_curr in df['time_label_clean']
                    labels = df["time_label_clean"].astype(str).unique().tolist()
                    t_prev_match = get_close_matches(
                        str(t_prev), labels, n=1, cutoff=0.6
                    )
                    t_curr_match = get_close_matches(
                        str(t_curr), labels, n=1, cutoff=0.6
                    )
                    t_prev_use = t_prev_match[0] if t_prev_match else t_prev
                    t_curr_use = t_curr_match[0] if t_curr_match else t_curr

                    if "state_benchmark_achievement" in df.columns:
                        curr = df.loc[
                            (df["time_label_clean"] == t_curr_use)
                            & (df["state_benchmark_achievement"] == bucket),
                            "pct",
                        ].sum()
                        prev = df.loc[
                            (df["time_label_clean"] == t_prev_use)
                            & (df["state_benchmark_achievement"] == bucket),
                            "pct",
                        ].sum()
                    else:
                        curr = df.loc[
                            (df["time_label_clean"] == t_curr_use)
                            & (df["relative_placement"] == bucket),
                            "pct",
                        ].sum()
                        prev = df.loc[
                            (df["time_label_clean"] == t_prev_use)
                            & (df["relative_placement"] == bucket),
                            "pct",
                        ].sum()
                    return curr - prev

                high_delta = _bucket_delta("Mid/Above", pct_df)
                lo_delta = _bucket_delta("3+ Below", pct_df) + _bucket_delta(
                    "2 Below", pct_df
                )
                score_delta = metrics.get("score_delta", 0)
                lines = [
                    "Comparisons based on current vs previous year:\n",
                    rf"$\Delta$ Mid/Above: $\mathbf{{{high_delta:+.1f}}}$ ppts",
                    rf"$\Delta$ 2+ Below: $\mathbf{{{lo_delta:+.1f}}}$ ppts",
                    rf"$\Delta$ Avg Scale Score: $\mathbf{{{score_delta:+.1f}}}$ pts",
                ]
            else:
                lines = [
                    "Comparisons based on current vs previous year:\n",
                    rf"$\Delta$ Mid/Above: $\mathbf{{{metrics['hi_delta']:+.1f}}}$ ppts",
                    rf"$\Delta$ 2+ Below: $\mathbf{{{metrics['lo_delta']:+.1f}}}$ ppts",
                    rf"$\Delta$ Avg Scale Score: $\mathbf{{{metrics['score_delta']:+.1f}}}$ pts",
                ]
        else:
            lines = ["Not enough history for change insights"]
        ax.text(
            0.5,
            0.5,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=11,
            color="#333",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbb"),
        )

    # Panels for each facet
    ax1 = fig.add_subplot(gs[0, 0])
    draw_stacked(ax1, pct_df_left)
    ax1.set_title("Overall Trends", fontsize=14, fontweight="bold")
    ax2 = fig.add_subplot(gs[0, 1])
    draw_stacked(ax2, pct_df_right)
    ax2.set_title("Cohort Trends", fontsize=14, fontweight="bold")
    ax3 = fig.add_subplot(gs[1, 0])
    draw_score(ax3, score_df_left, n_map_left)
    ax4 = fig.add_subplot(gs[1, 1])
    draw_score(ax4, score_df_right, n_map_right)
    ax5 = fig.add_subplot(gs[2, 0])
    draw_insight(ax5, metrics_left)
    ax6 = fig.add_subplot(gs[2, 1])
    # For cohort facet, use normalized time_label_clean in insight delta calculation
    draw_insight(ax6, metrics_right, pct_df=pct_df_right, use_clean_time_label=True)

    legend_handles = [
        Patch(
            facecolor=hf.IREADY_COLORS[k],
            edgecolor="none",
            label=hf.IREADY_LABEL_MAP.get(k, k),
        )
        for k in hf.IREADY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(hf.IREADY_ORDER),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        f"{scope_label} • Grade {int(current_grade)} • {subject_str} • {window_filter}",
        fontsize=20,
        fontweight="bold",
        y=1,
    )

    out_dir = Path("../charts") / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / f"{scope_label.replace(' ','_')}_section3_iready_grade{int(current_grade)}_{subject_str}_{window_filter.lower()}_trends.png"
    )
    hf._save_and_render(fig, out_path)
    if preview:
        plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# DRIVERS
# ---------------------------------------------------------------------
_base = iready_base.copy()
_base["academicyear"] = pd.to_numeric(_base["academicyear"], errors="coerce")
_base["student_grade"] = pd.to_numeric(_base["student_grade"], errors="coerce")
_anchor_year = int(_base["academicyear"].max())

scope_label = cfg.get("district_name", ["Districtwide"])[0]
for g in sorted(_base["student_grade"].dropna().unique()):
    for subj in ["ELA", "Math"]:
        plot_iready_blended_dashboard(
            _base.copy(),
            subj,
            int(g),
            "Winter",
            _anchor_year,
            scope_label=scope_label,
            preview=False,
        )

for raw_school in sorted(_base["school"].dropna().unique()):
    site_df = _base[_base["school"] == raw_school].copy()
    site_df["academicyear"] = pd.to_numeric(site_df["academicyear"], errors="coerce")
    site_df["student_grade"] = pd.to_numeric(site_df["student_grade"], errors="coerce")
    anchor = int(site_df["academicyear"].max())
    scope_label = hf._safe_normalize_school_name(raw_school, cfg)
    for g in sorted(site_df["student_grade"].dropna().unique()):
        for subj in ["ELA", "Math"]:
            plot_iready_blended_dashboard(
                site_df.copy(),
                subj,
                int(g),
                "Winter",
                anchor,
                scope_label=scope_label,
                preview=False,
            )

# %% SECTION 4 — Spring i-Ready Mid/Above → % CERS Met/Exceeded (≤2025)
# ---------------------------------------------------------------------
# Matches Section 2 exactly in layout and mathtext behavior
# ---------------------------------------------------------------------

_ME_LABELS = {"Level 3 - Standard Met", "Level 4 - Standard Exceeded"}
_SUBJECT_COLORS = {"ELA": "#0381a2", "Math": "#0381a2"}


def _prep_mid_above_to_cers(df_in: pd.DataFrame, subject: str) -> pd.DataFrame:
    d = df_in.copy()
    placement_col = (
        "relative_placement" if "relative_placement" in d.columns else "placement"
    )

    d = d[
        (d["domain"].astype(str).str.lower() == "overall")
        & (d["testwindow"].astype(str).str.lower() == "winter")
        & (d["cers_overall_performanceband"].notna())
        & (d[placement_col].notna())
    ].copy()

    d = d[d["subject"].astype(str).str.lower().str.contains(subject.lower())]
    d["academicyear"] = pd.to_numeric(d["academicyear"], errors="coerce")
    d = d[d["academicyear"] <= 2025]

    mid_vals = {"mid/above", "mid or above", "mid or above grade level"}
    d = d[d[placement_col].astype(str).str.strip().str.lower().isin(mid_vals)].copy()
    if d.empty:
        return pd.DataFrame()

    id_col = "student_id" if "student_id" in d.columns else "uniqueidentifier"
    denom = d.groupby("academicyear")[id_col].nunique().rename("n")
    numer = (
        d[d["cers_overall_performanceband"].isin(_ME_LABELS)]
        .groupby("academicyear")[id_col]
        .nunique()
        .rename("me")
    )
    trend = denom.to_frame().join(numer, how="left").fillna(0).reset_index()
    trend["pct_me"] = (trend["me"] / trend["n"]) * 100
    return trend.sort_values("academicyear")


def _plot_mid_above_to_cers_faceted(scope_df, scope_label, folder_name, preview=False):
    subjects = ["ELA", "Math"]
    trends = {s: _prep_mid_above_to_cers(scope_df, s) for s in subjects}

    if all(tr.empty for tr in trends.values()):
        print(f"[Section 4] No qualifying data for {scope_label}")
        return

    fig, axs = plt.subplots(2, 2, figsize=(16, 9), height_ratios=[2, 0.6])
    fig.subplots_adjust(hspace=0.45, wspace=0.25, top=0.88, bottom=0.1)

    for j, subj in enumerate(subjects):
        ax_bar = axs[0, j]
        ax_box = axs[1, j]

        tr = trends[subj]
        if tr.empty:
            ax_bar.axis("off")
            ax_box.axis("off")
            continue

        # --- top bar chart ---
        x = np.arange(len(tr))
        color = _SUBJECT_COLORS[subj]
        bars = ax_bar.bar(x, tr["pct_me"], color=color, edgecolor="white", width=0.55)

        for rect, yv, n in zip(bars, tr["pct_me"], tr["n"]):
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() / 2,
                f"{yv:.0f}%\n(n={int(n)})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

        ax_bar.set_ylim(0, 100)
        ax_bar.set_xlim(-0.5, len(tr) - 0.5)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(tr["academicyear"].astype(int))
        ax_bar.set_yticks(range(0, 101, 20))
        ax_bar.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
        ax_bar.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.set_ylabel("% Met or Exceeded")
        ax_bar.set_xlabel("Academic Year")
        ax_bar.set_title(subj, fontsize=14, fontweight="bold", pad=20)
        ax_bar.margins(x=0.15)

        # --- bottom insights box (true Section 2 style using LaTeX) ---
        ax_box.axis("off")
        overall_pct = 100 * tr["me"].sum() / tr["n"].sum()

        lines = [
            rf"Historically, $\mathbf{{{overall_pct:.1f}\%}}$ of students that meet ",
            r"$\mathbf{Mid\ or\ Above}$ Grade Level in Winter i-Ready tend to ",
            r"$\mathbf{Meet\ or\ Exceed\ Standard}$ on CAASPP for " + subj + ".",
        ]

        ax_box.text(
            0.5,
            0.5,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=13,
            color="#333",
            wrap=True,
            usetex=False,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#f5f5f5",
                edgecolor="#bbb",
                linewidth=0.8,
            ),
        )

    fig.suptitle(
        f"{scope_label} \n Winter i-Ready Mid/Above → % CERS Met/Exceeded",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )

    charts_dir = Path("../charts")
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    out_path = out_dir / f"{safe_scope}_section4_mid_plus_to_3plus.png"

    hf._save_and_render(fig, out_path)
    print(f"[Section 4] Saved: {out_path}")

    if preview:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# DRIVER — Section 4 (District + Sites)
# ---------------------------------------------------------------------
print("Running Section 4 — Winter Mid/Above → CERS Met/Exceeded")

scope_df = iready_base.copy()
district_label = cfg.get("district_name", ["Districtwide"])[0]
_plot_mid_above_to_cers_faceted(scope_df, district_label, folder_name="_district")

school_col = "school" if "school" in iready_base.columns else "schoolname"
for raw_school in sorted(iready_base[school_col].dropna().unique()):
    site_df = iready_base[iready_base[school_col] == raw_school].copy()
    scope_label = hf._safe_normalize_school_name(raw_school, cfg)
    folder = scope_label.replace(" ", "_")
    _plot_mid_above_to_cers_faceted(site_df, scope_label, folder_name=folder)


# %% SECTION 5 — MOY Growth Progress (Median % Progress + On-Track + BOY-Anchored Insights)
# ---------------------------------------------------------------------
# Purpose:
#   - Winter-only snapshot of growth progress toward annual goals.
#   - Top panel: Median % progress to Typical vs Stretch with 50% reference line.
#   - Middle panel: % of students at/above 50% progress to Typical vs Stretch.
#   - Bottom panel: BOY-anchored pathway update:
#       * Fall baseline cohort defines Mid+ pathway buckets (Already Mid+, Mid with Typical, Mid with Stretch, Beyond Stretch)
#       * Winter shows progress within those fixed Fall-defined cohorts.
# Notes:
#   - Uses current (max) academicyear.
#   - Anchors cohorts in Fall baseline_diagnostic == "Yes" (and Overall domain, Enrolled).
#   - Measures Winter progress using annual_typical_growth_percent / annual_stretch_growth_percent (0–100 integers).
# ---------------------------------------------------------------------


def run_section5_growth_progress_moy(
    df_scope, scope_label="Districtwide", folder="_district", preview=False
):
    print("\n>>> STARTING SECTION 5 <<<")

    # ----------------------------
    # Helpers
    # ----------------------------
    def _safe_id_col(df):
        # Prefer student_id for cross-window matching (uniqueidentifier can be unstable/missing across windows)
        return "student_id" if "student_id" in df.columns else "uniqueidentifier"

    def _dedupe_latest(df, id_col, sort_col_candidates):
        d = df.copy()
        sort_col = None
        for c in sort_col_candidates:
            if c in d.columns:
                sort_col = c
                break
        if sort_col is None:
            # fallback: stable order
            d = d.sort_values([id_col])
            return d.groupby([id_col], as_index=False).tail(1)

        d[sort_col] = pd.to_datetime(d[sort_col], errors="coerce")
        d = d.sort_values([id_col, sort_col])
        return d.groupby([id_col], as_index=False).tail(1)

    def _normalize_placement(d):
        if hasattr(hf, "IREADY_LABEL_MAP") and "relative_placement" in d.columns:
            d["relative_placement"] = d["relative_placement"].replace(
                hf.IREADY_LABEL_MAP
            )
        return d

    def _prep_section5_subject(df, subject):
        d0 = df.copy()

        # Current year only
        d0["academicyear"] = pd.to_numeric(d0.get("academicyear"), errors="coerce")
        year = int(d0["academicyear"].max())
        d0 = d0[d0["academicyear"] == year].copy()

        # Grade filter: default K–8 only (9–12 typically do not have growth targets)
        if "student_grade" in d0.columns:
            d0["student_grade"] = pd.to_numeric(
                d0.get("student_grade"), errors="coerce"
            )
            d0 = d0[d0["student_grade"] <= 8].copy()

        # Common filters
        subj = str(subject).strip().upper()

        base_mask = (d0["enrolled"].astype(str) == "Enrolled") & (
            d0["domain"].astype(str).str.lower() == "overall"
        )

        # i-Ready subjects are often labeled "Reading" and "Math"
        if subj == "ELA":
            subj_mask = (
                d0["subject"]
                .astype(str)
                .str.contains("ela|reading", case=False, na=False)
            )
        elif subj == "MATH":
            subj_mask = (
                d0["subject"].astype(str).str.contains("math", case=False, na=False)
            )
        else:
            subj_mask = (
                d0["subject"].astype(str).str.contains(subj, case=False, na=False)
            )

        d0 = d0[base_mask & subj_mask].copy()

        if d0.empty:
            return None

        id_col = _safe_id_col(d0)
        # Ensure ID types match across Fall/Winter (avoid int/float/object mismatches)
        d0[id_col] = d0[id_col].astype(str).str.strip()
        d0.loc[d0[id_col].isin(["nan", "None", "<NA>"]), id_col] = np.nan

        # ----------------------------
        # Fall baseline cohort (BOY-anchored)
        # ----------------------------
        fall = d0[
            (d0["testwindow"].astype(str).str.lower() == "fall")
            & (d0.get("baseline_diagnostic", "").astype(str).str.lower() == "yes")
        ].copy()

        if fall.empty:
            print(f"[WARN] Section 5: No Fall baseline rows for {subj} ({year})")
            return None

        fall = _normalize_placement(fall)
        fall = _dedupe_latest(fall, id_col, ["completion_date", "teststartdate"])

        # numeric coercion for pathway math
        ss = pd.to_numeric(fall.get("scale_score"), errors="coerce")
        typ = pd.to_numeric(fall.get("annual_typical_growth_measure"), errors="coerce")
        strg = pd.to_numeric(fall.get("annual_stretch_growth_measure"), errors="coerce")
        mid = pd.to_numeric(fall.get("mid_on_grade_level_scale_score"), errors="coerce")

        # After label normalization, Mid/Above is typically "Mid/Above"
        already_mid = (
            fall["relative_placement"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "mid/above",
                    "mid or above grade level",
                    "mid or above",
                    "mid or above grade level",
                    "mid/above grade level",
                ]
            )
        )
        typ_reach = (ss + typ) >= mid
        str_reach = (ss + strg) >= mid

        out = np.full(len(fall), np.nan, dtype=object)
        out[already_mid.to_numpy()] = "Already Mid+"
        m2 = (~already_mid) & typ_reach
        out[m2.to_numpy()] = "Mid with Typical"
        m3 = (~already_mid) & (~typ_reach) & str_reach
        out[m3.to_numpy()] = "Mid with Stretch"
        fall["mid_flag"] = pd.Series(out, index=fall.index).fillna("Mid Beyond Stretch")

        # Base counts (Fall)
        fall_counts = (
            fall["mid_flag"]
            .value_counts(dropna=False)
            .reindex(
                [
                    "Already Mid+",
                    "Mid with Typical",
                    "Mid with Stretch",
                    "Mid Beyond Stretch",
                ]
            )
            .fillna(0)
            .astype(int)
            .to_dict()
        )

        # ----------------------------
        # Winter rows for same Fall cohort
        # ----------------------------
        winter = d0[(d0["testwindow"].astype(str).str.lower() == "winter")].copy()
        winter = _normalize_placement(winter)
        winter = _dedupe_latest(winter, id_col, ["completion_date", "teststartdate"])

        # restrict to Fall cohort ids
        cohort_ids = set(fall[id_col].dropna().unique().tolist())

        # DQC: confirm IDs intersect across windows before filtering
        winter_ids = set(winter[id_col].dropna().unique().tolist())
        inter = cohort_ids.intersection(winter_ids)
        print(
            f"[DQC][S5] {subj} {year} id_col={id_col} | "
            f"fall_ids={len(cohort_ids):,} winter_ids={len(winter_ids):,} intersect={len(inter):,}"
        )
        if len(inter) == 0:
            print(f"[DQC][S5] fall id sample: {list(cohort_ids)[:5]}")
            print(f"[DQC][S5] winter id sample: {list(winter_ids)[:5]}")

        # apply filter
        winter = winter[winter[id_col].isin(cohort_ids)].copy()

        if winter.empty:
            print(f"[WARN] Section 5: No Winter rows for Fall cohort ({subj}, {year})")
            return None

        # attach fall mid_flag to winter for cohort-wise tracking
        winter = winter.merge(
            fall[[id_col, "mid_flag"]].drop_duplicates(subset=[id_col]),
            on=id_col,
            how="left",
        )

        # progress percents (0–100 ints) — use actual i-Ready columns
        winter["pct_typ"] = pd.to_numeric(
            winter.get("percent_progress_to_annual_typical_growth_"), errors="coerce"
        )
        winter["pct_str"] = pd.to_numeric(
            winter.get("percent_progress_to_annual_stretch_growth_"), errors="coerce"
        )

        # ----------------------------
        # Metrics for chart panels
        # ----------------------------
        med_typ = float(np.nanmedian(winter["pct_typ"].to_numpy()))
        med_str = float(np.nanmedian(winter["pct_str"].to_numpy()))
        pct50_typ = float((winter["pct_typ"] >= 50).mean() * 100)
        pct50_str = float((winter["pct_str"] >= 50).mean() * 100)

        # Mid/Above in Winter (for progress update)
        winter_mid = (
            winter["relative_placement"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "mid/above",
                    "mid or above grade level",
                    "mid or above",
                    "mid or above grade level",
                    "mid/above grade level",
                ]
            )
        )
        winter_mid_count = int(winter_mid.sum())
        winter_already_mid_count = winter_mid_count

        # Newly Mid+ by Winter (not Already Mid+ in Fall)
        fall_mid_ids = set(
            fall.loc[fall["mid_flag"] == "Already Mid+", id_col].tolist()
        )
        newly_mid = int(
            winter.loc[~winter[id_col].isin(fall_mid_ids), :][
                winter_mid.loc[~winter[id_col].isin(fall_mid_ids)].values
            ].shape[0]
        )

        # Cohort-wise on-track within Fall-defined groups
        def _on_track(group_name, col):
            base_ids = set(
                fall.loc[fall["mid_flag"] == group_name, id_col].dropna().tolist()
            )
            denom = len(base_ids)
            if denom == 0:
                return {"denom": 0, "num": 0, "pct": np.nan}
            w = winter[winter[id_col].isin(base_ids)].copy()
            num = int((w[col] >= 50).sum())
            pct = 100 * num / denom
            return {"denom": denom, "num": num, "pct": pct}

        on_typ = _on_track("Mid with Typical", "pct_typ")
        on_str = _on_track("Mid with Stretch", "pct_str")
        on_beyond = _on_track("Mid Beyond Stretch", "pct_str")

        metrics = dict(
            year=year,
            med_typ=med_typ,
            med_str=med_str,
            pct50_typ=pct50_typ,
            pct50_str=pct50_str,
            fall_counts=fall_counts,
            winter_mid_count=winter_mid_count,
            winter_already_mid_count=winter_already_mid_count,
            newly_mid=newly_mid,
            on_typ=on_typ,
            on_str=on_str,
            on_beyond=on_beyond,
            n_winter=int(winter[id_col].nunique()),
        )

        return metrics

    def _plot_section5(scope_label, folder, metrics_by_subject, preview=False):
        fig = plt.figure(figsize=(16, 9), dpi=300)
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.6])
        fig.subplots_adjust(hspace=0.35, wspace=0.25)

        # Colors (match existing i-Ready blues)
        c_typ = hf.IREADY_COLORS.get("Mid/Above", "#0381a2")  # darker
        c_str = hf.IREADY_COLORS.get("Early On", "#00baeb")  # lighter

        for i, subj in enumerate(["ELA", "Math"]):
            m = metrics_by_subject.get(subj)
            if not m:
                ax0 = fig.add_subplot(gs[0, i])
                ax0.axis("off")
                ax1 = fig.add_subplot(gs[1, i])
                ax1.axis("off")
                ax2 = fig.add_subplot(gs[2, i])
                ax2.axis("off")
                continue

            # ----------------------------
            # Top: Median % progress (Typical vs Stretch) with 50% line
            # ----------------------------
            ax_top = fig.add_subplot(gs[0, i])
            x = np.arange(2)
            vals = [m["med_typ"], m["med_str"]]
            bars = ax_top.bar(
                x,
                vals,
                color=[c_typ, c_str],
                edgecolor="white",
                linewidth=1.2,
                width=0.6,
            )
            for rect, v in zip(bars, vals):
                if pd.notna(v):
                    ax_top.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 1,
                        f"{v:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                        color="#333",
                    )
            ax_top.axhline(50, linestyle="--", linewidth=1.2, color="#666", alpha=0.8)
            ax_top.set_ylim(0, 100)
            ax_top.set_yticks(range(0, 101, 20))
            ax_top.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
            ax_top.set_xticks(x)
            ax_top.set_xticklabels(["Median % Typical", "Median % Stretch"])
            ax_top.set_ylabel("% Progress")
            ax_top.set_title(subj, fontsize=14, fontweight="bold")
            ax_top.grid(axis="y", alpha=0.2)
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)

            # ----------------------------
            # Middle: % of students >= 50% progress (Typical vs Stretch)
            # ----------------------------
            ax_mid = fig.add_subplot(gs[1, i])
            x2 = np.arange(2)
            vals2 = [m["pct50_typ"], m["pct50_str"]]
            bars2 = ax_mid.bar(
                x2,
                vals2,
                color=[c_typ, c_str],
                edgecolor="white",
                linewidth=1.2,
                width=0.6,
            )
            for rect, v in zip(bars2, vals2):
                if pd.notna(v):
                    ax_mid.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 1,
                        f"{v:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        color="#333",
                    )
            ax_mid.axhline(50, linestyle="--", linewidth=1.0, color="#666", alpha=0.8)
            ax_mid.set_ylim(0, 100)
            ax_mid.set_yticks(range(0, 101, 20))
            ax_mid.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
            ax_mid.set_xticks(x2)
            ax_mid.set_xticklabels([">=50% Typical", ">=50% Stretch"])
            ax_mid.set_ylabel("% of Students")
            ax_mid.grid(axis="y", alpha=0.2)
            ax_mid.spines["top"].set_visible(False)
            ax_mid.spines["right"].set_visible(False)

            # ----------------------------
            # Bottom: BOY-anchored insight summary
            # ----------------------------
            ax_bot = fig.add_subplot(gs[2, i])
            ax_bot.axis("off")

            fc = m["fall_counts"]
            on_typ = m["on_typ"]
            on_str = m["on_str"]
            # on_beyond is still computed, but not displayed
            # Percent strings (NOT bolded)
            typ_pct_str = (
                "NA"
                if (on_typ["denom"] == 0 or pd.isna(on_typ["pct"]))
                else f"{on_typ['pct']:.0f}%"
            )
            str_pct_str = (
                "NA"
                if (on_str["denom"] == 0 or pd.isna(on_str["pct"]))
                else f"{on_str['pct']:.0f}%"
            )

            fall_mid_ct = fc.get("Already Mid+", 0)
            winter_mid_ct = int(
                m.get("winter_already_mid_count", m.get("winter_mid_count", 0))
            )

            lines = [
                rf"Fall Already Mid+ = $\mathbf{{{fall_mid_ct:,}}}$",
                rf"Winter Already Mid+ = $\mathbf{{{winter_mid_ct:,}}}$",
                "",
                "Fall to Winter Mid On Level Update:",
                rf"Mid With Typical: $\mathbf{{{on_typ['num']:,}}}$ out of $\mathbf{{{on_typ['denom']:,}}}$ "
                rf"({typ_pct_str}) are at least 50% to typical growth",
                rf"Mid With Stretch: $\mathbf{{{on_str['num']:,}}}$ out of $\mathbf{{{on_str['denom']:,}}}$ "
                rf"({str_pct_str}) are at least 50% to stretch growth",
            ]

            ax_bot.text(
                0.5,
                0.5,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=11,
                color="#333",
                bbox=dict(
                    boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbb"
                ),
            )

        # Title + save
        year = next(iter(metrics_by_subject.values())).get("year", "")
        fig.suptitle(
            f"{scope_label} • Winter {year} • Growth Progress Toward Annual Goals",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )

        out_dir = Path("../charts") / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scope_label}_section5_growth_progress_moy_.png"
        hf._save_and_render(fig, out_path)
        print(f"[SAVE] Section 5 → {out_path}")

        if preview:
            plt.show()
        plt.close(fig)

    # ----------------------------
    # Build + plot
    # ----------------------------
    metrics_by_subject = {}
    for subj in ["ELA", "Math"]:
        m = _prep_section5_subject(df_scope, subj)
        if m is not None:
            metrics_by_subject[subj] = m

    if not metrics_by_subject:
        print(f"[WARN] No valid Section 5 data for {scope_label}")
        return

    _plot_section5(scope_label, folder, metrics_by_subject, preview=preview)


# ---------------------------------------------------------------------
# DRIVER — RUN SECTION 5 (District + Sites)
# ---------------------------------------------------------------------
print("Running Section 5 batch...")

# District-level
scope_df = iready_base.copy()
scope_label = cfg.get("district_name", ["Districtwide"])[0]
folder = "_district"
run_section5_growth_progress_moy(
    scope_df, scope_label=scope_label, folder=folder, preview=False
)

# Site-level
for raw_school in sorted(iready_base["school"].dropna().unique()):
    scope_df = iready_base[iready_base["school"] == raw_school].copy()
    scope_label = hf._safe_normalize_school_name(raw_school, cfg)
    folder = scope_label.replace(" ", "_")
    run_section5_growth_progress_moy(
        scope_df, scope_label=scope_label, folder=folder, preview=False
    )

print("Section 5 batch complete.")


# %%
