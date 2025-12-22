# %% Imports and config
# iready.py — charts and analytics
#
# NOTE: This is a legacy script. It is executed via `iready_moy_runner.py` in the app.
# The runner provides temp settings/config/data locations via env vars so this file can run
# without writing into the repo.
#
# IMPORTANT: Set non-interactive backend before importing pyplot.
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib import transforms as mtransforms
from matplotlib import lines as mlines
import helper_functions_iready as hf

# ---------------------------------------------------------------------
# Hardcoded, Slides-friendly style for this legacy script
# ---------------------------------------------------------------------
# We set a guaranteed-available font + high-contrast colors + larger default sizes.
# NOTE: explicit `fontsize=` calls in the code still override defaults; we bump the worst
# offenders below as well.
_IREADY_EOY_HARD_RC = {
    "font.family": "DejaVu Sans",
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "axes.titlecolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
}
mpl.rcParams.update(_IREADY_EOY_HARD_RC)
try:
    import skimpy  # type: ignore
    from skimpy import skim  # type: ignore
except Exception:  # pragma: no cover
    skimpy = None
    skim = None

try:
    import tabula as tb  # type: ignore
except Exception:  # pragma: no cover
    tb = None
import yaml
import os
import sys
import json
import re

# Libraries for modeling
try:
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score  # type: ignore
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # type: ignore
except Exception:  # pragma: no cover
    train_test_split = None
    r2_score = None
    mean_absolute_error = None
    accuracy_score = None
    f1_score = None
    RandomForestRegressor = None
    RandomForestClassifier = None

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None

pd.set_option("display.max_rows", None)

# Global threshold for inline % labels on stacked bars
LABEL_MIN_PCT = 5.0
# Toggle for cohort DQC printouts
COHORT_DEBUG = True

# ---------------------------------------------------------------------
# Load partner-specific config using settings.yaml pointer
# ---------------------------------------------------------------------
def _env_first(*keys: str) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    return None


# Prefer EOY env vars, but support MOY legacy names for backward compatibility.
SETTINGS_PATH = Path(
    _env_first("IREADY_EOY_SETTINGS_PATH")
    or (Path(__file__).resolve().parent / "settings.yaml")
)

# Step 1: read partner name from settings.yaml
with open(SETTINGS_PATH, "r") as f:
    base_cfg = yaml.safe_load(f)

partner_name = base_cfg.get("partner_name")
if not partner_name:
    raise ValueError("settings.yaml must include a 'partner_name' key")

# Step 2: load the partner config file from /config_files/{partner_name}.yaml
CONFIG_PATH = Path(
    _env_first("IREADY_EOY_CONFIG_PATH")
    or (Path(__file__).resolve().parent / "config_files" / f"{partner_name}.yaml")
)
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
hf.DEV_MODE = DEV_MODE
print(f"[INFO] Preview mode: {DEV_MODE}")

# %% Load Data
# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
DATA_DIR = Path(_env_first("IREADY_EOY_DATA_DIR") or "../data")
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
# Some partners omit district_name or provide an empty list; treat that as "District".
# Allow a display-only override that does NOT affect filtering / school matching.
_district_display_cfg = cfg.get("district_display_name")
if isinstance(_district_display_cfg, list) and len(_district_display_cfg) > 0 and _district_display_cfg[0]:
    district_label = str(_district_display_cfg[0])
elif isinstance(_district_display_cfg, str) and _district_display_cfg.strip():
    district_label = _district_display_cfg.strip()
else:
    _district_name_cfg = cfg.get("district_name")
    if isinstance(_district_name_cfg, list) and len(_district_name_cfg) > 0 and _district_name_cfg[0]:
        district_label = str(_district_name_cfg[0])
    else:
        district_label = "District"

# Base charts directory (overrideable by runner)
CHARTS_DIR = Path(_env_first("IREADY_EOY_CHARTS_DIR") or "../charts")

# ---------------------------------------------------------------------
# Scope selection (district-only vs district + schools vs selected schools)
#
# Env vars (set by runner / backend):
# - IREADY_MOY_SCOPE_MODE:
#     - "district_only" (skip all school loops)
#     - "selected_schools" (only loop selected schools; still include district)
#     - default/other: district + all schools
# - IREADY_MOY_SCHOOLS="School A,School B" (names can be raw or normalized)
# ---------------------------------------------------------------------
_scope_mode = str(_env_first("IREADY_EOY_SCOPE_MODE") or "").strip().lower()
_env_schools = _env_first("IREADY_EOY_SCHOOLS")
_selected_schools = []
if _env_schools:
    _selected_schools = [s.strip() for s in str(_env_schools).split(",") if s.strip()]


def _get_eoy_compare_windows() -> list[str]:
    """
    Controls within-year comparisons (e.g., Section 0.1 and Sections 6–8).

    Env:
      - IREADY_EOY_COMPARE_WINDOWS="Winter,Spring" (default)
      - IREADY_EOY_COMPARE_WINDOWS="Fall,Winter,Spring"
    Also supports legacy MOY key for compatibility.
    """
    raw = _env_first("IREADY_EOY_COMPARE_WINDOWS")
    if raw:
        parts = [p.strip().title() for p in str(raw).split(",") if p.strip()]
        parts = [p for p in parts if p in {"Fall", "Winter", "Spring"}]
        if parts:
            # De-dupe while preserving order
            seen: set[str] = set()
            out: list[str] = []
            for p in parts:
                if p not in seen:
                    out.append(p)
                    seen.add(p)
            return out
    return ["Winter", "Spring"]


def _include_school_charts() -> bool:
    return _scope_mode not in ("district_only", "district")


def _school_selected(raw_school: str) -> bool:
    """Match against raw school or normalized school display (case-insensitive)."""
    if not _selected_schools:
        return True
    raw = str(raw_school or "").strip()
    disp = ""
    try:
        disp = hf._safe_normalize_school_name(raw_school, cfg)
    except Exception:
        disp = raw
    raw_l = raw.lower()
    disp_l = str(disp or "").strip().lower()
    wanted = {s.lower() for s in _selected_schools}
    return raw_l in wanted or disp_l in wanted


def _iter_schools(df: pd.DataFrame):
    # Prefer `learning_center` if present (charter-style exports), otherwise fall back.
    school_col = None
    for col_name in ["learning_center", "school", "schoolname", "school_name"]:
        if col_name in df.columns:
            try:
                if df[col_name].dropna().astype(str).str.strip().ne("").any():
                    school_col = col_name
                    break
            except Exception:
                school_col = col_name
                break
    if not school_col:
        return
    for rs in sorted(df[school_col].dropna().unique()):
        if not _school_selected(rs):
            continue
        yield school_col, rs


def _write_chart_data(out_path: Path, chart_data: dict) -> None:
    """
    Write sidecar JSON data for chart analysis.

    `chart_analyzer.py` expects a file named `{chart_stem}_data.json` next to the PNG.
    """
    # Important: we serialize to a string first so we never leave partially-written JSON
    # files on disk (partial files cause chart_analyzer.py JSON parse failures).
    try:
        p = Path(out_path)
        data_path = p.parent / f"{p.stem}_data.json"
        tmp_path = p.parent / f"{p.stem}_data.json.tmp"

        payload = chart_data or {}

        # Auto-fill section_title (used by Slides divider generation)
        try:
            if "section_title" not in payload:
                sec = payload.get("section")
                if sec is not None:
                    try:
                        sec_f = float(sec)
                        sec_key = str(int(sec_f)) if sec_f.is_integer() else str(sec_f)
                    except Exception:
                        sec_key = str(sec)

                    IREADY_TITLES = {
                        "0": "i-Ready vs CERS (Predicted vs Actual)",
                        "0.1": "Fall → Winter Comparison",
                        "1": "Performance Trends",
                        "2": "Student Group Performance Trends",
                        "3": "Overall + Cohort Trends",
                        "4": "Winter i-Ready Mid/Above → % CERS Met/Exceeded",
                        "5": "Growth Progress Toward Annual Goals",
                        "6": "Window Compare by School",
                        "7": "Window Compare by Grade",
                        "8": "Window Compare by Student Group",
                        "9": "Median Progress by School",
                        "10": "Median Progress by Grade",
                        "11": "Median Progress by Student Group",
                    }
                    payload["section_title"] = IREADY_TITLES.get(sec_key, f"Section {sec_key}")
        except Exception:
            pass
        text = json.dumps(payload, indent=2, default=str)

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        # Atomic replace so readers never see partial JSON
        os.replace(tmp_path, data_path)
    except Exception:  # pragma: no cover
        try:
            if "tmp_path" in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        # Never fail chart creation due to data sidecar issues.
        return


def _jsonable(v):
    """Convert common pandas/numpy objects into JSON-friendly Python types."""
    try:
        import pandas as _pd  # type: ignore
    except Exception:  # pragma: no cover
        _pd = None
    try:
        import numpy as _np  # type: ignore
    except Exception:  # pragma: no cover
        _np = None

    if _pd is not None:
        if isinstance(v, _pd.DataFrame):
            return v.to_dict("records")
        if isinstance(v, _pd.Series):
            return v.to_dict()

    if _np is not None:
        if isinstance(v, (_np.integer, _np.floating)):
            return v.item()
        if isinstance(v, _np.ndarray):
            return v.tolist()

    return v

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
                (d["testwindow"].str.upper() == "SPRING")
                & (d["cers_overall_performanceband"].notna())
            ]["academicyear"]
            .dropna()
            .unique()
        )
        if len(valid_years) == 0:
            print(f"[WARN] No Spring rows with valid CERS data for {subj}")
            return None, None, None

        last_year = max(valid_years)
        d = d[
            (d["academicyear"] == last_year)
            & (d["testwindow"].str.upper() == "SPRING")
            & (d["subject"].str.upper() == subj)
            & (d["cers_overall_performanceband"].notna())
            & (d["domain"] == "Overall")
            & (d["relative_placement"].notna())
            & (d["enrolled"] == "Enrolled")
        ].copy()

        if d.empty:
            print(f"[WARN] No Spring {last_year} data for {subj}")
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

        # Hardcode font for this section (rcParams elsewhere in this script can be overridden).
        # DejaVu Sans is bundled with matplotlib and is reliably available on servers.
        section0_font = "DejaVu Sans"
        with mpl.rc_context(
            {
                "font.family": section0_font,
                "text.color": "#111111",
                "axes.labelcolor": "#111111",
                "axes.titlecolor": "#111111",
                "xtick.color": "#111111",
                "ytick.color": "#111111",
                "font.size": 20,
                "axes.titlesize": 20,
                "axes.labelsize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 14,
                "savefig.dpi": 300,
            }
        ):
            # Chart layout: 3 rows x 2 columns (ELA left, Math right)
            fig = plt.figure(figsize=(16, 9), dpi=300)
            gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.8, 0.65, 0.55])
            fig.subplots_adjust(hspace=0.40, wspace=0.25, top=0.86, bottom=0.08)

            # Legend for CERS bands (top center)
            handles = [
                Patch(facecolor=hf.CERS_LEVEL_COLORS.get(l, "#ccc"), label=l)
                for l in cers_levels
            ]
            leg = fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=4,
                frameon=False,
                fontsize=14,
            )
            for t in leg.get_texts():
                t.set_color("#111111")

            for i, (subj, (cross_pct, metrics)) in enumerate(data_dict.items()):
                print(
                    f"[Plot] Subject: {subj} — {len(cross_pct)} cells  Metrics: {metrics}"
                )

                # --- Stacked bar: % of students in each CERS band by i-Ready placement ---
                ax_top = fig.add_subplot(gs[0, i])
                bottom = np.zeros(len(placements))
                for lvl in cers_levels:
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
                    for j, v in enumerate(vals):
                        if v >= 5:
                            ax_top.text(
                                j,
                                bottom[j] + v / 2,
                                f"{v:.1f}%",
                                ha="center",
                                va="center",
                                fontsize=15,
                                fontweight="bold",
                                color="#111111",
                            )
                    bottom += np.array(vals)

                ax_top.set_ylim(0, 100)
                ax_top.set_ylabel("% of Students", color="#111111")
                ax_top.tick_params(axis="x", colors="#111111")
                ax_top.tick_params(axis="y", colors="#111111")
                ax_top.set_title(subj, fontsize=20, fontweight="bold", color="#111111")
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
                        fontsize=18,
                        fontweight="bold",
                        color="#111111",
                    )
                ax_mid.set_ylim(0, 100)
                ax_mid.set_ylabel("% of Students", color="#111111")
                ax_mid.tick_params(axis="x", colors="#111111")
                ax_mid.tick_params(axis="y", colors="#111111")
                ax_mid.grid(axis="y", alpha=0.2)
                ax_mid.spines["top"].set_visible(False)
                ax_mid.spines["right"].set_visible(False)

                # --- Insight panel: difference between i-Ready and CERS rates ---
                ax_bot = fig.add_subplot(gs[2, i])
                ax_bot.axis("off")
                insight_text = (
                    f"Spring i-Ready Mid/Above vs CERS Met/Exceed:\n"
                    rf"${metrics['iready_mid_above']:.1f}\% - {metrics['cers_met_exceed']:.1f}\% = "
                    rf"\mathbf{{{metrics['delta']:+.1f}}}$ pts"
                )
                ax_bot.text(
                    0.5,
                    0.5,
                    insight_text,
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="#111111",
                    bbox=dict(
                        boxstyle="round,pad=0.6",
                        facecolor="#eeeeee",
                        edgecolor="#666666",
                        linewidth=1.2,
                    ),
                )

            # --- Chart title and save ---
            year = next(iter(data_dict.values()))[1].get("year", "")
            fig.suptitle(
                f"{scope_label} • Spring {year} • i-Ready Placement vs CERS Performance",
                fontsize=26,
                fontweight="bold",
                y=0.99,
                color="#111111",
            )

            # Optional disclaimer/footer (can be blank/omitted)
            disclaimer = (
                (cfg or {}).get("iready_section0_disclaimer")
                or (cfg or {}).get("section0_disclaimer")
                or ""
            )
            if isinstance(disclaimer, str) and disclaimer.strip():
                fig.text(
                    0.5,
                    0.965,
                    disclaimer.strip(),
                    ha="center",
                    fontsize=12,
                    style="italic",
                    color="#111111",
                )

            out_dir = CHARTS_DIR / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{scope_label}_section0_iready_vs_cers_eoy.png"
            hf._save_and_render(fig, out_path)
            _write_chart_data(
                out_path,
                {
                    "chart_type": "iready_eoy_section0_iready_vs_cers",
                    "section": 0,
                    "scope": scope_label,
                    "folder": folder,
                    "subjects": list(data_dict.keys()),
                    "window_filter": "Spring",
                    "metrics": {k: _jsonable(v[1]) for k, v in data_dict.items()},
                    "cross_data": {k: _jsonable(v[0]) for k, v in data_dict.items()},
                },
            )
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
scope_label = district_label
folder = "_district"
run_section0_iready(scope_df, scope_label=scope_label, folder=folder, preview=False)

# Site-level
if _include_school_charts():
    for school_col, raw_school in _iter_schools(iready_base):
        scope_df = iready_base[iready_base[school_col] == raw_school].copy()
        scope_label = hf._safe_normalize_school_name(raw_school, cfg)
        folder = scope_label.replace(" ", "_")
        run_section0_iready(
            scope_df, scope_label=scope_label, folder=folder, preview=False
        )


print("Section 0 batch complete.")


# %% SECTION 0.1 — i-Ready Fall → Winter (Placement + Scale Score)
# =========================================================
# MUCH more basic than Section 0.
# Faceted ELA + Math dashboard (3 rows x 2 cols):
#   Row 1: 100% stacked bar of relative_placement for Fall vs Winter
#   Row 2: Avg scale score for Fall vs Winter
#   Row 3: Insight box showing change in avg scale score (Fall → Winter)
# =========================================================


def run_section0_1_iready_fall_winter_spring(
    df_scope, scope_label="Districtwide", folder="_district", preview=False
):
    print("\n>>> STARTING SECTION 0.1 <<<")
    win_order = _get_eoy_compare_windows()

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
            & (d["testwindow"].astype(str).str.title().isin(win_order))
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

        # Compute diff (end - start) for the selected compare windows
        start_w = win_order[0]
        end_w = win_order[-1]
        start_val = score_df.loc[score_df["window"] == start_w, "avg_score"]
        end_val = score_df.loc[score_df["window"] == end_w, "avg_score"]
        if len(start_val) == 0 or len(end_val) == 0:
            diff = np.nan
        else:
            diff = float(end_val.iloc[0]) - float(start_val.iloc[0])

        metrics = {"year": year, "diff": diff, "start_window": start_w, "end_window": end_w}
        return pct_df, score_df, metrics

    def _plot_section0_1(scope_label, folder, data_dict, preview=False):
        # Layout: 3 rows x 2 columns (ELA left, Math right)
        fig = plt.figure(figsize=(16, 9), dpi=300)
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.70, 0.65])
        # More top/bottom room so title + legend + insight text are readable in Slides
        fig.subplots_adjust(hspace=0.40, wspace=0.25, top=0.86, bottom=0.08)

        # Legend once (top center)
        legend_handles = [
            Patch(facecolor=hf.IREADY_COLORS[q], edgecolor="none", label=q)
            for q in hf.IREADY_ORDER
        ]
        leg = fig.legend(
            handles=legend_handles,
            labels=hf.IREADY_ORDER,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(hf.IREADY_ORDER),
            frameon=False,
            fontsize=11,
        )
        # Force darker legend text (some environments render defaults too light)
        for t in leg.get_texts():
            t.set_color("#111111")

        win_order = _get_eoy_compare_windows()

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
                            fontsize=10,
                            fontweight="bold",
                            color=(
                                "white"
                                if cat in ["3+ Below", "Mid/Above", "Early On"]
                                else "#333"
                            ),
                        )
                bottom += vals

            ax_top.set_ylim(0, 100)
            ax_top.set_ylabel("% of Students", color="#111111")
            ax_top.set_xticks(x)
            ax_top.set_xticklabels(win_order, color="#111111")
            ax_top.tick_params(axis="y", colors="#111111")
            ax_top.set_title(subj, fontsize=16, fontweight="bold", color="#111111")
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
                        fontsize=16,
                        fontweight="bold",
                        color="#111111",
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
            ax_mid.set_ylabel("Avg Scale Score", color="#111111")
            ax_mid.set_xticks(xx)
            ax_mid.set_xticklabels(labels_with_n, color="#111111")
            ax_mid.tick_params(axis="y", colors="#111111")
            ax_mid.grid(axis="y", alpha=0.2)
            ax_mid.spines["top"].set_visible(False)
            ax_mid.spines["right"].set_visible(False)

            # --- Row 3: Insight box (diff Winter - Fall) ---
            ax_bot = fig.add_subplot(gs[2, i])
            ax_bot.axis("off")
            diff = metrics.get("diff", np.nan)
            sw = metrics.get("start_window", win_order[0])
            ew = metrics.get("end_window", win_order[-1])
            diff_str = "NA" if pd.isna(diff) else f"{diff:+.1f}"
            insight_text = f"Change in Avg Scale Score {sw} to {ew}:\n{diff_str}"
            ax_bot.text(
                0.5,
                0.5,
                insight_text,
                ha="center",
                va="center",
                fontsize=14,
                color="#111111",
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="#eeeeee",
                    edgecolor="#666666",
                    linewidth=1.2,
                ),
            )

        # Title + save
        year = next(iter(data_dict.values()))[2].get("year", "")
        fig.suptitle(
            f"{scope_label} • {year} • i-Ready {win_order[0]} vs {win_order[-1]} Trends",
            fontsize=22,
            fontweight="bold",
            y=0.99,
            color="#111111",
        )

        out_dir = CHARTS_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scope_label}_section0_1_window_compare_eoy.png"
        hf._save_and_render(fig, out_path)
        _write_chart_data(
            out_path,
            {
                "chart_type": "iready_eoy_section0_1_fall_to_spring_snapshot",
                "section": "0.1",
                "scope": scope_label,
                "folder": folder,
                "subjects": list(data_dict.keys()),
                "window_filter": "/".join(win_order),
                "pct_data": [
                    {"subject": subj, "data": _jsonable(data_dict[subj][0])}
                    for subj in data_dict.keys()
                ],
                "score_data": [
                    {"subject": subj, "data": _jsonable(data_dict[subj][1])}
                    for subj in data_dict.keys()
                ],
                "metrics": [_jsonable(data_dict[subj][2]) for subj in data_dict.keys()],
            },
        )
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


# Backward-compatible alias (older driver code calls this name)
def run_section0_1_iready_fall_winter(
    df_scope, scope_label="Districtwide", folder="_district", preview=False
):
    return run_section0_1_iready_fall_winter_spring(
        df_scope, scope_label=scope_label, folder=folder, preview=preview
    )


# ---------------------------------------------------------------------
# DRIVER — RUN SECTION 0.1 (District + Sites)
# ---------------------------------------------------------------------
print("Running Section 0.1 batch...")

# District-level
scope_df = iready_base.copy()
scope_label = district_label
folder = "_district"
run_section0_1_iready_fall_winter(
    scope_df, scope_label=scope_label, folder=folder, preview=False
)

# Site-level
if _include_school_charts():
    for school_col, raw_school in _iter_schools(iready_base):
        scope_df = iready_base[iready_base[school_col] == raw_school].copy()
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
_grade_whitelist0_1 = {8, 11}
# If grades were selected in the frontend, the runner passes them via env var.
# Example: IREADY_MOY_GRADES="3,4,5"
_env_grades = _env_first("IREADY_EOY_GRADES" )
if _env_grades:
    try:
        _parsed = {int(x.strip()) for x in str(_env_grades).split(",") if x.strip()}
        if _parsed:
            _grade_whitelist0_1 = _parsed
    except Exception:
        pass

# ---- District-level (by grade) ----
scope_label_district = district_label
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
if _include_school_charts():
    for school_col, raw_school in _iter_schools(_base0_1):
        school_df = _base0_1[_base0_1[school_col] == raw_school].copy()
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
    window_filter: str = "Spring",
):
    """Prepare i-Ready data for dashboard plotting."""
    d = df.copy()

    # --- Normalize i-Ready placement labels ---
    if "relative_placement" in d.columns and hasattr(hf, "IREADY_LABEL_MAP"):
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

    # Hard-stop if we are missing critical columns that would otherwise crash.
    required_cols = ["uniqueidentifier", "academicyear", "subject", "testwindow", "domain", "enrolled", "relative_placement"]
    missing_required = [c for c in required_cols if c not in d.columns]
    if missing_required:
        print(f"[SKIP] _prep_iready_for_charts: missing required columns: {missing_required}")
        empty_metrics = {k: None for k in ["t_prev","t_curr","hi_now","hi_delta","lo_now","lo_delta","score_now","score_delta","high_now","high_delta"]}
        return pd.DataFrame(), pd.DataFrame(), empty_metrics, []

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

    # Build sidecar chart data payload as we go
    _pct_payload = []
    _score_payload = []
    _metrics_payload = []

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
                        fontsize=12,
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
                fontsize=14,
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

        # Collect analyzer-friendly data (avoid truthiness checks on Series/DataFrames)
        try:
            _pct_payload.append({"subject": title, "data": _jsonable(pct_df)})
            _score_payload.append({"subject": title, "data": _jsonable(score_df)})
            _metrics_payload.append(_jsonable(metrics))
        except Exception:
            pass

        ax1 = fig.add_subplot(gs[0, i])
        draw_stacked_bar(ax1, pct_df)
        ax1.set_title(title, fontsize=14, fontweight="bold", pad=30)

        ax2 = fig.add_subplot(gs[1, i])
        draw_score_bar(ax2, score_df, n_map)
        ax2.set_title("Avg Scale Score", fontsize=14, fontweight="bold", pad=10)

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
        fontsize=13,
    )

    # --- Title and save ---
    fig.suptitle(
        f"{scope_label} • {window_filter} Year-to-Year Trends",
        fontsize=20,
        fontweight="bold",
        y=1,
    )

    charts_dir = CHARTS_DIR
    folder_name = "_district" if school_raw is None else scope_label.replace(" ", "_")
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section1_iready_{window_filter.lower()}_trends_eoy.png"
    out_path = out_dir / out_name

    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "iready_eoy_section1_dual_subject_dashboard",
            "section": 1,
            "scope": scope_label,
            "window_filter": window_filter,
            "subjects": titles,
            "pct_data": _pct_payload,
            "score_data": _score_payload,
            "metrics": _metrics_payload,
        },
    )
    print(f"Saved: {out_path}")

    if preview:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------
# DRIVER — Dual Subject i-Ready Dashboard (District + Site)
# ---------------------------------------------------------------------

# ---- District-level ----
scope_df = iready_base.copy()
scope_label = district_label

plot_iready_dual_subject_dashboard(
    scope_df,
    window_filter="Winter",
    figsize=(16, 9),
    school_raw=None,
    scope_label=scope_label,
    preview=True,  # or False for batch
)

# ---- Site-level ----
if _include_school_charts():
    for school_col, raw_school in _iter_schools(iready_base):
        scope_df = iready_base[iready_base[school_col] == raw_school].copy()
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
    title_label = ((f"{district_label} (All Students)") if not school_display else school_display)

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
    # Give title/legend and insight boxes more room (Slides-friendly)
    fig.subplots_adjust(hspace=0.38, wspace=0.25, top=0.86, bottom=0.08)
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
                        fontsize=12,
                        fontweight="bold",
                        color=label_color,
                    )
            cumulative += band_vals
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("% of Students", color="#111111")
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, color="#111111")
        ax1.tick_params(axis="y", colors="#111111")
        ax1.grid(axis="y", alpha=0.2)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        # Title
        ax1.set_title(
            f"{subject_titles[i]}",
            fontsize=18,
            fontweight="bold",
            y=1.08,
            color="#111111",
        )
    # Place legend once, centered above both facets
    leg = fig.legend(
        handles=legend_handles,
        labels=hf.IREADY_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(hf.IREADY_ORDER),
        frameon=False,
        fontsize=14,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    for t in leg.get_texts():
        t.set_color("#111111")
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
                fontsize=18,
                fontweight="bold",
                color="#111111",
            )
        # --- Add n-counts under xticklabels ---
        n_map = n_maps[i] if i < len(n_maps) else {}
        base_labels = score_df["time_label"].astype(str).tolist()
        labels_with_n = [
            f"{lbl}\n(n = {int(n_map.get(lbl, 0))})" for lbl in base_labels
        ]
        ax2.set_ylabel("Avg Scale Score", color="#111111")
        ax2.set_xticks(rit_x)
        ax2.set_xticklabels(labels_with_n, color="#111111")
        ax2.tick_params(axis="y", colors="#111111")
        ax2.set_title(
            "Average Scale Score", fontsize=14, fontweight="bold", pad=10, color="#111111"
        )
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
            fontsize=16,
            fontweight="medium",
            color="#111111",
            ha="center",
            va="center",
            wrap=True,
            usetex=False,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#eeeeee",
                edgecolor="#666666",
                linewidth=1.2,
            ),
        )
    # Main title for the whole figure
    fig.suptitle(
        f"{title_label} • {group_name} • Winter Year-to-Year Trends",
        fontsize=26,
        fontweight="bold",
        y=0.99,
        color="#111111",
    )
    # --- Save ---
    charts_dir = CHARTS_DIR
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

    out_name = f"{safe_scope}_section2_{group_order_val:02d}_{safe_group}_Winter_i-Ready_trends_eoy.png"
    out_path = out_dir / out_name

    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "iready_eoy_section2_student_group_dashboard",
            "section": 2,
            "scope": title_label,
            "scope_label": scope_label,
            "group_name": group_name,
            "window_filter": window_filter,
            "subjects": subjects,
            "pct_data": [
                {"subject": subjects[i], "data": _jsonable(pct_dfs[i])}
                for i in range(len(subjects))
            ],
            "score_data": [
                {"subject": subjects[i], "data": _jsonable(score_dfs[i])}
                for i in range(len(subjects))
            ],
            "metrics": [_jsonable(m) for m in metrics_list],
            "time_orders": [_jsonable(t) for t in time_orders],
        },
    )
    print(f"Saved: {out_path}")

    if preview:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------
# DRIVER — Faceted Student Group Dashboards (District and Site)
# ---------------------------------------------------------------------
student_groups_cfg = cfg.get("student_groups", {})
group_order = cfg.get("student_group_order", {})

# Optional: restrict student-group dashboards based on frontend selection.
# The runner passes selected groups as: IREADY_MOY_STUDENT_GROUPS="English Learners,Students with Disabilities"
_env_groups = _env_first("IREADY_EOY_STUDENT_GROUPS")
_selected_groups = []
if _env_groups:
    _selected_groups = [g.strip() for g in str(_env_groups).split(",") if g.strip()]
    print(f"[FILTER] Student group selection from frontend: {_selected_groups}")

# ---- District-level ----
scope_df = iready_base.copy()
scope_label = district_label

for group_name, group_def in sorted(
    student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99)
):
    if group_def.get("type") == "all":
        continue
    if _selected_groups and group_name not in _selected_groups:
        continue
    plot_iready_subject_dashboard_by_group(
        scope_df.copy(),
        subject_str=None,
        window_filter="Spring",
        group_name=group_name,
        group_def=group_def,
        figsize=(16, 9),
        school_raw=None,
        scope_label=scope_label,
    )

# ---- Site-level ----
if _include_school_charts():
    for school_col, raw_school in _iter_schools(iready_base):
        scope_df = iready_base[iready_base[school_col] == raw_school].copy()
        scope_label = hf._safe_normalize_school_name(raw_school, cfg)

        for group_name, group_def in sorted(
            student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99)
        ):
            if group_def.get("type") == "all":
                continue
            if _selected_groups and group_name not in _selected_groups:
                continue
            plot_iready_subject_dashboard_by_group(
                scope_df.copy(),
                subject_str=None,
                window_filter="Spring",
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
    scope_label = scope_label or district_label
    folder_name = (
        "_district"
        if scope_label == district_label
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
                        fontsize=12,
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
        ax.set_xticklabels(pivot.index.tolist(), fontsize=12)
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
                fontsize=14,
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
        ax.set_xticklabels(labels, fontsize=12)
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

    out_dir = CHARTS_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / f"{scope_label.replace(' ','_')}_section3_iready_grade{int(current_grade)}_{subject_str}_{window_filter.lower()}_trends_eoy.png"
    )
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "iready_eoy_section3_blended_dashboard",
            "section": 3,
            "scope": scope_label,
            "window_filter": window_filter,
            "grade": int(current_grade),
            "subject": subject_str,
            "pct_data": [
                {"subject": "overall", "data": _jsonable(pct_df_left)},
                {"subject": "cohort", "data": _jsonable(pct_df_right)},
            ],
            "score_data": [
                {"subject": "overall", "data": _jsonable(score_df_left)},
                {"subject": "cohort", "data": _jsonable(score_df_right)},
            ],
            "metrics": {"overall": _jsonable(metrics_left), "cohort": _jsonable(metrics_right)},
        },
    )
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

# Optional: restrict grade-level dashboards based on frontend selection.
# The runner passes selected grades as: IREADY_MOY_GRADES="3,4,5"
_env_grades2 = _env_first("IREADY_EOY_GRADES")
_selected_grades2 = None
if _env_grades2:
    try:
        _selected_grades2 = {
            int(x.strip()) for x in str(_env_grades2).split(",") if x.strip()
        }
        if _selected_grades2:
            print(f"[FILTER] Grade selection from frontend: {sorted(_selected_grades2)}")
    except Exception:
        _selected_grades2 = None

scope_label = district_label
for g in sorted(_base["student_grade"].dropna().unique()):
    if _selected_grades2 and int(g) not in _selected_grades2:
        continue
    for subj in ["ELA", "Math"]:
        plot_iready_blended_dashboard(
            _base.copy(),
            subj,
            int(g),
            "Spring",
            _anchor_year,
            scope_label=scope_label,
            preview=False,
        )

if _include_school_charts():
    for school_col, raw_school in _iter_schools(_base):
        site_df = _base[_base[school_col] == raw_school].copy()
        site_df["academicyear"] = pd.to_numeric(site_df["academicyear"], errors="coerce")
        site_df["student_grade"] = pd.to_numeric(site_df["student_grade"], errors="coerce")
        anchor = int(site_df["academicyear"].max())
        scope_label = hf._safe_normalize_school_name(raw_school, cfg)
        for g in sorted(site_df["student_grade"].dropna().unique()):
            if _selected_grades2 and int(g) not in _selected_grades2:
                continue
            for subj in ["ELA", "Math"]:
                plot_iready_blended_dashboard(
                    site_df.copy(),
                    subj,
                    int(g),
                    "Spring",
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
        & (d["testwindow"].astype(str).str.lower() == "spring")
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
                fontsize=14,
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
            r"$\mathbf{Mid\ or\ Above}$ Grade Level in Spring i-Ready tend to ",
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
        f"{scope_label} \n Spring i-Ready Mid/Above → % CERS Met/Exceeded",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )

    charts_dir = CHARTS_DIR
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    out_path = out_dir / f"{safe_scope}_section4_mid_plus_to_3plus_eoy.png"

    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "iready_eoy_section4_mid_plus_to_3plus",
            "section": 4,
            "scope": scope_label,
            "subjects": subjects,
            # Provide the underlying trend tables per subject so the analyzer can summarize.
            # (Each record has academicyear, n, me, pct_me)
            "pct_data": [
                {"subject": subj, "data": _jsonable(trends.get(subj))}
                for subj in subjects
            ],
            "metrics": {
                subj: {
                    "overall_pct_me": float(
                        (100 * trends[subj]["me"].sum() / trends[subj]["n"].sum())
                    )
                    if (subj in trends and not trends[subj].empty)
                    else None
                }
                for subj in subjects
            },
        },
    )
    print(f"[Section 4] Saved: {out_path}")

    if preview:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# DRIVER — Section 4 (District + Sites)
# ---------------------------------------------------------------------
print("Running Section 4 — Spring Mid/Above → CERS Met/Exceeded")

scope_df = iready_base.copy()
district_label = district_label
_plot_mid_above_to_cers_faceted(scope_df, district_label, folder_name="_district")

if _include_school_charts():
    for school_col, raw_school in _iter_schools(iready_base):
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


def run_section5_growth_progress_eoy(
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
        # Spring rows for same Fall cohort
        # ----------------------------
        spring = d0[(d0["testwindow"].astype(str).str.lower() == "spring")].copy()
        spring = _normalize_placement(spring)
        spring = _dedupe_latest(spring, id_col, ["completion_date", "teststartdate"])

        # restrict to Fall cohort ids
        cohort_ids = set(fall[id_col].dropna().unique().tolist())

        # DQC: confirm IDs intersect across windows before filtering
        spring_ids = set(spring[id_col].dropna().unique().tolist())
        inter = cohort_ids.intersection(spring_ids)
        print(
            f"[DQC][S5] {subj} {year} id_col={id_col} | "
            f"fall_ids={len(cohort_ids):,} spring_ids={len(spring_ids):,} intersect={len(inter):,}"
        )
        if len(inter) == 0:
            print(f"[DQC][S5] fall id sample: {list(cohort_ids)[:5]}")
            print(f"[DQC][S5] spring id sample: {list(spring_ids)[:5]}")

        # apply filter
        spring = spring[spring[id_col].isin(cohort_ids)].copy()

        if spring.empty:
            print(f"[WARN] Section 5: No Spring rows for Fall cohort ({subj}, {year})")
            return None

        # attach fall mid_flag to spring for cohort-wise tracking
        spring = spring.merge(
            fall[[id_col, "mid_flag"]].drop_duplicates(subset=[id_col]),
            on=id_col,
            how="left",
        )

        # ----------------------------
        # Progress % toward annual targets
        #
        # Many i-Ready exports populate Spring "percent_progress_to_annual_*" as 100 for most rows,
        # which makes EOY charts uninformative. Prefer computing progress from Fall baseline:
        #   observed_growth = SpringScale - FallScale
        #   pct_typ = observed_growth / annual_typical_growth_measure * 100
        #   pct_str = observed_growth / annual_stretch_growth_measure * 100
        # Fall measures are BOY-anchored and vary by student/grade.
        # ----------------------------

        # progress percents (exports vary in naming / casing / underscores).
        def _norm_col_s5(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

        def _find_col_like_s5(df_in: pd.DataFrame, canonical: str) -> str | None:
            target = _norm_col_s5(canonical)
            for c in df_in.columns:
                if _norm_col_s5(c) == target:
                    return c
            return None

        # Compute from Fall baseline when possible
        fall_scale = pd.to_numeric(fall.get("scale_score"), errors="coerce")
        fall_typ = pd.to_numeric(fall.get("annual_typical_growth_measure"), errors="coerce")
        fall_strg = pd.to_numeric(fall.get("annual_stretch_growth_measure"), errors="coerce")
        fall_base = pd.DataFrame(
            {
                id_col: fall[id_col],
                "_fall_scale": fall_scale,
                "_fall_typ": fall_typ,
                "_fall_str": fall_strg,
            }
        ).drop_duplicates(subset=[id_col])

        spring = spring.merge(fall_base, on=id_col, how="left")

        spring_scale = pd.to_numeric(spring.get("scale_score"), errors="coerce")
        observed = spring_scale - pd.to_numeric(spring.get("_fall_scale"), errors="coerce")

        pct_typ_calc = (observed / pd.to_numeric(spring.get("_fall_typ"), errors="coerce")) * 100
        pct_str_calc = (observed / pd.to_numeric(spring.get("_fall_str"), errors="coerce")) * 100

        # Fall back to export-provided % columns if we can't compute (missing measures)
        typ_src = _find_col_like_s5(spring, "percent_progress_to_annual_typical_growth")
        str_src = _find_col_like_s5(spring, "percent_progress_to_annual_stretch_growth")
        pct_typ_fallback = pd.to_numeric(spring[typ_src], errors="coerce") if typ_src else pd.Series(dtype=float)
        pct_str_fallback = pd.to_numeric(spring[str_src], errors="coerce") if str_src else pd.Series(dtype=float)

        spring["pct_typ"] = pct_typ_calc.where(pct_typ_calc.notna(), pct_typ_fallback)
        spring["pct_str"] = pct_str_calc.where(pct_str_calc.notna(), pct_str_fallback)

        # Cap negatives at 0; allow >100 (students can exceed annual targets)
        spring["pct_typ"] = spring["pct_typ"].clip(lower=0)
        spring["pct_str"] = spring["pct_str"].clip(lower=0)

        # ----------------------------
        # Metrics for chart panels
        # ----------------------------
        med_typ = float(np.nanmedian(spring["pct_typ"].to_numpy()))
        med_str = float(np.nanmedian(spring["pct_str"].to_numpy()))
        pct50_typ = float((spring["pct_typ"] >= 50).mean() * 100)
        pct50_str = float((spring["pct_str"] >= 50).mean() * 100)

        # Mid/Above in Spring (for progress update)
        spring_mid = (
            spring["relative_placement"]
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
        spring_mid_count = int(spring_mid.sum())
        spring_already_mid_count = spring_mid_count

        # Newly Mid+ by Winter (not Already Mid+ in Fall)
        fall_mid_ids = set(
            fall.loc[fall["mid_flag"] == "Already Mid+", id_col].tolist()
        )
        newly_mid = int(
            spring.loc[~spring[id_col].isin(fall_mid_ids), :][
                spring_mid.loc[~spring[id_col].isin(fall_mid_ids)].values
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
            w = spring[spring[id_col].isin(base_ids)].copy()
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
            spring_mid_count=spring_mid_count,
            spring_already_mid_count=spring_already_mid_count,
            newly_mid=newly_mid,
            on_typ=on_typ,
            on_str=on_str,
            on_beyond=on_beyond,
            n_spring=int(spring[id_col].nunique()),
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
                        fontsize=14,
                        fontweight="bold",
                        color="#333",
                    )
            # Add 50% reference line (lighter)
            ax_top.axhline(50, linestyle="--", linewidth=1.0, color="#999", alpha=0.6)
            # Allow >100 when students exceed annual targets, but keep a sensible cap for readability.
            _top_cap = float(np.nanmax(np.array(vals, dtype=float))) if any(pd.notna(v) for v in vals) else 100.0
            ax_top.set_ylim(0, max(100.0, min(200.0, _top_cap + 10.0)))
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
                        fontsize=14,
                        fontweight="bold",
                        color="#333",
                    )
            ax_mid.axhline(50, linestyle="--", linewidth=1.0, color="#666", alpha=0.8)
            _mid_cap = float(np.nanmax(np.array(vals2, dtype=float))) if any(pd.notna(v) for v in vals2) else 100.0
            ax_mid.set_ylim(0, max(100.0, min(100.0, _mid_cap + 10.0)))
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
            spring_mid_ct = int(
                m.get("spring_already_mid_count", m.get("spring_mid_count", 0))
            )

            lines = [
                rf"Fall Already Mid+ = $\mathbf{{{fall_mid_ct:,}}}$",
                rf"Spring Already Mid+ = $\mathbf{{{spring_mid_ct:,}}}$",
                "",
                "Fall to Spring Mid On Level Update:",
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
            f"{scope_label} • Spring {year} • Growth Progress Toward Annual Goals",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )

        out_dir = CHARTS_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scope_label}_section5_growth_progress_eoy.png"
        hf._save_and_render(fig, out_path)
        _write_chart_data(
            out_path,
            {
                "chart_type": "iready_eoy_section5_growth_progress",
                "section": 5,
                "scope": scope_label,
                "folder": folder,
                "subjects": list(metrics_by_subject.keys()),
                "window_filter": "Spring",
                # Section 5 is mostly metric-driven; pass the full per-subject metrics dict.
                "metrics": _jsonable(metrics_by_subject),
            },
        )
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
scope_label = district_label
folder = "_district"
run_section5_growth_progress_eoy(
    scope_df, scope_label=scope_label, folder=folder, preview=False
)

# Site-level
if _include_school_charts():
    for school_col, raw_school in _iter_schools(iready_base):
        scope_df = iready_base[iready_base[school_col] == raw_school].copy()
        scope_label = hf._safe_normalize_school_name(raw_school, cfg)
        folder = scope_label.replace(" ", "_")
        run_section5_growth_progress_eoy(
            scope_df, scope_label=scope_label, folder=folder, preview=False
        )

print("Section 5 batch complete.")


# %% SECTION 6 — Window Compare by School (District Only)
# ---------------------------------------------------------------------
# District-only ELA + Math chart.
# X-axis = school
# For each school: two 100% stacked bars (Fall vs Winter) using i-Ready relative_placement.
# ---------------------------------------------------------------------


def _prep_iready_fall_winter_by_dimension(
    df: pd.DataFrame,
    subject_str: str,
    dim_col: str,
    min_n: int = 1,
):
    d = df.copy()

    # If caller asks for "school", prefer learning_center when present, otherwise fall back.
    if dim_col == "school":
        for candidate in ["learning_center", "school", "schoolname", "school_name"]:
            if candidate in d.columns:
                try:
                    if d[candidate].dropna().astype(str).str.strip().ne("").any():
                        dim_col = candidate
                        break
                except Exception:
                    dim_col = candidate
                    break

    # Normalize i-Ready placement labels for consistency
    if "relative_placement" in d.columns and hasattr(hf, "IREADY_LABEL_MAP"):
        d["relative_placement"] = d["relative_placement"].replace(hf.IREADY_LABEL_MAP)

    # Required columns for this section; skip gracefully if missing.
    required_cols = ["academicyear", "testwindow", "subject", "domain", "enrolled", "relative_placement", dim_col]
    missing_required = [c for c in required_cols if c not in d.columns]
    if missing_required:
        print(f"[Section 6–8] Skipped (missing columns): {missing_required}")
        return None

    # Current year only
    d["academicyear"] = pd.to_numeric(d.get("academicyear"), errors="coerce")
    year = int(d["academicyear"].max())

    subj = str(subject_str).strip().upper()

    # Core filters (match Section 0.1)
    win_order = _get_eoy_compare_windows()
    d = d[
        (d["academicyear"] == year)
        & (d["testwindow"].astype(str).str.title().isin(win_order))
        & (d["subject"].astype(str).str.upper() == subj)
        & (d["domain"].astype(str) == "Overall")
        & (d["enrolled"].astype(str) == "Enrolled")
        & (d["relative_placement"].notna())
    ].copy()

    if d.empty:
        return None

    # Dimension handling (do NOT dedupe; Section 0.1 does not dedupe)
    # Keep totals aligned with Section 0.1 where possible.
    if dim_col in ["learning_center", "school", "schoolname", "school_name"]:
        d[dim_col] = d[dim_col].fillna("(No School)")
        # Respect selected-schools scope if provided by the runner/frontend.
        if _selected_schools:
            d = d[d[dim_col].apply(lambda s: _school_selected(s))].copy()
    elif dim_col == "student_grade":
        d[dim_col] = pd.to_numeric(d[dim_col], errors="coerce")
        d = d[d[dim_col].notna()].copy()
    else:
        # For non-numeric dims (e.g., student_group), keep only labeled rows
        d = d[d[dim_col].notna()].copy()

    if d.empty:
        return None

    # Normalize window labels for plotting
    d["testwindow"] = d["testwindow"].astype(str).str.title()

    # Counts per (dimension, window)
    counts = (
        d.groupby([dim_col, "testwindow", "relative_placement"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals = d.groupby([dim_col, "testwindow"]).size().rename("N_total").reset_index()
    pct = counts.merge(totals, on=[dim_col, "testwindow"], how="left")
    pct["pct"] = 100 * pct["n"] / pct["N_total"]

    # Ensure full placement set exists for stacking
    win_order = _get_eoy_compare_windows()
    dim_order = sorted(pct[dim_col].dropna().unique().tolist())
    all_idx = pd.MultiIndex.from_product(
        [dim_order, win_order, hf.IREADY_ORDER],
        names=[dim_col, "testwindow", "relative_placement"],
    )
    pct = (
        pct.set_index([dim_col, "testwindow", "relative_placement"])
        .reindex(all_idx)
        .reset_index()
    )
    pct["pct"] = pct["pct"].fillna(0)
    pct["n"] = pct["n"].fillna(0)

    # Rebuild N_total after reindex
    n_map = totals.copy()
    n_map["testwindow"] = n_map["testwindow"].astype(str).str.title()
    pct = pct.merge(n_map, on=[dim_col, "testwindow"], how="left", suffixes=("", "_y"))
    if "N_total_y" in pct.columns:
        pct["N_total"] = pct["N_total"].fillna(pct["N_total_y"])
        pct.drop(columns=["N_total_y"], inplace=True)
    pct["N_total"] = pct.groupby([dim_col, "testwindow"])["N_total"].transform(
        lambda s: s.ffill().bfill()
    )

    # Keep columns tidy
    pct["testwindow"] = pct["testwindow"].astype(str).str.title()
    pct = pct[
        [dim_col, "testwindow", "relative_placement", "n", "N_total", "pct"]
    ].copy()

    # Apply min_n filter based on latest window within each dimension
    latest_window = win_order[-1]
    latest_n = (
        pct[pct["testwindow"] == latest_window]
        .groupby(dim_col)["N_total"]
        .max()
        .fillna(0)
        .astype(float)
    )
    keep_dims = latest_n[latest_n >= min_n].index.tolist()
    pct = pct[pct[dim_col].isin(keep_dims)].copy()

    if pct.empty:
        return None

    return {
        "year": year,
        "pct": pct,
        "dim_order": sorted(keep_dims),
        "win_order": win_order,
    }


def _plot_fall_winter_grouped_stacked(
    prep: dict,
    scope_label: str,
    title_suffix: str,
    subject_label: str,
    dim_labeler=None,
    out_name: str = "section6_window_compare_by_school_eoy.png",
    preview: bool = False,
):
    # One subject per chart (NOT faceted)
    pct = prep["pct"].copy()
    dim_col = [
        c
        for c in pct.columns
        if c
        not in {"testwindow", "relative_placement", "n", "N_total", "pct", "subject"}
    ][0]

    dim_order = prep["dim_order"]
    win_order = prep["win_order"]

    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.86, bottom=0.26)

    # Legend once
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

    # Filter to this subject (if present)
    if "subject" in pct.columns:
        pdf = pct[
            pct["subject"].astype(str).str.upper() == subject_label.upper()
        ].copy()
    else:
        pdf = pct.copy()

    if pdf.empty:
        print(f"[Skip] No data for {subject_label} ({scope_label})")
        plt.close(fig)
        return

    # Build x positions (supports 2 or 3 windows)
    base_x = np.arange(len(dim_order))
    n_w = len(win_order)
    if n_w == 2:
        offsets = [-0.22, 0.22]
        bar_w = 0.38
    elif n_w == 3:
        offsets = [-0.28, 0.00, 0.28]
        bar_w = 0.26
    else:
        offsets = list(np.linspace(-0.30, 0.30, max(1, n_w)))
        bar_w = max(0.18, 0.80 / max(1, n_w))

    window_to_x = {w: (base_x + offsets[i]) for i, w in enumerate(win_order)}

    # Visual emphasis: fade earlier windows, emphasize latest.
    window_to_alpha = {}
    if n_w == 1:
        window_to_alpha[win_order[0]] = 0.90
    else:
        for i, w in enumerate(win_order):
            window_to_alpha[w] = 0.35 + (0.55 * (i / (n_w - 1)))

    window_to_label = {"Fall": "F", "Winter": "W", "Spring": "S"}

    # --- n-counts for Fall/Winter bars (subject-filtered) ---
    n_lookup = (
        pdf[[dim_col, "testwindow", "N_total"]]
        .dropna(subset=[dim_col, "testwindow"])
        .drop_duplicates(subset=[dim_col, "testwindow"])
    )

    def _n_for(dim_val, win):
        try:
            v = (
                n_lookup.loc[
                    (n_lookup[dim_col] == dim_val)
                    & (n_lookup["testwindow"].astype(str).str.title() == win),
                    "N_total",
                ]
                .astype(float)
                .max()
            )
            return int(v) if pd.notna(v) else 0
        except Exception:
            return 0

    # STAR-style x tick labels: category + per-window n counts on a second line
    xlabels = []
    for v in dim_order:
        disp = dim_labeler(v) if callable(dim_labeler) else str(v)
        parts = []
        for w in win_order:
            short = window_to_label.get(w, w[:1].upper())
            parts.append(f"{short} n={_n_for(v, w):,}")
        xlabels.append(f"{disp}\n(" + " | ".join(parts) + ")")

    for win in win_order:
        sub = pdf[pdf["testwindow"] == win].copy()
        piv = (
            sub.pivot(index=dim_col, columns="relative_placement", values="pct")
            .reindex(index=dim_order)
            .reindex(columns=hf.IREADY_ORDER)
            .fillna(0)
        )
        bottom = np.zeros(len(dim_order))
        for cat in hf.IREADY_ORDER:
            vals = piv[cat].to_numpy()
            bars = ax.bar(
                window_to_x[win],
                vals,
                bottom=bottom,
                width=bar_w,
                color=hf.IREADY_COLORS[cat],
                edgecolor="white",
                linewidth=1.0,
                alpha=window_to_alpha[win],
            )
            for j, v in enumerate(vals):
                if v >= LABEL_MIN_PCT:
                    ax.text(
                        window_to_x[win][j],
                        bottom[j] + v / 2,
                        f"{v:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color=(
                            "white"
                            if cat in ["3+ Below", "Mid/Above", "Early On"]
                            else "#333"
                        ),
                    )
            bottom += vals

        # Window label ABOVE bars
        for j in range(len(dim_order)):
            ax.text(
                window_to_x[win][j],
                101,
                window_to_label.get(win, win),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#333",
            )

    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_xticks(base_x)
    ax.set_xticklabels(xlabels)

    # STAR-style rotation: rotate only school-ish + student_group
    if dim_col in ["learning_center", "school", "schoolname", "school_name", "student_grade", "student_group"]:
        for t in ax.get_xticklabels():
            t.set_rotation(35)
            t.set_ha("right")

    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{scope_label} i-Ready {win_order[0]} vs {win_order[-1]} {prep['year']} \n {subject_label} {title_suffix}",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )

    out_dir = CHARTS_DIR / "_district"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = re.sub(r"[^A-Za-z0-9_]+", "_", scope_label.replace(" ", "_"))
    out_file = f"{safe_scope}_{subject_label}_{out_name}"
    out_path = out_dir / out_file
    hf._save_and_render(fig, out_path)
    print(f"[SAVE] {out_path}")

    if preview:
        plt.show()
    plt.close(fig)


def run_section6_fall_winter_by_school(
    df_scope, scope_label=None, preview=False
):
    print("\n>>> STARTING SECTION 6 <<<")
    scope_label = scope_label or district_label

    for subj in ["ELA", "Math"]:
        prep = _prep_iready_fall_winter_by_dimension(
            df_scope,
            subject_str=subj,
            dim_col="school",
            min_n=1,
        )
        if prep is None:
            print(f"[Section 6] Skipped {subj} (no data)")
            continue
        prep["pct"]["subject"] = subj

        _plot_fall_winter_grouped_stacked(
            prep,
            scope_label=scope_label,
            title_suffix="Trends by School",
            subject_label=subj,
            dim_labeler=lambda s: hf._safe_normalize_school_name(s, cfg),
            out_name="section6_window_compare_by_school_eoy.png",
            preview=preview,
        )


# %% SECTION 7 — Fall vs Winter by Grade (District Only)
# ---------------------------------------------------------------------
# District-only ELA + Math chart.
# X-axis = grade
# For each grade: two 100% stacked bars (Fall vs Winter).
# ---------------------------------------------------------------------


def run_section7_fall_winter_by_grade(
    df_scope, scope_label=None, preview=False
):
    print("\n>>> STARTING SECTION 7 <<<")
    scope_label = scope_label or district_label

    def _glabel(g):
        try:
            return str(int(float(g)))
        except Exception:
            return str(g)

    for subj in ["ELA", "Math"]:
        prep = _prep_iready_fall_winter_by_dimension(
            df_scope,
            subject_str=subj,
            dim_col="student_grade",
            min_n=1,
        )
        if prep is None:
            print(f"[Section 7] Skipped {subj} (no data)")
            continue
        prep["pct"]["subject"] = subj

        _plot_fall_winter_grouped_stacked(
            prep,
            scope_label=scope_label,
            title_suffix="Trends by Grade",
            subject_label=subj,
            dim_labeler=_glabel,
            out_name="section7_window_compare_by_grade_eoy.png",
            preview=preview,
        )


# %% SECTION 8 — Fall vs Winter by Student Group (District Only)
# ---------------------------------------------------------------------
# District-only ELA + Math chart.
# X-axis = selected student groups (editable list below)
# For each group: two 100% stacked bars (Fall vs Winter).
# Only include groups with n >= 12 (latest window) per subject.
# ---------------------------------------------------------------------


def run_section8_fall_winter_by_student_group(
    df_scope,
    scope_label=None,
    preview=False,
    min_n=12,
):
    print("\n>>> STARTING SECTION 8 <<<")
    scope_label = scope_label or district_label

    # Editable include list (comment out to exclude)
    # Names MUST match keys in cfg['student_groups']
    included_groups = [
        "All Students",
        "English Learners",
        "Students with Disabilities",
        "Socioeconomically Disadvantaged",
        "Hispanic or Latino",
        "White",
        # "Foster",
        # "Homeless",
        # "Black or African American",
        # "Asian",
        # "Filipino",
        # "American Indian or Alaska Native",
        # "Native Hawaiian or Pacific Islander",
        # "Two or More Races",
        # "Not Stated",
    ]

    student_groups_cfg = cfg.get("student_groups", {})

    group_order_map = cfg.get("student_group_order", {})

    def _group_sort_key(g: str):
        # Default to 99 if not specified
        return (int(group_order_map.get(g, 99)), g)

    # Only keep groups that exist in config
    included_groups = [g for g in included_groups if g in student_groups_cfg]
    included_groups = sorted(included_groups, key=_group_sort_key)
    if not included_groups:
        print("[Section 8] No included groups found in cfg['student_groups']")
        return

    def _build_group_labeled_df(dfin: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for gname in included_groups:
            gdef = student_groups_cfg[gname]
            mask = _apply_student_group_mask(dfin, gname, gdef)
            tmp = dfin[mask].copy()
            if tmp.empty:
                continue
            tmp["student_group"] = gname
            rows.append(tmp)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    d0 = df_scope.copy()
    d0 = _build_group_labeled_df(d0)
    if d0.empty:
        print("[Section 8] Skipped (no rows after group masks)")
        return

    for subj in ["ELA", "Math"]:
        prep = _prep_iready_fall_winter_by_dimension(
            d0[d0["student_group"].notna()].copy(),
            subject_str=subj,
            dim_col="student_group",
            min_n=min_n,
        )
        if prep is None:
            print(f"[Section 8] Skipped {subj} (no data)")
            continue
        # Enforce YAML-defined plotting order (not alphabetical) and respect min_n survivors
        survivors = set(prep.get("dim_order", []))
        ordered = [g for g in included_groups if g in survivors]
        prep["dim_order"] = ordered if ordered else prep.get("dim_order", [])
        prep["pct"]["subject"] = subj

        _plot_fall_winter_grouped_stacked(
            prep,
            scope_label=scope_label,
            title_suffix="Trends by Student Group",
            subject_label=subj,
            dim_labeler=lambda s: str(s),
            out_name="section8_window_compare_by_student_group_eoy.png",
            preview=preview,
        )


# ---------------------------------------------------------------------
# DRIVER — RUN SECTIONS 6–8 (District Only)
# ---------------------------------------------------------------------
print("Running Sections 6–8 (district-level only)...")

_scope_df = iready_base.copy()
_scope_label = district_label

run_section6_fall_winter_by_school(_scope_df, scope_label=_scope_label, preview=False)
run_section7_fall_winter_by_grade(_scope_df, scope_label=_scope_label, preview=False)
run_section8_fall_winter_by_student_group(
    _scope_df, scope_label=_scope_label, preview=False, min_n=12
)

print("Sections 6–8 complete.")


# %%
# %% SECTION 9/10/11 — Median % Progress to Annual Growth (District-level)
# ---------------------------------------------------------------------
# Section 9: By School (Winter only)
# Section 10: By Grade (Winter only)
# Section 11: By Student Group (Winter only)
# Each: District-level only, ELA and Math, ONE figure per subject.
# Each figure: grouped bars per scope (Typical vs Stretch).
# Filters aligned to Section 0.1 context, but Winter + most_recent_diagnostic == 'Yes'.
# ---------------------------------------------------------------------


def _prep_progress_base(
    df: pd.DataFrame,
    subject_str: str,
    cfg: dict,
) -> tuple[pd.DataFrame, int, str, str]:
    """Base filter for Sections 9–11.

    Filters (aligned to Section 0.1 context, but Winter + most_recent_diagnostic == 'Yes'):
      - academicyear == max
      - testwindow == WINTER
      - subject == ELA or Math (i-Ready stable)
      - domain == Overall
      - enrolled == Enrolled
      - most_recent_diagnostic == Yes

    Returns: (filtered_df, year, subject_label, id_col)
    """
    d = df.copy()

    required_cols = ["academicyear", "testwindow", "subject", "domain", "enrolled", "student_grade"]
    missing_required = [c for c in required_cols if c not in d.columns]
    if missing_required:
        print(f"[Sections 9–11] Skipped (missing columns): {missing_required}")
        return pd.DataFrame(), np.nan, str(subject_str), "student_id"

    # Current year only
    d["academicyear"] = pd.to_numeric(d.get("academicyear"), errors="coerce")
    year = int(d["academicyear"].max()) if d["academicyear"].notna().any() else np.nan

    # Subject filtering (i-Ready is strictly ELA or Math)
    subj = str(subject_str).strip().upper()
    if subj == "ELA":
        subj_label = "ELA"
        subj_mask = d["subject"].astype(str).str.upper().eq("ELA")
    else:
        subj_label = "Math"
        subj_mask = d["subject"].astype(str).str.upper().eq("MATH")

    # Safe most_recent_diagnostic mask
    if "most_recent_diagnostic" in d.columns:
        mrd_mask = (
            d["most_recent_diagnostic"].astype(str).str.strip().str.lower().eq("yes")
        )
    else:
        # If the column is missing, nothing qualifies (keeps behavior explicit)
        mrd_mask = pd.Series(False, index=d.index)

    # Core filters
    d = d[
        (d["academicyear"] == year)
        & (d["testwindow"].astype(str).str.strip().str.upper() == "WINTER")
        & subj_mask
        & (d["domain"].astype(str).str.strip() == "Overall")
        & (d["enrolled"].astype(str).str.strip() == "Enrolled")
        & (d["student_grade"] < 9)
        & mrd_mask
    ].copy()

    # Choose ID column and normalize type
    id_col = "student_id" if "student_id" in d.columns else "uniqueidentifier"

    if d.empty:
        return pd.DataFrame(), year, subj_label, id_col

    d[id_col] = d[id_col].astype(str).str.strip()
    d.loc[d[id_col].isin(["nan", "None", "<NA>"]), id_col] = np.nan

    # Normalize + coerce progress columns (exports may have extra underscores / casing differences)
    def _norm_col(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

    def _find_col_like(canonical: str) -> str | None:
        target = _norm_col(canonical)
        for c in d.columns:
            if _norm_col(c) == target:
                return c
        return None

    def _is_all_zero_or_nan(series: pd.Series) -> bool:
        try:
            s = pd.to_numeric(series, errors="coerce")
            return bool(((s.fillna(0) == 0).all()))
        except Exception:
            return True

    def _pick_progress_source(candidates: list[str]) -> str | None:
        """
        Pick the best matching source column among candidates.
        Preference order:
        - first candidate with data (non-null and not all zeros)
        - otherwise first candidate that exists at all
        """
        found_existing: list[str] = []
        for cand in candidates:
            src = _find_col_like(cand)
            if not src:
                continue
            found_existing.append(src)
            if src in d.columns and not _is_all_zero_or_nan(d[src]):
                return src
        return found_existing[0] if found_existing else None

    rename_map = {}
    # Some exports provide annual progress columns; others provide non-annual variants.
    _typ_src = _pick_progress_source(
        [
            "percent_progress_to_annual_typical_growth",
            "percent_progress_to_typical_growth",
            "percent_progress_to_typical",
        ]
    )
    _str_src = _pick_progress_source(
        [
            "percent_progress_to_annual_stretch_growth",
            "percent_progress_to_stretch_growth",
            "percent_progress_to_stretch",
        ]
    )
    if _typ_src and _typ_src != "percent_progress_to_annual_typical_growth":
        rename_map[_typ_src] = "percent_progress_to_annual_typical_growth"
    if _str_src and _str_src != "percent_progress_to_annual_stretch_growth":
        rename_map[_str_src] = "percent_progress_to_annual_stretch_growth"
    if rename_map:
        d = d.rename(columns=rename_map)

    for c in [
        "percent_progress_to_annual_typical_growth",
        "percent_progress_to_annual_stretch_growth",
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan

    # School display helper (district charts still need normalized school names)
    school_col = None
    for col_name in ["learning_center", "school", "schoolname", "school_name"]:
        if col_name in d.columns:
            school_col = col_name
            break
    # Respect selected-schools scope if provided by the runner/frontend.
    if _selected_schools and school_col is not None:
        d = d[d[school_col].apply(lambda s: _school_selected(s))].copy()
    if school_col is None:
        d["school_display"] = "Unknown"
    else:
        d["school_display"] = d[school_col].apply(
            lambda s: (
                hf._safe_normalize_school_name(s, cfg) if pd.notna(s) else "Unknown"
            )
        )

    return d, year, subj_label, id_col


# ----------- CAP PERCENT AXIS HELPER -----------
PROGRESS_CAP_PCT = 120


def _apply_capped_percent_axis(ax, cap: float = PROGRESS_CAP_PCT, step: int = 20):
    """Cap axis at `cap` and label the top tick as ≥cap."""
    ax.set_ylim(0, cap)
    yticks = np.arange(0, cap + 1, step)
    ylabels = [f"{int(t)}%" for t in yticks]
    if len(ylabels) > 0:
        ylabels[-1] = f"\u2265{int(cap)}%"  # ≥120%
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)


def _dedupe_latest_per_scope(
    d: pd.DataFrame, id_col: str, scope_cols: list[str]
) -> pd.DataFrame:
    """Deduplicate to latest record per student within each scope for Winter."""
    if d.empty:
        return d

    sort_col = None
    if "completion_date" in d.columns:
        d["completion_date"] = pd.to_datetime(d["completion_date"], errors="coerce")
        sort_col = "completion_date"
    elif "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
        sort_col = "teststartdate"

    keys = [id_col] + scope_cols
    if sort_col is None:
        d = d.sort_values(keys)
        return d.groupby(keys, as_index=False).tail(1)

    d = d.sort_values(keys + [sort_col])
    return d.groupby(keys, as_index=False).tail(1)


# --- District-wide median progress helper ---
def _districtwide_medians(d: pd.DataFrame, id_col: str) -> tuple[float, float]:
    """Compute district-wide medians (Typical, Stretch) from the filtered Winter dataset.

    Uses latest Winter record per student (no additional scope grouping).
    """
    if d is None or d.empty:
        return (np.nan, np.nan)

    dd = _dedupe_latest_per_scope(d.copy(), id_col=id_col, scope_cols=[])

    typ = (
        pd.to_numeric(
            dd.get("percent_progress_to_annual_typical_growth"), errors="coerce"
        )
        if "percent_progress_to_annual_typical_growth" in dd.columns
        else pd.Series(dtype=float)
    )
    st = (
        pd.to_numeric(
            dd.get("percent_progress_to_annual_stretch_growth"), errors="coerce"
        )
        if "percent_progress_to_annual_stretch_growth" in dd.columns
        else pd.Series(dtype=float)
    )

    return (float(np.nanmedian(typ.to_numpy())), float(np.nanmedian(st.to_numpy())))


def _plot_grouped_typ_stretch(
    grp: pd.DataFrame,
    scope_col: str,
    scope_order: list,
    title: str,
    out_path: Path,
    ref_typical: float | None = None,
    ref_stretch: float | None = None,
    preview: bool = False,
    rotate: int = 45,
):
    """Single-axis grouped bars per scope: Typical vs Stretch."""
    if grp.empty:
        print(f"[Skip] Empty grouped data for {title}")
        return

    # Ensure ordering
    grp = grp.copy()
    grp[scope_col] = pd.Categorical(
        grp[scope_col], categories=scope_order, ordered=True
    )
    grp = grp.sort_values(scope_col)

    # --- n-counts for x-axis labels ---
    n_map = {}
    if "n_students" in grp.columns:
        try:
            n_map = (
                grp[[scope_col, "n_students"]]
                .dropna(subset=[scope_col])
                .drop_duplicates(subset=[scope_col])
                .set_index(scope_col)["n_students"]
                .to_dict()
            )
        except Exception:
            n_map = {}

    x_labels = [f"{str(v)}\n(n={int(n_map.get(v, 0)):,})" for v in scope_order]
    x = np.arange(len(scope_order))
    width = 0.38

    c_typ = hf.default_quartile_colors[3]
    c_str = hf.default_quartile_colors[2]

    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.88, bottom=0.20)

    typ = grp.set_index(scope_col)["median_typical"].reindex(scope_order).to_numpy()
    st = grp.set_index(scope_col)["median_stretch"].reindex(scope_order).to_numpy()

    # Cap displayed bars so outliers (e.g., 200%+) don't blow up the axis
    typ_true = typ
    st_true = st
    typ_disp = np.array(
        [np.nan if pd.isna(v) else min(float(v), PROGRESS_CAP_PCT) for v in typ_true],
        dtype=float,
    )
    st_disp = np.array(
        [np.nan if pd.isna(v) else min(float(v), PROGRESS_CAP_PCT) for v in st_true],
        dtype=float,
    )

    ax.bar(
        x - width / 2,
        typ_disp,
        width=width,
        color=c_typ,
        edgecolor="white",
        label="Typical",
    )
    ax.bar(
        x + width / 2,
        st_disp,
        width=width,
        color=c_str,
        edgecolor="white",
        label="Stretch",
    )

    # --- District-wide reference lines (Typical / Stretch) ---
    if ref_typical is not None and pd.notna(ref_typical):
        y_ref = min(float(ref_typical), PROGRESS_CAP_PCT)
        ax.axhline(y_ref, linestyle=":", linewidth=1.6, color=c_typ, alpha=0.95)

    if ref_stretch is not None and pd.notna(ref_stretch):
        y_ref = min(float(ref_stretch), PROGRESS_CAP_PCT)
        ax.axhline(y_ref, linestyle="--", linewidth=1.6, color=c_str, alpha=0.95)

    # Labels: show TRUE values (even if capped for display)
    for i, (tv_true, sv_true, tv_disp, sv_disp) in enumerate(
        zip(typ_true, st_true, typ_disp, st_disp)
    ):
        if pd.notna(tv_true):
            y = (tv_disp if pd.notna(tv_disp) else 0) + 1
            ax.text(
                i - width / 2,
                y,
                f"{float(tv_true):.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#333",
            )
        if pd.notna(sv_true):
            y = (sv_disp if pd.notna(sv_disp) else 0) + 1
            ax.text(
                i + width / 2,
                y,
                f"{float(sv_true):.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#333",
            )

    _apply_capped_percent_axis(ax)
    ax.set_ylabel("Median % Progress")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rotate, ha="right" if rotate else "center")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Legend with explicit handles including reference lines ---
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=c_typ, lw=10, label="Median % Typical"),
        Line2D([0], [0], color=c_str, lw=10, label="Median % Stretch"),
    ]
    if ref_typical is not None and pd.notna(ref_typical):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=c_typ,
                lw=1.6,
                linestyle=":",
                label="District Median Typical",
            )
        )
    if ref_stretch is not None and pd.notna(ref_stretch):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=c_str,
                lw=1.6,
                linestyle="--",
                label="District Median Stretch",
            )
        )
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    hf._save_and_render(fig, out_path)
    print(f"[SAVE] {out_path}")

    if preview:
        plt.show()
    plt.close(fig)


# SECTION 9 — Median Progress by School (Winter only)
def plot_section9_median_progress_by_school(
    df, subject_str, district_label, preview=False
):
    d, year, subj_label, id_col = _prep_progress_base(df, subject_str, cfg)
    if d.empty:
        print(
            f"[Section 9] Skipped {subj_label} (no data after Winter+MostRecent filters)"
        )
        return

    # Compute district-wide medians for reference lines
    ref_typ, ref_str = _districtwide_medians(d, id_col=id_col)

    # Deduplicate latest per student per school (Winter)
    d = _dedupe_latest_per_scope(d, id_col=id_col, scope_cols=["school_display"])

    grp = (
        d.groupby("school_display")
        .agg(
            median_typical=("percent_progress_to_annual_typical_growth", "median"),
            median_stretch=("percent_progress_to_annual_stretch_growth", "median"),
            n_students=(id_col, "nunique"),
        )
        .reset_index()
    )

    school_order = sorted(grp["school_display"].dropna().unique().tolist())
    title = (
        f"{district_label} • {year} • {subj_label} • Winter Median % Progress by School"
    )

    out_dir = CHARTS_DIR / "_district"
    out_path = (
        out_dir
        / f"{district_label.replace(' ', '_')}_section9_{subj_label}_winter_median_progress_by_school_eoy.png"
    )

    _plot_grouped_typ_stretch(
        grp,
        scope_col="school_display",
        scope_order=school_order,
        title=title,
        out_path=out_path,
        ref_typical=ref_typ,
        ref_stretch=ref_str,
        preview=preview,
        rotate=45,
    )


# SECTION 10 — Median Progress by Grade (Winter only)
def plot_section10_median_progress_by_grade(
    df, subject_str, district_label, preview=False
):
    d, year, subj_label, id_col = _prep_progress_base(df, subject_str, cfg)
    if d.empty:
        print(
            f"[Section 10] Skipped {subj_label} (no data after Winter+MostRecent filters)"
        )
        return

    # Compute district-wide medians for reference lines
    ref_typ, ref_str = _districtwide_medians(d, id_col=id_col)

    d["student_grade"] = pd.to_numeric(d.get("student_grade"), errors="coerce")
    d = d[d["student_grade"] < 9].copy()
    if d.empty:
        print(f"[Section 10] Skipped {subj_label} (no valid grades)")
        return

    # Deduplicate latest per student per grade (Winter)
    d = _dedupe_latest_per_scope(d, id_col=id_col, scope_cols=["student_grade"])

    grp = (
        d.groupby("student_grade")
        .agg(
            median_typical=("percent_progress_to_annual_typical_growth", "median"),
            median_stretch=("percent_progress_to_annual_stretch_growth", "median"),
            n_students=(id_col, "nunique"),
        )
        .reset_index()
    )

    grade_order_num = sorted(grp["student_grade"].dropna().unique().tolist())
    grade_order = ["K" if int(g) == 0 else str(int(g)) for g in grade_order_num]

    # Map numeric grades to labels
    grp["grade_label"] = grp["student_grade"].apply(
        lambda g: "K" if int(g) == 0 else str(int(g))
    )

    title = (
        f"{district_label} • {year} • {subj_label} • Winter Median % Progress by Grade"
    )

    out_dir = CHARTS_DIR / "_district"
    out_path = (
        out_dir
        / f"{district_label.replace(' ', '_')}_section10_{subj_label}_winter_median_progress_by_grade_eoy.png"
    )

    _plot_grouped_typ_stretch(
        grp,
        scope_col="grade_label",
        scope_order=grade_order,
        title=title,
        out_path=out_path,
        ref_typical=ref_typ,
        ref_stretch=ref_str,
        preview=preview,
        rotate=0,
    )


# SECTION 11 — Median Progress by Student Group (Winter only)
def plot_section11_median_progress_by_group(
    df,
    subject_str,
    district_label,
    cfg,
    preview=False,
    min_n: int = 12,
):
    d, year, subj_label, id_col = _prep_progress_base(df, subject_str, cfg)
    if d.empty:
        print(
            f"[Section 11] Skipped {subj_label} (no data after Winter+MostRecent filters)"
        )
        return

    # Compute district-wide medians for reference lines
    ref_typ, ref_str = _districtwide_medians(d, id_col=id_col)

    group_defs = cfg.get("student_groups", {})
    group_order_map = cfg.get("student_group_order", {})

    # Included groups:
    # - If the frontend selected Student Groups and/or Race/Ethnicity, the runner passes them
    #   via IREADY_EOY_STUDENT_GROUPS, and we respect that here.
    # - Otherwise, fall back to a default list (matches Section 8 conventions).
    if isinstance(globals().get("_selected_groups"), list) and len(globals().get("_selected_groups")) > 0:
        included_groups = list(globals().get("_selected_groups"))
    else:
        included_groups = [
            "All Students",
            "English Learners",
            "Students with Disabilities",
            "Socioeconomically Disadvantaged",
            "Hispanic or Latino",
            "White",
            # "Black or African American",
            # "Asian",
            # "Filipino",
            # "American Indian or Alaska Native",
            # "Native Hawaiian or Pacific Islander",
            # "Two or More Races",
            # "Not Stated",
            # "Foster",
            # "Homeless",
        ]

    # Only keep groups that exist in config
    included_groups = [g for g in included_groups if g in group_defs]
    if not included_groups:
        print("[Section 11] No included groups found in cfg['student_groups']")
        return

    # Enforce YAML-defined plotting order (not alphabetical)
    included_groups = sorted(
        included_groups, key=lambda k: int(group_order_map.get(k, 99))
    )

    records = []
    for group_name in included_groups:
        group_def = group_defs[group_name]
        mask = _apply_student_group_mask(d, group_name, group_def)
        sub = d[mask].copy()
        if sub.empty:
            continue

        # Deduplicate latest per student within group (Winter)
        sub = _dedupe_latest_per_scope(sub, id_col=id_col, scope_cols=[])

        n_students = int(sub[id_col].nunique())
        if n_students < min_n:
            continue

        records.append(
            {
                "student_group": group_name,
                "median_typical": sub[
                    "percent_progress_to_annual_typical_growth"
                ].median(),
                "median_stretch": sub[
                    "percent_progress_to_annual_stretch_growth"
                ].median(),
                "n_students": n_students,
            }
        )

    if not records:
        print(f"[Section 11] No qualifying groups for {subj_label} (min_n={min_n})")
        return

    grp = pd.DataFrame.from_records(records)

    # Plot only included groups that survived min_n, still in YAML order
    survivors = set(grp["student_group"].unique().tolist())
    plot_groups = [g for g in included_groups if g in survivors]

    title = f"{district_label} • {year} • {subj_label} • Winter Median % Progress by Student Group"

    out_dir = CHARTS_DIR / "_district"
    out_path = (
        out_dir
        / f"{district_label.replace(' ', '_')}_section11_{subj_label}_winter_median_progress_by_group_eoy.png"
    )

    _plot_grouped_typ_stretch(
        grp,
        scope_col="student_group",
        scope_order=plot_groups,
        title=title,
        out_path=out_path,
        ref_typical=ref_typ,
        ref_stretch=ref_str,
        preview=preview,
        rotate=30,
    )


# --- DRIVER for Sections 9, 10, 11 (district-level only) ---
print("Running Sections 9, 10, 11 (district-level winter median progress)...")
scope_df = iready_base.copy()
for subj in ["ELA", "Math"]:
    plot_section9_median_progress_by_school(
        scope_df, subj, district_label, preview=False
    )
    plot_section10_median_progress_by_grade(
        scope_df, subj, district_label, preview=False
    )
    plot_section11_median_progress_by_group(
        scope_df, subj, district_label, cfg, preview=False, min_n=12
    )
print("Sections 9, 10, 11 batch complete.")
# %%




# %%
