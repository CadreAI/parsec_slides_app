# %% Imports and config
# nwea.py — charts and analytics
# NOTE: This is a legacy script. It is executed via `nwea_boy_runner.py` in the app.
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
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib import transforms as mtransforms
from matplotlib import lines as mlines
import helper_functions_nwea as hf

_NWEA_BOY_HARD_RC = {
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
mpl.rcParams.update(_NWEA_BOY_HARD_RC)

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
import warnings
import logging
import json
import tempfile
from nwea_data import filter_nwea_subject_rows

warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress matplotlib font warnings
warnings.filterwarnings("ignore", message=".*findfont.*", category=UserWarning, module="matplotlib")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# Temp workspace defaults
#
# This legacy script historically defaulted to writing into ../charts and ../logs
# (relative to `backend/python/nwea/`), which lands inside the repo.
#
# In the app, runners set `NWEA_BOY_CHARTS_DIR` / `NWEA_BOY_LOG_DIR` to temp paths.
# When those env vars are NOT set (e.g., running the script manually), we now
# default to a per-run system temp directory to avoid polluting the repo.
# ---------------------------------------------------------------------
_env_charts_dir = os.getenv("NWEA_BOY_CHARTS_DIR")
_env_log_dir = os.getenv("NWEA_BOY_LOG_DIR")
_env_run_root = os.getenv("NWEA_BOY_RUN_ROOT")

_RUN_ROOT = None
if not _env_charts_dir or not _env_log_dir:
    try:
        _RUN_ROOT = Path(_env_run_root) if _env_run_root else Path(tempfile.mkdtemp(prefix="parsec_nwea_boy_"))
    except Exception:
        _RUN_ROOT = Path(tempfile.mkdtemp(prefix="parsec_nwea_boy_"))

# Setup logging - both console and file
LOG_DIR = Path(_env_log_dir) if _env_log_dir else (_RUN_ROOT / "logs" if _RUN_ROOT else Path("../logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"nwea_boy_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates
logger.handlers = []

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Logging initialized. Log file: {log_file}")

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

# ----------------------------
# Frontend-driven filter inputs
# ----------------------------
def _parse_env_csv(var_name: str):
    raw = os.getenv(var_name)
    if not raw:
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return parts or None


_selected_subjects = _parse_env_csv("NWEA_BOY_SUBJECTS")
if _selected_subjects:
    logger.info(f"[FILTER] Subject selection from frontend: {_selected_subjects}")


def _requested_subjects(default_subjects: list[str]):
    """Subjects to generate for per-subject sections (e.g., Section 3/4)."""
    return _selected_subjects if _selected_subjects else default_subjects


def _requested_core_subjects():
    """
    Core NWEA subjects used in many legacy sections.
    Returns a subset of ["Reading", "Mathematics"] if frontend selected subjects.
    """
    if not _selected_subjects:
        return ["Reading", "Mathematics"]
    subj_join = " | ".join(str(s).casefold() for s in _selected_subjects)
    out = []
    if any(k in subj_join for k in ["reading", "ela", "language arts"]):
        out.append("Reading")
    if "math" in subj_join:
        out.append("Mathematics")
    return out


def _requested_core_dual_subjects():
    """
    For legacy dual-subject dashboards that are hard-coded to 2 columns.

    If frontend selects subjects and it does NOT include both Reading and Math,
    we skip these dashboards to avoid partial/blank layouts.
    """
    if not _selected_subjects:
        return ["Reading", "Math K-12"], ["Reading", "Math"]

    subj_join = " | ".join(str(s).casefold() for s in _selected_subjects)
    has_read = any(k in subj_join for k in ["reading", "ela", "language arts"])
    has_math = "math" in subj_join
    if not (has_read and has_math):
        return [], []
    return ["Reading", "Math K-12"], ["Reading", "Math"]


# Global threshold for inline % labels on stacked bars
LABEL_MIN_PCT = 5.0
# Toggle for cohort DQC printouts
COHORT_DEBUG = True

# ---------------------------------------------------------------------
# Load partner-specific config using settings.yaml pointer
# ---------------------------------------------------------------------
SETTINGS_PATH = Path(
    os.getenv("NWEA_BOY_SETTINGS_PATH")
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
    os.getenv("NWEA_BOY_CONFIG_PATH")
    or (Path(__file__).resolve().parent / "config_files" / f"{partner_name}.yaml")
)
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Partner config not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

print(f"Loaded config for: {cfg.get('partner_name', partner_name)}")

# Charter-style partners may omit district_name or provide an empty list.
# Ensure downstream cfg.get("district_name")[0] calls always have a usable value.
try:
    _dn = cfg.get("district_name")
    _dn0 = _dn[0] if isinstance(_dn, list) and _dn else None
    if not _dn0 or not str(_dn0).strip():
        cfg["district_name"] = ["District"]
except Exception:
    cfg["district_name"] = ["District"]

# ---------------------------------------------------------------------
# SINGLE PREVIEW / DEV-MODE TOGGLE
# One source of truth: CLI or ENV override; defaults to False
# Usage:
#   python main.py --preview     → enables preview
#   python main.py --full        → disables preview
#   export PREVIEW=true          → enables preview globally
# ---------------------------------------------------------------------

DEV_MODE = False  # FALSE = Batch Run; TRUE = Preview Mode

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
DATA_DIR = Path(os.getenv("NWEA_BOY_DATA_DIR") or "../data")
csv_path = DATA_DIR / "nwea_data.csv"
LABEL_MIN_PCT = 5.0

if not csv_path.exists():
    raise FileNotFoundError(
        f"Expected CSV not found: {csv_path}. Please run data_ingest.py first."
    )

nwea_base = pd.read_csv(csv_path)
nwea_base.columns = nwea_base.columns.str.strip().str.lower()

# Charter-style exports sometimes put site/school in `learning_center` instead of `schoolname`.
# Normalize a single school key into `schoolname` so the rest of the script (loops, filtering,
# folder naming, matching) works unchanged.
_school_key_candidates = ["learning_center", "schoolname", "school", "school_name"]
_school_key = None
for _c in _school_key_candidates:
    if _c in nwea_base.columns:
        try:
            if nwea_base[_c].dropna().astype(str).str.strip().ne("").any():
                _school_key = _c
                break
        except Exception:
            _school_key = _c
            break

if _school_key and _school_key != "schoolname":
    nwea_base["schoolname"] = nwea_base[_school_key]
# --- Normalize projected proficiency labels globally ---
prof_prof_map = {
    "Not Met": "Level 1 - Standard Not Met",
    "Nearly Met": "Level 2 - Standard Nearly Met",
    "Met": "Level 3 - Standard Met",
    "Exceeded": "Level 4 - Standard Exceeded",
    "Level 1": "Level 1 - Standard Not Met",
    "Level 2": "Level 2 - Standard Nearly Met",
    "Level 3": "Level 3 - Standard Met",
    "Level 4": "Level 4 - Standard Exceeded",
}

# Base charts directory (overrideable by runner)
CHARTS_DIR = Path(_env_charts_dir) if _env_charts_dir else (_RUN_ROOT / "charts" if _RUN_ROOT else Path("../charts"))

# ---------------------------------------------------------------------
# Scope selection (district-only vs district + schools vs selected schools)
#
# Env vars (set by runner / backend):
# - NWEA_BOY_SCOPE_MODE:
#     - "district_only" (skip all school loops)
#     - "selected_schools" (only loop selected schools; still include district)
#     - default/other: district + all schools
# - NWEA_BOY_SCHOOLS="School A,School B" (names can be raw or normalized)
# ---------------------------------------------------------------------
_scope_mode = str(os.getenv("NWEA_BOY_SCOPE_MODE") or "").strip().lower()
_env_schools = os.getenv("NWEA_BOY_SCHOOLS")
_selected_schools = []
if _env_schools:
    _selected_schools = [s.strip() for s in str(_env_schools).split(",") if s.strip()]


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
    for rs in sorted(df["schoolname"].dropna().unique()):
        if not _school_selected(rs):
            continue
        yield rs


def _write_chart_data(out_path: Path, chart_data: dict) -> None:
    """
    Write sidecar JSON data for chart analysis.

    `chart_analyzer.py` expects a file named `{chart_stem}_data.json` next to the PNG.
    Uses atomic replace to avoid leaving partial JSON on disk.
    """
    try:
        p = Path(out_path)
        data_path = p.parent / f"{p.stem}_data.json"
        tmp_path = p.parent / f"{p.stem}_data.json.tmp"

        payload = chart_data or {}
        text = json.dumps(payload, indent=2, default=str)

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        os.replace(tmp_path, data_path)
    except Exception:  # pragma: no cover
        try:
            if "tmp_path" in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
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

nwea_base["projectedproficiencylevel2"] = nwea_base[
    "projectedproficiencylevel2"
].replace(prof_prof_map)
print(f"NWEA data loaded: {nwea_base.shape[0]:,} rows, {nwea_base.shape[1]} columns")
print(nwea_base["year"].value_counts().sort_index())
print(nwea_base.columns.tolist())

# Normalize district name for fallback in the title (allow display-only override)
try:
    _ddn = cfg.get("district_display_name")
    _ddn0 = _ddn[0] if isinstance(_ddn, list) and _ddn else (_ddn if isinstance(_ddn, str) else None)
    district_label = str(_ddn0).strip() if _ddn0 and str(_ddn0).strip() else cfg.get("district_name", ["Districtwide"])[0]
except Exception:
    district_label = cfg.get("district_name", ["Districtwide"])[0]

# Optional display label for "(All Students)" variants (falls back to "{district_label} (All Students)")
try:
    _das = cfg.get("district_all_students_label")
    if isinstance(_das, list) and _das and str(_das[0]).strip():
        district_all_students_label = str(_das[0]).strip()
    elif isinstance(_das, str) and _das.strip():
        district_all_students_label = _das.strip()
    else:
        district_all_students_label = f"{district_label} (All Students)"
except Exception:
    district_all_students_label = f"{district_label} (All Students)"

# Inspect categorical columns (quick QC) — expensive on large datasets.
# Only run when preview/dev mode is enabled.
if bool(getattr(hf, "DEV_MODE", False)) is True:
    cat_cols = [
        c
        for c in nwea_base.columns
        if nwea_base[c].dtype == "object" or nwea_base[c].dtype.name == "category"
    ]
    print("\n--- Unique values per categorical column ---")
    for c in cat_cols:
        uniq = nwea_base[c].dropna().unique()
        n = len(uniq)
        sample = uniq[:10]
        print(f"\n{c} ({n} unique): {sample}")


# %% SECTION 0 — Predicted vs Actual CAASPP (Spring)
# --------------------------------------------------
# Only Spring of max_year - 1
# Left = ELA, Right = Math
# Top row: 100% stacked — projected proficiency vs actual CAASPP
# Bottom: % Met/Exceed comparison
# Insight box: Predicted vs Actual Met/Exceed delta
# --------------------------------------------------


def _prep_section0(df, subject):
    logger.info(f"[FILTER] _prep_section0: Starting | Subject: {subject} | Input rows: {len(df):,}")
    d = df.copy()
    before_spring = len(d)
    d = d[d["testwindow"].str.upper() == "SPRING"].copy()
    logger.info(f"[FILTER] After Spring window filter: {len(d):,} rows (removed {before_spring - len(d):,})")

    # Bail out cleanly if nothing to work with
    if d.empty or d["year"].dropna().empty:
        return None, None, None, None

    # Subject filter
    before_subj = len(d)
    subj = subject.lower()
    if "math" in subj:
        d = d[d["course"].str.contains("math", case=False, na=False)]
        logger.info(f"[FILTER] After Math course filter: {len(d):,} rows (removed {before_subj - len(d):,})")
    else:
        d = d[d["course"].str.contains("read", case=False, na=False)]
        logger.info(f"[FILTER] After Reading course filter: {len(d):,} rows (removed {before_subj - len(d):,})")

    # If filtering killed the data, bail
    if d.empty or d["year"].dropna().empty:
        return None, None, None, None

    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    if d["year"].dropna().empty:  # safety net
        return None, None, None, None

    # Target year is the latest Spring test year present (no offset)
    target_year = int(d["year"].max())

    # Keep only the latest Spring year slice
    d = d[d["year"] == target_year].copy()

    # Dedupe — most recent Spring test per student
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
        d = d.sort_values("teststartdate").drop_duplicates(
            "uniqueidentifier", keep="last"
        )

    # Only keep valid values
    d = d.dropna(subset=["projectedproficiencylevel2", "cers_overall_performanceband"])
    if d.empty:
        return None, None, None, target_year

    # Order categories
    proj_order = sorted(d["projectedproficiencylevel2"].unique())
    act_order = hf.CERS_LEVELS

    # Calculate % distributions
    def pct_table(col, order):
        return (
            d.groupby(col)
            .size()
            .reindex(order, fill_value=0)
            .pipe(lambda s: 100 * s / s.sum())
        )

    proj_pct = pct_table("projectedproficiencylevel2", proj_order)
    act_pct = pct_table("cers_overall_performanceband", act_order)

    # Met or Exceed logic
    def pct_met_exceed(series, met_levels):
        return 100 * d[d[series].isin(met_levels)].shape[0] / d.shape[0]

    proj_met = pct_met_exceed(
        "projectedproficiencylevel2",
        ["Level 3 - Standard Met", "Level 4 - Standard Exceeded"],
    )
    act_met = pct_met_exceed(
        "cers_overall_performanceband",
        ["Level 3 - Standard Met", "Level 4 - Standard Exceeded"],
    )
    delta = proj_met - act_met

    metrics = {
        "proj_met": proj_met,
        "act_met": act_met,
        "delta": delta,
        "year": target_year,
        "proj_order": proj_order,
        "act_order": act_order,
        "proj_pct": proj_pct,
        "act_pct": act_pct,
    }

    return proj_pct, act_pct, metrics, target_year


def _plot_section0_dual(scope_label, folder, subj_payload, preview=False):
    """
    Render a two‑facet chart (Reading left, Math right) where each facet uses
    its own projected vs actual distributions and its own insight box.
    `subj_payload` is a dict keyed by subject name ("Reading", "Mathematics"),
    each value is a dict with keys: proj_pct, act_pct, metrics.
    """
    logger.info(f"[CHART] _plot_section0_dual: Starting | Scope: {scope_label} | Folder: {folder} | Subjects: {list(subj_payload.keys())}")
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    subjects = [s for s in ["Reading", "Mathematics"] if s in subj_payload]
    titles = {"Reading": "Reading", "Mathematics": "Math"}

    # Build legend from CERS levels
    # Use the first available subject's act_order
    first_metrics = next(iter(subj_payload.values()))["metrics"]
    handles = [
        Patch(facecolor=hf.CERS_LEVEL_COLORS[l], edgecolor="none", label=l)
        for l in first_metrics["act_order"]
    ]
    fig.legend(
        handles=handles,
        labels=first_metrics["act_order"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(first_metrics["act_order"]),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    for i, subject in enumerate(subjects):
        proj_pct = subj_payload[subject]["proj_pct"]
        act_pct = subj_payload[subject]["act_pct"]
        metrics = subj_payload[subject]["metrics"]

        # ---- upper stacked bars (Predicted vs Actual) ----
        bar_ax = fig.add_subplot(gs[0, i])
        # Predicted
        cumulative = 0
        for level in metrics["proj_order"]:
            val = float(proj_pct.get(level, 0))
            idx = metrics["proj_order"].index(level)
            mapped_level = (
                metrics["act_order"][idx]
                if idx < len(metrics["act_order"])
                else metrics["act_order"][-1]
            )
            col = hf.CERS_LEVEL_COLORS.get(mapped_level, "#cccccc")
            bars = bar_ax.bar(
                -0.2,
                val,
                bottom=cumulative,
                width=0.35,
                color=col,
                alpha=0.6,
                edgecolor="#434343",
                linewidth=1.2,
                linestyle="--",
            )
            rect = bars.patches[0]
            if val >= LABEL_MIN_PCT:
                bar_ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    cumulative + val / 2.0,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#434343",
                )
            cumulative += val
        # Actual
        cumulative = 0
        for level in metrics["act_order"]:
            val = float(act_pct.get(level, 0))
            col = hf.CERS_LEVEL_COLORS.get(level, "#cccccc")
            bars = bar_ax.bar(
                0.2,
                val,
                bottom=cumulative,
                width=0.35,
                color=col,
                edgecolor="white",
                linewidth=1.2,
            )
            rect = bars.patches[0]
            if val >= LABEL_MIN_PCT:
                txt_color = "#434343" if "Nearly" in level else "white"
                bar_ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    cumulative + val / 2.0,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=txt_color,
                )
            cumulative += val
        bar_ax.set_xticks([-0.2, 0.2])
        bar_ax.set_xticklabels(["Predicted", "Actual"])
        bar_ax.set_ylim(0, 100)
        bar_ax.set_ylabel("% of Students")
        bar_ax.set_title(titles[subject], fontsize=14, fontweight="bold", pad=30)
        bar_ax.grid(axis="y", alpha=0.5)
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)

        # ---- lower bars (% Met/Exceed) ----
        pct_ax = fig.add_subplot(gs[1, i])
        pct_ax.bar(
            "Pred Met/Exc",
            metrics["proj_met"],
            color=hf.CERS_LEVEL_COLORS["Level 4 - Standard Exceeded"],
            alpha=0.6,
            edgecolor="#434343",
            linewidth=1.2,
            linestyle="--",
        )
        pct_ax.bar(
            "Actual Met/Exc",
            metrics["act_met"],
            color=hf.CERS_LEVEL_COLORS["Level 4 - Standard Exceeded"],
            alpha=1.0,
            edgecolor=hf.CERS_LEVEL_COLORS["Level 4 - Standard Exceeded"],
            linewidth=1.2,
        )
        for x, v in zip([0, 1], [metrics["proj_met"], metrics["act_met"]]):
            pct_ax.text(
                x,
                v + 1,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#434343",
            )
        pct_ax.set_ylim(0, 100)
        pct_ax.set_ylabel("% Met/Exc")
        pct_ax.grid(axis="y", alpha=0.2)
        pct_ax.spines["top"].set_visible(False)
        pct_ax.spines["right"].set_visible(False)

        # ---- facet insight box ----
        ax3 = fig.add_subplot(gs[2, i])
        ax3.axis("off")
        # Use subject-specific metrics already computed above
        pred = float(metrics["proj_met"])  # % predicted Met/Exceeded
        act = float(metrics["act_met"])  # % actual Met/Exceeded
        delta = pred - act  # follow chart order: Predicted − Actual
        insight_text = (
            r"Predicted vs Actual Met/Exceed:"
            + "\n"
            + rf"${pred:.1f}\% - {act:.1f}\% = \mathbf{{{pred - act:+.1f}}}$ pts"
        )
        ax3.text(
            0.5,
            0.5,
            insight_text,
            fontsize=12,
            ha="center",
            va="center",
            color="#434343",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#f5f5f5",
                edgecolor="#ccc",
                linewidth=1.0,
            ),
        )

    # Title uses the year from first_metrics
    fig.suptitle(
        f"{scope_label} • Spring {first_metrics['year']} Prediction Accuracy",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )

    out_dir = CHARTS_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep naming consistent across scripts for chart split + analyzer.
    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section0_pred_vs_actual.png"
    out_path = out_dir / out_name
    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "nwea_boy_section0_pred_vs_actual",
            "section": 0,
            "scope": scope_label,
            "folder": folder,
            "window_filter": "Fall",
            "subjects": list(subj_payload.keys()) if isinstance(subj_payload, dict) else [],
            # Help downstream "Chart Value Check" detect this chart as meaningful.
            # (It looks for `predicted_vs_actual` specifically.)
            "predicted_vs_actual": subj_payload if isinstance(subj_payload, dict) else {},
            "subj_payload": {
                k: {
                    "metrics": v.get("metrics", {}) if isinstance(v, dict) else {},
                    "proj_pct": _jsonable(v.get("proj_pct")) if isinstance(v, dict) else None,
                    "act_pct": _jsonable(v.get("act_pct")) if isinstance(v, dict) else None,
                }
                for k, v in (subj_payload or {}).items()
            }
            if isinstance(subj_payload, dict)
            else {},
        },
    )
    logger.info(f"[CHART] Saved Section 0: {out_path}")
    print(f"Saved Section 0: {out_path}")
    if preview:
        plt.show()
    plt.close()


# ---- RUN SECTION 0 ----
_section0_schools = list(_iter_schools(nwea_base)) if _include_school_charts() else []
for raw in [None] + _section0_schools:
    if raw is None:
        scope_df = nwea_base
        scope_label = district_label
        folder = "_district"
    else:
        scope_df = nwea_base[nwea_base["schoolname"] == raw].copy()
        scope_label = hf._safe_normalize_school_name(raw, cfg)
        folder = scope_label.replace(" ", "_")

    # Align subject handling with `nwea_moy.py`:
    # Section 0 is a Reading/Math CAASPP comparison; if the frontend selected only
    # non-core subjects, skip instead of attempting to generate empty charts.
    payload = {}
    for subj in _requested_core_subjects():
        proj, act, metrics, _ = _prep_section0(scope_df, subj)
        if proj is None:
            continue
        payload[subj] = {"proj_pct": proj, "act_pct": act, "metrics": metrics}

    if payload:
        _plot_section0_dual(scope_label, folder, payload, preview=False)


# %% SECTION 1 - Fall Performance Trends
# Subject Dashboards by Year/Window
# Example labels: "Fall 22-23", "Fall 23-24", "Fall 24-25", "Fall 25-26"
# Rules:
#   - Window: Fall only (default, configurable)
#   - Subject filtering:
#       - Reading: course contains "reading" (case-insensitive), excludes language usage
#       - Math: course contains "math"
#   - Only valid achievementquintile rows retained
#   - Latest test per student per year used (based on teststartdate)
#   - Y-axis: percent of students in each quintile (100% stacked bar)
#   - Second panel: average RIT score per year
#   - Third panel: insight box with deltas (last 2 years)
#   - District chart includes all; school charts filtered to site
# ---------------------------------------------------------------------
def _prep_nwea_for_charts(
    df: pd.DataFrame,
    subject_str: str,
    window_filter: str = "Fall",
):
    """
    Filters and aggregates NWEA data for dashboard plotting.

    Rules:
      - Keep only the requested test window (ex: "Fall").
      - For Mathematics:
            keep rows where course contains "math".
      - For Reading:
            keep rows where course is "reading" or "reading (spanish)".
            exclude "language usage".
      - Drop rows with missing achievementquintile.
      - Keep the latest test per student per time_label using the most recent
        teststartdate.
      - Build time_label like "22-23 Fall".
      - Return:
            pct_df   = % by quintile per time window
            score_df = avg RIT per time window
            metrics  = delta metrics between last two windows
            time_order = ordered list of time labels
    """

    d = df.copy()
    initial_rows = len(d)
    logger.info(f"[FILTER] _prep_nwea_for_charts: Starting with {initial_rows:,} rows | Subject: {subject_str} | Window: {window_filter}")

    # 1. restrict to requested test window
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()].copy()
    after_window = len(d)
    logger.info(f"[FILTER] After window filter '{window_filter}': {after_window:,} rows (removed {initial_rows - after_window:,})")

    # 2. subject filtering (supports Reading/Math as well as optional MAP Growth subjects)
    before_subject = len(d)
    d = filter_nwea_subject_rows(d, subject_str)
    logger.info(
        f"[FILTER] After subject '{subject_str}' filter: {len(d):,} rows (removed {before_subject - len(d):,})"
    )

    # 3. require valid quintile bucket
    before_quintile = len(d)
    d = d[d["achievementquintile"].notna()].copy()
    after_quintile = len(d)
    logger.info(f"[FILTER] After quintile filter (removed nulls): {after_quintile:,} rows (removed {before_quintile - after_quintile:,})")

    # 4. build "22-23 Fall" style label
    def _short_year(y):
        """
        Return a YY-YY pair for an integer school year (e.g., 2026 -> '25-26').
        If already a 'YYYY-YYYY' string, keep the last-two/last-two.
        """
        ys = str(y)
        if "-" in ys:
            a, b = ys.split("-", 1)
            return f"{a[-2:]}-{b[-2:]}"
        yi = int(float(ys))
        return f"{str(yi-1)[-2:]}-{str(yi)[-2:]}"

    # ensure year numeric for sorting (handles "2024-2025" cases)
    d["year"] = pd.to_numeric(d["year"].astype(str).str[:4], errors="coerce")
    d["year_short"] = d["year"].apply(_short_year)
    d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["year_short"]

    # 5. dedupe to latest attempt per student per time_label
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    else:
        d["teststartdate"] = pd.NaT

    before_dedup = len(d)
    d.sort_values(["uniqueidentifier", "time_label", "teststartdate"], inplace=True)
    d = d.groupby(["uniqueidentifier", "time_label"], as_index=False).tail(1)
    after_dedup = len(d)
    logger.info(f"[FILTER] After deduplication (latest per student/time_label): {after_dedup:,} rows (removed {before_dedup - after_dedup:,} duplicates)")

    # 6. percent by quintile
    quint_counts = (
        d.groupby(["time_label", "achievementquintile"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_counts = d.groupby("time_label").size().rename("N_total").reset_index()

    pct_df = quint_counts.merge(total_counts, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]

    # ensure all quintiles exist for stacking in Low->High order
    all_idx = pd.MultiIndex.from_product(
        [pct_df["time_label"].unique(), hf.NWEA_ORDER],
        names=["time_label", "achievementquintile"],
    )
    pct_df = (
        pct_df.set_index(["time_label", "achievementquintile"])
        .reindex(all_idx)
        .reset_index()
    )
    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df.groupby("time_label")["N_total"].transform(
        lambda s: s.ffill().bfill()
    )

    # 7. avg RIT per time_label — restrict to deduped df used in pct_df
    score_df = (
        d[["time_label", "testritscore"]]
        .dropna(subset=["testritscore"])
        .groupby("time_label")["testritscore"]
        .mean()
        .rename("avg_score")
        .reset_index()
    )

    # 8. enforce chronological order
    time_order = sorted(pct_df["time_label"].unique().tolist())
    logger.info(f"[FILTER] Final data: {len(d):,} rows | Time periods: {time_order} | Unique students: {d['uniqueidentifier'].nunique():,}")

    pct_df["time_label"] = pd.Categorical(
        pct_df["time_label"],
        categories=time_order,
        ordered=True,
    )
    score_df["time_label"] = pd.Categorical(
        score_df["time_label"],
        categories=time_order,
        ordered=True,
    )

    pct_df.sort_values(["time_label", "achievementquintile"], inplace=True)
    score_df.sort_values(["time_label"], inplace=True)

    # --- Ensure year column is present in both pct_df and score_df ---
    if "year" not in pct_df.columns:
        # Try to map from d['time_label'] to d['year'] and merge
        time_label_to_year = d.drop_duplicates("time_label")[["time_label", "year"]]
        pct_df = pct_df.merge(time_label_to_year, on="time_label", how="left")
    if "year" not in score_df.columns:
        time_label_to_year = d.drop_duplicates("time_label")[["time_label", "year"]]
        score_df = score_df.merge(time_label_to_year, on="time_label", how="left")

    # 9. insight metrics from last two windows
    last_two = time_order[-2:] if len(time_order) >= 2 else time_order
    if len(last_two) == 2:
        t_prev, t_curr = last_two

        def pct_for(bucket_list, tlabel):
            return pct_df[
                (pct_df["time_label"] == tlabel)
                & (pct_df["achievementquintile"].isin(bucket_list))
            ]["pct"].sum()

        hi_curr = pct_for(hf.NWEA_HIGH_GROUP, t_curr)  # Avg+HiAvg+High
        hi_prev = pct_for(hf.NWEA_HIGH_GROUP, t_prev)
        lo_curr = pct_for(hf.NWEA_LOW_GROUP, t_curr)  # Low
        lo_prev = pct_for(hf.NWEA_LOW_GROUP, t_prev)

        # Add High only group
        high_curr = pct_for(["High"], t_curr)
        high_prev = pct_for(["High"], t_prev)
        # Add high_delta to metrics right after computing high_prev
        metrics = {}
        metrics["high_delta"] = high_curr - high_prev

        score_curr = float(
            score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
        )
        score_prev = float(
            score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0]
        )

        metrics.update(
            {
                "t_prev": t_prev,
                "t_curr": t_curr,
                "hi_now": hi_curr,
                "hi_delta": hi_curr - hi_prev,
                "lo_now": lo_curr,
                "lo_delta": lo_curr - lo_prev,
                "score_now": score_curr,
                "score_delta": score_curr - score_prev,
                "high_now": high_curr,
                # "high_delta" already set above
            }
        )
    else:
        metrics = {
            "t_prev": None,
            "t_curr": time_order[-1] if time_order else None,
            "hi_now": None,
            "hi_delta": None,
            "lo_now": None,
            "lo_delta": None,
            "score_now": None,
            "score_delta": None,
            "high_now": None,
            "high_delta": None,
        }

    return pct_df, score_df, metrics, time_order


# ---------------------------------------------------------------------
# Plot dashboard
# ---------------------------------------------------------------------
def plot_nwea_dual_subject_dashboard(
    df,
    window_filter="Fall",
    figsize=(16, 9),
    school_raw=None,
    scope_label=None,
    preview=False,
    *,
    _subjects_override: list[str] | None = None,
    _titles_override: list[str] | None = None,
):
    """
    Faceted dashboard for NWEA Section 1.

    Behavior:
      - If 2 subjects: render a 2-column dashboard (legacy "dual").
      - If 1 subject: render a 1-column dashboard.
      - If 3+ subjects: render one 1-column dashboard per subject (separate files).
    """
    logger.info(
        f"[CHART] plot_nwea_dual_subject_dashboard: Starting | Scope: {scope_label} | Window: {window_filter} | Input rows: {len(df):,}"
    )

    # Use explicit override (internal recursion helper) if provided.
    if isinstance(_subjects_override, list) and _subjects_override:
        subjects = _subjects_override
    else:
        # Use frontend-selected subjects if present; otherwise default to Reading + Mathematics.
        subjects = _selected_subjects if _selected_subjects else ["Reading", "Mathematics"]

    def _display_title(s: str) -> str:
        sl = str(s).strip().casefold()
        if "math" in sl:
            return "Math"
        if "reading" in sl or "ela" in sl or "language arts" in sl:
            return "Reading"
        return str(s).strip() or str(s)

    if isinstance(_titles_override, list) and _titles_override:
        titles = _titles_override
    else:
        titles = [_display_title(s) for s in subjects]

    # 3+ subjects => generate one single-subject chart per subject (avoids broken/wide layouts).
    if len(subjects) > 2 and not (isinstance(_subjects_override, list) and _subjects_override):
        out_paths = []
        for subj, title in zip(subjects, titles):
            out_paths.extend(
                plot_nwea_dual_subject_dashboard(
                    df,
                    window_filter=window_filter,
                    figsize=figsize,
                    school_raw=school_raw,
                    scope_label=scope_label,
                    preview=preview,
                    _subjects_override=[subj],
                    _titles_override=[title],
                )
                or []
            )
        return out_paths

    def draw_stacked_bar(ax, pct_df, score_df, labels):
        # Reshape for stacking
        stack_df = (
            pct_df.pivot(
                index="time_label", columns="achievementquintile", values="pct"
            )
            .reindex(columns=hf.NWEA_ORDER)
            .fillna(0)
        )
        x_labels = stack_df.index.tolist()
        x = np.arange(len(x_labels))
        cumulative = np.zeros(len(stack_df))
        for cat in hf.NWEA_ORDER:
            band_vals = stack_df[cat].to_numpy()
            bars = ax.bar(
                x,
                band_vals,
                bottom=cumulative,
                color=hf.NWEA_COLORS[cat],
                edgecolor="white",
                linewidth=1.2,
            )
            for idx, rect in enumerate(bars):
                h = band_vals[idx]
                if h >= LABEL_MIN_PCT:
                    bottom_before = cumulative[idx]
                    if cat == "High" or cat == "HiAvg":
                        label_color = "white"
                    elif cat == "Avg" or cat == "LoAvg":
                        label_color = "#434343"
                    elif cat == "Low":
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
            cumulative += band_vals
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of Students")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # legend_handles = [
        #     Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q)
        #     for q in hf.NWEA_ORDER
        # ]
        # ax.legend(
        #     handles=legend_handles,
        #     labels=hf.NWEA_ORDER,
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 1.1),
        #     ncol=len(hf.NWEA_ORDER),
        #     frameon=False,
        #     fontsize=9,
        #     handlelength=1.5,
        #     handletextpad=0.4,
        #     columnspacing=1.0,
        # )

    def draw_score_bar(ax, score_df, labels):
        rit_x = np.arange(len(score_df["time_label"]))
        rit_vals = score_df["avg_score"].to_numpy()
        rit_bars = ax.bar(
            rit_x,
            rit_vals,
            color=hf.default_quintile_colors[4],
            edgecolor="white",
            linewidth=1.2,
        )
        for rect, v in zip(rit_bars, rit_vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#434343",
            )
        # --- add n counts to x-axis labels ---
        # After defining rit_vals and rit_bars, before adding text labels
        import pandas as pd

        if "n" in score_df.columns:
            n_map = score_df[["time_label", "n"]]
        else:
            n_map = (
                pct_df.groupby("time_label")["n"].sum().reset_index()
                if "pct_df" in locals()
                else pd.DataFrame()
            )
        if not n_map.empty:
            label_map = {
                row["time_label"]: f"{row['time_label']}\n(n = {int(row['n'])})"
                for _, row in n_map.iterrows()
            }
            x_labels = [label_map.get(lbl, str(lbl)) for lbl in score_df["time_label"]]
        else:
            x_labels = score_df["time_label"].astype(str).tolist()
        ax.set_ylabel("Avg RIT")
        ax.set_xticks(rit_x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_insight_card(ax, metrics, title):
        ax.axis("off")

        if metrics.get("t_prev"):
            t_prev = metrics["t_prev"]
            t_curr = metrics["t_curr"]
            pct_df = metrics.get("pct_df")

            # Helper for bucket delta
            def _bucket_delta(bucket: str, pct_df: pd.DataFrame) -> float:
                curr = pct_df.loc[pct_df["achievementquintile"] == bucket]
                curr = curr.loc[curr["time_label"] == t_curr, "pct"].sum()
                prev = pct_df.loc[pct_df["achievementquintile"] == bucket]
                prev = prev.loc[prev["time_label"] == t_prev, "pct"].sum()
                return curr - prev

            # Use metrics['pct_df'] if present, else fallback to None
            if pct_df is None:
                insight_lines = ["(No pct_df for insight calculation)"]
            else:
                high_delta = _bucket_delta("High", pct_df)
                hi_delta = sum(
                    _bucket_delta(b, pct_df) for b in ["Avg", "HiAvg", "High"]
                )
                lo_delta = _bucket_delta("Low", pct_df)
                score_delta = metrics["score_delta"]
                title_line = "Change calculations from " + f"{t_prev} to {t_curr}:\n"
                line_high = rf"High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
                line_hiavg = (
                    rf"Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
                )
                line_low = rf"Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
                line_rit = rf"Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
                insight_lines = [title_line, line_high, line_hiavg, line_low, line_rit]
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
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#f5f5f5",
                edgecolor="#ccc",
                linewidth=0.8,
            ),
        )

    # Internal override for single-subject generation when 3+ subjects are requested
    if isinstance(_subjects_override, list) and _subjects_override:
        subjects = _subjects_override
    if isinstance(_titles_override, list) and _titles_override:
        titles = _titles_override

    ncols = len(subjects) if len(subjects) in (1, 2) else 2
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=ncols, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    _pct_payload = []
    _score_payload = []
    _metrics_payload = []
    _time_orders_payload = []

    for i, (course_filter, title) in enumerate(zip(subjects, titles)):
        pct_df, score_df, metrics, _ = _prep_nwea_for_charts(
            df,
            subject_str=course_filter,
            window_filter=window_filter,
        )
        # Limit to 4 years
        recent_years = sorted(pct_df["year"].unique())[-4:]
        pct_df = pct_df.query("year in @recent_years")
        score_df = score_df.query("year in @recent_years")

        # Add pct_df to metrics for draw_insight_card
        metrics = dict(metrics)  # copy to avoid mutating original
        metrics["pct_df"] = pct_df

        _pct_payload.append({"subject": title, "data": pct_df.to_dict("records") if not pct_df.empty else []})
        _score_payload.append({"subject": title, "data": score_df.to_dict("records") if not score_df.empty else []})
        _metrics_payload.append(metrics if isinstance(metrics, dict) else {})
        _time_orders_payload.append(sorted(pct_df["time_label"].astype(str).unique().tolist()) if (pct_df is not None and not pct_df.empty and "time_label" in pct_df.columns) else [])

        ax1 = fig.add_subplot(gs[0, i])
        draw_stacked_bar(ax1, pct_df, score_df, hf.NWEA_ORDER)
        ax1.set_title(f"{title}", fontsize=14, fontweight="bold", pad=30)

        ax2 = fig.add_subplot(gs[1, i])
        draw_score_bar(ax2, score_df, hf.NWEA_ORDER)
        ax2.set_title("Avg RIT Score", fontsize=8, fontweight="bold", pad=10)

        ax3 = fig.add_subplot(gs[2, i])
        draw_insight_card(ax3, metrics, title)

    # Shared legend across both subject panels (top center)
    legend_handles = [
        Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q)
        for q in hf.NWEA_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        labels=hf.NWEA_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(hf.NWEA_ORDER),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # Main title
    # Use the first district key from settings if school_raw is None, else use school_raw
    # Normalize display label from YAML school_name_map
    if school_raw:
        title_label = hf._safe_normalize_school_name(school_raw, cfg)
    else:
        # Use the configured district display label (module-level) when rendering districtwide charts
        title_label = district_label

    fig.suptitle(
        f"{title_label} • {window_filter} Year-to-Year Trends",
        fontsize=20,
        fontweight="bold",
        y=1,
    )
    # --- Save chart with scope_label for consistent naming ---
    charts_dir = CHARTS_DIR
    folder_name = "_district" if school_raw is None else scope_label.replace(" ", "_")
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(subjects) == 1:
        safe_subj = titles[0].replace(" ", "_").replace("/", "_")
        out_name = (
            f"{scope_label.replace(' ', '_')}_section1_{safe_subj}_dashboard.png"
        )
    else:
        out_name = f"{scope_label.replace(' ', '_')}_section1_dual_subject_dashboard.png"
    out_path = out_dir / out_name

    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "nwea_boy_section1_dual_subject_dashboard",
            "section": 1,
            "scope": scope_label,
            "window_filter": window_filter,
            "subjects": titles,
            # These fields are used by `decision_llm.py` chart value check.
            "pct_data": _pct_payload,
            "score_data": _score_payload,
            "metrics": _metrics_payload,
            "time_orders": _time_orders_payload,
        },
    )
    logger.info(f"[CHART] Saved: {out_path}")
    print(f"Saved: {out_path}")

    if preview:
        plt.show()
    plt.close()
    return [str(out_path)]


# ---------------------------------------------------------------------
# Dual Subject District Dashboard
# ---------------------------------------------------------------------
scope_label = district_label
folder = "_district"

plot_nwea_dual_subject_dashboard(
    nwea_base,
    window_filter="Fall",
    figsize=(16, 9),
    school_raw=None,
    scope_label=scope_label,  # <-- new param
    preview=True,  # or False for batch
)

# ---------------------------------------------------------------------
# Dual Subject Dashboard by School
# ---------------------------------------------------------------------
if _include_school_charts():
    for raw_school in _iter_schools(nwea_base):
        school_display = hf._safe_normalize_school_name(raw_school, cfg)
        folder = school_display.replace(" ", "_").replace("/", "_").replace("&", "and")

        school_df = nwea_base[nwea_base["schoolname"] == raw_school].copy()

        plot_nwea_dual_subject_dashboard(
            school_df,
            window_filter="Fall",
            figsize=(16, 9),
            school_raw=school_display,
            scope_label=school_display,  # <-- pass same label for title + save
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


def plot_nwea_subject_dashboard_by_group(
    df: pd.DataFrame,
    subject_str: str,
    window_filter: str,
    group_name: str,
    group_def: dict,
    figsize=(16, 9),
    school_raw: str | None = None,
    scope_label: str | None = None,
    preview: bool = False,
):
    """
    Same visual layout as the main dashboard and the grade dashboard
    but filtered to one student group.
    We also enforce min n >= 12 unique students in the current scope.
    If facet=True, plot both Reading and Mathematics side by side in a single chart.
    """

    # filter to this student group
    d0 = df.copy()

    # normalize school pretty label
    school_display = (
        hf._safe_normalize_school_name(school_raw, cfg) if school_raw else None
    )
    title_label = district_all_students_label if not school_display else school_display

    # Legacy group dashboard is a 2-column layout (Reading + Math). Respect frontend
    # subject filters by skipping if both aren't selected.
    if _selected_subjects:
        subj_join = " | ".join(str(s).casefold() for s in _selected_subjects)
        has_read = any(k in subj_join for k in ["reading", "ela", "language arts"])
        has_math = "math" in subj_join
        if not (has_read and has_math):
            logger.info(
                f"[CHART] Section 2: skipping group dashboard for '{group_name}' (frontend did not request both Reading and Math)"
            )
            return

    subjects = ["Reading", "Mathematics"]
    subject_titles = ["Reading", "Mathematics"]
    # Apply group mask first
    before_group_filter = len(d0)
    mask = _apply_student_group_mask(d0, group_name, group_def)
    d0 = d0[mask].copy()
    after_group_filter = len(d0)
    logger.info(f"[FILTER] After student group filter '{group_name}': {after_group_filter:,} rows (removed {before_group_filter - after_group_filter:,})")
    if d0.empty:
        logger.warning(f"[FILTER] No rows after group mask '{group_name}' ({school_raw or 'district'})")
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
    for subj in subjects:
        # Filter for subject
        before_subj_filter = len(d0)
        if subj == "Reading":
            subj_df = d0[
                d0["course"].astype(str).str.contains("reading", case=False, na=False)
            ].copy()
            logger.info(f"[FILTER] After Reading course filter (group '{group_name}'): {len(subj_df):,} rows (removed {before_subj_filter - len(subj_df):,})")
        elif subj == "Mathematics":
            subj_df = d0[
                d0["course"].astype(str).str.contains("math", case=False, na=False)
            ].copy()
            logger.info(f"[FILTER] After Math course filter (group '{group_name}'): {len(subj_df):,} rows (removed {before_subj_filter - len(subj_df):,})")
        else:
            subj_df = d0.copy()
        if subj_df.empty:
            pct_dfs.append(None)
            score_dfs.append(None)
            metrics_list.append(None)
            time_orders.append([])
            min_ns.append(0)
            continue
        pct_df, score_df, metrics, time_order = _prep_nwea_for_charts(
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

    # Setup subplots: 3 rows x 2 columns (Reading left, Math right)
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
        Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q)
        for q in hf.NWEA_ORDER
    ]
    for i, subj in enumerate(subjects):
        pct_df = pct_dfs[i]
        score_df = score_dfs[i]
        metrics = metrics_list[i]
        time_order = time_orders[i]
        stack_df = (
            pct_df.pivot(
                index="time_label", columns="achievementquintile", values="pct"
            )
            .reindex(columns=hf.NWEA_ORDER)
            .fillna(0)
        )
        x_labels = stack_df.index.tolist()
        x = np.arange(len(x_labels))
        ax1 = axes[0][i]
        cumulative = np.zeros(len(stack_df))
        for cat in hf.NWEA_ORDER:
            band_vals = stack_df[cat].to_numpy()
            bars = ax1.bar(
                x,
                band_vals,
                bottom=cumulative,
                color=hf.NWEA_COLORS[cat],
                edgecolor="white",
                linewidth=1.2,
            )
            for idx, rect in enumerate(bars):
                h = band_vals[idx]
                if h >= LABEL_MIN_PCT:
                    bottom_before = cumulative[idx]
                    if cat == "High" or cat == "HiAvg":
                        label_color = "white"
                    elif cat == "Avg" or cat == "LoAvg":
                        label_color = "#434343"
                    elif cat == "Low":
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
        labels=hf.NWEA_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(hf.NWEA_ORDER),
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    # Panel 2: Avg RIT by subject
    for i, subj in enumerate(subjects):
        score_df = score_dfs[i]
        pct_df = pct_dfs[i]
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
        # --- add n counts to x-axis labels ---
        # Use n from score_df if present, else from pct_df["N_total"], else no n
        if "n" in score_df.columns:
            n_map = score_df[["time_label", "n"]]
        elif "N_total" in pct_df.columns:
            n_map = (
                pct_df.groupby("time_label")["N_total"]
                .max()
                .reset_index()
                .rename(columns={"N_total": "n"})
            )
        else:
            n_map = pd.DataFrame(columns=["time_label", "n"])

        # Build label_map and x_labels aligned with score_df["time_label"]
        if not n_map.empty:
            # Ensure all time_labels in score_df are covered
            # n_map may be missing some time_labels; handle gracefully
            label_map = {
                row["time_label"]: f"{row['time_label']}\n(n = {int(row['n'])})"
                for _, row in n_map.iterrows()
                if not pd.isna(row["n"])
            }
            x_labels = [label_map.get(lbl, str(lbl)) for lbl in score_df["time_label"]]
        else:
            x_labels = score_df["time_label"].astype(str).tolist()

        ax2.set_ylabel("Avg RIT")
        ax2.set_xticks(rit_x)
        ax2.set_xticklabels(x_labels)
        ax2.set_title("Average RIT", fontsize=8, fontweight="bold", pad=10)
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
                    & (pct_df["achievementquintile"] == bucket_name)
                ]["pct"].sum()

            high_now = _pct_for_bucket("High", t_curr)
            high_prev = _pct_for_bucket("High", t_prev)
            high_delta = high_now - high_prev
            hi_delta = metrics["hi_delta"]
            lo_delta = metrics["lo_delta"]
            score_delta = metrics["score_delta"]
            title_line = "Change calculations from " + f"{t_prev} to {t_curr}:\n"
            line_high = rf"High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
            line_hiavg = rf"Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
            line_low = rf"Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
            line_rit = rf"Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
            insight_lines = [title_line, line_high, line_hiavg, line_low, line_rit]
        else:
            insight_lines = ["Not enough history for change insights"]
        ax3.text(
            0.5,
            0.5,
            "\n".join(insight_lines),
            fontsize=11,
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
        f"{title_label} • {group_name} • Fall Year-to-Year Trends",
        fontsize=20,
        fontweight="bold",
        y=1,
    )
    # --- Save chart with standardized scope_label ---
    charts_dir = CHARTS_DIR
    folder_name = "_district" if school_raw is None else scope_label.replace(" ", "_")
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    order_map = cfg.get("student_group_order", {})
    group_order_val = order_map.get(group_name, 99)
    safe_group = group_name.replace(" ", "_").replace("/", "_")
    out_name = f"{scope_label.replace(' ', '_')}_section2_{group_order_val:02d}_{safe_group}_fall_trends.png"
    out_path = out_dir / out_name

    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "nwea_boy_section2_student_group_dashboard",
            "section": 2,
            "scope": title_label,
            "scope_label": scope_label,
            "window_filter": window_filter,
            "group_name": group_name,
            "subjects": subject_titles,
            "pct_data": [
                {"subject": subject_titles[i], "data": _jsonable(pct_dfs[i])}
                for i in range(len(subject_titles))
            ],
            "score_data": [
                {"subject": subject_titles[i], "data": _jsonable(score_dfs[i])}
                for i in range(len(subject_titles))
            ],
            "metrics": _jsonable(metrics_list),
            "time_orders": _jsonable(time_orders),
        },
    )
    logger.info(f"[CHART] Saved Section 2: {out_path}")
    print(f"Saved Section 2: {out_path}")

    if preview:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------
# DRIVER — Faceted Student Group Dashboards (District and Site)
# ---------------------------------------------------------------------
student_groups_cfg = cfg.get("student_groups", {})
group_order = cfg.get("student_group_order", {})

# Optional: restrict student-group dashboards based on frontend selection.
# The runner passes selected groups as: NWEA_BOY_STUDENT_GROUPS="English Learners,Students with Disabilities"
_env_groups = os.getenv("NWEA_BOY_STUDENT_GROUPS")
_selected_groups = []
if _env_groups:
    _selected_groups = [g.strip() for g in str(_env_groups).split(",") if g.strip()]
    print(f"[FILTER] Student group selection from frontend: {_selected_groups}")

# Optional: restrict race/ethnicity dashboards based on frontend selection.
_env_race = os.getenv("NWEA_BOY_RACE")
_selected_races = []
if _env_race:
    _selected_races = [r.strip() for r in str(_env_race).split(",") if r.strip()]
    print(f"[FILTER] Race/Ethnicity selection from frontend: {_selected_races}")


def _get_ethnicity_col(df: pd.DataFrame) -> str | None:
    for c in ["ethnicityrace", "ethnicity_race", "race", "ethnicity"]:
        if c in df.columns:
            return c
    return None

# ---- District-level
scope_df = nwea_base
scope_label = district_label

_has_frontend_filters = bool(_selected_groups or _selected_races)

# Selected student groups
for group_name, group_def in sorted(
    student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99)
):
    if group_def.get("type") == "all":
        continue
    if _has_frontend_filters and group_name not in _selected_groups:
        continue
    plot_nwea_subject_dashboard_by_group(
        scope_df.copy(),
        subject_str=None,
        window_filter="Fall",
        group_name=group_name,
        group_def=group_def,
        figsize=(16, 9),
        school_raw=None,
        scope_label=scope_label,
    )

# Selected races
if _selected_races:
    eth_col = _get_ethnicity_col(scope_df)
    if not eth_col:
        logger.info(
            "[CHART] Section 2: race filters provided but no ethnicity/race column found; skipping race charts"
        )
    else:
        for race_name in _selected_races:
            mapped = (
                student_groups_cfg.get(race_name) if isinstance(student_groups_cfg, dict) else None
            )
            if isinstance(mapped, dict) and mapped.get("column") and mapped.get("in"):
                race_def = mapped
            else:
                race_def = {"column": eth_col, "in": [race_name]}
            plot_nwea_subject_dashboard_by_group(
                scope_df.copy(),
                subject_str=None,
                window_filter="Fall",
                group_name=race_name,
                group_def=race_def,
                figsize=(16, 9),
                school_raw=None,
                scope_label=scope_label,
            )

# ---- Site-level
if _include_school_charts():
    for raw_school in _iter_schools(nwea_base):
        scope_df = nwea_base[nwea_base["schoolname"] == raw_school].copy()
        scope_label = hf._safe_normalize_school_name(raw_school, cfg)

        # Selected student groups
        for group_name, group_def in sorted(
            student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99)
        ):
            if group_def.get("type") == "all":
                continue
            if _has_frontend_filters and group_name not in _selected_groups:
                continue
            plot_nwea_subject_dashboard_by_group(
                scope_df.copy(),
                subject_str=None,
                window_filter="Fall",
                group_name=group_name,
                group_def=group_def,
                figsize=(16, 9),
                school_raw=raw_school,
                scope_label=scope_label,
            )

        # Selected races
        if _selected_races:
            eth_col = _get_ethnicity_col(scope_df)
            if not eth_col:
                logger.info(
                    f"[CHART] Section 2: race filters provided but no ethnicity/race column found for school '{raw_school}'; skipping race charts"
                )
            else:
                for race_name in _selected_races:
                    mapped = (
                        student_groups_cfg.get(race_name)
                        if isinstance(student_groups_cfg, dict)
                        else None
                    )
                    if isinstance(mapped, dict) and mapped.get("column") and mapped.get("in"):
                        race_def = mapped
                    else:
                        race_def = {"column": eth_col, "in": [race_name]}
                    plot_nwea_subject_dashboard_by_group(
                        scope_df.copy(),
                        subject_str=None,
                        window_filter="Fall",
                        group_name=race_name,
                        group_def=race_def,
                        figsize=(16, 9),
                        school_raw=raw_school,
                        scope_label=scope_label,
                    )


# %% SECTION 3 - Overall + Cohort Trends
# Cohort Dashboards (current-year cohort across up to 4 years)
# Model and styling match the main dashboard section. X-axis shows cohort labels.
# Example labels: "Gr 3 • Fall 24-25", "Gr 4 • Fall 25-26", "Gr 5 • Fall 26-27"
# Rules:
#   - Window: Fall only
#   - Subject bucketing identical to _prep_nwea_for_charts
#   - For a given current grade G in latest available year Y that has G,
#     define the cohort as students in (year==Y & grade==G).
#   - Build up to four bars for that cohort by looking back 3 grades:
#         (grade, year) in [(G-3, Y-3), (G-2, Y-2), (G-1, Y-1), (G, Y)]
#     Skip any missing (grade, year) slices.
#   - District charts use all cohort students. Site charts filter to site’s students.
# ---------------------------------------------------------------------


def plot_nwea_blended_dashboard(
    df: pd.DataFrame,
    course_str: str,
    current_grade: int,
    window_filter: str = "Fall",
    cohort_year: int | None = None,
    figsize=(16, 9),
    school_raw: str | None = None,
    scope_label: str | None = None,
    preview: bool = False,
):
    """
    Produces a 2-column dashboard:
      - Left: grade-level panel (same as plot_nwea_subject_dashboard_by_grade)
      - Right: matched cohort panel (using _prep_nwea_matched_cohort_by_grade, but do NOT drop unmatched students)
    Both columns: 3 rows (stacked bar, avg RIT, insights).
    """
    logger.info(f"[CHART] plot_nwea_blended_dashboard: Starting | Scope: {scope_label} | Course: {course_str} | Grade: {current_grade} | Window: {window_filter} | Input rows: {len(df):,}")
    from matplotlib.gridspec import GridSpec

    # Normalize display + folder name
    school_display = (
        hf._safe_normalize_school_name(school_raw, cfg) if school_raw else None
    )
    folder_name = (
        "_district" if school_display is None else school_display.replace(" ", "_")
    )
    district_label = district_all_students_label if not school_display else school_display

    # --- Friendly title label ---
    course_str_for_title = course_str
    if str(course_str_for_title).strip().casefold() in ("math k-12", "mathematics", "math"):
        course_str_for_title = "Math"

    # -------------------------------
    # Left panel: grade-level
    # -------------------------------
    df_left = df.copy()
    logger.info(f"[FILTER] plot_nwea_blended_dashboard (left panel): Starting with {len(df_left):,} rows")
    df_left["grade"] = pd.to_numeric(df_left["grade"], errors="coerce")
    before_grade = len(df_left)
    df_left = df_left[df_left["grade"] == current_grade].copy()
    logger.info(f"[FILTER] After grade {current_grade} filter: {len(df_left):,} rows (removed {before_grade - len(df_left):,})")
    # Filter on course/subject (supports Reading/Math K-12 as well as optional MAP Growth subjects)
    before_course = len(df_left)
    df_left = filter_nwea_subject_rows(df_left, course_str)
    logger.info(
        f"[FILTER] After subject '{course_str}' filter: {len(df_left):,} rows (removed {before_course - len(df_left):,})"
    )
    pct_df_left, score_df_left, metrics_left, time_order_left = _prep_nwea_for_charts(
        df_left,
        subject_str=course_str,
        window_filter=window_filter,
    )
    if len(time_order_left) > 4:
        time_order_left = time_order_left[-4:]
        pct_df_left = pct_df_left[
            pct_df_left["time_label"].isin(time_order_left)
        ].copy()
        score_df_left = score_df_left[
            score_df_left["time_label"].isin(time_order_left)
        ].copy()

    # -------------------------------
    # Right panel: matched cohort
    # -------------------------------
    # Cohort (right panel)
    # Filter df for cohort panel by school_raw (needed for site-level charts)
    cohort_df = df.copy()
    if school_raw:
        cohort_df = cohort_df[cohort_df["schoolname"] == school_raw]

    def _prep_nwea_matched_cohort_by_grade(
        df, course_str, current_grade, window_filter, cohort_year
    ):
        """
        Returns pct_df, score_df, metrics, time_order for matched cohort panel.
        Uses df (already school-filtered) rather than _base.
        """
        base = df.copy()  # ✅ This now uses cohort_df
        base["year"] = pd.to_numeric(base.get("year"), errors="coerce")
        base["grade"] = pd.to_numeric(base.get("grade"), errors="coerce")
        if "teststartdate" in base.columns:
            base["teststartdate"] = pd.to_datetime(
                base["teststartdate"], errors="coerce"
            )
        else:
            base["teststartdate"] = pd.NaT

        # Anchor year
        if cohort_year is None:
            anchor_year = (
                int(base["year"].max()) if base["year"].notna().any() else None
            )
            if anchor_year is None:
                return pd.DataFrame(), pd.DataFrame(), {}, []
        else:
            anchor_year = int(cohort_year)

        cohort_grades = list(range(0, current_grade + 1))
        cohort_rows = []
        ordered_labels = []

        for grade in cohort_grades:
            offset = current_grade - grade
            year = anchor_year - offset
            if pd.isna(year):
                continue

            cohort_slice = base.copy()
            cohort_slice = cohort_slice[
                (
                    cohort_slice["testwindow"].astype(str).str.upper()
                    == window_filter.upper()
                )
                & (cohort_slice["grade"] == grade)
                & (cohort_slice["year"] == year)
            ].copy()

            # Subject filter (supports optional MAP Growth subjects)
            cohort_slice = filter_nwea_subject_rows(cohort_slice, course_str)
            if cohort_slice.empty:
                continue

            if "teststartdate" in cohort_slice.columns:
                cohort_slice["teststartdate"] = pd.to_datetime(
                    cohort_slice["teststartdate"], errors="coerce"
                )
            else:
                cohort_slice["teststartdate"] = pd.NaT

            cohort_slice.sort_values(
                ["uniqueidentifier", "teststartdate"], inplace=True
            )
            cohort_slice = cohort_slice.groupby(
                "uniqueidentifier", as_index=False
            ).tail(1)
            cohort_slice = cohort_slice[
                cohort_slice["achievementquintile"].notna()
            ].copy()
            if cohort_slice.empty:
                continue

            # Format time label as Fall YY-YY (e.g., Fall 25-26)
            year_str_prev = str(year - 1)[-2:]
            year_str_curr = str(year)[-2:]
            label_full = f"Gr {int(grade)} \u2022 Fall {year_str_prev}-{year_str_curr}"
            cohort_slice["cohort_label"] = label_full
            cohort_rows.append(cohort_slice)
            ordered_labels.append(label_full)

        if not cohort_rows:
            return pd.DataFrame(), pd.DataFrame(), {}, []

        cohort_df = pd.concat(cohort_rows, ignore_index=True)
        cohort_df["label"] = cohort_df["cohort_label"]

        def _extract_grade(label):
            try:
                return int(label.split()[1])
            except Exception:
                return 999

        cohort_df["cohort_label"] = pd.Categorical(
            cohort_df["cohort_label"],
            categories=sorted(cohort_df["cohort_label"].unique(), key=_extract_grade),
            ordered=True,
        )

        # Aggregate
        quint_counts = (
            cohort_df.groupby(["label", "achievementquintile"])
            .size()
            .rename("n")
            .reset_index()
        )
        total_counts = cohort_df.groupby("label").size().rename("N_total").reset_index()
        pct_df = quint_counts.merge(total_counts, on="label", how="left")
        pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]

        all_idx = pd.MultiIndex.from_product(
            [pct_df["label"].unique(), hf.NWEA_ORDER],
            names=["label", "achievementquintile"],
        )
        pct_df = (
            pct_df.set_index(["label", "achievementquintile"])
            .reindex(all_idx)
            .reset_index()
        )
        pct_df["pct"] = pct_df["pct"].fillna(0)
        pct_df["n"] = pct_df["n"].fillna(0)
        pct_df["N_total"] = pct_df.groupby("label")["N_total"].transform(
            lambda s: s.ffill().bfill()
        )

        score_df = (
            cohort_df[["label", "testritscore"]]
            .dropna(subset=["testritscore"])
            .groupby("label")["testritscore"]
            .mean()
            .rename("avg_score")
            .reset_index()
        )

        pct_df["label"] = pd.Categorical(
            pct_df["label"], categories=ordered_labels, ordered=True
        )
        pct_df = pct_df.sort_values("label").reset_index(drop=True)
        score_df["label"] = pd.Categorical(
            score_df["label"], categories=ordered_labels, ordered=True
        )
        score_df = score_df.sort_values("label").reset_index(drop=True)

        pct_df = pct_df.rename(columns={"label": "time_label"})
        score_df = score_df.rename(columns={"label": "time_label"})

        labels_order = ordered_labels
        last_two = labels_order[-2:] if len(labels_order) >= 2 else labels_order
        if len(last_two) == 2:
            t_prev, t_curr = last_two

            def pct_for(bucket_list, tlabel):
                return pct_df[
                    (pct_df["time_label"] == tlabel)
                    & (pct_df["achievementquintile"].isin(bucket_list))
                ]["pct"].sum()

            metrics = {
                "t_prev": t_prev,
                "t_curr": t_curr,
                "hi_now": pct_for(hf.NWEA_HIGH_GROUP, t_curr),
                "hi_delta": pct_for(hf.NWEA_HIGH_GROUP, t_curr)
                - pct_for(hf.NWEA_HIGH_GROUP, t_prev),
                "lo_now": pct_for(hf.NWEA_LOW_GROUP, t_curr),
                "lo_delta": pct_for(hf.NWEA_LOW_GROUP, t_curr)
                - pct_for(hf.NWEA_LOW_GROUP, t_prev),
                "score_now": float(
                    score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
                ),
                "score_delta": float(
                    score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
                )
                - float(
                    score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0]
                ),
            }
        else:
            metrics = {
                "t_prev": None,
                "t_curr": labels_order[-1] if labels_order else None,
                "hi_now": None,
                "hi_delta": None,
                "lo_now": None,
                "lo_delta": None,
                "score_now": None,
                "score_delta": None,
            }

        return pct_df, score_df, metrics, labels_order

    pct_df_right, score_df_right, metrics_right, time_order_right = (
        _prep_nwea_matched_cohort_by_grade(
            cohort_df,
            course_str=course_str,
            current_grade=current_grade,
            window_filter=window_filter,
            cohort_year=cohort_year,
        )
    )
    if len(time_order_right) > 4:
        time_order_right = time_order_right[-4:]
        pct_df_right = pct_df_right[
            pct_df_right["time_label"].isin(time_order_right)
        ].copy()
        score_df_right = score_df_right[
            score_df_right["time_label"].isin(time_order_right)
        ].copy()

    # --- Ensure cohort_df time_label is sorted in ascending cohort order numerically by grade (preserving grade 0) ---
    cohort_df = pct_df_right.copy()
    # Check for required columns and non-empty
    if cohort_df.empty or not all(
        col in cohort_df.columns
        for col in ["time_label", "achievementquintile", "pct", "n", "N_total"]
    ):
        print(
            f"[blended] required columns not found in cohort_df: {list(cohort_df.columns)}"
        )
        return
    # Use all relevant grades in the cohort from the data, including Grade 0
    df_cohort = df.copy()
    df_cohort["student_grade"] = pd.to_numeric(df_cohort["grade"], errors="coerce")
    cohort_grades = sorted(df_cohort["student_grade"].dropna().unique())

    # Sort time_label in ascending cohort order numerically by grade
    def _extract_grade(label):
        try:
            return int(label.split()[1])
        except Exception:
            return 999

    cohort_df["time_label"] = pd.Categorical(
        cohort_df["time_label"],
        categories=sorted(cohort_df["time_label"].unique(), key=_extract_grade),
        ordered=True,
    )
    # Deduplicate cohort_df before pivot to prevent ValueError
    cohort_df = cohort_df.groupby(
        ["time_label", "achievementquintile"], as_index=False
    ).agg({"pct": "mean", "n": "sum", "N_total": "max"})

    # If either panel is empty, bail
    if (
        pct_df_left.empty
        or score_df_left.empty
        or pct_df_right.empty
        or score_df_right.empty
    ):
        print(
            f"[blended] no data for Grade {current_grade} {course_str} ({school_display or 'district'})"
        )
        return

    # 3x2 grid
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    # Helper: stacked bar panel
    def draw_stacked_bar(ax, stack_df, pct_df, time_labels):
        x = np.arange(len(stack_df))
        cumulative = np.zeros(len(stack_df))
        for cat in hf.NWEA_ORDER:
            band_vals = stack_df[cat].to_numpy()
            bars = ax.bar(
                x,
                band_vals,
                bottom=cumulative,
                color=hf.NWEA_COLORS[cat],
                edgecolor="white",
                linewidth=1.2,
            )
            for idx, rect in enumerate(bars):
                h = band_vals[idx]
                if h >= LABEL_MIN_PCT:
                    bottom_before = cumulative[idx]
                    if cat == "High" or cat == "HiAvg":
                        label_color = "white"
                    elif cat == "Avg" or cat == "LoAvg":
                        label_color = "#434343"
                    elif cat == "Low":
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
            cumulative += band_vals
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of Students")
        ax.set_xticks(x)
        ax.set_xticklabels(stack_df.index.tolist())
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel 1: left stacked bar
    stack_df_left = (
        pct_df_left.pivot(
            index="time_label", columns="achievementquintile", values="pct"
        )
        .reindex(columns=hf.NWEA_ORDER)
        .fillna(0)
    )
    ax1 = fig.add_subplot(gs[0, 0])
    draw_stacked_bar(ax1, stack_df_left, pct_df_left, time_order_left)
    ax1.set_title("Overall Trends", fontsize=14, fontweight="bold", pad=30)
    legend_handles = [
        Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q)
        for q in hf.NWEA_ORDER
    ]

    # Centered legend across both charts
    fig.legend(
        handles=legend_handles,
        labels=hf.NWEA_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(hf.NWEA_ORDER),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # Panel 1: right stacked bar
    stack_df_right = (
        cohort_df.pivot(index="time_label", columns="achievementquintile", values="pct")
        .reindex(columns=hf.NWEA_ORDER)
        .fillna(0)
    )
    # Define x_labels_cohort before drawing stacked bar
    x_labels_cohort = stack_df_right.index.tolist()
    ax2 = fig.add_subplot(gs[0, 1])
    draw_stacked_bar(ax2, stack_df_right, cohort_df, x_labels_cohort)
    ax2.set_title("Cohort Trends", fontsize=14, fontweight="bold", pad=30)
    # Only legend on left

    # Panel 2: Avg RIT bars (left)
    ax3 = fig.add_subplot(gs[1, 0])
    rit_x = np.arange(len(score_df_left["time_label"]))
    rit_vals = score_df_left["avg_score"].to_numpy()
    rit_bars = ax3.bar(
        rit_x,
        rit_vals,
        color=hf.default_quintile_colors[4],
        edgecolor="white",
        linewidth=1.2,
    )
    for rect, v in zip(rit_bars, rit_vals):
        ax3.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#434343",
        )
    # --- Add n counts below x-axis labels for left panel ---
    # Use score_df_left and pct_df_left to build n_map
    if "n" in score_df_left.columns:
        n_map_left = score_df_left[["time_label", "n"]]
    elif "N_total" in pct_df_left.columns:
        n_map_left = (
            pct_df_left.groupby("time_label")["N_total"]
            .max()
            .reset_index()
            .rename(columns={"N_total": "n"})
        )
    else:
        n_map_left = pd.DataFrame(columns=["time_label", "n"])
    # Build label_map and x_labels for left panel
    if not n_map_left.empty:
        label_map_left = {
            row["time_label"]: f"{row['time_label']}\n(n = {int(row['n'])})"
            for _, row in n_map_left.iterrows()
            if not pd.isna(row["n"])
        }
        x_labels_left = [
            label_map_left.get(lbl, str(lbl)) for lbl in score_df_left["time_label"]
        ]
    else:
        x_labels_left = score_df_left["time_label"].astype(str).tolist()
    ax3.set_ylabel("Avg RIT", labelpad=10)
    ax3.set_xticks(rit_x)
    # Lower the label position slightly for alignment (y=-0.09 is used in other sections)
    ax3.set_xticklabels(x_labels_left, ha="center")
    for label in ax3.get_xticklabels():
        label.set_y(-0.09)
    ax3.set_title("Average RIT", fontsize=8, fontweight="bold", pad=10)
    ax3.grid(axis="y", alpha=0.2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Panel 2: Avg RIT bars (right)
    ax4 = fig.add_subplot(gs[1, 1])
    rit_xr = np.arange(len(score_df_right["time_label"]))
    rit_valsr = score_df_right["avg_score"].to_numpy()
    rit_barsr = ax4.bar(
        rit_xr,
        rit_valsr,
        color=hf.default_quintile_colors[4],
        edgecolor="white",
        linewidth=1.2,
    )
    for rect, v in zip(rit_barsr, rit_valsr):
        ax4.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#434343",
        )
    # --- Add n counts below x-axis labels for right panel (cohort) ---
    # Use cohort_df to get N_total for each time_label
    if "N_total" in cohort_df.columns:
        n_map_right = (
            cohort_df.groupby("time_label", observed=True)["N_total"]
            .max()
            .reset_index()
            .rename(columns={"N_total": "n"})
        )
    else:
        n_map_right = pd.DataFrame(columns=["time_label", "n"])
    if not n_map_right.empty:
        label_map_right = {
            row["time_label"]: f"{row['time_label']}\n(n = {int(row['n'])})"
            for _, row in n_map_right.iterrows()
            if not pd.isna(row["n"])
        }
        x_labels_right = [
            label_map_right.get(lbl, str(lbl)) for lbl in score_df_right["time_label"]
        ]
    else:
        x_labels_right = score_df_right["time_label"].astype(str).tolist()
    ax4.set_ylabel("Avg RIT", labelpad=10)
    ax4.set_xticks(rit_xr)
    ax4.set_xticklabels(x_labels_right, ha="center")
    for label in ax4.get_xticklabels():
        label.set_y(-0.09)
    ax4.set_title("Average RIT", fontsize=8, fontweight="bold", pad=10)
    ax4.grid(axis="y", alpha=0.2)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Panel 3: Insights (left)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")
    if metrics_left["t_prev"]:
        t_prev = metrics_left["t_prev"]
        t_curr = metrics_left["t_curr"]

        def _pct_for_bucket_left(bucket_name, tlabel):
            return pct_df_left[
                (pct_df_left["time_label"] == tlabel)
                & (pct_df_left["achievementquintile"] == bucket_name)
            ]["pct"].sum()

        high_now = _pct_for_bucket_left("High", t_curr)
        high_prev = _pct_for_bucket_left("High", t_prev)
        high_delta = high_now - high_prev
        hi_delta = metrics_left["hi_delta"]
        lo_delta = metrics_left["lo_delta"]
        score_delta = metrics_left["score_delta"]
        title_line = "Change calculations from previous to current year\n"
        line_high = rf"High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
        line_hiavg = rf"Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
        line_low = rf"Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
        line_rit = rf"Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
        insight_lines = [title_line, line_high, line_hiavg, line_low, line_rit]
    else:
        insight_lines = ["Not enough history for change insights"]
    ax5.text(
        0.5,
        0.5,
        "\n".join(insight_lines),
        fontsize=9,
        fontweight="normal",
        color="#434343",
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

    # Panel 3: Insights (right)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    if metrics_right.get("t_prev"):
        t_prev = metrics_right["t_prev"]
        t_curr = metrics_right["t_curr"]

        def _pct_for_bucket_right(bucket_name, tlabel):
            return pct_df_right[
                (pct_df_right["time_label"] == tlabel)
                & (pct_df_right["achievementquintile"] == bucket_name)
            ]["pct"].sum()

        high_now = _pct_for_bucket_right("High", t_curr)
        high_prev = _pct_for_bucket_right("High", t_prev)
        high_delta = high_now - high_prev
        hi_delta = metrics_right["hi_delta"]
        lo_delta = metrics_right["lo_delta"]
        score_delta = metrics_right["score_delta"]
        title_line = "Change calculations from previous to current year\n"
        line_high = rf"High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
        line_hiavg = rf"Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
        line_low = rf"Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
        line_rit = rf"Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
        insight_lines = [title_line, line_high, line_hiavg, line_low, line_rit]
    else:
        insight_lines = ["Not enough history for change insights"]
    ax6.text(
        0.5,
        0.5,
        "\n".join(insight_lines),
        fontsize=9,
        fontweight="normal",
        color="#434343",
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

    # Main title
    fig.suptitle(
        f"{district_label} • Grade {int(current_grade)} • {course_str_for_title}",
        fontsize=20,
        fontweight="bold",
        y=1,
    )
    #    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # Save
    charts_dir = CHARTS_DIR
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scope = scope_label or (
        district_label
        if school_raw is None
        else hf._safe_normalize_school_name(school_raw, cfg)
    )
    out_name = (
        f"{scope.replace(' ', '_')}_section3_grade{int(current_grade)}_"
        f"{course_str.lower().replace(' ', '_')}_fall_trends.png"
    )
    out_path = out_dir / out_name
    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "nwea_boy_section3_blended_dashboard",
            "section": 3,
            "scope": scope,
            "scope_label": scope_label,
            "window_filter": window_filter,
            "grade": int(current_grade),
            "course_str": course_str,
            "subjects": [course_str_for_title],
            "pct_data": [
                {"subject": "overall", "data": _jsonable(pct_df_left)},
                {"subject": "cohort", "data": _jsonable(pct_df_right)},
            ],
            "score_data": [
                {"subject": "overall", "data": _jsonable(score_df_left)},
                {"subject": "cohort", "data": _jsonable(score_df_right)},
            ],
            "metrics": {"overall": _jsonable(metrics_left), "cohort": _jsonable(metrics_right)},
            "time_orders": {"overall": _jsonable(time_order_left), "cohort": _jsonable(time_order_right)},
        },
    )
    logger.info(f"[CHART] Saved: {out_path}")
    if preview:
        plt.show()
    print(f"Saved: {out_path}")


# ---- Combined DRIVER for Section 3 ----
_base = nwea_base
_base["year"] = pd.to_numeric(_base["year"], errors="coerce")
_base["grade"] = pd.to_numeric(_base["grade"], errors="coerce")


def _run_scope(scope_df, scope_label, school_raw):
    if scope_df["year"].notna().any():
        anchor_year = int(scope_df["year"].max())
    else:
        anchor_year = None
    # Optional: restrict grade-level dashboards based on frontend selection.
    # The runner passes selected grades as: NWEA_BOY_GRADES="3,4,5"
    _env_grades = os.getenv("NWEA_BOY_GRADES")
    _selected_grades = None
    if _env_grades:
        try:
            _selected_grades = {int(x.strip()) for x in str(_env_grades).split(",") if x.strip()}
            if _selected_grades:
                print(f"[FILTER] Grade selection from frontend: {sorted(_selected_grades)}")
        except Exception:
            _selected_grades = None
    subjects_to_generate = _requested_subjects(["Reading", "Mathematics"])
    for g in sorted(scope_df["grade"].dropna().unique()):
        try:
            g_int = int(g)
        except Exception:
            continue
        if _selected_grades and g_int not in _selected_grades:
            continue
        for subject_str in subjects_to_generate:
            # Quick empty check per subject to avoid expensive chart work
            if filter_nwea_subject_rows(scope_df, subject_str).empty:
                logger.info(
                    f"[CHART] Section 3: skipping subject '{subject_str}' for Grade {g_int} (no matching rows)"
                )
                continue
            plot_nwea_blended_dashboard(
                scope_df.copy(),
                course_str=subject_str,
                current_grade=g_int,
                window_filter="Fall",
                cohort_year=anchor_year,
                figsize=(16, 9),
                school_raw=school_raw,
                preview=False,
                scope_label=scope_label,
            )


# ---- Run for district ----
_run_scope(_base.copy(), district_label, None)

# %%---- Run for schools ----
if _include_school_charts():
    for raw in _iter_schools(_base):
        site_df = _base[_base["schoolname"] == raw].copy()
        scope_label = hf._safe_normalize_school_name(raw, cfg)
        _run_scope(site_df, scope_label, raw)


# %% SECTION 4 — Overall Growth Trends by Site (CGP + CGI)
# ---------------------------------------------------------------------
# Historical Conditional Growth Percentile (CGP) and Growth Index (CGI)
# Uses `falltofallconditionalgrowthpercentile` and `falltofallconditionalgrowthindex`
# Creates dual-panel dashboards for district + each site.
# ---------------------------------------------------------------------


def _prep_cgp_trend(df: pd.DataFrame, subject_str: str) -> pd.DataFrame:
    """
    Return tidy frame with columns:
        scope_label, time_label, median_cgp, mean_cgi
    Only Fall window. Subject filter matches dashboard logic.
    """

    d = df.copy()
    d = d[d["testwindow"].astype(str).str.upper() == "FALL"].copy()
    d = filter_nwea_subject_rows(d, subject_str)

    if "falltofallconditionalgrowthpercentile" not in d.columns:
        logger.info(
            f"[CHART] Section 4: skipping subject '{subject_str}' for CGP trend "
            f"(missing falltofallconditionalgrowthpercentile column)"
        )
        return pd.DataFrame(
            columns=["scope_label", "time_label", "median_cgp", "mean_cgi"]
        )

    before_nonnull = len(d)
    d = d[d["falltofallconditionalgrowthpercentile"].notna()].copy()
    if d.empty:
        logger.info(
            f"[CHART] Section 4: skipping subject '{subject_str}' for CGP trend "
            f"(0/{before_nonnull:,} rows have falltofallconditionalgrowthpercentile)"
        )
        return pd.DataFrame(
            columns=["scope_label", "time_label", "median_cgp", "mean_cgi"]
        )

    def _short_year(y):
        ys = str(y)
        if "-" in ys:
            a, b = ys.split("-", 1)
            return f"{a[-2:]}-{b[-2:]}"
        yi = int(float(ys))
        return f"{str(yi-1)[-2:]}-{str(yi)[-2:]}"

    d["year_short"] = d["year"].apply(_short_year)
    d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["year_short"]
    d["subject"] = subject_str

    d["site_display"] = d["schoolname"].apply(
        lambda x: hf._safe_normalize_school_name(x, cfg)
    )
    dist_rows = d.copy()
    dist_rows["site_display"] = district_all_students_label

    both = pd.concat([d, dist_rows], ignore_index=True)
    has_cgi = "falltofallconditionalgrowthindex" in both.columns
    grp_cols = ["site_display", "time_label"]

    if has_cgi:
        out = (
            both.groupby(grp_cols, dropna=False)
            .agg(
                median_cgp=("falltofallconditionalgrowthpercentile", "median"),
                mean_cgi=("falltofallconditionalgrowthindex", "mean"),
            )
            .reset_index()
        )
    else:
        out = (
            both.groupby(grp_cols, dropna=False)
            .agg(median_cgp=("falltofallconditionalgrowthpercentile", "median"))
            .reset_index()
        )
        out["mean_cgi"] = np.nan

    time_order = sorted(out["time_label"].astype(str).unique().tolist())
    out["time_label"] = pd.Categorical(
        out["time_label"], categories=time_order, ordered=True
    )
    out.sort_values(["site_display", "time_label"], inplace=True)

    keep_list = []
    for scope_val, chunk in out.groupby("site_display", dropna=False):
        ordered = chunk["time_label"].cat.categories.tolist()
        present = chunk["time_label"].astype(str).unique().tolist()
        recent = set([t for t in ordered if t in present][-4:])
        keep_list.append(chunk[chunk["time_label"].astype(str).isin(recent)])
    out_recent = pd.concat(keep_list, ignore_index=True)
    out_recent["subject"] = subject_str
    return out_recent.rename(columns={"site_display": "scope_label"})


def _plot_cgp_trend(df, subject_str, scope_label, ax=None):
    """
    Bars = median CGP, line = mean CGI with blended transform.
    """
    sub = df[df["scope_label"] == scope_label].copy()
    if sub.empty:
        return

    sub["time_label"] = pd.Categorical(
        sub["time_label"],
        categories=sorted(sub["time_label"].astype(str).unique()),
        ordered=True,
    )
    sub.sort_values("time_label", inplace=True)
    x_vals = np.arange(len(sub))
    y_cgp = sub["median_cgp"].to_numpy(float)
    y_cgi = sub["mean_cgi"].to_numpy(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))

    # shaded band 42–58, bars for CGP
    ax.axhspan(0, 20, facecolor="#808080", alpha=0.5, zorder=0)
    ax.axhspan(20, 40, facecolor="#c5c5c5", alpha=0.5, zorder=0)
    ax.axhspan(40, 60, facecolor="#78daf4", alpha=0.5, zorder=0)
    ax.axhspan(60, 80, facecolor="#00baeb", alpha=0.5, zorder=0)
    ax.axhspan(80, 100, facecolor="#0381a2", alpha=0.5, zorder=0)
    for y in [42, 50, 58]:
        ax.axhline(y, linestyle="--", color="#6B7280", linewidth=1.2)

    bars = ax.bar(
        x_vals, y_cgp, color="#0381a2", edgecolor="white", linewidth=1.2, zorder=3
    )
    for rect, yv in zip(bars, y_cgp):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() / 2,
            f"{yv:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    ax.set_ylabel("Median Fall→Fall CGP")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(sub["time_label"].astype(str).tolist())
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # secondary axis (CGI line)
    ax2 = ax.twinx()
    ax2.set_ylim(-2.5, 2.5)
    ax2.patch.set_alpha(0)
    blend = mtransforms.BlendedGenericTransform(ax.transData, ax2.transData)
    x0, x1 = ax.get_xlim()

    # yellow CGI band
    band = mpl.patches.Rectangle(
        (x0, -0.2),
        x1 - x0,
        0.4,
        transform=blend,
        facecolor="#facc15",
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(band)
    for yb in [-0.2, 0.2]:
        ax.add_line(
            mlines.Line2D(
                [x0, x1],
                [yb, yb],
                transform=blend,
                linestyle="--",
                color="#eab308",
                linewidth=1.2,
            )
        )

    # CGI line and labels
    cgi_line = mlines.Line2D(
        x_vals,
        y_cgi,
        transform=blend,
        marker="o",
        linewidth=2,
        markersize=6,
        color="#ffa800",
        zorder=3,
    )
    ax.add_line(cgi_line)
    for xv, yv in zip(x_vals, y_cgi):
        if pd.notna(yv):
            ax.text(
                xv,
                yv + (0.12 if yv >= 0 else -0.12),
                f"{yv:.2f}",
                transform=blend,
                ha="center",
                va="bottom" if yv >= 0 else "top",
                fontsize=8,
                fontweight="bold",
                color="#ffa800",
            )

    ax2.set_ylabel("Avg Fall→Fall CGI")
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax.set_title(f"{subject_str}", fontweight="bold", fontsize=14, pad=10)


# ---------------------------------------------------------------------
# Standardized Save + Driver Logic
# ---------------------------------------------------------------------


def _save_cgp_chart(
    fig, scope_label, section_num=4, suffix="cgp_fall_to_fall_dualpanel"
):
    charts_dir = CHARTS_DIR
    folder_name = (
        "_district"
        if scope_label == district_all_students_label
        else scope_label.replace(" ", "_")
    )
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section{section_num}_{suffix}.png"
    out_path = out_dir / out_name
    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    logger.info(f"[CHART] Saved: {out_path}")
    print(f"Saved: {out_path}")


def _run_cgp_dual_trend(scope_df, scope_label):
    # This legacy chart is a 2-column layout (Reading + Math). If frontend filters
    # subjects and does not include both Reading and Math, skip to avoid a broken layout.
    if _selected_subjects:
        subj_join = " | ".join(str(s).casefold() for s in _selected_subjects)
        has_read = any(k in subj_join for k in ["reading", "ela", "language arts"])
        has_math = "math" in subj_join
        if not (has_read and has_math):
            logger.info(
                f"[CHART] Section 4 CGP: skipping for {scope_label} (frontend did not request both Reading and Math)"
            )
            return

    subjects_for_cgp = ["Reading", "Mathematics"]
    cgp_trend = pd.concat(
        [_prep_cgp_trend(scope_df, subj) for subj in subjects_for_cgp],
        ignore_index=True,
    )
    if cgp_trend.empty:
        logger.info(f"[CHART] Section 4 CGP: no CGP data for {scope_label} (nothing to plot)")
        return

    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.suptitle(
        f"{scope_label} • Fall→Fall Growth (All Students)",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )

    axes = []
    n_labels_axes = []
    for i, subject_str in enumerate(subjects_for_cgp):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        sub_df = cgp_trend[
            (cgp_trend["scope_label"] == scope_label)
            & (cgp_trend["subject"] == subject_str)
        ]
        if not sub_df.empty:
            _plot_cgp_trend(sub_df, subject_str, scope_label, ax=ax)
        # --- Add n-count labels under x-axis ticks for this facet ---
        # After bars and xtick labels are set
        # Try to get "n" count for each time_label
        # (Section 4: after both bar plots are created, add n-labels below x-ticks)
        # Use sub_df for this facet
        # If "n" in sub_df, use it; else if "N_total" in sub_df, use it; else skip
        # But in Section 4, columns are ["scope_label", "time_label", "median_cgp", "mean_cgi", "subject"]
        # So n-count is not present by default -- need to get from source
        # Here, we will try to get n from the underlying scope_df
        # Compute n for each time_label for this subject and scope
        d = scope_df.copy()
        d = d[d["testwindow"].astype(str).str.upper() == "FALL"].copy()
        d = filter_nwea_subject_rows(d, subject_str)

        # Use same year_short, time_label logic as _prep_cgp_trend
        def _short_year(y):
            ys = str(y)
            if "-" in ys:
                a, b = ys.split("-", 1)
                return f"{a[-2:]}-{b[-2:]}"
            yi = int(float(ys))
            return f"{str(yi-1)[-2:]}-{str(yi)[-2:]}"

        d["year_short"] = d["year"].apply(_short_year)
        d["time_label"] = (
            d["testwindow"].astype(str).str.title() + " " + d["year_short"]
        )
        # For district facet, need to match site_display to scope_label
        d["site_display"] = d["schoolname"].apply(
            lambda x: hf._safe_normalize_school_name(x, cfg)
        )
        if scope_label == district_all_students_label:
            d["site_display"] = district_all_students_label
        # Only keep rows matching this scope_label
        d = d[d["site_display"] == scope_label]
        # Now, for each time_label, get n
        n_map = (
            d.groupby("time_label")["uniqueidentifier"]
            .nunique()
            .reset_index()
            .rename(columns={"uniqueidentifier": "n"})
        )
        # Map time_label to n
        n_map_dict = dict(zip(n_map["time_label"], n_map["n"]))
        # Get the current x-tick labels (should match sub_df["time_label"])
        ticklabels = [str(lbl) for lbl in sub_df["time_label"]]
        # Build new labels with n-count
        labels_with_n = [
            f"{lbl}\n(n = {int(n_map_dict.get(lbl, 0))})" for lbl in ticklabels
        ]
        ax.set_xticklabels(labels_with_n)
        ax.tick_params(axis="x", pad=10)
        n_labels_axes.append(ax)

    # Add unified legend above both panels

    legend_handles = [
        Patch(facecolor="#0381a2", edgecolor="white", label="Median CGP"),
        Line2D(
            [0],
            [0],
            color="#ffa800",
            marker="o",
            linewidth=2,
            markersize=6,
            label="Mean CGI",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        labels=["Median CGP", "Mean CGI"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        handlelength=2,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    _save_cgp_chart(fig, scope_label)


# ---------------------------------------------------------------------
# DRIVER — District + School CGP Dual-Panel Dashboards
# ---------------------------------------------------------------------

_run_cgp_dual_trend(nwea_base, district_label)

if _include_school_charts():
    for raw_school in _iter_schools(nwea_base):
        scope_df = nwea_base[nwea_base["schoolname"] == raw_school].copy()
        scope_label = hf._safe_normalize_school_name(raw_school, cfg)
        _run_cgp_dual_trend(scope_df, scope_label)


# %% SECTION 5 — CGP/CGI Growth: Grade Trend + Backward Cohort (Unmatched)
# ---------------------------------------------------------------------
# Left facet: overall grade-level fall→fall growth across 4 years
# Right facet: same grade, same time span, but backward cohort (unmatched)
#   → Year = anchor_year - offset
#   → Grade = anchor_grade - offset
#   → Dedupe by test date, filter FALL only
#   → Median CGP + Mean CGI
# Replicates verified Sect‑2 cohort logic (no student matching)
# ---------------------------------------------------------------------
def _prep_cgp_by_grade(df, subject, grade):
    d = df
    d = d[d["testwindow"].str.upper() == "FALL"]
    d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
    d = d[d["grade"] == grade]

    subj = subject.lower()
    if "math" in subj:
        d = d[d["course"].str.contains("math", case=False, na=False)]
    else:
        d = d[d["course"].str.contains("read", case=False, na=False)]

    # Clean dates and deduplicate to most recent Fall test per student
    d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    d = d.dropna(subset=["teststartdate"])
    d = d.sort_values("teststartdate").drop_duplicates("uniqueidentifier", keep="last")

    # Only keep students with both values present
    before_growth = len(d)
    d = d.dropna(
        subset=[
            "falltofallconditionalgrowthpercentile",
            "falltofallconditionalgrowthindex",
        ]
    )

    if d.empty:
        logger.info(
            f"[CHART] Section 5: no CGP/CGI rows for Grade {grade} • {subject} "
            f"(0/{before_growth:,} rows have both falltofallconditionalgrowthpercentile & falltofallconditionalgrowthindex)"
        )
        return pd.DataFrame(columns=["time_label", "median_cgp", "mean_cgi"])

    # --- Use year_short logic from Section 4 ---
    def _short_year(y):
        ys = str(y)
        if "-" in ys:
            a, b = ys.split("-", 1)
            return f"{a[-2:]}-{b[-2:]}"
        yi = int(float(ys))
        return f"{str(yi-1)[-2:]}-{str(yi)[-2:]}"

    if "year_short" not in d.columns and "year" in d.columns:
        d["year_short"] = d["year"].apply(_short_year)

    # New label format includes grade for alignment with cohort panel
    d["time_label"] = (
        "Gr " + d["grade"].astype(int).astype(str) + " \u2022 Fall " + d["year_short"]
    )

    out = (
        d.groupby("time_label")
        .agg(
            median_cgp=("falltofallconditionalgrowthpercentile", "median"),
            mean_cgi=("falltofallconditionalgrowthindex", "mean"),
        )
        .reset_index()
    )

    # --- Add year_short column to out for sorting ---
    if "year_short" not in out.columns and "time_label" in out.columns:
        # Extract year_short from the time_label string
        # "Gr {grade} • Fall {yy-yy}"
        def _extract_year_short(label):
            try:
                return label.split("Fall")[-1].strip()
            except Exception:
                return ""

        out["year_short"] = out["time_label"].apply(_extract_year_short)

    # Sort by underlying school year rather than alphabetically
    out = out.sort_values("year_short").tail(4)
    out["time_label"] = pd.Categorical(
        out["time_label"], categories=out["time_label"], ordered=True
    )
    return out


def _plot_cgp_dual_facet(
    overall_df, cohort_df, grade, subject_str, scope_label, preview=False
):
    """
    Plots dual-facet CGP/CGI chart for Section 5.
    scope_label: pretty display name for district or school (for title and save).
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.subplots_adjust(wspace=0.28)

    def draw_panel(df, ax, title):
        if df.empty:
            return
        df = df.copy().sort_values("time_label")
        x_vals = np.arange(len(df))
        y_cgp = df["median_cgp"].to_numpy(dtype=float)
        y_cgi = df["mean_cgi"].to_numpy(dtype=float)

        ax.axhspan(0, 20, facecolor="#808080", alpha=0.5, zorder=0)
        ax.axhspan(20, 40, facecolor="#c5c5c5", alpha=0.5, zorder=0)
        ax.axhspan(40, 60, facecolor="#78daf4", alpha=0.5, zorder=0)
        ax.axhspan(60, 80, facecolor="#00baeb", alpha=0.5, zorder=0)
        ax.axhspan(80, 100, facecolor="#0381a2", alpha=0.5, zorder=0)
        for yref in [42, 50, 58]:
            ax.axhline(yref, linestyle="--", color="#6B7280", linewidth=1.2, zorder=0)

        bars = ax.bar(
            x_vals,
            y_cgp,
            color="#0381a2",
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        for rect, yv in zip(bars, y_cgp):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() / 2,
                f"{yv:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        # --- Add n-count to x-axis labels below each bar ---
        # Try to get n from underlying data for each time_label
        # We'll use the global nwea_base (district) or filter by school if needed
        # Use the time_label values in correct order
        labels_with_n = df["time_label"].astype(str).tolist()
        n_map = None
        # Try to get n for each time_label
        # Use global nwea_base; filter to FALL, subject, and grade for left; for right, filter to FALL, subject, grade/yr
        try:
            # Try to infer grade from time_label if possible
            import re

            # Figure out subject and grade from df/time_label
            # Example: "Gr 4 • Fall 23-24"
            time_labels = df["time_label"].astype(str).tolist()
            # We'll try to extract grade and year_short from each label
            n_dict = {}
            for lbl in time_labels:
                m = re.match(r"Gr (\d+) *• *Fall (\d{2}-\d{2})", lbl)
                if m:
                    gr = int(m.group(1))
                    yy = m.group(2)
                    # find rows in nwea_base with grade==gr, year_short==yy, FALL, and subject
                    d = nwea_base.copy()
                    d = d[d["testwindow"].str.upper() == "FALL"]
                    d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
                    d = d[d["grade"] == gr]

                    # Build year_short for each row
                    def _short_year(y):
                        ys = str(y)
                        if "-" in ys:
                            a, b = ys.split("-", 1)
                            return f"{a[-2:]}-{b[-2:]}"
                        yi = int(float(ys))
                        return f"{str(yi-1)[-2:]}-{str(yi)[-2:]}"

                    if "year_short" not in d.columns and "year" in d.columns:
                        d["year_short"] = d["year"].apply(_short_year)
                    d = d[d["year_short"] == yy]
                    # Subject filter
                    subj = (
                        subject_str
                        if "subject_str" in locals()
                        else subject if "subject" in locals() else ""
                    )
                    subj_lower = subj.lower()
                    if "math" in subj_lower:
                        d = d[d["course"].str.contains("math", case=False, na=False)]
                    else:
                        d = d[d["course"].str.contains("read", case=False, na=False)]
                    # Only count students with CGP/CGI present
                    d = d.dropna(
                        subset=[
                            "falltofallconditionalgrowthpercentile",
                            "falltofallconditionalgrowthindex",
                        ]
                    )
                    n_dict[lbl] = d["uniqueidentifier"].nunique()
                else:
                    n_dict[lbl] = 0
            labels_with_n = [
                f"{lbl}\n(n = {int(n_dict.get(lbl,0))})" for lbl in time_labels
            ]
        except Exception:
            # fallback: just use labels
            labels_with_n = df["time_label"].astype(str).tolist()

        ax.set_ylabel("Median Fall→Fall CGP")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels_with_n, ha="center")
        ax.tick_params(axis="x", pad=10)
        ax.set_ylim(0, 100)

        ax2 = ax.twinx()
        ax2.set_ylim(-2.5, 2.5)
        ax2.set_ylabel("Avg Fall→Fall CGI")
        ax2.set_yticks([-2, -1, 0, 1, 2])
        ax2.set_zorder(ax.get_zorder() - 1)
        ax2.patch.set_alpha(0)

        ax.set_xlim(-0.5, len(x_vals) - 0.5)
        x0, x1 = ax.get_xlim()
        blend = mtransforms.BlendedGenericTransform(ax.transData, ax2.transData)

        band = mpl.patches.Rectangle(
            (x0, -0.2),
            x1 - x0,
            0.4,
            transform=blend,
            facecolor="#facc15",
            alpha=0.35,
            zorder=1.5,
        )
        ax.add_patch(band)
        for yref in [-0.2, 0.2]:
            ax.add_line(
                mlines.Line2D(
                    [x0, x1],
                    [yref, yref],
                    transform=blend,
                    linestyle="--",
                    color="#eab308",
                    linewidth=1.2,
                    zorder=1.6,
                )
            )

        cgi_line = mlines.Line2D(
            x_vals,
            y_cgi,
            transform=blend,
            marker="o",
            linewidth=2,
            markersize=6,
            color="#ffa800",
            zorder=3,
        )
        ax.add_line(cgi_line)

        for xv, yv in zip(x_vals, y_cgi):
            if pd.isna(yv):
                continue
            ax.text(
                xv,
                yv + (0.12 if yv >= 0 else -0.12),
                f"{yv:.2f}",
                transform=blend,
                ha="center",
                va="bottom" if yv >= 0 else "top",
                fontsize=8,
                fontweight="bold",
                color="#ffa800",
                zorder=3.1,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)

    draw_panel(overall_df, axs[0], "Overall Growth Trends")
    draw_panel(cohort_df, axs[1], "Cohort Growth Trends")

    # Add unified legend above both facets (teal patch for Median CGP, orange line for Mean CGI)
    legend_handles = [
        Patch(facecolor="#0381a2", edgecolor="white", label="Median CGP"),
        mlines.Line2D(
            [0],
            [0],
            color="#ffa800",
            marker="o",
            linewidth=2,
            markersize=6,
            label="Mean CGI",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        labels=["Median CGP", "Mean CGI"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        frameon=False,
        handlelength=2,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    fig.suptitle(
        f"{scope_label} • {subject_str} • Grade {grade} • Fall→Fall Growth",
        fontsize=20,
        fontweight="bold",
        y=1,
    )

    _save_cgp_chart(
        fig,
        scope_label,
        section_num=5,
        suffix=f"cgp_cgi_grade_trends_grade{grade}_{subject_str.lower().replace(' ', '_')}",
    )
    if preview:
        plt.show()


# Optional: restrict Section 5 grade-level batches based on frontend selection.
# The runner passes selected grades as: NWEA_BOY_GRADES="3,4,5"
_env_grades = os.getenv("NWEA_BOY_GRADES")
_selected_grades = None
if _env_grades:
    try:
        _parsed = {int(x.strip()) for x in str(_env_grades).split(",") if x.strip()}
        if _parsed:
            _selected_grades = _parsed
            print(f"[FILTER] Grade selection from frontend: {sorted(_selected_grades)}")
    except Exception:
        _selected_grades = None

# SECTION 5 DRIVER — Districtwide
district_display = district_label
d0 = nwea_base.copy()
d0["year"] = pd.to_numeric(d0["year"], errors="coerce")
d0["grade"] = pd.to_numeric(d0["grade"], errors="coerce")
grades = sorted(d0["grade"].dropna().unique())
subjects = _requested_core_subjects()
preview = False  # or True for interactive preview

for grade in grades:
    try:
        grade = int(grade)
    except Exception:
        continue
    if _selected_grades is not None and grade not in _selected_grades:
        continue
    for subject in subjects:
        overall_df = _prep_cgp_by_grade(d0, subject, grade)
        if overall_df.empty:
            continue
        anchor_year = int(d0[d0["grade"] == grade]["year"].max())
        cohort_rows = []
        for offset in range(3, -1, -1):
            yr = anchor_year - offset
            gr = grade - offset
            if gr < 0:
                continue
            d = d0.copy()
            d = d[
                (d["year"] == yr)
                & (d["grade"] == gr)
                & (d["testwindow"].str.upper() == "FALL")
            ]
            if "teststartdate" in d.columns:
                d = d.sort_values("teststartdate").drop_duplicates(
                    subset=["uniqueidentifier", "year", "grade", "course", "subject"],
                    keep="last",
                )
            if subject.lower() == "mathematics":
                d = d[d["course"] == "Math K-12"]
            else:
                d = d[d["course"].str.contains("read", case=False, na=False)]
            d = d.dropna(
                subset=[
                    "falltofallconditionalgrowthpercentile",
                    "falltofallconditionalgrowthindex",
                ]
            )
            if d.empty:
                continue
            cohort_rows.append(
                {
                    "gr": gr,
                    "yr": yr,
                    "time_label": f"Gr {int(gr)} • Fall {str(yr - 1)[-2:]}-{str(yr)[-2:]}",
                    "median_cgp": d["falltofallconditionalgrowthpercentile"].median(),
                    "mean_cgi": d["falltofallconditionalgrowthindex"].mean(),
                }
            )
        if not cohort_rows:
            continue
        cohort_df = pd.DataFrame(cohort_rows)
        cohort_df = cohort_df.sort_values(["gr", "yr"])
        ordered_labels = cohort_df["time_label"].tolist()
        cohort_df["time_label"] = pd.Categorical(
            cohort_df["time_label"], categories=ordered_labels, ordered=True
        )
        _plot_cgp_dual_facet(
            overall_df,
            cohort_df,
            grade,
            subject,
            scope_label=district_display,
            preview=preview,
        )

# %% SECTION 5 DRIVER — By School
all_schools = list(_iter_schools(nwea_base)) if _include_school_charts() else []
grades = sorted(nwea_base["grade"].dropna().unique())
subjects = _requested_core_subjects()
preview = False  # set True if preview needed

for raw_school in all_schools:
    scope_label = hf._safe_normalize_school_name(raw_school, cfg)
    d0 = nwea_base[nwea_base["schoolname"] == raw_school].copy()
    d0["year"] = pd.to_numeric(d0["year"], errors="coerce")
    d0["grade"] = pd.to_numeric(d0["grade"], errors="coerce")
    for grade in grades:
        try:
            grade = int(grade)
        except Exception:
            continue
        if _selected_grades is not None and grade not in _selected_grades:
            continue
        for subject in subjects:
            overall_df = _prep_cgp_by_grade(d0, subject, grade)
            if overall_df.empty:
                continue
            anchor_year = d0.loc[d0["grade"] == grade, "year"].max()
            if pd.isna(anchor_year):
                continue
            anchor_year = int(anchor_year)
            cohort_rows = []
            for offset in range(3, -1, -1):
                yr = anchor_year - offset
                gr = grade - offset
                if gr < 0:
                    continue
                d = d0.copy()
                d = d[
                    (d["year"] == yr)
                    & (d["grade"] == gr)
                    & (d["testwindow"].str.upper() == "FALL")
                ]
                if "teststartdate" in d.columns:
                    d = d.sort_values("teststartdate").drop_duplicates(
                        subset=[
                            "uniqueidentifier",
                            "year",
                            "grade",
                            "course",
                            "subject",
                        ],
                        keep="last",
                    )
                if subject.lower() == "mathematics":
                    d = d[d["course"] == "Math K-12"]
                else:
                    d = d[d["course"].str.contains("read", case=False, na=False)]
                d = d.dropna(
                    subset=[
                        "falltofallconditionalgrowthpercentile",
                        "falltofallconditionalgrowthindex",
                    ]
                )
                if d.empty:
                    continue
                cohort_rows.append(
                    {
                        "gr": gr,
                        "yr": yr,
                        "time_label": f"Gr {int(gr)} • Fall {str(yr - 1)[-2:]}-{str(yr)[-2:]}",
                        "median_cgp": d[
                            "falltofallconditionalgrowthpercentile"
                        ].median(),
                        "mean_cgi": d["falltofallconditionalgrowthindex"].mean(),
                    }
                )
            if not cohort_rows:
                continue
            cohort_df = pd.DataFrame(cohort_rows)
            cohort_df = cohort_df.sort_values(["gr", "yr"])
            ordered_labels = cohort_df["time_label"].tolist()
            cohort_df["time_label"] = pd.Categorical(
                cohort_df["time_label"], categories=ordered_labels, ordered=True
            )
            _plot_cgp_dual_facet(
                overall_df,
                cohort_df,
                grade,
                subject,
                scope_label=scope_label,
                preview=preview,
            )


# ---------------------------------------------------------------------
# Helper for standardized CGP chart saving (used in Section 5 and 4)
# ---------------------------------------------------------------------
def _save_cgp_chart(
    fig, scope_label, section_num=4, suffix="cgp_fall_to_fall_dualpanel"
):
    charts_dir = CHARTS_DIR
    folder_name = (
        "_district"
        if scope_label == district_all_students_label
        else scope_label.replace(" ", "_")
    )
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section{section_num}_{suffix}.png"
    out_path = out_dir / out_name
    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    logger.info(f"[CHART] Saved: {out_path}")
    print(f"Saved: {out_path}")


# %%
"""
# %% SECTION 6 — Validate 2025 (Pred vs Actual) + Project 2026
# ------------------------------------------------------------
# Train on <=2024
# Predict 2025 and 2026
# Chart A: 2025 Predicted (left) vs Actual (right)
# Chart B: 2026 Projected
# ------------------------------------------------------------

_CASP_BAND_ORDER = [
    "Level 1 - Standard Not Met",
    "Level 2 - Standard Nearly Met",
    "Level 3 - Standard Met",
    "Level 4 - Standard Exceeded",
]


def _filter_fall_course_grades(df, subject, require_cers=True):
    d = df.copy()
    d = d[d["testwindow"].str.upper() == "FALL"]
    if "math" in subject.lower():
        d = d[d["course"].str.contains("math", case=False, na=False)]
    else:
        d = d[d["course"].str.contains("read", case=False, na=False)]
    d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
    d = d[d["grade"].isin([3, 4, 5, 6, 7, 8, 11])]
    if require_cers:
        d = d[d["cers_overall_performanceband"].notna()]
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    return d


def _train_model(df, subject):
    d0 = _filter_fall_course_grades(df, subject, require_cers=True)
    train = d0[d0["year"] <= 2024].copy()
    if train.empty:
        return None
    clf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42
    )
    clf.fit(train[["testritscore"]],
            train["cers_overall_performanceband"].astype(str))
    return clf


def _predict_2025(df, subject, clf):
    d0 = _filter_fall_course_grades(df, subject, require_cers=True)
    val = d0[d0["year"] == 2025].copy()
    if val.empty:
        return None, None, None
    y_true = val["cers_overall_performanceband"].astype(str)
    y_pred = clf.predict(val[["testritscore"]])
    labs = [l for l in _CASP_BAND_ORDER if l in y_true.values or l in y_pred]
    return y_true, y_pred, labs


def _predict_2026(df, subject, clf):
    # future data has no CAASPP labels yet
    d0 = _filter_fall_course_grades(df, subject, require_cers=False)
    fut = d0[d0["year"] == 2026].copy()
    if fut.empty:
        return None, None
    y_pred = clf.predict(fut[["testritscore"]])
    labs = [l for l in _CASP_BAND_ORDER if l in y_pred]
    return y_pred, labs

def _pct(arr, labels):
    arr = np.asarray(arr).astype(str)
    raw = np.array([(arr == lab).mean() * 100 if len(arr) else 0 for lab in labels])
    total = raw.sum()
    if total > 0:
        raw = raw * (100 / total)  # force total = 100
    return raw


def _plot_pred_vs_actual(scope_label, folder_name, results, preview=False):
    fig, axs = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(wspace=0.25)

    for ax, subject in zip(axs, ["Reading", "Mathematics"]):
        r = results.get(subject)
        if not r:
            ax.set_axis_off()
            continue

        y_true, y_pred, labs = r
        pct_true = _pct(y_true, labs)
        pct_pred = _pct(y_pred, labs)

        x_pred, x_act = -0.2, 0.2
        w = 0.35

        # Predicted (LEFT)
        bottom = 0
        for lab, val in zip(labs, pct_pred):
            ax.bar(
                x_pred,
                val,
                bottom=bottom,
                width=w,
                color=hf.CERS_LEVEL_COLORS[lab],
                edgecolor="black",
                linestyle="--",
                alpha=0.5,
            )
            if val > 3:
                ax.text(
                    x_pred,
                    bottom + val / 2,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )
            bottom += val

        # Actual (RIGHT)
        bottom = 0
        for lab, val in zip(labs, pct_true):
            ax.bar(
                x_act,
                val,
                bottom=bottom,
                width=w,
                color=hf.CERS_LEVEL_COLORS[lab],
                edgecolor="white",
            )
            if val > 3:
                ax.text(
                    x_act,
                    bottom + val / 2,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
            bottom += val

        ax.set_xticks([x_pred, x_act])
        ax.set_xticklabels(["2025\nPredicted", "2025\nActual"])
        ax.set_ylim(0, 100)
        ax.set_title(subject, fontweight="bold")
        ax.set_ylabel("% of Students")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{scope_label} \n 2025 Predicted vs Actual CAASPP (Fall NWEA)",
        fontsize=20,
        fontweight="bold",
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=hf.CERS_LEVEL_COLORS[l],
            markeredgecolor="white",
            markersize=10,
            label=l,
        )
        for l in _CASP_BAND_ORDER
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    out_dir = CHARTS_DIR / folder_name
    out_dir.mkdir(exist_ok=True, parents=True)
    safe_scope = scope_label.replace(" ", "_")
    out_path = out_dir / f"{safe_scope}_section6a_pred_vs_actual.png"
    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    logger.info(f"[CHART] Saved: {out_path}")
    print(f"Saved: {out_path}")
    if preview:
        plt.show()
    plt.close(fig)


def _plot_projection_2026(scope_label, folder_name, results, preview=False):
    fig, axs = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(wspace=0.25)

    for ax, subject in zip(axs, ["Reading", "Mathematics"]):
        r = results.get(subject)
        if not r:
            ax.set_axis_off()
            continue

        y_pred, labs = r
        pct = _pct(y_pred, labs)

        bottom = 0
        for lab, val in zip(labs, pct):
            ax.bar(
                0,
                val,
                bottom=bottom,
                width=0.55,
                color=hf.CERS_LEVEL_COLORS[lab],
                edgecolor="white",
                alpha=0.9,
            )
            if val > 3:
                ax.text(
                    0,
                    bottom + val / 2,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
            bottom += val

        ax.set_xticks([])
        ax.set_ylim(0, 100)
        ax.set_title(f"{subject} — 2026 Projected", fontweight="bold")
        ax.set_ylabel("% of Students")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{scope_label} \n Projected 2026 CAASPP (Fall 2026 NWEA)",
        fontsize=20,
        fontweight="bold"
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=hf.CERS_LEVEL_COLORS[l],
            markeredgecolor="white",
            markersize=10,
            label=l,
        )
        for l in _CASP_BAND_ORDER
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    out_dir = CHARTS_DIR / folder_name
    out_dir.mkdir(exist_ok=True, parents=True)
    safe_scope = scope_label.replace(" ", "_")
    out_path = out_dir / f"{safe_scope}_section6b_projection.png"
    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    logger.info(f"[CHART] Saved: {out_path}")
    print(f"Saved: {out_path}")
    if preview:
        plt.show()
    plt.close(fig)


# ---- RUN SECTION 6 ----
_section6_schools = list(_iter_schools(nwea_base)) if _include_school_charts() else []
for raw in [None] + _section6_schools:
    if raw is None:
        scope_df = nwea_base.copy()
        scope_label = district_label
        folder = "_district"
    else:
        scope_df = nwea_base[nwea_base["schoolname"] == raw].copy()
        scope_label = hf._safe_normalize_school_name(raw, cfg)
        folder = scope_label.replace(" ", "_")

    results_2025 = {}
    results_2026 = {}
    for subj in _requested_core_subjects():
        clf = _train_model(scope_df, subj)
        if clf is None:
            continue

        y_true25, y_pred25, labs25 = _predict_2025(scope_df, subj, clf)
        if y_true25 is not None:
            results_2025[subj] = (y_true25, y_pred25, labs25)

        y_pred26, labs26 = _predict_2026(scope_df, subj, clf)
        if y_pred26 is not None:
            results_2026[subj] = (y_pred26, labs26)

    if results_2025:
        _plot_pred_vs_actual(scope_label, folder, results_2025, preview=False)

    if results_2026:
        _plot_projection_2026(scope_label, folder, results_2026, preview=False)
# %%

"""
