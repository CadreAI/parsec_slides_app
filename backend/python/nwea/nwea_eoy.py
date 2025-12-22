# %% Imports and config
# nwea.py — charts and analytics
#
# NOTE: This is a legacy script. Like `iready_moy.py`, it executes a lot of work at
# import-time and assumes on-disk `settings.yaml`, `config_files/{partner}.yaml`,
# and `../data/nwea_data.csv`. The app runs it via a subprocess runner which injects
# temp settings/config/data locations via env vars.
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

_NWEA_EOY_HARD_RC = {
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
mpl.rcParams.update(_NWEA_EOY_HARD_RC)

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
import warnings
import logging
from nwea_data import filter_nwea_subject_rows

warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress matplotlib font warnings
warnings.filterwarnings("ignore", message=".*findfont.*", category=UserWarning, module="matplotlib")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Setup logging - both console and file
LOG_DIR = Path(os.getenv("NWEA_EOY_LOG_DIR") or "../logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"nwea_eoy_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"

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

# ----------------------------
# Frontend-driven filter inputs
# ----------------------------
def _parse_env_csv(var_name: str):
    raw = os.getenv(var_name)
    if not raw:
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return parts or None


_selected_subjects = _parse_env_csv("NWEA_EOY_SUBJECTS")
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


def _write_chart_data(out_path: Path, chart_data: dict) -> None:
    """
    Write sidecar JSON data for chart analysis.

    `chart_analyzer.py` expects a file named `{chart_stem}_data.json` next to the PNG.
    """
    # Important: serialize first so we never leave partially-written JSON on disk.
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

                    NWEA_TITLES = {
                        "0": "CAASPP Predicted vs Actual",
                        "1": "Performance Trends",
                        "2": "Student Group Performance Trends",
                        "3": "Overall + Cohort Trends",
                        "4": "Overall Growth Trends by Site",
                        "5": "CGP/CGI Growth: Grade Trend + Backward Cohort",
                        "6": "Window Compare by School",
                        "7": "Window Compare by Grade",
                        "8": "Window Compare by Student Group",
                        "9": "Growth by School",
                        "10": "Growth by Grade",
                        "11": "Growth by Student Group",
                    }
                    payload["section_title"] = NWEA_TITLES.get(sec_key, f"Section {sec_key}")
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

        os.replace(tmp_path, data_path)
        logger.info(f"[CHART] Wrote chart data: {data_path}")
    except Exception as e:  # pragma: no cover
        try:
            if "tmp_path" in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        # Never fail chart creation due to data sidecar issues.
        try:
            logger.warning(
                f"[CHART] Failed to write chart data sidecar for {out_path}: {e}"
            )
        except Exception:
            pass


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

# Libraries for modeling (optional; many environments won't have these installed)
try:
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import (  # type: ignore
        r2_score,
        mean_absolute_error,
        accuracy_score,
        f1_score,
    )
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
SETTINGS_PATH = Path(
    os.getenv("NWEA_EOY_SETTINGS_PATH") or (Path(__file__).resolve().parent / "settings.yaml")
)

# Step 1: read partner name from settings.yaml
with open(SETTINGS_PATH, "r") as f:
    base_cfg = yaml.safe_load(f)

partner_name = base_cfg.get("partner_name")
if not partner_name:
    raise ValueError("settings.yaml must include a 'partner_name' key")

# Step 2: load the partner config file from /config_files/{partner_name}.yaml
CONFIG_PATH = Path(
    os.getenv("NWEA_EOY_CONFIG_PATH")
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
DATA_DIR = Path(os.getenv("NWEA_EOY_DATA_DIR") or "../data")
csv_path = DATA_DIR / "nwea_data.csv"
LABEL_MIN_PCT = 5.0

if not csv_path.exists():
    raise FileNotFoundError(
        f"Expected CSV not found: {csv_path}. Please run data_ingest.py first."
    )

nwea_base = pd.read_csv(csv_path)
nwea_base.columns = nwea_base.columns.str.strip().str.lower()

# ===== DIAGNOSTIC LOGGING FOR SECTIONS 4, 5, 9-11 =====
logger.info(f"[DATA LOAD] Loaded {len(nwea_base):,} rows from CSV")
logger.info(f"[DATA LOAD] Total columns: {len(nwea_base.columns)}")

# Check for critical columns
has_testwindow = "testwindow" in nwea_base.columns
has_year = "year" in nwea_base.columns or "academicyear" in nwea_base.columns
logger.info(f"[DATA LOAD] Has testwindow: {has_testwindow}, Has year: {has_year}")

# Check test windows present
if has_testwindow:
    windows = nwea_base["testwindow"].str.upper().value_counts().to_dict()
    logger.info(f"[DATA LOAD] Test windows present: {windows}")

# Check for conditional growth columns
cond_growth_cols = sorted([c for c in nwea_base.columns if "conditionalgrowth" in c.lower()])
logger.info(f"[DATA LOAD] Found {len(cond_growth_cols)} conditional growth columns:")
for col in cond_growth_cols:
    non_null = nwea_base[col].notna().sum()
    logger.info(f"  - {col}: {non_null:,} non-null values")

# Check specifically for Spring-ended growth columns
spring_growth = [c for c in cond_growth_cols if "spring" in c.lower() and ("percentile" in c.lower() or "index" in c.lower())]
logger.info(f"[DATA LOAD] Spring-ended growth columns ({len(spring_growth)}): {spring_growth}")

# WRITE DIAGNOSTIC FILE TO CHARTS DIR (easy to find!)
try:
    diagnostic_path = CHARTS_DIR / "DIAGNOSTIC_nwea_eoy_columns.txt"
    with open(diagnostic_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NWEA EOY DATA DIAGNOSTIC\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total rows: {len(nwea_base):,}\n")
        f.write(f"Total columns: {len(nwea_base.columns)}\n\n")
        
        if has_testwindow:
            f.write("Test windows:\n")
            for window, count in sorted(windows.items()):
                f.write(f"  {window}: {count:,} rows\n")
            f.write("\n")
        
        f.write(f"Conditional growth columns found: {len(cond_growth_cols)}\n")
        if cond_growth_cols:
            f.write("\nAll conditional growth columns:\n")
            for col in cond_growth_cols:
                non_null = nwea_base[col].notna().sum()
                pct = (non_null / len(nwea_base) * 100) if len(nwea_base) > 0 else 0
                f.write(f"  {col}: {non_null:,} non-null ({pct:.1f}%)\n")
        
        f.write(f"\nSpring-ended growth columns: {len(spring_growth)}\n")
        if spring_growth:
            for col in spring_growth:
                non_null = nwea_base[col].notna().sum()
                pct = (non_null / len(nwea_base) * 100) if len(nwea_base) > 0 else 0
                f.write(f"  {col}: {non_null:,} non-null ({pct:.1f}%)\n")
        else:
            f.write("  ⚠️  NO SPRING-ENDED GROWTH COLUMNS FOUND!\n")
            f.write("  Sections 4, 5, 9-11 require these columns:\n")
            f.write("    - wintertospringconditionalgrowthpercentile\n")
            f.write("    - wintertospringconditionalgrowthindex\n")
            f.write("    - springtospringconditionalgrowthpercentile\n")
            f.write("    - springtospringconditionalgrowthindex\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("If sections 4, 5, 9-11 are not generating:\n")
        f.write("1. Check if Spring-ended columns exist above\n")
        f.write("2. Check if they have non-null values\n")
        f.write("3. Check if SPRING test window has rows\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"[DIAGNOSTIC] Wrote column diagnostic to: {diagnostic_path}")
    print(f"📊 DIAGNOSTIC FILE: {diagnostic_path}")
except Exception as e:
    logger.warning(f"[DIAGNOSTIC] Failed to write diagnostic file: {e}")
# ===== END DIAGNOSTIC LOGGING =====

# --- Charter datasets sometimes store school values in `learning_center` instead of `schoolname`/`school`.
# Many downstream sections assume a `schoolname` column, so we normalize the best available school column into `schoolname`.
def _pick_school_col(df: pd.DataFrame) -> str | None:
    # Prefer learning_center first, then fall back to other common variants
    for c in ["learning_center", "schoolname", "school", "school_name"]:
        if c in df.columns:
            try:
                if df[c].notna().any():
                    return c
            except Exception:
                return c
    return None


_school_col = _pick_school_col(nwea_base)
if _school_col and _school_col != "schoolname":
    # Always prefer `learning_center` when present so frontend-selected schools match reliably.
    nwea_base["schoolname"] = nwea_base[_school_col]

# --- Normalize school names using YAML school_name_map ---
school_map = cfg.get("school_name_map", {})
if "schoolname" in nwea_base.columns and school_map:
    # Clean incoming names and map to canonical YAML names
    nwea_base["schoolname"] = (
        nwea_base["schoolname"].astype(str).str.strip().replace(school_map)
    )

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

nwea_base["projectedproficiencylevel2"] = nwea_base[
    "projectedproficiencylevel2"
].replace(prof_prof_map)
print(f"NWEA data loaded: {nwea_base.shape[0]:,} rows, {nwea_base.shape[1]} columns")
print(nwea_base["year"].value_counts().sort_index())
print(nwea_base.columns.tolist())

def _cfg_first(cfg_obj: dict, key: str, default: str) -> str:
    """
    Return a stable display string from config.

    Handles cases like:
    - key missing
    - key present but [] (empty list)
    - key present as a single string
    """
    try:
        v = (cfg_obj or {}).get(key, None)
    except Exception:
        v = None

    if isinstance(v, list):
        if v and str(v[0]).strip():
            return str(v[0]).strip()
        return default
    if isinstance(v, str):
        return v.strip() or default
    if v is None:
        return default
    return str(v)


# District label (charter datasets may not have a district column; still produce an "all schools" aggregate)
# Allow a display-only override that does NOT affect filtering / school matching.
district_label = _cfg_first(cfg, "district_display_name", _cfg_first(cfg, "district_name", "District"))
district_all_students_label = _cfg_first(cfg, "district_all_students_label", f"{district_label} (All Students)")


def _is_district_scope(scope_label: str | None) -> bool:
    """True if a scope_label represents the districtwide aggregate."""
    if scope_label is None:
        return False
    return str(scope_label) in {str(district_label), str(district_all_students_label)}

# Base charts directory (overrideable by runner)
CHARTS_DIR = Path(os.getenv("NWEA_EOY_CHARTS_DIR") or "../charts")

# EOY window compare set (used by within-year snapshots)
def _get_eoy_compare_windows() -> list[str]:
    """
    Env:
      - NWEA_EOY_COMPARE_WINDOWS="WINTER,SPRING" (default)
      - NWEA_EOY_COMPARE_WINDOWS="FALL,WINTER,SPRING"
    """
    raw = os.getenv("NWEA_EOY_COMPARE_WINDOWS")
    if raw:
        parts = [p.strip().upper() for p in str(raw).split(",") if p.strip()]
        parts = [p for p in parts if p in {"FALL", "WINTER", "SPRING"}]
        if parts:
            seen: set[str] = set()
            out: list[str] = []
            for p in parts:
                if p not in seen:
                    out.append(p)
                    seen.add(p)
            return out
    return ["WINTER", "SPRING"]

# ---------------------------------------------------------------------
# Scope selection (district-only vs district + schools vs selected schools)
#
# Env vars (set by runner / backend):
# - NWEA_EOY_SCOPE_MODE:
#     - "district_only" (skip all school loops)
#     - "selected_schools" (only loop selected schools; still include district)
#     - default/other: district + all schools
# - NWEA_EOY_SCHOOLS="School A,School B" (names can be raw or normalized)
# ---------------------------------------------------------------------
_scope_mode = str(os.getenv("NWEA_EOY_SCOPE_MODE") or "").strip().lower()
_env_schools = os.getenv("NWEA_EOY_SCHOOLS")
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

# Optional: restrict grade-level dashboards based on frontend selection.
# The runner passes selected grades as: NWEA_EOY_GRADES="3,4,5"
_env_grades = os.getenv("NWEA_EOY_GRADES")
_selected_grades = None
if _env_grades:
    try:
        _parsed = {int(x.strip()) for x in str(_env_grades).split(",") if x.strip()}
        if _parsed:
            _selected_grades = _parsed
            print(f"[FILTER] Grade selection from frontend: {sorted(_selected_grades)}")
    except Exception:
        _selected_grades = None

# Inspect categorical columns (quick QC)
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
    # EOY focuses on Spring. Filter to Spring only.
    before_window = len(d)
    d["testwindow"] = d["testwindow"].astype(str)
    d = d[d["testwindow"].str.upper() == "SPRING"].copy()
    logger.info(f"[FILTER] After Spring window filter: {len(d):,} rows (removed {before_window - len(d):,})")

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

    # Coerce year to numeric
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    if d["year"].dropna().empty:  # safety net
        return None, None, None, None

    # Only consider rows that actually have both projected and actual CAASPP values
    before_caaspp = len(d)
    d = d.dropna(
        subset=["projectedproficiencylevel2", "cers_overall_performanceband"]
    ).copy()
    logger.info(f"[FILTER] After CAASPP data filter (removed nulls): {len(d):,} rows (removed {before_caaspp - len(d):,})")
    if d.empty or d["year"].dropna().empty:
        return None, None, None, None

    # Target year is the latest test year with valid projected + CAASPP join
    target_year = int(d["year"].max())

    # Keep only that target year slice
    before_year = len(d)
    d = d[d["year"] == target_year].copy()
    logger.info(f"[FILTER] After year filter (target_year={target_year}): {len(d):,} rows (removed {before_year - len(d):,})")

    # Dedupe — most recent test per student within the target year slice
    before_dedup = len(d)
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
        d = d.sort_values("teststartdate").drop_duplicates(
            "uniqueidentifier", keep="last"
        )
        logger.info(f"[FILTER] After deduplication (latest per student): {len(d):,} rows (removed {before_dedup - len(d):,} duplicates)")

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
        f"{scope_label} • Winter {first_metrics['year']} NWEA to CAASPP Prediction Accuracy",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )

    # Standardize folder naming:
    #   - District-level charts go to ../charts/_district
    #   - School-level charts go to ../charts/{School_Name_Safe}
    charts_dir = CHARTS_DIR
    district_name_cfg = district_label

    if scope_label == district_name_cfg:
        folder_name = "_district"
    else:
        folder_name = (
            scope_label.replace(" ", "_").replace("/", "_").replace("&", "and")
        )

    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section0_pred_vs_actual.png"
    out_path = out_dir / out_name

    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "nwea_eoy_section0_pred_vs_actual",
            "section": 0,
            "scope": scope_label,
            "folder": folder_name,
            "window_filter": "Spring",
            "subjects": subjects,
            "subj_payload": {
                k: {
                    "metrics": v.get("metrics", {}),
                    "proj_pct": _jsonable(v.get("proj_pct")),
                    "act_pct": _jsonable(v.get("act_pct")),
                }
                for k, v in (subj_payload or {}).items()
            },
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
        scope_df = nwea_base.copy()
        scope_label = district_label
        folder = "_district"
    else:
        scope_df = nwea_base[nwea_base["schoolname"] == raw].copy()
        scope_label = hf._safe_normalize_school_name(raw, cfg)
        folder = scope_label.replace(" ", "_")

    payload = {}
    for subj in _requested_core_subjects():
        proj, act, metrics, _ = _prep_section0(scope_df, subj)
        if proj is None:
            continue
        payload[subj] = {"proj_pct": proj, "act_pct": act, "metrics": metrics}

    if payload:
        _plot_section0_dual(scope_label, folder, payload, preview=False)

# %% SECTION 0.1 – Window Compare Performance Snapshot (Dual Subject)
# ---------------------------------------------------------------------
# Similar layout to Section 1, but compares selected windows within the
# same (latest) school year instead of year-to-year trends.
# Top row: 100% stacked bars — selected windows (latest year only)
# Middle row: Avg RIT — selected windows
# Bottom row: Insight card with deltas from first → last window
# ---------------------------------------------------------------------


def _prep_nwea_window_compare_snapshot(
    df: pd.DataFrame,
    subject_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """Prep frame for within-year window comparison in the latest year.

    Returns:
        pct_df    – % by quintile for each window (e.g., Winter, Spring or Fall, Winter, Spring)
        score_df  – Avg RIT for each window
        metrics   – delta metrics from first → last window
        time_order – ordered list of time labels
    """
    win_order = _get_eoy_compare_windows()
    logger.info(f"[FILTER] _prep_nwea_window_compare_snapshot: Starting | Subject: {subject_str} | Windows: {win_order} | Input rows: {len(df):,}")
    d = df.copy()
    initial_rows = len(d)

    # 1. Restrict to requested windows only
    d["testwindow"] = d["testwindow"].astype(str)
    mask_window = d["testwindow"].str.upper().isin(win_order)
    d = d[mask_window].copy()
    after_window = len(d)
    logger.info(f"[FILTER] After window filter: {after_window:,} rows (removed {initial_rows - after_window:,})")
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    # 2. Course-based filtering (mirror Section 1 semantics)
    before_course = len(d)
    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[d["course"].astype(str).str.contains("math", case=False, na=False)].copy()
        logger.info(f"[FILTER] After Math course filter: {len(d):,} rows (removed {before_course - len(d):,})")
    elif "reading" in subj_norm:
        d = d[
            d["course"].astype(str).str.contains("reading", case=False, na=False)
        ].copy()
        logger.info(f"[FILTER] After Reading course filter: {len(d):,} rows (removed {before_course - len(d):,})")

    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    # 3. Require valid quintile bucket
    before_quintile = len(d)
    d = d[d["achievementquintile"].notna()].copy()
    after_quintile = len(d)
    logger.info(f"[FILTER] After quintile filter (removed nulls): {after_quintile:,} rows (removed {before_quintile - after_quintile:,})")
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    # 4. Normalize year and choose latest
    before_year = len(d)
    d["year_num"] = pd.to_numeric(d["year"].astype(str).str[:4], errors="coerce")
    d = d[d["year_num"].notna()].copy()
    after_year_norm = len(d)
    logger.info(f"[FILTER] After year normalization (removed nulls): {after_year_norm:,} rows (removed {before_year - after_year_norm:,})")
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    target_year = int(d["year_num"].max())
    before_target_year = len(d)
    d = d[d["year_num"] == target_year].copy()
    logger.info(f"[FILTER] After target year filter (target_year={target_year}): {len(d):,} rows (removed {before_target_year - len(d):,})")
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []

    # 5. Build time_label like "Winter 25-26" / "Spring 25-26"
    def _short_year(y):
        ys = str(int(y))
        return f"{str(int(ys) - 1)[-2:]}-{ys[-2:]}"

    d["year_short"] = d["year_num"].apply(_short_year)
    d["time_label"] = d["testwindow"].str.title() + " " + d["year_short"]

    # 6. Dedupe to latest attempt per student per time_label
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    else:
        d["teststartdate"] = pd.NaT

    d.sort_values(["uniqueidentifier", "time_label", "teststartdate"], inplace=True)
    d = d.groupby(["uniqueidentifier", "time_label"], as_index=False).tail(1)

    # 7. Percent by quintile
    quint_counts = (
        d.groupby(["time_label", "achievementquintile"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_counts = d.groupby("time_label").size().rename("N_total").reset_index()

    pct_df = quint_counts.merge(total_counts, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]

    # Ensure all quintiles exist for requested windows, Low→High order
    time_labels = pct_df["time_label"].unique().tolist()
    all_idx = pd.MultiIndex.from_product(
        [time_labels, hf.NWEA_ORDER],
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

    # 8. Avg RIT per time_label based on deduped records
    score_df = (
        d[["time_label", "testritscore"]]
        .dropna(subset=["testritscore"])
        .groupby("time_label")["testritscore"]
        .mean()
        .rename("avg_score")
        .reset_index()
    )

    # 9. Enforce window ordering (Fall → Winter → Spring)
    def _sort_key(lbl: str) -> tuple[int, str]:
        # Ensure Fall comes before Winter, before Spring for the same year label
        if lbl.startswith("Fall"):
            season_order = 0
        elif lbl.startswith("Winter"):
            season_order = 1
        elif lbl.startswith("Spring"):
            season_order = 2
        else:
            season_order = 99
        return (season_order, lbl)

    time_order = sorted(
        pct_df["time_label"].dropna().astype(str).unique().tolist(), key=_sort_key
    )

    pct_df["time_label"] = pd.Categorical(
        pct_df["time_label"], categories=time_order, ordered=True
    )
    score_df["time_label"] = pd.Categorical(
        score_df["time_label"], categories=time_order, ordered=True
    )

    pct_df.sort_values(["time_label", "achievementquintile"], inplace=True)
    score_df.sort_values(["time_label"], inplace=True)

    # 10. Insight metrics from first → last window (if both present)
    if len(time_order) >= 2:
        t_prev, t_curr = time_order[0], time_order[-1]

        def pct_for(bucket_list, tlabel):
            return pct_df[
                (pct_df["time_label"] == tlabel)
                & (pct_df["achievementquintile"].isin(bucket_list))
            ]["pct"].sum()

        hi_curr = pct_for(hf.NWEA_HIGH_GROUP, t_curr)
        hi_prev = pct_for(hf.NWEA_HIGH_GROUP, t_prev)
        lo_curr = pct_for(hf.NWEA_LOW_GROUP, t_curr)
        lo_prev = pct_for(hf.NWEA_LOW_GROUP, t_prev)

        high_curr = pct_for(["High"], t_curr)
        high_prev = pct_for(["High"], t_prev)

        score_curr = float(
            score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
        )
        score_prev = float(
            score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0]
        )

        metrics = {
            "t_prev": t_prev,
            "t_curr": t_curr,
            "hi_now": hi_curr,
            "hi_delta": hi_curr - hi_prev,
            "lo_now": lo_curr,
            "lo_delta": lo_curr - lo_prev,
            "score_now": score_curr,
            "score_delta": score_curr - score_prev,
            "high_now": high_curr,
            "high_delta": high_curr - high_prev,
        }
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


def plot_nwea_dual_subject_window_compare_dashboard(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (16, 9),
    school_raw: str | None = None,
    scope_label: str | None = None,
    preview: bool = False,
    *,
    _subjects_override: list[str] | None = None,
    _titles_override: list[str] | None = None,
):
    """Faceted dashboard showing Reading and Math within-year window snapshot.

    Layout (same as Section 1):
      - Top row: 100% stacked bars (selected windows, latest year)
      - Middle row: Avg RIT (selected windows)
      - Bottom row: Insight card (first → last window deltas)
    """

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

    win_order = _get_eoy_compare_windows()

    # 3+ subjects => generate one single-subject chart per subject
    if len(subjects) > 2 and not (isinstance(_subjects_override, list) and _subjects_override):
        out_paths = []
        for subj, title in zip(subjects, titles):
            out_paths.extend(
                plot_nwea_dual_subject_window_compare_dashboard(
                    df,
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

    ncols = len(subjects) if len(subjects) in (1, 2) else 2
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=ncols, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    def draw_stacked_bar(ax, pct_df):
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
                    if cat in ("High", "HiAvg"):
                        label_color = "white"
                    elif cat in ("Avg", "LoAvg"):
                        label_color = "#434343"
                    elif cat == "Low":
                        label_color = "white"
                    else:
                        label_color = "#434343"
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        bottom_before + h / 2,
                        f"{h:.1f}%",
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

    def draw_score_bar(ax, pct_df, score_df):
        x = np.arange(len(score_df["time_label"]))
        vals = score_df["avg_score"].to_numpy()
        bars = ax.bar(
            x,
            vals,
            color=hf.default_quintile_colors[4],
            edgecolor="white",
            linewidth=1.2,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#434343",
            )

        # Optional n labels from pct_df
        if "N_total" in pct_df.columns:
            n_map = (
                pct_df.groupby("time_label")["N_total"]
                .max()
                .reset_index()
                .rename(columns={"N_total": "n"})
            )
        else:
            n_map = pd.DataFrame(columns=["time_label", "n"])

        if not n_map.empty:
            label_map = {
                row["time_label"]: f"{row['time_label']}\n(n = {int(row['n'])})"
                for _, row in n_map.iterrows()
                if not pd.isna(row["n"])
            }
            x_labels = [label_map.get(lbl, str(lbl)) for lbl in score_df["time_label"]]
        else:
            x_labels = score_df["time_label"].astype(str).tolist()

        ax.set_ylabel("Avg RIT")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_insight_card(ax, metrics):
        ax.axis("off")
        if metrics.get("t_prev") and metrics.get("t_curr"):
            t_prev = metrics["t_prev"]
            t_curr = metrics["t_curr"]
            high_delta = metrics.get("high_delta")
            hi_delta = metrics.get("hi_delta")
            lo_delta = metrics.get("lo_delta")
            score_delta = metrics.get("score_delta")
            title_line = f"Change from {t_prev} to {t_curr}:"

            def _fmt_delta(val):
                if val is None or pd.isna(val):
                    return "N/A"
                return f"{val:+.1f}"

            line_high = f"Δ High: {_fmt_delta(high_delta)} ppts"
            line_hiavg = f"Δ Avg+HiAvg+High: {_fmt_delta(hi_delta)} ppts"
            line_low = f"Δ Low: {_fmt_delta(lo_delta)} ppts"
            line_rit = f"Δ Avg RIT: {_fmt_delta(score_delta)} pts"
            insight_lines = [title_line, line_high, line_hiavg, line_low, line_rit]
        else:
            insight_lines = ["Not enough data for window-to-window insights"]

        ax.text(
            0.5,
            0.5,
            "\n".join(insight_lines),
            fontsize=10,
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

    win_order = _get_eoy_compare_windows()
    pct_dfs = []
    score_dfs = []
    metrics_list = []

    for subj in subjects:
        pct_df, score_df, metrics, _ = _prep_nwea_window_compare_snapshot(df, subj)
        pct_dfs.append(pct_df)
        score_dfs.append(score_df)
        metrics_list.append(metrics)

    # If either subject has no data, skip chart
    if any(
        (pct_df.empty or score_df.empty) for pct_df, score_df in zip(pct_dfs, score_dfs)
    ):
        label = scope_label or district_label
        print(f"[Section 0.1] Skipped window snapshot for {label} (missing data)")
        plt.close(fig)
        return

    # Draw panels for each subject
    for i, (pct_df, score_df, metrics, title) in enumerate(
        zip(pct_dfs, score_dfs, metrics_list, titles)
    ):
        ax1 = fig.add_subplot(gs[0, i])
        draw_stacked_bar(ax1, pct_df)
        ax1.set_title(title, fontsize=14, fontweight="bold", pad=30)

        ax2 = fig.add_subplot(gs[1, i])
        draw_score_bar(ax2, pct_df, score_df)
        ax2.set_title(
            f"Avg RIT ({win_order[0].title()} vs {win_order[-1].title()})" if len(win_order) >= 2 else "Avg RIT",
            fontsize=8,
            fontweight="bold",
            pad=10,
        )

        ax3 = fig.add_subplot(gs[2, i])
        draw_insight_card(ax3, metrics)

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

    # Title & saving
    if school_raw:
        school_display = hf._safe_normalize_school_name(school_raw, cfg)
    else:
        school_display = district_label

    main_label = school_display
    fig.suptitle(
        f"{main_label} • {win_order[0].title()} to {win_order[-1].title()} Performance Snapshot",
        fontsize=20,
        fontweight="bold",
        y=1,
    )

    charts_dir = CHARTS_DIR
    folder_name = (
        "_district" if school_raw is None else school_display.replace(" ", "_")
    )
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    scope = scope_label or main_label
    if len(subjects) == 1:
        safe_subj = titles[0].replace(" ", "_").replace("/", "_")
        out_name = (
            f"{scope.replace(' ', '_')}_section0_1_{safe_subj}_window_compare_eoy.png"
        )
    else:
        out_name = (
            f"{scope.replace(' ', '_')}_section0_1_dual_subject_window_compare_eoy.png"
        )
    out_path = out_dir / out_name

    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "nwea_eoy_section0_1_window_compare_snapshot",
            "section": "0.1",
            "scope": scope,
            "window_filter": "/".join([w.title() for w in win_order]),
            "subjects": titles,
            "pct_data": [
                {"subject": titles[i], "data": pct_dfs[i].to_dict("records")}
                for i in range(len(titles))
            ],
            "score_data": [
                {"subject": titles[i], "data": score_dfs[i].to_dict("records")}
                for i in range(len(titles))
            ],
            "metrics": metrics_list,
            "time_orders": [
                sorted(pct_dfs[i]["time_label"].astype(str).unique().tolist())
                if (pct_dfs[i] is not None and not pct_dfs[i].empty and "time_label" in pct_dfs[i].columns)
                else []
                for i in range(len(titles))
            ],
        },
    )
    logger.info(f"[CHART] Saved Section 0.1: {out_path}")
    print(f"Saved Section 0.1: {out_path}")

    if preview:
        plt.show()
    plt.close()
    return [str(out_path)]


# ---------------------------------------------------------------------
# Section 0.1 Drivers – Window Compare Snapshot (District + Sites)
# ---------------------------------------------------------------------

# District-level snapshot
scope_label_01 = district_label
plot_nwea_dual_subject_window_compare_dashboard(
    nwea_base.copy(),
    figsize=(16, 9),
    school_raw=None,
    scope_label=scope_label_01,
    preview=False,
)

# Site-level snapshots
if _include_school_charts():
    for raw_school_01 in _iter_schools(nwea_base):
        school_display_01 = hf._safe_normalize_school_name(raw_school_01, cfg)
        school_df_01 = nwea_base[nwea_base["schoolname"] == raw_school_01].copy()

        plot_nwea_dual_subject_window_compare_dashboard(
            school_df_01,
            figsize=(16, 9),
            school_raw=raw_school_01,
            scope_label=school_display_01,
            preview=False,
        )


# %% SECTION 1 - Spring Performance Trends
# Subject Dashboards by Year/Window
# Example labels: "Winter 22-23", "Winter 23-24", "Winter 24-25", "Winter 25-26"
# Rules:
#   - Window: Winter only (default, configurable)
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
    window_filter: str = "Spring",
):
    """
    Filters and aggregates NWEA data for dashboard plotting.

    Rules:
      - Keep only the requested test window (ex: "Winter").
      - For Mathematics:
            keep rows where course contains "math".
      - For Reading:
            keep rows where course is "reading" or "reading (spanish)".
            exclude "language usage".
      - Drop rows with missing achievementquintile.
      - Keep the latest test per student per time_label using the most recent
        teststartdate.
      - Build time_label like "22-23 Winter".
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

    # 4. build "22-23 Winter" style label
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
    window_filter="Spring",
    figsize=(16, 9),
    school_raw=None,
    scope_label=None,
    preview=False,
    *,
    _subjects_override: list[str] | None = None,
    _titles_override: list[str] | None = None,
):
    """
    Faceted dashboard showing both Math and Reading for a given scope (district or school).
    """
    logger.info(f"[CHART] plot_nwea_dual_subject_dashboard: Starting | Scope: {scope_label} | Window: {window_filter} | Input rows: {len(df):,}")
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

    # 3+ subjects => generate one single-subject chart per subject
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

    ncols = len(subjects) if len(subjects) in (1, 2) else 2
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=ncols, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    # Build sidecar chart data payload as we go
    _pct_payload = []
    _score_payload = []
    _metrics_payload = []
    _time_orders_payload = []

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
                line_high = rf"$\Delta$ High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
                line_hiavg = (
                    rf"$\Delta$ Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
                )
                line_low = rf"$\Delta$ Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
                line_rit = rf"$\Delta$ Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
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

        ax1 = fig.add_subplot(gs[0, i])
        draw_stacked_bar(ax1, pct_df, score_df, hf.NWEA_ORDER)
        ax1.set_title(f"{title}", fontsize=14, fontweight="bold", pad=30)

        ax2 = fig.add_subplot(gs[1, i])
        draw_score_bar(ax2, score_df, hf.NWEA_ORDER)
        ax2.set_title("Avg RIT Score", fontsize=8, fontweight="bold", pad=10)

        # Collect data for analyzer
        try:
            _pct_payload.append({"subject": title, "data": pct_df.to_dict("records")})
            _score_payload.append({"subject": title, "data": score_df.to_dict("records")})
            _metrics_payload.append(metrics)
            _time_orders_payload.append(
                sorted(pct_df["time_label"].astype(str).unique().tolist())
                if "time_label" in pct_df.columns
                else []
            )
        except Exception:
            pass

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

    # Main title (avoid reassigning `district_label` inside this function)
    if school_raw:
        title_label = hf._safe_normalize_school_name(school_raw, cfg)
    else:
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

    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "nwea_eoy_section1_dual_subject_dashboard",
            "section": 1,
            "scope": scope_label,
            "window_filter": window_filter,
            "subjects": titles,
            "pct_data": _pct_payload,
            "score_data": _score_payload,
            "metrics": _metrics_payload,
            "time_orders": _time_orders_payload,
        },
    )
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
    nwea_base.copy(),
    window_filter="Spring",
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

        before_school = len(nwea_base)
        school_df = nwea_base[nwea_base["schoolname"] == raw_school].copy()
        logger.info(f"[FILTER] Section 1: After school filter '{raw_school}': {len(school_df):,} rows (removed {before_school - len(school_df):,})")

        plot_nwea_dual_subject_dashboard(
            school_df,
            window_filter="Spring",
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
    logger.info(f"[FILTER] plot_nwea_subject_dashboard_by_group: Starting | Group: {group_name} | Subject: {subject_str} | Window: {window_filter} | Scope: {scope_label} | Input rows: {len(d0):,}")

    # normalize school pretty label
    school_display = (
        hf._safe_normalize_school_name(school_raw, cfg) if school_raw else None
    )
    title_label = (
        district_all_students_label
        if not school_display
        else school_display
    )

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
        subj_df = filter_nwea_subject_rows(d0, subj)
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
            line_high = rf"$\Delta$ High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
            line_hiavg = rf"$\Delta$ Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
            line_low = rf"$\Delta$ Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
            line_rit = rf"$\Delta$ Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
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
        f"{title_label} • {group_name} • Spring Year-to-Year Trends",
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
    out_name = (
        f"{scope_label.replace(' ', '_')}_section2_{safe_group}_Spring_trends.png"
    )
    out_path = out_dir / out_name

    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "nwea_eoy_section2_student_group_dashboard",
            "section": 2,
            "scope": title_label,
            "scope_label": scope_label,
            "window_filter": window_filter,
            "group_name": group_name,
            "subjects": subject_titles,
            "pct_data": [
                {"subject": subject_titles[i], "data": pct_dfs[i].to_dict("records")}
                for i in range(len(subject_titles))
            ],
            "score_data": [
                {"subject": subject_titles[i], "data": score_dfs[i].to_dict("records")}
                for i in range(len(subject_titles))
            ],
            "metrics": metrics_list,
            "time_orders": time_orders,
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
# The runner passes selected groups as: NWEA_EOY_STUDENT_GROUPS="English Learners,Students with Disabilities"
_env_groups = os.getenv("NWEA_EOY_STUDENT_GROUPS")
_selected_groups = []
if _env_groups:
    _selected_groups = [g.strip() for g in str(_env_groups).split(",") if g.strip()]
    print(f"[FILTER] Student group selection from frontend: {_selected_groups}")

# Optional: restrict race/ethnicity dashboards based on frontend selection.
# The runner passes selected races as: NWEA_EOY_RACE="Hispanic or Latino,White"
_env_race = os.getenv("NWEA_EOY_RACE")
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
scope_df = nwea_base.copy()
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
        window_filter="Spring",
        group_name=group_name,
        group_def=group_def,
        figsize=(16, 9),
        school_raw=None,
        scope_label=scope_label,
    )

# Selected races (may not exist as keys in student_groups_cfg when options are dynamic)
if _selected_races:
    eth_col = _get_ethnicity_col(scope_df)
    if not eth_col:
        logger.info(
            "[CHART] Section 2: race filters provided but no ethnicity/race column found; skipping race charts"
        )
    else:
        for race_name in _selected_races:
            # Prefer config mapping if it exists (allows synonyms), otherwise exact match on column.
            mapped = student_groups_cfg.get(race_name) if isinstance(student_groups_cfg, dict) else None
            if isinstance(mapped, dict) and mapped.get("column") and mapped.get("in"):
                race_def = mapped
            else:
                race_def = {"column": eth_col, "in": [race_name]}
            plot_nwea_subject_dashboard_by_group(
                scope_df.copy(),
                subject_str=None,
                window_filter="Spring",
                group_name=race_name,
                group_def=race_def,
                figsize=(16, 9),
                school_raw=None,
                scope_label=scope_label,
            )

# ---- Site-level
if _include_school_charts():
    for raw_school in _iter_schools(nwea_base):
        before_school = len(nwea_base)
        scope_df = nwea_base[nwea_base["schoolname"] == raw_school].copy()
        logger.info(f"[FILTER] Section 2: After school filter '{raw_school}': {len(scope_df):,} rows (removed {before_school - len(scope_df):,})")
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
                window_filter="Spring",
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
                        window_filter="Spring",
                        group_name=race_name,
                        group_def=race_def,
                        figsize=(16, 9),
                        school_raw=raw_school,
                        scope_label=scope_label,
                    )


# %% SECTION 3 - Overall + Cohort Trends
# Cohort Dashboards (current-year cohort across up to 4 years)
# Model and styling match the main dashboard section. X-axis shows cohort labels.
# Example labels: "Gr 3 • Spring 24-25", "Gr 4 • Spring 25-26", "Gr 5 • Spring 26-27"
# Rules:
#   - Window: Spring only
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
    window_filter: str = "Spring",
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
    district_label = (
        district_all_students_label
        if not school_display
        else school_display
    )

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

            # Format time label as Winter YY-YY (e.g., Winter 25-26)
            year_str_prev = str(year - 1)[-2:]
            year_str_curr = str(year)[-2:]
            label_full = (
                f"Gr {int(grade)} \u2022 Spring {year_str_prev}-{year_str_curr}"
            )
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
        ["time_label", "achievementquintile"],
        as_index=False,
        observed=False,
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
        ax.set_xticklabels(stack_df.index.tolist(), fontsize=8)
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
    ax3.set_xticklabels(x_labels_left, ha="center", fontsize=8)
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
            cohort_df.groupby("time_label", observed=False)["N_total"]
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
    ax4.set_xticklabels(x_labels_right, ha="center", fontsize=8)
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
        line_high = rf"$\Delta$ High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
        line_hiavg = rf"$\Delta$ Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
        line_low = rf"$\Delta$ Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
        line_rit = rf"$\Delta$ Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
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
        line_high = rf"$\Delta$ High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
        line_hiavg = rf"$\Delta$ Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
        line_low = rf"$\Delta$ Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
        line_rit = rf"$\Delta$ Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
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
        district_label if school_raw is None else hf._safe_normalize_school_name(school_raw, cfg)
    )
    out_name = (
        f"{scope.replace(' ', '_')}_section3_grade{int(current_grade)}_"
        f"{course_str.lower().replace(' ', '_')}_Spring_trends.png"
    )
    out_path = out_dir / out_name
    logger.info(f"[CHART] Generating chart: {out_path}")
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        {
            "chart_type": "nwea_eoy_section3_blended_dashboard",
            "section": 3,
            "scope": scope,
            "scope_label": scope_label,
            "window_filter": window_filter,
            "grade": int(current_grade),
            "course_str": course_str,
            "subjects": [course_str_for_title],
            "pct_data": [
                {"subject": "overall", "data": pct_df_left.to_dict("records")},
                {"subject": "cohort", "data": pct_df_right.to_dict("records")},
            ],
            "score_data": [
                {"subject": "overall", "data": score_df_left.to_dict("records")},
                {"subject": "cohort", "data": score_df_right.to_dict("records")},
            ],
            "metrics": {"overall": metrics_left, "cohort": metrics_right},
            "time_orders": {"overall": time_order_left, "cohort": time_order_right},
        },
    )
    logger.info(f"[CHART] Saved: {out_path}")
    if preview:
        plt.show()
    print(f"Saved: {out_path}")


# ---- Combined DRIVER for Section 3 ----
_base = nwea_base.copy()
_base["year"] = pd.to_numeric(_base["year"], errors="coerce")
_base["grade"] = pd.to_numeric(_base["grade"], errors="coerce")


def _run_scope(scope_df, scope_label, school_raw):
    if scope_df["year"].notna().any():
        anchor_year = int(scope_df["year"].max())
    else:
        anchor_year = None
    for g in sorted(scope_df["grade"].dropna().unique()):
        try:
            g_int = int(g)
        except Exception:
            continue
        if _selected_grades is not None and g_int not in _selected_grades:
            continue
        subjects_to_generate = _requested_subjects(["Reading", "Mathematics"])
        for subject_str in subjects_to_generate:
            if filter_nwea_subject_rows(scope_df, subject_str).empty:
                logger.info(
                    f"[CHART] Section 3: skipping subject '{subject_str}' for Grade {g_int} (no matching rows)"
                )
                continue
            plot_nwea_blended_dashboard(
                scope_df.copy(),
                course_str=subject_str,
                current_grade=g_int,
                window_filter="Spring",
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
# Uses `falltoWinterconditionalgrowthpercentile` and `falltoWinterconditionalgrowthindex`
# Creates dual-panel dashboards for district + each site.
# ---------------------------------------------------------------------

def _pick_cond_growth_cols(
    df: pd.DataFrame, *, end_window: str | None = None
) -> tuple[str | None, str | None, str, str]:
    """
    Choose the best available Conditional Growth columns.

    Returns: (cgp_col, cgi_col, growth_label, inferred_end_window)
    """
    # Map normalized -> actual column names so we can safely index
    col_map: dict[str, str] = {}
    for c in df.columns:
        key = str(c).strip().lower()
        col_map.setdefault(key, c)
    cols = set(col_map.keys())
    end = (end_window or "").strip().upper()

    candidates = [
        # EOY always uses Spring-ended growth (no Winter fallback)
        ("wintertospringconditionalgrowthpercentile", "wintertospringconditionalgrowthindex", "Winter→Spring", "SPRING"),
        ("springtospringconditionalgrowthpercentile", "springtospringconditionalgrowthindex", "Spring→Spring", "SPRING"),
        ("falltospringconditionalgrowthpercentile", "falltospringconditionalgrowthindex", "Fall→Spring", "SPRING"),
    ]
    # No dynamic reordering for EOY - always Spring-only

    for cgp, cgi, label, inferred_end in candidates:
        if cgp in cols:
            cgp_actual = col_map[cgp]
            cgi_actual = col_map.get(cgi) if cgi in cols else None
            return cgp_actual, cgi_actual, label, inferred_end

    return None, None, "Conditional Growth", end or "WINTER"


def _prep_cgp_trend(df: pd.DataFrame, subject_str: str) -> pd.DataFrame:
    """
    Return tidy frame with columns:
        scope_label, time_label, median_cgp, mean_cgi
    Only Winter window. Subject filter matches dashboard logic.
    """

    d = df.copy()
    # For EOY charts, use the last selected/available window (typically SPRING)
    end_window = _get_eoy_compare_windows()[-1].upper()
    before_window = len(d)
    d = d[d["testwindow"].astype(str).str.upper() == end_window].copy()
    if d.empty:
        logger.info(
            f"[CHART] Section 4 CGP: no rows after window filter ({end_window}) for subject '{subject_str}' "
            f"(started with {before_window:,} row(s))"
        )
    d = filter_nwea_subject_rows(d, subject_str)

    cgp_col, cgi_col, _growth_label, _inferred_end = _pick_cond_growth_cols(
        d, end_window=end_window
    )
    if not cgp_col:
        # Helpful hint: show any available ConditionalGrowth-like columns
        cg_cols = sorted([c for c in d.columns if "conditional" in str(c).lower()])
        logger.warning(
            f"[CHART] Section 4 CGP: missing conditional growth columns for '{subject_str}' "
            f"(window={end_window}); found {len(cg_cols)} conditional-like col(s): {cg_cols[:8]}"
        )
        return pd.DataFrame(
            columns=["scope_label", "time_label", "median_cgp", "mean_cgi"]
        )

    before_nonnull = len(d)
    d = d[d[cgp_col].notna()].copy()
    if d.empty:
        logger.info(
            f"[CHART] Section 4 CGP: column '{cgp_col}' is all NULL after filters for '{subject_str}' "
            f"(window={end_window}; rows before non-null filter={before_nonnull:,})"
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
    has_cgi = bool(cgi_col and cgi_col in both.columns)
    grp_cols = ["site_display", "time_label"]

    if has_cgi:
        out = (
            both.groupby(grp_cols, dropna=False)
            .agg(
                median_cgp=(cgp_col, "median"),
                mean_cgi=(cgi_col, "mean"),
            )
            .reset_index()
        )
    else:
        out = (
            both.groupby(grp_cols, dropna=False)
            .agg(median_cgp=(cgp_col, "median"))
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

    ax.set_ylabel("Median Conditional Growth Percentile (CGP)")
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

    ax2.set_ylabel("Avg Conditional Growth Index (CGI)")
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax.set_title(f"{subject_str}", fontweight="bold", fontsize=14, pad=10)


# ---------------------------------------------------------------------
# Standardized Save + Driver Logic
# ---------------------------------------------------------------------


def _save_cgp_chart(
    fig,
    scope_label,
    section_num=4,
    suffix="cgp_cgi_dualpanel",
    chart_data: dict | None = None,
):
    charts_dir = CHARTS_DIR
    folder_name = (
        "_district"
        if _is_district_scope(scope_label)
        else scope_label.replace(" ", "_")
    )
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section{section_num}_{suffix}.png"
    out_path = out_dir / out_name
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        chart_data
        or {
            "chart_type": "nwea_eoy_cgp_chart",
            "section": section_num,
            "scope": scope_label,
            "suffix": suffix,
        },
    )
    print(f"Saved: {out_path}")
    return out_path


def _run_cgp_dual_trend(scope_df, scope_label):
    # This chart started as a 2-column layout (Reading + Math). In EOY mode we allow
    # generating a 1-column version if the frontend only selected one subject.
    subjects_for_cgp = ["Reading", "Mathematics"]
    if _selected_subjects:
        subj_join = " | ".join(str(s).casefold() for s in _selected_subjects)
        has_read = any(k in subj_join for k in ["reading", "ela", "language arts"])
        has_math = "math" in subj_join
        if has_read and not has_math:
            subjects_for_cgp = ["Reading"]
        elif has_math and not has_read:
            subjects_for_cgp = ["Mathematics"]
        elif has_math and has_read:
            subjects_for_cgp = ["Reading", "Mathematics"]
        # else: keep default
    cgp_trend = pd.concat(
        [_prep_cgp_trend(scope_df, subj) for subj in subjects_for_cgp],
        ignore_index=True,
    )
    if cgp_trend.empty:
        logger.info(
            f"[CHART] Section 4 CGP: skipping for {scope_label} (no CGP data after filters; "
            f"end_window={_get_eoy_compare_windows()[-1].upper()})"
        )
        return

    fig = plt.figure(figsize=(16, 9), dpi=300)
    ncols = max(1, len(subjects_for_cgp))
    gs = fig.add_gridspec(nrows=3, ncols=ncols, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.suptitle(
        f"{scope_label} • Conditional Growth (All Students)",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )

    axes = []
    n_labels_axes = []
    end_window = _get_eoy_compare_windows()[-1].upper()
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
        d = d[d["testwindow"].astype(str).str.upper() == end_window].copy()
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
        if _is_district_scope(scope_label):
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

    _save_cgp_chart(
        fig,
        scope_label,
        section_num=4,
        suffix="cgp_cgi_dualpanel",
        chart_data={
            "chart_type": "nwea_eoy_section4_cgp_cgi_dualpanel",
            "section": 4,
            "scope": scope_label,
            "window_filter": end_window.title(),
            "subjects": subjects_for_cgp,
            # Analyzer will include this under sgp_data to avoid requiring custom keys.
            "sgp_data": cgp_trend.to_dict("records"),
        },
    )


# ---------------------------------------------------------------------
# DRIVER — District + School CGP Dual-Panel Dashboards
# ---------------------------------------------------------------------

_run_cgp_dual_trend(nwea_base.copy(), district_label)

if _include_school_charts():
    for raw_school in _iter_schools(nwea_base):
        before_school = len(nwea_base)
        scope_df = nwea_base[nwea_base["schoolname"] == raw_school].copy()
        logger.info(f"[FILTER] Section 4: After school filter '{raw_school}': {len(scope_df):,} rows (removed {before_school - len(scope_df):,})")
        scope_label = hf._safe_normalize_school_name(raw_school, cfg)
        _run_cgp_dual_trend(scope_df, scope_label)


# %% SECTION 5 — CGP/CGI Growth: Grade Trend + Backward Cohort (Unmatched)
# ---------------------------------------------------------------------
# Left facet: overall grade-level conditional growth across 4 years (EOY end window)
# Right facet: same grade, same time span, but backward cohort (unmatched)
#   → Year = anchor_year - offset
#   → Grade = anchor_grade - offset
#   → Dedupe by test date, filter to end window (typically SPRING)
#   → Median CGP + Mean CGI
# Replicates verified Sect‑2 cohort logic (no student matching)
# ---------------------------------------------------------------------
def _prep_cgp_by_grade(df, subject, grade):
    logger.info(f"[FILTER] _prep_cgp_by_grade: Starting | Subject: {subject} | Grade: {grade} | Input rows: {len(df):,}")
    # Use provided df (districtwide or school-filtered), not the global base.
    d = df.copy()
    initial_rows = len(d)
    end_window = _get_eoy_compare_windows()[-1].upper()
    d = d[d["testwindow"].str.upper() == end_window]
    after_window = len(d)
    logger.info(
        f"[FILTER] After end-window filter ({end_window}): {after_window:,} rows (removed {initial_rows - after_window:,})"
    )
    d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
    before_grade = len(d)
    d = d[d["grade"] == grade]
    logger.info(f"[FILTER] After grade {grade} filter: {len(d):,} rows (removed {before_grade - len(d):,})")

    before_course = len(d)
    subj = subject.lower()
    if "math" in subj:
        d = d[d["course"].str.contains("math", case=False, na=False)]
        logger.info(f"[FILTER] After Math course filter: {len(d):,} rows (removed {before_course - len(d):,})")
    else:
        d = d[d["course"].str.contains("read", case=False, na=False)]
        logger.info(f"[FILTER] After Reading course filter: {len(d):,} rows (removed {before_course - len(d):,})")

    # Clean dates and deduplicate to most recent end-window test per student
    before_dedup = len(d)
    d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    d = d.dropna(subset=["teststartdate"])
    d = d.sort_values("teststartdate").drop_duplicates("uniqueidentifier", keep="last")
    after_dedup = len(d)
    logger.info(f"[FILTER] After deduplication (latest per student): {after_dedup:,} rows (removed {before_dedup - after_dedup:,} duplicates)")

    # Only keep students with both values present
    before_cgp = len(d)
    cgp_col, cgi_col, _growth_label, _inferred_end = _pick_cond_growth_cols(
        d, end_window=end_window
    )
    if not cgp_col:
        cg_cols = sorted([c for c in d.columns if "conditional" in str(c).lower()])
        logger.warning(
            f"[CHART] Section 5 CGP: missing conditional growth columns (grade={grade}, subject={subject}, window={end_window}); "
            f"found {len(cg_cols)} conditional-like col(s): {cg_cols[:8]}"
        )
        return pd.DataFrame(columns=["time_label", "median_cgp", "mean_cgi"])

    subset_cols = [cgp_col] + ([cgi_col] if cgi_col else [])
    d = d.dropna(subset=subset_cols)
    logger.info(f"[FILTER] After CGP/CGI data filter (removed nulls): {len(d):,} rows (removed {before_cgp - len(d):,})")

    if d.empty:
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
        "Gr "
        + d["grade"].astype(int).astype(str)
        + f" \u2022 {end_window.title()} "
        + d["year_short"]
    )

    if cgi_col:
        out = (
            d.groupby("time_label")
            .agg(
                median_cgp=(cgp_col, "median"),
                mean_cgi=(cgi_col, "mean"),
            )
            .reset_index()
        )
    else:
        out = (
            d.groupby("time_label")
            .agg(median_cgp=(cgp_col, "median"))
            .reset_index()
        )
        out["mean_cgi"] = np.nan

    # --- Add year_short column to out for sorting ---
    if "year_short" not in out.columns and "time_label" in out.columns:
        # Extract year_short from the time_label string
        # "Gr {grade} • {WINDOW} {yy-yy}"
        def _extract_year_short(label):
            try:
                return label.split(end_window.title())[-1].strip()
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
        # Use global nwea_base; filter to Winter, subject, and grade for left; for right, filter to Winter, subject, grade/yr
        try:
            # Try to infer grade from time_label if possible
            import re

            # Figure out subject and grade from df/time_label
            # Example: "Gr 4 • Winter 23-24"
            time_labels = df["time_label"].astype(str).tolist()
            # We'll try to extract grade and year_short from each label
            n_dict = {}
            end_window = _get_eoy_compare_windows()[-1].upper()
            for lbl in time_labels:
                m = re.match(
                    rf"Gr (\d+) *• *{re.escape(end_window.title())} (\d{{2}}-\d{{2}})",
                    lbl,
                )
                if m:
                    gr = int(m.group(1))
                    yy = m.group(2)
                    # find rows in nwea_base with grade==gr, year_short==yy, end window, and subject
                    d = nwea_base.copy()
                    d = d[d["testwindow"].str.upper() == end_window]
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
                    # Only count students with Conditional Growth present (prefer W→S or F→S for EOY)
                    cgp_col, cgi_col, _growth_label, _inferred_end = _pick_cond_growth_cols(
                        d, end_window=end_window
                    )
                    if cgp_col:
                        subset_cols = [cgp_col] + ([cgi_col] if cgi_col else [])
                        d = d.dropna(subset=subset_cols)
                    n_dict[lbl] = d["uniqueidentifier"].nunique()
                else:
                    n_dict[lbl] = 0
            labels_with_n = [
                f"{lbl}\n(n = {int(n_dict.get(lbl,0))})" for lbl in time_labels
            ]
        except Exception:
            # fallback: just use labels
            labels_with_n = df["time_label"].astype(str).tolist()

        ax.set_ylabel(f"Median Conditional Growth Percentile ({end_window.title()} window)")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels_with_n, ha="center", fontsize=8)
        ax.tick_params(axis="x", pad=10)
        ax.set_ylim(0, 100)

        ax2 = ax.twinx()
        ax2.set_ylim(-2.5, 2.5)
        ax2.set_ylabel("Avg Conditional Growth Index")
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
        f"{scope_label} • {subject_str} • Grade {grade} • Conditional Growth",
        fontsize=20,
        fontweight="bold",
        y=1,
    )

    _save_cgp_chart(
        fig,
        scope_label,
        section_num=5,
        suffix=f"cgp_cgi_grade_trends_grade{grade}_{subject_str.lower().replace(' ', '_')}",
        chart_data={
            "chart_type": "nwea_eoy_section5_cgp_cgi_grade_and_cohort",
            "section": 5,
            "scope": scope_label,
            "subject": subject_str,
            "grade": int(grade),
            "window_filter": _get_eoy_compare_windows()[-1].title(),
            "grade_data": overall_df.to_dict("records") if hasattr(overall_df, "to_dict") else [],
            "cohort_data": cohort_df.to_dict("records") if hasattr(cohort_df, "to_dict") else [],
        },
    )
    if preview:
        plt.show()


# SECTION 5 DRIVER — Districtwide
district_display = district_label
d0 = nwea_base.copy()
d0["year"] = pd.to_numeric(d0["year"], errors="coerce")
d0["grade"] = pd.to_numeric(d0["grade"], errors="coerce")
grades = sorted(d0["grade"].dropna().unique())
if _selected_grades is not None:
    grades = [g for g in grades if int(g) in _selected_grades]
subjects = _requested_core_subjects()
preview = False  # or True for interactive preview

for grade in grades:
    try:
        grade = int(grade)
    except Exception:
        continue
    if _selected_grades is not None and int(grade) not in _selected_grades:
        continue
    for subject in subjects:
        overall_df = _prep_cgp_by_grade(d0, subject, grade)
        if overall_df.empty:
            continue
        anchor_year = int(d0[d0["grade"] == grade]["year"].max())
        end_window = _get_eoy_compare_windows()[-1].upper()
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
                & (d["testwindow"].str.upper() == end_window)
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
            cgp_col, cgi_col, _growth_label, _inferred_end = _pick_cond_growth_cols(
                d, end_window=end_window
            )
            if not cgp_col:
                continue
            subset_cols = [cgp_col] + ([cgi_col] if cgi_col else [])
            d = d.dropna(subset=subset_cols)
            if d.empty:
                continue
            cohort_rows.append(
                {
                    "gr": gr,
                    "yr": yr,
                    "time_label": f"Gr {int(gr)} • {end_window.title()} {str(yr - 1)[-2:]}-{str(yr)[-2:]}",
                    "median_cgp": d[cgp_col].median(),
                    "mean_cgi": (d[cgi_col].mean() if cgi_col else np.nan),
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
if _selected_grades is not None:
    grades = [g for g in grades if int(g) in _selected_grades]
subjects = _requested_core_subjects()
preview = False  # set True if preview needed

for raw_school in all_schools:
    scope_label = hf._safe_normalize_school_name(raw_school, cfg)
    before_school = len(nwea_base)
    d0 = nwea_base[nwea_base["schoolname"] == raw_school].copy()
    logger.info(f"[FILTER] Section 5: After school filter '{raw_school}': {len(d0):,} rows (removed {before_school - len(d0):,})")
    d0["year"] = pd.to_numeric(d0["year"], errors="coerce")
    d0["grade"] = pd.to_numeric(d0["grade"], errors="coerce")
    end_window = _get_eoy_compare_windows()[-1].upper()
    for grade in grades:
        try:
            grade = int(grade)
        except Exception:
            continue
        if _selected_grades is not None and int(grade) not in _selected_grades:
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
                    & (d["testwindow"].str.upper() == end_window)
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
                cgp_col, cgi_col, _growth_label, _inferred_end = _pick_cond_growth_cols(
                    d, end_window=end_window
                )
                if not cgp_col:
                    continue
                subset_cols = [cgp_col] + ([cgi_col] if cgi_col else [])
                d = d.dropna(subset=subset_cols)
                if d.empty:
                    continue
                cohort_rows.append(
                    {
                        "gr": gr,
                        "yr": yr,
                        "time_label": f"Gr {int(gr)} • {end_window.title()} {str(yr - 1)[-2:]}-{str(yr)[-2:]}",
                        "median_cgp": d[cgp_col].median(),
                        "mean_cgi": (d[cgi_col].mean() if cgi_col else np.nan),
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
    fig,
    scope_label,
    section_num=4,
    suffix="cgp_cgi_dualpanel",
    chart_data: dict | None = None,
):
    charts_dir = CHARTS_DIR
    folder_name = (
        "_district"
        if _is_district_scope(scope_label)
        else scope_label.replace(" ", "_")
    )
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section{section_num}_{suffix}.png"
    out_path = out_dir / out_name
    hf._save_and_render(fig, out_path)
    _write_chart_data(
        out_path,
        chart_data
        or {
            "chart_type": "nwea_eoy_cgp_chart",
            "section": section_num,
            "scope": scope_label,
            "suffix": suffix,
        },
    )
    print(f"Saved: {out_path}")
    return out_path

    # %%
#########################################################################
#
# Charts for District-level runs to include all schools on the same chart
# and all grade levels on the same chart
#
#########################################################################


# %% SECTION 6 — District Window Compare by School (ELA + Math)
# ---------------------------------------------------------------------
# Multi-indexed version of Section 0.1:
#   - Latest year only
#   - X-axis = schools
#   - For each school: first and last selected windows 100% stacked bars side-by-side
#   - Produce TWO full-size charts: one for Reading (ELA proxy) and one for Math
# ---------------------------------------------------------------------


def _prep_section6_window_compare_by_school(
    df: pd.DataFrame,
    subject_str: str,
) -> tuple[pd.DataFrame, int, list[str]]:
    """Prepare percent-by-quintile for selected windows, split by school.

    Returns:
        pct_df: columns [school_display, testwindow, achievementquintile, pct, n, N_total]
        target_year: int
        school_order: list of school_display in plotted order
    """

    d = df.copy()

    win_order = _get_eoy_compare_windows()
    # 1) Restrict to requested windows
    d["testwindow"] = d["testwindow"].astype(str)
    d = d[d["testwindow"].str.upper().isin(win_order)].copy()
    if d.empty:
        return pd.DataFrame(), -1, []

    # 2) Subject filter (mirror Section 0.1 semantics)
    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[d["course"].astype(str).str.contains("math", case=False, na=False)].copy()
    else:
        # Reading / ELA proxy
        d = d[
            d["course"].astype(str).str.contains("reading", case=False, na=False)
        ].copy()

    if d.empty:
        return pd.DataFrame(), -1, []

    # 3) Require valid quintile bucket
    d = d[d["achievementquintile"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), -1, []

    # 4) Latest year only
    d["year_num"] = pd.to_numeric(d["year"].astype(str).str[:4], errors="coerce")
    d = d[d["year_num"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), -1, []

    target_year = int(d["year_num"].max())
    d = d[d["year_num"] == target_year].copy()
    if d.empty:
        return pd.DataFrame(), target_year, []

    # 5) Normalize display school name (use helper for consistency)
    # Keep raw schoolname for grouping, but add a display field for labels.
    if "schoolname" in d.columns:
        d["school_display"] = d["schoolname"].apply(
            lambda s: hf._safe_normalize_school_name(s, cfg)
        )
    else:
        d["school_display"] = "(No School)"

    # 6) Dedupe to latest attempt per student per (school, window)
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    else:
        d["teststartdate"] = pd.NaT

    # Use uniqueidentifier when present; fallback to studentid if needed
    id_col = (
        "uniqueidentifier"
        if "uniqueidentifier" in d.columns
        else ("studentid" if "studentid" in d.columns else None)
    )
    if id_col is None:
        # No stable id; proceed without dedupe (still better than failing)
        id_col = "__row_id__"
        d[id_col] = np.arange(len(d))

    d.sort_values(
        ["school_display", "testwindow", id_col, "teststartdate"], inplace=True
    )
    d = d.groupby(["school_display", "testwindow", id_col], as_index=False).tail(1)

    # 7) Percent by quintile within each (school, window)
    quint_counts = (
        d.groupby(["school_display", "testwindow", "achievementquintile"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_counts = (
        d.groupby(["school_display", "testwindow"])
        .size()
        .rename("N_total")
        .reset_index()
    )

    pct_df = quint_counts.merge(
        total_counts, on=["school_display", "testwindow"], how="left"
    )
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]

    # 8) Ensure all quintiles exist for each (school, window)
    # Normalize window labels to FALL/WINTER for ordering
    pct_df["testwindow"] = pct_df["testwindow"].astype(str).str.upper()

    school_order = sorted(pct_df["school_display"].dropna().unique().tolist())
    window_order = win_order

    all_idx = pd.MultiIndex.from_product(
        [school_order, window_order, hf.NWEA_ORDER],
        names=["school_display", "testwindow", "achievementquintile"],
    )

    pct_df = (
        pct_df.set_index(["school_display", "testwindow", "achievementquintile"])
        .reindex(all_idx)
        .reset_index()
    )

    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)

    # Fill N_total within each (school, window)
    pct_df["N_total"] = pct_df.groupby(["school_display", "testwindow"])[
        "N_total"
    ].transform(lambda s: s.ffill().bfill())

    return pct_df, target_year, school_order


def _plot_section6_window_compare_by_school(
    pct_df: pd.DataFrame,
    subject_title: str,
    target_year: int,
    school_order: list[str],
    preview: bool = False,
):
    """Plot one full-size district chart for a single subject."""

    if pct_df.empty or not school_order:
        print(f"[Section 6] Skipped {subject_title} (no data)")
        return

    window_order = _get_eoy_compare_windows()

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    # Positions: grouped by school; windows side-by-side
    x = np.arange(len(school_order))
    n_w = len(window_order)
    if n_w == 2:
        w = 0.36
        offsets = {window_order[0]: -w / 2, window_order[1]: w / 2}
    elif n_w == 3:
        w = 0.26
        offsets = {window_order[0]: -w, window_order[1]: 0.0, window_order[2]: w}
    else:
        w = max(0.18, 0.80 / max(1, n_w))
        spread = np.linspace(-0.30, 0.30, max(1, n_w))
        offsets = {window_order[i]: float(spread[i]) for i in range(n_w)}

    # Pivot to (school, window) x quintile for fast lookup
    pivot = (
        pct_df.pivot_table(
            index=["school_display", "testwindow"],
            columns="achievementquintile",
            values="pct",
            aggfunc="sum",
        )
        .reindex(columns=hf.NWEA_ORDER)
        .fillna(0)
    )

    # Stack each window separately
    for window in window_order:
        cumulative = np.zeros(len(school_order))
        for cat in hf.NWEA_ORDER:
            vals = []
            for s in school_order:
                try:
                    vals.append(float(pivot.loc[(s, window), cat]))
                except Exception:
                    vals.append(0.0)
            vals = np.array(vals, dtype=float)

            bars = ax.bar(
                x + offsets[window],
                vals,
                bottom=cumulative,
                width=w,
                color=hf.NWEA_COLORS[cat],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.6 if window != window_order[-1] else 1.0,
            )

            # Inline % labels
            for i_bar, rect in enumerate(bars):
                h = float(vals[i_bar])
                if h >= LABEL_MIN_PCT:
                    bottom_before = float(cumulative[i_bar])
                    if cat in ("High", "HiAvg"):
                        label_color = "white"
                    elif cat in ("Avg", "LoAvg"):
                        label_color = "#434343"
                    elif cat == "Low":
                        label_color = "white"
                    else:
                        label_color = "#434343"
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        bottom_before + h / 2,
                        f"{h:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color=label_color,
                    )

            cumulative += vals

    # Axes styling
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of Students")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # School labels at group centers (with n-counts per window)
    n_map = (
        pct_df.groupby(["school_display", "testwindow"])["N_total"]
        .max()
        .dropna()
        .astype(int)
        .to_dict()
    )

    x_labels = []
    short = {"FALL": "F", "WINTER": "W", "SPRING": "S"}
    for s in school_order:
        parts = []
        for w_key in window_order:
            nval = int(n_map.get((s, w_key), 0) or 0)
            parts.append(f"{short.get(w_key, w_key[:1])} n={nval}")
        x_labels.append(f"{s}\n(" + " | ".join(parts) + ")")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")

    # Add small window tags ABOVE each group
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for xi in x:
        for w_key in window_order:
            ax.text(
                xi + offsets[w_key],
                1.01,
                w_key.title(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#434343",
                transform=trans,
            )

    # Legend (shared)
    legend_handles = [
        Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q)
        for q in hf.NWEA_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        labels=hf.NWEA_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=len(hf.NWEA_ORDER),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    district_name_cfg = district_label
    ax.set_title(
        f"{district_name_cfg} • {window_order[0].title()} vs {window_order[-1].title()} by School \n {subject_title} {window_order[-1].title()} {target_year}",
        fontsize=18,
        fontweight="bold",
        pad=22,
        y=1.05,
    )

    # Save to district folder (always use CHARTS_DIR so the runner can collect outputs)
    charts_dir = CHARTS_DIR
    out_dir = charts_dir / "_district"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_subj = subject_title.replace(" ", "_").replace("/", "_")
    out_path = (
        out_dir
        / f"{district_name_cfg.replace(' ', '_')}_section6_window_compare_by_school_{safe_subj}_eoy.png"
    )

    hf._save_and_render(fig, out_path)
    print(f"Saved Section 6 ({subject_title}): {out_path}")

    if preview:
        plt.show()
    plt.close(fig)


# ---- RUN SECTION 6 (district only) ----
try:
    district_label_06 = district_label

    # Reading / ELA
    pct_ela, year_ela, schools_ela = _prep_section6_window_compare_by_school(
        nwea_base.copy(),
        subject_str="Reading",
    )
    _plot_section6_window_compare_by_school(
        pct_ela,
        subject_title="Reading",
        target_year=year_ela,
        school_order=schools_ela,
        preview=False,
    )

    # Math
    pct_math, year_math, schools_math = _prep_section6_window_compare_by_school(
        nwea_base.copy(),
        subject_str="Math",
    )
    _plot_section6_window_compare_by_school(
        pct_math,
        subject_title="Math",
        target_year=year_math,
        school_order=schools_math,
        preview=False,
    )
except Exception as e:
    print(f"[Section 6] ERROR: {e}")


# %% SECTION 7 — District Window Compare by Grade (ELA + Math)
# ---------------------------------------------------------------------
# Same concept as Section 6, but:
#   - X-axis = grade levels (districtwide)
#   - For each grade: first and last selected windows 100% stacked bars side-by-side
#   - Produce TWO full-size charts: one for Reading (ELA proxy) and one for Math
# ---------------------------------------------------------------------


def _prep_section7_window_compare_by_grade(
    df: pd.DataFrame,
    subject_str: str,
) -> tuple[pd.DataFrame, int, list[int]]:
    """Prepare percent-by-quintile for selected windows, split by grade.

    Returns:
        pct_df: columns [grade_num, testwindow, achievementquintile, pct, n, N_total]
        target_year: int
        grade_order: list of grade_num in plotted order
    """

    d = df.copy()

    win_order = _get_eoy_compare_windows()
    # 1) Restrict to requested windows
    d["testwindow"] = d["testwindow"].astype(str)
    d = d[d["testwindow"].str.upper().isin(win_order)].copy()
    if d.empty:
        return pd.DataFrame(), -1, []

    # 2) Subject filter (mirror Section 0.1 semantics)
    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[d["course"].astype(str).str.contains("math", case=False, na=False)].copy()
    else:
        # Reading / ELA proxy
        d = d[
            d["course"].astype(str).str.contains("reading", case=False, na=False)
        ].copy()

    if d.empty:
        return pd.DataFrame(), -1, []

    # 3) Require valid quintile bucket
    d = d[d["achievementquintile"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), -1, []

    # 4) Latest year only
    d["year_num"] = pd.to_numeric(d["year"].astype(str).str[:4], errors="coerce")
    d = d[d["year_num"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), -1, []

    target_year = int(d["year_num"].max())
    d = d[d["year_num"] == target_year].copy()
    if d.empty:
        return pd.DataFrame(), target_year, []

    # 5) Grade normalization
    # Prefer `grade` (used throughout this script); fallback to `student_grade` if needed.
    grade_col = (
        "grade"
        if "grade" in d.columns
        else ("student_grade" if "student_grade" in d.columns else None)
    )
    if grade_col is None:
        return pd.DataFrame(), target_year, []

    d["grade_num"] = pd.to_numeric(d[grade_col], errors="coerce")
    d = d[d["grade_num"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), target_year, []

    # Cast to int-like for grouping/ordering (keeps K=0 intact)
    d["grade_num"] = d["grade_num"].astype(int)

    # 6) Dedupe to latest attempt per student per (grade, window)
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    else:
        d["teststartdate"] = pd.NaT

    id_col = (
        "uniqueidentifier"
        if "uniqueidentifier" in d.columns
        else ("studentid" if "studentid" in d.columns else None)
    )
    if id_col is None:
        id_col = "__row_id__"
        d[id_col] = np.arange(len(d))

    d.sort_values(["grade_num", "testwindow", id_col, "teststartdate"], inplace=True)
    d = d.groupby(["grade_num", "testwindow", id_col], as_index=False).tail(1)

    # 7) Percent by quintile within each (grade, window)
    quint_counts = (
        d.groupby(["grade_num", "testwindow", "achievementquintile"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_counts = (
        d.groupby(["grade_num", "testwindow"]).size().rename("N_total").reset_index()
    )

    pct_df = quint_counts.merge(
        total_counts, on=["grade_num", "testwindow"], how="left"
    )
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]

    # 8) Ensure all quintiles exist for each (grade, window)
    pct_df["testwindow"] = pct_df["testwindow"].astype(str).str.upper()

    grade_order = sorted(pct_df["grade_num"].dropna().unique().tolist())
    window_order = win_order

    all_idx = pd.MultiIndex.from_product(
        [grade_order, window_order, hf.NWEA_ORDER],
        names=["grade_num", "testwindow", "achievementquintile"],
    )

    pct_df = (
        pct_df.set_index(["grade_num", "testwindow", "achievementquintile"])
        .reindex(all_idx)
        .reset_index()
    )

    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df.groupby(["grade_num", "testwindow"])[
        "N_total"
    ].transform(lambda s: s.ffill().bfill())

    return pct_df, target_year, grade_order


def _plot_section7_window_compare_by_grade(
    pct_df: pd.DataFrame,
    subject_title: str,
    target_year: int,
    grade_order: list[int],
    preview: bool = False,
):
    """Plot one full-size district chart for a single subject."""

    if pct_df.empty or not grade_order:
        print(f"[Section 7] Skipped {subject_title} (no data)")
        return

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    window_order = _get_eoy_compare_windows()
    # Positions: grouped by grade; windows side-by-side
    x = np.arange(len(grade_order))
    n_w = len(window_order)
    if n_w == 2:
        w = 0.36
        offsets = {window_order[0]: -w / 2, window_order[1]: w / 2}
    elif n_w == 3:
        w = 0.26
        offsets = {window_order[0]: -w, window_order[1]: 0.0, window_order[2]: w}
    else:
        w = max(0.18, 0.80 / max(1, n_w))
        spread = np.linspace(-0.30, 0.30, max(1, n_w))
        offsets = {window_order[i]: float(spread[i]) for i in range(n_w)}

    pivot = (
        pct_df.pivot_table(
            index=["grade_num", "testwindow"],
            columns="achievementquintile",
            values="pct",
            aggfunc="sum",
        )
        .reindex(columns=hf.NWEA_ORDER)
        .fillna(0)
    )

    for window in window_order:
        cumulative = np.zeros(len(grade_order))
        for cat in hf.NWEA_ORDER:
            vals = []
            for g in grade_order:
                try:
                    vals.append(float(pivot.loc[(g, window), cat]))
                except Exception:
                    vals.append(0.0)
            vals = np.array(vals, dtype=float)

            bars = ax.bar(
                x + offsets[window],
                vals,
                bottom=cumulative,
                width=w,
                color=hf.NWEA_COLORS[cat],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.6 if window != window_order[-1] else 1.0,
            )

            for i_bar, rect in enumerate(bars):
                h = float(vals[i_bar])
                if h >= LABEL_MIN_PCT:
                    bottom_before = float(cumulative[i_bar])
                    if cat in ("High", "HiAvg"):
                        label_color = "white"
                    elif cat in ("Avg", "LoAvg"):
                        label_color = "#434343"
                    elif cat == "Low":
                        label_color = "white"
                    else:
                        label_color = "#434343"
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        bottom_before + h / 2,
                        f"{h:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color=label_color,
                    )

            cumulative += vals

    # Axes styling
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of Students")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grade labels at group centers
    def _grade_label(g: int) -> str:
        return "K" if int(g) == 0 else str(int(g))

    # Grade labels at group centers (with n-counts per window)
    n_map = (
        pct_df.groupby(["grade_num", "testwindow"])["N_total"]
        .max()
        .dropna()
        .astype(int)
        .to_dict()
    )

    x_labels = []
    short = {"FALL": "F", "WINTER": "W", "SPRING": "S"}
    for g in grade_order:
        parts = []
        for w_key in window_order:
            nval = int(n_map.get((g, w_key), 0) or 0)
            parts.append(f"{short.get(w_key, w_key[:1])} n={nval}")
        x_labels.append(f"{_grade_label(g)}\n(" + " | ".join(parts) + ")")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")

    # Add small window tags ABOVE each group
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for xi in x:
        for w_key in window_order:
            ax.text(
                xi + offsets[w_key],
                1.01,
                w_key.title(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#434343",
                transform=trans,
            )

    # Legend
    legend_handles = [
        Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q)
        for q in hf.NWEA_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        labels=hf.NWEA_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=len(hf.NWEA_ORDER),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    district_name_cfg = district_label
    ax.set_title(
        f"{district_name_cfg} • {window_order[0].title()} vs {window_order[-1].title()} by Grade\n{subject_title} {window_order[-1].title()} {target_year}",
        fontsize=18,
        fontweight="bold",
        pad=22,
        y=1.05,
    )

    # Save to district folder (always use CHARTS_DIR so the runner can collect outputs)
    charts_dir = CHARTS_DIR
    out_dir = charts_dir / "_district"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_subj = subject_title.replace(" ", "_").replace("/", "_")
    out_path = (
        out_dir
        / f"{district_name_cfg.replace(' ', '_')}_section7_window_compare_by_grade_{safe_subj}_eoy.png"
    )

    hf._save_and_render(fig, out_path)
    print(f"Saved Section 7 ({subject_title}): {out_path}")

    if preview:
        plt.show()
    plt.close(fig)


# ---- RUN SECTION 7 (district only) ----
try:
    # Reading / ELA
    pct_ela_g, year_ela_g, grades_ela = _prep_section7_window_compare_by_grade(
        nwea_base.copy(),
        subject_str="Reading",
    )
    _plot_section7_window_compare_by_grade(
        pct_ela_g,
        subject_title="Reading",
        target_year=year_ela_g,
        grade_order=grades_ela,
        preview=False,
    )

    # Math
    pct_math_g, year_math_g, grades_math = _prep_section7_window_compare_by_grade(
        nwea_base.copy(),
        subject_str="Math",
    )
    _plot_section7_window_compare_by_grade(
        pct_math_g,
        subject_title="Math",
        target_year=year_math_g,
        grade_order=grades_math,
        preview=False,
    )
except Exception as e:
    print(f"[Section 7] ERROR: {e}")

# %% SECTION 8 — District Window Compare by Student Group (ELA + Math)
# ---------------------------------------------------------------------
# X-axis = student groups (districtwide). For each group: Fall + Winter bars.
# Uses cfg['student_groups'] + _apply_student_group_mask for membership.
# ---------------------------------------------------------------------


def _prep_section8_window_compare_by_student_group(df: pd.DataFrame, subject_str: str):
    d = df.copy()

    win_order = _get_eoy_compare_windows()
    d["testwindow"] = d["testwindow"].astype(str)
    d = d[d["testwindow"].str.upper().isin(win_order)].copy()
    if d.empty:
        return pd.DataFrame(), -1, []

    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[d["course"].astype(str).str.contains("math", case=False, na=False)].copy()
    else:
        d = d[
            d["course"].astype(str).str.contains("reading", case=False, na=False)
        ].copy()

    d = d[d["achievementquintile"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), -1, []

    d["year_num"] = pd.to_numeric(d["year"].astype(str).str[:4], errors="coerce")
    d = d[d["year_num"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), -1, []

    target_year = int(d["year_num"].max())
    d = d[d["year_num"] == target_year].copy()
    if d.empty:
        return pd.DataFrame(), target_year, []

    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    else:
        d["teststartdate"] = pd.NaT

    id_col = (
        "uniqueidentifier"
        if "uniqueidentifier" in d.columns
        else ("studentid" if "studentid" in d.columns else None)
    )
    if id_col is None:
        id_col = "__row_id__"
        d[id_col] = np.arange(len(d))

    d.sort_values(["testwindow", id_col, "teststartdate"], inplace=True)
    d = d.groupby(["testwindow", id_col], as_index=False).tail(1)

    student_groups_cfg = cfg.get("student_groups", {})
    group_order_map = cfg.get("student_group_order", {})

    # --- Section 8: Choose which student groups to include ---
    # Prefer frontend selections if provided (env-driven), else use a sensible default set.
    DEFAULT_GROUPS = [
        "All Students",
        "Socioeconomically Disadvantaged",
        "Students with Disabilities",
        "English Learners",
        "Hispanic",
        "White",
    ]
    enabled_set = set(DEFAULT_GROUPS)
    try:
        if _selected_groups:
            enabled_set = set(["All Students"] + [str(g) for g in _selected_groups])
    except Exception:
        pass

    frames = []

    # All Students
    if enabled_set is None or "All Students" in enabled_set:
        all_mask = _apply_student_group_mask(d, "All Students", {"type": "all"})
        if all_mask is not None:
            d_all = d[all_mask].copy()
            if not d_all.empty:
                frames.append(d_all.assign(student_group="All Students"))

    # Configured groups
    for group_name, group_def in student_groups_cfg.items():
        if group_def.get("type") == "all":
            continue

        # Respect enabled list (if provided)
        if enabled_set is not None and group_name not in enabled_set:
            continue

        try:
            mask = _apply_student_group_mask(d, group_name, group_def)
        except Exception:
            continue
        dg = d[mask].copy()
        if dg.empty:
            continue
        frames.append(dg.assign(student_group=group_name))

    if not frames:
        return pd.DataFrame(), target_year, []

    d2 = pd.concat(frames, ignore_index=True)

    # --- Filter to groups with sufficient n (more than 11 students) ---
    # Keep groups with >11 students in at least one selected window (e.g., Winter OR Spring).
    _counts = (
        d2.groupby(["student_group", "testwindow"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    _counts["testwindow"] = _counts["testwindow"].astype(str).str.upper()
    _pivot = _counts.pivot_table(
        index="student_group", columns="testwindow", values="n", aggfunc="sum"
    ).fillna(0)
    eligible_groups = []
    for g in _pivot.index.tolist():
        if any(float(_pivot.loc[g].get(w, 0) or 0) > 11 for w in win_order):
            eligible_groups.append(g)

    # Always keep All Students if it exists and has sufficient n in either window
    if "All Students" in _pivot.index:
        if any(float(_pivot.loc["All Students"].get(w, 0) or 0) > 11 for w in win_order):
            if "All Students" not in eligible_groups:
                eligible_groups = ["All Students"] + eligible_groups

    d2 = d2[d2["student_group"].isin(eligible_groups)].copy()
    if d2.empty:
        logger.info(
            f"[CHART] Section 8: no student groups met n>11 threshold after filtering (windows={win_order})"
        )
        return pd.DataFrame(), target_year, []

    quint_counts = (
        d2.groupby(["student_group", "testwindow", "achievementquintile"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_counts = (
        d2.groupby(["student_group", "testwindow"])
        .size()
        .rename("N_total")
        .reset_index()
    )

    pct_df = quint_counts.merge(
        total_counts, on=["student_group", "testwindow"], how="left"
    )
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    pct_df["testwindow"] = pct_df["testwindow"].astype(str).str.upper()

    group_names = pct_df["student_group"].dropna().unique().tolist()

    def _gkey(g: str):
        if g == "All Students":
            return (-1, g)
        return (int(group_order_map.get(g, 99)), g)

    group_order = [g for g in sorted(group_names, key=_gkey)]

    all_idx = pd.MultiIndex.from_product(
        [group_order, win_order, hf.NWEA_ORDER],
        names=["student_group", "testwindow", "achievementquintile"],
    )

    pct_df = (
        pct_df.set_index(["student_group", "testwindow", "achievementquintile"])
        .reindex(all_idx)
        .reset_index()
    )
    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df.groupby(["student_group", "testwindow"])[
        "N_total"
    ].transform(lambda s: s.ffill().bfill())

    return pct_df, target_year, group_order


def _plot_section8_window_compare_by_student_group(
    pct_df, subject_title, target_year, group_order, preview=False
):
    if pct_df.empty or not group_order:
        print(f"[Section 8] Skipped {subject_title} (no data)")
        return

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    x = np.arange(len(group_order))
    window_order = _get_eoy_compare_windows()
    n_w = len(window_order)
    if n_w == 2:
        w = 0.36
        offsets = {window_order[0]: -w / 2, window_order[1]: w / 2}
    elif n_w == 3:
        w = 0.26
        offsets = {window_order[0]: -w, window_order[1]: 0.0, window_order[2]: w}
    else:
        w = max(0.18, 0.80 / max(1, n_w))
        spread = np.linspace(-0.30, 0.30, max(1, n_w))
        offsets = {window_order[i]: float(spread[i]) for i in range(n_w)}

    pivot = (
        pct_df.pivot_table(
            index=["student_group", "testwindow"],
            columns="achievementquintile",
            values="pct",
            aggfunc="sum",
        )
        .reindex(columns=hf.NWEA_ORDER)
        .fillna(0)
    )

    for window in window_order:
        cumulative = np.zeros(len(group_order))
        for cat in hf.NWEA_ORDER:
            vals = []
            for g in group_order:
                try:
                    vals.append(float(pivot.loc[(g, window), cat]))
                except Exception:
                    vals.append(0.0)
            vals = np.array(vals, dtype=float)

            bars = ax.bar(
                x + offsets[window],
                vals,
                bottom=cumulative,
                width=w,
                color=hf.NWEA_COLORS[cat],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.6 if window != window_order[-1] else 1.0,
            )

            for i_bar, rect in enumerate(bars):
                h = float(vals[i_bar])
                if h >= LABEL_MIN_PCT:
                    bottom_before = float(cumulative[i_bar])
                    if cat in ("High", "HiAvg"):
                        label_color = "white"
                    elif cat in ("Avg", "LoAvg"):
                        label_color = "#434343"
                    elif cat == "Low":
                        label_color = "white"
                    else:
                        label_color = "#434343"
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        bottom_before + h / 2,
                        f"{h:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color=label_color,
                    )

            cumulative += vals

    ax.set_ylim(0, 100)
    ax.set_ylabel("% of Students")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Student group labels at group centers (with n-counts per window)
    n_map = (
        pct_df.groupby(["student_group", "testwindow"])["N_total"]
        .max()
        .dropna()
        .astype(int)
        .to_dict()
    )

    x_labels = []
    for g in group_order:
        short = {"FALL": "F", "WINTER": "W", "SPRING": "S"}
        parts = []
        for w_key in window_order:
            nval = int(n_map.get((g, w_key), 0) or 0)
            parts.append(f"{short.get(w_key, w_key[:1])} n={nval}")
        x_labels.append(f"{g}\n(" + " | ".join(parts) + ")")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for xi in x:
        for w_key in window_order:
            ax.text(
                xi + offsets[w_key],
                1.02,
                w_key.title(),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#434343",
                transform=trans,
            )

    legend_handles = [
        Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q)
        for q in hf.NWEA_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        labels=hf.NWEA_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(hf.NWEA_ORDER),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    district_name_cfg = district_label
    ax.set_title(
        f"{district_name_cfg} • {window_order[0].title()} vs {window_order[-1].title()} by Student Group \n {subject_title} {window_order[-1].title()} {target_year}",
        fontsize=18,
        fontweight="bold",
        pad=22,
        y=1.05,
    )

    # Save to district folder (always use CHARTS_DIR so the runner can collect outputs)
    charts_dir = CHARTS_DIR
    out_dir = charts_dir / "_district"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_subj = subject_title.replace(" ", "_").replace("/", "_")
    out_path = (
        out_dir
        / f"{district_name_cfg.replace(' ', '_')}_section8_window_compare_by_group_{safe_subj}_eoy.png"
    )

    hf._save_and_render(fig, out_path)
    print(f"Saved Section 8 ({subject_title}): {out_path}")

    if preview:
        plt.show()
    plt.close(fig)


# ---- RUN SECTION 8 (district only) ----
try:
    pct_ela_grp, year_ela_grp, groups_ela = _prep_section8_window_compare_by_student_group(
        nwea_base.copy(),
        subject_str="Reading",
    )
    _plot_section8_window_compare_by_student_group(
        pct_ela_grp,
        subject_title="Reading",
        target_year=year_ela_grp,
        group_order=groups_ela,
        preview=False,
    )

    pct_math_grp, year_math_grp, groups_math = (
        _prep_section8_window_compare_by_student_group(
            nwea_base.copy(),
            subject_str="Math",
        )
    )
    _plot_section8_window_compare_by_student_group(
        pct_math_grp,
        subject_title="Math",
        target_year=year_math_grp,
        group_order=groups_math,
        preview=False,
    )
except Exception as e:
    print(f"[Section 8] ERROR: {e}")


# %%
# SECTION 9–11 — District Growth (NWEA)
#
# Mirrors the i-Ready Sections 9–11 structure, aligned to NWEA growth.
# Produces 4 charts per scope:
#   - Reading: Median CGP (percentile), Mean CGI (index)
#   - Math:    Median CGP (percentile), Mean CGI (index)
#
# Growth windows for EOY (Spring-ended):
#   - Winter→Spring (within-year)
#   - Spring→Spring (year-over-year, Spring-ended)
#
#########################################################################

# ----------------------------
# Growth windows (Sections 9–11 always output BOTH)
# ----------------------------
GROWTH_WINDOWS = [
    # EOY windows (Spring-ended)
    {
        "key": "fall_to_spring",
        "label": "Fall→Spring",
        "end_window": "SPRING",
        "cgp_col": "falltospringconditionalgrowthpercentile",
        "cgi_col": "falltospringconditionalgrowthindex",
    },
    {
        "key": "spring_to_spring",
        "label": "Spring→Spring",
        "end_window": "SPRING",
        "cgp_col": "springtospringconditionalgrowthpercentile",
        "cgi_col": "springtospringconditionalgrowthindex",
    },
]


def _latest_year_num(df_in: pd.DataFrame) -> int | None:
    if "year" not in df_in.columns:
        return None
    y = pd.to_numeric(df_in["year"].astype(str).str[:4], errors="coerce")
    if y.notna().any():
        return int(y.max())
    return None


def _filter_window_latest_year(
    df_in: pd.DataFrame, end_window: str
) -> tuple[pd.DataFrame, int | None]:
    d = df_in.copy()

    # End-window only (growth metrics are anchored on the ending test window)
    if "testwindow" in d.columns:
        d["testwindow"] = d["testwindow"].astype(str)
        d = d[d["testwindow"].str.upper() == str(end_window).upper()].copy()

    # Latest year only
    yr = _latest_year_num(d)
    if yr is not None:
        d["year_num"] = pd.to_numeric(d["year"].astype(str).str[:4], errors="coerce")
        d = d[d["year_num"] == yr].copy()

    return d, yr


def _filter_subject_nwea(df_in: pd.DataFrame, subject_title: str) -> pd.DataFrame:
    d = df_in.copy()
    subj_norm = subject_title.strip().casefold()

    if "course" not in d.columns:
        return d.iloc[0:0].copy()

    if "math" in subj_norm:
        d = d[d["course"].astype(str).str.contains("math", case=False, na=False)].copy()
    else:
        d = d[
            d["course"].astype(str).str.contains("reading", case=False, na=False)
        ].copy()

    return d


def _dedupe_latest_attempt(df_in: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    d = df_in.copy()

    id_col = (
        "uniqueidentifier"
        if "uniqueidentifier" in d.columns
        else ("studentid" if "studentid" in d.columns else None)
    )
    if id_col is None:
        id_col = "__row_id__"
        d[id_col] = np.arange(len(d))

    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    else:
        d["teststartdate"] = pd.NaT

    sort_cols = by_cols + [id_col, "teststartdate"]
    sort_cols = [c for c in sort_cols if c in d.columns]

    d = d.sort_values(sort_cols)

    group_cols = [c for c in (by_cols + [id_col]) if c in d.columns]
    if not group_cols:
        return d

    # Keep latest attempt per student within each scope
    d = d.groupby(group_cols, as_index=False).tail(1)
    return d


def _agg_growth_by_scope(
    df_in: pd.DataFrame,
    scope_col: str,
    cgp_col: str,
    cgi_col: str,
    end_window: str,
) -> pd.DataFrame:
    d, _ = _filter_window_latest_year(df_in, end_window=end_window)

    # Resolve CamelCase vs lowercase column names
    col_map: dict[str, str] = {}
    for c in d.columns:
        key = str(c).strip().lower()
        col_map.setdefault(key, c)

    scope_actual = col_map.get(str(scope_col).strip().lower(), scope_col)
    cgp_actual = col_map.get(str(cgp_col).strip().lower())
    cgi_actual = col_map.get(str(cgi_col).strip().lower())

    # Require metrics
    if scope_actual not in d.columns or not cgp_actual or cgp_actual not in d.columns:
        return pd.DataFrame(columns=[scope_col, "median_cgp", "mean_cgi", "n"])
    if not cgi_actual or cgi_actual not in d.columns:
        # CGI missing: still allow CGP-only output (mean_cgi becomes NaN)
        cgi_actual = None

    # Dedupe within scope
    d = _dedupe_latest_attempt(d, by_cols=[scope_actual])

    # Require non-null values for each metric
    d_cgp = d.dropna(subset=[cgp_actual]).copy()
    d_cgi = d.dropna(subset=[cgi_actual]).copy() if cgi_actual else d.iloc[0:0].copy()

    # n is count of unique students with metric present (per metric)
    id_col = (
        "uniqueidentifier"
        if "uniqueidentifier" in d.columns
        else ("studentid" if "studentid" in d.columns else None)
    )

    def _nunique(x):
        if id_col is None:
            return int(x.shape[0])
        return int(x[id_col].nunique())

    cgp_out = (
        d_cgp.groupby(scope_actual, dropna=False)
        .apply(
            lambda x: pd.Series(
                {
                    "median_cgp": float(np.nanmedian(x[cgp_actual])),
                    "n_cgp": _nunique(x),
                }
            )
        )
        .reset_index()
    )
    if cgi_actual:
        cgi_out = (
            d_cgi.groupby(scope_actual, dropna=False)
            .apply(
                lambda x: pd.Series(
                    {
                        "mean_cgi": float(np.nanmean(x[cgi_actual])),
                        "n_cgi": _nunique(x),
                    }
                )
            )
            .reset_index()
        )
    else:
        cgi_out = pd.DataFrame(columns=[scope_actual, "mean_cgi", "n_cgi"])

    out = cgp_out.merge(cgi_out, on=scope_actual, how="outer")
    out["n"] = out[["n_cgp", "n_cgi"]].max(axis=1)
    out.drop(columns=[c for c in ["n_cgp", "n_cgi"] if c in out.columns], inplace=True)

    # Normalize back to requested scope_col name for downstream plotting
    if scope_actual != scope_col and scope_actual in out.columns:
        out = out.rename(columns={scope_actual: scope_col})

    return out


def _save_growth_chart(fig, section_num: int, suffix: str):
    # Save to district folder (always use CHARTS_DIR so the runner can collect outputs)
    charts_dir = CHARTS_DIR
    out_dir = charts_dir / "_district"
    out_dir.mkdir(parents=True, exist_ok=True)

    district_name_cfg = district_label
    safe_d = district_name_cfg.replace(" ", "_")
    out_path = out_dir / f"{safe_d}_section{section_num}_{suffix}.png"
    hf._save_and_render(fig, out_path)
    print(f"Saved: {out_path}")


def _plot_scope_growth_bar(
    df_scope: pd.DataFrame,
    scope_col: str,
    scope_order: list,
    subject_title: str,
    metric: str,
    growth_label: str,
    section_num: int,
    suffix: str,
    y_lim: tuple[float, float] | None = None,
):
    if df_scope.empty:
        print(
            f"[Section {section_num}] Skipped {subject_title} {metric} ({scope_col}) — no data"
        )
        return

    # order
    if scope_order:
        df_scope = df_scope[df_scope[scope_col].isin(scope_order)].copy()
        df_scope[scope_col] = pd.Categorical(
            df_scope[scope_col], categories=scope_order, ordered=True
        )
        df_scope = df_scope.sort_values(scope_col)
    else:
        df_scope = df_scope.sort_values(scope_col)

    x = np.arange(len(df_scope))

    if metric == "median_cgp":
        y = df_scope["median_cgp"].to_numpy(dtype=float)
        ylab = f"Median {growth_label} CGP"
        title_metric = "Median CGP"
        if y_lim is None:
            y_lim = (0, 100)
    else:
        y = df_scope["mean_cgi"].to_numpy(dtype=float)
        ylab = f"Mean {growth_label} CGI"
        title_metric = "Mean CGI"
        if y_lim is None:
            y_lim = (-2.5, 2.5)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    # CGP: reference band 42–58 (with guide lines)
    if metric == "median_cgp":
        ax.axhspan(42, 58, facecolor="#78daf4", alpha=0.28, zorder=0)
        for yref in [42, 50, 58]:
            ax.axhline(yref, linestyle="--", color="#6B7280", linewidth=1.2, zorder=0)

    # CGI: add the standard yellow band around 0
    if metric == "mean_cgi":
        ax.axhspan(-0.2, 0.2, facecolor="#facc15", alpha=0.25, zorder=0)
        for yref in [-0.2, 0.0, 0.2]:
            ax.axhline(yref, linestyle="--", color="#eab308", linewidth=1.1, zorder=0)

    bars = ax.bar(x, y, edgecolor="white", linewidth=1.2)

    # Value labels
    for rect, v in zip(bars, y):
        if pd.isna(v):
            continue
        if metric == "median_cgp":
            label = f"{v:.1f}"
            y_text = (
                rect.get_height() / 2
                if rect.get_height() >= 0
                else rect.get_height() / 2
            )
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y_text,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if abs(v) >= 35 else "#434343",
            )
        else:
            label = f"{v:.2f}"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                label,
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
                fontweight="bold",
                color="#434343",
            )

    # X labels with n (best-effort)
    labels = df_scope[scope_col].astype(str).tolist()
    if "n" in df_scope.columns:
        labels = [
            f"{lbl}\n(n = {int(nv)})"
            for lbl, nv in zip(labels, df_scope["n"].fillna(0))
        ]

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=35 if scope_col != "grade_num" else 0,
        ha="right" if scope_col != "grade_num" else "center",
    )

    ax.set_ylabel(ylab)
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    district_name_cfg = district_label
    yr = _latest_year_num(nwea_base)
    year_txt = f"Spring {yr}" if yr is not None else "Spring"

    ax.set_title(
        f"{district_name_cfg} • {subject_title} • {title_metric}\n{year_txt} ({growth_label})",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )

    _save_growth_chart(fig, section_num=section_num, suffix=suffix)
    plt.close(fig)


# ---------------------------------------------------------------------
# SECTION 9 — District Growth by School (Reading + Math)
# ---------------------------------------------------------------------


def _run_section9_growth_by_school():
    # Debug: Check available columns for growth metrics
    growth_like_cols = sorted([c for c in nwea_base.columns if "conditional" in str(c).lower()])
    logger.info(f"[CHART] Section 9: Found {len(growth_like_cols)} conditional growth columns in nwea_base: {growth_like_cols[:12]}")
    
    for subject_title in ["Reading", "Math"]:
        safe_subj = "reading" if subject_title.lower().startswith("read") else "math"

        for w in GROWTH_WINDOWS:
            logger.info(f"[CHART] Section 9: Attempting {subject_title} / {w['label']} (end_window={w['end_window']})")
            logger.info(f"[CHART] Section 9: Looking for columns: {w['cgp_col']}, {w['cgi_col']}")
            
            d0, _ = _filter_window_latest_year(nwea_base.copy(), end_window=w["end_window"])
            if d0.empty:
                logger.warning(f"[CHART] Section 9: no {w['end_window']} data for {subject_title} ({w['label']}) — d0 empty after window filter")
                continue
            logger.info(f"[CHART] Section 9: After window filter: {len(d0):,} rows, {d0['teststartdate'].min()} to {d0['teststartdate'].max()}")
            
            if "schoolname" not in d0.columns:
                logger.warning(f"[CHART] Section 9: missing schoolname column for {subject_title} ({w['label']})")
                continue

            d0["school_display"] = d0["schoolname"].apply(
                lambda s: hf._safe_normalize_school_name(s, cfg)
            )
            school_order = sorted(d0["school_display"].dropna().unique().tolist())

            ds = _filter_subject_nwea(d0, subject_title)
            if ds.empty:
                continue

            agg = _agg_growth_by_scope(
                ds,
                scope_col="school_display",
                cgp_col=w["cgp_col"],
                cgi_col=w["cgi_col"],
                end_window=w["end_window"],
            )
            if agg.empty:
                continue

            # Median CGP
            _plot_scope_growth_bar(
                agg,
                scope_col="school_display",
                scope_order=school_order,
                subject_title=subject_title,
                metric="median_cgp",
                growth_label=w["label"],
                section_num=9,
                suffix=f"growth_by_school_{safe_subj}_{w['key']}_median_cgp",
                y_lim=(0, 100),
            )

            # Mean CGI
            _plot_scope_growth_bar(
                agg,
                scope_col="school_display",
                scope_order=school_order,
                subject_title=subject_title,
                metric="mean_cgi",
                growth_label=w["label"],
                section_num=9,
                suffix=f"growth_by_school_{safe_subj}_{w['key']}_mean_cgi",
                y_lim=(-2.5, 2.5),
            )


# ---------------------------------------------------------------------
# SECTION 10 — District Growth by Grade (Reading + Math)
# ---------------------------------------------------------------------


def _run_section10_growth_by_grade():
    def _glabel(g):
        return "K" if int(g) == 0 else str(int(g))

    for subject_title in ["Reading", "Math"]:
        for w in GROWTH_WINDOWS:
            d0, _ = _filter_window_latest_year(nwea_base.copy(), end_window=w["end_window"])
            if d0.empty:
                logger.info(f"[CHART] Section 10: no {w['end_window']} data for {subject_title} ({w['label']})")
                continue

            if "grade" not in d0.columns and "student_grade" not in d0.columns:
                logger.info("[CHART] Section 10: missing grade column")
                continue

            grade_col = "grade" if "grade" in d0.columns else "student_grade"
            d0["grade_num"] = pd.to_numeric(d0[grade_col], errors="coerce")
            d0 = d0[d0["grade_num"].notna()].copy()
            if d0.empty:
                continue
            d0["grade_num"] = d0["grade_num"].astype(int)
            grade_order = sorted(d0["grade_num"].dropna().unique().tolist())

            ds = _filter_subject_nwea(d0, subject_title)
            if ds.empty:
                continue

            agg = _agg_growth_by_scope(
                ds,
                scope_col="grade_num",
                cgp_col=w["cgp_col"],
                cgi_col=w["cgi_col"],
                end_window=w["end_window"],
            )
            if agg.empty:
                continue

            # Replace numeric grade with label for plotting
            agg = agg.copy()
            agg["grade_label"] = agg["grade_num"].apply(_glabel)
            label_order = [_glabel(g) for g in grade_order]
            agg = agg.rename(columns={"grade_label": "grade_display"})

            safe_subj = (
                "reading" if subject_title.lower().startswith("read") else "math"
            )

            # Median CGP
            _plot_scope_growth_bar(
                agg,
                scope_col="grade_display",
                scope_order=label_order,
                subject_title=subject_title,
                metric="median_cgp",
                growth_label=w["label"],
                section_num=10,
                suffix=f"growth_by_grade_{safe_subj}_{w['key']}_median_cgp",
                y_lim=(0, 100),
            )

            # Mean CGI
            _plot_scope_growth_bar(
                agg,
                scope_col="grade_display",
                scope_order=label_order,
                subject_title=subject_title,
                metric="mean_cgi",
                growth_label=w["label"],
                section_num=10,
                suffix=f"growth_by_grade_{safe_subj}_{w['key']}_mean_cgi",
                y_lim=(-2.5, 2.5),
            )


# ---------------------------------------------------------------------
# SECTION 11 — District Growth by Student Group (Reading + Math)
# ---------------------------------------------------------------------


def _run_section11_growth_by_student_group():
    DEFAULT_GROUPS = [
        "All Students",
        "Students with Disabilities",
        "Socioeconomically Disadvantaged",
        "English Learners",
        "Hispanic",
        "White",
    ]

    # Build a working set of rows for the enabled groups
    student_groups_cfg = cfg.get("student_groups", {})
    group_order_map = cfg.get("student_group_order", {})

    def _gkey(g: str):
        if g == "All Students":
            return (-1, g)
        return (int(group_order_map.get(g, 99)), g)

    for subject_title in ["Reading", "Math"]:
        safe_subj = "reading" if subject_title.lower().startswith("read") else "math"

        for w in GROWTH_WINDOWS:
            d0, _ = _filter_window_latest_year(nwea_base.copy(), end_window=w["end_window"])
            if d0.empty:
                logger.info(f"[CHART] Section 11: no {w['end_window']} data for {subject_title} ({w['label']})")
                continue

            enabled_set = set(DEFAULT_GROUPS)
            try:
                if _selected_groups:
                    enabled_set = set(["All Students"] + [str(g) for g in _selected_groups])
            except Exception:
                pass

            frames = []

            # All Students
            if "All Students" in enabled_set:
                all_mask = _apply_student_group_mask(d0, "All Students", {"type": "all"})
                if all_mask is not None:
                    d_all = d0[all_mask].copy()
                    if not d_all.empty:
                        frames.append(d_all.assign(student_group="All Students"))

            # Configured groups
            for group_name, group_def in student_groups_cfg.items():
                if group_def.get("type") == "all":
                    continue
                if group_name not in enabled_set:
                    continue
                try:
                    mask = _apply_student_group_mask(d0, group_name, group_def)
                except Exception:
                    continue
                dg = d0[mask].copy()
                if dg.empty:
                    continue
                frames.append(dg.assign(student_group=group_name))

            if not frames:
                continue

            d2 = pd.concat(frames, ignore_index=True)
            group_names = [
                g for g in list(enabled_set) if g in set(d2["student_group"].unique())
            ]
            if not group_names:
                continue
            group_order = sorted(group_names, key=_gkey)

            ds = _filter_subject_nwea(d2, subject_title)
            if ds.empty:
                continue

            agg = _agg_growth_by_scope(
                ds,
                scope_col="student_group",
                cgp_col=w["cgp_col"],
                cgi_col=w["cgi_col"],
                end_window=w["end_window"],
            )
            if agg.empty:
                continue

            # Median CGP
            _plot_scope_growth_bar(
                agg,
                scope_col="student_group",
                scope_order=group_order,
                subject_title=subject_title,
                metric="median_cgp",
                growth_label=w["label"],
                section_num=11,
                suffix=f"growth_by_group_{safe_subj}_{w['key']}_median_cgp",
                y_lim=(0, 100),
            )

            # Mean CGI
            _plot_scope_growth_bar(
                agg,
                scope_col="student_group",
                scope_order=group_order,
                subject_title=subject_title,
                metric="mean_cgi",
                growth_label=w["label"],
                section_num=11,
                suffix=f"growth_by_group_{safe_subj}_{w['key']}_mean_cgi",
                y_lim=(-2.5, 2.5),
            )


# ---------------------------------------------------------------------
# RUN Sections 9–11 (district only)
# ---------------------------------------------------------------------
try:
    _run_section9_growth_by_school()
    _run_section10_growth_by_grade()
    _run_section11_growth_by_student_group()
except Exception as e:
    print(f"[Sections 9–11] ERROR: {e}")
#########################################################################
#
