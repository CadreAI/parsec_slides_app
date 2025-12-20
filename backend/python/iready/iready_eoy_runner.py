import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def generate_iready_eoy_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = "./data",
    iready_data: Optional[List[Dict[str, Any]]] = None,
) -> list:
    """
    Runs legacy script-style `iready_eoy.py` as a subprocess, using temp files for config + data.

    Why: `iready_eoy.py` executes code at import-time and expects on-disk `settings.yaml`,
    `config_files/{partner}.yaml`, and `../data/iready_data.csv`.
    """
    if not iready_data:
        return []

    run_root = Path(tempfile.mkdtemp(prefix="parsec_iready_eoy_"))
    run_data_dir = run_root / "data"
    run_charts_dir = run_root / "charts"
    run_logs_dir = run_root / "logs"
    run_mpl_dir = run_root / "mplconfig"
    run_cfg_dir = run_root / "config_files"

    run_data_dir.mkdir(parents=True, exist_ok=True)
    run_charts_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    run_mpl_dir.mkdir(parents=True, exist_ok=True)
    run_cfg_dir.mkdir(parents=True, exist_ok=True)

    settings_path = run_root / "settings.yaml"
    config_path = run_cfg_dir / f"{partner_name}.yaml"
    csv_path = run_data_dir / "iready_data.csv"

    # Write settings.yaml
    # Write settings.yaml (avoid importing PyYAML at module import time)
    with open(settings_path, "w") as f:
        f.write(f"partner_name: {partner_name}\n")

    # Write partner config YAML (iReady script expects YAML)
    # Write partner config YAML (prefer PyYAML if installed; otherwise write JSON which YAML can parse)
    try:
        import yaml  # type: ignore

        with open(config_path, "w") as f:
            yaml.safe_dump(config or {}, f, sort_keys=False)
    except Exception:
        import json

        with open(config_path, "w") as f:
            json.dump(config or {}, f, indent=2, default=str)

    # Write iReady CSV
    pd.DataFrame(iready_data).to_csv(csv_path, index=False)

    script_path = Path(__file__).resolve().parent / "iready_eoy.py"
    run_cwd = str(Path(__file__).resolve().parent)

    env = os.environ.copy()
    env.update(
        {
            "IREADY_EOY_SETTINGS_PATH": str(settings_path),
            "IREADY_EOY_CONFIG_PATH": str(config_path),
            "IREADY_EOY_DATA_DIR": str(run_data_dir),
            "IREADY_EOY_CHARTS_DIR": str(run_charts_dir),
            "IREADY_EOY_LOG_DIR": str(run_logs_dir),
            "MPLCONFIGDIR": str(run_mpl_dir),
            "PREVIEW": "false",
        }
    )
    # Pass selected grades from frontend into the legacy script (used for grade-level batches)
    try:
        grades = (chart_filters or {}).get("grades") or []
        if isinstance(grades, list) and grades:
            env["IREADY_EOY_GRADES"] = ",".join(str(g) for g in grades)
    except Exception:
        pass

    # Pass selected student groups + race/ethnicity from frontend into the legacy script.
    # iReady treats race/ethnicity as additional "student group" keys in cfg['student_groups'].
    # Frontend uses:
    # - chart_filters["student_groups"]
    # - chart_filters["race"]
    try:
        cf = chart_filters or {}
        groups = cf.get("student_groups") or []
        races = cf.get("race") or []
        selected: list[str] = []
        for src in [groups, races]:
            if isinstance(src, str):
                src = [src]
            if not isinstance(src, list):
                continue
            for v in src:
                s = str(v).strip()
                if not s:
                    continue
                if s not in selected:
                    selected.append(s)

        if selected:
            env["IREADY_EOY_STUDENT_GROUPS"] = ",".join(selected)
    except Exception:
        pass

    # Within-year compare windows for EOY charts:
    # - Default: Winter + Spring
    # - If Fall is also selected: Fall + Winter + Spring
    try:
        quarters = (chart_filters or {}).get("quarters") or []
        if isinstance(quarters, str):
            quarters = [quarters]
        norm = [str(q).strip().lower() for q in (quarters if isinstance(quarters, list) else [])]
        include_fall = "fall" in norm
        # EOY comparisons always include Spring; include Winter if present, otherwise still default to Winter+Spring.
        env["IREADY_EOY_COMPARE_WINDOWS"] = "Fall,Winter,Spring" if include_fall else "Winter,Spring"
    except Exception:
        env["IREADY_EOY_COMPARE_WINDOWS"] = "Winter,Spring"

    # Scope selection (district vs schools)
    # Supported chart_filters options:
    # - chart_filters["district_only"] = True → district charts only
    # - chart_filters["schools"] = ["School A", "School B"] → only these schools (plus district)
    try:
        cf = chart_filters or {}
        if bool(cf.get("district_only")) is True:
            env["IREADY_EOY_SCOPE_MODE"] = "district_only"
        schools = cf.get("schools") or []
        if isinstance(schools, list) and len(schools) > 0:
            env["IREADY_EOY_SCHOOLS"] = ",".join(str(s) for s in schools if str(s).strip())
            if env.get("IREADY_EOY_SCOPE_MODE") != "district_only":
                env["IREADY_EOY_SCOPE_MODE"] = "selected_schools"
    except Exception:
        pass

    proc = subprocess.run(
        [sys.executable, str(script_path), "--full"],
        cwd=run_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    # Persist stdout/stderr for debugging
    try:
        (run_logs_dir / "iready_eoy_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (run_logs_dir / "iready_eoy_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    except Exception:
        pass

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-4000:]
        stdout_tail = (proc.stdout or "")[-4000:]
        raise RuntimeError(
            "iready_eoy.py failed.\n"
            f"returncode={proc.returncode}\n"
            f"stdout_tail:\n{stdout_tail}\n"
            f"stderr_tail:\n{stderr_tail}\n"
            f"run_root={run_root}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for src in sorted(run_charts_dir.rglob("*.png")):
        rel = src.relative_to(run_charts_dir)
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(str(dest))

        # Copy optional chart data sidecar if present (used by chart_analyzer.py)
        data_src = src.parent / f"{src.stem}_data.json"
        if data_src.exists():
            data_dest = dest.parent / f"{dest.stem}_data.json"
            try:
                shutil.copy2(data_src, data_dest)
            except Exception:
                pass

    # Cleanup temp workspace (charts have been copied out)
    shutil.rmtree(run_root, ignore_errors=True)

    return copied


# Backward-compatible alias (older imports may still use this name)
def generate_iready_winter_charts(*args, **kwargs) -> list:
    return generate_iready_eoy_charts(*args, **kwargs)


