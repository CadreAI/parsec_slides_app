import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def generate_iready_winter_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = "./data",
    iready_data: Optional[List[Dict[str, Any]]] = None,
) -> list:
    """
    Runs legacy script-style `iready_moy.py` as a subprocess, using temp files for config + data.

    Why: `iready_moy.py` executes code at import-time and expects on-disk `settings.yaml`,
    `config_files/{partner}.yaml`, and `../data/iready_data.csv`.
    """
    if not iready_data:
        return []

    run_root = Path(tempfile.mkdtemp(prefix="parsec_iready_moy_"))
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

    script_path = Path(__file__).resolve().parent / "iready_moy.py"
    run_cwd = str(Path(__file__).resolve().parent)

    env = os.environ.copy()
    env.update(
        {
            "IREADY_MOY_SETTINGS_PATH": str(settings_path),
            "IREADY_MOY_CONFIG_PATH": str(config_path),
            "IREADY_MOY_DATA_DIR": str(run_data_dir),
            "IREADY_MOY_CHARTS_DIR": str(run_charts_dir),
            "IREADY_MOY_LOG_DIR": str(run_logs_dir),
            "MPLCONFIGDIR": str(run_mpl_dir),
            "PREVIEW": "false",
        }
    )
    # Pass selected grades from frontend into the legacy script (used for grade-level batches)
    try:
        grades = (chart_filters or {}).get("grades") or []
        if isinstance(grades, list) and grades:
            env["IREADY_MOY_GRADES"] = ",".join(str(g) for g in grades)
    except Exception:
        pass

    # Pass selected student groups + race/ethnicity from frontend into the legacy script.
    # iReady treats race/ethnicity as additional "student group" keys in cfg['student_groups'].
    # Frontend uses:
    # - chart_filters["student_groups"]
    # - chart_filters["race"]
    try:
        cf = chart_filters or {}
        groups = cf.get("student_groups")
        races = cf.get("race")
        
        # Handle student groups
        if groups is not None:
            if isinstance(groups, list):
                if len(groups) == 0:
                    env["IREADY_MOY_STUDENT_GROUPS"] = "NONE"
                else:
                    env["IREADY_MOY_STUDENT_GROUPS"] = ",".join(str(g).strip() for g in groups if str(g).strip())
            elif isinstance(groups, str) and groups.strip():
                env["IREADY_MOY_STUDENT_GROUPS"] = groups.strip()
        
        # Handle races separately
        if races is not None:
            if isinstance(races, list):
                if len(races) == 0:
                    env["IREADY_MOY_RACE"] = "NONE"
                else:
                    env["IREADY_MOY_RACE"] = ",".join(str(r).strip() for r in races if str(r).strip())
            elif isinstance(races, str) and races.strip():
                env["IREADY_MOY_RACE"] = races.strip()
    except Exception:
        pass

    # Scope selection (district vs schools)
    # Supported chart_filters options:
    # - chart_filters["district_only"] = True → district charts only (including sections 6-11)
    # - chart_filters["schools_only"] = True → school charts only (skip district sections 6-11)
    # - chart_filters["schools"] = ["School A", "School B"] → only these schools (behavior depends on schools_only flag)
    try:
        cf = chart_filters or {}
        
        # Determine scope mode based on frontend flags
        if bool(cf.get("district_only")) is True:
            env["IREADY_MOY_SCOPE_MODE"] = "district_only"
        elif bool(cf.get("schools_only")) is True:
            # schools_only = True means skip district-level charts (including sections 6-11)
            env["IREADY_MOY_SCOPE_MODE"] = "schools_only"
        
        # Set school list if provided
        schools = cf.get("schools") or []
        if isinstance(schools, list) and len(schools) > 0:
            env["IREADY_MOY_SCHOOLS"] = ",".join(str(s) for s in schools if str(s).strip())
            # Only set selected_schools mode if no explicit mode was set
            # (selected_schools includes both district and school charts)
            if "IREADY_MOY_SCOPE_MODE" not in env:
                env["IREADY_MOY_SCOPE_MODE"] = "selected_schools"
            # Note: If schools_only was already set, we keep that mode (skip district)
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
        (run_logs_dir / "iready_moy_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (run_logs_dir / "iready_moy_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    except Exception:
        pass

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-4000:]
        stdout_tail = (proc.stdout or "")[-4000:]
        raise RuntimeError(
            "iready_moy.py failed.\n"
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


