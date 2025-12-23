import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def generate_nwea_fall_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = "./data",
    nwea_data: Optional[List[Dict[str, Any]]] = None,
) -> list:
    """
    Runs legacy script-style `nwea_boy.py` as a subprocess, using temp files for config + data.

    Why: `nwea_boy.py` executes code at import-time and expects on-disk `settings.yaml`,
    `config_files/{partner}.yaml`, and `../data/nwea_data.csv`.
    """
    if not nwea_data:
        return []

    run_root = Path(tempfile.mkdtemp(prefix="parsec_nwea_boy_"))
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
    csv_path = run_data_dir / "nwea_data.csv"

    with open(settings_path, "w", encoding="utf-8") as f:
        f.write(f"partner_name: {partner_name}\n")

    # Write partner config YAML (prefer PyYAML if installed; otherwise JSON which YAML can parse)
    try:
        import yaml  # type: ignore

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config or {}, f, sort_keys=False)
    except Exception:
        import json

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config or {}, f, indent=2, default=str)

    pd.DataFrame(nwea_data).to_csv(csv_path, index=False)

    script_path = Path(__file__).resolve().parent / "nwea_boy.py"
    run_cwd = str(Path(__file__).resolve().parent)

    env = os.environ.copy()
    env.update(
        {
            "NWEA_BOY_SETTINGS_PATH": str(settings_path),
            "NWEA_BOY_CONFIG_PATH": str(config_path),
            "NWEA_BOY_DATA_DIR": str(run_data_dir),
            "NWEA_BOY_CHARTS_DIR": str(run_charts_dir),
            "NWEA_BOY_LOG_DIR": str(run_logs_dir),
            "MPLCONFIGDIR": str(run_mpl_dir),
            "PREVIEW": "false",
        }
    )

    # Frontend override: district-only charts
    # Mirrors iready_*_runner behavior. Use a NWEA-specific key to avoid collisions when
    # multiple assessments are selected (e.g., iReady school charts + NWEA district-only).
    try:
        cf = chart_filters or {}
        if bool(cf.get("nwea_district_only")) is True:
            env["NWEA_BOY_SCOPE_MODE"] = "district_only"
    except Exception:
        pass

    # Scope control (district_only vs district + schools vs selected schools vs schools_only).
    # Prefer config.assessment_scopes['nwea'] if present.
    try:
        scopes = (config or {}).get("assessment_scopes") or {}
        nwea_scope = scopes.get("nwea") or {}
        include_districtwide = nwea_scope.get("includeDistrictwide")
        include_schools = nwea_scope.get("includeSchools")
        include_districtwide = True if include_districtwide is None else bool(include_districtwide)
        include_schools = True if include_schools is None else bool(include_schools)
        
        # Priority order: schools_only > district_only > selected_schools > default (both)
        if not include_districtwide and include_schools:
            # Schools only - skip district charts entirely
            env["NWEA_BOY_SCOPE_MODE"] = "schools_only"
        elif include_districtwide and not include_schools:
            # District only - skip school charts entirely
            env["NWEA_BOY_SCOPE_MODE"] = "district_only"
        
        schools = nwea_scope.get("schools") if include_schools else None
        if isinstance(schools, list) and schools:
            env["NWEA_BOY_SCHOOLS"] = ",".join(str(s) for s in schools if str(s).strip())
            if env.get("NWEA_BOY_SCOPE_MODE") not in ("district_only", "schools_only"):
                env["NWEA_BOY_SCOPE_MODE"] = "selected_schools"
    except Exception:
        pass

    # Pass selected grades from frontend into the legacy script
    try:
        grades = (chart_filters or {}).get("grades") or []
        if isinstance(grades, list) and grades:
            env["NWEA_BOY_GRADES"] = ",".join(str(g) for g in grades)
    except Exception:
        pass

    # Pass selected student groups from frontend into the legacy script
    try:
        groups = (chart_filters or {}).get("student_groups") or []
        if isinstance(groups, list) and groups:
            env["NWEA_BOY_STUDENT_GROUPS"] = ",".join(str(g) for g in groups)
    except Exception:
        pass

    # Pass selected race/ethnicity from frontend into the legacy script
    try:
        races = (chart_filters or {}).get("race") or []
        if isinstance(races, list) and races:
            env["NWEA_BOY_RACE"] = ",".join(str(r) for r in races if str(r).strip())
    except Exception:
        pass

    # Pass selected subjects from frontend into the legacy script.
    # Frontend uses chart_filters["subjects"] as an array of strings.
    try:
        subjects = (chart_filters or {}).get("subjects") or []
        if isinstance(subjects, list) and subjects:
            env["NWEA_BOY_SUBJECTS"] = ",".join(str(s) for s in subjects if str(s).strip())
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
        (run_logs_dir / "nwea_boy_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (run_logs_dir / "nwea_boy_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    except Exception:
        pass

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-4000:]
        stdout_tail = (proc.stdout or "")[-4000:]
        raise RuntimeError(
            "nwea_boy.py failed.\n"
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

    # Persist subprocess logs into output_dir so Celery logs aren't the only place to debug.
    try:
        logs_out_dir = out_dir / "_logs" / "nwea_boy"
        logs_out_dir.mkdir(parents=True, exist_ok=True)
        for log_src in sorted(run_logs_dir.glob("*.txt")):
            try:
                shutil.copy2(log_src, logs_out_dir / log_src.name)
            except Exception:
                pass
        # Also write a small tail summary for quick inspection.
        try:
            tail = (proc.stdout or "")[-8000:] + "\n\n" + (proc.stderr or "")[-8000:]
            (logs_out_dir / "nwea_boy_subprocess_tail.txt").write_text(tail, encoding="utf-8")
        except Exception:
            pass
    except Exception:
        pass

    shutil.rmtree(run_root, ignore_errors=True)
    return copied


