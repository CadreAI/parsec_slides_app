import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def generate_nwea_winter_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = "./data",
    nwea_data: Optional[List[Dict[str, Any]]] = None,
) -> list:
    """
    Runs legacy script-style `nwea_moy.py` as a subprocess, using temp files for config + data.

    Why: `nwea_moy.py` executes code at import-time and expects on-disk `settings.yaml`,
    `config_files/{partner}.yaml`, and `../data/nwea_data.csv`.
    """
    if not nwea_data:
        return []

    run_root = Path(tempfile.mkdtemp(prefix="parsec_nwea_moy_"))
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

    # Write settings.yaml (avoid importing PyYAML at module import time)
    with open(settings_path, "w") as f:
        f.write(f"partner_name: {partner_name}\n")

    # Write partner config YAML (prefer PyYAML if installed; otherwise write JSON which YAML can parse)
    try:
        import yaml  # type: ignore

        with open(config_path, "w") as f:
            yaml.safe_dump(config or {}, f, sort_keys=False)
    except Exception:
        import json

        with open(config_path, "w") as f:
            json.dump(config or {}, f, indent=2, default=str)

    # Write NWEA CSV
    pd.DataFrame(nwea_data).to_csv(csv_path, index=False)

    script_path = Path(__file__).resolve().parent / "nwea_moy.py"
    run_cwd = str(Path(__file__).resolve().parent)

    env = os.environ.copy()
    env.update(
        {
            "NWEA_MOY_SETTINGS_PATH": str(settings_path),
            "NWEA_MOY_CONFIG_PATH": str(config_path),
            "NWEA_MOY_DATA_DIR": str(run_data_dir),
            "NWEA_MOY_CHARTS_DIR": str(run_charts_dir),
            "NWEA_MOY_LOG_DIR": str(run_logs_dir),
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
            env["NWEA_MOY_SCOPE_MODE"] = "district_only"
    except Exception:
        pass

    # Scope control (district_only vs district + schools vs selected schools).
    # Prefer config.assessment_scopes['nwea'] if present.
    try:
        scopes = (config or {}).get("assessment_scopes") or {}
        nwea_scope = scopes.get("nwea") or {}
        include_districtwide = nwea_scope.get("includeDistrictwide")
        include_schools = nwea_scope.get("includeSchools")
        include_districtwide = True if include_districtwide is None else bool(include_districtwide)
        include_schools = True if include_schools is None else bool(include_schools)
        schools = nwea_scope.get("schools") if include_schools else None
        if isinstance(schools, list) and schools:
            env["NWEA_MOY_SCHOOLS"] = ",".join(str(s) for s in schools if str(s).strip())
            if env.get("NWEA_MOY_SCOPE_MODE") != "district_only":
                env["NWEA_MOY_SCOPE_MODE"] = "selected_schools"
        if include_districtwide and not include_schools:
            env["NWEA_MOY_SCOPE_MODE"] = "district_only"
    except Exception:
        pass

    # Pass selected student groups from frontend into the legacy script
    # Frontend uses chart_filters["student_groups"] (snake_case)
    try:
        groups = (chart_filters or {}).get("student_groups") or []
        if isinstance(groups, list) and groups:
            env["NWEA_MOY_STUDENT_GROUPS"] = ",".join(str(g) for g in groups)
    except Exception:
        pass

    # Pass selected race/ethnicity from frontend into the legacy script
    # Frontend uses chart_filters["race"] as an array of strings.
    try:
        races = (chart_filters or {}).get("race") or []
        if isinstance(races, list) and races:
            env["NWEA_MOY_RACE"] = ",".join(str(r) for r in races if str(r).strip())
    except Exception:
        pass

    # Pass selected grades from frontend into the legacy script (used for grade-level batches)
    try:
        grades = (chart_filters or {}).get("grades") or []
        if isinstance(grades, list) and grades:
            env["NWEA_MOY_GRADES"] = ",".join(str(g) for g in grades)
    except Exception:
        pass

    # Pass selected subjects from frontend into the legacy script.
    # Frontend uses chart_filters["subjects"] as an array of strings.
    try:
        subjects = (chart_filters or {}).get("subjects") or []
        if isinstance(subjects, list) and subjects:
            env["NWEA_MOY_SUBJECTS"] = ",".join(str(s) for s in subjects if str(s).strip())
    except Exception:
        pass

    # NOTE: `chart_filters` is otherwise unused by `nwea_moy.py` (legacy script).
    _ = data_dir

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
        (run_logs_dir / "nwea_moy_stdout.txt").write_text(
            proc.stdout or "", encoding="utf-8"
        )
        (run_logs_dir / "nwea_moy_stderr.txt").write_text(
            proc.stderr or "", encoding="utf-8"
        )
    except Exception:
        pass

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-4000:]
        stdout_tail = (proc.stdout or "")[-4000:]
        raise RuntimeError(
            "nwea_moy.py failed.\n"
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
        logs_out_dir = out_dir / "_logs" / "nwea_moy"
        logs_out_dir.mkdir(parents=True, exist_ok=True)
        for log_src in sorted(run_logs_dir.glob("*.txt")):
            try:
                shutil.copy2(log_src, logs_out_dir / log_src.name)
            except Exception:
                pass
        # Also write a small tail summary for quick inspection.
        try:
            tail = (proc.stdout or "")[-8000:] + "\n\n" + (proc.stderr or "")[-8000:]
            (logs_out_dir / "nwea_moy_subprocess_tail.txt").write_text(tail, encoding="utf-8")
        except Exception:
            pass
    except Exception:
        pass

    # Cleanup temp workspace (charts have been copied out)
    shutil.rmtree(run_root, ignore_errors=True)

    return copied
