"""
iready_sql.py
Generates SQL for i-Ready ingestion with optional dynamic column excludes.
DEPRECATED: Use sql_builders.sql_iready instead
"""

from __future__ import annotations
from typing import List
from .sql_builders import sql_iready


def get_iready_sql(table_id: str, exclude_cols: List[str] | None = None) -> str:
    """
    Build SQL for i-Ready pulls (wrapper for sql_builders.sql_iready).

    Args:
        table_id (str): Fully qualified BigQuery table name (project.dataset.table)
        exclude_cols (List[str] | None): Extra columns to exclude (from YAML)

    Returns:
        str: Final SQL string
    """
    # Delegate to sql_builders module
    return sql_iready(table_id, exclude_cols)
