"""
Data ingestion module for Python backend
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import concurrent.futures

from bigquery_client import get_bigquery_client, run_query
from nwea.sql_builders import sql_nwea


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names: lowercase, replace spaces with underscores
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    return df


def ingest_nwea(
    partner_name: str,
    config: Dict[str, Any],
    chart_filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Ingest NWEA data from BigQuery
    
    Args:
        partner_name: Partner name
        config: Partner configuration dict
        chart_filters: Optional filters for chart generation
    
    Returns:
        List of dictionaries representing NWEA data rows
    """
    chart_filters = chart_filters or {}
    
    # Map config-level filters to chart_filters if not already set
    # Frontend may send district_name and selected_schools at config level
    if not chart_filters.get('districts') and config.get('district_name'):
        districts = config.get('district_name')
        if isinstance(districts, list) and len(districts) > 0:
            chart_filters['districts'] = districts
            print(f"[Data Ingestion] Mapped district_name to chart_filters.districts: {districts}")
    
    if not chart_filters.get('schools') and config.get('selected_schools'):
        schools = config.get('selected_schools')
        if isinstance(schools, list) and len(schools) > 0:
            chart_filters['schools'] = schools
            print(f"[Data Ingestion] Mapped selected_schools to chart_filters.schools: {schools}")
    
    # Get BigQuery configuration
    gcp_config = config.get('gcp', {})
    project_id = gcp_config.get('project_id')
    location = gcp_config.get('location', 'US')
    
    if not project_id:
        raise ValueError("GCP project_id is required in config")
    
    # Get table ID
    # config.sources.nwea can be either:
    # 1. A string (the table_id directly)
    # 2. A dict with 'table_id' key
    sources = config.get('sources', {})
    nwea_source = sources.get('nwea')
    
    if isinstance(nwea_source, str):
        # Direct table_id string
        table_id = nwea_source
    elif isinstance(nwea_source, dict):
        # Dict with table_id key
        table_id = nwea_source.get('table_id')
    else:
        table_id = None
    
    if not table_id:
        raise ValueError("NWEA table_id is required in config.sources.nwea")
    
    # Build base filters (without years - we'll query each year separately)
    # Note: DistrictName column doesn't exist in NWEA table, so district filtering
    # must be done in Python after data retrieval
    base_filters = {}
    
    # School filter (column name is "School" not "SchoolName")
    if chart_filters.get('schools'):
        base_filters['schools'] = chart_filters['schools']
    
    # Determine if we should apply grade filter in SQL
    # Always apply grade filter in SQL when grades are specified (more efficient)
    # This reduces data fetched significantly
    apply_grade_filter_sql = bool(chart_filters.get('grades') and len(chart_filters.get('grades', [])) > 0)
    
    if apply_grade_filter_sql:
        base_filters['grades'] = chart_filters['grades']
        print(f"[Data Ingestion] Will apply grade filter in SQL: {chart_filters['grades']}")
    else:
        print(f"[Data Ingestion] No grade filter specified - will fetch all grades")
    
    # Note: districts filter is applied in Python after query (no DistrictName column in NWEA)
    
    # Warn if no filters are applied (will fetch all data)
    if not base_filters.get('schools'):
        print(f"[Data Ingestion] ⚠️  WARNING: No school filter applied!")
        print(f"[Data Ingestion] ⚠️  This will fetch data for ALL schools (may be very large)")
    
    # Determine years to query
    years = chart_filters.get('years')
    if not years or len(years) == 0:
        # Default: last 3 years
        current_date = datetime.now()
        current_year = current_date.year
        if current_date.month >= 7:
            current_year += 1
        years = [current_year - 2, current_year - 1, current_year]
        print(f"[Data Ingestion] No years specified, using default: {years}")
    else:
        print(f"[Data Ingestion] Querying years: {years}")
    
    # Function to query a single year
    def query_year(year: int) -> List[Dict[str, Any]]:
        """Query a single year with its own BigQuery client (thread-safe)"""
        import traceback
        try:
            print(f"[Data Ingestion] [Year {year}] Starting query setup...")
            # Create a new client for this thread (thread-safe)
            credentials_path = config.get('gcp', {}).get('credentials_path')
            client = get_bigquery_client(project_id, credentials_path)
            
            # Build filters for this year
            year_filters = {**base_filters, 'years': [year]}
            # Remove 'grades' from year_filters if we're applying it in SQL (already in base_filters)
            # The apply_grade_filter flag controls whether SQL builder uses it
            sql = sql_nwea(table_id, year_filters, apply_grade_filter=apply_grade_filter_sql)
            
            print(f"[Data Ingestion] [Year {year}] Executing query...")
            rows = run_query(sql, client, None)
            print(f"[Data Ingestion] [Year {year}] Retrieved {len(rows):,} rows")
            return rows
        except Exception as e:
            error_msg = str(e)
            print(f"[Data Ingestion] [Year {year}] Error: {error_msg}")
            print(f"[Data Ingestion] [Year {year}] Traceback:")
            traceback.print_exc()
            return []
    
    # Execute queries in parallel
    print(f"[Data Ingestion] Executing {len(years)} year queries in parallel...")
    print(f"[Data Ingestion] Table: {table_id}")
    print(f"[Data Ingestion] Base filters: {base_filters}")
    
    all_rows = []
    
    # Use ThreadPoolExecutor to run queries in parallel
    # Limit to max 5 concurrent queries to avoid overwhelming BigQuery
    max_workers = min(len(years), 5)
    
    print(f"[Data Ingestion] Using ThreadPoolExecutor with {max_workers} workers for {len(years)} years")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all year queries
        future_to_year = {executor.submit(query_year, year): year for year in years}
        print(f"[Data Ingestion] Submitted {len(future_to_year)} queries")
        
        # Collect results as they complete with timeout
        import time
        start_time = time.time()
        query_timeout = 300  # 5 minute timeout per query
        
        completed_years = set()
        
        try:
            # Wait for all futures with a reasonable timeout
            total_timeout = query_timeout * len(years) + 60  # Extra buffer
            for future in concurrent.futures.as_completed(future_to_year, timeout=total_timeout):
                year = future_to_year[future]
                elapsed = time.time() - start_time
                try:
                    rows = future.result(timeout=30)  # 30 second timeout for result retrieval
                    all_rows.extend(rows)
                    completed_years.add(year)
                    print(f"[Data Ingestion] [Year {year}] Completed: {len(rows):,} rows (Total: {len(all_rows):,}) in {elapsed:.1f}s")
                except concurrent.futures.TimeoutError:
                    print(f"[Data Ingestion] [Year {year}] TIMEOUT: Result retrieval took longer than 30s")
                except Exception as e:
                    print(f"[Data Ingestion] [Year {year}] Failed: {e}")
                    import traceback
                    traceback.print_exc()
        except concurrent.futures.TimeoutError:
            print(f"[Data Ingestion] TIMEOUT: Some queries exceeded {total_timeout}s total timeout")
            # Check which queries are still pending
            for future, year in future_to_year.items():
                if year not in completed_years:
                    if future.running():
                        print(f"[Data Ingestion] [Year {year}] Still running...")
                    elif future.done():
                        try:
                            rows = future.result(timeout=5)
                            all_rows.extend(rows)
                            print(f"[Data Ingestion] [Year {year}] Late completion: {len(rows):,} rows")
                        except:
                            print(f"[Data Ingestion] [Year {year}] Failed to retrieve result")
                    else:
                        print(f"[Data Ingestion] [Year {year}] Not started")
    
    print(f"[Data Ingestion] All queries completed. Total rows: {len(all_rows):,}")
    
    rows = all_rows
    
    if not rows:
        print("[Data Ingestion] Warning: No rows returned from query")
        return []
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    print(f"[Data Ingestion] Raw data: {len(df)} rows, {len(df.columns)} columns")
    
    # Clean column names
    df = clean_column_names(df)
    
    # Deduplication logic
    print("[Data Ingestion] Applying deduplication...")
    initial_count = len(df)
    
    # First, use more targeted deduplication based on key identifiers
    # This is more appropriate than checking ALL columns for exact matches
    if 'studentid' in df.columns and 'teststartdate' in df.columns:
        # Identify key columns for grouping (these define a unique test record)
        group_cols = ['studentid']
        if 'subject' in df.columns:
            group_cols.append('subject')
        if 'termname' in df.columns:
            group_cols.append('termname')
        if 'year' in df.columns:
            group_cols.append('year')
        if 'teststartdate' in df.columns:
            group_cols.append('teststartdate')
        
        # Check for duplicates based on these key columns
        before_key_dedup = len(df)
        df = df.sort_values('teststartdate', ascending=False).drop_duplicates(
            subset=group_cols,
            keep='first'
        )
        key_dupes_removed = before_key_dedup - len(df)
        if key_dupes_removed > 0:
            print(f"[Data Ingestion] Removed {key_dupes_removed} duplicates based on key columns: {group_cols}")
        
        # If there's a uniqueidentifier column, use that for final deduplication
        if 'uniqueidentifier' in df.columns:
            before_unique_dedup = len(df)
            df = df.drop_duplicates(subset=['uniqueidentifier'], keep='first')
            unique_dupes_removed = before_unique_dedup - len(df)
            if unique_dupes_removed > 0:
                print(f"[Data Ingestion] Removed {unique_dupes_removed} duplicates based on uniqueidentifier")
    else:
        # Fallback: remove exact duplicates only if we don't have the key columns
        print("[Data Ingestion] Warning: Missing key columns for deduplication, using exact match")
        df = df.drop_duplicates(keep='first')
    
    total_removed = initial_count - len(df)
    print(f"[Data Ingestion] Final data: {len(df)} rows after deduplication (removed {total_removed:,} duplicates, {total_removed/initial_count*100:.1f}%)")
    
    # Apply grade filter if specified (after deduplication)
    # Note: If grade filter was applied in SQL, this is a safety check/re-filter
    # If grade filter was NOT applied in SQL, this is the primary filter
    if chart_filters.get('grades') and 'grade' in df.columns:
        grades = chart_filters['grades']
        before_filter = len(df)
        df = df[df['grade'].isin(grades)]
        print(f"[Data Ingestion] Applied grade filter in Python: {before_filter} -> {len(df)} rows")
    
    # Convert back to list of dicts
    result = df.to_dict('records')
    
    return result

