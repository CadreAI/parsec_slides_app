"""
Data ingestion module for Python backend
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import concurrent.futures

from python.bigquery_client import get_bigquery_client, run_query
from python.nwea.sql_builders import sql_nwea
from python.iready.sql_builders import sql_iready
from python.star.sql_builders import sql_star
from python import helper_functions as hf


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

    # NWEA: Do NOT filter by grades or quarters during data collection
    # Only filter by years. Grade and quarter filtering happens during chart generation.
    apply_grade_filter_sql = False
    print(f"[Data Ingestion] NWEA: Fetching all grades (grade filtering happens during chart generation)")

    # Note: districts filter is applied in Python after query (no DistrictName column in NWEA)

    # Warn if no filters are applied (will fetch all data)
    if not base_filters.get('schools'):
        print(f"[Data Ingestion] ⚠️  WARNING: No school filter applied!")
        print(f"[Data Ingestion] ⚠️  This will fetch data for ALL schools (may be very large)")

    # NWEA: Use simplified query that fetches last 3 years in a single query
    print(f"[Data Ingestion] NWEA: Executing single query for last 3 years...")
    print(f"[Data Ingestion] Table: {table_id}")

    # Get exclude columns from config if specified
    exclude_cols = None
    if isinstance(nwea_source, dict) and 'exclude_cols' in nwea_source:
        exclude_cols = nwea_source.get('exclude_cols')
    # Also check config-level exclude_cols
    if not exclude_cols and isinstance(config.get('exclude_cols'), dict):
        exclude_cols = config.get('exclude_cols', {}).get('nwea', [])

    # Build and execute single SQL query
    try:
        credentials_path = config.get('gcp', {}).get('credentials_path')
        client = get_bigquery_client(project_id, credentials_path)

        # Ensure table_id is a string
        if not isinstance(table_id, str):
            raise ValueError(f"table_id must be a string, got {type(table_id)}: {table_id}")
        
        sql = sql_nwea(table_id=table_id, exclude_cols=exclude_cols)

        print(f"[Data Ingestion] SQL Query:")
        print(f"[Data Ingestion] {sql}")
        print(f"[Data Ingestion] Executing query...")
        rows = run_query(sql, client, None)
        print(f"[Data Ingestion] Retrieved {len(rows):,} total rows from query")
    except Exception as e:
        error_msg = str(e)
        print(f"[Data Ingestion] Error: {error_msg}")
        import traceback
        traceback.print_exc()
        rows = []

    if not rows:
        print("[Data Ingestion] Warning: No rows returned from queries")
        return []

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    print(f"[Data Ingestion] Raw data after combining all years: {len(df)} rows, {len(df.columns)} columns")
    
    # Check for year distribution before column name cleaning
    if 'Year' in df.columns:
        year_counts = df['Year'].value_counts().sort_index()
        print(f"[Data Ingestion] Year distribution in raw data:")
        for year, count in year_counts.items():
            print(f"[Data Ingestion]   Year {year}: {count:,} rows")

    # Clean column names
    df = clean_column_names(df)
    
    # Debug: Print available columns for deduplication
    print(f"[Data Ingestion] Available columns after cleaning: {sorted(df.columns.tolist())}")
    
    # Check for year distribution after column name cleaning
    if 'year' in df.columns:
        year_counts = df['year'].value_counts().sort_index()
        print(f"[Data Ingestion] Year distribution after column cleaning:")
        for year, count in year_counts.items():
            print(f"[Data Ingestion]   Year {year}: {count:,} rows")

    # Apply filters in Python (not in SQL)
    print("[Data Ingestion] Applying filters in Python...")

    # School filter
    # For charter schools, school name may be in learning_center column instead of school column
    # Use fuzzy matching similar to get_scopes() to handle name variations
    if chart_filters.get('schools'):
        schools = chart_filters['schools']
        school_col = None
        
        # Check learning_center first (for charter schools), then school (for regular schools)
        if 'learning_center' in df.columns:
            school_col = 'learning_center'
        elif 'school' in df.columns:
            school_col = 'school'
        
        if school_col:
            before_filter = len(df)
            # Use fuzzy matching instead of exact matching
            # Normalize selected schools
            selected_schools_normalized = [hf._safe_normalize_school_name(s, config) for s in schools]
            
            # Find matching schools using fuzzy logic
            available_schools = df[school_col].dropna().unique()
            matching_schools = []
            for school in available_schools:
                normalized_school = hf._safe_normalize_school_name(school, config)
                # Check normalized match
                if normalized_school in selected_schools_normalized:
                    matching_schools.append(school)
                # Also check direct case-insensitive/partial match
                elif any(school.lower() == selected.lower() or 
                        selected.lower() in school.lower() or 
                        school.lower() in selected.lower() 
                        for selected in schools):
                    matching_schools.append(school)
            
            if matching_schools:
                df = df[df[school_col].isin(matching_schools)]
                print(f"[Data Ingestion] Applied school filter using column '{school_col}': {before_filter} -> {len(df)} rows")
                print(f"[Data Ingestion] Matched schools: {matching_schools}")
            else:
                print(f"[Data Ingestion] ⚠️  No matching schools found for: {schools}")
                print(f"[Data Ingestion] Available schools in '{school_col}': {sorted(available_schools)}")
                print(f"[Data Ingestion] Keeping all data - will let get_scopes() handle filtering")
        else:
            print(f"[Data Ingestion] Warning: School filter specified but no school column found (checked: learning_center, school)")

    # District filter (applied in Python since DistrictName column doesn't exist in NWEA)
    # Handle both column name variations: DistrictName -> districtname, or district_name -> district_name
    if chart_filters.get('districts'):
        districts = chart_filters['districts']
        district_col = None
        
        # Check for both possible column names (after clean_column_names normalization)
        if 'districtname' in df.columns:
            district_col = 'districtname'
        elif 'district_name' in df.columns:
            district_col = 'district_name'
        
        if district_col:
            before_filter = len(df)
            df = df[df[district_col].isin(districts)]
            print(f"[Data Ingestion] Applied district filter using column '{district_col}': {before_filter} -> {len(df)} rows")
        else:
            print(f"[Data Ingestion] Warning: District filter specified but no district column found (checked: districtname, district_name)")

    # NWEA: No deduplication needed - SQL query uses SELECT DISTINCT
    # The database handles deduplication, so we keep all rows as-is
    initial_count = len(df)
    print(f"[Data Ingestion] NWEA data: {initial_count:,} rows (deduplication handled by SQL DISTINCT)")
    
    # Summary of filtering impact
    raw_row_count = len(rows)  # Store original count before filtering
    print(f"[Data Ingestion] Data reduction summary:")
    print(f"[Data Ingestion]   - Raw data from query (DISTINCT): {raw_row_count:,} rows")
    print(f"[Data Ingestion]   - After school/district filters: {initial_count:,} rows")
    print(f"[Data Ingestion]   - Final data: {len(df):,} rows")
    if raw_row_count != len(df):
        print(f"[Data Ingestion]   - Total reduction: {raw_row_count - len(df):,} rows ({100 * (raw_row_count - len(df)) / raw_row_count:.1f}%)")

    # NWEA: Do NOT apply grade filter during data collection
    # Grade filtering happens during chart generation when needed
    # This allows us to collect all grade data and filter later as needed
    # (Grade filtering removed per user request - only filter by years)

    # Convert back to list of dicts
    result = df.to_dict('records')

    return result


def ingest_iready(
    partner_name: str,
    config: Dict[str, Any],
    chart_filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Ingest iReady data from BigQuery

    Args:
        partner_name: Partner name
        config: Partner configuration dict
        chart_filters: Optional filters for chart generation (NOT applied in SQL - all filtering done in Python)

    Returns:
        List of dictionaries representing iReady data rows
    """
    chart_filters = chart_filters or {}

    # Map config-level filters to chart_filters if not already set
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
    sources = config.get('sources', {})
    iready_source = sources.get('iready')

    if isinstance(iready_source, str):
        table_id = iready_source
    elif isinstance(iready_source, dict):
        table_id = iready_source.get('table_id')
    else:
        table_id = None

    if not table_id:
        raise ValueError("iReady table_id is required in config.sources.iready")

    # Note: iReady SQL builder includes year filtering in SQL, but other filters done in Python
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

    # Get exclude columns from config if specified
    exclude_cols = None
    if isinstance(iready_source, dict) and 'exclude_cols' in iready_source:
        exclude_cols = iready_source.get('exclude_cols')

    # Function to query a single year
    def query_year(year: int) -> List[Dict[str, Any]]:
        """Query a single year with its own BigQuery client (thread-safe)"""
        import traceback
        try:
            print(f"[Data Ingestion] [Year {year}] Starting query setup...")
            # Create a new client for this thread (thread-safe)
            credentials_path = config.get('gcp', {}).get('credentials_path')
            client = get_bigquery_client(project_id, credentials_path)

            # Build SQL query for this specific year
            sql_filters = {'years': [year]}
            sql = sql_iready(table_id, exclude_cols, filters=sql_filters)

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
        print("[Data Ingestion] Warning: No rows returned from queries")
        return []

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    print(f"[Data Ingestion] Raw data: {len(df)} rows, {len(df.columns)} columns")

    # Clean column names
    df = clean_column_names(df)

    # Apply filters in Python (not in SQL)
    print("[Data Ingestion] Applying filters in Python...")

    # School filter
    # For charter schools, school name may be in learning_center column instead of school column
    if chart_filters.get('schools'):
        schools = chart_filters['schools']
        school_col = None
        
        # Check learning_center first (for charter schools), then school (for regular schools)
        if 'learning_center' in df.columns:
            school_col = 'learning_center'
        elif 'school' in df.columns:
            school_col = 'school'
        
        if school_col:
            before_filter = len(df)
            df = df[df[school_col].isin(schools)]
            print(f"[Data Ingestion] Applied school filter using column '{school_col}': {before_filter} -> {len(df)} rows")
        else:
            print(f"[Data Ingestion] Warning: School filter specified but no school column found (checked: learning_center, school)")

    # Note: Year filtering is done in SQL, not in Python
    # This ensures we only fetch the years we need from BigQuery
    # Note: Grade filtering and deduplication are NOT applied for iReady
    # All data is returned as-is from BigQuery (after year and school filtering)

    # Convert back to list of dicts
    result = df.to_dict('records')

    return result


def ingest_star(
    partner_name: str,
    config: Dict[str, Any],
    chart_filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Ingest STAR data from BigQuery

    Args:
        partner_name: Partner name
        config: Partner configuration dict
        chart_filters: Optional filters for chart generation (year filtering in SQL, others in Python)

    Returns:
        List of dictionaries representing STAR data rows
    """
    chart_filters = chart_filters or {}

    # Map config-level filters to chart_filters if not already set
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
    sources = config.get('sources', {})
    star_source = sources.get('star')

    if isinstance(star_source, str):
        table_id = star_source
    elif isinstance(star_source, dict):
        table_id = star_source.get('table_id')
    else:
        table_id = None

    if not table_id:
        raise ValueError("STAR table_id is required in config.sources.star")

    # STAR SQL builder includes year filtering in SQL
    # Determine years to query
    years = chart_filters.get('years')
    if not years or len(years) == 0:
        # Default: last 3 years (matching sql_star default)
        current_date = datetime.now()
        current_year = current_date.year
        if current_date.month >= 7:
            current_year += 1
        years = [current_year - 2, current_year - 1, current_year]
        print(f"[Data Ingestion] No years specified, using default: {years}")
    else:
        print(f"[Data Ingestion] Querying years: {years}")

    # Get exclude columns from config if specified
    exclude_cols = None
    if isinstance(star_source, dict) and 'exclude_cols' in star_source:
        exclude_cols = star_source.get('exclude_cols')

    # Build SQL query with year filters
    sql_filters = {'years': years}
    sql = sql_star(table_id, exclude_cols, filters=sql_filters)

    print(f"[Data Ingestion] Executing STAR query...")
    print(f"[Data Ingestion] Table: {table_id}")

    # Get BigQuery client
    credentials_path = config.get('gcp', {}).get('credentials_path')
    client = get_bigquery_client(project_id, credentials_path)

    # Execute query
    rows = run_query(sql, client, None)
    print(f"[Data Ingestion] Retrieved {len(rows):,} rows")

    if not rows:
        print("[Data Ingestion] Warning: No rows returned from STAR query")
        return []

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    print(f"[Data Ingestion] Raw data: {len(df)} rows, {len(df.columns)} columns")

    # Clean column names
    df = clean_column_names(df)

    # Apply filters in Python (school filtering, etc.)
    print("[Data Ingestion] Applying filters in Python...")

    # School filter (STAR uses School_Name column, normalized to school_name after clean_column_names)
    # For charter schools, school name may be in learning_center column instead of school column
    if chart_filters.get('schools'):
        schools = chart_filters['schools']
        # Check learning_center first (for charter schools), then other variations
        school_col = None
        for col_name in ['learning_center', 'school_name', 'schoolname', 'school']:
            if col_name in df.columns:
                school_col = col_name
                break

        if school_col:
            before_filter = len(df)
            df = df[df[school_col].isin(schools)]
            print(f"[Data Ingestion] Applied school filter using column '{school_col}': {before_filter} -> {len(df)} rows")
        else:
            print("[Data Ingestion] Warning: Could not find school column for filtering (checked: learning_center, school_name, schoolname, school)")

    # District filter (STAR uses District_Name column, normalized to district_name after clean_column_names)
    if chart_filters.get('districts'):
        districts = chart_filters['districts']
        # Try multiple column name variations
        district_col = None
        for col_name in ['district_name', 'districtname', 'district']:
            if col_name in df.columns:
                district_col = col_name
                break

        if district_col:
            before_filter = len(df)
            df = df[df[district_col].isin(districts)]
            print(f"[Data Ingestion] Applied district filter: {before_filter} -> {len(df)} rows")
        else:
            print("[Data Ingestion] Warning: Could not find district column for filtering")

    # Convert back to list of dicts
    result = df.to_dict('records')

    return result
