"""
SQL query builders for BigQuery - NWEA

Policy:
- Always filter by years in SQL.
- If schools are selected, also filter by schools in SQL to reduce result size.
"""
from typing import List, Optional, Dict, Any

# EXCLUDE_COLS can be imported from config if needed
EXCLUDE_COLS = {}


def sql_nwea(
    table_id: str,
    exclude_cols: Optional[List[str]] = None,
    year_column: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build SQL query for NWEA data
    
    Args:
        table_id: BigQuery table ID (project.dataset.table)
        exclude_cols: Optional list of additional columns to exclude from config
        year_column: Optional year column name ('Year' or 'AcademicYear'). If None, will use OR condition for both.
        filters: Optional filters dict with years and quarters (other filters applied in Python)
    
    Returns:
        SQL query string
    """
    filters = filters or {}
    # If Districtwide charts are requested, we must NOT filter by school name in SQL.
    # School-level slicing happens later during chart generation.
    include_districtwide = bool(filters.get("include_districtwide"))

    def _sql_escape(s: str) -> str:
        return str(s).replace("'", "\\'")

    def _sql_like_any(lower_expr: str, needles: list[str]) -> Optional[str]:
        pats = [n.strip().lower() for n in needles if n is not None and str(n).strip()]
        if not pats:
            return None
        ors = [f"{lower_expr} LIKE '%{_sql_escape(p)}%'" for p in pats]
        return "(" + " OR ".join(ors) + ")"

    available_cols = [str(c).lower() for c in (filters.get("available_columns") or [])]

    def _pick_col(candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c.lower() in available_cols:
                return c
        return None
    
    extra_excludes = EXCLUDE_COLS.get("nwea", [])
    if exclude_cols:
        extra_excludes = extra_excludes + exclude_cols

    # Some partners use non-standard production tables that include a large `email` column.
    # Exclude it ONLY when it exists, otherwise BigQuery will error on SELECT * EXCEPT(email).
    if "email" in available_cols and "email" not in [str(c).lower() for c in extra_excludes]:
        extra_excludes = extra_excludes + ["email"]
    
    # Determine which Growth columns to include based on selected quarters
    quarters = filters.get('quarters', [])
    if not isinstance(quarters, list):
        quarters = []
    
    # Map quarters to their Growth column prefixes (columns to INCLUDE, not exclude)
    # Note: Winter needs FallToWinter for Section 5 (Fall→Winter growth charts)
    growth_column_prefixes = {
        'Fall': ['FallToFall', 'FallToWinter', 'FallToSpring'],
        'Winter': ['FallToWinter', 'WinterToWinter', 'WinterToSpring'],
        'Spring': ['SpringToSpring', 'FallToSpring', 'WinterToSpring']
    }
    
    # Growth column suffixes for each type
    growth_suffixes = [
        'ProjectedGrowth',
        'ObservedGrowth',
        'ObservedGrowthSE',
        'MetProjectedGrowth',
        'ConditionalGrowthIndex',
        'ConditionalGrowthPercentile',
        'GrowthQuintile'
    ]
    
    # Build all possible growth columns for selected quarters
    columns_to_include = []
    for quarter in quarters:
        quarter_normalized = str(quarter).strip().capitalize()
        if quarter_normalized in growth_column_prefixes:
            prefixes = growth_column_prefixes[quarter_normalized]
            for prefix in prefixes:
                for suffix in growth_suffixes:
                    columns_to_include.append(f"{prefix}{suffix}")
    
    # Map quarters to their Typical Growth columns (columns to INCLUDE, not exclude)
    # Note: Winter needs TypicalFallToWinterGrowth for Section 5 (Fall→Winter growth charts)
    typical_growth_columns = {
        'Fall': ['TypicalFallToFallGrowth', 'TypicalFallToWinterGrowth', 'TypicalFallToSpringGrowth'],
        'Winter': ['TypicalFallToWinterGrowth', 'TypicalWinterToWinterGrowth', 'TypicalWinterToSpringGrowth'],
        'Spring': ['TypicalSpringToSpringGrowth', 'TypicalWinterToSpringGrowth', 'SpringToWTypicalFallToSpringGrowthinterGrowth']
    }
    
    # Collect all Typical Growth columns that should be INCLUDED (not excluded)
    typical_columns_to_include = []
    for quarter in quarters:
        quarter_normalized = str(quarter).strip().capitalize()
        if quarter_normalized in typical_growth_columns:
            typical_columns_to_include.extend(typical_growth_columns[quarter_normalized])
    
    # All possible Typical Growth columns
    all_typical_growth = [
        'TypicalFallToFallGrowth',
        'TypicalFallToWinterGrowth',
        'TypicalFallToSpringGrowth',
        'TypicalWinterToWinterGrowth',
        'TypicalWinterToSpringGrowth',
        'TypicalSpringToSpringGrowth'
    ]
    
    # All possible Growth columns (ProjectedGrowth, ObservedGrowth, etc.)
    all_growth_columns = []
    all_growth_prefixes = ['FallToFall', 'FallToWinter', 'FallToSpring', 'WinterToWinter', 'WinterToSpring', 'SpringToSpring']
    for prefix in all_growth_prefixes:
        for suffix in growth_suffixes:
            all_growth_columns.append(f"{prefix}{suffix}")
    
    # Build the Growth exclusion list (exclude all except the ones we want to include)
    growth_excludes = []
    if columns_to_include:
        # Exclude all growth columns EXCEPT the ones we want
        growth_excludes = [col for col in all_growth_columns if col not in columns_to_include]
    else:
        # If no quarters selected, exclude all growth columns
        growth_excludes = all_growth_columns
    
    # Build the Typical Growth exclusion list (exclude all except the ones we want to include)
    typical_growth_excludes = []
    if typical_columns_to_include:
        # Exclude all typical growth columns EXCEPT the ones we want
        typical_growth_excludes = [col for col in all_typical_growth if col not in typical_columns_to_include]
    else:
        # If no quarters selected, exclude all typical growth columns
        typical_growth_excludes = all_typical_growth
    
    # Format growth excludes for SQL
    growth_excludes_sql = ""
    if growth_excludes:
        growth_excludes_sql = ",\n        ".join(growth_excludes)
    
    # Format typical growth excludes for SQL
    typical_growth_excludes_sql = ""
    if typical_growth_excludes:
        typical_growth_excludes_sql = ",\n        ".join(typical_growth_excludes)
    
    dynamic_excludes = ",\n        ".join(extra_excludes) if extra_excludes else ""
    
    # Build year filter based on provided years or default to last 3 years
    if filters.get('years') and len(filters['years']) > 0:
        # Use selected years
        year_list = ', '.join(map(str, filters['years']))
        if year_column:
            # Use specified column
            year_filter = f"{year_column} IN ({year_list})"
        else:
            # Use OR condition to handle both Year and AcademicYear
            year_filter = f"""(
        -- Try Year column first (NWEA typically uses Year)
        (Year IN ({year_list}))
        -- Fallback to AcademicYear if Year doesn't exist (some tables use AcademicYear)
        OR (AcademicYear IN ({year_list}))
    )"""
    else:
        # Default: last 3 years (matching the provided SQL pattern)
        if year_column:
            # Use specified column
            year_filter = f"{year_column} >= (\n        CASE\n            WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7\n                THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1\n            ELSE EXTRACT(YEAR FROM CURRENT_DATE())\n        END\n    ) - 3"
        else:
            # Use OR condition to handle both Year and AcademicYear
            # This will work if at least one column exists
            year_filter = """(
        -- Try Year column first (NWEA typically uses Year)
        (Year >= (
            CASE
                WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                    THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
                ELSE EXTRACT(YEAR FROM CURRENT_DATE())
            END
        ) - 3)
        -- Fallback to AcademicYear if Year doesn't exist (some tables use AcademicYear)
        OR (AcademicYear >= (
            CASE
                WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                    THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
                ELSE EXTRACT(YEAR FROM CURRENT_DATE())
            END
        ) - 3)
    )"""

    # Optional school filter pushdown
    schools = [] if include_districtwide else (filters.get("schools") or [])
    if not isinstance(schools, list):
        schools = []
    school_clause = None
    if schools:
        school_col = _pick_col(["Learning_Center", "learning_center", "SchoolName", "School_Name", "School", "school"])
        if school_col:
            school_expr = f"LOWER(CAST({school_col} AS STRING))"
            school_clause = _sql_like_any(school_expr, schools)

    where_sql = year_filter
    if school_clause:
        where_sql = f"({year_filter}) AND {school_clause}"

    return f"""
        SELECT DISTINCT *
        EXCEPT (

        -- Comment the feilds below that you would like to include in the results

        -- For example, if you want to include relavant Growth Windows in your analysis, 

        -- just comment the line and it will be included in the output.

        RapidGuessingPercentage,

        -- Lexile and Quantile Scores if you actually use these in an analysis

        LexileScore,

        LexileMin,

        LexileMax,

        QuantileScore,

        QuantileMin,

        QuantileMax,

    

        -- Weeks of Instruction

        WISelectedAYFall,

        WISelectedAYWinter,

        WISelectedAYSpring,

        WIPreviousAYFall,

        WIPreviousAYWinter,

        WIPreviousAYSpring,

        

        -- Domains not applicable to Math and Reading

        Goal5Name,

        Goal5RitScore,

        Goal5StdErr,

        Goal5Range,

        Goal5Adjective,

        Goal6Name,

        Goal6RitScore,

        Goal6StdErr,

        Goal6Range,

        Goal6Adjective,

        Goal7Name,

        Goal7RitScore,

        Goal7StdErr,

        Goal7Range,

        Goal7Adjective,

        Goal8Name,

        Goal8RitScore,

        Goal8StdErr,

        Goal8Range,

        Goal8Adjective,

        AccommodationCategories,

        Accommodations,

    

        --Growth columns by Window. Dynamically included based on selected quarters
{f'{chr(10)}        {growth_excludes_sql},' if growth_excludes_sql else ''}

        --Typical Growth by Window. Dynamically included based on selected quarters
{f'{chr(10)}        {typical_growth_excludes_sql},' if typical_growth_excludes_sql else ''}

        

        -- Projected Prof Studies. Only Study 2 included because it is SBAC. Add others if interest (ACT, SAT, etc.)

        ProjectedProficiencyStudy1,

        ProjectedProficiencyLevel1,

        ProjectedProficiencyStudy3,

        ProjectedProficiencyLevel3,

        ProjectedProficiencyStudy4,

        ProjectedProficiencyLevel4,

        ProjectedProficiencyStudy5,

        ProjectedProficiencyLevel5,

        ProjectedProficiencyStudy6,

        ProjectedProficiencyLevel6,

        ProjectedProficiencyStudy7,

        ProjectedProficiencyLevel7,

        ProjectedProficiencyStudy8,

        ProjectedProficiencyLevel8,

        ProjectedProficiencyStudy9,

        ProjectedProficiencyLevel9,

        ProjectedProficiencyStudy10,

        ProjectedProficiencyLevel10,

        InstructionalDayWeight,

        

        -- These create LOTS of duplicate rows and were only created for 

        -- Matched Cohort Analysis on the Dashboards. Comment out with caution

        Match,

        MatchYear,

        MatchWindow,

        WindowOrder{',' if dynamic_excludes else ''}{dynamic_excludes}

      ),

      CASE

        WHEN Accommodations IS NOT NULL THEN 'Yes'

        ELSE 'No'

      END AS TestedWithAccommodations,

    

    -- You will need to update the table below with a new partner table

    FROM 

        `{table_id}`

    WHERE {where_sql}

    """
