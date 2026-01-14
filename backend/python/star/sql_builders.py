"""
SQL query builders for BigQuery - STAR

Policy:
- Always filter by years in SQL.
- If schools are selected, also filter by schools in SQL to reduce result size.
"""
from typing import List, Optional, Dict, Any


# Default exclude columns for STAR
DEFAULT_STAR_EXCLUDES = [
    "Teacher_State_ID",
    "Teacher_Identifier",
    "Lexile_Score",
    "Lexile_Range",
    "Quantile_Measure",
    "Quantile_Range",
    "WindowOrder",
    "Match",
    "MatchYear",
    "MatchWindow",
    "MatchOrder",
]


def sql_star(
    table_id: str,
    exclude_cols: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    year_column: Optional[str] = None
) -> str:
    """
    Build SQL query for STAR data
    
    Args:
        table_id: BigQuery table ID (project.dataset.table)
        exclude_cols: Optional list of additional columns to exclude
        filters: Optional filters dict (currently only years supported in SQL)
    
    Returns:
        SQL query string
    """
    filters = filters or {}

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
    
    # Base columns that should be included from STAR data
    base_columns = [
        "AcademicYear",
        "TestWindow",
        "School_Year_Start_Date",
        "School_Year_End_Date",
        "District_Name",
        "School_Name",
        "Teacher_State_ID",
        "Teacher_Identifier",
        "Teacher_First_Name",
        "Teacher_Last_Name",
        "Student_State_ID",
        "StudentID",
        "FirstName",
        "LastName",
        "StudentGrade",
        "Class_Name",
        "Activity_Type",
        "Activity_Completed_Date",
        "Used_Extended_Time",
        "Scaled_Score",
        "Unified_Scale",
        "Lexile_Score",
        "Lexile_Range",
        "Quantile_Measure",
        "Quantile_Range",
        "Percentile_Rank",
        "Normal_Curve_Equivalent",
        "Grade_Equivalent",
        "Screening_Window_Name",
        "Screening_Window_Start",
        "Screening_Window_End",
        "Literacy_Classification",
        "Instructional_Reading_Level",
        "School_Benchmark_Category",
        "State_Benchmark_Achievement",
        "State_Benchmark_Category",
        "District_Benchmark_Achievement",
        "District_Benchmark_Level",
        "District_Benchmark_Category",
        "Moderate_Growth_Rate_in_SS_Per_Week",
        "Ambitious_Growth_Rate_in_SS_Per_Week",
        "Maintain_PR_Growth_Rate_In_SS_Per_Week",
        "Projected_SS_Moderate",
        "Projected_SS_Ambitious",
        "Projected_SS_To_Maintain_PR",
        "Current_SGP_Vector",
        "Current_SGP",
        "SGP_normativegrowth",
        "SGP_normative_order",
        "SGP_growth",
        "SGP_growth2",
        "Current",
        "Window",
        "SGP_Test_Types",
        "Window_SGP_Prior_Test_1_Date",
        "Window_SGP_Prior_Test_1_Scaled_Score",
        "Window_SGP_Prior_Test_2_Date",
        "Window_SGP_Prior_Test_2_Scaled_Score",
        "SchoolCode",
        "SchoolName",
        "SSID",
        "LocalID",
        "StudentName",
        "Grade",
        "Gender",
        "EthnicityRace",
        "ELASDesignation",
        "EnglishLearner",
        "StudentswithDisabilities",
        "SocioEconomicallyDisadvantaged",
        "TitleIIIEligibleImmigrants",
        "TitleIPartCMigrant",
        "Foster",
        "Homeless",
        "Enrollment_Length",
        "Enrollment_Length_String",
        "cers_Overall_PerformanceBand",
        "cers_GradeLevelWhenAssessed",
        "cers_subject",
        "cers_scalescoreachievementlevel",
        "cers_scalescore",
        "cers_DFS",
        "Benchmark_Achievement_Category",
        "Benchmark_Type",
        "WindowOrder",
        "Match",
        "MatchYear",
        "MatchWindow",
        "MatchOrder",
        "learning_center",
        "program",
        "learning_studio",
    ]
    
    # Columns to exclude from the base columns
    exclude_list = DEFAULT_STAR_EXCLUDES.copy()
    
    # Add dynamic excludes if provided
    if exclude_cols:
        exclude_list.extend(exclude_cols)
    
    # Filter base columns to only include those that exist in the table and are not excluded
    final_columns = []
    for col in base_columns:
        # Check if column exists in available columns (case-insensitive)
        col_exists = col.lower() in available_cols if available_cols else True
        # Check if column is not in exclude list
        col_not_excluded = col not in exclude_list
        
        if col_not_excluded and (not available_cols or col_exists):
            final_columns.append(f"`{col}`")
    
    # If no columns remain after filtering, fall back to basic columns
    if not final_columns:
        final_columns = ["*"]
    
    columns_sql = ",\n        ".join(final_columns)
    
    # Build WHERE conditions
    where_conditions = []
    
    # Year filter (STAR might use AcademicYear or Year column)
    # Use detected column or COALESCE as fallback
    if year_column:
        # Use detected column name
        year_col_name = year_column
    else:
        # Fallback to COALESCE (will fail if neither column exists, but that's expected)
        year_col_name = "COALESCE(AcademicYear, Year)"
    
    if filters.get('years') and len(filters['years']) > 0:
        year_list = ', '.join(map(str, filters['years']))
        where_conditions.append(f"{year_col_name} IN ({year_list})")
    else:
        # Default: last 3 years (matching the provided SQL pattern)
        where_conditions.append(f"""{year_col_name} >= (
            CASE
                WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                    THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
                ELSE EXTRACT(YEAR FROM CURRENT_DATE())
            END
        ) - 3""")

    # Optional school filter pushdown
    schools = filters.get("schools") or []
    if not isinstance(schools, list):
        schools = []
    if schools:
        # Check all possible school name columns and combine them with OR
        school_col_candidates = [
            "School_Name", "SchoolName", "School", 
            "learning_center", "Learning_Center",
            "program", "Program",
            "learning_studio", "Learning_Studio"
        ]
        available_school_cols = [col for col in school_col_candidates if col.lower() in available_cols]
        
        if available_school_cols:
            # Build OR clause for all available school columns
            school_clauses = []
            for school_col in available_school_cols:
                school_expr = f"LOWER(CAST(`{school_col}` AS STRING))"
                like_clause = _sql_like_any(school_expr, schools)
                if like_clause:
                    school_clauses.append(like_clause)
            
            if school_clauses:
                # Combine all school column searches with OR
                combined_school_clause = "(" + " OR ".join(school_clauses) + ")"
                where_conditions.append(combined_school_clause)
    
    where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
    
    return f"""
    SELECT DISTINCT
        {columns_sql}
    FROM
      `{table_id}`
    {where_clause}
    """

