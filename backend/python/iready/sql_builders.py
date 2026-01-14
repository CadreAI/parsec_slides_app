"""
SQL query builders for BigQuery (iReady)

Policy:
- Always filter by years in SQL.
- If schools are selected, also filter by schools in SQL to reduce result size.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime


def sql_iready(
    table_id: str,
    exclude_cols: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    year_column: Optional[str] = None
) -> str:
    """
    Build SQL query for iReady data
    
    Args:
        table_id: BigQuery table ID (project.dataset.table)
        exclude_cols: Optional list of columns to exclude from the query
        filters: Optional filters dict with years (other filters applied in Python)
        year_column: Optional year column name ('Year' or 'AcademicYear'). If None, uses COALESCE.
    
    Returns:
        SQL query string with year filtering in SQL
    """
    filters = filters or {}

    def _sql_escape(s: str) -> str:
        return str(s).replace("'", "\\'")

    def _sql_like_tokens_any(lower_expr: str, phrases: list[str]) -> Optional[str]:
        """
        Token-based fuzzy LIKE matcher.
        - For each phrase (e.g. "ADDAMS ELEMENTARY SCHOOL"), build an AND of meaningful tokens.
        - Combine phrases with OR.
        This avoids brittle exact-substring matching when table values omit suffixes like "School".
        """
        import re

        stop = {
            "school",
            "elementary",
            "middle",
            "high",
            "academy",
            "charter",
            "k8",
            "k-8",
            "k12",
            "k-12",
            "of",
            "the",
        }

        clauses: list[str] = []
        for p in phrases or []:
            if p is None:
                continue
            raw = str(p).strip().lower()
            if not raw:
                continue
            # Normalize punctuation/whitespace
            raw = re.sub(r"[^a-z0-9\s]+", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
            tokens = [t for t in raw.split() if t and t not in stop and len(t) >= 3]
            if not tokens:
                tokens = [raw]
            ands = [f"{lower_expr} LIKE '%{_sql_escape(t)}%'" for t in tokens]
            clauses.append("(" + " AND ".join(ands) + ")")

        if not clauses:
            return None
        return "(" + " OR ".join(clauses) + ")"

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
    
    # Base columns that should be included from iReady data
    base_columns = [
        "School",
        "Last_Name",
        "First_Name",
        "Student_ID",
        "Student_Grade",
        "TestWindow",
        "AcademicYear",
        "Enrolled",
        "User_Name",
        "Sex",
        "Hispanic_or_Latino",
        "Race",
        "English_Language_Learner",
        "Special_Education",
        "Economically_Disadvantaged",
        "Migrant",
        "Class_es_",
        "Class_Teacher_s_",
        "Report_Group_s_",
        "Start_Date",
        "Completion_Date",
        "Duration__min_",
        "Rush_Flag",
        "Percentile",
        "Grouping",
        "Diagnostic_Gain",
        "Annual_Typical_Growth_Measure",
        "Annual_Stretch_Growth_Measure",
        "Mid_On_Grade_Level_Scale_Score",
        "Reading_Difficulty_Indicator__Y_N_",
        "Subject",
        "Proficiency_if_Student_Shows_No_Additional_Growth",
        "Projection_if_Student_Achieves_Typical_Growth",
        "Projection_if_Student_Achieves_Stretch_Growth",
        "Percent_Progress_to_Annual_Typical_Growth____",
        "Percent_Progress_to_Annual_Stretch_Growth____",
        "Scale_Score",
        "Placement",
        "Relative_Placement",
        "Domain",
        "Baseline_Diagnostic",
        "Most_Recent_Diagnostic",
        "Domain_Order",
        "Measure",
        "Range",
        "Annual_Typical_Growth_Percent",
        "Annual_Stretch_Growth_Percent",
        "Relative_Tier_Placement",
        "Relative_5Tier_Placement",
        "TestView",
        "Match",
        "MatchYear",
        "MatchWindow",
        "WindowOrder",
        "UniqueIdentifier",
        "cers_ScaleScoreAchievementLevel",
        "cers_ScaleScore",
        "cers_Overall_PerformanceBand",
        "Year",
        "SchoolCode",
        "SchoolName",
        "SSID",
        "LocalID",
        "StudentName",
        "Grade",
        "Gender",
        "EthnicityRace",
        "EnglishLearner",
        "StudentswithDisabilities",
        "SocioEconomicallyDisadvantaged",
        "TitleIIIEligibleImmigrants",
        "TitleIPartCMigrant",
        "ELASDesignation",
        "Foster",
        "Homeless",
        "Enrollment_Length",
        "Enrollment_Length_String",
        "learning_center",
        "program",
        "learning_studio",
    ]
    
    # Columns to exclude from the base columns
    exclude_list = [
        # Demographics (less robust than CALPADS)
        "Hispanic_or_Latino",
        "Race",
        "English_Language_Learner",
        "Special_Education",
        "Economically_Disadvantaged",
        "Migrant",
        
        # Report groups (teacher-created, meaningless for analysis)
        "Report_Group_s_",
        "Grouping",
        
        # Lexile / Quantile
        "Measure",
        "Range",
        
        # Redundant columns (iReady + Parsec)
        "Annual_Typical_Growth_Percent",
        "Annual_Stretch_Growth_Percent",
        "Relative_Tier_Placement",
        "Relative_5Tier_Placement",
        
        # Duplicate creators
        "TestView",
        "Match",
        "MatchYear",
        "MatchWindow",
        "WindowOrder",
    ]
    
    # Add email if present (check available_columns)
    if available_cols and "email" in available_cols:
        exclude_list.append("email")
    
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
    where_conditions = [ "Domain = 'Overall'"]
    
    # Year filter (iReady might use AcademicYear or Year column)
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
        # Default: last 3 years (matching the data ingestion script pattern)
        where_conditions.append(f"""{year_col_name} >= (
            CASE
                WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                    THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
                ELSE EXTRACT(YEAR FROM CURRENT_DATE())
            END
        ) - 3""")

    # Optional school filter pushdown.
    # If districtwide is included, do NOT push down school-name filtering:
    # district aggregate needs all schools; school selection happens later in chart generation.
    include_districtwide = bool(filters.get("include_districtwide"))
    schools = [] if include_districtwide else (filters.get("schools") or [])
    if not isinstance(schools, list):
        schools = []
    if schools:
        # Check all possible school name columns and combine them with OR
        school_col_candidates = ["SchoolName", "School_Name", "School", "learning_center", "Learning_Center","program",
        "learning_studio", "Learning_Studio","Program","Learning_Center","Learning_Studio"]
        available_school_cols = [col for col in school_col_candidates if col.lower() in available_cols]
        
        if available_school_cols:
            # Build OR clause for all available school columns
            school_clauses = []
            for school_col in available_school_cols:
                school_expr = f"LOWER(CAST(`{school_col}` AS STRING))"
                like_clause = _sql_like_tokens_any(school_expr, schools) or _sql_like_any(school_expr, schools)
                if like_clause:
                    school_clauses.append(like_clause)
            
            if school_clauses:
                # Combine all school column searches with OR
                combined_school_clause = "(" + " OR ".join(school_clauses) + ")"
                where_conditions.append(combined_school_clause)
    
    where_clause = " AND ".join(where_conditions)
    
    return f"""
    SELECT DISTINCT
        {columns_sql}
    FROM `{table_id}`
    WHERE {where_clause}
    """

