"""
SQL query builders for BigQuery
"""
from typing import List, Optional, Dict


def sql_nwea(
    table_id: str,
    filters: Optional[Dict] = None,
    apply_grade_filter: bool = False
) -> str:
    """
    Build SQL query for NWEA data
    
    Args:
        table_id: BigQuery table ID (project.dataset.table)
        filters: Optional filters dict with districts, years, schools, grades
        apply_grade_filter: If True, apply grade filter in SQL. If False, fetch all grades
                           (False for district aggregates, True for school-level queries)
    
    Returns:
        SQL query string
    """
    filters = filters or {}
    where_conditions = []
    
    # Year filter
    if filters.get('years') and len(filters['years']) > 0:
        year_list = ', '.join(map(str, filters['years']))
        where_conditions.append(f"Year IN ({year_list})")
    else:
        # Default: last 3 years
        where_conditions.append("""Year >= (
            CASE
                WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                    THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
                ELSE EXTRACT(YEAR FROM CURRENT_DATE())
            END
        ) - 3""")
    
    # Note: DistrictName column doesn't exist in NWEA table, so district filtering
    # must be done in Python after data retrieval based on school-to-district mapping
    
    # School filter (column name is "School" not "SchoolName")
    if filters.get('schools') and len(filters['schools']) > 0:
        school_list = "', '".join(filters['schools'])
        where_conditions.append(f"School IN ('{school_list}')")
    
    # Grade filter: Apply in SQL only for school-level queries (not district aggregates)
    if apply_grade_filter and filters.get('grades') and len(filters['grades']) > 0:
        grade_list = ', '.join(map(str, filters['grades']))
        where_conditions.append(f"Grade IN ({grade_list})")
        print(f"[SQL Builder] Applying grade filter in SQL: {filters['grades']}")
    
    where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
    
    sql = f"""
        SELECT *
        EXCEPT (
        RapidGuessingPercentage,
        LexileScore,
        LexileMin,
        LexileMax,
        QuantileScore,
        QuantileMin,
        QuantileMax,
        WISelectedAYFall,
        WISelectedAYWinter,
        WISelectedAYSpring,
        WIPreviousAYFall,
        WIPreviousAYWinter,
        WIPreviousAYSpring,
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
        Match,
        MatchYear,
        MatchWindow,
        WindowOrder
      ),
      CASE
        WHEN Accommodations IS NOT NULL THEN 'Yes'
        ELSE 'No'
      END AS TestedWithAccommodations
    FROM 
        `{table_id}`
    {where_clause}
    """
    
    return sql.strip()


