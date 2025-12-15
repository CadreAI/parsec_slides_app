"""
SQL query builders for BigQuery
"""
from typing import List, Optional

# EXCLUDE_COLS can be imported from config if needed
EXCLUDE_COLS = {}


def sql_nwea(table_id: str, exclude_cols: Optional[List[str]] = None, year_column: Optional[str] = None) -> str:
    """
    Build SQL query for NWEA data
    
    Args:
        table_id: BigQuery table ID (project.dataset.table)
        exclude_cols: Optional list of additional columns to exclude from config
        year_column: Optional year column name ('Year' or 'AcademicYear'). If None, will use OR condition for both.
    
    Returns:
        SQL query string
    """
    extra_excludes = EXCLUDE_COLS.get("nwea", [])
    if exclude_cols:
        extra_excludes = extra_excludes + exclude_cols
    
    dynamic_excludes = ",\n        ".join(extra_excludes) if extra_excludes else ""
    
    # Determine year column to use
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

    

        --Typical Growth by Window. Uncomment to keep relavant window

        --TypicalFallToFallGrowth,

        --TypicalFallToWinterGrowth,

        --TypicalFallToSpringGrowth,

        --TypicalWinterToWinterGrowth,

        --TypicalWinterToSpringGrowth,

        --TypicalSpringToSpringGrowth,

        

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

    WHERE {year_filter}

    """
