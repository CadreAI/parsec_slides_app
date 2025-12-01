// PartnerConfig type not used in this file but kept for future reference

export function sqlCers(tableId: string): string {
    return `
    SELECT DISTINCT *
    FROM \`${tableId}\`
    WHERE districtname IN UNNEST(@districts)
    AND schoolyear >= (
        CASE
            WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
            ELSE EXTRACT(YEAR FROM CURRENT_DATE())
        END
        ) - 3
    `
}

export function sqlIab(tableId: string): string {
    return `
    SELECT DISTINCT *
    FROM \`${tableId}\`
    WHERE LocalID <> 99999
    AND schoolyear >= (
        CASE
            WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
            ELSE EXTRACT(YEAR FROM CURRENT_DATE())
        END
        ) - 3
    `
}

export function sqlCalpads(tableId: string): string {
    return `
    SELECT DISTINCT *
    FROM \`${tableId}\`
    WHERE Year >= (
        CASE
            WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
            ELSE EXTRACT(YEAR FROM CURRENT_DATE())
        END
        ) - 3
    `
}

export function sqlNwea(tableId: string, excludeCols: string[] = []): string {
    const dynamicExcludes = excludeCols.length > 0 ? ',\n        ' + excludeCols.join(',\n        ') : ''

    return `
        SELECT DISTINCT *
        EXCEPT (
        -- Comment the fields below that you would like to include in the results
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
        WindowOrder${dynamicExcludes}
      ),
      CASE
        WHEN Accommodations IS NOT NULL THEN 'Yes'
        ELSE 'No'
      END AS TestedWithAccommodations
    FROM 
        \`${tableId}\`
    WHERE Year >= (
        CASE
            WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
            ELSE EXTRACT(YEAR FROM CURRENT_DATE())
        END
        ) - 3
    `
}

export function sqlStar(tableId: string, excludeCols: string[] = []): string {
    const dynamicExcludes = excludeCols.length > 0 ? ',\n        ' + excludeCols.join(',\n        ') : ''

    return `
    SELECT DISTINCT * 
      EXCEPT(
        Teacher_State_ID,
        Teacher_Identifier,
        Lexile_Score,
        Lexile_Range,
        Quantile_Measure,
        Quantile_Range,
        WindowOrder,
        Match,
        MatchYear,
        MatchWindow,
        MatchOrder${dynamicExcludes}
      )
    FROM
      \`${tableId}\`
    WHERE AcademicYear >= (
        CASE
            WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
            ELSE EXTRACT(YEAR FROM CURRENT_DATE())
        END
        ) - 3
    `
}

export function sqlIready(tableId: string, excludeCols: string[] = []): string {
    const dynamicExcludes = excludeCols.length > 0 ? ',\n        ' + excludeCols.join(',\n        ') : ''

    return `
    SELECT DISTINCT * 
      EXCEPT(
         Hispanic_or_Latino,
         Race,
         English_Language_Learner,
         Special_Education,
         Economically_Disadvantaged,
         Migrant,
        Report_Group_s_,
        \`Grouping\`,
        Measure,
        \`Range\`,
        Annual_Typical_Growth_Percent,
        Annual_Stretch_Growth_Percent,
        Relative_Tier_Placement,
        Relative_5Tier_Placement,
        TestView,
        Match,
        MatchYear,
        MatchWindow,
        WindowOrder${dynamicExcludes}
    )
    FROM \`${tableId}\` 
    WHERE Enrolled = 'Enrolled'
    AND AcademicYear >= (
        CASE
            WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
            ELSE EXTRACT(YEAR FROM CURRENT_DATE())
        END
        ) - 3
    `
}
