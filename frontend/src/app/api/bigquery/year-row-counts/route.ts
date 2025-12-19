import { createBigQueryClient } from '@/lib/bigquery'
import { auth } from '@clerk/nextjs/server'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/bigquery/year-row-counts
 *
 * Computes row counts by year for the selected assessment table(s).
 *
 * Query params:
 *   - projectId: string (required)
 *   - datasetId: string (required)
 *   - location?: string (optional, default: US)
 *   - assessments?: string (optional) - comma-separated list of assessments (nwea, iready, star)
 *   - tablePaths?: string (optional) - comma-separated list of specific table paths; route will
 *                extract table names and match them to selected assessments when possible
 *   - schools?: string (optional) - comma-separated list of selected school names; if provided,
 *                counts will be filtered to those schools (non-district scope)
 *   - windows?: string (optional) - comma-separated list of test windows to include (e.g. "Fall,Winter");
 *                if provided, counts will be filtered to those windows (case-insensitive).
 *
 * Returns:
 *   - success: boolean
 *   - totals_by_year: Record<string, number>
 *   - by_assessment: Record<string, Record<string, number>>
 *   - tables_used: Array<{ assessmentType: string, tableName: string, yearColumn: string }>
 */
export async function GET(req: NextRequest) {
    try {
        const { userId } = await auth()
        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        const toSafeNumber = (v: unknown): number => {
            try {
                if (typeof v === 'number') return Number.isFinite(v) ? v : 0
                if (typeof v === 'bigint') return Number(v)
                if (typeof v === 'string') {
                    const n = parseInt(v, 10)
                    return Number.isFinite(n) ? n : 0
                }
                if (v && typeof v === 'object') {
                    const maybe = v as { value?: unknown; toString?: () => string }
                    // BigQuery sometimes returns INT64 values as strings, or as objects that stringify.
                    if (maybe.value !== undefined) return toSafeNumber(maybe.value)
                    if (typeof maybe.toString === 'function') return toSafeNumber(maybe.toString())
                }
            } catch {
                // ignore
            }
            return 0
        }

        const projectId = req.nextUrl.searchParams.get('projectId')
        const datasetId = req.nextUrl.searchParams.get('datasetId')
        const location = req.nextUrl.searchParams.get('location') || 'US'
        const assessmentsParam = req.nextUrl.searchParams.get('assessments')
        const tablePathsParam = req.nextUrl.searchParams.get('tablePaths')
        const schoolsParam = req.nextUrl.searchParams.get('schools')
        const windowsParam = req.nextUrl.searchParams.get('windows')

        if (!projectId || !datasetId) {
            return NextResponse.json({ success: false, error: 'projectId and datasetId query parameters are required' }, { status: 400 })
        }

        const specificTablePaths = tablePathsParam ? tablePathsParam.split(',').map((t) => t.trim()) : null
        const selectedSchools = schoolsParam
            ? schoolsParam
                  .split(',')
                  .map((s) => s.trim())
                  .filter(Boolean)
            : []

        // Per-assessment school lists: schools_nwea=..., schools_iready=..., etc.
        const perAssessmentSchools: Record<string, string[]> = {}
        req.nextUrl.searchParams.forEach((value, key) => {
            if (!key.startsWith('schools_')) return
            const aid = key.slice('schools_'.length).trim().toLowerCase()
            if (!aid) return
            const vals = value
                .split(',')
                .map((s) => s.trim())
                .filter(Boolean)
            if (vals.length > 0) perAssessmentSchools[aid] = vals
        })
        const selectedWindows = windowsParam
            ? windowsParam
                  .split(',')
                  .map((w) => w.trim())
                  .filter(Boolean)
            : []
        const requestedAssessments = assessmentsParam
            ? assessmentsParam
                  .split(',')
                  .map((a) => a.trim().toLowerCase())
                  .filter(Boolean)
            : null

        const client = createBigQueryClient(projectId, location)
        const dataset = client.dataset(datasetId)

        // Map assessment IDs to table name patterns
        const assessmentTableMap: Record<string, string[]> = {
            nwea: ['nwea_production_calpads_v4_2', 'Nwea_production_calpads_v4_2'],
            iready: ['iready_production_calpads_v4_2', 'iReady_production_calpads_v4_2'],
            star: ['renaissance_production_calpads_v4_2', 'Renaissance_production_calpads_v4_2'],
            renaissance: ['renaissance_production_calpads_v4_2', 'Renaissance_production_calpads_v4_2']
        }

        const tablesToQuery: Array<{ tableName: string; assessmentType: string }> = []

        if (requestedAssessments && requestedAssessments.length > 0) {
            for (const assessment of requestedAssessments) {
                let patterns = assessmentTableMap[assessment] || []

                // If specific table paths are provided, use those instead
                if (specificTablePaths) {
                    const relevantTables = specificTablePaths.filter((path) => {
                        const tableName = path.split('.').pop() || path
                        return tableName.toLowerCase().includes(assessment.toLowerCase())
                    })
                    if (relevantTables.length > 0) {
                        patterns = relevantTables.map((path) => path.split('.').pop() || path)
                    }
                }

                for (const pattern of patterns) {
                    try {
                        const table = dataset.table(pattern)
                        const [exists] = await table.exists()
                        if (exists) {
                            tablesToQuery.push({ tableName: pattern, assessmentType: assessment })
                            break
                        }
                    } catch {
                        continue
                    }
                }
            }
        } else {
            // If no assessments specified, do nothing (caller should pass at least one)
            return NextResponse.json({
                success: true,
                totals_by_year: {},
                by_assessment: {},
                tables_used: []
            })
        }

        if (tablesToQuery.length === 0) {
            return NextResponse.json({
                success: true,
                totals_by_year: {},
                by_assessment: {},
                tables_used: []
            })
        }

        const yearColumns = ['AcademicYear', 'academicyear', 'Year', 'year', 'testyear']

        // Heuristic: detect a usable school expression for filtering.
        const schoolCandidates = ['learning_center', 'Learning_Center', 'SchoolName', 'School_Name', 'schoolname', 'school_name', 'School', 'school']

        const escapeSqlString = (s: string) => s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")
        const buildListSql = (vals: string[]) => (vals.length > 0 ? `UNNEST([${vals.map((s) => `'${escapeSqlString(s.toLowerCase())}'`).join(', ')}])` : null)
        const _schoolListSql = buildListSql(selectedSchools)

        // Heuristic: detect a usable test window column.
        const windowCandidates = ['testwindow', 'TestWindow', 'window', 'Window', 'benchmarkperiod', 'BenchmarkPeriod', 'term', 'Term']
        const windowsListSql = selectedWindows.length > 0 ? `UNNEST([${selectedWindows.map((w) => `'${escapeSqlString(w)}'`).join(', ')}])` : null

        const totalsByYear: Record<string, number> = {}
        const byAssessment: Record<string, Record<string, number>> = {}
        const tablesUsed: Array<{ assessmentType: string; tableName: string; yearColumn: string }> = []

        for (const { tableName, assessmentType } of tablesToQuery) {
            let schoolExpr: string | null = null
            let windowExpr: string | null = null
            let schemaFieldMap: Record<string, string> = {}

            // Discover the year column
            let yearColumn: string | null = null
            for (const yearCol of yearColumns) {
                try {
                    const testQuery = `SELECT \`${yearCol}\` FROM \`${projectId}.${datasetId}.${tableName}\` LIMIT 1`
                    await client.query({ query: testQuery, location })
                    yearColumn = yearCol
                    break
                } catch {
                    continue
                }
            }

            if (!yearColumn) {
                console.log(
                    `[Year Row Counts] No year column found for ${assessmentType} table ${projectId}.${datasetId}.${tableName} (tried: ${yearColumns.join(
                        ', '
                    )})`
                )
                continue
            }

            tablesUsed.push({ assessmentType, tableName, yearColumn })

            // Fetch schema (needed for safe EXCEPT(...) on non-standard tables).
            try {
                const [metadata] = await dataset.table(tableName).getMetadata()
                const schemaMeta = metadata as { schema?: { fields?: Array<{ name?: string }> } }
                const fields = schemaMeta?.schema?.fields || []
                schemaFieldMap = {}
                for (const f of fields) {
                    if (f?.name) schemaFieldMap[String(f.name).toLowerCase()] = String(f.name)
                }
            } catch {
                schemaFieldMap = {}
            }

            // If windows were provided, try to find a window column/expression.
            if (selectedWindows.length > 0 && windowsListSql) {
                for (const wcol of windowCandidates) {
                    try {
                        const testQuery = `SELECT \`${wcol}\` FROM \`${projectId}.${datasetId}.${tableName}\` LIMIT 1`
                        await client.query({ query: testQuery, location })
                        // normalize to lowercase for comparison
                        windowExpr = `LOWER(CAST(\`${wcol}\` AS STRING))`
                        break
                    } catch {
                        continue
                    }
                }
                if (!windowExpr) {
                    console.log(
                        `[Year Row Counts] windows filter provided but no window column found for ${assessmentType} table ${projectId}.${datasetId}.${tableName}`
                    )
                }
            }

            // If schools were provided, try to find a school column/expression.
            const schoolsForThisAssessment = perAssessmentSchools[assessmentType] || selectedSchools
            const schoolListSqlForThis = buildListSql(schoolsForThisAssessment)

            if (schoolsForThisAssessment.length > 0 && schoolListSqlForThis) {
                for (const schoolCol of schoolCandidates) {
                    try {
                        const testQuery = `SELECT \`${schoolCol}\` FROM \`${projectId}.${datasetId}.${tableName}\` LIMIT 1`
                        await client.query({ query: testQuery, location })
                        schoolExpr = `LOWER(TRIM(CAST(\`${schoolCol}\` AS STRING)))`
                        break
                    } catch {
                        continue
                    }
                }

                // NWEA sometimes uses learning_center + SchoolName; try a combined expression if both exist.
                if (!schoolExpr) {
                    try {
                        const testQuery = `SELECT COALESCE(learning_center, SchoolName) AS school FROM \`${projectId}.${datasetId}.${tableName}\` LIMIT 1`
                        await client.query({ query: testQuery, location })
                        schoolExpr = 'LOWER(TRIM(CAST(COALESCE(learning_center, Learning_Center, SchoolName, School_Name) AS STRING)))'
                    } catch {
                        // ignore
                    }
                }

                if (!schoolExpr) {
                    console.log(
                        `[Year Row Counts] schools filter provided but no school column found for ${assessmentType} table ${projectId}.${datasetId}.${tableName}`
                    )
                }
            }

            const windowClause = windowExpr && windowsListSql ? `AND ${windowExpr} IN (SELECT LOWER(CAST(x AS STRING)) FROM ${windowsListSql} AS x)` : ''

            const buildTokenLikeClause = (expr: string, schools: string[]) => {
                const stop = new Set(['school', 'elementary', 'middle', 'high', 'academy', 'charter', 'k8', 'k-8', 'k12', 'k-12', 'of', 'the'])
                const esc = (s: string) => escapeSqlString(s)
                const clauses: string[] = []
                for (const raw of schools) {
                    const norm = String(raw || '')
                        .trim()
                        .toLowerCase()
                        .replace(/[^a-z0-9\\s]+/g, ' ')
                        .replace(/\\s+/g, ' ')
                        .trim()
                    if (!norm) continue
                    let tokens = norm.split(' ').filter((t) => t && !stop.has(t) && t.length >= 3)
                    if (tokens.length === 0) tokens = [norm]
                    const ands = tokens.map((t) => `${expr} LIKE '%${esc(t)}%'`)
                    clauses.push('(' + ands.join(' AND ') + ')')
                }
                if (clauses.length === 0) return ''
                return `AND (${clauses.join(' OR ')})`
            }

            // iReady school names are often not exact matches (suffixes like "School" omitted),
            // so use token-based LIKE matching (mirrors backend iReady ingestion behavior).
            const isIreadySchoolFilter = assessmentType === 'iready'
            const schoolClause =
                schoolExpr && schoolsForThisAssessment.length > 0
                    ? isIreadySchoolFilter
                        ? buildTokenLikeClause(schoolExpr, schoolsForThisAssessment)
                        : `AND ${schoolExpr} IN ${schoolListSqlForThis}`
                    : ''

            const tableRef = `\`${projectId}.${datasetId}.${tableName}\``

            // For NWEA and iReady, mimic ingestion-style queries by excluding noisy columns
            // (including `email` if present) BEFORE counting rows by year. This keeps the
            // "size check" aligned with what we'd actually ingest.
            const isNwea = assessmentType === 'nwea'
            const isIready = assessmentType === 'iready'

            const query = (() => {
                if (!isNwea && !isIready) {
                    // Classic query shape (like the one you provided)
                    return `
                        SELECT
                            \`${yearColumn}\` AS Year,
                            COUNT(*) AS row_count
                        FROM ${tableRef}
                        WHERE \`${yearColumn}\` IS NOT NULL
                        ${windowClause}
                        ${schoolClause}
                        GROUP BY Year
                        ORDER BY Year
                    `
                }

                // Base exclude list per assessment + email if it exists.
                const staticExcludes = [
                    ...(isNwea
                        ? [
                              'RapidGuessingPercentage',
                              'LexileScore',
                              'LexileMin',
                              'LexileMax',
                              'QuantileScore',
                              'QuantileMin',
                              'QuantileMax',
                              'WISelectedAYFall',
                              'WISelectedAYWinter',
                              'WISelectedAYSpring',
                              'WIPreviousAYFall',
                              'WIPreviousAYWinter',
                              'WIPreviousAYSpring',
                              'Goal5Name',
                              'Goal5RitScore',
                              'Goal5StdErr',
                              'Goal5Range',
                              'Goal5Adjective',
                              'Goal6Name',
                              'Goal6RitScore',
                              'Goal6StdErr',
                              'Goal6Range',
                              'Goal6Adjective',
                              'Goal7Name',
                              'Goal7RitScore',
                              'Goal7StdErr',
                              'Goal7Range',
                              'Goal7Adjective',
                              'Goal8Name',
                              'Goal8RitScore',
                              'Goal8StdErr',
                              'Goal8Range',
                              'Goal8Adjective',
                              'AccommodationCategories',
                              'Accommodations',
                              'InstructionalDayWeight',
                              'Match',
                              'MatchYear',
                              'MatchWindow',
                              'WindowOrder'
                          ]
                        : []),
                    ...(isIready
                        ? [
                              // Matches backend/python/iready/sql_builders.py intent
                              'Hispanic_or_Latino',
                              'Race',
                              'English_Language_Learner',
                              'Special_Education',
                              'Economically_Disadvantaged',
                              'Migrant',
                              'Measure',
                              'Range',
                              'Annual_Typical_Growth_Percent',
                              'Annual_Stretch_Growth_Percent',
                              'Relative_Tier_Placement',
                              'Relative_5Tier_Placement',
                              'TestView',
                              'Match',
                              'MatchYear',
                              'MatchWindow',
                              'WindowOrder'
                          ]
                        : [])
                ]

                // Growth/Typical growth excludes based on selected windows (NWEA only).
                const growthSuffixes = [
                    'ProjectedGrowth',
                    'ObservedGrowth',
                    'ObservedGrowthSE',
                    'MetProjectedGrowth',
                    'ConditionalGrowthIndex',
                    'ConditionalGrowthPercentile',
                    'GrowthQuintile'
                ]
                const allGrowthPrefixes = ['FallToFall', 'FallToWinter', 'FallToSpring', 'WinterToWinter', 'WinterToSpring', 'SpringToSpring']
                const allGrowthCols: string[] = []
                for (const p of allGrowthPrefixes) for (const s of growthSuffixes) allGrowthCols.push(`${p}${s}`)

                const allTypicalCols = [
                    'TypicalFallToFallGrowth',
                    'TypicalFallToWinterGrowth',
                    'TypicalFallToSpringGrowth',
                    'TypicalWinterToWinterGrowth',
                    'TypicalWinterToSpringGrowth',
                    'TypicalSpringToSpringGrowth'
                ]

                const keepGrowthPrefixes = new Set<string>()
                const keepTypical = new Set<string>()
                const windowsNorm = selectedWindows.map((w) => String(w).trim().toLowerCase())
                const hasFall = windowsNorm.includes('fall')
                const hasWinter = windowsNorm.includes('winter')
                const hasSpring = windowsNorm.includes('spring')

                if (isNwea) {
                    if (hasFall) {
                        ;['FallToFall', 'FallToWinter', 'FallToSpring'].forEach((p) => keepGrowthPrefixes.add(p))
                        ;['TypicalFallToFallGrowth', 'TypicalFallToWinterGrowth', 'TypicalFallToSpringGrowth'].forEach((c) => keepTypical.add(c))
                    }
                    if (hasWinter) {
                        ;['FallToWinter', 'WinterToWinter', 'WinterToSpring'].forEach((p) => keepGrowthPrefixes.add(p))
                        ;['TypicalFallToWinterGrowth', 'TypicalWinterToWinterGrowth', 'TypicalWinterToSpringGrowth'].forEach((c) => keepTypical.add(c))
                    }
                    if (hasSpring) {
                        ;['SpringToSpring'].forEach((p) => keepGrowthPrefixes.add(p))
                        ;['TypicalSpringToSpringGrowth'].forEach((c) => keepTypical.add(c))
                    }
                }

                // Exclude all growth columns except those whose prefix is required for the selected windows.
                const growthExcludes = isNwea
                    ? keepGrowthPrefixes.size > 0
                        ? allGrowthCols.filter((c) => !Array.from(keepGrowthPrefixes).some((p) => c.startsWith(p)))
                        : allGrowthCols
                    : []

                const typicalExcludes = isNwea ? (keepTypical.size > 0 ? allTypicalCols.filter((c) => !keepTypical.has(c)) : allTypicalCols) : []

                const requested = [...staticExcludes, ...growthExcludes, ...typicalExcludes]
                // Always exclude email if present (non-standard production tables).
                if (schemaFieldMap['email']) requested.push(schemaFieldMap['email'])

                // Only exclude columns that actually exist in the table schema (avoid BigQuery errors).
                const existing: string[] = []
                for (const col of requested) {
                    const actual = schemaFieldMap[String(col).toLowerCase()]
                    if (actual) existing.push(actual)
                }
                const excludesSql = existing.length > 0 ? existing.map((c) => `\`${c}\``).join(', ') : ''

                const hasAccommodations = Boolean(schemaFieldMap['accommodations'])
                const computed =
                    isNwea && hasAccommodations
                        ? `, CASE WHEN \`${schemaFieldMap['accommodations']}\` IS NOT NULL THEN 'Yes' ELSE 'No' END AS TestedWithAccommodations`
                        : ''

                const selectDistinct = excludesSql.length > 0 ? `SELECT DISTINCT * EXCEPT (${excludesSql})${computed}` : `SELECT DISTINCT *${computed}`

                return `
                    WITH base AS (
                        ${selectDistinct}
                        FROM ${tableRef}
                        WHERE \`${yearColumn}\` IS NOT NULL
                        ${windowClause}
                        ${schoolClause}
                    )
                    SELECT
                        CAST(\`${yearColumn}\` AS STRING) AS Year,
                        COUNT(*) AS row_count
                    FROM base
                    GROUP BY Year
                    ORDER BY Year
                `
            })()

            console.log(
                `[Year Row Counts] Executing for ${assessmentType}: table=${projectId}.${datasetId}.${tableName} yearColumn=${yearColumn}${
                    schoolExpr && schoolsForThisAssessment.length > 0 ? ` schoolFilter=${schoolsForThisAssessment.length}` : ''
                }\n${query}`
            )

            const [rows] = await client.query({ query, location })

            if (!byAssessment[assessmentType]) byAssessment[assessmentType] = {}

            rows.forEach((r: { Year?: unknown; year?: unknown; row_count?: unknown }) => {
                const y = String((r.Year ?? r.year) || '').trim()
                const c = toSafeNumber(r.row_count)
                if (!y) return

                byAssessment[assessmentType][y] = (byAssessment[assessmentType][y] || 0) + c
                totalsByYear[y] = (totalsByYear[y] || 0) + c
            })

            const keys = Object.keys(byAssessment[assessmentType] || {})
            console.log(
                `[Year Row Counts] Result ${assessmentType}: ${keys.length} year(s) -> ${keys
                    .sort()
                    .slice(-8)
                    .map((k) => `${k}:${byAssessment[assessmentType][k]}`)
                    .join(', ')}${keys.length > 8 ? ' ...' : ''}`
            )
        }

        return NextResponse.json({
            success: true,
            totals_by_year: totalsByYear,
            by_assessment: byAssessment,
            tables_used: tablesUsed
        })
    } catch (error) {
        console.error('[Year Row Counts] Error:', error)
        return NextResponse.json({ success: false, error: 'Error fetching year row counts' }, { status: 500 })
    }
}
