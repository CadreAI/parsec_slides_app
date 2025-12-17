import { auth } from '@clerk/nextjs/server'
import { BigQuery } from '@google-cloud/bigquery'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/bigquery/form-options
 *
 * Fetches available grades and years from actual data tables
 *
 * Query params:
 *   - projectId: string (required) - GCP project ID
 *   - datasetId: string (required) - BigQuery dataset ID
 *   - location?: string (optional) - BigQuery location (default: US)
 *   - assessments?: string (optional) - Comma-separated list of assessment IDs (nwea, star, iready)
 *                                      If provided, queries only those tables and returns assessment-specific grade ranges
 *
 * Returns:
 *   - success: boolean
 *   - grades: string[] - Available grade options from data (filtered by assessment type)
 *   - years: string[] - Available year options from data
 */
export async function GET(req: NextRequest) {
    try {
        const { userId } = await auth()
        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        const projectId = req.nextUrl.searchParams.get('projectId')
        const datasetId = req.nextUrl.searchParams.get('datasetId')
        const location = req.nextUrl.searchParams.get('location') || 'US'
        const assessmentsParam = req.nextUrl.searchParams.get('assessments')
        const tablePathsParam = req.nextUrl.searchParams.get('tablePaths')

        // Parse specific table paths if provided
        const specificTablePaths = tablePathsParam ? tablePathsParam.split(',').map((t) => t.trim()) : null

        if (!projectId || !datasetId) {
            return NextResponse.json(
                {
                    success: false,
                    error: 'projectId and datasetId query parameters are required'
                },
                { status: 400 }
            )
        }

        // Set up BigQuery client
        process.env.GOOGLE_CLOUD_PROJECT = projectId

        // Try to resolve service account credentials
        try {
            const { resolveServiceAccountCredentialsPath } = await import('@/lib/credentials')
            const serviceAccountPath = resolveServiceAccountCredentialsPath()
            if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
                process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceAccountPath
            }
        } catch {
            // Credentials will use ADC if not found
        }

        const client = new BigQuery({
            projectId: projectId,
            location: location
        })

        const dataset = client.dataset(datasetId)

        // Parse assessments if provided
        const requestedAssessments = assessmentsParam ? assessmentsParam.split(',').map((a) => a.trim().toLowerCase()) : null
        console.log(`[Form Options] Requested assessments: ${requestedAssessments ? requestedAssessments.join(', ') : 'none (querying all)'}`)

        // Map assessment IDs to table name patterns
        const assessmentTableMap: Record<string, string[]> = {
            nwea: ['nwea_production_calpads_v4_2', 'Nwea_production_calpads_v4_2'],
            iready: ['iready_production_calpads_v4_2', 'iReady_production_calpads_v4_2'],
            star: ['renaissance_production_calpads_v4_2', 'Renaissance_production_calpads_v4_2'],
            renaissance: ['renaissance_production_calpads_v4_2', 'Renaissance_production_calpads_v4_2']
        }

        // Find tables to query based on requested assessments
        const tablesToQuery: Array<{ tableName: string; assessmentType: string }> = []

        if (requestedAssessments && requestedAssessments.length > 0) {
            // Query only requested assessment tables
            console.log(`[Form Options] Querying specific assessment tables for: ${requestedAssessments.join(', ')}`)
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

                console.log(`[Form Options] Looking for ${assessment} table with patterns: ${patterns.join(', ')}`)
                for (const pattern of patterns) {
                    try {
                        const table = dataset.table(pattern)
                        const [exists] = await table.exists()
                        if (exists) {
                            tablesToQuery.push({ tableName: pattern, assessmentType: assessment })
                            console.log(`[Form Options] ✓ Found ${assessment} table: ${pattern}`)
                            break // Found table for this assessment, move to next
                        }
                    } catch (err) {
                        console.log(`[Form Options] Table ${pattern} does not exist or error: ${err}`)
                        continue
                    }
                }
                if (!tablesToQuery.some((t) => t.assessmentType === assessment)) {
                    console.warn(`[Form Options] ⚠ No table found for assessment: ${assessment}`)
                }
            }
        } else {
            // No assessments specified - find first available table (backward compatibility)
            console.log(`[Form Options] No assessments specified, finding first available table...`)
            const allPatterns = [
                { pattern: 'nwea_production_calpads_v4_2', type: 'nwea' },
                { pattern: 'Nwea_production_calpads_v4_2', type: 'nwea' },
                { pattern: 'iready_production_calpads_v4_2', type: 'iready' },
                { pattern: 'iReady_production_calpads_v4_2', type: 'iready' },
                { pattern: 'renaissance_production_calpads_v4_2', type: 'star' },
                { pattern: 'Renaissance_production_calpads_v4_2', type: 'star' }
            ]

            for (const { pattern, type } of allPatterns) {
                try {
                    const table = dataset.table(pattern)
                    const [exists] = await table.exists()
                    if (exists) {
                        tablesToQuery.push({ tableName: pattern, assessmentType: type })
                        console.log(`[Form Options] ✓ Found first available table: ${pattern} (${type})`)
                        break
                    }
                } catch {
                    continue
                }
            }
        }

        console.log(
            `[Form Options] Will query ${tablesToQuery.length} table(s): ${tablesToQuery.map((t) => `${t.tableName} (${t.assessmentType})`).join(', ')}`
        )

        if (tablesToQuery.length === 0) {
            // Fallback to defaults if no table found
            console.warn(`[Form Options] ⚠ No tables found, using fallback grades`)
            const currentYear = new Date().getFullYear()
            const fallbackYears = [currentYear, currentYear + 1, currentYear + 2, currentYear + 3].map((y) => y.toString())
            return NextResponse.json({
                success: true,
                grades: ['-1', 'K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                years: fallbackYears
            })
        }

        // Query for distinct grades and years from all requested tables
        // Try different column name patterns - prioritize common column names
        // Note: iReady uses 'Grade' (capital G), NWEA uses 'Student_Grade' or 'grade'
        const gradeColumns = ['Grade', 'grade', 'Student_Grade', 'student_grade', 'StudentGrade']
        const yearColumns = ['AcademicYear', 'academicyear', 'Year', 'year', 'testyear']

        const allGrades = new Set<string>()
        const allYears = new Set<string>()

        // Query each table
        for (const { tableName, assessmentType } of tablesToQuery) {
            console.log(`[Form Options] Querying grades from ${tableName} (${assessmentType})...`)

            // Try to get grades - check which column exists first
            let gradeColumn: string | null = null
            for (const gradeCol of gradeColumns) {
                try {
                    // Check if column exists by trying a simple query
                    const testQuery = `SELECT \`${gradeCol}\` FROM \`${projectId}.${datasetId}.${tableName}\` LIMIT 1`
                    await client.query({ query: testQuery, location })
                    gradeColumn = gradeCol
                    console.log(`[Form Options]   Found grade column: ${gradeCol}`)
                    break
                } catch {
                    continue
                }
            }

            if (gradeColumn) {
                try {
                    const query = `
                        SELECT DISTINCT \`${gradeColumn}\` as grade
                        FROM \`${projectId}.${datasetId}.${tableName}\`
                        WHERE \`${gradeColumn}\` IS NOT NULL
                        LIMIT 100
                    `
                    console.log(`[Form Options]   Executing grade query for ${tableName}...`)
                    const [rows] = await client.query({ query, location })
                    console.log(`[Form Options]   Raw query returned ${rows.length} row(s)`)
                    const gradesFromTable = new Set<string>()

                    rows.forEach((row: { grade?: string | number }, index: number) => {
                        if (row.grade !== null && row.grade !== undefined) {
                            const gradeStr = String(row.grade).trim()
                            const originalValue = row.grade
                            console.log(
                                `[Form Options]   Row ${index}: raw value = ${originalValue} (type: ${typeof originalValue}), stringified = "${gradeStr}"`
                            )

                            if (gradeStr) {
                                // Normalize grade (K, 0, or number, -1 for pre-k)
                                if (gradeStr.toUpperCase() === 'K' || gradeStr === '0' || gradeStr === 'KINDERGARTEN') {
                                    allGrades.add('K')
                                    gradesFromTable.add('K')
                                    console.log(`[Form Options]   → Added grade: K (from "${gradeStr}")`)
                                } else {
                                    const numGrade = parseInt(gradeStr)
                                    console.log(
                                        `[Form Options]   → Parsed "${gradeStr}" as number: ${numGrade} (isNaN: ${isNaN(numGrade)}, >= -1: ${numGrade >= -1}, <= 12: ${numGrade <= 12})`
                                    )

                                    // Accept all valid grades from the data (Pre-K -1 to 12)
                                    // Don't filter by assessment type - just return what's in the data
                                    if (!isNaN(numGrade) && numGrade >= -1 && numGrade <= 12) {
                                        const gradeToAdd = numGrade.toString()
                                        allGrades.add(gradeToAdd)
                                        gradesFromTable.add(gradeToAdd)
                                        console.log(`[Form Options]   → Added grade: ${gradeToAdd}`)
                                    } else {
                                        console.log(
                                            `[Form Options]   → Skipped invalid grade: ${gradeStr} (parsed as ${numGrade}, isNaN: ${isNaN(numGrade)}, >= -1: ${numGrade >= -1}, <= 12: ${numGrade <= 12})`
                                        )
                                    }
                                }
                            } else {
                                console.log(`[Form Options]   → Skipped empty grade string from value: ${originalValue}`)
                            }
                        } else {
                            console.log(`[Form Options]   Row ${index}: grade is null or undefined`)
                        }
                    })

                    const sortedGrades = Array.from(gradesFromTable).sort((a, b) => {
                        if (a === '-1') return -1
                        if (b === '-1') return 1
                        if (a === 'K') return -1
                        if (b === 'K') return 1
                        return parseInt(a) - parseInt(b)
                    })
                    console.log(`[Form Options]   ✓ Found ${gradesFromTable.size} unique grade(s) from ${tableName}: [${sortedGrades.join(', ')}]`)
                } catch (err) {
                    console.warn(`[Form Options]   ✗ Could not fetch grades from ${tableName}:`, err)
                }
            } else {
                console.warn(`[Form Options]   ✗ No grade column found in ${tableName} (tried: ${gradeColumns.join(', ')})`)
            }

            // Try to get years - check which column exists first
            let yearColumn: string | null = null
            for (const yearCol of yearColumns) {
                try {
                    // Check if column exists by trying a simple query
                    const testQuery = `SELECT \`${yearCol}\` FROM \`${projectId}.${datasetId}.${tableName}\` LIMIT 1`
                    await client.query({ query: testQuery, location })
                    yearColumn = yearCol
                    break
                } catch {
                    continue
                }
            }

            if (yearColumn) {
                try {
                    const query = `
                        SELECT DISTINCT \`${yearColumn}\` as year
                        FROM \`${projectId}.${datasetId}.${tableName}\`
                        WHERE \`${yearColumn}\` IS NOT NULL
                        LIMIT 100
                    `
                    console.log(`[Form Options]   Executing year query for ${tableName}...`)
                    const [rows] = await client.query({ query, location })
                    const yearsFromTable = new Set<string>()

                    rows.forEach((row: { year?: string | number }) => {
                        if (row.year !== null && row.year !== undefined) {
                            const yearStr = String(row.year).trim()
                            // Extract 4-digit year
                            const yearMatch = yearStr.match(/\d{4}/)
                            if (yearMatch) {
                                const year = parseInt(yearMatch[0])
                                if (!isNaN(year) && year >= 2000 && year <= 2100) {
                                    allYears.add(year.toString())
                                    yearsFromTable.add(year.toString())
                                }
                            } else {
                                // Try parsing as number
                                const year = parseInt(yearStr)
                                if (!isNaN(year) && year >= 2000 && year <= 2100) {
                                    allYears.add(year.toString())
                                    yearsFromTable.add(year.toString())
                                }
                            }
                        }
                    })

                    const sortedYears = Array.from(yearsFromTable).sort((a, b) => parseInt(a) - parseInt(b))
                    console.log(`[Form Options]   ✓ Found ${yearsFromTable.size} unique year(s) from ${tableName}: [${sortedYears.join(', ')}]`)
                } catch (err) {
                    console.warn(`[Form Options]   ✗ Could not fetch years from ${tableName}:`, err)
                }
            } else {
                console.warn(`[Form Options]   ✗ No year column found in ${tableName} (tried: ${yearColumns.join(', ')})`)
            }
        }

        // Sort and format results
        const grades = Array.from(allGrades).sort((a, b) => {
            // Handle -1 (pre-k) first
            if (a === '-1') return -1
            if (b === '-1') return 1
            // Then K (kindergarten)
            if (a === 'K') return -1
            if (b === 'K') return 1
            return parseInt(a) - parseInt(b)
        })

        const years = Array.from(allYears).sort((a, b) => parseInt(a) - parseInt(b))

        console.log(`[Form Options] Combined results:`)
        console.log(`[Form Options]   Total unique grades: ${grades.length} - [${grades.join(', ')}]`)
        console.log(`[Form Options]   Total unique years: ${years.length} - [${years.join(', ')}]`)

        // Fallback to defaults if nothing found
        // Return all possible grades as fallback (Pre-K to 12)
        if (grades.length === 0) {
            console.warn(`[Form Options] ⚠ No grades found, using fallback`)
            grades.push(...['-1', 'K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        }
        if (years.length === 0) {
            console.warn(`[Form Options] ⚠ No years found, using fallback`)
            const currentYear = new Date().getFullYear()
            years.push(...[currentYear, currentYear + 1, currentYear + 2, currentYear + 3].map((y) => y.toString()))
        }

        console.log(`[Form Options] ✓ Returning ${grades.length} grades and ${years.length} years`)

        return NextResponse.json({
            success: true,
            grades,
            years
        })
    } catch (error) {
        console.error('Error fetching form options from BigQuery', error)

        // Fallback to defaults
        const currentYear = new Date().getFullYear()
        const fallbackYears = [currentYear, currentYear + 1, currentYear + 2, currentYear + 3].map((y) => y.toString())

        // Fallback to all possible grades (Pre-K to 12)
        const fallbackGrades = ['-1', 'K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

        return NextResponse.json({
            success: true, // Return success with fallback data
            grades: fallbackGrades,
            years: fallbackYears
        })
    }
}
