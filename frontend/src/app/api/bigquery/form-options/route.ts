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
 *
 * Returns:
 *   - success: boolean
 *   - grades: string[] - Available grade options from data
 *   - years: string[] - Available year options from data
 */
export async function GET(req: NextRequest) {
    try {
        const projectId = req.nextUrl.searchParams.get('projectId')
        const datasetId = req.nextUrl.searchParams.get('datasetId')
        const location = req.nextUrl.searchParams.get('location') || 'US'

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
                console.log(`[Form Options] Using service account credentials: ${serviceAccountPath}`)
            }
        } catch (_credError) {
            console.warn('[Form Options] Could not resolve credentials, will try Application Default Credentials')
        }

        const client = new BigQuery({
            projectId: projectId,
            location: location
        })

        const dataset = client.dataset(datasetId)

        // Find available assessment tables
        const assessmentTablePatterns = [
            'nwea_production_calpads_v4_2',
            'Nwea_production_calpads_v4_2',
            'iready_production_calpads_v4_2',
            'iReady_production_calpads_v4_2',
            'renaissance_production_calpads_v4_2',
            'Renaissance_production_calpads_v4_2',
            'cers_production',
            'cers'
        ]

        let foundTable: string | null = null
        for (const tableName of assessmentTablePatterns) {
            try {
                const table = dataset.table(tableName)
                const [exists] = await table.exists()
                if (exists) {
                    foundTable = tableName
                    console.log(`[Form Options] Found table: ${foundTable}`)
                    break
                }
            } catch {
                continue
            }
        }

        if (!foundTable) {
            // Fallback to defaults if no table found
            const currentYear = new Date().getFullYear()
            const fallbackYears = [currentYear, currentYear + 1, currentYear + 2, currentYear + 3].map((y) => y.toString())
            return NextResponse.json({
                success: true,
                grades: ['K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                years: fallbackYears
            })
        }

        // Query for distinct grades and years
        // Try different column name patterns - prioritize Student_Grade and AcademicYear as requested
        const gradeColumns = ['Student_Grade', 'student_grade', 'StudentGrade', 'Grade', 'grade']
        const yearColumns = ['AcademicYear', 'academicyear', 'Year', 'year', 'testyear']

        const allGrades = new Set<string>()
        const allYears = new Set<string>()

        // Try to get grades - check which column exists first
        let gradeColumn: string | null = null
        for (const gradeCol of gradeColumns) {
            try {
                // Check if column exists by trying a simple query
                const testQuery = `SELECT \`${gradeCol}\` FROM \`${projectId}.${datasetId}.${foundTable}\` LIMIT 1`
                await client.query({ query: testQuery, location })
                gradeColumn = gradeCol
                break
            } catch {
                continue
            }
        }

        if (gradeColumn) {
            try {
                const query = `
                    SELECT DISTINCT \`${gradeColumn}\` as grade
                    FROM \`${projectId}.${datasetId}.${foundTable}\`
                    WHERE \`${gradeColumn}\` IS NOT NULL
                    LIMIT 100
                `
                const [rows] = await client.query({ query, location })
                rows.forEach((row: { grade?: string | number }) => {
                    if (row.grade !== null && row.grade !== undefined) {
                        const gradeStr = String(row.grade).trim()
                        if (gradeStr) {
                            // Normalize grade (K, 0, or number)
                            if (gradeStr.toUpperCase() === 'K' || gradeStr === '0' || gradeStr === 'KINDERGARTEN') {
                                allGrades.add('K')
                            } else {
                                const numGrade = parseInt(gradeStr)
                                if (!isNaN(numGrade) && numGrade >= 0 && numGrade <= 12) {
                                    allGrades.add(numGrade.toString())
                                }
                            }
                        }
                    }
                })
            } catch {
                console.warn(`[Form Options] Could not fetch grades from ${gradeColumn}`)
            }
        }

        // Try to get years - check which column exists first
        let yearColumn: string | null = null
        for (const yearCol of yearColumns) {
            try {
                // Check if column exists by trying a simple query
                const testQuery = `SELECT \`${yearCol}\` FROM \`${projectId}.${datasetId}.${foundTable}\` LIMIT 1`
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
                    FROM \`${projectId}.${datasetId}.${foundTable}\`
                    WHERE \`${yearColumn}\` IS NOT NULL
                    LIMIT 100
                `
                const [rows] = await client.query({ query, location })
                rows.forEach((row: { year?: string | number }) => {
                    if (row.year !== null && row.year !== undefined) {
                        const yearStr = String(row.year).trim()
                        // Extract 4-digit year
                        const yearMatch = yearStr.match(/\d{4}/)
                        if (yearMatch) {
                            const year = parseInt(yearMatch[0])
                            if (!isNaN(year) && year >= 2000 && year <= 2100) {
                                allYears.add(year.toString())
                            }
                        } else {
                            // Try parsing as number
                            const year = parseInt(yearStr)
                            if (!isNaN(year) && year >= 2000 && year <= 2100) {
                                allYears.add(year.toString())
                            }
                        }
                    }
                })
            } catch {
                console.warn(`[Form Options] Could not fetch years from ${yearColumn}`)
            }
        }

        // Sort and format results
        const grades = Array.from(allGrades).sort((a, b) => {
            if (a === 'K') return -1
            if (b === 'K') return 1
            return parseInt(a) - parseInt(b)
        })

        const years = Array.from(allYears).sort((a, b) => parseInt(a) - parseInt(b))

        // Fallback to defaults if nothing found
        if (grades.length === 0) {
            grades.push(...['K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        }
        if (years.length === 0) {
            const currentYear = new Date().getFullYear()
            years.push(...[currentYear, currentYear + 1, currentYear + 2, currentYear + 3].map((y) => y.toString()))
        }

        console.log(`[Form Options] Found ${grades.length} grades and ${years.length} years from ${foundTable}`)

        return NextResponse.json({
            success: true,
            grades,
            years
        })
    } catch {
        console.error('Error fetching form options from BigQuery')

        // Fallback to defaults
        const currentYear = new Date().getFullYear()
        const fallbackYears = [currentYear, currentYear + 1, currentYear + 2, currentYear + 3].map((y) => y.toString())

        return NextResponse.json({
            success: true, // Return success with fallback data
            grades: ['K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            years: fallbackYears
        })
    }
}
