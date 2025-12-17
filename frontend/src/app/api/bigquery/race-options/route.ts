import { auth } from '@clerk/nextjs/server'
import { NextRequest, NextResponse } from 'next/server'
import { createBigQueryClient } from '@/lib/bigquery'

/**
 * GET /api/bigquery/race-options
 *
 * Fetches available race/ethnicity options from actual data tables
 *
 * Query params:
 *   - projectId: string (required) - GCP project ID
 *   - datasetId: string (required) - BigQuery dataset ID
 *   - assessments: string (required) - Comma-separated list of assessment IDs (e.g., "nwea,iready")
 *   - location?: string (optional) - BigQuery location (default: US)
 *   - tablePaths?: string (optional) - Comma-separated list of specific table paths
 *
 * Returns:
 *   - success: boolean
 *   - race_options: string[] - Array of unique race/ethnicity values from the tables
 */
export async function GET(req: NextRequest) {
    try {
        const { userId } = await auth()
        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        const projectId = req.nextUrl.searchParams.get('projectId')
        const datasetId = req.nextUrl.searchParams.get('datasetId')
        const assessments = req.nextUrl.searchParams.get('assessments') || ''
        const location = req.nextUrl.searchParams.get('location') || 'US'
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

        const client = createBigQueryClient(projectId, location)

        const dataset = client.dataset(datasetId)

        // Map assessment IDs to table name patterns
        const assessmentTableMap: Record<string, string[]> = {
            nwea: ['nwea_production_calpads_v4_2', 'Nwea_production_calpads_v4_2', 'nwea_production', 'nwea'],
            iready: ['iready_production_calpads_v4_2', 'iReady_production_calpads_v4_2', 'iready_production', 'iready'],
            star: ['renaissance_production_calpads_v4_2', 'Renaissance_production_calpads_v4_2', 'star_production', 'star', 'renaissance'],
            cers: ['cers_production', 'cers', 'CERS']
        }

        const requestedAssessments = assessments ? assessments.split(',').map((a) => a.trim()) : []
        const allRaceValues = new Set<string>()

        // Query each requested assessment
        for (const assessmentId of requestedAssessments) {
            let tablePatterns = assessmentTableMap[assessmentId] || []

            // If specific table paths are provided, filter to relevant tables for this assessment
            if (specificTablePaths) {
                const relevantTables = specificTablePaths.filter((path) => {
                    const tableName = path.split('.').pop() || path
                    return tableName.toLowerCase().includes(assessmentId.toLowerCase())
                })
                if (relevantTables.length > 0) {
                    tablePatterns = relevantTables.map((path) => path.split('.').pop() || path)
                }
            }

            let foundTable: string | null = null

            // Find the table for this assessment
            for (const tableName of tablePatterns) {
                try {
                    const table = dataset.table(tableName)
                    const [exists] = await table.exists()
                    if (exists) {
                        foundTable = tableName
                        break
                    }
                } catch {
                    continue
                }
            }

            if (!foundTable) {
                console.warn(`[Race Options] Table not found for ${assessmentId}`)
                continue
            }

            // Get table schema to find ethnicity column
            try {
                const [metadata] = await dataset.table(foundTable).getMetadata()
                const fields = metadata.schema?.fields || []

                // Try different column name patterns (case-insensitive)
                // Check for: EthnicityRace, Ethnicity_Race, ethnicityrace, Race, race, Ethnicity, ethnicity
                const ethnicityColumnPatterns = ['ethnicityrace', 'ethnicity_race', 'race', 'ethnicity']

                let ethnicityColumn: string | null = null

                for (const pattern of ethnicityColumnPatterns) {
                    const matchingField = fields.find((field: { name?: string }) => field.name?.toLowerCase().replace(/_/g, '') === pattern.replace(/_/g, ''))
                    if (matchingField && matchingField.name) {
                        ethnicityColumn = matchingField.name
                        break
                    }
                }

                if (!ethnicityColumn) {
                    console.warn(`[Race Options] No ethnicity column found in ${assessmentId} table`)
                    continue
                }

                // Query for distinct race values
                const query = `
                    SELECT DISTINCT \`${ethnicityColumn}\` as race
                    FROM \`${projectId}.${datasetId}.${foundTable}\`
                    WHERE \`${ethnicityColumn}\` IS NOT NULL
                      AND \`${ethnicityColumn}\` != ''
                    LIMIT 1000
                `

                console.log(`[Race Options] Querying ${assessmentId} table for race values using column: ${ethnicityColumn}`)

                const [rows] = await client.query({ query, location })

                // Add all race values to the set
                rows.forEach((row: { race?: string }) => {
                    if (row.race) {
                        const raceValue = String(row.race).trim()
                        if (raceValue) {
                            allRaceValues.add(raceValue)
                        }
                    }
                })

                console.log(`[Race Options] Found ${rows.length} distinct race values in ${assessmentId}`)
            } catch (error) {
                console.error(`[Race Options] Error querying ${assessmentId} table:`, error)
                continue
            }
        }

        // Convert set to sorted array
        const raceOptions = Array.from(allRaceValues).sort()

        // If no values found, return default fallback
        if (raceOptions.length === 0) {
            console.log('[Race Options] No race values found, returning default list')
            return NextResponse.json({
                success: true,
                race_options: [
                    'Hispanic or Latino',
                    'White',
                    'Black or African American',
                    'Asian',
                    'Filipino',
                    'American Indian or Alaska Native',
                    'Native Hawaiian or Pacific Islander',
                    'Two or More Races',
                    'Not Stated'
                ]
            })
        }

        console.log(`[Race Options] Returning ${raceOptions.length} unique race values`)

        return NextResponse.json({
            success: true,
            race_options: raceOptions
        })
    } catch (error: unknown) {
        console.error('Error fetching race options from BigQuery:', error)

        // Fallback to defaults
        return NextResponse.json({
            success: true,
            race_options: [
                'Hispanic or Latino',
                'White',
                'Black or African American',
                'Asian',
                'Filipino',
                'American Indian or Alaska Native',
                'Native Hawaiian or Pacific Islander',
                'Two or More Races',
                'Not Stated'
            ]
        })
    }
}
