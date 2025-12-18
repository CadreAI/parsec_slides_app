import { auth } from '@clerk/nextjs/server'
import { NextRequest, NextResponse } from 'next/server'
import { createBigQueryClient } from '@/lib/bigquery'

/**
 * GET /api/bigquery/assessment-tables
 *
 * Checks which assessment tables exist in a dataset
 *
 * Query params:
 *   - projectId: string (required) - GCP project ID
 *   - datasetId: string (required) - BigQuery dataset ID
 *   - location?: string (optional) - BigQuery location (default: US)
 *   - includeVariants?: string (optional) - If 'true', returns all table variants (default: false)
 *
 * Returns:
 *   - success: boolean
 *   - available_assessments: string[] - Array of assessment IDs that have tables (e.g., ['nwea', 'iready'])
 *   - tables: Record<string, string> - Map of assessment ID to full table ID
 *   - variants?: Record<string, Array<{table_name: string, full_path: string, is_default: boolean}>> - All table variants (when includeVariants=true)
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
        const includeVariants = req.nextUrl.searchParams.get('includeVariants') === 'true'

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

        // Define table name patterns for each assessment
        const assessmentTablePatterns: Record<string, string[]> = {
            nwea: ['nwea_production_calpads_v4_2', 'Nwea_production_calpads_v4_2', 'nwea_production', 'nwea', 'NWEA'],
            iready: ['iready_production_calpads_v4_2', 'iReady_production_calpads_v4_2', 'iready_production', 'iready', 'iReady'],
            star: ['renaissance_production_calpads_v4_2', 'Renaissance_production_calpads_v4_2', 'star_production', 'star', 'STAR', 'renaissance'],
            cers: ['cers_production', 'cers', 'CERS', 'cers_iab', 'CERS_IAB']
        }

        const availableAssessments: string[] = []
        const tables: Record<string, string> = {}
        const variantsMap: Record<
            string,
            Array<{
                table_name: string
                full_path: string
                is_default: boolean
            }>
        > = {}

        if (includeVariants) {
            const [allTables] = await dataset.getTables()
            const allTableIds = allTables.map((t) => t.id || '')

            // Check each assessment for all matching variants
            for (const [assessmentId, tableNames] of Object.entries(assessmentTablePatterns)) {
                const baseTableName = tableNames[0]

                // filter the table ids that start with the base table name
                const matchingTables = allTableIds.filter((tableId) => tableId.toLowerCase().startsWith(baseTableName.toLowerCase()))

                if (matchingTables.length > 0) {
                    // Sort to ensure base table is first
                    matchingTables.sort((a, b) => {
                        if (a === baseTableName) return -1
                        if (b === baseTableName) return 1
                        return a.localeCompare(b)
                    })

                    const fullTableId = `${projectId}.${datasetId}.${matchingTables[0]}`
                    if (!availableAssessments.includes(assessmentId)) {
                        availableAssessments.push(assessmentId)
                    }
                    tables[assessmentId] = fullTableId

                    // Store all variants
                    variantsMap[assessmentId] = matchingTables.map((tableName) => ({
                        table_name: tableName,
                        full_path: `${projectId}.${datasetId}.${tableName}`,
                        is_default: tableName.toLowerCase() === baseTableName.toLowerCase()
                    }))

                    console.log(`[Assessment Tables] Found ${matchingTables.length} variant(s) for ${assessmentId}: ${matchingTables.join(', ')}`)
                }
            }
        } else {
            // Original behavior: find first matching table only
            for (const [assessmentId, tableNames] of Object.entries(assessmentTablePatterns)) {
                for (const tableName of tableNames) {
                    try {
                        const table = dataset.table(tableName)
                        const [exists] = await table.exists()
                        if (exists) {
                            const fullTableId = `${projectId}.${datasetId}.${tableName}`
                            if (!availableAssessments.includes(assessmentId)) {
                                availableAssessments.push(assessmentId)
                            }
                            tables[assessmentId] = fullTableId
                            console.log(`[Assessment Tables] Found ${assessmentId} table: ${fullTableId}`)
                            break // Found a table for this assessment, move to next
                        }
                    } catch {
                        // Continue to next table name
                        continue
                    }
                }
            }
        }

        console.log(`[Assessment Tables] Available assessments in ${datasetId}: ${availableAssessments.join(', ')}`)

        return NextResponse.json({
            success: true,
            available_assessments: availableAssessments,
            tables: tables,
            ...(includeVariants && { variants: variantsMap })
        })
    } catch (error: unknown) {
        console.error('Error checking assessment tables:', error)
        const errorMessage = error instanceof Error ? error.message : 'Failed to check assessment tables'
        return NextResponse.json(
            {
                success: false,
                error: errorMessage,
                available_assessments: [],
                tables: {}
            },
            { status: 500 }
        )
    }
}
