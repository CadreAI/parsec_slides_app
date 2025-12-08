import { BigQuery } from '@google-cloud/bigquery'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/bigquery/assessment-tables
 *
 * Checks which assessment tables exist in a dataset
 *
 * Query params:
 *   - projectId: string (required) - GCP project ID
 *   - datasetId: string (required) - BigQuery dataset ID
 *   - location?: string (optional) - BigQuery location (default: US)
 *
 * Returns:
 *   - success: boolean
 *   - available_assessments: string[] - Array of assessment IDs that have tables (e.g., ['nwea', 'iready'])
 *   - tables: Record<string, string> - Map of assessment ID to full table ID
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
                console.log(`[Assessment Tables] Using service account credentials: ${serviceAccountPath}`)
            }
        } catch (credError) {
            console.warn('[Assessment Tables] Could not resolve credentials, will try Application Default Credentials')
        }

        const client = new BigQuery({
            projectId: projectId,
            location: location
        })

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

        // Check each assessment for table existence
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
                } catch (error) {
                    // Continue to next table name
                    continue
                }
            }
        }

        console.log(`[Assessment Tables] Available assessments in ${datasetId}: ${availableAssessments.join(', ')}`)

        return NextResponse.json({
            success: true,
            available_assessments: availableAssessments,
            tables: tables
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
