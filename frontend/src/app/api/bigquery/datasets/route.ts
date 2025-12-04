import { listDatasets } from '@/lib/bigquery'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/bigquery/datasets
 *
 * Lists all datasets in a BigQuery project
 *
 * Query params:
 *   - projectId: string (required) - GCP project ID
 *   - location?: string (optional) - BigQuery location (default: US)
 *
 * Returns:
 *   - success: boolean
 *   - datasets: string[] - Array of dataset IDs
 */
export async function GET(req: NextRequest) {
    try {
        const projectId = req.nextUrl.searchParams.get('projectId')
        const location = req.nextUrl.searchParams.get('location') || 'US'

        if (!projectId) {
            return NextResponse.json(
                {
                    success: false,
                    error: 'projectId query parameter is required'
                },
                { status: 400 }
            )
        }

        const datasets = await listDatasets(projectId, location)

        return NextResponse.json({
            success: true,
            datasets: datasets
        })
    } catch (error: unknown) {
        console.error('Error listing datasets:', error)
        const errorMessage = error instanceof Error ? error.message : 'Failed to list datasets'
        return NextResponse.json(
            {
                success: false,
                error: errorMessage
            },
            { status: 500 }
        )
    }
}
