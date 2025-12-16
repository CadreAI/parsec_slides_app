import { auth } from '@clerk/nextjs/server'
import { BigQuery } from '@google-cloud/bigquery'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/bigquery/student-group-options
 *
 * Fetches available student group options by checking which columns exist in selected assessment tables
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
 *   - student_group_options: string[] - Array of available student group options (always includes "All Students")
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

        // Set up BigQuery client
        process.env.GOOGLE_CLOUD_PROJECT = projectId

        // Try to resolve service account credentials
        try {
            const { resolveServiceAccountCredentialsPath } = await import('@/lib/credentials')
            const serviceAccountPath = resolveServiceAccountCredentialsPath()
            if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
                process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceAccountPath
            }
        } catch (_credError) {
            // Credentials will use ADC if not found
        }

        const client = new BigQuery({
            projectId: projectId,
            location: location
        })

        const dataset = client.dataset(datasetId)

        // Define student group column mappings
        // UI Label -> Normalized column name (lowercase, no underscores/spaces)
        const STUDENT_GROUP_COLUMNS: Record<string, string> = {
            'English Learners': 'englishlearner',
            'Students with Disabilities': 'studentswithdisabilities',
            'Socioeconomically Disadvantaged': 'socioeconomicallydisadvantaged',
            'Title II Eligible Immigrants': 'titleiieligibleimmigrants',
            'Title I Part C Migrant': 'titleipartcmigrant',
            'ELAS Designation': 'elasdesignation',
            'Foster': 'foster',
            'Homeless': 'homeless'
        }

        // Normalization function: lowercase and strip underscores/spaces
        function normalizeColumnName(name: string): string {
            return name.toLowerCase().replace(/[_\s]/g, '')
        }

        // Map assessment IDs to table name patterns
        const assessmentTableMap: Record<string, string[]> = {
            nwea: ['nwea_production_calpads_v4_2', 'Nwea_production_calpads_v4_2', 'nwea_production', 'nwea'],
            iready: ['iready_production_calpads_v4_2', 'iReady_production_calpads_v4_2', 'iready_production', 'iready'],
            star: ['renaissance_production_calpads_v4_2', 'Renaissance_production_calpads_v4_2', 'star_production', 'star', 'renaissance'],
            cers: ['cers_production', 'cers', 'CERS']
        }

        const requestedAssessments = assessments ? assessments.split(',').map((a) => a.trim()) : []
        const availableStudentGroups = new Set<string>()

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
                console.warn(`[Student Group Options] Table not found for ${assessmentId}`)
                continue
            }

            // Get table schema to find student group columns
            try {
                const [metadata] = await dataset.table(foundTable).getMetadata()
                const fields = metadata.schema?.fields || []

                // Normalize all column names in the table
                const normalizedTableColumns = fields.map((field: { name?: string }) => 
                    field.name ? normalizeColumnName(field.name) : ''
                ).filter(Boolean)

                console.log(`[Student Group Options] ${assessmentId} normalized columns:`, normalizedTableColumns.slice(0, 10))

                // Check which expected student group columns exist
                for (const [uiLabel, normalizedColumn] of Object.entries(STUDENT_GROUP_COLUMNS)) {
                    if (normalizedTableColumns.includes(normalizedColumn)) {
                        availableStudentGroups.add(uiLabel)
                        console.log(`[Student Group Options] Found column for "${uiLabel}" in ${assessmentId}`)
                    }
                }
            } catch (error) {
                console.error(`[Student Group Options] Error querying ${assessmentId} table:`, error)
                continue
            }
        }

        // Always prepend "All Students" (it's a special case that doesn't require a column)
        const studentGroupOptions = ['All Students', ...Array.from(availableStudentGroups).sort()]

        console.log(`[Student Group Options] Returning ${studentGroupOptions.length} options:`, studentGroupOptions)

        // If only "All Students" is available, return it (UI will decide whether to hide the field)
        return NextResponse.json({
            success: true,
            student_group_options: studentGroupOptions
        })
    } catch (error: unknown) {
        console.error('Error fetching student group options from BigQuery:', error)

        // Fallback to defaults
        return NextResponse.json({
            success: true,
            student_group_options: [
                'All Students',
                'English Learners',
                'Students with Disabilities',
                'Socioeconomically Disadvantaged',
                'Foster',
                'Homeless'
            ]
        })
    }
}

