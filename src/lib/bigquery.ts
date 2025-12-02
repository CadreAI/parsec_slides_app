import { resolveServiceAccountCredentialsPath } from '@/lib/credentials'
import type { PartnerConfig } from '@/types/config'
import { BigQuery } from '@google-cloud/bigquery'

let bqClient: BigQuery | null = null

/**
 * Get or create a BigQuery client instance
 */
export function getBigQueryClient(cfg: PartnerConfig): BigQuery {
    if (!bqClient) {
        process.env.GOOGLE_CLOUD_PROJECT = cfg.gcp.project_id

        // Try to use explicit service account credentials if available
        try {
            const serviceAccountPath = resolveServiceAccountCredentialsPath()
            // BigQuery client will automatically use GOOGLE_APPLICATION_CREDENTIALS
            // if set, otherwise it uses Application Default Credentials
            if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
                process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceAccountPath
            }
        } catch {
            // If service account credentials not found, BigQuery will use
            // Application Default Credentials (ADC) or GOOGLE_APPLICATION_CREDENTIALS
            // if already set
            console.log('[BigQuery] Using Application Default Credentials or GOOGLE_APPLICATION_CREDENTIALS')
        }

        bqClient = new BigQuery({
            projectId: cfg.gcp.project_id,
            location: cfg.gcp.location
        })
    }
    return bqClient
}

/**
 * List all datasets in a BigQuery project
 */
export async function listDatasets(projectId: string, location?: string): Promise<string[]> {
    try {
        // Create a temporary client just for listing datasets
        // We need to set up credentials first
        process.env.GOOGLE_CLOUD_PROJECT = projectId

        try {
            const serviceAccountPath = resolveServiceAccountCredentialsPath()
            if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
                process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceAccountPath
            }
        } catch {
            // Credentials will use ADC if not found
        }

        const client = new BigQuery({
            projectId: projectId,
            location: location || 'US'
        })

        const [datasets] = await client.getDatasets()
        const datasetIds = datasets
            .map((dataset) => {
                // Handle both id and datasetId properties
                const id = (dataset as { id?: string; datasetId?: string }).id || (dataset as { id?: string; datasetId?: string }).datasetId || ''
                return id
            })
            .filter(Boolean)

        console.log(`Found ${datasetIds.length} datasets in project ${projectId}`)
        return datasetIds.sort()
    } catch (error: unknown) {
        const err = error as { code?: number | string; message?: string }
        console.error('Error listing datasets:', err.message || 'Unknown error')
        throw error
    }
}

/**
 * Check if a BigQuery table exists
 */
export async function tableExists(tableId: string, client: BigQuery): Promise<boolean> {
    try {
        // Parse tableId: project.dataset.table
        const parts = tableId.split('.')
        if (parts.length !== 3) {
            throw new Error(`Invalid table ID format: ${tableId}. Expected format: project.dataset.table`)
        }

        const [projectId, datasetId, tableName] = parts

        // Get the dataset and table
        const dataset = client.dataset(datasetId, { projectId })
        const table = dataset.table(tableName)

        // Check if table exists by getting metadata
        await table.get()
        return true
    } catch (error: unknown) {
        // Handle various error codes that indicate table doesn't exist
        const err = error as { code?: number | string; message?: string }
        if (err.code === 404 || err.code === 403 || err.code === 'ENOTFOUND' || err.message?.includes('Not found')) {
            console.log(`  [DEBUG] Table not found or permission denied: ${tableId}`)
            return false
        }
        throw error
    }
}

/**
 * Run a BigQuery SQL query and return results as an array of objects
 */
export async function runQuery(
    sql: string,
    client: BigQuery,
    params?: { districts?: string[]; years?: number[]; schools?: string[] }
): Promise<Record<string, unknown>[]> {
    console.log('Executing query...')

    const options: {
        query: string
        location?: string
        queryParameters?: Array<{
            name: string
            parameterType: { arrayType: { type: string } }
            parameterValue: { arrayValues: Array<{ value: string }> }
        }>
        useQueryCache?: boolean
        useLegacySql?: boolean
        priority?: 'INTERACTIVE' | 'BATCH'
        jobTimeoutMs?: number
        maximumBytesBilled?: string
    } = {
        query: sql,
        useQueryCache: true, // Use cached results if available (faster)
        useLegacySql: false, // Use standard SQL (faster)
        priority: 'INTERACTIVE', // Interactive priority for faster execution
        jobTimeoutMs: 300000 // 5 minute timeout
    }

    // Build query parameters for parameterized queries
    const queryParameters: Array<{
        name: string
        parameterType: { arrayType: { type: string } }
        parameterValue: { arrayValues: Array<{ value: string }> }
    }> = []

    if (params?.districts && params.districts.length > 0) {
        queryParameters.push({
            name: 'districts',
            parameterType: { arrayType: { type: 'STRING' } },
            parameterValue: { arrayValues: params.districts.map((d) => ({ value: d })) }
        })
    }

    if (params?.schools && params.schools.length > 0) {
        queryParameters.push({
            name: 'schools',
            parameterType: { arrayType: { type: 'STRING' } },
            parameterValue: { arrayValues: params.schools.map((s) => ({ value: s })) }
        })
    }

    if (queryParameters.length > 0) {
        options.queryParameters = queryParameters
    }

    const startTime = Date.now()
    const [job] = await client.createQueryJob(options)
    console.log(`Query job ID: ${job.id}`)

    // Wait for job completion - getQueryResults() will automatically wait
    const [rows] = await job.getQueryResults({
        timeoutMs: 300000 // 5 minute timeout
    })

    // Get final job metadata for statistics
    const [metadata] = await job.getMetadata()
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)
    const bytesProcessed = metadata.statistics?.query?.totalBytesProcessed
    const mbProcessed = bytesProcessed ? (parseInt(bytesProcessed.toString()) / 1024 / 1024).toFixed(2) : 'unknown'

    console.log(`✓ Query completed in ${elapsed}s (${mbProcessed} MB processed), fetched ${rows.length.toLocaleString()} rows`)

    return rows
}
