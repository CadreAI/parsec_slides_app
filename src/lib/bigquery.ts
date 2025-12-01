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
export async function runQuery(sql: string, client: BigQuery, params?: { districts?: string[] }): Promise<Record<string, unknown>[]> {
    console.log('Executing query...')

    const options: {
        query: string
        location?: string
        queryParameters?: Array<{
            name: string
            parameterType: { arrayType: { type: string } }
            parameterValue: { arrayValues: Array<{ value: string }> }
        }>
    } = {
        query: sql
    }

    if (params?.districts) {
        options.queryParameters = [
            {
                name: 'districts',
                parameterType: { arrayType: { type: 'STRING' } },
                parameterValue: { arrayValues: params.districts.map((d) => ({ value: d })) }
            }
        ]
    }

    const [job] = await client.createQueryJob(options)
    console.log(`Query job ID: ${job.id}`)
    console.log('Waiting for query to complete...')

    // Use streaming results for large datasets to avoid memory issues
    const [rows] = await job.getQueryResults()
    console.log(`✓ Query completed, fetched ${rows.length.toLocaleString()} rows`)

    return rows
}
