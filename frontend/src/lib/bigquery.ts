import { resolveServiceAccountCredentialsPath } from '@/lib/credentials'
import type { PartnerConfig } from '@/types/config'
import { BigQuery } from '@google-cloud/bigquery'

let bqClient: BigQuery | null = null

function getServiceAccountCredentialsFromEnv(): {
    client_email: string
    private_key: string
} | null {
    const raw = process.env.GOOGLE_SERVICE_ACCOUNT_JSON
    if (!raw) return null

    const trimmed = raw.trim()

    function parseJson(s: string) {
        const obj = JSON.parse(s) as { client_email?: string; private_key?: string }
        if (!obj?.client_email || !obj?.private_key) return null
        return {
            client_email: obj.client_email,
            // Vercel env vars often store newlines escaped
            private_key: obj.private_key.replace(/\\n/g, '\n')
        }
    }

    // Support either raw JSON or base64-encoded JSON
    try {
        if (trimmed.startsWith('{')) return parseJson(trimmed)
        const decoded = Buffer.from(trimmed, 'base64').toString('utf8')
        return parseJson(decoded)
    } catch {
        return null
    }
}

/**
 * Create a BigQuery client for API routes that only have projectId/location.
 * Prefers GOOGLE_SERVICE_ACCOUNT_JSON when present (no filesystem dependency).
 */
export function createBigQueryClient(projectId: string, location: string = 'US'): BigQuery {
    process.env.GOOGLE_CLOUD_PROJECT = projectId

    const envCreds = getServiceAccountCredentialsFromEnv()
    if (envCreds) {
        return new BigQuery({ projectId, location, credentials: envCreds })
    }

    // Fallback to file-based/ADC credentials for local dev
    try {
        const serviceAccountPath = resolveServiceAccountCredentialsPath()
        if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
            process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceAccountPath
        }
    } catch {
        // ADC or pre-set GOOGLE_APPLICATION_CREDENTIALS
    }

    return new BigQuery({ projectId, location })
}

/**
 * Get or create a BigQuery client instance
 */
export function getBigQueryClient(cfg: PartnerConfig): BigQuery {
    if (!bqClient) {
        bqClient = createBigQueryClient(cfg.gcp.project_id, cfg.gcp.location || 'US')
    }
    return bqClient
}

/**
 * List all datasets in a BigQuery project
 */
export async function listDatasets(projectId: string, location?: string): Promise<string[]> {
    try {
        const client = createBigQueryClient(projectId, location || 'US')

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
        const errorMessage = err.message || 'Unknown error'

        // Provide helpful error message for credential issues
        if (errorMessage.includes('Could not load the default credentials') || errorMessage.includes('credentials')) {
            const helpfulMessage =
                `BigQuery authentication failed. Please set up credentials using one of these methods:\n` +
                `1. Set GOOGLE_SERVICE_ACCOUNT_JSON environment variable to the full service account JSON (recommended for Vercel)\n` +
                `2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable to your service account JSON file path\n` +
                `3. Set GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH environment variable\n` +
                `4. Place service_account.json in google/service_account.json\n` +
                `5. Configure Application Default Credentials (gcloud auth application-default login)\n\n` +
                `Original error: ${errorMessage}`

            console.error('Error listing datasets:', helpfulMessage)
            throw new Error(helpfulMessage)
        }

        console.error('Error listing datasets:', errorMessage)
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
