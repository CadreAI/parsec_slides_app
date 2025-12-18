import { getBigQueryAuthMode } from '@/lib/bigquery'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/bigquery/auth-status
 *
 * Safe debug endpoint to confirm which BigQuery auth mode the frontend server is using.
 * Does NOT return secrets.
 *
 * Returns:
 *   - success: boolean
 *   - auth_mode: 'env_json' | 'file' | 'existing_env_path' | 'adc'
 *   - has_env_json: boolean
 */
export async function GET(_req: NextRequest) {
    const hasEnvJson = !!process.env.GOOGLE_SERVICE_ACCOUNT_JSON
    const mode = getBigQueryAuthMode()

    return NextResponse.json({
        success: true,
        auth_mode: mode,
        has_env_json: hasEnvJson
    })
}
