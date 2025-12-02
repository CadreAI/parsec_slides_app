import { generateCharts } from '@/lib/chartGenerator'
import { loadConfig } from '@/lib/configLoader'
import { ingestAll } from '@/lib/dataIngestion'
import type { PartnerConfig } from '@/types/config'
import { NextRequest, NextResponse } from 'next/server'
import path from 'path'

/**
 * POST /api/data/ingest
 *
 * Ingests data from BigQuery based on partner configuration.
 *
 * Query params:
 *   - config?: string - Optional path to settings.yaml (defaults to auto-discovery)
 *
 * Returns:
 *   - success: boolean
 *   - data: Record<string, any[]> - Ingested data by source key
 *   - summary: Record<string, { rows: number; columns: number }>
 */
export async function POST(req: NextRequest) {
    try {
        const body = await req.json().catch(() => ({}))

        // Check if config object is provided directly (from UI)
        let cfg: PartnerConfig
        if (body.config && typeof body.config === 'object') {
            // Use config directly from request body
            cfg = body.config as PartnerConfig
        } else {
            // Fall back to loading from YAML file
            const configPath = body.config || req.nextUrl.searchParams.get('config') || undefined
            cfg = loadConfig(configPath)
        }

        // Ingest all data sources
        const data = await ingestAll(cfg)

        // Create summary
        const summary: Record<string, { rows: number; columns: number }> = {}
        for (const [key, rows] of Object.entries(data)) {
            summary[key] = {
                rows: rows.length,
                columns: rows.length > 0 ? Object.keys(rows[0]).length : 0
            }
        }

        // Generate charts after data ingestion
        let charts: string[] = []
        try {
            const dataDir = cfg.paths?.data_dir || './data'
            const resolvedDataDir = path.resolve(process.cwd(), dataDir)

            charts = await generateCharts({
                partnerName: cfg.partner_name,
                dataDir: resolvedDataDir,
                config: cfg,
                devMode: false // Always false to prevent charts from opening/previewing
            })
            console.log(`Generated ${charts.length} charts`)
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error'
            console.warn('Chart generation failed (non-critical):', errorMessage)
            // Don't fail the entire request if chart generation fails
        }

        // Don't return full data in response - it's too large (200k+ rows causes JSON.stringify RangeError)
        // Data is already saved to CSV files, so we just return summary
        return NextResponse.json({
            success: true,
            partner: cfg.partner_name,
            summary,
            charts: charts.length > 0 ? charts : undefined,
            message: 'Data ingested successfully. Data saved to CSV files.'
        })
    } catch (error: unknown) {
        console.error('Data ingestion error:', error)
        const errorMessage = error instanceof Error ? error.message : 'Failed to ingest data'
        const errorStack = error instanceof Error ? error.stack : undefined
        return NextResponse.json(
            {
                success: false,
                error: errorMessage,
                details: process.env.NODE_ENV === 'development' ? errorStack : undefined
            },
            { status: 500 }
        )
    }
}

/**
 * GET /api/data/ingest
 *
 * Same as POST but accepts config path as query parameter
 */
export async function GET(req: NextRequest) {
    try {
        const configPath = req.nextUrl.searchParams.get('config') || undefined

        const cfg = loadConfig(configPath)
        const data = await ingestAll(cfg)

        const summary: Record<string, { rows: number; columns: number }> = {}
        for (const [key, rows] of Object.entries(data)) {
            summary[key] = {
                rows: rows.length,
                columns: rows.length > 0 ? Object.keys(rows[0]).length : 0
            }
        }

        // Generate charts after data ingestion
        let charts: string[] = []
        try {
            const dataDir = cfg.paths?.data_dir || './data'
            const resolvedDataDir = path.resolve(process.cwd(), dataDir)

            charts = await generateCharts({
                partnerName: cfg.partner_name,
                dataDir: resolvedDataDir,
                config: cfg,
                devMode: false // Always false to prevent charts from opening/previewing
            })
            console.log(`Generated ${charts.length} charts`)
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error'
            console.warn('Chart generation failed (non-critical):', errorMessage)
            // Don't fail the entire request if chart generation fails
        }

        // Don't return full data in response - it's too large (200k+ rows causes JSON.stringify RangeError)
        // Data is already saved to CSV files, so we just return summary
        return NextResponse.json({
            success: true,
            partner: cfg.partner_name,
            summary,
            charts: charts.length > 0 ? charts : undefined,
            message: 'Data ingested successfully. Data saved to CSV files.'
        })
    } catch (error: unknown) {
        console.error('Data ingestion error:', error)
        const errorMessage = error instanceof Error ? error.message : 'Failed to ingest data'
        const errorStack = error instanceof Error ? error.stack : undefined
        return NextResponse.json(
            {
                success: false,
                error: errorMessage,
                details: process.env.NODE_ENV === 'development' ? errorStack : undefined
            },
            { status: 500 }
        )
    }
}
