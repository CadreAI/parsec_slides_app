import { loadConfig } from '@/lib/configLoader'
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

        // Call backend Flask API for data ingestion and chart generation
        const backendUrl = process.env.BACKEND_URL || 'http://localhost:5000'
        const outputDir = cfg.paths?.charts_dir || cfg.paths?.output_dir || './charts'
        const resolvedOutputDir = path.resolve(process.cwd(), outputDir)

        // Extract chart filters from config
        const chartFilters = cfg.chart_filters || {}

        console.log(`[Frontend] Calling backend API: ${backendUrl}/ingest-and-generate`)
        console.log(`[Frontend] Partner: ${cfg.partner_name}`)
        console.log(`[Frontend] Output dir: ${resolvedOutputDir}`)

        const backendResponse = await fetch(`${backendUrl}/ingest-and-generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                partnerName: cfg.partner_name,
                outputDir: resolvedOutputDir,
                config: cfg,
                chartFilters: chartFilters
            })
        })

        if (!backendResponse.ok) {
            const errorData = await backendResponse.json().catch(() => ({}))
            throw new Error(errorData.error || `Backend API error: ${backendResponse.status}`)
        }

        const backendData = await backendResponse.json()

        return NextResponse.json({
            success: true,
            partner: cfg.partner_name,
            summary: backendData.summary || {},
            charts: backendData.charts || [],
            message: 'Data ingested and charts generated successfully via backend.'
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
    // GET endpoint redirects to POST with config from query params
    try {
        const configPath = req.nextUrl.searchParams.get('config') || undefined
        const cfg = loadConfig(configPath)

        // Convert GET to POST by calling POST handler logic
        const mockRequest = {
            json: async () => ({ config: cfg })
        } as NextRequest

        return POST(mockRequest)
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
