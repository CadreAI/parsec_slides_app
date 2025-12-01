import type { PartnerConfig } from '@/types/config'
import { parse } from 'csv-parse/sync'
import { stringify } from 'csv-stringify/sync'
import fs from 'fs'
import path from 'path'
import { getBigQueryClient, runQuery, tableExists } from './bigquery'
import { resolveSource } from './configLoader'
import { sqlCalpads, sqlCers, sqlIab, sqlIready, sqlNwea, sqlStar } from './sqlBuilders'

type SqlBuilder = (tableId: string, excludeCols?: string[]) => string

/**
 * Get data directory path, creating it if needed
 */
function getDataDir(cfg: PartnerConfig): string {
    const dataDir = cfg.paths?.data_dir || './data'
    const resolvedPath = path.resolve(process.cwd(), dataDir)
    if (!fs.existsSync(resolvedPath)) {
        fs.mkdirSync(resolvedPath, { recursive: true })
    }
    return resolvedPath
}

/**
 * Clean column names (convert to lowercase, replace spaces with underscores)
 */
function cleanColumnNames(data: any[]): any[] {
    if (data.length === 0) return data

    // Build column name mapping once (more efficient for large datasets)
    const firstRow = data[0]
    const columnMap: Record<string, string> = {}

    for (const key of Object.keys(firstRow)) {
        columnMap[key] = key
            .toLowerCase()
            .replace(/\s+/g, '_')
            .replace(/[^a-z0-9_]/g, '')
    }

    // Apply mapping to all rows
    const cleaned = data.map((row) => {
        const newRow: any = {}
        for (const [key, value] of Object.entries(row)) {
            newRow[columnMap[key]] = value
        }
        return newRow
    })
    return cleaned
}

/**
 * Load CSV from cache or query BigQuery
 */
async function cacheOrQuery(csvPath: string, sql: string, client: any, params?: { districts?: string[] }, cfg?: PartnerConfig): Promise<any[]> {
    const useCache = cfg?.options?.cache_csv !== false

    if (useCache && fs.existsSync(csvPath)) {
        console.log(`Loading from cache: ${csvPath}`)
        const csvContent = fs.readFileSync(csvPath, 'utf-8')
        const records = parse(csvContent, {
            columns: true,
            skip_empty_lines: true
        })
        return cleanColumnNames(records)
    }

    const data = await runQuery(sql, client, params)
    console.log(`Processing ${data.length.toLocaleString()} rows...`)

    const cleaned = cleanColumnNames(data)
    console.log(`Cleaned column names for ${cleaned.length.toLocaleString()} rows`)

    // Write to cache
    if (cleaned.length > 0) {
        console.log(`Converting to CSV format...`)
        const csvContent = stringify(cleaned, { header: true })
        console.log(`Writing CSV to cache (${(csvContent.length / 1024 / 1024).toFixed(2)} MB)...`)
        fs.writeFileSync(csvPath, csvContent)
        console.log(`✓ Cached to: ${csvPath}`)
    }

    return cleaned
}

/**
 * Attempt to ingest data from a source if configured and table exists
 */
export async function ingestOptional(
    sourceKey: string,
    csvName: string,
    sqlBuilder: SqlBuilder,
    cfg: PartnerConfig,
    params?: { districts?: string[] }
): Promise<any[]> {
    const tableId = resolveSource(sourceKey, cfg)
    console.log(`[DEBUG] Resolved ${sourceKey} → ${tableId}`)

    if (!tableId) {
        console.log(`Skip ${sourceKey}: not configured in YAML`)
        return []
    }

    const client = getBigQueryClient(cfg)
    if (!(await tableExists(tableId, client))) {
        console.log(`Skip ${sourceKey}: table not found → ${tableId}`)
        return []
    }

    const excludeCols = cfg.exclude_cols?.[sourceKey] || []
    const sql = sqlBuilder(tableId, excludeCols)

    const dataDir = getDataDir(cfg)
    const csvPath = path.join(dataDir, csvName)

    try {
        let data = await cacheOrQuery(csvPath, sql, client, params, cfg)

        // Drop excluded columns if present
        if (excludeCols.length > 0 && data.length > 0) {
            const excludeSet = new Set(excludeCols.map((c) => c.toLowerCase().replace(/\s+/g, '_')))
            const firstRow = data[0]
            const columnsToDrop = Object.keys(firstRow).filter((col) => excludeSet.has(col))

            if (columnsToDrop.length > 0) {
                console.log(`Excluding columns for ${sourceKey}: ${columnsToDrop.join(', ')}`)
                data = data.map((row) => {
                    const newRow = { ...row }
                    for (const col of columnsToDrop) {
                        delete newRow[col]
                    }
                    return newRow
                })
            }
        }

        return data
    } catch (error: any) {
        console.log(`Skip ${sourceKey}: query error: ${error.message}`)
        return []
    }
}

/**
 * Main ingestion function - ingests all configured sources
 */
export async function ingestAll(cfg: PartnerConfig): Promise<Record<string, any[]>> {
    console.log(`Partner: ${cfg.partner_name}`)
    console.log(`Districts: ${cfg.district_name?.join(', ') || 'N/A'}`)

    const districts = cfg.district_name || []

    const results: Record<string, any[]> = {}

    // Ingest each source
    results.iab = await ingestOptional('iab', 'iab_data.csv', sqlIab, cfg)
    results.cers = await ingestOptional('cers', 'cers_data.csv', sqlCers, cfg, {
        districts
    })
    results.calpads = await ingestOptional('calpads', 'calpads_data.csv', sqlCalpads, cfg)
    results.nwea = await ingestOptional('nwea', 'nwea_data.csv', sqlNwea, cfg)
    results.star = await ingestOptional('star', 'star_data.csv', sqlStar, cfg)
    results.iready = await ingestOptional('iready', 'iready_data.csv', sqlIready, cfg)

    // Summary
    function summarize(name: string, data: any[]) {
        if (data.length === 0) {
            console.log(`${name}: skipped or empty\n`)
            return
        }

        let summaryText = `${name}: ${data.length.toLocaleString()} rows`

        // Try to find year column
        const yearCols = Object.keys(data[0] || {}).filter((c) => c.toLowerCase().includes('year'))
        if (yearCols.length > 0 && data.length > 0) {
            const yearCol = yearCols[0]
            const yearCounts: Record<string, number> = {}
            for (const row of data) {
                const year = String(row[yearCol] || '')
                if (year) {
                    yearCounts[year] = (yearCounts[year] || 0) + 1
                }
            }
            const sortedYears = Object.keys(yearCounts).sort()
            summaryText += ' | years:'
            for (const year of sortedYears) {
                summaryText += `\n  ${year}: ${yearCounts[year].toLocaleString()}`
            }
        }
        console.log(summaryText + '\n')
    }

    summarize('CERS', results.cers)
    summarize('IAB', results.iab)
    summarize('CALPADS', results.calpads)
    summarize('NWEA', results.nwea)
    summarize('STAR', results.star)
    summarize('i-Ready', results.iready)

    return results
}
