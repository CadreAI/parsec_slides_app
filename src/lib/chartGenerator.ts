import type { PartnerConfig } from '@/types/config'
import { spawn } from 'child_process'
import fs from 'fs'
import path from 'path'

/**
 * Options for chart generation
 */
export interface ChartGenerationOptions {
    partnerName: string
    dataDir: string
    outputDir?: string
    config?: PartnerConfig
    devMode?: boolean
}

/**
 * Generate charts using Python scripts after data ingestion
 */
export async function generateCharts(options: ChartGenerationOptions): Promise<string[]> {
    const { partnerName, dataDir, outputDir, config, devMode = false } = options

    // Ensure output directory exists
    const chartsOutputDir = outputDir || path.join(process.cwd(), 'charts')
    if (!fs.existsSync(chartsOutputDir)) {
        fs.mkdirSync(chartsOutputDir, { recursive: true })
    }

    // Check if Python is available
    const pythonCommand = await findPythonCommand()
    if (!pythonCommand) {
        throw new Error('Python not found. Please install Python 3.8+ to generate charts.')
    }

    // Check if required Python files exist
    const pythonDir = path.join(process.cwd(), 'python')
    const helperFunctionsPath = path.join(pythonDir, 'helper_functions.py')

    if (!fs.existsSync(helperFunctionsPath)) {
        throw new Error(`Python helper functions not found at: ${helperFunctionsPath}`)
    }

    const generatedCharts: string[] = []

    // Check which data sources are available
    const availableSources = checkAvailableDataSources(dataDir)

    // Generate charts for each available source
    for (const source of availableSources) {
        try {
            const chartPaths = await generateChartForSource({
                pythonCommand,
                source,
                partnerName,
                dataDir,
                chartsOutputDir,
                config,
                devMode
            })
            if (chartPaths && chartPaths.length > 0) {
                generatedCharts.push(...chartPaths)
            }
        } catch (error: any) {
            console.warn(`Failed to generate chart for ${source}:`, error.message)
        }
    }

    return generatedCharts
}

/**
 * Generate chart for a specific data source
 */
async function generateChartForSource(options: {
    pythonCommand: string
    source: string
    partnerName: string
    dataDir: string
    chartsOutputDir: string
    config?: PartnerConfig
    devMode: boolean
}): Promise<string[]> {
    const { pythonCommand, source, partnerName, dataDir, chartsOutputDir, config, devMode } = options

    // For now, we'll create a simple Python script that can be extended
    // This is a placeholder - you'll need to create actual chart generation scripts
    const scriptPath = path.join(process.cwd(), 'python', `${source}_charts.py`)

    // If source-specific script doesn't exist, skip
    if (!fs.existsSync(scriptPath)) {
        console.log(`No chart script found for ${source}, skipping...`)
        return []
    }

    return new Promise((resolve, reject) => {
        // Always set dev-mode to false to prevent charts from opening/previewing
        const args = [scriptPath, '--partner', partnerName, '--data-dir', dataDir, '--output-dir', chartsOutputDir, '--dev-mode', 'false']

        // Add config as JSON if provided
        if (config) {
            args.push('--config', JSON.stringify(config))
        }

        const pythonProcess = spawn(pythonCommand, args, {
            cwd: process.cwd(),
            stdio: ['pipe', 'pipe', 'pipe']
        })

        let stdout = ''
        let stderr = ''

        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString()
            console.log(`[${source} chart] ${data.toString().trim()}`)
        })

        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString()
            console.error(`[${source} chart error] ${data.toString().trim()}`)
        })

        pythonProcess.on('close', async (code) => {
            if (code === 0) {
                // Primary method: Read from chart_index.csv (most reliable)
                const chartIndexPath = path.join(chartsOutputDir, 'chart_index.csv')
                let normalizedPaths: string[] = []

                if (fs.existsSync(chartIndexPath)) {
                    try {
                        // Read CSV file - parse it properly handling quoted fields
                        const csvContent = fs.readFileSync(chartIndexPath, 'utf-8')
                        const lines = csvContent.split('\n').filter((line) => line.trim())

                        if (lines.length > 1) {
                            // Parse header to find file_path column index
                            const header = lines[0].split(',').map((col) => col.trim().replace(/^"|"$/g, ''))
                            const filePathIndex = header.indexOf('file_path')

                            if (filePathIndex >= 0) {
                                // Parse each data row
                                for (let i = 1; i < lines.length; i++) {
                                    const line = lines[i]
                                    // Simple CSV parsing - handle quoted fields
                                    const columns: string[] = []
                                    let current = ''
                                    let inQuotes = false

                                    for (let j = 0; j < line.length; j++) {
                                        const char = line[j]
                                        if (char === '"') {
                                            inQuotes = !inQuotes
                                        } else if (char === ',' && !inQuotes) {
                                            columns.push(current.trim())
                                            current = ''
                                        } else {
                                            current += char
                                        }
                                    }
                                    columns.push(current.trim()) // Add last column

                                    if (columns[filePathIndex]) {
                                        const filePath = columns[filePathIndex].replace(/^"|"$/g, '')
                                        // Resolve path - could be absolute or relative
                                        const resolvedPath = path.isAbsolute(filePath) ? filePath : path.resolve(chartsOutputDir, filePath)

                                        if (fs.existsSync(resolvedPath)) {
                                            normalizedPaths.push(resolvedPath)
                                        }
                                    }
                                }
                                console.log(`[${source} chart] Found ${normalizedPaths.length} chart(s) from chart_index.csv`)
                            } else {
                                console.warn(`[${source} chart] file_path column not found in chart_index.csv`)
                            }
                        }
                    } catch (error: any) {
                        console.warn(`[${source} chart] Failed to read chart_index.csv:`, error.message)
                    }
                }

                // Fallback: Extract from stdout if CSV didn't work or found fewer charts
                if (normalizedPaths.length === 0) {
                    const chartPatterns = [/(?:Chart saved to|Saved Section \d+):\s*(.+?\.png)/gi, /Saved:\s*(.+?\.png)/gi, /Saved:\s*([^\n]+\.png)/gi]

                    const chartPaths: string[] = []
                    for (const pattern of chartPatterns) {
                        let match
                        pattern.lastIndex = 0
                        while ((match = pattern.exec(stdout)) !== null) {
                            const chartPath = match[1]?.trim()
                            if (chartPath && chartPath.endsWith('.png') && !chartPaths.includes(chartPath)) {
                                chartPaths.push(chartPath)
                            }
                        }
                    }

                    // Also check lines directly
                    const lines = stdout.split('\n')
                    for (const line of lines) {
                        if (line.includes('Saved:') || line.includes('Saved Section')) {
                            const pngMatch = line.match(/([^\s]+\.png)/)
                            if (pngMatch && pngMatch[1] && !chartPaths.includes(pngMatch[1])) {
                                chartPaths.push(pngMatch[1])
                            }
                        }
                    }

                    normalizedPaths = chartPaths.map((p) => (path.isAbsolute(p) ? p : path.resolve(chartsOutputDir, p))).filter((p) => fs.existsSync(p))

                    console.log(`[${source} chart] Found ${normalizedPaths.length} chart(s) from stdout parsing`)
                }

                // Final fallback: Scan directory for PNG files created in the last few minutes
                if (normalizedPaths.length === 0) {
                    console.warn(`[${source} chart] No charts found in CSV or stdout, scanning directory...`)
                    try {
                        const now = Date.now()
                        const recentFiles: string[] = []

                        function scanDir(dir: string) {
                            const entries = fs.readdirSync(dir, { withFileTypes: true })
                            for (const entry of entries) {
                                const fullPath = path.join(dir, entry.name)
                                if (entry.isDirectory()) {
                                    scanDir(fullPath)
                                } else if (entry.isFile() && entry.name.endsWith('.png')) {
                                    const stats = fs.statSync(fullPath)
                                    // Only include files created in the last 10 minutes
                                    if (now - stats.mtimeMs < 10 * 60 * 1000) {
                                        recentFiles.push(fullPath)
                                    }
                                }
                            }
                        }

                        scanDir(chartsOutputDir)
                        normalizedPaths = recentFiles
                        console.log(`[${source} chart] Found ${normalizedPaths.length} recent chart(s) from directory scan`)
                    } catch (error: any) {
                        console.warn(`[${source} chart] Directory scan failed:`, error.message)
                    }
                }

                if (normalizedPaths.length > 0) {
                    console.log(`[${source} chart] Total charts found: ${normalizedPaths.length}`)
                    if (devMode) {
                        console.log(`[${source} chart] Chart paths:`, normalizedPaths.slice(0, 10))
                        if (normalizedPaths.length > 10) {
                            console.log(`[${source} chart] ... and ${normalizedPaths.length - 10} more`)
                        }
                    }
                } else {
                    console.warn(`[${source} chart] No charts found! Check Python script output above.`)
                }

                resolve(normalizedPaths)
            } else {
                reject(new Error(`Python script failed with code ${code}: ${stderr || stdout}`))
            }
        })

        pythonProcess.on('error', (error) => {
            reject(new Error(`Failed to spawn Python process: ${error.message}`))
        })
    })
}

/**
 * Check which data sources have CSV files available
 */
function checkAvailableDataSources(dataDir: string): string[] {
    const sources = ['calpads', 'nwea', 'iready', 'star', 'cers', 'iab']
    const available: string[] = []

    for (const source of sources) {
        const csvPath = path.join(dataDir, `${source}_data.csv`)
        if (fs.existsSync(csvPath)) {
            available.push(source)
        }
    }

    return available
}

/**
 * Find Python command (python3 or python)
 */
async function findPythonCommand(): Promise<string | null> {
    const commands = ['python3', 'python']

    for (const cmd of commands) {
        try {
            const result = await new Promise<{ success: boolean }>((resolve) => {
                const process = spawn(cmd, ['--version'], { stdio: 'pipe' })
                process.on('close', (code) => resolve({ success: code === 0 }))
                process.on('error', () => resolve({ success: false }))
            })

            if (result.success) {
                return cmd
            }
        } catch {
            continue
        }
    }

    return null
}
