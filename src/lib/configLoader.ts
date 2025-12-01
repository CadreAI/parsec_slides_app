import fs from 'fs'
import path from 'path'
import yaml from 'js-yaml'
import type { PartnerConfig, BaseSettings } from '@/types/config'

/**
 * Locate and load settings.yaml, then load partner-specific YAML from /config_files/{partner_name}.yaml.
 *
 * Resolution order for settings.yaml:
 *  1) Explicit path argument
 *  2) ENV var SETTINGS_YAML
 *  3) Sibling of project root inferred from this script: <repo_root>/settings.yaml
 *  4) Current working directory: ./settings.yaml
 *  5) Walk parents of this script and the CWD until a settings.yaml is found
 *
 * Raises FileNotFoundError with helpful message if missing.
 */
export function loadConfig(configPath?: string): PartnerConfig {
    const rootDir = process.cwd()

    // --- Step 1: locate settings.yaml ---
    let loadPath: string

    if (configPath) {
        const p = path.resolve(configPath)
        if (!fs.existsSync(p)) {
            throw new Error(`settings.yaml not found at explicit path: ${p}`)
        }
        loadPath = p
    } else {
        const envPath = process.env.SETTINGS_YAML
        if (envPath) {
            const p = path.resolve(envPath)
            if (!fs.existsSync(p)) {
                throw new Error(`SETTINGS_YAML points to a missing file: ${p}`)
            }
            loadPath = p
        } else {
            const scriptDir = __dirname
            const repoRoot = path.resolve(scriptDir, '../..')
            const candidates: string[] = [path.join(repoRoot, 'settings.yaml'), path.join(process.cwd(), 'settings.yaml')]

            // Walk up from script directory
            let currentDir = scriptDir
            while (currentDir !== path.dirname(currentDir)) {
                candidates.push(path.join(currentDir, 'settings.yaml'))
                currentDir = path.dirname(currentDir)
            }

            // Walk up from CWD
            let cwdDir = process.cwd()
            while (cwdDir !== path.dirname(cwdDir)) {
                candidates.push(path.join(cwdDir, 'settings.yaml'))
                cwdDir = path.dirname(cwdDir)
            }

            // Remove duplicates
            const uniqueCandidates = Array.from(new Set(candidates))

            const found = uniqueCandidates.find((c) => fs.existsSync(c))
            if (!found) {
                throw new Error(
                    'Could not locate settings.yaml. Looked in:\n' + uniqueCandidates.join('\n') + '\nTip: set env var SETTINGS_YAML to the full path.'
                )
            }
            loadPath = found
        }
    }

    // --- Step 2: load settings.yaml and read partner_name ---
    const baseCfgContent = fs.readFileSync(loadPath, 'utf-8')
    const baseCfg = yaml.load(baseCfgContent) as BaseSettings

    const partnerName = baseCfg.partner_name
    if (!partnerName) {
        throw new Error("settings.yaml must include a 'partner_name' key")
    }

    // --- Step 3: load partner-specific YAML from /config_files/{partner_name}.yaml ---
    const configFilesDir = path.resolve(rootDir, 'config_files')
    const partnerConfigPath = path.join(configFilesDir, `${partnerName}.yaml`)

    if (!fs.existsSync(partnerConfigPath)) {
        throw new Error(`Partner config not found: ${partnerConfigPath}`)
    }

    const partnerCfgContent = fs.readFileSync(partnerConfigPath, 'utf-8')
    const cfg = yaml.load(partnerCfgContent) as PartnerConfig

    // Recursively replace {partner_name} placeholders
    function replacePlaceholders(obj: any): any {
        if (typeof obj === 'string') {
            return obj.replace(/{partner_name}/g, partnerName)
        }
        if (Array.isArray(obj)) {
            return obj.map(replacePlaceholders)
        }
        if (obj && typeof obj === 'object') {
            const result: any = {}
            for (const [key, value] of Object.entries(obj)) {
                result[key] = replacePlaceholders(value)
            }
            return result
        }
        return obj
    }

    const processedCfg = replacePlaceholders(cfg) as PartnerConfig
    processedCfg.partner_name = partnerName

    console.log(`Loaded config for: ${partnerName}`)
    console.log(`Base settings: ${loadPath}`)
    console.log(`Partner config: ${partnerConfigPath}`)

    return processedCfg
}

/**
 * Resolve a fully-qualified table id for a given logical source key.
 *
 * Priority:
 *  1) sources_overrides[table_key] -> return formatted override
 *  2) sources[table_key] + optional sources_suffix[table_key]
 *  3) if none found, return null
 */
export function resolveSource(tableKey: string, cfg: PartnerConfig): string | null {
    // 1) explicit override
    const override = cfg.sources_overrides?.[tableKey]
    if (override) {
        return override
    }

    // 2) base + optional suffix
    const base = cfg.sources[tableKey]
    if (!base) {
        return null
    }
    const suffix = cfg.sources_suffix?.[tableKey] || ''
    const tableId = suffix ? `${base}${suffix}` : base
    return tableId
}
