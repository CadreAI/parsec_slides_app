import { lstatSync, readdirSync, rmSync } from 'node:fs'
import { join } from 'node:path'

const DIRECTORIES_TO_CLEAN = ['node_modules', '.next']
const DIRECTORIES_TO_IGNORE: string[] = []

function clean(directory: string, dryRun = false) {
    try {
        for (const file of readdirSync(directory)) {
            const path = join(directory, file)
            if (lstatSync(path).isDirectory()) {
                if (DIRECTORIES_TO_IGNORE.includes(file)) {
                    console.log('🟡 Ignoring:', path)
                    continue
                }
                if (DIRECTORIES_TO_CLEAN.includes(file)) {
                    console.log(`${dryRun ? '🔵 Would Delete' : '🔴 Deleting'}:`, path)
                    if (!dryRun) {
                        rmSync(path, { recursive: true, force: true })
                        console.log('✅ Deleted:', path)
                    }
                } else {
                    clean(path, dryRun)
                }
            }
        }
    } catch (error) {
        console.error('❌ Error while cleaning:', error)
    }
}

const dryRun = process.argv.includes('--dry-run')
clean(process.cwd(), dryRun)
