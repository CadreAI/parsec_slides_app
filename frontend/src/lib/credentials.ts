import fs from 'fs'
import path from 'path'

/**
 * Check if a credentials file matches a specific type
 */
function isCredentialsType(filePath: string, type: 'oauth' | 'service_account'): boolean {
    try {
        const content = fs.readFileSync(filePath, 'utf-8')
        const creds = JSON.parse(content)

        if (type === 'oauth') {
            return !!(creds.installed || creds.web)
        } else if (type === 'service_account') {
            return creds.type === 'service_account' && !!creds.private_key
        }
        return false
    } catch {
        return false
    }
}

/**
 * Resolve OAuth credentials.json path (for Google Slides API).
 *
 * Priority order:
 * 1. GOOGLE_OAUTH_CREDENTIALS_PATH environment variable
 * 2. GOOGLE_CREDENTIALS_PATH (if it's OAuth credentials)
 * 3. google/credentials.json (default location, if OAuth)
 * 4. ~/.config/google/oauth_credentials.json (backend-specific)
 *
 * @returns Path to OAuth credentials.json file
 * @throws Error if OAuth credentials file not found
 */
export function resolveOAuthCredentialsPath(): string {
    // Try to find project root (where google/ folder is)
    // If we're in frontend/, go up one level
    let rootDir = process.cwd()
    if (rootDir.endsWith('/frontend') || rootDir.endsWith('\\frontend')) {
        rootDir = path.resolve(rootDir, '..')
    }
    // Also check if google/ exists in current dir, if not try parent
    if (!fs.existsSync(path.join(rootDir, 'google', 'credentials.json'))) {
        const parentDir = path.resolve(rootDir, '..')
        if (fs.existsSync(path.join(parentDir, 'google', 'credentials.json'))) {
            rootDir = parentDir
        }
    }

    // Priority 1: Explicit OAuth environment variable
    const oauthEnvPath = process.env.GOOGLE_OAUTH_CREDENTIALS_PATH
    if (oauthEnvPath) {
        const resolved = path.resolve(oauthEnvPath)
        if (fs.existsSync(resolved)) {
            if (isCredentialsType(resolved, 'oauth')) {
                console.log(`[OAuth Credentials] Using GOOGLE_OAUTH_CREDENTIALS_PATH: ${resolved}`)
                return resolved
            }
            console.warn(`[OAuth Credentials] File found but not OAuth type: ${resolved}`)
        } else {
            console.warn(`[OAuth Credentials] GOOGLE_OAUTH_CREDENTIALS_PATH set but file not found: ${resolved}`)
        }
    }

    // Priority 2: Check GOOGLE_CREDENTIALS_PATH (if it's OAuth)
    const envPath = process.env.GOOGLE_CREDENTIALS_PATH
    if (envPath) {
        const resolved = path.resolve(envPath)
        if (fs.existsSync(resolved) && isCredentialsType(resolved, 'oauth')) {
            console.log(`[OAuth Credentials] Using GOOGLE_CREDENTIALS_PATH: ${resolved}`)
            return resolved
        }
    }

    // Priority 3: Default location in project
    const defaultPath = path.join(rootDir, 'google', 'credentials.json')
    if (fs.existsSync(defaultPath) && isCredentialsType(defaultPath, 'oauth')) {
        console.log(`[OAuth Credentials] Using default location: ${defaultPath}`)
        return defaultPath
    }

    // Priority 4: Backend-specific OAuth location
    const homeDir = process.env.HOME || process.env.USERPROFILE
    if (homeDir) {
        const backendPath = path.join(homeDir, '.config', 'google', 'oauth_credentials.json')
        if (fs.existsSync(backendPath) && isCredentialsType(backendPath, 'oauth')) {
            console.log(`[OAuth Credentials] Using backend location: ${backendPath}`)
            return backendPath
        }
    }

    // If none found, throw error with helpful message
    const triedPaths = [
        oauthEnvPath && `GOOGLE_OAUTH_CREDENTIALS_PATH: ${oauthEnvPath}`,
        envPath && `GOOGLE_CREDENTIALS_PATH: ${envPath}`,
        `Default: ${defaultPath}`,
        homeDir && `Backend: ${path.join(homeDir, '.config', 'google', 'oauth_credentials.json')}`
    ]
        .filter(Boolean)
        .join('\n  - ')

    throw new Error(
        `Google OAuth credentials.json not found. Tried:\n  - ${triedPaths}\n\n` +
            `Set GOOGLE_OAUTH_CREDENTIALS_PATH environment variable to point to your OAuth credentials file.`
    )
}

/**
 * Resolve service account credentials.json path (for BigQuery).
 *
 * Priority order:
 * 1. GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH environment variable
 * 2. GOOGLE_APPLICATION_CREDENTIALS (if it's a service account)
 * 3. google/service_account.json (default location)
 * 4. ~/.config/google/service_account.json (backend-specific)
 *
 * @returns Path to service account credentials.json file
 * @throws Error if service account credentials file not found
 */
export function resolveServiceAccountCredentialsPath(): string {
    // Try to find project root (where google/ folder is)
    // If we're in frontend/, go up one level
    let rootDir = process.cwd()
    if (rootDir.endsWith('/frontend') || rootDir.endsWith('\\frontend')) {
        rootDir = path.resolve(rootDir, '..')
    }
    // Also check if google/ exists in current dir, if not try parent
    if (!fs.existsSync(path.join(rootDir, 'google', 'service_account.json'))) {
        const parentDir = path.resolve(rootDir, '..')
        if (fs.existsSync(path.join(parentDir, 'google', 'service_account.json'))) {
            rootDir = parentDir
        }
    }

    // Priority 1: Explicit service account environment variable
    const saEnvPath = process.env.GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH
    if (saEnvPath) {
        const resolved = path.resolve(saEnvPath)
        if (fs.existsSync(resolved)) {
            if (isCredentialsType(resolved, 'service_account')) {
                console.log(`[Service Account Credentials] Using GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH: ${resolved}`)
                return resolved
            }
            console.warn(`[Service Account Credentials] File found but not service account type: ${resolved}`)
        } else {
            console.warn(`[Service Account Credentials] GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH set but file not found: ${resolved}`)
        }
    }

    // Priority 2: Check GOOGLE_APPLICATION_CREDENTIALS (standard for BigQuery)
    const appCredsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS
    if (appCredsPath && appCredsPath.endsWith('.json')) {
        const resolved = path.resolve(appCredsPath)
        if (fs.existsSync(resolved) && isCredentialsType(resolved, 'service_account')) {
            console.log(`[Service Account Credentials] Using GOOGLE_APPLICATION_CREDENTIALS: ${resolved}`)
            return resolved
        }
    }

    // Priority 3: Default service account location in project
    const defaultPath = path.join(rootDir, 'google', 'service_account.json')
    if (fs.existsSync(defaultPath) && isCredentialsType(defaultPath, 'service_account')) {
        console.log(`[Service Account Credentials] Using default location: ${defaultPath}`)
        return defaultPath
    }

    // Priority 4: Backend-specific service account location
    const homeDir = process.env.HOME || process.env.USERPROFILE
    if (homeDir) {
        const backendPath = path.join(homeDir, '.config', 'google', 'service_account.json')
        if (fs.existsSync(backendPath) && isCredentialsType(backendPath, 'service_account')) {
            console.log(`[Service Account Credentials] Using backend location: ${backendPath}`)
            return backendPath
        }
    }

    // If none found, throw error with helpful message
    const triedPaths = [
        saEnvPath && `GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH: ${saEnvPath}`,
        appCredsPath && `GOOGLE_APPLICATION_CREDENTIALS: ${appCredsPath}`,
        `Default: ${defaultPath}`,
        homeDir && `Backend: ${path.join(homeDir, '.config', 'google', 'service_account.json')}`
    ]
        .filter(Boolean)
        .join('\n  - ')

    throw new Error(
        `Google service account credentials.json not found. Tried:\n  - ${triedPaths}\n\n` +
            `Set GOOGLE_SERVICE_ACCOUNT_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS environment variable.`
    )
}

interface OAuthCredentials {
    installed?: {
        client_id: string
        client_secret: string
        redirect_uris: string[]
    }
    web?: {
        client_id: string
        client_secret: string
        redirect_uris: string[]
    }
}

interface ServiceAccountCredentials {
    type: string
    project_id: string
    private_key_id: string
    private_key: string
    client_email: string
    client_id: string
    auth_uri: string
    token_uri: string
    auth_provider_x509_cert_url: string
    client_x509_cert_url: string
}

/**
 * Load and parse OAuth credentials.json (for Google Slides)
 */
export function loadOAuthCredentials(): OAuthCredentials {
    const credsPath = resolveOAuthCredentialsPath()
    const content = fs.readFileSync(credsPath, 'utf-8')
    return JSON.parse(content) as OAuthCredentials
}

/**
 * Load and parse service account credentials.json (for BigQuery)
 */
export function loadServiceAccountCredentials(): ServiceAccountCredentials {
    const credsPath = resolveServiceAccountCredentialsPath()
    const content = fs.readFileSync(credsPath, 'utf-8')
    return JSON.parse(content) as ServiceAccountCredentials
}

/**
 * @deprecated Use loadOAuthCredentials() instead
 * Load and parse credentials.json from resolved path (OAuth only)
 */
export function loadCredentials(): OAuthCredentials {
    return loadOAuthCredentials()
}
