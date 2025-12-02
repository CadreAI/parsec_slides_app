// lib/googleSlides.ts
import fs from 'fs'
import { google } from 'googleapis'
import path from 'path'
import readline from 'readline'
import { loadOAuthCredentials } from './credentials'

const SCOPES = ['https://www.googleapis.com/auth/presentations', 'https://www.googleapis.com/auth/drive']

// Paths (adjust if needed)
const ROOT_DIR = process.cwd()
const TOKEN_PATH = path.join(ROOT_DIR, 'google', 'token.json')

function createOAuthClient() {
    const credentials = loadOAuthCredentials()

    const creds = credentials.installed || credentials.web
    if (!creds) {
        throw new Error('No OAuth credentials found')
    }
    const { client_secret, client_id, redirect_uris } = creds

    // For desktop apps, use the out-of-band redirect URI
    const redirectUri = credentials.installed ? 'urn:ietf:wg:oauth:2.0:oob' : redirect_uris[0]

    return new google.auth.OAuth2(client_id, client_secret, redirectUri)
}

async function getNewToken(oAuth2Client: ReturnType<typeof createOAuthClient>): Promise<ReturnType<typeof createOAuthClient>['credentials']> {
    const authUrl = oAuth2Client.generateAuthUrl({
        access_type: 'offline',
        scope: SCOPES
    })

    console.log('\n========================================')
    console.log('STEP 1: Authorize this app')
    console.log('========================================')
    console.log('\nVisit this URL in your browser:\n')
    console.log(authUrl)
    console.log('\n========================================')
    console.log('STEP 2: Get the authorization code')
    console.log('========================================')

    // Check if using web application OAuth
    const credentials = loadOAuthCredentials()

    if (credentials.web) {
        console.log('\n⚠️  You are using Web Application OAuth')
        console.log('\nAfter authorizing, you will be redirected to: http://localhost:3001')
        console.log('\nThe authorization code will be in the URL, like this:')
        console.log('  http://localhost:3001/?code=4/0Aean...')
        console.log('\n📋 HOW TO GET THE CODE:')
        console.log('  1. After clicking "Allow", check your browser URL bar')
        console.log('  2. Look for "code=" in the URL')
        console.log('  3. Copy everything after "code=" and before "&" (if present)')
        console.log('  4. If you see an error page, the code is still in the URL bar!')
        console.log('\n💡 TIP: Start your dev server (bun run dev) to see a helpful page')
    } else {
        console.log('\nAfter authorizing, you will see a page with a code.')
        console.log('Copy the entire code from that page.')
    }

    console.log('\nThe code will look like: 4/0Aean... or similar')
    console.log('\n========================================\n')

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    })

    const code: string = await new Promise((resolve) => {
        rl.question('Paste the authorization code here: ', (answer) => {
            rl.close()
            resolve(answer.trim())
        })
    })

    if (!code) {
        throw new Error('No authorization code provided')
    }

    console.log('\nExchanging code for token...')
    const { tokens } = await oAuth2Client.getToken(code)

    // Ensure token directory exists
    const tokenDir = path.dirname(TOKEN_PATH)
    if (!fs.existsSync(tokenDir)) {
        fs.mkdirSync(tokenDir, { recursive: true })
    }

    fs.writeFileSync(TOKEN_PATH, JSON.stringify(tokens, null, 2))
    console.log('✅ Token stored to', TOKEN_PATH)
    return tokens
}

export async function getSlidesClient() {
    const oAuth2Client = createOAuthClient()

    try {
        const token = JSON.parse(fs.readFileSync(TOKEN_PATH, 'utf-8'))
        oAuth2Client.setCredentials(token)
    } catch {
        // First-time setup: run in terminal to generate token
        const token = await getNewToken(oAuth2Client)
        oAuth2Client.setCredentials(token)
    }

    return google.slides({ version: 'v1', auth: oAuth2Client })
}

export async function getDriveClient() {
    const oAuth2Client = createOAuthClient()

    try {
        const token = JSON.parse(fs.readFileSync(TOKEN_PATH, 'utf-8'))
        oAuth2Client.setCredentials(token)
    } catch {
        // First-time setup: run in terminal to generate token
        const token = await getNewToken(oAuth2Client)
        oAuth2Client.setCredentials(token)
    }

    return google.drive({ version: 'v3', auth: oAuth2Client })
}

/**
 * Extract folder ID from Google Drive folder URL
 * Supports formats like:
 * - https://drive.google.com/drive/folders/FOLDER_ID
 * - https://drive.google.com/drive/folders/FOLDER_ID?dmr=1&ec=wgc-drive-hero-goto
 */
export function extractFolderIdFromUrl(url: string): string | null {
    if (!url) return null

    // Extract folder ID from URL
    const match = url.match(/\/folders\/([a-zA-Z0-9_-]+)/)
    return match ? match[1] : null
}

/**
 * Upload an image to Google Drive and return a URL that can be used in Google Slides
 * @param imagePath - Path to the image file
 * @param fileName - Optional custom file name
 * @param folderId - Optional Google Drive folder ID to upload to
 * @param makePublic - Whether to make the file publicly accessible (default: false)
 *                     If false, tries to use Drive file without public access (may not work)
 *                     If true, makes file public (more reliable but less private)
 * @returns Drive URL that can be used in Google Slides API
 *
 * Note: Once images are embedded into Google Slides, they become part of the presentation
 * file itself and don't require the Drive file to remain accessible. However, the Slides
 * API needs a URL to fetch the image initially.
 */
export async function uploadImageToDrive(imagePath: string, fileName?: string, folderId?: string | null, makePublic: boolean = false): Promise<string> {
    const drive = await getDriveClient()

    if (!fs.existsSync(imagePath)) {
        throw new Error(`Image file not found: ${imagePath}`)
    }

    const ext = path.extname(imagePath).toLowerCase()
    const mimeType = ext === '.png' ? 'image/png' : ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : 'image/png'
    const name = fileName || path.basename(imagePath)

    // Create a readable stream from the file
    const fileStream = fs.createReadStream(imagePath)

    // Upload file to Drive
    const fileMetadata: { name: string; mimeType: string; parents?: string[] } = {
        name: name,
        mimeType: mimeType
    }

    // Add folder ID if provided
    if (folderId) {
        fileMetadata.parents = [folderId]
    }

    const media = {
        mimeType: mimeType,
        body: fileStream
    }

    const file = await drive.files.create({
        requestBody: fileMetadata,
        media: media,
        fields: 'id'
    })

    if (!file.data.id) {
        throw new Error('Failed to upload image to Drive: No file ID returned')
    }

    const fileId = file.data.id

    if (makePublic) {
        // Make the file publicly accessible (more reliable for Slides API)
        await drive.permissions.create({
            fileId: fileId,
            requestBody: {
                role: 'reader',
                type: 'anyone'
            }
        })
    }
    // Note: Even if not made public, we return the URL format
    // Google Slides API may be able to access it if using same service account
    // If embedding fails, the caller should retry with makePublic=true

    // Return the direct image URL format
    // Once embedded in Slides, the image becomes part of the presentation file
    return `https://drive.google.com/uc?export=view&id=${fileId}`
}
