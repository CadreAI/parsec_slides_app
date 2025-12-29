import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'

// Use 127.0.0.1 instead of localhost for API routes to avoid connection issues
const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:5001'

export async function POST(request: Request, context: { params: Promise<{ taskId: string }> }) {
    try {
        // Authenticate with Clerk
        const { userId, getToken } = await auth()

        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        // Await params in Next.js 14+
        const { taskId } = await context.params

        if (!taskId) {
            return NextResponse.json({ success: false, error: 'Task ID is required' }, { status: 400 })
        }

        // Get token for backend API authentication
        let token: string | null = null
        try {
            token = await getToken({ template: 'backend' })
        } catch (_error) {
            // Template doesn't exist, fallback to default token
            console.log('[API Route /tasks/abort] Backend template not found, using default token')
        }
        if (!token) {
            token = await getToken()
        }

        // Forward to Flask backend
        const response = await fetch(`${BACKEND_URL}/tasks/abort/${taskId}`, {
            method: 'POST',
            headers: token ? { Authorization: `Bearer ${token}` } : {}
        })

        const data = await response.json()

        if (!response.ok) {
            return NextResponse.json({ success: false, error: data.error || 'Failed to abort task' }, { status: response.status })
        }

        return NextResponse.json(data, { status: 200 })
    } catch (error) {
        console.error('[API] Error aborting task:', error)
        return NextResponse.json({ success: false, error: 'Internal server error' }, { status: 500 })
    }
}
