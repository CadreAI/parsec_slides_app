import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || process.env.BACKEND_URL || 'http://localhost:5000'

export async function GET(_: Request, { params }: { params: Promise<{ taskId: string }> }) {
    const { taskId } = await params
    if (!taskId) {
        return NextResponse.json({ success: false, error: 'taskId is required' }, { status: 400 })
    }

    try {
        const { userId, getToken } = await auth()
        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        // Get token for backend API authentication
        // Try with backend template first, fallback to default if template doesn't exist
        let token: string | null = null
        try {
            token = await getToken({ template: 'backend' })
        } catch (error) {
            // Template doesn't exist, fallback to default token
            console.log('[API Route /tasks/status] Backend template not found, using default token')
        }
        if (!token) {
            token = await getToken()
        }

        const res = await fetch(`${BACKEND_BASE}/tasks/status/${taskId}`, {
            headers: token ? { Authorization: `Bearer ${token}` } : undefined
        })
        const data = await res.json().catch(() => ({}))
        return NextResponse.json(data, { status: res.status })
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch task status'
        return NextResponse.json({ success: false, error: message }, { status: 500 })
    }
}
