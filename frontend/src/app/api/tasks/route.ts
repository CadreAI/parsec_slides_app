import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000'

export async function GET(request: Request) {
    try {
        const { userId } = await auth()

        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        // Get optional status and limit filters from query params
        const { searchParams } = new URL(request.url)
        const status = searchParams.get('status') || ''
        const limit = searchParams.get('limit') || ''

        // Build query string
        let queryString = `clerkUserId=${userId}`
        if (status) {
            queryString += `&status=${status}`
        }
        if (limit) {
            queryString += `&limit=${limit}`
        }

        // Fetch tasks from backend
        const res = await fetch(`${BACKEND_BASE}/tasks?${queryString}`)
        const data = await res.json().catch(() => ({ success: false, tasks: [] }))

        return NextResponse.json(data, { status: res.status })
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch tasks'
        console.error('[API] Error:', err)
        return NextResponse.json({ success: false, error: message }, { status: 500 })
    }
}
