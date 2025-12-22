import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000'

export async function GET(request: Request) {
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
        } catch (_error) {
            // Template doesn't exist, fallback to default token
            console.log('[API Route /tasks] Backend template not found, using default token')
        }
        if (!token) {
            token = await getToken()
        }
        console.log('[API Route /tasks] Token retrieved:', token ? `Yes (length: ${token.length})` : 'No', 'userId:', userId)

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
        const headers: Record<string, string> = {}
        if (token) {
            headers['Authorization'] = `Bearer ${token}`
            console.log('[API Route /tasks] Sending request with Authorization header to:', `${BACKEND_BASE}/tasks?${queryString}`)
        } else {
            console.warn('[API Route /tasks] No token available, request may fail authentication')
        }

        const res = await fetch(`${BACKEND_BASE}/tasks?${queryString}`, {
            headers
        })
        console.log('[API Route /tasks] Backend response status:', res.status)
        const data = await res.json().catch(() => ({ success: false, tasks: [] }))

        return NextResponse.json(data, { status: res.status })
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch tasks'
        console.error('[API] Error:', err)
        return NextResponse.json({ success: false, error: message }, { status: 500 })
    }
}

export async function DELETE(request: Request) {
    try {
        const { userId, getToken } = await auth()

        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        // Get token for backend API authentication
        let token: string | null = null
        try {
            token = await getToken({ template: 'backend' })
        } catch (_error) {
            console.log('[API Route DELETE /tasks] Backend template not found, using default token')
        }
        if (!token) {
            token = await getToken()
        }

        // Get task_id from query params
        const { searchParams } = new URL(request.url)
        const taskId = searchParams.get('task_id')

        if (!taskId) {
            return NextResponse.json({ success: false, error: 'task_id is required' }, { status: 400 })
        }

        console.log('[API Route DELETE /tasks] Deleting task:', taskId, 'for user:', userId)

        // Delete task from backend
        const headers: Record<string, string> = {}
        if (token) {
            headers['Authorization'] = `Bearer ${token}`
        }

        const res = await fetch(`${BACKEND_BASE}/tasks?task_id=${taskId}&clerkUserId=${userId}`, {
            method: 'DELETE',
            headers
        })

        const data = await res.json().catch(() => ({ success: false }))
        console.log('[API Route DELETE /tasks] Backend response:', res.status, data)

        return NextResponse.json(data, { status: res.status })
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to delete task'
        console.error('[API DELETE /tasks] Error:', err)
        return NextResponse.json({ success: false, error: message }, { status: 500 })
    }
}
