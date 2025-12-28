import { auth } from '@clerk/nextjs/server'
import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000'

/**
 * POST /api/tasks/abort/[taskId]
 * Abort a running Celery task
 */
export async function POST(req: NextRequest, { params }: { params: Promise<{ taskId: string }> }) {
    try {
        const { userId } = await auth()

        if (!userId) {
            return NextResponse.json({ error: 'Not authenticated' }, { status: 401 })
        }

        const { taskId } = await params
        const { getToken } = await auth()
        const token = await getToken()

        if (!token) {
            return NextResponse.json({ error: 'Failed to get auth token' }, { status: 401 })
        }

        console.log(`[Tasks API] Aborting task: ${taskId}`)

        // Call backend to abort the task
        const response = await fetch(`${BACKEND_URL}/tasks/abort/${taskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${token}`
            }
        })

        const data = await response.json()

        if (!response.ok) {
            console.error('[Tasks API] Backend error aborting task:', data.error)
            return NextResponse.json({ error: data.error || 'Failed to abort task' }, { status: response.status })
        }

        console.log(`[Tasks API] Task ${taskId} aborted successfully`)
        return NextResponse.json(data, { status: 200 })
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        console.error('[Tasks API] Error:', errorMessage)
        return NextResponse.json({ error: errorMessage }, { status: 500 })
    }
}
