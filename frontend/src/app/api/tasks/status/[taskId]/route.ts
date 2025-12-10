import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000'

export async function GET(_: Request, { params }: { params: Promise<{ taskId: string }> }) {
    const { taskId } = await params
    if (!taskId) {
        return NextResponse.json({ success: false, error: 'taskId is required' }, { status: 400 })
    }

    try {
        const res = await fetch(`${BACKEND_URL}/tasks/status/${taskId}`)
        const data = await res.json().catch(() => ({}))
        return NextResponse.json(data, { status: res.status })
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch task status'
        return NextResponse.json({ success: false, error: message }, { status: 500 })
    }
}
