import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000'

export async function GET() {
    try {
        const { userId, getToken } = await auth()
        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        const token = await getToken()

        const res = await fetch(`${BACKEND_BASE}/health`, {
            cache: 'no-store',
            headers: { Authorization: `Bearer ${token}` }
        })
        const data = await res.json().catch(() => ({ status: 'unknown' }))

        return NextResponse.json(data, { status: res.status })
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Health check failed'
        return NextResponse.json({ status: 'unreachable', error: message }, { status: 503 })
    }
}
