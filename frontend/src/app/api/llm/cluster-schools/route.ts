import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000'

export async function POST(request: Request) {
    try {
        const { userId } = await auth()

        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        const body = await request.json().catch(() => ({}))

        // Extract required fields
        const schools = body?.schools
        const districtName = body?.district_name

        // Validate required fields
        if (!schools || !Array.isArray(schools)) {
            return NextResponse.json({ success: false, error: 'schools array is required' }, { status: 400 })
        }

        // Forward to Flask backend
        const res = await fetch(`${BACKEND_BASE}/llm/cluster-schools`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                schools,
                district_name: districtName
            })
        })

        const data = await res.json().catch(() => ({}))
        return NextResponse.json(data, { status: res.status })
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to cluster schools'
        return NextResponse.json({ success: false, error: message }, { status: 500 })
    }
}
