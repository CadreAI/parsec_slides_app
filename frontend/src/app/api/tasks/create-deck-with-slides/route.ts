import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'

const BACKEND_BASE = process.env.BACKEND_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000'

export async function POST(request: Request) {
    try {
        const { userId, getToken } = await auth()

        if (!userId) {
            return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
        }

        const token = await getToken()

        const body = await request.json().catch(() => ({}))

        // Extract all required fields
        const partnerName = body?.partnerName
        const config = body?.config
        const chartFilters = body?.chartFilters
        const title = body?.title

        // Log themeColor for debugging
        console.log('[API Route] Received themeColor from frontend:', body?.themeColor)

        // Validate required fields
        if (!partnerName) {
            return NextResponse.json({ success: false, error: 'partnerName is required' }, { status: 400 })
        }
        if (!title) {
            return NextResponse.json({ success: false, error: 'title is required' }, { status: 400 })
        }

        // Forward to backend with all parameters
        const requestPayload = {
            partnerName,
            config,
            chartFilters,
            title,
            clerkUserId: userId,
            driveFolderUrl: body?.driveFolderUrl,
            enableAIInsights: body?.enableAIInsights ?? true,
            userPrompt: body?.userPrompt,
            description: body?.description,
            themeColor: body?.themeColor // Forward themeColor to backend
        }
        console.log('[API Route] Forwarding themeColor to backend:', requestPayload.themeColor)

        const res = await fetch(`${BACKEND_BASE}/tasks/create-deck-with-slides`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(token ? { Authorization: `Bearer ${token}` } : {})
            },
            body: JSON.stringify({
                partnerName,
                config,
                chartFilters,
                title,
                clerkUserId: userId,
                driveFolderUrl: body?.driveFolderUrl,
                enableAIInsights: body?.enableAIInsights ?? true,
                userPrompt: body?.userPrompt,
                description: body?.description
            })
        })

        const data = await res.json().catch(() => ({}))
        return NextResponse.json(data, { status: res.status })
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to queue task'
        return NextResponse.json({ success: false, error: message }, { status: 500 })
    }
}
