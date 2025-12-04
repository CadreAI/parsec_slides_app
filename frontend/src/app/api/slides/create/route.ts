import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
    try {
        const body = await req.json()
        const { title, slides: slidesData, charts: chartPaths, driveFolderUrl, schoolName, quarters, partnerName, enableAIInsights, userPrompt } = body

        if (!title) {
            return NextResponse.json({ error: 'title is required' }, { status: 400 })
        }

        // Forward request to Python backend
        const backendUrl = process.env.BACKEND_URL || 'http://localhost:5000'
        console.log(`[Frontend] Forwarding slide creation to backend: ${backendUrl}/create-slides`)

        try {
            const backendResponse = await fetch(`${backendUrl}/create-slides`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title,
                    charts: chartPaths,
                    driveFolderUrl,
                    enableAIInsights: enableAIInsights !== false, // Default to true
                    userPrompt: userPrompt || undefined, // Optional user prompt for decision LLM
                    slides: slidesData,
                    schoolName,
                    quarters,
                    partnerName
                })
            })

            if (!backendResponse.ok) {
                const errorData = await backendResponse.json()
                throw new Error(errorData.error || `Backend error: ${backendResponse.statusText}`)
            }

            const result = await backendResponse.json()
            return NextResponse.json(result)
        } catch (backendError: unknown) {
            const errorMessage = backendError instanceof Error ? backendError.message : 'Unknown backend error'
            console.error('[Frontend] Backend error:', errorMessage)
            return NextResponse.json({ error: 'Failed to create presentation', details: errorMessage }, { status: 500 })
        }
    } catch (err: unknown) {
        console.error('Slides API error:', err)
        const errorObj = err as { message?: string; error?: { message?: string } }
        const errorMessage = errorObj?.message || errorObj?.error?.message || 'Unknown error'

        return NextResponse.json(
            {
                error: 'Failed to create presentation',
                details: errorMessage
            },
            { status: 500 }
        )
    }
}
