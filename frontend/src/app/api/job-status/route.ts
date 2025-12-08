import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/job-status?jobId=<job_id>
 *
 * Polls the backend for job status
 *
 * Query params:
 *   - jobId: string (required) - Celery task ID
 *
 * Returns:
 *   - success: boolean
 *   - jobId: string
 *   - status: string (PENDING, PROGRESS, SUCCESS, FAILURE)
 *   - step: string (current step description)
 *   - progress: number (0-100)
 *   - charts: string[] (when status is SUCCESS)
 *   - summary: object (when status is SUCCESS)
 *   - error: string (when status is FAILURE)
 */
export async function GET(req: NextRequest) {
    try {
        const jobId = req.nextUrl.searchParams.get('jobId')

        if (!jobId) {
            return NextResponse.json(
                {
                    success: false,
                    error: 'jobId query parameter is required'
                },
                { status: 400 }
            )
        }

        const backendUrl = process.env.BACKEND_URL || 'http://localhost:5000'
        const backendResponse = await fetch(`${backendUrl}/job-status/${jobId}`)

        if (!backendResponse.ok) {
            const errorData = await backendResponse.json().catch(() => ({}))
            throw new Error(errorData.error || `Backend API error: ${backendResponse.status}`)
        }

        const backendData = await backendResponse.json()

        return NextResponse.json({
            success: true,
            ...backendData
        })
    } catch (error: unknown) {
        console.error('Error checking job status:', error)
        const errorMessage = error instanceof Error ? error.message : 'Failed to check job status'
        return NextResponse.json(
            {
                success: false,
                error: errorMessage
            },
            { status: 500 }
        )
    }
}
