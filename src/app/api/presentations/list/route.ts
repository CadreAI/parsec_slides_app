import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

interface PresentationMetadata {
    presentationId: string
    title: string
    presentationUrl: string
    createdAt: string
    period: 'BOY' | 'MOY' | 'EOY'
    school: string
    partnerName?: string
    chartCount?: number
}

export async function GET(req: NextRequest) {
    try {
        const { searchParams } = new URL(req.url)
        const school = searchParams.get('school') || 'Parsec Academy'
        const period = searchParams.get('period') // Optional filter: BOY, MOY, EOY

        const metadataFile = path.join(process.cwd(), 'data', 'presentations.json')

        if (!fs.existsSync(metadataFile)) {
            return NextResponse.json({ presentations: [] })
        }

        const content = fs.readFileSync(metadataFile, 'utf-8')
        let presentations: PresentationMetadata[] = JSON.parse(content)

        // Filter by school
        presentations = presentations.filter((p) => p.school === school)

        // Filter by period if provided
        if (period && (period === 'BOY' || period === 'MOY' || period === 'EOY')) {
            presentations = presentations.filter((p) => p.period === period)
        }

        // Sort by creation date (newest first)
        presentations.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())

        return NextResponse.json({ presentations })
    } catch (error: unknown) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        console.error('Error fetching presentations:', errorMessage)
        return NextResponse.json({ error: errorMessage, presentations: [] }, { status: 500 })
    }
}
