import { getSupabaseClient } from '@/lib/supabase'
import { auth } from '@clerk/nextjs/server'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/decks/[id]
 * Get a specific deck by ID
 */
export async function GET(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
    try {
        const { userId } = await auth()

        if (!userId) {
            return NextResponse.json({ error: 'Not authenticated' }, { status: 401 })
        }

        const supabase = await getSupabaseClient()

        const { id } = await params
        const { data, error } = await supabase.from('decks').select('*').eq('id', id).eq('clerk_user_id', userId).single()

        if (error) {
            console.error('[Decks API] Error fetching deck:', error)
            return NextResponse.json({ error: error.message }, { status: 500 })
        }

        if (!data) {
            return NextResponse.json({ error: 'Deck not found' }, { status: 404 })
        }

        return NextResponse.json({ deck: data }, { status: 200 })
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        console.error('[Decks API] Error:', errorMessage)
        return NextResponse.json({ error: errorMessage }, { status: 500 })
    }
}

/**
 * PUT /api/decks/[id]
 * Update a deck
 */
export async function PUT(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
    try {
        const { userId } = await auth()

        if (!userId) {
            return NextResponse.json({ error: 'Not authenticated' }, { status: 401 })
        }

        const body = await req.json()
        const supabase = await getSupabaseClient()
        const { id } = await params

        // Remove clerk_user_id from body if present (shouldn't be changed)
        const { clerk_user_id: _clerk_user_id, ...updateData } = body

        const { data, error } = await supabase.from('decks').update(updateData).eq('id', id).eq('clerk_user_id', userId).select().single()

        if (error) {
            console.error('[Decks API] Error updating deck:', error)
            return NextResponse.json({ error: error.message }, { status: 500 })
        }

        return NextResponse.json({ deck: data }, { status: 200 })
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        console.error('[Decks API] Error:', errorMessage)
        return NextResponse.json({ error: errorMessage }, { status: 500 })
    }
}

/**
 * DELETE /api/decks/[id]
 * Delete a deck
 */
export async function DELETE(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
    try {
        const { userId } = await auth()

        if (!userId) {
            return NextResponse.json({ error: 'Not authenticated' }, { status: 401 })
        }

        const supabase = await getSupabaseClient()
        const { id } = await params

        const { error } = await supabase.from('decks').delete().eq('id', id).eq('clerk_user_id', userId)

        if (error) {
            console.error('[Decks API] Error deleting deck:', error)
            return NextResponse.json({ error: error.message }, { status: 500 })
        }

        return NextResponse.json({ success: true }, { status: 200 })
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        console.error('[Decks API] Error:', errorMessage)
        return NextResponse.json({ error: errorMessage }, { status: 500 })
    }
}
