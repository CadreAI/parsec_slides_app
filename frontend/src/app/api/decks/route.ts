import { getSupabaseClient } from '@/lib/supabase'
import { auth } from '@clerk/nextjs/server'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/decks
 * Get all decks for the authenticated user
 */
export async function GET(_req: NextRequest) {
    try {
        const { userId } = await auth()

        if (!userId) {
            return NextResponse.json({ error: 'Not authenticated' }, { status: 401 })
        }

        // Check environment variables
        const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
        const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

        if (!supabaseUrl || !supabaseKey) {
            console.error('[Decks API] Missing Supabase environment variables')
            return NextResponse.json(
                {
                    error: 'Supabase not configured',
                    details: {
                        hasUrl: !!supabaseUrl,
                        hasKey: !!supabaseKey
                    }
                },
                { status: 500 }
            )
        }

        let supabase
        try {
            supabase = await getSupabaseClient()
        } catch (supabaseError: unknown) {
            // Check if it's a Clerk JWT template error
            const error = supabaseError as { message?: string; status?: number; name?: string }
            if (error?.message?.includes('Not Found') || error?.status === 404 || error?.name === 'ClerkAPIResponseError') {
                console.error('[Decks API] Clerk JWT template "supabase" not found')
                return NextResponse.json(
                    {
                        error: 'Clerk JWT template not configured',
                        message: 'The "supabase" JWT template does not exist in Clerk. Please create it following the instructions in CLERK_SUPABASE_SETUP.md',
                        details: {
                            error: error.message || 'Unknown error',
                            fix: 'Go to Clerk Dashboard → JWT Templates → Create template named "supabase"'
                        }
                    },
                    { status: 500 }
                )
            }
            throw supabaseError
        }

        console.log('[Decks API] Fetching decks for user:', userId)

        const { data, error } = await supabase.from('decks').select('*').eq('clerk_user_id', userId).order('created_at', { ascending: false })

        if (error) {
            console.error('[Decks API] Error fetching decks:', error)
            console.error('[Decks API] Error code:', error.code)
            console.error('[Decks API] Error message:', error.message)
            console.error('[Decks API] Error details:', JSON.stringify(error, null, 2))
            console.error('[Decks API] User ID:', userId)

            // Check if it's a "table not found" error
            if (error.code === 'PGRST116' || error.message?.includes('relation') || error.message?.includes('does not exist')) {
                return NextResponse.json(
                    {
                        error: 'Table not found',
                        message: 'The decks table does not exist in Supabase. Please run SUPABASE_DECKS_SCHEMA.sql in your Supabase SQL Editor.',
                        details: error
                    },
                    { status: 500 }
                )
            }

            return NextResponse.json(
                {
                    error: error.message || 'Unknown error',
                    code: error.code,
                    details: error
                },
                { status: 500 }
            )
        }

        console.log('[Decks API] Successfully fetched decks:', data?.length || 0)
        return NextResponse.json({ decks: data || [] }, { status: 200 })
    } catch (error: unknown) {
        // Check if it's a Clerk JWT template error
        if (
            (error as { name?: string; message?: string })?.name === 'ClerkAPIResponseError' ||
            (error as { message?: string })?.message?.includes('Not Found')
        ) {
            console.error('[Decks API] Clerk JWT template error:', error.message)
            return NextResponse.json(
                {
                    error: 'Clerk JWT template not configured',
                    message: 'The "supabase" JWT template does not exist in Clerk. Please create it following the instructions in CLERK_SUPABASE_SETUP.md',
                    details: {
                        error: (error as { message?: string })?.message || 'Unknown error',
                        fix: 'Go to Clerk Dashboard → JWT Templates → Create template named "supabase"'
                    }
                },
                { status: 500 }
            )
        }

        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        console.error('[Decks API] Unexpected error:', errorMessage)
        if (error instanceof Error) {
            console.error('[Decks API] Stack:', error.stack)
        }
        return NextResponse.json({ error: errorMessage }, { status: 500 })
    }
}

/**
 * POST /api/decks
 * Create a new deck
 */
export async function POST(req: NextRequest) {
    try {
        const { userId } = await auth()

        if (!userId) {
            return NextResponse.json({ error: 'Not authenticated' }, { status: 401 })
        }

        const body = await req.json()
        const supabase = await getSupabaseClient()

        const deckData = {
            ...body,
            clerk_user_id: userId
        }

        const { data, error } = await supabase.from('decks').insert(deckData).select().single()

        if (error) {
            console.error('[Decks API] Error creating deck:', error)
            return NextResponse.json({ error: error.message }, { status: 500 })
        }

        return NextResponse.json({ deck: data }, { status: 201 })
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        console.error('[Decks API] Error:', errorMessage)
        return NextResponse.json({ error: errorMessage }, { status: 500 })
    }
}
