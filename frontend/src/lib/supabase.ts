import { auth } from '@clerk/nextjs/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error('Missing Supabase environment variables')
}

/**
 * Get Supabase client for server-side use with Clerk authentication
 * This client includes the Clerk JWT token for Row Level Security
 */
export async function getSupabaseClient() {
    try {
        const { getToken } = await auth()
        let token: string | null = null

        try {
            token = await getToken({ template: 'supabase' })
        } catch (tokenError: unknown) {
            // If template doesn't exist, Clerk will throw an error
            if ((tokenError as { status?: number; message?: string })?.status === 404 || (tokenError as { message?: string })?.message?.includes('Not Found')) {
                console.error('[Supabase] Clerk JWT template "supabase" not found.')
                console.error('[Supabase] Please create a JWT template named "supabase" in Clerk Dashboard.')
                console.error('[Supabase] See CLERK_SUPABASE_SETUP.md for instructions.')
                // Continue without token - RLS will block access but won't crash
            } else {
                throw tokenError
            }
        }

        const supabase = createClient(supabaseUrl, supabaseAnonKey, {
            global: {
                headers: token
                    ? {
                          Authorization: `Bearer ${token}`
                      }
                    : {}
            }
        })

        return supabase
    } catch (error) {
        console.error('[Supabase] Error creating Supabase client:', error)
        throw error
    }
}

/**
 * Get Supabase client for client-side use
 * Note: Client-side RLS relies on Clerk session token
 */
export function getSupabaseClientClient() {
    return createClient(supabaseUrl, supabaseAnonKey)
}
