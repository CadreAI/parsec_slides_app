'use client'

import { SignOutButton } from '@clerk/nextjs'
import { Calendar, FileText, Plus, ExternalLink, Loader2 } from 'lucide-react'
import Link from 'next/link'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface Deck {
    id: string
    clerk_user_id: string
    title: string
    description?: string
    slide_count?: number
    presentation_id?: string
    presentation_url?: string
    created_at: string
}

export default function Dashboard() {
    const [decks, setDecks] = useState<Deck[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetchDecks()
    }, [])

    const fetchDecks = async () => {
        try {
            setLoading(true)
            setError(null)
            const response = await fetch('/api/decks')
            const data = await response.json()

            if (!response.ok) {
                // Show the actual error message from the API
                const errorMessage = data.error || data.message || 'Failed to fetch decks'
                console.error('[Dashboard] API Error:', data)
                setError(errorMessage)
                setDecks([])
                return
            }

            // Success - set decks
            console.log('[Dashboard] Fetched decks:', data.decks?.length || 0)
            setDecks(data.decks || [])
            setError(null)
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to load decks'
            console.error('[Dashboard] Fetch error:', err)
            setError(errorMessage)
            setDecks([])
        } finally {
            setLoading(false)
        }
    }

    const formatDate = (dateString: string) => {
        const date = new Date(dateString)
        return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
    }

    return (
        <div className="min-h-screen p-8">
            <div className="mx-auto max-w-7xl">
                {/* Header */}
                <div className="mb-8 flex items-center justify-between">
                    <div>
                        <h1 className="mb-2 text-3xl font-bold">Parsec Academy Dashboard</h1>
                        <p className="text-muted-foreground">View and manage your slide presentations</p>
                    </div>
                    <div className="flex items-center gap-3">
                        <SignOutButton signOutOptions={{ redirectUrl: '/sign-in' }}>
                            <Button variant="outline">Sign Out</Button>
                        </SignOutButton>
                        <Link href="/create-deck">
                            <Button>
                                <Plus className="mr-2 h-4 w-4" />
                                Create Deck
                            </Button>
                        </Link>
                    </div>
                </div>

                {/* Decks Preview */}
                <div>
                    <h2 className="mb-6 text-2xl font-semibold">Your Decks</h2>

                    {loading && (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 className="text-muted-foreground h-8 w-8 animate-spin" />
                            <span className="text-muted-foreground ml-2">Loading decks...</span>
                        </div>
                    )}

                    {error && <div className="border-destructive bg-destructive/10 text-destructive rounded-lg border p-4">{error}</div>}

                    {!loading && !error && decks.length === 0 && (
                        <div className="rounded-lg border border-dashed p-12 text-center">
                            <FileText className="text-muted-foreground mx-auto mb-4 h-12 w-12" />
                            <h3 className="mb-2 text-lg font-semibold">No decks yet</h3>
                            <p className="text-muted-foreground mb-4">Create your first deck to get started</p>
                            <Link href="/create-deck">
                                <Button>
                                    <Plus className="mr-2 h-4 w-4" />
                                    Create Your First Deck
                                </Button>
                            </Link>
                        </div>
                    )}

                    {!loading && !error && decks.length > 0 && (
                        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
                            {decks.map((deck) => (
                                <Card
                                    key={deck.id}
                                    className="cursor-pointer transition-shadow hover:shadow-lg"
                                    onClick={() => {
                                        if (deck.presentation_url) {
                                            window.open(deck.presentation_url, '_blank')
                                        }
                                    }}
                                >
                                    <CardHeader>
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1">
                                                <CardTitle>{deck.title}</CardTitle>
                                                {deck.description && <CardDescription className="mt-2">{deck.description}</CardDescription>}
                                            </div>
                                        </div>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="space-y-4">
                                            <div className="text-muted-foreground flex items-center gap-4 text-sm">
                                                {deck.slide_count !== undefined && (
                                                    <div className="flex items-center gap-1">
                                                        <FileText className="h-4 w-4" />
                                                        <span>
                                                            {deck.slide_count} slide{deck.slide_count !== 1 ? 's' : ''}
                                                        </span>
                                                    </div>
                                                )}
                                                <div className="flex items-center gap-1">
                                                    <Calendar className="h-4 w-4" />
                                                    <span>{formatDate(deck.created_at)}</span>
                                                </div>
                                            </div>
                                            <div className="flex items-center justify-end">
                                                {deck.presentation_url && (
                                                    <Button
                                                        variant="ghost"
                                                        size="sm"
                                                        onClick={(e) => {
                                                            e.stopPropagation()
                                                            window.open(deck.presentation_url, '_blank')
                                                        }}
                                                    >
                                                        <ExternalLink className="mr-2 h-4 w-4" />
                                                        Open Slides
                                                    </Button>
                                                )}
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
