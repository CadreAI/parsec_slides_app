'use client'

import { SignOutButton } from '@clerk/nextjs'
import { Calendar, ChevronLeft, ChevronRight, ExternalLink, FileText, Loader2, Plus, X } from 'lucide-react'
import Link from 'next/link'
import { useCallback, useEffect, useRef, useState, useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select } from '@/components/ui/select'

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

interface Task {
    celery_task_id: string
    task_type: string
    status: string
    result?: unknown
    error_message?: string
}

interface ApiTask {
    celery_task_id: string
    task_type: string
    status: string
    result?: unknown
    error_message?: string
}

export default function Dashboard() {
    const [decks, setDecks] = useState<Deck[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [tasks, setTasks] = useState<Task[]>([])
    const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest')
    const [currentPage, setCurrentPage] = useState(1)
    const [itemsPerPage] = useState(9) // 3x3 grid
    const pollingRefs = useRef<Map<string, NodeJS.Timeout>>(new Map())

    const fetchDecks = useCallback(async () => {
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
    }, [])

    const startPolling = useCallback(
        (taskId: string, taskType: string) => {
            // Clear existing polling interval for this task
            const existingInterval = pollingRefs.current.get(taskId)
            if (existingInterval) {
                clearInterval(existingInterval)
            }

            const interval = setInterval(async () => {
                try {
                    const statusRes = await fetch(`/api/tasks/status/${taskId}`)
                    const statusJson = await statusRes.json()

                    if (!statusRes.ok || statusJson.error) {
                        console.error(`[Dashboard] Error polling task ${taskId}:`, statusJson.error)
                    }

                    // Update the task in the tasks array
                    setTasks((prevTasks) =>
                        prevTasks.map((task) =>
                            task.celery_task_id === taskId
                                ? {
                                      ...task,
                                      status: statusJson.state,
                                      result: statusJson.result || task.result,
                                      error_message: statusJson.error || task.error_message
                                  }
                                : task
                        )
                    )

                    if (['SUCCESS', 'FAILURE'].includes(statusJson.state)) {
                        // Stop polling this task
                        const intervalToStop = pollingRefs.current.get(taskId)
                        if (intervalToStop) {
                            clearInterval(intervalToStop)
                            pollingRefs.current.delete(taskId)
                        }

                        if (statusJson.state === 'SUCCESS') {
                            // Task completed successfully - refresh decks and hide task card after delay
                            console.log('[Dashboard] Task completed successfully, refreshing decks list')
                            if (taskType === 'create_deck_with_slides') {
                                fetchDecks()
                            }
                            // Remove the task from the list after 2 seconds
                            setTimeout(() => {
                                setTasks((prevTasks) => prevTasks.filter((t) => t.celery_task_id !== taskId))
                            }, 2000)
                        }
                        // If FAILURE, keep showing the task so user sees the error
                    }
                } catch (err) {
                    console.error(`[Dashboard] Error polling task ${taskId}:`, err)
                    const intervalToStop = pollingRefs.current.get(taskId)
                    if (intervalToStop) {
                        clearInterval(intervalToStop)
                        pollingRefs.current.delete(taskId)
                    }
                }
            }, 2000)

            pollingRefs.current.set(taskId, interval)
        },
        [fetchDecks]
    )

    useEffect(() => {
        const init = async () => {
            fetchDecks()

            // Fetch user's tasks from DB (all tasks, sorted by most recent)
            try {
                const response = await fetch('/api/tasks')
                const data = await response.json()

                if (response.ok && data.tasks && data.tasks.length > 0) {
                    // Filter for only active or failed tasks (not SUCCESS)
                    const activeTasks = (data.tasks as ApiTask[]).filter((t) => ['PENDING', 'STARTED', 'FAILURE'].includes(t.status))

                    console.log('[Dashboard] Found', activeTasks.length, 'active/failed tasks')

                    // Set all active tasks in state
                    setTasks(
                        activeTasks.map((t) => ({
                            celery_task_id: t.celery_task_id,
                            task_type: t.task_type,
                            status: t.status,
                            result: t.result,
                            error_message: t.error_message
                        }))
                    )

                    // Start polling for tasks that are still pending/started
                    activeTasks.forEach((task) => {
                        if (['PENDING', 'STARTED'].includes(task.status)) {
                            startPolling(task.celery_task_id, task.task_type)
                        }
                    })
                }
            } catch (err) {
                console.error('[Dashboard] Error fetching tasks:', err)
            }
        }

        init()

        return () => {
            // Clear all polling intervals
            const refs = pollingRefs.current
            refs.forEach((interval) => clearInterval(interval))
            refs.clear()
        }
    }, [startPolling, fetchDecks])

    const formatDate = (dateString: string) => {
        const date = new Date(dateString)
        return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
    }

    const dismissTask = async (taskId: string) => {
        try {
            // Call backend to delete the task
            const response = await fetch(`/api/tasks?task_id=${taskId}`, {
                method: 'DELETE'
            })

            if (response.ok) {
                console.log('[Dashboard] Successfully deleted task:', taskId)
                // Remove from UI
                setTasks((prevTasks) => prevTasks.filter((t) => t.celery_task_id !== taskId))

                // Clear any polling for this task
                const intervalToStop = pollingRefs.current.get(taskId)
                if (intervalToStop) {
                    clearInterval(intervalToStop)
                    pollingRefs.current.delete(taskId)
                }
            } else {
                const data = await response.json()
                console.error('[Dashboard] Failed to delete task:', data.error)
                // Still remove from UI even if backend delete fails
                setTasks((prevTasks) => prevTasks.filter((t) => t.celery_task_id !== taskId))
            }
        } catch (error) {
            console.error('[Dashboard] Error deleting task:', error)
            // Still remove from UI even if request fails
            setTasks((prevTasks) => prevTasks.filter((t) => t.celery_task_id !== taskId))
        }
    }

    // Sort and paginate decks
    const sortedDecks = useMemo(() => {
        const sorted = [...decks].sort((a, b) => {
            const dateA = new Date(a.created_at).getTime()
            const dateB = new Date(b.created_at).getTime()
            return sortOrder === 'newest' ? dateB - dateA : dateA - dateB
        })
        return sorted
    }, [decks, sortOrder])

    const totalPages = Math.ceil(sortedDecks.length / itemsPerPage)

    const paginatedDecks = useMemo(() => {
        const startIndex = (currentPage - 1) * itemsPerPage
        const endIndex = startIndex + itemsPerPage
        return sortedDecks.slice(startIndex, endIndex)
    }, [sortedDecks, currentPage, itemsPerPage])

    // Reset to page 1 when sort order changes
    useEffect(() => {
        setCurrentPage(1)
    }, [sortOrder])

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
                    <div className="mb-6 flex items-center justify-between">
                        <h2 className="text-2xl font-semibold">Your Decks</h2>
                        {decks.length > 0 && (
                            <div className="flex items-center gap-2">
                                <span className="text-muted-foreground text-sm">Sort by:</span>
                                <Select value={sortOrder} onChange={(e) => setSortOrder(e.target.value as 'newest' | 'oldest')} className="w-40">
                                    <option value="newest">Newest First</option>
                                    <option value="oldest">Oldest First</option>
                                </Select>
                            </div>
                        )}
                    </div>

                    {loading && (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 className="text-muted-foreground h-8 w-8 animate-spin" />
                            <span className="text-muted-foreground ml-2">Loading decks...</span>
                        </div>
                    )}

                    {error && <div className="border-destructive bg-destructive/10 text-destructive rounded-lg border p-4">{error}</div>}

                    {!loading && !error && decks.length === 0 && tasks.length === 0 && (
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

                    {!loading && !error && (decks.length > 0 || tasks.length > 0) && (
                        <>
                            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
                                {/* In-progress/Failed tasks as compact cards - always show on current page */}
                                {currentPage === 1 &&
                                    tasks.map((task) => (
                                        <Card
                                            key={task.celery_task_id}
                                            className={`relative ${
                                                task.status === 'FAILURE'
                                                    ? 'border-red-200 bg-red-50/50 dark:border-red-900 dark:bg-red-950/20'
                                                    : 'border-blue-200 bg-blue-50/50 dark:border-blue-900 dark:bg-blue-950/20'
                                            }`}
                                        >
                                            {task.status === 'FAILURE' && (
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    className="absolute right-2 top-2 h-6 w-6 p-0 text-red-600 hover:bg-red-100 hover:text-red-700 dark:text-red-400 dark:hover:bg-red-900/30"
                                                    onClick={() => dismissTask(task.celery_task_id)}
                                                >
                                                    <X className="h-4 w-4" />
                                                </Button>
                                            )}
                                            <CardHeader>
                                                <div className="flex items-start justify-between">
                                                    <div className="flex-1">
                                                        <CardTitle className="flex items-center gap-2 text-base">
                                                            {task.status === 'PENDING' && <Loader2 className="h-4 w-4 animate-spin text-blue-600" />}
                                                            {task.status === 'STARTED' && <Loader2 className="h-4 w-4 animate-spin text-blue-600" />}
                                                            {task.status === 'FAILURE' && <span className="text-red-600">⚠</span>}
                                                            {task.task_type === 'create_deck_with_slides' ? 'Creating Deck' : 'Processing Task'}
                                                        </CardTitle>
                                                    </div>
                                                </div>
                                            </CardHeader>
                                            <CardContent>
                                                <div className="space-y-3">
                                                    <div className="flex items-center justify-between">
                                                        <span className="text-sm font-medium">Status:</span>
                                                        <span
                                                            className={`rounded-full px-3 py-1 text-xs font-semibold ${
                                                                task.status === 'PENDING'
                                                                    ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                                                                    : task.status === 'STARTED'
                                                                      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400'
                                                                      : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                                                            }`}
                                                        >
                                                            {task.status === 'PENDING' && 'Waiting'}
                                                            {task.status === 'STARTED' && 'Generating...'}
                                                            {task.status === 'FAILURE' && 'Failed'}
                                                        </span>
                                                    </div>
                                                    {task.status === 'FAILURE' && (
                                                        <p className="text-xs text-red-600 dark:text-red-400">Generation failed. Please try again.</p>
                                                    )}
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}

                                {/* Completed decks - paginated */}
                                {paginatedDecks.map((deck) => (
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

                            {/* Pagination Controls */}
                            {totalPages > 1 && (
                                <div className="mt-8 flex flex-col items-center gap-4">
                                    <div className="text-muted-foreground text-sm">
                                        Page {currentPage} of {totalPages} • {sortedDecks.length} deck{sortedDecks.length !== 1 ? 's' : ''}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
                                            disabled={currentPage === 1}
                                        >
                                            <ChevronLeft className="h-4 w-4" />
                                            Previous
                                        </Button>

                                        <div className="flex items-center gap-1">
                                            {/* Show first page */}
                                            {totalPages <= 7 ? (
                                                // Show all pages if 7 or fewer
                                                Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                                                    <Button
                                                        key={page}
                                                        variant={currentPage === page ? 'default' : 'outline'}
                                                        size="sm"
                                                        onClick={() => setCurrentPage(page)}
                                                        className="h-8 w-8 p-0"
                                                    >
                                                        {page}
                                                    </Button>
                                                ))
                                            ) : (
                                                // Show ellipsis for many pages
                                                <>
                                                    <Button
                                                        variant={currentPage === 1 ? 'default' : 'outline'}
                                                        size="sm"
                                                        onClick={() => setCurrentPage(1)}
                                                        className="h-8 w-8 p-0"
                                                    >
                                                        1
                                                    </Button>

                                                    {currentPage > 3 && <span className="px-1">...</span>}

                                                    {/* Show pages around current page */}
                                                    {Array.from({ length: totalPages }, (_, i) => i + 1)
                                                        .filter((page) => page > 1 && page < totalPages && Math.abs(page - currentPage) <= 1)
                                                        .map((page) => (
                                                            <Button
                                                                key={page}
                                                                variant={currentPage === page ? 'default' : 'outline'}
                                                                size="sm"
                                                                onClick={() => setCurrentPage(page)}
                                                                className="h-8 w-8 p-0"
                                                            >
                                                                {page}
                                                            </Button>
                                                        ))}

                                                    {currentPage < totalPages - 2 && <span className="px-1">...</span>}

                                                    <Button
                                                        variant={currentPage === totalPages ? 'default' : 'outline'}
                                                        size="sm"
                                                        onClick={() => setCurrentPage(totalPages)}
                                                        className="h-8 w-8 p-0"
                                                    >
                                                        {totalPages}
                                                    </Button>
                                                </>
                                            )}
                                        </div>

                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
                                            disabled={currentPage === totalPages}
                                        >
                                            Next
                                            <ChevronRight className="h-4 w-4" />
                                        </Button>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}
