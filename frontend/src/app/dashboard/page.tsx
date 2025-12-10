'use client'

import { SignOutButton } from '@clerk/nextjs'
import { Calendar, ExternalLink, FileText, Loader2, Plus } from 'lucide-react'
import Link from 'next/link'
import { useCallback, useEffect, useRef, useState } from 'react'

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

                {/* Task Status Cards */}
                {tasks.length > 0 && (
                    <div className="mb-6 space-y-4">
                        {tasks.map((task) => (
                            <Card key={task.celery_task_id}>
                                <CardHeader>
                                    <CardTitle>{task.task_type === 'create_deck_with_slides' ? 'Deck Generation Progress' : 'Active Task'}</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        <div>
                                            <p className="font-semibold">{task.task_type === 'create_deck_with_slides' ? 'Creating Slide Deck' : 'Task'}</p>
                                            <p className="text-muted-foreground break-all text-sm">ID: {task.celery_task_id}</p>
                                        </div>
                                        <div className="text-sm">
                                            Status:{' '}
                                            <span
                                                className={`font-semibold ${
                                                    task.status === 'SUCCESS'
                                                        ? 'text-green-600'
                                                        : task.status === 'FAILURE'
                                                          ? 'text-red-600'
                                                          : task.status === 'STARTED' || task.status === 'PENDING'
                                                            ? 'text-blue-600'
                                                            : 'text-yellow-600'
                                                }`}
                                            >
                                                {(task.status === 'STARTED' || task.status === 'PENDING') &&
                                                    (task.task_type === 'create_deck_with_slides' ? 'Ingesting data and generating charts...' : 'Running...')}
                                                {task.status === 'SUCCESS' && 'Complete!'}
                                                {task.status === 'FAILURE' && 'Failed'}
                                                {!task.status && 'Pending...'}
                                            </span>
                                        </div>
                                        {task.status === 'SUCCESS' &&
                                        task.task_type === 'create_deck_with_slides' &&
                                        task.result &&
                                        typeof task.result === 'object' &&
                                        'presentationUrl' in task.result ? (
                                            <div className="mt-4">
                                                <Button
                                                    onClick={() => window.open((task.result as { presentationUrl: string }).presentationUrl, '_blank')}
                                                    className="w-full"
                                                >
                                                    <ExternalLink className="mr-2 h-4 w-4" />
                                                    View Slides
                                                </Button>
                                            </div>
                                        ) : null}
                                        {task.result !== null && task.task_type !== 'create_deck_with_slides' && (
                                            <div className="mt-2 text-sm text-green-700 dark:text-green-400">Result: {JSON.stringify(task.result)}</div>
                                        )}
                                        {task.error_message && <div className="mt-2 text-sm text-red-600 dark:text-red-400">Error: {task.error_message}</div>}
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                )}

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
