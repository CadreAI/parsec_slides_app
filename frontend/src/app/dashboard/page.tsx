'use client'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Calendar, CheckCircle2, FileText, Loader2, Plus, XCircle } from 'lucide-react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'

interface DeckJob {
    id: string
    title: string
    status: 'processing' | 'completed' | 'failed'
    startTime: number
    estimatedDuration: number
    step?: string
    progress?: number
    presentationId?: string
    presentationUrl?: string
    error?: string
    completedTime?: number
}

export default function Dashboard() {
    const searchParams = useSearchParams()
    const jobId = searchParams.get('job')
    const [activeJob, setActiveJob] = useState<DeckJob | null>(null)
    const [elapsedTime, setElapsedTime] = useState(0)
    const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState<number | null>(null)

    useEffect(() => {
        // Check for active job in localStorage
        const checkActiveJob = () => {
            try {
                const jobData = localStorage.getItem('activeDeckJob')
                if (jobData) {
                    const job: DeckJob = JSON.parse(jobData)
                    setActiveJob(job)

                    // Calculate elapsed time
                    const elapsed = Date.now() - job.startTime
                    setElapsedTime(Math.floor(elapsed / 1000))

                    // Calculate estimated time remaining
                    if (job.status === 'processing' && job.estimatedDuration) {
                        const remaining = Math.max(0, job.estimatedDuration - elapsed)
                        setEstimatedTimeRemaining(Math.floor(remaining / 1000))
                    } else {
                        setEstimatedTimeRemaining(null)
                    }

                    // If completed, show success notification
                    if (job.status === 'completed' && !toast.isActive(`job-${job.id}`)) {
                        toast.success(`✅ Deck "${job.title}" created successfully!`, {
                            id: `job-${job.id}`,
                            duration: 5000
                        })
                    }

                    // If failed, show error notification
                    if (job.status === 'failed' && !toast.isActive(`job-${job.id}`)) {
                        toast.error(`Failed to create deck: ${job.error || 'Unknown error'}`, {
                            id: `job-${job.id}`,
                            duration: 5000
                        })
                    }
                } else {
                    setActiveJob(null)
                }
            } catch (error) {
                console.error('Error reading job data:', error)
            }
        }

        // Check immediately
        checkActiveJob()

        // Poll for updates every second
        const interval = setInterval(() => {
            checkActiveJob()
            if (activeJob?.status === 'processing') {
                const elapsed = Date.now() - activeJob.startTime
                setElapsedTime(Math.floor(elapsed / 1000))
                if (activeJob.estimatedDuration) {
                    const remaining = Math.max(0, activeJob.estimatedDuration - elapsed)
                    setEstimatedTimeRemaining(Math.floor(remaining / 1000))
                }
            }
        }, 1000)

        return () => clearInterval(interval)
    }, [activeJob?.status, activeJob?.startTime, activeJob?.estimatedDuration, activeJob?.id])

    const formatTime = (seconds: number): string => {
        if (seconds < 60) return `${seconds}s`
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}m ${secs}s`
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
                    <Link href="/create-deck">
                        <Button>
                            <Plus className="mr-2 h-4 w-4" />
                            Create Deck
                        </Button>
                    </Link>
                </div>

                {/* Active Job Loading Card */}
                {activeJob && (
                    <Card className="mb-8 border-blue-200 bg-blue-50">
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    {activeJob.status === 'processing' && <Loader2 className="h-5 w-5 animate-spin text-blue-600" />}
                                    {activeJob.status === 'completed' && <CheckCircle2 className="h-5 w-5 text-green-600" />}
                                    {activeJob.status === 'failed' && <XCircle className="h-5 w-5 text-red-600" />}
                                    <CardTitle className="text-xl">
                                        {activeJob.status === 'processing' && 'Creating Slide Deck...'}
                                        {activeJob.status === 'completed' && 'Deck Created Successfully!'}
                                        {activeJob.status === 'failed' && 'Deck Creation Failed'}
                                    </CardTitle>
                                </div>
                                {activeJob.status === 'processing' && <Badge variant="secondary">{formatTime(elapsedTime)} elapsed</Badge>}
                            </div>
                            <CardDescription className="mt-2">{activeJob.title}</CardDescription>
                        </CardHeader>
                        <CardContent>
                            {activeJob.status === 'processing' && (
                                <div className="space-y-4">
                                    <div>
                                        <div className="mb-2 flex items-center justify-between text-sm">
                                            <span className="text-muted-foreground">{activeJob.step || 'Processing...'}</span>
                                            <span className="text-muted-foreground">
                                                {estimatedTimeRemaining !== null ? `~${formatTime(estimatedTimeRemaining)} remaining` : 'Calculating...'}
                                            </span>
                                        </div>
                                        <Progress value={activeJob.progress || 0} className="h-2" />
                                    </div>
                                    <div className="text-muted-foreground text-xs">This may take a few minutes depending on the number of charts...</div>
                                </div>
                            )}
                            {activeJob.status === 'completed' && activeJob.presentationUrl && (
                                <div className="space-y-2">
                                    <p className="text-muted-foreground text-sm">Your presentation is ready!</p>
                                    <Button asChild className="w-full">
                                        <a href={activeJob.presentationUrl} target="_blank" rel="noopener noreferrer">
                                            Open Presentation
                                        </a>
                                    </Button>
                                </div>
                            )}
                            {activeJob.status === 'failed' && (
                                <div className="space-y-2">
                                    <p className="text-destructive text-sm">{activeJob.error || 'An error occurred'}</p>
                                    <Button asChild variant="outline">
                                        <Link href="/create-deck">Try Again</Link>
                                    </Button>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                )}

                {/* Decks Preview */}
                <div>
                    <h2 className="mb-6 text-2xl font-semibold">Your Decks</h2>
                    <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
                        {/* BOY Deck Card */}
                        <Card className="cursor-pointer transition-shadow hover:shadow-lg">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <CardTitle>BOY</CardTitle>
                                        <CardDescription className="mt-2">Beginning of Year assessment presentation and performance overview</CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="text-muted-foreground flex items-center gap-4 text-sm">
                                        <div className="flex items-center gap-1">
                                            <FileText className="h-4 w-4" />
                                            <span>15 slides</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-4 w-4" />
                                            <span>Fall 2024</span>
                                        </div>
                                    </div>
                                    <Badge variant="secondary">Education</Badge>
                                </div>
                            </CardContent>
                        </Card>

                        {/* MOY Deck Card */}
                        <Card className="cursor-pointer transition-shadow hover:shadow-lg">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <CardTitle>MOY</CardTitle>
                                        <CardDescription className="mt-2">
                                            Middle of Year assessment presentation and mid-year performance analysis
                                        </CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="text-muted-foreground flex items-center gap-4 text-sm">
                                        <div className="flex items-center gap-1">
                                            <FileText className="h-4 w-4" />
                                            <span>18 slides</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-4 w-4" />
                                            <span>Winter 2024</span>
                                        </div>
                                    </div>
                                    <Badge variant="secondary">Education</Badge>
                                </div>
                            </CardContent>
                        </Card>

                        {/* EOY Deck Card */}
                        <Card className="cursor-pointer transition-shadow hover:shadow-lg">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <CardTitle>EOY</CardTitle>
                                        <CardDescription className="mt-2">End of Year assessment presentation and annual performance review</CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="text-muted-foreground flex items-center gap-4 text-sm">
                                        <div className="flex items-center gap-1">
                                            <FileText className="h-4 w-4" />
                                            <span>20 slides</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-4 w-4" />
                                            <span>Spring 2024</span>
                                        </div>
                                    </div>
                                    <Badge variant="secondary">Education</Badge>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    )
}
