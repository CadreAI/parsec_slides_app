'use client'

import { SignOutButton } from '@clerk/nextjs'
import { Calendar, FileText, Plus } from 'lucide-react'
import Link from 'next/link'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export default function Dashboard() {
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
