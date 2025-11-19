import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Calendar, FileText, Plus } from 'lucide-react'
import Link from 'next/link'

export default function Dashboard() {
    return (
        <div className="min-h-screen p-8">
            <div className="mx-auto max-w-7xl">
                {/* Header */}
                <div className="mb-8 flex items-center justify-between">
                    <div>
                        <h1 className="mb-2 text-3xl font-bold">Dashboard</h1>
                    </div>
                    <Link href="/create-deck">
                        <Button>
                            <Plus className="mr-2 h-4 w-4" />
                            Create Deck
                        </Button>
                    </Link>
                </div>
                {/* Decks Preview */}
                <div>
                    <h2 className="mb-6 text-2xl font-semibold">Your Decks</h2>
                    <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
                        {/* Deck Card 1 */}
                        <Card className="cursor-pointer transition-shadow hover:shadow-lg">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <CardTitle>Q4 Business Review</CardTitle>
                                        <CardDescription className="mt-2">Quarterly performance overview and strategic planning presentation</CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                        <div className="flex items-center gap-1">
                                            <FileText className="h-4 w-4" />
                                            <span>12 slides</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-4 w-4" />
                                            <span>Jan 15, 2024</span>
                                        </div>
                                    </div>
                                    <Badge variant="secondary">Business</Badge>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Deck Card 2 */}
                        <Card className="cursor-pointer transition-shadow hover:shadow-lg">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <CardTitle>Product Launch 2024</CardTitle>
                                        <CardDescription className="mt-2">New product features and marketing strategy presentation</CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                        <div className="flex items-center gap-1">
                                            <FileText className="h-4 w-4" />
                                            <span>8 slides</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-4 w-4" />
                                            <span>Jan 12, 2024</span>
                                        </div>
                                    </div>
                                    <Badge variant="secondary">Marketing</Badge>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Deck Card 3 */}
                        <Card className="cursor-pointer transition-shadow hover:shadow-lg">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <CardTitle>Team Training Workshop</CardTitle>
                                        <CardDescription className="mt-2">Onboarding materials and training resources for new team members</CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                        <div className="flex items-center gap-1">
                                            <FileText className="h-4 w-4" />
                                            <span>15 slides</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-4 w-4" />
                                            <span>Jan 10, 2024</span>
                                        </div>
                                    </div>
                                    <Badge variant="secondary">Education</Badge>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Deck Card 4 */}
                        <Card className="cursor-pointer transition-shadow hover:shadow-lg">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <CardTitle>Technology Roadmap</CardTitle>
                                        <CardDescription className="mt-2">Future technology initiatives and development timeline</CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                        <div className="flex items-center gap-1">
                                            <FileText className="h-4 w-4" />
                                            <span>20 slides</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-4 w-4" />
                                            <span>Jan 8, 2024</span>
                                        </div>
                                    </div>
                                    <Badge variant="secondary">Technology</Badge>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Deck Card 5 */}
                        <Card className="cursor-pointer transition-shadow hover:shadow-lg">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <CardTitle>Sales Performance</CardTitle>
                                        <CardDescription className="mt-2">Monthly sales metrics and revenue analysis dashboard</CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                        <div className="flex items-center gap-1">
                                            <FileText className="h-4 w-4" />
                                            <span>6 slides</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-4 w-4" />
                                            <span>Jan 5, 2024</span>
                                        </div>
                                    </div>
                                    <Badge variant="secondary">Business</Badge>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Deck Card 6 */}
                        <Card className="cursor-pointer transition-shadow hover:shadow-lg">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <CardTitle>Design System Overview</CardTitle>
                                        <CardDescription className="mt-2">UI/UX design guidelines and component library documentation</CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                        <div className="flex items-center gap-1">
                                            <FileText className="h-4 w-4" />
                                            <span>10 slides</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-4 w-4" />
                                            <span>Jan 3, 2024</span>
                                        </div>
                                    </div>
                                    <Badge variant="secondary">Design</Badge>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    )
}
