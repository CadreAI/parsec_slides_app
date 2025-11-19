'use client'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { useState } from 'react'

export default function CreateSlide() {
    const [formData, setFormData] = useState({
        title: '',
        description: '',
        slideType: '',
        category: '',
        priority: 'medium',
        tags: '',
        duration: '',
        targetAudience: '',
        notes: ''
    })

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { name, value } = e.target
        setFormData((prev) => ({
            ...prev,
            [name]: value
        }))
    }

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        // Handle form submission here
        console.log('Form submitted:', formData)
        alert('Slide information submitted!')
    }

    return (
        <div className="min-h-screen p-8">
            <div className="mx-auto max-w-3xl">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="mb-2 text-3xl font-bold">Create New Slide</h1>
                    <p className="text-muted-foreground">Fill out the information below to create your slide</p>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Basic Information</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="title">
                                    Slide Title <span className="text-destructive">*</span>
                                </Label>
                                <Input id="title" name="title" value={formData.title} onChange={handleChange} required placeholder="Enter slide title" />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="description">
                                    Description <span className="text-destructive">*</span>
                                </Label>
                                <Textarea
                                    id="description"
                                    name="description"
                                    value={formData.description}
                                    onChange={handleChange}
                                    required
                                    rows={4}
                                    placeholder="Describe the content and purpose of this slide"
                                />
                            </div>

                            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                                <div className="space-y-2">
                                    <Label htmlFor="slideType">
                                        Slide Type <span className="text-destructive">*</span>
                                    </Label>
                                    <Select id="slideType" name="slideType" value={formData.slideType} onChange={handleChange} required>
                                        <option value="">Select type</option>
                                        <option value="presentation">Presentation</option>
                                        <option value="infographic">Infographic</option>
                                        <option value="chart">Chart</option>
                                        <option value="text">Text Only</option>
                                        <option value="image">Image Focus</option>
                                        <option value="video">Video</option>
                                    </Select>
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="category">Category</Label>
                                    <Select id="category" name="category" value={formData.category} onChange={handleChange}>
                                        <option value="">Select category</option>
                                        <option value="business">Business</option>
                                        <option value="education">Education</option>
                                        <option value="marketing">Marketing</option>
                                        <option value="technology">Technology</option>
                                        <option value="design">Design</option>
                                        <option value="other">Other</option>
                                    </Select>
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Additional Details</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="priority">Priority</Label>
                                <Select id="priority" name="priority" value={formData.priority} onChange={handleChange}>
                                    <option value="low">Low</option>
                                    <option value="medium">Medium</option>
                                    <option value="high">High</option>
                                    <option value="urgent">Urgent</option>
                                </Select>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="tags">Tags</Label>
                                <Input id="tags" name="tags" value={formData.tags} onChange={handleChange} placeholder="Enter tags separated by commas" />
                                <p className="text-xs text-muted-foreground">Separate multiple tags with commas</p>
                            </div>

                            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                                <div className="space-y-2">
                                    <Label htmlFor="duration">Duration (minutes)</Label>
                                    <Input
                                        type="number"
                                        id="duration"
                                        name="duration"
                                        value={formData.duration}
                                        onChange={handleChange}
                                        min="1"
                                        placeholder="e.g., 5"
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="targetAudience">Target Audience</Label>
                                    <Input
                                        type="text"
                                        id="targetAudience"
                                        name="targetAudience"
                                        value={formData.targetAudience}
                                        onChange={handleChange}
                                        placeholder="e.g., Executives, Students"
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="notes">Additional Notes</Label>
                                <Textarea
                                    id="notes"
                                    name="notes"
                                    value={formData.notes}
                                    onChange={handleChange}
                                    rows={3}
                                    placeholder="Any additional information or requirements..."
                                />
                            </div>
                        </CardContent>
                    </Card>

                    {/* Form Actions */}
                    <div className="flex items-center justify-end gap-4">
                        <Button
                            type="button"
                            variant="outline"
                            onClick={() => {
                                setFormData({
                                    title: '',
                                    description: '',
                                    slideType: '',
                                    category: '',
                                    priority: 'medium',
                                    tags: '',
                                    duration: '',
                                    targetAudience: '',
                                    notes: ''
                                })
                            }}
                        >
                            Reset
                        </Button>
                        <Button type="submit">Create Slide</Button>
                    </div>
                </form>
            </div>
        </div>
    )
}
