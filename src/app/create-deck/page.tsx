'use client'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { MultiSelect } from '@/components/ui/multi-select'
import { Textarea } from '@/components/ui/textarea'
import { ArrowLeft } from 'lucide-react'
import { useState } from 'react'

export default function CreateSlide() {
    const [formData, setFormData] = useState({
        schoolDistricts: [] as string[],
        schools: [] as string[],
        grades: [] as string[],
        years: [] as string[],
        subjects: [] as string[],
        disadvantages: [] as string[],
        race: [] as string[],
        assessments: [] as string[],
        slidePrompt: ''
    })

    const handleSelectChange = (name: string, value: string) => {
        setFormData((prev) => {
            const currentArray = (prev[name as keyof typeof prev] as string[]) || []
            if (currentArray.includes(value)) {
                return {
                    ...prev,
                    [name]: currentArray.filter((item) => item !== value)
                }
            } else {
                return {
                    ...prev,
                    [name]: [...currentArray, value]
                }
            }
        })
    }

    const handleCheckboxChange = (name: string, value: string, checked: boolean) => {
        setFormData((prev) => {
            const currentArray = (prev[name as keyof typeof prev] as string[]) || []
            if (checked) {
                return {
                    ...prev,
                    [name]: [...currentArray, value]
                }
            } else {
                return {
                    ...prev,
                    [name]: currentArray.filter((item) => item !== value)
                }
            }
        })
    }

    const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setFormData((prev) => ({
            ...prev,
            slidePrompt: e.target.value
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
                    <h1 className="mb-2 text-3xl font-bold">Create New Slide Deck</h1>
                    <p className="text-muted-foreground">Fill out the information below to create your slide deck</p>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>School Information</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label>
                                    School District(s) <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={['District A', 'District B', 'District C', 'District D', 'District E', 'District F']}
                                    selected={formData.schoolDistricts}
                                    onChange={(selected) => setFormData((prev) => ({ ...prev, schoolDistricts: selected }))}
                                    placeholder="Select school district(s)..."
                                />
                            </div>

                            <div className="space-y-2">
                                <Label>
                                    School(s) <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={[
                                        'Lincoln Elementary',
                                        'Washington Middle',
                                        'Jefferson High',
                                        'Roosevelt Elementary',
                                        'Adams Middle',
                                        'Madison High',
                                        'Monroe Elementary'
                                    ]}
                                    selected={formData.schools}
                                    onChange={(selected) => setFormData((prev) => ({ ...prev, schools: selected }))}
                                    placeholder="Select school(s)..."
                                />
                            </div>

                            <div className="space-y-2">
                                <Label>
                                    Grade(s) <span className="text-destructive">*</span>
                                </Label>
                                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                                    {['K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'].map((grade) => (
                                        <div key={grade} className="flex items-center space-x-2">
                                            <Checkbox
                                                id={`grade-${grade}`}
                                                checked={formData.grades.includes(grade)}
                                                onChange={(e) => handleCheckboxChange('grades', grade, e.target.checked)}
                                            />
                                            <Label htmlFor={`grade-${grade}`} className="cursor-pointer font-normal">
                                                Grade {grade}
                                            </Label>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label>
                                    Year(s) <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']}
                                    selected={formData.years}
                                    onChange={(selected) => setFormData((prev) => ({ ...prev, years: selected }))}
                                    placeholder="Select year(s)..."
                                />
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Subjects & Demographics</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label>
                                    Subject(s) <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={[
                                        'Mathematics',
                                        'English Language Arts',
                                        'Science',
                                        'Social Studies',
                                        'Reading',
                                        'Writing',
                                        'History',
                                        'Geography',
                                        'Biology',
                                        'Chemistry',
                                        'Physics',
                                        'Algebra',
                                        'Geometry'
                                    ]}
                                    selected={formData.subjects}
                                    onChange={(selected) => setFormData((prev) => ({ ...prev, subjects: selected }))}
                                    placeholder="Select subject(s)..."
                                />
                            </div>

                            <div className="space-y-2">
                                <Label>
                                    Specific Disadvantages <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={[
                                        'Socioeconomically Disadvantaged',
                                        'Students with Disabilities',
                                        'English Learners',
                                        'Foster Youth',
                                        'Homeless',
                                        'Migrant',
                                        'Military Connected'
                                    ]}
                                    selected={formData.disadvantages}
                                    onChange={(selected) => setFormData((prev) => ({ ...prev, disadvantages: selected }))}
                                    placeholder="Select specific disadvantage(s)..."
                                />
                            </div>

                            <div className="space-y-2">
                                <Label>
                                    Race/Ethnicity <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={[
                                        'American Indian or Alaska Native',
                                        'Asian',
                                        'Black or African American',
                                        'Hispanic or Latino',
                                        'Native Hawaiian or Other Pacific Islander',
                                        'White',
                                        'Two or More Races',
                                        'Not Reported'
                                    ]}
                                    selected={formData.race}
                                    onChange={(selected) => setFormData((prev) => ({ ...prev, race: selected }))}
                                    placeholder="Select race/ethnicity..."
                                />
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Assessment Types</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-3">
                                <Label>
                                    Select Assessment Type(s) <span className="text-destructive">*</span>
                                </Label>
                                <div className="space-y-3">
                                    {['NWEA', 'STAR', 'iReady', 'CAASPP'].map((assessment) => (
                                        <div key={assessment} className="flex items-center space-x-2">
                                            <Checkbox
                                                id={`assessment-${assessment}`}
                                                checked={formData.assessments.includes(assessment)}
                                                onChange={(e) => handleCheckboxChange('assessments', assessment, e.target.checked)}
                                            />
                                            <Label htmlFor={`assessment-${assessment}`} className="cursor-pointer text-base font-normal">
                                                {assessment}
                                            </Label>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Slide Information</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="slidePrompt">
                                    Describe the slide information you want <span className="text-destructive">*</span>
                                </Label>
                                <Textarea
                                    id="slidePrompt"
                                    name="slidePrompt"
                                    value={formData.slidePrompt}
                                    onChange={handleTextareaChange}
                                    required
                                    rows={6}
                                    placeholder="Describe what specific type of slide information you would like to see in the slide deck. This will pull information from the selected districts, schools, grades, and assessments..."
                                />
                                <p className="text-xs text-muted-foreground">
                                    Provide details about the type of data, visualizations, or information you want included in your slide deck. The system will
                                    pull relevant information based on your selections above.
                                </p>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Form Actions */}
                    <div className="flex items-center justify-between gap-4">
                        <Button variant="outline" onClick={() => (window.location.href = '/dashboard')}>
                            <ArrowLeft className="mr-2 h-4 w-4" />
                            Back
                        </Button>
                        <div className="flex items-center justify-between gap-4">
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => {
                                    setFormData({
                                        schoolDistricts: [],
                                        schools: [],
                                        grades: [],
                                        years: [],
                                        subjects: [],
                                        disadvantages: [],
                                        race: [],
                                        assessments: [],
                                        slidePrompt: ''
                                    })
                                }}
                            >
                                Reset
                            </Button>
                            <Button type="submit">Create Slide Deck</Button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    )
}
