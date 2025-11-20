'use client'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { MultiSelect } from '@/components/ui/multi-select'
import { Select } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { ArrowLeft } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { toast } from 'sonner'

export default function CreateSlide() {
    const router = useRouter()
    const [formData, setFormData] = useState({
        partnerName: '',
        districtNames: [] as string[],
        schools: [] as string[],
        grades: [] as string[],
        years: [] as string[],
        subjects: [] as string[],
        disadvantages: [] as string[],
        race: [] as string[],
        assessments: [] as string[],
        slidePrompt: ''
    })

    // Partner configuration - maps partner_name to their districts and schools
    const partnerConfig: Record<string, { districts: string[]; schools: Record<string, string[]> }> = {
        california_pacific: {
            districts: [
                'California Pacific Charter Schools',
                'California Pacific Charter - San Diego',
                'California Pacific Charter - Sonoma',
                'California Pacific Charter- Los Angeles'
            ],
            schools: {
                'California Pacific Charter - San Diego': ['San Diego'],
                'California Pacific Charter - Sonoma': ['Sonoma'],
                'California Pacific Charter- Los Angeles': ['Los Angeles'],
                'California Pacific Charter - Los Angeles': ['Los Angeles']
            }
        }
        // Add more partners as needed
    }

    const partnerOptions = [
        { value: 'california_pacific', label: 'California Pacific Charter Schools' }
        // Add more partners as needed
    ]

    const getDistrictOptions = () => {
        if (!formData.partnerName || !partnerConfig[formData.partnerName]) {
            return []
        }
        return partnerConfig[formData.partnerName].districts
    }

    const getSchoolOptions = () => {
        if (!formData.partnerName || !partnerConfig[formData.partnerName]) {
            return []
        }
        const selectedDistricts = formData.districtNames
        const allSchools: string[] = []

        selectedDistricts.forEach((district) => {
            const schools = partnerConfig[formData.partnerName].schools[district] || []
            allSchools.push(...schools)
        })

        // Remove duplicates
        return Array.from(new Set(allSchools))
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

    const handlePartnerChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const partnerName = e.target.value
        setFormData((prev) => ({
            ...prev,
            partnerName,
            districtNames: [], // Reset districts when partner changes
            schools: [] // Reset schools when partner changes
        }))
    }

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        // Handle form submission here
        console.log('Form submitted:', formData)

        // Show toast notification
        toast.success('Deck is being created...')

        // Navigate to dashboard after a short delay
        setTimeout(() => {
            router.push('/dashboard')
        }, 500)
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
                            <CardTitle>Partner and District Setup</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="partnerName">
                                    Partner Name <span className="text-destructive">*</span>
                                </Label>
                                <Select id="partnerName" name="partnerName" value={formData.partnerName} onChange={handlePartnerChange} required>
                                    <option value="">Select partner...</option>
                                    {partnerOptions.map((partner) => (
                                        <option key={partner.value} value={partner.value}>
                                            {partner.label}
                                        </option>
                                    ))}
                                </Select>
                                <p className="text-xs text-muted-foreground">
                                    Matches the BigQuery dataset name: parsecgo.client_{formData.partnerName || '{partner_name}'}
                                </p>
                            </div>

                            <div className="space-y-2">
                                <Label>
                                    District Name(s) <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={getDistrictOptions()}
                                    selected={formData.districtNames}
                                    onChange={(selected) => {
                                        setFormData((prev) => ({
                                            ...prev,
                                            districtNames: selected,
                                            schools: [] // Reset schools when districts change
                                        }))
                                    }}
                                    placeholder="Select district(s)..."
                                    disabled={!formData.partnerName}
                                />
                                <p className="text-xs text-muted-foreground">
                                    List all LEAs under the charter organization. The first listed value should be the charter organization name.
                                </p>
                            </div>

                            <div className="space-y-2">
                                <Label>
                                    School(s) <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={getSchoolOptions()}
                                    selected={formData.schools}
                                    onChange={(selected) => setFormData((prev) => ({ ...prev, schools: selected }))}
                                    placeholder="Select school(s)..."
                                    disabled={!formData.partnerName || formData.districtNames.length === 0}
                                />
                                <p className="text-xs text-muted-foreground">School names are mapped from district variations to unified names.</p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>School Information</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
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
                                        partnerName: '',
                                        districtNames: [],
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
