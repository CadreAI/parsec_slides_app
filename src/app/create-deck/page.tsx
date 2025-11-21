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
        studentGroups: [] as string[],
        race: [] as string[],
        assessments: [] as string[],
        slidePrompt: ''
    })

    // Assessment BigQuery table mappings
    const assessmentTables: Record<string, (partnerName: string) => string> = {
        calpads: (partnerName: string) => `parsecgo.client_${partnerName}.calpads`,
        nwea: (partnerName: string) => `parsecgo.client_${partnerName}.Nwea_production_calpads_v4_2`,
        iready: () => `parsecgo.demodashboard.iready_production_calpads_v4_2`,
        star: (partnerName: string) => `parsecgo.client_${partnerName}.renaissance_production_calpads_v4_2`,
        cers: (partnerName: string) => `parsecgo.client_${partnerName}.cers_production`,
        iab: (partnerName: string) => `parsecgo.client_${partnerName}.cers_iab`
    }

    // Student group mappings for filtering
    const studentGroupMappings: Record<string, { type?: string; column?: string; in?: (string | number)[] }> = {
        'All Students': { type: 'all' },
        'English Learners': { column: 'englishlearner', in: ['Y', 'Yes', 'True', 1] },
        'Students with Disabilities': { column: 'studentswithdisabilities', in: ['Y', 'Yes', 'True', 1] },
        'Socioeconomically Disadvantaged': { column: 'socioeconomicallydisadvantaged', in: ['Y', 'Yes', 'True', 1] },
        'Hispanic or Latino': { column: 'ethnicityrace', in: ['Hispanic', 'Hispanic or Latino'] },
        White: { column: 'ethnicityrace', in: ['White'] },
        'Black or African American': { column: 'ethnicityrace', in: ['Black', 'African American', 'Black or African American'] },
        Asian: { column: 'ethnicityrace', in: ['Asian'] },
        Filipino: { column: 'ethnicityrace', in: ['Filipino'] },
        'American Indian or Alaska Native': { column: 'ethnicityrace', in: ['American Indian', 'Alaska Native', 'American Indian or Alaska Native'] },
        'Native Hawaiian or Pacific Islander': {
            column: 'ethnicityrace',
            in: ['Pacific Islander', 'Native Hawaiian', 'Native Hawaiian or Other Pacific Islander']
        },
        'Two or More Races': { column: 'ethnicityrace', in: ['Two or More Races', 'Multiracial', 'Multiple Races'] },
        'Not Stated': { column: 'ethnicityrace', in: ['Not Stated', 'Unknown', ''] },
        Foster: { column: 'foster', in: ['Y', 'Yes', 'True', 1] },
        Homeless: { column: 'homeless', in: ['Y', 'Yes', 'True', 1] }
    }

    // Student group order for consistent sorting
    const studentGroupOrder: Record<string, number> = {
        'All Students': 1,
        'English Learners': 2,
        'Students with Disabilities': 3,
        'Socioeconomically Disadvantaged': 4,
        'Hispanic or Latino': 5,
        White: 6,
        'Black or African American': 7,
        Asian: 8,
        Filipino: 9,
        'American Indian or Alaska Native': 10,
        'Native Hawaiian or Pacific Islander': 11,
        'Two or More Races': 12,
        'Not Stated': 13,
        Foster: 14,
        Homeless: 15
    }

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

        // Build assessment table names based on selected assessments and partner
        const assessmentTableNames: Record<string, string> = {}
        formData.assessments.forEach((assessment) => {
            if (assessmentTables[assessment]) {
                assessmentTableNames[assessment] = assessmentTables[assessment](formData.partnerName)
            }
        })

        // Build student group filters based on selected groups
        const studentGroupFilters: Record<string, { type?: string; column?: string; in?: (string | number)[] }> = {}
        formData.studentGroups.forEach((group) => {
            if (studentGroupMappings[group]) {
                studentGroupFilters[group] = studentGroupMappings[group]
            }
        })

        // Build race filters based on selected races
        const raceFilters: Record<string, { type?: string; column?: string; in?: (string | number)[] }> = {}
        formData.race.forEach((raceGroup) => {
            if (studentGroupMappings[raceGroup]) {
                raceFilters[raceGroup] = studentGroupMappings[raceGroup]
            }
        })

        // Combine all selected groups (student groups + race) and sort by order
        const allSelectedGroups = [...formData.studentGroups, ...formData.race]
        const sortedGroups = allSelectedGroups.sort((a, b) => {
            const orderA = studentGroupOrder[a] || 999
            const orderB = studentGroupOrder[b] || 999
            return orderA - orderB
        })

        // Combine all filters
        const allGroupFilters = { ...studentGroupFilters, ...raceFilters }

        // Prepare submission data with BigQuery table names and student group filters
        const submissionData = {
            ...formData,
            assessmentTables: assessmentTableNames,
            studentGroupFilters: allGroupFilters,
            studentGroups: sortedGroups
        }

        // Handle form submission here
        console.log('Form submitted:', submissionData)

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
                                    Student Groups <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={[
                                        'All Students',
                                        'English Learners',
                                        'Students with Disabilities',
                                        'Socioeconomically Disadvantaged',
                                        'Foster',
                                        'Homeless'
                                    ]}
                                    selected={formData.studentGroups}
                                    onChange={(selected) => setFormData((prev) => ({ ...prev, studentGroups: selected }))}
                                    placeholder="Select student group(s)..."
                                />
                                <p className="text-xs text-muted-foreground">Select one or more student groups to filter the data</p>
                            </div>

                            <div className="space-y-2">
                                <Label>
                                    Race/Ethnicity <span className="text-destructive">*</span>
                                </Label>
                                <MultiSelect
                                    options={[
                                        'Hispanic or Latino',
                                        'White',
                                        'Black or African American',
                                        'Asian',
                                        'Filipino',
                                        'American Indian or Alaska Native',
                                        'Native Hawaiian or Pacific Islander',
                                        'Two or More Races',
                                        'Not Stated'
                                    ]}
                                    selected={formData.race}
                                    onChange={(selected) => setFormData((prev) => ({ ...prev, race: selected }))}
                                    placeholder="Select race/ethnicity..."
                                />
                                <p className="text-xs text-muted-foreground">Select one or more race/ethnicity categories</p>
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
                                    {['calpads', 'nwea', 'iready', 'star', 'cers', 'iab'].map((assessment) => (
                                        <div key={assessment} className="flex items-center space-x-2">
                                            <Checkbox
                                                id={`assessment-${assessment}`}
                                                checked={formData.assessments.includes(assessment)}
                                                onChange={(e) => handleCheckboxChange('assessments', assessment, e.target.checked)}
                                            />
                                            <Label htmlFor={`assessment-${assessment}`} className="cursor-pointer text-base font-normal">
                                                {assessment.toUpperCase()}
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
                                        studentGroups: [],
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
