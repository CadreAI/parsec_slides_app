'use client'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { MultiSelect } from '@/components/ui/multi-select'
import { Select } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { ArrowLeft } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { toast } from 'sonner'

// Available assessment sources for data ingestion
const ASSESSMENT_SOURCES = [
    { id: 'calpads', label: 'CALPADS', defaultTable: 'parsecgo.demodashboard.calpads' },
    { id: 'nwea', label: 'NWEA', defaultTable: 'parsecgo.demodashboard.Nwea_production_calpads_v4_2' },
    { id: 'iready', label: 'iReady', defaultTable: 'parsecgo.demodashboard.iready_production_calpads_v4_2' },
    { id: 'star', label: 'STAR', defaultTable: 'parsecgo.demodashboard.renaissance_production_calpads_v4_2' },
    { id: 'cers', label: 'CERS', defaultTable: 'parsecgo.demodashboard.cers_production' },
    { id: 'iab', label: 'IAB', defaultTable: 'parsecgo.demodashboard.cers_iab' }
]

export default function CreateSlide() {
    const router = useRouter()
    const [isIngesting, setIsIngesting] = useState(false)
    const [isCreating, setIsCreating] = useState(false)
    const [formData, setFormData] = useState({
        // Partner & Data Configuration
        partnerName: '',
        projectId: 'parsecgo',
        location: 'US',
        selectedDataSources: [] as string[],
        customDataSources: {} as Record<string, string>,
        // Slide Configuration
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

    // Student group mappings for filtering (used in config)
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
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

    // Student group order for consistent sorting (used in config)
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
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
        demodashboard: {
            districts: ['Parsec Academy'],
            schools: {
                'Parsec Academy': ['Parsec Academy']
            }
        }
    }

    const partnerOptions = [{ value: 'demodashboard', label: 'Parsec Academy' }]

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
        const partner = e.target.value
        setFormData((prev) => ({
            ...prev,
            partnerName: partner,
            districtNames: [],
            schools: []
        }))
    }

    const handleDataSourceToggle = (sourceId: string) => {
        setFormData((prev) => ({
            ...prev,
            selectedDataSources: prev.selectedDataSources.includes(sourceId)
                ? prev.selectedDataSources.filter((id) => id !== sourceId)
                : [...prev.selectedDataSources, sourceId]
        }))
    }

    const handleCustomDataSourceChange = (sourceId: string, value: string) => {
        setFormData((prev) => ({
            ...prev,
            customDataSources: {
                ...prev.customDataSources,
                [sourceId]: value
            }
        }))
    }

    // Combined: Ingest data, generate charts, then create slide deck
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()

        if (!formData.partnerName.trim()) {
            toast.error('Please enter a partner name')
            return
        }

        if (formData.selectedDataSources.length === 0) {
            toast.error('Please select at least one data source/assessment')
            return
        }

        // Sync assessments with selected data sources
        if (formData.assessments.length === 0) {
            // Auto-select assessments based on data sources
            formData.assessments = [...formData.selectedDataSources]
        }

        setIsCreating(true)
        setIsIngesting(true)

        try {
            // Step 1: Ingest data and generate charts
            toast.info('Step 1/2: Ingesting data and generating charts...')

            // Build sources object
            const sources: Record<string, string> = {}
            formData.selectedDataSources.forEach((sourceId) => {
                const customTable = formData.customDataSources[sourceId]
                const defaultSource = ASSESSMENT_SOURCES.find((s) => s.id === sourceId)
                sources[sourceId] = customTable || defaultSource?.defaultTable || ''
            })

            // Use selected districts from scope selection
            const districtList = formData.districtNames.length > 0 ? formData.districtNames : ['Parsec Academy']

            // Build config object
            const config = {
                partner_name: formData.partnerName || 'demodashboard',
                district_name: districtList,
                school_name_map: {
                    'Parsec Academy': 'Parsec Academy',
                    '': 'No assigned program'
                },
                gcp: {
                    project_id: formData.projectId,
                    location: formData.location
                },
                sources: sources,
                exclude_cols: {},
                student_groups: {
                    'All Students': { type: 'all' },
                    'English Learners': { column: 'englishlearner', in: ['Y', 'Yes', 'True', 1] },
                    'Students with Disabilities': { column: 'studentswithdisabilities', in: ['Y', 'Yes', 'True', 1] },
                    'Socioeconomically Disadvantaged': { column: 'socioeconomicallydisadvantaged', in: ['Y', 'Yes', 'True', 1] },
                    'Hispanic or Latino': { column: 'ethnicityrace', in: ['Hispanic', 'Hispanic or Latino'] },
                    White: { column: 'ethnicityrace', in: ['White'] },
                    'Black or African American': { column: 'ethnicityrace', in: ['Black', 'African American', 'Black or African American'] },
                    Asian: { column: 'ethnicityrace', in: ['Asian'] },
                    Filipino: { column: 'ethnicityrace', in: ['Filipino'] },
                    'American Indian or Alaska Native': {
                        column: 'ethnicityrace',
                        in: ['American Indian', 'Alaska Native', 'American Indian or Alaska Native']
                    },
                    'Native Hawaiian or Pacific Islander': {
                        column: 'ethnicityrace',
                        in: ['Pacific Islander', 'Native Hawaiian', 'Native Hawaiian or Other Pacific Islander']
                    },
                    'Two or More Races': { column: 'ethnicityrace', in: ['Two or More Races', 'Multiracial', 'Multiple Races'] },
                    'Not Stated': { column: 'ethnicityrace', in: ['Not Stated', 'Unknown', ''] },
                    Foster: { column: 'foster', in: ['Y', 'Yes', 'True', 1] },
                    Homeless: { column: 'homeless', in: ['Y', 'Yes', 'True', 1] }
                },
                student_group_order: {
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
                },
                options: {
                    cache_csv: true,
                    preview: true
                },
                paths: {
                    data_dir: './data',
                    charts_dir: './charts',
                    config_dir: '.'
                },
                // Chart generation filters
                chart_filters: {
                    grades:
                        formData.grades.length > 0
                            ? formData.grades
                                  .map((g) => {
                                      if (g === 'K') return 0
                                      const parsed = parseInt(g)
                                      return isNaN(parsed) ? null : parsed
                                  })
                                  .filter((g) => g !== null)
                            : undefined,
                    years: formData.years.length > 0 ? formData.years.map((y) => parseInt(y)).filter((y) => !isNaN(y)) : undefined,
                    subjects: formData.subjects.length > 0 ? formData.subjects : undefined,
                    student_groups: formData.studentGroups.length > 0 ? formData.studentGroups : undefined,
                    race: formData.race.length > 0 ? formData.race : undefined
                },
                // Scope selection: only generate charts for selected schools/districts
                selected_schools: formData.schools.length > 0 ? formData.schools : [],
                include_district_scope: formData.districtNames.length > 0 // Include district scope if districts are selected
            }

            // Call the data ingestion API
            const ingestRes = await fetch('/api/data/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config })
            })

            const ingestData = await ingestRes.json()

            if (!ingestRes.ok) {
                throw new Error(ingestData.error || ingestData.details || 'Failed to ingest data')
            }

            const chartCount = ingestData.charts?.length || 0
            const charts = ingestData.charts || []

            toast.success(`Step 1 complete! Generated ${chartCount} charts.`)

            // Step 2: Create slide deck
            toast.info('Step 2/2: Creating slide deck...')

            const presentationTitle = `Slide Deck - ${formData.partnerName || 'Untitled'}`

            // Use selectedDataSources for assessments (they're now combined)
            const assessmentsToUse = formData.assessments.length > 0 ? formData.assessments : formData.selectedDataSources

            // Hardcoded Google Drive folder
            const driveFolderUrl = 'https://drive.google.com/drive/folders/1CUOM-Sz6ulyzD2mTREdcYoBXUJLrgngw'

            // Call the API route to create the presentation
            const res = await fetch('/api/slides/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: presentationTitle,
                    assessments: assessmentsToUse,
                    charts: charts.length > 0 ? charts : undefined,
                    driveFolderUrl: driveFolderUrl
                })
            })

            const data = await res.json()

            if (!res.ok) {
                const errorMsg = data.details ? `${data.error}: ${data.details}` : data.error || 'Failed to create presentation'
                console.error('API Error:', data)
                throw new Error(errorMsg)
            }

            console.log('Presentation created:', data.presentationId)
            toast.success(`✅ Complete! Presentation created. View it here: ${data.presentationUrl}`)

            setTimeout(() => {
                router.push('/dashboard')
            }, 2000)
        } catch (error: unknown) {
            console.error('Error:', error)
            const errorMessage = error instanceof Error ? error.message : 'Unknown error'
            toast.error(`Failed: ${errorMessage}`)
        } finally {
            setIsCreating(false)
            setIsIngesting(false)
        }
    }

    return (
        <div className="min-h-screen p-8">
            <div className="mx-auto max-w-4xl">
                {/* Header */}
                <div className="mb-8 flex items-center justify-between">
                    <div>
                        <h1 className="mb-2 text-3xl font-bold">Create Slide Deck</h1>
                        <p className="text-muted-foreground">Configure data ingestion and create your presentation</p>
                    </div>
                    <Button variant="ghost" onClick={() => router.back()}>
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back
                    </Button>
                </div>

                <form onSubmit={handleSubmit}>
                    {/* Unified Configuration */}
                    <Card className="mb-6">
                        <CardHeader>
                            <CardTitle>Configuration</CardTitle>
                            <CardDescription>Configure data ingestion and slide deck settings</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            {/* Basic Settings */}
                            <div className="space-y-4 border-b pb-4">
                                <h3 className="text-lg font-semibold">Basic Settings</h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="partnerName">
                                            Partner Name <span className="text-destructive">*</span>
                                        </Label>
                                        <Select id="partnerName" value={formData.partnerName} onChange={handlePartnerChange} required>
                                            <option value="">Select a partner...</option>
                                            {partnerOptions.map((partner) => (
                                                <option key={partner.value} value={partner.value}>
                                                    {partner.label}
                                                </option>
                                            ))}
                                        </Select>
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="projectId">GCP Project ID</Label>
                                        <Input
                                            id="projectId"
                                            value={formData.projectId}
                                            onChange={(e) => setFormData((prev) => ({ ...prev, projectId: e.target.value }))}
                                            placeholder="parsecgo"
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* Scope Selection */}
                            <div className="space-y-4 border-b pb-4">
                                <h3 className="text-lg font-semibold">Scope Selection</h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label>
                                            District(s) <span className="text-destructive">*</span>
                                        </Label>
                                        <MultiSelect
                                            options={getDistrictOptions()}
                                            selected={formData.districtNames}
                                            onChange={(selected) => {
                                                setFormData((prev) => ({
                                                    ...prev,
                                                    districtNames: selected,
                                                    schools: []
                                                }))
                                            }}
                                            placeholder="Select district(s)..."
                                            disabled={!formData.partnerName}
                                        />
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
                                    </div>
                                </div>
                            </div>

                            {/* Filters */}
                            <div className="space-y-4 border-b pb-4">
                                <h3 className="text-lg font-semibold">Filters</h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label>
                                            Grade(s) <span className="text-destructive">*</span>
                                        </Label>
                                        <MultiSelect
                                            options={['K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']}
                                            selected={formData.grades}
                                            onChange={(selected) => setFormData((prev) => ({ ...prev, grades: selected }))}
                                            placeholder="Select grade(s)..."
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>
                                            Year(s) <span className="text-destructive">*</span>
                                        </Label>
                                        <MultiSelect
                                            options={['2021', '2022', '2023', '2024', '2025']}
                                            selected={formData.years}
                                            onChange={(selected) => setFormData((prev) => ({ ...prev, years: selected }))}
                                            placeholder="Select year(s)..."
                                        />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <Label>
                                        Subject(s) <span className="text-destructive">*</span>
                                    </Label>
                                    <MultiSelect
                                        options={['Math', 'Reading', 'Science', 'Social Studies']}
                                        selected={formData.subjects}
                                        onChange={(selected) => setFormData((prev) => ({ ...prev, subjects: selected }))}
                                        placeholder="Select subject(s)..."
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label>Student Groups</Label>
                                        <MultiSelect
                                            options={['All Students', 'English Learners', 'Students with Disabilities', 'Socioeconomically Disadvantaged']}
                                            selected={formData.studentGroups}
                                            onChange={(selected) => setFormData((prev) => ({ ...prev, studentGroups: selected }))}
                                            placeholder="Select student group(s)..."
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>Race/Ethnicity</Label>
                                        <MultiSelect
                                            options={[
                                                'Hispanic or Latino',
                                                'White',
                                                'Black or African American',
                                                'Asian',
                                                'Filipino',
                                                'American Indian or Alaska Native',
                                                'Native Hawaiian or Pacific Islander',
                                                'Two or More Races'
                                            ]}
                                            selected={formData.race}
                                            onChange={(selected) => setFormData((prev) => ({ ...prev, race: selected }))}
                                            placeholder="Select race/ethnicity..."
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* Data Sources & Assessments */}
                            <div className="space-y-4 border-b pb-4">
                                <h3 className="text-lg font-semibold">Data Sources & Assessments</h3>
                                <div className="space-y-2">
                                    <Label>
                                        Select Data Sources/Assessments <span className="text-destructive">*</span>
                                    </Label>
                                    <p className="mb-2 text-xs text-muted-foreground">
                                        Selected sources will be used for data ingestion, chart generation, and slide content
                                    </p>
                                    <div className="grid grid-cols-2 gap-2 rounded-lg border p-4">
                                        {ASSESSMENT_SOURCES.map((source) => {
                                            const isSelected = formData.selectedDataSources.includes(source.id)
                                            const customTable = formData.customDataSources[source.id]
                                            const defaultTable = source.defaultTable

                                            return (
                                                <div key={source.id} className="space-y-1 rounded border p-2">
                                                    <div className="flex items-center space-x-2">
                                                        <Checkbox
                                                            id={`data-${source.id}`}
                                                            checked={isSelected}
                                                            onChange={(e) => {
                                                                handleDataSourceToggle(source.id)
                                                                // Also update assessments to match
                                                                handleCheckboxChange('assessments', source.id, e.target.checked)
                                                            }}
                                                        />
                                                        <Label htmlFor={`data-${source.id}`} className="cursor-pointer text-sm font-semibold">
                                                            {source.label}
                                                        </Label>
                                                    </div>
                                                    {isSelected && (
                                                        <Input
                                                            value={customTable || defaultTable}
                                                            onChange={(e) => handleCustomDataSourceChange(source.id, e.target.value)}
                                                            placeholder={defaultTable}
                                                            className="mt-1 ml-6 h-8 font-mono text-xs"
                                                        />
                                                    )}
                                                </div>
                                            )
                                        })}
                                    </div>
                                </div>
                            </div>

                            {/* Slide Content */}
                            <div className="space-y-4">
                                <h3 className="text-lg font-semibold">Slide Content</h3>
                                <div className="space-y-2">
                                    <Label htmlFor="slidePrompt">Slide Information (Optional)</Label>
                                    <Textarea
                                        id="slidePrompt"
                                        value={formData.slidePrompt}
                                        onChange={handleTextareaChange}
                                        placeholder="Enter any additional information for the slides..."
                                        rows={4}
                                    />
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Submit Button */}
                    <div className="flex justify-end gap-4">
                        <Button type="button" variant="outline" onClick={() => router.back()}>
                            Cancel
                        </Button>
                        <Button type="submit" disabled={isCreating || isIngesting || !formData.partnerName || formData.selectedDataSources.length === 0}>
                            {isIngesting ? 'Ingesting Data & Generating Charts...' : isCreating ? 'Creating Slide Deck...' : 'Create Slide Deck'}
                        </Button>
                    </div>
                </form>
            </div>
        </div>
    )
}
