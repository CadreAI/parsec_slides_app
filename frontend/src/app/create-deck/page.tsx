'use client'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { MultiSelect } from '@/components/ui/multi-select'
import { Progress } from '@/components/ui/progress'
import { Select } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { useAssessmentFilters } from '@/hooks/useAssessmentFilters'
import { useAssessmentTables } from '@/hooks/useAssessmentTables'
import { useAvailableAssessments } from '@/hooks/useAvailableAssessments'
import { useDatasets } from '@/hooks/useDatasets'
import { useDistrictsAndSchools } from '@/hooks/useDistrictsAndSchools'
import { useFormOptions } from '@/hooks/useFormOptions'
import { useStudentGroups } from '@/hooks/useStudentGroups'
import { getDistrictOptions, getSchoolOptions } from '@/utils/formHelpers'
import { ArrowLeft } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { toast } from 'sonner'

// Partner configuration - maps partner_name to their districts and schools
const PARTNER_CONFIG: Record<string, { districts: string[]; schools: Record<string, string[]> }> = {
    demodashboard: {
        districts: ['Parsec Academy'],
        schools: {
            'Parsec Academy': ['Parsec Academy']
        }
    }
}

export default function CreateSlide() {
    const router = useRouter()
    const [isIngesting, setIsIngesting] = useState(false)
    const [isCreating, setIsCreating] = useState(false)
    const [slideProgress, setSlideProgress] = useState({ value: 0, step: '' })
    const [formData, setFormData] = useState({
        // Partner & Data Configuration
        partnerName: '',
        projectId: 'parsecgo',
        location: 'US',
        selectedDataSources: [] as string[],
        customDataSources: {} as Record<string, string>,
        // Slide Configuration
        deckName: '',
        districtName: '',
        schools: [] as string[],
        grades: [] as string[],
        years: [] as string[],
        quarters: [] as string[],
        subjects: [] as string[],
        studentGroups: [] as string[],
        race: [] as string[],
        assessments: [] as string[],
        slidePrompt: ''
    })

    // Custom hooks for data fetching
    const { assessments: ASSESSMENT_SOURCES, isLoading: isLoadingAssessments } = useAvailableAssessments()
    const { grades: GRADES, years: YEARS } = useFormOptions(formData.projectId, formData.partnerName, formData.location)
    const { studentGroupOptions, raceOptions, studentGroupMappings, studentGroupOrder } = useStudentGroups()
    const { partnerOptions, isLoadingDatasets } = useDatasets(formData.projectId, formData.location)
    const { availableDistricts, availableSchools, districtSchoolMap, isLoadingDistrictsSchools } = useDistrictsAndSchools(
        formData.partnerName,
        formData.projectId,
        formData.location
    )
    const { availableAssessments, assessmentTables, isLoadingAssessmentTables } = useAssessmentTables(
        formData.partnerName,
        formData.projectId,
        formData.location,
        setFormData
    )
    const { availableSubjects, availableQuarters, supportsGrades, supportsStudentGroups, supportsRace, isLoadingFilters } = useAssessmentFilters(
        formData.assessments,
        formData.projectId,
        formData.partnerName,
        formData.location,
        setFormData
    )

    // Helper functions
    const districtOptions = getDistrictOptions(availableDistricts, formData.partnerName, PARTNER_CONFIG)
    const schoolOptions = getSchoolOptions(formData.districtName, districtSchoolMap, availableSchools, formData.partnerName, PARTNER_CONFIG)

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
            districtName: '',
            schools: [],
            quarters: []
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

        if (!formData.districtName.trim()) {
            toast.error('Please select a district')
            return
        }

        if (formData.assessments.length === 0) {
            toast.error('Please select at least one assessment first')
            return
        }

        if (formData.selectedDataSources.length === 0) {
            toast.error('Please select at least one data source/assessment')
            return
        }

        if (formData.years.length < 2) {
            toast.error('Please select at least 2 years (2023-2026)')
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
            const districtList = formData.districtName ? [formData.districtName] : ['Parsec Academy']

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
                student_groups: studentGroupMappings,
                student_group_order: studentGroupOrder,
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
                    quarters: formData.quarters.length > 0 ? formData.quarters : undefined,
                    subjects: formData.subjects.length > 0 ? formData.subjects : undefined,
                    student_groups: formData.studentGroups.length > 0 ? formData.studentGroups : undefined,
                    race: formData.race.length > 0 ? formData.race : undefined
                },
                // Scope selection: only generate charts for selected schools/districts
                selected_schools: formData.schools.length > 0 ? formData.schools : [],
                include_district_scope: !!formData.districtName // Include district scope if district is selected
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
            setSlideProgress({ value: 0, step: 'Initializing...' })

            const presentationTitle = formData.deckName.trim() || `Slide Deck - ${formData.partnerName || 'Untitled'}`

            // Use selectedDataSources for assessments (they're now combined)
            const assessmentsToUse = formData.assessments.length > 0 ? formData.assessments : formData.selectedDataSources

            // Hardcoded Google Drive folder
            const driveFolderUrl = 'https://drive.google.com/drive/folders/1CUOM-Sz6ulyzD2mTREdcYoBXUJLrgngw'

            // Estimate total steps for progress tracking
            const totalCharts = charts.length
            const estimatedSteps = Math.max(10, 5 + Math.ceil(totalCharts * 0.5)) // Base steps + chart processing
            let currentStep = 0

            const updateProgress = (step: string, increment: number = 1) => {
                currentStep += increment
                const progress = Math.min(Math.round((currentStep / estimatedSteps) * 100), 95) // Cap at 95% until complete
                setSlideProgress({ value: progress, step })
            }

            // Simulate progress updates during API call
            const progressInterval = setInterval(() => {
                if (currentStep < estimatedSteps - 1) {
                    // Gradually increase progress to show activity
                    const simulatedProgress = Math.min(currentStep + 0.3, estimatedSteps - 1)
                    const progress = Math.round((simulatedProgress / estimatedSteps) * 100)
                    setSlideProgress((prev) => ({
                        value: Math.max(prev.value, Math.min(progress, 95)),
                        step: prev.step || 'Processing...'
                    }))
                }
            }, 300)

            updateProgress('Creating presentation...', 1)
            setTimeout(() => updateProgress('Uploading charts to Drive...', 2), 500)
            setTimeout(() => updateProgress('Creating slides...', 2), 1000)
            setTimeout(() => updateProgress('Adding charts to slides...', 2), 1500)

            // Call the API route to create the presentation
            const res = await fetch('/api/slides/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: presentationTitle,
                    assessments: assessmentsToUse,
                    charts: charts.length > 0 ? charts : undefined,
                    driveFolderUrl: driveFolderUrl,
                    schoolName: 'Parsec Academy',
                    quarters: formData.quarters,
                    partnerName: formData.partnerName,
                    userPrompt: formData.slidePrompt || undefined, // User prompt for decision LLM
                    deckName: formData.deckName,
                    districtName: formData.districtName,
                    schools: formData.schools,
                    projectId: formData.projectId,
                    location: formData.location,
                    selectedDataSources: formData.selectedDataSources,
                    customDataSources: formData.customDataSources,
                    chartFilters: {
                        grades: formData.grades,
                        years: formData.years,
                        quarters: formData.quarters,
                        subjects: formData.subjects,
                        studentGroups: formData.studentGroups,
                        race: formData.race
                    }
                })
            })

            clearInterval(progressInterval)
            updateProgress('Finalizing...', estimatedSteps - currentStep)

            const data = await res.json()

            if (!res.ok) {
                const errorMsg = data.details ? `${data.error}: ${data.details}` : data.error || 'Failed to create presentation'
                console.error('API Error:', data)
                throw new Error(errorMsg)
            }

            console.log('Presentation created:', data.presentationId)
            setSlideProgress({ value: 100, step: 'Complete!' })
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
            setSlideProgress({ value: 0, step: '' })
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
                                <div className="space-y-2">
                                    <Label htmlFor="deckName">
                                        Deck Name <span className="text-muted-foreground">(Optional)</span>
                                    </Label>
                                    <Input
                                        id="deckName"
                                        value={formData.deckName}
                                        onChange={(e) => setFormData((prev) => ({ ...prev, deckName: e.target.value }))}
                                        placeholder={`Slide Deck - ${formData.partnerName || 'Untitled'}`}
                                    />
                                    <p className="text-muted-foreground text-xs">Leave empty to use default name</p>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="partnerName">
                                            Partner Name <span className="text-destructive">*</span>
                                        </Label>
                                        <Select
                                            id="partnerName"
                                            value={formData.partnerName}
                                            onChange={handlePartnerChange}
                                            required
                                            disabled={isLoadingDatasets}
                                        >
                                            <option value="">{isLoadingDatasets ? 'Loading datasets...' : 'Select a dataset/partner...'}</option>
                                            {partnerOptions.map((partner) => (
                                                <option key={partner.value} value={partner.value}>
                                                    {partner.label}
                                                </option>
                                            ))}
                                        </Select>
                                        {isLoadingDatasets && <p className="text-muted-foreground text-xs">Fetching datasets from BigQuery...</p>}
                                        {!isLoadingDatasets && partnerOptions.length === 1 && (
                                            <p className="text-muted-foreground text-xs">Enter a GCP Project ID above to load available datasets</p>
                                        )}
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
                                            District <span className="text-destructive">*</span>
                                        </Label>
                                        <Select
                                            id="districtName"
                                            value={formData.districtName}
                                            onChange={(e) => {
                                                setFormData((prev) => ({
                                                    ...prev,
                                                    districtName: e.target.value,
                                                    schools: []
                                                }))
                                            }}
                                            required
                                            disabled={isLoadingDistrictsSchools || !formData.partnerName}
                                        >
                                            <option value="">{isLoadingDistrictsSchools ? 'Loading districts...' : 'Select a district...'}</option>
                                            {districtOptions.map((district: string) => (
                                                <option key={district} value={district}>
                                                    {district}
                                                </option>
                                            ))}
                                        </Select>
                                        {isLoadingDistrictsSchools && <p className="text-muted-foreground text-xs">Fetching districts from NWEA table...</p>}
                                    </div>
                                    <div className="space-y-2">
                                        <Label>
                                            School(s) <span className="text-destructive">*</span>
                                        </Label>
                                        <MultiSelect
                                            options={schoolOptions}
                                            selected={formData.schools}
                                            onChange={(selected) => setFormData((prev) => ({ ...prev, schools: selected }))}
                                            placeholder={
                                                isLoadingDistrictsSchools
                                                    ? 'Loading schools...'
                                                    : !formData.districtName
                                                      ? 'Select district first...'
                                                      : 'Select school(s)...'
                                            }
                                            disabled={isLoadingDistrictsSchools || !formData.partnerName || !formData.districtName}
                                        />
                                        {isLoadingDistrictsSchools && <p className="text-muted-foreground text-xs">Fetching schools from NWEA table...</p>}
                                        {isLoadingDistrictsSchools && <p className="text-muted-foreground text-xs">Fetching schools from NWEA table...</p>}
                                    </div>
                                </div>
                            </div>

                            {/* Data Sources & Assessments - Only show when dataset is selected */}
                            {formData.partnerName && (
                                <div className="space-y-4 border-b pb-4">
                                    <h3 className="text-lg font-semibold">Assessments</h3>
                                    <div className="space-y-2">
                                        <Label>
                                            Select Assessments <span className="text-destructive">*</span>
                                        </Label>
                                        {isLoadingAssessments && <p className="text-muted-foreground mb-2 text-xs">Loading assessments...</p>}
                                        {isLoadingAssessmentTables && (
                                            <p className="text-muted-foreground mb-2 text-xs">Checking available assessment tables...</p>
                                        )}
                                        {!isLoadingAssessments && !isLoadingAssessmentTables && availableAssessments.length === 0 && (
                                            <p className="text-destructive mb-2 text-xs">No assessment tables found in this dataset</p>
                                        )}
                                        {!isLoadingAssessments && !isLoadingAssessmentTables && availableAssessments.length > 0 && (
                                            <p className="text-muted-foreground mb-2 text-xs">
                                                Available assessments in this dataset: {availableAssessments.join(', ')}
                                            </p>
                                        )}
                                        <div className="grid grid-cols-1 gap-2 rounded-lg border p-4">
                                            {ASSESSMENT_SOURCES.length === 0 && !isLoadingAssessments && (
                                                <p className="text-muted-foreground text-sm">No assessments available</p>
                                            )}
                                            {ASSESSMENT_SOURCES.map((source) => {
                                                // Only show assessment if it's available in the selected dataset
                                                const isAvailable = availableAssessments.length === 0 || availableAssessments.includes(source.id)
                                                const isSelected = formData.selectedDataSources.includes(source.id)
                                                const customTable = formData.customDataSources[source.id] || assessmentTables[source.id]
                                                const defaultTable = source.defaultTable

                                                if (!isAvailable) {
                                                    return null // Don't render unavailable assessments
                                                }

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
                                                                disabled={isLoadingAssessmentTables || isLoadingAssessments}
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
                                                                className="ml-6 mt-1 h-8 font-mono text-xs"
                                                            />
                                                        )}
                                                    </div>
                                                )
                                            })}
                                        </div>
                                        {availableAssessments.length > 0 && formData.assessments.length === 0 && (
                                            <p className="text-muted-foreground text-sm">Please select at least one assessment to continue</p>
                                        )}
                                        {isLoadingFilters && formData.assessments.length > 0 && (
                                            <p className="text-muted-foreground text-xs">Loading available filters...</p>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Filters - Dynamic based on selected assessments */}
                            {formData.assessments.length > 0 && (
                                <div className="space-y-4 border-b pb-4">
                                    <h3 className="text-lg font-semibold">Filters</h3>
                                    <p className="text-muted-foreground mb-4 text-xs">
                                        Available filters based on selected assessments: {formData.assessments.join(', ')}
                                    </p>
                                    <div className="grid grid-cols-2 gap-4">
                                        {supportsGrades && (
                                            <div className="space-y-2">
                                                <Label>
                                                    Grade(s) <span className="text-destructive">*</span>
                                                </Label>
                                                <MultiSelect
                                                    options={GRADES}
                                                    selected={formData.grades}
                                                    onChange={(selected) => setFormData((prev) => ({ ...prev, grades: selected }))}
                                                    placeholder="Select grade(s)..."
                                                    disabled={isLoadingFilters}
                                                />
                                            </div>
                                        )}
                                        <div className="space-y-2">
                                            <Label>
                                                Year(s) <span className="text-destructive">*</span>
                                            </Label>
                                            <MultiSelect
                                                options={YEARS}
                                                selected={formData.years}
                                                onChange={(selected) => setFormData((prev) => ({ ...prev, years: selected }))}
                                                placeholder="Select at least 2 year(s)..."
                                                disabled={isLoadingFilters}
                                            />
                                            {formData.years.length > 0 && formData.years.length < 2 && (
                                                <p className="text-destructive text-sm">Please select at least 2 years</p>
                                            )}
                                        </div>
                                    </div>
                                    {(availableQuarters.length > 0 || availableSubjects.length > 0) && (
                                        <div className="grid grid-cols-2 gap-4">
                                            {availableQuarters.length > 0 && (
                                                <div className="space-y-2">
                                                    <Label>
                                                        Quarter(s) <span className="text-destructive">*</span>
                                                    </Label>
                                                    <MultiSelect
                                                        options={availableQuarters}
                                                        selected={formData.quarters}
                                                        onChange={(selected) => setFormData((prev) => ({ ...prev, quarters: selected }))}
                                                        placeholder="Select quarter(s)..."
                                                        disabled={isLoadingFilters}
                                                    />
                                                </div>
                                            )}
                                            {availableSubjects.length > 0 && (
                                                <div className="space-y-2">
                                                    <Label>
                                                        Subject(s) <span className="text-destructive">*</span>
                                                    </Label>
                                                    <MultiSelect
                                                        options={availableSubjects}
                                                        selected={formData.subjects}
                                                        onChange={(selected) => setFormData((prev) => ({ ...prev, subjects: selected }))}
                                                        placeholder="Select subject(s)..."
                                                        disabled={isLoadingFilters}
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    )}
                                    {(supportsStudentGroups || supportsRace) && (
                                        <div className="grid grid-cols-2 gap-4">
                                            {supportsStudentGroups && (
                                                <div className="space-y-2">
                                                    <Label>Student Groups</Label>
                                                    <MultiSelect
                                                        options={
                                                            studentGroupOptions.length > 0
                                                                ? studentGroupOptions.filter((group) => !raceOptions.includes(group))
                                                                : [
                                                                      'All Students',
                                                                      'English Learners',
                                                                      'Students with Disabilities',
                                                                      'Socioeconomically Disadvantaged',
                                                                      'Foster',
                                                                      'Homeless'
                                                                  ]
                                                        }
                                                        selected={formData.studentGroups}
                                                        onChange={(selected) => setFormData((prev) => ({ ...prev, studentGroups: selected }))}
                                                        placeholder="Select student group(s)..."
                                                        disabled={isLoadingFilters}
                                                    />
                                                </div>
                                            )}
                                            {supportsRace && (
                                                <div className="space-y-2">
                                                    <Label>Race/Ethnicity</Label>
                                                    <MultiSelect
                                                        options={
                                                            raceOptions.length > 0
                                                                ? raceOptions
                                                                : [
                                                                      'Hispanic or Latino',
                                                                      'White',
                                                                      'Black or African American',
                                                                      'Asian',
                                                                      'Filipino',
                                                                      'American Indian or Alaska Native',
                                                                      'Native Hawaiian or Pacific Islander',
                                                                      'Two or More Races'
                                                                  ]
                                                        }
                                                        selected={formData.race}
                                                        onChange={(selected) => setFormData((prev) => ({ ...prev, race: selected }))}
                                                        placeholder="Select race/ethnicity..."
                                                        disabled={isLoadingFilters}
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}

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

                    {/* Progress Bar */}
                    {isCreating && slideProgress.value > 0 && (
                        <Card className="mb-6">
                            <CardContent className="pt-6">
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between text-sm">
                                        <span className="font-medium">{slideProgress.step}</span>
                                        <span className="text-muted-foreground">{slideProgress.value}%</span>
                                    </div>
                                    <Progress value={slideProgress.value} max={100} />
                                </div>
                            </CardContent>
                        </Card>
                    )}

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
