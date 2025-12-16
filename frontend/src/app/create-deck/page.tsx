'use client'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { MultiSelect } from '@/components/ui/multi-select'
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
import { getQuarterBackendValue, getQuarterDisplayLabel } from '@/utils/quarterLabels'
import { ArrowLeft } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { toast } from 'sonner'

// Helper function to format grade display labels
function getGradeDisplayLabel(grade: string): string {
    if (grade === '-1') return 'Pre-K'
    if (grade === 'K' || grade === '0') return 'Kindergarten'
    return `Grade ${grade}`
}

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
        districtOnly: false, // New: Filter to district-level charts only
        grades: [] as string[],
        years: [] as string[],
        quarters: '' as string,
        subjects: [] as string[],
        studentGroups: [] as string[],
        race: [] as string[],
        assessments: [] as string[],
        slidePrompt: '',
        enableAIInsights: true, // Toggle for AI analytics/insights
        themeColor: '#0094bd' // Default Parsec blue color
    })

    // Custom hooks for data fetching
    const { assessments: ASSESSMENT_SOURCES, isLoading: isLoadingAssessments } = useAvailableAssessments()
    const { grades: GRADES, years: YEARS } = useFormOptions(
        formData.projectId,
        formData.partnerName,
        formData.location,
        formData.assessments,
        formData.customDataSources
    )
    const { studentGroupOptions, raceOptions, studentGroupMappings, studentGroupOrder } = useStudentGroups()
    const { partnerOptions, isLoadingDatasets } = useDatasets(formData.projectId, formData.location)
    const { availableDistricts, availableSchools, districtSchoolMap, isLoadingDistrictsSchools } = useDistrictsAndSchools(
        formData.partnerName,
        formData.projectId,
        formData.location,
        formData.assessments,
        formData.customDataSources
    )
    const { availableAssessments, assessmentTables, variants, isLoadingAssessmentTables } = useAssessmentTables(
        formData.partnerName,
        formData.projectId,
        formData.location,
        setFormData,
        true
    )
    const { availableSubjects, availableQuarters, supportsGrades, supportsStudentGroups, supportsRace, isLoadingFilters } = useAssessmentFilters(
        formData.assessments,
        formData.projectId,
        formData.partnerName,
        formData.location,
        undefined,
        formData.customDataSources
    )

    // Combined loading state to prevent stale UI
    const isLoadingChoices = isLoadingDistrictsSchools || isLoadingFilters || isLoadingAssessmentTables || isLoadingDatasets

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
            quarters: ''
        }))
    }

    // Helper function to reset scope and filters when data changes
    const resetScopeAndFilters = () => ({
        districtName: '',
        schools: [],
        districtOnly: false,
        grades: [],
        quarters: '',
        subjects: [],
        studentGroups: [],
        race: []
    })

    const handleDataSourceToggle = (sourceId: string) => {
        setFormData((prev) => ({
            ...prev,
            selectedDataSources: prev.selectedDataSources.includes(sourceId)
                ? prev.selectedDataSources.filter((id) => id !== sourceId)
                : [...prev.selectedDataSources, sourceId],
            // Reset scope selections and filters when assessments change
            ...resetScopeAndFilters()
        }))
    }

    const handleCustomDataSourceChange = (sourceId: string, value: string) => {
        setFormData((prev) => ({
            ...prev,
            customDataSources: {
                ...prev.customDataSources,
                [sourceId]: value
            },
            // Reset scope selections and filters when variant changes
            ...resetScopeAndFilters()
        }))
    }

    // Combined: Ingest data, generate charts, then create slide deck
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()

        // Debug: Log current formData state at submit time
        console.log('[Frontend] handleSubmit - Current formData.themeColor:', formData.themeColor)

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
            toast.info('Queueing your deck generation task...')

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
                // Scope selection: only generate charts for selected schools/districts
                selected_schools: formData.districtOnly ? [] : formData.schools.length > 0 ? formData.schools : [], // Empty array if district only mode
                include_district_scope: !!formData.districtName // Include district scope if district is selected
            }

            // Build chart filters
            const chartFilters = {
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
                quarters: formData.quarters ? [formData.quarters] : undefined,
                subjects: formData.subjects.length > 0 ? formData.subjects : undefined,
                student_groups: formData.studentGroups.length > 0 ? formData.studentGroups : undefined,
                race: formData.race.length > 0 ? formData.race : undefined
            }

            const presentationTitle = formData.deckName.trim() || `Slide Deck - ${formData.partnerName || 'Untitled'}`

            // Hardcoded Google Drive folder
            const driveFolderUrl = 'https://drive.google.com/drive/folders/1CUOM-Sz6ulyzD2mTREdcYoBXUJLrgngw'

            // Call the new task API endpoint
            const requestBody = {
                partnerName: formData.partnerName,
                config: config,
                chartFilters: chartFilters,
                title: presentationTitle,
                driveFolderUrl: driveFolderUrl,
                enableAIInsights: formData.enableAIInsights,
                userPrompt: formData.slidePrompt || undefined,
                description: `Deck for ${formData.districtName || 'Parsec Academy'}`,
                themeColor: formData.themeColor // Always send the actual value from formData
            }
            console.log('[Frontend] FormData themeColor:', formData.themeColor)
            console.log('[Frontend] Sending themeColor in request:', requestBody.themeColor)

            const res = await fetch('/api/tasks/create-deck-with-slides', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            })

            const data = await res.json()

            if (!res.ok) {
                const errorMsg = data.error || 'Failed to queue task'
                console.error('API Error:', data)
                throw new Error(errorMsg)
            }

            toast.success('Task queued successfully! Returning to dashboard...')

            // Redirect to dashboard
            setTimeout(() => {
                router.push('/dashboard')
            }, 1000)
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
                                                            <div className="ml-6 mt-2">
                                                                {/* Case 1: Multiple variants - show radio buttons */}
                                                                {variants[source.id] && variants[source.id].length > 1 && (
                                                                    <div className="space-y-1.5">
                                                                        {variants[source.id].map((variant, idx) => (
                                                                            <div key={variant.table_name} className="flex items-center space-x-2">
                                                                                <input
                                                                                    type="radio"
                                                                                    id={`${source.id}-variant-${idx}`}
                                                                                    name={`${source.id}-variant`}
                                                                                    value={variant.full_path}
                                                                                    checked={formData.customDataSources[source.id] === variant.full_path}
                                                                                    onChange={() => handleCustomDataSourceChange(source.id, variant.full_path)}
                                                                                    className="h-3.5 w-3.5 cursor-pointer"
                                                                                />
                                                                                <label
                                                                                    htmlFor={`${source.id}-variant-${idx}`}
                                                                                    className="flex-1 cursor-pointer font-mono text-xs text-gray-700"
                                                                                >
                                                                                    {variant.full_path}
                                                                                    {variant.is_default && (
                                                                                        <span className="ml-2 rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-700">
                                                                                            Default
                                                                                        </span>
                                                                                    )}
                                                                                </label>
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                )}

                                                                {/* Case 2: Single variant - show as read-only */}
                                                                {variants[source.id] && variants[source.id].length === 1 && (
                                                                    <Input
                                                                        value={variants[source.id][0].full_path}
                                                                        readOnly
                                                                        className="h-8 cursor-not-allowed bg-gray-50 font-mono text-xs"
                                                                    />
                                                                )}

                                                                {/* Case 3: No variants (fallback to manual input) */}
                                                                {(!variants[source.id] || variants[source.id].length === 0) && (
                                                                    <>
                                                                        <Input
                                                                            value={customTable || defaultTable}
                                                                            onChange={(e) => handleCustomDataSourceChange(source.id, e.target.value)}
                                                                            placeholder={defaultTable}
                                                                            className="h-8 font-mono text-xs"
                                                                        />
                                                                        {variants[source.id] && variants[source.id].length === 0 && (
                                                                            <p className="text-destructive mt-1 text-xs">
                                                                                ⚠️ No table variants found. Please contact the Dev Team.
                                                                            </p>
                                                                        )}
                                                                    </>
                                                                )}
                                                            </div>
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

                            {/* Scope Selection - Only show when assessments are selected */}
                            {formData.assessments.length > 0 && (
                                <div className="space-y-4 border-b pb-4">
                                    <h3 className="text-lg font-semibold">Scope Selection</h3>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-2">
                                            <Label>
                                                District <span className="text-destructive">*</span>
                                            </Label>
                                            <Select
                                                key={`district-${formData.partnerName}-${formData.assessments.join(',')}-${Object.values(formData.customDataSources).join(',')}`}
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
                                                disabled={isLoadingChoices || !formData.partnerName || formData.assessments.length === 0}
                                            >
                                                <option value="">
                                                    {isLoadingChoices
                                                        ? 'Loading districts...'
                                                        : formData.assessments.length === 0
                                                          ? 'Select assessments first...'
                                                          : 'Select a district...'}
                                                </option>
                                                {districtOptions.map((district: string) => (
                                                    <option key={district} value={district}>
                                                        {district}
                                                    </option>
                                                ))}
                                            </Select>
                                            {isLoadingChoices && (
                                                <p className="text-muted-foreground text-xs">
                                                    Fetching districts from {formData.assessments.join(', ')} table(s)...
                                                </p>
                                            )}
                                        </div>
                                        <div className="space-y-2">
                                            <div className="mb-2 flex items-center space-x-2">
                                                <Checkbox
                                                    id="districtOnly"
                                                    checked={formData.districtOnly}
                                                    onChange={(e) => {
                                                        const checked = e.target.checked
                                                        setFormData((prev) => ({
                                                            ...prev,
                                                            districtOnly: checked,
                                                            schools: checked ? [] : prev.schools // Clear schools when enabling district only
                                                        }))
                                                    }}
                                                    disabled={!formData.districtName || isLoadingDistrictsSchools}
                                                />
                                                <Label htmlFor="districtOnly" className="cursor-pointer text-sm font-normal">
                                                    District Only (exclude school-level charts)
                                                </Label>
                                            </div>
                                            <Label>School(s) {!formData.districtOnly && <span className="text-destructive">*</span>}</Label>
                                            <MultiSelect
                                                key={`schools-${formData.districtName}-${Object.values(formData.customDataSources).join(',')}`}
                                                options={schoolOptions}
                                                selected={formData.schools}
                                                onChange={(selected) => setFormData((prev) => ({ ...prev, schools: selected }))}
                                                placeholder={
                                                    formData.districtOnly
                                                        ? 'District only mode - schools disabled'
                                                        : isLoadingChoices
                                                          ? 'Loading schools...'
                                                          : !formData.districtName
                                                            ? 'Select district first...'
                                                            : 'Select school(s)...'
                                                }
                                                disabled={isLoadingChoices || !formData.partnerName || !formData.districtName || formData.districtOnly}
                                            />
                                            {isLoadingChoices && (
                                                <p className="text-muted-foreground text-xs">
                                                    Fetching schools from {formData.assessments.join(', ')} table(s)...
                                                </p>
                                            )}
                                            {formData.districtOnly && (
                                                <p className="text-muted-foreground text-xs">
                                                    District only mode enabled - only district-level charts will be generated
                                                </p>
                                            )}
                                        </div>
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
                                                    key={`grades-${formData.assessments.join(',')}-${Object.values(formData.customDataSources).join(',')}`}
                                                    options={GRADES}
                                                    selected={formData.grades}
                                                    onChange={(selected) => setFormData((prev) => ({ ...prev, grades: selected }))}
                                                    placeholder={isLoadingChoices ? 'Loading grades...' : 'Select grade(s)...'}
                                                    disabled={isLoadingChoices}
                                                    getDisplayLabel={getGradeDisplayLabel}
                                                />
                                            </div>
                                        )}
                                        <div className="space-y-2">
                                            <Label>
                                                Year(s) <span className="text-destructive">*</span>
                                            </Label>
                                            <MultiSelect
                                                key={`years-${formData.assessments.join(',')}-${Object.values(formData.customDataSources).join(',')}`}
                                                options={YEARS}
                                                selected={formData.years}
                                                onChange={(selected) => setFormData((prev) => ({ ...prev, years: selected }))}
                                                placeholder={isLoadingChoices ? 'Loading years...' : 'Select at least 2 year(s)...'}
                                                disabled={isLoadingChoices}
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
                                                        Quarter <span className="text-destructive">*</span>
                                                    </Label>
                                                    <Select
                                                        value={formData.quarters ? getQuarterDisplayLabel(formData.quarters) : ''}
                                                        onChange={(e) => {
                                                            // Convert display label back to backend value for storage
                                                            const backendQuarter = getQuarterBackendValue(e.target.value)
                                                            setFormData((prev) => ({ ...prev, quarters: backendQuarter }))
                                                        }}
                                                        disabled={isLoadingChoices}
                                                    >
                                                        <option value="">Select Type of Slides...</option>
                                                        {availableQuarters.map((quarter) => (
                                                            <option key={quarter} value={getQuarterDisplayLabel(quarter)}>
                                                                {getQuarterDisplayLabel(quarter)}
                                                            </option>
                                                        ))}
                                                    </Select>
                                                </div>
                                            )}
                                            {availableSubjects.length > 0 && (
                                                <div className="space-y-2">
                                                    <Label>
                                                        Subject(s) <span className="text-destructive">*</span>
                                                    </Label>
                                                    <MultiSelect
                                                        key={`subjects-${formData.assessments.join(',')}-${Object.values(formData.customDataSources).join(',')}`}
                                                        options={availableSubjects}
                                                        selected={formData.subjects}
                                                        onChange={(selected) => setFormData((prev) => ({ ...prev, subjects: selected }))}
                                                        placeholder={isLoadingChoices ? 'Loading subjects...' : 'Select subject(s)...'}
                                                        disabled={isLoadingChoices}
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
                                                        key={`student-groups-${formData.assessments.join(',')}-${Object.values(formData.customDataSources).join(',')}`}
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
                                                        placeholder={isLoadingChoices ? 'Loading student groups...' : 'Select student group(s)...'}
                                                        disabled={isLoadingChoices}
                                                    />
                                                </div>
                                            )}
                                            {supportsRace && (
                                                <div className="space-y-2">
                                                    <Label>Race/Ethnicity</Label>
                                                    <MultiSelect
                                                        key={`race-${formData.assessments.join(',')}-${Object.values(formData.customDataSources).join(',')}`}
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
                                                        placeholder={isLoadingChoices ? 'Loading race/ethnicity...' : 'Select race/ethnicity...'}
                                                        disabled={isLoadingChoices}
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
                                <div className="space-y-4">
                                    <div className="flex items-center space-x-2">
                                        <Checkbox
                                            id="enableAIInsights"
                                            checked={formData.enableAIInsights}
                                            onChange={(e) => {
                                                setFormData((prev) => ({
                                                    ...prev,
                                                    enableAIInsights: e.target.checked
                                                }))
                                            }}
                                        />
                                        <Label htmlFor="enableAIInsights" className="cursor-pointer text-sm font-normal">
                                            Enable AI Analytics & Insights
                                        </Label>
                                    </div>
                                    {!formData.enableAIInsights && (
                                        <p className="text-muted-foreground text-xs">
                                            ⚡ AI analytics disabled - charts will be generated faster without AI analysis
                                        </p>
                                    )}
                                    <div className="space-y-2">
                                        <Label htmlFor="themeColor">Theme Color</Label>
                                        <div className="flex items-center gap-3">
                                            <input
                                                type="color"
                                                id="themeColor"
                                                value={formData.themeColor}
                                                onChange={(e) => {
                                                    const newColor = e.target.value
                                                    console.log('[Frontend] Color picker changed to:', newColor)
                                                    setFormData((prev) => {
                                                        console.log('[Frontend] Updating themeColor from', prev.themeColor, 'to', newColor)
                                                        return { ...prev, themeColor: newColor }
                                                    })
                                                }}
                                                className="h-10 w-20 cursor-pointer rounded border border-gray-300"
                                            />
                                            <Input
                                                type="text"
                                                value={formData.themeColor}
                                                onChange={(e) => {
                                                    const newColor = e.target.value
                                                    console.log('[Frontend] Color text input changed to:', newColor)
                                                    setFormData((prev) => {
                                                        console.log('[Frontend] Updating themeColor from', prev.themeColor, 'to', newColor)
                                                        return { ...prev, themeColor: newColor }
                                                    })
                                                }}
                                                placeholder="#0094bd"
                                                className="w-32"
                                            />
                                        </div>
                                        <p className="text-muted-foreground text-xs">
                                            Select a color for the top bar and cover slide background (default: Parsec blue)
                                        </p>
                                    </div>
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
