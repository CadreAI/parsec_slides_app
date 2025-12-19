'use client'

import { AssessmentScopeSelector } from '@/components/AssessmentScopeSelector'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Combobox } from '@/components/ui/combobox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { MultiSelect } from '@/components/ui/multi-select'
import { Textarea } from '@/components/ui/textarea'
import { useAssessmentFilters } from '@/hooks/useAssessmentFilters'
import { useAssessmentTables } from '@/hooks/useAssessmentTables'
import { useAvailableAssessments } from '@/hooks/useAvailableAssessments'
import { useDatasets } from '@/hooks/useDatasets'
import { useFormOptions } from '@/hooks/useFormOptions'
import { useRaceOptions } from '@/hooks/useRaceOptions'
import { useStudentGroupOptions } from '@/hooks/useStudentGroupOptions'
import { useStudentGroups } from '@/hooks/useStudentGroups'
import { getQuarterBackendValue } from '@/utils/quarterLabels'
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
        // Per-assessment scopes: { 'NWEA': { districts: [], schools: [] }, 'iReady': { ... } }
        assessmentScopes: {} as Record<
            string,
            {
                districts: string[]
                schools: string[] // UI-selected values (may be clustered names)
                resolvedSchools?: string[] // expanded raw school names (used for backend ingestion/charting)
                includeDistrictwide?: boolean
                includeSchools?: boolean
            }
        >,
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
    const {
        grades: GRADES,
        years: YEARS,
        isLoading: isLoadingFormOptions
    } = useFormOptions(formData.projectId, formData.partnerName, formData.location, formData.assessments, formData.customDataSources)
    const { studentGroupOptions, raceOptions, studentGroupMappings, studentGroupOrder } = useStudentGroups()
    const { partnerOptions, isLoadingDatasets } = useDatasets(formData.projectId, formData.location)
    // Note: District/school fetching is now done per-assessment (see AssessmentScopeSelector component below)
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
    const { raceOptions: dynamicRaceOptions, isLoadingRace } = useRaceOptions(
        formData.projectId,
        formData.partnerName,
        formData.location,
        formData.assessments,
        formData.customDataSources
    )
    const { studentGroupOptions: dynamicStudentGroupOptions, isLoadingStudentGroups } = useStudentGroupOptions(
        formData.projectId,
        formData.partnerName,
        formData.location,
        formData.assessments,
        formData.customDataSources
    )

    // Combined loading state to prevent stale UI
    const isLoadingChoices =
        isLoadingFilters || isLoadingAssessmentTables || isLoadingDatasets || isLoadingFormOptions || isLoadingRace || isLoadingStudentGroups

    // Note: District/school selection is now done per-assessment
    // See AssessmentScopeSelector component below

    const handleCheckboxChange = (name: string, value: string, checked: boolean) => {
        setFormData((prev) => {
            const currentArray = (prev[name as keyof typeof prev] as string[]) || []

            // Special handling for assessments - initialize or remove scope
            if (name === 'assessments') {
                const newAssessments = checked ? [...currentArray, value] : currentArray.filter((item) => item !== value)

                const newScopes = { ...prev.assessmentScopes }

                if (checked) {
                    // Initialize scope for new assessment
                    newScopes[value] = {
                        districts: [],
                        schools: [],
                        includeDistrictwide: true,
                        includeSchools: true
                    }
                } else {
                    // Remove scope for deselected assessment
                    delete newScopes[value]
                }

                return {
                    ...prev,
                    [name]: newAssessments,
                    assessmentScopes: newScopes
                }
            }

            // Default handling for other checkboxes
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
            districts: [],
            schools: [],
            quarters: ''
        }))
    }

    const handlePartnerComboboxChange = (partner: string) => {
        setFormData((prev) => ({
            ...prev,
            partnerName: partner,
            districts: [],
            schools: [],
            quarters: ''
        }))
    }

    // Helper function to reset scope and filters when data changes
    const resetScopeAndFilters = () => ({
        // Keep assessmentScopes intact - they are managed separately per assessment
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

    const handleAssessmentScopeChange = (
        assessmentId: string,
        scope: { districts: string[]; schools: string[]; includeDistrictwide?: boolean; includeSchools?: boolean }
    ) => {
        setFormData((prev) => ({
            ...prev,
            assessmentScopes: {
                ...prev.assessmentScopes,
                [assessmentId]: scope
            }
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

        if (formData.assessments.length === 0) {
            toast.error('Please select at least one assessment first')
            return
        }

        // Validate that each assessment has scope selected (districts or schools)
        for (const assessmentId of formData.assessments) {
            const scope = formData.assessmentScopes[assessmentId]
            if (!scope) {
                toast.error(`Please configure scope for ${assessmentId}`)
                return
            }
            const includeDistrictwide = scope.includeDistrictwide !== false
            const includeSchools = scope.includeSchools !== false
            // Must include districtwide and/or at least one school
            if (!includeDistrictwide && (!includeSchools || scope.schools.length === 0)) {
                toast.error(`Please enable Districtwide and/or select at least one school for ${assessmentId}`)
                return
            }
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

            // Aggregate scopes from all assessments
            const aggregatedDistricts = new Set<string>()
            const aggregatedSchools = new Set<string>()
            let anyDistrictwideOnly = false

            formData.assessments.forEach((assessmentId) => {
                const scope = formData.assessmentScopes[assessmentId]
                if (scope) {
                    scope.districts.forEach((d) => aggregatedDistricts.add(d))
                    const includeDistrictwide = scope.includeDistrictwide !== false
                    const includeSchools = scope.includeSchools !== false
                    if (includeSchools) {
                        const resolved = Array.isArray(scope.resolvedSchools) && scope.resolvedSchools.length > 0 ? scope.resolvedSchools : scope.schools
                        resolved.forEach((s) => aggregatedSchools.add(s))
                    }
                    if (includeDistrictwide && !includeSchools) {
                        anyDistrictwideOnly = true
                    }
                }
            })

            const districtList = Array.from(aggregatedDistricts).sort()
            const schoolList = Array.from(aggregatedSchools).sort()

            console.log('[Frontend] Aggregated scopes:', { districtList, schoolList, anyDistrictwideOnly, assessmentScopes: formData.assessmentScopes })

            // Resolve clustered school selections (if present) into raw school lists for backend use.
            const resolvedAssessmentScopes = Object.fromEntries(
                Object.entries(formData.assessmentScopes || {}).map(([aid, scope]) => {
                    const resolved = Array.isArray(scope?.resolvedSchools) && scope.resolvedSchools.length > 0 ? scope.resolvedSchools : scope?.schools || []
                    return [
                        aid,
                        {
                            ...scope,
                            schools: resolved
                        }
                    ]
                })
            )

            // Build config object
            const config = {
                partner_name: formData.partnerName || 'demodashboard',
                // Some partners (esp. charter-style datasets) don't have a district field; use Districtwide.
                district_name: districtList.length > 0 ? districtList : ['Districtwide'],
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
                // Scope selection: aggregated from all assessment scopes
                selected_schools: schoolList,
                // Include district scope if there are districts OR if this partner doesn't have districts but we have schools.
                include_district_scope: districtList.length > 0 || schoolList.length > 0 || anyDistrictwideOnly,
                // Store per-assessment scopes for future granular filtering (not yet implemented in backend)
                assessment_scopes: resolvedAssessmentScopes
            }

            // Build chart filters
            // Note: ingestion + runners increasingly rely on config.assessment_scopes for per-assessment scope.
            // chartFilters retains some legacy keys for runner-based scripts.

            const selectedAssessmentIds = Array.isArray(formData.assessments) ? formData.assessments : []
            const includeDistrictwideAny = selectedAssessmentIds.some((aid) => {
                const scope = formData.assessmentScopes[aid]
                return scope?.includeDistrictwide !== false
            })

            // Scope controls for runner-based iReady scripts:
            // - chartFilters.district_only: generate district charts only (skip all school loops)
            // - chartFilters.schools: generate district + only these school charts
            const hasIready = selectedAssessmentIds.includes('iready')
            const ireadyScope = hasIready ? formData.assessmentScopes['iready'] : undefined
            const ireadyIncludeDistrictwide = ireadyScope?.includeDistrictwide !== false
            const ireadyIncludeSchools = ireadyScope?.includeSchools !== false
            const ireadyDistrictOnly = Boolean(hasIready && ireadyIncludeDistrictwide && !ireadyIncludeSchools)
            const ireadyResolvedSchools =
                hasIready && Array.isArray(ireadyScope?.resolvedSchools) && ireadyScope!.resolvedSchools!.length > 0
                    ? ireadyScope!.resolvedSchools
                    : ireadyScope?.schools
            const ireadySelectedSchools =
                hasIready && !ireadyDistrictOnly && ireadyIncludeSchools && Array.isArray(ireadyResolvedSchools) && ireadyResolvedSchools.length > 0
                    ? ireadyResolvedSchools
                    : undefined

            // Scope controls for runner-based NWEA scripts (Fall/Winter legacy runners)
            // We keep this separate from iReady to avoid collisions when multiple assessments are selected.
            const hasNwea = selectedAssessmentIds.includes('nwea')
            const nweaScope = hasNwea ? formData.assessmentScopes['nwea'] : undefined
            const nweaIncludeDistrictwide = nweaScope?.includeDistrictwide !== false
            const nweaIncludeSchools = nweaScope?.includeSchools !== false
            const nweaDistrictOnly = Boolean(hasNwea && nweaIncludeDistrictwide && !nweaIncludeSchools)

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
                race: formData.race.length > 0 ? formData.race : undefined,

                // Districtwide aggregate selection (any assessment)
                // include_districtwide tells backend ingestion to avoid school-name SQL filters
                // when generating a districtwide aggregate (even if school charts are also requested).
                include_districtwide: includeDistrictwideAny ? true : undefined,

                // iReady scope selection (district vs schools)
                district_only: ireadyDistrictOnly ? true : undefined,
                schools: ireadySelectedSchools
                ,
                // NWEA scope selection (district vs schools)
                nwea_district_only: nweaDistrictOnly ? true : undefined
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
                description: `Deck for ${districtList.length > 0 ? districtList.join(', ') : schoolList.length > 0 ? schoolList.join(', ') : formData.partnerName}`,
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
                                        <Combobox
                                            options={partnerOptions}
                                            value={formData.partnerName}
                                            onChange={handlePartnerComboboxChange}
                                            placeholder={isLoadingDatasets ? 'Loading datasets...' : 'Select a dataset/partner...'}
                                            searchPlaceholder="Search datasets..."
                                            disabled={isLoadingDatasets}
                                        />
                                        {isLoadingDatasets && <p className="text-muted-foreground text-xs">Fetching datasets from BigQuery...</p>}
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

                                                        {/* Per-Assessment Scope Selection */}
                                                        {isSelected && formData.assessmentScopes[source.id] && (
                                                            <AssessmentScopeSelector
                                                                assessmentId={source.id}
                                                                partnerName={formData.partnerName}
                                                                projectId={formData.projectId}
                                                                location={formData.location}
                                                                customDataSource={formData.customDataSources[source.id]}
                                                                scope={formData.assessmentScopes[source.id]}
                                                                onScopeChangeAction={handleAssessmentScopeChange}
                                                                partnerConfig={PARTNER_CONFIG}
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

                            {/* Scope selection is now done per-assessment within each assessment checkbox above */}

                            {/* Quarter Selection - Buttons */}
                            {formData.assessments.length > 0 && availableQuarters.length > 0 && (
                                <div className="space-y-4 border-b pb-4">
                                    <h3 className="text-lg font-semibold">
                                        Type of Slides <span className="text-destructive">*</span>
                                    </h3>
                                    <div className="flex gap-4">
                                        {['BOY', 'MOY', 'EOY'].map((quarter) => {
                                            const backendValue = getQuarterBackendValue(quarter)
                                            const isSelected = formData.quarters === backendValue
                                            return (
                                                <Button
                                                    key={quarter}
                                                    type="button"
                                                    variant={isSelected ? 'default' : 'outline'}
                                                    onClick={() => setFormData((prev) => ({ ...prev, quarters: backendValue }))}
                                                    disabled={isLoadingChoices}
                                                    className={`flex-1 ${isSelected ? 'bg-primary text-primary-foreground' : ''}`}
                                                >
                                                    {quarter}
                                                </Button>
                                            )
                                        })}
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
                                    {availableSubjects.length > 0 && (
                                        <div className="grid grid-cols-2 gap-4">
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
                                        </div>
                                    )}
                                    {(supportsStudentGroups || supportsRace) && (
                                        <div className="grid grid-cols-2 gap-4">
                                            {supportsStudentGroups && dynamicStudentGroupOptions.length > 1 && (
                                                <div className="space-y-2">
                                                    <Label>Student Groups</Label>
                                                    <MultiSelect
                                                        key={`student-groups-${formData.assessments.join(',')}-${Object.values(formData.customDataSources).join(',')}`}
                                                        options={dynamicStudentGroupOptions}
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
                                                            dynamicRaceOptions.length > 0
                                                                ? dynamicRaceOptions
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
                                            Enable AI Analytics & Insights (included in slide notes)
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
