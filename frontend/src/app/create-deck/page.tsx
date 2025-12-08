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
import { ArrowLeft } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'

// Available assessment sources for data ingestion
const ASSESSMENT_SOURCES = [
    { id: 'nwea', label: 'NWEA Map Growth', defaultTable: 'parsecgo.demodashboard.Nwea_production_calpads_v4_2' },
    { id: 'iready', label: 'iReady', defaultTable: 'parsecgo.demodashboard.iready_production_calpads_v4_2' },
    { id: 'star', label: 'STAR', defaultTable: 'parsecgo.demodashboard.renaissance_production_calpads_v4_2' },
    { id: 'cers', label: 'CERS', defaultTable: 'parsecgo.demodashboard.cers_production' }
]

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
        },
        client_classicalacademies: {
            districts: [],
            schools: {}
        },
        client_plumascharter: {
            districts: [],
            schools: {}
        }
    }

    const [partnerOptions, setPartnerOptions] = useState<Array<{ value: string; label: string }>>([])
    const [isLoadingDatasets, setIsLoadingDatasets] = useState(false)
    const [availableDistricts, setAvailableDistricts] = useState<string[]>([])
    const [availableSchools, setAvailableSchools] = useState<string[]>([])
    const [districtSchoolMap, setDistrictSchoolMap] = useState<Record<string, string[]>>({})
    const [isLoadingDistrictsSchools, setIsLoadingDistrictsSchools] = useState(false)

    // Fetch datasets from BigQuery when projectId changes
    useEffect(() => {
        const fetchDatasets = async () => {
            if (!formData.projectId || formData.projectId.trim() === '') {
                return
            }

            setIsLoadingDatasets(true)
            try {
                const res = await fetch(
                    `/api/bigquery/datasets?projectId=${encodeURIComponent(formData.projectId)}&location=${encodeURIComponent(formData.location)}`
                )
                const data = await res.json()

                if (res.ok && data.success && data.datasets) {
                    const datasetOptions = data.datasets.map((datasetId: string) => ({
                        value: datasetId,
                        label: datasetId
                    }))
                    setPartnerOptions(datasetOptions)
                    console.log(
                        `Loaded ${datasetOptions.length} datasets from BigQuery:`,
                        datasetOptions.map((opt: { value: string }) => opt.value)
                    )
                } else {
                    console.warn('Failed to load datasets:', data.error || 'Unknown error')
                    // Show error message but don't set default options
                    toast.error(`Failed to load datasets: ${data.error || 'Unknown error'}`)
                }
            } catch (error) {
                console.error('Error fetching datasets:', error)
                // Keep default option on error
            } finally {
                setIsLoadingDatasets(false)
            }
        }

        fetchDatasets()
    }, [formData.projectId, formData.location])

    // Fetch districts and schools from NWEA table when partner is selected
    useEffect(() => {
        const fetchDistrictsAndSchools = async () => {
            if (!formData.partnerName || !formData.projectId || formData.partnerName.trim() === '') {
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
                return
            }

            setIsLoadingDistrictsSchools(true)
            try {
                const res = await fetch(
                    `/api/bigquery/districts-schools?projectId=${encodeURIComponent(formData.projectId)}&datasetId=${encodeURIComponent(formData.partnerName)}&location=${encodeURIComponent(formData.location)}`
                )
                const data = await res.json()

                if (res.ok && data.success) {
                    setAvailableDistricts(data.districts || [])
                    setAvailableSchools(data.schools || [])
                    setDistrictSchoolMap(data.districtSchoolMap || {})
                    console.log(`Loaded ${data.districts?.length || 0} districts and ${data.schools?.length || 0} schools`)
                } else {
                    console.warn('Failed to load districts/schools:', data.error || 'Unknown error')
                    setAvailableDistricts([])
                    setAvailableSchools([])
                    setDistrictSchoolMap({})
                }
            } catch (error) {
                console.error('Error fetching districts and schools:', error)
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
            } finally {
                setIsLoadingDistrictsSchools(false)
            }
        }

        fetchDistrictsAndSchools()
    }, [formData.partnerName, formData.projectId, formData.location])

    const getDistrictOptions = () => {
        // Use dynamically fetched districts if available, otherwise fall back to partnerConfig
        if (availableDistricts.length > 0) {
            return availableDistricts
        }
        if (!formData.partnerName || !partnerConfig[formData.partnerName]) {
            return []
        }
        return partnerConfig[formData.partnerName].districts
    }

    const getSchoolOptions = () => {
        // Use dynamically fetched schools if available
        if (formData.districtName && Object.keys(districtSchoolMap).length > 0) {
            // Filter schools by selected district
            const schools = districtSchoolMap[formData.districtName] || []
            return schools
        }
        if (availableSchools.length > 0) {
            return availableSchools
        }
        // Fall back to partnerConfig
        if (!formData.partnerName || !partnerConfig[formData.partnerName]) {
            return []
        }
        const schools = partnerConfig[formData.partnerName].schools[formData.districtName] || []
        return schools
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
                    quarters: formData.quarters.length > 0 ? formData.quarters : undefined,
                    subjects: formData.subjects.length > 0 ? formData.subjects : undefined,
                    student_groups: formData.studentGroups.length > 0 ? formData.studentGroups : undefined,
                    race: formData.race.length > 0 ? formData.race : undefined
                },
                // Scope selection: only generate charts for selected schools/districts
                selected_schools: formData.schools.length > 0 ? formData.schools : [],
                include_district_scope: !!formData.districtName // Include district scope if district is selected
            }

            // Prepare presentation title and other data before redirect
            const presentationTitle = formData.deckName.trim() || `Slide Deck - ${formData.partnerName || 'Untitled'}`
            const assessmentsToUse = formData.assessments.length > 0 ? formData.assessments : formData.selectedDataSources
            const driveFolderUrl = 'https://drive.google.com/drive/folders/1CUOM-Sz6ulyzD2mTREdcYoBXUJLrgngw'

            // Store job info in localStorage BEFORE redirecting
            const jobId = `deck_${Date.now()}`
            const jobInfo = {
                id: jobId,
                title: presentationTitle,
                status: 'processing',
                startTime: Date.now(),
                estimatedDuration: 300000, // Start with 5 minutes estimate, will update when we know chart count
                step: 'Ingesting data and generating charts...',
                progress: 0,
                config: config, // Store config for later use
                presentationTitle: presentationTitle,
                assessments: assessmentsToUse,
                driveFolderUrl: driveFolderUrl,
                quarters: formData.quarters,
                partnerName: formData.partnerName,
                userPrompt: formData.slidePrompt || undefined
            }
            localStorage.setItem('activeDeckJob', JSON.stringify(jobInfo))

            // Redirect to dashboard immediately - BEFORE making any API calls
            router.push('/dashboard?job=' + jobId)

            // Continue processing in background after redirect
            // Use setTimeout to ensure redirect happens first
            setTimeout(async () => {
                try {
                    // Step 1: Ingest data and generate charts
                    const ingestRes = await fetch('/api/data/ingest', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ config })
                    })

                    const ingestData = await ingestRes.json()

                    if (!ingestRes.ok) {
                        throw new Error(ingestData.error || ingestData.details || 'Failed to ingest data')
                    }

                    // Check if async job was started (has jobId)
                    if (ingestData.jobId) {
                        console.log(`[Frontend] Async job started with ID: ${ingestData.jobId}`)

                        // Poll for job status
                        const pollJobStatus = async (jobId: string): Promise<any> => {
                            return new Promise((resolve, reject) => {
                                const pollInterval = setInterval(async () => {
                                    try {
                                        const statusRes = await fetch(`/api/job-status?jobId=${jobId}`)
                                        const statusData = await statusRes.json()

                                        if (!statusRes.ok) {
                                            clearInterval(pollInterval)
                                            reject(new Error(statusData.error || 'Failed to check job status'))
                                            return
                                        }

                                        // Update job progress in localStorage
                                        try {
                                            const jobData = localStorage.getItem('activeDeckJob')
                                            if (jobData) {
                                                const job = JSON.parse(jobData)
                                                job.step = statusData.step || job.step
                                                job.progress = statusData.progress || job.progress
                                                job.status =
                                                    statusData.status === 'SUCCESS' ? 'completed' : statusData.status === 'FAILURE' ? 'failed' : 'processing'
                                                if (statusData.error) job.error = statusData.error
                                                localStorage.setItem('activeDeckJob', JSON.stringify(job))
                                            }
                                        } catch (error) {
                                            console.error('Error updating job progress:', error)
                                        }

                                        if (statusData.status === 'SUCCESS') {
                                            clearInterval(pollInterval)
                                            resolve(statusData)
                                        } else if (statusData.status === 'FAILURE') {
                                            clearInterval(pollInterval)
                                            reject(new Error(statusData.error || 'Job failed'))
                                        }
                                    } catch (error) {
                                        clearInterval(pollInterval)
                                        reject(error)
                                    }
                                }, 2000) // Poll every 2 seconds

                                // Timeout after 60 minutes
                                setTimeout(
                                    () => {
                                        clearInterval(pollInterval)
                                        reject(new Error('Job polling timeout after 60 minutes'))
                                    },
                                    60 * 60 * 1000
                                )
                            })
                        }

                        // Wait for job to complete
                        const jobResult = await pollJobStatus(ingestData.jobId)
                        var charts = jobResult.charts || []
                        const chartCount = charts.length

                        // Update job with final results
                        try {
                            const jobData = localStorage.getItem('activeDeckJob')
                            if (jobData) {
                                const job = JSON.parse(jobData)
                                job.charts = charts
                                job.progress = 30
                                job.estimatedDuration = Math.max(60000, chartCount * 2000 + 30000)
                                localStorage.setItem('activeDeckJob', JSON.stringify(job))
                            }
                        } catch (error) {
                            console.error('Error updating job progress:', error)
                        }
                    } else {
                        // Synchronous response (fallback)
                        const chartCount = ingestData.charts?.length || 0
                        var charts = ingestData.charts || []

                        // Update job progress
                        try {
                            const jobData = localStorage.getItem('activeDeckJob')
                            if (jobData) {
                                const job = JSON.parse(jobData)
                                job.step = 'Creating slide deck...'
                                job.progress = 30
                                job.estimatedDuration = Math.max(60000, chartCount * 2000 + 30000)
                                job.charts = charts
                                localStorage.setItem('activeDeckJob', JSON.stringify(job))
                            }
                        } catch (error) {
                            console.error('Error updating job progress:', error)
                        }
                    }

                    // Step 2: Create slide deck
                    // Charts are already defined above in the if/else block
                    const totalCharts = charts.length
                    const estimatedSteps = Math.max(10, 5 + Math.ceil(totalCharts * 0.5))
                    let currentStep = 0

                    const updateProgress = (step: string, increment: number = 1) => {
                        currentStep += increment
                        const progress = Math.min(30 + Math.round((currentStep / estimatedSteps) * 70), 95) // 30-95% range

                        // Update job progress in localStorage
                        try {
                            const jobData = localStorage.getItem('activeDeckJob')
                            if (jobData) {
                                const job = JSON.parse(jobData)
                                job.step = step
                                job.progress = progress
                                localStorage.setItem('activeDeckJob', JSON.stringify(job))
                            }
                        } catch (error) {
                            console.error('Error updating job progress:', error)
                        }
                    }

                    // Simulate progress updates during API call
                    const progressInterval = setInterval(() => {
                        if (currentStep < estimatedSteps - 1) {
                            const simulatedProgress = Math.min(currentStep + 0.3, estimatedSteps - 1)
                            const progress = Math.min(30 + Math.round((simulatedProgress / estimatedSteps) * 70), 95)

                            try {
                                const jobData = localStorage.getItem('activeDeckJob')
                                if (jobData) {
                                    const job = JSON.parse(jobData)
                                    job.progress = progress
                                    localStorage.setItem('activeDeckJob', JSON.stringify(job))
                                }
                            } catch (error) {
                                console.error('Error updating job progress:', error)
                            }
                        }
                    }, 300)

                    updateProgress('Creating presentation...', 1)
                    setTimeout(() => updateProgress('Uploading charts to Drive...', 2), 500)
                    setTimeout(() => updateProgress('Creating slides...', 2), 1000)
                    setTimeout(() => updateProgress('Adding charts to slides...', 2), 1500)

                    // Step 2: Call the API route to create the presentation
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
                            userPrompt: formData.slidePrompt || undefined // User prompt for decision LLM
                        })
                    })

                    clearInterval(progressInterval)

                    const data = await res.json()

                    if (!res.ok) {
                        const errorMsg = data.details ? `${data.error}: ${data.details}` : data.error || 'Failed to create presentation'
                        console.error('API Error:', data)

                        // Update job status to failed
                        const failedJob = {
                            ...jobInfo,
                            status: 'failed',
                            error: errorMsg,
                            progress: 0
                        }
                        localStorage.setItem('activeDeckJob', JSON.stringify(failedJob))
                        throw new Error(errorMsg)
                    }

                    console.log('Presentation created:', data.presentationId)

                    // Update job status to completed
                    const completedJob = {
                        ...jobInfo,
                        status: 'completed',
                        presentationId: data.presentationId,
                        presentationUrl: data.presentationUrl,
                        progress: 100,
                        step: 'Complete!',
                        completedTime: Date.now()
                    }
                    localStorage.setItem('activeDeckJob', JSON.stringify(completedJob))
                    localStorage.setItem('lastCompletedDeck', JSON.stringify(completedJob))

                    // Show success notification
                    toast.success(`✅ Complete! Presentation created. View it here: ${data.presentationUrl}`)
                } catch (fetchError: unknown) {
                    const errorMessage = fetchError instanceof Error ? fetchError.message : 'Unknown error'
                    const failedJob = {
                        ...jobInfo,
                        status: 'failed',
                        error: errorMessage,
                        progress: 0
                    }
                    localStorage.setItem('activeDeckJob', JSON.stringify(failedJob))
                    console.error('Background processing error:', fetchError)
                }
            }, 100) // Small delay to ensure redirect happens first
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
                                            {getDistrictOptions().map((district) => (
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
                                            options={getSchoolOptions()}
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
                                            options={['2023', '2024', '2025', '2026']}
                                            selected={formData.years}
                                            onChange={(selected) => setFormData((prev) => ({ ...prev, years: selected }))}
                                            placeholder="Select at least 2 year(s)..."
                                        />
                                        {formData.years.length > 0 && formData.years.length < 2 && (
                                            <p className="text-destructive text-sm">Please select at least 2 years</p>
                                        )}
                                    </div>
                                    <div className="space-y-2">
                                        <Label>
                                            Quarter(s) <span className="text-destructive">*</span>
                                        </Label>
                                        <MultiSelect
                                            options={['Fall', 'Winter', 'Spring']}
                                            selected={formData.quarters}
                                            onChange={(selected) => setFormData((prev) => ({ ...prev, quarters: selected }))}
                                            placeholder="Select quarter(s)..."
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
                                <h3 className="text-lg font-semibold">Assessments</h3>
                                <div className="space-y-2">
                                    <Label>
                                        Select Assessments <span className="text-destructive">*</span>
                                    </Label>
                                    <p className="text-muted-foreground mb-2 text-xs">
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
                                                            className="ml-6 mt-1 h-8 font-mono text-xs"
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
