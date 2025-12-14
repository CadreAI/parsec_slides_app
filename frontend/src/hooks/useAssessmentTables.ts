import { useEffect, useState } from 'react'

interface FormData {
    customDataSources: Record<string, string>
    [key: string]: unknown
}

interface TableVariant {
    table_name: string
    full_path: string
    is_default: boolean
}

export function useAssessmentTables(
    partnerName: string,
    projectId: string,
    location: string,
    setFormData: React.Dispatch<React.SetStateAction<FormData>>,
    includeVariants: boolean = true
) {
    const [availableAssessments, setAvailableAssessments] = useState<string[]>([])
    const [assessmentTables, setAssessmentTables] = useState<Record<string, string>>({})
    const [variants, setVariants] = useState<Record<string, TableVariant[]>>({})
    const [isLoadingAssessmentTables, setIsLoadingAssessmentTables] = useState(false)

    useEffect(() => {
        const fetchAssessmentTables = async () => {
            if (!partnerName || !projectId || partnerName.trim() === '') {
                setAvailableAssessments([])
                setAssessmentTables({})
                setVariants({})
                return
            }

            setIsLoadingAssessmentTables(true)
            try {
                const res = await fetch(
                    `/api/bigquery/assessment-tables?projectId=${encodeURIComponent(projectId)}&datasetId=${encodeURIComponent(partnerName)}&location=${encodeURIComponent(location)}&includeVariants=${includeVariants}`
                )
                const data = await res.json()

                if (res.ok && data.success) {
                    const assessments = data.available_assessments || []
                    const tables = data.tables || {}
                    const fetchedVariants = data.variants || {}
                    setAvailableAssessments(assessments)
                    setAssessmentTables(tables)
                    setVariants(fetchedVariants)

                    // Auto-update customDataSources with found tables
                    setFormData((prev: FormData) => {
                        const updatedCustomDataSources = { ...(prev.customDataSources || {}) }
                        for (const [assessmentId, tableId] of Object.entries(tables)) {
                            if (typeof tableId === 'string') {
                                updatedCustomDataSources[assessmentId] = tableId
                            }
                        }
                        return {
                            ...prev,
                            customDataSources: updatedCustomDataSources
                        }
                    })

                    console.log(`[Assessment Tables] Available assessments: ${assessments.join(', ')}`)
                } else {
                    console.warn('Failed to load assessment tables:', data.error || 'Unknown error')
                    setAvailableAssessments([])
                    setAssessmentTables({})
                    setVariants({})
                }
            } catch (error) {
                console.error('Error fetching assessment tables:', error)
                setAvailableAssessments([])
                setAssessmentTables({})
                setVariants({})
            } finally {
                setIsLoadingAssessmentTables(false)
            }
        }

        fetchAssessmentTables()
    }, [partnerName, projectId, location, includeVariants, setFormData])

    return {
        availableAssessments,
        assessmentTables,
        variants,
        isLoadingAssessmentTables
    }
}
