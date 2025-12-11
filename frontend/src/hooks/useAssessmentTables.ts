import { useEffect, useState } from 'react'

export function useAssessmentTables<T extends { customDataSources?: Record<string, string> }>(
    partnerName: string,
    projectId: string,
    location: string,
    setFormData: React.Dispatch<React.SetStateAction<T>>
) {
    const [availableAssessments, setAvailableAssessments] = useState<string[]>([])
    const [assessmentTables, setAssessmentTables] = useState<Record<string, string>>({})
    const [isLoadingAssessmentTables, setIsLoadingAssessmentTables] = useState(false)

    useEffect(() => {
        const fetchAssessmentTables = async () => {
            if (!partnerName || !projectId || partnerName.trim() === '') {
                setAvailableAssessments([])
                setAssessmentTables({})
                return
            }

            setIsLoadingAssessmentTables(true)
            try {
                const res = await fetch(
                    `/api/bigquery/assessment-tables?projectId=${encodeURIComponent(projectId)}&datasetId=${encodeURIComponent(partnerName)}&location=${encodeURIComponent(location)}`
                )
                const data = await res.json()

                if (res.ok && data.success) {
                    const assessments = data.available_assessments || []
                    const tables = data.tables || {}
                    setAvailableAssessments(assessments)
                    setAssessmentTables(tables)

                    // Auto-update customDataSources with found tables
                    setFormData((prev: T) => {
                        const updatedCustomDataSources = { ...(prev.customDataSources || {}) }
                        for (const [assessmentId, tableId] of Object.entries(tables)) {
                            if (typeof tableId === 'string') {
                                updatedCustomDataSources[assessmentId] = tableId
                            }
                        }
                        return {
                            ...prev,
                            customDataSources: updatedCustomDataSources
                        } as T
                    })

                    console.log(`[Assessment Tables] Available assessments: ${assessments.join(', ')}`)
                } else {
                    console.warn('Failed to load assessment tables:', data.error || 'Unknown error')
                    setAvailableAssessments([])
                    setAssessmentTables({})
                }
            } catch (error) {
                console.error('Error fetching assessment tables:', error)
                setAvailableAssessments([])
                setAssessmentTables({})
            } finally {
                setIsLoadingAssessmentTables(false)
            }
        }

        fetchAssessmentTables()
    }, [partnerName, projectId, location, setFormData])

    return {
        availableAssessments,
        assessmentTables,
        isLoadingAssessmentTables
    }
}
