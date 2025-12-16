import { useEffect, useState } from 'react'

const DEFAULT_FILTERS = {
    subjects: ['Reading', 'Mathematics', 'ELA', 'Math'],
    quarters: ['Fall', 'Winter', 'Spring'],
    supportsGrades: true,
    supportsStudentGroups: true,
    supportsRace: true
}

interface FormData {
    subjects: string[]
    quarters: string | string[]
    [key: string]: unknown
}

export function useAssessmentFilters(
    assessments: string[],
    projectId?: string,
    datasetId?: string,
    location?: string,
    setFormData?: React.Dispatch<React.SetStateAction<FormData>>,
    selectedTables?: Record<string, string>
) {
    const [availableSubjects, setAvailableSubjects] = useState<string[]>([])
    const [availableQuarters, setAvailableQuarters] = useState<string[]>([])
    const [supportsGrades, setSupportsGrades] = useState(true)
    const [supportsStudentGroups, setSupportsStudentGroups] = useState(true)
    const [supportsRace, setSupportsRace] = useState(true)
    const [isLoadingFilters, setIsLoadingFilters] = useState(false)

    useEffect(() => {
        const fetchAssessmentFilters = async () => {
            if (assessments.length === 0) {
                setAvailableSubjects([])
                setAvailableQuarters([])
                setSupportsGrades(false)
                setSupportsStudentGroups(false)
                setSupportsRace(false)
                return
            }

            setIsLoadingFilters(true)
            try {
                const assessmentsParam = assessments.join(',')

                // Fetch from actual data tables if we have projectId and datasetId
                if (projectId && datasetId && assessments.length > 0) {
                    // Add table paths to query params if provided
                    const tablePathsParam =
                        selectedTables && assessments.length > 0
                            ? '&tablePaths=' +
                              encodeURIComponent(
                                  assessments
                                      .map((a) => selectedTables[a])
                                      .filter(Boolean)
                                      .join(',')
                              )
                            : ''
                    const res = await fetch(
                        `/api/bigquery/assessment-filters?projectId=${encodeURIComponent(projectId)}&datasetId=${encodeURIComponent(datasetId)}&assessments=${encodeURIComponent(assessmentsParam)}&location=${encodeURIComponent(location || 'US')}${tablePathsParam}`
                    )
                    if (res.ok) {
                        const data = await res.json()
                        if (data.success && data.filters) {
                            const newSubjects = data.filters.subjects || []
                            const newQuarters = data.filters.quarters || []

                            setAvailableSubjects(newSubjects)
                            setAvailableQuarters(newQuarters)
                            setSupportsGrades(data.filters.supports_grades !== false)
                            setSupportsStudentGroups(data.filters.supports_student_groups !== false)
                            setSupportsRace(data.filters.supports_race !== false)

                            // Clear invalid filter selections if setFormData is provided
                            if (setFormData) {
                                setFormData((prev: FormData) => {
                                    const filteredSubjects = (prev.subjects || []).filter((s: string) => newSubjects.includes(s))
                                    // Handle both string and string[] for quarters
                                    const prevQuarters = Array.isArray(prev.quarters) ? prev.quarters : prev.quarters ? [prev.quarters] : []
                                    const filteredQuarters = prevQuarters.filter((q: string) => newQuarters.includes(q))
                                    const filteredQuartersValue =
                                        filteredQuarters.length > 0 ? (filteredQuarters.length === 1 ? filteredQuarters[0] : filteredQuarters) : ''

                                    const prevQuartersArray = Array.isArray(prev.quarters) ? prev.quarters : prev.quarters ? [prev.quarters] : []
                                    if (filteredSubjects.length !== (prev.subjects || []).length || filteredQuarters.length !== prevQuartersArray.length) {
                                        return {
                                            ...prev,
                                            subjects: filteredSubjects,
                                            quarters: filteredQuartersValue
                                        }
                                    }
                                    return prev
                                })
                            }
                            setIsLoadingFilters(false)
                            return
                        }
                    }
                }

                // Fallback to defaults if no dataset selected or query fails
                setAvailableSubjects(DEFAULT_FILTERS.subjects)
                setAvailableQuarters(DEFAULT_FILTERS.quarters)
                setSupportsGrades(DEFAULT_FILTERS.supportsGrades)
                setSupportsStudentGroups(DEFAULT_FILTERS.supportsStudentGroups)
                setSupportsRace(DEFAULT_FILTERS.supportsRace)
            } catch (error) {
                console.error('Error fetching assessment filters:', error)
                setAvailableSubjects(DEFAULT_FILTERS.subjects)
                setAvailableQuarters(DEFAULT_FILTERS.quarters)
                setSupportsGrades(DEFAULT_FILTERS.supportsGrades)
                setSupportsStudentGroups(DEFAULT_FILTERS.supportsStudentGroups)
                setSupportsRace(DEFAULT_FILTERS.supportsRace)
            } finally {
                setIsLoadingFilters(false)
            }
        }
        fetchAssessmentFilters()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        assessments.join(','),
        projectId,
        datasetId,
        location,
        Object.values(selectedTables || {})
            .sort()
            .join(',')
    ])

    return {
        availableSubjects,
        availableQuarters,
        supportsGrades,
        supportsStudentGroups,
        supportsRace,
        isLoadingFilters
    }
}
