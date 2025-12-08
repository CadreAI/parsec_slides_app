import { useEffect, useState } from 'react'

const DEFAULT_FILTERS = {
    subjects: ['Reading', 'Mathematics', 'ELA', 'Math'],
    quarters: ['Fall', 'Winter', 'Spring'],
    supportsGrades: true,
    supportsStudentGroups: true,
    supportsRace: true
}

export function useAssessmentFilters(
    assessments: string[],
    projectId?: string,
    datasetId?: string,
    location?: string,
    setFormData?: React.Dispatch<React.SetStateAction<any>>
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
                    const res = await fetch(
                        `/api/bigquery/assessment-filters?projectId=${encodeURIComponent(projectId)}&datasetId=${encodeURIComponent(datasetId)}&assessments=${encodeURIComponent(assessmentsParam)}&location=${encodeURIComponent(location || 'US')}`
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
                                setFormData((prev: any) => {
                                    const filteredSubjects = prev.subjects.filter((s: string) => newSubjects.includes(s))
                                    const filteredQuarters = prev.quarters.filter((q: string) => newQuarters.includes(q))

                                    if (filteredSubjects.length !== prev.subjects.length || filteredQuarters.length !== prev.quarters.length) {
                                        return {
                                            ...prev,
                                            subjects: filteredSubjects,
                                            quarters: filteredQuarters
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
    }, [assessments.join(','), projectId, datasetId, location])

    return {
        availableSubjects,
        availableQuarters,
        supportsGrades,
        supportsStudentGroups,
        supportsRace,
        isLoadingFilters
    }
}
