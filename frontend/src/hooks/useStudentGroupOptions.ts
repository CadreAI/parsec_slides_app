import { useEffect, useState } from 'react'

const DEFAULT_STUDENT_GROUP_OPTIONS = [
    'All Students',
    'English Learners',
    'Students with Disabilities',
    'Socioeconomically Disadvantaged',
    'Foster',
    'Homeless'
]

export function useStudentGroupOptions(
    projectId?: string,
    datasetId?: string,
    location?: string,
    assessments?: string[],
    selectedTables?: Record<string, string>
) {
    const [studentGroupOptions, setStudentGroupOptions] = useState<string[]>([])
    const [isLoadingStudentGroups, setIsLoadingStudentGroups] = useState(true)

    useEffect(() => {
        const abortController = new AbortController()

        const fetchStudentGroupOptions = async () => {
            // IMMEDIATELY clear state when dependencies change to prevent showing stale data
            setIsLoadingStudentGroups(true)
            setStudentGroupOptions([])

            try {
                // Fetch from actual data tables if we have projectId, datasetId, and assessments
                if (projectId && datasetId && assessments && assessments.length > 0) {
                    // Build query params
                    const params = new URLSearchParams({
                        projectId,
                        datasetId,
                        location: location || 'US',
                        assessments: assessments.join(',')
                    })

                    // Add specific table paths if provided
                    if (selectedTables && assessments.length > 0) {
                        const relevantTables = assessments.map((a) => selectedTables[a]).filter(Boolean)
                        if (relevantTables.length > 0) {
                            params.append('tablePaths', relevantTables.join(','))
                        }
                    }

                    const res = await fetch(`/api/bigquery/student-group-options?${params.toString()}`, {
                        signal: abortController.signal
                    })

                    if (res.ok) {
                        const data = await res.json()
                        if (data.success && data.student_group_options) {
                            setStudentGroupOptions(data.student_group_options)
                            setIsLoadingStudentGroups(false)
                            return
                        }
                    }
                }

                // Fallback to defaults if no dataset selected or query fails
                setStudentGroupOptions(DEFAULT_STUDENT_GROUP_OPTIONS)
            } catch (error) {
                // Ignore abort errors - these are intentional cancellations
                if (error instanceof Error && error.name === 'AbortError') {
                    console.log('[useStudentGroupOptions] Request aborted (expected behavior)')
                    return
                }
                console.error('Error fetching student group options:', error)
                // Fallback to defaults on error
                setStudentGroupOptions(DEFAULT_STUDENT_GROUP_OPTIONS)
            } finally {
                setIsLoadingStudentGroups(false)
            }
        }

        fetchStudentGroupOptions()

        // Cleanup: abort the request if dependencies change or component unmounts
        return () => {
            abortController.abort()
        }
    }, [projectId, datasetId, location, assessments?.join(','), Object.values(selectedTables || {}).sort().join(',')])

    return { studentGroupOptions, isLoadingStudentGroups }
}

