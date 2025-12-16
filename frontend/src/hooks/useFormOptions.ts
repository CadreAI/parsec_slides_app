import { useEffect, useState } from 'react'

const DEFAULT_GRADES = ['K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
const DEFAULT_YEARS = ['2023', '2024', '2025', '2026']

export function useFormOptions(projectId?: string, datasetId?: string, location?: string, assessments?: string[], selectedTables?: Record<string, string>) {
    const [grades, setGrades] = useState<string[]>(DEFAULT_GRADES)
    const [years, setYears] = useState<string[]>(DEFAULT_YEARS)
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const abortController = new AbortController()

        const fetchFormOptions = async () => {
            // IMMEDIATELY clear state when dependencies change to prevent showing stale data
            setIsLoading(true)
            setGrades([])
            setYears([])

            try {
                // Only fetch from actual data tables if we have projectId, datasetId, AND assessments selected
                if (projectId && datasetId && assessments && assessments.length > 0) {
                    // Build query params
                    const params = new URLSearchParams({
                        projectId,
                        datasetId,
                        location: location || 'US'
                    })

                    // Add assessments parameter
                    params.append('assessments', assessments.join(','))

                    // Add specific table paths if provided
                    if (selectedTables) {
                        const relevantTables = assessments.map((a) => selectedTables[a]).filter(Boolean)
                        if (relevantTables.length > 0) {
                            params.append('tablePaths', relevantTables.join(','))
                        }
                    }

                    const res = await fetch(
                        `/api/bigquery/form-options?${params.toString()}`,
                        { signal: abortController.signal }
                    )
                    if (res.ok) {
                        const data = await res.json()
                        if (data.success) {
                            setGrades(data.grades || DEFAULT_GRADES)
                            setYears(data.years || DEFAULT_YEARS)
                            setIsLoading(false)
                            return
                        }
                    }
                }

                // Fallback to defaults if no dataset/assessments selected or query fails
                // Return all possible grades (Pre-K to 12) - actual grades will come from query
                setGrades(['-1', 'K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
                const currentYear = new Date().getFullYear()
                setYears([currentYear, currentYear + 1, currentYear + 2, currentYear + 3].map((y) => y.toString()))
            } catch (error) {
                // Ignore abort errors - these are intentional cancellations
                if (error instanceof Error && error.name === 'AbortError') {
                    console.log('[useFormOptions] Request aborted (expected behavior)')
                    return
                }
                console.error('Error fetching form options:', error)
                // Fallback to all possible grades (Pre-K to 12)
                setGrades(['-1', 'K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
                const currentYear = new Date().getFullYear()
                setYears([currentYear, currentYear + 1, currentYear + 2, currentYear + 3].map((y) => y.toString()))
            } finally {
                setIsLoading(false)
            }
        }

        fetchFormOptions()

        // Cleanup: abort the request if dependencies change or component unmounts
        return () => {
            abortController.abort()
        }
    }, [
        projectId,
        datasetId,
        location,
        assessments?.join(','),
        Object.values(selectedTables || {})
            .sort()
            .join(',')
    ])

    return { grades, years, isLoading }
}
