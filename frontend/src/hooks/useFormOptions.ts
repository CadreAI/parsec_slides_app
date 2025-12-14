import { useEffect, useState } from 'react'

const DEFAULT_GRADES = ['K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
const DEFAULT_YEARS = ['2023', '2024', '2025', '2026']

export function useFormOptions(projectId?: string, datasetId?: string, location?: string, assessments?: string[]) {
    const [grades, setGrades] = useState<string[]>(DEFAULT_GRADES)
    const [years, setYears] = useState<string[]>(DEFAULT_YEARS)
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const fetchFormOptions = async () => {
            setIsLoading(true)
            try {
                // Fetch from actual data tables if we have projectId and datasetId
                if (projectId && datasetId) {
                    // Build query params
                    const params = new URLSearchParams({
                        projectId,
                        datasetId,
                        location: location || 'US'
                    })

                    // Add assessments parameter if provided
                    if (assessments && assessments.length > 0) {
                        params.append('assessments', assessments.join(','))
                    }

                    const res = await fetch(`/api/bigquery/form-options?${params.toString()}`)
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

                // Fallback to defaults if no dataset selected or query fails
                // Return all possible grades (Pre-K to 12) - actual grades will come from query
                setGrades(['-1', 'K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
                const currentYear = new Date().getFullYear()
                setYears([currentYear, currentYear + 1, currentYear + 2, currentYear + 3].map((y) => y.toString()))
            } catch (error) {
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
    }, [projectId, datasetId, location, assessments?.join(',')])

    return { grades, years, isLoading }
}
