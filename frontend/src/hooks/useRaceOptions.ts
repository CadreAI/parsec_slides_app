import { useEffect, useState } from 'react'

const DEFAULT_RACE_OPTIONS = [
    'Hispanic or Latino',
    'White',
    'Black or African American',
    'Asian',
    'Filipino',
    'American Indian or Alaska Native',
    'Native Hawaiian or Pacific Islander',
    'Two or More Races',
    'Not Stated'
]

export function useRaceOptions(
    projectId?: string,
    datasetId?: string,
    location?: string,
    assessments?: string[],
    selectedTables?: Record<string, string>
) {
    const [raceOptions, setRaceOptions] = useState<string[]>([])
    const [isLoadingRace, setIsLoadingRace] = useState(true)

    useEffect(() => {
        const abortController = new AbortController()

        const fetchRaceOptions = async () => {
            // IMMEDIATELY clear state when dependencies change to prevent showing stale data
            setIsLoadingRace(true)
            setRaceOptions([])

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

                    const res = await fetch(`/api/bigquery/race-options?${params.toString()}`, {
                        signal: abortController.signal
                    })

                    if (res.ok) {
                        const data = await res.json()
                        if (data.success && data.race_options) {
                            setRaceOptions(data.race_options)
                            setIsLoadingRace(false)
                            return
                        }
                    }
                }

                // Fallback to defaults if no dataset selected or query fails
                setRaceOptions(DEFAULT_RACE_OPTIONS)
            } catch (error) {
                // Ignore abort errors - these are intentional cancellations
                if (error instanceof Error && error.name === 'AbortError') {
                    console.log('[useRaceOptions] Request aborted (expected behavior)')
                    return
                }
                console.error('Error fetching race options:', error)
                // Fallback to defaults on error
                setRaceOptions(DEFAULT_RACE_OPTIONS)
            } finally {
                setIsLoadingRace(false)
            }
        }

        fetchRaceOptions()

        // Cleanup: abort the request if dependencies change or component unmounts
        return () => {
            abortController.abort()
        }
    }, [projectId, datasetId, location, assessments?.join(','), Object.values(selectedTables || {}).sort().join(',')])

    return { raceOptions, isLoadingRace }
}

