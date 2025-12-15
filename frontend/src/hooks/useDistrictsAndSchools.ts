import { useEffect, useState } from 'react'

export function useDistrictsAndSchools(partnerName: string, projectId: string, location: string, assessments?: string[], selectedTables?: Record<string, string>) {
    const [availableDistricts, setAvailableDistricts] = useState<string[]>([])
    const [availableSchools, setAvailableSchools] = useState<string[]>([])
    const [districtSchoolMap, setDistrictSchoolMap] = useState<Record<string, string[]>>({})
    const [isLoadingDistrictsSchools, setIsLoadingDistrictsSchools] = useState(false)

    useEffect(() => {
        const fetchDistrictsAndSchools = async () => {
            if (!partnerName || !projectId || partnerName.trim() === '') {
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
                return
            }

            // Don't fetch if assessments are required but not selected
            if (assessments && assessments.length === 0) {
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
                return
            }

            setIsLoadingDistrictsSchools(true)
            try {
                // Build query params
                const params = new URLSearchParams({
                    projectId: projectId,
                    datasetId: partnerName,
                    location: location
                })

                // Add assessments if provided
                if (assessments && assessments.length > 0) {
                    params.append('assessments', assessments.join(','))
                }

                // Add specific table paths if provided
                if (selectedTables && assessments) {
                    const relevantTables = assessments
                        .map(a => selectedTables[a])
                        .filter(Boolean)
                        .map(path => path.split('.').pop()) // Extract table name only
                    if (relevantTables.length > 0) {
                        params.append('tablePaths', relevantTables.join(','))
                    }
                }

                const res = await fetch(`/api/bigquery/districts-schools?${params.toString()}`)
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
    }, [partnerName, projectId, location, assessments?.join(','), Object.values(selectedTables || {}).sort().join(',')])

    return {
        availableDistricts,
        availableSchools,
        districtSchoolMap,
        isLoadingDistrictsSchools
    }
}
