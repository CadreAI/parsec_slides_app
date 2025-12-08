import { useEffect, useState } from 'react'

export function useDistrictsAndSchools(partnerName: string, projectId: string, location: string) {
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

            setIsLoadingDistrictsSchools(true)
            try {
                const res = await fetch(
                    `/api/bigquery/districts-schools?projectId=${encodeURIComponent(projectId)}&datasetId=${encodeURIComponent(partnerName)}&location=${encodeURIComponent(location)}`
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
    }, [partnerName, projectId, location])

    return {
        availableDistricts,
        availableSchools,
        districtSchoolMap,
        isLoadingDistrictsSchools
    }
}
