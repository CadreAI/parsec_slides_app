import { useEffect, useState } from 'react'

export function useDistrictsAndSchools(partnerName: string, projectId: string, location: string, assessments?: string[], selectedTables?: Record<string, string>, districtName?: string) {
    const [availableDistricts, setAvailableDistricts] = useState<string[]>([])
    const [availableSchools, setAvailableSchools] = useState<string[]>([])
    const [districtSchoolMap, setDistrictSchoolMap] = useState<Record<string, string[]>>({})
    const [isLoadingDistrictsSchools, setIsLoadingDistrictsSchools] = useState(false)
    
    // Clustering state
    const [schoolClusters, setSchoolClusters] = useState<Record<string, string[]>>({})
    const [clusteredSchools, setClusteredSchools] = useState<string[]>([])
    const [isLoadingClustering, setIsLoadingClustering] = useState(false)

    useEffect(() => {
        const abortController = new AbortController()

        const fetchDistrictsAndSchools = async () => {
            if (!partnerName || !projectId || partnerName.trim() === '') {
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
                setSchoolClusters({})
                setClusteredSchools([])
                return
            }

            // Don't fetch if assessments are required but not selected
            if (assessments && assessments.length === 0) {
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
                setSchoolClusters({})
                setClusteredSchools([])
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

                const res = await fetch(
                    `/api/bigquery/districts-schools?${params.toString()}`,
                    { signal: abortController.signal }
                )
                const data = await res.json()

                if (res.ok && data.success) {
                    setAvailableDistricts(data.districts || [])
                    setAvailableSchools(data.schools || [])
                    setDistrictSchoolMap(data.districtSchoolMap || {})
                    console.log(`Loaded ${data.districts?.length || 0} districts and ${data.schools?.length || 0} schools`)
                    // Don't cluster yet - wait for district selection
                } else {
                    console.warn('Failed to load districts/schools:', data.error || 'Unknown error')
                    setAvailableDistricts([])
                    setAvailableSchools([])
                    setDistrictSchoolMap({})
                    setSchoolClusters({})
                    setClusteredSchools([])
                }
            } catch (error) {
                // Ignore abort errors - these are intentional cancellations
                if (error instanceof Error && error.name === 'AbortError') {
                    console.log('[useDistrictsAndSchools] Request aborted (expected behavior)')
                    return
                }
                console.error('Error fetching districts and schools:', error)
                setAvailableDistricts([])
                setAvailableSchools([])
                setDistrictSchoolMap({})
                setSchoolClusters({})
                setClusteredSchools([])
            } finally {
                setIsLoadingDistrictsSchools(false)
            }
        }

        fetchDistrictsAndSchools()

        // Cleanup: abort the request if dependencies change or component unmounts
        return () => {
            abortController.abort()
        }
    }, [partnerName, projectId, location, assessments?.join(','), Object.values(selectedTables || {}).sort().join(',')])

    // Separate effect: Trigger clustering when district is selected
    useEffect(() => {
        const abortController = new AbortController()

        console.log('[Clustering Effect] Triggered:', { 
            districtName, 
            hasDistrictSchoolMap: Object.keys(districtSchoolMap).length > 0,
            districtSchoolMapKeys: Object.keys(districtSchoolMap)
        })

        const fetchSchoolClusters = async (schools: string[], districtName: string) => {
            console.log('[Clustering] Starting for district:', districtName, 'with', schools.length, 'schools')
            setIsLoadingClustering(true)
            try {
                const res = await fetch('/api/llm/cluster-schools', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        schools: schools,
                        district_name: districtName
                    }),
                    signal: abortController.signal
                })
                
                const data = await res.json()
                console.log('[Clustering] API Response:', data)
                
                if (res.ok && data.success) {
                    const clusters = data.clusters || {}
                    setSchoolClusters(clusters)
                    setClusteredSchools(Object.keys(clusters).sort())
                    console.log(`[Clustering] Success: ${schools.length} schools → ${Object.keys(clusters).length} clusters (source: ${data.source})`)
                    console.log('[Clustering] Cluster names:', Object.keys(clusters))
                } else {
                    console.warn('[Clustering] Failed, using identity mapping:', data.error || 'Unknown error')
                    // Fallback: use identity mapping (each school is its own cluster)
                    const identityClusters = Object.fromEntries(schools.map(s => [s, [s]]))
                    setSchoolClusters(identityClusters)
                    setClusteredSchools(schools.sort())
                }
            } catch (error) {
                // Ignore abort errors - these are intentional cancellations
                if (error instanceof Error && error.name === 'AbortError') {
                    console.log('[Clustering] Request aborted (expected behavior)')
                    return
                }
                console.error('[Clustering] Error:', error)
                // Fallback: use identity mapping
                const identityClusters = Object.fromEntries(schools.map(s => [s, [s]]))
                setSchoolClusters(identityClusters)
                setClusteredSchools(schools.sort())
            } finally {
                setIsLoadingClustering(false)
            }
        }

        // Only cluster when district is selected and we have the districtSchoolMap
        if (districtName && Object.keys(districtSchoolMap).length > 0) {
            const schoolsInDistrict = districtSchoolMap[districtName] || []
            console.log('[Clustering] Schools in district:', schoolsInDistrict.length, schoolsInDistrict)
            if (schoolsInDistrict.length > 0) {
                fetchSchoolClusters(schoolsInDistrict, districtName)
            } else {
                console.log('[Clustering] No schools in district, clearing')
                setSchoolClusters({})
                setClusteredSchools([])
            }
        } else {
            console.log('[Clustering] District not selected or no map, clearing')
            // Clear clustering when no district selected
            setSchoolClusters({})
            setClusteredSchools([])
        }

        // Cleanup: abort the request if dependencies change or component unmounts
        return () => {
            abortController.abort()
        }
    }, [districtName, districtSchoolMap])

    return {
        availableDistricts,
        availableSchools,
        clusteredSchools,
        schoolClusters,
        districtSchoolMap,
        isLoadingDistrictsSchools: isLoadingDistrictsSchools || isLoadingClustering
    }
}
